import datetime
import sys
import math
import os
import time
import warnings
import copy
import pickle
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.utils.data
import torchvision
import utils
from sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
from collections import OrderedDict
from typing import Dict, Callable, Optional, Any, Tuple, Union
import numpy as np
import statistics
import logging
from visdial import VisDial
import torch
import torch.utils.data
from torch.nn.functional import normalize
import losses
import torch.nn.functional as F
from transformers import AutoProcessor, BlipForImageTextRetrieval
os.environ['TOKENIZERS_PARALLELISM']='false'
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

class DiffGate(nn.Module):

    def __init__(self, dim=256):
        super().__init__()
        in_dim = dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, dim), nn.GELU(),
            nn.Linear(dim, dim // 2), nn.GELU(),
            nn.Linear(dim // 2, 1)
        )
  
    def init_weights(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.LayerNorm):
                nn.init.ones_(layer.weight); nn.init.zeros_(layer.bias)
    def forward(self, cap, u, detach_inputs=True):
        z = torch.cat([u, cap], dim=-1) 
        s = torch.sigmoid(self.mlp(z))     
        return s

class Guided_Block(nn.Module):

    def __init__(self, dim=256, rank=8, alpha_max=0.1, use_ln=True):
        super().__init__()
        self.dim, self.rank, self.alpha_max = dim, rank, alpha_max
        hid =  dim*2
        self.mlp_A = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, dim * rank))
        self.mlp_B = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, dim * rank))
        self.ln = nn.LayerNorm(dim) if use_ln else nn.Identity()
        self.gate = DiffGate()

    def init_weights(self):
        for m in (self.mlp_A, self.mlp_B):
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        if isinstance(self.ln, nn.LayerNorm):
            nn.init.ones_(self.ln.weight); nn.init.zeros_(self.ln.bias)
        self.gate.init_weights()
    @staticmethod
    def _col_l2_norm(M, eps=1e-6):
        return M / (M.norm(dim=1, keepdim=True).clamp_min(eps))
    def forward(self, u_base: torch.Tensor, cls_in: torch.Tensor,cap_feat) -> torch.Tensor:
        if u_base.dim() == 3:
            u = u_base[:, 0, :]                  
        else:
            u = u_base                             

        if cls_in.dim() == 2:
            x = cls_in.unsqueeze(1)               
        else:
            x = cls_in                               

        if cap_feat.dim() == 3:
            c = cap_feat[:, 0, :]                    
        else:
            c = cap_feat
        B = x.size(0); d = self.dim
        x = x.contiguous()
        u = u.to(x.dtype).contiguous()
        c = c.to(x.dtype).contiguous()
        A  = self.mlp_A(u).view(B, d, self.rank).contiguous()    
        Bm = self.mlp_B(u).view(B, d, self.rank).contiguous()   

        A = self._col_l2_norm(A)      
        Bm= self._col_l2_norm(Bm)
 
        t = torch.bmm(x, Bm)                                     
     
        delta = torch.bmm(t, A.transpose(1, 2))                 
                              
        s = self.gate(c, u)   
        
        out_s=s[0,:]                   
        s = s.to(x.dtype).unsqueeze(1) 
        y = x + s*delta            
        y = self.ln(y)                   
        return y,out_s

class BlipForRetrieval(BlipForImageTextRetrieval):
    def __init__(self, config):
        super().__init__(config)

        self.g_b=Guided_Block()

        self._weights_initialized = False  # 

    def _init_weights_(self):
        self.g_b.init_weights()
        self._weights_initialized = True

    def get_text_features(self, input_ids, attention_mask=None, return_dict=None, output_all=False):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        with torch.no_grad():
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)
            embeds = outputs[0] if not return_dict else outputs.last_hidden_state  
            feats = self.text_proj(embeds)  
            return feats[:, 0, :]      

    def get_image_features(self, pixel_values, output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        
        with torch.no_grad():
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            image_embeds = vision_outputs[0]
            image_feat = self.vision_proj(image_embeds[:, 0, :])
        return image_feat

    def labeled_feature(self, feat_input,label_input2,cap_feat):

        feat_input = feat_input.unsqueeze(1)   
        label_input2 = label_input2.unsqueeze(1)
        cap_feat=cap_feat.unsqueeze(1)
        feat_out,out_s =self.g_b(label_input2,feat_input,cap_feat)
        return feat_out.squeeze(1),out_s

    def get_text_image_features(self,text_feat,image_feat,label_feat2,cap_feat):

    
        if not self._weights_initialized:
            self._init_weights_()
        t_feat=normalize(self.labeled_feature(text_feat,label_feat2,cap_feat)[0],dim=-1)


        return t_feat
def criterion(img_features, txt_features, temp, device, args):
    if args.loss == 'contrastive':
        return losses.Contrastive(img_features, txt_features, temp, device)
    elif args.loss == 'recall':
        criterion = losses.RecallatK()
        return criterion(img_features, txt_features)
def gate_reg_loss(s, dialog_len, lam=1, base=0.1, max_s=0.2, max_len=10, p=1.0):

    dialog_len = torch.as_tensor(dialog_len, dtype=torch.float32, device=s.device)
    scale = ((dialog_len - 1) / (max_len - 1)) ** p
    s_max = base + scale * (max_s - base)

    while s_max.ndim < s.ndim:
        s_max = s_max.unsqueeze(-1)

    loss = lam * ((s - s_max).clamp(min=0)**2 ).mean()
    return loss

def multi_contrastive(t1, i2, bsz, tmp, device, dialog_len=None):
    # 不平均，逐样本计算 loss
    xent = nn.CrossEntropyLoss(reduction="none")
    targets = torch.arange(bsz, device=device)

    logits_t2i = torch.einsum("bd,bnd->bn", t1, i2) / tmp  
    loss_t2i = xent(logits_t2i, targets) 

    loss_t2i = loss_t2i.mean()

    return loss_t2i

def train_one_epoch(model, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None, processor=None,writer=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))
    global_step = epoch * len(data_loader)
    data_loader.epoch_ratio=epoch/args.epochs
    header = f"Epoch: [{epoch}]"
    for i, (imgs, txts, labels,labels2,dialog_len,cap) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        txts = processor(text=txts, padding=True, return_tensors='pt')
        caps=processor(text=cap, padding=True, return_tensors='pt')
        labels2 = processor(text=labels2, padding=True, return_tensors='pt')
        imgs = imgs.to(device)
        txts = txts.to(device)
        caps = caps.to(device)
        labels2 = labels2.to(device)
        batch_size = imgs.shape[0]
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            
            img_features = model.module.get_image_features(imgs)     
            caps_features=model.module.get_text_features(**caps)
            label2_features= model.module.get_text_features(**labels2)

            txt_features = model.module.get_text_features(**txts)
            txt_feats =model.module.get_text_image_features(txt_features,img_features,label2_features,caps_features)
            i_arr,t_arr,s_arr=[],[],[]
            for j in range(batch_size):

                label2=label2_features[j]
                label2_expand = label2.unsqueeze(0).expand(batch_size, label2_features.shape[-1])
                caption=caps_features[j]
                cap_expand = caption.unsqueeze(0).expand(batch_size, caps_features.shape[-1])
                i_space_feats,out_s=model.module.labeled_feature(img_features,label2_expand,cap_expand)
                s_arr.append(out_s)
                i_arr.append(normalize(i_space_feats,dim=-1))

            loss=multi_contrastive(txt_feats,torch.stack(i_arr, dim=0),batch_size,args.temp,device,dialog_len)
            g=gate_reg_loss(torch.stack(s_arr, dim=0),dialog_len)
            loss+=g
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

            optimizer.step()
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
        if writer is not None:
            
            writer.add_scalar('Loss/train', loss.item(), global_step + i)
            writer.add_scalar('LearningRate', optimizer.param_groups[0]["lr"], global_step + i)

def load_data(args, processor):
    # Data loading code
    logging.info("Loading data")

    logging.info("Loading training data")
    st = time.time()
    name = args.data_path.split('/')[-1]
    dataset_train = VisDial(args.data_path, split='train', transform=processor,recon_path=args.recon_path)
    dataset_test = VisDial(args.data_path, split='val', dialog_len=0, transform=processor)
    logging.info(f"Took {time.time() - st}")

    logging.info("Creating data loaders")
    logging.info('distributed:')
    logging.info(args.distributed)
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset_train, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset_train, dataset_test, train_sampler, test_sampler

def main(args):
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))
    checkpoint = None
    if args.output_dir:
        utils.mkdir(args.output_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'training.log')),
            logging.StreamHandler()
        ])
    logger = logging.getLogger()
    

    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    logging.info("Creating model")
    if args.torch_seed is not None:
        torch.manual_seed(args.torch_seed)
    model = BlipForRetrieval.from_pretrained("blip-itm-base-coco").to(device)
    processor = AutoProcessor.from_pretrained("blip-itm-base-coco")

    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    dataset, dataset_test, train_sampler, test_sampler = load_data(args, processor)
    logging.info(args)

    collate_fn = None
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=False,
        collate_fn=collate_fn
    )
    data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=args.test_batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))

    '''
    TODO: train only text encoder
    '''
    for name, p in model.named_parameters():
        if ("g_b" in name) :
            print(f'update {name}')
            p.requires_grad = True
        else:
            print(f'not update {name}')
            p.requires_grad = False

    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    '''
    TODO: loading test
    '''
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location="cpu",weights_only=False)
        checkpoint = checkpoint["model"]
        state_dict = {}
        for k, v in checkpoint.items():
            if 'heads' not in k:
                if 'labeled_encoder' in k:
                    state_dict[ "module." + "g_b."+k] = v
                else:
                    state_dict[ "module." + k] = v
        msg = model.load_state_dict(state_dict, strict=False)
        logging.info("Load pretrained model with msg: {}".format(msg))
    logging.info("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        logging.info('epoch %d'%epoch)
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, args, model_ema, scaler, processor=processor,writer=writer)
        lr_scheduler.step()
        checkpoint = {
            "model": model_without_ddp.state_dict(),
            "epoch": epoch,
            "args": args,
        }
        utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"epoch{epoch}.pth"))
        del checkpoint
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f"Training time {total_time_str}")
    writer.close()  
    if args.distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    ''' Add '''
    parser.add_argument("--torch-seed", type=int,default=42)
    parser.add_argument("--num-exp", type=int)
    parser.add_argument("--save-after-epoch", type=int, default=100)
    parser.add_argument("--pretrained", type=str)
    parser.add_argument("--loss", type=str, default="recall", choices=["contrastive", "recall"])
    parser.add_argument("--temp", type=float, default=1)
    parser.add_argument("--recon-path", type=str, help="recon path")
    parser.add_argument("--data-path", default="VisDial", type=str, help="dataset path")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=256, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--test-batch-size", default=100, type=int)
    parser.add_argument("--epochs", default=50, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--opt", default="adamw", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.0005, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--lr-scheduler", default="exponentiallr", type=str, help="the lr scheduler")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.93, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument(
        "--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224)"
    )
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
