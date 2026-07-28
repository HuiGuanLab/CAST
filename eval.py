import torch
import tqdm
import os.path
import json
from PIL import Image
import torch.nn.functional as F
from transformers import AutoProcessor, BlipForImageTextRetrieval
import argparse
import clip
from torch.nn.functional import normalize
from typing import Any, Optional, Tuple, Union
import logging
from finetune import utils
import numpy as np
import torch.nn as nn
os.environ['TOKENIZERS_PARALLELISM']='true'

parser = argparse.ArgumentParser()
parser.add_argument('--retriever', type=str, default='clip', choices=["clip", "blip"])
parser.add_argument('--queries-path', type=str, default='dialogues/VisDial_v1.0_queries_val.json')
parser.add_argument('--ft-model-path', type=str)
parser.add_argument('--cache-corpus', type=str)
parser.add_argument('--data-dir', type=str, default='visdial/corpus')
parser.add_argument('--output-dir', type=str, default='logs')
parser.add_argument('--K', type=int, default=5)
parser.add_argument('--num-rounds', type=int, default=11)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--split', action='store_true', help="load dialog (caption) in split")
parser.add_argument('--start-round', type=int, default=1, help="the dialogue round to start evaluation from")
parser.add_argument('--recon_path', type=str)
cfg = {'corpus_bs': 256,
       'queries_bs': 256,
       'num_workers': 8,
       'sep_token': ', ',  # Separation between dialog rounds
       'queries_path': None,
       'corpus_path': 'Protocol/Search_Space_val_50k.json',
       'device': 'cuda:0',  # 'cpu'
       }

args = parser.parse_args()
retriever = args.retriever
queries_path = args.queries_path
cfg['data_dir'] = args.data_dir
cfg['queries_path'] = queries_path
cfg['split'] = args.split
cfg['finetuned_model_path']=args.ft_model_path
cfg['cache_corpus']=args.cache_corpus
cfg['K']=args.K
cfg['queries_bs']=args.batch_size
device = "cuda" if torch.cuda.is_available() else "cpu"
 
if args.output_dir:
    utils.mkdir(args.output_dir)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.output_dir, 'test.log')),
        logging.StreamHandler()
    ])
logger = logging.getLogger()


class DiffGate(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        in_dim = dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, dim), nn.GELU(),
            nn.Linear(dim, dim // 2), nn.GELU(),
            nn.Linear(dim // 2, 1)
        )
  
    def forward(self, cap, u, detach_inputs=True):
        z = torch.cat([u, cap], dim=-1)  # [B,2D]
        s = torch.sigmoid(self.mlp(z))      # [B,1] in (0,1)
        return s


class Guided_Block(nn.Module):
    def __init__(self, dim=256, rank=8, alpha_max=0.1, use_ln=True):
        super().__init__()
        self.dim, self.rank, self.alpha_max = dim, rank, alpha_max
        hid = dim*2
        self.mlp_A = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, dim * rank))
        self.mlp_B = nn.Sequential(nn.Linear(dim, hid), nn.GELU(),nn.Linear(hid, dim * rank))
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

    @staticmethod
    def _col_l2_norm(M, eps=1e-6):
        return M / (M.norm(dim=1, keepdim=True).clamp_min(eps))

    def forward(self, u_base: torch.Tensor, cls_in: torch.Tensor, cap_feat) -> Tuple[torch.Tensor, torch.Tensor]:
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
        A  = self.mlp_A(u).view(B, d, self.rank).contiguous()
        Bm = self.mlp_B(u).view(B, d, self.rank).contiguous()
        A = self._col_l2_norm(A)
        Bm= self._col_l2_norm(Bm)
        t = torch.bmm(x, Bm) 
        delta = torch.bmm(t, A.transpose(1, 2)) 
        s = self.gate(c, u)   
        s = s.to(x.dtype).unsqueeze(1) 
        y = x + s*delta
        y = self.ln(y)
        return y, s   


class BlipForRetrieval(BlipForImageTextRetrieval):
    def __init__(self, config):
        super().__init__(config)
        self.g_b=Guided_Block()
    def get_image_features(self, pixel_values, return_dict=None):
        with torch.no_grad():
            vision_outputs = self.vision_model(pixel_values=pixel_values, return_dict=return_dict)
            image_embeds = vision_outputs[0]
            image_feat = self.vision_proj(image_embeds[:, 0, :])
        return image_feat 

    def get_text_features(self, input_ids, attention_mask=None, return_dict=None, output_all=True):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_ids=input_ids.to('cuda')
        attention_mask=attention_mask.to("cuda")
        with torch.no_grad():
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)
            embeds = outputs[0] if not return_dict else outputs.last_hidden_state  
            feats = self.text_proj(embeds)  
            if output_all:
                return feats, attention_mask 
            else:
                return feats[:, 0, :]


class ImageEmbedder:
    def __init__(self, model, preprocessor):
        self.model = model
        self.processor = preprocessor


class Corpus(torch.utils.data.Dataset):
    def __init__(self, data_dir, corpus_path, preprocessor):
        with open(corpus_path) as f:
            self.corpus = json.load(f)
        self.corpus = [os.path.join(data_dir, path) for path in self.corpus]
        self.preprocessor = preprocessor
        self.path2id = {self.corpus[i]: i for i in range(len(self.corpus))}

    def __len__(self):
        return len(self.corpus)

    def path_to_index(self, path):
        return self.path2id[path]

    def __getitem__(self, i):
        if retriever == 'blip':
            image = self.preprocessor(self.corpus[i])['pixel_values'][0]
        else:
            image = self.preprocessor(self.corpus[i])
        return {'id': i, 'image': image}


with open(args.recon_path, "r") as f:
    d = json.load(f)


class Queries(torch.utils.data.Dataset):
    def __init__(self, cfg, queries_path, txt_processors):
        with open(queries_path) as f:
            self.queries = json.load(f)
        self.dialog_length = None
        self.cfg = cfg
        self.txt_processors = txt_processors

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, i):
        assert self.dialog_length is not None
        target_path = os.path.join(self.cfg['data_dir'], self.queries[i]['img'])
        if self.cfg['split']:
            text = self.queries[i]['dialog'][self.dialog_length]
        else:
            text = self.cfg['sep_token'].join(self.queries[i]['dialog'][:self.dialog_length + 1])
        if self.dialog_length == 0:
            return {'text': text, 'target_path': target_path}
        return {'text': text, 'target_path': target_path, 'label2': d[i]['dialog'][self.dialog_length],
                'cap': d[i]['dialog'][0]}



def get_first_hitting_time(target_recall, hitting_recall=10):
    start_round = getattr(args, "start_round", 0)
    total_rounds = getattr(args, "num_rounds", 0)
    effective_rounds = total_rounds - start_round
    target_recalls = target_recall.view(effective_rounds, -1).T
    hits = (target_recalls < hitting_recall)
    final_hits = torch.inf * torch.ones(target_recalls.shape[0], device=target_recalls.device)
    hitting_times, temp_hitting_times = [], []
    for ro_i in range(effective_rounds):
        temp_hits = torch.inf * torch.ones(target_recalls.shape[0], device=target_recalls.device)
        rh = hits[:, ro_i]
        final_hits[rh] = torch.min(final_hits[rh], torch.ones(final_hits[rh].shape, device=target_recalls.device) * ro_i)
        temp_hits[rh] = torch.min(temp_hits[rh], torch.ones(temp_hits[rh].shape, device=target_recalls.device) * ro_i)
        hitting_times.append(final_hits.clone())
        temp_hitting_times.append(temp_hits)
    return torch.stack(hitting_times), torch.stack(temp_hitting_times)


def cumulative_hits_per_round_multi(target_recall, ranks, targets, hitting_recalls=(1, 5, 10)):
    all_hits, all_temp_hits = {}, {}
    for k in hitting_recalls:
        ht_times, temp_ht_times = get_first_hitting_time(target_recall, hitting_recall=k)
        all_hits[k] = (ht_times < torch.inf).sum(dim=-1) * 100 / ht_times[0].shape[0]
        all_temp_hits[k] = (temp_ht_times < torch.inf).sum(dim=-1) * 100 / temp_ht_times[0].shape[0]
    return all_hits, all_temp_hits



class PlugIREval:
    def __init__(self, cfg, dialog_encoder, image_embedder: ImageEmbedder, txt_processors, model):
        self.dialog_encoder = dialog_encoder
        self.image_embedder = image_embedder
        self.txt_processors = txt_processors
        self.model = model
        self.cfg = cfg
        self.corpus = None
        self.corpus_dataset = Corpus(self.cfg['data_dir'], self.cfg['corpus_path'], self.image_embedder.processor)
        self.scores = {}
        self.ranks = []
        self.targets = []
        self.s_per_round = [[] for _ in range(args.num_rounds)]
        self.sim_per_round = [[] for _ in range(args.num_rounds)]

    def _get_recalls(self, dataloader, dialog_length):
        dataloader.dataset.dialog_length = dialog_length
        recalls = []
        ranks = []
        targets = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm.tqdm(dataloader)):
                if dialog_length!=0:
                    device = self.cfg['device']
                    target_ids = torch.tensor(
                        [self.corpus_dataset.path_to_index(p) for p in batch['target_path']],
                        device=device
                    ).unsqueeze(1)  

                    text_feat = self.dialog_encoder(batch['text'])[0][:,0,:]
                    labels2= self.dialog_encoder(batch['label2'])[0][:,0,:]
                    cap=self.dialog_encoder(batch['cap'])[0][:,0,:]
                    text_feat = text_feat.unsqueeze(1).to(device)     
                    label2_input = labels2.unsqueeze(1).to(device) 
                    cap_input=cap.unsqueeze(1).to(device)

                    text_feat, s_vals = self.model.g_b(label2_input,text_feat,cap_input)
                    text_feat = text_feat.squeeze(1)
                    self.s_per_round[dialog_length].append(s_vals.mean().item())
                    # 计算 caption 和 query 的相似度
                    batch_sim = F.cosine_similarity(cap, labels2, dim=-1).mean().item()
                    self.sim_per_round[dialog_length].append(batch_sim)

                    image_feat = self.corpus[1].to(device)          
                    B1, B2, D = text_feat.size(0), image_feat.size(0), image_feat.size(1)

                    image_expand = image_feat.unsqueeze(0).expand(B1, -1, -1).to(device)    
                    label2_expand = labels2.unsqueeze(1).expand(-1, B2, -1).to(device)
                    cap_expand = cap.unsqueeze(1).expand(-1, B2, -1).to(device)

                    x_all = torch.stack([label2_expand,image_expand,cap_expand], dim=2)    
                    x_all = x_all.view(-1, 3, D).to(device)                     

                    batch_size = 2048
                    num_pairs = x_all.size(0)
                    feat_out_chunks = []

                    for i in range(0, num_pairs, batch_size):
                        x_chunk = x_all[i:i + batch_size]                      
                        out_chunk, s_vals_chunk = self.model.g_b(
                            x_chunk[:,0,:].unsqueeze(1),
                            x_chunk[:,1,:].unsqueeze(1),
                            x_chunk[:,2,:].unsqueeze(1)
                        )
                        feat_out_chunks.append(out_chunk.squeeze(1))
                        self.s_per_round[dialog_length].append(s_vals_chunk.mean().item())

                    feat_out = torch.cat(feat_out_chunks, dim=0).view(B1, B2, D).to(device)
                    final_img_feat = F.normalize(feat_out, dim=-1)
                    text_feat = F.normalize(text_feat, dim=-1)

                    t_feat = text_feat.unsqueeze(1)                            
                    sim_scores = torch.bmm(t_feat, final_img_feat.transpose(1, 2)).squeeze(1)  
                    self.scores[i] = sim_scores.cpu()
                    arg_ranks = torch.argsort(sim_scores, dim=1, descending=True)  
                    target_recall = ((arg_ranks - target_ids) == 0).nonzero()[:, 1]  

                    recalls.append(target_recall)
                    ranks.append(arg_ranks.cpu())
                    targets.append(target_ids.cpu())
                else:
    
                    target_ids = torch.tensor([self.corpus_dataset.path_to_index(p) for p in batch['target_path']]).unsqueeze(1).to(self.cfg['device'])
                    pred_vec = F.normalize(self.dialog_encoder(batch['text'])[0][:,0,:], dim=-1).to(self.cfg['device']) 
                    self.scores[i] = pred_vec @ F.normalize(self.corpus[1],dim=-1).to(self.cfg['device']).T
                    arg_ranks = torch.argsort(self.scores[i], descending=True, dim=1).long().to(self.cfg['device'])
                    target_recall = ((arg_ranks - target_ids) == 0).nonzero()[:, 1]
                    recalls.append(target_recall)
                    ranks.append(arg_ranks)
                    targets.append(target_ids)
                    self.scores[i] = self.scores[i].cpu()

        return torch.cat(recalls).cpu(), torch.cat(ranks).cpu(), torch.cat(targets).cpu()


    def run(self, hits_at):
        assert self.corpus, f"Prepare corpus first (self.index_corpus())"
        dataset = Queries(cfg, self.cfg['queries_path'], self.txt_processors)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg['queries_bs'],
            shuffle=False,
            num_workers=self.cfg['num_workers'],
            pin_memory=True,
            drop_last=False
        )

        start_round = getattr(args, "start_round", 0)
        total_rounds = args.num_rounds
        Ks = [1, 5, 10]

        logging.info(f"====== Evaluation Settings ======")
        logging.info(f"Start Round: {start_round}")
        logging.info(f"End Rounds: {total_rounds}")
        logging.info(f"Compute Recall@{Ks}")

        hits_results, ranks_results, targets_results, min_ranks = [], [], [], []

        for dl in range(start_round, total_rounds):
            logging.info(f"Calculating recalls for dialogue length {dl} ...")
            dialog_recalls, ranks, targets = self._get_recalls(dataloader, dialog_length=dl)
            if dl == start_round:
                min_ranks.append(dialog_recalls)
            else:
                min_ranks.append(torch.minimum(min_ranks[-1], dialog_recalls))
            hits_results.append(dialog_recalls)
            ranks_results.append(ranks)
            targets_results.append(targets)

        hits_dict, temp_hits_dict = cumulative_hits_per_round_multi(
            torch.cat(hits_results),
            torch.cat(ranks_results),
            torch.cat(targets_results),
            hitting_recalls=Ks
        )

        for k in Ks:
            logging.info(f"====== Results for Hits@{k} ======")
            for idx, dl in enumerate(range(start_round, total_rounds)):
                logging.info(f"\t Dialog Length: {dl}: {round(hits_dict[k][idx].item(), 2)}%")
            logging.info(f"====== Results for Recall@{k} ======")
            for idx, dl in enumerate(range(start_round, total_rounds)):
                logging.info(f"\t Dialog Length: {dl}: {round(temp_hits_dict[k][idx].item(), 2)}%")

        logging.info(f"====== Best log Rank Integral ======")
        bri = 0
        for idx in range(len(min_ranks) - 1):
            bri += ((torch.log(min_ranks[idx] + 1.) + torch.log(min_ranks[idx + 1] + 1.)) / 2).mean()
        bri /= max(1, len(min_ranks) - 1)
        logging.info(f"\t BRI: {bri}")

        logging.info(f"====== Average s & cosine similarity per round ======")
        for dl in range(start_round, total_rounds):
            if self.s_per_round[dl]:
                avg_s = np.mean(self.s_per_round[dl])
                avg_sim = np.mean(self.sim_per_round[dl]) if self.sim_per_round[dl] else float("nan")
                logging.info(f"\t Dialog Length {dl}: avg_s={round(avg_s,4)}, avg_sim={round(avg_sim,4)}")
            elif self.sim_per_round[dl]:
                logging.info(f"\t Dialog Length {dl}:avg_sim={round( np.mean(self.sim_per_round[dl]) ,4)}")
            else:
                logging.info(f"\t Dialog Length {dl}: N/A")

    def index_corpus(self):
        print(self.cfg['cache_corpus'])
        print(os.path.exists(self.cfg['cache_corpus']))
        if self.cfg['cache_corpus'] and os.path.exists(self.cfg['cache_corpus']):
            logging.info(f"<<<<Cached corpus has been loaded: {self.cfg['cache_corpus']} >>>>>")
            logging.info(f"Warning: Make sure this corpus has been indexed with the right image embedder!")
            self.corpus = torch.load(self.cfg['cache_corpus'])
            return
        dataloader = torch.utils.data.DataLoader(self.corpus_dataset,
                                                 batch_size=self.cfg['corpus_bs'],
                                                 shuffle=False,
                                                 num_workers=self.cfg['num_workers'],
                                                 pin_memory=True,
                                                 drop_last=False
                                                 )
        logging.info("Preparing corpus (search space)...")
        corpus_vectors = []
        corpus_ids = []
        for batch in tqdm.tqdm(dataloader):
            batch_vectors =self.image_embedder.model(batch['image'].to(self.cfg['device']))
            corpus_vectors.append(batch_vectors)
            corpus_ids.append(batch['id'].to(self.cfg['device']))

        corpus_vectors = torch.cat(corpus_vectors)
        corpus_ids = torch.cat(corpus_ids)
        arg_ids = torch.argsort(corpus_ids)
        arg_ids = arg_ids.to(self.cfg['device'])
        corpus_vectors = corpus_vectors.to(self.cfg['device'])
        corpus_ids = corpus_ids.to(self.cfg['device'])
        corpus_vectors = corpus_vectors[arg_ids]
        corpus_ids = corpus_ids[arg_ids]
        self.corpus = corpus_ids, corpus_vectors
        if self.cfg['cache_corpus']:
            torch.save(self.corpus, self.cfg['cache_corpus'])


def CLIP_ZERO_SHOT_BASELINE():
    model, preprocess = clip.load("ViT-B/16", device='cpu')
    model = model.to(device)
    image_embedder = ImageEmbedder(lambda img: model.encode_image(img), lambda path: preprocess(Image.open(path)))
    dialog_encoder = lambda text: model.encode_text(clip.tokenize(text, truncate=True).to(device))
    return dialog_encoder, image_embedder


def BLIP_ZERO_SHOT_BASELINE(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BlipForRetrieval.from_pretrained("Salesforce/blip-itm-large-coco")
    processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-large-coco")
    if cfg['finetuned_model_path']:
        ckpt = torch.load(cfg['finetuned_model_path'], map_location="cpu", weights_only=False)
        state_dict = ckpt['model']
        msg = model.load_state_dict(state_dict, strict=False)
        logging.info(f"Load pretrained model with msg: {msg}")
    model = model.to(device)
    model.eval()
    image_embedder = ImageEmbedder(lambda img: model.get_image_features(img),
                                   lambda path: processor(images=Image.open(path), return_tensors='pt'))
    dialog_encoder = lambda text: model.get_text_features(**processor(text=text,
                                                                      padding=True,
                                                                      truncation=True,
                                                                      return_tensors="pt"))
    return dialog_encoder, image_embedder, model


with torch.no_grad():
    txt_processors = None
    if retriever == 'blip':
        dialog_encoder, image_embedder, model = BLIP_ZERO_SHOT_BASELINE(cfg)
    else:
        dialog_encoder, image_embedder = CLIP_ZERO_SHOT_BASELINE()
        model = None
    model.eval()
    evaluator = PlugIREval(cfg, dialog_encoder, image_embedder, txt_processors, model)
    evaluator.index_corpus()
    evaluator.run(hits_at=(1, 5, 10))
