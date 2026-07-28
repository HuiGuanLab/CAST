import requests
import openai
import os
import json
import time
from func_timeout import func_timeout, FunctionTimedOut
API_KEY = "your api key"
API_URL = "api url"


def _image_id_to_file_name(split,data):
        image_id_to_file_name = {}
        if split == 'val':
            prefix = 'VisualDialog_val2018_'
            for dialog in data['data']['dialogs']:
                image_id = dialog['image_id']
                file_name = prefix + "%012d.jpg" % image_id
                image_id_to_file_name[image_id] = file_name
        else:
            prefix_train = 'COCO_train2014_'
            for dialog in data['data']['dialogs']:
                image_id = dialog['image_id']
                file_name = prefix_train + "%012d.jpg" % image_id

        return image_id_to_file_name

def reconstruct_dialog(dial):

    caption = dial[0]
    dialog = ', '.join(dial[1:])
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system",
                 "content": "Your role is to reconstruct the [Caption] with the additional information given by following [Dialogue]. "
                 "The reconstructed [New Caption] should be concise and in appropriate form to retrieve a target image from a pool of candidate images"},
           
        ]
    }
    dialog_examplar = ', '.join(["is this in a park? yes, i believe it is", "are there others around? no, she is alone",
                                 "does she have a collection bucket? no", "is her hair long? yes, pretty long",
                                 "is she wearing a dress? i don't think so, hard to tell",
                                 "does she have shoes on? yes, flip flops", "is there grass nearby? yes, everywhere",
                                 "is it a sunny day? yes", "are there trees? in the background there are trees",
                                 "is the guitar new? i don't think so"])

   
    data['messages'].append({"role": "user", "content": f"[Caption]: {caption} [Dialogue]: {dialog}  [New Caption]: "})
    def make_request():
        return requests.post(API_URL, headers=headers, json=data)
    st = time.time()
    try:
        response = func_timeout(30, make_request)
        
    except FunctionTimedOut:
        print("time out")
        return None, None
    except:
        print("fail")
        return None, None
        
    if response.status_code == 200:
        result = response.json()['choices'][0]['message']['content']
    else:
        print("status_code", response.status_code)
        print("message:", response.text)
        return None, None
    ed = time.time()

    return result, round(ed - st)
def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="Dialog Reconstruction", add_help=add_help)
    parser.add_argument("--run_idx", type=int)
    parser.add_argument("--split", type=str, default='train')
    return parser

args = get_args_parser().parse_args()
split = args.split
run_idx = args.run_idx
root = os.path.join('VisDial' , split)
_data_path = os.path.join(root)
with open(os.path.join(_data_path, f'visdial_1.0_{split}.json'), "r") as f:
    _data = json.load(f)
image_id_to_file_name = _image_id_to_file_name(split, _data)
_questions = _data['data']['questions']
_answers = _data['data']['answers']
save_root = os.path.join('dial_recon')
all_data_num = len(_data['data']['dialogs'])
split_num = 4
data_split = all_data_num // split_num
idx_interval = [[i*data_split, (i+1)*data_split] for i in range(split_num-1)]
idx_interval.append([idx_interval[-1][1], all_data_num])
cnt = 0
for i in range(idx_interval[run_idx][0], idx_interval[run_idx][1]):
    cnt += 1
    data = _data['data']['dialogs'][i]
    file_name = image_id_to_file_name[data['image_id']]
    file_name = file_name.split('.jpg')[0] + '.txt'
    captions = [data['caption']]
    save_path = os.path.join(save_root, file_name)
    if os.path.exists(save_path):
        continue
    else:
        recon_caps = []
        qa_caps = []
        time_all = 0
        for k in range(10):
            print('process dialog qa:', k)
            qa = _questions[data['dialog'][k]['question']] + '? ' + _answers[data['dialog'][k]['answer']]
            captions.append(qa)
            qa_caps.append(qa)
            recon, use_time = reconstruct_dialog(captions)
            while recon is None:
                print('reconstruct failed, retry')
                recon, use_time = reconstruct_dialog(captions)
            recon_caps.append(recon)
            time_all += use_time

        f1 = open(save_path, 'w')
        f1.write(data['caption'] + '\n')
        for k in range(len(recon_caps)):
            f1.write(recon_caps[k] + '\n')
        for k in range(len(qa_caps)):
            f1.write(qa_caps[k] + '\n')

        print(file_name+' finished, time:', time_all, 's', 'remain_dialogs:',
               idx_interval[run_idx][1] - idx_interval[run_idx][0] + 1 - cnt)