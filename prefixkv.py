import torch
import numpy as np
import json

def slice2d(x, start, end):
    return x[:, :, start:end, ...]

def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]

def slice1d(x, start, end):
    return x[:, start:end, ...]

DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}

def obtain_cdf_num(score_sum, target_num, protect):
    all_sorted_scores = []
    for layer in range(len(score_sum)):
        score = score_sum[layer]
        sorted_score, index = score.sort(descending=True)
        sorted_score = sorted_score / sorted_score.sum()
        sorted_score = sorted_score.cumsum(dim=0)
        all_sorted_scores.append(sorted_score)
        
    all_sorted_scores = torch.stack(all_sorted_scores)
    
    left = 0
    right = 1
    mid = 0
    while(left < right):
        mid = (left + right) / 2.0
        index = torch.searchsorted(all_sorted_scores, torch.full((all_sorted_scores.size(0),1), mid, device=all_sorted_scores.device), right=False)
        count = all_sorted_scores.shape[-1] - (index.squeeze(-1) + 1)
        count = count - protect
        
        count[count < 0] = 0
        count = count.sum()
        if abs(count - target_num) < 5:
            break
        elif count < target_num:
            right = mid
        else:
            left = mid
    index = torch.searchsorted(all_sorted_scores, torch.full((all_sorted_scores.size(0),1), mid, device=all_sorted_scores.device), right=False)
    count = all_sorted_scores.shape[-1] - (index.squeeze(-1) + 1)
    count = count - protect
    
    count[count < 0] = 0
    return count.cpu().numpy()

class PrefixKV:
    def __init__(
        self,
        model_name = None,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        ratio=0.,
        distance=-25,
        layer_num=40,
        batch_size=1,
        profile=False
    ):
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]

        self.batch_size = batch_size
        self.protect_size = 1
        self.distance = distance
        self.layer_num = layer_num

        self.selected_idxs = []
        
        self.ratio = ratio
        self.model_name = model_name
        
        self.profile = profile

    def __call__(self, past_key_values, num_of_token=None, attentions=None):
        if past_key_values is None:
            return None
        attn_score = [attention.mean(dim=1) for attention in attentions]
        seq_lens = np.array([p[0].size(self.k_seq_dim) for p in past_key_values])

        if attn_score[0].shape[-2] > 1:
            attn_score = torch.stack(attn_score).float()
            self.score_sum = attn_score.sum(dim=-2)
            flag = True
        else:
            flag = False

        if flag:
            assert(np.all(seq_lens == num_of_token))
            assert(np.all(seq_lens == seq_lens[0]))
            
            if self.profile:
                target_num = (seq_lens.sum() * (self.ratio))
                forget_nums = obtain_cdf_num(self.score_sum[:, 0, :attn_score.shape[-1]], target_num, self.start_size+self.protect_size)
                assert(forget_nums.sum() >= 0)
                if forget_nums.sum() != 0: 
                    forget_nums = forget_nums * target_num / forget_nums.sum()
                else:
                    assert(target_num == 0)
                forget_nums = forget_nums.astype(np.int32)
                self.ratios = forget_nums / seq_lens
                with open(f'samples/prefixkv_{self.model_name}_{str(self.ratio)}.jsonl', 'a') as f:
                    f.write(json.dumps([f / attn_score.shape[-1] for f in forget_nums.tolist()]) + '\n')
                    f.flush()
            else:
                with open(f'confs/prefixkv_{self.model_name}_{str(self.ratio)}.json', 'r') as f:
                    self.ratios = np.array(json.load(f))
                forget_nums = (self.ratios * seq_lens).round().astype(np.int32)
        else:
            forget_nums = (seq_lens - num_of_token * (1 - self.ratios)).astype(np.int32)
            forget_nums[forget_nums < 0] = 0
        if np.all(forget_nums <= 0):
            return past_key_values
        else:
            if flag:
                past_key_values_return = []
                
                for idx in range(self.layer_num):
                    forget_num = forget_nums[idx]
                    assert(forget_num >= 0)
                    seq_len = seq_lens[idx]
                    selected_idx = torch.argsort(self.score_sum[idx, :, self.start_size:(seq_len - self.protect_size)])[:, forget_num:] + self.start_size
                    selected_idx = selected_idx.sort().values

                    device = selected_idx.device
                    pre = torch.arange(self.start_size, device=device).unsqueeze(0).expand(self.batch_size, -1)
                    post = torch.tensor([seq_len - self.protect_size], device=device).unsqueeze(0).expand(self.batch_size, -1)
                    selected_idx = torch.cat([pre, selected_idx, post], dim=-1) # the last token is always kept

                    if self.distance > 0:
                        self.selected_idxs.append(self.distance)
                    else:
                        self.selected_idxs.append(seq_len - forget_num + self.distance)
                        if not self.selected_idxs[-1] >= 1:
                            assert(selected_idx.shape[-1] >= 3)
                            self.selected_idxs[-1] = (selected_idx.shape[-1] // 2)
                            assert(self.selected_idxs[-1] >= 1 and self.selected_idxs[-1] <= selected_idx.shape[-1]-2)

                    k, v = past_key_values[idx]
                    selected_idx = selected_idx.to(k.device)

                    k_select = k.gather(dim=-2, index=selected_idx.view(self.batch_size,1,-1,1).expand(-1, k.shape[1], -1 ,k.shape[-1]))
                    v_select = v.gather(dim=-2, index=selected_idx.view(self.batch_size,1,-1,1).expand(-1, v.shape[1], -1 ,v.shape[-1]))

                    past_key_values_return.append([k_select, v_select])
                    
                return past_key_values_return
            else:
                past_key_values_return = []
                for i, (k,v) in enumerate(past_key_values):
                    if forget_nums[i] == 0:
                        past_key_values_return.append([k, v])
                        continue
                    seq_len = seq_lens[i]
                    selected_idx = self.selected_idxs[i]
                    past_key_values_return.append([torch.cat([self.k_slice(k, 0, selected_idx), self.k_slice(k, (selected_idx+1), seq_len),],
                                dim=self.k_seq_dim,),
                            torch.cat([self.v_slice(v, 0, selected_idx), self.v_slice(v, (selected_idx+1), seq_len),],
                                dim=self.v_seq_dim,)])
                return past_key_values_return
            