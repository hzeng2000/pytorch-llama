from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from model import ModelArgs, Transformer
from tqdm import tqdm

class LLaMA:
    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.model_args = model_args
        
    @staticmethod
    def build(checkpoint_path: str, tokenizer_path: str, load_model: bool, 
              max_seq_len: int, max_batch_size, device: str):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoint_path).glob("*.pth"))
            assert len(checkpoints) > 0, "no checkpoint files found"   
            chk_path = checkpoints[0]
            print(f"loading checkpoint {chk_path}")
            checkpoint = torch.load(chk_path, map_location=device)     
            print(f"loaded checkpoint {chk_path} in {time.time() - prev_time:.2f}s")    
            prev_time = time.time()
            
        with open(Path(checkpoint_path) / "params.json", "r") as f:
            params = json.load(f)
            
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params,
        )
        
        tokenizer = SentencePieceProcessor()
        tokenizer.Load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()
        
        if device == "cuda":
            torch.set_default_dtype()(torch.bfloat16)
        else:
            torch.set_default_dtype(torch.float32)
        
        model = Transformer(model_args).to(device=device)
        if load_model:
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint, strict=True)
            print(f'load state dict in {(time.time() - prev_time):.2f}s')
            
        return LLaMA(model, tokenizer, model_args)
    
    def text_generation(self, prompts: list[str], temperature: float=0.6, 
                        top_p: float=0.9, max_gen_len: Optional[int]=None) -> list[str]:
        if max_gen_len is None:
            max_gen_len = self.model_args.max_seq_len
            
        # 将输入的prompt转换为token
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        batch_size = len(prompt_tokens)
        assert batch_size <= self.model_args.max_batch_size, "batch size too large"
        max_prompt_len = max(len(tokens) for tokens in prompt_tokens)
        # 确保输入的tokens长度没有超过最大长度
        assert max_prompt_len <= self.model_args.max_seq_len, "prompt length too long"
        total_len = min(max_prompt_len + max_gen_len, self.model_args.max_seq_len)
        
        # 创建包含生成tokens的list
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(prompt_tokens):
            # prompts填充
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)
            
        eos_reached = torch.tensor([False] * batch_size, device=device)
        prompt_tokens_mask = tokens != pad_id
        
        # 循环decode
        for cur_pos in tqdm(range(1, total_len), desc='Generating tokens'):
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos-1:cur_pos], cur_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)
            
            next_token = next_token.reshape(-1)
            # 当不是pad标记的时候才替换
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            # eos也同理
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id())
            if all(eos_reached):
                break
        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # 去除eos
            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id())
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        return (out_tokens, out_text)
    
    def _sample_top_p(self, probs, p):
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        prob_sum = torch.cumsum(sorted_probs, dim=-1)
        mask = prob_sum - sorted_probs > p
        sorted_probs[mask] = 0.0
        sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(sorted_probs, num_samples=1)
        next_token = torch.gather(sorted_indices, dim=-1, index=next_token)
        return next_token
        
if __name__ == "__main__":
    torch.manual_seed(0)
    
    # allow_cuda = False
    allow_cuda = True
    device = torch.device("cuda" if torch.cuda.is_available() and allow_cuda else "cpu")
    prompts = [
        "describe the mechanism of the deep learning network inference acceleration technique" 
    ]
    model = LLaMA.build(
        checkpoint_path="/home/hzeng/dataset/llama-2-7b",
        tokenizer_path="/home/hzeng/dataset/tokenizer.model",
        load_model=True,
        max_seq_len=512,
        max_batch_size=3,
        device=device,
    )
    # 推理
    out_tokens, out_text = model.text_generation(prompts, temperature=0.6, top_p=0.9, max_gen_len=64)
    assert len(out_tokens) == len(prompts)
    for i in range(len(prompts)):
        print(f"prompt: {prompts[i]}")
        print(f"output: {out_text[i]}")
        print(f"output tokens: {out_tokens[i]}")