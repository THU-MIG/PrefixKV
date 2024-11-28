import torch
from tqdm import tqdm
import os
import json
import argparse
import torch
from cache_generate import generate, sample, greedy_search
import types
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from rouge import Rouge
from PIL import Image
import requests
from PIL import Image
from io import BytesIO
from transformers import set_seed
from prefixkv import PrefixKV
from patch_attention_forward import patch_llama_attention_forward
import datetime
from accelerate import Accelerator
from accelerate.utils import gather_object
import random


def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def main(args):
    set_seed(0)
    accelerator = Accelerator()

    if args.method == "prefixkv":
        patch_llama_attention_forward()

    with open(args.data_path, "r") as f:
        data = json.load(f)

    model_name = get_model_name_from_path(args.model_path)

    if accelerator.num_processes == 1:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            args.model_path,
            args.model_base,
            model_name,
            args.load_8bit,
            args.load_4bit,
            device="cuda",
        )
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            args.model_path,
            args.model_base,
            model_name,
            args.load_8bit,
            args.load_4bit,
            device=accelerator.device,
            device_map=accelerator.device,
        )

    model.generate = types.MethodType(generate, model)
    model.sample = types.MethodType(sample, model)
    model.greedy_search = types.MethodType(greedy_search, model)

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    k_seq_dim = v_seq_dim = 2
    score_all = []
    outputs_all = []
    os.makedirs(f"logs/{args.exp_name}", exist_ok=True)

    data = data[: args.eval_samples]
    random.shuffle(data)

    with accelerator.split_between_processes(data, apply_padding=True) as batched_data:
        for item in tqdm(batched_data):
            if args.method == "prefixkv":
                kv_cache = PrefixKV(
                    model_name=model_name,
                    start_size=args.start_size,
                    recent_size=args.recent_size,
                    k_seq_dim=k_seq_dim,
                    v_seq_dim=v_seq_dim,
                    ratio=args.ratio,
                    layer_num=32 if "7b" in model_name else 40,
                    profile=args.profile,
                )
            else:
                kv_cache = None

            conv = conv_templates[args.conv_mode].copy()
            image_path = os.path.join(args.image_path, item["image"])
            question = item["question"]
            answer = item["answer"]

            image = load_image(image_path)
            image_tensor = process_images([image], image_processor, args)
            image_tensor = image_tensor.to(model.device, dtype=torch.bfloat16)

            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = (
                tokenizer_image_token(
                    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .cuda()
            )
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(
                keywords, tokenizer, input_ids)

            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if (args.temperature >
                                   0 and args.ratio == 0) else False,
                temperature=args.temperature if args.ratio == 0 else 0,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                kv_cache_criteria=kv_cache,
            )

            outputs_generate = tokenizer.decode(
                output_ids[0, input_ids.shape[1]:]
            ).strip()
            rouge = Rouge()
            scores = rouge.get_scores(outputs_generate, answer)
            score_all.append(scores[0]["rouge-l"]["f"])
            item["pred"] = outputs_generate
            item["score"] = scores[0]["rouge-l"]["f"]
            outputs_all.append(item)

    score_all = gather_object(score_all)[: len(data)]
    rouge = sum(score_all) / len(score_all)
    outputs_all = gather_object(outputs_all)[: len(data)]
    if accelerator.is_main_process:
        time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        with open(f"logs/{args.exp_name}/{time}.txt", "a") as f:
            f.write(f"Args:\n")
            f.write(json.dumps(vars(args), indent=2))
            f.write("\n")
            f.write(f"{rouge}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str,
                        default="./models/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data-path", type=str,
                        default="./data/mm-vet/mm-vet.json")
    parser.add_argument("--image-path", type=str,
                        default="./data/mm-vet/images")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default="pad")
    parser.add_argument("--start-size", type=int, default=1)
    parser.add_argument("--recent-size", type=int, default=2047)
    parser.add_argument("--eval-samples", type=int, default=100)
    parser.add_argument("--exp-name", type=str, default="llava-7b")
    parser.add_argument("--method", type=str, default="elastic")
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--profile", action="store_true", default=False)
    args = parser.parse_args()
    args.exp_name = args.exp_name + "/" + args.method + "/" + str(args.ratio)
    main(args)
