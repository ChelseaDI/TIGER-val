import argparse
import logging

import torch
import os

import json

from utils import set_seed, RecDataset, RecCollator, ensure_dir, Trie, prefix_allowed_tokens_fn, \
    setup_logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = 'false'


from experiment import CollisionEvaluator, rank_candidates, normalize_sid, parse_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ckpt & dataset
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints')
    parser.add_argument('--dataset', type=str, default='Games')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--plm_dir', type=str, default='../LLM/')
    parser.add_argument('--plm_name', type=str, default='t5-base')
    parser.add_argument('--tokenizer_plm', type=str, default='sentence-t5-base')
    parser.add_argument('--token_type', type=str, default='pretrained_nc', choices=['sid', 'cid', 'pretrained', 'sid_nc', 'pretrained_nc'])

    # hyper-param
    parser.add_argument('--num_beams', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--K', type=int, default=256, help='codebook size')
    parser.add_argument('--D', type=int, default=3, help='codebook num')
    parser.add_argument('--item_sep', type=str, default=',', help='separation between two item token sequence')
    parser.add_argument('--max_len', type=int, default=20, help='max length of user interation sequence')
    parser.add_argument('--max_sent_len', type=int, default=512, help='max length of model input sequence')
    parser.add_argument('--add_user_prefix', type=bool, default=False,
                        help='whether add user prefix in the begin of sequence')
    parser.add_argument('--user_prefix', type=str, default='<u_{}>', )

    # metrics
    parser.add_argument('--metrics', type=str, default="['hit@5', 'hit@10', 'ndcg@5', 'ndcg@10']")

    parser.add_argument('--results_file', type=str, default=None, help='Save JSON experiment results')
    args = parser.parse_args()
    parse_metrics(args.metrics)
    if args.num_beams < 1:
        parser.error('--num_beams must be positive')

    set_seed()
    setup_logging()

    logging.info('hyper-param:')
    for param, value in vars(args).items():
        logging.info("  %s: %s", param, value)

    # load checkpoint
    ckpt_dir = os.path.join(args.ckpt_dir, args.token_type)
    if args.token_type in ('sid', 'sid_nc'):
        dataset_name = args.dataset + '_' + args.tokenizer_plm
    else:
        dataset_name = args.dataset
    ckpt_dir = os.path.join(ckpt_dir, dataset_name)
    token_size = f'K={args.K}_D={args.D}'
    ckpt_dir = os.path.join(ckpt_dir, token_size)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    # model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir, torch_dtype=torch.bfloat16)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir, torch_dtype=torch.float32)

    model.to(args.device)
    model.eval()

    test_dataset = RecDataset(args, mode='test')

    all_items = test_dataset.get_all_items()

    evaluator = CollisionEvaluator(test_dataset.indices, args.D, args.token_type, args.metrics)
    if not len(test_dataset):
        raise ValueError('Test dataset is empty')
    if max(int(m.split('@')[1]) for m in evaluator.metrics) > args.num_beams:
        logging.warning('Some metric cutoffs exceed the beam count; ranks beyond beams are unavailable')
    candidate_ids = [tokenizer.encode(candidate) for candidate in sorted(all_items)]
    for code, ids in zip(sorted(all_items), candidate_ids):
        if normalize_sid(tokenizer.decode(ids, skip_special_tokens=True)) != code:
            raise ValueError('Checkpoint tokenizer does not represent this SID mapping: ' + code)
    candidate_trie = Trie(
        [
            [model.config.decoder_start_token_id] + ids for ids in candidate_ids
        ]
    )
    prefix_allowed_tokens_fn = prefix_allowed_tokens_fn(candidate_trie)

    collator = RecCollator(args, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collator, pin_memory=True,
                                 num_workers=4)


    with torch.no_grad():
        # for batch in tqdm(test_dataloader, desc='[Testing]', disable=True):
        for batch in tqdm(test_dataloader, desc='[Testing]', mininterval=5):
            input_ids, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['labels']
            input_ids, attention_mask = input_ids.to(args.device), attention_mask.to(args.device)

            # for the meaning of each parameter, please refer to https://huggingface.co/docs/transformers/v4.18.0/en/main_classes/text_generation
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max(len(ids) for ids in candidate_ids),
                num_beams=args.num_beams,
                num_return_sequences=args.num_beams,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                output_scores=True,
                return_dict_in_generate=True,
                early_stopping=True,
            )

            output_ids = output['sequences']
            # the credibility of the generated indices, also can be regarded as the user's preference for the item in the recommendation system(higher is better).
            # 分数是概率经过log后的结果，所以都是负数。由于log是单调函数，即概率越大，score越大(越接近于0)
            scores = output['sequences_scores']

            predictions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            targets = tokenizer.batch_decode(targets, skip_special_tokens=True)

            scores = scores.detach().tolist()
            for b, target in enumerate(targets):
                lo, hi = b * args.num_beams, (b + 1) * args.num_beams
                ranked = rank_candidates(predictions[lo:hi], scores[lo:hi], all_items)
                evaluator.add(ranked, target)

    result = evaluator.result()
    result['settings'] = vars(args)
    result['checkpoint'] = ckpt_dir
    logging.info('evaluation result:\n%s', json.dumps(result, indent=2, ensure_ascii=False))
    if args.results_file:
        ensure_dir(os.path.dirname(os.path.abspath(args.results_file)))
        with open(args.results_file, 'w') as fp:
            json.dump(result, fp, indent=2, ensure_ascii=False)
