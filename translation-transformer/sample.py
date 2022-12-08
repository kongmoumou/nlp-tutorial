import argparse
from tokenization import PretrainedTokenizer
import numpy as np
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import os, sys
parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules
from helper import indexes2sent
from inference import infer
from metrics import Metrics
from data_loader import APIDataset, load_dict, load_vecs


def evaluate(model, metrics, test_loader, vocab_desc, vocab_api, repeat, decode_mode, f_eval,
                tokenizer_src, tokenizer_tgt):
    ivocab_api = {v: k for k, v in vocab_api.items()}
    ivocab_desc = {v: k for k, v in vocab_desc.items()}
    device = next(model.parameters()).device
    # device = 'cuda:0'
    print('evaluate use device:', device)
    
    recall_bleus, prec_bleus = [], []
    local_t = 0
    import time
    for descs, decoder_input_ids, apiseqs in tqdm(test_loader):
        
        # if local_t>1000:
        #     break        
        
        desc_str = indexes2sent(descs[0].numpy(), vocab_desc)

        t = time.time()
        descs = descs.to(device)
        with torch.no_grad():
            # sample_words, sample_lens = model.sample(descs, desc_lens, repeat, decode_mode)
            sample_words = infer(model, descs, tokenizer_tgt)
        # print(f'sample time: {time.time()-t}')

        t = time.time()
        # nparray: [repeat x seq_len]
        pred_sents, _ = indexes2sent(sample_words[:,1:], vocab_api)
        pred_tokens = [sent.split(' ') for sent in pred_sents]
        ref_str, _ =indexes2sent(apiseqs[0].numpy()[:-1], vocab_api)
        ref_tokens = ref_str.split(' ')
        # print(f'detoken time: {time.time()-t}')
        
        max_bleu, avg_bleu = metrics.sim_bleu(pred_tokens, ref_tokens)
        recall_bleus.append(max_bleu)
        prec_bleus.append(avg_bleu)

        local_t += 1 
        f_eval.write("Batch %d \n" % (local_t))# print the context        
        f_eval.write(f"Query: {desc_str} \n")
        f_eval.write("Target >> %s\n" % (ref_str.replace(" ' ", "'")))# print the true outputs 
        for r_id, pred_sent in enumerate(pred_sents):
            f_eval.write("Sample %d >> %s\n" % (r_id, pred_sent.replace(" ' ", "'")))
        f_eval.write("\n")

    recall_bleu = float(np.mean(recall_bleus))
    prec_bleu = float(np.mean(prec_bleus))
    f1 = 2*(prec_bleu*recall_bleu) / (prec_bleu+recall_bleu+10e-12)
    
    report = "Avg recall BLEU %f, avg precision BLEU %f, F1 %f"% (recall_bleu, prec_bleu, f1)
    print(report)
    f_eval.write(report + "\n")
    print("Done testing")
    
    return recall_bleu, prec_bleu

def main(args):
    # conf = getattr(configs, 'config_'+args.model)()
    # Set the random seed manually for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        print('cuda is available~')
        torch.cuda.manual_seed(args.seed)
    else:
        print("Note that our pre-trained models require CUDA to evaluate.")

    # Load tokenizer
    tokenizer_src = PretrainedTokenizer(pretrained_model = None, vocab_file = args.vocab_file_src)
    tokenizer_tgt = PretrainedTokenizer(pretrained_model = None, vocab_file = args.vocab_file_tgt)
    
    # Load data
    test_set=APIDataset(args.data_path+'test.desc.h5', args.data_path+'test.apiseq.h5', args.max_seq_len)
    test_loader=torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=1)
    vocab_api = load_dict(args.data_path+'vocab.apiseq.json')
    vocab_desc = load_dict(args.data_path+'vocab.desc.json')
    metrics=Metrics()
    
    # Load model checkpoints   
    # model = getattr(models, args.model)(conf)
    # ckpt=f'./output/{args.model}/{args.expname}/{args.timestamp}/models/model_epo{args.reload_from}.pkl'
    ckpt=f'./output/model_epo120000.pkl'
    # fix: default load model to cpu, f**king slow
    device = torch.device("cuda")
    # model.load_state_dict(torch.load(ckpt, map_location="cuda:0"))
    # model.to(device)
    device = 'cuda'
    model_path = '.model/model_22-12-08_1256.ep6'
    model = torch.load(model_path).to(device)
    # model = torch.load(args.model).to(device)
    model.eval()

    import time
    # time in MM-DD-HH-mm format
    time_str = time.strftime("%m-%d-%H-%M", time.localtime())
    
    # f_eval = open(f"./output/{args.model}/{args.expname}/{time_str}results.txt".format(args.model, args.expname), "w")
    f_eval = open(f"./output/{args.model}/{time_str}results.txt".format(args.model, args.expname), "w")
    
    # save args to result
    f_eval.write(f"Args: {vars(args)} \n")

    evaluate(model, metrics, test_loader, vocab_desc, vocab_api, args.n_samples, args.decode_mode , f_eval,
                tokenizer_src, tokenizer_tgt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch DeepAPI for Eval')
    parser.add_argument('--data_path', type=str, default='./data/', help='location of the data corpus')
    parser.add_argument('--model', type=str, default='transformer', help='model name')
    parser.add_argument('--expname', type=str, default='basic', help='experiment name, disinguishing different parameter settings')
    parser.add_argument('--timestamp', type=str, default='201909270147', help='time stamp')
    parser.add_argument('--reload_from', type=int, default=10000, help='directory to load models from')
    
    parser.add_argument('--n_samples', type=int, default=10, help='Number of responses to sampling')
    parser.add_argument('--decode_mode', type=str, default='sample',
                        help='decoding mode for generation: beamsearch, greedy or sample')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')

    parser.add_argument('--vocab_file_src',          default='data/vocab.desc.json',     type=str, help='vocabulary path')
    parser.add_argument('--vocab_file_tgt',          default='data/vocab.apiseq.json',     type=str, help='vocabulary path')
    parser.add_argument('--max_seq_len',    default=50,  type=int,   help='the maximum size of the input sequence')

    args = parser.parse_args()
    print(vars(args))
    main(args)
