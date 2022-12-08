import argparse
from torch.utils.data import DataLoader

from data_utils import create_examples
from tokenization import Tokenizer, PretrainedTokenizer
from trainer import Trainer
from data_loader import APIDataset

import re

def main(args):
    print(args)

    # Load tokenizer
    tokenizer_src = PretrainedTokenizer(pretrained_model = args.pretrained_model_src, vocab_file = args.vocab_file_src)
    tokenizer_tgt = PretrainedTokenizer(pretrained_model = args.pretrained_model_tgt, vocab_file = args.vocab_file_tgt)
    
    # Build DataLoader
    # train_dataset = create_examples(args, tokenizer_src, tokenizer_tgt, mode='train')
    # test_dataset = create_examples(args, tokenizer_src, tokenizer_tgt, mode='test')

    train_dataset=APIDataset(args.data_path+'train.desc.h5', args.data_path+'train.apiseq.h5', args.max_seq_len)
    test_dataset=APIDataset(args.data_path+'test.desc.h5', args.data_path+'test.apiseq.h5', args.max_seq_len)
    # train_loader=torch.utils.data.DataLoader(dataset=train_set, batch_size=config['batch_size'], shuffle=True, num_workers=1)
    # valid_loader=torch.utils.data.DataLoader(dataset=valid_set, batch_size=config['batch_size'], shuffle=True, num_workers=1)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Build Trainer
    trainer = Trainer(args, train_loader, test_loader, tokenizer_src, tokenizer_tgt, model_path=args.from_model)

    # match ep(\d+) from args.from_model and covert to int
    start_epoch = int(re.match(r'ep(\d+)', args.from_model).group(1)) + 1 if args.from_model != '' else 1

    # Train & Validate
    for epoch in range(start_epoch, args.epochs+1):
        trainer.train(epoch)
        trainer.validate(epoch)
        trainer.save(epoch, args.output_model_prefix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',                 default='data/eng-fra.txt', type=str, help='dataset')
    parser.add_argument('--data_path',                 default='data/', type=str, help='data_path')
    parser.add_argument('--vocab_file_src',          default='./data/vocab.desc.json',        type=str, help='vocabulary path')
    parser.add_argument('--vocab_file_tgt',          default='./data/vocab.apiseq.json',        type=str, help='vocabulary path')
    parser.add_argument('--pretrained_model_src',    default='fra.model',        type=str, help='pretrained sentencepiece model path. used only when tokenizer=\'sentencepiece\'')
    parser.add_argument('--pretrained_model_tgt',    default='eng.model',        type=str, help='pretrained sentencepiece model path. used only when tokenizer=\'sentencepiece\'')
    parser.add_argument('--output_model_prefix',     default='model',            type=str, help='output model name prefix')
    # Input parameters
    parser.add_argument('--batch_size',     default=64,   type=int,   help='batch size')
    parser.add_argument('--max_seq_len',    default=50,   type=int,   help='the maximum size of the input sequence')
    # Train parameters
    parser.add_argument('--epochs',         default=30,   type=int,   help='the number of epochs')
    parser.add_argument('--lr',             default=2,    type=float, help='initial learning rate')
    parser.add_argument('--no_cuda',        action='store_true')
    parser.add_argument('--multi_gpu',      action='store_true')
    # Model parameters
    parser.add_argument('--hidden',         default=256,  type=int,   help='the number of expected features in the transformer')
    parser.add_argument('--n_layers',       default=4,    type=int,   help='the number of heads in the multi-head attention network')
    parser.add_argument('--n_attn_heads',   default=4,    type=int,   help='the number of multi-head attention heads')
    parser.add_argument('--dropout',        default=0.1,  type=float, help='the residual dropout value')
    parser.add_argument('--ffn_hidden',     default=1024, type=int,   help='the dimension of the feedforward network')

    parser.add_argument('--from-model',     default='',   type=str,   help='load last ckpt')

    args = parser.parse_args()

    main(args)