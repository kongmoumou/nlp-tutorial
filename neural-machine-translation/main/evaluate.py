import time
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import dataLoader as loader
import seq2seq
import train
from nltk.translate.bleu_score import sentence_bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(sentence, encoder, decoder, max_length=loader.MAX_LENGTH):
    with torch.no_grad():
        input_tensor = train.tensorFromSentence(input_lang, sentence)
        # |input_tensor| = (sentence_length, 1)
        input_length = input_tensor.size(0)
        
        encoder_hidden = (encoder.initHidden().to(device), encoder.initHidden().to(device))
        # |encoder_hidden[0]|, |encoder_hidden[1]| = (num_layers*num_directions, batch_size, hidden_size/2)
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size).to(device)
        # |encoder_outputs| = (max_length, hidden_size)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            # |encoder_output| = (batch_size, sequence_length, num_directions*(hidden_size/2))
            # |encoder_hidden| = (2, num_layers*num_directions, batch_size, hidden_size/2)
            # 2: respectively, hidden state and cell state.
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[loader.SOS_token]], device=device)
        # |decoder_input| = (1, 1)
        decoder_hidden = train.merge_encoder_hiddens(encoder_hidden)
        # |decoder_hidden|= (2, num_layers*num_directions, batch_size, hidden_size)
        # 2: respectively, hidden state and cell state.
        # Here, the lstm layer in decoder is uni-directional.

        decoded_words=[]
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input,
                                                                          decoder_hidden, encoder_outputs)
            # |decoder_output| = (sequence_length, output_lang.n_words)
            # |decoder_hidden| = (2, num_layers*num_directions, batch_size, hidden_size)
            # 2: respectively, hidden state and cell state.
            # Here, the lstm layer in decoder is uni-directional.
            # |decoder_attention| = (sequence_length, max_length)
            
            topv, topi = decoder_output.data.topk(1) # top-1 value, index
            # |topv|, |topi| = (1, 1)

            if topi.item() == loader.EOS_token:
                # decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()
            
    return decoded_words

def evaluateiters(pairs, encoder, decoder, train_pairs_seed=0):
    start = time.time()
    train_pairs, test_pairs = train_test_split(pairs, test_size=0.15, random_state=train_pairs_seed)
    # |test_pairs| = (n_pairs, 2, sentence_length, 1) # eng, fra

    scores = []
    for pi, pair in enumerate(train_pairs):
        # if pi == int(len(pairs)*0.3):
        #     break
    # for i in range(10):
    #     pair = pairs[i*3000]
        output_words = evaluate(pair[0], encoder, decoder)
        output_sentence = ' '.join(output_words)

        # print('From(source):\t{}\n To(answer):\t{}'.format(pair[0], pair[1])) 
        # print('To(predict):\t{}'.format(output_sentence), end='\n\n')
        # print('{}|{}'.format(pair[1], output_sentence))

        # for nltk.bleu
        ref = pair[1].split()
        hyp = output_words
        scores.append(sentence_bleu([ref], hyp) * 100.)
        
    print('BLEU: {:.4}'.format(sum(scores)/len(train_pairs)))
    # print('BLEU: {:.4}'.format(sum(scores)/int(len(pairs)*0.3)))

if __name__ == "__main__":
    '''
    Evaluation is mostly the same as training,
    but there are no targets so we simply feed the decoder's predictions back to itself for each step.
    Every time it predicts a word, we add it to the output string,
    and if it predicts the EOS token we stop there.
    '''
    embedding_size = 300
    hidden_size = 300

    input_lang, output_lang, pairs = loader.prepareData('eng', 'fra', True)
    
    input_emb_matrix, output_emb_matrix= np.load('input_emb_matrix.npy'), np.load('output_emb_matrix.npy')
    print('Embedding-matrix shape: {}, {}'.format(input_emb_matrix.shape, output_emb_matrix.shape))

    encoder = seq2seq.Encoder(input_size = input_lang.n_words,
                              embedding_size = embedding_size,
                              hidden_size = hidden_size,
                              embedding_matrix = input_emb_matrix,
                              n_layers = 2,
                              dropout_p = .1
                              ).to(device)
    decoder = seq2seq.AttnDecoder(output_size = output_lang.n_words,
                                  embedding_size = embedding_size,
                                  hidden_size = hidden_size,
                                  embedding_matrix = output_emb_matrix,
                                  n_layers = 2,
                                  dropout_p =.1
                                  ).to(device)

    encoder.load_state_dict(torch.load('encoder-n_layers2-hidden300.pth'))
    encoder.eval()
    decoder.load_state_dict(torch.load('decoder-n_layers2-hidden300.pth'))
    decoder.eval()

    evaluateiters(pairs, encoder, decoder)