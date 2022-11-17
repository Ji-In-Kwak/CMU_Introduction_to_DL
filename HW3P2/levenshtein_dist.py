import Levenshtein

def calculate_levenshtein(h, y, lh, ly, decoder, labels, debug = False):

    if debug:
        
        print(f"\n----- IN LEVENSHTEIN -----\n")
        print(h.shape, lh.shape)
        print(lh)
        print(y.shape, ly.shape)

        
    # TODO: look at docs for CTC.decoder and find out what is returned here
    h = h.permute(1, 0, 2)
    beam_results, beam_scores, timesteps, out_lens = decoder.decode(h, seq_lens=lh)
    
    
    batch_size = y.shape[0] # TODO
    distance = 0 # Initialize the distance to be 0 initially

    for i in range(batch_size): 
#         # TODO: Loop through each element in the batch
        decode_y = y[i][:ly[i]].detach().cpu().numpy()
        y_str = [labels[int(j)] for j in decode_y]
        y_str = ''.join(y_str)

        out = beam_results[i]
        decode_h = out[0][:out_lens[i][0]]
        decode_str = [labels[int(j)] for j in decode_h]
        decode_str = ''.join(decode_str)

        
        distance += Levenshtein.distance(y_str, decode_str)       
            

    distance /= batch_size # TODO: Uncomment this, but think about why we are doing this

    return distance