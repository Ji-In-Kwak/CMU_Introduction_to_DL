from tqdm import tqdm
from levenshtein_dist import calculate_levenshtein
import torch


def train_step(train_loader, model, optimizer, criterion, scheduler, scaler, device):
    
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train') 
    train_loss = 0
    model.train()

    for i, data in enumerate(train_loader):

        # TODO: Fill this with the help of your sanity check
        optimizer.zero_grad()

        x, y, lx, ly = data
        x = x.to(device)
        y = y.to(device)
        # lx = lx.detach().cpu()
        
        out, out_lengths = model(x, lx)
        out = out.permute(1, 0, 2)
        
        loss = criterion(out, y, out_lengths, ly)
#         loss.backward()
        
        train_loss += float(loss.item())

        # HINT: Are you using mixed precision? 

        batch_bar.set_postfix(
            loss = f"{train_loss/ (i+1):.4f}",
            lr = f"{optimizer.param_groups[0]['lr']}"
        )

        scaler.scale(loss).backward() # This is a replacement for loss.backward()
        scaler.step(optimizer) # This is a replacement for optimizer.step()
        scaler.update() 
        
        batch_bar.update()
        
    
    batch_bar.close()
    train_loss /= len(train_loader) # TODO

    return train_loss # And anything else you may wish to get out of this function


def evaluate(data_loader, model, criterion, decoder, device, LABELS):
    
    dist = 0
#     loss = 0
    total_loss = 0
    batch_bar = tqdm(total=len(data_loader), dynamic_ncols=True, leave=False, position=0, desc='Val') 
    # TODO Fill this function out, if you're using it.
    model.eval()
    
    for i, data in enumerate(data_loader):
        
        x, y, lx, ly = data
        x = x.to(device)
        y = y.to(device)
        
        # Get model outputs
        with torch.inference_mode():
            out, out_lengths = model(x, lx)
            out = out.permute(1, 0, 2)
            loss = criterion(out, y, out_lengths, ly)
            total_loss += float(loss.item())
            dist = calculate_levenshtein(out, y, out_lengths, ly, decoder, LABELS, debug = False)

        batch_bar.set_postfix(
            loss="{:.04f}".format(float(loss)),
            dist="{:.04f}".format(dist))

        batch_bar.update()
        
    batch_bar.close()
    
    total_loss /= len(data_loader)
    
    return total_loss, dist


def make_output(h, lh, decoder, LABELS, BATCH_SIZE):
  
    beam_results, beam_scores, timesteps, out_seq_len = decoder.decode(h, seq_lens = lh) #TODO: What parameters would the decode function take in?
    batch_size = BATCH_SIZE #What is the batch size

    dist = 0
    preds = []
    for i in range(batch_size): # Loop through each element in the batch
        try:
            h_sliced = beam_results[i][0][:out_seq_len[i][0]] #TODO: Obtain the beam results
            h_string = [LABELS[int(i)] for i in h_sliced] #TODO: Convert the beam results to phonemes
            h_string = ''.join(h_string)
            
        except:
            continue
            h_string = " "
        preds.append(h_string)
    
    return preds