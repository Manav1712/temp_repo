import torch
import numpy as np

def MAE(pred, target):
    loss = torch.abs(pred-target)
    return loss.mean()

def random_split_context_target(x,y, n_context):
    """Helper function to split randomly into context and target"""
    ind = np.arange(x.shape[0])
    mask = np.random.choice(ind, size=n_context, replace=False)
    return x[mask], y[mask], np.delete(x, mask, axis=0), np.delete(y, mask, axis=0)

def sample_z(mu, logvar, z_dim, n=1):
    """Reparameterisation trick."""
    if n == 1:
        eps = torch.autograd.Variable(logvar.data.new(z_dim).normal_())
    else:
        eps = torch.autograd.Variable(logvar.data.new(n,z_dim).normal_())
    
    std = 0.1+ 0.9*torch.sigmoid(logvar)
    return mu + std * eps

def data_to_z_params(x, y, dcrnn):
    """Helper to batch together some steps of the process."""
    xy = torch.cat([x,y], dim=1)
    rs = dcrnn.repr_encoder(xy)
    r_agg = rs.mean(dim=0) # Average over samples
    return dcrnn.z_encoder(r_agg) # Get mean and variance for q(z|...)

def test(x_train, y_train, x_test, device, dcrnn, zdim):
    with torch.no_grad():
      z_mu, z_logvar = data_to_z_params(x_train.to(device),y_train.to(device), dcrnn)
      
      output_list = []
      for i in range (len(x_test)):
          zsamples = sample_z(z_mu, z_logvar, zdim) 
          output = dcrnn.decoder(x_test[i:i+1].to(device), zsamples).cpu()
          output_list.append(output.detach().numpy())
    
    return np.concatenate(output_list)

def train(n_epochs, x_train, y_train, x_val, y_val, x_test, y_test, opt, dcrnn, device, zdim, n_display=500, patience = 5000): #7000, 1000
    train_losses = []
    val_losses = []
    test_losses = []

    means_test = []
    stds_test = []
    N = 100000 # Population
    min_loss = 0. # Early stopping
    wait = 0
    min_loss = float('inf')
    
    for t in range(n_epochs): 
        opt.zero_grad()

        # Generate data and process
        x_context, y_context, x_target, y_target = random_split_context_target(
                                x_train, y_train, int(len(y_train)*0.1)) # 0.25, 0.5, 0.05, 0.015, 0.01   

        x_c = torch.from_numpy(x_context).float().to(device)
        x_t = torch.from_numpy(x_target).float().to(device)
        y_c = torch.from_numpy(y_context).float().to(device)
        y_t = torch.from_numpy(y_target).float().to(device)

        x_ct = torch.cat([x_c, x_t], dim=0).float().to(device)
        y_ct = torch.cat([y_c, y_t], dim=0).float().to(device)

        y_pred = dcrnn(x_t, x_c, y_c, x_ct, y_ct)

        train_loss = N * MAE(y_pred, y_t)/100 + dcrnn.KLD_gaussian()
        mae_loss = N * MAE(y_pred, y_t)/100
        kld_loss = dcrnn.KLD_gaussian()
        
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(dcrnn.parameters(), 5) #10
        opt.step()
        
        # Val loss
        y_val_pred = test(torch.from_numpy(x_train).float(),torch.from_numpy(y_train).float(),
                      torch.from_numpy(x_val).float(), device, dcrnn, zdim)
        val_loss = N * MAE(torch.from_numpy(y_val_pred).float(),torch.from_numpy(y_val).float())/100
        
        # Test loss
        y_test_pred = test(torch.from_numpy(x_train).float(),torch.from_numpy(y_train).float(),
                      torch.from_numpy(x_test).float(), device, dcrnn, zdim)
        test_loss = N * MAE(torch.from_numpy(y_test_pred).float(),torch.from_numpy(y_test).float())/100

        if t % n_display ==0:
            print('train loss:', train_loss.item(), 'mae:', mae_loss.item(), 'kld:', kld_loss.item())
            print('val loss:', val_loss.item(), 'test loss:', test_loss.item())

        if t % (n_display/10) ==0:
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())
            test_losses.append(test_loss.item())

        # Early stopping
        if val_loss < min_loss:
            wait = 0
            min_loss = val_loss
            
        elif val_loss >= min_loss:
            wait += 1
            if wait == patience:
                print('Early stopping at epoch: %d' % t)
                return train_losses, val_losses, test_losses, dcrnn.z_mu_all, dcrnn.z_logvar_all
        
    return train_losses, val_losses, test_losses, dcrnn.z_mu_all, dcrnn.z_logvar_all