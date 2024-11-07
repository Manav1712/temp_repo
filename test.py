import numpy as np
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from call_api import getData, get_data
from dcrnn import DCRNNModel
from model_helper import train, test, data_to_z_params, sample_z, MAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""# CNP"""

# Reference: https://chrisorm.github.io/NGP.html
def select_data(x_train, y_train, beta_epsilon_all, score_array, selected_mask, num_days, num_simulations, batch_size):

    mask_score_array = score_array * (1 - selected_mask)
    sorted_mask_score_arr = np.argsort(mask_score_array)
    for i in range(batch_size):

        select_index = sorted_mask_score_arr[-1 * (i+1)]
        print('select_index:', select_index)


        selected_x = beta_epsilon_all[select_index:select_index+1]
        selected_y = getData("http://host.docker.internal:8080/", num_days, num_simulations, selected_x[0].tolist())
        #getData("http://host.docker.internal:8000/", num_days, num_simulations, selected_x[0].tolist())

        x_train1 = np.repeat(selected_x, num_simulations, axis=0)
        x_train = np.concatenate([x_train, x_train1], 0)
    
        selected_y = np.array(json.loads(selected_y)["train_set"])
        y_train1 = selected_y.reshape(-1,100)
        y_train = np.concatenate([y_train, y_train1], 0)
 
        selected_mask[select_index] = 1
    
    print("Successfully Chose Data")
    return x_train, y_train, selected_mask

def calculate_score(x_train, y_train, beta_epsilon_all):
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()

    # Query z_mu, z_var of the current training data
    with torch.no_grad():
        z_mu, z_logvar = data_to_z_params(x_train.to(device),y_train.to(device), dcrnn)

        score_list = []
        for i in range(len(beta_epsilon_all)):
            
            # Generate x_search
            x1 = beta_epsilon_all[i:i+1]
            x_search = np.repeat(x1,num_simulations,axis =0)
            x_search = torch.from_numpy(x_search).float()

            # Generate y_search based on z_mu, z_var of current training data
            output_list = []
            for j in range (len(x_search)):
                zsamples = sample_z(z_mu, z_logvar, z_dim) 
                output = dcrnn.decoder(x_search[j:j+1].to(device), zsamples).cpu()
                output_list.append(output.detach().numpy())

            y_search = np.concatenate(output_list)
            y_search = torch.from_numpy(y_search).float()

            x_search_all = torch.cat([x_train,x_search],dim=0)
            y_search_all = torch.cat([y_train,y_search],dim=0)

            # Generate z_mu_search, z_var_search
            z_mu_search, z_logvar_search = data_to_z_params(x_search_all.to(device),y_search_all.to(device), dcrnn)
            
            # Calculate and save kld
            mu_q, var_q, mu_p, var_p = z_mu_search,  0.1+ 0.9*torch.sigmoid(z_logvar_search), z_mu, 0.1+ 0.9*torch.sigmoid(z_logvar)

            std_q = torch.sqrt(var_q)
            std_p = torch.sqrt(var_p)

            p = torch.distributions.Normal(mu_p, std_p)
            q = torch.distributions.Normal(mu_q, std_q)
            score = torch.distributions.kl_divergence(p, q).sum()

            score_list.append(score.item())

        score_array = np.array(score_list)

    return score_array

"""BO search:"""

def MAE_MX(y_pred, y_test):
    N = 100000
    y_pred = y_pred.reshape(30,9, 30, 100)*N/100
    y_test = y_test.reshape(30,9, 30, 100)*N/100
    mae_matrix = np.mean(np.abs(y_pred - y_test),axis=(2,3))
    mae = np.mean(np.abs(y_pred - y_test))
    return mae_matrix, mae

large = 25; med = 19; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': 20,
          'figure.figsize': (27, 8),
          'axes.labelsize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': med}
plt.rcParams.update(params)

# Generate sample space
# ------------------------------------------------------------------------

num_days = 101
num_simulations = 30
beta = np.repeat(np.expand_dims(np.linspace(1.1, 4.0, 30),1),9,1)
epsilon = np.repeat(np.expand_dims(np.linspace(0.25, 0.65, 9),0),30,0)
beta_epsilon = np.stack([beta,epsilon],-1)
beta_epsilon_train = beta_epsilon.reshape(-1,2)

beta = np.repeat(np.expand_dims(np.linspace(1.14, 3.88, 5),1),3,1)
epsilon = np.repeat(np.expand_dims(np.linspace(0.29, 0.59, 3),0),5,0)
beta_epsilon = np.stack([beta,epsilon],-1)
beta_epsilon_val = beta_epsilon.reshape(-1,2)

beta = np.repeat(np.expand_dims(np.linspace(1.24, 3.98, 5),1),3,1)
epsilon = np.repeat(np.expand_dims(np.linspace(0.31, 0.61, 3),0),5,0)
beta_epsilon = np.stack([beta,epsilon],-1)
beta_epsilon_test = beta_epsilon.reshape(-1,2)

# CREATE TRAINING AND TESTING DATA
# ----------------------------------------------------------------------

beta_epsilon_all = beta_epsilon_train
x_all = np.repeat(beta_epsilon_all,num_simulations,axis =0)
y_val = get_data(num_days,num_simulations,beta_epsilon_val)
y_val = y_val.reshape(-1,100)
x_val = np.repeat(beta_epsilon_val,num_simulations,axis =0)
y_test = get_data(num_days,num_simulations,beta_epsilon_test)
y_test = y_test.reshape(-1,100)
x_test = np.repeat(beta_epsilon_test,num_simulations,axis =0)

np.random.seed(3)
mask_init = np.zeros(len(beta_epsilon_all))
mask_init[:2] = 1

np.random.shuffle(mask_init)
selected_beta_epsilon = beta_epsilon_all[mask_init.astype('bool')]
x_train_init = np.repeat(selected_beta_epsilon,num_simulations,axis =0)

selected_y = get_data(num_days,num_simulations,selected_beta_epsilon)

y_train_init = selected_y.reshape(selected_y.shape[0]*selected_y.shape[1],selected_y.shape[2])

# SIMULATION
# ------------------------------------------------------------------------------------
r_dim = 8
z_dim = 8 # 8
x_dim = 2 #
y_dim = 100 # 50
N = 100000 # Population
dcrnn = DCRNNModel(x_dim, y_dim, r_dim, z_dim, device).to(device)
opt = torch.optim.Adam(dcrnn.parameters(), 1e-3)

def simulation():
    ypred_allset = []
    ypred_testset = []
    mae_testset = []
    score_set = []
    mask_set = []
    y_pred_test_list = []
    y_pred_all_list = []
    test_mae_list = []
    score_list = []
    mask_list = []

    x_train,y_train = x_train_init, y_train_init
    selected_mask = np.copy(mask_init)
        
    for i in range(8): #8
        print('training data shape:', x_train.shape, y_train.shape)
        mask_list.append(np.copy(selected_mask))

        train_losses, val_losses, test_losses, z_mu, z_logvar = train(3, x_train, y_train, x_val, y_val, 
                                                                x_test, y_test, opt, dcrnn, device, z_dim, 500, 1500) #20000, 5000
        
        y_pred_test = test(torch.from_numpy(x_train).float(),torch.from_numpy(y_train).float(),
                            torch.from_numpy(x_test).float(), device, dcrnn, z_dim)
        y_pred_test_list.append(y_pred_test)

        test_mae = N * MAE(torch.from_numpy(y_pred_test).float(),torch.from_numpy(y_test).float())/100
        test_mae_list.append(test_mae.item())
        print('Test MAE:',test_mae.item())

        y_pred_all = test(torch.from_numpy(x_train).float(),torch.from_numpy(y_train).float(),
                        torch.from_numpy(x_all).float(), device, dcrnn, z_dim)
        y_pred_all_list.append(y_pred_all)

        score_array = calculate_score(x_train, y_train, beta_epsilon_all)
        score_array = (score_array - np.min(score_array))/(np.max(score_array) - np.min(score_array))
            
        score_list.append(score_array)
        x_train, y_train, selected_mask = select_data(x_train, y_train, beta_epsilon_all, score_array, selected_mask, num_days, num_simulations, 3)

    y_pred_all_arr = np.stack(y_pred_all_list,0)
    y_pred_test_arr = np.stack(y_pred_test_list,0)
    test_mae_arr = np.stack(test_mae_list,0)
    score_arr = np.stack(score_list,0)
    mask_arr = np.stack(mask_list,0)

    ypred_allset.append(y_pred_all_arr)
    ypred_testset.append(y_pred_test_arr)
    mae_testset.append(test_mae_arr)
    score_set.append(score_arr)
    mask_set.append(mask_arr)

    ypred_allarr = np.stack(ypred_allset,0)
    ypred_testarr = np.stack(ypred_testset,0)
    mae_testarr = np.stack(mae_testset,0)
    score_arr = np.stack(score_set,0)
    mask_arr = np.stack(mask_set,0)

    np.save('mae_testarr.npy',mae_testarr)
    np.save('score_arr.npy',score_arr)
    np.save('mask_arr.npy',mask_arr)

    np.save('y_pred_all_arr.npy',ypred_allarr)
    np.save('y_pred_test_arr.npy',ypred_testarr)

    np.save('y_test.npy',y_test)

simulation()