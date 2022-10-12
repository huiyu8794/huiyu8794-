from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from models.ViT import *
import torch
from matplotlib import markers

gpu_id = 'cuda:0'
data_name = 'replay'
savefig_path = '/home/huiyu8794/Generalization/tsne_ls_dm/test_I/'

datas_live = np.load('/shared/alwayswithme/data/domain-generalization/' + data_name + '_images_live.npy')#[:400]
datas_spoof = np.load('/shared/alwayswithme/data/domain-generalization/' + data_name + '_images_spoof.npy')
datas_spoof = datas_spoof[:len(datas_live)]

trainset_D = torch.utils.data.TensorDataset(torch.tensor(np.transpose(datas_live, (0, 3, 1, 2))),
                                            torch.tensor(np.transpose(datas_spoof, (0, 3, 1, 2))),)
                                            # torch.tensor(np.transpose(datas_live2spoof, (0, 3, 1, 2))), )
trainloader_D = torch.utils.data.DataLoader(trainset_D, batch_size=1, shuffle=True)

Fas_Net = vit_base_patch16_224(pretrained=True).to(gpu_id)


### tsne
# protocol = 'results_R_O_M_fixed/ResNet_LS_Domain_invariant/LS_92_loss_0.00000_nloss_0.00075.tar'
# protocol = 'results_R_C_O_fixed/ResNet_LS_Domain_invariant/LS_99_loss_0.00000_nloss_0.00000.tar'
# protocol = 'results_R_C_M_fixed/ResNet_LS_Domain_invariant/LS_99_loss_0.00000_nloss_0.00000.tar'
protocol = 'results_O_C_M_fixed/ResNet_LS_Domain_invariant/LS_99_loss_0.00000_nloss_0.00000.tar'
LS_path = '/shared/alwayswithme/generalization/' + protocol

### tsne_ls_dm
# ID_model = ResNet_ID(output=3).to(gpu_id)
# protocol = 'R_C_M/ID_99_loss_0.00006.tar'
protocol = 'O_C_M/ID_47_loss_0.00000.tar'
# protocol = 'O_C_R/ID_99_loss_0.00000.tar'
# protocol = 'O_M_R/ID_99_loss_0.00000.tar'
ID_path = '/shared/huiyu8794/disentangle_results/ResNet_Domain/'+ protocol
ID_model.load_state_dict(torch.load(ID_path,map_location=gpu_id))
ID_model.eval()


results_filename = 'test_M_cdsa_mcdsa_patch16'
results_path = '/shared/huiyu8794/Transformer/' + results_filename

### tsne_simple
# protocol = 'test_I/ID_99_loss_0.00000.tar'
# LS_path = '/shared/huiyu8794/classifier_result/'+ protocol

### tsne_simple_ls_dm
# protocol = 'test_O/ID_99_loss_0.06159.tar'
# protocol = 'test_I/ID_99_loss_0.20507.tar'
# protocol = 'test_M/ID_99_loss_0.19128.tar'
# protocol = 'test_C/ID_99_loss_0.19965.tar'
# LS_path = '/shared/huiyu8794/share_layer_result/'+ protocol

LS_model.load_state_dict(torch.load(LS_path,map_location=gpu_id))
LS_model.eval()

for tsne_num in range(200):
    print(tsne_num)
    data_list = []
    label_list = []
    for i, data in enumerate(trainloader_D, 0):
        lives, spoofs = data
        lives = lives.to(gpu_id)
        spoofs = spoofs.to(gpu_id)
        feat_l = LS_model(lives)
        feat_s = LS_model(spoofs)
        # print(feat_l.size()) # torch.Size([2, 1000])
        
        feat_idl = ID_model(lives)
        feat_ids = ID_model(spoofs)
        # print(feat_d2.size()) # torch.Size([2, 3])
        
        data_list.append(torch.flatten(feat_l[0]).detach().cpu().numpy())
        label_list.append(1)
        data_list.append(torch.flatten(feat_s[0]).detach().cpu().numpy())
        label_list.append(2)
        
        data_list.append(torch.flatten(feat_idl[0]).detach().cpu().numpy())
        label_list.append(3)
        data_list.append(torch.flatten(feat_ids[0]).detach().cpu().numpy())
        label_list.append(3)
        
        # data_list.append(torch.flatten(feat_d1[1]).detach().cpu().numpy())
        # label_list.append(4)
        # data_list.append(torch.flatten(feat_d2[1]).detach().cpu().numpy())
        # label_list.append(4)
        # data_list.append(torch.flatten(feat_d1[2]).detach().cpu().numpy())
        # label_list.append(5)
        # data_list.append(torch.flatten(feat_d2[2]).detach().cpu().numpy())
        # label_list.append(5)
        # data_list.append(torch.flatten(feat_d3[0]).detach().cpu().numpy())
        # label_list.append(3)
        # if len(data_list)==100:
        #     break
    
    X_tsne = manifold.TSNE().fit_transform(data_list)
    # X_tsne = manifold.TSNE(perplexity=15, n_iter=250, n_iter_without_progress=30).fit_transform(data_list)
    # print(X_tsne)
    # #Data Visualization
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  #Normalize
    plt.figure(figsize=(6, 6))
    for i in range(X_tsne.shape[0]):
        # plt.text(X_norm[i, 0], X_norm[i, 1], str(label_list[i]), color=plt.cm.Set1(label_list[i]),)
        if label_list[i]== 1:
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color="g", s=3)
        elif label_list[i]== 2:
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color="r", s=3)
        # elif label_list[i]== 3:
        #     plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color="r", s=3, marker=".")
        # elif label_list[i]== 4:
        #     plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color="r", s=1, marker="x")
        else:
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color="m", s=3)
    plt.xticks([])
    plt.yticks([])
    plt.axis("off")
    plt.savefig(savefig_path + str(tsne_num) + ".png", bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # conda run -n base --no-capture-output --live-stream python /home/huiyu8794/Generalization/tsne_live_spoof.py