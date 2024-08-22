import utils
utils.set_seed(utils.seed)

import dataloaders
import  nets
import  deeplearning
import  torch
import  torch.nn                as  nn          # basic building block for neural networks
import  matplotlib.pyplot       as  plt         # for plotting
import  numpy                   as  np            
from    tqdm                    import  tqdm
from    sklearn.metrics         import  accuracy_score         
#%%

model = nets.VisionTransformer(
                                    img_size        =   utils.input_image_size,
                                    patch_size      =   utils.patch_size,
                                    in_chans        =   utils.in_chanel,
                                    n_classes       =   utils.num_classes,  
                                    embed_dim       =   utils.embed_dim,
                                    depth           =   utils.depth,
                                    n_heads         =   utils.n_heads,
                                    mlp_ratio       =   4.0,
                                    qkv_bias        =   True,
                                    p               =   utils.p,
                                    attn_p          =   utils.attention_p
                                ).to(utils.device)

optimizer = torch.optim.AdamW(model.parameters(),
                              lr=utils.learning_rate,
                              betas=(0.9, 0.999),
                              eps=1e-08,
                              weight_decay=utils.weight_decay,)
criterion = nn.BCELoss() 

#%%
# immg = next(iter(dataloaders.train_loader))[0][0]
# # Plot tensors in a 4x4 grid
# fig, axes = plt.subplots(4, 4, figsize=(10, 10))

# for i, batch in enumerate(dataloaders.train_loader):
#     for j, tensor in enumerate(batch):
#         if i * 16 + j < 16:
#             # Convert tensor to NumPy array for plotting
#             image_array = tensor.permute(1, 2, 0).cpu().numpy()  # Assuming CHW format, adjust if needed
#             axes[i, j].imshow(image_array)
#             axes[i, j].axis('off')

# plt.show()
#%%
model, optimizer, report = deeplearning.train(
                                                test_ealuate    = False,
                                                train_loader    = dataloaders.train_loader,
                                                val_loader      = dataloaders.val_loader,
                                                tets_loader     = None,
                                                model       = model,
                                                model_name  = f"n_heads={utils.n_heads}_depth={utils.depth}_learning_rate={utils.learning_rate}_weight_decay={utils.weight_decay}_embed_dim={utils.embed_dim}",
                                                
                                                epochs          = utils.num_epochs,#utils.num_epochs,
                                                device          = utils.device,
                                                load_saved_model= False,
                                                ckpt_save_freq  = utils.ckpt_save_freq,
                                                ckpt_save_path  = r'./model/checkpoint',
                                                ckpt_path       = r'./model/',
                                                report_path     = r'./model/',

                                                optimizer       = optimizer,
                                                criterion       = criterion,

                                                lr_schedulerr_setting =60,
                                                lr_schedulerr  = None,
                                                validation_threshold = 0.98
                                                )
