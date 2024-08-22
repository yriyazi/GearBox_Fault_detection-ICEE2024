
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

criterion = nn.BCELoss() 

model .load_state_dict(torch.load(r'.\model\n_heads=12_depth=1_learning_rate=0.001_weight_decay=0.03_embed_dim=192_valid_acc 0.9906531531531532.pt'))
model.eval()

