import  nets, utils
import  torch
from    thop                    import profile,clever_format
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

#%%
input = torch.randn(1, utils.in_chanel, utils.input_image_size, utils.input_image_size).to(utils.device)
macs, params = profile(model, inputs=(input, ))
macs, params = clever_format([macs, params], "%.3f")
print(macs, params)