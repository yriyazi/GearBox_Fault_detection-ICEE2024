import yaml

# Load config file
with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Access hyperparameters
inference_mode      = config['inference_mode']
num_epochs          = config['num_epochs']
seed                = config['seed']
ckpt_save_freq      = config['ckpt_save_freq']

# Access dataset parameters
dataset_path        = config['dataset']['path']
train_split         = config['dataset']['train_split']
validation_split    = config['dataset']['validation_split']
test_split          = config['dataset']['test_split']
classes             = config['dataset']['classes']
batch_size          = config['dataset']['batch_size']

embed_dim= config['Tokenizer']['embed_dim']

# Access model architecture parameters
num_classes                 = config['model']['num_classes']
n_heads                     = config['model']['n_heads']
depth                       = config['model']['depth']
input_image_size            = config['model']['input_image_size']
patch_size                  = config['model']['patch_size']
in_chanel                   = config['model']['in_chanel']
p                           = config['model']['p']
attention_p                 = config['model']['attention_p']



# Access optimizer parameters
optimizer_name      = config['optimizer']['name']
weight_decay        = config['optimizer']['weight_decay']
learning_rate        = config['optimizer']['learning_rate']
# opt_momentum        = config['optimizer']['momentum']
# Access scheduler parameters
scheduler_name  = config['scheduler']['name']
start_factor    = config['scheduler']['start_factor']
end_factor      = config['scheduler']['end_factor']


# print("configuration has been loaded!!! \n successfully")