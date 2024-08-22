#%% Plotting the Results
import pandas as pd
import numpy as np
import utils

adress = r'model\n_heads=12_depth=1_learning_rate=0.001_weight_decay=0.03_embed_dim=384_report.csv'
df=pd.read_csv(adress)


train=df.query('mode == "train"').query('batch_index == 171')
test=df.query('mode == "val"').query('batch_index == 37')

Model_name = f'{adress}'

utils.plot.result_plot(Model_name+"loss","loss",
                        np.array(train['loss_batch']),
                        np.array(test['loss_batch']),
                        " ",
                        DPI=400,
                        y_lim = [0,0.8])

utils.plot.result_plot(Model_name+"Accuracy","Accuracy",
                        np.array(train['avg_train_acc_till_current_batch']),
                        np.array(test['avg_val_acc_till_current_batch']),
                        " ",
                        DPI=400,
                        y_lim = [0.6,1.05])