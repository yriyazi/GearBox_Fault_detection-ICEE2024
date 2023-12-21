import os
import csv
import numpy as np
import pandas as pd
from scipy.io import savemat



def read_csv_files(folder_path):
    csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', newline='') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    data = list(csv_reader)
                    csv_files.append((file_path, data))
    return csv_files
    
def array_iefier(csv_files):
    _set = {}
    for i in range(10):
        name = str(csv_files[i][0].split('\\')[-1].split('.')[0])
        _set[name] = None

        float_list = [] 
        for _row in csv_files[i][1][1:]:
            float_list .append([float(element) for element in _row])

        _set[name] = np.array(float_list)
    return _set

def _unifier(_set,_min):
    for i in _set.keys():
        _set[i] = _set[i][:_min,:]
    return _set


csv_files_Healthy = read_csv_files(r'Gear Orginal\Healthy')

sensor = {}
for sensor_index in range(4):
    float_list = [] 
    for i in range(10):
        float_list .append([float(element[sensor_index]) for element in csv_files_Healthy[i][1][1:]][:88320])
    sensor[f"sensor {sensor_index}"] = np.array(float_list).reshape(1,-1)


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cwt, morlet
from PIL import Image
import numpy as np
import tqdm
import os

# Specify the directory path
output_directory = r'C:\temp'
# Ensure the directory exists
os.makedirs(output_directory, exist_ok=True)



_image_dim = 224
widths = np.arange(1, 57)
for i in tqdm.tqdm(range(883200//_image_dim)):

    signal1 = sensor[f"sensor 0"][0][i*_image_dim:(i+1)*_image_dim]
    signal2 = sensor[f"sensor 1"][0][i*_image_dim:(i+1)*_image_dim]
    signal3 = sensor[f"sensor 2"][0][i*_image_dim:(i+1)*_image_dim]
    signal4 = sensor[f"sensor 3"][0][i*_image_dim:(i+1)*_image_dim]

    cwt_result1 = cwt(signal1, morlet, widths)
    cwt_result2 = cwt(signal2, morlet, widths)
    cwt_result3 = cwt(signal3, morlet, widths)
    cwt_result4 = cwt(signal4, morlet, widths)

    result_array = np.concatenate((np.abs(cwt_result1),
                                   np.abs(cwt_result2),
                                   np.abs(cwt_result3),
                                   np.abs(cwt_result4)), axis=0)

    
    plt.imshow(np.abs(result_array), aspect='auto', cmap='jet', interpolation='bilinear')
    plt.axis('off')

    # Save the plot as an image file without margins
    output_file_path = os.path.join(output_directory, f'{i:04}.png')

    # Save the plot as an image file (e.g., PNG)
    plt.savefig(output_file_path, pad_inches=0)
    plt.close()

    
    # plt.imshow(np.abs(cwt_result1), aspect='auto', cmap='jet', interpolation='bilinear')
    # plt.axis('off')
    # output_file_path = os.path.join(output_directory, f'1.png')
    # plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0)
    # plt.close()

    # plt.imshow(np.abs(cwt_result2), aspect='auto', cmap='jet', interpolation='bilinear')
    # plt.axis('off')
    # output_file_path = os.path.join(output_directory, f'2.png')
    # plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0)
    # plt.close()

    # plt.imshow(np.abs(cwt_result3), aspect='auto', cmap='jet', interpolation='bilinear')
    # plt.axis('off')
    # output_file_path = os.path.join(output_directory, f'3.png')
    # plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0)
    # plt.close()

    # plt.imshow(np.abs(cwt_result4), aspect='auto', cmap='jet', interpolation='bilinear')
    # plt.axis('off')
    # output_file_path = os.path.join(output_directory, f'4.png')
    # plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0)
    # plt.close()
    # Show the plot (optional)
    # plt.show()
    break
