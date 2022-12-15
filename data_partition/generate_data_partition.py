import os
import glob
import numpy as np

filepath = "../data/pt_data_mustang_deployment_milcom_aug"
output_suffix = "aug"

# locations = ["siebel_10-16", "siebel_10-28", "statefarm_10-28", "sand_10-28"]
locations = ["siebel_10-28", "statefarm_10-28", "sand_10-28"]

# siebel_1016_filenames = glob.glob(os.path.join("/home/tianshi/data/VehicleDetection/pt_files/pt_data_mustang_deployment_milcom_aug/siebel_10-16", "siebel_10-16*.pt"))
siebel_1028_filenames = glob.glob(os.path.join(filepath, "siebel_10-28*.pt"))
statefarm_filenames = glob.glob(os.path.join(filepath, "statefarm*.pt"))
sand_filenames = glob.glob(os.path.join(filepath, "sand*.pt"))

# loc_filenames = [siebel_1016_filenames, siebel_1028_filenames, statefarm_filenames, sand_filenames]
loc_filenames = [siebel_1028_filenames, statefarm_filenames, sand_filenames]

for i, location in enumerate(locations):
    print(f"====location: {location}====")
    train_data = []
    val_data = []
    test_data = []

    for fn in loc_filenames[i]:
        train_data.append(fn)
        val_data.append(fn)
        test_data.append(fn)

    print("train_data: ", len(train_data))
    print("val_data: ", len(val_data))
    print("test_data: ", len(test_data))

    if not os.path.exists(f"time_data_partition_{location}_test_{output_suffix}"):
        os.makedirs(f"time_data_partition_{location}_test_{output_suffix}")

    with open(f"time_data_partition_{location}_test_{output_suffix}/train_index.txt", "w") as f:
        for fn in train_data:
            f.write(fn + "\n")
    
    with open(f"time_data_partition_{location}_test_{output_suffix}/val_index.txt", "w") as f:
        for fn in val_data:
            f.write(fn + "\n")
    
    with open(f"time_data_partition_{location}_test_{output_suffix}/test_index.txt", "w") as f:
        for fn in test_data:
            f.write(fn + "\n")
    