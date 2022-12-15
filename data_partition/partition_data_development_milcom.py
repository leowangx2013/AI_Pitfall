from cgi import test
import os
import getpass
import glob

from tqdm import tqdm

DATA_SEPARATION = [0.8, 0.1, 0.1]

PRESERVED_FOLDERS = {
    "motor",
    "mustang0528",
    "walk",
    "walk2",
    "tesla",
    "Polaris0150pm",
    "Polaris0215pm",
    "Polaris0235pm-NoLineOfSight",
    "Warhog1135am",
    "Warhog1149am",
    "Warhog-NoLineOfSight",
    "Silverado0255pm",
    "Silverado0315pm",
}

HUMVEE_VALID_RUN = "run4"
HUMVEE_TEST_RUN = "run5"

UPSAMPLE_RATIO = 15

def extract_user_list(input_path):
    """Extract the user list in the given path.

    Args:
        input_path (_type_): _description_
    """
    user_list = []

    for e in os.listdir(input_path):
        if os.path.isdir(os.path.join(input_path, e)):
            user_list.append(e)

    return user_list


def partition_data(parkland_data_path, siebel_data_path, output_path):
    # for users in training set, only preserve their data samples with complete modalities
    all_samples = glob.glob(os.path.join(parkland_data_path, "*.pt"))
    parkland_data_samples = []
    for sample in all_samples:
        if 'humvee' not in os.path.basename(sample):
            parkland_data_samples.append(sample)

    train_samples = []
    val_samples = []
    test_samples = []
    
    # Parkland data 
    type_to_samples = {}
    for vehicle_type in PRESERVED_FOLDERS:
        type_to_samples[vehicle_type] = []

    print("Extracting Parkland data samples...")
    for sample in tqdm(parkland_data_samples):
        sample_target = os.path.basename(sample).split("_")[0]
        if sample_target not in PRESERVED_FOLDERS:
            continue
            
        file_path = sample
        type_to_samples[sample_target].append(file_path)

    for vehicle_type in type_to_samples:
        type_to_samples[vehicle_type] = sorted(type_to_samples[vehicle_type], key=lambda x: int(x.split("_")[-1].split(".")[0]))
        total_size = len(type_to_samples[vehicle_type])
        if vehicle_type == "mustang0528":
            for _ in range(7):
                train_samples += type_to_samples[vehicle_type][:int(total_size*DATA_SEPARATION[0])]
        else:
            train_samples += type_to_samples[vehicle_type][:int(total_size*DATA_SEPARATION[0])]
        val_samples += type_to_samples[vehicle_type][int(total_size*DATA_SEPARATION[0]):int(total_size*(DATA_SEPARATION[0]+DATA_SEPARATION[1]))]
        test_samples += type_to_samples[vehicle_type][int(total_size*(DATA_SEPARATION[0]+DATA_SEPARATION[1])):]

    # Siebel data
    siebel_data_samples = glob.glob(os.path.join(siebel_data_path, "*.pt"))
    type_to_samples = {}
    for sample_type in ["driving", "engine", "quiet"]:
        type_to_samples[sample_type] = []
    
    for sample in siebel_data_samples:
        sample_type = os.path.basename(sample).split("_")[1]
        type_to_samples[sample_type].append(sample)

    for sample_type in type_to_samples:
        type_to_samples[sample_type] = sorted(type_to_samples[sample_type], key=lambda x: int(x.split("_")[-1].split(".")[0]))
        total_size = len(type_to_samples[sample_type])
        if sample_type == "driving":
            for _ in range(7):
                train_samples += type_to_samples[sample_type][:int(total_size*DATA_SEPARATION[0])]
        else:
            train_samples += type_to_samples[sample_type][:int(total_size*DATA_SEPARATION[0])]

        val_samples += type_to_samples[sample_type][int(total_size*DATA_SEPARATION[0]):int(total_size*(DATA_SEPARATION[0]+DATA_SEPARATION[1]))]
        test_samples += type_to_samples[sample_type][int(total_size*(DATA_SEPARATION[0]+DATA_SEPARATION[1])):]

    # save the index file
    print(
        f"Number of training samples: {len(train_samples)}, \
        number of validation samples: {len(val_samples)}, \
        number of testing samples: {len(test_samples)}."
    )
    
    with open(os.path.join(output_path, "train_index.txt"), "w") as f:
        for sample_file in train_samples:
            f.write(sample_file + "\n")
    with open(os.path.join(output_path, "val_index.txt"), "w") as f:
        for sample_file in val_samples:
            f.write(sample_file + "\n")
    with open(os.path.join(output_path, "test_index.txt"), "w") as f:
        for sample_file in test_samples:
            f.write(sample_file + "\n")


if __name__ == "__main__":
    username = getpass.getuser()
    parkland_data_path = "../data/pt_data_mustang_development_milcom_noaug"
    siebel_data_path = "../data/pt_mustang_siebel_10-16"
    output_path = "./time_data_partition_mustang_development_milcom_noaug"
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # partition the dta
    partition_data(parkland_data_path, siebel_data_path, output_path)
