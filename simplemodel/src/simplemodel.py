# from audioop import add
from cProfile import label
import os,random
from tabnanny import verbose

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import numpy as np
# from src.dataset2016 import load_data
from src.dataset_arl import load_data
from statistics import mean
from src.utils import *
import argparse
import xgboost as xgb
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error,precision_score,recall_score,f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split,cross_validate,StratifiedKFold
from itertools import product
import pickle
from scipy import signal
from cgi import test
import os
import torch
import random
import getpass
import pickle as pkl
import numpy as np
import shap
from matplotlib import pyplot
import wandb
from tqdm import tqdm
from sklearn.metrics import multilabel_confusion_matrix


parser = argparse.ArgumentParser()

# dataset config
parser.add_argument("--gpu", type=str, default="0", help="Visible GPU")
parser.add_argument("--mode", type=str, default="train", help="Mode for train or test with pretrained model")
parser.add_argument("--scenario", type=str, default="TEST_E", help="Different Deployment Scenarios i.e. E-F-G")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# ### Set Random Seed
seed = 1994
random.seed(seed)  
np.random.seed(seed) 
tf.random.set_seed(seed)


SAMPLE_LEN = 1024
CROSS_TRAIN = False # use one environment as test set
WANDB_ACTIVE = False
if WANDB_ACTIVE:
    wandb.init(project="IoBT-vehicleclassification", entity="uiuc-dkara")
    # wandb.run.name = "XGBoost"

def convertLabels(Y):
    # takes a list of one hot vectors and converts to integer labels
    Y=np.argmax(Y,axis=1)
    # add 1 for index
    # Y = Y+1
    return Y

def createFeatures(X_acoustic, X_seismic,sample_len=SAMPLE_LEN):
    # takes a single second dataframe and returns basic features
    # return pse with welch method for x
    from added_features import applyAndReturnAllFeatures
    feature_names = []
    ## acoustic
    X = X_acoustic
    sample_len = 16000
    features_acoustic = []
    nperseg= 2000 # fft length up to 500 Hz
    for index in range(len(X)):
        x = X[index] 
        f, Pxx_den = signal.welch(x, sample_len, nperseg=nperseg)
        # take up to 1000 Hz
        len_to_take = (1*len(f)) // 8 # (3*len(f)) // 4
        len_to_take = (3*len(f)) // 4
        
        if WANDB_ACTIVE:
            wandb.log({"len_to_take": len_to_take})
            wandb.log({"nperseg": nperseg})
            wandb.log({"f": f[:len_to_take]})
        pse=Pxx_den[:len_to_take]
        
        additonal_features = applyAndReturnAllFeatures(x)
        additonal_feature_names = [k for k, v in sorted(additonal_features.items())] 
        additonal_features = [v for k, v in sorted(additonal_features.items())] #list(additonal_features.values())
        pse = np.concatenate((pse,additonal_features))

        features_acoustic.append(np.asarray(pse).flatten())
        pass
    # append to feature names
    for i in range(len_to_take):
        feature_names.append("pse_acoustic_f"+str(f[i]))
    for i in range(len(additonal_features)):
        feature_names.append('Acoustic'+additonal_feature_names[i])

    ## seismic
    X = X_seismic
    sample_len = 200
    features_seismic = []
    nperseg= 25 # fft length up to 500 Hz
    for index in range(len(X)):
        x = X[index] 
        f, Pxx_den = signal.welch(x, sample_len, nperseg=nperseg)
        # take up to 100 Hz
        len_to_take = len(f) # (1*len(f)) // 8
        # wandb.log({"len_to_take": len_to_take})
        # wandb.log({"nperseg": nperseg})
        
        pse=Pxx_den[:len_to_take]

        additonal_features = applyAndReturnAllFeatures(x)
        additonal_feature_names = [k for k, v in sorted(additonal_features.items())] 
        additonal_features = [v for k, v in sorted(additonal_features.items())] #list(additonal_features.values())
        pse = np.concatenate((pse,additonal_features))

        features_seismic.append(np.asarray(pse).flatten())

    # merge acoustic and seismic features
    features = []
    for i in range(len(features_acoustic)):
        features.append(np.concatenate((features_acoustic[i],features_seismic[i])))

    # append to feature names
    for i in range(len_to_take):
        feature_names.append("pse_seismic_f"+str(f[i]))
    for i in range(len(additonal_features)):
        feature_names.append('Seismic'+additonal_feature_names[i])

    return np.asarray(features),feature_names
    pass

def train_supervised_basic(X_train_acoustic, X_train_seismic, Y_train, X_val_acoustic, X_val_seismic, Y_val,model_name='model', sample_len=SAMPLE_LEN):
    
    X_train,feature_names = createFeatures(X_train_acoustic,X_train_seismic)
    X_val,feature_names = createFeatures(X_val_acoustic,X_val_seismic)
    # Y_train = convertLabels(Y_train)
    # Y_val = convertLabels(Y_val)


    
    # model = xgb.XGBClassifier(objective='binary:logistic')#,verbosity=3)
    model = xgb.XGBClassifier(objective='binary:logistic',n_estimators=400)#,verbosity=3)
    model.fit(X_train, Y_train,
            eval_set=[(X_train, Y_train), (X_val, Y_val)], 
            early_stopping_rounds=20)

    model.get_booster().feature_names = feature_names
    # Interpret model
    interpretModel(model,X_val,feature_names,name="Val")
    interpretModel(model,X_train,feature_names,name="Train")
    if True:
        #model2= xgb.XGBClassifier(**model.get_params())
        print('Choosing best n_estimators as 50')
        model2= xgb.XGBClassifier(n_estimators=80,objective='binary:logistic')
        full_train =np.concatenate((X_train,X_val))
        full_label = np.concatenate((Y_train,Y_val))
        model2.fit(full_train, full_label)
        pkl.dump(model2, open(model_name+".pkl", "wb"))
        return model2
    else:
        pkl.dump(model, open(model_name+".pkl", "wb"))
        return model
    pass

def eval_supervised_basic(model,X_val_acoustic,X_val_seismic, Y_val, model_name='model', sample_len=SAMPLE_LEN,files=None):
    import time
    print("Evaluating start time: ",time.time())
    if not model:
        #model = pkl.load(open(model_name+".pkl", "rb"))
        pass
    X_test,feature_names = createFeatures(X_val_acoustic,X_val_seismic)
    # y_test = convertLabels(Y_val) +1
    y_test = Y_val
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)
    interpretModel(model,X_test,feature_names,name="Test")
    print("Evaluating end time: ",time.time())

    val_files = os.path.join(files, "val_index.txt")
    files =[]
    with open(val_files, 'r') as f:
        for line in f:
            files.append(line.strip())

    if files and False:
        print(len(files))
        #print files where prediction is wrong
        for i in range(len(y_pred)):
            if y_pred[i] != y_test[i]:
                print(files[i])
    
    print(multilabel_confusion_matrix(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: %.3f' % accuracy)
    
    # find the sequence number of current window length
    precision = precision_score(y_test, y_pred, average='binary')
    print('Precision: %.3f' % precision)
    recall = recall_score(y_test, y_pred, average='binary')
    print('Recall: %.3f' % recall)
    
    f_score = f1_score(y_test,y_pred,average='binary')
    print('F1-Score: %.3f' % f_score)
    
    ## better confusion matrix
    X_val_labeled = X_test
    Y_val_labeled = y_test
    
    con_mat = confusion_matrix(y_test, y_pred)
    con_mat = con_mat / con_mat.sum(axis=1, keepdims=True)
    df_cm = pd.DataFrame(con_mat, range(len(set(y_test))), range(len(set(y_test))))
    plt.figure(figsize=(10,7))
    plt.title(f"Window Size = {1024}, Overall Accuracy = {accuracy}")
    s = sn.heatmap(df_cm, annot=True)
    s.set(xlabel='Prediction', ylabel='True Label')
    plt.savefig(f"./n_win={1024}.png")
    if WANDB_ACTIVE:
        wandb.log({"Confusion Matrix": wandb.Image(f"./n_win={1024}.png")})
        wandb.log({"Accuracy": accuracy})
        wandb.log({"Precision": precision})
        wandb.log({"Recall": recall})
        wandb.log({"F1-Score": f_score})
    """
    print(f"Correctness = {correctness}, incorrectness = {incorrectness}, accuracy = {correctness / (correctness + incorrectness)}")
    wandb.log({"Accuracy": correctness / (correctness + incorrectness),
                "Correctness": correctness,
                "Incorrectness": incorrectness})

    print("Accuracy by runs: \n")
    for n, (cor, incor, fn) in enumerate(zip(correctness_by_runs, incorrectness_by_runs, filenames)):
        print(n, fn, cor, incor, cor/(cor+incor))
        wandb.log({f"Accuracy by runs {fn}": cor/(cor+incor),
                    f"Correctness by runs {fn}": cor,
                    f"Incorrectness by runs {fn}": incor})
    """
    pass

def load_data_sedan(filepath, sample_len=256):

    def loaderHelper(index_filepath):
        train_index = []
    
        with open(index_filepath, "r") as file:
            for line in file:
                # last part of the line directory is the filename
                # append the last two directories to the path
                train_index.append(os.path.join(line.split("/")[-2].strip(),line.split("/")[-1].strip()))
                #train_index.append(line.split("/")[-1].strip())
        # read training data from filepath
        X_train_acoustic = []
        X_train_seismic = []
        Y_train = []
        for file in train_index:
            try:
                #sample = torch.load(os.path.join(filepath, file))
                sample= torch.load(file)
                seismic= torch.flatten(sample['data']['shake']['seismic']).numpy()
                acoustic = torch.flatten(sample['data']['shake']['audio']).numpy()
                
                if True: # do 1vsrest with humvee
                    if "driving" in file:
                        label = np.array(1)
                    elif "engine" in file:
                        label = np.array(1)
                    elif 'mustang' in file:
                        label = np.array(1)
                    
                    #elif 'quiet' in file:
                    #    label = np.array(0)
                    
                    #elif 'humvee' in file:
                    #    label = np.array(1)
                    else:
                        label = np.array(0)
                    pass
                else:
                    label = sample['label'].numpy()
                
                X_train_acoustic.append(acoustic)
                X_train_seismic.append(seismic)
                Y_train.append(label)
            
            except:
                print("Error reading file: ", file)
                continue
        return X_train_acoustic, X_train_seismic, Y_train

    
    # preliminaries
    train_index_file = 'time_data_partition_mustang_development_milcom_noaug/train_index.txt'#"time_data_partition_mustang/train_index.txt"
    val_index_file = 'time_data_partition_mustang_development_milcom_noaug/val_index.txt'#"time_data_partition_mustang/val_index.txt"
    test_index_file = 'time_data_partition_mustang_development_milcom_noaug/test_index.txt'#"time_data_partition_mustang/test_index.txt"
    # sample_rate_acoustic = 8000
    # sample_rate_seismic = 100 

    X_train_acoustic, X_train_seismic, Y_train = loaderHelper(train_index_file)
    X_val_acoustic, X_val_seismic, Y_val = loaderHelper(val_index_file)
    X_test_acoustic, X_test_seismic, Y_test = loaderHelper(test_index_file)

    
    X_train_acoustic = np.array(X_train_acoustic)
    X_train_seismic = np.array(X_train_seismic)
    Y_train = np.array(Y_train)
    X_val_acoustic = np.array(X_val_acoustic)
    X_val_seismic = np.array(X_val_seismic)
    Y_val = np.array(Y_val)
    X_test_acoustic = np.array(X_test_acoustic)
    X_test_seismic = np.array(X_test_seismic)
    Y_test = np.array(Y_test)
    
    '''
    # X_train_shape = (11495, 1024, 5)
    # Y_train.shape = (11495, 9)
    for i in range(len(X_val_acoustic)):
        m = np.max(np.absolute(X_val_acoustic[i]))
        X_val_acoustic[i] = X_val_acoustic[i]/m
    for i in range(len(X_val_seismic)):
        m = np.max(np.absolute(X_val_seismic[i]))
        X_val_seismic[i] = X_val_seismic[i]/m
    
    for i in range(len(X_train_acoustic)):
        m = np.max(np.absolute(X_train_acoustic[i]))
        X_train_acoustic[i] = X_train_acoustic[i]/m
    for i in range(len(X_train_seismic)):
        m = np.max(np.absolute(X_train_seismic[i]))
        X_train_seismic[i] = X_train_seismic[i]/m

    for i in range(len(X_test_acoustic)):
        m = np.max(np.absolute(X_test_acoustic[i]))
        X_test_acoustic[i] = X_test_acoustic[i]/m
    for i in range(len(X_test_seismic)):
        m = np.max(np.absolute(X_test_seismic[i]))
        X_test_seismic[i] = X_test_seismic[i]/m
    
    '''

    print("X_train_acoustic shape: ", X_train_acoustic.shape)
    print("X_train_seismic shape: ", X_train_seismic.shape)
    print("Y_train shape: ", Y_train.shape)
    print("X_val_acoustic shape: ", X_val_acoustic.shape)
    print("X_val_seismic shape: ", X_val_seismic.shape)
    print("Y_val shape: ", Y_val.shape)
    print("X_test_acoustic shape: ", X_test_acoustic.shape)
    print("X_test_seismic shape: ", X_test_seismic.shape)
    print("Y_test shape: ", Y_test.shape)
    return X_train_acoustic, X_train_seismic, Y_train, X_val_acoustic, X_val_seismic, Y_val, X_test_acoustic, X_test_seismic, Y_test



def load_data_all(filepath):

    def loaderHelper(index_filepath):
        train_index = []
    
        with open(index_filepath, "r") as file:
            for line in file:
                # last part of the line directory is the filename
                train_index.append(line.split("/")[-1].strip())
        # read training data from filepath
        X_train_acoustic = []
        X_train_seismic = []
        Y_train = []
        for file in train_index:

            # do selection here
            if "non-humvee" in file or 'no-humvee' in file:
                pass # remove
            else:
                pass

            try:
                sample = torch.load(os.path.join(filepath, file))
                seismic= torch.flatten(sample['data']['shake']['seismic']).numpy()
                acoustic = torch.flatten(sample['data']['shake']['audio']).numpy()
                
                if True: # do 1vsrest with humvee
                    if "non-humvee" in file or 'no-humvee' in file:
                        label = np.array(0)
                    elif "driving" in file:
                        label = np.array(1)
                    else:
                        label = np.array(0)
                    pass
                else:
                    label = sample['label'].numpy()
                
                X_train_acoustic.append(acoustic)
                X_train_seismic.append(seismic)
                Y_train.append(label)
            
            except:
                print("Error reading file: ", file)
                continue
        return X_train_acoustic, X_train_seismic, Y_train

    
    # preliminaries
    #train_index_file = "time_data_partition_positive+negative_t=1000/train_index.txt"
    #val_index_file = "time_data_partition_positive+negative_t=1000/val_index.txt"
    #test_index_file = "time_data_partition_positive+negative_t=1000/test_index.txt"
    
    
    #train_index_file = "time_data_partition_positive+negative_t=700/train_index.txt"
    #val_index_file = "time_data_partition_positive+negative_t=700/val_index.txt"
    #test_index_file = "time_data_partition_positive+negative_t=700/test_index.txt"
    
    #X_train_acoustic, X_train_seismic, Y_train = loaderHelper(train_index_file)
    #X_val_acoustic, X_val_seismic, Y_val = loaderHelper(val_index_file)
    #X_test_acoustic, X_test_seismic, Y_test = loaderHelper(test_index_file)

    
    X_train_acoustic = np.array(X_train_acoustic)
    X_train_seismic = np.array(X_train_seismic)
    Y_train = np.array(Y_train)
    X_val_acoustic = np.array(X_val_acoustic)
    X_val_seismic = np.array(X_val_seismic)
    Y_val = np.array(Y_val)
    X_test_acoustic = np.array(X_test_acoustic)
    X_test_seismic = np.array(X_test_seismic)
    Y_test = np.array(Y_test)
    
    '''
    # X_train_shape = (11495, 1024, 5)
    # Y_train.shape = (11495, 9)
    for i in range(len(X_val_acoustic)):
        m = np.max(np.absolute(X_val_acoustic[i]))
        X_val_acoustic[i] = X_val_acoustic[i]/m
    for i in range(len(X_val_seismic)):
        m = np.max(np.absolute(X_val_seismic[i]))
        X_val_seismic[i] = X_val_seismic[i]/m
    
    for i in range(len(X_train_acoustic)):
        m = np.max(np.absolute(X_train_acoustic[i]))
        X_train_acoustic[i] = X_train_acoustic[i]/m
    for i in range(len(X_train_seismic)):
        m = np.max(np.absolute(X_train_seismic[i]))
        X_train_seismic[i] = X_train_seismic[i]/m

    for i in range(len(X_test_acoustic)):
        m = np.max(np.absolute(X_test_acoustic[i]))
        X_test_acoustic[i] = X_test_acoustic[i]/m
    for i in range(len(X_test_seismic)):
        m = np.max(np.absolute(X_test_seismic[i]))
        X_test_seismic[i] = X_test_seismic[i]/m
    
    '''

    print("X_train_acoustic shape: ", X_train_acoustic.shape)
    print("X_train_seismic shape: ", X_train_seismic.shape)
    print("Y_train shape: ", Y_train.shape)
    print("X_val_acoustic shape: ", X_val_acoustic.shape)
    print("X_val_seismic shape: ", X_val_seismic.shape)
    print("Y_val shape: ", Y_val.shape)
    print("X_test_acoustic shape: ", X_test_acoustic.shape)
    print("X_test_seismic shape: ", X_test_seismic.shape)
    print("Y_test shape: ", Y_test.shape)
    return X_train_acoustic, X_train_seismic, Y_train, X_val_acoustic, X_val_seismic, Y_val, X_test_acoustic, X_test_seismic, Y_test


def load_data_parkinglot(filepath, sample_len=256):

    def loaderHelper(index_filepath):
        train_index = []
    
        with open(index_filepath, "r") as file:
            for line in file:
                if 'txt' in line:
                    continue
                # last part of the line directory is the filename
                train_index.append(line.split("/")[-1].strip())
        # read training data from filepath
        X_train_acoustic = []
        X_train_seismic = []
        Y_train = []
        for file in train_index:
            try:
                sample = torch.load(os.path.join(filepath, file))
                seismic= torch.flatten(sample['data']['shake']['seismic']).numpy()
                acoustic = torch.flatten(sample['data']['shake']['audio']).numpy()
                
                if True: # do 1vsrest with humvee
                    if "driving" in file:
                        label = np.array(1)
                    elif "engine" in file:
                        label = np.array(1)
                    elif 'mustang' in file:
                        label = np.array(1)
                    else:
                        label = np.array(0)
                    pass
                else:
                    label = sample['label'].numpy()
                
                X_train_acoustic.append(acoustic)
                X_train_seismic.append(seismic)
                Y_train.append(label)
            
            except:
                print("Error reading file: ", file)
                continue
        return X_train_acoustic, X_train_seismic, Y_train

    
    
    def train_test_val_split(filelist):
        random.seed(42)
        data= random.shuffle(filelist)

        val_set = 'sand_'
        val_set = 'statefarm_'
        val_set = 'siebel_'
        # val_set = None
        if not CROSS_TRAIN:
            train = filelist[:int(len(filelist)*0.7)]
            test = filelist[int(len(filelist)*0.7):int(len(filelist)*0.8)]
            val = filelist[int(len(filelist)*0.8):]
        else:
            print("Current val_set: ", val_set)
            train = []
            val = []
            test = []
            for file in filelist:
                if val_set in file:
                    val.append(file)
                else:
                    train.append(file)
            pass
        return train, val, test

    def createIndexes(filepath):
        # from the files in the directory, create a list of files to read as train-test-val sets
        # create a list of all files in the directory
        files = os.listdir(filepath)
        # list of files including 'quiet'
        files_quiet = []
        # files including 'driving'
        files_driving = []
        # files including 'engine'
        files_engine = []
        for file in files:
            #if not filepath == 'siebel_10-16':
            #if 'pt_data_mustang_10-28' in filepath:
            if 'pt_data_mustang_testing_milcom_aug' in filepath:
                if args.scenario == 'TEST_G':
                    val_set = 'sand_'
                elif args.scenario == 'TEST_F':
                    val_set = 'statefarm_'
                elif args.scenario == 'TEST_E':
                    val_set = 'siebel_'
                
                if not val_set in file:
                    continue
                
            if "quiet" in file:
                files_quiet.append(file)
            elif "driving" in file:
                files_driving.append(file)
            elif "engine" in file:
                files_engine.append(file)
            elif 'txt' in file:
                continue
            elif '.pt' in file: # added this for use with no index file
                files_driving.append(file)
            else:
                print("Error: file not in quiet, driving, engine: ", file)

        training_set = []
        test_set = []
        val_set = []
        driving_train, driving_val, driving_test = train_test_val_split(files_driving)
        quiet_train, quiet_val, quiet_test = train_test_val_split(files_quiet)
        engine_train, engine_val, engine_test = train_test_val_split(files_engine)
        training_set.extend(driving_train)
        training_set.extend(quiet_train)
        training_set.extend(engine_train)
        val_set.extend(driving_val)
        val_set.extend(quiet_val)
        val_set.extend(engine_val)
        test_set.extend(driving_test)
        test_set.extend(quiet_test)
        test_set.extend(engine_test)
        
        # write the sets to files
        with open(os.path.join(filepath, "train_index.txt"), "w") as file:
            for line in training_set:
                file.write(line + "\n")
        with open(os.path.join(filepath, "test_index.txt"), "w") as file:
            for line in test_set:
                file.write(line + "\n")
        with open(os.path.join(filepath, "val_index.txt"), "w") as file:
            for line in val_set:
                file.write(line + "\n")

        if WANDB_ACTIVE:
            # save txt files to wandb
            wandb.save(os.path.join(filepath, "train_index.txt"))
            wandb.save(os.path.join(filepath, "test_index.txt"))
            wandb.save(os.path.join(filepath, "val_index.txt"))

            
        # return filepaths
        return os.path.join(filepath, "train_index.txt"), os.path.join(filepath, "test_index.txt"), os.path.join(filepath, "val_index.txt")
        #return training_set, test_set, val_set


    # create indexes if they don't exist
    #if not os.path.exists(os.path.join(filepath, "train_index.txt")):
    train_index_file, test_index_file, val_index_file = createIndexes(filepath)

    X_train_acoustic, X_train_seismic, Y_train = loaderHelper(train_index_file)
    X_val_acoustic, X_val_seismic, Y_val = loaderHelper(val_index_file)
    X_test_acoustic, X_test_seismic, Y_test = loaderHelper(test_index_file)

    
    X_train_acoustic = np.array(X_train_acoustic)
    X_train_seismic = np.array(X_train_seismic)
    Y_train = np.array(Y_train)
    X_val_acoustic = np.array(X_val_acoustic)
    X_val_seismic = np.array(X_val_seismic)
    Y_val = np.array(Y_val)
    X_test_acoustic = np.array(X_test_acoustic)
    X_test_seismic = np.array(X_test_seismic)
    Y_test = np.array(Y_test)
    
    

    print("X_train_acoustic shape: ", X_train_acoustic.shape)
    print("X_train_seismic shape: ", X_train_seismic.shape)
    print("Y_train shape: ", Y_train.shape)
    print("X_val_acoustic shape: ", X_val_acoustic.shape)
    print("X_val_seismic shape: ", X_val_seismic.shape)
    print("Y_val shape: ", Y_val.shape)
    print("X_test_acoustic shape: ", X_test_acoustic.shape)
    print("X_test_seismic shape: ", X_test_seismic.shape)
    print("Y_test shape: ", Y_test.shape)
    return X_train_acoustic, X_train_seismic, Y_train, X_val_acoustic, X_val_seismic, Y_val, X_test_acoustic, X_test_seismic, Y_test


def load_shake_data(filepath, sample_len=256):

    def loaderHelper(filepath):
        
        # read all file names from filepath ending with .pt
        files = []
        for file in os.listdir(filepath):
            if file.endswith(".pt"):
                files.append(file)
        
        # read training data from filepath
        X_train_acoustic = []
        X_train_seismic = []
        Y_train = []
        for file in files:
            try:
                sample = torch.load(os.path.join(filepath, file))
                seismic= torch.flatten(sample['data']['shake']['seismic']).numpy()
                acoustic = torch.flatten(sample['data']['shake']['audio']).numpy()
                
                if False: # do 1vsrest with humvee
                    if "humv" in file:
                        label = np.array(1)
                    else:
                        label = np.array(0)
                    pass
                else:
                    label = np.array(0) # 0 or 1 
                
                X_train_acoustic.append(acoustic)
                X_train_seismic.append(seismic)
                Y_train.append(label)
            
            except:
                print("Error reading file: ", file)
                continue
        return X_train_acoustic, X_train_seismic, Y_train, files

    
    # preliminaries
    # sample_rate_acoustic = 8000
    # sample_rate_seismic = 100 

    X_train_acoustic, X_train_seismic, Y_train, files = loaderHelper(filepath)
    
    
    X_train_acoustic = np.array(X_train_acoustic)
    X_train_seismic = np.array(X_train_seismic)
    Y_train = np.array(Y_train)
    
    
    '''
    for i in range(len(X_train_acoustic)):
        m = np.max(np.absolute(X_train_acoustic[i]))
        X_train_acoustic[i] = X_train_acoustic[i]/m
    for i in range(len(X_train_seismic)):
        m = np.max(np.absolute(X_train_seismic[i]))
        X_train_seismic[i] = X_train_seismic[i]/m
    '''
    
    print("X_train_acoustic shape: ", X_train_acoustic.shape)
    print("X_train_seismic shape: ", X_train_seismic.shape)
    print("Y_train shape: ", Y_train.shape)
    return X_train_acoustic, X_train_seismic, Y_train, files

def interpretModel(model,X_test,feature_names,name=None):
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test,feature_names= feature_names,show=False)
    if name:
        figname= name + "Shapley_summary_plot.png"
    else:
        figname= "Shapley_summary_plot.png"
    pyplot.savefig(figname,bbox_inches = "tight")
    if WANDB_ACTIVE:
        wandb.log({"Interpretation": wandb.Image(figname)})
        
    pyplot.close()

def countNumParams(file_name):
    import json
    with open(file_name, 'r', encoding='utf-8') as f:
        my_data = json.load(f)
    trees = my_data['learner']['gradient_booster']['model']['trees']

    param_size = 0
    for tree in trees:
        for entry in tree:
            if entry == 'loss_changes':
                continue
            try:
                current_size = len(tree[entry])
                param_size += current_size
            except TypeError:
                param_size += 0
            pass

#49700
#49600 without id
# 44680 without id and without loss changes
# 
    pass
if __name__ == "__main__":

    if args.mode == 'test':# mode=='3':
        
        # Figure 4
        figure_4 = True
        filepath = 'pt_data_mustang_10-28'
        filepath2 = 'pt_data_mustang_testing_milcom_aug'
        
        #X_train_acoustic, X_train_seismic, Y_train, X_val_acoustic, X_val_seismic, Y_val, X_test_acoustic, X_test_seismic, Y_test = load_data_parkinglot(filepath)
        
        if figure_4:
            X_train_acoustic2, X_train_seismic2, Y_train2, X_val_acoustic2, X_val_seismic2, Y_val2, X_test_acoustic2, X_test_seismic2, Y_test2 = load_data_parkinglot(filepath2)

            X_val_acoustic = np.concatenate((X_train_acoustic2, X_test_acoustic2,X_val_acoustic2), axis=0)
            X_val_seismic = np.concatenate((X_train_seismic2, X_test_seismic2,X_val_seismic2), axis=0)
            Y_val = np.concatenate((Y_train2, Y_test2,Y_val2), axis=0)
        

        print("X_train_acoustic shape: ", X_val_acoustic.shape)
        print("X_train_seismic shape: ", X_val_seismic.shape)
        print("Y_train shape: ", Y_val.shape)

        model_name = filepath
        
        sup_model = pkl.load(open('simple_model'+".pkl", "rb"))        
        eval_supervised_basic(sup_model,X_val_acoustic,X_val_seismic, Y_val,model_name,files = filepath)

    elif args.mode == 'train': #mode=='4':

        # Figure 4
        figure_4 = False
        
        filepath = 'pt_data_mustang_10-28'
        filepath2 = 'pt_data'
        
        X_train_acoustic, X_train_seismic, Y_train, X_val_acoustic, X_val_seismic, Y_val, X_test_acoustic, X_test_seismic, Y_test = load_data_sedan(filepath)
        
        
        if not CROSS_TRAIN and not figure_4:
            # concatenate train and test data
            X_train_acoustic = np.concatenate((X_train_acoustic, X_test_acoustic), axis=0)
            X_train_seismic = np.concatenate((X_train_seismic, X_test_seismic), axis=0)
            Y_train = np.concatenate((Y_train, Y_test), axis=0)

        print("X_train_acoustic shape: ", X_train_acoustic.shape)
        print("X_train_seismic shape: ", X_train_seismic.shape)
        print("Y_train shape: ", Y_train.shape)

        model_name = 'simple_model'# filepath
        
        sup_model = train_supervised_basic(X_train_acoustic,X_train_seismic, Y_train, X_val_acoustic,X_val_seismic,Y_val,model_name)
        eval_supervised_basic(sup_model,X_val_acoustic,X_val_seismic, Y_val,model_name,files = filepath)
        