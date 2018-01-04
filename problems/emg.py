import numpy as np
from .dataset import Dataset
import os
import pdb
from scipy import signal
import matplotlib.pyplot as plt
from sets import Set

def normalize(X):
  if X.shape[1] == 1:
      X = (X - np.mean(X))/np.std(X)
  else:
      X[:, 0] = (X[:, 0] - np.mean(X[:, 0]))/np.std(X[:, 0])      
      X[:, 1] = (X[:, 1] - np.mean(X[:, 1]))/np.std(X[:, 1])      
  return X

class EMGProblemDataset(Dataset):
    def __init__(self, num_samples, sample_len):
        super(EMGProblemDataset,self).__init__(num_samples, sample_len)
        #pdb.set_trace()
        data = {}
        for file_name in os.listdir("emg_data/"):
            if file_name.startswith('imu_') or file_name.startswith('emg_'):
                with open("emg_data/"+file_name, 'r') as f:
                    array = []
                    for line in f: # read rest of lines
                        array.append([float(x) for x in line.split()])
                data[file_name] = np.asarray(array)
       
        """
        for file_name in os.listdir("emg_data/"):
            if file_name.endswith(".txt") and file_name.startswith('emg_'):
                with open("emg_data/"+file_name, 'r') as f:
                    emg_array = []
                    for line in f: # read rest of lines
                        emg_array.append([float(x) for x in line.split()])
                
            elif file_name.endswith(".txt") and file_name.startswith('imu_'):
                with open("emg_data/"+file_name, 'r') as f:
                    imu_array = []
                    for line in f: # read rest of lines
                        imu_array.append([float(x) for x in line.split()])
        """
        #pdb.set_trace()
        self.X = []
        self.Y = []
        for i in range(1, 4):
            for j in ["A", "B", "C"]:
                tmp = "_" + str(i) + "_" + j + ".txt"
                names = Set(data.keys())
                if "emg"+tmp in names:
                    self.X.append(normalize(data["emg"+tmp]))
                    self.Y.append(normalize(signal.resample(data["imu"+tmp], 64000)))
        self.X = np.reshape(np.array(self.X), [64000*7, 1])
        self.Y = np.reshape(np.array(self.Y), [64000*7, 2])
        #pdb.set_trace()
        #self.X = (self.X - np.mean(self.X[:3000]))/np.std(self.X[:3000])
        #plt.plot(self.X)
        #plt.show()
        #self.Y[:, 0] = (self.Y[:, 0] - np.mean(self.Y[:3000, 0]))/np.std(self.Y[:3000, 0])
        #self.Y[:, 1] = (self.Y[:, 1] - np.mean(self.Y[:3000, 1]))/np.std(self.Y[:3000, 1])


        #plt.plot(self.Y)
        #plt.show()
        #pdb.set_trace()
        #self.Y = signal.resample(self.Y, self.X.shape[0])
        #plt.plot(self.X[:, 0])
        #plt.show()
        #pdb.set_trace()
        self.X = np.reshape(self.X, [-1, 100, 1]) # signal len = 20
        self.Y = np.reshape(self.Y, [-1, 100, 2])
        self.X_train = self.X[:3000, :, :]
        self.Y_train = self.Y[:3000, :]
        self.X_valid = self.X[3000:, :, :]
        self.Y_valid = self.Y[3000:, :]

    def generate(self, num_samples):
        pass 

    def get_batch(self, batch_idx, batch_size):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size

        X_batch = self.X_train[start_idx:end_idx, :, :]
        Y_batch = self.Y_train[start_idx:end_idx, :]

        return X_batch, Y_batch
