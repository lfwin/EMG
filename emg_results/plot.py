import matplotlib.pyplot as plt

def deserialize(path):
	loss = [float(l) for l in open(path, 'r').read().split('\n')[:-1]]
	return loss

def plot():
	emg_lstm = deserialize('emg_lstm-1')


	#plt.axis([0, 6000, 0, 2])

	plt.plot(emg_lstm, 'r-')
	plt.show()


plot()
