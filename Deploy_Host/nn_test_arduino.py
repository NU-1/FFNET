import tensorflow as tf
import numpy as np
from nn_layer import Layer
from nn import NeuralNet
from read_data import Episode
from agent import Agent
import scipy.io as sio

# data path and ground truth path.
feat_path = 'input/'
gt_path ='input/'

# names of test videos
test_name = ['MP7']
test_num = 1

for i in range(test_num):

	video = Episode(i,test_num, test_name, feat_path, gt_path)
	frame_num = np.shape(video.feat)[0]

	summary = np.zeros(frame_num)
	Q_value = []
	id_curr = 0
	while id_curr < frame_num :
		action_value = Q.forward([video.feat[id_curr]])
		a_index = np.argmax(action_value[0])
		id_next = id_curr + a_index+1
		if id_next >frame_num-1 :
			break
		summary[id_next]=1
		Q_value.append(max(action_value[0]))
		id_curr = id_next

	name = 'output/arduino_sum_'+test_name[i%test_num]
	sio.savemat(name,{'arduino_summary': summary})
print('Arduino Test done.')
