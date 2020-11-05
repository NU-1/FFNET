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

# define neural network layout
l1 = Layer(4096,400,'relu')
l2 = Layer(400,200,'relu')
l3 = Layer(200,100,'relu')
l4 = Layer(100,25,'linear')
layers = [l1,l2,l3,l4]
learning_rate = 0.0002
loss_type = 'mean_square'
opt_type = 'RMSprop'

print('Loading Graph')

Q = NeuralNet(layers,learning_rate,loss_type, opt_type)
Q.recover('model/','Q_net_all_11_0_1000')

Q.show()
for layer in Q.layers:
	print(layer)


# -------------Freezing Graph, Converting to TFLite---------------------------


#initialize variables 
init = tf.initialize_all_variables()
Q.sess.run(init)

# Freeze the graph: need to confirm output nodes 
frozen_graph_def = tf.graph_util.convert_variables_to_constants(
    Q.sess,
    Q.sess.graph_def, 
    ['add_3'])

#['Variable_' + str(i) for i in range(1,8)]

# Save the frozen graph
with open('output_graph.pb', 'wb') as f:
  f.write(frozen_graph_def.SerializeToString())

converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file='./output_graph.pb', 
    input_arrays= ['Placeholder'], 
    output_arrays=  ['add_3']
)

tflite_model = converter.convert()

tflite_model_size = open('model.tflite', 'wb').write(tflite_model)
print('TFLite Model is %d bytes' % tflite_model_size)
