import tensorflow as tf
import numpy as np
from read_data import Episode
from agent import Agent
import scipy.io as sio

# data path and ground truth path.
feat_path = 'input/'
gt_path = 'input/'

# names of test videos
test_name = ['MP7']
test_num = 1

# Load and Prepare TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


for i in range(test_num):
    video = Episode(i, test_num, test_name, feat_path, gt_path)
    frame_num = np.shape(video.feat)[0]

    summary = np.zeros(frame_num)
    Q_value = []
    id_curr = 0
    while id_curr < frame_num:
        #Load data into TFLite model
        input_data = video.feat[id_curr]

        # Random Test Data Works
        # input_shape = input_details[0]['shape']
        # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

        interpreter.set_tensor(input_details[0]['index'], input_data)

        #Run TFLite Model
        interpreter.invoke()

        #Get TFLite Model Output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        action_value = output_data

        a_index = np.argmax(action_value[0])
        id_next = id_curr + a_index + 1
        if id_next > frame_num - 1:
            break
        summary[id_next] = 1
        Q_value.append(max(action_value[0]))
        id_curr = id_next

    name = 'output/sum_' + test_name[i % test_num]
    sio.savemat(name, {'summary': summary})
print('Test done.')
