import tensorflow as tf
import numpy as np

#Load TFLite Model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

#Get Input Details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print input shape and type
print(interpreter.get_input_details()[0]['shape'])
print(interpreter.get_input_details()[0]['dtype'])

# Print output shape and type
print(interpreter.get_output_details()[0]['shape']) 
print(interpreter.get_output_details()[0]['dtype'])  

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = [np.arange(1.0, 1025.0, dtype=np.float32)]
interpreter.set_tensor(input_details[0]['index'], input_data)

#Run Inference
interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
