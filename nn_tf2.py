import tensorflow as tf

#TF2 adaptation of nn.py to maintain compatibility
#loss type should be "mse" instead of "mean_square"

#https://www.tensorflow.org/api_docs/python/tf/keras/Model
class NeuralNet(tf.keras.Model):
    def __init__(self,layers,learning_rate_,loss_type,opt_type):

        super(NeuralNet, self).__init__()

        self.ave_value = []

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(layers[0].num_input,))

        self.hidden_layers = []
        for i in layers[:-1]:
            self.hidden_layers.append(tf.keras.layers.Dense(
                layers[i].num_output, activation=layers[i].activation_type,
                kernel_initializer='RandomNormal'))

        self.output_layer = tf.keras.layers.Dense(
            layers[-1].num_output , activation=layers[-1].activation_type,
            kernel_initializer='RandomNormal')
        
        if opt_type=='SGD':
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate_)
	else: 
	    opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate_)

        self.compile(optimizer=opt, loss=loss_type)
    
    
    def forward(self, data):
    	return self.call(data)


    @tf.function
    def call(self, data):
        z = self.input_layer(data)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output


    def train(self, x):
    	self.fit(x, y)


    def recover(self,path,file):
        try:
            self.load_weights(path + file)
        except:
            print('Error when restoring weights!')


    def saving(self,path,file):
        try:
            self.save_weights(path + file)
        except:
            print('Error when saving weights!')


    def to_TFLite(self):
        #convert 
        converter = tf.lite.TFLiteConverter.from_keras_model(self)
        tflite_model = converter.convert()

        # Save the model.
        with open('model.tflite', 'wb') as f:
            f.write(tflite_model)

