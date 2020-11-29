Steps: 


1. Convert .tflite file to C array: Use TFLite_to_C or Unix xxd program (xxd -i ./model.tflite > ./model.cc)

2. Copy the output array (declare it as const) into FFNET/Deploy_Host/main.cc 

3. Inspect Model's Input and Output Tensors (Inspect_Model)

4. For Arduino <--> Python communication, pip install pyserialtransfer, and get arduino library 

