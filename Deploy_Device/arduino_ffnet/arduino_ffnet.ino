#include "Arduino.h"

#include "model.h"

#include <TensorFlowLite.h>
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"


//Constants
//Arduino's built-in LED pin #
int led = LED_BUILTIN;


namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

// Memory for input, output, and intermediate arrays.
const int kModelArenaSize = 2468;
const int kExtraArenaSize = 560 + 16 + 100;
const int kTensorArenaSize = kModelArenaSize + kExtraArenaSize;
uint8_t tensor_arena[kTensorArenaSize];
} 



void setup(){
	
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  //Make model usable
  model = tflite::GetModel(FFNet_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);


  
  // Keep track of how many inferences we have performed.
  inference_count = 0;
}


void loop(){
	blink();

Serial.print(interpreter->get_input_details())
  

//Run Interpreter
TfLiteStatus invoke_status = interpreter->Invoke();

int8_t y_quantized = output->data.int8[0];


inference_count += 1;

 }



void blink(){
  analogWrite(led, 255);
  delay(2000);
  analogWrite(led,0);
  delay(2000);
  //Serial.print("Hello\n");
  analogWrite(led, 255);
  delay(1000);
  analogWrite(led,0);
  delay(1000);
}



 
