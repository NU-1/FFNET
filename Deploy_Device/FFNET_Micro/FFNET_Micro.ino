/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <TensorFlowLite.h>

#include "main_functions.h"
#include "constants.h"
#include "model.h"
#include "output_handler.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

// Increased arena size for FFNET Model
//Arena Size Estimator thinks we need: {"arena_size": 5872}
const int kModelArenaSize = 50*1024;
// Extra headroom for model + alignment + future interpreter changes.
const int kExtraArenaSize = 560 + 16 + 100;
const int kTensorArenaSize = kModelArenaSize + kExtraArenaSize;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
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

// The name of this function is important for Arduino compatibility.
void loop() {

  //Using Error Reporter to Print Input Tensor Information to Serial
  TF_LITE_REPORT_ERROR(error_reporter, "Input Tensor Number of Dimensions: %d\n", input->dims->size);
  TF_LITE_REPORT_ERROR(error_reporter, "Input First Dimension Size: %d\n", input->dims->data[0]);
  TF_LITE_REPORT_ERROR(error_reporter, "Input Second Dimension Size: %d\n", input->dims->data[1]);

  //Using Error Reporter to Print Output Tensor Information to Serial
  TF_LITE_REPORT_ERROR(error_reporter, "Output Tensor Number of Dimensions: %d\n", input->dims->size);
  TF_LITE_REPORT_ERROR(error_reporter, "Output First Dimension Size: %d\n", output->dims->data[0]);
  TF_LITE_REPORT_ERROR(error_reporter, "Output Second Dimension Size: %d\n", output->dims->data[1]);

  //Confirming input tensor type
  //TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, input->type);
 
  //dummy input/output
  float dummy_input[1028]; 
  float dummy_output[25];

  
  //create dummy input
   for(int i=0; i<1028; i++){
    dummy_input[i] = 1.0 + i ;
  }
  
  // Place dummy x values in the model's input tensor
  for(int i=0; i<1028; i++){
    input->data.f[i] = dummy_input[i];
    //TF_LITE_REPORT_ERROR(error_reporter, "Input Value, Idx %d: %f\n", i, dummy_input[i]);
  }


  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed %f\n",
                         NULL);
    return;
  }

  // Read the predicted y value from the model's output tensor
  
  for(int i=0; i<25; i++){
   dummy_output[i] = output->data.f[i];
   
   TF_LITE_REPORT_ERROR(error_reporter, "Output Value, Idx %d: %f\n", i, dummy_output[i]);
  }

  // Increment the inference_counter
  inference_count += 1;

  TF_LITE_REPORT_ERROR(error_reporter, "------------\n\nNext Inference\n", NULL );
    
}
