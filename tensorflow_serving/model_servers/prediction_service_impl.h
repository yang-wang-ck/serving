/* Copyright 2018 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_SERVING_MODEL_SERVERS_PREDICTION_SERVICE_IMPL_H_
#define TENSORFLOW_SERVING_MODEL_SERVERS_PREDICTION_SERVICE_IMPL_H_

#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#include "tensorflow_serving/model_servers/server_core.h"
#include "tensorflow_serving/servables/tensorflow/predict_impl.h"

#include "boost/asio.hpp"
using namespace boost::asio;

io_service io_service;
ip::udp::socket socket(io_service);
ip::udp::endpoint remote_endpoint;

socket.open(ip::udp::v4());

remote_endpoint = ip::udp::endpoint(ip::address::from_string("192.168.0.4"), 9000);

boost::system::error_code err;
socket.send_to(buffer("Jane Doe", 8), remote_endpoint, 0, err);

socket.close();

namespace tensorflow {
namespace serving {

class PredictionServiceImpl final : public PredictionService::Service {
 public:
  explicit PredictionServiceImpl(ServerCore* core, bool use_saved_model)
      : core_(core),
        predictor_(new TensorflowPredictor(use_saved_model)),
        use_saved_model_(use_saved_model) {}

  ::grpc::Status Predict(::grpc::ServerContext* context,
                         const PredictRequest* request,
                         PredictResponse* response) override;

  ::grpc::Status GetModelMetadata(::grpc::ServerContext* context,
                                  const GetModelMetadataRequest* request,
                                  GetModelMetadataResponse* response) override;

  ::grpc::Status Classify(::grpc::ServerContext* context,
                          const ClassificationRequest* request,
                          ClassificationResponse* response) override;

  ::grpc::Status Regress(::grpc::ServerContext* context,
                         const RegressionRequest* request,
                         RegressionResponse* response) override;

  ::grpc::Status MultiInference(::grpc::ServerContext* context,
                                const MultiInferenceRequest* request,
                                MultiInferenceResponse* response) override;

 private:
  ServerCore* core_;
  std::unique_ptr<TensorflowPredictor> predictor_;
  const bool use_saved_model_;
};

}  // namespace serving
}  // namespace tensorflow

#endif  // TENSORFLOW_SERVING_MODEL_SERVERS_PREDICTION_SERVICE_IMPL_H_
