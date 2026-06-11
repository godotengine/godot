// Copyright 2016 The Draco Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#ifndef DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_MESH_PREDICTION_SCHEME_CONSTRAINED_MULTI_PARALLELOGRAM_SHARED_H_
#define DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_MESH_PREDICTION_SCHEME_CONSTRAINED_MULTI_PARALLELOGRAM_SHARED_H_

namespace draco {

// Data shared between constrained multi-parallelogram encoder and decoder.
namespace constrained_multi_parallelogram {

enum Mode {
  // Selects the optimal multi-parallelogram from up to 4 available
  // parallelograms.
  OPTIMAL_MULTI_PARALLELOGRAM = 0,
};

static constexpr int kMaxNumParallelograms = 4;

}  // namespace constrained_multi_parallelogram
}  // namespace draco

#endif  // DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_MESH_PREDICTION_SCHEME_CONSTRAINED_MULTI_PARALLELOGRAM_SHARED_H_
