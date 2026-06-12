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
#ifndef DRACO_COMPRESSION_ATTRIBUTES_LINEAR_SEQUENCER_H_
#define DRACO_COMPRESSION_ATTRIBUTES_LINEAR_SEQUENCER_H_

#include "draco/compression/attributes/points_sequencer.h"

namespace draco {

// A simple sequencer that generates a linear sequence [0, num_points - 1].
// I.e., the order of the points is preserved for the input data.
class LinearSequencer : public PointsSequencer {
 public:
  explicit LinearSequencer(int32_t num_points) : num_points_(num_points) {}

  bool UpdatePointToAttributeIndexMapping(PointAttribute *attribute) override {
    attribute->SetIdentityMapping();
    return true;
  }

 protected:
  bool GenerateSequenceInternal() override {
    if (num_points_ < 0) {
      return false;
    }
    out_point_ids()->resize(num_points_);
    for (int i = 0; i < num_points_; ++i) {
      out_point_ids()->at(i) = PointIndex(i);
    }
    return true;
  }

 private:
  int32_t num_points_;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_ATTRIBUTES_LINEAR_SEQUENCER_H_
