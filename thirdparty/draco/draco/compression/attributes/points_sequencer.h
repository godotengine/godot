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
#ifndef DRACO_COMPRESSION_ATTRIBUTES_POINTS_SEQUENCER_H_
#define DRACO_COMPRESSION_ATTRIBUTES_POINTS_SEQUENCER_H_

#include <vector>

#include "draco/attributes/point_attribute.h"

namespace draco {

// Class for generating a sequence of point ids that can be used to encode
// or decode attribute values in a specific order.
// See sequential_attribute_encoders/decoders_controller.h for more details.
class PointsSequencer {
 public:
  PointsSequencer() : out_point_ids_(nullptr) {}
  virtual ~PointsSequencer() = default;

  // Fills the |out_point_ids| with the generated sequence of point ids.
  bool GenerateSequence(std::vector<PointIndex> *out_point_ids) {
    out_point_ids_ = out_point_ids;
    return GenerateSequenceInternal();
  }

  // Appends a point to the sequence.
  void AddPointId(PointIndex point_id) { out_point_ids_->push_back(point_id); }

  // Sets the correct mapping between point ids and value ids. I.e., the inverse
  // of the |out_point_ids|. In general, |out_point_ids_| does not contain
  // sufficient information to compute the inverse map, because not all point
  // ids are necessarily contained within the map.
  // Must be implemented for sequencers that are used by attribute decoders.
  virtual bool UpdatePointToAttributeIndexMapping(PointAttribute * /* attr */) {
    return false;
  }

 protected:
  // Method that needs to be implemented by the derived classes. The
  // implementation is responsible for filling |out_point_ids_| with the valid
  // sequence of point ids.
  virtual bool GenerateSequenceInternal() = 0;
  std::vector<PointIndex> *out_point_ids() const { return out_point_ids_; }

 private:
  std::vector<PointIndex> *out_point_ids_;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_ATTRIBUTES_POINTS_SEQUENCER_H_
