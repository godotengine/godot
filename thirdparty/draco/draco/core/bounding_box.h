// Copyright 2018 The Draco Authors.
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
#ifndef DRACO_CORE_BOUNDING_BOX_H_
#define DRACO_CORE_BOUNDING_BOX_H_

#include "draco/core/vector_d.h"

namespace draco {

// Class for computing the bounding box of points in 3D space.
class BoundingBox {
 public:
  // Creates bounding box object with minimum and maximum points initialized to
  // the largest positive and the smallest negative values, respectively. The
  // resulting abstract bounding box effectively has no points and can be
  // updated by providing any point to Update() method.
  BoundingBox();

  // Creates bounding box object with minimum and maximum points initialized to
  // |min_point| and |max_point|, respectively.
  BoundingBox(const Vector3f &min_point, const Vector3f &max_point);

  // Returns the minimum point of the bounding box.
  inline const Vector3f &GetMinPoint() const { return min_point_; }

  // Returns the maximum point of the bounding box.
  inline const Vector3f &GetMaxPoint() const { return max_point_; }

  // Checks if the bounding box object was created with the default constructor
  // then never updated. Internally, checks if the bounding box minimum and
  // maximum points hold the largest positive and smallest negative values.
  const bool IsValid() const;

  // Conditionally updates the bounding box with a given |new_point|.
  void Update(const Vector3f &new_point) {
    for (int i = 0; i < 3; i++) {
      if (new_point[i] < min_point_[i]) {
        min_point_[i] = new_point[i];
      }
      if (new_point[i] > max_point_[i]) {
        max_point_[i] = new_point[i];
      }
    }
  }

  // Updates bounding box with minimum and maximum points of the |other|
  // bounding box.
  void Update(const BoundingBox &other) {
    Update(other.GetMinPoint());
    Update(other.GetMaxPoint());
  }

  // Returns the size of the bounding box along each axis.
  Vector3f Size() const { return max_point_ - min_point_; }

  // Returns the center of the bounding box.
  Vector3f Center() const { return (min_point_ + max_point_) / 2; }

 private:
  Vector3f min_point_;
  Vector3f max_point_;
};
}  // namespace draco

#endif  //  DRACO_CORE_BOUNDING_BOX_H_
