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
#ifndef DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_PREDICTION_SCHEME_INTERFACE_H_
#define DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_PREDICTION_SCHEME_INTERFACE_H_

#include "draco/attributes/point_attribute.h"
#include "draco/compression/config/compression_shared.h"

// Prediction schemes can be used during encoding and decoding of attributes
// to predict attribute values based on the previously encoded/decoded data.
// See prediction_scheme.h for more details.
namespace draco {

// Abstract interface for all prediction schemes used during attribute encoding.
class PredictionSchemeInterface {
 public:
  virtual ~PredictionSchemeInterface() = default;
  virtual PredictionSchemeMethod GetPredictionMethod() const = 0;

  // Returns the encoded attribute.
  virtual const PointAttribute *GetAttribute() const = 0;

  // Returns true when the prediction scheme is initialized with all data it
  // needs.
  virtual bool IsInitialized() const = 0;

  // Returns the number of parent attributes that are needed for the prediction.
  virtual int GetNumParentAttributes() const = 0;

  // Returns the type of each of the parent attribute.
  virtual GeometryAttribute::Type GetParentAttributeType(int i) const = 0;

  // Sets the required parent attribute.
  // Returns false if the attribute doesn't meet the requirements of the
  // prediction scheme.
  virtual bool SetParentAttribute(const PointAttribute *att) = 0;

  // Method should return true if the prediction scheme guarantees that all
  // correction values are always positive (or at least non-negative).
  virtual bool AreCorrectionsPositive() = 0;

  // Returns the transform type used by the prediction scheme.
  virtual PredictionSchemeTransformType GetTransformType() const = 0;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_PREDICTION_SCHEME_INTERFACE_H_
