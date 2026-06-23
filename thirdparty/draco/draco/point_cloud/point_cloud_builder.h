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
#ifndef DRACO_POINT_CLOUD_POINT_CLOUD_BUILDER_H_
#define DRACO_POINT_CLOUD_POINT_CLOUD_BUILDER_H_

#include <utility>

#include "draco/point_cloud/point_cloud.h"

namespace draco {

// A helper class for constructing PointCloud instances from other data sources.
// Usage:
//   PointCloudBuilder builder;
//   // Initialize the builder for a given number of points (required).
//   builder.Start(num_points);
//   // Specify desired attributes.
//   int pos_att_id =
//       builder.AddAttribute(GeometryAttribute::POSITION, 3, DT_FLOAT32);
//   // Add attribute values.
//   for (PointIndex i(0); i < num_points; ++i) {
//     builder.SetAttributeValueForPoint(pos_att_id, i, input_pos[i.value()]);
//   }
//   // Get the final PointCloud.
//   constexpr bool deduplicate_points = false;
//   std::unique_ptr<PointCloud> pc = builder.Finalize(deduplicate_points);

class PointCloudBuilder {
 public:
  // Index type of the inserted element.
  typedef PointIndex ElementIndex;

  PointCloudBuilder();

  // Starts collecting point cloud data.
  // The behavior of other functions is undefined before this method is called.
  void Start(PointIndex::ValueType num_points);

  int AddAttribute(GeometryAttribute::Type attribute_type,
                   int8_t num_components, DataType data_type);
  int AddAttribute(GeometryAttribute::Type attribute_type,
                   int8_t num_components, DataType data_type, bool normalized);

  // Sets attribute value for a specific point.
  // |attribute_value| must contain data in the format specified by the
  // AddAttribute method.
  void SetAttributeValueForPoint(int att_id, PointIndex point_index,
                                 const void *attribute_value);

  // Sets attribute values for all points. All the values must be stored in the
  // input |attribute_values| buffer. |stride| can be used to define the byte
  // offset between two consecutive attribute values. If |stride| is set to 0,
  // the stride is automatically computed based on the format of the given
  // attribute.
  void SetAttributeValuesForAllPoints(int att_id, const void *attribute_values,
                                      int stride);

  // Sets the unique ID for an attribute created with AddAttribute().
  void SetAttributeUniqueId(int att_id, uint32_t unique_id);

#ifdef DRACO_TRANSCODER_SUPPORTED
  // Sets attribute name.
  void SetAttributeName(int att_id, const std::string &name);
#endif  // DRACO_TRANSCODER_SUPPORTED

  // Finalizes the PointCloud or returns nullptr on error.
  // If |deduplicate_points| is set to true, the following happens:
  //   1. Attribute values with duplicate entries are deduplicated.
  //   2. Point ids that are mapped to the same attribute values are
  //      deduplicated.
  // Therefore, if |deduplicate_points| is true the final PointCloud can have
  // a different number of point from the value specified in the Start method.
  // Once this function is called, the builder becomes invalid and cannot be
  // used until the method Start() is called again.
  std::unique_ptr<PointCloud> Finalize(bool deduplicate_points);

  // Add metadata for an attribute.
  void AddAttributeMetadata(int32_t att_id,
                            std::unique_ptr<AttributeMetadata> metadata) {
    point_cloud_->AddAttributeMetadata(att_id, std::move(metadata));
  }

 private:
  std::unique_ptr<PointCloud> point_cloud_;
};

}  // namespace draco

#endif  // DRACO_POINT_CLOUD_POINT_CLOUD_BUILDER_H_
