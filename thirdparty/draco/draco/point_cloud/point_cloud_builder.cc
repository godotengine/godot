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
#include "draco/point_cloud/point_cloud_builder.h"

#include <string>
#include <utility>

namespace draco {

PointCloudBuilder::PointCloudBuilder() {}

void PointCloudBuilder::Start(PointIndex::ValueType num_points) {
  point_cloud_ = std::unique_ptr<PointCloud>(new PointCloud());
  point_cloud_->set_num_points(num_points);
}

int PointCloudBuilder::AddAttribute(GeometryAttribute::Type attribute_type,
                                    int8_t num_components, DataType data_type) {
  return AddAttribute(attribute_type, num_components, data_type, false);
}

int PointCloudBuilder::AddAttribute(GeometryAttribute::Type attribute_type,
                                    int8_t num_components, DataType data_type,
                                    bool normalized) {
  GeometryAttribute ga;
  ga.Init(attribute_type, nullptr, num_components, data_type, normalized,
          DataTypeLength(data_type) * num_components, 0);
  return point_cloud_->AddAttribute(ga, true, point_cloud_->num_points());
}

void PointCloudBuilder::SetAttributeValueForPoint(int att_id,
                                                  PointIndex point_index,
                                                  const void *attribute_value) {
  PointAttribute *const att = point_cloud_->attribute(att_id);
  att->SetAttributeValue(att->mapped_index(point_index), attribute_value);
}

void PointCloudBuilder::SetAttributeValuesForAllPoints(
    int att_id, const void *attribute_values, int stride) {
  PointAttribute *const att = point_cloud_->attribute(att_id);
  const int data_stride =
      DataTypeLength(att->data_type()) * att->num_components();
  if (stride == 0) {
    stride = data_stride;
  }
  if (stride == data_stride) {
    // Fast copy path.
    att->buffer()->Write(0, attribute_values,
                         point_cloud_->num_points() * data_stride);
  } else {
    // Copy attribute entries one by one.
    for (PointIndex i(0); i < point_cloud_->num_points(); ++i) {
      att->SetAttributeValue(
          att->mapped_index(i),
          static_cast<const uint8_t *>(attribute_values) + stride * i.value());
    }
  }
}

std::unique_ptr<PointCloud> PointCloudBuilder::Finalize(
    bool deduplicate_points) {
  if (deduplicate_points) {
#ifdef DRACO_ATTRIBUTE_VALUES_DEDUPLICATION_SUPPORTED
    point_cloud_->DeduplicateAttributeValues();
#endif
#ifdef DRACO_ATTRIBUTE_INDICES_DEDUPLICATION_SUPPORTED
    point_cloud_->DeduplicatePointIds();
#endif
  }
  return std::move(point_cloud_);
}

void PointCloudBuilder::SetAttributeUniqueId(int att_id, uint32_t unique_id) {
  point_cloud_->attribute(att_id)->set_unique_id(unique_id);
}

#ifdef DRACO_TRANSCODER_SUPPORTED
void PointCloudBuilder::SetAttributeName(int att_id, const std::string &name) {
  point_cloud_->attribute(att_id)->set_name(name);
}
#endif  // DRACO_TRANSCODER_SUPPORTED

}  // namespace draco
