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
#ifndef DRACO_MESH_TRIANGLE_SOUP_MESH_BUILDER_H_
#define DRACO_MESH_TRIANGLE_SOUP_MESH_BUILDER_H_

#include <utility>
#include <vector>

#include "draco/draco_features.h"

#ifdef DRACO_TRANSCODER_SUPPORTED
#include "draco/core/status.h"
#endif
#include "draco/mesh/mesh.h"

namespace draco {

// Class for building meshes directly from attribute values that can be
// specified for each face corner. All attributes are automatically
// deduplicated.
class TriangleSoupMeshBuilder {
 public:
  // Index type of the inserted element.
  typedef FaceIndex ElementIndex;

  // Starts mesh building for a given number of faces.
  // TODO(ostava): Currently it's necessary to select the correct number of
  // faces upfront. This should be generalized, but it will require us to
  // rewrite our attribute resizing functions.
  void Start(int num_faces);

#ifdef DRACO_TRANSCODER_SUPPORTED
  // Sets mesh name.
  void SetName(const std::string &name);
#endif  // DRACO_TRANSCODER_SUPPORTED

  // Adds an empty attribute to the mesh. Returns the new attribute's id.
  int AddAttribute(GeometryAttribute::Type attribute_type,
                   int8_t num_components, DataType data_type);
  int AddAttribute(GeometryAttribute::Type attribute_type,
                   int8_t num_components, DataType data_type, bool normalized);

  // Sets values for a given attribute on all corners of a given face.
  void SetAttributeValuesForFace(int att_id, FaceIndex face_id,
                                 const void *corner_value_0,
                                 const void *corner_value_1,
                                 const void *corner_value_2);

#ifdef DRACO_TRANSCODER_SUPPORTED
  // Converts input values of type T into internal representation used by
  // |att_id|. Each input value needs to have |input_num_components| entries.
  template <typename T>
  Status ConvertAndSetAttributeValuesForFace(int att_id, FaceIndex face_id,
                                             int input_num_components,
                                             const T *corner_value_0,
                                             const T *corner_value_1,
                                             const T *corner_value_2);
#endif

  // Sets value for a per-face attribute. If all faces of a given attribute are
  // set with this method, the attribute will be marked as per-face, otherwise
  // it will be marked as per-corner attribute.
  void SetPerFaceAttributeValueForFace(int att_id, FaceIndex face_id,
                                       const void *value);

  // Add metadata.
  void AddMetadata(std::unique_ptr<GeometryMetadata> metadata) {
    mesh_->AddMetadata(std::move(metadata));
  }

  // Sets the unique ID for an attribute created with AddAttribute().
  void SetAttributeUniqueId(int att_id, uint32_t unique_id);

#ifdef DRACO_TRANSCODER_SUPPORTED
  // Sets attribute name.
  void SetAttributeName(int att_id, const std::string &name);
#endif  // DRACO_TRANSCODER_SUPPORTED

  // Add metadata for an attribute.
  void AddAttributeMetadata(int32_t att_id,
                            std::unique_ptr<AttributeMetadata> metadata) {
    mesh_->AddAttributeMetadata(att_id, std::move(metadata));
  }

  // Finalizes the mesh or returns nullptr on error.
  // Once this function is called, the builder becomes invalid and cannot be
  // used until the method Start() is called again.
  std::unique_ptr<Mesh> Finalize();

 private:
  std::vector<int8_t> attribute_element_types_;

  std::unique_ptr<Mesh> mesh_;
};

#ifdef DRACO_TRANSCODER_SUPPORTED
template <typename T>
Status TriangleSoupMeshBuilder::ConvertAndSetAttributeValuesForFace(
    int att_id, FaceIndex face_id, int input_num_components,
    const T *corner_value_0, const T *corner_value_1, const T *corner_value_2) {
  const int start_index = 3 * face_id.value();
  PointAttribute *const att = mesh_->attribute(att_id);
  DRACO_RETURN_IF_ERROR(
      att->ConvertAndSetAttributeValue(AttributeValueIndex(start_index + 0),
                                       input_num_components, corner_value_0));
  DRACO_RETURN_IF_ERROR(
      att->ConvertAndSetAttributeValue(AttributeValueIndex(start_index + 1),
                                       input_num_components, corner_value_1));
  DRACO_RETURN_IF_ERROR(
      att->ConvertAndSetAttributeValue(AttributeValueIndex(start_index + 2),
                                       input_num_components, corner_value_2));
  mesh_->SetFace(face_id,
                 {{PointIndex(start_index), PointIndex(start_index + 1),
                   PointIndex(start_index + 2)}});
  attribute_element_types_[att_id] = MESH_CORNER_ATTRIBUTE;
  return OkStatus();
}
#endif  // DRACO_TRANSCODER_SUPPORTED

}  // namespace draco

#endif  // DRACO_MESH_TRIANGLE_SOUP_MESH_BUILDER_H_
