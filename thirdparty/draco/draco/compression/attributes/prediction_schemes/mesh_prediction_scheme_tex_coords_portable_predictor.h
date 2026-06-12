// Copyright 2017 The Draco Authors.
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
#ifndef DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_MESH_PREDICTION_SCHEME_TEX_COORDS_PORTABLE_PREDICTOR_H_
#define DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_MESH_PREDICTION_SCHEME_TEX_COORDS_PORTABLE_PREDICTOR_H_

#include <math.h>

#include <algorithm>
#include <limits>

#include "draco/attributes/point_attribute.h"
#include "draco/core/math_utils.h"
#include "draco/core/vector_d.h"
#include "draco/mesh/corner_table.h"

namespace draco {

// Predictor functionality used for portable UV prediction by both encoder and
// decoder.
template <typename DataTypeT, class MeshDataT>
class MeshPredictionSchemeTexCoordsPortablePredictor {
 public:
  static constexpr int kNumComponents = 2;

  explicit MeshPredictionSchemeTexCoordsPortablePredictor(const MeshDataT &md)
      : pos_attribute_(nullptr),
        entry_to_point_id_map_(nullptr),
        mesh_data_(md) {}
  void SetPositionAttribute(const PointAttribute &position_attribute) {
    pos_attribute_ = &position_attribute;
  }
  void SetEntryToPointIdMap(const PointIndex *map) {
    entry_to_point_id_map_ = map;
  }
  bool IsInitialized() const { return pos_attribute_ != nullptr; }

  VectorD<int64_t, 3> GetPositionForEntryId(int entry_id) const {
    const PointIndex point_id = entry_to_point_id_map_[entry_id];
    VectorD<int64_t, 3> pos;
    pos_attribute_->ConvertValue(pos_attribute_->mapped_index(point_id),
                                 &pos[0]);
    return pos;
  }

  VectorD<int64_t, 2> GetTexCoordForEntryId(int entry_id,
                                            const DataTypeT *data) const {
    const int data_offset = entry_id * kNumComponents;
    return VectorD<int64_t, 2>(data[data_offset], data[data_offset + 1]);
  }

  // Computes predicted UV coordinates on a given corner. The coordinates are
  // stored in |predicted_value_| member.
  template <bool is_encoder_t>
  bool ComputePredictedValue(CornerIndex corner_id, const DataTypeT *data,
                             int data_id);

  const DataTypeT *predicted_value() const { return predicted_value_; }
  bool orientation(int i) const { return orientations_[i]; }
  void set_orientation(int i, bool v) { orientations_[i] = v; }
  size_t num_orientations() const { return orientations_.size(); }
  void ResizeOrientations(int num_orientations) {
    orientations_.resize(num_orientations);
  }

 private:
  const PointAttribute *pos_attribute_;
  const PointIndex *entry_to_point_id_map_;
  DataTypeT predicted_value_[kNumComponents];
  // Encoded / decoded array of UV flips.
  // TODO(ostava): We should remove this and replace this with in-place encoding
  // and decoding to avoid unnecessary copy.
  std::vector<bool> orientations_;
  MeshDataT mesh_data_;
};

template <typename DataTypeT, class MeshDataT>
template <bool is_encoder_t>
bool MeshPredictionSchemeTexCoordsPortablePredictor<
    DataTypeT, MeshDataT>::ComputePredictedValue(CornerIndex corner_id,
                                                 const DataTypeT *data,
                                                 int data_id) {
  // Compute the predicted UV coordinate from the positions on all corners
  // of the processed triangle. For the best prediction, the UV coordinates
  // on the next/previous corners need to be already encoded/decoded.
  const CornerIndex next_corner_id = mesh_data_.corner_table()->Next(corner_id);
  const CornerIndex prev_corner_id =
      mesh_data_.corner_table()->Previous(corner_id);
  // Get the encoded data ids from the next and previous corners.
  // The data id is the encoding order of the UV coordinates.
  int next_data_id, prev_data_id;

  int next_vert_id, prev_vert_id;
  next_vert_id = mesh_data_.corner_table()->Vertex(next_corner_id).value();
  prev_vert_id = mesh_data_.corner_table()->Vertex(prev_corner_id).value();

  next_data_id = mesh_data_.vertex_to_data_map()->at(next_vert_id);
  prev_data_id = mesh_data_.vertex_to_data_map()->at(prev_vert_id);

  typedef VectorD<int64_t, 2> Vec2;
  typedef VectorD<int64_t, 3> Vec3;
  typedef VectorD<uint64_t, 2> Vec2u;

  if (prev_data_id < data_id && next_data_id < data_id) {
    // Both other corners have available UV coordinates for prediction.
    const Vec2 n_uv = GetTexCoordForEntryId(next_data_id, data);
    const Vec2 p_uv = GetTexCoordForEntryId(prev_data_id, data);
    if (p_uv == n_uv) {
      // We cannot do a reliable prediction on degenerated UV triangles.
      predicted_value_[0] = p_uv[0];
      predicted_value_[1] = p_uv[1];
      return true;
    }

    // Get positions at all corners.
    const Vec3 tip_pos = GetPositionForEntryId(data_id);
    const Vec3 next_pos = GetPositionForEntryId(next_data_id);
    const Vec3 prev_pos = GetPositionForEntryId(prev_data_id);
    // We use the positions of the above triangle to predict the texture
    // coordinate on the tip corner C.
    // To convert the triangle into the UV coordinate system we first compute
    // position X on the vector |prev_pos - next_pos| that is the projection of
    // point C onto vector |prev_pos - next_pos|:
    //
    //              C
    //             /.  \
    //            / .     \
    //           /  .        \
    //          N---X----------P
    //
    // Where next_pos is point (N), prev_pos is point (P) and tip_pos is the
    // position of predicted coordinate (C).
    //
    const Vec3 pn = prev_pos - next_pos;
    const uint64_t pn_norm2_squared = pn.SquaredNorm();
    if (pn_norm2_squared != 0) {
      // Compute the projection of C onto PN by computing dot product of CN with
      // PN and normalizing it by length of PN. This gives us a factor |s| where
      // |s = PN.Dot(CN) / PN.SquaredNorm2()|. This factor can be used to
      // compute X in UV space |X_UV| as |X_UV = N_UV + s * PN_UV|.
      const Vec3 cn = tip_pos - next_pos;
      const int64_t cn_dot_pn = pn.Dot(cn);

      const Vec2 pn_uv = p_uv - n_uv;
      // Because we perform all computations with integers, we don't explicitly
      // compute the normalized factor |s|, but rather we perform all operations
      // over UV vectors in a non-normalized coordinate system scaled with a
      // scaling factor |pn_norm2_squared|:
      //
      //      x_uv = X_UV * PN.Norm2Squared()
      //
      const int64_t n_uv_absmax_element =
          std::max(std::abs(n_uv[0]), std::abs(n_uv[1]));
      if (n_uv_absmax_element >
          std::numeric_limits<int64_t>::max() / pn_norm2_squared) {
        // Return false if the below multiplication would overflow.
        return false;
      }
      const int64_t pn_uv_absmax_element =
          std::max(std::abs(pn_uv[0]), std::abs(pn_uv[1]));
      if (std::abs(cn_dot_pn) >
          std::numeric_limits<int64_t>::max() / pn_uv_absmax_element) {
        // Return false if squared length calculation would overflow.
        return false;
      }
      const Vec2 x_uv = n_uv * pn_norm2_squared + (cn_dot_pn * pn_uv);
      const int64_t pn_absmax_element =
          std::max(std::max(std::abs(pn[0]), std::abs(pn[1])), std::abs(pn[2]));
      if (std::abs(cn_dot_pn) >
          std::numeric_limits<int64_t>::max() / pn_absmax_element) {
        // Return false if squared length calculation would overflow.
        return false;
      }

      // Compute squared length of vector CX in position coordinate system:
      const Vec3 x_pos = next_pos + (cn_dot_pn * pn) / pn_norm2_squared;
      const uint64_t cx_norm2_squared = (tip_pos - x_pos).SquaredNorm();

      // Compute vector CX_UV in the uv space by rotating vector PN_UV by 90
      // degrees and scaling it with factor CX.Norm2() / PN.Norm2():
      //
      //     CX_UV = (CX.Norm2() / PN.Norm2()) * Rot(PN_UV)
      //
      // To preserve precision, we perform all operations in scaled space as
      // explained above, so we want the final vector to be:
      //
      //     cx_uv = CX_UV * PN.Norm2Squared()
      //
      // We can then rewrite the formula as:
      //
      //     cx_uv = CX.Norm2() * PN.Norm2() * Rot(PN_UV)
      //
      Vec2 cx_uv(pn_uv[1], -pn_uv[0]);  // Rotated PN_UV.
      // Compute CX.Norm2() * PN.Norm2()
      const uint64_t norm_squared =
          IntSqrt(cx_norm2_squared * pn_norm2_squared);
      // Final cx_uv in the scaled coordinate space.
      cx_uv = cx_uv * norm_squared;

      // Predicted uv coordinate is then computed by either adding or
      // subtracting CX_UV to/from X_UV.
      Vec2 predicted_uv;
      if (is_encoder_t) {
        // When encoding, compute both possible vectors and determine which one
        // results in a better prediction.
        // Both vectors need to be transformed back from the scaled space to
        // the real UV coordinate space.
        const Vec2 predicted_uv_0((x_uv + cx_uv) / pn_norm2_squared);
        const Vec2 predicted_uv_1((x_uv - cx_uv) / pn_norm2_squared);
        const Vec2 c_uv = GetTexCoordForEntryId(data_id, data);
        if ((c_uv - predicted_uv_0).SquaredNorm() <
            (c_uv - predicted_uv_1).SquaredNorm()) {
          predicted_uv = predicted_uv_0;
          orientations_.push_back(true);
        } else {
          predicted_uv = predicted_uv_1;
          orientations_.push_back(false);
        }
      } else {
        // When decoding the data, we already know which orientation to use.
        if (orientations_.empty()) {
          return false;
        }
        const bool orientation = orientations_.back();
        orientations_.pop_back();
        // Perform operations in unsigned type to avoid signed integer overflow.
        // Note that the result will be the same (for non-overflowing values).
        if (orientation) {
          predicted_uv = Vec2(Vec2u(x_uv) + Vec2u(cx_uv)) / pn_norm2_squared;
        } else {
          predicted_uv = Vec2(Vec2u(x_uv) - Vec2u(cx_uv)) / pn_norm2_squared;
        }
      }
      predicted_value_[0] = static_cast<int>(predicted_uv[0]);
      predicted_value_[1] = static_cast<int>(predicted_uv[1]);
      return true;
    }
  }
  // Else we don't have available textures on both corners or the position data
  // is invalid. For such cases we can't use positions for predicting the uv
  // value and we resort to delta coding.
  int data_offset = 0;
  if (prev_data_id < data_id) {
    // Use the value on the previous corner as the prediction.
    data_offset = prev_data_id * kNumComponents;
  }
  if (next_data_id < data_id) {
    // Use the value on the next corner as the prediction.
    data_offset = next_data_id * kNumComponents;
  } else {
    // None of the other corners have a valid value. Use the last encoded value
    // as the prediction if possible.
    if (data_id > 0) {
      data_offset = (data_id - 1) * kNumComponents;
    } else {
      // We are encoding the first value. Predict 0.
      for (int i = 0; i < kNumComponents; ++i) {
        predicted_value_[i] = 0;
      }
      return true;
    }
  }
  for (int i = 0; i < kNumComponents; ++i) {
    predicted_value_[i] = data[data_offset + i];
  }
  return true;
}

}  // namespace draco

#endif  // DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_MESH_PREDICTION_SCHEME_TEX_COORDS_PORTABLE_PREDICTOR_H_
