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
#ifndef DRACO_COMPRESSION_CONFIG_COMPRESSION_SHARED_H_
#define DRACO_COMPRESSION_CONFIG_COMPRESSION_SHARED_H_

#include <stdint.h>

#include "draco/core/macros.h"
#include "draco/draco_features.h"

namespace draco {

// Latest Draco bit-stream version.
static constexpr uint8_t kDracoPointCloudBitstreamVersionMajor = 2;
static constexpr uint8_t kDracoPointCloudBitstreamVersionMinor = 3;
static constexpr uint8_t kDracoMeshBitstreamVersionMajor = 2;
static constexpr uint8_t kDracoMeshBitstreamVersionMinor = 2;

// Concatenated latest bit-stream version.
static constexpr uint16_t kDracoPointCloudBitstreamVersion =
    DRACO_BITSTREAM_VERSION(kDracoPointCloudBitstreamVersionMajor,
                            kDracoPointCloudBitstreamVersionMinor);

static constexpr uint16_t kDracoMeshBitstreamVersion = DRACO_BITSTREAM_VERSION(
    kDracoMeshBitstreamVersionMajor, kDracoMeshBitstreamVersionMinor);

// Currently, we support point cloud and triangular mesh encoding.
// TODO(draco-eng) Convert enum to enum class (safety, not performance).
enum EncodedGeometryType {
  INVALID_GEOMETRY_TYPE = -1,
  POINT_CLOUD = 0,
  TRIANGULAR_MESH,
  NUM_ENCODED_GEOMETRY_TYPES
};

// List of encoding methods for point clouds.
enum PointCloudEncodingMethod {
  POINT_CLOUD_SEQUENTIAL_ENCODING = 0,
  POINT_CLOUD_KD_TREE_ENCODING
};

// List of encoding methods for meshes.
enum MeshEncoderMethod {
  MESH_SEQUENTIAL_ENCODING = 0,
  MESH_EDGEBREAKER_ENCODING,
};

// List of various attribute encoders supported by our framework. The entries
// are used as unique identifiers of the encoders and their values should not
// be changed!
enum AttributeEncoderType {
  BASIC_ATTRIBUTE_ENCODER = 0,
  MESH_TRAVERSAL_ATTRIBUTE_ENCODER,
  KD_TREE_ATTRIBUTE_ENCODER,
};

// List of various sequential attribute encoder/decoders that can be used in our
// pipeline. The values represent unique identifiers used by the decoder and
// they should not be changed.
enum SequentialAttributeEncoderType {
  SEQUENTIAL_ATTRIBUTE_ENCODER_GENERIC = 0,
  SEQUENTIAL_ATTRIBUTE_ENCODER_INTEGER,
  SEQUENTIAL_ATTRIBUTE_ENCODER_QUANTIZATION,
  SEQUENTIAL_ATTRIBUTE_ENCODER_NORMALS,
};

// List of all prediction methods currently supported by our framework.
enum PredictionSchemeMethod {
  // Special value indicating that no prediction scheme was used.
  PREDICTION_NONE = -2,
  // Used when no specific prediction scheme is required.
  PREDICTION_UNDEFINED = -1,
  PREDICTION_DIFFERENCE = 0,
  MESH_PREDICTION_PARALLELOGRAM = 1,
  MESH_PREDICTION_MULTI_PARALLELOGRAM = 2,
  MESH_PREDICTION_TEX_COORDS_DEPRECATED = 3,
  MESH_PREDICTION_CONSTRAINED_MULTI_PARALLELOGRAM = 4,
  MESH_PREDICTION_TEX_COORDS_PORTABLE = 5,
  MESH_PREDICTION_GEOMETRIC_NORMAL = 6,
  NUM_PREDICTION_SCHEMES
};

// List of all prediction scheme transforms used by our framework.
enum PredictionSchemeTransformType {
  PREDICTION_TRANSFORM_NONE = -1,
  // Basic delta transform where the prediction is computed as difference the
  // predicted and original value.
  PREDICTION_TRANSFORM_DELTA = 0,
  // An improved delta transform where all computed delta values are wrapped
  // around a fixed interval which lowers the entropy.
  PREDICTION_TRANSFORM_WRAP = 1,
  // Specialized transform for normal coordinates using inverted tiles.
  PREDICTION_TRANSFORM_NORMAL_OCTAHEDRON = 2,
  // Specialized transform for normal coordinates using canonicalized inverted
  // tiles.
  PREDICTION_TRANSFORM_NORMAL_OCTAHEDRON_CANONICALIZED = 3,
  // The number of valid (non-negative) prediction scheme transform types.
  NUM_PREDICTION_SCHEME_TRANSFORM_TYPES
};

// List of all mesh traversal methods supported by Draco framework.
enum MeshTraversalMethod {
  MESH_TRAVERSAL_DEPTH_FIRST = 0,
  MESH_TRAVERSAL_PREDICTION_DEGREE = 1,
  NUM_TRAVERSAL_METHODS
};

// List of all variant of the edgebreaker method that is used for compression
// of mesh connectivity.
enum MeshEdgebreakerConnectivityEncodingMethod {
  MESH_EDGEBREAKER_STANDARD_ENCODING = 0,
  MESH_EDGEBREAKER_PREDICTIVE_ENCODING = 1,  // Deprecated.
  MESH_EDGEBREAKER_VALENCE_ENCODING = 2,
};

// Draco header V1
struct DracoHeader {
  int8_t draco_string[5];
  uint8_t version_major;
  uint8_t version_minor;
  uint8_t encoder_type;
  uint8_t encoder_method;
  uint16_t flags;
};

enum NormalPredictionMode {
  ONE_TRIANGLE = 0,  // To be deprecated.
  TRIANGLE_AREA = 1,
};

// Different methods used for symbol entropy encoding.
enum SymbolCodingMethod {
  SYMBOL_CODING_TAGGED = 0,
  SYMBOL_CODING_RAW = 1,
  NUM_SYMBOL_CODING_METHODS,
};

// Mask for setting and getting the bit for metadata in |flags| of header.
#define METADATA_FLAG_MASK 0x8000

}  // namespace draco

#endif  // DRACO_COMPRESSION_CONFIG_COMPRESSION_SHARED_H_
