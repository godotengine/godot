// SPDX-License-Identifier: Apache 2.0
// Copyright 2022 - 2023, Syoyo Fujita.
// Copyright 2023 - Present, Light Transport Entertainment Inc.
//
// Render data structure suited for WebGL and Raytracing render
//
#pragma once

#include <algorithm>
#include <cmath>
#include <unordered_map>

#include "asset-resolution.hh"
#include "nonstd/expected.hpp"
#include "usdGeom.hh"
#include "usdShade.hh"
#include "usdSkel.hh"
#include "value-types.hh"

// tydra
#include "scene-access.hh"

namespace tinyusdz {

// forward decl
class Stage;
class Prim;
struct Material;
struct GeomMesh;
struct Xform;
struct AssetInfo;
class Path;
struct UsdPreviewSurface;
struct UsdUVTexture;

template <typename T>
struct UsdPrimvarReader;

using UsdPrimvarReader_int = UsdPrimvarReader<int>;
using UsdPrimvarReader_float = UsdPrimvarReader<float>;
using UsdPrimvarReader_float3 = UsdPrimvarReader<value::float3>;
using UsdPrimvarReader_float3 = UsdPrimvarReader<value::float3>;
using UsdPrimvarReader_string = UsdPrimvarReader<std::string>;
using UsdPrimvarReader_matrix4d = UsdPrimvarReader<value::matrix4d>;

namespace tydra {

// GLSL like data types
using vec2 = value::float2;
using vec3 = value::float3;
using vec4 = value::float4;
using quat = value::float4; // (x, y, z, w)
using mat2 = value::matrix2f;
using mat3 = value::matrix3f;
using mat4 = value::matrix4f;
using dmat4 = value::matrix4d;

// Simple string <-> id map
struct StringAndIdMap {
  void add(uint64_t key, const std::string &val) {
    _i_to_s[key] = val;
    _s_to_i[val] = key;
  }

  void add(const std::string &key, uint64_t val) {
    _s_to_i[key] = val;
    _i_to_s[val] = key;
  }

  bool empty() const { return _i_to_s.empty(); }

  size_t count(uint64_t i) const { return _i_to_s.count(i); }

  size_t count(const std::string &s) const { return _s_to_i.count(s); }

  std::string at(uint64_t i) const { return _i_to_s.at(i); }

  uint64_t at(std::string s) const { return _s_to_i.at(s); }

  std::map<uint64_t, std::string>::const_iterator find(uint64_t key) const {
    return _i_to_s.find(key);
  }

  std::map<std::string, uint64_t>::const_iterator find(
      const std::string &key) const {
    return _s_to_i.find(key);
  }

  std::map<std::string, uint64_t>::const_iterator s_begin() const {
    return _s_to_i.begin();
  }

  std::map<std::string, uint64_t>::const_iterator s_end() const {
    return _s_to_i.end();
  }

  std::map<uint64_t, std::string>::const_iterator i_begin() const {
    return _i_to_s.begin();
  }

  std::map<uint64_t, std::string>::const_iterator i_end() const {
    return _i_to_s.end();
  }

  size_t size() const {
    // size should be same, but just in case.
    if (_i_to_s.size() == _s_to_i.size()) {
      return _i_to_s.size();
    }

    return 0;
  }

  std::map<uint64_t, std::string> _i_to_s;  // index -> string
  std::map<std::string, uint64_t> _s_to_i;  // string -> index
};

// timeSamples in USD
// TODO: AttributeBlock support
template <typename T>
struct AnimationSample {
  float t{0.0};  // time is represented as float
  T value;
};

enum class VertexVariability {
  Constant,  // one value for all geometric elements
  Uniform,   // one value for each geometric elements(e.g. `face`, `UV patch`)
  Varying,   // per-vertex for each geometric elements. Bilinear interpolation.
  Vertex,  // Equvalent to `Varying` for Polygon mesh. The basis function of the
           // surface is used for the interpolation(Curves, Subdivision Surface,
           // etc).
  FaceVarying,  // per-Vertex per face. Bilinear interpolation.
  Indexed,      // Dedicated index buffer provided(unflattened Indexed Primvar).
};

std::string to_string(VertexVariability variability);

enum class NodeType {
  Xform,
  Mesh,  // Polygon mesh
  Camera,
  Skeleton, // SkelHierarchy
  PointLight,
  DirectionalLight,
  EnvmapLight, // DomeLight in USD
  // TODO(more lights)...
};

std::string to_string(NodeType ntype);

enum class ComponentType {
  UInt8,
  Int8,
  UInt16,
  Int16,
  UInt32,
  Int32,
  Half,
  Float,
  Double,
};

std::string to_string(ComponentType ty);

// glTF-like BufferData
struct BufferData {
  ComponentType componentType{ComponentType::UInt8};
  //uint8_t count{1};           // # of components. up to 256
  std::vector<uint8_t> data;  // binary data. size is dividable by sizeof(componentType)

  // TODO: Stride?
};

// Compound of ComponentType x component
enum class VertexAttributeFormat {
  Bool,     // bool(1 byte)
  Char,     // int8
  Char2,    // int8x2
  Char3,    // int8x3
  Char4,    // int8x4
  Byte,     // uint8
  Byte2,    // uint8x2
  Byte3,    // uint8x3
  Byte4,    // uint8x4
  Short,    // int16
  Short2,   // int16x2
  Short3,   // int16x2
  Short4,   // int16x2
  Ushort,   // uint16
  Ushort2,  // uint16x2
  Ushort3,  // uint16x2
  Ushort4,  // uint16x2
  Half,     // half
  Half2,    // half2
  Half3,    // half3
  Half4,    // half4
  Float,    // float
  Vec2,     // float2
  Vec3,     // float3
  Vec4,     // float4
  Int,      // int
  Ivec2,    // int2
  Ivec3,    // int3
  Ivec4,    // int4
  Uint,     // uint
  Uvec2,    // uint2
  Uvec3,    // uint3
  Uvec4,    // uint4
  Double,   // double
  Dvec2,    // double2
  Dvec3,    // double3
  Dvec4,    // double4
  Mat2,     // float 2x2
  Mat3,     // float 3x3
  Mat4,     // float 4x4
  Dmat2,    // double 2x2
  Dmat3,    // double 3x3
  Dmat4,    // double 4x4
};

static size_t VertexAttributeFormatSize(VertexAttributeFormat f) {
  size_t elemsize{0};

  switch (f) {
    case VertexAttributeFormat::Bool: {
      elemsize = 1;
      break;
    }
    case VertexAttributeFormat::Char: {
      elemsize = 1;
      break;
    }
    case VertexAttributeFormat::Char2: {
      elemsize = 2;
      break;
    }
    case VertexAttributeFormat::Char3: {
      elemsize = 3;
      break;
    }
    case VertexAttributeFormat::Char4: {
      elemsize = 4;
      break;
    }
    case VertexAttributeFormat::Byte: {
      elemsize = 1;
      break;
    }
    case VertexAttributeFormat::Byte2: {
      elemsize = 2;
      break;
    }
    case VertexAttributeFormat::Byte3: {
      elemsize = 3;
      break;
    }
    case VertexAttributeFormat::Byte4: {
      elemsize = 4;
      break;
    }
    case VertexAttributeFormat::Short: {
      elemsize = 2;
      break;
    }
    case VertexAttributeFormat::Short2: {
      elemsize = 4;
      break;
    }
    case VertexAttributeFormat::Short3: {
      elemsize = 6;
      break;
    }
    case VertexAttributeFormat::Short4: {
      elemsize = 8;
      break;
    }
    case VertexAttributeFormat::Ushort: {
      elemsize = 2;
      break;
    }
    case VertexAttributeFormat::Ushort2: {
      elemsize = 4;
      break;
    }
    case VertexAttributeFormat::Ushort3: {
      elemsize = 6;
      break;
    }
    case VertexAttributeFormat::Ushort4: {
      elemsize = 8;
      break;
    }
    case VertexAttributeFormat::Half: {
      elemsize = 2;
      break;
    }
    case VertexAttributeFormat::Half2: {
      elemsize = 4;
      break;
    }
    case VertexAttributeFormat::Half3: {
      elemsize = 6;
      break;
    }
    case VertexAttributeFormat::Half4: {
      elemsize = 8;
      break;
    }
    case VertexAttributeFormat::Mat2: {
      elemsize = 4 * 4;
      break;
    }
    case VertexAttributeFormat::Mat3: {
      elemsize = 4 * 9;
      break;
    }
    case VertexAttributeFormat::Mat4: {
      elemsize = 4 * 16;
      break;
    }
    case VertexAttributeFormat::Dmat2: {
      elemsize = 8 * 4;
      break;
    }
    case VertexAttributeFormat::Dmat3: {
      elemsize = 8 * 9;
      break;
    }
    case VertexAttributeFormat::Dmat4: {
      elemsize = 8 * 16;
      break;
    }
    case VertexAttributeFormat::Float: {
      elemsize = 4;
      break;
    }
    case VertexAttributeFormat::Vec2: {
      elemsize = sizeof(float) * 2;
      break;
    }
    case VertexAttributeFormat::Vec3: {
      elemsize = sizeof(float) * 3;
      break;
    }
    case VertexAttributeFormat::Vec4: {
      elemsize = sizeof(float) * 4;
      break;
    }
    case VertexAttributeFormat::Int: {
      elemsize = 4;
      break;
    }
    case VertexAttributeFormat::Ivec2: {
      elemsize = sizeof(int) * 2;
      break;
    }
    case VertexAttributeFormat::Ivec3: {
      elemsize = sizeof(int) * 3;
      break;
    }
    case VertexAttributeFormat::Ivec4: {
      elemsize = sizeof(int) * 4;
      break;
    }
    case VertexAttributeFormat::Uint: {
      elemsize = 4;
      break;
    }
    case VertexAttributeFormat::Uvec2: {
      elemsize = sizeof(uint32_t) * 2;
      break;
    }
    case VertexAttributeFormat::Uvec3: {
      elemsize = sizeof(uint32_t) * 3;
      break;
    }
    case VertexAttributeFormat::Uvec4: {
      elemsize = sizeof(uint32_t) * 4;
      break;
    }
    case VertexAttributeFormat::Double: {
      elemsize = sizeof(double);
      break;
    }
    case VertexAttributeFormat::Dvec2: {
      elemsize = sizeof(double) * 2;
      break;
    }
    case VertexAttributeFormat::Dvec3: {
      elemsize = sizeof(double) * 3;
      break;
    }
    case VertexAttributeFormat::Dvec4: {
      elemsize = sizeof(double) * 4;
      break;
    }
  }

  return elemsize;
}

std::string to_string(VertexAttributeFormat f);

///
/// Vertex attribute array. Stores raw vertex attribute data.
///
/// arrayLength = elementSize * vertexCount
/// arrayBytes = formatSize * elementSize * vertexCount
///
/// Example:
///    positions(float3, elementSize=1, n=2): [1.0, 1.1, 1.2,  0.4, 0.3, 0.2]
///    skinWeights(float, elementSize=4, n=2): [1.0, 1.0, 1.0, 1.0,  0.5, 0.5,
///    0.5, 0.5]
///
struct VertexAttribute {
  std::string name;  // Attribute(primvar) name. Optional. Can be empty.
  VertexAttributeFormat format{VertexAttributeFormat::Vec3};
  uint32_t elementSize{1};  // `elementSize` in USD terminology(i.e. # of
                            // samples per vertex data)
  uint32_t stride{0};  //  We don't support packed(interleaved) vertex data, so
                       //  stride is usually sizeof(VertexAttributeFormat) *
                       //  elementSize. 0 = tightly packed.
  std::vector<uint8_t> data;  // raw binary data(TODO: Use Buffer ID?)
  std::vector<uint32_t>
      indices;  // Dedicated Index buffer. Set when variability == Indexed.
                // empty = Use externally provided vertex index buffer
  VertexVariability variability{VertexVariability::Vertex};
  uint64_t handle{0};  // Handle ID for Graphics API. 0 = invalid

  //
  // Returns the number of vertex items(excludes `elementSize`).
  //
  // We use compound type for the format.
  // For example, this returns 1 when the buffer is
  // composed of 3 floats and `format` is float3(in any elementSize >= 1).
  size_t vertex_count() const {
    if (stride != 0) {
      // TODO: return 0 when (data.size() % stride) != 0?
      return data.size() / stride;
    }

    size_t itemSize = stride_bytes();

    if ((data.size() % itemSize) != 0) {
      // data size mismatch
      return 0;
    }

    return data.size() / itemSize;
  }

  inline bool empty() const {
    return data.empty();
  }

  size_t num_bytes() const { return data.size(); }

  const void *buffer() const {
    return reinterpret_cast<const void *>(data.data());
  }

  void set_buffer(const uint8_t *addr, size_t n) {
    data.resize(n);
    memcpy(data.data(), addr, n);
  }

  const std::vector<uint8_t> &get_data() const { return data; }

  std::vector<uint8_t> &get_data() { return data; }

  //
  // Bytes for each vertex item.
  // Returns `formatSize * elementSize` when `stride` is 0.
  // Returns `stride` when `stride` is not zero.
  //
  size_t stride_bytes() const {
    if (stride != 0) {
      return stride;
    }

    return element_size() * VertexAttributeFormatSize(format);
  }

  size_t element_size() const { return elementSize; }

  size_t format_size() const { return VertexAttributeFormatSize(format); }

  bool is_constant() const {
    return (variability == VertexVariability::Constant);
  }

  bool is_uniform() const {
    return (variability == VertexVariability::Constant);
  }

  // includes 'varying'
  bool is_vertex() const {
    return (variability == VertexVariability::Vertex) ||
           (variability == VertexVariability::Varying);
  }

  bool is_facevarying() const {
    return (variability == VertexVariability::FaceVarying);
  }

  bool is_indexed() const { return variability == VertexVariability::Indexed; }
};

#if 0  // TODO: Implement
///
/// Flatten(expand by vertexCounts and vertexIndices) VertexAttribute.
///
/// @param[in] src Input VertexAttribute.
/// @param[in] faceVertexCounts Array of faceVertex counts.
/// @param[in] faceVertexIndices Array of faceVertex indices.
/// @param[out] dst flattened VertexAttribute data.
/// @param[out] itemCount # of vertex items = dst.size() / src.stride_bytes().
///
static bool FlattenVertexAttribute(
    const VertexAttribute &src,
    const std::vector<uint32_t> &faceVertexCounts,
    const std::vector<uint32_t> &faceVertexIndices,
    std::vector<uint8_t> &dst,
    size_t &itemCount);
#else

#if 0  // TODO: Implement
///
/// Convert variability of `src` VertexAttribute to "facevarying".
///
/// @param[in] src Input VertexAttribute.
/// @param[in] faceVertexCounts  # of vertex per face. When the size is empty
/// and faceVertexIndices is not empty, treat `faceVertexIndices` as
/// triangulated mesh indices.
/// @param[in] faceVertexIndices
/// @param[out] dst VertexAttribute with facevarying variability. `dst.vertex_count()` become `sum(faceVertexCounts)`
///
static bool ToFacevaringVertexAttribute(
    const VertexAttribute &src, VertexAttribute &dst,
    const std::vector<uint32_t> &faceVertexCounts,
    const std::vector<uint32_t> &faceVertexIndices);
#endif
#endif

//
// Convert PrimVar(type-erased value) at specified time to VertexAttribute
//
// Input Primvar's name, variability(interpolation) and elementSize are
// preserved. Use Primvar's underlying type to set the type of VertexAttribute.
// (example: 'color3f'(underlying type 'float3') -> Vec3)
//
// @param[in] pvar GeomPrimvar.
// @param[out] dst Output VertexAttribute.
// @param[out] err Error messsage. can be nullptr
// @param[in] t timecode
// @param[in] tinterp Interpolation for timesamples
//
// @return true upon success.
//
bool ToVertexAttribute(const GeomPrimvar &pvar, VertexAttribute &dst,
                       std::string *err,
                       const double t = value::TimeCode::Default(),
                       const value::TimeSampleInterpolationType tinterp =
                           value::TimeSampleInterpolationType::Linear);

enum class ColorSpace {
  sRGB,
  Lin_sRGB,     // Linear sRGB(D65)
  Rec709,
  Raw,        // Raw(physical quantity) value(e.g. normal maps, ao maps)
  Lin_ACEScg, // ACES CG colorspace(AP1. D50)
  OCIO,
  Lin_DisplayP3,   // colorSpace 'lin_displayp3'
  sRGB_DisplayP3,  // colorSpace 'srgb_displayp3'
  Custom,          // TODO: Custom colorspace
  Unknown,         // Unknown color space. 
};

std::string to_string(ColorSpace cs);

// Infer colorspace from token value.
bool InferColorSpace(const value::token &tok, ColorSpace *result);

struct TextureImage {
  std::string asset_identifier;  // (resolved) filename or asset identifier.

  ComponentType texelComponentType{
      ComponentType::UInt8};  // texel bit depth of `buffer_id`
  ComponentType assetTexelComponentType{
      ComponentType::UInt8};  // texel bit depth of UsdUVTexture asset

  ColorSpace colorSpace{ColorSpace::sRGB};  // color space of texel data.
  ColorSpace usdColorSpace{
      ColorSpace::sRGB};  // original color space info in UsdUVTexture(asset meta or sourceColorSpace attrib)

  int32_t width{-1};
  int32_t height{-1};
  int32_t channels{-1};  // e.g. 3 for RGB.
  int32_t miplevel{0};

  int64_t buffer_id{-1};  // index to buffer_id(texel data)

  bool decoded{false}; // true if texture data(buffer_id) is decoded. false if buffer_id contains raw image data(e.g. JPEG data)
  uint64_t handle{0};  // Handle ID for Graphics API. 0 = invalid
};

struct Cubemap
{
  // face id mapping(based on OpenGL)
  // https://www.khronos.org/opengl/wiki/Cubemap_Texture
  //
  // 0: +X (right)
  // 1: -X (left)
  // 2: +Y (top)
  // 3: -Y (bottom)
  // 4: +Z (back)
  // 5: -Z (front)

  // LoD of cubemap
  std::vector<std::array<TextureImage, 6>> faces_lod;
};

// Envmap lightsource
struct EnvmapLight
{
  enum class Coordinate {
    LatLong,  // "latlong"
    Angular,  // "angular"
    // MirroredBall, // TODO: "mirroredBall"
    Cubemap,  // TinyUSDZ Tydra specific.
  };

  std::string element_name;
  std::string abs_path;
  std::string display_name;

  double guideRadius{1.0e5};
  std::string asset_name; // 'inputs:texture:file'

  std::vector<TextureImage> texture_lod;

  // Utility
  bool to_cubemap(Cubemap &cubemap);

};

// glTF-lie animation data

// TOOD: Implement Animation sample resampler.

// In USD, timeSamples are linearly interpolated by default.
template <typename T>
struct AnimationSampler {
  nonstd::optional<T> static_value; // value at static time('default' time) if exist
  std::vector<AnimationSample<T>> samples;

  // No cubicSpline in USD
  enum class Interpolation {
    Linear,
    Step,  // Held in USD
  };

  Interpolation interpolation{Interpolation::Linear};
};

// We store animation data in AoS(array of structure) approach(glTF-like), i.e. animation channel is provided per joint, instead of
// SoA(structure of array) approach(USD SkelAnimation)
// TODO: Use VertexAttribute-like data structure
struct AnimationChannel {
  enum class ChannelType { Transform, Translation, Rotation, Scale, Weight };

  AnimationChannel() = default;

  AnimationChannel(ChannelType ty) : type(ty) {
  }

  ChannelType type;
  // The following AnimationSampler is filled depending on ChannelType.
  // Example: Rotation => Only `rotations` are filled.

  // Matrix precision is reduced to float-precision
  // NOTE: transform is not supported in glTF(you need to decompose transform
  // matrix into TRS)
  AnimationSampler<mat4> transforms;

  AnimationSampler<vec3> translations;
  AnimationSampler<quat> rotations;  // Rotation is represented as quaternions
  AnimationSampler<vec3> scales; // half-types are upcasted to float precision
  AnimationSampler<float> weights;

  //std::string joint_name; // joint name(UsdSkel::joints)
  //int64_t joint_id{-1};  // joint index in SkelHierarchy
};

// USD SkelAnimation
struct Animation {
  std::string prim_name; // Prim name(element name)
  std::string abs_path;  // Target USD Prim path
  std::string display_name;  // `displayName` prim meta

  // key = joint, value = (key: channel_type, value: channel_value)
  std::map<std::string, std::map<AnimationChannel::ChannelType, AnimationChannel>> channels_map;

  // For blendshapes
  // key = blendshape name, value = timesamped weights
  // TODO: in-between weight
  std::map<std::string, AnimationSampler<float>> blendshape_weights_map;
};

struct Node {
  std::string prim_name;     // Prim name(element name)
  std::string abs_path;      // Absolute prim path
  std::string display_name;  // `displayName` prim meta

  NodeType nodeType{NodeType::Xform};

  int32_t id{-1};  // Index to node content(e.g. meshes[id] when nodeTypes ==
                   // Mesh). -1 = no corresponding content exists for this node.

  std::vector<Node> children;

  // Every node have its transform at specified timecode.
  // `resetXform` is encoded in global matrix.
  value::matrix4d local_matrix;
  value::matrix4d global_matrix;  // = local_matrix * parent_matrix (USD use
                                  // row-major(pre-multiply))

  bool has_resetXform{false}; // true: When updating the transform of the node, need to reset parent's matrix to compute global matrix.

  bool is_identity_matrix() { return is_identity(local_matrix); }

  std::vector<AnimationChannel>
      node_animations;  // xform animations(timesamples)

  uint64_t handle{0};  // Handle ID for Graphics API. 0 = invalid
};

// BlendShape shape target.

struct InbetweenShapeTarget {
  std::vector<vec3> pointOffsets;
  std::vector<vec3> normalOffsets;
  float weight{0.5f};  // TODO: Init with invalid weight?
};

struct ShapeTarget {
  std::string prim_name;     // Prim name
  std::string abs_path;      // Absolute prim path
  std::string display_name;  // `displayName` prim meta

  std::vector<uint32_t> pointIndices;
  std::vector<vec3> pointOffsets;
  std::vector<vec3> normalOffsets;

  // key = weight
  std::unordered_map<float, InbetweenShapeTarget> inbetweens;
};

struct JointAndWeight {
  value::matrix4d geomBindTransform{
      value::matrix4d::identity()};  // matrix4d primvars:skel:geomBindTransform

  //
  // NOTE: variability of jointIndices and jointWeights are 'vertex'
  // NOTE: Values in jointIndices and jointWeights will be reordered when `MeshConverterConfig::build_vertex_indices` is set true.
  //
  std::vector<int> jointIndices;  // int[] primvars:skel:jointIndices

  // NOTE: weight is converted from USD as-is. not normalized.
  std::vector<float> jointWeights;  // float[] primvars:skel:jointWeight;

  int elementSize{1}; // # of weights per vertex
};

struct MaterialPath {
  std::string material_path;           // USD Material Prim path.
  std::string backface_material_path;  // USD Material Prim path.

  // Default RenderMaterial Id to assign when
  // material_path/backface_material_path is empty. -1 = no material will be
  // assigned.
  int default_material_id{-1};
  int default_backface_material_id{-1};

  // primvar name used for texcoords when default RenderMaterial is used.
  // Currently we don't support different texcoord for each frontface and
  // backface material.
  std::string default_texcoords_primvar_name{"st"};
};

// GeomSubset whose familyName is 'materialBind'.
// For per-face material mapping.
struct MaterialSubset {
  std::string prim_name;     // Prim name in Stage
  std::string abs_path;      // Absolute Prim path in Stage
  std::string display_name;  // `displayName` Prim meta
  int64_t prim_index{-1};    // Prim index in Stage

  // Index to RenderScene::materials
  int material_id{-1};
  int backface_material_id{-1};

  // USD GeomSubset.indices. Index to a facet, i.e. index to GeomMesh.faceVertexCounts[], in USD GeomSubset
  std::vector<int> usdIndices;

  // Triangulated indices. Filled when `MeshConverterConfig::triangualte` is true
  std::vector<int> triangulatedIndices;

  const std::vector<int> &indices() const {
    return triangulatedIndices.size() ? triangulatedIndices : usdIndices;
  }

};

// Currently normals and texcoords are converted as facevarying attribute.
struct RenderMesh {
#if 0 // deprecated.
  //
  // Type of Vertex attributes of this mesh.
  //
  // `Indexed` preferred. `Facevarying` as the last resport.
  //
  enum class VertexArrayType {
    Indexed,  // 'vertex'-varying. i.e, use faceVertexIndices to draw mesh. All
              // vertex attributes must be representatable by single
              // indices(i.e, no `facevertex`-varying attribute)
    Facevarying,  // 'facevertx'-varying. When any of mesh attribute has
                  // 'facevertex' varying, we cannot represent the mesh with
                  // single indices, so decompose all vertex attribute to
                  // Facevaring(no VertexArray indices). This would impact
                  // rendering performance.
  };
#endif

  std::string prim_name;     // Prim name
  std::string abs_path;      // Absolute Prim path in Stage
  std::string display_name;  // `displayName` Prim metadataum

  // true: all vertex attributes are 'vertex'-varying. i.e, an App can simply use faceVertexIndices to draw mesh.
  // false: some vertex attributes are 'facevarying'-varying. An app need to decompose 'points' and 'vertex'-varying attribute to 'facevarying' variability to draw a mesh.
  bool is_single_indexable{false};

  //VertexArrayType vertexArrayType{VertexArrayType::Facevarying};

  std::vector<vec3> points;  // varying is always 'vertex'.

  ///
  /// Initialized with USD faceVertexIndices/faceVertexCounts in GeomMesh.
  /// When the mesh is triangulated, these attribute does not change.
  ///
  /// But will be modified when `MeshConverterCondig::build_vertex_indices` is set to true
  /// (To make vertex attributes of the mesh single-indexable)
  ///
  ///
  std::vector<uint32_t> usdFaceVertexIndices;
  std::vector<uint32_t> usdFaceVertexCounts;

  ///
  /// Triangulated faceVertexIndices, faceVerteCounts and auxiality state
  /// required to triangulate primvars in the app.
  ///
  /// trinangulated*** variables will be empty when the mesh is not
  /// triangulated.
  ///
  /// Topology could be changed(modified) when `MeshConverterCondig::build_vertex_indices` is set to true.
  ///
  std::vector<uint32_t> triangulatedFaceVertexIndices;
  std::vector<uint32_t> triangulatedFaceVertexCounts;

  std::vector<size_t>
      triangulatedToOrigFaceVertexIndexMap;  // used for rearrange facevertex
                                             // attrib
  std::vector<uint32_t>
      triangulatedFaceCounts;  // used for rearrange face indices(e.g GeomSubset
                               // indices)

  const std::vector<uint32_t> &faceVertexIndices() const {
    return is_triangulated() ? triangulatedFaceVertexIndices : usdFaceVertexIndices;
  }

  const std::vector<uint32_t> &faceVertexCounts() const {
    return is_triangulated() ? triangulatedFaceVertexCounts : usdFaceVertexCounts;
  }

  bool is_triangulated() const {
    return triangulatedFaceVertexIndices.size() && triangulatedFaceVertexCounts.size();
  }

  // `normals` or `primvar:normals`. Empty when no normals exist in the
  // GeomMesh.
  VertexAttribute normals;

  // key = slot ID. Usually 0 = primary
  std::unordered_map<uint32_t, VertexAttribute> texcoords;
  StringAndIdMap texcoordSlotIdMap;  // st primvarname to slotID map

  //
  // tangents and binormals(single-frame only)
  //
  // When `normals`(or `normals` primvar) is not present in the GeomMesh,
  // tangents and normals are not computed.
  //
  // When `normals` is supplied, but neither `tangents` nor `binormals` are
  // supplied in primvars, Tydra computes it based on:
  // https://learnopengl.com/Advanced-Lighting/Normal-Mapping (when
  // MeshConverterConfig::compute_tangents_and_binormals is set to `true`)
  //
  // For UsdPreviewSurface, geom primvar name of `tangents` and `binormals` are
  // read from Material's inputs::frame:tangentsPrimvarName(default "tangents"),
  // inputs::frame::binormalsPrimvarName(default "binormals")
  // https://learnopengl.com/Advanced-Lighting/Normal-Mapping
  //
  VertexAttribute tangents;
  VertexAttribute binormals;

  bool doubleSided{false};  // false = backface-cull.
  value::color3f displayColor{
      0.18f, 0.18f,
      0.18f};  // displayColor primvar(The number of array elements = 1) in USD.
               // default is set to the same in UsdPreviewSurface::diffuseColor
  float displayOpacity{
      1.0};  // displayOpacity primvar(The number of array elements = 1) in USD
  bool is_rightHanded{true};  // orientation attribute in USD.

  VertexAttribute
      vertex_colors;  // vertex color(displayColor primvar in USD). vec3.
  VertexAttribute
      vertex_opacities;  // opacity(alpha) component of vertex
                         // color(displayOpacity primvar in USD). float

  // For vertex skinning
  JointAndWeight joint_and_weights;
  int skel_id{-1}; // index to RenderScene::skeletons

  // BlendShapes
  // key = USD BlendShape prim name.
  std::map<std::string, ShapeTarget> targets;

  // Index to RenderScene::materials
  int material_id{-1};  // Material applied to whole faces in the mesh. per-face
                        // material by GeomSubset is stored in
                        // `material_subsetMap`
  int backface_material_id{
      -1};  // Backface material. Look up `rel material:binding:<BACKFACENAME>`
            // in GeomMesh. BACKFACENAME is a user-supplied setting. Default =
            // MaterialConverterConfig::default_backface_material_purpose_name

  // Key = GeomSubset name
  std::map<std::string, MaterialSubset>
      material_subsetMap;  // GeomSubset whose famiyName is 'materialBind'

  // If you want to access user-defined primvars or custom property,
  // Plese look into corresponding Prim( stage::find_prim_at_path(abs_path) )

  uint64_t handle{0};  // Handle ID for Graphics API. 0 = invalid
};

enum class UVReaderFloatComponentType {
  COMPONENT_FLOAT,
  COMPONENT_FLOAT2,
  COMPONENT_FLOAT3,
  COMPONENT_FLOAT4,
};

std::string to_string(UVReaderFloatComponentType ty);

// TODO: Deprecate UVReaderFloat.
// float, float2, float3 or float4 only
struct UVReaderFloat {
  UVReaderFloatComponentType componentType{
      UVReaderFloatComponentType::COMPONENT_FLOAT2};
  int64_t mesh_id{-1};   // index to RenderMesh
  int64_t coord_id{-1};  // index to RenderMesh::facevaryingTexcoords

#if 0
  // Returns interpolated UV coordinate with UV transform
  // # of components filled are equal to `componentType`.
  vec4 fetchUV(size_t faceId, float varyu, float varyv);
#endif
};

struct UVTexture {
  // NOTE: it looks no 'rgba' in UsdUvTexture
  enum class Channel { R, G, B, A, RGB, RGBA };

  std::string prim_name; // element Prim name
  std::string abs_path; // Absolute Prim path
  std::string display_name; // displayName prim metadatum

  // TextureWrap `black` in UsdUVTexture is mapped to `CLAMP_TO_BORDER`(app must
  // set border color to black) default is CLAMP_TO_EDGE and `useMetadata` wrap
  // mode is ignored.
  enum class WrapMode { CLAMP_TO_EDGE, REPEAT, MIRROR, CLAMP_TO_BORDER };

  WrapMode wrapS{WrapMode::CLAMP_TO_EDGE};
  WrapMode wrapT{WrapMode::CLAMP_TO_EDGE};

  // Do CPU texture mapping. For baking texels with transform, texturing in
  // raytracer(bake lighting), etc.
  //
  // This method accounts for `tranform` and `bias/scale`
  //
  // NOTE: for R, G, B channel, The value is replicated to output[0], output[1]
  // and output[2]. For A channel, The value is returned to output[3]
  vec4 fetch_uv(size_t faceId, float varyu, float varyv);

  // `fetch_uv` with user-specified channel. `outputChannel` is ignored.
  vec4 fetch_uv_channel(size_t faceId, float varyu, float varyv,
                        Channel channel);

  // UVW version of `fetch_uv`.
  vec4 fetch_uvw(size_t faceId, float varyu, float varyv, float varyw);
  vec4 fetch_uvw_channel(size_t faceId, float varyu, float varyv, float varyw,
                         Channel channel);

  // Connected output channel(determined by connectionPath in UsdPreviewSurface)
  Channel connectedOutputChannel{Channel::RGB};

  std::set<Channel> authoredOutputChannels; // Authored `output:***` attribute in UsdUVTexture

  // bias and scale for texel value
  vec4 bias{0.0f, 0.0f, 0.0f, 0.0f};
  vec4 scale{1.0f, 1.0f, 1.0f, 1.0f};

  UVReaderFloat uvreader;
  vec4 fallback_uv{0.0f, 0.0f, 0.0f, 0.0f};

  // UsdTransform2d
  // https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_texture_transform
  // = scale * rotate + translation
  bool has_transform2d{false};  // true = `transform`, `tx_rotation`, `tx_scale`
                                // and `tx_translation` are filled;
  mat3 transform{value::matrix3f::identity()};

  // raw transform2d value
  float tx_rotation{0.0f};
  vec2 tx_scale{1.0f, 1.0f};
  vec2 tx_translation{0.0f, 0.0f};

  // UV primvars name(UsdPrimvarReader's inputs:varname)
  std::string varname_uv;

  int64_t texture_image_id{-1};  // Index to TextureImage
  uint64_t handle{0};            // Handle ID for Graphics API. 0 = invalid
};

std::string to_string(UVTexture::WrapMode ty);

struct UDIMTexture {
  enum class Channel { R, G, B, RGB, RGBA };

  std::string prim_name; // element Prim name
  std::string abs_path; // Absolute Prim path
  std::string display_name; // displayName prim metadatum

  // NOTE: for single channel(e.g. R) fetch, Only [0] will be filled for the
  // return value.
  vec4 fetch(size_t faceId, float varyu, float varyv, float varyw = 1.0f,
             Channel channel = Channel::RGB);

  // key = UDIM id(e.g. 1001)
  std::unordered_map<uint32_t, int32_t> imageTileIds;
};

// workaround for GCC
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

// T or TextureId
template <typename T>
class ShaderParam {
 public:
  ShaderParam() = default;
  ShaderParam(const T &t) : value(t) { }

  bool is_texture() const { return texture_id >= 0; }

  template <typename STy>
  void set_value(const STy &val) {
    // Currently we assume T == Sty.
    // TODO: support more type variant
    static_assert(value::TypeTraits<T>::underlying_type_id() ==
                      value::TypeTraits<STy>::underlying_type_id(),
                  "");
    static_assert(sizeof(T) >= sizeof(STy), "");
    memcpy(&value, &val, sizeof(T));
  }

 //private:
  T value{};
  int32_t texture_id{-1};  // negative = invalid
};

// UsdPreviewSurface
class PreviewSurfaceShader {
 public:
  bool useSpecularWorkflow{false};

  ShaderParam<vec3> diffuseColor{{0.18f, 0.18f, 0.18f}};
  ShaderParam<vec3> emissiveColor{{0.0f, 0.0f, 0.0f}};
  ShaderParam<vec3> specularColor{{0.0f, 0.0f, 0.0f}};
  ShaderParam<float> metallic{0.0f};
  ShaderParam<float> roughness{0.5f};
  ShaderParam<float> clearcoat{0.0f};
  ShaderParam<float> clearcoatRoughness{0.01f};
  ShaderParam<float> opacity{1.0f};
  ShaderParam<float> opacityThreshold{0.0f};
  ShaderParam<float> ior{1.5f};
  ShaderParam<vec3> normal{{0.0f, 0.0f, 1.0f}};
  ShaderParam<float> displacement{0.0f};
  ShaderParam<float> occlusion{0.0f};

  uint64_t handle{0};  // Handle ID for Graphics API. 0 = invalid
};

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

// Material + Shader
struct RenderMaterial {
  std::string name;  // elementName in USD (e.g. "pbrMat")
  std::string
      abs_path;  // abosolute Prim path in USD (e.g. "/_material/scope/pbrMat")
  std::string display_name;

  PreviewSurfaceShader surfaceShader;
  // TODO: displacement, volume.

  uint64_t handle{0};  // Handle ID for Graphics API. 0 = invalid
};

// Simple Camera
//
// https://openusd.org/dev/api/class_usd_geom_camera.html
//
// NOTE: Node's matrix is used for Camera matrix
// NOTE: "Y up" coordinate, right-handed coordinate space in USD.
// NOTE: Unit uses tenths of a scene unit(i.e. [mm] by default).
//       RenderSceneConverter adjusts property value to [mm] accounting for Stage's unitsPerMeter
struct RenderCamera {

  std::string name;  // elementName in USD (e.g. "frontCamera")
  std::string
      abs_path;  // abosolute GeomCamera Prim path in USD (e.g. "/xform/camera")
  std::string display_name;

  float znear{0.1f}; // clippingRange[0]
  float zfar{1000000.0f}; // clippingRange[1]
  float verticalAspectRatio{1.0}; // vertical aspect ratio

  // for Ortho camera
  float xmag{1.0f}; // horizontal maginification
  float ymag{1.0f}; // vertical maginification

  float focalLength{50.0f}; // EFL(Effective Focal Length). [mm]
  float verticalAperture{15.2908f}; // [mm]
  float horizontalAperture{20.965f}; // [mm]

  // vertical FOV in radian
  inline float yfov() {
    return 2.0f * std::atan(0.5f * verticalAperture / focalLength);
  }

  // horizontal FOV in radian
  float xfov() {
    return 2.0f * std::atan(0.5f * horizontalAperture / focalLength);
  }

  GeomCamera::Projection projection{GeomCamera::Projection::Perspective};
  GeomCamera::StereoRole stereoRole{GeomCamera::StereoRole::Mono};

  double shutterOpen{0.0};
  double shutterClose{0.0};

};

// Simple light
struct RenderLight
{
  std::string name;  // elementName in USD (e.g. "frontCamera")
  std::string
      abs_path;  // abosolute GeomCamera Prim path in USD (e.g. "/xform/camera")


  // TODO..
};

struct SceneMetadata
{
  std::string copyright;
  std::string comment;

  std::string upAxis{"Y"}; // "X", "Y" or "Z"
  nonstd::optional<double> startTimeCode;
  nonstd::optional<double> endTimeCode;
  double framesPerSecond{24.0};
  double timeCodesPerSecond{24.0};
  double metersPerUnit{1.0}; // default [m]

  bool autoPlay{true};

  // If you want to lookup more thing on USD Stage Metadata, Use Stage::metas()
};

// Simple glTF-like Scene Graph
class RenderScene {
 public:
  std::string usd_filename;

  SceneMetadata meta;

  uint32_t default_root_node{0}; // index to `nodes`.

  std::vector<Node> nodes;
  std::vector<TextureImage> images;
  std::vector<RenderMaterial> materials;
  std::vector<RenderCamera> cameras;
  std::vector<RenderLight> lights;
  std::vector<UVTexture> textures;
  std::vector<RenderMesh> meshes;
  std::vector<Animation> animations;
  std::vector<SkelHierarchy> skeletons;
  std::vector<BufferData>
      buffers;  // Various data storage(e.g. texel/image data).

};

///
/// Texture image loader callback
///
/// The callback function should return TextureImage and Raw image data.
///
/// NOTE: TextureImage::buffer_id is filled in Tydra side after calling this
/// callback. NOTE: TextureImage::colorSpace will be overwritten if
/// `asset:sourceColorSpace` is authored in UsdUVTexture.
///
/// @param[in] asset Asset path
/// @param[in] assetInfo AssetInfo
/// @param[in] assetResolver AssetResolutionResolver context. Please pass
/// DefaultAssetResolutionResolver() if you don't have custom
/// AssetResolutionResolver.
/// @param[out] texImageOut TextureImage info.
/// @param[out] imageData Raw texture image data.
/// @param[inout] userdata User data.
/// @param[out] warn Optional. Warning message.
/// @param[out] error Optional. Error message.
///
/// @return true upon success.
/// termination of visiting Prims.
///
typedef bool (*TextureImageLoaderFunction)(
    const value::AssetPath &assetPath, const AssetInfo &assetInfo,
    const AssetResolutionResolver &assetResolver, TextureImage *imageOut,
    std::vector<uint8_t> *imageData, void *userdata, std::string *warn,
    std::string *err);

bool DefaultTextureImageLoaderFunction(const value::AssetPath &assetPath,
                                       const AssetInfo &assetInfo,
                                       const AssetResolutionResolver &assetResolver,
                                       TextureImage *imageOut,
                                       std::vector<uint8_t> *imageData,
                                       void *userdata, std::string *warn,
                                       std::string *err);

///
/// TODO: UDIM loder
///

struct MeshConverterConfig {
  bool triangulate{true};

  bool validate_geomsubset{true};  // Validate GeomSubset.

  // We may want texcoord data even if the Mesh does not have bound Material.
  // But we don't know which primvar is used as a texture coordinate when no
  // Texture assigned to the mesh(no PrimVar Reader assigned to) Use
  // UsdPreviewSurface setting for it.
  //
  // https://openusd.org/release/spec_usdpreviewsurface.html#usd-sample
  //
  // Also for tangnents/binormals.
  //
  // 'primvars' namespace is omitted.
  //
  std::string default_texcoords_primvar_name{"st"};
  std::string default_texcoords1_primvar_name{
      "st1"};  // for multi texture(available from iOS 16/macOS 13)
  std::string default_tangents_primvar_name{"tangents"};
  std::string default_binormals_primvar_name{"binormals"};

  // TODO: tangents1/binormals1 for multi-frame normal mapping?

  // Upperlimit of the number of skin weights per vertex.
  // For realtime app, usually up to 64
  uint32_t max_skin_elementSize = 1024ull * 256ull;

  //
  // Build vertex indices when vertex attributes are converted to `faceverying`?
  // Similar vertices are merged into single vertex index.
  // (convert vertex attributes from 'facevarying' to 'vertex' variability)
  //
  // Building indices is preferred for renderers which supports single
  // index-buffer only (e.g. OpenGL/Vulkan)
  //
  bool build_vertex_indices{true};

  //
  // Compute normals if not present in the mesh.
  // The algorithm computes smoothed normal for shared vertex.
  // Normals are also computed when `compute_tangents_and_binormals` is true
  // and normals primvar is not present in the mesh.
  //
  bool compute_normals{true};

  //
  // Compute tangents and binormals for tangent space normal mapping.
  // But when primary texcoords primvar is not present, tangents and binormals are not computed.
  //
  // NOTE: The algorithm is not robust to compute tangent/binormal for quad/polygons.
  // Set `triangulate` preferred when you want let Tydra compute tangent/binormal.
  //
  // NOTE: Computing tangent frame for multi-texcoord is not supported.
  //
  bool compute_tangents_and_binormals{true};

  //
  // Allowed relative error to check if vertex data is the same.
  // Used for 'facevarying' variability to `vertex` variability conversion in
  // ConvertMesh. Only effective to floating-point vertex data.
  //
  float facevarying_to_vertex_eps = std::numeric_limits<float>::epsilon();
};

struct MaterialConverterConfig {
  // purpose name for two-sided material mapping.
  // https://github.com/syoyo/tinyusdz/issues/120
  std::string default_backface_material_purpose_name{"back"};

  // DefaultTextureImageLoader will be used when nullptr;
  TextureImageLoaderFunction texture_image_loader_function{nullptr};
  void *texture_image_loader_function_userdata{nullptr};

  // For UsdUVTexture.
  //
  // Default configuration:
  //
  // - The converter converts 8bit texture to floating point image and texel
  // value is converted to linear space.
  // - Allow missing asset(texture) and asset load failure.
  //
  // Recommended configuration for mobile/WebGL
  //
  // - `preserve_texel_bitdepth` true
  //   - No floating-point image conversion.
  // - `linearize_color_space` true
  //   - Linearlize in CPU, and no sRGB -> Linear conversion in a shader
  //   required.

  // In the UsdUVTexture spec, 8bit texture image is converted to floating point
  // image of range `[0.0, 1.0]`. When this flag is set to false, 8bit and 16bit
  // texture image is converted to floating point image. When this flag is set
  // to true, 8bit and 16bit texture data is stored as-is to save memory.
  // Setting true is good if you want to render USD scene on mobile, WebGL, etc.
  bool preserve_texel_bitdepth{false};

  // Apply the inverse of a color space to make texture image in linear space.
  // When `preserve_texel_bitdepth` is set to true, linearization also preserse
  // texel bit depth (i.e, for 8bit sRGB image, 8bit linear-space image is
  // produced)
  bool linearize_color_space{false};

  //
  // Set scene(working space) colorspace. This space must be linear colorspace.
  // Possible choice is: Linear_sRGB(linear_srgb), Lin_ACEScg(ACEScg/AP1), Lin_DisplayP3(linear_displayp3)
  // W.I.P: Curently Lin_sRGB is only supported.
  //
  ColorSpace scene_color_space{ColorSpace::Lin_sRGB};

  // Allow asset(texture, shader, etc) path with Windows backslashes(e.g.
  // ".\textures\cat.png")? When true, convert it to forward slash('/') on
  // Posixish system(otherwise character is escaped(e.g. '\t' -> tab).
  bool allow_backslash_in_asset_path{true};

  // Allow texture load failure?
  bool allow_texture_load_failure{true};

  // Allow asset(e.g. texture file/shader file) which does not exit?
  bool allow_missing_asset{true};

};

struct RenderSceneConverterConfig {
  // Load texture image data on convert.
  // false: no actual texture file/asset access.
  // App/User must setup TextureImage manually after the conversion.
  bool load_texture_assets{true};
};

//
// Simple packed vertex struct & comparator for dedup.
// https://github.com/huamulan/OpenGL-tutorial/blob/master/common/vboindexer.cpp
//
// Up to 2 texcoords.
// tangent and binormal is included in VertexData, considering the situation
// that tangent and binormal is supplied through user-defined primvar.
//
// TODO: Use spatial hash for robust dedup(consider floating-point eps)
// TODO: Polish interface to support arbitrary vertex configuration.
//
struct DefaultPackedVertexData {
  //value::float3 position;
  uint32_t point_index;
  value::float3 normal;
  value::float2 uv0;
  value::float2 uv1;
  value::float3 tangent;
  value::float3 binormal;
  value::float3 color;
  float opacity;

  // comparator for std::map
  bool operator<(const DefaultPackedVertexData &rhs) const {
    return memcmp(reinterpret_cast<const void *>(this),
                  reinterpret_cast<const void *>(&rhs),
                  sizeof(DefaultPackedVertexData)) > 0;
  }
};

struct DefaultPackedVertexDataHasher {
  inline size_t operator()(const DefaultPackedVertexData &v) const {
    // Simple hasher using FNV1 32bit
    // TODO: Use 64bit FNV1?
    // TODO: Use spatial hash or LSH(LocallySensitiveHash) for position value.
    static constexpr uint32_t kFNV_Prime = 0x01000193;
    static constexpr uint32_t kFNV_Offset_Basis = 0x811c9dc5;

    const uint8_t *ptr = reinterpret_cast<const uint8_t *>(&v);
    size_t n = sizeof(DefaultPackedVertexData);

    uint32_t hash = kFNV_Offset_Basis;
    for (size_t i = 0; i < n; i++) {
      hash = (kFNV_Prime * hash) ^ (ptr[i]);
    }

    return size_t(hash);
  }
};

struct DefaultPackedVertexDataEqual {
  bool operator()(const DefaultPackedVertexData &lhs,
                  const DefaultPackedVertexData &rhs) const {
    return memcmp(reinterpret_cast<const void *>(&lhs),
                  reinterpret_cast<const void *>(&rhs),
                  sizeof(DefaultPackedVertexData)) == 0;
  }
};

template <class PackedVert>
struct DefaultVertexInput {
  //std::vector<value::float3> positions;
  std::vector<uint32_t> point_indices;
  std::vector<value::float3> normals;
  std::vector<value::float2> uv0s;
  std::vector<value::float2> uv1s;
  std::vector<value::float3> tangents;
  std::vector<value::float3> binormals;
  std::vector<value::float3> colors;
  std::vector<float> opacities;

  size_t size() const { return point_indices.size(); }

  void get(size_t idx, PackedVert &output) const {
    if (idx < point_indices.size()) {
      output.point_index = point_indices[idx];
    } else {
      output.point_index = ~0u; // this case should not happen though
    }
    if (idx < normals.size()) {
      output.normal = normals[idx];
    } else {
      output.normal = {0.0f, 0.0f, 0.0f};
    }
    if (idx < uv0s.size()) {
      output.uv0 = uv0s[idx];
    } else {
      output.uv0 = {0.0f, 0.0f};
    }
    if (idx < uv1s.size()) {
      output.uv1 = uv1s[idx];
    } else {
      output.uv1 = {0.0f, 0.0f};
    }
    if (idx < tangents.size()) {
      output.tangent = tangents[idx];
    } else {
      output.tangent = {0.0f, 0.0f, 0.0f};
    }
    if (idx < binormals.size()) {
      output.binormal = binormals[idx];
    } else {
      output.binormal = {0.0f, 0.0f, 0.0f};
    }
    if (idx < colors.size()) {
      output.color = colors[idx];
    } else {
      output.color = {0.0f, 0.0f, 0.0f};
    }
    if (idx < opacities.size()) {
      output.opacity = opacities[idx];
    } else {
      output.opacity = 0.0f;  // FIXME: Use 1.0?
    }
  }
};

template <class PackedVert>
struct DefaultVertexOutput {
  //std::vector<value::float3> positions;
  std::vector<uint32_t> point_indices;
  std::vector<value::float3> normals;
  std::vector<value::float2> uv0s;
  std::vector<value::float2> uv1s;
  std::vector<value::float3> tangents;
  std::vector<value::float3> binormals;
  std::vector<value::float3> colors;
  std::vector<float> opacities;

  size_t size() const { return point_indices.size(); }

  void push_back(const PackedVert &v) {
    point_indices.push_back(v.point_index);
    normals.push_back(v.normal);
    uv0s.push_back(v.uv0);
    uv1s.push_back(v.uv1);
    tangents.push_back(v.tangent);
    binormals.push_back(v.binormal);
    colors.push_back(v.color);
    opacities.push_back(v.opacity);
  }
};

//
// out_vertex_indices_remap: corresponding vertexIndex in input.
//
template <class VertexInput, class VertexOutput, class PackedVert,
          class PackedVertHasher, class PackedVertEqual>
void BuildIndices(const VertexInput &input, VertexOutput &output,
                  std::vector<uint32_t> &out_indices, std::vector<uint32_t> &out_point_indices)
{
  // TODO: Use LSH(locally sensitive hashing) or BVH for kNN point query.
  std::unordered_map<PackedVert, uint32_t, PackedVertHasher, PackedVertEqual>
      vertexToIndexMap;

  auto GetSimilarVertex = [&](const PackedVert &v, uint32_t &out_idx) -> bool {
    auto it = vertexToIndexMap.find(v);
    if (it == vertexToIndexMap.end()) {
      return false;
    }

    out_idx = it->second;
    return true;
  };

  for (size_t i = 0; i < input.size(); i++) {
    PackedVert v;
    input.get(i, v);

    uint32_t index{0};
    bool found = GetSimilarVertex(v, index);
    if (found) {
      out_indices.push_back(index);
    } else {
      uint32_t new_index = uint32_t(output.size());
      out_indices.push_back(new_index);
      output.push_back(v);
      vertexToIndexMap[v] = new_index;
    }
    out_point_indices.push_back(v.point_index);
  }
}

class RenderSceneConverterEnv {
 public:
  RenderSceneConverterEnv(const Stage &_stage) : stage(_stage) {}

  RenderSceneConverterConfig scene_config;
  MeshConverterConfig mesh_config;
  MaterialConverterConfig material_config;

  AssetResolutionResolver asset_resolver;

  std::string usd_filename; // Corresponding USD filename to Stage.

  void set_search_paths(const std::vector<std::string> &paths) {
    asset_resolver.set_search_paths(paths);
  }

  const Stage &stage;  // Point to valid Stage object at constructor

  double timecode{value::TimeCode::Default()};
  value::TimeSampleInterpolationType tinterp{
      value::TimeSampleInterpolationType::Linear};

};

//
// Convert USD scenegraph at specified time
// TODO: Use RenderSceneConverterEnv(RenderSceneConverterEnv::timecode)
//
class RenderSceneConverter {
 public:
  RenderSceneConverter() = default;
  RenderSceneConverter(const RenderSceneConverter &rhs) = delete;
  RenderSceneConverter(RenderSceneConverter &&rhs) = delete;

  ///
  /// All-in-one Stage to RenderScene conversion.
  ///
  /// Convert Stage to RenderScene.
  /// Must be called after SetStage, SetMaterialConverterConfig(optional)
  ///
  bool ConvertToRenderScene(const RenderSceneConverterEnv &env, RenderScene *scene);

  const std::string &GetInfo() const { return _info; }
  const std::string &GetWarning() const { return _warn; }
  const std::string &GetError() const { return _err; }

  // Prim path <-> index for corresponding array
  // e.g. meshMap: primPath/index to `meshes`.

  // TODO: Move to private?
  StringAndIdMap root_nodeMap;
  StringAndIdMap meshMap;
  StringAndIdMap materialMap;
  StringAndIdMap cameraMap;
  StringAndIdMap lightMap;
  StringAndIdMap textureMap;
  StringAndIdMap imageMap;
  StringAndIdMap bufferMap;
  StringAndIdMap animationMap;

  int default_node{-1};

  std::vector<Node> root_nodes;
  std::vector<RenderMesh> meshes;
  std::vector<RenderMaterial> materials;
  std::vector<RenderCamera> cameras;
  std::vector<RenderLight> lights;
  std::vector<UVTexture> textures;
  std::vector<TextureImage> images;
  std::vector<BufferData> buffers;
  std::vector<SkelHierarchy> skeletons;
  std::vector<Animation> animations;

  ///
  /// Convert GeomMesh to renderer-friendly mesh.
  /// Also apply triangulation when MeshConverterConfig::triangulate is set to
  /// true.
  ///
  /// normals, texcoords, vertexcolors/opacities vertex attributes(built-in
  /// primvars) are converterd to either `vertex` variability(i.e. can be drawn
  /// with single vertex indices) or `facevarying` variability(any of primvars
  /// is `facevarying`. It can be drawn with no indices, but less
  /// efficient(especially vertex has skin weights and blendshapes)).
  ///
  /// Since preferred variability for OpenGL/Vulkan renderer is `vertex`,
  /// ConvertMesh tries to convert `facevarying` attribute to `vertex` attribute
  /// when all shared vertex data is the same. If it fails, but
  /// `MeshConverterConfig.build_indices` is set to true, ConvertMesh builds
  /// vertex indices from `facevarying` and convert variability to 'vertex'.
  ///
  /// Note that `points`, skin weights and BlendShape attributes are remains
  /// with `vertex` variability. (so that we can apply some processing per
  /// point-wise)
  ///
  /// Thus, if you want to render a mesh whose normal/texcoord/etc variability
  /// is `facevarying`, `points`, skin weights and BlendShape attributes would
  /// also need to be converted to `facevarying` to draw.
  ///
  /// Other user defined primvars are not touched by ConvertMesh.
  /// The app need to manually triangulate, change variability of user-defined
  /// primvar if required.
  ///
  /// It is recommended first convert Materials assigned(bounded) to this
  /// GeomMesh(and GeomSubsets) or create your own Materials, and supply
  /// material info with `material_path` and `rmaterial_map`. You may supply
  /// empty material info and assign Material after ConvertMesh manually, but it
  /// will need some steps(Need to find texcoord primvar, triangulate texcoord,
  /// etc). See the implementation of ConvertMesh for details)
  ///
  ///
  /// @param[in] mesh_abs_path USD prim path to this GeomMesh
  /// @param[in] mesh Input GeomMesh
  /// @param[in] material_path USD Material Prim path assigned(bound) to this
  /// GeomMesh. Use tydra::GetBoundPath to get Material path actually assigned
  /// to the mesh.
  /// @param[in] subset_material_path_map USD Material Prim path assigned(bound)
  /// to GeomSubsets in this GeomMesh. key = GeomSubset Prim name.
  /// @param[in] rmaterial_map USD Material Prim path <-> RenderMaterial index
  /// map. Use empty map if no material assigned to this Mesh. If the mesh has
  /// bounded material(including material from GeomSubset), RenderMaterial index
  /// must be obrained using ConvertMaterial method before calling ConvertMesh.
  /// @param[in] material_subsets GeomSubset assigned to this Mesh. Can be empty
  /// when no materialBind GeomSuset assigned to this mesh.
  /// @param[in] blendshapes BlendShape Prims assigned to this Mesh. Can be
  /// empty when no BlendShape assigned to this mesh.
  /// @param[out] dst RenderMesh output
  ///
  /// @return true when success.
  ///
  ///
  bool ConvertMesh(
      const RenderSceneConverterEnv &env, const tinyusdz::Path &mesh_abs_path,
      const tinyusdz::GeomMesh &mesh, const MaterialPath &material_path,
      const std::map<std::string, MaterialPath> &subset_material_path_map,
      //const std::map<std::string, int64_t> &rmaterial_map,
      const StringAndIdMap &rmaterial_map,
      const std::vector<const tinyusdz::GeomSubset *> &material_subsets,
      const std::vector<std::pair<std::string, const tinyusdz::BlendShape *>>
          &blendshapes,
      RenderMesh *dst);

  ///
  /// Convert USD Material/Shader to renderer-friendly Material
  ///
  /// @return true when success.
  ///
  bool ConvertMaterial(const RenderSceneConverterEnv &env,
                       const tinyusdz::Path &abs_mat_path,
                       const tinyusdz::Material &material,
                       RenderMaterial *rmat_out);

  ///
  /// Convert UsdPreviewSurface Shader to renderer-friendly PreviewSurfaceShader
  ///
  /// @param[in] shader_abs_path USD Path to Shader Prim with UsdPreviewSurface
  /// info:id.
  /// @param[in] shader UsdPreviewSurface
  /// @param[in] pss_put PreviewSurfaceShader
  ///
  /// @return true when success.
  ///
  bool ConvertPreviewSurfaceShader(const RenderSceneConverterEnv &env,
                                   const tinyusdz::Path &shader_abs_path,
                                   const tinyusdz::UsdPreviewSurface &shader,
                                   PreviewSurfaceShader *pss_out);

  ///
  /// Convert UsdUvTexture to renderer-friendly UVTexture
  ///
  /// @param[in] tex_abs_path USD Path to Shader Prim with UsdUVTexture info:id.
  /// @param[in] assetInfo assetInfo Prim metadata of given Shader Prim
  /// @param[in] texture UsdUVTexture
  /// @param[in] tex_out UVTexture
  ///
  /// TODO: Retrieve assetInfo from `tex_abs_path`?
  ///
  /// @return true when success.
  ///
  bool ConvertUVTexture(const RenderSceneConverterEnv &env,
                        const Path &tex_abs_path, const AssetInfo &assetInfo,
                        const UsdUVTexture &texture, UVTexture *tex_out);

  ///
  /// Convert SkelAnimation to Tydra Animation.
  ///
  /// @param[in] abs_path USD Path to SkelAnimation Prim
  /// @param[in] skelAnim SkelAnimatio
  /// @param[in] anim_out Animation
  ///
  bool ConvertSkelAnimation(const RenderSceneConverterEnv &env,
                        const Path &abs_path, const SkelAnimation &skelAnim,
                        Animation *anim_out);

  ///
  /// @param[in] env
  /// @param[in] root XformNode
  ///
  bool BuildNodeHierarchy(const RenderSceneConverterEnv &env, const XformNode &node);

 private:
  ///
  /// Convert variability of vertex data to 'vertex' or 'facevarying'.
  ///
  /// @param[inout] vattr Input/Output VertexAttribute
  /// @param[in] to_vertex_varing Convert to `vertex` varying when true.
  /// `facevarying` when false.
  /// @param[in] faceVertexCounts faceVertexCounts
  /// @param[in] faceVertexIndices faceVertexIndices
  ///
  /// @return true upon success.
  ///
  bool ConvertVertexVariabilityImpl(
      VertexAttribute &vattr, const bool to_vertex_varying,
      const std::vector<uint32_t> &faceVertexCounts,
      const std::vector<uint32_t> &faceVertexIndices);

  template <typename T, typename Dty>
  bool ConvertPreviewSurfaceShaderParam(
      const RenderSceneConverterEnv &env, const Path &shader_abs_path,
      const TypedAttributeWithFallback<Animatable<T>> &param,
      const std::string &param_name, ShaderParam<Dty> &dst_param);

  ///
  /// Build (single) vertex indices for RenderMesh.
  /// existing `RenderMesh::faceVertexIndices` will be replaced with built indices.
  /// All vertex attributes are converted to 'vertex' variability.
  ///
  /// Limitation: Currently we only supports texcoords up to two(primary(0) and secondary(1)).
  ///
  /// @param[inout] mesh
  ///
  bool BuildVertexIndicesImpl(RenderMesh &mesh);

  //
  // Get Skeleton assigned to the GeomMesh Prim and convert it to SkelHierarchy.
  // Also get SkelAnimation attached to Skeleton(if exists)
  //
  bool ConvertSkeletonImpl(const RenderSceneConverterEnv &env, const tinyusdz::GeomMesh &mesh,
                       SkelHierarchy *out_skel, nonstd::optional<Animation> *out_anim);

  bool BuildNodeHierarchyImpl(
    const RenderSceneConverterEnv &env,
    const std::string &parentPrimPath,
    const XformNode &node,
    Node &out_rnode);

  void PushInfo(const std::string &msg) { _info += msg; }
  void PushWarn(const std::string &msg) { _warn += msg; }
  void PushError(const std::string &msg) { _err += msg; }

  std::string _info;
  std::string _err;
  std::string _warn;
};

// For debug
// Supported format: "kdl" (default. https://kdl.dev/), "json"
//
std::string DumpRenderScene(const RenderScene &scene,
                            const std::string &format = "kdl");

}  // namespace tydra
}  // namespace tinyusdz
