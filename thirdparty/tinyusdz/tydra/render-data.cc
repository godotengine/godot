// SPDX-License-Identifier: Apache 2.0
// Copyright 2022 - 2023, Syoyo Fujita.
// Copyright 2023 - Present, Light Transport Entertainment Inc.
//
// TODO:
//   - [ ] Subdivision surface to polygon mesh conversion.
//     - [ ] Correctly handle primvar with 'vertex' interpolation(Use the basis
//     function of subd surface)
//   - [x] Support time-varying shader attribute(timeSamples)
//   - [ ] Wide gamut colorspace conversion support
//     - [ ] linear sRGB <-> linear DisplayP3
//   - [x] Compute tangentes and binormals
//   - [x] displayColor, displayOpacity primvar(vertex color)
//   - [x] Support Skeleton
//   - [x] Support SkelAnimation
//     - [x] joint animation
//     - [x] blendshape animation
//     - [x] explicit joint order
//   - [ ] Support Inbetween BlendShape
//   - [ ] Support material binding collection(Collection API)
//   - [ ] Support multiple skel animation
//   https://github.com/PixarAnimationStudios/OpenUSD/issues/2246
//   - [ ] Adjust normal vector computation with handness?
//   - [ ] Node xform animation
//   - [ ] Better build of index buffer
//     - [ ] Preserve the order of 'points' variable(mesh.points, Skin
//     indices/weights, BlendShape points, ...) as much as possible.
//     - Implement spatial hash
//
#include <numeric>

#include "image-loader.hh"
#include "image-util.hh"
#include "image-types.hh"
#include "linear-algebra.hh"
#include "math-util.inc"
#include "pprinter.hh"
#include "prim-types.hh"
#include "str-util.hh"
#include "tiny-format.hh"
#include "tinyusdz.hh"
#include "usdGeom.hh"
#include "usdShade.hh"
#include "value-pprint.hh"

#if defined(TINYUSDZ_WITH_COLORIO)
#include "external/tiny-color-io.h"
#endif

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

// For tangent/binormal computation
// NOTE: HalfEdge is not used atm.
#include "external/half-edge.hh"

// For triangulation.
// TODO: Use tinyobjloader's triangulation
#include "external/mapbox/earcut/earcut.hpp"

// For kNN point search
// #include "external/nanoflann.hpp"

#ifdef __clang__
#pragma clang diagnostic pop
#endif

//
#include "common-macros.inc"
#include "math-util.inc"

//
#include "tydra/attribute-eval.hh"
#include "tydra/render-data.hh"
#include "tydra/scene-access.hh"
#include "tydra/shader-network.hh"

namespace tinyusdz {

namespace tydra {

namespace {

#define PushError(msg) \
  if (err) {           \
    (*err) += msg;     \
  }

inline std::string to_string(const UVTexture::Channel channel) {
  if (channel == UVTexture::Channel::RGB) {
    return "rgb";
  } else if (channel == UVTexture::Channel::R) {
    return "r";
  } else if (channel == UVTexture::Channel::G) {
    return "g";
  } else if (channel == UVTexture::Channel::B) {
    return "b";
  } else if (channel == UVTexture::Channel::A) {
    return "a";
  }

  return "[[InternalError. Invalid UVTexture::Channel]]";
}

//
// Convert vertex attribute with Uniform variability(interpolation) to
// facevarying variability, by replicating uniform value per face over face
// vertices.
//
#if 0  // unused atm
template <typename T>
nonstd::expected<std::vector<T>, std::string> UniformToFaceVarying(
    const std::vector<T> &inputs,
    const std::vector<uint32_t> &faceVertexCounts) {
  std::vector<T> dst;

  if (inputs.size() == faceVertexCounts.size()) {
    return nonstd::make_unexpected(
        fmt::format("The number of inputs {} must be the same with "
                    "faceVertexCounts.size() {}",
                    inputs.size(), faceVertexCounts.size()));
  }

  for (size_t i = 0; i < faceVertexCounts.size(); i++) {
    size_t cnt = faceVertexCounts[i];

    // repeat cnt times.
    for (size_t k = 0; k < cnt; k++) {
      dst.emplace_back(inputs[i]);
    }
  }

  return dst;
}
#endif

//
// Convert vertex attribute with Uniform variability(interpolation) to vertex
// variability, by replicating uniform value for vertices of a face. For shared
// vertex, the value will be overwritten.
//
#if 0  // unused atm
template <typename T>
nonstd::expected<std::vector<T>, std::string> UniformToVertex(
    const std::vector<T> &inputs, const size_t elementSize,
    const std::vector<uint32_t> &faceVertexCounts,
    const std::vector<uint32_t> &faceVertexIndices) {
  std::vector<T> dst;

  if (faceVertexIndices.size() < 3) {
    return nonstd::make_unexpected(
        fmt::format("faceVertexIndices.size must be 3 or greater, but got {}.",
                    faceVertexCounts.size()));
  }

  if (faceVertexCounts.empty()) {
    return nonstd::make_unexpected("faceVertexCounts.size is zero");
  }

  if (elementSize == 0) {
    return nonstd::make_unexpected("`elementSize` is zero.");
  }

  if ((inputs.size() % elementSize) != 0) {
    return nonstd::make_unexpected(
        fmt::format("input bytes {} must be dividable by elementSize {}.",
                    inputs.size(), elementSize));
  }

  size_t num_uniforms = faceVertexCounts.size();

  dst.resize(num_uniforms * elementSize);

  size_t fvIndexOffset{0};

  for (size_t i = 0; i < faceVertexCounts.size(); i++) {
    size_t cnt = faceVertexCounts[i];

    if ((fvIndexOffset + cnt) > faceVertexIndices.size()) {
      return nonstd::make_unexpected(
          fmt::format("faceVertexCounts[{}] {} gives buffer-overrun to "
                      "faceVertexIndices.size {}.",
                      i, cnt, faceVertexIndices.size()));
    }

    for (size_t k = 0; k < cnt; k++) {
      uint32_t v_idx = faceVertexIndices[fvIndexOffset + k];

      if (v_idx >= inputs.size()) {
        return nonstd::make_unexpected(
            fmt::format("vertexIndex {} is out-of-range for inputs.size {}.",
                        v_idx, inputs.size()));
      }

      // may overwrite the value
      memcpy(&dst[v_idx * elementSize], &inputs[i * elementSize],
             sizeof(T) * elementSize);
    }

    fvIndexOffset += cnt;
  }

  return dst;
}
#endif

nonstd::expected<std::vector<uint8_t>, std::string> UniformToVertex(
    const std::vector<uint8_t> &inputs, const size_t stride_bytes,
    const std::vector<uint32_t> &faceVertexCounts,
    const std::vector<uint32_t> &faceVertexIndices) {
  // NOTE: Uniform -> Vertex convertsion may give wrong result.
  std::vector<uint8_t> dst;

  if (stride_bytes == 0) {
    return nonstd::make_unexpected(fmt::format("stride_bytes is zero."));
  }

  if (faceVertexIndices.size() < 3) {
    return nonstd::make_unexpected(
        fmt::format("faceVertexIndices.size must be 3 or greater, but got {}.",
                    faceVertexCounts.size()));
  }

  if ((inputs.size() % stride_bytes) != 0) {
    return nonstd::make_unexpected(
        fmt::format("input bytes {} must be dividable by stride_bytes {}.",
                    inputs.size(), stride_bytes));
  }

  size_t num_uniforms = inputs.size() / stride_bytes;

  if (num_uniforms != faceVertexCounts.size()) {
    return nonstd::make_unexpected(fmt::format(
        "The number of input uniform attributes {} must be the same with "
        "faceVertexCounts.size() {}",
        num_uniforms, faceVertexCounts.size()));
  }

  const uint32_t num_vertices =
      *std::max_element(faceVertexIndices.cbegin(), faceVertexIndices.cend()) + 1;

  dst.resize(num_vertices * stride_bytes);

  size_t fvIndexOffset{0};

  for (size_t i = 0; i < faceVertexCounts.size(); i++) {
    size_t cnt = faceVertexCounts[i];

    if ((fvIndexOffset + cnt) > faceVertexIndices.size()) {
      return nonstd::make_unexpected(
          fmt::format("faceVertexCounts[{}] {} gives buffer-overrun to "
                      "faceVertexIndices.size {}.",
                      i, cnt, faceVertexIndices.size()));
    }

    for (size_t k = 0; k < cnt; k++) {
      uint32_t v_idx = faceVertexIndices[fvIndexOffset + k];

      // may overwrite the value when a vertex is referenced from multiple facet.
      memcpy(dst.data() + v_idx * stride_bytes,
             inputs.data() + i * stride_bytes, stride_bytes);
    }

    fvIndexOffset += cnt;
  }

  return dst;
}

// Generic uniform to facevarying conversion
nonstd::expected<std::vector<uint8_t>, std::string> UniformToFaceVarying(
    const std::vector<uint8_t> &src, const size_t stride_bytes,
    const std::vector<uint32_t> &faceVertexCounts) {
  std::vector<uint8_t> dst;

  if (stride_bytes == 0) {
    return nonstd::make_unexpected("stride_bytes is zero.");
  }

  if ((src.size() % stride_bytes) != 0) {
    return nonstd::make_unexpected(
        fmt::format("input bytes {} must be the multiple of stride_bytes {}",
                    src.size(), stride_bytes));
  }

  size_t num_uniforms = src.size() / stride_bytes;

  if (num_uniforms != faceVertexCounts.size()) {
    return nonstd::make_unexpected(fmt::format(
        "The number of input uniform attributes {} must be the same with "
        "faceVertexCounts.size() {}",
        num_uniforms, faceVertexCounts.size()));
  }

  std::vector<uint8_t> buf;
  buf.resize(stride_bytes);

  for (size_t i = 0; i < faceVertexCounts.size(); i++) {
    size_t cnt = faceVertexCounts[i];

    memcpy(buf.data(), src.data() + i * stride_bytes, stride_bytes);

    // repeat cnt times.
    for (size_t k = 0; k < cnt; k++) {
      dst.insert(dst.end(), buf.begin(), buf.end());
    }
  }

  return dst;
}

//
// Convert vertex attribute with Vertex variability(interpolation) to
// facevarying attribute, by expanding(flatten) the value per vertex per face.
//
#if 0  // unsued atm
template <typename T>
nonstd::expected<std::vector<T>, std::string> VertexToFaceVarying(
    const std::vector<T> &inputs, const std::vector<uint32_t> &faceVertexCounts,
    const std::vector<uint32_t> &faceVertexIndices) {
  std::vector<T> dst;

  size_t face_offset{0};
  for (size_t i = 0; i < faceVertexCounts.size(); i++) {
    size_t cnt = faceVertexCounts[i];

    for (size_t k = 0; k < cnt; k++) {
      size_t idx = k + face_offset;

      if (idx >= faceVertexIndices.size()) {
        return nonstd::make_unexpected(fmt::format(
            "faeVertexIndex out-of-range at faceVertexCount[{}]", i));
      }

      size_t v_idx = faceVertexIndices[idx];

      if (v_idx >= inputs.size()) {
        return nonstd::make_unexpected(
            fmt::format("faeVertexIndices[{}] {} exceeds input array size {}",
                        idx, v_idx, inputs.size()));
      }

      dst.emplace_back(inputs[v_idx]);
    }

    face_offset += cnt;
  }

  return dst;
}
#endif

// Generic vertex to facevarying conversion
nonstd::expected<std::vector<uint8_t>, std::string> VertexToFaceVarying(
    const std::vector<uint8_t> &src, const size_t stride_bytes,
    const std::vector<uint32_t> &faceVertexCounts,
    const std::vector<uint32_t> &faceVertexIndices) {
  std::vector<uint8_t> dst;

  if (src.empty()) {
    return nonstd::make_unexpected("src data is empty.");
  }

  if (stride_bytes == 0) {
    return nonstd::make_unexpected("stride_bytes must be non-zero.");
  }

  if ((src.size() % stride_bytes) != 0) {
    return nonstd::make_unexpected(
        fmt::format("src size {} must be the multiple of stride_bytes {}",
                    src.size(), stride_bytes));
  }

  const size_t num_vertices = src.size() / stride_bytes;

  std::vector<uint8_t> buf;
  buf.resize(stride_bytes);

  size_t faceVertexIndexOffset{0};

  for (size_t i = 0; i < faceVertexCounts.size(); i++) {
    size_t cnt = faceVertexCounts[i];

    for (size_t k = 0; k < cnt; k++) {
      size_t fv_idx = k + faceVertexIndexOffset;

      if (fv_idx >= faceVertexIndices.size()) {
        return nonstd::make_unexpected(
            fmt::format("faeVertexIndex {} out-of-range at faceVertexCount[{}]",
                        fv_idx, i));
      }

      size_t v_idx = faceVertexIndices[fv_idx];

      if (v_idx >= num_vertices) {
        return nonstd::make_unexpected(fmt::format(
            "faeVertexIndices[{}] {} exceeds the number of vertices {}", fv_idx,
            v_idx, num_vertices));
      }

      memcpy(buf.data(), src.data() + v_idx * stride_bytes, stride_bytes);
      dst.insert(dst.end(), buf.begin(), buf.end());
    }

    faceVertexIndexOffset += cnt;
  }

  return dst;
}

#if 0  // unused a.t.m
// Copy single value to facevarying vertices.
template <typename T>
static nonstd::expected<std::vector<T>, std::string> ConstantToFaceVarying(
    const T &input, const std::vector<uint32_t> &faceVertexCounts) {
  std::vector<T> dst;

  for (size_t i = 0; i < faceVertexCounts.size(); i++) {
    size_t cnt = faceVertexCounts[i];

    for (size_t k = 0; k < cnt; k++) {
      dst.emplace_back(input);
    }
  }

  return dst;
}
#endif

static nonstd::expected<std::vector<uint8_t>, std::string> ConstantToVertex(
    const std::vector<uint8_t> &src, const size_t stride_bytes,
    const std::vector<uint32_t> &faceVertexCounts,
    const std::vector<uint32_t> &faceVertexIndices) {
  if (faceVertexCounts.empty()) {
    return nonstd::make_unexpected("faceVertexCounts is empty.");
  }

  if (faceVertexIndices.size() < 3) {
    return nonstd::make_unexpected(
        fmt::format("faceVertexIndices.size must be at least 3, but got {}.",
                    faceVertexIndices.size()));
  }

  const uint32_t num_vertices =
      *std::max_element(faceVertexIndices.cbegin(), faceVertexIndices.cend()) + 1;

  std::vector<uint8_t> dst;

  if (src.empty()) {
    return nonstd::make_unexpected("src data is empty.");
  }

  if (stride_bytes == 0) {
    return nonstd::make_unexpected("stride_bytes must be non-zero.");
  }

  if (src.size() != stride_bytes) {
    return nonstd::make_unexpected(
        fmt::format("src size {} must be equal to stride_bytes {}", src.size(),
                    stride_bytes));
  }

  dst.resize(stride_bytes * num_vertices);

  size_t faceVertexIndexOffset = 0;
  for (size_t i = 0; i < faceVertexCounts.size(); i++) {
    uint32_t cnt = faceVertexCounts[i];
    if (cnt < 3) {
      return nonstd::make_unexpected(fmt::format(
          "faeVertexCounts[{}] must be equal to or greater than 3, but got {}",
          i, cnt));
    }

    for (size_t k = 0; k < cnt; k++) {
      size_t fv_idx = k + faceVertexIndexOffset;

      if (fv_idx >= faceVertexIndices.size()) {
        return nonstd::make_unexpected(
            fmt::format("faeVertexIndex {} out-of-range at faceVertexCount[{}]",
                        fv_idx, i));
      }

      size_t v_idx = faceVertexIndices[fv_idx];

      if (v_idx >= num_vertices) {  // this should not happen. just in case.
        return nonstd::make_unexpected(fmt::format(
            "faeVertexIndices[{}] {} exceeds the number of vertices {}", fv_idx,
            v_idx, num_vertices));
      }

      memcpy(dst.data() + v_idx * stride_bytes, src.data(), stride_bytes);
    }

    faceVertexIndexOffset += cnt;
  }

  return dst;
}

#if 0
static nonstd::expected<std::vector<uint8_t>, std::string>
ConstantToFaceVarying(const std::vector<uint8_t> &src,
                      const size_t stride_bytes,
                      const std::vector<uint32_t> &faceVertexCounts) {
  std::vector<uint8_t> dst;

  if (src.empty()) {
    return nonstd::make_unexpected("src data is empty.");
  }

  if (stride_bytes == 0) {
    return nonstd::make_unexpected("stride_bytes must be non-zero.");
  }

  if ((src.size() != stride_bytes)) {
    return nonstd::make_unexpected(
        fmt::format("src size {} must be equal to stride_bytes {}", src.size(),
                    stride_bytes));
  }

  std::vector<uint8_t> buf;
  buf.resize(stride_bytes);

  for (size_t i = 0; i < faceVertexCounts.size(); i++) {
    size_t cnt = faceVertexCounts[i];

    for (size_t k = 0; k < cnt; k++) {
      dst.insert(dst.end(), buf.begin(), buf.end());
    }
  }

  return dst;
}
#endif

// T = int
template <typename T>
bool TryConvertFacevaryingToVertexInt(
    const std::vector<T> &src, std::vector<T> *dst,
    const std::vector<uint32_t> &faceVertexIndices) {
  if (!dst) {
    return false;
  }

  if (src.size() != faceVertexIndices.size()) {
    return false;
  }

  // size must be at least 1 triangle(3 verts).
  if (faceVertexIndices.size() < 3) {
    return false;
  }

  // vidx, value
  std::unordered_map<uint32_t, T> vdata;

  uint32_t max_vidx = 0;
  for (size_t i = 0; i < faceVertexIndices.size(); i++) {
    uint32_t vidx = faceVertexIndices[i];
    max_vidx = (std::max)(vidx, max_vidx);

    if (vdata.count(vidx)) {
      if (!math::is_close(vdata[vidx], src[i])) {
        return false;
      }
    } else {
      vdata[vidx] = src[i];
    }
  }

  dst->resize(max_vidx + 1);
  memset(dst->data(), 0, (max_vidx + 1) * sizeof(T));

  for (const auto &v : vdata) {
    (*dst)[v.first] = v.second;
  }

  return true;
}

// T = float, double, float2, ...
template <typename T, typename EpsTy>
bool TryConvertFacevaryingToVertexFloat(
    const std::vector<T> &src, std::vector<T> *dst,
    const std::vector<uint32_t> &faceVertexIndices, const EpsTy eps) {
  DCOUT("TryConvertFacevaryingToVertexFloat");
  if (!dst) {
    return false;
  }

  if (src.size() != faceVertexIndices.size()) {
    DCOUT("size mismatch.");
    return false;
  }

  // size must be at least 1 triangle(3 verts).
  if (faceVertexIndices.size() < 3) {
    return false;
  }

  // vidx, value
  std::unordered_map<uint32_t, T> vdata;

  uint32_t max_vidx = 0;
  for (size_t i = 0; i < faceVertexIndices.size(); i++) {
    uint32_t vidx = faceVertexIndices[i];
    max_vidx = (std::max)(vidx, max_vidx);

    if (vdata.count(vidx)) {
      if (!math::is_close(vdata[vidx], src[i], eps)) {
        DCOUT("diff at faceVertexIndices[" << i << "]");
        return false;
      }
    } else {
      vdata[vidx] = src[i];
    }
  }

  dst->resize(max_vidx + 1);
  memset(dst->data(), 0, (max_vidx + 1) * sizeof(T));

  for (const auto &v : vdata) {
    (*dst)[v.first] = v.second;
  }

  return true;
}

// T = matrix type.
template <typename T>
bool TryConvertFacevaryingToVertexMat(
    const std::vector<T> &src, std::vector<T> *dst,
    const std::vector<uint32_t> &faceVertexIndices) {
  if (!dst) {
    return false;
  }

  if (src.size() != faceVertexIndices.size()) {
    return false;
  }

  // size must be at least 1 triangle(3 verts).
  if (faceVertexIndices.size() < 3) {
    return false;
  }

  // vidx, value
  std::unordered_map<uint32_t, T> vdata;

  uint32_t max_vidx = 0;
  for (size_t i = 0; i < faceVertexIndices.size(); i++) {
    uint32_t vidx = faceVertexIndices[i];
    max_vidx = (std::max)(vidx, max_vidx);

    if (vdata.count(vidx)) {
      if (!is_close(vdata[vidx], src[i])) {
        return false;
      }
    } else {
      vdata[vidx] = src[i];
    }
  }

  dst->assign(max_vidx + 1, T::identity());

  for (const auto &v : vdata) {
    (*dst)[v.first] = v.second;
  }

  return true;
}

///
/// Try to convert 'facevarying' vertex attribute to 'vertex' attribute.
/// Inspect each vertex value is the same(with given eps)
///
/// Current limitation:
/// - stride must be 0 or tightly packed.
/// - elementSize must be 1
///
/// @return true when 'facevarying' vertex attribute successfully converted to
/// 'vertex'
///
static bool TryConvertFacevaryingToVertex(
    const VertexAttribute &src, VertexAttribute *dst,
    const std::vector<uint32_t> &faceVertexIndices, std::string *err,
    const float eps) {
  DCOUT("TryConvertFacevaryingToVertex");
  if (!dst) {
    PUSH_ERROR_AND_RETURN("Output `dst` is nullptr.");
  }

  if (!src.is_facevarying()) {
    PUSH_ERROR_AND_RETURN("Input must be 'facevarying' attribute");
  }

  if (src.element_size() != 1) {
    PUSH_ERROR_AND_RETURN("Input's element_size must be 1.");
  }

  if ((src.stride != 0) && (src.stride_bytes() != src.format_size())) {
    PUSH_ERROR_AND_RETURN(
        "Input attribute must be tightly packed. stride_bytes = "
        << src.stride_bytes() << ", format_size = " << src.format_size());
  }

#define CONVERT_FUN_INT(__fmt, __ty)                                      \
  if (src.format == __fmt) {                                              \
    std::vector<__ty> vsrc;                                               \
    vsrc.resize(src.vertex_count());                                      \
    memcpy(vsrc.data(), src.get_data().data(), src.get_data().size());    \
    std::vector<__ty> vdst;                                               \
    bool ret = TryConvertFacevaryingToVertexInt<__ty>(vsrc, &vdst,        \
                                                      faceVertexIndices); \
    if (!ret) {                                                           \
      return false;                                                       \
    }                                                                     \
    dst->name = src.name;                                                 \
    dst->elementSize = 1;                                                 \
    dst->format = src.format;                                             \
    dst->variability = VertexVariability::Vertex;                         \
    dst->data.resize(vdst.size() * src.format_size());                    \
    memcpy(dst->data.data(), vdst.data(), dst->data.size());              \
    return true;                                                          \
  } else

#define CONVERT_FUN_FLOAT(__fmt, __ty, __epsty)                        \
  if (src.format == __fmt) {                                           \
    std::vector<__ty> vsrc;                                            \
    vsrc.resize(src.vertex_count());                                   \
    memcpy(vsrc.data(), src.get_data().data(), src.get_data().size()); \
    std::vector<__ty> vdst;                                            \
    bool ret = TryConvertFacevaryingToVertexFloat<__ty, __epsty>(      \
        vsrc, &vdst, faceVertexIndices, __epsty(eps));                 \
    if (!ret) {                                                        \
      return false;                                                    \
    }                                                                  \
    dst->name = src.name;                                                 \
    dst->elementSize = 1;                                              \
    dst->format = src.format;                                          \
    dst->variability = VertexVariability::Vertex;                      \
    dst->data.resize(vdst.size() * src.format_size());                 \
    memcpy(dst->data.data(), vdst.data(), dst->data.size());           \
    return true;                                                       \
  } else

#define CONVERT_FUN_MAT(__fmt, __ty)                                      \
  if (src.format == __fmt) {                                              \
    std::vector<__ty> vsrc;                                               \
    vsrc.resize(src.vertex_count());                                      \
    memcpy(vsrc.data(), src.get_data().data(), src.get_data().size());    \
    std::vector<__ty> vdst;                                               \
    bool ret = TryConvertFacevaryingToVertexMat<__ty>(vsrc, &vdst,        \
                                                      faceVertexIndices); \
    if (!ret) {                                                           \
      return false;                                                       \
    }                                                                     \
    dst->name = src.name;                                                 \
    dst->elementSize = 1;                                                 \
    dst->format = src.format;                                             \
    dst->variability = VertexVariability::Vertex;                         \
    dst->data.resize(vdst.size() * src.format_size());                    \
    memcpy(dst->data.data(), vdst.data(), dst->data.size());              \
    return true;                                                          \
  } else

  // NOTE: VertexAttributeFormat::Bool is preserved
  CONVERT_FUN_INT(VertexAttributeFormat::Bool, uint8_t)
  CONVERT_FUN_FLOAT(VertexAttributeFormat::Float, float, float)
  CONVERT_FUN_FLOAT(VertexAttributeFormat::Vec2, value::float2, float)
  CONVERT_FUN_FLOAT(VertexAttributeFormat::Vec3, value::float3, float)
  CONVERT_FUN_FLOAT(VertexAttributeFormat::Vec4, value::float4, float)
  CONVERT_FUN_INT(VertexAttributeFormat::Char, signed char)
  // CONVERT_FUN(VertexAttributeFormat::Char2, value::char2)
  // CONVERT_FUN(VertexAttributeFormat::Char3, value::char3)
  // CONVERT_FUN(VertexAttributeFormat::Char4,    // int8x4
  CONVERT_FUN_INT(VertexAttributeFormat::Byte, uint8_t)
  // CONVERT_FUN(VertexAttributeFormat::Byte2,    // uint8x2
  // CONVERT_FUN(VertexAttributeFormat::Byte3,    // uint8x3
  // CONVERT_FUN(VertexAttributeFormat::Byte4,    // uint8x4
  CONVERT_FUN_INT(VertexAttributeFormat::Short, int16_t)
  // CONVERT_FUN(VertexAttributeFormat::Short2, value::short2)
  // CONVERT_FUN(VertexAttributeFormat::Short3, value::short3)
  // CONVERT_FUN(VertexAttributeFormat::Short4, value::short4)
  CONVERT_FUN_INT(VertexAttributeFormat::Ushort, uint16_t)
  // CONVERT_FUN(VertexAttributeFormat::Ushort2, uint16_t)
  // CONVERT_FUN(VertexAttributeFormat::Ushort3, uint16_t)
  // CONVERT_FUN(VertexAttributeFormat::Ushort4, uint16_t)
  CONVERT_FUN_FLOAT(VertexAttributeFormat::Half, value::half, float)
  CONVERT_FUN_FLOAT(VertexAttributeFormat::Half2, value::half2, float)
  CONVERT_FUN_FLOAT(VertexAttributeFormat::Half3, value::half3, float)
  CONVERT_FUN_FLOAT(VertexAttributeFormat::Half4, value::half4, float)
  CONVERT_FUN_INT(VertexAttributeFormat::Int, int)
  CONVERT_FUN_INT(VertexAttributeFormat::Ivec2, value::int2)
  CONVERT_FUN_INT(VertexAttributeFormat::Ivec3, value::int3)
  CONVERT_FUN_INT(VertexAttributeFormat::Ivec4, value::int4)
  CONVERT_FUN_INT(VertexAttributeFormat::Uint, uint32_t)
  CONVERT_FUN_INT(VertexAttributeFormat::Uvec2, value::uint2)
  CONVERT_FUN_INT(VertexAttributeFormat::Uvec3, value::uint3)
  CONVERT_FUN_INT(VertexAttributeFormat::Uvec4, value::uint4)
  // NOTE: Use float precision eps is upcasted to double precision.
  CONVERT_FUN_FLOAT(VertexAttributeFormat::Double, double, double)
  CONVERT_FUN_FLOAT(VertexAttributeFormat::Dvec2, value::double2, double)
  CONVERT_FUN_FLOAT(VertexAttributeFormat::Dvec3, value::double3, double)
  CONVERT_FUN_FLOAT(VertexAttributeFormat::Dvec4, value::double4, double)
  CONVERT_FUN_MAT(VertexAttributeFormat::Mat2, value::matrix2f)
  CONVERT_FUN_MAT(VertexAttributeFormat::Mat3, value::matrix3f)
  CONVERT_FUN_MAT(VertexAttributeFormat::Mat4, value::matrix4f)
  CONVERT_FUN_MAT(VertexAttributeFormat::Dmat2, value::matrix2d)
  CONVERT_FUN_MAT(VertexAttributeFormat::Dmat3, value::matrix3d)
  CONVERT_FUN_MAT(VertexAttributeFormat::Dmat4, value::matrix4d) {
    if (err) {
      (*err) +=
          fmt::format("Unsupported/Unimplemented VertexAttributeFormat: {}",
                      to_string(src.format));
    }
  }

#undef CONVERT_FUN_INT
#undef CONVERT_FUN_FLOAT
#undef CONVERT_FUN_MAT

  return false;
}

#if 0  // Not used atm.
static bool ToFaceVaryingAttribute(const std::string &attr_name,
  const VertexAttribute &src,
  const std::vector<uint32_t> &faceVertexCounts,
  const std::vector<uint32_t> &faceVertexIndices,
  VertexAttribute *dst,
  std::string *err) {

#define PushError(msg) \
  if (err) {           \
    (*err) += msg;     \
  }

  if (!dst) {
    PUSH_ERROR_AND_RETURN("'dest' parameter is nullptr.");
  }

  if (src.variability == VertexVariability::Indexed) {
    PUSH_ERROR_AND_RETURN(fmt::format("'indexed' variability for {} is not supported.", attr_name));
  } else if (src.variability == VertexVariability::Constant) {

    auto result = ConstantToFaceVarying(src.get_data(), src.stride_bytes(),
            faceVertexCounts);

    if (!result) {
      PUSH_ERROR_AND_RETURN(fmt::format("Failed to convert vertex data with 'constant' variability to 'facevarying': name {}.", attr_name));
    }

    dst->data = result.value();
    dst->elementSize = src.elementSize;
    dst->format = src.format;
    dst->stride = src.stride;
    dst->variability = VertexVariability::FaceVarying;

    return true;

  } else if (src.variability == VertexVariability::Uniform) {

    auto result = UniformToFaceVarying(src.get_data(), src.stride_bytes(),
            faceVertexCounts);

    if (!result) {
      PUSH_ERROR_AND_RETURN(fmt::format("Failed to convert vertex data with 'uniform' variability to 'facevarying': name {}.", attr_name));
    }

    dst->data = result.value();
    dst->elementSize = src.elementSize;
    dst->format = src.format;
    dst->stride = src.stride;
    dst->variability = VertexVariability::FaceVarying;

    return true;

  } else if (src.variability == VertexVariability::Vertex) {

    auto result = VertexToFaceVarying(src.get_data(), src.stride_bytes(),
            faceVertexCounts, faceVertexIndices);

    if (!result) {
      PUSH_ERROR_AND_RETURN(fmt::format("Failed to convert vertex data with 'vertex' variability to 'facevarying': name {}.", attr_name));
    }

    dst->data = result.value();
    dst->elementSize = src.elementSize;
    dst->format = src.format;
    dst->stride = src.stride;
    dst->variability = VertexVariability::FaceVarying;

    return true;
  } else if (src.variability == VertexVariability::FaceVarying) {
    (*dst) = src;
    return true;
  }

#undef PushError

  return false;
}

static bool ToVertexVaryingAttribute(
  const std::string &attr_name,
  const VertexAttribute &src,
  const std::vector<uint32_t> &faceVertexCounts,
  const std::vector<uint32_t> &faceVertexIndices,
  VertexAttribute *dst,
  std::string *err) {

#define PushError(msg) \
  if (err) {           \
    (*err) += msg;     \
  }

  if (!dst) {
    PUSH_ERROR_AND_RETURN("'dest' parameter is nullptr.");
  }

  if (src.variability == VertexVariability::Indexed) {
    PUSH_ERROR_AND_RETURN(fmt::format("'indexed' variability for {} is not supported.", attr_name));
  } else if (src.variability == VertexVariability::Constant) {

    auto result = ConstantToVertex(src.get_data(), src.stride_bytes(),
            faceVertexCounts, faceVertexIndices);

    if (!result) {
      PUSH_ERROR_AND_RETURN(fmt::format("Failed to convert vertex data with 'constant' variability to 'facevarying': name {}.", attr_name));
    }

    dst->data = result.value();
    dst->elementSize = src.elementSize;
    dst->format = src.format;
    dst->stride = src.stride;
    dst->variability = VertexVariability::Vertex;

    return true;

  } else if (src.variability == VertexVariability::Uniform) {

    auto result = UniformToVertex(src.get_data(), src.stride_bytes(),
            faceVertexCounts, faceVertexIndices);

    if (!result) {
      PUSH_ERROR_AND_RETURN(fmt::format("Failed to convert vertex data with 'uniform' variability to 'facevarying': name {}.", attr_name));
    }

    dst->data = result.value();
    dst->elementSize = src.elementSize;
    dst->format = src.format;
    dst->stride = src.stride;
    dst->variability = VertexVariability::Vertex;

    return true;

  } else if (src.variability == VertexVariability::Vertex) {

    (*dst) = src;
    return true;
  } else if (src.variability == VertexVariability::FaceVarying) {

    PUSH_ERROR_AND_RETURN(fmt::format("'facevarying' variability cannot be converted to 'vertex' variability: name {}.", attr_name));

  }

#undef PushError

  return false;

}
#endif

///
/// Triangulate VeretexAttribute data.
///
static bool TriangulateVertexAttribute(
    VertexAttribute &vattr, const std::vector<uint32_t> &faceVertexCounts,
    const std::vector<size_t> &triangulatedToOrigFaceVertexIndexMap,
    const std::vector<uint32_t> &triangulatedFaceCounts,
    const std::vector<uint32_t> &triangulatedFaceVertexIndices,
    std::string *err) {
  if (vattr.vertex_count() == 0) {
    return true;
  }

  if (triangulatedFaceCounts.empty()) {
    PUSH_ERROR_AND_RETURN("triangulatedFaceCounts is empty.");
  }

  if (faceVertexCounts.size() != triangulatedFaceCounts.size()) {
    PUSH_ERROR_AND_RETURN(
        "faceVertexCounts.size must be equal to triangulatedFaceCounts.size.");
  }

  if ((triangulatedFaceVertexIndices.size() % 3) != 0) {
    PUSH_ERROR_AND_RETURN("Invalid size for triangulatedFaceVertexIndices.");
  }

  if (vattr.is_facevarying()) {
    if (triangulatedToOrigFaceVertexIndexMap.size() !=
        triangulatedFaceVertexIndices.size()) {
      PUSH_ERROR_AND_RETURN(
          "triangulatedToOrigFaceVertexIndexMap.size must be equal to "
          "triangulatedFaceVertexIndices.");
    }

    size_t num_vs = vattr.vertex_count();
    std::vector<uint8_t> buf;

    for (uint32_t f = 0; f < triangulatedFaceVertexIndices.size(); f++) {
      // Array index to faceVertexIndices(before triangulation).
      size_t src_fvIdx = triangulatedToOrigFaceVertexIndexMap[f];

      if (src_fvIdx >= num_vs) {
        PUSH_ERROR_AND_RETURN(
            fmt::format("triangulatedToOrigFaceVertexIndexMap[{}] {} exceeds num_vs {}.", f, src_fvIdx, num_vs));
      }

      buf.insert(
          buf.end(), vattr.get_data().data() + src_fvIdx * vattr.stride_bytes(),
          vattr.get_data().data() + (1 + src_fvIdx) * vattr.stride_bytes());
    }

    vattr.data = std::move(buf);
  } else if (vattr.is_vertex()) {
    // # of vertices does not change, so nothing is required.
    return true;
  } else if (vattr.is_indexed()) {
    PUSH_ERROR_AND_RETURN("Indexed VertexAttribute is not supported.");
  } else if (vattr.is_constant()) {
    std::vector<uint8_t> buf;

    for (size_t f = 0; f < triangulatedFaceCounts.size(); f++) {
      uint32_t nf = triangulatedFaceCounts[f];

      // copy `nf` times.
      for (size_t k = 0; k < nf; k++) {
        buf.insert(buf.end(),
                   vattr.get_data().data() + f * vattr.stride_bytes(),
                   vattr.get_data().data() + (1 + f) * vattr.stride_bytes());
      }
    }

    vattr.data = std::move(buf);
  } else if (vattr.is_uniform()) {
    // nothing is required
    return true;
  }

  return true;
}

std::vector<const tinyusdz::GeomSubset *> GetMaterialBindGeomSubsets(
    const tinyusdz::Prim &prim) {
  std::vector<const tinyusdz::GeomSubset *> dst;

  // GeomSubet Prim must be a child Prim of GeomMesh.
  for (const auto &child : prim.children()) {
    if (const tinyusdz::GeomSubset *psubset =
            child.as<tinyusdz::GeomSubset>()) {
      value::token tok;
      if (!psubset->familyName.get_value(&tok)) {
        continue;
      }

      if (tok.str() != "materialBind") {
        continue;
      }

      dst.push_back(psubset);
    }
  }

  return dst;
}

//
// name does not include "primvars:" prefix.
//
nonstd::expected<VertexAttribute, std::string> GetTextureCoordinate(
    const Stage &stage, const GeomMesh &mesh, const std::string &name,
    const double t, const value::TimeSampleInterpolationType tinterp) {
  VertexAttribute vattr;

  (void)stage;

  std::string err;
  GeomPrimvar primvar;
  if (!GetGeomPrimvar(stage, &mesh, name, &primvar, &err)) {
    return nonstd::make_unexpected(err);
  }

  if (!primvar.has_value()) {
    return nonstd::make_unexpected("No value exist for primvars:" + name +
                                   "\n");
  }

  // TODO: allow float2?
  if (primvar.get_type_id() !=
      value::TypeTraits<std::vector<value::texcoord2f>>::type_id()) {
    return nonstd::make_unexpected(
        "Texture coordinate primvar must be texCoord2f[] type, but got " +
        primvar.get_type_name() + "\n");
  }

  std::vector<value::texcoord2f> uvs;
  if (!primvar.flatten_with_indices(t, &uvs, tinterp)) {
    return nonstd::make_unexpected(
        "Failed to retrieve texture coordinate primvar with concrete type.\n");
  }

  if (primvar.get_interpolation() == Interpolation::Varying) {
    vattr.variability = VertexVariability::Varying;
  } else if (primvar.get_interpolation() == Interpolation::Constant) {
    vattr.variability = VertexVariability::Constant;
  } else if (primvar.get_interpolation() == Interpolation::Uniform) {
    vattr.variability = VertexVariability::Uniform;
  } else if (primvar.get_interpolation() == Interpolation::Vertex) {
    vattr.variability = VertexVariability::Vertex;
  } else if (primvar.get_interpolation() == Interpolation::FaceVarying) {
    vattr.variability = VertexVariability::FaceVarying;
  }


  DCOUT("texcoord " << name << " : " << uvs);

  vattr.format = VertexAttributeFormat::Vec2;
  vattr.data.resize(uvs.size() * sizeof(value::texcoord2f));
  memcpy(vattr.data.data(), uvs.data(), vattr.data.size());
  vattr.indices.clear();  // just in case.

  vattr.name = name;  // TODO: add "primvars:" namespace?

  return std::move(vattr);
}

#if 0  // not used at the moment.
///
/// For GeomSubset. Build offset table to corresponding array index in
/// mesh.faceVertexIndices. No need to use this function for triangulated mesh,
/// since the index can be easily computed as `3 * subset.indices[i]`
///
bool BuildFaceVertexIndexOffsets(const std::vector<uint32_t> &faceVertexCounts,
                                 std::vector<size_t> &faceVertexIndexOffsets) {
  size_t offset = 0;
  for (size_t i = 0; i < faceVertexCounts.size(); i++) {
    uint32_t npolys = faceVertexCounts[i];

    faceVertexIndexOffsets.push_back(offset);
    offset += npolys;
  }

  return true;
}
#endif

namespace {

template <typename UnderlyingTy>
bool ScalarValueToVertexAttribute(const value::Value &value,
                                  const std::string &name,
                                  const VertexAttributeFormat format,
                                  VertexAttribute &dst, std::string *err) {
  if (VertexAttributeFormatSize(format) != sizeof(UnderlyingTy)) {
    PUSH_ERROR_AND_RETURN("format size mismatch.");
    return false;
  }

  if (auto pv = value.as<UnderlyingTy>()) {
    dst.data.resize(sizeof(UnderlyingTy));
    memcpy(dst.data.data(), pv, sizeof(UnderlyingTy));

    dst.elementSize = 1;
    dst.stride = 0;
    dst.format = format;
    dst.variability = VertexVariability::Constant;
    dst.name = name;
    dst.indices.clear();
    return true;
  }

  PUSH_ERROR_AND_RETURN("[Internal error] value is not scalar-typed value.");
}

template <typename UnderlyingTy>
bool ArrayValueToVertexAttribute(
    const value::Value &value, const std::string &name,
    const uint32_t elementSize, const VertexVariability variability,
    const uint32_t num_vertices, const uint32_t num_face_counts,
    const uint32_t num_face_vertex_indices, const VertexAttributeFormat format,
    VertexAttribute &dst, std::string *err) {
  if (!value::TypeTraits<UnderlyingTy>::is_array()) {
    PUSH_ERROR_AND_RETURN(
        "[Internal error] UnderlyingTy template parameter must be array type.");
  }

  size_t baseTySize = value::TypeTraits<UnderlyingTy>::size();

  size_t value_counts = value.array_size();
  if (value_counts == 0) {
    PUSH_ERROR_AND_RETURN("Empty array size");
  }

  if (variability == VertexVariability::Indexed) {
    PUSH_ERROR_AND_RETURN("Indexed variability is not supported.");
  }

  if (VertexAttributeFormatSize(format) != baseTySize) {
    PUSH_ERROR_AND_RETURN("format size mismatch. expected "
                          << VertexAttributeFormatSize(format) << " but got "
                          << baseTySize);
    return false;
  }

  DCOUT("value.type = " << value.type_name());
  DCOUT("UnderlyingTy = " << value::TypeTraits<UnderlyingTy>::type_name());
  const auto p = value.as<UnderlyingTy>();
  if (!p) {
    DCOUT("p is nullptr");
  }

  if (auto pv = value.as<UnderlyingTy>()) {
    switch (variability) {
    case VertexVariability::Constant: {
      if (value_counts != elementSize) {
        PUSH_ERROR_AND_RETURN(fmt::format(
            "{} # of items {} expected, but got {}. Variability = Constant",
            name, elementSize, value_counts));
      }
      break;
    }
    case VertexVariability::Uniform: {
      if (value_counts != (elementSize * num_face_counts)) {
        PUSH_ERROR_AND_RETURN(fmt::format(
            "{} # of items {} expected, but got {}. Variability = Uniform",
            name, elementSize * num_face_counts, value_counts));
      }
      break;
    }
    case VertexVariability::Vertex: {
      if (value_counts != (elementSize * num_vertices)) {
        PUSH_ERROR_AND_RETURN(fmt::format(
            "{} # of items {} expected, but got {}. Variability = Vertex",
            name, elementSize * num_vertices, value_counts));
      }
      break;
    case VertexVariability::Varying: {
      if (value_counts != (elementSize * num_vertices)) {
        PUSH_ERROR_AND_RETURN(fmt::format(
            "{} # of items {} expected, but got {}. Variability = Varying",
            name, elementSize * num_vertices, value_counts));
      }
      break;
    }
    case VertexVariability::FaceVarying: {
      if (value_counts != (elementSize * num_face_vertex_indices)) {
        PUSH_ERROR_AND_RETURN(fmt::format(
            "# of items {} expected, but got {}. Variability = FaceVarying",
            elementSize * num_face_vertex_indices, value_counts));
      }
      break;
    }
    case VertexVariability::Indexed: {
      PUSH_ERROR_AND_RETURN(fmt::format(
            "{} Internal error. 'Indexed' variability is not supported."));
      }
      break;
    }
    }

    dst.data.resize(value_counts * baseTySize);
    memcpy(dst.data.data(), pv->data(), value_counts * baseTySize);

    dst.elementSize = elementSize;
    dst.stride = 0;
    dst.format = format;
    dst.variability = variability;
    dst.name = name;
    dst.indices.clear();
    return true;
  }

  PUSH_ERROR_AND_RETURN(fmt::format(
      "Requested underlying type {} but input `value` has underlying type {}.",
      value::TypeTraits<UnderlyingTy>::type_name(),
      value.underlying_type_name()));
}

}  // namespace

bool ToVertexAttribute(const GeomPrimvar &primvar, const std::string &name,
                       const uint32_t num_vertices,
                       const uint32_t num_face_counts,
                       const uint32_t num_face_vertex_indices,
                       VertexAttribute &dst, std::string *err, const double t,
                       const value::TimeSampleInterpolationType tinterp) {
  uint32_t elementSize = uint32_t(primvar.get_elementSize());
  if (elementSize == 0) {
    PUSH_ERROR_AND_RETURN(
        fmt::format("elementSize is zero for primvar: {}", primvar.name()));
  }

  VertexAttribute vattr;

  const tinyusdz::Attribute &attr = primvar.get_attribute();

  value::Value value;
  if (!primvar.flatten_with_indices(t, &value, tinterp)) {
    PUSH_ERROR_AND_RETURN("Failed to flatten primvar");
  }

  bool is_array = value.type_id() & value::TYPE_ID_1D_ARRAY_BIT;
  DCOUT("is_array " << (is_array ? "true" : "false"));

  VertexVariability variability;
  if (primvar.get_interpolation() == Interpolation::Varying) {
    variability = VertexVariability::Varying;
  } else if (primvar.get_interpolation() == Interpolation::Constant) {
    variability = VertexVariability::Constant;
  } else if (primvar.get_interpolation() == Interpolation::Uniform) {
    variability = VertexVariability::Uniform;
  } else if (primvar.get_interpolation() == Interpolation::Vertex) {
    variability = VertexVariability::Vertex;
  } else if (primvar.get_interpolation() == Interpolation::FaceVarying) {
    variability = VertexVariability::FaceVarying;
  } else {
    PUSH_ERROR_AND_RETURN("[Internal Error] Invalid `interpolation` type.");
  }

  uint32_t baseUnderlyingTypeId =
      value.underlying_type_id() & (~value::TYPE_ID_1D_ARRAY_BIT);
  DCOUT("flattened primvar type: " << value.type_name() << ", underlying type "
                                   << value::GetTypeName(baseUnderlyingTypeId));

  // Cast to underlying type

#define TO_TYPED_VALUE(__underlying_ty, __vfmt)                                \
  if (baseUnderlyingTypeId == value::TypeTraits<__underlying_ty>::type_id()) { \
    if (is_array) {                                                            \
      return ArrayValueToVertexAttribute<std::vector<__underlying_ty>>(        \
          value, name, elementSize, variability, num_vertices,                 \
          num_face_counts, num_face_vertex_indices, __vfmt, dst, err);         \
    } else {                                                                   \
      return ScalarValueToVertexAttribute<__underlying_ty>(value, name,        \
                                                           __vfmt, dst, err);  \
    }                                                                          \
  } else

  // specialization for bool type: bool is represented as uint8 in USD primvar
  if (baseUnderlyingTypeId == value::TypeTraits<bool>::type_id()) {
    if (is_array) {
      return ArrayValueToVertexAttribute<std::vector<uint8_t>>(
          value, name, elementSize, variability, num_vertices, num_face_counts,
          num_face_vertex_indices, VertexAttributeFormat::Bool, dst, err);
    } else {
      return ScalarValueToVertexAttribute<uint8_t>(
          value, name, VertexAttributeFormat::Bool, dst, err);
    }
  } else
    TO_TYPED_VALUE(uint8_t, VertexAttributeFormat::Byte)
  TO_TYPED_VALUE(value::uchar2, VertexAttributeFormat::Byte2)
  TO_TYPED_VALUE(value::uchar3, VertexAttributeFormat::Byte3)
  TO_TYPED_VALUE(value::uchar4, VertexAttributeFormat::Byte4)
  TO_TYPED_VALUE(char, VertexAttributeFormat::Char)
  TO_TYPED_VALUE(value::char2, VertexAttributeFormat::Char2)
  TO_TYPED_VALUE(value::char3, VertexAttributeFormat::Char3)
  TO_TYPED_VALUE(value::char4, VertexAttributeFormat::Char4)
  TO_TYPED_VALUE(short, VertexAttributeFormat::Short)
  TO_TYPED_VALUE(value::short2, VertexAttributeFormat::Short2)
  TO_TYPED_VALUE(value::short3, VertexAttributeFormat::Short3)
  TO_TYPED_VALUE(value::short4, VertexAttributeFormat::Short4)
  TO_TYPED_VALUE(uint16_t, VertexAttributeFormat::Ushort)
  TO_TYPED_VALUE(value::ushort2, VertexAttributeFormat::Ushort2)
  TO_TYPED_VALUE(value::ushort3, VertexAttributeFormat::Ushort3)
  TO_TYPED_VALUE(value::ushort4, VertexAttributeFormat::Ushort4)
  TO_TYPED_VALUE(int, VertexAttributeFormat::Int)
  TO_TYPED_VALUE(value::int2, VertexAttributeFormat::Ivec2)
  TO_TYPED_VALUE(value::int3, VertexAttributeFormat::Ivec3)
  TO_TYPED_VALUE(value::int4, VertexAttributeFormat::Ivec4)
  TO_TYPED_VALUE(uint32_t, VertexAttributeFormat::Uint)
  TO_TYPED_VALUE(value::uint2, VertexAttributeFormat::Uvec2)
  TO_TYPED_VALUE(value::uint3, VertexAttributeFormat::Uvec3)
  TO_TYPED_VALUE(value::uint4, VertexAttributeFormat::Uvec4)
  TO_TYPED_VALUE(float, VertexAttributeFormat::Float)
  TO_TYPED_VALUE(value::float2, VertexAttributeFormat::Vec2)
  TO_TYPED_VALUE(value::float3, VertexAttributeFormat::Vec3)
  TO_TYPED_VALUE(value::float4, VertexAttributeFormat::Vec4)
  TO_TYPED_VALUE(value::half, VertexAttributeFormat::Half)
  TO_TYPED_VALUE(value::half2, VertexAttributeFormat::Half2)
  TO_TYPED_VALUE(value::half3, VertexAttributeFormat::Half3)
  TO_TYPED_VALUE(value::half4, VertexAttributeFormat::Half4)
  TO_TYPED_VALUE(double, VertexAttributeFormat::Double)
  TO_TYPED_VALUE(value::double2, VertexAttributeFormat::Dvec2)
  TO_TYPED_VALUE(value::double3, VertexAttributeFormat::Dvec3)
  TO_TYPED_VALUE(value::double4, VertexAttributeFormat::Dvec4)
  TO_TYPED_VALUE(value::matrix2f, VertexAttributeFormat::Mat2)
  TO_TYPED_VALUE(value::matrix3f, VertexAttributeFormat::Mat3)
  TO_TYPED_VALUE(value::matrix4f, VertexAttributeFormat::Mat4)
  TO_TYPED_VALUE(value::matrix2d, VertexAttributeFormat::Dmat2)
  TO_TYPED_VALUE(value::matrix3d, VertexAttributeFormat::Dmat3)
  TO_TYPED_VALUE(value::matrix4d, VertexAttributeFormat::Dmat4) {
    PUSH_ERROR_AND_RETURN(
        fmt::format("Unknown or unsupported data type for Geom PrimVar: {}",
                    attr.type_name()));
  }

#undef TO_TYPED_VALUE
}

#if 0  // TODO: Remove. The following could be done using ToVertexAttribute +
       // TriangulateVertexAttribute
///
/// Triangulate Geom primvar.
///
/// triangulatted indices are computed in `TriangulatePolygon` API.
///
/// @param[in] mesh Geom mesh
/// @param[in] name Geom Primvar name.
/// @param[in] triangulatedFaceVertexIndices Triangulated faceVertexIndices(len
/// = 3 * triangles)
/// @param[in] triangulatedToOrigFaceVertexIndexMap Triangulated faceVertexIndex
/// to original faceVertexIndex remapping table. len = 3 * triangles.
///
nonstd::expected<VertexAttribute, std::string> TriangulateGeomPrimvar(
    const GeomMesh &mesh, const std::string &name,
    const std::vector<uint32_t> &faceVertexCounts,
    const std::vector<uint32_t> &faceVertexIndices,
    const std::vector<uint32_t> &triangulatedFaceVertexIndices,
    const std::vector<size_t> &triangulatedToOrigFaceVertexIndexMap) {
  GeomPrimvar primvar;

  if (triangulatedFaceVertexIndices.size() % 3 != 0) {
    return nonstd::make_unexpected(fmt::format(
        "triangulatedFaceVertexIndices.size {} must be the multiple of 3.\n",
        triangulatedFaceVertexIndices.size()));
  }

  if (!GetGeomPrimvar(name, &primvar)) {
    return nonstd::make_unexpected(
        fmt::format("No primvars:{} found in GeomMesh {}\n", name, mesh.name));
  }

  if (!primvar.has_value()) {
    // TODO: Create empty VertexAttribute?
    return nonstd::make_unexpected(
        fmt::format("No value exist for primvars:{}\n", name));
  }

  //
  // Flatten Indexed PrimVar(return raw primvar for non-Indexed PrimVar)
  //
  std::string err;
  value::Value flattened;
  if (!primvar.flatten_with_indices(t, &flattened, tinterp, &err)) {
    return nonstd::make_unexpected(fmt::format(
        "Failed to flatten Indexed PrimVar: {}. Error = {}\n", name, err));
  }

  VertexAttribute vattr;

  if (!ToVertexAttributeData(primvar, &vattr, &err)) {
    return nonstd::make_unexpected(fmt::format(
        "Failed to convert Geom PrimVar to VertexAttribute for {}. Error = {}\n", name, err));
  }

  return vattr;
}
#endif

#if 1
///
/// Input: points, faceVertexCounts, faceVertexIndices
/// Output: triangulated faceVertexCounts(all filled with 3), triangulated
/// faceVertexIndices, triangulatedToOrigFaceVertexIndexMap (length =
/// triangulated faceVertexIndices. triangulatedToOrigFaceVertexIndexMap[i]
/// stores an array index to original faceVertexIndices. For remapping
/// facevarying primvar attributes.)
///
/// triangulatedFaceVertexCounts: len = len(faceVertexCounts). Records the
/// number of triangle faces. 1 = triangle. 2 = quad, ... For remapping face
/// indices(e.g. GeomSubset::indices)
///
/// triangulated*** output is generated even when input mesh is fully composed
/// from triangles(`faceVertexCounts` are all filled with 3) Return false when a
/// polygon is degenerated. No overlap check at the moment
///
/// Example:
///   - faceVertexCounts = [4]
///   - faceVertexIndices = [0, 1, 3, 2]
///
///   - triangulatedFaceVertexCounts = [3, 3]
///   - triangulatedFaceVertexIndices = [0, 1, 3, 0, 3, 2]
///   - triangulatedToOrigFaceVertexIndexMap = [0, 1, 2, 0, 2, 3]
///
/// T = value::float3 or value::double3
/// BaseTy = float or double
template <typename T, typename BaseTy>
bool TriangulatePolygon(
    const std::vector<T> &points, const std::vector<uint32_t> &faceVertexCounts,
    const std::vector<uint32_t> &faceVertexIndices,
    std::vector<uint32_t> &triangulatedFaceVertexCounts,
    std::vector<uint32_t> &triangulatedFaceVertexIndices,
    std::vector<size_t> &triangulatedToOrigFaceVertexIndexMap,
    std::vector<uint32_t> &triangulatedFaceCounts, std::string &err) {
  triangulatedFaceVertexCounts.clear();
  triangulatedFaceVertexIndices.clear();

  triangulatedToOrigFaceVertexIndexMap.clear();

  size_t faceIndexOffset = 0;

  // For each polygon(face)
  for (size_t i = 0; i < faceVertexCounts.size(); i++) {
    uint32_t npolys = faceVertexCounts[i];

    if (npolys < 3) {
      err = fmt::format(
          "faceVertex count must be 3(triangle) or "
          "more(polygon), but got faceVertexCounts[{}] = {}\n",
          i, npolys);
      return false;
    }

    if (faceIndexOffset + npolys > faceVertexIndices.size()) {
      err = fmt::format(
          "Invalid faceVertexIndices or faceVertexCounts. faceVertex index "
          "exceeds faceVertexIndices.size() at [{}]\n",
          i);
      return false;
    }

    if (npolys == 3) {
      // No need for triangulation.
      triangulatedFaceVertexCounts.push_back(3);
      triangulatedFaceVertexIndices.push_back(
          faceVertexIndices[faceIndexOffset + 0]);
      triangulatedFaceVertexIndices.push_back(
          faceVertexIndices[faceIndexOffset + 1]);
      triangulatedFaceVertexIndices.push_back(
          faceVertexIndices[faceIndexOffset + 2]);
      triangulatedToOrigFaceVertexIndexMap.push_back(faceIndexOffset + 0);
      triangulatedToOrigFaceVertexIndexMap.push_back(faceIndexOffset + 1);
      triangulatedToOrigFaceVertexIndexMap.push_back(faceIndexOffset + 2);
      triangulatedFaceCounts.push_back(1);
#if 1
    } else if (npolys == 4) {
      // Use simple split
      // TODO: Split at shortest edge for better triangulation.
      triangulatedFaceVertexCounts.push_back(3);
      triangulatedFaceVertexCounts.push_back(3);

      triangulatedFaceVertexIndices.push_back(
          faceVertexIndices[faceIndexOffset + 0]);
      triangulatedFaceVertexIndices.push_back(
          faceVertexIndices[faceIndexOffset + 1]);
      triangulatedFaceVertexIndices.push_back(
          faceVertexIndices[faceIndexOffset + 2]);

      triangulatedFaceVertexIndices.push_back(
          faceVertexIndices[faceIndexOffset + 0]);
      triangulatedFaceVertexIndices.push_back(
          faceVertexIndices[faceIndexOffset + 2]);
      triangulatedFaceVertexIndices.push_back(
          faceVertexIndices[faceIndexOffset + 3]);

      triangulatedToOrigFaceVertexIndexMap.push_back(faceIndexOffset + 0);
      triangulatedToOrigFaceVertexIndexMap.push_back(faceIndexOffset + 1);
      triangulatedToOrigFaceVertexIndexMap.push_back(faceIndexOffset + 2);
      triangulatedToOrigFaceVertexIndexMap.push_back(faceIndexOffset + 0);
      triangulatedToOrigFaceVertexIndexMap.push_back(faceIndexOffset + 2);
      triangulatedToOrigFaceVertexIndexMap.push_back(faceIndexOffset + 3);
      triangulatedFaceCounts.push_back(2);
#endif
    } else {
      // Use double for accuracy. `float` precision may classify small-are polygon as degenerated.
      // Find the normal axis of the polygon using Newell's method
      value::double3 n = {0, 0, 0};

      size_t vi0;
      size_t vi0_2;

      for (size_t k = 0; k < npolys; ++k) {
        vi0 = faceVertexIndices[faceIndexOffset + k];

        size_t j = (k + 1) % npolys;
        vi0_2 = faceVertexIndices[faceIndexOffset + j];

        if (vi0 >= points.size()) {
          err = fmt::format("Invalid vertex index.\n");
          return false;
        }

        if (vi0_2 >= points.size()) {
          err = fmt::format("Invalid vertex index.\n");
          return false;
        }

        T v0 = points[vi0];
        T v1 = points[vi0_2];

        const T point1 = {v0[0], v0[1], v0[2]};
        const T point2 = {v1[0], v1[1], v1[2]};

        T a = {point1[0] - point2[0], point1[1] - point2[1],
               point1[2] - point2[2]};
        T b = {point1[0] + point2[0], point1[1] + point2[1],
               point1[2] + point2[2]};

        n[0] += double(a[1] * b[2]);
        n[1] += double(a[2] * b[0]);
        n[2] += double(a[0] * b[1]);
        DCOUT("v0 " << v0);
        DCOUT("v1 " << v1);
        DCOUT("n " << n);
      }
      //BaseTy length_n = vlength(n);
      double length_n = vlength(n);

      // Check if zero length normal
      if (std::fabs(length_n) < std::numeric_limits<double>::epsilon()) {
        DCOUT("length_n " << length_n);
        err = "Degenerated polygon found.\n";
        return false;
      }

      // Negative is to flip the normal to the correct direction
      n = vnormalize(n);

      T axis_w, axis_v, axis_u;
      axis_w[0] = BaseTy(n[0]);
      axis_w[1] = BaseTy(n[1]);
      axis_w[2] = BaseTy(n[2]);
      T a;
      if (std::fabs(axis_w[0]) > BaseTy(0.9999999)) {  // TODO: use 1.0 - eps?
        a = {BaseTy(0), BaseTy(1), BaseTy(0)};
      } else {
        a = {BaseTy(1), BaseTy(0), BaseTy(0)};
      }
      axis_v = vnormalize(vcross(axis_w, a));
      axis_u = vcross(axis_w, axis_v);

      using Point3D = std::array<BaseTy, 3>;
      using Point2D = std::array<BaseTy, 2>;
      std::vector<Point2D> polyline;

      // TMW change: Find best normal and project v0x and v0y to those
      // coordinates, instead of picking a plane aligned with an axis (which
      // can flip polygons).

      // Fill polygon data.
      for (size_t k = 0; k < npolys; k++) {
        size_t vidx = faceVertexIndices[faceIndexOffset + k];

        value::float3 v = points[vidx];
        // Point3 polypoint = {v0[0],v0[1],v0[2]};

        // world to local
        Point3D loc = {vdot(v, axis_u), vdot(v, axis_v), vdot(v, axis_w)};

        polyline.push_back({loc[0], loc[1]});
      }

      std::vector<std::vector<Point2D>> polygon_2d;
      // Single polygon only(no holes)

      std::vector<uint32_t> indices = mapbox::earcut<uint32_t>(polygon_2d);
      //  => result = 3 * faces, clockwise

      if ((indices.size() % 3) != 0) {
        // This should not be happen, though.
        err = "Failed to triangulate.\n";
        return false;
      }

      size_t ntris = indices.size() / 3;

      // Up to 2GB tris.
      if (ntris > size_t((std::numeric_limits<int32_t>::max)())) {
        err = "Too many triangles are generated.\n";
        return false;
      }

      for (size_t k = 0; k < ntris; k++) {
        triangulatedFaceVertexCounts.push_back(3);
        triangulatedFaceVertexIndices.push_back(
            faceVertexIndices[faceIndexOffset + indices[3 * k + 0]]);
        triangulatedFaceVertexIndices.push_back(
            faceVertexIndices[faceIndexOffset + indices[3 * k + 1]]);
        triangulatedFaceVertexIndices.push_back(
            faceVertexIndices[faceIndexOffset + indices[3 * k + 2]]);

        triangulatedToOrigFaceVertexIndexMap.push_back(faceIndexOffset +
                                                       indices[3 * k + 0]);
        triangulatedToOrigFaceVertexIndexMap.push_back(faceIndexOffset +
                                                       indices[3 * k + 1]);
        triangulatedToOrigFaceVertexIndexMap.push_back(faceIndexOffset +
                                                       indices[3 * k + 2]);
      }
      triangulatedFaceCounts.push_back(uint32_t(ntris));
    }

    faceIndexOffset += npolys;
  }

  return true;
}
#endif

#if 0  // not used atm.
// Building an Orthonormal Basis, Revisited
// http://jcgt.org/published/0006/01/01/
static void GenerateBasis(const vec3 &n, vec3 *tangent,
                         vec3 *binormal)
{
  if (n[2] < 0.0f) {
    const float a = 1.0f / (1.0f - n[2]);
    const float b = n[0] * n[1] * a;
    (*tangent) = vec3{1.0f - n[0] * n[0] * a, -b, n[0]};
    (*binormal) = vec3{b, n[1] * n[1] * a - 1.0f, -n[1]};
  } else {
    const float a = 1.0f / (1.0f + n[2]);
    const float b = -n[0] * n[1] * a;
    (*tangent) = vec3{1.0f - n[0] * n[0] * a, b, -n[0]};
    (*binormal) = vec3{b, 1.0f - n[1] * n[1] * a, -n[1]};
  }
}
#endif

struct ComputeTangentPackedVertexData {
  // value::float3 position;
  uint32_t point_index;
  value::float3 normal;
  value::float2 uv;

  // comparator for std::map
  bool operator<(const DefaultPackedVertexData &rhs) const {
    return memcmp(reinterpret_cast<const void *>(this),
                  reinterpret_cast<const void *>(&rhs),
                  sizeof(DefaultPackedVertexData)) > 0;
  }
};

struct ComputeTangentPackedVertexDataHasher {
  inline size_t operator()(const ComputeTangentPackedVertexData &v) const {
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

struct ComputeTangentPackedVertexDataEqual {
  bool operator()(const ComputeTangentPackedVertexData &lhs,
                  const ComputeTangentPackedVertexData &rhs) const {
    return memcmp(reinterpret_cast<const void *>(&lhs),
                  reinterpret_cast<const void *>(&rhs),
                  sizeof(ComputeTangentPackedVertexData)) == 0;
  }
};

template <class PackedVert>
struct ComputeTangentVertexInput {
  // std::vector<value::float3> positions;
  std::vector<uint32_t> point_indices;
  std::vector<value::float3> normals;
  std::vector<value::float2> uvs;

  size_t size() const { return point_indices.size(); }

  void get(size_t idx, PackedVert &output) const {
    if (idx < point_indices.size()) {
      output.point_index = point_indices[idx];
    } else {
      output.point_index = ~0u;  // never should reach here though.
    }
    if (idx < normals.size()) {
      output.normal = normals[idx];
    } else {
      output.normal = {0.0f, 0.0f, 0.0f};
    }
    if (idx < uvs.size()) {
      output.uv = uvs[idx];
    } else {
      output.uv = {0.0f, 0.0f};
    }
  }
};

template <class PackedVert>
struct ComputeTangentVertexOutput {
  // std::vector<value::float3> positions;
  std::vector<uint32_t> point_indices;
  std::vector<value::float3> normals;
  std::vector<value::float2> uvs;

  size_t size() const { return point_indices.size(); }

  void push_back(const PackedVert &v) {
    // positions.push_back(v.position);
    point_indices.push_back(v.point_index);
    normals.push_back(v.normal);
    uvs.push_back(v.uv);
  }
};

///
/// Compute facevarying tangent and facevarying binormal.
///
/// Reference:
/// http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-13-normal-mapping
///
/// Implemented code uses two adjacent edge composed from three vertices v_{i},
/// v_{i+1}, v_{i+2} for i < (N - 1) , where N is the number of vertices per
/// facet.
///
/// This may produce unwanted tangent/binormal frame for ill-defined
/// polygon(quad, pentagon, ...). Also, we assume input mesh has well-formed and
/// has no or few vertices with similar property(position, uvs and normals)
///
/// TODO:
/// - [ ] Implement better getSimilarVertexIndex in the above opengl-tutorial to
/// better average tangent/binormal.
///   - Use kNN search(e.g. nanoflann https://github.com/jlblancoc/nanoflann ),
///   or point-query by building BVH over the mesh points.
///     - BVH builder candidate:
///       - NanoRT https://github.com/lighttransport/nanort
///       - bvh https://github.com/madmann91/bvh
///   - Or we can quantize vertex attributes and compute locally sensitive
///   hashsing? https://dl.acm.org/doi/10.1145/3188745.3188846
/// - [ ] Support robusut computing tangent/binormal on arbitrary mesh.
///  - e.g. vector field calculation, use instance-mesh algorithm, etc...
//   - Use half-edges to find adjacent face/vertex.
///
///
/// @param[in] vertices Vertex points(`vertex` variability).
/// @param[in] faceVertexCounts faceVertexCounts of the mesh.
/// @param[in] faceVertexIndices faceVertexIndices of the mesh.
/// @param[in] texcoords Primary texcoords.
/// @param[in] normals normals.
/// @param[in] is_facevarying_input false = texcoords and normals are 'vertex'
/// variability. true = 'facevarying' variability.
/// @param[out] tangents Computed tangents;
/// @param[out] binormals Computed binormals;
/// @param[out] out_vertex_indices Vertex indices.
/// @param[out] err Error message.
///
static bool ComputeTangentsAndBinormals(
    const std::vector<vec3> &vertices,
    const std::vector<uint32_t> &faceVertexCounts,
    const std::vector<uint32_t> &faceVertexIndices,
    const std::vector<vec2> &texcoords, const std::vector<vec3> &normals,
    bool is_facevarying_input,  // false: 'vertex' varying
    std::vector<vec3> *tangents, std::vector<vec3> *binormals,
    std::vector<uint32_t> *out_vertex_indices, std::string *err) {
  if (!tangents) {
    PUSH_ERROR_AND_RETURN("tangents arg is nullptr.");
  }

  if (!binormals) {
    PUSH_ERROR_AND_RETURN("binormals arg is nullptr.");
  }

  if (!out_vertex_indices) {
    PUSH_ERROR_AND_RETURN("out_indices arg is nullptr.");
  }

  if (vertices.empty()) {
    PUSH_ERROR_AND_RETURN("vertices is empty.");
  }

  // At least 1 triangle face should exist.
  if (faceVertexIndices.size() < 3) {
    PUSH_ERROR_AND_RETURN("faceVertexIndices.size < 3");
  }

  if (texcoords.empty()) {
    PUSH_ERROR_AND_RETURN("texcoords is empty");
  }

  if (normals.empty()) {
    PUSH_ERROR_AND_RETURN("normals is empty");
  }

  if (is_facevarying_input) {
    if (vertices.size() != faceVertexIndices.size()) {
      PUSH_ERROR_AND_RETURN("Invalid vertices.size.");
    }
    if (texcoords.size() != faceVertexIndices.size()) {
      PUSH_ERROR_AND_RETURN("Invalid texcoords.size.");
    }
    if (normals.size() != faceVertexIndices.size()) {
      PUSH_ERROR_AND_RETURN("Invalid normals.size.");
    }
  } else {
    uint32_t max_vert_index =
        *std::max_element(faceVertexIndices.begin(), faceVertexIndices.end());
    if (max_vert_index >= vertices.size()) {
      PUSH_ERROR_AND_RETURN("Invalid vertices.size.");
    }
    if (max_vert_index >= texcoords.size()) {
      PUSH_ERROR_AND_RETURN("Invalid texcoords.size.");
    }
    if (max_vert_index >= normals.size()) {
      PUSH_ERROR_AND_RETURN("Invalid normals.size.");
    }
  }

  bool hasFaceVertexCounts = true;
  if (faceVertexCounts.size() == 0) {
    // Assume all triangle faces.
    if ((faceVertexIndices.size() % 3) != 0) {
      PUSH_ERROR_AND_RETURN(
          "Invalid faceVertexIndices. It must be all triangles: "
          "faceVertexIndices.size % 3 == 0");
    }
    hasFaceVertexCounts = false;
  }

  // tn, bn = facevarying
  std::vector<value::normal3f> tn(faceVertexIndices.size());
  memset(&tn.at(0), 0, sizeof(value::normal3f) * tn.size());
  std::vector<value::normal3f> bn(faceVertexIndices.size());
  memset(&bn.at(0), 0, sizeof(value::normal3f) * bn.size());

  //
  // 1. Compute facevarying tangent/binormal for each faceVertex.
  //
  size_t faceVertexIndexOffset{0};
  for (size_t i = 0; i < faceVertexCounts.size(); i++) {
    size_t nv = hasFaceVertexCounts ? faceVertexCounts[i] : 3;

    if ((faceVertexIndexOffset + nv) >= faceVertexIndices.size()) {
      // Invalid faceVertexIndices
      PUSH_ERROR_AND_RETURN("Invalid value in faceVertexOffset.");
    }

    if (nv < 3) {
      PUSH_ERROR_AND_RETURN("Degenerated facet found.");
    }

    // Process each two-edges per facet.
    //
    // Example:
    //
    // fv3
    //  o----------------o fv2
    //   \              /
    //    \            /
    //     o----------o
    //    fv0         fv1

    // facet0:  fv0, fv1, fv2
    // facet1:  fv1, fv2, fv3

    for (size_t f = 0; f < nv - 2; f++) {
      size_t fid0 = faceVertexIndexOffset + f;
      size_t fid1 = faceVertexIndexOffset + f + 1;
      size_t fid2 = faceVertexIndexOffset + f + 2;

      uint32_t vf0 =
          is_facevarying_input ? uint32_t(fid0) : faceVertexIndices[fid0];
      uint32_t vf1 =
          is_facevarying_input ? uint32_t(fid1) : faceVertexIndices[fid1];
      uint32_t vf2 =
          is_facevarying_input ? uint32_t(fid2) : faceVertexIndices[fid2];

      if ((vf0 >= vertices.size()) || (vf1 >= vertices.size()) ||
          (vf2 >= vertices.size())) {
        // index out-of-range
        PUSH_ERROR_AND_RETURN(
            "Invalid value in faceVertexIndices. some exceeds vertices.size()");
      }

      vec3 v1 = vertices[vf0];
      vec3 v2 = vertices[vf1];
      vec3 v3 = vertices[vf2];

      float v1x = v1[0];
      float v1y = v1[1];
      float v1z = v1[2];

      float v2x = v2[0];
      float v2y = v2[1];
      float v2z = v2[2];

      float v3x = v3[0];
      float v3y = v3[1];
      float v3z = v3[2];

      float w1x = 0.0f;
      float w1y = 0.0f;
      float w2x = 0.0f;
      float w2y = 0.0f;
      float w3x = 0.0f;
      float w3y = 0.0f;

      if ((vf0 >= texcoords.size()) || (vf1 >= texcoords.size()) ||
          (vf2 >= texcoords.size())) {
        // index out-of-range
        PUSH_ERROR_AND_RETURN("Invalid index. some exceeds texcoords.size()");
      }

      {
        vec2 uv1 = texcoords[vf0];
        vec2 uv2 = texcoords[vf1];
        vec2 uv3 = texcoords[vf2];

        w1x = uv1[0];
        w1y = uv1[1];
        w2x = uv2[0];
        w2y = uv2[1];
        w3x = uv3[0];
        w3y = uv3[1];
      }

      float x1 = v2x - v1x;
      float x2 = v3x - v1x;
      float y1 = v2y - v1y;
      float y2 = v3y - v1y;
      float z1 = v2z - v1z;
      float z2 = v3z - v1z;

      float s1 = w2x - w1x;
      float s2 = w3x - w1x;
      float t1 = w2y - w1y;
      float t2 = w3y - w1y;

      float r = 1.0;

      if (std::fabs(double(s1 * t2 - s2 * t1)) > 1.0e-20) {
        r /= (s1 * t2 - s2 * t1);
      }

      vec3 tdir{(t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r,
                (t2 * z1 - t1 * z2) * r};
      vec3 bdir{(s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r,
                (s1 * z2 - s2 * z1) * r};

      //
      // NOTE: for quad or polygon mesh, this overwrites previous 2 facevarying
      // points for each face.
      //       And this would not be a good way to compute tangents for
      //       quad/polygon.
      //

      tn[fid0][0] = tdir[0];
      tn[fid0][1] = tdir[1];
      tn[fid0][2] = tdir[2];

      tn[fid1][0] = tdir[0];
      tn[fid1][1] = tdir[1];
      tn[fid1][2] = tdir[2];

      tn[fid2][0] = tdir[0];
      tn[fid2][1] = tdir[1];
      tn[fid2][2] = tdir[2];

      bn[fid0][0] = bdir[0];
      bn[fid0][1] = bdir[1];
      bn[fid0][2] = bdir[2];

      bn[fid1][0] = bdir[0];
      bn[fid1][1] = bdir[1];
      bn[fid1][2] = bdir[2];

      bn[fid2][0] = bdir[0];
      bn[fid2][1] = bdir[1];
      bn[fid2][2] = bdir[2];
    }

    faceVertexIndexOffset += nv;
  }

  //
  // 2. Build indices(use same index for shared-vertex)
  //
  std::vector<uint32_t> vertex_indices;  // len = faceVertexIndices.size()
  {
    ComputeTangentVertexInput<ComputeTangentPackedVertexData> vertex_input;
    ComputeTangentVertexOutput<ComputeTangentPackedVertexData> vertex_output;

    if (is_facevarying_input) {
      // input position is still in 'vertex' variability.
      for (size_t i = 0; i < faceVertexIndices.size(); i++) {
        vertex_input.point_indices.push_back(faceVertexIndices[i]);
      }
      vertex_input.normals = normals;
      vertex_input.uvs = texcoords;
    } else {
      // expand to facevarying.
      for (size_t i = 0; i < faceVertexIndices.size(); i++) {
        vertex_input.point_indices.push_back(faceVertexIndices[i]);
        vertex_input.normals.push_back(normals[faceVertexIndices[i]]);
        vertex_input.uvs.push_back(texcoords[faceVertexIndices[i]]);
      }
    }

    std::vector<uint32_t> vertex_point_indices;

    BuildIndices<ComputeTangentVertexInput<ComputeTangentPackedVertexData>,
                 ComputeTangentVertexOutput<ComputeTangentPackedVertexData>,
                 ComputeTangentPackedVertexData,
                 ComputeTangentPackedVertexDataHasher,
                 ComputeTangentPackedVertexDataEqual>(
        vertex_input, vertex_output, vertex_indices, vertex_point_indices);

    DCOUT("faceVertexIndices.size : " << faceVertexIndices.size());
    DCOUT("# of indices after the build: "
          << vertex_indices.size() << ", reduced "
          << (faceVertexIndices.size() - vertex_indices.size()) << " indices.");
    // We only need indices. Discard vertex_output and vertrex_point_indices
  }

  const uint32_t num_verts =
      *std::max_element(vertex_indices.begin(), vertex_indices.end());

  //
  // 3. normalize * orthogonalize;
  //

  // per-vertex tangents/binormals
  std::vector<value::normal3f> v_tn;
  v_tn.assign(num_verts, {0.0f, 0.0f, 0.0f});

  std::vector<value::normal3f> v_bn;
  v_bn.assign(num_verts, {0.0f, 0.0f, 0.0f});

  for (size_t i = 0; i < vertex_indices.size(); i++) {
    value::normal3f Tn = tn[vertex_indices[i]];
    value::normal3f Bn = bn[vertex_indices[i]];

    v_tn[vertex_indices[i]][0] += Tn[0];
    v_tn[vertex_indices[i]][1] += Tn[1];
    v_tn[vertex_indices[i]][2] += Tn[2];

    v_bn[vertex_indices[i]][0] += Bn[0];
    v_bn[vertex_indices[i]][1] += Bn[1];
    v_bn[vertex_indices[i]][2] += Bn[2];
  }

  for (size_t i = 0; i < size_t(num_verts); i++) {
    if (vlength(v_tn[i]) > 0.0f) {
      v_tn[i] = vnormalize(v_tn[i]);
    }
    if (vlength(v_bn[i]) > 0.0f) {
      v_bn[i] = vnormalize(v_bn[i]);
    }
  }

  tangents->assign(num_verts, {0.0f, 0.0f, 0.0f});
  binormals->assign(num_verts, {0.0f, 0.0f, 0.0f});

  for (size_t i = 0; i < vertex_indices.size(); i++) {
    value::normal3f n;

    // http://www.terathon.com/code/tangent.html

    n[0] = normals[vertex_indices[i]][0];
    n[1] = normals[vertex_indices[i]][1];
    n[2] = normals[vertex_indices[i]][2];

    value::normal3f Tn = v_tn[vertex_indices[i]];
    value::normal3f Bn = v_bn[vertex_indices[i]];

    // Gram-Schmidt orthogonalize
    Tn = (Tn - n * vdot(n, Tn));
    if (vlength(Tn) > 0.0f) {
      Tn = vnormalize(Tn);
    }

    // Calculate handedness
    if (vdot(vcross(n, Tn), Bn) < 0.0f) {
      Tn = Tn * -1.0f;
    }

    ((*tangents)[vertex_indices[i]])[0] = Tn[0];
    ((*tangents)[vertex_indices[i]])[1] = Tn[1];
    ((*tangents)[vertex_indices[i]])[2] = Tn[2];

    ((*binormals)[vertex_indices[i]])[0] = Bn[0];
    ((*binormals)[vertex_indices[i]])[1] = Bn[1];
    ((*binormals)[vertex_indices[i]])[2] = Bn[2];
  }

  (*out_vertex_indices) = vertex_indices;

  return true;
}

//
// Compute geometric normal in CCW(Counter Clock-Wise) manner
// Also computes the area of the input triangle.
//
inline static value::float3 GeometricNormal(const value::float3 v0,
                                            const value::float3 v1,
                                            const value::float3 v2,
                                            float &area) {
  const value::float3 v10 = v1 - v0;
  const value::float3 v20 = v2 - v0;

  value::float3 Nf = vcross(v10, v20);  // CCW
  area = 0.5f * vlength(Nf);
  Nf = vnormalize(Nf);

  return Nf;
}

//
// Compute a normal for vertices.
// Normal vector is computed as weighted(by the area of the triangle) vector.
//
// TODO: Implement better normal calculation. ref.
// http://www.bytehazard.com/articles/vertnorm.html
//
static bool ComputeNormals(const std::vector<vec3> &vertices,
                           const std::vector<uint32_t> &faceVertexCounts,
                           const std::vector<uint32_t> &faceVertexIndices,
                           std::vector<vec3> &normals, std::string *err) {
  normals.assign(vertices.size(), {0.0f, 0.0f, 0.0f});

  size_t faceVertexIndexOffset{0};
  for (size_t f = 0; f < faceVertexCounts.size(); f++) {
    size_t nv = faceVertexCounts[f];

    if (nv < 3) {
      PUSH_ERROR_AND_RETURN(
          fmt::format("Invalid face num {} at faceVertexCounts[{}]", nv, f));
    }

    // For quad/polygon, first three vertices are used to compute face normal
    // (Assume quad/polygon plane is co-planar)
    uint32_t vidx0 = faceVertexIndices[faceVertexIndexOffset + 0];
    uint32_t vidx1 = faceVertexIndices[faceVertexIndexOffset + 1];
    uint32_t vidx2 = faceVertexIndices[faceVertexIndexOffset + 2];

    if (vidx0 >= vertices.size()) {
      PUSH_ERROR_AND_RETURN(
          fmt::format("vertexIndex0 {} exceeds vertices.size {}", vidx0, vertices.size()));
    }

    if (vidx1 >= vertices.size()) {
      PUSH_ERROR_AND_RETURN(
          fmt::format("vertexIndex1 {} exceeds vertices.size {}", vidx1, vertices.size()));
    }

    if (vidx2 >= vertices.size()) {
      PUSH_ERROR_AND_RETURN(
          fmt::format("vertexIndex2 {} exceeds vertices.size {}", vidx2, vertices.size()));
    }

    float area{0.0f};
    value::float3 Nf = GeometricNormal(vertices[vidx0], vertices[vidx1],
                                       vertices[vidx2], area);

    for (size_t v = 0; v < nv; v++) {
      uint32_t vidx = faceVertexIndices[faceVertexIndexOffset + v];
      if (vidx >= vertices.size()) {
        PUSH_ERROR_AND_RETURN(fmt::format(
            "vertexIndex exceeds vertices.size {}", vertices.size()));
      }
      normals[vidx] += area * Nf;
    }

    faceVertexIndexOffset += nv;
  }

  for (size_t v = 0; v < normals.size(); v++) {
    normals[v] = vnormalize(normals[v]);
  }

  return true;
}

}  // namespace

#if 0
// Currently float2 only
std::vector<UsdPrimvarReader_float2> ExtractPrimvarReadersFromMaterialNode(
    const Prim &node) {
  std::vector<UsdPrimvarReader_float2> dst;

  if (!node.is<Material>()) {
    return dst;
  }

  for (const auto &child : node.children()) {
    (void)child;
  }

  // Traverse and find PrimvarReader_float2 shader.
  return dst;
}

nonstd::expected<Node, std::string> ConvertXform(const Stage &stage,
                                            const Xform &xform) {
  (void)stage;

  // TODO: timeSamples

  Node node;
  if (auto m = xform.GetLocalMatrix()) {
    node.local_matrix = m.value();
  }

  return std::move(node);
}
#endif

namespace {

bool ListUVNames(const RenderMaterial &material,
                 const std::vector<UVTexture> &textures,
                 StringAndIdMap &si_map) {
  // TODO: Use auto
  auto fun_vec3 = [&](const ShaderParam<vec3> &param) {
    int32_t texId = param.texture_id;
    if ((texId >= 0) && (size_t(texId) < textures.size())) {
      const UVTexture &tex = textures[size_t(texId)];
      if (tex.varname_uv.size()) {
        if (!si_map.count(tex.varname_uv)) {
          uint64_t slotId = si_map.size();
          DCOUT("Add textureSlot: " << tex.varname_uv << ", " << slotId);
          si_map.add(tex.varname_uv, slotId);
        }
      }
    }
  };

  auto fun_float = [&](const ShaderParam<float> &param) {
    int32_t texId = param.texture_id;
    if ((texId >= 0) && (size_t(texId) < textures.size())) {
      const UVTexture &tex = textures[size_t(texId)];
      if (tex.varname_uv.size()) {
        if (!si_map.count(tex.varname_uv)) {
          uint64_t slotId = si_map.size();
          DCOUT("Add textureSlot: " << tex.varname_uv << ", " << slotId);
          si_map.add(tex.varname_uv, slotId);
        }
      }
    }
  };

  fun_vec3(material.surfaceShader.diffuseColor);
  fun_vec3(material.surfaceShader.normal);
  fun_float(material.surfaceShader.metallic);
  fun_float(material.surfaceShader.roughness);
  fun_float(material.surfaceShader.clearcoat);
  fun_float(material.surfaceShader.clearcoatRoughness);
  fun_float(material.surfaceShader.opacity);
  fun_float(material.surfaceShader.opacityThreshold);
  fun_float(material.surfaceShader.ior);
  fun_float(material.surfaceShader.displacement);
  fun_float(material.surfaceShader.occlusion);

  return true;
}

#undef PushError

}  // namespace

///
/// Convert vertex variability either 'vertex' or 'facevarying'
///
/// @param[in] to_vertex_varying true: Convert to 'vertrex' varying. false:
/// Convert to 'facevarying'
///
bool RenderSceneConverter::ConvertVertexVariabilityImpl(
    VertexAttribute &vattr, const bool to_vertex_varying,
    const std::vector<uint32_t> &faceVertexCounts,
    const std::vector<uint32_t> &faceVertexIndices) {
  if (vattr.data.empty()) {
    return true;
  }

  if (vattr.variability == VertexVariability::Uniform) {
    if (to_vertex_varying) {
      auto result = UniformToVertex(vattr.get_data(), vattr.stride_bytes(),
                                    faceVertexCounts, faceVertexIndices);

      if (!result) {
        PUSH_ERROR_AND_RETURN(
            fmt::format("Convert `{}` attribute with uniform-varying "
                        "to vertex-varying failed: {}",
                        vattr.name, result.error()));
      }

      vattr.data = result.value();
      vattr.variability = VertexVariability::Vertex;

    } else {
      auto result = UniformToFaceVarying(vattr.get_data(), vattr.stride_bytes(),
                                         faceVertexCounts);
      if (!result) {
        PUSH_ERROR_AND_RETURN(
            fmt::format("Convert uniform `{}` attribute to failed: {}",
                        vattr.name, result.error()));
      }

      vattr.data = result.value();
      vattr.variability = VertexVariability::FaceVarying;
    }
  } else if (vattr.variability == VertexVariability::Constant) {
    if (to_vertex_varying) {
      auto result = ConstantToVertex(vattr.get_data(), vattr.stride_bytes(),
                                     faceVertexCounts, faceVertexIndices);

      if (!result) {
        PUSH_ERROR_AND_RETURN(
            fmt::format("Convert `{}` attribute with uniform-varying "
                        "to vertex-varying failed: {}",
                        vattr.name, result.error()));
      }

      vattr.data = result.value();
      vattr.variability = VertexVariability::Vertex;

    } else {
      auto result = UniformToFaceVarying(vattr.get_data(), vattr.stride_bytes(),
                                         faceVertexCounts);
      if (!result) {
        PUSH_ERROR_AND_RETURN(
            fmt::format("Convert uniform `{}` attribute to failed: {}",
                        vattr.name, result.error()));
      }

      vattr.data = result.value();
      vattr.variability = VertexVariability::FaceVarying;
    }

  } else if ((vattr.variability == VertexVariability::Vertex) ||
             (vattr.variability == VertexVariability::Varying)) {
    if (!to_vertex_varying) {
      auto result = VertexToFaceVarying(vattr.get_data(), vattr.stride_bytes(),
                                        faceVertexCounts, faceVertexIndices);
      if (!result) {
        PUSH_ERROR_AND_RETURN(
            fmt::format("Convert vertex/varying `{}` attribute to failed: {}",
                        vattr.name, result.error()));
      }

      vattr.data = result.value();
      vattr.variability = VertexVariability::FaceVarying;
    }

  } else if (vattr.variability == VertexVariability::FaceVarying) {
    if (to_vertex_varying) {
      PUSH_ERROR_AND_RETURN(
          "Internal error. `to_vertex_varying` should not be true when "
          "FaceVarying.");
    }

  } else {
    PUSH_ERROR_AND_RETURN(
        fmt::format("Unsupported/unimplemented interpolation: {} ",
                    to_string(vattr.variability)));
  }

  return true;
}

bool RenderSceneConverter::BuildVertexIndicesImpl(RenderMesh &mesh) {
  //
  // - If mesh is triangulated, use triangulatedFaceVertexIndices, otherwise use
  // faceVertxIndices.
  // - Make vertex attributes 'facevarying' variability
  // - Assign same id for similar(currently identitical) vertex attribute.
  // - Reorder vertex attributes to 'vertex' variability.
  //

  const std::vector<uint32_t> &fvIndices =
      mesh.triangulatedFaceVertexIndices.size()
          ? mesh.triangulatedFaceVertexIndices
          : mesh.usdFaceVertexIndices;

  DefaultVertexInput<DefaultPackedVertexData> vertex_input;

  size_t num_fvs = fvIndices.size();
  vertex_input.point_indices = fvIndices;
  vertex_input.uv0s.assign(num_fvs, {0.0f, 0.0f});
  vertex_input.uv1s.assign(num_fvs, {0.0f, 0.0f});
  vertex_input.normals.assign(num_fvs, {0.0f, 0.0f, 0.0f});
  vertex_input.tangents.assign(num_fvs, {0.0f, 0.0f, 0.0f});
  vertex_input.binormals.assign(num_fvs, {0.0f, 0.0f, 0.0f});
  vertex_input.colors.assign(num_fvs, {0.0f, 0.0f, 0.0f});
  vertex_input.opacities.assign(num_fvs, 0.0f);

  if (mesh.normals.vertex_count()) {
    if (!mesh.normals.is_facevarying()) {
      PUSH_ERROR_AND_RETURN(
          "Internal error. normals must be 'facevarying' variability.");
    }
    if (mesh.normals.vertex_count() != num_fvs) {
      PUSH_ERROR_AND_RETURN(
          "Internal error. The number of normal items does not match with "
          "the number of facevarying items.");
    }
  }

  const value::float2 *texcoord0_ptr = nullptr;
  const value::float2 *texcoord1_ptr = nullptr;

  for (const auto &it : mesh.texcoords) {
    if (it.second.vertex_count() > 0) {
      if (!it.second.is_facevarying()) {
        PUSH_ERROR_AND_RETURN(
            "Internal error. texcoords must be 'facevarying' variability.");
      }
      if (it.second.vertex_count() != num_fvs) {
        PUSH_ERROR_AND_RETURN(
            "Internal error. The number of texcoord items does not match "
            "with the number of facevarying items.");
      }

      if (it.first == 0) {
        texcoord0_ptr = reinterpret_cast<const value::float2 *>(
            it.second.get_data().data());
      } else if (it.first == 1) {
        texcoord1_ptr = reinterpret_cast<const value::float2 *>(
            it.second.get_data().data());
      } else {
        // ignore.
      }
    }
  }

  const value::float3 *tangents_ptr = nullptr;
  const value::float3 *binormals_ptr = nullptr;

  if (texcoord0_ptr) {
    if (mesh.tangents.vertex_count()) {
      if (!mesh.tangents.is_facevarying()) {
        PUSH_ERROR_AND_RETURN(
            "Internal error. tangents must be 'facevarying' variability.");
      }
      if (mesh.tangents.vertex_count() != num_fvs) {
        PUSH_ERROR_AND_RETURN(
            "Internal error. The number of tangents items does not match "
            "with the number of facevarying items.");
      }

      tangents_ptr = reinterpret_cast<const value::float3 *>(
          mesh.tangents.get_data().data());
    }

    if (mesh.binormals.vertex_count()) {
      if (!mesh.binormals.is_facevarying()) {
        PUSH_ERROR_AND_RETURN(
            "Internal error. binormals must be 'facevarying' variability.");
      }
      if (mesh.binormals.vertex_count() != num_fvs) {
        PUSH_ERROR_AND_RETURN(
            "Internal error. The number of binormals items does not match "
            "with the number of facevarying items.");
      }
      binormals_ptr = reinterpret_cast<const value::float3 *>(
          mesh.binormals.get_data().data());
    }
  }

  if (mesh.vertex_colors.vertex_count()) {
    if (!mesh.vertex_colors.is_facevarying()) {
      PUSH_ERROR_AND_RETURN(
          "Internal error. vertex_colors must be 'facevarying' variability.");
    }
    if (mesh.vertex_colors.vertex_count() != num_fvs) {
      PUSH_ERROR_AND_RETURN(
          "Internal error. The number of vertex_color items does not match "
          "with the number of facevarying items.");
    }
  }

  if (mesh.vertex_opacities.vertex_count()) {
    if (!mesh.vertex_opacities.is_facevarying()) {
      PUSH_ERROR_AND_RETURN(
          "Internal error. vertex_opacities must be 'facevarying' "
          "variability.");
    }
    if (mesh.vertex_colors.vertex_count() != num_fvs) {
      PUSH_ERROR_AND_RETURN(
          "Internal error. The number of vertex_opacity items does not match "
          "with the number of facevarying items.");
    }
  }

  const value::float3 *normals_ptr =
      (mesh.normals.vertex_count() > 0)
          ? reinterpret_cast<const value::float3 *>(
                mesh.normals.get_data().data())
          : nullptr;
  const value::float3 *colors_ptr =
      (mesh.vertex_colors.vertex_count() > 0)
          ? reinterpret_cast<const value::float3 *>(
                mesh.vertex_colors.get_data().data())
          : nullptr;
  const float *opacities_ptr =
      (mesh.vertex_opacities.vertex_count() > 0)
          ? reinterpret_cast<const float *>(
                mesh.vertex_opacities.get_data().data())
          : nullptr;

  for (size_t i = 0; i < num_fvs; i++) {
    size_t fvi = fvIndices[i];
    if (fvi >= num_fvs) {
      PUSH_ERROR_AND_RETURN(fmt::format(
          "Invalid faceVertexIndex {}. Must be less than {}", fvi, num_fvs));
    }

    if (normals_ptr) {
      vertex_input.normals[i] = normals_ptr[i];
    }
    if (texcoord0_ptr) {
      vertex_input.uv0s[i] = texcoord0_ptr[i];
    }
    if (texcoord1_ptr) {
      vertex_input.uv1s[i] = texcoord1_ptr[i];
    }
    if (tangents_ptr) {
      vertex_input.tangents[i] = tangents_ptr[i];
    }
    if (binormals_ptr) {
      vertex_input.binormals[i] = binormals_ptr[i];
    }
    if (colors_ptr) {
      vertex_input.colors[i] = colors_ptr[i];
    }
    if (opacities_ptr) {
      vertex_input.opacities[i] = opacities_ptr[i];
    }
  }

  std::vector<uint32_t> out_indices;
  std::vector<uint32_t> out_point_indices;  // to reorder position data
  DefaultVertexOutput<DefaultPackedVertexData> vertex_output;

  BuildIndices<DefaultVertexInput<DefaultPackedVertexData>,
               DefaultVertexOutput<DefaultPackedVertexData>,
               DefaultPackedVertexData, DefaultPackedVertexDataHasher,
               DefaultPackedVertexDataEqual>(vertex_input, vertex_output,
                                             out_indices, out_point_indices);

  if (out_indices.size() != out_point_indices.size()) {
    PUSH_ERROR_AND_RETURN(
        "Internal error. out_indices.size != out_point_indices.");
  }

  DCOUT("faceVertexIndices.size : " << fvIndices.size());
  DCOUT("# of indices after the build: "
        << out_indices.size() << ", reduced "
        << (fvIndices.size() - out_indices.size()) << " indices.");

  if (mesh.is_triangulated()) {
    mesh.triangulatedFaceVertexIndices = out_indices;
  } else {
    mesh.usdFaceVertexIndices = out_indices;
  }

  //
  // Reorder 'vertex' varying attributes(points, jointIndices/jointWeights,
  // BlendShape points, ...)
  // TODO: Preserve input order as much as possible.
  //
  {
    uint32_t numPoints =
        *std::max_element(out_indices.begin(), out_indices.end()) + 1;
    {
      std::vector<value::float3> tmp_points(numPoints);
      // TODO: Use vertex_output[i].point_index?
      for (size_t i = 0; i < out_point_indices.size(); i++) {
        if (out_point_indices[i] >= mesh.points.size()) {
          PUSH_ERROR_AND_RETURN("Internal error. point index out-of-range.");
        }
        tmp_points[out_indices[i]] = mesh.points[out_point_indices[i]];
      }
      mesh.points.swap(tmp_points);
    }

    if (mesh.joint_and_weights.jointIndices.size()) {
      if (mesh.joint_and_weights.elementSize < 1) {
        PUSH_ERROR_AND_RETURN(
            "Internal error. Invalid elementSize in mesh.joint_and_weights.");
      }
      uint32_t elementSize = uint32_t(mesh.joint_and_weights.elementSize);
      std::vector<int> tmp_indices(size_t(numPoints) * size_t(elementSize));
      std::vector<float> tmp_weights(size_t(numPoints) * size_t(elementSize));
      for (size_t i = 0; i < out_point_indices.size(); i++) {
        if ((elementSize * out_point_indices[i]) >=
            mesh.joint_and_weights.jointIndices.size()) {
          PUSH_ERROR_AND_RETURN(
              "Internal error. point index exceeds jointIndices.size.");
        }
        for (size_t k = 0; k < elementSize; k++) {
          tmp_indices[size_t(elementSize) * size_t(out_indices[i]) + k] =
              mesh.joint_and_weights
                  .jointIndices[size_t(elementSize) * size_t(out_point_indices[i]) + k];
        }

        if ((elementSize * out_point_indices[i]) >=
            mesh.joint_and_weights.jointWeights.size()) {
          PUSH_ERROR_AND_RETURN(
              "Internal error. point index exceeds jointWeights.size.");
        }

        for (size_t k = 0; k < elementSize; k++) {
          tmp_weights[size_t(elementSize) * size_t(out_indices[i]) + k] =
              mesh.joint_and_weights
                  .jointWeights[size_t(elementSize) * size_t(out_point_indices[i]) + k];
        }
      }
      mesh.joint_and_weights.jointIndices.swap(tmp_indices);
      mesh.joint_and_weights.jointWeights.swap(tmp_weights);
    }

    if (mesh.targets.size()) {
      // For BlendShape, reordering pointIndices, pointOffsets and normalOffsets is not enough.
      // Some points could be duplicated, so we need to find a mapping of org pointIdx -> pointIdx list in reordered points,
      // Then splat point attributes accordingly.

      // org pointIdx -> List of pointIdx in reordered points.
      std::unordered_map<uint32_t, std::vector<uint32_t>> pointIdxRemap;

      for (size_t i = 0; i < vertex_output.size(); i++) {
        pointIdxRemap[vertex_output.point_indices[i]].push_back(uint32_t(i));
      }

      for (auto &target : mesh.targets) {

        std::vector<value::float3> tmpPointOffsets;
        std::vector<value::float3> tmpNormalOffsets;
        std::vector<uint32_t> tmpPointIndices;

        for (size_t i = 0; i < target.second.pointIndices.size(); i++) {

          uint32_t orgPointIdx = target.second.pointIndices[i];
          if (!pointIdxRemap.count(orgPointIdx)) {
            PUSH_ERROR_AND_RETURN("Invalid pointIndices value.");
          }
          const std::vector<uint32_t> &dstPointIndices = pointIdxRemap.at(orgPointIdx);

          for (size_t k = 0; k < dstPointIndices.size(); k++) {
            if (target.second.pointOffsets.size()) {
              if (i >= target.second.pointOffsets.size()) {
                PUSH_ERROR_AND_RETURN("Invalid pointOffsets.size.");
              }
              tmpPointOffsets.push_back(target.second.pointOffsets[i]);
            }
            if (target.second.normalOffsets.size()) {
              if (i >= target.second.normalOffsets.size()) {
                PUSH_ERROR_AND_RETURN("Invalid normalOffsets.size.");
              }
              tmpNormalOffsets.push_back(target.second.normalOffsets[i]);
            }

            tmpPointIndices.push_back(dstPointIndices[k]);
          }
        }

        target.second.pointIndices.swap(tmpPointIndices);
        target.second.pointOffsets.swap(tmpPointOffsets);
        target.second.normalOffsets.swap(tmpNormalOffsets);

      }

      // TODO: Inbetween BlendShapes

    }

  }

  // Other 'facevarying' attributes are now 'vertex' variability
  if (normals_ptr) {
    mesh.normals.set_buffer(
        reinterpret_cast<const uint8_t *>(vertex_output.normals.data()),
        vertex_output.normals.size() * sizeof(value::float3));
    mesh.normals.variability = VertexVariability::Vertex;
  }

  if (texcoord0_ptr) {
    mesh.texcoords[0].set_buffer(
        reinterpret_cast<const uint8_t *>(vertex_output.uv0s.data()),
        vertex_output.uv0s.size() * sizeof(value::float2));
    mesh.texcoords[0].variability = VertexVariability::Vertex;
  }

  if (texcoord1_ptr) {
    mesh.texcoords[1].set_buffer(
        reinterpret_cast<const uint8_t *>(vertex_output.uv1s.data()),
        vertex_output.uv1s.size() * sizeof(value::float2));
    mesh.texcoords[1].variability = VertexVariability::Vertex;
  }

  if (tangents_ptr) {
    mesh.tangents.set_buffer(
        reinterpret_cast<const uint8_t *>(vertex_output.tangents.data()),
        vertex_output.tangents.size() * sizeof(value::float3));
    mesh.tangents.variability = VertexVariability::Vertex;
  }

  if (binormals_ptr) {
    mesh.binormals.set_buffer(
        reinterpret_cast<const uint8_t *>(vertex_output.binormals.data()),
        vertex_output.binormals.size() * sizeof(value::float3));
    mesh.binormals.variability = VertexVariability::Vertex;
  }

  if (colors_ptr) {
    mesh.vertex_colors.set_buffer(
        reinterpret_cast<const uint8_t *>(vertex_output.colors.data()),
        vertex_output.colors.size() * sizeof(value::float3));
    mesh.vertex_colors.variability = VertexVariability::Vertex;
  }

  if (opacities_ptr) {
    mesh.vertex_opacities.set_buffer(
        reinterpret_cast<const uint8_t *>(vertex_output.opacities.data()),
        vertex_output.opacities.size() * sizeof(float));
    mesh.vertex_opacities.variability = VertexVariability::Vertex;
  }

  return true;
}

bool RenderSceneConverter::ConvertMesh(
    const RenderSceneConverterEnv &env, const Path &abs_prim_path,
    const GeomMesh &mesh, const MaterialPath &material_path,
    const std::map<std::string, MaterialPath> &subset_material_path_map,
    // const std::map<std::string, int64_t> &rmaterial_idMap,
    const StringAndIdMap &rmaterial_map,
    const std::vector<const tinyusdz::GeomSubset *> &material_subsets,
    const std::vector<std::pair<std::string, const tinyusdz::BlendShape *>>
        &blendshapes,
    RenderMesh *dstMesh) {
  //
  // Steps:
  //
  // 1. Get points, faceVertexIndices and faceVertexOffsets at specified time.
  //   - Validate GeomSubsets
  // 2. Assign Material and list up texcoord primvars
  // 3. convert texcoord, normals, vetexcolor(displaycolors)
  //   - First try to convert it to `vertex` varying(Can be drawn with single
  //   index buffer)
  //   - Otherwise convert to `facevarying` as the last resort.
  // 4. Triangulate indices  when `triangulate` is enabled.
  //   - Triangulate texcoord, normals, vertexcolor.
  // 5. Convert Skin weights
  // 6. Convert BlendShape
  // 7. Build indices(convert 'facevarying' to 'vertrex')
  // 8. Calcualte normals(if not present in the mesh)
  // 9. Build tangent frame(for normal mapping)
  //
  //

  if (!dstMesh) {
    PUSH_ERROR_AND_RETURN("`dst` mesh pointer is nullptr");
  }

  RenderMesh dst;

  dst.is_rightHanded =
      (mesh.orientation.get_value() == tinyusdz::Orientation::RightHanded);
  dst.doubleSided = mesh.doubleSided.get_value();

  //
  // 1. Mandatory attribute: points, faceVertexCounts and faceVertexIndices.
  //
  // TODO: Make error when Mesh's indices is empty?
  //

  {
    std::vector<value::point3f> points;
    bool ret = EvaluateTypedAnimatableAttribute(
        env.stage, mesh.points, "points", &points, &_err, env.timecode,
        value::TimeSampleInterpolationType::Linear);
    if (!ret) {
      return false;
    }

    if (points.empty()) {
      PUSH_ERROR_AND_RETURN(
          fmt::format("`points` is empty. Prim {}", abs_prim_path));
    }

    dst.points.resize(points.size());
    memcpy(dst.points.data(), points.data(),
           sizeof(value::float3) * points.size());
  }

  {
    std::vector<int32_t> indices;
    bool ret = EvaluateTypedAnimatableAttribute(
        env.stage, mesh.faceVertexIndices, "faceVertexIndices", &indices, &_err,
        env.timecode, value::TimeSampleInterpolationType::Held);
    if (!ret) {
      return false;
    }

    for (size_t i = 0; i < indices.size(); i++) {
      if (indices[i] < 0) {
        PUSH_ERROR_AND_RETURN(fmt::format(
            "faceVertexIndices[{}] contains negative index value {}.", i,
            indices[i]));
      }
      if (size_t(indices[i]) > dst.points.size()) {
        PUSH_ERROR_AND_RETURN(
            fmt::format("faceVertexIndices[{}] {} exceeds points.size {}.", i,
                        indices[i], dst.points.size()));
      }
      dst.usdFaceVertexIndices.push_back(uint32_t(indices[i]));
    }
  }

  {
    std::vector<int> counts;
    bool ret = EvaluateTypedAnimatableAttribute(
        env.stage, mesh.faceVertexCounts, "faceVertexCounts", &counts, &_err,
        env.timecode, value::TimeSampleInterpolationType::Held);
    if (!ret) {
      return false;
    }

    size_t sumCounts = 0;
    dst.usdFaceVertexCounts.clear();
    for (size_t i = 0; i < counts.size(); i++) {
      if (counts[i] < 3) {
        PUSH_ERROR_AND_RETURN(
            fmt::format("faceVertexCounts[{}] contains invalid value {}. The "
                        "count value must be >= 3",
                        i, counts[i]));
      }

      if ((sumCounts + size_t(counts[i])) > dst.usdFaceVertexIndices.size()) {
        PUSH_ERROR_AND_RETURN(fmt::format(
            "faceVertexCounts[{}] exceeds faceVertexIndices.size {}.", i,
            dst.usdFaceVertexIndices.size()));
      }
      dst.usdFaceVertexCounts.push_back(uint32_t(counts[i]));
      sumCounts += size_t(counts[i]);
    }
  }

  //
  // 2. bindMaterial GeoMesh and GeomSubset.
  //
  // Assume Material conversion is done before ConvertMesh.
  // Here we only assign RenderMaterial id and extract GeomSubset::indices
  // information.
  //

  DCOUT("rmaterial_ap.size " << rmaterial_map.size());
  if (rmaterial_map.count(material_path.material_path)) {
    dst.material_id = int(rmaterial_map.at(material_path.material_path));
  }

  if (rmaterial_map.count(material_path.backface_material_path)) {
    dst.backface_material_id =
        int(rmaterial_map.at(material_path.backface_material_path));
  }

  if (env.mesh_config.validate_geomsubset) {
    size_t elementCount = dst.usdFaceVertexCounts.size();

    if (material_subsets.size() &&
        mesh.subsetFamilyTypeMap.count(value::token("materialBind"))) {
      const GeomSubset::FamilyType familyType =
          mesh.subsetFamilyTypeMap.at(value::token("materialBind"));
      if (!GeomSubset::ValidateSubsets(material_subsets, elementCount,
                                       familyType, &_err)) {
        PUSH_ERROR_AND_RETURN("GeomSubset validation failed.");
      }
    }
  }

  for (const auto &psubset : material_subsets) {
    MaterialSubset ms;
    ms.prim_name = psubset->name;
    // ms.prim_index = // TODO
    ms.abs_path = abs_prim_path.prim_part() + std::string("/") + psubset->name;
    ms.display_name = psubset->meta.displayName.value_or("");

    // TODO: Raise error when indices is empty?
    if (psubset->indices.authored()) {
      std::vector<int> indices;  // index to faceVertexCounts
      bool ret = EvaluateTypedAnimatableAttribute(
          env.stage, psubset->indices, "indices", &indices, &_err, env.timecode,
          value::TimeSampleInterpolationType::Held);
      if (!ret) {
        return false;
      }

      ms.usdIndices = indices;
    }

    if (subset_material_path_map.count(psubset->name)) {
      const auto &mp = subset_material_path_map.at(psubset->name);
      if (rmaterial_map.count(mp.material_path)) {
        ms.material_id = int(rmaterial_map.at(mp.material_path));
        DCOUT("MaterialSubset " << psubset->name << " : material_id "
                                << ms.material_id);
      }
      if (rmaterial_map.count(mp.backface_material_path)) {
        ms.backface_material_id =
            int(rmaterial_map.at(mp.backface_material_path));
        DCOUT("MaterialSubset " << psubset->name << " : backface_material_id "
                                << ms.backface_material_id);
      }
    }

    // TODO: Ensure prim_name is unique.
    dst.material_subsetMap[ms.prim_name] = ms;
  }

  uint32_t num_vertices = uint32_t(dst.points.size());
  uint32_t num_faces = uint32_t(dst.usdFaceVertexCounts.size());
  uint32_t num_face_vertex_indices = uint32_t(dst.usdFaceVertexIndices.size());

  //
  // List up texcoords in this mesh.
  // - If no material assigned to this mesh, look into
  // `default_texcoords_primvar_name`
  // - If materials are assigned, find all corresponding UV primvars in this
  // mesh.
  //

  // key:slotId, value:texcoord data
  std::unordered_map<uint32_t, VertexAttribute> uvAttrs;

  // We need Material info to get corresponding primvar name.
  if (rmaterial_map.empty()) {
    // No material assigned to the Mesh, but we may still want texcoords solely(
    // assign material after the conversion)
    // So find a primvar whose name matches default texcoord name.
    if (mesh.has_primvar(env.mesh_config.default_texcoords_primvar_name)) {
      DCOUT("uv primvar  with default_texcoords_primvar_name found.");
      auto ret = GetTextureCoordinate(
          env.stage, mesh, env.mesh_config.default_texcoords_primvar_name,
          env.timecode, env.tinterp);
      if (ret) {
        const VertexAttribute &vattr = ret.value();

        // Use slotId 0
        uvAttrs[0] = vattr;
      } else {
        PUSH_WARN("Failed to get texture coordinate for `"
                  << env.mesh_config.default_texcoords_primvar_name
                  << "` : " << ret.error());
      }
    }
  } else {
    for (auto mit = rmaterial_map.i_begin(); mit != rmaterial_map.i_end();
         mit++) {
      int64_t rmaterial_id = int64_t(mit->first);

      if ((rmaterial_id > -1) && (size_t(rmaterial_id) < materials.size())) {
        const RenderMaterial &material = materials[size_t(rmaterial_id)];

        StringAndIdMap uvname_map;
        if (!ListUVNames(material, textures, uvname_map)) {
          DCOUT("Failed to list UV names");
          return false;
        }

        for (auto it = uvname_map.i_begin(); it != uvname_map.i_end(); it++) {
          uint64_t slotId = it->first;
          std::string uvname = it->second;

          if (!uvAttrs.count(uint32_t(slotId))) {
            // FIXME: Use GetGeomPrimvar() & ToVertexAttribute()
            auto ret = GetTextureCoordinate(env.stage, mesh, uvname,
                                            env.timecode, env.tinterp);
            if (ret) {
              const VertexAttribute &vattr = ret.value();

              if (vattr.is_vertex()) {
                if (vattr.vertex_count() != num_vertices) {
                  PUSH_ERROR_AND_RETURN(fmt::format("Array length of texture coordinate `{}`(Prim path {}) must be {}, but got {}", uvname, abs_prim_path.prim_part(), num_vertices, vattr.vertex_count()));
                }
              } else if (vattr.is_constant()) {
                if (vattr.vertex_count() != 1) {
                  PUSH_ERROR_AND_RETURN(fmt::format("Array length of texture coordinate `{}`(Prim path {}) must be {}, but got {}", uvname, abs_prim_path.prim_part(), 1, vattr.vertex_count()));
                }
              } else if (vattr.is_uniform()) {
                if (vattr.vertex_count() != num_faces) {
                  PUSH_ERROR_AND_RETURN(fmt::format("Array length of texture coordinate `{}`(Prim path {}) must be {}, but got {}", uvname, abs_prim_path.prim_part(), num_faces, vattr.vertex_count()));
                }
              } else if (vattr.is_facevarying()) {
                if (vattr.vertex_count() != num_face_vertex_indices) {
                  PUSH_ERROR_AND_RETURN(fmt::format("Array length of texture coordinate `{}`(Prim path {}) must be {}, but got {}", uvname, abs_prim_path.prim_part(), num_face_vertex_indices, vattr.vertex_count()));
                }
              } else {
                PUSH_ERROR_AND_RETURN("Internal error. Unknown variability of texcoord attribute.");
                return false;
              }

              uvAttrs[uint32_t(slotId)] = vattr;
            } else {
              PUSH_WARN("Failed to get texture coordinate for `"
                        << uvname << "` : " << ret.error());
            }
          }
        }
      }
    }
  }


  if (mesh.has_primvar(env.mesh_config.default_tangents_primvar_name)) {
    GeomPrimvar pvar;

    if (!GetGeomPrimvar(env.stage, &mesh,
                        env.mesh_config.default_tangents_primvar_name, &pvar,
                        &_err)) {
      return false;
    }

    if (!ToVertexAttribute(pvar, env.mesh_config.default_tangents_primvar_name,
                           num_vertices, num_faces, num_face_vertex_indices,
                           dst.tangents, &_err, env.timecode, env.tinterp)) {
      return false;
    }
  }

  if (mesh.has_primvar(env.mesh_config.default_binormals_primvar_name)) {
    GeomPrimvar pvar;

    if (!GetGeomPrimvar(env.stage, &mesh,
                        env.mesh_config.default_binormals_primvar_name, &pvar,
                        &_err)) {
      return false;
    }

    if (!ToVertexAttribute(pvar, env.mesh_config.default_binormals_primvar_name,
                           num_vertices, num_faces, num_face_vertex_indices,
                           dst.binormals, &_err, env.timecode, env.tinterp)) {
      return false;
    }
  }

  constexpr auto kDisplayColor = "displayColor";
  if (mesh.has_primvar(kDisplayColor)) {
    GeomPrimvar pvar;

    if (!GetGeomPrimvar(env.stage, &mesh, kDisplayColor, &pvar, &_err)) {
      return false;
    }

    VertexAttribute vcolor;
    if (!ToVertexAttribute(pvar, kDisplayColor, num_vertices, num_faces,
                           num_face_vertex_indices, vcolor, &_err, env.timecode,
                           env.tinterp)) {
      return false;
    }

    if ((vcolor.elementSize == 1) && (vcolor.vertex_count() == 1) &&
        (vcolor.stride_bytes() == 3 * 4)) {
      memcpy(&dst.displayColor, vcolor.data.data(), vcolor.stride_bytes());
    } else {
      dst.vertex_colors = vcolor;
    }
  }

  constexpr auto kDisplayOpacity = "displayOpacity";
  if (mesh.has_primvar(kDisplayOpacity)) {
    GeomPrimvar pvar;
    if (!GetGeomPrimvar(env.stage, &mesh, kDisplayOpacity, &pvar, &_err)) {
      return false;
    }

    VertexAttribute vopacity;
    if (!ToVertexAttribute(pvar, kDisplayOpacity, num_vertices, num_faces,
                           num_face_vertex_indices, vopacity, &_err,
                           env.timecode, env.tinterp)) {
      return false;
    }

    if ((vopacity.elementSize == 1) && (vopacity.vertex_count() == 1) &&
        (vopacity.stride_bytes() == 4)) {
      memcpy(&dst.displayOpacity, vopacity.data.data(),
             vopacity.stride_bytes());
    } else {
      dst.vertex_opacities = vopacity;
    }
  }

  //
  // Check if the Mesh can be drawn with single index buffer during converting
  // normals/texcoords/displayColors/displayOpacities, since OpenGL and Vulkan
  // does not support drawing a primitive with multiple index buffers.
  //
  // If the Mesh contains any face-varying attribute,
  // First try to convert it 'vertex' variabily, if it fails, all attribute are
  // converted to face-varying so that the Mesh can be drawn without index
  // buffer. This will hurt the performance of rendering in OpenGL/Vulkan,
  // especially when the Mesh is animated with skinning.
  //
  // We leave user-defined primvar as-is, so no check for it.
  //
  bool is_single_indexable{true};

  //
  // Convert normals
  //
  {
    Interpolation interp = mesh.get_normalsInterpolation();
    std::vector<value::normal3f> normals;

    if (mesh.has_primvar("normals")) {  // primvars:normals
      GeomPrimvar pvar;
      if (!GetGeomPrimvar(env.stage, &mesh, "normals", &pvar, &_err)) {
        return false;
      }

      if (!pvar.flatten_with_indices(env.timecode, &normals, env.tinterp,
                                     &_err)) {
        PUSH_ERROR_AND_RETURN("Failed to expand `normals` primvar.");
      }

    } else if (mesh.normals.authored()) {  // look 'normals'
      if (!EvaluateTypedAnimatableAttribute(env.stage, mesh.normals, "normals",
                                            &normals, &_err, env.timecode,
                                            env.tinterp)) {
      }
    }

    dst.normals.get_data().resize(normals.size() * sizeof(value::normal3f));
    memcpy(dst.normals.get_data().data(), normals.data(),
           normals.size() * sizeof(value::normal3f));
    dst.normals.elementSize = 1;
    dst.normals.stride = sizeof(value::normal3f);
    dst.normals.format = VertexAttributeFormat::Vec3;

    if (interp == Interpolation::Varying) {
      dst.normals.variability = VertexVariability::Varying;
    } else if (interp == Interpolation::Constant) {
      dst.normals.variability = VertexVariability::Constant;
    } else if (interp == Interpolation::Uniform) {
      dst.normals.variability = VertexVariability::Uniform;
    } else if (interp == Interpolation::Vertex) {
      dst.normals.variability = VertexVariability::Vertex;
    } else if (interp == Interpolation::FaceVarying) {
      dst.normals.variability = VertexVariability::FaceVarying;
    } else {
      PUSH_ERROR_AND_RETURN(
          "[Internal Error] Invalid interpolation value for normals.");
    }
    dst.normals.indices.clear();
    dst.normals.name = "normals";

    if (is_single_indexable &&
        (dst.normals.variability == VertexVariability::FaceVarying)) {
      VertexAttribute va_normals;
      if (TryConvertFacevaryingToVertex(
              dst.normals, &va_normals, dst.usdFaceVertexIndices, &_warn,
              env.mesh_config.facevarying_to_vertex_eps)) {
        DCOUT("normals is converted to 'vertex' varying.");
        dst.normals = std::move(va_normals);
      } else {
        DCOUT(
            "normals cannot be converted to 'vertex' varying. Staying "
            "'facevarying'");
        DCOUT("warn = " << _warn);
        is_single_indexable = false;
      }
    }
  }

  //
  // Convert UVs
  //

  for (const auto &it : uvAttrs) {
    uint64_t slotId = it.first;
    const VertexAttribute &vattr = it.second;

    if (vattr.format != VertexAttributeFormat::Vec2) {
      PUSH_ERROR_AND_RETURN(
          fmt::format("Texcoord VertexAttribute must be Vec2 type.\n"));
    }

    if (vattr.element_size() != 1) {
      PUSH_ERROR_AND_RETURN(
          fmt::format("elementSize must be 1 for Texcoord attribute."));
    }

    DCOUT("Add texcoord attr `" << vattr.name << "` to slot Id " << slotId);

    if (is_single_indexable &&
        (vattr.variability == VertexVariability::FaceVarying)) {
      VertexAttribute va_uvs;
      if (TryConvertFacevaryingToVertex(
              vattr, &va_uvs, dst.usdFaceVertexIndices, &_warn,
              env.mesh_config.facevarying_to_vertex_eps)) {
        DCOUT("texcoord[" << slotId << "] is converted to 'vertex' varying.");
        dst.texcoords[uint32_t(slotId)] = va_uvs;
      } else {
        DCOUT("texcoord[" << slotId
                          << "] cannot be converted to 'vertex' varying. "
                             "Staying 'facevarying'");
        is_single_indexable = false;
        dst.texcoords[uint32_t(slotId)] = vattr;
      }
    } else {
      dst.texcoords[uint32_t(slotId)] = vattr;
    }
  }

  if (dst.vertex_colors.vertex_count() > 1) {
    VertexAttribute vattr = dst.vertex_colors;  // copy

    if (vattr.format != VertexAttributeFormat::Vec3) {
      PUSH_ERROR_AND_RETURN(
          fmt::format("Color VertexAttribute must be Vec3 type.\n"));
    }

    if (vattr.element_size() != 1) {
      PUSH_ERROR_AND_RETURN(
          fmt::format("elementSize = 1 expected for VertexColor, but got {}",
                      vattr.element_size()));
    }

    if (is_single_indexable &&
        (dst.vertex_colors.variability == VertexVariability::FaceVarying)) {
      VertexAttribute va;
      if (TryConvertFacevaryingToVertex(
              dst.vertex_colors, &va, dst.usdFaceVertexIndices, &_warn,
              env.mesh_config.facevarying_to_vertex_eps)) {
        dst.vertex_colors = std::move(va);
      } else {
        DCOUT(
            "vertex_colors cannot be converted to 'vertex' varying. Staying "
            "'facevarying'");
        is_single_indexable = false;
      }
    }
  }

  if (dst.vertex_opacities.vertex_count() > 1) {
    VertexAttribute vattr = dst.vertex_opacities;  // copy

    if (vattr.format != VertexAttributeFormat::Float) {
      PUSH_ERROR_AND_RETURN(
          fmt::format("Opacity VertexAttribute must be Float type.\n"));
    }

    if (vattr.element_size() != 1) {
      PUSH_ERROR_AND_RETURN(
          fmt::format("elementSize = 1 expected for VertexOpacity, but got {}",
                      vattr.element_size()));
    }

    if (is_single_indexable &&
        (dst.vertex_opacities.variability == VertexVariability::FaceVarying)) {
      VertexAttribute va;
      if (TryConvertFacevaryingToVertex(
              dst.vertex_opacities, &va, dst.usdFaceVertexIndices, &_warn,
              env.mesh_config.facevarying_to_vertex_eps)) {
        dst.vertex_opacities = std::move(va);
      } else {
        DCOUT(
            "vertex_opacities cannot be converted to 'vertex' varying. Staying "
            "'facevarying'");
        is_single_indexable = false;
      }
    }
  }

  DCOUT(mesh.name << " : is_single_indexable = " << is_single_indexable);

  //
  // Convert built-in vertex attributes to either 'vertex' or 'facevarying'
  //
  {
    if (!ConvertVertexVariabilityImpl(dst.normals, is_single_indexable,
                                      dst.usdFaceVertexCounts,
                                      dst.usdFaceVertexIndices)) {
      return false;
    }
    for (auto &it : dst.texcoords) {
      if (!ConvertVertexVariabilityImpl(it.second, is_single_indexable,
                                        dst.usdFaceVertexCounts,
                                        dst.usdFaceVertexIndices)) {
        return false;
      }
    }

    if (!ConvertVertexVariabilityImpl(dst.vertex_colors, is_single_indexable,
                                      dst.usdFaceVertexCounts,
                                      dst.usdFaceVertexIndices)) {
      return false;
    }
    if (!ConvertVertexVariabilityImpl(dst.vertex_opacities, is_single_indexable,
                                      dst.usdFaceVertexCounts,
                                      dst.usdFaceVertexIndices)) {
      return false;
    }
  }

  ///
  /// 4. Triangulate
  ///  - triangulate faceVertexCounts, faceVertexIndices
  ///  - Remap faceIndex in MaterialSubset(GeomSubset).
  ///  - Triangulate vertex attributes(normals, uvcoords, vertex
  ///  colors/opacities).
  ///
  bool triangulate = env.mesh_config.triangulate;
  if (triangulate) {
    DCOUT("Triangulate mesh");
    std::vector<uint32_t> triangulatedFaceVertexCounts;  // should be all 3's
    std::vector<uint32_t> triangulatedFaceVertexIndices;
    std::vector<size_t>
        triangulatedToOrigFaceVertexIndexMap;  // used for rearrange facevertex
                                               // attrib
    std::vector<uint32_t>
        triangulatedFaceCounts;  // used for rearrange face indices(e.g
                                 // GeomSubset indices)

    std::string err;

    if (!TriangulatePolygon<value::float3, float>(
            dst.points, dst.usdFaceVertexCounts, dst.usdFaceVertexIndices,
            triangulatedFaceVertexCounts, triangulatedFaceVertexIndices,
            triangulatedToOrigFaceVertexIndexMap, triangulatedFaceCounts,
            err)) {
      PUSH_ERROR_AND_RETURN("Triangulation failed: " + err);
    }

    if (dst.material_subsetMap.size()) {
      // Remap faceId in GeomSubsets

      //
      // size: len(triangulatedFaceCounts)
      // value: array index in triangulatedFaceVertexCounts
      // Up to 4GB faces.
      //
      std::vector<uint32_t> faceIndexOffsets;
      faceIndexOffsets.resize(triangulatedFaceCounts.size());

      size_t faceIndexOffset = 0;
      for (size_t i = 0; i < triangulatedFaceCounts.size(); i++) {
        size_t ncount = triangulatedFaceCounts[i];

        faceIndexOffsets[i] = uint32_t(faceIndexOffset);
        // DCOUT("faceIndexOffset[" << i << "] = " << faceIndexOffsets[i]);

        faceIndexOffset += ncount;

        if (faceIndexOffset >= std::numeric_limits<uint32_t>::max()) {
          PUSH_ERROR_AND_RETURN("Triangulated Mesh contains 4G or more faces.");
        }
      }

      // Remap indices in MaterialSubset
      //
      // example:
      //
      // faceVertexCounts = [4, 4]
      // faceVertexIndices = [0, 1, 2, 3, 4, 5, 6, 7]
      //
      // triangulatedFaceVertexCounts = [3, 3, 3, 3]
      // triangulatedFaceVertexIndices = [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]
      // triangulatedFaceCounts = [2, 2]
      //
      // geomsubset.indices = [0, 1] # index to faceVertexCounts
      // faceIndexOffsets = [0, 2]
      //
      // => triangulated geomsubset.indices = [0, 1, 2, 3] # index to
      // triangulatedFaceVertexCounts
      //
      //
      for (auto &it : dst.material_subsetMap) {
        std::vector<int> triangulated_indices;

        for (size_t i = 0; i < it.second.usdIndices.size(); i++) {
          int32_t srcIndex = it.second.usdIndices[i];
          if (srcIndex < 0) {
            PUSH_ERROR_AND_RETURN("Invalid index value in GeomSubset.");
          }

          uint32_t baseFaceIndex = faceIndexOffsets[size_t(srcIndex)];
          // DCOUT(i << ", baseFaceIndex = " << baseFaceIndex);

          for (size_t k = 0; k < triangulatedFaceCounts[uint32_t(srcIndex)];
               k++) {
            if ((baseFaceIndex + k) > size_t((std::numeric_limits<int32_t>::max)())) {
              PUSH_ERROR_AND_RETURN(fmt::format("Index value exceeds 2GB."));
            }
            // assume triangulated faceIndex in each polygon is monotonically
            // increasing.
            triangulated_indices.push_back(int(baseFaceIndex + k));
          }
        }

        it.second.triangulatedIndices = std::move(triangulated_indices);
      }
    }

    //
    // Triangulate built-in vertex attributes.
    //
    {
      if (!TriangulateVertexAttribute(dst.normals, dst.usdFaceVertexCounts,
                                      triangulatedToOrigFaceVertexIndexMap,
                                      triangulatedFaceCounts,
                                      triangulatedFaceVertexIndices, &_err)) {
        PUSH_ERROR_AND_RETURN("Failed to triangulate normals attribute.");
      }

      if (!TriangulateVertexAttribute(dst.tangents, dst.usdFaceVertexCounts,
                                      triangulatedToOrigFaceVertexIndexMap,
                                      triangulatedFaceCounts,
                                      triangulatedFaceVertexIndices, &_err)) {
        PUSH_ERROR_AND_RETURN("Failed to triangulate tangents attribute.");
      }

      if (!TriangulateVertexAttribute(dst.binormals, dst.usdFaceVertexCounts,
                                      triangulatedToOrigFaceVertexIndexMap,
                                      triangulatedFaceCounts,
                                      triangulatedFaceVertexIndices, &_err)) {
        PUSH_ERROR_AND_RETURN("Failed to triangulate binormals attribute.");
      }

      for (auto &it : dst.texcoords) {
        if (!TriangulateVertexAttribute(it.second, dst.usdFaceVertexCounts,
                                        triangulatedToOrigFaceVertexIndexMap,
                                        triangulatedFaceCounts,
                                        triangulatedFaceVertexIndices, &_err)) {
          PUSH_ERROR_AND_RETURN(fmt::format(
              "Failed to triangulate texcoords[{}] attribute.", it.first));
        }
      }

      if (!TriangulateVertexAttribute(
              dst.vertex_colors, dst.usdFaceVertexCounts,
              triangulatedToOrigFaceVertexIndexMap, triangulatedFaceCounts,
              triangulatedFaceVertexIndices, &_err)) {
        PUSH_ERROR_AND_RETURN("Failed to triangulate vertex_colors attribute.");
      }

      if (!TriangulateVertexAttribute(
              dst.vertex_opacities, dst.usdFaceVertexCounts,
              triangulatedToOrigFaceVertexIndexMap, triangulatedFaceCounts,
              triangulatedFaceVertexIndices, &_err)) {
        PUSH_ERROR_AND_RETURN(
            "Failed to triangulate vertopacitiesex_colors attribute.");
      }
    }

    dst.triangulatedFaceVertexCounts = std::move(triangulatedFaceVertexCounts);
    dst.triangulatedFaceVertexIndices =
        std::move(triangulatedFaceVertexIndices);

    dst.triangulatedToOrigFaceVertexIndexMap =
        std::move(triangulatedToOrigFaceVertexIndexMap);
    dst.triangulatedFaceCounts = std::move(triangulatedFaceCounts);
  }

  //
  // 5. Vertex skin weights(jointIndex and jointWeights)
  //
  if (mesh.has_primvar("skel:jointIndices") &&
      mesh.has_primvar("skel:jointWeights")) {
    DCOUT("Convert skin weights");
    GeomPrimvar jointIndices;
    GeomPrimvar jointWeights;

    if (!GetGeomPrimvar(env.stage, &mesh, "skel:jointIndices", &jointIndices,
                        &_err)) {
      return false;
    }

    if (!GetGeomPrimvar(env.stage, &mesh, "skel:jointWeights", &jointWeights,
                        &_err)) {
      return false;
    }

    // interpolation must be 'vertex'
    if (!jointIndices.has_interpolation()) {
      PUSH_ERROR_AND_RETURN(
          fmt::format("`skel:jointIndices` primvar must author `interpolation` "
                      "metadata(and set it to `vertex`)"));
    }

    // TODO: Disallow Varying?
    if ((jointIndices.get_interpolation() != Interpolation::Vertex) &&
        (jointIndices.get_interpolation() != Interpolation::Varying)) {
      PUSH_ERROR_AND_RETURN(
          fmt::format("`skel:jointIndices` primvar must use `vertex` for "
                      "`interpolation` metadata, but got `{}`.",
                      to_string(jointIndices.get_interpolation())));
    }

    if (!jointWeights.has_interpolation()) {
      PUSH_ERROR_AND_RETURN(
          fmt::format("`skel:jointWeights` primvar must author `interpolation` "
                      "metadata(and set it to `vertex`)"));
    }

    // TODO: Disallow Varying?
    if ((jointWeights.get_interpolation() != Interpolation::Vertex) &&
        (jointWeights.get_interpolation() != Interpolation::Varying)) {
      PUSH_ERROR_AND_RETURN(
          fmt::format("`skel:jointWeights` primvar must use `vertex` for "
                      "`interpolation` metadata, but got `{}`.",
                      to_string(jointWeights.get_interpolation())));
    }

    uint32_t jointIndicesElementSize = jointIndices.get_elementSize();
    uint32_t jointWeightsElementSize = jointWeights.get_elementSize();

    if (jointIndicesElementSize == 0) {
      PUSH_ERROR_AND_RETURN(fmt::format(
          "`elementSize` metadata of `skel:jointIndices` is zero."));
    }

    if (jointWeightsElementSize == 0) {
      PUSH_ERROR_AND_RETURN(fmt::format(
          "`elementSize` metadata of `skel:jointWeights` is zero."));
    }

    if (jointIndicesElementSize > env.mesh_config.max_skin_elementSize) {
      PUSH_ERROR_AND_RETURN(fmt::format(
          "`elementSize` {} of `skel:jointIndices` too large. Max allowed is "
          "set to {}",
          jointIndicesElementSize, env.mesh_config.max_skin_elementSize));
    }

    if (jointWeightsElementSize > env.mesh_config.max_skin_elementSize) {
      PUSH_ERROR_AND_RETURN(fmt::format(
          "`elementSize` {} of `skel:jointWeights` too large. Max allowed is "
          "set to {}",
          jointWeightsElementSize, env.mesh_config.max_skin_elementSize));
    }

    if (jointIndicesElementSize != jointWeightsElementSize) {
      PUSH_ERROR_AND_RETURN(
          fmt::format("`elementSize` {} of `skel:jointIndices` must equal to "
                      "`elementSize` {} of `skel:jointWeights`",
                      jointIndicesElementSize, jointWeightsElementSize));
    }

    std::vector<int> jointIndicesArray;
    if (!jointIndices.flatten_with_indices(env.timecode, &jointIndicesArray,
                                           env.tinterp)) {
      PUSH_ERROR_AND_RETURN(
          fmt::format("Failed to flatten Indexed Primvar `skel:jointIndices`. "
                      "Ensure `skel:jointIndices` is type `int[]`"));
    }

    std::vector<float> jointWeightsArray;
    if (!jointWeights.flatten_with_indices(env.timecode, &jointWeightsArray,
                                           env.tinterp)) {
      PUSH_ERROR_AND_RETURN(
          fmt::format("Failed to flatten Indexed Primvar `skel:jointWeights`. "
                      "Ensure `skel:jointWeights` is type `float[]`"));
    }

    if (jointIndicesArray.size() != jointWeightsArray.size()) {
      PUSH_ERROR_AND_RETURN(
          fmt::format("`skel:jointIndices` nitems {} must be equal to "
                      "`skel:jointWeights` ntems {}",
                      jointIndicesArray.size(), jointWeightsArray.size()));
    }

    if (jointIndicesArray.empty()) {
      PUSH_ERROR_AND_RETURN(fmt::format("`skel:jointIndices` is empty array."));
    }

    // TODO: Validate jointIndex.

    dst.joint_and_weights.jointIndices = jointIndicesArray;
    dst.joint_and_weights.jointWeights = jointWeightsArray;
    dst.joint_and_weights.elementSize = int(jointIndicesElementSize);

    if (mesh.skeleton.has_value()) {
      DCOUT("Convert Skeleton");
      Path skelPath;

      if (mesh.skeleton.value().is_path()) {
        skelPath = mesh.skeleton.value().targetPath;
      } else if (mesh.skeleton.value().is_pathvector()) {
        // Use the first tone
        if (mesh.skeleton.value().targetPathVector.size()) {
          skelPath = mesh.skeleton.value().targetPathVector[0];
        } else {
          PUSH_WARN("`skel:skeleton` has invalid definition.");
        }
      } else {
        PUSH_WARN("`skel:skeleton` has invalid definition.");
      }

      if (skelPath.is_valid()) {
        SkelHierarchy skel;
        nonstd::optional<Animation> anim;
        // TODO: cache skeleton conversion
        if (!ConvertSkeletonImpl(env, mesh, &skel, &anim)) {
          return false;
        }
        DCOUT("Converted skeleton attached to : " << abs_prim_path);

        auto skel_it = std::find_if(skeletons.begin(), skeletons.end(), [&skelPath](const SkelHierarchy &sk) {
          DCOUT("sk.abs_path " << sk.abs_path << ", skel_path " << skelPath.full_path_name());
          return sk.abs_path == skelPath.full_path_name();
        });

        if (anim) {

          const auto &animAbsPath = anim.value().abs_path;
          auto anim_it = std::find_if(animations.begin(), animations.end(), [&animAbsPath](const Animation &a) {
            DCOUT("a.abs_path " << a.abs_path << ", anim_path " << animAbsPath);
            return a.abs_path == animAbsPath;
          });

          if (anim_it != animations.end()) {
            skel.anim_id = int(std::distance(animations.begin(), anim_it));
          } else {
            skel.anim_id = int(animations.size());
            animations.emplace_back(anim.value());
          }
        }

        int skel_id{0};
        if (skel_it != skeletons.end()) {
          skel_id = int(std::distance(skeletons.begin(), skel_it));
        } else {
          skel_id = int(skeletons.size());
          skeletons.emplace_back(std::move(skel));
          DCOUT("add skeleton\n");
        }

        dst.skel_id = skel_id;

      }
    }

    // Explicit joint orders
    // If the mesh has `skel:joints`, remap jointIndex.
    {
      std::vector<value::token> joints = mesh.get_joints();
      //if ((dst.skel_id >= 0) && joints.size()) {
      //  DCOUT("has explicit joint orders.\n");
      //}

      const auto &skel = skeletons[size_t(dst.skel_id)];

      std::map<std::string, int> name_to_index_map = BuildSkelNameToIndexMap(skel);

      std::unordered_map<int, int> index_remap;

      for (size_t i = 0; i < joints.size(); i++) {
        std::string joint_name = joints[i].str();
        
        if (!name_to_index_map.count(joint_name)) {
          PUSH_ERROR_AND_RETURN(fmt::format("joint_name {} not found in Skeleton", joint_name));
        }

        int dst_idx = name_to_index_map.at(joint_name);
        index_remap[int(i)] = dst_idx;

        //DCOUT("remap " << i << " to " << dst_idx);
      }

      for (size_t i = 0; i < dst.joint_and_weights.jointIndices.size(); i++) {
        int src_idx = dst.joint_and_weights.jointIndices[i];
        if (index_remap.count(src_idx)) {
          int dst_idx = index_remap[src_idx];

          dst.joint_and_weights.jointIndices[i] = dst_idx;
          //DCOUT("jointIndex modified: remap " << src_idx << " to " << dst_idx);
        }
      }

    }

    // geomBindTransform(optional).
    if (mesh.has_primvar("skel:geomBindTransform")) {
      GeomPrimvar bindTransformPvar;

      if (!GetGeomPrimvar(env.stage, &mesh, "skel:geomBindTransform",
                          &bindTransformPvar, &_err)) {
        return false;
      }

      value::matrix4d bindTransform;
      if (!bindTransformPvar.get_value(&bindTransform)) {
        PUSH_ERROR_AND_RETURN(
            fmt::format("Failed to get `skel:geomBindTransform` attribute. "
                        "Ensure `skel:geomBindTransform` is type `matrix4d`"));
      }

      dst.joint_and_weights.geomBindTransform = bindTransform;
    }
  }

  //
  // 6. BlendShapes
  //
  //    NOTE: (Built-in) BlendShape attributes are per-point, so it is not
  //    affected by triangulation and single-indexable indices build.
  //
  for (const auto &it : blendshapes) {
    const std::string &bs_path = it.first;
    const BlendShape *bs = it.second;

    if (!bs) {
      continue;
    }

    //
    // TODO: in-between attribs
    //

    std::vector<int> vertex_indices;
    std::vector<value::vector3f> normal_offsets;
    std::vector<value::vector3f> vertex_offsets;

    bs->pointIndices.get_value(&vertex_indices);
    bs->normalOffsets.get_value(&normal_offsets);
    bs->offsets.get_value(&vertex_offsets);

    ShapeTarget shapeTarget;
    shapeTarget.abs_path = bs_path;
    shapeTarget.prim_name = bs->name;
    shapeTarget.display_name = bs->metas().displayName.value_or("");

    if (vertex_indices.empty()) {
      PUSH_WARN(
          fmt::format("`pointIndices` in BlendShape `{}` is not authored or "
                      "empty. Skipping.",
                      bs->name));
    }

    // Check if index is valid.
    std::vector<uint32_t> indices;
    indices.resize(vertex_indices.size());

    for (size_t i = 0; i < vertex_indices.size(); i++) {
      if (vertex_indices[i] < 0) {
        PUSH_ERROR_AND_RETURN(fmt::format(
            "negative index in `pointIndices`. Prim path: `{}`", bs_path));
      }

      if (uint32_t(vertex_indices[i]) > dst.points.size()) {
        PUSH_ERROR_AND_RETURN(
            fmt::format("pointIndices[{}] {} exceeds the number of points in "
                        "GeomMesh {}. Prim path: `{}`",
                        i, vertex_indices[i], dst.points.size(), bs_path));
      }

      indices[i] = uint32_t(vertex_indices[i]);
    }
    shapeTarget.pointIndices = indices;

    if (vertex_offsets.size() &&
        (vertex_offsets.size() == vertex_indices.size())) {
      shapeTarget.pointOffsets.resize(vertex_offsets.size());
      memcpy(shapeTarget.pointOffsets.data(), vertex_offsets.data(),
             sizeof(value::normal3f) * vertex_offsets.size());
    }

    if (normal_offsets.size() &&
        (normal_offsets.size() == vertex_indices.size())) {
      shapeTarget.normalOffsets.resize(normal_offsets.size());
      memcpy(shapeTarget.normalOffsets.data(), normal_offsets.data(),
             sizeof(value::normal3f) * normal_offsets.size());
    }

    // TODO inbetweens

    // TODO: key duplicate check
    dst.targets[bs->name] = shapeTarget;
    DCOUT("Converted blendshape target: " << bs->name);
  }

  //
  // 7. Compute normals
  //
  //    Compute normals when normals is not present or compute tangents
  //    requiested but normals is not present. Normals are computed with
  //    'vertex' variability to compute smooth normals for shared vertex.
  //
  //    When triangulated, normals are computed for triangulated mesh.
  //
  bool compute_normals =
      (env.mesh_config.compute_normals && dst.normals.empty());
  bool compute_tangents =
      (env.mesh_config.compute_tangents_and_binormals &&
       (dst.binormals.empty() == 0 && dst.tangents.empty() == 0));

  if (compute_normals || (compute_tangents && dst.normals.empty())) {
    DCOUT("Compute normals");
    std::vector<vec3> normals;
    if (!ComputeNormals(dst.points, dst.faceVertexCounts(),
                       dst.faceVertexIndices(), normals, &_err)) {
      DCOUT("compute normals failed.");
      return false;
    }

    dst.normals.set_buffer(reinterpret_cast<const uint8_t *>(normals.data()),
                           normals.size() * sizeof(vec3));
    dst.normals.elementSize = 1;
    dst.normals.variability = VertexVariability::Vertex;
    dst.normals.format = VertexAttributeFormat::Vec3;
    dst.normals.stride = 0;
    dst.normals.indices.clear();
    dst.normals.name = "normals";

    if (!is_single_indexable) {
      auto result = VertexToFaceVarying(
          dst.normals.get_data(), dst.normals.stride_bytes(),
          dst.faceVertexCounts(), dst.faceVertexIndices());
      if (!result) {
        PUSH_ERROR_AND_RETURN(fmt::format(
            "Convert vertex/varying `normals` attribute to failed: {}",
            result.error()));
      }
      dst.normals.data = result.value();
      dst.normals.variability = VertexVariability::FaceVarying;
    }
  }

  //
  // 8. Build indices
  //
  if (env.mesh_config.build_vertex_indices && (!is_single_indexable)) {
    DCOUT("Build vertex indices");

    if (!BuildVertexIndicesImpl(dst)) {
      return false;
    }

    is_single_indexable = true;
  }

  //
  // 8. Compute tangents.
  //
  if (compute_tangents) {
    DCOUT("Compute tangents.");
    std::vector<vec2> texcoords;
    std::vector<vec3> normals;

    // TODO: Support arbitrary slotID
    if (!dst.texcoords.count(0)) {
      PUSH_ERROR_AND_RETURN(
          "texcoord is required to compute tangents/binormals.\n");
    }

    texcoords.resize(dst.texcoords[0].vertex_count());
    normals.resize(dst.normals.vertex_count());

    memcpy(texcoords.data(), dst.texcoords[0].buffer(),
           dst.texcoords[0].num_bytes());
    memcpy(normals.data(), dst.normals.buffer(), dst.normals.num_bytes());

    std::vector<vec3> tangents;
    std::vector<vec3> binormals;
    std::vector<uint32_t> vertex_indices;

    if (!ComputeTangentsAndBinormals(dst.points, dst.faceVertexCounts(),
                                     dst.faceVertexIndices(), texcoords,
                                     normals, !is_single_indexable, &tangents,
                                     &binormals, &vertex_indices, &_err)) {
      PUSH_ERROR_AND_RETURN("Failed to compute tangents/binormals.");
    }

    // 1. Firstly, always convert tangents/binormals to 'facevarying'
    // variability
    {
      std::vector<vec3> facevarying_tangents;
      std::vector<vec3> facevarying_binormals;
      facevarying_tangents.assign(vertex_indices.size(), {0.0f, 0.0f, 0.0f});
      facevarying_binormals.assign(vertex_indices.size(), {0.0f, 0.0f, 0.0f});
      for (size_t i = 0; i < vertex_indices.size(); i++) {
        facevarying_tangents[i] = tangents[vertex_indices[i]];
        facevarying_binormals[i] = binormals[vertex_indices[i]];
      }

      dst.tangents.data.resize(facevarying_tangents.size() * sizeof(vec3));
      memcpy(dst.tangents.data.data(), facevarying_tangents.data(),
             facevarying_tangents.size() * sizeof(vec3));

      dst.tangents.format = VertexAttributeFormat::Vec3;
      dst.tangents.stride = 0;
      dst.tangents.elementSize = 1;
      dst.tangents.variability = VertexVariability::FaceVarying;

      dst.binormals.data.resize(facevarying_binormals.size() * sizeof(vec3));
      memcpy(dst.binormals.data.data(), facevarying_binormals.data(),
             facevarying_binormals.size() * sizeof(vec3));

      dst.binormals.format = VertexAttributeFormat::Vec3;
      dst.binormals.stride = 0;
      dst.binormals.elementSize = 1;
      dst.binormals.variability = VertexVariability::FaceVarying;
    }

    // 2. Build single vertex indices if `build_vertex_indices` is true.
    if (env.mesh_config.build_vertex_indices) {
      if (!BuildVertexIndicesImpl(dst)) {
        return false;
      }
      is_single_indexable = true;
    }
  }

  dst.is_single_indexable = is_single_indexable;

  dst.prim_name = mesh.name;
  dst.abs_path = abs_prim_path.full_path_name();
  dst.display_name = mesh.metas().displayName.value_or("");

  (*dstMesh) = std::move(dst);

  return true;
}

namespace {

// Convert UsdTransform2d -> PrimvarReader_float2 shader network.
nonstd::expected<bool, std::string> ConvertTexTransform2d(
    const Stage &stage, const Path &tx_abs_path, const UsdTransform2d &tx,
    UVTexture *tex_out, double timecode) {
  float rotation;  // in angles
  if (!tx.rotation.get_value().get(timecode, &rotation)) {
    return nonstd::make_unexpected(
        fmt::format("Failed to retrieve rotation attribute from {}\n",
                    tx_abs_path.full_path_name()));
  }

  value::float2 scale;
  if (!tx.scale.get_value().get(timecode, &scale)) {
    return nonstd::make_unexpected(
        fmt::format("Failed to retrieve scale attribute from {}\n",
                    tx_abs_path.full_path_name()));
  }

  value::float2 translation;
  if (!tx.translation.get_value().get(timecode, &translation)) {
    return nonstd::make_unexpected(
        fmt::format("Failed to retrieve translation attribute from {}\n",
                    tx_abs_path.full_path_name()));
  }

  // must be authored and connected to PrimvarReader.
  if (!tx.in.authored()) {
    return nonstd::make_unexpected("`inputs:in` must be authored.\n");
  }

  if (!tx.in.is_connection()) {
    return nonstd::make_unexpected("`inputs:in` must be a connection.\n");
  }

  const auto &paths = tx.in.get_connections();
  if (paths.size() != 1) {
    return nonstd::make_unexpected(
        "`inputs:in` must be a single connection Path.\n");
  }

  std::string prim_part = paths[0].prim_part();
  std::string prop_part = paths[0].prop_part();

  if (prop_part != "outputs:result") {
    return nonstd::make_unexpected(
        "`inputs:in` connection Path's property part must be "
        "`outputs:result`\n");
  }

  std::string err;

  const Prim *pprim{nullptr};
  if (!stage.find_prim_at_path(Path(prim_part, ""), pprim, &err)) {
    return nonstd::make_unexpected(fmt::format(
        "`inputs:in` connection Path not found in the Stage. {}\n", prim_part));
  }

  if (!pprim) {
    return nonstd::make_unexpected(
        fmt::format("[InternalError] Prim is nullptr: {}\n", prim_part));
  }

  const Shader *pshader = pprim->as<Shader>();
  if (!pshader) {
    return nonstd::make_unexpected(
        fmt::format("{} must be Shader Prim, but got {}\n", prim_part,
                    pprim->prim_type_name()));
  }

  const UsdPrimvarReader_float2 *preader =
      pshader->value.as<UsdPrimvarReader_float2>();
  if (!preader) {
    return nonstd::make_unexpected(fmt::format(
        "Shader {} must be UsdPrimvarReader_float2 type, but got {}(internal type {})\n",
        prim_part, pshader->info_id, pshader->value.type_name()));
  }

  // Get value producing attribute(i.e, follow .connection and return
  // terminal Attribute value)
  //value::token varname;

  // 'string' for inputs:varname preferred.
  std::string varname;
#if 0
  if (!tydra::EvaluateShaderAttribute(stage, *pshader, "inputs:varname",
                                      &varname, &err)) {
    return nonstd::make_unexpected(
        fmt::format("Failed to evaluate UsdPrimvarReader_float2's "
                    "inputs:varname: {}\n",
                    err));
  }
#else
  TerminalAttributeValue attr;
  if (!tydra::EvaluateAttribute(stage, *pprim, "inputs:varname", &attr, &err)) {
    return nonstd::make_unexpected(
        "`inputs:varname` evaluation failed: " + err + "\n");
  }
  if (auto pvt = attr.as<value::token>()) {
    varname = pvt->str();
  } else if (auto pvs = attr.as<std::string>()) {
    varname = *pvs;
  } else if (auto pvsd = attr.as<value::StringData>()) {
    varname = (*pvsd).value;
  } else {
    return nonstd::make_unexpected(
        "`inputs:varname` must be `token` or `string` type, but got " + attr.type_name() +
        "\n");
  }
  if (varname.empty()) {
    return nonstd::make_unexpected("`inputs:varname` is empty token\n");
  }
  DCOUT("inputs:varname = " << varname);
#endif

  // Build transform matrix.
  // https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_texture_transform
  // Since USD uses post-multiply,
  //
  // matrix = scale * rotate * translate
  //
  {
    mat3 s;
    s.set_scale(scale[0], scale[1], 1.0f);

    mat3 r = mat3::identity();

    r.m[0][0] = std::cos(math::radian(rotation));
    r.m[0][1] = std::sin(math::radian(rotation));

    r.m[1][0] = -std::sin(math::radian(rotation));
    r.m[1][1] = std::cos(math::radian(rotation));

    mat3 t = mat3::identity();
    t.set_translation(translation[0], translation[1], 1.0f);

    tex_out->transform = s * r * t;
  }

  tex_out->tx_rotation = rotation;
  tex_out->tx_translation = translation;
  tex_out->tx_scale = scale;
  tex_out->has_transform2d = true;

  tex_out->varname_uv = varname;

  return true;
}

template <typename T>
nonstd::expected<bool, std::string> GetConnectedUVTexture(
    const Stage &stage, const TypedAnimatableAttributeWithFallback<T> &src,
    Path *tex_abs_path, const UsdUVTexture **dst, const Shader **shader_out) {
  if (!dst) {
    return nonstd::make_unexpected("[InternalError] dst is nullptr.\n");
  }

  if (!src.is_connection()) {
    return nonstd::make_unexpected("Attribute must be connection.\n");
  }

  if (src.get_connections().size() != 1) {
    return nonstd::make_unexpected(
        "Attribute connections must be single connection Path.\n");
  }

  //
  // Example: color3f inputs:diffuseColor.connect = </path/to/tex.outputs:rgb>
  //
  // => path.prim_part : /path/to/tex
  // => path.prop_part : outputs:rgb
  //

  const Path &path = src.get_connections()[0];

  const std::string prim_part = path.prim_part();
  const std::string prop_part = path.prop_part();

  // NOTE: no `outputs:rgba` in the spec.
  constexpr auto kOutputsRGB = "outputs:rgb";
  constexpr auto kOutputsR = "outputs:r";
  constexpr auto kOutputsG = "outputs:g";
  constexpr auto kOutputsB = "outputs:b";
  constexpr auto kOutputsA = "outputs:a";

  if (prop_part == kOutputsRGB) {
    // ok
  } else if (prop_part == kOutputsR) {
    // ok
  } else if (prop_part == kOutputsG) {
    // ok
  } else if (prop_part == kOutputsB) {
    // ok
  } else if (prop_part == kOutputsA) {
    // ok
  } else {
    return nonstd::make_unexpected(fmt::format(
        "connection Path's property part must be `{}`, `{}`, `{}` or `{}` "
        "for "
        "UsdUVTexture, but got `{}`\n",
        kOutputsRGB, kOutputsR, kOutputsG, kOutputsB, kOutputsA, prop_part));
  }

  const Prim *prim{nullptr};
  std::string err;
  if (!stage.find_prim_at_path(Path(prim_part, ""), prim, &err)) {
    return nonstd::make_unexpected(
        fmt::format("Prim {} not found in the Stage: {}\n", prim_part, err));
  }

  if (!prim) {
    return nonstd::make_unexpected("[InternalError] Prim ptr is null.\n");
  }

  if (tex_abs_path) {
    (*tex_abs_path) = Path(prim_part, prop_part);
  }

  if (const Shader *pshader = prim->as<Shader>()) {
    if (const UsdUVTexture *ptex = pshader->value.as<UsdUVTexture>()) {
      DCOUT("ptex = " << ptex);
      (*dst) = ptex;

      if (shader_out) {
        (*shader_out) = pshader;
      }

      return true;
    }
  }

  return nonstd::make_unexpected(
      fmt::format("Prim {} must be `Shader` Prim type, but got `{}`", prim_part,
                  prim->prim_type_name()));
}

static bool RawAssetRead(
    const value::AssetPath &assetPath, const AssetInfo &assetInfo,
    const AssetResolutionResolver &assetResolver,
    Asset *assetOut,
    std::string &resolvedPathOut,
    void *userdata, std::string *warn,
    std::string *err) {
  if (!assetOut) {
    if (err) {
      (*err) = "`assetOut` argument is nullptr\n";
    }
    return false;
  }

  // TODO: assetInfo
  (void)assetInfo;
  (void)userdata;
  (void)warn;

  std::string resolvedPath = assetResolver.resolve(assetPath.GetAssetPath());

  if (resolvedPath.empty()) {
    if (err) {
      (*err) += fmt::format("Failed to resolve asset path: {}\n",
                            assetPath.GetAssetPath());
    }
    return false;
  }

  Asset asset;
  bool ret = assetResolver.open_asset(resolvedPath, assetPath.GetAssetPath(),
                                      &asset, warn, err);
  if (!ret) {
    if (err) {
      (*err) += fmt::format("Failed to open asset: {}", resolvedPath);
    }
    return false;
  }

  DCOUT("Resolved asset path = " << resolvedPath);

  resolvedPathOut = resolvedPath;
  (*assetOut) = std::move(asset);

  return true;
}

}  // namespace

// Convert UsdUVTexture shader node.
// @return true upon conversion success(textures.back() contains the converted
// UVTexture)
//
// Possible network configuration
//
// - UsdUVTexture -> UsdPrimvarReader
// - UsdUVTexture -> UsdTransform2d -> UsdPrimvarReader
bool RenderSceneConverter::ConvertUVTexture(const RenderSceneConverterEnv &env,
                                            const Path &tex_abs_path,
                                            const AssetInfo &assetInfo,
                                            const UsdUVTexture &texture,
                                            UVTexture *tex_out) {
  DCOUT("ConvertUVTexture " << tex_abs_path);

  if (!tex_out) {
    PUSH_ERROR_AND_RETURN("tex_out arg is nullptr.");
  }
  std::string err;

  UVTexture tex;

  if (!texture.file.authored()) {
    PUSH_ERROR_AND_RETURN(fmt::format("`asset:file` is not authored. Path = {}",
                                      tex_abs_path.prim_part()));
  }

  value::AssetPath assetPath;
  if (auto apath = texture.file.get_value()) {
    if (!apath.value().get(env.timecode, &assetPath)) {
      PUSH_ERROR_AND_RETURN(fmt::format(
          "Failed to get `asset:file` value from Path {} at time {}",
          tex_abs_path.prim_part(), env.timecode));
    }
  } else {
    PUSH_ERROR_AND_RETURN(
        fmt::format("Failed to get `asset:file` value from Path {}",
                    tex_abs_path.prim_part()));
  }

  // TextureImage and BufferData
  {
    TextureImage texImage;
    BufferData assetImageBuffer;

    // Texel data is treated as byte array
    assetImageBuffer.componentType = ComponentType::UInt8;

    bool tex_loaded{false};

    if (env.scene_config.load_texture_assets) {
      DCOUT("load texture : " << assetPath.GetAssetPath());
      std::string warn;

      TextureImageLoaderFunction tex_loader_fun =
          env.material_config.texture_image_loader_function;

      if (!tex_loader_fun) {
        tex_loader_fun = DefaultTextureImageLoaderFunction;
      }

      tex_loaded = tex_loader_fun(
          assetPath, assetInfo, env.asset_resolver, &texImage,
          &assetImageBuffer.data,
          env.material_config.texture_image_loader_function_userdata, &warn,
          &err);

      if (warn.size()) {
        DCOUT("WARN: " << warn);
        PushWarn(warn);
      }

      if (!tex_loaded && !env.material_config.allow_texture_load_failure) {
        PUSH_ERROR_AND_RETURN(fmt::format("Failed to load texture image: `{}` err = {}", assetPath.GetAssetPath(), err));
      }


      if (err.size()) {
        // report as warn.
        PUSH_WARN(fmt::format("Failed to load texture image: `{}`. Skip loading. reason = {} ", assetPath.GetAssetPath(), err));
      }

      // store unresolved asset path.
      texImage.asset_identifier = assetPath.GetAssetPath();
      texImage.decoded = true;

    } else {

      Asset asset;
      std::string resolvedPath;
      if (RawAssetRead(assetPath, assetInfo, env.asset_resolver, &asset, resolvedPath, /* userdata */nullptr, /* warn */nullptr, &err )) {
        
        // store resolved asset path.
        texImage.asset_identifier = resolvedPath;
        

        BufferData imageBuffer;
        imageBuffer.componentType = tydra::ComponentType::UInt8;

        imageBuffer.data.resize(asset.size());
        memcpy(imageBuffer.data.data(), asset.data(), asset.size());

        // Assign buffer id
        texImage.buffer_id = int64_t(buffers.size());

        // TODO: Share image data as much as possible.
        // e.g. Texture A and B uses same image file, but texturing parameter is
        // different.
        buffers.emplace_back(imageBuffer);
        
        texImage.decoded = false;
        DCOUT("texture image is read, but not decoded.");
      
      } else {
        // store resolved asset path.
        texImage.asset_identifier = env.asset_resolver.resolve(assetPath.GetAssetPath());
        texImage.decoded = false;

        DCOUT("store asset path.");
      }

    }

    // colorSpace.
    // First look into `colorSpace` metadata of asset, then
    // look into `inputs:sourceColorSpace' attribute.
    // When both `colorSpace` metadata and `inputs:sourceColorSpace' attribute
    // exists, `colorSpace` metadata supercedes.
    // NOTE: `inputs:sourceColorSpace` attribute should be deprecated in favor of `colorSpace` metadata.
    bool inferColorSpaceFailed = false;
    if (texture.file.metas().has_colorSpace()) {
      ColorSpace cs;
      value::token cs_token = texture.file.metas().get_colorSpace();
      if (InferColorSpace(cs_token, &cs)) {
        texImage.usdColorSpace = cs;
        DCOUT("Inferred colorSpace: " << to_string(cs));
      } else {
        inferColorSpaceFailed = true;
      }
    }

    bool sourceColorSpaceSet = false;
    if (inferColorSpaceFailed || !texture.file.metas().has_colorSpace()) {
      if (texture.sourceColorSpace.authored()) {
        UsdUVTexture::SourceColorSpace cs;
        if (texture.sourceColorSpace.get_value().get(env.timecode, &cs)) {
          if (cs == UsdUVTexture::SourceColorSpace::SRGB) {
            texImage.usdColorSpace = tydra::ColorSpace::sRGB;
            sourceColorSpaceSet = true;
          } else if (cs == UsdUVTexture::SourceColorSpace::Raw) {
            texImage.usdColorSpace = tydra::ColorSpace::Raw;
            sourceColorSpaceSet = true;
          } else if (cs == UsdUVTexture::SourceColorSpace::Auto) {

            if (tex_loaded) {

              // The spec says: https://openusd.org/release/spec_usdpreviewsurface.html
              //
              // auto : Check for gamma/color space metadata in the texture file itself; if metadata is indicative of sRGB, mark texture as sRGB . If no relevant metadata is found, mark texture as sRGB if it is either 8-bit and has 3 channels or if it is 8-bit and has 4 channels. Otherwise, do not mark texture as sRGB and use texture data as it was read from the texture.
              //
              if (((texImage.assetTexelComponentType == ComponentType::UInt8) ||
                  (texImage.assetTexelComponentType == ComponentType::Int8)) &&
                ((texImage.channels == 3) || (texImage.channels ==4))) {
                texImage.usdColorSpace = tydra::ColorSpace::sRGB;
                sourceColorSpaceSet = true;
              } else {
                PUSH_WARN(fmt::format("Infer colorSpace failed for {}. Set to Raw for now. Results may be wrong.", assetPath.GetAssetPath()));
                // At least 'not' sRGB. For now set to Raw.

                texImage.usdColorSpace = tydra::ColorSpace::Raw;
                sourceColorSpaceSet = true;
              }
            } else {
              texImage.usdColorSpace = tydra::ColorSpace::Unknown;
              sourceColorSpaceSet = true;
            }
          }
        }
      }
    }

    if (!sourceColorSpaceSet && inferColorSpaceFailed) {
      value::token cs_token = texture.file.metas().get_colorSpace();
      PUSH_ERROR_AND_RETURN(
          fmt::format("Invalid or unknown colorSpace metadataum: {}. Please "
                      "report an issue to TinyUSDZ github repo.",
                      cs_token.str()));
    }

    if (tex_loaded) {
      BufferData imageBuffer;

      // Linearlization and widen texel bit depth if required.
      if (env.material_config.linearize_color_space) {
        // TODO: Support ACEScg and Lin_DisplayP3
        DCOUT("linearlize colorspace.");
        size_t width = size_t(texImage.width);
        size_t height = size_t(texImage.height);
        size_t channels = size_t(texImage.channels);

        if (channels > 4) {
          PUSH_ERROR_AND_RETURN(
              fmt::format("TODO: Multiband color channels(5 or more) are not "
                          "supported(yet)."));
        }

        if (assetImageBuffer.componentType == tydra::ComponentType::UInt8) {
          if (texImage.usdColorSpace == tydra::ColorSpace::sRGB) {
            if (env.material_config.preserve_texel_bitdepth) {
              // u8 sRGB -> u8 Linear
              imageBuffer.componentType = tydra::ComponentType::UInt8;

              bool ret = srgb_8bit_to_linear_8bit(
                  assetImageBuffer.data, width, height, channels,
                  /* channel stride */ channels, &imageBuffer.data, &_err);
              if (!ret) {
                PUSH_ERROR_AND_RETURN(
                    "Failed to convert sRGB u8 image to Linear u8 image.");
              }

            } else {
              DCOUT("u8 sRGB -> fp32 linear.");
              // u8 sRGB -> fp32 Linear
              imageBuffer.componentType = tydra::ComponentType::Float;

              std::vector<float> buf;
              bool ret = srgb_8bit_to_linear_f32(
                  assetImageBuffer.data, width, height, channels,
                  /* channel stride */ channels, &buf, &_err);
              if (!ret) {
                PUSH_ERROR_AND_RETURN(
                    "Failed to convert sRGB u8 image to Linear f32 image.");
              }

              DCOUT("sz = " << buf.size());
              imageBuffer.data.resize(buf.size() * sizeof(float));
              memcpy(imageBuffer.data.data(), buf.data(),
                     sizeof(float) * buf.size());
            }

            texImage.colorSpace = tydra::ColorSpace::Lin_sRGB;

          } else if (texImage.usdColorSpace == tydra::ColorSpace::Lin_sRGB) {
            if (env.material_config.preserve_texel_bitdepth) {
              // no op.
              imageBuffer = std::move(assetImageBuffer);

            } else {
              // u8 -> fp32
              imageBuffer.componentType = tydra::ComponentType::Float;

              std::vector<float> buf;
              bool ret = u8_to_f32_image(assetImageBuffer.data, width, height,
                                         channels, &buf, &_err);
              if (!ret) {
                PUSH_ERROR_AND_RETURN("Failed to convert u8 image to f32 image.");
              }

              imageBuffer.data.resize(buf.size() * sizeof(float));
              memcpy(imageBuffer.data.data(), buf.data(),
                     sizeof(float) * buf.size());
            }

            texImage.colorSpace = tydra::ColorSpace::Lin_sRGB;

          } else {
            PUSH_ERROR(fmt::format("TODO: Color space {}",
                                   to_string(texImage.usdColorSpace)));
          }

        } else if (assetImageBuffer.componentType ==
                   tydra::ComponentType::Float) {
          // ignore preserve_texel_bitdepth

          if (texImage.usdColorSpace == tydra::ColorSpace::sRGB) {
            // srgb f32 -> linear f32
            std::vector<float> in_buf;
            std::vector<float> out_buf;
            in_buf.resize(assetImageBuffer.data.size() / sizeof(float));
            memcpy(in_buf.data(), assetImageBuffer.data.data(),
                   in_buf.size() * sizeof(float));

            out_buf.resize(assetImageBuffer.data.size() / sizeof(float));

            // TODO: scale factor & bias
            float scale_factor = 1.0f;
            float bias = 0.0f;
            float alpha_scale_factor = 1.0f;
            float alpha_bias = 0.0f;

            bool ret =
                srgb_f32_to_linear_f32(in_buf, width, height, channels,
                                       /* channel stride */ channels, &out_buf, scale_factor, bias, alpha_scale_factor, alpha_bias, &_err);

            if (!ret) {
              PUSH_ERROR_AND_RETURN(
                  "Failed to convert sRGB f32 image to Linear f32 image.");
            }

            imageBuffer.data.resize(assetImageBuffer.data.size());
            memcpy(imageBuffer.data.data(), out_buf.data(),
                   imageBuffer.data.size());


          } else if (texImage.usdColorSpace == tydra::ColorSpace::Lin_sRGB) {
            // no op
            imageBuffer = std::move(assetImageBuffer);

          } else {
            PUSH_ERROR(fmt::format("TODO: Color space {}",
                                   to_string(texImage.usdColorSpace)));
          }

        } else {
          PUSH_ERROR(fmt::format("TODO: asset texture texel format {}",
                                 to_string(assetImageBuffer.componentType)));
        }

      } else {
        // Same color space.
        DCOUT("assetImageBuffer.sz = " << assetImageBuffer.data.size());

        if (assetImageBuffer.componentType == tydra::ComponentType::UInt8) {
          if (env.material_config.preserve_texel_bitdepth) {
            // Do nothing.
            imageBuffer = std::move(assetImageBuffer);

          } else {
            size_t width = size_t(texImage.width);
            size_t height = size_t(texImage.height);
            size_t channels = size_t(texImage.channels);

            // u8 to f32, but no sRGB -> linear conversion(this would break
            // UsdPreviewSurface's spec though)
            PUSH_WARN(
                "8bit sRGB texture is converted to fp32 sRGB texture(without "
                "linearlization)");
            std::vector<float> buf;
            bool ret = u8_to_f32_image(assetImageBuffer.data, width, height,
                                       channels, &buf, &_err);
            if (!ret) {
              PUSH_ERROR_AND_RETURN("Failed to convert u8 image to f32 image.");
            }
            imageBuffer.componentType = tydra::ComponentType::Float;

            imageBuffer.data.resize(buf.size() * sizeof(float));
            memcpy(imageBuffer.data.data(), buf.data(),
                   sizeof(float) * buf.size());
          }

          texImage.colorSpace = texImage.usdColorSpace;

        } else if (assetImageBuffer.componentType ==
                   tydra::ComponentType::Float) {
          // ignore preserve_texel_bitdepth

          // f32 to f32, so no op
          imageBuffer = std::move(assetImageBuffer);

        } else {
          PUSH_ERROR(fmt::format("TODO: asset texture texel format {}",
                                 to_string(assetImageBuffer.componentType)));
        }
      }

      // Assign buffer id
      texImage.buffer_id = int64_t(buffers.size());

      // TODO: Share image data as much as possible.
      // e.g. Texture A and B uses same image file, but texturing parameter is
      // different.
      buffers.emplace_back(imageBuffer);

      tex.texture_image_id = int64_t(images.size());

      images.emplace_back(texImage);

      std::stringstream ss;
      ss << "Loaded texture image " << assetPath.GetAssetPath()
         << " : buffer_id " + std::to_string(texImage.buffer_id) << "\n";
      ss << "  width x height x components " << texImage.width << " x "
         << texImage.height << " x " << texImage.channels << "\n";
      ss << "  colorSpace " << tinyusdz::tydra::to_string(texImage.colorSpace)
         << "\n";
      PushInfo(ss.str());
    } else {

      tex.texture_image_id = int64_t(images.size());

      images.emplace_back(texImage);

      std::stringstream ss;
      ss << "Loaded texture image " << assetPath.GetAssetPath()
         << " : buffer_id " + std::to_string(texImage.buffer_id) << "\n";
      ss << "  width x height x components " << texImage.width << " x "
         << texImage.height << " x " << texImage.channels << "\n";
      ss << "  colorSpace " << tinyusdz::tydra::to_string(texImage.colorSpace)
         << "\n";
      PushInfo(ss.str());

    }
  }

  //
  // Set authored outputChannels
  //
  if (texture.outputsRGB.authored()) {
    tex.authoredOutputChannels.insert(UVTexture::Channel::RGB);
  }

  if (texture.outputsA.authored()) {
    tex.authoredOutputChannels.insert(UVTexture::Channel::A);
  }

  if (texture.outputsR.authored()) {
    tex.authoredOutputChannels.insert(UVTexture::Channel::R);
  }

  if (texture.outputsG.authored()) {
    tex.authoredOutputChannels.insert(UVTexture::Channel::G);
  }

  if (texture.outputsB.authored()) {
    tex.authoredOutputChannels.insert(UVTexture::Channel::B);
  }

#if 0 // TODO
  if (tex.authoredOutputChannels.empty()) {
    PUSH_WARN("No valid output channel attribute authored. Default to RGB");
    tex.authoredOutputChannels.insert(UVTexture::Channel::RGB);
  }
#endif

  //
  // Convert other UVTexture parameters
  //

  if (texture.bias.authored()) {
    tex.bias = texture.bias.get_value();
  }

  if (texture.scale.authored()) {
    tex.scale = texture.scale.get_value();
  }

  if (texture.st.authored()) {
    if (texture.st.is_connection()) {
      const auto &paths = texture.st.get_connections();
      if (paths.size() != 1) {
        PUSH_ERROR_AND_RETURN(
            "UsdUVTexture inputs:st connection must be single Path.");
      }
      const Path &path = paths[0];

      const Prim *readerPrim{nullptr};
      if (!env.stage.find_prim_at_path(Path(path.prim_part(), ""), readerPrim,
                                       &err)) {
        PUSH_ERROR_AND_RETURN(
            "UsdUVTexture inputs:st connection targetPath not found in the "
            "Stage: " +
            err);
      }

      if (!readerPrim) {
        PUSH_ERROR_AND_RETURN(
            "[InternlError] Invalid Prim connected to inputs:st");
      }

      const Shader *pshader = readerPrim->as<Shader>();
      if (!pshader) {
        PUSH_ERROR_AND_RETURN(
            fmt::format("UsdUVTexture inputs:st connected Prim must be "
                        "Shader Prim, but got {} Prim",
                        readerPrim->prim_type_name()));
      }

      // currently UsdTranform2d or PrimvarReaer_float2 only for inputs:st
      if (const UsdPrimvarReader_float2 *preader =
              pshader->value.as<UsdPrimvarReader_float2>()) {
        if (!preader) {
          PUSH_ERROR_AND_RETURN(
              fmt::format("Shader's info:id must be UsdPrimvarReader_float2, "
                          "but got {}",
                          pshader->info_id));
        }

        // Get value producing attribute(i.e, follow .connection and return
        // terminal Attribute value)
        std::string varname;
        TerminalAttributeValue attr;
        if (!tydra::EvaluateAttribute(env.stage, *readerPrim, "inputs:varname",
                                      &attr, &err)) {
          PUSH_ERROR_AND_RETURN(
              fmt::format("Failed to evaluate UsdPrimvarReader_float2's "
                          "inputs:varname.\n{}",
                          err));
        }

        if (auto pv = attr.as<value::token>()) {
          varname = (*pv).str();
        } else if (auto pvs = attr.as<std::string>()) {
          varname = (*pvs);
        } else if (auto pvsd = attr.as<value::StringData>()) {
          varname = (*pvsd).value;
        } else {
          PUSH_ERROR_AND_RETURN(
              "`inputs:varname` must be `string` or `token` type, but got " +
              attr.type_name());
        }
        if (varname.empty()) {
          PUSH_ERROR_AND_RETURN("`inputs:varname` is empty token.");
        }
        DCOUT("inputs:varname = " << varname);

        tex.varname_uv = varname;
      } else if (const UsdTransform2d *ptransform =
                     pshader->value.as<UsdTransform2d>()) {
        auto result = ConvertTexTransform2d(env.stage, path, *ptransform, &tex,
                                            env.timecode);
        if (!result) {
          PUSH_ERROR_AND_RETURN(result.error());
        }
      } else {
        PUSH_ERROR_AND_RETURN(
            "Unsupported Shader type for `inputs:st` connection: " +
            pshader->info_id + "\n");
      }

    } else {
      Animatable<value::texcoord2f> fallbacks = texture.st.get_value();
      value::texcoord2f uv;
      if (fallbacks.get(env.timecode, &uv)) {
        tex.fallback_uv[0] = uv[0];
        tex.fallback_uv[1] = uv[1];
      } else {
        // TODO: report warning.
        PUSH_WARN("Failed to get fallback `st` texcoord attribute.");
      }
    }
  }

  if (texture.wrapS.authored()) {
    tinyusdz::UsdUVTexture::Wrap wrap;

    if (!texture.wrapS.get_value().get(env.timecode, &wrap)) {
      PUSH_ERROR_AND_RETURN("Invalid UsdUVTexture inputs:wrapS value.");
    }

    if (wrap == UsdUVTexture::Wrap::Repeat) {
      tex.wrapS = UVTexture::WrapMode::REPEAT;
    } else if (wrap == UsdUVTexture::Wrap::Mirror) {
      tex.wrapS = UVTexture::WrapMode::MIRROR;
    } else if (wrap == UsdUVTexture::Wrap::Clamp) {
      tex.wrapS = UVTexture::WrapMode::CLAMP_TO_EDGE;
    } else if (wrap == UsdUVTexture::Wrap::Black) {
      tex.wrapS = UVTexture::WrapMode::CLAMP_TO_BORDER;
    } else {
      tex.wrapS = UVTexture::WrapMode::CLAMP_TO_EDGE;
    }
  }

  if (texture.wrapT.authored()) {
    tinyusdz::UsdUVTexture::Wrap wrap;

    if (!texture.wrapT.get_value().get(env.timecode, &wrap)) {
      PUSH_ERROR_AND_RETURN("Invalid UsdUVTexture inputs:wrapT value.");
    }

    if (wrap == UsdUVTexture::Wrap::Repeat) {
      tex.wrapT = UVTexture::WrapMode::REPEAT;
    } else if (wrap == UsdUVTexture::Wrap::Mirror) {
      tex.wrapT = UVTexture::WrapMode::MIRROR;
    } else if (wrap == UsdUVTexture::Wrap::Clamp) {
      tex.wrapT = UVTexture::WrapMode::CLAMP_TO_EDGE;
    } else if (wrap == UsdUVTexture::Wrap::Black) {
      tex.wrapT = UVTexture::WrapMode::CLAMP_TO_BORDER;
    } else {
      tex.wrapT = UVTexture::WrapMode::CLAMP_TO_EDGE;
    }
  }

  DCOUT("Converted UVTexture.");

  (*tex_out) = tex;
  return true;
}

template <typename T, typename Dty>
bool RenderSceneConverter::ConvertPreviewSurfaceShaderParam(
    const RenderSceneConverterEnv &env, const Path &shader_abs_path,
    const TypedAttributeWithFallback<Animatable<T>> &param,
    const std::string &param_name, ShaderParam<Dty> &dst_param) {
  if (!param.authored()) {
    return true;
  }

  if (param.is_blocked()) {
    PUSH_ERROR_AND_RETURN(fmt::format("{} attribute is blocked.", param_name));
  } else if (param.is_connection()) {
    DCOUT(fmt::format("{} is attribute connection.", param_name));

    const UsdUVTexture *ptex{nullptr};
    const Shader *pshader{nullptr};
    Path texPath;
    auto result =
        GetConnectedUVTexture(env.stage, param, &texPath, &ptex, &pshader);

    if (!result) {
      PUSH_ERROR_AND_RETURN(result.error());
    }

    if (!ptex) {
      PUSH_ERROR_AND_RETURN("[InternalError] ptex is nullptr.");
    }
    DCOUT("ptex = " << ptex->name);

    if (!pshader) {
      PUSH_ERROR_AND_RETURN("[InternalError] pshader is nullptr.");
    }

    DCOUT("Get connected UsdUVTexture Prim: " << texPath);

    UVTexture rtex;
    const AssetInfo &assetInfo = pshader->metas().get_assetInfo();
    if (!ConvertUVTexture(env, texPath, assetInfo, *ptex, &rtex)) {
      PUSH_ERROR_AND_RETURN(fmt::format(
          "Failed to convert UVTexture connected to {}", param_name));
    }

    // Extract connected outputChannel from prop part.
    std::string prop_part = texPath.prop_part();

    // TODO: Attribute type check.
    if (prop_part == "outputs:r") {
      rtex.connectedOutputChannel = tydra::UVTexture::Channel::R;
    } else if (prop_part == "outputs:g") {
      rtex.connectedOutputChannel = tydra::UVTexture::Channel::G;
    } else if (prop_part == "outputs:b") {
      rtex.connectedOutputChannel = tydra::UVTexture::Channel::B;
    } else if (prop_part == "outputs:a") {
      rtex.connectedOutputChannel = tydra::UVTexture::Channel::A;
    } else if (prop_part == "outputs:rgb") {
      rtex.connectedOutputChannel = tydra::UVTexture::Channel::RGB;
    } else {
      PUSH_ERROR_AND_RETURN(fmt::format("Unknown or invalid connection to a property of output channel: {}(Abs path {})", prop_part, texPath.full_path_name()));
    }


    uint64_t texId = textures.size();
    textures.push_back(rtex);

    textureMap.add(texId, shader_abs_path.prim_part() + "." + param_name);

    DCOUT(fmt::format("TexId {}.{} = {}",
                      shader_abs_path.prim_part(), param_name, texId));

    dst_param.texture_id = int32_t(texId);

    return true;
  } else {
    T val;
    if (!param.get_value().get(env.timecode, &val)) {
      PUSH_ERROR_AND_RETURN(
          fmt::format("Failed to get {} at `default` timecode.", param_name));
    }

    dst_param.set_value(val);

    return true;
  }
}

bool RenderSceneConverter::ConvertPreviewSurfaceShader(
    const RenderSceneConverterEnv &env, const Path &shader_abs_path,
    const UsdPreviewSurface &shader, PreviewSurfaceShader *rshader_out) {
  if (!rshader_out) {
    PUSH_ERROR_AND_RETURN("rshader_out arg is nullptr.");
  }

  PreviewSurfaceShader rshader;

  if (shader.useSpecularWorkflow.authored()) {
    if (shader.useSpecularWorkflow.is_blocked()) {
      PUSH_ERROR_AND_RETURN(
          fmt::format("useSpecularWorkflow attribute is blocked."));
    } else if (shader.useSpecularWorkflow.is_connection()) {
      PUSH_ERROR_AND_RETURN(
          fmt::format("TODO: useSpecularWorkflow with connection."));
    } else {
      int val;
      if (!shader.useSpecularWorkflow.get_value().get(env.timecode, &val)) {
        PUSH_ERROR_AND_RETURN(
            fmt::format("Failed to get useSpcularWorkflow value at time `{}`.",
                        env.timecode));
      }

      rshader.useSpecularWorkflow = val ? true : false;
    }
  }

  if (!ConvertPreviewSurfaceShaderParam(env, shader_abs_path,
                                        shader.diffuseColor, "diffuseColor",
                                        rshader.diffuseColor)) {
    return false;
  }

  if (!ConvertPreviewSurfaceShaderParam(env, shader_abs_path,
                                        shader.emissiveColor, "emissiveColor",
                                        rshader.emissiveColor)) {
    return false;
  }

  if (!ConvertPreviewSurfaceShaderParam(env, shader_abs_path,
                                        shader.specularColor, "specularColor",
                                        rshader.specularColor)) {
    return false;
  }

  if (!ConvertPreviewSurfaceShaderParam(env, shader_abs_path, shader.normal,
                                        "normal", rshader.normal)) {
    return false;
  }

  if (!ConvertPreviewSurfaceShaderParam(env, shader_abs_path, shader.roughness,
                                        "roughness", rshader.roughness)) {
    return false;
  }

  if (!ConvertPreviewSurfaceShaderParam(env, shader_abs_path, shader.metallic,
                                        "metallic", rshader.metallic)) {
    return false;
  }

  if (!ConvertPreviewSurfaceShaderParam(env, shader_abs_path, shader.clearcoat,
                                        "clearcoat", rshader.clearcoat)) {
    return false;
  }

  if (!ConvertPreviewSurfaceShaderParam(
          env, shader_abs_path, shader.clearcoatRoughness, "clearcoatRoughness",
          rshader.clearcoatRoughness)) {
    return false;
  }
  if (!ConvertPreviewSurfaceShaderParam(env, shader_abs_path, shader.opacity,
                                        "opacity", rshader.opacity)) {
    return false;
  }
  if (!ConvertPreviewSurfaceShaderParam(
          env, shader_abs_path, shader.opacityThreshold, "opacityThreshold",
          rshader.opacityThreshold)) {
    return false;
  }

  if (!ConvertPreviewSurfaceShaderParam(env, shader_abs_path, shader.ior, "ior",
                                        rshader.ior)) {
    return false;
  }

  if (!ConvertPreviewSurfaceShaderParam(env, shader_abs_path, shader.occlusion,
                                        "occlusion", rshader.occlusion)) {
    return false;
  }

  if (!ConvertPreviewSurfaceShaderParam(env, shader_abs_path,
                                        shader.displacement, "displacement",
                                        rshader.displacement)) {
    return false;
  }

  (*rshader_out) = rshader;
  return true;
}

bool RenderSceneConverter::ConvertMaterial(const RenderSceneConverterEnv &env,
                                           const Path &mat_abs_path,
                                           const tinyusdz::Material &material,
                                           RenderMaterial *rmat_out) {
  if (!rmat_out) {
    PUSH_ERROR_AND_RETURN("rmat_out argument is nullptr.");
  }

  RenderMaterial rmat;
  rmat.abs_path = mat_abs_path.prim_part();
  rmat.name = mat_abs_path.element_name();
  DCOUT("rmat.abs_path = " << rmat.abs_path);
  DCOUT("rmat.name = " << rmat.name);
  std::string err;
  Path surfacePath;

  //
  // surface shader
  {
    if (material.surface.authored()) {
      auto paths = material.surface.get_connections();
      DCOUT("paths = " << paths);
      // must have single targetPath.
      if (paths.size() != 1) {
        PUSH_ERROR_AND_RETURN(
            fmt::format("{}'s outputs:surface must be connection with single "
                        "target Path.\n",
                        mat_abs_path.full_path_name()));
      }
      surfacePath = paths[0];
    } else {
      // May be PhysicsMaterial?
      // Create dummy material

      PUSH_WARN(fmt::format("{}'s outputs:surface isn't authored, so not a valid Material/Shader. Create a default Material\n",
                      mat_abs_path.full_path_name()));


      (*rmat_out) = rmat;
      return true;

      PUSH_ERROR_AND_RETURN(
          fmt::format("{}'s outputs:surface isn't authored.\n",
                      mat_abs_path.full_path_name()));
    }

    const Prim *shaderPrim{nullptr};
    if (!env.stage.find_prim_at_path(
            Path(surfacePath.prim_part(), /* prop part */ ""), shaderPrim,
            &err)) {
      PUSH_ERROR_AND_RETURN(fmt::format(
          "{}'s outputs:surface isn't connected to exising Prim path.\n",
          mat_abs_path.full_path_name()));
    }

    if (!shaderPrim) {
      // this should not happen though.
      PUSH_ERROR_AND_RETURN("[InternalError] invalid Shader Prim.\n");
    }

    const Shader *shader = shaderPrim->as<Shader>();

    if (!shader) {
      PUSH_ERROR_AND_RETURN(
          fmt::format("{}'s outputs:surface must be connected to Shader Prim, "
                      "but connected to `{}` Prim.\n",
                      shaderPrim->prim_type_name()));
    }

    // Currently must be UsdPreviewSurface
    const UsdPreviewSurface *psurface = shader->value.as<UsdPreviewSurface>();
    if (!psurface) {
      PUSH_ERROR_AND_RETURN(
          fmt::format("Shader's info:id must be UsdPreviewSurface, but got {}",
                      shader->info_id));
    }

    // prop part must be `outputs:surface` for now.
    if (surfacePath.prop_part() != "outputs:surface") {
      PUSH_ERROR_AND_RETURN(
          fmt::format("{}'s outputs:surface connection must point to property "
                      "`outputs:surface`, but got `{}`",
                      mat_abs_path.full_path_name(), surfacePath.prop_part()));
    }

    PreviewSurfaceShader pss;
    if (!ConvertPreviewSurfaceShader(env, surfacePath, *psurface, &pss)) {
      PUSH_ERROR_AND_RETURN(fmt::format(
          "Failed to convert UsdPreviewSurface : {}", surfacePath.prim_part()));
    }

    rmat.surfaceShader = pss;
  }

  DCOUT("Converted Material: " << mat_abs_path);

  (*rmat_out) = rmat;
  return true;
}

namespace {

struct MeshVisitorEnv {
  RenderSceneConverter *converter{nullptr};
  const RenderSceneConverterEnv *env{nullptr};
};

bool MeshVisitor(const tinyusdz::Path &abs_path, const tinyusdz::Prim &prim,
                 const int32_t level, void *userdata, std::string *err) {
  if (!userdata) {
    if (err) {
      (*err) += "userdata pointer must be filled.";
    }
    return false;
  }

  MeshVisitorEnv *visitorEnv = reinterpret_cast<MeshVisitorEnv *>(userdata);

  if (level > 1024 * 1024) {
    if (err) {
      (*err) += "Scene graph is too deep.\n";
    }
    // Too deep
    return false;
  }

  if (const tinyusdz::GeomMesh *pmesh = prim.as<tinyusdz::GeomMesh>()) {
    // Collect GeomSubsets
    // std::vector<const tinyusdz::GeomSubset *> subsets = GetGeomSubsets(;

    DCOUT("Mesh: " << abs_path);

    if (!pmesh->points.authored()) {
      // Maybe Collider mesh? Ignore for now.
      DCOUT(fmt::format("Mesh {} does not author `points` attribute(Maybe Collider mesh?). Ignore it for now", abs_path));
      return true;
    }

    //
    // First convert Material assigned to GeomMesh.
    //
    // - If prim has GeomSubset with materialBind, convert it to per-face
    // material.
    // - If prim has materialBind, convert it to RenderMesh's material.
    //

    auto ConvertBoundMaterial = [&](const Path &bound_material_path,
                                    const tinyusdz::Material *bound_material,
                                    int64_t &rmaterial_id) -> bool {
      std::vector<RenderMaterial> &rmaterials =
          visitorEnv->converter->materials;

      const auto matIt = visitorEnv->converter->materialMap.find(
          bound_material_path.full_path_name());

      if (matIt != visitorEnv->converter->materialMap.s_end()) {
        // Got material in the cache.
        uint64_t mat_id = matIt->second;
        if (mat_id >= visitorEnv->converter->materials
                          .size()) {  // this should not happen though
          if (err) {
            (*err) += "Material index out-of-range.\n";
          }
          return false;
        }

        if (mat_id >= size_t((std::numeric_limits<int32_t>::max)())) {
          if (err) {
            (*err) += "Material index too large.\n";
          }
          return false;
        }

        rmaterial_id = int64_t(mat_id);

      } else {
        RenderMaterial rmat;
        if (!visitorEnv->converter->ConvertMaterial(*visitorEnv->env,
                                                    bound_material_path,
                                                    *bound_material, &rmat)) {
          if (err) {
            (*err) += fmt::format("Material conversion failed: {}",
                                  bound_material_path);
          }
          return false;
        }

        // Assign new material ID
        uint64_t mat_id = rmaterials.size();

        if (mat_id >= uint64_t((std::numeric_limits<int32_t>::max)())) {
          if (err) {
            (*err) += "Material index too large.\n";
          }
          return false;
        }
        rmaterial_id = int64_t(mat_id);

        visitorEnv->converter->materialMap.add(
            bound_material_path.full_path_name(), uint64_t(rmaterial_id));
        DCOUT("Added renderMaterial: " << mat_id << " " << rmat.abs_path
                                       << " ( " << rmat.name << " ) ");

        rmaterials.push_back(rmat);
      }

      return true;
    };

    // Convert bound materials in GeomSubsets
    //
    // key: subset Prim name
    std::map<std::string, MaterialPath> subset_material_path_map;
    std::vector<const GeomSubset *> material_subsets;
    {
      material_subsets = GetMaterialBindGeomSubsets(prim);

      for (const auto &psubset : material_subsets) {
        MaterialPath mpath;
        mpath.default_texcoords_primvar_name =
            visitorEnv->env->mesh_config.default_texcoords_primvar_name;

        Path subset_abs_path = abs_path.AppendElement(psubset->name);

        // front and back
        {
          tinyusdz::Path bound_material_path;
          const tinyusdz::Material *bound_material{nullptr};
          bool ret = tinyusdz::tydra::GetBoundMaterial(
              visitorEnv->env->stage,
              /* GeomSubset prim path */ subset_abs_path,
              /* purpose */ "", &bound_material_path, &bound_material, err);

          if (ret && bound_material) {
            int64_t rmaterial_id = -1;  // not used.

            if (!ConvertBoundMaterial(bound_material_path, bound_material,
                                      rmaterial_id)) {
              if (err) {
                (*err) += "Convert boundMaterial failed: " + bound_material_path.full_path_name();
              }
              return false;
            }

            mpath.material_path = bound_material_path.full_path_name();
            DCOUT("GeomSubset " << subset_abs_path << " : Bound material path: "
                                << mpath.backface_material_path);
          }
        }

        std::string backface_purpose =
            visitorEnv->env->material_config
                .default_backface_material_purpose_name;

        if (!backface_purpose.empty() &&
            psubset->has_materialBinding(value::token(backface_purpose))) {
          DCOUT("backface_material_purpose "
                << visitorEnv->env->material_config
                       .default_backface_material_purpose_name);
          tinyusdz::Path bound_material_path;
          const tinyusdz::Material *bound_material{nullptr};
          bool ret = tinyusdz::tydra::GetBoundMaterial(
              visitorEnv->env->stage,
              /* GeomSubset prim path */ subset_abs_path,
              /* purpose */
              visitorEnv->env->material_config
                  .default_backface_material_purpose_name,
              &bound_material_path, &bound_material, err);

          if (ret && bound_material) {
            int64_t rmaterial_id = -1;  // not used

            if (!ConvertBoundMaterial(bound_material_path, bound_material,
                                      rmaterial_id)) {
              if (err) {
                (*err) += "Convert boundMaterial failed: " + bound_material_path.full_path_name();
              }
              return false;
            }

            mpath.backface_material_path = bound_material_path.full_path_name();
            DCOUT("GeomSubset " << subset_abs_path
                                << " : Bound backface material path: "
                                << mpath.backface_material_path);
          }
        }

        subset_material_path_map[psubset->name] = mpath;
      }
    }

    MaterialPath material_path;
    material_path.default_texcoords_primvar_name =
        visitorEnv->env->mesh_config.default_texcoords_primvar_name;
    // TODO: Implement feature to assign default material
    // id(MaterialPath::default_material_id) when no bound material found.

    {
      const std::string mesh_path_str = abs_path.full_path_name();

      // Front and back material.
      {
        tinyusdz::Path bound_material_path;
        const tinyusdz::Material *bound_material{nullptr};
        bool ret = tinyusdz::tydra::GetBoundMaterial(
            visitorEnv->env->stage, /* GeomMesh prim path */ abs_path,
            /* purpose */ "", &bound_material_path, &bound_material, err);

        DCOUT("Bound material found: " << ret);
        if (ret && bound_material) {
          int64_t rmaterial_id = -1;  // not used

          if (!ConvertBoundMaterial(bound_material_path, bound_material,
                                    rmaterial_id)) {
            if (err) {
              (*err) += "Convert boundMaterial failed: " + bound_material_path.full_path_name();
            }
            return false;
          }

          material_path.material_path = bound_material_path.full_path_name();
          DCOUT("Bound material path: " << material_path.material_path);
        }
      }

      std::string backface_purpose =
          visitorEnv->env->material_config
              .default_backface_material_purpose_name;

      if (!backface_purpose.empty() &&
          pmesh->has_materialBinding(value::token(backface_purpose))) {
        tinyusdz::Path bound_material_path;
        const tinyusdz::Material *bound_material{nullptr};
        bool ret = tinyusdz::tydra::GetBoundMaterial(
            visitorEnv->env->stage, /* GeomMesh prim path */ abs_path,
            /* purpose */
            visitorEnv->env->material_config
                .default_backface_material_purpose_name,
            &bound_material_path, &bound_material, err);

        if (ret && bound_material) {
          int64_t rmaterial_id = -1;  // not used

          if (!ConvertBoundMaterial(bound_material_path, bound_material,
                                    rmaterial_id)) {
            if (err) {
              (*err) += "Convert boundMaterial failed: " + bound_material_path.full_path_name();
            }
            return false;
          }

          material_path.backface_material_path =
              bound_material_path.full_path_name();
          DCOUT("Bound backface material path: "
                << material_path.backface_material_path);
        }
      }

      // BlendShapes
      std::vector<std::pair<std::string, const BlendShape *>> blendshapes;
      {
        std::string local_err;
        blendshapes = GetBlendShapes(visitorEnv->env->stage, prim, &local_err);
        if (local_err.size()) {
          if (err) {
            (*err) += fmt::format("Failed to get BlendShapes prims. err = {}", local_err);
          }
        }
      }
      DCOUT("# of blendshapes : " << blendshapes.size());

      RenderMesh rmesh;

      if (!visitorEnv->converter->ConvertMesh(
              *visitorEnv->env, abs_path, *pmesh, material_path,
              subset_material_path_map, visitorEnv->converter->materialMap,
              material_subsets, blendshapes, &rmesh)) {
        if (err) {
          (*err) += fmt::format("Mesh conversion failed: {}",
                                abs_path.full_path_name());
          (*err) += "\n" + visitorEnv->converter->GetError() + "\n";

        }
        return false;
      }

      uint64_t mesh_id = uint64_t(visitorEnv->converter->meshes.size());
      if (mesh_id >= size_t((std::numeric_limits<int32_t>::max)())) {
        if (err) {
          (*err) += "Mesh index too large.\n";
        }
        return false;
      }
      visitorEnv->converter->meshMap.add(abs_path.full_path_name(), mesh_id);

      visitorEnv->converter->meshes.emplace_back(std::move(rmesh));
    }
  }

  return true;  // continue traversal
}

}  // namespace

bool RenderSceneConverter::ConvertSkelAnimation(const RenderSceneConverterEnv &env,
                                            const Path &abs_path,
                                            const SkelAnimation &skelAnim,
                                            Animation *anim_out) {
  // The spec says
  // """
  // An animation source is only valid if its translation, rotation, and scale components are all authored, storing arrays size to the same size as the authored joints array.
  // """
  //
  // SkelAnimation contains
  // - Joint animations(translation, rotation, scale)
  // - BlendShape animations(weights)

  std::vector<value::token> joints;

  if (skelAnim.joints.authored()) {
    if (!EvaluateTypedAttribute(env.stage, skelAnim.joints, "joints", &joints, &_err)) {
      PUSH_ERROR_AND_RETURN(fmt::format("Failed to evaluate `joints` in SkelAnimation Prim : {}", abs_path));
    }

    if (!skelAnim.rotations.authored() ||
        !skelAnim.translations.authored() ||
        !skelAnim.scales.authored()) {

      PUSH_ERROR_AND_RETURN(fmt::format("`translations`, `rotations` and `scales` must be all authored for SkelAnimation Prim {}. authored flags: translations {}, rotations {}, scales {}", abs_path, skelAnim.translations.authored() ? "yes" : "no",
      skelAnim.rotations.authored() ? "yes" : "no",
      skelAnim.scales.authored() ? "yes" : "no"));
    }
  }

  // TODO: inbetweens BlendShape
  std::vector<value::token> blendShapes;
  if (skelAnim.blendShapes.authored()) {
    if (!EvaluateTypedAttribute(env.stage, skelAnim.blendShapes, "blendShapes", &blendShapes, &_err)) {
      PUSH_ERROR_AND_RETURN(fmt::format("Failed to evaluate `blendShapes` in SkelAnimation Prim : {}", abs_path));
    }

    if (!skelAnim.blendShapeWeights.authored()) {
      PUSH_ERROR_AND_RETURN(fmt::format("`blendShapeWeights` must be authored for SkelAnimation Prim {}", abs_path));
    }

  }


  //
  // Reorder values[channels][timeCode][jointId] into values[jointId][channels][timeCode]
  //

  std::map<std::string, std::map<AnimationChannel::ChannelType, AnimationChannel>> channelMap;

  // Joint animations
  if (joints.size()) {
    StringAndIdMap jointIdMap;

    for (const auto &joint : joints) {
      uint64_t id = jointIdMap.size();
      jointIdMap.add(joint.str(), id);
    }

    Animatable<std::vector<value::float3>> translations;
    if (!skelAnim.translations.get_value(&translations)) {
      PUSH_ERROR_AND_RETURN(fmt::format("Failed to get `translations` attribute of SkelAnimation. Maybe ValueBlock or connection? : {}", abs_path));
    }

    Animatable<std::vector<value::quatf>> rotations;
    if (!skelAnim.rotations.get_value(&rotations)) {
      PUSH_ERROR_AND_RETURN(fmt::format("Failed to get `rotations` attribute of SkelAnimation. Maybe ValueBlock or connection? : {}", abs_path));
    }

    Animatable<std::vector<value::half3>> scales;
    if (!skelAnim.scales.get_value(&scales)) {
      PUSH_ERROR_AND_RETURN(fmt::format("Failed to get `scales` attribute of SkelAnimation. Maybe ValueBlock or connection? : {}", abs_path));
    }

    DCOUT("translations: has_timesamples " << translations.has_timesamples());
    DCOUT("translations: has_default " << translations.has_default());
    DCOUT("rotations: has_timesamples " << rotations.has_timesamples());
    DCOUT("rotations: has_default " << rotations.has_default());
    DCOUT("scales: has_timesamples " << scales.has_timesamples());
    DCOUT("scales: has_default " << scales.has_default());

    //
    // timesamples value
    //

    if (translations.has_timesamples()) {
      DCOUT("Convert ttranslations");
      const TypedTimeSamples<std::vector<value::float3>> &ts_txs = translations.get_timesamples();

      if (ts_txs.get_samples().empty()) {
        PUSH_ERROR_AND_RETURN(fmt::format("`translations` timeSamples in SkelAnimation is empty : {}", abs_path));
      }

      for (const auto &sample : ts_txs.get_samples()) {
        if (!sample.blocked) {
          // length check
          if (sample.value.size() != joints.size()) {
            PUSH_ERROR_AND_RETURN(fmt::format("Array length mismatch in SkelAnimation. timeCode {} translations.size {} must be equal to joints.size {} : {}", sample.t, sample.value.size(), joints.size(), abs_path));
          }

          for (size_t j = 0; j < sample.value.size(); j++) {
            AnimationSample<value::float3> s;
            s.t = float(sample.t);
            s.value = sample.value[j];

            std::string jointName = jointIdMap.at(j);
            auto &it = channelMap[jointName][AnimationChannel::ChannelType::Translation];
            if (it.translations.samples.empty()) {
              it.type = AnimationChannel::ChannelType::Translation;
            }
            it.translations.samples.push_back(s);
          }

        }
      }
    }

    if (rotations.has_timesamples()) {
      const TypedTimeSamples<std::vector<value::quatf>> &ts_rots = rotations.get_timesamples();
      DCOUT("Convert rotations");
      for (const auto &sample : ts_rots.get_samples()) {
        if (!sample.blocked) {
          if (sample.value.size() != joints.size()) {
            PUSH_ERROR_AND_RETURN(fmt::format("Array length mismatch in SkelAnimation. timeCode {} rotations.size {} must be equal to joints.size {} : {}", sample.t, sample.value.size(), joints.size(), abs_path));
          }
          for (size_t j = 0; j < sample.value.size(); j++) {
            AnimationSample<value::float4> s;
            s.t = float(sample.t);
            s.value[0] = sample.value[j][0];
            s.value[1] = sample.value[j][1];
            s.value[2] = sample.value[j][2];
            s.value[3] = sample.value[j][3];

            std::string jointName = jointIdMap.at(j);
            auto &it = channelMap[jointName][AnimationChannel::ChannelType::Rotation];
            if (it.rotations.samples.empty()) {
              it.type = AnimationChannel::ChannelType::Rotation;
            }
            it.rotations.samples.push_back(s);
          }

        }
      }
    }

    if (scales.has_timesamples()) {
      const TypedTimeSamples<std::vector<value::half3>> &ts_scales = scales.get_timesamples();
      DCOUT("Convert scales");
      for (const auto &sample : ts_scales.get_samples()) {
        if (!sample.blocked) {
          if (sample.value.size() != joints.size()) {
            PUSH_ERROR_AND_RETURN(fmt::format("Array length mismatch in SkelAnimation. timeCode {} scales.size {} must be equal to joints.size {} : {}", sample.t, sample.value.size(), joints.size(), abs_path));
          }

          for (size_t j = 0; j < sample.value.size(); j++) {
            AnimationSample<value::float3> s;
            s.t = float(sample.t);
            s.value[0] = value::half_to_float(sample.value[j][0]);
            s.value[1] = value::half_to_float(sample.value[j][1]);
            s.value[2] = value::half_to_float(sample.value[j][2]);

            std::string jointName = jointIdMap.at(j);
            auto &it = channelMap[jointName][AnimationChannel::ChannelType::Scale];
            if (it.scales.samples.empty()) {
              it.type = AnimationChannel::ChannelType::Scale;
            }
            it.scales.samples.push_back(s);
          }

        }
      }
    }

    //
    // value at 'default' time.
    //

    // Get value and also do length check for scalar(non timeSampled) animation value.
    if (translations.has_default()) {
      DCOUT("translation at default time");
      std::vector<value::float3> translation;
      if (!translations.get_scalar(&translation)) {
        PUSH_ERROR_AND_RETURN(fmt::format("Failed to get `translations` attribute in SkelAnimation: {}", abs_path));
      }
      if (translation.size() != joints.size()) {
        PUSH_ERROR_AND_RETURN(fmt::format("Array length mismatch in SkelAnimation. translations.default.size {} must be equal to joints.size {} : {}", translation.size(), joints.size(), abs_path));
      }

      for (size_t j = 0; j < joints.size(); j++) {
        std::string jointName = jointIdMap.at(j);
        auto &it = channelMap[jointName][AnimationChannel::ChannelType::Translation];
        it.translations.static_value = translation[j];
      }
    }

    if (rotations.has_default()) {
      DCOUT("rotations at default time");
      std::vector<value::float4> rotation;
      std::vector<value::quatf> _rotation;
      if (!rotations.get_scalar(&_rotation)) {
        PUSH_ERROR_AND_RETURN(fmt::format("Failed to get `rotations` attribute in SkelAnimation: {}", abs_path));
      }
      if (_rotation.size() != joints.size()) {
        PUSH_ERROR_AND_RETURN(fmt::format("Array length mismatch in SkelAnimation. rotations.default.size {} must be equal to joints.size {} : {}", _rotation.size(), joints.size(), abs_path));
      }
      std::transform(_rotation.begin(), _rotation.end(), std::back_inserter(rotation), [](const value::quatf &v) {
        value::float4 ret;
        // pxrUSD's TfQuat also uses xyzw memory order.
        ret[0] = v[0];
        ret[1] = v[1];
        ret[2] = v[2];
        ret[3] = v[3];
        return ret;
      });

      for (size_t j = 0; j < joints.size(); j++) {
        std::string jointName = jointIdMap.at(j);
        auto &it = channelMap[jointName][AnimationChannel::ChannelType::Rotation];
        it.rotations.static_value = rotation[j];
      }
    }

    if (scales.has_default()) {
      DCOUT("scales at default time");
      std::vector<value::float3> scale;
      std::vector<value::half3> _scale;
      if (!scales.get_scalar(&_scale)) {
        PUSH_ERROR_AND_RETURN(fmt::format("Failed to get `scales` attribute in SkelAnimation: {}", abs_path));
      }
      if (_scale.size() != joints.size()) {
        PUSH_ERROR_AND_RETURN(fmt::format("Array length mismatch in SkelAnimation. scale.default.size {} must be equal to joints.size {} : {}", _scale.size(), joints.size(), abs_path));
      }
      // half -> float
      std::transform(_scale.begin(), _scale.end(), std::back_inserter(scale), [](const value::half3 &v) {
        value::float3 ret;
        ret[0] = value::half_to_float(v[0]);
        ret[1] = value::half_to_float(v[1]);
        ret[2] = value::half_to_float(v[2]);
        return ret;
      });
      for (size_t j = 0; j < joints.size(); j++) {
        std::string jointName = jointIdMap.at(j);
        auto &it = channelMap[jointName][AnimationChannel::ChannelType::Scale];
        it.scales.static_value = scale[j];
      }
    }


#if 0 // TODO: remove
    if (!is_translations_timesamples) {
      DCOUT("Reorder translation samples");
      // Create a channel value with single-entry
      // Use USD TimeCode::Default for static sample.
      for (const auto &joint : joints) {
        channelMap[joint.str()][AnimationChannel::ChannelType::Translation].type = AnimationChannel::ChannelType::Translation;

        AnimationSample<value::float3> s;
        s.t = std::numeric_limits<float>::quiet_NaN();
        uint64_t joint_id = jointIdMap.at(joint.str());
        s.value = translation[joint_id];
        channelMap[joint.str()][AnimationChannel::ChannelType::Translation].translations.samples.clear();
        channelMap[joint.str()][AnimationChannel::ChannelType::Translation].translations.samples.push_back(s);
      }
    }

    if (!is_rotations_timesamples) {
      DCOUT("Reorder rotation samples");
      for (const auto &joint : joints) {
        channelMap[joint.str()][AnimationChannel::ChannelType::Rotation].type = AnimationChannel::ChannelType::Rotation;

        AnimationSample<value::float4> s;
        s.t = std::numeric_limits<float>::quiet_NaN();
        uint64_t joint_id = jointIdMap.at(joint.str());
        DCOUT("rot joint_id " << joint_id);
        s.value = rotation[joint_id];
        channelMap[joint.str()][AnimationChannel::ChannelType::Rotation].rotations.samples.clear();
        channelMap[joint.str()][AnimationChannel::ChannelType::Rotation].rotations.samples.push_back(s);
      }
    }

    if (!is_scales_timesamples) {
      DCOUT("Reorder scale samples");
      for (const auto &joint : joints) {
        channelMap[joint.str()][AnimationChannel::ChannelType::Scale].type = AnimationChannel::ChannelType::Scale;

        AnimationSample<value::float3> s;
        s.t = std::numeric_limits<float>::quiet_NaN();
        uint64_t joint_id = jointIdMap.at(joint.str());
        s.value = scale[joint_id];
        channelMap[joint.str()][AnimationChannel::ChannelType::Scale].scales.samples.clear();
        channelMap[joint.str()][AnimationChannel::ChannelType::Scale].scales.samples.push_back(s);
      }
    }
#endif
  }

  // BlendShape animations
  if (blendShapes.size()) {

    std::map<std::string, AnimationSampler<float>> weightsMap;

    // Blender 4.1 may export empty bendShapeWeights. We'll accept it.
    //
    // float[] blendShapeWeights
    if (skelAnim.blendShapeWeights.is_value_empty()) {
      for (const auto &bs : blendShapes) {
        weightsMap[bs.str()].static_value = 1.0f;
      }
    } else {

      Animatable<std::vector<float>> weights;
      if (!skelAnim.blendShapeWeights.get_value(&weights)) {
        PUSH_ERROR_AND_RETURN(fmt::format("Failed to get `blendShapeWeights` attribute of SkelAnimation. Maybe ValueBlock or connection? : {}", abs_path));
      }

      if (weights.has_timesamples()) {

        const TypedTimeSamples<std::vector<float>> &ts_weights = weights.get_timesamples();
        DCOUT("Convert timeSampledd weights");
        for (const auto &sample : ts_weights.get_samples()) {
          if (!sample.blocked) {
            if (sample.value.size() != blendShapes.size()) {
              PUSH_ERROR_AND_RETURN(fmt::format("Array length mismatch in SkelAnimation. timeCode {} blendShapeWeights.size {} must be equal to blendShapes.size {} : {}", sample.t, sample.value.size(), blendShapes.size(), abs_path));
            }

            for (size_t j = 0; j < sample.value.size(); j++) {
              AnimationSample<float> s;
              s.t = float(sample.t);
              s.value = sample.value[j];

              const std::string &targetName = blendShapes[j].str();
              weightsMap[targetName].samples.push_back(s);
            }

          }
        }
      }

      if (weights.has_default()) {
        std::vector<float> ws;
        if (!weights.get_scalar(&ws)) {
          PUSH_ERROR_AND_RETURN(fmt::format("Failed to get default value of `blendShapeWeights` attribute of SkelAnimation is invalid : {}", abs_path));
        }

        if (ws.size() != blendShapes.size()) {
          PUSH_ERROR_AND_RETURN(fmt::format("blendShapeWeights.size {} must be equal to blendShapes.size {} : {}", ws.size(), blendShapes.size(), abs_path));
        }

        for (size_t i = 0; i < blendShapes.size(); i++) {
          weightsMap[blendShapes[i].str()].static_value = ws[i];
        }

      } else {
        PUSH_ERROR_AND_RETURN(fmt::format("Internal error. `blendShapeWeights` attribute of SkelAnimation is invalid : {}", abs_path));
      }

    }

    anim_out->blendshape_weights_map = std::move(weightsMap);
  }

  anim_out->abs_path = abs_path.full_path_name();
  anim_out->prim_name = skelAnim.name;
  anim_out->display_name = skelAnim.metas().displayName.value_or("");

  anim_out->channels_map = std::move(channelMap);

  return true;
}

bool RenderSceneConverter::BuildNodeHierarchyImpl(
    const RenderSceneConverterEnv &env, const std::string &parentPrimPath,
    const XformNode &node, Node &out_rnode) {
  Node rnode;

  std::string primPath;
  if (parentPrimPath.empty()) {
    primPath = "/" + node.element_name;
  } else {
    primPath = parentPrimPath + "/" + node.element_name;
  }

  const tinyusdz::Prim *prim = node.prim;
  if (prim) {
    rnode.prim_name = prim->element_name();
    rnode.abs_path = primPath;
    rnode.display_name = prim->metas().displayName.value_or("");

    DCOUT("rnode.prim_name " << rnode.prim_name);
    DCOUT("node.local_mat " << node.get_local_matrix());
    DCOUT("node.has_resetXform " << node.has_resetXformStack());
    DCOUT("prim.type_name " << prim->type_name());
    DCOUT("prim.type_id " << prim->type_id());
    DCOUT("xform " << value::TYPE_ID_GEOM_XFORM);

    if (prim->type_id() == value::TYPE_ID_GEOM_MESH) {
      // GeomMesh(GPrim) also has xform.
      rnode.local_matrix = node.get_local_matrix();
      rnode.global_matrix = node.get_world_matrix();
      rnode.nodeType = NodeType::Mesh;
      rnode.has_resetXform = node.has_resetXformStack();

      if (meshMap.count(primPath)) {
        rnode.id = int32_t(meshMap.at(primPath));
      } else {
        rnode.id = -1;
      }
    } else if (prim->type_id() == value::TYPE_ID_GEOM_CAMERA) {
      rnode.local_matrix = node.get_local_matrix();
      rnode.global_matrix = node.get_world_matrix();
      rnode.nodeType = NodeType::Mesh;
      rnode.has_resetXform = node.has_resetXformStack();
      rnode.nodeType = NodeType::Camera;
      rnode.id = -1;  // TODO: Assign index to cameras
    } else if (prim->type_id() == value::TYPE_ID_GEOM_XFORM) {
      rnode.local_matrix = node.get_local_matrix();
      rnode.global_matrix = node.get_world_matrix();
      DCOUT("rnode.local_matrix " << rnode.local_matrix);
      rnode.global_matrix = node.get_world_matrix();
      rnode.has_resetXform = node.has_resetXformStack();
      rnode.nodeType = NodeType::Xform;
    } else if (prim->type_id() == value::TYPE_ID_SCOPE) {
      // NOTE: get_local_matrix() should return identity matrix.
      rnode.local_matrix = node.get_local_matrix();
      rnode.global_matrix = node.get_world_matrix();
      rnode.has_resetXform = node.has_resetXformStack();
      rnode.nodeType = NodeType::Xform;
    } else if (prim->type_id() == value::TYPE_ID_MODEL) {
      rnode.local_matrix = node.get_local_matrix();
      rnode.global_matrix = node.get_world_matrix();
      rnode.has_resetXform = node.has_resetXformStack();
      rnode.nodeType = NodeType::Xform;
    } else if ((prim->type_id() > value::TYPE_ID_MODEL_BEGIN) && (prim->type_id() < value::TYPE_ID_GEOM_END)) {
      // Other Geom prims(e.g. GeomCube)
      rnode.local_matrix = node.get_local_matrix();
      rnode.global_matrix = node.get_world_matrix();
      rnode.has_resetXform = node.has_resetXformStack();
      rnode.nodeType = NodeType::Xform;
    } else if (IsLightPrim(*prim)) {
      rnode.local_matrix = node.get_local_matrix();
      rnode.global_matrix = node.get_world_matrix();
      rnode.has_resetXform = node.has_resetXformStack();
      if (prim->type_id() == value::TYPE_ID_LUX_DISTANT) {
        rnode.nodeType = NodeType::DirectionalLight;
      } else if (prim->type_id() == value::TYPE_ID_LUX_SPHERE) {
        // treat sphereLight as pointLight
        rnode.nodeType = NodeType::PointLight;
      } else {
        // TODO
        rnode.nodeType = NodeType::Xform;
      }
      rnode.id = -1;  // TODO: index to lights
    } else {
      // ignore other node types.
      DCOUT("Unknown/Unsupported prim. " << prim->type_name());

      // Setup as xform for now.
      rnode.local_matrix = node.get_local_matrix();
      rnode.global_matrix = node.get_world_matrix();
      rnode.has_resetXform = node.has_resetXformStack();
      rnode.nodeType = NodeType::Xform;
    }
  }

  for (const auto &child : node.children) {
    Node child_rnode;
    if (!BuildNodeHierarchyImpl(env, primPath, child, child_rnode)) {
      return false;
    }

    rnode.children.emplace_back(std::move(child_rnode));
  }

  out_rnode = std::move(rnode);

  return true;
}

bool RenderSceneConverter::BuildNodeHierarchy(
    const RenderSceneConverterEnv &env, const XformNode &root) {
  std::string defaultRootNode = env.stage.metas().defaultPrim.str();

  default_node = -1;

  for (const auto &rootNode : root.children) {
    Node root_node;
    if (!BuildNodeHierarchyImpl(env, /* root */ "", rootNode, root_node)) {
      return false;
    }

    if (defaultRootNode == rootNode.element_name) {
      default_node = int(root_nodes.size());
    }

    root_nodeMap.add("/" + rootNode.element_name, root_nodes.size());
    root_nodes.push_back(root_node);
  }

  return true;
}

bool RenderSceneConverter::ConvertToRenderScene(
    const RenderSceneConverterEnv &env, RenderScene *scene) {
  if (!scene) {
    PUSH_ERROR_AND_RETURN("nullptr for RenderScene argument.");
  }

  // 1. Convert Xform
  // 2. Convert Material/Texture
  // 3. Convert Mesh/SkinWeights/BlendShapes
  // 4. Convert Skeleton(bones)
  // 5. Build node hierarchy
  // TODO: Convert lights

  //
  // 1. Build Xform at specified time.
  //    Each Prim in Stage is converted to XformNode.
  //
  XformNode xform_node;
  if (!BuildXformNodeFromStage(env.stage, &xform_node, env.timecode)) {
    PUSH_ERROR_AND_RETURN("Failed to build Xform node hierarchy.\n");
  }

  std::string err;

  //
  // 2. Convert Material/Texture
  // 3. Convert Mesh/SkinWeights/BlendShapes
  // 4. Convert Skeleton(bones) and SkelAnimation
  //
  // Material conversion will be done in MeshVisitor.
  //
  MeshVisitorEnv menv;
  menv.env = &env;
  menv.converter = this;

  bool ret = tydra::VisitPrims(env.stage, MeshVisitor, &menv, &err);

  if (!ret) {
    PUSH_ERROR_AND_RETURN(err);
  }

  //
  // 5. Build node hierarchy from XformNode and meshes, materials, skeletons,
  // etc.
  //
  if (!BuildNodeHierarchy(env, xform_node)) {
    return false;
  }

  // render_scene.meshMap = std::move(meshMap);
  // render_scene.materialMap = std::move(materialMap);
  // render_scene.textureMap = std::move(textureMap);
  // render_scene.imageMap = std::move(imageMap);
  // render_scene.bufferMap = std::move(bufferMap);

  RenderScene render_scene;
  render_scene.usd_filename = env.usd_filename;
  render_scene.default_root_node = 0;
  if (default_node > -1) {
    if (size_t(default_node) >= root_nodes.size()) {
      PushWarn("Invalid default_node id. Use 0 for default_node id.");
    } else {
      render_scene.default_root_node = uint32_t(default_node);
    }
  }

  render_scene.nodes = std::move(root_nodes);
  render_scene.meshes = std::move(meshes);
  render_scene.textures = std::move(textures);
  render_scene.images = std::move(images);
  render_scene.buffers = std::move(buffers);
  render_scene.materials = std::move(materials);
  render_scene.skeletons = std::move(skeletons);
  render_scene.animations = std::move(animations);

  (*scene) = std::move(render_scene);
  return true;
}

bool RenderSceneConverter::ConvertSkeletonImpl(const RenderSceneConverterEnv &env, const tinyusdz::GeomMesh &mesh,
                       SkelHierarchy *out_skel, nonstd::optional<Animation> *out_anim) {

  if (!out_skel) {
    return false;
  }

  if (!mesh.skeleton.has_value()) {
    return false;
  }

  Path skelPath;

  if (mesh.skeleton.value().is_path()) {
    skelPath = mesh.skeleton.value().targetPath;
  } else if (mesh.skeleton.value().is_pathvector()) {
    // Use the first one
    if (mesh.skeleton.value().targetPathVector.size()) {
      skelPath = mesh.skeleton.value().targetPathVector[0];
    } else {
      PUSH_WARN("`skel:skeleton` has invalid definition.");
    }
  } else {
    PUSH_WARN("`skel:skeleton` has invalid definition.");
  }

  if (skelPath.is_valid()) {
    const Prim *skelPrim{nullptr};
    if (!env.stage.find_prim_at_path(skelPath, skelPrim, &_err)) {
      return false;
    }

    SkelHierarchy dst;
    if (const auto pskel = skelPrim->as<Skeleton>()) {
      SkelNode root;
      if (!BuildSkelHierarchy((*pskel), root, &_err)) {
        return false;
      }
      dst.abs_path = skelPath.prim_part();
      dst.prim_name = skelPrim->element_name();
      dst.display_name = pskel->metas().displayName.value_or("");
      dst.root_node = root;

      if (pskel->animationSource.has_value()) {
        DCOUT("skel:animationSource");

        const Relationship &animSourceRel = pskel->animationSource.value();

        Path animSourcePath;

        if (animSourceRel.is_path()) {
          animSourcePath = animSourceRel.targetPath;
        } else if (animSourceRel.is_pathvector()) {
          // Use the first one
          if (animSourceRel.targetPathVector.size()) {
            animSourcePath = animSourceRel.targetPathVector[0];
          } else {
            PUSH_ERROR_AND_RETURN("`skel:animationSource` has invalid definition.");
          }
        } else {
          PUSH_ERROR_AND_RETURN("`skel:animationSource` has invalid definition.");
        }

        const Prim *animSourcePrim{nullptr};
        if (!env.stage.find_prim_at_path(animSourcePath, animSourcePrim, &_err)) {
          return false;
        }

        if (const auto panim = animSourcePrim->as<SkelAnimation>()) {
          DCOUT("Convert SkelAnimation");
          Animation anim;
          if (!ConvertSkelAnimation(env, animSourcePath, *panim, &anim)) {
            return false;
          }

          DCOUT("Converted SkelAnimation");
          (*out_anim) = anim;

        } else {
          PUSH_ERROR_AND_RETURN(fmt::format("Target Prim of `skel:animationSource` must be `SkelAnimation` Prim, but got `{}`.", animSourcePrim->prim_type_name()));
        }


      }
    } else {
      PUSH_ERROR_AND_RETURN("Prim is not Skeleton.");
    }

    (*out_skel) = dst;
    return true;
  }

  PUSH_ERROR_AND_RETURN("`skel:skeleton` path is invalid.");
}

bool DefaultTextureImageLoaderFunction(
    const value::AssetPath &assetPath, const AssetInfo &assetInfo,
    const AssetResolutionResolver &assetResolver, TextureImage *texImageOut,
    std::vector<uint8_t> *imageData, void *userdata, std::string *warn,
    std::string *err) {
  if (!texImageOut) {
    if (err) {
      (*err) = "`imageOut` argument is nullptr\n";
    }
    return false;
  }

  if (!imageData) {
    if (err) {
      (*err) = "`imageData` argument is nullptr\n";
    }
    return false;
  }

  // TODO: assetInfo
  (void)assetInfo;
  (void)userdata;
  (void)warn;

  std::string resolvedPath = assetResolver.resolve(assetPath.GetAssetPath());

  if (resolvedPath.empty()) {
    if (err) {
      (*err) += fmt::format("Failed to resolve asset path: {}\n",
                            assetPath.GetAssetPath());
    }
    return false;
  }

  Asset asset;
  bool ret = assetResolver.open_asset(resolvedPath, assetPath.GetAssetPath(),
                                      &asset, warn, err);
  if (!ret) {
    if (err) {
      (*err) += fmt::format("Failed to open asset: {}", resolvedPath);
    }
    return false;
  }

  DCOUT("Resolved asset path = " << resolvedPath);

  // TODO: user-defined image loader handler.
  auto result = tinyusdz::image::LoadImageFromMemory(asset.data(), asset.size(),
                                                     resolvedPath);
  if (!result) {
    if (err) {
      (*err) += "Failed to load image file: " + result.error() + "\n";
    }
    return false;
  }

  TextureImage texImage;

  texImage.asset_identifier = resolvedPath;
  texImage.channels = result.value().image.channels;

  const auto &imgret = result.value();

  if (imgret.image.bpp == 8) {
    // assume uint8
    texImage.assetTexelComponentType = ComponentType::UInt8;
  } else if (imgret.image.bpp == 16) {
    if (imgret.image.format == Image::PixelFormat::UInt) {
      texImage.assetTexelComponentType = ComponentType::UInt16;
    } else if (imgret.image.format == Image::PixelFormat::Int) {
      texImage.assetTexelComponentType = ComponentType::Int16;
    } else if (imgret.image.format == Image::PixelFormat::Float) {
      texImage.assetTexelComponentType = ComponentType::Half;
    } else {
      if (err) {
        (*err) += "Invalid image.pixelformat: " + tinyusdz::to_string(imgret.image.format) + "\n";
      }
      return false;
    }

  } else if (imgret.image.bpp == 16) {
    if (imgret.image.format == Image::PixelFormat::UInt) {
      texImage.assetTexelComponentType = ComponentType::UInt32;
    } else if (imgret.image.format == Image::PixelFormat::Int) {
      texImage.assetTexelComponentType = ComponentType::Int32;
    } else if (imgret.image.format == Image::PixelFormat::Float) {
      texImage.assetTexelComponentType = ComponentType::Float;
    } else {
      if (err) {
        (*err) += "Invalid image.pixelformat: " + tinyusdz::to_string(imgret.image.format) + "\n";
      }
      return false;
    }
  } else {
    DCOUT("TODO: bpp = " << result.value().image.bpp);
    if (err) {
      (*err) += "TODO or unsupported bpp: " +
               std::to_string(result.value().image.bpp) + "\n";
    }
    return false;
  }

  texImage.channels = result.value().image.channels;
  texImage.width = result.value().image.width;
  texImage.height = result.value().image.height;

  (*texImageOut) = texImage;

  // raw image data
  (*imageData) = result.value().image.data;

  return true;
}


std::string to_string(ColorSpace cty) {
  std::string s;
  switch (cty) {
    case ColorSpace::sRGB: {
      s = "srgb";
      break;
    }
    case ColorSpace::Lin_sRGB: {
      s = "lin_srgb";
      break;
    }
    case ColorSpace::Raw: {
      s = "raw";
      break;
    }
    case ColorSpace::Rec709: {
      s = "rec709";
      break;
    }
    case ColorSpace::OCIO: {
      s = "ocio";
      break;
    }
    case ColorSpace::Lin_ACEScg: {
      s = "lin_acescg";
      break;
    }
    case ColorSpace::Lin_DisplayP3: {
      s = "lin_displayp3";
      break;
    }
    case ColorSpace::sRGB_DisplayP3: {
      s = "srgb_displayp3";
      break;
    }
    case ColorSpace::Custom: {
      s = "custom";
      break;
    }
    case ColorSpace::Unknown: {
      s = "unknown";
      break;
    }
  }

  return s;
}

bool InferColorSpace(const value::token &tok, ColorSpace *cty) {
  if (!cty) {
    return false;
  }

  if (tok.str() == "raw") {
    (*cty) = ColorSpace::Raw;
  } else if (tok.str() == "Raw") {
    (*cty) = ColorSpace::Raw;
  } else if (tok.str() == "srgb") {
    (*cty) = ColorSpace::sRGB;
  } else if (tok.str() == "sRGB") {
    (*cty) = ColorSpace::sRGB;
  } else if (tok.str() == "linear") { // guess linear_srgb
    (*cty) = ColorSpace::Lin_sRGB;
  } else if (tok.str() == "lin_srgb") {
    (*cty) = ColorSpace::Lin_sRGB;
  } else if (tok.str() == "rec709") {
    (*cty) = ColorSpace::Rec709;
  } else if (tok.str() == "ocio") {
    (*cty) = ColorSpace::OCIO;
  } else if (tok.str() == "lin_displayp3") {
    (*cty) = ColorSpace::Lin_DisplayP3;
  } else if (tok.str() == "srgb_displayp3") {
    (*cty) = ColorSpace::sRGB_DisplayP3;

    //
    // seen in Apple's USDZ model(or OCIO?)
    //

  } else if (tok.str() == "ACES - ACEScg") {
    (*cty) = ColorSpace::Lin_ACEScg;
  } else if (tok.str() == "Input - Texture - sRGB - Display P3") {
    (*cty) = ColorSpace::sRGB_DisplayP3;
  } else if (tok.str() == "Input - Texture - sRGB - sRGB") {
    (*cty) = ColorSpace::sRGB;
  } else if (tok.str() == "custom") {
    (*cty) = ColorSpace::Custom;
  } else {
    return false;
  }

  return true;
}

std::string to_string(NodeType ntype) {
  if (ntype == NodeType::Xform) {
    return "xform";
  } else if (ntype == NodeType::Mesh) {
    return "mesh";
  } else if (ntype == NodeType::Camera) {
    return "camera";
  } else if (ntype == NodeType::PointLight) {
    return "pointLight";
  } else if (ntype == NodeType::DirectionalLight) {
    return "directionalLight";
  } else if (ntype == NodeType::Skeleton) {
    return "skeleton";
  }
  return "???";
}

std::string to_string(ComponentType cty) {
  std::string s;
  switch (cty) {
    case ComponentType::UInt8: {
      s = "uint8";
      break;
    }
    case ComponentType::Int8: {
      s = "int8";
      break;
    }
    case ComponentType::UInt16: {
      s = "uint16";
      break;
    }
    case ComponentType::Int16: {
      s = "int16";
      break;
    }
    case ComponentType::UInt32: {
      s = "uint32";
      break;
    }
    case ComponentType::Int32: {
      s = "int32";
      break;
    }
    case ComponentType::Half: {
      s = "half";
      break;
    }
    case ComponentType::Float: {
      s = "float";
      break;
    }
    case ComponentType::Double: {
      s = "double";
      break;
    }
  }

  return s;
}

std::string to_string(UVTexture::WrapMode mode) {
  std::string s;
  switch (mode) {
    case UVTexture::WrapMode::REPEAT: {
      s = "repeat";
      break;
    }
    case UVTexture::WrapMode::CLAMP_TO_BORDER: {
      s = "clamp_to_border";
      break;
    }
    case UVTexture::WrapMode::CLAMP_TO_EDGE: {
      s = "clamp_to_edge";
      break;
    }
    case UVTexture::WrapMode::MIRROR: {
      s = "mirror";
      break;
    }
  }

  return s;
}

std::string to_string(VertexVariability v) {
  std::string s;

  switch (v) {
    case VertexVariability::Constant: {
      s = "constant";
      break;
    }
    case VertexVariability::Uniform: {
      s = "uniform";
      break;
    }
    case VertexVariability::Varying: {
      s = "varying";
      break;
    }
    case VertexVariability::Vertex: {
      s = "vertex";
      break;
    }
    case VertexVariability::FaceVarying: {
      s = "facevarying";
      break;
    }
    case VertexVariability::Indexed: {
      s = "indexed";
      break;
    }
  }

  return s;
}

std::string to_string(VertexAttributeFormat f) {
  std::string s;

  switch (f) {
    case VertexAttributeFormat::Bool: {
      s = "bool";
      break;
    }
    case VertexAttributeFormat::Char: {
      s = "int8";
      break;
    }
    case VertexAttributeFormat::Char2: {
      s = "int8x2";
      break;
    }
    case VertexAttributeFormat::Char3: {
      s = "int8x3";
      break;
    }
    case VertexAttributeFormat::Char4: {
      s = "int8x4";
      break;
    }
    case VertexAttributeFormat::Byte: {
      s = "uint8";
      break;
    }
    case VertexAttributeFormat::Byte2: {
      s = "uint8x2";
      break;
    }
    case VertexAttributeFormat::Byte3: {
      s = "uint8x3";
      break;
    }
    case VertexAttributeFormat::Byte4: {
      s = "uint8x4";
      break;
    }
    case VertexAttributeFormat::Short: {
      s = "int16";
      break;
    }
    case VertexAttributeFormat::Short2: {
      s = "int16x2";
      break;
    }
    case VertexAttributeFormat::Short3: {
      s = "int16x2";
      break;
    }
    case VertexAttributeFormat::Short4: {
      s = "int16x2";
      break;
    }
    case VertexAttributeFormat::Ushort: {
      s = "uint16";
      break;
    }
    case VertexAttributeFormat::Ushort2: {
      s = "uint16x2";
      break;
    }
    case VertexAttributeFormat::Ushort3: {
      s = "uint16x2";
      break;
    }
    case VertexAttributeFormat::Ushort4: {
      s = "uint16x2";
      break;
    }
    case VertexAttributeFormat::Half: {
      s = "half";
      break;
    }
    case VertexAttributeFormat::Half2: {
      s = "half2";
      break;
    }
    case VertexAttributeFormat::Half3: {
      s = "half3";
      break;
    }
    case VertexAttributeFormat::Half4: {
      s = "half4";
      break;
    }
    case VertexAttributeFormat::Float: {
      s = "float";
      break;
    }
    case VertexAttributeFormat::Vec2: {
      s = "float2";
      break;
    }
    case VertexAttributeFormat::Vec3: {
      s = "float3";
      break;
    }
    case VertexAttributeFormat::Vec4: {
      s = "float4";
      break;
    }
    case VertexAttributeFormat::Int: {
      s = "int";
      break;
    }
    case VertexAttributeFormat::Ivec2: {
      s = "int2";
      break;
    }
    case VertexAttributeFormat::Ivec3: {
      s = "int3";
      break;
    }
    case VertexAttributeFormat::Ivec4: {
      s = "int4";
      break;
    }
    case VertexAttributeFormat::Uint: {
      s = "uint";
      break;
    }
    case VertexAttributeFormat::Uvec2: {
      s = "uint2";
      break;
    }
    case VertexAttributeFormat::Uvec3: {
      s = "uint3";
      break;
    }
    case VertexAttributeFormat::Uvec4: {
      s = "uint4";
      break;
    }
    case VertexAttributeFormat::Double: {
      s = "double";
      break;
    }
    case VertexAttributeFormat::Dvec2: {
      s = "double2";
      break;
    }
    case VertexAttributeFormat::Dvec3: {
      s = "double3";
      break;
    }
    case VertexAttributeFormat::Dvec4: {
      s = "double4";
      break;
    }
    case VertexAttributeFormat::Mat2: {
      s = "mat2";
      break;
    }
    case VertexAttributeFormat::Mat3: {
      s = "mat3";
      break;
    }
    case VertexAttributeFormat::Mat4: {
      s = "mat4";
      break;
    }
    case VertexAttributeFormat::Dmat2: {
      s = "dmat2";
      break;
    }
    case VertexAttributeFormat::Dmat3: {
      s = "dmat3";
      break;
    }
    case VertexAttributeFormat::Dmat4: {
      s = "dmat4";
      break;
    }
  }

  return s;
}

namespace {

template <typename T>
std::string DumpVertexAttributeDataImpl(const T *data, const size_t nbytes,
                                        const size_t stride_bytes,
                                        uint32_t indent) {
  size_t itemsize;

  if (stride_bytes != 0) {
    if ((nbytes % stride_bytes) != 0) {
      return fmt::format(
          "[Invalid VertexAttributeData. input bytes {} must be dividable by "
          "stride_bytes {}(Type {})]",
          nbytes, stride_bytes, value::TypeTraits<T>::type_name());
    }
    itemsize = stride_bytes;
  } else {
    if ((nbytes % sizeof(T)) != 0) {
      return fmt::format(
          "[Invalid VertexAttributeData. input bytes {} must be dividable by "
          "size {}(Type {})]",
          nbytes, sizeof(T), value::TypeTraits<T>::type_name());
    }
    itemsize = sizeof(T);
  }

  size_t nitems = nbytes / itemsize;
  std::string s;
  s += pprint::Indent(indent);
  s += value::print_strided_array_snipped<T>(
      reinterpret_cast<const uint8_t *>(data), stride_bytes, nitems);
  return s;
}

std::string DumpVertexAttributeData(const VertexAttribute &vattr,
                                    uint32_t indent) {
  // Ignore elementSize
#define APPLY_FUNC(__fmt, __basety)                            \
  if (__fmt == vattr.format) {                                 \
    return DumpVertexAttributeDataImpl(                        \
        reinterpret_cast<const __basety *>(vattr.data.data()), \
        vattr.data.size(), vattr.stride, indent);              \
  }

  APPLY_FUNC(VertexAttributeFormat::Bool, uint8_t)
  APPLY_FUNC(VertexAttributeFormat::Char, char)
  APPLY_FUNC(VertexAttributeFormat::Char2, value::char2)
  APPLY_FUNC(VertexAttributeFormat::Char3, value::char3)
  APPLY_FUNC(VertexAttributeFormat::Char4, value::char4)
  APPLY_FUNC(VertexAttributeFormat::Byte, uint8_t)
  APPLY_FUNC(VertexAttributeFormat::Byte2, value::uchar2)
  APPLY_FUNC(VertexAttributeFormat::Byte3, value::uchar3)
  APPLY_FUNC(VertexAttributeFormat::Byte4, value::uchar4)
  APPLY_FUNC(VertexAttributeFormat::Short, int16_t)
  APPLY_FUNC(VertexAttributeFormat::Short2, value::short2)
  APPLY_FUNC(VertexAttributeFormat::Short3, value::short3)
  APPLY_FUNC(VertexAttributeFormat::Short4, value::short4)
  APPLY_FUNC(VertexAttributeFormat::Ushort, uint16_t)
  APPLY_FUNC(VertexAttributeFormat::Ushort2, value::ushort2)
  APPLY_FUNC(VertexAttributeFormat::Ushort3, value::ushort3)
  APPLY_FUNC(VertexAttributeFormat::Ushort4, value::ushort4)
  APPLY_FUNC(VertexAttributeFormat::Half, value::half)
  APPLY_FUNC(VertexAttributeFormat::Half2, value::half2)
  APPLY_FUNC(VertexAttributeFormat::Half3, value::half3)
  APPLY_FUNC(VertexAttributeFormat::Half4, value::half4)
  APPLY_FUNC(VertexAttributeFormat::Float, float)
  APPLY_FUNC(VertexAttributeFormat::Vec2, value::float2)
  APPLY_FUNC(VertexAttributeFormat::Vec3, value::float3)
  APPLY_FUNC(VertexAttributeFormat::Vec4, value::float4)
  APPLY_FUNC(VertexAttributeFormat::Int, int)
  APPLY_FUNC(VertexAttributeFormat::Ivec2, value::int2)
  APPLY_FUNC(VertexAttributeFormat::Ivec3, value::int3)
  APPLY_FUNC(VertexAttributeFormat::Ivec4, value::int4)
  APPLY_FUNC(VertexAttributeFormat::Uint, uint32_t)
  APPLY_FUNC(VertexAttributeFormat::Uvec2, value::half)
  APPLY_FUNC(VertexAttributeFormat::Uvec3, value::half)
  APPLY_FUNC(VertexAttributeFormat::Uvec4, value::half)
  APPLY_FUNC(VertexAttributeFormat::Double, double)
  APPLY_FUNC(VertexAttributeFormat::Dvec2, value::double2)
  APPLY_FUNC(VertexAttributeFormat::Dvec3, value::double2)
  APPLY_FUNC(VertexAttributeFormat::Dvec4, value::double2)
  APPLY_FUNC(VertexAttributeFormat::Mat2, value::matrix2f)
  APPLY_FUNC(VertexAttributeFormat::Mat3, value::matrix3f)
  APPLY_FUNC(VertexAttributeFormat::Mat4, value::matrix4f)
  APPLY_FUNC(VertexAttributeFormat::Dmat2, value::matrix2d)
  APPLY_FUNC(VertexAttributeFormat::Dmat3, value::matrix3d)
  APPLY_FUNC(VertexAttributeFormat::Dmat4, value::matrix4d)
  else {
    return fmt::format("[InternalError. Invalid VertexAttributeFormat: Id{}]",
                       int(vattr.format));
  }

#undef APPLY_FUNC
}

std::string DumpVertexAttribute(const VertexAttribute &vattr, uint32_t indent) {
  std::stringstream ss;

  ss << pprint::Indent(indent) << "count " << vattr.get_data().size() << "\n";
  ss << pprint::Indent(indent) << "format " << quote(to_string(vattr.format))
     << "\n";
  ss << pprint::Indent(indent) << "variability "
     << quote(to_string(vattr.variability)) << "\n";
  ss << pprint::Indent(indent) << "elementSize " << vattr.elementSize << "\n";
  ss << pprint::Indent(indent) << "value "
     << quote(DumpVertexAttributeData(vattr, /* indent */ 0)) << "\n";
  if (vattr.indices.size()) {
    ss << pprint::Indent(indent) << "indices "
       << quote(value::print_array_snipped(vattr.indices)) << "\n";
  }

  return ss.str();
}


std::string DumpNode(const Node &node, uint32_t indent) {
  std::stringstream ss;

  ss << pprint::Indent(indent) << "node {\n";

  ss << pprint::Indent(indent + 1) << "type " << quote(to_string(node.nodeType))
     << "\n";

  ss << pprint::Indent(indent + 1) << "id " << node.id << "\n";

  ss << pprint::Indent(indent + 1) << "prim_name " << quote(node.prim_name)
     << "\n";
  ss << pprint::Indent(indent + 1) << "abs_path " << quote(node.abs_path)
     << "\n";
  ss << pprint::Indent(indent + 1) << "display_name "
     << quote(node.display_name) << "\n";
  ss << pprint::Indent(indent + 1) << "local_matrix "
     << quote(tinyusdz::to_string(node.local_matrix)) << "\n";
  ss << pprint::Indent(indent + 1) << "global_matrix "
     << quote(tinyusdz::to_string(node.global_matrix)) << "\n";

  if (node.children.size()) {
    ss << pprint::Indent(indent + 1) << "children {\n";
    for (const auto &child : node.children) {
      ss << DumpNode(child, indent + 1);
    }
    ss << pprint::Indent(indent + 1) << "}\n";
  }

  ss << pprint::Indent(indent) << "}\n";

  return ss.str();
}

void DumpMaterialSubset(std::stringstream &ss, const MaterialSubset &msubset,
                        uint32_t indent) {
  ss << pprint::Indent(indent) << "material_subset {\n";
  ss << pprint::Indent(indent + 1) << "material_id " << msubset.material_id
     << "\n";
  ss << pprint::Indent(indent + 1) << "indices "
     << quote(value::print_array_snipped(msubset.indices())) << "\n";
  ss << pprint::Indent(indent) << "}\n";
}

std::string DumpMesh(const RenderMesh &mesh, uint32_t indent) {
  std::stringstream ss;

  ss << pprint::Indent(indent) << "mesh {\n";

  ss << pprint::Indent(indent + 1) << "prim_name " << quote(mesh.prim_name)
     << "\n";
  ss << pprint::Indent(indent + 1) << "abs_path " << quote(mesh.abs_path)
     << "\n";
  ss << pprint::Indent(indent + 1) << "display_name "
     << quote(mesh.display_name) << "\n";
  ss << pprint::Indent(indent + 1) << "num_points "
     << std::to_string(mesh.points.size()) << "\n";
  ss << pprint::Indent(indent + 1) << "points \""
     << value::print_array_snipped(mesh.points) << "\"\n";
  ss << pprint::Indent(indent + 1) << "num_faceVertexCounts "
     << std::to_string(mesh.faceVertexCounts().size()) << "\n";
  ss << pprint::Indent(indent + 1) << "faceVertexCounts \""
     << value::print_array_snipped(mesh.faceVertexCounts()) << "\"\n";
  ss << pprint::Indent(indent + 1) << "num_faceVertexIndices "
     << std::to_string(mesh.faceVertexIndices().size()) << "\n";
  ss << pprint::Indent(indent + 1) << "faceVertexIndices \""
     << value::print_array_snipped(mesh.faceVertexIndices()) << "\"\n";
  ss << pprint::Indent(indent + 1) << "materialId "
     << std::to_string(mesh.material_id) << "\n";
  ss << pprint::Indent(indent + 1) << "normals {\n"
     << DumpVertexAttribute(mesh.normals, indent + 2) << "\n";
  ss << pprint::Indent(indent + 1) << "}\n";
  ss << pprint::Indent(indent + 1) << "num_texcoordSlots "
     << std::to_string(mesh.texcoords.size()) << "\n";
  for (const auto &uvs : mesh.texcoords) {
    ss << pprint::Indent(indent + 1) << "texcoords_"
       << std::to_string(uvs.first) << " {\n"
       << DumpVertexAttribute(uvs.second, indent + 2) << "\n";
    ss << pprint::Indent(indent + 1) << "}\n";
  }
  if (mesh.binormals.data.size()) {
    ss << pprint::Indent(indent + 1) << "binormals {\n"
       << DumpVertexAttribute(mesh.binormals, indent + 2) << "\n";
    ss << pprint::Indent(indent + 1) << "}\n";
  }
  if (mesh.tangents.data.size()) {
    ss << pprint::Indent(indent + 1) << "tangents {\n"
       << DumpVertexAttribute(mesh.tangents, indent + 2) << "\n";
    ss << pprint::Indent(indent + 1) << "}\n";
  }

  ss << pprint::Indent(indent + 1) << "skel_id " << mesh.skel_id << "\n";

  if (mesh.joint_and_weights.jointIndices.size()) {
    ss << pprint::Indent(indent + 1) << "skin {\n";
    ss << pprint::Indent(indent + 2) << "geomBindTransform "
       << quote(tinyusdz::to_string(mesh.joint_and_weights.geomBindTransform))
       << "\n";
    ss << pprint::Indent(indent + 2) << "elementSize "
       << mesh.joint_and_weights.elementSize << "\n";
    ss << pprint::Indent(indent + 2) << "jointIndices "
       << quote(value::print_array_snipped(mesh.joint_and_weights.jointIndices))
       << "\n";
    ss << pprint::Indent(indent + 2) << "jointWeights "
       << quote(value::print_array_snipped(mesh.joint_and_weights.jointWeights))
       << "\n";
    ss << pprint::Indent(indent + 1) << "}\n";
  }
  if (mesh.targets.size()) {
    ss << pprint::Indent(indent + 1) << "shapeTargets {\n";

    for (const auto &target : mesh.targets) {
      ss << pprint::Indent(indent + 2) << target.first << " {\n";
      ss << pprint::Indent(indent + 3) << "prim_name " << quote(target.second.prim_name) << "\n";
      ss << pprint::Indent(indent + 3) << "abs_path " << quote(target.second.abs_path) << "\n";
      ss << pprint::Indent(indent + 3) << "display_name " << quote(target.second.display_name) << "\n";
      ss << pprint::Indent(indent + 3) << "pointIndices " << quote(value::print_array_snipped(target.second.pointIndices)) << "\n";
      ss << pprint::Indent(indent + 3) << "pointOffsets " << quote(value::print_array_snipped(target.second.pointOffsets)) << "\n";
      ss << pprint::Indent(indent + 3) << "normalOffsets " << quote(value::print_array_snipped(target.second.normalOffsets)) << "\n";
      ss << pprint::Indent(indent + 2) << "}\n";
    }

    ss << pprint::Indent(indent + 1) << "}\n";

  }
  if (mesh.material_subsetMap.size()) {
    ss << pprint::Indent(indent + 1) << "material_subsets {\n";
    for (const auto &msubset : mesh.material_subsetMap) {
      DumpMaterialSubset(ss, msubset.second, indent + 2);
    }
    ss << pprint::Indent(indent + 1) << "}\n";
  }

  // TODO: primvars

  ss << "\n";

  ss << pprint::Indent(indent) << "}\n";

  return ss.str();
}

namespace detail {

void DumpSkelNode(std::stringstream &ss, const SkelNode &node, uint32_t indent) {

  ss << pprint::Indent(indent) << node.joint_name << " {\n";

  ss << pprint::Indent(indent + 1) << "joint_path " << quote(node.joint_path) << "\n";
  ss << pprint::Indent(indent + 1) << "joint_id " << node.joint_id << "\n";
  ss << pprint::Indent(indent + 1) << "bind_transform " << quote(tinyusdz::to_string(node.bind_transform)) << "\n";
  ss << pprint::Indent(indent + 1) << "rest_transform " << quote(tinyusdz::to_string(node.rest_transform)) << "\n";

  if (node.children.size()) {
    ss << pprint::Indent(indent + 1) << "children {\n";
    for (const auto &child : node.children) {
      DumpSkelNode(ss, child, indent + 2);
    }
    ss << pprint::Indent(indent + 1) << "}\n";
  }

  ss << pprint::Indent(indent) << "}\n";
}


} // namespace detail

std::string DumpSkeleton(const SkelHierarchy &skel, uint32_t indent) {
  std::stringstream ss;

  ss << pprint::Indent(indent) << "skeleton {\n";

  ss << pprint::Indent(indent + 1) << "name " << quote(skel.prim_name) << "\n";
  ss << pprint::Indent(indent + 1) << "abs_path " << quote(skel.abs_path)
     << "\n";
  ss << pprint::Indent(indent + 1) << "anim_id " << skel.anim_id
     << "\n";
  ss << pprint::Indent(indent + 1) << "display_name "
     << quote(skel.display_name) << "\n";

  detail::DumpSkelNode(ss, skel.root_node, indent + 1);

  ss << "\n";

  ss << pprint::Indent(indent) << "}\n";

  return ss.str();
}

namespace detail {

template<typename T>
std::string PrintAnimationSamples(const std::vector<AnimationSample<T>> &samples) {
  std::stringstream ss;

  ss << "[";
  for (size_t i = 0; i < samples.size(); i++) {
    if (i > 0) {
      ss << ", ";
    }

    ss << "(" << samples[i].t << ", " << samples[i].value << ")";
  }
  ss << "]";

  return ss.str();
}

void DumpAnimChannel(std::stringstream &ss, const std::string &name, const std::map<AnimationChannel::ChannelType, AnimationChannel> &channels, uint32_t indent) {

  ss << pprint::Indent(indent) << name << " {\n";

  for (const auto &channel : channels) {
    if (channel.first == AnimationChannel::ChannelType::Translation) {
      ss << pprint::Indent(indent + 1) << "translations " << quote(detail::PrintAnimationSamples(channel.second.translations.samples)) << "\n";
    } else if (channel.first == AnimationChannel::ChannelType::Rotation) {
      ss << pprint::Indent(indent + 1) << "rotations " << quote(detail::PrintAnimationSamples(channel.second.rotations.samples)) << "\n";
    } else if (channel.first == AnimationChannel::ChannelType::Scale) {
      ss << pprint::Indent(indent + 1) << "scales " << quote(detail::PrintAnimationSamples(channel.second.scales.samples)) << "\n";
    }
  }

  ss << pprint::Indent(indent) << "}\n";
}


} // namespace detail

std::string DumpAnimation(const Animation &anim, uint32_t indent) {
  std::stringstream ss;

  ss << pprint::Indent(indent) << "animation {\n";

  ss << pprint::Indent(indent + 1) << "name " << quote(anim.prim_name) << "\n";
  ss << pprint::Indent(indent + 1) << "abs_path " << quote(anim.abs_path)
     << "\n";
  ss << pprint::Indent(indent + 1) << "display_name "
     << quote(anim.display_name) << "\n";

  for (const auto &channel : anim.channels_map) {
    detail::DumpAnimChannel(ss, channel.first, channel.second, indent + 1);
  }

  ss << "\n";

  ss << pprint::Indent(indent) << "}\n";

  return ss.str();
}

std::string DumpCamera(const RenderCamera &camera, uint32_t indent) {
  std::stringstream ss;

  ss << pprint::Indent(indent) << "camera {\n";

  ss << pprint::Indent(indent + 1) << "name " << quote(camera.name) << "\n";
  ss << pprint::Indent(indent + 1) << "abs_path " << quote(camera.abs_path)
     << "\n";
  ss << pprint::Indent(indent + 1) << "display_name "
     << quote(camera.display_name) << "\n";
  ss << pprint::Indent(indent + 1) << "shutterOpen "
     << std::to_string(camera.shutterOpen) << "\n";
  ss << pprint::Indent(indent + 1) << "shutterClose "
     << std::to_string(camera.shutterClose) << "\n";

  ss << "\n";

  ss << pprint::Indent(indent) << "}\n";

  return ss.str();
}

std::string DumpPreviewSurface(const PreviewSurfaceShader &shader,
                               uint32_t indent) {
  std::stringstream ss;

  ss << "PreviewSurfaceShader {\n";

  ss << pprint::Indent(indent + 1)
     << "useSpecularWorkflow = " << std::to_string(shader.useSpecularWorkflow)
     << "\n";

  ss << pprint::Indent(indent + 1) << "diffuseColor = ";
  if (shader.diffuseColor.is_texture()) {
    ss << "texture_id[" << shader.diffuseColor.texture_id << "]";
  } else {
    ss << shader.diffuseColor.value;
  }
  ss << "\n";

  ss << pprint::Indent(indent + 1) << "metallic = ";
  if (shader.metallic.is_texture()) {
    ss << "texture_id[" << shader.metallic.texture_id << "]";
  } else {
    ss << shader.metallic.value;
  }
  ss << "\n";

  ss << pprint::Indent(indent + 1) << "roughness = ";
  if (shader.roughness.is_texture()) {
    ss << "texture_id[" << shader.roughness.texture_id << "]";
  } else {
    ss << shader.roughness.value;
  }
  ss << "\n";

  ss << pprint::Indent(indent + 1) << "ior = ";
  if (shader.ior.is_texture()) {
    ss << "texture_id[" << shader.ior.texture_id << "]";
  } else {
    ss << shader.ior.value;
  }
  ss << "\n";

  ss << pprint::Indent(indent + 1) << "clearcoat = ";
  if (shader.clearcoat.is_texture()) {
    ss << "texture_id[" << shader.clearcoat.texture_id << "]";
  } else {
    ss << shader.clearcoat.value;
  }
  ss << "\n";

  ss << pprint::Indent(indent + 1) << "clearcoatRoughness = ";
  if (shader.clearcoatRoughness.is_texture()) {
    ss << "texture_id[" << shader.clearcoatRoughness.texture_id << "]";
  } else {
    ss << shader.clearcoatRoughness.value;
  }
  ss << "\n";

  ss << pprint::Indent(indent + 1) << "opacity = ";
  if (shader.opacity.is_texture()) {
    ss << "texture_id[" << shader.opacity.texture_id << "]";
  } else {
    ss << shader.opacity.value;
  }
  ss << "\n";

  ss << pprint::Indent(indent + 1) << "opacityThreshold = ";
  if (shader.opacityThreshold.is_texture()) {
    ss << "texture_id[" << shader.opacityThreshold.texture_id << "]";
  } else {
    ss << shader.opacityThreshold.value;
  }
  ss << "\n";

  ss << pprint::Indent(indent + 1) << "normal = ";
  if (shader.normal.is_texture()) {
    ss << "texture_id[" << shader.normal.texture_id << "]";
  } else {
    ss << shader.normal.value;
  }
  ss << "\n";

  ss << pprint::Indent(indent + 1) << "displacement = ";
  if (shader.displacement.is_texture()) {
    ss << "texture_id[" << shader.displacement.texture_id << "]";
  } else {
    ss << shader.displacement.value;
  }
  ss << "\n";

  ss << pprint::Indent(indent + 1) << "occlusion = ";
  if (shader.occlusion.is_texture()) {
    ss << "texture_id[" << shader.occlusion.texture_id << "]";
  } else {
    ss << shader.occlusion.value;
  }
  ss << "\n";

  ss << pprint::Indent(indent) << "}\n";

  return ss.str();
}

std::string DumpMaterial(const RenderMaterial &material, uint32_t indent) {
  std::stringstream ss;

  ss << pprint::Indent(indent) << "material {\n";

  ss << pprint::Indent(indent + 1) << "name " << quote(material.name) << "\n";
  ss << pprint::Indent(indent + 1) << "abs_path " << quote(material.abs_path)
     << "\n";
  ss << pprint::Indent(indent + 1) << "display_name "
     << quote(material.display_name) << "\n";

  ss << pprint::Indent(indent + 1) << "surfaceShader = ";
  ss << DumpPreviewSurface(material.surfaceShader, indent + 1);
  ss << "\n";

  ss << pprint::Indent(indent) << "}\n";

  return ss.str();
}

std::string DumpUVTexture(const UVTexture &texture, uint32_t indent) {
  std::stringstream ss;

  // TODO
  ss << "UVTexture {\n";
  ss << pprint::Indent(indent + 1) << "primvar_name " << texture.varname_uv
     << "\n";
  ss << pprint::Indent(indent + 1) << "connectedOutputChannel ";
     ss << to_string(texture.connectedOutputChannel) << "\n";

  ss << pprint::Indent(indent + 1) << "authoredOutputChannels ";

  for (const auto &c : texture.authoredOutputChannels) {
     ss << to_string(c) << " ";
  }
  ss << "\n";

  ss << pprint::Indent(indent + 1) << "bias " << texture.bias << "\n";
  ss << pprint::Indent(indent + 1) << "scale " << texture.scale << "\n";
  ss << pprint::Indent(indent + 1) << "wrapS " << to_string(texture.wrapS)
     << "\n";
  ss << pprint::Indent(indent + 1) << "wrapT " << to_string(texture.wrapT)
     << "\n";
  ss << pprint::Indent(indent + 1) << "fallback_uv " << texture.fallback_uv
     << "\n";
  ss << pprint::Indent(indent + 1) << "textureImageID "
     << std::to_string(texture.texture_image_id) << "\n";
  ss << pprint::Indent(indent + 1) << "has UsdTransform2d "
     << std::to_string(texture.has_transform2d) << "\n";
  if (texture.has_transform2d) {
    ss << pprint::Indent(indent + 2) << "rotation " << texture.tx_rotation
       << "\n";
    ss << pprint::Indent(indent + 2) << "scale " << texture.tx_scale << "\n";
    ss << pprint::Indent(indent + 2) << "translation " << texture.tx_translation
       << "\n";
    ss << pprint::Indent(indent + 2) << "computed_transform "
       << texture.transform << "\n";
  }

  ss << "\n";

  ss << pprint::Indent(indent) << "}\n";

  return ss.str();
}

std::string DumpImage(const TextureImage &image, uint32_t indent) {
  std::stringstream ss;

  ss << "TextureImage {\n";
  ss << pprint::Indent(indent + 1) << "asset_identifier \""
     << image.asset_identifier << "\"\n";
  ss << pprint::Indent(indent + 1) << "decoded \""
     << image.decoded << "\"\n";
  ss << pprint::Indent(indent + 1) << "channels "
     << std::to_string(image.channels) << "\n";
  ss << pprint::Indent(indent + 1) << "width " << std::to_string(image.width)
     << "\n";
  ss << pprint::Indent(indent + 1) << "height " << std::to_string(image.height)
     << "\n";
  ss << pprint::Indent(indent + 1) << "miplevel "
     << std::to_string(image.miplevel) << "\n";
  ss << pprint::Indent(indent + 1) << "colorSpace "
     << to_string(image.colorSpace) << "\n";
  ss << pprint::Indent(indent + 1) << "usdColorSpace "
     << to_string(image.usdColorSpace) << "\n";
  ss << pprint::Indent(indent + 1) << "bufferID "
     << std::to_string(image.buffer_id) << "\n";

  ss << "\n";

  ss << pprint::Indent(indent) << "}\n";

  return ss.str();
}

std::string DumpBuffer(const BufferData &buffer, uint32_t indent) {
  std::stringstream ss;

  ss << "Buffer {\n";
  ss << pprint::Indent(indent + 1) << "bytes " << buffer.data.size() << "\n";
  ss << pprint::Indent(indent + 1) << "componentType "
     << to_string(buffer.componentType) << "\n";

  ss << "\n";

  ss << pprint::Indent(indent) << "}\n";

  return ss.str();
}

}  // namespace

std::string DumpRenderScene(const RenderScene &scene,
                            const std::string &format) {
  std::stringstream ss;

  if (format == "json") {
    // TODO:
    // Currently kdl only.
    ss << "// `json` format is not supported yet. Use KDL format\n";
  }

  ss << "title " << quote(scene.usd_filename) << "\n";
  ss << "default_root_node " << scene.default_root_node << "\n";
  ss << "// # of Root Nodes : " << scene.nodes.size() << "\n";
  ss << "// # of Meshes : " << scene.meshes.size() << "\n";
  ss << "// # of Skeletons : " << scene.skeletons.size() << "\n";
  ss << "// # of Animations : " << scene.animations.size() << "\n";
  ss << "// # of Cameras : " << scene.cameras.size() << "\n";
  ss << "// # of Materials : " << scene.materials.size() << "\n";
  ss << "// # of UVTextures : " << scene.textures.size() << "\n";
  ss << "// # of TextureImages : " << scene.images.size() << "\n";
  ss << "// # of Buffers : " << scene.buffers.size() << "\n";

  ss << "\n";

  ss << "nodes {\n";
  for (size_t i = 0; i < scene.nodes.size(); i++) {
    ss << DumpNode(scene.nodes[i], 1);
  }
  ss << "}\n";

  ss << "meshes {\n";
  for (size_t i = 0; i < scene.meshes.size(); i++) {
    ss << "[" << i << "] " << DumpMesh(scene.meshes[i], 1);
  }
  ss << "}\n";

  ss << "skeletons {\n";
  for (size_t i = 0; i < scene.skeletons.size(); i++) {
    ss << "[" << i << "] " << DumpSkeleton(scene.skeletons[i], 1);
  }
  ss << "}\n";

  ss << "animations {\n";
  for (size_t i = 0; i < scene.animations.size(); i++) {
    ss << "[" << i << "] " << DumpAnimation(scene.animations[i], 1);
  }
  ss << "}\n";

  ss << "cameras {\n";
  for (size_t i = 0; i < scene.cameras.size(); i++) {
    ss << "[" << i << "] " << DumpCamera(scene.cameras[i], 1);
  }
  ss << "}\n";

  ss << "\n";
  ss << "materials {\n";
  for (size_t i = 0; i < scene.materials.size(); i++) {
    ss << "[" << i << "] " << DumpMaterial(scene.materials[i], 1);
  }
  ss << "}\n";

  ss << "\n";
  ss << "textures {\n";
  for (size_t i = 0; i < scene.textures.size(); i++) {
    ss << "[" << i << "] " << DumpUVTexture(scene.textures[i], 1);
  }
  ss << "}\n";

  ss << "\n";
  ss << "images {\n";
  for (size_t i = 0; i < scene.images.size(); i++) {
    ss << "[" << i << "] " << DumpImage(scene.images[i], 1);
  }
  ss << "}\n";

  ss << "\n";
  ss << "buffers {\n";
  for (size_t i = 0; i < scene.buffers.size(); i++) {
    ss << "[" << i << "] " << DumpBuffer(scene.buffers[i], 1);
  }
  ss << "}\n";

  // ss << "TODO: AnimationChannel, ...\n";

  return ss.str();
}

}  // namespace tydra
}  // namespace tinyusdz
