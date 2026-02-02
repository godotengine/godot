// SPDX-License-Identifier: Apache 2.0
// Copyright 2022 - 2023, Syoyo Fujita.
// Copyright 2023 - Present, Light Transport Entertainment Inc.

#pragma once

#include <algorithm>
#include <iostream>
#include <sstream>

#include "value-types.hh"

// forward decl
namespace tinyusdz {

// in prim-types.hh
class Path;
struct Reference;
struct Payload;
struct LayerOffset;
struct SubLayer;
class Collection;

}  // namespace tinyusdz

namespace std {

std::ostream &operator<<(std::ostream &os, const tinyusdz::value::char2 &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::char3 &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::char4 &v);

std::ostream &operator<<(std::ostream &os, const tinyusdz::value::uchar2 &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::uchar3 &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::uchar4 &v);

std::ostream &operator<<(std::ostream &os, const tinyusdz::value::short2 &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::short3 &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::short4 &v);

std::ostream &operator<<(std::ostream &os, const tinyusdz::value::ushort2 &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::ushort3 &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::ushort4 &v);

std::ostream &operator<<(std::ostream &os, const tinyusdz::value::int2 &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::int3 &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::int4 &v);

std::ostream &operator<<(std::ostream &os, const tinyusdz::value::uint2 &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::uint3 &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::uint4 &v);

std::ostream &operator<<(std::ostream &os, const tinyusdz::value::half &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::half2 &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::half3 &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::half4 &v);

std::ostream &operator<<(std::ostream &os, const tinyusdz::value::float2 &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::float3 &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::float4 &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::double2 &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::double3 &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::double4 &v);

std::ostream &operator<<(std::ostream &os, const tinyusdz::value::point3h &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::point3f &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::point3d &v);

std::ostream &operator<<(std::ostream &os, const tinyusdz::value::normal3h &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::normal3f &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::normal3d &v);

std::ostream &operator<<(std::ostream &os, const tinyusdz::value::vector3h &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::vector3f &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::vector3d &v);

std::ostream &operator<<(std::ostream &os, const tinyusdz::value::color3h &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::color3f &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::color3d &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::color4h &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::color4f &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::color4d &v);

std::ostream &operator<<(std::ostream &os,
                         const tinyusdz::value::texcoord2h &v);
std::ostream &operator<<(std::ostream &os,
                         const tinyusdz::value::texcoord2f &v);
std::ostream &operator<<(std::ostream &os,
                         const tinyusdz::value::texcoord2d &v);

std::ostream &operator<<(std::ostream &os,
                         const tinyusdz::value::texcoord3h &v);
std::ostream &operator<<(std::ostream &os,
                         const tinyusdz::value::texcoord3f &v);
std::ostream &operator<<(std::ostream &os,
                         const tinyusdz::value::texcoord3d &v);

std::ostream &operator<<(std::ostream &os, const tinyusdz::value::quath &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::quatf &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::quatd &v);

std::ostream &operator<<(std::ostream &os, const tinyusdz::value::token &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::dict &v);
std::ostream &operator<<(std::ostream &os,
                         const tinyusdz::value::TimeSamples &ts);

std::ostream &operator<<(std::ostream &os, const tinyusdz::value::matrix2f &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::matrix3f &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::matrix4f &v);

std::ostream &operator<<(std::ostream &os, const tinyusdz::value::matrix2d &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::matrix3d &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::value::matrix4d &v);

std::ostream &operator<<(std::ostream &os, const tinyusdz::value::frame4d &v);

std::ostream &operator<<(std::ostream &os, const tinyusdz::value::AssetPath &v);

// NOTE: Implemented in pprinter.cc
std::ostream &operator<<(std::ostream &os,
                         const tinyusdz::value::StringData &v);

// NOTE: Implemented in pprinter.cc
std::ostream &operator<<(std::ostream &os, const tinyusdz::Path &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::Reference &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::Payload &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::LayerOffset &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::SubLayer &v);
std::ostream &operator<<(std::ostream &os, const tinyusdz::Collection &v);

// 1D array
template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
  os << "[";
  for (size_t i = 0; i < v.size(); i++) {
    os << v[i];
    if (i != (v.size() - 1)) {
      os << ", ";
    }
  }
  os << "]";
  return os;
}

// Provide specialized version for int and float array.
template <>
std::ostream &operator<<(std::ostream &os, const std::vector<double> &v);

template <>
std::ostream &operator<<(std::ostream &os, const std::vector<float> &v);

template <>
std::ostream &operator<<(std::ostream &os, const std::vector<int32_t> &v);

template <>
std::ostream &operator<<(std::ostream &os, const std::vector<uint32_t> &v);

template <>
std::ostream &operator<<(std::ostream &os, const std::vector<int64_t> &v);

template <>
std::ostream &operator<<(std::ostream &os, const std::vector<uint64_t> &v);

}  // namespace std

namespace tinyusdz {

std::string to_string(bool v);
std::string to_string(int32_t v);
std::string to_string(uint32_t v);
std::string to_string(int64_t v);
std::string to_string(uint64_t v);

std::string to_string(const value::char2 &v);
std::string to_string(const value::char3 &v);
std::string to_string(const value::char4 &v);
std::string to_string(const value::short2 &v);
std::string to_string(const value::short3 &v);
std::string to_string(const value::short4 &v);
std::string to_string(const value::int2 &v);
std::string to_string(const value::int3 &v);
std::string to_string(const value::int4 &v);
std::string to_string(const value::uint2 &v);
std::string to_string(const value::uint3 &v);
std::string to_string(const value::uint4 &v);
std::string to_string(const value::float2 &v);
std::string to_string(const value::float3 &v);
std::string to_string(const value::float4 &v);
std::string to_string(const value::double2 &v);
std::string to_string(const value::double3 &v);
std::string to_string(const value::double4 &v);
std::string to_string(const value::texcoord2h &v);
std::string to_string(const value::texcoord2f &v);
std::string to_string(const value::texcoord2d &v);
std::string to_string(const value::texcoord3h &v);
std::string to_string(const value::texcoord3f &v);
std::string to_string(const value::texcoord3d &v);
std::string to_string(const value::StringData &s);
std::string to_string(const value::token &s);
std::string to_string(const std::string &s); // do USD specific escaping
std::string to_string(const value::quath &v);
std::string to_string(const value::quatf &v);
std::string to_string(const value::quatd &v);
std::string to_string(const value::matrix2f &v);
std::string to_string(const value::matrix3f &v);
std::string to_string(const value::matrix4f &v);
std::string to_string(const value::matrix2d &v);
std::string to_string(const value::matrix3d &v);
std::string to_string(const value::matrix4d &v);
std::string to_string(const value::frame4d &v);
std::string to_string(const value::half &v);
std::string to_string(const value::half2 &v);
std::string to_string(const value::half3 &v);
std::string to_string(const value::half4 &v);
std::string to_string(const value::normal3h &v);
std::string to_string(const value::normal3f &v);
std::string to_string(const value::normal3d &v);
std::string to_string(const value::vector3h &v);
std::string to_string(const value::vector3f &v);
std::string to_string(const value::vector3d &v);
std::string to_string(const value::point3h &v);
std::string to_string(const value::point3f &v);
std::string to_string(const value::point3d &v);
std::string to_string(const value::color3f &v);
std::string to_string(const value::color3d &v);
std::string to_string(const value::color4h &v);
std::string to_string(const value::color4f &v);
std::string to_string(const value::color4d &v);

namespace value {

std::string pprint_value(const tinyusdz::value::Value &v,
                         const uint32_t indent = 0, bool closing_brace = true);

// Print first N and last N items.
// 0 = print all items.
// Callee must ensure access to `vals` does not trigger out-of-bounds error.
template <typename T>
std::string print_array_snipped(const T *vals, size_t n, size_t N = 16) {
  std::stringstream os;

  if ((N == 0) || ((N * 2) >= n)) {
    os << "[";
    for (size_t i = 0; i < n; i++) {
      if (i > 0) {
        os << ", ";
      }
      os << vals[i];
    }
    os << "]";
  } else {
    size_t head_end = (std::min)(N, n);
    size_t tail_start = (std::max)(n - N, head_end);

    os << "[";

    for (size_t i = 0; i < head_end; i++) {
      if (i > 0) {
        os << ", ";
      }
      os << vals[i];
    }

    os << ", ..., ";

    for (size_t i = tail_start; i < n; i++) {
      if (i > tail_start) {
        os << ", ";
      }
      os << vals[i];
    }

    os << "]";
  }
  return os.str();
}

// Account for stride.
// stride 0 => use sizeof(T)
// Callee must ensure access to `vals` does not trigger out-of-bounds error.
template <typename T>
std::string print_strided_array_snipped(const uint8_t *vals, size_t stride_bytes, const size_t n, size_t N = 16) {
  std::stringstream os;

  if ((stride_bytes == 0) || (stride_bytes == sizeof(T))) { // tightly packed.
    return print_array_snipped(reinterpret_cast<const T*>(vals), n, N);
  }

  if ((N == 0) || ((N * 2) >= n)) {
    os << "[";
    for (size_t i = 0; i < n; i++) {
      if (i > 0) {
        os << ", ";
      }
      os << *reinterpret_cast<const T *>(&vals[i * stride_bytes]);
    }
    os << "]";
  } else {
    size_t head_end = (std::min)(N, n);
    size_t tail_start = (std::max)(n - N, head_end);

    os << "[";

    for (size_t i = 0; i < head_end; i++) {
      if (i > 0) {
        os << ", ";
      }
      os << *reinterpret_cast<const T *>(&vals[i * stride_bytes]);
    }

    os << ", ..., ";

    for (size_t i = tail_start; i < n; i++) {
      if (i > tail_start) {
        os << ", ";
      }
      os << *reinterpret_cast<const T *>(&vals[i * stride_bytes]);
    }

    os << "]";
  }
  return os.str();
}

// Print first N and last N items.
// 0 = print all items.
// Useful when dump
template <typename T>
std::string print_array_snipped(const std::vector<T> &vals, size_t N = 16) {
  std::stringstream os;

  if ((N == 0) || ((N * 2) >= vals.size())) {
    os << "[";
    for (size_t i = 0; i < vals.size(); i++) {
      if (i > 0) {
        os << ", ";
      }
      os << vals[i];
    }
    os << "]";
  } else {
    size_t head_end = (std::min)(N, vals.size());
    size_t tail_start = (std::max)(vals.size() - N, head_end);

    os << "[";

    for (size_t i = 0; i < head_end; i++) {
      if (i > 0) {
        os << ", ";
      }
      os << vals[i];
    }

    os << ", ..., ";

    for (size_t i = tail_start; i < vals.size(); i++) {
      if (i > tail_start) {
        os << ", ";
      }
      os << vals[i];
    }

    os << "]";
  }
  return os.str();
}

// TODO: Remove
// std::string pprint_any(const linb::any &v, const uint32_t indent = 0, bool
// closing_brace = true);

}  // namespace value
}  // namespace tinyusdz
