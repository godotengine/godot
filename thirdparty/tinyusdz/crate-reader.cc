// SPDX-License-Identifier: Apache 2.0
// Copyright 2022-2022 Syoyo Fujita.
// Copyright 2023-Present Light Transport Entertainment Inc.
//
// Crate(binary format) reader
//
//
// TODO:
// - [] Unify BuildDecompressedPathsImpl and BuildNodeHierarchy

#ifdef _MSC_VER
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "crate-reader.hh"

#ifdef __wasi__
#else
#include <thread>
#endif

#include <unordered_set>
#include <stack>

#include "crate-format.hh"
#include "crate-pprint.hh"
#include "integerCoding.h"
#include "lz4-compression.hh"
#include "path-util.hh"
#include "pprinter.hh"
#include "prim-types.hh"
#include "stream-reader.hh"
#include "tinyusdz.hh"
#include "value-pprint.hh"
#include "value-types.hh"
#include "tiny-format.hh"
#include "str-util.hh"

//
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

#include "nonstd/expected.hpp"

#ifdef __clang__
#pragma clang diagnostic pop
#endif

//

#include "common-macros.inc"

namespace tinyusdz {
namespace crate {

//constexpr auto kTypeName = "typeName";
//constexpr auto kToken = "Token";
//constexpr auto kDefault = "default";

#define kTag "[Crate]"

#define CHECK_MEMORY_USAGE(__nbytes) do { \
  _memoryUsage += (__nbytes); \
  if (_memoryUsage > _config.maxMemoryBudget) { \
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Reached to max memory budget."); \
  }  \
  } while(0)

#define REDUCE_MEMORY_USAGE(__nbytes) do { \
  if (_memoryUsage < (__nbytes)) { \
    _memoryUsage -= (__nbytes); \
  } \
  } while(0)



#define VERSION_LESS_THAN_0_8_0(__version) ((_version[0] == 0) && (_version[1] < 7))

//
// --
//
CrateReader::CrateReader(StreamReader *sr, const CrateReaderConfig &config) : _sr(sr), _impl(nullptr) {
  _config = config;
  if (_config.numThreads == -1) {
#if defined(__wasi__)
#else
    _config.numThreads = (std::max)(1, int(std::thread::hardware_concurrency()));
    PUSH_WARN("# of thread to use: " << std::to_string(_config.numThreads));
#endif
  }


#if defined(__wasi__)
  PUSH_WARN("Threading is disabled for WASI build.");
  _config.numThreads = 1;
#else

  // Limit to 1024 threads.
  _config.numThreads = (std::min)(1024, _config.numThreads);
#endif

  //_impl = new Impl();

}

CrateReader::~CrateReader() {
  //delete _impl;
  //_impl = nullptr;
}

std::string CrateReader::GetError() { return _err; }

std::string CrateReader::GetWarning() { return _warn; }

bool CrateReader::HasField(const std::string &key) const {
  // Simple linear search
  for (const auto &field : _fields) {
    if (auto fv = GetToken(field.token_index)) {
      if (fv.value().str().compare(key) == 0) {
        return true;
      }
    }
  }
  return false;
}

nonstd::optional<crate::Field> CrateReader::GetField(crate::Index index) const {

  if (index.value < _fields.size()) {
    return _fields[index.value];
  } else {
    return nonstd::nullopt;
  }
}

const nonstd::optional<value::token> CrateReader::GetToken(
    crate::Index token_index) const {
  if (token_index.value < _tokens.size()) {
    return _tokens[token_index.value];
  } else {
    return nonstd::nullopt;
  }
}

// Get string token from string index.
const nonstd::optional<value::token> CrateReader::GetStringToken(
    crate::Index string_index) const {

  if (string_index.value < _string_indices.size()) {
    crate::Index s_idx = _string_indices[string_index.value];
    return GetToken(s_idx);
  } else {
    PUSH_ERROR("String index out of range: " +
               std::to_string(string_index.value));
    return value::token();
  }
}

nonstd::optional<Path> CrateReader::GetPath(crate::Index index) const {

  if (index.value < _paths.size()) {
    // ok
  } else {
    return nonstd::nullopt;
  }

  return _paths[index.value];
}

nonstd::optional<Path> CrateReader::GetElementPath(crate::Index index) const {
  if (index.value < _elemPaths.size()) {
    // ok
  } else {
    return nonstd::nullopt;
  }

  return _elemPaths[index.value];
}

nonstd::optional<std::string> CrateReader::GetPathString(
    crate::Index index) const {
  if (index.value < _paths.size()) {
    // ok
  } else {
    return nonstd::nullopt;
  }

  const Path &p = _paths[index.value];

  return p.full_path_name();
}

bool CrateReader::ReadIndex(crate::Index *i) {
  // string is serialized as StringIndex
  uint32_t value;
  if (!_sr->read4(&value)) {
    PUSH_ERROR("Failed to read Index");
    return false;
  }

  CHECK_MEMORY_USAGE(sizeof(uint32_t));

  (*i) = crate::Index(value);
  return true;
}

bool CrateReader::ReadIndices(std::vector<crate::Index> *indices) {
  uint64_t n;
  if (!_sr->read8(&n)) {
    return false;
  }

  if (n > _config.maxNumIndices) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Too many indices.");
  }

  if (n == 0) {
    return true;
  }

  DCOUT("ReadIndices: n = " << n);

  size_t datalen = size_t(n) * sizeof(crate::Index);

  if (datalen > _sr->size()) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Indices data exceeds USDC size.");
  }

  CHECK_MEMORY_USAGE(datalen);

  indices->resize(size_t(n));

  if (datalen != _sr->read(datalen, datalen,
                          reinterpret_cast<uint8_t *>(indices->data()))) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read Indices array.");
  }

  return true;
}

bool CrateReader::ReadString(std::string *s) {
  // string is serialized as StringIndex
  crate::Index string_index;
  if (!ReadIndex(&string_index)) {
    PUSH_ERROR("Failed to read Index for string data.");
    return false;
  }

  if (auto tok = GetStringToken(string_index)) {
    (*s) = tok.value().str();
    CHECK_MEMORY_USAGE(s->size());
    return true;
  }


  PUSH_ERROR("Invalid StringIndex.");
  return false;
}

nonstd::optional<std::string> CrateReader::GetSpecString(
    crate::Index index) const {
  if (index.value < _specs.size()) {
    // ok
  } else {
    return nonstd::nullopt;
  }

  const crate::Spec &spec = _specs[index.value];

  if (auto pathv = GetPathString(spec.path_index)) {
    std::string path_str = pathv.value();
    std::string specty_str = to_string(spec.spec_type);

    return "[Spec] path: " + path_str +
           ", fieldset id: " + std::to_string(spec.fieldset_index.value) +
           ", spec_type: " + specty_str;
  }

  return nonstd::nullopt;
}

bool CrateReader::ReadValueRep(crate::ValueRep *rep) {
  if (!_sr->read8(reinterpret_cast<uint64_t *>(rep))) {
    PUSH_ERROR("Failed to read ValueRep.");
    return false;
  }

  CHECK_MEMORY_USAGE(sizeof(uint64_t));

  DCOUT("ValueRep value = " << rep->GetData());

  return true;
}

template <class Int>
bool CrateReader::ReadCompressedInts(Int *out,
                                     size_t num_ints) {
  if (num_ints > _config.maxInts) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("# of ints {} too large. maxInts is set to {}", num_ints, _config.maxInts));
  }

  using Compressor =
      typename std::conditional<sizeof(Int) == 4, Usd_IntegerCompression,
                                Usd_IntegerCompression64>::type;


  // TODO: Read compressed data from _sr directly
  size_t compBufferSize = Compressor::GetCompressedBufferSize(num_ints);
  CHECK_MEMORY_USAGE(compBufferSize);

  uint64_t compSize;
  if (!_sr->read8(&compSize)) {
    return false;
  }

  if (compSize > compBufferSize) {
    // Truncate
    // TODO: return error?
    compSize = compBufferSize;
  }

  if (compSize > _sr->size()) {
    return false;
  }

  if (compSize < 4) {
    // Too small
    return false;
  }

  std::vector<char> compBuffer;
  compBuffer.resize(compBufferSize);
  if (!_sr->read(size_t(compSize), size_t(compSize),
                reinterpret_cast<uint8_t *>(compBuffer.data()))) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read compressedInts.");
  }

  bool ret = Compressor::DecompressFromBuffer(
      compBuffer.data(), size_t(compSize), out, num_ints, &_err);

  REDUCE_MEMORY_USAGE(compBufferSize);

  return ret;
}

template <typename T>
bool CrateReader::ReadIntArray(bool is_compressed, std::vector<T> *d) {

  size_t length{0}; // uncompressed array elements.
  // < ver 0.7.0  use 32bit
  if (VERSION_LESS_THAN_0_8_0(_version)) {
      uint32_t shapesize; // not used
      if (!_sr->read4(&shapesize)) {
        PUSH_ERROR("Failed to read the number of array elements.");
        return false;
      }
    uint32_t n;
    if (!_sr->read4(&n)) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read the number of array elements.");
    }
    length = size_t(n);
  } else {
    uint64_t n;
    if (!_sr->read8(&n)) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read the number of array elements.");
      return false;
    }

    DCOUT("array.len = " << n);
    length = size_t(n);
  }

  DCOUT("array.len = " << length);
  if (length == 0) {
    d->clear();
    return true;
  }

  if (length > _config.maxArrayElements) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Too large array elements.");
  }

  CHECK_MEMORY_USAGE(sizeof(T) * length);

  d->resize(length);

  if (!is_compressed) {

    // TODO(syoyo): Zero-copy
    if (!_sr->read(sizeof(T) * length, sizeof(T) * length,
                   reinterpret_cast<uint8_t *>(d->data()))) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read integer array data.");
    }

    return true;

  } else {

    if (length < crate::kMinCompressedArraySize) {
      size_t sz = sizeof(T) * length;
      // Not stored in compressed for smaller data
      if (!_sr->read(sz, sz, reinterpret_cast<uint8_t *>(d->data()))) {
        PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read uncompressed integer array data.");
      }
      return true;
    }

    return ReadCompressedInts(d->data(), d->size());
  }
}

bool CrateReader::ReadHalfArray(bool is_compressed,
                                std::vector<value::half> *d) {
  size_t length;
  // < ver 0.7.0  use 32bit
  if (VERSION_LESS_THAN_0_8_0(_version)) {
      uint32_t shapesize; // not used
      if (!_sr->read4(&shapesize)) {
        PUSH_ERROR("Failed to read the number of array elements.");
        return false;
      }
    uint32_t n;
    if (!_sr->read4(&n)) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read the number of array elements.");
    }
    length = size_t(n);
  } else {
    uint64_t n;
    if (!_sr->read8(&n)) {
      _err += "Failed to read the number of array elements.\n";
      return false;
    }

    length = size_t(n);
  }

  if (length > _config.maxArrayElements) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Too many array elements {}.", length));
  }

  CHECK_MEMORY_USAGE(length * sizeof(uint16_t));

  d->resize(length);

  if (!is_compressed) {


    // TODO(syoyo): Zero-copy
    if (!_sr->read(sizeof(uint16_t) * length, sizeof(uint16_t) * length,
                   reinterpret_cast<uint8_t *>(d->data()))) {
      _err += "Failed to read half array data.\n";
      return false;
    }

    return true;
  } else {

    //
    // compressed data is represented by integers or look-up table.
    //

    if (length < crate::kMinCompressedArraySize) {
      size_t sz = sizeof(uint16_t) * length;
      // Not stored in compressed.
      // reader.ReadContiguous(odata, osize);
      if (!_sr->read(sz, sz, reinterpret_cast<uint8_t *>(d->data()))) {
        _err += "Failed to read uncompressed array data.\n";
        return false;
      }
      return true;
    }

    // Read the code
    char code;
    if (!_sr->read1(&code)) {
      _err += "Failed to read the code.\n";
      return false;
    }

    if (code == 'i') {
      // Compressed integers.
      std::vector<int32_t> ints;
      ints.resize(length);
      if (!ReadCompressedInts(ints.data(), ints.size())) {
        _err += "Failed to read compressed ints in ReadHalfArray.\n";
        return false;
      }
      for (size_t i = 0; i < length; i++) {
        float f = float(ints[i]);
        value::half h = value::float_to_half_full(f);
        (*d)[i] = h;
      }
    } else if (code == 't') {
      // Lookup table & indexes.
      uint32_t lutSize;
      if (!_sr->read4(&lutSize)) {
        _err += "Failed to read lutSize in ReadHalfArray.\n";
        return false;
      }

      std::vector<value::half> lut;
      lut.resize(lutSize);
      if (!_sr->read(sizeof(value::half) * lutSize, sizeof(value::half) * lutSize,
                     reinterpret_cast<uint8_t *>(lut.data()))) {
        _err += "Failed to read lut table in ReadHalfArray.\n";
        return false;
      }

      std::vector<uint32_t> indexes;
      indexes.resize(length);
      if (!ReadCompressedInts(indexes.data(), indexes.size())) {
        _err += "Failed to read lut indices in ReadHalfArray.\n";
        return false;
      }

      auto o = d->data();
      for (auto index : indexes) {
        *o++ = lut[index];
      }
    } else {
      _err += "Invalid code. Data is currupted\n";
      return false;
    }

    return true;
  }

}

bool CrateReader::ReadFloatArray(bool is_compressed, std::vector<float> *d) {

  size_t length;
  // < ver 0.7.0  use 32bit
  if (VERSION_LESS_THAN_0_8_0(_version)) {
      uint32_t shapesize; // not used
      if (!_sr->read4(&shapesize)) {
        PUSH_ERROR("Failed to read the number of array elements.");
        return false;
      }
    uint32_t n;
    if (!_sr->read4(&n)) {
      _err += "Failed to read the number of array elements.\n";
      return false;
    }
    length = size_t(n);
  } else {
    uint64_t n;
    if (!_sr->read8(&n)) {
      _err += "Failed to read the number of array elements.\n";
      return false;
    }

    length = size_t(n);
  }

  if (length > _config.maxArrayElements) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Too many array elements.");
  }

  CHECK_MEMORY_USAGE(length * sizeof(float));

  d->resize(length);

  if (!is_compressed) {

    // TODO(syoyo): Zero-copy
    if (!_sr->read(sizeof(float) * length, sizeof(float) * length,
                   reinterpret_cast<uint8_t *>(d->data()))) {
      _err += "Failed to read float array data.\n";
      return false;
    }

    return true;
  } else {

    //
    // compressed data is represented by integers or look-up table.
    //

    if (length < crate::kMinCompressedArraySize) {
      size_t sz = sizeof(float) * length;
      // Not stored in compressed.
      // reader.ReadContiguous(odata, osize);
      if (!_sr->read(sz, sz, reinterpret_cast<uint8_t *>(d->data()))) {
        _err += "Failed to read uncompressed array data.\n";
        return false;
      }
      return true;
    }

    // Read the code
    char code;
    if (!_sr->read1(&code)) {
      _err += "Failed to read the code.\n";
      return false;
    }

    if (code == 'i') {
      // Compressed integers.
      std::vector<int32_t> ints;
      ints.resize(length);
      if (!ReadCompressedInts(ints.data(), ints.size())) {
        _err += "Failed to read compressed ints in ReadFloatArray.\n";
        return false;
      }
      for (size_t i = 0; i < length; i++) {
        d->data()[i] = float(ints[i]);
      }
    } else if (code == 't') {
      // Lookup table & indexes.
      uint32_t lutSize;
      if (!_sr->read4(&lutSize)) {
        _err += "Failed to read lutSize in ReadFloatArray.\n";
        return false;
      }

      std::vector<float> lut;
      lut.resize(lutSize);
      if (!_sr->read(sizeof(float) * lutSize, sizeof(float) * lutSize,
                     reinterpret_cast<uint8_t *>(lut.data()))) {
        _err += "Failed to read lut table in ReadFloatArray.\n";
        return false;
      }

      std::vector<uint32_t> indexes;
      indexes.resize(length);
      if (!ReadCompressedInts(indexes.data(), indexes.size())) {
        _err += "Failed to read lut indices in ReadFloatArray.\n";
        return false;
      }

      auto o = d->data();
      for (auto index : indexes) {
        *o++ = lut[index];
      }
    } else {
      _err += "Invalid code. Data is currupted\n";
      return false;
    }

    return true;
  }

}

bool CrateReader::ReadDoubleArray(bool is_compressed, std::vector<double> *d) {

  size_t length;
  // < ver 0.7.0  use 32bit
  if (VERSION_LESS_THAN_0_8_0(_version)) {
      uint32_t shapesize; // not used
      if (!_sr->read4(&shapesize)) {
        PUSH_ERROR("Failed to read the number of array elements.");
        return false;
      }
    uint32_t n;
    if (!_sr->read4(&n)) {
      _err += "Failed to read the number of array elements.\n";
      return false;
    }
    length = size_t(n);
  } else {
    uint64_t n;
    if (!_sr->read8(&n)) {
      _err += "Failed to read the number of array elements.\n";
      return false;
    }

    length = size_t(n);
  }

  if (length == 0) {
    d->clear();
    return true;
  }

  if (length > _config.maxArrayElements) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Too many array elements.");
  }

  CHECK_MEMORY_USAGE(length * sizeof(double));

  d->resize(length);

  if (!is_compressed) {

    // TODO(syoyo): Zero-copy
    if (!_sr->read(sizeof(double) * length, sizeof(double) * length,
                   reinterpret_cast<uint8_t *>(d->data()))) {
      _err += "Failed to read double array data.\n";
      return false;
    }

    return true;
  } else {

    //
    // compressed data is represented by integers or look-up table.
    //

    d->resize(length);

    if (length < crate::kMinCompressedArraySize) {
      size_t sz = sizeof(double) * length;
      // Not stored in compressed.
      // reader.ReadContiguous(odata, osize);
      if (!_sr->read(sz, sz, reinterpret_cast<uint8_t *>(d->data()))) {
        _err += "Failed to read uncompressed array data.\n";
        return false;
      }
      return true;
    }

    // Read the code
    char code;
    if (!_sr->read1(&code)) {
      _err += "Failed to read the code.\n";
      return false;
    }

    if (code == 'i') {
      // Compressed integers.
      std::vector<int32_t> ints;
      ints.resize(length);
      if (!ReadCompressedInts(ints.data(), ints.size())) {
        _err += "Failed to read compressed ints in ReadDoubleArray.\n";
        return false;
      }
      std::copy(ints.begin(), ints.end(), d->data());
    } else if (code == 't') {
      // Lookup table & indexes.
      uint32_t lutSize;
      if (!_sr->read4(&lutSize)) {
        _err += "Failed to read lutSize in ReadDoubleArray.\n";
        return false;
      }

      std::vector<double> lut;
      lut.resize(lutSize);
      if (!_sr->read(sizeof(double) * lutSize, sizeof(double) * lutSize,
                     reinterpret_cast<uint8_t *>(lut.data()))) {
        _err += "Failed to read lut table in ReadDoubleArray.\n";
        return false;
      }

      std::vector<uint32_t> indexes;
      indexes.resize(length);
      if (!ReadCompressedInts(indexes.data(), indexes.size())) {
        _err += "Failed to read lut indices in ReadDoubleArray.\n";
        return false;
      }

      auto o = d->data();
      for (auto index : indexes) {
        *o++ = lut[index];
      }
    } else {
      _err += "Invalid code. Data is currupted\n";
      return false;
    }

    return true;
  }
}

bool CrateReader::ReadDoubleVector(std::vector<double> *d) {
  size_t length;

  uint64_t n;
  if (!_sr->read8(&n)) {
    _err += "Failed to read the number of array elements.\n";
    return false;
  }

  length = size_t(n);

  if (length == 0) {
    d->clear();
    return true;
  }

  if (length > _config.maxArrayElements) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Too many array elements.");
  }

  CHECK_MEMORY_USAGE(length * sizeof(double));

  d->resize(length);

  // TODO(syoyo): Zero-copy
  if (!_sr->read(sizeof(double) * length, sizeof(double) * length,
                 reinterpret_cast<uint8_t *>(d->data()))) {
    _err += "Failed to read double vector data.\n";
    return false;
  }

  return true;
}

bool CrateReader::ReadTimeSamples(value::TimeSamples *d) {

  // Layout
  //
  // - `times`(double[])
  // - NumValueReps(int64)
  // - ArrayOfValueRep
  //

  // TODO(syoyo): Deferred loading of TimeSamples?(See USD's implementation for details)

  DCOUT("ReadTimeSamples: offt before tell = " << _sr->tell());

  // 8byte for the offset for recursive value. See RecursiveRead() in
  // https://github.com/PixarAnimationStudios/USD/blob/release/pxr/usd/usd/crateFile.cpp for details.
  int64_t offset{0};
  if (!_sr->read8(&offset)) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read the offset for value in Dictionary.");
    return false;
  }

  DCOUT("TimeSample times value offset = " << offset);
  DCOUT("TimeSample tell = " << _sr->tell());

  // -8 to compensate sizeof(offset)
  if (!_sr->seek_from_current(offset - 8)) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to seek to TimeSample times. Invalid offset value: " +
            std::to_string(offset));
  }

  // TODO(syoyo): Deduplicate times?

  crate::ValueRep times_rep{0};
  if (!ReadValueRep(&times_rep)) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read ValueRep for TimeSample' `times` element.");
  }

  // Save offset
  auto values_offset = _sr->tell();

  // TODO: Enable Check if  type `double[]`
#if 0
  if (times_rep.GetType() == crate::CrateDataTypeId::CRATE_DATA_TYPE_DOUBLE_VECTOR) {
    // ok
  } else if ((times_rep.GetType() == crate::CrateDataTypeId::CRATE_DATA_TYPE_DOUBOLE) && times_rep.IsArray()) {
    // ok
  } else {
    PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("`times` value must be type `double[]`, but got type `{}`", times_rep.GetTypeName()));
  }
#endif

  crate::CrateValue times_value;
  if (!UnpackValueRep(times_rep, &times_value)) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to unpack value of TimeSample's `times` element.");
  }

  // must be an array of double.
  DCOUT("TimeSample times:" << times_value.type_name());

  std::vector<double> times;
  if (auto pv = times_value.get_value<std::vector<double>>()) {
    times = pv.value();
    DCOUT("`times` = " << times);
  } else {
    PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("`times` in TimeSamples must be type `double[]`, but got type `{}`", times_value.type_name()));
  }

  //
  // Parse values(elements) of TimeSamples.
  //

  // seek position will be changed in `_UnpackValueRep`, so revert it.
  if (!_sr->seek_set(values_offset)) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to seek to TimeSamples values.");
  }

  // 8byte for the offset for recursive value. See RecursiveRead() in
  // crateFile.cpp for details.
  if (!_sr->read8(&offset)) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read the offset for value in TimeSamples.");
    return false;
  }

  DCOUT("TimeSample value offset = " << offset);
  DCOUT("TimeSample tell = " << _sr->tell());

  // -8 to compensate sizeof(offset)
  if (!_sr->seek_from_current(offset - 8)) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to seek to TimeSample values. Invalid offset value: " + std::to_string(offset));
  }

  uint64_t num_values{0};
  if (!_sr->read8(&num_values)) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read the number of values from TimeSamples.");
    return false;
  }

  DCOUT("Number of values = " << num_values);

  if (times.size() != num_values) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "# of `times` elements and # of values in Crate differs.");
  }

  for (size_t i = 0; i < num_values; i++) {

    crate::ValueRep rep;
    if (!ReadValueRep(&rep)) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read ValueRep for TimeSample' value element.");
    }

    auto next_vrep_loc = _sr->tell();

    ///
    /// Type check of the content of `value` will be done at ReconstructPrim() in usdc-reader.cc.
    ///
    crate::CrateValue value;
    if (!UnpackValueRep(rep, &value)) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to unpack value of TimeSample's value element.");
    }

    d->add_sample(times[i], value.get_raw());

    // UnpackValueRep() will change StreamReader's read position.
    // Revert to next ValueRep location here.
    _sr->seek_set(next_vrep_loc);
  }

  // Move to next location.
  // sizeof(uint64) = sizeof(ValueRep)
  _sr->seek_set(values_offset);
  if (!_sr->seek_from_current(int64_t(sizeof(uint64_t) * num_values))) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to seek over TimeSamples's values.");
  }


  return true;
}

bool CrateReader::ReadStringArray(std::vector<std::string> *d) {
  // array data is not compressed
  auto ReadFn = [this](std::vector<std::string> &result) -> bool {
    uint64_t n{0};
    if (!_sr->read8(&n)) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read the number of array elements.");
      return false;
    }

    if (n > _config.maxArrayElements) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "Too many array elements.");
    }

    CHECK_MEMORY_USAGE(size_t(n) * sizeof(crate::Index));

    std::vector<crate::Index> ivalue(static_cast<size_t>(n));

    if (!_sr->read(size_t(n) * sizeof(crate::Index),
                   size_t(n) * sizeof(crate::Index),
                   reinterpret_cast<uint8_t *>(ivalue.data()))) {
      PUSH_ERROR("Failed to read STRING_VECTOR data.");
      return false;
    }

    // reconstruct
    CHECK_MEMORY_USAGE(size_t(n) * sizeof(void *));
    result.resize(static_cast<size_t>(n));
    for (size_t i = 0; i < n; i++) {
      if (auto v = GetStringToken(ivalue[i])) {
        std::string s = v.value().str();
        CHECK_MEMORY_USAGE(s.size());
        result[i] = s;
      } else {
        PUSH_ERROR("Invalid StringIndex.");
      }
    }

    return true;
  };

  std::vector<std::string> items;
  if (!ReadFn(items)) {
    return false;
  }

  (*d) = items;

  return true;
}

bool CrateReader::ReadReference(Reference *d) {

  if (!d) {
    return false;
  }

  // assetPath : string
  // primPath : Path
  // layerOffset : LayerOffset
  // customData : Dict

  std::string assetPath;
  if (!ReadString(&assetPath)) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read assetPath in Reference ValueRep.");
  }

  crate::PathIndex index;
  if (!ReadIndex(&index)) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read primPath Index in Reference ValueRep.");
  }

  auto path = GetPath(index);
  if (!path) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Invalid Path index in Reference ValueRep.");
  }

  LayerOffset layerOffset;
  if (!ReadLayerOffset(&layerOffset)) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read LayerOffset in Reference ValueRep.");
  }

  CustomDataType customData;
  if (!ReadCustomData(&customData)) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read CustomData(Dict) in Reference ValueRep.");
  }

  d->asset_path = assetPath;
  d->prim_path = path.value();
  d->layerOffset = layerOffset;
  d->customData = customData;

  return true;
}

bool CrateReader::ReadPayload(Payload *d) {

  if (!d) {
    return false;
  }

  // assetPath : string
  // primPath : Path

  std::string assetPath;
  if (!ReadString(&assetPath)) {
    return false;
  }


  crate::PathIndex index;
  if (!ReadIndex(&index)) {
    return false;
  }

  auto path = GetPath(index);
  if (!path) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Invalid Path index in Payload ValueRep.");
  }

  // LayerOffset from 0.8.0
  if (VersionGreaterThanOrEqualTo_0_8_0()) {
    LayerOffset layerOffset;
    if (!ReadLayerOffset(&layerOffset)) {
      return false;
    }
    d->layerOffset = layerOffset;
  }

  d->asset_path = assetPath;
  d->prim_path = path.value();

  return true;
}

bool CrateReader::ReadLayerOffset(LayerOffset *d) {
  static_assert(sizeof(LayerOffset) == 8 * 2, "LayerOffset must be 16bytes");

  // double x 2
  if (!_sr->read(sizeof(double), sizeof(double), reinterpret_cast<uint8_t *>(&(d->_offset)))) {
    return false;
  }
  if (!_sr->read(sizeof(double), sizeof(double), reinterpret_cast<uint8_t *>(&(d->_scale)))) {
    return false;
  }

  return true;
}

bool CrateReader::ReadLayerOffsetArray(std::vector<LayerOffset> *d) {
  // array data is not compressed

  uint64_t n{0};
  if (!_sr->read8(&n)) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read the number of array elements.");
    return false;
  }

  if (n > _config.maxArrayElements) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Too many array elements.");
  }

  if (n == 0) {
    return true;
  }

  CHECK_MEMORY_USAGE(size_t(n) * sizeof(LayerOffset));

  d->resize(size_t(n));

  if (!_sr->read(size_t(n) * sizeof(LayerOffset),
                 size_t(n) * sizeof(LayerOffset),
                 reinterpret_cast<uint8_t *>(d->data()))) {
    PUSH_ERROR("Failed to read LayerOffset[] data.");
    return false;
  }

  return true;
}

bool CrateReader::ReadPathArray(std::vector<Path> *d) {
  // array data is not compressed
  auto ReadFn = [this](std::vector<Path> &result) -> bool {
    uint64_t n{0};
    if (!_sr->read8(&n)) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read the number of array elements.");
      return false;
    }

    if (n > _config.maxArrayElements) {
      _err += "Too many Path array elements.\n";
      return false;
    }

    CHECK_MEMORY_USAGE(size_t(n) * sizeof(crate::Index));

    std::vector<crate::Index> ivalue(static_cast<size_t>(n));

    if (!_sr->read(size_t(n) * sizeof(crate::Index),
                   size_t(n) * sizeof(crate::Index),
                   reinterpret_cast<uint8_t *>(ivalue.data()))) {
      _err += "Failed to read ListOp data.\n";
      return false;
    }

    // reconstruct
    result.resize(static_cast<size_t>(n));
    for (size_t i = 0; i < n; i++) {
      if (auto pv = GetPath(ivalue[i])) {
        result[i] = pv.value();
      } else {
        PUSH_ERROR("Invalid Index for Path.");
        return false;
      }
    }

    return true;
  };

  std::vector<Path> items;
  if (!ReadFn(items)) {
    _err += "Failed to read Path vector.\n";
    return false;
  }

  (*d) = items;

  return true;
}

bool CrateReader::ReadTokenListOp(ListOp<value::token> *d) {
  // read ListOpHeader
  ListOpHeader h;
  if (!_sr->read1(&h.bits)) {
    _err += "Failed to read ListOpHeader\n";
    return false;
  }

  if (h.IsExplicit()) {
    d->ClearAndMakeExplicit();
  }

  // array data is not compressed
  auto ReadFn = [this](std::vector<value::token> &result) -> bool {
    uint64_t n;
    if (!_sr->read8(&n)) {
      _err += "Failed to read # of elements in ListOp.\n";
      return false;
    }

    if (n > _config.maxArrayElements) {
      _err += "Too many ListOp elements.\n";
      return false;
    }

    CHECK_MEMORY_USAGE(size_t(n) * sizeof(crate::Index));

    std::vector<crate::Index> ivalue(static_cast<size_t>(n));

    if (!_sr->read(size_t(n) * sizeof(crate::Index),
                   size_t(n) * sizeof(crate::Index),
                   reinterpret_cast<uint8_t *>(ivalue.data()))) {
      _err += "Failed to read ListOp data.\n";
      return false;
    }

    // reconstruct
    result.resize(static_cast<size_t>(n));
    for (size_t i = 0; i < n; i++) {
      if (auto v = GetToken(ivalue[i])) {
        result[i] = v.value();
      } else {
        return false;
      }
    }

    return true;
  };

  if (h.HasExplicitItems()) {
    std::vector<value::token> items;
    if (!ReadFn(items)) {
      _err += "Failed to read ListOp::ExplicitItems.\n";
      return false;
    }

    d->SetExplicitItems(items);
  }

  if (h.HasAddedItems()) {
    std::vector<value::token> items;
    if (!ReadFn(items)) {
      _err += "Failed to read ListOp::AddedItems.\n";
      return false;
    }

    d->SetAddedItems(items);
  }

  if (h.HasPrependedItems()) {
    std::vector<value::token> items;
    if (!ReadFn(items)) {
      _err += "Failed to read ListOp::PrependedItems.\n";
      return false;
    }

    d->SetPrependedItems(items);
  }

  if (h.HasAppendedItems()) {
    std::vector<value::token> items;
    if (!ReadFn(items)) {
      _err += "Failed to read ListOp::AppendedItems.\n";
      return false;
    }

    d->SetAppendedItems(items);
  }

  if (h.HasDeletedItems()) {
    std::vector<value::token> items;
    if (!ReadFn(items)) {
      _err += "Failed to read ListOp::DeletedItems.\n";
      return false;
    }

    d->SetDeletedItems(items);
  }

  if (h.HasOrderedItems()) {
    std::vector<value::token> items;
    if (!ReadFn(items)) {
      _err += "Failed to read ListOp::OrderedItems.\n";
      return false;
    }

    d->SetOrderedItems(items);
  }

  return true;
}

bool CrateReader::ReadStringListOp(ListOp<std::string> *d) {
  // read ListOpHeader
  ListOpHeader h;
  if (!_sr->read1(&h.bits)) {
    _err += "Failed to read ListOpHeader\n";
    return false;
  }

  if (h.IsExplicit()) {
    d->ClearAndMakeExplicit();
  }

  // array data is not compressed
  auto ReadFn = [this](std::vector<std::string> &result) -> bool {
    uint64_t n{0};
    if (!_sr->read8(&n)) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read the number of array elements.");
      return false;
    }


    if (n > _config.maxArrayElements) {
      _err += "Too many ListOp elements.\n";
      return false;
    }

    CHECK_MEMORY_USAGE(size_t(n) * sizeof(crate::Index));

    std::vector<crate::Index> ivalue(static_cast<size_t>(n));

    if (!_sr->read(size_t(n) * sizeof(crate::Index),
                   size_t(n) * sizeof(crate::Index),
                   reinterpret_cast<uint8_t *>(ivalue.data()))) {
      _err += "Failed to read ListOp data.\n";
      return false;
    }

    // reconstruct
    result.resize(static_cast<size_t>(n));
    for (size_t i = 0; i < n; i++) {
      if (auto v = GetStringToken(ivalue[i])) {
        result[i] = v.value().str();
      } else {
        return false;
      }
    }

    return true;
  };

  if (h.HasExplicitItems()) {
    std::vector<std::string> items;
    if (!ReadFn(items)) {
      _err += "Failed to read ListOp::ExplicitItems.\n";
      return false;
    }

    d->SetExplicitItems(items);
  }

  if (h.HasAddedItems()) {
    std::vector<std::string> items;
    if (!ReadFn(items)) {
      _err += "Failed to read ListOp::AddedItems.\n";
      return false;
    }

    d->SetAddedItems(items);
  }

  if (h.HasPrependedItems()) {
    std::vector<std::string> items;
    if (!ReadFn(items)) {
      _err += "Failed to read ListOp::PrependedItems.\n";
      return false;
    }

    d->SetPrependedItems(items);
  }

  if (h.HasAppendedItems()) {
    std::vector<std::string> items;
    if (!ReadFn(items)) {
      _err += "Failed to read ListOp::AppendedItems.\n";
      return false;
    }

    d->SetAppendedItems(items);
  }

  if (h.HasDeletedItems()) {
    std::vector<std::string> items;
    if (!ReadFn(items)) {
      _err += "Failed to read ListOp::DeletedItems.\n";
      return false;
    }

    d->SetDeletedItems(items);
  }

  if (h.HasOrderedItems()) {
    std::vector<std::string> items;
    if (!ReadFn(items)) {
      _err += "Failed to read ListOp::OrderedItems.\n";
      return false;
    }

    d->SetOrderedItems(items);
  }

  return true;
}

bool CrateReader::ReadPathListOp(ListOp<Path> *d) {
  // read ListOpHeader
  ListOpHeader h;
  if (!_sr->read1(&h.bits)) {
    PUSH_ERROR("Failed to read ListOpHeader.");
    return false;
  }

  if (h.IsExplicit()) {
    DCOUT("IsExplicit()");
    d->ClearAndMakeExplicit();
  }

  // array data is not compressed
  auto ReadFn = [this](std::vector<Path> &result) -> bool {
    uint64_t n{0};
    if (!_sr->read8(&n)) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read the number of array elements.");
      return false;
    }

    if (n > _config.maxArrayElements) {
      _err += "Too many ListOp elements.\n";
      return false;
    }

    CHECK_MEMORY_USAGE(size_t(n) * sizeof(crate::Index));

    std::vector<crate::Index> ivalue(static_cast<size_t>(n));

    if (!_sr->read(size_t(n) * sizeof(crate::Index),
                   size_t(n) * sizeof(crate::Index),
                   reinterpret_cast<uint8_t *>(ivalue.data()))) {
      PUSH_ERROR("Failed to read ListOp data..");
      return false;
    }

    // reconstruct
    result.resize(static_cast<size_t>(n));
    for (size_t i = 0; i < n; i++) {
      if (auto pv = GetPath(ivalue[i])) {
        result[i] = pv.value();
      } else {
        PUSH_ERROR("Invalid Index for Path.");
        return false;
      }
    }

    return true;
  };

  if (h.HasExplicitItems()) {
    std::vector<Path> items;
    if (!ReadFn(items)) {
      _err += "Failed to read ListOp::ExplicitItems.\n";
      return false;
    }

    d->SetExplicitItems(items);
  }

  if (h.HasAddedItems()) {
    std::vector<Path> items;
    if (!ReadFn(items)) {
      _err += "Failed to read ListOp::AddedItems.\n";
      return false;
    }

    d->SetAddedItems(items);
  }

  if (h.HasPrependedItems()) {
    std::vector<Path> items;
    if (!ReadFn(items)) {
      _err += "Failed to read ListOp::PrependedItems.\n";
      return false;
    }

    d->SetPrependedItems(items);
  }

  if (h.HasAppendedItems()) {
    std::vector<Path> items;
    if (!ReadFn(items)) {
      _err += "Failed to read ListOp::AppendedItems.\n";
      return false;
    }

    d->SetAppendedItems(items);
  }

  if (h.HasDeletedItems()) {
    std::vector<Path> items;
    if (!ReadFn(items)) {
      _err += "Failed to read ListOp::DeletedItems.\n";
      return false;
    }

    d->SetDeletedItems(items);
  }

  if (h.HasOrderedItems()) {
    std::vector<Path> items;
    if (!ReadFn(items)) {
      _err += "Failed to read ListOp::OrderedItems.\n";
      return false;
    }

    d->SetOrderedItems(items);
  }

  return true;
}

template<>
bool CrateReader::ReadArray(std::vector<Reference> *d) {

  if (!d) {
    return false;
  }

  uint64_t n{0};
  if (!_sr->read8(&n)) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read the number of array elements.");
    return false;
  }

  if (n > _config.maxArrayElements) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Too many array elements.");
  }

  CHECK_MEMORY_USAGE(sizeof(Reference) * n);

  for (size_t i = 0; i < n; i++) {
    Reference p;
    if (!ReadReference(&p)) {
      return false;
    }
    d->emplace_back(p);
  }

  return true;
}

template<>
bool CrateReader::ReadArray(std::vector<Payload> *d) {

  if (!d) {
    return false;
  }

  uint64_t n{0};
  if (VERSION_LESS_THAN_0_8_0(_version)) {
    uint32_t shapesize; // not used
    if (!_sr->read4(&shapesize)) {
      PUSH_ERROR("Failed to read the number of array elements.");
      return false;
    }
    uint32_t _n;
    if (!_sr->read4(&_n)) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read the number of array elements.");
    }
    n = _n;
  } else {
    if (!_sr->read8(&n)) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read the number of array elements.");
      return false;
    }
  }

  if (n > _config.maxArrayElements) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Too many array elements.");
  }

  CHECK_MEMORY_USAGE(sizeof(Payload) * n);

  for (size_t i = 0; i < n; i++) {
    Payload p;
    if (!ReadPayload(&p)) {
      return false;
    }
    d->emplace_back(p);
  }

  return true;
}

// T = int, uint, int64, uint64
template<typename T>
//typename std::enable_if<CrateReader::IsIntType<T>::value, bool>::type
bool CrateReader::ReadArray(std::vector<T> *d) {

  if (!d) {
    return false;
  }

  uint64_t n{0};
  if (VERSION_LESS_THAN_0_8_0(_version)) {
    uint32_t shapesize; // not used
    if (!_sr->read4(&shapesize)) {
      PUSH_ERROR("Failed to read the number of array elements.");
      return false;
    }
    uint32_t _n;
    if (!_sr->read4(&_n)) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read the number of array elements.");
    }
    n = _n;
  } else {
    if (!_sr->read8(&n)) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read the number of array elements.");
      return false;
    }
  }

  if (n > _config.maxArrayElements) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Too many array elements.");
  }

  if (n == 0) {
    return true;
  }

  CHECK_MEMORY_USAGE(sizeof(T) * size_t(n));

  d->resize(size_t(n));
  if (_sr->read(sizeof(T) * n, sizeof(T) * size_t(n), reinterpret_cast<uint8_t *>(d->data()))) {
    return false;
  }

  return true;
}

template<typename T>
bool CrateReader::ReadListOp(ListOp<T> *d) {
  // read ListOpHeader
  ListOpHeader h;
  if (!_sr->read1(&h.bits)) {
    PUSH_ERROR("Failed to read ListOpHeader.");
    return false;
  }

  if (h.IsExplicit()) {
    d->ClearAndMakeExplicit();
  }

  //
  // NOTE: array data is not compressed even for Int type
  //

  if (h.HasExplicitItems()) {
    std::vector<T> items;
    if (!ReadArray(&items)) {
      _err += "Failed to read ListOp::ExplicitItems.\n";
      return false;
    }

    d->SetExplicitItems(items);
  }

  if (h.HasAddedItems()) {
    std::vector<T> items;
    if (!ReadArray(&items)) {
      _err += "Failed to read ListOp::AddedItems.\n";
      return false;
    }

    d->SetAddedItems(items);
  }

  if (h.HasPrependedItems()) {
    std::vector<T> items;
    if (!ReadArray(&items)) {
      _err += "Failed to read ListOp::PrependedItems.\n";
      return false;
    }

    d->SetPrependedItems(items);
  }

  if (h.HasAppendedItems()) {
    std::vector<T> items;
    if (!ReadArray(&items)) {
      _err += "Failed to read ListOp::AppendedItems.\n";
      return false;
    }

    d->SetAppendedItems(items);
  }

  if (h.HasDeletedItems()) {
    std::vector<T> items;
    if (!ReadArray(&items)) {
      _err += "Failed to read ListOp::DeletedItems.\n";
      return false;
    }

    d->SetDeletedItems(items);
  }

  if (h.HasOrderedItems()) {
    std::vector<T> items;
    if (!ReadArray(&items)) {
      _err += "Failed to read ListOp::OrderedItems.\n";
      return false;
    }

    d->SetOrderedItems(items);
  }

  return true;
}


bool CrateReader::ReadVariantSelectionMap(VariantSelectionMap *d) {

  if (!d) {
    return false;
  }

  // map<string, string>

  // n
  // [key, value] * n

  uint64_t sz;
  if (!_sr->read8(&sz)) {
    _err += "Failed to read the number of elements for VariantsMap data.\n";
    return false;
  }

  if (sz > _config.maxVariantsMapElements) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "The number of elements for VariantsMap data is too large. Max = " << std::to_string(_config.maxVariantsMapElements) << ", but got " << std::to_string(sz));
  }

  for (size_t i = 0; i < sz; i++) {
    std::string key;
    if (!ReadString(&key)) {
      return false;
    }

    std::string value;
    if (!ReadString(&value)) {
      return false;
    }

    // TODO: Duplicate key check?
    d->emplace(key, value);
  }

  return true;
}

bool CrateReader::ReadCustomData(CustomDataType *d) {
  CustomDataType dict;
  uint64_t sz;
  if (!_sr->read8(&sz)) {
    _err += "Failed to read the number of elements for Dictionary data.\n";
    return false;
  }

  if (sz > _config.maxDictElements) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "The number of elements for Dictionary data is too large. Max = " << std::to_string(_config.maxDictElements) << ", but got " << std::to_string(sz));
  }

  DCOUT("# o elements in dict" << sz);

  while (sz--) {
    // key(StringIndex)
    std::string key;

    if (!ReadString(&key)) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read key string for Dictionary element.");
    }

    // 8byte for the offset for recursive value. See RecursiveRead() in
    // crateFile.cpp for details.
    int64_t offset{0};
    if (!_sr->read8(&offset)) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read the offset for value in Dictionary.");
    }

    // -8 to compensate sizeof(offset)
    if (!_sr->seek_from_current(offset - 8)) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to seek. Invalid offset value: " + std::to_string(offset));
    }

    DCOUT("key = " << key);

    crate::ValueRep rep{0};
    if (!ReadValueRep(&rep)) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read value for Dictionary element.");
    }

    DCOUT("vrep =" << crate::GetCrateDataTypeName(rep.GetType()));

    auto saved_position = _sr->tell();

    crate::CrateValue value;
    if (!UnpackValueRep(rep, &value)) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to unpack value of Dictionary element.");
    }

    if (dict.count(key)) {
      // Duplicated key. maybe ok?
    }
    // CrateValue -> MetaVariable
    MetaVariable var;

    var.set_value(key, value.get_raw());

    dict[key] = var;

    if (!_sr->seek_set(saved_position)) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to set seek.");
    }
  }

  (*d) = std::move(dict);
  return true;
}

bool CrateReader::UnpackInlinedValueRep(const crate::ValueRep &rep,
                                        crate::CrateValue *value) {
  if (!rep.IsInlined()) {
    PUSH_ERROR("ValueRep must be inlined value representation.");
    return false;
  }

  const auto tyRet = crate::GetCrateDataType(rep.GetType());
  if (!tyRet) {
    PUSH_ERROR(tyRet.error());
    return false;
  }

  if (rep.IsCompressed()) {
    PUSH_ERROR("Inlinved value must not be compressed.");
    return false;
  }

  if (rep.IsArray()) {
    PUSH_ERROR("Inlined value must not be an array.");
    return false;
  }

  const auto dty = tyRet.value();
  DCOUT(crate::GetCrateDataTypeRepr(dty));

  uint32_t d = (rep.GetPayload() & ((1ull << (sizeof(uint32_t) * 8)) - 1));
  DCOUT("d = " << d);

  // TODO(syoyo): Use template SFINE?
  switch (dty.dtype_id) {
    case crate::CrateDataTypeId::NumDataTypes:
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_INVALID: {
      PUSH_ERROR("`Invalid` DataType.");
      return false;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_BOOL: {
      value->Set(d ? true : false);
      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_ASSET_PATH: {
      // AssetPath = TokenIndex for inlined value.
      if (auto v = GetToken(crate::Index(d))) {
        std::string str = v.value().str();

        value::AssetPath assetp(str);
        value->Set(assetp);
        return true;
      } else {
        PUSH_ERROR("Invalid Index for AssetPath.");
        return false;
      }
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_TOKEN: {
      if (auto v = GetToken(crate::Index(d))) {
        value::token tok = v.value();

        DCOUT("value.token = " << tok);

        value->Set(tok);

        return true;
      } else {
        PUSH_ERROR("Invalid Index for Token.");
        return false;
      }
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_STRING: {
      if (auto v = GetStringToken(crate::Index(d))) {
        std::string str = v.value().str();

        DCOUT("value.string = " << str);

        value->Set(str);

        return true;
      } else {
        PUSH_ERROR("Invalid Index for StringToken.");
        return false;
      }
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_SPECIFIER: {
      if (d >= static_cast<int>(Specifier::Invalid)) {
        _err += "Invalid value for Specifier\n";
        return false;
      }
      Specifier val = static_cast<Specifier>(d);

      value->Set(val);

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_PERMISSION: {
      if (d >= static_cast<int>(Permission::Invalid)) {
        _err += "Invalid value for Permission\n";
        return false;
      }
      Permission val = static_cast<Permission>(d);

      value->Set(val);

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_VARIABILITY: {
      if (d >= static_cast<int>(Variability::Invalid)) {
        _err += "Invalid value for Variability\n";
        return false;
      }
      Variability val = static_cast<Variability>(d);

      value->Set(val);

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_UCHAR: {
      uint8_t val;
      memcpy(&val, &d, 1);

      DCOUT("value.uchar = " << val);

      value->Set(val);

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_INT: {
      int ival;
      memcpy(&ival, &d, sizeof(int));

      DCOUT("value.int = " << ival);

      value->Set(ival);

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_UINT: {
      uint32_t val;
      memcpy(&val, &d, sizeof(uint32_t));

      DCOUT("value.uint = " << val);

      value->Set(val);

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_INT64: {
      // stored as int
      int _ival;
      memcpy(&_ival, &d, sizeof(int));

      DCOUT("value.int = " << _ival);

      int64_t ival = static_cast<int64_t>(_ival);

      value->Set(ival);

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_UINT64: {
      // stored as uint32
      uint32_t _ival;
      memcpy(&_ival, &d, sizeof(uint32_t));

      DCOUT("value.int = " << _ival);

      uint64_t ival = static_cast<uint64_t>(_ival);

      value->Set(ival);

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_HALF: {
      value::half f;
      memcpy(&f, &d, sizeof(value::half));

      DCOUT("value.half = " << f);

      value->Set(f);

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_FLOAT: {
      float f;
      memcpy(&f, &d, sizeof(float));

      DCOUT("value.float = " << f);

      value->Set(f);

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_DOUBLE: {
      // stored as float
      float _f;
      memcpy(&_f, &d, sizeof(float));

      double f = static_cast<double>(_f);

      value->Set(f);

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_MATRIX2D: {
      // Matrix contains diagnonal components only, and values are represented
      // in int8
      int8_t data[2];
      memcpy(&data, &d, 2);

      value::matrix2d v;
      memset(v.m, 0, sizeof(value::matrix2d));
      v.m[0][0] = static_cast<double>(data[0]);
      v.m[1][1] = static_cast<double>(data[1]);

      DCOUT("value.matrix(diag) = " << v.m[0][0] << ", " << v.m[1][1] << "\n");

      value->Set(v);

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_MATRIX3D: {
      // Matrix contains diagnonal components only, and values are represented
      // in int8
      int8_t data[3];
      memcpy(&data, &d, 3);

      value::matrix3d v;
      memset(v.m, 0, sizeof(value::matrix3d));
      v.m[0][0] = static_cast<double>(data[0]);
      v.m[1][1] = static_cast<double>(data[1]);
      v.m[2][2] = static_cast<double>(data[2]);

      DCOUT("value.matrix(diag) = " << v.m[0][0] << ", " << v.m[1][1] << ", "
                                    << v.m[2][2] << "\n");

      value->Set(v);

      return true;
    }

    case crate::CrateDataTypeId::CRATE_DATA_TYPE_MATRIX4D: {
      // Matrix contains diagnonal components only, and values are represented
      // in int8
      int8_t data[4];
      memcpy(&data, &d, 4);

      value::matrix4d v;
      memset(v.m, 0, sizeof(value::matrix4d));
      v.m[0][0] = static_cast<double>(data[0]);
      v.m[1][1] = static_cast<double>(data[1]);
      v.m[2][2] = static_cast<double>(data[2]);
      v.m[3][3] = static_cast<double>(data[3]);

      DCOUT("value.matrix(diag) = " << v.m[0][0] << ", " << v.m[1][1] << ", "
                                    << v.m[2][2] << ", " << v.m[3][3]);

      value->Set(v);

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_QUATD:
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_QUATF:
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_QUATH: {
      // Seems quaternion type is not allowed for Inlined Value.
      PUSH_ERROR("Quaternion type is not allowed for Inlined Value.");
      return false;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_VEC2D: {
      // Value is represented in int8
      int8_t data[2];
      memcpy(&data, &d, 2);

      value::double2 v;
      v[0] = double(data[0]);
      v[1] = double(data[1]);

      DCOUT("value.double2 = " << v);

      value->Set(v);

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_VEC2F: {
      // Value is represented in int8
      int8_t data[2];
      memcpy(&data, &d, 2);

      value::float2 v;
      v[0] = float(data[0]);
      v[1] = float(data[1]);

      DCOUT("value.float2 = " << v);

      value->Set(v);

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_VEC2H: {
      // Value is represented in int8
      int8_t data[2];
      memcpy(&data, &d, 2);

      value::half3 v;
      v[0] = value::float_to_half_full(float(data[0]));
      v[1] = value::float_to_half_full(float(data[1]));

      DCOUT("value.half2 = " << v);

      value->Set(v);

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_VEC2I: {
      // Value is represented in int8
      int8_t data[2];
      memcpy(&data, &d, 2);

      value::int2 v;
      v[0] = int(data[0]);
      v[1] = int(data[1]);

      DCOUT("value.int2 = " << v);

      value->Set(v);

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_VEC3D: {
      // Value is represented in int8
      int8_t data[3];
      memcpy(&data, &d, 3);

      value::double3 v;
      v[0] = double(data[0]);
      v[1] = double(data[1]);
      v[2] = double(data[2]);

      DCOUT("value.double3 = " << v);

      value->Set(v);

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_VEC3F: {
      // Value is represented in int8
      int8_t data[3];
      memcpy(&data, &d, 3);

      value::float3 v;
      v[0] = float(data[0]);
      v[1] = float(data[1]);
      v[2] = float(data[2]);

      DCOUT("value.float3 = " << v);

      value->Set(v);

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_VEC3H: {
      // Value is represented in int8
      int8_t data[3];
      memcpy(&data, &d, 3);

      value::half3 v;
      v[0] = value::float_to_half_full(float(data[0]));
      v[1] = value::float_to_half_full(float(data[1]));
      v[2] = value::float_to_half_full(float(data[2]));

      DCOUT("value.half3 = " << v);

      value->Set(v);

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_VEC3I: {
      // Value is represented in int8
      int8_t data[3];
      memcpy(&data, &d, 3);

      value::int3 v;
      v[0] = static_cast<int32_t>(data[0]);
      v[1] = static_cast<int32_t>(data[1]);
      v[2] = static_cast<int32_t>(data[2]);

      DCOUT("value.int3 = " << v);

      value->Set(v);

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_VEC4D: {
      // Value is represented in int8
      int8_t data[4];
      memcpy(&data, &d, 4);

      value::double4 v;
      v[0] = static_cast<double>(data[0]);
      v[1] = static_cast<double>(data[1]);
      v[2] = static_cast<double>(data[2]);
      v[3] = static_cast<double>(data[3]);

      DCOUT("value.doublef = " << v);

      value->Set(v);

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_VEC4F: {
      // Value is represented in int8
      int8_t data[4];
      memcpy(&data, &d, 4);

      value::float4 v;
      v[0] = static_cast<float>(data[0]);
      v[1] = static_cast<float>(data[1]);
      v[2] = static_cast<float>(data[2]);
      v[3] = static_cast<float>(data[3]);

      DCOUT("value.vec4f = " << v);

      value->Set(v);

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_VEC4H: {
      // Value is represented in int8
      int8_t data[4];
      memcpy(&data, &d, 4);

      value::half4 v;
      v[0] = value::float_to_half_full(float(data[0]));
      v[1] = value::float_to_half_full(float(data[0]));
      v[2] = value::float_to_half_full(float(data[0]));
      v[3] = value::float_to_half_full(float(data[0]));

      DCOUT("value.vec4h = " << v);

      value->Set(v);

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_VEC4I: {
      // Value is represented in int8
      int8_t data[4];
      memcpy(&data, &d, 4);

      value::int4 v;
      v[0] = static_cast<int32_t>(data[0]);
      v[1] = static_cast<int32_t>(data[1]);
      v[2] = static_cast<int32_t>(data[2]);
      v[3] = static_cast<int32_t>(data[3]);

      DCOUT("value.vec4i = " << v);

      value->Set(v);

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_DICTIONARY: {
      // empty dict is allowed
      // TODO: empty(zero value) check?
      //crate::CrateValue::Dictionary dict;
      CustomDataType dict; // use CustomDataType for Dict
      value->Set(dict);
      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_VALUE_BLOCK: {
      // Guess No content for ValueBlock
      value::ValueBlock block;
      value->Set(block);
      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_TOKEN_LIST_OP:
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_STRING_LIST_OP:
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_PATH_LIST_OP:
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_REFERENCE_LIST_OP:
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_INT_LIST_OP:
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_INT64_LIST_OP:
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_UINT_LIST_OP:
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_UINT64_LIST_OP: {
      PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("ListOp data type `{}` cannot be inlined.",
          crate::GetCrateDataTypeName(dty.dtype_id)));
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_PATH_VECTOR:
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_TOKEN_VECTOR:
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_VARIANT_SELECTION_MAP:
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_TIME_SAMPLES:
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_DOUBLE_VECTOR:
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_PAYLOAD:
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_PAYLOAD_LIST_OP:
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_LAYER_OFFSET_VECTOR:
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_STRING_VECTOR: {
      PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Data type `{}` cannot be inlined.",
          crate::GetCrateDataTypeName(dty.dtype_id)));
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_VALUE:
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_UNREGISTERED_VALUE:
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_UNREGISTERED_VALUE_LIST_OP:
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_TIME_CODE: {
      PUSH_ERROR(
          "Invalid data type(or maybe not supported in TinyUSDZ yet) for "
          "Inlined value: " +
          crate::GetCrateDataTypeName(dty.dtype_id));
      return false;
    }
  }

  // Should never reach here.
  return false;
}

#if 0
template<T>
CrateReader::UnpackArrayValue(CrateDataTypeId dty, crate::CrateValue *value_out) {
  uint64_t n;
  if (!_sr->read8(&n)) {
    PUSH_ERROR("Failed to read the number of array elements.");
    return false;
  }

  std::vector<crate::Index> v(static_cast<size_t>(n));
  if (!_sr->read(size_t(n) * sizeof(crate::Index),
                 size_t(n) * sizeof(crate::Index),
                 reinterpret_cast<uint8_t *>(v.data()))) {
    PUSH_ERROR("Failed to read array data.");
    return false;
  }

  return true;
}
#endif

bool CrateReader::UnpackValueRep(const crate::ValueRep &rep,
                                 crate::CrateValue *value) {
  if (rep.IsInlined()) {
    return UnpackInlinedValueRep(rep, value);
  }

  DCOUT("ValueRep type value = " << rep.GetType());
  auto tyRet = crate::GetCrateDataType(rep.GetType());
  if (!tyRet) {
    PUSH_ERROR(tyRet.error());
  }

  const auto dty = tyRet.value();

#define TODO_IMPLEMENT(__dty)                                            \
  {                                                                      \
    PUSH_ERROR("TODO: '" + crate::GetCrateDataTypeName(__dty.dtype_id) + \
               "' data is not yet implemented.");                        \
    return false;                                                        \
  }

#define COMPRESS_UNSUPPORTED_CHECK(__dty)                                     \
  if (rep.IsCompressed()) {                                                   \
    PUSH_ERROR("Compressed [" + crate::GetCrateDataTypeName(__dty.dtype_id) + \
               "' data is not yet supported.");                               \
    return false;                                                             \
  }

#define NON_ARRAY_UNSUPPORTED_CHECK(__dty)                                   \
  if (!rep.IsArray()) {                                                      \
    PUSH_ERROR("Non array '" + crate::GetCrateDataTypeName(__dty.dtype_id) + \
               "' data is not yet supported.");                              \
    return false;                                                            \
  }

#define ARRAY_UNSUPPORTED_CHECK(__dty)                                      \
  if (rep.IsArray()) {                                                      \
    PUSH_ERROR("Array of '" + crate::GetCrateDataTypeName(__dty.dtype_id) + \
               "' data type is not yet supported.");                        \
    return false;                                                           \
  }

  // payload is the offset to data.
  uint64_t offset = rep.GetPayload();
  if (!_sr->seek_set(offset)) {
    PUSH_ERROR("Invalid offset.");
    return false;
  }

  switch (dty.dtype_id) {
    case crate::CrateDataTypeId::NumDataTypes:
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_INVALID: {
      DCOUT("dtype_id = " << to_string(uint32_t(dty.dtype_id)));
      PUSH_ERROR("`Invalid` DataType.");
      return false;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_BOOL: {
      COMPRESS_UNSUPPORTED_CHECK(dty)
      NON_ARRAY_UNSUPPORTED_CHECK(dty)

      if (rep.IsArray()) {
        std::vector<bool> v;

        if (rep.GetPayload() == 0) { // empty array
          value->Set(v);
          return true;
        }

        // bool is encoded as 8bit value.

        uint64_t n;
        if (!_sr->read8(&n)) {
          PUSH_ERROR("Failed to read the number of array elements.");
          return false;
        }

        if (n > _config.maxArrayElements) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("# of bool array too large. TinyUSDZ limites it up to {}", _config.maxArrayElements));
        }

        CHECK_MEMORY_USAGE(n * sizeof(uint8_t));

        std::vector<uint8_t> data(static_cast<size_t>(n));
        if (!_sr->read(size_t(n) * sizeof(uint8_t),
                       size_t(n) * sizeof(uint8_t),
                       reinterpret_cast<uint8_t *>(data.data()))) {
          PUSH_ERROR("Failed to read bool array.");
          return false;
        }

        // to std::vector<bool>, whose underlying storage may use 1bit.
        v.resize(size_t(n));
        for (size_t i = 0; i < n; i++) {
          v[i] = data[i] ? true : false;
        }

        value->Set(v);
        return true;

      } else {
        // non array bool should be inline encoded.
        PUSH_ERROR_AND_RETURN_TAG(kTag, "bool value must be inlined.");
      }
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_ASSET_PATH: {
      COMPRESS_UNSUPPORTED_CHECK(dty)

      // AssetPath is encoded as StringIndex for uninlined and array value
      // NOTE: inlined value uses TokenIndex.

      if (rep.IsArray()) {

        if (rep.GetPayload() == 0) { // empty array
          value->Set(std::vector<value::AssetPath>());
          return true;
        }

        uint64_t n{0};
        if (VERSION_LESS_THAN_0_8_0(_version)) {
          uint32_t shapesize; // not used
          if (!_sr->read4(&shapesize)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
          uint32_t _n;
          if (!_sr->read4(&_n)) {
            PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read the number of array elements.");
          }
          n = _n;
        } else {
          if (!_sr->read8(&n)) {
            PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read the number of array elements.");
            return false;
          }
        }

        if (n > _config.maxAssetPathElements) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("# of AssetPaths too large. TinyUSDZ limites it up to {}", _config.maxAssetPathElements));
        }

        CHECK_MEMORY_USAGE(n * sizeof(crate::Index));

        std::vector<crate::Index> v(static_cast<size_t>(n));
        if (!_sr->read(size_t(n) * sizeof(crate::Index),
                       size_t(n) * sizeof(crate::Index),
                       reinterpret_cast<uint8_t *>(v.data()))) {
          PUSH_ERROR("Failed to read StringIndex array.");
          return false;
        }

        std::vector<value::AssetPath> apaths(static_cast<size_t>(n));

        for (size_t i = 0; i < n; i++) {
          if (auto tokv = GetStringToken(v[i])) {
            DCOUT("StringToken[" << i << "] = " << tokv.value());
            apaths[i] = value::AssetPath(tokv.value().str());
          } else {
            return false;
          }
        }

        value->Set(apaths);
        return true;
      } else {

        CHECK_MEMORY_USAGE(sizeof(crate::Index));

        crate::Index v;
        if (!_sr->read(sizeof(crate::Index), sizeof(crate::Index),
                       reinterpret_cast<uint8_t *>(&v))) {
          PUSH_ERROR("Failed to read uint64 data.");
          return false;
        }

        DCOUT("StrIndex = " << v);

        if (auto tokv = GetStringToken(v)) {
          DCOUT("StringToken = " << tokv.value());
          value::AssetPath apath(tokv.value().str());
          value->Set(apath);
        } else {
          PUSH_ERROR_AND_RETURN("Invalid StringToken found.");
          return false;
        }

        return true;
      }
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_TOKEN: {
      COMPRESS_UNSUPPORTED_CHECK(dty)
      NON_ARRAY_UNSUPPORTED_CHECK(dty)

      if (rep.IsArray()) {

        if (rep.GetPayload() == 0) { // empty array
          value->Set(std::vector<value::token>());
          return true;
        }

        uint64_t n;
        if (!_sr->read8(&n)) {
          PUSH_ERROR("Failed to read the number of array elements.");
          return false;
        }

        if (n > _config.maxArrayElements) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Token array too large. TinyUSDZ limits it up to {}", _config.maxArrayElements));
        }

        CHECK_MEMORY_USAGE(n * sizeof(crate::Index));

        std::vector<crate::Index> v;
        v.resize(static_cast<size_t>(n));
        if (!_sr->read(size_t(n) * sizeof(crate::Index),
                       size_t(n) * sizeof(crate::Index),
                       reinterpret_cast<uint8_t *>(v.data()))) {
          PUSH_ERROR("Failed to read TokenIndex array.");
          return false;
        }

        std::vector<value::token> tokens(static_cast<size_t>(n));

        for (size_t i = 0; i < n; i++) {
          if (auto tokv = GetToken(v[i])) {
            DCOUT("Token[" << i << "] = " << tokv.value());
            tokens[i] = tokv.value();
          } else {
            return false;
          }
        }

        value->Set(tokens);
        return true;
      } else {
        return false;
      }
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_STRING: {
      COMPRESS_UNSUPPORTED_CHECK(dty)

      if (rep.IsArray()) {
        uint64_t n;
        if (!_sr->read8(&n)) {
          PUSH_ERROR("Failed to read the number of array elements.");
          return false;
        }

        if (n > _config.maxArrayElements) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("String array too large. TinyUSDZ limites it up to {}", _config.maxArrayElements));
        }

        CHECK_MEMORY_USAGE(n * sizeof(crate::Index));

        std::vector<crate::Index> v(static_cast<size_t>(n));
        if (!_sr->read(size_t(n) * sizeof(crate::Index),
                       size_t(n) * sizeof(crate::Index),
                       reinterpret_cast<uint8_t *>(v.data()))) {
          PUSH_ERROR("Failed to read TokenIndex array.");
          return false;
        }

        std::vector<std::string> stringArray(static_cast<size_t>(n));

        for (size_t i = 0; i < n; i++) {
          if (auto stok = GetStringToken(v[i])) {
            stringArray[i] = stok.value().str();
          } else {
            return false;
          }
        }

        DCOUT("stringArray = " << stringArray);

        // TODO: Use token type?
        value->Set(stringArray);

        return true;
      } else {
        return false;
      }
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_SPECIFIER:
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_PERMISSION:
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_VARIABILITY: {
      PUSH_ERROR("TODO: Specifier/Permission/Variability. isArray "
                 << rep.IsArray() << ", isCompressed " << rep.IsCompressed());
      return false;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_UCHAR: {
      NON_ARRAY_UNSUPPORTED_CHECK(dty)
      TODO_IMPLEMENT(dty)
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_INT: {
      NON_ARRAY_UNSUPPORTED_CHECK(dty)

      if (rep.IsArray()) {
        std::vector<int32_t> v;
        if (rep.GetPayload() == 0) { // empty array
          value->Set(v);
          return true;
        }
        if (!ReadIntArray(rep.IsCompressed(), &v)) {
          PUSH_ERROR("Failed to read Int array.");
          return false;
        }

        if (v.empty()) {
          PUSH_ERROR("Empty int array.");
          return false;
        }

        DCOUT("IntArray = " << value::print_array_snipped(v));

        value->Set(v);
        return true;
      } else {
        return false;
      }
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_UINT: {
      NON_ARRAY_UNSUPPORTED_CHECK(dty)

      if (rep.IsArray()) {
        std::vector<uint32_t> v;
        if (rep.GetPayload() == 0) { // empty array
          value->Set(v);
          return true;
        }
        if (!ReadIntArray(rep.IsCompressed(), &v)) {
          PUSH_ERROR("Failed to read UInt array.");
          return false;
        }

        if (v.empty()) {
          PUSH_ERROR("Empty uint array.");
          return false;
        }

        DCOUT("UIntArray = " << value::print_array_snipped(v));

        value->Set(v);
        return true;
      } else {
        return false;
      }
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_INT64: {
      if (rep.IsArray()) {
        std::vector<int64_t> v;
        if (rep.GetPayload() == 0) { // empty array
          value->Set(v);
          return true;
        }
        if (!ReadIntArray(rep.IsCompressed(), &v)) {
          PUSH_ERROR("Failed to read Int64 array.");
          return false;
        }

        if (v.empty()) {
          PUSH_ERROR("Empty int64 array.");
          return false;
        }

        DCOUT("Int64Array = " << v);

        value->Set(v);
        return true;
      } else {
        COMPRESS_UNSUPPORTED_CHECK(dty)

        CHECK_MEMORY_USAGE(sizeof(int64_t));

        int64_t v;
        if (!_sr->read(sizeof(int64_t), sizeof(int64_t),
                       reinterpret_cast<uint8_t *>(&v))) {
          PUSH_ERROR("Failed to read int64 data.");
          return false;
        }

        DCOUT("int64 = " << v);

        value->Set(v);
        return true;
      }
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_UINT64: {
      if (rep.IsArray()) {
        std::vector<uint64_t> v;
        if (rep.GetPayload() == 0) { // empty array
          value->Set(v);
          return true;
        }

        if (!ReadIntArray(rep.IsCompressed(), &v)) {
          PUSH_ERROR("Failed to read UInt64 array.");
          return false;
        }

        if (v.empty()) {
          PUSH_ERROR("Empty uint64 array.");
          return false;
        }

        DCOUT("UInt64Array = " << value::print_array_snipped(v));

        value->Set(v);
        return true;
      } else {
        COMPRESS_UNSUPPORTED_CHECK(dty)

        CHECK_MEMORY_USAGE(sizeof(uint64_t));

        uint64_t v;
        if (!_sr->read(sizeof(uint64_t), sizeof(uint64_t),
                       reinterpret_cast<uint8_t *>(&v))) {
          PUSH_ERROR("Failed to read uint64 data.");
          return false;
        }

        DCOUT("uint64 = " << v);

        value->Set(v);
        return true;
      }
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_HALF: {
      if (rep.IsArray()) {
        std::vector<value::half> v;
        if (rep.GetPayload() == 0) { // empty array
          value->Set(v);
          return true;
        }
        if (!ReadHalfArray(rep.IsCompressed(), &v)) {
          PUSH_ERROR("Failed to read half array value.");
          return false;
        }

        value->Set(v);

        return true;
      } else {
        PUSH_ERROR("Non-inlined, non-array Half value is invalid.");
        return false;
      }
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_FLOAT: {
      if (rep.IsArray()) {
        std::vector<float> v;
        if (rep.GetPayload() == 0) { // empty array
          value->Set(v);
          return true;
        }
        if (!ReadFloatArray(rep.IsCompressed(), &v)) {
          PUSH_ERROR("Failed to read float array value.");
          return false;
        }

        DCOUT("FloatArray = " << value::print_array_snipped(v));

        value->Set(v);

        return true;
      } else {
        COMPRESS_UNSUPPORTED_CHECK(dty)

        PUSH_ERROR("Non-inlined, non-array Float value is not supported.");
        return false;
      }
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_DOUBLE: {
      if (rep.IsArray()) {
        std::vector<double> v;
        if (rep.GetPayload() == 0) { // empty array
          value->Set(v);
          return true;
        }
        if (!ReadDoubleArray(rep.IsCompressed(), &v)) {
          PUSH_ERROR("Failed to read Double value.");
          return false;
        }

        DCOUT("DoubleArray = " << value::print_array_snipped(v));
        value->Set(v);

        return true;
      } else {
        COMPRESS_UNSUPPORTED_CHECK(dty)

        CHECK_MEMORY_USAGE(sizeof(double));

        double v{0.0};
        if (!_sr->read_double(&v)) {
          PUSH_ERROR("Failed to read Double value.");
          return false;
        }

        DCOUT("Double " << v);

        value->Set(v);

        return true;
      }
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_MATRIX2D: {
      COMPRESS_UNSUPPORTED_CHECK(dty)

      if (rep.IsArray()) {
        std::vector<value::matrix2d> v;
        if (rep.GetPayload() == 0) { // empty array
          value->Set(v);
          return true;
        }

        uint64_t n{0};
        if (VERSION_LESS_THAN_0_8_0(_version)) {
          uint32_t shapesize; // not used
          if (!_sr->read4(&shapesize)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
          uint32_t _n;
          if (!_sr->read4(&_n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
          n = _n;
        } else {
          if (!_sr->read8(&n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
        }

        if (n == 0) {
          value->Set(v);
          return true;
        }

        if (n > _config.maxArrayElements) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Array size {} too large. maxArrayElements is set to {}. Please increase maxArrayElements in CrateReaderConfig.", n, _config.maxArrayElements));
        }

        CHECK_MEMORY_USAGE(n * sizeof(value::matrix2d));


        v.resize(static_cast<size_t>(n));
        if (!_sr->read(size_t(n) * sizeof(value::matrix2d),
                       size_t(n) * sizeof(value::matrix2d),
                       reinterpret_cast<uint8_t *>(v.data()))) {
          PUSH_ERROR("Failed to read Matrix2d array.");
          return false;
        }

        value->Set(v);

      } else {
        static_assert(sizeof(value::matrix2d) == (8 * 4), "");

        CHECK_MEMORY_USAGE(sizeof(value::matrix2d));

        value::matrix4d v;
        if (!_sr->read(sizeof(value::matrix2d), sizeof(value::matrix2d),
                       reinterpret_cast<uint8_t *>(v.m))) {
          _err += "Failed to read value of `matrix2d` type\n";
          return false;
        }

        DCOUT("value.matrix2d = " << v);

        value->Set(v);
      }

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_MATRIX3D: {
      COMPRESS_UNSUPPORTED_CHECK(dty)

      if (rep.IsArray()) {
        std::vector<value::matrix3d> v;
        if (rep.GetPayload() == 0) { // empty array
          value->Set(v);
          return true;
        }

        uint64_t n{0};
        if (VERSION_LESS_THAN_0_8_0(_version)) {
          uint32_t shapesize; // not used
          if (!_sr->read4(&shapesize)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
          uint32_t _n;
          if (!_sr->read4(&_n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
          n = _n;
        } else {
          if (!_sr->read8(&n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
        }

        if (n == 0) {
          value->Set(v);
          return true;
        }

        if (n > _config.maxArrayElements) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Array size {} too large. maxArrayElements is set to {}. Please increase maxArrayElements in CrateReaderConfig.", n, _config.maxArrayElements));
        }

        CHECK_MEMORY_USAGE(n * sizeof(value::matrix3d));

        v.resize(static_cast<size_t>(n));
        if (!_sr->read(size_t(n) * sizeof(value::matrix3d),
                       size_t(n) * sizeof(value::matrix3d),
                       reinterpret_cast<uint8_t *>(v.data()))) {
          PUSH_ERROR("Failed to read Matrix3d array.");
          return false;
        }

        value->Set(v);

      } else {
        static_assert(sizeof(value::matrix3d) == (8 * 9), "");

        CHECK_MEMORY_USAGE(sizeof(value::matrix3d));

        value::matrix3d v;
        if (!_sr->read(sizeof(value::matrix3d), sizeof(value::matrix3d),
                       reinterpret_cast<uint8_t *>(v.m))) {
          _err += "Failed to read value of `matrix3d` type\n";
          return false;
        }

        DCOUT("value.matrix3d = " << v);

        value->Set(v);
      }

      return true;
    }

    case crate::CrateDataTypeId::CRATE_DATA_TYPE_MATRIX4D: {
      COMPRESS_UNSUPPORTED_CHECK(dty)

      if (rep.IsArray()) {
        std::vector<value::matrix4d> v;
        if (rep.GetPayload() == 0) { // empty array
          value->Set(v);
          return true;
        }

        uint64_t n{0};
        if (VERSION_LESS_THAN_0_8_0(_version)) {
          uint32_t shapesize; // not used
          if (!_sr->read4(&shapesize)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
          uint32_t _n;
          if (!_sr->read4(&_n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
          n = _n;
        } else {
          if (!_sr->read8(&n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
        }

        if (n == 0) {
          value->Set(v);
          return true;
        }

        if (n > _config.maxArrayElements) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Array size {} too large. maxArrayElements is set to {}. Please increase maxArrayElements in CrateReaderConfig.", n, _config.maxArrayElements));
        }

        CHECK_MEMORY_USAGE(n * sizeof(value::matrix4d));

        v.resize(static_cast<size_t>(n));
        if (!_sr->read(size_t(n) * sizeof(value::matrix4d),
                       size_t(n) * sizeof(value::matrix4d),
                       reinterpret_cast<uint8_t *>(v.data()))) {
          PUSH_ERROR("Failed to read Matrix4d array.");
          return false;
        }

        value->Set(v);

      } else {
        static_assert(sizeof(value::matrix4d) == (8 * 16), "");

        CHECK_MEMORY_USAGE(sizeof(value::matrix4d));

        value::matrix4d v;
        if (!_sr->read(sizeof(value::matrix4d), sizeof(value::matrix4d),
                       reinterpret_cast<uint8_t *>(v.m))) {
          _err += "Failed to read value of `matrix4d` type\n";
          return false;
        }

        DCOUT("value.matrix4d = " << v);

        value->Set(v);
      }

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_QUATD: {
      if (rep.IsArray()) {
        std::vector<value::quatd> v;
        if (rep.GetPayload() == 0) { // empty array
          value->Set(v);
          return true;
        }
        uint64_t n{0};
        if (VERSION_LESS_THAN_0_8_0(_version)) {
          uint32_t shapesize; // not used
          if (!_sr->read4(&shapesize)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
          uint32_t _n;
          if (!_sr->read4(&_n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
          n = _n;
        } else {
          if (!_sr->read8(&n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
        }

        if (n == 0) {
          value->Set(v);
          return true;
        }

        if (n > _config.maxArrayElements) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Array size {} too large. maxArrayElements is set to {}. Please increase maxArrayElements in CrateReaderConfig.", n, _config.maxArrayElements));
        }

        CHECK_MEMORY_USAGE(n * sizeof(value::quatd));

        v.resize(static_cast<size_t>(n));
        if (!_sr->read(size_t(n) * sizeof(value::quatd),
                       size_t(n) * sizeof(value::quatd),
                       reinterpret_cast<uint8_t *>(v.data()))) {
          PUSH_ERROR("Failed to read Quatf array.");
          return false;
        }

        DCOUT("Quatf[] = " << v);

        value->Set(v);

      } else {
        COMPRESS_UNSUPPORTED_CHECK(dty)

        CHECK_MEMORY_USAGE(sizeof(value::quatd));

        value::quatd v;
        if (!_sr->read(sizeof(value::quatd), sizeof(value::quatd),
                       reinterpret_cast<uint8_t *>(&v))) {
          _err += "Failed to read Quatd value\n";
          return false;
        }

        DCOUT("Quatd = " << v);
        value->Set(v);
      }

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_QUATF: {
      if (rep.IsArray()) {
        std::vector<value::quatf> v;
        if (rep.GetPayload() == 0) { // empty array
          value->Set(v);
          return true;
        }
        uint64_t n{0};
        if (VERSION_LESS_THAN_0_8_0(_version)) {
          uint32_t shapesize; // not used
          if (!_sr->read4(&shapesize)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
          uint32_t _n;
          if (!_sr->read4(&_n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
          n = _n;
        } else {
          if (!_sr->read8(&n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
        }

        if (n == 0) {
          value->Set(v);
          return true;
        }

        if (n > _config.maxArrayElements) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Array size {} too large. maxArrayElements is set to {}. Please increase maxArrayElements in CrateReaderConfig.", n, _config.maxArrayElements));
        }

        CHECK_MEMORY_USAGE(n * sizeof(value::quatf));

        v.resize(static_cast<size_t>(n));
        if (!_sr->read(size_t(n) * sizeof(value::quatf),
                       size_t(n) * sizeof(value::quatf),
                       reinterpret_cast<uint8_t *>(v.data()))) {
          PUSH_ERROR("Failed to read Quatf array.");
          return false;
        }

        DCOUT("Quatf[] = " << v);

        value->Set(v);

      } else {
        COMPRESS_UNSUPPORTED_CHECK(dty)

        CHECK_MEMORY_USAGE(sizeof(value::quatf));

        value::quatf v;
        if (!_sr->read(sizeof(value::quatf), sizeof(value::quatf),
                       reinterpret_cast<uint8_t *>(&v))) {
          _err += "Failed to read Quatf value\n";
          return false;
        }

        DCOUT("Quatf = " << v);
        value->Set(v);
      }

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_QUATH: {
      if (rep.IsArray()) {
        std::vector<value::quath> v;
        if (rep.GetPayload() == 0) { // empty array
          value->Set(v);
          return true;
        }

        uint64_t n{0};
        if (VERSION_LESS_THAN_0_8_0(_version)) {
          uint32_t shapesize; // not used
          if (!_sr->read4(&shapesize)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
          uint32_t _n;
          if (!_sr->read4(&_n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
          n = _n;
        } else {
          if (!_sr->read8(&n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
        }

        if (n == 0) {
          value->Set(v);
          return true;
        }

        if (n > _config.maxArrayElements) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Array size {} too large. maxArrayElements is set to {}. Please increase maxArrayElements in CrateReaderConfig.", n, _config.maxArrayElements));
        }

        CHECK_MEMORY_USAGE(n * sizeof(value::quath));

        v.resize(static_cast<size_t>(n));
        if (!_sr->read(size_t(n) * sizeof(value::quath),
                       size_t(n) * sizeof(value::quath),
                       reinterpret_cast<uint8_t *>(v.data()))) {
          PUSH_ERROR("Failed to read Quath array.");
          return false;
        }

        DCOUT("Quath[] = " << v);

        value->Set(v);

      } else {
        COMPRESS_UNSUPPORTED_CHECK(dty)

        CHECK_MEMORY_USAGE(sizeof(value::quath));

        value::quath v;
        if (!_sr->read(sizeof(value::quath), sizeof(value::quath),
                       reinterpret_cast<uint8_t *>(&v))) {
          _err += "Failed to read Quath value\n";
          return false;
        }

        DCOUT("Quath = " << v);
        value->Set(v);
      }

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_VEC2D: {
      COMPRESS_UNSUPPORTED_CHECK(dty)

      if (rep.IsArray()) {
        std::vector<value::double2> v;
        if (rep.GetPayload() == 0) { // empty array
          value->Set(v);
          return true;
        }

        uint64_t n{0};
        if (VERSION_LESS_THAN_0_8_0(_version)) {
          uint32_t shapesize; // not used
          if (!_sr->read4(&shapesize)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
          uint32_t _n;
          if (!_sr->read4(&_n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
          n = _n;
        } else {
          if (!_sr->read8(&n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
        }

        if (n == 0) {
          value->Set(v);
          return true;
        }

        if (n > _config.maxArrayElements) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Array size {} too large. maxArrayElements is set to {}. Please increase maxArrayElements in CrateReaderConfig.", n, _config.maxArrayElements));
        }

        CHECK_MEMORY_USAGE(n * sizeof(value::double2));

        v.resize(static_cast<size_t>(n));
        if (!_sr->read(size_t(n) * sizeof(value::double2),
                       size_t(n) * sizeof(value::double2),
                       reinterpret_cast<uint8_t *>(v.data()))) {
          PUSH_ERROR("Failed to read double2 array.");
          return false;
        }

        DCOUT("double2[] = " << value::print_array_snipped(v));

        value->Set(v);
        return true;
      } else {
        CHECK_MEMORY_USAGE(sizeof(value::double2));
        value::double2 v;
        if (!_sr->read(sizeof(value::double2), sizeof(value::double2),
                       reinterpret_cast<uint8_t *>(&v))) {
          PUSH_ERROR("Failed to read double2 data.");
          return false;
        }

        DCOUT("double2 = " << v);

        value->Set(v);
        return true;
      }
    }

    case crate::CrateDataTypeId::CRATE_DATA_TYPE_VEC2F: {
      COMPRESS_UNSUPPORTED_CHECK(dty)

      if (rep.IsArray()) {
        std::vector<value::float2> v;
        if (rep.GetPayload() == 0) { // empty array
          value->Set(v);
          return true;
        }

        uint64_t n{0};
        if (VERSION_LESS_THAN_0_8_0(_version)) {
          uint32_t shapesize; // not used
          if (!_sr->read4(&shapesize)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
          uint32_t _n;
          if (!_sr->read4(&_n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
          n = _n;
        } else {
          if (!_sr->read8(&n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
        }

        if (n > _config.maxArrayElements) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Array size {} too large. maxArrayElements is set to {}. Please increase maxArrayElements in CrateReaderConfig.", n, _config.maxArrayElements));
        }

        if (n == 0) {
          value->Set(v);
          return true;
        }

        CHECK_MEMORY_USAGE(n * sizeof(value::float2));

        v.resize(static_cast<size_t>(n));
        if (!_sr->read(size_t(n) * sizeof(value::float2),
                       size_t(n) * sizeof(value::float2),
                       reinterpret_cast<uint8_t *>(v.data()))) {
          PUSH_ERROR("Failed to read float2 array.");
          return false;
        }

        DCOUT("float2[] = " << value::print_array_snipped(v));

        value->Set(v);
        return true;
      } else {
        CHECK_MEMORY_USAGE(sizeof(value::float2));
        value::float2 v;
        if (!_sr->read(sizeof(value::float2), sizeof(value::float2),
                       reinterpret_cast<uint8_t *>(&v))) {
          PUSH_ERROR("Failed to read float2 data.");
          return false;
        }

        DCOUT("float2 = " << v);

        value->Set(v);
        return true;
      }
    }

    case crate::CrateDataTypeId::CRATE_DATA_TYPE_VEC2H: {
      COMPRESS_UNSUPPORTED_CHECK(dty)

      if (rep.IsArray()) {
        std::vector<value::half2> v;
        if (rep.GetPayload() == 0) { // empty array
          value->Set(v);
          return true;
        }
        uint64_t n{0};
        if (VERSION_LESS_THAN_0_8_0(_version)) {
          uint32_t shapesize; // not used
          if (!_sr->read4(&shapesize)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
          uint32_t _n;
          if (!_sr->read4(&_n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
          n = _n;
        } else {
          if (!_sr->read8(&n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
        }

        if (n > _config.maxArrayElements) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Array size {} too large. maxArrayElements is set to {}. Please increase maxArrayElements in CrateReaderConfig.", n, _config.maxArrayElements));
        }

        CHECK_MEMORY_USAGE(n * sizeof(value::half2));

        v.resize(static_cast<size_t>(n));
        if (!_sr->read(size_t(n) * sizeof(value::half2),
                       size_t(n) * sizeof(value::half2),
                       reinterpret_cast<uint8_t *>(v.data()))) {
          PUSH_ERROR("Failed to read half2 array.");
          return false;
        }

        DCOUT("half2[] = " << value::print_array_snipped(v));
        value->Set(v);

      } else {
        CHECK_MEMORY_USAGE(sizeof(value::half2));
        value::half2 v;
        if (!_sr->read(sizeof(value::half2), sizeof(value::half2),
                       reinterpret_cast<uint8_t *>(&v))) {
          PUSH_ERROR("Failed to read half2");
          return false;
        }

        DCOUT("half2 = " << v);

        value->Set(v);
      }

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_VEC2I: {
      COMPRESS_UNSUPPORTED_CHECK(dty)

      if (rep.IsArray()) {
        std::vector<value::int2> v;
        if (rep.GetPayload() == 0) { // empty array
          value->Set(v);
          return true;
        }

        uint64_t n{0};
        if (VERSION_LESS_THAN_0_8_0(_version)) {
          uint32_t shapesize; // not used
          if (!_sr->read4(&shapesize)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
          uint32_t _n;
          if (!_sr->read4(&_n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
          n = _n;
        } else {
          if (!_sr->read8(&n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
        }

        if (n > _config.maxArrayElements) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Array size {} too large. maxArrayElements is set to {}. Please increase maxArrayElements in CrateReaderConfig.", n, _config.maxArrayElements));
        }

        CHECK_MEMORY_USAGE(n * sizeof(value::int2));

        v.resize(static_cast<size_t>(n));
        if (!_sr->read(size_t(n) * sizeof(value::int2),
                       size_t(n) * sizeof(value::int2),
                       reinterpret_cast<uint8_t *>(v.data()))) {
          PUSH_ERROR("Failed to read int2 array.");
          return false;
        }

        DCOUT("int2[] = " << value::print_array_snipped(v));
        value->Set(v);

      } else {
        CHECK_MEMORY_USAGE(sizeof(value::int2));
        value::int2 v;
        if (!_sr->read(sizeof(value::int2), sizeof(value::int2),
                       reinterpret_cast<uint8_t *>(&v))) {
          PUSH_ERROR("Failed to read int2");
          return false;
        }

        DCOUT("int2 = " << v);

        value->Set(v);
      }

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_VEC3D: {
      COMPRESS_UNSUPPORTED_CHECK(dty)

      if (rep.IsArray()) {
        std::vector<value::double3> v;
        if (rep.GetPayload() == 0) { // empty array
          value->Set(v);
          return true;
        }

        uint64_t n{0};
        if (VERSION_LESS_THAN_0_8_0(_version)) {
          uint32_t shapesize; // not used
          if (!_sr->read4(&shapesize)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
          uint32_t _n;
          if (!_sr->read4(&_n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
          n = _n;
        } else {
          if (!_sr->read8(&n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
        }

        if (n > _config.maxArrayElements) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Array size {} too large. maxArrayElements is set to {}. Please increase maxArrayElements in CrateReaderConfig.", n, _config.maxArrayElements));
        }

        CHECK_MEMORY_USAGE(n * sizeof(value::double3));

        v.resize(static_cast<size_t>(n));
        if (!_sr->read(size_t(n) * sizeof(value::double3),
                       size_t(n) * sizeof(value::double3),
                       reinterpret_cast<uint8_t *>(v.data()))) {
          PUSH_ERROR("Failed to read double3 array.");
          return false;
        }

        DCOUT("double3[] = " << value::print_array_snipped(v));
        value->Set(v);

      } else {
        CHECK_MEMORY_USAGE(sizeof(value::double3));
        value::double3 v;
        if (!_sr->read(sizeof(value::double3), sizeof(value::double3),
                       reinterpret_cast<uint8_t *>(&v))) {
          PUSH_ERROR("Failed to read double3");
          return false;
        }

        DCOUT("double3 = " << v);

        value->Set(v);
      }

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_VEC3F: {
      COMPRESS_UNSUPPORTED_CHECK(dty)

      if (rep.IsArray()) {
        std::vector<value::float3> v;
        if (rep.GetPayload() == 0) { // empty array
          value->Set(v);
          return true;
        }

        uint64_t n{0};
        if (VERSION_LESS_THAN_0_8_0(_version)) {
          uint32_t shapesize; // not used
          if (!_sr->read4(&shapesize)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
          uint32_t _n;
          if (!_sr->read4(&_n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
          n = _n;
        } else {
          if (!_sr->read8(&n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
        }

        if (n > _config.maxArrayElements) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Array size {} too large. maxArrayElements is set to {}. Please increase maxArrayElements in CrateReaderConfig.", n, _config.maxArrayElements));
        }

        CHECK_MEMORY_USAGE(n * sizeof(value::float3));

        v.resize(static_cast<size_t>(n));
        if (!_sr->read(size_t(n) * sizeof(value::float3),
                       size_t(n) * sizeof(value::float3),
                       reinterpret_cast<uint8_t *>(v.data()))) {
          PUSH_ERROR("Failed to read float3 array.");
          return false;
        }

        DCOUT("float3f[] = " << value::print_array_snipped(v));
        value->Set(v);

      } else {
        CHECK_MEMORY_USAGE(sizeof(value::float3));
        value::float3 v;
        if (!_sr->read(sizeof(value::float3), sizeof(value::float3),
                       reinterpret_cast<uint8_t *>(&v))) {
          PUSH_ERROR("Failed to read float3");
          return false;
        }

        DCOUT("float3 = " << v);

        value->Set(v);
      }

      return true;
    }

    case crate::CrateDataTypeId::CRATE_DATA_TYPE_VEC3H: {
      COMPRESS_UNSUPPORTED_CHECK(dty)

      if (rep.IsArray()) {
        std::vector<value::half3> v;
        if (rep.GetPayload() == 0) { // empty array
          value->Set(v);
          return true;
        }
        uint64_t n{0};
        if (VERSION_LESS_THAN_0_8_0(_version)) {
          uint32_t shapesize; // not used
          if (!_sr->read4(&shapesize)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
          uint32_t _n;
          if (!_sr->read4(&_n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
          n = _n;
        } else {
          if (!_sr->read8(&n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
        }

        if (n > _config.maxArrayElements) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Array size {} too large. maxArrayElements is set to {}. Please increase maxArrayElements in CrateReaderConfig.", n, _config.maxArrayElements));
        }

        CHECK_MEMORY_USAGE(n * sizeof(value::half3));

        v.resize(static_cast<size_t>(n));
        if (!_sr->read(size_t(n) * sizeof(value::half3),
                       size_t(n) * sizeof(value::half3),
                       reinterpret_cast<uint8_t *>(v.data()))) {
          PUSH_ERROR("Failed to read half3 array.");
          return false;
        }

        DCOUT("half3[] = " << value::print_array_snipped(v));
        value->Set(v);

      } else {
        CHECK_MEMORY_USAGE(sizeof(value::half3));
        value::half3 v;
        if (!_sr->read(sizeof(value::half3), sizeof(value::half3),
                       reinterpret_cast<uint8_t *>(&v))) {
          PUSH_ERROR("Failed to read half3");
          return false;
        }

        DCOUT("half3 = " << v);

        value->Set(v);
      }

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_VEC3I: {
      COMPRESS_UNSUPPORTED_CHECK(dty)

      if (rep.IsArray()) {
        std::vector<value::int3> v;
        if (rep.GetPayload() == 0) { // empty array
          value->Set(v);
          return true;
        }
        uint64_t n{0};
        if (VERSION_LESS_THAN_0_8_0(_version)) {
      uint32_t shapesize; // not used
      if (!_sr->read4(&shapesize)) {
        PUSH_ERROR("Failed to read the number of array elements.");
        return false;
      }
          uint32_t _n;
          if (!_sr->read4(&_n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
          n = _n;
        } else {
          if (!_sr->read8(&n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
        }

        if (n > _config.maxArrayElements) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Array size {} too large. maxArrayElements is set to {}. Please increase maxArrayElements in CrateReaderConfig.", n, _config.maxArrayElements));
        }

        CHECK_MEMORY_USAGE(n * sizeof(value::int3));

        v.resize(static_cast<size_t>(n));
        if (!_sr->read(size_t(n) * sizeof(value::int3),
                       size_t(n) * sizeof(value::int3),
                       reinterpret_cast<uint8_t *>(v.data()))) {
          PUSH_ERROR("Failed to read int3 array.");
          return false;
        }

        DCOUT("int3[] = " << value::print_array_snipped(v));
        value->Set(v);

      } else {
        CHECK_MEMORY_USAGE(sizeof(value::int3));
        value::int3 v;
        if (!_sr->read(sizeof(value::int3), sizeof(value::int3),
                       reinterpret_cast<uint8_t *>(&v))) {
          PUSH_ERROR("Failed to read int3");
          return false;
        }

        DCOUT("int3 = " << v);

        value->Set(v);
      }

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_VEC4D: {
      COMPRESS_UNSUPPORTED_CHECK(dty)

      if (rep.IsArray()) {
        std::vector<value::double4> v;
        if (rep.GetPayload() == 0) { // empty array
          value->Set(v);
          return true;
        }

        uint64_t n{0};
        if (VERSION_LESS_THAN_0_8_0(_version)) {
      uint32_t shapesize; // not used
      if (!_sr->read4(&shapesize)) {
        PUSH_ERROR("Failed to read the number of array elements.");
        return false;
      }
          uint32_t _n;
          if (!_sr->read4(&_n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
          n = _n;
        } else {
          if (!_sr->read8(&n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
        }

        if (n > _config.maxArrayElements) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Array size {} too large. maxArrayElements is set to {}. Please increase maxArrayElements in CrateReaderConfig.", n, _config.maxArrayElements));
        }

        CHECK_MEMORY_USAGE(n * sizeof(value::double4));

        v.resize(static_cast<size_t>(n));
        if (!_sr->read(size_t(n) * sizeof(value::double4),
                       size_t(n) * sizeof(value::double4),
                       reinterpret_cast<uint8_t *>(v.data()))) {
          PUSH_ERROR("Failed to read double4 array.");
          return false;
        }

        DCOUT("double4[] = " << value::print_array_snipped(v));
        value->Set(v);

      } else {
        CHECK_MEMORY_USAGE(sizeof(value::double4));
        value::double4 v;
        if (!_sr->read(sizeof(value::double4), sizeof(value::double4),
                       reinterpret_cast<uint8_t *>(&v))) {
          PUSH_ERROR("Failed to read double4");
          return false;
        }

        DCOUT("double4 = " << v);

        value->Set(v);
      }

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_VEC4F: {
      COMPRESS_UNSUPPORTED_CHECK(dty)

      if (rep.IsArray()) {
        std::vector<value::float4> v;
        if (rep.GetPayload() == 0) { // empty array
          value->Set(v);
          return true;
        }
        uint64_t n{0};
        if (VERSION_LESS_THAN_0_8_0(_version)) {
      uint32_t shapesize; // not used
      if (!_sr->read4(&shapesize)) {
        PUSH_ERROR("Failed to read the number of array elements.");
        return false;
      }
          uint32_t _n;
          if (!_sr->read4(&_n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
          n = _n;
        } else {
          if (!_sr->read8(&n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
        }

        if (n > _config.maxArrayElements) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Array size {} too large. maxArrayElements is set to {}. Please increase maxArrayElements in CrateReaderConfig.", n, _config.maxArrayElements));
        }

        CHECK_MEMORY_USAGE(n * sizeof(value::float4));

        v.resize(static_cast<size_t>(n));
        if (!_sr->read(size_t(n) * sizeof(value::float4),
                       size_t(n) * sizeof(value::float4),
                       reinterpret_cast<uint8_t *>(v.data()))) {
          PUSH_ERROR("Failed to read float4 array.");
          return false;
        }

        DCOUT("float4[] = " << value::print_array_snipped(v));
        value->Set(v);

      } else {
        CHECK_MEMORY_USAGE(sizeof(value::float4));
        value::float4 v;
        if (!_sr->read(sizeof(value::float4), sizeof(value::float4),
                       reinterpret_cast<uint8_t *>(&v))) {
          PUSH_ERROR("Failed to read float4");
          return false;
        }

        DCOUT("float4 = " << v);

        value->Set(v);
      }

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_VEC4H: {
      COMPRESS_UNSUPPORTED_CHECK(dty)

      if (rep.IsArray()) {
        std::vector<value::half4> v;
        if (rep.GetPayload() == 0) { // empty array
          value->Set(v);
          return true;
        }
        uint64_t n{0};
        if (VERSION_LESS_THAN_0_8_0(_version)) {
      uint32_t shapesize; // not used
      if (!_sr->read4(&shapesize)) {
        PUSH_ERROR("Failed to read the number of array elements.");
        return false;
      }
          uint32_t _n;
          if (!_sr->read4(&_n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
          n = _n;
        } else {
          if (!_sr->read8(&n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
        }

        if (n > _config.maxArrayElements) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Array size {} too large. maxArrayElements is set to {}. Please increase maxArrayElements in CrateReaderConfig.", n, _config.maxArrayElements));
        }

        CHECK_MEMORY_USAGE(n * sizeof(value::half4));

        v.resize(static_cast<size_t>(n));
        if (!_sr->read(size_t(n) * sizeof(value::half4),
                       size_t(n) * sizeof(value::half4),
                       reinterpret_cast<uint8_t *>(v.data()))) {
          PUSH_ERROR("Failed to read half4 array.");
          return false;
        }

        DCOUT("half4[] = " << value::print_array_snipped(v));
        value->Set(v);

      } else {
        CHECK_MEMORY_USAGE(sizeof(value::half4));
        value::half4 v;
        if (!_sr->read(sizeof(value::half4), sizeof(value::half4),
                       reinterpret_cast<uint8_t *>(&v))) {
          PUSH_ERROR("Failed to read half4");
          return false;
        }

        DCOUT("half4 = " << v);

        value->Set(v);
      }

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_VEC4I: {
      COMPRESS_UNSUPPORTED_CHECK(dty)

      if (rep.IsArray()) {
        std::vector<value::int4> v;
        if (rep.GetPayload() == 0) { // empty array
          value->Set(v);
          return true;
        }
        uint64_t n{0};
        if (VERSION_LESS_THAN_0_8_0(_version)) {
      uint32_t shapesize; // not used
      if (!_sr->read4(&shapesize)) {
        PUSH_ERROR("Failed to read the number of array elements.");
        return false;
      }
          uint32_t _n;
          if (!_sr->read4(&_n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
          n = _n;
        } else {
          if (!_sr->read8(&n)) {
            PUSH_ERROR("Failed to read the number of array elements.");
            return false;
          }
        }

        if (n > _config.maxArrayElements) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Array size {} too large. maxArrayElements is set to {}. Please increase maxArrayElements in CrateReaderConfig.", n, _config.maxArrayElements));
        }

        CHECK_MEMORY_USAGE(n * sizeof(value::int4));

        v.resize(static_cast<size_t>(n));
        if (!_sr->read(size_t(n) * sizeof(value::int4),
                       size_t(n) * sizeof(value::int4),
                       reinterpret_cast<uint8_t *>(v.data()))) {
          PUSH_ERROR("Failed to read int4 array.");
          return false;
        }

        DCOUT("int4[] = " << value::print_array_snipped(v));
        value->Set(v);

      } else {
        CHECK_MEMORY_USAGE(sizeof(value::int4));
        value::int4 v;
        if (!_sr->read(sizeof(value::int4), sizeof(value::int4),
                       reinterpret_cast<uint8_t *>(&v))) {
          PUSH_ERROR("Failed to read int4");
          return false;
        }

        DCOUT("int4 = " << v);

        value->Set(v);
      }

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_DICTIONARY: {
      COMPRESS_UNSUPPORTED_CHECK(dty)
      ARRAY_UNSUPPORTED_CHECK(dty)

      //crate::CrateValue::Dictionary dict;
      CustomDataType dict;

      if (!ReadCustomData(&dict)) {
        _err += "Failed to read Dictionary value\n";
        return false;
      }

      DCOUT("Dict. nelems = " << dict.size());

      value->Set(dict);

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_TOKEN_LIST_OP: {
      ListOp<value::token> lst;

      if (!ReadTokenListOp(&lst)) {
        PUSH_ERROR("Failed to read TokenListOp data");
        return false;
      }

      value->Set(lst);
      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_PATH_LIST_OP: {
      COMPRESS_UNSUPPORTED_CHECK(dty)

      // SdfListOp<class SdfPath>
      // => underliying storage is the array of ListOp[PathIndex]
      ListOp<Path> lst;

      if (!ReadPathListOp(&lst)) {
        PUSH_ERROR("Failed to read PathListOp data.");
        return false;
      }

      value->Set(lst);

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_STRING_LIST_OP: {
      ListOp<std::string> lst;

      if (!ReadStringListOp(&lst)) {
        PUSH_ERROR("Failed to read StringListOp data");
        return false;
      }

      value->Set(lst);
      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_PATH_VECTOR: {
      COMPRESS_UNSUPPORTED_CHECK(dty)

      std::vector<Path> v;
      if (!ReadPathArray(&v)) {
        _err += "Failed to read PathVector value\n";
        return false;
      }

      DCOUT("PathVector = " << to_string(v));

      value->Set(v);

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_TOKEN_VECTOR: {
      COMPRESS_UNSUPPORTED_CHECK(dty)
      // std::vector<Index>
      uint64_t n{0};
      if (!_sr->read8(&n)) {
        PUSH_ERROR("Failed to read the number of array elements.");
        return false;
      }

      if (n > _config.maxArrayElements) {
        PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Array size {} too large. maxArrayElements is set to {}. Please increase maxArrayElements in CrateReaderConfig.", n, _config.maxArrayElements));
      }

      CHECK_MEMORY_USAGE(n * sizeof(crate::Index));

      std::vector<crate::Index> indices(static_cast<size_t>(n));
      if (!_sr->read(static_cast<size_t>(n) * sizeof(crate::Index),
                     static_cast<size_t>(n) * sizeof(crate::Index),
                     reinterpret_cast<uint8_t *>(indices.data()))) {
        PUSH_ERROR("Failed to read TokenVector value.");
        return false;
      }

      DCOUT("TokenVector(index) = " << indices);

      std::vector<value::token> tokens(indices.size());
      for (size_t i = 0; i < indices.size(); i++) {
        if (auto tokv = GetToken(indices[i])) {
          tokens[i] = tokv.value();
        } else {
          return false;
        }
      }

      DCOUT("TokenVector = " << tokens);

      value->Set(tokens);

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_TIME_SAMPLES: {
      COMPRESS_UNSUPPORTED_CHECK(dty)

      value::TimeSamples ts;
      if (!ReadTimeSamples(&ts)) {
        PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read TimeSamples data");
      }

      value->Set(ts);

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_DOUBLE_VECTOR: {
      std::vector<double> v;
      if (!ReadDoubleVector(&v)) {
        _err += "Failed to read DoubleVector value\n";
        return false;
      }

      DCOUT("DoubleArray = " << v);

      value->Set(v);

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_STRING_VECTOR: {
      COMPRESS_UNSUPPORTED_CHECK(dty)

      std::vector<std::string> v;
      if (!ReadStringArray(&v)) {
        PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read StringVector value");
      }

      DCOUT("StringArray = " << v);

      value->Set(v);

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_VARIANT_SELECTION_MAP: {
      COMPRESS_UNSUPPORTED_CHECK(dty)

      VariantSelectionMap m;
      if (!ReadVariantSelectionMap(&m)) {
        PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read VariantSelectionMap value");
      }

      DCOUT("VariantSelectionMap = " << print_variantSelectionMap(m, 0));

      value->Set(m);

      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_LAYER_OFFSET_VECTOR: {
      COMPRESS_UNSUPPORTED_CHECK(dty)
      // LayerOffset[]

      std::vector<LayerOffset> v;
      if (!ReadLayerOffsetArray(&v)) {
        PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read LayerOffsetVector value");
      }

      DCOUT("LayerOffsetVector = " << v);

      value->Set(v);

      return true;

    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_PAYLOAD: {
      COMPRESS_UNSUPPORTED_CHECK(dty)

      // Payload
      Payload v;
      if (!ReadPayload(&v)) {
        PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read Payload value");
      }

      DCOUT("Payload = " << v);

      value->Set(v);
      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_PAYLOAD_LIST_OP: {
      ListOp<Payload> lst;

      if (!ReadListOp(&lst)) {
        PUSH_ERROR("Failed to read PayloadListOp data");
        return false;
      }

      value->Set(lst);
      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_REFERENCE_LIST_OP: {
      ListOp<Reference> lst;

      if (!ReadListOp(&lst)) {
        PUSH_ERROR("Failed to read ReferenceListOp data");
        return false;
      }

      value->Set(lst);
      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_INT_LIST_OP: {
      ListOp<int32_t> lst;

      if (!ReadListOp(&lst)) {
        PUSH_ERROR("Failed to read IntListOp data");
        return false;
      }

      value->Set(lst);
      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_INT64_LIST_OP: {
      ListOp<int64_t> lst;

      if (!ReadListOp(&lst)) {
        PUSH_ERROR("Failed to read Int64ListOp data");
        return false;
      }

      value->Set(lst);
      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_UINT_LIST_OP: {
      ListOp<uint32_t> lst;

      if (!ReadListOp(&lst)) {
        PUSH_ERROR("Failed to read UIntListOp data");
        return false;
      }

      value->Set(lst);
      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_UINT64_LIST_OP: {
      ListOp<uint64_t> lst;

      if (!ReadListOp(&lst)) {
        PUSH_ERROR("Failed to read UInt64ListOp data");
        return false;
      }

      value->Set(lst);
      return true;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_VALUE_BLOCK: {
      PUSH_ERROR(
          "ValueBlock must be defined in Inlined ValueRep.");
      return false;
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_VALUE: {

      crate::ValueRep local_rep{0};
      if (!ReadValueRep(&local_rep)) {
        PUSH_ERROR(
            "Failed to read ValueRep for VALUE type.");
        return false;
      }

      if (unpackRecursionGuard.size() > _config.maxValueRecursion) {
        // To many recursive stacks. We report error
        PUSH_ERROR(
            "Too many recursion when decoding generic VALUE data.");
        return false;
      }

      // TODO: use crate::ValueRep for set container type.
      if (unpackRecursionGuard.count(local_rep.GetData())) {
        // Recursion detected.
        PUSH_ERROR(
            "Corrupted Value data detected.");
        return false;
      } else {
        crate::CrateValue local_val;
        bool ret = UnpackValueRep(local_rep, &local_val);
        if (!ret) {
          return false;
        }

        (*value) = local_val;

        unpackRecursionGuard.erase(local_rep.GetData());
        return true;
      }
    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_UNREGISTERED_VALUE: {
      COMPRESS_UNSUPPORTED_CHECK(dty)
      ARRAY_UNSUPPORTED_CHECK(dty)

      // 8byte for the offset for recursive value. See RecursiveRead() in
      // https://github.com/PixarAnimationStudios/USD/blob/release/pxr/usd/usd/crateFile.cpp for details.
      int64_t local_offset{0};
      if (!_sr->read8(&local_offset)) {
        PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read the offset for value in Dictionary.");
        return false;
      }

      DCOUT("UnregisteredValue  offset = " << local_offset);
      DCOUT("tell = " << _sr->tell());

      // -8 to compensate sizeof(offset)
      if (!_sr->seek_from_current(local_offset - 8)) {
        PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to seek to UNREGISTERD_VALUE content. Invalid offset value: " +
                std::to_string(local_offset));
      }

      uint64_t saved_position = _sr->tell();

      crate::ValueRep local_rep{0};
      if (!ReadValueRep(&local_rep)) {
        PUSH_ERROR(
            "Failed to read ValueRep for UNREGISTERED_VALUE type.");
        return false;
      }

      auto local_tyRet = crate::GetCrateDataType(local_rep.GetType());
      if (!local_tyRet) {
        PUSH_ERROR(local_tyRet.error());
        return false;
      }

      const auto local_dty = local_tyRet.value();

      // Should be STRING or DICTIONARY for UNREGISTERED_VALUE.
      if (local_dty.dtype_id == crate::CrateDataTypeId::CRATE_DATA_TYPE_STRING) {
        COMPRESS_UNSUPPORTED_CHECK(local_dty)
        ARRAY_UNSUPPORTED_CHECK(local_dty)

        if (local_rep.IsInlined()) {
          uint32_t local_d = (local_rep.GetPayload() & ((1ull << (sizeof(uint32_t) * 8)) - 1));
          if (auto v = GetStringToken(crate::Index(local_d))) {
            std::string str = v.value().str();

            DCOUT("UNREGISTERED_VALUE.string = " << str);

            // NOTE: string may contain double-quotes.
            // We remove it at here, but it'd be better not to do it.
            std::string unquoted = unwrap(str);
            value->Set(unquoted);

            if (!_sr->seek_set(saved_position)) {
              PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to set seek.");
            }
            return true;
          } else {
            PUSH_ERROR("Failed to decode String.");
            return false;
          }
        } else {
          PUSH_ERROR("String value must be inlined.");
          return false;
        }

      } else if (local_dty.dtype_id == crate::CrateDataTypeId::CRATE_DATA_TYPE_DICTIONARY) {
        COMPRESS_UNSUPPORTED_CHECK(local_dty)
        ARRAY_UNSUPPORTED_CHECK(local_dty)

        CustomDataType dict;

        if (local_rep.IsInlined()) {
          // empty dict
        }  else{
          if (!ReadCustomData(&dict)) {
            _err += "Failed to read Dictionary value\n";
            return false;
          }
        }
        value->Set(dict);
        if (!_sr->seek_set(saved_position)) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to set seek.");
        }
        return true;

      } else {
        PUSH_ERROR_AND_RETURN(fmt::format("UNREGISTERD_VALUE type must be string or dictionary, but got other data type: {}(id {}).", GetCrateDataTypeName(local_dty.dtype_id), local_rep.GetType()));
      }

    }
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_UNREGISTERED_VALUE_LIST_OP:
    case crate::CrateDataTypeId::CRATE_DATA_TYPE_TIME_CODE: {
      PUSH_ERROR(
          "Invalid data type(or maybe not supported in TinyUSDZ yet) for "
          "Uninlined value: " +
          crate::GetCrateDataTypeName(dty.dtype_id));
      return false;
    }
  }

#undef TODO_IMPLEMENT
#undef COMPRESS_UNSUPPORTED_CHECK
#undef NON_ARRAY_UNSUPPORTED_CHECK

  // Never should reach here.
  return false;
}

#if defined(TINYUSDZ_CRATE_USE_FOR_BASED_PATH_INDEX_DECODER)
bool CrateReader::BuildDecompressedPathsImpl(
    BuildDecompressedPathsArg *arg) {

  if (!arg) {
    return false;
  }

  Path parentPath = arg->parentPath;
  if (!arg->pathIndexes) {
    return false;
  }
  if (!arg->elementTokenIndexes) {
    return false;
  }
  if (!arg->jumps) {
    return false;
  }
  if (!arg->visit_table) {
    return false;
  }
  auto &pathIndexes = *arg->pathIndexes;
  auto &elementTokenIndexes = *arg->elementTokenIndexes;
  auto &jumps = *arg->jumps;
  auto &visit_table = *arg->visit_table;

  auto rootPath = Path::make_root_path();

  const size_t maxIter = _config.maxPathIndicesDecodeIteration;

  std::stack<size_t> startIndexStack;
  std::stack<size_t> endIndexStack;
  std::stack<Path> parentPathStack;

  size_t nIter = 0;

  size_t startIndex = arg->startIndex;
  size_t endIndex = arg->endIndex;

  while (nIter < maxIter) {

    DCOUT("startIndex = " << startIndex << ", endIdx = " << endIndex);

    for (size_t thisIndex = startIndex; thisIndex < (endIndex + 1); thisIndex++) {
      //auto thisIndex = curIndex++;
      DCOUT("thisIndex = " << thisIndex << ", pathIndexes.size = " << pathIndexes.size());
      if (parentPath.is_empty()) {
        // root node.
        // Assume single root node in the scene.
        DCOUT("paths[" << pathIndexes[thisIndex]
                       << "] is parent. name = " << parentPath.full_path_name());
        parentPath = rootPath;

        if (thisIndex >= pathIndexes.size()) {
          PUSH_ERROR("Index exceeds pathIndexes.size()");
          return false;
        }

        size_t idx = pathIndexes[thisIndex];
        if (idx >= _paths.size()) {
          PUSH_ERROR("Index is out-of-range");
          return false;
        }

        if (idx < visit_table.size()) {
          if (visit_table[idx]) {
            PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Circular referencing of Path index {}(thisIndex {}) detected. Invalid Paths data.", idx, thisIndex));
          }
        }

        _paths[idx] = parentPath;
        visit_table[idx] = true;
      } else {
        if (thisIndex >= elementTokenIndexes.size()) {
          PUSH_ERROR("Index exceeds elementTokenIndexes.size()");
          return false;
        }
        int32_t _tokenIndex = elementTokenIndexes[thisIndex];
        DCOUT("elementTokenIndex = " << _tokenIndex);
        bool isPrimPropertyPath = _tokenIndex < 0;
        // ~0 returns -2147483648, so cast to uint32
        uint32_t tokenIndex = uint32_t(isPrimPropertyPath ? -_tokenIndex : _tokenIndex);

        DCOUT("tokenIndex = " << tokenIndex << ", _tokens.size = " << _tokens.size());
        if (tokenIndex >= _tokens.size()) {
          PUSH_ERROR("Invalid tokenIndex in BuildDecompressedPathsImpl.");
          return false;
        }
        auto const &elemToken = _tokens[size_t(tokenIndex)];
        DCOUT("elemToken = " << elemToken);
        DCOUT("[" << pathIndexes[thisIndex] << "].append = " << elemToken);

        size_t idx = pathIndexes[thisIndex];
        if (idx >= _paths.size()) {
          PUSH_ERROR("Index is out-of-range");
          return false;
        }

        if (idx >= _elemPaths.size()) {
          PUSH_ERROR("Index is out-of-range");
          return false;
        }

        if (idx < visit_table.size()) {
          if (visit_table[idx]) {
            PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Circular referencing of Path index {}(thisIndex {}) detected. Invalid Paths data.", idx, thisIndex));
          }
        }

        // Reconstruct full path
        _paths[idx] =
            isPrimPropertyPath ? parentPath.AppendProperty(elemToken.str())
                               : parentPath.AppendElement(elemToken.str()); // prim, variantSelection, etc.

        // also set leaf path for 'primChildren' check
        _elemPaths[idx] = Path(elemToken.str(), "");
        //_paths[pathIndexes[thisIndex]].SetLocalPart(elemToken.str());

        visit_table[idx] = true;
      }

      // If we have either a child or a sibling but not both, then just
      // continue to the neighbor.  If we have both then spawn a task for the
      // sibling and do the child ourself.  We think that our path trees tend
      // to be broader more often than deep.

      if (thisIndex >= jumps.size()) {
        PUSH_ERROR("Index is out-of-range");
        return false;
      }

      bool hasChild = (jumps[thisIndex] > 0) || (jumps[thisIndex] == -1);
      bool hasSibling = (jumps[thisIndex] >= 0);
      DCOUT("hasChild = " << hasChild << ", hasSibling = " << hasSibling);

      if (hasChild) {
        if (hasSibling) {
          // NOTE(syoyo): This recursive call can be parallelized
          auto siblingIndex = thisIndex + size_t(jumps[thisIndex]);

          if (siblingIndex >= jumps.size()) {
            PUSH_ERROR_AND_RETURN("jump index corrupted.");
          }

          // Find subtree end.
          size_t subtreeStartIdx = siblingIndex;
          size_t subtreeIdx = subtreeStartIdx;

          for (; subtreeIdx < jumps.size(); subtreeIdx++) {

            bool has_child = (jumps[subtreeIdx] > 0) || (jumps[subtreeIdx] == -1);
            bool has_sibling = (jumps[subtreeIdx] >= 0);

            if (has_child || has_sibling) {
              continue;
            }
            break;
          }

          size_t subtreeEndIdx = subtreeIdx;
          if (subtreeEndIdx >= jumps.size()) {
            // Guess corrupted.
            PUSH_ERROR_AND_RETURN("jump indices seems corrupted.");
          }

          DCOUT("subtree startIdx " << subtreeStartIdx << ", subtree endIndex " << subtreeEndIdx);

          if (subtreeEndIdx >= subtreeStartIdx) {

            // index range after traversing subtree
            if (jumps[thisIndex] > 1) {

                // Setup stacks to resume loop from [Cont.]
                startIndexStack.push(thisIndex+1);
                // jumps should be always positive, so no siblingIndex < thisIndex
                endIndexStack.push(siblingIndex-1); // endIndex is inclusive so subtract 1.

                {
                  size_t idx = pathIndexes[thisIndex];
                  if (idx >= _paths.size()) {
                    PUSH_ERROR("Index is out-of-range");
                    return false;
                  }

                  parentPathStack.push(_paths[idx]);
                }
            }

            startIndexStack.push(subtreeStartIdx);
            endIndexStack.push(subtreeEndIdx);

            parentPathStack.push(parentPath);
            DCOUT("stack size: " << startIndexStack.size());

            nIter++;

            break; // goto `(A)`
          }

        }

        // [Cont.]
        size_t idx = pathIndexes[thisIndex];
        if (idx >= _paths.size()) {
          PUSH_ERROR("Index is out-of-range");
          return false;
        }

        parentPath = _paths[idx];

      }
    }

    // (A)

    if (startIndexStack.empty()) {
      break; // end traversal
    }

    startIndex = startIndexStack.top();
    startIndexStack.pop();

    endIndex = endIndexStack.top();
    endIndexStack.pop();

    parentPath = parentPathStack.top();
    parentPathStack.pop();

    nIter++;
  }

  if (nIter >= maxIter) {
    PUSH_ERROR_AND_RETURN("PathIndex tree Too deep.");
  }

  return true;
}
#else
bool CrateReader::BuildDecompressedPathsImpl(
    std::vector<uint32_t> const &pathIndexes,
    std::vector<int32_t> const &elementTokenIndexes,
    std::vector<int32_t> const &jumps,
    std::vector<bool> &visit_table,
    size_t curIndex, const Path &_parentPath) {

  Path parentPath = _parentPath;

  bool hasChild = false, hasSibling = false;
  do {
    auto thisIndex = curIndex++;
    DCOUT("thisIndex = " << thisIndex << ", pathIndexes.size = " << pathIndexes.size());
    if (parentPath.is_empty()) {
      // root node.
      // Assume single root node in the scene.
      DCOUT("paths[" << pathIndexes[thisIndex]
                     << "] is parent. name = " << parentPath.full_path_name());
      parentPath = Path::make_root_path();

      if (thisIndex >= pathIndexes.size()) {
        PUSH_ERROR("Index exceeds pathIndexes.size()");
        return false;
      }

      size_t idx = pathIndexes[thisIndex];
      if (idx >= _paths.size()) {
        PUSH_ERROR("Index is out-of-range");
        return false;
      }

      if (idx < visit_table.size()) {
        if (visit_table[idx]) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, "Circular referencing of Path index tree detected. Invalid Paths data.");
        }
      }

      _paths[idx] = parentPath;
      visit_table[idx] = true;
    } else {
      if (thisIndex >= elementTokenIndexes.size()) {
        PUSH_ERROR("Index exceeds elementTokenIndexes.size()");
        return false;
      }
      int32_t _tokenIndex = elementTokenIndexes[thisIndex];
      DCOUT("elementTokenIndex = " << _tokenIndex);
      bool isPrimPropertyPath = _tokenIndex < 0;
      // ~0 returns -2147483648, so cast to uint32
      uint32_t tokenIndex = uint32_t(isPrimPropertyPath ? -_tokenIndex : _tokenIndex);

      DCOUT("tokenIndex = " << tokenIndex << ", _tokens.size = " << _tokens.size());
      if (tokenIndex >= _tokens.size()) {
        PUSH_ERROR("Invalid tokenIndex in BuildDecompressedPathsImpl.");
        return false;
      }
      auto const &elemToken = _tokens[size_t(tokenIndex)];
      DCOUT("elemToken = " << elemToken);
      DCOUT("[" << pathIndexes[thisIndex] << "].append = " << elemToken);

      size_t idx = pathIndexes[thisIndex];
      if (idx >= _paths.size()) {
        PUSH_ERROR("Index is out-of-range");
        return false;
      }

      if (idx >= _elemPaths.size()) {
        PUSH_ERROR("Index is out-of-range");
        return false;
      }

      if (idx < visit_table.size()) {
        if (visit_table[idx]) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, "Circular referencing of Path index tree detected. Invalid Paths data.");
        }
      }

      // Reconstruct full path
      _paths[idx] =
          isPrimPropertyPath ? parentPath.AppendProperty(elemToken.str())
                             : parentPath.AppendElement(elemToken.str()); // prim, variantSelection, etc.

      // also set leaf path for 'primChildren' check
      _elemPaths[idx] = Path(elemToken.str(), "");
      //_paths[pathIndexes[thisIndex]].SetLocalPart(elemToken.str());

      visit_table[idx] = true;
    }

    // If we have either a child or a sibling but not both, then just
    // continue to the neighbor.  If we have both then spawn a task for the
    // sibling and do the child ourself.  We think that our path trees tend
    // to be broader more often than deep.

    if (thisIndex >= jumps.size()) {
      PUSH_ERROR("Index is out-of-range");
      return false;
    }

    hasChild = (jumps[thisIndex] > 0) || (jumps[thisIndex] == -1);
    hasSibling = (jumps[thisIndex] >= 0);
    DCOUT("hasChild = " << hasChild << ", hasSibling = " << hasSibling);

    DCOUT(fmt::format("hasChild {}, hasSibling {}", hasChild, hasSibling));

    if (hasChild) {
      if (hasSibling) {
        // NOTE(syoyo): This recursive call can be parallelized
        auto siblingIndex = thisIndex + size_t(jumps[thisIndex]);
        if (!BuildDecompressedPathsImpl(pathIndexes, elementTokenIndexes, jumps, visit_table,
                                        siblingIndex, parentPath)) {
          return false;
        }
      }

      size_t idx = pathIndexes[thisIndex];
      if (idx >= _paths.size()) {
        PUSH_ERROR("Index is out-of-range");
        return false;
      }

      // Have a child (may have also had a sibling). Reset parent path.
      parentPath = _paths[idx];
    }
    // If we had only a sibling, we just continue since the parent path is
    // unchanged and the next thing in the reader stream is the sibling's
    // header.
  } while (hasChild || hasSibling);

  return true;
}
#endif

#if defined(TINYUSDZ_CRATE_USE_FOR_BASED_PATH_INDEX_DECODER)
bool CrateReader::BuildNodeHierarchy(
    std::vector<uint32_t> const &pathIndexes,
    std::vector<int32_t> const &elementTokenIndexes,
    std::vector<int32_t> const &jumps,
    std::vector<bool> &visit_table, /* inout */
    size_t _curIndex,
    int64_t _parentNodeIndex) {

  (void)elementTokenIndexes;

  std::stack<int64_t> parentNodeIndexStack;
  std::stack<size_t> startIndexStack;
  std::stack<size_t> endIndexStack;

  size_t nIter = 0;
  const size_t maxIter = _config.maxPathIndicesDecodeIteration;

  size_t startIndex = _curIndex;
  size_t endIndex = pathIndexes.size() - 1;
  int64_t parentNodeIndex = _parentNodeIndex;

  // NOTE: Need to indirectly lookup index through pathIndexes[] when accessing
  // `_nodes`
  while (nIter < maxIter) {

    for (size_t thisIndex = startIndex; thisIndex < (endIndex + 1); thisIndex++) {
      if (parentNodeIndex == -1) {
        // root node.
        // Assume single root node in the scene.
        //assert(thisIndex == 0);
        if (thisIndex != 0) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, "TODO: Multiple root nodes.");
        }

        if (thisIndex >= pathIndexes.size()) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, "Index out-of-range.");
        }

        size_t pathIdx = pathIndexes[thisIndex];
        if (pathIdx >= _paths.size()) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, "PathIndex out-of-range.");
        }

        if (pathIdx >= _nodes.size()) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, "PathIndex out-of-range.");
        }

        if (pathIdx >= visit_table.size()) {
          // This should not be happan though
          PUSH_ERROR_AND_RETURN_TAG(kTag, "[InternalError] out-of-range.");
        }

        if (visit_table[pathIdx]) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, "Circular referencing detected. Invalid Prim tree representation.");
        }

        _nodes[pathIdx] = Node(parentNodeIndex, _paths[pathIdx]);
        visit_table[pathIdx] = true;

        parentNodeIndex = int64_t(thisIndex);

      } else {
        //if (parentNodeIndex >= int64_t(_nodes.size())) {
        //  PUSH_ERROR_AND_RETURN_TAG(kTag, "Parent Index out-of-range.");
        //}

        if (parentNodeIndex >= int64_t(pathIndexes.size())) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, "Parent Index out-of-range.");
        }

        if (thisIndex >= pathIndexes.size()) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, "Index out-of-range.");
        }

        DCOUT("Hierarchy. parent[" << pathIndexes[size_t(parentNodeIndex)]
                                   << "].add_child = " << pathIndexes[thisIndex]);

        size_t pathIdx = pathIndexes[thisIndex];
        if (pathIdx >= _paths.size()) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, "PathIndex out-of-range.");
        }

        if (pathIdx >= _nodes.size()) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, "PathIndex out-of-range.");
        }

        if (pathIdx >= visit_table.size()) {
          // This should not be happan though
          PUSH_ERROR_AND_RETURN_TAG(kTag, "[InternalError] out-of-range.");
        }

        if (visit_table[pathIdx]) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, "Circular referencing detected. Invalid Prim tree representation.");
        }


        // Ensure parent is not set yet.
        if (_nodes[pathIdx].GetParent() != -2) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, "???: Maybe corrupted path hierarchy?.");
        }

        Node node(parentNodeIndex, _paths[pathIdx]);
        _nodes[pathIdx] = node;

        visit_table[pathIdx] = true;

        if (pathIdx >= _elemPaths.size()) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, "PathIndex out-of-range.");
        }

        //std::string name = _paths[pathIndexes[thisIndex]].local_path_name();
        std::string name = _elemPaths[pathIdx].full_path_name();
        DCOUT("childName = " << name);

        size_t parentNodeIdx = size_t(parentNodeIndex);
        if (parentNodeIdx >= pathIndexes.size()) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, "ParentNodeIdx out-of-range.");
        }

        size_t parentPathIdx = pathIndexes[parentNodeIdx];
        if (parentPathIdx >= _nodes.size()) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, "PathIndex out-of-range.");
        }

        if (!_nodes[parentPathIdx].AddChildren(
            name, pathIdx)) {
          PUSH_ERROR_AND_RETURN_TAG(kTag, "Invalid path index.");
        }
      }

      if (thisIndex >= jumps.size()) {
        PUSH_ERROR_AND_RETURN_TAG(kTag, "Index is out-of-range");
      }

      bool hasChild = (jumps[thisIndex] > 0) || (jumps[thisIndex] == -1);
      bool hasSibling = (jumps[thisIndex] >= 0);

      if (hasChild) {
        if (hasSibling) {
          auto siblingIndex = thisIndex + size_t(jumps[thisIndex]);

          if (siblingIndex >= jumps.size()) {
            PUSH_ERROR_AND_RETURN("jump index corrupted.");
          }

          // Find subtree end.
          size_t subtreeStartIdx = siblingIndex;
          size_t subtreeIdx = subtreeStartIdx;

          for (; subtreeIdx < jumps.size(); subtreeIdx++) {

            bool has_child = (jumps[subtreeIdx] > 0) || (jumps[subtreeIdx] == -1);
            bool has_sibling = (jumps[subtreeIdx] >= 0);

            if (has_child || has_sibling) {
              continue;
            }
            break;
          }

          size_t subtreeEndIdx = subtreeIdx;
          if (subtreeEndIdx >= jumps.size()) {
            // Guess corrupted.
            PUSH_ERROR_AND_RETURN("jump indices seems corrupted.");
          }

          DCOUT("subtree startIdx " << subtreeStartIdx << ", subtree endIndex " << subtreeEndIdx);

          if (subtreeEndIdx >= subtreeStartIdx) {

            // index range after traversing subtree
            if (jumps[thisIndex] > 1) {
                startIndexStack.push(thisIndex+1);
                // jumps should be always positive, so no siblingIndex < thisIndex
                endIndexStack.push(siblingIndex-1); // endIndex is inclusive so subtract 1.
                parentNodeIndexStack.push(int64_t(thisIndex));
            }

            startIndexStack.push(subtreeStartIdx);
            endIndexStack.push(subtreeEndIdx);
            parentNodeIndexStack.push(parentNodeIndex);

            DCOUT("stack size: " << startIndexStack.size());

            nIter++;

            break; // goto `(A)`
          }

        }
        // Have a child (may have also had a sibling). Reset parent node index
        parentNodeIndex = int64_t(thisIndex);
        DCOUT("parentNodeIndex = " << parentNodeIndex);
      }
    }

    // (A)

    if (startIndexStack.empty()) {
      break; // end traversal
    }

    startIndex = startIndexStack.top();
    startIndexStack.pop();

    endIndex = endIndexStack.top();
    endIndexStack.pop();

    parentNodeIndex = parentNodeIndexStack.top();
    parentNodeIndexStack.pop();

    nIter++;
  }

  if (nIter >= maxIter) {
    PUSH_ERROR_AND_RETURN("PathIndex tree Too deep.");
  }

  return true;
}
#else
// TODO(syoyo): Refactor. Code is mostly identical to BuildDecompressedPathsImpl
bool CrateReader::BuildNodeHierarchy(
    std::vector<uint32_t> const &pathIndexes,
    std::vector<int32_t> const &elementTokenIndexes,
    std::vector<int32_t> const &jumps,
    std::vector<bool> &visit_table, /* inout */
    size_t curIndex,
    int64_t parentNodeIndex) {
  bool hasChild = false, hasSibling = false;

  // NOTE: Need to indirectly lookup index through pathIndexes[] when accessing
  // `_nodes`
  do {
    auto thisIndex = curIndex++;
    DCOUT("thisIndex = " << thisIndex << ", curIndex = " << curIndex);
    if (parentNodeIndex == -1) {
      // root node.
      // Assume single root node in the scene.
      //assert(thisIndex == 0);
      if (thisIndex != 0) {
        PUSH_ERROR_AND_RETURN_TAG(kTag, "TODO: Multiple root nodes.");
      }

      if (thisIndex >= pathIndexes.size()) {
        PUSH_ERROR_AND_RETURN_TAG(kTag, "Index out-of-range.");
      }

      size_t pathIdx = pathIndexes[thisIndex];
      if (pathIdx >= _paths.size()) {
        PUSH_ERROR_AND_RETURN_TAG(kTag, "PathIndex out-of-range.");
      }

      if (pathIdx >= _nodes.size()) {
        PUSH_ERROR_AND_RETURN_TAG(kTag, "PathIndex out-of-range.");
      }

      if (pathIdx >= visit_table.size()) {
        // This should not be happan though
        PUSH_ERROR_AND_RETURN_TAG(kTag, "[InternalError] out-of-range.");
      }

      if (visit_table[pathIdx]) {
        PUSH_ERROR_AND_RETURN_TAG(kTag, "Circular referencing detected. Invalid Prim tree representation.");
      }

      Node root(parentNodeIndex, _paths[pathIdx]);

      _nodes[pathIdx] = root;
      visit_table[pathIdx] = true;

      parentNodeIndex = int64_t(thisIndex);

    } else {
      //if (parentNodeIndex >= int64_t(_nodes.size())) {
      //  PUSH_ERROR_AND_RETURN_TAG(kTag, "Parent Index out-of-range.");
      //}

      if (parentNodeIndex >= int64_t(pathIndexes.size())) {
        PUSH_ERROR_AND_RETURN_TAG(kTag, "Parent Index out-of-range.");
      }

      if (thisIndex >= pathIndexes.size()) {
        PUSH_ERROR_AND_RETURN_TAG(kTag, "Index out-of-range.");
      }

      DCOUT("Hierarchy. parent[" << pathIndexes[size_t(parentNodeIndex)]
                                 << "].add_child = " << pathIndexes[thisIndex]);

      size_t pathIdx = pathIndexes[thisIndex];
      if (pathIdx >= _paths.size()) {
        PUSH_ERROR_AND_RETURN_TAG(kTag, "PathIndex out-of-range.");
      }

      if (pathIdx >= _nodes.size()) {
        PUSH_ERROR_AND_RETURN_TAG(kTag, "PathIndex out-of-range.");
      }

      if (pathIdx >= visit_table.size()) {
        // This should not be happan though
        PUSH_ERROR_AND_RETURN_TAG(kTag, "[InternalError] out-of-range.");
      }

      if (visit_table[pathIdx]) {
        PUSH_ERROR_AND_RETURN_TAG(kTag, "Circular referencing detected. Invalid Prim tree representation.");
      }

      Node node(parentNodeIndex, _paths[pathIdx]);

      // Ensure parent is not set yet.
      if (_nodes[pathIdx].GetParent() != -2) {
        PUSH_ERROR_AND_RETURN_TAG(kTag, "???: Maybe corrupted path hierarchy?.");
      }

      _nodes[pathIdx] = node;
      visit_table[pathIdx] = true;

      if (pathIdx >= _elemPaths.size()) {
        PUSH_ERROR_AND_RETURN_TAG(kTag, "PathIndex out-of-range.");
      }

      //std::string name = _paths[pathIndexes[thisIndex]].local_path_name();
      std::string name = _elemPaths[pathIdx].full_path_name();
      DCOUT("childName = " << name);

      size_t parentNodeIdx = size_t(parentNodeIndex);
      if (parentNodeIdx >= pathIndexes.size()) {
        PUSH_ERROR_AND_RETURN_TAG(kTag, "ParentNodeIdx out-of-range.");
      }

      size_t parentPathIdx = pathIndexes[parentNodeIdx];
      if (parentPathIdx >= _nodes.size()) {
        PUSH_ERROR_AND_RETURN_TAG(kTag, "PathIndex out-of-range.");
      }

      if (!_nodes[parentPathIdx].AddChildren(
          name, pathIdx)) {
        PUSH_ERROR_AND_RETURN_TAG(kTag, "Invalid path index.");
      }
    }

    if (thisIndex >= jumps.size()) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "Index is out-of-range");
    }

    hasChild = (jumps[thisIndex] > 0) || (jumps[thisIndex] == -1);
    hasSibling = (jumps[thisIndex] >= 0);

    if (hasChild) {
      if (hasSibling) {
        auto siblingIndex = thisIndex + size_t(jumps[thisIndex]);
        if (!BuildNodeHierarchy(pathIndexes, elementTokenIndexes, jumps, visit_table,
                                siblingIndex, parentNodeIndex)) {
          return false;
        }
      }
      // Have a child (may have also had a sibling). Reset parent node index
      parentNodeIndex = int64_t(thisIndex);
      DCOUT("parentNodeIndex = " << parentNodeIndex);
    }
    // If we had only a sibling, we just continue since the parent path is
    // unchanged and the next thing in the reader stream is the sibling's
    // header.
  } while (hasChild || hasSibling);

  return true;
}
#endif

bool CrateReader::ReadCompressedPaths(const uint64_t maxNumPaths) {
  std::vector<uint32_t> pathIndexes;
  std::vector<int32_t> elementTokenIndexes;
  std::vector<int32_t> jumps;

  // Read number of encoded paths.
  uint64_t numEncodedPaths;
  if (!_sr->read8(&numEncodedPaths)) {
    _err += "Failed to read the number of encoded paths.\n";
    return false;
  }

  DCOUT("maxNumPaths : " << maxNumPaths);
  DCOUT("numEncodedPaths : " << numEncodedPaths);

  // Number of compressed paths could be less than maxNumPaths,
  // but should not be greater.
  if (maxNumPaths < numEncodedPaths) {
    _err += "Size mismatch of numEncodedPaths at `PATHS` section.\n";
    return false;
  }


  // 3 = pathIndex, elementTokenIndex, jump
  CHECK_MEMORY_USAGE(size_t(numEncodedPaths) * sizeof(int32_t) * 3);

  pathIndexes.resize(static_cast<size_t>(numEncodedPaths));
  elementTokenIndexes.resize(static_cast<size_t>(numEncodedPaths));
  jumps.resize(static_cast<size_t>(numEncodedPaths));

  size_t compBufferSize = Usd_IntegerCompression::GetCompressedBufferSize(static_cast<size_t>(numEncodedPaths));
  size_t workspaceBufferSize = Usd_IntegerCompression::GetDecompressionWorkingSpaceSize(static_cast<size_t>(numEncodedPaths));
  CHECK_MEMORY_USAGE(compBufferSize);
  CHECK_MEMORY_USAGE(workspaceBufferSize);

  // Create temporary space for decompressing.
  std::vector<char> compBuffer(compBufferSize);
  std::vector<char> workingSpace(workspaceBufferSize);

  // pathIndexes.
  {
    uint64_t compPathIndexesSize;
    if (!_sr->read8(&compPathIndexesSize)) {
      _err += "Failed to read pathIndexesSize.\n";
      return false;
    }

    if (compPathIndexesSize > compBufferSize) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "Invalid Compressed PathIndexes size.");
    }

    CHECK_MEMORY_USAGE(size_t(compPathIndexesSize));

    if (compPathIndexesSize !=
        _sr->read(size_t(compPathIndexesSize), size_t(compPathIndexesSize),
                  reinterpret_cast<uint8_t *>(compBuffer.data()))) {
      _err += "Failed to read compressed pathIndexes data.\n";
      return false;
    }

    DCOUT("comBuffer.size = " << compBuffer.size());
    DCOUT("compPathIndexesSize = " << compPathIndexesSize);

    std::string err;
    Usd_IntegerCompression::DecompressFromBuffer(
        compBuffer.data(), size_t(compPathIndexesSize), pathIndexes.data(),
        size_t(numEncodedPaths), &err, workingSpace.data());
    if (!err.empty()) {
      _err += "Failed to decode pathIndexes\n" + err;
      return false;
    }
  }

  // elementTokenIndexes.
  {
    uint64_t compElementTokenIndexesSize;
    if (!_sr->read8(&compElementTokenIndexesSize)) {
      _err += "Failed to read elementTokenIndexesSize.\n";
      return false;
    }

    if (compElementTokenIndexesSize > compBufferSize) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "Invalid Compressed elementTokenIndexes size.");
    }

    CHECK_MEMORY_USAGE(size_t(compElementTokenIndexesSize));

    if (compElementTokenIndexesSize !=
        _sr->read(size_t(compElementTokenIndexesSize),
                  size_t(compElementTokenIndexesSize),
                  reinterpret_cast<uint8_t *>(compBuffer.data()))) {
      PUSH_ERROR("Failed to read elementTokenIndexes data.");
      return false;
    }

    std::string err;
    Usd_IntegerCompression::DecompressFromBuffer(
        compBuffer.data(), size_t(compElementTokenIndexesSize),
        elementTokenIndexes.data(), size_t(numEncodedPaths), &err,
        workingSpace.data());

    if (!err.empty()) {
      PUSH_ERROR("Failed to decode elementTokenIndexes.");
      return false;
    }
  }

  // jumps.
  {
    uint64_t compJumpsSize;
    if (!_sr->read8(&compJumpsSize)) {
      PUSH_ERROR("Failed to read compressed jumpsSize.");
      return false;
    }

    if (compJumpsSize > compBufferSize) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "Invalid Compressed elementTokenIndexes size.");
    }

    CHECK_MEMORY_USAGE(size_t(compJumpsSize));

    if (compJumpsSize !=
        _sr->read(size_t(compJumpsSize), size_t(compJumpsSize),
                  reinterpret_cast<uint8_t *>(compBuffer.data()))) {
      PUSH_ERROR("Failed to read compressed jumps data.");
      return false;
    }

    std::string err;
    Usd_IntegerCompression::DecompressFromBuffer(
        compBuffer.data(), size_t(compJumpsSize), jumps.data(), size_t(numEncodedPaths),
        &err, workingSpace.data());

    if (!err.empty()) {
      PUSH_ERROR("Failed to decode jumps.");
      return false;
    }
  }

#ifdef TINYUSDZ_LOCAL_DEBUG_PRINT
  for (size_t i = 0; i < pathIndexes.size(); i++) {
    DCOUT("pathIndexes[" << i << "] = " << pathIndexes[i]);
  }

  for (size_t i = 0; i < elementTokenIndexes.size(); i++) {
    std::stringstream ss;
    ss << "elementTokenIndexes[" << i << "] = " << elementTokenIndexes[i];
    int32_t tokIdx = elementTokenIndexes[i];
    if (tokIdx < 0) {
      // Property Path. Need to negate it.
      tokIdx = -tokIdx;
    }
    if (auto tokv = GetToken(crate::Index(uint32_t(tokIdx)))) {
      ss << "(" << tokv.value() << ")";
    }
    ss << "\n";
    DCOUT(ss.str());
  }

  for (size_t i = 0; i < jumps.size(); i++) {
    DCOUT(fmt::format("jumps[{}] = {}", i, jumps[i]));
  }
#endif

  // For circular tree check
  std::vector<bool> visit_table;
  CHECK_MEMORY_USAGE(_paths.size()); // TODO: divide by 8?

  // `_paths` is already initialized just before calling this ReadCompressedPaths
  visit_table.resize(_paths.size());
  for (size_t i = 0; i < visit_table.size(); i++) {
    visit_table[i] = false;
  }

  // Now build the paths.
#if defined(TINYUSDZ_CRATE_USE_FOR_BASED_PATH_INDEX_DECODER)
  BuildDecompressedPathsArg arg;
  arg.pathIndexes = &pathIndexes;
  arg.elementTokenIndexes = &elementTokenIndexes;
  arg.jumps = &jumps;
  arg.visit_table = &visit_table;
  arg.startIndex = 0;
  arg.endIndex = pathIndexes.size() - 1; // or numEncodedPaths - 1
  arg.parentPath = Path();
  if (!BuildDecompressedPathsImpl(&arg)) {
    return false;
  }

#else
  if (!BuildDecompressedPathsImpl(pathIndexes, elementTokenIndexes, jumps, visit_table,
                                  /* curIndex */ 0, Path())) {
    return false;
  }
#endif

  //
  // Ensure decoded numEncodedPaths.
  //
  size_t sumDecodedPaths = 0;
  for (size_t i = 0; i < visit_table.size(); i++) {
    if (visit_table[i]) {
      sumDecodedPaths++;
    }
  }
  if (sumDecodedPaths != numEncodedPaths) {
    PUSH_ERROR_AND_RETURN(fmt::format("Decoded {} paths but numEncodedPaths in Crate is {}. Possible corruption of Crate data.",
      sumDecodedPaths, numEncodedPaths));
  }

  // Now build node hierarchy.

  // Circular referencing check should be done in BuildDecompressedPathsImpl,
  // but do check it again just in case.
  for (size_t i = 0; i < visit_table.size(); i++) {
    visit_table[i] = false;
  }
  if (!BuildNodeHierarchy(pathIndexes, elementTokenIndexes, jumps, visit_table,
                          /* curIndex */ 0, /* parent node index */ -1)) {
    return false;
  }

  sumDecodedPaths = 0;
  for (size_t i = 0; i < visit_table.size(); i++) {
    if (visit_table[i]) {
      sumDecodedPaths++;
    }
  }
  if (sumDecodedPaths != numEncodedPaths) {
    PUSH_ERROR_AND_RETURN(fmt::format("Decoded {} paths but numEncodedPaths in BuildNodeHierarchy is {}. Possible corruption of Crate data.",
      sumDecodedPaths, numEncodedPaths));
  }

  return true;
}

bool CrateReader::ReadSection(crate::Section *s) {
  size_t name_len = crate::kSectionNameMaxLength + 1;

  if (name_len !=
      _sr->read(name_len, name_len, reinterpret_cast<uint8_t *>(s->name))) {
    _err += "Failed to read section.name.\n";
    return false;
  }

  if (!_sr->read8(&s->start)) {
    _err += "Failed to read section.start.\n";
    return false;
  }

  if (size_t(s->start) > _sr->size()) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Section start offset exceeds USDC file size.");
  }

  if (!_sr->read8(&s->size)) {
    _err += "Failed to read section.size.\n";
    return false;
  }

  if (size_t(s->start + s->size) > _sr->size()) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Section end offset exceeds USDC file size.");
  }


  return true;
}

bool CrateReader::ReadTokens() {
  if ((_tokens_index < 0) || (_tokens_index >= int64_t(_toc.sections.size()))) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Invalid index for `TOKENS` section.");
  }

  if ((_version[0] == 0) && (_version[1] < 4)) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Version must be 0.4.0 or later, but got {}.{}.{}",
      _version[0], _version[1], _version[2]));
  }

  const crate::Section &sec = _toc.sections[size_t(_tokens_index)];
  if (!_sr->seek_set(uint64_t(sec.start))) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to move to `TOKENS` section.");
    return false;
  }

  if (sec.size < 4) {
     PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("`TOKENS` section data size is zero or too small."));
  }

  DCOUT("sec.start = " << sec.start);
  DCOUT("sec.size = " << sec.size);

  // # of tokens.
  uint64_t num_tokens;
  if (!_sr->read8(&num_tokens)) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read # of tokens at `TOKENS` section.");
  }

  DCOUT("# of tokens = " << num_tokens);

  if (num_tokens == 0) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Empty tokens.");
  }

  if (num_tokens > _config.maxNumTokens) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Too many Tokens.");
  }

  // Tokens are lz4 compressed starting from version 0.4.0

  // Compressed token data.
  uint64_t uncompressedSize;
  if (!_sr->read8(&uncompressedSize)) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read uncompressedSize at `TOKENS` section.");
  }

  DCOUT("uncompressedSize = " << uncompressedSize);


  // Must be larger than len(';-)') + all empty string case.
  // 3 = ';-)'
  // num_tokens = '\0' delimiter
  if ((3 + num_tokens) > uncompressedSize) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "`TOKENS` section corrupted.");
  }

  // At least min size should be 16 both for compress and uncompress.
  if (uncompressedSize < 4) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "uncompressedSize too small or zero bytes.");
  }

  uint64_t compressedSize;
  if (!_sr->read8(&compressedSize)) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read compressedSize at `TOKENS` section.");
  }

  DCOUT("compressedSize = " << compressedSize);

  if (compressedSize < 4) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "compressedSize is too small or zero bytes.");
  }

  if (compressedSize > _sr->size()) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Compressed data size exceeds input file size.");
  }

  if (size_t(compressedSize) > size_t(sec.size)) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Compressed data size exceeds `TOKENS` section size.");
  }

  // To combat with heap-buffer flow in lz4 cuased by corrupted lz4 compressed data,
  // We allocate same size of uncompressedSize(or larger one),
  // And further, extra 128 bytes for safety(LZ4_FAST_DEC_LOOP does 16 bytes stride memcpy)

  uint64_t bufSize = (std::max)(compressedSize, uncompressedSize);
  CHECK_MEMORY_USAGE(bufSize+128);
  CHECK_MEMORY_USAGE(uncompressedSize);


  // dst
  std::vector<char> chars(static_cast<size_t>(uncompressedSize));
  memset(chars.data(), 0, chars.size());

  std::vector<char> compressed(static_cast<size_t>(bufSize + 128));
  memset(compressed.data(), 0, compressed.size());

  if (compressedSize !=
      _sr->read(size_t(compressedSize), size_t(compressedSize),
                reinterpret_cast<uint8_t *>(compressed.data()))) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read compressed data at `TOKENS` section.");
    return false;
  }

  if (uncompressedSize !=
      LZ4Compression::DecompressFromBuffer(compressed.data(), chars.data(),
                                           size_t(compressedSize),
                                           size_t(uncompressedSize), &_err)) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to decompress data of Tokens.");
  }

  // Split null terminated string into _tokens.
  const char *ps = chars.data();
  const char *pe = chars.data() + chars.size();
  const char *pcurr = ps;
  size_t nbytes_remain = size_t(chars.size());

  auto my_strnlen = [](const char *s, const size_t max_length) -> size_t {
    if (!s) return 0;

    size_t i = 0;
    for (; i < max_length; i++) {
      if (s[i] == '\0') {
        return i;
      }
    }

    // null character not found.
    return i;
  };

  // TODO(syoyo): Check if input string has exactly `n` tokens(`n` null
  // characters)
  for (size_t i = 0; i < num_tokens; i++) {
    DCOUT("n_remain = " << nbytes_remain);

    size_t len = my_strnlen(pcurr, nbytes_remain);
    DCOUT("len = " << len);

    if ((pcurr + (len+1)) > pe) {
      _err += "Invalid token string array.\n";
      return false;
    }

    std::string str;
    if (len > 0) {
      str = std::string(pcurr, len);
    } else {
      // Empty string allowed
      str = std::string();
    }

    pcurr += len + 1;  // +1 = '\0'
    nbytes_remain = size_t(pe - pcurr);
    if (pcurr > pe) {
      _err += "Invalid token string array.\n";
      return false;
    }

    value::token tok(str);

    DCOUT("token[" << i << "] = " << tok);
    _tokens.push_back(tok);

    if (nbytes_remain == 0) {
      // reached to the string buffer end.
      break;
    }
  }

  if (_tokens.size() != num_tokens) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("The number of tokens parsed {} does not match the requested one {}", _tokens.size(), num_tokens));
  }

  return true;
}

bool CrateReader::ReadStrings() {
  if ((_strings_index < 0) ||
      (_strings_index >= int64_t(_toc.sections.size()))) {
    _err += "Invalid index for `STRINGS` section.\n";
    return false;
  }

  const crate::Section &s = _toc.sections[size_t(_strings_index)];

  if (s.size == 0) {
    // empty `STRINGS`?
    return true;
  }

  if (!_sr->seek_set(uint64_t(s.start))) {
    _err += "Failed to move to `STRINGS` section.\n";
    return false;
  }

  // `STRINGS` are not compressed.
  if (!ReadIndices(&_string_indices)) {
    _err += "Failed to read StringIndex array.\n";
    return false;
  }

  for (size_t i = 0; i < _string_indices.size(); i++) {
    DCOUT("StringIndex[" << i << "] = " << _string_indices[i].value);
  }

  return true;
}

bool CrateReader::ReadFields() {
  if ((_fields_index < 0) || (_fields_index >= int64_t(_toc.sections.size()))) {
    _err += "Invalid index for `FIELDS` section.\n";
    return false;
  }

  if ((_version[0] == 0) && (_version[1] < 4)) {
    _err += "Version must be 0.4.0 or later, but got " +
            std::to_string(_version[0]) + "." + std::to_string(_version[1]) +
            "." + std::to_string(_version[2]) + "\n";
    return false;
  }

  const crate::Section &s = _toc.sections[size_t(_fields_index)];

  if (s.size == 0) {
    // accepts Empty FIELDS size.
    return true;
  }

  if (!_sr->seek_set(uint64_t(s.start))) {
    _err += "Failed to move to `FIELDS` section.\n";
    return false;
  }

  uint64_t num_fields;
  if (!_sr->read8(&num_fields)) {
    _err += "Failed to read # of fields at `FIELDS` section.\n";
    return false;
  }

  DCOUT("num_fields = " << num_fields);

  if (num_fields == 0) {
    // Fields may be empty, so OK
    return true;
  }

  if (num_fields > _config.maxNumFields) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Too many fields in `FIELDS` section.");
  }

  if (sizeof(void *) == 4) {
    // 32bit
    if (num_fields > std::numeric_limits<int32_t>::max() / sizeof(uint32_t)) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "Too many fields in `FIELDS` section.");
    }
  }

  CHECK_MEMORY_USAGE(size_t(num_fields) * sizeof(Field));

  _fields.resize(static_cast<size_t>(num_fields));

  // indices
  {

    CHECK_MEMORY_USAGE(size_t(num_fields) * sizeof(uint32_t));

    std::vector<uint32_t> tmp;
    tmp.resize(size_t(num_fields));
    if (!ReadCompressedInts(tmp.data(), size_t(num_fields))) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read Field token_index array.");
    }

    for (size_t i = 0; i < num_fields; i++) {
      _fields[i].token_index.value = tmp[i];
    }

    REDUCE_MEMORY_USAGE(size_t(num_fields) * sizeof(uint32_t));

  }

  // Value reps(LZ4 compressed)
  {
    uint64_t reps_size; // compressed size
    if (!_sr->read8(&reps_size)) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read value reps legnth at `FIELDS` section.");
    }

    if (reps_size > size_t(s.size)) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "Invalid byte size of Value reps data.");
    }

    if (reps_size > _sr->size()) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "Compressed Value reps size exceeds USDC data.");
    }

    CHECK_MEMORY_USAGE(size_t(reps_size));

    // TODO: Decompress from _sr directly.
    std::vector<char> comp_buffer(static_cast<size_t>(reps_size));

    if (reps_size !=
        _sr->read(size_t(reps_size), size_t(reps_size),
                  reinterpret_cast<uint8_t *>(comp_buffer.data()))) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read reps data at `FIELDS` section.");
    }

    // reps datasize = LZ4 compressed. uncompressed size = num_fields * 8 bytes
    size_t uncompressed_size = size_t(num_fields) * sizeof(uint64_t);
    CHECK_MEMORY_USAGE(uncompressed_size);

    std::vector<uint64_t> reps_data;
    reps_data.resize(static_cast<size_t>(num_fields));


    if (uncompressed_size != LZ4Compression::DecompressFromBuffer(
                                 comp_buffer.data(),
                                 reinterpret_cast<char *>(reps_data.data()),
                                 size_t(reps_size), uncompressed_size, &_err)) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read Fields ValueRep data.");
    }

    for (size_t i = 0; i < num_fields; i++) {
      _fields[i].value_rep = crate::ValueRep(reps_data[i]);
    }

    REDUCE_MEMORY_USAGE(uncompressed_size);
    REDUCE_MEMORY_USAGE(size_t(reps_size)); // comp_buffer
  }

  DCOUT("num_fields = " << num_fields);
  for (size_t i = 0; i < num_fields; i++) {
    if (auto tokv = GetToken(_fields[i].token_index)) {
      DCOUT("field[" << i << "] name = " << tokv.value()
                     << ", value = " << _fields[i].value_rep.GetStringRepr());
    }
  }

  return true;
}

bool CrateReader::ReadFieldSets() {
  if ((_fieldsets_index < 0) ||
      (_fieldsets_index >= int64_t(_toc.sections.size()))) {
    _err += "Invalid index for `FIELDSETS` section.\n";
    return false;
  }

  if ((_version[0] == 0) && (_version[1] < 4)) {
    _err += "Version must be 0.4.0 or later, but got " +
            std::to_string(_version[0]) + "." + std::to_string(_version[1]) +
            "." + std::to_string(_version[2]) + "\n";
    return false;
  }

  const crate::Section &s = _toc.sections[size_t(_fieldsets_index)];

  if (!_sr->seek_set(uint64_t(s.start))) {
    _err += "Failed to move to `FIELDSETS` section.\n";
    return false;
  }

  uint64_t num_fieldsets;
  if (!_sr->read8(&num_fieldsets)) {
    _err += "Failed to read # of fieldsets at `FIELDSETS` section.\n";
    return false;
  }

  if (num_fieldsets == 0) {
    // At least 1 FieldIndex(separator(~0)) must exist.
    PUSH_ERROR("`FIELDSETS` is empty.");
    return false;
  }

  if (num_fieldsets > _config.maxNumFieldSets) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Too many FieldSets {}. maxNumFieldSets is set to {}", num_fieldsets, _config.maxNumFieldSets));
  }

  CHECK_MEMORY_USAGE(size_t(num_fieldsets) * sizeof(uint32_t));

  _fieldset_indices.resize(static_cast<size_t>(num_fieldsets));

  // Create temporary space for decompressing.
  size_t compBufferSize = Usd_IntegerCompression::GetCompressedBufferSize(
      static_cast<size_t>(num_fieldsets));

  CHECK_MEMORY_USAGE(compBufferSize);

  std::vector<char> comp_buffer;
  comp_buffer.resize(compBufferSize);

  CHECK_MEMORY_USAGE(sizeof(uint32_t) * size_t(num_fieldsets));
  std::vector<uint32_t> tmp;
  tmp.resize(static_cast<size_t>(num_fieldsets));

  size_t workBufferSize = Usd_IntegerCompression::GetDecompressionWorkingSpaceSize(
          static_cast<size_t>(num_fieldsets));

  CHECK_MEMORY_USAGE(workBufferSize);
  std::vector<char> working_space;
  working_space.resize(workBufferSize);

  uint64_t fsets_size;
  if (!_sr->read8(&fsets_size)) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read fieldsets size at `FIELDSETS` section.");
  }

  DCOUT("num_fieldsets = " << num_fieldsets << ", fsets_size = " << fsets_size
                           << ", comp_buffer.size = " << comp_buffer.size());

  if (fsets_size > comp_buffer.size()) {
    // Maybe corrupted?
    fsets_size = comp_buffer.size();
  }

  if (fsets_size > _sr->size()) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "FieldSets compressed data exceeds USDC data.");
  }

  if (fsets_size !=
      _sr->read(size_t(fsets_size), size_t(fsets_size),
                reinterpret_cast<uint8_t *>(comp_buffer.data()))) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Failed to read fieldsets data at `FIELDSETS` section.");
  }

  std::string err;
  Usd_IntegerCompression::DecompressFromBuffer(
      comp_buffer.data(), size_t(fsets_size), tmp.data(), size_t(num_fieldsets),
      &err, working_space.data());

  if (!err.empty()) {
    _err += err;
    return false;
  }

  for (size_t i = 0; i != num_fieldsets; ++i) {
    DCOUT("fieldset_index[" << i << "] = " << tmp[i]);
    _fieldset_indices[i].value = tmp[i];
  }

  REDUCE_MEMORY_USAGE(workBufferSize);
  REDUCE_MEMORY_USAGE(compBufferSize);

  return true;
}

bool CrateReader::BuildLiveFieldSets() {
  for (auto fsBegin = _fieldset_indices.begin(),
            fsEnd = std::find(fsBegin, _fieldset_indices.end(), crate::Index());
       fsBegin != _fieldset_indices.end();
       fsBegin = fsEnd + 1, fsEnd = std::find(fsBegin, _fieldset_indices.end(),
                                              crate::Index())) {
    auto &pairs = _live_fieldsets[crate::Index(
        uint32_t(fsBegin - _fieldset_indices.begin()))];

    pairs.resize(size_t(fsEnd - fsBegin));
    DCOUT("range size = " << (fsEnd - fsBegin));
    // TODO(syoyo): Parallelize.
    for (size_t i = 0; fsBegin != fsEnd; ++fsBegin, ++i) {
      if (fsBegin->value < _fields.size()) {
        // ok
      } else {
        PUSH_ERROR("Invalid live field set data.");
        return false;
      }

      DCOUT("fieldIndex = " << (fsBegin->value));
      auto const &field = _fields[fsBegin->value];
      if (auto tokv = GetToken(field.token_index)) {
        pairs[i].first = tokv.value().str();

        if (!UnpackValueRep(field.value_rep, &pairs[i].second)) {
          PUSH_ERROR("BuildLiveFieldSets: Failed to unpack ValueRep : "
                     << field.value_rep.GetStringRepr());
          return false;
        }
      } else {
        PUSH_ERROR("Invalid token index.");
      }
    }
  }

  DCOUT("# of live fieldsets = " << _live_fieldsets.size());

#ifdef TINYUSDZ_LOCAL_DEBUG_PRINT
  size_t sum = 0;
  for (const auto &item : _live_fieldsets) {
    DCOUT("livefieldsets[" << item.first.value
                           << "].count = " << item.second.size());
    sum += item.second.size();

    for (size_t i = 0; i < item.second.size(); i++) {
      DCOUT(" [" << i << "] name = " << item.second[i].first);
    }
  }
  DCOUT("Total fields used = " << sum);
#endif

  return true;
}

bool CrateReader::ReadSpecs() {
  if ((_specs_index < 0) || (_specs_index >= int64_t(_toc.sections.size()))) {
    PUSH_ERROR("Invalid index for `SPECS` section.");
    return false;
  }

  if ((_version[0] == 0) && (_version[1] < 4)) {
    PUSH_ERROR("Version must be 0.4.0 or later, but got " +
               std::to_string(_version[0]) + "." + std::to_string(_version[1]) +
               "." + std::to_string(_version[2]));
    return false;
  }

  const crate::Section &s = _toc.sections[size_t(_specs_index)];

  if (!_sr->seek_set(uint64_t(s.start))) {
    PUSH_ERROR("Failed to move to `SPECS` section.");
    return false;
  }

  uint64_t num_specs;
  if (!_sr->read8(&num_specs)) {
    PUSH_ERROR("Failed to read # of specs size at `SPECS` section.");
    return false;
  }

  if (num_specs > _config.maxNumSpecifiers) {
    PUSH_ERROR("Too many specs in `SPECS` section.");
    return false;
  }

  if (num_specs == 0) {
    // At least 1 Spec(Root Prim '/') must exist.
    PUSH_ERROR("`SPECS` is empty.");
    return false;
  }

  DCOUT("num_specs " << num_specs);

  CHECK_MEMORY_USAGE(size_t(num_specs) * sizeof(Spec));

  _specs.resize(static_cast<size_t>(num_specs));

  // TODO: Memory size check

  // Create temporary space for decompressing.
  size_t compBufferSize= Usd_IntegerCompression::GetCompressedBufferSize(
      static_cast<size_t>(num_specs));

  CHECK_MEMORY_USAGE(compBufferSize);

  std::vector<char> comp_buffer;
  comp_buffer.resize(compBufferSize);

  CHECK_MEMORY_USAGE(size_t(num_specs) * sizeof(uint32_t)); // tmp

  std::vector<uint32_t> tmp(static_cast<size_t>(num_specs));

  size_t workBufferSize= Usd_IntegerCompression::GetDecompressionWorkingSpaceSize(
          static_cast<size_t>(num_specs));

  CHECK_MEMORY_USAGE(workBufferSize);
  std::vector<char> working_space;
  working_space.resize(workBufferSize);

  // path indices
  {
    uint64_t path_indexes_size;
    if (!_sr->read8(&path_indexes_size)) {
      PUSH_ERROR("Failed to read path indexes size at `SPECS` section.");
      return false;
    }

    if (path_indexes_size > comp_buffer.size()) {
      // Maybe corrupted?
      path_indexes_size = comp_buffer.size();
    }

    if (path_indexes_size !=
        _sr->read(size_t(path_indexes_size), size_t(path_indexes_size),
                  reinterpret_cast<uint8_t *>(comp_buffer.data()))) {
      PUSH_ERROR("Failed to read path indexes data at `SPECS` section.");
      return false;
    }

    std::string err;  // not used
    if (!Usd_IntegerCompression::DecompressFromBuffer(
            comp_buffer.data(), size_t(path_indexes_size), tmp.data(),
            size_t(num_specs), &err, working_space.data())) {
      PUSH_ERROR("Failed to decode pathIndexes at `SPECS` section.");
      return false;
    }

    for (size_t i = 0; i < num_specs; ++i) {
      DCOUT("spec[" << i << "].path_index = " << tmp[i]);
      _specs[i].path_index.value = tmp[i];
    }
  }

  // fieldset indices
  {
    uint64_t fset_indexes_size;
    if (!_sr->read8(&fset_indexes_size)) {
      PUSH_ERROR("Failed to read fieldset indexes size at `SPECS` section.");
      return false;
    }

    if (fset_indexes_size > comp_buffer.size()) {
      // Maybe corrupted?
      fset_indexes_size = comp_buffer.size();
    }

    if (fset_indexes_size !=
        _sr->read(size_t(fset_indexes_size), size_t(fset_indexes_size),
                  reinterpret_cast<uint8_t *>(comp_buffer.data()))) {
      PUSH_ERROR("Failed to read fieldset indexes data at `SPECS` section.");
      return false;
    }

    std::string err;  // not used
    if (!Usd_IntegerCompression::DecompressFromBuffer(
            comp_buffer.data(), size_t(fset_indexes_size), tmp.data(),
            size_t(num_specs), &err, working_space.data())) {
      PUSH_ERROR("Failed to decode fieldset indices at `SPECS` section.");
      return false;
    }

    for (size_t i = 0; i != num_specs; ++i) {
      DCOUT("specs[" << i << "].fieldset_index = " << tmp[i]);
      _specs[i].fieldset_index.value = tmp[i];
    }
  }

  // spec types
  {
    uint64_t spectype_size;
    if (!_sr->read8(&spectype_size)) {
      PUSH_ERROR("Failed to read spectype size at `SPECS` section.");
      return false;
    }

    if (spectype_size > comp_buffer.size()) {
      // Maybe corrupted?
      spectype_size = comp_buffer.size();
    }

    if (spectype_size !=
        _sr->read(size_t(spectype_size), size_t(spectype_size),
                  reinterpret_cast<uint8_t *>(comp_buffer.data()))) {
      PUSH_ERROR("Failed to read spectype data at `SPECS` section.");
      return false;
    }

    std::string err;  // not used.
    if (!Usd_IntegerCompression::DecompressFromBuffer(
            comp_buffer.data(), size_t(spectype_size), tmp.data(),
            size_t(num_specs), &err, working_space.data())) {
      PUSH_ERROR("Failed to decode fieldset indices at `SPECS` section.\n");
      return false;
    }

    for (size_t i = 0; i != num_specs; ++i) {
      // std::cout << "spectype = " << tmp[i] << "\n";
      _specs[i].spec_type = static_cast<SpecType>(tmp[i]);
    }
  }

#ifdef TINYUSDZ_LOCAL_DEBUG_PRINT
  for (size_t i = 0; i != num_specs; ++i) {
    DCOUT("spec[" << i << "].pathIndex  = " << _specs[i].path_index.value
                  << ", fieldset_index = " << _specs[i].fieldset_index.value
                  << ", spec_type = "
                  << tinyusdz::to_string(_specs[i].spec_type));
    if (auto specstr = GetSpecString(crate::Index(uint32_t(i)))) {
      DCOUT("spec[" << i << "] string_repr = " << specstr.value());
    }
  }
#endif

  REDUCE_MEMORY_USAGE(compBufferSize);
  REDUCE_MEMORY_USAGE(workBufferSize);
  REDUCE_MEMORY_USAGE(size_t(num_specs) * sizeof(uint32_t)); // tmp

  return true;
}

bool CrateReader::ReadPaths() {
  if ((_paths_index < 0) || (_paths_index >= int64_t(_toc.sections.size()))) {
    PUSH_ERROR("Invalid index for `PATHS` section.");
    return false;
  }

  if ((_version[0] == 0) && (_version[1] < 4)) {
    PUSH_ERROR("Version must be 0.4.0 or later, but got " +
               std::to_string(_version[0]) + "." + std::to_string(_version[1]) +
               "." + std::to_string(_version[2]));
    return false;
  }

  const crate::Section &s = _toc.sections[size_t(_paths_index)];

  if (!_sr->seek_set(uint64_t(s.start))) {
    PUSH_ERROR("Failed to move to `PATHS` section.");
    return false;
  }

  uint64_t num_paths;
  if (!_sr->read8(&num_paths)) {
    PUSH_ERROR("Failed to read # of paths at `PATHS` section.");
    return false;
  }

  if (num_paths == 0) {
    // At least root path exits.
    PUSH_ERROR_AND_RETURN_TAG(kTag, "`PATHS` is empty.");
  }

  if (num_paths > _config.maxNumPaths) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, "Too many Paths in `PATHS` section.");
  }

  CHECK_MEMORY_USAGE(size_t(num_paths) * sizeof(Path)); // conservative estimation
  CHECK_MEMORY_USAGE(size_t(num_paths) * sizeof(Path)); // conservative estimation
  CHECK_MEMORY_USAGE(size_t(num_paths) * sizeof(Node)); // conservative estimation

  _paths.resize(static_cast<size_t>(num_paths));
  _elemPaths.resize(static_cast<size_t>(num_paths));
  _nodes.resize(static_cast<size_t>(num_paths));

  if (!ReadCompressedPaths(num_paths)) {
    PUSH_ERROR("Failed to read compressed paths.");
    return false;
  }

#ifdef TINYUSDZ_LOCAL_DEBUG_PRINT
  DCOUT("# of paths " << _paths.size());

  for (size_t i = 0; i < _paths.size(); i++) {
    DCOUT("path[" << i << "] = " << _paths[i].full_path_name());
  }
#endif

  return true;
}

bool CrateReader::ReadBootStrap() {
  // parse header.
  uint8_t magic[8];
  if (8 != _sr->read(/* req */ 8, /* dst len */ 8, magic)) {
    PUSH_ERROR("Failed to read magic number.");
    return false;
  }

  if (memcmp(magic, "PXR-USDC", 8)) {
    PUSH_ERROR("Invalid magic number. Expected 'PXR-USDC' but got '" +
               std::string(magic, magic + 8) + "'");
    return false;
  }

  // parse version(first 3 bytes from 8 bytes)
  uint8_t version[8];
  if (8 != _sr->read(8, 8, version)) {
    PUSH_ERROR("Failed to read magic number.");
    return false;
  }

  DCOUT("version = " << int(version[0]) << "." << int(version[1]) << "."
                     << int(version[2]));

  _version[0] = version[0];
  _version[1] = version[1];
  _version[2] = version[2];

  // We only support version 0.4.0 or later.
  if ((version[0] == 0) && (version[1] < 4)) {
    PUSH_ERROR("Version must be 0.4.0 or later, but got " +
               std::to_string(version[0]) + "." + std::to_string(version[1]) +
               "." + std::to_string(version[2]));
    return false;
  }

  // Currently up to 0.9.0
  if ((version[0] == 0) && (version[1] < 10)) {
    // ok
  } else {
    PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Unsupported version {}.{}.{}. TinyUSDZ supports version up to 0.9.0",
      _version[0], _version[1], _version[2]));
  }

  _toc_offset = 0;
  if (!_sr->read8(&_toc_offset)) {
    PUSH_ERROR("Failed to read TOC offset.");
    return false;
  }

  if ((_toc_offset <= 88) || (_toc_offset >= int64_t(_sr->size()))) {
    PUSH_ERROR("Invalid TOC offset value: " + std::to_string(_toc_offset) +
               ", filesize = " + std::to_string(_sr->size()) + ".");
    return false;
  }

  DCOUT("toc offset = " << _toc_offset);

  return true;
}

bool CrateReader::ReadTOC() {

  DCOUT(fmt::format("Memory budget: {} bytes", _config.maxMemoryBudget));

  if ((_toc_offset <= 88) || (_toc_offset >= int64_t(_sr->size()))) {
    PUSH_ERROR("Invalid toc offset.");
    return false;
  }

  if (!_sr->seek_set(uint64_t(_toc_offset))) {
    PUSH_ERROR("Failed to move to TOC offset.");
    return false;
  }

  // read # of sections.
  uint64_t num_sections{0};
  if (!_sr->read8(&num_sections)) {
    PUSH_ERROR("Failed to read TOC(# of sections).");
    return false;
  }
  if (num_sections >= _config.maxTOCSections) {
    PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("# of sections {} are too large. maxTOCSections is set to {}", num_sections, _config.maxTOCSections));
  }

  DCOUT("toc sections = " << num_sections);

  _toc.sections.resize(static_cast<size_t>(num_sections));

  CHECK_MEMORY_USAGE(num_sections * sizeof(Section));

  for (size_t i = 0; i < num_sections; i++) {
    if (!ReadSection(&_toc.sections[i])) {
      PUSH_ERROR("Failed to read TOC section at " + std::to_string(i));
      return false;
    }
    DCOUT("section[" << i << "] name = " << _toc.sections[i].name
                     << ", start = " << _toc.sections[i].start
                     << ", size = " << _toc.sections[i].size);

    if (_toc.sections[i].start < 0) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Invalid section start byte offset."));
    }

    if (_toc.sections[i].size <= 0) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Invalid or empty section size."));
    }

    if (size_t(_toc.sections[i].size) > _sr->size()) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Section size exceeds input USDC data size."));
    }

    if (size_t(_toc.sections[i].start) > _sr->size()) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Section start byte offset exceeds input USDC data size."));
    }

    // TODO: handle integer overflow.
    size_t end_offset = size_t(_toc.sections[i].start + _toc.sections[i].size);
    if (sizeof(void *) == 4) { // 32bit
      if (end_offset > size_t(std::numeric_limits<int32_t>::max())) {
        PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Section end offset exceeds 32bit max."));
      }
    }
    if (end_offset > _sr->size()) {
      PUSH_ERROR_AND_RETURN_TAG(kTag, fmt::format("Section byte offset + size exceeds input USDC data size."));
    }


    if (0 == strncmp(_toc.sections[i].name, "TOKENS",
                     crate::kSectionNameMaxLength)) {
      _tokens_index = int64_t(i);
    } else if (0 == strncmp(_toc.sections[i].name, "STRINGS",
                            crate::kSectionNameMaxLength)) {
      _strings_index = int64_t(i);
    } else if (0 == strncmp(_toc.sections[i].name, "FIELDS",
                            crate::kSectionNameMaxLength)) {
      _fields_index = int64_t(i);
    } else if (0 == strncmp(_toc.sections[i].name, "FIELDSETS",
                            crate::kSectionNameMaxLength)) {
      _fieldsets_index = int64_t(i);
    } else if (0 == strncmp(_toc.sections[i].name, "SPECS",
                            crate::kSectionNameMaxLength)) {
      _specs_index = int64_t(i);
    } else if (0 == strncmp(_toc.sections[i].name, "PATHS",
                            crate::kSectionNameMaxLength)) {
      _paths_index = int64_t(i);
    }
  }

  DCOUT("TOC read success");
  return true;
}

///
/// Find if a field with (`name`, `tyname`) exists in FieldValuePairVector.
///
bool CrateReader::HasFieldValuePair(const FieldValuePairVector &fvs,
                                    const std::string &name,
                                    const std::string &tyname) {
  for (const auto &fv : fvs) {
    if ((fv.first == name) && (fv.second.type_name() == tyname)) {
      return true;
    }
  }

  return false;
}

///
/// Find if a field with `name`(type can be arbitrary) exists in
/// FieldValuePairVector.
///
bool CrateReader::HasFieldValuePair(const FieldValuePairVector &fvs,
                                    const std::string &name) {
  for (const auto &fv : fvs) {
    if (fv.first == name) {
      return true;
    }
  }

  return false;
}

nonstd::expected<FieldValuePair, std::string>
CrateReader::GetFieldValuePair(const FieldValuePairVector &fvs,
                               const std::string &name,
                               const std::string &tyname) {
  for (const auto &fv : fvs) {
    if ((fv.first == name) && (fv.second.type_name() == tyname)) {
      return fv;
    }
  }

  return nonstd::make_unexpected("FieldValuePair not found with name: `" +
                                 name + "` and specified type: `" + tyname +
                                 "`");
}

nonstd::expected<FieldValuePair, std::string>
CrateReader::GetFieldValuePair(const FieldValuePairVector &fvs,
                               const std::string &name) {
  for (const auto &fv : fvs) {
    if (fv.first == name) {
      return fv;
    }
  }

  return nonstd::make_unexpected("FieldValuePair not found with name: `" +
                                 name + "`");
}


}  // namespace crate
}  // namespace tinyusdz
