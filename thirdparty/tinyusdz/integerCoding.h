//
// Copyright 2017 Pixar
//
// Licensed under the Apache License, Version 2.0 (the "Apache License")
// with the following modification; you may not use this file except in
// compliance with the Apache License and the following modification to it:
// Section 6. Trademarks. is deleted and replaced with:
//
// 6. Trademarks. This License does not grant permission to use the trade
//    names, trademarks, service marks, or product names of the Licensor
//    and its affiliates, except as required to comply with Section 4(c) of
//    the License and to reproduce the content of the NOTICE file.
//
// You may obtain a copy of the Apache License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the Apache License with the above modification is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the Apache License for the specific
// language governing permissions and limitations under the Apache License.
//
#ifndef USD_INTEGERCODING_H
#define USD_INTEGERCODING_H

//#include "pxr/pxr.h"
//#include "pxr/usd/usd/api.h"

#include <cstdint>
#include <memory>

#define USD_API 
//PXR_NAMESPACE_OPEN_SCOPE
namespace tinyusdz {

class Usd_IntegerCompression
{
public:
    // Return the max compression buffer size required for \p numInts 32-bit
    // integers.
    USD_API
    static size_t GetCompressedBufferSize(size_t numInts);

    // Return the max decompression working space size required for \p numInts
    // 32-bit integers.
    USD_API
    static size_t GetDecompressionWorkingSpaceSize(size_t numInts);

    // Compress \p numInts ints from \p ints to \p compressed.  The
    // \p compressed space must point to at least
    // GetCompressedBufferSize(numInts) bytes.  Return the actual number
    // of bytes written to \p compressed.
    USD_API
    static size_t CompressToBuffer(
        int32_t const *ints, size_t numInts, char *compressed, std::string *err);

    // Compress \p numInts ints from \p ints to \p compressed.  The
    // \p compressed space must point to at least
    // GetCompressedBufferSize(numInts) bytes.  Return the actual number
    // of bytes written to \p compressed.
    USD_API
    static size_t CompressToBuffer(
        uint32_t const *ints, size_t numInts, char *compressed, std::string *err);

    // Decompress \p compressedSize bytes from \p compressed to produce
    // \p numInts 32-bit integers into \p ints.  Clients may supply
    // \p workingSpace to save allocations if several decompressions will be
    // done but it isn't required.  If supplied it must point to at least
    // GetDecompressionWorkingSpaceSize(numInts) bytes.
    USD_API
    static size_t DecompressFromBuffer(
        char const *compressed, size_t compressedSize,
        int32_t *ints, size_t numInts, std::string *err,
        char *workingSpace=nullptr);

    // Decompress \p compressedSize bytes from \p compressed to produce
    // \p numInts 32-bit integers into \p ints.  Clients may supply
    // \p workingSpace to save allocations if several decompressions will be
    // done but it isn't required.  If supplied it must point to at least
    // GetDecompressionWorkingSpaceSize(numInts) bytes.
    USD_API
    static size_t DecompressFromBuffer(
        char const *compressed, size_t compressedSize,
        uint32_t *ints, size_t numInts, std::string *err,
        char *workingSpace=nullptr);
};

class Usd_IntegerCompression64
{
public:
    // Return the max compression buffer size required for \p numInts 64-bit
    // integers.
    USD_API
    static size_t GetCompressedBufferSize(size_t numInts);

    // Return the max decompression working space size required for \p numInts
    // 64-bit integers.
    USD_API
    static size_t GetDecompressionWorkingSpaceSize(size_t numInts);

    // Compress \p numInts ints from \p ints to \p compressed.  The
    // \p compressed space must point to at least
    // GetCompressedBufferSize(numInts) bytes.  Return the actual number
    // of bytes written to \p compressed.
    USD_API
    static size_t CompressToBuffer(
        int64_t const *ints, size_t numInts, char *compressed, std::string *err);

    // Compress \p numInts ints from \p ints to \p compressed.  The
    // \p compressed space must point to at least
    // GetCompressedBufferSize(numInts) bytes.  Return the actual number
    // of bytes written to \p compressed.
    USD_API
    static size_t CompressToBuffer(
        uint64_t const *ints, size_t numInts, char *compressed, std::string *err);

    // Decompress \p compressedSize bytes from \p compressed to produce
    // \p numInts 64-bit integers into \p ints.  Clients may supply
    // \p workingSpace to save allocations if several decompressions will be
    // done but it isn't required.  If supplied it must point to at least
    // GetDecompressionWorkingSpaceSize(numInts) bytes.
    USD_API
    static size_t DecompressFromBuffer(
        char const *compressed, size_t compressedSize,
        int64_t *ints, size_t numInts, std::string *err,
        char *workingSpace=nullptr);

    // Decompress \p compressedSize bytes from \p compressed to produce
    // \p numInts 64-bit integers into \p ints.  Clients may supply
    // \p workingSpace to save allocations if several decompressions will be
    // done but it isn't required.  If supplied it must point to at least
    // GetDecompressionWorkingSpaceSize(numInts) bytes.
    USD_API
    static size_t DecompressFromBuffer(
        char const *compressed, size_t compressedSize,
        uint64_t *ints, size_t numInts, std::string *err,
        char *workingSpace=nullptr);
};

//PXR_NAMESPACE_CLOSE_SCOPE
} // namespace tinyusdz

#endif // USD_INTEGERCODING_H
