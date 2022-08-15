///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilContainer.h                                                           //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Provides declarations for the DXIL container format.                      //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#ifndef __DXC_CONTAINER__
#define __DXC_CONTAINER__

#include <stdint.h>
#include <iterator>
#include "dxc/DXIL/DxilConstants.h"
#include "dxc/Support/WinAdapter.h"

struct IDxcContainerReflection;
namespace llvm { class Module; }

namespace hlsl {

class AbstractMemoryStream;
class RootSignatureHandle;
class DxilModule;

#pragma pack(push, 1)

static const size_t DxilContainerHashSize = 16;
static const uint16_t DxilContainerVersionMajor = 1;  // Current major version
static const uint16_t DxilContainerVersionMinor = 0;  // Current minor version
static const uint32_t DxilContainerMaxSize = 0x80000000; // Max size for container.

/// Use this type to represent the hash for the full container.
struct DxilContainerHash {
  uint8_t Digest[DxilContainerHashSize];
};

enum class DxilShaderHashFlags : uint32_t {
  None = 0,           // No flags defined.
  IncludesSource = 1, // This flag indicates that the shader hash was computed
                      // taking into account source information (-Zss)
};

typedef struct DxilShaderHash {
  uint32_t Flags; // DxilShaderHashFlags
  uint8_t Digest[DxilContainerHashSize];
} DxilShaderHash;

struct DxilContainerVersion {
  uint16_t Major;
  uint16_t Minor;
};

/// Use this type to describe a DXIL container of parts.
struct DxilContainerHeader {
  uint32_t              HeaderFourCC;
  DxilContainerHash     Hash;
  DxilContainerVersion  Version;
  uint32_t              ContainerSizeInBytes; // From start of this header
  uint32_t              PartCount;
  // Structure is followed by uint32_t PartOffset[PartCount];
  // The offset is to a DxilPartHeader.
};

/// Use this type to describe the size and type of a DXIL container part.
struct DxilPartHeader {
  uint32_t  PartFourCC; // Four char code for part type.
  uint32_t  PartSize;   // Byte count for PartData.
  // Structure is followed by uint8_t PartData[PartSize].
};

#define DXIL_FOURCC(ch0, ch1, ch2, ch3) (                            \
  (uint32_t)(uint8_t)(ch0)        | (uint32_t)(uint8_t)(ch1) << 8  | \
  (uint32_t)(uint8_t)(ch2) << 16  | (uint32_t)(uint8_t)(ch3) << 24   \
  )

enum DxilFourCC {
  DFCC_Container                = DXIL_FOURCC('D', 'X', 'B', 'C'), // for back-compat with tools that look for DXBC containers
  DFCC_ResourceDef              = DXIL_FOURCC('R', 'D', 'E', 'F'),
  DFCC_InputSignature           = DXIL_FOURCC('I', 'S', 'G', '1'),
  DFCC_OutputSignature          = DXIL_FOURCC('O', 'S', 'G', '1'),
  DFCC_PatchConstantSignature   = DXIL_FOURCC('P', 'S', 'G', '1'),
  DFCC_ShaderStatistics         = DXIL_FOURCC('S', 'T', 'A', 'T'),
  DFCC_ShaderDebugInfoDXIL      = DXIL_FOURCC('I', 'L', 'D', 'B'),
  DFCC_ShaderDebugName          = DXIL_FOURCC('I', 'L', 'D', 'N'),
  DFCC_FeatureInfo              = DXIL_FOURCC('S', 'F', 'I', '0'),
  DFCC_PrivateData              = DXIL_FOURCC('P', 'R', 'I', 'V'),
  DFCC_RootSignature            = DXIL_FOURCC('R', 'T', 'S', '0'),
  DFCC_DXIL                     = DXIL_FOURCC('D', 'X', 'I', 'L'),
  DFCC_PipelineStateValidation  = DXIL_FOURCC('P', 'S', 'V', '0'),
  DFCC_RuntimeData              = DXIL_FOURCC('R', 'D', 'A', 'T'),
  DFCC_ShaderHash               = DXIL_FOURCC('H', 'A', 'S', 'H'),
  DFCC_ShaderSourceInfo         = DXIL_FOURCC('S', 'R', 'C', 'I'),
  DFCC_CompilerVersion          = DXIL_FOURCC('V', 'E', 'R', 'S'),
};

#undef DXIL_FOURCC

struct DxilShaderFeatureInfo {
  uint64_t FeatureFlags;
};

// DXIL program information.
struct DxilBitcodeHeader {
  uint32_t DxilMagic;       // ACSII "DXIL".
  uint32_t DxilVersion;     // DXIL version.
  uint32_t BitcodeOffset;   // Offset to LLVM bitcode (from start of header).
  uint32_t BitcodeSize;     // Size of LLVM bitcode.
};
static const uint32_t DxilMagicValue = 0x4C495844; // 'DXIL'

struct DxilProgramHeader {
  uint32_t          ProgramVersion;   /// Major and minor version, including type.
  uint32_t          SizeInUint32;     /// Size in uint32_t units including this header.
  DxilBitcodeHeader BitcodeHeader;    /// Bitcode-specific header.
  // Followed by uint8_t[BitcodeHeader.BitcodeOffset]
};

struct DxilProgramSignature {
  uint32_t ParamCount;
  uint32_t ParamOffset;
};

enum class DxilProgramSigMinPrecision : uint32_t {
  Default = 0,
  Float16 = 1,
  Float2_8 = 2,
  Reserved = 3,
  SInt16 = 4,
  UInt16 = 5,
  Any16 = 0xf0,
  Any10 = 0xf1
};

// Corresponds to D3D_NAME and D3D10_SB_NAME
enum class DxilProgramSigSemantic : uint32_t {
  Undefined = 0,
  Position = 1,
  ClipDistance = 2,
  CullDistance = 3,
  RenderTargetArrayIndex = 4,
  ViewPortArrayIndex = 5,
  VertexID = 6,
  PrimitiveID = 7,
  InstanceID = 8,
  IsFrontFace = 9,
  SampleIndex = 10,
  FinalQuadEdgeTessfactor = 11,
  FinalQuadInsideTessfactor = 12,
  FinalTriEdgeTessfactor = 13,
  FinalTriInsideTessfactor = 14,
  FinalLineDetailTessfactor = 15,
  FinalLineDensityTessfactor = 16,
  Barycentrics = 23,
  ShadingRate = 24,
  CullPrimitive = 25,
  Target = 64,
  Depth = 65,
  Coverage = 66,
  DepthGE = 67,
  DepthLE = 68,
  StencilRef = 69,
  InnerCoverage = 70,
};

enum class DxilProgramSigCompType : uint32_t {
  Unknown = 0,
  UInt32 = 1,
  SInt32 = 2,
  Float32 = 3,
  UInt16 = 4,
  SInt16 = 5,
  Float16 = 6,
  UInt64 = 7,
  SInt64 = 8,
  Float64 = 9,
};

struct DxilProgramSignatureElement {
  uint32_t Stream;                    // Stream index (parameters must appear in non-decreasing stream order)
  uint32_t SemanticName;              // Offset to LPCSTR from start of DxilProgramSignature.
  uint32_t SemanticIndex;             // Semantic Index
  DxilProgramSigSemantic SystemValue; // Semantic type. Similar to DxilSemantic::Kind, but a serialized rather than processing rep.
  DxilProgramSigCompType CompType;    // Type of bits.
  uint32_t Register;                  // Register Index (row index)
  uint8_t Mask;                       // Mask (column allocation)
  union                         // Unconditional cases useful for validation of shader linkage.
  {
    uint8_t NeverWrites_Mask;   // For an output signature, the shader the signature belongs to never
                                // writes the masked components of the output register.
    uint8_t AlwaysReads_Mask;   // For an input signature, the shader the signature belongs to always
                                // reads the masked components of the input register.
  };
  uint16_t Pad;
  DxilProgramSigMinPrecision MinPrecision; // Minimum precision of input/output data
};

// Easy to get this wrong. Earlier assertions can help determine
static_assert(sizeof(DxilProgramSignatureElement) == 0x20, "else DxilProgramSignatureElement is misaligned");

struct DxilShaderDebugName {
  uint16_t Flags;       // Reserved, must be set to zero.
  uint16_t NameLength;  // Length of the debug name, without null terminator.
  // Followed by NameLength bytes of the UTF-8-encoded name.
  // Followed by a null terminator.
  // Followed by [0-3] zero bytes to align to a 4-byte boundary.
};
static const size_t MinDxilShaderDebugNameSize = sizeof(DxilShaderDebugName) + 4;

struct DxilCompilerVersion {
  uint16_t Major;
  uint16_t Minor;
  uint32_t VersionFlags;
  uint32_t CommitCount;
  uint32_t VersionStringListSizeInBytes;
  // Followed by VersionStringListSizeInBytes bytes, containing up to two null-terminated strings, sequentially:
  //  1. CommitSha
  //  1. CustomVersionString
  // Followed by [0-3] zero bytes to align to a 4-byte boundary.
};

// Source Info part has the following top level structure:
//
//   DxilSourceInfo
//
//      DxilSourceInfoSection
//         char Data[]
//         (0-3 zero bytes to align to a 4-byte boundary)
//
//      DxilSourceInfoSection
//         char Data[]
//         (0-3 zero bytes to align to a 4-byte boundary)
//
//      ...
//
//      DxilSourceInfoSection
//         char Data[]
//         (0-3 zero bytes to align to a 4-byte boundary)
//
// Each DxilSourceInfoSection is followed by a blob of Data.
// The each type of data has its own internal structure:
//
// ================ 1. Source Names ==================================
//
//  DxilSourceInfo_SourceNames
//
//     DxilSourceInfo_SourceNamesEntry
//        char Name[ NameSizeInBytes ]
//        (0-3 zero bytes to align to a 4-byte boundary)
//
//     DxilSourceInfo_SourceNamesEntry
//        char Name[ NameSizeInBytes ]
//        (0-3 zero bytes to align to a 4-byte boundary)
//
//      ...
//
//     DxilSourceInfo_SourceNamesEntry
//        char Name[ NameSizeInBytes ]
//        (0-3 zero bytes to align to a 4-byte boundary)
//
// ================ 2. Source Contents ==================================
// 
//  DxilSourceInfo_SourceContents
//    char Entries[CompressedEntriesSizeInBytes]
//   
// `Entries` may be compressed. Here is the uncompressed structure:
//
//     DxilSourceInfo_SourcesContentsEntry
//        char Content[ ContentSizeInBytes ]
//        (0-3 zero bytes to align to a 4-byte boundary)
//
//     DxilSourceInfo_SourcesContentsEntry
//        char Content[ ContentSizeInBytes ]
//        (0-3 zero bytes to align to a 4-byte boundary)
//
//     ...
//
//     DxilSourceInfo_SourcesContentsEntry
//        char Content[ ContentSizeInBytes ]
//        (0-3 zero bytes to align to a 4-byte boundary)
//
// ================ 3. Args ==================================
//
//   DxilSourceInfo_Args
//
//      char ArgName[]; char NullTerm;
//      char ArgValue[]; char NullTerm;
//
//      char ArgName[]; char NullTerm;
//      char ArgValue[]; char NullTerm;
//
//      ...
//
//      char ArgName[]; char NullTerm;
//      char ArgValue[]; char NullTerm;
//

struct DxilSourceInfo {
  uint32_t AlignedSizeInBytes;  // Total size of the contents including this header
  uint16_t Flags;               // Reserved, must be set to zero.
  uint16_t SectionCount;        // The number of sections in the source info.
};

enum class DxilSourceInfoSectionType : uint16_t {
  SourceContents = 0,
  SourceNames    = 1,
  Args           = 2,
};

struct DxilSourceInfoSection {
  uint32_t AlignedSizeInBytes;      // Size of the section, including this header, and the padding. Aligned to 4-byte boundary.
  uint16_t Flags;                   // Reserved, must be set to zero.
  DxilSourceInfoSectionType Type;   // The type of data following this header.
};

struct DxilSourceInfo_Args {
  uint32_t Flags;       // Reserved, must be set to zero.
  uint32_t SizeInBytes; // Length of all argument pairs, including their null terminators, not including this header.
  uint32_t Count;       // Number of arguments.

  // Followed by `Count` argument pairs.
  //
  // For example, given the following arguments:
  //    /T ps_6_0 -EMain -D MyDefine=1 /DMyOtherDefine=2 -Zi MyShader.hlsl
  //
  // The argument pair data becomes:
  //    T\0ps_6_0\0
  //    E\0Main\0
  //    D\0MyDefine=1\0
  //    D\0MyOtherDefine=2\0
  //    Zi\0\0
  //    \0MyShader.hlsl\0
  //
};

struct DxilSourceInfo_SourceNames {
  uint32_t Flags;                                   // Reserved, must be set to 0.
  uint32_t Count;                                   // The number of data entries
  uint16_t EntriesSizeInBytes;                      // The total size of the data entries following this header.

  // Followed by `Count` data entries with the header DxilSourceInfo_SourceNamesEntry
};

struct DxilSourceInfo_SourceNamesEntry {
  uint32_t AlignedSizeInBytes;                      // Size of the data including this header and padding. Aligned to 4-byte boundary.
  uint32_t Flags;                                   // Reserved, must be set to 0.
  uint32_t NameSizeInBytes;                         // Size of the file name, *including* the null terminator.
  uint32_t ContentSizeInBytes;                      // Size of the file content, *including* the null terminator.
  // Followed by NameSizeInBytes bytes of the UTF-8-encoded file name (including null terminator).
  // Followed by [0-3] zero bytes to align to a 4-byte boundary.
};

enum class DxilSourceInfo_SourceContentsCompressType : uint16_t {
  None,
  Zlib
};

struct DxilSourceInfo_SourceContents {
  uint32_t AlignedSizeInBytes;                             // Size of the entry including this header. Aligned to 4-byte boundary.
  uint16_t Flags;                                          // Reserved, must be set to 0.
  DxilSourceInfo_SourceContentsCompressType CompressType;  // The type of compression used to compress the data
  uint32_t EntriesSizeInBytes;                             // The size of the data entries following this header.
  uint32_t UncompressedEntriesSizeInBytes;                 // Total size of the data entries when uncompressed.
  uint32_t Count;                                          // The number of data entries
  // Followed by (compressed) `Count` data entries with the header DxilSourceInfo_SourceContentsEntry
};

struct DxilSourceInfo_SourceContentsEntry {
  uint32_t AlignedSizeInBytes;                             // Size of the entry including this header and padding. Aligned to 4-byte boundary.
  uint32_t Flags;                                          // Reserved, must be set to 0.
  uint32_t ContentSizeInBytes;                             // Size of the data following this header, *including* the null terminator
  // Followed by ContentSizeInBytes bytes of the UTF-8-encoded content (including null terminator).
  // Followed by [0-3] zero bytes to align to a 4-byte boundary.
};

#pragma pack(pop)

/// Gets a part header by index.
inline const DxilPartHeader *
GetDxilContainerPart(const DxilContainerHeader *pHeader, uint32_t index) {
  const uint8_t *pLinearContainer = reinterpret_cast<const uint8_t *>(pHeader);
  const uint32_t *pPartOffsetTable =
      reinterpret_cast<const uint32_t *>(pHeader + 1);
  return reinterpret_cast<const DxilPartHeader *>(
      pLinearContainer + pPartOffsetTable[index]);
}

/// Gets a part header by index.
inline DxilPartHeader *GetDxilContainerPart(DxilContainerHeader *pHeader,
                                            uint32_t index) {
  return const_cast<DxilPartHeader *>(GetDxilContainerPart(
      reinterpret_cast<const DxilContainerHeader *>(pHeader), index));
}

/// Gets the part data from the header.
inline const char *GetDxilPartData(const DxilPartHeader *pPart) {
  return reinterpret_cast<const char *>(pPart + 1);
}

/// Gets the part data from the header.
inline char *GetDxilPartData(DxilPartHeader *pPart) {
  return reinterpret_cast<char *>(pPart + 1);
}
/// Gets a part header by fourCC
DxilPartHeader *GetDxilPartByType(DxilContainerHeader *pHeader,
                                           DxilFourCC fourCC);
/// Gets a part header by fourCC 
const DxilPartHeader *
GetDxilPartByType(const DxilContainerHeader *pHeader,
                           DxilFourCC fourCC);

/// Returns valid DxilProgramHeader. nullptr if does not exist.
DxilProgramHeader *GetDxilProgramHeader(DxilContainerHeader *pHeader, DxilFourCC fourCC);

/// Returns valid DxilProgramHeader. nullptr if does not exist.
const DxilProgramHeader *
GetDxilProgramHeader(const DxilContainerHeader *pHeader, DxilFourCC fourCC);

/// Initializes container with the specified values.
void InitDxilContainer(_Out_ DxilContainerHeader *pHeader, uint32_t partCount,
                       uint32_t containerSizeInBytes);

/// Checks whether pHeader claims by signature to be a DXIL container
/// and the length is at least sizeof(DxilContainerHeader).
const DxilContainerHeader *IsDxilContainerLike(const void *ptr, size_t length);
DxilContainerHeader *IsDxilContainerLike(void *ptr, size_t length);

/// Checks whether the DXIL container is valid and in-bounds.
bool IsValidDxilContainer(const DxilContainerHeader *pHeader, size_t length);

/// Use this type as a unary predicate functor.
struct DxilPartIsType {
  uint32_t IsFourCC;
  DxilPartIsType(uint32_t FourCC) : IsFourCC(FourCC) { }
  bool operator()(const DxilPartHeader *pPart) const {
    return pPart->PartFourCC == IsFourCC;
  }
};

/// Use this type as an iterator over the part headers.
struct DxilPartIterator : public std::iterator<std::input_iterator_tag,
                                               const DxilContainerHeader *> {
  const DxilContainerHeader *pHeader;
  uint32_t index;

  DxilPartIterator(const DxilContainerHeader *h, uint32_t i)
      : pHeader(h), index(i) {}

  // increment
  DxilPartIterator &operator++() {
    ++index;
    return *this;
  }
  DxilPartIterator operator++(int) {
    DxilPartIterator result(pHeader, index);
    ++index;
    return result;
  }

  // input iterator - compare and deref
  bool operator==(const DxilPartIterator &other) const {
    return index == other.index && pHeader == other.pHeader;
  }
  bool operator!=(const DxilPartIterator &other) const {
    return index != other.index || pHeader != other.pHeader;
  }
  const DxilPartHeader *operator*() const {
    return GetDxilContainerPart(pHeader, index);
  }
};

DxilPartIterator begin(const DxilContainerHeader *pHeader);
DxilPartIterator end(const DxilContainerHeader *pHeader);

inline bool IsValidDxilBitcodeHeader(const DxilBitcodeHeader *pHeader,
                                     uint32_t length) {
  return length > sizeof(DxilBitcodeHeader) &&
         pHeader->BitcodeOffset + pHeader->BitcodeSize >
             pHeader->BitcodeOffset &&
         length >= pHeader->BitcodeOffset + pHeader->BitcodeSize &&
         pHeader->DxilMagic == DxilMagicValue;
}

inline void InitBitcodeHeader(DxilBitcodeHeader &header,
  uint32_t dxilVersion,
  uint32_t bitcodeSize) {
  header.DxilMagic = DxilMagicValue;
  header.DxilVersion = dxilVersion;
  header.BitcodeOffset = sizeof(DxilBitcodeHeader);
  header.BitcodeSize = bitcodeSize;
}

inline void GetDxilProgramBitcode(const DxilProgramHeader *pHeader,
                                  const char **pBitcode,
                                  uint32_t *pBitcodeLength) {
  *pBitcode = reinterpret_cast<const char *>(&pHeader->BitcodeHeader) +
              pHeader->BitcodeHeader.BitcodeOffset;
  *pBitcodeLength = pHeader->BitcodeHeader.BitcodeSize;
}

inline bool IsValidDxilProgramHeader(const DxilProgramHeader *pHeader,
  uint32_t length) {
  return length >= sizeof(DxilProgramHeader) &&
    length >= (pHeader->SizeInUint32 * sizeof(uint32_t)) &&
    IsValidDxilBitcodeHeader(
      &pHeader->BitcodeHeader,
      length - offsetof(DxilProgramHeader, BitcodeHeader));
}

inline void InitProgramHeader(DxilProgramHeader &header, uint32_t shaderVersion,
                              uint32_t dxilVersion,
                              uint32_t bitcodeSize) {
  header.ProgramVersion = shaderVersion;
  header.SizeInUint32 =
    sizeof(DxilProgramHeader) / sizeof(uint32_t) +
    bitcodeSize / sizeof(uint32_t) + ((bitcodeSize % 4) ? 1 : 0);
  InitBitcodeHeader(header.BitcodeHeader, dxilVersion, bitcodeSize);
}

inline const char *GetDxilBitcodeData(const DxilProgramHeader *pHeader) {
  const DxilBitcodeHeader *pBCHdr = &(pHeader->BitcodeHeader);
  return (const char *)pBCHdr + pBCHdr->BitcodeOffset;
}

inline uint32_t GetDxilBitcodeSize(const DxilProgramHeader *pHeader) {
  return pHeader->BitcodeHeader.BitcodeSize;
}

/// Extract the shader type from the program version value.
inline DXIL::ShaderKind GetVersionShaderType(uint32_t programVersion) {
  return (DXIL::ShaderKind)((programVersion & 0xffff0000) >> 16);
}
inline uint32_t GetVersionMajor(uint32_t programVersion) {
  return (programVersion & 0xf0) >> 4;
}
inline uint32_t GetVersionMinor(uint32_t programVersion) {
  return (programVersion & 0xf);
}
inline uint32_t EncodeVersion(DXIL::ShaderKind shaderType, uint32_t major,
  uint32_t minor) {
  return ((unsigned)shaderType << 16) | (major << 4) | minor;
}

inline bool IsDxilShaderDebugNameValid(const DxilPartHeader *pPart) {
  if (pPart->PartFourCC != DFCC_ShaderDebugName) return false;
  if (pPart->PartSize < MinDxilShaderDebugNameSize) return false;
  const DxilShaderDebugName *pDebugNameContent = reinterpret_cast<const DxilShaderDebugName *>(GetDxilPartData(pPart));
  uint16_t ExpectedSize = sizeof(DxilShaderDebugName) + pDebugNameContent->NameLength + 1;
  if (ExpectedSize & 0x3) {
    ExpectedSize += 0x4;
    ExpectedSize &= ~(0x3);
  }
  if (pPart->PartSize != ExpectedSize) return false;
  return true;
}

inline bool GetDxilShaderDebugName(const DxilPartHeader *pDebugNamePart,
  const char **ppUtf8Name, _Out_opt_ uint16_t *pUtf8NameLen) {
  *ppUtf8Name = nullptr;
  if (!IsDxilShaderDebugNameValid(pDebugNamePart)) {
    return false;
  }
  const DxilShaderDebugName *pDebugNameContent = reinterpret_cast<const DxilShaderDebugName *>(GetDxilPartData(pDebugNamePart));
  if (pUtf8NameLen) {
    *pUtf8NameLen = pDebugNameContent->NameLength;
  }
  *ppUtf8Name = (const char *)(pDebugNameContent + 1);
  return true;
}

enum class SerializeDxilFlags : uint32_t {
  None                        = 0,      // No flags defined.
  IncludeDebugInfoPart        = 1 << 0, // Include the debug info part in the container.
  IncludeDebugNamePart        = 1 << 1, // Include the debug name part in the container.
  DebugNameDependOnSource     = 1 << 2, // Make the debug name depend on source (and not just final module).
  StripReflectionFromDxilPart = 1 << 3, // Strip Reflection info from DXIL part.
  IncludeReflectionPart       = 1 << 4, // Include reflection in STAT part.
  StripRootSignature          = 1 << 5, // Strip Root Signature from main shader container.
};
inline SerializeDxilFlags& operator |=(SerializeDxilFlags& l, const SerializeDxilFlags& r) {
  l = static_cast<SerializeDxilFlags>(static_cast<int>(l) | static_cast<int>(r));
  return l;
}
inline SerializeDxilFlags& operator &=(SerializeDxilFlags& l, const SerializeDxilFlags& r) {
  l = static_cast<SerializeDxilFlags>(static_cast<int>(l) & static_cast<int>(r));
  return l;
}
inline int operator&(SerializeDxilFlags l, SerializeDxilFlags r) {
  return static_cast<int>(l) & static_cast<int>(r);
}
inline SerializeDxilFlags operator~(SerializeDxilFlags l) {
  return static_cast<SerializeDxilFlags>(~static_cast<uint32_t>(l));
}

void CreateDxcContainerReflection(IDxcContainerReflection **ppResult);

// Converts uint32_t partKind to char array object.
inline char * PartKindToCharArray(uint32_t partKind, _Out_writes_(5) char* pText) {
  pText[0] = (char)((partKind & 0x000000FF) >> 0);
  pText[1] = (char)((partKind & 0x0000FF00) >> 8);
  pText[2] = (char)((partKind & 0x00FF0000) >> 16);
  pText[3] = (char)((partKind & 0xFF000000) >> 24);
  pText[4] = '\0';
  return pText;
}

inline size_t GetOffsetTableSize(uint32_t partCount) {
  return sizeof(uint32_t) * partCount;
}
// Compute total size of the dxil container from parts information
inline size_t GetDxilContainerSizeFromParts(uint32_t partCount, uint32_t partsSize) {
  return partsSize + (uint32_t)sizeof(DxilContainerHeader) +
         GetOffsetTableSize(partCount) +
         (uint32_t)sizeof(DxilPartHeader) * partCount;
}

} // namespace hlsl

#endif // __DXC_CONTAINER__
