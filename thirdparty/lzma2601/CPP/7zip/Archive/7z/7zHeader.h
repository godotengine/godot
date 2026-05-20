// 7z/7zHeader.h

#ifndef ZIP7_INC_7Z_HEADER_H
#define ZIP7_INC_7Z_HEADER_H

#include "../../../Common/MyTypes.h"

namespace NArchive {
namespace N7z {

const unsigned kSignatureSize = 6;
extern Byte kSignature[kSignatureSize];

// #define Z7_7Z_VOL
// 7z-MultiVolume is not finished yet.
// It can work already, but I still do not like some
// things of that new multivolume format.
// So please keep it commented.

#ifdef Z7_7Z_VOL
extern Byte kFinishSignature[kSignatureSize];
#endif

struct CArchiveVersion
{
  Byte Major;
  Byte Minor;
};

const Byte kMajorVersion = 0;

struct CStartHeader
{
  UInt64 NextHeaderOffset;
  UInt64 NextHeaderSize;
  UInt32 NextHeaderCRC;
};

const UInt32 kStartHeaderSize = 20;

#ifdef Z7_7Z_VOL
struct CFinishHeader: public CStartHeader
{
  UInt64 ArchiveStartOffset;  // data offset from end if that struct
  UInt64 AdditionalStartBlockSize; // start  signature & start header size
};

const UInt32 kFinishHeaderSize = kStartHeaderSize + 16;
#endif

namespace NID
{
  enum EEnum
  {
    kEnd,

    kHeader,

    kArchiveProperties,
    
    kAdditionalStreamsInfo,
    kMainStreamsInfo,
    kFilesInfo,
    
    kPackInfo,
    kUnpackInfo,
    kSubStreamsInfo,

    kSize,
    kCRC,

    kFolder,

    kCodersUnpackSize,
    kNumUnpackStream,

    kEmptyStream,
    kEmptyFile,
    kAnti,

    kName,
    kCTime,
    kATime,
    kMTime,
    kWinAttrib,
    kComment,

    kEncodedHeader,

    kStartPos,
    kDummy

    // kNtSecure,
    // kParent,
    // kIsAux
  };
}


const UInt32 k_Copy = 0;
const UInt32 k_Delta = 3;
const UInt32 k_ARM64 = 0xa;
const UInt32 k_RISCV = 0xb;

const UInt32 k_LZMA2 = 0x21;

const UInt32 k_SWAP2 = 0x20302;
const UInt32 k_SWAP4 = 0x20304;

const UInt32 k_LZMA  = 0x30101;
const UInt32 k_PPMD  = 0x30401;

const UInt32 k_Deflate   = 0x40108;
const UInt32 k_Deflate64 = 0x40109;
const UInt32 k_BZip2     = 0x40202;

const UInt32 k_BCJ   = 0x3030103;
const UInt32 k_BCJ2  = 0x303011B;
const UInt32 k_PPC   = 0x3030205;
const UInt32 k_IA64  = 0x3030401;
const UInt32 k_ARM   = 0x3030501;
const UInt32 k_ARMT  = 0x3030701;
const UInt32 k_SPARC = 0x3030805;

const UInt32 k_AES   = 0x6F10701;

// const UInt32 k_ZSTD = 0x4015D; // winzip zstd
// 0x4F71101, 7z-zstd

inline bool IsFilterMethod(UInt64 m)
{
  if (m > (UInt32)0xFFFFFFFF)
    return false;
  switch ((UInt32)m)
  {
    case k_Delta:
    case k_ARM64:
    case k_RISCV:
    case k_BCJ:
    case k_BCJ2:
    case k_PPC:
    case k_IA64:
    case k_ARM:
    case k_ARMT:
    case k_SPARC:
    case k_SWAP2:
    case k_SWAP4:
      return true;
    default: break;
  }
  return false;
}

}}

#endif
