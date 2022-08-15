//===-- TargetParser - Parser for target features ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a target parser to recognise hardware features such as
// FPU/CPU/ARCH names as well as specific support such as HDIV, etc.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_TARGETPARSER_H
#define LLVM_SUPPORT_TARGETPARSER_H

// FIXME: vector is used because that's what clang uses for subtarget feature
// lists, but SmallVector would probably be better
#include <vector>

namespace llvm {
  class StringRef;

// Target specific information into their own namespaces. These should be
// generated from TableGen because the information is already there, and there
// is where new information about targets will be added.
// FIXME: To TableGen this we need to make some table generated files available
// even if the back-end is not compiled with LLVM, plus we need to create a new
// back-end to TableGen to create these clean tables.
namespace ARM {
  // FPU names.
  enum FPUKind {
    FK_INVALID = 0,
    FK_NONE,
    FK_VFP,
    FK_VFPV2,
    FK_VFPV3,
    FK_VFPV3_FP16,
    FK_VFPV3_D16,
    FK_VFPV3_D16_FP16,
    FK_VFPV3XD,
    FK_VFPV3XD_FP16,
    FK_VFPV4,
    FK_VFPV4_D16,
    FK_FPV4_SP_D16,
    FK_FPV5_D16,
    FK_FPV5_SP_D16,
    FK_FP_ARMV8,
    FK_NEON,
    FK_NEON_FP16,
    FK_NEON_VFPV4,
    FK_NEON_FP_ARMV8,
    FK_CRYPTO_NEON_FP_ARMV8,
    FK_SOFTVFP,
    FK_LAST
  };

  // FPU Version
  enum FPUVersion {
    FV_NONE = 0,
    FV_VFPV2,
    FV_VFPV3,
    FV_VFPV3_FP16,
    FV_VFPV4,
    FV_VFPV5
  };

  // An FPU name implies one of three levels of Neon support:
  enum NeonSupportLevel {
    NS_None = 0, ///< No Neon
    NS_Neon,     ///< Neon
    NS_Crypto    ///< Neon with Crypto
  };

  // An FPU name restricts the FPU in one of three ways:
  enum FPURestriction {
    FR_None = 0, ///< No restriction
    FR_D16,      ///< Only 16 D registers
    FR_SP_D16    ///< Only single-precision instructions, with 16 D registers
  };

  // Arch names.
  enum ArchKind {
    AK_INVALID = 0,
    AK_ARMV2,
    AK_ARMV2A,
    AK_ARMV3,
    AK_ARMV3M,
    AK_ARMV4,
    AK_ARMV4T,
    AK_ARMV5T,
    AK_ARMV5TE,
    AK_ARMV5TEJ,
    AK_ARMV6,
    AK_ARMV6K,
    AK_ARMV6T2,
    AK_ARMV6Z,
    AK_ARMV6ZK,
    AK_ARMV6M,
    AK_ARMV6SM,
    AK_ARMV7A,
    AK_ARMV7R,
    AK_ARMV7M,
    AK_ARMV7EM,
    AK_ARMV8A,
    AK_ARMV8_1A,
    // Non-standard Arch names.
    AK_IWMMXT,
    AK_IWMMXT2,
    AK_XSCALE,
    AK_ARMV5,
    AK_ARMV5E,
    AK_ARMV6J,
    AK_ARMV6HL,
    AK_ARMV7,
    AK_ARMV7L,
    AK_ARMV7HL,
    AK_ARMV7S,
    AK_LAST
  };

  // Arch extension modifiers for CPUs.
  enum ArchExtKind {
    AEK_INVALID = 0,
    AEK_CRC,
    AEK_CRYPTO,
    AEK_FP,
    AEK_HWDIV,
    AEK_MP,
    AEK_SIMD,
    AEK_SEC,
    AEK_VIRT,
    // Unsupported extensions.
    AEK_OS,
    AEK_IWMMXT,
    AEK_IWMMXT2,
    AEK_MAVERICK,
    AEK_XSCALE,
    AEK_LAST
  };

  // ISA kinds.
  enum ISAKind {
    IK_INVALID = 0,
    IK_ARM,
    IK_THUMB,
    IK_AARCH64
  };

  // Endianness
  // FIXME: BE8 vs. BE32?
  enum EndianKind {
    EK_INVALID = 0,
    EK_LITTLE,
    EK_BIG
  };

  // v6/v7/v8 Profile
  enum ProfileKind {
    PK_INVALID = 0,
    PK_A,
    PK_R,
    PK_M
  };
} // namespace ARM

// Target Parsers, one per architecture.
class ARMTargetParser {
  static StringRef getFPUSynonym(StringRef FPU);
  static StringRef getArchSynonym(StringRef Arch);

public:
  static StringRef getCanonicalArchName(StringRef Arch);

  // Information by ID
  static const char * getFPUName(unsigned FPUKind);
  static     unsigned getFPUVersion(unsigned FPUKind);
  static     unsigned getFPUNeonSupportLevel(unsigned FPUKind);
  static     unsigned getFPURestriction(unsigned FPUKind);
  // FIXME: This should be moved to TargetTuple once it exists
  static       bool   getFPUFeatures(unsigned FPUKind,
                                     std::vector<const char*> &Features);
  static const char * getArchName(unsigned ArchKind);
  static   unsigned   getArchAttr(unsigned ArchKind);
  static const char * getCPUAttr(unsigned ArchKind);
  static const char * getSubArch(unsigned ArchKind);
  static const char * getArchExtName(unsigned ArchExtKind);
  static const char * getDefaultCPU(StringRef Arch);

  // Parser
  static unsigned parseFPU(StringRef FPU);
  static unsigned parseArch(StringRef Arch);
  static unsigned parseArchExt(StringRef ArchExt);
  static unsigned parseCPUArch(StringRef CPU);
  static unsigned parseArchISA(StringRef Arch);
  static unsigned parseArchEndian(StringRef Arch);
  static unsigned parseArchProfile(StringRef Arch);
  static unsigned parseArchVersion(StringRef Arch);

};

} // namespace llvm

#endif
