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

#include "llvm/Support/ARMBuildAttributes.h"
#include "llvm/Support/TargetParser.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include <cctype>

using namespace llvm;

namespace {

// List of canonical FPU names (use getFPUSynonym) and which architectural
// features they correspond to (use getFPUFeatures).
// FIXME: TableGen this.
// The entries must appear in the order listed in ARM::FPUKind for correct indexing
struct {
  const char * Name;
  ARM::FPUKind ID;
  ARM::FPUVersion FPUVersion;
  ARM::NeonSupportLevel NeonSupport;
  ARM::FPURestriction Restriction;
} FPUNames[] = {
  { "invalid",        ARM::FK_INVALID,        ARM::FV_NONE,       ARM::NS_None,   ARM::FR_None},
  { "none",           ARM::FK_NONE,           ARM::FV_NONE,       ARM::NS_None,   ARM::FR_None},
  { "vfp",            ARM::FK_VFP,            ARM::FV_VFPV2,      ARM::NS_None,   ARM::FR_None},
  { "vfpv2",          ARM::FK_VFPV2,          ARM::FV_VFPV2,      ARM::NS_None,   ARM::FR_None},
  { "vfpv3",          ARM::FK_VFPV3,          ARM::FV_VFPV3,      ARM::NS_None,   ARM::FR_None},
  { "vfpv3-fp16",     ARM::FK_VFPV3_FP16,     ARM::FV_VFPV3_FP16, ARM::NS_None,   ARM::FR_None},
  { "vfpv3-d16",      ARM::FK_VFPV3_D16,      ARM::FV_VFPV3,      ARM::NS_None,   ARM::FR_D16},
  { "vfpv3-d16-fp16", ARM::FK_VFPV3_D16_FP16, ARM::FV_VFPV3_FP16, ARM::NS_None,   ARM::FR_D16},
  { "vfpv3xd",        ARM::FK_VFPV3XD,        ARM::FV_VFPV3,      ARM::NS_None,   ARM::FR_SP_D16},
  { "vfpv3xd-fp16",   ARM::FK_VFPV3XD_FP16,   ARM::FV_VFPV3_FP16, ARM::NS_None,   ARM::FR_SP_D16},
  { "vfpv4",          ARM::FK_VFPV4,          ARM::FV_VFPV4,      ARM::NS_None,   ARM::FR_None},
  { "vfpv4-d16",      ARM::FK_VFPV4_D16,      ARM::FV_VFPV4,      ARM::NS_None,   ARM::FR_D16},
  { "fpv4-sp-d16",    ARM::FK_FPV4_SP_D16,    ARM::FV_VFPV4,      ARM::NS_None,   ARM::FR_SP_D16},
  { "fpv5-d16",       ARM::FK_FPV5_D16,       ARM::FV_VFPV5,      ARM::NS_None,   ARM::FR_D16},
  { "fpv5-sp-d16",    ARM::FK_FPV5_SP_D16,    ARM::FV_VFPV5,      ARM::NS_None,   ARM::FR_SP_D16},
  { "fp-armv8",       ARM::FK_FP_ARMV8,       ARM::FV_VFPV5,      ARM::NS_None,   ARM::FR_None},
  { "neon",           ARM::FK_NEON,           ARM::FV_VFPV3,      ARM::NS_Neon,   ARM::FR_None},
  { "neon-fp16",      ARM::FK_NEON_FP16,      ARM::FV_VFPV3_FP16, ARM::NS_Neon,   ARM::FR_None},
  { "neon-vfpv4",     ARM::FK_NEON_VFPV4,     ARM::FV_VFPV4,      ARM::NS_Neon,   ARM::FR_None},
  { "neon-fp-armv8",  ARM::FK_NEON_FP_ARMV8,  ARM::FV_VFPV5,      ARM::NS_Neon,   ARM::FR_None},
  { "crypto-neon-fp-armv8",
               ARM::FK_CRYPTO_NEON_FP_ARMV8,  ARM::FV_VFPV5,      ARM::NS_Crypto, ARM::FR_None},
  { "softvfp",        ARM::FK_SOFTVFP,        ARM::FV_NONE,       ARM::NS_None,   ARM::FR_None},
};

// List of canonical arch names (use getArchSynonym).
// This table also provides the build attribute fields for CPU arch
// and Arch ID, according to the Addenda to the ARM ABI, chapters
// 2.4 and 2.3.5.2 respectively.
// FIXME: SubArch values were simplified to fit into the expectations
// of the triples and are not conforming with their official names.
// Check to see if the expectation should be changed.
// FIXME: TableGen this.
struct {
  const char *Name;
  ARM::ArchKind ID;
  const char *CPUAttr; // CPU class in build attributes.
  const char *SubArch; // Sub-Arch name.
  ARMBuildAttrs::CPUArch ArchAttr; // Arch ID in build attributes.
} ARCHNames[] = {
  { "invalid",   ARM::AK_INVALID,  nullptr,   nullptr, ARMBuildAttrs::CPUArch::Pre_v4 },
  { "armv2",     ARM::AK_ARMV2,    "2",       "v2",    ARMBuildAttrs::CPUArch::Pre_v4 },
  { "armv2a",    ARM::AK_ARMV2A,   "2A",      "v2a",   ARMBuildAttrs::CPUArch::Pre_v4 },
  { "armv3",     ARM::AK_ARMV3,    "3",       "v3",    ARMBuildAttrs::CPUArch::Pre_v4 },
  { "armv3m",    ARM::AK_ARMV3M,   "3M",      "v3m",   ARMBuildAttrs::CPUArch::Pre_v4 },
  { "armv4",     ARM::AK_ARMV4,    "4",       "v4",    ARMBuildAttrs::CPUArch::v4 },
  { "armv4t",    ARM::AK_ARMV4T,   "4T",      "v4t",   ARMBuildAttrs::CPUArch::v4T },
  { "armv5t",    ARM::AK_ARMV5T,   "5T",      "v5",    ARMBuildAttrs::CPUArch::v5T },
  { "armv5te",   ARM::AK_ARMV5TE,  "5TE",     "v5e",   ARMBuildAttrs::CPUArch::v5TE },
  { "armv5tej",  ARM::AK_ARMV5TEJ, "5TEJ",    "v5e",   ARMBuildAttrs::CPUArch::v5TEJ },
  { "armv6",     ARM::AK_ARMV6,    "6",       "v6",    ARMBuildAttrs::CPUArch::v6 },
  { "armv6k",    ARM::AK_ARMV6K,   "6K",      "v6k",   ARMBuildAttrs::CPUArch::v6K },
  { "armv6t2",   ARM::AK_ARMV6T2,  "6T2",     "v6t2",  ARMBuildAttrs::CPUArch::v6T2 },
  { "armv6z",    ARM::AK_ARMV6Z,   "6Z",      "v6z",   ARMBuildAttrs::CPUArch::v6KZ },
  { "armv6zk",   ARM::AK_ARMV6ZK,  "6ZK",     "v6zk",  ARMBuildAttrs::CPUArch::v6KZ },
  { "armv6-m",   ARM::AK_ARMV6M,   "6-M",     "v6m",   ARMBuildAttrs::CPUArch::v6_M },
  { "armv6s-m",  ARM::AK_ARMV6SM,  "6S-M",    "v6sm",  ARMBuildAttrs::CPUArch::v6S_M },
  { "armv7-a",   ARM::AK_ARMV7A,   "7-A",     "v7",    ARMBuildAttrs::CPUArch::v7 },
  { "armv7-r",   ARM::AK_ARMV7R,   "7-R",     "v7r",   ARMBuildAttrs::CPUArch::v7 },
  { "armv7-m",   ARM::AK_ARMV7M,   "7-M",     "v7m",   ARMBuildAttrs::CPUArch::v7 },
  { "armv7e-m",  ARM::AK_ARMV7EM,  "7E-M",    "v7em",  ARMBuildAttrs::CPUArch::v7E_M },
  { "armv8-a",   ARM::AK_ARMV8A,   "8-A",     "v8",    ARMBuildAttrs::CPUArch::v8 },
  { "armv8.1-a", ARM::AK_ARMV8_1A, "8.1-A",   "v8.1a", ARMBuildAttrs::CPUArch::v8 },
  // Non-standard Arch names.
  { "iwmmxt",    ARM::AK_IWMMXT,   "iwmmxt",  "",      ARMBuildAttrs::CPUArch::v5TE },
  { "iwmmxt2",   ARM::AK_IWMMXT2,  "iwmmxt2", "",      ARMBuildAttrs::CPUArch::v5TE },
  { "xscale",    ARM::AK_XSCALE,   "xscale",  "",      ARMBuildAttrs::CPUArch::v5TE },
  { "armv5",     ARM::AK_ARMV5,    "5T",      "v5",    ARMBuildAttrs::CPUArch::v5T },
  { "armv5e",    ARM::AK_ARMV5E,   "5TE",     "v5e",   ARMBuildAttrs::CPUArch::v5TE },
  { "armv6j",    ARM::AK_ARMV6J,   "6J",      "v6",    ARMBuildAttrs::CPUArch::v6 },
  { "armv6hl",   ARM::AK_ARMV6HL,  "6-M",     "v6hl",  ARMBuildAttrs::CPUArch::v6_M },
  { "armv7",     ARM::AK_ARMV7,    "7",       "v7",    ARMBuildAttrs::CPUArch::v7 },
  { "armv7l",    ARM::AK_ARMV7L,   "7-L",     "v7l",   ARMBuildAttrs::CPUArch::v7 },
  { "armv7hl",   ARM::AK_ARMV7HL,  "7-L",     "v7hl",  ARMBuildAttrs::CPUArch::v7 },
  { "armv7s",    ARM::AK_ARMV7S,   "7-S",     "v7s",   ARMBuildAttrs::CPUArch::v7 }
};
// List of Arch Extension names.
// FIXME: TableGen this.
struct {
  const char *Name;
  ARM::ArchExtKind ID;
} ARCHExtNames[] = {
  { "invalid",  ARM::AEK_INVALID },
  { "crc",      ARM::AEK_CRC },
  { "crypto",   ARM::AEK_CRYPTO },
  { "fp",       ARM::AEK_FP },
  { "idiv",     ARM::AEK_HWDIV },
  { "mp",       ARM::AEK_MP },
  { "simd",     ARM::AEK_SIMD },
  { "sec",      ARM::AEK_SEC },
  { "virt",     ARM::AEK_VIRT },
  { "os",       ARM::AEK_OS },
  { "iwmmxt",   ARM::AEK_IWMMXT },
  { "iwmmxt2",  ARM::AEK_IWMMXT2 },
  { "maverick", ARM::AEK_MAVERICK },
  { "xscale",   ARM::AEK_XSCALE }
};
// List of CPU names and their arches.
// The same CPU can have multiple arches and can be default on multiple arches.
// When finding the Arch for a CPU, first-found prevails. Sort them accordingly.
// When this becomes table-generated, we'd probably need two tables.
// FIXME: TableGen this.
struct {
  const char *Name;
  ARM::ArchKind ArchID;
  bool Default;
} CPUNames[] = {
  { "arm2",          ARM::AK_ARMV2,    true },
  { "arm3",          ARM::AK_ARMV2A,   true },
  { "arm6",          ARM::AK_ARMV3,    true },
  { "arm7m",         ARM::AK_ARMV3M,   true },
  { "arm8",          ARM::AK_ARMV4,    false },
  { "arm810",        ARM::AK_ARMV4,    false },
  { "strongarm",     ARM::AK_ARMV4,    true },
  { "strongarm110",  ARM::AK_ARMV4,    false },
  { "strongarm1100", ARM::AK_ARMV4,    false },
  { "strongarm1110", ARM::AK_ARMV4,    false },
  { "arm7tdmi",      ARM::AK_ARMV4T,   true },
  { "arm7tdmi-s",    ARM::AK_ARMV4T,   false },
  { "arm710t",       ARM::AK_ARMV4T,   false },
  { "arm720t",       ARM::AK_ARMV4T,   false },
  { "arm9",          ARM::AK_ARMV4T,   false },
  { "arm9tdmi",      ARM::AK_ARMV4T,   false },
  { "arm920",        ARM::AK_ARMV4T,   false },
  { "arm920t",       ARM::AK_ARMV4T,   false },
  { "arm922t",       ARM::AK_ARMV4T,   false },
  { "arm9312",       ARM::AK_ARMV4T,   false },
  { "arm940t",       ARM::AK_ARMV4T,   false },
  { "ep9312",        ARM::AK_ARMV4T,   false },
  { "arm10tdmi",     ARM::AK_ARMV5T,   true },
  { "arm1020t",      ARM::AK_ARMV5T,   false },
  { "arm9e",         ARM::AK_ARMV5TE,  false },
  { "arm946e-s",     ARM::AK_ARMV5TE,  false },
  { "arm966e-s",     ARM::AK_ARMV5TE,  false },
  { "arm968e-s",     ARM::AK_ARMV5TE,  false },
  { "arm10e",        ARM::AK_ARMV5TE,  false },
  { "arm1020e",      ARM::AK_ARMV5TE,  false },
  { "arm1022e",      ARM::AK_ARMV5TE,  true },
  { "iwmmxt",        ARM::AK_ARMV5TE,  false },
  { "xscale",        ARM::AK_ARMV5TE,  false },
  { "arm926ej-s",    ARM::AK_ARMV5TEJ, true },
  { "arm1136jf-s",   ARM::AK_ARMV6,    true },
  { "arm1176j-s",    ARM::AK_ARMV6K,   false },
  { "arm1176jz-s",   ARM::AK_ARMV6K,   false },
  { "mpcore",        ARM::AK_ARMV6K,   false },
  { "mpcorenovfp",   ARM::AK_ARMV6K,   false },
  { "arm1176jzf-s",  ARM::AK_ARMV6K,   true },
  { "arm1176jzf-s",  ARM::AK_ARMV6Z,   true },
  { "arm1176jzf-s",  ARM::AK_ARMV6ZK,  true },
  { "arm1156t2-s",   ARM::AK_ARMV6T2,  true },
  { "arm1156t2f-s",  ARM::AK_ARMV6T2,  false },
  { "cortex-m0",     ARM::AK_ARMV6M,   true },
  { "cortex-m0plus", ARM::AK_ARMV6M,   false },
  { "cortex-m1",     ARM::AK_ARMV6M,   false },
  { "sc000",         ARM::AK_ARMV6M,   false },
  { "cortex-a5",     ARM::AK_ARMV7A,   false },
  { "cortex-a7",     ARM::AK_ARMV7A,   false },
  { "cortex-a8",     ARM::AK_ARMV7A,   true },
  { "cortex-a9",     ARM::AK_ARMV7A,   false },
  { "cortex-a12",    ARM::AK_ARMV7A,   false },
  { "cortex-a15",    ARM::AK_ARMV7A,   false },
  { "cortex-a17",    ARM::AK_ARMV7A,   false },
  { "krait",         ARM::AK_ARMV7A,   false },
  { "cortex-r4",     ARM::AK_ARMV7R,   true },
  { "cortex-r4f",    ARM::AK_ARMV7R,   false },
  { "cortex-r5",     ARM::AK_ARMV7R,   false },
  { "cortex-r7",     ARM::AK_ARMV7R,   false },
  { "sc300",         ARM::AK_ARMV7M,   false },
  { "cortex-m3",     ARM::AK_ARMV7M,   true },
  { "cortex-m4",     ARM::AK_ARMV7EM,  true },
  { "cortex-m7",     ARM::AK_ARMV7EM,  false },
  { "cortex-a53",    ARM::AK_ARMV8A,   true },
  { "cortex-a57",    ARM::AK_ARMV8A,   false },
  { "cortex-a72",    ARM::AK_ARMV8A,   false },
  { "cyclone",       ARM::AK_ARMV8A,   false },
  { "generic",       ARM::AK_ARMV8_1A, true },
  // Non-standard Arch names.
  { "iwmmxt",        ARM::AK_IWMMXT,   true },
  { "xscale",        ARM::AK_XSCALE,   true },
  { "arm10tdmi",     ARM::AK_ARMV5,    true },
  { "arm1022e",      ARM::AK_ARMV5E,   true },
  { "arm1136j-s",    ARM::AK_ARMV6J,   true },
  { "arm1136jz-s",   ARM::AK_ARMV6J,   false },
  { "cortex-m0",     ARM::AK_ARMV6SM,  true },
  { "arm1176jzf-s",  ARM::AK_ARMV6HL,  true },
  { "cortex-a8",     ARM::AK_ARMV7,    true },
  { "cortex-a8",     ARM::AK_ARMV7L,   true },
  { "cortex-a8",     ARM::AK_ARMV7HL,  true },
  { "cortex-m4",     ARM::AK_ARMV7EM,  true },
  { "swift",         ARM::AK_ARMV7S,   true },
  // Invalid CPU
  { "invalid",       ARM::AK_INVALID,  true }
};

} // namespace

// ======================================================= //
// Information by ID
// ======================================================= //

const char *ARMTargetParser::getFPUName(unsigned FPUKind) {
  if (FPUKind >= ARM::FK_LAST)
    return nullptr;
  return FPUNames[FPUKind].Name;
}

unsigned ARMTargetParser::getFPUVersion(unsigned FPUKind) {
  if (FPUKind >= ARM::FK_LAST)
    return 0;
  return FPUNames[FPUKind].FPUVersion;
}

unsigned ARMTargetParser::getFPUNeonSupportLevel(unsigned FPUKind) {
  if (FPUKind >= ARM::FK_LAST)
    return 0;
  return FPUNames[FPUKind].NeonSupport;
}

unsigned ARMTargetParser::getFPURestriction(unsigned FPUKind) {
  if (FPUKind >= ARM::FK_LAST)
    return 0;
  return FPUNames[FPUKind].Restriction;
}

bool ARMTargetParser::getFPUFeatures(unsigned FPUKind,
                                     std::vector<const char *> &Features) {

  if (FPUKind >= ARM::FK_LAST || FPUKind == ARM::FK_INVALID)
    return false;

  // fp-only-sp and d16 subtarget features are independent of each other, so we
  // must enable/disable both.
  switch (FPUNames[FPUKind].Restriction) {
  case ARM::FR_SP_D16:
    Features.push_back("+fp-only-sp");
    Features.push_back("+d16");
    break;
  case ARM::FR_D16:
    Features.push_back("-fp-only-sp");
    Features.push_back("+d16");
    break;
  case ARM::FR_None:
    Features.push_back("-fp-only-sp");
    Features.push_back("-d16");
    break;
  }

  // FPU version subtarget features are inclusive of lower-numbered ones, so
  // enable the one corresponding to this version and disable all that are
  // higher. We also have to make sure to disable fp16 when vfp4 is disabled,
  // as +vfp4 implies +fp16 but -vfp4 does not imply -fp16.
  switch (FPUNames[FPUKind].FPUVersion) {
  case ARM::FV_VFPV5:
    Features.push_back("+fp-armv8");
    break;
  case ARM::FV_VFPV4:
    Features.push_back("+vfp4");
    Features.push_back("-fp-armv8");
    break;
  case ARM::FV_VFPV3_FP16:
    Features.push_back("+vfp3");
    Features.push_back("+fp16");
    Features.push_back("-vfp4");
    Features.push_back("-fp-armv8");
    break;
  case ARM::FV_VFPV3:
    Features.push_back("+vfp3");
    Features.push_back("-fp16");
    Features.push_back("-vfp4");
    Features.push_back("-fp-armv8");
    break;
  case ARM::FV_VFPV2:
    Features.push_back("+vfp2");
    Features.push_back("-vfp3");
    Features.push_back("-fp16");
    Features.push_back("-vfp4");
    Features.push_back("-fp-armv8");
    break;
  case ARM::FV_NONE:
    Features.push_back("-vfp2");
    Features.push_back("-vfp3");
    Features.push_back("-fp16");
    Features.push_back("-vfp4");
    Features.push_back("-fp-armv8");
    break;
  }

  // crypto includes neon, so we handle this similarly to FPU version.
  switch (FPUNames[FPUKind].NeonSupport) {
  case ARM::NS_Crypto:
    Features.push_back("+crypto");
    break;
  case ARM::NS_Neon:
    Features.push_back("+neon");
    Features.push_back("-crypto");
    break;
  case ARM::NS_None:
    Features.push_back("-neon");
    Features.push_back("-crypto");
    break;
  }

  return true;
}

const char *ARMTargetParser::getArchName(unsigned ArchKind) {
  if (ArchKind >= ARM::AK_LAST)
    return nullptr;
  return ARCHNames[ArchKind].Name;
}

const char *ARMTargetParser::getCPUAttr(unsigned ArchKind) {
  if (ArchKind >= ARM::AK_LAST)
    return nullptr;
  return ARCHNames[ArchKind].CPUAttr;
}

const char *ARMTargetParser::getSubArch(unsigned ArchKind) {
  if (ArchKind >= ARM::AK_LAST)
    return nullptr;
  return ARCHNames[ArchKind].SubArch;
}

unsigned ARMTargetParser::getArchAttr(unsigned ArchKind) {
  if (ArchKind >= ARM::AK_LAST)
    return ARMBuildAttrs::CPUArch::Pre_v4;
  return ARCHNames[ArchKind].ArchAttr;
}

const char *ARMTargetParser::getArchExtName(unsigned ArchExtKind) {
  if (ArchExtKind >= ARM::AEK_LAST)
    return nullptr;
  return ARCHExtNames[ArchExtKind].Name;
}

const char *ARMTargetParser::getDefaultCPU(StringRef Arch) {
  unsigned AK = parseArch(Arch);
  if (AK == ARM::AK_INVALID)
    return nullptr;

  // Look for multiple AKs to find the default for pair AK+Name.
  for (const auto CPU : CPUNames) {
    if (CPU.ArchID == AK && CPU.Default)
      return CPU.Name;
  }
  return nullptr;
}

// ======================================================= //
// Parsers
// ======================================================= //

StringRef ARMTargetParser::getFPUSynonym(StringRef FPU) {
  return StringSwitch<StringRef>(FPU)
    .Cases("fpa", "fpe2", "fpe3", "maverick", "invalid") // Unsupported
    .Case("vfp2", "vfpv2")
    .Case("vfp3", "vfpv3")
    .Case("vfp4", "vfpv4")
    .Case("vfp3-d16", "vfpv3-d16")
    .Case("vfp4-d16", "vfpv4-d16")
    .Cases("fp4-sp-d16", "vfpv4-sp-d16", "fpv4-sp-d16")
    .Cases("fp4-dp-d16", "fpv4-dp-d16", "vfpv4-d16")
    .Case("fp5-sp-d16", "fpv5-sp-d16")
    .Cases("fp5-dp-d16", "fpv5-dp-d16", "fpv5-d16")
    // FIXME: Clang uses it, but it's bogus, since neon defaults to vfpv3.
    .Case("neon-vfpv3", "neon")
    .Default(FPU);
}

StringRef ARMTargetParser::getArchSynonym(StringRef Arch) {
  return StringSwitch<StringRef>(Arch)
    .Case("v6sm", "v6s-m")
    .Case("v6m", "v6-m")
    .Case("v7a", "v7-a")
    .Case("v7r", "v7-r")
    .Case("v7m", "v7-m")
    .Case("v7em", "v7e-m")
    .Cases("v8", "v8a", "aarch64", "arm64", "v8-a")
    .Case("v8.1a", "v8.1-a")
    .Default(Arch);
}

// MArch is expected to be of the form (arm|thumb)?(eb)?(v.+)?(eb)?, but
// (iwmmxt|xscale)(eb)? is also permitted. If the former, return
// "v.+", if the latter, return unmodified string, minus 'eb'.
// If invalid, return empty string.
StringRef ARMTargetParser::getCanonicalArchName(StringRef Arch) {
  size_t offset = StringRef::npos;
  StringRef A = Arch;
  StringRef Error = "";

  // Begins with "arm" / "thumb", move past it.
  if (A.startswith("arm64"))
    offset = 5;
  else if (A.startswith("arm"))
    offset = 3;
  else if (A.startswith("thumb"))
    offset = 5;
  else if (A.startswith("aarch64")) {
    offset = 7;
    // AArch64 uses "_be", not "eb" suffix.
    if (A.find("eb") != StringRef::npos)
      return Error;
    if (A.substr(offset,3) == "_be")
      offset += 3;
  }

  // Ex. "armebv7", move past the "eb".
  if (offset != StringRef::npos && A.substr(offset, 2) == "eb")
    offset += 2;
  // Or, if it ends with eb ("armv7eb"), chop it off.
  else if (A.endswith("eb"))
    A = A.substr(0, A.size() - 2);
  // Trim the head
  if (offset != StringRef::npos)
    A = A.substr(offset);

  // Empty string means offset reached the end, which means it's valid.
  if (A.empty())
    return Arch;

  // Only match non-marketing names
  if (offset != StringRef::npos) {
  // Must start with 'vN'.
    if (A[0] != 'v' || !std::isdigit(A[1]))
      return Error;
    // Can't have an extra 'eb'.
    if (A.find("eb") != StringRef::npos)
      return Error;
  }

  // Arch will either be a 'v' name (v7a) or a marketing name (xscale).
  return A;
}

unsigned ARMTargetParser::parseFPU(StringRef FPU) {
  StringRef Syn = getFPUSynonym(FPU);
  for (const auto F : FPUNames) {
    if (Syn == F.Name)
      return F.ID;
  }
  return ARM::FK_INVALID;
}

// Allows partial match, ex. "v7a" matches "armv7a".
unsigned ARMTargetParser::parseArch(StringRef Arch) {
  Arch = getCanonicalArchName(Arch);
  StringRef Syn = getArchSynonym(Arch);
  for (const auto A : ARCHNames) {
    if (StringRef(A.Name).endswith(Syn))
      return A.ID;
  }
  return ARM::AK_INVALID;
}

unsigned ARMTargetParser::parseArchExt(StringRef ArchExt) {
  for (const auto A : ARCHExtNames) {
    if (ArchExt == A.Name)
      return A.ID;
  }
  return ARM::AEK_INVALID;
}

unsigned ARMTargetParser::parseCPUArch(StringRef CPU) {
  for (const auto C : CPUNames) {
    if (CPU == C.Name)
      return C.ArchID;
  }
  return ARM::AK_INVALID;
}

// ARM, Thumb, AArch64
unsigned ARMTargetParser::parseArchISA(StringRef Arch) {
  return StringSwitch<unsigned>(Arch)
      .StartsWith("aarch64", ARM::IK_AARCH64)
      .StartsWith("arm64",   ARM::IK_AARCH64)
      .StartsWith("thumb",   ARM::IK_THUMB)
      .StartsWith("arm",     ARM::IK_ARM)
      .Default(ARM::EK_INVALID);
}

// Little/Big endian
unsigned ARMTargetParser::parseArchEndian(StringRef Arch) {
  if (Arch.startswith("armeb") ||
      Arch.startswith("thumbeb") ||
      Arch.startswith("aarch64_be"))
    return ARM::EK_BIG;

  if (Arch.startswith("arm") || Arch.startswith("thumb")) {
    if (Arch.endswith("eb"))
      return ARM::EK_BIG;
    else
      return ARM::EK_LITTLE;
  }

  if (Arch.startswith("aarch64"))
    return ARM::EK_LITTLE;

  return ARM::EK_INVALID;
}

// Profile A/R/M
unsigned ARMTargetParser::parseArchProfile(StringRef Arch) {
  Arch = getCanonicalArchName(Arch);
  switch(parseArch(Arch)) {
  case ARM::AK_ARMV6M:
  case ARM::AK_ARMV7M:
  case ARM::AK_ARMV6SM:
  case ARM::AK_ARMV7EM:
    return ARM::PK_M;
  case ARM::AK_ARMV7R:
    return ARM::PK_R;
  case ARM::AK_ARMV7:
  case ARM::AK_ARMV7A:
  case ARM::AK_ARMV8A:
  case ARM::AK_ARMV8_1A:
    return ARM::PK_A;
  }
  return ARM::PK_INVALID;
}

// Version number (ex. v7 = 7).
unsigned ARMTargetParser::parseArchVersion(StringRef Arch) {
  Arch = getCanonicalArchName(Arch);
  switch(parseArch(Arch)) {
  case ARM::AK_ARMV2:
  case ARM::AK_ARMV2A:
    return 2;
  case ARM::AK_ARMV3:
  case ARM::AK_ARMV3M:
    return 3;
  case ARM::AK_ARMV4:
  case ARM::AK_ARMV4T:
    return 4;
  case ARM::AK_ARMV5:
  case ARM::AK_ARMV5T:
  case ARM::AK_ARMV5TE:
  case ARM::AK_IWMMXT:
  case ARM::AK_IWMMXT2:
  case ARM::AK_XSCALE:
  case ARM::AK_ARMV5E:
  case ARM::AK_ARMV5TEJ:
    return 5;
  case ARM::AK_ARMV6:
  case ARM::AK_ARMV6J:
  case ARM::AK_ARMV6K:
  case ARM::AK_ARMV6T2:
  case ARM::AK_ARMV6Z:
  case ARM::AK_ARMV6ZK:
  case ARM::AK_ARMV6M:
  case ARM::AK_ARMV6SM:
  case ARM::AK_ARMV6HL:
    return 6;
  case ARM::AK_ARMV7:
  case ARM::AK_ARMV7A:
  case ARM::AK_ARMV7R:
  case ARM::AK_ARMV7M:
  case ARM::AK_ARMV7L:
  case ARM::AK_ARMV7HL:
  case ARM::AK_ARMV7S:
  case ARM::AK_ARMV7EM:
    return 7;
  case ARM::AK_ARMV8A:
  case ARM::AK_ARMV8_1A:
    return 8;
  }
  return 0;
}
