//===-- Host.cpp - Implement OS Host Concept --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the operating system Host concept.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Host.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Config/config.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <string.h>

// Include the platform-specific parts of this class.
#ifdef LLVM_ON_UNIX
#include "Unix/Host.inc"
#endif
#ifdef LLVM_ON_WIN32
#include "Windows/Host.inc"
#endif
#ifdef _MSC_VER
#include <intrin.h>
#endif
#if defined(__APPLE__) && (defined(__ppc__) || defined(__powerpc__))
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/host_info.h>
#include <mach/machine.h>
#endif

#define DEBUG_TYPE "host-detection"

//===----------------------------------------------------------------------===//
//
//  Implementations of the CPU detection routines
//
//===----------------------------------------------------------------------===//

using namespace llvm;

#if defined(__linux__)
static ssize_t LLVM_ATTRIBUTE_UNUSED readCpuInfo(void *Buf, size_t Size) {
  // Note: We cannot mmap /proc/cpuinfo here and then process the resulting
  // memory buffer because the 'file' has 0 size (it can be read from only
  // as a stream).

  int FD;
  std::error_code EC = sys::fs::openFileForRead("/proc/cpuinfo", FD);
  if (EC) {
    DEBUG(dbgs() << "Unable to open /proc/cpuinfo: " << EC.message() << "\n");
    return -1;
  }
  int Ret = read(FD, Buf, Size);
  int CloseStatus = close(FD);
  if (CloseStatus)
    return -1;
  return Ret;
}
#endif

#if defined(i386) || defined(__i386__) || defined(__x86__) || defined(_M_IX86)\
 || defined(__x86_64__) || defined(_M_AMD64) || defined (_M_X64)

/// GetX86CpuIDAndInfo - Execute the specified cpuid and return the 4 values in the
/// specified arguments.  If we can't run cpuid on the host, return true.
static bool GetX86CpuIDAndInfo(unsigned value, unsigned *rEAX, unsigned *rEBX,
                               unsigned *rECX, unsigned *rEDX) {
#if defined(__GNUC__) || defined(__clang__)
  #if defined(__x86_64__) || defined(_M_AMD64) || defined (_M_X64)
    // gcc doesn't know cpuid would clobber ebx/rbx. Preseve it manually.
    asm ("movq\t%%rbx, %%rsi\n\t"
         "cpuid\n\t"
         "xchgq\t%%rbx, %%rsi\n\t"
         : "=a" (*rEAX),
           "=S" (*rEBX),
           "=c" (*rECX),
           "=d" (*rEDX)
         :  "a" (value));
    return false;
  #elif defined(i386) || defined(__i386__) || defined(__x86__) || defined(_M_IX86)
    asm ("movl\t%%ebx, %%esi\n\t"
         "cpuid\n\t"
         "xchgl\t%%ebx, %%esi\n\t"
         : "=a" (*rEAX),
           "=S" (*rEBX),
           "=c" (*rECX),
           "=d" (*rEDX)
         :  "a" (value));
    return false;
// pedantic #else returns to appease -Wunreachable-code (so we don't generate
// postprocessed code that looks like "return true; return false;")
  #else
    return true;
  #endif
#elif defined(_MSC_VER)
  // The MSVC intrinsic is portable across x86 and x64.
  int registers[4];
  __cpuid(registers, value);
  *rEAX = registers[0];
  *rEBX = registers[1];
  *rECX = registers[2];
  *rEDX = registers[3];
  return false;
#else
  return true;
#endif
}

/// GetX86CpuIDAndInfoEx - Execute the specified cpuid with subleaf and return the
/// 4 values in the specified arguments.  If we can't run cpuid on the host,
/// return true.
static bool GetX86CpuIDAndInfoEx(unsigned value, unsigned subleaf,
                                 unsigned *rEAX, unsigned *rEBX, unsigned *rECX,
                                 unsigned *rEDX) {
#if defined(__x86_64__) || defined(_M_AMD64) || defined (_M_X64)
  #if defined(__GNUC__)
    // gcc doesn't know cpuid would clobber ebx/rbx. Preseve it manually.
    asm ("movq\t%%rbx, %%rsi\n\t"
         "cpuid\n\t"
         "xchgq\t%%rbx, %%rsi\n\t"
         : "=a" (*rEAX),
           "=S" (*rEBX),
           "=c" (*rECX),
           "=d" (*rEDX)
         :  "a" (value),
            "c" (subleaf));
    return false;
  #elif defined(_MSC_VER)
    int registers[4];
    __cpuidex(registers, value, subleaf);
    *rEAX = registers[0];
    *rEBX = registers[1];
    *rECX = registers[2];
    *rEDX = registers[3];
    return false;
  #else
    return true;
  #endif
#elif defined(i386) || defined(__i386__) || defined(__x86__) || defined(_M_IX86)
  #if defined(__GNUC__)
    asm ("movl\t%%ebx, %%esi\n\t"
         "cpuid\n\t"
         "xchgl\t%%ebx, %%esi\n\t"
         : "=a" (*rEAX),
           "=S" (*rEBX),
           "=c" (*rECX),
           "=d" (*rEDX)
         :  "a" (value),
            "c" (subleaf));
    return false;
  #elif defined(_MSC_VER)
    __asm {
      mov   eax,value
      mov   ecx,subleaf
      cpuid
      mov   esi,rEAX
      mov   dword ptr [esi],eax
      mov   esi,rEBX
      mov   dword ptr [esi],ebx
      mov   esi,rECX
      mov   dword ptr [esi],ecx
      mov   esi,rEDX
      mov   dword ptr [esi],edx
    }
    return false;
  #else
    return true;
  #endif
#else
  return true;
#endif
}

static bool GetX86XCR0(unsigned *rEAX, unsigned *rEDX) {
#if defined(__GNUC__)
  // Check xgetbv; this uses a .byte sequence instead of the instruction
  // directly because older assemblers do not include support for xgetbv and
  // there is no easy way to conditionally compile based on the assembler used.
  __asm__ (".byte 0x0f, 0x01, 0xd0" : "=a" (*rEAX), "=d" (*rEDX) : "c" (0));
  return false;
#elif defined(_MSC_FULL_VER) && defined(_XCR_XFEATURE_ENABLED_MASK) && !defined(_M_ARM64EC)
  unsigned long long Result = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
  *rEAX = Result;
  *rEDX = Result >> 32;
  return false;
#else
  return true;
#endif
}

static void DetectX86FamilyModel(unsigned EAX, unsigned &Family,
                                 unsigned &Model) {
  Family = (EAX >> 8) & 0xf; // Bits 8 - 11
  Model  = (EAX >> 4) & 0xf; // Bits 4 - 7
  if (Family == 6 || Family == 0xf) {
    if (Family == 0xf)
      // Examine extended family ID if family ID is F.
      Family += (EAX >> 20) & 0xff;    // Bits 20 - 27
    // Examine extended model ID if family ID is 6 or F.
    Model += ((EAX >> 16) & 0xf) << 4; // Bits 16 - 19
  }
}

StringRef sys::getHostCPUName() {
  unsigned EAX = 0, EBX = 0, ECX = 0, EDX = 0;
  if (GetX86CpuIDAndInfo(0x1, &EAX, &EBX, &ECX, &EDX))
    return "generic";
  unsigned Family = 0;
  unsigned Model  = 0;
  DetectX86FamilyModel(EAX, Family, Model);

  union {
    unsigned u[3];
    char     c[12];
  } text;

  unsigned MaxLeaf;
  GetX86CpuIDAndInfo(0, &MaxLeaf, text.u+0, text.u+2, text.u+1);

  bool HasMMX   = (EDX >> 23) & 1;
  bool HasSSE   = (EDX >> 25) & 1;
  bool HasSSE2  = (EDX >> 26) & 1;
  bool HasSSE3  = (ECX >>  0) & 1;
  bool HasSSSE3 = (ECX >>  9) & 1;
  bool HasSSE41 = (ECX >> 19) & 1;
  bool HasSSE42 = (ECX >> 20) & 1;
  bool HasMOVBE = (ECX >> 22) & 1;
  // If CPUID indicates support for XSAVE, XRESTORE and AVX, and XGETBV
  // indicates that the AVX registers will be saved and restored on context
  // switch, then we have full AVX support.
  const unsigned AVXBits = (1 << 27) | (1 << 28);
  bool HasAVX = ((ECX & AVXBits) == AVXBits) && !GetX86XCR0(&EAX, &EDX) &&
                ((EAX & 0x6) == 0x6);
  bool HasAVX512Save = HasAVX && ((EAX & 0xe0) == 0xe0);
  bool HasLeaf7 = MaxLeaf >= 0x7 &&
                  !GetX86CpuIDAndInfoEx(0x7, 0x0, &EAX, &EBX, &ECX, &EDX);
  bool HasADX = HasLeaf7 && ((EBX >> 19) & 1);
  bool HasAVX2 = HasAVX && HasLeaf7 && (EBX & 0x20);
  bool HasAVX512 = HasLeaf7 && HasAVX512Save && ((EBX >> 16) & 1);

  GetX86CpuIDAndInfo(0x80000001, &EAX, &EBX, &ECX, &EDX);
  bool Em64T = (EDX >> 29) & 0x1;
  bool HasTBM = (ECX >> 21) & 0x1;

  if (memcmp(text.c, "GenuineIntel", 12) == 0) {
    switch (Family) {
    case 3:
      return "i386";
    case 4:
      switch (Model) {
      case 0: // Intel486 DX processors
      case 1: // Intel486 DX processors
      case 2: // Intel486 SX processors
      case 3: // Intel487 processors, IntelDX2 OverDrive processors,
              // IntelDX2 processors
      case 4: // Intel486 SL processor
      case 5: // IntelSX2 processors
      case 7: // Write-Back Enhanced IntelDX2 processors
      case 8: // IntelDX4 OverDrive processors, IntelDX4 processors
      default: return "i486";
      }
    case 5:
      switch (Model) {
      case  1: // Pentium OverDrive processor for Pentium processor (60, 66),
               // Pentium processors (60, 66)
      case  2: // Pentium OverDrive processor for Pentium processor (75, 90,
               // 100, 120, 133), Pentium processors (75, 90, 100, 120, 133,
               // 150, 166, 200)
      case  3: // Pentium OverDrive processors for Intel486 processor-based
               // systems
        return "pentium";

      case  4: // Pentium OverDrive processor with MMX technology for Pentium
               // processor (75, 90, 100, 120, 133), Pentium processor with
               // MMX technology (166, 200)
        return "pentium-mmx";

      default: return "pentium";
      }
    case 6:
      switch (Model) {
      case  1: // Pentium Pro processor
        return "pentiumpro";

      case  3: // Intel Pentium II OverDrive processor, Pentium II processor,
               // model 03
      case  5: // Pentium II processor, model 05, Pentium II Xeon processor,
               // model 05, and Intel Celeron processor, model 05
      case  6: // Celeron processor, model 06
        return "pentium2";

      case  7: // Pentium III processor, model 07, and Pentium III Xeon
               // processor, model 07
      case  8: // Pentium III processor, model 08, Pentium III Xeon processor,
               // model 08, and Celeron processor, model 08
      case 10: // Pentium III Xeon processor, model 0Ah
      case 11: // Pentium III processor, model 0Bh
        return "pentium3";

      case  9: // Intel Pentium M processor, Intel Celeron M processor model 09.
      case 13: // Intel Pentium M processor, Intel Celeron M processor, model
               // 0Dh. All processors are manufactured using the 90 nm process.
      case 21: // Intel EP80579 Integrated Processor and Intel EP80579
               // Integrated Processor with Intel QuickAssist Technology
        return "pentium-m";

      case 14: // Intel Core Duo processor, Intel Core Solo processor, model
               // 0Eh. All processors are manufactured using the 65 nm process.
        return "yonah";

      case 15: // Intel Core 2 Duo processor, Intel Core 2 Duo mobile
               // processor, Intel Core 2 Quad processor, Intel Core 2 Quad
               // mobile processor, Intel Core 2 Extreme processor, Intel
               // Pentium Dual-Core processor, Intel Xeon processor, model
               // 0Fh. All processors are manufactured using the 65 nm process.
      case 22: // Intel Celeron processor model 16h. All processors are
               // manufactured using the 65 nm process
        return "core2";

      case 23: // Intel Core 2 Extreme processor, Intel Xeon processor, model
               // 17h. All processors are manufactured using the 45 nm process.
               //
               // 45nm: Penryn , Wolfdale, Yorkfield (XE)
      case 29: // Intel Xeon processor MP. All processors are manufactured using
               // the 45 nm process.
        return "penryn";

      case 26: // Intel Core i7 processor and Intel Xeon processor. All
               // processors are manufactured using the 45 nm process.
      case 30: // Intel(R) Core(TM) i7 CPU         870  @ 2.93GHz.
               // As found in a Summer 2010 model iMac.
      case 46: // Nehalem EX
        return "nehalem";
      case 37: // Intel Core i7, laptop version.
      case 44: // Intel Core i7 processor and Intel Xeon processor. All
               // processors are manufactured using the 32 nm process.
      case 47: // Westmere EX
        return "westmere";

      // SandyBridge:
      case 42: // Intel Core i7 processor. All processors are manufactured
               // using the 32 nm process.
      case 45:
        return "sandybridge";

      // Ivy Bridge:
      case 58:
      case 62: // Ivy Bridge EP
        return "ivybridge";

      // Haswell:
      case 60:
      case 63:
      case 69:
      case 70:
        return "haswell";

      // Broadwell:
      case 61:
        return "broadwell";

      case 28: // Most 45 nm Intel Atom processors
      case 38: // 45 nm Atom Lincroft
      case 39: // 32 nm Atom Medfield
      case 53: // 32 nm Atom Midview
      case 54: // 32 nm Atom Midview
        return "bonnell";

      // Atom Silvermont codes from the Intel software optimization guide.
      case 55:
      case 74:
      case 77:
        return "silvermont";

      default: // Unknown family 6 CPU, try to guess.
        if (HasAVX512)
          return "knl";
        if (HasADX)
          return "broadwell";
        if (HasAVX2)
          return "haswell";
        if (HasAVX)
          return "sandybridge";
        if (HasSSE42)
          return HasMOVBE ? "silvermont" : "nehalem";
        if (HasSSE41)
          return "penryn";
        if (HasSSSE3)
          return HasMOVBE ? "bonnell" : "core2";
        if (Em64T)
          return "x86-64";
        if (HasSSE2)
          return "pentium-m";
        if (HasSSE)
          return "pentium3";
        if (HasMMX)
          return "pentium2";
        return "pentiumpro";
      }
    case 15: {
      switch (Model) {
      case  0: // Pentium 4 processor, Intel Xeon processor. All processors are
               // model 00h and manufactured using the 0.18 micron process.
      case  1: // Pentium 4 processor, Intel Xeon processor, Intel Xeon
               // processor MP, and Intel Celeron processor. All processors are
               // model 01h and manufactured using the 0.18 micron process.
      case  2: // Pentium 4 processor, Mobile Intel Pentium 4 processor - M,
               // Intel Xeon processor, Intel Xeon processor MP, Intel Celeron
               // processor, and Mobile Intel Celeron processor. All processors
               // are model 02h and manufactured using the 0.13 micron process.
        return (Em64T) ? "x86-64" : "pentium4";

      case  3: // Pentium 4 processor, Intel Xeon processor, Intel Celeron D
               // processor. All processors are model 03h and manufactured using
               // the 90 nm process.
      case  4: // Pentium 4 processor, Pentium 4 processor Extreme Edition,
               // Pentium D processor, Intel Xeon processor, Intel Xeon
               // processor MP, Intel Celeron D processor. All processors are
               // model 04h and manufactured using the 90 nm process.
      case  6: // Pentium 4 processor, Pentium D processor, Pentium processor
               // Extreme Edition, Intel Xeon processor, Intel Xeon processor
               // MP, Intel Celeron D processor. All processors are model 06h
               // and manufactured using the 65 nm process.
        return (Em64T) ? "nocona" : "prescott";

      default:
        return (Em64T) ? "x86-64" : "pentium4";
      }
    }

    default:
      return "generic";
    }
  } else if (memcmp(text.c, "AuthenticAMD", 12) == 0) {
    // FIXME: this poorly matches the generated SubtargetFeatureKV table.  There
    // appears to be no way to generate the wide variety of AMD-specific targets
    // from the information returned from CPUID.
    switch (Family) {
      case 4:
        return "i486";
      case 5:
        switch (Model) {
        case 6:
        case 7:  return "k6";
        case 8:  return "k6-2";
        case 9:
        case 13: return "k6-3";
        case 10: return "geode";
        default: return "pentium";
        }
      case 6:
        switch (Model) {
        case 4:  return "athlon-tbird";
        case 6:
        case 7:
        case 8:  return "athlon-mp";
        case 10: return "athlon-xp";
        default: return "athlon";
        }
      case 15:
        if (HasSSE3)
          return "k8-sse3";
        switch (Model) {
        case 1:  return "opteron";
        case 5:  return "athlon-fx"; // also opteron
        default: return "athlon64";
        }
      case 16:
        return "amdfam10";
      case 20:
        return "btver1";
      case 21:
        if (!HasAVX) // If the OS doesn't support AVX provide a sane fallback.
          return "btver1";
        if (Model >= 0x50)
          return "bdver4"; // 50h-6Fh: Excavator
        if (Model >= 0x30)
          return "bdver3"; // 30h-3Fh: Steamroller
        if (Model >= 0x10 || HasTBM)
          return "bdver2"; // 10h-1Fh: Piledriver
        return "bdver1";   // 00h-0Fh: Bulldozer
      case 22:
        if (!HasAVX) // If the OS doesn't support AVX provide a sane fallback.
          return "btver1";
        return "btver2";
    default:
      return "generic";
    }
  }
  return "generic";
}
#elif defined(__APPLE__) && (defined(__ppc__) || defined(__powerpc__))
StringRef sys::getHostCPUName() {
  host_basic_info_data_t hostInfo;
  mach_msg_type_number_t infoCount;

  infoCount = HOST_BASIC_INFO_COUNT;
  host_info(mach_host_self(), HOST_BASIC_INFO, (host_info_t)&hostInfo, 
            &infoCount);
            
  if (hostInfo.cpu_type != CPU_TYPE_POWERPC) return "generic";

  switch(hostInfo.cpu_subtype) {
  case CPU_SUBTYPE_POWERPC_601:   return "601";
  case CPU_SUBTYPE_POWERPC_602:   return "602";
  case CPU_SUBTYPE_POWERPC_603:   return "603";
  case CPU_SUBTYPE_POWERPC_603e:  return "603e";
  case CPU_SUBTYPE_POWERPC_603ev: return "603ev";
  case CPU_SUBTYPE_POWERPC_604:   return "604";
  case CPU_SUBTYPE_POWERPC_604e:  return "604e";
  case CPU_SUBTYPE_POWERPC_620:   return "620";
  case CPU_SUBTYPE_POWERPC_750:   return "750";
  case CPU_SUBTYPE_POWERPC_7400:  return "7400";
  case CPU_SUBTYPE_POWERPC_7450:  return "7450";
  case CPU_SUBTYPE_POWERPC_970:   return "970";
  default: ;
  }
  
  return "generic";
}
#elif defined(__linux__) && (defined(__ppc__) || defined(__powerpc__))
StringRef sys::getHostCPUName() {
  // Access to the Processor Version Register (PVR) on PowerPC is privileged,
  // and so we must use an operating-system interface to determine the current
  // processor type. On Linux, this is exposed through the /proc/cpuinfo file.
  const char *generic = "generic";

  // The cpu line is second (after the 'processor: 0' line), so if this
  // buffer is too small then something has changed (or is wrong).
  char buffer[1024];
  ssize_t CPUInfoSize = readCpuInfo(buffer, sizeof(buffer));
  if (CPUInfoSize == -1)
    return generic;

  const char *CPUInfoStart = buffer;
  const char *CPUInfoEnd = buffer + CPUInfoSize;

  const char *CIP = CPUInfoStart;

  const char *CPUStart = 0;
  size_t CPULen = 0;

  // We need to find the first line which starts with cpu, spaces, and a colon.
  // After the colon, there may be some additional spaces and then the cpu type.
  while (CIP < CPUInfoEnd && CPUStart == 0) {
    if (CIP < CPUInfoEnd && *CIP == '\n')
      ++CIP;

    if (CIP < CPUInfoEnd && *CIP == 'c') {
      ++CIP;
      if (CIP < CPUInfoEnd && *CIP == 'p') {
        ++CIP;
        if (CIP < CPUInfoEnd && *CIP == 'u') {
          ++CIP;
          while (CIP < CPUInfoEnd && (*CIP == ' ' || *CIP == '\t'))
            ++CIP;
  
          if (CIP < CPUInfoEnd && *CIP == ':') {
            ++CIP;
            while (CIP < CPUInfoEnd && (*CIP == ' ' || *CIP == '\t'))
              ++CIP;
  
            if (CIP < CPUInfoEnd) {
              CPUStart = CIP;
              while (CIP < CPUInfoEnd && (*CIP != ' ' && *CIP != '\t' &&
                                          *CIP != ',' && *CIP != '\n'))
                ++CIP;
              CPULen = CIP - CPUStart;
            }
          }
        }
      }
    }

    if (CPUStart == 0)
      while (CIP < CPUInfoEnd && *CIP != '\n')
        ++CIP;
  }

  if (CPUStart == 0)
    return generic;

  return StringSwitch<const char *>(StringRef(CPUStart, CPULen))
    .Case("604e", "604e")
    .Case("604", "604")
    .Case("7400", "7400")
    .Case("7410", "7400")
    .Case("7447", "7400")
    .Case("7455", "7450")
    .Case("G4", "g4")
    .Case("POWER4", "970")
    .Case("PPC970FX", "970")
    .Case("PPC970MP", "970")
    .Case("G5", "g5")
    .Case("POWER5", "g5")
    .Case("A2", "a2")
    .Case("POWER6", "pwr6")
    .Case("POWER7", "pwr7")
    .Case("POWER8", "pwr8")
    .Case("POWER8E", "pwr8")
    .Default(generic);
}
#elif defined(__linux__) && defined(__arm__)
StringRef sys::getHostCPUName() {
  // The cpuid register on arm is not accessible from user space. On Linux,
  // it is exposed through the /proc/cpuinfo file.

  // Read 1024 bytes from /proc/cpuinfo, which should contain the CPU part line
  // in all cases.
  char buffer[1024];
  ssize_t CPUInfoSize = readCpuInfo(buffer, sizeof(buffer));
  if (CPUInfoSize == -1)
    return "generic";

  StringRef Str(buffer, CPUInfoSize);

  SmallVector<StringRef, 32> Lines;
  Str.split(Lines, "\n");

  // Look for the CPU implementer line.
  StringRef Implementer;
  for (unsigned I = 0, E = Lines.size(); I != E; ++I)
    if (Lines[I].startswith("CPU implementer"))
      Implementer = Lines[I].substr(15).ltrim("\t :");

  if (Implementer == "0x41") // ARM Ltd.
    // Look for the CPU part line.
    for (unsigned I = 0, E = Lines.size(); I != E; ++I)
      if (Lines[I].startswith("CPU part"))
        // The CPU part is a 3 digit hexadecimal number with a 0x prefix. The
        // values correspond to the "Part number" in the CP15/c0 register. The
        // contents are specified in the various processor manuals.
        return StringSwitch<const char *>(Lines[I].substr(8).ltrim("\t :"))
          .Case("0x926", "arm926ej-s")
          .Case("0xb02", "mpcore")
          .Case("0xb36", "arm1136j-s")
          .Case("0xb56", "arm1156t2-s")
          .Case("0xb76", "arm1176jz-s")
          .Case("0xc08", "cortex-a8")
          .Case("0xc09", "cortex-a9")
          .Case("0xc0f", "cortex-a15")
          .Case("0xc20", "cortex-m0")
          .Case("0xc23", "cortex-m3")
          .Case("0xc24", "cortex-m4")
          .Default("generic");

  if (Implementer == "0x51") // Qualcomm Technologies, Inc.
    // Look for the CPU part line.
    for (unsigned I = 0, E = Lines.size(); I != E; ++I)
      if (Lines[I].startswith("CPU part"))
        // The CPU part is a 3 digit hexadecimal number with a 0x prefix. The
        // values correspond to the "Part number" in the CP15/c0 register. The
        // contents are specified in the various processor manuals.
        return StringSwitch<const char *>(Lines[I].substr(8).ltrim("\t :"))
          .Case("0x06f", "krait") // APQ8064
          .Default("generic");

  return "generic";
}
#elif defined(__linux__) && defined(__s390x__)
StringRef sys::getHostCPUName() {
  // STIDP is a privileged operation, so use /proc/cpuinfo instead.

  // The "processor 0:" line comes after a fair amount of other information,
  // including a cache breakdown, but this should be plenty.
  char buffer[2048];
  ssize_t CPUInfoSize = readCpuInfo(buffer, sizeof(buffer));
  if (CPUInfoSize == -1)
    return "generic";

  StringRef Str(buffer, CPUInfoSize);
  SmallVector<StringRef, 32> Lines;
  Str.split(Lines, "\n");

  // Look for the CPU features.
  SmallVector<StringRef, 32> CPUFeatures;
  for (unsigned I = 0, E = Lines.size(); I != E; ++I)
    if (Lines[I].startswith("features")) {
      size_t Pos = Lines[I].find(":");
      if (Pos != StringRef::npos) {
        Lines[I].drop_front(Pos + 1).split(CPUFeatures, " ");
        break;
      }
    }

  // We need to check for the presence of vector support independently of
  // the machine type, since we may only use the vector register set when
  // supported by the kernel (and hypervisor).
  bool HaveVectorSupport = false;
  for (unsigned I = 0, E = CPUFeatures.size(); I != E; ++I) {
    if (CPUFeatures[I] == "vx")
      HaveVectorSupport = true;
  }

  // Now check the processor machine type.
  for (unsigned I = 0, E = Lines.size(); I != E; ++I) {
    if (Lines[I].startswith("processor ")) {
      size_t Pos = Lines[I].find("machine = ");
      if (Pos != StringRef::npos) {
        Pos += sizeof("machine = ") - 1;
        unsigned int Id;
        if (!Lines[I].drop_front(Pos).getAsInteger(10, Id)) {
          if (Id >= 2964 && HaveVectorSupport)
            return "z13";
          if (Id >= 2827)
            return "zEC12";
          if (Id >= 2817)
            return "z196";
        }
      }
      break;
    }
  }
  
  return "generic";
}
#else
StringRef sys::getHostCPUName() {
  return "generic";
}
#endif

#if defined(i386) || defined(__i386__) || defined(__x86__) || defined(_M_IX86)\
 || defined(__x86_64__) || defined(_M_AMD64) || defined (_M_X64)
bool sys::getHostCPUFeatures(StringMap<bool> &Features) {
  unsigned EAX = 0, EBX = 0, ECX = 0, EDX = 0;
  unsigned MaxLevel;
  union {
    unsigned u[3];
    char     c[12];
  } text;

  if (GetX86CpuIDAndInfo(0, &MaxLevel, text.u+0, text.u+2, text.u+1) ||
      MaxLevel < 1)
    return false;

  GetX86CpuIDAndInfo(1, &EAX, &EBX, &ECX, &EDX);

  Features["cmov"]   = (EDX >> 15) & 1;
  Features["mmx"]    = (EDX >> 23) & 1;
  Features["sse"]    = (EDX >> 25) & 1;
  Features["sse2"]   = (EDX >> 26) & 1;
  Features["sse3"]   = (ECX >>  0) & 1;
  Features["ssse3"]  = (ECX >>  9) & 1;
  Features["sse4.1"] = (ECX >> 19) & 1;
  Features["sse4.2"] = (ECX >> 20) & 1;

  Features["pclmul"] = (ECX >>  1) & 1;
  Features["cx16"]   = (ECX >> 13) & 1;
  Features["movbe"]  = (ECX >> 22) & 1;
  Features["popcnt"] = (ECX >> 23) & 1;
  Features["aes"]    = (ECX >> 25) & 1;
  Features["rdrnd"]  = (ECX >> 30) & 1;

  // If CPUID indicates support for XSAVE, XRESTORE and AVX, and XGETBV
  // indicates that the AVX registers will be saved and restored on context
  // switch, then we have full AVX support.
  bool HasAVX = ((ECX >> 27) & 1) && ((ECX >> 28) & 1) &&
                !GetX86XCR0(&EAX, &EDX) && ((EAX & 0x6) == 0x6);
  Features["avx"]    = HasAVX;
  Features["fma"]    = HasAVX && (ECX >> 12) & 1;
  Features["f16c"]   = HasAVX && (ECX >> 29) & 1;

  // AVX512 requires additional context to be saved by the OS.
  bool HasAVX512Save = HasAVX && ((EAX & 0xe0) == 0xe0);

  unsigned MaxExtLevel;
  GetX86CpuIDAndInfo(0x80000000, &MaxExtLevel, &EBX, &ECX, &EDX);

  bool HasExtLeaf1 = MaxExtLevel >= 0x80000001 &&
                     !GetX86CpuIDAndInfo(0x80000001, &EAX, &EBX, &ECX, &EDX);
  Features["lzcnt"]  = HasExtLeaf1 && ((ECX >>  5) & 1);
  Features["sse4a"]  = HasExtLeaf1 && ((ECX >>  6) & 1);
  Features["prfchw"] = HasExtLeaf1 && ((ECX >>  8) & 1);
  Features["xop"]    = HasAVX && HasExtLeaf1 && ((ECX >> 11) & 1);
  Features["fma4"]   = HasAVX && HasExtLeaf1 && ((ECX >> 16) & 1);
  Features["tbm"]    = HasExtLeaf1 && ((ECX >> 21) & 1);

  bool HasLeaf7 = MaxLevel >= 7 &&
                  !GetX86CpuIDAndInfoEx(0x7, 0x0, &EAX, &EBX, &ECX, &EDX);

  // AVX2 is only supported if we have the OS save support from AVX.
  Features["avx2"]     = HasAVX && HasLeaf7 && (EBX >>  5) & 1;

  Features["fsgsbase"] = HasLeaf7 && ((EBX >>  0) & 1);
  Features["bmi"]      = HasLeaf7 && ((EBX >>  3) & 1);
  Features["hle"]      = HasLeaf7 && ((EBX >>  4) & 1);
  Features["bmi2"]     = HasLeaf7 && ((EBX >>  8) & 1);
  Features["rtm"]      = HasLeaf7 && ((EBX >> 11) & 1);
  Features["rdseed"]   = HasLeaf7 && ((EBX >> 18) & 1);
  Features["adx"]      = HasLeaf7 && ((EBX >> 19) & 1);
  Features["sha"]      = HasLeaf7 && ((EBX >> 29) & 1);

  // AVX512 is only supported if the OS supports the context save for it.
  Features["avx512f"]  = HasLeaf7 && ((EBX >> 16) & 1) && HasAVX512Save;
  Features["avx512dq"] = HasLeaf7 && ((EBX >> 17) & 1) && HasAVX512Save;
  Features["avx512pf"] = HasLeaf7 && ((EBX >> 26) & 1) && HasAVX512Save;
  Features["avx512er"] = HasLeaf7 && ((EBX >> 27) & 1) && HasAVX512Save;
  Features["avx512cd"] = HasLeaf7 && ((EBX >> 28) & 1) && HasAVX512Save;
  Features["avx512bw"] = HasLeaf7 && ((EBX >> 30) & 1) && HasAVX512Save;
  Features["avx512vl"] = HasLeaf7 && ((EBX >> 31) & 1) && HasAVX512Save;

  return true;
}
#elif defined(__linux__) && (defined(__arm__) || defined(__aarch64__))
bool sys::getHostCPUFeatures(StringMap<bool> &Features) {
  // Read 1024 bytes from /proc/cpuinfo, which should contain the Features line
  // in all cases.
  char buffer[1024];
  ssize_t CPUInfoSize = readCpuInfo(buffer, sizeof(buffer));
  if (CPUInfoSize == -1)
    return false;

  StringRef Str(buffer, CPUInfoSize);

  SmallVector<StringRef, 32> Lines;
  Str.split(Lines, "\n");

  SmallVector<StringRef, 32> CPUFeatures;

  // Look for the CPU features.
  for (unsigned I = 0, E = Lines.size(); I != E; ++I)
    if (Lines[I].startswith("Features")) {
      Lines[I].split(CPUFeatures, " ");
      break;
    }

#if defined(__aarch64__)
  // Keep track of which crypto features we have seen
  enum {
    CAP_AES   = 0x1,
    CAP_PMULL = 0x2,
    CAP_SHA1  = 0x4,
    CAP_SHA2  = 0x8
  };
  uint32_t crypto = 0;
#endif

  for (unsigned I = 0, E = CPUFeatures.size(); I != E; ++I) {
    StringRef LLVMFeatureStr = StringSwitch<StringRef>(CPUFeatures[I])
#if defined(__aarch64__)
      .Case("asimd", "neon")
      .Case("fp", "fp-armv8")
      .Case("crc32", "crc")
#else
      .Case("half", "fp16")
      .Case("neon", "neon")
      .Case("vfpv3", "vfp3")
      .Case("vfpv3d16", "d16")
      .Case("vfpv4", "vfp4")
      .Case("idiva", "hwdiv-arm")
      .Case("idivt", "hwdiv")
#endif
      .Default("");

#if defined(__aarch64__)
    // We need to check crypto separately since we need all of the crypto
    // extensions to enable the subtarget feature
    if (CPUFeatures[I] == "aes")
      crypto |= CAP_AES;
    else if (CPUFeatures[I] == "pmull")
      crypto |= CAP_PMULL;
    else if (CPUFeatures[I] == "sha1")
      crypto |= CAP_SHA1;
    else if (CPUFeatures[I] == "sha2")
      crypto |= CAP_SHA2;
#endif

    if (LLVMFeatureStr != "")
      Features[LLVMFeatureStr] = true;
  }

#if defined(__aarch64__)
  // If we have all crypto bits we can add the feature
  if (crypto == (CAP_AES | CAP_PMULL | CAP_SHA1 | CAP_SHA2))
    Features["crypto"] = true;
#endif

  return true;
}
#else
bool sys::getHostCPUFeatures(StringMap<bool> &Features){
  return false;
}
#endif

std::string sys::getProcessTriple() {
  Triple PT(Triple::normalize(LLVM_HOST_TRIPLE));

  if (sizeof(void *) == 8 && PT.isArch32Bit())
    PT = PT.get64BitArchVariant();
  if (sizeof(void *) == 4 && PT.isArch64Bit())
    PT = PT.get32BitArchVariant();

  return PT.str();
}
