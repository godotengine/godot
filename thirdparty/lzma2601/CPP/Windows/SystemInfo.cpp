// Windows/SystemInfo.cpp

#include "StdAfx.h"

#include "../../C/CpuArch.h"

#include "../Common/IntToString.h"
#include "../Common/StringConvert.h"
#include "../Common/StringToInt.h"

#ifdef _WIN32

#include "Registry.h"

#else

#include <unistd.h>
#include <sys/utsname.h>
#ifdef __APPLE__
#include <sys/sysctl.h>

#elif !defined(_AIX)

#if defined(__GLIBC__) && (__GLIBC__ * 100 + __GLIBC_MINOR__ >= 216)
  #define Z7_GETAUXV_AVAILABLE
#elif !defined(__QNXNTO__)
//  #pragma message("=== is not NEW GLIBC === ")
  #if defined __has_include
  #if __has_include (<sys/auxv.h>)
//    #pragma message("=== sys/auxv.h is avail=== ")
    #define Z7_GETAUXV_AVAILABLE
  #endif
  #endif
#endif

#ifdef Z7_GETAUXV_AVAILABLE
// #if defined __has_include
// #if __has_include (<sys/auxv.h>)
#include <sys/auxv.h>
#define USE_HWCAP
// #endif
// #endif

// #undef AT_HWCAP    // to debug
// #undef AT_HWCAP2   // to debug

/* the following patch for some debian systems.
   Is it OK to define AT_HWCAP and AT_HWCAP2 here with these constant numbers? */
/*
#if defined(__FreeBSD_kernel__) && defined(__GLIBC__)
  #ifndef AT_HWCAP
    #define AT_HWCAP 16
  #endif
  #ifndef AT_HWCAP2
    #define AT_HWCAP2 26
  #endif
#endif
*/

#ifdef USE_HWCAP

#if defined(__FreeBSD__) || defined(__OpenBSD__)

// #if (__FreeBSD__ >= 13) // (FreeBSD 12.01 is required for elf_aux_info() ???)
static unsigned long MY_getauxval(int aux)
{
  unsigned long val;
  if (elf_aux_info(aux, &val, sizeof(val)))
    return 0;
  return val;
}

#else // ! __FreeBSD__

#ifdef MY_CPU_ARM_OR_ARM64
  #if defined __has_include
  #if __has_include (<asm/hwcap.h>)
#include <asm/hwcap.h>
  #endif
  #endif
#endif

#if defined(AT_HWCAP) || defined(AT_HWCAP2)
#define MY_getauxval  getauxval
#endif

#endif // ! __FreeBSD__
#endif // USE_HWCAP
#endif // Z7_GETAUXV_AVAILABLE

#endif // !defined(_AIX)

#ifdef __linux__
#include "../Windows/FileIO.h"
#endif

#endif // WIN32

#include "SystemInfo.h"
#include "System.h"

using namespace NWindows;

#ifdef __linux__

static bool ReadFile_to_Buffer(CFSTR fileName, CByteBuffer &buf)
{
  buf.Free();
  NWindows::NFile::NIO::CInFile file;
  if (!file.Open(fileName))
    return false;
  /*
  UInt64 size;
  if (!file.GetLength(size))
  {
    // GetLength() doesn't work "/proc/cpuinfo"
    return false;
  }
  if (size >= ((UInt32)1 << 29))
    return false;
  */
  size_t size = 0;
  size_t addSize = (size_t)1 << 12;
  for (;;)
  {
    // printf("\nsize = %d\n", (unsigned)size);
    buf.ChangeSize_KeepData(size + addSize, size);
    size_t processed;
    if (!file.ReadFull(buf.NonConstData() + size, addSize, processed))
      return false;
    if (processed == 0)
    {
      buf.ChangeSize_KeepData(size, size);
      return true;
    }
    size += processed;
    addSize *= 2;
  }
}

static bool ReadFile_to_String(CFSTR fileName, AString &s)
{
  CByteBuffer buf;
  if (!ReadFile_to_Buffer(fileName, buf))
    return false;
  s.SetFrom_CalcLen((const char *)(const void *)(const Byte *)buf, (unsigned)buf.Size());
  return true;
}

#endif


#if defined(_WIN32) || defined(AT_HWCAP) || defined(AT_HWCAP2)
static void PrintHex(AString &s, UInt64 v)
{
  char temp[32];
  ConvertUInt64ToHex(v, temp);
  s += temp;
}
#endif

#ifdef MY_CPU_X86_OR_AMD64

Z7_NO_INLINE
static void PrintCpuChars(AString &s, UInt32 v)
{
  for (unsigned j = 0; j < 4; j++)
  {
    const Byte b = (Byte)(v & 0xFF);
    v >>= 8;
    if (b == 0)
      break;
    if (b >= 0x20 && b <= 0x7f)
      s.Add_Char((char)b);
    else
    {
      s.Add_Char('[');
      char temp[16];
      ConvertUInt32ToHex(b, temp);
      s += temp;
      s.Add_Char(']');
    }
  }
}


static void x86cpuid_to_String(AString &s)
{
  s.Empty();

  UInt32 a[4];
  // cpuid was called already. So we don't check for cpuid availability here
  z7_x86_cpuid(a, 0x80000000);

  if (a[0] >= 0x80000004) // if (maxFunc2 >= hi+4) the full name is available
  {
    for (unsigned i = 0; i < 3; i++)
    {
      z7_x86_cpuid(a, (UInt32)(0x80000002 + i));
      for (unsigned j = 0; j < 4; j++)
        PrintCpuChars(s, a[j]);
    }
  }

  s.Trim();
  
  if (s.IsEmpty())
  {
    z7_x86_cpuid(a, 0);
    for (unsigned i = 1; i < 4; i++)
    {
      const unsigned j = (i ^ (i >> 1));
      PrintCpuChars(s, a[j]);
    }
    s.Trim();
  }
}

/*
static void x86cpuid_all_to_String(AString &s)
{
  Cx86cpuid p;
  if (!x86cpuid_CheckAndRead(&p))
    return;
  s += "x86cpuid maxFunc = ";
  s.Add_UInt32(p.maxFunc);
  for (unsigned j = 0; j <= p.maxFunc; j++)
  {
    s.Add_LF();
    // s.Add_UInt32(j); // align
    {
      char temp[32];
      ConvertUInt32ToString(j, temp);
      unsigned len = (unsigned)strlen(temp);
      while (len < 8)
      {
        len++;
        s.Add_Space();
      }
      s += temp;
    }

    s += ":";
    UInt32 d[4] = { 0 };
    MyCPUID(j, &d[0], &d[1], &d[2], &d[3]);
    for (unsigned i = 0; i < 4; i++)
    {
      char temp[32];
      ConvertUInt32ToHex8Digits(d[i], temp);
      s.Add_Space();
      s += temp;
    }
  }
}
*/

#endif



#ifdef _WIN32

static const char * const k_PROCESSOR_ARCHITECTURE[] =
{
    "x86" // "INTEL"
  , "MIPS"
  , "ALPHA"
  , "PPC"
  , "SHX"
  , "ARM"
  , "IA64"
  , "ALPHA64"
  , "MSIL"
  , "x64" // "AMD64"
  , "IA32_ON_WIN64"
  , "NEUTRAL"
  , "ARM64"
  , "ARM32_ON_WIN64"
};

#define Z7_WIN_PROCESSOR_ARCHITECTURE_INTEL 0
#define Z7_WIN_PROCESSOR_ARCHITECTURE_AMD64 9


#define Z7_WIN_PROCESSOR_INTEL_PENTIUM  586
#define Z7_WIN_PROCESSOR_AMD_X8664      8664

/*
static const CUInt32PCharPair k_PROCESSOR[] =
{
  { 2200, "IA64" },
  { 8664, "x64" }
};

#define PROCESSOR_INTEL_386      386
#define PROCESSOR_INTEL_486      486
#define PROCESSOR_INTEL_PENTIUM  586
#define PROCESSOR_INTEL_860      860
#define PROCESSOR_INTEL_IA64     2200
#define PROCESSOR_AMD_X8664      8664
#define PROCESSOR_MIPS_R2000     2000
#define PROCESSOR_MIPS_R3000     3000
#define PROCESSOR_MIPS_R4000     4000
#define PROCESSOR_ALPHA_21064    21064
#define PROCESSOR_PPC_601        601
#define PROCESSOR_PPC_603        603
#define PROCESSOR_PPC_604        604
#define PROCESSOR_PPC_620        620
#define PROCESSOR_HITACHI_SH3    10003
#define PROCESSOR_HITACHI_SH3E   10004
#define PROCESSOR_HITACHI_SH4    10005
#define PROCESSOR_MOTOROLA_821   821
#define PROCESSOR_SHx_SH3        103
#define PROCESSOR_SHx_SH4        104
#define PROCESSOR_STRONGARM      2577    // 0xA11
#define PROCESSOR_ARM720         1824    // 0x720
#define PROCESSOR_ARM820         2080    // 0x820
#define PROCESSOR_ARM920         2336    // 0x920
#define PROCESSOR_ARM_7TDMI      70001
#define PROCESSOR_OPTIL          18767   // 0x494f
*/


/*
static const char * const k_PF[] =
{
    "FP_ERRATA"
  , "FP_EMU"
  , "CMPXCHG"
  , "MMX"
  , "PPC_MOVEMEM_64BIT"
  , "ALPHA_BYTE"
  , "SSE"
  , "3DNOW"
  , "RDTSC"
  , "PAE"
  , "SSE2"
  , "SSE_DAZ"
  , "NX"
  , "SSE3"
  , "CMPXCHG16B"
  , "CMP8XCHG16"
  , "CHANNELS"
  , "XSAVE"
  , "ARM_VFP_32"
  , "ARM_NEON"
  , "L2AT"
  , "VIRT_FIRMWARE"
  , "RDWRFSGSBASE"
  , "FASTFAIL"
  , "ARM_DIVIDE"
  , "ARM_64BIT_LOADSTORE_ATOMIC"
  , "ARM_EXTERNAL_CACHE"
  , "ARM_FMAC"
  , "RDRAND"
  , "ARM_V8"
  , "ARM_V8_CRYPTO"
  , "ARM_V8_CRC32"
  , "RDTSCP"
  , "RDPID"
  , "ARM_V81_ATOMIC"
  , "MONITORX"
};
*/

#endif


static void PrintPage(AString &s, UInt64 v)
{
  const char *t = "B";
       if ((v & ((1 << 20) - 1)) == 0) { v >>= 20;  t = "MB"; }
  else if ((v & ((1 << 10) - 1)) == 0) { v >>= 10;  t = "KB"; }
  s.Add_UInt64(v);
  s += t;
}

#ifdef _WIN32

static AString TypeToString2(const char * const table[], unsigned num, UInt32 value)
{
  char sz[16];
  const char *p = NULL;
  if (value < num)
    p = table[value];
  if (!p)
  {
    ConvertUInt32ToString(value, sz);
    p = sz;
  }
  return (AString)p;
}

// #if defined(Z7_LARGE_PAGES) || defined(_WIN32)
// #ifdef _WIN32
void PrintSize_KMGT_Or_Hex(AString &s, UInt64 v)
{
  char c = 0;
  if ((v & 0x3FF) == 0) { v >>= 10; c = 'K';
  if ((v & 0x3FF) == 0) { v >>= 10; c = 'M';
  if ((v & 0x3FF) == 0) { v >>= 10; c = 'G';
  if ((v & 0x3FF) == 0) { v >>= 10; c = 'T';
  }}}}
  else
  {
    // s += "0x";
    PrintHex(s, v);
    return;
  }
  s.Add_UInt64(v);
  if (c)
    s.Add_Char(c);
  s.Add_Char('B');
}
// #endif
// #endif

static void SysInfo_To_String(AString &s, const SYSTEM_INFO &si)
{
  s += TypeToString2(k_PROCESSOR_ARCHITECTURE, Z7_ARRAY_SIZE(k_PROCESSOR_ARCHITECTURE), si.wProcessorArchitecture);

  if (!( (si.wProcessorArchitecture == Z7_WIN_PROCESSOR_ARCHITECTURE_INTEL && si.dwProcessorType == Z7_WIN_PROCESSOR_INTEL_PENTIUM)
      || (si.wProcessorArchitecture == Z7_WIN_PROCESSOR_ARCHITECTURE_AMD64 && si.dwProcessorType == Z7_WIN_PROCESSOR_AMD_X8664)))
  {
    s.Add_Space();
    // s += TypePairToString(k_PROCESSOR, Z7_ARRAY_SIZE(k_PROCESSOR), si.dwProcessorType);
    s.Add_UInt32(si.dwProcessorType);
  }
  s.Add_Space();
  PrintHex(s, si.wProcessorLevel);
  s.Add_Dot();
  PrintHex(s, si.wProcessorRevision);
  if ((UInt64)si.dwActiveProcessorMask + 1 != ((UInt64)1 << si.dwNumberOfProcessors))
  if ((UInt64)si.dwActiveProcessorMask + 1 != 0 || si.dwNumberOfProcessors != sizeof(UInt64) * 8)
  {
    s += " act:";
    PrintHex(s, si.dwActiveProcessorMask);
  }
  s += " threads:";
  s.Add_UInt32(si.dwNumberOfProcessors);
  if (si.dwPageSize != 1 << 12)
  {
    s += " page:";
    PrintPage(s, si.dwPageSize);
  }
  if (si.dwAllocationGranularity != 1 << 16)
  {
    s += " gran:";
    PrintPage(s, si.dwAllocationGranularity);
  }
  s.Add_Space();

  const DWORD_PTR minAdd = (DWORD_PTR)si.lpMinimumApplicationAddress;
  UInt64 maxSize = (UInt64)(DWORD_PTR)si.lpMaximumApplicationAddress + 1;
  const UInt32 kReserveSize = ((UInt32)1 << 16);
  if (minAdd != kReserveSize)
  {
    PrintSize_KMGT_Or_Hex(s, minAdd);
    s.Add_Minus();
  }
  else
  {
    if ((maxSize & (kReserveSize - 1)) == 0)
      maxSize += kReserveSize;
  }
  PrintSize_KMGT_Or_Hex(s, maxSize);
}

#ifndef _WIN64
EXTERN_C_BEGIN
typedef VOID (WINAPI *Func_GetNativeSystemInfo)(LPSYSTEM_INFO lpSystemInfo);
EXTERN_C_END
#endif

#endif

#ifdef __APPLE__
#ifndef MY_CPU_X86_OR_AMD64
static void Add_sysctlbyname_to_String(const char *name, AString &s)
{
  size_t bufSize = 256;
  char buf[256];
  if (z7_sysctlbyname_Get(name, &buf, &bufSize) == 0)
    s += buf;
}
#endif
#endif

void GetSysInfo(AString &s1, AString &s2);
void GetSysInfo(AString &s1, AString &s2)
{
  s1.Empty();
  s2.Empty();

  #ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    {
      SysInfo_To_String(s1, si);
      // s += " : ";
    }
    
    #if !defined(_WIN64) && !defined(UNDER_CE)
    const
    Func_GetNativeSystemInfo fn = Z7_GET_PROC_ADDRESS(
    Func_GetNativeSystemInfo, GetModuleHandleA("kernel32.dll"),
        "GetNativeSystemInfo");
    if (fn)
    {
      SYSTEM_INFO si2;
      fn(&si2);
      // if (memcmp(&si, &si2, sizeof(si)) != 0)
      {
        // s += " - ";
        SysInfo_To_String(s2, si2);
      }
    }
    #endif
  #endif
}


static void AddBracedString(AString &dest, AString &src)
{
  if (!src.IsEmpty())
  {
    dest.Add_Space_if_NotEmpty();
    dest.Add_Char('(');
    dest += src;
    dest.Add_Char(')');
  }
}

struct CCpuName
{
  AString CpuName;
  AString Revision;
  AString Microcode;
  AString LargePages;

#ifdef _WIN32
  UInt32 MHz;

#ifdef MY_CPU_ARM64
#define Z7_SYS_INFO_SHOW_ARM64_REGS
#endif
#ifdef Z7_SYS_INFO_SHOW_ARM64_REGS
  bool Arm64_ISAR0_EL1_Defined;
  UInt64 Arm64_ISAR0_EL1;
#endif
#endif

#ifdef _WIN32
  CCpuName():
      MHz(0)
#ifdef Z7_SYS_INFO_SHOW_ARM64_REGS
    , Arm64_ISAR0_EL1_Defined(false)
    , Arm64_ISAR0_EL1(0)
#endif
    {}
#endif

  void Fill();

  void Get_Revision_Microcode_LargePages(AString &s)
  {
    s.Empty();
    AddBracedString(s, Revision);
    AddBracedString(s, Microcode);
#ifdef _WIN32
    if (MHz != 0)
    {
      s.Add_Space_if_NotEmpty();
      s.Add_UInt32(MHz);
      s += " MHz";
    }
#endif
    if (!LargePages.IsEmpty())
      s.Add_OptSpaced(LargePages);
  }

#ifdef Z7_SYS_INFO_SHOW_ARM64_REGS
  void Get_Registers(AString &s)
  {
    if (Arm64_ISAR0_EL1_Defined)
    {
      // ID_AA64ISAR0_EL1
      s.Add_OptSpaced("cp4030:");
      PrintHex(s, Arm64_ISAR0_EL1);
      {
        const unsigned sha2 = ((unsigned)(Arm64_ISAR0_EL1 >> 12) & 0xf) - 1;
        if (sha2 < 2)
        {
          s += ":SHA256";
          if (sha2)
            s += ":SHA512";
        }
      }
    }
  }
#endif
};

void CCpuName::Fill()
{
  // CpuName.Empty();
  // Revision.Empty();
  // Microcode.Empty();
  // LargePages.Empty();

  AString &s = CpuName;

  #ifdef MY_CPU_X86_OR_AMD64
  {
    #if !defined(MY_CPU_AMD64)
    if (z7_x86_cpuid_GetMaxFunc())
    #endif
    {
      x86cpuid_to_String(s);
      {
        UInt32 a[4];
        z7_x86_cpuid(a, 1);
        char temp[16];
        ConvertUInt32ToHex(a[0], temp);
        Revision += temp;
      }
    }
  }
  #elif defined(__APPLE__)
  {
    Add_sysctlbyname_to_String("machdep.cpu.brand_string", s);
  }
  #elif defined(MY_CPU_E2K) && defined(Z7_MCST_LCC_VERSION) && (Z7_MCST_LCC_VERSION >= 12323)
  {
    s += "mcst ";
    s += __builtin_cpu_name();
    s.Add_Space();
    s += __builtin_cpu_arch();
  }
  #endif


#ifdef _WIN32
  {
    NRegistry::CKey key;
    if (key.Open(HKEY_LOCAL_MACHINE, TEXT("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"), KEY_READ) == ERROR_SUCCESS)
    {
      // s.Empty(); // for debug
      {
        CSysString name;
        if (s.IsEmpty())
        if (key.QueryValue(TEXT("ProcessorNameString"), name) == ERROR_SUCCESS)
        {
          s += GetAnsiString(name);
        }
        if (key.QueryValue(TEXT("Identifier"), name) == ERROR_SUCCESS)
        {
          if (!Revision.IsEmpty())
            Revision += " : ";
          Revision += GetAnsiString(name);
        }
      }
#ifdef _WIN32
      key.GetValue_UInt32_IfOk(TEXT("~MHz"), MHz);
#ifdef Z7_SYS_INFO_SHOW_ARM64_REGS
/*
mapping arm64 registers to Windows registry:
CP 4000: MIDR_EL1
CP 4020: ID_AA64PFR0_EL1
CP 4021: ID_AA64PFR1_EL1
CP 4028: ID_AA64DFR0_EL1
CP 4029: ID_AA64DFR1_EL1
CP 402C: ID_AA64AFR0_EL1
CP 402D: ID_AA64AFR1_EL1
CP 4030: ID_AA64ISAR0_EL1
CP 4031: ID_AA64ISAR1_EL1
CP 4038: ID_AA64MMFR0_EL1
CP 4039: ID_AA64MMFR1_EL1
CP 403A: ID_AA64MMFR2_EL1
*/
      if (key.GetValue_UInt64_IfOk(TEXT("CP 4030"), Arm64_ISAR0_EL1) == ERROR_SUCCESS)
        Arm64_ISAR0_EL1_Defined = true;
#endif
#endif
      LONG res[2];
      CByteBuffer bufs[2];
      res[0] = key.QueryValue_Binary(TEXT("Previous Update Revision"), bufs[0]);
      res[1] = key.QueryValue_Binary(TEXT("Update Revision"),          bufs[1]);
      if (res[0] == ERROR_SUCCESS || res[1] == ERROR_SUCCESS)
      {
        for (unsigned i = 0; i < 2; i++)
        {
          if (i == 1)
            Microcode += "->";
          if (res[i] != ERROR_SUCCESS)
            continue;
          const CByteBuffer &buf = bufs[i];
          if (buf.Size() == 8)
          {
            const UInt32 high = GetUi32(buf);
            if (high != 0)
            {
              PrintHex(Microcode, high);
              Microcode.Add_Dot();
            }
            PrintHex(Microcode, GetUi32(buf + 4));
          }
        }
      }
    }
  }
#endif

  if (s.IsEmpty())
  {
    #ifdef MY_CPU_NAME
      s += MY_CPU_NAME;
    #endif
  }
  
  #ifdef __APPLE__
  {
    AString s2;
    UInt32 v = 0;
    if (z7_sysctlbyname_Get_UInt32("machdep.cpu.core_count", &v) == 0)
    {
      s2.Add_UInt32(v);
      s2.Add_Char('C');
    }
    if (z7_sysctlbyname_Get_UInt32("machdep.cpu.thread_count", &v) == 0)
    {
      s2.Add_UInt32(v);
      s2.Add_Char('T');
    }
    if (!s2.IsEmpty())
    {
      s.Add_Space_if_NotEmpty();
      s += s2;
    }
  }
  #endif

  #ifdef Z7_LARGE_PAGES
  Add_LargePages_String(LargePages);
  #endif
}


#if 0 && defined(Z7_LARGE_PAGES) && defined(__linux__)
bool Get_HugePageSize(UInt64 &pageSize);
bool Get_HugePageSize(UInt64 &pageSize)
{
  AString s2;
  if (ReadFile_to_String("/sys/kernel/mm/transparent_hugepage/hpage_pmd_size", s2))
  {
    pageSize = ConvertStringToUInt64(s2.Ptr(), NULL);
    if (pageSize)
      return true;
  }
  return false;
}
#endif


void AddCpuFeatures(AString &s);
void AddCpuFeatures(AString &s)
{
  #ifdef _WIN32
  // const unsigned kNumFeatures_Extra = 32; // we check also for unknown features
  // const unsigned kNumFeatures = Z7_ARRAY_SIZE(k_PF) + kNumFeatures_Extra;
  const unsigned kNumFeatures = 64;
  UInt64 flags = 0;
  for (unsigned i = 0; i < kNumFeatures; i++)
  {
    if (IsProcessorFeaturePresent((DWORD)i))
    {
      flags += (UInt64)1 << i;
      // s.Add_Space_if_NotEmpty();
      // s += TypeToString2(k_PF, Z7_ARRAY_SIZE(k_PF), i);
    }
  }
  s.Add_OptSpaced("f:");
  PrintHex(s, flags);
  
  #elif defined(__APPLE__)
  {
    UInt32 v = 0;
    if (z7_sysctlbyname_Get_UInt32("hw.pagesize", &v) == 0)
    {
      s.Add_OptSpaced("PageSize:");
      PrintPage(s, v);
    }
  }

  #else

  const long v = sysconf(_SC_PAGESIZE);
  if (v != -1)
  {
    s.Add_OptSpaced("PageSize:");
    PrintPage(s, (unsigned long)v);
  }

  #if !defined(_AIX)

  #ifdef __linux__

  {
    AString s2;
    if (ReadFile_to_String("/proc/meminfo", s2))
    {
      const int pos = s2.Find("Hugepagesize:");
      if (pos >= 0)
      {
        s.Add_OptSpaced("HPS:");
        s2.DeleteFrontal((unsigned)pos + 13); // 13 == strlen("Hugepagesize:")
        s2.TrimLeft();
        // const int pos2 = s2.Find("kB");
        const UInt64 size = ConvertStringToUInt64(s2.Ptr(), NULL);
        if (size)
          PrintPage(s, size << 10);
      }
    }
    
    if (ReadFile_to_String("/sys/kernel/mm/transparent_hugepage/hpage_pmd_size", s2))
    {
      s.Add_OptSpaced("THPS:");
      const UInt64 size = ConvertStringToUInt64(s2.Ptr(), NULL);
      if (size)
        PrintPage(s, size);
    }
    /*
    {
      UInt64 pagesSize;
      if (Get_HugePageSize(pagesSize) && pagesSize)
      {
        s.Add_OptSpaced("THPS:");
        PrintPage(s, pagesSize);
      }
    }
    */
    
    if (ReadFile_to_String("/sys/kernel/mm/transparent_hugepage/enabled", s2))
    {
      s.Add_OptSpaced("THP:");
      const int pos = s2.Find('[');
      if (pos >= 0)
      {
        const int pos2 = s2.Find(']', (unsigned)pos + 1);
        if (pos2 >= 0)
        {
          s2.DeleteFrom((unsigned)pos2);
          s2.DeleteFrontal((unsigned)pos + 1);
        }
      }
      s += s2;
    }
  }
  // else throw CSystemException(MY_SRes_HRESULT_FROM_WRes(errno));

  #endif


  #ifdef AT_HWCAP
  s.Add_OptSpaced("hwcap:");
  {
    unsigned long h = MY_getauxval(AT_HWCAP);
    PrintHex(s, h);
    #ifdef MY_CPU_ARM64
#ifndef HWCAP_SHA3
#define HWCAP_SHA3    (1 << 17)
#endif
#ifndef HWCAP_SHA512
#define HWCAP_SHA512  (1 << 21)
// #pragma message("=== HWCAP_SHA512 define === ")
#endif
    if (h & HWCAP_CRC32)  s += ":CRC32";
    if (h & HWCAP_SHA1)   s += ":SHA1";
    if (h & HWCAP_SHA2)   s += ":SHA2";
    if (h & HWCAP_SHA3)   s += ":SHA3";
    if (h & HWCAP_SHA512) s += ":SHA512";
    if (h & HWCAP_AES)    s += ":AES";
    if (h & HWCAP_ASIMD)  s += ":ASIMD";
    #elif defined(MY_CPU_ARM)
    if (h & HWCAP_NEON)   s += ":NEON";
    #endif
  }
  #endif // AT_HWCAP
 
  #ifdef AT_HWCAP2
  {
    unsigned long h = MY_getauxval(AT_HWCAP2);
    #ifndef MY_CPU_ARM
    if (h != 0)
    #endif
    {
      s += " hwcap2:";
      PrintHex(s, h);
      #ifdef MY_CPU_ARM
      if (h & HWCAP2_CRC32)  s += ":CRC32";
      if (h & HWCAP2_SHA1)   s += ":SHA1";
      if (h & HWCAP2_SHA2)   s += ":SHA2";
      if (h & HWCAP2_AES)    s += ":AES";
      #endif
    }
  }
  #endif // AT_HWCAP2
  #endif // _AIX
  #endif // _WIN32
}


#ifdef _WIN32
#ifndef UNDER_CE

Z7_DIAGNOSTIC_IGNORE_CAST_FUNCTION

EXTERN_C_BEGIN
typedef void (WINAPI * Func_RtlGetVersion) (OSVERSIONINFOEXW *);
EXTERN_C_END

static BOOL My_RtlGetVersion(OSVERSIONINFOEXW *vi)
{
  const HMODULE ntdll = ::GetModuleHandleW(L"ntdll.dll");
  if (!ntdll)
    return FALSE;
  const
  Func_RtlGetVersion func = Z7_GET_PROC_ADDRESS(
  Func_RtlGetVersion, ntdll,
      "RtlGetVersion");
  if (!func)
    return FALSE;
  func(vi);
  return TRUE;
}

#endif
#endif


void GetOsInfoText(AString &sRes)
{
  sRes.Empty();
    AString s;

    #ifdef _WIN32
    #ifndef UNDER_CE
      // OSVERSIONINFO vi;
      OSVERSIONINFOEXW vi;
      vi.dwOSVersionInfoSize = sizeof(vi);
      // if (::GetVersionEx(&vi))
      if (My_RtlGetVersion(&vi))
      {
        s += "Windows";
        if (vi.dwPlatformId != VER_PLATFORM_WIN32_NT)
          s.Add_UInt32(vi.dwPlatformId);
        s.Add_Space(); s.Add_UInt32(vi.dwMajorVersion);
        s.Add_Dot();   s.Add_UInt32(vi.dwMinorVersion);
        s.Add_Space(); s.Add_UInt32(vi.dwBuildNumber);

        if (vi.wServicePackMajor != 0 || vi.wServicePackMinor != 0)
        {
          s += " SP:"; s.Add_UInt32(vi.wServicePackMajor);
          s.Add_Dot(); s.Add_UInt32(vi.wServicePackMinor);
        }
        // s += " Suite:"; PrintHex(s, vi.wSuiteMask);
        // s += " Type:"; s.Add_UInt32(vi.wProductType);
        // s.Add_Space(); s += GetOemString(vi.szCSDVersion);
      }
      /*
      {
        s += " OEMCP:"; s.Add_UInt32(GetOEMCP());
        s += " ACP:";   s.Add_UInt32(GetACP());
      }
      */
    #endif
    #else // _WIN32

      if (!s.IsEmpty())
        s.Add_LF();
      struct utsname un;
      if (uname(&un) == 0)
      {
        s += un.sysname;
        // s += " : "; s += un.nodename; // we don't want to show name of computer
        s += " : "; s += un.release;
        s += " : "; s += un.version;
        s += " : "; s += un.machine;

        #ifdef __APPLE__
          // Add_sysctlbyname_to_String("kern.version", s);
          // it's same as "utsname.version"
        #endif
      }
    #endif  // _WIN32

    sRes += s;
  #ifdef MY_CPU_X86_OR_AMD64
  {
    AString s2;
    GetVirtCpuid(s2);
    if (!s2.IsEmpty())
    {
      sRes += " : ";
      sRes += s2;
    }
  }
  #endif
}



void GetSystemInfoText(AString &sRes)
{
  GetOsInfoText(sRes);
  sRes.Add_LF();

    {
      AString s, s1, s2;
      GetSysInfo(s1, s2);
      if (!s1.IsEmpty() || !s2.IsEmpty())
      {
        s = s1;
        if (s1 != s2 && !s2.IsEmpty())
        {
          s += " - ";
          s += s2;
        }
      }
      {
        AddCpuFeatures(s);
        if (!s.IsEmpty())
        {
          sRes += s;
          sRes.Add_LF();
        }
      }
    }
    {
      AString s, registers;
      GetCpuName_MultiLine(s, registers);
      if (!s.IsEmpty())
      {
        sRes += s;
        sRes.Add_LF();
      }
      if (!registers.IsEmpty())
      {
        sRes += registers;
        sRes.Add_LF();
      }
    }
    /*
    #ifdef MY_CPU_X86_OR_AMD64
    {
      AString s;
      x86cpuid_all_to_String(s);
      if (!s.IsEmpty())
      {
        printCallback->Print(s);
        printCallback->NewLine();
      }
    }
    #endif
    */
}


void GetCpuName_MultiLine(AString &s, AString &registers);
void GetCpuName_MultiLine(AString &s, AString &registers)
{
  CCpuName cpuName;
  cpuName.Fill();
  s = cpuName.CpuName;
  AString s2;
  cpuName.Get_Revision_Microcode_LargePages(s2);
  if (!s2.IsEmpty())
  {
    s.Add_LF();
    s += s2;
  }
  registers.Empty();
#ifdef Z7_SYS_INFO_SHOW_ARM64_REGS
  cpuName.Get_Registers(registers);
#endif
}


#ifdef MY_CPU_X86_OR_AMD64

void GetVirtCpuid(AString &s)
{
  const UInt32 kHv = 0x40000000;

  Z7_IF_X86_CPUID_SUPPORTED
  {
    UInt32 a[4];
    z7_x86_cpuid(a, kHv);
    
    if (a[0] < kHv || a[0] >= kHv + (1 << 16))
      return;
    {
      {
        for (unsigned j = 1; j < 4; j++)
          PrintCpuChars(s, a[j]);
      }
    }
    if (a[0] >= kHv + 1)
    {
      UInt32 d[4];
      z7_x86_cpuid(d, kHv + 1);
      s += " : ";
      PrintCpuChars(s, d[0]);
      if (a[0] >= kHv + 2)
      {
        z7_x86_cpuid(d, kHv + 2);
        s += " : ";
        s.Add_UInt32(d[1] >> 16);
        s.Add_Dot();  s.Add_UInt32(d[1] & 0xffff);
        s.Add_Dot();  s.Add_UInt32(d[0]);
        s.Add_Dot();  s.Add_UInt32(d[2]);
        s.Add_Dot();  s.Add_UInt32(d[3] >> 24);
        s.Add_Dot();  s.Add_UInt32(d[3] & 0xffffff);
      }
      /*
      if (a[0] >= kHv + 5)
      {
        z7_x86_cpuid(d, kHv + 5);
        s += " : ";
        s.Add_UInt32(d[0]);
        s += "p";
        s.Add_UInt32(d[1]);
        s += "t";
      }
      */
    }
  }
}

#endif


void GetCompiler(AString &s)
{
  #ifdef __clang__
    s += " CLANG ";
    s.Add_UInt32(__clang_major__);
    s.Add_Dot();
    s.Add_UInt32(__clang_minor__);
    s.Add_Dot();
    s.Add_UInt32(__clang_patchlevel__);
  #endif

  #ifdef __xlC__
    s += " XLC ";
    s.Add_UInt32(__xlC__ >> 8);
    s.Add_Dot();
    s.Add_UInt32(__xlC__ & 0xFF);
    #ifdef __xlC_ver__
      s.Add_Dot();
      s.Add_UInt32(__xlC_ver__ >> 8);
      s.Add_Dot();
      s.Add_UInt32(__xlC_ver__ & 0xFF);
    #endif
  #endif

  // #define __LCC__ 126
  // #define __LCC_MINOR__ 20
  // #define __MCST__ 1
  #ifdef __MCST__
    s += " MCST";
  #endif
  #ifdef __LCC__
    s += " LCC ";
    s.Add_UInt32(__LCC__ / 100);
    s.Add_Dot();
    s.Add_UInt32(__LCC__ % 100 / 10);
    s.Add_UInt32(__LCC__ % 10);
    #ifdef __LCC_MINOR__
      s.Add_Dot();
      s.Add_UInt32(__LCC_MINOR__ / 10);
      s.Add_UInt32(__LCC_MINOR__ % 10);
    #endif
  #endif

  // #define __EDG_VERSION__ 602
  #ifdef __EDG_VERSION__
    s += " EDG ";
    s.Add_UInt32(__EDG_VERSION__ / 100);
    s.Add_Dot();
    s.Add_UInt32(__EDG_VERSION__ % 100 / 10);
    s.Add_UInt32(__EDG_VERSION__ % 10);
  #endif

  #ifdef __VERSION__
    s.Add_Space();
    s += "ver:";
    s += __VERSION__;
  #endif

  #ifdef __GNUC__
    s += " GCC ";
    s.Add_UInt32(__GNUC__);
    s.Add_Dot();
    s.Add_UInt32(__GNUC_MINOR__);
    s.Add_Dot();
    s.Add_UInt32(__GNUC_PATCHLEVEL__);
  #endif


  #ifdef _MSC_VER
    s += " MSC ";
    s.Add_UInt32(_MSC_VER);
    #ifdef _MSC_FULL_VER
      s.Add_Dot();
      s.Add_UInt32(_MSC_FULL_VER);
    #endif
      
  #endif

    #if defined(__AVX512F__)
      #if defined(__AVX512VL__)
        #define MY_CPU_COMPILE_ISA "AVX512VL"
      #else
        #define MY_CPU_COMPILE_ISA "AVX512F"
      #endif
    #elif defined(__AVX2__)
      #define MY_CPU_COMPILE_ISA "AVX2"
    #elif defined(__AVX__)
      #define MY_CPU_COMPILE_ISA "AVX"
    #elif defined(__SSE2__)
      #define MY_CPU_COMPILE_ISA "SSE2"
    #elif defined(_M_IX86_FP) && (_M_IX86_FP >= 2)
      #define MY_CPU_COMPILE_ISA "SSE2"
    #elif defined(__SSE__)
      #define MY_CPU_COMPILE_ISA "SSE"
    #elif defined(_M_IX86_FP) && (_M_IX86_FP >= 1)
      #define MY_CPU_COMPILE_ISA "SSE"
    #elif defined(__i686__)
      #define MY_CPU_COMPILE_ISA "i686"
    #elif defined(__i586__)
      #define MY_CPU_COMPILE_ISA "i586"
    #elif defined(__i486__)
      #define MY_CPU_COMPILE_ISA "i486"
    #elif defined(__i386__)
      #define MY_CPU_COMPILE_ISA "i386"
    #elif defined(_M_IX86_FP)
      #define MY_CPU_COMPILE_ISA "IA32"
    #endif

  AString s2;

  #ifdef MY_CPU_COMPILE_ISA
    s2.Add_OptSpaced(MY_CPU_COMPILE_ISA);
  #endif

#ifndef MY_CPU_ARM64
  #ifdef __ARM_FP
    s2.Add_OptSpaced("FP");
  #endif
  #ifdef __ARM_NEON
    s2.Add_OptSpaced("NEON");
  #endif
  #ifdef __NEON__
    s2.Add_OptSpaced("__NEON__");
  #endif
  #ifdef __ARM_FEATURE_SIMD32
    s2.Add_OptSpaced("SIMD32");
  #endif
#endif

  #ifdef __ARM_FEATURE_CRYPTO
    s2.Add_OptSpaced("CRYPTO");
  #endif

  #ifdef __ARM_FEATURE_SHA2
    s2.Add_OptSpaced("SHA2");
  #endif

  #ifdef __ARM_FEATURE_AES
    s2.Add_OptSpaced("AES");
  #endif

  #ifdef __ARM_FEATURE_CRC32
    s2.Add_OptSpaced("CRC32");
  #endif

  #ifdef __ARM_FEATURE_UNALIGNED
    s2.Add_OptSpaced("UNALIGNED");
  #endif


  #ifdef MY_CPU_BE
    s2.Add_OptSpaced("BE");
  #endif

  #if defined(MY_CPU_LE_UNALIGN) \
      && !defined(MY_CPU_X86_OR_AMD64) \
      && !defined(MY_CPU_ARM64)
    s2.Add_OptSpaced("LE-unaligned");
  #endif

  if (!s2.IsEmpty())
  {
    s.Add_OptSpaced(": ");
    s += s2;
  }
}
