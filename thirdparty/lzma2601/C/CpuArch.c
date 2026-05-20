/* CpuArch.c -- CPU specific code
Igor Pavlov : Public domain */

#include "Precomp.h"

// #include <stdio.h>

#include "CpuArch.h"

#ifdef MY_CPU_X86_OR_AMD64

#undef NEED_CHECK_FOR_CPUID
#if !defined(MY_CPU_AMD64)
#define NEED_CHECK_FOR_CPUID
#endif

/*
  cpuid instruction supports (subFunction) parameter in ECX,
  that is used only with some specific (function) parameter values.
  most functions use only (subFunction==0).
*/
/*
  __cpuid(): MSVC and GCC/CLANG use same function/macro name
             but parameters are different.
   We use MSVC __cpuid() parameters style for our z7_x86_cpuid() function.
*/

#if defined(__GNUC__) /* && (__GNUC__ >= 10) */ \
    || defined(__clang__) /* && (__clang_major__ >= 10) */

/* there was some CLANG/GCC compilers that have issues with
   rbx(ebx) handling in asm blocks in -fPIC mode (__PIC__ is defined).
   compiler's <cpuid.h> contains the macro __cpuid() that is similar to our code.
   The history of __cpuid() changes in CLANG/GCC:
   GCC:
     2007: it preserved ebx for (__PIC__ && __i386__)
     2013: it preserved rbx and ebx for __PIC__
     2014: it doesn't preserves rbx and ebx anymore
     we suppose that (__GNUC__ >= 5) fixed that __PIC__ ebx/rbx problem.
   CLANG:
     2014+: it preserves rbx, but only for 64-bit code. No __PIC__ check.
   Why CLANG cares about 64-bit mode only, and doesn't care about ebx (in 32-bit)?
   Do we need __PIC__ test for CLANG or we must care about rbx even if
   __PIC__ is not defined?
*/

#define ASM_LN "\n"
   
#if defined(MY_CPU_AMD64) && defined(__PIC__) \
    && ((defined (__GNUC__) && (__GNUC__ < 5)) || defined(__clang__))

  /* "=&r" selects free register. It can select even rbx, if that register is free.
     "=&D" for (RDI) also works, but the code can be larger with "=&D"
     "2"(subFun) : 2 is (zero-based) index in the output constraint list "=c" (ECX). */

#define x86_cpuid_MACRO_2(p, func, subFunc) { \
  __asm__ __volatile__ ( \
    ASM_LN   "mov     %%rbx, %q1"  \
    ASM_LN   "cpuid"               \
    ASM_LN   "xchg    %%rbx, %q1"  \
    : "=a" ((p)[0]), "=&r" ((p)[1]), "=c" ((p)[2]), "=d" ((p)[3]) : "0" (func), "2"(subFunc)); }

#elif defined(MY_CPU_X86) && defined(__PIC__) \
    && ((defined (__GNUC__) && (__GNUC__ < 5)) || defined(__clang__))

#define x86_cpuid_MACRO_2(p, func, subFunc) { \
  __asm__ __volatile__ ( \
    ASM_LN   "mov     %%ebx, %k1"  \
    ASM_LN   "cpuid"               \
    ASM_LN   "xchg    %%ebx, %k1"  \
    : "=a" ((p)[0]), "=&r" ((p)[1]), "=c" ((p)[2]), "=d" ((p)[3]) : "0" (func), "2"(subFunc)); }

#else

#define x86_cpuid_MACRO_2(p, func, subFunc) { \
  __asm__ __volatile__ ( \
    ASM_LN   "cpuid"               \
    : "=a" ((p)[0]), "=b" ((p)[1]), "=c" ((p)[2]), "=d" ((p)[3]) : "0" (func), "2"(subFunc)); }

#endif

#define x86_cpuid_MACRO(p, func)  x86_cpuid_MACRO_2(p, func, 0)

void Z7_FASTCALL z7_x86_cpuid(UInt32 p[4], UInt32 func)
{
  x86_cpuid_MACRO(p, func)
}

static
void Z7_FASTCALL z7_x86_cpuid_subFunc(UInt32 p[4], UInt32 func, UInt32 subFunc)
{
  x86_cpuid_MACRO_2(p, func, subFunc)
}


Z7_NO_INLINE
UInt32 Z7_FASTCALL z7_x86_cpuid_GetMaxFunc(void)
{
 #if defined(NEED_CHECK_FOR_CPUID)
  #define EFALGS_CPUID_BIT 21
  UInt32 a;
  __asm__ __volatile__ (
    ASM_LN   "pushf"
    ASM_LN   "pushf"
    ASM_LN   "pop     %0"
    // ASM_LN   "movl    %0, %1"
    // ASM_LN   "xorl    $0x200000, %0"
    ASM_LN   "btc     %1, %0"
    ASM_LN   "push    %0"
    ASM_LN   "popf"
    ASM_LN   "pushf"
    ASM_LN   "pop     %0"
    ASM_LN   "xorl    (%%esp), %0"

    ASM_LN   "popf"
    ASM_LN
    : "=&r" (a) // "=a"
    : "i" (EFALGS_CPUID_BIT)
    );
  if ((a & (1 << EFALGS_CPUID_BIT)) == 0)
    return 0;
 #endif
  {
    UInt32 p[4];
    x86_cpuid_MACRO(p, 0)
    return p[0];
  }
}

#undef ASM_LN

#elif !defined(_MSC_VER)

/*
// for gcc/clang and other: we can try to use __cpuid macro:
#include <cpuid.h>
void Z7_FASTCALL z7_x86_cpuid(UInt32 p[4], UInt32 func)
{
  __cpuid(func, p[0], p[1], p[2], p[3]);
}
UInt32 Z7_FASTCALL z7_x86_cpuid_GetMaxFunc(void)
{
  return (UInt32)__get_cpuid_max(0, NULL);
}
*/
// for unsupported cpuid:
void Z7_FASTCALL z7_x86_cpuid(UInt32 p[4], UInt32 func)
{
  UNUSED_VAR(func)
  p[0] = p[1] = p[2] = p[3] = 0;
}
UInt32 Z7_FASTCALL z7_x86_cpuid_GetMaxFunc(void)
{
  return 0;
}

#else // _MSC_VER

#if !defined(MY_CPU_AMD64)

UInt32 __declspec(naked) Z7_FASTCALL z7_x86_cpuid_GetMaxFunc(void)
{
  #if defined(NEED_CHECK_FOR_CPUID)
  #define EFALGS_CPUID_BIT 21
  __asm   pushfd
  __asm   pushfd
  /*
  __asm   pop     eax
  // __asm   mov     edx, eax
  __asm   btc     eax, EFALGS_CPUID_BIT
  __asm   push    eax
  */
  __asm   btc     dword ptr [esp], EFALGS_CPUID_BIT
  __asm   popfd
  __asm   pushfd
  __asm   pop     eax
  // __asm   xor     eax, edx
  __asm   xor     eax, [esp]
  // __asm   push    edx
  __asm   popfd
  __asm   and     eax, (1 shl EFALGS_CPUID_BIT)
  __asm   jz end_func
  #endif
  __asm   push    ebx
  __asm   xor     eax, eax    // func
  __asm   xor     ecx, ecx    // subFunction (optional) for (func == 0)
  __asm   cpuid
  __asm   pop     ebx
  #if defined(NEED_CHECK_FOR_CPUID)
  end_func:
  #endif
  __asm   ret 0
}

void __declspec(naked) Z7_FASTCALL z7_x86_cpuid(UInt32 p[4], UInt32 func)
{
  UNUSED_VAR(p)
  UNUSED_VAR(func)
  __asm   push    ebx
  __asm   push    edi
  __asm   mov     edi, ecx    // p
  __asm   mov     eax, edx    // func
  __asm   xor     ecx, ecx    // subfunction (optional) for (func == 0)
  __asm   cpuid
  __asm   mov     [edi     ], eax
  __asm   mov     [edi +  4], ebx
  __asm   mov     [edi +  8], ecx
  __asm   mov     [edi + 12], edx
  __asm   pop     edi
  __asm   pop     ebx
  __asm   ret     0
}

static
void __declspec(naked) Z7_FASTCALL z7_x86_cpuid_subFunc(UInt32 p[4], UInt32 func, UInt32 subFunc)
{
  UNUSED_VAR(p)
  UNUSED_VAR(func)
  UNUSED_VAR(subFunc)
  __asm   push    ebx
  __asm   push    edi
  __asm   mov     edi, ecx    // p
  __asm   mov     eax, edx    // func
  __asm   mov     ecx, [esp + 12]  // subFunc
  __asm   cpuid
  __asm   mov     [edi     ], eax
  __asm   mov     [edi +  4], ebx
  __asm   mov     [edi +  8], ecx
  __asm   mov     [edi + 12], edx
  __asm   pop     edi
  __asm   pop     ebx
  __asm   ret     4
}

#else // MY_CPU_AMD64

    #if _MSC_VER >= 1600
      #include <intrin.h>
      #define MY_cpuidex  __cpuidex

static
void Z7_FASTCALL z7_x86_cpuid_subFunc(UInt32 p[4], UInt32 func, UInt32 subFunc)
{
  __cpuidex((int *)p, func, subFunc);
}

    #else
/*
 __cpuid (func == (0 or 7)) requires subfunction number in ECX.
  MSDN: The __cpuid intrinsic clears the ECX register before calling the cpuid instruction.
   __cpuid() in new MSVC clears ECX.
   __cpuid() in old MSVC (14.00) x64 doesn't clear ECX
 We still can use __cpuid for low (func) values that don't require ECX,
 but __cpuid() in old MSVC will be incorrect for some func values: (func == 7).
 So here we use the hack for old MSVC to send (subFunction) in ECX register to cpuid instruction,
 where ECX value is first parameter for FASTCALL / NO_INLINE func.
 So the caller of MY_cpuidex_HACK() sets ECX as subFunction, and
 old MSVC for __cpuid() doesn't change ECX and cpuid instruction gets (subFunction) value.
 
DON'T remove Z7_NO_INLINE and Z7_FASTCALL for MY_cpuidex_HACK(): !!!
*/
static
Z7_NO_INLINE void Z7_FASTCALL MY_cpuidex_HACK(Int32 subFunction, Int32 func, Int32 *CPUInfo)
{
  UNUSED_VAR(subFunction)
  __cpuid(CPUInfo, func);
}
      #define MY_cpuidex(info, func, func2)  MY_cpuidex_HACK(func2, func, info)
      #pragma message("======== MY_cpuidex_HACK WAS USED ========")
static
void Z7_FASTCALL z7_x86_cpuid_subFunc(UInt32 p[4], UInt32 func, UInt32 subFunc)
{
  MY_cpuidex_HACK(subFunc, func, (Int32 *)p);
}
    #endif // _MSC_VER >= 1600

#if !defined(MY_CPU_AMD64)
/* inlining for __cpuid() in MSVC x86 (32-bit) produces big ineffective code,
   so we disable inlining here */
Z7_NO_INLINE
#endif
void Z7_FASTCALL z7_x86_cpuid(UInt32 p[4], UInt32 func)
{
  MY_cpuidex((Int32 *)p, (Int32)func, 0);
}

Z7_NO_INLINE
UInt32 Z7_FASTCALL z7_x86_cpuid_GetMaxFunc(void)
{
  Int32 a[4];
  MY_cpuidex(a, 0, 0);
  return a[0];
}

#endif // MY_CPU_AMD64
#endif // _MSC_VER

#if defined(NEED_CHECK_FOR_CPUID)
#define CHECK_CPUID_IS_SUPPORTED { if (z7_x86_cpuid_GetMaxFunc() == 0) return 0; }
#else
#define CHECK_CPUID_IS_SUPPORTED
#endif
#undef NEED_CHECK_FOR_CPUID


static
BoolInt x86cpuid_Func_1(UInt32 *p)
{
  CHECK_CPUID_IS_SUPPORTED
  z7_x86_cpuid(p, 1);
  return True;
}

/*
static const UInt32 kVendors[][1] =
{
  { 0x756E6547 }, // , 0x49656E69, 0x6C65746E },
  { 0x68747541 }, // , 0x69746E65, 0x444D4163 },
  { 0x746E6543 }  // , 0x48727561, 0x736C7561 }
};
*/

/*
typedef struct
{
  UInt32 maxFunc;
  UInt32 vendor[3];
  UInt32 ver;
  UInt32 b;
  UInt32 c;
  UInt32 d;
} Cx86cpuid;

enum
{
  CPU_FIRM_INTEL,
  CPU_FIRM_AMD,
  CPU_FIRM_VIA
};
int x86cpuid_GetFirm(const Cx86cpuid *p);
#define x86cpuid_ver_GetFamily(ver) (((ver >> 16) & 0xff0) | ((ver >> 8) & 0xf))
#define x86cpuid_ver_GetModel(ver)  (((ver >> 12) &  0xf0) | ((ver >> 4) & 0xf))
#define x86cpuid_ver_GetStepping(ver) (ver & 0xf)

int x86cpuid_GetFirm(const Cx86cpuid *p)
{
  unsigned i;
  for (i = 0; i < sizeof(kVendors) / sizeof(kVendors[0]); i++)
  {
    const UInt32 *v = kVendors[i];
    if (v[0] == p->vendor[0]
        // && v[1] == p->vendor[1]
        // && v[2] == p->vendor[2]
        )
      return (int)i;
  }
  return -1;
}

BoolInt CPU_Is_InOrder()
{
  Cx86cpuid p;
  UInt32 family, model;
  if (!x86cpuid_CheckAndRead(&p))
    return True;

  family = x86cpuid_ver_GetFamily(p.ver);
  model = x86cpuid_ver_GetModel(p.ver);

  switch (x86cpuid_GetFirm(&p))
  {
    case CPU_FIRM_INTEL: return (family < 6 || (family == 6 && (
        // In-Order Atom CPU
           model == 0x1C  // 45 nm, N4xx, D4xx, N5xx, D5xx, 230, 330
        || model == 0x26  // 45 nm, Z6xx
        || model == 0x27  // 32 nm, Z2460
        || model == 0x35  // 32 nm, Z2760
        || model == 0x36  // 32 nm, N2xxx, D2xxx
        )));
    case CPU_FIRM_AMD: return (family < 5 || (family == 5 && (model < 6 || model == 0xA)));
    case CPU_FIRM_VIA: return (family < 6 || (family == 6 && model < 0xF));
  }
  return False; // v23 : unknown processors are not In-Order
}
*/

#ifdef _WIN32
#include "7zWindows.h"
#endif

#if !defined(MY_CPU_AMD64) && defined(_WIN32)

/* for legacy SSE ia32: there is no user-space cpu instruction to check
   that OS supports SSE register storing/restoring on context switches.
   So we need some OS-specific function to check that it's safe to use SSE registers.
*/

Z7_FORCE_INLINE
static BoolInt CPU_Sys_Is_SSE_Supported(void)
{
#ifdef _MSC_VER
  #pragma warning(push)
  #pragma warning(disable : 4996) // `GetVersion': was declared deprecated
#endif
  /* low byte is major version of Windows
     We suppose that any Windows version since
     Windows2000 (major == 5) supports SSE registers */
  return (Byte)GetVersion() >= 5;
#if defined(_MSC_VER)
  #pragma warning(pop)
#endif
}
#define CHECK_SYS_SSE_SUPPORT if (!CPU_Sys_Is_SSE_Supported()) return False;
#else
#define CHECK_SYS_SSE_SUPPORT
#endif


#if !defined(MY_CPU_AMD64)

BoolInt CPU_IsSupported_CMOV(void)
{
  UInt32 a[4];
  if (!x86cpuid_Func_1(&a[0]))
    return 0;
  return (BoolInt)(a[3] >> 15) & 1;
}

BoolInt CPU_IsSupported_SSE(void)
{
  UInt32 a[4];
  CHECK_SYS_SSE_SUPPORT
  if (!x86cpuid_Func_1(&a[0]))
    return 0;
  return (BoolInt)(a[3] >> 25) & 1;
}

BoolInt CPU_IsSupported_SSE2(void)
{
  UInt32 a[4];
  CHECK_SYS_SSE_SUPPORT
  if (!x86cpuid_Func_1(&a[0]))
    return 0;
  return (BoolInt)(a[3] >> 26) & 1;
}

#endif


static UInt32 x86cpuid_Func_1_ECX(void)
{
  UInt32 a[4];
  CHECK_SYS_SSE_SUPPORT
  if (!x86cpuid_Func_1(&a[0]))
    return 0;
  return a[2];
}

BoolInt CPU_IsSupported_AES(void)
{
  return (BoolInt)(x86cpuid_Func_1_ECX() >> 25) & 1;
}

BoolInt CPU_IsSupported_SSSE3(void)
{
  return (BoolInt)(x86cpuid_Func_1_ECX() >> 9) & 1;
}

BoolInt CPU_IsSupported_SSE41(void)
{
  return (BoolInt)(x86cpuid_Func_1_ECX() >> 19) & 1;
}

BoolInt CPU_IsSupported_SHA(void)
{
  CHECK_SYS_SSE_SUPPORT

  if (z7_x86_cpuid_GetMaxFunc() < 7)
    return False;
  {
    UInt32 d[4];
    z7_x86_cpuid(d, 7);
    return (BoolInt)(d[1] >> 29) & 1;
  }
}


BoolInt CPU_IsSupported_SHA512(void)
{
  if (!CPU_IsSupported_AVX2()) return False; // maybe CPU_IsSupported_AVX() is enough here

  if (z7_x86_cpuid_GetMaxFunc() < 7)
    return False;
  {
    UInt32 d[4];
    z7_x86_cpuid_subFunc(d, 7, 0);
    if (d[0] < 1) // d[0] - is max supported subleaf value
      return False;
    z7_x86_cpuid_subFunc(d, 7, 1);
    return (BoolInt)(d[0]) & 1;
  }
}

/*
MSVC: _xgetbv() intrinsic is available since VS2010SP1.
   MSVC also defines (_XCR_XFEATURE_ENABLED_MASK) macro in
   <immintrin.h> that we can use or check.
   For any 32-bit x86 we can use asm code in MSVC,
   but MSVC asm code is huge after compilation.
   So _xgetbv() is better

ICC: _xgetbv() intrinsic is available (in what version of ICC?)
   ICC defines (__GNUC___) and it supports gnu assembler
   also ICC supports MASM style code with -use-msasm switch.
   but ICC doesn't support __attribute__((__target__))

GCC/CLANG 9:
  _xgetbv() is macro that works via __builtin_ia32_xgetbv()
  and we need __attribute__((__target__("xsave")).
  But with __target__("xsave") the function will be not
  inlined to function that has no __target__("xsave") attribute.
  If we want _xgetbv() call inlining, then we should use asm version
  instead of calling _xgetbv().
  Note:intrinsic is broke before GCC 8.2:
    https://gcc.gnu.org/bugzilla/show_bug.cgi?id=85684
*/

#if    defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 1100) \
    || defined(_MSC_VER) && (_MSC_VER >= 1600) && (_MSC_FULL_VER >= 160040219)  \
    || defined(__GNUC__) && (__GNUC__ >= 9) \
    || defined(__clang__) && (__clang_major__ >= 9)
// we define ATTRIB_XGETBV, if we want to use predefined _xgetbv() from compiler
#if defined(__INTEL_COMPILER)
#define ATTRIB_XGETBV
#elif defined(__GNUC__) || defined(__clang__)
// we don't define ATTRIB_XGETBV here, because asm version is better for inlining.
// #define ATTRIB_XGETBV __attribute__((__target__("xsave")))
#else
#define ATTRIB_XGETBV
#endif
#endif

#if defined(ATTRIB_XGETBV)
#include <immintrin.h>
#endif


// XFEATURE_ENABLED_MASK/XCR0
#define MY_XCR_XFEATURE_ENABLED_MASK 0

#if defined(ATTRIB_XGETBV)
ATTRIB_XGETBV
#endif
static UInt64 x86_xgetbv_0(UInt32 num)
{
#if defined(ATTRIB_XGETBV)
  {
    return
      #if (defined(_MSC_VER))
        _xgetbv(num);
      #else
        __builtin_ia32_xgetbv(
          #if !defined(__clang__)
            (int)
          #endif
            num);
      #endif
  }

#elif defined(__GNUC__) || defined(__clang__) || defined(__SUNPRO_CC)

  UInt32 a, d;
 #if defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 4))
  __asm__
  (
    "xgetbv"
    : "=a"(a), "=d"(d) : "c"(num) : "cc"
  );
 #else // is old gcc
  __asm__
  (
    ".byte 0x0f, 0x01, 0xd0" "\n\t"
    : "=a"(a), "=d"(d) : "c"(num) : "cc"
  );
 #endif
  return ((UInt64)d << 32) | a;
  // return a;

#elif defined(_MSC_VER) && !defined(MY_CPU_AMD64)
  
  UInt32 a, d;
  __asm {
    push eax
    push edx
    push ecx
    mov ecx, num;
    // xor ecx, ecx // = MY_XCR_XFEATURE_ENABLED_MASK
    _emit 0x0f
    _emit 0x01
    _emit 0xd0
    mov a, eax
    mov d, edx
    pop ecx
    pop edx
    pop eax
  }
  return ((UInt64)d << 32) | a;
  // return a;

#else // it's unknown compiler
  // #error "Need xgetbv function"
  UNUSED_VAR(num)
  // for MSVC-X64 we could call external function from external file.
  /* Actually we had checked OSXSAVE/AVX in cpuid before.
     So it's expected that OS supports at least AVX and below. */
  // if (num != MY_XCR_XFEATURE_ENABLED_MASK) return 0; // if not XCR0
  return
      // (1 << 0) |  // x87
        (1 << 1)   // SSE
      | (1 << 2);  // AVX
  
#endif
}

#ifdef _WIN32
/*
  Windows versions do not know about new ISA extensions that
  can be introduced. But we still can use new extensions,
  even if Windows doesn't report about supporting them,
  But we can use new extensions, only if Windows knows about new ISA extension
  that changes the number or size of registers: SSE, AVX/XSAVE, AVX512
  So it's enough to check
    MY_PF_AVX_INSTRUCTIONS_AVAILABLE
      instead of
    MY_PF_AVX2_INSTRUCTIONS_AVAILABLE
*/
#define MY_PF_XSAVE_ENABLED                            17
// #define MY_PF_SSSE3_INSTRUCTIONS_AVAILABLE             36
// #define MY_PF_SSE4_1_INSTRUCTIONS_AVAILABLE            37
// #define MY_PF_SSE4_2_INSTRUCTIONS_AVAILABLE            38
// #define MY_PF_AVX_INSTRUCTIONS_AVAILABLE               39
// #define MY_PF_AVX2_INSTRUCTIONS_AVAILABLE              40
// #define MY_PF_AVX512F_INSTRUCTIONS_AVAILABLE           41
#endif

BoolInt CPU_IsSupported_AVX(void)
{
  #ifdef _WIN32
  if (!IsProcessorFeaturePresent(MY_PF_XSAVE_ENABLED))
    return False;
  /* PF_AVX_INSTRUCTIONS_AVAILABLE probably is supported starting from
     some latest Win10 revisions. But we need AVX in older Windows also.
     So we don't use the following check: */
  /*
  if (!IsProcessorFeaturePresent(MY_PF_AVX_INSTRUCTIONS_AVAILABLE))
    return False;
  */
  #endif

  /*
    OS must use new special XSAVE/XRSTOR instructions to save
    AVX registers when it required for context switching.
    At OS statring:
      OS sets CR4.OSXSAVE flag to signal the processor that OS supports the XSAVE extensions.
      Also OS sets bitmask in XCR0 register that defines what
      registers will be processed by XSAVE instruction:
        XCR0.SSE[bit 0] - x87 registers and state
        XCR0.SSE[bit 1] - SSE registers and state
        XCR0.AVX[bit 2] - AVX registers and state
    CR4.OSXSAVE is reflected to CPUID.1:ECX.OSXSAVE[bit 27].
       So we can read that bit in user-space.
    XCR0 is available for reading in user-space by new XGETBV instruction.
  */
  {
    const UInt32 c = x86cpuid_Func_1_ECX();
    if (0 == (1
        & (c >> 28)   // AVX instructions are supported by hardware
        & (c >> 27))) // OSXSAVE bit: XSAVE and related instructions are enabled by OS.
      return False;
  }

  /* also we can check
     CPUID.1:ECX.XSAVE [bit 26] : that shows that
        XSAVE, XRESTOR, XSETBV, XGETBV instructions are supported by hardware.
     But that check is redundant, because if OSXSAVE bit is set, then XSAVE is also set */

  /* If OS have enabled XSAVE extension instructions (OSXSAVE == 1),
     in most cases we expect that OS also will support storing/restoring
     for AVX and SSE states at least.
     But to be ensure for that we call user-space instruction
     XGETBV(0) to get XCR0 value that contains bitmask that defines
     what exact states(registers) OS have enabled for storing/restoring.
  */

  {
    const UInt32 bm = (UInt32)x86_xgetbv_0(MY_XCR_XFEATURE_ENABLED_MASK);
    // printf("\n=== XGetBV=0x%x\n", bm);
    return 1
        & (BoolInt)(bm >> 1)  // SSE state is supported (set by OS) for storing/restoring
        & (BoolInt)(bm >> 2); // AVX state is supported (set by OS) for storing/restoring
  }
  // since Win7SP1: we can use GetEnabledXStateFeatures();
}


BoolInt CPU_IsSupported_AVX2(void)
{
  if (!CPU_IsSupported_AVX())
    return False;
  if (z7_x86_cpuid_GetMaxFunc() < 7)
    return False;
  {
    UInt32 d[4];
    z7_x86_cpuid(d, 7);
    // printf("\ncpuid(7): ebx=%8x ecx=%8x\n", d[1], d[2]);
    return 1
      & (BoolInt)(d[1] >> 5); // avx2
  }
}

#if 0
BoolInt CPU_IsSupported_AVX512F_AVX512VL(void)
{
  if (!CPU_IsSupported_AVX())
    return False;
  if (z7_x86_cpuid_GetMaxFunc() < 7)
    return False;
  {
    UInt32 d[4];
    BoolInt v;
    z7_x86_cpuid(d, 7);
    // printf("\ncpuid(7): ebx=%8x ecx=%8x\n", d[1], d[2]);
    v = 1
      & (BoolInt)(d[1] >> 16)  // avx512f
      & (BoolInt)(d[1] >> 31); // avx512vl
    if (!v)
      return False;
  }
  {
    const UInt32 bm = (UInt32)x86_xgetbv_0(MY_XCR_XFEATURE_ENABLED_MASK);
    // printf("\n=== XGetBV=0x%x\n", bm);
    return 1
        & (BoolInt)(bm >> 5)  // OPMASK
        & (BoolInt)(bm >> 6)  // ZMM upper 256-bit
        & (BoolInt)(bm >> 7); // ZMM16 ... ZMM31
  }
}
#endif

BoolInt CPU_IsSupported_VAES_AVX2(void)
{
  if (!CPU_IsSupported_AVX())
    return False;
  if (z7_x86_cpuid_GetMaxFunc() < 7)
    return False;
  {
    UInt32 d[4];
    z7_x86_cpuid(d, 7);
    // printf("\ncpuid(7): ebx=%8x ecx=%8x\n", d[1], d[2]);
    return 1
      & (BoolInt)(d[1] >> 5) // avx2
      // & (d[1] >> 31) // avx512vl
      & (BoolInt)(d[2] >> 9); // vaes // VEX-256/EVEX
  }
}

BoolInt CPU_IsSupported_PageGB(void)
{
  CHECK_CPUID_IS_SUPPORTED
  {
    UInt32 d[4];
    z7_x86_cpuid(d, 0x80000000);
    if (d[0] < 0x80000001)
      return False;
    z7_x86_cpuid(d, 0x80000001);
    return (BoolInt)(d[3] >> 26) & 1;
  }
}


#elif defined(MY_CPU_ARM_OR_ARM64)

#ifdef _WIN32

#include "7zWindows.h"

BoolInt CPU_IsSupported_CRC32(void)  { return IsProcessorFeaturePresent(PF_ARM_V8_CRC32_INSTRUCTIONS_AVAILABLE) ? 1 : 0; }
BoolInt CPU_IsSupported_CRYPTO(void) { return IsProcessorFeaturePresent(PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE) ? 1 : 0; }
BoolInt CPU_IsSupported_NEON(void)   { return IsProcessorFeaturePresent(PF_ARM_NEON_INSTRUCTIONS_AVAILABLE) ? 1 : 0; }

#else

#if defined(__APPLE__)

/*
#include <stdio.h>
#include <string.h>
static void Print_sysctlbyname(const char *name)
{
  size_t bufSize = 256;
  char buf[256];
  int res = sysctlbyname(name, &buf, &bufSize, NULL, 0);
  {
    int i;
    printf("\nres = %d : %s : '%s' : bufSize = %d, numeric", res, name, buf, (unsigned)bufSize);
    for (i = 0; i < 20; i++)
      printf(" %2x", (unsigned)(Byte)buf[i]);

  }
}
*/
/*
  Print_sysctlbyname("hw.pagesize");
  Print_sysctlbyname("machdep.cpu.brand_string");
*/

static BoolInt z7_sysctlbyname_Get_BoolInt(const char *name)
{
  UInt32 val = 0;
  if (z7_sysctlbyname_Get_UInt32(name, &val) == 0 && val == 1)
    return 1;
  return 0;
}

BoolInt CPU_IsSupported_CRC32(void)
{
  return z7_sysctlbyname_Get_BoolInt("hw.optional.armv8_crc32");
}

BoolInt CPU_IsSupported_NEON(void)
{
  return z7_sysctlbyname_Get_BoolInt("hw.optional.neon");
}

BoolInt CPU_IsSupported_SHA512(void)
{
  return z7_sysctlbyname_Get_BoolInt("hw.optional.armv8_2_sha512");
}

/*
BoolInt CPU_IsSupported_SHA3(void)
{
  return z7_sysctlbyname_Get_BoolInt("hw.optional.armv8_2_sha3");
}
*/

#ifdef MY_CPU_ARM64
#define APPLE_CRYPTO_SUPPORT_VAL 1
#else
#define APPLE_CRYPTO_SUPPORT_VAL 0
#endif

BoolInt CPU_IsSupported_SHA1(void) { return APPLE_CRYPTO_SUPPORT_VAL; }
BoolInt CPU_IsSupported_SHA2(void) { return APPLE_CRYPTO_SUPPORT_VAL; }
BoolInt CPU_IsSupported_AES (void) { return APPLE_CRYPTO_SUPPORT_VAL; }


#else // __APPLE__

#if defined(__GLIBC__) && (__GLIBC__ * 100 + __GLIBC_MINOR__ >= 216)
  #define Z7_GETAUXV_AVAILABLE
#elif !defined(__QNXNTO__)
// #pragma message("=== is not NEW GLIBC === ")
  #if defined __has_include
  #if __has_include (<sys/auxv.h>)
// #pragma message("=== sys/auxv.h is avail=== ")
    #define Z7_GETAUXV_AVAILABLE
  #endif
  #endif
#endif

#ifdef Z7_GETAUXV_AVAILABLE
// #pragma message("=== Z7_GETAUXV_AVAILABLE === ")
#include <sys/auxv.h>
#define USE_HWCAP
#endif

#ifdef USE_HWCAP

#if defined(__FreeBSD__) || defined(__OpenBSD__)
static unsigned long MY_getauxval(int aux)
{
  unsigned long val;
  if (elf_aux_info(aux, &val, sizeof(val)))
    return 0;
  return val;
}
#else
#define MY_getauxval  getauxval
  #if defined __has_include
  #if __has_include (<asm/hwcap.h>)
#include <asm/hwcap.h>
  #endif
  #endif
#endif

  #define MY_HWCAP_CHECK_FUNC_2(name1, name2) \
  BoolInt CPU_IsSupported_ ## name1(void) { return (MY_getauxval(AT_HWCAP)  & (HWCAP_  ## name2)); }

#ifdef MY_CPU_ARM64
  #define MY_HWCAP_CHECK_FUNC(name) \
  MY_HWCAP_CHECK_FUNC_2(name, name)
#if 1 || defined(__ARM_NEON)
  BoolInt CPU_IsSupported_NEON(void) { return True; }
#else
  MY_HWCAP_CHECK_FUNC_2(NEON, ASIMD)
#endif
// MY_HWCAP_CHECK_FUNC (ASIMD)
#elif defined(MY_CPU_ARM)
  #define MY_HWCAP_CHECK_FUNC(name) \
  BoolInt CPU_IsSupported_ ## name(void) { return (MY_getauxval(AT_HWCAP2) & (HWCAP2_ ## name)); }
  MY_HWCAP_CHECK_FUNC_2(NEON, NEON)
#endif

#else // USE_HWCAP

  #define MY_HWCAP_CHECK_FUNC(name) \
  BoolInt CPU_IsSupported_ ## name(void) { return 0; }
#if defined(__ARM_NEON)
  BoolInt CPU_IsSupported_NEON(void) { return True; }
#else
  MY_HWCAP_CHECK_FUNC(NEON)
#endif

#endif // USE_HWCAP

MY_HWCAP_CHECK_FUNC (CRC32)
MY_HWCAP_CHECK_FUNC (SHA1)
MY_HWCAP_CHECK_FUNC (SHA2)
MY_HWCAP_CHECK_FUNC (AES)
#ifdef MY_CPU_ARM64
// <hwcap.h> supports HWCAP_SHA512 and HWCAP_SHA3 since 2017.
// we define them here, if they are not defined
#ifndef HWCAP_SHA3
// #define HWCAP_SHA3    (1 << 17)
#endif
#ifndef HWCAP_SHA512
// #pragma message("=== HWCAP_SHA512 define === ")
#define HWCAP_SHA512  (1 << 21)
#endif
MY_HWCAP_CHECK_FUNC (SHA512)
// MY_HWCAP_CHECK_FUNC (SHA3)
#endif

#endif // __APPLE__
#endif // _WIN32

#endif // MY_CPU_ARM_OR_ARM64



#ifdef __APPLE__

#include <sys/sysctl.h>

int z7_sysctlbyname_Get(const char *name, void *buf, size_t *bufSize)
{
  return sysctlbyname(name, buf, bufSize, NULL, 0);
}

int z7_sysctlbyname_Get_UInt32(const char *name, UInt32 *val)
{
  size_t bufSize = sizeof(*val);
  const int res = z7_sysctlbyname_Get(name, val, &bufSize);
  if (res == 0 && bufSize != sizeof(*val))
    return EFAULT;
  return res;
}

#endif
