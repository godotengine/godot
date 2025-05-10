/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

#include "SDL_internal.h"

#include "SDL_cpuinfo_c.h"

#if defined(SDL_PLATFORM_WINDOWS)
#include "../core/windows/SDL_windows.h"
#endif

// CPU feature detection for SDL

#ifdef HAVE_SYSCONF
#include <unistd.h>
#endif
#ifdef HAVE_SYSCTLBYNAME
#include <sys/types.h>
#include <sys/sysctl.h>
#endif
#if defined(SDL_PLATFORM_MACOS) && (defined(__ppc__) || defined(__ppc64__))
#include <sys/sysctl.h> // For AltiVec check
#elif defined(SDL_PLATFORM_OPENBSD) && defined(__powerpc__)
#include <sys/types.h>
#include <sys/sysctl.h> // For AltiVec check
#include <machine/cpu.h>
#elif defined(SDL_PLATFORM_FREEBSD) && defined(__powerpc__)
#include <machine/cpu.h>
#include <sys/auxv.h>
#elif defined(SDL_ALTIVEC_BLITTERS) && defined(HAVE_SETJMP)
#include <signal.h>
#include <setjmp.h>
#endif

#if (defined(SDL_PLATFORM_LINUX) || defined(SDL_PLATFORM_ANDROID)) && defined(__arm__)
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <elf.h>

// #include <asm/hwcap.h>
#ifndef AT_HWCAP
#define AT_HWCAP 16
#endif
#ifndef AT_PLATFORM
#define AT_PLATFORM 15
#endif
#ifndef HWCAP_NEON
#define HWCAP_NEON (1 << 12)
#endif
#endif

#if defined (SDL_PLATFORM_FREEBSD)
#include <sys/param.h>
#endif

#if defined(SDL_PLATFORM_ANDROID) && defined(__arm__) && !defined(HAVE_GETAUXVAL)
#include <cpu-features.h>
#endif

#if defined(HAVE_GETAUXVAL) || defined(HAVE_ELF_AUX_INFO)
#include <sys/auxv.h>
#endif

#ifdef SDL_PLATFORM_RISCOS
#include <kernel.h>
#include <swis.h>
#endif
#ifdef SDL_PLATFORM_3DS
#include <3ds.h>
#endif
#ifdef SDL_PLATFORM_PS2
#include <kernel.h>
#endif

#ifdef SDL_PLATFORM_HAIKU
#include <kernel/OS.h>
#endif

#define CPU_HAS_ALTIVEC  (1 << 0)
#define CPU_HAS_MMX      (1 << 1)
#define CPU_HAS_SSE      (1 << 2)
#define CPU_HAS_SSE2     (1 << 3)
#define CPU_HAS_SSE3     (1 << 4)
#define CPU_HAS_SSE41    (1 << 5)
#define CPU_HAS_SSE42    (1 << 6)
#define CPU_HAS_AVX      (1 << 7)
#define CPU_HAS_AVX2     (1 << 8)
#define CPU_HAS_NEON     (1 << 9)
#define CPU_HAS_AVX512F  (1 << 10)
#define CPU_HAS_ARM_SIMD (1 << 11)
#define CPU_HAS_LSX      (1 << 12)
#define CPU_HAS_LASX     (1 << 13)

#define CPU_CFG2      0x2
#define CPU_CFG2_LSX  (1 << 6)
#define CPU_CFG2_LASX (1 << 7)

#if defined(SDL_ALTIVEC_BLITTERS) && defined(HAVE_SETJMP) && !defined(SDL_PLATFORM_MACOS) && !defined(SDL_PLATFORM_OPENBSD) && !defined(SDL_PLATFORM_FREEBSD)
/* This is the brute force way of detecting instruction sets...
   the idea is borrowed from the libmpeg2 library - thanks!
 */
static jmp_buf jmpbuf;
static void illegal_instruction(int sig)
{
    longjmp(jmpbuf, 1);
}
#endif // HAVE_SETJMP

static int CPU_haveCPUID(void)
{
    int has_CPUID = 0;

/* *INDENT-OFF* */ // clang-format off
#ifndef SDL_PLATFORM_EMSCRIPTEN
#if (defined(__GNUC__) || defined(__llvm__)) && defined(__i386__)
    __asm__ (
"        pushfl                      # Get original EFLAGS             \n"
"        popl    %%eax                                                 \n"
"        movl    %%eax,%%ecx                                           \n"
"        xorl    $0x200000,%%eax     # Flip ID bit in EFLAGS           \n"
"        pushl   %%eax               # Save new EFLAGS value on stack  \n"
"        popfl                       # Replace current EFLAGS value    \n"
"        pushfl                      # Get new EFLAGS                  \n"
"        popl    %%eax               # Store new EFLAGS in EAX         \n"
"        xorl    %%ecx,%%eax         # Can not toggle ID bit,          \n"
"        jz      1f                  # Processor=80486                 \n"
"        movl    $1,%0               # We have CPUID support           \n"
"1:                                                                    \n"
    : "=m" (has_CPUID)
    :
    : "%eax", "%ecx"
    );
#elif (defined(__GNUC__) || defined(__llvm__)) && defined(__x86_64__)
/* Technically, if this is being compiled under __x86_64__ then it has
   CPUid by definition.  But it's nice to be able to prove it.  :)      */
    __asm__ (
"        pushfq                      # Get original EFLAGS             \n"
"        popq    %%rax                                                 \n"
"        movq    %%rax,%%rcx                                           \n"
"        xorl    $0x200000,%%eax     # Flip ID bit in EFLAGS           \n"
"        pushq   %%rax               # Save new EFLAGS value on stack  \n"
"        popfq                       # Replace current EFLAGS value    \n"
"        pushfq                      # Get new EFLAGS                  \n"
"        popq    %%rax               # Store new EFLAGS in EAX         \n"
"        xorl    %%ecx,%%eax         # Can not toggle ID bit,          \n"
"        jz      1f                  # Processor=80486                 \n"
"        movl    $1,%0               # We have CPUID support           \n"
"1:                                                                    \n"
    : "=m" (has_CPUID)
    :
    : "%rax", "%rcx"
    );
#elif (defined(_MSC_VER) && defined(_M_IX86)) || defined(__WATCOMC__)
    __asm {
        pushfd                      ; Get original EFLAGS
        pop     eax
        mov     ecx, eax
        xor     eax, 200000h        ; Flip ID bit in EFLAGS
        push    eax                 ; Save new EFLAGS value on stack
        popfd                       ; Replace current EFLAGS value
        pushfd                      ; Get new EFLAGS
        pop     eax                 ; Store new EFLAGS in EAX
        xor     eax, ecx            ; Can not toggle ID bit,
        jz      done                ; Processor=80486
        mov     has_CPUID,1         ; We have CPUID support
done:
    }
#elif defined(_MSC_VER) && defined(_M_X64)
    has_CPUID = 1;
#elif defined(__sun) && defined(__i386)
    __asm (
"       pushfl                 \n"
"       popl    %eax           \n"
"       movl    %eax,%ecx      \n"
"       xorl    $0x200000,%eax \n"
"       pushl   %eax           \n"
"       popfl                  \n"
"       pushfl                 \n"
"       popl    %eax           \n"
"       xorl    %ecx,%eax      \n"
"       jz      1f             \n"
"       movl    $1,-8(%ebp)    \n"
"1:                            \n"
    );
#elif defined(__sun) && defined(__amd64)
    __asm (
"       pushfq                 \n"
"       popq    %rax           \n"
"       movq    %rax,%rcx      \n"
"       xorl    $0x200000,%eax \n"
"       pushq   %rax           \n"
"       popfq                  \n"
"       pushfq                 \n"
"       popq    %rax           \n"
"       xorl    %ecx,%eax      \n"
"       jz      1f             \n"
"       movl    $1,-8(%rbp)    \n"
"1:                            \n"
    );
#endif
#endif // !SDL_PLATFORM_EMSCRIPTEN
/* *INDENT-ON* */ // clang-format on
    return has_CPUID;
}

#if (defined(__GNUC__) || defined(__llvm__)) && defined(__i386__)
#define cpuid(func, a, b, c, d)              \
    __asm__ __volatile__(                    \
        "        pushl %%ebx        \n"      \
        "        xorl %%ecx,%%ecx   \n"      \
        "        cpuid              \n"      \
        "        movl %%ebx, %%esi  \n"      \
        "        popl %%ebx         \n"      \
        : "=a"(a), "=S"(b), "=c"(c), "=d"(d) \
        : "a"(func))
#elif (defined(__GNUC__) || defined(__llvm__)) && defined(__x86_64__)
#define cpuid(func, a, b, c, d)              \
    __asm__ __volatile__(                    \
        "        pushq %%rbx        \n"      \
        "        xorq %%rcx,%%rcx   \n"      \
        "        cpuid              \n"      \
        "        movq %%rbx, %%rsi  \n"      \
        "        popq %%rbx         \n"      \
        : "=a"(a), "=S"(b), "=c"(c), "=d"(d) \
        : "a"(func))
#elif (defined(_MSC_VER) && defined(_M_IX86)) || defined(__WATCOMC__)
#define cpuid(func, a, b, c, d) \
    __asm { \
        __asm mov eax, func \
        __asm xor ecx, ecx \
        __asm cpuid \
        __asm mov a, eax \
        __asm mov b, ebx \
        __asm mov c, ecx \
        __asm mov d, edx                   \
    }
#elif (defined(_MSC_VER) && defined(_M_X64))
// Use __cpuidex instead of __cpuid because ICL does not clear ecx register
#define cpuid(func, a, b, c, d)      \
    {                                \
        int CPUInfo[4];              \
        __cpuidex(CPUInfo, func, 0); \
        a = CPUInfo[0];              \
        b = CPUInfo[1];              \
        c = CPUInfo[2];              \
        d = CPUInfo[3];              \
    }
#else
#define cpuid(func, a, b, c, d) \
    do {                        \
        a = b = c = d = 0;      \
        (void)a;                \
        (void)b;                \
        (void)c;                \
        (void)d;                \
    } while (0)
#endif

static int CPU_CPUIDFeatures[4];
static int CPU_CPUIDMaxFunction = 0;
static bool CPU_OSSavesYMM = false;
static bool CPU_OSSavesZMM = false;

static void CPU_calcCPUIDFeatures(void)
{
    static bool checked = false;
    if (!checked) {
        checked = true;
        if (CPU_haveCPUID()) {
            int a, b, c, d;
            cpuid(0, a, b, c, d);
            CPU_CPUIDMaxFunction = a;
            if (CPU_CPUIDMaxFunction >= 1) {
                cpuid(1, a, b, c, d);
                CPU_CPUIDFeatures[0] = a;
                CPU_CPUIDFeatures[1] = b;
                CPU_CPUIDFeatures[2] = c;
                CPU_CPUIDFeatures[3] = d;

                // Check to make sure we can call xgetbv
                if (c & 0x08000000) {
                    // Call xgetbv to see if YMM (etc) register state is saved
#if (defined(__GNUC__) || defined(__llvm__)) && (defined(__i386__) || defined(__x86_64__))
                    __asm__(".byte 0x0f, 0x01, 0xd0"
                            : "=a"(a)
                            : "c"(0)
                            : "%edx");
#elif defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64)) && (_MSC_FULL_VER >= 160040219) // VS2010 SP1
                    a = (int)_xgetbv(0);
#elif (defined(_MSC_VER) && defined(_M_IX86)) || defined(__WATCOMC__)
                    __asm
                        {
                        xor ecx, ecx
                        _asm _emit 0x0f _asm _emit 0x01 _asm _emit 0xd0
                        mov a, eax
                        }
#endif
                    CPU_OSSavesYMM = ((a & 6) == 6) ? true : false;
                    CPU_OSSavesZMM = (CPU_OSSavesYMM && ((a & 0xe0) == 0xe0)) ? true : false;
                }
            }
        }
    }
}

static int CPU_haveAltiVec(void)
{
    volatile int altivec = 0;
#ifndef SDL_CPUINFO_DISABLED
#if (defined(SDL_PLATFORM_MACOS) && (defined(__ppc__) || defined(__ppc64__))) || (defined(SDL_PLATFORM_OPENBSD) && defined(__powerpc__))
#ifdef SDL_PLATFORM_OPENBSD
    int selectors[2] = { CTL_MACHDEP, CPU_ALTIVEC };
#else
    int selectors[2] = { CTL_HW, HW_VECTORUNIT };
#endif
    int hasVectorUnit = 0;
    size_t length = sizeof(hasVectorUnit);
    int error = sysctl(selectors, 2, &hasVectorUnit, &length, NULL, 0);
    if (0 == error) {
        altivec = (hasVectorUnit != 0);
    }
#elif defined(SDL_PLATFORM_FREEBSD) && defined(__powerpc__)
    unsigned long cpufeatures = 0;
    elf_aux_info(AT_HWCAP, &cpufeatures, sizeof(cpufeatures));
    altivec = cpufeatures & PPC_FEATURE_HAS_ALTIVEC;
    return altivec;
#elif defined(SDL_ALTIVEC_BLITTERS) && defined(HAVE_SETJMP)
    void (*handler)(int sig);
    handler = signal(SIGILL, illegal_instruction);
    if (setjmp(jmpbuf) == 0) {
        asm volatile("mtspr 256, %0\n\t"
                     "vand %%v0, %%v0, %%v0" ::"r"(-1));
        altivec = 1;
    }
    signal(SIGILL, handler);
#endif
#endif
    return altivec;
}

#if (defined(__ARM_ARCH) && (__ARM_ARCH >= 6)) || defined(__aarch64__)
static int CPU_haveARMSIMD(void)
{
    return 1;
}

#elif !defined(__arm__)
static int CPU_haveARMSIMD(void)
{
    return 0;
}

#elif defined(SDL_PLATFORM_LINUX)
static int CPU_haveARMSIMD(void)
{
    int arm_simd = 0;
    int fd;

    fd = open("/proc/self/auxv", O_RDONLY | O_CLOEXEC);
    if (fd >= 0) {
        Elf32_auxv_t aux;
        while (read(fd, &aux, sizeof(aux)) == sizeof(aux)) {
            if (aux.a_type == AT_PLATFORM) {
                const char *plat = (const char *)aux.a_un.a_val;
                if (plat) {
                    arm_simd = SDL_strncmp(plat, "v6l", 3) == 0 ||
                               SDL_strncmp(plat, "v7l", 3) == 0;
                }
            }
        }
        close(fd);
    }
    return arm_simd;
}

#elif defined(SDL_PLATFORM_RISCOS)
static int CPU_haveARMSIMD(void)
{
    _kernel_swi_regs regs;
    regs.r[0] = 0;
    if (_kernel_swi(OS_PlatformFeatures, &regs, &regs) != NULL) {
        return 0;
    }

    if (!(regs.r[0] & (1 << 31))) {
        return 0;
    }

    regs.r[0] = 34;
    regs.r[1] = 29;
    if (_kernel_swi(OS_PlatformFeatures, &regs, &regs) != NULL) {
        return 0;
    }

    return regs.r[0];
}

#else
static int CPU_haveARMSIMD(void)
{
#warning SDL_HasARMSIMD is not implemented for this ARM platform. Write me.
    return 0;
}
#endif

#if defined(SDL_PLATFORM_LINUX) && defined(__arm__) && !defined(HAVE_GETAUXVAL)
static int readProcAuxvForNeon(void)
{
    int neon = 0;
    int fd;

    fd = open("/proc/self/auxv", O_RDONLY | O_CLOEXEC);
    if (fd >= 0) {
        Elf32_auxv_t aux;
        while (read(fd, &aux, sizeof(aux)) == sizeof(aux)) {
            if (aux.a_type == AT_HWCAP) {
                neon = (aux.a_un.a_val & HWCAP_NEON) == HWCAP_NEON;
                break;
            }
        }
        close(fd);
    }
    return neon;
}
#endif

static int CPU_haveNEON(void)
{
/* The way you detect NEON is a privileged instruction on ARM, so you have
   query the OS kernel in a platform-specific way. :/ */
#if defined(SDL_PLATFORM_WINDOWS) && (defined(_M_ARM) || defined(_M_ARM64))
// Visual Studio, for ARM, doesn't define __ARM_ARCH. Handle this first.
// Seems to have been removed
#ifndef PF_ARM_NEON_INSTRUCTIONS_AVAILABLE
#define PF_ARM_NEON_INSTRUCTIONS_AVAILABLE 19
#endif
    // All WinRT ARM devices are required to support NEON, but just in case.
    return IsProcessorFeaturePresent(PF_ARM_NEON_INSTRUCTIONS_AVAILABLE) != 0;
#elif (defined(__ARM_ARCH) && (__ARM_ARCH >= 8)) || defined(__aarch64__)
    return 1; // ARMv8 always has non-optional NEON support.
#elif defined(SDL_PLATFORM_VITA)
    return 1;
#elif defined(SDL_PLATFORM_3DS)
    return 0;
#elif defined(SDL_PLATFORM_APPLE) && defined(__ARM_ARCH) && (__ARM_ARCH >= 7)
    // (note that sysctlbyname("hw.optional.neon") doesn't work!)
    return 1; // all Apple ARMv7 chips and later have NEON.
#elif defined(SDL_PLATFORM_APPLE)
    return 0; // assume anything else from Apple doesn't have NEON.
#elif !defined(__arm__)
    return 0; // not an ARM CPU at all.
#elif defined(SDL_PLATFORM_OPENBSD)
    return 1; // OpenBSD only supports ARMv7 CPUs that have NEON.
#elif defined(HAVE_ELF_AUX_INFO)
    unsigned long hasneon = 0;
    if (elf_aux_info(AT_HWCAP, (void *)&hasneon, (int)sizeof(hasneon)) != 0) {
        return 0;
    }
    return (hasneon & HWCAP_NEON) == HWCAP_NEON;
#elif (defined(SDL_PLATFORM_LINUX) || defined(SDL_PLATFORM_ANDROID)) && defined(HAVE_GETAUXVAL)
    return (getauxval(AT_HWCAP) & HWCAP_NEON) == HWCAP_NEON;
#elif defined(SDL_PLATFORM_LINUX)
    return readProcAuxvForNeon();
#elif defined(SDL_PLATFORM_ANDROID)
    // Use NDK cpufeatures to read either /proc/self/auxv or /proc/cpuinfo
    {
        AndroidCpuFamily cpu_family = android_getCpuFamily();
        if (cpu_family == ANDROID_CPU_FAMILY_ARM) {
            uint64_t cpu_features = android_getCpuFeatures();
            if (cpu_features & ANDROID_CPU_ARM_FEATURE_NEON) {
                return 1;
            }
        }
        return 0;
    }
#elif defined(SDL_PLATFORM_RISCOS)
    // Use the VFPSupport_Features SWI to access the MVFR registers
    {
        _kernel_swi_regs regs;
        regs.r[0] = 0;
        if (_kernel_swi(VFPSupport_Features, &regs, &regs) == NULL) {
            if ((regs.r[2] & 0xFFF000) == 0x111000) {
                return 1;
            }
        }
        return 0;
    }
#elif defined(SDL_PLATFORM_EMSCRIPTEN)
    return 0;
#else
#warning SDL_HasNEON is not implemented for this ARM platform. Write me.
    return 0;
#endif
}

static int CPU_readCPUCFG(void)
{
    uint32_t cfg2 = 0;
#if defined __loongarch__
    __asm__ volatile(
        "cpucfg %0, %1 \n\t"
        : "+&r"(cfg2)
        : "r"(CPU_CFG2));
#endif
    return cfg2;
}

#define CPU_haveLSX()  (CPU_readCPUCFG() & CPU_CFG2_LSX)
#define CPU_haveLASX() (CPU_readCPUCFG() & CPU_CFG2_LASX)

#ifdef __e2k__
#ifdef __MMX__
#define CPU_haveMMX() (1)
#else
#define CPU_haveMMX() (0)
#endif
#ifdef __SSE__
#define CPU_haveSSE() (1)
#else
#define CPU_haveSSE() (0)
#endif
#ifdef __SSE2__
#define CPU_haveSSE2() (1)
#else
#define CPU_haveSSE2() (0)
#endif
#ifdef __SSE3__
#define CPU_haveSSE3() (1)
#else
#define CPU_haveSSE3() (0)
#endif
#ifdef __SSE4_1__
#define CPU_haveSSE41() (1)
#else
#define CPU_haveSSE41() (0)
#endif
#ifdef __SSE4_2__
#define CPU_haveSSE42() (1)
#else
#define CPU_haveSSE42() (0)
#endif
#ifdef __AVX__
#define CPU_haveAVX() (1)
#else
#define CPU_haveAVX() (0)
#endif
#else
#define CPU_haveMMX()   (CPU_CPUIDFeatures[3] & 0x00800000)
#define CPU_haveSSE()   (CPU_CPUIDFeatures[3] & 0x02000000)
#define CPU_haveSSE2()  (CPU_CPUIDFeatures[3] & 0x04000000)
#define CPU_haveSSE3()  (CPU_CPUIDFeatures[2] & 0x00000001)
#define CPU_haveSSE41() (CPU_CPUIDFeatures[2] & 0x00080000)
#define CPU_haveSSE42() (CPU_CPUIDFeatures[2] & 0x00100000)
#define CPU_haveAVX()   (CPU_OSSavesYMM && (CPU_CPUIDFeatures[2] & 0x10000000))
#endif

#ifdef __e2k__
inline int
CPU_haveAVX2(void)
{
#ifdef __AVX2__
    return 1;
#else
    return 0;
#endif
}
#else
static int CPU_haveAVX2(void)
{
    if (CPU_OSSavesYMM && (CPU_CPUIDMaxFunction >= 7)) {
        int a, b, c, d;
        (void)a;
        (void)b;
        (void)c;
        (void)d; // compiler warnings...
        cpuid(7, a, b, c, d);
        return b & 0x00000020;
    }
    return 0;
}
#endif

#ifdef __e2k__
inline int
CPU_haveAVX512F(void)
{
    return 0;
}
#else
static int CPU_haveAVX512F(void)
{
    if (CPU_OSSavesZMM && (CPU_CPUIDMaxFunction >= 7)) {
        int a, b, c, d;
        (void)a;
        (void)b;
        (void)c;
        (void)d; // compiler warnings...
        cpuid(7, a, b, c, d);
        return b & 0x00010000;
    }
    return 0;
}
#endif

static int SDL_NumLogicalCPUCores = 0;

int SDL_GetNumLogicalCPUCores(void)
{
    if (!SDL_NumLogicalCPUCores) {
#if defined(HAVE_SYSCONF) && defined(_SC_NPROCESSORS_ONLN)
        if (SDL_NumLogicalCPUCores <= 0) {
            SDL_NumLogicalCPUCores = (int)sysconf(_SC_NPROCESSORS_ONLN);
        }
#endif
#ifdef HAVE_SYSCTLBYNAME
        if (SDL_NumLogicalCPUCores <= 0) {
            size_t size = sizeof(SDL_NumLogicalCPUCores);
            sysctlbyname("hw.ncpu", &SDL_NumLogicalCPUCores, &size, NULL, 0);
        }
#endif
#if defined(SDL_PLATFORM_WINDOWS)
        if (SDL_NumLogicalCPUCores <= 0) {
            SYSTEM_INFO info;
            GetSystemInfo(&info);
            SDL_NumLogicalCPUCores = info.dwNumberOfProcessors;
        }
#endif
#ifdef SDL_PLATFORM_3DS
        if (SDL_NumLogicalCPUCores <= 0) {
            bool isNew3DS = false;
            APT_CheckNew3DS(&isNew3DS);
            // 1 core is always dedicated to the OS
            // Meaning that the New3DS has 3 available core, and the Old3DS only one.
            SDL_NumLogicalCPUCores = isNew3DS ? 4 : 2;
        }
#endif
        // There has to be at least 1, right? :)
        if (SDL_NumLogicalCPUCores <= 0) {
            SDL_NumLogicalCPUCores = 1;
        }
    }
    return SDL_NumLogicalCPUCores;
}

#ifdef __e2k__
inline const char *
SDL_GetCPUType(void)
{
    static char SDL_CPUType[13];

    SDL_strlcpy(SDL_CPUType, "E2K MACHINE", sizeof(SDL_CPUType));

    return SDL_CPUType;
}
#else
// Oh, such a sweet sweet trick, just not very useful. :)
static const char *SDL_GetCPUType(void)
{
    static char SDL_CPUType[13];

    if (!SDL_CPUType[0]) {
        int i = 0;

        CPU_calcCPUIDFeatures();
        if (CPU_CPUIDMaxFunction > 0) { // do we have CPUID at all?
            int a, b, c, d;
            cpuid(0x00000000, a, b, c, d);
            (void)a;
            SDL_CPUType[i++] = (char)(b & 0xff);
            b >>= 8;
            SDL_CPUType[i++] = (char)(b & 0xff);
            b >>= 8;
            SDL_CPUType[i++] = (char)(b & 0xff);
            b >>= 8;
            SDL_CPUType[i++] = (char)(b & 0xff);

            SDL_CPUType[i++] = (char)(d & 0xff);
            d >>= 8;
            SDL_CPUType[i++] = (char)(d & 0xff);
            d >>= 8;
            SDL_CPUType[i++] = (char)(d & 0xff);
            d >>= 8;
            SDL_CPUType[i++] = (char)(d & 0xff);

            SDL_CPUType[i++] = (char)(c & 0xff);
            c >>= 8;
            SDL_CPUType[i++] = (char)(c & 0xff);
            c >>= 8;
            SDL_CPUType[i++] = (char)(c & 0xff);
            c >>= 8;
            SDL_CPUType[i++] = (char)(c & 0xff);
        }
        if (!SDL_CPUType[0]) {
            SDL_strlcpy(SDL_CPUType, "Unknown", sizeof(SDL_CPUType));
        }
    }
    return SDL_CPUType;
}
#endif

#if 0
!!! FIXME: Not used at the moment. */
#ifdef __e2k__
inline const char *
SDL_GetCPUName(void)
{
    static char SDL_CPUName[48];

    SDL_strlcpy(SDL_CPUName, __builtin_cpu_name(), sizeof(SDL_CPUName));

    return SDL_CPUName;
}
#else
static const char *SDL_GetCPUName(void)
{
    static char SDL_CPUName[48];

    if (!SDL_CPUName[0]) {
        int i = 0;
        int a, b, c, d;

        CPU_calcCPUIDFeatures();
        if (CPU_CPUIDMaxFunction > 0) { // do we have CPUID at all?
            cpuid(0x80000000, a, b, c, d);
            if (a >= 0x80000004) {
                cpuid(0x80000002, a, b, c, d);
                SDL_CPUName[i++] = (char)(a & 0xff);
                a >>= 8;
                SDL_CPUName[i++] = (char)(a & 0xff);
                a >>= 8;
                SDL_CPUName[i++] = (char)(a & 0xff);
                a >>= 8;
                SDL_CPUName[i++] = (char)(a & 0xff);
                a >>= 8;
                SDL_CPUName[i++] = (char)(b & 0xff);
                b >>= 8;
                SDL_CPUName[i++] = (char)(b & 0xff);
                b >>= 8;
                SDL_CPUName[i++] = (char)(b & 0xff);
                b >>= 8;
                SDL_CPUName[i++] = (char)(b & 0xff);
                b >>= 8;
                SDL_CPUName[i++] = (char)(c & 0xff);
                c >>= 8;
                SDL_CPUName[i++] = (char)(c & 0xff);
                c >>= 8;
                SDL_CPUName[i++] = (char)(c & 0xff);
                c >>= 8;
                SDL_CPUName[i++] = (char)(c & 0xff);
                c >>= 8;
                SDL_CPUName[i++] = (char)(d & 0xff);
                d >>= 8;
                SDL_CPUName[i++] = (char)(d & 0xff);
                d >>= 8;
                SDL_CPUName[i++] = (char)(d & 0xff);
                d >>= 8;
                SDL_CPUName[i++] = (char)(d & 0xff);
                d >>= 8;
                cpuid(0x80000003, a, b, c, d);
                SDL_CPUName[i++] = (char)(a & 0xff);
                a >>= 8;
                SDL_CPUName[i++] = (char)(a & 0xff);
                a >>= 8;
                SDL_CPUName[i++] = (char)(a & 0xff);
                a >>= 8;
                SDL_CPUName[i++] = (char)(a & 0xff);
                a >>= 8;
                SDL_CPUName[i++] = (char)(b & 0xff);
                b >>= 8;
                SDL_CPUName[i++] = (char)(b & 0xff);
                b >>= 8;
                SDL_CPUName[i++] = (char)(b & 0xff);
                b >>= 8;
                SDL_CPUName[i++] = (char)(b & 0xff);
                b >>= 8;
                SDL_CPUName[i++] = (char)(c & 0xff);
                c >>= 8;
                SDL_CPUName[i++] = (char)(c & 0xff);
                c >>= 8;
                SDL_CPUName[i++] = (char)(c & 0xff);
                c >>= 8;
                SDL_CPUName[i++] = (char)(c & 0xff);
                c >>= 8;
                SDL_CPUName[i++] = (char)(d & 0xff);
                d >>= 8;
                SDL_CPUName[i++] = (char)(d & 0xff);
                d >>= 8;
                SDL_CPUName[i++] = (char)(d & 0xff);
                d >>= 8;
                SDL_CPUName[i++] = (char)(d & 0xff);
                d >>= 8;
                cpuid(0x80000004, a, b, c, d);
                SDL_CPUName[i++] = (char)(a & 0xff);
                a >>= 8;
                SDL_CPUName[i++] = (char)(a & 0xff);
                a >>= 8;
                SDL_CPUName[i++] = (char)(a & 0xff);
                a >>= 8;
                SDL_CPUName[i++] = (char)(a & 0xff);
                a >>= 8;
                SDL_CPUName[i++] = (char)(b & 0xff);
                b >>= 8;
                SDL_CPUName[i++] = (char)(b & 0xff);
                b >>= 8;
                SDL_CPUName[i++] = (char)(b & 0xff);
                b >>= 8;
                SDL_CPUName[i++] = (char)(b & 0xff);
                b >>= 8;
                SDL_CPUName[i++] = (char)(c & 0xff);
                c >>= 8;
                SDL_CPUName[i++] = (char)(c & 0xff);
                c >>= 8;
                SDL_CPUName[i++] = (char)(c & 0xff);
                c >>= 8;
                SDL_CPUName[i++] = (char)(c & 0xff);
                c >>= 8;
                SDL_CPUName[i++] = (char)(d & 0xff);
                d >>= 8;
                SDL_CPUName[i++] = (char)(d & 0xff);
                d >>= 8;
                SDL_CPUName[i++] = (char)(d & 0xff);
                d >>= 8;
                SDL_CPUName[i++] = (char)(d & 0xff);
                d >>= 8;
            }
        }
        if (!SDL_CPUName[0]) {
            SDL_strlcpy(SDL_CPUName, "Unknown", sizeof(SDL_CPUName));
        }
    }
    return SDL_CPUName;
}
#endif
#endif

int SDL_GetCPUCacheLineSize(void)
{
    const char *cpuType = SDL_GetCPUType();
    int cacheline_size = SDL_CACHELINE_SIZE; // initial guess
    int a, b, c, d;
    (void)a;
    (void)b;
    (void)c;
    (void)d;
    if (SDL_strcmp(cpuType, "GenuineIntel") == 0 || SDL_strcmp(cpuType, "CentaurHauls") == 0 || SDL_strcmp(cpuType, "  Shanghai  ") == 0) {
        cpuid(0x00000001, a, b, c, d);
        cacheline_size = ((b >> 8) & 0xff) * 8;
    } else if (SDL_strcmp(cpuType, "AuthenticAMD") == 0 || SDL_strcmp(cpuType, "HygonGenuine") == 0) {
        cpuid(0x80000005, a, b, c, d);
        cacheline_size = c & 0xff;
    } else {
#if defined(HAVE_SYSCONF) && defined(_SC_LEVEL1_DCACHE_LINESIZE)
        if ((cacheline_size = (int)sysconf(_SC_LEVEL1_DCACHE_LINESIZE)) > 0) {
            return cacheline_size;
        } else {
            cacheline_size = SDL_CACHELINE_SIZE;
        }
#endif
#if defined(SDL_PLATFORM_LINUX)
        {
            FILE *f = fopen("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size", "r");
            if (f) {
                int size;
                if (fscanf(f, "%d", &size) == 1) {
                    cacheline_size = size;
                }
                fclose(f);
            }
        }
#elif defined(__FREEBSD__) && defined(CACHE_LINE_SIZE)
        cacheline_size = CACHE_LINE_SIZE;
#endif
    }
    return cacheline_size;
}

#define SDL_CPUFEATURES_RESET_VALUE 0xFFFFFFFF

static Uint32 SDL_CPUFeatures = SDL_CPUFEATURES_RESET_VALUE;
static Uint32 SDL_SIMDAlignment = 0xFFFFFFFF;

static bool ref_string_equals(const char *ref, const char *test, const char *end_test) {
    size_t len_test = end_test - test;
    return SDL_strncmp(ref, test, len_test) == 0 && ref[len_test] == '\0' && (test[len_test] == '\0' || test[len_test] == ',');
}

static Uint32 SDLCALL SDL_CPUFeatureMaskFromHint(void)
{
    Uint32 result_mask = SDL_CPUFEATURES_RESET_VALUE;

    const char *hint = SDL_GetHint(SDL_HINT_CPU_FEATURE_MASK);

    if (hint) {
        for (const char *spot = hint, *next; *spot; spot = next) {
            const char *end = SDL_strchr(spot, ',');
            Uint32 spot_mask;
            bool add_spot_mask = true;
            if (end) {
                next = end + 1;
            } else {
                size_t len = SDL_strlen(spot);
                end = spot + len;
                next = end;
            }
            if (spot[0] == '+') {
                add_spot_mask = true;
                spot += 1;
            } else if (spot[0] == '-') {
                add_spot_mask = false;
                spot += 1;
            }
            if (ref_string_equals("all", spot, end)) {
                spot_mask = SDL_CPUFEATURES_RESET_VALUE;
            } else if (ref_string_equals("altivec", spot, end)) {
                spot_mask= CPU_HAS_ALTIVEC;
            } else if (ref_string_equals("mmx", spot, end)) {
                spot_mask = CPU_HAS_MMX;
            } else if (ref_string_equals("sse", spot, end)) {
                spot_mask = CPU_HAS_SSE;
            } else if (ref_string_equals("sse2", spot, end)) {
                spot_mask = CPU_HAS_SSE2;
            } else if (ref_string_equals("sse3", spot, end)) {
                spot_mask = CPU_HAS_SSE3;
            } else if (ref_string_equals("sse41", spot, end)) {
                spot_mask = CPU_HAS_SSE41;
            } else if (ref_string_equals("sse42", spot, end)) {
                spot_mask = CPU_HAS_SSE42;
            } else if (ref_string_equals("avx", spot, end)) {
                spot_mask = CPU_HAS_AVX;
            } else if (ref_string_equals("avx2", spot, end)) {
                spot_mask = CPU_HAS_AVX2;
            } else if (ref_string_equals("avx512f", spot, end)) {
                spot_mask = CPU_HAS_AVX512F;
            } else if (ref_string_equals("arm-simd", spot, end)) {
                spot_mask = CPU_HAS_ARM_SIMD;
            } else if (ref_string_equals("neon", spot, end)) {
                spot_mask = CPU_HAS_NEON;
            } else if (ref_string_equals("lsx", spot, end)) {
                spot_mask = CPU_HAS_LSX;
            } else if (ref_string_equals("lasx", spot, end)) {
                spot_mask = CPU_HAS_LASX;
            } else {
                // Ignore unknown/incorrect cpu feature(s)
                continue;
            }
            if (add_spot_mask) {
                result_mask |= spot_mask;
            } else {
                result_mask &= ~spot_mask;
            }
        }
    }
    return result_mask;
}

static Uint32 SDL_GetCPUFeatures(void)
{
    if (SDL_CPUFeatures == SDL_CPUFEATURES_RESET_VALUE) {
        CPU_calcCPUIDFeatures();
        SDL_CPUFeatures = 0;
        SDL_SIMDAlignment = sizeof(void *); // a good safe base value
        if (CPU_haveAltiVec()) {
            SDL_CPUFeatures |= CPU_HAS_ALTIVEC;
            SDL_SIMDAlignment = SDL_max(SDL_SIMDAlignment, 16);
        }
        if (CPU_haveMMX()) {
            SDL_CPUFeatures |= CPU_HAS_MMX;
            SDL_SIMDAlignment = SDL_max(SDL_SIMDAlignment, 8);
        }
        if (CPU_haveSSE()) {
            SDL_CPUFeatures |= CPU_HAS_SSE;
            SDL_SIMDAlignment = SDL_max(SDL_SIMDAlignment, 16);
        }
        if (CPU_haveSSE2()) {
            SDL_CPUFeatures |= CPU_HAS_SSE2;
            SDL_SIMDAlignment = SDL_max(SDL_SIMDAlignment, 16);
        }
        if (CPU_haveSSE3()) {
            SDL_CPUFeatures |= CPU_HAS_SSE3;
            SDL_SIMDAlignment = SDL_max(SDL_SIMDAlignment, 16);
        }
        if (CPU_haveSSE41()) {
            SDL_CPUFeatures |= CPU_HAS_SSE41;
            SDL_SIMDAlignment = SDL_max(SDL_SIMDAlignment, 16);
        }
        if (CPU_haveSSE42()) {
            SDL_CPUFeatures |= CPU_HAS_SSE42;
            SDL_SIMDAlignment = SDL_max(SDL_SIMDAlignment, 16);
        }
        if (CPU_haveAVX()) {
            SDL_CPUFeatures |= CPU_HAS_AVX;
            SDL_SIMDAlignment = SDL_max(SDL_SIMDAlignment, 32);
        }
        if (CPU_haveAVX2()) {
            SDL_CPUFeatures |= CPU_HAS_AVX2;
            SDL_SIMDAlignment = SDL_max(SDL_SIMDAlignment, 32);
        }
        if (CPU_haveAVX512F()) {
            SDL_CPUFeatures |= CPU_HAS_AVX512F;
            SDL_SIMDAlignment = SDL_max(SDL_SIMDAlignment, 64);
        }
        if (CPU_haveARMSIMD()) {
            SDL_CPUFeatures |= CPU_HAS_ARM_SIMD;
            SDL_SIMDAlignment = SDL_max(SDL_SIMDAlignment, 16);
        }
        if (CPU_haveNEON()) {
            SDL_CPUFeatures |= CPU_HAS_NEON;
            SDL_SIMDAlignment = SDL_max(SDL_SIMDAlignment, 16);
        }
        if (CPU_haveLSX()) {
            SDL_CPUFeatures |= CPU_HAS_LSX;
            SDL_SIMDAlignment = SDL_max(SDL_SIMDAlignment, 16);
        }
        if (CPU_haveLASX()) {
            SDL_CPUFeatures |= CPU_HAS_LASX;
            SDL_SIMDAlignment = SDL_max(SDL_SIMDAlignment, 32);
        }
        SDL_CPUFeatures &= SDL_CPUFeatureMaskFromHint();
    }
    return SDL_CPUFeatures;
}

void SDL_QuitCPUInfo(void) {
    SDL_CPUFeatures = SDL_CPUFEATURES_RESET_VALUE;
}

#define CPU_FEATURE_AVAILABLE(f) ((SDL_GetCPUFeatures() & (f)) ? true : false)

bool SDL_HasAltiVec(void)
{
    return CPU_FEATURE_AVAILABLE(CPU_HAS_ALTIVEC);
}

bool SDL_HasMMX(void)
{
    return CPU_FEATURE_AVAILABLE(CPU_HAS_MMX);
}

bool SDL_HasSSE(void)
{
    return CPU_FEATURE_AVAILABLE(CPU_HAS_SSE);
}

bool SDL_HasSSE2(void)
{
    return CPU_FEATURE_AVAILABLE(CPU_HAS_SSE2);
}

bool SDL_HasSSE3(void)
{
    return CPU_FEATURE_AVAILABLE(CPU_HAS_SSE3);
}

bool SDL_HasSSE41(void)
{
    return CPU_FEATURE_AVAILABLE(CPU_HAS_SSE41);
}

bool SDL_HasSSE42(void)
{
    return CPU_FEATURE_AVAILABLE(CPU_HAS_SSE42);
}

bool SDL_HasAVX(void)
{
    return CPU_FEATURE_AVAILABLE(CPU_HAS_AVX);
}

bool SDL_HasAVX2(void)
{
    return CPU_FEATURE_AVAILABLE(CPU_HAS_AVX2);
}

bool SDL_HasAVX512F(void)
{
    return CPU_FEATURE_AVAILABLE(CPU_HAS_AVX512F);
}

bool SDL_HasARMSIMD(void)
{
    return CPU_FEATURE_AVAILABLE(CPU_HAS_ARM_SIMD);
}

bool SDL_HasNEON(void)
{
    return CPU_FEATURE_AVAILABLE(CPU_HAS_NEON);
}

bool SDL_HasLSX(void)
{
    return CPU_FEATURE_AVAILABLE(CPU_HAS_LSX);
}

bool SDL_HasLASX(void)
{
    return CPU_FEATURE_AVAILABLE(CPU_HAS_LASX);
}

static int SDL_SystemRAM = 0;

int SDL_GetSystemRAM(void)
{
    if (!SDL_SystemRAM) {
#if defined(HAVE_SYSCONF) && defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
        if (SDL_SystemRAM <= 0) {
            SDL_SystemRAM = (int)((Sint64)sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGESIZE) / (1024 * 1024));
        }
#endif
#ifdef HAVE_SYSCTLBYNAME
        if (SDL_SystemRAM <= 0) {
#ifdef HW_PHYSMEM64
            // (64-bit): NetBSD since 2003, OpenBSD
            int mib[2] = { CTL_HW, HW_PHYSMEM64 };
#elif defined(HW_REALMEM)
            // (64-bit): FreeBSD since 2005, DragonFly
            int mib[2] = { CTL_HW, HW_REALMEM };
#elif defined(HW_MEMSIZE)
            // (64-bit): Darwin
            int mib[2] = { CTL_HW, HW_MEMSIZE };
#else
            // (32-bit): very old BSD, might only report up to 2 GiB
            int mib[2] = { CTL_HW, HW_PHYSMEM };
#endif // HW_PHYSMEM64
            Uint64 memsize = 0;
            size_t len = sizeof(memsize);

            if (sysctl(mib, 2, &memsize, &len, NULL, 0) == 0) {
                SDL_SystemRAM = (int)(memsize / (1024 * 1024));
            }
        }
#endif
#if defined(SDL_PLATFORM_WINDOWS)
        if (SDL_SystemRAM <= 0) {
            MEMORYSTATUSEX stat;
            stat.dwLength = sizeof(stat);
            if (GlobalMemoryStatusEx(&stat)) {
                SDL_SystemRAM = (int)(stat.ullTotalPhys / (1024 * 1024));
            }
        }
#endif
#ifdef SDL_PLATFORM_RISCOS
        if (SDL_SystemRAM <= 0) {
            _kernel_swi_regs regs;
            regs.r[0] = 0x108;
            if (_kernel_swi(OS_Memory, &regs, &regs) == NULL) {
                SDL_SystemRAM = (int)(regs.r[1] * regs.r[2] / (1024 * 1024));
            }
        }
#endif
#ifdef SDL_PLATFORM_3DS
        if (SDL_SystemRAM <= 0) {
            // The New3DS has 255MiB, the Old3DS 127MiB
            SDL_SystemRAM = (int)(osGetMemRegionSize(MEMREGION_ALL) / (1024 * 1024));
        }
#endif
#ifdef SDL_PLATFORM_VITA
        if (SDL_SystemRAM <= 0) {
            /* Vita has 512MiB on SoC, that's split into 256MiB(+109MiB in extended memory mode) for app
               +26MiB of physically continuous memory, +112MiB of CDRAM(VRAM) + system reserved memory. */
            SDL_SystemRAM = 536870912;
        }
#endif
#ifdef SDL_PLATFORM_PS2
        if (SDL_SystemRAM <= 0) {
            // PlayStation 2 has 32MiB however there are some special models with 64 and 128
            SDL_SystemRAM = GetMemorySize();
        }
#endif
#ifdef SDL_PLATFORM_HAIKU
        if (SDL_SystemRAM <= 0) {
            system_info info;
            if (get_system_info(&info) == B_OK) {
                /* To have an accurate amount, we also take in account the inaccessible pages (aka ignored)
                  which is a bit handier compared to the legacy system's api (i.e. used_pages).*/
                SDL_SystemRAM = (int)SDL_round((info.max_pages + info.ignored_pages > 0 ? info.ignored_pages : 0) * B_PAGE_SIZE / 1048576.0);
            }
        }
#endif
    }
    return SDL_SystemRAM;
}

size_t SDL_GetSIMDAlignment(void)
{
    if (SDL_SIMDAlignment == 0xFFFFFFFF) {
        SDL_GetCPUFeatures(); // make sure this has been calculated
    }
    SDL_assert(SDL_SIMDAlignment != 0);
    return SDL_SIMDAlignment;
}
