#ifndef _C4_CPU_HPP_
#define _C4_CPU_HPP_

/** @file cpu.hpp Provides processor information macros
 * @ingroup basic_headers */

// see also https://sourceforge.net/p/predef/wiki/Architectures/
// see also https://sourceforge.net/p/predef/wiki/Endianness/
// see also https://github.com/googlesamples/android-ndk/blob/android-mk/hello-jni/jni/hello-jni.c
// see http://code.qt.io/cgit/qt/qtbase.git/tree/src/corelib/global/qprocessordetection.h

#ifdef __ORDER_LITTLE_ENDIAN__
#   define _C4EL __ORDER_LITTLE_ENDIAN__
#else
#   define _C4EL 1234
#endif

#ifdef __ORDER_BIG_ENDIAN__
#   define _C4EB __ORDER_BIG_ENDIAN__
#else
#   define _C4EB 4321
#endif

// mixed byte order (eg, PowerPC or ia64)
#define _C4EM 1111

#if defined(__x86_64) || defined(__x86_64__) || defined(__amd64) || defined(_M_X64)
#    define C4_CPU_X86_64
#    define C4_WORDSIZE 8
#    define C4_BYTE_ORDER _C4EL

#elif defined(__i386) || defined(__i386__) || defined(_M_IX86)
#    define C4_CPU_X86
#    define C4_WORDSIZE 4
#    define C4_BYTE_ORDER _C4EL

#elif defined(__arm__) || defined(_M_ARM) \
    || defined(__TARGET_ARCH_ARM) || defined(__aarch64__) || defined(_M_ARM64)
#   if defined(__aarch64__) || defined(_M_ARM64)
#       define C4_CPU_ARM64
#       define C4_CPU_ARMV8
#       define C4_WORDSIZE 8
#   else
#       define C4_CPU_ARM
#       define C4_WORDSIZE 4
#       if defined(__ARM_ARCH_8__) || defined(__ARM_ARCH_8A__)  \
        || (defined(__ARCH_ARM) && __ARCH_ARM >= 8) \
        || (defined(__TARGET_ARCH_ARM) && __TARGET_ARCH_ARM >= 8)
#           define C4_CPU_ARMV8
#       elif defined(__ARM_ARCH_7__) || defined(_ARM_ARCH_7)    \
        || defined(__ARM_ARCH_7A__) || defined(__ARM_ARCH_7R__) \
        || defined(__ARM_ARCH_7M__) || defined(__ARM_ARCH_7S__) \
        || defined(__ARM_ARCH_7EM__) \
        || (defined(__TARGET_ARCH_ARM) && __TARGET_ARCH_ARM >= 7) \
        || (defined(_M_ARM) && _M_ARM >= 7)
#           define C4_CPU_ARMV7
#       elif defined(__ARM_ARCH_6__) || defined(__ARM_ARCH_6J__) \
        || defined(__ARM_ARCH_6T2__) || defined(__ARM_ARCH_6Z__) \
        || defined(__ARM_ARCH_6K__)  || defined(__ARM_ARCH_6ZK__) \
        || defined(__ARM_ARCH_6M__) || defined(__ARM_ARCH_6KZ__) \
        || (defined(__TARGET_ARCH_ARM) && __TARGET_ARCH_ARM >= 6)
#           define C4_CPU_ARMV6
#       elif defined(__ARM_ARCH_5TEJ__) \
        || defined(__ARM_ARCH_5TE__) \
        || (defined(__TARGET_ARCH_ARM) && __TARGET_ARCH_ARM >= 5)
#           define C4_CPU_ARMV5
#       elif defined(__ARM_ARCH_4T__) \
        || (defined(__TARGET_ARCH_ARM) && __TARGET_ARCH_ARM >= 4)
#           define C4_CPU_ARMV4
#       else
#           error "unknown CPU architecture: ARM"
#       endif
#   endif
#   if defined(__ARMEL__) || defined(__LITTLE_ENDIAN__) || defined(__AARCH64EL__) \
       || (defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)) \
       || defined(_MSC_VER) // winarm64 does not provide any of the above macros,
                            // but advises little-endianess:
                            // https://docs.microsoft.com/en-us/cpp/build/overview-of-arm-abi-conventions?view=msvc-170
                            // So if it is visual studio compiling, we'll assume little endian.
#       define C4_BYTE_ORDER _C4EL
#   elif defined(__ARMEB__) || defined(__BIG_ENDIAN__) || defined(__AARCH64EB__) \
       || (defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__))
#       define C4_BYTE_ORDER _C4EB
#   elif defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_PDP_ENDIAN__)
#       define C4_BYTE_ORDER _C4EM
#   else
#       error "unknown endianness"
#   endif

#elif defined(__ia64) || defined(__ia64__) || defined(_M_IA64)
#   define C4_CPU_IA64
#   define C4_WORDSIZE 8
#   define C4_BYTE_ORDER _C4EM
   // itanium is bi-endian - check byte order below

#elif defined(__ppc__) || defined(__ppc) || defined(__powerpc__)       \
    || defined(_ARCH_COM) || defined(_ARCH_PWR) || defined(_ARCH_PPC)  \
    || defined(_M_MPPC) || defined(_M_PPC)
#   if defined(__ppc64__) || defined(__powerpc64__) || defined(__64BIT__)
#       define C4_CPU_PPC64
#       define C4_WORDSIZE 8
#   else
#       define C4_CPU_PPC
#       define C4_WORDSIZE 4
#   endif
#   define C4_BYTE_ORDER _C4EM
   // ppc is bi-endian - check byte order below

#elif defined(__s390x__) || defined(__zarch__) || defined(__SYSC_ZARCH_)
#   define C4_CPU_S390_X
#   define C4_WORDSIZE 8
#   define C4_BYTE_ORDER _C4EB

#elif defined(__xtensa__) || defined(__XTENSA__)
#   define C4_CPU_XTENSA
#   define C4_WORDSIZE 4
// not sure about this...
#   if defined(__XTENSA_EL__) || defined(__xtensa_el__)
#       define C4_BYTE_ORDER _C4EL
#   else
#       define C4_BYTE_ORDER _C4EB
#   endif

#elif defined(__riscv)
#   if __riscv_xlen == 64
#       define C4_CPU_RISCV64
#       define C4_WORDSIZE 8
#   else
#       define C4_CPU_RISCV32
#       define C4_WORDSIZE 4
#   endif
#   define C4_BYTE_ORDER _C4EL

#elif defined(__EMSCRIPTEN__)
#   define C4_BYTE_ORDER _C4EL
#   define C4_WORDSIZE 4

#elif defined(SWIG)
#   error "please define CPU architecture macros when compiling with swig"

#else
#   error "unknown CPU architecture"
#endif

#define C4_LITTLE_ENDIAN (C4_BYTE_ORDER == _C4EL)
#define C4_BIG_ENDIAN (C4_BYTE_ORDER == _C4EB)
#define C4_MIXED_ENDIAN (C4_BYTE_ORDER == _C4EM)

#endif /* _C4_CPU_HPP_ */
