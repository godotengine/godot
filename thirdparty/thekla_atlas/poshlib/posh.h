/**
@file posh.h
@author Brian Hook
@version 1.3.001

Header file for POSH, the Portable Open Source Harness project.

NOTE: Unlike most header files, this one is designed to be included
multiple times, which is why it does not have the @#ifndef/@#define
preamble.

POSH relies on environment specified preprocessor symbols in order
to infer as much as possible about the target OS/architecture and
the host compiler capabilities.

NOTE: POSH is simple and focused. It attempts to provide basic
functionality and information, but it does NOT attempt to emulate
missing functionality.  I am also not willing to make POSH dirty
and hackish to support truly ancient and/or outmoded and/or bizarre
technologies such as non-ANSI compilers, systems with non-IEEE
floating point formats, segmented 16-bit operating systems, etc.

Please refer to the accompanying HTML documentation or visit
http://www.poshlib.org for more information on how to use POSH.

LICENSE:

Copyright (c) 2004, Brian Hook
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.

    * The names of this package'ss contributors contributors may not
      be used to endorse or promote products derived from this
      software without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REVISION:

I've been lax about revision histories, so this starts at, um, 1.3.001.
Sorry for any inconveniences.

1.3.001 - 2/23/2006 - Incorporated fix for bug reported by Bill Cary,
                      where I was not detecting Visual Studio
                      compilation on x86-64 systems.  Added check for
                      _M_X64 which should fix that.

*/
/*
I have yet to find an authoritative reference on preprocessor
symbols, but so far this is what I've gleaned:

GNU GCC/G++:
   - __GNUC__: GNU C version
   - __GNUG__: GNU C++ compiler
   - __sun__ : on Sun platforms
   - __svr4__: on Solaris and other SysV R4 platforms
   - __mips__: on MIPS processor platforms
   - __sparc_v9__: on Sparc 64-bit CPUs
   - __sparcv9: 64-bit Solaris
   - __MIPSEL__: mips processor, compiled for little endian
   - __MIPSEB__: mips processor, compiled for big endian
   - _R5900: MIPS/Sony/Toshiba R5900 (PS2)
   - mc68000: 68K
   - m68000: 68K
   - m68k: 68K
   - __palmos__: PalmOS

Intel C/C++ Compiler:
   - __ECC      : compiler version, IA64 only
   - __EDG__
   - __ELF__
   - __GXX_ABI_VERSION
   - __i386     : IA-32 only
   - __i386__   : IA-32 only
   - i386       : IA-32 only
   - __ia64     : IA-64 only
   - __ia64__   : IA-64 only
   - ia64       : IA-64 only
   - __ICC      : IA-32 only
   - __INTEL_COMPILER : IA-32 or IA-64, newer versions only

Apple's C/C++ Compiler for OS X:
   - __APPLE_CC__
   - __APPLE__
   - __BIG_ENDIAN__
   - __APPLE__
   - __ppc__
   - __MACH__

DJGPP:
   - __MSDOS__
   - __unix__
   - __unix
   - __GNUC__
   - __GO32
   - DJGPP
   - __i386, __i386, i386

Cray's C compiler:
   - _ADDR64: if 64-bit pointers
   - _UNICOS: 
   - __unix:

SGI's CC compiler predefines the following (and more) with -ansi:
   - __sgi
   - __unix
   - __host_mips
   - _SYSTYPE_SVR4
   - __mips
   - _MIPSEB
   - anyone know if there is a predefined symbol for the compiler?!

MinGW:
   - as GnuC but also defines _WIN32, __WIN32, WIN32, _X86_, __i386, __i386__, and several others
   - __MINGW32__

Cygwin:
   - as Gnu C, but also
   - __unix__
   - __CYGWIN32__

Microsoft Visual Studio predefines the following:
   - _MSC_VER
   - _WIN32: on Win32
   - _M_IX6 (on x86 systems)
   - _M_X64: on x86-64 systems
   - _M_ALPHA (on DEC AXP systems)
   - _SH3: WinCE, Hitachi SH-3
   - _MIPS: WinCE, MIPS
   - _ARM: WinCE, ARM

Sun's C Compiler:
   - sun and _sun
   - unix and _unix
   - sparc and _sparc (SPARC systems only)
   - i386 and _i386 (x86 systems only)
   - __SVR4 (Solaris only)
   - __sparcv9: 64-bit solaris
   - __SUNPRO_C
   - _LP64: defined in 64-bit LP64 mode, but only if <sys/types.h> is included

Borland C/C++ predefines the following:
   - __BORLANDC__:

DEC/Compaq C/C++ on Alpha:
   - __alpha
   - __arch64__
   - __unix__ (on Tru64 Unix)
   - __osf__
   - __DECC
   - __DECCXX (C++ compilation)
   - __DECC_VER
   - __DECCXX_VER

IBM's AIX compiler:
   - __64BIT__ if 64-bit mode
   - _AIX
   - __IBMC__: C compiler version
   - __IBMCPP__: C++ compiler version
   - _LONG_LONG: compiler allows long long

Watcom:
   - __WATCOMC__
   - __DOS__ : if targeting DOS
   - __386__ : if 32-bit support
   - __WIN32__ : if targetin 32-bit Windows

HP-UX C/C++ Compiler:
   - __hpux
   - __unix
   - __hppa (on PA-RISC)
   - __LP64__: if compiled in 64-bit mode

Metrowerks:
   - __MWERKS__
   - __powerpc__
   - _powerc
   - __MC68K__
   - macintosh when compiling for MacOS
   - __INTEL__ for x86 targets
   - __POWERPC__

*/

/*
** ----------------------------------------------------------------------------
** Include <limits.h> optionally
** ----------------------------------------------------------------------------
*/
#ifdef POSH_USE_LIMITS_H
#  include <limits.h>
#endif

/*
** ----------------------------------------------------------------------------
** Determine compilation environment
** ----------------------------------------------------------------------------
*/
#if defined __ECC || defined __ICC || defined __INTEL_COMPILER
#  define POSH_COMPILER_STRING "Intel C/C++"
#  define POSH_COMPILER_INTEL 1
#endif

#if ( defined __host_mips || defined __sgi ) && !defined __GNUC__
#  define POSH_COMPILER_STRING    "MIPSpro C/C++"
#  define POSH_COMPILER_MIPSPRO 1 
#endif

#if defined __hpux && !defined __GNUC__
#  define POSH_COMPILER_STRING "HP-UX CC"
#  define POSH_COMPILER_HPCC 1 
#endif

#if defined __GNUC__ && !defined __clang__
#  define POSH_COMPILER_STRING "Gnu GCC"
#  define POSH_COMPILER_GCC 1
#endif

#if defined __clang__
#  define POSH_COMPILER_STRING "Clang"
#  define POSH_COMPILER_CLANG 1
#endif

#if defined __APPLE_CC__
   /* we don't define the compiler string here, let it be GNU */
#  define POSH_COMPILER_APPLECC 1
#endif

#if defined __IBMC__ || defined __IBMCPP__
#  define POSH_COMPILER_STRING "IBM C/C++"
#  define POSH_COMPILER_IBM 1
#endif

#if defined _MSC_VER
#  define POSH_COMPILER_STRING "Microsoft Visual C++"
#  define POSH_COMPILER_MSVC 1
#endif

#if defined __SUNPRO_C
#  define POSH_COMPILER_STRING "Sun Pro" 
#  define POSH_COMPILER_SUN 1
#endif

#if defined __BORLANDC__
#  define POSH_COMPILER_STRING "Borland C/C++"
#  define POSH_COMPILER_BORLAND 1
#endif

#if defined __MWERKS__
#  define POSH_COMPILER_STRING     "MetroWerks CodeWarrior"
#  define POSH_COMPILER_METROWERKS 1
#endif

#if defined __DECC || defined __DECCXX
#  define POSH_COMPILER_STRING "Compaq/DEC C/C++"
#  define POSH_COMPILER_DEC 1
#endif

#if defined __WATCOMC__
#  define POSH_COMPILER_STRING "Watcom C/C++"
#  define POSH_COMPILER_WATCOM 1
#endif

#if !defined POSH_COMPILER_STRING
#  define POSH_COMPILER_STRING "Unknown compiler"
#endif

/*
** ----------------------------------------------------------------------------
** Determine target operating system
** ----------------------------------------------------------------------------
*/
#if defined linux || defined __linux__
#  define POSH_OS_LINUX 1 
#  define POSH_OS_STRING "Linux"
#endif

#if defined __FreeBSD__
#  define POSH_OS_FREEBSD 1 
#  define POSH_OS_STRING "FreeBSD"
#endif

#if defined __CYGWIN32__
#  define POSH_OS_CYGWIN32 1
#  define POSH_OS_STRING "Cygwin"
#endif

#if defined GEKKO
#  define POSH_OS_GAMECUBE
#  define __powerpc__
#  define POSH_OS_STRING "GameCube"
#endif

#if defined __MINGW32__
#  define POSH_OS_MINGW 1
#  define POSH_OS_STRING "MinGW"
#endif

#if defined GO32 && defined DJGPP && defined __MSDOS__
#  define POSH_OS_GO32 1
#  define POSH_OS_STRING "GO32/MS-DOS"
#endif

/* NOTE: make sure you use /bt=DOS if compiling for 32-bit DOS,
   otherwise Watcom assumes host=target */
#if defined __WATCOMC__  && defined __386__ && defined __DOS__
#  define POSH_OS_DOS32 1
#  define POSH_OS_STRING "DOS/32-bit"
#endif

#if defined _UNICOS
#  define POSH_OS_UNICOS 1
#  define POSH_OS_STRING "UNICOS"
#endif

//ACS if we're in xcode, look at the target conditionals to figure out if this is ios or osx
#if defined __APPLE__
#  include "TargetConditionals.h"
#endif
#if TARGET_OS_IPHONE
#    define POSH_OS_IOS 1
#    define POSH_OS_STRING "iOS"
#else
#  if ( defined __MWERKS__ && defined __powerc && !defined macintosh ) || defined __APPLE_CC__ || defined macosx
#    define POSH_OS_OSX 1
#    define POSH_OS_STRING "MacOS X"
#  endif
#endif

#if defined __sun__ || defined sun || defined __sun || defined __solaris__
#  if defined __SVR4 || defined __svr4__ || defined __solaris__
#     define POSH_OS_STRING "Solaris"
#     define POSH_OS_SOLARIS 1
#  endif
#  if !defined POSH_OS_STRING
#     define POSH_OS_STRING "SunOS"
#     define POSH_OS_SUNOS 1
#  endif
#endif

#if defined __sgi__ || defined sgi || defined __sgi
#  define POSH_OS_IRIX 1
#  define POSH_OS_STRING "Irix"
#endif

#if defined __hpux__ || defined __hpux
#  define POSH_OS_HPUX 1
#  define POSH_OS_STRING "HP-UX"
#endif

#if defined _AIX
#  define POSH_OS_AIX 1
#  define POSH_OS_STRING "AIX"
#endif

#if ( defined __alpha && defined __osf__ )
#  define POSH_OS_TRU64 1
#  define POSH_OS_STRING "Tru64"
#endif

#if defined __BEOS__ || defined __beos__
#  define POSH_OS_BEOS 1
#  define POSH_OS_STRING "BeOS"
#endif

#if defined amiga || defined amigados || defined AMIGA || defined _AMIGA
#  define POSH_OS_AMIGA 1
#  define POSH_OS_STRING "Amiga"
#endif

#if defined __unix__
#  define POSH_OS_UNIX 1 
#  if !defined POSH_OS_STRING
#     define POSH_OS_STRING "Unix-like(generic)"
#  endif
#endif

#if defined _WIN32_WCE
#  define POSH_OS_WINCE 1
#  define POSH_OS_STRING "Windows CE"
#endif

#if defined _XBOX || defined _XBOX_VER
#  define POSH_OS_XBOX 1
#  define POSH_OS_STRING "XBOX"
#endif

#if defined __ORBIS__
#   define POSH_OS_ORBIS
#endif

#if defined _WIN32 || defined WIN32 || defined __NT__ || defined __WIN32__
#  if !defined POSH_OS_XBOX
#  define POSH_OS_WIN32 1
#     if defined _WIN64
#        define POSH_OS_WIN64 1
#        define POSH_OS_STRING "Win64"
#     else
#        if !defined POSH_OS_STRING
#           define POSH_OS_STRING "Win32"
#        endif
#     endif
#  endif
#endif

#if defined __palmos__
#  define POSH_OS_PALM 1
#  define POSH_OS_STRING "PalmOS"
#endif

#if defined THINK_C || defined macintosh
#  define POSH_OS_MACOS 1
#  define POSH_OS_STRING "MacOS"
#endif

/*
** -----------------------------------------------------------------------------
** Determine target CPU
** -----------------------------------------------------------------------------
*/

#if defined GEKKO
#  define POSH_CPU_PPC750 1
#  define POSH_CPU_STRING "IBM PowerPC 750 (NGC)"
#endif

#if defined mc68000 || defined m68k || defined __MC68K__ || defined m68000
#  define POSH_CPU_68K 1
#  define POSH_CPU_STRING "MC68000"
#endif

#if defined __PPC__ || defined __POWERPC__  || defined powerpc || defined _POWER || defined __ppc__ || defined __powerpc__ || defined _M_PPC
#  define POSH_CPU_PPC 1
#  if !defined POSH_CPU_STRING
#    if defined __powerpc64__
#       define POSH_CPU_STRING "PowerPC64"
#    else
#       define POSH_CPU_STRING "PowerPC"
#    endif
#  endif
#endif

#if defined _CRAYT3E || defined _CRAYMPP
#  define POSH_CPU_CRAYT3E 1 /* target processor is a DEC Alpha 21164 used in a Cray T3E*/
#  define POSH_CPU_STRING "Cray T3E (Alpha 21164)"
#endif

#if defined CRAY || defined _CRAY && !defined _CRAYT3E
#  error Non-AXP Cray systems not supported
#endif

#if defined _SH3
#  define POSH_CPU_SH3 1
#  define POSH_CPU_STRING "Hitachi SH-3"
#endif

#if defined __sh4__ || defined __SH4__
#  define POSH_CPU_SH3 1
#  define POSH_CPU_SH4 1
#  define POSH_CPU_STRING "Hitachi SH-4"
#endif

#if defined __sparc__ || defined __sparc
#  if defined __arch64__ || defined __sparcv9 || defined __sparc_v9__
#     define POSH_CPU_SPARC64 1 
#     define POSH_CPU_STRING "Sparc/64"
#  else
#     define POSH_CPU_STRING "Sparc/32"
#  endif
#  define POSH_CPU_SPARC 1
#endif

#if defined ARM || defined __arm__ || defined _ARM
#  define POSH_CPU_STRONGARM 1
#  define POSH_CPU_STRING "ARM"
#endif

#if defined mips || defined __mips__ || defined __MIPS__ || defined _MIPS
#  define POSH_CPU_MIPS 1 
#  if defined _R5900
#    define POSH_CPU_STRING "MIPS R5900 (PS2)"
#  else
#    define POSH_CPU_STRING "MIPS"
#  endif
#endif

#if defined __ia64 || defined _M_IA64 || defined __ia64__ 
#  define POSH_CPU_IA64 1
#  define POSH_CPU_STRING "IA64"
#endif

#if defined __X86__ || defined __i386__ || defined i386 || defined _M_IX86 || defined __386__ || defined __x86_64__ || defined _M_X64
#  define POSH_CPU_X86 1
#  if defined __x86_64__ || defined _M_X64
#     define POSH_CPU_X86_64 1 
#  endif
#  if defined POSH_CPU_X86_64
#     define POSH_CPU_STRING "AMD x86-64"
#  else
#     define POSH_CPU_STRING "Intel 386+"
#  endif
#endif

#if defined __alpha || defined alpha || defined _M_ALPHA || defined __alpha__
#  define POSH_CPU_AXP 1
#  define POSH_CPU_STRING "AXP"
#endif

#if defined __hppa || defined hppa
#  define POSH_CPU_HPPA 1
#  define POSH_CPU_STRING "PA-RISC"
#endif

#if !defined POSH_CPU_STRING
#  error POSH cannot determine target CPU
#  define POSH_CPU_STRING "Unknown" /* this is here for Doxygen's benefit */
#endif

/*
** -----------------------------------------------------------------------------
** Attempt to autodetect building for embedded on Sony PS2
** -----------------------------------------------------------------------------
*/
#if !defined POSH_OS_STRING
#  if !defined FORCE_DOXYGEN
#    define POSH_OS_EMBEDDED 1 
#  endif
#  if defined _R5900
#     define POSH_OS_STRING "Sony PS2(embedded)"
#  else
#     define POSH_OS_STRING "Embedded/Unknown"
#  endif
#endif

/*
** ---------------------------------------------------------------------------
** Handle cdecl, stdcall, fastcall, etc.
** ---------------------------------------------------------------------------
*/
#if defined POSH_CPU_X86 && !defined POSH_CPU_X86_64
#  if defined __GNUC__
#     define POSH_CDECL __attribute__((cdecl))
#     define POSH_STDCALL __attribute__((stdcall))
#     define POSH_FASTCALL __attribute__((fastcall))
#  elif ( defined _MSC_VER || defined __WATCOMC__ || defined __BORLANDC__ || defined __MWERKS__ )
#     define POSH_CDECL    __cdecl
#     define POSH_STDCALL  __stdcall
#     define POSH_FASTCALL __fastcall
#  endif
#else
#  define POSH_CDECL    
#  define POSH_STDCALL  
#  define POSH_FASTCALL 
#endif

/*
** ---------------------------------------------------------------------------
** Define POSH_IMPORTEXPORT signature based on POSH_DLL and POSH_BUILDING_LIB
** ---------------------------------------------------------------------------
*/

/*
** We undefine this so that multiple inclusions will work
*/
#if defined POSH_IMPORTEXPORT
#  undef POSH_IMPORTEXPORT
#endif

#if defined POSH_DLL
#   if defined POSH_OS_WIN32
#      if defined _MSC_VER 
#         if ( _MSC_VER >= 800 )
#            if defined POSH_BUILDING_LIB
#               define POSH_IMPORTEXPORT __declspec( dllexport )
#            else
#               define POSH_IMPORTEXPORT __declspec( dllimport )
#            endif
#         else
#            if defined POSH_BUILDING_LIB
#               define POSH_IMPORTEXPORT __export
#            else
#               define POSH_IMPORTEXPORT 
#            endif
#         endif
#      endif  /* defined _MSC_VER */
#      if defined __BORLANDC__
#         if ( __BORLANDC__ >= 0x500 )
#            if defined POSH_BUILDING_LIB 
#               define POSH_IMPORTEXPORT __declspec( dllexport )
#            else
#               define POSH_IMPORTEXPORT __declspec( dllimport )
#            endif
#         else
#            if defined POSH_BUILDING_LIB
#               define POSH_IMPORTEXPORT __export
#            else
#               define POSH_IMPORTEXPORT 
#            endif
#         endif
#      endif /* defined __BORLANDC__ */
       /* for all other compilers, we're just making a blanket assumption */
#      if defined __GNUC__ || defined __WATCOMC__ || defined __MWERKS__
#         if defined POSH_BUILDING_LIB
#            define POSH_IMPORTEXPORT __declspec( dllexport )
#         else
#            define POSH_IMPORTEXPORT __declspec( dllimport )
#         endif
#      endif /* all other compilers */
#      if !defined POSH_IMPORTEXPORT
#         error Building DLLs not supported on this compiler (poshlib@poshlib.org if you know how)
#      endif
#   endif /* defined POSH_OS_WIN32 */
#endif

/* On pretty much everything else, we can thankfully just ignore this */
#if !defined POSH_IMPORTEXPORT
#  define POSH_IMPORTEXPORT
#endif

#if defined FORCE_DOXYGEN
#  define POSH_DLL    
#  define POSH_BUILDING_LIB
#  undef POSH_DLL
#  undef POSH_BUILDING_LIB
#endif

/*
** ----------------------------------------------------------------------------
** (Re)define POSH_PUBLIC_API export signature 
** ----------------------------------------------------------------------------
*/
#ifdef POSH_PUBLIC_API
#  undef POSH_PUBLIC_API
#endif

#if ( ( defined _MSC_VER ) && ( _MSC_VER < 800 ) ) || ( defined __BORLANDC__ && ( __BORLANDC__ < 0x500 ) )
#  define POSH_PUBLIC_API(rtype) extern rtype POSH_IMPORTEXPORT 
#else
#  define POSH_PUBLIC_API(rtype) extern POSH_IMPORTEXPORT rtype
#endif

/*
** ----------------------------------------------------------------------------
** Try to infer endianess.  Basically we just go through the CPUs we know are
** little endian, and assume anything that isn't one of those is big endian.
** As a sanity check, we also do this with operating systems we know are
** little endian, such as Windows.  Some processors are bi-endian, such as 
** the MIPS series, so we have to be careful about those.
** ----------------------------------------------------------------------------
*/
#if defined POSH_CPU_X86 || defined POSH_CPU_AXP || defined POSH_CPU_STRONGARM || defined POSH_OS_WIN32 || defined POSH_OS_WINCE || defined __MIPSEL__
#  define POSH_ENDIAN_STRING "little"
#  define POSH_LITTLE_ENDIAN 1
#else
#  define POSH_ENDIAN_STRING "big"
#  define POSH_BIG_ENDIAN 1
#endif

#if defined FORCE_DOXYGEN
#  define POSH_LITTLE_ENDIAN
#endif

/*
** ----------------------------------------------------------------------------
** Cross-platform compile time assertion macro
** ----------------------------------------------------------------------------
*/
#define POSH_COMPILE_TIME_ASSERT(name, x) typedef int _POSH_dummy_ ## name[(x) ? 1 : -1 ]

/*
** ----------------------------------------------------------------------------
** 64-bit Integer
**
** We don't require 64-bit support, nor do we emulate its functionality, we
** simply export it if it's available.  Since we can't count on <limits.h>
** for 64-bit support, we ignore the POSH_USE_LIMITS_H directive.
** ----------------------------------------------------------------------------
*/
#if defined ( __LP64__ ) || defined ( __powerpc64__ ) || defined POSH_CPU_SPARC64
#  define POSH_64BIT_INTEGER 1
typedef long posh_i64_t; 
typedef unsigned long posh_u64_t;
#  define POSH_I64( x ) ((posh_i64_t)x)
#  define POSH_U64( x ) ((posh_u64_t)x)
#  define POSH_I64_PRINTF_PREFIX "l"
#elif defined _MSC_VER || defined __BORLANDC__ || defined __WATCOMC__ || ( defined __alpha && defined __DECC )
#  define POSH_64BIT_INTEGER 1
typedef __int64 posh_i64_t;
typedef unsigned __int64 posh_u64_t;
#  define POSH_I64( x ) ((posh_i64_t)x)
#  define POSH_U64( x ) ((posh_u64_t)x)
#  define POSH_I64_PRINTF_PREFIX "I64"
#elif defined __GNUC__ || defined __MWERKS__ || defined __SUNPRO_C || defined __SUNPRO_CC || defined __APPLE_CC__ || defined POSH_OS_IRIX || defined _LONG_LONG || defined _CRAYC
#  define POSH_64BIT_INTEGER 1
typedef long long posh_i64_t;
typedef unsigned long long posh_u64_t;
#  define POSH_U64( x ) ((posh_u64_t)(x##LL))
#  define POSH_I64( x ) ((posh_i64_t)(x##LL))
#  define POSH_I64_PRINTF_PREFIX "ll"
#endif

/* hack */
/*#ifdef __MINGW32__
#undef POSH_I64
#undef POSH_U64
#undef POSH_I64_PRINTF_PREFIX
#define POSH_I64( x ) ((posh_i64_t)x)
#define POSH_U64( x ) ((posh_u64_t)x)
#define POSH_I64_PRINTF_PREFIX "I64"
#endif*/

#ifdef FORCE_DOXYGEN
typedef long long posh_i64_t;
typedef unsigned long posh_u64_t;
#  define POSH_64BIT_INTEGER
#  define POSH_I64_PRINTF_PREFIX
#  define POSH_I64(x)
#  define POSH_U64(x)
#endif

/** Minimum value for a 64-bit signed integer */
#define POSH_I64_MIN  POSH_I64(0x8000000000000000)
/** Maximum value for a 64-bit signed integer */
#define POSH_I64_MAX  POSH_I64(0x7FFFFFFFFFFFFFFF)
/** Minimum value for a 64-bit unsigned integer */
#define POSH_U64_MIN  POSH_U64(0)
/** Maximum value for a 64-bit unsigned integer */
#define POSH_U64_MAX  POSH_U64(0xFFFFFFFFFFFFFFFF)

/* ----------------------------------------------------------------------------
** Basic Sized Types
**
** These types are expected to be EXACTLY sized so you can use them for
** serialization.
** ----------------------------------------------------------------------------
*/
#define POSH_FALSE 0 
#define POSH_TRUE  1 

typedef int            posh_bool_t;
typedef unsigned char  posh_byte_t;

/* NOTE: These assume that CHAR_BIT is 8!! */
typedef unsigned char  posh_u8_t;
typedef signed char    posh_i8_t;

#if defined POSH_USE_LIMITS_H
#  if CHAR_BITS > 8
#    error This machine uses 9-bit characters.  This is a warning, you can comment this out now.
#  endif /* CHAR_BITS > 8 */

/* 16-bit */
#  if ( USHRT_MAX == 65535 ) 
   typedef unsigned short posh_u16_t;
   typedef short          posh_i16_t;
#  else
   /* Yes, in theory there could still be a 16-bit character type and shorts are
      32-bits in size...if you find such an architecture, let me know =P */
#    error No 16-bit type found
#  endif

/* 32-bit */
#  if ( INT_MAX == 2147483647 )
  typedef unsigned       posh_u32_t;
  typedef int            posh_i32_t;
#  elif ( LONG_MAX == 2147483647 )
  typedef unsigned long  posh_u32_t;
  typedef long           posh_i32_t;
#  else
      error No 32-bit type found
#  endif

#else /* POSH_USE_LIMITS_H */

  typedef unsigned short posh_u16_t;
  typedef short          posh_i16_t;

#  if !defined POSH_OS_PALM
  typedef unsigned       posh_u32_t;
  typedef int            posh_i32_t;
#  else
  typedef unsigned long  posh_u32_t;
  typedef long           posh_i32_t;
#  endif
#endif

/** Minimum value for a byte */
#define POSH_BYTE_MIN    0
/** Maximum value for an 8-bit unsigned value */
#define POSH_BYTE_MAX    255
/** Minimum value for a byte */
#define POSH_I16_MIN     ( ( posh_i16_t ) 0x8000 )
/** Maximum value for a 16-bit signed value */
#define POSH_I16_MAX     ( ( posh_i16_t ) 0x7FFF ) 
/** Minimum value for a 16-bit unsigned value */
#define POSH_U16_MIN     0
/** Maximum value for a 16-bit unsigned value */
#define POSH_U16_MAX     ( ( posh_u16_t ) 0xFFFF )
/** Minimum value for a 32-bit signed value */
#define POSH_I32_MIN     ( ( posh_i32_t ) 0x80000000 )
/** Maximum value for a 32-bit signed value */
#define POSH_I32_MAX     ( ( posh_i32_t ) 0x7FFFFFFF )
/** Minimum value for a 32-bit unsigned value */
#define POSH_U32_MIN     0
/** Maximum value for a 32-bit unsigned value */
#define POSH_U32_MAX     ( ( posh_u32_t ) 0xFFFFFFFF )

/*
** ----------------------------------------------------------------------------
** Sanity checks on expected sizes
** ----------------------------------------------------------------------------
*/
#if !defined FORCE_DOXYGEN

POSH_COMPILE_TIME_ASSERT(posh_byte_t, sizeof(posh_byte_t) == 1);
POSH_COMPILE_TIME_ASSERT(posh_u8_t, sizeof(posh_u8_t) == 1);
POSH_COMPILE_TIME_ASSERT(posh_i8_t, sizeof(posh_i8_t) == 1);
POSH_COMPILE_TIME_ASSERT(posh_u16_t, sizeof(posh_u16_t) == 2);
POSH_COMPILE_TIME_ASSERT(posh_i16_t, sizeof(posh_i16_t) == 2);
POSH_COMPILE_TIME_ASSERT(posh_u32_t, sizeof(posh_u32_t) == 4);
POSH_COMPILE_TIME_ASSERT(posh_i32_t, sizeof(posh_i32_t) == 4);

#if !defined POSH_NO_FLOAT
   POSH_COMPILE_TIME_ASSERT(posh_testfloat_t, sizeof(float)==4 );
   POSH_COMPILE_TIME_ASSERT(posh_testdouble_t, sizeof(double)==8);
#endif

#if defined POSH_64BIT_INTEGER
   POSH_COMPILE_TIME_ASSERT(posh_u64_t, sizeof(posh_u64_t) == 8);
   POSH_COMPILE_TIME_ASSERT(posh_i64_t, sizeof(posh_i64_t) == 8);
#endif

#endif

/*
** ----------------------------------------------------------------------------
** 64-bit pointer support
** ----------------------------------------------------------------------------
*/
#if defined POSH_CPU_AXP && ( defined POSH_OS_TRU64 || defined POSH_OS_LINUX )
#  define POSH_64BIT_POINTER 1
#endif

#if defined POSH_CPU_X86_64 && defined POSH_OS_LINUX
#  define POSH_64BIT_POINTER 1
#endif

#if defined POSH_CPU_SPARC64 || defined POSH_OS_WIN64 || defined __64BIT__ || defined __LP64 || defined _LP64 || defined __LP64__ || defined _ADDR64 || defined _CRAYC
#   define POSH_64BIT_POINTER 1
#endif

#if defined POSH_64BIT_POINTER
   POSH_COMPILE_TIME_ASSERT( posh_64bit_pointer, sizeof( void * ) == 8 );
#elif !defined FORCE_DOXYGEN
/* if this assertion is hit then you're on a system that either has 64-bit
   addressing and we didn't catch it, or you're on a system with 16-bit
   pointers.  In the latter case, POSH doesn't actually care, we're just
   triggering this assertion to make sure you're aware of the situation,
   so feel free to delete it.

   If this assertion is triggered on a known 32 or 64-bit platform, 
   please let us know (poshlib@poshlib.org) */
   POSH_COMPILE_TIME_ASSERT( posh_32bit_pointer, sizeof( void * ) == 4 );
#endif

#if defined FORCE_DOXYGEN
#  define POSH_64BIT_POINTER
#endif

/*
** ----------------------------------------------------------------------------
** POSH Utility Functions
**
** These are optional POSH utility functions that are not required if you don't
** need anything except static checking of your host and target environment.
** 
** These functions are NOT wrapped with POSH_PUBLIC_API because I didn't want
** to enforce their export if your own library is only using them internally.
** ----------------------------------------------------------------------------
*/
#ifdef __cplusplus
extern "C" {
#endif

const char *POSH_GetArchString( void );

#if !defined POSH_NO_FLOAT

posh_u32_t  POSH_LittleFloatBits( float f );
posh_u32_t  POSH_BigFloatBits( float f );
float       POSH_FloatFromLittleBits( posh_u32_t bits );
float       POSH_FloatFromBigBits( posh_u32_t bits );

void        POSH_DoubleBits( double d, posh_byte_t dst[ 8 ] );
double      POSH_DoubleFromBits( const posh_byte_t src[ 8 ] );

/* unimplemented
float      *POSH_WriteFloatToLittle( void *dst, float f );
float      *POSH_WriteFloatToBig( void *dst, float f );
float       POSH_ReadFloatFromLittle( const void *src );
float       POSH_ReadFloatFromBig( const void *src );

double     *POSH_WriteDoubleToLittle( void *dst, double d );
double     *POSH_WriteDoubleToBig( void *dst, double d );
double      POSH_ReadDoubleFromLittle( const void *src );
double      POSH_ReadDoubleFromBig( const void *src );
*/
#endif /* !defined POSH_NO_FLOAT */

#if defined FORCE_DOXYGEN
#  define POSH_NO_FLOAT
#  undef  POSH_NO_FLOAT
#endif

extern posh_u16_t  POSH_SwapU16( posh_u16_t u );
extern posh_i16_t  POSH_SwapI16( posh_i16_t u );
extern posh_u32_t  POSH_SwapU32( posh_u32_t u );
extern posh_i32_t  POSH_SwapI32( posh_i32_t u );

#if defined POSH_64BIT_INTEGER

extern posh_u64_t  POSH_SwapU64( posh_u64_t u );
extern posh_i64_t  POSH_SwapI64( posh_i64_t u );

#endif /*POSH_64BIT_INTEGER */

extern posh_u16_t *POSH_WriteU16ToLittle( void *dst, posh_u16_t value );
extern posh_i16_t *POSH_WriteI16ToLittle( void *dst, posh_i16_t value );
extern posh_u32_t *POSH_WriteU32ToLittle( void *dst, posh_u32_t value );
extern posh_i32_t *POSH_WriteI32ToLittle( void *dst, posh_i32_t value );

extern posh_u16_t *POSH_WriteU16ToBig( void *dst, posh_u16_t value );
extern posh_i16_t *POSH_WriteI16ToBig( void *dst, posh_i16_t value );
extern posh_u32_t *POSH_WriteU32ToBig( void *dst, posh_u32_t value );
extern posh_i32_t *POSH_WriteI32ToBig( void *dst, posh_i32_t value );

extern posh_u16_t  POSH_ReadU16FromLittle( const void *src );
extern posh_i16_t  POSH_ReadI16FromLittle( const void *src );
extern posh_u32_t  POSH_ReadU32FromLittle( const void *src );
extern posh_i32_t  POSH_ReadI32FromLittle( const void *src );

extern posh_u16_t  POSH_ReadU16FromBig( const void *src );
extern posh_i16_t  POSH_ReadI16FromBig( const void *src );
extern posh_u32_t  POSH_ReadU32FromBig( const void *src );
extern posh_i32_t  POSH_ReadI32FromBig( const void *src );

#if defined POSH_64BIT_INTEGER
extern posh_u64_t *POSH_WriteU64ToLittle( void *dst, posh_u64_t value );
extern posh_i64_t *POSH_WriteI64ToLittle( void *dst, posh_i64_t value );
extern posh_u64_t *POSH_WriteU64ToBig( void *dst, posh_u64_t value );
extern posh_i64_t *POSH_WriteI64ToBig( void *dst, posh_i64_t value );

extern posh_u64_t  POSH_ReadU64FromLittle( const void *src );
extern posh_i64_t  POSH_ReadI64FromLittle( const void *src );
extern posh_u64_t  POSH_ReadU64FromBig( const void *src );
extern posh_i64_t  POSH_ReadI64FromBig( const void *src );
#endif /* POSH_64BIT_INTEGER */

#if defined POSH_LITTLE_ENDIAN

#  define POSH_LittleU16(x) (x)
#  define POSH_LittleU32(x) (x)
#  define POSH_LittleI16(x) (x)
#  define POSH_LittleI32(x) (x)
#  if defined POSH_64BIT_INTEGER
#    define POSH_LittleU64(x) (x)
#    define POSH_LittleI64(x) (x)
#  endif /* defined POSH_64BIT_INTEGER */

#  define POSH_BigU16(x) POSH_SwapU16(x)
#  define POSH_BigU32(x) POSH_SwapU32(x)
#  define POSH_BigI16(x) POSH_SwapI16(x)
#  define POSH_BigI32(x) POSH_SwapI32(x)
#  if defined POSH_64BIT_INTEGER
#    define POSH_BigU64(x) POSH_SwapU64(x)
#    define POSH_BigI64(x) POSH_SwapI64(x)
#  endif /* defined POSH_64BIT_INTEGER */

#else

#  define POSH_BigU16(x) (x)
#  define POSH_BigU32(x) (x)
#  define POSH_BigI16(x) (x)
#  define POSH_BigI32(x) (x)

#  if defined POSH_64BIT_INTEGER
#    define POSH_BigU64(x) (x)
#    define POSH_BigI64(x) (x)
#  endif /* POSH_64BIT_INTEGER */

#  define POSH_LittleU16(x) POSH_SwapU16(x)
#  define POSH_LittleU32(x) POSH_SwapU32(x)
#  define POSH_LittleI16(x) POSH_SwapI16(x)
#  define POSH_LittleI32(x) POSH_SwapI32(x)

#  if defined POSH_64BIT_INTEGER
#    define POSH_LittleU64(x) POSH_SwapU64(x)
#    define POSH_LittleI64(x) POSH_SwapI64(x)
#  endif /* POSH_64BIT_INTEGER */

#endif

#ifdef __cplusplus
}
#endif


