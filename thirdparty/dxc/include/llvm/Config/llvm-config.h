/*===------- llvm/Config/llvm-config.h - llvm configuration -------*- C -*-===*/
/*                                                                            */
/*                     The LLVM Compiler Infrastructure                       */
/*                                                                            */
/* This file is distributed under the University of Illinois Open Source      */
/* License. See LICENSE.TXT for details.                                      */
/*                                                                            */
/*===----------------------------------------------------------------------===*/

/* This file enumerates variables from the LLVM configuration so that they
   can be in exported headers and won't override package specific directives.
   This is a C header that can be included in the llvm-c headers. */

#ifndef LLVM_CONFIG_H
#define LLVM_CONFIG_H

/* Installation directory for binary executables */
#define LLVM_BINDIR ""

/* Time at which LLVM was configured */
#define LLVM_CONFIGTIME ""

/* Installation directory for data files */
#define LLVM_DATADIR ""

/* Target triple LLVM will generate code for by default */
#define LLVM_DEFAULT_TARGET_TRIPLE "dxil-ms-dx"

/* Installation directory for documentation */
#define LLVM_DOCSDIR ""

/* Define if LLVM is built with asserts and checks that change the layout of
   client-visible data structures.  */
#undef LLVM_ENABLE_ABI_BREAKING_CHECKS

/* Define if threads enabled */
#define LLVM_ENABLE_THREADS 0

/* Installation directory for config files */
#define LLVM_ETCDIR ""

/* Has gcc/MSVC atomic intrinsics */
#define LLVM_HAS_ATOMICS 1

/* Host triple LLVM will be executed on */
#define LLVM_HOST_TRIPLE ""

/* Installation directory for include files */
#define LLVM_INCLUDEDIR ""

/* Installation directory for .info files */
#define LLVM_INFODIR ""

/* Installation directory for man pages */
#define LLVM_MANDIR ""

/* LLVM architecture name for the native architecture, if available */
#undef LLVM_NATIVE_ARCH

/* LLVM name for the native AsmParser init function, if available */
#undef LLVM_NATIVE_ASMPARSER

/* LLVM name for the native AsmPrinter init function, if available */
#undef LLVM_NATIVE_ASMPRINTER

/* LLVM name for the native Disassembler init function, if available */
#undef LLVM_NATIVE_DISASSEMBLER

/* LLVM name for the native Target init function, if available */
#undef LLVM_NATIVE_TARGET

/* LLVM name for the native TargetInfo init function, if available */
#undef LLVM_NATIVE_TARGETINFO

/* LLVM name for the native target MC init function, if available */
#undef LLVM_NATIVE_TARGETMC

/* Define if this is Unixish platform */
#ifdef UNIX_ENABLED
#define LLVM_ON_UNIX 1
#endif

/* Define if this is Win32ish platform */
#ifdef WINDOWS_ENABLED
#define LLVM_ON_WIN32 1
#endif

/* Installation prefix directory */
#define LLVM_PREFIX ""

/* Define if we have the Intel JIT API runtime support library */
#define LLVM_USE_INTEL_JITEVENTS 0

/* Define if we have the oprofile JIT-support library */
#define LLVM_USE_OPROFILE 0

/* Major version of the LLVM API */
#define LLVM_VERSION_MAJOR 3

/* Minor version of the LLVM API */
#define LLVM_VERSION_MINOR 7

/* Patch version of the LLVM API */
#define LLVM_VERSION_PATCH 0

/* LLVM version string */
#define LLVM_VERSION_STRING "3.7-v1.7.2207"

/* Define if we link Polly to the tools */
#undef LINK_POLLY_INTO_TOOLS

#endif
