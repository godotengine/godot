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
#cmakedefine LLVM_BINDIR "${LLVM_BINDIR}"

/* Time at which LLVM was configured */
#cmakedefine LLVM_CONFIGTIME "${LLVM_CONFIGTIME}"

/* Installation directory for data files */
#cmakedefine LLVM_DATADIR "${LLVM_DATADIR}"

/* Target triple LLVM will generate code for by default */
#cmakedefine LLVM_DEFAULT_TARGET_TRIPLE "${LLVM_DEFAULT_TARGET_TRIPLE}"

/* Installation directory for documentation */
#cmakedefine LLVM_DOCSDIR "${LLVM_DOCSDIR}"

/* Define if LLVM is built with asserts and checks that change the layout of
   client-visible data structures.  */
#cmakedefine LLVM_ENABLE_ABI_BREAKING_CHECKS

/* Define if threads enabled */
#cmakedefine01 LLVM_ENABLE_THREADS

/* Installation directory for config files */
#cmakedefine LLVM_ETCDIR "${LLVM_ETCDIR}"

/* Has gcc/MSVC atomic intrinsics */
#cmakedefine01 LLVM_HAS_ATOMICS

/* Host triple LLVM will be executed on */
#cmakedefine LLVM_HOST_TRIPLE "${LLVM_HOST_TRIPLE}"

/* Installation directory for include files */
#cmakedefine LLVM_INCLUDEDIR "${LLVM_INCLUDEDIR}"

/* Installation directory for .info files */
#cmakedefine LLVM_INFODIR "${LLVM_INFODIR}"

/* Installation directory for man pages */
#cmakedefine LLVM_MANDIR "${LLVM_MANDIR}"

/* LLVM architecture name for the native architecture, if available */
#cmakedefine LLVM_NATIVE_ARCH ${LLVM_NATIVE_ARCH}

/* LLVM name for the native AsmParser init function, if available */
#cmakedefine LLVM_NATIVE_ASMPARSER LLVMInitialize${LLVM_NATIVE_ARCH}AsmParser

/* LLVM name for the native AsmPrinter init function, if available */
#cmakedefine LLVM_NATIVE_ASMPRINTER LLVMInitialize${LLVM_NATIVE_ARCH}AsmPrinter

/* LLVM name for the native Disassembler init function, if available */
#cmakedefine LLVM_NATIVE_DISASSEMBLER LLVMInitialize${LLVM_NATIVE_ARCH}Disassembler

/* LLVM name for the native Target init function, if available */
#cmakedefine LLVM_NATIVE_TARGET LLVMInitialize${LLVM_NATIVE_ARCH}Target

/* LLVM name for the native TargetInfo init function, if available */
#cmakedefine LLVM_NATIVE_TARGETINFO LLVMInitialize${LLVM_NATIVE_ARCH}TargetInfo

/* LLVM name for the native target MC init function, if available */
#cmakedefine LLVM_NATIVE_TARGETMC LLVMInitialize${LLVM_NATIVE_ARCH}TargetMC

/* Define if this is Unixish platform */
#cmakedefine LLVM_ON_UNIX ${LLVM_ON_UNIX}

/* Define if this is Win32ish platform */
#cmakedefine LLVM_ON_WIN32 ${LLVM_ON_WIN32}

/* Installation prefix directory */
#cmakedefine LLVM_PREFIX "${LLVM_PREFIX}"

/* Define if we have the Intel JIT API runtime support library */
#cmakedefine LLVM_USE_INTEL_JITEVENTS 1

/* Define if we have the oprofile JIT-support library */
#cmakedefine LLVM_USE_OPROFILE 1

/* Major version of the LLVM API */
#define LLVM_VERSION_MAJOR ${LLVM_VERSION_MAJOR}

/* Minor version of the LLVM API */
#define LLVM_VERSION_MINOR ${LLVM_VERSION_MINOR}

/* Patch version of the LLVM API */
#define LLVM_VERSION_PATCH ${LLVM_VERSION_PATCH}

/* LLVM version string */
#define LLVM_VERSION_STRING "${PACKAGE_VERSION}"

/* Define if we link Polly to the tools */
#cmakedefine LINK_POLLY_INTO_TOOLS

#endif
