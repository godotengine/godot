/*===- llvm/Support/Solaris.h ------------------------------------*- C++ -*-===*
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 *===----------------------------------------------------------------------===*
 *
 * This file contains portability fixes for Solaris hosts.
 *
 *===----------------------------------------------------------------------===*/
#ifndef LLVM_SUPPORT_SOLARIS_H
#define LLVM_SUPPORT_SOLARIS_H

#include <sys/types.h>
#include <sys/regset.h>

/* Solaris doesn't have endian.h. SPARC is the only supported big-endian ISA. */
#define BIG_ENDIAN 4321
#define LITTLE_ENDIAN 1234
#if defined(__sparc) || defined(__sparc__)
#define BYTE_ORDER BIG_ENDIAN
#else
#define BYTE_ORDER LITTLE_ENDIAN
#endif

#undef CS
#undef DS
#undef ES
#undef FS
#undef GS
#undef SS
#undef EAX
#undef ECX
#undef EDX
#undef EBX
#undef ESP
#undef EBP
#undef ESI
#undef EDI
#undef EIP
#undef UESP
#undef EFL
#undef ERR
#undef TRAPNO

#endif
