/*===-- llvm-c/Linker.h - Module Linker C Interface -------------*- C++ -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This file defines the C interface to the module/file/archive linker.       *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_C_LINKER_H
#define LLVM_C_LINKER_H

#include "llvm-c/Core.h"

#ifdef __cplusplus
extern "C" {
#endif

/* This enum is provided for backwards-compatibility only. It has no effect. */
typedef enum {
  LLVMLinkerDestroySource = 0, /* This is the default behavior. */
  LLVMLinkerPreserveSource_Removed = 1 /* This option has been deprecated and
                                          should not be used. */
} LLVMLinkerMode;

/* Links the source module into the destination module, taking ownership
 * of the source module away from the caller. Optionally returns a
 * human-readable description of any errors that occurred in linking.
 * OutMessage must be disposed with LLVMDisposeMessage. The return value
 * is true if an error occurred, false otherwise.
 *
 * Note that the linker mode parameter \p Unused is no longer used, and has
 * no effect. */
LLVMBool LLVMLinkModules(LLVMModuleRef Dest, LLVMModuleRef Src,
                         LLVMLinkerMode Unused, _Outptr_result_maybenull_ char **OutMessage);

#ifdef __cplusplus
}
#endif

#endif
