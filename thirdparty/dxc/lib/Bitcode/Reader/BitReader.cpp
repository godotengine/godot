//===-- BitReader.cpp -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm-c/BitReader.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>
#include <string>

using namespace llvm;

/* Builds a module from the bitcode in the specified memory buffer, returning a
   reference to the module via the OutModule parameter. Returns 0 on success.
   Optionally returns a human-readable error message via OutMessage. */
LLVMBool LLVMParseBitcode(LLVMMemoryBufferRef MemBuf,
                          LLVMModuleRef *OutModule, char **OutMessage) {
  return LLVMParseBitcodeInContext(wrap(&getGlobalContext()), MemBuf, OutModule,
                                   OutMessage);
}

LLVMBool LLVMParseBitcodeInContext(LLVMContextRef ContextRef,
                                   LLVMMemoryBufferRef MemBuf,
                                   LLVMModuleRef *OutModule,
                                   char **OutMessage) {
  MemoryBufferRef Buf = unwrap(MemBuf)->getMemBufferRef();
  LLVMContext &Ctx = *unwrap(ContextRef);

  std::string Message;
  raw_string_ostream Stream(Message);
  DiagnosticPrinterRawOStream DP(Stream);

  ErrorOr<std::unique_ptr<Module>> ModuleOrErr = parseBitcodeFile(
      Buf, Ctx, [&](const DiagnosticInfo &DI) { DI.print(DP); });
  if (ModuleOrErr.getError()) {
    if (OutMessage) {
      Stream.flush();
      *OutMessage = strdup(Message.c_str());
    }
    *OutModule = wrap((Module*)nullptr);
    return 1;
  }

  *OutModule = wrap(ModuleOrErr.get().release());
  return 0;
}

/* Reads a module from the specified path, returning via the OutModule parameter
   a module provider which performs lazy deserialization. Returns 0 on success.
   Optionally returns a human-readable error message via OutMessage. */
LLVMBool LLVMGetBitcodeModuleInContext(LLVMContextRef ContextRef,
                                       LLVMMemoryBufferRef MemBuf,
                                       LLVMModuleRef *OutM,
                                       char **OutMessage) {
  std::string Message;
  std::unique_ptr<MemoryBuffer> Owner(unwrap(MemBuf));

  ErrorOr<std::unique_ptr<Module>> ModuleOrErr =
      getLazyBitcodeModule(std::move(Owner), *unwrap(ContextRef));
  Owner.release();

  if (std::error_code EC = ModuleOrErr.getError()) {
    *OutM = wrap((Module *)nullptr);
    if (OutMessage)
      *OutMessage = strdup(EC.message().c_str());
    return 1;
  }

  *OutM = wrap(ModuleOrErr.get().release());

  return 0;

}

LLVMBool LLVMGetBitcodeModule(LLVMMemoryBufferRef MemBuf, LLVMModuleRef *OutM,
                              char **OutMessage) {
  return LLVMGetBitcodeModuleInContext(LLVMGetGlobalContext(), MemBuf, OutM,
                                       OutMessage);
}

/* Deprecated: Use LLVMGetBitcodeModuleInContext instead. */
LLVMBool LLVMGetBitcodeModuleProviderInContext(LLVMContextRef ContextRef,
                                               LLVMMemoryBufferRef MemBuf,
                                               LLVMModuleProviderRef *OutMP,
                                               char **OutMessage) {
  return LLVMGetBitcodeModuleInContext(ContextRef, MemBuf,
                                       reinterpret_cast<LLVMModuleRef*>(OutMP),
                                       OutMessage);
}

/* Deprecated: Use LLVMGetBitcodeModule instead. */
LLVMBool LLVMGetBitcodeModuleProvider(LLVMMemoryBufferRef MemBuf,
                                      LLVMModuleProviderRef *OutMP,
                                      char **OutMessage) {
  return LLVMGetBitcodeModuleProviderInContext(LLVMGetGlobalContext(), MemBuf,
                                               OutMP, OutMessage);
}
