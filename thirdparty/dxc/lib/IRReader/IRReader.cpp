//===---- IRReader.cpp - Reader for LLVM IR files -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IRReader/IRReader.h"
// #include "llvm-c/Core.h"
// #include "llvm-c/IRReader.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include <system_error>

using namespace llvm;

namespace llvm {
  extern bool TimePassesIsEnabled;
}

static const char *const TimeIRParsingGroupName = "LLVM IR Parsing";
static const char *const TimeIRParsingName = "Parse IR";

static std::unique_ptr<Module>
getLazyIRModule(std::unique_ptr<MemoryBuffer> Buffer, SMDiagnostic &Err,
                LLVMContext &Context) {
  if (isBitcode((const unsigned char *)Buffer->getBufferStart(),
                (const unsigned char *)Buffer->getBufferEnd())) {
    ErrorOr<std::unique_ptr<Module>> ModuleOrErr =
        getLazyBitcodeModule(std::move(Buffer), Context);
    if (std::error_code EC = ModuleOrErr.getError()) {
      Err = SMDiagnostic(Buffer->getBufferIdentifier(), SourceMgr::DK_Error,
                         EC.message());
      return nullptr;
    }
    return std::move(ModuleOrErr.get());
  }

  return parseAssembly(Buffer->getMemBufferRef(), Err, Context);
}

std::unique_ptr<Module> llvm::getLazyIRFileModule(StringRef Filename,
                                                  SMDiagnostic &Err,
                                                  LLVMContext &Context) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> FileOrErr =
      MemoryBuffer::getFileOrSTDIN(Filename);
  if (std::error_code EC = FileOrErr.getError()) {
    Err = SMDiagnostic(Filename, SourceMgr::DK_Error,
                       "Could not open input file: " + EC.message());
    return nullptr;
  }

  return getLazyIRModule(std::move(FileOrErr.get()), Err, Context);
}

std::unique_ptr<Module> llvm::parseIR(MemoryBufferRef Buffer, SMDiagnostic &Err,
                                      LLVMContext &Context) {
  NamedRegionTimer T(TimeIRParsingName, TimeIRParsingGroupName,
                     TimePassesIsEnabled);
  if (isBitcode((const unsigned char *)Buffer.getBufferStart(),
                (const unsigned char *)Buffer.getBufferEnd())) {
    ErrorOr<std::unique_ptr<Module>> ModuleOrErr =
        parseBitcodeFile(Buffer, Context);
    if (std::error_code EC = ModuleOrErr.getError()) {
      Err = SMDiagnostic(Buffer.getBufferIdentifier(), SourceMgr::DK_Error,
                         EC.message());
      return nullptr;
    }
    return std::move(ModuleOrErr.get());
  }

  return parseAssembly(Buffer, Err, Context);
}

std::unique_ptr<Module> llvm::parseIRFile(StringRef Filename, SMDiagnostic &Err,
                                          LLVMContext &Context) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> FileOrErr =
      MemoryBuffer::getFileOrSTDIN(Filename);
  if (std::error_code EC = FileOrErr.getError()) {
    Err = SMDiagnostic(Filename, SourceMgr::DK_Error,
                       "Could not open input file: " + EC.message());
    return nullptr;
  }

  return parseIR(FileOrErr.get()->getMemBufferRef(), Err, Context);
}

#if 0 // HLSL Change Starts

//===----------------------------------------------------------------------===//
// C API.
//===----------------------------------------------------------------------===//

LLVMBool LLVMParseIRInContext(LLVMContextRef ContextRef,
                              LLVMMemoryBufferRef MemBuf, LLVMModuleRef *OutM,
                              char **OutMessage) {
  SMDiagnostic Diag;

  std::unique_ptr<MemoryBuffer> MB(unwrap(MemBuf));
  *OutM =
      wrap(parseIR(MB->getMemBufferRef(), Diag, *unwrap(ContextRef)).release());

  if(!*OutM) {
    if (OutMessage) {
      std::string buf;
      raw_string_ostream os(buf);

      Diag.print(nullptr, os, false);
      os.flush();

      *OutMessage = _strdup(buf.c_str());
    }
    return 1;
  }

  return 0;
}

#endif // HLSL Change Ends
