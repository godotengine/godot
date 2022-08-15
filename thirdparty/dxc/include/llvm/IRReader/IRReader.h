//===---- llvm/IRReader/IRReader.h - Reader for LLVM IR files ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines functions for reading LLVM IR. They support both
// Bitcode and Assembly, automatically detecting the input format.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IRREADER_IRREADER_H
#define LLVM_IRREADER_IRREADER_H

#include "llvm/Support/MemoryBuffer.h"
#include <string>

namespace llvm {

class Module;
class SMDiagnostic;
class LLVMContext;

/// If the given file holds a bitcode image, return a Module
/// for it which does lazy deserialization of function bodies.  Otherwise,
/// attempt to parse it as LLVM Assembly and return a fully populated
/// Module.
std::unique_ptr<Module> getLazyIRFileModule(StringRef Filename,
                                            SMDiagnostic &Err,
                                            LLVMContext &Context);

/// If the given MemoryBuffer holds a bitcode image, return a Module
/// for it.  Otherwise, attempt to parse it as LLVM Assembly and return
/// a Module for it.
std::unique_ptr<Module> parseIR(MemoryBufferRef Buffer, SMDiagnostic &Err,
                                LLVMContext &Context);

/// If the given file holds a bitcode image, return a Module for it.
/// Otherwise, attempt to parse it as LLVM Assembly and return a Module
/// for it.
std::unique_ptr<Module> parseIRFile(StringRef Filename, SMDiagnostic &Err,
                                    LLVMContext &Context);
}

#endif
