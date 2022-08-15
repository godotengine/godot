//===-- Parser.h - Parser for LLVM IR text assembly files -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  These classes are implemented by the lib/AsmParser library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ASMPARSER_PARSER_H
#define LLVM_ASMPARSER_PARSER_H

#include "llvm/Support/MemoryBuffer.h"

namespace llvm {

class LLVMContext;
class Module;
struct SlotMapping;
class SMDiagnostic;

/// This function is the main interface to the LLVM Assembly Parser. It parses
/// an ASCII file that (presumably) contains LLVM Assembly code. It returns a
/// Module (intermediate representation) with the corresponding features. Note
/// that this does not verify that the generated Module is valid, so you should
/// run the verifier after parsing the file to check that it is okay.
/// \brief Parse LLVM Assembly from a file
/// \param Filename The name of the file to parse
/// \param Error Error result info.
/// \param Context Context in which to allocate globals info.
/// \param Slots The optional slot mapping that will be initialized during
///              parsing.
std::unique_ptr<Module> parseAssemblyFile(StringRef Filename,
                                          SMDiagnostic &Error,
                                          LLVMContext &Context,
                                          SlotMapping *Slots = nullptr);

/// The function is a secondary interface to the LLVM Assembly Parser. It parses
/// an ASCII string that (presumably) contains LLVM Assembly code. It returns a
/// Module (intermediate representation) with the corresponding features. Note
/// that this does not verify that the generated Module is valid, so you should
/// run the verifier after parsing the file to check that it is okay.
/// \brief Parse LLVM Assembly from a string
/// \param AsmString The string containing assembly
/// \param Error Error result info.
/// \param Context Context in which to allocate globals info.
/// \param Slots The optional slot mapping that will be initialized during
///              parsing.
std::unique_ptr<Module> parseAssemblyString(StringRef AsmString,
                                            SMDiagnostic &Error,
                                            LLVMContext &Context,
                                            SlotMapping *Slots = nullptr);

/// parseAssemblyFile and parseAssemblyString are wrappers around this function.
/// \brief Parse LLVM Assembly from a MemoryBuffer.
/// \param F The MemoryBuffer containing assembly
/// \param Err Error result info.
/// \param Slots The optional slot mapping that will be initialized during
///              parsing.
std::unique_ptr<Module> parseAssembly(MemoryBufferRef F, SMDiagnostic &Err,
                                      LLVMContext &Context,
                                      SlotMapping *Slots = nullptr);

/// This function is the low-level interface to the LLVM Assembly Parser.
/// This is kept as an independent function instead of being inlined into
/// parseAssembly for the convenience of interactive users that want to add
/// recently parsed bits to an existing module.
///
/// \param F The MemoryBuffer containing assembly
/// \param M The module to add data to.
/// \param Err Error result info.
/// \param Slots The optional slot mapping that will be initialized during
///              parsing.
/// \return true on error.
bool parseAssemblyInto(MemoryBufferRef F, Module &M, SMDiagnostic &Err,
                       SlotMapping *Slots = nullptr);

} // End llvm namespace

#endif
