//===-- llvm/CodeGen/MachineCombinerPattern.h - Instruction pattern supported by
// combiner  ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines instruction pattern supported by combiner
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINECOMBINERPATTERN_H
#define LLVM_CODEGEN_MACHINECOMBINERPATTERN_H

namespace llvm {

/// Enumeration of instruction pattern supported by machine combiner
///
///
namespace MachineCombinerPattern {
// Forward declaration
enum MC_PATTERN : int;
} // end namespace MachineCombinerPattern
} // end namespace llvm

#endif
