//===- llvm/Transforms/Utils/BypassSlowDivision.h --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains an optimization for div and rem on architectures that
// execute short instructions significantly faster than longer instructions.
// For example, on Intel Atom 32-bit divides are slow enough that during
// runtime it is profitable to check the value of the operands, and if they are
// positive and less than 256 use an unsigned 8-bit divide.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_BYPASSSLOWDIVISION_H
#define LLVM_TRANSFORMS_UTILS_BYPASSSLOWDIVISION_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Function.h"

namespace llvm {

/// This optimization identifies DIV instructions that can be
/// profitably bypassed and carried out with a shorter, faster divide.
bool bypassSlowDivision(Function &F,
                        Function::iterator &I,
                        const DenseMap<unsigned int, unsigned int> &BypassWidth);

} // End llvm namespace

#endif
