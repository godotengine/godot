//===- llvm/IR/UseListOrder.h - LLVM Use List Order -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file has structures and command-line options for preserving use-list
// order.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_USELISTORDER_H
#define LLVM_IR_USELISTORDER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <vector>

namespace llvm {

class Module;
class Function;
class Value;

/// \brief Structure to hold a use-list order.
struct UseListOrder {
  const Value *V;
  const Function *F;
  std::vector<unsigned> Shuffle;

  UseListOrder(const Value *V, const Function *F, size_t ShuffleSize)
      : V(V), F(F), Shuffle(ShuffleSize) {}

  UseListOrder() : V(0), F(0) {}
  UseListOrder(UseListOrder &&X)
      : V(X.V), F(X.F), Shuffle(std::move(X.Shuffle)) {}
  UseListOrder &operator=(UseListOrder &&X) {
    V = X.V;
    F = X.F;
    Shuffle = std::move(X.Shuffle);
    return *this;
  }

private:
  UseListOrder(const UseListOrder &X) = delete;
  UseListOrder &operator=(const UseListOrder &X) = delete;
};

typedef std::vector<UseListOrder> UseListOrderStack;

} // end namespace llvm

#endif
