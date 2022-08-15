//===-- DxilTargetTransformInfo.h - DXIL specific TTI -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares a TargetTransformInfo analysis pass specific to the DXIL.
/// Only implemented isSourceOfDivergence for DivergenceAnalysis.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/CodeGen/BasicTTIImpl.h"

namespace hlsl {
class DxilModule;
class OP;
}

namespace llvm {

class DxilTTIImpl final : public BasicTTIImplBase<DxilTTIImpl> {
  typedef BasicTTIImplBase<DxilTTIImpl> BaseT;
  typedef TargetTransformInfo TTI;
  friend BaseT;
  hlsl::OP *m_pHlslOP;
  bool m_isThreadGroup;
  const TargetSubtargetInfo *getST() const { return nullptr; }
  const TargetLowering *getTLI() const { return nullptr; }

public:
  explicit DxilTTIImpl(const TargetMachine *TM, const Function &F,
                       hlsl::DxilModule &DM, bool ThreadGroup);

  bool hasBranchDivergence() { return true; }
  bool isSourceOfDivergence(const Value *V) const;
};

} // end namespace llvm
