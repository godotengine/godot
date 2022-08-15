//===-- DxilTargetTransformInfo.cpp - DXIL specific TTI pass     ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// \file
// This file implements a TargetTransformInfo analysis pass specific to the
// DXIL. Only implemented isSourceOfDivergence for DivergenceAnalysis.
//
//===----------------------------------------------------------------------===//

#include "DxilTargetTransformInfo.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/DXIL/DxilOperations.h"
#include "llvm/CodeGen/BasicTTIImpl.h"

using namespace llvm;
using namespace hlsl;

#define DEBUG_TYPE "DXILtti"

// For BasicTTImpl
cl::opt<unsigned>
    llvm::PartialUnrollingThreshold("partial-unrolling-threshold", cl::init(0),
                                    cl::desc("Threshold for partial unrolling"),
                                    cl::Hidden);

DxilTTIImpl::DxilTTIImpl(const TargetMachine *TM, const Function &F,
                         hlsl::DxilModule &DM, bool ThreadGroup)
    : BaseT(TM, F.getParent()->getDataLayout()), m_pHlslOP(DM.GetOP()),
      m_isThreadGroup(ThreadGroup) {}

namespace {
bool IsDxilOpSourceOfDivergence(const CallInst *CI, OP *hlslOP,
                                bool ThreadGroup) {

  DXIL::OpCode opcode = hlslOP->GetDxilOpFuncCallInst(CI);
  switch (opcode) {
  case DXIL::OpCode::AtomicBinOp:
  case DXIL::OpCode::AtomicCompareExchange:
  case DXIL::OpCode::LoadInput:
  case DXIL::OpCode::BufferUpdateCounter:
  case DXIL::OpCode::CycleCounterLegacy:
  case DXIL::OpCode::DomainLocation:
  case DXIL::OpCode::Coverage:
  case DXIL::OpCode::EvalCentroid:
  case DXIL::OpCode::EvalSampleIndex:
  case DXIL::OpCode::EvalSnapped:
  case DXIL::OpCode::FlattenedThreadIdInGroup:
  case DXIL::OpCode::GSInstanceID:
  case DXIL::OpCode::InnerCoverage:
  case DXIL::OpCode::LoadOutputControlPoint:
  case DXIL::OpCode::LoadPatchConstant:
  case DXIL::OpCode::OutputControlPointID:
  case DXIL::OpCode::PrimitiveID:
  case DXIL::OpCode::RenderTargetGetSampleCount:
  case DXIL::OpCode::RenderTargetGetSamplePosition:
  case DXIL::OpCode::ThreadId:
  case DXIL::OpCode::ThreadIdInGroup:
    return true;
  case DXIL::OpCode::GroupId:
    return !ThreadGroup;
  default:
    return false;
  }
}
}

///
/// \returns true if the result of the value could potentially be
/// different across dispatch or thread group.
bool DxilTTIImpl::isSourceOfDivergence(const Value *V) const {

  if (dyn_cast<Argument>(V))
    return true;

  // Atomics are divergent because they are executed sequentially: when an
  // atomic operation refers to the same address in each thread, then each
  // thread after the first sees the value written by the previous thread as
  // original value.
  if (isa<AtomicRMWInst>(V) || isa<AtomicCmpXchgInst>(V))
    return true;

  if (const CallInst *CI = dyn_cast<CallInst>(V)) {
    // Assume none dxil instrincis function calls are a source of divergence.
    if (!m_pHlslOP->IsDxilOpFuncCallInst(CI))
      return true;
    return IsDxilOpSourceOfDivergence(CI, m_pHlslOP, m_isThreadGroup);
  }

  return false;
}
