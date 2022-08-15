///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// HLUtil.cpp                                                                //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// HL helper functions.                                                      //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/HLSL/HLUtil.h"
#include "dxc/HLSL/HLOperations.h"
#include "dxc/DXIL/DxilTypeSystem.h"

#include "dxc/Support/Global.h"

#include "llvm/IR/Operator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"

using namespace llvm;
using namespace hlsl;
using namespace hlsl::hlutil;

namespace {
void analyzePointer(const Value *V, PointerStatus &PS, DxilTypeSystem &typeSys,
                    bool bStructElt, bool bLdStOnly) {
  // Early return when only care load store.
  if (bLdStOnly) {
    if (PS.HasLoaded() && PS.HasStored())
       return;
  }
  for (const User *U : V->users()) {
    if (const Instruction *I = dyn_cast<Instruction>(U)) {
      const Function *F = I->getParent()->getParent();
      if (!PS.AccessingFunction) {
        PS.AccessingFunction = F;
      } else {
        if (F != PS.AccessingFunction)
          PS.HasMultipleAccessingFunctions = true;
      }
    }

    if (const BitCastOperator *BC = dyn_cast<BitCastOperator>(U)) {
      analyzePointer(BC, PS, typeSys, bStructElt, bLdStOnly);
    } else if (const MemCpyInst *MC = dyn_cast<MemCpyInst>(U)) {
      // Do not collect memcpy on struct GEP use.
      // These memcpy will be flattened in next level.
      if (!bStructElt) {
        MemCpyInst *MI = const_cast<MemCpyInst *>(MC);
        PS.memcpySet.insert(MI);
        bool bFullCopy = false;
        if (ConstantInt *Length = dyn_cast<ConstantInt>(MC->getLength())) {
          bFullCopy = PS.Size == Length->getLimitedValue() || PS.Size == 0 ||
                      Length->getLimitedValue() == 0; // handle unbounded arrays
        }
        if (MC->getRawDest() == V) {
          if (bFullCopy &&
              PS.storedType == PointerStatus::StoredType::NotStored) {
            PS.storedType = PointerStatus::StoredType::MemcopyDestOnce;
            PS.StoringMemcpy = MI;
          } else {
            PS.MarkAsStored();
            PS.StoringMemcpy = nullptr;
          }
        } else if (MC->getRawSource() == V) {
          if (bFullCopy &&
              PS.loadedType == PointerStatus::LoadedType::NotLoaded) {
            PS.loadedType = PointerStatus::LoadedType::MemcopySrcOnce;
            PS.LoadingMemcpy = MI;
          } else {
            PS.MarkAsLoaded();
            PS.LoadingMemcpy = nullptr;
          }
        }
      } else {
        if (MC->getRawDest() == V) {
          PS.MarkAsStored();
        } else {
          DXASSERT(MC->getRawSource() == V, "must be source here");
          PS.MarkAsLoaded();
        }
      }
    } else if (const GEPOperator *GEP = dyn_cast<GEPOperator>(U)) {
      gep_type_iterator GEPIt = gep_type_begin(GEP);
      gep_type_iterator GEPEnd = gep_type_end(GEP);
      // Skip pointer idx.
      GEPIt++;
      // Struct elt will be flattened in next level.
      bool bStructElt = (GEPIt != GEPEnd) && GEPIt->isStructTy();
      analyzePointer(GEP, PS, typeSys, bStructElt, bLdStOnly);
    } else if (const StoreInst *SI = dyn_cast<StoreInst>(U)) {
      Value *V = SI->getOperand(0);

      if (PS.storedType == PointerStatus::StoredType::NotStored) {
        PS.storedType = PointerStatus::StoredType::StoredOnce;
        PS.StoredOnceValue = V;
      } else {
        PS.MarkAsStored();
      }
    } else if (dyn_cast<LoadInst>(U)) {
      PS.MarkAsLoaded();
    } else if (const CallInst *CI = dyn_cast<CallInst>(U)) {
      Function *F = CI->getCalledFunction();
      if (F->isIntrinsic()) {
        if (F->getIntrinsicID() == Intrinsic::lifetime_start ||
            F->getIntrinsicID() == Intrinsic::lifetime_end)
          continue;
      }
      DxilFunctionAnnotation *annotation = typeSys.GetFunctionAnnotation(F);
      if (!annotation) {
        HLOpcodeGroup group = hlsl::GetHLOpcodeGroupByName(F);
        switch (group) {
        case HLOpcodeGroup::HLMatLoadStore: {
          HLMatLoadStoreOpcode opcode =
              static_cast<HLMatLoadStoreOpcode>(hlsl::GetHLOpcode(CI));
          switch (opcode) {
          case HLMatLoadStoreOpcode::ColMatLoad:
          case HLMatLoadStoreOpcode::RowMatLoad:
            PS.MarkAsLoaded();
            break;
          case HLMatLoadStoreOpcode::ColMatStore:
          case HLMatLoadStoreOpcode::RowMatStore:
            PS.MarkAsStored();
            break;
          default:
            DXASSERT(0, "invalid opcode");
            PS.MarkAsStored();
            PS.MarkAsLoaded();
          }
        } break;
        case HLOpcodeGroup::HLSubscript: {
          HLSubscriptOpcode opcode =
              static_cast<HLSubscriptOpcode>(hlsl::GetHLOpcode(CI));
          switch (opcode) {
          case HLSubscriptOpcode::VectorSubscript:
          case HLSubscriptOpcode::ColMatElement:
          case HLSubscriptOpcode::ColMatSubscript:
          case HLSubscriptOpcode::RowMatElement:
          case HLSubscriptOpcode::RowMatSubscript:
            analyzePointer(CI, PS, typeSys, bStructElt, bLdStOnly);
            break;
          default:
            // Rest are resource ptr like buf[i].
            // Only read of resource handle.
            PS.MarkAsLoaded();
            break;
          }
        } break;
        default: {
          // If not sure its out param or not. Take as out param.
          PS.MarkAsStored();
          PS.MarkAsLoaded();
        }
        }
        continue;
      }

      unsigned argSize = F->arg_size();
      for (unsigned i = 0; i < argSize; i++) {
        Value *arg = CI->getArgOperand(i);
        if (V == arg) {
          if (bLdStOnly) {
            auto &paramAnnot = annotation->GetParameterAnnotation(i);
            switch (paramAnnot.GetParamInputQual()) {
            default:
              PS.MarkAsStored();
              PS.MarkAsLoaded();
              break;
            case DxilParamInputQual::Out:
              PS.MarkAsStored();
              break;
            case DxilParamInputQual::In:
              PS.MarkAsLoaded();
              break;
            }
          } else {
            // Do not replace struct arg.
            // Mark stored and loaded to disable replace.
            PS.MarkAsStored();
            PS.MarkAsLoaded();
          }
        }
      }
    }
  }
}
}

namespace hlsl {
namespace hlutil {

void PointerStatus::analyze(DxilTypeSystem &typeSys, bool bStructElt) {
  analyzePointer(Ptr, *this, typeSys, bStructElt, bLoadStoreOnly);
}

PointerStatus::PointerStatus(llvm::Value *ptr, unsigned size, bool bLdStOnly)
    : storedType(StoredType::NotStored), loadedType(LoadedType::NotLoaded),
      StoredOnceValue(nullptr), StoringMemcpy(nullptr), LoadingMemcpy(nullptr),
      AccessingFunction(nullptr), HasMultipleAccessingFunctions(false),
      Size(size), Ptr(ptr), bLoadStoreOnly(bLdStOnly) {}

void PointerStatus::MarkAsStored() {
  storedType = StoredType::Stored;
  StoredOnceValue = nullptr;
}
void PointerStatus::MarkAsLoaded() { loadedType = LoadedType::Loaded; }
bool PointerStatus::HasStored() {
  return storedType != StoredType::NotStored &&
         storedType != StoredType::InitializerStored;
}
bool PointerStatus::HasLoaded() { return loadedType != LoadedType::NotLoaded; }

} // namespace hlutil
} // namespace hlsl