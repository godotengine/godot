///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// HLLowerUDT.cpp                                                            //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Lower user defined type used directly by certain intrinsic operations.    //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/HLSL/HLLowerUDT.h"
#include "dxc/Support/Global.h"
#include "dxc/DXIL/DxilConstants.h"
#include "dxc/HLSL/HLModule.h"
#include "dxc/HLSL/HLOperations.h"
#include "dxc/DXIL/DxilTypeSystem.h"
#include "dxc/HLSL/HLMatrixLowerHelper.h"
#include "dxc/HLSL/HLMatrixType.h"
#include "dxc/HlslIntrinsicOp.h"
#include "dxc/DXIL/DxilUtil.h"

#include "HLMatrixSubscriptUseReplacer.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

using namespace llvm;
using namespace hlsl;

// Lowered UDT is the same layout, but with vectors and matrices translated to
// arrays.
// Returns nullptr for failure due to embedded HLSL object type.
StructType *hlsl::GetLoweredUDT(StructType *structTy, DxilTypeSystem *pTypeSys) {
  bool changed = false;
  SmallVector<Type*, 8> NewElTys(structTy->getNumContainedTypes());

  for (unsigned iField = 0; iField < NewElTys.size(); ++iField) {
    Type *FieldTy = structTy->getContainedType(iField);

    // Default to original type
    NewElTys[iField] = FieldTy;

    // Unwrap arrays:
    SmallVector<unsigned, 4> OuterToInnerLengths;
    Type *EltTy = dxilutil::StripArrayTypes(FieldTy, &OuterToInnerLengths);
    Type *NewTy = EltTy;

    // Lower element if necessary
    if (FixedVectorType *VT = dyn_cast<FixedVectorType>(EltTy)) {
      NewTy = ArrayType::get(VT->getElementType(),
                             VT->getNumElements());
    } else if (HLMatrixType Mat = HLMatrixType::dyn_cast(EltTy)) {
      NewTy = ArrayType::get(Mat.getElementType(/*MemRepr*/true),
                             Mat.getNumElements());
    } else if (dxilutil::IsHLSLObjectType(EltTy) ||
               dxilutil::IsHLSLRayQueryType(EltTy)) {
      // We cannot lower a structure with an embedded object type
      return nullptr;
    } else if (StructType *ST = dyn_cast<StructType>(EltTy)) {
      NewTy = GetLoweredUDT(ST);
      if (nullptr == NewTy)
        return nullptr; // Propagate failure back to root
    } else if (EltTy->isIntegerTy(1)) {
      // Must translate bool to mem type
      EltTy = IntegerType::get(EltTy->getContext(), 32);
    }

    // if unchanged, skip field
    if (NewTy == EltTy)
      continue;

    // Rewrap Arrays:
    for (auto itLen = OuterToInnerLengths.rbegin(),
                  E = OuterToInnerLengths.rend();
         itLen != E; ++itLen) {
      NewTy = ArrayType::get(NewTy, *itLen);
    }

    // Update field, and set changed
    NewElTys[iField] = NewTy;
    changed = true;
  }

  if (changed) {
    StructType *newStructTy = StructType::create(
      structTy->getContext(), NewElTys, structTy->getStructName());
    if (DxilStructAnnotation *pSA = pTypeSys ?
          pTypeSys->GetStructAnnotation(structTy) : nullptr) {
      if (!pTypeSys->GetStructAnnotation(newStructTy)) {
        DxilStructAnnotation &NewSA = *pTypeSys->AddStructAnnotation(newStructTy);
        for (unsigned iField = 0; iField < NewElTys.size(); ++iField) {
          NewSA.GetFieldAnnotation(iField) = pSA->GetFieldAnnotation(iField);
        }
      }
    }
    return newStructTy;
  }

  return structTy;
}

Constant *hlsl::TranslateInitForLoweredUDT(
    Constant *Init, Type *NewTy,
    // We need orientation for matrix fields
    DxilTypeSystem *pTypeSys,
    MatrixOrientation matOrientation) {

  // handle undef and zero init
  if (isa<UndefValue>(Init))
    return UndefValue::get(NewTy);
  else if (Init->getType()->isAggregateType() && Init->isZeroValue())
    return ConstantAggregateZero::get(NewTy);

  // unchanged
  Type *Ty = Init->getType();
  if (Ty == NewTy)
    return Init;

  SmallVector<Constant*, 16> values;
  if (Ty->isArrayTy()) {
    values.reserve(Ty->getArrayNumElements());
    ConstantArray *CA = cast<ConstantArray>(Init);
    for (unsigned i = 0; i < Ty->getArrayNumElements(); ++i)
      values.emplace_back(
        TranslateInitForLoweredUDT(
          CA->getAggregateElement(i),
          NewTy->getArrayElementType(),
          pTypeSys, matOrientation));
    return ConstantArray::get(cast<ArrayType>(NewTy), values);
  } else if (FixedVectorType *VT = dyn_cast<FixedVectorType>(Ty)) {
    values.reserve(VT->getNumElements());
    ConstantVector *CV = cast<ConstantVector>(Init);
    for (unsigned i = 0; i < VT->getNumElements(); ++i)
      values.emplace_back(CV->getAggregateElement(i));
    return ConstantArray::get(cast<ArrayType>(NewTy), values);
  } else if (HLMatrixType Mat = HLMatrixType::dyn_cast(Ty)) {
    values.reserve(Mat.getNumElements());
    ConstantArray *MatArray = cast<ConstantArray>(
      cast<ConstantStruct>(Init)->getOperand(0));
    for (unsigned row = 0; row < Mat.getNumRows(); ++row) {
      ConstantVector *RowVector = cast<ConstantVector>(
        MatArray->getOperand(row));
      for (unsigned col = 0; col < Mat.getNumColumns(); ++col) {
        unsigned index = matOrientation == MatrixOrientation::ColumnMajor ?
          Mat.getColumnMajorIndex(row, col) : Mat.getRowMajorIndex(row, col);
        values[index] = RowVector->getOperand(col);
      }
    }
  } else if (StructType *ST = dyn_cast<StructType>(Ty)) {
    DxilStructAnnotation *pStructAnnotation =
      pTypeSys ? pTypeSys->GetStructAnnotation(ST) : nullptr;
    values.reserve(ST->getNumContainedTypes());
    ConstantStruct *CS = cast<ConstantStruct>(Init);
    for (unsigned i = 0; i < ST->getStructNumElements(); ++i) {
      MatrixOrientation matFieldOrientation = matOrientation;
      if (pStructAnnotation) {
        DxilFieldAnnotation &FA = pStructAnnotation->GetFieldAnnotation(i);
        if (FA.HasMatrixAnnotation()) {
          matFieldOrientation = FA.GetMatrixAnnotation().Orientation;
        }
      }
      values.emplace_back(
        TranslateInitForLoweredUDT(
          cast<Constant>(CS->getAggregateElement(i)),
          NewTy->getStructElementType(i),
          pTypeSys, matFieldOrientation));
    }
    return ConstantStruct::get(cast<StructType>(NewTy), values);
  }
  return Init;
}

void hlsl::ReplaceUsesForLoweredUDT(Value *V, Value *NewV) {
  Type *Ty = V->getType();
  Type *NewTy = NewV->getType();

  if (Ty == NewTy) {
    V->replaceAllUsesWith(NewV);
    if (Instruction *I = dyn_cast<Instruction>(V))
      I->dropAllReferences();
    if (Constant *CV = dyn_cast<Constant>(V))
      CV->removeDeadConstantUsers();
    return;
  }

  if (Ty->isPointerTy())
    Ty = Ty->getPointerElementType();
  if (NewTy->isPointerTy())
    NewTy = NewTy->getPointerElementType();

  while (!V->use_empty()) {
    Use &use = *V->use_begin();
    User *user = use.getUser();
    if (Instruction *I = dyn_cast<Instruction>(user)) {
      use.set(UndefValue::get(I->getType()));
    }

    if (LoadInst *LI = dyn_cast<LoadInst>(user)) {
      // Load for non-matching type should only be vector
      FixedVectorType *VT = dyn_cast<FixedVectorType>(Ty);
      DXASSERT(VT && NewTy->isArrayTy() &&
        VT->getNumElements() == NewTy->getArrayNumElements(),
        "unexpected load of non-matching type");
      IRBuilder<> Builder(LI);
      Value *result = UndefValue::get(Ty);
      for (unsigned i = 0; i < VT->getNumElements(); ++i) {
        Value *GEP = Builder.CreateInBoundsGEP(NewV,
          {Builder.getInt32(0), Builder.getInt32(i)});
        Value *El = Builder.CreateLoad(GEP);
        result = Builder.CreateInsertElement(result, El, i);
      }
      LI->replaceAllUsesWith(result);
      LI->eraseFromParent();

    } else if (StoreInst *SI = dyn_cast<StoreInst>(user)) {
      // Store for non-matching type should only be vector
      FixedVectorType *VT = dyn_cast<FixedVectorType>(Ty);
      DXASSERT(VT && NewTy->isArrayTy() &&
        VT->getNumElements() == NewTy->getArrayNumElements(),
        "unexpected load of non-matching type");
      IRBuilder<> Builder(SI);
      for (unsigned i = 0; i < VT->getNumElements(); ++i) {
        Value *EE = Builder.CreateExtractElement(SI->getValueOperand(), i);
        Value *GEP = Builder.CreateInBoundsGEP(
          NewV, {Builder.getInt32(0), Builder.getInt32(i)});
        Builder.CreateStore(EE, GEP);
      }
      SI->eraseFromParent();

    } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(user)) {
      // Non-constant GEP
      IRBuilder<> Builder(GEP);
      SmallVector<Value*, 4> idxList(GEP->idx_begin(), GEP->idx_end());
      Value *NewGEP = Builder.CreateGEP(NewV, idxList);
      ReplaceUsesForLoweredUDT(GEP, NewGEP);
      GEP->eraseFromParent();

    } else if (GEPOperator *GEP = dyn_cast<GEPOperator>(user)) {
      // Has to be constant GEP, NewV better be constant
      SmallVector<Value*, 4> idxList(GEP->idx_begin(), GEP->idx_end());
      Constant *NewGEP = ConstantExpr::getGetElementPtr(
        nullptr, cast<Constant>(NewV), idxList, true);
      ReplaceUsesForLoweredUDT(GEP, NewGEP);

    } else if (AddrSpaceCastInst *AC = dyn_cast<AddrSpaceCastInst>(user)) {
      // Address space cast
      IRBuilder<> Builder(AC);
      unsigned AddrSpace = AC->getType()->getPointerAddressSpace();
      Value *NewAC = Builder.CreateAddrSpaceCast(
          NewV, PointerType::get(NewTy, AddrSpace));
      ReplaceUsesForLoweredUDT(user, NewAC);
      AC->eraseFromParent();

    } else if (BitCastInst *BC = dyn_cast<BitCastInst>(user)) {
      IRBuilder<> Builder(BC);
      if (BC->getType()->getPointerElementType() == NewTy) {
        // if alreday bitcast to new type, just replace the bitcast
        // with the new value (already translated user function)
        BC->replaceAllUsesWith(NewV);
        BC->eraseFromParent();
      } else {
        // Could be i8 for memcpy?
        // Replace bitcast argument with new value
        use.set(NewV);
      }

    } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(user)) {
      // Constant AddrSpaceCast, or BitCast
      if (CE->getOpcode() == Instruction::AddrSpaceCast) {
        unsigned AddrSpace = CE->getType()->getPointerAddressSpace();
        ReplaceUsesForLoweredUDT(user,
          ConstantExpr::getAddrSpaceCast(cast<Constant>(NewV),
            PointerType::get(NewTy, AddrSpace)));
      } else if (CE->getOpcode() == Instruction::BitCast) {
        if (CE->getType()->getPointerElementType() == NewTy) {
          // if alreday bitcast to new type, just replace the bitcast
          // with the new value
          CE->replaceAllUsesWith(NewV);
        } else {
          // Could be i8 for memcpy?
          // Replace bitcast argument with new value
          CE->replaceAllUsesWith(
              ConstantExpr::getBitCast(cast<Constant>(NewV), CE->getType()));
        }
      } else {
        DXASSERT(0, "unhandled constant expr for lowered UDT");
        // better than infinite loop on release
        CE->replaceAllUsesWith(UndefValue::get(CE->getType()));
      }

    } else if (CallInst *CI = dyn_cast<CallInst>(user)) {
      // Lower some matrix intrinsics that access pointers early, and
      // cast arguments for user functions or special UDT intrinsics
      // for later translation.
      Function *F = CI->getCalledFunction();
      HLOpcodeGroup group = GetHLOpcodeGroupByName(F);
      HLMatrixType Mat = HLMatrixType::dyn_cast(Ty);
      bool bColMajor = false;

      switch (group) {
      case HLOpcodeGroup::HLMatLoadStore: {
        DXASSERT(Mat, "otherwise, matrix operation on non-matrix value");
        IRBuilder<> Builder(CI);
        HLMatLoadStoreOpcode opcode =
            static_cast<HLMatLoadStoreOpcode>(hlsl::GetHLOpcode(CI));
        switch (opcode) {
        case HLMatLoadStoreOpcode::ColMatLoad:
          bColMajor = true;
          __fallthrough;
        case HLMatLoadStoreOpcode::RowMatLoad: {
          Value *val = UndefValue::get(
            VectorType::get(NewTy->getArrayElementType(),
                            NewTy->getArrayNumElements()));
          for (unsigned i = 0; i < NewTy->getArrayNumElements(); ++i) {
            Value *GEP = Builder.CreateGEP(NewV,
              {Builder.getInt32(0), Builder.getInt32(i)});
            Value *elt = Builder.CreateLoad(GEP);
            val = Builder.CreateInsertElement(val, elt, i);
          }
          if (bColMajor) {
            // transpose matrix to match expected value orientation for
            // default cast to matrix type
            SmallVector<int, 16> ShuffleIndices;
            for (unsigned RowIdx = 0; RowIdx < Mat.getNumRows(); ++RowIdx)
              for (unsigned ColIdx = 0; ColIdx < Mat.getNumColumns(); ++ColIdx)
                ShuffleIndices.emplace_back(
                  static_cast<int>(Mat.getColumnMajorIndex(RowIdx, ColIdx)));
            val = Builder.CreateShuffleVector(val, val, ShuffleIndices);
          }
          // lower mem to reg type
          val = Mat.emitLoweredMemToReg(val, Builder);
          // cast vector back to matrix value (DefaultCast expects row major)
          unsigned newOpcode = (unsigned)HLCastOpcode::DefaultCast;
          val = callHLFunction(*F->getParent(), HLOpcodeGroup::HLCast, newOpcode,
                               Ty, { Builder.getInt32(newOpcode), val }, Builder);
          if (bColMajor) {
            // emit cast row to col to match original result
            newOpcode = (unsigned)HLCastOpcode::RowMatrixToColMatrix;
            val = callHLFunction(*F->getParent(), HLOpcodeGroup::HLCast, newOpcode,
              Ty, { Builder.getInt32(newOpcode), val }, Builder);
          }
          // replace use of HLMatLoadStore with loaded vector
          CI->replaceAllUsesWith(val);
        } break;
        case HLMatLoadStoreOpcode::ColMatStore:
          bColMajor = true;
          __fallthrough;
        case HLMatLoadStoreOpcode::RowMatStore: {
          // HLCast matrix value to vector
          unsigned newOpcode = (unsigned)(bColMajor ?
              HLCastOpcode::ColMatrixToVecCast :
              HLCastOpcode::RowMatrixToVecCast);
          Value *val = callHLFunction(*F->getParent(),
            HLOpcodeGroup::HLCast, newOpcode,
            Mat.getLoweredVectorType(false),
            { Builder.getInt32(newOpcode),
              CI->getArgOperand(HLOperandIndex::kMatStoreValOpIdx) },
            Builder);
          // lower reg to mem type
          val = Mat.emitLoweredRegToMem(val, Builder);
          for (unsigned i = 0; i < NewTy->getArrayNumElements(); ++i) {
            Value *elt = Builder.CreateExtractElement(val, i);
            Value *GEP = Builder.CreateGEP(NewV,
              {Builder.getInt32(0), Builder.getInt32(i)});
            Builder.CreateStore(elt, GEP);
          }
        } break;
        default:
          DXASSERT(0, "invalid opcode");
        }
        CI->eraseFromParent();
      } break;

      case HLOpcodeGroup::HLSubscript: {
        SmallVector<Value*, 4> ElemIndices;
        HLSubscriptOpcode opcode =
            static_cast<HLSubscriptOpcode>(hlsl::GetHLOpcode(CI));
        switch (opcode) {
        case HLSubscriptOpcode::VectorSubscript:
          DXASSERT(0, "not handled yet");
          break;
        case HLSubscriptOpcode::ColMatElement:
          bColMajor = true;
          __fallthrough;
        case HLSubscriptOpcode::RowMatElement: {
          ConstantDataSequential *cIdx = cast<ConstantDataSequential>(
            CI->getArgOperand(HLOperandIndex::kMatSubscriptSubOpIdx));
          for (unsigned i = 0; i < cIdx->getNumElements(); ++i) {
            ElemIndices.push_back(cIdx->getElementAsConstant(i));
          }
        } break;
        case HLSubscriptOpcode::ColMatSubscript:
          bColMajor = true;
          __fallthrough;
        case HLSubscriptOpcode::RowMatSubscript: {
          for (unsigned Idx = HLOperandIndex::kMatSubscriptSubOpIdx; Idx < CI->getNumArgOperands(); ++Idx) {
            ElemIndices.emplace_back(CI->getArgOperand(Idx));
          }
        } break;
        default:
          DXASSERT(0, "invalid opcode");
        }

        std::vector<Instruction*> DeadInsts;
        HLMatrixSubscriptUseReplacer UseReplacer(
          CI, NewV, /*TempLoweredMatrix*/nullptr, ElemIndices, /*AllowLoweredPtrGEPs*/true, DeadInsts);
        DXASSERT(CI->use_empty(),
                 "Expected all matrix subscript uses to have been replaced.");
        CI->eraseFromParent();
        while (!DeadInsts.empty()) {
          DeadInsts.back()->eraseFromParent();
          DeadInsts.pop_back();
        }
      } break;

      //case HLOpcodeGroup::NotHL:  // TODO: Support lib functions
      case HLOpcodeGroup::HLIntrinsic: {
        // Just bitcast for now
        IRBuilder<> Builder(CI);
        use.set(Builder.CreateBitCast(NewV, V->getType()));
        continue;
      } break;

      default:
        DXASSERT(0, "invalid opcode");
        // Replace user with undef to prevent infinite loop on unhandled case.
        user->replaceAllUsesWith(UndefValue::get(user->getType()));
      }
    } else {
      // What else?
      DXASSERT(false, "case not handled.");
      // Replace user with undef to prevent infinite loop on unhandled case.
      user->replaceAllUsesWith(UndefValue::get(user->getType()));
    }
    // Clean up dead constant users to prevent infinite loop
    if (Constant *CV = dyn_cast<Constant>(V))
      CV->removeDeadConstantUsers();
  }
}
