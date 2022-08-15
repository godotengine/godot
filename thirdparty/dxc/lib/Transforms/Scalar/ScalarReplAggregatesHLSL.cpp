//===- ScalarReplAggregatesHLSL.cpp - Scalar Replacement of Aggregates ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
// Based on ScalarReplAggregates.cpp. The difference is HLSL version will keep
// array so it can break up all structure.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "dxc/HLSL/HLOperations.h"
#include "dxc/DXIL/DxilConstants.h"
#include "dxc/HLSL/HLModule.h"
#include "dxc/DXIL/DxilUtil.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/HlslIntrinsicOp.h"
#include "dxc/DXIL/DxilTypeSystem.h"
#include "dxc/HLSL/HLMatrixLowerHelper.h"
#include "dxc/HLSL/HLMatrixType.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/HLSL/HLLowerUDT.h"
#include "dxc/HLSL/HLUtil.h"
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <queue>

using namespace llvm;
using namespace hlsl;
#define DEBUG_TYPE "scalarreplhlsl"

STATISTIC(NumReplaced, "Number of allocas broken up");

namespace {

class SROA_Helper {
public:
  // Split V into AllocaInsts with Builder and save the new AllocaInsts into Elts.
  // Then do SROA on V.
  static bool DoScalarReplacement(Value *V, std::vector<Value *> &Elts,
                                  Type *&BrokenUpTy, uint64_t &NumInstances,
                                  IRBuilder<> &Builder, bool bFlatVector,
                                  bool hasPrecise, DxilTypeSystem &typeSys,
                                  const DataLayout &DL,
                                  SmallVector<Value *, 32> &DeadInsts,
                                  DominatorTree *DT);

  static bool DoScalarReplacement(GlobalVariable *GV, std::vector<Value *> &Elts,
                                  IRBuilder<> &Builder, bool bFlatVector,
                                  bool hasPrecise, DxilTypeSystem &typeSys,
                                  const DataLayout &DL,
                                  SmallVector<Value *, 32> &DeadInsts,
                                  DominatorTree *DT);
  static unsigned GetEltAlign(unsigned ValueAlign, const DataLayout &DL,
                              Type *EltTy, unsigned Offset);
  // Lower memcpy related to V.
  static bool LowerMemcpy(Value *V, DxilFieldAnnotation *annotation,
                          DxilTypeSystem &typeSys, const DataLayout &DL,
                          DominatorTree *DT, bool bAllowReplace);
  static void MarkEmptyStructUsers(Value *V,
                                   SmallVector<Value *, 32> &DeadInsts);
  static bool IsEmptyStructType(Type *Ty, DxilTypeSystem &typeSys);
private:
  SROA_Helper(Value *V, ArrayRef<Value *> Elts,
              SmallVector<Value *, 32> &DeadInsts, DxilTypeSystem &ts,
              const DataLayout &dl, DominatorTree *dt)
    : OldVal(V), NewElts(Elts), DeadInsts(DeadInsts), typeSys(ts), DL(dl), DT(dt) {}
  void RewriteForScalarRepl(Value *V, IRBuilder<> &Builder);

private:
  // Must be a pointer type val.
  Value * OldVal;
  // Flattened elements for OldVal.
  ArrayRef<Value*> NewElts;
  SmallVector<Value *, 32> &DeadInsts;
  DxilTypeSystem  &typeSys;
  const DataLayout &DL;
  DominatorTree *DT;

  void RewriteForConstExpr(ConstantExpr *user, IRBuilder<> &Builder);
  void RewriteForGEP(GEPOperator *GEP, IRBuilder<> &Builder);
  void RewriteForAddrSpaceCast(Value *user, IRBuilder<> &Builder);
  void RewriteForLoad(LoadInst *loadInst);
  void RewriteForStore(StoreInst *storeInst);
  void RewriteMemIntrin(MemIntrinsic *MI, Value *OldV);
  void RewriteCall(CallInst *CI);
  void RewriteBitCast(BitCastInst *BCI);
  void RewriteCallArg(CallInst *CI, unsigned ArgIdx, bool bIn, bool bOut);
};

}

static unsigned getNestedLevelInStruct(const Type *ty) {
  unsigned lvl = 0;
  while (ty->isStructTy()) {
    if (ty->getStructNumElements() != 1)
      break;
    ty = ty->getStructElementType(0);
    lvl++;
  }
  return lvl;
}

// After SROA'ing a given value into a series of elements,
// creates the debug info for the storage of the individual elements.
static void addDebugInfoForElements(Value *ParentVal,
    Type *BrokenUpTy, uint64_t NumInstances,
    ArrayRef<Value*> Elems, const DataLayout &DatLayout,
    DIBuilder *DbgBuilder) {

  // Extract the data we need from the parent value,
  // depending on whether it is an alloca, argument or global variable.

  if (isa<GlobalVariable>(ParentVal)) {
    llvm_unreachable("Not implemented: sroa debug info propagation for global vars.");
  }
  else {
    Type *ParentTy = nullptr;

    if (AllocaInst *ParentAlloca = dyn_cast<AllocaInst>(ParentVal))
      ParentTy = ParentAlloca->getAllocatedType();
    else
      ParentTy = cast<Argument>(ParentVal)->getType();

    SmallVector<DbgDeclareInst *, 4> Declares;
    llvm::FindAllocaDbgDeclare(ParentVal, Declares);

    for (DbgDeclareInst *ParentDbgDeclare : Declares) {
      unsigned ParentBitPieceOffset = 0;
      DIVariable *ParentDbgVariable = nullptr;
      DIExpression *ParentDbgExpr = nullptr;
      DILocation *ParentDbgLocation = nullptr;
      Instruction *DbgDeclareInsertPt = nullptr;

      std::vector<DxilDIArrayDim> DIArrayDims;
      // Get the bit piece offset
      if ((ParentDbgExpr = ParentDbgDeclare->getExpression())) {
        if (ParentDbgExpr->isBitPiece()) {
          ParentBitPieceOffset = ParentDbgExpr->getBitPieceOffset();
        }
      }
      
      ParentDbgVariable = ParentDbgDeclare->getVariable();
      ParentDbgLocation = ParentDbgDeclare->getDebugLoc();
      DbgDeclareInsertPt = ParentDbgDeclare;

      // Read the extra layout metadata, if any
      unsigned ParentBitPieceOffsetFromMD = 0;
      if (DxilMDHelper::GetVariableDebugLayout(ParentDbgDeclare, ParentBitPieceOffsetFromMD, DIArrayDims)) {
        // The offset is redundant for local variables and only necessary for global variables.
        DXASSERT(ParentBitPieceOffsetFromMD == ParentBitPieceOffset,
          "Bit piece offset mismatch between llvm.dbg.declare and DXIL metadata.");
      }

      // If the type that was broken up is nested in arrays,
      // then each element will also be an array,
      // but the continuity between successive elements of the original aggregate
      // will have been broken, such that we must store the stride to rebuild it.
      // For example: [2 x {i32, float}] => [2 x i32], [2 x float], each with stride 64 bits
      if (NumInstances > 1 && Elems.size() > 1) {
        // Existing dimensions already account for part of the stride
        uint64_t NewDimNumElements = NumInstances;
        for (const DxilDIArrayDim& ArrayDim : DIArrayDims) {
          DXASSERT(NewDimNumElements % ArrayDim.NumElements == 0,
            "Debug array stride is inconsistent with the number of elements.");
          NewDimNumElements /= ArrayDim.NumElements;
        }

        // Add a stride dimension
        DxilDIArrayDim NewDIArrayDim = {};
        NewDIArrayDim.StrideInBits = (unsigned)DatLayout.getTypeAllocSizeInBits(BrokenUpTy);
        NewDIArrayDim.NumElements = (unsigned)NewDimNumElements;
        DIArrayDims.emplace_back(NewDIArrayDim);
      }
      else {
        DIArrayDims.clear();
      }

      // Create the debug info for each element
      for (unsigned ElemIdx = 0; ElemIdx < Elems.size(); ++ElemIdx) {
        // Figure out the offset of the element in the broken up type
        unsigned ElemBitPieceOffset = ParentBitPieceOffset;
        if (StructType *ParentStructTy = dyn_cast<StructType>(BrokenUpTy)) {
          DXASSERT_NOMSG(Elems.size() == ParentStructTy->getNumElements());
          ElemBitPieceOffset += (unsigned)DatLayout.getStructLayout(ParentStructTy)->getElementOffsetInBits(ElemIdx);
        }
        else if (VectorType *ParentVecTy = dyn_cast<VectorType>(BrokenUpTy)) {
          DXASSERT_NOMSG(Elems.size() == ParentVecTy->getNumElements());
          ElemBitPieceOffset += (unsigned)DatLayout.getTypeStoreSizeInBits(ParentVecTy->getElementType()) * ElemIdx;
        }
        else if (ArrayType *ParentArrayTy = dyn_cast<ArrayType>(BrokenUpTy)) {
          DXASSERT_NOMSG(Elems.size() == ParentArrayTy->getNumElements());
          ElemBitPieceOffset += (unsigned)DatLayout.getTypeStoreSizeInBits(ParentArrayTy->getElementType()) * ElemIdx;
        }

        // The bit_piece can only represent the leading contiguous bytes.
        // If strides are involved, we'll need additional metadata.
        Type *ElemTy = Elems[ElemIdx]->getType()->getPointerElementType();
        unsigned ElemBitPieceSize = (unsigned)DatLayout.getTypeStoreSizeInBits(ElemTy);
        for (const DxilDIArrayDim& ArrayDim : DIArrayDims)
          ElemBitPieceSize /= ArrayDim.NumElements;

        if (AllocaInst *ElemAlloca = dyn_cast<AllocaInst>(Elems[ElemIdx])) {
          // Local variables get an @llvm.dbg.declare plus optional metadata for layout stride information.
          DIExpression *ElemDbgExpr = nullptr;
          if (ElemBitPieceOffset == 0 && DatLayout.getTypeAllocSizeInBits(ParentTy) == ElemBitPieceSize) {
            ElemDbgExpr = DbgBuilder->createExpression();
          }
          else {
            ElemDbgExpr = DbgBuilder->createBitPieceExpression(ElemBitPieceOffset, ElemBitPieceSize);
          }

          DXASSERT_NOMSG(DbgBuilder != nullptr);
          DbgDeclareInst *EltDDI = cast<DbgDeclareInst>(DbgBuilder->insertDeclare(
            ElemAlloca, cast<DILocalVariable>(ParentDbgVariable), ElemDbgExpr, ParentDbgLocation, DbgDeclareInsertPt));

          if (!DIArrayDims.empty()) DxilMDHelper::SetVariableDebugLayout(EltDDI, ElemBitPieceOffset, DIArrayDims);
        }
        else {
          llvm_unreachable("Non-AllocaInst SROA'd elements.");
        }
      }
    }
  }
}

/// Returns first GEP index that indexes a struct member, or 0 otherwise.
/// Ignores initial ptr index.
static unsigned FindFirstStructMemberIdxInGEP(GEPOperator *GEP) {
  StructType *ST = dyn_cast<StructType>(
    GEP->getPointerOperandType()->getPointerElementType());
  int index = 1;
  for (auto it = gep_type_begin(GEP), E = gep_type_end(GEP); it != E;
       ++it, ++index) {
    if (ST) {
      DXASSERT(!HLMatrixType::isa(ST) && !dxilutil::IsHLSLObjectType(ST),
               "otherwise, indexing into hlsl object");
      return index;
    }
    ST = dyn_cast<StructType>(it->getPointerElementType());
  }
  return 0;
}

/// Return true when ptr should not be SROA'd or copied, but used directly
/// by a function in its lowered form.  Also collect uses for translation.
/// What is meant by directly here:
///   Possibly accessed through GEP array index or address space cast, but
///   not under another struct member (always allow SROA of outer struct).
typedef SmallMapVector<CallInst*, unsigned, 4> FunctionUseMap;
static unsigned IsPtrUsedByLoweredFn(
    Value *V, FunctionUseMap &CollectedUses) {
  bool bFound = false;
  for (Use &U : V->uses()) {
    User *user = U.getUser();

    if (CallInst *CI = dyn_cast<CallInst>(user)) {
      unsigned foundIdx = (unsigned)-1;
      Function *F = CI->getCalledFunction();
      Type *Ty = V->getType();
      if (F->isDeclaration() && !F->isIntrinsic() &&
          Ty->isPointerTy()) {
        HLOpcodeGroup group = hlsl::GetHLOpcodeGroupByName(F);
        if (group == HLOpcodeGroup::HLIntrinsic) {
          unsigned opIdx = U.getOperandNo();
          switch ((IntrinsicOp)hlsl::GetHLOpcode(CI)) {
            // TODO: Lower these as well, along with function parameter types
            //case IntrinsicOp::IOP_TraceRay:
            //  if (opIdx != HLOperandIndex::kTraceRayPayLoadOpIdx)
            //    continue;
            //  break;
            //case IntrinsicOp::IOP_ReportHit:
            //  if (opIdx != HLOperandIndex::kReportIntersectionAttributeOpIdx)
            //    continue;
            //  break;
            //case IntrinsicOp::IOP_CallShader:
            //  if (opIdx != HLOperandIndex::kCallShaderPayloadOpIdx)
            //    continue;
            //  break;
            case IntrinsicOp::IOP_DispatchMesh:
              if (opIdx != HLOperandIndex::kDispatchMeshOpPayload)
                continue;
              break;
            default:
              continue;
          }
          foundIdx = opIdx;

        // TODO: Lower these as well, along with function parameter types
        //} else if (group == HLOpcodeGroup::NotHL) {
        //  foundIdx = U.getOperandNo();
        }
      }
      if (foundIdx != (unsigned)-1) {
        bFound = true;
        auto insRes = CollectedUses.insert(std::make_pair(CI, foundIdx));
        DXASSERT_LOCALVAR(insRes, insRes.second,
            "otherwise, multiple uses in single call");
      }

    } else if (GEPOperator *GEP = dyn_cast<GEPOperator>(user)) {
      // Not what we are looking for if GEP result is not [array of] struct.
      // If use is under struct member, we can still SROA the outer struct.
      if (!dxilutil::StripArrayTypes(GEP->getType()->getPointerElementType())
            ->isStructTy() ||
          FindFirstStructMemberIdxInGEP(GEP))
        continue;
      if (IsPtrUsedByLoweredFn(user, CollectedUses))
        bFound = true;

    } else if (AddrSpaceCastInst *AC = dyn_cast<AddrSpaceCastInst>(user)) {
      if (IsPtrUsedByLoweredFn(user, CollectedUses))
        bFound = true;

    } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(user)) {
      unsigned opcode = CE->getOpcode();
      if (opcode == Instruction::AddrSpaceCast)
        if (IsPtrUsedByLoweredFn(user, CollectedUses))
          bFound = true;
    }
  }
  return bFound;
}

/// Rewrite call to natively use an argument with addrspace cast/bitcast
static CallInst *RewriteIntrinsicCallForCastedArg(CallInst *CI, unsigned argIdx) {
  Function *F = CI->getCalledFunction();
  HLOpcodeGroup group = GetHLOpcodeGroupByName(F);
  DXASSERT_NOMSG(group == HLOpcodeGroup::HLIntrinsic);
  unsigned opcode = GetHLOpcode(CI);
  SmallVector<Type *, 8> newArgTypes(CI->getFunctionType()->param_begin(),
                                     CI->getFunctionType()->param_end());
  SmallVector<Value *, 8> newArgs(CI->arg_operands());

  Value *newArg = CI->getOperand(argIdx)->stripPointerCasts();
  newArgTypes[argIdx] = newArg->getType();
  newArgs[argIdx] = newArg;

  FunctionType *newFuncTy = FunctionType::get(CI->getType(), newArgTypes, false);
  Function *newF = GetOrCreateHLFunction(*F->getParent(), newFuncTy, group, opcode,
                                         F->getAttributes().getFnAttributes());

  IRBuilder<> Builder(CI);
  return Builder.CreateCall(newF, newArgs);
}

/// Translate pointer for cases where intrinsics use UDT pointers directly
/// Return existing or new ptr if needs preserving,
/// otherwise nullptr to proceed with existing checks and SROA.
static Value *TranslatePtrIfUsedByLoweredFn(
    Value *Ptr, DxilTypeSystem &TypeSys) {
  if (!Ptr->getType()->isPointerTy())
    return nullptr;
  Type *Ty = Ptr->getType()->getPointerElementType();
  SmallVector<unsigned, 4> outerToInnerLengths;
  Ty = dxilutil::StripArrayTypes(Ty, &outerToInnerLengths);
  if (!Ty->isStructTy())
    return nullptr;
  if (HLMatrixType::isa(Ty) || dxilutil::IsHLSLObjectType(Ty))
    return nullptr;
  unsigned AddrSpace = Ptr->getType()->getPointerAddressSpace();
  FunctionUseMap FunctionUses;
  if (!IsPtrUsedByLoweredFn(Ptr, FunctionUses))
    return nullptr;
  // Translate vectors to arrays in type, but don't SROA
  Type *NewTy = GetLoweredUDT(cast<StructType>(Ty), &TypeSys);

  // No work to do here, but prevent SROA.
  if (Ty == NewTy && AddrSpace != DXIL::kTGSMAddrSpace)
    return Ptr;

  // If type changed, replace value, otherwise casting may still
  // require a rewrite of the calls.
  Value *NewPtr = Ptr;
  if (Ty != NewTy) {
    NewTy = dxilutil::WrapInArrayTypes(NewTy, outerToInnerLengths);
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Ptr)) {
      Module &M = *GV->getParent();
      // Rewrite init expression for arrays instead of vectors
      Constant *Init = GV->hasInitializer() ?
        GV->getInitializer() : UndefValue::get(Ptr->getType());
      Constant *NewInit = TranslateInitForLoweredUDT(
        Init, NewTy, &TypeSys);
      // Replace with new GV, and rewrite vector load/store users
      GlobalVariable *NewGV = new GlobalVariable(
          M, NewTy, GV->isConstant(), GV->getLinkage(),
          NewInit, GV->getName(), /*InsertBefore*/ GV,
          GV->getThreadLocalMode(), AddrSpace);
      NewPtr = NewGV;
    } else if (AllocaInst *AI = dyn_cast<AllocaInst>(Ptr)) {
      IRBuilder<> Builder(AI);
      AllocaInst * NewAI = Builder.CreateAlloca(NewTy, nullptr, AI->getName());
      NewPtr = NewAI;
    } else {
      DXASSERT(false, "Ptr must be global or alloca");
    }
    // This will rewrite vector load/store users
    // and insert bitcasts for CallInst users
    ReplaceUsesForLoweredUDT(Ptr, NewPtr);
  }

  // Rewrite the HLIntrinsic calls
  for (auto it : FunctionUses) {
    CallInst *CI = it.first;
    HLOpcodeGroup group = GetHLOpcodeGroupByName(CI->getCalledFunction());
    if (group == HLOpcodeGroup::NotHL)
      continue;
    CallInst *newCI = RewriteIntrinsicCallForCastedArg(CI, it.second);
    CI->replaceAllUsesWith(newCI);
    CI->eraseFromParent();
  }

  return NewPtr;
}

/// isHomogeneousAggregate - Check if type T is a struct or array containing
/// elements of the same type (which is always true for arrays).  If so,
/// return true with NumElts and EltTy set to the number of elements and the
/// element type, respectively.
static bool isHomogeneousAggregate(Type *T, unsigned &NumElts, Type *&EltTy) {
  if (ArrayType *AT = dyn_cast<ArrayType>(T)) {
    NumElts = AT->getNumElements();
    EltTy = (NumElts == 0 ? nullptr : AT->getElementType());
    return true;
  }
  if (StructType *ST = dyn_cast<StructType>(T)) {
    NumElts = ST->getNumContainedTypes();
    EltTy = (NumElts == 0 ? nullptr : ST->getContainedType(0));
    for (unsigned n = 1; n < NumElts; ++n) {
      if (ST->getContainedType(n) != EltTy)
        return false;
    }
    return true;
  }
  return false;
}

/// isCompatibleAggregate - Check if T1 and T2 are either the same type or are
/// "homogeneous" aggregates with the same element type and number of elements.
static bool isCompatibleAggregate(Type *T1, Type *T2) {
  if (T1 == T2)
    return true;

  unsigned NumElts1, NumElts2;
  Type *EltTy1, *EltTy2;
  if (isHomogeneousAggregate(T1, NumElts1, EltTy1) &&
      isHomogeneousAggregate(T2, NumElts2, EltTy2) && NumElts1 == NumElts2 &&
      EltTy1 == EltTy2)
    return true;

  return false;
}


/// LoadVectorArray - Load vector array like [2 x <4 x float>] from
///  arrays like 4 [2 x float] or struct array like
///  [2 x { <4 x float>, < 4 x uint> }]
/// from arrays like [ 2 x <4 x float> ], [ 2 x <4 x uint> ].
static Value *LoadVectorOrStructArray(ArrayType *AT, ArrayRef<Value *> NewElts,
                              SmallVector<Value *, 8> &idxList,
                              IRBuilder<> &Builder) {
  Type *EltTy = AT->getElementType();
  Value *retVal = llvm::UndefValue::get(AT);
  Type *i32Ty = Type::getInt32Ty(EltTy->getContext());

  uint32_t arraySize = AT->getNumElements();
  for (uint32_t i = 0; i < arraySize; i++) {
    Constant *idx = ConstantInt::get(i32Ty, i);
    idxList.emplace_back(idx);

    if (ArrayType *EltAT = dyn_cast<ArrayType>(EltTy)) {
      Value *EltVal = LoadVectorOrStructArray(EltAT, NewElts, idxList, Builder);
      retVal = Builder.CreateInsertValue(retVal, EltVal, i);
    } else {
      assert((EltTy->isVectorTy() ||
              EltTy->isStructTy()) && "must be a vector or struct type");
      bool isVectorTy = EltTy->isVectorTy();
      Value *retVec = llvm::UndefValue::get(EltTy);

      if (isVectorTy) {
        for (uint32_t c = 0; c < EltTy->getVectorNumElements(); c++) {
          Value *GEP = Builder.CreateInBoundsGEP(NewElts[c], idxList);
          Value *elt = Builder.CreateLoad(GEP);
          retVec = Builder.CreateInsertElement(retVec, elt, c);
        }
      } else {
        for (uint32_t c = 0; c < EltTy->getStructNumElements(); c++) {
          Value *GEP = Builder.CreateInBoundsGEP(NewElts[c], idxList);
          Value *elt = Builder.CreateLoad(GEP);
          retVec = Builder.CreateInsertValue(retVec, elt, c);
        }
      }

      retVal = Builder.CreateInsertValue(retVal, retVec, i);
    }
    idxList.pop_back();
  }
  return retVal;
}

/// LoadVectorArray - Store vector array like [2 x <4 x float>] to
///  arrays like 4 [2 x float] or struct array like
///  [2 x { <4 x float>, < 4 x uint> }]
/// from arrays like [ 2 x <4 x float> ], [ 2 x <4 x uint> ].
static void StoreVectorOrStructArray(ArrayType *AT, Value *val,
                             ArrayRef<Value *> NewElts,
                             SmallVector<Value *, 8> &idxList,
                             IRBuilder<> &Builder) {
  Type *EltTy = AT->getElementType();
  Type *i32Ty = Type::getInt32Ty(EltTy->getContext());

  uint32_t arraySize = AT->getNumElements();
  for (uint32_t i = 0; i < arraySize; i++) {
    Value *elt = Builder.CreateExtractValue(val, i);

    Constant *idx = ConstantInt::get(i32Ty, i);
    idxList.emplace_back(idx);

    if (ArrayType *EltAT = dyn_cast<ArrayType>(EltTy)) {
      StoreVectorOrStructArray(EltAT, elt, NewElts, idxList, Builder);
    } else {
      assert((EltTy->isVectorTy() ||
              EltTy->isStructTy()) && "must be a vector or struct type");
      bool isVectorTy = EltTy->isVectorTy();
      if (isVectorTy) {
        for (uint32_t c = 0; c < EltTy->getVectorNumElements(); c++) {
          Value *component = Builder.CreateExtractElement(elt, c);
          Value *GEP = Builder.CreateInBoundsGEP(NewElts[c], idxList);
          Builder.CreateStore(component, GEP);
        }
      } else {
        for (uint32_t c = 0; c < EltTy->getStructNumElements(); c++) {
          Value *field = Builder.CreateExtractValue(elt, c);
          Value *GEP = Builder.CreateInBoundsGEP(NewElts[c], idxList);
          Builder.CreateStore(field, GEP);
        }
      }
    }
    idxList.pop_back();
  }
}

namespace {

// Simple struct to split memcpy into ld/st
struct MemcpySplitter {
  llvm::LLVMContext &m_context;
  DxilTypeSystem &m_typeSys;

public:
  MemcpySplitter(llvm::LLVMContext &context, DxilTypeSystem &typeSys)
      : m_context(context), m_typeSys(typeSys) {}
  void Split(llvm::Function &F);

  static void PatchMemCpyWithZeroIdxGEP(Module &M);
  static void PatchMemCpyWithZeroIdxGEP(MemCpyInst *MI, const DataLayout &DL);
  static void SplitMemCpy(MemCpyInst *MI, const DataLayout &DL,
                          DxilFieldAnnotation *fieldAnnotation,
                          DxilTypeSystem &typeSys,
                          const bool bEltMemCpy = true);
};

// Copy data from srcPtr to destPtr.
void SimplePtrCopy(Value *DestPtr, Value *SrcPtr,
                   llvm::SmallVector<llvm::Value *, 16> &idxList,
                   IRBuilder<> &Builder) {
  if (idxList.size() > 1) {
    DestPtr = Builder.CreateInBoundsGEP(DestPtr, idxList);
    SrcPtr = Builder.CreateInBoundsGEP(SrcPtr, idxList);
  }
  llvm::LoadInst *ld = Builder.CreateLoad(SrcPtr);
  Builder.CreateStore(ld, DestPtr);
}

// Copy srcVal to destPtr.
void SimpleValCopy(Value *DestPtr, Value *SrcVal,
                   llvm::SmallVector<llvm::Value *, 16> &idxList,
                   IRBuilder<> &Builder) {
  Value *DestGEP = Builder.CreateInBoundsGEP(DestPtr, idxList);
  Value *Val = SrcVal;
  // Skip beginning pointer type.
  for (unsigned i = 1; i < idxList.size(); i++) {
    ConstantInt *idx = cast<ConstantInt>(idxList[i]);
    Type *Ty = Val->getType();
    if (Ty->isAggregateType()) {
      Val = Builder.CreateExtractValue(Val, idx->getLimitedValue());
    }
  }

  Builder.CreateStore(Val, DestGEP);
}

void SimpleCopy(Value *Dest, Value *Src,
                llvm::SmallVector<llvm::Value *, 16> &idxList,
                IRBuilder<> &Builder) {
  if (Src->getType()->isPointerTy())
    SimplePtrCopy(Dest, Src, idxList, Builder);
  else
    SimpleValCopy(Dest, Src, idxList, Builder);
}

Value *CreateMergedGEP(Value *Ptr, SmallVector<Value *, 16> &idxList,
                       IRBuilder<> &Builder) {
  if (GEPOperator *GEPPtr = dyn_cast<GEPOperator>(Ptr)) {
    SmallVector<Value *, 2> IdxList(GEPPtr->idx_begin(), GEPPtr->idx_end());
    // skip idxLIst.begin() because it is included in GEPPtr idx.
    IdxList.append(idxList.begin() + 1, idxList.end());
    return Builder.CreateInBoundsGEP(GEPPtr->getPointerOperand(), IdxList);
  } else {
    return Builder.CreateInBoundsGEP(Ptr, idxList);
  }
}

void EltMemCpy(Type *Ty, Value *Dest, Value *Src,
               SmallVector<Value *, 16> &idxList, IRBuilder<> &Builder,
               const DataLayout &DL) {
  Value *DestGEP = CreateMergedGEP(Dest, idxList, Builder);
  Value *SrcGEP = CreateMergedGEP(Src, idxList, Builder);
  unsigned size = DL.getTypeAllocSize(Ty);
  Builder.CreateMemCpy(DestGEP, SrcGEP, size, /* Align */ 1);
}

bool IsMemCpyTy(Type *Ty, DxilTypeSystem &typeSys) {
  if (!Ty->isAggregateType())
    return false;
  if (HLMatrixType::isa(Ty))
    return false;
  if (dxilutil::IsHLSLObjectType(Ty))
    return false;
  if (StructType *ST = dyn_cast<StructType>(Ty)) {
    DxilStructAnnotation *STA = typeSys.GetStructAnnotation(ST);
    DXASSERT(STA, "require annotation here");
    if (STA->IsEmptyStruct())
      return false;
    // Skip 1 element struct which the element is basic type.
    // Because create memcpy will create gep on the struct, memcpy the basic
    // type only.
    if (ST->getNumElements() == 1)
      return IsMemCpyTy(ST->getElementType(0), typeSys);
  }
  return true;
}

// Split copy into ld/st.
void SplitCpy(Type *Ty, Value *Dest, Value *Src,
              SmallVector<Value *, 16> &idxList, IRBuilder<> &Builder,
              const DataLayout &DL, DxilTypeSystem &typeSys,
              const DxilFieldAnnotation *fieldAnnotation,
              const bool bEltMemCpy = true) {
  if (PointerType *PT = dyn_cast<PointerType>(Ty)) {
    Constant *idx = Constant::getIntegerValue(
        IntegerType::get(Ty->getContext(), 32), APInt(32, 0));
    idxList.emplace_back(idx);

    SplitCpy(PT->getElementType(), Dest, Src, idxList, Builder, DL, typeSys,
             fieldAnnotation, bEltMemCpy);

    idxList.pop_back();
  } else if (HLMatrixType::isa(Ty)) {
    // If no fieldAnnotation, use row major as default.
    // Only load then store immediately should be fine.
    bool bRowMajor = true;
    if (fieldAnnotation) {
      DXASSERT(fieldAnnotation->HasMatrixAnnotation(),
               "must has matrix annotation");
      bRowMajor = fieldAnnotation->GetMatrixAnnotation().Orientation ==
                  MatrixOrientation::RowMajor;
    }
    Module *M = Builder.GetInsertPoint()->getModule();

    Value *DestMatPtr;
    Value *SrcMatPtr;
    if (idxList.size() == 1 &&
        idxList[0] == ConstantInt::get(IntegerType::get(Ty->getContext(), 32),
                                       APInt(32, 0))) {
      // Avoid creating GEP(0)
      DestMatPtr = Dest;
      SrcMatPtr = Src;
    } else {
      DestMatPtr = Builder.CreateInBoundsGEP(Dest, idxList);
      SrcMatPtr = Builder.CreateInBoundsGEP(Src, idxList);
    }

    HLMatLoadStoreOpcode loadOp = bRowMajor ? HLMatLoadStoreOpcode::RowMatLoad
                                            : HLMatLoadStoreOpcode::ColMatLoad;
    HLMatLoadStoreOpcode storeOp = bRowMajor
                                       ? HLMatLoadStoreOpcode::RowMatStore
                                       : HLMatLoadStoreOpcode::ColMatStore;

    Value *Load = HLModule::EmitHLOperationCall(
        Builder, HLOpcodeGroup::HLMatLoadStore, static_cast<unsigned>(loadOp),
        Ty, {SrcMatPtr}, *M);
    HLModule::EmitHLOperationCall(Builder, HLOpcodeGroup::HLMatLoadStore,
                                  static_cast<unsigned>(storeOp), Ty,
                                  {DestMatPtr, Load}, *M);
  } else if (StructType *ST = dyn_cast<StructType>(Ty)) {
    if (dxilutil::IsHLSLObjectType(ST)) {
      // Avoid split HLSL object.
      SimpleCopy(Dest, Src, idxList, Builder);
      return;
    }
    // Built-in structs have no type annotation
    DxilStructAnnotation *STA = typeSys.GetStructAnnotation(ST);
    if (STA && STA->IsEmptyStruct())
      return;
    for (uint32_t i = 0; i < ST->getNumElements(); i++) {
      llvm::Type *ET = ST->getElementType(i);
      Constant *idx = llvm::Constant::getIntegerValue(
          IntegerType::get(Ty->getContext(), 32), APInt(32, i));
      idxList.emplace_back(idx);
      if (bEltMemCpy && IsMemCpyTy(ET, typeSys)) {
        EltMemCpy(ET, Dest, Src, idxList, Builder, DL);
      } else {
        DxilFieldAnnotation *EltAnnotation =
            STA ? &STA->GetFieldAnnotation(i) : nullptr;
        SplitCpy(ET, Dest, Src, idxList, Builder, DL, typeSys, EltAnnotation,
                 bEltMemCpy);
      }

      idxList.pop_back();
    }

  } else if (ArrayType *AT = dyn_cast<ArrayType>(Ty)) {
    Type *ET = AT->getElementType();

    for (uint32_t i = 0; i < AT->getNumElements(); i++) {
      Constant *idx = Constant::getIntegerValue(
          IntegerType::get(Ty->getContext(), 32), APInt(32, i));
      idxList.emplace_back(idx);
      if (bEltMemCpy && IsMemCpyTy(ET, typeSys)) {
        EltMemCpy(ET, Dest, Src, idxList, Builder, DL);
      } else {
        SplitCpy(ET, Dest, Src, idxList, Builder, DL, typeSys, fieldAnnotation,
                 bEltMemCpy);
      }

      idxList.pop_back();
    }
  } else {
    SimpleCopy(Dest, Src, idxList, Builder);
  }
}

// Given a pointer to a value, produces a list of pointers to
// all scalar elements of that value and their field annotations, at any nesting
// level.
void SplitPtr(
    Value *Ptr,                        // The root value pointer
    SmallVectorImpl<Value *> &IdxList, // GEP indices stack during recursion
    Type *Ty, // Type at the current GEP indirection level
    const DxilFieldAnnotation
        &Annotation, // Annotation at the current GEP indirection level
    SmallVectorImpl<Value *>
        &EltPtrList, // Accumulates pointers to each element found
    SmallVectorImpl<const DxilFieldAnnotation *>
        &EltAnnotationList, // Accumulates field annotations for each element
                            // found
    DxilTypeSystem &TypeSys, IRBuilder<> &Builder) {

  if (PointerType *PT = dyn_cast<PointerType>(Ty)) {
    Constant *idx = Constant::getIntegerValue(
        IntegerType::get(Ty->getContext(), 32), APInt(32, 0));
    IdxList.emplace_back(idx);

    SplitPtr(Ptr, IdxList, PT->getElementType(), Annotation, EltPtrList,
             EltAnnotationList, TypeSys, Builder);

    IdxList.pop_back();
    return;
  }

  if (StructType *ST = dyn_cast<StructType>(Ty)) {
    if (!HLMatrixType::isa(Ty) && !dxilutil::IsHLSLObjectType(ST)) {
      const DxilStructAnnotation *SA = TypeSys.GetStructAnnotation(ST);

      for (uint32_t i = 0; i < ST->getNumElements(); i++) {
        llvm::Type *EltTy = ST->getElementType(i);

        Constant *idx = llvm::Constant::getIntegerValue(
            IntegerType::get(Ty->getContext(), 32), APInt(32, i));
        IdxList.emplace_back(idx);

        SplitPtr(Ptr, IdxList, EltTy, SA->GetFieldAnnotation(i), EltPtrList,
                 EltAnnotationList, TypeSys, Builder);

        IdxList.pop_back();
      }
      return;
    }
  }

  if (ArrayType *AT = dyn_cast<ArrayType>(Ty)) {
    if (AT->getArrayNumElements() == 0) {
      // Skip cases like [0 x %struct], nothing to copy
      return;
    }

    Type *ElTy = AT->getElementType();
    SmallVector<ArrayType *, 4> nestArrayTys;

    nestArrayTys.emplace_back(AT);
    // support multi level of array
    while (ElTy->isArrayTy()) {
      ArrayType *ElAT = cast<ArrayType>(ElTy);
      nestArrayTys.emplace_back(ElAT);
      ElTy = ElAT->getElementType();
    }

    if (ElTy->isStructTy() && !HLMatrixType::isa(ElTy)) {
      DXASSERT(0, "Not support array of struct when split pointers.");
      return;
    }
  }

  // Return a pointer to the current element and its annotation
  Value *GEP = Builder.CreateInBoundsGEP(Ptr, IdxList);
  EltPtrList.emplace_back(GEP);
  EltAnnotationList.emplace_back(&Annotation);
}

// Support case when bitcast (gep ptr, 0,0) is transformed into bitcast ptr.
unsigned MatchSizeByCheckElementType(Type *Ty, const DataLayout &DL,
                                     unsigned size, unsigned level) {
  unsigned ptrSize = DL.getTypeAllocSize(Ty);
  // Size match, return current level.
  if (ptrSize == size) {
    // Do not go deeper for matrix or object.
    if (HLMatrixType::isa(Ty) || dxilutil::IsHLSLObjectType(Ty))
      return level;
    // For struct, go deeper if size not change.
    // This will leave memcpy to deeper level when flatten.
    if (StructType *ST = dyn_cast<StructType>(Ty)) {
      if (ST->getNumElements() == 1) {
        return MatchSizeByCheckElementType(ST->getElementType(0), DL, size,
                                           level + 1);
      }
    }
    // Don't do this for array.
    // Array will be flattened as struct of array.
    return level;
  }
  // Add ZeroIdx cannot make ptrSize bigger.
  if (ptrSize < size)
    return 0;
  // ptrSize > size.
  // Try to use element type to make size match.
  if (StructType *ST = dyn_cast<StructType>(Ty)) {
    return MatchSizeByCheckElementType(ST->getElementType(0), DL, size,
                                       level + 1);
  } else if (ArrayType *AT = dyn_cast<ArrayType>(Ty)) {
    return MatchSizeByCheckElementType(AT->getElementType(), DL, size,
                                       level + 1);
  } else {
    return 0;
  }
}

void PatchZeroIdxGEP(Value *Ptr, Value *RawPtr, MemCpyInst *MI, unsigned level,
                     IRBuilder<> &Builder) {
  Value *zeroIdx = Builder.getInt32(0);
  Value *GEP = nullptr;
  if (GEPOperator *GEPPtr = dyn_cast<GEPOperator>(Ptr)) {
    SmallVector<Value *, 2> IdxList(GEPPtr->idx_begin(), GEPPtr->idx_end());
    // level not + 1 because it is included in GEPPtr idx.
    IdxList.append(level, zeroIdx);
    GEP = Builder.CreateInBoundsGEP(GEPPtr->getPointerOperand(), IdxList);
  } else {
    SmallVector<Value *, 2> IdxList(level + 1, zeroIdx);
    GEP = Builder.CreateInBoundsGEP(Ptr, IdxList);
  }
  // Use BitCastInst::Create to prevent idxList from being optimized.
  CastInst *Cast =
      BitCastInst::Create(Instruction::BitCast, GEP, RawPtr->getType());
  Builder.Insert(Cast);
  MI->replaceUsesOfWith(RawPtr, Cast);
  // Remove RawPtr if possible.
  if (RawPtr->user_empty()) {
    if (Instruction *I = dyn_cast<Instruction>(RawPtr)) {
      I->eraseFromParent();
    }
  }
}

void MemcpySplitter::PatchMemCpyWithZeroIdxGEP(MemCpyInst *MI,
                                               const DataLayout &DL) {
  Value *Dest = MI->getRawDest();
  Value *Src = MI->getRawSource();
  // Only remove one level bitcast generated from inline.
  if (BitCastOperator *BC = dyn_cast<BitCastOperator>(Dest))
    Dest = BC->getOperand(0);
  if (BitCastOperator *BC = dyn_cast<BitCastOperator>(Src))
    Src = BC->getOperand(0);

  IRBuilder<> Builder(MI);
  ConstantInt *zero = Builder.getInt32(0);
  Type *DestTy = Dest->getType()->getPointerElementType();
  Type *SrcTy = Src->getType()->getPointerElementType();
  // Support case when bitcast (gep ptr, 0,0) is transformed into
  // bitcast ptr.
  // Also replace (gep ptr, 0) with ptr.
  ConstantInt *Length = cast<ConstantInt>(MI->getLength());
  unsigned size = Length->getLimitedValue();
  if (unsigned level = MatchSizeByCheckElementType(DestTy, DL, size, 0)) {
    PatchZeroIdxGEP(Dest, MI->getRawDest(), MI, level, Builder);
  } else if (GEPOperator *GEP = dyn_cast<GEPOperator>(Dest)) {
    if (GEP->getNumIndices() == 1) {
      Value *idx = *GEP->idx_begin();
      if (idx == zero) {
        GEP->replaceAllUsesWith(GEP->getPointerOperand());
      }
    }
  }
  if (unsigned level = MatchSizeByCheckElementType(SrcTy, DL, size, 0)) {
    PatchZeroIdxGEP(Src, MI->getRawSource(), MI, level, Builder);
  } else if (GEPOperator *GEP = dyn_cast<GEPOperator>(Src)) {
    if (GEP->getNumIndices() == 1) {
      Value *idx = *GEP->idx_begin();
      if (idx == zero) {
        GEP->replaceAllUsesWith(GEP->getPointerOperand());
      }
    }
  }
}

void MemcpySplitter::PatchMemCpyWithZeroIdxGEP(Module &M) {
  const DataLayout &DL = M.getDataLayout();
  for (Function &F : M.functions()) {
    for (Function::iterator BB = F.begin(), BBE = F.end(); BB != BBE; ++BB) {
      for (BasicBlock::iterator BI = BB->begin(), BE = BB->end(); BI != BE;) {
        // Avoid invalidating the iterator.
        Instruction *I = BI++;

        if (MemCpyInst *MI = dyn_cast<MemCpyInst>(I)) {
          PatchMemCpyWithZeroIdxGEP(MI, DL);
        }
      }
    }
  }
}

void DeleteMemcpy(MemCpyInst *MI) {
  Value *Op0 = MI->getOperand(0);
  Value *Op1 = MI->getOperand(1);
  // delete memcpy
  MI->eraseFromParent();
  if (Instruction *op0 = dyn_cast<Instruction>(Op0)) {
    if (op0->user_empty())
      op0->eraseFromParent();
  }
  if (Instruction *op1 = dyn_cast<Instruction>(Op1)) {
    if (op1->user_empty())
      op1->eraseFromParent();
  }
}

// If user is function call, return param annotation to get matrix major.
DxilFieldAnnotation *FindAnnotationFromMatUser(Value *Mat,
                                               DxilTypeSystem &typeSys) {
  for (User *U : Mat->users()) {
    if (CallInst *CI = dyn_cast<CallInst>(U)) {
      Function *F = CI->getCalledFunction();
      if (DxilFunctionAnnotation *Anno = typeSys.GetFunctionAnnotation(F)) {
        for (unsigned i = 0; i < CI->getNumArgOperands(); i++) {
          if (CI->getArgOperand(i) == Mat) {
            return &Anno->GetParameterAnnotation(i);
          }
        }
      }
    }
  }
  return nullptr;
}

namespace {
bool isCBVec4ArrayToScalarArray(Type *TyV, Value *Src, Type *TySrc, const DataLayout &DL) {
  Value *SrcPtr = Src;
  while (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(SrcPtr)) {
    SrcPtr = GEP->getPointerOperand();
  }
  CallInst *CI = dyn_cast<CallInst>(SrcPtr);
  if (!CI)
    return false;

  Function *F = CI->getCalledFunction();
  if (hlsl::GetHLOpcodeGroupByName(F) != HLOpcodeGroup::HLSubscript)
    return false;

  if (hlsl::GetHLOpcode(CI) != (unsigned)HLSubscriptOpcode::CBufferSubscript)
    return false;

  ArrayType *AT = dyn_cast<ArrayType>(TySrc);
  if (!AT)
    return false;
  VectorType *VT = dyn_cast<VectorType>(AT->getElementType());

  if (!VT)
    return false;

  if (DL.getTypeSizeInBits(VT) != 128)
    return false;

  ArrayType *DstAT = dyn_cast<ArrayType>(TyV);
  if (!DstAT)
    return false;

  if (VT->getElementType() != DstAT->getElementType())
    return false;

  unsigned sizeInBits = DL.getTypeSizeInBits(VT->getElementType());
  if (sizeInBits < 32)
    return false;
  return true;
}

bool trySplitCBVec4ArrayToScalarArray(Value *Dest, Type *TyV, Value *Src,
                                      Type *TySrc, const DataLayout &DL,
                                      IRBuilder<> &B) {
  if (!isCBVec4ArrayToScalarArray(TyV, Src, TySrc, DL))
    return false;

  ArrayType *AT = cast<ArrayType>(TyV);
  Type *EltTy = AT->getElementType();
  unsigned sizeInBits = DL.getTypeSizeInBits(EltTy);
  unsigned vecSize = 4;
  if (sizeInBits == 64)
    vecSize = 2;
  unsigned arraySize = AT->getNumElements();
  unsigned vecArraySize = arraySize / vecSize;
  Value *zeroIdx = B.getInt32(0);
  for (unsigned a = 0; a < vecArraySize; a++) {
    Value *SrcGEP = B.CreateGEP(Src, {zeroIdx, B.getInt32(a)});
    Value *Ld = B.CreateLoad(SrcGEP);
    for (unsigned v = 0; v < vecSize; v++) {
      Value *Elt = B.CreateExtractElement(Ld, v);

      Value *DestGEP =
          B.CreateGEP(Dest, {zeroIdx, B.getInt32(a * vecSize + v)});
      B.CreateStore(Elt, DestGEP);
    }
  }

  return true;
}

}

void MemcpySplitter::SplitMemCpy(MemCpyInst *MI, const DataLayout &DL,
                                 DxilFieldAnnotation *fieldAnnotation,
                                 DxilTypeSystem &typeSys,
                                 const bool bEltMemCpy) {
  Value *Dest = MI->getRawDest();
  Value *Src = MI->getRawSource();
  // Only remove one level bitcast generated from inline.
  if (BitCastOperator *BC = dyn_cast<BitCastOperator>(Dest))
    Dest = BC->getOperand(0);
  if (BitCastOperator *BC = dyn_cast<BitCastOperator>(Src))
    Src = BC->getOperand(0);

  if (Dest == Src) {
    // delete self copy.
    DeleteMemcpy(MI);
    return;
  }

  IRBuilder<> Builder(MI);
  Type *DestTy = Dest->getType()->getPointerElementType();
  Type *SrcTy = Src->getType()->getPointerElementType();

  // Allow copy between different address space.
  if (DestTy != SrcTy) {
    if (trySplitCBVec4ArrayToScalarArray(Dest, DestTy, Src, SrcTy, DL,
                                         Builder)) {
      // delete memcpy
      DeleteMemcpy(MI);
    }
    return;
  }
  // Try to find fieldAnnotation from user of Dest/Src.
  if (!fieldAnnotation) {
    Type *EltTy = dxilutil::GetArrayEltTy(DestTy);
    if (HLMatrixType::isa(EltTy)) {
      fieldAnnotation = FindAnnotationFromMatUser(Dest, typeSys);
    }
  }

  llvm::SmallVector<llvm::Value *, 16> idxList;
  // split
  // Matrix is treated as scalar type, will not use memcpy.
  // So use nullptr for fieldAnnotation should be safe here.
  SplitCpy(Dest->getType(), Dest, Src, idxList, Builder, DL, typeSys,
           fieldAnnotation, bEltMemCpy);
  // delete memcpy
  DeleteMemcpy(MI);
}

void MemcpySplitter::Split(llvm::Function &F) {
  const DataLayout &DL = F.getParent()->getDataLayout();
  SmallVector<Function *, 2> memcpys;
  for (Function &Fn : F.getParent()->functions()) {
    if (Fn.getIntrinsicID() == Intrinsic::memcpy) {
      memcpys.emplace_back(&Fn);
    }
  }
  for (Function *memcpy : memcpys) {
    for (auto U = memcpy->user_begin(); U != memcpy->user_end();) {
      MemCpyInst *MI = cast<MemCpyInst>(*(U++));
      if (MI->getParent()->getParent() != &F)
        continue;
      // Matrix is treated as scalar type, will not use memcpy.
      // So use nullptr for fieldAnnotation should be safe here.
      SplitMemCpy(MI, DL, /*fieldAnnotation*/ nullptr, m_typeSys,
                  /*bEltMemCpy*/ false);
    }
  }
}

} // namespace

namespace {

/// DeleteDeadInstructions - Erase instructions on the DeadInstrs list,
/// recursively including all their operands that become trivially dead.
void DeleteDeadInstructions(SmallVector<Value *, 32> &DeadInsts) {
  while (!DeadInsts.empty()) {
    Instruction *I = cast<Instruction>(DeadInsts.pop_back_val());

    for (User::op_iterator OI = I->op_begin(), E = I->op_end(); OI != E; ++OI)
      if (Instruction *U = dyn_cast<Instruction>(*OI)) {
        // Zero out the operand and see if it becomes trivially dead.
        // (But, don't add allocas to the dead instruction list -- they are
        // already on the worklist and will be deleted separately.)
        *OI = nullptr;
        if (isInstructionTriviallyDead(U) && !isa<AllocaInst>(U))
          DeadInsts.push_back(U);
      }

    I->eraseFromParent();
  }
}

// markPrecise - To save the precise attribute on alloca inst which might be
// removed by promote, mark precise attribute with function call on alloca inst
// stores.
bool markPrecise(Function &F) {
  bool Changed = false;
  BasicBlock &BB = F.getEntryBlock();
  for (BasicBlock::iterator I = BB.begin(), E = BB.end(); I != E; ++I)
    if (AllocaInst *A = dyn_cast<AllocaInst>(I)) {
      // TODO: Only do this on basic types.
      if (HLModule::HasPreciseAttributeWithMetadata(A)) {
        HLModule::MarkPreciseAttributeOnPtrWithFunctionCall(A,
                                                            *(F.getParent()));
        Changed = true;
      }
    }
  return Changed;
}

bool Cleanup(Function &F, DxilTypeSystem &typeSys) {
  // change rest memcpy into ld/st.
  MemcpySplitter splitter(F.getContext(), typeSys);
  splitter.Split(F);
  return markPrecise(F);
}
} // namespace

namespace {

/// ShouldAttemptScalarRepl - Decide if an alloca is a good candidate for
/// SROA.  It must be a struct or array type with a small number of elements.
bool ShouldAttemptScalarRepl(AllocaInst *AI) {
  Type *T = AI->getAllocatedType();
  // promote every struct.
  if (dyn_cast<StructType>(T))
    return true;
  // promote every array.
  if (dyn_cast<ArrayType>(T))
    return true;
  return false;
}

/// AllocaInfo - When analyzing uses of an alloca instruction, this captures
/// information about the uses.  All these fields are initialized to false
/// and set to true when something is learned.
struct AllocaInfo {
  /// The alloca to promote.
  AllocaInst *AI;

  /// CheckedPHIs - This is a set of verified PHI nodes, to prevent infinite
  /// looping and avoid redundant work.
  SmallPtrSet<PHINode *, 8> CheckedPHIs;

  /// isUnsafe - This is set to true if the alloca cannot be SROA'd.
  bool isUnsafe : 1;

  /// isMemCpySrc - This is true if this aggregate is memcpy'd from.
  bool isMemCpySrc : 1;

  /// isMemCpyDst - This is true if this aggregate is memcpy'd into.
  bool isMemCpyDst : 1;

  /// hasSubelementAccess - This is true if a subelement of the alloca is
  /// ever accessed, or false if the alloca is only accessed with mem
  /// intrinsics or load/store that only access the entire alloca at once.
  bool hasSubelementAccess : 1;

  /// hasALoadOrStore - This is true if there are any loads or stores to it.
  /// The alloca may just be accessed with memcpy, for example, which would
  /// not set this.
  bool hasALoadOrStore : 1;

  /// hasArrayIndexing - This is true if there are any dynamic array
  /// indexing to it.
  bool hasArrayIndexing : 1;

  /// hasVectorIndexing - This is true if there are any dynamic vector
  /// indexing to it.
  bool hasVectorIndexing : 1;

  explicit AllocaInfo(AllocaInst *ai)
      : AI(ai), isUnsafe(false), isMemCpySrc(false), isMemCpyDst(false),
        hasSubelementAccess(false), hasALoadOrStore(false),
        hasArrayIndexing(false), hasVectorIndexing(false) {}
};

/// TypeHasComponent - Return true if T has a component type with the
/// specified offset and size.  If Size is zero, do not check the size.
bool TypeHasComponent(Type *T, uint64_t Offset, uint64_t Size,
                      const DataLayout &DL) {
  Type *EltTy;
  uint64_t EltSize;
  if (StructType *ST = dyn_cast<StructType>(T)) {
    const StructLayout *Layout = DL.getStructLayout(ST);
    unsigned EltIdx = Layout->getElementContainingOffset(Offset);
    EltTy = ST->getContainedType(EltIdx);
    EltSize = DL.getTypeAllocSize(EltTy);
    Offset -= Layout->getElementOffset(EltIdx);
  } else if (ArrayType *AT = dyn_cast<ArrayType>(T)) {
    EltTy = AT->getElementType();
    EltSize = DL.getTypeAllocSize(EltTy);
    if (Offset >= AT->getNumElements() * EltSize)
      return false;
    Offset %= EltSize;
  } else if (VectorType *VT = dyn_cast<VectorType>(T)) {
    EltTy = VT->getElementType();
    EltSize = DL.getTypeAllocSize(EltTy);
    if (Offset >= VT->getNumElements() * EltSize)
      return false;
    Offset %= EltSize;
  } else {
    return false;
  }
  if (Offset == 0 && (Size == 0 || EltSize == Size))
    return true;
  // Check if the component spans multiple elements.
  if (Offset + Size > EltSize)
    return false;
  return TypeHasComponent(EltTy, Offset, Size, DL);
}

void MarkUnsafe(AllocaInfo &I, Instruction *User) {
  I.isUnsafe = true;
  DEBUG(dbgs() << "  Transformation preventing inst: " << *User << '\n');
}

/// isSafeGEP - Check if a GEP instruction can be handled for scalar
/// replacement.  It is safe when all the indices are constant, in-bounds
/// references, and when the resulting offset corresponds to an element within
/// the alloca type.  The results are flagged in the Info parameter.  Upon
/// return, Offset is adjusted as specified by the GEP indices.
void isSafeGEP(GetElementPtrInst *GEPI, uint64_t &Offset, AllocaInfo &Info) {
  gep_type_iterator GEPIt = gep_type_begin(GEPI), E = gep_type_end(GEPI);
  if (GEPIt == E)
    return;
  bool NonConstant = false;
  unsigned NonConstantIdxSize = 0;

  // Compute the offset due to this GEP and check if the alloca has a
  // component element at that offset.
  SmallVector<Value *, 8> Indices(GEPI->op_begin() + 1, GEPI->op_end());
  auto indicesIt = Indices.begin();

  // Walk through the GEP type indices, checking the types that this indexes
  // into.
  uint32_t arraySize = 0;
  bool isArrayIndexing = false;

  for (; GEPIt != E; ++GEPIt) {
    Type *Ty = *GEPIt;
    if (Ty->isStructTy() && !HLMatrixType::isa(Ty)) {
      // Don't go inside struct when mark hasArrayIndexing and
      // hasVectorIndexing. The following level won't affect scalar repl on the
      // struct.
      break;
    }
    if (GEPIt->isArrayTy()) {
      arraySize = GEPIt->getArrayNumElements();
      isArrayIndexing = true;
    }
    if (GEPIt->isVectorTy()) {
      arraySize = GEPIt->getVectorNumElements();
      isArrayIndexing = false;
    }
    // Allow dynamic indexing
    ConstantInt *IdxVal = dyn_cast<ConstantInt>(GEPIt.getOperand());
    if (!IdxVal) {
      // for dynamic index, use array size - 1 to check the offset
      *indicesIt = Constant::getIntegerValue(
          Type::getInt32Ty(GEPI->getContext()), APInt(32, arraySize - 1));
      if (isArrayIndexing)
        Info.hasArrayIndexing = true;
      else
        Info.hasVectorIndexing = true;
      NonConstant = true;
    }
    indicesIt++;
  }
  // Continue iterate only for the NonConstant.
  for (; GEPIt != E; ++GEPIt) {
    Type *Ty = *GEPIt;
    if (Ty->isArrayTy()) {
      arraySize = GEPIt->getArrayNumElements();
    }
    if (Ty->isVectorTy()) {
      arraySize = GEPIt->getVectorNumElements();
    }
    // Allow dynamic indexing
    ConstantInt *IdxVal = dyn_cast<ConstantInt>(GEPIt.getOperand());
    if (!IdxVal) {
      // for dynamic index, use array size - 1 to check the offset
      *indicesIt = Constant::getIntegerValue(
          Type::getInt32Ty(GEPI->getContext()), APInt(32, arraySize - 1));
      NonConstant = true;
    }
    indicesIt++;
  }
  // If this GEP is non-constant then the last operand must have been a
  // dynamic index into a vector.  Pop this now as it has no impact on the
  // constant part of the offset.
  if (NonConstant)
    Indices.pop_back();

  const DataLayout &DL = GEPI->getModule()->getDataLayout();
  Offset += DL.getIndexedOffset(GEPI->getPointerOperandType(), Indices);
  if (!TypeHasComponent(Info.AI->getAllocatedType(), Offset, NonConstantIdxSize,
                        DL))
    MarkUnsafe(Info, GEPI);
}

/// isSafeMemAccess - Check if a load/store/memcpy operates on the entire AI
/// alloca or has an offset and size that corresponds to a component element
/// within it.  The offset checked here may have been formed from a GEP with a
/// pointer bitcasted to a different type.
///
/// If AllowWholeAccess is true, then this allows uses of the entire alloca as a
/// unit.  If false, it only allows accesses known to be in a single element.
void isSafeMemAccess(uint64_t Offset, uint64_t MemSize, Type *MemOpType,
                     bool isStore, AllocaInfo &Info, Instruction *TheAccess,
                     bool AllowWholeAccess) {
  // What hlsl cares is Info.hasVectorIndexing.
  // Do nothing here.
}

/// isSafePHIUseForScalarRepl - If we see a PHI node or select using a pointer
/// derived from the alloca, we can often still split the alloca into elements.
/// This is useful if we have a large alloca where one element is phi'd
/// together somewhere: we can SRoA and promote all the other elements even if
/// we end up not being able to promote this one.
///
/// All we require is that the uses of the PHI do not index into other parts of
/// the alloca.  The most important use case for this is single load and stores
/// that are PHI'd together, which can happen due to code sinking.
void isSafePHISelectUseForScalarRepl(Instruction *I, uint64_t Offset,
                                     AllocaInfo &Info) {
  // If we've already checked this PHI, don't do it again.
  if (PHINode *PN = dyn_cast<PHINode>(I))
    if (!Info.CheckedPHIs.insert(PN).second)
      return;

  const DataLayout &DL = I->getModule()->getDataLayout();
  for (User *U : I->users()) {
    Instruction *UI = cast<Instruction>(U);

    if (BitCastInst *BC = dyn_cast<BitCastInst>(UI)) {
      isSafePHISelectUseForScalarRepl(BC, Offset, Info);
    } else if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(UI)) {
      // Only allow "bitcast" GEPs for simplicity.  We could generalize this,
      // but would have to prove that we're staying inside of an element being
      // promoted.
      if (!GEPI->hasAllZeroIndices())
        return MarkUnsafe(Info, UI);
      isSafePHISelectUseForScalarRepl(GEPI, Offset, Info);
    } else if (LoadInst *LI = dyn_cast<LoadInst>(UI)) {
      if (!LI->isSimple())
        return MarkUnsafe(Info, UI);
      Type *LIType = LI->getType();
      isSafeMemAccess(Offset, DL.getTypeAllocSize(LIType), LIType, false, Info,
                      LI, false /*AllowWholeAccess*/);
      Info.hasALoadOrStore = true;

    } else if (StoreInst *SI = dyn_cast<StoreInst>(UI)) {
      // Store is ok if storing INTO the pointer, not storing the pointer
      if (!SI->isSimple() || SI->getOperand(0) == I)
        return MarkUnsafe(Info, UI);

      Type *SIType = SI->getOperand(0)->getType();
      isSafeMemAccess(Offset, DL.getTypeAllocSize(SIType), SIType, true, Info,
                      SI, false /*AllowWholeAccess*/);
      Info.hasALoadOrStore = true;
    } else if (isa<PHINode>(UI) || isa<SelectInst>(UI)) {
      isSafePHISelectUseForScalarRepl(UI, Offset, Info);
    } else {
      return MarkUnsafe(Info, UI);
    }
    if (Info.isUnsafe)
      return;
  }
}

/// isSafeForScalarRepl - Check if instruction I is a safe use with regard to
/// performing scalar replacement of alloca AI.  The results are flagged in
/// the Info parameter.  Offset indicates the position within AI that is
/// referenced by this instruction.
void isSafeForScalarRepl(Instruction *I, uint64_t Offset, AllocaInfo &Info) {
  if (I->getType()->isPointerTy()) {
    // Don't check object pointers.
    if (dxilutil::IsHLSLObjectType(I->getType()->getPointerElementType()))
      return;
  }
  const DataLayout &DL = I->getModule()->getDataLayout();
  for (Use &U : I->uses()) {
    Instruction *User = cast<Instruction>(U.getUser());

    if (BitCastInst *BC = dyn_cast<BitCastInst>(User)) {
      isSafeForScalarRepl(BC, Offset, Info);
    } else if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(User)) {
      uint64_t GEPOffset = Offset;
      isSafeGEP(GEPI, GEPOffset, Info);
      if (!Info.isUnsafe)
        isSafeForScalarRepl(GEPI, GEPOffset, Info);
    } else if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(User)) {
      ConstantInt *Length = dyn_cast<ConstantInt>(MI->getLength());
      if (!Length || Length->isNegative())
        return MarkUnsafe(Info, User);

      isSafeMemAccess(Offset, Length->getZExtValue(), nullptr,
                      U.getOperandNo() == 0, Info, MI,
                      true /*AllowWholeAccess*/);
    } else if (LoadInst *LI = dyn_cast<LoadInst>(User)) {
      if (!LI->isSimple())
        return MarkUnsafe(Info, User);
      Type *LIType = LI->getType();
      isSafeMemAccess(Offset, DL.getTypeAllocSize(LIType), LIType, false, Info,
                      LI, true /*AllowWholeAccess*/);
      Info.hasALoadOrStore = true;

    } else if (StoreInst *SI = dyn_cast<StoreInst>(User)) {
      // Store is ok if storing INTO the pointer, not storing the pointer
      if (!SI->isSimple() || SI->getOperand(0) == I)
        return MarkUnsafe(Info, User);

      Type *SIType = SI->getOperand(0)->getType();
      isSafeMemAccess(Offset, DL.getTypeAllocSize(SIType), SIType, true, Info,
                      SI, true /*AllowWholeAccess*/);
      Info.hasALoadOrStore = true;
    } else if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(User)) {
      if (II->getIntrinsicID() != Intrinsic::lifetime_start &&
          II->getIntrinsicID() != Intrinsic::lifetime_end)
        return MarkUnsafe(Info, User);
    } else if (isa<PHINode>(User) || isa<SelectInst>(User)) {
      isSafePHISelectUseForScalarRepl(User, Offset, Info);
    } else if (CallInst *CI = dyn_cast<CallInst>(User)) {
      HLOpcodeGroup group = GetHLOpcodeGroupByName(CI->getCalledFunction());
      // Most HL functions are safe for scalar repl.
      if (HLOpcodeGroup::NotHL == group)
        return MarkUnsafe(Info, User);
      else if (HLOpcodeGroup::HLIntrinsic == group) {
        // TODO: should we check HL parameter type for UDT overload instead of
        // basing on IOP?
        IntrinsicOp opcode = static_cast<IntrinsicOp>(GetHLOpcode(CI));
        if (IntrinsicOp::IOP_TraceRay == opcode ||
            IntrinsicOp::IOP_ReportHit == opcode ||
            IntrinsicOp::IOP_CallShader == opcode) {
          return MarkUnsafe(Info, User);
        }
      }
    } else {
      return MarkUnsafe(Info, User);
    }
    if (Info.isUnsafe)
      return;
  }
}

/// HasPadding - Return true if the specified type has any structure or
/// alignment padding in between the elements that would be split apart
/// by SROA; return false otherwise.
static bool HasPadding(Type *Ty, const DataLayout &DL) {
  if (ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
    Ty = ATy->getElementType();
    return DL.getTypeSizeInBits(Ty) != DL.getTypeAllocSizeInBits(Ty);
  }

  // SROA currently handles only Arrays and Structs.
  StructType *STy = cast<StructType>(Ty);
  const StructLayout *SL = DL.getStructLayout(STy);
  unsigned PrevFieldBitOffset = 0;
  for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
    unsigned FieldBitOffset = SL->getElementOffsetInBits(i);

    // Check to see if there is any padding between this element and the
    // previous one.
    if (i) {
      unsigned PrevFieldEnd =
          PrevFieldBitOffset + DL.getTypeSizeInBits(STy->getElementType(i - 1));
      if (PrevFieldEnd < FieldBitOffset)
        return true;
    }
    PrevFieldBitOffset = FieldBitOffset;
  }
  // Check for tail padding.
  if (unsigned EltCount = STy->getNumElements()) {
    unsigned PrevFieldEnd =
        PrevFieldBitOffset +
        DL.getTypeSizeInBits(STy->getElementType(EltCount - 1));
    if (PrevFieldEnd < SL->getSizeInBits())
      return true;
  }
  return false;
}

/// isSafeStructAllocaToScalarRepl - Check to see if the specified allocation of
/// an aggregate can be broken down into elements.  Return 0 if not, 3 if safe,
/// or 1 if safe after canonicalization has been performed.
bool isSafeAllocaToScalarRepl(AllocaInst *AI) {
  // Loop over the use list of the alloca.  We can only transform it if all of
  // the users are safe to transform.
  AllocaInfo Info(AI);

  isSafeForScalarRepl(AI, 0, Info);
  if (Info.isUnsafe) {
    DEBUG(dbgs() << "Cannot transform: " << *AI << '\n');
    return false;
  }

  // vector indexing need translate vector into array
  if (Info.hasVectorIndexing)
    return false;

  const DataLayout &DL = AI->getModule()->getDataLayout();

  // Okay, we know all the users are promotable.  If the aggregate is a memcpy
  // source and destination, we have to be careful.  In particular, the memcpy
  // could be moving around elements that live in structure padding of the LLVM
  // types, but may actually be used.  In these cases, we refuse to promote the
  // struct.
  if (Info.isMemCpySrc && Info.isMemCpyDst &&
      HasPadding(AI->getAllocatedType(), DL))
    return false;

  return true;
}
} // namespace

namespace {

struct GVDbgOffset {
  GlobalVariable *base;
  unsigned debugOffset;
};

bool hasDynamicVectorIndexing(Value *V) {
  for (User *U : V->users()) {
    if (!U->getType()->isPointerTy())
      continue;

    if (dyn_cast<GEPOperator>(U)) {

      gep_type_iterator GEPIt = gep_type_begin(U), E = gep_type_end(U);

      for (; GEPIt != E; ++GEPIt) {
        if (isa<VectorType>(*GEPIt)) {
          Value *VecIdx = GEPIt.getOperand();
          if (!isa<ConstantInt>(VecIdx))
            return true;
        }
      }
    }
  }
  return false;
}

} // namespace

namespace {

void RemoveUnusedInternalGlobalVariable(Module &M) {
  std::vector<GlobalVariable *> staticGVs;
  for (GlobalVariable &GV : M.globals()) {
    if (dxilutil::IsStaticGlobal(&GV) || dxilutil::IsSharedMemoryGlobal(&GV)) {
      staticGVs.emplace_back(&GV);
    }
  }

  for (GlobalVariable *GV : staticGVs) {
    bool onlyStoreUse = true;
    for (User *user : GV->users()) {
      if (isa<StoreInst>(user))
        continue;
      if (isa<ConstantExpr>(user) && user->user_empty())
        continue;

      // Check matrix store.
      if (HLMatrixType::isa(GV->getType()->getPointerElementType())) {
        if (CallInst *CI = dyn_cast<CallInst>(user)) {
          if (GetHLOpcodeGroupByName(CI->getCalledFunction()) ==
              HLOpcodeGroup::HLMatLoadStore) {
            HLMatLoadStoreOpcode opcode =
                static_cast<HLMatLoadStoreOpcode>(GetHLOpcode(CI));
            if (opcode == HLMatLoadStoreOpcode::ColMatStore ||
                opcode == HLMatLoadStoreOpcode::RowMatStore)
              continue;
          }
        }
      }

      onlyStoreUse = false;
      break;
    }
    if (onlyStoreUse) {
      for (auto UserIt = GV->user_begin(); UserIt != GV->user_end();) {
        Value *User = *(UserIt++);
        if (Instruction *I = dyn_cast<Instruction>(User)) {
          I->eraseFromParent();
        } else {
          ConstantExpr *CE = cast<ConstantExpr>(User);
          CE->dropAllReferences();
        }
      }
      GV->eraseFromParent();
    }
  }
}

bool isGroupShareOrConstStaticArray(GlobalVariable *GV) {
  // Disable scalarization of groupshared/const_static vector arrays
  if (!(GV->getType()->getAddressSpace() == DXIL::kTGSMAddrSpace ||
       (GV->isConstant() && GV->hasInitializer() &&
        GV->getLinkage() == GlobalValue::LinkageTypes::InternalLinkage)))
    return false;
  Type *Ty = GV->getType()->getPointerElementType();
  return Ty->isArrayTy();
}

bool SROAGlobalAndAllocas(HLModule &HLM, bool bHasDbgInfo) {
  Module &M = *HLM.GetModule();
  DxilTypeSystem &typeSys = HLM.GetTypeSystem();

  const DataLayout &DL = M.getDataLayout();
  // Make sure big alloca split first.
  // This will simplify memcpy check between part of big alloca and small
  // alloca. Big alloca will be split to smaller piece first, when process the
  // alloca, it will be alloca flattened from big alloca instead of a GEP of
  // big alloca.
  auto size_cmp = [&DL](const Value *a0, const Value *a1) -> bool {
    Type *a0ty = a0->getType()->getPointerElementType();
    Type *a1ty = a1->getType()->getPointerElementType();
    bool isUnitSzStruct0 =
        a0ty->isStructTy() && a0ty->getStructNumElements() == 1;
    bool isUnitSzStruct1 =
        a1ty->isStructTy() && a1ty->getStructNumElements() == 1;
    auto sz0 = DL.getTypeAllocSize(a0ty);
    auto sz1 = DL.getTypeAllocSize(a1ty);
    if (sz0 == sz1 && (isUnitSzStruct0 || isUnitSzStruct1)) {
      sz0 = getNestedLevelInStruct(a0ty);
      sz1 = getNestedLevelInStruct(a1ty);
    }
    // If sizes are equal, tiebreak with alphabetically lesser at higher priority
    return sz0 < sz1 || (sz0 == sz1 && isa<GlobalVariable>(a0) &&
                         isa<GlobalVariable>(a1) &&
                         a0->getName() > a1->getName());
  };

  std::priority_queue<Value *, std::vector<Value *>,
                      std::function<bool(Value *, Value *)>>
      WorkList(size_cmp);

  // Flatten internal global.
  llvm::SetVector<GlobalVariable *> staticGVs;

  DenseMap<GlobalVariable *, GVDbgOffset> GVDbgOffsetMap;
  for (GlobalVariable &GV : M.globals()) {
    if (dxilutil::IsStaticGlobal(&GV) || dxilutil::IsSharedMemoryGlobal(&GV)) {
      staticGVs.insert(&GV);
      GVDbgOffset &dbgOffset = GVDbgOffsetMap[&GV];
      dbgOffset.base = &GV;
      dbgOffset.debugOffset = 0;
    } else {
      // merge GEP use for global.
      dxilutil::MergeGepUse(&GV);
    }
  }
  // Add static GVs to work list.
  for (GlobalVariable *GV : staticGVs)
    WorkList.push(GV);

  DenseMap<Function *, DominatorTree> domTreeMap;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    // Collect domTree.
    domTreeMap[&F].recalculate(F);

    // Scan the entry basic block, adding allocas to the worklist.
    BasicBlock &BB = F.getEntryBlock();
    for (BasicBlock::iterator I = BB.begin(), E = BB.end(); I != E; ++I)
      if (AllocaInst *A = dyn_cast<AllocaInst>(I)) {
        if (!A->user_empty()) {
          WorkList.push(A);
          // merge GEP use for the allocs
          dxilutil::MergeGepUse(A);
        }
      }
  }

  // Establish debug metadata layout name in the context in advance so the name
  // is serialized in both debug and non-debug compilations.
  (void)M.getContext().getMDKindID(
      DxilMDHelper::kDxilVariableDebugLayoutMDName);
  DIBuilder DIB(M, /*AllowUnresolved*/ false);

  /// DeadInsts - Keep track of instructions we have made dead, so that
  /// we can remove them after we are done working.
  SmallVector<Value *, 32> DeadInsts;

  // Only used to create ConstantExpr.
  IRBuilder<> Builder(M.getContext());
  std::unordered_map<Value *, StringRef> EltNameMap;

  bool Changed = false;
  while (!WorkList.empty()) {
    Value *V = WorkList.top();
    WorkList.pop();

    if (AllocaInst *AI = dyn_cast<AllocaInst>(V)) {
      // Handle dead allocas trivially.  These can be formed by SROA'ing arrays
      // with unused elements.
      if (AI->use_empty()) {
        AI->eraseFromParent();
        Changed = true;
        continue;
      }
      Function *F = AI->getParent()->getParent();
      const bool bAllowReplace = true;
      DominatorTree &DT = domTreeMap[F];
      if (SROA_Helper::LowerMemcpy(AI, /*annotation*/ nullptr, typeSys, DL, &DT,
                                   bAllowReplace)) {
        Changed = true;
        continue;
      }

      // If this alloca is impossible for us to promote, reject it early.
      if (AI->isArrayAllocation() || !AI->getAllocatedType()->isSized())
        continue;

      // Check to see if we can perform the core SROA transformation.  We cannot
      // transform the allocation instruction if it is an array allocation
      // (allocations OF arrays are ok though), and an allocation of a scalar
      // value cannot be decomposed at all.
      uint64_t AllocaSize = DL.getTypeAllocSize(AI->getAllocatedType());

      // Do not promote [0 x %struct].
      if (AllocaSize == 0)
        continue;

      Type *Ty = AI->getAllocatedType();
      // Skip empty struct type.
      if (SROA_Helper::IsEmptyStructType(Ty, typeSys)) {
        SROA_Helper::MarkEmptyStructUsers(AI, DeadInsts);
        DeleteDeadInstructions(DeadInsts);
        continue;
      }

      if (Value *NewV = TranslatePtrIfUsedByLoweredFn(AI, typeSys)) {
        if (NewV != AI) {
          DXASSERT(AI->getNumUses() == 0, "must have zero users.");
          // Update debug declare.

          SmallVector<DbgDeclareInst *, 4> dbgDecls;
          llvm::FindAllocaDbgDeclare(AI, dbgDecls);
          for (DbgDeclareInst *DDI : dbgDecls) {
            DDI->setArgOperand(0, MetadataAsValue::get(NewV->getContext(), ValueAsMetadata::get(NewV)));
          }
          AI->eraseFromParent();
          Changed = true;
        }
        continue;
      }

      // If the alloca looks like a good candidate for scalar replacement, and
      // if
      // all its users can be transformed, then split up the aggregate into its
      // separate elements.
      if (ShouldAttemptScalarRepl(AI) && isSafeAllocaToScalarRepl(AI)) {
        std::vector<Value *> Elts;
        IRBuilder<> Builder(dxilutil::FindAllocaInsertionPt(AI));
        bool hasPrecise = HLModule::HasPreciseAttributeWithMetadata(AI);

        Type *BrokenUpTy = nullptr;
        uint64_t NumInstances = 1;
        bool SROAed = SROA_Helper::DoScalarReplacement(
            AI, Elts, BrokenUpTy, NumInstances, Builder,
            /*bFlatVector*/ true, hasPrecise, typeSys, DL, DeadInsts, &DT);

        if (SROAed) {
          Type *Ty = AI->getAllocatedType();
          // Skip empty struct parameters.
          if (StructType *ST = dyn_cast<StructType>(Ty)) {
            if (!HLMatrixType::isa(Ty)) {
              DxilStructAnnotation *SA = typeSys.GetStructAnnotation(ST);
              if (SA && SA->IsEmptyStruct()) {
                for (User *U : AI->users()) {
                  if (StoreInst *SI = dyn_cast<StoreInst>(U))
                    DeadInsts.emplace_back(SI);
                }
                DeleteDeadInstructions(DeadInsts);
                AI->replaceAllUsesWith(UndefValue::get(AI->getType()));
                AI->eraseFromParent();
                continue;
              }
            }
          }

          addDebugInfoForElements(AI, BrokenUpTy, NumInstances, Elts, DL, &DIB);

          // Push Elts into workList.
          for (unsigned EltIdx = 0; EltIdx < Elts.size(); ++EltIdx) {
            AllocaInst *EltAlloca = cast<AllocaInst>(Elts[EltIdx]);
            WorkList.push(EltAlloca);
          }

          // Now erase any instructions that were made dead while rewriting the
          // alloca.
          DeleteDeadInstructions(DeadInsts);
          ++NumReplaced;
          DXASSERT(AI->getNumUses() == 0, "must have zero users.");
          AI->eraseFromParent();
          Changed = true;
          continue;
        }
      }
    } else {
      GlobalVariable *GV = cast<GlobalVariable>(V);
      // Handle dead GVs trivially. These can be formed by RAUWing one GV
      // with another, leaving the original in the worklist
      if (GV->use_empty()) {
        GV->eraseFromParent();
        Changed = true;
        continue;
      }

      if (staticGVs.count(GV)) {
        Type *Ty = GV->getType()->getPointerElementType();
        // Skip basic types.
        if (!Ty->isAggregateType() && !Ty->isVectorTy())
          continue;
        // merge GEP use for global.
        dxilutil::MergeGepUse(GV);
      }

      const bool bAllowReplace = true;
      // SROA_Parameter_HLSL has no access to a domtree, if one is needed, it'll
      // be generated
      if (SROA_Helper::LowerMemcpy(GV, /*annotation*/ nullptr, typeSys, DL,
                                   nullptr /*DT */, bAllowReplace)) {
        continue;
      }

      // Flat Global vector if no dynamic vector indexing.
      bool bFlatVector = !hasDynamicVectorIndexing(GV);

      if (bFlatVector) {
        GVDbgOffset &dbgOffset = GVDbgOffsetMap[GV];
        GlobalVariable *baseGV = dbgOffset.base;
        // Disable scalarization of groupshared/const_static vector arrays
        if (isGroupShareOrConstStaticArray(baseGV))
          bFlatVector = false;
      }

      std::vector<Value *> Elts;
      bool SROAed = false;
      if (GlobalVariable *NewEltGV = dyn_cast_or_null<GlobalVariable>(
              TranslatePtrIfUsedByLoweredFn(GV, typeSys))) {
        GVDbgOffset dbgOffset = GVDbgOffsetMap[GV];
        // Don't need to update when skip SROA on base GV.
        if (NewEltGV == dbgOffset.base)
          continue;

        if (GV != NewEltGV) {
          GVDbgOffsetMap[NewEltGV] = dbgOffset;
          // Remove GV from GVDbgOffsetMap.
          GVDbgOffsetMap.erase(GV);
          if (GV != dbgOffset.base) {
            // Remove GV when it is replaced by NewEltGV and is not a base GV.
            GV->removeDeadConstantUsers();
            GV->eraseFromParent();
          }
          GV = NewEltGV;
        }
      } else {
        // SROA_Parameter_HLSL has no access to a domtree, if one is needed,
        // it'll be generated
        SROAed = SROA_Helper::DoScalarReplacement(
            GV, Elts, Builder, bFlatVector,
            // TODO: set precise.
            /*hasPrecise*/ false, typeSys, DL, DeadInsts, /*DT*/ nullptr);
      }

      if (SROAed) {
        GVDbgOffset dbgOffset = GVDbgOffsetMap[GV];
        unsigned offset = 0;
        // Push Elts into workList.
        for (auto iter = Elts.begin(); iter != Elts.end(); iter++) {
          WorkList.push(*iter);
          GlobalVariable *EltGV = cast<GlobalVariable>(*iter);
          if (bHasDbgInfo) {
            StringRef OriginEltName = EltGV->getName();
            StringRef OriginName = dbgOffset.base->getName();
            StringRef EltName = OriginEltName.substr(OriginName.size());
            StringRef EltParentName = OriginEltName.substr(0, OriginName.size());
            DXASSERT_LOCALVAR(EltParentName, EltParentName == OriginName, "parent name mismatch");
            EltNameMap[EltGV] = EltName;
          }
          GVDbgOffset &EltDbgOffset = GVDbgOffsetMap[EltGV];
          EltDbgOffset.base = dbgOffset.base;
          EltDbgOffset.debugOffset = dbgOffset.debugOffset + offset;
          unsigned size =
              DL.getTypeAllocSizeInBits(EltGV->getType()->getElementType());
          offset += size;
        }
        GV->removeDeadConstantUsers();
        // Now erase any instructions that were made dead while rewriting the
        // alloca.
        DeleteDeadInstructions(DeadInsts);
        ++NumReplaced;
      } else {
        // Add debug info for flattened globals.
        if (bHasDbgInfo && staticGVs.count(GV) == 0) {
          GVDbgOffset &dbgOffset = GVDbgOffsetMap[GV];
          DebugInfoFinder &Finder = HLM.GetOrCreateDebugInfoFinder();
          Type *Ty = GV->getType()->getElementType();
          unsigned size = DL.getTypeAllocSizeInBits(Ty);
          unsigned align = DL.getPrefTypeAlignment(Ty);
          HLModule::CreateElementGlobalVariableDebugInfo(
              dbgOffset.base, Finder, GV, size, align, dbgOffset.debugOffset,
              EltNameMap[GV]);
        }
      }
      // Remove GV from GVDbgOffsetMap.
      GVDbgOffsetMap.erase(GV);
    }
  }

  // Remove unused internal global.
  RemoveUnusedInternalGlobalVariable(M);
  // Cleanup memcpy for allocas and mark precise.
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    Cleanup(F, typeSys);
  }

  return true;
}
} // namespace

//===----------------------------------------------------------------------===//
// SRoA Helper
//===----------------------------------------------------------------------===//

/// RewriteGEP - Rewrite the GEP to be relative to new element when can find a
/// new element which is struct field. If cannot find, create new element GEPs
/// and try to rewrite GEP with new GEPS.
void SROA_Helper::RewriteForGEP(GEPOperator *GEP, IRBuilder<> &Builder) {

  assert(OldVal == GEP->getPointerOperand() && "");

  Value *NewPointer = nullptr;
  SmallVector<Value *, 8> NewArgs;

  gep_type_iterator GEPIt = gep_type_begin(GEP), E = gep_type_end(GEP);
  for (; GEPIt != E; ++GEPIt) {
    if (GEPIt->isStructTy()) {
      // must be const
      ConstantInt *IdxVal = dyn_cast<ConstantInt>(GEPIt.getOperand());
      assert(IdxVal->getLimitedValue() < NewElts.size() && "");
      NewPointer = NewElts[IdxVal->getLimitedValue()];
      // The idx is used for NewPointer, not part of newGEP idx,
      GEPIt++;
      break;
    } else if (GEPIt->isArrayTy()) {
      // Add array idx.
      NewArgs.push_back(GEPIt.getOperand());
    } else if (GEPIt->isPointerTy()) {
      // Add pointer idx.
      NewArgs.push_back(GEPIt.getOperand());
    } else if (GEPIt->isVectorTy()) {
      // Add vector idx.
      NewArgs.push_back(GEPIt.getOperand());
    } else {
      llvm_unreachable("should break from structTy");
    }
  }

  if (NewPointer) {
    // Struct split.
    // Add rest of idx.
    for (; GEPIt != E; ++GEPIt) {
      NewArgs.push_back(GEPIt.getOperand());
    }
    // If only 1 level struct, just use the new pointer.
    Value *NewGEP = NewPointer;
    if (NewArgs.size() > 1) {
      NewGEP = Builder.CreateInBoundsGEP(NewPointer, NewArgs);
      NewGEP->takeName(GEP);
    }

    assert(NewGEP->getType() == GEP->getType() && "type mismatch");
    
    GEP->replaceAllUsesWith(NewGEP);
  } else {
    // End at array of basic type.
    Type *Ty = GEP->getType()->getPointerElementType();
    if (Ty->isVectorTy() ||
        (Ty->isStructTy() && !dxilutil::IsHLSLObjectType(Ty)) ||
        Ty->isArrayTy()) {
      SmallVector<Value *, 8> NewArgs;
      NewArgs.append(GEP->idx_begin(), GEP->idx_end());

      SmallVector<Value *, 8> NewGEPs;
      // create new geps
      for (unsigned i = 0, e = NewElts.size(); i != e; ++i) {
        Value *NewGEP = Builder.CreateGEP(nullptr, NewElts[i], NewArgs);
        NewGEPs.emplace_back(NewGEP);
      }
      const bool bAllowReplace = isa<AllocaInst>(OldVal);
      if (!SROA_Helper::LowerMemcpy(GEP, /*annotation*/ nullptr, typeSys, DL, DT, bAllowReplace)) {
        SROA_Helper helper(GEP, NewGEPs, DeadInsts, typeSys, DL, DT);
        helper.RewriteForScalarRepl(GEP, Builder);
        for (Value *NewGEP : NewGEPs) {
          if (NewGEP->user_empty() && isa<Instruction>(NewGEP)) {
            // Delete unused newGEP.
            cast<Instruction>(NewGEP)->eraseFromParent();
          }
        }
      }
    } else {
      Value *vecIdx = NewArgs.back();
      if (ConstantInt *immVecIdx = dyn_cast<ConstantInt>(vecIdx)) {
        // Replace vecArray[arrayIdx][immVecIdx]
        // with scalarArray_immVecIdx[arrayIdx]

        // Pop the vecIdx.
        NewArgs.pop_back();
        Value *NewGEP = NewElts[immVecIdx->getLimitedValue()];
        if (NewArgs.size() > 1) {
          NewGEP = Builder.CreateInBoundsGEP(NewGEP, NewArgs);
          NewGEP->takeName(GEP);
        }

        assert(NewGEP->getType() == GEP->getType() && "type mismatch");

        GEP->replaceAllUsesWith(NewGEP);
      } else {
        // dynamic vector indexing.
        assert(0 && "should not reach here");
      }
    }
  }

  // Remove the use so that the caller can keep iterating over its other users
  DXASSERT(GEP->user_empty(), "All uses of the GEP should have been eliminated");
  if (isa<Instruction>(GEP)) {
    GEP->setOperand(GEP->getPointerOperandIndex(), UndefValue::get(GEP->getPointerOperand()->getType()));
    DeadInsts.push_back(GEP);
  }
  else {
    cast<Constant>(GEP)->destroyConstant();
  }
}

/// isVectorOrStructArray - Check if T is array of vector or struct.
static bool isVectorOrStructArray(Type *T) {
  if (!T->isArrayTy())
    return false;

  T = dxilutil::GetArrayEltTy(T);

  return T->isStructTy() || T->isVectorTy();
}

static void SimplifyStructValUsage(Value *StructVal, std::vector<Value *> Elts,
                                   SmallVectorImpl<Value *> &DeadInsts) {
  for (User *user : StructVal->users()) {
    if (ExtractValueInst *Extract = dyn_cast<ExtractValueInst>(user)) {
      DXASSERT(Extract->getNumIndices() == 1, "only support 1 index case");
      unsigned index = Extract->getIndices()[0];
      Value *Elt = Elts[index];
      Extract->replaceAllUsesWith(Elt);
      DeadInsts.emplace_back(Extract);
    } else if (InsertValueInst *Insert = dyn_cast<InsertValueInst>(user)) {
      DXASSERT(Insert->getNumIndices() == 1, "only support 1 index case");
      unsigned index = Insert->getIndices()[0];
      if (Insert->getAggregateOperand() == StructVal) {
        // Update field.
        std::vector<Value *> NewElts = Elts;
        NewElts[index] = Insert->getInsertedValueOperand();
        SimplifyStructValUsage(Insert, NewElts, DeadInsts);
      } else {
        // Insert to another bigger struct.
        IRBuilder<> Builder(Insert);
        Value *TmpStructVal = UndefValue::get(StructVal->getType());
        for (unsigned i = 0; i < Elts.size(); i++) {
          TmpStructVal =
              Builder.CreateInsertValue(TmpStructVal, Elts[i], {i});
        }
        Insert->replaceUsesOfWith(StructVal, TmpStructVal);
      }
    }
  }
}

/// RewriteForLoad - Replace OldVal with flattened NewElts in LoadInst.
void SROA_Helper::RewriteForLoad(LoadInst *LI) {
  Type *LIType = LI->getType();
  Type *ValTy = OldVal->getType()->getPointerElementType();
  IRBuilder<> Builder(LI);
  if (LIType->isVectorTy()) {
    // Replace:
    //   %res = load { 2 x i32 }* %alloc
    // with:
    //   %load.0 = load i32* %alloc.0
    //   %insert.0 insertvalue { 2 x i32 } zeroinitializer, i32 %load.0, 0
    //   %load.1 = load i32* %alloc.1
    //   %insert = insertvalue { 2 x i32 } %insert.0, i32 %load.1, 1
    Value *Insert = UndefValue::get(LIType);
    for (unsigned i = 0, e = NewElts.size(); i != e; ++i) {
      Value *Load = Builder.CreateLoad(NewElts[i], "load");
      Insert = Builder.CreateInsertElement(Insert, Load, i, "insert");
    }
    LI->replaceAllUsesWith(Insert);
  } else if (isCompatibleAggregate(LIType, ValTy)) {
    if (isVectorOrStructArray(LIType)) {
      // Replace:
      //   %res = load [2 x <2 x float>] * %alloc
      // with:
      //   %load.0 = load [4 x float]* %alloc.0
      //   %insert.0 insertvalue [4 x float] zeroinitializer,i32 %load.0,0
      //   %load.1 = load [4 x float]* %alloc.1
      //   %insert = insertvalue [4 x float] %insert.0, i32 %load.1, 1
      //  ...
      Type *i32Ty = Type::getInt32Ty(LIType->getContext());
      Value *zero = ConstantInt::get(i32Ty, 0);
      SmallVector<Value *, 8> idxList;
      idxList.emplace_back(zero);
      Value *newLd =
          LoadVectorOrStructArray(cast<ArrayType>(LIType), NewElts, idxList, Builder);
      LI->replaceAllUsesWith(newLd);
    } else {
      // Replace:
      //   %res = load { i32, i32 }* %alloc
      // with:
      //   %load.0 = load i32* %alloc.0
      //   %insert.0 insertvalue { i32, i32 } zeroinitializer, i32 %load.0,
      //   0
      //   %load.1 = load i32* %alloc.1
      //   %insert = insertvalue { i32, i32 } %insert.0, i32 %load.1, 1
      // (Also works for arrays instead of structs)
      Module *M = LI->getModule();
      Value *Insert = UndefValue::get(LIType);
      std::vector<Value *> LdElts(NewElts.size());
      for (unsigned i = 0, e = NewElts.size(); i != e; ++i) {
        Value *Ptr = NewElts[i];
        Type *Ty = Ptr->getType()->getPointerElementType();
        Value *Load = nullptr;
        if (!HLMatrixType::isa(Ty))
          Load = Builder.CreateLoad(Ptr, "load");
        else {
          // Generate Matrix Load.
          Load = HLModule::EmitHLOperationCall(
              Builder, HLOpcodeGroup::HLMatLoadStore,
              static_cast<unsigned>(HLMatLoadStoreOpcode::RowMatLoad), Ty,
              {Ptr}, *M);
        }
        LdElts[i] = Load;
        Insert = Builder.CreateInsertValue(Insert, Load, i, "insert");
      }
      LI->replaceAllUsesWith(Insert);
      if (LIType->isStructTy()) {
        SimplifyStructValUsage(Insert, LdElts, DeadInsts);
      }
    }
  } else {
    llvm_unreachable("other type don't need rewrite");
  }

  // Remove the use so that the caller can keep iterating over its other users
  LI->setOperand(LI->getPointerOperandIndex(), UndefValue::get(LI->getPointerOperand()->getType()));
  DeadInsts.push_back(LI);
}

/// RewriteForStore - Replace OldVal with flattened NewElts in StoreInst.
void SROA_Helper::RewriteForStore(StoreInst *SI) {
  Value *Val = SI->getOperand(0);
  Type *SIType = Val->getType();
  IRBuilder<> Builder(SI);
  Type *ValTy = OldVal->getType()->getPointerElementType();
  if (SIType->isVectorTy()) {
    // Replace:
    //   store <2 x float> %val, <2 x float>* %alloc
    // with:
    //   %val.0 = extractelement { 2 x float } %val, 0
    //   store i32 %val.0, i32* %alloc.0
    //   %val.1 = extractelement { 2 x float } %val, 1
    //   store i32 %val.1, i32* %alloc.1

    for (unsigned i = 0, e = NewElts.size(); i != e; ++i) {
      Value *Extract = Builder.CreateExtractElement(Val, i, Val->getName());
      Builder.CreateStore(Extract, NewElts[i]);
    }
  } else if (isCompatibleAggregate(SIType, ValTy)) {
    if (isVectorOrStructArray(SIType)) {
      // Replace:
      //   store [2 x <2 x i32>] %val, [2 x <2 x i32>]* %alloc, align 16
      // with:
      //   %val.0 = extractvalue [2 x <2 x i32>] %val, 0
      //   %all0c.0.0 = getelementptr inbounds [2 x i32], [2 x i32]* %alloc.0,
      //   i32 0, i32 0
      //   %val.0.0 = extractelement <2 x i32> %243, i64 0
      //   store i32 %val.0.0, i32* %all0c.0.0
      //   %alloc.1.0 = getelementptr inbounds [2 x i32], [2 x i32]* %alloc.1,
      //   i32 0, i32 0
      //   %val.0.1 = extractelement <2 x i32> %243, i64 1
      //   store i32 %val.0.1, i32* %alloc.1.0
      //   %val.1 = extractvalue [2 x <2 x i32>] %val, 1
      //   %alloc.0.0 = getelementptr inbounds [2 x i32], [2 x i32]* %alloc.0,
      //   i32 0, i32 1
      //   %val.1.0 = extractelement <2 x i32> %248, i64 0
      //   store i32 %val.1.0, i32* %alloc.0.0
      //   %all0c.1.1 = getelementptr inbounds [2 x i32], [2 x i32]* %alloc.1,
      //   i32 0, i32 1
      //   %val.1.1 = extractelement <2 x i32> %248, i64 1
      //   store i32 %val.1.1, i32* %all0c.1.1
      ArrayType *AT = cast<ArrayType>(SIType);
      Type *i32Ty = Type::getInt32Ty(SIType->getContext());
      Value *zero = ConstantInt::get(i32Ty, 0);
      SmallVector<Value *, 8> idxList;
      idxList.emplace_back(zero);
      StoreVectorOrStructArray(AT, Val, NewElts, idxList, Builder);
    } else {
      // Replace:
      //   store { i32, i32 } %val, { i32, i32 }* %alloc
      // with:
      //   %val.0 = extractvalue { i32, i32 } %val, 0
      //   store i32 %val.0, i32* %alloc.0
      //   %val.1 = extractvalue { i32, i32 } %val, 1
      //   store i32 %val.1, i32* %alloc.1
      // (Also works for arrays instead of structs)
      Module *M = SI->getModule();
      for (unsigned i = 0, e = NewElts.size(); i != e; ++i) {
        Value *Extract = Builder.CreateExtractValue(Val, i, Val->getName());
        if (!HLMatrixType::isa(Extract->getType())) {
          Builder.CreateStore(Extract, NewElts[i]);
        } else {
          // Generate Matrix Store.
          HLModule::EmitHLOperationCall(
              Builder, HLOpcodeGroup::HLMatLoadStore,
              static_cast<unsigned>(HLMatLoadStoreOpcode::RowMatStore),
              Extract->getType(), {NewElts[i], Extract}, *M);
        }
      }
    }
  } else {
    llvm_unreachable("other type don't need rewrite");
  }

  // Remove the use so that the caller can keep iterating over its other users
  SI->setOperand(SI->getPointerOperandIndex(), UndefValue::get(SI->getPointerOperand()->getType()));
  DeadInsts.push_back(SI);
}
/// RewriteMemIntrin - MI is a memcpy/memset/memmove from or to AI.
/// Rewrite it to copy or set the elements of the scalarized memory.
void SROA_Helper::RewriteMemIntrin(MemIntrinsic *MI, Value *OldV) {
  // If this is a memcpy/memmove, construct the other pointer as the
  // appropriate type.  The "Other" pointer is the pointer that goes to memory
  // that doesn't have anything to do with the alloca that we are promoting. For
  // memset, this Value* stays null.
  Value *OtherPtr = nullptr;
  unsigned MemAlignment = MI->getAlignment();
  if (MemTransferInst *MTI = dyn_cast<MemTransferInst>(MI)) { // memmove/memcopy
    if (OldV == MTI->getRawDest())
      OtherPtr = MTI->getRawSource();
    else {
      assert(OldV == MTI->getRawSource());
      OtherPtr = MTI->getRawDest();
    }
  }

  // If there is an other pointer, we want to convert it to the same pointer
  // type as AI has, so we can GEP through it safely.
  if (OtherPtr) {
    unsigned AddrSpace =
        cast<PointerType>(OtherPtr->getType())->getAddressSpace();

    // Remove bitcasts and all-zero GEPs from OtherPtr.  This is an
    // optimization, but it's also required to detect the corner case where
    // both pointer operands are referencing the same memory, and where
    // OtherPtr may be a bitcast or GEP that currently being rewritten.  (This
    // function is only called for mem intrinsics that access the whole
    // aggregate, so non-zero GEPs are not an issue here.)
    OtherPtr = OtherPtr->stripPointerCasts();

    // Copying the alloca to itself is a no-op: just delete it.
    if (OtherPtr == OldVal || OtherPtr == NewElts[0]) {
      // This code will run twice for a no-op memcpy -- once for each operand.
      // Put only one reference to MI on the DeadInsts list.
      for (SmallVectorImpl<Value *>::const_iterator I = DeadInsts.begin(),
                                                    E = DeadInsts.end();
           I != E; ++I)
        if (*I == MI)
          return;

      // Remove the uses so that the caller can keep iterating over its other users
      MI->setOperand(0, UndefValue::get(MI->getOperand(0)->getType()));
      MI->setOperand(1, UndefValue::get(MI->getOperand(1)->getType()));
      DeadInsts.push_back(MI);
      return;
    }

    // If the pointer is not the right type, insert a bitcast to the right
    // type.
    Type *NewTy =
        PointerType::get(OldVal->getType()->getPointerElementType(), AddrSpace);

    if (OtherPtr->getType() != NewTy)
      OtherPtr = new BitCastInst(OtherPtr, NewTy, OtherPtr->getName(), MI);
  }

  // Process each element of the aggregate.
  bool SROADest = MI->getRawDest() == OldV;

  Constant *Zero = Constant::getNullValue(Type::getInt32Ty(MI->getContext()));
  const DataLayout &DL = MI->getModule()->getDataLayout();

  for (unsigned i = 0, e = NewElts.size(); i != e; ++i) {
    // If this is a memcpy/memmove, emit a GEP of the other element address.
    Value *OtherElt = nullptr;
    unsigned OtherEltAlign = MemAlignment;

    if (OtherPtr) {
      Value *Idx[2] = {Zero,
                       ConstantInt::get(Type::getInt32Ty(MI->getContext()), i)};
      OtherElt = GetElementPtrInst::CreateInBounds(
          OtherPtr, Idx, OtherPtr->getName() + "." + Twine(i), MI);
      uint64_t EltOffset;
      PointerType *OtherPtrTy = cast<PointerType>(OtherPtr->getType());
      Type *OtherTy = OtherPtrTy->getElementType();
      if (StructType *ST = dyn_cast<StructType>(OtherTy)) {
        EltOffset = DL.getStructLayout(ST)->getElementOffset(i);
      } else {
        Type *EltTy = cast<SequentialType>(OtherTy)->getElementType();
        EltOffset = DL.getTypeAllocSize(EltTy) * i;
      }

      // The alignment of the other pointer is the guaranteed alignment of the
      // element, which is affected by both the known alignment of the whole
      // mem intrinsic and the alignment of the element.  If the alignment of
      // the memcpy (f.e.) is 32 but the element is at a 4-byte offset, then the
      // known alignment is just 4 bytes.
      OtherEltAlign = (unsigned)MinAlign(OtherEltAlign, EltOffset);
    }

    Value *EltPtr = NewElts[i];
    Type *EltTy = cast<PointerType>(EltPtr->getType())->getElementType();

    // If we got down to a scalar, insert a load or store as appropriate.
    if (EltTy->isSingleValueType()) {
      if (isa<MemTransferInst>(MI)) {
        if (SROADest) {
          // From Other to Alloca.
          Value *Elt = new LoadInst(OtherElt, "tmp", false, OtherEltAlign, MI);
          new StoreInst(Elt, EltPtr, MI);
        } else {
          // From Alloca to Other.
          Value *Elt = new LoadInst(EltPtr, "tmp", MI);
          new StoreInst(Elt, OtherElt, false, OtherEltAlign, MI);
        }
        continue;
      }
      assert(isa<MemSetInst>(MI));

      // If the stored element is zero (common case), just store a null
      // constant.
      Constant *StoreVal;
      if (ConstantInt *CI = dyn_cast<ConstantInt>(MI->getArgOperand(1))) {
        if (CI->isZero()) {
          StoreVal = Constant::getNullValue(EltTy); // 0.0, null, 0, <0,0>
        } else {
          // If EltTy is a vector type, get the element type.
          Type *ValTy = EltTy->getScalarType();

          // Construct an integer with the right value.
          unsigned EltSize = DL.getTypeSizeInBits(ValTy);
          APInt OneVal(EltSize, CI->getZExtValue());
          APInt TotalVal(OneVal);
          // Set each byte.
          for (unsigned i = 0; 8 * i < EltSize; ++i) {
            TotalVal = TotalVal.shl(8);
            TotalVal |= OneVal;
          }

          // Convert the integer value to the appropriate type.
          StoreVal = ConstantInt::get(CI->getContext(), TotalVal);
          if (ValTy->isPointerTy())
            StoreVal = ConstantExpr::getIntToPtr(StoreVal, ValTy);
          else if (ValTy->isFloatingPointTy())
            StoreVal = ConstantExpr::getBitCast(StoreVal, ValTy);
          assert(StoreVal->getType() == ValTy && "Type mismatch!");

          // If the requested value was a vector constant, create it.
          if (EltTy->isVectorTy()) {
            unsigned NumElts = cast<VectorType>(EltTy)->getNumElements();
            StoreVal = ConstantVector::getSplat(NumElts, StoreVal);
          }
        }
        new StoreInst(StoreVal, EltPtr, MI);
        continue;
      }
      // Otherwise, if we're storing a byte variable, use a memset call for
      // this element.
    }

    unsigned EltSize = DL.getTypeAllocSize(EltTy);
    if (!EltSize)
      continue;

    IRBuilder<> Builder(MI);

    // Finally, insert the meminst for this element.
    if (isa<MemSetInst>(MI)) {
      Builder.CreateMemSet(EltPtr, MI->getArgOperand(1), EltSize,
                           MI->isVolatile());
    } else {
      assert(isa<MemTransferInst>(MI));
      Value *Dst = SROADest ? EltPtr : OtherElt; // Dest ptr
      Value *Src = SROADest ? OtherElt : EltPtr; // Src ptr

      if (isa<MemCpyInst>(MI))
        Builder.CreateMemCpy(Dst, Src, EltSize, OtherEltAlign,
                             MI->isVolatile());
      else
        Builder.CreateMemMove(Dst, Src, EltSize, OtherEltAlign,
                              MI->isVolatile());
    }
  }

  // Remove the use so that the caller can keep iterating over its other users
  MI->setOperand(0, UndefValue::get(MI->getOperand(0)->getType()));
  if (isa<MemTransferInst>(MI))
    MI->setOperand(1, UndefValue::get(MI->getOperand(1)->getType()));
  DeadInsts.push_back(MI);
}

void SROA_Helper::RewriteBitCast(BitCastInst *BCI) {
  // Unused bitcast may be leftover from temporary memcpy
  if (BCI->use_empty()) {
    BCI->eraseFromParent();
    return;
  }

  Type *DstTy = BCI->getType();
  Value *Val = BCI->getOperand(0);
  Type *SrcTy = Val->getType();
  if (!DstTy->isPointerTy()) {
    assert(0 && "Type mismatch.");
    return;
  }
  if (!SrcTy->isPointerTy()) {
    assert(0 && "Type mismatch.");
    return;
  }

  DstTy = DstTy->getPointerElementType();
  SrcTy = SrcTy->getPointerElementType();

  if (!DstTy->isStructTy()) {
    // This is an llvm.lifetime.* intrinsic. Replace bitcast by a bitcast for each element.
    SmallVector<IntrinsicInst*, 16> ToReplace;

    DXASSERT(onlyUsedByLifetimeMarkers(BCI),
             "expected struct bitcast to only be used by lifetime intrinsics");

    for (User *User : BCI->users()) {
      IntrinsicInst *Intrin = cast<IntrinsicInst>(User);
      ToReplace.push_back(Intrin);
    }

    const DataLayout &DL = BCI->getModule()->getDataLayout();
    for (IntrinsicInst *Intrin : ToReplace) {
      IRBuilder<> Builder(Intrin);

      for (Value *Elt : NewElts) {
        assert(Elt->getType()->isPointerTy());
        Type     *ElPtrTy = Elt->getType();
        Type     *ElTy    = ElPtrTy->getPointerElementType();
        Value    *SizeV   = Builder.getInt64( DL.getTypeAllocSize(ElTy) );
        Value    *Ptr     = Builder.CreateBitCast(Elt, Builder.getInt8PtrTy());
        Value    *Args[]  = {SizeV, Ptr};
        CallInst *C       = Builder.CreateCall(Intrin->getCalledFunction(), Args);
        C->setDoesNotThrow();
      }

      assert(Intrin->use_empty());
      Intrin->eraseFromParent();
    }

    assert(BCI->use_empty());
    BCI->eraseFromParent();
    return;
  }

  if (!SrcTy->isStructTy()) {
    assert(0 && "Type mismatch.");
    return;
  }
  // Only support bitcast to parent struct type.
  StructType *DstST = cast<StructType>(DstTy);
  StructType *SrcST = cast<StructType>(SrcTy);

  bool bTypeMatch = false;
  unsigned level = 0;
  while (SrcST) {
    level++;
    Type *EltTy = SrcST->getElementType(0);
    if (EltTy == DstST) {
      bTypeMatch = true;
      break;
    }
    SrcST = dyn_cast<StructType>(EltTy);
  }

  if (!bTypeMatch) {
    // If the layouts match, just replace the type
    SrcST = cast<StructType>(SrcTy);
    if (SrcST->isLayoutIdentical(DstST)) {
      BCI->mutateType(Val->getType());
      BCI->replaceAllUsesWith(Val);
      BCI->eraseFromParent();
      return;
    }
    assert(0 && "Type mismatch.");
    return;
  }

  std::vector<Value*> idxList(level+1);
  ConstantInt *zeroIdx = ConstantInt::get(Type::getInt32Ty(Val->getContext()), 0);
  for (unsigned i=0;i<(level+1);i++)
    idxList[i] = zeroIdx;

  IRBuilder<> Builder(BCI);
  Builder.AllowFolding = false; // We need an Instruction, so make sure we don't get a constant
  Instruction *GEP = cast<Instruction>(Builder.CreateInBoundsGEP(Val, idxList));
  BCI->replaceAllUsesWith(GEP);
  BCI->eraseFromParent();

  IRBuilder<> GEPBuilder(GEP);
  RewriteForGEP(cast<GEPOperator>(GEP), GEPBuilder);
}

/// RewriteCallArg - For Functions which don't flat,
///                  replace OldVal with alloca and
///                  copy in copy out data between alloca and flattened NewElts
///                  in CallInst.
void SROA_Helper::RewriteCallArg(CallInst *CI, unsigned ArgIdx, bool bIn,
                                 bool bOut) {
  Function *F = CI->getParent()->getParent();
  IRBuilder<> AllocaBuilder(dxilutil::FindAllocaInsertionPt(F));
  const DataLayout &DL = F->getParent()->getDataLayout();

  Value *userTyV = CI->getArgOperand(ArgIdx);
  PointerType *userTy = cast<PointerType>(userTyV->getType());
  Type *userTyElt = userTy->getElementType();
  Value *Alloca = AllocaBuilder.CreateAlloca(userTyElt);
  IRBuilder<> Builder(CI);
  if (bIn) {
    MemCpyInst *cpy = cast<MemCpyInst>(Builder.CreateMemCpy(
        Alloca, userTyV, DL.getTypeAllocSize(userTyElt), false));
    RewriteMemIntrin(cpy, cpy->getRawSource());
  }
  CI->setArgOperand(ArgIdx, Alloca);
  if (bOut) {
    Builder.SetInsertPoint(CI->getNextNode());
    MemCpyInst *cpy = cast<MemCpyInst>(Builder.CreateMemCpy(
        userTyV, Alloca, DL.getTypeAllocSize(userTyElt), false));
    RewriteMemIntrin(cpy, cpy->getRawSource());
  }
}

// Flatten matching OldVal arg to NewElts, optionally loading values (loadElts).
// Does not replace or clean up old CallInst.
static CallInst *CreateFlattenedHLIntrinsicCall(
    CallInst *CI, Value* OldVal, ArrayRef<Value*> NewElts, bool loadElts) {
  HLOpcodeGroup group = GetHLOpcodeGroupByName(CI->getCalledFunction());
  Function *F = CI->getCalledFunction();
  DXASSERT_NOMSG(group == HLOpcodeGroup::HLIntrinsic);
  unsigned opcode = GetHLOpcode(CI);
  IRBuilder<> Builder(CI);

  SmallVector<Value *, 4> flatArgs;
  for (Value *arg : CI->arg_operands()) {
    if (arg == OldVal) {
      for (Value *Elt : NewElts) {
        if (loadElts && Elt->getType()->isPointerTy())
          Elt = Builder.CreateLoad(Elt);
        flatArgs.emplace_back(Elt);
      }
    } else
      flatArgs.emplace_back(arg);
  }

  SmallVector<Type *, 4> flatParamTys;
  for (Value *arg : flatArgs)
    flatParamTys.emplace_back(arg->getType());
  FunctionType *flatFuncTy =
      FunctionType::get(CI->getType(), flatParamTys, false);
  Function *flatF =
    GetOrCreateHLFunction(*F->getParent(), flatFuncTy, group, opcode,
                          F->getAttributes().getFnAttributes());

  return Builder.CreateCall(flatF, flatArgs);
}

static CallInst *RewriteWithFlattenedHLIntrinsicCall(
    CallInst *CI, Value* OldVal, ArrayRef<Value*> NewElts, bool loadElts) {
  CallInst *flatCI = CreateFlattenedHLIntrinsicCall(
    CI, OldVal, NewElts, /*loadElts*/loadElts);
  CI->replaceAllUsesWith(flatCI);
  // Clear CI operands so we don't try to translate old call again
  for (auto& opit : CI->operands())
    opit.set(UndefValue::get(opit->getType()));
  return flatCI;
}

/// RewriteCall - Replace OldVal with flattened NewElts in CallInst.
void SROA_Helper::RewriteCall(CallInst *CI) {
  HLOpcodeGroup group = GetHLOpcodeGroupByName(CI->getCalledFunction());
  if (group != HLOpcodeGroup::NotHL) {
    unsigned opcode = GetHLOpcode(CI);
    if (group == HLOpcodeGroup::HLIntrinsic) {
      IntrinsicOp IOP = static_cast<IntrinsicOp>(opcode);
      switch (IOP) {
      case IntrinsicOp::MOP_Append: {
        // Buffer Append already expand in code gen.
        // Must be OutputStream Append here.
        // Every Elt has a pointer type.
        // For Append, this is desired, so don't load.
        RewriteWithFlattenedHLIntrinsicCall(CI, OldVal, NewElts, /*loadElts*/false);
        DeadInsts.push_back(CI);
      } break;
      case IntrinsicOp::IOP_TraceRay: {
        if (OldVal ==
            CI->getArgOperand(HLOperandIndex::kTraceRayRayDescOpIdx)) {
          RewriteCallArg(CI, HLOperandIndex::kTraceRayRayDescOpIdx,
                         /*bIn*/ true, /*bOut*/ false);
        } else {
          DXASSERT(OldVal ==
                       CI->getArgOperand(HLOperandIndex::kTraceRayPayLoadOpIdx),
                   "else invalid TraceRay");
          RewriteCallArg(CI, HLOperandIndex::kTraceRayPayLoadOpIdx,
                         /*bIn*/ true, /*bOut*/ true);
        }
      } break;
      case IntrinsicOp::IOP_ReportHit: {
        RewriteCallArg(CI, HLOperandIndex::kReportIntersectionAttributeOpIdx,
                       /*bIn*/ true, /*bOut*/ false);
      } break;
      case IntrinsicOp::IOP_CallShader: {
        RewriteCallArg(CI, HLOperandIndex::kCallShaderPayloadOpIdx,
                       /*bIn*/ true, /*bOut*/ true);
      } break;
      case IntrinsicOp::MOP_TraceRayInline: {
        if (OldVal ==
            CI->getArgOperand(HLOperandIndex::kTraceRayInlineRayDescOpIdx)) {
          RewriteWithFlattenedHLIntrinsicCall(CI, OldVal, NewElts, /*loadElts*/true);
          DeadInsts.push_back(CI);
          break;
        }
      }
      __fallthrough;
      default:
        // RayQuery this pointer replacement.
        if (OldVal->getType()->isPointerTy() &&
            CI->getNumArgOperands() >= HLOperandIndex::kHandleOpIdx &&
            OldVal == CI->getArgOperand(HLOperandIndex::kHandleOpIdx) &&
            dxilutil::IsHLSLRayQueryType(
              OldVal->getType()->getPointerElementType())) {
          // For RayQuery methods, we want to replace the RayQuery this pointer
          // with a load and use of the underlying handle value.
          // This will allow elimination of RayQuery types earlier.
          RewriteWithFlattenedHLIntrinsicCall(CI, OldVal, NewElts, /*loadElts*/true);
          DeadInsts.push_back(CI);
          break;
        }
        DXASSERT(0, "cannot flatten hlsl intrinsic.");
      }
    }
    // TODO: check other high level dx operations if need to.
  } else {
    DXASSERT(0, "should done at inline");
  }
}

/// RewriteForAddrSpaceCast - Rewrite the AddrSpaceCast, either ConstExpr or Inst.
void SROA_Helper::RewriteForAddrSpaceCast(Value *CE,
                                          IRBuilder<> &Builder) {
  SmallVector<Value *, 8> NewCasts;
  // create new AddrSpaceCast.
  for (unsigned i = 0, e = NewElts.size(); i != e; ++i) {
    Value *NewCast = Builder.CreateAddrSpaceCast(
        NewElts[i],
        PointerType::get(NewElts[i]->getType()->getPointerElementType(),
                         CE->getType()->getPointerAddressSpace()));
    NewCasts.emplace_back(NewCast);
  }
  SROA_Helper helper(CE, NewCasts, DeadInsts, typeSys, DL, DT);
  helper.RewriteForScalarRepl(CE, Builder);

  // Remove the use so that the caller can keep iterating over its other users
  DXASSERT(CE->user_empty(), "All uses of the addrspacecast should have been eliminated");
  if (Instruction *I = dyn_cast<Instruction>(CE))
    I->eraseFromParent();
  else
    cast<Constant>(CE)->destroyConstant();
}

/// RewriteForConstExpr - Rewrite the GEP which is ConstantExpr.
void SROA_Helper::RewriteForConstExpr(ConstantExpr *CE, IRBuilder<> &Builder) {
  if (GEPOperator *GEP = dyn_cast<GEPOperator>(CE)) {
    if (OldVal == GEP->getPointerOperand()) {
      // Flatten GEP.
      RewriteForGEP(GEP, Builder);
      return;
    }
  }
  if (CE->getOpcode() == Instruction::AddrSpaceCast) {
    if (OldVal == CE->getOperand(0)) {
      // Flatten AddrSpaceCast.
      RewriteForAddrSpaceCast(CE, Builder);
      return;
    }
  }
  for (Value::use_iterator UI = CE->use_begin(), E = CE->use_end(); UI != E;) {
    Use &TheUse = *UI++;
    if (Instruction *I = dyn_cast<Instruction>(TheUse.getUser())) {
      IRBuilder<> tmpBuilder(I);
      // Replace CE with constInst.
      Instruction *tmpInst = CE->getAsInstruction();
      tmpBuilder.Insert(tmpInst);
      TheUse.set(tmpInst);
    }
    else {
      RewriteForConstExpr(cast<ConstantExpr>(TheUse.getUser()), Builder);
    }
  }

  // Remove the use so that the caller can keep iterating over its other users
  DXASSERT(CE->user_empty(), "All uses of the constantexpr should have been eliminated");
  CE->destroyConstant();
}
/// RewriteForScalarRepl - OldVal is being split into NewElts, so rewrite
/// users of V, which references it, to use the separate elements.
void SROA_Helper::RewriteForScalarRepl(Value *V, IRBuilder<> &Builder) {
  // Don't iterate upon the uses explicitly because we'll be removing them,
  // and potentially adding new ones (if expanding memcpys) during the iteration.
  Use* PrevUse = nullptr;
  while (!V->use_empty()) {
    Use &TheUse = *V->use_begin();

    DXASSERT_LOCALVAR(PrevUse, &TheUse != PrevUse,
      "Infinite loop while SROA'ing value, use isn't getting eliminated.");
    PrevUse = &TheUse;

    // Each of these must either call ->eraseFromParent()
    // or null out the use of V so that we make progress.
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(TheUse.getUser())) {
      RewriteForConstExpr(CE, Builder);
    }
    else {
      Instruction *User = cast<Instruction>(TheUse.getUser());
      if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(User)) {
        IRBuilder<> Builder(GEP);
        RewriteForGEP(cast<GEPOperator>(GEP), Builder);
      } else if (LoadInst *ldInst = dyn_cast<LoadInst>(User))
        RewriteForLoad(ldInst);
      else if (StoreInst *stInst = dyn_cast<StoreInst>(User))
        RewriteForStore(stInst);
      else if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(User))
        RewriteMemIntrin(MI, V);
      else if (CallInst *CI = dyn_cast<CallInst>(User)) 
        RewriteCall(CI);
      else if (BitCastInst *BCI = dyn_cast<BitCastInst>(User))
        RewriteBitCast(BCI);
      else if (AddrSpaceCastInst *CI = dyn_cast<AddrSpaceCastInst>(User)) {
        RewriteForAddrSpaceCast(CI, Builder);
      } else {
        assert(0 && "not support.");
      }
    }
  }
}

static ArrayType *CreateNestArrayTy(Type *FinalEltTy,
                                    ArrayRef<ArrayType *> nestArrayTys) {
  Type *newAT = FinalEltTy;
  for (auto ArrayTy = nestArrayTys.rbegin(), E=nestArrayTys.rend(); ArrayTy != E;
       ++ArrayTy)
    newAT = ArrayType::get(newAT, (*ArrayTy)->getNumElements());
  return cast<ArrayType>(newAT);
}

/// DoScalarReplacement - Split V into AllocaInsts with Builder and save the new AllocaInsts into Elts.
/// Then do SROA on V.
bool SROA_Helper::DoScalarReplacement(Value *V, std::vector<Value *> &Elts,
                                      Type *&BrokenUpTy, uint64_t &NumInstances,
                                      IRBuilder<> &Builder, bool bFlatVector,
                                      bool hasPrecise, DxilTypeSystem &typeSys,
                                      const DataLayout &DL,
                                      SmallVector<Value *, 32> &DeadInsts,
                                      DominatorTree *DT) {
  DEBUG(dbgs() << "Found inst to SROA: " << *V << '\n');
  Type *Ty = V->getType();
  // Skip none pointer types.
  if (!Ty->isPointerTy())
    return false;

  Ty = Ty->getPointerElementType();
  // Skip none aggregate types.
  if (!Ty->isAggregateType())
    return false;
  // Skip matrix types.
  if (HLMatrixType::isa(Ty))
    return false;

  IRBuilder<> AllocaBuilder(dxilutil::FindAllocaInsertionPt(Builder.GetInsertPoint()));

  if (StructType *ST = dyn_cast<StructType>(Ty)) {
    // Skip HLSL object types and RayQuery.
    if (dxilutil::IsHLSLObjectType(ST)) {
      return false;
    }

    BrokenUpTy = ST;
    NumInstances = 1;

    unsigned numTypes = ST->getNumContainedTypes();
    Elts.reserve(numTypes);
    DxilStructAnnotation *SA = typeSys.GetStructAnnotation(ST);
    // Skip empty struct.
    if (SA && SA->IsEmptyStruct())
      return true;
    for (int i = 0, e = numTypes; i != e; ++i) {
      AllocaInst *NA = AllocaBuilder.CreateAlloca(ST->getContainedType(i), nullptr, V->getName() + "." + Twine(i));
      bool markPrecise = hasPrecise;
      if (SA) {
        DxilFieldAnnotation &FA = SA->GetFieldAnnotation(i);
        markPrecise |= FA.IsPrecise();
      }
      if (markPrecise)
        HLModule::MarkPreciseAttributeWithMetadata(NA);
      Elts.push_back(NA);
    }
  } else {
    ArrayType *AT = cast<ArrayType>(Ty);
    if (AT->getNumContainedTypes() == 0) {
      // Skip case like [0 x %struct].
      return false;
    }
    Type *ElTy = AT->getElementType();
    SmallVector<ArrayType *, 4> nestArrayTys;
    nestArrayTys.emplace_back(AT);
    NumInstances = AT->getNumElements();
    // support multi level of array
    while (ElTy->isArrayTy()) {
      ArrayType *ElAT = cast<ArrayType>(ElTy);
      nestArrayTys.emplace_back(ElAT);
      NumInstances *= ElAT->getNumElements();
      ElTy = ElAT->getElementType();
    }
    BrokenUpTy = ElTy;

    if (ElTy->isStructTy() &&
        // Skip Matrix type.
        !HLMatrixType::isa(ElTy)) {
      if (!dxilutil::IsHLSLObjectType(ElTy)) {
        // for array of struct
        // split into arrays of struct elements
        StructType *ElST = cast<StructType>(ElTy);
        unsigned numTypes = ElST->getNumContainedTypes();
        Elts.reserve(numTypes);
        DxilStructAnnotation *SA = typeSys.GetStructAnnotation(ElST);
        // Skip empty struct.
        if (SA && SA->IsEmptyStruct())
          return true;
        for (int i = 0, e = numTypes; i != e; ++i) {
          AllocaInst *NA = AllocaBuilder.CreateAlloca(
              CreateNestArrayTy(ElST->getContainedType(i), nestArrayTys),
              nullptr, V->getName() + "." + Twine(i));
          bool markPrecise = hasPrecise;
          if (SA) {
            DxilFieldAnnotation &FA = SA->GetFieldAnnotation(i);
            markPrecise |= FA.IsPrecise();
          }
          if (markPrecise)
            HLModule::MarkPreciseAttributeWithMetadata(NA);
          Elts.push_back(NA);
        }
      } else {
        // For local resource array which not dynamic indexing,
        // split it.
        if (dxilutil::HasDynamicIndexing(V) ||
            // Only support 1 dim split.
            nestArrayTys.size() > 1)
          return false;
        BrokenUpTy = AT;
        NumInstances = 1;
        for (int i = 0, e = AT->getNumElements(); i != e; ++i) {
          AllocaInst *NA = AllocaBuilder.CreateAlloca(ElTy, nullptr,
                                                V->getName() + "." + Twine(i));
          Elts.push_back(NA);
        }
      }
    } else if (ElTy->isVectorTy()) {
      // Skip vector if required.
      if (!bFlatVector)
        return false;

      // for array of vector
      // split into arrays of scalar
      VectorType *ElVT = cast<VectorType>(ElTy);
      BrokenUpTy = ElVT;
      Elts.reserve(ElVT->getNumElements());

      ArrayType *scalarArrayTy = CreateNestArrayTy(ElVT->getElementType(), nestArrayTys);

      for (int i = 0, e = ElVT->getNumElements(); i != e; ++i) {
        AllocaInst *NA = AllocaBuilder.CreateAlloca(scalarArrayTy, nullptr,
                           V->getName() + "." + Twine(i));
        if (hasPrecise)
          HLModule::MarkPreciseAttributeWithMetadata(NA);
        Elts.push_back(NA);
      }
    } else
      // Skip array of basic types.
      return false;
  }
  
  // Now that we have created the new alloca instructions, rewrite all the
  // uses of the old alloca.
  SROA_Helper helper(V, Elts, DeadInsts, typeSys, DL, DT);
  helper.RewriteForScalarRepl(V, Builder);

  return true;
}

static Constant *GetEltInit(Type *Ty, Constant *Init, unsigned idx,
                            Type *EltTy) {
  if (isa<UndefValue>(Init))
    return UndefValue::get(EltTy);

  if (dyn_cast<StructType>(Ty)) {
    return Init->getAggregateElement(idx);
  } else if (dyn_cast<VectorType>(Ty)) {
    return Init->getAggregateElement(idx);
  } else {
    ArrayType *AT = cast<ArrayType>(Ty);
    ArrayType *EltArrayTy = cast<ArrayType>(EltTy);
    std::vector<Constant *> Elts;
    if (!AT->getElementType()->isArrayTy()) {
      for (unsigned i = 0; i < AT->getNumElements(); i++) {
        // Get Array[i]
        Constant *InitArrayElt = Init->getAggregateElement(i);
        // Get Array[i].idx
        InitArrayElt = InitArrayElt->getAggregateElement(idx);
        Elts.emplace_back(InitArrayElt);
      }
      return ConstantArray::get(EltArrayTy, Elts);
    } else {
      Type *EltTy = AT->getElementType();
      ArrayType *NestEltArrayTy = cast<ArrayType>(EltArrayTy->getElementType());
      // Nested array.
      for (unsigned i = 0; i < AT->getNumElements(); i++) {
        // Get Array[i]
        Constant *InitArrayElt = Init->getAggregateElement(i);
        // Get Array[i].idx
        InitArrayElt = GetEltInit(EltTy, InitArrayElt, idx, NestEltArrayTy);
        Elts.emplace_back(InitArrayElt);
      }
      return ConstantArray::get(EltArrayTy, Elts);
    }
  }
}

unsigned SROA_Helper::GetEltAlign(unsigned ValueAlign, const DataLayout &DL,
                                  Type *EltTy, unsigned Offset) {
  unsigned Alignment = ValueAlign;
  if (ValueAlign == 0) {
    // The minimum alignment which users can rely on when the explicit
    // alignment is omitted or zero is that required by the ABI for this
    // type.
    Alignment = DL.getABITypeAlignment(EltTy);
  }
  return MinAlign(Alignment, Offset);
}

/// DoScalarReplacement - Split V into AllocaInsts with Builder and save the new AllocaInsts into Elts.
/// Then do SROA on V.
bool SROA_Helper::DoScalarReplacement(GlobalVariable *GV,
                                      std::vector<Value *> &Elts,
                                      IRBuilder<> &Builder, bool bFlatVector,
                                      bool hasPrecise, DxilTypeSystem &typeSys,
                                      const DataLayout &DL,
                                      SmallVector<Value *, 32> &DeadInsts,
                                      DominatorTree *DT) {
  DEBUG(dbgs() << "Found inst to SROA: " << *GV << '\n');
  Type *Ty = GV->getType();
  // Skip none pointer types.
  if (!Ty->isPointerTy())
    return false;

  Ty = Ty->getPointerElementType();
  // Skip none aggregate types.
  if (!Ty->isAggregateType() && !bFlatVector)
    return false;
  // Skip basic types.
  if (Ty->isSingleValueType() && !Ty->isVectorTy())
    return false;
  // Skip matrix types.
  if (HLMatrixType::isa(Ty))
    return false;

  Module *M = GV->getParent();
  Constant *Init = GV->hasInitializer() ? GV->getInitializer() : UndefValue::get(Ty);
  bool isConst = GV->isConstant();

  GlobalVariable::ThreadLocalMode TLMode = GV->getThreadLocalMode();
  unsigned AddressSpace = GV->getType()->getAddressSpace();
  GlobalValue::LinkageTypes linkage = GV->getLinkage();
  const unsigned Alignment = GV->getAlignment();
  if (StructType *ST = dyn_cast<StructType>(Ty)) {
    // Skip HLSL object types.
    if (dxilutil::IsHLSLObjectType(ST))
      return false;
    unsigned numTypes = ST->getNumContainedTypes();
    Elts.reserve(numTypes);
    unsigned Offset = 0;
    //DxilStructAnnotation *SA = typeSys.GetStructAnnotation(ST);
    for (int i = 0, e = numTypes; i != e; ++i) {
      Type *EltTy = ST->getElementType(i);
      Constant *EltInit = GetEltInit(Ty, Init, i, EltTy);
      GlobalVariable *EltGV = new llvm::GlobalVariable(
          *M, ST->getContainedType(i), /*IsConstant*/ isConst, linkage,
          /*InitVal*/ EltInit, GV->getName() + "." + Twine(i),
          /*InsertBefore*/ nullptr, TLMode, AddressSpace);
      EltGV->setAlignment(GetEltAlign(Alignment, DL, EltTy, Offset));
      Offset += DL.getTypeAllocSize(EltTy);
      //DxilFieldAnnotation &FA = SA->GetFieldAnnotation(i);
      // TODO: set precise.
      // if (hasPrecise || FA.IsPrecise())
      //  HLModule::MarkPreciseAttributeWithMetadata(NA);
      Elts.push_back(EltGV);
    }
  } else if (VectorType *VT = dyn_cast<VectorType>(Ty)) {
    // TODO: support dynamic indexing on vector by change it to array.
    unsigned numElts = VT->getNumElements();
    Elts.reserve(numElts);
    Type *EltTy = VT->getElementType();
    unsigned Offset = 0;
    //DxilStructAnnotation *SA = typeSys.GetStructAnnotation(ST);
    for (int i = 0, e = numElts; i != e; ++i) {
      Constant *EltInit = GetEltInit(Ty, Init, i, EltTy);
      GlobalVariable *EltGV = new llvm::GlobalVariable(
          *M, EltTy, /*IsConstant*/ isConst, linkage,
          /*InitVal*/ EltInit, GV->getName() + "." + Twine(i),
          /*InsertBefore*/ nullptr, TLMode, AddressSpace);
      EltGV->setAlignment(GetEltAlign(Alignment, DL, EltTy, Offset));
      Offset += DL.getTypeAllocSize(EltTy);
      //DxilFieldAnnotation &FA = SA->GetFieldAnnotation(i);
      // TODO: set precise.
      // if (hasPrecise || FA.IsPrecise())
      //  HLModule::MarkPreciseAttributeWithMetadata(NA);
      Elts.push_back(EltGV);
    }
  } else {
    ArrayType *AT = cast<ArrayType>(Ty);
    if (AT->getNumContainedTypes() == 0) {
      // Skip case like [0 x %struct].
      return false;
    }
    Type *ElTy = AT->getElementType();
    SmallVector<ArrayType *, 4> nestArrayTys;

    nestArrayTys.emplace_back(AT);
    // support multi level of array
    while (ElTy->isArrayTy()) {
      ArrayType *ElAT = cast<ArrayType>(ElTy);
      nestArrayTys.emplace_back(ElAT);
      ElTy = ElAT->getElementType();
    }

    if (ElTy->isStructTy() &&
        // Skip Matrix and Resource type.
        !HLMatrixType::isa(ElTy) &&
        !dxilutil::IsHLSLResourceType(ElTy)) {
      // for array of struct
      // split into arrays of struct elements
      StructType *ElST = cast<StructType>(ElTy);
      unsigned numTypes = ElST->getNumContainedTypes();
      Elts.reserve(numTypes);
      unsigned Offset = 0;
      //DxilStructAnnotation *SA = typeSys.GetStructAnnotation(ElST);
      for (int i = 0, e = numTypes; i != e; ++i) {
        Type *EltTy =
            CreateNestArrayTy(ElST->getContainedType(i), nestArrayTys);
        Constant *EltInit = GetEltInit(Ty, Init, i, EltTy);
        GlobalVariable *EltGV = new llvm::GlobalVariable(
            *M, EltTy, /*IsConstant*/ isConst, linkage,
            /*InitVal*/ EltInit, GV->getName() + "." + Twine(i),
            /*InsertBefore*/ nullptr, TLMode, AddressSpace);

        EltGV->setAlignment(GetEltAlign(Alignment, DL, EltTy, Offset));
        Offset += DL.getTypeAllocSize(EltTy);
        //DxilFieldAnnotation &FA = SA->GetFieldAnnotation(i);
        // TODO: set precise.
        // if (hasPrecise || FA.IsPrecise())
        //  HLModule::MarkPreciseAttributeWithMetadata(NA);
        Elts.push_back(EltGV);
      }
    } else if (ElTy->isVectorTy()) {
      // Skip vector if required.
      if (!bFlatVector)
        return false;

      // for array of vector
      // split into arrays of scalar
      VectorType *ElVT = cast<VectorType>(ElTy);
      Elts.reserve(ElVT->getNumElements());

      ArrayType *scalarArrayTy =
          CreateNestArrayTy(ElVT->getElementType(), nestArrayTys);
      unsigned Offset = 0;

      for (int i = 0, e = ElVT->getNumElements(); i != e; ++i) {
        Constant *EltInit = GetEltInit(Ty, Init, i, scalarArrayTy);
        GlobalVariable *EltGV = new llvm::GlobalVariable(
            *M, scalarArrayTy, /*IsConstant*/ isConst, linkage,
            /*InitVal*/ EltInit, GV->getName() + "." + Twine(i),
            /*InsertBefore*/ nullptr, TLMode, AddressSpace);
        // TODO: set precise.
        // if (hasPrecise)
        //  HLModule::MarkPreciseAttributeWithMetadata(NA);
        EltGV->setAlignment(GetEltAlign(Alignment, DL, scalarArrayTy, Offset));
        Offset += DL.getTypeAllocSize(scalarArrayTy);
        Elts.push_back(EltGV);
      }
    } else
      // Skip array of basic types.
      return false;
  }

  // Now that we have created the new alloca instructions, rewrite all the
  // uses of the old alloca.
  SROA_Helper helper(GV, Elts, DeadInsts, typeSys, DL, DT);
  helper.RewriteForScalarRepl(GV, Builder);

  return true;
}

static void ReplaceConstantWithInst(Constant *C, Value *V, IRBuilder<> &Builder) {
  Function *F = Builder.GetInsertBlock()->getParent();
  for (auto it = C->user_begin(); it != C->user_end(); ) {
    User *U = *(it++);
    if (Instruction *I = dyn_cast<Instruction>(U)) {
      if (I->getParent()->getParent() != F)
        continue;
      I->replaceUsesOfWith(C, V);
    } else {
      // Skip unused ConstantExpr.
      if (U->user_empty())
        continue;
      ConstantExpr *CE = cast<ConstantExpr>(U);
      Instruction *Inst = CE->getAsInstruction();
      Builder.Insert(Inst);
      Inst->replaceUsesOfWith(C, V);
      ReplaceConstantWithInst(CE, Inst, Builder);
    }
  }
  C->removeDeadConstantUsers();
}

static void ReplaceUnboundedArrayUses(Value *V, Value *Src) {
  for (auto it = V->user_begin(); it != V->user_end(); ) {
    User *U = *(it++);
    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(U)) {
      SmallVector<Value *, 4> idxList(GEP->idx_begin(), GEP->idx_end());
      // Must set the insert point to the GEP itself (instead of the memcpy),
      // because the indices might not dominate the memcpy.
      IRBuilder<> Builder(GEP);
      Value *NewGEP = Builder.CreateGEP(Src, idxList);
      GEP->replaceAllUsesWith(NewGEP);
    } else if (BitCastInst *BC = dyn_cast<BitCastInst>(U)) {
      BC->setOperand(0, Src);
    } else {
      DXASSERT(false, "otherwise unbounded array used in unexpected instruction");
    }
  }
}

static bool IsUnboundedArrayMemcpy(Type *destTy, Type *srcTy) {
  return (destTy->isArrayTy() && srcTy->isArrayTy()) &&
    (destTy->getArrayNumElements() == 0 || srcTy->getArrayNumElements() == 0);
}

static bool ArePointersToStructsOfIdenticalLayouts(Type *DstTy, Type *SrcTy) {
  if (!SrcTy->isPointerTy() || !DstTy->isPointerTy())
    return false;
  DstTy = DstTy->getPointerElementType();
  SrcTy = SrcTy->getPointerElementType();
  if (!SrcTy->isStructTy() || !DstTy->isStructTy())
    return false;
  StructType *DstST = cast<StructType>(DstTy);
  StructType *SrcST = cast<StructType>(SrcTy);
  return SrcST->isLayoutIdentical(DstST);
}

static std::vector<Value *> GetConstValueIdxList(IRBuilder<> &builder,
  std::vector<unsigned> idxlist) {
  std::vector<Value *> idxConstList;
  for (unsigned idx : idxlist) {
    idxConstList.push_back(ConstantInt::get(builder.getInt32Ty(), idx));
  }
  return idxConstList;
}

static void CopyElementsOfStructsWithIdenticalLayout(
  IRBuilder<> &builder, Value *destPtr, Value *srcPtr, Type *ty,
  std::vector<unsigned>& idxlist) {
  if (ty->isStructTy()) {
    for (unsigned i = 0; i < ty->getStructNumElements(); i++) {
      idxlist.push_back(i);
      CopyElementsOfStructsWithIdenticalLayout(
        builder, destPtr, srcPtr, ty->getStructElementType(i), idxlist);
      idxlist.pop_back();
    }
  }
  else if (ty->isArrayTy()) {
    for (unsigned i = 0; i < ty->getArrayNumElements(); i++) {
      idxlist.push_back(i);
      CopyElementsOfStructsWithIdenticalLayout(
        builder, destPtr, srcPtr, ty->getArrayElementType(), idxlist);
      idxlist.pop_back();
    }
  }
  else if (ty->isIntegerTy() || ty->isFloatTy() || ty->isDoubleTy() ||
    ty->isHalfTy() || ty->isVectorTy()) {
    Value *srcGEP =
      builder.CreateInBoundsGEP(srcPtr, GetConstValueIdxList(builder, idxlist));
    Value *destGEP =
      builder.CreateInBoundsGEP(destPtr, GetConstValueIdxList(builder, idxlist));
    LoadInst *LI = builder.CreateLoad(srcGEP);
    builder.CreateStore(LI, destGEP);
  }
  else {
    DXASSERT(0, "encountered unsupported type when copying elements of identical structs.");
  }
}

static void removeLifetimeUsers(Value *V) {
  std::set<Value*> users(V->users().begin(), V->users().end());
  for (Value *U : users) {
    if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(U)) {
      if (II->getIntrinsicID() == Intrinsic::lifetime_start ||
          II->getIntrinsicID() == Intrinsic::lifetime_end) {
        II->eraseFromParent();
      }
    } else if (isa<BitCastInst>(U) ||
               isa<AddrSpaceCastInst>(U) ||
               isa<GetElementPtrInst>(U)) {
      // Recurse into bitcast, addrspacecast, GEP.
      removeLifetimeUsers(U);
      if (U->use_empty())
        cast<Instruction>(U)->eraseFromParent();
    }
  }
}

// Conservatively remove all lifetime users of both source and target.
// Otherwise, wrong lifetimes could be inherited either way.
// TODO: We should be merging the lifetimes. For convenience, just remove them
//       for now to be safe.
static void updateLifetimeForReplacement(Value *From, Value *To)
{
    removeLifetimeUsers(From);
    removeLifetimeUsers(To);
}

static bool DominateAllUsers(Instruction *I, Value *V, DominatorTree *DT);

namespace {
void replaceScalarArrayGEPWithVectorArrayGEP(User *GEP, Value *VectorArray,
                                             IRBuilder<> &Builder,
                                             unsigned sizeInDwords) {
  gep_type_iterator GEPIt = gep_type_begin(GEP), E = gep_type_end(GEP);

  Value *PtrOffset = GEPIt.getOperand();
  ++GEPIt;
  Value *ArrayIdx = GEPIt.getOperand();
  ++GEPIt;
  ArrayIdx = Builder.CreateAdd(PtrOffset, ArrayIdx);
  DXASSERT_LOCALVAR(E, GEPIt == E, "invalid gep on scalar array");

  unsigned shift = 2;
  unsigned mask = 0x3;
  switch (sizeInDwords) {
  case 2:
    shift = 1;
    mask = 1;
    break;
  case 1:
    shift = 2;
    mask = 0x3;
    break;
  default:
    DXASSERT(0, "invalid scalar size");
    break;
  }

  Value *VecIdx = Builder.CreateLShr(ArrayIdx, shift);
  Value *VecPtr = Builder.CreateGEP(
      VectorArray, {ConstantInt::get(VecIdx->getType(), 0), VecIdx});
  Value *CompIdx = Builder.CreateAnd(ArrayIdx, mask);
  Value *NewGEP = Builder.CreateGEP(
      VecPtr, {ConstantInt::get(CompIdx->getType(), 0), CompIdx});
  GEP->replaceAllUsesWith(NewGEP);
}

void replaceScalarArrayWithVectorArray(Value *ScalarArray, Value *VectorArray,
                                       MemCpyInst *MC, unsigned sizeInDwords) {
  LLVMContext &Context = ScalarArray->getContext();
  // All users should be element type.
  // Replace users of AI or GV.
  for (auto it = ScalarArray->user_begin(); it != ScalarArray->user_end();) {
    User *U = *(it++);
    if (U->user_empty())
      continue;
    if (BitCastInst *BCI = dyn_cast<BitCastInst>(U)) {
      BCI->setOperand(0, VectorArray);
      continue;
    }

    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(U)) {
      IRBuilder<> Builder(Context);
      if (GEPOperator *GEP = dyn_cast<GEPOperator>(U)) {
        // NewGEP must be GEPOperator too.
        // No instruction will be build.
        replaceScalarArrayGEPWithVectorArrayGEP(U, VectorArray, Builder,
                                                sizeInDwords);
      } else if (CE->getOpcode() == Instruction::AddrSpaceCast) {
        Value *NewAddrSpaceCast = Builder.CreateAddrSpaceCast(
            VectorArray,
            PointerType::get(VectorArray->getType()->getPointerElementType(),
                             CE->getType()->getPointerAddressSpace()));
        replaceScalarArrayWithVectorArray(CE, NewAddrSpaceCast, MC,
                                          sizeInDwords);
      } else if (CE->hasOneUse() && CE->user_back() == MC) {
        continue;
      } else {
        DXASSERT(0, "not implemented");
      }
    } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(U)) {
      IRBuilder<> Builder(GEP);
      replaceScalarArrayGEPWithVectorArrayGEP(U, VectorArray, Builder,
                                              sizeInDwords);
      GEP->eraseFromParent();
    } else {
      DXASSERT(0, "not implemented");
    }
  }
}

// For pattern like
// float4 cb[16];
// float v[64] = cb;
bool tryToReplaceCBVec4ArrayToScalarArray(Value *V, Type *TyV, Value *Src,
                                          Type *TySrc, MemCpyInst *MC,
                                          const DataLayout &DL) {
  if (!isCBVec4ArrayToScalarArray(TyV, Src, TySrc, DL))
    return false;

  ArrayType *AT = cast<ArrayType>(TyV);
  Type *EltTy = AT->getElementType();
  unsigned sizeInBits = DL.getTypeSizeInBits(EltTy);
  // Convert array of float4 to array of float.
  replaceScalarArrayWithVectorArray(V, Src, MC, sizeInBits >> 5);
  return true;
}

} // namespace

static bool ReplaceMemcpy(Value *V, Value *Src, MemCpyInst *MC,
                          DxilFieldAnnotation *annotation, DxilTypeSystem &typeSys,
                          const DataLayout &DL, DominatorTree *DT) {
  // If the only user of the src and dst is the memcpy,
  // this memcpy was probably produced by splitting another.
  // Regardless, the goal here is to replace, not remove the memcpy
  // we won't have enough information to determine if we can do that before mem2reg
  if (V != Src && V->hasOneUse() && Src->hasOneUse())
    return false;

  // If the source of the memcpy (Src) doesn't dominate all users of dest (V),
  // full replacement isn't possible without complicated PHI insertion
  // This will likely replace with ld/st which will be replaced in mem2reg
  if (Instruction *SrcI = dyn_cast<Instruction>(Src))
    if (!DominateAllUsers(SrcI, V, DT))
      return false;

  Type *TyV = V->getType()->getPointerElementType();
  Type *TySrc = Src->getType()->getPointerElementType();
  if (Constant *C = dyn_cast<Constant>(V)) {
    updateLifetimeForReplacement(V, Src);
    if (TyV == TySrc) {
      if (isa<Constant>(Src)) {
        V->replaceAllUsesWith(Src);
      } else {
        // Replace Constant with a non-Constant.
        IRBuilder<> Builder(MC);
        ReplaceConstantWithInst(C, Src, Builder);
      }
    } else {
      // Try convert special pattern for cbuffer which copy array of float4 to
      // array of float.
      if (!tryToReplaceCBVec4ArrayToScalarArray(V, TyV, Src, TySrc, MC, DL)) {
        IRBuilder<> Builder(MC);
        Src = Builder.CreateBitCast(Src, V->getType());
        ReplaceConstantWithInst(C, Src, Builder);
      }
    }
  } else {
    if (TyV == TySrc) {
      if (V != Src) {
        updateLifetimeForReplacement(V, Src);
        V->replaceAllUsesWith(Src);
      }
    } else if (!IsUnboundedArrayMemcpy(TyV, TySrc)) {
      Value* DestVal = MC->getRawDest();
      Value* SrcVal = MC->getRawSource();
      if (!isa<BitCastInst>(SrcVal) || !isa<BitCastInst>(DestVal)) {
        DXASSERT(0, "Encountered unexpected instruction sequence");
        return false;
      }

      BitCastInst *DestBCI = cast<BitCastInst>(DestVal);
      BitCastInst *SrcBCI = cast<BitCastInst>(SrcVal);

      Type* DstTy = DestBCI->getSrcTy();
      Type *SrcTy = SrcBCI->getSrcTy();
      if (ArePointersToStructsOfIdenticalLayouts(DstTy, SrcTy)) {
        const DataLayout &DL = SrcBCI->getModule()->getDataLayout();
        unsigned SrcSize = DL.getTypeAllocSize(
            SrcBCI->getOperand(0)->getType()->getPointerElementType());
        unsigned MemcpySize = cast<ConstantInt>(MC->getLength())->getZExtValue();
        if (SrcSize != MemcpySize) {
          DXASSERT(0, "Cannot handle partial memcpy");
          return false;
        }

        if (DestBCI->hasOneUse() && SrcBCI->hasOneUse()) {
          IRBuilder<> Builder(MC);
          StructType *srcStTy = cast<StructType>(
              SrcBCI->getOperand(0)->getType()->getPointerElementType());
          std::vector<unsigned> idxlist = {0};
          CopyElementsOfStructsWithIdenticalLayout(
              Builder, DestBCI->getOperand(0), SrcBCI->getOperand(0), srcStTy,
              idxlist);
        }
      } else {
        if (DstTy == SrcTy) {
          Value *DstPtr = DestBCI->getOperand(0);
          Value *SrcPtr = SrcBCI->getOperand(0);
          if (isa<GEPOperator>(DstPtr) || isa<GEPOperator>(SrcPtr)) {
            MemcpySplitter::SplitMemCpy(MC, DL, annotation, typeSys);
            return true;
          } else {
            updateLifetimeForReplacement(V, Src);
            DstPtr->replaceAllUsesWith(SrcPtr);
          }
        } else {
          DXASSERT(0, "Can't handle structs of different layouts");
          return false;
        }
      }
    } else {
      updateLifetimeForReplacement(V, Src);
      DXASSERT(IsUnboundedArrayMemcpy(TyV, TySrc), "otherwise mismatched types in memcpy are not unbounded array");
      ReplaceUnboundedArrayUses(V, Src);
    }
  }

  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(Src)) {
    // For const GV, if has stored, mark as non-constant.
    if (GV->isConstant()) {
      hlutil::PointerStatus PS(GV, 0, /*bLdStOnly*/ true);
      PS.analyze(typeSys, /*bStructElt*/ false);
      if (PS.HasStored())
        GV->setConstant(false);
    }
  }
  Value *RawDest = MC->getOperand(0);
  Value *RawSrc = MC->getOperand(1);
  MC->eraseFromParent();
  if (Instruction *I = dyn_cast<Instruction>(RawDest)) {
    if (I->user_empty())
      I->eraseFromParent();
  }
  if (Instruction *I = dyn_cast<Instruction>(RawSrc)) {
    if (I->user_empty())
      I->eraseFromParent();
  }

  return true;
}

static bool ReplaceUseOfZeroInitEntry(Instruction *I, Value *V) {
  BasicBlock *BB = I->getParent();
  Function *F = I->getParent()->getParent();
  for (auto U = V->user_begin(); U != V->user_end(); ) {
    Instruction *UI = dyn_cast<Instruction>(*(U++));
    if (!UI)
      continue;

    if (UI->getParent()->getParent() != F)
      continue;

    if (isa<GetElementPtrInst>(UI) || isa<BitCastInst>(UI)) {
      if (!ReplaceUseOfZeroInitEntry(I, UI))
        return false;
      else
        continue;
    }
    if (BB != UI->getParent() || UI == I)
      continue;
    // I is the last inst in the block after split.
    // Any inst in current block is before I.
    if (LoadInst *LI = dyn_cast<LoadInst>(UI)) {
      LI->replaceAllUsesWith(ConstantAggregateZero::get(LI->getType()));
      LI->eraseFromParent();
      continue;
    }
    return false;
  }
  return true;
}

// If a V user is dominated by memcpy (I),
//    skip it - memcpy dest can simply alias to src for this user.
// If the V user may follow the memcpy (I),
//    return false - memcpy dest not safe to replace with src.
// Otherwise,
//    replace use with zeroinitializer.
static bool ReplaceUseOfZeroInit(Instruction *I, Value *V,
                                 DominatorTree &DT,
                                 SmallPtrSet<BasicBlock*, 8> &Reachable) {
  BasicBlock *BB = I->getParent();
  Function *F = I->getParent()->getParent();
  for (auto U = V->user_begin(); U != V->user_end(); ) {
    Instruction *UI = dyn_cast<Instruction>(*(U++));
    if (!UI || UI == I)
      continue;
    if (UI->getParent()->getParent() != F)
      continue;

    // Skip properly dominated users
    if (DT.properlyDominates(BB, UI->getParent()))
      continue;

    // If user is found in memcpy successor list
    // then the user is not safe to replace with zeroinitializer.
    if (Reachable.count(UI->getParent()))
      return false;

    // Remaining cases are where I:
    // - is at the end of the same block
    // - does not precede UI on any path
    if (isa<GetElementPtrInst>(UI) || isa<BitCastInst>(UI)) {
      if (ReplaceUseOfZeroInit(I, UI, DT, Reachable))
        continue;
    } else if (LoadInst *LI = dyn_cast<LoadInst>(UI)) {
      LI->replaceAllUsesWith(ConstantAggregateZero::get(LI->getType()));
      LI->eraseFromParent();
      continue;
    }
    return false;
  }
  return true;
}

// Recursively collect all successors of BB and BB's successors.
// BB will not be in set unless it's reachable through its successors.
static void CollectReachableBBs(BasicBlock *BB, SmallPtrSet<BasicBlock*, 8> &Reachable) {
  for (auto S : successors(BB)) {
    if (Reachable.insert(S).second)
      CollectReachableBBs(S, Reachable);
  }
}

// When zero initialized GV has only one define, all uses before the def should
// use zero.
static bool ReplaceUseOfZeroInitBeforeDef(Instruction *I, GlobalVariable *GV) {
  BasicBlock *BB = I->getParent();
  Function *F = I->getParent()->getParent();
  // Make sure I is the last inst for BB.
  BasicBlock *NewBB = nullptr;
  if (I != BB->getTerminator())
    NewBB = BB->splitBasicBlock(I->getNextNode());

  bool bSuccess = false;
  if (&F->getEntryBlock() == I->getParent()) {
    bSuccess = ReplaceUseOfZeroInitEntry(I, GV);
  } else {
    DominatorTree DT;
    DT.recalculate(*F);
    SmallPtrSet<BasicBlock*, 8> Reachable;
    CollectReachableBBs(BB, Reachable);
    bSuccess = ReplaceUseOfZeroInit(I, GV, DT, Reachable);
  }

  // Re-merge basic block to keep things simpler
  if (NewBB)
    llvm::MergeBlockIntoPredecessor(NewBB);

  return bSuccess;
}

// Use `DT` to trace all users and make sure `I`'s BB dominates them all
static bool DominateAllUsersDom(Instruction *I, Value *V, DominatorTree *DT) {
  BasicBlock *BB = I->getParent();
  Function *F = I->getParent()->getParent();
  for (auto U = V->user_begin(); U != V->user_end(); ) {
    Instruction *UI = dyn_cast<Instruction>(*(U++));
    // If not an instruction or from a differnt function, nothing to check, move along.
    if (!UI || UI->getParent()->getParent() != F)
      continue;

    if (!DT->dominates(BB, UI->getParent()))
      return false;

    if (isa<GetElementPtrInst>(UI) || isa<BitCastInst>(UI)) {
      if (!DominateAllUsersDom(I, UI, DT))
        return false;
    }
  }
  return true;
}

// Determine if `I` dominates all the users of `V`
static bool DominateAllUsers(Instruction *I, Value *V, DominatorTree *DT) {
  Function *F = I->getParent()->getParent();

  // The Entry Block dominates everything, trivially true
  if (&F->getEntryBlock() == I->getParent())
    return true;

  if (!DT) {
    DominatorTree TempDT;
    TempDT.recalculate(*F);
    return DominateAllUsersDom(I, V, &TempDT);
  } else {
    return DominateAllUsersDom(I, V, DT);
  }
}

// Return resource properties if handle value is HLAnnotateHandle
static DxilResourceProperties GetResPropsFromHLAnnotateHandle(Value *handle) {
  if (CallInst *handleCI = dyn_cast<CallInst>(handle)) {
    hlsl::HLOpcodeGroup group =
        hlsl::GetHLOpcodeGroup(handleCI->getCalledFunction());
    if (group == HLOpcodeGroup::HLAnnotateHandle) {
      Constant *Props = cast<Constant>(handleCI->getArgOperand(
                            HLOperandIndex::kAnnotateHandleResourcePropertiesOpIdx));
      return resource_helper::loadPropsFromConstant(*Props);
    }
  }
  return {};
}

static bool isReadOnlyResSubscriptOrLoad(CallInst *PtrCI) {
  hlsl::HLOpcodeGroup group =
      hlsl::GetHLOpcodeGroup(PtrCI->getCalledFunction());
  if (group == HLOpcodeGroup::HLSubscript) {
    HLSubscriptOpcode opcode =
        static_cast<HLSubscriptOpcode>(hlsl::GetHLOpcode(PtrCI));
    if (opcode == HLSubscriptOpcode::CBufferSubscript) {
      // Ptr from CBuffer is readonly.
      return true;
    } else if (opcode == HLSubscriptOpcode::DefaultSubscript) {
      DxilResourceProperties RP = GetResPropsFromHLAnnotateHandle(
          PtrCI->getArgOperand(HLOperandIndex::kSubscriptObjectOpIdx));
      if (RP.getResourceClass() == DXIL::ResourceClass::SRV)
        return true;
    }
  } else if (group == HLOpcodeGroup::HLIntrinsic) {
    IntrinsicOp hlOpcode = (IntrinsicOp)GetHLOpcode(PtrCI);
    if (hlOpcode == IntrinsicOp::MOP_Load) {
      // This is templated BAB.Load<UDT>() case.
      DxilResourceProperties RP = GetResPropsFromHLAnnotateHandle(
        PtrCI->getArgOperand(HLOperandIndex::kHandleOpIdx));
      if (RP.getResourceClass() == DXIL::ResourceClass::SRV)
        return true;
    }
  }
  return false;
}

bool SROA_Helper::LowerMemcpy(Value *V, DxilFieldAnnotation *annotation,
                              DxilTypeSystem &typeSys, const DataLayout &DL,
                              DominatorTree *DT, bool bAllowReplace) {
  Type *Ty = V->getType();
  if (!Ty->isPointerTy()) {
    return false;
  }
  // Get access status and collect memcpy uses.
  // if MemcpyOnce, replace with dest with src if dest is not out param.
  // else flat memcpy.
  unsigned size = DL.getTypeAllocSize(Ty->getPointerElementType());
  hlutil::PointerStatus PS(V, size, /*bLdStOnly*/ false);
  const bool bStructElt = false;
  PS.analyze(typeSys, bStructElt);

  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(V)) {
    if (GV->hasInitializer() && !isa<UndefValue>(GV->getInitializer())) {
      if (PS.storedType == hlutil::PointerStatus::StoredType::NotStored) {
        PS.storedType = hlutil::PointerStatus::StoredType::InitializerStored;
      } else if (PS.storedType ==
                 hlutil::PointerStatus::StoredType::MemcopyDestOnce) {
        // For single mem store, if the store does not dominate all users.
        // Mark it as Stored.
        // In cases like:
        // struct A { float4 x[25]; };
        // A a;
        // static A a2;
        // void set(A aa) { aa = a; }
        // call set inside entry function then use a2.
        if (isa<ConstantAggregateZero>(GV->getInitializer())) {
          Instruction * Memcpy = PS.StoringMemcpy;
          if (!ReplaceUseOfZeroInitBeforeDef(Memcpy, GV)) {
            PS.storedType = hlutil::PointerStatus::StoredType::Stored;
          }
        }
      } else {
        PS.storedType = hlutil::PointerStatus::StoredType::Stored;
      }
    }
  }

  if (bAllowReplace && !PS.HasMultipleAccessingFunctions) {
    if (PS.storedType == hlutil::PointerStatus::StoredType::MemcopyDestOnce &&
        // Skip argument for input argument has input value, it is not dest once anymore.
        !isa<Argument>(V)) {
      // Replace with src of memcpy.
      MemCpyInst *MC = PS.StoringMemcpy;
      if (MC->getSourceAddressSpace() == MC->getDestAddressSpace()) {
        Value *Src = MC->getOperand(1);
        // Only remove one level bitcast generated from inline.
        if (BitCastOperator *BC = dyn_cast<BitCastOperator>(Src))
          Src = BC->getOperand(0);

        if (GEPOperator *GEP = dyn_cast<GEPOperator>(Src)) {
          // For GEP, the ptr could have other GEP read/write.
          // Only scan one GEP is not enough.
          Value *Ptr = GEP->getPointerOperand();
          while (GEPOperator *NestedGEP = dyn_cast<GEPOperator>(Ptr))
            Ptr = NestedGEP->getPointerOperand();

          if (CallInst *PtrCI = dyn_cast<CallInst>(Ptr)) {
            if (isReadOnlyResSubscriptOrLoad(PtrCI)) {
              // Ptr from CBuffer/SRV is safe.
              if (ReplaceMemcpy(V, Src, MC, annotation, typeSys, DL, DT)) {
                if (V->user_empty())
                  return true;
                return LowerMemcpy(V, annotation, typeSys, DL, DT, bAllowReplace);
              }
            }
          }
        } else if (!isa<CallInst>(Src)) {
          // Resource ptr should not be replaced.
          // Need to make sure src not updated after current memcpy.
          // Check Src only have 1 store now.
          hlutil::PointerStatus SrcPS(Src, size, /*bLdStOnly*/ false);
          SrcPS.analyze(typeSys, bStructElt);
          if (SrcPS.storedType != hlutil::PointerStatus::StoredType::Stored) {
            if (ReplaceMemcpy(V, Src, MC, annotation, typeSys, DL, DT)) {
              if (V->user_empty())
                return true;
              return LowerMemcpy(V, annotation, typeSys, DL, DT, bAllowReplace);
            }
          }
        }
      }
    } else if (PS.loadedType ==
               hlutil::PointerStatus::LoadedType::MemcopySrcOnce) {
      // Replace dst of memcpy.
      MemCpyInst *MC = PS.LoadingMemcpy;
      if (MC->getSourceAddressSpace() == MC->getDestAddressSpace()) {
        Value *Dest = MC->getOperand(0);
        // Only remove one level bitcast generated from inline.
        if (BitCastOperator *BC = dyn_cast<BitCastOperator>(Dest))
          Dest = BC->getOperand(0);
        // For GEP, the ptr could have other GEP read/write.
        // Only scan one GEP is not enough.
        // And resource ptr should not be replaced.
        // Nor should (output) argument ptr be replaced.
        if (!isa<GEPOperator>(Dest) && !isa<CallInst>(Dest) &&
            !isa<BitCastOperator>(Dest) && !isa<Argument>(Dest)) {
          // Need to make sure Dest not updated after current memcpy.
          // Check Dest only have 1 store now.
          hlutil::PointerStatus DestPS(Dest, size, /*bLdStOnly*/ false);
          DestPS.analyze(typeSys, bStructElt);
          if (DestPS.storedType != hlutil::PointerStatus::StoredType::Stored) {
            if (ReplaceMemcpy(Dest, V, MC, annotation, typeSys, DL, DT)) {
              // V still needs to be flattened.
              // Lower memcpy come from Dest.
              return LowerMemcpy(V, annotation, typeSys, DL, DT, bAllowReplace);
            }
          }
        }
      }
    }
  }

  for (MemCpyInst *MC : PS.memcpySet) {
    MemcpySplitter::SplitMemCpy(MC, DL, annotation, typeSys);
  }
  return false;
}

/// MarkEmptyStructUsers - Add instruction related to Empty struct to DeadInsts.
void SROA_Helper::MarkEmptyStructUsers(Value *V, SmallVector<Value *, 32> &DeadInsts) {
  UndefValue *undef = UndefValue::get(V->getType());
  for (auto itU = V->user_begin(), E = V->user_end(); itU != E;) {
    Value *U = *(itU++);
    // Kill memcpy, set operands to undef for call and ret, and recurse
    if (MemCpyInst *MC = dyn_cast<MemCpyInst>(U)) {
      DeadInsts.emplace_back(MC);
    } else if (CallInst *CI = dyn_cast<CallInst>(U)) {
      for (auto &operand : CI->operands()) {
        if (operand == V)
          operand.set(undef);
      }
    } else if (ReturnInst *Ret = dyn_cast<ReturnInst>(U)) {
      Ret->setOperand(0, undef);
    } else if (isa<Constant>(U) || isa<GetElementPtrInst>(U) ||
               isa<BitCastInst>(U) || isa<LoadInst>(U) || isa<StoreInst>(U)) {
      // Recurse users
      MarkEmptyStructUsers(U, DeadInsts);
    } else {
      DXASSERT(false, "otherwise, recursing unexpected empty struct user");
    }
  }

  if (Instruction *I = dyn_cast<Instruction>(V)) {
    // Only need to add no use inst here.
    // DeleteDeadInst will delete everything.
    if (I->user_empty())
      DeadInsts.emplace_back(I);
  }
}

bool SROA_Helper::IsEmptyStructType(Type *Ty, DxilTypeSystem &typeSys) {
  if (isa<ArrayType>(Ty))
    Ty = Ty->getArrayElementType();

  if (StructType *ST = dyn_cast<StructType>(Ty)) {
    if (!HLMatrixType::isa(Ty)) {
      DxilStructAnnotation *SA = typeSys.GetStructAnnotation(ST);
      if (SA && SA->IsEmptyStruct())
        return true;
    }
  }
  return false;
}

// Recursively search all loads and stores of Ptr, and record all the scopes
// they are in.
static void FindAllScopesOfLoadsAndStores(Value *Ptr, std::unordered_set<MDNode *> *OutScopes) {
  for (User *U : Ptr->users()) {
    if (isa<GEPOperator>(U) || isa<BitCastOperator>(U)) {
      FindAllScopesOfLoadsAndStores(U, OutScopes);
      continue;
    }
    Instruction *I = dyn_cast<Instruction>(U);
    if (!I)
      continue;
    DebugLoc DL = I->getDebugLoc();
    if (!DL)
      continue;

    if (isa<LoadInst>(I) || isa<StoreInst>(I) || isa<MemCpyInst>(I) ||
        isa<CallInst>(I)) // Could be some arbitrary HL op
    {
      DILocation *loc = DL.get();
      while (loc) {
        DILocalScope *scope = dyn_cast<DILocalScope>(loc->getScope());
        while (scope) {
          OutScopes->insert(scope);
          if (auto lexicalScope = dyn_cast<DILexicalBlockBase>(scope))
            scope = lexicalScope->getScope();
          else if (isa<DISubprogram>(scope))
            break;
        }
        loc = loc->getInlinedAt();
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Delete all dbg.declare instructions for allocas that are never touched in
// scopes. This could greatly improve compilation speed and binary size at the
// cost of in-scope but unused (part of) variables being invisible in certain
// functions. For example:
//
//     struct Context {
//        float a, b, c;
//     };
//     void bar(inout Context ctx) {
//       ctx.b = ctx.a * 2;
//     }
//     void foo(inout Context ctx) {
//       ctx.a = 10;
//       bar(ctx);
//     }
//     
//     float main() : SV_Target {
//       Context ctx = (Context)0;
//       foo(ctx);
//       return ctx.a + ctx.b + ctx.c;
//     }
//
// Before running this, shader would generate dbg.declare for members 'a', 'b',
// and 'c' for variable 'ctx' in every scope ('main', 'foo', and 'bar'). In
// the call stack with 'foo' and 'bar', member 'c' is never used in any way, so
// it's a waste to generate dbg.declare for member 'c' for the 'ctx' in scope
// 'foo' and scope 'bar', so they can be removed.
//===----------------------------------------------------------------------===//
static void DeleteOutOfScopeDebugInfo(Function &F) {
  if (!llvm::hasDebugInfo(*F.getParent()))
    return;

  std::unordered_set<MDNode *> Scopes;
  for (Instruction &I : F.getEntryBlock()) {
    auto AI = dyn_cast<AllocaInst>(&I);
    if (!AI)
      continue;

    Scopes.clear();
    FindAllScopesOfLoadsAndStores(AI, &Scopes);
    for (auto it = hlsl::dxilutil::mdv_users_begin(AI),
              end = hlsl::dxilutil::mdv_users_end(AI);
        it != end;)
    {
      User *U = *(it++);
      if (DbgDeclareInst *DDI = dyn_cast<DbgDeclareInst>(U)) {
        DILocalVariable *var = DDI->getVariable();
        DILocalScope *scope = var->getScope();
        if (!Scopes.count(scope)) {
          DDI->eraseFromParent();
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// SROA on function parameters.
//===----------------------------------------------------------------------===//

static void LegalizeDxilInputOutputs(Function *F,
                                     DxilFunctionAnnotation *EntryAnnotation,
                                     const DataLayout &DL,
                                     DxilTypeSystem &typeSys);
static void InjectReturnAfterNoReturnPreserveOutput(HLModule &HLM);

namespace {
class SROA_Parameter_HLSL : public ModulePass {
  HLModule *m_pHLModule;

public:
  static char ID; // Pass identification, replacement for typeid
  explicit SROA_Parameter_HLSL() : ModulePass(ID) {}
  StringRef getPassName() const override { return "SROA Parameter HLSL"; }
  static void RewriteBitcastWithIdenticalStructs(Function *F);
  static void RewriteBitcastWithIdenticalStructs(BitCastInst *BCI);
  static bool DeleteSimpleStoreOnlyAlloca(AllocaInst *AI);
  static bool IsSimpleStoreOnlyAlloca(AllocaInst *AI);

  bool runOnModule(Module &M) override {
    // Patch memcpy to cover case bitcast (gep ptr, 0,0) is transformed into
    // bitcast ptr.
    MemcpySplitter::PatchMemCpyWithZeroIdxGEP(M);

    m_pHLModule = &M.GetOrCreateHLModule();
    const DataLayout &DL = M.getDataLayout();
    // Load up debug information, to cross-reference values and the instructions
    // used to load them.
    m_HasDbgInfo = nullptr != M.getNamedMetadata("llvm.dbg.cu");

    InjectReturnAfterNoReturnPreserveOutput(*m_pHLModule);

    std::deque<Function *> WorkList;
    std::vector<Function *> DeadHLFunctions;
    for (Function &F : M.functions()) {
      HLOpcodeGroup group = GetHLOpcodeGroup(&F);
      // Skip HL operations.
      if (group != HLOpcodeGroup::NotHL ||
          group == HLOpcodeGroup::HLExtIntrinsic) {
        if (F.user_empty())
          DeadHLFunctions.emplace_back(&F);
        continue;
      }

      if (F.isDeclaration()) {
        // Skip llvm intrinsic.
        if (F.isIntrinsic())
          continue;
        // Skip unused external function.
        if (F.user_empty())
          continue;
      }
      // Skip void(void) functions.
      if (F.getReturnType()->isVoidTy() && F.arg_size() == 0)
        continue;

      // Skip library function, except to LegalizeDxilInputOutputs
      if (&F != m_pHLModule->GetEntryFunction() &&
          !m_pHLModule->IsEntryThatUsesSignatures(&F)) {
        if (!F.isDeclaration())
          LegalizeDxilInputOutputs(&F, m_pHLModule->GetFunctionAnnotation(&F),
                                   DL, m_pHLModule->GetTypeSystem());
        continue;
      }

      WorkList.emplace_back(&F);
    }

    // Remove dead hl functions here.
    // This is for hl functions which has body and always inline.
    for (Function *F : DeadHLFunctions) {
      F->eraseFromParent();
    }

    // Preprocess aggregate function param used as function call arg.
    for (Function *F : WorkList) {
      preprocessArgUsedInCall(F);
    }

    // Process the worklist
    while (!WorkList.empty()) {
      Function *F = WorkList.front();
      WorkList.pop_front();
      RewriteBitcastWithIdenticalStructs(F);
      createFlattenedFunction(F);
    }

    // Replace functions with flattened version when we flat all the functions.
    for (auto Iter : funcMap)
      replaceCall(Iter.first, Iter.second);

    // Update patch constant function.
    for (Function &F : M.functions()) {
      if (F.isDeclaration())
        continue;
      if (!m_pHLModule->HasDxilFunctionProps(&F))
        continue;
      DxilFunctionProps &funcProps = m_pHLModule->GetDxilFunctionProps(&F);
      if (funcProps.shaderKind == DXIL::ShaderKind::Hull) {
        Function *oldPatchConstantFunc =
            funcProps.ShaderProps.HS.patchConstantFunc;
        if (funcMap.count(oldPatchConstantFunc))
          m_pHLModule->SetPatchConstantFunctionForHS(&F, funcMap[oldPatchConstantFunc]);
      }
    }

    // Remove flattened functions.
    for (auto Iter : funcMap) {
      Function *F = Iter.first;
      Function *flatF = Iter.second;
      flatF->takeName(F);
      F->eraseFromParent();
    }

    // SROA globals and allocas.
    SROAGlobalAndAllocas(*m_pHLModule, m_HasDbgInfo);

    // Move up allocas that might have been pushed down by instruction inserts
    SmallVector<AllocaInst *, 16> simpleStoreOnlyAllocas;
    for (Function &F : M) {
      if (F.isDeclaration())
        continue;
      Instruction *insertPt = nullptr;
      simpleStoreOnlyAllocas.clear();
      // SROA only potentially "incorrectly" inserts non-allocas into the entry block.
      for (llvm::Instruction &I : F.getEntryBlock()) {
        // In really pathologically huge shaders, there could be thousands of
        // unused allocas (and hundreds of thousands of dbg.declares). Record
        // and remove them now so they don't horrifically slow down the
        // compilation.
        if (isa<AllocaInst>(I) && IsSimpleStoreOnlyAlloca(cast<AllocaInst>(&I)))
          simpleStoreOnlyAllocas.push_back(cast<AllocaInst>(&I));

        if (!insertPt) {
          // Find the first non-alloca to move the allocas above
          if (!isa<AllocaInst>(I) && !isa<DbgInfoIntrinsic>(I))
            insertPt = &I;
        } else if (isa<AllocaInst>(I)) {
          // Move any alloca to before the first non-alloca
          I.moveBefore(insertPt);
        }
      }
      for (AllocaInst *AI : simpleStoreOnlyAllocas) {
        DeleteSimpleStoreOnlyAlloca(AI);
      }

      DeleteOutOfScopeDebugInfo(F);
    }
    return true;
  }

private:
  void DeleteDeadInstructions();
  void preprocessArgUsedInCall(Function *F);
  void moveFunctionBody(Function *F, Function *flatF);
  void replaceCall(Function *F, Function *flatF);
  void createFlattenedFunction(Function *F);
  void
  flattenArgument(Function *F, Value *Arg, bool bForParam,
                  DxilParameterAnnotation &paramAnnotation,
                  std::vector<Value *> &FlatParamList,
                  std::vector<DxilParameterAnnotation> &FlatRetAnnotationList,
                  BasicBlock *EntryBlock, ArrayRef<DbgDeclareInst *> DDIs);
  Value *castResourceArgIfRequired(Value *V, Type *Ty, bool bOut,
                                   DxilParamInputQual inputQual,
                                   IRBuilder<> &Builder);
  Value *castArgumentIfRequired(Value *V, Type *Ty, bool bOut,
                                DxilParamInputQual inputQual,
                                DxilFieldAnnotation &annotation,
                                IRBuilder<> &Builder,
                                DxilTypeSystem &TypeSys);
  // Replace use of parameter which changed type when flatten.
  // Also add information to Arg if required.
  void replaceCastParameter(Value *NewParam, Value *OldParam, Function &F,
                            Argument *Arg, const DxilParamInputQual inputQual,
                            IRBuilder<> &Builder);
  void allocateSemanticIndex(
    std::vector<DxilParameterAnnotation> &FlatAnnotationList,
    unsigned startArgIndex, llvm::StringMap<Type *> &semanticTypeMap);

  //static std::vector<Value*> GetConstValueIdxList(IRBuilder<>& builder, std::vector<unsigned> idxlist);
  /// DeadInsts - Keep track of instructions we have made dead, so that
  /// we can remove them after we are done working.
  SmallVector<Value *, 32> DeadInsts;
  // Map from orginal function to the flatten version.
  MapVector<Function *, Function *> funcMap; // Need deterministic order of iteration
  // Map from original arg/param to flatten cast version.
  std::unordered_map<Value *, std::pair<Value*, DxilParamInputQual>> castParamMap;
  // Map form first element of a vector the list of all elements of the vector.
  std::unordered_map<Value *, SmallVector<Value*, 4> > vectorEltsMap;
  // Set for row major matrix parameter.
  std::unordered_set<Value *> castRowMajorParamMap;
  bool m_HasDbgInfo;
};

// When replacing aggregates by its scalar elements,
// the first element will preserve the original semantic,
// and the subsequent ones will temporarily use this value.
// We then run a pass to fix the semantics and properly renumber them
// once the aggregate has been fully expanded.
// 
// For example:
// struct Foo { float a; float b; };
// void main(Foo foo : TEXCOORD0, float bar : TEXCOORD0)
//
// Will be expanded to
// void main(float a : TEXCOORD0, float b : *, float bar : TEXCOORD0)
//
// And then fixed up to
// void main(float a : TEXCOORD0, float b : TEXCOORD1, float bar : TEXCOORD0)
//
// (which will later on fail validation due to duplicate semantics).
constexpr const char *ContinuedPseudoSemantic = "*";
}

char SROA_Parameter_HLSL::ID = 0;

INITIALIZE_PASS(SROA_Parameter_HLSL, "scalarrepl-param-hlsl",
  "Scalar Replacement of Aggregates HLSL (parameters)", false,
  false)

void SROA_Parameter_HLSL::RewriteBitcastWithIdenticalStructs(Function *F) {
  if (F->isDeclaration())
    return;
  // Gather list of bitcast involving src and dest structs with identical layout
  std::vector<BitCastInst*> worklist;
  for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
    if (BitCastInst *BCI = dyn_cast<BitCastInst>(&*I)) {
      Type *DstTy = BCI->getDestTy();
      Type *SrcTy = BCI->getSrcTy();
      if(ArePointersToStructsOfIdenticalLayouts(DstTy, SrcTy))
        worklist.push_back(BCI);
    }
  }

  // Replace bitcast involving src and dest structs with identical layout
  while (!worklist.empty()) {
    BitCastInst *BCI = worklist.back();
    worklist.pop_back();
    RewriteBitcastWithIdenticalStructs(BCI);
  }
}

bool SROA_Parameter_HLSL::IsSimpleStoreOnlyAlloca(AllocaInst *AI) {
  if (!AI->getAllocatedType()->isSingleValueType())
    return false;
  for (User *U : AI->users()) {
    if (!isa<StoreInst>(U))
      return false;
  }
  return true;
}

bool SROA_Parameter_HLSL::DeleteSimpleStoreOnlyAlloca(AllocaInst *AI) {
  assert(IsSimpleStoreOnlyAlloca(AI));
  for (auto it = AI->user_begin(), end = AI->user_end(); it != end;) {
    StoreInst *Store = cast<StoreInst>(*(it++));
    Store->eraseFromParent();
  }

  // Delete dbg.declare's too
  for (auto it = hlsl::dxilutil::mdv_users_begin(AI), end = hlsl::dxilutil::mdv_users_end(AI); it != end;) {
    User *U = *(it++);
    if (DbgDeclareInst *DDI = dyn_cast<DbgDeclareInst>(U))
      DDI->eraseFromParent();
  }

  AI->eraseFromParent();
  return true;
}

void SROA_Parameter_HLSL::RewriteBitcastWithIdenticalStructs(BitCastInst *BCI) {
  StructType *srcStTy = cast<StructType>(BCI->getSrcTy()->getPointerElementType());
  StructType *destStTy = cast<StructType>(BCI->getDestTy()->getPointerElementType());
  Value* srcPtr = BCI->getOperand(0);
  IRBuilder<> AllocaBuilder(dxilutil::FindAllocaInsertionPt(BCI->getParent()->getParent()));
  AllocaInst *destPtr = AllocaBuilder.CreateAlloca(destStTy);
  IRBuilder<> InstBuilder(BCI);
  std::vector<unsigned> idxlist = { 0 };
  CopyElementsOfStructsWithIdenticalLayout(InstBuilder, destPtr, srcPtr, srcStTy, idxlist);
  BCI->replaceAllUsesWith(destPtr);
  BCI->eraseFromParent();
}

/// DeleteDeadInstructions - Erase instructions on the DeadInstrs list,
/// recursively including all their operands that become trivially dead.
void SROA_Parameter_HLSL::DeleteDeadInstructions() {
  while (!DeadInsts.empty()) {
    Instruction *I = cast<Instruction>(DeadInsts.pop_back_val());

    for (User::op_iterator OI = I->op_begin(), E = I->op_end(); OI != E; ++OI)
      if (Instruction *U = dyn_cast<Instruction>(*OI)) {
        // Zero out the operand and see if it becomes trivially dead.
        // (But, don't add allocas to the dead instruction list -- they are
        // already on the worklist and will be deleted separately.)
        *OI = nullptr;
        if (isInstructionTriviallyDead(U) && !isa<AllocaInst>(U))
          DeadInsts.push_back(U);
      }

    I->eraseFromParent();
  }
}

static DxilFieldAnnotation &GetEltAnnotation(Type *Ty, unsigned idx, DxilFieldAnnotation &annotation, DxilTypeSystem &dxilTypeSys) {
  while (Ty->isArrayTy())
    Ty = Ty->getArrayElementType();
  if (StructType *ST = dyn_cast<StructType>(Ty)) {
    if (HLMatrixType::isa(Ty))
      return annotation;
    DxilStructAnnotation *SA = dxilTypeSys.GetStructAnnotation(ST);
    if (SA) {
      DxilFieldAnnotation &FA = SA->GetFieldAnnotation(idx);
      return FA;
    }
  }
  return annotation;  
}

// Note: Semantic index allocation.
// Semantic index is allocated base on linear layout.
// For following code
/*
    struct S {
     float4 m;
     float4 m2;
    };
    S  s[2] : semantic;

    struct S2 {
     float4 m[2];
     float4 m2[2];
    };

    S2  s2 : semantic;
*/
//  The semantic index is like this:
//  s[0].m  : semantic0
//  s[0].m2 : semantic1
//  s[1].m  : semantic2
//  s[1].m2 : semantic3

//  s2.m[0]  : semantic0
//  s2.m[1]  : semantic1
//  s2.m2[0] : semantic2
//  s2.m2[1] : semantic3

//  But when flatten argument, the result is like this:
//  float4 s_m[2], float4 s_m2[2].
//  float4 s2_m[2], float4 s2_m2[2].

// To do the allocation, need to map from each element to its flattened argument.
// Say arg index of float4 s_m[2] is 0, float4 s_m2[2] is 1.
// Need to get 0 from s[0].m and s[1].m, get 1 from s[0].m2 and s[1].m2.


// Allocate the argments with same semantic string from type where the
// semantic starts( S2 for s2.m[2] and s2.m2[2]).
// Iterate each elements of the type, save the semantic index and update it.
// The map from element to the arg ( s[0].m2 -> s.m2[2]) is done by argIdx.
// ArgIdx only inc by 1 when finish a struct field.
static unsigned AllocateSemanticIndex(
    Type *Ty, unsigned &semIndex, unsigned argIdx, unsigned endArgIdx,
    std::vector<DxilParameterAnnotation> &FlatAnnotationList) {
  if (Ty->isPointerTy()) {
    return AllocateSemanticIndex(Ty->getPointerElementType(), semIndex, argIdx,
                                 endArgIdx, FlatAnnotationList);
  } else if (Ty->isArrayTy()) {
    unsigned arraySize = Ty->getArrayNumElements();
    unsigned updatedArgIdx = argIdx;
    Type *EltTy = Ty->getArrayElementType();
    for (unsigned i = 0; i < arraySize; i++) {
      updatedArgIdx = AllocateSemanticIndex(EltTy, semIndex, argIdx, endArgIdx,
                                            FlatAnnotationList);
    }
    return updatedArgIdx;
  } else if (Ty->isStructTy() && !HLMatrixType::isa(Ty)) {
    unsigned fieldsCount = Ty->getStructNumElements();
    for (unsigned i = 0; i < fieldsCount; i++) {
      Type *EltTy = Ty->getStructElementType(i);
      argIdx = AllocateSemanticIndex(EltTy, semIndex, argIdx, endArgIdx,
                                     FlatAnnotationList);
      if (!(EltTy->isStructTy() && !HLMatrixType::isa(EltTy))) {
        // Update argIdx only when it is a leaf node.
        argIdx++;
      }
    }
    return argIdx;
  } else {
    DXASSERT(argIdx < endArgIdx, "arg index out of bound");
    DxilParameterAnnotation &paramAnnotation = FlatAnnotationList[argIdx];
    // Get element size.
    unsigned rows = 1;
    if (paramAnnotation.HasMatrixAnnotation()) {
      const DxilMatrixAnnotation &matrix =
          paramAnnotation.GetMatrixAnnotation();
      if (matrix.Orientation == MatrixOrientation::RowMajor) {
        rows = matrix.Rows;
      } else {
        DXASSERT_NOMSG(matrix.Orientation == MatrixOrientation::ColumnMajor);
        rows = matrix.Cols;
      }
    }
    // Save semIndex.
    for (unsigned i = 0; i < rows; i++)
      paramAnnotation.AppendSemanticIndex(semIndex + i);
    // Update semIndex.
    semIndex += rows;

    return argIdx;
  }
}

void SROA_Parameter_HLSL::allocateSemanticIndex(
    std::vector<DxilParameterAnnotation> &FlatAnnotationList,
    unsigned startArgIndex, llvm::StringMap<Type *> &semanticTypeMap) {
  unsigned endArgIndex = FlatAnnotationList.size();

  // Allocate semantic index.
  for (unsigned i = startArgIndex; i < endArgIndex; ++i) {
    // Group by semantic names.
    DxilParameterAnnotation &flatParamAnnotation = FlatAnnotationList[i];
    const std::string &semantic = flatParamAnnotation.GetSemanticString();

    // If semantic is undefined, an error will be emitted elsewhere.  For now,
    // we should avoid asserting.
    if (semantic.empty())
      continue;

    StringRef baseSemName; // The 'FOO' in 'FOO1'.
    uint32_t semIndex;     // The '1' in 'FOO1'
    // Split semName and index.
    Semantic::DecomposeNameAndIndex(semantic, &baseSemName, &semIndex);

    unsigned semGroupEnd = i + 1;
    while (semGroupEnd < endArgIndex &&
           FlatAnnotationList[semGroupEnd].GetSemanticString() == ContinuedPseudoSemantic) {
      FlatAnnotationList[semGroupEnd].SetSemanticString(baseSemName);
      ++semGroupEnd;
    }

    DXASSERT(semanticTypeMap.count(semantic) > 0, "Must has semantic type");
    Type *semanticTy = semanticTypeMap[semantic];

    AllocateSemanticIndex(semanticTy, semIndex, /*argIdx*/ i,
                          /*endArgIdx*/ semGroupEnd, FlatAnnotationList);
    // Update i.
    i = semGroupEnd - 1;
  }
}

//
// Cast parameters.
//

static void CopyHandleToResourcePtr(Value *Handle, Value *ResPtr, HLModule &HLM,
                                    IRBuilder<> &Builder) {
  // Cast it to resource.
  Type *ResTy = ResPtr->getType()->getPointerElementType();
  Value *Res = HLM.EmitHLOperationCall(Builder, HLOpcodeGroup::HLCast,
                                       (unsigned)HLCastOpcode::HandleToResCast,
                                       ResTy, {Handle}, *HLM.GetModule());
  // Store casted resource to OldArg.
  Builder.CreateStore(Res, ResPtr);
}

static void CopyHandlePtrToResourcePtr(Value *HandlePtr, Value *ResPtr,
                                       HLModule &HLM, IRBuilder<> &Builder) {
  // Load the handle.
  Value *Handle = Builder.CreateLoad(HandlePtr);
  CopyHandleToResourcePtr(Handle, ResPtr, HLM, Builder);
}

static Value *CastResourcePtrToHandle(Value *Res, Type *HandleTy, HLModule &HLM,
                                      IRBuilder<> &Builder) {
  // Load OldArg.
  Value *LdRes = Builder.CreateLoad(Res);
  Value *Handle = HLM.EmitHLOperationCall(
      Builder, HLOpcodeGroup::HLCreateHandle,
      /*opcode*/ 0, HandleTy, {LdRes}, *HLM.GetModule());
  return Handle;
}

static void CopyResourcePtrToHandlePtr(Value *Res, Value *HandlePtr,
                                       HLModule &HLM, IRBuilder<> &Builder) {
  Type *HandleTy = HandlePtr->getType()->getPointerElementType();
  Value *Handle = CastResourcePtrToHandle(Res, HandleTy, HLM, Builder);
  Builder.CreateStore(Handle, HandlePtr);
}

static void CopyVectorPtrToEltsPtr(Value *VecPtr, ArrayRef<Value *> elts,
                                   unsigned vecSize, IRBuilder<> &Builder) {
  Value *Vec = Builder.CreateLoad(VecPtr);
  for (unsigned i = 0; i < vecSize; i++) {
    Value *Elt = Builder.CreateExtractElement(Vec, i);
    Builder.CreateStore(Elt, elts[i]);
  }
}

static void CopyEltsPtrToVectorPtr(ArrayRef<Value *> elts, Value *VecPtr,
                                   Type *VecTy, unsigned vecSize,
                                   IRBuilder<> &Builder) {
  Value *Vec = UndefValue::get(VecTy);
  for (unsigned i = 0; i < vecSize; i++) {
    Value *Elt = Builder.CreateLoad(elts[i]);
    Vec = Builder.CreateInsertElement(Vec, Elt, i);
  }
  Builder.CreateStore(Vec, VecPtr);
}

static void CopyMatToArrayPtr(Value *Mat, Value *ArrayPtr,
                              unsigned arrayBaseIdx, HLModule &HLM,
                              IRBuilder<> &Builder, bool bRowMajor) {
  // Mat val is row major.
  HLMatrixType MatTy = HLMatrixType::cast(Mat->getType());
  Type *VecTy = MatTy.getLoweredVectorTypeForReg();
  Value *Vec =
      HLM.EmitHLOperationCall(Builder, HLOpcodeGroup::HLCast,
                              (unsigned)HLCastOpcode::RowMatrixToVecCast, VecTy,
                              {Mat}, *HLM.GetModule());
  Value *zero = Builder.getInt32(0);

  for (unsigned r = 0; r < MatTy.getNumRows(); r++) {
    for (unsigned c = 0; c < MatTy.getNumColumns(); c++) {
      unsigned matIdx = MatTy.getColumnMajorIndex(r, c);
      Value *Elt = Builder.CreateExtractElement(Vec, matIdx);
      Value *Ptr = Builder.CreateInBoundsGEP(
          ArrayPtr, {zero, Builder.getInt32(arrayBaseIdx + matIdx)});
      Builder.CreateStore(Elt, Ptr);
    }
  }
}
static void CopyMatPtrToArrayPtr(Value *MatPtr, Value *ArrayPtr,
                                 unsigned arrayBaseIdx, HLModule &HLM,
                                 IRBuilder<> &Builder, bool bRowMajor) {
  Type *Ty = MatPtr->getType()->getPointerElementType();
  Value *Mat = nullptr;
  if (bRowMajor) {
    Mat = HLM.EmitHLOperationCall(Builder, HLOpcodeGroup::HLMatLoadStore,
                                  (unsigned)HLMatLoadStoreOpcode::RowMatLoad,
                                  Ty, {MatPtr}, *HLM.GetModule());
  } else {
    Mat = HLM.EmitHLOperationCall(Builder, HLOpcodeGroup::HLMatLoadStore,
                                  (unsigned)HLMatLoadStoreOpcode::ColMatLoad,
                                  Ty, {MatPtr}, *HLM.GetModule());
    // Matrix value should be row major.
    Mat = HLM.EmitHLOperationCall(Builder, HLOpcodeGroup::HLCast,
                                  (unsigned)HLCastOpcode::ColMatrixToRowMatrix,
                                  Ty, {Mat}, *HLM.GetModule());
  }
  CopyMatToArrayPtr(Mat, ArrayPtr, arrayBaseIdx, HLM, Builder, bRowMajor);
}

static Value *LoadArrayPtrToMat(Value *ArrayPtr, unsigned arrayBaseIdx,
                                Type *Ty, HLModule &HLM, IRBuilder<> &Builder,
                                bool bRowMajor) {
  HLMatrixType MatTy = HLMatrixType::cast(Ty);
  // HLInit operands are in row major.
  SmallVector<Value *, 16> Elts;
  Value *zero = Builder.getInt32(0);
  for (unsigned r = 0; r < MatTy.getNumRows(); r++) {
    for (unsigned c = 0; c < MatTy.getNumColumns(); c++) {
      unsigned matIdx = bRowMajor
        ? MatTy.getRowMajorIndex(r, c)
        : MatTy.getColumnMajorIndex(r, c);
      Value *Ptr = Builder.CreateInBoundsGEP(
          ArrayPtr, {zero, Builder.getInt32(arrayBaseIdx + matIdx)});
      Value *Elt = Builder.CreateLoad(Ptr);
      Elts.emplace_back(Elt);
    }
  }
  return HLM.EmitHLOperationCall(Builder, HLOpcodeGroup::HLInit,
                                 /*opcode*/ 0, Ty, {Elts}, *HLM.GetModule());
}

static void CopyArrayPtrToMatPtr(Value *ArrayPtr, unsigned arrayBaseIdx,
                                 Value *MatPtr, HLModule &HLM,
                                 IRBuilder<> &Builder, bool bRowMajor) {
  Type *Ty = MatPtr->getType()->getPointerElementType();
  Value *Mat =
      LoadArrayPtrToMat(ArrayPtr, arrayBaseIdx, Ty, HLM, Builder, bRowMajor);
  if (bRowMajor) {
    HLM.EmitHLOperationCall(Builder, HLOpcodeGroup::HLMatLoadStore,
                            (unsigned)HLMatLoadStoreOpcode::RowMatStore, Ty,
                            {MatPtr, Mat}, *HLM.GetModule());
  } else {
    // Mat is row major.
    // Cast it to col major before store.
    Mat = HLM.EmitHLOperationCall(Builder, HLOpcodeGroup::HLCast,
                                  (unsigned)HLCastOpcode::RowMatrixToColMatrix,
                                  Ty, {Mat}, *HLM.GetModule());
    HLM.EmitHLOperationCall(Builder, HLOpcodeGroup::HLMatLoadStore,
                            (unsigned)HLMatLoadStoreOpcode::ColMatStore, Ty,
                            {MatPtr, Mat}, *HLM.GetModule());
  }
}

using CopyFunctionTy = void(Value *FromPtr, Value *ToPtr, HLModule &HLM,
                            Type *HandleTy, IRBuilder<> &Builder,
                            bool bRowMajor);

static void
CastCopyArrayMultiDimTo1Dim(Value *FromArray, Value *ToArray, Type *CurFromTy,
                            std::vector<Value *> &idxList, unsigned calcIdx,
                            Type *HandleTy, HLModule &HLM, IRBuilder<> &Builder,
                            CopyFunctionTy CastCopyFn, bool bRowMajor) {
  if (CurFromTy->isVectorTy()) {
    // Copy vector to array.
    Value *FromPtr = Builder.CreateInBoundsGEP(FromArray, idxList);
    Value *V = Builder.CreateLoad(FromPtr);
    unsigned vecSize = CurFromTy->getVectorNumElements();
    Value *zeroIdx = Builder.getInt32(0);
    for (unsigned i = 0; i < vecSize; i++) {
      Value *ToPtr = Builder.CreateInBoundsGEP(
          ToArray, {zeroIdx, Builder.getInt32(calcIdx++)});
      Value *Elt = Builder.CreateExtractElement(V, i);
      Builder.CreateStore(Elt, ToPtr);
    }
  } else if (HLMatrixType MatTy = HLMatrixType::dyn_cast(CurFromTy)) {
    // Copy matrix to array.
    // Calculate the offset.
    unsigned offset = calcIdx * MatTy.getNumElements();
    Value *FromPtr = Builder.CreateInBoundsGEP(FromArray, idxList);
    CopyMatPtrToArrayPtr(FromPtr, ToArray, offset, HLM, Builder, bRowMajor);
  } else if (!CurFromTy->isArrayTy()) {
    Value *FromPtr = Builder.CreateInBoundsGEP(FromArray, idxList);
    Value *ToPtr = Builder.CreateInBoundsGEP(
        ToArray, {Builder.getInt32(0), Builder.getInt32(calcIdx)});
    CastCopyFn(FromPtr, ToPtr, HLM, HandleTy, Builder, bRowMajor);
  } else {
    unsigned size = CurFromTy->getArrayNumElements();
    Type *FromEltTy = CurFromTy->getArrayElementType();
    for (unsigned i = 0; i < size; i++) {
      idxList.push_back(Builder.getInt32(i));
      unsigned idx = calcIdx * size + i;
      CastCopyArrayMultiDimTo1Dim(FromArray, ToArray, FromEltTy, idxList, idx,
                                  HandleTy, HLM, Builder, CastCopyFn,
                                  bRowMajor);
      idxList.pop_back();
    }
  }
}

static void
CastCopyArray1DimToMultiDim(Value *FromArray, Value *ToArray, Type *CurToTy,
                            std::vector<Value *> &idxList, unsigned calcIdx,
                            Type *HandleTy, HLModule &HLM, IRBuilder<> &Builder,
                            CopyFunctionTy CastCopyFn, bool bRowMajor) {
  if (CurToTy->isVectorTy()) {
    // Copy array to vector.
    Value *V = UndefValue::get(CurToTy);
    unsigned vecSize = CurToTy->getVectorNumElements();
    // Calculate the offset.
    unsigned offset = calcIdx * vecSize;
    Value *zeroIdx = Builder.getInt32(0);
    Value *ToPtr = Builder.CreateInBoundsGEP(ToArray, idxList);
    for (unsigned i = 0; i < vecSize; i++) {
      Value *FromPtr = Builder.CreateInBoundsGEP(
          FromArray, {zeroIdx, Builder.getInt32(offset++)});
      Value *Elt = Builder.CreateLoad(FromPtr);
      V = Builder.CreateInsertElement(V, Elt, i);
    }
    Builder.CreateStore(V, ToPtr);
  } else if (HLMatrixType MatTy = HLMatrixType::cast(CurToTy)) {
    // Copy array to matrix.
    // Calculate the offset.
    unsigned offset = calcIdx * MatTy.getNumElements();
    Value *ToPtr = Builder.CreateInBoundsGEP(ToArray, idxList);
    CopyArrayPtrToMatPtr(FromArray, offset, ToPtr, HLM, Builder, bRowMajor);
  } else if (!CurToTy->isArrayTy()) {
    Value *FromPtr = Builder.CreateInBoundsGEP(
        FromArray, {Builder.getInt32(0), Builder.getInt32(calcIdx)});
    Value *ToPtr = Builder.CreateInBoundsGEP(ToArray, idxList);
    CastCopyFn(FromPtr, ToPtr, HLM, HandleTy, Builder, bRowMajor);
  } else {
    unsigned size = CurToTy->getArrayNumElements();
    Type *ToEltTy = CurToTy->getArrayElementType();
    for (unsigned i = 0; i < size; i++) {
      idxList.push_back(Builder.getInt32(i));
      unsigned idx = calcIdx * size + i;
      CastCopyArray1DimToMultiDim(FromArray, ToArray, ToEltTy, idxList, idx,
                                  HandleTy, HLM, Builder, CastCopyFn,
                                  bRowMajor);
      idxList.pop_back();
    }
  }
}

static void CastCopyOldPtrToNewPtr(Value *OldPtr, Value *NewPtr, HLModule &HLM,
                                   Type *HandleTy, IRBuilder<> &Builder,
                                   bool bRowMajor) {
  Type *NewTy = NewPtr->getType()->getPointerElementType();
  Type *OldTy = OldPtr->getType()->getPointerElementType();
  if (NewTy == HandleTy) {
    CopyResourcePtrToHandlePtr(OldPtr, NewPtr, HLM, Builder);
  } else if (OldTy->isVectorTy()) {
    // Copy vector to array.
    Value *V = Builder.CreateLoad(OldPtr);
    unsigned vecSize = OldTy->getVectorNumElements();
    Value *zeroIdx = Builder.getInt32(0);
    for (unsigned i = 0; i < vecSize; i++) {
      Value *EltPtr = Builder.CreateGEP(NewPtr, {zeroIdx, Builder.getInt32(i)});
      Value *Elt = Builder.CreateExtractElement(V, i);
      Builder.CreateStore(Elt, EltPtr);
    }
  } else if (HLMatrixType::isa(OldTy)) {
    CopyMatPtrToArrayPtr(OldPtr, NewPtr, /*arrayBaseIdx*/ 0, HLM, Builder,
                         bRowMajor);
  } else if (OldTy->isArrayTy()) {
    std::vector<Value *> idxList;
    idxList.emplace_back(Builder.getInt32(0));
    CastCopyArrayMultiDimTo1Dim(OldPtr, NewPtr, OldTy, idxList, /*calcIdx*/ 0,
                                HandleTy, HLM, Builder, CastCopyOldPtrToNewPtr,
                                bRowMajor);
  }
}

static void CastCopyNewPtrToOldPtr(Value *NewPtr, Value *OldPtr, HLModule &HLM,
                                   Type *HandleTy, IRBuilder<> &Builder,
                                   bool bRowMajor) {
  Type *NewTy = NewPtr->getType()->getPointerElementType();
  Type *OldTy = OldPtr->getType()->getPointerElementType();
  if (NewTy == HandleTy) {
    CopyHandlePtrToResourcePtr(NewPtr, OldPtr, HLM, Builder);
  } else if (OldTy->isVectorTy()) {
    // Copy array to vector.
    Value *V = UndefValue::get(OldTy);
    unsigned vecSize = OldTy->getVectorNumElements();
    Value *zeroIdx = Builder.getInt32(0);
    for (unsigned i = 0; i < vecSize; i++) {
      Value *EltPtr = Builder.CreateGEP(NewPtr, {zeroIdx, Builder.getInt32(i)});
      Value *Elt = Builder.CreateLoad(EltPtr);
      V = Builder.CreateInsertElement(V, Elt, i);
    }
    Builder.CreateStore(V, OldPtr);
  } else if (HLMatrixType::isa(OldTy)) {
    CopyArrayPtrToMatPtr(NewPtr, /*arrayBaseIdx*/ 0, OldPtr, HLM, Builder,
                         bRowMajor);
  } else if (OldTy->isArrayTy()) {
    std::vector<Value *> idxList;
    idxList.emplace_back(Builder.getInt32(0));
    CastCopyArray1DimToMultiDim(NewPtr, OldPtr, OldTy, idxList, /*calcIdx*/ 0,
                                HandleTy, HLM, Builder, CastCopyNewPtrToOldPtr,
                                bRowMajor);
  }
}

void SROA_Parameter_HLSL::replaceCastParameter(
    Value *NewParam, Value *OldParam, Function &F, Argument *Arg,
    const DxilParamInputQual inputQual, IRBuilder<> &Builder) {
  Type *HandleTy = m_pHLModule->GetOP()->GetHandleType();

  Type *NewTy = NewParam->getType();
  Type *OldTy = OldParam->getType();

  bool bIn = inputQual == DxilParamInputQual::Inout ||
             inputQual == DxilParamInputQual::In;
  bool bOut = inputQual == DxilParamInputQual::Inout ||
              inputQual == DxilParamInputQual::Out;

  // Make sure InsertPoint after OldParam inst.
  if (Instruction *I = dyn_cast<Instruction>(OldParam)) {
    Builder.SetInsertPoint(I->getNextNode());
  }

  SmallVector<DbgDeclareInst *, 4> dbgDecls;
  llvm::FindAllocaDbgDeclare(OldParam, dbgDecls);
  for (DbgDeclareInst *DDI : dbgDecls) {
    // Add debug info to new param.
    DIBuilder DIB(*F.getParent(), /*AllowUnresolved*/ false);
    DIExpression *DDIExp = DDI->getExpression();
    DIB.insertDeclare(NewParam, DDI->getVariable(), DDIExp, DDI->getDebugLoc(),
                      Builder.GetInsertPoint());
  }

  if (isa<Argument>(OldParam) && OldTy->isPointerTy()) {
    // OldParam will be removed with Old function.
    // Create alloca to replace it.
    IRBuilder<> AllocaBuilder(dxilutil::FindAllocaInsertionPt(&F));
    Value *AllocParam = AllocaBuilder.CreateAlloca(OldTy->getPointerElementType());
    OldParam->replaceAllUsesWith(AllocParam);
    OldParam = AllocParam;
  }

  if (NewTy == HandleTy) {
    CopyHandleToResourcePtr(NewParam, OldParam, *m_pHLModule, Builder);
  } else if (vectorEltsMap.count(NewParam)) {
    // Vector is flattened to scalars.
    Type *VecTy = OldTy;
    if (VecTy->isPointerTy())
      VecTy = VecTy->getPointerElementType();

    // Flattened vector.
    SmallVector<Value *, 4> &elts = vectorEltsMap[NewParam];
    unsigned vecSize = elts.size();

    if (NewTy->isPointerTy()) {
      if (bIn) {
        // Copy NewParam to OldParam at entry.
        CopyEltsPtrToVectorPtr(elts, OldParam, VecTy, vecSize, Builder);
      }
      // bOut must be true here.
      // Store the OldParam to NewParam before every return.
      for (auto &BB : F.getBasicBlockList()) {
        if (ReturnInst *RI = dyn_cast<ReturnInst>(BB.getTerminator())) {
          IRBuilder<> RetBuilder(RI);
          CopyVectorPtrToEltsPtr(OldParam, elts, vecSize, RetBuilder);
        }
      }
    } else {
      // Must be in parameter.
      // Copy NewParam to OldParam at entry.
      Value *Vec = UndefValue::get(VecTy);
      for (unsigned i = 0; i < vecSize; i++) {
        Vec = Builder.CreateInsertElement(Vec, elts[i], i);
      }
      if (OldTy->isPointerTy()) {
        Builder.CreateStore(Vec, OldParam);
      } else {
        OldParam->replaceAllUsesWith(Vec);
      }
    }
    // Don't need elts anymore.
    vectorEltsMap.erase(NewParam);
  } else if (!NewTy->isPointerTy()) {
    // Ptr param is cast to non-ptr param.
    // Must be in param.
    // Store NewParam to OldParam at entry.
    Builder.CreateStore(NewParam, OldParam);
  } else if (HLMatrixType::isa(OldTy)) {
    bool bRowMajor = castRowMajorParamMap.count(NewParam);
    Value *Mat = LoadArrayPtrToMat(NewParam, /*arrayBaseIdx*/ 0, OldTy,
                                   *m_pHLModule, Builder, bRowMajor);
    OldParam->replaceAllUsesWith(Mat);
  } else {
    bool bRowMajor = castRowMajorParamMap.count(NewParam);
    // NewTy is pointer type.
    if (bIn) {
      // Copy NewParam to OldParam at entry.
      CastCopyNewPtrToOldPtr(NewParam, OldParam, *m_pHLModule, HandleTy,
                             Builder, bRowMajor);
    }
    if (bOut) {
      // Store the OldParam to NewParam before every return.
      for (auto &BB : F.getBasicBlockList()) {
        if (ReturnInst *RI = dyn_cast<ReturnInst>(BB.getTerminator())) {
          IRBuilder<> RetBuilder(RI);
          CastCopyOldPtrToNewPtr(OldParam, NewParam, *m_pHLModule, HandleTy,
                                 RetBuilder, bRowMajor);
        }
      }
    }
  }
}

Value *SROA_Parameter_HLSL::castResourceArgIfRequired(
    Value *V, Type *Ty, bool bOut,
    DxilParamInputQual inputQual,
    IRBuilder<> &Builder) {
  Type *HandleTy = m_pHLModule->GetOP()->GetHandleType();
  Module &M = *m_pHLModule->GetModule();
  IRBuilder<> AllocaBuilder(dxilutil::FindAllocaInsertionPt(Builder.GetInsertPoint()));

  // Lower resource type to handle ty.
  if (dxilutil::IsHLSLResourceType(Ty)) {
    Value *Res = V;
    if (!bOut) {
      Value *LdRes = Builder.CreateLoad(Res);
      V = m_pHLModule->EmitHLOperationCall(Builder,
        HLOpcodeGroup::HLCreateHandle,
        /*opcode*/ 0, HandleTy, { LdRes }, M);
    }
    else {
      V = AllocaBuilder.CreateAlloca(HandleTy);
    }
    castParamMap[V] = std::make_pair(Res, inputQual);
  }
  else if (Ty->isArrayTy()) {
    unsigned arraySize = 1;
    Type *AT = Ty;
    while (AT->isArrayTy()) {
      arraySize *= AT->getArrayNumElements();
      AT = AT->getArrayElementType();
    }
    if (dxilutil::IsHLSLResourceType(AT)) {
      Value *Res = V;
      Type *Ty = ArrayType::get(HandleTy, arraySize);
      V = AllocaBuilder.CreateAlloca(Ty);
      castParamMap[V] = std::make_pair(Res, inputQual);
    }
  }
  return V;
}

Value *SROA_Parameter_HLSL::castArgumentIfRequired(
    Value *V, Type *Ty, bool bOut,
    DxilParamInputQual inputQual, DxilFieldAnnotation &annotation,
    IRBuilder<> &Builder,
    DxilTypeSystem &TypeSys) {
  Module &M = *m_pHLModule->GetModule();
  IRBuilder<> AllocaBuilder(dxilutil::FindAllocaInsertionPt(Builder.GetInsertPoint()));

  if (inputQual == DxilParamInputQual::InPayload) {
    DXASSERT_NOMSG(isa<StructType>(Ty));
    // Lower payload type here
    StructType *LoweredTy = GetLoweredUDT(cast<StructType>(Ty), &TypeSys);
    if (LoweredTy != Ty) {
      Value *Ptr = AllocaBuilder.CreateAlloca(LoweredTy);
      ReplaceUsesForLoweredUDT(V, Ptr);
      castParamMap[V] = std::make_pair(Ptr, inputQual);
      V = Ptr;
    }
    return V;
  }

  // Remove pointer for vector/scalar which is not out.
  if (V->getType()->isPointerTy() && !Ty->isAggregateType() && !bOut) {
    Value *Ptr = AllocaBuilder.CreateAlloca(Ty);
    V->replaceAllUsesWith(Ptr);
    // Create load here to make correct type.
    // The Ptr will be store with correct value in replaceCastParameter.
    if (Ptr->hasOneUse()) {
      // Load after existing user for call arg replace.
      // If not, call arg will load undef.
      // This will not hurt parameter, new load is only after first load.
      // It still before all the load users.
      Instruction *User = cast<Instruction>(*(Ptr->user_begin()));
      IRBuilder<> CallBuilder(User->getNextNode());
      V = CallBuilder.CreateLoad(Ptr);
    } else {
      V = Builder.CreateLoad(Ptr);
    }
    castParamMap[V] = std::make_pair(Ptr, inputQual);
  }

  V = castResourceArgIfRequired(V, Ty, bOut, inputQual, Builder);

  // Entry function matrix value parameter has major.
  // Make sure its user use row major matrix value.
  bool updateToColMajor = annotation.HasMatrixAnnotation() &&
                          annotation.GetMatrixAnnotation().Orientation ==
                              MatrixOrientation::ColumnMajor;
  if (updateToColMajor) {
    if (V->getType()->isPointerTy()) {
      for (User *user : V->users()) {
        CallInst *CI = dyn_cast<CallInst>(user);
        if (!CI)
          continue;

        HLOpcodeGroup group = GetHLOpcodeGroupByName(CI->getCalledFunction());
        if (group != HLOpcodeGroup::HLMatLoadStore)
          continue;
        HLMatLoadStoreOpcode opcode =
            static_cast<HLMatLoadStoreOpcode>(GetHLOpcode(CI));
        Type *opcodeTy = Builder.getInt32Ty();
        switch (opcode) {
        case HLMatLoadStoreOpcode::RowMatLoad: {
          // Update matrix function opcode to col major version.
          Value *rowOpArg = ConstantInt::get(
              opcodeTy,
              static_cast<unsigned>(HLMatLoadStoreOpcode::ColMatLoad));
          CI->setOperand(HLOperandIndex::kOpcodeIdx, rowOpArg);
          // Cast it to row major.
          CallInst *RowMat = HLModule::EmitHLOperationCall(
              Builder, HLOpcodeGroup::HLCast,
              (unsigned)HLCastOpcode::ColMatrixToRowMatrix, Ty, {CI}, M);
          CI->replaceAllUsesWith(RowMat);
          // Set arg to CI again.
          RowMat->setArgOperand(HLOperandIndex::kUnaryOpSrc0Idx, CI);
        } break;
        case HLMatLoadStoreOpcode::RowMatStore:
          // Update matrix function opcode to col major version.
          Value *rowOpArg = ConstantInt::get(
              opcodeTy,
              static_cast<unsigned>(HLMatLoadStoreOpcode::ColMatStore));
          CI->setOperand(HLOperandIndex::kOpcodeIdx, rowOpArg);
          Value *Mat = CI->getArgOperand(HLOperandIndex::kMatStoreValOpIdx);
          // Cast it to col major.
          CallInst *RowMat = HLModule::EmitHLOperationCall(
              Builder, HLOpcodeGroup::HLCast,
              (unsigned)HLCastOpcode::RowMatrixToColMatrix, Ty, {Mat}, M);
          CI->setArgOperand(HLOperandIndex::kMatStoreValOpIdx, RowMat);
          break;
        }
      }
    } else {
      CallInst *RowMat = HLModule::EmitHLOperationCall(
          Builder, HLOpcodeGroup::HLCast,
          (unsigned)HLCastOpcode::ColMatrixToRowMatrix, Ty, {V}, M);
      V->replaceAllUsesWith(RowMat);
      // Set arg to V again.
      RowMat->setArgOperand(HLOperandIndex::kUnaryOpSrc0Idx, V);
    }
  }
  return V;
}

struct AnnotatedValue {
  llvm::Value *Value;
  DxilFieldAnnotation Annotation;
};

void SROA_Parameter_HLSL::flattenArgument(
    Function *F, Value *Arg, bool bForParam,
    DxilParameterAnnotation &paramAnnotation,
    std::vector<Value *> &FlatParamList,
    std::vector<DxilParameterAnnotation> &FlatAnnotationList,
    BasicBlock *EntryBlock, ArrayRef<DbgDeclareInst *> DDIs) {
  std::deque<AnnotatedValue> WorkList;
  WorkList.push_back({ Arg, paramAnnotation });

  unsigned startArgIndex = FlatAnnotationList.size();

  DxilTypeSystem &dxilTypeSys = m_pHLModule->GetTypeSystem();

  const std::string &semantic = paramAnnotation.GetSemanticString();

  DxilParamInputQual inputQual = paramAnnotation.GetParamInputQual();
  bool bOut = inputQual == DxilParamInputQual::Out ||
              inputQual == DxilParamInputQual::Inout ||
              inputQual == DxilParamInputQual::OutStream0 ||
              inputQual == DxilParamInputQual::OutStream1 ||
              inputQual == DxilParamInputQual::OutStream2 ||
              inputQual == DxilParamInputQual::OutStream3;

  // Map from semantic string to type.
  llvm::StringMap<Type *> semanticTypeMap;
  // Original semantic type.
  if (!semantic.empty()) {
    // Unwrap top-level array if primitive
    if (inputQual == DxilParamInputQual::InputPatch ||
        inputQual == DxilParamInputQual::OutputPatch ||
        inputQual == DxilParamInputQual::InputPrimitive) {
      Type *Ty = Arg->getType();
      if (Ty->isPointerTy())
        Ty = Ty->getPointerElementType();
      if (Ty->isArrayTy())
        semanticTypeMap[semantic] = Ty->getArrayElementType();
    } else {
      semanticTypeMap[semantic] = Arg->getType();
    }
  }

  std::vector<Instruction*> deadAllocas;

  DIBuilder DIB(*F->getParent(), /*AllowUnresolved*/ false);
  unsigned debugOffset = 0;
  const DataLayout &DL = F->getParent()->getDataLayout();

  // Process the worklist
  while (!WorkList.empty()) {
    AnnotatedValue AV = WorkList.front();
    WorkList.pop_front();

    // Do not skip unused parameter.
    Value *V = AV.Value;
    DxilFieldAnnotation &annotation = AV.Annotation;

    // We can never replace memcpy for arguments because they have an implicit
    // first memcpy that happens from argument passing, and pointer analysis
    // will not reveal that, especially if we've done a first SROA pass on V.
    // No DomTree needed for that reason
    const bool bAllowReplace = false;
    SROA_Helper::LowerMemcpy(V, &annotation, dxilTypeSys, DL, nullptr /*DT */, bAllowReplace);

    // Now is safe to create the IRBuilder.
    // If we create it before LowerMemcpy, the insertion pointer instruction may get deleted
    IRBuilder<> Builder(dxilutil::FindAllocaInsertionPt(EntryBlock));

    std::vector<Value *> Elts;

    // Not flat vector for entry function currently.
    bool SROAed = false;
    Type *BrokenUpTy = nullptr;
    uint64_t NumInstances = 1;
    if (inputQual != DxilParamInputQual::InPayload) {
      // DomTree isn't used by arguments
      SROAed = SROA_Helper::DoScalarReplacement(
        V, Elts, BrokenUpTy, NumInstances, Builder, 
        /*bFlatVector*/ false, annotation.IsPrecise(),
        dxilTypeSys, DL, DeadInsts, /*DT*/ nullptr);
    }

    if (SROAed) {
      Type *Ty = V->getType()->getPointerElementType();
      // Skip empty struct parameters.
      if (SROA_Helper::IsEmptyStructType(Ty, dxilTypeSys)) {
        SROA_Helper::MarkEmptyStructUsers(V, DeadInsts);
        DeleteDeadInstructions();
        continue;
      }

      bool precise = annotation.IsPrecise();
      const std::string &semantic = annotation.GetSemanticString();
      hlsl::InterpolationMode interpMode = annotation.GetInterpolationMode();

      // Find index of first non-empty field.
      unsigned firstNonEmptyIx = Elts.size();
      for (unsigned ri = 0; ri < Elts.size(); ri++) {
        if (DL.getTypeSizeInBits(Ty->getContainedType(ri)) > 0) {
          firstNonEmptyIx = ri;
          break;
        }
      }

      // Push Elts into workList from right to left to preserve the order.
      for (unsigned ri=0;ri<Elts.size();ri++) {
        unsigned i = Elts.size() - ri - 1;
        DxilFieldAnnotation EltAnnotation = GetEltAnnotation(Ty, i, annotation, dxilTypeSys);
        const std::string &eltSem = EltAnnotation.GetSemanticString();
        if (!semantic.empty()) {
          if (!eltSem.empty()) {
            // It doesn't look like we can provide source location information from here
            F->getContext().emitWarning(
              Twine("semantic '") + eltSem + "' on field overridden by function or enclosing type");
          }

          // Inherit semantic from parent, but only preserve it for the first
          // non-zero-sized element.
          // Subsequent elements are noted with a special value that gets resolved
          // once the argument is completely flattened.
          EltAnnotation.SetSemanticString(
            i == firstNonEmptyIx ? semantic : ContinuedPseudoSemantic);
        } else if (!eltSem.empty() &&
                 semanticTypeMap.count(eltSem) == 0) {
          Type *EltTy = dxilutil::GetArrayEltTy(Ty);
          DXASSERT(EltTy->isStructTy(), "must be a struct type to has semantic.");
          semanticTypeMap[eltSem] = EltTy->getStructElementType(i);
        }

        if (precise)
          EltAnnotation.SetPrecise();

        if (EltAnnotation.GetInterpolationMode().GetKind() == DXIL::InterpolationMode::Undefined)
          EltAnnotation.SetInterpolationMode(interpMode);

        WorkList.push_front({ Elts[i], EltAnnotation });
      }

      ++NumReplaced;
      if (Instruction *I = dyn_cast<Instruction>(V))
        deadAllocas.emplace_back(I);
    } else {
      Type *Ty = V->getType();
      if (Ty->isPointerTy())
        Ty = Ty->getPointerElementType();

      // Flatten array of SV_Target.
      StringRef semanticStr = annotation.GetSemanticString();
      if (semanticStr.upper().find("SV_TARGET") == 0 &&
          Ty->isArrayTy()) {
        Type *Ty = cast<ArrayType>(V->getType()->getPointerElementType());
        StringRef targetStr;
        unsigned  targetIndex;
        Semantic::DecomposeNameAndIndex(semanticStr, &targetStr, &targetIndex);
        // Replace target parameter with local target.
        AllocaInst *localTarget = Builder.CreateAlloca(Ty);
        V->replaceAllUsesWith(localTarget);
        unsigned arraySize = 1;
        std::vector<unsigned> arraySizeList;
        while (Ty->isArrayTy()) {
          unsigned size = Ty->getArrayNumElements();
          arraySizeList.emplace_back(size);
          arraySize *= size;
          Ty = Ty->getArrayElementType();
        }

        unsigned arrayLevel = arraySizeList.size();
        std::vector<unsigned> arrayIdxList(arrayLevel, 0);

        // Create flattened target.
        DxilFieldAnnotation EltAnnotation = annotation;
        for (unsigned i=0;i<arraySize;i++) {
          Value *Elt = Builder.CreateAlloca(Ty);
          EltAnnotation.SetSemanticString(targetStr.str()+std::to_string(targetIndex+i));

          // Add semantic type.
          semanticTypeMap[EltAnnotation.GetSemanticString()] = Ty;

          WorkList.push_front({ Elt, EltAnnotation });
          // Copy local target to flattened target.
          std::vector<Value*> idxList(arrayLevel+1);
          idxList[0] = Builder.getInt32(0);
          for (unsigned idx=0;idx<arrayLevel; idx++) {
            idxList[idx+1] = Builder.getInt32(arrayIdxList[idx]);
          }

          if (bForParam) {
            // If Argument, copy before each return.
            for (auto &BB : F->getBasicBlockList()) {
              TerminatorInst *TI = BB.getTerminator();
              if (isa<ReturnInst>(TI)) {
                IRBuilder<> RetBuilder(TI);
                Value *Ptr = RetBuilder.CreateGEP(localTarget, idxList);
                Value *V = RetBuilder.CreateLoad(Ptr);
                RetBuilder.CreateStore(V, Elt);
              }
            }
          } else {
            // Else, copy with Builder.
            Value *Ptr = Builder.CreateGEP(localTarget, idxList);
            Value *V = Builder.CreateLoad(Ptr);
            Builder.CreateStore(V, Elt);
          }

          // Update arrayIdxList.
          for (unsigned idx=arrayLevel;idx>0;idx--) {
            arrayIdxList[idx-1]++;
            if (arrayIdxList[idx-1] < arraySizeList[idx-1])
              break;
            arrayIdxList[idx-1] = 0;
          }
        }
        continue;
      }

      // Cast vector/matrix/resource parameter.
      V = castArgumentIfRequired(V, Ty, bOut, inputQual,
                                  annotation, Builder, dxilTypeSys);

      // Cannot SROA, save it to final parameter list.
      FlatParamList.emplace_back(V);
      // Create ParamAnnotation for V.
      FlatAnnotationList.emplace_back(DxilParameterAnnotation());
      DxilParameterAnnotation &flatParamAnnotation = FlatAnnotationList.back();

      flatParamAnnotation.SetParamInputQual(paramAnnotation.GetParamInputQual());
            
      flatParamAnnotation.SetInterpolationMode(annotation.GetInterpolationMode());
      flatParamAnnotation.SetSemanticString(annotation.GetSemanticString());
      flatParamAnnotation.SetCompType(annotation.GetCompType().GetKind());
      flatParamAnnotation.SetMatrixAnnotation(annotation.GetMatrixAnnotation());
      flatParamAnnotation.SetPrecise(annotation.IsPrecise());
      flatParamAnnotation.SetResourceAttribute(annotation.GetResourceAttribute());

      // Add debug info.
      if (DDIs.size() && V != Arg) {
        Value *TmpV = V;
        // If V is casted, add debug into to original V.
        if (castParamMap.count(V)) {
          TmpV = castParamMap[V].first;
          // One more level for ptr of input vector.
          // It cast from ptr to non-ptr then cast to scalars.
          if (castParamMap.count(TmpV)) {
            TmpV = castParamMap[TmpV].first;
          }
        }
        Type *Ty = TmpV->getType();
        if (Ty->isPointerTy())
          Ty = Ty->getPointerElementType();
        unsigned size = DL.getTypeAllocSize(Ty);
#if 0 // HLSL Change
        DIExpression *DDIExp = DIB.createBitPieceExpression(debugOffset, size);
#else // HLSL Change
        Type *argTy = Arg->getType();
        if (argTy->isPointerTy())
          argTy = argTy->getPointerElementType();
        DIExpression *DDIExp = nullptr;
        if (debugOffset == 0 && DL.getTypeAllocSize(argTy) == size) {
          DDIExp = DIB.createExpression();
        }
        else {
          DDIExp = DIB.createBitPieceExpression(debugOffset * 8, size * 8);
        }
#endif // HLSL Change
        debugOffset += size;
        for (DbgDeclareInst *DDI : DDIs) {
          DIB.insertDeclare(TmpV, DDI->getVariable(), DDIExp, DDI->getDebugLoc(),
            Builder.GetInsertPoint());
        }
      }

      // Flatten stream out.
      if (HLModule::IsStreamOutputPtrType(V->getType())) {
        // For stream output objects.
        // Create a value as output value.
        Type *outputType = V->getType()->getPointerElementType()->getStructElementType(0);
        Value *outputVal = Builder.CreateAlloca(outputType);

        // For each stream.Append(data)
        // transform into
        //   d = load data
        //   store outputVal, d
        //   stream.Append(outputVal)
        for (User *user : V->users()) {
          if (CallInst *CI = dyn_cast<CallInst>(user)) {
            unsigned opcode = GetHLOpcode(CI);
            if (opcode == static_cast<unsigned>(IntrinsicOp::MOP_Append)) {
              // At this point, the stream append data argument might or not have been SROA'd
              Value *firstDataPtr = CI->getArgOperand(HLOperandIndex::kStreamAppendDataOpIndex);
              DXASSERT(firstDataPtr->getType()->isPointerTy(), "Append value must be a pointer.");
              if (firstDataPtr->getType()->getPointerElementType() == outputType) {
                // The data has not been SROA'd
                DXASSERT(CI->getNumArgOperands() == (HLOperandIndex::kStreamAppendDataOpIndex + 1),
                  "Unexpected number of arguments for non-SROA'd StreamOutput.Append");
                IRBuilder<> Builder(CI);

                llvm::SmallVector<llvm::Value *, 16> idxList;
                SplitCpy(firstDataPtr->getType(), outputVal, firstDataPtr, idxList, Builder, DL,
                          dxilTypeSys, &flatParamAnnotation);

                CI->setArgOperand(HLOperandIndex::kStreamAppendDataOpIndex, outputVal);
              }
              else {
                // Append has been SROA'd, we might be operating on multiple values
                // with types differing from the stream output type.
                // Flatten store outputVal.
                // Must be struct to be flatten.
                IRBuilder<> Builder(CI);

                llvm::SmallVector<llvm::Value *, 16> IdxList;
                llvm::SmallVector<llvm::Value *, 16> EltPtrList;
                llvm::SmallVector<const DxilFieldAnnotation*, 16> EltAnnotationList;
                // split
                SplitPtr(outputVal, IdxList, outputVal->getType(), flatParamAnnotation,
                  EltPtrList, EltAnnotationList, dxilTypeSys, Builder);

                unsigned eltCount = CI->getNumArgOperands()-2;
                DXASSERT_LOCALVAR(eltCount, eltCount == EltPtrList.size(), "invalid element count");

                for (unsigned i = HLOperandIndex::kStreamAppendDataOpIndex; i < CI->getNumArgOperands(); i++) {
                  Value *DataPtr = CI->getArgOperand(i);
                  Value *EltPtr = EltPtrList[i - HLOperandIndex::kStreamAppendDataOpIndex];
                  const DxilFieldAnnotation *EltAnnotation = EltAnnotationList[i - HLOperandIndex::kStreamAppendDataOpIndex];

                  llvm::SmallVector<llvm::Value *, 16> IdxList;
                  SplitCpy(DataPtr->getType(), EltPtr, DataPtr, IdxList,
                            Builder, DL, dxilTypeSys, EltAnnotation);
                  CI->setArgOperand(i, EltPtr);
                }
              }
            }
          }
        }

        // Then split output value to generate ParamQual.
        WorkList.push_front({ outputVal, annotation });
      }
    }
  }

  // Now erase any instructions that were made dead while rewriting the
  // alloca.
  DeleteDeadInstructions();
  // Erase dead allocas after all uses deleted.
  for (Instruction *I : deadAllocas)
    I->eraseFromParent();

  unsigned endArgIndex = FlatAnnotationList.size();
  if (bForParam && startArgIndex < endArgIndex) {
    DxilParamInputQual inputQual = paramAnnotation.GetParamInputQual();
    if (inputQual == DxilParamInputQual::OutStream0 ||
        inputQual == DxilParamInputQual::OutStream1 ||
        inputQual == DxilParamInputQual::OutStream2 ||
        inputQual == DxilParamInputQual::OutStream3)
      startArgIndex++;

    DxilParameterAnnotation &flatParamAnnotation =
        FlatAnnotationList[startArgIndex];
    const std::string &semantic = flatParamAnnotation.GetSemanticString();
    if (!semantic.empty())
      allocateSemanticIndex(FlatAnnotationList, startArgIndex,
                            semanticTypeMap);
  }

}

static bool IsUsedAsCallArg(Value *V) {
  for (User *U : V->users()) {
    if (CallInst *CI = dyn_cast<CallInst>(U)) {
      Function *CalledF = CI->getCalledFunction();
      HLOpcodeGroup group = GetHLOpcodeGroup(CalledF);
      // Skip HL operations.
      if (group != HLOpcodeGroup::NotHL ||
          group == HLOpcodeGroup::HLExtIntrinsic) {
        continue;
      }
      // Skip llvm intrinsic.
      if (CalledF->isIntrinsic())
        continue;

      return true;
    }
    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(U)) {
      if (IsUsedAsCallArg(GEP))
        return true;
    }
  }
  return false;
}

// For function parameter which used in function call and need to be flattened.
// Replace with tmp alloca.
void SROA_Parameter_HLSL::preprocessArgUsedInCall(Function *F) {
  if (F->isDeclaration())
    return;

  const DataLayout &DL = m_pHLModule->GetModule()->getDataLayout();

  DxilTypeSystem &typeSys = m_pHLModule->GetTypeSystem();
  DxilFunctionAnnotation *pFuncAnnot = typeSys.GetFunctionAnnotation(F);
  DXASSERT(pFuncAnnot, "else invalid function");

  IRBuilder<> Builder(dxilutil::FindAllocaInsertionPt(F));

  SmallVector<ReturnInst*, 2> retList;
  for (BasicBlock &bb : F->getBasicBlockList()) {
    if (ReturnInst *RI = dyn_cast<ReturnInst>(bb.getTerminator())) {
      retList.emplace_back(RI);
    }
  }

  for (Argument &arg : F->args()) {
    Type *Ty = arg.getType();
    // Only check pointer types.
    if (!Ty->isPointerTy())
      continue;
    Ty = Ty->getPointerElementType();
    // Skip scalar types.
    if (!Ty->isAggregateType() &&
        Ty->getScalarType() == Ty)
      continue;

    bool bUsedInCall = IsUsedAsCallArg(&arg);

    if (bUsedInCall) {
      // Create tmp.
      Value *TmpArg = Builder.CreateAlloca(Ty);
      // Replace arg with tmp.
      arg.replaceAllUsesWith(TmpArg);

      DxilParameterAnnotation &paramAnnot = pFuncAnnot->GetParameterAnnotation(arg.getArgNo());
      DxilParamInputQual inputQual = paramAnnot.GetParamInputQual();
      unsigned size = DL.getTypeAllocSize(Ty);
      // Copy between arg and tmp.
      if (inputQual == DxilParamInputQual::In ||
          inputQual == DxilParamInputQual::Inout) {
        // copy arg to tmp.
        CallInst *argToTmp = Builder.CreateMemCpy(TmpArg, &arg, size, 0);
        // Split the memcpy.
        MemcpySplitter::SplitMemCpy(cast<MemCpyInst>(argToTmp), DL, nullptr,
                                    typeSys);
      }
      if (inputQual == DxilParamInputQual::Out ||
          inputQual == DxilParamInputQual::Inout) {
        for (ReturnInst *RI : retList) {
          IRBuilder<> RetBuilder(RI);
          // copy tmp to arg.
          CallInst *tmpToArg =
              RetBuilder.CreateMemCpy(&arg, TmpArg, size, 0);
          // Split the memcpy.
          MemcpySplitter::SplitMemCpy(cast<MemCpyInst>(tmpToArg), DL, nullptr,
                                      typeSys);
        }
      }
      // TODO: support other DxilParamInputQual.

    }
  }
}

/// moveFunctionBlocks - Move body of F to flatF.
void SROA_Parameter_HLSL::moveFunctionBody(Function *F, Function *flatF) {
  bool updateRetType = F->getReturnType() != flatF->getReturnType();

  // Splice the body of the old function right into the new function.
  flatF->getBasicBlockList().splice(flatF->begin(), F->getBasicBlockList());

  // Update Block uses.
  if (updateRetType) {
    for (BasicBlock &BB : flatF->getBasicBlockList()) {
      if (updateRetType) {
        // Replace ret with ret void.
        if (ReturnInst *RI = dyn_cast<ReturnInst>(BB.getTerminator())) {
          // Create store for return.
          IRBuilder<> Builder(RI);
          Builder.CreateRetVoid();
          RI->eraseFromParent();
        }
      }
    }
  }
}

static void SplitArrayCopy(Value *V, const DataLayout &DL,
                           DxilTypeSystem &typeSys,
                           DxilFieldAnnotation *fieldAnnotation) {
  for (auto U = V->user_begin(); U != V->user_end();) {
    User *user = *(U++);
    if (StoreInst *ST = dyn_cast<StoreInst>(user)) {
      Value *ptr = ST->getPointerOperand();
      Value *val = ST->getValueOperand();
      IRBuilder<> Builder(ST);
      SmallVector<Value *, 16> idxList;
      SplitCpy(ptr->getType(), ptr, val, idxList, Builder, DL, typeSys,
               fieldAnnotation);
      ST->eraseFromParent();
    }
  }
}

static void CheckArgUsage(Value *V, bool &bLoad, bool &bStore) {
  if (bLoad && bStore)
    return;
  for (User *user : V->users()) {
    if (dyn_cast<LoadInst>(user)) {
      bLoad = true;
    } else if (dyn_cast<StoreInst>(user)) {
      bStore = true;
    } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(user)) {
      CheckArgUsage(GEP, bLoad, bStore);
    } else if (CallInst *CI = dyn_cast<CallInst>(user)) {
      if (CI->getType()->isPointerTy())
        CheckArgUsage(CI, bLoad, bStore);
      else {
        HLOpcodeGroup group = GetHLOpcodeGroupByName(CI->getCalledFunction());
        if (group == HLOpcodeGroup::HLMatLoadStore) {
          HLMatLoadStoreOpcode opcode =
              static_cast<HLMatLoadStoreOpcode>(GetHLOpcode(CI));
          switch (opcode) {
          case HLMatLoadStoreOpcode::ColMatLoad:
          case HLMatLoadStoreOpcode::RowMatLoad:
            bLoad = true;
            break;
          case HLMatLoadStoreOpcode::ColMatStore:
          case HLMatLoadStoreOpcode::RowMatStore:
            bStore = true;
            break;
          }
        }
      }
    }
  }
}

// AcceptHitAndEndSearch and IgnoreHit both will not return, but require
// outputs to have been written before the call.  Do this by:
//  - inject a return immediately after the call if not there already
//  - LegalizeDxilInputOutputs will inject writes from temp alloca to
//    outputs before each return.
//  - in HLOperationLower, after lowering the intrinsic, move the intrinsic
//    to just before the return.
static void InjectReturnAfterNoReturnPreserveOutput(HLModule &HLM) {
  for (Function &F : HLM.GetModule()->functions()) {
    if (GetHLOpcodeGroup(&F) == HLOpcodeGroup::HLIntrinsic) {
      for (auto U : F.users()) {
        if (CallInst *CI = dyn_cast<CallInst>(U)) {
          unsigned OpCode = GetHLOpcode(CI);
          if (OpCode == (unsigned)IntrinsicOp::IOP_AcceptHitAndEndSearch ||
              OpCode == (unsigned)IntrinsicOp::IOP_IgnoreHit) {
            Instruction *pNextI = CI->getNextNode();
            // Skip if already has a return immediatly following call
            if (isa<ReturnInst>(pNextI))
              continue;
            // split block and add return:
            BasicBlock *BB = CI->getParent();
            BB->splitBasicBlock(pNextI);
            TerminatorInst *Term = BB->getTerminator();
            Term->eraseFromParent();
            IRBuilder<> Builder(BB);
            llvm::Type *RetTy = CI->getParent()->getParent()->getReturnType();
            if (RetTy->isVoidTy())
              Builder.CreateRetVoid();
            else
              Builder.CreateRet(UndefValue::get(RetTy));
          }
        }
      }
    }
  }
}

// Support store to input and load from output.
static void LegalizeDxilInputOutputs(Function *F,
                                     DxilFunctionAnnotation *EntryAnnotation,
                                     const DataLayout &DL,
                                     DxilTypeSystem &typeSys) {
  BasicBlock &EntryBlk = F->getEntryBlock();
  Module *M = F->getParent();
  // Map from output to the temp created for it.
  MapVector<Argument *, Value*> outputTempMap; // Need deterministic order of iteration
  for (Argument &arg : F->args()) {
    dxilutil::MergeGepUse(&arg);
    Type *Ty = arg.getType();

    DxilParameterAnnotation &paramAnnotation = EntryAnnotation->GetParameterAnnotation(arg.getArgNo());
    DxilParamInputQual qual = paramAnnotation.GetParamInputQual();

    bool isColMajor = false;

    // Skip arg which is not a pointer.
    if (!Ty->isPointerTy()) {
      if (HLMatrixType::isa(Ty)) {
        // Replace matrix arg with cast to vec. It will be lowered in
        // DxilGenerationPass.
        isColMajor = paramAnnotation.GetMatrixAnnotation().Orientation ==
                     MatrixOrientation::ColumnMajor;
        IRBuilder<> Builder(dxilutil::FindAllocaInsertionPt(F));

        HLCastOpcode opcode = isColMajor ? HLCastOpcode::ColMatrixToVecCast
                                         : HLCastOpcode::RowMatrixToVecCast;
        Value *undefVal = UndefValue::get(Ty);

        Value *Cast = HLModule::EmitHLOperationCall(
            Builder, HLOpcodeGroup::HLCast, static_cast<unsigned>(opcode), Ty,
            {undefVal}, *M);
        arg.replaceAllUsesWith(Cast);
        // Set arg as the operand.
        CallInst *CI = cast<CallInst>(Cast);
        CI->setArgOperand(HLOperandIndex::kUnaryOpSrc0Idx, &arg);
      }
      continue;
    }

    Ty = Ty->getPointerElementType();

    bool bLoad = false;
    bool bStore = false;
    CheckArgUsage(&arg, bLoad, bStore);

    bool bStoreInputToTemp = false;
    bool bLoadOutputFromTemp = false;

    if (qual == DxilParamInputQual::In && bStore) {
      bStoreInputToTemp = true;
    } else if (qual == DxilParamInputQual::Out && bLoad) {
      bLoadOutputFromTemp = true;
    } else if (bLoad && bStore) {
      switch (qual) {
      case DxilParamInputQual::InPayload:
      case DxilParamInputQual::InputPrimitive:
      case DxilParamInputQual::InputPatch:
      case DxilParamInputQual::OutputPatch: {
        bStoreInputToTemp = true;
      } break;
      case DxilParamInputQual::Inout:
        break;
      default:
        DXASSERT(0, "invalid input qual here");
      }
    } else if (qual == DxilParamInputQual::Inout) {
      // Only replace inout when (bLoad && bStore) == false.
      bLoadOutputFromTemp = true;
      bStoreInputToTemp = true;
    }

    if (HLMatrixType::isa(Ty)) {
      if (qual == DxilParamInputQual::In)
        bStoreInputToTemp = bLoad;
      else if (qual == DxilParamInputQual::Out)
        bLoadOutputFromTemp = bStore;
      else if (qual == DxilParamInputQual::Inout) {
        bStoreInputToTemp = true;
        bLoadOutputFromTemp = true;
      }
    }

    if (bStoreInputToTemp || bLoadOutputFromTemp) {
      IRBuilder<> Builder(EntryBlk.getFirstInsertionPt());

      AllocaInst *temp = Builder.CreateAlloca(Ty);
      // Replace all uses with temp.
      arg.replaceAllUsesWith(temp);

      // Copy input to temp.
      if (bStoreInputToTemp) {
        llvm::SmallVector<llvm::Value *, 16> idxList;
        // split copy.
        SplitCpy(temp->getType(), temp, &arg, idxList, Builder, DL, typeSys,
                 &paramAnnotation);
      }

      // Generate store output, temp later.
      if (bLoadOutputFromTemp) {
        outputTempMap[&arg] = temp;
      }
    }
  }

  for (BasicBlock &BB : F->getBasicBlockList()) {
    if (ReturnInst *RI = dyn_cast<ReturnInst>(BB.getTerminator())) {
      IRBuilder<> Builder(RI);
      // Copy temp to output.
      for (auto It : outputTempMap) {
        Argument *output = It.first;
        Value *temp = It.second;
        llvm::SmallVector<llvm::Value *, 16> idxList;

        DxilParameterAnnotation &paramAnnotation =
            EntryAnnotation->GetParameterAnnotation(output->getArgNo());

        auto Iter = Builder.GetInsertPoint();
        if (RI != BB.begin())
          Iter--;
        // split copy.
        SplitCpy(output->getType(), output, temp, idxList, Builder, DL, typeSys,
                 &paramAnnotation);
      }
      // Clone the return.
      Builder.CreateRet(RI->getReturnValue());
      RI->eraseFromParent();
    }
  }
}

void SROA_Parameter_HLSL::createFlattenedFunction(Function *F) {
  DxilTypeSystem &typeSys = m_pHLModule->GetTypeSystem();

  DXASSERT(F == m_pHLModule->GetEntryFunction() ||
           m_pHLModule->IsEntryThatUsesSignatures(F),
    "otherwise, createFlattenedFunction called on library function "
    "that should not be flattened.");

  const DataLayout &DL = m_pHLModule->GetModule()->getDataLayout();

  // Skip void (void) function.
  if (F->getReturnType()->isVoidTy() && F->getArgumentList().empty()) {
    return;
  }
  // Clear maps for cast.
  castParamMap.clear();
  vectorEltsMap.clear();

  DxilFunctionAnnotation *funcAnnotation = m_pHLModule->GetFunctionAnnotation(F);
  DXASSERT(funcAnnotation, "must find annotation for function");

  std::deque<Value *> WorkList;

  LLVMContext &Ctx = m_pHLModule->GetCtx();
  std::unique_ptr<BasicBlock> TmpBlockForFuncDecl;
  BasicBlock *EntryBlock;
  if (F->isDeclaration()) {
    // We still want to SROA the parameters, so creaty a dummy
    // function body block to avoid special cases.
    TmpBlockForFuncDecl.reset(BasicBlock::Create(Ctx));
    // Create return as terminator.
    IRBuilder<> RetBuilder(TmpBlockForFuncDecl.get());
    RetBuilder.CreateRetVoid();
    EntryBlock = TmpBlockForFuncDecl.get();
  } else {
    EntryBlock = &F->getEntryBlock();
  }

  std::vector<Value *> FlatParamList;
  std::vector<DxilParameterAnnotation> FlatParamAnnotationList;
  std::vector<int> FlatParamOriArgNoList;

  const bool bForParamTrue = true;

  // Add all argument to worklist.
  for (Argument &Arg : F->args()) {
    // merge GEP use for arg.
    dxilutil::MergeGepUse(&Arg);

    unsigned prevFlatParamCount = FlatParamList.size();

    DxilParameterAnnotation &paramAnnotation =
        funcAnnotation->GetParameterAnnotation(Arg.getArgNo());
    SmallVector<DbgDeclareInst *, 4> DDIs;
    llvm::FindAllocaDbgDeclare(&Arg, DDIs);
    flattenArgument(F, &Arg, bForParamTrue, paramAnnotation, FlatParamList,
                    FlatParamAnnotationList, EntryBlock, DDIs);

    unsigned newFlatParamCount = FlatParamList.size() - prevFlatParamCount;
    for (unsigned i = 0; i < newFlatParamCount; i++) {
      FlatParamOriArgNoList.emplace_back(Arg.getArgNo());
    }
  }

  Type *retType = F->getReturnType();

  std::vector<Value *> FlatRetList;
  std::vector<DxilParameterAnnotation> FlatRetAnnotationList;
  // Split and change to out parameter.
  if (!retType->isVoidTy()) {
    IRBuilder<> Builder(dxilutil::FindAllocaInsertionPt(EntryBlock));
    Value *retValAddr = Builder.CreateAlloca(retType);
    DxilParameterAnnotation &retAnnotation =
        funcAnnotation->GetRetTypeAnnotation();
    Module &M = *m_pHLModule->GetModule();
    Type *voidTy = Type::getVoidTy(m_pHLModule->GetCtx());
#if 0 // We don't really want this to show up in debug info.
    // Create DbgDecl for the ret value.
    if (DISubprogram *funcDI = getDISubprogram(F)) {
        DITypeRef RetDITyRef = funcDI->getType()->getTypeArray()[0];
        DITypeIdentifierMap EmptyMap;
        DIType * RetDIType = RetDITyRef.resolve(EmptyMap);
        DIBuilder DIB(*F->getParent(), /*AllowUnresolved*/ false);
        DILocalVariable *RetVar = DIB.createLocalVariable(llvm::dwarf::Tag::DW_TAG_arg_variable, funcDI, F->getName().str() + ".Ret", funcDI->getFile(),
            funcDI->getLine(), RetDIType);
        DIExpression *Expr = DIB.createExpression();
        // TODO: how to get col?
        DILocation *DL = DILocation::get(F->getContext(), funcDI->getLine(), 0, funcDI);
        DIB.insertDeclare(retValAddr, RetVar, Expr, DL, Builder.GetInsertPoint());
    }
#endif
    for (BasicBlock &BB : F->getBasicBlockList()) {
      if (ReturnInst *RI = dyn_cast<ReturnInst>(BB.getTerminator())) {
        // Create store for return.
        IRBuilder<> RetBuilder(RI);
        if (!retAnnotation.HasMatrixAnnotation()) {
          RetBuilder.CreateStore(RI->getReturnValue(), retValAddr);
        } else {
          bool isRowMajor = retAnnotation.GetMatrixAnnotation().Orientation ==
                            MatrixOrientation::RowMajor;
          Value *RetVal = RI->getReturnValue();
          if (!isRowMajor) {
            // Matrix value is row major. ColMatStore require col major.
            // Cast before store.
            RetVal = HLModule::EmitHLOperationCall(
                RetBuilder, HLOpcodeGroup::HLCast,
                static_cast<unsigned>(HLCastOpcode::RowMatrixToColMatrix),
                RetVal->getType(), {RetVal}, M);
          }
          unsigned opcode = static_cast<unsigned>(
              isRowMajor ? HLMatLoadStoreOpcode::RowMatStore
                          : HLMatLoadStoreOpcode::ColMatStore);
          HLModule::EmitHLOperationCall(RetBuilder,
                                        HLOpcodeGroup::HLMatLoadStore, opcode,
                                        voidTy, {retValAddr, RetVal}, M);
        }
      }
    }
    // Create a fake store to keep retValAddr so it can be flattened.
    if (retValAddr->user_empty()) {
      Builder.CreateStore(UndefValue::get(retType), retValAddr);
    }

    SmallVector<DbgDeclareInst *, 4> DDIs;
    llvm::FindAllocaDbgDeclare(retValAddr, DDIs);
    flattenArgument(F, retValAddr, bForParamTrue,
                    funcAnnotation->GetRetTypeAnnotation(), FlatRetList,
                    FlatRetAnnotationList, EntryBlock, DDIs);

    const int kRetArgNo = -1;
    for (unsigned i = 0; i < FlatRetList.size(); i++) {
      FlatParamOriArgNoList.insert(FlatParamOriArgNoList.begin(), kRetArgNo);
    }
  }

  // Always change return type as parameter.
  // By doing this, no need to check return when generate storeOutput.
  if (FlatRetList.size() ||
      // For empty struct return type.
      !retType->isVoidTy()) {
    // Return value is flattened.
    // Change return value into out parameter.
    retType = Type::getVoidTy(retType->getContext());
    // Merge return data info param data.

    FlatParamList.insert(FlatParamList.begin(), FlatRetList.begin(), FlatRetList.end());
    FlatParamAnnotationList.insert(FlatParamAnnotationList.begin(),
                                    FlatRetAnnotationList.begin(),
                                    FlatRetAnnotationList.end());
  }


  std::vector<Type *> FinalTypeList;
  for (Value * arg : FlatParamList) {
    FinalTypeList.emplace_back(arg->getType());
  }

  unsigned extraParamSize = 0;
  if (m_pHLModule->HasDxilFunctionProps(F)) {
    DxilFunctionProps &funcProps = m_pHLModule->GetDxilFunctionProps(F);
    if (funcProps.shaderKind == ShaderModel::Kind::Vertex) {
      auto &VS = funcProps.ShaderProps.VS;
      Type *outFloatTy = Type::getFloatPtrTy(F->getContext());
      // Add out float parameter for each clip plane.
      unsigned i=0;
      for (; i < DXIL::kNumClipPlanes; i++) {
        if (!VS.clipPlanes[i])
          break;
        FinalTypeList.emplace_back(outFloatTy);
      }
      extraParamSize = i;
    }
  }

  FunctionType *flatFuncTy = FunctionType::get(retType, FinalTypeList, false);
  // Return if nothing changed.
  if (flatFuncTy == F->getFunctionType()) {
    // Copy semantic allocation.
    if (!FlatParamAnnotationList.empty()) {
      if (!FlatParamAnnotationList[0].GetSemanticString().empty()) {
        for (unsigned i = 0; i < FlatParamAnnotationList.size(); i++) {
          DxilParameterAnnotation &paramAnnotation = funcAnnotation->GetParameterAnnotation(i);
          DxilParameterAnnotation &flatParamAnnotation = FlatParamAnnotationList[i];
          paramAnnotation.SetSemanticIndexVec(flatParamAnnotation.GetSemanticIndexVec());
          paramAnnotation.SetSemanticString(flatParamAnnotation.GetSemanticString());
        }
      }
    }
    if (!F->isDeclaration()) {
      // Support store to input and load from output.
      LegalizeDxilInputOutputs(F, funcAnnotation, DL, typeSys);
    }
    return;
  }

  std::string flatName = F->getName().str() + ".flat";
  DXASSERT(nullptr == F->getParent()->getFunction(flatName),
           "else overwriting existing function");
  Function *flatF =
      cast<Function>(F->getParent()->getOrInsertFunction(flatName, flatFuncTy));
  funcMap[F] = flatF;

  // Update function debug info.
  if (DISubprogram *funcDI = getDISubprogram(F))
    funcDI->replaceFunction(flatF);

  // Create FunctionAnnotation for flatF.
  DxilFunctionAnnotation *flatFuncAnnotation = m_pHLModule->AddFunctionAnnotation(flatF);
  
  // Don't need to set Ret Info, flatF always return void now.

  // Param Info
  for (unsigned ArgNo = 0; ArgNo < FlatParamAnnotationList.size(); ++ArgNo) {
    DxilParameterAnnotation &paramAnnotation = flatFuncAnnotation->GetParameterAnnotation(ArgNo);
    paramAnnotation = FlatParamAnnotationList[ArgNo];
  }

  // Function Attr and Parameter Attr.
  // Remove sret first.
  if (F->hasStructRetAttr())
    F->removeFnAttr(Attribute::StructRet);
  for (Argument &arg : F->args()) {
    if (arg.hasStructRetAttr()) {
      Attribute::AttrKind SRet [] = {Attribute::StructRet};
      AttributeSet SRetAS = AttributeSet::get(Ctx, arg.getArgNo() + 1, SRet);
      arg.removeAttr(SRetAS);
    }
  }

  AttributeSet AS = F->getAttributes();
  AttrBuilder FnAttrs(AS.getFnAttributes(), AttributeSet::FunctionIndex);
  AttributeSet flatAS;
  flatAS = flatAS.addAttributes(
      Ctx, AttributeSet::FunctionIndex,
      AttributeSet::get(Ctx, AttributeSet::FunctionIndex, FnAttrs));
  if (!F->isDeclaration()) {
    // Only set Param attribute for function has a body.
    for (unsigned ArgNo = 0; ArgNo < FlatParamAnnotationList.size(); ++ArgNo) {
      unsigned oriArgNo = FlatParamOriArgNoList[ArgNo] + 1;
      AttrBuilder paramAttr(AS, oriArgNo);
      if (oriArgNo == AttributeSet::ReturnIndex)
        paramAttr.addAttribute(Attribute::AttrKind::NoAlias);
      flatAS = flatAS.addAttributes(
          Ctx, ArgNo + 1, AttributeSet::get(Ctx, ArgNo + 1, paramAttr));
    }
  }
  flatF->setAttributes(flatAS);

  DXASSERT_LOCALVAR(extraParamSize, flatF->arg_size() == (extraParamSize + FlatParamAnnotationList.size()), "parameter count mismatch");
  // ShaderProps.
  if (m_pHLModule->HasDxilFunctionProps(F)) {
    DxilFunctionProps &funcProps = m_pHLModule->GetDxilFunctionProps(F);
    std::unique_ptr<DxilFunctionProps> flatFuncProps = llvm::make_unique<DxilFunctionProps>();
    *flatFuncProps = funcProps;
    m_pHLModule->AddDxilFunctionProps(flatF, flatFuncProps);
    if (funcProps.shaderKind == ShaderModel::Kind::Vertex) {
      auto &VS = funcProps.ShaderProps.VS;
      unsigned clipArgIndex = FlatParamAnnotationList.size();
      // Add out float SV_ClipDistance for each clip plane.
      for (unsigned i = 0; i < DXIL::kNumClipPlanes; i++) {
        if (!VS.clipPlanes[i])
          break;
        DxilParameterAnnotation &paramAnnotation =
            flatFuncAnnotation->GetParameterAnnotation(clipArgIndex+i);
        paramAnnotation.SetParamInputQual(DxilParamInputQual::Out);
        Twine semName = Twine("SV_ClipDistance") + Twine(i);
        paramAnnotation.SetSemanticString(semName.str());
        paramAnnotation.SetCompType(DXIL::ComponentType::F32);
        paramAnnotation.AppendSemanticIndex(i);
      }
    }
  }

  if (!F->isDeclaration()) {
    // Move function body into flatF.
    moveFunctionBody(F, flatF);

    // Replace old parameters with flatF Arguments.
    auto argIter = flatF->arg_begin();
    auto flatArgIter = FlatParamList.begin();
    LLVMContext &Context = F->getContext();

    // Parameter cast come from begining of entry block.
    IRBuilder<> Builder(dxilutil::FindAllocaInsertionPt(flatF));

    while (argIter != flatF->arg_end()) {
      Argument *Arg = argIter++;
      if (flatArgIter == FlatParamList.end()) {
        DXASSERT(extraParamSize > 0, "parameter count mismatch");
        break;
      }
      Value *flatArg = *(flatArgIter++);

      if (castParamMap.count(flatArg)) {
        replaceCastParameter(flatArg, castParamMap[flatArg].first, *flatF, Arg,
                             castParamMap[flatArg].second, Builder);
      }

      // Update arg debug info.
      SmallVector<DbgDeclareInst *, 4> DDIs;
      llvm::FindAllocaDbgDeclare(flatArg, DDIs);
      if (DDIs.size()) {
        Value *VMD = nullptr;
        if (!flatArg->getType()->isPointerTy()) {
          // Create alloca to hold the debug info.
          Value *allocaArg = nullptr;
          if (flatArg->hasOneUse() && isa<StoreInst>(*flatArg->user_begin())) {
            StoreInst *SI = cast<StoreInst>(*flatArg->user_begin());
            allocaArg = SI->getPointerOperand();
          } else {
            allocaArg = Builder.CreateAlloca(flatArg->getType());
            StoreInst *initArg = Builder.CreateStore(flatArg, allocaArg);
            Value *ldArg = Builder.CreateLoad(allocaArg);
            flatArg->replaceAllUsesWith(ldArg);
            initArg->setOperand(0, flatArg);
          }
          VMD = MetadataAsValue::get(Context, ValueAsMetadata::get(allocaArg));
        } else {
          VMD = MetadataAsValue::get(Context, ValueAsMetadata::get(Arg));
        }
        for (DbgDeclareInst *DDI : DDIs) {
          DDI->setArgOperand(0, VMD);
        }
      }

      flatArg->replaceAllUsesWith(Arg);
      if (isa<Instruction>(flatArg))
        DeadInsts.emplace_back(flatArg);

      dxilutil::MergeGepUse(Arg);
      // Flatten store of array parameter.
      if (Arg->getType()->isPointerTy()) {
        Type *Ty = Arg->getType()->getPointerElementType();
        if (Ty->isArrayTy())
          SplitArrayCopy(
              Arg, DL, typeSys,
              &flatFuncAnnotation->GetParameterAnnotation(Arg->getArgNo()));
      }
    }
    // Support store to input and load from output.
    LegalizeDxilInputOutputs(flatF, flatFuncAnnotation, DL, typeSys);
  }
}

void SROA_Parameter_HLSL::replaceCall(Function *F, Function *flatF) {
  // Update entry function.
  if (F == m_pHLModule->GetEntryFunction()) {
    m_pHLModule->SetEntryFunction(flatF);
  }

  DXASSERT(F->user_empty(), "otherwise we flattened a library function.");
}

// Public interface to the SROA_Parameter_HLSL pass
ModulePass *llvm::createSROA_Parameter_HLSL() {
  return new SROA_Parameter_HLSL();
}

//===----------------------------------------------------------------------===//
// Lower static global into Alloca.
//===----------------------------------------------------------------------===//

namespace {
class LowerStaticGlobalIntoAlloca : public ModulePass {
  DebugInfoFinder m_DbgFinder;

public:
  static char ID; // Pass identification, replacement for typeid
  explicit LowerStaticGlobalIntoAlloca() : ModulePass(ID) {}
  StringRef getPassName() const override { return "Lower static global into Alloca"; }

  bool runOnModule(Module &M) override {
    m_DbgFinder.processModule(M);
    Type *handleTy = nullptr;
    DxilTypeSystem *pTypeSys = nullptr;
    SetVector<Function *> entryAndInitFunctionSet;
    if (M.HasHLModule()) {
      auto &HLM = M.GetHLModule();
      pTypeSys = &HLM.GetTypeSystem();
      handleTy = HLM.GetOP()->GetHandleType();
      if (!HLM.GetShaderModel()->IsLib()) {
        entryAndInitFunctionSet.insert(HLM.GetEntryFunction());
        if (HLM.GetShaderModel()->IsHS()) {
          entryAndInitFunctionSet.insert(HLM.GetPatchConstantFunction());
        }
      } else {
        for (Function &F : M) {
          if (F.isDeclaration() || !HLM.IsEntry(&F)) {
            continue;
          }
          entryAndInitFunctionSet.insert(&F);
        }
      }
    } else {
      DXASSERT(M.HasDxilModule(), "must have dxilModle or HLModule");
      auto &DM = M.GetDxilModule();
      pTypeSys = &DM.GetTypeSystem();
      handleTy = DM.GetOP()->GetHandleType();
      if (!DM.GetShaderModel()->IsLib()) {
        entryAndInitFunctionSet.insert(DM.GetEntryFunction());
        if (DM.GetShaderModel()->IsHS()) {
          entryAndInitFunctionSet.insert(DM.GetPatchConstantFunction());
        }
      } else {
        for (Function &F : M) {
          if (F.isDeclaration() || !DM.IsEntry(&F))
            continue;
          entryAndInitFunctionSet.insert(&F);
        }
      }
    }

    // Collect init functions for static globals.
    if (GlobalVariable *Ctors = M.getGlobalVariable("llvm.global_ctors")) {
      if (ConstantArray *CA =
              dyn_cast<ConstantArray>(Ctors->getInitializer())) {
        for (User::op_iterator i = CA->op_begin(), e = CA->op_end(); i != e;
             ++i) {
          if (isa<ConstantAggregateZero>(*i))
            continue;
          ConstantStruct *CS = cast<ConstantStruct>(*i);
          if (isa<ConstantPointerNull>(CS->getOperand(1)))
            continue;

          // Must have a function or null ptr.
          if (!isa<Function>(CS->getOperand(1)))
            continue;
          Function *Ctor = cast<Function>(CS->getOperand(1));
          assert(Ctor->getReturnType()->isVoidTy() && Ctor->arg_size() == 0 &&
                 "function type must be void (void)");
          // Add Ctor.
          entryAndInitFunctionSet.insert(Ctor);
        }
      }
    }

    // Lower static global into allocas.
    std::vector<GlobalVariable *> staticGVs;
    for (GlobalVariable &GV : M.globals()) {
      // only for non-constant static globals
      if (!dxilutil::IsStaticGlobal(&GV) || GV.isConstant())
        continue;
      // Skip dx.ishelper
      if (GV.getName().compare(DXIL::kDxIsHelperGlobalName) == 0)
        continue;
      // Skip if GV used in functions other than entry.
      if (!usedOnlyInEntry(&GV, entryAndInitFunctionSet))
        continue;
      Type *EltTy = GV.getType()->getElementType();
      if (!EltTy->isAggregateType()) {
        staticGVs.emplace_back(&GV);
      } else {
        EltTy = dxilutil::GetArrayEltTy(EltTy);
        // Lower static [array of] resources
        if (dxilutil::IsHLSLObjectType(EltTy) ||
            EltTy == handleTy) {
          staticGVs.emplace_back(&GV);
        }
      }
    }
    bool bUpdated = false;

    const DataLayout &DL = M.getDataLayout();
    // Create AI for each GV in each entry.
    // Replace all users of GV with AI.
    // Collect all users of GV within each entry.
    // Remove unused AI in the end.
    for (GlobalVariable *GV : staticGVs) {
      bUpdated |= lowerStaticGlobalIntoAlloca(GV, DL, *pTypeSys, entryAndInitFunctionSet);
    }

    return bUpdated;
  }

private:
  bool lowerStaticGlobalIntoAlloca(GlobalVariable *GV, const DataLayout &DL,
                                   DxilTypeSystem &typeSys,
                                   SetVector<Function *> &entryAndInitFunctionSet);
  bool usedOnlyInEntry(Value *V, SetVector<Function *> &entryAndInitFunctionSet);
};
}

// Go through the base type chain of TyA and see if
// we eventually get to TyB
//
// Note: Not necessarily about inheritance. Could be
// typedef, const type, ref type, MEMBER type (TyA
// being a member of TyB).
//
static bool IsDerivedTypeOf(DIType *TyA, DIType *TyB) {
  DITypeIdentifierMap EmptyMap;
  while (TyA) {
    if (DIDerivedType *Derived = dyn_cast<DIDerivedType>(TyA)) {
      if (Derived->getBaseType() == TyB)
        return true;
      else
        TyA = Derived->getBaseType().resolve(EmptyMap);
    }
    else {
      break;
    }
  }

  return false;
}

// See if 'DGV' a member type of some other variable, and return that variable
// and the offset and size DGV is into it.
//
// If DGV is not a member, just return nullptr.
//
static DIGlobalVariable *FindGlobalVariableFragment(const DebugInfoFinder &DbgFinder, DIGlobalVariable *DGV, unsigned *Out_OffsetInBits, unsigned *Out_SizeInBits) {
  DITypeIdentifierMap EmptyMap;

  StringRef FullName = DGV->getName();
  size_t FirstDot = FullName.find_first_of('.');
  if (FirstDot == StringRef::npos)
    return nullptr;

  StringRef BaseName = FullName.substr(0, FirstDot);
  assert(BaseName.size());

  DIType *Ty = DGV->getType().resolve(EmptyMap);
  assert(isa<DIDerivedType>(Ty) && Ty->getTag() == dwarf::DW_TAG_member);

  DIGlobalVariable *FinalResult = nullptr;

  for (DIGlobalVariable *DGV_It : DbgFinder.global_variables()) {
    if (DGV_It->getName() == BaseName &&
      IsDerivedTypeOf(Ty, DGV_It->getType().resolve(EmptyMap)))
    {
      FinalResult = DGV_It;
      break;
    }
  }

  if (FinalResult) {
    *Out_OffsetInBits = Ty->getOffsetInBits();
    *Out_SizeInBits = Ty->getSizeInBits();
  }

  return FinalResult;
}

// Create a fake local variable for the GlobalVariable GV that has just been
// lowered to local Alloca.
//
static
void PatchDebugInfo(DebugInfoFinder &DbgFinder, Function *F, GlobalVariable *GV, AllocaInst *AI) {
  if (!DbgFinder.compile_unit_count())
    return;

  // Find the subprogram for function
  DISubprogram *Subprogram = nullptr;
  for (DISubprogram *SP : DbgFinder.subprograms()) {
    if (SP->getFunction() == F) {
      Subprogram = SP;
      break;
    }
  }

  DIGlobalVariable *DGV = dxilutil::FindGlobalVariableDebugInfo(GV, DbgFinder);
  if (!DGV)
    return;

  DITypeIdentifierMap EmptyMap;
  DIBuilder DIB(*GV->getParent());
  DIScope *Scope = Subprogram;
  DebugLoc Loc = DebugLoc::get(DGV->getLine(), 0, Scope);

  // If the variable is a member of another variable, find the offset and size
  bool IsFragment = false;
  unsigned OffsetInBits = 0,
           SizeInBits = 0;
  if (DIGlobalVariable *UnsplitDGV = FindGlobalVariableFragment(DbgFinder, DGV, &OffsetInBits, &SizeInBits)) {
    DGV = UnsplitDGV;
    IsFragment = true;
  }

  std::string Name = "global.";
  Name += DGV->getName();
  // Using arg_variable instead of auto_variable because arg variables can use
  // Subprogram as its scope, so we don't have to make one up for it.
  llvm::dwarf::Tag Tag = llvm::dwarf::Tag::DW_TAG_arg_variable;

  DIType *Ty = DGV->getType().resolve(EmptyMap);
  DXASSERT(Ty->getTag() != dwarf::DW_TAG_member, "Member type is not allowed for variables.");
  DILocalVariable *ConvertedLocalVar =
    DIB.createLocalVariable(Tag, Scope,
      Name, DGV->getFile(), DGV->getLine(), Ty);

  DIExpression *Expr = nullptr;
  if (IsFragment) {
    Expr = DIB.createBitPieceExpression(OffsetInBits, SizeInBits);
  }
  else {
    Expr = DIB.createExpression(ArrayRef<int64_t>());
  }

  DIB.insertDeclare(AI, ConvertedLocalVar, Expr, Loc, AI->getNextNode());
}

//Collect instructions using GV and the value used by the instruction.
//For direct use, the value == GV
//For constant operator like GEP/Bitcast, the value is the operator used by the instruction.
//This requires recursion to unwrap nested constant operators using the GV.
static void collectGVInstUsers(Value *V,
                               DenseMap<Instruction *, Value *> &InstUserMap) {
  for (User *U : V->users()) {
    if (Instruction *I = dyn_cast<Instruction>(U)) {
      InstUserMap[I] = V;
    } else {
      collectGVInstUsers(U, InstUserMap);
    }
  }
}

static Instruction *replaceGVUseWithAI(GlobalVariable *GV, AllocaInst *AI,
                                       Value *U, IRBuilder<> &B) {
  if (U == GV)
    return AI;
  if (GEPOperator *GEP = dyn_cast<GEPOperator>(U)) {
    Instruction *PtrInst =
        replaceGVUseWithAI(GV, AI, GEP->getPointerOperand(), B);
    SmallVector<Value *, 2> Index(GEP->idx_begin(), GEP->idx_end());
    return cast<Instruction>(B.CreateGEP(PtrInst, Index));
  }

  if (BitCastOperator *BCO = dyn_cast<BitCastOperator>(U)) {
    Instruction *SrcInst = replaceGVUseWithAI(GV, AI, BCO->getOperand(0), B);
    return cast<Instruction>(B.CreateBitCast(SrcInst, BCO->getType()));
  }
  DXASSERT(false, "unsupported user of static global");
  return nullptr;
}

bool LowerStaticGlobalIntoAlloca::lowerStaticGlobalIntoAlloca(
    GlobalVariable *GV, const DataLayout &DL, DxilTypeSystem &typeSys,
    SetVector<Function *> &entryAndInitFunctionSet) {
  GV->removeDeadConstantUsers();
  bool bIsObjectTy = dxilutil::IsHLSLObjectType(
      dxilutil::StripArrayTypes(GV->getType()->getElementType()));
  // Create alloca for each entry.
  DenseMap<Function *, AllocaInst *> allocaMap;
  for (Function *F : entryAndInitFunctionSet) {
    IRBuilder<> Builder(dxilutil::FindAllocaInsertionPt(F));
    AllocaInst *AI = Builder.CreateAlloca(GV->getType()->getElementType());
    allocaMap[F] = AI;
    // Store initializer is exist.
    if (GV->hasInitializer() && !isa<UndefValue>(GV->getInitializer()) &&
        !bIsObjectTy) { // Do not zerio-initialize object allocas
      Builder.CreateStore(GV->getInitializer(), GV);
    }
  }

  DenseMap<Instruction *, Value *> InstUserMap;
  collectGVInstUsers(GV, InstUserMap);

  for (auto it : InstUserMap) {
    Instruction *I = it.first;
    Value *U = it.second;

    Function *F = I->getParent()->getParent();
    AllocaInst *AI = allocaMap[F];

    IRBuilder<> B(I);

    Instruction *UI = replaceGVUseWithAI(GV, AI, U, B);
    I->replaceUsesOfWith(U, UI);
  }

  for (Function *F : entryAndInitFunctionSet) {
    AllocaInst *AI = allocaMap[F];
    if (AI->user_empty())
      AI->eraseFromParent();
    else
      PatchDebugInfo(m_DbgFinder, F, GV, AI);
  }

  GV->removeDeadConstantUsers();
  if (GV->user_empty())
    GV->eraseFromParent();
  return true;
}

bool LowerStaticGlobalIntoAlloca::usedOnlyInEntry(
    Value *V, SetVector<Function *> &entryAndInitFunctionSet) {
  bool bResult = true;
  for (User *U : V->users()) {
    if (Instruction *I = dyn_cast<Instruction>(U)) {
      Function *F = I->getParent()->getParent();

      if (entryAndInitFunctionSet.count(F) == 0) {
        bResult = false;
        break;
      }
    } else {
      bResult = usedOnlyInEntry(U, entryAndInitFunctionSet);
      if (!bResult)
        break;
    }
  }
  return bResult;
}

char LowerStaticGlobalIntoAlloca::ID = 0;

INITIALIZE_PASS(LowerStaticGlobalIntoAlloca, "static-global-to-alloca",
  "Lower static global into Alloca", false,
  false)

// Public interface to the LowerStaticGlobalIntoAlloca pass
ModulePass *llvm::createLowerStaticGlobalIntoAlloca() {
  return new LowerStaticGlobalIntoAlloca();
}
