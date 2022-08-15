///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// HLMatrixLowerPass.cpp                                                     //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// HLMatrixLowerPass implementation.                                         //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/HLSL/HLMatrixLowerPass.h"
#include "dxc/HLSL/HLMatrixLowerHelper.h"
#include "dxc/HLSL/HLMatrixType.h"
#include "dxc/HLSL/HLOperations.h"
#include "dxc/HLSL/HLModule.h"
#include "dxc/HlslIntrinsicOp.h"
#include "dxc/Support/Global.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DXIL/DxilTypeSystem.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/DXIL/DxilUtil.h"
#include "HLMatrixSubscriptUseReplacer.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/ValueTracking.h"
#include <unordered_set>
#include <vector>

using namespace llvm;
using namespace hlsl;
using namespace hlsl::HLMatrixLower;
namespace hlsl {
namespace HLMatrixLower {

Value *BuildVector(Type *EltTy, ArrayRef<llvm::Value *> elts, IRBuilder<> &Builder) {
  Value *Vec = UndefValue::get(VectorType::get(EltTy, static_cast<unsigned>(elts.size())));
  for (unsigned i = 0; i < elts.size(); i++)
    Vec = Builder.CreateInsertElement(Vec, elts[i], i);
  return Vec;
}

} // namespace HLMatrixLower
} // namespace hlsl

namespace {

// Creates and manages a set of temporary overloaded functions keyed on the function type,
// and which should be destroyed when the pool gets out of scope.
class TempOverloadPool {
public:
  TempOverloadPool(llvm::Module &Module, const char* BaseName)
    : Module(Module), BaseName(BaseName) {}
  ~TempOverloadPool() { clear(); }

  Function *get(FunctionType *Ty);
  bool contains(FunctionType *Ty) const { return Funcs.count(Ty) != 0; }
  bool contains(Function *Func) const;
  void clear();

private:
  llvm::Module &Module;
  const char* BaseName;
  llvm::DenseMap<FunctionType*, Function*> Funcs;
};

Function *TempOverloadPool::get(FunctionType *Ty) {
  auto It = Funcs.find(Ty);
  if (It != Funcs.end()) return It->second;

  std::string MangledName;
  raw_string_ostream MangledNameStream(MangledName);
  MangledNameStream << BaseName;
  MangledNameStream << '.';
  Ty->print(MangledNameStream);
  MangledNameStream.flush();

  Function* Func = cast<Function>(Module.getOrInsertFunction(MangledName, Ty));
  Funcs.insert(std::make_pair(Ty, Func));
  return Func;
}

bool TempOverloadPool::contains(Function *Func) const {
  auto It = Funcs.find(Func->getFunctionType());
  return It != Funcs.end() && It->second == Func;
}

void TempOverloadPool::clear() {
  for (auto Entry : Funcs) {
    DXASSERT(Entry.second->use_empty(), "Temporary function still used during pool destruction.");
    Entry.second->eraseFromParent();
  }
  Funcs.clear();
}

// High-level matrix lowering pass.
//
// This pass converts matrices to their lowered vector representations,
// including global variables, local variables and operations,
// but not function signatures (arguments and return types) - left to HLSignatureLower and HLMatrixBitcastLower,
// nor matrices obtained from resources or constant - left to HLOperationLower.
//
// Algorithm overview:
// 1. Find all matrix and matrix array global variables and lower them to vectors.
//    Walk any GEPs and insert vec-to-mat translation stubs so that consuming
//    instructions keep dealing with matrix types for the moment.
// 2. For each function
// 2a. Lower all matrix and matrix array allocas, just like global variables.
// 2b. Lower all other instructions producing or consuming matrices
//
// Conversion stubs are used to allow converting instructions in isolation,
// and in an order-independent manner:
//
// Initial: MatInst1(MatInst2(MatInst3))
// After lowering MatInst2: MatInst1(VecToMat(VecInst2(MatToVec(MatInst3))))
// After lowering MatInst1: VecInst1(VecInst2(MatToVec(MatInst3)))
// After lowering MatInst3: VecInst1(VecInst2(VecInst3))
class HLMatrixLowerPass : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  explicit HLMatrixLowerPass() : ModulePass(ID) {}

  StringRef getPassName() const override { return "HL matrix lower"; }
  bool runOnModule(Module &M) override;

private:
  void runOnFunction(Function &Func);
  void addToDeadInsts(Instruction *Inst) { m_deadInsts.emplace_back(Inst); }
  void deleteDeadInsts();

  void getMatrixAllocasAndOtherInsts(Function &Func,
    std::vector<AllocaInst*> &MatAllocas, std::vector<Instruction*> &MatInsts);
  Value *getLoweredByValOperand(Value *Val, IRBuilder<> &Builder, bool DiscardStub = false);
  Value *tryGetLoweredPtrOperand(Value *Ptr, IRBuilder<> &Builder, bool DiscardStub = false);
  Value *bitCastValue(Value *SrcVal, Type* DstTy, bool DstTyAlloca, IRBuilder<> &Builder);
  void replaceAllUsesByLoweredValue(Instruction *MatInst, Value *VecVal);
  void replaceAllVariableUses(Value* MatPtr, Value* LoweredPtr);
  void replaceAllVariableUses(SmallVectorImpl<Value*> &GEPIdxStack, Value *StackTopPtr, Value* LoweredPtr);
  Value *translateScalarMatMul(Value *scalar, Value *mat, IRBuilder<> &Builder, bool isLhsScalar = true);

  void lowerGlobal(GlobalVariable *Global);
  Constant *lowerConstInitVal(Constant *Val);
  AllocaInst *lowerAlloca(AllocaInst *MatAlloca);
  void lowerInstruction(Instruction* Inst);
  void lowerReturn(ReturnInst* Return);
  Value *lowerCall(CallInst *Call);
  Value *lowerNonHLCall(CallInst *Call);
  void lowerPreciseCall(CallInst *Call, IRBuilder<> Builder);
  Value *lowerHLOperation(CallInst *Call, HLOpcodeGroup OpcodeGroup);
  Value *lowerHLIntrinsic(CallInst *Call, IntrinsicOp Opcode);
  Value *lowerHLMulIntrinsic(Value* Lhs, Value *Rhs, bool Unsigned, IRBuilder<> &Builder);
  Value *lowerHLTransposeIntrinsic(Value *MatVal, IRBuilder<> &Builder);
  Value *lowerHLDeterminantIntrinsic(Value *MatVal, IRBuilder<> &Builder);
  Value *lowerHLUnaryOperation(Value *MatVal, HLUnaryOpcode Opcode, IRBuilder<> &Builder);
  Value *lowerHLBinaryOperation(Value *Lhs, Value *Rhs, HLBinaryOpcode Opcode, IRBuilder<> &Builder);
  Value *lowerHLLoadStore(CallInst *Call, HLMatLoadStoreOpcode Opcode);
  Value *lowerHLLoad(CallInst *Call, Value *MatPtr, bool RowMajor, IRBuilder<> &Builder);
  Value *lowerHLStore(CallInst *Call, Value *MatVal, Value *MatPtr, bool RowMajor, bool Return, IRBuilder<> &Builder);
  Value *lowerHLCast(CallInst *Call, Value *Src, Type *DstTy, HLCastOpcode Opcode, IRBuilder<> &Builder);
  Value *lowerHLSubscript(CallInst *Call, HLSubscriptOpcode Opcode);
  Value *lowerHLMatElementSubscript(CallInst *Call, bool RowMajor);
  Value *lowerHLMatSubscript(CallInst *Call, bool RowMajor);
  void lowerHLMatSubscript(CallInst *Call, Value *MatPtr, SmallVectorImpl<Value*> &ElemIndices);
  Value *lowerHLInit(CallInst *Call);
  Value *lowerHLSelect(CallInst *Call);

private:
  Module *m_pModule;
  HLModule *m_pHLModule;
  bool m_HasDbgInfo;

  // Pools for the translation stubs
  TempOverloadPool *m_matToVecStubs = nullptr;
  TempOverloadPool *m_vecToMatStubs = nullptr;

  std::vector<Instruction *> m_deadInsts;
};
}

char HLMatrixLowerPass::ID = 0;

ModulePass *llvm::createHLMatrixLowerPass() { return new HLMatrixLowerPass(); }

INITIALIZE_PASS(HLMatrixLowerPass, "hlmatrixlower", "HLSL High-Level Matrix Lower", false, false)

bool HLMatrixLowerPass::runOnModule(Module &M) {
  TempOverloadPool matToVecStubs(M, "hlmatrixlower.mat2vec");
  TempOverloadPool vecToMatStubs(M, "hlmatrixlower.vec2mat");

  m_pModule = &M;
  m_pHLModule = &m_pModule->GetOrCreateHLModule();
  // Load up debug information, to cross-reference values and the instructions
  // used to load them.
  m_HasDbgInfo = hasDebugInfo(M);
  m_matToVecStubs = &matToVecStubs;
  m_vecToMatStubs = &vecToMatStubs;

  // First, lower static global variables.
  // We need to accumulate them locally because we'll be creating new ones as we lower them.
  std::vector<GlobalVariable*> Globals;
  for (GlobalVariable &Global : M.globals()) {
    if ((dxilutil::IsStaticGlobal(&Global) || dxilutil::IsSharedMemoryGlobal(&Global))
      && HLMatrixType::isMatrixPtrOrArrayPtr(Global.getType())) {
      Globals.emplace_back(&Global);
    }
  }

  for (GlobalVariable *Global : Globals)
    lowerGlobal(Global);

  for (Function &F : M.functions()) {
    if (F.isDeclaration()) continue;
    runOnFunction(F);
  }

  m_pModule = nullptr;
  m_pHLModule = nullptr;
  m_matToVecStubs = nullptr;
  m_vecToMatStubs = nullptr;

  // If you hit an assert during TempOverloadPool destruction,
  // it means that either a matrix producer was lowered,
  // causing a translation stub to be created,
  // but the consumer of that matrix was never (properly) lowered.
  // Or the opposite: a matrix consumer was lowered and not its producer.

  return true;
}

void HLMatrixLowerPass::runOnFunction(Function &Func) {
  // Skip hl function definition (like createhandle)
  if (hlsl::GetHLOpcodeGroupByName(&Func) != HLOpcodeGroup::NotHL)
    return;

  // Save the matrix instructions first since the translation process
  // will temporarily create other instructions consuming/producing matrix types.
  std::vector<AllocaInst*> MatAllocas;
  std::vector<Instruction*> MatInsts;
  getMatrixAllocasAndOtherInsts(Func, MatAllocas, MatInsts);

  // First lower all allocas and take care of their GEP chains
  for (AllocaInst* MatAlloca : MatAllocas) {
    AllocaInst* LoweredAlloca = lowerAlloca(MatAlloca);
    replaceAllVariableUses(MatAlloca, LoweredAlloca);
    addToDeadInsts(MatAlloca);
  }

  // Now lower all other matrix instructions
  for (Instruction *MatInst : MatInsts)
    lowerInstruction(MatInst);

  deleteDeadInsts();
}

void HLMatrixLowerPass::deleteDeadInsts() {
  while (!m_deadInsts.empty()) {
    Instruction *Inst = m_deadInsts.back();
    m_deadInsts.pop_back();

    DXASSERT_NOMSG(Inst->use_empty());
    for (Value *Operand : Inst->operand_values()) {
      Instruction *OperandInst = dyn_cast<Instruction>(Operand);
      if (OperandInst && ++OperandInst->user_begin() == OperandInst->user_end()) {
        // We were its only user, erase recursively.
        // This will get rid of translation stubs:
        // Original: MatConsumer(MatProducer)
        // Producer lowered: MatConsumer(VecToMat(VecProducer)), MatProducer dead
        // Consumer lowered: VecConsumer(VecProducer)), MatConsumer(VecToMat) dead
        // Only by recursing on MatConsumer's operand do we delete the VecToMat stub.
        DXASSERT_NOMSG(*OperandInst->user_begin() == Inst);
        m_deadInsts.emplace_back(OperandInst);
      }
    }

    Inst->eraseFromParent();
  }
}

// Find all instructions consuming or producing matrices,
// directly or through pointers/arrays.
void HLMatrixLowerPass::getMatrixAllocasAndOtherInsts(Function &Func,
    std::vector<AllocaInst*> &MatAllocas, std::vector<Instruction*> &MatInsts){
  for (BasicBlock &BasicBlock : Func) {
    for (Instruction &Inst : BasicBlock) {
      // Don't lower GEPs directly, we'll handle them as we lower the root pointer,
      // typically a global variable or alloca.
      if (isa<GetElementPtrInst>(&Inst)) continue;

      // Don't lower lifetime intrinsics here, we'll handle them as we lower the alloca.
      IntrinsicInst *Intrin = dyn_cast<IntrinsicInst>(&Inst);
      if (Intrin && Intrin->getIntrinsicID() == Intrinsic::lifetime_start) continue;
      if (Intrin && Intrin->getIntrinsicID() == Intrinsic::lifetime_end) continue;

      if (AllocaInst *Alloca = dyn_cast<AllocaInst>(&Inst)) {
        if (HLMatrixType::isMatrixOrPtrOrArrayPtr(Alloca->getType())) {
          MatAllocas.emplace_back(Alloca);
        }
        continue;
      }
      
      if (CallInst *Call = dyn_cast<CallInst>(&Inst)) {
        // Lowering of global variables will have introduced
        // vec-to-mat translation stubs, which we deal with indirectly,
        // as we lower the instructions consuming them.
        if (m_vecToMatStubs->contains(Call->getCalledFunction()))
          continue;

        // Mat-to-vec stubs should only be introduced during instruction lowering.
        // Globals lowering won't introduce any because their only operand is
        // their initializer, which we can fully lower without stubbing since it is constant.
        DXASSERT(!m_matToVecStubs->contains(Call->getCalledFunction()),
          "Unexpected mat-to-vec stubbing before function instruction lowering.");

        // Match matrix producers
        if (HLMatrixType::isMatrixOrPtrOrArrayPtr(Inst.getType())) {
          MatInsts.emplace_back(Call);
          continue;
        }

        // Match matrix consumers
        for (Value *Operand : Inst.operand_values()) {
          if (HLMatrixType::isMatrixOrPtrOrArrayPtr(Operand->getType())) {
            MatInsts.emplace_back(Call);
            break;
          }
        }

        continue;
      }

      if (ReturnInst *Return = dyn_cast<ReturnInst>(&Inst)) {
        Value *ReturnValue = Return->getReturnValue();
        if (ReturnValue != nullptr && HLMatrixType::isMatrixOrPtrOrArrayPtr(ReturnValue->getType()))
          MatInsts.emplace_back(Return);
        continue;
      }

      // Nothing else should produce or consume matrices
    }
  }
}

// Gets the matrix-lowered representation of a value, potentially adding a translation stub.
// DiscardStub causes any vec-to-mat translation stubs to be deleted,
// it should be true only if the original instruction will be modified and kept alive.
// If a new instruction is created and the original marked as dead,
// then the remove dead instructions pass will take care of removing the stub.
Value* HLMatrixLowerPass::getLoweredByValOperand(Value *Val, IRBuilder<> &Builder, bool DiscardStub) {
  Type *Ty = Val->getType();

  // We're only lowering byval matrices.
  // Since structs and arrays are always accessed by pointer,
  // we do not need to worry about a matrix being hidden inside a more complex type.
  DXASSERT(!Ty->isPointerTy(), "Value cannot be a pointer.");
  HLMatrixType MatTy = HLMatrixType::dyn_cast(Ty);
  if (!MatTy) return Val;

  Type *LoweredTy = MatTy.getLoweredVectorTypeForReg();
  
  // Check if the value is already a vec-to-mat translation stub
  if (CallInst *Call = dyn_cast<CallInst>(Val)) {
    if (m_vecToMatStubs->contains(Call->getCalledFunction())) {
      if (DiscardStub && Call->getNumUses() == 1) {
        Call->use_begin()->set(UndefValue::get(Call->getType()));
        addToDeadInsts(Call);
      }

      Value *LoweredVal = Call->getArgOperand(0);
      DXASSERT(LoweredVal->getType() == LoweredTy, "Unexpected already-lowered value type.");
      return LoweredVal;
    }
  }
  // Lower mat 0 to vec 0.
  if (isa<ConstantAggregateZero>(Val))
    return ConstantAggregateZero::get(LoweredTy);

  // Return a mat-to-vec translation stub
  FunctionType *TranslationStubTy = FunctionType::get(LoweredTy, { Ty }, /* isVarArg */ false);
  Function *TranslationStub = m_matToVecStubs->get(TranslationStubTy);
  return Builder.CreateCall(TranslationStub, { Val });
}

// Attempts to retrieve the lowered vector pointer equivalent to a matrix pointer.
// Returns nullptr if the pointed-to matrix lives in memory that cannot be lowered at this time,
// for example a buffer or shader inputs/outputs, which are lowered during signature lowering.
Value *HLMatrixLowerPass::tryGetLoweredPtrOperand(Value *Ptr, IRBuilder<> &Builder, bool DiscardStub) {
  if (!HLMatrixType::isMatrixPtrOrArrayPtr(Ptr->getType()))
    return nullptr;

  // Matrix pointers can only be derived from Allocas, GlobalVariables or resource accesses.
  // The first two cases are what this pass must be able to lower, and we should already
  // have replaced their uses by vector to matrix pointer translation stubs.
  if (CallInst *Call = dyn_cast<CallInst>(Ptr)) {
    if (m_vecToMatStubs->contains(Call->getCalledFunction())) {
      if (DiscardStub && Call->getNumUses() == 1) {
        Call->use_begin()->set(UndefValue::get(Call->getType()));
        addToDeadInsts(Call);
      }
      return Call->getArgOperand(0);
    }
  }

  // There's one more case to handle.
  // When compiling shader libraries, signatures won't have been lowered yet.
  // So we can have a matrix in a struct as an argument,
  // or an alloca'd struct holding the return value of a call and containing a matrix.
  Value *RootPtr = Ptr;
  while (GEPOperator *GEP = dyn_cast<GEPOperator>(RootPtr))
    RootPtr = GEP->getPointerOperand();

  Argument *Arg = dyn_cast<Argument>(RootPtr);
  bool IsNonShaderArg = Arg != nullptr && !m_pHLModule->IsEntryThatUsesSignatures(Arg->getParent());
  if (IsNonShaderArg || isa<AllocaInst>(RootPtr)) {
    // Bitcast the matrix pointer to its lowered equivalent.
    // The HLMatrixBitcast pass will take care of this later.
    return Builder.CreateBitCast(Ptr, HLMatrixType::getLoweredType(Ptr->getType()));
  }

  // The pointer must be derived from a resource, we don't handle it in this pass.
  return nullptr;
}

// Bitcasts a value from matrix to vector or vice-versa.
// This is used to convert to/from arguments/return values since we don't
// lower signatures in this pass. The later HLMatrixBitcastLower pass fixes this.
Value *HLMatrixLowerPass::bitCastValue(Value *SrcVal, Type* DstTy, bool DstTyAlloca, IRBuilder<> &Builder) {
  Type *SrcTy = SrcVal->getType();
  DXASSERT_NOMSG(!SrcTy->isPointerTy());

  // We store and load from a temporary alloca, bitcasting either on the store pointer
  // or on the load pointer.
  IRBuilder<> AllocaBuilder(dxilutil::FindAllocaInsertionPt(Builder.GetInsertPoint()));
  Value *Alloca = AllocaBuilder.CreateAlloca(DstTyAlloca ? DstTy : SrcTy);
  Value *BitCastedAlloca = Builder.CreateBitCast(Alloca, (DstTyAlloca ? SrcTy : DstTy)->getPointerTo());
  Builder.CreateStore(SrcVal, DstTyAlloca ? BitCastedAlloca : Alloca);
  return Builder.CreateLoad(DstTyAlloca ? Alloca : BitCastedAlloca);
}

// Replaces all uses of a matrix value by its lowered vector form,
// inserting translation stubs for users which still expect a matrix value.
void HLMatrixLowerPass::replaceAllUsesByLoweredValue(Instruction* MatInst, Value* VecVal) {
  if (VecVal == nullptr || VecVal == MatInst) return;

  DXASSERT(HLMatrixType::getLoweredType(MatInst->getType()) == VecVal->getType(),
    "Unexpected lowered value type.");

  Instruction *VecToMatStub = nullptr;

  while (!MatInst->use_empty()) {
    Use &ValUse = *MatInst->use_begin();

    // Handle non-matrix cases, just point to the new value.
    if (MatInst->getType() == VecVal->getType()) {
      ValUse.set(VecVal);
      continue;
    }

    // If the user is already a matrix-to-vector translation stub,
    // we can now replace it by the proper vector value.
    if (CallInst *Call = dyn_cast<CallInst>(ValUse.getUser())) {
      if (m_matToVecStubs->contains(Call->getCalledFunction())) {
        Call->replaceAllUsesWith(VecVal);
        ValUse.set(UndefValue::get(MatInst->getType()));
        addToDeadInsts(Call);
        continue;
      }
    }

    // Otherwise, the user should point to a vector-to-matrix translation
    // stub of the new vector value.
    if (VecToMatStub == nullptr) {
      FunctionType *TranslationStubTy = FunctionType::get(
        MatInst->getType(), { VecVal->getType() }, /* isVarArg */ false);
      Function *TranslationStub = m_vecToMatStubs->get(TranslationStubTy);

      Instruction *PrevInst = dyn_cast<Instruction>(VecVal);
      if (PrevInst == nullptr) PrevInst = MatInst;

      IRBuilder<> Builder(PrevInst->getNextNode());
      VecToMatStub = Builder.CreateCall(TranslationStub, { VecVal });
    }

    ValUse.set(VecToMatStub);
  }
}

// Replaces all uses of a matrix or matrix array alloca or global variable by its lowered equivalent.
// This doesn't lower the users, but will insert a translation stub from the lowered value pointer
// back to the matrix value pointer, and recreate any GEPs around the new pointer.
// Before: User(GEP(MatrixArrayAlloca))
// After: User(VecToMatPtrStub(GEP'(VectorArrayAlloca)))
void HLMatrixLowerPass::replaceAllVariableUses(Value* MatPtr, Value* LoweredPtr) {
  DXASSERT_NOMSG(HLMatrixType::isMatrixPtrOrArrayPtr(MatPtr->getType()));
  DXASSERT_NOMSG(LoweredPtr->getType() == HLMatrixType::getLoweredType(MatPtr->getType()));

  SmallVector<Value*, 4> GEPIdxStack;
  GEPIdxStack.emplace_back(ConstantInt::get(Type::getInt32Ty(MatPtr->getContext()), 0));
  replaceAllVariableUses(GEPIdxStack, MatPtr, LoweredPtr);
}

void HLMatrixLowerPass::replaceAllVariableUses(
    SmallVectorImpl<Value*> &GEPIdxStack, Value *StackTopPtr, Value* LoweredPtr) {
  while (!StackTopPtr->use_empty()) {
    llvm::Use &Use = *StackTopPtr->use_begin();
    if (GEPOperator *GEP = dyn_cast<GEPOperator>(Use.getUser())) {
      DXASSERT(GEP->getNumIndices() >= 1, "Unexpected degenerate GEP.");
      DXASSERT(cast<ConstantInt>(*GEP->idx_begin())->isZero(), "Unexpected non-zero first GEP index.");

      // Recurse in GEP to find actual users
      for (auto It = GEP->idx_begin() + 1; It != GEP->idx_end(); ++It)
        GEPIdxStack.emplace_back(*It);
      replaceAllVariableUses(GEPIdxStack, GEP, LoweredPtr);
      GEPIdxStack.erase(GEPIdxStack.end() - (GEP->getNumIndices() - 1), GEPIdxStack.end());
      
      // Discard the GEP
      DXASSERT_NOMSG(GEP->use_empty());
      if (GetElementPtrInst *GEPInst = dyn_cast<GetElementPtrInst>(GEP)) {
        Use.set(UndefValue::get(Use->getType()));
        addToDeadInsts(GEPInst);
      } else {
        // constant GEP
        cast<Constant>(GEP)->destroyConstant();
      }
      continue;
    }

    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Use.getUser())) {
      DXASSERT(CE->getOpcode() == Instruction::AddrSpaceCast ||
        CE->use_empty(), "Unexpected constant user");
      replaceAllVariableUses(GEPIdxStack, CE, LoweredPtr);
      DXASSERT_NOMSG(CE->use_empty());
      CE->destroyConstant();
      continue;
    }

    if (AddrSpaceCastInst *CI = dyn_cast<AddrSpaceCastInst>(Use.getUser())) {
      replaceAllVariableUses(GEPIdxStack, CI, LoweredPtr);
      Use.set(UndefValue::get(Use->getType()));
      addToDeadInsts(CI);
      continue;
    }

    if (BitCastInst *BCI = dyn_cast<BitCastInst>(Use.getUser())) {
      // Replace bitcasts to i8* for lifetime intrinsics.
      if (BCI->getType()->isPointerTy() && BCI->getType()->getPointerElementType()->isIntegerTy(8))
      {
        DXASSERT(onlyUsedByLifetimeMarkers(BCI),
                 "bitcast to i8* must only be used by lifetime intrinsics");
        Value *NewBCI = IRBuilder<>(BCI).CreateBitCast(LoweredPtr, BCI->getType());
        // Replace all uses of the use.
        BCI->replaceAllUsesWith(NewBCI);
        // Remove the current use to end iteration.
        Use.set(UndefValue::get(Use->getType()));
        addToDeadInsts(BCI);
        continue;
      }
    }

    // Recreate the same GEP sequence, if any, on the lowered pointer
    IRBuilder<> Builder(cast<Instruction>(Use.getUser()));
    Value *LoweredStackTopPtr = GEPIdxStack.size() == 1
      ? LoweredPtr : Builder.CreateGEP(LoweredPtr, GEPIdxStack);

    // Generate a stub translating the vector pointer back to a matrix pointer,
    // such that consuming instructions are unaffected.
    FunctionType *TranslationStubTy = FunctionType::get(
      StackTopPtr->getType(), { LoweredStackTopPtr->getType() }, /* isVarArg */ false);
    Function *TranslationStub = m_vecToMatStubs->get(TranslationStubTy);
    Use.set(Builder.CreateCall(TranslationStub, { LoweredStackTopPtr }));
  }
}

void HLMatrixLowerPass::lowerGlobal(GlobalVariable *Global) {
  if (Global->user_empty()) return;

  PointerType *LoweredPtrTy = cast<PointerType>(HLMatrixType::getLoweredType(Global->getType()));
  DXASSERT_NOMSG(LoweredPtrTy != Global->getType());

  Constant *LoweredInitVal = Global->hasInitializer()
    ? lowerConstInitVal(Global->getInitializer()) : nullptr;
  GlobalVariable *LoweredGlobal = new GlobalVariable(*m_pModule, LoweredPtrTy->getElementType(),
    Global->isConstant(), Global->getLinkage(), LoweredInitVal,
    Global->getName() + ".v", /*InsertBefore*/ nullptr, Global->getThreadLocalMode(),
    Global->getType()->getAddressSpace());

  // Add debug info.
  if (m_HasDbgInfo) {
    DebugInfoFinder &Finder = m_pHLModule->GetOrCreateDebugInfoFinder();
    HLModule::UpdateGlobalVariableDebugInfo(Global, Finder, LoweredGlobal);
  }

  replaceAllVariableUses(Global, LoweredGlobal);
  Global->removeDeadConstantUsers();
  Global->eraseFromParent();
}

Constant *HLMatrixLowerPass::lowerConstInitVal(Constant *Val) {
  Type *Ty = Val->getType();

  // If it's an array of matrices, recurse for each element or nested array
  if (ArrayType *ArrayTy = dyn_cast<ArrayType>(Ty)) {
    SmallVector<Constant*, 4> LoweredElems;
    unsigned NumElems = ArrayTy->getNumElements();
    LoweredElems.reserve(NumElems);
    for (unsigned ElemIdx = 0; ElemIdx < NumElems; ++ElemIdx) {
      Constant *ArrayElem = Val->getAggregateElement(ElemIdx);
      LoweredElems.emplace_back(lowerConstInitVal(ArrayElem));
    }

    Type *LoweredElemTy = HLMatrixType::getLoweredType(ArrayTy->getElementType(), /*MemRepr*/true);
    ArrayType *LoweredArrayTy = ArrayType::get(LoweredElemTy, NumElems);
    return ConstantArray::get(LoweredArrayTy, LoweredElems);
  }

  // Otherwise it's a matrix, lower it to a vector
  HLMatrixType MatTy = HLMatrixType::cast(Ty);
  DXASSERT_NOMSG(isa<StructType>(Ty));
  Constant *RowArrayVal = Val->getAggregateElement((unsigned)0);

  // Original initializer should have been produced in row/column-major order
  // depending on the qualifiers of the target variable, so preserve the order.
  SmallVector<Constant*, 16> MatElems;
  for (unsigned RowIdx = 0; RowIdx < MatTy.getNumRows(); ++RowIdx) {
    Constant *RowVal = RowArrayVal->getAggregateElement(RowIdx);
    for (unsigned ColIdx = 0; ColIdx < MatTy.getNumColumns(); ++ColIdx) {
      MatElems.emplace_back(RowVal->getAggregateElement(ColIdx));
    }
  }

  Constant *Vec = ConstantVector::get(MatElems);
  
  // Matrix elements are always in register representation,
  // but the lowered global variable is of vector type in
  // its memory representation, so we must convert here.

  // This will produce a constant so we can use an IRBuilder without a valid insertion point.
  IRBuilder<> DummyBuilder(Val->getContext());
  return cast<Constant>(MatTy.emitLoweredRegToMem(Vec, DummyBuilder));
}

AllocaInst *HLMatrixLowerPass::lowerAlloca(AllocaInst *MatAlloca) {
  PointerType *LoweredAllocaTy = cast<PointerType>(HLMatrixType::getLoweredType(MatAlloca->getType()));

  IRBuilder<> Builder(MatAlloca);
  AllocaInst *LoweredAlloca = Builder.CreateAlloca(
    LoweredAllocaTy->getElementType(), nullptr, MatAlloca->getName());

  // Update debug info.
  if (DbgDeclareInst *DbgDeclare = llvm::FindAllocaDbgDeclare(MatAlloca)) {
    DILocalVariable *DbgDeclareVar  = DbgDeclare->getVariable();
    DIExpression    *DbgDeclareExpr = DbgDeclare->getExpression();
    DIBuilder DIB(*MatAlloca->getModule());
    DIB.insertDeclare(LoweredAlloca, DbgDeclareVar, DbgDeclareExpr, DbgDeclare->getDebugLoc(), DbgDeclare);
  }

  if (HLModule::HasPreciseAttributeWithMetadata(MatAlloca))
    HLModule::MarkPreciseAttributeWithMetadata(LoweredAlloca);

  replaceAllVariableUses(MatAlloca, LoweredAlloca);

  return LoweredAlloca;
}

void HLMatrixLowerPass::lowerInstruction(Instruction* Inst) {
  if (CallInst *Call = dyn_cast<CallInst>(Inst)) {
    Value *LoweredValue = lowerCall(Call);

    // lowerCall returns the lowered value iff we should discard
    // the original matrix instruction and replace all of its uses
    // by the lowered value. It returns nullptr to opt-out of this.
    if (LoweredValue != nullptr) {
      replaceAllUsesByLoweredValue(Call, LoweredValue);
      addToDeadInsts(Inst);
    }
  }
  else if (ReturnInst *Return = dyn_cast<ReturnInst>(Inst)) {
    lowerReturn(Return);
  }
  else
    llvm_unreachable("Unexpected matrix instruction type.");
}

void HLMatrixLowerPass::lowerReturn(ReturnInst* Return) {
  Value *RetVal = Return->getReturnValue();
  Type *RetTy = RetVal->getType();
  DXASSERT_LOCALVAR(RetTy, !RetTy->isPointerTy(), "Unexpected matrix returned by pointer.");

  IRBuilder<> Builder(Return);
  Value *LoweredRetVal = getLoweredByValOperand(RetVal, Builder, /* DiscardStub */ true);

  // Since we're not lowering the signature, we can't return the lowered value directly,
  // so insert a bitcast, which HLMatrixBitcastLower knows how to eliminate.
  Value *BitCastedRetVal = bitCastValue(LoweredRetVal, RetVal->getType(), /* DstTyAlloca */ false, Builder);
  Return->setOperand(0, BitCastedRetVal);
}

Value *HLMatrixLowerPass::lowerCall(CallInst *Call) {
  HLOpcodeGroup OpcodeGroup = GetHLOpcodeGroupByName(Call->getCalledFunction());
  return OpcodeGroup == HLOpcodeGroup::NotHL
    ? lowerNonHLCall(Call) : lowerHLOperation(Call, OpcodeGroup);
}

// Special function to lower precise call applied to a matrix
// The matrix should be lowered and the call regenerated with vector arg
void HLMatrixLowerPass::lowerPreciseCall(CallInst *Call, IRBuilder<> Builder) {
  DXASSERT(Call->getNumArgOperands() == 1, "Only one arg expected for precise matrix call");
  Value *Arg = Call->getArgOperand(0);
  Value *LoweredArg = getLoweredByValOperand(Arg, Builder);
  HLModule::MarkPreciseAttributeOnValWithFunctionCall(LoweredArg, Builder, *m_pModule);
  addToDeadInsts(Call);
}

Value *HLMatrixLowerPass::lowerNonHLCall(CallInst *Call) {
  // First, handle any operand of matrix-derived type
  // We don't lower the callee's signature in this pass,
  // so, for any matrix-typed parameter, we create a bitcast from the
  // lowered vector back to the matrix type, which the later HLMatrixBitcastLower
  // pass knows how to eliminate.
  IRBuilder<> PreCallBuilder(Call);
  unsigned NumArgs = Call->getNumArgOperands();
  Function *Func = Call->getCalledFunction();
  if (Func && HLModule::HasPreciseAttribute(Func)) {
    lowerPreciseCall(Call, PreCallBuilder);
    return nullptr;
  }

  for (unsigned ArgIdx = 0; ArgIdx < NumArgs; ++ArgIdx) {
    Use &ArgUse = Call->getArgOperandUse(ArgIdx);
    if (ArgUse->getType()->isPointerTy()) {
      // Byref arg
      Value *LoweredArg = tryGetLoweredPtrOperand(ArgUse.get(), PreCallBuilder, /* DiscardStub */ true);
      if (LoweredArg != nullptr) {
        // Pointer to a matrix we've lowered, insert a bitcast back to matrix pointer type.
        Value *BitCastedArg = PreCallBuilder.CreateBitCast(LoweredArg, ArgUse->getType());
        ArgUse.set(BitCastedArg);
      }
    }
    else {
      // Byvalue arg
      Value *LoweredArg = getLoweredByValOperand(ArgUse.get(), PreCallBuilder, /* DiscardStub */ true);
      if (LoweredArg == ArgUse.get()) continue;

      Value *BitCastedArg = bitCastValue(LoweredArg, ArgUse->getType(), /* DstTyAlloca */ false, PreCallBuilder);
      ArgUse.set(BitCastedArg);
    }
  }

  // Now check the return type
  HLMatrixType RetMatTy = HLMatrixType::dyn_cast(Call->getType());
  if (!RetMatTy) {
    DXASSERT(!HLMatrixType::isMatrixPtrOrArrayPtr(Call->getType()),
      "Unexpected user call returning a matrix by pointer.");
    // Nothing to replace, other instructions can consume a non-matrix return type.
    return nullptr;
  }

  // The callee returns a matrix, and we don't lower signatures in this pass.
  // We perform a sketchy bitcast to the lowered register-representation type,
  // which the later HLMatrixBitcastLower pass knows how to eliminate.
  IRBuilder<> AllocaBuilder(dxilutil::FindAllocaInsertionPt(Call));
  Value *LoweredAlloca = AllocaBuilder.CreateAlloca(RetMatTy.getLoweredVectorTypeForReg());
  
  IRBuilder<> PostCallBuilder(Call->getNextNode());
  Value *BitCastedAlloca = PostCallBuilder.CreateBitCast(LoweredAlloca, Call->getType()->getPointerTo());
  
  // This is slightly tricky
  // We want to replace all uses of the matrix-returning call by the bitcasted value,
  // but the store to the bitcasted pointer itself is a use of that matrix,
  // so we need to create the load, replace the uses, and then insert the store.
  LoadInst *LoweredVal = PostCallBuilder.CreateLoad(LoweredAlloca);
  replaceAllUsesByLoweredValue(Call, LoweredVal);

  // Now we can insert the store. Make sure to do so before the load.
  PostCallBuilder.SetInsertPoint(LoweredVal);
  PostCallBuilder.CreateStore(Call, BitCastedAlloca);
  
  // Return nullptr since we did our own uses replacement and we don't want
  // the matrix instruction to be marked as dead since we're still using it.
  return nullptr;
}

Value *HLMatrixLowerPass::lowerHLOperation(CallInst *Call, HLOpcodeGroup OpcodeGroup) {
  IRBuilder<> Builder(Call);
  switch (OpcodeGroup) {
  case HLOpcodeGroup::HLIntrinsic:
    return lowerHLIntrinsic(Call, static_cast<IntrinsicOp>(GetHLOpcode(Call)));

  case HLOpcodeGroup::HLBinOp:
    return lowerHLBinaryOperation(
      Call->getArgOperand(HLOperandIndex::kBinaryOpSrc0Idx),
      Call->getArgOperand(HLOperandIndex::kBinaryOpSrc1Idx),
      static_cast<HLBinaryOpcode>(GetHLOpcode(Call)), Builder);

  case HLOpcodeGroup::HLUnOp:
    return lowerHLUnaryOperation(
      Call->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx),
      static_cast<HLUnaryOpcode>(GetHLOpcode(Call)), Builder);

  case HLOpcodeGroup::HLMatLoadStore:
    return lowerHLLoadStore(Call, static_cast<HLMatLoadStoreOpcode>(GetHLOpcode(Call)));

  case HLOpcodeGroup::HLCast:
    return lowerHLCast(Call,
      Call->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx), Call->getType(),
      static_cast<HLCastOpcode>(GetHLOpcode(Call)), Builder);

  case HLOpcodeGroup::HLSubscript:
    return lowerHLSubscript(Call, static_cast<HLSubscriptOpcode>(GetHLOpcode(Call)));

  case HLOpcodeGroup::HLInit:
    return lowerHLInit(Call);

  case HLOpcodeGroup::HLSelect:
    return lowerHLSelect(Call);

  default:
    llvm_unreachable("Unexpected matrix opcode");
  }
}

Value *HLMatrixLowerPass::lowerHLIntrinsic(CallInst *Call, IntrinsicOp Opcode) {
  IRBuilder<> Builder(Call);

  // See if this is a matrix-specific intrinsic which we should expand here
  switch (Opcode) {
  case IntrinsicOp::IOP_umul:
  case IntrinsicOp::IOP_mul:
    return lowerHLMulIntrinsic(
      Call->getArgOperand(HLOperandIndex::kBinaryOpSrc0Idx),
      Call->getArgOperand(HLOperandIndex::kBinaryOpSrc1Idx),
      /* Unsigned */ Opcode == IntrinsicOp::IOP_umul, Builder);
  case IntrinsicOp::IOP_transpose:
    return lowerHLTransposeIntrinsic(Call->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx), Builder);
  case IntrinsicOp::IOP_determinant:
    return lowerHLDeterminantIntrinsic(Call->getArgOperand(HLOperandIndex::kUnaryOpSrc0Idx), Builder);
  }

  // Delegate to a lowered intrinsic call
  SmallVector<Value*, 4> LoweredArgs;
  LoweredArgs.reserve(Call->getNumArgOperands());
  for (Value *Arg : Call->arg_operands()) {
    if (Arg->getType()->isPointerTy()) {
      // ByRef parameter (for example, frexp's second parameter)
      // If the argument points to a lowered matrix variable, replace it here,
      // otherwise preserve the matrix type and let further passes handle the lowering.
      Value *LoweredArg = tryGetLoweredPtrOperand(Arg, Builder);
      if (LoweredArg == nullptr) LoweredArg = Arg;
      LoweredArgs.emplace_back(LoweredArg);
    }
    else {
      LoweredArgs.emplace_back(getLoweredByValOperand(Arg, Builder));
    }
  }

  Type *LoweredRetTy = HLMatrixType::getLoweredType(Call->getType());
  return callHLFunction(*m_pModule, HLOpcodeGroup::HLIntrinsic, static_cast<unsigned>(Opcode),
                        LoweredRetTy, LoweredArgs,
                        Call->getCalledFunction()->getAttributes().getFnAttributes(), Builder);
}

// Handles multiplcation of a scalar with a matrix
Value *HLMatrixLowerPass::translateScalarMatMul(Value *Lhs, Value *Rhs, IRBuilder<> &Builder, bool isLhsScalar) {
  Value *Mat = isLhsScalar ? Rhs : Lhs;
  Value *Scalar = isLhsScalar ? Lhs : Rhs;
  Value* LoweredMat = getLoweredByValOperand(Mat, Builder);
  Type *ScalarTy = Scalar->getType();
  FixedVectorType *VT = dyn_cast<FixedVectorType>(LoweredMat->getType());
  // Perform the scalar-matrix multiplication!
  Type *ElemTy = VT->getElementType();
  bool isIntMulOp = ScalarTy->isIntegerTy() && ElemTy->isIntegerTy();
  bool isFloatMulOp = ScalarTy->isFloatingPointTy() && ElemTy->isFloatingPointTy();
  DXASSERT(ScalarTy == ElemTy, "Scalar type must match the matrix component type.");
  Value *Result = Builder.CreateVectorSplat(VT->getNumElements(), Scalar);

  if (isFloatMulOp) {
    // Preserve the order of operation for floats
    Result = isLhsScalar ? Builder.CreateFMul(Result, LoweredMat) : Builder.CreateFMul(LoweredMat, Result);
  }
  else if (isIntMulOp) {
    // Doesn't matter for integers but still preserve the order of operation
    Result = isLhsScalar ? Builder.CreateMul(Result, LoweredMat) : Builder.CreateMul(LoweredMat, Result);
  }
  else {
    DXASSERT(0, "Unknown type encountered when doing scalar-matrix multiplication.");
  }

  return Result;
}

Value *HLMatrixLowerPass::lowerHLMulIntrinsic(Value* Lhs, Value *Rhs,
    bool Unsigned, IRBuilder<> &Builder) {
  HLMatrixType LhsMatTy = HLMatrixType::dyn_cast(Lhs->getType());
  HLMatrixType RhsMatTy = HLMatrixType::dyn_cast(Rhs->getType());
  Value* LoweredLhs = getLoweredByValOperand(Lhs, Builder);
  Value* LoweredRhs = getLoweredByValOperand(Rhs, Builder);

  // Translate multiplication of scalar with matrix
  bool isLhsScalar = !LoweredLhs->getType()->isVectorTy();
  bool isRhsScalar = !LoweredRhs->getType()->isVectorTy();
  bool isScalar = isLhsScalar || isRhsScalar;
  if (isScalar)
    return translateScalarMatMul(Lhs, Rhs, Builder, isLhsScalar);

  DXASSERT(LoweredLhs->getType()->getScalarType() == LoweredRhs->getType()->getScalarType(),
    "Unexpected element type mismatch in mul intrinsic.");
  DXASSERT(cast<VectorType>(LoweredLhs->getType()) && cast<VectorType>(LoweredRhs->getType()),
    "Unexpected scalar in lowered matrix mul intrinsic operands.");

  Type* ElemTy = LoweredLhs->getType()->getScalarType();

  // Figure out the dimensions of each side
  unsigned LhsNumRows, LhsNumCols, RhsNumRows, RhsNumCols;
  if (LhsMatTy && RhsMatTy) {
    LhsNumRows = LhsMatTy.getNumRows();
    LhsNumCols = LhsMatTy.getNumColumns();
    RhsNumRows = RhsMatTy.getNumRows();
    RhsNumCols = RhsMatTy.getNumColumns();
  }
  else if (LhsMatTy) {
    LhsNumRows = LhsMatTy.getNumRows();
    LhsNumCols = LhsMatTy.getNumColumns();
    FixedVectorType *VT = dyn_cast<FixedVectorType>(LoweredRhs->getType());
    RhsNumRows = VT->getNumElements();
    RhsNumCols = 1;
  }
  else if (RhsMatTy) {
    LhsNumRows = 1;
    FixedVectorType *VT = dyn_cast<FixedVectorType>(LoweredLhs->getType());
    LhsNumCols = VT->getNumElements();
    RhsNumRows = RhsMatTy.getNumRows();
    RhsNumCols = RhsMatTy.getNumColumns();
  }
  else {
    llvm_unreachable("mul intrinsic was identified as a matrix operation but neither operand is a matrix.");
  }

  DXASSERT(LhsNumCols == RhsNumRows, "Matrix mul intrinsic operands dimensions mismatch.");
  HLMatrixType ResultMatTy(ElemTy, LhsNumRows, RhsNumCols);
  unsigned AccCount = LhsNumCols;

  // Get the multiply-and-add intrinsic function, we'll need it
  IntrinsicOp MadOpcode = Unsigned ? IntrinsicOp::IOP_umad : IntrinsicOp::IOP_mad;
  FunctionType *MadFuncTy = FunctionType::get(ElemTy, { Builder.getInt32Ty(), ElemTy, ElemTy, ElemTy }, false);
  Function *MadFunc = GetOrCreateHLFunction(*m_pModule, MadFuncTy, HLOpcodeGroup::HLIntrinsic, (unsigned)MadOpcode);
  Constant *MadOpcodeVal = Builder.getInt32((unsigned)MadOpcode);

  // Perform the multiplication!
  Value *Result = UndefValue::get(VectorType::get(ElemTy, LhsNumRows * RhsNumCols));
  for (unsigned ResultRowIdx = 0; ResultRowIdx < ResultMatTy.getNumRows(); ++ResultRowIdx) {
    for (unsigned ResultColIdx = 0; ResultColIdx < ResultMatTy.getNumColumns(); ++ResultColIdx) {
      unsigned ResultElemIdx = ResultMatTy.getRowMajorIndex(ResultRowIdx, ResultColIdx);
      Value *ResultElem = nullptr;

      for (unsigned AccIdx = 0; AccIdx < AccCount; ++AccIdx) {
        unsigned LhsElemIdx = HLMatrixType::getRowMajorIndex(ResultRowIdx, AccIdx, LhsNumRows, LhsNumCols);
        unsigned RhsElemIdx = HLMatrixType::getRowMajorIndex(AccIdx, ResultColIdx, RhsNumRows, RhsNumCols);
        Value* LhsElem = Builder.CreateExtractElement(LoweredLhs, static_cast<uint64_t>(LhsElemIdx));
        Value* RhsElem = Builder.CreateExtractElement(LoweredRhs, static_cast<uint64_t>(RhsElemIdx));
        if (ResultElem == nullptr) {
          ResultElem = ElemTy->isFloatingPointTy()
            ? Builder.CreateFMul(LhsElem, RhsElem)
            : Builder.CreateMul(LhsElem, RhsElem);
        }
        else {
          ResultElem = Builder.CreateCall(MadFunc, { MadOpcodeVal, LhsElem, RhsElem, ResultElem });
        }
      }

      Result = Builder.CreateInsertElement(Result, ResultElem, static_cast<uint64_t>(ResultElemIdx));
    }
  }

  return Result;
}

Value *HLMatrixLowerPass::lowerHLTransposeIntrinsic(Value* MatVal, IRBuilder<> &Builder) {
  HLMatrixType MatTy = HLMatrixType::cast(MatVal->getType());
  Value *LoweredVal = getLoweredByValOperand(MatVal, Builder);
  return MatTy.emitLoweredVectorRowToCol(LoweredVal, Builder);
}

static Value *determinant2x2(Value *M00, Value *M01, Value *M10, Value *M11, IRBuilder<> &Builder) {
  Value *Mul0 = Builder.CreateFMul(M00, M11);
  Value *Mul1 = Builder.CreateFMul(M01, M10);
  return Builder.CreateFSub(Mul0, Mul1);
}

static Value *determinant3x3(Value *M00, Value *M01, Value *M02,
    Value *M10, Value *M11, Value *M12,
    Value *M20, Value *M21, Value *M22,
    IRBuilder<> &Builder) {
  Value *Det00 = determinant2x2(M11, M12, M21, M22, Builder);
  Value *Det01 = determinant2x2(M10, M12, M20, M22, Builder);
  Value *Det02 = determinant2x2(M10, M11, M20, M21, Builder);
  Det00 = Builder.CreateFMul(M00, Det00);
  Det01 = Builder.CreateFMul(M01, Det01);
  Det02 = Builder.CreateFMul(M02, Det02);
  Value *Result = Builder.CreateFSub(Det00, Det01);
  Result = Builder.CreateFAdd(Result, Det02);
  return Result;
}

static Value *determinant4x4(Value *M00, Value *M01, Value *M02, Value *M03,
    Value *M10, Value *M11, Value *M12, Value *M13,
    Value *M20, Value *M21, Value *M22, Value *M23,
    Value *M30, Value *M31, Value *M32, Value *M33,
    IRBuilder<> &Builder) {
  Value *Det00 = determinant3x3(M11, M12, M13, M21, M22, M23, M31, M32, M33, Builder);
  Value *Det01 = determinant3x3(M10, M12, M13, M20, M22, M23, M30, M32, M33, Builder);
  Value *Det02 = determinant3x3(M10, M11, M13, M20, M21, M23, M30, M31, M33, Builder);
  Value *Det03 = determinant3x3(M10, M11, M12, M20, M21, M22, M30, M31, M32, Builder);
  Det00 = Builder.CreateFMul(M00, Det00);
  Det01 = Builder.CreateFMul(M01, Det01);
  Det02 = Builder.CreateFMul(M02, Det02);
  Det03 = Builder.CreateFMul(M03, Det03);
  Value *Result = Builder.CreateFSub(Det00, Det01);
  Result = Builder.CreateFAdd(Result, Det02);
  Result = Builder.CreateFSub(Result, Det03);
  return Result;
}

Value *HLMatrixLowerPass::lowerHLDeterminantIntrinsic(Value* MatVal, IRBuilder<> &Builder) {
  HLMatrixType MatTy = HLMatrixType::cast(MatVal->getType());
  DXASSERT_NOMSG(MatTy.getNumColumns() == MatTy.getNumRows());

  Value *LoweredVal = getLoweredByValOperand(MatVal, Builder);

  // Extract all matrix elements
  SmallVector<Value*, 16> Elems;
  for (unsigned ElemIdx = 0; ElemIdx < MatTy.getNumElements(); ++ElemIdx)
    Elems.emplace_back(Builder.CreateExtractElement(LoweredVal, static_cast<uint64_t>(ElemIdx)));

  // Delegate to appropriate determinant function
  switch (MatTy.getNumColumns()) {
  case 1:
    return Elems[0];

  case 2:
    return determinant2x2(
      Elems[0], Elems[1],
      Elems[2], Elems[3],
      Builder);

  case 3:
    return determinant3x3(
      Elems[0], Elems[1], Elems[2],
      Elems[3], Elems[4], Elems[5],
      Elems[6], Elems[7], Elems[8],
      Builder);

  case 4:
    return determinant4x4(
      Elems[0], Elems[1], Elems[2], Elems[3],
      Elems[4], Elems[5], Elems[6], Elems[7],
      Elems[8], Elems[9], Elems[10], Elems[11],
      Elems[12], Elems[13], Elems[14], Elems[15],
      Builder);

  default:
    llvm_unreachable("Unexpected matrix dimensions.");
  }
}

Value *HLMatrixLowerPass::lowerHLUnaryOperation(Value *MatVal, HLUnaryOpcode Opcode, IRBuilder<> &Builder) {
  Value *LoweredVal = getLoweredByValOperand(MatVal, Builder);
  VectorType *VecTy = cast<VectorType>(LoweredVal->getType());
  bool IsFloat = VecTy->getElementType()->isFloatingPointTy();
  
  switch (Opcode) {
  case HLUnaryOpcode::Plus: return LoweredVal; // No-op

  case HLUnaryOpcode::Minus:
    return IsFloat
      ? Builder.CreateFSub(Constant::getNullValue(VecTy), LoweredVal)
      : Builder.CreateSub(Constant::getNullValue(VecTy), LoweredVal);

  case HLUnaryOpcode::LNot:
    return IsFloat
      ? Builder.CreateFCmp(CmpInst::FCMP_UEQ, LoweredVal, Constant::getNullValue(VecTy))
      : Builder.CreateICmp(CmpInst::ICMP_EQ, LoweredVal, Constant::getNullValue(VecTy));

  case HLUnaryOpcode::Not:
    return Builder.CreateXor(LoweredVal, Constant::getAllOnesValue(VecTy));

  case HLUnaryOpcode::PostInc:
  case HLUnaryOpcode::PreInc:
  case HLUnaryOpcode::PostDec:
  case HLUnaryOpcode::PreDec: {
    Constant *ScalarOne = IsFloat
      ? ConstantFP::get(VecTy->getElementType(), 1)
      : ConstantInt::get(VecTy->getElementType(), 1);
    Constant *VecOne = ConstantVector::getSplat(VecTy->getNumElements(), ScalarOne);

    // CodeGen already emitted the load and following store, our job is only to produce
    // the updated value.
    if (Opcode == HLUnaryOpcode::PostInc || Opcode == HLUnaryOpcode::PreInc) {
      return IsFloat
        ? Builder.CreateFAdd(LoweredVal, VecOne)
        : Builder.CreateAdd(LoweredVal, VecOne);
    }
    else {
      return IsFloat
        ? Builder.CreateFSub(LoweredVal, VecOne)
        : Builder.CreateSub(LoweredVal, VecOne);
    }
  }
  default:
    llvm_unreachable("Unsupported unary matrix operator");
  }
}

Value *HLMatrixLowerPass::lowerHLBinaryOperation(Value *Lhs, Value *Rhs, HLBinaryOpcode Opcode, IRBuilder<> &Builder) {
  Value *LoweredLhs = getLoweredByValOperand(Lhs, Builder);
  Value *LoweredRhs = getLoweredByValOperand(Rhs, Builder);

  DXASSERT(LoweredLhs->getType()->isVectorTy() && LoweredRhs->getType()->isVectorTy(),
    "Expected lowered binary operation operands to be vectors");
  DXASSERT(LoweredLhs->getType() == LoweredRhs->getType(),
    "Expected lowered binary operation operands to have matching types.");
  FixedVectorType *VT = dyn_cast<FixedVectorType>(LoweredLhs->getType());
  bool IsFloat = VT->getElementType()->isFloatingPointTy();

  switch (Opcode) {
  case HLBinaryOpcode::Add:
    return IsFloat
      ? Builder.CreateFAdd(LoweredLhs, LoweredRhs)
      : Builder.CreateAdd(LoweredLhs, LoweredRhs);

  case HLBinaryOpcode::Sub:
    return IsFloat
      ? Builder.CreateFSub(LoweredLhs, LoweredRhs)
      : Builder.CreateSub(LoweredLhs, LoweredRhs);

  case HLBinaryOpcode::Mul:
    return IsFloat
      ? Builder.CreateFMul(LoweredLhs, LoweredRhs)
      : Builder.CreateMul(LoweredLhs, LoweredRhs);

  case HLBinaryOpcode::Div:
    return IsFloat
      ? Builder.CreateFDiv(LoweredLhs, LoweredRhs)
      : Builder.CreateSDiv(LoweredLhs, LoweredRhs);

  case HLBinaryOpcode::Rem:
    return IsFloat
      ? Builder.CreateFRem(LoweredLhs, LoweredRhs)
      : Builder.CreateSRem(LoweredLhs, LoweredRhs);

  case HLBinaryOpcode::And:
    return Builder.CreateAnd(LoweredLhs, LoweredRhs);

  case HLBinaryOpcode::Or:
    return Builder.CreateOr(LoweredLhs, LoweredRhs);

  case HLBinaryOpcode::Xor:
    return Builder.CreateXor(LoweredLhs, LoweredRhs);

  case HLBinaryOpcode::Shl:
    return Builder.CreateShl(LoweredLhs, LoweredRhs);

  case HLBinaryOpcode::Shr:
    return Builder.CreateAShr(LoweredLhs, LoweredRhs);

  case HLBinaryOpcode::LT:
    return IsFloat
      ? Builder.CreateFCmp(CmpInst::FCMP_OLT, LoweredLhs, LoweredRhs)
      : Builder.CreateICmp(CmpInst::ICMP_SLT, LoweredLhs, LoweredRhs);

  case HLBinaryOpcode::GT:
    return IsFloat
      ? Builder.CreateFCmp(CmpInst::FCMP_OGT, LoweredLhs, LoweredRhs)
      : Builder.CreateICmp(CmpInst::ICMP_SGT, LoweredLhs, LoweredRhs);

  case HLBinaryOpcode::LE:
    return IsFloat
      ? Builder.CreateFCmp(CmpInst::FCMP_OLE, LoweredLhs, LoweredRhs)
      : Builder.CreateICmp(CmpInst::ICMP_SLE, LoweredLhs, LoweredRhs);

  case HLBinaryOpcode::GE:
    return IsFloat
      ? Builder.CreateFCmp(CmpInst::FCMP_OGE, LoweredLhs, LoweredRhs)
      : Builder.CreateICmp(CmpInst::ICMP_SGE, LoweredLhs, LoweredRhs);

  case HLBinaryOpcode::EQ:
    return IsFloat
      ? Builder.CreateFCmp(CmpInst::FCMP_OEQ, LoweredLhs, LoweredRhs)
      : Builder.CreateICmp(CmpInst::ICMP_EQ, LoweredLhs, LoweredRhs);

  case HLBinaryOpcode::NE:
    return IsFloat
      ? Builder.CreateFCmp(CmpInst::FCMP_ONE, LoweredLhs, LoweredRhs)
      : Builder.CreateICmp(CmpInst::ICMP_NE, LoweredLhs, LoweredRhs);

  case HLBinaryOpcode::UDiv:
    return Builder.CreateUDiv(LoweredLhs, LoweredRhs);

  case HLBinaryOpcode::URem:
    return Builder.CreateURem(LoweredLhs, LoweredRhs);

  case HLBinaryOpcode::UShr:
    return Builder.CreateLShr(LoweredLhs, LoweredRhs);

  case HLBinaryOpcode::ULT:
    return Builder.CreateICmp(CmpInst::ICMP_ULT, LoweredLhs, LoweredRhs);

  case HLBinaryOpcode::UGT:
    return Builder.CreateICmp(CmpInst::ICMP_UGT, LoweredLhs, LoweredRhs);

  case HLBinaryOpcode::ULE:
    return Builder.CreateICmp(CmpInst::ICMP_ULE, LoweredLhs, LoweredRhs);

  case HLBinaryOpcode::UGE:
    return Builder.CreateICmp(CmpInst::ICMP_UGE, LoweredLhs, LoweredRhs);

  case HLBinaryOpcode::LAnd:
  case HLBinaryOpcode::LOr: {
    Value *Zero = Constant::getNullValue(LoweredLhs->getType());
    Value *LhsCmp = IsFloat
      ? Builder.CreateFCmp(CmpInst::FCMP_ONE, LoweredLhs, Zero)
      : Builder.CreateICmp(CmpInst::ICMP_NE, LoweredLhs, Zero);
    Value *RhsCmp = IsFloat
      ? Builder.CreateFCmp(CmpInst::FCMP_ONE, LoweredRhs, Zero)
      : Builder.CreateICmp(CmpInst::ICMP_NE, LoweredRhs, Zero);
    return Opcode == HLBinaryOpcode::LOr
      ? Builder.CreateOr(LhsCmp, RhsCmp)
      : Builder.CreateAnd(LhsCmp, RhsCmp);
  }
  default:
    llvm_unreachable("Unsupported binary matrix operator");
  }
}

Value *HLMatrixLowerPass::lowerHLLoadStore(CallInst *Call, HLMatLoadStoreOpcode Opcode) {
  IRBuilder<> Builder(Call);
  switch (Opcode) {
  case HLMatLoadStoreOpcode::RowMatLoad:
  case HLMatLoadStoreOpcode::ColMatLoad:
    return lowerHLLoad(Call, Call->getArgOperand(HLOperandIndex::kMatLoadPtrOpIdx),
      /* RowMajor */ Opcode == HLMatLoadStoreOpcode::RowMatLoad, Builder);

  case HLMatLoadStoreOpcode::RowMatStore:
  case HLMatLoadStoreOpcode::ColMatStore:
    return lowerHLStore(Call,
      Call->getArgOperand(HLOperandIndex::kMatStoreValOpIdx),
      Call->getArgOperand(HLOperandIndex::kMatStoreDstPtrOpIdx),
      /* RowMajor */ Opcode == HLMatLoadStoreOpcode::RowMatStore,
      /* Return */ !Call->getType()->isVoidTy(), Builder);

  default:
    llvm_unreachable("Unsupported matrix load/store operation");
  }
}

Value *HLMatrixLowerPass::lowerHLLoad(CallInst *Call, Value *MatPtr, bool RowMajor, IRBuilder<> &Builder) {
  HLMatrixType MatTy = HLMatrixType::cast(MatPtr->getType()->getPointerElementType());

  Value *LoweredPtr = tryGetLoweredPtrOperand(MatPtr, Builder);
  if (LoweredPtr == nullptr) {
    // Can't lower this here, defer to HL signature lower
    HLMatLoadStoreOpcode Opcode = RowMajor ? HLMatLoadStoreOpcode::RowMatLoad : HLMatLoadStoreOpcode::ColMatLoad;
    return callHLFunction(
      *m_pModule, HLOpcodeGroup::HLMatLoadStore, static_cast<unsigned>(Opcode),
      MatTy.getLoweredVectorTypeForReg(), { Builder.getInt32((uint32_t)Opcode), MatPtr },
      Call->getCalledFunction()->getAttributes().getFnAttributes(), Builder);
  }

  return MatTy.emitLoweredLoad(LoweredPtr, Builder);
}

Value *HLMatrixLowerPass::lowerHLStore(CallInst *Call, Value *MatVal, Value *MatPtr,
                                       bool RowMajor, bool Return, IRBuilder<> &Builder) {
  DXASSERT(MatVal->getType() == MatPtr->getType()->getPointerElementType(),
    "Matrix store value/pointer type mismatch.");

  Value *LoweredPtr = tryGetLoweredPtrOperand(MatPtr, Builder);
  Value *LoweredVal = getLoweredByValOperand(MatVal, Builder);
  if (LoweredPtr == nullptr) {
    // Can't lower the pointer here, defer to HL signature lower
    HLMatLoadStoreOpcode Opcode = RowMajor ? HLMatLoadStoreOpcode::RowMatStore : HLMatLoadStoreOpcode::ColMatStore;
    return callHLFunction(
      *m_pModule, HLOpcodeGroup::HLMatLoadStore, static_cast<unsigned>(Opcode),
      Return ? LoweredVal->getType() : Builder.getVoidTy(),
      { Builder.getInt32((uint32_t)Opcode), MatPtr, LoweredVal },
      Call->getCalledFunction()->getAttributes().getFnAttributes(), Builder);
  }

  HLMatrixType MatTy = HLMatrixType::cast(MatPtr->getType()->getPointerElementType());
  StoreInst *LoweredStore = MatTy.emitLoweredStore(LoweredVal, LoweredPtr, Builder);

  // If the intrinsic returned a value, return the stored lowered value
  return Return ? LoweredVal : LoweredStore;
}

static Value *convertScalarOrVector(Value *SrcVal, Type *DstTy, HLCastOpcode Opcode, IRBuilder<> Builder) {
  DXASSERT(SrcVal->getType()->isVectorTy() == DstTy->isVectorTy(),
    "Scalar/vector type mismatch in numerical conversion.");
  Type *SrcTy = SrcVal->getType();

  // Conversions between equivalent types are no-ops,
  // even between signed/unsigned variants.
  if (SrcTy == DstTy) return SrcVal;

  // Conversions to bools are comparisons
  if (DstTy->getScalarSizeInBits() == 1) {
    // fcmp une is what regular clang uses in C++ for (bool)f;
    return SrcTy->isIntOrIntVectorTy()
               ? Builder.CreateICmpNE(
                     SrcVal, llvm::Constant::getNullValue(SrcTy), "tobool")
               : Builder.CreateFCmpUNE(
                     SrcVal, llvm::Constant::getNullValue(SrcTy), "tobool");
  }

  // Cast necessary
  bool SrcIsUnsigned = Opcode == HLCastOpcode::FromUnsignedCast ||
    Opcode == HLCastOpcode::UnsignedUnsignedCast;
  bool DstIsUnsigned = Opcode == HLCastOpcode::ToUnsignedCast ||
    Opcode == HLCastOpcode::UnsignedUnsignedCast;
  auto CastOp = static_cast<Instruction::CastOps>(HLModule::GetNumericCastOp(
    SrcTy, SrcIsUnsigned, DstTy, DstIsUnsigned));
  return Builder.CreateCast(CastOp, SrcVal, DstTy);
}

Value *HLMatrixLowerPass::lowerHLCast(CallInst *Call, Value *Src, Type *DstTy,
                                      HLCastOpcode Opcode, IRBuilder<> &Builder) {
  // The opcode really doesn't mean much here, the types involved are what drive most of the casting.
  DXASSERT(Opcode != HLCastOpcode::HandleToResCast, "Unexpected matrix cast opcode.");

  if (dxilutil::IsIntegerOrFloatingPointType(Src->getType())) {
    // Scalar to matrix splat
    HLMatrixType MatDstTy = HLMatrixType::cast(DstTy);

    // Apply element conversion
    Value *Result = convertScalarOrVector(Src,
      MatDstTy.getElementTypeForReg(), Opcode, Builder);

    // Splat to a vector
    Result = Builder.CreateInsertElement(
      UndefValue::get(VectorType::get(Result->getType(), 1)),
      Result, static_cast<uint64_t>(0));
    return Builder.CreateShuffleVector(Result, Result,
      ConstantVector::getSplat(MatDstTy.getNumElements(), Builder.getInt32(0)));
  }
  else if (VectorType *SrcVecTy = dyn_cast<VectorType>(Src->getType())) {
    // Vector to matrix
    HLMatrixType MatDstTy = HLMatrixType::cast(DstTy);
    Value *Result = Src;

    // We might need to truncate
    if (MatDstTy.getNumElements() < SrcVecTy->getNumElements()) {
      SmallVector<int, 4> ShuffleIndices;
      for (unsigned Idx = 0; Idx < MatDstTy.getNumElements(); ++Idx)
        ShuffleIndices.emplace_back(static_cast<int>(Idx));
      Result = Builder.CreateShuffleVector(Src, Src, ShuffleIndices);
    }

    // Apply element conversion
    return convertScalarOrVector(Result,
      MatDstTy.getLoweredVectorTypeForReg(), Opcode, Builder);
  }

  // Source must now be a matrix
  HLMatrixType MatSrcTy = HLMatrixType::cast(Src->getType());
  VectorType* LoweredSrcTy = MatSrcTy.getLoweredVectorTypeForReg();

  Value *LoweredSrc;
  if (isa<Argument>(Src)) {
    // Function arguments are lowered in HLSignatureLower.
    // Initial codegen first generates those cast intrinsics to tell us how to lower them into vectors.
    // Preserve them, but change the return type to vector.
    DXASSERT(Opcode == HLCastOpcode::ColMatrixToVecCast || Opcode == HLCastOpcode::RowMatrixToVecCast,
      "Unexpected cast of matrix argument.");
    LoweredSrc = callHLFunction(*m_pModule, HLOpcodeGroup::HLCast, static_cast<unsigned>(Opcode),
      LoweredSrcTy, { Builder.getInt32((uint32_t)Opcode), Src },
      Call->getCalledFunction()->getAttributes().getFnAttributes(), Builder);
  }
  else {
    LoweredSrc = getLoweredByValOperand(Src, Builder);
  }
  DXASSERT_NOMSG(LoweredSrc->getType() == LoweredSrcTy);

  Value* Result = LoweredSrc;
  Type* LoweredDstTy = DstTy;
  if (dxilutil::IsIntegerOrFloatingPointType(DstTy)) {
    // Matrix to scalar
    Result = Builder.CreateExtractElement(LoweredSrc, static_cast<uint64_t>(0));
  }
  else if (FixedVectorType *DstVecTy = dyn_cast<FixedVectorType>(DstTy)) {
    // Matrix to vector
    DXASSERT(DstVecTy->getNumElements() <= LoweredSrcTy->getNumElements(),
      "Cannot cast matrix to a larger vector.");

    // We might have to truncate
    if (DstVecTy->getNumElements() < LoweredSrcTy->getNumElements()) {
      SmallVector<int, 3> ShuffleIndices;
      for (unsigned Idx = 0; Idx < DstVecTy->getNumElements(); ++Idx)
        ShuffleIndices.emplace_back(static_cast<int>(Idx));
      Result = Builder.CreateShuffleVector(Result, Result, ShuffleIndices);
    }
  }
  else {
    // Destination must now be a matrix too
    HLMatrixType MatDstTy = HLMatrixType::cast(DstTy);

    // Apply any changes at the matrix level: orientation changes and truncation
    if (Opcode == HLCastOpcode::ColMatrixToRowMatrix)
      Result = MatSrcTy.emitLoweredVectorColToRow(Result, Builder);
    else if (Opcode == HLCastOpcode::RowMatrixToColMatrix)
      Result = MatSrcTy.emitLoweredVectorRowToCol(Result, Builder);
    else if (MatDstTy.getNumRows() != MatSrcTy.getNumRows()
      || MatDstTy.getNumColumns() != MatSrcTy.getNumColumns()) {
      // Apply truncation
      DXASSERT(MatDstTy.getNumRows() <= MatSrcTy.getNumRows()
        && MatDstTy.getNumColumns() <= MatSrcTy.getNumColumns(),
        "Unexpected matrix cast between incompatible dimensions.");
      SmallVector<int, 16> ShuffleIndices;
      for (unsigned RowIdx = 0; RowIdx < MatDstTy.getNumRows(); ++RowIdx)
        for (unsigned ColIdx = 0; ColIdx < MatDstTy.getNumColumns(); ++ColIdx)
          ShuffleIndices.emplace_back(static_cast<int>(MatSrcTy.getRowMajorIndex(RowIdx, ColIdx)));
      Result = Builder.CreateShuffleVector(Result, Result, ShuffleIndices);
    }

    LoweredDstTy = MatDstTy.getLoweredVectorTypeForReg();
    DXASSERT(cast<FixedVectorType>(Result->getType())->getNumElements() ==
                 cast<FixedVectorType>(LoweredDstTy)->getNumElements(),
             "Unexpected matrix src/dst lowered element count mismatch after "
             "truncation.");
  }

  // Apply element conversion
  return convertScalarOrVector(Result, LoweredDstTy, Opcode, Builder);
}

Value *HLMatrixLowerPass::lowerHLSubscript(CallInst *Call, HLSubscriptOpcode Opcode) {
  switch (Opcode) {
  case HLSubscriptOpcode::RowMatElement:
  case HLSubscriptOpcode::ColMatElement:
    return lowerHLMatElementSubscript(Call,
      /* RowMajor */ Opcode == HLSubscriptOpcode::RowMatElement);

  case HLSubscriptOpcode::RowMatSubscript:
  case HLSubscriptOpcode::ColMatSubscript:
    return lowerHLMatSubscript(Call,
      /* RowMajor */ Opcode == HLSubscriptOpcode::RowMatSubscript);

  case HLSubscriptOpcode::DefaultSubscript:
  case HLSubscriptOpcode::CBufferSubscript:
    // Those get lowered during HLOperationLower,
    // and the return type must stay unchanged (as a matrix)
    // to provide the metadata to properly emit the loads.
    return nullptr;

  default:
    llvm_unreachable("Unexpected matrix subscript opcode.");
  }
}

Value *HLMatrixLowerPass::lowerHLMatElementSubscript(CallInst *Call, bool RowMajor) {
  (void)RowMajor; // It doesn't look like we actually need this?

  Value *MatPtr = Call->getArgOperand(HLOperandIndex::kMatSubscriptMatOpIdx);
  Constant *IdxVec = cast<Constant>(Call->getArgOperand(HLOperandIndex::kMatSubscriptSubOpIdx));
  VectorType *IdxVecTy = cast<VectorType>(IdxVec->getType());

  // Get the loaded lowered vector element indices
  SmallVector<Value*, 4> ElemIndices;
  ElemIndices.reserve(IdxVecTy->getNumElements());
  for (unsigned VecIdx = 0; VecIdx < IdxVecTy->getNumElements(); ++VecIdx) {
    ElemIndices.emplace_back(IdxVec->getAggregateElement(VecIdx));
  }

  lowerHLMatSubscript(Call, MatPtr, ElemIndices);

  // We did our own replacement of uses, opt-out of having the caller does it for us.
  return nullptr;
}

Value *HLMatrixLowerPass::lowerHLMatSubscript(CallInst *Call, bool RowMajor) {
  (void)RowMajor; // It doesn't look like we actually need this?

  Value *MatPtr = Call->getArgOperand(HLOperandIndex::kMatSubscriptMatOpIdx);

  // Gather the indices, checking if they are all constant
  SmallVector<Value*, 4> ElemIndices;
  for (unsigned Idx = HLOperandIndex::kMatSubscriptSubOpIdx; Idx < Call->getNumArgOperands(); ++Idx) {
    ElemIndices.emplace_back(Call->getArgOperand(Idx));
  }

  lowerHLMatSubscript(Call, MatPtr, ElemIndices);

  // We did our own replacement of uses, opt-out of having the caller does it for us.
  return nullptr;
}

void HLMatrixLowerPass::lowerHLMatSubscript(CallInst *Call, Value *MatPtr, SmallVectorImpl<Value*> &ElemIndices) {
  DXASSERT_NOMSG(HLMatrixType::isMatrixPtr(MatPtr->getType()));

  IRBuilder<> CallBuilder(Call);
  Value *LoweredPtr = tryGetLoweredPtrOperand(MatPtr, CallBuilder);
  Value *LoweredMatrix = nullptr;
  Value *RootPtr = LoweredPtr? LoweredPtr: MatPtr;
  while (GEPOperator *GEP = dyn_cast<GEPOperator>(RootPtr))
    RootPtr = GEP->getPointerOperand();

  if (LoweredPtr == nullptr) {
    if (!isa<Argument>(RootPtr))
      return;

    // For a shader input, load the matrix into a lowered ptr
    // The load will be handled by LowerSignature
    HLMatLoadStoreOpcode Opcode = (HLSubscriptOpcode)GetHLOpcode(Call) == HLSubscriptOpcode::RowMatSubscript ?
                                   HLMatLoadStoreOpcode::RowMatLoad : HLMatLoadStoreOpcode::ColMatLoad;
    HLMatrixType MatTy = HLMatrixType::cast(MatPtr->getType()->getPointerElementType());
    LoweredMatrix = callHLFunction(
      *m_pModule, HLOpcodeGroup::HLMatLoadStore, static_cast<unsigned>(Opcode),
      MatTy.getLoweredVectorTypeForReg(), { CallBuilder.getInt32((uint32_t)Opcode), MatPtr },
      Call->getCalledFunction()->getAttributes().getFnAttributes(), CallBuilder);
  }
  // For global variables, we can GEP directly into the lowered vector pointer.
  // This is necessary to support group shared memory atomics and the likes.
  bool AllowLoweredPtrGEPs = isa<GlobalVariable>(RootPtr);
  
  // Just constructing this does all the work
  HLMatrixSubscriptUseReplacer UseReplacer(Call, LoweredPtr, LoweredMatrix,
                                           ElemIndices, AllowLoweredPtrGEPs, m_deadInsts);

  DXASSERT(Call->use_empty(), "Expected all matrix subscript uses to have been replaced.");
  addToDeadInsts(Call);
}

Value *HLMatrixLowerPass::lowerHLInit(CallInst *Call) {
  DXASSERT(GetHLOpcode(Call) == 0, "Unexpected matrix init opcode.");

  // Figure out the result type
  HLMatrixType MatTy = HLMatrixType::cast(Call->getType());
  VectorType *LoweredTy = MatTy.getLoweredVectorTypeForReg();

  // Handle case where produced by EmitHLSLFlatConversion where there's one
  // vector argument, instead of scalar arguments.
  if (1 == Call->getNumArgOperands() - HLOperandIndex::kInitFirstArgOpIdx &&
      Call->getArgOperand(HLOperandIndex::kInitFirstArgOpIdx)->
              getType()->isVectorTy()) {
    Value *LoweredVec = Call->getArgOperand(HLOperandIndex::kInitFirstArgOpIdx);
    DXASSERT(LoweredTy->getNumElements() ==
                cast<FixedVectorType>(LoweredVec->getType())->getNumElements(),
             "Invalid matrix init argument vector element count.");
    return LoweredVec;
  }

  DXASSERT(LoweredTy->getNumElements() == Call->getNumArgOperands() - HLOperandIndex::kInitFirstArgOpIdx,
    "Invalid matrix init argument count.");

  // Build the result vector from the init args.
  // Both the args and the result vector are in row-major order, so no shuffling is necessary.
  IRBuilder<> Builder(Call);
  Value *LoweredVec = UndefValue::get(LoweredTy);
  for (unsigned VecElemIdx = 0; VecElemIdx < LoweredTy->getNumElements(); ++VecElemIdx) {
    Value *ArgVal = Call->getArgOperand(HLOperandIndex::kInitFirstArgOpIdx + VecElemIdx);
    DXASSERT(dxilutil::IsIntegerOrFloatingPointType(ArgVal->getType()),
      "Expected only scalars in matrix initialization.");
    LoweredVec = Builder.CreateInsertElement(LoweredVec, ArgVal, static_cast<uint64_t>(VecElemIdx));
  }

  return LoweredVec;
}

Value *HLMatrixLowerPass::lowerHLSelect(CallInst *Call) {
  DXASSERT(GetHLOpcode(Call) == 0, "Unexpected matrix init opcode.");

  Value *Cond = Call->getArgOperand(HLOperandIndex::kTrinaryOpSrc0Idx);
  Value *TrueMat = Call->getArgOperand(HLOperandIndex::kTrinaryOpSrc1Idx);
  Value *FalseMat = Call->getArgOperand(HLOperandIndex::kTrinaryOpSrc2Idx);

  DXASSERT(TrueMat->getType() == FalseMat->getType(),
    "Unexpected type mismatch between matrix ternary operator values.");

#ifndef NDEBUG
  // Assert that if the condition is a matrix, it matches the dimensions of the values
  if (HLMatrixType MatCondTy = HLMatrixType::dyn_cast(Cond->getType())) {
    HLMatrixType ValMatTy = HLMatrixType::cast(TrueMat->getType());
    DXASSERT(MatCondTy.getNumRows() == ValMatTy.getNumRows()
      && MatCondTy.getNumColumns() == ValMatTy.getNumColumns(),
      "Unexpected mismatch between ternary operator condition and value matrix dimensions.");
  }
#endif

  IRBuilder<> Builder(Call);
  Value *LoweredCond = getLoweredByValOperand(Cond, Builder);
  Value *LoweredTrueVec = getLoweredByValOperand(TrueMat, Builder);
  Value *LoweredFalseVec = getLoweredByValOperand(FalseMat, Builder);
  Value *Result = UndefValue::get(LoweredTrueVec->getType());

  bool IsScalarCond = !LoweredCond->getType()->isVectorTy();

  unsigned NumElems = cast<FixedVectorType>(Result->getType())->getNumElements();
  for (uint64_t ElemIdx = 0; ElemIdx < NumElems; ++ElemIdx) {
    Value *ElemCond = IsScalarCond ? LoweredCond
      : Builder.CreateExtractElement(LoweredCond, ElemIdx);
    Value *ElemTrueVal = Builder.CreateExtractElement(LoweredTrueVec, ElemIdx);
    Value *ElemFalseVal = Builder.CreateExtractElement(LoweredFalseVec, ElemIdx);
    Value *ResultElem = Builder.CreateSelect(ElemCond, ElemTrueVal, ElemFalseVal);
    Result = Builder.CreateInsertElement(Result, ResultElem, ElemIdx);
  }

  return Result;
}
