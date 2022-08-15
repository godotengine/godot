//===- DxilConditionalMem2Reg.cpp - Mem2Reg that selectively promotes Allocas ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Pass.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DIBuilder.h"

#include "dxc/DXIL/DxilUtil.h"
#include "dxc/HLSL/HLModule.h"
#include "llvm/Analysis/DxilValueCache.h"
#include "llvm/Analysis/ValueTracking.h"

using namespace llvm;
using namespace hlsl;

static bool ContainsFloatingPointType(Type *Ty) {
  if (Ty->isFloatingPointTy()) {
    return true;
  }
  else if (Ty->isArrayTy()) {
    return ContainsFloatingPointType(Ty->getArrayElementType());
  }
  else if (Ty->isVectorTy()) {
    return ContainsFloatingPointType(Ty->getVectorElementType());
  }
  else if (Ty->isStructTy()) {
    for (unsigned i = 0, NumStructElms = Ty->getStructNumElements(); i < NumStructElms; i++) {
      if (ContainsFloatingPointType(Ty->getStructElementType(i)))
        return true;
    }
  }
  return false;
}

static bool Mem2Reg(Function &F, DominatorTree &DT, AssumptionCache &AC) {
  BasicBlock &BB = F.getEntryBlock();  // Get the entry node for the function
  bool Changed  = false;
  std::vector<AllocaInst*> Allocas;
  while (1) {
    Allocas.clear();

    // Find allocas that are safe to promote, by looking at all instructions in
    // the entry node
    for (BasicBlock::iterator I = BB.begin(), E = --BB.end(); I != E; ++I)
      if (AllocaInst *AI = dyn_cast<AllocaInst>(I))       // Is it an alloca?
        if (isAllocaPromotable(AI) &&
          (!HLModule::HasPreciseAttributeWithMetadata(AI) || !ContainsFloatingPointType(AI->getAllocatedType())))
          Allocas.push_back(AI);

    if (Allocas.empty()) break;

    PromoteMemToReg(Allocas, DT, nullptr, &AC);
    Changed = true;
  }

  return Changed;
}

//
// Special Mem2Reg pass that conditionally promotes or transforms Alloca's.
//
// Anything marked 'dx.precise', will not be promoted because precise markers
// are not propagated to the dxil operations yet and will be lost if alloca
// is removed right now.
//
// Precise Allocas of vectors get scalarized here. It's important we do that
// before Scalarizer pass because promoting the allocas later than that will
// produce vector phi's (disallowed by the validator), which need another
// Scalarizer pass to clean up.
//
class DxilConditionalMem2Reg : public FunctionPass {
public:
  static char ID;

  // Function overrides that resolve options when used for DxOpt
  void applyOptions(PassOptions O) override {
    GetPassOptionBool(O, "NoOpt", &NoOpt, false);
  }
  void dumpConfig(raw_ostream &OS) override {
    FunctionPass::dumpConfig(OS);
    OS << ",NoOpt=" << NoOpt;
  }

  bool NoOpt = false;
  explicit DxilConditionalMem2Reg(bool NoOpt=false) : FunctionPass(ID), NoOpt(NoOpt)
  {
    initializeDxilConditionalMem2RegPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<AssumptionCacheTracker>();
    AU.setPreservesCFG();
  }

  //
  // Turns all allocas of vector types that are marked with 'dx.precise'
  // and turn them into scalars. For example:
  //
  //    x = alloca <f32 x 4> !dx.precise
  //
  // becomes:
  //
  //    x1 = alloca f32 !dx.precise
  //    x2 = alloca f32 !dx.precise
  //    x3 = alloca f32 !dx.precise
  //    x4 = alloca f32 !dx.precise
  //
  // This function also replaces all stores and loads but leaves everything
  // else alone by generating insertelement and extractelement as appropriate.
  //
  static bool ScalarizePreciseVectorAlloca(Function &F) {
    BasicBlock *Entry = &*F.begin();

    bool Changed = false;
    for (auto it = Entry->begin(); it != Entry->end();) {
      Instruction *I = &*(it++);
      AllocaInst *AI = dyn_cast<AllocaInst>(I);
      if (!AI || !AI->getAllocatedType()->isVectorTy()) continue;
      if (!HLModule::HasPreciseAttributeWithMetadata(AI)) continue;


      IRBuilder<> B(AI);
      VectorType *VTy = cast<VectorType>(AI->getAllocatedType());
      Type *ScalarTy = VTy->getVectorElementType();

      const unsigned VectorSize = VTy->getVectorNumElements();
      SmallVector<AllocaInst *, 32> Elements;

      for (unsigned i = 0; i < VectorSize; i++) {
        AllocaInst *Elem = B.CreateAlloca(ScalarTy);
        hlsl::DxilMDHelper::CopyMetadata(*Elem, *AI);
        Elements.push_back(Elem);
      }

      for (auto it = AI->user_begin(); it != AI->user_end();) {
        User *U = *(it++);
        if (LoadInst *LI = dyn_cast<LoadInst>(U)) {
          B.SetInsertPoint(LI);
          Value *Vec = UndefValue::get(VTy);
          for (unsigned i = 0; i < VectorSize; i++) {
            LoadInst *Elem = B.CreateLoad(Elements[i]);
            hlsl::DxilMDHelper::CopyMetadata(*Elem, *LI);
            Vec = B.CreateInsertElement(Vec, Elem, i);
          }

          LI->replaceAllUsesWith(Vec);
          LI->eraseFromParent();
        }
        else if (StoreInst *Store = dyn_cast<StoreInst>(U)) {
          B.SetInsertPoint(Store);
          Value *Vec = Store->getValueOperand();
          for (unsigned i = 0; i < VectorSize; i++) {
            Value *Elem = B.CreateExtractElement(Vec, i);
            StoreInst *ElemStore = B.CreateStore(Elem, Elements[i]);
            hlsl::DxilMDHelper::CopyMetadata(*ElemStore, *Store);
          }
          Store->eraseFromParent();
        }
        else if (BitCastInst *BCI = dyn_cast<BitCastInst>(U)) {
          DXASSERT(onlyUsedByLifetimeMarkers(BCI),
                   "expected bitcast to only be used by lifetime intrinsics");
          for (auto BCIU = BCI->user_begin(), BCIE = BCI->user_end(); BCIU != BCIE;) {
            IntrinsicInst *II = cast<IntrinsicInst>(*(BCIU++));
            II->eraseFromParent();
          }
          BCI->eraseFromParent();
        }
        else {
          llvm_unreachable("Cannot handle non-store/load on precise vector allocas");
        }
      }

      AI->eraseFromParent();
      Changed = true;
    }
    return Changed;
  }

  struct StoreInfo {
    Value *V;
    unsigned Offset;
  };
  static bool FindAllStores(Module &M, Value *V, SmallVectorImpl<StoreInfo> *Stores) {
    SmallVector<StoreInfo, 8> Worklist;
    std::set<Value *> Seen;

    auto Add = [&](Value *V, unsigned OffsetInBits) {
      if (Seen.insert(V).second)
        Worklist.push_back({ V, OffsetInBits });
    };

    Add(V, 0);

    const DataLayout &DL = M.getDataLayout();

    while (Worklist.size()) {
      auto Info = Worklist.pop_back_val();
      auto *Elem = Info.V;

      if (auto GEP = dyn_cast<GEPOperator>(Elem)) {
        if (GEP->getNumIndices() != 2)
          continue;

        unsigned ElemSize = 0;

        Type *GEPPtrType = GEP->getPointerOperand()->getType();
        Type *PtrElemType = GEPPtrType->getPointerElementType();
        if (ArrayType *ArrayTy = dyn_cast<ArrayType>(PtrElemType)) {
          ElemSize = DL.getTypeAllocSizeInBits(ArrayTy->getElementType());
        }
        else if (VectorType *VectorTy = dyn_cast<VectorType>(PtrElemType)) {
          ElemSize = DL.getTypeAllocSizeInBits(VectorTy->getElementType());
        }
        else {
          return false;
        }

        unsigned OffsetInBits = 0;
        for (unsigned i = 0; i < GEP->getNumIndices(); i++) {
          auto IdxOp = dyn_cast<ConstantInt>(GEP->getOperand(i+1));
          if (!IdxOp) {
            return false;
          }
          uint64_t Idx = IdxOp->getLimitedValue();
          if (i == 0) {
            if (Idx != 0)
              return false;
          }
          else {
            OffsetInBits = Idx * ElemSize;
          }
        }

        for (User *U : Elem->users())
          Add(U, Info.Offset + OffsetInBits);
      }
      else if (auto *Store = dyn_cast<StoreInst>(Elem)) {
        Stores->push_back({ Store, Info.Offset });
      }
    }

    return true;
  }

  // Function to rewrite debug info for output argument.
  // Sometimes, normal local variables that get returned from functions get rewritten as
  // a pointer argument.
  //
  // Right now, we generally have a single dbg.declare for the Argument, but as we lower
  // it to storeOutput, the dbg.declare and the Argument both get removed, leavning no
  // debug info for the local variable.
  //
  // Solution here is to rewrite the dbg.declare as dbg.value's by finding all the stores
  // and writing a dbg.value immediately before the store. Fairly conservative at the moment 
  // about what cases to rewrite (only scalars and vectors, and arrays of scalars and vectors).
  //
  bool RewriteOutputArgsDebugInfo(Function &F) {
    bool Changed = false;
    Module *M = F.getParent();
    DIBuilder DIB(*M);

    SmallVector<StoreInfo, 4> Stores;
    LLVMContext &Ctx = F.getContext();
    for (Argument &Arg : F.args()) {
      if (!Arg.getType()->isPointerTy())
        continue;
      Type *Ty = Arg.getType()->getPointerElementType();

      bool IsSimpleType =
        Ty->isSingleValueType() ||
        Ty->isVectorTy() ||
        (Ty->isArrayTy() && (Ty->getArrayElementType()->isVectorTy() || Ty->getArrayElementType()->isSingleValueType()));

      if (!IsSimpleType)
        continue;

      Stores.clear();
      for (User *U : Arg.users()) {
        if (!FindAllStores(*M, U, &Stores)) {
          Stores.clear();
          break;
        }
      }

      if (Stores.empty())
        continue;

      DbgDeclareInst *Declare = nullptr;
      if (auto *L = LocalAsMetadata::getIfExists(&Arg)) {
        if (auto *DINode = MetadataAsValue::getIfExists(Ctx, L)) {
          if (!DINode->user_empty() && std::next(DINode->user_begin()) == DINode->user_end()) {
            Declare = dyn_cast<DbgDeclareInst>(*DINode->user_begin());
          }
        }
      }

      if (Declare) {
        DITypeIdentifierMap EmptyMap;
        DILocalVariable *Var = Declare->getVariable();
        DIExpression *Expr = Declare->getExpression();
        DIType *VarTy = Var->getType().resolve(EmptyMap);
        uint64_t VarSize = VarTy->getSizeInBits();
        uint64_t Offset = 0;
        if (Expr->isBitPiece())
          Offset = Expr->getBitPieceOffset();

        for (auto &Info : Stores) {
          auto *Store = cast<StoreInst>(Info.V);
          auto Val = Store->getValueOperand();
          auto Loc = Store->getDebugLoc();
          auto &M = *F.getParent();
          unsigned ValSize = M.getDataLayout().getTypeAllocSizeInBits(Val->getType());

          DIExpression *NewExpr = nullptr;
          if (Offset || VarSize > ValSize) {
            uint64_t Elems[] = { dwarf::DW_OP_bit_piece, Offset + Info.Offset, ValSize };
            NewExpr = DIExpression::get(Ctx, Elems);
          }
          else {
            NewExpr = DIExpression::get(Ctx, {});
          }
          if (Loc->getScope()->getSubprogram() == Var->getScope()->getSubprogram())
            DIB.insertDbgValueIntrinsic(Val, 0, Var, NewExpr, Loc, Store);
        }

        Declare->eraseFromParent();
        Changed = true;
      }
    }

    return Changed;
  }

  bool runOnFunction(Function &F) override {
    DominatorTree *DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    AssumptionCache *AC = &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
    bool Changed = false;

    Changed |= RewriteOutputArgsDebugInfo(F);
    Changed |= dxilutil::DeleteDeadAllocas(F);
    Changed |= ScalarizePreciseVectorAlloca(F);
    Changed |= Mem2Reg(F, *DT, *AC);

    return Changed;
  }
};
char DxilConditionalMem2Reg::ID;

Pass *llvm::createDxilConditionalMem2RegPass(bool NoOpt) {
  return new DxilConditionalMem2Reg(NoOpt);
}

INITIALIZE_PASS_BEGIN(DxilConditionalMem2Reg, "dxil-cond-mem2reg", "Dxil Conditional Mem2Reg", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_END(DxilConditionalMem2Reg, "dxil-cond-mem2reg", "Dxil Conditional Mem2Reg", false, false)
