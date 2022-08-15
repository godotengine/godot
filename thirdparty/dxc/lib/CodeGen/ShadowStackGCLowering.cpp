//===-- ShadowStackGCLowering.cpp - Custom lowering for shadow-stack gc ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the custom lowering code required by the shadow-stack GC
// strategy.  
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/GCStrategy.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"

using namespace llvm;

#define DEBUG_TYPE "shadowstackgclowering"

namespace {

class ShadowStackGCLowering : public FunctionPass {
  /// RootChain - This is the global linked-list that contains the chain of GC
  /// roots.
  GlobalVariable *Head;

  /// StackEntryTy - Abstract type of a link in the shadow stack.
  ///
  StructType *StackEntryTy;
  StructType *FrameMapTy;

  /// Roots - GC roots in the current function. Each is a pair of the
  /// intrinsic call and its corresponding alloca.
  std::vector<std::pair<CallInst *, AllocaInst *>> Roots;

public:
  static char ID;
  ShadowStackGCLowering();

  bool doInitialization(Module &M) override;
  bool runOnFunction(Function &F) override;

private:
  bool IsNullValue(Value *V);
  Constant *GetFrameMap(Function &F);
  Type *GetConcreteStackEntryType(Function &F);
  void CollectRoots(Function &F);
  static GetElementPtrInst *CreateGEP(LLVMContext &Context, IRBuilder<> &B,
                                      Type *Ty, Value *BasePtr, int Idx1,
                                      const char *Name);
  static GetElementPtrInst *CreateGEP(LLVMContext &Context, IRBuilder<> &B,
                                      Type *Ty, Value *BasePtr, int Idx1, int Idx2,
                                      const char *Name);
};
}

INITIALIZE_PASS_BEGIN(ShadowStackGCLowering, "shadow-stack-gc-lowering",
                      "Shadow Stack GC Lowering", false, false)
INITIALIZE_PASS_DEPENDENCY(GCModuleInfo)
INITIALIZE_PASS_END(ShadowStackGCLowering, "shadow-stack-gc-lowering",
                    "Shadow Stack GC Lowering", false, false)

FunctionPass *llvm::createShadowStackGCLoweringPass() { return new ShadowStackGCLowering(); }

char ShadowStackGCLowering::ID = 0;

ShadowStackGCLowering::ShadowStackGCLowering()
  : FunctionPass(ID), Head(nullptr), StackEntryTy(nullptr),
    FrameMapTy(nullptr) {
  initializeShadowStackGCLoweringPass(*PassRegistry::getPassRegistry());
}

namespace {
/// EscapeEnumerator - This is a little algorithm to find all escape points
/// from a function so that "finally"-style code can be inserted. In addition
/// to finding the existing return and unwind instructions, it also (if
/// necessary) transforms any call instructions into invokes and sends them to
/// a landing pad.
///
/// It's wrapped up in a state machine using the same transform C# uses for
/// 'yield return' enumerators, This transform allows it to be non-allocating.
class EscapeEnumerator {
  Function &F;
  const char *CleanupBBName;

  // State.
  int State;
  Function::iterator StateBB, StateE;
  IRBuilder<> Builder;

public:
  EscapeEnumerator(Function &F, const char *N = "cleanup")
      : F(F), CleanupBBName(N), State(0), Builder(F.getContext()) {}

  IRBuilder<> *Next() {
    switch (State) {
    default:
      return nullptr;

    case 0:
      StateBB = F.begin();
      StateE = F.end();
      State = 1;

    case 1:
      // Find all 'return', 'resume', and 'unwind' instructions.
      while (StateBB != StateE) {
        BasicBlock *CurBB = StateBB++;

        // Branches and invokes do not escape, only unwind, resume, and return
        // do.
        TerminatorInst *TI = CurBB->getTerminator();
        if (!isa<ReturnInst>(TI) && !isa<ResumeInst>(TI))
          continue;

        Builder.SetInsertPoint(TI->getParent(), TI);
        return &Builder;
      }

      State = 2;

      // Find all 'call' instructions.
      SmallVector<Instruction *, 16> Calls;
      for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
        for (BasicBlock::iterator II = BB->begin(), EE = BB->end(); II != EE;
             ++II)
          if (CallInst *CI = dyn_cast<CallInst>(II))
            if (!CI->getCalledFunction() ||
                !CI->getCalledFunction()->getIntrinsicID())
              Calls.push_back(CI);

      if (Calls.empty())
        return nullptr;

      // Create a cleanup block.
      LLVMContext &C = F.getContext();
      BasicBlock *CleanupBB = BasicBlock::Create(C, CleanupBBName, &F);
      Type *ExnTy =
          StructType::get(Type::getInt8PtrTy(C), Type::getInt32Ty(C), nullptr);
      if (!F.hasPersonalityFn()) {
        Constant *PersFn = F.getParent()->getOrInsertFunction(
            "__gcc_personality_v0",
            FunctionType::get(Type::getInt32Ty(C), true));
        F.setPersonalityFn(PersFn);
      }
      LandingPadInst *LPad =
          LandingPadInst::Create(ExnTy, 1, "cleanup.lpad", CleanupBB);
      LPad->setCleanup(true);
      ResumeInst *RI = ResumeInst::Create(LPad, CleanupBB);

      // Transform the 'call' instructions into 'invoke's branching to the
      // cleanup block. Go in reverse order to make prettier BB names.
      SmallVector<Value *, 16> Args;
      for (unsigned I = Calls.size(); I != 0;) {
        CallInst *CI = cast<CallInst>(Calls[--I]);

        // Split the basic block containing the function call.
        BasicBlock *CallBB = CI->getParent();
        BasicBlock *NewBB =
            CallBB->splitBasicBlock(CI, CallBB->getName() + ".cont");

        // Remove the unconditional branch inserted at the end of CallBB.
        CallBB->getInstList().pop_back();
        NewBB->getInstList().remove(CI);

        // Create a new invoke instruction.
        Args.clear();
        CallSite CS(CI);
        Args.append(CS.arg_begin(), CS.arg_end());

        InvokeInst *II =
            InvokeInst::Create(CI->getCalledValue(), NewBB, CleanupBB, Args,
                               CI->getName(), CallBB);
        II->setCallingConv(CI->getCallingConv());
        II->setAttributes(CI->getAttributes());
        CI->replaceAllUsesWith(II);
        delete CI;
      }

      Builder.SetInsertPoint(RI->getParent(), RI);
      return &Builder;
    }
  }
};
}


Constant *ShadowStackGCLowering::GetFrameMap(Function &F) {
  // doInitialization creates the abstract type of this value.
  Type *VoidPtr = Type::getInt8PtrTy(F.getContext());

  // Truncate the ShadowStackDescriptor if some metadata is null.
  unsigned NumMeta = 0;
  SmallVector<Constant *, 16> Metadata;
  for (unsigned I = 0; I != Roots.size(); ++I) {
    Constant *C = cast<Constant>(Roots[I].first->getArgOperand(1));
    if (!C->isNullValue())
      NumMeta = I + 1;
    Metadata.push_back(ConstantExpr::getBitCast(C, VoidPtr));
  }
  Metadata.resize(NumMeta);

  Type *Int32Ty = Type::getInt32Ty(F.getContext());

  Constant *BaseElts[] = {
      ConstantInt::get(Int32Ty, Roots.size(), false),
      ConstantInt::get(Int32Ty, NumMeta, false),
  };

  Constant *DescriptorElts[] = {
      ConstantStruct::get(FrameMapTy, BaseElts),
      ConstantArray::get(ArrayType::get(VoidPtr, NumMeta), Metadata)};

  Type *EltTys[] = {DescriptorElts[0]->getType(), DescriptorElts[1]->getType()};
  StructType *STy = StructType::create(EltTys, "gc_map." + utostr(NumMeta));

  Constant *FrameMap = ConstantStruct::get(STy, DescriptorElts);

  // FIXME: Is this actually dangerous as WritingAnLLVMPass.html claims? Seems
  //        that, short of multithreaded LLVM, it should be safe; all that is
  //        necessary is that a simple Module::iterator loop not be invalidated.
  //        Appending to the GlobalVariable list is safe in that sense.
  //
  //        All of the output passes emit globals last. The ExecutionEngine
  //        explicitly supports adding globals to the module after
  //        initialization.
  //
  //        Still, if it isn't deemed acceptable, then this transformation needs
  //        to be a ModulePass (which means it cannot be in the 'llc' pipeline
  //        (which uses a FunctionPassManager (which segfaults (not asserts) if
  //        provided a ModulePass))).
  Constant *GV = new GlobalVariable(*F.getParent(), FrameMap->getType(), true,
                                    GlobalVariable::InternalLinkage, FrameMap,
                                    "__gc_" + F.getName());

  Constant *GEPIndices[2] = {
      ConstantInt::get(Type::getInt32Ty(F.getContext()), 0),
      ConstantInt::get(Type::getInt32Ty(F.getContext()), 0)};
  return ConstantExpr::getGetElementPtr(FrameMap->getType(), GV, GEPIndices);
}

Type *ShadowStackGCLowering::GetConcreteStackEntryType(Function &F) {
  // doInitialization creates the generic version of this type.
  std::vector<Type *> EltTys;
  EltTys.push_back(StackEntryTy);
  for (size_t I = 0; I != Roots.size(); I++)
    EltTys.push_back(Roots[I].second->getAllocatedType());

  return StructType::create(EltTys, ("gc_stackentry." + F.getName()).str());
}

/// doInitialization - If this module uses the GC intrinsics, find them now. If
/// not, exit fast.
bool ShadowStackGCLowering::doInitialization(Module &M) {
  bool Active = false;
  for (Function &F : M) {
    if (F.hasGC() && F.getGC() == std::string("shadow-stack")) {
      Active = true;
      break;
    }
  }
  if (!Active)
    return false;
  
  // struct FrameMap {
  //   int32_t NumRoots; // Number of roots in stack frame.
  //   int32_t NumMeta;  // Number of metadata descriptors. May be < NumRoots.
  //   void *Meta[];     // May be absent for roots without metadata.
  // };
  std::vector<Type *> EltTys;
  // 32 bits is ok up to a 32GB stack frame. :)
  EltTys.push_back(Type::getInt32Ty(M.getContext()));
  // Specifies length of variable length array.
  EltTys.push_back(Type::getInt32Ty(M.getContext()));
  FrameMapTy = StructType::create(EltTys, "gc_map");
  PointerType *FrameMapPtrTy = PointerType::getUnqual(FrameMapTy);

  // struct StackEntry {
  //   ShadowStackEntry *Next; // Caller's stack entry.
  //   FrameMap *Map;          // Pointer to constant FrameMap.
  //   void *Roots[];          // Stack roots (in-place array, so we pretend).
  // };

  StackEntryTy = StructType::create(M.getContext(), "gc_stackentry");

  EltTys.clear();
  EltTys.push_back(PointerType::getUnqual(StackEntryTy));
  EltTys.push_back(FrameMapPtrTy);
  StackEntryTy->setBody(EltTys);
  PointerType *StackEntryPtrTy = PointerType::getUnqual(StackEntryTy);

  // Get the root chain if it already exists.
  Head = M.getGlobalVariable("llvm_gc_root_chain");
  if (!Head) {
    // If the root chain does not exist, insert a new one with linkonce
    // linkage!
    Head = new GlobalVariable(
        M, StackEntryPtrTy, false, GlobalValue::LinkOnceAnyLinkage,
        Constant::getNullValue(StackEntryPtrTy), "llvm_gc_root_chain");
  } else if (Head->hasExternalLinkage() && Head->isDeclaration()) {
    Head->setInitializer(Constant::getNullValue(StackEntryPtrTy));
    Head->setLinkage(GlobalValue::LinkOnceAnyLinkage);
  }

  return true;
}

bool ShadowStackGCLowering::IsNullValue(Value *V) {
  if (Constant *C = dyn_cast<Constant>(V))
    return C->isNullValue();
  return false;
}

void ShadowStackGCLowering::CollectRoots(Function &F) {
  // FIXME: Account for original alignment. Could fragment the root array.
  //   Approach 1: Null initialize empty slots at runtime. Yuck.
  //   Approach 2: Emit a map of the array instead of just a count.

  assert(Roots.empty() && "Not cleaned up?");

  SmallVector<std::pair<CallInst *, AllocaInst *>, 16> MetaRoots;

  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
    for (BasicBlock::iterator II = BB->begin(), E = BB->end(); II != E;)
      if (IntrinsicInst *CI = dyn_cast<IntrinsicInst>(II++))
        if (Function *F = CI->getCalledFunction())
          if (F->getIntrinsicID() == Intrinsic::gcroot) {
            std::pair<CallInst *, AllocaInst *> Pair = std::make_pair(
                CI,
                cast<AllocaInst>(CI->getArgOperand(0)->stripPointerCasts()));
            if (IsNullValue(CI->getArgOperand(1)))
              Roots.push_back(Pair);
            else
              MetaRoots.push_back(Pair);
          }

  // Number roots with metadata (usually empty) at the beginning, so that the
  // FrameMap::Meta array can be elided.
  Roots.insert(Roots.begin(), MetaRoots.begin(), MetaRoots.end());
}

GetElementPtrInst *ShadowStackGCLowering::CreateGEP(LLVMContext &Context,
                                                    IRBuilder<> &B, Type *Ty,
                                                    Value *BasePtr, int Idx,
                                                    int Idx2,
                                                    const char *Name) {
  Value *Indices[] = {ConstantInt::get(Type::getInt32Ty(Context), 0),
                      ConstantInt::get(Type::getInt32Ty(Context), Idx),
                      ConstantInt::get(Type::getInt32Ty(Context), Idx2)};
  Value *Val = B.CreateGEP(Ty, BasePtr, Indices, Name);

  assert(isa<GetElementPtrInst>(Val) && "Unexpected folded constant");

  return dyn_cast<GetElementPtrInst>(Val);
}

GetElementPtrInst *ShadowStackGCLowering::CreateGEP(LLVMContext &Context,
                                            IRBuilder<> &B, Type *Ty, Value *BasePtr,
                                            int Idx, const char *Name) {
  Value *Indices[] = {ConstantInt::get(Type::getInt32Ty(Context), 0),
                      ConstantInt::get(Type::getInt32Ty(Context), Idx)};
  Value *Val = B.CreateGEP(Ty, BasePtr, Indices, Name);

  assert(isa<GetElementPtrInst>(Val) && "Unexpected folded constant");

  return dyn_cast<GetElementPtrInst>(Val);
}

/// runOnFunction - Insert code to maintain the shadow stack.
bool ShadowStackGCLowering::runOnFunction(Function &F) {
  // Quick exit for functions that do not use the shadow stack GC.
  if (!F.hasGC() ||
      F.getGC() != std::string("shadow-stack"))
    return false;
  
  LLVMContext &Context = F.getContext();

  // Find calls to llvm.gcroot.
  CollectRoots(F);

  // If there are no roots in this function, then there is no need to add a
  // stack map entry for it.
  if (Roots.empty())
    return false;

  // Build the constant map and figure the type of the shadow stack entry.
  Value *FrameMap = GetFrameMap(F);
  Type *ConcreteStackEntryTy = GetConcreteStackEntryType(F);

  // Build the shadow stack entry at the very start of the function.
  BasicBlock::iterator IP = F.getEntryBlock().begin();
  IRBuilder<> AtEntry(IP->getParent(), IP);

  Instruction *StackEntry =
      AtEntry.CreateAlloca(ConcreteStackEntryTy, nullptr, "gc_frame");

  while (isa<AllocaInst>(IP))
    ++IP;
  AtEntry.SetInsertPoint(IP->getParent(), IP);

  // Initialize the map pointer and load the current head of the shadow stack.
  Instruction *CurrentHead = AtEntry.CreateLoad(Head, "gc_currhead");
  Instruction *EntryMapPtr = CreateGEP(Context, AtEntry, ConcreteStackEntryTy,
                                       StackEntry, 0, 1, "gc_frame.map");
  AtEntry.CreateStore(FrameMap, EntryMapPtr);

  // After all the allocas...
  for (unsigned I = 0, E = Roots.size(); I != E; ++I) {
    // For each root, find the corresponding slot in the aggregate...
    Value *SlotPtr = CreateGEP(Context, AtEntry, ConcreteStackEntryTy,
                               StackEntry, 1 + I, "gc_root");

    // And use it in lieu of the alloca.
    AllocaInst *OriginalAlloca = Roots[I].second;
    SlotPtr->takeName(OriginalAlloca);
    OriginalAlloca->replaceAllUsesWith(SlotPtr);
  }

  // Move past the original stores inserted by GCStrategy::InitRoots. This isn't
  // really necessary (the collector would never see the intermediate state at
  // runtime), but it's nicer not to push the half-initialized entry onto the
  // shadow stack.
  while (isa<StoreInst>(IP))
    ++IP;
  AtEntry.SetInsertPoint(IP->getParent(), IP);

  // Push the entry onto the shadow stack.
  Instruction *EntryNextPtr = CreateGEP(Context, AtEntry, ConcreteStackEntryTy,
                                        StackEntry, 0, 0, "gc_frame.next");
  Instruction *NewHeadVal = CreateGEP(Context, AtEntry, ConcreteStackEntryTy,
                                      StackEntry, 0, "gc_newhead");
  AtEntry.CreateStore(CurrentHead, EntryNextPtr);
  AtEntry.CreateStore(NewHeadVal, Head);

  // For each instruction that escapes...
  EscapeEnumerator EE(F, "gc_cleanup");
  while (IRBuilder<> *AtExit = EE.Next()) {
    // Pop the entry from the shadow stack. Don't reuse CurrentHead from
    // AtEntry, since that would make the value live for the entire function.
    Instruction *EntryNextPtr2 =
        CreateGEP(Context, *AtExit, ConcreteStackEntryTy, StackEntry, 0, 0,
                  "gc_frame.next");
    Value *SavedHead = AtExit->CreateLoad(EntryNextPtr2, "gc_savedhead");
    AtExit->CreateStore(SavedHead, Head);
  }

  // Delete the original allocas (which are no longer used) and the intrinsic
  // calls (which are no longer valid). Doing this last avoids invalidating
  // iterators.
  for (unsigned I = 0, E = Roots.size(); I != E; ++I) {
    Roots[I].first->eraseFromParent();
    Roots[I].second->eraseFromParent();
  }

  Roots.clear();
  return true;
}
