//===- CloneFunction.cpp - Clone a function into another function ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CloneFunctionInto interface, which is used as the
// low-level function cloner.  This is used by the CloneFunction and function
// inliner to do the dirty work of copying the body of a function around.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <map>
using namespace llvm;

/// See comments in Cloning.h.
BasicBlock *llvm::CloneBasicBlock(const BasicBlock *BB,
                                  ValueToValueMapTy &VMap,
                                  const Twine &NameSuffix, Function *F,
                                  ClonedCodeInfo *CodeInfo) {
  BasicBlock *NewBB = BasicBlock::Create(BB->getContext(), "", F);
  if (BB->hasName()) NewBB->setName(BB->getName()+NameSuffix);

  bool hasCalls = false, hasDynamicAllocas = false, hasStaticAllocas = false;
  
  // Loop over all instructions, and copy them over.
  for (BasicBlock::const_iterator II = BB->begin(), IE = BB->end();
       II != IE; ++II) {
    Instruction *NewInst = II->clone();
    if (II->hasName())
      NewInst->setName(II->getName()+NameSuffix);
    NewBB->getInstList().push_back(NewInst);
    VMap[II] = NewInst;                // Add instruction map to value.
    
    hasCalls |= (isa<CallInst>(II) && !isa<DbgInfoIntrinsic>(II));
    if (const AllocaInst *AI = dyn_cast<AllocaInst>(II)) {
      if (isa<ConstantInt>(AI->getArraySize()))
        hasStaticAllocas = true;
      else
        hasDynamicAllocas = true;
    }
  }
  
  if (CodeInfo) {
    CodeInfo->ContainsCalls          |= hasCalls;
    CodeInfo->ContainsDynamicAllocas |= hasDynamicAllocas;
    CodeInfo->ContainsDynamicAllocas |= hasStaticAllocas && 
                                        BB != &BB->getParent()->getEntryBlock();
  }
  return NewBB;
}

// Clone OldFunc into NewFunc, transforming the old arguments into references to
// VMap values.
//
void llvm::CloneFunctionInto(Function *NewFunc, const Function *OldFunc,
                             ValueToValueMapTy &VMap,
                             bool ModuleLevelChanges,
                             SmallVectorImpl<ReturnInst*> &Returns,
                             const char *NameSuffix, ClonedCodeInfo *CodeInfo,
                             ValueMapTypeRemapper *TypeMapper,
                             ValueMaterializer *Materializer) {
  assert(NameSuffix && "NameSuffix cannot be null!");

#ifndef NDEBUG
  for (Function::const_arg_iterator I = OldFunc->arg_begin(), 
       E = OldFunc->arg_end(); I != E; ++I)
    assert(VMap.count(I) && "No mapping from source argument specified!");
#endif

  // Copy all attributes other than those stored in the AttributeSet.  We need
  // to remap the parameter indices of the AttributeSet.
  AttributeSet NewAttrs = NewFunc->getAttributes();
  NewFunc->copyAttributesFrom(OldFunc);
  NewFunc->setAttributes(NewAttrs);

  AttributeSet OldAttrs = OldFunc->getAttributes();
  // Clone any argument attributes that are present in the VMap.
  for (const Argument &OldArg : OldFunc->args())
    if (Argument *NewArg = dyn_cast<Argument>(VMap[&OldArg])) {
      AttributeSet attrs =
          OldAttrs.getParamAttributes(OldArg.getArgNo() + 1);
      if (attrs.getNumSlots() > 0)
        NewArg->addAttr(attrs);
    }

  NewFunc->setAttributes(
      NewFunc->getAttributes()
          .addAttributes(NewFunc->getContext(), AttributeSet::ReturnIndex,
                         OldAttrs.getRetAttributes())
          .addAttributes(NewFunc->getContext(), AttributeSet::FunctionIndex,
                         OldAttrs.getFnAttributes()));

  // Loop over all of the basic blocks in the function, cloning them as
  // appropriate.  Note that we save BE this way in order to handle cloning of
  // recursive functions into themselves.
  //
  for (Function::const_iterator BI = OldFunc->begin(), BE = OldFunc->end();
       BI != BE; ++BI) {
    const BasicBlock &BB = *BI;

    // Create a new basic block and copy instructions into it!
    BasicBlock *CBB = CloneBasicBlock(&BB, VMap, NameSuffix, NewFunc, CodeInfo);

    // Add basic block mapping.
    VMap[&BB] = CBB;

    // It is only legal to clone a function if a block address within that
    // function is never referenced outside of the function.  Given that, we
    // want to map block addresses from the old function to block addresses in
    // the clone. (This is different from the generic ValueMapper
    // implementation, which generates an invalid blockaddress when
    // cloning a function.)
    if (BB.hasAddressTaken()) {
      Constant *OldBBAddr = BlockAddress::get(const_cast<Function*>(OldFunc),
                                              const_cast<BasicBlock*>(&BB));
      VMap[OldBBAddr] = BlockAddress::get(NewFunc, CBB);                                         
    }

    // Note return instructions for the caller.
    if (ReturnInst *RI = dyn_cast<ReturnInst>(CBB->getTerminator()))
      Returns.push_back(RI);
  }

  // Loop over all of the instructions in the function, fixing up operand
  // references as we go.  This uses VMap to do all the hard work.
  for (Function::iterator BB = cast<BasicBlock>(VMap[OldFunc->begin()]),
         BE = NewFunc->end(); BB != BE; ++BB)
    // Loop over all instructions, fixing each one as we find it...
    for (BasicBlock::iterator II = BB->begin(); II != BB->end(); ++II)
      RemapInstruction(II, VMap,
                       ModuleLevelChanges ? RF_None : RF_NoModuleLevelChanges,
                       TypeMapper, Materializer);
}

// Find the MDNode which corresponds to the subprogram data that described F.
static DISubprogram *FindSubprogram(const Function *F,
                                    DebugInfoFinder &Finder) {
  for (DISubprogram *Subprogram : Finder.subprograms()) {
    if (Subprogram->describes(F))
      return Subprogram;
  }
  return nullptr;
}

// Add an operand to an existing MDNode. The new operand will be added at the
// back of the operand list.
static void AddOperand(DICompileUnit *CU, DISubprogramArray SPs,
                       Metadata *NewSP) {
  SmallVector<Metadata *, 16> NewSPs;
  NewSPs.reserve(SPs.size() + 1);
  for (auto *SP : SPs)
    NewSPs.push_back(SP);
  NewSPs.push_back(NewSP);
  CU->replaceSubprograms(MDTuple::get(CU->getContext(), NewSPs));
}

// Clone the module-level debug info associated with OldFunc. The cloned data
// will point to NewFunc instead.
static void CloneDebugInfoMetadata(Function *NewFunc, const Function *OldFunc,
                            ValueToValueMapTy &VMap) {
  DebugInfoFinder Finder;
  Finder.processModule(*OldFunc->getParent());

  const DISubprogram *OldSubprogramMDNode = FindSubprogram(OldFunc, Finder);
  if (!OldSubprogramMDNode) return;

  // Ensure that OldFunc appears in the map.
  // (if it's already there it must point to NewFunc anyway)
  VMap[OldFunc] = NewFunc;
  auto *NewSubprogram =
      cast<DISubprogram>(MapMetadata(OldSubprogramMDNode, VMap));

  for (auto *CU : Finder.compile_units()) {
    auto Subprograms = CU->getSubprograms();
    // If the compile unit's function list contains the old function, it should
    // also contain the new one.
    for (auto *SP : Subprograms) {
      if (SP == OldSubprogramMDNode) {
        AddOperand(CU, Subprograms, NewSubprogram);
        break;
      }
    }
  }
}

/// Return a copy of the specified function, but without
/// embedding the function into another module.  Also, any references specified
/// in the VMap are changed to refer to their mapped value instead of the
/// original one.  If any of the arguments to the function are in the VMap,
/// the arguments are deleted from the resultant function.  The VMap is
/// updated to include mappings from all of the instructions and basicblocks in
/// the function from their old to new values.
///
Function *llvm::CloneFunction(const Function *F, ValueToValueMapTy &VMap,
                              bool ModuleLevelChanges,
                              ClonedCodeInfo *CodeInfo) {
  std::vector<Type*> ArgTypes;

  // The user might be deleting arguments to the function by specifying them in
  // the VMap.  If so, we need to not add the arguments to the arg ty vector
  //
  for (Function::const_arg_iterator I = F->arg_begin(), E = F->arg_end();
       I != E; ++I)
    if (VMap.count(I) == 0)  // Haven't mapped the argument to anything yet?
      ArgTypes.push_back(I->getType());

  // Create a new function type...
  FunctionType *FTy = FunctionType::get(F->getFunctionType()->getReturnType(),
                                    ArgTypes, F->getFunctionType()->isVarArg());

  // Create the new function...
  Function *NewF = Function::Create(FTy, F->getLinkage(), F->getName());

  // Loop over the arguments, copying the names of the mapped arguments over...
  Function::arg_iterator DestI = NewF->arg_begin();
  for (Function::const_arg_iterator I = F->arg_begin(), E = F->arg_end();
       I != E; ++I)
    if (VMap.count(I) == 0) {   // Is this argument preserved?
      DestI->setName(I->getName()); // Copy the name over...
      VMap[I] = DestI++;        // Add mapping to VMap
    }

  if (ModuleLevelChanges)
    CloneDebugInfoMetadata(NewF, F, VMap);

  SmallVector<ReturnInst*, 8> Returns;  // Ignore returns cloned.
  CloneFunctionInto(NewF, F, VMap, ModuleLevelChanges, Returns, "", CodeInfo);
  return NewF;
}



namespace {
  /// This is a private class used to implement CloneAndPruneFunctionInto.
  struct PruningFunctionCloner {
    Function *NewFunc;
    const Function *OldFunc;
    ValueToValueMapTy &VMap;
    bool ModuleLevelChanges;
    const char *NameSuffix;
    ClonedCodeInfo *CodeInfo;
    CloningDirector *Director;
    ValueMapTypeRemapper *TypeMapper;
    ValueMaterializer *Materializer;

  public:
    PruningFunctionCloner(Function *newFunc, const Function *oldFunc,
                          ValueToValueMapTy &valueMap, bool moduleLevelChanges,
                          const char *nameSuffix, ClonedCodeInfo *codeInfo,
                          CloningDirector *Director)
        : NewFunc(newFunc), OldFunc(oldFunc), VMap(valueMap),
          ModuleLevelChanges(moduleLevelChanges), NameSuffix(nameSuffix),
          CodeInfo(codeInfo), Director(Director) {
      // These are optional components.  The Director may return null.
      if (Director) {
        TypeMapper = Director->getTypeRemapper();
        Materializer = Director->getValueMaterializer();
      } else {
        TypeMapper = nullptr;
        Materializer = nullptr;
      }
    }

    /// The specified block is found to be reachable, clone it and
    /// anything that it can reach.
    void CloneBlock(const BasicBlock *BB, 
                    BasicBlock::const_iterator StartingInst,
                    std::vector<const BasicBlock*> &ToClone);
  };
}

/// The specified block is found to be reachable, clone it and
/// anything that it can reach.
void PruningFunctionCloner::CloneBlock(const BasicBlock *BB,
                                       BasicBlock::const_iterator StartingInst,
                                       std::vector<const BasicBlock*> &ToClone){
  WeakVH &BBEntry = VMap[BB];

  // Have we already cloned this block?
  if (BBEntry) return;
  
  // Nope, clone it now.
  BasicBlock *NewBB;
  BBEntry = NewBB = BasicBlock::Create(BB->getContext());
  if (BB->hasName()) NewBB->setName(BB->getName()+NameSuffix);

  // It is only legal to clone a function if a block address within that
  // function is never referenced outside of the function.  Given that, we
  // want to map block addresses from the old function to block addresses in
  // the clone. (This is different from the generic ValueMapper
  // implementation, which generates an invalid blockaddress when
  // cloning a function.)
  //
  // Note that we don't need to fix the mapping for unreachable blocks;
  // the default mapping there is safe.
  if (BB->hasAddressTaken()) {
    Constant *OldBBAddr = BlockAddress::get(const_cast<Function*>(OldFunc),
                                            const_cast<BasicBlock*>(BB));
    VMap[OldBBAddr] = BlockAddress::get(NewFunc, NewBB);
  }

  bool hasCalls = false, hasDynamicAllocas = false, hasStaticAllocas = false;

  // Loop over all instructions, and copy them over, DCE'ing as we go.  This
  // loop doesn't include the terminator.
  for (BasicBlock::const_iterator II = StartingInst, IE = --BB->end();
       II != IE; ++II) {
    // If the "Director" remaps the instruction, don't clone it.
    if (Director) {
      CloningDirector::CloningAction Action 
                              = Director->handleInstruction(VMap, II, NewBB);
      // If the cloning director says stop, we want to stop everything, not
      // just break out of the loop (which would cause the terminator to be
      // cloned).  The cloning director is responsible for inserting a proper
      // terminator into the new basic block in this case.
      if (Action == CloningDirector::StopCloningBB)
        return;
      // If the cloning director says skip, continue to the next instruction.
      // In this case, the cloning director is responsible for mapping the
      // skipped instruction to some value that is defined in the new
      // basic block.
      if (Action == CloningDirector::SkipInstruction)
        continue;
    }

    Instruction *NewInst = II->clone();

    // Eagerly remap operands to the newly cloned instruction, except for PHI
    // nodes for which we defer processing until we update the CFG.
    if (!isa<PHINode>(NewInst)) {
      RemapInstruction(NewInst, VMap,
                       ModuleLevelChanges ? RF_None : RF_NoModuleLevelChanges,
                       TypeMapper, Materializer);

      // If we can simplify this instruction to some other value, simply add
      // a mapping to that value rather than inserting a new instruction into
      // the basic block.
      if (Value *V =
              SimplifyInstruction(NewInst, BB->getModule()->getDataLayout())) {
        // On the off-chance that this simplifies to an instruction in the old
        // function, map it back into the new function.
        if (Value *MappedV = VMap.lookup(V))
          V = MappedV;

        VMap[II] = V;
        delete NewInst;
        continue;
      }
    }

    if (II->hasName())
      NewInst->setName(II->getName()+NameSuffix);
    VMap[II] = NewInst;                // Add instruction map to value.
    NewBB->getInstList().push_back(NewInst);
    hasCalls |= (isa<CallInst>(II) && !isa<DbgInfoIntrinsic>(II));
    if (const AllocaInst *AI = dyn_cast<AllocaInst>(II)) {
      if (isa<ConstantInt>(AI->getArraySize()))
        hasStaticAllocas = true;
      else
        hasDynamicAllocas = true;
    }
  }
  
  // Finally, clone over the terminator.
  const TerminatorInst *OldTI = BB->getTerminator();
  bool TerminatorDone = false;
  if (Director) {
    CloningDirector::CloningAction Action 
                           = Director->handleInstruction(VMap, OldTI, NewBB);
    // If the cloning director says stop, we want to stop everything, not
    // just break out of the loop (which would cause the terminator to be
    // cloned).  The cloning director is responsible for inserting a proper
    // terminator into the new basic block in this case.
    if (Action == CloningDirector::StopCloningBB)
      return;
    if (Action == CloningDirector::CloneSuccessors) {
      // If the director says to skip with a terminate instruction, we still
      // need to clone this block's successors.
      const TerminatorInst *TI = NewBB->getTerminator();
      for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i)
        ToClone.push_back(TI->getSuccessor(i));
      return;
    }
    assert(Action != CloningDirector::SkipInstruction && 
           "SkipInstruction is not valid for terminators.");
  }
  if (const BranchInst *BI = dyn_cast<BranchInst>(OldTI)) {
    if (BI->isConditional()) {
      // If the condition was a known constant in the callee...
      ConstantInt *Cond = dyn_cast<ConstantInt>(BI->getCondition());
      // Or is a known constant in the caller...
      if (!Cond) {
        Value *V = VMap[BI->getCondition()];
        Cond = dyn_cast_or_null<ConstantInt>(V);
      }

      // Constant fold to uncond branch!
      if (Cond) {
        BasicBlock *Dest = BI->getSuccessor(!Cond->getZExtValue());
        VMap[OldTI] = BranchInst::Create(Dest, NewBB);
        ToClone.push_back(Dest);
        TerminatorDone = true;
      }
    }
  } else if (const SwitchInst *SI = dyn_cast<SwitchInst>(OldTI)) {
    // If switching on a value known constant in the caller.
    ConstantInt *Cond = dyn_cast<ConstantInt>(SI->getCondition());
    if (!Cond) { // Or known constant after constant prop in the callee...
      Value *V = VMap[SI->getCondition()];
      Cond = dyn_cast_or_null<ConstantInt>(V);
    }
    if (Cond) {     // Constant fold to uncond branch!
      SwitchInst::ConstCaseIt Case = SI->findCaseValue(Cond);
      BasicBlock *Dest = const_cast<BasicBlock*>(Case.getCaseSuccessor());
      VMap[OldTI] = BranchInst::Create(Dest, NewBB);
      ToClone.push_back(Dest);
      TerminatorDone = true;
    }
  }
  
  if (!TerminatorDone) {
    Instruction *NewInst = OldTI->clone();
    if (OldTI->hasName())
      NewInst->setName(OldTI->getName()+NameSuffix);
    NewBB->getInstList().push_back(NewInst);
    VMap[OldTI] = NewInst;             // Add instruction map to value.
    
    // Recursively clone any reachable successor blocks.
    const TerminatorInst *TI = BB->getTerminator();
    for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i)
      ToClone.push_back(TI->getSuccessor(i));
  }
  
  if (CodeInfo) {
    CodeInfo->ContainsCalls          |= hasCalls;
    CodeInfo->ContainsDynamicAllocas |= hasDynamicAllocas;
    CodeInfo->ContainsDynamicAllocas |= hasStaticAllocas && 
      BB != &BB->getParent()->front();
  }
}

/// This works like CloneAndPruneFunctionInto, except that it does not clone the
/// entire function. Instead it starts at an instruction provided by the caller
/// and copies (and prunes) only the code reachable from that instruction.
void llvm::CloneAndPruneIntoFromInst(Function *NewFunc, const Function *OldFunc,
                                     const Instruction *StartingInst,
                                     ValueToValueMapTy &VMap,
                                     bool ModuleLevelChanges,
                                     SmallVectorImpl<ReturnInst *> &Returns,
                                     const char *NameSuffix, 
                                     ClonedCodeInfo *CodeInfo,
                                     CloningDirector *Director) {
  assert(NameSuffix && "NameSuffix cannot be null!");

  ValueMapTypeRemapper *TypeMapper = nullptr;
  ValueMaterializer *Materializer = nullptr;

  if (Director) {
    TypeMapper = Director->getTypeRemapper();
    Materializer = Director->getValueMaterializer();
  }

#ifndef NDEBUG
  // If the cloning starts at the begining of the function, verify that
  // the function arguments are mapped.
  if (!StartingInst)
    for (Function::const_arg_iterator II = OldFunc->arg_begin(),
         E = OldFunc->arg_end(); II != E; ++II)
      assert(VMap.count(II) && "No mapping from source argument specified!");
#endif

  PruningFunctionCloner PFC(NewFunc, OldFunc, VMap, ModuleLevelChanges,
                            NameSuffix, CodeInfo, Director);
  const BasicBlock *StartingBB;
  if (StartingInst)
    StartingBB = StartingInst->getParent();
  else {
    StartingBB = &OldFunc->getEntryBlock();
    StartingInst = StartingBB->begin();
  }

  // Clone the entry block, and anything recursively reachable from it.
  std::vector<const BasicBlock*> CloneWorklist;
  PFC.CloneBlock(StartingBB, StartingInst, CloneWorklist);
  while (!CloneWorklist.empty()) {
    const BasicBlock *BB = CloneWorklist.back();
    CloneWorklist.pop_back();
    PFC.CloneBlock(BB, BB->begin(), CloneWorklist);
  }
  
  // Loop over all of the basic blocks in the old function.  If the block was
  // reachable, we have cloned it and the old block is now in the value map:
  // insert it into the new function in the right order.  If not, ignore it.
  //
  // Defer PHI resolution until rest of function is resolved.
  SmallVector<const PHINode*, 16> PHIToResolve;
  for (Function::const_iterator BI = OldFunc->begin(), BE = OldFunc->end();
       BI != BE; ++BI) {
    Value *V = VMap[BI];
    BasicBlock *NewBB = cast_or_null<BasicBlock>(V);
    if (!NewBB) continue;  // Dead block.

    // Add the new block to the new function.
    NewFunc->getBasicBlockList().push_back(NewBB);

    // Handle PHI nodes specially, as we have to remove references to dead
    // blocks.
    for (BasicBlock::const_iterator I = BI->begin(), E = BI->end(); I != E; ++I) {
      // PHI nodes may have been remapped to non-PHI nodes by the caller or
      // during the cloning process.
      if (const PHINode *PN = dyn_cast<PHINode>(I)) {
        if (isa<PHINode>(VMap[PN]))
          PHIToResolve.push_back(PN);
        else
          break;
      } else {
        break;
      }
    }

    // Finally, remap the terminator instructions, as those can't be remapped
    // until all BBs are mapped.
    RemapInstruction(NewBB->getTerminator(), VMap,
                     ModuleLevelChanges ? RF_None : RF_NoModuleLevelChanges,
                     TypeMapper, Materializer);
  }
  
  // Defer PHI resolution until rest of function is resolved, PHI resolution
  // requires the CFG to be up-to-date.
  for (unsigned phino = 0, e = PHIToResolve.size(); phino != e; ) {
    const PHINode *OPN = PHIToResolve[phino];
    unsigned NumPreds = OPN->getNumIncomingValues();
    const BasicBlock *OldBB = OPN->getParent();
    BasicBlock *NewBB = cast<BasicBlock>(VMap[OldBB]);

    // Map operands for blocks that are live and remove operands for blocks
    // that are dead.
    for (; phino != PHIToResolve.size() &&
         PHIToResolve[phino]->getParent() == OldBB; ++phino) {
      OPN = PHIToResolve[phino];
      PHINode *PN = cast<PHINode>(VMap[OPN]);
      for (unsigned pred = 0, e = NumPreds; pred != e; ++pred) {
        Value *V = VMap[PN->getIncomingBlock(pred)];
        if (BasicBlock *MappedBlock = cast_or_null<BasicBlock>(V)) {
          Value *InVal = MapValue(PN->getIncomingValue(pred),
                                  VMap, 
                        ModuleLevelChanges ? RF_None : RF_NoModuleLevelChanges);
          assert(InVal && "Unknown input value?");
          PN->setIncomingValue(pred, InVal);
          PN->setIncomingBlock(pred, MappedBlock);
        } else {
          PN->removeIncomingValue(pred, false);
          --pred, --e;  // Revisit the next entry.
        }
      } 
    }
    
    // The loop above has removed PHI entries for those blocks that are dead
    // and has updated others.  However, if a block is live (i.e. copied over)
    // but its terminator has been changed to not go to this block, then our
    // phi nodes will have invalid entries.  Update the PHI nodes in this
    // case.
    PHINode *PN = cast<PHINode>(NewBB->begin());
    NumPreds = std::distance(pred_begin(NewBB), pred_end(NewBB));
    if (NumPreds != PN->getNumIncomingValues()) {
      assert(NumPreds < PN->getNumIncomingValues());
      // Count how many times each predecessor comes to this block.
      std::map<BasicBlock*, unsigned> PredCount;
      for (pred_iterator PI = pred_begin(NewBB), E = pred_end(NewBB);
           PI != E; ++PI)
        --PredCount[*PI];
      
      // Figure out how many entries to remove from each PHI.
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
        ++PredCount[PN->getIncomingBlock(i)];
      
      // At this point, the excess predecessor entries are positive in the
      // map.  Loop over all of the PHIs and remove excess predecessor
      // entries.
      BasicBlock::iterator I = NewBB->begin();
      for (; (PN = dyn_cast<PHINode>(I)); ++I) {
        for (std::map<BasicBlock*, unsigned>::iterator PCI =PredCount.begin(),
             E = PredCount.end(); PCI != E; ++PCI) {
          BasicBlock *Pred     = PCI->first;
          for (unsigned NumToRemove = PCI->second; NumToRemove; --NumToRemove)
            PN->removeIncomingValue(Pred, false);
        }
      }
    }
    
    // If the loops above have made these phi nodes have 0 or 1 operand,
    // replace them with undef or the input value.  We must do this for
    // correctness, because 0-operand phis are not valid.
    PN = cast<PHINode>(NewBB->begin());
    if (PN->getNumIncomingValues() == 0) {
      BasicBlock::iterator I = NewBB->begin();
      BasicBlock::const_iterator OldI = OldBB->begin();
      while ((PN = dyn_cast<PHINode>(I++))) {
        Value *NV = UndefValue::get(PN->getType());
        PN->replaceAllUsesWith(NV);
        assert(VMap[OldI] == PN && "VMap mismatch");
        VMap[OldI] = NV;
        PN->eraseFromParent();
        ++OldI;
      }
    }
  }

  // Make a second pass over the PHINodes now that all of them have been
  // remapped into the new function, simplifying the PHINode and performing any
  // recursive simplifications exposed. This will transparently update the
  // WeakVH in the VMap. Notably, we rely on that so that if we coalesce
  // two PHINodes, the iteration over the old PHIs remains valid, and the
  // mapping will just map us to the new node (which may not even be a PHI
  // node).
  for (unsigned Idx = 0, Size = PHIToResolve.size(); Idx != Size; ++Idx)
    if (PHINode *PN = dyn_cast<PHINode>(VMap[PHIToResolve[Idx]]))
      recursivelySimplifyInstruction(PN);

  // Now that the inlined function body has been fully constructed, go through
  // and zap unconditional fall-through branches. This happens all the time when
  // specializing code: code specialization turns conditional branches into
  // uncond branches, and this code folds them.
  Function::iterator Begin = cast<BasicBlock>(VMap[StartingBB]);
  Function::iterator I = Begin;
  while (I != NewFunc->end()) {
    // Check if this block has become dead during inlining or other
    // simplifications. Note that the first block will appear dead, as it has
    // not yet been wired up properly.
    if (I != Begin && (pred_begin(I) == pred_end(I) ||
                       I->getSinglePredecessor() == I)) {
      BasicBlock *DeadBB = I++;
      DeleteDeadBlock(DeadBB);
      continue;
    }

    // We need to simplify conditional branches and switches with a constant
    // operand. We try to prune these out when cloning, but if the
    // simplification required looking through PHI nodes, those are only
    // available after forming the full basic block. That may leave some here,
    // and we still want to prune the dead code as early as possible.
    ConstantFoldTerminator(I);

    BranchInst *BI = dyn_cast<BranchInst>(I->getTerminator());
    if (!BI || BI->isConditional()) { ++I; continue; }
    
    BasicBlock *Dest = BI->getSuccessor(0);
    if (!Dest->getSinglePredecessor()) {
      ++I; continue;
    }

    // We shouldn't be able to get single-entry PHI nodes here, as instsimplify
    // above should have zapped all of them..
    assert(!isa<PHINode>(Dest->begin()));

    // We know all single-entry PHI nodes in the inlined function have been
    // removed, so we just need to splice the blocks.
    BI->eraseFromParent();
    
    // Make all PHI nodes that referred to Dest now refer to I as their source.
    Dest->replaceAllUsesWith(I);

    // Move all the instructions in the succ to the pred.
    I->getInstList().splice(I->end(), Dest->getInstList());
    
    // Remove the dest block.
    Dest->eraseFromParent();
    
    // Do not increment I, iteratively merge all things this block branches to.
  }

  // Make a final pass over the basic blocks from the old function to gather
  // any return instructions which survived folding. We have to do this here
  // because we can iteratively remove and merge returns above.
  for (Function::iterator I = cast<BasicBlock>(VMap[StartingBB]),
                          E = NewFunc->end();
       I != E; ++I)
    if (ReturnInst *RI = dyn_cast<ReturnInst>(I->getTerminator()))
      Returns.push_back(RI);
}


/// This works exactly like CloneFunctionInto,
/// except that it does some simple constant prop and DCE on the fly.  The
/// effect of this is to copy significantly less code in cases where (for
/// example) a function call with constant arguments is inlined, and those
/// constant arguments cause a significant amount of code in the callee to be
/// dead.  Since this doesn't produce an exact copy of the input, it can't be
/// used for things like CloneFunction or CloneModule.
void llvm::CloneAndPruneFunctionInto(Function *NewFunc, const Function *OldFunc,
                                     ValueToValueMapTy &VMap,
                                     bool ModuleLevelChanges,
                                     SmallVectorImpl<ReturnInst*> &Returns,
                                     const char *NameSuffix, 
                                     ClonedCodeInfo *CodeInfo,
                                     Instruction *TheCall) {
  CloneAndPruneIntoFromInst(NewFunc, OldFunc, OldFunc->front().begin(), VMap,
                            ModuleLevelChanges, Returns, NameSuffix, CodeInfo,
                            nullptr);
}

/// \brief Remaps instructions in \p Blocks using the mapping in \p VMap.
void llvm::remapInstructionsInBlocks(
    const SmallVectorImpl<BasicBlock *> &Blocks, ValueToValueMapTy &VMap) {
  // Rewrite the code to refer to itself.
  for (auto *BB : Blocks)
    for (auto &Inst : *BB)
      RemapInstruction(&Inst, VMap,
                       RF_NoModuleLevelChanges | RF_IgnoreMissingEntries);
}

/// \brief Clones a loop \p OrigLoop.  Returns the loop and the blocks in \p
/// Blocks.
///
/// Updates LoopInfo and DominatorTree assuming the loop is dominated by block
/// \p LoopDomBB.  Insert the new blocks before block specified in \p Before.
Loop *llvm::cloneLoopWithPreheader(BasicBlock *Before, BasicBlock *LoopDomBB,
                                   Loop *OrigLoop, ValueToValueMapTy &VMap,
                                   const Twine &NameSuffix, LoopInfo *LI,
                                   DominatorTree *DT,
                                   SmallVectorImpl<BasicBlock *> &Blocks) {
  Function *F = OrigLoop->getHeader()->getParent();
  Loop *ParentLoop = OrigLoop->getParentLoop();

  Loop *NewLoop = new Loop();
  if (ParentLoop)
    ParentLoop->addChildLoop(NewLoop);
  else
    LI->addTopLevelLoop(NewLoop);

  BasicBlock *OrigPH = OrigLoop->getLoopPreheader();
  assert(OrigPH && "No preheader");
  BasicBlock *NewPH = CloneBasicBlock(OrigPH, VMap, NameSuffix, F);
  // To rename the loop PHIs.
  VMap[OrigPH] = NewPH;
  Blocks.push_back(NewPH);

  // Update LoopInfo.
  if (ParentLoop)
    ParentLoop->addBasicBlockToLoop(NewPH, *LI);

  // Update DominatorTree.
  DT->addNewBlock(NewPH, LoopDomBB);

  for (BasicBlock *BB : OrigLoop->getBlocks()) {
    BasicBlock *NewBB = CloneBasicBlock(BB, VMap, NameSuffix, F);
    VMap[BB] = NewBB;

    // Update LoopInfo.
    NewLoop->addBasicBlockToLoop(NewBB, *LI);

    // Update DominatorTree.
    BasicBlock *IDomBB = DT->getNode(BB)->getIDom()->getBlock();
    DT->addNewBlock(NewBB, cast<BasicBlock>(VMap[IDomBB]));

    Blocks.push_back(NewBB);
  }

  // Move them physically from the end of the block list.
  F->getBasicBlockList().splice(Before, F->getBasicBlockList(), NewPH);
  F->getBasicBlockList().splice(Before, F->getBasicBlockList(),
                                NewLoop->getHeader(), F->end());

  return NewLoop;
}
