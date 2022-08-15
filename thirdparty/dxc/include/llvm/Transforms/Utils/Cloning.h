//===- Cloning.h - Clone various parts of LLVM programs ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines various functions that are used to clone chunks of LLVM
// code for various purposes.  This varies from copying whole modules into new
// modules, to cloning functions with different arguments, to inlining
// functions, to copying basic blocks to support loop unrolling or superblock
// formation, etc.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_CLONING_H
#define LLVM_TRANSFORMS_UTILS_CLONING_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

namespace llvm {

class Module;
class Function;
class Instruction;
class Pass;
class LPPassManager;
class BasicBlock;
class Value;
class CallInst;
class InvokeInst;
class ReturnInst;
class CallSite;
class Trace;
class CallGraph;
class DataLayout;
class Loop;
class LoopInfo;
class AllocaInst;
class AliasAnalysis;
class AssumptionCacheTracker;
class DominatorTree;

/// CloneModule - Return an exact copy of the specified module
///
Module *CloneModule(const Module *M);
Module *CloneModule(const Module *M, ValueToValueMapTy &VMap);

/// ClonedCodeInfo - This struct can be used to capture information about code
/// being cloned, while it is being cloned.
struct ClonedCodeInfo {
  /// ContainsCalls - This is set to true if the cloned code contains a normal
  /// call instruction.
  bool ContainsCalls;

  /// ContainsDynamicAllocas - This is set to true if the cloned code contains
  /// a 'dynamic' alloca.  Dynamic allocas are allocas that are either not in
  /// the entry block or they are in the entry block but are not a constant
  /// size.
  bool ContainsDynamicAllocas;

  ClonedCodeInfo() : ContainsCalls(false), ContainsDynamicAllocas(false) {}
};

/// CloneBasicBlock - Return a copy of the specified basic block, but without
/// embedding the block into a particular function.  The block returned is an
/// exact copy of the specified basic block, without any remapping having been
/// performed.  Because of this, this is only suitable for applications where
/// the basic block will be inserted into the same function that it was cloned
/// from (loop unrolling would use this, for example).
///
/// Also, note that this function makes a direct copy of the basic block, and
/// can thus produce illegal LLVM code.  In particular, it will copy any PHI
/// nodes from the original block, even though there are no predecessors for the
/// newly cloned block (thus, phi nodes will have to be updated).  Also, this
/// block will branch to the old successors of the original block: these
/// successors will have to have any PHI nodes updated to account for the new
/// incoming edges.
///
/// The correlation between instructions in the source and result basic blocks
/// is recorded in the VMap map.
///
/// If you have a particular suffix you'd like to use to add to any cloned
/// names, specify it as the optional third parameter.
///
/// If you would like the basic block to be auto-inserted into the end of a
/// function, you can specify it as the optional fourth parameter.
///
/// If you would like to collect additional information about the cloned
/// function, you can specify a ClonedCodeInfo object with the optional fifth
/// parameter.
///
BasicBlock *CloneBasicBlock(const BasicBlock *BB, ValueToValueMapTy &VMap,
                            const Twine &NameSuffix = "", Function *F = nullptr,
                            ClonedCodeInfo *CodeInfo = nullptr);

/// CloneFunction - Return a copy of the specified function, but without
/// embedding the function into another module.  Also, any references specified
/// in the VMap are changed to refer to their mapped value instead of the
/// original one.  If any of the arguments to the function are in the VMap,
/// the arguments are deleted from the resultant function.  The VMap is
/// updated to include mappings from all of the instructions and basicblocks in
/// the function from their old to new values.  The final argument captures
/// information about the cloned code if non-null.
///
/// If ModuleLevelChanges is false, VMap contains no non-identity GlobalValue
/// mappings, and debug info metadata will not be cloned.
///
Function *CloneFunction(const Function *F, ValueToValueMapTy &VMap,
                        bool ModuleLevelChanges,
                        ClonedCodeInfo *CodeInfo = nullptr);

/// Clone OldFunc into NewFunc, transforming the old arguments into references
/// to VMap values.  Note that if NewFunc already has basic blocks, the ones
/// cloned into it will be added to the end of the function.  This function
/// fills in a list of return instructions, and can optionally remap types
/// and/or append the specified suffix to all values cloned.
///
/// If ModuleLevelChanges is false, VMap contains no non-identity GlobalValue
/// mappings.
///
void CloneFunctionInto(Function *NewFunc, const Function *OldFunc,
                       ValueToValueMapTy &VMap, bool ModuleLevelChanges,
                       SmallVectorImpl<ReturnInst*> &Returns,
                       const char *NameSuffix = "",
                       ClonedCodeInfo *CodeInfo = nullptr,
                       ValueMapTypeRemapper *TypeMapper = nullptr,
                       ValueMaterializer *Materializer = nullptr);

/// A helper class used with CloneAndPruneIntoFromInst to change the default
/// behavior while instructions are being cloned.
class CloningDirector {
public:
  /// This enumeration describes the way CloneAndPruneIntoFromInst should
  /// proceed after the CloningDirector has examined an instruction.
  enum CloningAction {
    ///< Continue cloning the instruction (default behavior).
    CloneInstruction,
    ///< Skip this instruction but continue cloning the current basic block.
    SkipInstruction,
    ///< Skip this instruction and stop cloning the current basic block.
    StopCloningBB,
    ///< Don't clone the terminator but clone the current block's successors.
    CloneSuccessors
  };

  virtual ~CloningDirector() {}

  /// Subclasses must override this function to customize cloning behavior.
  virtual CloningAction handleInstruction(ValueToValueMapTy &VMap,
                                          const Instruction *Inst,
                                          BasicBlock *NewBB) = 0;

  virtual ValueMapTypeRemapper *getTypeRemapper() { return nullptr; }
  virtual ValueMaterializer *getValueMaterializer() { return nullptr; }
};

void CloneAndPruneIntoFromInst(Function *NewFunc, const Function *OldFunc,
                               const Instruction *StartingInst,
                               ValueToValueMapTy &VMap, bool ModuleLevelChanges,
                               SmallVectorImpl<ReturnInst*> &Returns,
                               const char *NameSuffix = "", 
                               ClonedCodeInfo *CodeInfo = nullptr,
                               CloningDirector *Director = nullptr);


/// CloneAndPruneFunctionInto - This works exactly like CloneFunctionInto,
/// except that it does some simple constant prop and DCE on the fly.  The
/// effect of this is to copy significantly less code in cases where (for
/// example) a function call with constant arguments is inlined, and those
/// constant arguments cause a significant amount of code in the callee to be
/// dead.  Since this doesn't produce an exactly copy of the input, it can't be
/// used for things like CloneFunction or CloneModule.
///
/// If ModuleLevelChanges is false, VMap contains no non-identity GlobalValue
/// mappings.
///
void CloneAndPruneFunctionInto(Function *NewFunc, const Function *OldFunc,
                               ValueToValueMapTy &VMap, bool ModuleLevelChanges,
                               SmallVectorImpl<ReturnInst*> &Returns,
                               const char *NameSuffix = "",
                               ClonedCodeInfo *CodeInfo = nullptr,
                               Instruction *TheCall = nullptr);

/// InlineFunctionInfo - This class captures the data input to the
/// InlineFunction call, and records the auxiliary results produced by it.
class InlineFunctionInfo {
public:
  explicit InlineFunctionInfo(CallGraph *cg = nullptr,
                              AliasAnalysis *AA = nullptr,
                              AssumptionCacheTracker *ACT = nullptr)
      : CG(cg), AA(AA), ACT(ACT) {}

  /// CG - If non-null, InlineFunction will update the callgraph to reflect the
  /// changes it makes.
  CallGraph *CG;
  AliasAnalysis *AA;
  AssumptionCacheTracker *ACT;

  /// StaticAllocas - InlineFunction fills this in with all static allocas that
  /// get copied into the caller.
  SmallVector<AllocaInst *, 4> StaticAllocas;

  /// InlinedCalls - InlineFunction fills this in with callsites that were
  /// inlined from the callee.  This is only filled in if CG is non-null.
  SmallVector<WeakVH, 8> InlinedCalls;

  void reset() {
    StaticAllocas.clear();
    InlinedCalls.clear();
  }
};

/// InlineFunction - This function inlines the called function into the basic
/// block of the caller.  This returns false if it is not possible to inline
/// this call.  The program is still in a well defined state if this occurs
/// though.
///
/// Note that this only does one level of inlining.  For example, if the
/// instruction 'call B' is inlined, and 'B' calls 'C', then the call to 'C' now
/// exists in the instruction stream.  Similarly this will inline a recursive
/// function by one level.
///
bool InlineFunction(CallInst *C, InlineFunctionInfo &IFI,
                    bool InsertLifetime = true);
bool InlineFunction(InvokeInst *II, InlineFunctionInfo &IFI,
                    bool InsertLifetime = true);
bool InlineFunction(CallSite CS, InlineFunctionInfo &IFI,
                    bool InsertLifetime = true);

/// \brief Clones a loop \p OrigLoop.  Returns the loop and the blocks in \p
/// Blocks.
///
/// Updates LoopInfo and DominatorTree assuming the loop is dominated by block
/// \p LoopDomBB.  Insert the new blocks before block specified in \p Before.
Loop *cloneLoopWithPreheader(BasicBlock *Before, BasicBlock *LoopDomBB,
                             Loop *OrigLoop, ValueToValueMapTy &VMap,
                             const Twine &NameSuffix, LoopInfo *LI,
                             DominatorTree *DT,
                             SmallVectorImpl<BasicBlock *> &Blocks);

/// \brief Remaps instructions in \p Blocks using the mapping in \p VMap.
void remapInstructionsInBlocks(const SmallVectorImpl<BasicBlock *> &Blocks,
                               ValueToValueMapTy &VMap);

} // End llvm namespace

#endif
