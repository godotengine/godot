//===-- DeadArgumentElimination.cpp - Eliminate dead arguments ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass deletes dead arguments from internal functions.  Dead argument
// elimination removes arguments which are directly dead, as well as arguments
// only passed into function calls as dead arguments of other functions.  This
// pass also deletes dead return values in a similar way.
//
// This pass is often useful as a cleanup pass to run after aggressive
// interprocedural passes, which add possibly-dead arguments or return values.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <set>
#include <tuple>
using namespace llvm;

#define DEBUG_TYPE "deadargelim"

STATISTIC(NumArgumentsEliminated, "Number of unread args removed");
STATISTIC(NumRetValsEliminated  , "Number of unused return values removed");
STATISTIC(NumArgumentsReplacedWithUndef, 
          "Number of unread args replaced with undef");
namespace {
  /// DAE - The dead argument elimination pass.
  ///
  class DAE : public ModulePass {
  public:

    /// Struct that represents (part of) either a return value or a function
    /// argument.  Used so that arguments and return values can be used
    /// interchangeably.
    struct RetOrArg {
      RetOrArg(const Function *F, unsigned Idx, bool IsArg) : F(F), Idx(Idx),
               IsArg(IsArg) {}
      const Function *F;
      unsigned Idx;
      bool IsArg;

      /// Make RetOrArg comparable, so we can put it into a map.
      bool operator<(const RetOrArg &O) const {
        return std::tie(F, Idx, IsArg) < std::tie(O.F, O.Idx, O.IsArg);
      }

      /// Make RetOrArg comparable, so we can easily iterate the multimap.
      bool operator==(const RetOrArg &O) const {
        return F == O.F && Idx == O.Idx && IsArg == O.IsArg;
      }

      std::string getDescription() const {
        return (Twine(IsArg ? "Argument #" : "Return value #") + utostr(Idx) +
                " of function " + F->getName()).str();
      }
    };

    /// Liveness enum - During our initial pass over the program, we determine
    /// that things are either alive or maybe alive. We don't mark anything
    /// explicitly dead (even if we know they are), since anything not alive
    /// with no registered uses (in Uses) will never be marked alive and will
    /// thus become dead in the end.
    enum Liveness { Live, MaybeLive };

    /// Convenience wrapper
    RetOrArg CreateRet(const Function *F, unsigned Idx) {
      return RetOrArg(F, Idx, false);
    }
    /// Convenience wrapper
    RetOrArg CreateArg(const Function *F, unsigned Idx) {
      return RetOrArg(F, Idx, true);
    }

    typedef std::multimap<RetOrArg, RetOrArg> UseMap;
    /// This maps a return value or argument to any MaybeLive return values or
    /// arguments it uses. This allows the MaybeLive values to be marked live
    /// when any of its users is marked live.
    /// For example (indices are left out for clarity):
    ///  - Uses[ret F] = ret G
    ///    This means that F calls G, and F returns the value returned by G.
    ///  - Uses[arg F] = ret G
    ///    This means that some function calls G and passes its result as an
    ///    argument to F.
    ///  - Uses[ret F] = arg F
    ///    This means that F returns one of its own arguments.
    ///  - Uses[arg F] = arg G
    ///    This means that G calls F and passes one of its own (G's) arguments
    ///    directly to F.
    UseMap Uses;

    typedef std::set<RetOrArg> LiveSet;
    typedef std::set<const Function*> LiveFuncSet;

    /// This set contains all values that have been determined to be live.
    LiveSet LiveValues;
    /// This set contains all values that are cannot be changed in any way.
    LiveFuncSet LiveFunctions;

    typedef SmallVector<RetOrArg, 5> UseVector;

    // Map each LLVM function to corresponding metadata with debug info. If
    // the function is replaced with another one, we should patch the pointer
    // to LLVM function in metadata.
    // As the code generation for module is finished (and DIBuilder is
    // finalized) we assume that subprogram descriptors won't be changed, and
    // they are stored in map for short duration anyway.
    DenseMap<const Function *, DISubprogram *> FunctionDIs;

  protected:
    // DAH uses this to specify a different ID.
    explicit DAE(char &ID) : ModulePass(ID) {}

  public:
    static char ID; // Pass identification, replacement for typeid
    DAE() : ModulePass(ID) {
      initializeDAEPass(*PassRegistry::getPassRegistry());
    }

    bool runOnModule(Module &M) override;

    virtual bool ShouldHackArguments() const { return false; }

  private:
    Liveness MarkIfNotLive(RetOrArg Use, UseVector &MaybeLiveUses);
    Liveness SurveyUse(const Use *U, UseVector &MaybeLiveUses,
                       unsigned RetValNum = -1U);
    Liveness SurveyUses(const Value *V, UseVector &MaybeLiveUses);

    void SurveyFunction(const Function &F);
    void MarkValue(const RetOrArg &RA, Liveness L,
                   const UseVector &MaybeLiveUses);
    void MarkLive(const RetOrArg &RA);
    void MarkLive(const Function &F);
    void PropagateLiveness(const RetOrArg &RA);
    bool RemoveDeadStuffFromFunction(Function *F);
    bool DeleteDeadVarargs(Function &Fn);
    bool RemoveDeadArgumentsFromCallers(Function &Fn);
  };
}


char DAE::ID = 0;
INITIALIZE_PASS(DAE, "deadargelim", "Dead Argument Elimination", false, false)

namespace {
  /// DAH - DeadArgumentHacking pass - Same as dead argument elimination, but
  /// deletes arguments to functions which are external.  This is only for use
  /// by bugpoint.
  struct DAH : public DAE {
    static char ID;
    DAH() : DAE(ID) {}

    bool ShouldHackArguments() const override { return true; }
  };
}

char DAH::ID = 0;
INITIALIZE_PASS(DAH, "deadarghaX0r", 
                "Dead Argument Hacking (BUGPOINT USE ONLY; DO NOT USE)",
                false, false)

/// createDeadArgEliminationPass - This pass removes arguments from functions
/// which are not used by the body of the function.
///
ModulePass *llvm::createDeadArgEliminationPass() { return new DAE(); }
ModulePass *llvm::createDeadArgHackingPass() { return new DAH(); }

/// DeleteDeadVarargs - If this is an function that takes a ... list, and if
/// llvm.vastart is never called, the varargs list is dead for the function.
bool DAE::DeleteDeadVarargs(Function &Fn) {
  assert(Fn.getFunctionType()->isVarArg() && "Function isn't varargs!");
  if (Fn.isDeclaration() || !Fn.hasLocalLinkage()) return false;

  // Ensure that the function is only directly called.
  if (Fn.hasAddressTaken())
    return false;

  // Okay, we know we can transform this function if safe.  Scan its body
  // looking for calls marked musttail or calls to llvm.vastart.
  for (Function::iterator BB = Fn.begin(), E = Fn.end(); BB != E; ++BB) {
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
      CallInst *CI = dyn_cast<CallInst>(I);
      if (!CI)
        continue;
      if (CI->isMustTailCall())
        return false;
      if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(CI)) {
        if (II->getIntrinsicID() == Intrinsic::vastart)
          return false;
      }
    }
  }

  // If we get here, there are no calls to llvm.vastart in the function body,
  // remove the "..." and adjust all the calls.

  // Start by computing a new prototype for the function, which is the same as
  // the old function, but doesn't have isVarArg set.
  FunctionType *FTy = Fn.getFunctionType();

  std::vector<Type*> Params(FTy->param_begin(), FTy->param_end());
  FunctionType *NFTy = FunctionType::get(FTy->getReturnType(),
                                                Params, false);
  unsigned NumArgs = Params.size();

  // Create the new function body and insert it into the module...
  Function *NF = Function::Create(NFTy, Fn.getLinkage());
  NF->copyAttributesFrom(&Fn);
  Fn.getParent()->getFunctionList().insert(&Fn, NF);
  NF->takeName(&Fn);

  // Loop over all of the callers of the function, transforming the call sites
  // to pass in a smaller number of arguments into the new function.
  //
  std::vector<Value*> Args;
  for (Value::user_iterator I = Fn.user_begin(), E = Fn.user_end(); I != E; ) {
    CallSite CS(*I++);
    if (!CS)
      continue;
    Instruction *Call = CS.getInstruction();

    // Pass all the same arguments.
    Args.assign(CS.arg_begin(), CS.arg_begin() + NumArgs);

    // Drop any attributes that were on the vararg arguments.
    AttributeSet PAL = CS.getAttributes();
    if (!PAL.isEmpty() && PAL.getSlotIndex(PAL.getNumSlots() - 1) > NumArgs) {
      SmallVector<AttributeSet, 8> AttributesVec;
      for (unsigned i = 0; PAL.getSlotIndex(i) <= NumArgs; ++i)
        AttributesVec.push_back(PAL.getSlotAttributes(i));
      if (PAL.hasAttributes(AttributeSet::FunctionIndex))
        AttributesVec.push_back(AttributeSet::get(Fn.getContext(),
                                                  PAL.getFnAttributes()));
      PAL = AttributeSet::get(Fn.getContext(), AttributesVec);
    }

    Instruction *New;
    if (InvokeInst *II = dyn_cast<InvokeInst>(Call)) {
      New = InvokeInst::Create(NF, II->getNormalDest(), II->getUnwindDest(),
                               Args, "", Call);
      cast<InvokeInst>(New)->setCallingConv(CS.getCallingConv());
      cast<InvokeInst>(New)->setAttributes(PAL);
    } else {
      New = CallInst::Create(NF, Args, "", Call);
      cast<CallInst>(New)->setCallingConv(CS.getCallingConv());
      cast<CallInst>(New)->setAttributes(PAL);
      if (cast<CallInst>(Call)->isTailCall())
        cast<CallInst>(New)->setTailCall();
    }
    New->setDebugLoc(Call->getDebugLoc());

    Args.clear();

    if (!Call->use_empty())
      Call->replaceAllUsesWith(New);

    New->takeName(Call);

    // Finally, remove the old call from the program, reducing the use-count of
    // F.
    Call->eraseFromParent();
  }

  // Since we have now created the new function, splice the body of the old
  // function right into the new function, leaving the old rotting hulk of the
  // function empty.
  NF->getBasicBlockList().splice(NF->begin(), Fn.getBasicBlockList());

  // Loop over the argument list, transferring uses of the old arguments over to
  // the new arguments, also transferring over the names as well.  While we're at
  // it, remove the dead arguments from the DeadArguments list.
  //
  for (Function::arg_iterator I = Fn.arg_begin(), E = Fn.arg_end(),
       I2 = NF->arg_begin(); I != E; ++I, ++I2) {
    // Move the name and users over to the new version.
    I->replaceAllUsesWith(I2);
    I2->takeName(I);
  }

  // Patch the pointer to LLVM function in debug info descriptor.
  auto DI = FunctionDIs.find(&Fn);
  if (DI != FunctionDIs.end()) {
    DISubprogram *SP = DI->second;
    SP->replaceFunction(NF);
    // Ensure the map is updated so it can be reused on non-varargs argument
    // eliminations of the same function.
    FunctionDIs.erase(DI);
    FunctionDIs[NF] = SP;
  }

  // Fix up any BlockAddresses that refer to the function.
  Fn.replaceAllUsesWith(ConstantExpr::getBitCast(NF, Fn.getType()));
  // Delete the bitcast that we just created, so that NF does not
  // appear to be address-taken.
  NF->removeDeadConstantUsers();
  // Finally, nuke the old function.
  Fn.eraseFromParent();
  return true;
}

/// RemoveDeadArgumentsFromCallers - Checks if the given function has any 
/// arguments that are unused, and changes the caller parameters to be undefined
/// instead.
bool DAE::RemoveDeadArgumentsFromCallers(Function &Fn)
{
  // We cannot change the arguments if this TU does not define the function or
  // if the linker may choose a function body from another TU, even if the
  // nominal linkage indicates that other copies of the function have the same
  // semantics. In the below example, the dead load from %p may not have been
  // eliminated from the linker-chosen copy of f, so replacing %p with undef
  // in callers may introduce undefined behavior.
  //
  // define linkonce_odr void @f(i32* %p) {
  //   %v = load i32 %p
  //   ret void
  // }
  if (!Fn.isStrongDefinitionForLinker())
    return false;

  // Functions with local linkage should already have been handled, except the
  // fragile (variadic) ones which we can improve here.
  if (Fn.hasLocalLinkage() && !Fn.getFunctionType()->isVarArg())
    return false;

  if (Fn.use_empty())
    return false;

  SmallVector<unsigned, 8> UnusedArgs;
  for (Function::arg_iterator I = Fn.arg_begin(), E = Fn.arg_end(); 
       I != E; ++I) {
    Argument *Arg = I;

    if (Arg->use_empty() && !Arg->hasByValOrInAllocaAttr())
      UnusedArgs.push_back(Arg->getArgNo());
  }

  if (UnusedArgs.empty())
    return false;

  bool Changed = false;

  for (Use &U : Fn.uses()) {
    CallSite CS(U.getUser());
    if (!CS || !CS.isCallee(&U))
      continue;

    // Now go through all unused args and replace them with "undef".
    for (unsigned I = 0, E = UnusedArgs.size(); I != E; ++I) {
      unsigned ArgNo = UnusedArgs[I];

      Value *Arg = CS.getArgument(ArgNo);
      CS.setArgument(ArgNo, UndefValue::get(Arg->getType()));
      ++NumArgumentsReplacedWithUndef;
      Changed = true;
    }
  }

  return Changed;
}

/// Convenience function that returns the number of return values. It returns 0
/// for void functions and 1 for functions not returning a struct. It returns
/// the number of struct elements for functions returning a struct.
static unsigned NumRetVals(const Function *F) {
  Type *RetTy = F->getReturnType();
  if (RetTy->isVoidTy())
    return 0;
  else if (StructType *STy = dyn_cast<StructType>(RetTy))
    return STy->getNumElements();
  else if (ArrayType *ATy = dyn_cast<ArrayType>(RetTy))
    return ATy->getNumElements();
  else
    return 1;
}

/// Returns the sub-type a function will return at a given Idx. Should
/// correspond to the result type of an ExtractValue instruction executed with
/// just that one Idx (i.e. only top-level structure is considered).
static Type *getRetComponentType(const Function *F, unsigned Idx) {
  Type *RetTy = F->getReturnType();
  assert(!RetTy->isVoidTy() && "void type has no subtype");

  if (StructType *STy = dyn_cast<StructType>(RetTy))
    return STy->getElementType(Idx);
  else if (ArrayType *ATy = dyn_cast<ArrayType>(RetTy))
    return ATy->getElementType();
  else
    return RetTy;
}

/// MarkIfNotLive - This checks Use for liveness in LiveValues. If Use is not
/// live, it adds Use to the MaybeLiveUses argument. Returns the determined
/// liveness of Use.
DAE::Liveness DAE::MarkIfNotLive(RetOrArg Use, UseVector &MaybeLiveUses) {
  // We're live if our use or its Function is already marked as live.
  if (LiveFunctions.count(Use.F) || LiveValues.count(Use))
    return Live;

  // We're maybe live otherwise, but remember that we must become live if
  // Use becomes live.
  MaybeLiveUses.push_back(Use);
  return MaybeLive;
}


/// SurveyUse - This looks at a single use of an argument or return value
/// and determines if it should be alive or not. Adds this use to MaybeLiveUses
/// if it causes the used value to become MaybeLive.
///
/// RetValNum is the return value number to use when this use is used in a
/// return instruction. This is used in the recursion, you should always leave
/// it at 0.
DAE::Liveness DAE::SurveyUse(const Use *U,
                             UseVector &MaybeLiveUses, unsigned RetValNum) {
    const User *V = U->getUser();
    if (const ReturnInst *RI = dyn_cast<ReturnInst>(V)) {
      // The value is returned from a function. It's only live when the
      // function's return value is live. We use RetValNum here, for the case
      // that U is really a use of an insertvalue instruction that uses the
      // original Use.
      const Function *F = RI->getParent()->getParent();
      if (RetValNum != -1U) {
        RetOrArg Use = CreateRet(F, RetValNum);
        // We might be live, depending on the liveness of Use.
        return MarkIfNotLive(Use, MaybeLiveUses);
      } else {
        DAE::Liveness Result = MaybeLive;
        for (unsigned i = 0; i < NumRetVals(F); ++i) {
          RetOrArg Use = CreateRet(F, i);
          // We might be live, depending on the liveness of Use. If any
          // sub-value is live, then the entire value is considered live. This
          // is a conservative choice, and better tracking is possible.
          DAE::Liveness SubResult = MarkIfNotLive(Use, MaybeLiveUses);
          if (Result != Live)
            Result = SubResult;
        }
        return Result;
      }
    }
    if (const InsertValueInst *IV = dyn_cast<InsertValueInst>(V)) {
      if (U->getOperandNo() != InsertValueInst::getAggregateOperandIndex()
          && IV->hasIndices())
        // The use we are examining is inserted into an aggregate. Our liveness
        // depends on all uses of that aggregate, but if it is used as a return
        // value, only index at which we were inserted counts.
        RetValNum = *IV->idx_begin();

      // Note that if we are used as the aggregate operand to the insertvalue,
      // we don't change RetValNum, but do survey all our uses.

      Liveness Result = MaybeLive;
      for (const Use &UU : IV->uses()) {
        Result = SurveyUse(&UU, MaybeLiveUses, RetValNum);
        if (Result == Live)
          break;
      }
      return Result;
    }

    if (auto CS = ImmutableCallSite(V)) {
      const Function *F = CS.getCalledFunction();
      if (F) {
        // Used in a direct call.

        // Find the argument number. We know for sure that this use is an
        // argument, since if it was the function argument this would be an
        // indirect call and the we know can't be looking at a value of the
        // label type (for the invoke instruction).
        unsigned ArgNo = CS.getArgumentNo(U);

        if (ArgNo >= F->getFunctionType()->getNumParams())
          // The value is passed in through a vararg! Must be live.
          return Live;

        assert(CS.getArgument(ArgNo)
               == CS->getOperand(U->getOperandNo())
               && "Argument is not where we expected it");

        // Value passed to a normal call. It's only live when the corresponding
        // argument to the called function turns out live.
        RetOrArg Use = CreateArg(F, ArgNo);
        return MarkIfNotLive(Use, MaybeLiveUses);
      }
    }
    // Used in any other way? Value must be live.
    return Live;
}

/// SurveyUses - This looks at all the uses of the given value
/// Returns the Liveness deduced from the uses of this value.
///
/// Adds all uses that cause the result to be MaybeLive to MaybeLiveRetUses. If
/// the result is Live, MaybeLiveUses might be modified but its content should
/// be ignored (since it might not be complete).
DAE::Liveness DAE::SurveyUses(const Value *V, UseVector &MaybeLiveUses) {
  // Assume it's dead (which will only hold if there are no uses at all..).
  Liveness Result = MaybeLive;
  // Check each use.
  for (const Use &U : V->uses()) {
    Result = SurveyUse(&U, MaybeLiveUses);
    if (Result == Live)
      break;
  }
  return Result;
}

// SurveyFunction - This performs the initial survey of the specified function,
// checking out whether or not it uses any of its incoming arguments or whether
// any callers use the return value.  This fills in the LiveValues set and Uses
// map.
//
// We consider arguments of non-internal functions to be intrinsically alive as
// well as arguments to functions which have their "address taken".
//
void DAE::SurveyFunction(const Function &F) {
  // Functions with inalloca parameters are expecting args in a particular
  // register and memory layout.
  if (F.getAttributes().hasAttrSomewhere(Attribute::InAlloca)) {
    MarkLive(F);
    return;
  }

  unsigned RetCount = NumRetVals(&F);
  // Assume all return values are dead
  typedef SmallVector<Liveness, 5> RetVals;
  RetVals RetValLiveness(RetCount, MaybeLive);

  typedef SmallVector<UseVector, 5> RetUses;
  // These vectors map each return value to the uses that make it MaybeLive, so
  // we can add those to the Uses map if the return value really turns out to be
  // MaybeLive. Initialized to a list of RetCount empty lists.
  RetUses MaybeLiveRetUses(RetCount);

  for (Function::const_iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
    if (const ReturnInst *RI = dyn_cast<ReturnInst>(BB->getTerminator()))
      if (RI->getNumOperands() != 0 && RI->getOperand(0)->getType()
          != F.getFunctionType()->getReturnType()) {
        // We don't support old style multiple return values.
        MarkLive(F);
        return;
      }

  if (!F.hasLocalLinkage() && (!ShouldHackArguments() || F.isIntrinsic())) {
    MarkLive(F);
    return;
  }

  DEBUG(dbgs() << "DAE - Inspecting callers for fn: " << F.getName() << "\n");
  // Keep track of the number of live retvals, so we can skip checks once all
  // of them turn out to be live.
  unsigned NumLiveRetVals = 0;
  // Loop all uses of the function.
  for (const Use &U : F.uses()) {
    // If the function is PASSED IN as an argument, its address has been
    // taken.
    ImmutableCallSite CS(U.getUser());
    if (!CS || !CS.isCallee(&U)) {
      MarkLive(F);
      return;
    }

    // If this use is anything other than a call site, the function is alive.
    const Instruction *TheCall = CS.getInstruction();
    if (!TheCall) {   // Not a direct call site?
      MarkLive(F);
      return;
    }

    // If we end up here, we are looking at a direct call to our function.

    // Now, check how our return value(s) is/are used in this caller. Don't
    // bother checking return values if all of them are live already.
    if (NumLiveRetVals == RetCount)
      continue;

    // Check all uses of the return value.
    for (const Use &U : TheCall->uses()) {
      if (ExtractValueInst *Ext = dyn_cast<ExtractValueInst>(U.getUser())) {
        // This use uses a part of our return value, survey the uses of
        // that part and store the results for this index only.
        unsigned Idx = *Ext->idx_begin();
        if (RetValLiveness[Idx] != Live) {
          RetValLiveness[Idx] = SurveyUses(Ext, MaybeLiveRetUses[Idx]);
          if (RetValLiveness[Idx] == Live)
            NumLiveRetVals++;
        }
      } else {
        // Used by something else than extractvalue. Survey, but assume that the
        // result applies to all sub-values.
        UseVector MaybeLiveAggregateUses;
        if (SurveyUse(&U, MaybeLiveAggregateUses) == Live) {
          NumLiveRetVals = RetCount;
          RetValLiveness.assign(RetCount, Live);
          break;
        } else {
          for (unsigned i = 0; i != RetCount; ++i) {
            if (RetValLiveness[i] != Live)
              MaybeLiveRetUses[i].append(MaybeLiveAggregateUses.begin(),
                                         MaybeLiveAggregateUses.end());
          }
        }
      }
    }
  }

  // Now we've inspected all callers, record the liveness of our return values.
  for (unsigned i = 0; i != RetCount; ++i)
    MarkValue(CreateRet(&F, i), RetValLiveness[i], MaybeLiveRetUses[i]);

  DEBUG(dbgs() << "DAE - Inspecting args for fn: " << F.getName() << "\n");

  // Now, check all of our arguments.
  unsigned i = 0;
  UseVector MaybeLiveArgUses;
  for (Function::const_arg_iterator AI = F.arg_begin(),
       E = F.arg_end(); AI != E; ++AI, ++i) {
    Liveness Result;
    if (F.getFunctionType()->isVarArg()) {
      // Variadic functions will already have a va_arg function expanded inside
      // them, making them potentially very sensitive to ABI changes resulting
      // from removing arguments entirely, so don't. For example AArch64 handles
      // register and stack HFAs very differently, and this is reflected in the
      // IR which has already been generated.
      Result = Live;
    } else {
      // See what the effect of this use is (recording any uses that cause
      // MaybeLive in MaybeLiveArgUses). 
      Result = SurveyUses(AI, MaybeLiveArgUses);
    }

    // Mark the result.
    MarkValue(CreateArg(&F, i), Result, MaybeLiveArgUses);
    // Clear the vector again for the next iteration.
    MaybeLiveArgUses.clear();
  }
}

/// MarkValue - This function marks the liveness of RA depending on L. If L is
/// MaybeLive, it also takes all uses in MaybeLiveUses and records them in Uses,
/// such that RA will be marked live if any use in MaybeLiveUses gets marked
/// live later on.
void DAE::MarkValue(const RetOrArg &RA, Liveness L,
                    const UseVector &MaybeLiveUses) {
  switch (L) {
    case Live: MarkLive(RA); break;
    case MaybeLive:
    {
      // Note any uses of this value, so this return value can be
      // marked live whenever one of the uses becomes live.
      for (UseVector::const_iterator UI = MaybeLiveUses.begin(),
           UE = MaybeLiveUses.end(); UI != UE; ++UI)
        Uses.insert(std::make_pair(*UI, RA));
      break;
    }
  }
}

/// MarkLive - Mark the given Function as alive, meaning that it cannot be
/// changed in any way. Additionally,
/// mark any values that are used as this function's parameters or by its return
/// values (according to Uses) live as well.
void DAE::MarkLive(const Function &F) {
  DEBUG(dbgs() << "DAE - Intrinsically live fn: " << F.getName() << "\n");
  // Mark the function as live.
  LiveFunctions.insert(&F);
  // Mark all arguments as live.
  for (unsigned i = 0, e = F.arg_size(); i != e; ++i)
    PropagateLiveness(CreateArg(&F, i));
  // Mark all return values as live.
  for (unsigned i = 0, e = NumRetVals(&F); i != e; ++i)
    PropagateLiveness(CreateRet(&F, i));
}

/// MarkLive - Mark the given return value or argument as live. Additionally,
/// mark any values that are used by this value (according to Uses) live as
/// well.
void DAE::MarkLive(const RetOrArg &RA) {
  if (LiveFunctions.count(RA.F))
    return; // Function was already marked Live.

  if (!LiveValues.insert(RA).second)
    return; // We were already marked Live.

  DEBUG(dbgs() << "DAE - Marking " << RA.getDescription() << " live\n");
  PropagateLiveness(RA);
}

/// PropagateLiveness - Given that RA is a live value, propagate it's liveness
/// to any other values it uses (according to Uses).
void DAE::PropagateLiveness(const RetOrArg &RA) {
  // We don't use upper_bound (or equal_range) here, because our recursive call
  // to ourselves is likely to cause the upper_bound (which is the first value
  // not belonging to RA) to become erased and the iterator invalidated.
  UseMap::iterator Begin = Uses.lower_bound(RA);
  UseMap::iterator E = Uses.end();
  UseMap::iterator I;
  for (I = Begin; I != E && I->first == RA; ++I)
    MarkLive(I->second);

  // Erase RA from the Uses map (from the lower bound to wherever we ended up
  // after the loop).
  Uses.erase(Begin, I);
}

// RemoveDeadStuffFromFunction - Remove any arguments and return values from F
// that are not in LiveValues. Transform the function and all of the callees of
// the function to not have these arguments and return values.
//
bool DAE::RemoveDeadStuffFromFunction(Function *F) {
  // Don't modify fully live functions
  if (LiveFunctions.count(F))
    return false;

  // Start by computing a new prototype for the function, which is the same as
  // the old function, but has fewer arguments and a different return type.
  FunctionType *FTy = F->getFunctionType();
  std::vector<Type*> Params;

  // Keep track of if we have a live 'returned' argument
  bool HasLiveReturnedArg = false;

  // Set up to build a new list of parameter attributes.
  SmallVector<AttributeSet, 8> AttributesVec;
  const AttributeSet &PAL = F->getAttributes();

  // Remember which arguments are still alive.
  SmallVector<bool, 10> ArgAlive(FTy->getNumParams(), false);
  // Construct the new parameter list from non-dead arguments. Also construct
  // a new set of parameter attributes to correspond. Skip the first parameter
  // attribute, since that belongs to the return value.
  unsigned i = 0;
  for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end();
       I != E; ++I, ++i) {
    RetOrArg Arg = CreateArg(F, i);
    if (LiveValues.erase(Arg)) {
      Params.push_back(I->getType());
      ArgAlive[i] = true;

      // Get the original parameter attributes (skipping the first one, that is
      // for the return value.
      if (PAL.hasAttributes(i + 1)) {
        AttrBuilder B(PAL, i + 1);
        if (B.contains(Attribute::Returned))
          HasLiveReturnedArg = true;
        AttributesVec.
          push_back(AttributeSet::get(F->getContext(), Params.size(), B));
      }
    } else {
      ++NumArgumentsEliminated;
      DEBUG(dbgs() << "DAE - Removing argument " << i << " (" << I->getName()
            << ") from " << F->getName() << "\n");
    }
  }

  // Find out the new return value.
  Type *RetTy = FTy->getReturnType();
  Type *NRetTy = nullptr;
  unsigned RetCount = NumRetVals(F);

  // -1 means unused, other numbers are the new index
  SmallVector<int, 5> NewRetIdxs(RetCount, -1);
  std::vector<Type*> RetTypes;

  // If there is a function with a live 'returned' argument but a dead return
  // value, then there are two possible actions:
  // 1) Eliminate the return value and take off the 'returned' attribute on the
  //    argument.
  // 2) Retain the 'returned' attribute and treat the return value (but not the
  //    entire function) as live so that it is not eliminated.
  // 
  // It's not clear in the general case which option is more profitable because,
  // even in the absence of explicit uses of the return value, code generation
  // is free to use the 'returned' attribute to do things like eliding
  // save/restores of registers across calls. Whether or not this happens is
  // target and ABI-specific as well as depending on the amount of register
  // pressure, so there's no good way for an IR-level pass to figure this out.
  //
  // Fortunately, the only places where 'returned' is currently generated by
  // the FE are places where 'returned' is basically free and almost always a
  // performance win, so the second option can just be used always for now.
  //
  // This should be revisited if 'returned' is ever applied more liberally.
  if (RetTy->isVoidTy() || HasLiveReturnedArg) {
    NRetTy = RetTy;
  } else {
    // Look at each of the original return values individually.
    for (unsigned i = 0; i != RetCount; ++i) {
      RetOrArg Ret = CreateRet(F, i);
      if (LiveValues.erase(Ret)) {
        RetTypes.push_back(getRetComponentType(F, i));
        NewRetIdxs[i] = RetTypes.size() - 1;
      } else {
        ++NumRetValsEliminated;
        DEBUG(dbgs() << "DAE - Removing return value " << i << " from "
              << F->getName() << "\n");
      }
    }
    if (RetTypes.size() > 1) {
      // More than one return type? Reduce it down to size.
      if (StructType *STy = dyn_cast<StructType>(RetTy)) {
        // Make the new struct packed if we used to return a packed struct
        // already.
        NRetTy = StructType::get(STy->getContext(), RetTypes, STy->isPacked());
      } else {
        assert(isa<ArrayType>(RetTy) && "unexpected multi-value return");
        NRetTy = ArrayType::get(RetTypes[0], RetTypes.size());
      }
    } else if (RetTypes.size() == 1)
      // One return type? Just a simple value then, but only if we didn't use to
      // return a struct with that simple value before.
      NRetTy = RetTypes.front();
    else if (RetTypes.size() == 0)
      // No return types? Make it void, but only if we didn't use to return {}.
      NRetTy = Type::getVoidTy(F->getContext());
  }

  assert(NRetTy && "No new return type found?");

  // The existing function return attributes.
  AttributeSet RAttrs = PAL.getRetAttributes();

  // Remove any incompatible attributes, but only if we removed all return
  // values. Otherwise, ensure that we don't have any conflicting attributes
  // here. Currently, this should not be possible, but special handling might be
  // required when new return value attributes are added.
  if (NRetTy->isVoidTy())
    RAttrs = RAttrs.removeAttributes(NRetTy->getContext(),
                                     AttributeSet::ReturnIndex,
                                     AttributeFuncs::typeIncompatible(NRetTy));
  else
    assert(!AttrBuilder(RAttrs, AttributeSet::ReturnIndex).
             overlaps(AttributeFuncs::typeIncompatible(NRetTy)) &&
           "Return attributes no longer compatible?");

  if (RAttrs.hasAttributes(AttributeSet::ReturnIndex))
    AttributesVec.push_back(AttributeSet::get(NRetTy->getContext(), RAttrs));

  if (PAL.hasAttributes(AttributeSet::FunctionIndex))
    AttributesVec.push_back(AttributeSet::get(F->getContext(),
                                              PAL.getFnAttributes()));

  // Reconstruct the AttributesList based on the vector we constructed.
  AttributeSet NewPAL = AttributeSet::get(F->getContext(), AttributesVec);

  // Create the new function type based on the recomputed parameters.
  FunctionType *NFTy = FunctionType::get(NRetTy, Params, FTy->isVarArg());

  // No change?
  if (NFTy == FTy)
    return false;

  // Create the new function body and insert it into the module...
  Function *NF = Function::Create(NFTy, F->getLinkage());
  NF->copyAttributesFrom(F);
  NF->setAttributes(NewPAL);
  // Insert the new function before the old function, so we won't be processing
  // it again.
  F->getParent()->getFunctionList().insert(F, NF);
  NF->takeName(F);

  // Loop over all of the callers of the function, transforming the call sites
  // to pass in a smaller number of arguments into the new function.
  //
  std::vector<Value*> Args;
  while (!F->use_empty()) {
    CallSite CS(F->user_back());
    Instruction *Call = CS.getInstruction();

    AttributesVec.clear();
    const AttributeSet &CallPAL = CS.getAttributes();

    // The call return attributes.
    AttributeSet RAttrs = CallPAL.getRetAttributes();

    // Adjust in case the function was changed to return void.
    RAttrs = RAttrs.removeAttributes(NRetTy->getContext(),
                                     AttributeSet::ReturnIndex,
                        AttributeFuncs::typeIncompatible(NF->getReturnType()));
    if (RAttrs.hasAttributes(AttributeSet::ReturnIndex))
      AttributesVec.push_back(AttributeSet::get(NF->getContext(), RAttrs));

    // Declare these outside of the loops, so we can reuse them for the second
    // loop, which loops the varargs.
    CallSite::arg_iterator I = CS.arg_begin();
    unsigned i = 0;
    // Loop over those operands, corresponding to the normal arguments to the
    // original function, and add those that are still alive.
    for (unsigned e = FTy->getNumParams(); i != e; ++I, ++i)
      if (ArgAlive[i]) {
        Args.push_back(*I);
        // Get original parameter attributes, but skip return attributes.
        if (CallPAL.hasAttributes(i + 1)) {
          AttrBuilder B(CallPAL, i + 1);
          // If the return type has changed, then get rid of 'returned' on the
          // call site. The alternative is to make all 'returned' attributes on
          // call sites keep the return value alive just like 'returned'
          // attributes on function declaration but it's less clearly a win
          // and this is not an expected case anyway
          if (NRetTy != RetTy && B.contains(Attribute::Returned))
            B.removeAttribute(Attribute::Returned);
          AttributesVec.
            push_back(AttributeSet::get(F->getContext(), Args.size(), B));
        }
      }

    // Push any varargs arguments on the list. Don't forget their attributes.
    for (CallSite::arg_iterator E = CS.arg_end(); I != E; ++I, ++i) {
      Args.push_back(*I);
      if (CallPAL.hasAttributes(i + 1)) {
        AttrBuilder B(CallPAL, i + 1);
        AttributesVec.
          push_back(AttributeSet::get(F->getContext(), Args.size(), B));
      }
    }

    if (CallPAL.hasAttributes(AttributeSet::FunctionIndex))
      AttributesVec.push_back(AttributeSet::get(Call->getContext(),
                                                CallPAL.getFnAttributes()));

    // Reconstruct the AttributesList based on the vector we constructed.
    AttributeSet NewCallPAL = AttributeSet::get(F->getContext(), AttributesVec);

    Instruction *New;
    if (InvokeInst *II = dyn_cast<InvokeInst>(Call)) {
      New = InvokeInst::Create(NF, II->getNormalDest(), II->getUnwindDest(),
                               Args, "", Call);
      cast<InvokeInst>(New)->setCallingConv(CS.getCallingConv());
      cast<InvokeInst>(New)->setAttributes(NewCallPAL);
    } else {
      New = CallInst::Create(NF, Args, "", Call);
      cast<CallInst>(New)->setCallingConv(CS.getCallingConv());
      cast<CallInst>(New)->setAttributes(NewCallPAL);
      if (cast<CallInst>(Call)->isTailCall())
        cast<CallInst>(New)->setTailCall();
    }
    New->setDebugLoc(Call->getDebugLoc());

    Args.clear();

    if (!Call->use_empty()) {
      if (New->getType() == Call->getType()) {
        // Return type not changed? Just replace users then.
        Call->replaceAllUsesWith(New);
        New->takeName(Call);
      } else if (New->getType()->isVoidTy()) {
        // Our return value has uses, but they will get removed later on.
        // Replace by null for now.
        if (!Call->getType()->isX86_MMXTy())
          Call->replaceAllUsesWith(Constant::getNullValue(Call->getType()));
      } else {
        assert((RetTy->isStructTy() || RetTy->isArrayTy()) &&
               "Return type changed, but not into a void. The old return type"
               " must have been a struct or an array!");
        Instruction *InsertPt = Call;
        if (InvokeInst *II = dyn_cast<InvokeInst>(Call)) {
          BasicBlock::iterator IP = II->getNormalDest()->begin();
          while (isa<PHINode>(IP)) ++IP;
          InsertPt = IP;
        }

        // We used to return a struct or array. Instead of doing smart stuff
        // with all the uses, we will just rebuild it using extract/insertvalue
        // chaining and let instcombine clean that up.
        //
        // Start out building up our return value from undef
        Value *RetVal = UndefValue::get(RetTy);
        for (unsigned i = 0; i != RetCount; ++i)
          if (NewRetIdxs[i] != -1) {
            Value *V;
            if (RetTypes.size() > 1)
              // We are still returning a struct, so extract the value from our
              // return value
              V = ExtractValueInst::Create(New, NewRetIdxs[i], "newret",
                                           InsertPt);
            else
              // We are now returning a single element, so just insert that
              V = New;
            // Insert the value at the old position
            RetVal = InsertValueInst::Create(RetVal, V, i, "oldret", InsertPt);
          }
        // Now, replace all uses of the old call instruction with the return
        // struct we built
        Call->replaceAllUsesWith(RetVal);
        New->takeName(Call);
      }
    }

    // Finally, remove the old call from the program, reducing the use-count of
    // F.
    Call->eraseFromParent();
  }

  // Since we have now created the new function, splice the body of the old
  // function right into the new function, leaving the old rotting hulk of the
  // function empty.
  NF->getBasicBlockList().splice(NF->begin(), F->getBasicBlockList());

  // Loop over the argument list, transferring uses of the old arguments over to
  // the new arguments, also transferring over the names as well.
  i = 0;
  for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end(),
       I2 = NF->arg_begin(); I != E; ++I, ++i)
    if (ArgAlive[i]) {
      // If this is a live argument, move the name and users over to the new
      // version.
      I->replaceAllUsesWith(I2);
      I2->takeName(I);
      ++I2;
    } else {
      // If this argument is dead, replace any uses of it with null constants
      // (these are guaranteed to become unused later on).
      if (!I->getType()->isX86_MMXTy())
        I->replaceAllUsesWith(Constant::getNullValue(I->getType()));
    }

  // If we change the return value of the function we must rewrite any return
  // instructions.  Check this now.
  if (F->getReturnType() != NF->getReturnType())
    for (Function::iterator BB = NF->begin(), E = NF->end(); BB != E; ++BB)
      if (ReturnInst *RI = dyn_cast<ReturnInst>(BB->getTerminator())) {
        Value *RetVal;

        if (NFTy->getReturnType()->isVoidTy()) {
          RetVal = nullptr;
        } else {
          assert(RetTy->isStructTy() || RetTy->isArrayTy());
          // The original return value was a struct or array, insert
          // extractvalue/insertvalue chains to extract only the values we need
          // to return and insert them into our new result.
          // This does generate messy code, but we'll let it to instcombine to
          // clean that up.
          Value *OldRet = RI->getOperand(0);
          // Start out building up our return value from undef
          RetVal = UndefValue::get(NRetTy);
          for (unsigned i = 0; i != RetCount; ++i)
            if (NewRetIdxs[i] != -1) {
              ExtractValueInst *EV = ExtractValueInst::Create(OldRet, i,
                                                              "oldret", RI);
              if (RetTypes.size() > 1) {
                // We're still returning a struct, so reinsert the value into
                // our new return value at the new index

                RetVal = InsertValueInst::Create(RetVal, EV, NewRetIdxs[i],
                                                 "newret", RI);
              } else {
                // We are now only returning a simple value, so just return the
                // extracted value.
                RetVal = EV;
              }
            }
        }
        // Replace the return instruction with one returning the new return
        // value (possibly 0 if we became void).
        ReturnInst::Create(F->getContext(), RetVal, RI);
        BB->getInstList().erase(RI);
      }

  // Patch the pointer to LLVM function in debug info descriptor.
  auto DI = FunctionDIs.find(F);
  if (DI != FunctionDIs.end())
    DI->second->replaceFunction(NF);

  // Now that the old function is dead, delete it.
  F->eraseFromParent();

  return true;
}

bool DAE::runOnModule(Module &M) {
  bool Changed = false;

  // Collect debug info descriptors for functions.
  FunctionDIs = makeSubprogramMap(M);

  // First pass: Do a simple check to see if any functions can have their "..."
  // removed.  We can do this if they never call va_start.  This loop cannot be
  // fused with the next loop, because deleting a function invalidates
  // information computed while surveying other functions.
  DEBUG(dbgs() << "DAE - Deleting dead varargs\n");
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ) {
    Function &F = *I++;
    if (F.getFunctionType()->isVarArg())
      Changed |= DeleteDeadVarargs(F);
  }

  // Second phase:loop through the module, determining which arguments are live.
  // We assume all arguments are dead unless proven otherwise (allowing us to
  // determine that dead arguments passed into recursive functions are dead).
  //
  DEBUG(dbgs() << "DAE - Determining liveness\n");
  for (auto &F : M)
    SurveyFunction(F);

  // Now, remove all dead arguments and return values from each function in
  // turn.
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ) {
    // Increment now, because the function will probably get removed (ie.
    // replaced by a new one).
    Function *F = I++;
    Changed |= RemoveDeadStuffFromFunction(F);
  }

  // Finally, look for any unused parameters in functions with non-local
  // linkage and replace the passed in parameters with undef.
  for (auto &F : M)
    Changed |= RemoveDeadArgumentsFromCallers(F);

  return Changed;
}
