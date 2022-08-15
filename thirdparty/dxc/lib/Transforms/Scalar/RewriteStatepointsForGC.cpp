//===- RewriteStatepointsForGC.cpp - Make GC relocations explicit ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Rewrite an existing set of gc.statepoints such that they make potential
// relocations performed by the garbage collector explicit in the IR.
//
//===----------------------------------------------------------------------===//

#include "llvm/Pass.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Statepoint.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"

#define DEBUG_TYPE "rewrite-statepoints-for-gc"

using namespace llvm;

// Print tracing output
static cl::opt<bool> TraceLSP("trace-rewrite-statepoints", cl::Hidden,
                              cl::init(false));

// Print the liveset found at the insert location
static cl::opt<bool> PrintLiveSet("spp-print-liveset", cl::Hidden,
                                  cl::init(false));
static cl::opt<bool> PrintLiveSetSize("spp-print-liveset-size", cl::Hidden,
                                      cl::init(false));
// Print out the base pointers for debugging
static cl::opt<bool> PrintBasePointers("spp-print-base-pointers", cl::Hidden,
                                       cl::init(false));

// Cost threshold measuring when it is profitable to rematerialize value instead
// of relocating it
static cl::opt<unsigned>
RematerializationThreshold("spp-rematerialization-threshold", cl::Hidden,
                           cl::init(6));

#ifdef XDEBUG
static bool ClobberNonLive = true;
#else
static bool ClobberNonLive = false;
#endif
static cl::opt<bool, true> ClobberNonLiveOverride("rs4gc-clobber-non-live",
                                                  cl::location(ClobberNonLive),
                                                  cl::Hidden);

namespace {
struct RewriteStatepointsForGC : public ModulePass {
  static char ID; // Pass identification, replacement for typeid

  RewriteStatepointsForGC() : ModulePass(ID) {
    initializeRewriteStatepointsForGCPass(*PassRegistry::getPassRegistry());
  }
  bool runOnFunction(Function &F);
  bool runOnModule(Module &M) override {
    bool Changed = false;
    for (Function &F : M)
      Changed |= runOnFunction(F);

    if (Changed) {
      // stripDereferenceabilityInfo asserts that shouldRewriteStatepointsIn
      // returns true for at least one function in the module.  Since at least
      // one function changed, we know that the precondition is satisfied.
      stripDereferenceabilityInfo(M);
    }

    return Changed;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    // We add and rewrite a bunch of instructions, but don't really do much
    // else.  We could in theory preserve a lot more analyses here.
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
  }

  /// The IR fed into RewriteStatepointsForGC may have had attributes implying
  /// dereferenceability that are no longer valid/correct after
  /// RewriteStatepointsForGC has run.  This is because semantically, after
  /// RewriteStatepointsForGC runs, all calls to gc.statepoint "free" the entire
  /// heap.  stripDereferenceabilityInfo (conservatively) restores correctness
  /// by erasing all attributes in the module that externally imply
  /// dereferenceability.
  ///
  void stripDereferenceabilityInfo(Module &M);

  // Helpers for stripDereferenceabilityInfo
  void stripDereferenceabilityInfoFromBody(Function &F);
  void stripDereferenceabilityInfoFromPrototype(Function &F);
};
} // namespace

char RewriteStatepointsForGC::ID = 0;

ModulePass *llvm::createRewriteStatepointsForGCPass() {
  return new RewriteStatepointsForGC();
}

INITIALIZE_PASS_BEGIN(RewriteStatepointsForGC, "rewrite-statepoints-for-gc",
                      "Make relocations explicit at statepoints", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(RewriteStatepointsForGC, "rewrite-statepoints-for-gc",
                    "Make relocations explicit at statepoints", false, false)

namespace {
struct GCPtrLivenessData {
  /// Values defined in this block.
  DenseMap<BasicBlock *, DenseSet<Value *>> KillSet;
  /// Values used in this block (and thus live); does not included values
  /// killed within this block.
  DenseMap<BasicBlock *, DenseSet<Value *>> LiveSet;

  /// Values live into this basic block (i.e. used by any
  /// instruction in this basic block or ones reachable from here)
  DenseMap<BasicBlock *, DenseSet<Value *>> LiveIn;

  /// Values live out of this basic block (i.e. live into
  /// any successor block)
  DenseMap<BasicBlock *, DenseSet<Value *>> LiveOut;
};

// The type of the internal cache used inside the findBasePointers family
// of functions.  From the callers perspective, this is an opaque type and
// should not be inspected.
//
// In the actual implementation this caches two relations:
// - The base relation itself (i.e. this pointer is based on that one)
// - The base defining value relation (i.e. before base_phi insertion)
// Generally, after the execution of a full findBasePointer call, only the
// base relation will remain.  Internally, we add a mixture of the two
// types, then update all the second type to the first type
typedef DenseMap<Value *, Value *> DefiningValueMapTy;
typedef DenseSet<llvm::Value *> StatepointLiveSetTy;
typedef DenseMap<Instruction *, Value *> RematerializedValueMapTy;

struct PartiallyConstructedSafepointRecord {
  /// The set of values known to be live accross this safepoint
  StatepointLiveSetTy liveset;

  /// Mapping from live pointers to a base-defining-value
  DenseMap<llvm::Value *, llvm::Value *> PointerToBase;

  /// The *new* gc.statepoint instruction itself.  This produces the token
  /// that normal path gc.relocates and the gc.result are tied to.
  Instruction *StatepointToken;

  /// Instruction to which exceptional gc relocates are attached
  /// Makes it easier to iterate through them during relocationViaAlloca.
  Instruction *UnwindToken;

  /// Record live values we are rematerialized instead of relocating.
  /// They are not included into 'liveset' field.
  /// Maps rematerialized copy to it's original value.
  RematerializedValueMapTy RematerializedValues;
};
}

/// Compute the live-in set for every basic block in the function
static void computeLiveInValues(DominatorTree &DT, Function &F,
                                GCPtrLivenessData &Data);

/// Given results from the dataflow liveness computation, find the set of live
/// Values at a particular instruction.
static void findLiveSetAtInst(Instruction *inst, GCPtrLivenessData &Data,
                              StatepointLiveSetTy &out);

// TODO: Once we can get to the GCStrategy, this becomes
// Optional<bool> isGCManagedPointer(const Value *V) const override {

static bool isGCPointerType(const Type *T) {
  if (const PointerType *PT = dyn_cast<PointerType>(T))
    // For the sake of this example GC, we arbitrarily pick addrspace(1) as our
    // GC managed heap.  We know that a pointer into this heap needs to be
    // updated and that no other pointer does.
    return (1 == PT->getAddressSpace());
  return false;
}

// Return true if this type is one which a) is a gc pointer or contains a GC
// pointer and b) is of a type this code expects to encounter as a live value.
// (The insertion code will assert that a type which matches (a) and not (b)
// is not encountered.)
static bool isHandledGCPointerType(Type *T) {
  // We fully support gc pointers
  if (isGCPointerType(T))
    return true;
  // We partially support vectors of gc pointers. The code will assert if it
  // can't handle something.
  if (auto VT = dyn_cast<VectorType>(T))
    if (isGCPointerType(VT->getElementType()))
      return true;
  return false;
}

#ifndef NDEBUG
/// Returns true if this type contains a gc pointer whether we know how to
/// handle that type or not.
static bool containsGCPtrType(Type *Ty) {
  if (isGCPointerType(Ty))
    return true;
  if (VectorType *VT = dyn_cast<VectorType>(Ty))
    return isGCPointerType(VT->getScalarType());
  if (ArrayType *AT = dyn_cast<ArrayType>(Ty))
    return containsGCPtrType(AT->getElementType());
  if (StructType *ST = dyn_cast<StructType>(Ty))
    return std::any_of(
        ST->subtypes().begin(), ST->subtypes().end(),
        [](Type *SubType) { return containsGCPtrType(SubType); });
  return false;
}

// Returns true if this is a type which a) is a gc pointer or contains a GC
// pointer and b) is of a type which the code doesn't expect (i.e. first class
// aggregates).  Used to trip assertions.
static bool isUnhandledGCPointerType(Type *Ty) {
  return containsGCPtrType(Ty) && !isHandledGCPointerType(Ty);
}
#endif

static bool order_by_name(llvm::Value *a, llvm::Value *b) {
  if (a->hasName() && b->hasName()) {
    return -1 == a->getName().compare(b->getName());
  } else if (a->hasName() && !b->hasName()) {
    return true;
  } else if (!a->hasName() && b->hasName()) {
    return false;
  } else {
    // Better than nothing, but not stable
    return a < b;
  }
}

// Conservatively identifies any definitions which might be live at the
// given instruction. The  analysis is performed immediately before the
// given instruction. Values defined by that instruction are not considered
// live.  Values used by that instruction are considered live.
static void analyzeParsePointLiveness(
    DominatorTree &DT, GCPtrLivenessData &OriginalLivenessData,
    const CallSite &CS, PartiallyConstructedSafepointRecord &result) {
  Instruction *inst = CS.getInstruction();

  StatepointLiveSetTy liveset;
  findLiveSetAtInst(inst, OriginalLivenessData, liveset);

  if (PrintLiveSet) {
    // Note: This output is used by several of the test cases
    // The order of elemtns in a set is not stable, put them in a vec and sort
    // by name
    SmallVector<Value *, 64> temp;
    temp.insert(temp.end(), liveset.begin(), liveset.end());
    std::sort(temp.begin(), temp.end(), order_by_name);
    errs() << "Live Variables:\n";
    for (Value *V : temp) {
      errs() << " " << V->getName(); // no newline
      V->dump();
    }
  }
  if (PrintLiveSetSize) {
    errs() << "Safepoint For: " << CS.getCalledValue()->getName() << "\n";
    errs() << "Number live values: " << liveset.size() << "\n";
  }
  result.liveset = liveset;
}

static Value *findBaseDefiningValue(Value *I);

/// Return a base defining value for the 'Index' element of the given vector
/// instruction 'I'.  If Index is null, returns a BDV for the entire vector
/// 'I'.  As an optimization, this method will try to determine when the 
/// element is known to already be a base pointer.  If this can be established,
/// the second value in the returned pair will be true.  Note that either a
/// vector or a pointer typed value can be returned.  For the former, the
/// vector returned is a BDV (and possibly a base) of the entire vector 'I'.
/// If the later, the return pointer is a BDV (or possibly a base) for the
/// particular element in 'I'.  
static std::pair<Value *, bool>
findBaseDefiningValueOfVector(Value *I, Value *Index = nullptr) {
  assert(I->getType()->isVectorTy() &&
         cast<VectorType>(I->getType())->getElementType()->isPointerTy() &&
         "Illegal to ask for the base pointer of a non-pointer type");

  // Each case parallels findBaseDefiningValue below, see that code for
  // detailed motivation.

  if (isa<Argument>(I))
    // An incoming argument to the function is a base pointer
    return std::make_pair(I, true);

  // We shouldn't see the address of a global as a vector value?
  assert(!isa<GlobalVariable>(I) &&
         "unexpected global variable found in base of vector");

  // inlining could possibly introduce phi node that contains
  // undef if callee has multiple returns
  if (isa<UndefValue>(I))
    // utterly meaningless, but useful for dealing with partially optimized
    // code.
    return std::make_pair(I, true);

  // Due to inheritance, this must be _after_ the global variable and undef
  // checks
  if (Constant *Con = dyn_cast<Constant>(I)) {
    assert(!isa<GlobalVariable>(I) && !isa<UndefValue>(I) &&
           "order of checks wrong!");
    assert(Con->isNullValue() && "null is the only case which makes sense");
    return std::make_pair(Con, true);
  }
  
  if (isa<LoadInst>(I))
    return std::make_pair(I, true);
  
  // For an insert element, we might be able to look through it if we know
  // something about the indexes.
  if (InsertElementInst *IEI = dyn_cast<InsertElementInst>(I)) {
    if (Index) {
      Value *InsertIndex = IEI->getOperand(2);
      // This index is inserting the value, look for its BDV
      if (InsertIndex == Index)
        return std::make_pair(findBaseDefiningValue(IEI->getOperand(1)), false);
      // Both constant, and can't be equal per above. This insert is definitely
      // not relevant, look back at the rest of the vector and keep trying.
      if (isa<ConstantInt>(Index) && isa<ConstantInt>(InsertIndex))
        return findBaseDefiningValueOfVector(IEI->getOperand(0), Index);
    }
    
    // We don't know whether this vector contains entirely base pointers or
    // not.  To be conservatively correct, we treat it as a BDV and will
    // duplicate code as needed to construct a parallel vector of bases.
    return std::make_pair(IEI, false);
  }

  if (isa<ShuffleVectorInst>(I))
    // We don't know whether this vector contains entirely base pointers or
    // not.  To be conservatively correct, we treat it as a BDV and will
    // duplicate code as needed to construct a parallel vector of bases.
    // TODO: There a number of local optimizations which could be applied here
    // for particular sufflevector patterns.
    return std::make_pair(I, false);

  // A PHI or Select is a base defining value.  The outer findBasePointer
  // algorithm is responsible for constructing a base value for this BDV.
  assert((isa<SelectInst>(I) || isa<PHINode>(I)) &&
         "unknown vector instruction - no base found for vector element");
  return std::make_pair(I, false);
}

static bool isKnownBaseResult(Value *V);

/// Helper function for findBasePointer - Will return a value which either a)
/// defines the base pointer for the input or b) blocks the simple search
/// (i.e. a PHI or Select of two derived pointers)
static Value *findBaseDefiningValue(Value *I) {
  if (I->getType()->isVectorTy())
    return findBaseDefiningValueOfVector(I).first;
  
  assert(I->getType()->isPointerTy() &&
         "Illegal to ask for the base pointer of a non-pointer type");

  // This case is a bit of a hack - it only handles extracts from vectors which
  // trivially contain only base pointers or cases where we can directly match
  // the index of the original extract element to an insertion into the vector.
  // See note inside the function for how to improve this.
  if (auto *EEI = dyn_cast<ExtractElementInst>(I)) {
    Value *VectorOperand = EEI->getVectorOperand();
    Value *Index = EEI->getIndexOperand();
    std::pair<Value *, bool> pair =
      findBaseDefiningValueOfVector(VectorOperand, Index);
    Value *VectorBase = pair.first;
    if (VectorBase->getType()->isPointerTy())
      // We found a BDV for this specific element with the vector.  This is an
      // optimization, but in practice it covers most of the useful cases
      // created via scalarization.
      return VectorBase;
    else {
      assert(VectorBase->getType()->isVectorTy());
      if (pair.second)
        // If the entire vector returned is known to be entirely base pointers,
        // then the extractelement is valid base for this value.
        return EEI;
      else {
        // Otherwise, we have an instruction which potentially produces a
        // derived pointer and we need findBasePointers to clone code for us
        // such that we can create an instruction which produces the
        // accompanying base pointer.
        // Note: This code is currently rather incomplete.  We don't currently
        // support the general form of shufflevector of insertelement.
        // Conceptually, these are just 'base defining values' of the same
        // variety as phi or select instructions.  We need to update the
        // findBasePointers algorithm to insert new 'base-only' versions of the
        // original instructions. This is relative straight forward to do, but
        // the case which would motivate the work hasn't shown up in real
        // workloads yet.  
        assert((isa<PHINode>(VectorBase) || isa<SelectInst>(VectorBase)) &&
               "need to extend findBasePointers for generic vector"
               "instruction cases");
        return VectorBase;
      }
    }
  }

  if (isa<Argument>(I))
    // An incoming argument to the function is a base pointer
    // We should have never reached here if this argument isn't an gc value
    return I;

  if (isa<GlobalVariable>(I))
    // base case
    return I;

  // inlining could possibly introduce phi node that contains
  // undef if callee has multiple returns
  if (isa<UndefValue>(I))
    // utterly meaningless, but useful for dealing with
    // partially optimized code.
    return I;

  // Due to inheritance, this must be _after_ the global variable and undef
  // checks
  if (Constant *Con = dyn_cast<Constant>(I)) {
    assert(!isa<GlobalVariable>(I) && !isa<UndefValue>(I) &&
           "order of checks wrong!");
    // Note: Finding a constant base for something marked for relocation
    // doesn't really make sense.  The most likely case is either a) some
    // screwed up the address space usage or b) your validating against
    // compiled C++ code w/o the proper separation.  The only real exception
    // is a null pointer.  You could have generic code written to index of
    // off a potentially null value and have proven it null.  We also use
    // null pointers in dead paths of relocation phis (which we might later
    // want to find a base pointer for).
    assert(isa<ConstantPointerNull>(Con) &&
           "null is the only case which makes sense");
    return Con;
  }

  if (CastInst *CI = dyn_cast<CastInst>(I)) {
    Value *Def = CI->stripPointerCasts();
    // If we find a cast instruction here, it means we've found a cast which is
    // not simply a pointer cast (i.e. an inttoptr).  We don't know how to
    // handle int->ptr conversion.
    assert(!isa<CastInst>(Def) && "shouldn't find another cast here");
    return findBaseDefiningValue(Def);
  }

  if (isa<LoadInst>(I))
    return I; // The value loaded is an gc base itself

  if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(I))
    // The base of this GEP is the base
    return findBaseDefiningValue(GEP->getPointerOperand());

  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
    switch (II->getIntrinsicID()) {
    case Intrinsic::experimental_gc_result_ptr:
    default:
      // fall through to general call handling
      break;
    case Intrinsic::experimental_gc_statepoint:
    case Intrinsic::experimental_gc_result_float:
    case Intrinsic::experimental_gc_result_int:
      llvm_unreachable("these don't produce pointers");
    case Intrinsic::experimental_gc_relocate: {
      // Rerunning safepoint insertion after safepoints are already
      // inserted is not supported.  It could probably be made to work,
      // but why are you doing this?  There's no good reason.
      llvm_unreachable("repeat safepoint insertion is not supported");
    }
    case Intrinsic::gcroot:
      // Currently, this mechanism hasn't been extended to work with gcroot.
      // There's no reason it couldn't be, but I haven't thought about the
      // implications much.
      llvm_unreachable(
          "interaction with the gcroot mechanism is not supported");
    }
  }
  // We assume that functions in the source language only return base
  // pointers.  This should probably be generalized via attributes to support
  // both source language and internal functions.
  if (isa<CallInst>(I) || isa<InvokeInst>(I))
    return I;

  // I have absolutely no idea how to implement this part yet.  It's not
  // neccessarily hard, I just haven't really looked at it yet.
  assert(!isa<LandingPadInst>(I) && "Landing Pad is unimplemented");

  if (isa<AtomicCmpXchgInst>(I))
    // A CAS is effectively a atomic store and load combined under a
    // predicate.  From the perspective of base pointers, we just treat it
    // like a load.
    return I;

  assert(!isa<AtomicRMWInst>(I) && "Xchg handled above, all others are "
                                   "binary ops which don't apply to pointers");

  // The aggregate ops.  Aggregates can either be in the heap or on the
  // stack, but in either case, this is simply a field load.  As a result,
  // this is a defining definition of the base just like a load is.
  if (isa<ExtractValueInst>(I))
    return I;

  // We should never see an insert vector since that would require we be
  // tracing back a struct value not a pointer value.
  assert(!isa<InsertValueInst>(I) &&
         "Base pointer for a struct is meaningless");

  // The last two cases here don't return a base pointer.  Instead, they
  // return a value which dynamically selects from amoung several base
  // derived pointers (each with it's own base potentially).  It's the job of
  // the caller to resolve these.
  assert((isa<SelectInst>(I) || isa<PHINode>(I)) &&
         "missing instruction case in findBaseDefiningValing");
  return I;
}

/// Returns the base defining value for this value.
static Value *findBaseDefiningValueCached(Value *I, DefiningValueMapTy &Cache) {
  Value *&Cached = Cache[I];
  if (!Cached) {
    Cached = findBaseDefiningValue(I);
  }
  assert(Cache[I] != nullptr);

  if (TraceLSP) {
    dbgs() << "fBDV-cached: " << I->getName() << " -> " << Cached->getName()
           << "\n";
  }
  return Cached;
}

/// Return a base pointer for this value if known.  Otherwise, return it's
/// base defining value.
static Value *findBaseOrBDV(Value *I, DefiningValueMapTy &Cache) {
  Value *Def = findBaseDefiningValueCached(I, Cache);
  auto Found = Cache.find(Def);
  if (Found != Cache.end()) {
    // Either a base-of relation, or a self reference.  Caller must check.
    return Found->second;
  }
  // Only a BDV available
  return Def;
}

/// Given the result of a call to findBaseDefiningValue, or findBaseOrBDV,
/// is it known to be a base pointer?  Or do we need to continue searching.
static bool isKnownBaseResult(Value *V) {
  if (!isa<PHINode>(V) && !isa<SelectInst>(V)) {
    // no recursion possible
    return true;
  }
  if (isa<Instruction>(V) &&
      cast<Instruction>(V)->getMetadata("is_base_value")) {
    // This is a previously inserted base phi or select.  We know
    // that this is a base value.
    return true;
  }

  // We need to keep searching
  return false;
}

// TODO: find a better name for this
namespace {
class PhiState {
public:
  enum Status { Unknown, Base, Conflict };

  PhiState(Status s, Value *b = nullptr) : status(s), base(b) {
    assert(status != Base || b);
  }
  PhiState(Value *b) : status(Base), base(b) {}
  PhiState() : status(Unknown), base(nullptr) {}

  Status getStatus() const { return status; }
  Value *getBase() const { return base; }

  bool isBase() const { return getStatus() == Base; }
  bool isUnknown() const { return getStatus() == Unknown; }
  bool isConflict() const { return getStatus() == Conflict; }

  bool operator==(const PhiState &other) const {
    return base == other.base && status == other.status;
  }

  bool operator!=(const PhiState &other) const { return !(*this == other); }

  void dump() {
    errs() << status << " (" << base << " - "
           << (base ? base->getName() : "nullptr") << "): ";
  }

private:
  Status status;
  Value *base; // non null only if status == base
};

typedef DenseMap<Value *, PhiState> ConflictStateMapTy;
// Values of type PhiState form a lattice, and this is a helper
// class that implementes the meet operation.  The meat of the meet
// operation is implemented in MeetPhiStates::pureMeet
class MeetPhiStates {
public:
  // phiStates is a mapping from PHINodes and SelectInst's to PhiStates.
  explicit MeetPhiStates(const ConflictStateMapTy &phiStates)
      : phiStates(phiStates) {}

  // Destructively meet the current result with the base V.  V can
  // either be a merge instruction (SelectInst / PHINode), in which
  // case its status is looked up in the phiStates map; or a regular
  // SSA value, in which case it is assumed to be a base.
  void meetWith(Value *V) {
    PhiState otherState = getStateForBDV(V);
    assert((MeetPhiStates::pureMeet(otherState, currentResult) ==
            MeetPhiStates::pureMeet(currentResult, otherState)) &&
           "math is wrong: meet does not commute!");
    currentResult = MeetPhiStates::pureMeet(otherState, currentResult);
  }

  PhiState getResult() const { return currentResult; }

private:
  const ConflictStateMapTy &phiStates;
  PhiState currentResult;

  /// Return a phi state for a base defining value.  We'll generate a new
  /// base state for known bases and expect to find a cached state otherwise
  PhiState getStateForBDV(Value *baseValue) {
    if (isKnownBaseResult(baseValue)) {
      return PhiState(baseValue);
    } else {
      return lookupFromMap(baseValue);
    }
  }

  PhiState lookupFromMap(Value *V) {
    auto I = phiStates.find(V);
    assert(I != phiStates.end() && "lookup failed!");
    return I->second;
  }

  static PhiState pureMeet(const PhiState &stateA, const PhiState &stateB) {
    switch (stateA.getStatus()) {
    case PhiState::Unknown:
      return stateB;

    case PhiState::Base:
      assert(stateA.getBase() && "can't be null");
      if (stateB.isUnknown())
        return stateA;

      if (stateB.isBase()) {
        if (stateA.getBase() == stateB.getBase()) {
          assert(stateA == stateB && "equality broken!");
          return stateA;
        }
        return PhiState(PhiState::Conflict);
      }
      assert(stateB.isConflict() && "only three states!");
      return PhiState(PhiState::Conflict);

    case PhiState::Conflict:
      return stateA;
    }
    llvm_unreachable("only three states!");
  }
};
}
/// For a given value or instruction, figure out what base ptr it's derived
/// from.  For gc objects, this is simply itself.  On success, returns a value
/// which is the base pointer.  (This is reliable and can be used for
/// relocation.)  On failure, returns nullptr.
static Value *findBasePointer(Value *I, DefiningValueMapTy &cache) {
  Value *def = findBaseOrBDV(I, cache);

  if (isKnownBaseResult(def)) {
    return def;
  }

  // Here's the rough algorithm:
  // - For every SSA value, construct a mapping to either an actual base
  //   pointer or a PHI which obscures the base pointer.
  // - Construct a mapping from PHI to unknown TOP state.  Use an
  //   optimistic algorithm to propagate base pointer information.  Lattice
  //   looks like:
  //   UNKNOWN
  //   b1 b2 b3 b4
  //   CONFLICT
  //   When algorithm terminates, all PHIs will either have a single concrete
  //   base or be in a conflict state.
  // - For every conflict, insert a dummy PHI node without arguments.  Add
  //   these to the base[Instruction] = BasePtr mapping.  For every
  //   non-conflict, add the actual base.
  //  - For every conflict, add arguments for the base[a] of each input
  //   arguments.
  //
  // Note: A simpler form of this would be to add the conflict form of all
  // PHIs without running the optimistic algorithm.  This would be
  // analougous to pessimistic data flow and would likely lead to an
  // overall worse solution.

  ConflictStateMapTy states;
  states[def] = PhiState();
  // Recursively fill in all phis & selects reachable from the initial one
  // for which we don't already know a definite base value for
  // TODO: This should be rewritten with a worklist
  bool done = false;
  while (!done) {
    done = true;
    // Since we're adding elements to 'states' as we run, we can't keep
    // iterators into the set.
    SmallVector<Value *, 16> Keys;
    Keys.reserve(states.size());
    for (auto Pair : states) {
      Value *V = Pair.first;
      Keys.push_back(V);
    }
    for (Value *v : Keys) {
      assert(!isKnownBaseResult(v) && "why did it get added?");
      if (PHINode *phi = dyn_cast<PHINode>(v)) {
        assert(phi->getNumIncomingValues() > 0 &&
               "zero input phis are illegal");
        for (Value *InVal : phi->incoming_values()) {
          Value *local = findBaseOrBDV(InVal, cache);
          if (!isKnownBaseResult(local) && states.find(local) == states.end()) {
            states[local] = PhiState();
            done = false;
          }
        }
      } else if (SelectInst *sel = dyn_cast<SelectInst>(v)) {
        Value *local = findBaseOrBDV(sel->getTrueValue(), cache);
        if (!isKnownBaseResult(local) && states.find(local) == states.end()) {
          states[local] = PhiState();
          done = false;
        }
        local = findBaseOrBDV(sel->getFalseValue(), cache);
        if (!isKnownBaseResult(local) && states.find(local) == states.end()) {
          states[local] = PhiState();
          done = false;
        }
      }
    }
  }

  if (TraceLSP) {
    errs() << "States after initialization:\n";
    for (auto Pair : states) {
      Instruction *v = cast<Instruction>(Pair.first);
      PhiState state = Pair.second;
      state.dump();
      v->dump();
    }
  }

  // TODO: come back and revisit the state transitions around inputs which
  // have reached conflict state.  The current version seems too conservative.

  bool progress = true;
  while (progress) {
#ifndef NDEBUG
    size_t oldSize = states.size();
#endif
    progress = false;
    // We're only changing keys in this loop, thus safe to keep iterators
    for (auto Pair : states) {
      MeetPhiStates calculateMeet(states);
      Value *v = Pair.first;
      assert(!isKnownBaseResult(v) && "why did it get added?");
      if (SelectInst *select = dyn_cast<SelectInst>(v)) {
        calculateMeet.meetWith(findBaseOrBDV(select->getTrueValue(), cache));
        calculateMeet.meetWith(findBaseOrBDV(select->getFalseValue(), cache));
      } else
        for (Value *Val : cast<PHINode>(v)->incoming_values())
          calculateMeet.meetWith(findBaseOrBDV(Val, cache));

      PhiState oldState = states[v];
      PhiState newState = calculateMeet.getResult();
      if (oldState != newState) {
        progress = true;
        states[v] = newState;
      }
    }

    assert(oldSize <= states.size());
    assert(oldSize == states.size() || progress);
  }

  if (TraceLSP) {
    errs() << "States after meet iteration:\n";
    for (auto Pair : states) {
      Instruction *v = cast<Instruction>(Pair.first);
      PhiState state = Pair.second;
      state.dump();
      v->dump();
    }
  }

  // Insert Phis for all conflicts
  // We want to keep naming deterministic in the loop that follows, so
  // sort the keys before iteration.  This is useful in allowing us to
  // write stable tests. Note that there is no invalidation issue here.
  SmallVector<Value *, 16> Keys;
  Keys.reserve(states.size());
  for (auto Pair : states) {
    Value *V = Pair.first;
    Keys.push_back(V);
  }
  std::sort(Keys.begin(), Keys.end(), order_by_name);
  // TODO: adjust naming patterns to avoid this order of iteration dependency
  for (Value *V : Keys) {
    Instruction *v = cast<Instruction>(V);
    PhiState state = states[V];
    assert(!isKnownBaseResult(v) && "why did it get added?");
    assert(!state.isUnknown() && "Optimistic algorithm didn't complete!");
    if (!state.isConflict())
      continue;

    if (isa<PHINode>(v)) {
      int num_preds =
          std::distance(pred_begin(v->getParent()), pred_end(v->getParent()));
      assert(num_preds > 0 && "how did we reach here");
      PHINode *phi = PHINode::Create(v->getType(), num_preds, "base_phi", v);
      // Add metadata marking this as a base value
      auto *const_1 = ConstantInt::get(
          Type::getInt32Ty(
              v->getParent()->getParent()->getParent()->getContext()),
          1);
      auto MDConst = ConstantAsMetadata::get(const_1);
      MDNode *md = MDNode::get(
          v->getParent()->getParent()->getParent()->getContext(), MDConst);
      phi->setMetadata("is_base_value", md);
      states[v] = PhiState(PhiState::Conflict, phi);
    } else {
      SelectInst *sel = cast<SelectInst>(v);
      // The undef will be replaced later
      UndefValue *undef = UndefValue::get(sel->getType());
      SelectInst *basesel = SelectInst::Create(sel->getCondition(), undef,
                                               undef, "base_select", sel);
      // Add metadata marking this as a base value
      auto *const_1 = ConstantInt::get(
          Type::getInt32Ty(
              v->getParent()->getParent()->getParent()->getContext()),
          1);
      auto MDConst = ConstantAsMetadata::get(const_1);
      MDNode *md = MDNode::get(
          v->getParent()->getParent()->getParent()->getContext(), MDConst);
      basesel->setMetadata("is_base_value", md);
      states[v] = PhiState(PhiState::Conflict, basesel);
    }
  }

  // Fixup all the inputs of the new PHIs
  for (auto Pair : states) {
    Instruction *v = cast<Instruction>(Pair.first);
    PhiState state = Pair.second;

    assert(!isKnownBaseResult(v) && "why did it get added?");
    assert(!state.isUnknown() && "Optimistic algorithm didn't complete!");
    if (!state.isConflict())
      continue;

    if (PHINode *basephi = dyn_cast<PHINode>(state.getBase())) {
      PHINode *phi = cast<PHINode>(v);
      unsigned NumPHIValues = phi->getNumIncomingValues();
      for (unsigned i = 0; i < NumPHIValues; i++) {
        Value *InVal = phi->getIncomingValue(i);
        BasicBlock *InBB = phi->getIncomingBlock(i);

        // If we've already seen InBB, add the same incoming value
        // we added for it earlier.  The IR verifier requires phi
        // nodes with multiple entries from the same basic block
        // to have the same incoming value for each of those
        // entries.  If we don't do this check here and basephi
        // has a different type than base, we'll end up adding two
        // bitcasts (and hence two distinct values) as incoming
        // values for the same basic block.

        int blockIndex = basephi->getBasicBlockIndex(InBB);
        if (blockIndex != -1) {
          Value *oldBase = basephi->getIncomingValue(blockIndex);
          basephi->addIncoming(oldBase, InBB);
#ifndef NDEBUG
          Value *base = findBaseOrBDV(InVal, cache);
          if (!isKnownBaseResult(base)) {
            // Either conflict or base.
            assert(states.count(base));
            base = states[base].getBase();
            assert(base != nullptr && "unknown PhiState!");
          }

          // In essense this assert states: the only way two
          // values incoming from the same basic block may be
          // different is by being different bitcasts of the same
          // value.  A cleanup that remains TODO is changing
          // findBaseOrBDV to return an llvm::Value of the correct
          // type (and still remain pure).  This will remove the
          // need to add bitcasts.
          assert(base->stripPointerCasts() == oldBase->stripPointerCasts() &&
                 "sanity -- findBaseOrBDV should be pure!");
#endif
          continue;
        }

        // Find either the defining value for the PHI or the normal base for
        // a non-phi node
        Value *base = findBaseOrBDV(InVal, cache);
        if (!isKnownBaseResult(base)) {
          // Either conflict or base.
          assert(states.count(base));
          base = states[base].getBase();
          assert(base != nullptr && "unknown PhiState!");
        }
        assert(base && "can't be null");
        // Must use original input BB since base may not be Instruction
        // The cast is needed since base traversal may strip away bitcasts
        if (base->getType() != basephi->getType()) {
          base = new BitCastInst(base, basephi->getType(), "cast",
                                 InBB->getTerminator());
        }
        basephi->addIncoming(base, InBB);
      }
      assert(basephi->getNumIncomingValues() == NumPHIValues);
    } else {
      SelectInst *basesel = cast<SelectInst>(state.getBase());
      SelectInst *sel = cast<SelectInst>(v);
      // Operand 1 & 2 are true, false path respectively. TODO: refactor to
      // something more safe and less hacky.
      for (int i = 1; i <= 2; i++) {
        Value *InVal = sel->getOperand(i);
        // Find either the defining value for the PHI or the normal base for
        // a non-phi node
        Value *base = findBaseOrBDV(InVal, cache);
        if (!isKnownBaseResult(base)) {
          // Either conflict or base.
          assert(states.count(base));
          base = states[base].getBase();
          assert(base != nullptr && "unknown PhiState!");
        }
        assert(base && "can't be null");
        // Must use original input BB since base may not be Instruction
        // The cast is needed since base traversal may strip away bitcasts
        if (base->getType() != basesel->getType()) {
          base = new BitCastInst(base, basesel->getType(), "cast", basesel);
        }
        basesel->setOperand(i, base);
      }
    }
  }

  // Cache all of our results so we can cheaply reuse them
  // NOTE: This is actually two caches: one of the base defining value
  // relation and one of the base pointer relation!  FIXME
  for (auto item : states) {
    Value *v = item.first;
    Value *base = item.second.getBase();
    assert(v && base);
    assert(!isKnownBaseResult(v) && "why did it get added?");

    if (TraceLSP) {
      std::string fromstr =
          cache.count(v) ? (cache[v]->hasName() ? cache[v]->getName() : "")
                         : "none";
      errs() << "Updating base value cache"
             << " for: " << (v->hasName() ? v->getName() : "")
             << " from: " << fromstr
             << " to: " << (base->hasName() ? base->getName() : "") << "\n";
    }

    assert(isKnownBaseResult(base) &&
           "must be something we 'know' is a base pointer");
    if (cache.count(v)) {
      // Once we transition from the BDV relation being store in the cache to
      // the base relation being stored, it must be stable
      assert((!isKnownBaseResult(cache[v]) || cache[v] == base) &&
             "base relation should be stable");
    }
    cache[v] = base;
  }
  assert(cache.find(def) != cache.end());
  return cache[def];
}

// For a set of live pointers (base and/or derived), identify the base
// pointer of the object which they are derived from.  This routine will
// mutate the IR graph as needed to make the 'base' pointer live at the
// definition site of 'derived'.  This ensures that any use of 'derived' can
// also use 'base'.  This may involve the insertion of a number of
// additional PHI nodes.
//
// preconditions: live is a set of pointer type Values
//
// side effects: may insert PHI nodes into the existing CFG, will preserve
// CFG, will not remove or mutate any existing nodes
//
// post condition: PointerToBase contains one (derived, base) pair for every
// pointer in live.  Note that derived can be equal to base if the original
// pointer was a base pointer.
static void
findBasePointers(const StatepointLiveSetTy &live,
                 DenseMap<llvm::Value *, llvm::Value *> &PointerToBase,
                 DominatorTree *DT, DefiningValueMapTy &DVCache) {
  // For the naming of values inserted to be deterministic - which makes for
  // much cleaner and more stable tests - we need to assign an order to the
  // live values.  DenseSets do not provide a deterministic order across runs.
  SmallVector<Value *, 64> Temp;
  Temp.insert(Temp.end(), live.begin(), live.end());
  std::sort(Temp.begin(), Temp.end(), order_by_name);
  for (Value *ptr : Temp) {
    Value *base = findBasePointer(ptr, DVCache);
    assert(base && "failed to find base pointer");
    PointerToBase[ptr] = base;
    assert((!isa<Instruction>(base) || !isa<Instruction>(ptr) ||
            DT->dominates(cast<Instruction>(base)->getParent(),
                          cast<Instruction>(ptr)->getParent())) &&
           "The base we found better dominate the derived pointer");

    // If you see this trip and like to live really dangerously, the code should
    // be correct, just with idioms the verifier can't handle.  You can try
    // disabling the verifier at your own substaintial risk.
    assert(!isa<ConstantPointerNull>(base) &&
           "the relocation code needs adjustment to handle the relocation of "
           "a null pointer constant without causing false positives in the "
           "safepoint ir verifier.");
  }
}

/// Find the required based pointers (and adjust the live set) for the given
/// parse point.
static void findBasePointers(DominatorTree &DT, DefiningValueMapTy &DVCache,
                             const CallSite &CS,
                             PartiallyConstructedSafepointRecord &result) {
  DenseMap<llvm::Value *, llvm::Value *> PointerToBase;
  findBasePointers(result.liveset, PointerToBase, &DT, DVCache);

  if (PrintBasePointers) {
    // Note: Need to print these in a stable order since this is checked in
    // some tests.
    errs() << "Base Pairs (w/o Relocation):\n";
    SmallVector<Value *, 64> Temp;
    Temp.reserve(PointerToBase.size());
    for (auto Pair : PointerToBase) {
      Temp.push_back(Pair.first);
    }
    std::sort(Temp.begin(), Temp.end(), order_by_name);
    for (Value *Ptr : Temp) {
      Value *Base = PointerToBase[Ptr];
      errs() << " derived %" << Ptr->getName() << " base %" << Base->getName()
             << "\n";
    }
  }

  result.PointerToBase = PointerToBase;
}

/// Given an updated version of the dataflow liveness results, update the
/// liveset and base pointer maps for the call site CS.
static void recomputeLiveInValues(GCPtrLivenessData &RevisedLivenessData,
                                  const CallSite &CS,
                                  PartiallyConstructedSafepointRecord &result);

static void recomputeLiveInValues(
    Function &F, DominatorTree &DT, Pass *P, ArrayRef<CallSite> toUpdate,
    MutableArrayRef<struct PartiallyConstructedSafepointRecord> records) {
  // TODO-PERF: reuse the original liveness, then simply run the dataflow
  // again.  The old values are still live and will help it stablize quickly.
  GCPtrLivenessData RevisedLivenessData;
  computeLiveInValues(DT, F, RevisedLivenessData);
  for (size_t i = 0; i < records.size(); i++) {
    struct PartiallyConstructedSafepointRecord &info = records[i];
    const CallSite &CS = toUpdate[i];
    recomputeLiveInValues(RevisedLivenessData, CS, info);
  }
}

// When inserting gc.relocate calls, we need to ensure there are no uses
// of the original value between the gc.statepoint and the gc.relocate call.
// One case which can arise is a phi node starting one of the successor blocks.
// We also need to be able to insert the gc.relocates only on the path which
// goes through the statepoint.  We might need to split an edge to make this
// possible.
static BasicBlock *
normalizeForInvokeSafepoint(BasicBlock *BB, BasicBlock *InvokeParent,
                            DominatorTree &DT) {
  BasicBlock *Ret = BB;
  if (!BB->getUniquePredecessor()) {
    Ret = SplitBlockPredecessors(BB, InvokeParent, "", nullptr, &DT);
  }

  // Now that 'ret' has unique predecessor we can safely remove all phi nodes
  // from it
  FoldSingleEntryPHINodes(Ret);
  assert(!isa<PHINode>(Ret->begin()));

  // At this point, we can safely insert a gc.relocate as the first instruction
  // in Ret if needed.
  return Ret;
}

static int find_index(ArrayRef<Value *> livevec, Value *val) {
  auto itr = std::find(livevec.begin(), livevec.end(), val);
  assert(livevec.end() != itr);
  size_t index = std::distance(livevec.begin(), itr);
  assert(index < livevec.size());
  return index;
}

// Create new attribute set containing only attributes which can be transfered
// from original call to the safepoint.
static AttributeSet legalizeCallAttributes(AttributeSet AS) {
  AttributeSet ret;

  for (unsigned Slot = 0; Slot < AS.getNumSlots(); Slot++) {
    unsigned index = AS.getSlotIndex(Slot);

    if (index == AttributeSet::ReturnIndex ||
        index == AttributeSet::FunctionIndex) {

      for (auto it = AS.begin(Slot), it_end = AS.end(Slot); it != it_end;
           ++it) {
        Attribute attr = *it;

        // Do not allow certain attributes - just skip them
        // Safepoint can not be read only or read none.
        if (attr.hasAttribute(Attribute::ReadNone) ||
            attr.hasAttribute(Attribute::ReadOnly))
          continue;

        ret = ret.addAttributes(
            AS.getContext(), index,
            AttributeSet::get(AS.getContext(), index, AttrBuilder(attr)));
      }
    }

    // Just skip parameter attributes for now
  }

  return ret;
}

/// Helper function to place all gc relocates necessary for the given
/// statepoint.
/// Inputs:
///   liveVariables - list of variables to be relocated.
///   liveStart - index of the first live variable.
///   basePtrs - base pointers.
///   statepointToken - statepoint instruction to which relocates should be
///   bound.
///   Builder - Llvm IR builder to be used to construct new calls.
static void CreateGCRelocates(ArrayRef<llvm::Value *> LiveVariables,
                              const int LiveStart,
                              ArrayRef<llvm::Value *> BasePtrs,
                              Instruction *StatepointToken,
                              IRBuilder<> Builder) {
  SmallVector<Instruction *, 64> NewDefs;
  NewDefs.reserve(LiveVariables.size());

  Module *M = StatepointToken->getParent()->getParent()->getParent();

  for (unsigned i = 0; i < LiveVariables.size(); i++) {
    // We generate a (potentially) unique declaration for every pointer type
    // combination.  This results is some blow up the function declarations in
    // the IR, but removes the need for argument bitcasts which shrinks the IR
    // greatly and makes it much more readable.
    SmallVector<Type *, 1> Types;                 // one per 'any' type
    // All gc_relocate are set to i8 addrspace(1)* type. This could help avoid
    // cases where the actual value's type mangling is not supported by llvm. A
    // bitcast is added later to convert gc_relocate to the actual value's type.
    Types.push_back(Type::getInt8PtrTy(M->getContext(), 1));
    Value *GCRelocateDecl = Intrinsic::getDeclaration(
        M, Intrinsic::experimental_gc_relocate, Types);

    // Generate the gc.relocate call and save the result
    Value *BaseIdx =
        ConstantInt::get(Type::getInt32Ty(M->getContext()),
                         LiveStart + find_index(LiveVariables, BasePtrs[i]));
    Value *LiveIdx = ConstantInt::get(
        Type::getInt32Ty(M->getContext()),
        LiveStart + find_index(LiveVariables, LiveVariables[i]));

    // only specify a debug name if we can give a useful one
    Value *Reloc = Builder.CreateCall(
        GCRelocateDecl, {StatepointToken, BaseIdx, LiveIdx},
        LiveVariables[i]->hasName() ? LiveVariables[i]->getName() + ".relocated"
                                    : "");
    // Trick CodeGen into thinking there are lots of free registers at this
    // fake call.
    cast<CallInst>(Reloc)->setCallingConv(CallingConv::Cold);

    NewDefs.push_back(cast<Instruction>(Reloc));
  }
  assert(NewDefs.size() == LiveVariables.size() &&
         "missing or extra redefinition at safepoint");
}

static void
makeStatepointExplicitImpl(const CallSite &CS, /* to replace */
                           const SmallVectorImpl<llvm::Value *> &basePtrs,
                           const SmallVectorImpl<llvm::Value *> &liveVariables,
                           Pass *P,
                           PartiallyConstructedSafepointRecord &result) {
  assert(basePtrs.size() == liveVariables.size());
  assert(isStatepoint(CS) &&
         "This method expects to be rewriting a statepoint");

  BasicBlock *BB = CS.getInstruction()->getParent();
  assert(BB);
  Function *F = BB->getParent();
  assert(F && "must be set");
  Module *M = F->getParent();
  (void)M;
  assert(M && "must be set");

  // We're not changing the function signature of the statepoint since the gc
  // arguments go into the var args section.
  Function *gc_statepoint_decl = CS.getCalledFunction();

  // Then go ahead and use the builder do actually do the inserts.  We insert
  // immediately before the previous instruction under the assumption that all
  // arguments will be available here.  We can't insert afterwards since we may
  // be replacing a terminator.
  Instruction *insertBefore = CS.getInstruction();
  IRBuilder<> Builder(insertBefore);
  // Copy all of the arguments from the original statepoint - this includes the
  // target, call args, and deopt args
  SmallVector<llvm::Value *, 64> args;
  args.insert(args.end(), CS.arg_begin(), CS.arg_end());
  // TODO: Clear the 'needs rewrite' flag

  // add all the pointers to be relocated (gc arguments)
  // Capture the start of the live variable list for use in the gc_relocates
  const int live_start = args.size();
  args.insert(args.end(), liveVariables.begin(), liveVariables.end());

  // Create the statepoint given all the arguments
  Instruction *token = nullptr;
  AttributeSet return_attributes;
  if (CS.isCall()) {
    CallInst *toReplace = cast<CallInst>(CS.getInstruction());
    CallInst *call =
        Builder.CreateCall(gc_statepoint_decl, args, "safepoint_token");
    call->setTailCall(toReplace->isTailCall());
    call->setCallingConv(toReplace->getCallingConv());

    // Currently we will fail on parameter attributes and on certain
    // function attributes.
    AttributeSet new_attrs = legalizeCallAttributes(toReplace->getAttributes());
    // In case if we can handle this set of sttributes - set up function attrs
    // directly on statepoint and return attrs later for gc_result intrinsic.
    call->setAttributes(new_attrs.getFnAttributes());
    return_attributes = new_attrs.getRetAttributes();

    token = call;

    // Put the following gc_result and gc_relocate calls immediately after the
    // the old call (which we're about to delete)
    BasicBlock::iterator next(toReplace);
    assert(BB->end() != next && "not a terminator, must have next");
    next++;
    Instruction *IP = &*(next);
    Builder.SetInsertPoint(IP);
    Builder.SetCurrentDebugLocation(IP->getDebugLoc());

  } else {
    InvokeInst *toReplace = cast<InvokeInst>(CS.getInstruction());

    // Insert the new invoke into the old block.  We'll remove the old one in a
    // moment at which point this will become the new terminator for the
    // original block.
    InvokeInst *invoke = InvokeInst::Create(
        gc_statepoint_decl, toReplace->getNormalDest(),
        toReplace->getUnwindDest(), args, "", toReplace->getParent());
    invoke->setCallingConv(toReplace->getCallingConv());

    // Currently we will fail on parameter attributes and on certain
    // function attributes.
    AttributeSet new_attrs = legalizeCallAttributes(toReplace->getAttributes());
    // In case if we can handle this set of sttributes - set up function attrs
    // directly on statepoint and return attrs later for gc_result intrinsic.
    invoke->setAttributes(new_attrs.getFnAttributes());
    return_attributes = new_attrs.getRetAttributes();

    token = invoke;

    // Generate gc relocates in exceptional path
    BasicBlock *unwindBlock = toReplace->getUnwindDest();
    assert(!isa<PHINode>(unwindBlock->begin()) &&
           unwindBlock->getUniquePredecessor() &&
           "can't safely insert in this block!");

    Instruction *IP = &*(unwindBlock->getFirstInsertionPt());
    Builder.SetInsertPoint(IP);
    Builder.SetCurrentDebugLocation(toReplace->getDebugLoc());

    // Extract second element from landingpad return value. We will attach
    // exceptional gc relocates to it.
    const unsigned idx = 1;
    Instruction *exceptional_token =
        cast<Instruction>(Builder.CreateExtractValue(
            unwindBlock->getLandingPadInst(), idx, "relocate_token"));
    result.UnwindToken = exceptional_token;

    // Just throw away return value. We will use the one we got for normal
    // block.
    (void)CreateGCRelocates(liveVariables, live_start, basePtrs,
                            exceptional_token, Builder);

    // Generate gc relocates and returns for normal block
    BasicBlock *normalDest = toReplace->getNormalDest();
    assert(!isa<PHINode>(normalDest->begin()) &&
           normalDest->getUniquePredecessor() &&
           "can't safely insert in this block!");

    IP = &*(normalDest->getFirstInsertionPt());
    Builder.SetInsertPoint(IP);

    // gc relocates will be generated later as if it were regular call
    // statepoint
  }
  assert(token);

  // Take the name of the original value call if it had one.
  token->takeName(CS.getInstruction());

// The GCResult is already inserted, we just need to find it
#ifndef NDEBUG
  Instruction *toReplace = CS.getInstruction();
  assert((toReplace->hasNUses(0) || toReplace->hasNUses(1)) &&
         "only valid use before rewrite is gc.result");
  assert(!toReplace->hasOneUse() ||
         isGCResult(cast<Instruction>(*toReplace->user_begin())));
#endif

  // Update the gc.result of the original statepoint (if any) to use the newly
  // inserted statepoint.  This is safe to do here since the token can't be
  // considered a live reference.
  CS.getInstruction()->replaceAllUsesWith(token);

  result.StatepointToken = token;

  // Second, create a gc.relocate for every live variable
  CreateGCRelocates(liveVariables, live_start, basePtrs, token, Builder);
}

namespace {
struct name_ordering {
  Value *base;
  Value *derived;
  bool operator()(name_ordering const &a, name_ordering const &b) {
    return -1 == a.derived->getName().compare(b.derived->getName());
  }
};
}
static void stablize_order(SmallVectorImpl<Value *> &basevec,
                           SmallVectorImpl<Value *> &livevec) {
  assert(basevec.size() == livevec.size());

  SmallVector<name_ordering, 64> temp;
  for (size_t i = 0; i < basevec.size(); i++) {
    name_ordering v;
    v.base = basevec[i];
    v.derived = livevec[i];
    temp.push_back(v);
  }
  std::sort(temp.begin(), temp.end(), name_ordering());
  for (size_t i = 0; i < basevec.size(); i++) {
    basevec[i] = temp[i].base;
    livevec[i] = temp[i].derived;
  }
}

// Replace an existing gc.statepoint with a new one and a set of gc.relocates
// which make the relocations happening at this safepoint explicit.
//
// WARNING: Does not do any fixup to adjust users of the original live
// values.  That's the callers responsibility.
static void
makeStatepointExplicit(DominatorTree &DT, const CallSite &CS, Pass *P,
                       PartiallyConstructedSafepointRecord &result) {
  auto liveset = result.liveset;
  auto PointerToBase = result.PointerToBase;

  // Convert to vector for efficient cross referencing.
  SmallVector<Value *, 64> basevec, livevec;
  livevec.reserve(liveset.size());
  basevec.reserve(liveset.size());
  for (Value *L : liveset) {
    livevec.push_back(L);

    assert(PointerToBase.find(L) != PointerToBase.end());
    Value *base = PointerToBase[L];
    basevec.push_back(base);
  }
  assert(livevec.size() == basevec.size());

  // To make the output IR slightly more stable (for use in diffs), ensure a
  // fixed order of the values in the safepoint (by sorting the value name).
  // The order is otherwise meaningless.
  stablize_order(basevec, livevec);

  // Do the actual rewriting and delete the old statepoint
  makeStatepointExplicitImpl(CS, basevec, livevec, P, result);
  CS.getInstruction()->eraseFromParent();
}

// Helper function for the relocationViaAlloca.
// It receives iterator to the statepoint gc relocates and emits store to the
// assigned
// location (via allocaMap) for the each one of them.
// Add visited values into the visitedLiveValues set we will later use them
// for sanity check.
static void
insertRelocationStores(iterator_range<Value::user_iterator> GCRelocs,
                       DenseMap<Value *, Value *> &AllocaMap,
                       DenseSet<Value *> &VisitedLiveValues) {

  for (User *U : GCRelocs) {
    if (!isa<IntrinsicInst>(U))
      continue;

    IntrinsicInst *RelocatedValue = cast<IntrinsicInst>(U);

    // We only care about relocates
    if (RelocatedValue->getIntrinsicID() !=
        Intrinsic::experimental_gc_relocate) {
      continue;
    }

    GCRelocateOperands RelocateOperands(RelocatedValue);
    Value *OriginalValue =
        const_cast<Value *>(RelocateOperands.getDerivedPtr());
    assert(AllocaMap.count(OriginalValue));
    Value *Alloca = AllocaMap[OriginalValue];

    // Emit store into the related alloca
    // All gc_relocate are i8 addrspace(1)* typed, and it must be bitcasted to
    // the correct type according to alloca.
    assert(RelocatedValue->getNextNode() && "Should always have one since it's not a terminator");
    IRBuilder<> Builder(RelocatedValue->getNextNode());
    Value *CastedRelocatedValue =
        Builder.CreateBitCast(RelocatedValue, cast<AllocaInst>(Alloca)->getAllocatedType(),
        RelocatedValue->hasName() ? RelocatedValue->getName() + ".casted" : "");

    StoreInst *Store = new StoreInst(CastedRelocatedValue, Alloca);
    Store->insertAfter(cast<Instruction>(CastedRelocatedValue));

#ifndef NDEBUG
    VisitedLiveValues.insert(OriginalValue);
#endif
  }
}

// Helper function for the "relocationViaAlloca". Similar to the
// "insertRelocationStores" but works for rematerialized values.
static void
insertRematerializationStores(
  RematerializedValueMapTy RematerializedValues,
  DenseMap<Value *, Value *> &AllocaMap,
  DenseSet<Value *> &VisitedLiveValues) {

  for (auto RematerializedValuePair: RematerializedValues) {
    Instruction *RematerializedValue = RematerializedValuePair.first;
    Value *OriginalValue = RematerializedValuePair.second;

    assert(AllocaMap.count(OriginalValue) &&
           "Can not find alloca for rematerialized value");
    Value *Alloca = AllocaMap[OriginalValue];

    StoreInst *Store = new StoreInst(RematerializedValue, Alloca);
    Store->insertAfter(RematerializedValue);

#ifndef NDEBUG
    VisitedLiveValues.insert(OriginalValue);
#endif
  }
}

/// do all the relocation update via allocas and mem2reg
static void relocationViaAlloca(
    Function &F, DominatorTree &DT, ArrayRef<Value *> Live,
    ArrayRef<struct PartiallyConstructedSafepointRecord> Records) {
#ifndef NDEBUG
  // record initial number of (static) allocas; we'll check we have the same
  // number when we get done.
  int InitialAllocaNum = 0;
  for (auto I = F.getEntryBlock().begin(), E = F.getEntryBlock().end(); I != E;
       I++)
    if (isa<AllocaInst>(*I))
      InitialAllocaNum++;
#endif

  // TODO-PERF: change data structures, reserve
  DenseMap<Value *, Value *> AllocaMap;
  SmallVector<AllocaInst *, 200> PromotableAllocas;
  // Used later to chack that we have enough allocas to store all values
  std::size_t NumRematerializedValues = 0;
  PromotableAllocas.reserve(Live.size());

  // Emit alloca for "LiveValue" and record it in "allocaMap" and
  // "PromotableAllocas"
  auto emitAllocaFor = [&](Value *LiveValue) {
    AllocaInst *Alloca = new AllocaInst(LiveValue->getType(), "",
                                        F.getEntryBlock().getFirstNonPHI());
    AllocaMap[LiveValue] = Alloca;
    PromotableAllocas.push_back(Alloca);
  };

  // emit alloca for each live gc pointer
  for (unsigned i = 0; i < Live.size(); i++) {
    emitAllocaFor(Live[i]);
  }

  // emit allocas for rematerialized values
  for (size_t i = 0; i < Records.size(); i++) {
    const struct PartiallyConstructedSafepointRecord &Info = Records[i];

    for (auto RematerializedValuePair : Info.RematerializedValues) {
      Value *OriginalValue = RematerializedValuePair.second;
      if (AllocaMap.count(OriginalValue) != 0)
        continue;

      emitAllocaFor(OriginalValue);
      ++NumRematerializedValues;
    }
  }

  // The next two loops are part of the same conceptual operation.  We need to
  // insert a store to the alloca after the original def and at each
  // redefinition.  We need to insert a load before each use.  These are split
  // into distinct loops for performance reasons.

  // update gc pointer after each statepoint
  // either store a relocated value or null (if no relocated value found for
  // this gc pointer and it is not a gc_result)
  // this must happen before we update the statepoint with load of alloca
  // otherwise we lose the link between statepoint and old def
  for (size_t i = 0; i < Records.size(); i++) {
    const struct PartiallyConstructedSafepointRecord &Info = Records[i];
    Value *Statepoint = Info.StatepointToken;

    // This will be used for consistency check
    DenseSet<Value *> VisitedLiveValues;

    // Insert stores for normal statepoint gc relocates
    insertRelocationStores(Statepoint->users(), AllocaMap, VisitedLiveValues);

    // In case if it was invoke statepoint
    // we will insert stores for exceptional path gc relocates.
    if (isa<InvokeInst>(Statepoint)) {
      insertRelocationStores(Info.UnwindToken->users(), AllocaMap,
                             VisitedLiveValues);
    }

    // Do similar thing with rematerialized values
    insertRematerializationStores(Info.RematerializedValues, AllocaMap,
                                  VisitedLiveValues);

    if (ClobberNonLive) {
      // As a debuging aid, pretend that an unrelocated pointer becomes null at
      // the gc.statepoint.  This will turn some subtle GC problems into
      // slightly easier to debug SEGVs.  Note that on large IR files with
      // lots of gc.statepoints this is extremely costly both memory and time
      // wise.
      SmallVector<AllocaInst *, 64> ToClobber;
      for (auto Pair : AllocaMap) {
        Value *Def = Pair.first;
        AllocaInst *Alloca = cast<AllocaInst>(Pair.second);

        // This value was relocated
        if (VisitedLiveValues.count(Def)) {
          continue;
        }
        ToClobber.push_back(Alloca);
      }

      auto InsertClobbersAt = [&](Instruction *IP) {
        for (auto *AI : ToClobber) {
          auto AIType = cast<PointerType>(AI->getType());
          auto PT = cast<PointerType>(AIType->getElementType());
          Constant *CPN = ConstantPointerNull::get(PT);
          StoreInst *Store = new StoreInst(CPN, AI);
          Store->insertBefore(IP);
        }
      };

      // Insert the clobbering stores.  These may get intermixed with the
      // gc.results and gc.relocates, but that's fine.
      if (auto II = dyn_cast<InvokeInst>(Statepoint)) {
        InsertClobbersAt(II->getNormalDest()->getFirstInsertionPt());
        InsertClobbersAt(II->getUnwindDest()->getFirstInsertionPt());
      } else {
        BasicBlock::iterator Next(cast<CallInst>(Statepoint));
        Next++;
        InsertClobbersAt(Next);
      }
    }
  }
  // update use with load allocas and add store for gc_relocated
  for (auto Pair : AllocaMap) {
    Value *Def = Pair.first;
    Value *Alloca = Pair.second;

    // we pre-record the uses of allocas so that we dont have to worry about
    // later update
    // that change the user information.
    SmallVector<Instruction *, 20> Uses;
    // PERF: trade a linear scan for repeated reallocation
    Uses.reserve(std::distance(Def->user_begin(), Def->user_end()));
    for (User *U : Def->users()) {
      if (!isa<ConstantExpr>(U)) {
        // If the def has a ConstantExpr use, then the def is either a
        // ConstantExpr use itself or null.  In either case
        // (recursively in the first, directly in the second), the oop
        // it is ultimately dependent on is null and this particular
        // use does not need to be fixed up.
        Uses.push_back(cast<Instruction>(U));
      }
    }

    std::sort(Uses.begin(), Uses.end());
    auto Last = std::unique(Uses.begin(), Uses.end());
    Uses.erase(Last, Uses.end());

    for (Instruction *Use : Uses) {
      if (isa<PHINode>(Use)) {
        PHINode *Phi = cast<PHINode>(Use);
        for (unsigned i = 0; i < Phi->getNumIncomingValues(); i++) {
          if (Def == Phi->getIncomingValue(i)) {
            LoadInst *Load = new LoadInst(
                Alloca, "", Phi->getIncomingBlock(i)->getTerminator());
            Phi->setIncomingValue(i, Load);
          }
        }
      } else {
        LoadInst *Load = new LoadInst(Alloca, "", Use);
        Use->replaceUsesOfWith(Def, Load);
      }
    }

    // emit store for the initial gc value
    // store must be inserted after load, otherwise store will be in alloca's
    // use list and an extra load will be inserted before it
    StoreInst *Store = new StoreInst(Def, Alloca);
    if (Instruction *Inst = dyn_cast<Instruction>(Def)) {
      if (InvokeInst *Invoke = dyn_cast<InvokeInst>(Inst)) {
        // InvokeInst is a TerminatorInst so the store need to be inserted
        // into its normal destination block.
        BasicBlock *NormalDest = Invoke->getNormalDest();
        Store->insertBefore(NormalDest->getFirstNonPHI());
      } else {
        assert(!Inst->isTerminator() &&
               "The only TerminatorInst that can produce a value is "
               "InvokeInst which is handled above.");
        Store->insertAfter(Inst);
      }
    } else {
      assert(isa<Argument>(Def));
      Store->insertAfter(cast<Instruction>(Alloca));
    }
  }

  assert(PromotableAllocas.size() == Live.size() + NumRematerializedValues &&
         "we must have the same allocas with lives");
  if (!PromotableAllocas.empty()) {
    // apply mem2reg to promote alloca to SSA
    PromoteMemToReg(PromotableAllocas, DT);
  }

#ifndef NDEBUG
  for (auto I = F.getEntryBlock().begin(), E = F.getEntryBlock().end(); I != E;
       I++)
    if (isa<AllocaInst>(*I))
      InitialAllocaNum--;
  assert(InitialAllocaNum == 0 && "We must not introduce any extra allocas");
#endif
}

/// Implement a unique function which doesn't require we sort the input
/// vector.  Doing so has the effect of changing the output of a couple of
/// tests in ways which make them less useful in testing fused safepoints.
template <typename T> static void unique_unsorted(SmallVectorImpl<T> &Vec) {
  SmallSet<T, 8> Seen;
  Vec.erase(std::remove_if(Vec.begin(), Vec.end(), [&](const T &V) {
              return !Seen.insert(V).second;
            }), Vec.end());
}

/// Insert holders so that each Value is obviously live through the entire
/// lifetime of the call.
static void insertUseHolderAfter(CallSite &CS, const ArrayRef<Value *> Values,
                                 SmallVectorImpl<CallInst *> &Holders) {
  if (Values.empty())
    // No values to hold live, might as well not insert the empty holder
    return;

  Module *M = CS.getInstruction()->getParent()->getParent()->getParent();
  // Use a dummy vararg function to actually hold the values live
  Function *Func = cast<Function>(M->getOrInsertFunction(
      "__tmp_use", FunctionType::get(Type::getVoidTy(M->getContext()), true)));
  if (CS.isCall()) {
    // For call safepoints insert dummy calls right after safepoint
    BasicBlock::iterator Next(CS.getInstruction());
    Next++;
    Holders.push_back(CallInst::Create(Func, Values, "", Next));
    return;
  }
  // For invoke safepooints insert dummy calls both in normal and
  // exceptional destination blocks
  auto *II = cast<InvokeInst>(CS.getInstruction());
  Holders.push_back(CallInst::Create(
      Func, Values, "", II->getNormalDest()->getFirstInsertionPt()));
  Holders.push_back(CallInst::Create(
      Func, Values, "", II->getUnwindDest()->getFirstInsertionPt()));
}

static void findLiveReferences(
    Function &F, DominatorTree &DT, Pass *P, ArrayRef<CallSite> toUpdate,
    MutableArrayRef<struct PartiallyConstructedSafepointRecord> records) {
  GCPtrLivenessData OriginalLivenessData;
  computeLiveInValues(DT, F, OriginalLivenessData);
  for (size_t i = 0; i < records.size(); i++) {
    struct PartiallyConstructedSafepointRecord &info = records[i];
    const CallSite &CS = toUpdate[i];
    analyzeParsePointLiveness(DT, OriginalLivenessData, CS, info);
  }
}

/// Remove any vector of pointers from the liveset by scalarizing them over the
/// statepoint instruction.  Adds the scalarized pieces to the liveset.  It
/// would be preferrable to include the vector in the statepoint itself, but
/// the lowering code currently does not handle that.  Extending it would be
/// slightly non-trivial since it requires a format change.  Given how rare
/// such cases are (for the moment?) scalarizing is an acceptable comprimise.
static void splitVectorValues(Instruction *StatepointInst,
                              StatepointLiveSetTy &LiveSet,
                              DenseMap<Value *, Value *>& PointerToBase,
                              DominatorTree &DT) {
  SmallVector<Value *, 16> ToSplit;
  for (Value *V : LiveSet)
    if (isa<VectorType>(V->getType()))
      ToSplit.push_back(V);

  if (ToSplit.empty())
    return;

  DenseMap<Value *, SmallVector<Value *, 16>> ElementMapping;

  Function &F = *(StatepointInst->getParent()->getParent());

  DenseMap<Value *, AllocaInst *> AllocaMap;
  // First is normal return, second is exceptional return (invoke only)
  DenseMap<Value *, std::pair<Value *, Value *>> Replacements;
  for (Value *V : ToSplit) {
    AllocaInst *Alloca =
        new AllocaInst(V->getType(), "", F.getEntryBlock().getFirstNonPHI());
    AllocaMap[V] = Alloca;

    VectorType *VT = cast<VectorType>(V->getType());
    IRBuilder<> Builder(StatepointInst);
    SmallVector<Value *, 16> Elements;
    for (unsigned i = 0; i < VT->getNumElements(); i++)
      Elements.push_back(Builder.CreateExtractElement(V, Builder.getInt32(i)));
    ElementMapping[V] = Elements;

    auto InsertVectorReform = [&](Instruction *IP) {
      Builder.SetInsertPoint(IP);
      Builder.SetCurrentDebugLocation(IP->getDebugLoc());
      Value *ResultVec = UndefValue::get(VT);
      for (unsigned i = 0; i < VT->getNumElements(); i++)
        ResultVec = Builder.CreateInsertElement(ResultVec, Elements[i],
                                                Builder.getInt32(i));
      return ResultVec;
    };

    if (isa<CallInst>(StatepointInst)) {
      BasicBlock::iterator Next(StatepointInst);
      Next++;
      Instruction *IP = &*(Next);
      Replacements[V].first = InsertVectorReform(IP);
      Replacements[V].second = nullptr;
    } else {
      InvokeInst *Invoke = cast<InvokeInst>(StatepointInst);
      // We've already normalized - check that we don't have shared destination
      // blocks
      BasicBlock *NormalDest = Invoke->getNormalDest();
      assert(!isa<PHINode>(NormalDest->begin()));
      BasicBlock *UnwindDest = Invoke->getUnwindDest();
      assert(!isa<PHINode>(UnwindDest->begin()));
      // Insert insert element sequences in both successors
      Instruction *IP = &*(NormalDest->getFirstInsertionPt());
      Replacements[V].first = InsertVectorReform(IP);
      IP = &*(UnwindDest->getFirstInsertionPt());
      Replacements[V].second = InsertVectorReform(IP);
    }
  }

  for (Value *V : ToSplit) {
    AllocaInst *Alloca = AllocaMap[V];

    // Capture all users before we start mutating use lists
    SmallVector<Instruction *, 16> Users;
    for (User *U : V->users())
      Users.push_back(cast<Instruction>(U));

    for (Instruction *I : Users) {
      if (auto Phi = dyn_cast<PHINode>(I)) {
        for (unsigned i = 0; i < Phi->getNumIncomingValues(); i++)
          if (V == Phi->getIncomingValue(i)) {
            LoadInst *Load = new LoadInst(
                Alloca, "", Phi->getIncomingBlock(i)->getTerminator());
            Phi->setIncomingValue(i, Load);
          }
      } else {
        LoadInst *Load = new LoadInst(Alloca, "", I);
        I->replaceUsesOfWith(V, Load);
      }
    }

    // Store the original value and the replacement value into the alloca
    StoreInst *Store = new StoreInst(V, Alloca);
    if (auto I = dyn_cast<Instruction>(V))
      Store->insertAfter(I);
    else
      Store->insertAfter(Alloca);

    // Normal return for invoke, or call return
    Instruction *Replacement = cast<Instruction>(Replacements[V].first);
    (new StoreInst(Replacement, Alloca))->insertAfter(Replacement);
    // Unwind return for invoke only
    Replacement = cast_or_null<Instruction>(Replacements[V].second);
    if (Replacement)
      (new StoreInst(Replacement, Alloca))->insertAfter(Replacement);
  }

  // apply mem2reg to promote alloca to SSA
  SmallVector<AllocaInst *, 16> Allocas;
  for (Value *V : ToSplit)
    Allocas.push_back(AllocaMap[V]);
  PromoteMemToReg(Allocas, DT);

  // Update our tracking of live pointers and base mappings to account for the
  // changes we just made.
  for (Value *V : ToSplit) {
    auto &Elements = ElementMapping[V];

    LiveSet.erase(V);
    LiveSet.insert(Elements.begin(), Elements.end());
    // We need to update the base mapping as well.
    assert(PointerToBase.count(V));
    Value *OldBase = PointerToBase[V];
    auto &BaseElements = ElementMapping[OldBase];
    PointerToBase.erase(V);
    assert(Elements.size() == BaseElements.size());
    for (unsigned i = 0; i < Elements.size(); i++) {
      Value *Elem = Elements[i];
      PointerToBase[Elem] = BaseElements[i];
    }
  }
}

// Helper function for the "rematerializeLiveValues". It walks use chain
// starting from the "CurrentValue" until it meets "BaseValue". Only "simple"
// values are visited (currently it is GEP's and casts). Returns true if it
// sucessfully reached "BaseValue" and false otherwise.
// Fills "ChainToBase" array with all visited values. "BaseValue" is not
// recorded.
static bool findRematerializableChainToBasePointer(
  SmallVectorImpl<Instruction*> &ChainToBase,
  Value *CurrentValue, Value *BaseValue) {

  // We have found a base value
  if (CurrentValue == BaseValue) {
    return true;
  }

  if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(CurrentValue)) {
    ChainToBase.push_back(GEP);
    return findRematerializableChainToBasePointer(ChainToBase,
                                                  GEP->getPointerOperand(),
                                                  BaseValue);
  }

  if (CastInst *CI = dyn_cast<CastInst>(CurrentValue)) {
    Value *Def = CI->stripPointerCasts();

    // This two checks are basically similar. First one is here for the
    // consistency with findBasePointers logic.
    assert(!isa<CastInst>(Def) && "not a pointer cast found");
    if (!CI->isNoopCast(CI->getModule()->getDataLayout()))
      return false;

    ChainToBase.push_back(CI);
    return findRematerializableChainToBasePointer(ChainToBase, Def, BaseValue);
  }

  // Not supported instruction in the chain
  return false;
}

// Helper function for the "rematerializeLiveValues". Compute cost of the use
// chain we are going to rematerialize.
static unsigned
chainToBasePointerCost(SmallVectorImpl<Instruction*> &Chain,
                       TargetTransformInfo &TTI) {
  unsigned Cost = 0;

  for (Instruction *Instr : Chain) {
    if (CastInst *CI = dyn_cast<CastInst>(Instr)) {
      assert(CI->isNoopCast(CI->getModule()->getDataLayout()) &&
             "non noop cast is found during rematerialization");

      Type *SrcTy = CI->getOperand(0)->getType();
      Cost += TTI.getCastInstrCost(CI->getOpcode(), CI->getType(), SrcTy);

    } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Instr)) {
      // Cost of the address calculation
      Type *ValTy = GEP->getPointerOperandType()->getPointerElementType();
      Cost += TTI.getAddressComputationCost(ValTy);

      // And cost of the GEP itself
      // TODO: Use TTI->getGEPCost here (it exists, but appears to be not
      //       allowed for the external usage)
      if (!GEP->hasAllConstantIndices())
        Cost += 2;

    } else {
      llvm_unreachable("unsupported instruciton type during rematerialization");
    }
  }

  return Cost;
}

// From the statepoint liveset pick values that are cheaper to recompute then to
// relocate. Remove this values from the liveset, rematerialize them after
// statepoint and record them in "Info" structure. Note that similar to
// relocated values we don't do any user adjustments here.
static void rematerializeLiveValues(CallSite CS,
                                    PartiallyConstructedSafepointRecord &Info,
                                    TargetTransformInfo &TTI) {
  const unsigned int ChainLengthThreshold = 10;

  // Record values we are going to delete from this statepoint live set.
  // We can not di this in following loop due to iterator invalidation.
  SmallVector<Value *, 32> LiveValuesToBeDeleted;

  for (Value *LiveValue: Info.liveset) {
    // For each live pointer find it's defining chain
    SmallVector<Instruction *, 3> ChainToBase;
    assert(Info.PointerToBase.find(LiveValue) != Info.PointerToBase.end());
    bool FoundChain =
      findRematerializableChainToBasePointer(ChainToBase,
                                             LiveValue,
                                             Info.PointerToBase[LiveValue]);
    // Nothing to do, or chain is too long
    if (!FoundChain ||
        ChainToBase.size() == 0 ||
        ChainToBase.size() > ChainLengthThreshold)
      continue;

    // Compute cost of this chain
    unsigned Cost = chainToBasePointerCost(ChainToBase, TTI);
    // TODO: We can also account for cases when we will be able to remove some
    //       of the rematerialized values by later optimization passes. I.e if
    //       we rematerialized several intersecting chains. Or if original values
    //       don't have any uses besides this statepoint.

    // For invokes we need to rematerialize each chain twice - for normal and
    // for unwind basic blocks. Model this by multiplying cost by two.
    if (CS.isInvoke()) {
      Cost *= 2;
    }
    // If it's too expensive - skip it
    if (Cost >= RematerializationThreshold)
      continue;

    // Remove value from the live set
    LiveValuesToBeDeleted.push_back(LiveValue);

    // Clone instructions and record them inside "Info" structure

    // Walk backwards to visit top-most instructions first
    std::reverse(ChainToBase.begin(), ChainToBase.end());

    // Utility function which clones all instructions from "ChainToBase"
    // and inserts them before "InsertBefore". Returns rematerialized value
    // which should be used after statepoint.
    auto rematerializeChain = [&ChainToBase](Instruction *InsertBefore) {
      Instruction *LastClonedValue = nullptr;
      Instruction *LastValue = nullptr;
      for (Instruction *Instr: ChainToBase) {
        // Only GEP's and casts are suported as we need to be careful to not
        // introduce any new uses of pointers not in the liveset.
        // Note that it's fine to introduce new uses of pointers which were
        // otherwise not used after this statepoint.
        assert(isa<GetElementPtrInst>(Instr) || isa<CastInst>(Instr));

        Instruction *ClonedValue = Instr->clone();
        ClonedValue->insertBefore(InsertBefore);
        ClonedValue->setName(Instr->getName() + ".remat");

        // If it is not first instruction in the chain then it uses previously
        // cloned value. We should update it to use cloned value.
        if (LastClonedValue) {
          assert(LastValue);
          ClonedValue->replaceUsesOfWith(LastValue, LastClonedValue);
#ifndef NDEBUG
          // Assert that cloned instruction does not use any instructions from
          // this chain other than LastClonedValue
          for (auto OpValue : ClonedValue->operand_values()) {
            assert(std::find(ChainToBase.begin(), ChainToBase.end(), OpValue) ==
                       ChainToBase.end() &&
                   "incorrect use in rematerialization chain");
          }
#endif
        }

        LastClonedValue = ClonedValue;
        LastValue = Instr;
      }
      assert(LastClonedValue);
      return LastClonedValue;
    };

    // Different cases for calls and invokes. For invokes we need to clone
    // instructions both on normal and unwind path.
    if (CS.isCall()) {
      Instruction *InsertBefore = CS.getInstruction()->getNextNode();
      assert(InsertBefore);
      Instruction *RematerializedValue = rematerializeChain(InsertBefore);
      Info.RematerializedValues[RematerializedValue] = LiveValue;
    } else {
      InvokeInst *Invoke = cast<InvokeInst>(CS.getInstruction());

      Instruction *NormalInsertBefore =
          Invoke->getNormalDest()->getFirstInsertionPt();
      Instruction *UnwindInsertBefore =
          Invoke->getUnwindDest()->getFirstInsertionPt();

      Instruction *NormalRematerializedValue =
          rematerializeChain(NormalInsertBefore);
      Instruction *UnwindRematerializedValue =
          rematerializeChain(UnwindInsertBefore);

      Info.RematerializedValues[NormalRematerializedValue] = LiveValue;
      Info.RematerializedValues[UnwindRematerializedValue] = LiveValue;
    }
  }

  // Remove rematerializaed values from the live set
  for (auto LiveValue: LiveValuesToBeDeleted) {
    Info.liveset.erase(LiveValue);
  }
}

static bool insertParsePoints(Function &F, DominatorTree &DT, Pass *P,
                              SmallVectorImpl<CallSite> &toUpdate) {
#ifndef NDEBUG
  // sanity check the input
  std::set<CallSite> uniqued;
  uniqued.insert(toUpdate.begin(), toUpdate.end());
  assert(uniqued.size() == toUpdate.size() && "no duplicates please!");

  for (size_t i = 0; i < toUpdate.size(); i++) {
    CallSite &CS = toUpdate[i];
    assert(CS.getInstruction()->getParent()->getParent() == &F);
    assert(isStatepoint(CS) && "expected to already be a deopt statepoint");
  }
#endif

  // When inserting gc.relocates for invokes, we need to be able to insert at
  // the top of the successor blocks.  See the comment on
  // normalForInvokeSafepoint on exactly what is needed.  Note that this step
  // may restructure the CFG.
  for (CallSite CS : toUpdate) {
    if (!CS.isInvoke())
      continue;
    InvokeInst *invoke = cast<InvokeInst>(CS.getInstruction());
    normalizeForInvokeSafepoint(invoke->getNormalDest(), invoke->getParent(),
                                DT);
    normalizeForInvokeSafepoint(invoke->getUnwindDest(), invoke->getParent(),
                                DT);
  }

  // A list of dummy calls added to the IR to keep various values obviously
  // live in the IR.  We'll remove all of these when done.
  SmallVector<CallInst *, 64> holders;

  // Insert a dummy call with all of the arguments to the vm_state we'll need
  // for the actual safepoint insertion.  This ensures reference arguments in
  // the deopt argument list are considered live through the safepoint (and
  // thus makes sure they get relocated.)
  for (size_t i = 0; i < toUpdate.size(); i++) {
    CallSite &CS = toUpdate[i];
    Statepoint StatepointCS(CS);

    SmallVector<Value *, 64> DeoptValues;
    for (Use &U : StatepointCS.vm_state_args()) {
      Value *Arg = cast<Value>(&U);
      assert(!isUnhandledGCPointerType(Arg->getType()) &&
             "support for FCA unimplemented");
      if (isHandledGCPointerType(Arg->getType()))
        DeoptValues.push_back(Arg);
    }
    insertUseHolderAfter(CS, DeoptValues, holders);
  }

  SmallVector<struct PartiallyConstructedSafepointRecord, 64> records;
  records.reserve(toUpdate.size());
  for (size_t i = 0; i < toUpdate.size(); i++) {
    struct PartiallyConstructedSafepointRecord info;
    records.push_back(info);
  }
  assert(records.size() == toUpdate.size());

  // A) Identify all gc pointers which are staticly live at the given call
  // site.
  findLiveReferences(F, DT, P, toUpdate, records);

  // B) Find the base pointers for each live pointer
  /* scope for caching */ {
    // Cache the 'defining value' relation used in the computation and
    // insertion of base phis and selects.  This ensures that we don't insert
    // large numbers of duplicate base_phis.
    DefiningValueMapTy DVCache;

    for (size_t i = 0; i < records.size(); i++) {
      struct PartiallyConstructedSafepointRecord &info = records[i];
      CallSite &CS = toUpdate[i];
      findBasePointers(DT, DVCache, CS, info);
    }
  } // end of cache scope

  // The base phi insertion logic (for any safepoint) may have inserted new
  // instructions which are now live at some safepoint.  The simplest such
  // example is:
  // loop:
  //   phi a  <-- will be a new base_phi here
  //   safepoint 1 <-- that needs to be live here
  //   gep a + 1
  //   safepoint 2
  //   br loop
  // We insert some dummy calls after each safepoint to definitely hold live
  // the base pointers which were identified for that safepoint.  We'll then
  // ask liveness for _every_ base inserted to see what is now live.  Then we
  // remove the dummy calls.
  holders.reserve(holders.size() + records.size());
  for (size_t i = 0; i < records.size(); i++) {
    struct PartiallyConstructedSafepointRecord &info = records[i];
    CallSite &CS = toUpdate[i];

    SmallVector<Value *, 128> Bases;
    for (auto Pair : info.PointerToBase) {
      Bases.push_back(Pair.second);
    }
    insertUseHolderAfter(CS, Bases, holders);
  }

  // By selecting base pointers, we've effectively inserted new uses. Thus, we
  // need to rerun liveness.  We may *also* have inserted new defs, but that's
  // not the key issue.
  recomputeLiveInValues(F, DT, P, toUpdate, records);

  if (PrintBasePointers) {
    for (size_t i = 0; i < records.size(); i++) {
      struct PartiallyConstructedSafepointRecord &info = records[i];
      errs() << "Base Pairs: (w/Relocation)\n";
      for (auto Pair : info.PointerToBase) {
        errs() << " derived %" << Pair.first->getName() << " base %"
               << Pair.second->getName() << "\n";
      }
    }
  }
  for (size_t i = 0; i < holders.size(); i++) {
    holders[i]->eraseFromParent();
    holders[i] = nullptr;
  }
  holders.clear();

  // Do a limited scalarization of any live at safepoint vector values which
  // contain pointers.  This enables this pass to run after vectorization at
  // the cost of some possible performance loss.  TODO: it would be nice to
  // natively support vectors all the way through the backend so we don't need
  // to scalarize here.
  for (size_t i = 0; i < records.size(); i++) {
    struct PartiallyConstructedSafepointRecord &info = records[i];
    Instruction *statepoint = toUpdate[i].getInstruction();
    splitVectorValues(cast<Instruction>(statepoint), info.liveset,
                      info.PointerToBase, DT);
  }

  // In order to reduce live set of statepoint we might choose to rematerialize
  // some values instead of relocating them. This is purelly an optimization and
  // does not influence correctness.
  TargetTransformInfo &TTI =
    P->getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);

  for (size_t i = 0; i < records.size(); i++) {
    struct PartiallyConstructedSafepointRecord &info = records[i];
    CallSite &CS = toUpdate[i];

    rematerializeLiveValues(CS, info, TTI);
  }

  // Now run through and replace the existing statepoints with new ones with
  // the live variables listed.  We do not yet update uses of the values being
  // relocated. We have references to live variables that need to
  // survive to the last iteration of this loop.  (By construction, the
  // previous statepoint can not be a live variable, thus we can and remove
  // the old statepoint calls as we go.)
  for (size_t i = 0; i < records.size(); i++) {
    struct PartiallyConstructedSafepointRecord &info = records[i];
    CallSite &CS = toUpdate[i];
    makeStatepointExplicit(DT, CS, P, info);
  }
  toUpdate.clear(); // prevent accident use of invalid CallSites

  // Do all the fixups of the original live variables to their relocated selves
  SmallVector<Value *, 128> live;
  for (size_t i = 0; i < records.size(); i++) {
    struct PartiallyConstructedSafepointRecord &info = records[i];
    // We can't simply save the live set from the original insertion.  One of
    // the live values might be the result of a call which needs a safepoint.
    // That Value* no longer exists and we need to use the new gc_result.
    // Thankfully, the liveset is embedded in the statepoint (and updated), so
    // we just grab that.
    Statepoint statepoint(info.StatepointToken);
    live.insert(live.end(), statepoint.gc_args_begin(),
                statepoint.gc_args_end());
#ifndef NDEBUG
    // Do some basic sanity checks on our liveness results before performing
    // relocation.  Relocation can and will turn mistakes in liveness results
    // into non-sensical code which is must harder to debug.
    // TODO: It would be nice to test consistency as well
    assert(DT.isReachableFromEntry(info.StatepointToken->getParent()) &&
           "statepoint must be reachable or liveness is meaningless");
    for (Value *V : statepoint.gc_args()) {
      if (!isa<Instruction>(V))
        // Non-instruction values trivial dominate all possible uses
        continue;
      auto LiveInst = cast<Instruction>(V);
      assert(DT.isReachableFromEntry(LiveInst->getParent()) &&
             "unreachable values should never be live");
      assert(DT.dominates(LiveInst, info.StatepointToken) &&
             "basic SSA liveness expectation violated by liveness analysis");
    }
#endif
  }
  unique_unsorted(live);

#ifndef NDEBUG
  // sanity check
  for (auto ptr : live) {
    assert(isGCPointerType(ptr->getType()) && "must be a gc pointer type");
  }
#endif

  relocationViaAlloca(F, DT, live, records);
  return !records.empty();
}

// Handles both return values and arguments for Functions and CallSites.
template <typename AttrHolder>
static void RemoveDerefAttrAtIndex(LLVMContext &Ctx, AttrHolder &AH,
                                   unsigned Index) {
  AttrBuilder R;
  if (AH.getDereferenceableBytes(Index))
    R.addAttribute(Attribute::get(Ctx, Attribute::Dereferenceable,
                                  AH.getDereferenceableBytes(Index)));
  if (AH.getDereferenceableOrNullBytes(Index))
    R.addAttribute(Attribute::get(Ctx, Attribute::DereferenceableOrNull,
                                  AH.getDereferenceableOrNullBytes(Index)));

  if (!R.empty())
    AH.setAttributes(AH.getAttributes().removeAttributes(
        Ctx, Index, AttributeSet::get(Ctx, Index, R)));
}

void
RewriteStatepointsForGC::stripDereferenceabilityInfoFromPrototype(Function &F) {
  LLVMContext &Ctx = F.getContext();

  for (Argument &A : F.args())
    if (isa<PointerType>(A.getType()))
      RemoveDerefAttrAtIndex(Ctx, F, A.getArgNo() + 1);

  if (isa<PointerType>(F.getReturnType()))
    RemoveDerefAttrAtIndex(Ctx, F, AttributeSet::ReturnIndex);
}

void RewriteStatepointsForGC::stripDereferenceabilityInfoFromBody(Function &F) {
  if (F.empty())
    return;

  LLVMContext &Ctx = F.getContext();
  MDBuilder Builder(Ctx);

  for (Instruction &I : inst_range(F)) {
    if (const MDNode *MD = I.getMetadata(LLVMContext::MD_tbaa)) {
      assert(MD->getNumOperands() < 5 && "unrecognized metadata shape!");
      bool IsImmutableTBAA =
          MD->getNumOperands() == 4 &&
          mdconst::extract<ConstantInt>(MD->getOperand(3))->getValue() == 1;

      if (!IsImmutableTBAA)
        continue; // no work to do, MD_tbaa is already marked mutable

      MDNode *Base = cast<MDNode>(MD->getOperand(0));
      MDNode *Access = cast<MDNode>(MD->getOperand(1));
      uint64_t Offset =
          mdconst::extract<ConstantInt>(MD->getOperand(2))->getZExtValue();

      MDNode *MutableTBAA =
          Builder.createTBAAStructTagNode(Base, Access, Offset);
      I.setMetadata(LLVMContext::MD_tbaa, MutableTBAA);
    }

    if (CallSite CS = CallSite(&I)) {
      for (int i = 0, e = CS.arg_size(); i != e; i++)
        if (isa<PointerType>(CS.getArgument(i)->getType()))
          RemoveDerefAttrAtIndex(Ctx, CS, i + 1);
      if (isa<PointerType>(CS.getType()))
        RemoveDerefAttrAtIndex(Ctx, CS, AttributeSet::ReturnIndex);
    }
  }
}

/// Returns true if this function should be rewritten by this pass.  The main
/// point of this function is as an extension point for custom logic.
static bool shouldRewriteStatepointsIn(Function &F) {
  // TODO: This should check the GCStrategy
  if (F.hasGC()) {
    const char *FunctionGCName = F.getGC();
    const StringRef StatepointExampleName("statepoint-example");
    const StringRef CoreCLRName("coreclr");
    return (StatepointExampleName == FunctionGCName) ||
           (CoreCLRName == FunctionGCName);
  } else
    return false;
}

void RewriteStatepointsForGC::stripDereferenceabilityInfo(Module &M) {
#ifndef NDEBUG
  assert(std::any_of(M.begin(), M.end(), shouldRewriteStatepointsIn) &&
         "precondition!");
#endif

  for (Function &F : M)
    stripDereferenceabilityInfoFromPrototype(F);

  for (Function &F : M)
    stripDereferenceabilityInfoFromBody(F);
}

bool RewriteStatepointsForGC::runOnFunction(Function &F) {
  // Nothing to do for declarations.
  if (F.isDeclaration() || F.empty())
    return false;

  // Policy choice says not to rewrite - the most common reason is that we're
  // compiling code without a GCStrategy.
  if (!shouldRewriteStatepointsIn(F))
    return false;

  DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>(F).getDomTree();

  // Gather all the statepoints which need rewritten.  Be careful to only
  // consider those in reachable code since we need to ask dominance queries
  // when rewriting.  We'll delete the unreachable ones in a moment.
  SmallVector<CallSite, 64> ParsePointNeeded;
  bool HasUnreachableStatepoint = false;
  for (Instruction &I : inst_range(F)) {
    // TODO: only the ones with the flag set!
    if (isStatepoint(I)) {
      if (DT.isReachableFromEntry(I.getParent()))
        ParsePointNeeded.push_back(CallSite(&I));
      else
        HasUnreachableStatepoint = true;
    }
  }

  bool MadeChange = false;

  // Delete any unreachable statepoints so that we don't have unrewritten
  // statepoints surviving this pass.  This makes testing easier and the
  // resulting IR less confusing to human readers.  Rather than be fancy, we
  // just reuse a utility function which removes the unreachable blocks.
  if (HasUnreachableStatepoint)
    MadeChange |= removeUnreachableBlocks(F);

  // Return early if no work to do.
  if (ParsePointNeeded.empty())
    return MadeChange;

  // As a prepass, go ahead and aggressively destroy single entry phi nodes.
  // These are created by LCSSA.  They have the effect of increasing the size
  // of liveness sets for no good reason.  It may be harder to do this post
  // insertion since relocations and base phis can confuse things.
  for (BasicBlock &BB : F)
    if (BB.getUniquePredecessor()) {
      MadeChange = true;
      FoldSingleEntryPHINodes(&BB);
    }

  MadeChange |= insertParsePoints(F, DT, this, ParsePointNeeded);
  return MadeChange;
}

// liveness computation via standard dataflow
// -------------------------------------------------------------------

// TODO: Consider using bitvectors for liveness, the set of potentially
// interesting values should be small and easy to pre-compute.

/// Compute the live-in set for the location rbegin starting from
/// the live-out set of the basic block
static void computeLiveInValues(BasicBlock::reverse_iterator rbegin,
                                BasicBlock::reverse_iterator rend,
                                DenseSet<Value *> &LiveTmp) {

  for (BasicBlock::reverse_iterator ritr = rbegin; ritr != rend; ritr++) {
    Instruction *I = &*ritr;

    // KILL/Def - Remove this definition from LiveIn
    LiveTmp.erase(I);

    // Don't consider *uses* in PHI nodes, we handle their contribution to
    // predecessor blocks when we seed the LiveOut sets
    if (isa<PHINode>(I))
      continue;

    // USE - Add to the LiveIn set for this instruction
    for (Value *V : I->operands()) {
      assert(!isUnhandledGCPointerType(V->getType()) &&
             "support for FCA unimplemented");
      if (isHandledGCPointerType(V->getType()) && !isa<Constant>(V)) {
        // The choice to exclude all things constant here is slightly subtle.
        // There are two idependent reasons:
        // - We assume that things which are constant (from LLVM's definition)
        // do not move at runtime.  For example, the address of a global
        // variable is fixed, even though it's contents may not be.
        // - Second, we can't disallow arbitrary inttoptr constants even
        // if the language frontend does.  Optimization passes are free to
        // locally exploit facts without respect to global reachability.  This
        // can create sections of code which are dynamically unreachable and
        // contain just about anything.  (see constants.ll in tests)
        LiveTmp.insert(V);
      }
    }
  }
}

static void computeLiveOutSeed(BasicBlock *BB, DenseSet<Value *> &LiveTmp) {

  for (BasicBlock *Succ : successors(BB)) {
    const BasicBlock::iterator E(Succ->getFirstNonPHI());
    for (BasicBlock::iterator I = Succ->begin(); I != E; I++) {
      PHINode *Phi = cast<PHINode>(&*I);
      Value *V = Phi->getIncomingValueForBlock(BB);
      assert(!isUnhandledGCPointerType(V->getType()) &&
             "support for FCA unimplemented");
      if (isHandledGCPointerType(V->getType()) && !isa<Constant>(V)) {
        LiveTmp.insert(V);
      }
    }
  }
}

static DenseSet<Value *> computeKillSet(BasicBlock *BB) {
  DenseSet<Value *> KillSet;
  for (Instruction &I : *BB)
    if (isHandledGCPointerType(I.getType()))
      KillSet.insert(&I);
  return KillSet;
}

#ifndef NDEBUG
/// Check that the items in 'Live' dominate 'TI'.  This is used as a basic
/// sanity check for the liveness computation.
static void checkBasicSSA(DominatorTree &DT, DenseSet<Value *> &Live,
                          TerminatorInst *TI, bool TermOkay = false) {
  for (Value *V : Live) {
    if (auto *I = dyn_cast<Instruction>(V)) {
      // The terminator can be a member of the LiveOut set.  LLVM's definition
      // of instruction dominance states that V does not dominate itself.  As
      // such, we need to special case this to allow it.
      if (TermOkay && TI == I)
        continue;
      assert(DT.dominates(I, TI) &&
             "basic SSA liveness expectation violated by liveness analysis");
    }
  }
}

/// Check that all the liveness sets used during the computation of liveness
/// obey basic SSA properties.  This is useful for finding cases where we miss
/// a def.
static void checkBasicSSA(DominatorTree &DT, GCPtrLivenessData &Data,
                          BasicBlock &BB) {
  checkBasicSSA(DT, Data.LiveSet[&BB], BB.getTerminator());
  checkBasicSSA(DT, Data.LiveOut[&BB], BB.getTerminator(), true);
  checkBasicSSA(DT, Data.LiveIn[&BB], BB.getTerminator());
}
#endif

static void computeLiveInValues(DominatorTree &DT, Function &F,
                                GCPtrLivenessData &Data) {

  SmallSetVector<BasicBlock *, 200> Worklist;
  auto AddPredsToWorklist = [&](BasicBlock *BB) {
    // We use a SetVector so that we don't have duplicates in the worklist.
    Worklist.insert(pred_begin(BB), pred_end(BB));
  };
  auto NextItem = [&]() {
    BasicBlock *BB = Worklist.back();
    Worklist.pop_back();
    return BB;
  };

  // Seed the liveness for each individual block
  for (BasicBlock &BB : F) {
    Data.KillSet[&BB] = computeKillSet(&BB);
    Data.LiveSet[&BB].clear();
    computeLiveInValues(BB.rbegin(), BB.rend(), Data.LiveSet[&BB]);

#ifndef NDEBUG
    for (Value *Kill : Data.KillSet[&BB])
      assert(!Data.LiveSet[&BB].count(Kill) && "live set contains kill");
#endif

    Data.LiveOut[&BB] = DenseSet<Value *>();
    computeLiveOutSeed(&BB, Data.LiveOut[&BB]);
    Data.LiveIn[&BB] = Data.LiveSet[&BB];
    set_union(Data.LiveIn[&BB], Data.LiveOut[&BB]);
    set_subtract(Data.LiveIn[&BB], Data.KillSet[&BB]);
    if (!Data.LiveIn[&BB].empty())
      AddPredsToWorklist(&BB);
  }

  // Propagate that liveness until stable
  while (!Worklist.empty()) {
    BasicBlock *BB = NextItem();

    // Compute our new liveout set, then exit early if it hasn't changed
    // despite the contribution of our successor.
    DenseSet<Value *> LiveOut = Data.LiveOut[BB];
    const auto OldLiveOutSize = LiveOut.size();
    for (BasicBlock *Succ : successors(BB)) {
      assert(Data.LiveIn.count(Succ));
      set_union(LiveOut, Data.LiveIn[Succ]);
    }
    // assert OutLiveOut is a subset of LiveOut
    if (OldLiveOutSize == LiveOut.size()) {
      // If the sets are the same size, then we didn't actually add anything
      // when unioning our successors LiveIn  Thus, the LiveIn of this block
      // hasn't changed.
      continue;
    }
    Data.LiveOut[BB] = LiveOut;

    // Apply the effects of this basic block
    DenseSet<Value *> LiveTmp = LiveOut;
    set_union(LiveTmp, Data.LiveSet[BB]);
    set_subtract(LiveTmp, Data.KillSet[BB]);

    assert(Data.LiveIn.count(BB));
    const DenseSet<Value *> &OldLiveIn = Data.LiveIn[BB];
    // assert: OldLiveIn is a subset of LiveTmp
    if (OldLiveIn.size() != LiveTmp.size()) {
      Data.LiveIn[BB] = LiveTmp;
      AddPredsToWorklist(BB);
    }
  } // while( !worklist.empty() )

#ifndef NDEBUG
  // Sanity check our ouput against SSA properties.  This helps catch any
  // missing kills during the above iteration.
  for (BasicBlock &BB : F) {
    checkBasicSSA(DT, Data, BB);
  }
#endif
}

static void findLiveSetAtInst(Instruction *Inst, GCPtrLivenessData &Data,
                              StatepointLiveSetTy &Out) {

  BasicBlock *BB = Inst->getParent();

  // Note: The copy is intentional and required
  assert(Data.LiveOut.count(BB));
  DenseSet<Value *> LiveOut = Data.LiveOut[BB];

  // We want to handle the statepoint itself oddly.  It's
  // call result is not live (normal), nor are it's arguments
  // (unless they're used again later).  This adjustment is
  // specifically what we need to relocate
  BasicBlock::reverse_iterator rend(Inst);
  computeLiveInValues(BB->rbegin(), rend, LiveOut);
  LiveOut.erase(Inst);
  Out.insert(LiveOut.begin(), LiveOut.end());
}

static void recomputeLiveInValues(GCPtrLivenessData &RevisedLivenessData,
                                  const CallSite &CS,
                                  PartiallyConstructedSafepointRecord &Info) {
  Instruction *Inst = CS.getInstruction();
  StatepointLiveSetTy Updated;
  findLiveSetAtInst(Inst, RevisedLivenessData, Updated);

#ifndef NDEBUG
  DenseSet<Value *> Bases;
  for (auto KVPair : Info.PointerToBase) {
    Bases.insert(KVPair.second);
  }
#endif
  // We may have base pointers which are now live that weren't before.  We need
  // to update the PointerToBase structure to reflect this.
  for (auto V : Updated)
    if (!Info.PointerToBase.count(V)) {
      assert(Bases.count(V) && "can't find base for unexpected live value");
      Info.PointerToBase[V] = V;
      continue;
    }

#ifndef NDEBUG
  for (auto V : Updated) {
    assert(Info.PointerToBase.count(V) &&
           "must be able to find base for live value");
  }
#endif

  // Remove any stale base mappings - this can happen since our liveness is
  // more precise then the one inherent in the base pointer analysis
  DenseSet<Value *> ToErase;
  for (auto KVPair : Info.PointerToBase)
    if (!Updated.count(KVPair.first))
      ToErase.insert(KVPair.first);
  for (auto V : ToErase)
    Info.PointerToBase.erase(V);

#ifndef NDEBUG
  for (auto KVPair : Info.PointerToBase)
    assert(Updated.count(KVPair.first) && "record for non-live value");
#endif

  Info.liveset = Updated;
}
