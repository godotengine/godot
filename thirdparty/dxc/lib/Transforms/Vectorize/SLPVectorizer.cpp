//===- SLPVectorizer.cpp - A bottom up SLP Vectorizer ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This pass implements the Bottom Up SLP vectorizer. It detects consecutive
// stores that can be put together into vector-stores. Next, it attempts to
// construct vectorizable tree using the use-def chains. If a profitable tree
// was found, the SLP vectorizer performs vectorization on the tree.
//
// The pass is inspired by the work described in the paper:
//  "Loop-Aware SLP in GCC" by Ira Rosen, Dorit Nuzman, Ayal Zaks.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/VectorUtils.h"
#include <algorithm>
#include <map>
#include <memory>
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
using namespace llvm;

#define SV_NAME "slp-vectorizer"
#define DEBUG_TYPE "SLP"

STATISTIC(NumVectorInstructions, "Number of vector instructions generated");

static cl::opt<int>
    SLPCostThreshold("slp-threshold", cl::init(0), cl::Hidden,
                     cl::desc("Only vectorize if you gain more than this "
                              "number "));

static cl::opt<bool>
ShouldVectorizeHor("slp-vectorize-hor", cl::init(false), cl::Hidden,
                   cl::desc("Attempt to vectorize horizontal reductions"));

static cl::opt<bool> ShouldStartVectorizeHorAtStore(
    "slp-vectorize-hor-store", cl::init(false), cl::Hidden,
    cl::desc(
        "Attempt to vectorize horizontal reductions feeding into a store"));

static cl::opt<int>
MaxVectorRegSizeOption("slp-max-reg-size", cl::init(128), cl::Hidden,
    cl::desc("Attempt to vectorize for this register size in bits"));

namespace {

// FIXME: Set this via cl::opt to allow overriding.
static const unsigned MinVecRegSize = 128;

static const unsigned RecursionMaxDepth = 12;

// Limit the number of alias checks. The limit is chosen so that
// it has no negative effect on the llvm benchmarks.
static const unsigned AliasedCheckLimit = 10;

// Another limit for the alias checks: The maximum distance between load/store
// instructions where alias checks are done.
// This limit is useful for very large basic blocks.
static const unsigned MaxMemDepDistance = 160;

/// \brief Predicate for the element types that the SLP vectorizer supports.
///
/// The most important thing to filter here are types which are invalid in LLVM
/// vectors. We also filter target specific types which have absolutely no
/// meaningful vectorization path such as x86_fp80 and ppc_f128. This just
/// avoids spending time checking the cost model and realizing that they will
/// be inevitably scalarized.
static bool isValidElementType(Type *Ty) {
  return VectorType::isValidElementType(Ty) && !Ty->isX86_FP80Ty() &&
         !Ty->isPPC_FP128Ty();
}

/// \returns the parent basic block if all of the instructions in \p VL
/// are in the same block or null otherwise.
static BasicBlock *getSameBlock(ArrayRef<Value *> VL) {
  Instruction *I0 = dyn_cast<Instruction>(VL[0]);
  if (!I0)
    return nullptr;
  BasicBlock *BB = I0->getParent();
  for (int i = 1, e = VL.size(); i < e; i++) {
    Instruction *I = dyn_cast<Instruction>(VL[i]);
    if (!I)
      return nullptr;

    if (BB != I->getParent())
      return nullptr;
  }
  return BB;
}

/// \returns True if all of the values in \p VL are constants.
static bool allConstant(ArrayRef<Value *> VL) {
  for (unsigned i = 0, e = VL.size(); i < e; ++i)
    if (!isa<Constant>(VL[i]))
      return false;
  return true;
}

/// \returns True if all of the values in \p VL are identical.
static bool isSplat(ArrayRef<Value *> VL) {
  for (unsigned i = 1, e = VL.size(); i < e; ++i)
    if (VL[i] != VL[0])
      return false;
  return true;
}

///\returns Opcode that can be clubbed with \p Op to create an alternate
/// sequence which can later be merged as a ShuffleVector instruction.
static unsigned getAltOpcode(unsigned Op) {
  switch (Op) {
  case Instruction::FAdd:
    return Instruction::FSub;
  case Instruction::FSub:
    return Instruction::FAdd;
  case Instruction::Add:
    return Instruction::Sub;
  case Instruction::Sub:
    return Instruction::Add;
  default:
    return 0;
  }
}

///\returns bool representing if Opcode \p Op can be part
/// of an alternate sequence which can later be merged as
/// a ShuffleVector instruction.
static bool canCombineAsAltInst(unsigned Op) {
  if (Op == Instruction::FAdd || Op == Instruction::FSub ||
      Op == Instruction::Sub || Op == Instruction::Add)
    return true;
  return false;
}

/// \returns ShuffleVector instruction if intructions in \p VL have
///  alternate fadd,fsub / fsub,fadd/add,sub/sub,add sequence.
/// (i.e. e.g. opcodes of fadd,fsub,fadd,fsub...)
static unsigned isAltInst(ArrayRef<Value *> VL) {
  Instruction *I0 = dyn_cast<Instruction>(VL[0]);
  unsigned Opcode = I0->getOpcode();
  unsigned AltOpcode = getAltOpcode(Opcode);
  for (int i = 1, e = VL.size(); i < e; i++) {
    Instruction *I = dyn_cast<Instruction>(VL[i]);
    if (!I || I->getOpcode() != ((i & 1) ? AltOpcode : Opcode))
      return 0;
  }
  return Instruction::ShuffleVector;
}

/// \returns The opcode if all of the Instructions in \p VL have the same
/// opcode, or zero.
static unsigned getSameOpcode(ArrayRef<Value *> VL) {
  Instruction *I0 = dyn_cast<Instruction>(VL[0]);
  if (!I0)
    return 0;
  unsigned Opcode = I0->getOpcode();
  for (int i = 1, e = VL.size(); i < e; i++) {
    Instruction *I = dyn_cast<Instruction>(VL[i]);
    if (!I || Opcode != I->getOpcode()) {
      if (canCombineAsAltInst(Opcode) && i == 1)
        return isAltInst(VL);
      return 0;
    }
  }
  return Opcode;
}

/// Get the intersection (logical and) of all of the potential IR flags
/// of each scalar operation (VL) that will be converted into a vector (I).
/// Flag set: NSW, NUW, exact, and all of fast-math.
static void propagateIRFlags(Value *I, ArrayRef<Value *> VL) {
  if (auto *VecOp = dyn_cast<BinaryOperator>(I)) {
    if (auto *Intersection = dyn_cast<BinaryOperator>(VL[0])) {
      // Intersection is initialized to the 0th scalar,
      // so start counting from index '1'.
      for (int i = 1, e = VL.size(); i < e; ++i) {
        if (auto *Scalar = dyn_cast<BinaryOperator>(VL[i]))
          Intersection->andIRFlags(Scalar);
      }
      VecOp->copyIRFlags(Intersection);
    }
  }
}
  
/// \returns \p I after propagating metadata from \p VL.
static Instruction *propagateMetadata(Instruction *I, ArrayRef<Value *> VL) {
  Instruction *I0 = cast<Instruction>(VL[0]);
  SmallVector<std::pair<unsigned, MDNode *>, 4> Metadata;
  I0->getAllMetadataOtherThanDebugLoc(Metadata);

  for (unsigned i = 0, n = Metadata.size(); i != n; ++i) {
    unsigned Kind = Metadata[i].first;
    MDNode *MD = Metadata[i].second;

    for (int i = 1, e = VL.size(); MD && i != e; i++) {
      Instruction *I = cast<Instruction>(VL[i]);
      MDNode *IMD = I->getMetadata(Kind);

      switch (Kind) {
      default:
        MD = nullptr; // Remove unknown metadata
        break;
      case LLVMContext::MD_tbaa:
        MD = MDNode::getMostGenericTBAA(MD, IMD);
        break;
      case LLVMContext::MD_alias_scope:
        MD = MDNode::getMostGenericAliasScope(MD, IMD);
        break;
      case LLVMContext::MD_noalias:
        MD = MDNode::intersect(MD, IMD);
        break;
      case LLVMContext::MD_fpmath:
        MD = MDNode::getMostGenericFPMath(MD, IMD);
        break;
      }
    }
    I->setMetadata(Kind, MD);
  }
  return I;
}

/// \returns The type that all of the values in \p VL have or null if there
/// are different types.
static Type* getSameType(ArrayRef<Value *> VL) {
  Type *Ty = VL[0]->getType();
  for (int i = 1, e = VL.size(); i < e; i++)
    if (VL[i]->getType() != Ty)
      return nullptr;

  return Ty;
}

/// \returns True if the ExtractElement instructions in VL can be vectorized
/// to use the original vector.
static bool CanReuseExtract(ArrayRef<Value *> VL) {
  assert(Instruction::ExtractElement == getSameOpcode(VL) && "Invalid opcode");
  // Check if all of the extracts come from the same vector and from the
  // correct offset.
  Value *VL0 = VL[0];
  ExtractElementInst *E0 = cast<ExtractElementInst>(VL0);
  Value *Vec = E0->getOperand(0);

  // We have to extract from the same vector type.
  unsigned NElts = Vec->getType()->getVectorNumElements();

  if (NElts != VL.size())
    return false;

  // Check that all of the indices extract from the correct offset.
  ConstantInt *CI = dyn_cast<ConstantInt>(E0->getOperand(1));
  if (!CI || CI->getZExtValue())
    return false;

  for (unsigned i = 1, e = VL.size(); i < e; ++i) {
    ExtractElementInst *E = cast<ExtractElementInst>(VL[i]);
    ConstantInt *CI = dyn_cast<ConstantInt>(E->getOperand(1));

    if (!CI || CI->getZExtValue() != i || E->getOperand(0) != Vec)
      return false;
  }

  return true;
}

/// \returns True if in-tree use also needs extract. This refers to
/// possible scalar operand in vectorized instruction.
static bool InTreeUserNeedToExtract(Value *Scalar, Instruction *UserInst,
                                    TargetLibraryInfo *TLI) {

  unsigned Opcode = UserInst->getOpcode();
  switch (Opcode) {
  case Instruction::Load: {
    LoadInst *LI = cast<LoadInst>(UserInst);
    return (LI->getPointerOperand() == Scalar);
  }
  case Instruction::Store: {
    StoreInst *SI = cast<StoreInst>(UserInst);
    return (SI->getPointerOperand() == Scalar);
  }
  case Instruction::Call: {
    CallInst *CI = cast<CallInst>(UserInst);
    Intrinsic::ID ID = getIntrinsicIDForCall(CI, TLI);
    if (hasVectorInstrinsicScalarOpd(ID, 1)) {
      return (CI->getArgOperand(1) == Scalar);
    }
  }
  default:
    return false;
  }
}

/// \returns the AA location that is being access by the instruction.
static MemoryLocation getLocation(Instruction *I, AliasAnalysis *AA) {
  if (StoreInst *SI = dyn_cast<StoreInst>(I))
    return MemoryLocation::get(SI);
  if (LoadInst *LI = dyn_cast<LoadInst>(I))
    return MemoryLocation::get(LI);
  return MemoryLocation();
}

/// \returns True if the instruction is not a volatile or atomic load/store.
static bool isSimple(Instruction *I) {
  if (LoadInst *LI = dyn_cast<LoadInst>(I))
    return LI->isSimple();
  if (StoreInst *SI = dyn_cast<StoreInst>(I))
    return SI->isSimple();
  if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(I))
    return !MI->isVolatile();
  return true;
}

/// Bottom Up SLP Vectorizer.
class BoUpSLP {
public:
  typedef SmallVector<Value *, 8> ValueList;
  typedef SmallVector<Instruction *, 16> InstrList;
  typedef SmallPtrSet<Value *, 16> ValueSet;
  typedef SmallVector<StoreInst *, 8> StoreList;

  BoUpSLP(Function *Func, ScalarEvolution *Se, TargetTransformInfo *Tti,
          TargetLibraryInfo *TLi, AliasAnalysis *Aa, LoopInfo *Li,
          DominatorTree *Dt, AssumptionCache *AC)
      : NumLoadsWantToKeepOrder(0), NumLoadsWantToChangeOrder(0), F(Func),
        SE(Se), TTI(Tti), TLI(TLi), AA(Aa), LI(Li), DT(Dt),
        Builder(Se->getContext()) {
    CodeMetrics::collectEphemeralValues(F, AC, EphValues);
  }

  /// \brief Vectorize the tree that starts with the elements in \p VL.
  /// Returns the vectorized root.
  Value *vectorizeTree();

  /// \returns the cost incurred by unwanted spills and fills, caused by
  /// holding live values over call sites.
  int getSpillCost();

  /// \returns the vectorization cost of the subtree that starts at \p VL.
  /// A negative number means that this is profitable.
  int getTreeCost();

  /// Construct a vectorizable tree that starts at \p Roots, ignoring users for
  /// the purpose of scheduling and extraction in the \p UserIgnoreLst.
  void buildTree(ArrayRef<Value *> Roots,
                 ArrayRef<Value *> UserIgnoreLst = None);

  /// Clear the internal data structures that are created by 'buildTree'.
  void deleteTree() {
    VectorizableTree.clear();
    ScalarToTreeEntry.clear();
    MustGather.clear();
    ExternalUses.clear();
    NumLoadsWantToKeepOrder = 0;
    NumLoadsWantToChangeOrder = 0;
    for (auto &Iter : BlocksSchedules) {
      BlockScheduling *BS = Iter.second.get();
      BS->clear();
    }
  }

  /// \returns true if the memory operations A and B are consecutive.
  bool isConsecutiveAccess(Value *A, Value *B, const DataLayout &DL);

  /// \brief Perform LICM and CSE on the newly generated gather sequences.
  void optimizeGatherSequence();

  /// \returns true if it is benefitial to reverse the vector order.
  bool shouldReorder() const {
    return NumLoadsWantToChangeOrder > NumLoadsWantToKeepOrder;
  }

private:
  struct TreeEntry;

  /// \returns the cost of the vectorizable entry.
  int getEntryCost(TreeEntry *E);

  /// This is the recursive part of buildTree.
  void buildTree_rec(ArrayRef<Value *> Roots, unsigned Depth);

  /// Vectorize a single entry in the tree.
  Value *vectorizeTree(TreeEntry *E);

  /// Vectorize a single entry in the tree, starting in \p VL.
  Value *vectorizeTree(ArrayRef<Value *> VL);

  /// \returns the pointer to the vectorized value if \p VL is already
  /// vectorized, or NULL. They may happen in cycles.
  Value *alreadyVectorized(ArrayRef<Value *> VL) const;

  /// \brief Take the pointer operand from the Load/Store instruction.
  /// \returns NULL if this is not a valid Load/Store instruction.
  static Value *getPointerOperand(Value *I);

  /// \brief Take the address space operand from the Load/Store instruction.
  /// \returns -1 if this is not a valid Load/Store instruction.
  static unsigned getAddressSpaceOperand(Value *I);

  /// \returns the scalarization cost for this type. Scalarization in this
  /// context means the creation of vectors from a group of scalars.
  int getGatherCost(Type *Ty);

  /// \returns the scalarization cost for this list of values. Assuming that
  /// this subtree gets vectorized, we may need to extract the values from the
  /// roots. This method calculates the cost of extracting the values.
  int getGatherCost(ArrayRef<Value *> VL);

  /// \brief Set the Builder insert point to one after the last instruction in
  /// the bundle
  void setInsertPointAfterBundle(ArrayRef<Value *> VL);

  /// \returns a vector from a collection of scalars in \p VL.
  Value *Gather(ArrayRef<Value *> VL, VectorType *Ty);

  /// \returns whether the VectorizableTree is fully vectoriable and will
  /// be beneficial even the tree height is tiny.
  bool isFullyVectorizableTinyTree();

  /// \reorder commutative operands in alt shuffle if they result in
  ///  vectorized code.
  void reorderAltShuffleOperands(ArrayRef<Value *> VL,
                                 SmallVectorImpl<Value *> &Left,
                                 SmallVectorImpl<Value *> &Right);
  /// \reorder commutative operands to get better probability of
  /// generating vectorized code.
  void reorderInputsAccordingToOpcode(ArrayRef<Value *> VL,
                                      SmallVectorImpl<Value *> &Left,
                                      SmallVectorImpl<Value *> &Right);
  struct TreeEntry {
    TreeEntry() : Scalars(), VectorizedValue(nullptr),
    NeedToGather(0) {}

    /// \returns true if the scalars in VL are equal to this entry.
    bool isSame(ArrayRef<Value *> VL) const {
      assert(VL.size() == Scalars.size() && "Invalid size");
      return std::equal(VL.begin(), VL.end(), Scalars.begin());
    }

    /// A vector of scalars.
    ValueList Scalars;

    /// The Scalars are vectorized into this value. It is initialized to Null.
    Value *VectorizedValue;

    /// Do we need to gather this sequence ?
    bool NeedToGather;
  };

  /// Create a new VectorizableTree entry.
  TreeEntry *newTreeEntry(ArrayRef<Value *> VL, bool Vectorized) {
    VectorizableTree.emplace_back();
    int idx = VectorizableTree.size() - 1;
    TreeEntry *Last = &VectorizableTree[idx];
    Last->Scalars.insert(Last->Scalars.begin(), VL.begin(), VL.end());
    Last->NeedToGather = !Vectorized;
    if (Vectorized) {
      for (int i = 0, e = VL.size(); i != e; ++i) {
        assert(!ScalarToTreeEntry.count(VL[i]) && "Scalar already in tree!");
        ScalarToTreeEntry[VL[i]] = idx;
      }
    } else {
      MustGather.insert(VL.begin(), VL.end());
    }
    return Last;
  }
  
  /// -- Vectorization State --
  /// Holds all of the tree entries.
  std::vector<TreeEntry> VectorizableTree;

  /// Maps a specific scalar to its tree entry.
  SmallDenseMap<Value*, int> ScalarToTreeEntry;

  /// A list of scalars that we found that we need to keep as scalars.
  ValueSet MustGather;

  /// This POD struct describes one external user in the vectorized tree.
  struct ExternalUser {
    ExternalUser (Value *S, llvm::User *U, int L) :
      Scalar(S), User(U), Lane(L){};
    // Which scalar in our function.
    Value *Scalar;
    // Which user that uses the scalar.
    llvm::User *User;
    // Which lane does the scalar belong to.
    int Lane;
  };
  typedef SmallVector<ExternalUser, 16> UserList;

  /// Checks if two instructions may access the same memory.
  ///
  /// \p Loc1 is the location of \p Inst1. It is passed explicitly because it
  /// is invariant in the calling loop.
  bool isAliased(const MemoryLocation &Loc1, Instruction *Inst1,
                 Instruction *Inst2) {

    // First check if the result is already in the cache.
    AliasCacheKey key = std::make_pair(Inst1, Inst2);
    Optional<bool> &result = AliasCache[key];
    if (result.hasValue()) {
      return result.getValue();
    }
    MemoryLocation Loc2 = getLocation(Inst2, AA);
    bool aliased = true;
    if (Loc1.Ptr && Loc2.Ptr && isSimple(Inst1) && isSimple(Inst2)) {
      // Do the alias check.
      aliased = AA->alias(Loc1, Loc2);
    }
    // Store the result in the cache.
    result = aliased;
    return aliased;
  }

  typedef std::pair<Instruction *, Instruction *> AliasCacheKey;

  /// Cache for alias results.
  /// TODO: consider moving this to the AliasAnalysis itself.
  DenseMap<AliasCacheKey, Optional<bool>> AliasCache;

  /// Removes an instruction from its block and eventually deletes it.
  /// It's like Instruction::eraseFromParent() except that the actual deletion
  /// is delayed until BoUpSLP is destructed.
  /// This is required to ensure that there are no incorrect collisions in the
  /// AliasCache, which can happen if a new instruction is allocated at the
  /// same address as a previously deleted instruction.
  void eraseInstruction(Instruction *I) {
    I->removeFromParent();
    I->dropAllReferences();
    DeletedInstructions.push_back(std::unique_ptr<Instruction>(I));
  }

  /// Temporary store for deleted instructions. Instructions will be deleted
  /// eventually when the BoUpSLP is destructed.
  SmallVector<std::unique_ptr<Instruction>, 8> DeletedInstructions;

  /// A list of values that need to extracted out of the tree.
  /// This list holds pairs of (Internal Scalar : External User).
  UserList ExternalUses;

  /// Values used only by @llvm.assume calls.
  SmallPtrSet<const Value *, 32> EphValues;

  /// Holds all of the instructions that we gathered.
  SetVector<Instruction *> GatherSeq;
  /// A list of blocks that we are going to CSE.
  SetVector<BasicBlock *> CSEBlocks;

  /// Contains all scheduling relevant data for an instruction.
  /// A ScheduleData either represents a single instruction or a member of an
  /// instruction bundle (= a group of instructions which is combined into a
  /// vector instruction).
  struct ScheduleData {

    // The initial value for the dependency counters. It means that the
    // dependencies are not calculated yet.
    enum { InvalidDeps = -1 };

    ScheduleData()
        : Inst(nullptr), FirstInBundle(nullptr), NextInBundle(nullptr),
          NextLoadStore(nullptr), SchedulingRegionID(0), SchedulingPriority(0),
          Dependencies(InvalidDeps), UnscheduledDeps(InvalidDeps),
          UnscheduledDepsInBundle(InvalidDeps), IsScheduled(false) {}

    void init(int BlockSchedulingRegionID) {
      FirstInBundle = this;
      NextInBundle = nullptr;
      NextLoadStore = nullptr;
      IsScheduled = false;
      SchedulingRegionID = BlockSchedulingRegionID;
      UnscheduledDepsInBundle = UnscheduledDeps;
      clearDependencies();
    }

    /// Returns true if the dependency information has been calculated.
    bool hasValidDependencies() const { return Dependencies != InvalidDeps; }

    /// Returns true for single instructions and for bundle representatives
    /// (= the head of a bundle).
    bool isSchedulingEntity() const { return FirstInBundle == this; }

    /// Returns true if it represents an instruction bundle and not only a
    /// single instruction.
    bool isPartOfBundle() const {
      return NextInBundle != nullptr || FirstInBundle != this;
    }

    /// Returns true if it is ready for scheduling, i.e. it has no more
    /// unscheduled depending instructions/bundles.
    bool isReady() const {
      assert(isSchedulingEntity() &&
             "can't consider non-scheduling entity for ready list");
      return UnscheduledDepsInBundle == 0 && !IsScheduled;
    }

    /// Modifies the number of unscheduled dependencies, also updating it for
    /// the whole bundle.
    int incrementUnscheduledDeps(int Incr) {
      UnscheduledDeps += Incr;
      return FirstInBundle->UnscheduledDepsInBundle += Incr;
    }

    /// Sets the number of unscheduled dependencies to the number of
    /// dependencies.
    void resetUnscheduledDeps() {
      incrementUnscheduledDeps(Dependencies - UnscheduledDeps);
    }

    /// Clears all dependency information.
    void clearDependencies() {
      Dependencies = InvalidDeps;
      resetUnscheduledDeps();
      MemoryDependencies.clear();
    }

    void dump(raw_ostream &os) const {
      if (!isSchedulingEntity()) {
        os << "/ " << *Inst;
      } else if (NextInBundle) {
        os << '[' << *Inst;
        ScheduleData *SD = NextInBundle;
        while (SD) {
          os << ';' << *SD->Inst;
          SD = SD->NextInBundle;
        }
        os << ']';
      } else {
        os << *Inst;
      }
    }

    Instruction *Inst;

    /// Points to the head in an instruction bundle (and always to this for
    /// single instructions).
    ScheduleData *FirstInBundle;

    /// Single linked list of all instructions in a bundle. Null if it is a
    /// single instruction.
    ScheduleData *NextInBundle;

    /// Single linked list of all memory instructions (e.g. load, store, call)
    /// in the block - until the end of the scheduling region.
    ScheduleData *NextLoadStore;

    /// The dependent memory instructions.
    /// This list is derived on demand in calculateDependencies().
    SmallVector<ScheduleData *, 4> MemoryDependencies;

    /// This ScheduleData is in the current scheduling region if this matches
    /// the current SchedulingRegionID of BlockScheduling.
    int SchedulingRegionID;

    /// Used for getting a "good" final ordering of instructions.
    int SchedulingPriority;

    /// The number of dependencies. Constitutes of the number of users of the
    /// instruction plus the number of dependent memory instructions (if any).
    /// This value is calculated on demand.
    /// If InvalidDeps, the number of dependencies is not calculated yet.
    ///
    int Dependencies;

    /// The number of dependencies minus the number of dependencies of scheduled
    /// instructions. As soon as this is zero, the instruction/bundle gets ready
    /// for scheduling.
    /// Note that this is negative as long as Dependencies is not calculated.
    int UnscheduledDeps;

    /// The sum of UnscheduledDeps in a bundle. Equals to UnscheduledDeps for
    /// single instructions.
    int UnscheduledDepsInBundle;

    /// True if this instruction is scheduled (or considered as scheduled in the
    /// dry-run).
    bool IsScheduled;
  };

#ifndef NDEBUG
  friend raw_ostream &operator<<(raw_ostream &os,
                                 const BoUpSLP::ScheduleData &SD);
#endif

  /// Contains all scheduling data for a basic block.
  ///
  struct BlockScheduling {

    BlockScheduling(BasicBlock *BB)
        : BB(BB), ChunkSize(BB->size()), ChunkPos(ChunkSize),
          ScheduleStart(nullptr), ScheduleEnd(nullptr),
          FirstLoadStoreInRegion(nullptr), LastLoadStoreInRegion(nullptr),
          // Make sure that the initial SchedulingRegionID is greater than the
          // initial SchedulingRegionID in ScheduleData (which is 0).
          SchedulingRegionID(1) {}

    void clear() {
      ReadyInsts.clear();
      ScheduleStart = nullptr;
      ScheduleEnd = nullptr;
      FirstLoadStoreInRegion = nullptr;
      LastLoadStoreInRegion = nullptr;

      // Make a new scheduling region, i.e. all existing ScheduleData is not
      // in the new region yet.
      ++SchedulingRegionID;
    }

    ScheduleData *getScheduleData(Value *V) {
      ScheduleData *SD = ScheduleDataMap[V];
      if (SD && SD->SchedulingRegionID == SchedulingRegionID)
        return SD;
      return nullptr;
    }

    bool isInSchedulingRegion(ScheduleData *SD) {
      return SD->SchedulingRegionID == SchedulingRegionID;
    }

    /// Marks an instruction as scheduled and puts all dependent ready
    /// instructions into the ready-list.
    template <typename ReadyListType>
    void schedule(ScheduleData *SD, ReadyListType &ReadyList) {
      SD->IsScheduled = true;
      DEBUG(dbgs() << "SLP:   schedule " << *SD << "\n");

      ScheduleData *BundleMember = SD;
      while (BundleMember) {
        // Handle the def-use chain dependencies.
        for (Use &U : BundleMember->Inst->operands()) {
          ScheduleData *OpDef = getScheduleData(U.get());
          if (OpDef && OpDef->hasValidDependencies() &&
              OpDef->incrementUnscheduledDeps(-1) == 0) {
            // There are no more unscheduled dependencies after decrementing,
            // so we can put the dependent instruction into the ready list.
            ScheduleData *DepBundle = OpDef->FirstInBundle;
            assert(!DepBundle->IsScheduled &&
                   "already scheduled bundle gets ready");
            ReadyList.insert(DepBundle);
            DEBUG(dbgs() << "SLP:    gets ready (def): " << *DepBundle << "\n");
          }
        }
        // Handle the memory dependencies.
        for (ScheduleData *MemoryDepSD : BundleMember->MemoryDependencies) {
          if (MemoryDepSD->incrementUnscheduledDeps(-1) == 0) {
            // There are no more unscheduled dependencies after decrementing,
            // so we can put the dependent instruction into the ready list.
            ScheduleData *DepBundle = MemoryDepSD->FirstInBundle;
            assert(!DepBundle->IsScheduled &&
                   "already scheduled bundle gets ready");
            ReadyList.insert(DepBundle);
            DEBUG(dbgs() << "SLP:    gets ready (mem): " << *DepBundle << "\n");
          }
        }
        BundleMember = BundleMember->NextInBundle;
      }
    }

    /// Put all instructions into the ReadyList which are ready for scheduling.
    template <typename ReadyListType>
    void initialFillReadyList(ReadyListType &ReadyList) {
      for (auto *I = ScheduleStart; I != ScheduleEnd; I = I->getNextNode()) {
        ScheduleData *SD = getScheduleData(I);
        if (SD->isSchedulingEntity() && SD->isReady()) {
          ReadyList.insert(SD);
          DEBUG(dbgs() << "SLP:    initially in ready list: " << *I << "\n");
        }
      }
    }

    /// Checks if a bundle of instructions can be scheduled, i.e. has no
    /// cyclic dependencies. This is only a dry-run, no instructions are
    /// actually moved at this stage.
    bool tryScheduleBundle(ArrayRef<Value *> VL, BoUpSLP *SLP);

    /// Un-bundles a group of instructions.
    void cancelScheduling(ArrayRef<Value *> VL);

    /// Extends the scheduling region so that V is inside the region.
    void extendSchedulingRegion(Value *V);

    /// Initialize the ScheduleData structures for new instructions in the
    /// scheduling region.
    void initScheduleData(Instruction *FromI, Instruction *ToI,
                          ScheduleData *PrevLoadStore,
                          ScheduleData *NextLoadStore);

    /// Updates the dependency information of a bundle and of all instructions/
    /// bundles which depend on the original bundle.
    void calculateDependencies(ScheduleData *SD, bool InsertInReadyList,
                               BoUpSLP *SLP);

    /// Sets all instruction in the scheduling region to un-scheduled.
    void resetSchedule();

    BasicBlock *BB;

    /// Simple memory allocation for ScheduleData.
    std::vector<std::unique_ptr<ScheduleData[]>> ScheduleDataChunks;

    /// The size of a ScheduleData array in ScheduleDataChunks.
    int ChunkSize;

    /// The allocator position in the current chunk, which is the last entry
    /// of ScheduleDataChunks.
    int ChunkPos;

    /// Attaches ScheduleData to Instruction.
    /// Note that the mapping survives during all vectorization iterations, i.e.
    /// ScheduleData structures are recycled.
    DenseMap<Value *, ScheduleData *> ScheduleDataMap;

    struct ReadyList : SmallVector<ScheduleData *, 8> {
      void insert(ScheduleData *SD) { push_back(SD); }
    };

    /// The ready-list for scheduling (only used for the dry-run).
    ReadyList ReadyInsts;

    /// The first instruction of the scheduling region.
    Instruction *ScheduleStart;

    /// The first instruction _after_ the scheduling region.
    Instruction *ScheduleEnd;

    /// The first memory accessing instruction in the scheduling region
    /// (can be null).
    ScheduleData *FirstLoadStoreInRegion;

    /// The last memory accessing instruction in the scheduling region
    /// (can be null).
    ScheduleData *LastLoadStoreInRegion;

    /// The ID of the scheduling region. For a new vectorization iteration this
    /// is incremented which "removes" all ScheduleData from the region.
    int SchedulingRegionID;
  };

  /// Attaches the BlockScheduling structures to basic blocks.
  MapVector<BasicBlock *, std::unique_ptr<BlockScheduling>> BlocksSchedules;

  /// Performs the "real" scheduling. Done before vectorization is actually
  /// performed in a basic block.
  void scheduleBlock(BlockScheduling *BS);

  /// List of users to ignore during scheduling and that don't need extracting.
  ArrayRef<Value *> UserIgnoreList;

  // Number of load-bundles, which contain consecutive loads.
  int NumLoadsWantToKeepOrder;

  // Number of load-bundles of size 2, which are consecutive loads if reversed.
  int NumLoadsWantToChangeOrder;

  // Analysis and block reference.
  Function *F;
  ScalarEvolution *SE;
  TargetTransformInfo *TTI;
  TargetLibraryInfo *TLI;
  AliasAnalysis *AA;
  LoopInfo *LI;
  DominatorTree *DT;
  /// Instruction builder to construct the vectorized tree.
  IRBuilder<> Builder;
};

#ifndef NDEBUG
raw_ostream &operator<<(raw_ostream &os, const BoUpSLP::ScheduleData &SD) {
  SD.dump(os);
  return os;
}
#endif

void BoUpSLP::buildTree(ArrayRef<Value *> Roots,
                        ArrayRef<Value *> UserIgnoreLst) {
  deleteTree();
  UserIgnoreList = UserIgnoreLst;
  if (!getSameType(Roots))
    return;
  buildTree_rec(Roots, 0);

  // Collect the values that we need to extract from the tree.
  for (int EIdx = 0, EE = VectorizableTree.size(); EIdx < EE; ++EIdx) {
    TreeEntry *Entry = &VectorizableTree[EIdx];

    // For each lane:
    for (int Lane = 0, LE = Entry->Scalars.size(); Lane != LE; ++Lane) {
      Value *Scalar = Entry->Scalars[Lane];

      // No need to handle users of gathered values.
      if (Entry->NeedToGather)
        continue;

      for (User *U : Scalar->users()) {
        DEBUG(dbgs() << "SLP: Checking user:" << *U << ".\n");

        Instruction *UserInst = dyn_cast<Instruction>(U);
        if (!UserInst)
          continue;

        // Skip in-tree scalars that become vectors
        if (ScalarToTreeEntry.count(U)) {
          int Idx = ScalarToTreeEntry[U];
          TreeEntry *UseEntry = &VectorizableTree[Idx];
          Value *UseScalar = UseEntry->Scalars[0];
          // Some in-tree scalars will remain as scalar in vectorized
          // instructions. If that is the case, the one in Lane 0 will
          // be used.
          if (UseScalar != U ||
              !InTreeUserNeedToExtract(Scalar, UserInst, TLI)) {
            DEBUG(dbgs() << "SLP: \tInternal user will be removed:" << *U
                         << ".\n");
            assert(!VectorizableTree[Idx].NeedToGather && "Bad state");
            continue;
          }
        }

        // Ignore users in the user ignore list.
        if (std::find(UserIgnoreList.begin(), UserIgnoreList.end(), UserInst) !=
            UserIgnoreList.end())
          continue;

        DEBUG(dbgs() << "SLP: Need to extract:" << *U << " from lane " <<
              Lane << " from " << *Scalar << ".\n");
        ExternalUses.push_back(ExternalUser(Scalar, U, Lane));
      }
    }
  }
}


void BoUpSLP::buildTree_rec(ArrayRef<Value *> VL, unsigned Depth) {
  bool SameTy = getSameType(VL); (void)SameTy;
  bool isAltShuffle = false;
  assert(SameTy && "Invalid types!");

  if (Depth == RecursionMaxDepth) {
    DEBUG(dbgs() << "SLP: Gathering due to max recursion depth.\n");
    newTreeEntry(VL, false);
    return;
  }

  // Don't handle vectors.
  if (VL[0]->getType()->isVectorTy()) {
    DEBUG(dbgs() << "SLP: Gathering due to vector type.\n");
    newTreeEntry(VL, false);
    return;
  }

  if (StoreInst *SI = dyn_cast<StoreInst>(VL[0]))
    if (SI->getValueOperand()->getType()->isVectorTy()) {
      DEBUG(dbgs() << "SLP: Gathering due to store vector type.\n");
      newTreeEntry(VL, false);
      return;
    }
  unsigned Opcode = getSameOpcode(VL);

  // Check that this shuffle vector refers to the alternate
  // sequence of opcodes.
  if (Opcode == Instruction::ShuffleVector) {
    Instruction *I0 = dyn_cast<Instruction>(VL[0]);
    unsigned Op = I0->getOpcode();
    if (Op != Instruction::ShuffleVector)
      isAltShuffle = true;
  }

  // If all of the operands are identical or constant we have a simple solution.
  if (allConstant(VL) || isSplat(VL) || !getSameBlock(VL) || !Opcode) {
    DEBUG(dbgs() << "SLP: Gathering due to C,S,B,O. \n");
    newTreeEntry(VL, false);
    return;
  }

  // We now know that this is a vector of instructions of the same type from
  // the same block.

  // Don't vectorize ephemeral values.
  for (unsigned i = 0, e = VL.size(); i != e; ++i) {
    if (EphValues.count(VL[i])) {
      DEBUG(dbgs() << "SLP: The instruction (" << *VL[i] <<
            ") is ephemeral.\n");
      newTreeEntry(VL, false);
      return;
    }
  }

  // Check if this is a duplicate of another entry.
  if (ScalarToTreeEntry.count(VL[0])) {
    int Idx = ScalarToTreeEntry[VL[0]];
    TreeEntry *E = &VectorizableTree[Idx];
    for (unsigned i = 0, e = VL.size(); i != e; ++i) {
      DEBUG(dbgs() << "SLP: \tChecking bundle: " << *VL[i] << ".\n");
      if (E->Scalars[i] != VL[i]) {
        DEBUG(dbgs() << "SLP: Gathering due to partial overlap.\n");
        newTreeEntry(VL, false);
        return;
      }
    }
    DEBUG(dbgs() << "SLP: Perfect diamond merge at " << *VL[0] << ".\n");
    return;
  }

  // Check that none of the instructions in the bundle are already in the tree.
  for (unsigned i = 0, e = VL.size(); i != e; ++i) {
    if (ScalarToTreeEntry.count(VL[i])) {
      DEBUG(dbgs() << "SLP: The instruction (" << *VL[i] <<
            ") is already in tree.\n");
      newTreeEntry(VL, false);
      return;
    }
  }

  // If any of the scalars is marked as a value that needs to stay scalar then
  // we need to gather the scalars.
  for (unsigned i = 0, e = VL.size(); i != e; ++i) {
    if (MustGather.count(VL[i])) {
      DEBUG(dbgs() << "SLP: Gathering due to gathered scalar.\n");
      newTreeEntry(VL, false);
      return;
    }
  }

  // Check that all of the users of the scalars that we want to vectorize are
  // schedulable.
  Instruction *VL0 = cast<Instruction>(VL[0]);
  BasicBlock *BB = cast<Instruction>(VL0)->getParent();

  if (!DT->isReachableFromEntry(BB)) {
    // Don't go into unreachable blocks. They may contain instructions with
    // dependency cycles which confuse the final scheduling.
    DEBUG(dbgs() << "SLP: bundle in unreachable block.\n");
    newTreeEntry(VL, false);
    return;
  }
  
  // Check that every instructions appears once in this bundle.
  for (unsigned i = 0, e = VL.size(); i < e; ++i)
    for (unsigned j = i+1; j < e; ++j)
      if (VL[i] == VL[j]) {
        DEBUG(dbgs() << "SLP: Scalar used twice in bundle.\n");
        newTreeEntry(VL, false);
        return;
      }

  auto &BSRef = BlocksSchedules[BB];
  if (!BSRef) {
    BSRef = llvm::make_unique<BlockScheduling>(BB);
  }
  BlockScheduling &BS = *BSRef.get();

  if (!BS.tryScheduleBundle(VL, this)) {
    DEBUG(dbgs() << "SLP: We are not able to schedule this bundle!\n");
    BS.cancelScheduling(VL);
    newTreeEntry(VL, false);
    return;
  }
  DEBUG(dbgs() << "SLP: We are able to schedule this bundle.\n");

  switch (Opcode) {
    case Instruction::PHI: {
      PHINode *PH = dyn_cast<PHINode>(VL0);

      // Check for terminator values (e.g. invoke).
      for (unsigned j = 0; j < VL.size(); ++j)
        for (unsigned i = 0, e = PH->getNumIncomingValues(); i < e; ++i) {
          TerminatorInst *Term = dyn_cast<TerminatorInst>(
              cast<PHINode>(VL[j])->getIncomingValueForBlock(PH->getIncomingBlock(i)));
          if (Term) {
            DEBUG(dbgs() << "SLP: Need to swizzle PHINodes (TerminatorInst use).\n");
            BS.cancelScheduling(VL);
            newTreeEntry(VL, false);
            return;
          }
        }

      newTreeEntry(VL, true);
      DEBUG(dbgs() << "SLP: added a vector of PHINodes.\n");

      for (unsigned i = 0, e = PH->getNumIncomingValues(); i < e; ++i) {
        ValueList Operands;
        // Prepare the operand vector.
        for (unsigned j = 0; j < VL.size(); ++j)
          Operands.push_back(cast<PHINode>(VL[j])->getIncomingValueForBlock(
              PH->getIncomingBlock(i)));

        buildTree_rec(Operands, Depth + 1);
      }
      return;
    }
    case Instruction::ExtractElement: {
      bool Reuse = CanReuseExtract(VL);
      if (Reuse) {
        DEBUG(dbgs() << "SLP: Reusing extract sequence.\n");
      } else {
        BS.cancelScheduling(VL);
      }
      newTreeEntry(VL, Reuse);
      return;
    }
    case Instruction::Load: {
      // Check if the loads are consecutive or of we need to swizzle them.
      for (unsigned i = 0, e = VL.size() - 1; i < e; ++i) {
        LoadInst *L = cast<LoadInst>(VL[i]);
        if (!L->isSimple()) {
          BS.cancelScheduling(VL);
          newTreeEntry(VL, false);
          DEBUG(dbgs() << "SLP: Gathering non-simple loads.\n");
          return;
        }
        const DataLayout &DL = F->getParent()->getDataLayout();
        if (!isConsecutiveAccess(VL[i], VL[i + 1], DL)) {
          if (VL.size() == 2 && isConsecutiveAccess(VL[1], VL[0], DL)) {
            ++NumLoadsWantToChangeOrder;
          }
          BS.cancelScheduling(VL);
          newTreeEntry(VL, false);
          DEBUG(dbgs() << "SLP: Gathering non-consecutive loads.\n");
          return;
        }
      }
      ++NumLoadsWantToKeepOrder;
      newTreeEntry(VL, true);
      DEBUG(dbgs() << "SLP: added a vector of loads.\n");
      return;
    }
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
    case Instruction::FPExt:
    case Instruction::PtrToInt:
    case Instruction::IntToPtr:
    case Instruction::SIToFP:
    case Instruction::UIToFP:
    case Instruction::Trunc:
    case Instruction::FPTrunc:
    case Instruction::BitCast: {
      Type *SrcTy = VL0->getOperand(0)->getType();
      for (unsigned i = 0; i < VL.size(); ++i) {
        Type *Ty = cast<Instruction>(VL[i])->getOperand(0)->getType();
        if (Ty != SrcTy || !isValidElementType(Ty)) {
          BS.cancelScheduling(VL);
          newTreeEntry(VL, false);
          DEBUG(dbgs() << "SLP: Gathering casts with different src types.\n");
          return;
        }
      }
      newTreeEntry(VL, true);
      DEBUG(dbgs() << "SLP: added a vector of casts.\n");

      for (unsigned i = 0, e = VL0->getNumOperands(); i < e; ++i) {
        ValueList Operands;
        // Prepare the operand vector.
        for (unsigned j = 0; j < VL.size(); ++j)
          Operands.push_back(cast<Instruction>(VL[j])->getOperand(i));

        buildTree_rec(Operands, Depth+1);
      }
      return;
    }
    case Instruction::ICmp:
    case Instruction::FCmp: {
      // Check that all of the compares have the same predicate.
      CmpInst::Predicate P0 = cast<CmpInst>(VL0)->getPredicate();
      Type *ComparedTy = cast<Instruction>(VL[0])->getOperand(0)->getType();
      for (unsigned i = 1, e = VL.size(); i < e; ++i) {
        CmpInst *Cmp = cast<CmpInst>(VL[i]);
        if (Cmp->getPredicate() != P0 ||
            Cmp->getOperand(0)->getType() != ComparedTy) {
          BS.cancelScheduling(VL);
          newTreeEntry(VL, false);
          DEBUG(dbgs() << "SLP: Gathering cmp with different predicate.\n");
          return;
        }
      }

      newTreeEntry(VL, true);
      DEBUG(dbgs() << "SLP: added a vector of compares.\n");

      for (unsigned i = 0, e = VL0->getNumOperands(); i < e; ++i) {
        ValueList Operands;
        // Prepare the operand vector.
        for (unsigned j = 0; j < VL.size(); ++j)
          Operands.push_back(cast<Instruction>(VL[j])->getOperand(i));

        buildTree_rec(Operands, Depth+1);
      }
      return;
    }
    case Instruction::Select:
    case Instruction::Add:
    case Instruction::FAdd:
    case Instruction::Sub:
    case Instruction::FSub:
    case Instruction::Mul:
    case Instruction::FMul:
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::FDiv:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::FRem:
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor: {
      newTreeEntry(VL, true);
      DEBUG(dbgs() << "SLP: added a vector of bin op.\n");

      // Sort operands of the instructions so that each side is more likely to
      // have the same opcode.
      if (isa<BinaryOperator>(VL0) && VL0->isCommutative()) {
        ValueList Left, Right;
        reorderInputsAccordingToOpcode(VL, Left, Right);
        buildTree_rec(Left, Depth + 1);
        buildTree_rec(Right, Depth + 1);
        return;
      }

      for (unsigned i = 0, e = VL0->getNumOperands(); i < e; ++i) {
        ValueList Operands;
        // Prepare the operand vector.
        for (unsigned j = 0; j < VL.size(); ++j)
          Operands.push_back(cast<Instruction>(VL[j])->getOperand(i));

        buildTree_rec(Operands, Depth+1);
      }
      return;
    }
    case Instruction::GetElementPtr: {
      // We don't combine GEPs with complicated (nested) indexing.
      for (unsigned j = 0; j < VL.size(); ++j) {
        if (cast<Instruction>(VL[j])->getNumOperands() != 2) {
          DEBUG(dbgs() << "SLP: not-vectorizable GEP (nested indexes).\n");
          BS.cancelScheduling(VL);
          newTreeEntry(VL, false);
          return;
        }
      }

      // We can't combine several GEPs into one vector if they operate on
      // different types.
      Type *Ty0 = cast<Instruction>(VL0)->getOperand(0)->getType();
      for (unsigned j = 0; j < VL.size(); ++j) {
        Type *CurTy = cast<Instruction>(VL[j])->getOperand(0)->getType();
        if (Ty0 != CurTy) {
          DEBUG(dbgs() << "SLP: not-vectorizable GEP (different types).\n");
          BS.cancelScheduling(VL);
          newTreeEntry(VL, false);
          return;
        }
      }

      // We don't combine GEPs with non-constant indexes.
      for (unsigned j = 0; j < VL.size(); ++j) {
        auto Op = cast<Instruction>(VL[j])->getOperand(1);
        if (!isa<ConstantInt>(Op)) {
          DEBUG(
              dbgs() << "SLP: not-vectorizable GEP (non-constant indexes).\n");
          BS.cancelScheduling(VL);
          newTreeEntry(VL, false);
          return;
        }
      }

      newTreeEntry(VL, true);
      DEBUG(dbgs() << "SLP: added a vector of GEPs.\n");
      for (unsigned i = 0, e = 2; i < e; ++i) {
        ValueList Operands;
        // Prepare the operand vector.
        for (unsigned j = 0; j < VL.size(); ++j)
          Operands.push_back(cast<Instruction>(VL[j])->getOperand(i));

        buildTree_rec(Operands, Depth + 1);
      }
      return;
    }
    case Instruction::Store: {
      const DataLayout &DL = F->getParent()->getDataLayout();
      // Check if the stores are consecutive or of we need to swizzle them.
      for (unsigned i = 0, e = VL.size() - 1; i < e; ++i)
        if (!isConsecutiveAccess(VL[i], VL[i + 1], DL)) {
          BS.cancelScheduling(VL);
          newTreeEntry(VL, false);
          DEBUG(dbgs() << "SLP: Non-consecutive store.\n");
          return;
        }

      newTreeEntry(VL, true);
      DEBUG(dbgs() << "SLP: added a vector of stores.\n");

      ValueList Operands;
      for (unsigned j = 0; j < VL.size(); ++j)
        Operands.push_back(cast<Instruction>(VL[j])->getOperand(0));

      buildTree_rec(Operands, Depth + 1);
      return;
    }
    case Instruction::Call: {
      // Check if the calls are all to the same vectorizable intrinsic.
      CallInst *CI = cast<CallInst>(VL[0]);
      // Check if this is an Intrinsic call or something that can be
      // represented by an intrinsic call
      Intrinsic::ID ID = getIntrinsicIDForCall(CI, TLI);
      if (!isTriviallyVectorizable(ID)) {
        BS.cancelScheduling(VL);
        newTreeEntry(VL, false);
        DEBUG(dbgs() << "SLP: Non-vectorizable call.\n");
        return;
      }
      Function *Int = CI->getCalledFunction();
      Value *A1I = nullptr;
      if (hasVectorInstrinsicScalarOpd(ID, 1))
        A1I = CI->getArgOperand(1);
      for (unsigned i = 1, e = VL.size(); i != e; ++i) {
        CallInst *CI2 = dyn_cast<CallInst>(VL[i]);
        if (!CI2 || CI2->getCalledFunction() != Int ||
            getIntrinsicIDForCall(CI2, TLI) != ID) {
          BS.cancelScheduling(VL);
          newTreeEntry(VL, false);
          DEBUG(dbgs() << "SLP: mismatched calls:" << *CI << "!=" << *VL[i]
                       << "\n");
          return;
        }
        // ctlz,cttz and powi are special intrinsics whose second argument
        // should be same in order for them to be vectorized.
        if (hasVectorInstrinsicScalarOpd(ID, 1)) {
          Value *A1J = CI2->getArgOperand(1);
          if (A1I != A1J) {
            BS.cancelScheduling(VL);
            newTreeEntry(VL, false);
            DEBUG(dbgs() << "SLP: mismatched arguments in call:" << *CI
                         << " argument "<< A1I<<"!=" << A1J
                         << "\n");
            return;
          }
        }
      }

      newTreeEntry(VL, true);
      for (unsigned i = 0, e = CI->getNumArgOperands(); i != e; ++i) {
        ValueList Operands;
        // Prepare the operand vector.
        for (unsigned j = 0; j < VL.size(); ++j) {
          CallInst *CI2 = dyn_cast<CallInst>(VL[j]);
          Operands.push_back(CI2->getArgOperand(i));
        }
        buildTree_rec(Operands, Depth + 1);
      }
      return;
    }
    case Instruction::ShuffleVector: {
      // If this is not an alternate sequence of opcode like add-sub
      // then do not vectorize this instruction.
      if (!isAltShuffle) {
        BS.cancelScheduling(VL);
        newTreeEntry(VL, false);
        DEBUG(dbgs() << "SLP: ShuffleVector are not vectorized.\n");
        return;
      }
      newTreeEntry(VL, true);
      DEBUG(dbgs() << "SLP: added a ShuffleVector op.\n");

      // Reorder operands if reordering would enable vectorization.
      if (isa<BinaryOperator>(VL0)) {
        ValueList Left, Right;
        reorderAltShuffleOperands(VL, Left, Right);
        buildTree_rec(Left, Depth + 1);
        buildTree_rec(Right, Depth + 1);
        return;
      }

      for (unsigned i = 0, e = VL0->getNumOperands(); i < e; ++i) {
        ValueList Operands;
        // Prepare the operand vector.
        for (unsigned j = 0; j < VL.size(); ++j)
          Operands.push_back(cast<Instruction>(VL[j])->getOperand(i));

        buildTree_rec(Operands, Depth + 1);
      }
      return;
    }
    default:
      BS.cancelScheduling(VL);
      newTreeEntry(VL, false);
      DEBUG(dbgs() << "SLP: Gathering unknown instruction.\n");
      return;
  }
}

int BoUpSLP::getEntryCost(TreeEntry *E) {
  ArrayRef<Value*> VL = E->Scalars;

  Type *ScalarTy = VL[0]->getType();
  if (StoreInst *SI = dyn_cast<StoreInst>(VL[0]))
    ScalarTy = SI->getValueOperand()->getType();
  VectorType *VecTy = VectorType::get(ScalarTy, VL.size());

  if (E->NeedToGather) {
    if (allConstant(VL))
      return 0;
    if (isSplat(VL)) {
      return TTI->getShuffleCost(TargetTransformInfo::SK_Broadcast, VecTy, 0);
    }
    return getGatherCost(E->Scalars);
  }
  unsigned Opcode = getSameOpcode(VL);
  assert(Opcode && getSameType(VL) && getSameBlock(VL) && "Invalid VL");
  Instruction *VL0 = cast<Instruction>(VL[0]);
  switch (Opcode) {
    case Instruction::PHI: {
      return 0;
    }
    case Instruction::ExtractElement: {
      if (CanReuseExtract(VL)) {
        int DeadCost = 0;
        for (unsigned i = 0, e = VL.size(); i < e; ++i) {
          ExtractElementInst *E = cast<ExtractElementInst>(VL[i]);
          if (E->hasOneUse())
            // Take credit for instruction that will become dead.
            DeadCost +=
                TTI->getVectorInstrCost(Instruction::ExtractElement, VecTy, i);
        }
        return -DeadCost;
      }
      return getGatherCost(VecTy);
    }
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
    case Instruction::FPExt:
    case Instruction::PtrToInt:
    case Instruction::IntToPtr:
    case Instruction::SIToFP:
    case Instruction::UIToFP:
    case Instruction::Trunc:
    case Instruction::FPTrunc:
    case Instruction::BitCast: {
      Type *SrcTy = VL0->getOperand(0)->getType();

      // Calculate the cost of this instruction.
      int ScalarCost = VL.size() * TTI->getCastInstrCost(VL0->getOpcode(),
                                                         VL0->getType(), SrcTy);

      VectorType *SrcVecTy = VectorType::get(SrcTy, VL.size());
      int VecCost = TTI->getCastInstrCost(VL0->getOpcode(), VecTy, SrcVecTy);
      return VecCost - ScalarCost;
    }
    case Instruction::FCmp:
    case Instruction::ICmp:
    case Instruction::Select:
    case Instruction::Add:
    case Instruction::FAdd:
    case Instruction::Sub:
    case Instruction::FSub:
    case Instruction::Mul:
    case Instruction::FMul:
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::FDiv:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::FRem:
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor: {
      // Calculate the cost of this instruction.
      int ScalarCost = 0;
      int VecCost = 0;
      if (Opcode == Instruction::FCmp || Opcode == Instruction::ICmp ||
          Opcode == Instruction::Select) {
        VectorType *MaskTy = VectorType::get(Builder.getInt1Ty(), VL.size());
        ScalarCost = VecTy->getNumElements() *
        TTI->getCmpSelInstrCost(Opcode, ScalarTy, Builder.getInt1Ty());
        VecCost = TTI->getCmpSelInstrCost(Opcode, VecTy, MaskTy);
      } else {
        // Certain instructions can be cheaper to vectorize if they have a
        // constant second vector operand.
        TargetTransformInfo::OperandValueKind Op1VK =
            TargetTransformInfo::OK_AnyValue;
        TargetTransformInfo::OperandValueKind Op2VK =
            TargetTransformInfo::OK_UniformConstantValue;
        TargetTransformInfo::OperandValueProperties Op1VP =
            TargetTransformInfo::OP_None;
        TargetTransformInfo::OperandValueProperties Op2VP =
            TargetTransformInfo::OP_None;

        // If all operands are exactly the same ConstantInt then set the
        // operand kind to OK_UniformConstantValue.
        // If instead not all operands are constants, then set the operand kind
        // to OK_AnyValue. If all operands are constants but not the same,
        // then set the operand kind to OK_NonUniformConstantValue.
        ConstantInt *CInt = nullptr;
        for (unsigned i = 0; i < VL.size(); ++i) {
          const Instruction *I = cast<Instruction>(VL[i]);
          if (!isa<ConstantInt>(I->getOperand(1))) {
            Op2VK = TargetTransformInfo::OK_AnyValue;
            break;
          }
          if (i == 0) {
            CInt = cast<ConstantInt>(I->getOperand(1));
            continue;
          }
          if (Op2VK == TargetTransformInfo::OK_UniformConstantValue &&
              CInt != cast<ConstantInt>(I->getOperand(1)))
            Op2VK = TargetTransformInfo::OK_NonUniformConstantValue;
        }
        // FIXME: Currently cost of model modification for division by
        // power of 2 is handled only for X86. Add support for other targets.
        if (Op2VK == TargetTransformInfo::OK_UniformConstantValue && CInt &&
            CInt->getValue().isPowerOf2())
          Op2VP = TargetTransformInfo::OP_PowerOf2;

        ScalarCost = VecTy->getNumElements() *
                     TTI->getArithmeticInstrCost(Opcode, ScalarTy, Op1VK, Op2VK,
                                                 Op1VP, Op2VP);
        VecCost = TTI->getArithmeticInstrCost(Opcode, VecTy, Op1VK, Op2VK,
                                              Op1VP, Op2VP);
      }
      return VecCost - ScalarCost;
    }
    case Instruction::GetElementPtr: {
      TargetTransformInfo::OperandValueKind Op1VK =
          TargetTransformInfo::OK_AnyValue;
      TargetTransformInfo::OperandValueKind Op2VK =
          TargetTransformInfo::OK_UniformConstantValue;

      int ScalarCost =
          VecTy->getNumElements() *
          TTI->getArithmeticInstrCost(Instruction::Add, ScalarTy, Op1VK, Op2VK);
      int VecCost =
          TTI->getArithmeticInstrCost(Instruction::Add, VecTy, Op1VK, Op2VK);

      return VecCost - ScalarCost;
    }
    case Instruction::Load: {
      // Cost of wide load - cost of scalar loads.
      int ScalarLdCost = VecTy->getNumElements() *
      TTI->getMemoryOpCost(Instruction::Load, ScalarTy, 1, 0);
      int VecLdCost = TTI->getMemoryOpCost(Instruction::Load, VecTy, 1, 0);
      return VecLdCost - ScalarLdCost;
    }
    case Instruction::Store: {
      // We know that we can merge the stores. Calculate the cost.
      int ScalarStCost = VecTy->getNumElements() *
      TTI->getMemoryOpCost(Instruction::Store, ScalarTy, 1, 0);
      int VecStCost = TTI->getMemoryOpCost(Instruction::Store, VecTy, 1, 0);
      return VecStCost - ScalarStCost;
    }
    case Instruction::Call: {
      CallInst *CI = cast<CallInst>(VL0);
      Intrinsic::ID ID = getIntrinsicIDForCall(CI, TLI);

      // Calculate the cost of the scalar and vector calls.
      SmallVector<Type*, 4> ScalarTys, VecTys;
      for (unsigned op = 0, opc = CI->getNumArgOperands(); op!= opc; ++op) {
        ScalarTys.push_back(CI->getArgOperand(op)->getType());
        VecTys.push_back(VectorType::get(CI->getArgOperand(op)->getType(),
                                         VecTy->getNumElements()));
      }

      int ScalarCallCost = VecTy->getNumElements() *
          TTI->getIntrinsicInstrCost(ID, ScalarTy, ScalarTys);

      int VecCallCost = TTI->getIntrinsicInstrCost(ID, VecTy, VecTys);

      DEBUG(dbgs() << "SLP: Call cost "<< VecCallCost - ScalarCallCost
            << " (" << VecCallCost  << "-" <<  ScalarCallCost << ")"
            << " for " << *CI << "\n");

      return VecCallCost - ScalarCallCost;
    }
    case Instruction::ShuffleVector: {
      TargetTransformInfo::OperandValueKind Op1VK =
          TargetTransformInfo::OK_AnyValue;
      TargetTransformInfo::OperandValueKind Op2VK =
          TargetTransformInfo::OK_AnyValue;
      int ScalarCost = 0;
      int VecCost = 0;
      for (unsigned i = 0; i < VL.size(); ++i) {
        Instruction *I = cast<Instruction>(VL[i]);
        if (!I)
          break;
        ScalarCost +=
            TTI->getArithmeticInstrCost(I->getOpcode(), ScalarTy, Op1VK, Op2VK);
      }
      // VecCost is equal to sum of the cost of creating 2 vectors
      // and the cost of creating shuffle.
      Instruction *I0 = cast<Instruction>(VL[0]);
      VecCost =
          TTI->getArithmeticInstrCost(I0->getOpcode(), VecTy, Op1VK, Op2VK);
      Instruction *I1 = cast<Instruction>(VL[1]);
      VecCost +=
          TTI->getArithmeticInstrCost(I1->getOpcode(), VecTy, Op1VK, Op2VK);
      VecCost +=
          TTI->getShuffleCost(TargetTransformInfo::SK_Alternate, VecTy, 0);
      return VecCost - ScalarCost;
    }
    default:
      llvm_unreachable("Unknown instruction");
  }
}

bool BoUpSLP::isFullyVectorizableTinyTree() {
  DEBUG(dbgs() << "SLP: Check whether the tree with height " <<
        VectorizableTree.size() << " is fully vectorizable .\n");

  // We only handle trees of height 2.
  if (VectorizableTree.size() != 2)
    return false;

  // Handle splat and all-constants stores.
  if (!VectorizableTree[0].NeedToGather &&
      (allConstant(VectorizableTree[1].Scalars) ||
       isSplat(VectorizableTree[1].Scalars)))
    return true;

  // Gathering cost would be too much for tiny trees.
  if (VectorizableTree[0].NeedToGather || VectorizableTree[1].NeedToGather)
    return false;

  return true;
}

int BoUpSLP::getSpillCost() {
  // Walk from the bottom of the tree to the top, tracking which values are
  // live. When we see a call instruction that is not part of our tree,
  // query TTI to see if there is a cost to keeping values live over it
  // (for example, if spills and fills are required).
  unsigned BundleWidth = VectorizableTree.front().Scalars.size();
  int Cost = 0;

  SmallPtrSet<Instruction*, 4> LiveValues;
  Instruction *PrevInst = nullptr; 

  for (unsigned N = 0; N < VectorizableTree.size(); ++N) {
    Instruction *Inst = dyn_cast<Instruction>(VectorizableTree[N].Scalars[0]);
    if (!Inst)
      continue;

    if (!PrevInst) {
      PrevInst = Inst;
      continue;
    }

    DEBUG(
      dbgs() << "SLP: #LV: " << LiveValues.size();
      for (auto *X : LiveValues)
        dbgs() << " " << X->getName();
      dbgs() << ", Looking at ";
      Inst->dump();
      );

    // Update LiveValues.
    LiveValues.erase(PrevInst);
    for (auto &J : PrevInst->operands()) {
      if (isa<Instruction>(&*J) && ScalarToTreeEntry.count(&*J))
        LiveValues.insert(cast<Instruction>(&*J));
    }    

    // Now find the sequence of instructions between PrevInst and Inst.
    BasicBlock::reverse_iterator InstIt(Inst), PrevInstIt(PrevInst);
    --PrevInstIt;
    while (InstIt != PrevInstIt) {
      if (PrevInstIt == PrevInst->getParent()->rend()) {
        PrevInstIt = Inst->getParent()->rbegin();
        continue;
      }

      if (isa<CallInst>(&*PrevInstIt) && &*PrevInstIt != PrevInst) {
        SmallVector<Type*, 4> V;
        for (auto *II : LiveValues)
          V.push_back(VectorType::get(II->getType(), BundleWidth));
        Cost += TTI->getCostOfKeepingLiveOverCall(V);
      }

      ++PrevInstIt;
    }

    PrevInst = Inst;
  }

  DEBUG(dbgs() << "SLP: SpillCost=" << Cost << "\n");
  return Cost;
}

int BoUpSLP::getTreeCost() {
  int Cost = 0;
  DEBUG(dbgs() << "SLP: Calculating cost for tree of size " <<
        VectorizableTree.size() << ".\n");

  // We only vectorize tiny trees if it is fully vectorizable.
  if (VectorizableTree.size() < 3 && !isFullyVectorizableTinyTree()) {
    if (VectorizableTree.empty()) {
      assert(!ExternalUses.size() && "We should not have any external users");
    }
    return INT_MAX;
  }

  unsigned BundleWidth = VectorizableTree[0].Scalars.size();

  for (unsigned i = 0, e = VectorizableTree.size(); i != e; ++i) {
    int C = getEntryCost(&VectorizableTree[i]);
    DEBUG(dbgs() << "SLP: Adding cost " << C << " for bundle that starts with "
          << *VectorizableTree[i].Scalars[0] << " .\n");
    Cost += C;
  }

  SmallSet<Value *, 16> ExtractCostCalculated;
  int ExtractCost = 0;
  for (UserList::iterator I = ExternalUses.begin(), E = ExternalUses.end();
       I != E; ++I) {
    // We only add extract cost once for the same scalar.
    if (!ExtractCostCalculated.insert(I->Scalar).second)
      continue;

    // Uses by ephemeral values are free (because the ephemeral value will be
    // removed prior to code generation, and so the extraction will be
    // removed as well).
    if (EphValues.count(I->User))
      continue;

    VectorType *VecTy = VectorType::get(I->Scalar->getType(), BundleWidth);
    ExtractCost += TTI->getVectorInstrCost(Instruction::ExtractElement, VecTy,
                                           I->Lane);
  }

  Cost += getSpillCost();

  DEBUG(dbgs() << "SLP: Total Cost " << Cost + ExtractCost<< ".\n");
  return  Cost + ExtractCost;
}

int BoUpSLP::getGatherCost(Type *Ty) {
  int Cost = 0;
  for (unsigned i = 0, e = cast<VectorType>(Ty)->getNumElements(); i < e; ++i)
    Cost += TTI->getVectorInstrCost(Instruction::InsertElement, Ty, i);
  return Cost;
}

int BoUpSLP::getGatherCost(ArrayRef<Value *> VL) {
  // Find the type of the operands in VL.
  Type *ScalarTy = VL[0]->getType();
  if (StoreInst *SI = dyn_cast<StoreInst>(VL[0]))
    ScalarTy = SI->getValueOperand()->getType();
  VectorType *VecTy = VectorType::get(ScalarTy, VL.size());
  // Find the cost of inserting/extracting values from the vector.
  return getGatherCost(VecTy);
}

Value *BoUpSLP::getPointerOperand(Value *I) {
  if (LoadInst *LI = dyn_cast<LoadInst>(I))
    return LI->getPointerOperand();
  if (StoreInst *SI = dyn_cast<StoreInst>(I))
    return SI->getPointerOperand();
  return nullptr;
}

unsigned BoUpSLP::getAddressSpaceOperand(Value *I) {
  if (LoadInst *L = dyn_cast<LoadInst>(I))
    return L->getPointerAddressSpace();
  if (StoreInst *S = dyn_cast<StoreInst>(I))
    return S->getPointerAddressSpace();
  return -1;
}

bool BoUpSLP::isConsecutiveAccess(Value *A, Value *B, const DataLayout &DL) {
  Value *PtrA = getPointerOperand(A);
  Value *PtrB = getPointerOperand(B);
  unsigned ASA = getAddressSpaceOperand(A);
  unsigned ASB = getAddressSpaceOperand(B);

  // Check that the address spaces match and that the pointers are valid.
  if (!PtrA || !PtrB || (ASA != ASB))
    return false;

  // Make sure that A and B are different pointers of the same type.
  if (PtrA == PtrB || PtrA->getType() != PtrB->getType())
    return false;

  unsigned PtrBitWidth = DL.getPointerSizeInBits(ASA);
  Type *Ty = cast<PointerType>(PtrA->getType())->getElementType();
  APInt Size(PtrBitWidth, DL.getTypeStoreSize(Ty));

  APInt OffsetA(PtrBitWidth, 0), OffsetB(PtrBitWidth, 0);
  PtrA = PtrA->stripAndAccumulateInBoundsConstantOffsets(DL, OffsetA);
  PtrB = PtrB->stripAndAccumulateInBoundsConstantOffsets(DL, OffsetB);

  APInt OffsetDelta = OffsetB - OffsetA;

  // Check if they are based on the same pointer. That makes the offsets
  // sufficient.
  if (PtrA == PtrB)
    return OffsetDelta == Size;

  // Compute the necessary base pointer delta to have the necessary final delta
  // equal to the size.
  APInt BaseDelta = Size - OffsetDelta;

  // Otherwise compute the distance with SCEV between the base pointers.
  const SCEV *PtrSCEVA = SE->getSCEV(PtrA);
  const SCEV *PtrSCEVB = SE->getSCEV(PtrB);
  const SCEV *C = SE->getConstant(BaseDelta);
  const SCEV *X = SE->getAddExpr(PtrSCEVA, C);
  return X == PtrSCEVB;
}

// Reorder commutative operations in alternate shuffle if the resulting vectors
// are consecutive loads. This would allow us to vectorize the tree.
// If we have something like-
// load a[0] - load b[0]
// load b[1] + load a[1]
// load a[2] - load b[2]
// load a[3] + load b[3]
// Reordering the second load b[1]  load a[1] would allow us to vectorize this
// code.
void BoUpSLP::reorderAltShuffleOperands(ArrayRef<Value *> VL,
                                        SmallVectorImpl<Value *> &Left,
                                        SmallVectorImpl<Value *> &Right) {
  const DataLayout &DL = F->getParent()->getDataLayout();

  // Push left and right operands of binary operation into Left and Right
  for (unsigned i = 0, e = VL.size(); i < e; ++i) {
    Left.push_back(cast<Instruction>(VL[i])->getOperand(0));
    Right.push_back(cast<Instruction>(VL[i])->getOperand(1));
  }

  // Reorder if we have a commutative operation and consecutive access
  // are on either side of the alternate instructions.
  for (unsigned j = 0; j < VL.size() - 1; ++j) {
    if (LoadInst *L = dyn_cast<LoadInst>(Left[j])) {
      if (LoadInst *L1 = dyn_cast<LoadInst>(Right[j + 1])) {
        Instruction *VL1 = cast<Instruction>(VL[j]);
        Instruction *VL2 = cast<Instruction>(VL[j + 1]);
        if (isConsecutiveAccess(L, L1, DL) && VL1->isCommutative()) {
          std::swap(Left[j], Right[j]);
          continue;
        } else if (isConsecutiveAccess(L, L1, DL) && VL2->isCommutative()) {
          std::swap(Left[j + 1], Right[j + 1]);
          continue;
        }
        // else unchanged
      }
    }
    if (LoadInst *L = dyn_cast<LoadInst>(Right[j])) {
      if (LoadInst *L1 = dyn_cast<LoadInst>(Left[j + 1])) {
        Instruction *VL1 = cast<Instruction>(VL[j]);
        Instruction *VL2 = cast<Instruction>(VL[j + 1]);
        if (isConsecutiveAccess(L, L1, DL) && VL1->isCommutative()) {
          std::swap(Left[j], Right[j]);
          continue;
        } else if (isConsecutiveAccess(L, L1, DL) && VL2->isCommutative()) {
          std::swap(Left[j + 1], Right[j + 1]);
          continue;
        }
        // else unchanged
      }
    }
  }
}

void BoUpSLP::reorderInputsAccordingToOpcode(ArrayRef<Value *> VL,
                                             SmallVectorImpl<Value *> &Left,
                                             SmallVectorImpl<Value *> &Right) {

  SmallVector<Value *, 16> OrigLeft, OrigRight;

  bool AllSameOpcodeLeft = true;
  bool AllSameOpcodeRight = true;
  for (unsigned i = 0, e = VL.size(); i != e; ++i) {
    Instruction *I = cast<Instruction>(VL[i]);
    Value *VLeft = I->getOperand(0);
    Value *VRight = I->getOperand(1);

    OrigLeft.push_back(VLeft);
    OrigRight.push_back(VRight);

    Instruction *ILeft = dyn_cast<Instruction>(VLeft);
    Instruction *IRight = dyn_cast<Instruction>(VRight);

    // Check whether all operands on one side have the same opcode. In this case
    // we want to preserve the original order and not make things worse by
    // reordering.
    if (i && AllSameOpcodeLeft && ILeft) {
      if (Instruction *PLeft = dyn_cast<Instruction>(OrigLeft[i - 1])) {
        if (PLeft->getOpcode() != ILeft->getOpcode())
          AllSameOpcodeLeft = false;
      } else
        AllSameOpcodeLeft = false;
    }
    if (i && AllSameOpcodeRight && IRight) {
      if (Instruction *PRight = dyn_cast<Instruction>(OrigRight[i - 1])) {
        if (PRight->getOpcode() != IRight->getOpcode())
          AllSameOpcodeRight = false;
      } else
        AllSameOpcodeRight = false;
    }

    // Sort two opcodes. In the code below we try to preserve the ability to use
    // broadcast of values instead of individual inserts.
    // vl1 = load
    // vl2 = phi
    // vr1 = load
    // vr2 = vr2
    //    = vl1 x vr1
    //    = vl2 x vr2
    // If we just sorted according to opcode we would leave the first line in
    // tact but we would swap vl2 with vr2 because opcode(phi) > opcode(load).
    //    = vl1 x vr1
    //    = vr2 x vl2
    // Because vr2 and vr1 are from the same load we loose the opportunity of a
    // broadcast for the packed right side in the backend: we have [vr1, vl2]
    // instead of [vr1, vr2=vr1].
    if (ILeft && IRight) {
      if (!i && ILeft->getOpcode() > IRight->getOpcode()) {
        Left.push_back(IRight);
        Right.push_back(ILeft);
      } else if (i && ILeft->getOpcode() > IRight->getOpcode() &&
                 Right[i - 1] != IRight) {
        // Try not to destroy a broad cast for no apparent benefit.
        Left.push_back(IRight);
        Right.push_back(ILeft);
      } else if (i && ILeft->getOpcode() == IRight->getOpcode() &&
                 Right[i - 1] == ILeft) {
        // Try preserve broadcasts.
        Left.push_back(IRight);
        Right.push_back(ILeft);
      } else if (i && ILeft->getOpcode() == IRight->getOpcode() &&
                 Left[i - 1] == IRight) {
        // Try preserve broadcasts.
        Left.push_back(IRight);
        Right.push_back(ILeft);
      } else {
        Left.push_back(ILeft);
        Right.push_back(IRight);
      }
      continue;
    }
    // One opcode, put the instruction on the right.
    if (ILeft) {
      Left.push_back(VRight);
      Right.push_back(ILeft);
      continue;
    }
    Left.push_back(VLeft);
    Right.push_back(VRight);
  }

  bool LeftBroadcast = isSplat(Left);
  bool RightBroadcast = isSplat(Right);

  // If operands end up being broadcast return this operand order.
  if (LeftBroadcast || RightBroadcast)
    return;

  // Don't reorder if the operands where good to begin.
  if (AllSameOpcodeRight || AllSameOpcodeLeft) {
    Left = OrigLeft;
    Right = OrigRight;
  }

  const DataLayout &DL = F->getParent()->getDataLayout();

  // Finally check if we can get longer vectorizable chain by reordering
  // without breaking the good operand order detected above.
  // E.g. If we have something like-
  // load a[0]  load b[0]
  // load b[1]  load a[1]
  // load a[2]  load b[2]
  // load a[3]  load b[3]
  // Reordering the second load b[1]  load a[1] would allow us to vectorize
  // this code and we still retain AllSameOpcode property.
  // FIXME: This load reordering might break AllSameOpcode in some rare cases
  // such as-
  // add a[0],c[0]  load b[0]
  // add a[1],c[2]  load b[1]
  // b[2]           load b[2]
  // add a[3],c[3]  load b[3]
  for (unsigned j = 0; j < VL.size() - 1; ++j) {
    if (LoadInst *L = dyn_cast<LoadInst>(Left[j])) {
      if (LoadInst *L1 = dyn_cast<LoadInst>(Right[j + 1])) {
        if (isConsecutiveAccess(L, L1, DL)) {
          std::swap(Left[j + 1], Right[j + 1]);
          continue;
        }
      }
    }
    if (LoadInst *L = dyn_cast<LoadInst>(Right[j])) {
      if (LoadInst *L1 = dyn_cast<LoadInst>(Left[j + 1])) {
        if (isConsecutiveAccess(L, L1, DL)) {
          std::swap(Left[j + 1], Right[j + 1]);
          continue;
        }
      }
    }
    // else unchanged
  }
}

void BoUpSLP::setInsertPointAfterBundle(ArrayRef<Value *> VL) {
  Instruction *VL0 = cast<Instruction>(VL[0]);
  BasicBlock::iterator NextInst = VL0;
  ++NextInst;
  Builder.SetInsertPoint(VL0->getParent(), NextInst);
  Builder.SetCurrentDebugLocation(VL0->getDebugLoc());
}

Value *BoUpSLP::Gather(ArrayRef<Value *> VL, VectorType *Ty) {
  Value *Vec = UndefValue::get(Ty);
  // Generate the 'InsertElement' instruction.
  for (unsigned i = 0; i < Ty->getNumElements(); ++i) {
    Vec = Builder.CreateInsertElement(Vec, VL[i], Builder.getInt32(i));
    if (Instruction *Insrt = dyn_cast<Instruction>(Vec)) {
      GatherSeq.insert(Insrt);
      CSEBlocks.insert(Insrt->getParent());

      // Add to our 'need-to-extract' list.
      if (ScalarToTreeEntry.count(VL[i])) {
        int Idx = ScalarToTreeEntry[VL[i]];
        TreeEntry *E = &VectorizableTree[Idx];
        // Find which lane we need to extract.
        int FoundLane = -1;
        for (unsigned Lane = 0, LE = VL.size(); Lane != LE; ++Lane) {
          // Is this the lane of the scalar that we are looking for ?
          if (E->Scalars[Lane] == VL[i]) {
            FoundLane = Lane;
            break;
          }
        }
        assert(FoundLane >= 0 && "Could not find the correct lane");
        ExternalUses.push_back(ExternalUser(VL[i], Insrt, FoundLane));
      }
    }
  }

  return Vec;
}

Value *BoUpSLP::alreadyVectorized(ArrayRef<Value *> VL) const {
  SmallDenseMap<Value*, int>::const_iterator Entry
    = ScalarToTreeEntry.find(VL[0]);
  if (Entry != ScalarToTreeEntry.end()) {
    int Idx = Entry->second;
    const TreeEntry *En = &VectorizableTree[Idx];
    if (En->isSame(VL) && En->VectorizedValue)
      return En->VectorizedValue;
  }
  return nullptr;
}

Value *BoUpSLP::vectorizeTree(ArrayRef<Value *> VL) {
  if (ScalarToTreeEntry.count(VL[0])) {
    int Idx = ScalarToTreeEntry[VL[0]];
    TreeEntry *E = &VectorizableTree[Idx];
    if (E->isSame(VL))
      return vectorizeTree(E);
  }

  Type *ScalarTy = VL[0]->getType();
  if (StoreInst *SI = dyn_cast<StoreInst>(VL[0]))
    ScalarTy = SI->getValueOperand()->getType();
  VectorType *VecTy = VectorType::get(ScalarTy, VL.size());

  return Gather(VL, VecTy);
}

Value *BoUpSLP::vectorizeTree(TreeEntry *E) {
  IRBuilder<>::InsertPointGuard Guard(Builder);

  if (E->VectorizedValue) {
    DEBUG(dbgs() << "SLP: Diamond merged for " << *E->Scalars[0] << ".\n");
    return E->VectorizedValue;
  }

  Instruction *VL0 = cast<Instruction>(E->Scalars[0]);
  Type *ScalarTy = VL0->getType();
  if (StoreInst *SI = dyn_cast<StoreInst>(VL0))
    ScalarTy = SI->getValueOperand()->getType();
  VectorType *VecTy = VectorType::get(ScalarTy, E->Scalars.size());

  if (E->NeedToGather) {
    setInsertPointAfterBundle(E->Scalars);
    return Gather(E->Scalars, VecTy);
  }

  const DataLayout &DL = F->getParent()->getDataLayout();
  unsigned Opcode = getSameOpcode(E->Scalars);

  switch (Opcode) {
    case Instruction::PHI: {
      PHINode *PH = dyn_cast<PHINode>(VL0);
      Builder.SetInsertPoint(PH->getParent()->getFirstNonPHI());
      Builder.SetCurrentDebugLocation(PH->getDebugLoc());
      PHINode *NewPhi = Builder.CreatePHI(VecTy, PH->getNumIncomingValues());
      E->VectorizedValue = NewPhi;

      // PHINodes may have multiple entries from the same block. We want to
      // visit every block once.
      SmallSet<BasicBlock*, 4> VisitedBBs;

      for (unsigned i = 0, e = PH->getNumIncomingValues(); i < e; ++i) {
        ValueList Operands;
        BasicBlock *IBB = PH->getIncomingBlock(i);

        if (!VisitedBBs.insert(IBB).second) {
          NewPhi->addIncoming(NewPhi->getIncomingValueForBlock(IBB), IBB);
          continue;
        }

        // Prepare the operand vector.
        for (Value *V : E->Scalars)
          Operands.push_back(cast<PHINode>(V)->getIncomingValueForBlock(IBB));

        Builder.SetInsertPoint(IBB->getTerminator());
        Builder.SetCurrentDebugLocation(PH->getDebugLoc());
        Value *Vec = vectorizeTree(Operands);
        NewPhi->addIncoming(Vec, IBB);
      }

      assert(NewPhi->getNumIncomingValues() == PH->getNumIncomingValues() &&
             "Invalid number of incoming values");
      return NewPhi;
    }

    case Instruction::ExtractElement: {
      if (CanReuseExtract(E->Scalars)) {
        Value *V = VL0->getOperand(0);
        E->VectorizedValue = V;
        return V;
      }
      return Gather(E->Scalars, VecTy);
    }
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
    case Instruction::FPExt:
    case Instruction::PtrToInt:
    case Instruction::IntToPtr:
    case Instruction::SIToFP:
    case Instruction::UIToFP:
    case Instruction::Trunc:
    case Instruction::FPTrunc:
    case Instruction::BitCast: {
      ValueList INVL;
      for (Value *V : E->Scalars)
        INVL.push_back(cast<Instruction>(V)->getOperand(0));

      setInsertPointAfterBundle(E->Scalars);

      Value *InVec = vectorizeTree(INVL);

      if (Value *V = alreadyVectorized(E->Scalars))
        return V;

      CastInst *CI = dyn_cast<CastInst>(VL0);
      Value *V = Builder.CreateCast(CI->getOpcode(), InVec, VecTy);
      E->VectorizedValue = V;
      ++NumVectorInstructions;
      return V;
    }
    case Instruction::FCmp:
    case Instruction::ICmp: {
      ValueList LHSV, RHSV;
      for (Value *V : E->Scalars) {
        LHSV.push_back(cast<Instruction>(V)->getOperand(0));
        RHSV.push_back(cast<Instruction>(V)->getOperand(1));
      }

      setInsertPointAfterBundle(E->Scalars);

      Value *L = vectorizeTree(LHSV);
      Value *R = vectorizeTree(RHSV);

      if (Value *V = alreadyVectorized(E->Scalars))
        return V;

      CmpInst::Predicate P0 = cast<CmpInst>(VL0)->getPredicate();
      Value *V;
      if (Opcode == Instruction::FCmp)
        V = Builder.CreateFCmp(P0, L, R);
      else
        V = Builder.CreateICmp(P0, L, R);

      E->VectorizedValue = V;
      ++NumVectorInstructions;
      return V;
    }
    case Instruction::Select: {
      ValueList TrueVec, FalseVec, CondVec;
      for (Value *V : E->Scalars) {
        CondVec.push_back(cast<Instruction>(V)->getOperand(0));
        TrueVec.push_back(cast<Instruction>(V)->getOperand(1));
        FalseVec.push_back(cast<Instruction>(V)->getOperand(2));
      }

      setInsertPointAfterBundle(E->Scalars);

      Value *Cond = vectorizeTree(CondVec);
      Value *True = vectorizeTree(TrueVec);
      Value *False = vectorizeTree(FalseVec);

      if (Value *V = alreadyVectorized(E->Scalars))
        return V;

      Value *V = Builder.CreateSelect(Cond, True, False);
      E->VectorizedValue = V;
      ++NumVectorInstructions;
      return V;
    }
    case Instruction::Add:
    case Instruction::FAdd:
    case Instruction::Sub:
    case Instruction::FSub:
    case Instruction::Mul:
    case Instruction::FMul:
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::FDiv:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::FRem:
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor: {
      ValueList LHSVL, RHSVL;
      if (isa<BinaryOperator>(VL0) && VL0->isCommutative())
        reorderInputsAccordingToOpcode(E->Scalars, LHSVL, RHSVL);
      else
        for (Value *V : E->Scalars) {
          LHSVL.push_back(cast<Instruction>(V)->getOperand(0));
          RHSVL.push_back(cast<Instruction>(V)->getOperand(1));
        }

      setInsertPointAfterBundle(E->Scalars);

      Value *LHS = vectorizeTree(LHSVL);
      Value *RHS = vectorizeTree(RHSVL);

      if (LHS == RHS && isa<Instruction>(LHS)) {
        assert((VL0->getOperand(0) == VL0->getOperand(1)) && "Invalid order");
      }

      if (Value *V = alreadyVectorized(E->Scalars))
        return V;

      BinaryOperator *BinOp = cast<BinaryOperator>(VL0);
      Value *V = Builder.CreateBinOp(BinOp->getOpcode(), LHS, RHS);
      E->VectorizedValue = V;
      propagateIRFlags(E->VectorizedValue, E->Scalars);
      ++NumVectorInstructions;

      if (Instruction *I = dyn_cast<Instruction>(V))
        return propagateMetadata(I, E->Scalars);

      return V;
    }
    case Instruction::Load: {
      // Loads are inserted at the head of the tree because we don't want to
      // sink them all the way down past store instructions.
      setInsertPointAfterBundle(E->Scalars);

      LoadInst *LI = cast<LoadInst>(VL0);
      Type *ScalarLoadTy = LI->getType();
      unsigned AS = LI->getPointerAddressSpace();

      Value *VecPtr = Builder.CreateBitCast(LI->getPointerOperand(),
                                            VecTy->getPointerTo(AS));

      // The pointer operand uses an in-tree scalar so we add the new BitCast to
      // ExternalUses list to make sure that an extract will be generated in the
      // future.
      if (ScalarToTreeEntry.count(LI->getPointerOperand()))
        ExternalUses.push_back(
            ExternalUser(LI->getPointerOperand(), cast<User>(VecPtr), 0));

      unsigned Alignment = LI->getAlignment();
      LI = Builder.CreateLoad(VecPtr);
      if (!Alignment) {
        Alignment = DL.getABITypeAlignment(ScalarLoadTy);
      }
      LI->setAlignment(Alignment);
      E->VectorizedValue = LI;
      ++NumVectorInstructions;
      return propagateMetadata(LI, E->Scalars);
    }
    case Instruction::Store: {
      StoreInst *SI = cast<StoreInst>(VL0);
      unsigned Alignment = SI->getAlignment();
      unsigned AS = SI->getPointerAddressSpace();

      ValueList ValueOp;
      for (Value *V : E->Scalars)
        ValueOp.push_back(cast<StoreInst>(V)->getValueOperand());

      setInsertPointAfterBundle(E->Scalars);

      Value *VecValue = vectorizeTree(ValueOp);
      Value *VecPtr = Builder.CreateBitCast(SI->getPointerOperand(),
                                            VecTy->getPointerTo(AS));
      StoreInst *S = Builder.CreateStore(VecValue, VecPtr);

      // The pointer operand uses an in-tree scalar so we add the new BitCast to
      // ExternalUses list to make sure that an extract will be generated in the
      // future.
      if (ScalarToTreeEntry.count(SI->getPointerOperand()))
        ExternalUses.push_back(
            ExternalUser(SI->getPointerOperand(), cast<User>(VecPtr), 0));

      if (!Alignment) {
        Alignment = DL.getABITypeAlignment(SI->getValueOperand()->getType());
      }
      S->setAlignment(Alignment);
      E->VectorizedValue = S;
      ++NumVectorInstructions;
      return propagateMetadata(S, E->Scalars);
    }
    case Instruction::GetElementPtr: {
      setInsertPointAfterBundle(E->Scalars);

      ValueList Op0VL;
      for (Value *V : E->Scalars)
        Op0VL.push_back(cast<GetElementPtrInst>(V)->getOperand(0));

      Value *Op0 = vectorizeTree(Op0VL);

      std::vector<Value *> OpVecs;
      for (int j = 1, e = cast<GetElementPtrInst>(VL0)->getNumOperands(); j < e;
           ++j) {
        ValueList OpVL;
        for (Value *V : E->Scalars)
          OpVL.push_back(cast<GetElementPtrInst>(V)->getOperand(j));

        Value *OpVec = vectorizeTree(OpVL);
        OpVecs.push_back(OpVec);
      }

      Value *V = Builder.CreateGEP(
          cast<GetElementPtrInst>(VL0)->getSourceElementType(), Op0, OpVecs);
      E->VectorizedValue = V;
      ++NumVectorInstructions;

      if (Instruction *I = dyn_cast<Instruction>(V))
        return propagateMetadata(I, E->Scalars);

      return V;
    }
    case Instruction::Call: {
      CallInst *CI = cast<CallInst>(VL0);
      setInsertPointAfterBundle(E->Scalars);
      Function *FI;
      Intrinsic::ID IID  = Intrinsic::not_intrinsic;
      Value *ScalarArg = nullptr;
      if (CI && (FI = CI->getCalledFunction())) {
        IID = FI->getIntrinsicID();
      }
      std::vector<Value *> OpVecs;
      for (int j = 0, e = CI->getNumArgOperands(); j < e; ++j) {
        ValueList OpVL;
        // ctlz,cttz and powi are special intrinsics whose second argument is
        // a scalar. This argument should not be vectorized.
        if (hasVectorInstrinsicScalarOpd(IID, 1) && j == 1) {
          CallInst *CEI = cast<CallInst>(E->Scalars[0]);
          ScalarArg = CEI->getArgOperand(j);
          OpVecs.push_back(CEI->getArgOperand(j));
          continue;
        }
        for (Value *V : E->Scalars) {
          CallInst *CEI = cast<CallInst>(V);
          OpVL.push_back(CEI->getArgOperand(j));
        }

        Value *OpVec = vectorizeTree(OpVL);
        DEBUG(dbgs() << "SLP: OpVec[" << j << "]: " << *OpVec << "\n");
        OpVecs.push_back(OpVec);
      }

      Module *M = F->getParent();
      Intrinsic::ID ID = getIntrinsicIDForCall(CI, TLI);
      Type *Tys[] = { VectorType::get(CI->getType(), E->Scalars.size()) };
      Function *CF = Intrinsic::getDeclaration(M, ID, Tys);
      Value *V = Builder.CreateCall(CF, OpVecs);

      // The scalar argument uses an in-tree scalar so we add the new vectorized
      // call to ExternalUses list to make sure that an extract will be
      // generated in the future.
      if (ScalarArg && ScalarToTreeEntry.count(ScalarArg))
        ExternalUses.push_back(ExternalUser(ScalarArg, cast<User>(V), 0));

      E->VectorizedValue = V;
      ++NumVectorInstructions;
      return V;
    }
    case Instruction::ShuffleVector: {
      ValueList LHSVL, RHSVL;
      assert(isa<BinaryOperator>(VL0) && "Invalid Shuffle Vector Operand");
      reorderAltShuffleOperands(E->Scalars, LHSVL, RHSVL);
      setInsertPointAfterBundle(E->Scalars);

      Value *LHS = vectorizeTree(LHSVL);
      Value *RHS = vectorizeTree(RHSVL);

      if (Value *V = alreadyVectorized(E->Scalars))
        return V;

      // Create a vector of LHS op1 RHS
      BinaryOperator *BinOp0 = cast<BinaryOperator>(VL0);
      Value *V0 = Builder.CreateBinOp(BinOp0->getOpcode(), LHS, RHS);

      // Create a vector of LHS op2 RHS
      Instruction *VL1 = cast<Instruction>(E->Scalars[1]);
      BinaryOperator *BinOp1 = cast<BinaryOperator>(VL1);
      Value *V1 = Builder.CreateBinOp(BinOp1->getOpcode(), LHS, RHS);

      // Create shuffle to take alternate operations from the vector.
      // Also, gather up odd and even scalar ops to propagate IR flags to
      // each vector operation.
      ValueList OddScalars, EvenScalars;
      unsigned e = E->Scalars.size();
      SmallVector<Constant *, 8> Mask(e);
      for (unsigned i = 0; i < e; ++i) {
        if (i & 1) {
          Mask[i] = Builder.getInt32(e + i);
          OddScalars.push_back(E->Scalars[i]);
        } else {
          Mask[i] = Builder.getInt32(i);
          EvenScalars.push_back(E->Scalars[i]);
        }
      }

      Value *ShuffleMask = ConstantVector::get(Mask);
      propagateIRFlags(V0, EvenScalars);
      propagateIRFlags(V1, OddScalars);

      Value *V = Builder.CreateShuffleVector(V0, V1, ShuffleMask);
      E->VectorizedValue = V;
      ++NumVectorInstructions;
      if (Instruction *I = dyn_cast<Instruction>(V))
        return propagateMetadata(I, E->Scalars);

      return V;
    }
    default:
    llvm_unreachable("unknown inst");
  }
  return nullptr;
}

Value *BoUpSLP::vectorizeTree() {
  
  // All blocks must be scheduled before any instructions are inserted.
  for (auto &BSIter : BlocksSchedules) {
    scheduleBlock(BSIter.second.get());
  }

  Builder.SetInsertPoint(F->getEntryBlock().begin());
  vectorizeTree(&VectorizableTree[0]);

  DEBUG(dbgs() << "SLP: Extracting " << ExternalUses.size() << " values .\n");

  // Extract all of the elements with the external uses.
  for (UserList::iterator it = ExternalUses.begin(), e = ExternalUses.end();
       it != e; ++it) {
    Value *Scalar = it->Scalar;
    llvm::User *User = it->User;

    // Skip users that we already RAUW. This happens when one instruction
    // has multiple uses of the same value.
    if (std::find(Scalar->user_begin(), Scalar->user_end(), User) ==
        Scalar->user_end())
      continue;
    assert(ScalarToTreeEntry.count(Scalar) && "Invalid scalar");

    int Idx = ScalarToTreeEntry[Scalar];
    TreeEntry *E = &VectorizableTree[Idx];
    assert(!E->NeedToGather && "Extracting from a gather list");

    Value *Vec = E->VectorizedValue;
    assert(Vec && "Can't find vectorizable value");

    Value *Lane = Builder.getInt32(it->Lane);
    // Generate extracts for out-of-tree users.
    // Find the insertion point for the extractelement lane.
    if (isa<Instruction>(Vec)){
      if (PHINode *PH = dyn_cast<PHINode>(User)) {
        for (int i = 0, e = PH->getNumIncomingValues(); i != e; ++i) {
          if (PH->getIncomingValue(i) == Scalar) {
            Builder.SetInsertPoint(PH->getIncomingBlock(i)->getTerminator());
            Value *Ex = Builder.CreateExtractElement(Vec, Lane);
            CSEBlocks.insert(PH->getIncomingBlock(i));
            PH->setOperand(i, Ex);
          }
        }
      } else {
        Builder.SetInsertPoint(cast<Instruction>(User));
        Value *Ex = Builder.CreateExtractElement(Vec, Lane);
        CSEBlocks.insert(cast<Instruction>(User)->getParent());
        User->replaceUsesOfWith(Scalar, Ex);
     }
    } else {
      Builder.SetInsertPoint(F->getEntryBlock().begin());
      Value *Ex = Builder.CreateExtractElement(Vec, Lane);
      CSEBlocks.insert(&F->getEntryBlock());
      User->replaceUsesOfWith(Scalar, Ex);
    }

    DEBUG(dbgs() << "SLP: Replaced:" << *User << ".\n");
  }

  // For each vectorized value:
  for (int EIdx = 0, EE = VectorizableTree.size(); EIdx < EE; ++EIdx) {
    TreeEntry *Entry = &VectorizableTree[EIdx];

    // For each lane:
    for (int Lane = 0, LE = Entry->Scalars.size(); Lane != LE; ++Lane) {
      Value *Scalar = Entry->Scalars[Lane];
      // No need to handle users of gathered values.
      if (Entry->NeedToGather)
        continue;

      assert(Entry->VectorizedValue && "Can't find vectorizable value");

      Type *Ty = Scalar->getType();
      if (!Ty->isVoidTy()) {
#ifndef NDEBUG
        for (User *U : Scalar->users()) {
          DEBUG(dbgs() << "SLP: \tvalidating user:" << *U << ".\n");

          assert((ScalarToTreeEntry.count(U) ||
                  // It is legal to replace users in the ignorelist by undef.
                  (std::find(UserIgnoreList.begin(), UserIgnoreList.end(), U) !=
                   UserIgnoreList.end())) &&
                 "Replacing out-of-tree value with undef");
        }
#endif
        Value *Undef = UndefValue::get(Ty);
        Scalar->replaceAllUsesWith(Undef);
      }
      DEBUG(dbgs() << "SLP: \tErasing scalar:" << *Scalar << ".\n");
      eraseInstruction(cast<Instruction>(Scalar));
    }
  }

  Builder.ClearInsertionPoint();

  return VectorizableTree[0].VectorizedValue;
}

void BoUpSLP::optimizeGatherSequence() {
  DEBUG(dbgs() << "SLP: Optimizing " << GatherSeq.size()
        << " gather sequences instructions.\n");
  // LICM InsertElementInst sequences.
  for (SetVector<Instruction *>::iterator it = GatherSeq.begin(),
       e = GatherSeq.end(); it != e; ++it) {
    InsertElementInst *Insert = dyn_cast<InsertElementInst>(*it);

    if (!Insert)
      continue;

    // Check if this block is inside a loop.
    Loop *L = LI->getLoopFor(Insert->getParent());
    if (!L)
      continue;

    // Check if it has a preheader.
    BasicBlock *PreHeader = L->getLoopPreheader();
    if (!PreHeader)
      continue;

    // If the vector or the element that we insert into it are
    // instructions that are defined in this basic block then we can't
    // hoist this instruction.
    Instruction *CurrVec = dyn_cast<Instruction>(Insert->getOperand(0));
    Instruction *NewElem = dyn_cast<Instruction>(Insert->getOperand(1));
    if (CurrVec && L->contains(CurrVec))
      continue;
    if (NewElem && L->contains(NewElem))
      continue;

    // We can hoist this instruction. Move it to the pre-header.
    Insert->moveBefore(PreHeader->getTerminator());
  }

  // Make a list of all reachable blocks in our CSE queue.
  SmallVector<const DomTreeNode *, 8> CSEWorkList;
  CSEWorkList.reserve(CSEBlocks.size());
  for (BasicBlock *BB : CSEBlocks)
    if (DomTreeNode *N = DT->getNode(BB)) {
      assert(DT->isReachableFromEntry(N));
      CSEWorkList.push_back(N);
    }

  // Sort blocks by domination. This ensures we visit a block after all blocks
  // dominating it are visited.
  std::stable_sort(CSEWorkList.begin(), CSEWorkList.end(),
                   [this](const DomTreeNode *A, const DomTreeNode *B) {
    return DT->properlyDominates(A, B);
  });

  // Perform O(N^2) search over the gather sequences and merge identical
  // instructions. TODO: We can further optimize this scan if we split the
  // instructions into different buckets based on the insert lane.
  SmallVector<Instruction *, 16> Visited;
  for (auto I = CSEWorkList.begin(), E = CSEWorkList.end(); I != E; ++I) {
    assert((I == CSEWorkList.begin() || !DT->dominates(*I, *std::prev(I))) &&
           "Worklist not sorted properly!");
    BasicBlock *BB = (*I)->getBlock();
    // For all instructions in blocks containing gather sequences:
    for (BasicBlock::iterator it = BB->begin(), e = BB->end(); it != e;) {
      Instruction *In = it++;
      if (!isa<InsertElementInst>(In) && !isa<ExtractElementInst>(In))
        continue;

      // Check if we can replace this instruction with any of the
      // visited instructions.
      for (SmallVectorImpl<Instruction *>::iterator v = Visited.begin(),
                                                    ve = Visited.end();
           v != ve; ++v) {
        if (In->isIdenticalTo(*v) &&
            DT->dominates((*v)->getParent(), In->getParent())) {
          In->replaceAllUsesWith(*v);
          eraseInstruction(In);
          In = nullptr;
          break;
        }
      }
      if (In) {
        assert(std::find(Visited.begin(), Visited.end(), In) == Visited.end());
        Visited.push_back(In);
      }
    }
  }
  CSEBlocks.clear();
  GatherSeq.clear();
}

// Groups the instructions to a bundle (which is then a single scheduling entity)
// and schedules instructions until the bundle gets ready.
bool BoUpSLP::BlockScheduling::tryScheduleBundle(ArrayRef<Value *> VL,
                                                 BoUpSLP *SLP) {
  if (isa<PHINode>(VL[0]))
    return true;

  // Initialize the instruction bundle.
  Instruction *OldScheduleEnd = ScheduleEnd;
  ScheduleData *PrevInBundle = nullptr;
  ScheduleData *Bundle = nullptr;
  bool ReSchedule = false;
  DEBUG(dbgs() << "SLP:  bundle: " << *VL[0] << "\n");
  for (Value *V : VL) {
    extendSchedulingRegion(V);
    ScheduleData *BundleMember = getScheduleData(V);
    assert(BundleMember &&
           "no ScheduleData for bundle member (maybe not in same basic block)");
    if (BundleMember->IsScheduled) {
      // A bundle member was scheduled as single instruction before and now
      // needs to be scheduled as part of the bundle. We just get rid of the
      // existing schedule.
      DEBUG(dbgs() << "SLP:  reset schedule because " << *BundleMember
                   << " was already scheduled\n");
      ReSchedule = true;
    }
    assert(BundleMember->isSchedulingEntity() &&
           "bundle member already part of other bundle");
    if (PrevInBundle) {
      PrevInBundle->NextInBundle = BundleMember;
    } else {
      Bundle = BundleMember;
    }
    BundleMember->UnscheduledDepsInBundle = 0;
    Bundle->UnscheduledDepsInBundle += BundleMember->UnscheduledDeps;

    // Group the instructions to a bundle.
    BundleMember->FirstInBundle = Bundle;
    PrevInBundle = BundleMember;
  }
  if (ScheduleEnd != OldScheduleEnd) {
    // The scheduling region got new instructions at the lower end (or it is a
    // new region for the first bundle). This makes it necessary to
    // recalculate all dependencies.
    // It is seldom that this needs to be done a second time after adding the
    // initial bundle to the region.
    for (auto *I = ScheduleStart; I != ScheduleEnd; I = I->getNextNode()) {
      ScheduleData *SD = getScheduleData(I);
      SD->clearDependencies();
    }
    ReSchedule = true;
  }
  if (ReSchedule) {
    resetSchedule();
    initialFillReadyList(ReadyInsts);
  }

  DEBUG(dbgs() << "SLP: try schedule bundle " << *Bundle << " in block "
               << BB->getName() << "\n");

  calculateDependencies(Bundle, true, SLP);

  // Now try to schedule the new bundle. As soon as the bundle is "ready" it
  // means that there are no cyclic dependencies and we can schedule it.
  // Note that's important that we don't "schedule" the bundle yet (see
  // cancelScheduling).
  while (!Bundle->isReady() && !ReadyInsts.empty()) {

    ScheduleData *pickedSD = ReadyInsts.back();
    ReadyInsts.pop_back();

    if (pickedSD->isSchedulingEntity() && pickedSD->isReady()) {
      schedule(pickedSD, ReadyInsts);
    }
  }
  return Bundle->isReady();
}

void BoUpSLP::BlockScheduling::cancelScheduling(ArrayRef<Value *> VL) {
  if (isa<PHINode>(VL[0]))
    return;

  ScheduleData *Bundle = getScheduleData(VL[0]);
  DEBUG(dbgs() << "SLP:  cancel scheduling of " << *Bundle << "\n");
  assert(!Bundle->IsScheduled &&
         "Can't cancel bundle which is already scheduled");
  assert(Bundle->isSchedulingEntity() && Bundle->isPartOfBundle() &&
         "tried to unbundle something which is not a bundle");

  // Un-bundle: make single instructions out of the bundle.
  ScheduleData *BundleMember = Bundle;
  while (BundleMember) {
    assert(BundleMember->FirstInBundle == Bundle && "corrupt bundle links");
    BundleMember->FirstInBundle = BundleMember;
    ScheduleData *Next = BundleMember->NextInBundle;
    BundleMember->NextInBundle = nullptr;
    BundleMember->UnscheduledDepsInBundle = BundleMember->UnscheduledDeps;
    if (BundleMember->UnscheduledDepsInBundle == 0) {
      ReadyInsts.insert(BundleMember);
    }
    BundleMember = Next;
  }
}

void BoUpSLP::BlockScheduling::extendSchedulingRegion(Value *V) {
  if (getScheduleData(V))
    return;
  Instruction *I = dyn_cast<Instruction>(V);
  assert(I && "bundle member must be an instruction");
  assert(!isa<PHINode>(I) && "phi nodes don't need to be scheduled");
  if (!ScheduleStart) {
    // It's the first instruction in the new region.
    initScheduleData(I, I->getNextNode(), nullptr, nullptr);
    ScheduleStart = I;
    ScheduleEnd = I->getNextNode();
    assert(ScheduleEnd && "tried to vectorize a TerminatorInst?");
    DEBUG(dbgs() << "SLP:  initialize schedule region to " << *I << "\n");
    return;
  }
  // Search up and down at the same time, because we don't know if the new
  // instruction is above or below the existing scheduling region.
  BasicBlock::reverse_iterator UpIter(ScheduleStart);
  BasicBlock::reverse_iterator UpperEnd = BB->rend();
  BasicBlock::iterator DownIter(ScheduleEnd);
  BasicBlock::iterator LowerEnd = BB->end();
  for (;;) {
    if (UpIter != UpperEnd) {
      if (&*UpIter == I) {
        initScheduleData(I, ScheduleStart, nullptr, FirstLoadStoreInRegion);
        ScheduleStart = I;
        DEBUG(dbgs() << "SLP:  extend schedule region start to " << *I << "\n");
        return;
      }
      UpIter++;
    }
    if (DownIter != LowerEnd) {
      if (&*DownIter == I) {
        initScheduleData(ScheduleEnd, I->getNextNode(), LastLoadStoreInRegion,
                         nullptr);
        ScheduleEnd = I->getNextNode();
        assert(ScheduleEnd && "tried to vectorize a TerminatorInst?");
        DEBUG(dbgs() << "SLP:  extend schedule region end to " << *I << "\n");
        return;
      }
      DownIter++;
    }
    assert((UpIter != UpperEnd || DownIter != LowerEnd) &&
           "instruction not found in block");
  }
}

void BoUpSLP::BlockScheduling::initScheduleData(Instruction *FromI,
                                                Instruction *ToI,
                                                ScheduleData *PrevLoadStore,
                                                ScheduleData *NextLoadStore) {
  ScheduleData *CurrentLoadStore = PrevLoadStore;
  for (Instruction *I = FromI; I != ToI; I = I->getNextNode()) {
    ScheduleData *SD = ScheduleDataMap[I];
    if (!SD) {
      // Allocate a new ScheduleData for the instruction.
      if (ChunkPos >= ChunkSize) {
        ScheduleDataChunks.push_back(
            llvm::make_unique<ScheduleData[]>(ChunkSize));
        ChunkPos = 0;
      }
      SD = &(ScheduleDataChunks.back()[ChunkPos++]);
      ScheduleDataMap[I] = SD;
      SD->Inst = I;
    }
    assert(!isInSchedulingRegion(SD) &&
           "new ScheduleData already in scheduling region");
    SD->init(SchedulingRegionID);

    if (I->mayReadOrWriteMemory()) {
      // Update the linked list of memory accessing instructions.
      if (CurrentLoadStore) {
        CurrentLoadStore->NextLoadStore = SD;
      } else {
        FirstLoadStoreInRegion = SD;
      }
      CurrentLoadStore = SD;
    }
  }
  if (NextLoadStore) {
    if (CurrentLoadStore)
      CurrentLoadStore->NextLoadStore = NextLoadStore;
  } else {
    LastLoadStoreInRegion = CurrentLoadStore;
  }
}

void BoUpSLP::BlockScheduling::calculateDependencies(ScheduleData *SD,
                                                     bool InsertInReadyList,
                                                     BoUpSLP *SLP) {
  assert(SD->isSchedulingEntity());

  SmallVector<ScheduleData *, 10> WorkList;
  WorkList.push_back(SD);

  while (!WorkList.empty()) {
    ScheduleData *SD = WorkList.back();
    WorkList.pop_back();

    ScheduleData *BundleMember = SD;
    while (BundleMember) {
      assert(isInSchedulingRegion(BundleMember));
      if (!BundleMember->hasValidDependencies()) {

        DEBUG(dbgs() << "SLP:       update deps of " << *BundleMember << "\n");
        BundleMember->Dependencies = 0;
        BundleMember->resetUnscheduledDeps();

        // Handle def-use chain dependencies.
        for (User *U : BundleMember->Inst->users()) {
          if (isa<Instruction>(U)) {
            ScheduleData *UseSD = getScheduleData(U);
            if (UseSD && isInSchedulingRegion(UseSD->FirstInBundle)) {
              BundleMember->Dependencies++;
              ScheduleData *DestBundle = UseSD->FirstInBundle;
              if (!DestBundle->IsScheduled) {
                BundleMember->incrementUnscheduledDeps(1);
              }
              if (!DestBundle->hasValidDependencies()) {
                WorkList.push_back(DestBundle);
              }
            }
          } else {
            // I'm not sure if this can ever happen. But we need to be safe.
            // This lets the instruction/bundle never be scheduled and eventally
            // disable vectorization.
            BundleMember->Dependencies++;
            BundleMember->incrementUnscheduledDeps(1);
          }
        }

        // Handle the memory dependencies.
        ScheduleData *DepDest = BundleMember->NextLoadStore;
        if (DepDest) {
          Instruction *SrcInst = BundleMember->Inst;
          MemoryLocation SrcLoc = getLocation(SrcInst, SLP->AA);
          bool SrcMayWrite = BundleMember->Inst->mayWriteToMemory();
          unsigned numAliased = 0;
          unsigned DistToSrc = 1;

          while (DepDest) {
            assert(isInSchedulingRegion(DepDest));

            // We have two limits to reduce the complexity:
            // 1) AliasedCheckLimit: It's a small limit to reduce calls to
            //    SLP->isAliased (which is the expensive part in this loop).
            // 2) MaxMemDepDistance: It's for very large blocks and it aborts
            //    the whole loop (even if the loop is fast, it's quadratic).
            //    It's important for the loop break condition (see below) to
            //    check this limit even between two read-only instructions.
            if (DistToSrc >= MaxMemDepDistance ||
                    ((SrcMayWrite || DepDest->Inst->mayWriteToMemory()) &&
                     (numAliased >= AliasedCheckLimit ||
                      SLP->isAliased(SrcLoc, SrcInst, DepDest->Inst)))) {

              // We increment the counter only if the locations are aliased
              // (instead of counting all alias checks). This gives a better
              // balance between reduced runtime and accurate dependencies.
              numAliased++;

              DepDest->MemoryDependencies.push_back(BundleMember);
              BundleMember->Dependencies++;
              ScheduleData *DestBundle = DepDest->FirstInBundle;
              if (!DestBundle->IsScheduled) {
                BundleMember->incrementUnscheduledDeps(1);
              }
              if (!DestBundle->hasValidDependencies()) {
                WorkList.push_back(DestBundle);
              }
            }
            DepDest = DepDest->NextLoadStore;

            // Example, explaining the loop break condition: Let's assume our
            // starting instruction is i0 and MaxMemDepDistance = 3.
            //
            //                      +--------v--v--v
            //             i0,i1,i2,i3,i4,i5,i6,i7,i8
            //             +--------^--^--^
            //
            // MaxMemDepDistance let us stop alias-checking at i3 and we add
            // dependencies from i0 to i3,i4,.. (even if they are not aliased).
            // Previously we already added dependencies from i3 to i6,i7,i8
            // (because of MaxMemDepDistance). As we added a dependency from
            // i0 to i3, we have transitive dependencies from i0 to i6,i7,i8
            // and we can abort this loop at i6.
            if (DistToSrc >= 2 * MaxMemDepDistance)
                break;
            DistToSrc++;
          }
        }
      }
      BundleMember = BundleMember->NextInBundle;
    }
    if (InsertInReadyList && SD->isReady()) {
      ReadyInsts.push_back(SD);
      DEBUG(dbgs() << "SLP:     gets ready on update: " << *SD->Inst << "\n");
    }
  }
}

void BoUpSLP::BlockScheduling::resetSchedule() {
  assert(ScheduleStart &&
         "tried to reset schedule on block which has not been scheduled");
  for (Instruction *I = ScheduleStart; I != ScheduleEnd; I = I->getNextNode()) {
    ScheduleData *SD = getScheduleData(I);
    assert(isInSchedulingRegion(SD));
    SD->IsScheduled = false;
    SD->resetUnscheduledDeps();
  }
  ReadyInsts.clear();
}

void BoUpSLP::scheduleBlock(BlockScheduling *BS) {
  
  if (!BS->ScheduleStart)
    return;
  
  DEBUG(dbgs() << "SLP: schedule block " << BS->BB->getName() << "\n");

  BS->resetSchedule();

  // For the real scheduling we use a more sophisticated ready-list: it is
  // sorted by the original instruction location. This lets the final schedule
  // be as  close as possible to the original instruction order.
  struct ScheduleDataCompare {
    bool operator()(ScheduleData *SD1, ScheduleData *SD2) {
      return SD2->SchedulingPriority < SD1->SchedulingPriority;
    }
  };
  std::set<ScheduleData *, ScheduleDataCompare> ReadyInsts;

  // Ensure that all depencency data is updated and fill the ready-list with
  // initial instructions.
  int Idx = 0;
  int NumToSchedule = 0;
  for (auto *I = BS->ScheduleStart; I != BS->ScheduleEnd;
       I = I->getNextNode()) {
    ScheduleData *SD = BS->getScheduleData(I);
    assert(
        SD->isPartOfBundle() == (ScalarToTreeEntry.count(SD->Inst) != 0) &&
        "scheduler and vectorizer have different opinion on what is a bundle");
    SD->FirstInBundle->SchedulingPriority = Idx++;
    if (SD->isSchedulingEntity()) {
      BS->calculateDependencies(SD, false, this);
      NumToSchedule++;
    }
  }
  BS->initialFillReadyList(ReadyInsts);

  Instruction *LastScheduledInst = BS->ScheduleEnd;

  // Do the "real" scheduling.
  while (!ReadyInsts.empty()) {
    ScheduleData *picked = *ReadyInsts.begin();
    ReadyInsts.erase(ReadyInsts.begin());

    // Move the scheduled instruction(s) to their dedicated places, if not
    // there yet.
    ScheduleData *BundleMember = picked;
    while (BundleMember) {
      Instruction *pickedInst = BundleMember->Inst;
      if (LastScheduledInst->getNextNode() != pickedInst) {
        BS->BB->getInstList().remove(pickedInst);
        BS->BB->getInstList().insert(LastScheduledInst, pickedInst);
      }
      LastScheduledInst = pickedInst;
      BundleMember = BundleMember->NextInBundle;
    }

    BS->schedule(picked, ReadyInsts);
    NumToSchedule--;
  }
  assert(NumToSchedule == 0 && "could not schedule all instructions");

  // Avoid duplicate scheduling of the block.
  BS->ScheduleStart = nullptr;
}

/// The SLPVectorizer Pass.
struct SLPVectorizer : public FunctionPass {
  typedef SmallVector<StoreInst *, 8> StoreList;
  typedef MapVector<Value *, StoreList> StoreListMap;

  /// Pass identification, replacement for typeid
  static char ID;

  explicit SLPVectorizer() : FunctionPass(ID) {
    initializeSLPVectorizerPass(*PassRegistry::getPassRegistry());
  }

  ScalarEvolution *SE;
  TargetTransformInfo *TTI;
  TargetLibraryInfo *TLI;
  AliasAnalysis *AA;
  LoopInfo *LI;
  DominatorTree *DT;
  AssumptionCache *AC;

  bool runOnFunction(Function &F) override {
    if (skipOptnoneFunction(F))
      return false;

    SE = &getAnalysis<ScalarEvolution>();
    TTI = &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    auto *TLIP = getAnalysisIfAvailable<TargetLibraryInfoWrapperPass>();
    TLI = TLIP ? &TLIP->getTLI() : nullptr;
    AA = &getAnalysis<AliasAnalysis>();
    LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    AC = &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);

    StoreRefs.clear();
    bool Changed = false;

    // If the target claims to have no vector registers don't attempt
    // vectorization.
    if (!TTI->getNumberOfRegisters(true))
      return false;

    // Use the vector register size specified by the target unless overridden
    // by a command-line option.
    // TODO: It would be better to limit the vectorization factor based on
    //       data type rather than just register size. For example, x86 AVX has
    //       256-bit registers, but it does not support integer operations
    //       at that width (that requires AVX2).
    if (MaxVectorRegSizeOption.getNumOccurrences())
      MaxVecRegSize = MaxVectorRegSizeOption;
    else
      MaxVecRegSize = TTI->getRegisterBitWidth(true);

    // Don't vectorize when the attribute NoImplicitFloat is used.
    if (F.hasFnAttribute(Attribute::NoImplicitFloat))
      return false;

    DEBUG(dbgs() << "SLP: Analyzing blocks in " << F.getName() << ".\n");

    // Use the bottom up slp vectorizer to construct chains that start with
    // store instructions.
    BoUpSLP R(&F, SE, TTI, TLI, AA, LI, DT, AC);

    // A general note: the vectorizer must use BoUpSLP::eraseInstruction() to
    // delete instructions.

    // Scan the blocks in the function in post order.
    for (auto BB : post_order(&F.getEntryBlock())) {
      // Vectorize trees that end at stores.
      if (unsigned count = collectStores(BB, R)) {
        (void)count;
        DEBUG(dbgs() << "SLP: Found " << count << " stores to vectorize.\n");
        Changed |= vectorizeStoreChains(R);
      }

      // Vectorize trees that end at reductions.
      Changed |= vectorizeChainsInBlock(BB, R);
    }

    if (Changed) {
      R.optimizeGatherSequence();
      DEBUG(dbgs() << "SLP: vectorized \"" << F.getName() << "\"\n");
      DEBUG(verifyFunction(F));
    }
    return Changed;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    FunctionPass::getAnalysisUsage(AU);
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<ScalarEvolution>();
    AU.addRequired<AliasAnalysis>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addPreserved<LoopInfoWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.setPreservesCFG();
  }

private:

  /// \brief Collect memory references and sort them according to their base
  /// object. We sort the stores to their base objects to reduce the cost of the
  /// quadratic search on the stores. TODO: We can further reduce this cost
  /// if we flush the chain creation every time we run into a memory barrier.
  unsigned collectStores(BasicBlock *BB, BoUpSLP &R);

  /// \brief Try to vectorize a chain that starts at two arithmetic instrs.
  bool tryToVectorizePair(Value *A, Value *B, BoUpSLP &R);

  /// \brief Try to vectorize a list of operands.
  /// \@param BuildVector A list of users to ignore for the purpose of
  ///                     scheduling and that don't need extracting.
  /// \returns true if a value was vectorized.
  bool tryToVectorizeList(ArrayRef<Value *> VL, BoUpSLP &R,
                          ArrayRef<Value *> BuildVector = None,
                          bool allowReorder = false);

  /// \brief Try to vectorize a chain that may start at the operands of \V;
  bool tryToVectorize(BinaryOperator *V, BoUpSLP &R);

  /// \brief Vectorize the stores that were collected in StoreRefs.
  bool vectorizeStoreChains(BoUpSLP &R);

  /// \brief Scan the basic block and look for patterns that are likely to start
  /// a vectorization chain.
  bool vectorizeChainsInBlock(BasicBlock *BB, BoUpSLP &R);

  bool vectorizeStoreChain(ArrayRef<Value *> Chain, int CostThreshold,
                           BoUpSLP &R, unsigned VecRegSize);

  bool vectorizeStores(ArrayRef<StoreInst *> Stores, int costThreshold,
                       BoUpSLP &R);
private:
  StoreListMap StoreRefs;
  unsigned MaxVecRegSize; // This is set by TTI or overridden by cl::opt.
};

/// \brief Check that the Values in the slice in VL array are still existent in
/// the WeakVH array.
/// Vectorization of part of the VL array may cause later values in the VL array
/// to become invalid. We track when this has happened in the WeakVH array.
static bool hasValueBeenRAUWed(ArrayRef<Value *> VL, ArrayRef<WeakVH> VH,
                               unsigned SliceBegin, unsigned SliceSize) {
  VL = VL.slice(SliceBegin, SliceSize);
  VH = VH.slice(SliceBegin, SliceSize);
  return !std::equal(VL.begin(), VL.end(), VH.begin());
}

bool SLPVectorizer::vectorizeStoreChain(ArrayRef<Value *> Chain,
                                        int CostThreshold, BoUpSLP &R,
                                        unsigned VecRegSize) {
  unsigned ChainLen = Chain.size();
  DEBUG(dbgs() << "SLP: Analyzing a store chain of length " << ChainLen
        << "\n");
  Type *StoreTy = cast<StoreInst>(Chain[0])->getValueOperand()->getType();
  auto &DL = cast<StoreInst>(Chain[0])->getModule()->getDataLayout();
  unsigned Sz = DL.getTypeSizeInBits(StoreTy);
  unsigned VF = VecRegSize / Sz;

  if (!isPowerOf2_32(Sz) || VF < 2)
    return false;

  // Keep track of values that were deleted by vectorizing in the loop below.
  SmallVector<WeakVH, 8> TrackValues(Chain.begin(), Chain.end());

  bool Changed = false;
  // Look for profitable vectorizable trees at all offsets, starting at zero.
  for (unsigned i = 0, e = ChainLen; i < e; ++i) {
    if (i + VF > e)
      break;

    // Check that a previous iteration of this loop did not delete the Value.
    if (hasValueBeenRAUWed(Chain, TrackValues, i, VF))
      continue;

    DEBUG(dbgs() << "SLP: Analyzing " << VF << " stores at offset " << i
          << "\n");
    ArrayRef<Value *> Operands = Chain.slice(i, VF);

    R.buildTree(Operands);

    int Cost = R.getTreeCost();

    DEBUG(dbgs() << "SLP: Found cost=" << Cost << " for VF=" << VF << "\n");
    if (Cost < CostThreshold) {
      DEBUG(dbgs() << "SLP: Decided to vectorize cost=" << Cost << "\n");
      R.vectorizeTree();

      // Move to the next bundle.
      i += VF - 1;
      Changed = true;
    }
  }

  return Changed;
}

bool SLPVectorizer::vectorizeStores(ArrayRef<StoreInst *> Stores,
                                    int costThreshold, BoUpSLP &R) {
  SetVector<StoreInst *> Heads, Tails;
  SmallDenseMap<StoreInst *, StoreInst *> ConsecutiveChain;

  // We may run into multiple chains that merge into a single chain. We mark the
  // stores that we vectorized so that we don't visit the same store twice.
  BoUpSLP::ValueSet VectorizedStores;
  bool Changed = false;

  // Do a quadratic search on all of the given stores and find
  // all of the pairs of stores that follow each other.
  for (unsigned i = 0, e = Stores.size(); i < e; ++i) {
    for (unsigned j = 0; j < e; ++j) {
      if (i == j)
        continue;
      const DataLayout &DL = Stores[i]->getModule()->getDataLayout();
      if (R.isConsecutiveAccess(Stores[i], Stores[j], DL)) {
        Tails.insert(Stores[j]);
        Heads.insert(Stores[i]);
        ConsecutiveChain[Stores[i]] = Stores[j];
      }
    }
  }

  // For stores that start but don't end a link in the chain:
  for (SetVector<StoreInst *>::iterator it = Heads.begin(), e = Heads.end();
       it != e; ++it) {
    if (Tails.count(*it))
      continue;

    // We found a store instr that starts a chain. Now follow the chain and try
    // to vectorize it.
    BoUpSLP::ValueList Operands;
    StoreInst *I = *it;
    // Collect the chain into a list.
    while (Tails.count(I) || Heads.count(I)) {
      if (VectorizedStores.count(I))
        break;
      Operands.push_back(I);
      // Move to the next value in the chain.
      I = ConsecutiveChain[I];
    }

    // FIXME: Is division-by-2 the correct step? Should we assert that the
    // register size is a power-of-2?
    for (unsigned Size = MaxVecRegSize; Size >= MinVecRegSize; Size /= 2) {
      if (vectorizeStoreChain(Operands, costThreshold, R, Size)) {
        // Mark the vectorized stores so that we don't vectorize them again.
        VectorizedStores.insert(Operands.begin(), Operands.end());
        Changed = true;
        break;
      }
    }
  }

  return Changed;
}


unsigned SLPVectorizer::collectStores(BasicBlock *BB, BoUpSLP &R) {
  unsigned count = 0;
  StoreRefs.clear();
  const DataLayout &DL = BB->getModule()->getDataLayout();
  for (Instruction &I : *BB) {
    StoreInst *SI = dyn_cast<StoreInst>(&I);
    if (!SI)
      continue;

    // Don't touch volatile stores.
    if (!SI->isSimple())
      continue;

    // Check that the pointer points to scalars.
    Type *Ty = SI->getValueOperand()->getType();
    if (!isValidElementType(Ty))
      continue;

    // Find the base pointer.
    Value *Ptr = GetUnderlyingObject(SI->getPointerOperand(), DL);

    // Save the store locations.
    StoreRefs[Ptr].push_back(SI);
    count++;
  }
  return count;
}

bool SLPVectorizer::tryToVectorizePair(Value *A, Value *B, BoUpSLP &R) {
  if (!A || !B)
    return false;
  Value *VL[] = { A, B };
  return tryToVectorizeList(VL, R, None, true);
}

bool SLPVectorizer::tryToVectorizeList(ArrayRef<Value *> VL, BoUpSLP &R,
                                       ArrayRef<Value *> BuildVector,
                                       bool allowReorder) {
  if (VL.size() < 2)
    return false;

  DEBUG(dbgs() << "SLP: Vectorizing a list of length = " << VL.size() << ".\n");

  // Check that all of the parts are scalar instructions of the same type.
  Instruction *I0 = dyn_cast<Instruction>(VL[0]);
  if (!I0)
    return false;

  unsigned Opcode0 = I0->getOpcode();
  const DataLayout &DL = I0->getModule()->getDataLayout();

  Type *Ty0 = I0->getType();
  unsigned Sz = DL.getTypeSizeInBits(Ty0);
  // FIXME: Register size should be a parameter to this function, so we can
  // try different vectorization factors.
  unsigned VF = MinVecRegSize / Sz;

  for (Value *V : VL) {
    Type *Ty = V->getType();
    if (!isValidElementType(Ty))
      return false;
    Instruction *Inst = dyn_cast<Instruction>(V);
    if (!Inst || Inst->getOpcode() != Opcode0)
      return false;
  }

  bool Changed = false;

  // Keep track of values that were deleted by vectorizing in the loop below.
  SmallVector<WeakVH, 8> TrackValues(VL.begin(), VL.end());

  for (unsigned i = 0, e = VL.size(); i < e; ++i) {
    unsigned OpsWidth = 0;

    if (i + VF > e)
      OpsWidth = e - i;
    else
      OpsWidth = VF;

    if (!isPowerOf2_32(OpsWidth) || OpsWidth < 2)
      break;

    // Check that a previous iteration of this loop did not delete the Value.
    if (hasValueBeenRAUWed(VL, TrackValues, i, OpsWidth))
      continue;

    DEBUG(dbgs() << "SLP: Analyzing " << OpsWidth << " operations "
                 << "\n");
    ArrayRef<Value *> Ops = VL.slice(i, OpsWidth);

    ArrayRef<Value *> BuildVectorSlice;
    if (!BuildVector.empty())
      BuildVectorSlice = BuildVector.slice(i, OpsWidth);

    R.buildTree(Ops, BuildVectorSlice);
    // TODO: check if we can allow reordering also for other cases than
    // tryToVectorizePair()
    if (allowReorder && R.shouldReorder()) {
      assert(Ops.size() == 2);
      assert(BuildVectorSlice.empty());
      Value *ReorderedOps[] = { Ops[1], Ops[0] };
      R.buildTree(ReorderedOps, None);
    }
    int Cost = R.getTreeCost();

    if (Cost < -SLPCostThreshold) {
      DEBUG(dbgs() << "SLP: Vectorizing list at cost:" << Cost << ".\n");
      Value *VectorizedRoot = R.vectorizeTree();

      // Reconstruct the build vector by extracting the vectorized root. This
      // way we handle the case where some elements of the vector are undefined.
      //  (return (inserelt <4 xi32> (insertelt undef (opd0) 0) (opd1) 2))
      if (!BuildVectorSlice.empty()) {
        // The insert point is the last build vector instruction. The vectorized
        // root will precede it. This guarantees that we get an instruction. The
        // vectorized tree could have been constant folded.
        Instruction *InsertAfter = cast<Instruction>(BuildVectorSlice.back());
        unsigned VecIdx = 0;
        for (auto &V : BuildVectorSlice) {
          IRBuilder<true, NoFolder> Builder(
              ++BasicBlock::iterator(InsertAfter));
          InsertElementInst *IE = cast<InsertElementInst>(V);
          Instruction *Extract = cast<Instruction>(Builder.CreateExtractElement(
              VectorizedRoot, Builder.getInt32(VecIdx++)));
          IE->setOperand(1, Extract);
          IE->removeFromParent();
          IE->insertAfter(Extract);
          InsertAfter = IE;
        }
      }
      // Move to the next bundle.
      i += VF - 1;
      Changed = true;
    }
  }

  return Changed;
}

bool SLPVectorizer::tryToVectorize(BinaryOperator *V, BoUpSLP &R) {
  if (!V)
    return false;

  // Try to vectorize V.
  if (tryToVectorizePair(V->getOperand(0), V->getOperand(1), R))
    return true;

  BinaryOperator *A = dyn_cast<BinaryOperator>(V->getOperand(0));
  BinaryOperator *B = dyn_cast<BinaryOperator>(V->getOperand(1));
  // Try to skip B.
  if (B && B->hasOneUse()) {
    BinaryOperator *B0 = dyn_cast<BinaryOperator>(B->getOperand(0));
    BinaryOperator *B1 = dyn_cast<BinaryOperator>(B->getOperand(1));
    if (tryToVectorizePair(A, B0, R)) {
      return true;
    }
    if (tryToVectorizePair(A, B1, R)) {
      return true;
    }
  }

  // Try to skip A.
  if (A && A->hasOneUse()) {
    BinaryOperator *A0 = dyn_cast<BinaryOperator>(A->getOperand(0));
    BinaryOperator *A1 = dyn_cast<BinaryOperator>(A->getOperand(1));
    if (tryToVectorizePair(A0, B, R)) {
      return true;
    }
    if (tryToVectorizePair(A1, B, R)) {
      return true;
    }
  }
  return 0;
}

/// \brief Generate a shuffle mask to be used in a reduction tree.
///
/// \param VecLen The length of the vector to be reduced.
/// \param NumEltsToRdx The number of elements that should be reduced in the
///        vector.
/// \param IsPairwise Whether the reduction is a pairwise or splitting
///        reduction. A pairwise reduction will generate a mask of 
///        <0,2,...> or <1,3,..> while a splitting reduction will generate
///        <2,3, undef,undef> for a vector of 4 and NumElts = 2.
/// \param IsLeft True will generate a mask of even elements, odd otherwise.
static Value *createRdxShuffleMask(unsigned VecLen, unsigned NumEltsToRdx,
                                   bool IsPairwise, bool IsLeft,
                                   IRBuilder<> &Builder) {
  assert((IsPairwise || !IsLeft) && "Don't support a <0,1,undef,...> mask");

  SmallVector<Constant *, 32> ShuffleMask(
      VecLen, UndefValue::get(Builder.getInt32Ty()));

  if (IsPairwise)
    // Build a mask of 0, 2, ... (left) or 1, 3, ... (right).
    for (unsigned i = 0; i != NumEltsToRdx; ++i)
      ShuffleMask[i] = Builder.getInt32(2 * i + !IsLeft);
  else
    // Move the upper half of the vector to the lower half.
    for (unsigned i = 0; i != NumEltsToRdx; ++i)
      ShuffleMask[i] = Builder.getInt32(NumEltsToRdx + i);

  return ConstantVector::get(ShuffleMask);
}


/// Model horizontal reductions.
///
/// A horizontal reduction is a tree of reduction operations (currently add and
/// fadd) that has operations that can be put into a vector as its leaf.
/// For example, this tree:
///
/// mul mul mul mul
///  \  /    \  /
///   +       +
///    \     /
///       +
/// This tree has "mul" as its reduced values and "+" as its reduction
/// operations. A reduction might be feeding into a store or a binary operation
/// feeding a phi.
///    ...
///    \  /
///     +
///     |
///  phi +=
///
///  Or:
///    ...
///    \  /
///     +
///     |
///   *p =
///
class HorizontalReduction {
  SmallVector<Value *, 16> ReductionOps;
  SmallVector<Value *, 32> ReducedVals;

  BinaryOperator *ReductionRoot;
  PHINode *ReductionPHI;

  /// The opcode of the reduction.
  unsigned ReductionOpcode;
  /// The opcode of the values we perform a reduction on.
  unsigned ReducedValueOpcode;
  /// The width of one full horizontal reduction operation.
  unsigned ReduxWidth;
  /// Should we model this reduction as a pairwise reduction tree or a tree that
  /// splits the vector in halves and adds those halves.
  bool IsPairwiseReduction;

public:
  HorizontalReduction()
    : ReductionRoot(nullptr), ReductionPHI(nullptr), ReductionOpcode(0),
    ReducedValueOpcode(0), ReduxWidth(0), IsPairwiseReduction(false) {}

  /// \brief Try to find a reduction tree.
  bool matchAssociativeReduction(PHINode *Phi, BinaryOperator *B) {
    assert((!Phi ||
            std::find(Phi->op_begin(), Phi->op_end(), B) != Phi->op_end()) &&
           "Thi phi needs to use the binary operator");

    // We could have a initial reductions that is not an add.
    //  r *= v1 + v2 + v3 + v4
    // In such a case start looking for a tree rooted in the first '+'.
    if (Phi) {
      if (B->getOperand(0) == Phi) {
        Phi = nullptr;
        B = dyn_cast<BinaryOperator>(B->getOperand(1));
      } else if (B->getOperand(1) == Phi) {
        Phi = nullptr;
        B = dyn_cast<BinaryOperator>(B->getOperand(0));
      }
    }

    if (!B)
      return false;

    Type *Ty = B->getType();
    if (!isValidElementType(Ty))
      return false;

    const DataLayout &DL = B->getModule()->getDataLayout();
    ReductionOpcode = B->getOpcode();
    ReducedValueOpcode = 0;
    // FIXME: Register size should be a parameter to this function, so we can
    // try different vectorization factors.
    ReduxWidth = MinVecRegSize / DL.getTypeSizeInBits(Ty);
    ReductionRoot = B;
    ReductionPHI = Phi;

    if (ReduxWidth < 4)
      return false;

    // We currently only support adds.
    if (ReductionOpcode != Instruction::Add &&
        ReductionOpcode != Instruction::FAdd)
      return false;

    // Post order traverse the reduction tree starting at B. We only handle true
    // trees containing only binary operators.
    SmallVector<std::pair<BinaryOperator *, unsigned>, 32> Stack;
    Stack.push_back(std::make_pair(B, 0));
    while (!Stack.empty()) {
      BinaryOperator *TreeN = Stack.back().first;
      unsigned EdgeToVist = Stack.back().second++;
      bool IsReducedValue = TreeN->getOpcode() != ReductionOpcode;

      // Only handle trees in the current basic block.
      if (TreeN->getParent() != B->getParent())
        return false;

      // Each tree node needs to have one user except for the ultimate
      // reduction.
      if (!TreeN->hasOneUse() && TreeN != B)
        return false;

      // Postorder vist.
      if (EdgeToVist == 2 || IsReducedValue) {
        if (IsReducedValue) {
          // Make sure that the opcodes of the operations that we are going to
          // reduce match.
          if (!ReducedValueOpcode)
            ReducedValueOpcode = TreeN->getOpcode();
          else if (ReducedValueOpcode != TreeN->getOpcode())
            return false;
          ReducedVals.push_back(TreeN);
        } else {
          // We need to be able to reassociate the adds.
          if (!TreeN->isAssociative())
            return false;
          ReductionOps.push_back(TreeN);
        }
        // Retract.
        Stack.pop_back();
        continue;
      }

      // Visit left or right.
      Value *NextV = TreeN->getOperand(EdgeToVist);
      BinaryOperator *Next = dyn_cast<BinaryOperator>(NextV);
      if (Next)
        Stack.push_back(std::make_pair(Next, 0));
      else if (NextV != Phi)
        return false;
    }
    return true;
  }

  /// \brief Attempt to vectorize the tree found by
  /// matchAssociativeReduction.
  bool tryToReduce(BoUpSLP &V, TargetTransformInfo *TTI) {
    if (ReducedVals.empty())
      return false;

    unsigned NumReducedVals = ReducedVals.size();
    if (NumReducedVals < ReduxWidth)
      return false;

    Value *VectorizedTree = nullptr;
    IRBuilder<> Builder(ReductionRoot);
    FastMathFlags Unsafe;
    Unsafe.setUnsafeAlgebra();
    Builder.SetFastMathFlags(Unsafe);
    unsigned i = 0;

    for (; i < NumReducedVals - ReduxWidth + 1; i += ReduxWidth) {
      V.buildTree(makeArrayRef(&ReducedVals[i], ReduxWidth), ReductionOps);

      // Estimate cost.
      int Cost = V.getTreeCost() + getReductionCost(TTI, ReducedVals[i]);
      if (Cost >= -SLPCostThreshold)
        break;

      DEBUG(dbgs() << "SLP: Vectorizing horizontal reduction at cost:" << Cost
                   << ". (HorRdx)\n");

      // Vectorize a tree.
      DebugLoc Loc = cast<Instruction>(ReducedVals[i])->getDebugLoc();
      Value *VectorizedRoot = V.vectorizeTree();

      // Emit a reduction.
      Value *ReducedSubTree = emitReduction(VectorizedRoot, Builder);
      if (VectorizedTree) {
        Builder.SetCurrentDebugLocation(Loc);
        VectorizedTree = createBinOp(Builder, ReductionOpcode, VectorizedTree,
                                     ReducedSubTree, "bin.rdx");
      } else
        VectorizedTree = ReducedSubTree;
    }

    if (VectorizedTree) {
      // Finish the reduction.
      for (; i < NumReducedVals; ++i) {
        Builder.SetCurrentDebugLocation(
          cast<Instruction>(ReducedVals[i])->getDebugLoc());
        VectorizedTree = createBinOp(Builder, ReductionOpcode, VectorizedTree,
                                     ReducedVals[i]);
      }
      // Update users.
      if (ReductionPHI) {
        assert(ReductionRoot && "Need a reduction operation");
        ReductionRoot->setOperand(0, VectorizedTree);
        ReductionRoot->setOperand(1, ReductionPHI);
      } else
        ReductionRoot->replaceAllUsesWith(VectorizedTree);
    }
    return VectorizedTree != nullptr;
  }

private:

  /// \brief Calcuate the cost of a reduction.
  int getReductionCost(TargetTransformInfo *TTI, Value *FirstReducedVal) {
    Type *ScalarTy = FirstReducedVal->getType();
    Type *VecTy = VectorType::get(ScalarTy, ReduxWidth);

    int PairwiseRdxCost = TTI->getReductionCost(ReductionOpcode, VecTy, true);
    int SplittingRdxCost = TTI->getReductionCost(ReductionOpcode, VecTy, false);

    IsPairwiseReduction = PairwiseRdxCost < SplittingRdxCost;
    int VecReduxCost = IsPairwiseReduction ? PairwiseRdxCost : SplittingRdxCost;

    int ScalarReduxCost =
        ReduxWidth * TTI->getArithmeticInstrCost(ReductionOpcode, VecTy);

    DEBUG(dbgs() << "SLP: Adding cost " << VecReduxCost - ScalarReduxCost
                 << " for reduction that starts with " << *FirstReducedVal
                 << " (It is a "
                 << (IsPairwiseReduction ? "pairwise" : "splitting")
                 << " reduction)\n");

    return VecReduxCost - ScalarReduxCost;
  }

  static Value *createBinOp(IRBuilder<> &Builder, unsigned Opcode, Value *L,
                            Value *R, const Twine &Name = "") {
    if (Opcode == Instruction::FAdd)
      return Builder.CreateFAdd(L, R, Name);
    return Builder.CreateBinOp((Instruction::BinaryOps)Opcode, L, R, Name);
  }

  /// \brief Emit a horizontal reduction of the vectorized value.
  Value *emitReduction(Value *VectorizedValue, IRBuilder<> &Builder) {
    assert(VectorizedValue && "Need to have a vectorized tree node");
    assert(isPowerOf2_32(ReduxWidth) &&
           "We only handle power-of-two reductions for now");

    Value *TmpVec = VectorizedValue;
    for (unsigned i = ReduxWidth / 2; i != 0; i >>= 1) {
      if (IsPairwiseReduction) {
        Value *LeftMask =
          createRdxShuffleMask(ReduxWidth, i, true, true, Builder);
        Value *RightMask =
          createRdxShuffleMask(ReduxWidth, i, true, false, Builder);

        Value *LeftShuf = Builder.CreateShuffleVector(
          TmpVec, UndefValue::get(TmpVec->getType()), LeftMask, "rdx.shuf.l");
        Value *RightShuf = Builder.CreateShuffleVector(
          TmpVec, UndefValue::get(TmpVec->getType()), (RightMask),
          "rdx.shuf.r");
        TmpVec = createBinOp(Builder, ReductionOpcode, LeftShuf, RightShuf,
                             "bin.rdx");
      } else {
        Value *UpperHalf =
          createRdxShuffleMask(ReduxWidth, i, false, false, Builder);
        Value *Shuf = Builder.CreateShuffleVector(
          TmpVec, UndefValue::get(TmpVec->getType()), UpperHalf, "rdx.shuf");
        TmpVec = createBinOp(Builder, ReductionOpcode, TmpVec, Shuf, "bin.rdx");
      }
    }

    // The result is in the first element of the vector.
    return Builder.CreateExtractElement(TmpVec, Builder.getInt32(0));
  }
};

/// \brief Recognize construction of vectors like
///  %ra = insertelement <4 x float> undef, float %s0, i32 0
///  %rb = insertelement <4 x float> %ra, float %s1, i32 1
///  %rc = insertelement <4 x float> %rb, float %s2, i32 2
///  %rd = insertelement <4 x float> %rc, float %s3, i32 3
///
/// Returns true if it matches
///
static bool findBuildVector(InsertElementInst *FirstInsertElem,
                            SmallVectorImpl<Value *> &BuildVector,
                            SmallVectorImpl<Value *> &BuildVectorOpds) {
  if (!isa<UndefValue>(FirstInsertElem->getOperand(0)))
    return false;

  InsertElementInst *IE = FirstInsertElem;
  while (true) {
    BuildVector.push_back(IE);
    BuildVectorOpds.push_back(IE->getOperand(1));

    if (IE->use_empty())
      return false;

    InsertElementInst *NextUse = dyn_cast<InsertElementInst>(IE->user_back());
    if (!NextUse)
      return true;

    // If this isn't the final use, make sure the next insertelement is the only
    // use. It's OK if the final constructed vector is used multiple times
    if (!IE->hasOneUse())
      return false;

    IE = NextUse;
  }

  return false;
}

static bool PhiTypeSorterFunc(Value *V, Value *V2) {
  return V->getType() < V2->getType();
}

bool SLPVectorizer::vectorizeChainsInBlock(BasicBlock *BB, BoUpSLP &R) {
  bool Changed = false;
  SmallVector<Value *, 4> Incoming;
  SmallSet<Value *, 16> VisitedInstrs;

  bool HaveVectorizedPhiNodes = true;
  while (HaveVectorizedPhiNodes) {
    HaveVectorizedPhiNodes = false;

    // Collect the incoming values from the PHIs.
    Incoming.clear();
    for (BasicBlock::iterator instr = BB->begin(), ie = BB->end(); instr != ie;
         ++instr) {
      PHINode *P = dyn_cast<PHINode>(instr);
      if (!P)
        break;

      if (!VisitedInstrs.count(P))
        Incoming.push_back(P);
    }

    // Sort by type.
    std::stable_sort(Incoming.begin(), Incoming.end(), PhiTypeSorterFunc);

    // Try to vectorize elements base on their type.
    for (SmallVector<Value *, 4>::iterator IncIt = Incoming.begin(),
                                           E = Incoming.end();
         IncIt != E;) {

      // Look for the next elements with the same type.
      SmallVector<Value *, 4>::iterator SameTypeIt = IncIt;
      while (SameTypeIt != E &&
             (*SameTypeIt)->getType() == (*IncIt)->getType()) {
        VisitedInstrs.insert(*SameTypeIt);
        ++SameTypeIt;
      }

      // Try to vectorize them.
      unsigned NumElts = (SameTypeIt - IncIt);
      DEBUG(errs() << "SLP: Trying to vectorize starting at PHIs (" << NumElts << ")\n");
      if (NumElts > 1 && tryToVectorizeList(makeArrayRef(IncIt, NumElts), R)) {
        // Success start over because instructions might have been changed.
        HaveVectorizedPhiNodes = true;
        Changed = true;
        break;
      }

      // Start over at the next instruction of a different type (or the end).
      IncIt = SameTypeIt;
    }
  }

  VisitedInstrs.clear();

  for (BasicBlock::iterator it = BB->begin(), e = BB->end(); it != e; it++) {
    // We may go through BB multiple times so skip the one we have checked.
    if (!VisitedInstrs.insert(it).second)
      continue;

    if (isa<DbgInfoIntrinsic>(it))
      continue;

    // Try to vectorize reductions that use PHINodes.
    if (PHINode *P = dyn_cast<PHINode>(it)) {
      // Check that the PHI is a reduction PHI.
      if (P->getNumIncomingValues() != 2)
        return Changed;
      Value *Rdx =
          (P->getIncomingBlock(0) == BB
               ? (P->getIncomingValue(0))
               : (P->getIncomingBlock(1) == BB ? P->getIncomingValue(1)
                                               : nullptr));
      // Check if this is a Binary Operator.
      BinaryOperator *BI = dyn_cast_or_null<BinaryOperator>(Rdx);
      if (!BI)
        continue;

      // Try to match and vectorize a horizontal reduction.
      HorizontalReduction HorRdx;
      if (ShouldVectorizeHor && HorRdx.matchAssociativeReduction(P, BI) &&
          HorRdx.tryToReduce(R, TTI)) {
        Changed = true;
        it = BB->begin();
        e = BB->end();
        continue;
      }

     Value *Inst = BI->getOperand(0);
      if (Inst == P)
        Inst = BI->getOperand(1);

      if (tryToVectorize(dyn_cast<BinaryOperator>(Inst), R)) {
        // We would like to start over since some instructions are deleted
        // and the iterator may become invalid value.
        Changed = true;
        it = BB->begin();
        e = BB->end();
        continue;
      }

      continue;
    }

    // Try to vectorize horizontal reductions feeding into a store.
    if (ShouldStartVectorizeHorAtStore)
      if (StoreInst *SI = dyn_cast<StoreInst>(it))
        if (BinaryOperator *BinOp =
                dyn_cast<BinaryOperator>(SI->getValueOperand())) {
          HorizontalReduction HorRdx;
          if (((HorRdx.matchAssociativeReduction(nullptr, BinOp) &&
                HorRdx.tryToReduce(R, TTI)) ||
               tryToVectorize(BinOp, R))) {
            Changed = true;
            it = BB->begin();
            e = BB->end();
            continue;
          }
        }

    // Try to vectorize horizontal reductions feeding into a return.
    if (ReturnInst *RI = dyn_cast<ReturnInst>(it))
      if (RI->getNumOperands() != 0)
        if (BinaryOperator *BinOp =
                dyn_cast<BinaryOperator>(RI->getOperand(0))) {
          DEBUG(dbgs() << "SLP: Found a return to vectorize.\n");
          if (tryToVectorizePair(BinOp->getOperand(0),
                                 BinOp->getOperand(1), R)) {
            Changed = true;
            it = BB->begin();
            e = BB->end();
            continue;
          }
        }

    // Try to vectorize trees that start at compare instructions.
    if (CmpInst *CI = dyn_cast<CmpInst>(it)) {
      if (tryToVectorizePair(CI->getOperand(0), CI->getOperand(1), R)) {
        Changed = true;
        // We would like to start over since some instructions are deleted
        // and the iterator may become invalid value.
        it = BB->begin();
        e = BB->end();
        continue;
      }

      for (int i = 0; i < 2; ++i) {
        if (BinaryOperator *BI = dyn_cast<BinaryOperator>(CI->getOperand(i))) {
          if (tryToVectorizePair(BI->getOperand(0), BI->getOperand(1), R)) {
            Changed = true;
            // We would like to start over since some instructions are deleted
            // and the iterator may become invalid value.
            it = BB->begin();
            e = BB->end();
            break;
          }
        }
      }
      continue;
    }

    // Try to vectorize trees that start at insertelement instructions.
    if (InsertElementInst *FirstInsertElem = dyn_cast<InsertElementInst>(it)) {
      SmallVector<Value *, 16> BuildVector;
      SmallVector<Value *, 16> BuildVectorOpds;
      if (!findBuildVector(FirstInsertElem, BuildVector, BuildVectorOpds))
        continue;

      // Vectorize starting with the build vector operands ignoring the
      // BuildVector instructions for the purpose of scheduling and user
      // extraction.
      if (tryToVectorizeList(BuildVectorOpds, R, BuildVector)) {
        Changed = true;
        it = BB->begin();
        e = BB->end();
      }

      continue;
    }
  }

  return Changed;
}

bool SLPVectorizer::vectorizeStoreChains(BoUpSLP &R) {
  bool Changed = false;
  // Attempt to sort and vectorize each of the store-groups.
  for (StoreListMap::iterator it = StoreRefs.begin(), e = StoreRefs.end();
       it != e; ++it) {
    if (it->second.size() < 2)
      continue;

    DEBUG(dbgs() << "SLP: Analyzing a store chain of length "
          << it->second.size() << ".\n");

    // Process the stores in chunks of 16.
    // TODO: The limit of 16 inhibits greater vectorization factors.
    //       For example, AVX2 supports v32i8. Increasing this limit, however,
    //       may cause a significant compile-time increase.
    for (unsigned CI = 0, CE = it->second.size(); CI < CE; CI+=16) {
      unsigned Len = std::min<unsigned>(CE - CI, 16);
      Changed |= vectorizeStores(makeArrayRef(&it->second[CI], Len),
                                 -SLPCostThreshold, R);
    }
  }
  return Changed;
}

} // end anonymous namespace

char SLPVectorizer::ID = 0;
static const char lv_name[] = "SLP Vectorizer";
INITIALIZE_PASS_BEGIN(SLPVectorizer, SV_NAME, lv_name, false, false)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_END(SLPVectorizer, SV_NAME, lv_name, false, false)

namespace llvm {
Pass *createSLPVectorizerPass() { return new SLPVectorizer(); }
}
