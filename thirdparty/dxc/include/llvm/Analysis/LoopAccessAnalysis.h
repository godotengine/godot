//===- llvm/Analysis/LoopAccessAnalysis.h -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface for the loop memory dependence framework that
// was originally developed for the Loop Vectorizer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LOOPACCESSANALYSIS_H
#define LLVM_ANALYSIS_LOOPACCESSANALYSIS_H

#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class Value;
class DataLayout;
class AliasAnalysis;
class ScalarEvolution;
class Loop;
class SCEV;

/// Optimization analysis message produced during vectorization. Messages inform
/// the user why vectorization did not occur.
class LoopAccessReport {
  std::string Message;
  const Instruction *Instr;

protected:
  LoopAccessReport(const Twine &Message, const Instruction *I)
      : Message(Message.str()), Instr(I) {}

public:
  LoopAccessReport(const Instruction *I = nullptr) : Instr(I) {}

  template <typename A> LoopAccessReport &operator<<(const A &Value) {
    raw_string_ostream Out(Message);
    Out << Value;
    return *this;
  }

  const Instruction *getInstr() const { return Instr; }

  std::string &str() { return Message; }
  const std::string &str() const { return Message; }
  operator Twine() { return Message; }

  /// \brief Emit an analysis note for \p PassName with the debug location from
  /// the instruction in \p Message if available.  Otherwise use the location of
  /// \p TheLoop.
  static void emitAnalysis(const LoopAccessReport &Message,
                           const Function *TheFunction,
                           const Loop *TheLoop,
                           const char *PassName);
};

/// \brief Collection of parameters shared beetween the Loop Vectorizer and the
/// Loop Access Analysis.
struct VectorizerParams {
  /// \brief Maximum SIMD width.
  static const unsigned MaxVectorWidth;

  /// \brief VF as overridden by the user.
  static unsigned VectorizationFactor;
  /// \brief Interleave factor as overridden by the user.
  static unsigned VectorizationInterleave;
  /// \brief True if force-vector-interleave was specified by the user.
  static bool isInterleaveForced();

  /// \\brief When performing memory disambiguation checks at runtime do not
  /// make more than this number of comparisons.
  static unsigned RuntimeMemoryCheckThreshold;
};

/// \brief Checks memory dependences among accesses to the same underlying
/// object to determine whether there vectorization is legal or not (and at
/// which vectorization factor).
///
/// Note: This class will compute a conservative dependence for access to
/// different underlying pointers. Clients, such as the loop vectorizer, will
/// sometimes deal these potential dependencies by emitting runtime checks.
///
/// We use the ScalarEvolution framework to symbolically evalutate access
/// functions pairs. Since we currently don't restructure the loop we can rely
/// on the program order of memory accesses to determine their safety.
/// At the moment we will only deem accesses as safe for:
///  * A negative constant distance assuming program order.
///
///      Safe: tmp = a[i + 1];     OR     a[i + 1] = x;
///            a[i] = tmp;                y = a[i];
///
///   The latter case is safe because later checks guarantuee that there can't
///   be a cycle through a phi node (that is, we check that "x" and "y" is not
///   the same variable: a header phi can only be an induction or a reduction, a
///   reduction can't have a memory sink, an induction can't have a memory
///   source). This is important and must not be violated (or we have to
///   resort to checking for cycles through memory).
///
///  * A positive constant distance assuming program order that is bigger
///    than the biggest memory access.
///
///     tmp = a[i]        OR              b[i] = x
///     a[i+2] = tmp                      y = b[i+2];
///
///     Safe distance: 2 x sizeof(a[0]), and 2 x sizeof(b[0]), respectively.
///
///  * Zero distances and all accesses have the same size.
///
class MemoryDepChecker {
public:
  typedef PointerIntPair<Value *, 1, bool> MemAccessInfo;
  typedef SmallPtrSet<MemAccessInfo, 8> MemAccessInfoSet;
  /// \brief Set of potential dependent memory accesses.
  typedef EquivalenceClasses<MemAccessInfo> DepCandidates;

  /// \brief Dependece between memory access instructions.
  struct Dependence {
    /// \brief The type of the dependence.
    enum DepType {
      // No dependence.
      NoDep,
      // We couldn't determine the direction or the distance.
      Unknown,
      // Lexically forward.
      Forward,
      // Forward, but if vectorized, is likely to prevent store-to-load
      // forwarding.
      ForwardButPreventsForwarding,
      // Lexically backward.
      Backward,
      // Backward, but the distance allows a vectorization factor of
      // MaxSafeDepDistBytes.
      BackwardVectorizable,
      // Same, but may prevent store-to-load forwarding.
      BackwardVectorizableButPreventsForwarding
    };

    /// \brief String version of the types.
    static const char *DepName[];

    /// \brief Index of the source of the dependence in the InstMap vector.
    unsigned Source;
    /// \brief Index of the destination of the dependence in the InstMap vector.
    unsigned Destination;
    /// \brief The type of the dependence.
    DepType Type;

    Dependence(unsigned Source, unsigned Destination, DepType Type)
        : Source(Source), Destination(Destination), Type(Type) {}

    /// \brief Dependence types that don't prevent vectorization.
    static bool isSafeForVectorization(DepType Type);

    /// \brief Dependence types that can be queried from the analysis.
    static bool isInterestingDependence(DepType Type);

    /// \brief Lexically backward dependence types.
    bool isPossiblyBackward() const;

    /// \brief Print the dependence.  \p Instr is used to map the instruction
    /// indices to instructions.
    void print(raw_ostream &OS, unsigned Depth,
               const SmallVectorImpl<Instruction *> &Instrs) const;
  };

  MemoryDepChecker(ScalarEvolution *Se, const Loop *L)
      : SE(Se), InnermostLoop(L), AccessIdx(0),
        ShouldRetryWithRuntimeCheck(false), SafeForVectorization(true),
        RecordInterestingDependences(true) {}

  /// \brief Register the location (instructions are given increasing numbers)
  /// of a write access.
  void addAccess(StoreInst *SI) {
    Value *Ptr = SI->getPointerOperand();
    Accesses[MemAccessInfo(Ptr, true)].push_back(AccessIdx);
    InstMap.push_back(SI);
    ++AccessIdx;
  }

  /// \brief Register the location (instructions are given increasing numbers)
  /// of a write access.
  void addAccess(LoadInst *LI) {
    Value *Ptr = LI->getPointerOperand();
    Accesses[MemAccessInfo(Ptr, false)].push_back(AccessIdx);
    InstMap.push_back(LI);
    ++AccessIdx;
  }

  /// \brief Check whether the dependencies between the accesses are safe.
  ///
  /// Only checks sets with elements in \p CheckDeps.
  bool areDepsSafe(DepCandidates &AccessSets, MemAccessInfoSet &CheckDeps,
                   const ValueToValueMap &Strides);

  /// \brief No memory dependence was encountered that would inhibit
  /// vectorization.
  bool isSafeForVectorization() const { return SafeForVectorization; }

  /// \brief The maximum number of bytes of a vector register we can vectorize
  /// the accesses safely with.
  unsigned getMaxSafeDepDistBytes() { return MaxSafeDepDistBytes; }

  /// \brief In same cases when the dependency check fails we can still
  /// vectorize the loop with a dynamic array access check.
  bool shouldRetryWithRuntimeCheck() { return ShouldRetryWithRuntimeCheck; }

  /// \brief Returns the interesting dependences.  If null is returned we
  /// exceeded the MaxInterestingDependence threshold and this information is
  /// not available.
  const SmallVectorImpl<Dependence> *getInterestingDependences() const {
    return RecordInterestingDependences ? &InterestingDependences : nullptr;
  }

  void clearInterestingDependences() { InterestingDependences.clear(); }

  /// \brief The vector of memory access instructions.  The indices are used as
  /// instruction identifiers in the Dependence class.
  const SmallVectorImpl<Instruction *> &getMemoryInstructions() const {
    return InstMap;
  }

  /// \brief Find the set of instructions that read or write via \p Ptr.
  SmallVector<Instruction *, 4> getInstructionsForAccess(Value *Ptr,
                                                         bool isWrite) const;

private:
  ScalarEvolution *SE;
  const Loop *InnermostLoop;

  /// \brief Maps access locations (ptr, read/write) to program order.
  DenseMap<MemAccessInfo, std::vector<unsigned> > Accesses;

  /// \brief Memory access instructions in program order.
  SmallVector<Instruction *, 16> InstMap;

  /// \brief The program order index to be used for the next instruction.
  unsigned AccessIdx;

  // We can access this many bytes in parallel safely.
  unsigned MaxSafeDepDistBytes;

  /// \brief If we see a non-constant dependence distance we can still try to
  /// vectorize this loop with runtime checks.
  bool ShouldRetryWithRuntimeCheck;

  /// \brief No memory dependence was encountered that would inhibit
  /// vectorization.
  bool SafeForVectorization;

  //// \brief True if InterestingDependences reflects the dependences in the
  //// loop.  If false we exceeded MaxInterestingDependence and
  //// InterestingDependences is invalid.
  bool RecordInterestingDependences;

  /// \brief Interesting memory dependences collected during the analysis as
  /// defined by isInterestingDependence.  Only valid if
  /// RecordInterestingDependences is true.
  SmallVector<Dependence, 8> InterestingDependences;

  /// \brief Check whether there is a plausible dependence between the two
  /// accesses.
  ///
  /// Access \p A must happen before \p B in program order. The two indices
  /// identify the index into the program order map.
  ///
  /// This function checks  whether there is a plausible dependence (or the
  /// absence of such can't be proved) between the two accesses. If there is a
  /// plausible dependence but the dependence distance is bigger than one
  /// element access it records this distance in \p MaxSafeDepDistBytes (if this
  /// distance is smaller than any other distance encountered so far).
  /// Otherwise, this function returns true signaling a possible dependence.
  Dependence::DepType isDependent(const MemAccessInfo &A, unsigned AIdx,
                                  const MemAccessInfo &B, unsigned BIdx,
                                  const ValueToValueMap &Strides);

  /// \brief Check whether the data dependence could prevent store-load
  /// forwarding.
  bool couldPreventStoreLoadForward(unsigned Distance, unsigned TypeByteSize);
};

/// \brief Holds information about the memory runtime legality checks to verify
/// that a group of pointers do not overlap.
class RuntimePointerChecking {
public:
  struct PointerInfo {
    /// Holds the pointer value that we need to check.
    TrackingVH<Value> PointerValue;
    /// Holds the pointer value at the beginning of the loop.
    const SCEV *Start;
    /// Holds the pointer value at the end of the loop.
    const SCEV *End;
    /// Holds the information if this pointer is used for writing to memory.
    bool IsWritePtr;
    /// Holds the id of the set of pointers that could be dependent because of a
    /// shared underlying object.
    unsigned DependencySetId;
    /// Holds the id of the disjoint alias set to which this pointer belongs.
    unsigned AliasSetId;
    /// SCEV for the access.
    const SCEV *Expr;

    PointerInfo(Value *PointerValue, const SCEV *Start, const SCEV *End,
                bool IsWritePtr, unsigned DependencySetId, unsigned AliasSetId,
                const SCEV *Expr)
        : PointerValue(PointerValue), Start(Start), End(End),
          IsWritePtr(IsWritePtr), DependencySetId(DependencySetId),
          AliasSetId(AliasSetId), Expr(Expr) {}
  };

  RuntimePointerChecking(ScalarEvolution *SE) : Need(false), SE(SE) {}

  /// Reset the state of the pointer runtime information.
  void reset() {
    Need = false;
    Pointers.clear();
  }

  /// Insert a pointer and calculate the start and end SCEVs.
  void insert(Loop *Lp, Value *Ptr, bool WritePtr, unsigned DepSetId,
              unsigned ASId, const ValueToValueMap &Strides);

  /// \brief No run-time memory checking is necessary.
  bool empty() const { return Pointers.empty(); }

  /// A grouping of pointers. A single memcheck is required between
  /// two groups.
  struct CheckingPtrGroup {
    /// \brief Create a new pointer checking group containing a single
    /// pointer, with index \p Index in RtCheck.
    CheckingPtrGroup(unsigned Index, RuntimePointerChecking &RtCheck)
        : RtCheck(RtCheck), High(RtCheck.Pointers[Index].End),
          Low(RtCheck.Pointers[Index].Start) {
      Members.push_back(Index);
    }

    /// \brief Tries to add the pointer recorded in RtCheck at index
    /// \p Index to this pointer checking group. We can only add a pointer
    /// to a checking group if we will still be able to get
    /// the upper and lower bounds of the check. Returns true in case
    /// of success, false otherwise.
    bool addPointer(unsigned Index);

    /// Constitutes the context of this pointer checking group. For each
    /// pointer that is a member of this group we will retain the index
    /// at which it appears in RtCheck.
    RuntimePointerChecking &RtCheck;
    /// The SCEV expression which represents the upper bound of all the
    /// pointers in this group.
    const SCEV *High;
    /// The SCEV expression which represents the lower bound of all the
    /// pointers in this group.
    const SCEV *Low;
    /// Indices of all the pointers that constitute this grouping.
    SmallVector<unsigned, 2> Members;
  };

  /// \brief Groups pointers such that a single memcheck is required
  /// between two different groups. This will clear the CheckingGroups vector
  /// and re-compute it. We will only group dependecies if \p UseDependencies
  /// is true, otherwise we will create a separate group for each pointer.
  void groupChecks(MemoryDepChecker::DepCandidates &DepCands,
                   bool UseDependencies);

  /// \brief Decide if we need to add a check between two groups of pointers,
  /// according to needsChecking.
  bool needsChecking(const CheckingPtrGroup &M, const CheckingPtrGroup &N,
                     const SmallVectorImpl<int> *PtrPartition) const;

  /// \brief Return true if any pointer requires run-time checking according
  /// to needsChecking.
  bool needsAnyChecking(const SmallVectorImpl<int> *PtrPartition) const;

  /// \brief Returns the number of run-time checks required according to
  /// needsChecking.
  unsigned getNumberOfChecks(const SmallVectorImpl<int> *PtrPartition) const;

  /// \brief Print the list run-time memory checks necessary.
  ///
  /// If \p PtrPartition is set, it contains the partition number for
  /// pointers (-1 if the pointer belongs to multiple partitions).  In this
  /// case omit checks between pointers belonging to the same partition.
  void print(raw_ostream &OS, unsigned Depth = 0,
             const SmallVectorImpl<int> *PtrPartition = nullptr) const;

  /// This flag indicates if we need to add the runtime check.
  bool Need;

  /// Information about the pointers that may require checking.
  SmallVector<PointerInfo, 2> Pointers;

  /// Holds a partitioning of pointers into "check groups".
  SmallVector<CheckingPtrGroup, 2> CheckingGroups;

private:
  /// \brief Decide whether we need to issue a run-time check for pointer at
  /// index \p I and \p J to prove their independence.
  ///
  /// If \p PtrPartition is set, it contains the partition number for
  /// pointers (-1 if the pointer belongs to multiple partitions).  In this
  /// case omit checks between pointers belonging to the same partition.
  bool needsChecking(unsigned I, unsigned J,
                     const SmallVectorImpl<int> *PtrPartition) const;

  /// Holds a pointer to the ScalarEvolution analysis.
  ScalarEvolution *SE;
};

/// \brief Drive the analysis of memory accesses in the loop
///
/// This class is responsible for analyzing the memory accesses of a loop.  It
/// collects the accesses and then its main helper the AccessAnalysis class
/// finds and categorizes the dependences in buildDependenceSets.
///
/// For memory dependences that can be analyzed at compile time, it determines
/// whether the dependence is part of cycle inhibiting vectorization.  This work
/// is delegated to the MemoryDepChecker class.
///
/// For memory dependences that cannot be determined at compile time, it
/// generates run-time checks to prove independence.  This is done by
/// AccessAnalysis::canCheckPtrAtRT and the checks are maintained by the
/// RuntimePointerCheck class.
class LoopAccessInfo {
public:
  LoopAccessInfo(Loop *L, ScalarEvolution *SE, const DataLayout &DL,
                 const TargetLibraryInfo *TLI, AliasAnalysis *AA,
                 DominatorTree *DT, LoopInfo *LI,
                 const ValueToValueMap &Strides);

  /// Return true we can analyze the memory accesses in the loop and there are
  /// no memory dependence cycles.
  bool canVectorizeMemory() const { return CanVecMem; }

  const RuntimePointerChecking *getRuntimePointerChecking() const {
    return &PtrRtChecking;
  }

  /// \brief Number of memchecks required to prove independence of otherwise
  /// may-alias pointers.
  unsigned getNumRuntimePointerChecks(
    const SmallVectorImpl<int> *PtrPartition = nullptr) const {
    return PtrRtChecking.getNumberOfChecks(PtrPartition);
  }

  /// Return true if the block BB needs to be predicated in order for the loop
  /// to be vectorized.
  static bool blockNeedsPredication(BasicBlock *BB, Loop *TheLoop,
                                    DominatorTree *DT);

  /// Returns true if the value V is uniform within the loop.
  bool isUniform(Value *V) const;

  unsigned getMaxSafeDepDistBytes() const { return MaxSafeDepDistBytes; }
  unsigned getNumStores() const { return NumStores; }
  unsigned getNumLoads() const { return NumLoads;}

  /// \brief Add code that checks at runtime if the accessed arrays overlap.
  ///
  /// Returns a pair of instructions where the first element is the first
  /// instruction generated in possibly a sequence of instructions and the
  /// second value is the final comparator value or NULL if no check is needed.
  ///
  /// If \p PtrPartition is set, it contains the partition number for pointers
  /// (-1 if the pointer belongs to multiple partitions).  In this case omit
  /// checks between pointers belonging to the same partition.
  std::pair<Instruction *, Instruction *>
  addRuntimeCheck(Instruction *Loc,
                  const SmallVectorImpl<int> *PtrPartition = nullptr) const;

  /// \brief The diagnostics report generated for the analysis.  E.g. why we
  /// couldn't analyze the loop.
  const Optional<LoopAccessReport> &getReport() const { return Report; }

  /// \brief the Memory Dependence Checker which can determine the
  /// loop-independent and loop-carried dependences between memory accesses.
  const MemoryDepChecker &getDepChecker() const { return DepChecker; }

  /// \brief Return the list of instructions that use \p Ptr to read or write
  /// memory.
  SmallVector<Instruction *, 4> getInstructionsForAccess(Value *Ptr,
                                                         bool isWrite) const {
    return DepChecker.getInstructionsForAccess(Ptr, isWrite);
  }

  /// \brief Print the information about the memory accesses in the loop.
  void print(raw_ostream &OS, unsigned Depth = 0) const;

  /// \brief Used to ensure that if the analysis was run with speculating the
  /// value of symbolic strides, the client queries it with the same assumption.
  /// Only used in DEBUG build but we don't want NDEBUG-dependent ABI.
  unsigned NumSymbolicStrides;

  /// \brief Checks existence of store to invariant address inside loop.
  /// If the loop has any store to invariant address, then it returns true,
  /// else returns false.
  bool hasStoreToLoopInvariantAddress() const {
    return StoreToLoopInvariantAddress;
  }

private:
  /// \brief Analyze the loop.  Substitute symbolic strides using Strides.
  void analyzeLoop(const ValueToValueMap &Strides);

  /// \brief Check if the structure of the loop allows it to be analyzed by this
  /// pass.
  bool canAnalyzeLoop();

  void emitAnalysis(LoopAccessReport &Message);

  /// We need to check that all of the pointers in this list are disjoint
  /// at runtime.
  RuntimePointerChecking PtrRtChecking;

  /// \brief the Memory Dependence Checker which can determine the
  /// loop-independent and loop-carried dependences between memory accesses.
  MemoryDepChecker DepChecker;

  Loop *TheLoop;
  ScalarEvolution *SE;
  const DataLayout &DL;
  const TargetLibraryInfo *TLI;
  AliasAnalysis *AA;
  DominatorTree *DT;
  LoopInfo *LI;

  unsigned NumLoads;
  unsigned NumStores;

  unsigned MaxSafeDepDistBytes;

  /// \brief Cache the result of analyzeLoop.
  bool CanVecMem;

  /// \brief Indicator for storing to uniform addresses.
  /// If a loop has write to a loop invariant address then it should be true.
  bool StoreToLoopInvariantAddress;

  /// \brief The diagnostics report generated for the analysis.  E.g. why we
  /// couldn't analyze the loop.
  Optional<LoopAccessReport> Report;
};

Value *stripIntegerCast(Value *V);

///\brief Return the SCEV corresponding to a pointer with the symbolic stride
///replaced with constant one.
///
/// If \p OrigPtr is not null, use it to look up the stride value instead of \p
/// Ptr.  \p PtrToStride provides the mapping between the pointer value and its
/// stride as collected by LoopVectorizationLegality::collectStridedAccess.
const SCEV *replaceSymbolicStrideSCEV(ScalarEvolution *SE,
                                      const ValueToValueMap &PtrToStride,
                                      Value *Ptr, Value *OrigPtr = nullptr);

/// \brief Check the stride of the pointer and ensure that it does not wrap in
/// the address space.
int isStridedPtr(ScalarEvolution *SE, Value *Ptr, const Loop *Lp,
                 const ValueToValueMap &StridesMap);

/// \brief This analysis provides dependence information for the memory accesses
/// of a loop.
///
/// It runs the analysis for a loop on demand.  This can be initiated by
/// querying the loop access info via LAA::getInfo.  getInfo return a
/// LoopAccessInfo object.  See this class for the specifics of what information
/// is provided.
class LoopAccessAnalysis : public FunctionPass {
public:
  static char ID;

  LoopAccessAnalysis() : FunctionPass(ID) {
    initializeLoopAccessAnalysisPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  /// \brief Query the result of the loop access information for the loop \p L.
  ///
  /// If the client speculates (and then issues run-time checks) for the values
  /// of symbolic strides, \p Strides provides the mapping (see
  /// replaceSymbolicStrideSCEV).  If there is no cached result available run
  /// the analysis.
  const LoopAccessInfo &getInfo(Loop *L, const ValueToValueMap &Strides);

  void releaseMemory() override {
    // Invalidate the cache when the pass is freed.
    LoopAccessInfoMap.clear();
  }

  /// \brief Print the result of the analysis when invoked with -analyze.
  void print(raw_ostream &OS, const Module *M = nullptr) const override;

private:
  /// \brief The cache.
  DenseMap<Loop *, std::unique_ptr<LoopAccessInfo>> LoopAccessInfoMap;

  // The used analysis passes.
  ScalarEvolution *SE;
  const TargetLibraryInfo *TLI;
  AliasAnalysis *AA;
  DominatorTree *DT;
  LoopInfo *LI;
};
} // End llvm namespace

#endif
