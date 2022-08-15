//===- llvm/Analysis/AliasAnalysis.h - Alias Analysis Interface -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the generic AliasAnalysis interface, which is used as the
// common interface used by all clients of alias analysis information, and
// implemented by all alias analysis implementations.  Mod/Ref information is
// also captured by this interface.
//
// Implementations of this interface must implement the various virtual methods,
// which automatically provides functionality for the entire suite of client
// APIs.
//
// This API identifies memory regions with the MemoryLocation class. The pointer
// component specifies the base memory address of the region. The Size specifies
// the maximum size (in address units) of the memory region, or
// MemoryLocation::UnknownSize if the size is not known. The TBAA tag
// identifies the "type" of the memory reference; see the
// TypeBasedAliasAnalysis class for details.
//
// Some non-obvious details include:
//  - Pointers that point to two completely different objects in memory never
//    alias, regardless of the value of the Size component.
//  - NoAlias doesn't imply inequal pointers. The most obvious example of this
//    is two pointers to constant memory. Even if they are equal, constant
//    memory is never stored to, so there will never be any dependencies.
//    In this and other situations, the pointers may be both NoAlias and
//    MustAlias at the same time. The current API can only return one result,
//    though this is rarely a problem in practice.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_ALIASANALYSIS_H
#define LLVM_ANALYSIS_ALIASANALYSIS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Analysis/MemoryLocation.h"

namespace llvm {

class LoadInst;
class StoreInst;
class VAArgInst;
class DataLayout;
class TargetLibraryInfo;
class Pass;
class AnalysisUsage;
class MemTransferInst;
class MemIntrinsic;
class DominatorTree;

/// The possible results of an alias query.
///
/// These results are always computed between two MemoryLocation objects as
/// a query to some alias analysis.
///
/// Note that these are unscoped enumerations because we would like to support
/// implicitly testing a result for the existence of any possible aliasing with
/// a conversion to bool, but an "enum class" doesn't support this. The
/// canonical names from the literature are suffixed and unique anyways, and so
/// they serve as global constants in LLVM for these results.
///
/// See docs/AliasAnalysis.html for more information on the specific meanings
/// of these values.
enum AliasResult {
  /// The two locations do not alias at all.
  ///
  /// This value is arranged to convert to false, while all other values
  /// convert to true. This allows a boolean context to convert the result to
  /// a binary flag indicating whether there is the possibility of aliasing.
  NoAlias = 0,
  /// The two locations may or may not alias. This is the least precise result.
  MayAlias,
  /// The two locations alias, but only due to a partial overlap.
  PartialAlias,
  /// The two locations precisely alias each other.
  MustAlias,
};

class AliasAnalysis {
protected:
  const DataLayout *DL;
  const TargetLibraryInfo *TLI;

private:
  AliasAnalysis *AA;       // Previous Alias Analysis to chain to.

protected:
  /// InitializeAliasAnalysis - Subclasses must call this method to initialize
  /// the AliasAnalysis interface before any other methods are called.  This is
  /// typically called by the run* methods of these subclasses.  This may be
  /// called multiple times.
  ///
  void InitializeAliasAnalysis(Pass *P, const DataLayout *DL);

  /// getAnalysisUsage - All alias analysis implementations should invoke this
  /// directly (using AliasAnalysis::getAnalysisUsage(AU)).
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;

public:
  static char ID; // Class identification, replacement for typeinfo
  AliasAnalysis() : DL(nullptr), TLI(nullptr), AA(nullptr) {}
  virtual ~AliasAnalysis();  // We want to be subclassed

  /// getTargetLibraryInfo - Return a pointer to the current TargetLibraryInfo
  /// object, or null if no TargetLibraryInfo object is available.
  ///
  const TargetLibraryInfo *getTargetLibraryInfo() const { return TLI; }

  /// getTypeStoreSize - Return the DataLayout store size for the given type,
  /// if known, or a conservative value otherwise.
  ///
  uint64_t getTypeStoreSize(Type *Ty);

  //===--------------------------------------------------------------------===//
  /// Alias Queries...
  ///

  /// alias - The main low level interface to the alias analysis implementation.
  /// Returns an AliasResult indicating whether the two pointers are aliased to
  /// each other.  This is the interface that must be implemented by specific
  /// alias analysis implementations.
  virtual AliasResult alias(const MemoryLocation &LocA,
                            const MemoryLocation &LocB);

  /// alias - A convenience wrapper.
  AliasResult alias(const Value *V1, uint64_t V1Size,
                    const Value *V2, uint64_t V2Size) {
    return alias(MemoryLocation(V1, V1Size), MemoryLocation(V2, V2Size));
  }

  /// alias - A convenience wrapper.
  AliasResult alias(const Value *V1, const Value *V2) {
    return alias(V1, MemoryLocation::UnknownSize, V2,
                 MemoryLocation::UnknownSize);
  }

  /// isNoAlias - A trivial helper function to check to see if the specified
  /// pointers are no-alias.
  bool isNoAlias(const MemoryLocation &LocA, const MemoryLocation &LocB) {
    return alias(LocA, LocB) == NoAlias;
  }

  /// isNoAlias - A convenience wrapper.
  bool isNoAlias(const Value *V1, uint64_t V1Size,
                 const Value *V2, uint64_t V2Size) {
    return isNoAlias(MemoryLocation(V1, V1Size), MemoryLocation(V2, V2Size));
  }
  
  /// isNoAlias - A convenience wrapper.
  bool isNoAlias(const Value *V1, const Value *V2) {
    return isNoAlias(MemoryLocation(V1), MemoryLocation(V2));
  }
  
  /// isMustAlias - A convenience wrapper.
  bool isMustAlias(const MemoryLocation &LocA, const MemoryLocation &LocB) {
    return alias(LocA, LocB) == MustAlias;
  }

  /// isMustAlias - A convenience wrapper.
  bool isMustAlias(const Value *V1, const Value *V2) {
    return alias(V1, 1, V2, 1) == MustAlias;
  }
  
  /// pointsToConstantMemory - If the specified memory location is
  /// known to be constant, return true. If OrLocal is true and the
  /// specified memory location is known to be "local" (derived from
  /// an alloca), return true. Otherwise return false.
  virtual bool pointsToConstantMemory(const MemoryLocation &Loc,
                                      bool OrLocal = false);

  /// pointsToConstantMemory - A convenient wrapper.
  bool pointsToConstantMemory(const Value *P, bool OrLocal = false) {
    return pointsToConstantMemory(MemoryLocation(P), OrLocal);
  }

  //===--------------------------------------------------------------------===//
  /// Simple mod/ref information...
  ///

  /// ModRefResult - Represent the result of a mod/ref query.  Mod and Ref are
  /// bits which may be or'd together.
  ///
  enum ModRefResult { NoModRef = 0, Ref = 1, Mod = 2, ModRef = 3 };

  /// These values define additional bits used to define the
  /// ModRefBehavior values.
  enum { Nowhere = 0, ArgumentPointees = 4, Anywhere = 8 | ArgumentPointees };

  /// ModRefBehavior - Summary of how a function affects memory in the program.
  /// Loads from constant globals are not considered memory accesses for this
  /// interface.  Also, functions may freely modify stack space local to their
  /// invocation without having to report it through these interfaces.
  enum ModRefBehavior {
    /// DoesNotAccessMemory - This function does not perform any non-local loads
    /// or stores to memory.
    ///
    /// This property corresponds to the GCC 'const' attribute.
    /// This property corresponds to the LLVM IR 'readnone' attribute.
    /// This property corresponds to the IntrNoMem LLVM intrinsic flag.
    DoesNotAccessMemory = Nowhere | NoModRef,

    /// OnlyReadsArgumentPointees - The only memory references in this function
    /// (if it has any) are non-volatile loads from objects pointed to by its
    /// pointer-typed arguments, with arbitrary offsets.
    ///
    /// This property corresponds to the LLVM IR 'argmemonly' attribute combined
    /// with 'readonly' attribute.
    /// This property corresponds to the IntrReadArgMem LLVM intrinsic flag.
    OnlyReadsArgumentPointees = ArgumentPointees | Ref,

    /// OnlyAccessesArgumentPointees - The only memory references in this
    /// function (if it has any) are non-volatile loads and stores from objects
    /// pointed to by its pointer-typed arguments, with arbitrary offsets.
    ///
    /// This property corresponds to the LLVM IR 'argmemonly' attribute.
    /// This property corresponds to the IntrReadWriteArgMem LLVM intrinsic flag.
    OnlyAccessesArgumentPointees = ArgumentPointees | ModRef,

    /// OnlyReadsMemory - This function does not perform any non-local stores or
    /// volatile loads, but may read from any memory location.
    ///
    /// This property corresponds to the GCC 'pure' attribute.
    /// This property corresponds to the LLVM IR 'readonly' attribute.
    /// This property corresponds to the IntrReadMem LLVM intrinsic flag.
    OnlyReadsMemory = Anywhere | Ref,

    /// UnknownModRefBehavior - This indicates that the function could not be
    /// classified into one of the behaviors above.
    UnknownModRefBehavior = Anywhere | ModRef
  };

  /// Get the ModRef info associated with a pointer argument of a callsite. The
  /// result's bits are set to indicate the allowed aliasing ModRef kinds. Note
  /// that these bits do not necessarily account for the overall behavior of
  /// the function, but rather only provide additional per-argument
  /// information.
  virtual ModRefResult getArgModRefInfo(ImmutableCallSite CS, unsigned ArgIdx);

  /// getModRefBehavior - Return the behavior when calling the given call site.
  virtual ModRefBehavior getModRefBehavior(ImmutableCallSite CS);

  /// getModRefBehavior - Return the behavior when calling the given function.
  /// For use when the call site is not known.
  virtual ModRefBehavior getModRefBehavior(const Function *F);

  /// doesNotAccessMemory - If the specified call is known to never read or
  /// write memory, return true.  If the call only reads from known-constant
  /// memory, it is also legal to return true.  Calls that unwind the stack
  /// are legal for this predicate.
  ///
  /// Many optimizations (such as CSE and LICM) can be performed on such calls
  /// without worrying about aliasing properties, and many calls have this
  /// property (e.g. calls to 'sin' and 'cos').
  ///
  /// This property corresponds to the GCC 'const' attribute.
  ///
  bool doesNotAccessMemory(ImmutableCallSite CS) {
    return getModRefBehavior(CS) == DoesNotAccessMemory;
  }

  /// doesNotAccessMemory - If the specified function is known to never read or
  /// write memory, return true.  For use when the call site is not known.
  ///
  bool doesNotAccessMemory(const Function *F) {
    return getModRefBehavior(F) == DoesNotAccessMemory;
  }

  /// onlyReadsMemory - If the specified call is known to only read from
  /// non-volatile memory (or not access memory at all), return true.  Calls
  /// that unwind the stack are legal for this predicate.
  ///
  /// This property allows many common optimizations to be performed in the
  /// absence of interfering store instructions, such as CSE of strlen calls.
  ///
  /// This property corresponds to the GCC 'pure' attribute.
  ///
  bool onlyReadsMemory(ImmutableCallSite CS) {
    return onlyReadsMemory(getModRefBehavior(CS));
  }

  /// onlyReadsMemory - If the specified function is known to only read from
  /// non-volatile memory (or not access memory at all), return true.  For use
  /// when the call site is not known.
  ///
  bool onlyReadsMemory(const Function *F) {
    return onlyReadsMemory(getModRefBehavior(F));
  }

  /// onlyReadsMemory - Return true if functions with the specified behavior are
  /// known to only read from non-volatile memory (or not access memory at all).
  ///
  static bool onlyReadsMemory(ModRefBehavior MRB) {
    return !(MRB & Mod);
  }

  /// onlyAccessesArgPointees - Return true if functions with the specified
  /// behavior are known to read and write at most from objects pointed to by
  /// their pointer-typed arguments (with arbitrary offsets).
  ///
  static bool onlyAccessesArgPointees(ModRefBehavior MRB) {
    return !(MRB & Anywhere & ~ArgumentPointees);
  }

  /// doesAccessArgPointees - Return true if functions with the specified
  /// behavior are known to potentially read or write from objects pointed
  /// to be their pointer-typed arguments (with arbitrary offsets).
  ///
  static bool doesAccessArgPointees(ModRefBehavior MRB) {
    return (MRB & ModRef) && (MRB & ArgumentPointees);
  }

  /// getModRefInfo - Return information about whether or not an
  /// instruction may read or write memory (without regard to a
  /// specific location)
  ModRefResult getModRefInfo(const Instruction *I) {
    if (auto CS = ImmutableCallSite(I)) {
      auto MRB = getModRefBehavior(CS);
      if (MRB & ModRef)
        return ModRef;
      else if (MRB & Ref)
        return Ref;
      else if (MRB & Mod)
        return Mod;
      return NoModRef;
    }

    return getModRefInfo(I, MemoryLocation());
  }

  /// getModRefInfo - Return information about whether or not an instruction may
  /// read or write the specified memory location.  An instruction
  /// that doesn't read or write memory may be trivially LICM'd for example.
  ModRefResult getModRefInfo(const Instruction *I, const MemoryLocation &Loc) {
    switch (I->getOpcode()) {
    case Instruction::VAArg:  return getModRefInfo((const VAArgInst*)I, Loc);
    case Instruction::Load:   return getModRefInfo((const LoadInst*)I,  Loc);
    case Instruction::Store:  return getModRefInfo((const StoreInst*)I, Loc);
    case Instruction::Fence:  return getModRefInfo((const FenceInst*)I, Loc);
    case Instruction::AtomicCmpXchg:
      return getModRefInfo((const AtomicCmpXchgInst*)I, Loc);
    case Instruction::AtomicRMW:
      return getModRefInfo((const AtomicRMWInst*)I, Loc);
    case Instruction::Call:   return getModRefInfo((const CallInst*)I,  Loc);
    case Instruction::Invoke: return getModRefInfo((const InvokeInst*)I,Loc);
    default:                  return NoModRef;
    }
  }

  /// getModRefInfo - A convenience wrapper.
  ModRefResult getModRefInfo(const Instruction *I,
                             const Value *P, uint64_t Size) {
    return getModRefInfo(I, MemoryLocation(P, Size));
  }

  /// getModRefInfo (for call sites) - Return information about whether
  /// a particular call site modifies or reads the specified memory location.
  virtual ModRefResult getModRefInfo(ImmutableCallSite CS,
                                     const MemoryLocation &Loc);

  /// getModRefInfo (for call sites) - A convenience wrapper.
  ModRefResult getModRefInfo(ImmutableCallSite CS,
                             const Value *P, uint64_t Size) {
    return getModRefInfo(CS, MemoryLocation(P, Size));
  }

  /// getModRefInfo (for calls) - Return information about whether
  /// a particular call modifies or reads the specified memory location.
  ModRefResult getModRefInfo(const CallInst *C, const MemoryLocation &Loc) {
    return getModRefInfo(ImmutableCallSite(C), Loc);
  }

  /// getModRefInfo (for calls) - A convenience wrapper.
  ModRefResult getModRefInfo(const CallInst *C, const Value *P, uint64_t Size) {
    return getModRefInfo(C, MemoryLocation(P, Size));
  }

  /// getModRefInfo (for invokes) - Return information about whether
  /// a particular invoke modifies or reads the specified memory location.
  ModRefResult getModRefInfo(const InvokeInst *I, const MemoryLocation &Loc) {
    return getModRefInfo(ImmutableCallSite(I), Loc);
  }

  /// getModRefInfo (for invokes) - A convenience wrapper.
  ModRefResult getModRefInfo(const InvokeInst *I,
                             const Value *P, uint64_t Size) {
    return getModRefInfo(I, MemoryLocation(P, Size));
  }

  /// getModRefInfo (for loads) - Return information about whether
  /// a particular load modifies or reads the specified memory location.
  ModRefResult getModRefInfo(const LoadInst *L, const MemoryLocation &Loc);

  /// getModRefInfo (for loads) - A convenience wrapper.
  ModRefResult getModRefInfo(const LoadInst *L, const Value *P, uint64_t Size) {
    return getModRefInfo(L, MemoryLocation(P, Size));
  }

  /// getModRefInfo (for stores) - Return information about whether
  /// a particular store modifies or reads the specified memory location.
  ModRefResult getModRefInfo(const StoreInst *S, const MemoryLocation &Loc);

  /// getModRefInfo (for stores) - A convenience wrapper.
  ModRefResult getModRefInfo(const StoreInst *S, const Value *P, uint64_t Size){
    return getModRefInfo(S, MemoryLocation(P, Size));
  }

  /// getModRefInfo (for fences) - Return information about whether
  /// a particular store modifies or reads the specified memory location.
  ModRefResult getModRefInfo(const FenceInst *S, const MemoryLocation &Loc) {
    // Conservatively correct.  (We could possibly be a bit smarter if
    // Loc is a alloca that doesn't escape.)
    return ModRef;
  }

  /// getModRefInfo (for fences) - A convenience wrapper.
  ModRefResult getModRefInfo(const FenceInst *S, const Value *P, uint64_t Size){
    return getModRefInfo(S, MemoryLocation(P, Size));
  }

  /// getModRefInfo (for cmpxchges) - Return information about whether
  /// a particular cmpxchg modifies or reads the specified memory location.
  ModRefResult getModRefInfo(const AtomicCmpXchgInst *CX,
                             const MemoryLocation &Loc);

  /// getModRefInfo (for cmpxchges) - A convenience wrapper.
  ModRefResult getModRefInfo(const AtomicCmpXchgInst *CX,
                             const Value *P, unsigned Size) {
    return getModRefInfo(CX, MemoryLocation(P, Size));
  }

  /// getModRefInfo (for atomicrmws) - Return information about whether
  /// a particular atomicrmw modifies or reads the specified memory location.
  ModRefResult getModRefInfo(const AtomicRMWInst *RMW,
                             const MemoryLocation &Loc);

  /// getModRefInfo (for atomicrmws) - A convenience wrapper.
  ModRefResult getModRefInfo(const AtomicRMWInst *RMW,
                             const Value *P, unsigned Size) {
    return getModRefInfo(RMW, MemoryLocation(P, Size));
  }

  /// getModRefInfo (for va_args) - Return information about whether
  /// a particular va_arg modifies or reads the specified memory location.
  ModRefResult getModRefInfo(const VAArgInst *I, const MemoryLocation &Loc);

  /// getModRefInfo (for va_args) - A convenience wrapper.
  ModRefResult getModRefInfo(const VAArgInst* I, const Value* P, uint64_t Size){
    return getModRefInfo(I, MemoryLocation(P, Size));
  }
  /// getModRefInfo - Return information about whether a call and an instruction
  /// may refer to the same memory locations.
  ModRefResult getModRefInfo(Instruction *I,
                             ImmutableCallSite Call);

  /// getModRefInfo - Return information about whether two call sites may refer
  /// to the same set of memory locations.  See 
  ///   http://llvm.org/docs/AliasAnalysis.html#ModRefInfo
  /// for details.
  virtual ModRefResult getModRefInfo(ImmutableCallSite CS1,
                                     ImmutableCallSite CS2);

  /// callCapturesBefore - Return information about whether a particular call 
  /// site modifies or reads the specified memory location.
  ModRefResult callCapturesBefore(const Instruction *I,
                                  const MemoryLocation &MemLoc,
                                  DominatorTree *DT);

  /// callCapturesBefore - A convenience wrapper.
  ModRefResult callCapturesBefore(const Instruction *I, const Value *P,
                                  uint64_t Size, DominatorTree *DT) {
    return callCapturesBefore(I, MemoryLocation(P, Size), DT);
  }

  //===--------------------------------------------------------------------===//
  /// Higher level methods for querying mod/ref information.
  ///

  /// canBasicBlockModify - Return true if it is possible for execution of the
  /// specified basic block to modify the location Loc.
  bool canBasicBlockModify(const BasicBlock &BB, const MemoryLocation &Loc);

  /// canBasicBlockModify - A convenience wrapper.
  bool canBasicBlockModify(const BasicBlock &BB, const Value *P, uint64_t Size){
    return canBasicBlockModify(BB, MemoryLocation(P, Size));
  }

  /// canInstructionRangeModRef - Return true if it is possible for the
  /// execution of the specified instructions to mod\ref (according to the
  /// mode) the location Loc. The instructions to consider are all
  /// of the instructions in the range of [I1,I2] INCLUSIVE.
  /// I1 and I2 must be in the same basic block.
  bool canInstructionRangeModRef(const Instruction &I1, const Instruction &I2,
                                 const MemoryLocation &Loc,
                                 const ModRefResult Mode);

  /// canInstructionRangeModRef - A convenience wrapper.
  bool canInstructionRangeModRef(const Instruction &I1,
                                 const Instruction &I2, const Value *Ptr,
                                 uint64_t Size, const ModRefResult Mode) {
    return canInstructionRangeModRef(I1, I2, MemoryLocation(Ptr, Size), Mode);
  }

  //===--------------------------------------------------------------------===//
  /// Methods that clients should call when they transform the program to allow
  /// alias analyses to update their internal data structures.  Note that these
  /// methods may be called on any instruction, regardless of whether or not
  /// they have pointer-analysis implications.
  ///

  /// deleteValue - This method should be called whenever an LLVM Value is
  /// deleted from the program, for example when an instruction is found to be
  /// redundant and is eliminated.
  ///
  virtual void deleteValue(Value *V);

  /// addEscapingUse - This method should be used whenever an escaping use is
  /// added to a pointer value.  Analysis implementations may either return
  /// conservative responses for that value in the future, or may recompute
  /// some or all internal state to continue providing precise responses.
  ///
  /// Escaping uses are considered by anything _except_ the following:
  ///  - GEPs or bitcasts of the pointer
  ///  - Loads through the pointer
  ///  - Stores through (but not of) the pointer
  virtual void addEscapingUse(Use &U);

  /// replaceWithNewValue - This method is the obvious combination of the two
  /// above, and it provided as a helper to simplify client code.
  ///
  void replaceWithNewValue(Value *Old, Value *New) {
    deleteValue(Old);
  }
};

/// isNoAliasCall - Return true if this pointer is returned by a noalias
/// function.
bool isNoAliasCall(const Value *V);

/// isNoAliasArgument - Return true if this is an argument with the noalias
/// attribute.
bool isNoAliasArgument(const Value *V);

/// isIdentifiedObject - Return true if this pointer refers to a distinct and
/// identifiable object.  This returns true for:
///    Global Variables and Functions (but not Global Aliases)
///    Allocas
///    ByVal and NoAlias Arguments
///    NoAlias returns (e.g. calls to malloc)
///
bool isIdentifiedObject(const Value *V);

/// isIdentifiedFunctionLocal - Return true if V is umabigously identified
/// at the function-level. Different IdentifiedFunctionLocals can't alias.
/// Further, an IdentifiedFunctionLocal can not alias with any function
/// arguments other than itself, which is not necessarily true for
/// IdentifiedObjects.
bool isIdentifiedFunctionLocal(const Value *V);

} // End llvm namespace

#endif
