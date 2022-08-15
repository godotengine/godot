//===- ScalarReplAggregates.cpp - Scalar Replacement of Aggregates --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This transformation implements the well known scalar replacement of
// aggregates transformation.  This xform breaks up alloca instructions of
// aggregate type (structure or array) into individual alloca instructions for
// each member (if possible).  Then, if possible, it transforms the individual
// alloca instructions into nice clean scalar SSA form.
//
// This combines a simple SRoA algorithm with the Mem2Reg algorithm because they
// often interact, especially for C++ programs.  As such, iterating between
// SRoA, then Mem2Reg until we run out of things to promote works well.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
using namespace llvm;

#define DEBUG_TYPE "scalarrepl"

STATISTIC(NumReplaced,  "Number of allocas broken up");
STATISTIC(NumPromoted,  "Number of allocas promoted");
STATISTIC(NumAdjusted,  "Number of scalar allocas adjusted to allow promotion");
STATISTIC(NumConverted, "Number of aggregates converted to scalar");

namespace {
  struct SROA : public FunctionPass {
    SROA(int T, bool hasDT, char &ID, int ST, int AT, int SLT)
      : FunctionPass(ID), HasDomTree(hasDT) {
      if (T == -1)
        SRThreshold = 128;
      else
        SRThreshold = T;
      if (ST == -1)
        StructMemberThreshold = 32;
      else
        StructMemberThreshold = ST;
      if (AT == -1)
        ArrayElementThreshold = 8;
      else
        ArrayElementThreshold = AT;
      if (SLT == -1)
        // Do not limit the scalar integer load size if no threshold is given.
        ScalarLoadThreshold = -1;
      else
        ScalarLoadThreshold = SLT;
    }

    bool runOnFunction(Function &F) override;

    bool performScalarRepl(Function &F);
    bool performPromotion(Function &F);

  private:
    bool HasDomTree;

    /// DeadInsts - Keep track of instructions we have made dead, so that
    /// we can remove them after we are done working.
    SmallVector<Value*, 32> DeadInsts;

    /// AllocaInfo - When analyzing uses of an alloca instruction, this captures
    /// information about the uses.  All these fields are initialized to false
    /// and set to true when something is learned.
    struct AllocaInfo {
      /// The alloca to promote.
      AllocaInst *AI;

      /// CheckedPHIs - This is a set of verified PHI nodes, to prevent infinite
      /// looping and avoid redundant work.
      SmallPtrSet<PHINode*, 8> CheckedPHIs;

      /// isUnsafe - This is set to true if the alloca cannot be SROA'd.
      bool isUnsafe : 1;

      /// isMemCpySrc - This is true if this aggregate is memcpy'd from.
      bool isMemCpySrc : 1;

      /// isMemCpyDst - This is true if this aggregate is memcpy'd into.
      bool isMemCpyDst : 1;

      /// hasSubelementAccess - This is true if a subelement of the alloca is
      /// ever accessed, or false if the alloca is only accessed with mem
      /// intrinsics or load/store that only access the entire alloca at once.
      bool hasSubelementAccess : 1;

      /// hasALoadOrStore - This is true if there are any loads or stores to it.
      /// The alloca may just be accessed with memcpy, for example, which would
      /// not set this.
      bool hasALoadOrStore : 1;

      explicit AllocaInfo(AllocaInst *ai)
        : AI(ai), isUnsafe(false), isMemCpySrc(false), isMemCpyDst(false),
          hasSubelementAccess(false), hasALoadOrStore(false) {}
    };

    /// SRThreshold - The maximum alloca size to considered for SROA.
    unsigned SRThreshold;

    /// StructMemberThreshold - The maximum number of members a struct can
    /// contain to be considered for SROA.
    unsigned StructMemberThreshold;

    /// ArrayElementThreshold - The maximum number of elements an array can
    /// have to be considered for SROA.
    unsigned ArrayElementThreshold;

    /// ScalarLoadThreshold - The maximum size in bits of scalars to load when
    /// converting to scalar
    unsigned ScalarLoadThreshold;

    void MarkUnsafe(AllocaInfo &I, Instruction *User) {
      I.isUnsafe = true;
      DEBUG(dbgs() << "  Transformation preventing inst: " << *User << '\n');
    }

    bool isSafeAllocaToScalarRepl(AllocaInst *AI);

    void isSafeForScalarRepl(Instruction *I, uint64_t Offset, AllocaInfo &Info);
    void isSafePHISelectUseForScalarRepl(Instruction *User, uint64_t Offset,
                                         AllocaInfo &Info);
    void isSafeGEP(GetElementPtrInst *GEPI, uint64_t &Offset, AllocaInfo &Info);
    void isSafeMemAccess(uint64_t Offset, uint64_t MemSize,
                         Type *MemOpType, bool isStore, AllocaInfo &Info,
                         Instruction *TheAccess, bool AllowWholeAccess);
    bool TypeHasComponent(Type *T, uint64_t Offset, uint64_t Size,
                          const DataLayout &DL);
    uint64_t FindElementAndOffset(Type *&T, uint64_t &Offset, Type *&IdxTy,
                                  const DataLayout &DL);

    void DoScalarReplacement(AllocaInst *AI,
                             std::vector<AllocaInst*> &WorkList);
    void DeleteDeadInstructions();

    void RewriteForScalarRepl(Instruction *I, AllocaInst *AI, uint64_t Offset,
                              SmallVectorImpl<AllocaInst *> &NewElts);
    void RewriteBitCast(BitCastInst *BC, AllocaInst *AI, uint64_t Offset,
                        SmallVectorImpl<AllocaInst *> &NewElts);
    void RewriteGEP(GetElementPtrInst *GEPI, AllocaInst *AI, uint64_t Offset,
                    SmallVectorImpl<AllocaInst *> &NewElts);
    void RewriteLifetimeIntrinsic(IntrinsicInst *II, AllocaInst *AI,
                                  uint64_t Offset,
                                  SmallVectorImpl<AllocaInst *> &NewElts);
    void RewriteMemIntrinUserOfAlloca(MemIntrinsic *MI, Instruction *Inst,
                                      AllocaInst *AI,
                                      SmallVectorImpl<AllocaInst *> &NewElts);
    void RewriteStoreUserOfWholeAlloca(StoreInst *SI, AllocaInst *AI,
                                       SmallVectorImpl<AllocaInst *> &NewElts);
    void RewriteLoadUserOfWholeAlloca(LoadInst *LI, AllocaInst *AI,
                                      SmallVectorImpl<AllocaInst *> &NewElts);
    bool ShouldAttemptScalarRepl(AllocaInst *AI);
  };

  // SROA_DT - SROA that uses DominatorTree.
  struct SROA_DT : public SROA {
    static char ID;
  public:
    SROA_DT(int T = -1, int ST = -1, int AT = -1, int SLT = -1) :
        SROA(T, true, ID, ST, AT, SLT) {
      initializeSROA_DTPass(*PassRegistry::getPassRegistry());
    }

    // getAnalysisUsage - This pass does not require any passes, but we know it
    // will not alter the CFG, so say so.
    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<AssumptionCacheTracker>();
      AU.addRequired<DominatorTreeWrapperPass>();
      AU.setPreservesCFG();
    }
  };

  // SROA_SSAUp - SROA that uses SSAUpdater.
  struct SROA_SSAUp : public SROA {
    static char ID;
  public:
    SROA_SSAUp(int T = -1, int ST = -1, int AT = -1, int SLT = -1) :
        SROA(T, false, ID, ST, AT, SLT) {
      initializeSROA_SSAUpPass(*PassRegistry::getPassRegistry());
    }

    // getAnalysisUsage - This pass does not require any passes, but we know it
    // will not alter the CFG, so say so.
    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<AssumptionCacheTracker>();
      AU.setPreservesCFG();
    }
  };

}

char SROA_DT::ID = 0;
char SROA_SSAUp::ID = 0;

INITIALIZE_PASS_BEGIN(SROA_DT, "scalarrepl",
                "Scalar Replacement of Aggregates (DT)", false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(SROA_DT, "scalarrepl",
                "Scalar Replacement of Aggregates (DT)", false, false)

INITIALIZE_PASS_BEGIN(SROA_SSAUp, "scalarrepl-ssa",
                      "Scalar Replacement of Aggregates (SSAUp)", false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_END(SROA_SSAUp, "scalarrepl-ssa",
                    "Scalar Replacement of Aggregates (SSAUp)", false, false)

// Public interface to the ScalarReplAggregates pass
FunctionPass *llvm::createScalarReplAggregatesPass(int Threshold,
                                                   bool UseDomTree,
                                                   int StructMemberThreshold,
                                                   int ArrayElementThreshold,
                                                   int ScalarLoadThreshold) {
  if (UseDomTree)
    return new SROA_DT(Threshold, StructMemberThreshold, ArrayElementThreshold,
                       ScalarLoadThreshold);
  return new SROA_SSAUp(Threshold, StructMemberThreshold,
                        ArrayElementThreshold, ScalarLoadThreshold);
}


//===----------------------------------------------------------------------===//
// Convert To Scalar Optimization.
//===----------------------------------------------------------------------===//

namespace {
/// ConvertToScalarInfo - This class implements the "Convert To Scalar"
/// optimization, which scans the uses of an alloca and determines if it can
/// rewrite it in terms of a single new alloca that can be mem2reg'd.
class ConvertToScalarInfo {
  /// AllocaSize - The size of the alloca being considered in bytes.
  unsigned AllocaSize;
  const DataLayout &DL;
  unsigned ScalarLoadThreshold;

  /// IsNotTrivial - This is set to true if there is some access to the object
  /// which means that mem2reg can't promote it.
  bool IsNotTrivial;

  /// ScalarKind - Tracks the kind of alloca being considered for promotion,
  /// computed based on the uses of the alloca rather than the LLVM type system.
  enum {
    Unknown,

    // Accesses via GEPs that are consistent with element access of a vector
    // type. This will not be converted into a vector unless there is a later
    // access using an actual vector type.
    ImplicitVector,

    // Accesses via vector operations and GEPs that are consistent with the
    // layout of a vector type.
    Vector,

    // An integer bag-of-bits with bitwise operations for insertion and
    // extraction. Any combination of types can be converted into this kind
    // of scalar.
    Integer
  } ScalarKind;

  /// VectorTy - This tracks the type that we should promote the vector to if
  /// it is possible to turn it into a vector.  This starts out null, and if it
  /// isn't possible to turn into a vector type, it gets set to VoidTy.
  VectorType *VectorTy;

  /// HadNonMemTransferAccess - True if there is at least one access to the
  /// alloca that is not a MemTransferInst.  We don't want to turn structs into
  /// large integers unless there is some potential for optimization.
  bool HadNonMemTransferAccess;

  /// HadDynamicAccess - True if some element of this alloca was dynamic.
  /// We don't yet have support for turning a dynamic access into a large
  /// integer.
  bool HadDynamicAccess;

public:
  explicit ConvertToScalarInfo(unsigned Size, const DataLayout &DL,
                               unsigned SLT)
    : AllocaSize(Size), DL(DL), ScalarLoadThreshold(SLT), IsNotTrivial(false),
    ScalarKind(Unknown), VectorTy(nullptr), HadNonMemTransferAccess(false),
    HadDynamicAccess(false) { }

  AllocaInst *TryConvert(AllocaInst *AI);

private:
  bool CanConvertToScalar(Value *V, uint64_t Offset, Value* NonConstantIdx);
  void MergeInTypeForLoadOrStore(Type *In, uint64_t Offset);
  bool MergeInVectorType(VectorType *VInTy, uint64_t Offset);
  void ConvertUsesToScalar(Value *Ptr, AllocaInst *NewAI, uint64_t Offset,
                           Value *NonConstantIdx);

  Value *ConvertScalar_ExtractValue(Value *NV, Type *ToType,
                                    uint64_t Offset, Value* NonConstantIdx,
                                    IRBuilder<> &Builder);
  Value *ConvertScalar_InsertValue(Value *StoredVal, Value *ExistingVal,
                                   uint64_t Offset, Value* NonConstantIdx,
                                   IRBuilder<> &Builder);
};
} // end anonymous namespace.


/// TryConvert - Analyze the specified alloca, and if it is safe to do so,
/// rewrite it to be a new alloca which is mem2reg'able.  This returns the new
/// alloca if possible or null if not.
AllocaInst *ConvertToScalarInfo::TryConvert(AllocaInst *AI) {
  // If we can't convert this scalar, or if mem2reg can trivially do it, bail
  // out.
  if (!CanConvertToScalar(AI, 0, nullptr) || !IsNotTrivial)
    return nullptr;

  // If an alloca has only memset / memcpy uses, it may still have an Unknown
  // ScalarKind. Treat it as an Integer below.
  if (ScalarKind == Unknown)
    ScalarKind = Integer;

  if (ScalarKind == Vector && VectorTy->getBitWidth() != AllocaSize * 8)
    ScalarKind = Integer;

  // If we were able to find a vector type that can handle this with
  // insert/extract elements, and if there was at least one use that had
  // a vector type, promote this to a vector.  We don't want to promote
  // random stuff that doesn't use vectors (e.g. <9 x double>) because then
  // we just get a lot of insert/extracts.  If at least one vector is
  // involved, then we probably really do have a union of vector/array.
  Type *NewTy;
  if (ScalarKind == Vector) {
    assert(VectorTy && "Missing type for vector scalar.");
    DEBUG(dbgs() << "CONVERT TO VECTOR: " << *AI << "\n  TYPE = "
          << *VectorTy << '\n');
    NewTy = VectorTy;  // Use the vector type.
  } else {
    unsigned BitWidth = AllocaSize * 8;

    // Do not convert to scalar integer if the alloca size exceeds the
    // scalar load threshold.
    if (BitWidth > ScalarLoadThreshold)
      return nullptr;

    if ((ScalarKind == ImplicitVector || ScalarKind == Integer) &&
        !HadNonMemTransferAccess && !DL.fitsInLegalInteger(BitWidth))
      return nullptr;
    // Dynamic accesses on integers aren't yet supported.  They need us to shift
    // by a dynamic amount which could be difficult to work out as we might not
    // know whether to use a left or right shift.
    if (ScalarKind == Integer && HadDynamicAccess)
      return nullptr;

    DEBUG(dbgs() << "CONVERT TO SCALAR INTEGER: " << *AI << "\n");
    // Create and insert the integer alloca.
    NewTy = IntegerType::get(AI->getContext(), BitWidth);
  }
  AllocaInst *NewAI = new AllocaInst(NewTy, nullptr, "",
                                     AI->getParent()->begin());
  ConvertUsesToScalar(AI, NewAI, 0, nullptr);
  return NewAI;
}

/// MergeInTypeForLoadOrStore - Add the 'In' type to the accumulated vector type
/// (VectorTy) so far at the offset specified by Offset (which is specified in
/// bytes).
///
/// There are two cases we handle here:
///   1) A union of vector types of the same size and potentially its elements.
///      Here we turn element accesses into insert/extract element operations.
///      This promotes a <4 x float> with a store of float to the third element
///      into a <4 x float> that uses insert element.
///   2) A fully general blob of memory, which we turn into some (potentially
///      large) integer type with extract and insert operations where the loads
///      and stores would mutate the memory.  We mark this by setting VectorTy
///      to VoidTy.
void ConvertToScalarInfo::MergeInTypeForLoadOrStore(Type *In,
                                                    uint64_t Offset) {
  // If we already decided to turn this into a blob of integer memory, there is
  // nothing to be done.
  if (ScalarKind == Integer)
    return;

  // If this could be contributing to a vector, analyze it.

  // If the In type is a vector that is the same size as the alloca, see if it
  // matches the existing VecTy.
  if (VectorType *VInTy = dyn_cast<VectorType>(In)) {
    if (MergeInVectorType(VInTy, Offset))
      return;
  } else if (In->isFloatTy() || In->isDoubleTy() ||
             (In->isIntegerTy() && In->getPrimitiveSizeInBits() >= 8 &&
              isPowerOf2_32(In->getPrimitiveSizeInBits()))) {
    // Full width accesses can be ignored, because they can always be turned
    // into bitcasts.
    unsigned EltSize = In->getPrimitiveSizeInBits()/8;
    if (EltSize == AllocaSize)
      return;

    // If we're accessing something that could be an element of a vector, see
    // if the implied vector agrees with what we already have and if Offset is
    // compatible with it.
    if (Offset % EltSize == 0 && AllocaSize % EltSize == 0 &&
        (!VectorTy || EltSize == VectorTy->getElementType()
                                         ->getPrimitiveSizeInBits()/8)) {
      if (!VectorTy) {
        ScalarKind = ImplicitVector;
        VectorTy = VectorType::get(In, AllocaSize/EltSize);
      }
      return;
    }
  }

  // Otherwise, we have a case that we can't handle with an optimized vector
  // form.  We can still turn this into a large integer.
  ScalarKind = Integer;
}

/// MergeInVectorType - Handles the vector case of MergeInTypeForLoadOrStore,
/// returning true if the type was successfully merged and false otherwise.
bool ConvertToScalarInfo::MergeInVectorType(VectorType *VInTy,
                                            uint64_t Offset) {
  if (VInTy->getBitWidth()/8 == AllocaSize && Offset == 0) {
    // If we're storing/loading a vector of the right size, allow it as a
    // vector.  If this the first vector we see, remember the type so that
    // we know the element size. If this is a subsequent access, ignore it
    // even if it is a differing type but the same size. Worst case we can
    // bitcast the resultant vectors.
    if (!VectorTy)
      VectorTy = VInTy;
    ScalarKind = Vector;
    return true;
  }

  return false;
}

/// CanConvertToScalar - V is a pointer.  If we can convert the pointee and all
/// its accesses to a single vector type, return true and set VecTy to
/// the new type.  If we could convert the alloca into a single promotable
/// integer, return true but set VecTy to VoidTy.  Further, if the use is not a
/// completely trivial use that mem2reg could promote, set IsNotTrivial.  Offset
/// is the current offset from the base of the alloca being analyzed.
///
/// If we see at least one access to the value that is as a vector type, set the
/// SawVec flag.
bool ConvertToScalarInfo::CanConvertToScalar(Value *V, uint64_t Offset,
                                             Value* NonConstantIdx) {
  for (User *U : V->users()) {
    Instruction *UI = cast<Instruction>(U);

    if (LoadInst *LI = dyn_cast<LoadInst>(UI)) {
      // Don't break volatile loads.
      if (!LI->isSimple())
        return false;
      // Don't touch MMX operations.
      if (LI->getType()->isX86_MMXTy())
        return false;
      HadNonMemTransferAccess = true;
      MergeInTypeForLoadOrStore(LI->getType(), Offset);
      continue;
    }

    if (StoreInst *SI = dyn_cast<StoreInst>(UI)) {
      // Storing the pointer, not into the value?
      if (SI->getOperand(0) == V || !SI->isSimple()) return false;
      // Don't touch MMX operations.
      if (SI->getOperand(0)->getType()->isX86_MMXTy())
        return false;
      HadNonMemTransferAccess = true;
      MergeInTypeForLoadOrStore(SI->getOperand(0)->getType(), Offset);
      continue;
    }

    if (BitCastInst *BCI = dyn_cast<BitCastInst>(UI)) {
      if (!onlyUsedByLifetimeMarkers(BCI))
        IsNotTrivial = true;  // Can't be mem2reg'd.
      if (!CanConvertToScalar(BCI, Offset, NonConstantIdx))
        return false;
      continue;
    }

    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(UI)) {
      // If this is a GEP with a variable indices, we can't handle it.
      PointerType* PtrTy = dyn_cast<PointerType>(GEP->getPointerOperandType());
      if (!PtrTy)
        return false;

      // Compute the offset that this GEP adds to the pointer.
      SmallVector<Value*, 8> Indices(GEP->op_begin()+1, GEP->op_end());
      Value *GEPNonConstantIdx = nullptr;
      if (!GEP->hasAllConstantIndices()) {
        if (!isa<VectorType>(PtrTy->getElementType()))
          return false;
        if (NonConstantIdx)
          return false;
        GEPNonConstantIdx = Indices.pop_back_val();
        if (!GEPNonConstantIdx->getType()->isIntegerTy(32))
          return false;
        HadDynamicAccess = true;
      } else
        GEPNonConstantIdx = NonConstantIdx;
      uint64_t GEPOffset = DL.getIndexedOffset(PtrTy,
                                               Indices);
      // See if all uses can be converted.
      if (!CanConvertToScalar(GEP, Offset+GEPOffset, GEPNonConstantIdx))
        return false;
      IsNotTrivial = true;  // Can't be mem2reg'd.
      HadNonMemTransferAccess = true;
      continue;
    }

    // If this is a constant sized memset of a constant value (e.g. 0) we can
    // handle it.
    if (MemSetInst *MSI = dyn_cast<MemSetInst>(UI)) {
      // Store to dynamic index.
      if (NonConstantIdx)
        return false;
      // Store of constant value.
      if (!isa<ConstantInt>(MSI->getValue()))
        return false;

      // Store of constant size.
      ConstantInt *Len = dyn_cast<ConstantInt>(MSI->getLength());
      if (!Len)
        return false;

      // If the size differs from the alloca, we can only convert the alloca to
      // an integer bag-of-bits.
      // FIXME: This should handle all of the cases that are currently accepted
      // as vector element insertions.
      if (Len->getZExtValue() != AllocaSize || Offset != 0)
        ScalarKind = Integer;

      IsNotTrivial = true;  // Can't be mem2reg'd.
      HadNonMemTransferAccess = true;
      continue;
    }

    // If this is a memcpy or memmove into or out of the whole allocation, we
    // can handle it like a load or store of the scalar type.
    if (MemTransferInst *MTI = dyn_cast<MemTransferInst>(UI)) {
      // Store to dynamic index.
      if (NonConstantIdx)
        return false;
      ConstantInt *Len = dyn_cast<ConstantInt>(MTI->getLength());
      if (!Len || Len->getZExtValue() != AllocaSize || Offset != 0)
        return false;

      IsNotTrivial = true;  // Can't be mem2reg'd.
      continue;
    }

    // If this is a lifetime intrinsic, we can handle it.
    if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(UI)) {
      if (II->getIntrinsicID() == Intrinsic::lifetime_start ||
          II->getIntrinsicID() == Intrinsic::lifetime_end) {
        continue;
      }
    }

    // Otherwise, we cannot handle this!
    return false;
  }

  return true;
}

/// ConvertUsesToScalar - Convert all of the users of Ptr to use the new alloca
/// directly.  This happens when we are converting an "integer union" to a
/// single integer scalar, or when we are converting a "vector union" to a
/// vector with insert/extractelement instructions.
///
/// Offset is an offset from the original alloca, in bits that need to be
/// shifted to the right.  By the end of this, there should be no uses of Ptr.
void ConvertToScalarInfo::ConvertUsesToScalar(Value *Ptr, AllocaInst *NewAI,
                                              uint64_t Offset,
                                              Value* NonConstantIdx) {
  while (!Ptr->use_empty()) {
    Instruction *User = cast<Instruction>(Ptr->user_back());

    if (BitCastInst *CI = dyn_cast<BitCastInst>(User)) {
      ConvertUsesToScalar(CI, NewAI, Offset, NonConstantIdx);
      CI->eraseFromParent();
      continue;
    }

    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(User)) {
      // Compute the offset that this GEP adds to the pointer.
      SmallVector<Value*, 8> Indices(GEP->op_begin()+1, GEP->op_end());
      Value* GEPNonConstantIdx = nullptr;
      if (!GEP->hasAllConstantIndices()) {
        assert(!NonConstantIdx &&
               "Dynamic GEP reading from dynamic GEP unsupported");
        GEPNonConstantIdx = Indices.pop_back_val();
      } else
        GEPNonConstantIdx = NonConstantIdx;
      uint64_t GEPOffset = DL.getIndexedOffset(GEP->getPointerOperandType(),
                                               Indices);
      ConvertUsesToScalar(GEP, NewAI, Offset+GEPOffset*8, GEPNonConstantIdx);
      GEP->eraseFromParent();
      continue;
    }

    IRBuilder<> Builder(User);

    if (LoadInst *LI = dyn_cast<LoadInst>(User)) {
      // The load is a bit extract from NewAI shifted right by Offset bits.
      Value *LoadedVal = Builder.CreateLoad(NewAI);
      Value *NewLoadVal
        = ConvertScalar_ExtractValue(LoadedVal, LI->getType(), Offset,
                                     NonConstantIdx, Builder);
      LI->replaceAllUsesWith(NewLoadVal);
      LI->eraseFromParent();
      continue;
    }

    if (StoreInst *SI = dyn_cast<StoreInst>(User)) {
      assert(SI->getOperand(0) != Ptr && "Consistency error!");
      Instruction *Old = Builder.CreateLoad(NewAI, NewAI->getName()+".in");
      Value *New = ConvertScalar_InsertValue(SI->getOperand(0), Old, Offset,
                                             NonConstantIdx, Builder);
      Builder.CreateStore(New, NewAI);
      SI->eraseFromParent();

      // If the load we just inserted is now dead, then the inserted store
      // overwrote the entire thing.
      if (Old->use_empty())
        Old->eraseFromParent();
      continue;
    }

    // If this is a constant sized memset of a constant value (e.g. 0) we can
    // transform it into a store of the expanded constant value.
    if (MemSetInst *MSI = dyn_cast<MemSetInst>(User)) {
      assert(MSI->getRawDest() == Ptr && "Consistency error!");
      assert(!NonConstantIdx && "Cannot replace dynamic memset with insert");
      int64_t SNumBytes = cast<ConstantInt>(MSI->getLength())->getSExtValue();
      if (SNumBytes > 0 && (SNumBytes >> 32) == 0) {
        unsigned NumBytes = static_cast<unsigned>(SNumBytes);
        unsigned Val = cast<ConstantInt>(MSI->getValue())->getZExtValue();

        // Compute the value replicated the right number of times.
        APInt APVal(NumBytes*8, Val);

        // Splat the value if non-zero.
        if (Val)
          for (unsigned i = 1; i != NumBytes; ++i)
            APVal |= APVal << 8;

        Instruction *Old = Builder.CreateLoad(NewAI, NewAI->getName()+".in");
        Value *New = ConvertScalar_InsertValue(
                                    ConstantInt::get(User->getContext(), APVal),
                                               Old, Offset, nullptr, Builder);
        Builder.CreateStore(New, NewAI);

        // If the load we just inserted is now dead, then the memset overwrote
        // the entire thing.
        if (Old->use_empty())
          Old->eraseFromParent();
      }
      MSI->eraseFromParent();
      continue;
    }

    // If this is a memcpy or memmove into or out of the whole allocation, we
    // can handle it like a load or store of the scalar type.
    if (MemTransferInst *MTI = dyn_cast<MemTransferInst>(User)) {
      assert(Offset == 0 && "must be store to start of alloca");
      assert(!NonConstantIdx && "Cannot replace dynamic transfer with insert");

      // If the source and destination are both to the same alloca, then this is
      // a noop copy-to-self, just delete it.  Otherwise, emit a load and store
      // as appropriate.
      AllocaInst *OrigAI = cast<AllocaInst>(GetUnderlyingObject(Ptr, DL, 0));

      if (GetUnderlyingObject(MTI->getSource(), DL, 0) != OrigAI) {
        // Dest must be OrigAI, change this to be a load from the original
        // pointer (bitcasted), then a store to our new alloca.
        assert(MTI->getRawDest() == Ptr && "Neither use is of pointer?");
        Value *SrcPtr = MTI->getSource();
        PointerType* SPTy = cast<PointerType>(SrcPtr->getType());
        PointerType* AIPTy = cast<PointerType>(NewAI->getType());
        if (SPTy->getAddressSpace() != AIPTy->getAddressSpace()) {
          AIPTy = PointerType::get(AIPTy->getElementType(),
                                   SPTy->getAddressSpace());
        }
        SrcPtr = Builder.CreateBitCast(SrcPtr, AIPTy);

        LoadInst *SrcVal = Builder.CreateLoad(SrcPtr, "srcval");
        SrcVal->setAlignment(MTI->getAlignment());
        Builder.CreateStore(SrcVal, NewAI);
      } else if (GetUnderlyingObject(MTI->getDest(), DL, 0) != OrigAI) {
        // Src must be OrigAI, change this to be a load from NewAI then a store
        // through the original dest pointer (bitcasted).
        assert(MTI->getRawSource() == Ptr && "Neither use is of pointer?");
        LoadInst *SrcVal = Builder.CreateLoad(NewAI, "srcval");

        PointerType* DPTy = cast<PointerType>(MTI->getDest()->getType());
        PointerType* AIPTy = cast<PointerType>(NewAI->getType());
        if (DPTy->getAddressSpace() != AIPTy->getAddressSpace()) {
          AIPTy = PointerType::get(AIPTy->getElementType(),
                                   DPTy->getAddressSpace());
        }
        Value *DstPtr = Builder.CreateBitCast(MTI->getDest(), AIPTy);

        StoreInst *NewStore = Builder.CreateStore(SrcVal, DstPtr);
        NewStore->setAlignment(MTI->getAlignment());
      } else {
        // Noop transfer. Src == Dst
      }

      MTI->eraseFromParent();
      continue;
    }

    if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(User)) {
      if (II->getIntrinsicID() == Intrinsic::lifetime_start ||
          II->getIntrinsicID() == Intrinsic::lifetime_end) {
        // There's no need to preserve these, as the resulting alloca will be
        // converted to a register anyways.
        II->eraseFromParent();
        continue;
      }
    }

    llvm_unreachable("Unsupported operation!");
  }
}

/// ConvertScalar_ExtractValue - Extract a value of type ToType from an integer
/// or vector value FromVal, extracting the bits from the offset specified by
/// Offset.  This returns the value, which is of type ToType.
///
/// This happens when we are converting an "integer union" to a single
/// integer scalar, or when we are converting a "vector union" to a vector with
/// insert/extractelement instructions.
///
/// Offset is an offset from the original alloca, in bits that need to be
/// shifted to the right.
Value *ConvertToScalarInfo::
ConvertScalar_ExtractValue(Value *FromVal, Type *ToType,
                           uint64_t Offset, Value* NonConstantIdx,
                           IRBuilder<> &Builder) {
  // If the load is of the whole new alloca, no conversion is needed.
  Type *FromType = FromVal->getType();
  if (FromType == ToType && Offset == 0)
    return FromVal;

  // If the result alloca is a vector type, this is either an element
  // access or a bitcast to another vector type of the same size.
  if (VectorType *VTy = dyn_cast<VectorType>(FromType)) {
    unsigned FromTypeSize = DL.getTypeAllocSize(FromType);
    unsigned ToTypeSize = DL.getTypeAllocSize(ToType);
    if (FromTypeSize == ToTypeSize)
        return Builder.CreateBitCast(FromVal, ToType);

    // Otherwise it must be an element access.
    unsigned Elt = 0;
    if (Offset) {
      unsigned EltSize = DL.getTypeAllocSizeInBits(VTy->getElementType());
      Elt = Offset/EltSize;
      assert(EltSize*Elt == Offset && "Invalid modulus in validity checking");
    }
    // Return the element extracted out of it.
    Value *Idx;
    if (NonConstantIdx) {
      if (Elt)
        Idx = Builder.CreateAdd(NonConstantIdx,
                                Builder.getInt32(Elt),
                                "dyn.offset");
      else
        Idx = NonConstantIdx;
    } else
      Idx = Builder.getInt32(Elt);
    Value *V = Builder.CreateExtractElement(FromVal, Idx);
    if (V->getType() != ToType)
      V = Builder.CreateBitCast(V, ToType);
    return V;
  }

  // If ToType is a first class aggregate, extract out each of the pieces and
  // use insertvalue's to form the FCA.
  if (StructType *ST = dyn_cast<StructType>(ToType)) {
    assert(!NonConstantIdx &&
           "Dynamic indexing into struct types not supported");
    const StructLayout &Layout = *DL.getStructLayout(ST);
    Value *Res = UndefValue::get(ST);
    for (unsigned i = 0, e = ST->getNumElements(); i != e; ++i) {
      Value *Elt = ConvertScalar_ExtractValue(FromVal, ST->getElementType(i),
                                        Offset+Layout.getElementOffsetInBits(i),
                                              nullptr, Builder);
      Res = Builder.CreateInsertValue(Res, Elt, i);
    }
    return Res;
  }

  if (ArrayType *AT = dyn_cast<ArrayType>(ToType)) {
    assert(!NonConstantIdx &&
           "Dynamic indexing into array types not supported");
    uint64_t EltSize = DL.getTypeAllocSizeInBits(AT->getElementType());
    Value *Res = UndefValue::get(AT);
    for (unsigned i = 0, e = AT->getNumElements(); i != e; ++i) {
      Value *Elt = ConvertScalar_ExtractValue(FromVal, AT->getElementType(),
                                              Offset+i*EltSize, nullptr,
                                              Builder);
      Res = Builder.CreateInsertValue(Res, Elt, i);
    }
    return Res;
  }

  // Otherwise, this must be a union that was converted to an integer value.
  IntegerType *NTy = cast<IntegerType>(FromVal->getType());

  // If this is a big-endian system and the load is narrower than the
  // full alloca type, we need to do a shift to get the right bits.
  int ShAmt = 0;
  if (DL.isBigEndian()) {
    // On big-endian machines, the lowest bit is stored at the bit offset
    // from the pointer given by getTypeStoreSizeInBits.  This matters for
    // integers with a bitwidth that is not a multiple of 8.
    ShAmt = DL.getTypeStoreSizeInBits(NTy) -
            DL.getTypeStoreSizeInBits(ToType) - Offset;
  } else {
    ShAmt = Offset;
  }

  // Note: we support negative bitwidths (with shl) which are not defined.
  // We do this to support (f.e.) loads off the end of a structure where
  // only some bits are used.
  if (ShAmt > 0 && (unsigned)ShAmt < NTy->getBitWidth())
    FromVal = Builder.CreateLShr(FromVal,
                                 ConstantInt::get(FromVal->getType(), ShAmt));
  else if (ShAmt < 0 && (unsigned)-ShAmt < NTy->getBitWidth())
    FromVal = Builder.CreateShl(FromVal,
                                ConstantInt::get(FromVal->getType(), -ShAmt));

  // Finally, unconditionally truncate the integer to the right width.
  unsigned LIBitWidth = DL.getTypeSizeInBits(ToType);
  if (LIBitWidth < NTy->getBitWidth())
    FromVal =
      Builder.CreateTrunc(FromVal, IntegerType::get(FromVal->getContext(),
                                                    LIBitWidth));
  else if (LIBitWidth > NTy->getBitWidth())
    FromVal =
       Builder.CreateZExt(FromVal, IntegerType::get(FromVal->getContext(),
                                                    LIBitWidth));

  // If the result is an integer, this is a trunc or bitcast.
  if (ToType->isIntegerTy()) {
    // Should be done.
  } else if (ToType->isFloatingPointTy() || ToType->isVectorTy()) {
    // Just do a bitcast, we know the sizes match up.
    FromVal = Builder.CreateBitCast(FromVal, ToType);
  } else {
    // Otherwise must be a pointer.
    FromVal = Builder.CreateIntToPtr(FromVal, ToType);
  }
  assert(FromVal->getType() == ToType && "Didn't convert right?");
  return FromVal;
}

/// ConvertScalar_InsertValue - Insert the value "SV" into the existing integer
/// or vector value "Old" at the offset specified by Offset.
///
/// This happens when we are converting an "integer union" to a
/// single integer scalar, or when we are converting a "vector union" to a
/// vector with insert/extractelement instructions.
///
/// Offset is an offset from the original alloca, in bits that need to be
/// shifted to the right.
///
/// NonConstantIdx is an index value if there was a GEP with a non-constant
/// index value.  If this is 0 then all GEPs used to find this insert address
/// are constant.
Value *ConvertToScalarInfo::
ConvertScalar_InsertValue(Value *SV, Value *Old,
                          uint64_t Offset, Value* NonConstantIdx,
                          IRBuilder<> &Builder) {
  // Convert the stored type to the actual type, shift it left to insert
  // then 'or' into place.
  Type *AllocaType = Old->getType();
  LLVMContext &Context = Old->getContext();

  if (VectorType *VTy = dyn_cast<VectorType>(AllocaType)) {
    uint64_t VecSize = DL.getTypeAllocSizeInBits(VTy);
    uint64_t ValSize = DL.getTypeAllocSizeInBits(SV->getType());

    // Changing the whole vector with memset or with an access of a different
    // vector type?
    if (ValSize == VecSize)
        return Builder.CreateBitCast(SV, AllocaType);

    // Must be an element insertion.
    Type *EltTy = VTy->getElementType();
    if (SV->getType() != EltTy)
      SV = Builder.CreateBitCast(SV, EltTy);
    uint64_t EltSize = DL.getTypeAllocSizeInBits(EltTy);
    unsigned Elt = Offset/EltSize;
    Value *Idx;
    if (NonConstantIdx) {
      if (Elt)
        Idx = Builder.CreateAdd(NonConstantIdx,
                                Builder.getInt32(Elt),
                                "dyn.offset");
      else
        Idx = NonConstantIdx;
    } else
      Idx = Builder.getInt32(Elt);
    return Builder.CreateInsertElement(Old, SV, Idx);
  }

  // If SV is a first-class aggregate value, insert each value recursively.
  if (StructType *ST = dyn_cast<StructType>(SV->getType())) {
    assert(!NonConstantIdx &&
           "Dynamic indexing into struct types not supported");
    const StructLayout &Layout = *DL.getStructLayout(ST);
    for (unsigned i = 0, e = ST->getNumElements(); i != e; ++i) {
      Value *Elt = Builder.CreateExtractValue(SV, i);
      Old = ConvertScalar_InsertValue(Elt, Old,
                                      Offset+Layout.getElementOffsetInBits(i),
                                      nullptr, Builder);
    }
    return Old;
  }

  if (ArrayType *AT = dyn_cast<ArrayType>(SV->getType())) {
    assert(!NonConstantIdx &&
           "Dynamic indexing into array types not supported");
    uint64_t EltSize = DL.getTypeAllocSizeInBits(AT->getElementType());
    for (unsigned i = 0, e = AT->getNumElements(); i != e; ++i) {
      Value *Elt = Builder.CreateExtractValue(SV, i);
      Old = ConvertScalar_InsertValue(Elt, Old, Offset+i*EltSize, nullptr,
                                      Builder);
    }
    return Old;
  }

  // If SV is a float, convert it to the appropriate integer type.
  // If it is a pointer, do the same.
  unsigned SrcWidth = DL.getTypeSizeInBits(SV->getType());
  unsigned DestWidth = DL.getTypeSizeInBits(AllocaType);
  unsigned SrcStoreWidth = DL.getTypeStoreSizeInBits(SV->getType());
  unsigned DestStoreWidth = DL.getTypeStoreSizeInBits(AllocaType);
  if (SV->getType()->isFloatingPointTy() || SV->getType()->isVectorTy())
    SV = Builder.CreateBitCast(SV, IntegerType::get(SV->getContext(),SrcWidth));
  else if (SV->getType()->isPointerTy())
    SV = Builder.CreatePtrToInt(SV, DL.getIntPtrType(SV->getType()));

  // Zero extend or truncate the value if needed.
  if (SV->getType() != AllocaType) {
    if (SV->getType()->getPrimitiveSizeInBits() <
             AllocaType->getPrimitiveSizeInBits())
      SV = Builder.CreateZExt(SV, AllocaType);
    else {
      // Truncation may be needed if storing more than the alloca can hold
      // (undefined behavior).
      SV = Builder.CreateTrunc(SV, AllocaType);
      SrcWidth = DestWidth;
      SrcStoreWidth = DestStoreWidth;
    }
  }

  // If this is a big-endian system and the store is narrower than the
  // full alloca type, we need to do a shift to get the right bits.
  int ShAmt = 0;
  if (DL.isBigEndian()) {
    // On big-endian machines, the lowest bit is stored at the bit offset
    // from the pointer given by getTypeStoreSizeInBits.  This matters for
    // integers with a bitwidth that is not a multiple of 8.
    ShAmt = DestStoreWidth - SrcStoreWidth - Offset;
  } else {
    ShAmt = Offset;
  }

  // Note: we support negative bitwidths (with shr) which are not defined.
  // We do this to support (f.e.) stores off the end of a structure where
  // only some bits in the structure are set.
  APInt Mask(APInt::getLowBitsSet(DestWidth, SrcWidth));
  if (ShAmt > 0 && (unsigned)ShAmt < DestWidth) {
    SV = Builder.CreateShl(SV, ConstantInt::get(SV->getType(), ShAmt));
    Mask <<= ShAmt;
  } else if (ShAmt < 0 && (unsigned)-ShAmt < DestWidth) {
    SV = Builder.CreateLShr(SV, ConstantInt::get(SV->getType(), -ShAmt));
    Mask = Mask.lshr(-ShAmt);
  }

  // Mask out the bits we are about to insert from the old value, and or
  // in the new bits.
  if (SrcWidth != DestWidth) {
    assert(DestWidth > SrcWidth);
    Old = Builder.CreateAnd(Old, ConstantInt::get(Context, ~Mask), "mask");
    SV = Builder.CreateOr(Old, SV, "ins");
  }
  return SV;
}


//===----------------------------------------------------------------------===//
// SRoA Driver
//===----------------------------------------------------------------------===//


bool SROA::runOnFunction(Function &F) {
  if (skipOptnoneFunction(F))
    return false;

  bool Changed = performPromotion(F);

  while (1) {
    bool LocalChange = performScalarRepl(F);
    if (!LocalChange) break;   // No need to repromote if no scalarrepl
    Changed = true;
    LocalChange = performPromotion(F);
    if (!LocalChange) break;   // No need to re-scalarrepl if no promotion
  }

  return Changed;
}

namespace {
class AllocaPromoter : public LoadAndStorePromoter {
  AllocaInst *AI;
  DIBuilder *DIB;
  SmallVector<DbgDeclareInst *, 4> DDIs;
  SmallVector<DbgValueInst *, 4> DVIs;
public:
  AllocaPromoter(ArrayRef<Instruction*> Insts, SSAUpdater &S,
                 DIBuilder *DB)
    : LoadAndStorePromoter(Insts, S), AI(nullptr), DIB(DB) {}

  void run(AllocaInst *AI, const SmallVectorImpl<Instruction*> &Insts) {
    // Remember which alloca we're promoting (for isInstInList).
    this->AI = AI;
    if (auto *L = LocalAsMetadata::getIfExists(AI)) {
      if (auto *DINode = MetadataAsValue::getIfExists(AI->getContext(), L)) {
        for (User *U : DINode->users())
          if (DbgDeclareInst *DDI = dyn_cast<DbgDeclareInst>(U))
            DDIs.push_back(DDI);
          else if (DbgValueInst *DVI = dyn_cast<DbgValueInst>(U))
            DVIs.push_back(DVI);
      }
    }

    LoadAndStorePromoter::run(Insts);
    AI->eraseFromParent();
    for (SmallVectorImpl<DbgDeclareInst *>::iterator I = DDIs.begin(),
           E = DDIs.end(); I != E; ++I) {
      DbgDeclareInst *DDI = *I;
      DDI->eraseFromParent();
    }
    for (SmallVectorImpl<DbgValueInst *>::iterator I = DVIs.begin(),
           E = DVIs.end(); I != E; ++I) {
      DbgValueInst *DVI = *I;
      DVI->eraseFromParent();
    }
  }

  bool isInstInList(Instruction *I,
                    const SmallVectorImpl<Instruction*> &Insts) const override {
    if (LoadInst *LI = dyn_cast<LoadInst>(I))
      return LI->getOperand(0) == AI;
    return cast<StoreInst>(I)->getPointerOperand() == AI;
  }

  void updateDebugInfo(Instruction *Inst) const override {
    for (SmallVectorImpl<DbgDeclareInst *>::const_iterator I = DDIs.begin(),
           E = DDIs.end(); I != E; ++I) {
      DbgDeclareInst *DDI = *I;
      if (StoreInst *SI = dyn_cast<StoreInst>(Inst))
        ConvertDebugDeclareToDebugValue(DDI, SI, *DIB);
      else if (LoadInst *LI = dyn_cast<LoadInst>(Inst))
        ConvertDebugDeclareToDebugValue(DDI, LI, *DIB);
    }
    for (SmallVectorImpl<DbgValueInst *>::const_iterator I = DVIs.begin(),
           E = DVIs.end(); I != E; ++I) {
      DbgValueInst *DVI = *I;
      Value *Arg = nullptr;
      if (StoreInst *SI = dyn_cast<StoreInst>(Inst)) {
        // If an argument is zero extended then use argument directly. The ZExt
        // may be zapped by an optimization pass in future.
        if (ZExtInst *ZExt = dyn_cast<ZExtInst>(SI->getOperand(0)))
          Arg = dyn_cast<Argument>(ZExt->getOperand(0));
        if (SExtInst *SExt = dyn_cast<SExtInst>(SI->getOperand(0)))
          Arg = dyn_cast<Argument>(SExt->getOperand(0));
        if (!Arg)
          Arg = SI->getOperand(0);
      } else if (LoadInst *LI = dyn_cast<LoadInst>(Inst)) {
        Arg = LI->getOperand(0);
      } else {
        continue;
      }
      DIB->insertDbgValueIntrinsic(Arg, 0, DVI->getVariable(),
                                   DVI->getExpression(), DVI->getDebugLoc(),
                                   Inst);
    }
  }
};
} // end anon namespace

/// isSafeSelectToSpeculate - Select instructions that use an alloca and are
/// subsequently loaded can be rewritten to load both input pointers and then
/// select between the result, allowing the load of the alloca to be promoted.
/// From this:
///   %P2 = select i1 %cond, i32* %Alloca, i32* %Other
///   %V = load i32* %P2
/// to:
///   %V1 = load i32* %Alloca      -> will be mem2reg'd
///   %V2 = load i32* %Other
///   %V = select i1 %cond, i32 %V1, i32 %V2
///
/// We can do this to a select if its only uses are loads and if the operand to
/// the select can be loaded unconditionally.
static bool isSafeSelectToSpeculate(SelectInst *SI) {
  const DataLayout &DL = SI->getModule()->getDataLayout();
  bool TDerefable = isDereferenceablePointer(SI->getTrueValue(), DL);
  bool FDerefable = isDereferenceablePointer(SI->getFalseValue(), DL);

  for (User *U : SI->users()) {
    LoadInst *LI = dyn_cast<LoadInst>(U);
    if (!LI || !LI->isSimple()) return false;

    // Both operands to the select need to be dereferencable, either absolutely
    // (e.g. allocas) or at this point because we can see other accesses to it.
    if (!TDerefable &&
        !isSafeToLoadUnconditionally(SI->getTrueValue(), LI,
                                     LI->getAlignment()))
      return false;
    if (!FDerefable &&
        !isSafeToLoadUnconditionally(SI->getFalseValue(), LI,
                                     LI->getAlignment()))
      return false;
  }

  return true;
}

/// isSafePHIToSpeculate - PHI instructions that use an alloca and are
/// subsequently loaded can be rewritten to load both input pointers in the pred
/// blocks and then PHI the results, allowing the load of the alloca to be
/// promoted.
/// From this:
///   %P2 = phi [i32* %Alloca, i32* %Other]
///   %V = load i32* %P2
/// to:
///   %V1 = load i32* %Alloca      -> will be mem2reg'd
///   ...
///   %V2 = load i32* %Other
///   ...
///   %V = phi [i32 %V1, i32 %V2]
///
/// We can do this to a select if its only uses are loads and if the operand to
/// the select can be loaded unconditionally.
static bool isSafePHIToSpeculate(PHINode *PN) {
  // For now, we can only do this promotion if the load is in the same block as
  // the PHI, and if there are no stores between the phi and load.
  // TODO: Allow recursive phi users.
  // TODO: Allow stores.
  BasicBlock *BB = PN->getParent();
  unsigned MaxAlign = 0;
  for (User *U : PN->users()) {
    LoadInst *LI = dyn_cast<LoadInst>(U);
    if (!LI || !LI->isSimple()) return false;

    // For now we only allow loads in the same block as the PHI.  This is a
    // common case that happens when instcombine merges two loads through a PHI.
    if (LI->getParent() != BB) return false;

    // Ensure that there are no instructions between the PHI and the load that
    // could store.
    for (BasicBlock::iterator BBI = PN; &*BBI != LI; ++BBI)
      if (BBI->mayWriteToMemory())
        return false;

    MaxAlign = std::max(MaxAlign, LI->getAlignment());
  }

  const DataLayout &DL = PN->getModule()->getDataLayout();

  // Okay, we know that we have one or more loads in the same block as the PHI.
  // We can transform this if it is safe to push the loads into the predecessor
  // blocks.  The only thing to watch out for is that we can't put a possibly
  // trapping load in the predecessor if it is a critical edge.
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
    BasicBlock *Pred = PN->getIncomingBlock(i);
    Value *InVal = PN->getIncomingValue(i);

    // If the terminator of the predecessor has side-effects (an invoke),
    // there is no safe place to put a load in the predecessor.
    if (Pred->getTerminator()->mayHaveSideEffects())
      return false;

    // If the value is produced by the terminator of the predecessor
    // (an invoke), there is no valid place to put a load in the predecessor.
    if (Pred->getTerminator() == InVal)
      return false;

    // If the predecessor has a single successor, then the edge isn't critical.
    if (Pred->getTerminator()->getNumSuccessors() == 1)
      continue;

    // If this pointer is always safe to load, or if we can prove that there is
    // already a load in the block, then we can move the load to the pred block.
    if (isDereferenceablePointer(InVal, DL) ||
        isSafeToLoadUnconditionally(InVal, Pred->getTerminator(), MaxAlign))
      continue;

    return false;
  }

  return true;
}


/// tryToMakeAllocaBePromotable - This returns true if the alloca only has
/// direct (non-volatile) loads and stores to it.  If the alloca is close but
/// not quite there, this will transform the code to allow promotion.  As such,
/// it is a non-pure predicate.
static bool tryToMakeAllocaBePromotable(AllocaInst *AI, const DataLayout &DL) {
  SetVector<Instruction*, SmallVector<Instruction*, 4>,
            SmallPtrSet<Instruction*, 4> > InstsToRewrite;
  for (User *U : AI->users()) {
    if (LoadInst *LI = dyn_cast<LoadInst>(U)) {
      if (!LI->isSimple())
        return false;
      continue;
    }

    if (StoreInst *SI = dyn_cast<StoreInst>(U)) {
      if (SI->getOperand(0) == AI || !SI->isSimple())
        return false;   // Don't allow a store OF the AI, only INTO the AI.
      continue;
    }

    if (SelectInst *SI = dyn_cast<SelectInst>(U)) {
      // If the condition being selected on is a constant, fold the select, yes
      // this does (rarely) happen early on.
      if (ConstantInt *CI = dyn_cast<ConstantInt>(SI->getCondition())) {
        Value *Result = SI->getOperand(1+CI->isZero());
        SI->replaceAllUsesWith(Result);
        SI->eraseFromParent();

        // This is very rare and we just scrambled the use list of AI, start
        // over completely.
        return tryToMakeAllocaBePromotable(AI, DL);
      }

      // If it is safe to turn "load (select c, AI, ptr)" into a select of two
      // loads, then we can transform this by rewriting the select.
      if (!isSafeSelectToSpeculate(SI))
        return false;

      InstsToRewrite.insert(SI);
      continue;
    }

    if (PHINode *PN = dyn_cast<PHINode>(U)) {
      if (PN->use_empty()) {  // Dead PHIs can be stripped.
        InstsToRewrite.insert(PN);
        continue;
      }

      // If it is safe to turn "load (phi [AI, ptr, ...])" into a PHI of loads
      // in the pred blocks, then we can transform this by rewriting the PHI.
      if (!isSafePHIToSpeculate(PN))
        return false;

      InstsToRewrite.insert(PN);
      continue;
    }

    if (BitCastInst *BCI = dyn_cast<BitCastInst>(U)) {
      if (onlyUsedByLifetimeMarkers(BCI)) {
        InstsToRewrite.insert(BCI);
        continue;
      }
    }

    return false;
  }

  // If there are no instructions to rewrite, then all uses are load/stores and
  // we're done!
  if (InstsToRewrite.empty())
    return true;

  // If we have instructions that need to be rewritten for this to be promotable
  // take care of it now.
  for (unsigned i = 0, e = InstsToRewrite.size(); i != e; ++i) {
    if (BitCastInst *BCI = dyn_cast<BitCastInst>(InstsToRewrite[i])) {
      // This could only be a bitcast used by nothing but lifetime intrinsics.
      for (BitCastInst::user_iterator I = BCI->user_begin(), E = BCI->user_end();
           I != E;)
        cast<Instruction>(*I++)->eraseFromParent();
      BCI->eraseFromParent();
      continue;
    }

    if (SelectInst *SI = dyn_cast<SelectInst>(InstsToRewrite[i])) {
      // Selects in InstsToRewrite only have load uses.  Rewrite each as two
      // loads with a new select.
      while (!SI->use_empty()) {
        LoadInst *LI = cast<LoadInst>(SI->user_back());

        IRBuilder<> Builder(LI);
        LoadInst *TrueLoad =
          Builder.CreateLoad(SI->getTrueValue(), LI->getName()+".t");
        LoadInst *FalseLoad =
          Builder.CreateLoad(SI->getFalseValue(), LI->getName()+".f");

        // Transfer alignment and AA info if present.
        TrueLoad->setAlignment(LI->getAlignment());
        FalseLoad->setAlignment(LI->getAlignment());

        AAMDNodes Tags;
        LI->getAAMetadata(Tags);
        if (Tags) {
          TrueLoad->setAAMetadata(Tags);
          FalseLoad->setAAMetadata(Tags);
        }

        Value *V = Builder.CreateSelect(SI->getCondition(), TrueLoad, FalseLoad);
        V->takeName(LI);
        LI->replaceAllUsesWith(V);
        LI->eraseFromParent();
      }

      // Now that all the loads are gone, the select is gone too.
      SI->eraseFromParent();
      continue;
    }

    // Otherwise, we have a PHI node which allows us to push the loads into the
    // predecessors.
    PHINode *PN = cast<PHINode>(InstsToRewrite[i]);
    if (PN->use_empty()) {
      PN->eraseFromParent();
      continue;
    }

    Type *LoadTy = cast<PointerType>(PN->getType())->getElementType();
    PHINode *NewPN = PHINode::Create(LoadTy, PN->getNumIncomingValues(),
                                     PN->getName()+".ld", PN);

    // Get the AA tags and alignment to use from one of the loads.  It doesn't
    // matter which one we get and if any differ, it doesn't matter.
    LoadInst *SomeLoad = cast<LoadInst>(PN->user_back());

    AAMDNodes AATags;
    SomeLoad->getAAMetadata(AATags);
    unsigned Align = SomeLoad->getAlignment();

    // Rewrite all loads of the PN to use the new PHI.
    while (!PN->use_empty()) {
      LoadInst *LI = cast<LoadInst>(PN->user_back());
      LI->replaceAllUsesWith(NewPN);
      LI->eraseFromParent();
    }

    // Inject loads into all of the pred blocks.  Keep track of which blocks we
    // insert them into in case we have multiple edges from the same block.
    DenseMap<BasicBlock*, LoadInst*> InsertedLoads;

    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
      BasicBlock *Pred = PN->getIncomingBlock(i);
      LoadInst *&Load = InsertedLoads[Pred];
      if (!Load) {
        Load = new LoadInst(PN->getIncomingValue(i),
                            PN->getName() + "." + Pred->getName(),
                            Pred->getTerminator());
        Load->setAlignment(Align);
        if (AATags) Load->setAAMetadata(AATags);
      }

      NewPN->addIncoming(Load, Pred);
    }

    PN->eraseFromParent();
  }

  ++NumAdjusted;
  return true;
}

bool SROA::performPromotion(Function &F) {
  std::vector<AllocaInst*> Allocas;
  const DataLayout &DL = F.getParent()->getDataLayout();
  DominatorTree *DT = nullptr;
  if (HasDomTree)
    DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  AssumptionCache &AC =
      getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);

  BasicBlock &BB = F.getEntryBlock();  // Get the entry node for the function
  DIBuilder DIB(*F.getParent(), /*AllowUnresolved*/ false);
  bool Changed = false;
  SmallVector<Instruction*, 64> Insts;
  while (1) {
    Allocas.clear();

    // Find allocas that are safe to promote, by looking at all instructions in
    // the entry node
    for (BasicBlock::iterator I = BB.begin(), E = --BB.end(); I != E; ++I)
      if (AllocaInst *AI = dyn_cast<AllocaInst>(I))       // Is it an alloca?
        if (tryToMakeAllocaBePromotable(AI, DL))
          Allocas.push_back(AI);

    if (Allocas.empty()) break;

    if (HasDomTree)
      PromoteMemToReg(Allocas, *DT, nullptr, &AC);
    else {
      SSAUpdater SSA;
      for (unsigned i = 0, e = Allocas.size(); i != e; ++i) {
        AllocaInst *AI = Allocas[i];

        // Build list of instructions to promote.
        for (User *U : AI->users())
          Insts.push_back(cast<Instruction>(U));
        AllocaPromoter(Insts, SSA, &DIB).run(AI, Insts);
        Insts.clear();
      }
    }
    NumPromoted += Allocas.size();
    Changed = true;
  }

  return Changed;
}


/// ShouldAttemptScalarRepl - Decide if an alloca is a good candidate for
/// SROA.  It must be a struct or array type with a small number of elements.
bool SROA::ShouldAttemptScalarRepl(AllocaInst *AI) {
  Type *T = AI->getAllocatedType();
  // Do not promote any struct that has too many members.
  if (StructType *ST = dyn_cast<StructType>(T))
    return ST->getNumElements() <= StructMemberThreshold;
  // Do not promote any array that has too many elements.
  if (ArrayType *AT = dyn_cast<ArrayType>(T))
    return AT->getNumElements() <= ArrayElementThreshold;
  return false;
}

// performScalarRepl - This algorithm is a simple worklist driven algorithm,
// which runs on all of the alloca instructions in the entry block, removing
// them if they are only used by getelementptr instructions.
//
bool SROA::performScalarRepl(Function &F) {
  std::vector<AllocaInst*> WorkList;
  const DataLayout &DL = F.getParent()->getDataLayout();

  // Scan the entry basic block, adding allocas to the worklist.
  BasicBlock &BB = F.getEntryBlock();
  for (BasicBlock::iterator I = BB.begin(), E = BB.end(); I != E; ++I)
    if (AllocaInst *A = dyn_cast<AllocaInst>(I))
      WorkList.push_back(A);

  // Process the worklist
  bool Changed = false;
  while (!WorkList.empty()) {
    AllocaInst *AI = WorkList.back();
    WorkList.pop_back();

    // Handle dead allocas trivially.  These can be formed by SROA'ing arrays
    // with unused elements.
    if (AI->use_empty()) {
      AI->eraseFromParent();
      Changed = true;
      continue;
    }

    // If this alloca is impossible for us to promote, reject it early.
    if (AI->isArrayAllocation() || !AI->getAllocatedType()->isSized())
      continue;

    // Check to see if we can perform the core SROA transformation.  We cannot
    // transform the allocation instruction if it is an array allocation
    // (allocations OF arrays are ok though), and an allocation of a scalar
    // value cannot be decomposed at all.
    uint64_t AllocaSize = DL.getTypeAllocSize(AI->getAllocatedType());

    // Do not promote [0 x %struct].
    if (AllocaSize == 0) continue;

    // Do not promote any struct whose size is too big.
    if (AllocaSize > SRThreshold) continue;

    // If the alloca looks like a good candidate for scalar replacement, and if
    // all its users can be transformed, then split up the aggregate into its
    // separate elements.
    if (ShouldAttemptScalarRepl(AI) && isSafeAllocaToScalarRepl(AI)) {
      DoScalarReplacement(AI, WorkList);
      Changed = true;
      continue;
    }

    // If we can turn this aggregate value (potentially with casts) into a
    // simple scalar value that can be mem2reg'd into a register value.
    // IsNotTrivial tracks whether this is something that mem2reg could have
    // promoted itself.  If so, we don't want to transform it needlessly.  Note
    // that we can't just check based on the type: the alloca may be of an i32
    // but that has pointer arithmetic to set byte 3 of it or something.
    if (AllocaInst *NewAI =
            ConvertToScalarInfo((unsigned)AllocaSize, DL, ScalarLoadThreshold)
                .TryConvert(AI)) {
      NewAI->takeName(AI);
      AI->eraseFromParent();
      ++NumConverted;
      Changed = true;
      continue;
    }

    // Otherwise, couldn't process this alloca.
  }

  return Changed;
}

/// DoScalarReplacement - This alloca satisfied the isSafeAllocaToScalarRepl
/// predicate, do SROA now.
void SROA::DoScalarReplacement(AllocaInst *AI,
                               std::vector<AllocaInst*> &WorkList) {
  DEBUG(dbgs() << "Found inst to SROA: " << *AI << '\n');
  SmallVector<AllocaInst*, 32> ElementAllocas;
  if (StructType *ST = dyn_cast<StructType>(AI->getAllocatedType())) {
    ElementAllocas.reserve(ST->getNumContainedTypes());
    for (unsigned i = 0, e = ST->getNumContainedTypes(); i != e; ++i) {
      AllocaInst *NA = new AllocaInst(ST->getContainedType(i), nullptr,
                                      AI->getAlignment(),
                                      AI->getName() + "." + Twine(i), AI);
      ElementAllocas.push_back(NA);
      WorkList.push_back(NA);  // Add to worklist for recursive processing
    }
  } else {
    ArrayType *AT = cast<ArrayType>(AI->getAllocatedType());
    ElementAllocas.reserve(AT->getNumElements());
    Type *ElTy = AT->getElementType();
    for (unsigned i = 0, e = AT->getNumElements(); i != e; ++i) {
      AllocaInst *NA = new AllocaInst(ElTy, nullptr, AI->getAlignment(),
                                      AI->getName() + "." + Twine(i), AI);
      ElementAllocas.push_back(NA);
      WorkList.push_back(NA);  // Add to worklist for recursive processing
    }
  }

  // Now that we have created the new alloca instructions, rewrite all the
  // uses of the old alloca.
  RewriteForScalarRepl(AI, AI, 0, ElementAllocas);

  // Now erase any instructions that were made dead while rewriting the alloca.
  DeleteDeadInstructions();
  AI->eraseFromParent();

  ++NumReplaced;
}

/// DeleteDeadInstructions - Erase instructions on the DeadInstrs list,
/// recursively including all their operands that become trivially dead.
void SROA::DeleteDeadInstructions() {
  while (!DeadInsts.empty()) {
    Instruction *I = cast<Instruction>(DeadInsts.pop_back_val());

    for (User::op_iterator OI = I->op_begin(), E = I->op_end(); OI != E; ++OI)
      if (Instruction *U = dyn_cast<Instruction>(*OI)) {
        // Zero out the operand and see if it becomes trivially dead.
        // (But, don't add allocas to the dead instruction list -- they are
        // already on the worklist and will be deleted separately.)
        *OI = nullptr;
        if (isInstructionTriviallyDead(U) && !isa<AllocaInst>(U))
          DeadInsts.push_back(U);
      }

    I->eraseFromParent();
  }
}

/// isSafeForScalarRepl - Check if instruction I is a safe use with regard to
/// performing scalar replacement of alloca AI.  The results are flagged in
/// the Info parameter.  Offset indicates the position within AI that is
/// referenced by this instruction.
void SROA::isSafeForScalarRepl(Instruction *I, uint64_t Offset,
                               AllocaInfo &Info) {
  const DataLayout &DL = I->getModule()->getDataLayout();
  for (Use &U : I->uses()) {
    Instruction *User = cast<Instruction>(U.getUser());

    if (BitCastInst *BC = dyn_cast<BitCastInst>(User)) {
      isSafeForScalarRepl(BC, Offset, Info);
    } else if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(User)) {
      uint64_t GEPOffset = Offset;
      isSafeGEP(GEPI, GEPOffset, Info);
      if (!Info.isUnsafe)
        isSafeForScalarRepl(GEPI, GEPOffset, Info);
    } else if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(User)) {
      ConstantInt *Length = dyn_cast<ConstantInt>(MI->getLength());
      if (!Length || Length->isNegative())
        return MarkUnsafe(Info, User);

      isSafeMemAccess(Offset, Length->getZExtValue(), nullptr,
                      U.getOperandNo() == 0, Info, MI,
                      true /*AllowWholeAccess*/);
    } else if (LoadInst *LI = dyn_cast<LoadInst>(User)) {
      if (!LI->isSimple())
        return MarkUnsafe(Info, User);
      Type *LIType = LI->getType();
      isSafeMemAccess(Offset, DL.getTypeAllocSize(LIType), LIType, false, Info,
                      LI, true /*AllowWholeAccess*/);
      Info.hasALoadOrStore = true;

    } else if (StoreInst *SI = dyn_cast<StoreInst>(User)) {
      // Store is ok if storing INTO the pointer, not storing the pointer
      if (!SI->isSimple() || SI->getOperand(0) == I)
        return MarkUnsafe(Info, User);

      Type *SIType = SI->getOperand(0)->getType();
      isSafeMemAccess(Offset, DL.getTypeAllocSize(SIType), SIType, true, Info,
                      SI, true /*AllowWholeAccess*/);
      Info.hasALoadOrStore = true;
    } else if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(User)) {
      if (II->getIntrinsicID() != Intrinsic::lifetime_start &&
          II->getIntrinsicID() != Intrinsic::lifetime_end)
        return MarkUnsafe(Info, User);
    } else if (isa<PHINode>(User) || isa<SelectInst>(User)) {
      isSafePHISelectUseForScalarRepl(User, Offset, Info);
    } else {
      return MarkUnsafe(Info, User);
    }
    if (Info.isUnsafe) return;
  }
}


/// isSafePHIUseForScalarRepl - If we see a PHI node or select using a pointer
/// derived from the alloca, we can often still split the alloca into elements.
/// This is useful if we have a large alloca where one element is phi'd
/// together somewhere: we can SRoA and promote all the other elements even if
/// we end up not being able to promote this one.
///
/// All we require is that the uses of the PHI do not index into other parts of
/// the alloca.  The most important use case for this is single load and stores
/// that are PHI'd together, which can happen due to code sinking.
void SROA::isSafePHISelectUseForScalarRepl(Instruction *I, uint64_t Offset,
                                           AllocaInfo &Info) {
  // If we've already checked this PHI, don't do it again.
  if (PHINode *PN = dyn_cast<PHINode>(I))
    if (!Info.CheckedPHIs.insert(PN).second)
      return;

  const DataLayout &DL = I->getModule()->getDataLayout();
  for (User *U : I->users()) {
    Instruction *UI = cast<Instruction>(U);

    if (BitCastInst *BC = dyn_cast<BitCastInst>(UI)) {
      isSafePHISelectUseForScalarRepl(BC, Offset, Info);
    } else if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(UI)) {
      // Only allow "bitcast" GEPs for simplicity.  We could generalize this,
      // but would have to prove that we're staying inside of an element being
      // promoted.
      if (!GEPI->hasAllZeroIndices())
        return MarkUnsafe(Info, UI);
      isSafePHISelectUseForScalarRepl(GEPI, Offset, Info);
    } else if (LoadInst *LI = dyn_cast<LoadInst>(UI)) {
      if (!LI->isSimple())
        return MarkUnsafe(Info, UI);
      Type *LIType = LI->getType();
      isSafeMemAccess(Offset, DL.getTypeAllocSize(LIType), LIType, false, Info,
                      LI, false /*AllowWholeAccess*/);
      Info.hasALoadOrStore = true;

    } else if (StoreInst *SI = dyn_cast<StoreInst>(UI)) {
      // Store is ok if storing INTO the pointer, not storing the pointer
      if (!SI->isSimple() || SI->getOperand(0) == I)
        return MarkUnsafe(Info, UI);

      Type *SIType = SI->getOperand(0)->getType();
      isSafeMemAccess(Offset, DL.getTypeAllocSize(SIType), SIType, true, Info,
                      SI, false /*AllowWholeAccess*/);
      Info.hasALoadOrStore = true;
    } else if (isa<PHINode>(UI) || isa<SelectInst>(UI)) {
      isSafePHISelectUseForScalarRepl(UI, Offset, Info);
    } else {
      return MarkUnsafe(Info, UI);
    }
    if (Info.isUnsafe) return;
  }
}

/// isSafeGEP - Check if a GEP instruction can be handled for scalar
/// replacement.  It is safe when all the indices are constant, in-bounds
/// references, and when the resulting offset corresponds to an element within
/// the alloca type.  The results are flagged in the Info parameter.  Upon
/// return, Offset is adjusted as specified by the GEP indices.
void SROA::isSafeGEP(GetElementPtrInst *GEPI,
                     uint64_t &Offset, AllocaInfo &Info) {
  gep_type_iterator GEPIt = gep_type_begin(GEPI), E = gep_type_end(GEPI);
  if (GEPIt == E)
    return;
  bool NonConstant = false;
  unsigned NonConstantIdxSize = 0;

  // Walk through the GEP type indices, checking the types that this indexes
  // into.
  for (; GEPIt != E; ++GEPIt) {
    // Ignore struct elements, no extra checking needed for these.
    if ((*GEPIt)->isStructTy())
      continue;

    ConstantInt *IdxVal = dyn_cast<ConstantInt>(GEPIt.getOperand());
    if (!IdxVal)
      return MarkUnsafe(Info, GEPI);
  }

  // Compute the offset due to this GEP and check if the alloca has a
  // component element at that offset.
  SmallVector<Value*, 8> Indices(GEPI->op_begin() + 1, GEPI->op_end());
  // If this GEP is non-constant then the last operand must have been a
  // dynamic index into a vector.  Pop this now as it has no impact on the
  // constant part of the offset.
  if (NonConstant)
    Indices.pop_back();

  const DataLayout &DL = GEPI->getModule()->getDataLayout();
  Offset += DL.getIndexedOffset(GEPI->getPointerOperandType(), Indices);
  if (!TypeHasComponent(Info.AI->getAllocatedType(), Offset, NonConstantIdxSize,
                        DL))
    MarkUnsafe(Info, GEPI);
}

/// isHomogeneousAggregate - Check if type T is a struct or array containing
/// elements of the same type (which is always true for arrays).  If so,
/// return true with NumElts and EltTy set to the number of elements and the
/// element type, respectively.
static bool isHomogeneousAggregate(Type *T, unsigned &NumElts,
                                   Type *&EltTy) {
  if (ArrayType *AT = dyn_cast<ArrayType>(T)) {
    NumElts = AT->getNumElements();
    EltTy = (NumElts == 0 ? nullptr : AT->getElementType());
    return true;
  }
  if (StructType *ST = dyn_cast<StructType>(T)) {
    NumElts = ST->getNumContainedTypes();
    EltTy = (NumElts == 0 ? nullptr : ST->getContainedType(0));
    for (unsigned n = 1; n < NumElts; ++n) {
      if (ST->getContainedType(n) != EltTy)
        return false;
    }
    return true;
  }
  return false;
}

/// isCompatibleAggregate - Check if T1 and T2 are either the same type or are
/// "homogeneous" aggregates with the same element type and number of elements.
static bool isCompatibleAggregate(Type *T1, Type *T2) {
  if (T1 == T2)
    return true;

  unsigned NumElts1, NumElts2;
  Type *EltTy1, *EltTy2;
  if (isHomogeneousAggregate(T1, NumElts1, EltTy1) &&
      isHomogeneousAggregate(T2, NumElts2, EltTy2) &&
      NumElts1 == NumElts2 &&
      EltTy1 == EltTy2)
    return true;

  return false;
}

/// isSafeMemAccess - Check if a load/store/memcpy operates on the entire AI
/// alloca or has an offset and size that corresponds to a component element
/// within it.  The offset checked here may have been formed from a GEP with a
/// pointer bitcasted to a different type.
///
/// If AllowWholeAccess is true, then this allows uses of the entire alloca as a
/// unit.  If false, it only allows accesses known to be in a single element.
void SROA::isSafeMemAccess(uint64_t Offset, uint64_t MemSize,
                           Type *MemOpType, bool isStore,
                           AllocaInfo &Info, Instruction *TheAccess,
                           bool AllowWholeAccess) {
  const DataLayout &DL = TheAccess->getModule()->getDataLayout();
  // Check if this is a load/store of the entire alloca.
  if (Offset == 0 && AllowWholeAccess &&
      MemSize == DL.getTypeAllocSize(Info.AI->getAllocatedType())) {
    // This can be safe for MemIntrinsics (where MemOpType is 0) and integer
    // loads/stores (which are essentially the same as the MemIntrinsics with
    // regard to copying padding between elements).  But, if an alloca is
    // flagged as both a source and destination of such operations, we'll need
    // to check later for padding between elements.
    if (!MemOpType || MemOpType->isIntegerTy()) {
      if (isStore)
        Info.isMemCpyDst = true;
      else
        Info.isMemCpySrc = true;
      return;
    }
    // This is also safe for references using a type that is compatible with
    // the type of the alloca, so that loads/stores can be rewritten using
    // insertvalue/extractvalue.
    if (isCompatibleAggregate(MemOpType, Info.AI->getAllocatedType())) {
      Info.hasSubelementAccess = true;
      return;
    }
  }
  // Check if the offset/size correspond to a component within the alloca type.
  Type *T = Info.AI->getAllocatedType();
  if (TypeHasComponent(T, Offset, MemSize, DL)) {
    Info.hasSubelementAccess = true;
    return;
  }

  return MarkUnsafe(Info, TheAccess);
}

/// TypeHasComponent - Return true if T has a component type with the
/// specified offset and size.  If Size is zero, do not check the size.
bool SROA::TypeHasComponent(Type *T, uint64_t Offset, uint64_t Size,
                            const DataLayout &DL) {
  Type *EltTy;
  uint64_t EltSize;
  if (StructType *ST = dyn_cast<StructType>(T)) {
    const StructLayout *Layout = DL.getStructLayout(ST);
    unsigned EltIdx = Layout->getElementContainingOffset(Offset);
    EltTy = ST->getContainedType(EltIdx);
    EltSize = DL.getTypeAllocSize(EltTy);
    Offset -= Layout->getElementOffset(EltIdx);
  } else if (ArrayType *AT = dyn_cast<ArrayType>(T)) {
    EltTy = AT->getElementType();
    EltSize = DL.getTypeAllocSize(EltTy);
    if (Offset >= AT->getNumElements() * EltSize)
      return false;
    Offset %= EltSize;
  } else if (VectorType *VT = dyn_cast<VectorType>(T)) {
    EltTy = VT->getElementType();
    EltSize = DL.getTypeAllocSize(EltTy);
    if (Offset >= VT->getNumElements() * EltSize)
      return false;
    Offset %= EltSize;
  } else {
    return false;
  }
  if (Offset == 0 && (Size == 0 || EltSize == Size))
    return true;
  // Check if the component spans multiple elements.
  if (Offset + Size > EltSize)
    return false;
  return TypeHasComponent(EltTy, Offset, Size, DL);
}

/// RewriteForScalarRepl - Alloca AI is being split into NewElts, so rewrite
/// the instruction I, which references it, to use the separate elements.
/// Offset indicates the position within AI that is referenced by this
/// instruction.
void SROA::RewriteForScalarRepl(Instruction *I, AllocaInst *AI, uint64_t Offset,
                                SmallVectorImpl<AllocaInst *> &NewElts) {
  const DataLayout &DL = I->getModule()->getDataLayout();
  for (Value::use_iterator UI = I->use_begin(), E = I->use_end(); UI!=E;) {
    Use &TheUse = *UI++;
    Instruction *User = cast<Instruction>(TheUse.getUser());

    if (BitCastInst *BC = dyn_cast<BitCastInst>(User)) {
      RewriteBitCast(BC, AI, Offset, NewElts);
      continue;
    }

    if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(User)) {
      RewriteGEP(GEPI, AI, Offset, NewElts);
      continue;
    }

    if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(User)) {
      ConstantInt *Length = dyn_cast<ConstantInt>(MI->getLength());
      uint64_t MemSize = Length->getZExtValue();
      if (Offset == 0 && MemSize == DL.getTypeAllocSize(AI->getAllocatedType()))
        RewriteMemIntrinUserOfAlloca(MI, I, AI, NewElts);
      // Otherwise the intrinsic can only touch a single element and the
      // address operand will be updated, so nothing else needs to be done.
      continue;
    }

    if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(User)) {
      if (II->getIntrinsicID() == Intrinsic::lifetime_start ||
          II->getIntrinsicID() == Intrinsic::lifetime_end) {
        RewriteLifetimeIntrinsic(II, AI, Offset, NewElts);
      }
      continue;
    }

    if (LoadInst *LI = dyn_cast<LoadInst>(User)) {
      Type *LIType = LI->getType();

      if (isCompatibleAggregate(LIType, AI->getAllocatedType())) {
        // Replace:
        //   %res = load { i32, i32 }* %alloc
        // with:
        //   %load.0 = load i32* %alloc.0
        //   %insert.0 insertvalue { i32, i32 } zeroinitializer, i32 %load.0, 0
        //   %load.1 = load i32* %alloc.1
        //   %insert = insertvalue { i32, i32 } %insert.0, i32 %load.1, 1
        // (Also works for arrays instead of structs)
        Value *Insert = UndefValue::get(LIType);
        IRBuilder<> Builder(LI);
        for (unsigned i = 0, e = NewElts.size(); i != e; ++i) {
          Value *Load = Builder.CreateLoad(NewElts[i], "load");
          Insert = Builder.CreateInsertValue(Insert, Load, i, "insert");
        }
        LI->replaceAllUsesWith(Insert);
        DeadInsts.push_back(LI);
      } else if (LIType->isIntegerTy() &&
                 DL.getTypeAllocSize(LIType) ==
                     DL.getTypeAllocSize(AI->getAllocatedType())) {
        // If this is a load of the entire alloca to an integer, rewrite it.
        RewriteLoadUserOfWholeAlloca(LI, AI, NewElts);
      }
      continue;
    }

    if (StoreInst *SI = dyn_cast<StoreInst>(User)) {
      Value *Val = SI->getOperand(0);
      Type *SIType = Val->getType();
      if (isCompatibleAggregate(SIType, AI->getAllocatedType())) {
        // Replace:
        //   store { i32, i32 } %val, { i32, i32 }* %alloc
        // with:
        //   %val.0 = extractvalue { i32, i32 } %val, 0
        //   store i32 %val.0, i32* %alloc.0
        //   %val.1 = extractvalue { i32, i32 } %val, 1
        //   store i32 %val.1, i32* %alloc.1
        // (Also works for arrays instead of structs)
        IRBuilder<> Builder(SI);
        for (unsigned i = 0, e = NewElts.size(); i != e; ++i) {
          Value *Extract = Builder.CreateExtractValue(Val, i, Val->getName());
          Builder.CreateStore(Extract, NewElts[i]);
        }
        DeadInsts.push_back(SI);
      } else if (SIType->isIntegerTy() &&
                 DL.getTypeAllocSize(SIType) ==
                     DL.getTypeAllocSize(AI->getAllocatedType())) {
        // If this is a store of the entire alloca from an integer, rewrite it.
        RewriteStoreUserOfWholeAlloca(SI, AI, NewElts);
      }
      continue;
    }

    if (isa<SelectInst>(User) || isa<PHINode>(User)) {
      // If we have a PHI user of the alloca itself (as opposed to a GEP or
      // bitcast) we have to rewrite it.  GEP and bitcast uses will be RAUW'd to
      // the new pointer.
      if (!isa<AllocaInst>(I)) continue;

      assert(Offset == 0 && NewElts[0] &&
             "Direct alloca use should have a zero offset");

      // If we have a use of the alloca, we know the derived uses will be
      // utilizing just the first element of the scalarized result.  Insert a
      // bitcast of the first alloca before the user as required.
      AllocaInst *NewAI = NewElts[0];
      BitCastInst *BCI = new BitCastInst(NewAI, AI->getType(), "", NewAI);
      NewAI->moveBefore(BCI);
      TheUse = BCI;
      continue;
    }
  }
}

/// RewriteBitCast - Update a bitcast reference to the alloca being replaced
/// and recursively continue updating all of its uses.
void SROA::RewriteBitCast(BitCastInst *BC, AllocaInst *AI, uint64_t Offset,
                          SmallVectorImpl<AllocaInst *> &NewElts) {
  RewriteForScalarRepl(BC, AI, Offset, NewElts);
  if (BC->getOperand(0) != AI)
    return;

  // The bitcast references the original alloca.  Replace its uses with
  // references to the alloca containing offset zero (which is normally at
  // index zero, but might not be in cases involving structs with elements
  // of size zero).
  Type *T = AI->getAllocatedType();
  uint64_t EltOffset = 0;
  Type *IdxTy;
  uint64_t Idx = FindElementAndOffset(T, EltOffset, IdxTy,
                                      BC->getModule()->getDataLayout());
  Instruction *Val = NewElts[Idx];
  if (Val->getType() != BC->getDestTy()) {
    Val = new BitCastInst(Val, BC->getDestTy(), "", BC);
    Val->takeName(BC);
  }
  BC->replaceAllUsesWith(Val);
  DeadInsts.push_back(BC);
}

/// FindElementAndOffset - Return the index of the element containing Offset
/// within the specified type, which must be either a struct or an array.
/// Sets T to the type of the element and Offset to the offset within that
/// element.  IdxTy is set to the type of the index result to be used in a
/// GEP instruction.
uint64_t SROA::FindElementAndOffset(Type *&T, uint64_t &Offset, Type *&IdxTy,
                                    const DataLayout &DL) {
  uint64_t Idx = 0;

  if (StructType *ST = dyn_cast<StructType>(T)) {
    const StructLayout *Layout = DL.getStructLayout(ST);
    Idx = Layout->getElementContainingOffset(Offset);
    T = ST->getContainedType(Idx);
    Offset -= Layout->getElementOffset(Idx);
    IdxTy = Type::getInt32Ty(T->getContext());
    return Idx;
  } else if (ArrayType *AT = dyn_cast<ArrayType>(T)) {
    T = AT->getElementType();
    uint64_t EltSize = DL.getTypeAllocSize(T);
    Idx = Offset / EltSize;
    Offset -= Idx * EltSize;
    IdxTy = Type::getInt64Ty(T->getContext());
    return Idx;
  }
  VectorType *VT = cast<VectorType>(T);
  T = VT->getElementType();
  uint64_t EltSize = DL.getTypeAllocSize(T);
  Idx = Offset / EltSize;
  Offset -= Idx * EltSize;
  IdxTy = Type::getInt64Ty(T->getContext());
  return Idx;
}

/// RewriteGEP - Check if this GEP instruction moves the pointer across
/// elements of the alloca that are being split apart, and if so, rewrite
/// the GEP to be relative to the new element.
void SROA::RewriteGEP(GetElementPtrInst *GEPI, AllocaInst *AI, uint64_t Offset,
                      SmallVectorImpl<AllocaInst *> &NewElts) {
  uint64_t OldOffset = Offset;
  const DataLayout &DL = GEPI->getModule()->getDataLayout();
  SmallVector<Value*, 8> Indices(GEPI->op_begin() + 1, GEPI->op_end());
  // If the GEP was dynamic then it must have been a dynamic vector lookup.
  // In this case, it must be the last GEP operand which is dynamic so keep that
  // aside until we've found the constant GEP offset then add it back in at the
  // end.
  Value* NonConstantIdx = nullptr;
  if (!GEPI->hasAllConstantIndices())
    NonConstantIdx = Indices.pop_back_val();
  Offset += DL.getIndexedOffset(GEPI->getPointerOperandType(), Indices);

  RewriteForScalarRepl(GEPI, AI, Offset, NewElts);

  Type *T = AI->getAllocatedType();
  Type *IdxTy;
  uint64_t OldIdx = FindElementAndOffset(T, OldOffset, IdxTy, DL);
  if (GEPI->getOperand(0) == AI)
    OldIdx = ~0ULL; // Force the GEP to be rewritten.

  T = AI->getAllocatedType();
  uint64_t EltOffset = Offset;
  uint64_t Idx = FindElementAndOffset(T, EltOffset, IdxTy, DL);

  // If this GEP does not move the pointer across elements of the alloca
  // being split, then it does not needs to be rewritten.
  if (Idx == OldIdx)
    return;

  Type *i32Ty = Type::getInt32Ty(AI->getContext());
  SmallVector<Value*, 8> NewArgs;
  NewArgs.push_back(Constant::getNullValue(i32Ty));
  while (EltOffset != 0) {
    uint64_t EltIdx = FindElementAndOffset(T, EltOffset, IdxTy, DL);
    NewArgs.push_back(ConstantInt::get(IdxTy, EltIdx));
  }
  if (NonConstantIdx) {
    Type* GepTy = T;
    // This GEP has a dynamic index.  We need to add "i32 0" to index through
    // any structs or arrays in the original type until we get to the vector
    // to index.
    while (!isa<VectorType>(GepTy)) {
      NewArgs.push_back(Constant::getNullValue(i32Ty));
      GepTy = cast<CompositeType>(GepTy)->getTypeAtIndex(0U);
    }
    NewArgs.push_back(NonConstantIdx);
  }
  Instruction *Val = NewElts[Idx];
  if (NewArgs.size() > 1) {
    Val = GetElementPtrInst::CreateInBounds(Val, NewArgs, "", GEPI);
    Val->takeName(GEPI);
  }
  if (Val->getType() != GEPI->getType())
    Val = new BitCastInst(Val, GEPI->getType(), Val->getName(), GEPI);
  GEPI->replaceAllUsesWith(Val);
  DeadInsts.push_back(GEPI);
}

/// RewriteLifetimeIntrinsic - II is a lifetime.start/lifetime.end. Rewrite it
/// to mark the lifetime of the scalarized memory.
void SROA::RewriteLifetimeIntrinsic(IntrinsicInst *II, AllocaInst *AI,
                                    uint64_t Offset,
                                    SmallVectorImpl<AllocaInst *> &NewElts) {
  ConstantInt *OldSize = cast<ConstantInt>(II->getArgOperand(0));
  // Put matching lifetime markers on everything from Offset up to
  // Offset+OldSize.
  Type *AIType = AI->getAllocatedType();
  const DataLayout &DL = II->getModule()->getDataLayout();
  uint64_t NewOffset = Offset;
  Type *IdxTy;
  uint64_t Idx = FindElementAndOffset(AIType, NewOffset, IdxTy, DL);

  IRBuilder<> Builder(II);
  uint64_t Size = OldSize->getLimitedValue();

  if (NewOffset) {
    // Splice the first element and index 'NewOffset' bytes in.  SROA will
    // split the alloca again later.
    unsigned AS = AI->getType()->getAddressSpace();
    Value *V = Builder.CreateBitCast(NewElts[Idx], Builder.getInt8PtrTy(AS));
    V = Builder.CreateGEP(Builder.getInt8Ty(), V, Builder.getInt64(NewOffset));

    IdxTy = NewElts[Idx]->getAllocatedType();
    uint64_t EltSize = DL.getTypeAllocSize(IdxTy) - NewOffset;
    if (EltSize > Size) {
      EltSize = Size;
      Size = 0;
    } else {
      Size -= EltSize;
    }
    if (II->getIntrinsicID() == Intrinsic::lifetime_start)
      Builder.CreateLifetimeStart(V, Builder.getInt64(EltSize));
    else
      Builder.CreateLifetimeEnd(V, Builder.getInt64(EltSize));
    ++Idx;
  }

  for (; Idx != NewElts.size() && Size; ++Idx) {
    IdxTy = NewElts[Idx]->getAllocatedType();
    uint64_t EltSize = DL.getTypeAllocSize(IdxTy);
    if (EltSize > Size) {
      EltSize = Size;
      Size = 0;
    } else {
      Size -= EltSize;
    }
    if (II->getIntrinsicID() == Intrinsic::lifetime_start)
      Builder.CreateLifetimeStart(NewElts[Idx],
                                  Builder.getInt64(EltSize));
    else
      Builder.CreateLifetimeEnd(NewElts[Idx],
                                Builder.getInt64(EltSize));
  }
  DeadInsts.push_back(II);
}

/// RewriteMemIntrinUserOfAlloca - MI is a memcpy/memset/memmove from or to AI.
/// Rewrite it to copy or set the elements of the scalarized memory.
void
SROA::RewriteMemIntrinUserOfAlloca(MemIntrinsic *MI, Instruction *Inst,
                                   AllocaInst *AI,
                                   SmallVectorImpl<AllocaInst *> &NewElts) {
  // If this is a memcpy/memmove, construct the other pointer as the
  // appropriate type.  The "Other" pointer is the pointer that goes to memory
  // that doesn't have anything to do with the alloca that we are promoting. For
  // memset, this Value* stays null.
  Value *OtherPtr = nullptr;
  unsigned MemAlignment = MI->getAlignment();
  if (MemTransferInst *MTI = dyn_cast<MemTransferInst>(MI)) { // memmove/memcopy
    if (Inst == MTI->getRawDest())
      OtherPtr = MTI->getRawSource();
    else {
      assert(Inst == MTI->getRawSource());
      OtherPtr = MTI->getRawDest();
    }
  }

  // If there is an other pointer, we want to convert it to the same pointer
  // type as AI has, so we can GEP through it safely.
  if (OtherPtr) {
    unsigned AddrSpace =
      cast<PointerType>(OtherPtr->getType())->getAddressSpace();

    // Remove bitcasts and all-zero GEPs from OtherPtr.  This is an
    // optimization, but it's also required to detect the corner case where
    // both pointer operands are referencing the same memory, and where
    // OtherPtr may be a bitcast or GEP that currently being rewritten.  (This
    // function is only called for mem intrinsics that access the whole
    // aggregate, so non-zero GEPs are not an issue here.)
    OtherPtr = OtherPtr->stripPointerCasts();

    // Copying the alloca to itself is a no-op: just delete it.
    if (OtherPtr == AI || OtherPtr == NewElts[0]) {
      // This code will run twice for a no-op memcpy -- once for each operand.
      // Put only one reference to MI on the DeadInsts list.
      for (SmallVectorImpl<Value *>::const_iterator I = DeadInsts.begin(),
             E = DeadInsts.end(); I != E; ++I)
        if (*I == MI) return;
      DeadInsts.push_back(MI);
      return;
    }

    // If the pointer is not the right type, insert a bitcast to the right
    // type.
    Type *NewTy =
      PointerType::get(AI->getType()->getElementType(), AddrSpace);

    if (OtherPtr->getType() != NewTy)
      OtherPtr = new BitCastInst(OtherPtr, NewTy, OtherPtr->getName(), MI);
  }

  // Process each element of the aggregate.
  bool SROADest = MI->getRawDest() == Inst;

  Constant *Zero = Constant::getNullValue(Type::getInt32Ty(MI->getContext()));
  const DataLayout &DL = MI->getModule()->getDataLayout();

  for (unsigned i = 0, e = NewElts.size(); i != e; ++i) {
    // If this is a memcpy/memmove, emit a GEP of the other element address.
    Value *OtherElt = nullptr;
    unsigned OtherEltAlign = MemAlignment;

    if (OtherPtr) {
      Value *Idx[2] = { Zero,
                      ConstantInt::get(Type::getInt32Ty(MI->getContext()), i) };
      OtherElt = GetElementPtrInst::CreateInBounds(OtherPtr, Idx,
                                              OtherPtr->getName()+"."+Twine(i),
                                                   MI);
      uint64_t EltOffset;
      PointerType *OtherPtrTy = cast<PointerType>(OtherPtr->getType());
      Type *OtherTy = OtherPtrTy->getElementType();
      if (StructType *ST = dyn_cast<StructType>(OtherTy)) {
        EltOffset = DL.getStructLayout(ST)->getElementOffset(i);
      } else {
        Type *EltTy = cast<SequentialType>(OtherTy)->getElementType();
        EltOffset = DL.getTypeAllocSize(EltTy) * i;
      }

      // The alignment of the other pointer is the guaranteed alignment of the
      // element, which is affected by both the known alignment of the whole
      // mem intrinsic and the alignment of the element.  If the alignment of
      // the memcpy (f.e.) is 32 but the element is at a 4-byte offset, then the
      // known alignment is just 4 bytes.
      OtherEltAlign = (unsigned)MinAlign(OtherEltAlign, EltOffset);
    }

    Value *EltPtr = NewElts[i];
    Type *EltTy = cast<PointerType>(EltPtr->getType())->getElementType();

    // If we got down to a scalar, insert a load or store as appropriate.
    if (EltTy->isSingleValueType()) {
      if (isa<MemTransferInst>(MI)) {
        if (SROADest) {
          // From Other to Alloca.
          Value *Elt = new LoadInst(OtherElt, "tmp", false, OtherEltAlign, MI);
          new StoreInst(Elt, EltPtr, MI);
        } else {
          // From Alloca to Other.
          Value *Elt = new LoadInst(EltPtr, "tmp", MI);
          new StoreInst(Elt, OtherElt, false, OtherEltAlign, MI);
        }
        continue;
      }
      assert(isa<MemSetInst>(MI));

      // If the stored element is zero (common case), just store a null
      // constant.
      Constant *StoreVal;
      if (ConstantInt *CI = dyn_cast<ConstantInt>(MI->getArgOperand(1))) {
        if (CI->isZero()) {
          StoreVal = Constant::getNullValue(EltTy);  // 0.0, null, 0, <0,0>
        } else {
          // If EltTy is a vector type, get the element type.
          Type *ValTy = EltTy->getScalarType();

          // Construct an integer with the right value.
          unsigned EltSize = DL.getTypeSizeInBits(ValTy);
          APInt OneVal(EltSize, CI->getZExtValue());
          APInt TotalVal(OneVal);
          // Set each byte.
          for (unsigned i = 0; 8*i < EltSize; ++i) {
            TotalVal = TotalVal.shl(8);
            TotalVal |= OneVal;
          }

          // Convert the integer value to the appropriate type.
          StoreVal = ConstantInt::get(CI->getContext(), TotalVal);
          if (ValTy->isPointerTy())
            StoreVal = ConstantExpr::getIntToPtr(StoreVal, ValTy);
          else if (ValTy->isFloatingPointTy())
            StoreVal = ConstantExpr::getBitCast(StoreVal, ValTy);
          assert(StoreVal->getType() == ValTy && "Type mismatch!");

          // If the requested value was a vector constant, create it.
          if (EltTy->isVectorTy()) {
            unsigned NumElts = cast<VectorType>(EltTy)->getNumElements();
            StoreVal = ConstantVector::getSplat(NumElts, StoreVal);
          }
        }
        new StoreInst(StoreVal, EltPtr, MI);
        continue;
      }
      // Otherwise, if we're storing a byte variable, use a memset call for
      // this element.
    }

    unsigned EltSize = DL.getTypeAllocSize(EltTy);
    if (!EltSize)
      continue;

    IRBuilder<> Builder(MI);

    // Finally, insert the meminst for this element.
    if (isa<MemSetInst>(MI)) {
      Builder.CreateMemSet(EltPtr, MI->getArgOperand(1), EltSize,
                           MI->isVolatile());
    } else {
      assert(isa<MemTransferInst>(MI));
      Value *Dst = SROADest ? EltPtr : OtherElt;  // Dest ptr
      Value *Src = SROADest ? OtherElt : EltPtr;  // Src ptr

      if (isa<MemCpyInst>(MI))
        Builder.CreateMemCpy(Dst, Src, EltSize, OtherEltAlign,MI->isVolatile());
      else
        Builder.CreateMemMove(Dst, Src, EltSize,OtherEltAlign,MI->isVolatile());
    }
  }
  DeadInsts.push_back(MI);
}

/// RewriteStoreUserOfWholeAlloca - We found a store of an integer that
/// overwrites the entire allocation.  Extract out the pieces of the stored
/// integer and store them individually.
void
SROA::RewriteStoreUserOfWholeAlloca(StoreInst *SI, AllocaInst *AI,
                                    SmallVectorImpl<AllocaInst *> &NewElts) {
  // Extract each element out of the integer according to its structure offset
  // and store the element value to the individual alloca.
  Value *SrcVal = SI->getOperand(0);
  Type *AllocaEltTy = AI->getAllocatedType();
  const DataLayout &DL = SI->getModule()->getDataLayout();
  uint64_t AllocaSizeBits = DL.getTypeAllocSizeInBits(AllocaEltTy);

  IRBuilder<> Builder(SI);

  // Handle tail padding by extending the operand
  if (DL.getTypeSizeInBits(SrcVal->getType()) != AllocaSizeBits)
    SrcVal = Builder.CreateZExt(SrcVal,
                            IntegerType::get(SI->getContext(), AllocaSizeBits));

  DEBUG(dbgs() << "PROMOTING STORE TO WHOLE ALLOCA: " << *AI << '\n' << *SI
               << '\n');

  // There are two forms here: AI could be an array or struct.  Both cases
  // have different ways to compute the element offset.
  if (StructType *EltSTy = dyn_cast<StructType>(AllocaEltTy)) {
    const StructLayout *Layout = DL.getStructLayout(EltSTy);

    for (unsigned i = 0, e = NewElts.size(); i != e; ++i) {
      // Get the number of bits to shift SrcVal to get the value.
      Type *FieldTy = EltSTy->getElementType(i);
      uint64_t Shift = Layout->getElementOffsetInBits(i);

      if (DL.isBigEndian())
        Shift = AllocaSizeBits - Shift - DL.getTypeAllocSizeInBits(FieldTy);

      Value *EltVal = SrcVal;
      if (Shift) {
        Value *ShiftVal = ConstantInt::get(EltVal->getType(), Shift);
        EltVal = Builder.CreateLShr(EltVal, ShiftVal, "sroa.store.elt");
      }

      // Truncate down to an integer of the right size.
      uint64_t FieldSizeBits = DL.getTypeSizeInBits(FieldTy);

      // Ignore zero sized fields like {}, they obviously contain no data.
      if (FieldSizeBits == 0) continue;

      if (FieldSizeBits != AllocaSizeBits)
        EltVal = Builder.CreateTrunc(EltVal,
                             IntegerType::get(SI->getContext(), FieldSizeBits));
      Value *DestField = NewElts[i];
      if (EltVal->getType() == FieldTy) {
        // Storing to an integer field of this size, just do it.
      } else if (FieldTy->isFloatingPointTy() || FieldTy->isVectorTy()) {
        // Bitcast to the right element type (for fp/vector values).
        EltVal = Builder.CreateBitCast(EltVal, FieldTy);
      } else {
        // Otherwise, bitcast the dest pointer (for aggregates).
        DestField = Builder.CreateBitCast(DestField,
                                     PointerType::getUnqual(EltVal->getType()));
      }
      new StoreInst(EltVal, DestField, SI);
    }

  } else {
    ArrayType *ATy = cast<ArrayType>(AllocaEltTy);
    Type *ArrayEltTy = ATy->getElementType();
    uint64_t ElementOffset = DL.getTypeAllocSizeInBits(ArrayEltTy);
    uint64_t ElementSizeBits = DL.getTypeSizeInBits(ArrayEltTy);

    uint64_t Shift;

    if (DL.isBigEndian())
      Shift = AllocaSizeBits-ElementOffset;
    else
      Shift = 0;

    for (unsigned i = 0, e = NewElts.size(); i != e; ++i) {
      // Ignore zero sized fields like {}, they obviously contain no data.
      if (ElementSizeBits == 0) continue;

      Value *EltVal = SrcVal;
      if (Shift) {
        Value *ShiftVal = ConstantInt::get(EltVal->getType(), Shift);
        EltVal = Builder.CreateLShr(EltVal, ShiftVal, "sroa.store.elt");
      }

      // Truncate down to an integer of the right size.
      if (ElementSizeBits != AllocaSizeBits)
        EltVal = Builder.CreateTrunc(EltVal,
                                     IntegerType::get(SI->getContext(),
                                                      ElementSizeBits));
      Value *DestField = NewElts[i];
      if (EltVal->getType() == ArrayEltTy) {
        // Storing to an integer field of this size, just do it.
      } else if (ArrayEltTy->isFloatingPointTy() ||
                 ArrayEltTy->isVectorTy()) {
        // Bitcast to the right element type (for fp/vector values).
        EltVal = Builder.CreateBitCast(EltVal, ArrayEltTy);
      } else {
        // Otherwise, bitcast the dest pointer (for aggregates).
        DestField = Builder.CreateBitCast(DestField,
                                     PointerType::getUnqual(EltVal->getType()));
      }
      new StoreInst(EltVal, DestField, SI);

      if (DL.isBigEndian())
        Shift -= ElementOffset;
      else
        Shift += ElementOffset;
    }
  }

  DeadInsts.push_back(SI);
}

/// RewriteLoadUserOfWholeAlloca - We found a load of the entire allocation to
/// an integer.  Load the individual pieces to form the aggregate value.
void
SROA::RewriteLoadUserOfWholeAlloca(LoadInst *LI, AllocaInst *AI,
                                   SmallVectorImpl<AllocaInst *> &NewElts) {
  // Extract each element out of the NewElts according to its structure offset
  // and form the result value.
  Type *AllocaEltTy = AI->getAllocatedType();
  const DataLayout &DL = LI->getModule()->getDataLayout();
  uint64_t AllocaSizeBits = DL.getTypeAllocSizeInBits(AllocaEltTy);

  DEBUG(dbgs() << "PROMOTING LOAD OF WHOLE ALLOCA: " << *AI << '\n' << *LI
               << '\n');

  // There are two forms here: AI could be an array or struct.  Both cases
  // have different ways to compute the element offset.
  const StructLayout *Layout = nullptr;
  uint64_t ArrayEltBitOffset = 0;
  if (StructType *EltSTy = dyn_cast<StructType>(AllocaEltTy)) {
    Layout = DL.getStructLayout(EltSTy);
  } else {
    Type *ArrayEltTy = cast<ArrayType>(AllocaEltTy)->getElementType();
    ArrayEltBitOffset = DL.getTypeAllocSizeInBits(ArrayEltTy);
  }

  Value *ResultVal =
    Constant::getNullValue(IntegerType::get(LI->getContext(), AllocaSizeBits));

  for (unsigned i = 0, e = NewElts.size(); i != e; ++i) {
    // Load the value from the alloca.  If the NewElt is an aggregate, cast
    // the pointer to an integer of the same size before doing the load.
    Value *SrcField = NewElts[i];
    Type *FieldTy =
      cast<PointerType>(SrcField->getType())->getElementType();
    uint64_t FieldSizeBits = DL.getTypeSizeInBits(FieldTy);

    // Ignore zero sized fields like {}, they obviously contain no data.
    if (FieldSizeBits == 0) continue;

    IntegerType *FieldIntTy = IntegerType::get(LI->getContext(),
                                                     FieldSizeBits);
    if (!FieldTy->isIntegerTy() && !FieldTy->isFloatingPointTy() &&
        !FieldTy->isVectorTy())
      SrcField = new BitCastInst(SrcField,
                                 PointerType::getUnqual(FieldIntTy),
                                 "", LI);
    SrcField = new LoadInst(SrcField, "sroa.load.elt", LI);

    // If SrcField is a fp or vector of the right size but that isn't an
    // integer type, bitcast to an integer so we can shift it.
    if (SrcField->getType() != FieldIntTy)
      SrcField = new BitCastInst(SrcField, FieldIntTy, "", LI);

    // Zero extend the field to be the same size as the final alloca so that
    // we can shift and insert it.
    if (SrcField->getType() != ResultVal->getType())
      SrcField = new ZExtInst(SrcField, ResultVal->getType(), "", LI);

    // Determine the number of bits to shift SrcField.
    uint64_t Shift;
    if (Layout) // Struct case.
      Shift = Layout->getElementOffsetInBits(i);
    else  // Array case.
      Shift = i*ArrayEltBitOffset;

    if (DL.isBigEndian())
      Shift = AllocaSizeBits-Shift-FieldIntTy->getBitWidth();

    if (Shift) {
      Value *ShiftVal = ConstantInt::get(SrcField->getType(), Shift);
      SrcField = BinaryOperator::CreateShl(SrcField, ShiftVal, "", LI);
    }

    // Don't create an 'or x, 0' on the first iteration.
    if (!isa<Constant>(ResultVal) ||
        !cast<Constant>(ResultVal)->isNullValue())
      ResultVal = BinaryOperator::CreateOr(SrcField, ResultVal, "", LI);
    else
      ResultVal = SrcField;
  }

  // Handle tail padding by truncating the result
  if (DL.getTypeSizeInBits(LI->getType()) != AllocaSizeBits)
    ResultVal = new TruncInst(ResultVal, LI->getType(), "", LI);

  LI->replaceAllUsesWith(ResultVal);
  DeadInsts.push_back(LI);
}

/// HasPadding - Return true if the specified type has any structure or
/// alignment padding in between the elements that would be split apart
/// by SROA; return false otherwise.
static bool HasPadding(Type *Ty, const DataLayout &DL) {
  if (ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
    Ty = ATy->getElementType();
    return DL.getTypeSizeInBits(Ty) != DL.getTypeAllocSizeInBits(Ty);
  }

  // SROA currently handles only Arrays and Structs.
  StructType *STy = cast<StructType>(Ty);
  const StructLayout *SL = DL.getStructLayout(STy);
  unsigned PrevFieldBitOffset = 0;
  for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
    unsigned FieldBitOffset = SL->getElementOffsetInBits(i);

    // Check to see if there is any padding between this element and the
    // previous one.
    if (i) {
      unsigned PrevFieldEnd =
        PrevFieldBitOffset+DL.getTypeSizeInBits(STy->getElementType(i-1));
      if (PrevFieldEnd < FieldBitOffset)
        return true;
    }
    PrevFieldBitOffset = FieldBitOffset;
  }
  // Check for tail padding.
  if (unsigned EltCount = STy->getNumElements()) {
    unsigned PrevFieldEnd = PrevFieldBitOffset +
      DL.getTypeSizeInBits(STy->getElementType(EltCount-1));
    if (PrevFieldEnd < SL->getSizeInBits())
      return true;
  }
  return false;
}

/// isSafeStructAllocaToScalarRepl - Check to see if the specified allocation of
/// an aggregate can be broken down into elements.  Return 0 if not, 3 if safe,
/// or 1 if safe after canonicalization has been performed.
bool SROA::isSafeAllocaToScalarRepl(AllocaInst *AI) {
  // Loop over the use list of the alloca.  We can only transform it if all of
  // the users are safe to transform.
  AllocaInfo Info(AI);

  isSafeForScalarRepl(AI, 0, Info);
  if (Info.isUnsafe) {
    DEBUG(dbgs() << "Cannot transform: " << *AI << '\n');
    return false;
  }

  const DataLayout &DL = AI->getModule()->getDataLayout();

  // Okay, we know all the users are promotable.  If the aggregate is a memcpy
  // source and destination, we have to be careful.  In particular, the memcpy
  // could be moving around elements that live in structure padding of the LLVM
  // types, but may actually be used.  In these cases, we refuse to promote the
  // struct.
  if (Info.isMemCpySrc && Info.isMemCpyDst &&
      HasPadding(AI->getAllocatedType(), DL))
    return false;

  // If the alloca never has an access to just *part* of it, but is accessed
  // via loads and stores, then we should use ConvertToScalarInfo to promote
  // the alloca instead of promoting each piece at a time and inserting fission
  // and fusion code.
  if (!Info.hasSubelementAccess && Info.hasALoadOrStore) {
    // If the struct/array just has one element, use basic SRoA.
    if (StructType *ST = dyn_cast<StructType>(AI->getAllocatedType())) {
      if (ST->getNumElements() > 1) return false;
    } else {
      if (cast<ArrayType>(AI->getAllocatedType())->getNumElements() > 1)
        return false;
    }
  }

  return true;
}
