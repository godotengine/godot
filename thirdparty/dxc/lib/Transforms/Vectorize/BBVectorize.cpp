//===- BBVectorize.cpp - A Basic-Block Vectorizer -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a basic-block vectorization pass. The algorithm was
// inspired by that used by the Vienna MAP Vectorizor by Franchetti and Kral,
// et al. It works by looking for chains of pairable operations and then
// pairing them.
//
//===----------------------------------------------------------------------===//

#define BBV_NAME "bb-vectorize"
#include "llvm/Transforms/Vectorize.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Local.h"
#include <algorithm>
using namespace llvm;

#define DEBUG_TYPE BBV_NAME

static cl::opt<bool>
IgnoreTargetInfo("bb-vectorize-ignore-target-info",  cl::init(false),
  cl::Hidden, cl::desc("Ignore target information"));

static cl::opt<unsigned>
ReqChainDepth("bb-vectorize-req-chain-depth", cl::init(6), cl::Hidden,
  cl::desc("The required chain depth for vectorization"));

static cl::opt<bool>
UseChainDepthWithTI("bb-vectorize-use-chain-depth",  cl::init(false),
  cl::Hidden, cl::desc("Use the chain depth requirement with"
                       " target information"));

static cl::opt<unsigned>
SearchLimit("bb-vectorize-search-limit", cl::init(400), cl::Hidden,
  cl::desc("The maximum search distance for instruction pairs"));

static cl::opt<bool>
SplatBreaksChain("bb-vectorize-splat-breaks-chain", cl::init(false), cl::Hidden,
  cl::desc("Replicating one element to a pair breaks the chain"));

static cl::opt<unsigned>
VectorBits("bb-vectorize-vector-bits", cl::init(128), cl::Hidden,
  cl::desc("The size of the native vector registers"));

static cl::opt<unsigned>
MaxIter("bb-vectorize-max-iter", cl::init(0), cl::Hidden,
  cl::desc("The maximum number of pairing iterations"));

static cl::opt<bool>
Pow2LenOnly("bb-vectorize-pow2-len-only", cl::init(false), cl::Hidden,
  cl::desc("Don't try to form non-2^n-length vectors"));

static cl::opt<unsigned>
MaxInsts("bb-vectorize-max-instr-per-group", cl::init(500), cl::Hidden,
  cl::desc("The maximum number of pairable instructions per group"));

static cl::opt<unsigned>
MaxPairs("bb-vectorize-max-pairs-per-group", cl::init(3000), cl::Hidden,
  cl::desc("The maximum number of candidate instruction pairs per group"));

static cl::opt<unsigned>
MaxCandPairsForCycleCheck("bb-vectorize-max-cycle-check-pairs", cl::init(200),
  cl::Hidden, cl::desc("The maximum number of candidate pairs with which to use"
                       " a full cycle check"));

static cl::opt<bool>
NoBools("bb-vectorize-no-bools", cl::init(false), cl::Hidden,
  cl::desc("Don't try to vectorize boolean (i1) values"));

static cl::opt<bool>
NoInts("bb-vectorize-no-ints", cl::init(false), cl::Hidden,
  cl::desc("Don't try to vectorize integer values"));

static cl::opt<bool>
NoFloats("bb-vectorize-no-floats", cl::init(false), cl::Hidden,
  cl::desc("Don't try to vectorize floating-point values"));

// FIXME: This should default to false once pointer vector support works.
static cl::opt<bool>
NoPointers("bb-vectorize-no-pointers", cl::init(/*false*/ true), cl::Hidden,
  cl::desc("Don't try to vectorize pointer values"));

static cl::opt<bool>
NoCasts("bb-vectorize-no-casts", cl::init(false), cl::Hidden,
  cl::desc("Don't try to vectorize casting (conversion) operations"));

static cl::opt<bool>
NoMath("bb-vectorize-no-math", cl::init(false), cl::Hidden,
  cl::desc("Don't try to vectorize floating-point math intrinsics"));

static cl::opt<bool>
  NoBitManipulation("bb-vectorize-no-bitmanip", cl::init(false), cl::Hidden,
  cl::desc("Don't try to vectorize BitManipulation intrinsics"));

static cl::opt<bool>
NoFMA("bb-vectorize-no-fma", cl::init(false), cl::Hidden,
  cl::desc("Don't try to vectorize the fused-multiply-add intrinsic"));

static cl::opt<bool>
NoSelect("bb-vectorize-no-select", cl::init(false), cl::Hidden,
  cl::desc("Don't try to vectorize select instructions"));

static cl::opt<bool>
NoCmp("bb-vectorize-no-cmp", cl::init(false), cl::Hidden,
  cl::desc("Don't try to vectorize comparison instructions"));

static cl::opt<bool>
NoGEP("bb-vectorize-no-gep", cl::init(false), cl::Hidden,
  cl::desc("Don't try to vectorize getelementptr instructions"));

static cl::opt<bool>
NoMemOps("bb-vectorize-no-mem-ops", cl::init(false), cl::Hidden,
  cl::desc("Don't try to vectorize loads and stores"));

static cl::opt<bool>
AlignedOnly("bb-vectorize-aligned-only", cl::init(false), cl::Hidden,
  cl::desc("Only generate aligned loads and stores"));

static cl::opt<bool>
NoMemOpBoost("bb-vectorize-no-mem-op-boost",
  cl::init(false), cl::Hidden,
  cl::desc("Don't boost the chain-depth contribution of loads and stores"));

static cl::opt<bool>
FastDep("bb-vectorize-fast-dep", cl::init(false), cl::Hidden,
  cl::desc("Use a fast instruction dependency analysis"));

#ifndef NDEBUG
static cl::opt<bool>
DebugInstructionExamination("bb-vectorize-debug-instruction-examination",
  cl::init(false), cl::Hidden,
  cl::desc("When debugging is enabled, output information on the"
           " instruction-examination process"));
static cl::opt<bool>
DebugCandidateSelection("bb-vectorize-debug-candidate-selection",
  cl::init(false), cl::Hidden,
  cl::desc("When debugging is enabled, output information on the"
           " candidate-selection process"));
static cl::opt<bool>
DebugPairSelection("bb-vectorize-debug-pair-selection",
  cl::init(false), cl::Hidden,
  cl::desc("When debugging is enabled, output information on the"
           " pair-selection process"));
static cl::opt<bool>
DebugCycleCheck("bb-vectorize-debug-cycle-check",
  cl::init(false), cl::Hidden,
  cl::desc("When debugging is enabled, output information on the"
           " cycle-checking process"));

static cl::opt<bool>
PrintAfterEveryPair("bb-vectorize-debug-print-after-every-pair",
  cl::init(false), cl::Hidden,
  cl::desc("When debugging is enabled, dump the basic block after"
           " every pair is fused"));
#endif

STATISTIC(NumFusedOps, "Number of operations fused by bb-vectorize");

namespace {
  struct BBVectorize : public BasicBlockPass {
    static char ID; // Pass identification, replacement for typeid

    const VectorizeConfig Config;

    BBVectorize(const VectorizeConfig &C = VectorizeConfig())
      : BasicBlockPass(ID), Config(C) {
      initializeBBVectorizePass(*PassRegistry::getPassRegistry());
    }

    BBVectorize(Pass *P, Function &F, const VectorizeConfig &C)
      : BasicBlockPass(ID), Config(C) {
      AA = &P->getAnalysis<AliasAnalysis>();
      DT = &P->getAnalysis<DominatorTreeWrapperPass>().getDomTree();
      SE = &P->getAnalysis<ScalarEvolution>();
      TTI = IgnoreTargetInfo
                ? nullptr
                : &P->getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    }

    typedef std::pair<Value *, Value *> ValuePair;
    typedef std::pair<ValuePair, int> ValuePairWithCost;
    typedef std::pair<ValuePair, size_t> ValuePairWithDepth;
    typedef std::pair<ValuePair, ValuePair> VPPair; // A ValuePair pair
    typedef std::pair<VPPair, unsigned> VPPairWithType;

    AliasAnalysis *AA;
    DominatorTree *DT;
    ScalarEvolution *SE;
    const TargetTransformInfo *TTI;

    // FIXME: const correct?

    bool vectorizePairs(BasicBlock &BB, bool NonPow2Len = false);

    bool getCandidatePairs(BasicBlock &BB,
                       BasicBlock::iterator &Start,
                       DenseMap<Value *, std::vector<Value *> > &CandidatePairs,
                       DenseSet<ValuePair> &FixedOrderPairs,
                       DenseMap<ValuePair, int> &CandidatePairCostSavings,
                       std::vector<Value *> &PairableInsts, bool NonPow2Len);

    // FIXME: The current implementation does not account for pairs that
    // are connected in multiple ways. For example:
    //   C1 = A1 / A2; C2 = A2 / A1 (which may be both direct and a swap)
    enum PairConnectionType {
      PairConnectionDirect,
      PairConnectionSwap,
      PairConnectionSplat
    };

    void computeConnectedPairs(
             DenseMap<Value *, std::vector<Value *> > &CandidatePairs,
             DenseSet<ValuePair> &CandidatePairsSet,
             std::vector<Value *> &PairableInsts,
             DenseMap<ValuePair, std::vector<ValuePair> > &ConnectedPairs,
             DenseMap<VPPair, unsigned> &PairConnectionTypes);

    void buildDepMap(BasicBlock &BB,
             DenseMap<Value *, std::vector<Value *> > &CandidatePairs,
             std::vector<Value *> &PairableInsts,
             DenseSet<ValuePair> &PairableInstUsers);

    void choosePairs(DenseMap<Value *, std::vector<Value *> > &CandidatePairs,
             DenseSet<ValuePair> &CandidatePairsSet,
             DenseMap<ValuePair, int> &CandidatePairCostSavings,
             std::vector<Value *> &PairableInsts,
             DenseSet<ValuePair> &FixedOrderPairs,
             DenseMap<VPPair, unsigned> &PairConnectionTypes,
             DenseMap<ValuePair, std::vector<ValuePair> > &ConnectedPairs,
             DenseMap<ValuePair, std::vector<ValuePair> > &ConnectedPairDeps,
             DenseSet<ValuePair> &PairableInstUsers,
             DenseMap<Value *, Value *>& ChosenPairs);

    void fuseChosenPairs(BasicBlock &BB,
             std::vector<Value *> &PairableInsts,
             DenseMap<Value *, Value *>& ChosenPairs,
             DenseSet<ValuePair> &FixedOrderPairs,
             DenseMap<VPPair, unsigned> &PairConnectionTypes,
             DenseMap<ValuePair, std::vector<ValuePair> > &ConnectedPairs,
             DenseMap<ValuePair, std::vector<ValuePair> > &ConnectedPairDeps);


    bool isInstVectorizable(Instruction *I, bool &IsSimpleLoadStore);

    bool areInstsCompatible(Instruction *I, Instruction *J,
                       bool IsSimpleLoadStore, bool NonPow2Len,
                       int &CostSavings, int &FixedOrder);

    bool trackUsesOfI(DenseSet<Value *> &Users,
                      AliasSetTracker &WriteSet, Instruction *I,
                      Instruction *J, bool UpdateUsers = true,
                      DenseSet<ValuePair> *LoadMoveSetPairs = nullptr);

  void computePairsConnectedTo(
             DenseMap<Value *, std::vector<Value *> > &CandidatePairs,
             DenseSet<ValuePair> &CandidatePairsSet,
             std::vector<Value *> &PairableInsts,
             DenseMap<ValuePair, std::vector<ValuePair> > &ConnectedPairs,
             DenseMap<VPPair, unsigned> &PairConnectionTypes,
             ValuePair P);

    bool pairsConflict(ValuePair P, ValuePair Q,
             DenseSet<ValuePair> &PairableInstUsers,
             DenseMap<ValuePair, std::vector<ValuePair> >
               *PairableInstUserMap = nullptr,
             DenseSet<VPPair> *PairableInstUserPairSet = nullptr);

    bool pairWillFormCycle(ValuePair P,
             DenseMap<ValuePair, std::vector<ValuePair> > &PairableInstUsers,
             DenseSet<ValuePair> &CurrentPairs);

    void pruneDAGFor(
             DenseMap<Value *, std::vector<Value *> > &CandidatePairs,
             std::vector<Value *> &PairableInsts,
             DenseMap<ValuePair, std::vector<ValuePair> > &ConnectedPairs,
             DenseSet<ValuePair> &PairableInstUsers,
             DenseMap<ValuePair, std::vector<ValuePair> > &PairableInstUserMap,
             DenseSet<VPPair> &PairableInstUserPairSet,
             DenseMap<Value *, Value *> &ChosenPairs,
             DenseMap<ValuePair, size_t> &DAG,
             DenseSet<ValuePair> &PrunedDAG, ValuePair J,
             bool UseCycleCheck);

    void buildInitialDAGFor(
             DenseMap<Value *, std::vector<Value *> > &CandidatePairs,
             DenseSet<ValuePair> &CandidatePairsSet,
             std::vector<Value *> &PairableInsts,
             DenseMap<ValuePair, std::vector<ValuePair> > &ConnectedPairs,
             DenseSet<ValuePair> &PairableInstUsers,
             DenseMap<Value *, Value *> &ChosenPairs,
             DenseMap<ValuePair, size_t> &DAG, ValuePair J);

    void findBestDAGFor(
             DenseMap<Value *, std::vector<Value *> > &CandidatePairs,
             DenseSet<ValuePair> &CandidatePairsSet,
             DenseMap<ValuePair, int> &CandidatePairCostSavings,
             std::vector<Value *> &PairableInsts,
             DenseSet<ValuePair> &FixedOrderPairs,
             DenseMap<VPPair, unsigned> &PairConnectionTypes,
             DenseMap<ValuePair, std::vector<ValuePair> > &ConnectedPairs,
             DenseMap<ValuePair, std::vector<ValuePair> > &ConnectedPairDeps,
             DenseSet<ValuePair> &PairableInstUsers,
             DenseMap<ValuePair, std::vector<ValuePair> > &PairableInstUserMap,
             DenseSet<VPPair> &PairableInstUserPairSet,
             DenseMap<Value *, Value *> &ChosenPairs,
             DenseSet<ValuePair> &BestDAG, size_t &BestMaxDepth,
             int &BestEffSize, Value *II, std::vector<Value *>&JJ,
             bool UseCycleCheck);

    Value *getReplacementPointerInput(LLVMContext& Context, Instruction *I,
                     Instruction *J, unsigned o);

    void fillNewShuffleMask(LLVMContext& Context, Instruction *J,
                     unsigned MaskOffset, unsigned NumInElem,
                     unsigned NumInElem1, unsigned IdxOffset,
                     std::vector<Constant*> &Mask);

    Value *getReplacementShuffleMask(LLVMContext& Context, Instruction *I,
                     Instruction *J);

    bool expandIEChain(LLVMContext& Context, Instruction *I, Instruction *J,
                       unsigned o, Value *&LOp, unsigned numElemL,
                       Type *ArgTypeL, Type *ArgTypeR, bool IBeforeJ,
                       unsigned IdxOff = 0);

    Value *getReplacementInput(LLVMContext& Context, Instruction *I,
                     Instruction *J, unsigned o, bool IBeforeJ);

    void getReplacementInputsForPair(LLVMContext& Context, Instruction *I,
                     Instruction *J, SmallVectorImpl<Value *> &ReplacedOperands,
                     bool IBeforeJ);

    void replaceOutputsOfPair(LLVMContext& Context, Instruction *I,
                     Instruction *J, Instruction *K,
                     Instruction *&InsertionPt, Instruction *&K1,
                     Instruction *&K2);

    void collectPairLoadMoveSet(BasicBlock &BB,
                     DenseMap<Value *, Value *> &ChosenPairs,
                     DenseMap<Value *, std::vector<Value *> > &LoadMoveSet,
                     DenseSet<ValuePair> &LoadMoveSetPairs,
                     Instruction *I);

    void collectLoadMoveSet(BasicBlock &BB,
                     std::vector<Value *> &PairableInsts,
                     DenseMap<Value *, Value *> &ChosenPairs,
                     DenseMap<Value *, std::vector<Value *> > &LoadMoveSet,
                     DenseSet<ValuePair> &LoadMoveSetPairs);

    bool canMoveUsesOfIAfterJ(BasicBlock &BB,
                     DenseSet<ValuePair> &LoadMoveSetPairs,
                     Instruction *I, Instruction *J);

    void moveUsesOfIAfterJ(BasicBlock &BB,
                     DenseSet<ValuePair> &LoadMoveSetPairs,
                     Instruction *&InsertionPt,
                     Instruction *I, Instruction *J);

    bool vectorizeBB(BasicBlock &BB) {
      if (skipOptnoneFunction(BB))
        return false;
      if (!DT->isReachableFromEntry(&BB)) {
        DEBUG(dbgs() << "BBV: skipping unreachable " << BB.getName() <<
              " in " << BB.getParent()->getName() << "\n");
        return false;
      }

      DEBUG(if (TTI) dbgs() << "BBV: using target information\n");

      bool changed = false;
      // Iterate a sufficient number of times to merge types of size 1 bit,
      // then 2 bits, then 4, etc. up to half of the target vector width of the
      // target vector register.
      unsigned n = 1;
      for (unsigned v = 2;
           (TTI || v <= Config.VectorBits) &&
           (!Config.MaxIter || n <= Config.MaxIter);
           v *= 2, ++n) {
        DEBUG(dbgs() << "BBV: fusing loop #" << n <<
              " for " << BB.getName() << " in " <<
              BB.getParent()->getName() << "...\n");
        if (vectorizePairs(BB))
          changed = true;
        else
          break;
      }

      if (changed && !Pow2LenOnly) {
        ++n;
        for (; !Config.MaxIter || n <= Config.MaxIter; ++n) {
          DEBUG(dbgs() << "BBV: fusing for non-2^n-length vectors loop #: " <<
                n << " for " << BB.getName() << " in " <<
                BB.getParent()->getName() << "...\n");
          if (!vectorizePairs(BB, true)) break;
        }
      }

      DEBUG(dbgs() << "BBV: done!\n");
      return changed;
    }

    bool runOnBasicBlock(BasicBlock &BB) override {
      // OptimizeNone check deferred to vectorizeBB().

      AA = &getAnalysis<AliasAnalysis>();
      DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
      SE = &getAnalysis<ScalarEvolution>();
      TTI = IgnoreTargetInfo
                ? nullptr
                : &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(
                      *BB.getParent());

      return vectorizeBB(BB);
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      BasicBlockPass::getAnalysisUsage(AU);
      AU.addRequired<AliasAnalysis>();
      AU.addRequired<DominatorTreeWrapperPass>();
      AU.addRequired<ScalarEvolution>();
      AU.addRequired<TargetTransformInfoWrapperPass>();
      AU.addPreserved<AliasAnalysis>();
      AU.addPreserved<DominatorTreeWrapperPass>();
      AU.addPreserved<ScalarEvolution>();
      AU.setPreservesCFG();
    }

    static inline VectorType *getVecTypeForPair(Type *ElemTy, Type *Elem2Ty) {
      assert(ElemTy->getScalarType() == Elem2Ty->getScalarType() &&
             "Cannot form vector from incompatible scalar types");
      Type *STy = ElemTy->getScalarType();

      unsigned numElem;
      if (VectorType *VTy = dyn_cast<VectorType>(ElemTy)) {
        numElem = VTy->getNumElements();
      } else {
        numElem = 1;
      }

      if (VectorType *VTy = dyn_cast<VectorType>(Elem2Ty)) {
        numElem += VTy->getNumElements();
      } else {
        numElem += 1;
      }

      return VectorType::get(STy, numElem);
    }

    static inline void getInstructionTypes(Instruction *I,
                                           Type *&T1, Type *&T2) {
      if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
        // For stores, it is the value type, not the pointer type that matters
        // because the value is what will come from a vector register.
  
        Value *IVal = SI->getValueOperand();
        T1 = IVal->getType();
      } else {
        T1 = I->getType();
      }
  
      if (CastInst *CI = dyn_cast<CastInst>(I))
        T2 = CI->getSrcTy();
      else
        T2 = T1;

      if (SelectInst *SI = dyn_cast<SelectInst>(I)) {
        T2 = SI->getCondition()->getType();
      } else if (ShuffleVectorInst *SI = dyn_cast<ShuffleVectorInst>(I)) {
        T2 = SI->getOperand(0)->getType();
      } else if (CmpInst *CI = dyn_cast<CmpInst>(I)) {
        T2 = CI->getOperand(0)->getType();
      }
    }

    // Returns the weight associated with the provided value. A chain of
    // candidate pairs has a length given by the sum of the weights of its
    // members (one weight per pair; the weight of each member of the pair
    // is assumed to be the same). This length is then compared to the
    // chain-length threshold to determine if a given chain is significant
    // enough to be vectorized. The length is also used in comparing
    // candidate chains where longer chains are considered to be better.
    // Note: when this function returns 0, the resulting instructions are
    // not actually fused.
    inline size_t getDepthFactor(Value *V) {
      // InsertElement and ExtractElement have a depth factor of zero. This is
      // for two reasons: First, they cannot be usefully fused. Second, because
      // the pass generates a lot of these, they can confuse the simple metric
      // used to compare the dags in the next iteration. Thus, giving them a
      // weight of zero allows the pass to essentially ignore them in
      // subsequent iterations when looking for vectorization opportunities
      // while still tracking dependency chains that flow through those
      // instructions.
      if (isa<InsertElementInst>(V) || isa<ExtractElementInst>(V))
        return 0;

      // Give a load or store half of the required depth so that load/store
      // pairs will vectorize.
      if (!Config.NoMemOpBoost && (isa<LoadInst>(V) || isa<StoreInst>(V)))
        return Config.ReqChainDepth/2;

      return 1;
    }

    // Returns the cost of the provided instruction using TTI.
    // This does not handle loads and stores.
    unsigned getInstrCost(unsigned Opcode, Type *T1, Type *T2,
                          TargetTransformInfo::OperandValueKind Op1VK = 
                              TargetTransformInfo::OK_AnyValue,
                          TargetTransformInfo::OperandValueKind Op2VK =
                              TargetTransformInfo::OK_AnyValue) {
      switch (Opcode) {
      default: break;
      case Instruction::GetElementPtr:
        // We mark this instruction as zero-cost because scalar GEPs are usually
        // lowered to the instruction addressing mode. At the moment we don't
        // generate vector GEPs.
        return 0;
      case Instruction::Br:
        return TTI->getCFInstrCost(Opcode);
      case Instruction::PHI:
        return 0;
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
      case Instruction::Xor:
        return TTI->getArithmeticInstrCost(Opcode, T1, Op1VK, Op2VK);
      case Instruction::Select:
      case Instruction::ICmp:
      case Instruction::FCmp:
        return TTI->getCmpSelInstrCost(Opcode, T1, T2);
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
      case Instruction::BitCast:
      case Instruction::ShuffleVector:
        return TTI->getCastInstrCost(Opcode, T1, T2);
      }

      return 1;
    }

    // This determines the relative offset of two loads or stores, returning
    // true if the offset could be determined to be some constant value.
    // For example, if OffsetInElmts == 1, then J accesses the memory directly
    // after I; if OffsetInElmts == -1 then I accesses the memory
    // directly after J.
    bool getPairPtrInfo(Instruction *I, Instruction *J,
        Value *&IPtr, Value *&JPtr, unsigned &IAlignment, unsigned &JAlignment,
        unsigned &IAddressSpace, unsigned &JAddressSpace,
        int64_t &OffsetInElmts, bool ComputeOffset = true) {
      OffsetInElmts = 0;
      if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
        LoadInst *LJ = cast<LoadInst>(J);
        IPtr = LI->getPointerOperand();
        JPtr = LJ->getPointerOperand();
        IAlignment = LI->getAlignment();
        JAlignment = LJ->getAlignment();
        IAddressSpace = LI->getPointerAddressSpace();
        JAddressSpace = LJ->getPointerAddressSpace();
      } else {
        StoreInst *SI = cast<StoreInst>(I), *SJ = cast<StoreInst>(J);
        IPtr = SI->getPointerOperand();
        JPtr = SJ->getPointerOperand();
        IAlignment = SI->getAlignment();
        JAlignment = SJ->getAlignment();
        IAddressSpace = SI->getPointerAddressSpace();
        JAddressSpace = SJ->getPointerAddressSpace();
      }

      if (!ComputeOffset)
        return true;

      const SCEV *IPtrSCEV = SE->getSCEV(IPtr);
      const SCEV *JPtrSCEV = SE->getSCEV(JPtr);

      // If this is a trivial offset, then we'll get something like
      // 1*sizeof(type). With target data, which we need anyway, this will get
      // constant folded into a number.
      const SCEV *OffsetSCEV = SE->getMinusSCEV(JPtrSCEV, IPtrSCEV);
      if (const SCEVConstant *ConstOffSCEV =
            dyn_cast<SCEVConstant>(OffsetSCEV)) {
        ConstantInt *IntOff = ConstOffSCEV->getValue();
        int64_t Offset = IntOff->getSExtValue();
        const DataLayout &DL = I->getModule()->getDataLayout();
        Type *VTy = IPtr->getType()->getPointerElementType();
        int64_t VTyTSS = (int64_t)DL.getTypeStoreSize(VTy);

        Type *VTy2 = JPtr->getType()->getPointerElementType();
        if (VTy != VTy2 && Offset < 0) {
          int64_t VTy2TSS = (int64_t)DL.getTypeStoreSize(VTy2);
          OffsetInElmts = Offset/VTy2TSS;
          return (std::abs(Offset) % VTy2TSS) == 0;
        }

        OffsetInElmts = Offset/VTyTSS;
        return (std::abs(Offset) % VTyTSS) == 0;
      }

      return false;
    }

    // Returns true if the provided CallInst represents an intrinsic that can
    // be vectorized.
    bool isVectorizableIntrinsic(CallInst* I) {
      Function *F = I->getCalledFunction();
      if (!F) return false;

      Intrinsic::ID IID = F->getIntrinsicID();
      if (!IID) return false;

      switch(IID) {
      default:
        return false;
      case Intrinsic::sqrt:
      case Intrinsic::powi:
      case Intrinsic::sin:
      case Intrinsic::cos:
      case Intrinsic::log:
      case Intrinsic::log2:
      case Intrinsic::log10:
      case Intrinsic::exp:
      case Intrinsic::exp2:
      case Intrinsic::pow:
      case Intrinsic::round:
      case Intrinsic::copysign:
      case Intrinsic::ceil:
      case Intrinsic::nearbyint:
      case Intrinsic::rint:
      case Intrinsic::trunc:
      case Intrinsic::floor:
      case Intrinsic::fabs:
      case Intrinsic::minnum:
      case Intrinsic::maxnum:
        return Config.VectorizeMath;
      case Intrinsic::bswap:
      case Intrinsic::ctpop:
      case Intrinsic::ctlz:
      case Intrinsic::cttz:
        return Config.VectorizeBitManipulations;
      case Intrinsic::fma:
      case Intrinsic::fmuladd:
        return Config.VectorizeFMA;
      }
    }

    bool isPureIEChain(InsertElementInst *IE) {
      InsertElementInst *IENext = IE;
      do {
        if (!isa<UndefValue>(IENext->getOperand(0)) &&
            !isa<InsertElementInst>(IENext->getOperand(0))) {
          return false;
        }
      } while ((IENext =
                 dyn_cast<InsertElementInst>(IENext->getOperand(0))));

      return true;
    }
  };

  // This function implements one vectorization iteration on the provided
  // basic block. It returns true if the block is changed.
  bool BBVectorize::vectorizePairs(BasicBlock &BB, bool NonPow2Len) {
    bool ShouldContinue;
    BasicBlock::iterator Start = BB.getFirstInsertionPt();

    std::vector<Value *> AllPairableInsts;
    DenseMap<Value *, Value *> AllChosenPairs;
    DenseSet<ValuePair> AllFixedOrderPairs;
    DenseMap<VPPair, unsigned> AllPairConnectionTypes;
    DenseMap<ValuePair, std::vector<ValuePair> > AllConnectedPairs,
                                                 AllConnectedPairDeps;

    do {
      std::vector<Value *> PairableInsts;
      DenseMap<Value *, std::vector<Value *> > CandidatePairs;
      DenseSet<ValuePair> FixedOrderPairs;
      DenseMap<ValuePair, int> CandidatePairCostSavings;
      ShouldContinue = getCandidatePairs(BB, Start, CandidatePairs,
                                         FixedOrderPairs,
                                         CandidatePairCostSavings,
                                         PairableInsts, NonPow2Len);
      if (PairableInsts.empty()) continue;

      // Build the candidate pair set for faster lookups.
      DenseSet<ValuePair> CandidatePairsSet;
      for (DenseMap<Value *, std::vector<Value *> >::iterator I =
           CandidatePairs.begin(), E = CandidatePairs.end(); I != E; ++I)
        for (std::vector<Value *>::iterator J = I->second.begin(),
             JE = I->second.end(); J != JE; ++J)
          CandidatePairsSet.insert(ValuePair(I->first, *J));

      // Now we have a map of all of the pairable instructions and we need to
      // select the best possible pairing. A good pairing is one such that the
      // users of the pair are also paired. This defines a (directed) forest
      // over the pairs such that two pairs are connected iff the second pair
      // uses the first.

      // Note that it only matters that both members of the second pair use some
      // element of the first pair (to allow for splatting).

      DenseMap<ValuePair, std::vector<ValuePair> > ConnectedPairs,
                                                   ConnectedPairDeps;
      DenseMap<VPPair, unsigned> PairConnectionTypes;
      computeConnectedPairs(CandidatePairs, CandidatePairsSet,
                            PairableInsts, ConnectedPairs, PairConnectionTypes);
      if (ConnectedPairs.empty()) continue;

      for (DenseMap<ValuePair, std::vector<ValuePair> >::iterator
           I = ConnectedPairs.begin(), IE = ConnectedPairs.end();
           I != IE; ++I)
        for (std::vector<ValuePair>::iterator J = I->second.begin(),
             JE = I->second.end(); J != JE; ++J)
          ConnectedPairDeps[*J].push_back(I->first);

      // Build the pairable-instruction dependency map
      DenseSet<ValuePair> PairableInstUsers;
      buildDepMap(BB, CandidatePairs, PairableInsts, PairableInstUsers);

      // There is now a graph of the connected pairs. For each variable, pick
      // the pairing with the largest dag meeting the depth requirement on at
      // least one branch. Then select all pairings that are part of that dag
      // and remove them from the list of available pairings and pairable
      // variables.

      DenseMap<Value *, Value *> ChosenPairs;
      choosePairs(CandidatePairs, CandidatePairsSet,
        CandidatePairCostSavings,
        PairableInsts, FixedOrderPairs, PairConnectionTypes,
        ConnectedPairs, ConnectedPairDeps,
        PairableInstUsers, ChosenPairs);

      if (ChosenPairs.empty()) continue;
      AllPairableInsts.insert(AllPairableInsts.end(), PairableInsts.begin(),
                              PairableInsts.end());
      AllChosenPairs.insert(ChosenPairs.begin(), ChosenPairs.end());

      // Only for the chosen pairs, propagate information on fixed-order pairs,
      // pair connections, and their types to the data structures used by the
      // pair fusion procedures.
      for (DenseMap<Value *, Value *>::iterator I = ChosenPairs.begin(),
           IE = ChosenPairs.end(); I != IE; ++I) {
        if (FixedOrderPairs.count(*I))
          AllFixedOrderPairs.insert(*I);
        else if (FixedOrderPairs.count(ValuePair(I->second, I->first)))
          AllFixedOrderPairs.insert(ValuePair(I->second, I->first));

        for (DenseMap<Value *, Value *>::iterator J = ChosenPairs.begin();
             J != IE; ++J) {
          DenseMap<VPPair, unsigned>::iterator K =
            PairConnectionTypes.find(VPPair(*I, *J));
          if (K != PairConnectionTypes.end()) {
            AllPairConnectionTypes.insert(*K);
          } else {
            K = PairConnectionTypes.find(VPPair(*J, *I));
            if (K != PairConnectionTypes.end())
              AllPairConnectionTypes.insert(*K);
          }
        }
      }

      for (DenseMap<ValuePair, std::vector<ValuePair> >::iterator
           I = ConnectedPairs.begin(), IE = ConnectedPairs.end();
           I != IE; ++I)
        for (std::vector<ValuePair>::iterator J = I->second.begin(),
          JE = I->second.end(); J != JE; ++J)
          if (AllPairConnectionTypes.count(VPPair(I->first, *J))) {
            AllConnectedPairs[I->first].push_back(*J);
            AllConnectedPairDeps[*J].push_back(I->first);
          }
    } while (ShouldContinue);

    if (AllChosenPairs.empty()) return false;
    NumFusedOps += AllChosenPairs.size();

    // A set of pairs has now been selected. It is now necessary to replace the
    // paired instructions with vector instructions. For this procedure each
    // operand must be replaced with a vector operand. This vector is formed
    // by using build_vector on the old operands. The replaced values are then
    // replaced with a vector_extract on the result.  Subsequent optimization
    // passes should coalesce the build/extract combinations.

    fuseChosenPairs(BB, AllPairableInsts, AllChosenPairs, AllFixedOrderPairs,
                    AllPairConnectionTypes,
                    AllConnectedPairs, AllConnectedPairDeps);

    // It is important to cleanup here so that future iterations of this
    // function have less work to do.
    (void)SimplifyInstructionsInBlock(&BB, AA->getTargetLibraryInfo());
    return true;
  }

  // This function returns true if the provided instruction is capable of being
  // fused into a vector instruction. This determination is based only on the
  // type and other attributes of the instruction.
  bool BBVectorize::isInstVectorizable(Instruction *I,
                                         bool &IsSimpleLoadStore) {
    IsSimpleLoadStore = false;

    if (CallInst *C = dyn_cast<CallInst>(I)) {
      if (!isVectorizableIntrinsic(C))
        return false;
    } else if (LoadInst *L = dyn_cast<LoadInst>(I)) {
      // Vectorize simple loads if possbile:
      IsSimpleLoadStore = L->isSimple();
      if (!IsSimpleLoadStore || !Config.VectorizeMemOps)
        return false;
    } else if (StoreInst *S = dyn_cast<StoreInst>(I)) {
      // Vectorize simple stores if possbile:
      IsSimpleLoadStore = S->isSimple();
      if (!IsSimpleLoadStore || !Config.VectorizeMemOps)
        return false;
    } else if (CastInst *C = dyn_cast<CastInst>(I)) {
      // We can vectorize casts, but not casts of pointer types, etc.
      if (!Config.VectorizeCasts)
        return false;

      Type *SrcTy = C->getSrcTy();
      if (!SrcTy->isSingleValueType())
        return false;

      Type *DestTy = C->getDestTy();
      if (!DestTy->isSingleValueType())
        return false;
    } else if (isa<SelectInst>(I)) {
      if (!Config.VectorizeSelect)
        return false;
    } else if (isa<CmpInst>(I)) {
      if (!Config.VectorizeCmp)
        return false;
    } else if (GetElementPtrInst *G = dyn_cast<GetElementPtrInst>(I)) {
      if (!Config.VectorizeGEP)
        return false;

      // Currently, vector GEPs exist only with one index.
      if (G->getNumIndices() != 1)
        return false;
    } else if (!(I->isBinaryOp() || isa<ShuffleVectorInst>(I) ||
        isa<ExtractElementInst>(I) || isa<InsertElementInst>(I))) {
      return false;
    }

    Type *T1, *T2;
    getInstructionTypes(I, T1, T2);

    // Not every type can be vectorized...
    if (!(VectorType::isValidElementType(T1) || T1->isVectorTy()) ||
        !(VectorType::isValidElementType(T2) || T2->isVectorTy()))
      return false;

    if (T1->getScalarSizeInBits() == 1) {
      if (!Config.VectorizeBools)
        return false;
    } else {
      if (!Config.VectorizeInts && T1->isIntOrIntVectorTy())
        return false;
    }

    if (T2->getScalarSizeInBits() == 1) {
      if (!Config.VectorizeBools)
        return false;
    } else {
      if (!Config.VectorizeInts && T2->isIntOrIntVectorTy())
        return false;
    }

    if (!Config.VectorizeFloats
        && (T1->isFPOrFPVectorTy() || T2->isFPOrFPVectorTy()))
      return false;

    // Don't vectorize target-specific types.
    if (T1->isX86_FP80Ty() || T1->isPPC_FP128Ty() || T1->isX86_MMXTy())
      return false;
    if (T2->isX86_FP80Ty() || T2->isPPC_FP128Ty() || T2->isX86_MMXTy())
      return false;

    if (!Config.VectorizePointers && (T1->getScalarType()->isPointerTy() ||
                                      T2->getScalarType()->isPointerTy()))
      return false;

    if (!TTI && (T1->getPrimitiveSizeInBits() >= Config.VectorBits ||
                 T2->getPrimitiveSizeInBits() >= Config.VectorBits))
      return false;

    return true;
  }

  // This function returns true if the two provided instructions are compatible
  // (meaning that they can be fused into a vector instruction). This assumes
  // that I has already been determined to be vectorizable and that J is not
  // in the use dag of I.
  bool BBVectorize::areInstsCompatible(Instruction *I, Instruction *J,
                       bool IsSimpleLoadStore, bool NonPow2Len,
                       int &CostSavings, int &FixedOrder) {
    DEBUG(if (DebugInstructionExamination) dbgs() << "BBV: looking at " << *I <<
                     " <-> " << *J << "\n");

    CostSavings = 0;
    FixedOrder = 0;

    // Loads and stores can be merged if they have different alignments,
    // but are otherwise the same.
    if (!J->isSameOperationAs(I, Instruction::CompareIgnoringAlignment |
                      (NonPow2Len ? Instruction::CompareUsingScalarTypes : 0)))
      return false;

    Type *IT1, *IT2, *JT1, *JT2;
    getInstructionTypes(I, IT1, IT2);
    getInstructionTypes(J, JT1, JT2);
    unsigned MaxTypeBits = std::max(
      IT1->getPrimitiveSizeInBits() + JT1->getPrimitiveSizeInBits(),
      IT2->getPrimitiveSizeInBits() + JT2->getPrimitiveSizeInBits());
    if (!TTI && MaxTypeBits > Config.VectorBits)
      return false;

    // FIXME: handle addsub-type operations!

    if (IsSimpleLoadStore) {
      Value *IPtr, *JPtr;
      unsigned IAlignment, JAlignment, IAddressSpace, JAddressSpace;
      int64_t OffsetInElmts = 0;
      if (getPairPtrInfo(I, J, IPtr, JPtr, IAlignment, JAlignment,
                         IAddressSpace, JAddressSpace, OffsetInElmts) &&
          std::abs(OffsetInElmts) == 1) {
        FixedOrder = (int) OffsetInElmts;
        unsigned BottomAlignment = IAlignment;
        if (OffsetInElmts < 0) BottomAlignment = JAlignment;

        Type *aTypeI = isa<StoreInst>(I) ?
          cast<StoreInst>(I)->getValueOperand()->getType() : I->getType();
        Type *aTypeJ = isa<StoreInst>(J) ?
          cast<StoreInst>(J)->getValueOperand()->getType() : J->getType();
        Type *VType = getVecTypeForPair(aTypeI, aTypeJ);

        if (Config.AlignedOnly) {
          // An aligned load or store is possible only if the instruction
          // with the lower offset has an alignment suitable for the
          // vector type.
          const DataLayout &DL = I->getModule()->getDataLayout();
          unsigned VecAlignment = DL.getPrefTypeAlignment(VType);
          if (BottomAlignment < VecAlignment)
            return false;
        }

        if (TTI) {
          unsigned ICost = TTI->getMemoryOpCost(I->getOpcode(), aTypeI,
                                                IAlignment, IAddressSpace);
          unsigned JCost = TTI->getMemoryOpCost(J->getOpcode(), aTypeJ,
                                                JAlignment, JAddressSpace);
          unsigned VCost = TTI->getMemoryOpCost(I->getOpcode(), VType,
                                                BottomAlignment,
                                                IAddressSpace);

          ICost += TTI->getAddressComputationCost(aTypeI);
          JCost += TTI->getAddressComputationCost(aTypeJ);
          VCost += TTI->getAddressComputationCost(VType);

          if (VCost > ICost + JCost)
            return false;

          // We don't want to fuse to a type that will be split, even
          // if the two input types will also be split and there is no other
          // associated cost.
          unsigned VParts = TTI->getNumberOfParts(VType);
          if (VParts > 1)
            return false;
          else if (!VParts && VCost == ICost + JCost)
            return false;

          CostSavings = ICost + JCost - VCost;
        }
      } else {
        return false;
      }
    } else if (TTI) {
      unsigned ICost = getInstrCost(I->getOpcode(), IT1, IT2);
      unsigned JCost = getInstrCost(J->getOpcode(), JT1, JT2);
      Type *VT1 = getVecTypeForPair(IT1, JT1),
           *VT2 = getVecTypeForPair(IT2, JT2);
      TargetTransformInfo::OperandValueKind Op1VK =
          TargetTransformInfo::OK_AnyValue;
      TargetTransformInfo::OperandValueKind Op2VK =
          TargetTransformInfo::OK_AnyValue;

      // On some targets (example X86) the cost of a vector shift may vary
      // depending on whether the second operand is a Uniform or
      // NonUniform Constant.
      switch (I->getOpcode()) {
      default : break;
      case Instruction::Shl:
      case Instruction::LShr:
      case Instruction::AShr:

        // If both I and J are scalar shifts by constant, then the
        // merged vector shift count would be either a constant splat value
        // or a non-uniform vector of constants.
        if (ConstantInt *CII = dyn_cast<ConstantInt>(I->getOperand(1))) {
          if (ConstantInt *CIJ = dyn_cast<ConstantInt>(J->getOperand(1)))
            Op2VK = CII == CIJ ? TargetTransformInfo::OK_UniformConstantValue :
                               TargetTransformInfo::OK_NonUniformConstantValue;
        } else {
          // Check for a splat of a constant or for a non uniform vector
          // of constants.
          Value *IOp = I->getOperand(1);
          Value *JOp = J->getOperand(1);
          if ((isa<ConstantVector>(IOp) || isa<ConstantDataVector>(IOp)) &&
              (isa<ConstantVector>(JOp) || isa<ConstantDataVector>(JOp))) {
            Op2VK = TargetTransformInfo::OK_NonUniformConstantValue;
            Constant *SplatValue = cast<Constant>(IOp)->getSplatValue();
            if (SplatValue != nullptr &&
                SplatValue == cast<Constant>(JOp)->getSplatValue())
              Op2VK = TargetTransformInfo::OK_UniformConstantValue;
          }
        }
      }

      // Note that this procedure is incorrect for insert and extract element
      // instructions (because combining these often results in a shuffle),
      // but this cost is ignored (because insert and extract element
      // instructions are assigned a zero depth factor and are not really
      // fused in general).
      unsigned VCost = getInstrCost(I->getOpcode(), VT1, VT2, Op1VK, Op2VK);

      if (VCost > ICost + JCost)
        return false;

      // We don't want to fuse to a type that will be split, even
      // if the two input types will also be split and there is no other
      // associated cost.
      unsigned VParts1 = TTI->getNumberOfParts(VT1),
               VParts2 = TTI->getNumberOfParts(VT2);
      if (VParts1 > 1 || VParts2 > 1)
        return false;
      else if ((!VParts1 || !VParts2) && VCost == ICost + JCost)
        return false;

      CostSavings = ICost + JCost - VCost;
    }

    // The powi,ctlz,cttz intrinsics are special because only the first
    // argument is vectorized, the second arguments must be equal.
    CallInst *CI = dyn_cast<CallInst>(I);
    Function *FI;
    if (CI && (FI = CI->getCalledFunction())) {
      Intrinsic::ID IID = FI->getIntrinsicID();
      if (IID == Intrinsic::powi || IID == Intrinsic::ctlz ||
          IID == Intrinsic::cttz) {
        Value *A1I = CI->getArgOperand(1),
              *A1J = cast<CallInst>(J)->getArgOperand(1);
        const SCEV *A1ISCEV = SE->getSCEV(A1I),
                   *A1JSCEV = SE->getSCEV(A1J);
        return (A1ISCEV == A1JSCEV);
      }

      if (IID && TTI) {
        SmallVector<Type*, 4> Tys;
        for (unsigned i = 0, ie = CI->getNumArgOperands(); i != ie; ++i)
          Tys.push_back(CI->getArgOperand(i)->getType());
        unsigned ICost = TTI->getIntrinsicInstrCost(IID, IT1, Tys);

        Tys.clear();
        CallInst *CJ = cast<CallInst>(J);
        for (unsigned i = 0, ie = CJ->getNumArgOperands(); i != ie; ++i)
          Tys.push_back(CJ->getArgOperand(i)->getType());
        unsigned JCost = TTI->getIntrinsicInstrCost(IID, JT1, Tys);

        Tys.clear();
        assert(CI->getNumArgOperands() == CJ->getNumArgOperands() &&
               "Intrinsic argument counts differ");
        for (unsigned i = 0, ie = CI->getNumArgOperands(); i != ie; ++i) {
          if ((IID == Intrinsic::powi || IID == Intrinsic::ctlz ||
               IID == Intrinsic::cttz) && i == 1)
            Tys.push_back(CI->getArgOperand(i)->getType());
          else
            Tys.push_back(getVecTypeForPair(CI->getArgOperand(i)->getType(),
                                            CJ->getArgOperand(i)->getType()));
        }

        Type *RetTy = getVecTypeForPair(IT1, JT1);
        unsigned VCost = TTI->getIntrinsicInstrCost(IID, RetTy, Tys);

        if (VCost > ICost + JCost)
          return false;

        // We don't want to fuse to a type that will be split, even
        // if the two input types will also be split and there is no other
        // associated cost.
        unsigned RetParts = TTI->getNumberOfParts(RetTy);
        if (RetParts > 1)
          return false;
        else if (!RetParts && VCost == ICost + JCost)
          return false;

        for (unsigned i = 0, ie = CI->getNumArgOperands(); i != ie; ++i) {
          if (!Tys[i]->isVectorTy())
            continue;

          unsigned NumParts = TTI->getNumberOfParts(Tys[i]);
          if (NumParts > 1)
            return false;
          else if (!NumParts && VCost == ICost + JCost)
            return false;
        }

        CostSavings = ICost + JCost - VCost;
      }
    }

    return true;
  }

  // Figure out whether or not J uses I and update the users and write-set
  // structures associated with I. Specifically, Users represents the set of
  // instructions that depend on I. WriteSet represents the set
  // of memory locations that are dependent on I. If UpdateUsers is true,
  // and J uses I, then Users is updated to contain J and WriteSet is updated
  // to contain any memory locations to which J writes. The function returns
  // true if J uses I. By default, alias analysis is used to determine
  // whether J reads from memory that overlaps with a location in WriteSet.
  // If LoadMoveSet is not null, then it is a previously-computed map
  // where the key is the memory-based user instruction and the value is
  // the instruction to be compared with I. So, if LoadMoveSet is provided,
  // then the alias analysis is not used. This is necessary because this
  // function is called during the process of moving instructions during
  // vectorization and the results of the alias analysis are not stable during
  // that process.
  bool BBVectorize::trackUsesOfI(DenseSet<Value *> &Users,
                       AliasSetTracker &WriteSet, Instruction *I,
                       Instruction *J, bool UpdateUsers,
                       DenseSet<ValuePair> *LoadMoveSetPairs) {
    bool UsesI = false;

    // This instruction may already be marked as a user due, for example, to
    // being a member of a selected pair.
    if (Users.count(J))
      UsesI = true;

    if (!UsesI)
      for (User::op_iterator JU = J->op_begin(), JE = J->op_end();
           JU != JE; ++JU) {
        Value *V = *JU;
        if (I == V || Users.count(V)) {
          UsesI = true;
          break;
        }
      }
    if (!UsesI && J->mayReadFromMemory()) {
      if (LoadMoveSetPairs) {
        UsesI = LoadMoveSetPairs->count(ValuePair(J, I));
      } else {
        for (AliasSetTracker::iterator W = WriteSet.begin(),
             WE = WriteSet.end(); W != WE; ++W) {
          if (W->aliasesUnknownInst(J, *AA)) {
            UsesI = true;
            break;
          }
        }
      }
    }

    if (UsesI && UpdateUsers) {
      if (J->mayWriteToMemory()) WriteSet.add(J);
      Users.insert(J);
    }

    return UsesI;
  }

  // This function iterates over all instruction pairs in the provided
  // basic block and collects all candidate pairs for vectorization.
  bool BBVectorize::getCandidatePairs(BasicBlock &BB,
                       BasicBlock::iterator &Start,
                       DenseMap<Value *, std::vector<Value *> > &CandidatePairs,
                       DenseSet<ValuePair> &FixedOrderPairs,
                       DenseMap<ValuePair, int> &CandidatePairCostSavings,
                       std::vector<Value *> &PairableInsts, bool NonPow2Len) {
    size_t TotalPairs = 0;
    BasicBlock::iterator E = BB.end();
    if (Start == E) return false;

    bool ShouldContinue = false, IAfterStart = false;
    for (BasicBlock::iterator I = Start++; I != E; ++I) {
      if (I == Start) IAfterStart = true;

      bool IsSimpleLoadStore;
      if (!isInstVectorizable(I, IsSimpleLoadStore)) continue;

      // Look for an instruction with which to pair instruction *I...
      DenseSet<Value *> Users;
      AliasSetTracker WriteSet(*AA);
      if (I->mayWriteToMemory()) WriteSet.add(I);

      bool JAfterStart = IAfterStart;
      BasicBlock::iterator J = std::next(I);
      for (unsigned ss = 0; J != E && ss <= Config.SearchLimit; ++J, ++ss) {
        if (J == Start) JAfterStart = true;

        // Determine if J uses I, if so, exit the loop.
        bool UsesI = trackUsesOfI(Users, WriteSet, I, J, !Config.FastDep);
        if (Config.FastDep) {
          // Note: For this heuristic to be effective, independent operations
          // must tend to be intermixed. This is likely to be true from some
          // kinds of grouped loop unrolling (but not the generic LLVM pass),
          // but otherwise may require some kind of reordering pass.

          // When using fast dependency analysis,
          // stop searching after first use:
          if (UsesI) break;
        } else {
          if (UsesI) continue;
        }

        // J does not use I, and comes before the first use of I, so it can be
        // merged with I if the instructions are compatible.
        int CostSavings, FixedOrder;
        if (!areInstsCompatible(I, J, IsSimpleLoadStore, NonPow2Len,
            CostSavings, FixedOrder)) continue;

        // J is a candidate for merging with I.
        if (PairableInsts.empty() ||
             PairableInsts[PairableInsts.size()-1] != I) {
          PairableInsts.push_back(I);
        }

        CandidatePairs[I].push_back(J);
        ++TotalPairs;
        if (TTI)
          CandidatePairCostSavings.insert(ValuePairWithCost(ValuePair(I, J),
                                                            CostSavings));

        if (FixedOrder == 1)
          FixedOrderPairs.insert(ValuePair(I, J));
        else if (FixedOrder == -1)
          FixedOrderPairs.insert(ValuePair(J, I));

        // The next call to this function must start after the last instruction
        // selected during this invocation.
        if (JAfterStart) {
          Start = std::next(J);
          IAfterStart = JAfterStart = false;
        }

        DEBUG(if (DebugCandidateSelection) dbgs() << "BBV: candidate pair "
                     << *I << " <-> " << *J << " (cost savings: " <<
                     CostSavings << ")\n");

        // If we have already found too many pairs, break here and this function
        // will be called again starting after the last instruction selected
        // during this invocation.
        if (PairableInsts.size() >= Config.MaxInsts ||
            TotalPairs >= Config.MaxPairs) {
          ShouldContinue = true;
          break;
        }
      }

      if (ShouldContinue)
        break;
    }

    DEBUG(dbgs() << "BBV: found " << PairableInsts.size()
           << " instructions with candidate pairs\n");

    return ShouldContinue;
  }

  // Finds candidate pairs connected to the pair P = <PI, PJ>. This means that
  // it looks for pairs such that both members have an input which is an
  // output of PI or PJ.
  void BBVectorize::computePairsConnectedTo(
                  DenseMap<Value *, std::vector<Value *> > &CandidatePairs,
                  DenseSet<ValuePair> &CandidatePairsSet,
                  std::vector<Value *> &PairableInsts,
                  DenseMap<ValuePair, std::vector<ValuePair> > &ConnectedPairs,
                  DenseMap<VPPair, unsigned> &PairConnectionTypes,
                  ValuePair P) {
    StoreInst *SI, *SJ;

    // For each possible pairing for this variable, look at the uses of
    // the first value...
    for (Value::user_iterator I = P.first->user_begin(),
                              E = P.first->user_end();
         I != E; ++I) {
      User *UI = *I;
      if (isa<LoadInst>(UI)) {
        // A pair cannot be connected to a load because the load only takes one
        // operand (the address) and it is a scalar even after vectorization.
        continue;
      } else if ((SI = dyn_cast<StoreInst>(UI)) &&
                 P.first == SI->getPointerOperand()) {
        // Similarly, a pair cannot be connected to a store through its
        // pointer operand.
        continue;
      }

      // For each use of the first variable, look for uses of the second
      // variable...
      for (User *UJ : P.second->users()) {
        if ((SJ = dyn_cast<StoreInst>(UJ)) &&
            P.second == SJ->getPointerOperand())
          continue;

        // Look for <I, J>:
        if (CandidatePairsSet.count(ValuePair(UI, UJ))) {
          VPPair VP(P, ValuePair(UI, UJ));
          ConnectedPairs[VP.first].push_back(VP.second);
          PairConnectionTypes.insert(VPPairWithType(VP, PairConnectionDirect));
        }

        // Look for <J, I>:
        if (CandidatePairsSet.count(ValuePair(UJ, UI))) {
          VPPair VP(P, ValuePair(UJ, UI));
          ConnectedPairs[VP.first].push_back(VP.second);
          PairConnectionTypes.insert(VPPairWithType(VP, PairConnectionSwap));
        }
      }

      if (Config.SplatBreaksChain) continue;
      // Look for cases where just the first value in the pair is used by
      // both members of another pair (splatting).
      for (Value::user_iterator J = P.first->user_begin(); J != E; ++J) {
        User *UJ = *J;
        if ((SJ = dyn_cast<StoreInst>(UJ)) &&
            P.first == SJ->getPointerOperand())
          continue;

        if (CandidatePairsSet.count(ValuePair(UI, UJ))) {
          VPPair VP(P, ValuePair(UI, UJ));
          ConnectedPairs[VP.first].push_back(VP.second);
          PairConnectionTypes.insert(VPPairWithType(VP, PairConnectionSplat));
        }
      }
    }

    if (Config.SplatBreaksChain) return;
    // Look for cases where just the second value in the pair is used by
    // both members of another pair (splatting).
    for (Value::user_iterator I = P.second->user_begin(),
                              E = P.second->user_end();
         I != E; ++I) {
      User *UI = *I;
      if (isa<LoadInst>(UI))
        continue;
      else if ((SI = dyn_cast<StoreInst>(UI)) &&
               P.second == SI->getPointerOperand())
        continue;

      for (Value::user_iterator J = P.second->user_begin(); J != E; ++J) {
        User *UJ = *J;
        if ((SJ = dyn_cast<StoreInst>(UJ)) &&
            P.second == SJ->getPointerOperand())
          continue;

        if (CandidatePairsSet.count(ValuePair(UI, UJ))) {
          VPPair VP(P, ValuePair(UI, UJ));
          ConnectedPairs[VP.first].push_back(VP.second);
          PairConnectionTypes.insert(VPPairWithType(VP, PairConnectionSplat));
        }
      }
    }
  }

  // This function figures out which pairs are connected.  Two pairs are
  // connected if some output of the first pair forms an input to both members
  // of the second pair.
  void BBVectorize::computeConnectedPairs(
                  DenseMap<Value *, std::vector<Value *> > &CandidatePairs,
                  DenseSet<ValuePair> &CandidatePairsSet,
                  std::vector<Value *> &PairableInsts,
                  DenseMap<ValuePair, std::vector<ValuePair> > &ConnectedPairs,
                  DenseMap<VPPair, unsigned> &PairConnectionTypes) {
    for (std::vector<Value *>::iterator PI = PairableInsts.begin(),
         PE = PairableInsts.end(); PI != PE; ++PI) {
      DenseMap<Value *, std::vector<Value *> >::iterator PP =
        CandidatePairs.find(*PI);
      if (PP == CandidatePairs.end())
        continue;

      for (std::vector<Value *>::iterator P = PP->second.begin(),
           E = PP->second.end(); P != E; ++P)
        computePairsConnectedTo(CandidatePairs, CandidatePairsSet,
                                PairableInsts, ConnectedPairs,
                                PairConnectionTypes, ValuePair(*PI, *P));
    }

    DEBUG(size_t TotalPairs = 0;
          for (DenseMap<ValuePair, std::vector<ValuePair> >::iterator I =
               ConnectedPairs.begin(), IE = ConnectedPairs.end(); I != IE; ++I)
            TotalPairs += I->second.size();
          dbgs() << "BBV: found " << TotalPairs
                 << " pair connections.\n");
  }

  // This function builds a set of use tuples such that <A, B> is in the set
  // if B is in the use dag of A. If B is in the use dag of A, then B
  // depends on the output of A.
  void BBVectorize::buildDepMap(
                      BasicBlock &BB,
                      DenseMap<Value *, std::vector<Value *> > &CandidatePairs,
                      std::vector<Value *> &PairableInsts,
                      DenseSet<ValuePair> &PairableInstUsers) {
    DenseSet<Value *> IsInPair;
    for (DenseMap<Value *, std::vector<Value *> >::iterator C =
         CandidatePairs.begin(), E = CandidatePairs.end(); C != E; ++C) {
      IsInPair.insert(C->first);
      IsInPair.insert(C->second.begin(), C->second.end());
    }

    // Iterate through the basic block, recording all users of each
    // pairable instruction.

    BasicBlock::iterator E = BB.end(), EL =
      BasicBlock::iterator(cast<Instruction>(PairableInsts.back()));
    for (BasicBlock::iterator I = BB.getFirstInsertionPt(); I != E; ++I) {
      if (IsInPair.find(I) == IsInPair.end()) continue;

      DenseSet<Value *> Users;
      AliasSetTracker WriteSet(*AA);
      if (I->mayWriteToMemory()) WriteSet.add(I);

      for (BasicBlock::iterator J = std::next(I); J != E; ++J) {
        (void) trackUsesOfI(Users, WriteSet, I, J);

        if (J == EL)
          break;
      }

      for (DenseSet<Value *>::iterator U = Users.begin(), E = Users.end();
           U != E; ++U) {
        if (IsInPair.find(*U) == IsInPair.end()) continue;
        PairableInstUsers.insert(ValuePair(I, *U));
      }

      if (I == EL)
        break;
    }
  }

  // Returns true if an input to pair P is an output of pair Q and also an
  // input of pair Q is an output of pair P. If this is the case, then these
  // two pairs cannot be simultaneously fused.
  bool BBVectorize::pairsConflict(ValuePair P, ValuePair Q,
             DenseSet<ValuePair> &PairableInstUsers,
             DenseMap<ValuePair, std::vector<ValuePair> > *PairableInstUserMap,
             DenseSet<VPPair> *PairableInstUserPairSet) {
    // Two pairs are in conflict if they are mutual Users of eachother.
    bool QUsesP = PairableInstUsers.count(ValuePair(P.first,  Q.first))  ||
                  PairableInstUsers.count(ValuePair(P.first,  Q.second)) ||
                  PairableInstUsers.count(ValuePair(P.second, Q.first))  ||
                  PairableInstUsers.count(ValuePair(P.second, Q.second));
    bool PUsesQ = PairableInstUsers.count(ValuePair(Q.first,  P.first))  ||
                  PairableInstUsers.count(ValuePair(Q.first,  P.second)) ||
                  PairableInstUsers.count(ValuePair(Q.second, P.first))  ||
                  PairableInstUsers.count(ValuePair(Q.second, P.second));
    if (PairableInstUserMap) {
      // FIXME: The expensive part of the cycle check is not so much the cycle
      // check itself but this edge insertion procedure. This needs some
      // profiling and probably a different data structure.
      if (PUsesQ) {
        if (PairableInstUserPairSet->insert(VPPair(Q, P)).second)
          (*PairableInstUserMap)[Q].push_back(P);
      }
      if (QUsesP) {
        if (PairableInstUserPairSet->insert(VPPair(P, Q)).second)
          (*PairableInstUserMap)[P].push_back(Q);
      }
    }

    return (QUsesP && PUsesQ);
  }

  // This function walks the use graph of current pairs to see if, starting
  // from P, the walk returns to P.
  bool BBVectorize::pairWillFormCycle(ValuePair P,
             DenseMap<ValuePair, std::vector<ValuePair> > &PairableInstUserMap,
             DenseSet<ValuePair> &CurrentPairs) {
    DEBUG(if (DebugCycleCheck)
            dbgs() << "BBV: starting cycle check for : " << *P.first << " <-> "
                   << *P.second << "\n");
    // A lookup table of visisted pairs is kept because the PairableInstUserMap
    // contains non-direct associations.
    DenseSet<ValuePair> Visited;
    SmallVector<ValuePair, 32> Q;
    // General depth-first post-order traversal:
    Q.push_back(P);
    do {
      ValuePair QTop = Q.pop_back_val();
      Visited.insert(QTop);

      DEBUG(if (DebugCycleCheck)
              dbgs() << "BBV: cycle check visiting: " << *QTop.first << " <-> "
                     << *QTop.second << "\n");
      DenseMap<ValuePair, std::vector<ValuePair> >::iterator QQ =
        PairableInstUserMap.find(QTop);
      if (QQ == PairableInstUserMap.end())
        continue;

      for (std::vector<ValuePair>::iterator C = QQ->second.begin(),
           CE = QQ->second.end(); C != CE; ++C) {
        if (*C == P) {
          DEBUG(dbgs()
                 << "BBV: rejected to prevent non-trivial cycle formation: "
                 << QTop.first << " <-> " << C->second << "\n");
          return true;
        }

        if (CurrentPairs.count(*C) && !Visited.count(*C))
          Q.push_back(*C);
      }
    } while (!Q.empty());

    return false;
  }

  // This function builds the initial dag of connected pairs with the
  // pair J at the root.
  void BBVectorize::buildInitialDAGFor(
                  DenseMap<Value *, std::vector<Value *> > &CandidatePairs,
                  DenseSet<ValuePair> &CandidatePairsSet,
                  std::vector<Value *> &PairableInsts,
                  DenseMap<ValuePair, std::vector<ValuePair> > &ConnectedPairs,
                  DenseSet<ValuePair> &PairableInstUsers,
                  DenseMap<Value *, Value *> &ChosenPairs,
                  DenseMap<ValuePair, size_t> &DAG, ValuePair J) {
    // Each of these pairs is viewed as the root node of a DAG. The DAG
    // is then walked (depth-first). As this happens, we keep track of
    // the pairs that compose the DAG and the maximum depth of the DAG.
    SmallVector<ValuePairWithDepth, 32> Q;
    // General depth-first post-order traversal:
    Q.push_back(ValuePairWithDepth(J, getDepthFactor(J.first)));
    do {
      ValuePairWithDepth QTop = Q.back();

      // Push each child onto the queue:
      bool MoreChildren = false;
      size_t MaxChildDepth = QTop.second;
      DenseMap<ValuePair, std::vector<ValuePair> >::iterator QQ =
        ConnectedPairs.find(QTop.first);
      if (QQ != ConnectedPairs.end())
        for (std::vector<ValuePair>::iterator k = QQ->second.begin(),
             ke = QQ->second.end(); k != ke; ++k) {
          // Make sure that this child pair is still a candidate:
          if (CandidatePairsSet.count(*k)) {
            DenseMap<ValuePair, size_t>::iterator C = DAG.find(*k);
            if (C == DAG.end()) {
              size_t d = getDepthFactor(k->first);
              Q.push_back(ValuePairWithDepth(*k, QTop.second+d));
              MoreChildren = true;
            } else {
              MaxChildDepth = std::max(MaxChildDepth, C->second);
            }
          }
        }

      if (!MoreChildren) {
        // Record the current pair as part of the DAG:
        DAG.insert(ValuePairWithDepth(QTop.first, MaxChildDepth));
        Q.pop_back();
      }
    } while (!Q.empty());
  }

  // Given some initial dag, prune it by removing conflicting pairs (pairs
  // that cannot be simultaneously chosen for vectorization).
  void BBVectorize::pruneDAGFor(
              DenseMap<Value *, std::vector<Value *> > &CandidatePairs,
              std::vector<Value *> &PairableInsts,
              DenseMap<ValuePair, std::vector<ValuePair> > &ConnectedPairs,
              DenseSet<ValuePair> &PairableInstUsers,
              DenseMap<ValuePair, std::vector<ValuePair> > &PairableInstUserMap,
              DenseSet<VPPair> &PairableInstUserPairSet,
              DenseMap<Value *, Value *> &ChosenPairs,
              DenseMap<ValuePair, size_t> &DAG,
              DenseSet<ValuePair> &PrunedDAG, ValuePair J,
              bool UseCycleCheck) {
    SmallVector<ValuePairWithDepth, 32> Q;
    // General depth-first post-order traversal:
    Q.push_back(ValuePairWithDepth(J, getDepthFactor(J.first)));
    do {
      ValuePairWithDepth QTop = Q.pop_back_val();
      PrunedDAG.insert(QTop.first);

      // Visit each child, pruning as necessary...
      SmallVector<ValuePairWithDepth, 8> BestChildren;
      DenseMap<ValuePair, std::vector<ValuePair> >::iterator QQ =
        ConnectedPairs.find(QTop.first);
      if (QQ == ConnectedPairs.end())
        continue;

      for (std::vector<ValuePair>::iterator K = QQ->second.begin(),
           KE = QQ->second.end(); K != KE; ++K) {
        DenseMap<ValuePair, size_t>::iterator C = DAG.find(*K);
        if (C == DAG.end()) continue;

        // This child is in the DAG, now we need to make sure it is the
        // best of any conflicting children. There could be multiple
        // conflicting children, so first, determine if we're keeping
        // this child, then delete conflicting children as necessary.

        // It is also necessary to guard against pairing-induced
        // dependencies. Consider instructions a .. x .. y .. b
        // such that (a,b) are to be fused and (x,y) are to be fused
        // but a is an input to x and b is an output from y. This
        // means that y cannot be moved after b but x must be moved
        // after b for (a,b) to be fused. In other words, after
        // fusing (a,b) we have y .. a/b .. x where y is an input
        // to a/b and x is an output to a/b: x and y can no longer
        // be legally fused. To prevent this condition, we must
        // make sure that a child pair added to the DAG is not
        // both an input and output of an already-selected pair.

        // Pairing-induced dependencies can also form from more complicated
        // cycles. The pair vs. pair conflicts are easy to check, and so
        // that is done explicitly for "fast rejection", and because for
        // child vs. child conflicts, we may prefer to keep the current
        // pair in preference to the already-selected child.
        DenseSet<ValuePair> CurrentPairs;

        bool CanAdd = true;
        for (SmallVectorImpl<ValuePairWithDepth>::iterator C2
              = BestChildren.begin(), E2 = BestChildren.end();
             C2 != E2; ++C2) {
          if (C2->first.first == C->first.first ||
              C2->first.first == C->first.second ||
              C2->first.second == C->first.first ||
              C2->first.second == C->first.second ||
              pairsConflict(C2->first, C->first, PairableInstUsers,
                            UseCycleCheck ? &PairableInstUserMap : nullptr,
                            UseCycleCheck ? &PairableInstUserPairSet
                                          : nullptr)) {
            if (C2->second >= C->second) {
              CanAdd = false;
              break;
            }

            CurrentPairs.insert(C2->first);
          }
        }
        if (!CanAdd) continue;

        // Even worse, this child could conflict with another node already
        // selected for the DAG. If that is the case, ignore this child.
        for (DenseSet<ValuePair>::iterator T = PrunedDAG.begin(),
             E2 = PrunedDAG.end(); T != E2; ++T) {
          if (T->first == C->first.first ||
              T->first == C->first.second ||
              T->second == C->first.first ||
              T->second == C->first.second ||
              pairsConflict(*T, C->first, PairableInstUsers,
                            UseCycleCheck ? &PairableInstUserMap : nullptr,
                            UseCycleCheck ? &PairableInstUserPairSet
                                          : nullptr)) {
            CanAdd = false;
            break;
          }

          CurrentPairs.insert(*T);
        }
        if (!CanAdd) continue;

        // And check the queue too...
        for (SmallVectorImpl<ValuePairWithDepth>::iterator C2 = Q.begin(),
             E2 = Q.end(); C2 != E2; ++C2) {
          if (C2->first.first == C->first.first ||
              C2->first.first == C->first.second ||
              C2->first.second == C->first.first ||
              C2->first.second == C->first.second ||
              pairsConflict(C2->first, C->first, PairableInstUsers,
                            UseCycleCheck ? &PairableInstUserMap : nullptr,
                            UseCycleCheck ? &PairableInstUserPairSet
                                          : nullptr)) {
            CanAdd = false;
            break;
          }

          CurrentPairs.insert(C2->first);
        }
        if (!CanAdd) continue;

        // Last but not least, check for a conflict with any of the
        // already-chosen pairs.
        for (DenseMap<Value *, Value *>::iterator C2 =
              ChosenPairs.begin(), E2 = ChosenPairs.end();
             C2 != E2; ++C2) {
          if (pairsConflict(*C2, C->first, PairableInstUsers,
                            UseCycleCheck ? &PairableInstUserMap : nullptr,
                            UseCycleCheck ? &PairableInstUserPairSet
                                          : nullptr)) {
            CanAdd = false;
            break;
          }

          CurrentPairs.insert(*C2);
        }
        if (!CanAdd) continue;

        // To check for non-trivial cycles formed by the addition of the
        // current pair we've formed a list of all relevant pairs, now use a
        // graph walk to check for a cycle. We start from the current pair and
        // walk the use dag to see if we again reach the current pair. If we
        // do, then the current pair is rejected.

        // FIXME: It may be more efficient to use a topological-ordering
        // algorithm to improve the cycle check. This should be investigated.
        if (UseCycleCheck &&
            pairWillFormCycle(C->first, PairableInstUserMap, CurrentPairs))
          continue;

        // This child can be added, but we may have chosen it in preference
        // to an already-selected child. Check for this here, and if a
        // conflict is found, then remove the previously-selected child
        // before adding this one in its place.
        for (SmallVectorImpl<ValuePairWithDepth>::iterator C2
              = BestChildren.begin(); C2 != BestChildren.end();) {
          if (C2->first.first == C->first.first ||
              C2->first.first == C->first.second ||
              C2->first.second == C->first.first ||
              C2->first.second == C->first.second ||
              pairsConflict(C2->first, C->first, PairableInstUsers))
            C2 = BestChildren.erase(C2);
          else
            ++C2;
        }

        BestChildren.push_back(ValuePairWithDepth(C->first, C->second));
      }

      for (SmallVectorImpl<ValuePairWithDepth>::iterator C
            = BestChildren.begin(), E2 = BestChildren.end();
           C != E2; ++C) {
        size_t DepthF = getDepthFactor(C->first.first);
        Q.push_back(ValuePairWithDepth(C->first, QTop.second+DepthF));
      }
    } while (!Q.empty());
  }

  // This function finds the best dag of mututally-compatible connected
  // pairs, given the choice of root pairs as an iterator range.
  void BBVectorize::findBestDAGFor(
              DenseMap<Value *, std::vector<Value *> > &CandidatePairs,
              DenseSet<ValuePair> &CandidatePairsSet,
              DenseMap<ValuePair, int> &CandidatePairCostSavings,
              std::vector<Value *> &PairableInsts,
              DenseSet<ValuePair> &FixedOrderPairs,
              DenseMap<VPPair, unsigned> &PairConnectionTypes,
              DenseMap<ValuePair, std::vector<ValuePair> > &ConnectedPairs,
              DenseMap<ValuePair, std::vector<ValuePair> > &ConnectedPairDeps,
              DenseSet<ValuePair> &PairableInstUsers,
              DenseMap<ValuePair, std::vector<ValuePair> > &PairableInstUserMap,
              DenseSet<VPPair> &PairableInstUserPairSet,
              DenseMap<Value *, Value *> &ChosenPairs,
              DenseSet<ValuePair> &BestDAG, size_t &BestMaxDepth,
              int &BestEffSize, Value *II, std::vector<Value *>&JJ,
              bool UseCycleCheck) {
    for (std::vector<Value *>::iterator J = JJ.begin(), JE = JJ.end();
         J != JE; ++J) {
      ValuePair IJ(II, *J);
      if (!CandidatePairsSet.count(IJ))
        continue;

      // Before going any further, make sure that this pair does not
      // conflict with any already-selected pairs (see comment below
      // near the DAG pruning for more details).
      DenseSet<ValuePair> ChosenPairSet;
      bool DoesConflict = false;
      for (DenseMap<Value *, Value *>::iterator C = ChosenPairs.begin(),
           E = ChosenPairs.end(); C != E; ++C) {
        if (pairsConflict(*C, IJ, PairableInstUsers,
                          UseCycleCheck ? &PairableInstUserMap : nullptr,
                          UseCycleCheck ? &PairableInstUserPairSet : nullptr)) {
          DoesConflict = true;
          break;
        }

        ChosenPairSet.insert(*C);
      }
      if (DoesConflict) continue;

      if (UseCycleCheck &&
          pairWillFormCycle(IJ, PairableInstUserMap, ChosenPairSet))
        continue;

      DenseMap<ValuePair, size_t> DAG;
      buildInitialDAGFor(CandidatePairs, CandidatePairsSet,
                          PairableInsts, ConnectedPairs,
                          PairableInstUsers, ChosenPairs, DAG, IJ);

      // Because we'll keep the child with the largest depth, the largest
      // depth is still the same in the unpruned DAG.
      size_t MaxDepth = DAG.lookup(IJ);

      DEBUG(if (DebugPairSelection) dbgs() << "BBV: found DAG for pair {"
                   << *IJ.first << " <-> " << *IJ.second << "} of depth " <<
                   MaxDepth << " and size " << DAG.size() << "\n");

      // At this point the DAG has been constructed, but, may contain
      // contradictory children (meaning that different children of
      // some dag node may be attempting to fuse the same instruction).
      // So now we walk the dag again, in the case of a conflict,
      // keep only the child with the largest depth. To break a tie,
      // favor the first child.

      DenseSet<ValuePair> PrunedDAG;
      pruneDAGFor(CandidatePairs, PairableInsts, ConnectedPairs,
                   PairableInstUsers, PairableInstUserMap,
                   PairableInstUserPairSet,
                   ChosenPairs, DAG, PrunedDAG, IJ, UseCycleCheck);

      int EffSize = 0;
      if (TTI) {
        DenseSet<Value *> PrunedDAGInstrs;
        for (DenseSet<ValuePair>::iterator S = PrunedDAG.begin(),
             E = PrunedDAG.end(); S != E; ++S) {
          PrunedDAGInstrs.insert(S->first);
          PrunedDAGInstrs.insert(S->second);
        }

        // The set of pairs that have already contributed to the total cost.
        DenseSet<ValuePair> IncomingPairs;

        // If the cost model were perfect, this might not be necessary; but we
        // need to make sure that we don't get stuck vectorizing our own
        // shuffle chains.
        bool HasNontrivialInsts = false;

        // The node weights represent the cost savings associated with
        // fusing the pair of instructions.
        for (DenseSet<ValuePair>::iterator S = PrunedDAG.begin(),
             E = PrunedDAG.end(); S != E; ++S) {
          if (!isa<ShuffleVectorInst>(S->first) &&
              !isa<InsertElementInst>(S->first) &&
              !isa<ExtractElementInst>(S->first))
            HasNontrivialInsts = true;

          bool FlipOrder = false;

          if (getDepthFactor(S->first)) {
            int ESContrib = CandidatePairCostSavings.find(*S)->second;
            DEBUG(if (DebugPairSelection) dbgs() << "\tweight {"
                   << *S->first << " <-> " << *S->second << "} = " <<
                   ESContrib << "\n");
            EffSize += ESContrib;
          }

          // The edge weights contribute in a negative sense: they represent
          // the cost of shuffles.
          DenseMap<ValuePair, std::vector<ValuePair> >::iterator SS =
            ConnectedPairDeps.find(*S);
          if (SS != ConnectedPairDeps.end()) {
            unsigned NumDepsDirect = 0, NumDepsSwap = 0;
            for (std::vector<ValuePair>::iterator T = SS->second.begin(),
                 TE = SS->second.end(); T != TE; ++T) {
              VPPair Q(*S, *T);
              if (!PrunedDAG.count(Q.second))
                continue;
              DenseMap<VPPair, unsigned>::iterator R =
                PairConnectionTypes.find(VPPair(Q.second, Q.first));
              assert(R != PairConnectionTypes.end() &&
                     "Cannot find pair connection type");
              if (R->second == PairConnectionDirect)
                ++NumDepsDirect;
              else if (R->second == PairConnectionSwap)
                ++NumDepsSwap;
            }

            // If there are more swaps than direct connections, then
            // the pair order will be flipped during fusion. So the real
            // number of swaps is the minimum number.
            FlipOrder = !FixedOrderPairs.count(*S) &&
              ((NumDepsSwap > NumDepsDirect) ||
                FixedOrderPairs.count(ValuePair(S->second, S->first)));

            for (std::vector<ValuePair>::iterator T = SS->second.begin(),
                 TE = SS->second.end(); T != TE; ++T) {
              VPPair Q(*S, *T);
              if (!PrunedDAG.count(Q.second))
                continue;
              DenseMap<VPPair, unsigned>::iterator R =
                PairConnectionTypes.find(VPPair(Q.second, Q.first));
              assert(R != PairConnectionTypes.end() &&
                     "Cannot find pair connection type");
              Type *Ty1 = Q.second.first->getType(),
                   *Ty2 = Q.second.second->getType();
              Type *VTy = getVecTypeForPair(Ty1, Ty2);
              if ((R->second == PairConnectionDirect && FlipOrder) ||
                  (R->second == PairConnectionSwap && !FlipOrder)  ||
                  R->second == PairConnectionSplat) {
                int ESContrib = (int) getInstrCost(Instruction::ShuffleVector,
                                                   VTy, VTy);

                if (VTy->getVectorNumElements() == 2) {
                  if (R->second == PairConnectionSplat)
                    ESContrib = std::min(ESContrib, (int) TTI->getShuffleCost(
                      TargetTransformInfo::SK_Broadcast, VTy));
                  else
                    ESContrib = std::min(ESContrib, (int) TTI->getShuffleCost(
                      TargetTransformInfo::SK_Reverse, VTy));
                }

                DEBUG(if (DebugPairSelection) dbgs() << "\tcost {" <<
                  *Q.second.first << " <-> " << *Q.second.second <<
                    "} -> {" <<
                  *S->first << " <-> " << *S->second << "} = " <<
                   ESContrib << "\n");
                EffSize -= ESContrib;
              }
            }
          }

          // Compute the cost of outgoing edges. We assume that edges outgoing
          // to shuffles, inserts or extracts can be merged, and so contribute
          // no additional cost.
          if (!S->first->getType()->isVoidTy()) {
            Type *Ty1 = S->first->getType(),
                 *Ty2 = S->second->getType();
            Type *VTy = getVecTypeForPair(Ty1, Ty2);

            bool NeedsExtraction = false;
            for (User *U : S->first->users()) {
              if (ShuffleVectorInst *SI = dyn_cast<ShuffleVectorInst>(U)) {
                // Shuffle can be folded if it has no other input
                if (isa<UndefValue>(SI->getOperand(1)))
                  continue;
              }
              if (isa<ExtractElementInst>(U))
                continue;
              if (PrunedDAGInstrs.count(U))
                continue;
              NeedsExtraction = true;
              break;
            }

            if (NeedsExtraction) {
              int ESContrib;
              if (Ty1->isVectorTy()) {
                ESContrib = (int) getInstrCost(Instruction::ShuffleVector,
                                               Ty1, VTy);
                ESContrib = std::min(ESContrib, (int) TTI->getShuffleCost(
                  TargetTransformInfo::SK_ExtractSubvector, VTy, 0, Ty1));
              } else
                ESContrib = (int) TTI->getVectorInstrCost(
                                    Instruction::ExtractElement, VTy, 0);

              DEBUG(if (DebugPairSelection) dbgs() << "\tcost {" <<
                *S->first << "} = " << ESContrib << "\n");
              EffSize -= ESContrib;
            }

            NeedsExtraction = false;
            for (User *U : S->second->users()) {
              if (ShuffleVectorInst *SI = dyn_cast<ShuffleVectorInst>(U)) {
                // Shuffle can be folded if it has no other input
                if (isa<UndefValue>(SI->getOperand(1)))
                  continue;
              }
              if (isa<ExtractElementInst>(U))
                continue;
              if (PrunedDAGInstrs.count(U))
                continue;
              NeedsExtraction = true;
              break;
            }

            if (NeedsExtraction) {
              int ESContrib;
              if (Ty2->isVectorTy()) {
                ESContrib = (int) getInstrCost(Instruction::ShuffleVector,
                                               Ty2, VTy);
                ESContrib = std::min(ESContrib, (int) TTI->getShuffleCost(
                  TargetTransformInfo::SK_ExtractSubvector, VTy,
                  Ty1->isVectorTy() ? Ty1->getVectorNumElements() : 1, Ty2));
              } else
                ESContrib = (int) TTI->getVectorInstrCost(
                                    Instruction::ExtractElement, VTy, 1);
              DEBUG(if (DebugPairSelection) dbgs() << "\tcost {" <<
                *S->second << "} = " << ESContrib << "\n");
              EffSize -= ESContrib;
            }
          }

          // Compute the cost of incoming edges.
          if (!isa<LoadInst>(S->first) && !isa<StoreInst>(S->first)) {
            Instruction *S1 = cast<Instruction>(S->first),
                        *S2 = cast<Instruction>(S->second);
            for (unsigned o = 0; o < S1->getNumOperands(); ++o) {
              Value *O1 = S1->getOperand(o), *O2 = S2->getOperand(o);

              // Combining constants into vector constants (or small vector
              // constants into larger ones are assumed free).
              if (isa<Constant>(O1) && isa<Constant>(O2))
                continue;

              if (FlipOrder)
                std::swap(O1, O2);

              ValuePair VP  = ValuePair(O1, O2);
              ValuePair VPR = ValuePair(O2, O1);

              // Internal edges are not handled here.
              if (PrunedDAG.count(VP) || PrunedDAG.count(VPR))
                continue;

              Type *Ty1 = O1->getType(),
                   *Ty2 = O2->getType();
              Type *VTy = getVecTypeForPair(Ty1, Ty2);

              // Combining vector operations of the same type is also assumed
              // folded with other operations.
              if (Ty1 == Ty2) {
                // If both are insert elements, then both can be widened.
                InsertElementInst *IEO1 = dyn_cast<InsertElementInst>(O1),
                                  *IEO2 = dyn_cast<InsertElementInst>(O2);
                if (IEO1 && IEO2 && isPureIEChain(IEO1) && isPureIEChain(IEO2))
                  continue;
                // If both are extract elements, and both have the same input
                // type, then they can be replaced with a shuffle
                ExtractElementInst *EIO1 = dyn_cast<ExtractElementInst>(O1),
                                   *EIO2 = dyn_cast<ExtractElementInst>(O2);
                if (EIO1 && EIO2 &&
                    EIO1->getOperand(0)->getType() ==
                      EIO2->getOperand(0)->getType())
                  continue;
                // If both are a shuffle with equal operand types and only two
                // unqiue operands, then they can be replaced with a single
                // shuffle
                ShuffleVectorInst *SIO1 = dyn_cast<ShuffleVectorInst>(O1),
                                  *SIO2 = dyn_cast<ShuffleVectorInst>(O2);
                if (SIO1 && SIO2 &&
                    SIO1->getOperand(0)->getType() ==
                      SIO2->getOperand(0)->getType()) {
                  SmallSet<Value *, 4> SIOps;
                  SIOps.insert(SIO1->getOperand(0));
                  SIOps.insert(SIO1->getOperand(1));
                  SIOps.insert(SIO2->getOperand(0));
                  SIOps.insert(SIO2->getOperand(1));
                  if (SIOps.size() <= 2)
                    continue;
                }
              }

              int ESContrib;
              // This pair has already been formed.
              if (IncomingPairs.count(VP)) {
                continue;
              } else if (IncomingPairs.count(VPR)) {
                ESContrib = (int) getInstrCost(Instruction::ShuffleVector,
                                               VTy, VTy);

                if (VTy->getVectorNumElements() == 2)
                  ESContrib = std::min(ESContrib, (int) TTI->getShuffleCost(
                    TargetTransformInfo::SK_Reverse, VTy));
              } else if (!Ty1->isVectorTy() && !Ty2->isVectorTy()) {
                ESContrib = (int) TTI->getVectorInstrCost(
                                    Instruction::InsertElement, VTy, 0);
                ESContrib += (int) TTI->getVectorInstrCost(
                                     Instruction::InsertElement, VTy, 1);
              } else if (!Ty1->isVectorTy()) {
                // O1 needs to be inserted into a vector of size O2, and then
                // both need to be shuffled together.
                ESContrib = (int) TTI->getVectorInstrCost(
                                    Instruction::InsertElement, Ty2, 0);
                ESContrib += (int) getInstrCost(Instruction::ShuffleVector,
                                                VTy, Ty2);
              } else if (!Ty2->isVectorTy()) {
                // O2 needs to be inserted into a vector of size O1, and then
                // both need to be shuffled together.
                ESContrib = (int) TTI->getVectorInstrCost(
                                    Instruction::InsertElement, Ty1, 0);
                ESContrib += (int) getInstrCost(Instruction::ShuffleVector,
                                                VTy, Ty1);
              } else {
                Type *TyBig = Ty1, *TySmall = Ty2;
                if (Ty2->getVectorNumElements() > Ty1->getVectorNumElements())
                  std::swap(TyBig, TySmall);

                ESContrib = (int) getInstrCost(Instruction::ShuffleVector,
                                               VTy, TyBig);
                if (TyBig != TySmall)
                  ESContrib += (int) getInstrCost(Instruction::ShuffleVector,
                                                  TyBig, TySmall);
              }

              DEBUG(if (DebugPairSelection) dbgs() << "\tcost {"
                     << *O1 << " <-> " << *O2 << "} = " <<
                     ESContrib << "\n");
              EffSize -= ESContrib;
              IncomingPairs.insert(VP);
            }
          }
        }

        if (!HasNontrivialInsts) {
          DEBUG(if (DebugPairSelection) dbgs() <<
                "\tNo non-trivial instructions in DAG;"
                " override to zero effective size\n");
          EffSize = 0;
        }
      } else {
        for (DenseSet<ValuePair>::iterator S = PrunedDAG.begin(),
             E = PrunedDAG.end(); S != E; ++S)
          EffSize += (int) getDepthFactor(S->first);
      }

      DEBUG(if (DebugPairSelection)
             dbgs() << "BBV: found pruned DAG for pair {"
             << *IJ.first << " <-> " << *IJ.second << "} of depth " <<
             MaxDepth << " and size " << PrunedDAG.size() <<
            " (effective size: " << EffSize << ")\n");
      if (((TTI && !UseChainDepthWithTI) ||
            MaxDepth >= Config.ReqChainDepth) &&
          EffSize > 0 && EffSize > BestEffSize) {
        BestMaxDepth = MaxDepth;
        BestEffSize = EffSize;
        BestDAG = PrunedDAG;
      }
    }
  }

  // Given the list of candidate pairs, this function selects those
  // that will be fused into vector instructions.
  void BBVectorize::choosePairs(
                DenseMap<Value *, std::vector<Value *> > &CandidatePairs,
                DenseSet<ValuePair> &CandidatePairsSet,
                DenseMap<ValuePair, int> &CandidatePairCostSavings,
                std::vector<Value *> &PairableInsts,
                DenseSet<ValuePair> &FixedOrderPairs,
                DenseMap<VPPair, unsigned> &PairConnectionTypes,
                DenseMap<ValuePair, std::vector<ValuePair> > &ConnectedPairs,
                DenseMap<ValuePair, std::vector<ValuePair> > &ConnectedPairDeps,
                DenseSet<ValuePair> &PairableInstUsers,
                DenseMap<Value *, Value *>& ChosenPairs) {
    bool UseCycleCheck =
     CandidatePairsSet.size() <= Config.MaxCandPairsForCycleCheck;

    DenseMap<Value *, std::vector<Value *> > CandidatePairs2;
    for (DenseSet<ValuePair>::iterator I = CandidatePairsSet.begin(),
         E = CandidatePairsSet.end(); I != E; ++I) {
      std::vector<Value *> &JJ = CandidatePairs2[I->second];
      if (JJ.empty()) JJ.reserve(32);
      JJ.push_back(I->first);
    }

    DenseMap<ValuePair, std::vector<ValuePair> > PairableInstUserMap;
    DenseSet<VPPair> PairableInstUserPairSet;
    for (std::vector<Value *>::iterator I = PairableInsts.begin(),
         E = PairableInsts.end(); I != E; ++I) {
      // The number of possible pairings for this variable:
      size_t NumChoices = CandidatePairs.lookup(*I).size();
      if (!NumChoices) continue;

      std::vector<Value *> &JJ = CandidatePairs[*I];

      // The best pair to choose and its dag:
      size_t BestMaxDepth = 0;
      int BestEffSize = 0;
      DenseSet<ValuePair> BestDAG;
      findBestDAGFor(CandidatePairs, CandidatePairsSet,
                      CandidatePairCostSavings,
                      PairableInsts, FixedOrderPairs, PairConnectionTypes,
                      ConnectedPairs, ConnectedPairDeps,
                      PairableInstUsers, PairableInstUserMap,
                      PairableInstUserPairSet, ChosenPairs,
                      BestDAG, BestMaxDepth, BestEffSize, *I, JJ,
                      UseCycleCheck);

      if (BestDAG.empty())
        continue;

      // A dag has been chosen (or not) at this point. If no dag was
      // chosen, then this instruction, I, cannot be paired (and is no longer
      // considered).

      DEBUG(dbgs() << "BBV: selected pairs in the best DAG for: "
                   << *cast<Instruction>(*I) << "\n");

      for (DenseSet<ValuePair>::iterator S = BestDAG.begin(),
           SE2 = BestDAG.end(); S != SE2; ++S) {
        // Insert the members of this dag into the list of chosen pairs.
        ChosenPairs.insert(ValuePair(S->first, S->second));
        DEBUG(dbgs() << "BBV: selected pair: " << *S->first << " <-> " <<
               *S->second << "\n");

        // Remove all candidate pairs that have values in the chosen dag.
        std::vector<Value *> &KK = CandidatePairs[S->first];
        for (std::vector<Value *>::iterator K = KK.begin(), KE = KK.end();
             K != KE; ++K) {
          if (*K == S->second)
            continue;

          CandidatePairsSet.erase(ValuePair(S->first, *K));
        }

        std::vector<Value *> &LL = CandidatePairs2[S->second];
        for (std::vector<Value *>::iterator L = LL.begin(), LE = LL.end();
             L != LE; ++L) {
          if (*L == S->first)
            continue;

          CandidatePairsSet.erase(ValuePair(*L, S->second));
        }

        std::vector<Value *> &MM = CandidatePairs[S->second];
        for (std::vector<Value *>::iterator M = MM.begin(), ME = MM.end();
             M != ME; ++M) {
          assert(*M != S->first && "Flipped pair in candidate list?");
          CandidatePairsSet.erase(ValuePair(S->second, *M));
        }

        std::vector<Value *> &NN = CandidatePairs2[S->first];
        for (std::vector<Value *>::iterator N = NN.begin(), NE = NN.end();
             N != NE; ++N) {
          assert(*N != S->second && "Flipped pair in candidate list?");
          CandidatePairsSet.erase(ValuePair(*N, S->first));
        }
      }
    }

    DEBUG(dbgs() << "BBV: selected " << ChosenPairs.size() << " pairs.\n");
  }

  std::string getReplacementName(Instruction *I, bool IsInput, unsigned o,
                     unsigned n = 0) {
    if (!I->hasName())
      return "";

    return (I->getName() + (IsInput ? ".v.i" : ".v.r") + utostr(o) +
             (n > 0 ? "." + utostr(n) : "")).str();
  }

  // Returns the value that is to be used as the pointer input to the vector
  // instruction that fuses I with J.
  Value *BBVectorize::getReplacementPointerInput(LLVMContext& Context,
                     Instruction *I, Instruction *J, unsigned o) {
    Value *IPtr, *JPtr;
    unsigned IAlignment, JAlignment, IAddressSpace, JAddressSpace;
    int64_t OffsetInElmts;

    // Note: the analysis might fail here, that is why the pair order has
    // been precomputed (OffsetInElmts must be unused here).
    (void) getPairPtrInfo(I, J, IPtr, JPtr, IAlignment, JAlignment,
                          IAddressSpace, JAddressSpace,
                          OffsetInElmts, false);

    // The pointer value is taken to be the one with the lowest offset.
    Value *VPtr = IPtr;

    Type *ArgTypeI = IPtr->getType()->getPointerElementType();
    Type *ArgTypeJ = JPtr->getType()->getPointerElementType();
    Type *VArgType = getVecTypeForPair(ArgTypeI, ArgTypeJ);
    Type *VArgPtrType
      = PointerType::get(VArgType,
                         IPtr->getType()->getPointerAddressSpace());
    return new BitCastInst(VPtr, VArgPtrType, getReplacementName(I, true, o),
                        /* insert before */ I);
  }

  void BBVectorize::fillNewShuffleMask(LLVMContext& Context, Instruction *J,
                     unsigned MaskOffset, unsigned NumInElem,
                     unsigned NumInElem1, unsigned IdxOffset,
                     std::vector<Constant*> &Mask) {
    unsigned NumElem1 = J->getType()->getVectorNumElements();
    for (unsigned v = 0; v < NumElem1; ++v) {
      int m = cast<ShuffleVectorInst>(J)->getMaskValue(v);
      if (m < 0) {
        Mask[v+MaskOffset] = UndefValue::get(Type::getInt32Ty(Context));
      } else {
        unsigned mm = m + (int) IdxOffset;
        if (m >= (int) NumInElem1)
          mm += (int) NumInElem;

        Mask[v+MaskOffset] =
          ConstantInt::get(Type::getInt32Ty(Context), mm);
      }
    }
  }

  // Returns the value that is to be used as the vector-shuffle mask to the
  // vector instruction that fuses I with J.
  Value *BBVectorize::getReplacementShuffleMask(LLVMContext& Context,
                     Instruction *I, Instruction *J) {
    // This is the shuffle mask. We need to append the second
    // mask to the first, and the numbers need to be adjusted.

    Type *ArgTypeI = I->getType();
    Type *ArgTypeJ = J->getType();
    Type *VArgType = getVecTypeForPair(ArgTypeI, ArgTypeJ);

    unsigned NumElemI = ArgTypeI->getVectorNumElements();

    // Get the total number of elements in the fused vector type.
    // By definition, this must equal the number of elements in
    // the final mask.
    unsigned NumElem = VArgType->getVectorNumElements();
    std::vector<Constant*> Mask(NumElem);

    Type *OpTypeI = I->getOperand(0)->getType();
    unsigned NumInElemI = OpTypeI->getVectorNumElements();
    Type *OpTypeJ = J->getOperand(0)->getType();
    unsigned NumInElemJ = OpTypeJ->getVectorNumElements();

    // The fused vector will be:
    // -----------------------------------------------------
    // | NumInElemI | NumInElemJ | NumInElemI | NumInElemJ |
    // -----------------------------------------------------
    // from which we'll extract NumElem total elements (where the first NumElemI
    // of them come from the mask in I and the remainder come from the mask
    // in J.

    // For the mask from the first pair...
    fillNewShuffleMask(Context, I, 0,        NumInElemJ, NumInElemI,
                       0,          Mask);

    // For the mask from the second pair...
    fillNewShuffleMask(Context, J, NumElemI, NumInElemI, NumInElemJ,
                       NumInElemI, Mask);

    return ConstantVector::get(Mask);
  }

  bool BBVectorize::expandIEChain(LLVMContext& Context, Instruction *I,
                                  Instruction *J, unsigned o, Value *&LOp,
                                  unsigned numElemL,
                                  Type *ArgTypeL, Type *ArgTypeH,
                                  bool IBeforeJ, unsigned IdxOff) {
    bool ExpandedIEChain = false;
    if (InsertElementInst *LIE = dyn_cast<InsertElementInst>(LOp)) {
      // If we have a pure insertelement chain, then this can be rewritten
      // into a chain that directly builds the larger type.
      if (isPureIEChain(LIE)) {
        SmallVector<Value *, 8> VectElemts(numElemL,
          UndefValue::get(ArgTypeL->getScalarType()));
        InsertElementInst *LIENext = LIE;
        do {
          unsigned Idx =
            cast<ConstantInt>(LIENext->getOperand(2))->getSExtValue();
          VectElemts[Idx] = LIENext->getOperand(1);
        } while ((LIENext =
                   dyn_cast<InsertElementInst>(LIENext->getOperand(0))));

        LIENext = nullptr;
        Value *LIEPrev = UndefValue::get(ArgTypeH);
        for (unsigned i = 0; i < numElemL; ++i) {
          if (isa<UndefValue>(VectElemts[i])) continue;
          LIENext = InsertElementInst::Create(LIEPrev, VectElemts[i],
                             ConstantInt::get(Type::getInt32Ty(Context),
                                              i + IdxOff),
                             getReplacementName(IBeforeJ ? I : J,
                                                true, o, i+1));
          LIENext->insertBefore(IBeforeJ ? J : I);
          LIEPrev = LIENext;
        }

        LOp = LIENext ? (Value*) LIENext : UndefValue::get(ArgTypeH);
        ExpandedIEChain = true;
      }
    }

    return ExpandedIEChain;
  }

  static unsigned getNumScalarElements(Type *Ty) {
    if (VectorType *VecTy = dyn_cast<VectorType>(Ty))
      return VecTy->getNumElements();
    return 1;
  }

  // Returns the value to be used as the specified operand of the vector
  // instruction that fuses I with J.
  Value *BBVectorize::getReplacementInput(LLVMContext& Context, Instruction *I,
                     Instruction *J, unsigned o, bool IBeforeJ) {
    Value *CV0 = ConstantInt::get(Type::getInt32Ty(Context), 0);
    Value *CV1 = ConstantInt::get(Type::getInt32Ty(Context), 1);

    // Compute the fused vector type for this operand
    Type *ArgTypeI = I->getOperand(o)->getType();
    Type *ArgTypeJ = J->getOperand(o)->getType();
    VectorType *VArgType = getVecTypeForPair(ArgTypeI, ArgTypeJ);

    Instruction *L = I, *H = J;
    Type *ArgTypeL = ArgTypeI, *ArgTypeH = ArgTypeJ;

    unsigned numElemL = getNumScalarElements(ArgTypeL);
    unsigned numElemH = getNumScalarElements(ArgTypeH);

    Value *LOp = L->getOperand(o);
    Value *HOp = H->getOperand(o);
    unsigned numElem = VArgType->getNumElements();

    // First, we check if we can reuse the "original" vector outputs (if these
    // exist). We might need a shuffle.
    ExtractElementInst *LEE = dyn_cast<ExtractElementInst>(LOp);
    ExtractElementInst *HEE = dyn_cast<ExtractElementInst>(HOp);
    ShuffleVectorInst *LSV = dyn_cast<ShuffleVectorInst>(LOp);
    ShuffleVectorInst *HSV = dyn_cast<ShuffleVectorInst>(HOp);

    // FIXME: If we're fusing shuffle instructions, then we can't apply this
    // optimization. The input vectors to the shuffle might be a different
    // length from the shuffle outputs. Unfortunately, the replacement
    // shuffle mask has already been formed, and the mask entries are sensitive
    // to the sizes of the inputs.
    bool IsSizeChangeShuffle =
      isa<ShuffleVectorInst>(L) &&
        (LOp->getType() != L->getType() || HOp->getType() != H->getType());

    if ((LEE || LSV) && (HEE || HSV) && !IsSizeChangeShuffle) {
      // We can have at most two unique vector inputs.
      bool CanUseInputs = true;
      Value *I1, *I2 = nullptr;
      if (LEE) {
        I1 = LEE->getOperand(0);
      } else {
        I1 = LSV->getOperand(0);
        I2 = LSV->getOperand(1);
        if (I2 == I1 || isa<UndefValue>(I2))
          I2 = nullptr;
      }
  
      if (HEE) {
        Value *I3 = HEE->getOperand(0);
        if (!I2 && I3 != I1)
          I2 = I3;
        else if (I3 != I1 && I3 != I2)
          CanUseInputs = false;
      } else {
        Value *I3 = HSV->getOperand(0);
        if (!I2 && I3 != I1)
          I2 = I3;
        else if (I3 != I1 && I3 != I2)
          CanUseInputs = false;

        if (CanUseInputs) {
          Value *I4 = HSV->getOperand(1);
          if (!isa<UndefValue>(I4)) {
            if (!I2 && I4 != I1)
              I2 = I4;
            else if (I4 != I1 && I4 != I2)
              CanUseInputs = false;
          }
        }
      }

      if (CanUseInputs) {
        unsigned LOpElem =
          cast<Instruction>(LOp)->getOperand(0)->getType()
            ->getVectorNumElements();

        unsigned HOpElem =
          cast<Instruction>(HOp)->getOperand(0)->getType()
            ->getVectorNumElements();

        // We have one or two input vectors. We need to map each index of the
        // operands to the index of the original vector.
        SmallVector<std::pair<int, int>, 8>  II(numElem);
        for (unsigned i = 0; i < numElemL; ++i) {
          int Idx, INum;
          if (LEE) {
            Idx =
              cast<ConstantInt>(LEE->getOperand(1))->getSExtValue();
            INum = LEE->getOperand(0) == I1 ? 0 : 1;
          } else {
            Idx = LSV->getMaskValue(i);
            if (Idx < (int) LOpElem) {
              INum = LSV->getOperand(0) == I1 ? 0 : 1;
            } else {
              Idx -= LOpElem;
              INum = LSV->getOperand(1) == I1 ? 0 : 1;
            }
          }

          II[i] = std::pair<int, int>(Idx, INum);
        }
        for (unsigned i = 0; i < numElemH; ++i) {
          int Idx, INum;
          if (HEE) {
            Idx =
              cast<ConstantInt>(HEE->getOperand(1))->getSExtValue();
            INum = HEE->getOperand(0) == I1 ? 0 : 1;
          } else {
            Idx = HSV->getMaskValue(i);
            if (Idx < (int) HOpElem) {
              INum = HSV->getOperand(0) == I1 ? 0 : 1;
            } else {
              Idx -= HOpElem;
              INum = HSV->getOperand(1) == I1 ? 0 : 1;
            }
          }

          II[i + numElemL] = std::pair<int, int>(Idx, INum);
        }

        // We now have an array which tells us from which index of which
        // input vector each element of the operand comes.
        VectorType *I1T = cast<VectorType>(I1->getType());
        unsigned I1Elem = I1T->getNumElements();

        if (!I2) {
          // In this case there is only one underlying vector input. Check for
          // the trivial case where we can use the input directly.
          if (I1Elem == numElem) {
            bool ElemInOrder = true;
            for (unsigned i = 0; i < numElem; ++i) {
              if (II[i].first != (int) i && II[i].first != -1) {
                ElemInOrder = false;
                break;
              }
            }

            if (ElemInOrder)
              return I1;
          }

          // A shuffle is needed.
          std::vector<Constant *> Mask(numElem);
          for (unsigned i = 0; i < numElem; ++i) {
            int Idx = II[i].first;
            if (Idx == -1)
              Mask[i] = UndefValue::get(Type::getInt32Ty(Context));
            else
              Mask[i] = ConstantInt::get(Type::getInt32Ty(Context), Idx);
          }

          Instruction *S =
            new ShuffleVectorInst(I1, UndefValue::get(I1T),
                                  ConstantVector::get(Mask),
                                  getReplacementName(IBeforeJ ? I : J,
                                                     true, o));
          S->insertBefore(IBeforeJ ? J : I);
          return S;
        }

        VectorType *I2T = cast<VectorType>(I2->getType());
        unsigned I2Elem = I2T->getNumElements();

        // This input comes from two distinct vectors. The first step is to
        // make sure that both vectors are the same length. If not, the
        // smaller one will need to grow before they can be shuffled together.
        if (I1Elem < I2Elem) {
          std::vector<Constant *> Mask(I2Elem);
          unsigned v = 0;
          for (; v < I1Elem; ++v)
            Mask[v] = ConstantInt::get(Type::getInt32Ty(Context), v);
          for (; v < I2Elem; ++v)
            Mask[v] = UndefValue::get(Type::getInt32Ty(Context));

          Instruction *NewI1 =
            new ShuffleVectorInst(I1, UndefValue::get(I1T),
                                  ConstantVector::get(Mask),
                                  getReplacementName(IBeforeJ ? I : J,
                                                     true, o, 1));
          NewI1->insertBefore(IBeforeJ ? J : I);
          I1 = NewI1;
          I1Elem = I2Elem;
        } else if (I1Elem > I2Elem) {
          std::vector<Constant *> Mask(I1Elem);
          unsigned v = 0;
          for (; v < I2Elem; ++v)
            Mask[v] = ConstantInt::get(Type::getInt32Ty(Context), v);
          for (; v < I1Elem; ++v)
            Mask[v] = UndefValue::get(Type::getInt32Ty(Context));

          Instruction *NewI2 =
            new ShuffleVectorInst(I2, UndefValue::get(I2T),
                                  ConstantVector::get(Mask),
                                  getReplacementName(IBeforeJ ? I : J,
                                                     true, o, 1));
          NewI2->insertBefore(IBeforeJ ? J : I);
          I2 = NewI2;
        }

        // Now that both I1 and I2 are the same length we can shuffle them
        // together (and use the result).
        std::vector<Constant *> Mask(numElem);
        for (unsigned v = 0; v < numElem; ++v) {
          if (II[v].first == -1) {
            Mask[v] = UndefValue::get(Type::getInt32Ty(Context));
          } else {
            int Idx = II[v].first + II[v].second * I1Elem;
            Mask[v] = ConstantInt::get(Type::getInt32Ty(Context), Idx);
          }
        }

        Instruction *NewOp =
          new ShuffleVectorInst(I1, I2, ConstantVector::get(Mask),
                                getReplacementName(IBeforeJ ? I : J, true, o));
        NewOp->insertBefore(IBeforeJ ? J : I);
        return NewOp;
      }
    }

    Type *ArgType = ArgTypeL;
    if (numElemL < numElemH) {
      if (numElemL == 1 && expandIEChain(Context, I, J, o, HOp, numElemH,
                                         ArgTypeL, VArgType, IBeforeJ, 1)) {
        // This is another short-circuit case: we're combining a scalar into
        // a vector that is formed by an IE chain. We've just expanded the IE
        // chain, now insert the scalar and we're done.

        Instruction *S = InsertElementInst::Create(HOp, LOp, CV0,
                           getReplacementName(IBeforeJ ? I : J, true, o));
        S->insertBefore(IBeforeJ ? J : I);
        return S;
      } else if (!expandIEChain(Context, I, J, o, LOp, numElemL, ArgTypeL,
                                ArgTypeH, IBeforeJ)) {
        // The two vector inputs to the shuffle must be the same length,
        // so extend the smaller vector to be the same length as the larger one.
        Instruction *NLOp;
        if (numElemL > 1) {
  
          std::vector<Constant *> Mask(numElemH);
          unsigned v = 0;
          for (; v < numElemL; ++v)
            Mask[v] = ConstantInt::get(Type::getInt32Ty(Context), v);
          for (; v < numElemH; ++v)
            Mask[v] = UndefValue::get(Type::getInt32Ty(Context));
    
          NLOp = new ShuffleVectorInst(LOp, UndefValue::get(ArgTypeL),
                                       ConstantVector::get(Mask),
                                       getReplacementName(IBeforeJ ? I : J,
                                                          true, o, 1));
        } else {
          NLOp = InsertElementInst::Create(UndefValue::get(ArgTypeH), LOp, CV0,
                                           getReplacementName(IBeforeJ ? I : J,
                                                              true, o, 1));
        }
  
        NLOp->insertBefore(IBeforeJ ? J : I);
        LOp = NLOp;
      }

      ArgType = ArgTypeH;
    } else if (numElemL > numElemH) {
      if (numElemH == 1 && expandIEChain(Context, I, J, o, LOp, numElemL,
                                         ArgTypeH, VArgType, IBeforeJ)) {
        Instruction *S =
          InsertElementInst::Create(LOp, HOp, 
                                    ConstantInt::get(Type::getInt32Ty(Context),
                                                     numElemL),
                                    getReplacementName(IBeforeJ ? I : J,
                                                       true, o));
        S->insertBefore(IBeforeJ ? J : I);
        return S;
      } else if (!expandIEChain(Context, I, J, o, HOp, numElemH, ArgTypeH,
                                ArgTypeL, IBeforeJ)) {
        Instruction *NHOp;
        if (numElemH > 1) {
          std::vector<Constant *> Mask(numElemL);
          unsigned v = 0;
          for (; v < numElemH; ++v)
            Mask[v] = ConstantInt::get(Type::getInt32Ty(Context), v);
          for (; v < numElemL; ++v)
            Mask[v] = UndefValue::get(Type::getInt32Ty(Context));
    
          NHOp = new ShuffleVectorInst(HOp, UndefValue::get(ArgTypeH),
                                       ConstantVector::get(Mask),
                                       getReplacementName(IBeforeJ ? I : J,
                                                          true, o, 1));
        } else {
          NHOp = InsertElementInst::Create(UndefValue::get(ArgTypeL), HOp, CV0,
                                           getReplacementName(IBeforeJ ? I : J,
                                                              true, o, 1));
        }

        NHOp->insertBefore(IBeforeJ ? J : I);
        HOp = NHOp;
      }
    }

    if (ArgType->isVectorTy()) {
      unsigned numElem = VArgType->getVectorNumElements();
      std::vector<Constant*> Mask(numElem);
      for (unsigned v = 0; v < numElem; ++v) {
        unsigned Idx = v;
        // If the low vector was expanded, we need to skip the extra
        // undefined entries.
        if (v >= numElemL && numElemH > numElemL)
          Idx += (numElemH - numElemL);
        Mask[v] = ConstantInt::get(Type::getInt32Ty(Context), Idx);
      }

      Instruction *BV = new ShuffleVectorInst(LOp, HOp,
                          ConstantVector::get(Mask),
                          getReplacementName(IBeforeJ ? I : J, true, o));
      BV->insertBefore(IBeforeJ ? J : I);
      return BV;
    }

    Instruction *BV1 = InsertElementInst::Create(
                                          UndefValue::get(VArgType), LOp, CV0,
                                          getReplacementName(IBeforeJ ? I : J,
                                                             true, o, 1));
    BV1->insertBefore(IBeforeJ ? J : I);
    Instruction *BV2 = InsertElementInst::Create(BV1, HOp, CV1,
                                          getReplacementName(IBeforeJ ? I : J,
                                                             true, o, 2));
    BV2->insertBefore(IBeforeJ ? J : I);
    return BV2;
  }

  // This function creates an array of values that will be used as the inputs
  // to the vector instruction that fuses I with J.
  void BBVectorize::getReplacementInputsForPair(LLVMContext& Context,
                     Instruction *I, Instruction *J,
                     SmallVectorImpl<Value *> &ReplacedOperands,
                     bool IBeforeJ) {
    unsigned NumOperands = I->getNumOperands();

    for (unsigned p = 0, o = NumOperands-1; p < NumOperands; ++p, --o) {
      // Iterate backward so that we look at the store pointer
      // first and know whether or not we need to flip the inputs.

      if (isa<LoadInst>(I) || (o == 1 && isa<StoreInst>(I))) {
        // This is the pointer for a load/store instruction.
        ReplacedOperands[o] = getReplacementPointerInput(Context, I, J, o);
        continue;
      } else if (isa<CallInst>(I)) {
        Function *F = cast<CallInst>(I)->getCalledFunction();
        Intrinsic::ID IID = F->getIntrinsicID();
        if (o == NumOperands-1) {
          BasicBlock &BB = *I->getParent();

          Module *M = BB.getParent()->getParent();
          Type *ArgTypeI = I->getType();
          Type *ArgTypeJ = J->getType();
          Type *VArgType = getVecTypeForPair(ArgTypeI, ArgTypeJ);

          ReplacedOperands[o] = Intrinsic::getDeclaration(M, IID, VArgType);
          continue;
        } else if ((IID == Intrinsic::powi || IID == Intrinsic::ctlz ||
                    IID == Intrinsic::cttz) && o == 1) {
          // The second argument of powi/ctlz/cttz is a single integer/constant
          // and we've already checked that both arguments are equal.
          // As a result, we just keep I's second argument.
          ReplacedOperands[o] = I->getOperand(o);
          continue;
        }
      } else if (isa<ShuffleVectorInst>(I) && o == NumOperands-1) {
        ReplacedOperands[o] = getReplacementShuffleMask(Context, I, J);
        continue;
      }

      ReplacedOperands[o] = getReplacementInput(Context, I, J, o, IBeforeJ);
    }
  }

  // This function creates two values that represent the outputs of the
  // original I and J instructions. These are generally vector shuffles
  // or extracts. In many cases, these will end up being unused and, thus,
  // eliminated by later passes.
  void BBVectorize::replaceOutputsOfPair(LLVMContext& Context, Instruction *I,
                     Instruction *J, Instruction *K,
                     Instruction *&InsertionPt,
                     Instruction *&K1, Instruction *&K2) {
    if (isa<StoreInst>(I)) {
      AA->replaceWithNewValue(I, K);
      AA->replaceWithNewValue(J, K);
    } else {
      Type *IType = I->getType();
      Type *JType = J->getType();

      VectorType *VType = getVecTypeForPair(IType, JType);
      unsigned numElem = VType->getNumElements();

      unsigned numElemI = getNumScalarElements(IType);
      unsigned numElemJ = getNumScalarElements(JType);

      if (IType->isVectorTy()) {
        std::vector<Constant*> Mask1(numElemI), Mask2(numElemI);
        for (unsigned v = 0; v < numElemI; ++v) {
          Mask1[v] = ConstantInt::get(Type::getInt32Ty(Context), v);
          Mask2[v] = ConstantInt::get(Type::getInt32Ty(Context), numElemJ+v);
        }

        K1 = new ShuffleVectorInst(K, UndefValue::get(VType),
                                   ConstantVector::get( Mask1),
                                   getReplacementName(K, false, 1));
      } else {
        Value *CV0 = ConstantInt::get(Type::getInt32Ty(Context), 0);
        K1 = ExtractElementInst::Create(K, CV0,
                                          getReplacementName(K, false, 1));
      }

      if (JType->isVectorTy()) {
        std::vector<Constant*> Mask1(numElemJ), Mask2(numElemJ);
        for (unsigned v = 0; v < numElemJ; ++v) {
          Mask1[v] = ConstantInt::get(Type::getInt32Ty(Context), v);
          Mask2[v] = ConstantInt::get(Type::getInt32Ty(Context), numElemI+v);
        }

        K2 = new ShuffleVectorInst(K, UndefValue::get(VType),
                                   ConstantVector::get( Mask2),
                                   getReplacementName(K, false, 2));
      } else {
        Value *CV1 = ConstantInt::get(Type::getInt32Ty(Context), numElem-1);
        K2 = ExtractElementInst::Create(K, CV1,
                                          getReplacementName(K, false, 2));
      }

      K1->insertAfter(K);
      K2->insertAfter(K1);
      InsertionPt = K2;
    }
  }

  // Move all uses of the function I (including pairing-induced uses) after J.
  bool BBVectorize::canMoveUsesOfIAfterJ(BasicBlock &BB,
                     DenseSet<ValuePair> &LoadMoveSetPairs,
                     Instruction *I, Instruction *J) {
    // Skip to the first instruction past I.
    BasicBlock::iterator L = std::next(BasicBlock::iterator(I));

    DenseSet<Value *> Users;
    AliasSetTracker WriteSet(*AA);
    if (I->mayWriteToMemory()) WriteSet.add(I);

    for (; cast<Instruction>(L) != J; ++L)
      (void) trackUsesOfI(Users, WriteSet, I, L, true, &LoadMoveSetPairs);

    assert(cast<Instruction>(L) == J &&
      "Tracking has not proceeded far enough to check for dependencies");
    // If J is now in the use set of I, then trackUsesOfI will return true
    // and we have a dependency cycle (and the fusing operation must abort).
    return !trackUsesOfI(Users, WriteSet, I, J, true, &LoadMoveSetPairs);
  }

  // Move all uses of the function I (including pairing-induced uses) after J.
  void BBVectorize::moveUsesOfIAfterJ(BasicBlock &BB,
                     DenseSet<ValuePair> &LoadMoveSetPairs,
                     Instruction *&InsertionPt,
                     Instruction *I, Instruction *J) {
    // Skip to the first instruction past I.
    BasicBlock::iterator L = std::next(BasicBlock::iterator(I));

    DenseSet<Value *> Users;
    AliasSetTracker WriteSet(*AA);
    if (I->mayWriteToMemory()) WriteSet.add(I);

    for (; cast<Instruction>(L) != J;) {
      if (trackUsesOfI(Users, WriteSet, I, L, true, &LoadMoveSetPairs)) {
        // Move this instruction
        Instruction *InstToMove = L; ++L;

        DEBUG(dbgs() << "BBV: moving: " << *InstToMove <<
                        " to after " << *InsertionPt << "\n");
        InstToMove->removeFromParent();
        InstToMove->insertAfter(InsertionPt);
        InsertionPt = InstToMove;
      } else {
        ++L;
      }
    }
  }

  // Collect all load instruction that are in the move set of a given first
  // pair member.  These loads depend on the first instruction, I, and so need
  // to be moved after J (the second instruction) when the pair is fused.
  void BBVectorize::collectPairLoadMoveSet(BasicBlock &BB,
                     DenseMap<Value *, Value *> &ChosenPairs,
                     DenseMap<Value *, std::vector<Value *> > &LoadMoveSet,
                     DenseSet<ValuePair> &LoadMoveSetPairs,
                     Instruction *I) {
    // Skip to the first instruction past I.
    BasicBlock::iterator L = std::next(BasicBlock::iterator(I));

    DenseSet<Value *> Users;
    AliasSetTracker WriteSet(*AA);
    if (I->mayWriteToMemory()) WriteSet.add(I);

    // Note: We cannot end the loop when we reach J because J could be moved
    // farther down the use chain by another instruction pairing. Also, J
    // could be before I if this is an inverted input.
    for (BasicBlock::iterator E = BB.end(); cast<Instruction>(L) != E; ++L) {
      if (trackUsesOfI(Users, WriteSet, I, L)) {
        if (L->mayReadFromMemory()) {
          LoadMoveSet[L].push_back(I);
          LoadMoveSetPairs.insert(ValuePair(L, I));
        }
      }
    }
  }

  // In cases where both load/stores and the computation of their pointers
  // are chosen for vectorization, we can end up in a situation where the
  // aliasing analysis starts returning different query results as the
  // process of fusing instruction pairs continues. Because the algorithm
  // relies on finding the same use dags here as were found earlier, we'll
  // need to precompute the necessary aliasing information here and then
  // manually update it during the fusion process.
  void BBVectorize::collectLoadMoveSet(BasicBlock &BB,
                     std::vector<Value *> &PairableInsts,
                     DenseMap<Value *, Value *> &ChosenPairs,
                     DenseMap<Value *, std::vector<Value *> > &LoadMoveSet,
                     DenseSet<ValuePair> &LoadMoveSetPairs) {
    for (std::vector<Value *>::iterator PI = PairableInsts.begin(),
         PIE = PairableInsts.end(); PI != PIE; ++PI) {
      DenseMap<Value *, Value *>::iterator P = ChosenPairs.find(*PI);
      if (P == ChosenPairs.end()) continue;

      Instruction *I = cast<Instruction>(P->first);
      collectPairLoadMoveSet(BB, ChosenPairs, LoadMoveSet,
                             LoadMoveSetPairs, I);
    }
  }

  // This function fuses the chosen instruction pairs into vector instructions,
  // taking care preserve any needed scalar outputs and, then, it reorders the
  // remaining instructions as needed (users of the first member of the pair
  // need to be moved to after the location of the second member of the pair
  // because the vector instruction is inserted in the location of the pair's
  // second member).
  void BBVectorize::fuseChosenPairs(BasicBlock &BB,
             std::vector<Value *> &PairableInsts,
             DenseMap<Value *, Value *> &ChosenPairs,
             DenseSet<ValuePair> &FixedOrderPairs,
             DenseMap<VPPair, unsigned> &PairConnectionTypes,
             DenseMap<ValuePair, std::vector<ValuePair> > &ConnectedPairs,
             DenseMap<ValuePair, std::vector<ValuePair> > &ConnectedPairDeps) {
    LLVMContext& Context = BB.getContext();

    // During the vectorization process, the order of the pairs to be fused
    // could be flipped. So we'll add each pair, flipped, into the ChosenPairs
    // list. After a pair is fused, the flipped pair is removed from the list.
    DenseSet<ValuePair> FlippedPairs;
    for (DenseMap<Value *, Value *>::iterator P = ChosenPairs.begin(),
         E = ChosenPairs.end(); P != E; ++P)
      FlippedPairs.insert(ValuePair(P->second, P->first));
    for (DenseSet<ValuePair>::iterator P = FlippedPairs.begin(),
         E = FlippedPairs.end(); P != E; ++P)
      ChosenPairs.insert(*P);

    DenseMap<Value *, std::vector<Value *> > LoadMoveSet;
    DenseSet<ValuePair> LoadMoveSetPairs;
    collectLoadMoveSet(BB, PairableInsts, ChosenPairs,
                       LoadMoveSet, LoadMoveSetPairs);

    DEBUG(dbgs() << "BBV: initial: \n" << BB << "\n");

    for (BasicBlock::iterator PI = BB.getFirstInsertionPt(); PI != BB.end();) {
      DenseMap<Value *, Value *>::iterator P = ChosenPairs.find(PI);
      if (P == ChosenPairs.end()) {
        ++PI;
        continue;
      }

      if (getDepthFactor(P->first) == 0) {
        // These instructions are not really fused, but are tracked as though
        // they are. Any case in which it would be interesting to fuse them
        // will be taken care of by InstCombine.
        --NumFusedOps;
        ++PI;
        continue;
      }

      Instruction *I = cast<Instruction>(P->first),
        *J = cast<Instruction>(P->second);

      DEBUG(dbgs() << "BBV: fusing: " << *I <<
             " <-> " << *J << "\n");

      // Remove the pair and flipped pair from the list.
      DenseMap<Value *, Value *>::iterator FP = ChosenPairs.find(P->second);
      assert(FP != ChosenPairs.end() && "Flipped pair not found in list");
      ChosenPairs.erase(FP);
      ChosenPairs.erase(P);

      if (!canMoveUsesOfIAfterJ(BB, LoadMoveSetPairs, I, J)) {
        DEBUG(dbgs() << "BBV: fusion of: " << *I <<
               " <-> " << *J <<
               " aborted because of non-trivial dependency cycle\n");
        --NumFusedOps;
        ++PI;
        continue;
      }

      // If the pair must have the other order, then flip it.
      bool FlipPairOrder = FixedOrderPairs.count(ValuePair(J, I));
      if (!FlipPairOrder && !FixedOrderPairs.count(ValuePair(I, J))) {
        // This pair does not have a fixed order, and so we might want to
        // flip it if that will yield fewer shuffles. We count the number
        // of dependencies connected via swaps, and those directly connected,
        // and flip the order if the number of swaps is greater.
        bool OrigOrder = true;
        DenseMap<ValuePair, std::vector<ValuePair> >::iterator IJ =
          ConnectedPairDeps.find(ValuePair(I, J));
        if (IJ == ConnectedPairDeps.end()) {
          IJ = ConnectedPairDeps.find(ValuePair(J, I));
          OrigOrder = false;
        }

        if (IJ != ConnectedPairDeps.end()) {
          unsigned NumDepsDirect = 0, NumDepsSwap = 0;
          for (std::vector<ValuePair>::iterator T = IJ->second.begin(),
               TE = IJ->second.end(); T != TE; ++T) {
            VPPair Q(IJ->first, *T);
            DenseMap<VPPair, unsigned>::iterator R =
              PairConnectionTypes.find(VPPair(Q.second, Q.first));
            assert(R != PairConnectionTypes.end() &&
                   "Cannot find pair connection type");
            if (R->second == PairConnectionDirect)
              ++NumDepsDirect;
            else if (R->second == PairConnectionSwap)
              ++NumDepsSwap;
          }

          if (!OrigOrder)
            std::swap(NumDepsDirect, NumDepsSwap);

          if (NumDepsSwap > NumDepsDirect) {
            FlipPairOrder = true;
            DEBUG(dbgs() << "BBV: reordering pair: " << *I <<
                            " <-> " << *J << "\n");
          }
        }
      }

      Instruction *L = I, *H = J;
      if (FlipPairOrder)
        std::swap(H, L);

      // If the pair being fused uses the opposite order from that in the pair
      // connection map, then we need to flip the types.
      DenseMap<ValuePair, std::vector<ValuePair> >::iterator HL =
        ConnectedPairs.find(ValuePair(H, L));
      if (HL != ConnectedPairs.end())
        for (std::vector<ValuePair>::iterator T = HL->second.begin(),
             TE = HL->second.end(); T != TE; ++T) {
          VPPair Q(HL->first, *T);
          DenseMap<VPPair, unsigned>::iterator R = PairConnectionTypes.find(Q);
          assert(R != PairConnectionTypes.end() &&
                 "Cannot find pair connection type");
          if (R->second == PairConnectionDirect)
            R->second = PairConnectionSwap;
          else if (R->second == PairConnectionSwap)
            R->second = PairConnectionDirect;
        }

      bool LBeforeH = !FlipPairOrder;
      unsigned NumOperands = I->getNumOperands();
      SmallVector<Value *, 3> ReplacedOperands(NumOperands);
      getReplacementInputsForPair(Context, L, H, ReplacedOperands,
                                  LBeforeH);

      // Make a copy of the original operation, change its type to the vector
      // type and replace its operands with the vector operands.
      Instruction *K = L->clone();
      if (L->hasName())
        K->takeName(L);
      else if (H->hasName())
        K->takeName(H);

      if (auto CS = CallSite(K)) {
        SmallVector<Type *, 3> Tys;
        FunctionType *Old = CS.getFunctionType();
        unsigned NumOld = Old->getNumParams();
        assert(NumOld <= ReplacedOperands.size());
        for (unsigned i = 0; i != NumOld; ++i)
          Tys.push_back(ReplacedOperands[i]->getType());
        CS.mutateFunctionType(
            FunctionType::get(getVecTypeForPair(L->getType(), H->getType()),
                              Tys, Old->isVarArg()));
      } else if (!isa<StoreInst>(K))
        K->mutateType(getVecTypeForPair(L->getType(), H->getType()));

      unsigned KnownIDs[] = {
        LLVMContext::MD_tbaa,
        LLVMContext::MD_alias_scope,
        LLVMContext::MD_noalias,
        LLVMContext::MD_fpmath
      };
      combineMetadata(K, H, KnownIDs);
      K->intersectOptionalDataWith(H);

      for (unsigned o = 0; o < NumOperands; ++o)
        K->setOperand(o, ReplacedOperands[o]);

      K->insertAfter(J);

      // Instruction insertion point:
      Instruction *InsertionPt = K;
      Instruction *K1 = nullptr, *K2 = nullptr;
      replaceOutputsOfPair(Context, L, H, K, InsertionPt, K1, K2);

      // The use dag of the first original instruction must be moved to after
      // the location of the second instruction. The entire use dag of the
      // first instruction is disjoint from the input dag of the second
      // (by definition), and so commutes with it.

      moveUsesOfIAfterJ(BB, LoadMoveSetPairs, InsertionPt, I, J);

      if (!isa<StoreInst>(I)) {
        L->replaceAllUsesWith(K1);
        H->replaceAllUsesWith(K2);
        AA->replaceWithNewValue(L, K1);
        AA->replaceWithNewValue(H, K2);
      }

      // Instructions that may read from memory may be in the load move set.
      // Once an instruction is fused, we no longer need its move set, and so
      // the values of the map never need to be updated. However, when a load
      // is fused, we need to merge the entries from both instructions in the
      // pair in case those instructions were in the move set of some other
      // yet-to-be-fused pair. The loads in question are the keys of the map.
      if (I->mayReadFromMemory()) {
        std::vector<ValuePair> NewSetMembers;
        DenseMap<Value *, std::vector<Value *> >::iterator II =
          LoadMoveSet.find(I);
        if (II != LoadMoveSet.end())
          for (std::vector<Value *>::iterator N = II->second.begin(),
               NE = II->second.end(); N != NE; ++N)
            NewSetMembers.push_back(ValuePair(K, *N));
        DenseMap<Value *, std::vector<Value *> >::iterator JJ =
          LoadMoveSet.find(J);
        if (JJ != LoadMoveSet.end())
          for (std::vector<Value *>::iterator N = JJ->second.begin(),
               NE = JJ->second.end(); N != NE; ++N)
            NewSetMembers.push_back(ValuePair(K, *N));
        for (std::vector<ValuePair>::iterator A = NewSetMembers.begin(),
             AE = NewSetMembers.end(); A != AE; ++A) {
          LoadMoveSet[A->first].push_back(A->second);
          LoadMoveSetPairs.insert(*A);
        }
      }

      // Before removing I, set the iterator to the next instruction.
      PI = std::next(BasicBlock::iterator(I));
      if (cast<Instruction>(PI) == J)
        ++PI;

      SE->forgetValue(I);
      SE->forgetValue(J);
      I->eraseFromParent();
      J->eraseFromParent();

      DEBUG(if (PrintAfterEveryPair) dbgs() << "BBV: block is now: \n" <<
                                               BB << "\n");
    }

    DEBUG(dbgs() << "BBV: final: \n" << BB << "\n");
  }
}

char BBVectorize::ID = 0;
static const char bb_vectorize_name[] = "Basic-Block Vectorization";
INITIALIZE_PASS_BEGIN(BBVectorize, BBV_NAME, bb_vectorize_name, false, false)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_PASS_END(BBVectorize, BBV_NAME, bb_vectorize_name, false, false)

BasicBlockPass *llvm::createBBVectorizePass(const VectorizeConfig &C) {
  return new BBVectorize(C);
}

bool
llvm::vectorizeBasicBlock(Pass *P, BasicBlock &BB, const VectorizeConfig &C) {
  BBVectorize BBVectorizer(P, *BB.getParent(), C);
  return BBVectorizer.vectorizeBB(BB);
}

//===----------------------------------------------------------------------===//
VectorizeConfig::VectorizeConfig() {
  VectorBits = ::VectorBits;
  VectorizeBools = !::NoBools;
  VectorizeInts = !::NoInts;
  VectorizeFloats = !::NoFloats;
  VectorizePointers = !::NoPointers;
  VectorizeCasts = !::NoCasts;
  VectorizeMath = !::NoMath;
  VectorizeBitManipulations = !::NoBitManipulation;
  VectorizeFMA = !::NoFMA;
  VectorizeSelect = !::NoSelect;
  VectorizeCmp = !::NoCmp;
  VectorizeGEP = !::NoGEP;
  VectorizeMemOps = !::NoMemOps;
  AlignedOnly = ::AlignedOnly;
  ReqChainDepth= ::ReqChainDepth;
  SearchLimit = ::SearchLimit;
  MaxCandPairsForCycleCheck = ::MaxCandPairsForCycleCheck;
  SplatBreaksChain = ::SplatBreaksChain;
  MaxInsts = ::MaxInsts;
  MaxPairs = ::MaxPairs;
  MaxIter = ::MaxIter;
  Pow2LenOnly = ::Pow2LenOnly;
  NoMemOpBoost = ::NoMemOpBoost;
  FastDep = ::FastDep;
}
