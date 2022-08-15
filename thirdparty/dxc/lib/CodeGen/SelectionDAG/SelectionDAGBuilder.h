//===-- SelectionDAGBuilder.h - Selection-DAG building --------*- C++ -*---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements routines for translating from LLVM IR into SelectionDAG IR.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_SELECTIONDAG_SELECTIONDAGBUILDER_H
#define LLVM_LIB_CODEGEN_SELECTIONDAG_SELECTIONDAGBUILDER_H

#include "StatepointLowering.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Statepoint.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetLowering.h"
#include <vector>

namespace llvm {

class AddrSpaceCastInst;
class AliasAnalysis;
class AllocaInst;
class BasicBlock;
class BitCastInst;
class BranchInst;
class CallInst;
class DbgValueInst;
class ExtractElementInst;
class ExtractValueInst;
class FCmpInst;
class FPExtInst;
class FPToSIInst;
class FPToUIInst;
class FPTruncInst;
class Function;
class FunctionLoweringInfo;
class GetElementPtrInst;
class GCFunctionInfo;
class ICmpInst;
class IntToPtrInst;
class IndirectBrInst;
class InvokeInst;
class InsertElementInst;
class InsertValueInst;
class Instruction;
class LoadInst;
class MachineBasicBlock;
class MachineInstr;
class MachineRegisterInfo;
class MDNode;
class MVT;
class PHINode;
class PtrToIntInst;
class ReturnInst;
class SDDbgValue;
class SExtInst;
class SelectInst;
class ShuffleVectorInst;
class SIToFPInst;
class StoreInst;
class SwitchInst;
class DataLayout;
class TargetLibraryInfo;
class TargetLowering;
class TruncInst;
class UIToFPInst;
class UnreachableInst;
class VAArgInst;
class ZExtInst;
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
/// SelectionDAGBuilder - This is the common target-independent lowering
/// implementation that is parameterized by a TargetLowering object.
///
class SelectionDAGBuilder {
  /// CurInst - The current instruction being visited
  const Instruction *CurInst;

  DenseMap<const Value*, SDValue> NodeMap;

  /// UnusedArgNodeMap - Maps argument value for unused arguments. This is used
  /// to preserve debug information for incoming arguments.
  DenseMap<const Value*, SDValue> UnusedArgNodeMap;

  /// DanglingDebugInfo - Helper type for DanglingDebugInfoMap.
  class DanglingDebugInfo {
    const DbgValueInst* DI;
    DebugLoc dl;
    unsigned SDNodeOrder;
  public:
    DanglingDebugInfo() : DI(nullptr), dl(DebugLoc()), SDNodeOrder(0) { }
    DanglingDebugInfo(const DbgValueInst *di, DebugLoc DL, unsigned SDNO) :
      DI(di), dl(DL), SDNodeOrder(SDNO) { }
    const DbgValueInst* getDI() { return DI; }
    DebugLoc getdl() { return dl; }
    unsigned getSDNodeOrder() { return SDNodeOrder; }
  };

  /// DanglingDebugInfoMap - Keeps track of dbg_values for which we have not
  /// yet seen the referent.  We defer handling these until we do see it.
  DenseMap<const Value*, DanglingDebugInfo> DanglingDebugInfoMap;

public:
  /// PendingLoads - Loads are not emitted to the program immediately.  We bunch
  /// them up and then emit token factor nodes when possible.  This allows us to
  /// get simple disambiguation between loads without worrying about alias
  /// analysis.
  SmallVector<SDValue, 8> PendingLoads;

  /// State used while lowering a statepoint sequence (gc_statepoint,
  /// gc_relocate, and gc_result).  See StatepointLowering.hpp/cpp for details.
  StatepointLoweringState StatepointLowering;
private:

  /// PendingExports - CopyToReg nodes that copy values to virtual registers
  /// for export to other blocks need to be emitted before any terminator
  /// instruction, but they have no other ordering requirements. We bunch them
  /// up and the emit a single tokenfactor for them just before terminator
  /// instructions.
  SmallVector<SDValue, 8> PendingExports;

  /// SDNodeOrder - A unique monotonically increasing number used to order the
  /// SDNodes we create.
  unsigned SDNodeOrder;

  enum CaseClusterKind {
    /// A cluster of adjacent case labels with the same destination, or just one
    /// case.
    CC_Range,
    /// A cluster of cases suitable for jump table lowering.
    CC_JumpTable,
    /// A cluster of cases suitable for bit test lowering.
    CC_BitTests
  };

  /// A cluster of case labels.
  struct CaseCluster {
    CaseClusterKind Kind;
    const ConstantInt *Low, *High;
    union {
      MachineBasicBlock *MBB;
      unsigned JTCasesIndex;
      unsigned BTCasesIndex;
    };
    uint32_t Weight;

    static CaseCluster range(const ConstantInt *Low, const ConstantInt *High,
                             MachineBasicBlock *MBB, uint32_t Weight) {
      CaseCluster C;
      C.Kind = CC_Range;
      C.Low = Low;
      C.High = High;
      C.MBB = MBB;
      C.Weight = Weight;
      return C;
    }

    static CaseCluster jumpTable(const ConstantInt *Low,
                                 const ConstantInt *High, unsigned JTCasesIndex,
                                 uint32_t Weight) {
      CaseCluster C;
      C.Kind = CC_JumpTable;
      C.Low = Low;
      C.High = High;
      C.JTCasesIndex = JTCasesIndex;
      C.Weight = Weight;
      return C;
    }

    static CaseCluster bitTests(const ConstantInt *Low, const ConstantInt *High,
                                unsigned BTCasesIndex, uint32_t Weight) {
      CaseCluster C;
      C.Kind = CC_BitTests;
      C.Low = Low;
      C.High = High;
      C.BTCasesIndex = BTCasesIndex;
      C.Weight = Weight;
      return C;
    }
  };

  typedef std::vector<CaseCluster> CaseClusterVector;
  typedef CaseClusterVector::iterator CaseClusterIt;

  struct CaseBits {
    uint64_t Mask;
    MachineBasicBlock* BB;
    unsigned Bits;
    uint32_t ExtraWeight;

    CaseBits(uint64_t mask, MachineBasicBlock* bb, unsigned bits,
             uint32_t Weight):
      Mask(mask), BB(bb), Bits(bits), ExtraWeight(Weight) { }

    CaseBits() : Mask(0), BB(nullptr), Bits(0), ExtraWeight(0) {}
  };

  typedef std::vector<CaseBits> CaseBitsVector;

  /// Sort Clusters and merge adjacent cases.
  void sortAndRangeify(CaseClusterVector &Clusters);

  /// CaseBlock - This structure is used to communicate between
  /// SelectionDAGBuilder and SDISel for the code generation of additional basic
  /// blocks needed by multi-case switch statements.
  struct CaseBlock {
    CaseBlock(ISD::CondCode cc, const Value *cmplhs, const Value *cmprhs,
              const Value *cmpmiddle,
              MachineBasicBlock *truebb, MachineBasicBlock *falsebb,
              MachineBasicBlock *me,
              uint32_t trueweight = 0, uint32_t falseweight = 0)
      : CC(cc), CmpLHS(cmplhs), CmpMHS(cmpmiddle), CmpRHS(cmprhs),
        TrueBB(truebb), FalseBB(falsebb), ThisBB(me),
        TrueWeight(trueweight), FalseWeight(falseweight) { }

    // CC - the condition code to use for the case block's setcc node
    ISD::CondCode CC;

    // CmpLHS/CmpRHS/CmpMHS - The LHS/MHS/RHS of the comparison to emit.
    // Emit by default LHS op RHS. MHS is used for range comparisons:
    // If MHS is not null: (LHS <= MHS) and (MHS <= RHS).
    const Value *CmpLHS, *CmpMHS, *CmpRHS;

    // TrueBB/FalseBB - the block to branch to if the setcc is true/false.
    MachineBasicBlock *TrueBB, *FalseBB;

    // ThisBB - the block into which to emit the code for the setcc and branches
    MachineBasicBlock *ThisBB;

    // TrueWeight/FalseWeight - branch weights.
    uint32_t TrueWeight, FalseWeight;
  };

  struct JumpTable {
    JumpTable(unsigned R, unsigned J, MachineBasicBlock *M,
              MachineBasicBlock *D): Reg(R), JTI(J), MBB(M), Default(D) {}

    /// Reg - the virtual register containing the index of the jump table entry
    //. to jump to.
    unsigned Reg;
    /// JTI - the JumpTableIndex for this jump table in the function.
    unsigned JTI;
    /// MBB - the MBB into which to emit the code for the indirect jump.
    MachineBasicBlock *MBB;
    /// Default - the MBB of the default bb, which is a successor of the range
    /// check MBB.  This is when updating PHI nodes in successors.
    MachineBasicBlock *Default;
  };
  struct JumpTableHeader {
    JumpTableHeader(APInt F, APInt L, const Value *SV, MachineBasicBlock *H,
                    bool E = false):
      First(F), Last(L), SValue(SV), HeaderBB(H), Emitted(E) {}
    APInt First;
    APInt Last;
    const Value *SValue;
    MachineBasicBlock *HeaderBB;
    bool Emitted;
  };
  typedef std::pair<JumpTableHeader, JumpTable> JumpTableBlock;

  struct BitTestCase {
    BitTestCase(uint64_t M, MachineBasicBlock* T, MachineBasicBlock* Tr,
                uint32_t Weight):
      Mask(M), ThisBB(T), TargetBB(Tr), ExtraWeight(Weight) { }
    uint64_t Mask;
    MachineBasicBlock *ThisBB;
    MachineBasicBlock *TargetBB;
    uint32_t ExtraWeight;
  };

  typedef SmallVector<BitTestCase, 3> BitTestInfo;

  struct BitTestBlock {
    BitTestBlock(APInt F, APInt R, const Value* SV,
                 unsigned Rg, MVT RgVT, bool E,
                 MachineBasicBlock* P, MachineBasicBlock* D,
                 BitTestInfo C):
      First(F), Range(R), SValue(SV), Reg(Rg), RegVT(RgVT), Emitted(E),
      Parent(P), Default(D), Cases(std::move(C)) { }
    APInt First;
    APInt Range;
    const Value *SValue;
    unsigned Reg;
    MVT RegVT;
    bool Emitted;
    MachineBasicBlock *Parent;
    MachineBasicBlock *Default;
    BitTestInfo Cases;
  };

  /// Minimum jump table density, in percent.
  enum { MinJumpTableDensity = 40 };

  /// Check whether a range of clusters is dense enough for a jump table.
  bool isDense(const CaseClusterVector &Clusters, unsigned *TotalCases,
               unsigned First, unsigned Last);

  /// Build a jump table cluster from Clusters[First..Last]. Returns false if it
  /// decides it's not a good idea.
  bool buildJumpTable(CaseClusterVector &Clusters, unsigned First,
                      unsigned Last, const SwitchInst *SI,
                      MachineBasicBlock *DefaultMBB, CaseCluster &JTCluster);

  /// Find clusters of cases suitable for jump table lowering.
  void findJumpTables(CaseClusterVector &Clusters, const SwitchInst *SI,
                      MachineBasicBlock *DefaultMBB);

  /// Check whether the range [Low,High] fits in a machine word.
  bool rangeFitsInWord(const APInt &Low, const APInt &High);

  /// Check whether these clusters are suitable for lowering with bit tests based
  /// on the number of destinations, comparison metric, and range.
  bool isSuitableForBitTests(unsigned NumDests, unsigned NumCmps,
                             const APInt &Low, const APInt &High);

  /// Build a bit test cluster from Clusters[First..Last]. Returns false if it
  /// decides it's not a good idea.
  bool buildBitTests(CaseClusterVector &Clusters, unsigned First, unsigned Last,
                     const SwitchInst *SI, CaseCluster &BTCluster);

  /// Find clusters of cases suitable for bit test lowering.
  void findBitTestClusters(CaseClusterVector &Clusters, const SwitchInst *SI);

  struct SwitchWorkListItem {
    MachineBasicBlock *MBB;
    CaseClusterIt FirstCluster;
    CaseClusterIt LastCluster;
    const ConstantInt *GE;
    const ConstantInt *LT;
  };
  typedef SmallVector<SwitchWorkListItem, 4> SwitchWorkList;

  /// Determine the rank by weight of CC in [First,Last]. If CC has more weight
  /// than each cluster in the range, its rank is 0.
  static unsigned caseClusterRank(const CaseCluster &CC, CaseClusterIt First,
                                  CaseClusterIt Last);

  /// Emit comparison and split W into two subtrees.
  void splitWorkItem(SwitchWorkList &WorkList, const SwitchWorkListItem &W,
                     Value *Cond, MachineBasicBlock *SwitchMBB);

  /// Lower W.
  void lowerWorkItem(SwitchWorkListItem W, Value *Cond,
                     MachineBasicBlock *SwitchMBB,
                     MachineBasicBlock *DefaultMBB);


  /// A class which encapsulates all of the information needed to generate a
  /// stack protector check and signals to isel via its state being initialized
  /// that a stack protector needs to be generated.
  ///
  /// *NOTE* The following is a high level documentation of SelectionDAG Stack
  /// Protector Generation. The reason that it is placed here is for a lack of
  /// other good places to stick it.
  ///
  /// High Level Overview of SelectionDAG Stack Protector Generation:
  ///
  /// Previously, generation of stack protectors was done exclusively in the
  /// pre-SelectionDAG Codegen LLVM IR Pass "Stack Protector". This necessitated
  /// splitting basic blocks at the IR level to create the success/failure basic
  /// blocks in the tail of the basic block in question. As a result of this,
  /// calls that would have qualified for the sibling call optimization were no
  /// longer eligible for optimization since said calls were no longer right in
  /// the "tail position" (i.e. the immediate predecessor of a ReturnInst
  /// instruction).
  ///
  /// Then it was noticed that since the sibling call optimization causes the
  /// callee to reuse the caller's stack, if we could delay the generation of
  /// the stack protector check until later in CodeGen after the sibling call
  /// decision was made, we get both the tail call optimization and the stack
  /// protector check!
  ///
  /// A few goals in solving this problem were:
  ///
  ///   1. Preserve the architecture independence of stack protector generation.
  ///
  ///   2. Preserve the normal IR level stack protector check for platforms like
  ///      OpenBSD for which we support platform-specific stack protector
  ///      generation.
  ///
  /// The main problem that guided the present solution is that one can not
  /// solve this problem in an architecture independent manner at the IR level
  /// only. This is because:
  ///
  ///   1. The decision on whether or not to perform a sibling call on certain
  ///      platforms (for instance i386) requires lower level information
  ///      related to available registers that can not be known at the IR level.
  ///
  ///   2. Even if the previous point were not true, the decision on whether to
  ///      perform a tail call is done in LowerCallTo in SelectionDAG which
  ///      occurs after the Stack Protector Pass. As a result, one would need to
  ///      put the relevant callinst into the stack protector check success
  ///      basic block (where the return inst is placed) and then move it back
  ///      later at SelectionDAG/MI time before the stack protector check if the
  ///      tail call optimization failed. The MI level option was nixed
  ///      immediately since it would require platform-specific pattern
  ///      matching. The SelectionDAG level option was nixed because
  ///      SelectionDAG only processes one IR level basic block at a time
  ///      implying one could not create a DAG Combine to move the callinst.
  ///
  /// To get around this problem a few things were realized:
  ///
  ///   1. While one can not handle multiple IR level basic blocks at the
  ///      SelectionDAG Level, one can generate multiple machine basic blocks
  ///      for one IR level basic block. This is how we handle bit tests and
  ///      switches.
  ///
  ///   2. At the MI level, tail calls are represented via a special return
  ///      MIInst called "tcreturn". Thus if we know the basic block in which we
  ///      wish to insert the stack protector check, we get the correct behavior
  ///      by always inserting the stack protector check right before the return
  ///      statement. This is a "magical transformation" since no matter where
  ///      the stack protector check intrinsic is, we always insert the stack
  ///      protector check code at the end of the BB.
  ///
  /// Given the aforementioned constraints, the following solution was devised:
  ///
  ///   1. On platforms that do not support SelectionDAG stack protector check
  ///      generation, allow for the normal IR level stack protector check
  ///      generation to continue.
  ///
  ///   2. On platforms that do support SelectionDAG stack protector check
  ///      generation:
  ///
  ///     a. Use the IR level stack protector pass to decide if a stack
  ///        protector is required/which BB we insert the stack protector check
  ///        in by reusing the logic already therein. If we wish to generate a
  ///        stack protector check in a basic block, we place a special IR
  ///        intrinsic called llvm.stackprotectorcheck right before the BB's
  ///        returninst or if there is a callinst that could potentially be
  ///        sibling call optimized, before the call inst.
  ///
  ///     b. Then when a BB with said intrinsic is processed, we codegen the BB
  ///        normally via SelectBasicBlock. In said process, when we visit the
  ///        stack protector check, we do not actually emit anything into the
  ///        BB. Instead, we just initialize the stack protector descriptor
  ///        class (which involves stashing information/creating the success
  ///        mbbb and the failure mbb if we have not created one for this
  ///        function yet) and export the guard variable that we are going to
  ///        compare.
  ///
  ///     c. After we finish selecting the basic block, in FinishBasicBlock if
  ///        the StackProtectorDescriptor attached to the SelectionDAGBuilder is
  ///        initialized, we first find a splice point in the parent basic block
  ///        before the terminator and then splice the terminator of said basic
  ///        block into the success basic block. Then we code-gen a new tail for
  ///        the parent basic block consisting of the two loads, the comparison,
  ///        and finally two branches to the success/failure basic blocks. We
  ///        conclude by code-gening the failure basic block if we have not
  ///        code-gened it already (all stack protector checks we generate in
  ///        the same function, use the same failure basic block).
  class StackProtectorDescriptor {
  public:
    StackProtectorDescriptor() : ParentMBB(nullptr), SuccessMBB(nullptr),
                                 FailureMBB(nullptr), Guard(nullptr),
                                 GuardReg(0) { }

    /// Returns true if all fields of the stack protector descriptor are
    /// initialized implying that we should/are ready to emit a stack protector.
    bool shouldEmitStackProtector() const {
      return ParentMBB && SuccessMBB && FailureMBB && Guard;
    }

    /// Initialize the stack protector descriptor structure for a new basic
    /// block.
    void initialize(const BasicBlock *BB,
                    MachineBasicBlock *MBB,
                    const CallInst &StackProtCheckCall) {
      // Make sure we are not initialized yet.
      assert(!shouldEmitStackProtector() && "Stack Protector Descriptor is "
             "already initialized!");
      ParentMBB = MBB;
      SuccessMBB = AddSuccessorMBB(BB, MBB, /* IsLikely */ true);
      FailureMBB = AddSuccessorMBB(BB, MBB, /* IsLikely */ false, FailureMBB);
      if (!Guard)
        Guard = StackProtCheckCall.getArgOperand(0);
    }

    /// Reset state that changes when we handle different basic blocks.
    ///
    /// This currently includes:
    ///
    /// 1. The specific basic block we are generating a
    /// stack protector for (ParentMBB).
    ///
    /// 2. The successor machine basic block that will contain the tail of
    /// parent mbb after we create the stack protector check (SuccessMBB). This
    /// BB is visited only on stack protector check success.
    void resetPerBBState() {
      ParentMBB = nullptr;
      SuccessMBB = nullptr;
    }

    /// Reset state that only changes when we switch functions.
    ///
    /// This currently includes:
    ///
    /// 1. FailureMBB since we reuse the failure code path for all stack
    /// protector checks created in an individual function.
    ///
    /// 2.The guard variable since the guard variable we are checking against is
    /// always the same.
    void resetPerFunctionState() {
      FailureMBB = nullptr;
      Guard = nullptr;
    }

    MachineBasicBlock *getParentMBB() { return ParentMBB; }
    MachineBasicBlock *getSuccessMBB() { return SuccessMBB; }
    MachineBasicBlock *getFailureMBB() { return FailureMBB; }
    const Value *getGuard() { return Guard; }

    unsigned getGuardReg() const { return GuardReg; }
    void setGuardReg(unsigned R) { GuardReg = R; }

  private:
    /// The basic block for which we are generating the stack protector.
    ///
    /// As a result of stack protector generation, we will splice the
    /// terminators of this basic block into the successor mbb SuccessMBB and
    /// replace it with a compare/branch to the successor mbbs
    /// SuccessMBB/FailureMBB depending on whether or not the stack protector
    /// was violated.
    MachineBasicBlock *ParentMBB;

    /// A basic block visited on stack protector check success that contains the
    /// terminators of ParentMBB.
    MachineBasicBlock *SuccessMBB;

    /// This basic block visited on stack protector check failure that will
    /// contain a call to __stack_chk_fail().
    MachineBasicBlock *FailureMBB;

    /// The guard variable which we will compare against the stored value in the
    /// stack protector stack slot.
    const Value *Guard;

    /// The virtual register holding the stack guard value.
    unsigned GuardReg;

    /// Add a successor machine basic block to ParentMBB. If the successor mbb
    /// has not been created yet (i.e. if SuccMBB = 0), then the machine basic
    /// block will be created. Assign a large weight if IsLikely is true.
    MachineBasicBlock *AddSuccessorMBB(const BasicBlock *BB,
                                       MachineBasicBlock *ParentMBB,
                                       bool IsLikely,
                                       MachineBasicBlock *SuccMBB = nullptr);
  };

private:
  const TargetMachine &TM;
public:
  /// Lowest valid SDNodeOrder. The special case 0 is reserved for scheduling
  /// nodes without a corresponding SDNode.
  static const unsigned LowestSDNodeOrder = 1;

  SelectionDAG &DAG;
  const DataLayout *DL;
  AliasAnalysis *AA;
  const TargetLibraryInfo *LibInfo;

  /// SwitchCases - Vector of CaseBlock structures used to communicate
  /// SwitchInst code generation information.
  std::vector<CaseBlock> SwitchCases;
  /// JTCases - Vector of JumpTable structures used to communicate
  /// SwitchInst code generation information.
  std::vector<JumpTableBlock> JTCases;
  /// BitTestCases - Vector of BitTestBlock structures used to communicate
  /// SwitchInst code generation information.
  std::vector<BitTestBlock> BitTestCases;
  /// A StackProtectorDescriptor structure used to communicate stack protector
  /// information in between SelectBasicBlock and FinishBasicBlock.
  StackProtectorDescriptor SPDescriptor;

  // Emit PHI-node-operand constants only once even if used by multiple
  // PHI nodes.
  DenseMap<const Constant *, unsigned> ConstantsOut;

  /// FuncInfo - Information about the function as a whole.
  ///
  FunctionLoweringInfo &FuncInfo;

  /// OptLevel - What optimization level we're generating code for.
  ///
  CodeGenOpt::Level OptLevel;

  /// GFI - Garbage collection metadata for the function.
  GCFunctionInfo *GFI;

  /// LPadToCallSiteMap - Map a landing pad to the call site indexes.
  DenseMap<MachineBasicBlock*, SmallVector<unsigned, 4> > LPadToCallSiteMap;

  /// HasTailCall - This is set to true if a call in the current
  /// block has been translated as a tail call. In this case,
  /// no subsequent DAG nodes should be created.
  ///
  bool HasTailCall;

  LLVMContext *Context;

  SelectionDAGBuilder(SelectionDAG &dag, FunctionLoweringInfo &funcinfo,
                      CodeGenOpt::Level ol)
    : CurInst(nullptr), SDNodeOrder(LowestSDNodeOrder), TM(dag.getTarget()),
      DAG(dag), FuncInfo(funcinfo), OptLevel(ol),
      HasTailCall(false) {
  }

  void init(GCFunctionInfo *gfi, AliasAnalysis &aa,
            const TargetLibraryInfo *li);

  /// clear - Clear out the current SelectionDAG and the associated
  /// state and prepare this SelectionDAGBuilder object to be used
  /// for a new block. This doesn't clear out information about
  /// additional blocks that are needed to complete switch lowering
  /// or PHI node updating; that information is cleared out as it is
  /// consumed.
  void clear();

  /// clearDanglingDebugInfo - Clear the dangling debug information
  /// map. This function is separated from the clear so that debug
  /// information that is dangling in a basic block can be properly
  /// resolved in a different basic block. This allows the
  /// SelectionDAG to resolve dangling debug information attached
  /// to PHI nodes.
  void clearDanglingDebugInfo();

  /// getRoot - Return the current virtual root of the Selection DAG,
  /// flushing any PendingLoad items. This must be done before emitting
  /// a store or any other node that may need to be ordered after any
  /// prior load instructions.
  ///
  SDValue getRoot();

  /// getControlRoot - Similar to getRoot, but instead of flushing all the
  /// PendingLoad items, flush all the PendingExports items. It is necessary
  /// to do this before emitting a terminator instruction.
  ///
  SDValue getControlRoot();

  SDLoc getCurSDLoc() const {
    return SDLoc(CurInst, SDNodeOrder);
  }

  DebugLoc getCurDebugLoc() const {
    return CurInst ? CurInst->getDebugLoc() : DebugLoc();
  }

  unsigned getSDNodeOrder() const { return SDNodeOrder; }

  void CopyValueToVirtualRegister(const Value *V, unsigned Reg);

  void visit(const Instruction &I);

  void visit(unsigned Opcode, const User &I);

  /// getCopyFromRegs - If there was virtual register allocated for the value V
  /// emit CopyFromReg of the specified type Ty. Return empty SDValue() otherwise.
  SDValue getCopyFromRegs(const Value *V, Type *Ty);

  // resolveDanglingDebugInfo - if we saw an earlier dbg_value referring to V,
  // generate the debug data structures now that we've seen its definition.
  void resolveDanglingDebugInfo(const Value *V, SDValue Val);
  SDValue getValue(const Value *V);
  bool findValue(const Value *V) const;

  SDValue getNonRegisterValue(const Value *V);
  SDValue getValueImpl(const Value *V);

  void setValue(const Value *V, SDValue NewN) {
    SDValue &N = NodeMap[V];
    assert(!N.getNode() && "Already set a value for this node!");
    N = NewN;
  }

  void setUnusedArgValue(const Value *V, SDValue NewN) {
    SDValue &N = UnusedArgNodeMap[V];
    assert(!N.getNode() && "Already set a value for this node!");
    N = NewN;
  }

  void FindMergedConditions(const Value *Cond, MachineBasicBlock *TBB,
                            MachineBasicBlock *FBB, MachineBasicBlock *CurBB,
                            MachineBasicBlock *SwitchBB, unsigned Opc,
                            uint32_t TW, uint32_t FW);
  void EmitBranchForMergedCondition(const Value *Cond, MachineBasicBlock *TBB,
                                    MachineBasicBlock *FBB,
                                    MachineBasicBlock *CurBB,
                                    MachineBasicBlock *SwitchBB,
                                    uint32_t TW, uint32_t FW);
  bool ShouldEmitAsBranches(const std::vector<CaseBlock> &Cases);
  bool isExportableFromCurrentBlock(const Value *V, const BasicBlock *FromBB);
  void CopyToExportRegsIfNeeded(const Value *V);
  void ExportFromCurrentBlock(const Value *V);
  void LowerCallTo(ImmutableCallSite CS, SDValue Callee, bool IsTailCall,
                   MachineBasicBlock *LandingPad = nullptr);

  std::pair<SDValue, SDValue> lowerCallOperands(
          ImmutableCallSite CS,
          unsigned ArgIdx,
          unsigned NumArgs,
          SDValue Callee,
          Type *ReturnTy,
          MachineBasicBlock *LandingPad = nullptr,
          bool IsPatchPoint = false);

  /// UpdateSplitBlock - When an MBB was split during scheduling, update the
  /// references that need to refer to the last resulting block.
  void UpdateSplitBlock(MachineBasicBlock *First, MachineBasicBlock *Last);

  // This function is responsible for the whole statepoint lowering process.
  // It uniformly handles invoke and call statepoints.
  void LowerStatepoint(ImmutableStatepoint Statepoint,
                       MachineBasicBlock *LandingPad = nullptr);
private:
  std::pair<SDValue, SDValue> lowerInvokable(
          TargetLowering::CallLoweringInfo &CLI,
          MachineBasicBlock *LandingPad);

  // Terminator instructions.
  void visitRet(const ReturnInst &I);
  void visitBr(const BranchInst &I);
  void visitSwitch(const SwitchInst &I);
  void visitIndirectBr(const IndirectBrInst &I);
  void visitUnreachable(const UnreachableInst &I);

  uint32_t getEdgeWeight(const MachineBasicBlock *Src,
                         const MachineBasicBlock *Dst) const;
  void addSuccessorWithWeight(MachineBasicBlock *Src, MachineBasicBlock *Dst,
                              uint32_t Weight = 0);
public:
  void visitSwitchCase(CaseBlock &CB,
                       MachineBasicBlock *SwitchBB);
  void visitSPDescriptorParent(StackProtectorDescriptor &SPD,
                               MachineBasicBlock *ParentBB);
  void visitSPDescriptorFailure(StackProtectorDescriptor &SPD);
  void visitBitTestHeader(BitTestBlock &B, MachineBasicBlock *SwitchBB);
  void visitBitTestCase(BitTestBlock &BB,
                        MachineBasicBlock* NextMBB,
                        uint32_t BranchWeightToNext,
                        unsigned Reg,
                        BitTestCase &B,
                        MachineBasicBlock *SwitchBB);
  void visitJumpTable(JumpTable &JT);
  void visitJumpTableHeader(JumpTable &JT, JumpTableHeader &JTH,
                            MachineBasicBlock *SwitchBB);

private:
  // These all get lowered before this pass.
  void visitInvoke(const InvokeInst &I);
  void visitResume(const ResumeInst &I);

  void visitBinary(const User &I, unsigned OpCode);
  void visitShift(const User &I, unsigned Opcode);
  void visitAdd(const User &I)  { visitBinary(I, ISD::ADD); }
  void visitFAdd(const User &I) { visitBinary(I, ISD::FADD); }
  void visitSub(const User &I)  { visitBinary(I, ISD::SUB); }
  void visitFSub(const User &I);
  void visitMul(const User &I)  { visitBinary(I, ISD::MUL); }
  void visitFMul(const User &I) { visitBinary(I, ISD::FMUL); }
  void visitURem(const User &I) { visitBinary(I, ISD::UREM); }
  void visitSRem(const User &I) { visitBinary(I, ISD::SREM); }
  void visitFRem(const User &I) { visitBinary(I, ISD::FREM); }
  void visitUDiv(const User &I) { visitBinary(I, ISD::UDIV); }
  void visitSDiv(const User &I);
  void visitFDiv(const User &I) { visitBinary(I, ISD::FDIV); }
  void visitAnd (const User &I) { visitBinary(I, ISD::AND); }
  void visitOr  (const User &I) { visitBinary(I, ISD::OR); }
  void visitXor (const User &I) { visitBinary(I, ISD::XOR); }
  void visitShl (const User &I) { visitShift(I, ISD::SHL); }
  void visitLShr(const User &I) { visitShift(I, ISD::SRL); }
  void visitAShr(const User &I) { visitShift(I, ISD::SRA); }
  void visitICmp(const User &I);
  void visitFCmp(const User &I);
  // Visit the conversion instructions
  void visitTrunc(const User &I);
  void visitZExt(const User &I);
  void visitSExt(const User &I);
  void visitFPTrunc(const User &I);
  void visitFPExt(const User &I);
  void visitFPToUI(const User &I);
  void visitFPToSI(const User &I);
  void visitUIToFP(const User &I);
  void visitSIToFP(const User &I);
  void visitPtrToInt(const User &I);
  void visitIntToPtr(const User &I);
  void visitBitCast(const User &I);
  void visitAddrSpaceCast(const User &I);

  void visitExtractElement(const User &I);
  void visitInsertElement(const User &I);
  void visitShuffleVector(const User &I);

  void visitExtractValue(const ExtractValueInst &I);
  void visitInsertValue(const InsertValueInst &I);
  void visitLandingPad(const LandingPadInst &I);

  void visitGetElementPtr(const User &I);
  void visitSelect(const User &I);

  void visitAlloca(const AllocaInst &I);
  void visitLoad(const LoadInst &I);
  void visitStore(const StoreInst &I);
  void visitMaskedLoad(const CallInst &I);
  void visitMaskedStore(const CallInst &I);
  void visitMaskedGather(const CallInst &I);
  void visitMaskedScatter(const CallInst &I);
  void visitAtomicCmpXchg(const AtomicCmpXchgInst &I);
  void visitAtomicRMW(const AtomicRMWInst &I);
  void visitFence(const FenceInst &I);
  void visitPHI(const PHINode &I);
  void visitCall(const CallInst &I);
  bool visitMemCmpCall(const CallInst &I);
  bool visitMemChrCall(const CallInst &I);
  bool visitStrCpyCall(const CallInst &I, bool isStpcpy);
  bool visitStrCmpCall(const CallInst &I);
  bool visitStrLenCall(const CallInst &I);
  bool visitStrNLenCall(const CallInst &I);
  bool visitUnaryFloatCall(const CallInst &I, unsigned Opcode);
  bool visitBinaryFloatCall(const CallInst &I, unsigned Opcode);
  void visitAtomicLoad(const LoadInst &I);
  void visitAtomicStore(const StoreInst &I);

  void visitInlineAsm(ImmutableCallSite CS);
  const char *visitIntrinsicCall(const CallInst &I, unsigned Intrinsic);
  void visitTargetIntrinsic(const CallInst &I, unsigned Intrinsic);

  void visitVAStart(const CallInst &I);
  void visitVAArg(const VAArgInst &I);
  void visitVAEnd(const CallInst &I);
  void visitVACopy(const CallInst &I);
  void visitStackmap(const CallInst &I);
  void visitPatchpoint(ImmutableCallSite CS,
                       MachineBasicBlock *LandingPad = nullptr);

  // These three are implemented in StatepointLowering.cpp
  void visitStatepoint(const CallInst &I);
  void visitGCRelocate(const CallInst &I);
  void visitGCResult(const CallInst &I);

  void visitUserOp1(const Instruction &I) {
    llvm_unreachable("UserOp1 should not exist at instruction selection time!");
  }
  void visitUserOp2(const Instruction &I) {
    llvm_unreachable("UserOp2 should not exist at instruction selection time!");
  }

  void processIntegerCallValue(const Instruction &I,
                               SDValue Value, bool IsSigned);

  void HandlePHINodesInSuccessorBlocks(const BasicBlock *LLVMBB);

  /// EmitFuncArgumentDbgValue - If V is an function argument then create
  /// corresponding DBG_VALUE machine instruction for it now. At the end of
  /// instruction selection, they will be inserted to the entry BB.
  bool EmitFuncArgumentDbgValue(const Value *V, DILocalVariable *Variable,
                                DIExpression *Expr, DILocation *DL,
                                int64_t Offset, bool IsIndirect,
                                const SDValue &N);

  /// Return the next block after MBB, or nullptr if there is none.
  MachineBasicBlock *NextBlock(MachineBasicBlock *MBB);

  /// Update the DAG and DAG builder with the relevant information after
  /// a new root node has been created which could be a tail call.
  void updateDAGForMaybeTailCall(SDValue MaybeTC);
};

/// RegsForValue - This struct represents the registers (physical or virtual)
/// that a particular set of values is assigned, and the type information about
/// the value. The most common situation is to represent one value at a time,
/// but struct or array values are handled element-wise as multiple values.  The
/// splitting of aggregates is performed recursively, so that we never have
/// aggregate-typed registers. The values at this point do not necessarily have
/// legal types, so each value may require one or more registers of some legal
/// type.
///
struct RegsForValue {
  /// ValueVTs - The value types of the values, which may not be legal, and
  /// may need be promoted or synthesized from one or more registers.
  ///
  SmallVector<EVT, 4> ValueVTs;

  /// RegVTs - The value types of the registers. This is the same size as
  /// ValueVTs and it records, for each value, what the type of the assigned
  /// register or registers are. (Individual values are never synthesized
  /// from more than one type of register.)
  ///
  /// With virtual registers, the contents of RegVTs is redundant with TLI's
  /// getRegisterType member function, however when with physical registers
  /// it is necessary to have a separate record of the types.
  ///
  SmallVector<MVT, 4> RegVTs;

  /// Regs - This list holds the registers assigned to the values.
  /// Each legal or promoted value requires one register, and each
  /// expanded value requires multiple registers.
  ///
  SmallVector<unsigned, 4> Regs;

  RegsForValue();

  RegsForValue(const SmallVector<unsigned, 4> &regs, MVT regvt, EVT valuevt);

  RegsForValue(LLVMContext &Context, const TargetLowering &TLI,
               const DataLayout &DL, unsigned Reg, Type *Ty);

  /// append - Add the specified values to this one.
  void append(const RegsForValue &RHS) {
    ValueVTs.append(RHS.ValueVTs.begin(), RHS.ValueVTs.end());
    RegVTs.append(RHS.RegVTs.begin(), RHS.RegVTs.end());
    Regs.append(RHS.Regs.begin(), RHS.Regs.end());
  }

  /// getCopyFromRegs - Emit a series of CopyFromReg nodes that copies from
  /// this value and returns the result as a ValueVTs value.  This uses
  /// Chain/Flag as the input and updates them for the output Chain/Flag.
  /// If the Flag pointer is NULL, no flag is used.
  SDValue getCopyFromRegs(SelectionDAG &DAG, FunctionLoweringInfo &FuncInfo,
                          SDLoc dl,
                          SDValue &Chain, SDValue *Flag,
                          const Value *V = nullptr) const;

  /// getCopyToRegs - Emit a series of CopyToReg nodes that copies the specified
  /// value into the registers specified by this object.  This uses Chain/Flag
  /// as the input and updates them for the output Chain/Flag.  If the Flag
  /// pointer is nullptr, no flag is used.  If V is not nullptr, then it is used
  /// in printing better diagnostic messages on error.
  void
  getCopyToRegs(SDValue Val, SelectionDAG &DAG, SDLoc dl, SDValue &Chain,
                SDValue *Flag, const Value *V = nullptr,
                ISD::NodeType PreferredExtendType = ISD::ANY_EXTEND) const;

  /// AddInlineAsmOperands - Add this value to the specified inlineasm node
  /// operand list.  This adds the code marker, matching input operand index
  /// (if applicable), and includes the number of values added into it.
  void AddInlineAsmOperands(unsigned Kind,
                            bool HasMatching, unsigned MatchingIdx, SDLoc dl,
                            SelectionDAG &DAG,
                            std::vector<SDValue> &Ops) const;
};

} // end namespace llvm

#endif
