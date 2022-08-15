//===-- PeepholeOptimizer.cpp - Peephole Optimizations --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Perform peephole optimizations on the machine code:
//
// - Optimize Extensions
//
//     Optimization of sign / zero extension instructions. It may be extended to
//     handle other instructions with similar properties.
//
//     On some targets, some instructions, e.g. X86 sign / zero extension, may
//     leave the source value in the lower part of the result. This optimization
//     will replace some uses of the pre-extension value with uses of the
//     sub-register of the results.
//
// - Optimize Comparisons
//
//     Optimization of comparison instructions. For instance, in this code:
//
//       sub r1, 1
//       cmp r1, 0
//       bz  L1
//
//     If the "sub" instruction all ready sets (or could be modified to set) the
//     same flag that the "cmp" instruction sets and that "bz" uses, then we can
//     eliminate the "cmp" instruction.
//
//     Another instance, in this code:
//
//       sub r1, r3 | sub r1, imm
//       cmp r3, r1 or cmp r1, r3 | cmp r1, imm
//       bge L1
//
//     If the branch instruction can use flag from "sub", then we can replace
//     "sub" with "subs" and eliminate the "cmp" instruction.
//
// - Optimize Loads:
//
//     Loads that can be folded into a later instruction. A load is foldable
//     if it loads to virtual registers and the virtual register defined has 
//     a single use.
//
// - Optimize Copies and Bitcast (more generally, target specific copies):
//
//     Rewrite copies and bitcasts to avoid cross register bank copies
//     when possible.
//     E.g., Consider the following example, where capital and lower
//     letters denote different register file:
//     b = copy A <-- cross-bank copy
//     C = copy b <-- cross-bank copy
//   =>
//     b = copy A <-- cross-bank copy
//     C = copy A <-- same-bank copy
//
//     E.g., for bitcast:
//     b = bitcast A <-- cross-bank copy
//     C = bitcast b <-- cross-bank copy
//   =>
//     b = bitcast A <-- cross-bank copy
//     C = copy A    <-- same-bank copy
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include <utility>
using namespace llvm;

#define DEBUG_TYPE "peephole-opt"

// Optimize Extensions
static cl::opt<bool>
Aggressive("aggressive-ext-opt", cl::Hidden,
           cl::desc("Aggressive extension optimization"));

static cl::opt<bool>
DisablePeephole("disable-peephole", cl::Hidden, cl::init(false),
                cl::desc("Disable the peephole optimizer"));

static cl::opt<bool>
DisableAdvCopyOpt("disable-adv-copy-opt", cl::Hidden, cl::init(false),
                  cl::desc("Disable advanced copy optimization"));

STATISTIC(NumReuse,      "Number of extension results reused");
STATISTIC(NumCmps,       "Number of compares eliminated");
STATISTIC(NumImmFold,    "Number of move immediate folded");
STATISTIC(NumLoadFold,   "Number of loads folded");
STATISTIC(NumSelects,    "Number of selects optimized");
STATISTIC(NumUncoalescableCopies, "Number of uncoalescable copies optimized");
STATISTIC(NumRewrittenCopies, "Number of copies rewritten");

namespace {
  class PeepholeOptimizer : public MachineFunctionPass {
    const TargetInstrInfo *TII;
    const TargetRegisterInfo *TRI;
    MachineRegisterInfo   *MRI;
    MachineDominatorTree  *DT;  // Machine dominator tree

  public:
    static char ID; // Pass identification
    PeepholeOptimizer() : MachineFunctionPass(ID) {
      initializePeepholeOptimizerPass(*PassRegistry::getPassRegistry());
    }

    bool runOnMachineFunction(MachineFunction &MF) override;

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesCFG();
      MachineFunctionPass::getAnalysisUsage(AU);
      if (Aggressive) {
        AU.addRequired<MachineDominatorTree>();
        AU.addPreserved<MachineDominatorTree>();
      }
    }

  private:
    bool optimizeCmpInstr(MachineInstr *MI, MachineBasicBlock *MBB);
    bool optimizeExtInstr(MachineInstr *MI, MachineBasicBlock *MBB,
                          SmallPtrSetImpl<MachineInstr*> &LocalMIs);
    bool optimizeSelect(MachineInstr *MI,
                        SmallPtrSetImpl<MachineInstr *> &LocalMIs);
    bool optimizeCondBranch(MachineInstr *MI);
    bool optimizeCopyOrBitcast(MachineInstr *MI);
    bool optimizeCoalescableCopy(MachineInstr *MI);
    bool optimizeUncoalescableCopy(MachineInstr *MI,
                                   SmallPtrSetImpl<MachineInstr *> &LocalMIs);
    bool findNextSource(unsigned &Reg, unsigned &SubReg);
    bool isMoveImmediate(MachineInstr *MI,
                         SmallSet<unsigned, 4> &ImmDefRegs,
                         DenseMap<unsigned, MachineInstr*> &ImmDefMIs);
    bool foldImmediate(MachineInstr *MI, MachineBasicBlock *MBB,
                       SmallSet<unsigned, 4> &ImmDefRegs,
                       DenseMap<unsigned, MachineInstr*> &ImmDefMIs);
    bool isLoadFoldable(MachineInstr *MI,
                        SmallSet<unsigned, 16> &FoldAsLoadDefCandidates);

    /// \brief Check whether \p MI is understood by the register coalescer
    /// but may require some rewriting.
    bool isCoalescableCopy(const MachineInstr &MI) {
      // SubregToRegs are not interesting, because they are already register
      // coalescer friendly.
      return MI.isCopy() || (!DisableAdvCopyOpt &&
                             (MI.isRegSequence() || MI.isInsertSubreg() ||
                              MI.isExtractSubreg()));
    }

    /// \brief Check whether \p MI is a copy like instruction that is
    /// not recognized by the register coalescer.
    bool isUncoalescableCopy(const MachineInstr &MI) {
      return MI.isBitcast() ||
             (!DisableAdvCopyOpt &&
              (MI.isRegSequenceLike() || MI.isInsertSubregLike() ||
               MI.isExtractSubregLike()));
    }
  };

  /// \brief Helper class to track the possible sources of a value defined by
  /// a (chain of) copy related instructions.
  /// Given a definition (instruction and definition index), this class
  /// follows the use-def chain to find successive suitable sources.
  /// The given source can be used to rewrite the definition into
  /// def = COPY src.
  ///
  /// For instance, let us consider the following snippet:
  /// v0 =
  /// v2 = INSERT_SUBREG v1, v0, sub0
  /// def = COPY v2.sub0
  ///
  /// Using a ValueTracker for def = COPY v2.sub0 will give the following
  /// suitable sources:
  /// v2.sub0 and v0.
  /// Then, def can be rewritten into def = COPY v0.
  class ValueTracker {
  private:
    /// The current point into the use-def chain.
    const MachineInstr *Def;
    /// The index of the definition in Def.
    unsigned DefIdx;
    /// The sub register index of the definition.
    unsigned DefSubReg;
    /// The register where the value can be found.
    unsigned Reg;
    /// Specifiy whether or not the value tracking looks through
    /// complex instructions. When this is false, the value tracker
    /// bails on everything that is not a copy or a bitcast.
    ///
    /// Note: This could have been implemented as a specialized version of
    /// the ValueTracker class but that would have complicated the code of
    /// the users of this class.
    bool UseAdvancedTracking;
    /// MachineRegisterInfo used to perform tracking.
    const MachineRegisterInfo &MRI;
    /// Optional TargetInstrInfo used to perform some complex
    /// tracking.
    const TargetInstrInfo *TII;

    /// \brief Dispatcher to the right underlying implementation of
    /// getNextSource.
    bool getNextSourceImpl(unsigned &SrcReg, unsigned &SrcSubReg);
    /// \brief Specialized version of getNextSource for Copy instructions.
    bool getNextSourceFromCopy(unsigned &SrcReg, unsigned &SrcSubReg);
    /// \brief Specialized version of getNextSource for Bitcast instructions.
    bool getNextSourceFromBitcast(unsigned &SrcReg, unsigned &SrcSubReg);
    /// \brief Specialized version of getNextSource for RegSequence
    /// instructions.
    bool getNextSourceFromRegSequence(unsigned &SrcReg, unsigned &SrcSubReg);
    /// \brief Specialized version of getNextSource for InsertSubreg
    /// instructions.
    bool getNextSourceFromInsertSubreg(unsigned &SrcReg, unsigned &SrcSubReg);
    /// \brief Specialized version of getNextSource for ExtractSubreg
    /// instructions.
    bool getNextSourceFromExtractSubreg(unsigned &SrcReg, unsigned &SrcSubReg);
    /// \brief Specialized version of getNextSource for SubregToReg
    /// instructions.
    bool getNextSourceFromSubregToReg(unsigned &SrcReg, unsigned &SrcSubReg);

  public:
    /// \brief Create a ValueTracker instance for the value defined by \p Reg.
    /// \p DefSubReg represents the sub register index the value tracker will
    /// track. It does not need to match the sub register index used in the
    /// definition of \p Reg.
    /// \p UseAdvancedTracking specifies whether or not the value tracker looks
    /// through complex instructions. By default (false), it handles only copy
    /// and bitcast instructions.
    /// If \p Reg is a physical register, a value tracker constructed with
    /// this constructor will not find any alternative source.
    /// Indeed, when \p Reg is a physical register that constructor does not
    /// know which definition of \p Reg it should track.
    /// Use the next constructor to track a physical register.
    ValueTracker(unsigned Reg, unsigned DefSubReg,
                 const MachineRegisterInfo &MRI,
                 bool UseAdvancedTracking = false,
                 const TargetInstrInfo *TII = nullptr)
        : Def(nullptr), DefIdx(0), DefSubReg(DefSubReg), Reg(Reg),
          UseAdvancedTracking(UseAdvancedTracking), MRI(MRI), TII(TII) {
      if (!TargetRegisterInfo::isPhysicalRegister(Reg)) {
        Def = MRI.getVRegDef(Reg);
        DefIdx = MRI.def_begin(Reg).getOperandNo();
      }
    }

    /// \brief Create a ValueTracker instance for the value defined by
    /// the pair \p MI, \p DefIdx.
    /// Unlike the other constructor, the value tracker produced by this one
    /// may be able to find a new source when the definition is a physical
    /// register.
    /// This could be useful to rewrite target specific instructions into
    /// generic copy instructions.
    ValueTracker(const MachineInstr &MI, unsigned DefIdx, unsigned DefSubReg,
                 const MachineRegisterInfo &MRI,
                 bool UseAdvancedTracking = false,
                 const TargetInstrInfo *TII = nullptr)
        : Def(&MI), DefIdx(DefIdx), DefSubReg(DefSubReg),
          UseAdvancedTracking(UseAdvancedTracking), MRI(MRI), TII(TII) {
      assert(DefIdx < Def->getDesc().getNumDefs() &&
             Def->getOperand(DefIdx).isReg() && "Invalid definition");
      Reg = Def->getOperand(DefIdx).getReg();
    }

    /// \brief Following the use-def chain, get the next available source
    /// for the tracked value.
    /// When the returned value is not nullptr, \p SrcReg gives the register
    /// that contain the tracked value.
    /// \note The sub register index returned in \p SrcSubReg must be used
    /// on \p SrcReg to access the actual value.
    /// \return Unless the returned value is nullptr (i.e., no source found),
    /// \p SrcReg gives the register of the next source used in the returned
    /// instruction and \p SrcSubReg the sub-register index to be used on that
    /// source to get the tracked value. When nullptr is returned, no
    /// alternative source has been found.
    const MachineInstr *getNextSource(unsigned &SrcReg, unsigned &SrcSubReg);

    /// \brief Get the last register where the initial value can be found.
    /// Initially this is the register of the definition.
    /// Then, after each successful call to getNextSource, this is the
    /// register of the last source.
    unsigned getReg() const { return Reg; }
  };
}

char PeepholeOptimizer::ID = 0;
char &llvm::PeepholeOptimizerID = PeepholeOptimizer::ID;
INITIALIZE_PASS_BEGIN(PeepholeOptimizer, "peephole-opts",
                "Peephole Optimizations", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_END(PeepholeOptimizer, "peephole-opts",
                "Peephole Optimizations", false, false)

/// optimizeExtInstr - If instruction is a copy-like instruction, i.e. it reads
/// a single register and writes a single register and it does not modify the
/// source, and if the source value is preserved as a sub-register of the
/// result, then replace all reachable uses of the source with the subreg of the
/// result.
///
/// Do not generate an EXTRACT that is used only in a debug use, as this changes
/// the code. Since this code does not currently share EXTRACTs, just ignore all
/// debug uses.
bool PeepholeOptimizer::
optimizeExtInstr(MachineInstr *MI, MachineBasicBlock *MBB,
                 SmallPtrSetImpl<MachineInstr*> &LocalMIs) {
  unsigned SrcReg, DstReg, SubIdx;
  if (!TII->isCoalescableExtInstr(*MI, SrcReg, DstReg, SubIdx))
    return false;

  if (TargetRegisterInfo::isPhysicalRegister(DstReg) ||
      TargetRegisterInfo::isPhysicalRegister(SrcReg))
    return false;

  if (MRI->hasOneNonDBGUse(SrcReg))
    // No other uses.
    return false;

  // Ensure DstReg can get a register class that actually supports
  // sub-registers. Don't change the class until we commit.
  const TargetRegisterClass *DstRC = MRI->getRegClass(DstReg);
  DstRC = TRI->getSubClassWithSubReg(DstRC, SubIdx);
  if (!DstRC)
    return false;

  // The ext instr may be operating on a sub-register of SrcReg as well.
  // PPC::EXTSW is a 32 -> 64-bit sign extension, but it reads a 64-bit
  // register.
  // If UseSrcSubIdx is Set, SubIdx also applies to SrcReg, and only uses of
  // SrcReg:SubIdx should be replaced.
  bool UseSrcSubIdx =
      TRI->getSubClassWithSubReg(MRI->getRegClass(SrcReg), SubIdx) != nullptr;

  // The source has other uses. See if we can replace the other uses with use of
  // the result of the extension.
  SmallPtrSet<MachineBasicBlock*, 4> ReachedBBs;
  for (MachineInstr &UI : MRI->use_nodbg_instructions(DstReg))
    ReachedBBs.insert(UI.getParent());

  // Uses that are in the same BB of uses of the result of the instruction.
  SmallVector<MachineOperand*, 8> Uses;

  // Uses that the result of the instruction can reach.
  SmallVector<MachineOperand*, 8> ExtendedUses;

  bool ExtendLife = true;
  for (MachineOperand &UseMO : MRI->use_nodbg_operands(SrcReg)) {
    MachineInstr *UseMI = UseMO.getParent();
    if (UseMI == MI)
      continue;

    if (UseMI->isPHI()) {
      ExtendLife = false;
      continue;
    }

    // Only accept uses of SrcReg:SubIdx.
    if (UseSrcSubIdx && UseMO.getSubReg() != SubIdx)
      continue;

    // It's an error to translate this:
    //
    //    %reg1025 = <sext> %reg1024
    //     ...
    //    %reg1026 = SUBREG_TO_REG 0, %reg1024, 4
    //
    // into this:
    //
    //    %reg1025 = <sext> %reg1024
    //     ...
    //    %reg1027 = COPY %reg1025:4
    //    %reg1026 = SUBREG_TO_REG 0, %reg1027, 4
    //
    // The problem here is that SUBREG_TO_REG is there to assert that an
    // implicit zext occurs. It doesn't insert a zext instruction. If we allow
    // the COPY here, it will give us the value after the <sext>, not the
    // original value of %reg1024 before <sext>.
    if (UseMI->getOpcode() == TargetOpcode::SUBREG_TO_REG)
      continue;

    MachineBasicBlock *UseMBB = UseMI->getParent();
    if (UseMBB == MBB) {
      // Local uses that come after the extension.
      if (!LocalMIs.count(UseMI))
        Uses.push_back(&UseMO);
    } else if (ReachedBBs.count(UseMBB)) {
      // Non-local uses where the result of the extension is used. Always
      // replace these unless it's a PHI.
      Uses.push_back(&UseMO);
    } else if (Aggressive && DT->dominates(MBB, UseMBB)) {
      // We may want to extend the live range of the extension result in order
      // to replace these uses.
      ExtendedUses.push_back(&UseMO);
    } else {
      // Both will be live out of the def MBB anyway. Don't extend live range of
      // the extension result.
      ExtendLife = false;
      break;
    }
  }

  if (ExtendLife && !ExtendedUses.empty())
    // Extend the liveness of the extension result.
    Uses.append(ExtendedUses.begin(), ExtendedUses.end());

  // Now replace all uses.
  bool Changed = false;
  if (!Uses.empty()) {
    SmallPtrSet<MachineBasicBlock*, 4> PHIBBs;

    // Look for PHI uses of the extended result, we don't want to extend the
    // liveness of a PHI input. It breaks all kinds of assumptions down
    // stream. A PHI use is expected to be the kill of its source values.
    for (MachineInstr &UI : MRI->use_nodbg_instructions(DstReg))
      if (UI.isPHI())
        PHIBBs.insert(UI.getParent());

    const TargetRegisterClass *RC = MRI->getRegClass(SrcReg);
    for (unsigned i = 0, e = Uses.size(); i != e; ++i) {
      MachineOperand *UseMO = Uses[i];
      MachineInstr *UseMI = UseMO->getParent();
      MachineBasicBlock *UseMBB = UseMI->getParent();
      if (PHIBBs.count(UseMBB))
        continue;

      // About to add uses of DstReg, clear DstReg's kill flags.
      if (!Changed) {
        MRI->clearKillFlags(DstReg);
        MRI->constrainRegClass(DstReg, DstRC);
      }

      unsigned NewVR = MRI->createVirtualRegister(RC);
      MachineInstr *Copy = BuildMI(*UseMBB, UseMI, UseMI->getDebugLoc(),
                                   TII->get(TargetOpcode::COPY), NewVR)
        .addReg(DstReg, 0, SubIdx);
      // SubIdx applies to both SrcReg and DstReg when UseSrcSubIdx is set.
      if (UseSrcSubIdx) {
        Copy->getOperand(0).setSubReg(SubIdx);
        Copy->getOperand(0).setIsUndef();
      }
      UseMO->setReg(NewVR);
      ++NumReuse;
      Changed = true;
    }
  }

  return Changed;
}

/// optimizeCmpInstr - If the instruction is a compare and the previous
/// instruction it's comparing against all ready sets (or could be modified to
/// set) the same flag as the compare, then we can remove the comparison and use
/// the flag from the previous instruction.
bool PeepholeOptimizer::optimizeCmpInstr(MachineInstr *MI,
                                         MachineBasicBlock *MBB) {
  // If this instruction is a comparison against zero and isn't comparing a
  // physical register, we can try to optimize it.
  unsigned SrcReg, SrcReg2;
  int CmpMask, CmpValue;
  if (!TII->analyzeCompare(MI, SrcReg, SrcReg2, CmpMask, CmpValue) ||
      TargetRegisterInfo::isPhysicalRegister(SrcReg) ||
      (SrcReg2 != 0 && TargetRegisterInfo::isPhysicalRegister(SrcReg2)))
    return false;

  // Attempt to optimize the comparison instruction.
  if (TII->optimizeCompareInstr(MI, SrcReg, SrcReg2, CmpMask, CmpValue, MRI)) {
    ++NumCmps;
    return true;
  }

  return false;
}

/// Optimize a select instruction.
bool PeepholeOptimizer::optimizeSelect(MachineInstr *MI,
                            SmallPtrSetImpl<MachineInstr *> &LocalMIs) {
  unsigned TrueOp = 0;
  unsigned FalseOp = 0;
  bool Optimizable = false;
  SmallVector<MachineOperand, 4> Cond;
  if (TII->analyzeSelect(MI, Cond, TrueOp, FalseOp, Optimizable))
    return false;
  if (!Optimizable)
    return false;
  if (!TII->optimizeSelect(MI, LocalMIs))
    return false;
  MI->eraseFromParent();
  ++NumSelects;
  return true;
}

/// \brief Check if a simpler conditional branch can be
// generated
bool PeepholeOptimizer::optimizeCondBranch(MachineInstr *MI) {
  return TII->optimizeCondBranch(MI);
}

/// \brief Check if the registers defined by the pair (RegisterClass, SubReg)
/// share the same register file.
static bool shareSameRegisterFile(const TargetRegisterInfo &TRI,
                                  const TargetRegisterClass *DefRC,
                                  unsigned DefSubReg,
                                  const TargetRegisterClass *SrcRC,
                                  unsigned SrcSubReg) {
  // Same register class.
  if (DefRC == SrcRC)
    return true;

  // Both operands are sub registers. Check if they share a register class.
  unsigned SrcIdx, DefIdx;
  if (SrcSubReg && DefSubReg)
    return TRI.getCommonSuperRegClass(SrcRC, SrcSubReg, DefRC, DefSubReg,
                                      SrcIdx, DefIdx) != nullptr;
  // At most one of the register is a sub register, make it Src to avoid
  // duplicating the test.
  if (!SrcSubReg) {
    std::swap(DefSubReg, SrcSubReg);
    std::swap(DefRC, SrcRC);
  }

  // One of the register is a sub register, check if we can get a superclass.
  if (SrcSubReg)
    return TRI.getMatchingSuperRegClass(SrcRC, DefRC, SrcSubReg) != nullptr;
  // Plain copy.
  return TRI.getCommonSubClass(DefRC, SrcRC) != nullptr;
}

/// \brief Try to find the next source that share the same register file
/// for the value defined by \p Reg and \p SubReg.
/// When true is returned, \p Reg and \p SubReg are updated with the
/// register number and sub-register index of the new source.
/// \return False if no alternative sources are available. True otherwise.
bool PeepholeOptimizer::findNextSource(unsigned &Reg, unsigned &SubReg) {
  // Do not try to find a new source for a physical register.
  // So far we do not have any motivating example for doing that.
  // Thus, instead of maintaining untested code, we will revisit that if
  // that changes at some point.
  if (TargetRegisterInfo::isPhysicalRegister(Reg))
    return false;

  const TargetRegisterClass *DefRC = MRI->getRegClass(Reg);
  unsigned DefSubReg = SubReg;

  unsigned Src;
  unsigned SrcSubReg;
  bool ShouldRewrite = false;

  // Follow the chain of copies until we reach the top of the use-def chain
  // or find a more suitable source.
  ValueTracker ValTracker(Reg, DefSubReg, *MRI, !DisableAdvCopyOpt, TII);
  do {
    unsigned CopySrcReg, CopySrcSubReg;
    if (!ValTracker.getNextSource(CopySrcReg, CopySrcSubReg))
      break;
    Src = CopySrcReg;
    SrcSubReg = CopySrcSubReg;

    // Do not extend the live-ranges of physical registers as they add
    // constraints to the register allocator.
    // Moreover, if we want to extend the live-range of a physical register,
    // unlike SSA virtual register, we will have to check that they are not
    // redefine before the related use.
    if (TargetRegisterInfo::isPhysicalRegister(Src))
      break;

    const TargetRegisterClass *SrcRC = MRI->getRegClass(Src);

    // If this source does not incur a cross register bank copy, use it.
    ShouldRewrite = shareSameRegisterFile(*TRI, DefRC, DefSubReg, SrcRC,
                                          SrcSubReg);
  } while (!ShouldRewrite);

  // If we did not find a more suitable source, there is nothing to optimize.
  if (!ShouldRewrite || Src == Reg)
    return false;

  Reg = Src;
  SubReg = SrcSubReg;
  return true;
}

namespace {
/// \brief Helper class to rewrite the arguments of a copy-like instruction.
class CopyRewriter {
protected:
  /// The copy-like instruction.
  MachineInstr &CopyLike;
  /// The index of the source being rewritten.
  unsigned CurrentSrcIdx;

public:
  CopyRewriter(MachineInstr &MI) : CopyLike(MI), CurrentSrcIdx(0) {}

  virtual ~CopyRewriter() {}

  /// \brief Get the next rewritable source (SrcReg, SrcSubReg) and
  /// the related value that it affects (TrackReg, TrackSubReg).
  /// A source is considered rewritable if its register class and the
  /// register class of the related TrackReg may not be register
  /// coalescer friendly. In other words, given a copy-like instruction
  /// not all the arguments may be returned at rewritable source, since
  /// some arguments are none to be register coalescer friendly.
  ///
  /// Each call of this method moves the current source to the next
  /// rewritable source.
  /// For instance, let CopyLike be the instruction to rewrite.
  /// CopyLike has one definition and one source:
  /// dst.dstSubIdx = CopyLike src.srcSubIdx.
  ///
  /// The first call will give the first rewritable source, i.e.,
  /// the only source this instruction has:
  /// (SrcReg, SrcSubReg) = (src, srcSubIdx).
  /// This source defines the whole definition, i.e.,
  /// (TrackReg, TrackSubReg) = (dst, dstSubIdx).
  ///
  /// The second and subsequent calls will return false, has there is only one
  /// rewritable source.
  ///
  /// \return True if a rewritable source has been found, false otherwise.
  /// The output arguments are valid if and only if true is returned.
  virtual bool getNextRewritableSource(unsigned &SrcReg, unsigned &SrcSubReg,
                                       unsigned &TrackReg,
                                       unsigned &TrackSubReg) {
    // If CurrentSrcIdx == 1, this means this function has already been
    // called once. CopyLike has one defintiion and one argument, thus,
    // there is nothing else to rewrite.
    if (!CopyLike.isCopy() || CurrentSrcIdx == 1)
      return false;
    // This is the first call to getNextRewritableSource.
    // Move the CurrentSrcIdx to remember that we made that call.
    CurrentSrcIdx = 1;
    // The rewritable source is the argument.
    const MachineOperand &MOSrc = CopyLike.getOperand(1);
    SrcReg = MOSrc.getReg();
    SrcSubReg = MOSrc.getSubReg();
    // What we track are the alternative sources of the definition.
    const MachineOperand &MODef = CopyLike.getOperand(0);
    TrackReg = MODef.getReg();
    TrackSubReg = MODef.getSubReg();
    return true;
  }

  /// \brief Rewrite the current source with \p NewReg and \p NewSubReg
  /// if possible.
  /// \return True if the rewritting was possible, false otherwise.
  virtual bool RewriteCurrentSource(unsigned NewReg, unsigned NewSubReg) {
    if (!CopyLike.isCopy() || CurrentSrcIdx != 1)
      return false;
    MachineOperand &MOSrc = CopyLike.getOperand(CurrentSrcIdx);
    MOSrc.setReg(NewReg);
    MOSrc.setSubReg(NewSubReg);
    return true;
  }
};

/// \brief Specialized rewriter for INSERT_SUBREG instruction.
class InsertSubregRewriter : public CopyRewriter {
public:
  InsertSubregRewriter(MachineInstr &MI) : CopyRewriter(MI) {
    assert(MI.isInsertSubreg() && "Invalid instruction");
  }

  /// \brief See CopyRewriter::getNextRewritableSource.
  /// Here CopyLike has the following form:
  /// dst = INSERT_SUBREG Src1, Src2.src2SubIdx, subIdx.
  /// Src1 has the same register class has dst, hence, there is
  /// nothing to rewrite.
  /// Src2.src2SubIdx, may not be register coalescer friendly.
  /// Therefore, the first call to this method returns:
  /// (SrcReg, SrcSubReg) = (Src2, src2SubIdx).
  /// (TrackReg, TrackSubReg) = (dst, subIdx).
  ///
  /// Subsequence calls will return false.
  bool getNextRewritableSource(unsigned &SrcReg, unsigned &SrcSubReg,
                               unsigned &TrackReg,
                               unsigned &TrackSubReg) override {
    // If we already get the only source we can rewrite, return false.
    if (CurrentSrcIdx == 2)
      return false;
    // We are looking at v2 = INSERT_SUBREG v0, v1, sub0.
    CurrentSrcIdx = 2;
    const MachineOperand &MOInsertedReg = CopyLike.getOperand(2);
    SrcReg = MOInsertedReg.getReg();
    SrcSubReg = MOInsertedReg.getSubReg();
    const MachineOperand &MODef = CopyLike.getOperand(0);

    // We want to track something that is compatible with the
    // partial definition.
    TrackReg = MODef.getReg();
    if (MODef.getSubReg())
      // Bails if we have to compose sub-register indices.
      return false;
    TrackSubReg = (unsigned)CopyLike.getOperand(3).getImm();
    return true;
  }
  bool RewriteCurrentSource(unsigned NewReg, unsigned NewSubReg) override {
    if (CurrentSrcIdx != 2)
      return false;
    // We are rewriting the inserted reg.
    MachineOperand &MO = CopyLike.getOperand(CurrentSrcIdx);
    MO.setReg(NewReg);
    MO.setSubReg(NewSubReg);
    return true;
  }
};

/// \brief Specialized rewriter for EXTRACT_SUBREG instruction.
class ExtractSubregRewriter : public CopyRewriter {
  const TargetInstrInfo &TII;

public:
  ExtractSubregRewriter(MachineInstr &MI, const TargetInstrInfo &TII)
      : CopyRewriter(MI), TII(TII) {
    assert(MI.isExtractSubreg() && "Invalid instruction");
  }

  /// \brief See CopyRewriter::getNextRewritableSource.
  /// Here CopyLike has the following form:
  /// dst.dstSubIdx = EXTRACT_SUBREG Src, subIdx.
  /// There is only one rewritable source: Src.subIdx,
  /// which defines dst.dstSubIdx.
  bool getNextRewritableSource(unsigned &SrcReg, unsigned &SrcSubReg,
                               unsigned &TrackReg,
                               unsigned &TrackSubReg) override {
    // If we already get the only source we can rewrite, return false.
    if (CurrentSrcIdx == 1)
      return false;
    // We are looking at v1 = EXTRACT_SUBREG v0, sub0.
    CurrentSrcIdx = 1;
    const MachineOperand &MOExtractedReg = CopyLike.getOperand(1);
    SrcReg = MOExtractedReg.getReg();
    // If we have to compose sub-register indices, bails out.
    if (MOExtractedReg.getSubReg())
      return false;

    SrcSubReg = CopyLike.getOperand(2).getImm();

    // We want to track something that is compatible with the definition.
    const MachineOperand &MODef = CopyLike.getOperand(0);
    TrackReg = MODef.getReg();
    TrackSubReg = MODef.getSubReg();
    return true;
  }

  bool RewriteCurrentSource(unsigned NewReg, unsigned NewSubReg) override {
    // The only source we can rewrite is the input register.
    if (CurrentSrcIdx != 1)
      return false;

    CopyLike.getOperand(CurrentSrcIdx).setReg(NewReg);

    // If we find a source that does not require to extract something,
    // rewrite the operation with a copy.
    if (!NewSubReg) {
      // Move the current index to an invalid position.
      // We do not want another call to this method to be able
      // to do any change.
      CurrentSrcIdx = -1;
      // Rewrite the operation as a COPY.
      // Get rid of the sub-register index.
      CopyLike.RemoveOperand(2);
      // Morph the operation into a COPY.
      CopyLike.setDesc(TII.get(TargetOpcode::COPY));
      return true;
    }
    CopyLike.getOperand(CurrentSrcIdx + 1).setImm(NewSubReg);
    return true;
  }
};

/// \brief Specialized rewriter for REG_SEQUENCE instruction.
class RegSequenceRewriter : public CopyRewriter {
public:
  RegSequenceRewriter(MachineInstr &MI) : CopyRewriter(MI) {
    assert(MI.isRegSequence() && "Invalid instruction");
  }

  /// \brief See CopyRewriter::getNextRewritableSource.
  /// Here CopyLike has the following form:
  /// dst = REG_SEQUENCE Src1.src1SubIdx, subIdx1, Src2.src2SubIdx, subIdx2.
  /// Each call will return a different source, walking all the available
  /// source.
  ///
  /// The first call returns:
  /// (SrcReg, SrcSubReg) = (Src1, src1SubIdx).
  /// (TrackReg, TrackSubReg) = (dst, subIdx1).
  ///
  /// The second call returns:
  /// (SrcReg, SrcSubReg) = (Src2, src2SubIdx).
  /// (TrackReg, TrackSubReg) = (dst, subIdx2).
  ///
  /// And so on, until all the sources have been traversed, then
  /// it returns false.
  bool getNextRewritableSource(unsigned &SrcReg, unsigned &SrcSubReg,
                               unsigned &TrackReg,
                               unsigned &TrackSubReg) override {
    // We are looking at v0 = REG_SEQUENCE v1, sub1, v2, sub2, etc.

    // If this is the first call, move to the first argument.
    if (CurrentSrcIdx == 0) {
      CurrentSrcIdx = 1;
    } else {
      // Otherwise, move to the next argument and check that it is valid.
      CurrentSrcIdx += 2;
      if (CurrentSrcIdx >= CopyLike.getNumOperands())
        return false;
    }
    const MachineOperand &MOInsertedReg = CopyLike.getOperand(CurrentSrcIdx);
    SrcReg = MOInsertedReg.getReg();
    // If we have to compose sub-register indices, bails out.
    if ((SrcSubReg = MOInsertedReg.getSubReg()))
      return false;

    // We want to track something that is compatible with the related
    // partial definition.
    TrackSubReg = CopyLike.getOperand(CurrentSrcIdx + 1).getImm();

    const MachineOperand &MODef = CopyLike.getOperand(0);
    TrackReg = MODef.getReg();
    // If we have to compose sub-registers, bails.
    return MODef.getSubReg() == 0;
  }

  bool RewriteCurrentSource(unsigned NewReg, unsigned NewSubReg) override {
    // We cannot rewrite out of bound operands.
    // Moreover, rewritable sources are at odd positions.
    if ((CurrentSrcIdx & 1) != 1 || CurrentSrcIdx > CopyLike.getNumOperands())
      return false;

    MachineOperand &MO = CopyLike.getOperand(CurrentSrcIdx);
    MO.setReg(NewReg);
    MO.setSubReg(NewSubReg);
    return true;
  }
};
} // End namespace.

/// \brief Get the appropriated CopyRewriter for \p MI.
/// \return A pointer to a dynamically allocated CopyRewriter or nullptr
/// if no rewriter works for \p MI.
static CopyRewriter *getCopyRewriter(MachineInstr &MI,
                                     const TargetInstrInfo &TII) {
  switch (MI.getOpcode()) {
  default:
    return nullptr;
  case TargetOpcode::COPY:
    return new CopyRewriter(MI);
  case TargetOpcode::INSERT_SUBREG:
    return new InsertSubregRewriter(MI);
  case TargetOpcode::EXTRACT_SUBREG:
    return new ExtractSubregRewriter(MI, TII);
  case TargetOpcode::REG_SEQUENCE:
    return new RegSequenceRewriter(MI);
  }
  llvm_unreachable(nullptr);
}

/// \brief Optimize generic copy instructions to avoid cross
/// register bank copy. The optimization looks through a chain of
/// copies and tries to find a source that has a compatible register
/// class.
/// Two register classes are considered to be compatible if they share
/// the same register bank.
/// New copies issued by this optimization are register allocator
/// friendly. This optimization does not remove any copy as it may
/// overconstraint the register allocator, but replaces some operands
/// when possible.
/// \pre isCoalescableCopy(*MI) is true.
/// \return True, when \p MI has been rewritten. False otherwise.
bool PeepholeOptimizer::optimizeCoalescableCopy(MachineInstr *MI) {
  assert(MI && isCoalescableCopy(*MI) && "Invalid argument");
  assert(MI->getDesc().getNumDefs() == 1 &&
         "Coalescer can understand multiple defs?!");
  const MachineOperand &MODef = MI->getOperand(0);
  // Do not rewrite physical definitions.
  if (TargetRegisterInfo::isPhysicalRegister(MODef.getReg()))
    return false;

  bool Changed = false;
  // Get the right rewriter for the current copy.
  std::unique_ptr<CopyRewriter> CpyRewriter(getCopyRewriter(*MI, *TII));
  // If none exists, bails out.
  if (!CpyRewriter)
    return false;
  // Rewrite each rewritable source.
  unsigned SrcReg, SrcSubReg, TrackReg, TrackSubReg;
  while (CpyRewriter->getNextRewritableSource(SrcReg, SrcSubReg, TrackReg,
                                              TrackSubReg)) {
    unsigned NewSrc = TrackReg;
    unsigned NewSubReg = TrackSubReg;
    // Try to find a more suitable source.
    // If we failed to do so, or get the actual source,
    // move to the next source.
    if (!findNextSource(NewSrc, NewSubReg) || SrcReg == NewSrc)
      continue;
    // Rewrite source.
    if (CpyRewriter->RewriteCurrentSource(NewSrc, NewSubReg)) {
      // We may have extended the live-range of NewSrc, account for that.
      MRI->clearKillFlags(NewSrc);
      Changed = true;
    }
  }
  // TODO: We could have a clean-up method to tidy the instruction.
  // E.g., v0 = INSERT_SUBREG v1, v1.sub0, sub0
  // => v0 = COPY v1
  // Currently we haven't seen motivating example for that and we
  // want to avoid untested code.
  NumRewrittenCopies += Changed;
  return Changed;
}

/// \brief Optimize copy-like instructions to create
/// register coalescer friendly instruction.
/// The optimization tries to kill-off the \p MI by looking
/// through a chain of copies to find a source that has a compatible
/// register class.
/// If such a source is found, it replace \p MI by a generic COPY
/// operation.
/// \pre isUncoalescableCopy(*MI) is true.
/// \return True, when \p MI has been optimized. In that case, \p MI has
/// been removed from its parent.
/// All COPY instructions created, are inserted in \p LocalMIs.
bool PeepholeOptimizer::optimizeUncoalescableCopy(
    MachineInstr *MI, SmallPtrSetImpl<MachineInstr *> &LocalMIs) {
  assert(MI && isUncoalescableCopy(*MI) && "Invalid argument");

  // Check if we can rewrite all the values defined by this instruction.
  SmallVector<
      std::pair<TargetInstrInfo::RegSubRegPair, TargetInstrInfo::RegSubRegPair>,
      4> RewritePairs;
  for (const MachineOperand &MODef : MI->defs()) {
    if (MODef.isDead())
      // We can ignore those.
      continue;

    // If a physical register is here, this is probably for a good reason.
    // Do not rewrite that.
    if (TargetRegisterInfo::isPhysicalRegister(MODef.getReg()))
      return false;

    // If we do not know how to rewrite this definition, there is no point
    // in trying to kill this instruction.
    TargetInstrInfo::RegSubRegPair Def(MODef.getReg(), MODef.getSubReg());
    TargetInstrInfo::RegSubRegPair Src = Def;
    if (!findNextSource(Src.Reg, Src.SubReg))
      return false;
    RewritePairs.push_back(std::make_pair(Def, Src));
  }
  // The change is possible for all defs, do it.
  for (const auto &PairDefSrc : RewritePairs) {
    const auto &Def = PairDefSrc.first;
    const auto &Src = PairDefSrc.second;
    // Rewrite the "copy" in a way the register coalescer understands.
    assert(!TargetRegisterInfo::isPhysicalRegister(Def.Reg) &&
           "We do not rewrite physical registers");
    const TargetRegisterClass *DefRC = MRI->getRegClass(Def.Reg);
    unsigned NewVR = MRI->createVirtualRegister(DefRC);
    MachineInstr *NewCopy = BuildMI(*MI->getParent(), MI, MI->getDebugLoc(),
                                    TII->get(TargetOpcode::COPY),
                                    NewVR).addReg(Src.Reg, 0, Src.SubReg);
    NewCopy->getOperand(0).setSubReg(Def.SubReg);
    if (Def.SubReg)
      NewCopy->getOperand(0).setIsUndef();
    LocalMIs.insert(NewCopy);
    MRI->replaceRegWith(Def.Reg, NewVR);
    MRI->clearKillFlags(NewVR);
    // We extended the lifetime of Src.
    // Clear the kill flags to account for that.
    MRI->clearKillFlags(Src.Reg);
  }
  // MI is now dead.
  MI->eraseFromParent();
  ++NumUncoalescableCopies;
  return true;
}

/// isLoadFoldable - Check whether MI is a candidate for folding into a later
/// instruction. We only fold loads to virtual registers and the virtual
/// register defined has a single use.
bool PeepholeOptimizer::isLoadFoldable(
                              MachineInstr *MI,
                              SmallSet<unsigned, 16> &FoldAsLoadDefCandidates) {
  if (!MI->canFoldAsLoad() || !MI->mayLoad())
    return false;
  const MCInstrDesc &MCID = MI->getDesc();
  if (MCID.getNumDefs() != 1)
    return false;

  unsigned Reg = MI->getOperand(0).getReg();
  // To reduce compilation time, we check MRI->hasOneNonDBGUse when inserting
  // loads. It should be checked when processing uses of the load, since
  // uses can be removed during peephole.
  if (!MI->getOperand(0).getSubReg() &&
      TargetRegisterInfo::isVirtualRegister(Reg) &&
      MRI->hasOneNonDBGUse(Reg)) {
    FoldAsLoadDefCandidates.insert(Reg);
    return true;
  }
  return false;
}

bool PeepholeOptimizer::isMoveImmediate(MachineInstr *MI,
                                        SmallSet<unsigned, 4> &ImmDefRegs,
                                 DenseMap<unsigned, MachineInstr*> &ImmDefMIs) {
  const MCInstrDesc &MCID = MI->getDesc();
  if (!MI->isMoveImmediate())
    return false;
  if (MCID.getNumDefs() != 1)
    return false;
  unsigned Reg = MI->getOperand(0).getReg();
  if (TargetRegisterInfo::isVirtualRegister(Reg)) {
    ImmDefMIs.insert(std::make_pair(Reg, MI));
    ImmDefRegs.insert(Reg);
    return true;
  }

  return false;
}

/// foldImmediate - Try folding register operands that are defined by move
/// immediate instructions, i.e. a trivial constant folding optimization, if
/// and only if the def and use are in the same BB.
bool PeepholeOptimizer::foldImmediate(MachineInstr *MI, MachineBasicBlock *MBB,
                                      SmallSet<unsigned, 4> &ImmDefRegs,
                                 DenseMap<unsigned, MachineInstr*> &ImmDefMIs) {
  for (unsigned i = 0, e = MI->getDesc().getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || MO.isDef())
      continue;
    unsigned Reg = MO.getReg();
    if (!TargetRegisterInfo::isVirtualRegister(Reg))
      continue;
    if (ImmDefRegs.count(Reg) == 0)
      continue;
    DenseMap<unsigned, MachineInstr*>::iterator II = ImmDefMIs.find(Reg);
    assert(II != ImmDefMIs.end());
    if (TII->FoldImmediate(MI, II->second, Reg, MRI)) {
      ++NumImmFold;
      return true;
    }
  }
  return false;
}

bool PeepholeOptimizer::runOnMachineFunction(MachineFunction &MF) {
  if (skipOptnoneFunction(*MF.getFunction()))
    return false;

  DEBUG(dbgs() << "********** PEEPHOLE OPTIMIZER **********\n");
  DEBUG(dbgs() << "********** Function: " << MF.getName() << '\n');

  if (DisablePeephole)
    return false;

  TII = MF.getSubtarget().getInstrInfo();
  TRI = MF.getSubtarget().getRegisterInfo();
  MRI = &MF.getRegInfo();
  DT  = Aggressive ? &getAnalysis<MachineDominatorTree>() : nullptr;

  bool Changed = false;

  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I) {
    MachineBasicBlock *MBB = &*I;

    bool SeenMoveImm = false;

    // During this forward scan, at some point it needs to answer the question
    // "given a pointer to an MI in the current BB, is it located before or
    // after the current instruction".
    // To perform this, the following set keeps track of the MIs already seen
    // during the scan, if a MI is not in the set, it is assumed to be located
    // after. Newly created MIs have to be inserted in the set as well.
    SmallPtrSet<MachineInstr*, 16> LocalMIs;
    SmallSet<unsigned, 4> ImmDefRegs;
    DenseMap<unsigned, MachineInstr*> ImmDefMIs;
    SmallSet<unsigned, 16> FoldAsLoadDefCandidates;

    for (MachineBasicBlock::iterator
           MII = I->begin(), MIE = I->end(); MII != MIE; ) {
      MachineInstr *MI = &*MII;
      // We may be erasing MI below, increment MII now.
      ++MII;
      LocalMIs.insert(MI);

      // Skip debug values. They should not affect this peephole optimization.
      if (MI->isDebugValue())
          continue;

      // If there exists an instruction which belongs to the following
      // categories, we will discard the load candidates.
      if (MI->isPosition() || MI->isPHI() || MI->isImplicitDef() ||
          MI->isKill() || MI->isInlineAsm() ||
          MI->hasUnmodeledSideEffects()) {
        FoldAsLoadDefCandidates.clear();
        continue;
      }
      if (MI->mayStore() || MI->isCall())
        FoldAsLoadDefCandidates.clear();

      if ((isUncoalescableCopy(*MI) &&
           optimizeUncoalescableCopy(MI, LocalMIs)) ||
          (MI->isCompare() && optimizeCmpInstr(MI, MBB)) ||
          (MI->isSelect() && optimizeSelect(MI, LocalMIs))) {
        // MI is deleted.
        LocalMIs.erase(MI);
        Changed = true;
        continue;
      }

      if (MI->isConditionalBranch() && optimizeCondBranch(MI)) {
        Changed = true;
        continue;
      }

      if (isCoalescableCopy(*MI) && optimizeCoalescableCopy(MI)) {
        // MI is just rewritten.
        Changed = true;
        continue;
      }

      if (isMoveImmediate(MI, ImmDefRegs, ImmDefMIs)) {
        SeenMoveImm = true;
      } else {
        Changed |= optimizeExtInstr(MI, MBB, LocalMIs);
        // optimizeExtInstr might have created new instructions after MI
        // and before the already incremented MII. Adjust MII so that the
        // next iteration sees the new instructions.
        MII = MI;
        ++MII;
        if (SeenMoveImm)
          Changed |= foldImmediate(MI, MBB, ImmDefRegs, ImmDefMIs);
      }

      // Check whether MI is a load candidate for folding into a later
      // instruction. If MI is not a candidate, check whether we can fold an
      // earlier load into MI.
      if (!isLoadFoldable(MI, FoldAsLoadDefCandidates) &&
          !FoldAsLoadDefCandidates.empty()) {
        const MCInstrDesc &MIDesc = MI->getDesc();
        for (unsigned i = MIDesc.getNumDefs(); i != MIDesc.getNumOperands();
             ++i) {
          const MachineOperand &MOp = MI->getOperand(i);
          if (!MOp.isReg())
            continue;
          unsigned FoldAsLoadDefReg = MOp.getReg();
          if (FoldAsLoadDefCandidates.count(FoldAsLoadDefReg)) {
            // We need to fold load after optimizeCmpInstr, since
            // optimizeCmpInstr can enable folding by converting SUB to CMP.
            // Save FoldAsLoadDefReg because optimizeLoadInstr() resets it and
            // we need it for markUsesInDebugValueAsUndef().
            unsigned FoldedReg = FoldAsLoadDefReg;
            MachineInstr *DefMI = nullptr;
            MachineInstr *FoldMI = TII->optimizeLoadInstr(MI, MRI,
                                                          FoldAsLoadDefReg,
                                                          DefMI);
            if (FoldMI) {
              // Update LocalMIs since we replaced MI with FoldMI and deleted
              // DefMI.
              DEBUG(dbgs() << "Replacing: " << *MI);
              DEBUG(dbgs() << "     With: " << *FoldMI);
              LocalMIs.erase(MI);
              LocalMIs.erase(DefMI);
              LocalMIs.insert(FoldMI);
              MI->eraseFromParent();
              DefMI->eraseFromParent();
              MRI->markUsesInDebugValueAsUndef(FoldedReg);
              FoldAsLoadDefCandidates.erase(FoldedReg);
              ++NumLoadFold;
              // MI is replaced with FoldMI.
              Changed = true;
              break;
            }
          }
        }
      }
    }
  }

  return Changed;
}

bool ValueTracker::getNextSourceFromCopy(unsigned &SrcReg,
                                         unsigned &SrcSubReg) {
  assert(Def->isCopy() && "Invalid definition");
  // Copy instruction are supposed to be: Def = Src.
  // If someone breaks this assumption, bad things will happen everywhere.
  assert(Def->getNumOperands() == 2 && "Invalid number of operands");

  if (Def->getOperand(DefIdx).getSubReg() != DefSubReg)
    // If we look for a different subreg, it means we want a subreg of src.
    // Bails as we do not support composing subreg yet.
    return false;
  // Otherwise, we want the whole source.
  const MachineOperand &Src = Def->getOperand(1);
  SrcReg = Src.getReg();
  SrcSubReg = Src.getSubReg();
  return true;
}

bool ValueTracker::getNextSourceFromBitcast(unsigned &SrcReg,
                                            unsigned &SrcSubReg) {
  assert(Def->isBitcast() && "Invalid definition");

  // Bail if there are effects that a plain copy will not expose.
  if (Def->hasUnmodeledSideEffects())
    return false;

  // Bitcasts with more than one def are not supported.
  if (Def->getDesc().getNumDefs() != 1)
    return false;
  if (Def->getOperand(DefIdx).getSubReg() != DefSubReg)
    // If we look for a different subreg, it means we want a subreg of the src.
    // Bails as we do not support composing subreg yet.
    return false;

  unsigned SrcIdx = Def->getNumOperands();
  for (unsigned OpIdx = DefIdx + 1, EndOpIdx = SrcIdx; OpIdx != EndOpIdx;
       ++OpIdx) {
    const MachineOperand &MO = Def->getOperand(OpIdx);
    if (!MO.isReg() || !MO.getReg())
      continue;
    assert(!MO.isDef() && "We should have skipped all the definitions by now");
    if (SrcIdx != EndOpIdx)
      // Multiple sources?
      return false;
    SrcIdx = OpIdx;
  }
  const MachineOperand &Src = Def->getOperand(SrcIdx);
  SrcReg = Src.getReg();
  SrcSubReg = Src.getSubReg();
  return true;
}

bool ValueTracker::getNextSourceFromRegSequence(unsigned &SrcReg,
                                                unsigned &SrcSubReg) {
  assert((Def->isRegSequence() || Def->isRegSequenceLike()) &&
         "Invalid definition");

  if (Def->getOperand(DefIdx).getSubReg())
    // If we are composing subreg, bails out.
    // The case we are checking is Def.<subreg> = REG_SEQUENCE.
    // This should almost never happen as the SSA property is tracked at
    // the register level (as opposed to the subreg level).
    // I.e.,
    // Def.sub0 =
    // Def.sub1 =
    // is a valid SSA representation for Def.sub0 and Def.sub1, but not for
    // Def. Thus, it must not be generated.
    // However, some code could theoretically generates a single
    // Def.sub0 (i.e, not defining the other subregs) and we would
    // have this case.
    // If we can ascertain (or force) that this never happens, we could
    // turn that into an assertion.
    return false;

  if (!TII)
    // We could handle the REG_SEQUENCE here, but we do not want to
    // duplicate the code from the generic TII.
    return false;

  SmallVector<TargetInstrInfo::RegSubRegPairAndIdx, 8> RegSeqInputRegs;
  if (!TII->getRegSequenceInputs(*Def, DefIdx, RegSeqInputRegs))
    return false;

  // We are looking at:
  // Def = REG_SEQUENCE v0, sub0, v1, sub1, ...
  // Check if one of the operand defines the subreg we are interested in.
  for (auto &RegSeqInput : RegSeqInputRegs) {
    if (RegSeqInput.SubIdx == DefSubReg) {
      if (RegSeqInput.SubReg)
        // Bails if we have to compose sub registers.
        return false;

      SrcReg = RegSeqInput.Reg;
      SrcSubReg = RegSeqInput.SubReg;
      return true;
    }
  }

  // If the subreg we are tracking is super-defined by another subreg,
  // we could follow this value. However, this would require to compose
  // the subreg and we do not do that for now.
  return false;
}

bool ValueTracker::getNextSourceFromInsertSubreg(unsigned &SrcReg,
                                                 unsigned &SrcSubReg) {
  assert((Def->isInsertSubreg() || Def->isInsertSubregLike()) &&
         "Invalid definition");

  if (Def->getOperand(DefIdx).getSubReg())
    // If we are composing subreg, bails out.
    // Same remark as getNextSourceFromRegSequence.
    // I.e., this may be turned into an assert.
    return false;

  if (!TII)
    // We could handle the REG_SEQUENCE here, but we do not want to
    // duplicate the code from the generic TII.
    return false;

  TargetInstrInfo::RegSubRegPair BaseReg;
  TargetInstrInfo::RegSubRegPairAndIdx InsertedReg;
  if (!TII->getInsertSubregInputs(*Def, DefIdx, BaseReg, InsertedReg))
    return false;

  // We are looking at:
  // Def = INSERT_SUBREG v0, v1, sub1
  // There are two cases:
  // 1. DefSubReg == sub1, get v1.
  // 2. DefSubReg != sub1, the value may be available through v0.

  // #1 Check if the inserted register matches the required sub index.
  if (InsertedReg.SubIdx == DefSubReg) {
    SrcReg = InsertedReg.Reg;
    SrcSubReg = InsertedReg.SubReg;
    return true;
  }
  // #2 Otherwise, if the sub register we are looking for is not partial
  // defined by the inserted element, we can look through the main
  // register (v0).
  const MachineOperand &MODef = Def->getOperand(DefIdx);
  // If the result register (Def) and the base register (v0) do not
  // have the same register class or if we have to compose
  // subregisters, bails out.
  if (MRI.getRegClass(MODef.getReg()) != MRI.getRegClass(BaseReg.Reg) ||
      BaseReg.SubReg)
    return false;

  // Get the TRI and check if the inserted sub-register overlaps with the
  // sub-register we are tracking.
  const TargetRegisterInfo *TRI = MRI.getTargetRegisterInfo();
  if (!TRI ||
      (TRI->getSubRegIndexLaneMask(DefSubReg) &
       TRI->getSubRegIndexLaneMask(InsertedReg.SubIdx)) != 0)
    return false;
  // At this point, the value is available in v0 via the same subreg
  // we used for Def.
  SrcReg = BaseReg.Reg;
  SrcSubReg = DefSubReg;
  return true;
}

bool ValueTracker::getNextSourceFromExtractSubreg(unsigned &SrcReg,
                                                  unsigned &SrcSubReg) {
  assert((Def->isExtractSubreg() ||
          Def->isExtractSubregLike()) && "Invalid definition");
  // We are looking at:
  // Def = EXTRACT_SUBREG v0, sub0

  // Bails if we have to compose sub registers.
  // Indeed, if DefSubReg != 0, we would have to compose it with sub0.
  if (DefSubReg)
    return false;

  if (!TII)
    // We could handle the EXTRACT_SUBREG here, but we do not want to
    // duplicate the code from the generic TII.
    return false;

  TargetInstrInfo::RegSubRegPairAndIdx ExtractSubregInputReg;
  if (!TII->getExtractSubregInputs(*Def, DefIdx, ExtractSubregInputReg))
    return false;

  // Bails if we have to compose sub registers.
  // Likewise, if v0.subreg != 0, we would have to compose v0.subreg with sub0.
  if (ExtractSubregInputReg.SubReg)
    return false;
  // Otherwise, the value is available in the v0.sub0.
  SrcReg = ExtractSubregInputReg.Reg;
  SrcSubReg = ExtractSubregInputReg.SubIdx;
  return true;
}

bool ValueTracker::getNextSourceFromSubregToReg(unsigned &SrcReg,
                                                unsigned &SrcSubReg) {
  assert(Def->isSubregToReg() && "Invalid definition");
  // We are looking at:
  // Def = SUBREG_TO_REG Imm, v0, sub0

  // Bails if we have to compose sub registers.
  // If DefSubReg != sub0, we would have to check that all the bits
  // we track are included in sub0 and if yes, we would have to
  // determine the right subreg in v0.
  if (DefSubReg != Def->getOperand(3).getImm())
    return false;
  // Bails if we have to compose sub registers.
  // Likewise, if v0.subreg != 0, we would have to compose it with sub0.
  if (Def->getOperand(2).getSubReg())
    return false;

  SrcReg = Def->getOperand(2).getReg();
  SrcSubReg = Def->getOperand(3).getImm();
  return true;
}

bool ValueTracker::getNextSourceImpl(unsigned &SrcReg, unsigned &SrcSubReg) {
  assert(Def && "This method needs a valid definition");

  assert(
      (DefIdx < Def->getDesc().getNumDefs() || Def->getDesc().isVariadic()) &&
      Def->getOperand(DefIdx).isDef() && "Invalid DefIdx");
  if (Def->isCopy())
    return getNextSourceFromCopy(SrcReg, SrcSubReg);
  if (Def->isBitcast())
    return getNextSourceFromBitcast(SrcReg, SrcSubReg);
  // All the remaining cases involve "complex" instructions.
  // Bails if we did not ask for the advanced tracking.
  if (!UseAdvancedTracking)
    return false;
  if (Def->isRegSequence() || Def->isRegSequenceLike())
    return getNextSourceFromRegSequence(SrcReg, SrcSubReg);
  if (Def->isInsertSubreg() || Def->isInsertSubregLike())
    return getNextSourceFromInsertSubreg(SrcReg, SrcSubReg);
  if (Def->isExtractSubreg() || Def->isExtractSubregLike())
    return getNextSourceFromExtractSubreg(SrcReg, SrcSubReg);
  if (Def->isSubregToReg())
    return getNextSourceFromSubregToReg(SrcReg, SrcSubReg);
  return false;
}

const MachineInstr *ValueTracker::getNextSource(unsigned &SrcReg,
                                                unsigned &SrcSubReg) {
  // If we reach a point where we cannot move up in the use-def chain,
  // there is nothing we can get.
  if (!Def)
    return nullptr;

  const MachineInstr *PrevDef = nullptr;
  // Try to find the next source.
  if (getNextSourceImpl(SrcReg, SrcSubReg)) {
    // Update definition, definition index, and subregister for the
    // next call of getNextSource.
    // Update the current register.
    Reg = SrcReg;
    // Update the return value before moving up in the use-def chain.
    PrevDef = Def;
    // If we can still move up in the use-def chain, move to the next
    // defintion.
    if (!TargetRegisterInfo::isPhysicalRegister(Reg)) {
      Def = MRI.getVRegDef(Reg);
      DefIdx = MRI.def_begin(Reg).getOperandNo();
      DefSubReg = SrcSubReg;
      return PrevDef;
    }
  }
  // If we end up here, this means we will not be able to find another source
  // for the next iteration.
  // Make sure any new call to getNextSource bails out early by cutting the
  // use-def chain.
  Def = nullptr;
  return PrevDef;
}
