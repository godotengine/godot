//===-- llvm/CodeGen/AsmPrinter/DbgValueHistoryCalculator.cpp -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DbgValueHistoryCalculator.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include <algorithm>
#include <map>
using namespace llvm;

#define DEBUG_TYPE "dwarfdebug"

// \brief If @MI is a DBG_VALUE with debug value described by a
// defined register, returns the number of this register.
// In the other case, returns 0.
static unsigned isDescribedByReg(const MachineInstr &MI) {
  assert(MI.isDebugValue());
  assert(MI.getNumOperands() == 4);
  // If location of variable is described using a register (directly or
  // indirecltly), this register is always a first operand.
  return MI.getOperand(0).isReg() ? MI.getOperand(0).getReg() : 0;
}

void DbgValueHistoryMap::startInstrRange(InlinedVariable Var,
                                         const MachineInstr &MI) {
  // Instruction range should start with a DBG_VALUE instruction for the
  // variable.
  assert(MI.isDebugValue() && "not a DBG_VALUE");
  auto &Ranges = VarInstrRanges[Var];
  if (!Ranges.empty() && Ranges.back().second == nullptr &&
      Ranges.back().first->isIdenticalTo(&MI)) {
    DEBUG(dbgs() << "Coalescing identical DBG_VALUE entries:\n"
                 << "\t" << Ranges.back().first << "\t" << MI << "\n");
    return;
  }
  Ranges.push_back(std::make_pair(&MI, nullptr));
}

void DbgValueHistoryMap::endInstrRange(InlinedVariable Var,
                                       const MachineInstr &MI) {
  auto &Ranges = VarInstrRanges[Var];
  // Verify that the current instruction range is not yet closed.
  assert(!Ranges.empty() && Ranges.back().second == nullptr);
  // For now, instruction ranges are not allowed to cross basic block
  // boundaries.
  assert(Ranges.back().first->getParent() == MI.getParent());
  Ranges.back().second = &MI;
}

unsigned DbgValueHistoryMap::getRegisterForVar(InlinedVariable Var) const {
  const auto &I = VarInstrRanges.find(Var);
  if (I == VarInstrRanges.end())
    return 0;
  const auto &Ranges = I->second;
  if (Ranges.empty() || Ranges.back().second != nullptr)
    return 0;
  return isDescribedByReg(*Ranges.back().first);
}

namespace {
// Maps physreg numbers to the variables they describe.
typedef DbgValueHistoryMap::InlinedVariable InlinedVariable;
typedef std::map<unsigned, SmallVector<InlinedVariable, 1>> RegDescribedVarsMap;
}

// \brief Claim that @Var is not described by @RegNo anymore.
static void dropRegDescribedVar(RegDescribedVarsMap &RegVars, unsigned RegNo,
                                InlinedVariable Var) {
  const auto &I = RegVars.find(RegNo);
  assert(RegNo != 0U && I != RegVars.end());
  auto &VarSet = I->second;
  const auto &VarPos = std::find(VarSet.begin(), VarSet.end(), Var);
  assert(VarPos != VarSet.end());
  VarSet.erase(VarPos);
  // Don't keep empty sets in a map to keep it as small as possible.
  if (VarSet.empty())
    RegVars.erase(I);
}

// \brief Claim that @Var is now described by @RegNo.
static void addRegDescribedVar(RegDescribedVarsMap &RegVars, unsigned RegNo,
                               InlinedVariable Var) {
  assert(RegNo != 0U);
  auto &VarSet = RegVars[RegNo];
  assert(std::find(VarSet.begin(), VarSet.end(), Var) == VarSet.end());
  VarSet.push_back(Var);
}

// \brief Terminate the location range for variables described by register at
// @I by inserting @ClobberingInstr to their history.
static void clobberRegisterUses(RegDescribedVarsMap &RegVars,
                                RegDescribedVarsMap::iterator I,
                                DbgValueHistoryMap &HistMap,
                                const MachineInstr &ClobberingInstr) {
  // Iterate over all variables described by this register and add this
  // instruction to their history, clobbering it.
  for (const auto &Var : I->second)
    HistMap.endInstrRange(Var, ClobberingInstr);
  RegVars.erase(I);
}

// \brief Terminate the location range for variables described by register
// @RegNo by inserting @ClobberingInstr to their history.
static void clobberRegisterUses(RegDescribedVarsMap &RegVars, unsigned RegNo,
                                DbgValueHistoryMap &HistMap,
                                const MachineInstr &ClobberingInstr) {
  const auto &I = RegVars.find(RegNo);
  if (I == RegVars.end())
    return;
  clobberRegisterUses(RegVars, I, HistMap, ClobberingInstr);
}

// \brief Collect all registers clobbered by @MI and apply the functor
// @Func to their RegNo.
// @Func should be a functor with a void(unsigned) signature. We're
// not using std::function here for performance reasons. It has a
// small but measurable impact. By using a functor instead of a
// std::set& here, we can avoid the overhead of constructing
// temporaries in calculateDbgValueHistory, which has a significant
// performance impact.
template<typename Callable>
static void applyToClobberedRegisters(const MachineInstr &MI,
                                      const TargetRegisterInfo *TRI,
                                      Callable Func) {
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg() || !MO.isDef() || !MO.getReg())
      continue;
    for (MCRegAliasIterator AI(MO.getReg(), TRI, true); AI.isValid(); ++AI)
      Func(*AI);
  }
}

// \brief Returns the first instruction in @MBB which corresponds to
// the function epilogue, or nullptr if @MBB doesn't contain an epilogue.
static const MachineInstr *getFirstEpilogueInst(const MachineBasicBlock &MBB) {
  auto LastMI = MBB.getLastNonDebugInstr();
  if (LastMI == MBB.end() || !LastMI->isReturn())
    return nullptr;
  // Assume that epilogue starts with instruction having the same debug location
  // as the return instruction.
  DebugLoc LastLoc = LastMI->getDebugLoc();
  auto Res = LastMI;
  for (MachineBasicBlock::const_reverse_iterator I(std::next(LastMI)),
       E = MBB.rend();
       I != E; ++I) {
    if (I->getDebugLoc() != LastLoc)
      return Res;
    Res = &*I;
  }
  // If all instructions have the same debug location, assume whole MBB is
  // an epilogue.
  return MBB.begin();
}

// \brief Collect registers that are modified in the function body (their
// contents is changed outside of the prologue and epilogue).
static void collectChangingRegs(const MachineFunction *MF,
                                const TargetRegisterInfo *TRI,
                                BitVector &Regs) {
  for (const auto &MBB : *MF) {
    auto FirstEpilogueInst = getFirstEpilogueInst(MBB);

    for (const auto &MI : MBB) {
      if (&MI == FirstEpilogueInst)
        break;
      if (!MI.getFlag(MachineInstr::FrameSetup))
        applyToClobberedRegisters(MI, TRI, [&](unsigned r) { Regs.set(r); });
    }
  }
}

void llvm::calculateDbgValueHistory(const MachineFunction *MF,
                                    const TargetRegisterInfo *TRI,
                                    DbgValueHistoryMap &Result) {
  BitVector ChangingRegs(TRI->getNumRegs());
  collectChangingRegs(MF, TRI, ChangingRegs);

  RegDescribedVarsMap RegVars;
  for (const auto &MBB : *MF) {
    for (const auto &MI : MBB) {
      if (!MI.isDebugValue()) {
        // Not a DBG_VALUE instruction. It may clobber registers which describe
        // some variables.
        applyToClobberedRegisters(MI, TRI, [&](unsigned RegNo) {
          if (ChangingRegs.test(RegNo))
            clobberRegisterUses(RegVars, RegNo, Result, MI);
        });
        continue;
      }

      assert(MI.getNumOperands() > 1 && "Invalid DBG_VALUE instruction!");
      // Use the base variable (without any DW_OP_piece expressions)
      // as index into History. The full variables including the
      // piece expressions are attached to the MI.
      const DILocalVariable *RawVar = MI.getDebugVariable();
      assert(RawVar->isValidLocationForIntrinsic(MI.getDebugLoc()) &&
             "Expected inlined-at fields to agree");
      InlinedVariable Var(RawVar, MI.getDebugLoc()->getInlinedAt());

      if (unsigned PrevReg = Result.getRegisterForVar(Var))
        dropRegDescribedVar(RegVars, PrevReg, Var);

      Result.startInstrRange(Var, MI);

      if (unsigned NewReg = isDescribedByReg(MI))
        addRegDescribedVar(RegVars, NewReg, Var);
    }

    // Make sure locations for register-described variables are valid only
    // until the end of the basic block (unless it's the last basic block, in
    // which case let their liveness run off to the end of the function).
    if (!MBB.empty() && &MBB != &MF->back()) {
      for (auto I = RegVars.begin(), E = RegVars.end(); I != E;) {
        auto CurElem = I++; // CurElem can be erased below.
        if (ChangingRegs.test(CurElem->first))
          clobberRegisterUses(RegVars, CurElem, Result, MBB.back());
      }
    }
  }
}
