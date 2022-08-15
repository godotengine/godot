//===---- ScheduleDAGInstrs.cpp - MachineInstr Rescheduling ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the ScheduleDAGInstrs class, which implements re-scheduling
// of MachineInstrs.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/RegisterPressure.h"
#include "llvm/CodeGen/ScheduleDFS.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include <queue>

using namespace llvm;

#define DEBUG_TYPE "misched"

static cl::opt<bool> EnableAASchedMI("enable-aa-sched-mi", cl::Hidden,
    cl::ZeroOrMore, cl::init(false),
    cl::desc("Enable use of AA during MI DAG construction"));

static cl::opt<bool> UseTBAA("use-tbaa-in-sched-mi", cl::Hidden,
    cl::init(true), cl::desc("Enable use of TBAA during MI DAG construction"));

ScheduleDAGInstrs::ScheduleDAGInstrs(MachineFunction &mf,
                                     const MachineLoopInfo *mli,
                                     bool IsPostRAFlag, bool RemoveKillFlags,
                                     LiveIntervals *lis)
    : ScheduleDAG(mf), MLI(mli), MFI(mf.getFrameInfo()), LIS(lis),
      IsPostRA(IsPostRAFlag), RemoveKillFlags(RemoveKillFlags),
      CanHandleTerminators(false), FirstDbgValue(nullptr) {
  assert((IsPostRA || LIS) && "PreRA scheduling requires LiveIntervals");
  DbgValues.clear();
  assert(!(IsPostRA && MRI.getNumVirtRegs()) &&
         "Virtual registers must be removed prior to PostRA scheduling");

  const TargetSubtargetInfo &ST = mf.getSubtarget();
  SchedModel.init(ST.getSchedModel(), &ST, TII);
}

/// getUnderlyingObjectFromInt - This is the function that does the work of
/// looking through basic ptrtoint+arithmetic+inttoptr sequences.
static const Value *getUnderlyingObjectFromInt(const Value *V) {
  do {
    if (const Operator *U = dyn_cast<Operator>(V)) {
      // If we find a ptrtoint, we can transfer control back to the
      // regular getUnderlyingObjectFromInt.
      if (U->getOpcode() == Instruction::PtrToInt)
        return U->getOperand(0);
      // If we find an add of a constant, a multiplied value, or a phi, it's
      // likely that the other operand will lead us to the base
      // object. We don't have to worry about the case where the
      // object address is somehow being computed by the multiply,
      // because our callers only care when the result is an
      // identifiable object.
      if (U->getOpcode() != Instruction::Add ||
          (!isa<ConstantInt>(U->getOperand(1)) &&
           Operator::getOpcode(U->getOperand(1)) != Instruction::Mul &&
           !isa<PHINode>(U->getOperand(1))))
        return V;
      V = U->getOperand(0);
    } else {
      return V;
    }
    assert(V->getType()->isIntegerTy() && "Unexpected operand type!");
  } while (1);
}

/// getUnderlyingObjects - This is a wrapper around GetUnderlyingObjects
/// and adds support for basic ptrtoint+arithmetic+inttoptr sequences.
static void getUnderlyingObjects(const Value *V,
                                 SmallVectorImpl<Value *> &Objects,
                                 const DataLayout &DL) {
  SmallPtrSet<const Value *, 16> Visited;
  SmallVector<const Value *, 4> Working(1, V);
  do {
    V = Working.pop_back_val();

    SmallVector<Value *, 4> Objs;
    GetUnderlyingObjects(const_cast<Value *>(V), Objs, DL);

    for (SmallVectorImpl<Value *>::iterator I = Objs.begin(), IE = Objs.end();
         I != IE; ++I) {
      V = *I;
      if (!Visited.insert(V).second)
        continue;
      if (Operator::getOpcode(V) == Instruction::IntToPtr) {
        const Value *O =
          getUnderlyingObjectFromInt(cast<User>(V)->getOperand(0));
        if (O->getType()->isPointerTy()) {
          Working.push_back(O);
          continue;
        }
      }
      Objects.push_back(const_cast<Value *>(V));
    }
  } while (!Working.empty());
}

typedef PointerUnion<const Value *, const PseudoSourceValue *> ValueType;
typedef SmallVector<PointerIntPair<ValueType, 1, bool>, 4>
UnderlyingObjectsVector;

/// getUnderlyingObjectsForInstr - If this machine instr has memory reference
/// information and it can be tracked to a normal reference to a known
/// object, return the Value for that object.
static void getUnderlyingObjectsForInstr(const MachineInstr *MI,
                                         const MachineFrameInfo *MFI,
                                         UnderlyingObjectsVector &Objects,
                                         const DataLayout &DL) {
  if (!MI->hasOneMemOperand() ||
      (!(*MI->memoperands_begin())->getValue() &&
       !(*MI->memoperands_begin())->getPseudoValue()) ||
      (*MI->memoperands_begin())->isVolatile())
    return;

  if (const PseudoSourceValue *PSV =
      (*MI->memoperands_begin())->getPseudoValue()) {
    // Function that contain tail calls don't have unique PseudoSourceValue
    // objects. Two PseudoSourceValues might refer to the same or overlapping
    // locations. The client code calling this function assumes this is not the
    // case. So return a conservative answer of no known object.
    if (MFI->hasTailCall())
      return;

    // For now, ignore PseudoSourceValues which may alias LLVM IR values
    // because the code that uses this function has no way to cope with
    // such aliases.
    if (!PSV->isAliased(MFI)) {
      bool MayAlias = PSV->mayAlias(MFI);
      Objects.push_back(UnderlyingObjectsVector::value_type(PSV, MayAlias));
    }
    return;
  }

  const Value *V = (*MI->memoperands_begin())->getValue();
  if (!V)
    return;

  SmallVector<Value *, 4> Objs;
  getUnderlyingObjects(V, Objs, DL);

  for (Value *V : Objs) {
    if (!isIdentifiedObject(V)) {
      Objects.clear();
      return;
    }

    Objects.push_back(UnderlyingObjectsVector::value_type(V, true));
  }
}

void ScheduleDAGInstrs::startBlock(MachineBasicBlock *bb) {
  BB = bb;
}

void ScheduleDAGInstrs::finishBlock() {
  // Subclasses should no longer refer to the old block.
  BB = nullptr;
}

/// Initialize the DAG and common scheduler state for the current scheduling
/// region. This does not actually create the DAG, only clears it. The
/// scheduling driver may call BuildSchedGraph multiple times per scheduling
/// region.
void ScheduleDAGInstrs::enterRegion(MachineBasicBlock *bb,
                                    MachineBasicBlock::iterator begin,
                                    MachineBasicBlock::iterator end,
                                    unsigned regioninstrs) {
  assert(bb == BB && "startBlock should set BB");
  RegionBegin = begin;
  RegionEnd = end;
  NumRegionInstrs = regioninstrs;
}

/// Close the current scheduling region. Don't clear any state in case the
/// driver wants to refer to the previous scheduling region.
void ScheduleDAGInstrs::exitRegion() {
  // Nothing to do.
}

/// addSchedBarrierDeps - Add dependencies from instructions in the current
/// list of instructions being scheduled to scheduling barrier by adding
/// the exit SU to the register defs and use list. This is because we want to
/// make sure instructions which define registers that are either used by
/// the terminator or are live-out are properly scheduled. This is
/// especially important when the definition latency of the return value(s)
/// are too high to be hidden by the branch or when the liveout registers
/// used by instructions in the fallthrough block.
void ScheduleDAGInstrs::addSchedBarrierDeps() {
  MachineInstr *ExitMI = RegionEnd != BB->end() ? &*RegionEnd : nullptr;
  ExitSU.setInstr(ExitMI);
  bool AllDepKnown = ExitMI &&
    (ExitMI->isCall() || ExitMI->isBarrier());
  if (ExitMI && AllDepKnown) {
    // If it's a call or a barrier, add dependencies on the defs and uses of
    // instruction.
    for (unsigned i = 0, e = ExitMI->getNumOperands(); i != e; ++i) {
      const MachineOperand &MO = ExitMI->getOperand(i);
      if (!MO.isReg() || MO.isDef()) continue;
      unsigned Reg = MO.getReg();
      if (Reg == 0) continue;

      if (TRI->isPhysicalRegister(Reg))
        Uses.insert(PhysRegSUOper(&ExitSU, -1, Reg));
      else {
        assert(!IsPostRA && "Virtual register encountered after regalloc.");
        if (MO.readsReg()) // ignore undef operands
          addVRegUseDeps(&ExitSU, i);
      }
    }
  } else {
    // For others, e.g. fallthrough, conditional branch, assume the exit
    // uses all the registers that are livein to the successor blocks.
    assert(Uses.empty() && "Uses in set before adding deps?");
    for (MachineBasicBlock::succ_iterator SI = BB->succ_begin(),
           SE = BB->succ_end(); SI != SE; ++SI)
      for (MachineBasicBlock::livein_iterator I = (*SI)->livein_begin(),
             E = (*SI)->livein_end(); I != E; ++I) {
        unsigned Reg = *I;
        if (!Uses.contains(Reg))
          Uses.insert(PhysRegSUOper(&ExitSU, -1, Reg));
      }
  }
}

/// MO is an operand of SU's instruction that defines a physical register. Add
/// data dependencies from SU to any uses of the physical register.
void ScheduleDAGInstrs::addPhysRegDataDeps(SUnit *SU, unsigned OperIdx) {
  const MachineOperand &MO = SU->getInstr()->getOperand(OperIdx);
  assert(MO.isDef() && "expect physreg def");

  // Ask the target if address-backscheduling is desirable, and if so how much.
  const TargetSubtargetInfo &ST = MF.getSubtarget();

  for (MCRegAliasIterator Alias(MO.getReg(), TRI, true);
       Alias.isValid(); ++Alias) {
    if (!Uses.contains(*Alias))
      continue;
    for (Reg2SUnitsMap::iterator I = Uses.find(*Alias); I != Uses.end(); ++I) {
      SUnit *UseSU = I->SU;
      if (UseSU == SU)
        continue;

      // Adjust the dependence latency using operand def/use information,
      // then allow the target to perform its own adjustments.
      int UseOp = I->OpIdx;
      MachineInstr *RegUse = nullptr;
      SDep Dep;
      if (UseOp < 0)
        Dep = SDep(SU, SDep::Artificial);
      else {
        // Set the hasPhysRegDefs only for physreg defs that have a use within
        // the scheduling region.
        SU->hasPhysRegDefs = true;
        Dep = SDep(SU, SDep::Data, *Alias);
        RegUse = UseSU->getInstr();
      }
      Dep.setLatency(
        SchedModel.computeOperandLatency(SU->getInstr(), OperIdx, RegUse,
                                         UseOp));

      ST.adjustSchedDependency(SU, UseSU, Dep);
      UseSU->addPred(Dep);
    }
  }
}

/// addPhysRegDeps - Add register dependencies (data, anti, and output) from
/// this SUnit to following instructions in the same scheduling region that
/// depend the physical register referenced at OperIdx.
void ScheduleDAGInstrs::addPhysRegDeps(SUnit *SU, unsigned OperIdx) {
  MachineInstr *MI = SU->getInstr();
  MachineOperand &MO = MI->getOperand(OperIdx);

  // Optionally add output and anti dependencies. For anti
  // dependencies we use a latency of 0 because for a multi-issue
  // target we want to allow the defining instruction to issue
  // in the same cycle as the using instruction.
  // TODO: Using a latency of 1 here for output dependencies assumes
  //       there's no cost for reusing registers.
  SDep::Kind Kind = MO.isUse() ? SDep::Anti : SDep::Output;
  for (MCRegAliasIterator Alias(MO.getReg(), TRI, true);
       Alias.isValid(); ++Alias) {
    if (!Defs.contains(*Alias))
      continue;
    for (Reg2SUnitsMap::iterator I = Defs.find(*Alias); I != Defs.end(); ++I) {
      SUnit *DefSU = I->SU;
      if (DefSU == &ExitSU)
        continue;
      if (DefSU != SU &&
          (Kind != SDep::Output || !MO.isDead() ||
           !DefSU->getInstr()->registerDefIsDead(*Alias))) {
        if (Kind == SDep::Anti)
          DefSU->addPred(SDep(SU, Kind, /*Reg=*/*Alias));
        else {
          SDep Dep(SU, Kind, /*Reg=*/*Alias);
          Dep.setLatency(
            SchedModel.computeOutputLatency(MI, OperIdx, DefSU->getInstr()));
          DefSU->addPred(Dep);
        }
      }
    }
  }

  if (!MO.isDef()) {
    SU->hasPhysRegUses = true;
    // Either insert a new Reg2SUnits entry with an empty SUnits list, or
    // retrieve the existing SUnits list for this register's uses.
    // Push this SUnit on the use list.
    Uses.insert(PhysRegSUOper(SU, OperIdx, MO.getReg()));
    if (RemoveKillFlags)
      MO.setIsKill(false);
  }
  else {
    addPhysRegDataDeps(SU, OperIdx);
    unsigned Reg = MO.getReg();

    // clear this register's use list
    if (Uses.contains(Reg))
      Uses.eraseAll(Reg);

    if (!MO.isDead()) {
      Defs.eraseAll(Reg);
    } else if (SU->isCall) {
      // Calls will not be reordered because of chain dependencies (see
      // below). Since call operands are dead, calls may continue to be added
      // to the DefList making dependence checking quadratic in the size of
      // the block. Instead, we leave only one call at the back of the
      // DefList.
      Reg2SUnitsMap::RangePair P = Defs.equal_range(Reg);
      Reg2SUnitsMap::iterator B = P.first;
      Reg2SUnitsMap::iterator I = P.second;
      for (bool isBegin = I == B; !isBegin; /* empty */) {
        isBegin = (--I) == B;
        if (!I->SU->isCall)
          break;
        I = Defs.erase(I);
      }
    }

    // Defs are pushed in the order they are visited and never reordered.
    Defs.insert(PhysRegSUOper(SU, OperIdx, Reg));
  }
}

/// addVRegDefDeps - Add register output and data dependencies from this SUnit
/// to instructions that occur later in the same scheduling region if they read
/// from or write to the virtual register defined at OperIdx.
///
/// TODO: Hoist loop induction variable increments. This has to be
/// reevaluated. Generally, IV scheduling should be done before coalescing.
void ScheduleDAGInstrs::addVRegDefDeps(SUnit *SU, unsigned OperIdx) {
  const MachineInstr *MI = SU->getInstr();
  unsigned Reg = MI->getOperand(OperIdx).getReg();

  // Singly defined vregs do not have output/anti dependencies.
  // The current operand is a def, so we have at least one.
  // Check here if there are any others...
  if (MRI.hasOneDef(Reg))
    return;

  // Add output dependence to the next nearest def of this vreg.
  //
  // Unless this definition is dead, the output dependence should be
  // transitively redundant with antidependencies from this definition's
  // uses. We're conservative for now until we have a way to guarantee the uses
  // are not eliminated sometime during scheduling. The output dependence edge
  // is also useful if output latency exceeds def-use latency.
  VReg2SUnitMap::iterator DefI = VRegDefs.find(Reg);
  if (DefI == VRegDefs.end())
    VRegDefs.insert(VReg2SUnit(Reg, SU));
  else {
    SUnit *DefSU = DefI->SU;
    if (DefSU != SU && DefSU != &ExitSU) {
      SDep Dep(SU, SDep::Output, Reg);
      Dep.setLatency(
        SchedModel.computeOutputLatency(MI, OperIdx, DefSU->getInstr()));
      DefSU->addPred(Dep);
    }
    DefI->SU = SU;
  }
}

/// addVRegUseDeps - Add a register data dependency if the instruction that
/// defines the virtual register used at OperIdx is mapped to an SUnit. Add a
/// register antidependency from this SUnit to instructions that occur later in
/// the same scheduling region if they write the virtual register.
///
/// TODO: Handle ExitSU "uses" properly.
void ScheduleDAGInstrs::addVRegUseDeps(SUnit *SU, unsigned OperIdx) {
  MachineInstr *MI = SU->getInstr();
  unsigned Reg = MI->getOperand(OperIdx).getReg();

  // Record this local VReg use.
  VReg2UseMap::iterator UI = VRegUses.find(Reg);
  for (; UI != VRegUses.end(); ++UI) {
    if (UI->SU == SU)
      break;
  }
  if (UI == VRegUses.end())
    VRegUses.insert(VReg2SUnit(Reg, SU));

  // Lookup this operand's reaching definition.
  assert(LIS && "vreg dependencies requires LiveIntervals");
  LiveQueryResult LRQ
    = LIS->getInterval(Reg).Query(LIS->getInstructionIndex(MI));
  VNInfo *VNI = LRQ.valueIn();

  // VNI will be valid because MachineOperand::readsReg() is checked by caller.
  assert(VNI && "No value to read by operand");
  MachineInstr *Def = LIS->getInstructionFromIndex(VNI->def);
  // Phis and other noninstructions (after coalescing) have a NULL Def.
  if (Def) {
    SUnit *DefSU = getSUnit(Def);
    if (DefSU) {
      // The reaching Def lives within this scheduling region.
      // Create a data dependence.
      SDep dep(DefSU, SDep::Data, Reg);
      // Adjust the dependence latency using operand def/use information, then
      // allow the target to perform its own adjustments.
      int DefOp = Def->findRegisterDefOperandIdx(Reg);
      dep.setLatency(SchedModel.computeOperandLatency(Def, DefOp, MI, OperIdx));

      const TargetSubtargetInfo &ST = MF.getSubtarget();
      ST.adjustSchedDependency(DefSU, SU, const_cast<SDep &>(dep));
      SU->addPred(dep);
    }
  }

  // Add antidependence to the following def of the vreg it uses.
  VReg2SUnitMap::iterator DefI = VRegDefs.find(Reg);
  if (DefI != VRegDefs.end() && DefI->SU != SU)
    DefI->SU->addPred(SDep(SU, SDep::Anti, Reg));
}

/// Return true if MI is an instruction we are unable to reason about
/// (like a call or something with unmodeled side effects).
static inline bool isGlobalMemoryObject(AliasAnalysis *AA, MachineInstr *MI) {
  if (MI->isCall() || MI->hasUnmodeledSideEffects() ||
      (MI->hasOrderedMemoryRef() &&
       (!MI->mayLoad() || !MI->isInvariantLoad(AA))))
    return true;
  return false;
}

// This MI might have either incomplete info, or known to be unsafe
// to deal with (i.e. volatile object).
static inline bool isUnsafeMemoryObject(MachineInstr *MI,
                                        const MachineFrameInfo *MFI,
                                        const DataLayout &DL) {
  if (!MI || MI->memoperands_empty())
    return true;
  // We purposefully do no check for hasOneMemOperand() here
  // in hope to trigger an assert downstream in order to
  // finish implementation.
  if ((*MI->memoperands_begin())->isVolatile() ||
       MI->hasUnmodeledSideEffects())
    return true;

  if ((*MI->memoperands_begin())->getPseudoValue()) {
    // Similarly to getUnderlyingObjectForInstr:
    // For now, ignore PseudoSourceValues which may alias LLVM IR values
    // because the code that uses this function has no way to cope with
    // such aliases.
    return true;
  }

  const Value *V = (*MI->memoperands_begin())->getValue();
  if (!V)
    return true;

  SmallVector<Value *, 4> Objs;
  getUnderlyingObjects(V, Objs, DL);
  for (Value *V : Objs) {
    // Does this pointer refer to a distinct and identifiable object?
    if (!isIdentifiedObject(V))
      return true;
  }

  return false;
}

/// This returns true if the two MIs need a chain edge betwee them.
/// If these are not even memory operations, we still may need
/// chain deps between them. The question really is - could
/// these two MIs be reordered during scheduling from memory dependency
/// point of view.
static bool MIsNeedChainEdge(AliasAnalysis *AA, const MachineFrameInfo *MFI,
                             const DataLayout &DL, MachineInstr *MIa,
                             MachineInstr *MIb) {
  const MachineFunction *MF = MIa->getParent()->getParent();
  const TargetInstrInfo *TII = MF->getSubtarget().getInstrInfo();

  // Cover a trivial case - no edge is need to itself.
  if (MIa == MIb)
    return false;
 
  // Let the target decide if memory accesses cannot possibly overlap.
  if ((MIa->mayLoad() || MIa->mayStore()) &&
      (MIb->mayLoad() || MIb->mayStore()))
    if (TII->areMemAccessesTriviallyDisjoint(MIa, MIb, AA))
      return false;

  // FIXME: Need to handle multiple memory operands to support all targets.
  if (!MIa->hasOneMemOperand() || !MIb->hasOneMemOperand())
    return true;

  if (isUnsafeMemoryObject(MIa, MFI, DL) || isUnsafeMemoryObject(MIb, MFI, DL))
    return true;

  // If we are dealing with two "normal" loads, we do not need an edge
  // between them - they could be reordered.
  if (!MIa->mayStore() && !MIb->mayStore())
    return false;

  // To this point analysis is generic. From here on we do need AA.
  if (!AA)
    return true;

  MachineMemOperand *MMOa = *MIa->memoperands_begin();
  MachineMemOperand *MMOb = *MIb->memoperands_begin();

  if (!MMOa->getValue() || !MMOb->getValue())
    return true;

  // The following interface to AA is fashioned after DAGCombiner::isAlias
  // and operates with MachineMemOperand offset with some important
  // assumptions:
  //   - LLVM fundamentally assumes flat address spaces.
  //   - MachineOperand offset can *only* result from legalization and
  //     cannot affect queries other than the trivial case of overlap
  //     checking.
  //   - These offsets never wrap and never step outside
  //     of allocated objects.
  //   - There should never be any negative offsets here.
  //
  // FIXME: Modify API to hide this math from "user"
  // FIXME: Even before we go to AA we can reason locally about some
  // memory objects. It can save compile time, and possibly catch some
  // corner cases not currently covered.

  assert ((MMOa->getOffset() >= 0) && "Negative MachineMemOperand offset");
  assert ((MMOb->getOffset() >= 0) && "Negative MachineMemOperand offset");

  int64_t MinOffset = std::min(MMOa->getOffset(), MMOb->getOffset());
  int64_t Overlapa = MMOa->getSize() + MMOa->getOffset() - MinOffset;
  int64_t Overlapb = MMOb->getSize() + MMOb->getOffset() - MinOffset;

  AliasResult AAResult =
      AA->alias(MemoryLocation(MMOa->getValue(), Overlapa,
                               UseTBAA ? MMOa->getAAInfo() : AAMDNodes()),
                MemoryLocation(MMOb->getValue(), Overlapb,
                               UseTBAA ? MMOb->getAAInfo() : AAMDNodes()));

  return (AAResult != NoAlias);
}

/// This recursive function iterates over chain deps of SUb looking for
/// "latest" node that needs a chain edge to SUa.
static unsigned iterateChainSucc(AliasAnalysis *AA, const MachineFrameInfo *MFI,
                                 const DataLayout &DL, SUnit *SUa, SUnit *SUb,
                                 SUnit *ExitSU, unsigned *Depth,
                                 SmallPtrSetImpl<const SUnit *> &Visited) {
  if (!SUa || !SUb || SUb == ExitSU)
    return *Depth;

  // Remember visited nodes.
  if (!Visited.insert(SUb).second)
      return *Depth;
  // If there is _some_ dependency already in place, do not
  // descend any further.
  // TODO: Need to make sure that if that dependency got eliminated or ignored
  // for any reason in the future, we would not violate DAG topology.
  // Currently it does not happen, but makes an implicit assumption about
  // future implementation.
  //
  // Independently, if we encounter node that is some sort of global
  // object (like a call) we already have full set of dependencies to it
  // and we can stop descending.
  if (SUa->isSucc(SUb) ||
      isGlobalMemoryObject(AA, SUb->getInstr()))
    return *Depth;

  // If we do need an edge, or we have exceeded depth budget,
  // add that edge to the predecessors chain of SUb,
  // and stop descending.
  if (*Depth > 200 ||
      MIsNeedChainEdge(AA, MFI, DL, SUa->getInstr(), SUb->getInstr())) {
    SUb->addPred(SDep(SUa, SDep::MayAliasMem));
    return *Depth;
  }
  // Track current depth.
  (*Depth)++;
  // Iterate over memory dependencies only.
  for (SUnit::const_succ_iterator I = SUb->Succs.begin(), E = SUb->Succs.end();
       I != E; ++I)
    if (I->isNormalMemoryOrBarrier())
      iterateChainSucc(AA, MFI, DL, SUa, I->getSUnit(), ExitSU, Depth, Visited);
  return *Depth;
}

/// This function assumes that "downward" from SU there exist
/// tail/leaf of already constructed DAG. It iterates downward and
/// checks whether SU can be aliasing any node dominated
/// by it.
static void adjustChainDeps(AliasAnalysis *AA, const MachineFrameInfo *MFI,
                            const DataLayout &DL, SUnit *SU, SUnit *ExitSU,
                            std::set<SUnit *> &CheckList,
                            unsigned LatencyToLoad) {
  if (!SU)
    return;

  SmallPtrSet<const SUnit*, 16> Visited;
  unsigned Depth = 0;

  for (std::set<SUnit *>::iterator I = CheckList.begin(), IE = CheckList.end();
       I != IE; ++I) {
    if (SU == *I)
      continue;
    if (MIsNeedChainEdge(AA, MFI, DL, SU->getInstr(), (*I)->getInstr())) {
      SDep Dep(SU, SDep::MayAliasMem);
      Dep.setLatency(((*I)->getInstr()->mayLoad()) ? LatencyToLoad : 0);
      (*I)->addPred(Dep);
    }

    // Iterate recursively over all previously added memory chain
    // successors. Keep track of visited nodes.
    for (SUnit::const_succ_iterator J = (*I)->Succs.begin(),
         JE = (*I)->Succs.end(); J != JE; ++J)
      if (J->isNormalMemoryOrBarrier())
        iterateChainSucc(AA, MFI, DL, SU, J->getSUnit(), ExitSU, &Depth,
                         Visited);
  }
}

/// Check whether two objects need a chain edge, if so, add it
/// otherwise remember the rejected SU.
static inline void addChainDependency(AliasAnalysis *AA,
                                      const MachineFrameInfo *MFI,
                                      const DataLayout &DL, SUnit *SUa,
                                      SUnit *SUb, std::set<SUnit *> &RejectList,
                                      unsigned TrueMemOrderLatency = 0,
                                      bool isNormalMemory = false) {
  // If this is a false dependency,
  // do not add the edge, but rememeber the rejected node.
  if (MIsNeedChainEdge(AA, MFI, DL, SUa->getInstr(), SUb->getInstr())) {
    SDep Dep(SUa, isNormalMemory ? SDep::MayAliasMem : SDep::Barrier);
    Dep.setLatency(TrueMemOrderLatency);
    SUb->addPred(Dep);
  }
  else {
    // Duplicate entries should be ignored.
    RejectList.insert(SUb);
    DEBUG(dbgs() << "\tReject chain dep between SU("
          << SUa->NodeNum << ") and SU("
          << SUb->NodeNum << ")\n");
  }
}

/// Create an SUnit for each real instruction, numbered in top-down toplological
/// order. The instruction order A < B, implies that no edge exists from B to A.
///
/// Map each real instruction to its SUnit.
///
/// After initSUnits, the SUnits vector cannot be resized and the scheduler may
/// hang onto SUnit pointers. We may relax this in the future by using SUnit IDs
/// instead of pointers.
///
/// MachineScheduler relies on initSUnits numbering the nodes by their order in
/// the original instruction list.
void ScheduleDAGInstrs::initSUnits() {
  // We'll be allocating one SUnit for each real instruction in the region,
  // which is contained within a basic block.
  SUnits.reserve(NumRegionInstrs);

  for (MachineBasicBlock::iterator I = RegionBegin; I != RegionEnd; ++I) {
    MachineInstr *MI = I;
    if (MI->isDebugValue())
      continue;

    SUnit *SU = newSUnit(MI);
    MISUnitMap[MI] = SU;

    SU->isCall = MI->isCall();
    SU->isCommutable = MI->isCommutable();

    // Assign the Latency field of SU using target-provided information.
    SU->Latency = SchedModel.computeInstrLatency(SU->getInstr());

    // If this SUnit uses a reserved or unbuffered resource, mark it as such.
    //
    // Reserved resources block an instruction from issuing and stall the
    // entire pipeline. These are identified by BufferSize=0.
    //
    // Unbuffered resources prevent execution of subsequent instructions that
    // require the same resources. This is used for in-order execution pipelines
    // within an out-of-order core. These are identified by BufferSize=1.
    if (SchedModel.hasInstrSchedModel()) {
      const MCSchedClassDesc *SC = getSchedClass(SU);
      for (TargetSchedModel::ProcResIter
             PI = SchedModel.getWriteProcResBegin(SC),
             PE = SchedModel.getWriteProcResEnd(SC); PI != PE; ++PI) {
        switch (SchedModel.getProcResource(PI->ProcResourceIdx)->BufferSize) {
        case 0:
          SU->hasReservedResource = true;
          break;
        case 1:
          SU->isUnbuffered = true;
          break;
        default:
          break;
        }
      }
    }
  }
}

/// If RegPressure is non-null, compute register pressure as a side effect. The
/// DAG builder is an efficient place to do it because it already visits
/// operands.
void ScheduleDAGInstrs::buildSchedGraph(AliasAnalysis *AA,
                                        RegPressureTracker *RPTracker,
                                        PressureDiffs *PDiffs) {
  const TargetSubtargetInfo &ST = MF.getSubtarget();
  bool UseAA = EnableAASchedMI.getNumOccurrences() > 0 ? EnableAASchedMI
                                                       : ST.useAA();
  AliasAnalysis *AAForDep = UseAA ? AA : nullptr;

  MISUnitMap.clear();
  ScheduleDAG::clearDAG();

  // Create an SUnit for each real instruction.
  initSUnits();

  if (PDiffs)
    PDiffs->init(SUnits.size());

  // We build scheduling units by walking a block's instruction list from bottom
  // to top.

  // Remember where a generic side-effecting instruction is as we procede.
  SUnit *BarrierChain = nullptr, *AliasChain = nullptr;

  // Memory references to specific known memory locations are tracked
  // so that they can be given more precise dependencies. We track
  // separately the known memory locations that may alias and those
  // that are known not to alias
  MapVector<ValueType, std::vector<SUnit *> > AliasMemDefs, NonAliasMemDefs;
  MapVector<ValueType, std::vector<SUnit *> > AliasMemUses, NonAliasMemUses;
  std::set<SUnit*> RejectMemNodes;

  // Remove any stale debug info; sometimes BuildSchedGraph is called again
  // without emitting the info from the previous call.
  DbgValues.clear();
  FirstDbgValue = nullptr;

  assert(Defs.empty() && Uses.empty() &&
         "Only BuildGraph should update Defs/Uses");
  Defs.setUniverse(TRI->getNumRegs());
  Uses.setUniverse(TRI->getNumRegs());

  assert(VRegDefs.empty() && "Only BuildSchedGraph may access VRegDefs");
  VRegUses.clear();
  VRegDefs.setUniverse(MRI.getNumVirtRegs());
  VRegUses.setUniverse(MRI.getNumVirtRegs());

  // Model data dependencies between instructions being scheduled and the
  // ExitSU.
  addSchedBarrierDeps();

  // Walk the list of instructions, from bottom moving up.
  MachineInstr *DbgMI = nullptr;
  for (MachineBasicBlock::iterator MII = RegionEnd, MIE = RegionBegin;
       MII != MIE; --MII) {
    MachineInstr *MI = std::prev(MII);
    if (MI && DbgMI) {
      DbgValues.push_back(std::make_pair(DbgMI, MI));
      DbgMI = nullptr;
    }

    if (MI->isDebugValue()) {
      DbgMI = MI;
      continue;
    }
    SUnit *SU = MISUnitMap[MI];
    assert(SU && "No SUnit mapped to this MI");

    if (RPTracker) {
      PressureDiff *PDiff = PDiffs ? &(*PDiffs)[SU->NodeNum] : nullptr;
      RPTracker->recede(/*LiveUses=*/nullptr, PDiff);
      assert(RPTracker->getPos() == std::prev(MII) &&
             "RPTracker can't find MI");
    }

    assert(
        (CanHandleTerminators || (!MI->isTerminator() && !MI->isPosition())) &&
        "Cannot schedule terminators or labels!");

    // Add register-based dependencies (data, anti, and output).
    bool HasVRegDef = false;
    for (unsigned j = 0, n = MI->getNumOperands(); j != n; ++j) {
      const MachineOperand &MO = MI->getOperand(j);
      if (!MO.isReg()) continue;
      unsigned Reg = MO.getReg();
      if (Reg == 0) continue;

      if (TRI->isPhysicalRegister(Reg))
        addPhysRegDeps(SU, j);
      else {
        assert(!IsPostRA && "Virtual register encountered!");
        if (MO.isDef()) {
          HasVRegDef = true;
          addVRegDefDeps(SU, j);
        }
        else if (MO.readsReg()) // ignore undef operands
          addVRegUseDeps(SU, j);
      }
    }
    // If we haven't seen any uses in this scheduling region, create a
    // dependence edge to ExitSU to model the live-out latency. This is required
    // for vreg defs with no in-region use, and prefetches with no vreg def.
    //
    // FIXME: NumDataSuccs would be more precise than NumSuccs here. This
    // check currently relies on being called before adding chain deps.
    if (SU->NumSuccs == 0 && SU->Latency > 1
        && (HasVRegDef || MI->mayLoad())) {
      SDep Dep(SU, SDep::Artificial);
      Dep.setLatency(SU->Latency - 1);
      ExitSU.addPred(Dep);
    }

    // Add chain dependencies.
    // Chain dependencies used to enforce memory order should have
    // latency of 0 (except for true dependency of Store followed by
    // aliased Load... we estimate that with a single cycle of latency
    // assuming the hardware will bypass)
    // Note that isStoreToStackSlot and isLoadFromStackSLot are not usable
    // after stack slots are lowered to actual addresses.
    // TODO: Use an AliasAnalysis and do real alias-analysis queries, and
    // produce more precise dependence information.
    unsigned TrueMemOrderLatency = MI->mayStore() ? 1 : 0;
    if (isGlobalMemoryObject(AA, MI)) {
      // Be conservative with these and add dependencies on all memory
      // references, even those that are known to not alias.
      for (MapVector<ValueType, std::vector<SUnit *> >::iterator I =
             NonAliasMemDefs.begin(), E = NonAliasMemDefs.end(); I != E; ++I) {
        for (unsigned i = 0, e = I->second.size(); i != e; ++i) {
          I->second[i]->addPred(SDep(SU, SDep::Barrier));
        }
      }
      for (MapVector<ValueType, std::vector<SUnit *> >::iterator I =
             NonAliasMemUses.begin(), E = NonAliasMemUses.end(); I != E; ++I) {
        for (unsigned i = 0, e = I->second.size(); i != e; ++i) {
          SDep Dep(SU, SDep::Barrier);
          Dep.setLatency(TrueMemOrderLatency);
          I->second[i]->addPred(Dep);
        }
      }
      // Add SU to the barrier chain.
      if (BarrierChain)
        BarrierChain->addPred(SDep(SU, SDep::Barrier));
      BarrierChain = SU;
      // This is a barrier event that acts as a pivotal node in the DAG,
      // so it is safe to clear list of exposed nodes.
      adjustChainDeps(AA, MFI, *TM.getDataLayout(), SU, &ExitSU, RejectMemNodes,
                      TrueMemOrderLatency);
      RejectMemNodes.clear();
      NonAliasMemDefs.clear();
      NonAliasMemUses.clear();

      // fall-through
    new_alias_chain:
      // Chain all possibly aliasing memory references through SU.
      if (AliasChain) {
        unsigned ChainLatency = 0;
        if (AliasChain->getInstr()->mayLoad())
          ChainLatency = TrueMemOrderLatency;
        addChainDependency(AAForDep, MFI, *TM.getDataLayout(), SU, AliasChain,
                           RejectMemNodes, ChainLatency);
      }
      AliasChain = SU;
      for (unsigned k = 0, m = PendingLoads.size(); k != m; ++k)
        addChainDependency(AAForDep, MFI, *TM.getDataLayout(), SU,
                           PendingLoads[k], RejectMemNodes,
                           TrueMemOrderLatency);
      for (MapVector<ValueType, std::vector<SUnit *> >::iterator I =
           AliasMemDefs.begin(), E = AliasMemDefs.end(); I != E; ++I) {
        for (unsigned i = 0, e = I->second.size(); i != e; ++i)
          addChainDependency(AAForDep, MFI, *TM.getDataLayout(), SU,
                             I->second[i], RejectMemNodes);
      }
      for (MapVector<ValueType, std::vector<SUnit *> >::iterator I =
           AliasMemUses.begin(), E = AliasMemUses.end(); I != E; ++I) {
        for (unsigned i = 0, e = I->second.size(); i != e; ++i)
          addChainDependency(AAForDep, MFI, *TM.getDataLayout(), SU,
                             I->second[i], RejectMemNodes, TrueMemOrderLatency);
      }
      adjustChainDeps(AA, MFI, *TM.getDataLayout(), SU, &ExitSU, RejectMemNodes,
                      TrueMemOrderLatency);
      PendingLoads.clear();
      AliasMemDefs.clear();
      AliasMemUses.clear();
    } else if (MI->mayStore()) {
      // Add dependence on barrier chain, if needed.
      // There is no point to check aliasing on barrier event. Even if
      // SU and barrier _could_ be reordered, they should not. In addition,
      // we have lost all RejectMemNodes below barrier.
      if (BarrierChain)
        BarrierChain->addPred(SDep(SU, SDep::Barrier));

      UnderlyingObjectsVector Objs;
      getUnderlyingObjectsForInstr(MI, MFI, Objs, *TM.getDataLayout());

      if (Objs.empty()) {
        // Treat all other stores conservatively.
        goto new_alias_chain;
      }

      bool MayAlias = false;
      for (UnderlyingObjectsVector::iterator K = Objs.begin(), KE = Objs.end();
           K != KE; ++K) {
        ValueType V = K->getPointer();
        bool ThisMayAlias = K->getInt();
        if (ThisMayAlias)
          MayAlias = true;

        // A store to a specific PseudoSourceValue. Add precise dependencies.
        // Record the def in MemDefs, first adding a dep if there is
        // an existing def.
        MapVector<ValueType, std::vector<SUnit *> >::iterator I =
          ((ThisMayAlias) ? AliasMemDefs.find(V) : NonAliasMemDefs.find(V));
        MapVector<ValueType, std::vector<SUnit *> >::iterator IE =
          ((ThisMayAlias) ? AliasMemDefs.end() : NonAliasMemDefs.end());
        if (I != IE) {
          for (unsigned i = 0, e = I->second.size(); i != e; ++i)
            addChainDependency(AAForDep, MFI, *TM.getDataLayout(), SU,
                               I->second[i], RejectMemNodes, 0, true);

          // If we're not using AA, then we only need one store per object.
          if (!AAForDep)
            I->second.clear();
          I->second.push_back(SU);
        } else {
          if (ThisMayAlias) {
            if (!AAForDep)
              AliasMemDefs[V].clear();
            AliasMemDefs[V].push_back(SU);
          } else {
            if (!AAForDep)
              NonAliasMemDefs[V].clear();
            NonAliasMemDefs[V].push_back(SU);
          }
        }
        // Handle the uses in MemUses, if there are any.
        MapVector<ValueType, std::vector<SUnit *> >::iterator J =
          ((ThisMayAlias) ? AliasMemUses.find(V) : NonAliasMemUses.find(V));
        MapVector<ValueType, std::vector<SUnit *> >::iterator JE =
          ((ThisMayAlias) ? AliasMemUses.end() : NonAliasMemUses.end());
        if (J != JE) {
          for (unsigned i = 0, e = J->second.size(); i != e; ++i)
            addChainDependency(AAForDep, MFI, *TM.getDataLayout(), SU,
                               J->second[i], RejectMemNodes,
                               TrueMemOrderLatency, true);
          J->second.clear();
        }
      }
      if (MayAlias) {
        // Add dependencies from all the PendingLoads, i.e. loads
        // with no underlying object.
        for (unsigned k = 0, m = PendingLoads.size(); k != m; ++k)
          addChainDependency(AAForDep, MFI, *TM.getDataLayout(), SU,
                             PendingLoads[k], RejectMemNodes,
                             TrueMemOrderLatency);
        // Add dependence on alias chain, if needed.
        if (AliasChain)
          addChainDependency(AAForDep, MFI, *TM.getDataLayout(), SU, AliasChain,
                             RejectMemNodes);
      }
      adjustChainDeps(AA, MFI, *TM.getDataLayout(), SU, &ExitSU, RejectMemNodes,
                      TrueMemOrderLatency);
    } else if (MI->mayLoad()) {
      bool MayAlias = true;
      if (MI->isInvariantLoad(AA)) {
        // Invariant load, no chain dependencies needed!
      } else {
        UnderlyingObjectsVector Objs;
        getUnderlyingObjectsForInstr(MI, MFI, Objs, *TM.getDataLayout());

        if (Objs.empty()) {
          // A load with no underlying object. Depend on all
          // potentially aliasing stores.
          for (MapVector<ValueType, std::vector<SUnit *> >::iterator I =
                 AliasMemDefs.begin(), E = AliasMemDefs.end(); I != E; ++I)
            for (unsigned i = 0, e = I->second.size(); i != e; ++i)
              addChainDependency(AAForDep, MFI, *TM.getDataLayout(), SU,
                                 I->second[i], RejectMemNodes);

          PendingLoads.push_back(SU);
          MayAlias = true;
        } else {
          MayAlias = false;
        }

        for (UnderlyingObjectsVector::iterator
             J = Objs.begin(), JE = Objs.end(); J != JE; ++J) {
          ValueType V = J->getPointer();
          bool ThisMayAlias = J->getInt();

          if (ThisMayAlias)
            MayAlias = true;

          // A load from a specific PseudoSourceValue. Add precise dependencies.
          MapVector<ValueType, std::vector<SUnit *> >::iterator I =
            ((ThisMayAlias) ? AliasMemDefs.find(V) : NonAliasMemDefs.find(V));
          MapVector<ValueType, std::vector<SUnit *> >::iterator IE =
            ((ThisMayAlias) ? AliasMemDefs.end() : NonAliasMemDefs.end());
          if (I != IE)
            for (unsigned i = 0, e = I->second.size(); i != e; ++i)
              addChainDependency(AAForDep, MFI, *TM.getDataLayout(), SU,
                                 I->second[i], RejectMemNodes, 0, true);
          if (ThisMayAlias)
            AliasMemUses[V].push_back(SU);
          else
            NonAliasMemUses[V].push_back(SU);
        }
        if (MayAlias)
          adjustChainDeps(AA, MFI, *TM.getDataLayout(), SU, &ExitSU,
                          RejectMemNodes, /*Latency=*/0);
        // Add dependencies on alias and barrier chains, if needed.
        if (MayAlias && AliasChain)
          addChainDependency(AAForDep, MFI, *TM.getDataLayout(), SU, AliasChain,
                             RejectMemNodes);
        if (BarrierChain)
          BarrierChain->addPred(SDep(SU, SDep::Barrier));
      }
    }
  }
  if (DbgMI)
    FirstDbgValue = DbgMI;

  Defs.clear();
  Uses.clear();
  VRegDefs.clear();
  PendingLoads.clear();
}

/// \brief Initialize register live-range state for updating kills.
void ScheduleDAGInstrs::startBlockForKills(MachineBasicBlock *BB) {
  // Start with no live registers.
  LiveRegs.reset();

  // Examine the live-in regs of all successors.
  for (MachineBasicBlock::succ_iterator SI = BB->succ_begin(),
       SE = BB->succ_end(); SI != SE; ++SI) {
    for (MachineBasicBlock::livein_iterator I = (*SI)->livein_begin(),
         E = (*SI)->livein_end(); I != E; ++I) {
      unsigned Reg = *I;
      // Repeat, for reg and all subregs.
      for (MCSubRegIterator SubRegs(Reg, TRI, /*IncludeSelf=*/true);
           SubRegs.isValid(); ++SubRegs)
        LiveRegs.set(*SubRegs);
    }
  }
}

/// \brief If we change a kill flag on the bundle instruction implicit register
/// operands, then we also need to propagate that to any instructions inside
/// the bundle which had the same kill state.
static void toggleBundleKillFlag(MachineInstr *MI, unsigned Reg,
                                 bool NewKillState) {
  if (MI->getOpcode() != TargetOpcode::BUNDLE)
    return;

  // Walk backwards from the last instruction in the bundle to the first.
  // Once we set a kill flag on an instruction, we bail out, as otherwise we
  // might set it on too many operands.  We will clear as many flags as we
  // can though.
  MachineBasicBlock::instr_iterator Begin = MI;
  MachineBasicBlock::instr_iterator End = getBundleEnd(MI);
  while (Begin != End) {
    for (MachineOperand &MO : (--End)->operands()) {
      if (!MO.isReg() || MO.isDef() || Reg != MO.getReg())
        continue;

      // DEBUG_VALUE nodes do not contribute to code generation and should
      // always be ignored.  Failure to do so may result in trying to modify
      // KILL flags on DEBUG_VALUE nodes, which is distressing.
      if (MO.isDebug())
        continue;

      // If the register has the internal flag then it could be killing an
      // internal def of the register.  In this case, just skip.  We only want
      // to toggle the flag on operands visible outside the bundle.
      if (MO.isInternalRead())
        continue;

      if (MO.isKill() == NewKillState)
        continue;
      MO.setIsKill(NewKillState);
      if (NewKillState)
        return;
    }
  }
}

bool ScheduleDAGInstrs::toggleKillFlag(MachineInstr *MI, MachineOperand &MO) {
  // Setting kill flag...
  if (!MO.isKill()) {
    MO.setIsKill(true);
    toggleBundleKillFlag(MI, MO.getReg(), true);
    return false;
  }

  // If MO itself is live, clear the kill flag...
  if (LiveRegs.test(MO.getReg())) {
    MO.setIsKill(false);
    toggleBundleKillFlag(MI, MO.getReg(), false);
    return false;
  }

  // If any subreg of MO is live, then create an imp-def for that
  // subreg and keep MO marked as killed.
  MO.setIsKill(false);
  toggleBundleKillFlag(MI, MO.getReg(), false);
  bool AllDead = true;
  const unsigned SuperReg = MO.getReg();
  MachineInstrBuilder MIB(MF, MI);
  for (MCSubRegIterator SubRegs(SuperReg, TRI); SubRegs.isValid(); ++SubRegs) {
    if (LiveRegs.test(*SubRegs)) {
      MIB.addReg(*SubRegs, RegState::ImplicitDefine);
      AllDead = false;
    }
  }

  if(AllDead) {
    MO.setIsKill(true);
    toggleBundleKillFlag(MI, MO.getReg(), true);
  }
  return false;
}

// FIXME: Reuse the LivePhysRegs utility for this.
void ScheduleDAGInstrs::fixupKills(MachineBasicBlock *MBB) {
  DEBUG(dbgs() << "Fixup kills for BB#" << MBB->getNumber() << '\n');

  LiveRegs.resize(TRI->getNumRegs());
  BitVector killedRegs(TRI->getNumRegs());

  startBlockForKills(MBB);

  // Examine block from end to start...
  unsigned Count = MBB->size();
  for (MachineBasicBlock::iterator I = MBB->end(), E = MBB->begin();
       I != E; --Count) {
    MachineInstr *MI = --I;
    if (MI->isDebugValue())
      continue;

    // Update liveness.  Registers that are defed but not used in this
    // instruction are now dead. Mark register and all subregs as they
    // are completely defined.
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (MO.isRegMask())
        LiveRegs.clearBitsNotInMask(MO.getRegMask());
      if (!MO.isReg()) continue;
      unsigned Reg = MO.getReg();
      if (Reg == 0) continue;
      if (!MO.isDef()) continue;
      // Ignore two-addr defs.
      if (MI->isRegTiedToUseOperand(i)) continue;

      // Repeat for reg and all subregs.
      for (MCSubRegIterator SubRegs(Reg, TRI, /*IncludeSelf=*/true);
           SubRegs.isValid(); ++SubRegs)
        LiveRegs.reset(*SubRegs);
    }

    // Examine all used registers and set/clear kill flag. When a
    // register is used multiple times we only set the kill flag on
    // the first use. Don't set kill flags on undef operands.
    killedRegs.reset();
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg() || !MO.isUse() || MO.isUndef()) continue;
      unsigned Reg = MO.getReg();
      if ((Reg == 0) || MRI.isReserved(Reg)) continue;

      bool kill = false;
      if (!killedRegs.test(Reg)) {
        kill = true;
        // A register is not killed if any subregs are live...
        for (MCSubRegIterator SubRegs(Reg, TRI); SubRegs.isValid(); ++SubRegs) {
          if (LiveRegs.test(*SubRegs)) {
            kill = false;
            break;
          }
        }

        // If subreg is not live, then register is killed if it became
        // live in this instruction
        if (kill)
          kill = !LiveRegs.test(Reg);
      }

      if (MO.isKill() != kill) {
        DEBUG(dbgs() << "Fixing " << MO << " in ");
        // Warning: toggleKillFlag may invalidate MO.
        toggleKillFlag(MI, MO);
        DEBUG(MI->dump());
        DEBUG(if (MI->getOpcode() == TargetOpcode::BUNDLE) {
          MachineBasicBlock::instr_iterator Begin = MI;
          MachineBasicBlock::instr_iterator End = getBundleEnd(MI);
          while (++Begin != End)
            DEBUG(Begin->dump());
        });
      }

      killedRegs.set(Reg);
    }

    // Mark any used register (that is not using undef) and subregs as
    // now live...
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg() || !MO.isUse() || MO.isUndef()) continue;
      unsigned Reg = MO.getReg();
      if ((Reg == 0) || MRI.isReserved(Reg)) continue;

      for (MCSubRegIterator SubRegs(Reg, TRI, /*IncludeSelf=*/true);
           SubRegs.isValid(); ++SubRegs)
        LiveRegs.set(*SubRegs);
    }
  }
}

void ScheduleDAGInstrs::dumpNode(const SUnit *SU) const {
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  SU->getInstr()->dump();
#endif
}

std::string ScheduleDAGInstrs::getGraphNodeLabel(const SUnit *SU) const {
  std::string s;
  raw_string_ostream oss(s);
  if (SU == &EntrySU)
    oss << "<entry>";
  else if (SU == &ExitSU)
    oss << "<exit>";
  else
    SU->getInstr()->print(oss, /*SkipOpers=*/true);
  return oss.str();
}

/// Return the basic block label. It is not necessarilly unique because a block
/// contains multiple scheduling regions. But it is fine for visualization.
std::string ScheduleDAGInstrs::getDAGName() const {
  return "dag." + BB->getFullName();
}

//===----------------------------------------------------------------------===//
// SchedDFSResult Implementation
//===----------------------------------------------------------------------===//

namespace llvm {
/// \brief Internal state used to compute SchedDFSResult.
class SchedDFSImpl {
  SchedDFSResult &R;

  /// Join DAG nodes into equivalence classes by their subtree.
  IntEqClasses SubtreeClasses;
  /// List PredSU, SuccSU pairs that represent data edges between subtrees.
  std::vector<std::pair<const SUnit*, const SUnit*> > ConnectionPairs;

  struct RootData {
    unsigned NodeID;
    unsigned ParentNodeID;  // Parent node (member of the parent subtree).
    unsigned SubInstrCount; // Instr count in this tree only, not children.

    RootData(unsigned id): NodeID(id),
                           ParentNodeID(SchedDFSResult::InvalidSubtreeID),
                           SubInstrCount(0) {}

    unsigned getSparseSetIndex() const { return NodeID; }
  };

  SparseSet<RootData> RootSet;

public:
  SchedDFSImpl(SchedDFSResult &r): R(r), SubtreeClasses(R.DFSNodeData.size()) {
    RootSet.setUniverse(R.DFSNodeData.size());
  }

  /// Return true if this node been visited by the DFS traversal.
  ///
  /// During visitPostorderNode the Node's SubtreeID is assigned to the Node
  /// ID. Later, SubtreeID is updated but remains valid.
  bool isVisited(const SUnit *SU) const {
    return R.DFSNodeData[SU->NodeNum].SubtreeID
      != SchedDFSResult::InvalidSubtreeID;
  }

  /// Initialize this node's instruction count. We don't need to flag the node
  /// visited until visitPostorder because the DAG cannot have cycles.
  void visitPreorder(const SUnit *SU) {
    R.DFSNodeData[SU->NodeNum].InstrCount =
      SU->getInstr()->isTransient() ? 0 : 1;
  }

  /// Called once for each node after all predecessors are visited. Revisit this
  /// node's predecessors and potentially join them now that we know the ILP of
  /// the other predecessors.
  void visitPostorderNode(const SUnit *SU) {
    // Mark this node as the root of a subtree. It may be joined with its
    // successors later.
    R.DFSNodeData[SU->NodeNum].SubtreeID = SU->NodeNum;
    RootData RData(SU->NodeNum);
    RData.SubInstrCount = SU->getInstr()->isTransient() ? 0 : 1;

    // If any predecessors are still in their own subtree, they either cannot be
    // joined or are large enough to remain separate. If this parent node's
    // total instruction count is not greater than a child subtree by at least
    // the subtree limit, then try to join it now since splitting subtrees is
    // only useful if multiple high-pressure paths are possible.
    unsigned InstrCount = R.DFSNodeData[SU->NodeNum].InstrCount;
    for (SUnit::const_pred_iterator
           PI = SU->Preds.begin(), PE = SU->Preds.end(); PI != PE; ++PI) {
      if (PI->getKind() != SDep::Data)
        continue;
      unsigned PredNum = PI->getSUnit()->NodeNum;
      if ((InstrCount - R.DFSNodeData[PredNum].InstrCount) < R.SubtreeLimit)
        joinPredSubtree(*PI, SU, /*CheckLimit=*/false);

      // Either link or merge the TreeData entry from the child to the parent.
      if (R.DFSNodeData[PredNum].SubtreeID == PredNum) {
        // If the predecessor's parent is invalid, this is a tree edge and the
        // current node is the parent.
        if (RootSet[PredNum].ParentNodeID == SchedDFSResult::InvalidSubtreeID)
          RootSet[PredNum].ParentNodeID = SU->NodeNum;
      }
      else if (RootSet.count(PredNum)) {
        // The predecessor is not a root, but is still in the root set. This
        // must be the new parent that it was just joined to. Note that
        // RootSet[PredNum].ParentNodeID may either be invalid or may still be
        // set to the original parent.
        RData.SubInstrCount += RootSet[PredNum].SubInstrCount;
        RootSet.erase(PredNum);
      }
    }
    RootSet[SU->NodeNum] = RData;
  }

  /// Called once for each tree edge after calling visitPostOrderNode on the
  /// predecessor. Increment the parent node's instruction count and
  /// preemptively join this subtree to its parent's if it is small enough.
  void visitPostorderEdge(const SDep &PredDep, const SUnit *Succ) {
    R.DFSNodeData[Succ->NodeNum].InstrCount
      += R.DFSNodeData[PredDep.getSUnit()->NodeNum].InstrCount;
    joinPredSubtree(PredDep, Succ);
  }

  /// Add a connection for cross edges.
  void visitCrossEdge(const SDep &PredDep, const SUnit *Succ) {
    ConnectionPairs.push_back(std::make_pair(PredDep.getSUnit(), Succ));
  }

  /// Set each node's subtree ID to the representative ID and record connections
  /// between trees.
  void finalize() {
    SubtreeClasses.compress();
    R.DFSTreeData.resize(SubtreeClasses.getNumClasses());
    assert(SubtreeClasses.getNumClasses() == RootSet.size()
           && "number of roots should match trees");
    for (SparseSet<RootData>::const_iterator
           RI = RootSet.begin(), RE = RootSet.end(); RI != RE; ++RI) {
      unsigned TreeID = SubtreeClasses[RI->NodeID];
      if (RI->ParentNodeID != SchedDFSResult::InvalidSubtreeID)
        R.DFSTreeData[TreeID].ParentTreeID = SubtreeClasses[RI->ParentNodeID];
      R.DFSTreeData[TreeID].SubInstrCount = RI->SubInstrCount;
      // Note that SubInstrCount may be greater than InstrCount if we joined
      // subtrees across a cross edge. InstrCount will be attributed to the
      // original parent, while SubInstrCount will be attributed to the joined
      // parent.
    }
    R.SubtreeConnections.resize(SubtreeClasses.getNumClasses());
    R.SubtreeConnectLevels.resize(SubtreeClasses.getNumClasses());
    DEBUG(dbgs() << R.getNumSubtrees() << " subtrees:\n");
    for (unsigned Idx = 0, End = R.DFSNodeData.size(); Idx != End; ++Idx) {
      R.DFSNodeData[Idx].SubtreeID = SubtreeClasses[Idx];
      DEBUG(dbgs() << "  SU(" << Idx << ") in tree "
            << R.DFSNodeData[Idx].SubtreeID << '\n');
    }
    for (std::vector<std::pair<const SUnit*, const SUnit*> >::const_iterator
           I = ConnectionPairs.begin(), E = ConnectionPairs.end();
         I != E; ++I) {
      unsigned PredTree = SubtreeClasses[I->first->NodeNum];
      unsigned SuccTree = SubtreeClasses[I->second->NodeNum];
      if (PredTree == SuccTree)
        continue;
      unsigned Depth = I->first->getDepth();
      addConnection(PredTree, SuccTree, Depth);
      addConnection(SuccTree, PredTree, Depth);
    }
  }

protected:
  /// Join the predecessor subtree with the successor that is its DFS
  /// parent. Apply some heuristics before joining.
  bool joinPredSubtree(const SDep &PredDep, const SUnit *Succ,
                       bool CheckLimit = true) {
    assert(PredDep.getKind() == SDep::Data && "Subtrees are for data edges");

    // Check if the predecessor is already joined.
    const SUnit *PredSU = PredDep.getSUnit();
    unsigned PredNum = PredSU->NodeNum;
    if (R.DFSNodeData[PredNum].SubtreeID != PredNum)
      return false;

    // Four is the magic number of successors before a node is considered a
    // pinch point.
    unsigned NumDataSucs = 0;
    for (SUnit::const_succ_iterator SI = PredSU->Succs.begin(),
           SE = PredSU->Succs.end(); SI != SE; ++SI) {
      if (SI->getKind() == SDep::Data) {
        if (++NumDataSucs >= 4)
          return false;
      }
    }
    if (CheckLimit && R.DFSNodeData[PredNum].InstrCount > R.SubtreeLimit)
      return false;
    R.DFSNodeData[PredNum].SubtreeID = Succ->NodeNum;
    SubtreeClasses.join(Succ->NodeNum, PredNum);
    return true;
  }

  /// Called by finalize() to record a connection between trees.
  void addConnection(unsigned FromTree, unsigned ToTree, unsigned Depth) {
    if (!Depth)
      return;

    do {
      SmallVectorImpl<SchedDFSResult::Connection> &Connections =
        R.SubtreeConnections[FromTree];
      for (SmallVectorImpl<SchedDFSResult::Connection>::iterator
             I = Connections.begin(), E = Connections.end(); I != E; ++I) {
        if (I->TreeID == ToTree) {
          I->Level = std::max(I->Level, Depth);
          return;
        }
      }
      Connections.push_back(SchedDFSResult::Connection(ToTree, Depth));
      FromTree = R.DFSTreeData[FromTree].ParentTreeID;
    } while (FromTree != SchedDFSResult::InvalidSubtreeID);
  }
};
} // namespace llvm

namespace {
/// \brief Manage the stack used by a reverse depth-first search over the DAG.
class SchedDAGReverseDFS {
  std::vector<std::pair<const SUnit*, SUnit::const_pred_iterator> > DFSStack;
public:
  bool isComplete() const { return DFSStack.empty(); }

  void follow(const SUnit *SU) {
    DFSStack.push_back(std::make_pair(SU, SU->Preds.begin()));
  }
  void advance() { ++DFSStack.back().second; }

  const SDep *backtrack() {
    DFSStack.pop_back();
    return DFSStack.empty() ? nullptr : std::prev(DFSStack.back().second);
  }

  const SUnit *getCurr() const { return DFSStack.back().first; }

  SUnit::const_pred_iterator getPred() const { return DFSStack.back().second; }

  SUnit::const_pred_iterator getPredEnd() const {
    return getCurr()->Preds.end();
  }
};
} // anonymous

static bool hasDataSucc(const SUnit *SU) {
  for (SUnit::const_succ_iterator
         SI = SU->Succs.begin(), SE = SU->Succs.end(); SI != SE; ++SI) {
    if (SI->getKind() == SDep::Data && !SI->getSUnit()->isBoundaryNode())
      return true;
  }
  return false;
}

/// Compute an ILP metric for all nodes in the subDAG reachable via depth-first
/// search from this root.
void SchedDFSResult::compute(ArrayRef<SUnit> SUnits) {
  if (!IsBottomUp)
    llvm_unreachable("Top-down ILP metric is unimplemnted");

  SchedDFSImpl Impl(*this);
  for (ArrayRef<SUnit>::const_iterator
         SI = SUnits.begin(), SE = SUnits.end(); SI != SE; ++SI) {
    const SUnit *SU = &*SI;
    if (Impl.isVisited(SU) || hasDataSucc(SU))
      continue;

    SchedDAGReverseDFS DFS;
    Impl.visitPreorder(SU);
    DFS.follow(SU);
    for (;;) {
      // Traverse the leftmost path as far as possible.
      while (DFS.getPred() != DFS.getPredEnd()) {
        const SDep &PredDep = *DFS.getPred();
        DFS.advance();
        // Ignore non-data edges.
        if (PredDep.getKind() != SDep::Data
            || PredDep.getSUnit()->isBoundaryNode()) {
          continue;
        }
        // An already visited edge is a cross edge, assuming an acyclic DAG.
        if (Impl.isVisited(PredDep.getSUnit())) {
          Impl.visitCrossEdge(PredDep, DFS.getCurr());
          continue;
        }
        Impl.visitPreorder(PredDep.getSUnit());
        DFS.follow(PredDep.getSUnit());
      }
      // Visit the top of the stack in postorder and backtrack.
      const SUnit *Child = DFS.getCurr();
      const SDep *PredDep = DFS.backtrack();
      Impl.visitPostorderNode(Child);
      if (PredDep)
        Impl.visitPostorderEdge(*PredDep, DFS.getCurr());
      if (DFS.isComplete())
        break;
    }
  }
  Impl.finalize();
}

/// The root of the given SubtreeID was just scheduled. For all subtrees
/// connected to this tree, record the depth of the connection so that the
/// nearest connected subtrees can be prioritized.
void SchedDFSResult::scheduleTree(unsigned SubtreeID) {
  for (SmallVectorImpl<Connection>::const_iterator
         I = SubtreeConnections[SubtreeID].begin(),
         E = SubtreeConnections[SubtreeID].end(); I != E; ++I) {
    SubtreeConnectLevels[I->TreeID] =
      std::max(SubtreeConnectLevels[I->TreeID], I->Level);
    DEBUG(dbgs() << "  Tree: " << I->TreeID
          << " @" << SubtreeConnectLevels[I->TreeID] << '\n');
  }
}

LLVM_DUMP_METHOD
void ILPValue::print(raw_ostream &OS) const {
  OS << InstrCount << " / " << Length << " = ";
  if (!Length)
    OS << "BADILP";
  else
    OS << format("%g", ((double)InstrCount / Length));
}

LLVM_DUMP_METHOD
void ILPValue::dump() const {
  dbgs() << *this << '\n';
}

namespace llvm {

LLVM_DUMP_METHOD
raw_ostream &operator<<(raw_ostream &OS, const ILPValue &Val) {
  Val.print(OS);
  return OS;
}

} // namespace llvm
