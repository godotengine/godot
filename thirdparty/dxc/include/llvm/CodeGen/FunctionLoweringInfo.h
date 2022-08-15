//===-- FunctionLoweringInfo.h - Lower functions from LLVM IR to CodeGen --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements routines for translating functions from LLVM IR into
// Machine IR.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_FUNCTIONLOWERINGINFO_H
#define LLVM_CODEGEN_FUNCTIONLOWERINGINFO_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include <vector>

namespace llvm {

class AllocaInst;
class BasicBlock;
class BranchProbabilityInfo;
class CallInst;
class Function;
class GlobalVariable;
class Instruction;
class MachineInstr;
class MachineBasicBlock;
class MachineFunction;
class MachineModuleInfo;
class MachineRegisterInfo;
class SelectionDAG;
class MVT;
class TargetLowering;
class Value;

//===--------------------------------------------------------------------===//
/// FunctionLoweringInfo - This contains information that is global to a
/// function that is used when lowering a region of the function.
///
class FunctionLoweringInfo {
public:
  const Function *Fn;
  MachineFunction *MF;
  const TargetLowering *TLI;
  MachineRegisterInfo *RegInfo;
  BranchProbabilityInfo *BPI;
  /// CanLowerReturn - true iff the function's return value can be lowered to
  /// registers.
  bool CanLowerReturn;

  /// DemoteRegister - if CanLowerReturn is false, DemoteRegister is a vreg
  /// allocated to hold a pointer to the hidden sret parameter.
  unsigned DemoteRegister;

  /// MBBMap - A mapping from LLVM basic blocks to their machine code entry.
  DenseMap<const BasicBlock*, MachineBasicBlock *> MBBMap;

  /// ValueMap - Since we emit code for the function a basic block at a time,
  /// we must remember which virtual registers hold the values for
  /// cross-basic-block values.
  DenseMap<const Value*, unsigned> ValueMap;

  // Keep track of frame indices allocated for statepoints as they could be used
  // across basic block boundaries.
  // Key of the map is statepoint instruction, value is a map from spilled
  // llvm Value to the optional stack stack slot index.
  // If optional is unspecified it means that we have visited this value
  // but didn't spill it.
  typedef DenseMap<const Value*, Optional<int>> StatepointSpilledValueMapTy;
  DenseMap<const Instruction*, StatepointSpilledValueMapTy>
    StatepointRelocatedValues;

  /// StaticAllocaMap - Keep track of frame indices for fixed sized allocas in
  /// the entry block.  This allows the allocas to be efficiently referenced
  /// anywhere in the function.
  DenseMap<const AllocaInst*, int> StaticAllocaMap;

  /// ByValArgFrameIndexMap - Keep track of frame indices for byval arguments.
  DenseMap<const Argument*, int> ByValArgFrameIndexMap;

  /// ArgDbgValues - A list of DBG_VALUE instructions created during isel for
  /// function arguments that are inserted after scheduling is completed.
  SmallVector<MachineInstr*, 8> ArgDbgValues;

  /// RegFixups - Registers which need to be replaced after isel is done.
  DenseMap<unsigned, unsigned> RegFixups;

  /// StatepointStackSlots - A list of temporary stack slots (frame indices) 
  /// used to spill values at a statepoint.  We store them here to enable
  /// reuse of the same stack slots across different statepoints in different
  /// basic blocks.
  SmallVector<unsigned, 50> StatepointStackSlots;

  /// MBB - The current block.
  MachineBasicBlock *MBB;

  /// MBB - The current insert position inside the current block.
  MachineBasicBlock::iterator InsertPt;

#ifndef NDEBUG
  SmallPtrSet<const Instruction *, 8> CatchInfoLost;
  SmallPtrSet<const Instruction *, 8> CatchInfoFound;
#endif

  struct LiveOutInfo {
    unsigned NumSignBits : 31;
    bool IsValid : 1;
    APInt KnownOne, KnownZero;
    LiveOutInfo() : NumSignBits(0), IsValid(true), KnownOne(1, 0),
                    KnownZero(1, 0) {}
  };

  /// Record the preferred extend type (ISD::SIGN_EXTEND or ISD::ZERO_EXTEND)
  /// for a value.
  DenseMap<const Value *, ISD::NodeType> PreferredExtendType;

  /// VisitedBBs - The set of basic blocks visited thus far by instruction
  /// selection.
  SmallPtrSet<const BasicBlock*, 4> VisitedBBs;

  /// PHINodesToUpdate - A list of phi instructions whose operand list will
  /// be updated after processing the current basic block.
  /// TODO: This isn't per-function state, it's per-basic-block state. But
  /// there's no other convenient place for it to live right now.
  std::vector<std::pair<MachineInstr*, unsigned> > PHINodesToUpdate;
  unsigned OrigNumPHINodesToUpdate;

  /// If the current MBB is a landing pad, the exception pointer and exception
  /// selector registers are copied into these virtual registers by
  /// SelectionDAGISel::PrepareEHLandingPad().
  unsigned ExceptionPointerVirtReg, ExceptionSelectorVirtReg;

  /// set - Initialize this FunctionLoweringInfo with the given Function
  /// and its associated MachineFunction.
  ///
  void set(const Function &Fn, MachineFunction &MF, SelectionDAG *DAG);

  /// clear - Clear out all the function-specific state. This returns this
  /// FunctionLoweringInfo to an empty state, ready to be used for a
  /// different function.
  void clear();

  /// isExportedInst - Return true if the specified value is an instruction
  /// exported from its block.
  bool isExportedInst(const Value *V) {
    return ValueMap.count(V);
  }

  unsigned CreateReg(MVT VT);
  
  unsigned CreateRegs(Type *Ty);
  
  unsigned InitializeRegForValue(const Value *V) {
    unsigned &R = ValueMap[V];
    assert(R == 0 && "Already initialized this value register!");
    return R = CreateRegs(V->getType());
  }

  /// GetLiveOutRegInfo - Gets LiveOutInfo for a register, returning NULL if the
  /// register is a PHI destination and the PHI's LiveOutInfo is not valid.
  const LiveOutInfo *GetLiveOutRegInfo(unsigned Reg) {
    if (!LiveOutRegInfo.inBounds(Reg))
      return nullptr;

    const LiveOutInfo *LOI = &LiveOutRegInfo[Reg];
    if (!LOI->IsValid)
      return nullptr;

    return LOI;
  }

  /// GetLiveOutRegInfo - Gets LiveOutInfo for a register, returning NULL if the
  /// register is a PHI destination and the PHI's LiveOutInfo is not valid. If
  /// the register's LiveOutInfo is for a smaller bit width, it is extended to
  /// the larger bit width by zero extension. The bit width must be no smaller
  /// than the LiveOutInfo's existing bit width.
  const LiveOutInfo *GetLiveOutRegInfo(unsigned Reg, unsigned BitWidth);

  /// AddLiveOutRegInfo - Adds LiveOutInfo for a register.
  void AddLiveOutRegInfo(unsigned Reg, unsigned NumSignBits,
                         const APInt &KnownZero, const APInt &KnownOne) {
    // Only install this information if it tells us something.
    if (NumSignBits == 1 && KnownZero == 0 && KnownOne == 0)
      return;

    LiveOutRegInfo.grow(Reg);
    LiveOutInfo &LOI = LiveOutRegInfo[Reg];
    LOI.NumSignBits = NumSignBits;
    LOI.KnownOne = KnownOne;
    LOI.KnownZero = KnownZero;
  }

  /// ComputePHILiveOutRegInfo - Compute LiveOutInfo for a PHI's destination
  /// register based on the LiveOutInfo of its operands.
  void ComputePHILiveOutRegInfo(const PHINode*);

  /// InvalidatePHILiveOutRegInfo - Invalidates a PHI's LiveOutInfo, to be
  /// called when a block is visited before all of its predecessors.
  void InvalidatePHILiveOutRegInfo(const PHINode *PN) {
    // PHIs with no uses have no ValueMap entry.
    DenseMap<const Value*, unsigned>::const_iterator It = ValueMap.find(PN);
    if (It == ValueMap.end())
      return;

    unsigned Reg = It->second;
    if (Reg == 0)
      return;

    LiveOutRegInfo.grow(Reg);
    LiveOutRegInfo[Reg].IsValid = false;
  }

  /// setArgumentFrameIndex - Record frame index for the byval
  /// argument.
  void setArgumentFrameIndex(const Argument *A, int FI);

  /// getArgumentFrameIndex - Get frame index for the byval argument.
  int getArgumentFrameIndex(const Argument *A);

private:
  void addSEHHandlersForLPads(ArrayRef<const LandingPadInst *> LPads);

  /// LiveOutRegInfo - Information about live out vregs.
  IndexedMap<LiveOutInfo, VirtReg2IndexFunctor> LiveOutRegInfo;
};

/// ComputeUsesVAFloatArgument - Determine if any floating-point values are
/// being passed to this variadic function, and set the MachineModuleInfo's
/// usesVAFloatArgument flag if so. This flag is used to emit an undefined
/// reference to _fltused on Windows, which will link in MSVCRT's
/// floating-point support.
void ComputeUsesVAFloatArgument(const CallInst &I, MachineModuleInfo *MMI);

/// AddLandingPadInfo - Extract the exception handling information from the
/// landingpad instruction and add them to the specified machine module info.
void AddLandingPadInfo(const LandingPadInst &I, MachineModuleInfo &MMI,
                       MachineBasicBlock *MBB);

} // end namespace llvm

#endif
