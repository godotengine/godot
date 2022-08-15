//===-- FunctionLoweringInfo.cpp ------------------------------------------===//
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

#include "llvm/CodeGen/FunctionLoweringInfo.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/WinEHFuncInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include <algorithm>
using namespace llvm;

#define DEBUG_TYPE "function-lowering-info"

/// isUsedOutsideOfDefiningBlock - Return true if this instruction is used by
/// PHI nodes or outside of the basic block that defines it, or used by a
/// switch or atomic instruction, which may expand to multiple basic blocks.
static bool isUsedOutsideOfDefiningBlock(const Instruction *I) {
  if (I->use_empty()) return false;
  if (isa<PHINode>(I)) return true;
  const BasicBlock *BB = I->getParent();
  for (const User *U : I->users())
    if (cast<Instruction>(U)->getParent() != BB || isa<PHINode>(U))
      return true;

  return false;
}

static ISD::NodeType getPreferredExtendForValue(const Value *V) {
  // For the users of the source value being used for compare instruction, if
  // the number of signed predicate is greater than unsigned predicate, we
  // prefer to use SIGN_EXTEND.
  //
  // With this optimization, we would be able to reduce some redundant sign or
  // zero extension instruction, and eventually more machine CSE opportunities
  // can be exposed.
  ISD::NodeType ExtendKind = ISD::ANY_EXTEND;
  unsigned NumOfSigned = 0, NumOfUnsigned = 0;
  for (const User *U : V->users()) {
    if (const auto *CI = dyn_cast<CmpInst>(U)) {
      NumOfSigned += CI->isSigned();
      NumOfUnsigned += CI->isUnsigned();
    }
  }
  if (NumOfSigned > NumOfUnsigned)
    ExtendKind = ISD::SIGN_EXTEND;

  return ExtendKind;
}

void FunctionLoweringInfo::set(const Function &fn, MachineFunction &mf,
                               SelectionDAG *DAG) {
  Fn = &fn;
  MF = &mf;
  TLI = MF->getSubtarget().getTargetLowering();
  RegInfo = &MF->getRegInfo();
  MachineModuleInfo &MMI = MF->getMMI();

  // Check whether the function can return without sret-demotion.
  SmallVector<ISD::OutputArg, 4> Outs;
  GetReturnInfo(Fn->getReturnType(), Fn->getAttributes(), Outs, *TLI,
                mf.getDataLayout());
  CanLowerReturn = TLI->CanLowerReturn(Fn->getCallingConv(), *MF,
                                       Fn->isVarArg(), Outs, Fn->getContext());

  // Initialize the mapping of values to registers.  This is only set up for
  // instruction values that are used outside of the block that defines
  // them.
  Function::const_iterator BB = Fn->begin(), EB = Fn->end();
  for (; BB != EB; ++BB)
    for (BasicBlock::const_iterator I = BB->begin(), E = BB->end();
         I != E; ++I) {
      if (const AllocaInst *AI = dyn_cast<AllocaInst>(I)) {
        // Static allocas can be folded into the initial stack frame adjustment.
        if (AI->isStaticAlloca()) {
          const ConstantInt *CUI = cast<ConstantInt>(AI->getArraySize());
          Type *Ty = AI->getAllocatedType();
          uint64_t TySize = MF->getDataLayout().getTypeAllocSize(Ty);
          unsigned Align =
              std::max((unsigned)MF->getDataLayout().getPrefTypeAlignment(Ty),
                       AI->getAlignment());

          TySize *= CUI->getZExtValue();   // Get total allocated size.
          if (TySize == 0) TySize = 1; // Don't create zero-sized stack objects.

          StaticAllocaMap[AI] =
            MF->getFrameInfo()->CreateStackObject(TySize, Align, false, AI);

        } else {
          unsigned Align =
              std::max((unsigned)MF->getDataLayout().getPrefTypeAlignment(
                           AI->getAllocatedType()),
                       AI->getAlignment());
          unsigned StackAlign =
              MF->getSubtarget().getFrameLowering()->getStackAlignment();
          if (Align <= StackAlign)
            Align = 0;
          // Inform the Frame Information that we have variable-sized objects.
          MF->getFrameInfo()->CreateVariableSizedObject(Align ? Align : 1, AI);
        }
      }

      // Look for inline asm that clobbers the SP register.
      if (isa<CallInst>(I) || isa<InvokeInst>(I)) {
        ImmutableCallSite CS(I);
        if (isa<InlineAsm>(CS.getCalledValue())) {
          unsigned SP = TLI->getStackPointerRegisterToSaveRestore();
          const TargetRegisterInfo *TRI = MF->getSubtarget().getRegisterInfo();
          std::vector<TargetLowering::AsmOperandInfo> Ops =
              TLI->ParseConstraints(Fn->getParent()->getDataLayout(), TRI, CS);
          for (size_t I = 0, E = Ops.size(); I != E; ++I) {
            TargetLowering::AsmOperandInfo &Op = Ops[I];
            if (Op.Type == InlineAsm::isClobber) {
              // Clobbers don't have SDValue operands, hence SDValue().
              TLI->ComputeConstraintToUse(Op, SDValue(), DAG);
              std::pair<unsigned, const TargetRegisterClass *> PhysReg =
                  TLI->getRegForInlineAsmConstraint(TRI, Op.ConstraintCode,
                                                    Op.ConstraintVT);
              if (PhysReg.first == SP)
                MF->getFrameInfo()->setHasOpaqueSPAdjustment(true);
            }
          }
        }
      }

      // Look for calls to the @llvm.va_start intrinsic. We can omit some
      // prologue boilerplate for variadic functions that don't examine their
      // arguments.
      if (const auto *II = dyn_cast<IntrinsicInst>(I)) {
        if (II->getIntrinsicID() == Intrinsic::vastart)
          MF->getFrameInfo()->setHasVAStart(true);
      }

      // If we have a musttail call in a variadic funciton, we need to ensure we
      // forward implicit register parameters.
      if (const auto *CI = dyn_cast<CallInst>(I)) {
        if (CI->isMustTailCall() && Fn->isVarArg())
          MF->getFrameInfo()->setHasMustTailInVarArgFunc(true);
      }

      // Mark values used outside their block as exported, by allocating
      // a virtual register for them.
      if (isUsedOutsideOfDefiningBlock(I))
        if (!isa<AllocaInst>(I) ||
            !StaticAllocaMap.count(cast<AllocaInst>(I)))
          InitializeRegForValue(I);

      // Collect llvm.dbg.declare information. This is done now instead of
      // during the initial isel pass through the IR so that it is done
      // in a predictable order.
      if (const DbgDeclareInst *DI = dyn_cast<DbgDeclareInst>(I)) {
        assert(DI->getVariable() && "Missing variable");
        assert(DI->getDebugLoc() && "Missing location");
        if (MMI.hasDebugInfo()) {
          // Don't handle byval struct arguments or VLAs, for example.
          // Non-byval arguments are handled here (they refer to the stack
          // temporary alloca at this point).
          const Value *Address = DI->getAddress();
          if (Address) {
            if (const BitCastInst *BCI = dyn_cast<BitCastInst>(Address))
              Address = BCI->getOperand(0);
            if (const AllocaInst *AI = dyn_cast<AllocaInst>(Address)) {
              DenseMap<const AllocaInst *, int>::iterator SI =
                StaticAllocaMap.find(AI);
              if (SI != StaticAllocaMap.end()) { // Check for VLAs.
                int FI = SI->second;
                MMI.setVariableDbgInfo(DI->getVariable(), DI->getExpression(),
                                       FI, DI->getDebugLoc());
              }
            }
          }
        }
      }

      // Decide the preferred extend type for a value.
      PreferredExtendType[I] = getPreferredExtendForValue(I);
    }

  // Create an initial MachineBasicBlock for each LLVM BasicBlock in F.  This
  // also creates the initial PHI MachineInstrs, though none of the input
  // operands are populated.
  for (BB = Fn->begin(); BB != EB; ++BB) {
    MachineBasicBlock *MBB = mf.CreateMachineBasicBlock(BB);
    MBBMap[BB] = MBB;
    MF->push_back(MBB);

    // Transfer the address-taken flag. This is necessary because there could
    // be multiple MachineBasicBlocks corresponding to one BasicBlock, and only
    // the first one should be marked.
    if (BB->hasAddressTaken())
      MBB->setHasAddressTaken();

    // Create Machine PHI nodes for LLVM PHI nodes, lowering them as
    // appropriate.
    for (BasicBlock::const_iterator I = BB->begin();
         const PHINode *PN = dyn_cast<PHINode>(I); ++I) {
      if (PN->use_empty()) continue;

      // Skip empty types
      if (PN->getType()->isEmptyTy())
        continue;

      DebugLoc DL = PN->getDebugLoc();
      unsigned PHIReg = ValueMap[PN];
      assert(PHIReg && "PHI node does not have an assigned virtual register!");

      SmallVector<EVT, 4> ValueVTs;
      ComputeValueVTs(*TLI, MF->getDataLayout(), PN->getType(), ValueVTs);
      for (unsigned vti = 0, vte = ValueVTs.size(); vti != vte; ++vti) {
        EVT VT = ValueVTs[vti];
        unsigned NumRegisters = TLI->getNumRegisters(Fn->getContext(), VT);
        const TargetInstrInfo *TII = MF->getSubtarget().getInstrInfo();
        for (unsigned i = 0; i != NumRegisters; ++i)
          BuildMI(MBB, DL, TII->get(TargetOpcode::PHI), PHIReg + i);
        PHIReg += NumRegisters;
      }
    }
  }

  // Mark landing pad blocks.
  SmallVector<const LandingPadInst *, 4> LPads;
  for (BB = Fn->begin(); BB != EB; ++BB) {
    if (const auto *Invoke = dyn_cast<InvokeInst>(BB->getTerminator()))
      MBBMap[Invoke->getSuccessor(1)]->setIsLandingPad();
    if (BB->isLandingPad())
      LPads.push_back(BB->getLandingPadInst());
  }

  // If this is an MSVC EH personality, we need to do a bit more work.
  EHPersonality Personality = EHPersonality::Unknown;
  if (Fn->hasPersonalityFn())
    Personality = classifyEHPersonality(Fn->getPersonalityFn());
  if (!isMSVCEHPersonality(Personality))
    return;

  if (Personality == EHPersonality::MSVC_Win64SEH ||
      Personality == EHPersonality::MSVC_X86SEH) {
    addSEHHandlersForLPads(LPads);
  }

  WinEHFuncInfo &EHInfo = MMI.getWinEHFuncInfo(&fn);
  if (Personality == EHPersonality::MSVC_CXX) {
    const Function *WinEHParentFn = MMI.getWinEHParent(&fn);
    calculateWinCXXEHStateNumbers(WinEHParentFn, EHInfo);
  }

  // Copy the state numbers to LandingPadInfo for the current function, which
  // could be a handler or the parent. This should happen for 32-bit SEH and
  // C++ EH.
  if (Personality == EHPersonality::MSVC_CXX ||
      Personality == EHPersonality::MSVC_X86SEH) {
    for (const LandingPadInst *LP : LPads) {
      MachineBasicBlock *LPadMBB = MBBMap[LP->getParent()];
      MMI.addWinEHState(LPadMBB, EHInfo.LandingPadStateMap[LP]);
    }
  }
}

void FunctionLoweringInfo::addSEHHandlersForLPads(
    ArrayRef<const LandingPadInst *> LPads) {
  MachineModuleInfo &MMI = MF->getMMI();

  // Iterate over all landing pads with llvm.eh.actions calls.
  for (const LandingPadInst *LP : LPads) {
    const IntrinsicInst *ActionsCall =
        dyn_cast<IntrinsicInst>(LP->getNextNode());
    if (!ActionsCall ||
        ActionsCall->getIntrinsicID() != Intrinsic::eh_actions)
      continue;

    // Parse the llvm.eh.actions call we found.
    MachineBasicBlock *LPadMBB = MBBMap[LP->getParent()];
    SmallVector<std::unique_ptr<ActionHandler>, 4> Actions;
    parseEHActions(ActionsCall, Actions);

    // Iterate EH actions from most to least precedence, which means
    // iterating in reverse.
    for (auto I = Actions.rbegin(), E = Actions.rend(); I != E; ++I) {
      ActionHandler *Action = I->get();
      if (auto *CH = dyn_cast<CatchHandler>(Action)) {
        const auto *Filter =
            dyn_cast<Function>(CH->getSelector()->stripPointerCasts());
        assert((Filter || CH->getSelector()->isNullValue()) &&
               "expected function or catch-all");
        const auto *RecoverBA =
            cast<BlockAddress>(CH->getHandlerBlockOrFunc());
        MMI.addSEHCatchHandler(LPadMBB, Filter, RecoverBA);
      } else {
        assert(isa<CleanupHandler>(Action));
        const auto *Fini = cast<Function>(Action->getHandlerBlockOrFunc());
        MMI.addSEHCleanupHandler(LPadMBB, Fini);
      }
    }
  }
}

/// clear - Clear out all the function-specific state. This returns this
/// FunctionLoweringInfo to an empty state, ready to be used for a
/// different function.
void FunctionLoweringInfo::clear() {
  assert(CatchInfoFound.size() == CatchInfoLost.size() &&
         "Not all catch info was assigned to a landing pad!");

  MBBMap.clear();
  ValueMap.clear();
  StaticAllocaMap.clear();
#ifndef NDEBUG
  CatchInfoLost.clear();
  CatchInfoFound.clear();
#endif
  LiveOutRegInfo.clear();
  VisitedBBs.clear();
  ArgDbgValues.clear();
  ByValArgFrameIndexMap.clear();
  RegFixups.clear();
  StatepointStackSlots.clear();
  StatepointRelocatedValues.clear();
  PreferredExtendType.clear();
}

/// CreateReg - Allocate a single virtual register for the given type.
unsigned FunctionLoweringInfo::CreateReg(MVT VT) {
  return RegInfo->createVirtualRegister(
      MF->getSubtarget().getTargetLowering()->getRegClassFor(VT));
}

/// CreateRegs - Allocate the appropriate number of virtual registers of
/// the correctly promoted or expanded types.  Assign these registers
/// consecutive vreg numbers and return the first assigned number.
///
/// In the case that the given value has struct or array type, this function
/// will assign registers for each member or element.
///
unsigned FunctionLoweringInfo::CreateRegs(Type *Ty) {
  const TargetLowering *TLI = MF->getSubtarget().getTargetLowering();

  SmallVector<EVT, 4> ValueVTs;
  ComputeValueVTs(*TLI, MF->getDataLayout(), Ty, ValueVTs);

  unsigned FirstReg = 0;
  for (unsigned Value = 0, e = ValueVTs.size(); Value != e; ++Value) {
    EVT ValueVT = ValueVTs[Value];
    MVT RegisterVT = TLI->getRegisterType(Ty->getContext(), ValueVT);

    unsigned NumRegs = TLI->getNumRegisters(Ty->getContext(), ValueVT);
    for (unsigned i = 0; i != NumRegs; ++i) {
      unsigned R = CreateReg(RegisterVT);
      if (!FirstReg) FirstReg = R;
    }
  }
  return FirstReg;
}

/// GetLiveOutRegInfo - Gets LiveOutInfo for a register, returning NULL if the
/// register is a PHI destination and the PHI's LiveOutInfo is not valid. If
/// the register's LiveOutInfo is for a smaller bit width, it is extended to
/// the larger bit width by zero extension. The bit width must be no smaller
/// than the LiveOutInfo's existing bit width.
const FunctionLoweringInfo::LiveOutInfo *
FunctionLoweringInfo::GetLiveOutRegInfo(unsigned Reg, unsigned BitWidth) {
  if (!LiveOutRegInfo.inBounds(Reg))
    return nullptr;

  LiveOutInfo *LOI = &LiveOutRegInfo[Reg];
  if (!LOI->IsValid)
    return nullptr;

  if (BitWidth > LOI->KnownZero.getBitWidth()) {
    LOI->NumSignBits = 1;
    LOI->KnownZero = LOI->KnownZero.zextOrTrunc(BitWidth);
    LOI->KnownOne = LOI->KnownOne.zextOrTrunc(BitWidth);
  }

  return LOI;
}

/// ComputePHILiveOutRegInfo - Compute LiveOutInfo for a PHI's destination
/// register based on the LiveOutInfo of its operands.
void FunctionLoweringInfo::ComputePHILiveOutRegInfo(const PHINode *PN) {
  Type *Ty = PN->getType();
  if (!Ty->isIntegerTy() || Ty->isVectorTy())
    return;

  SmallVector<EVT, 1> ValueVTs;
  ComputeValueVTs(*TLI, MF->getDataLayout(), Ty, ValueVTs);
  assert(ValueVTs.size() == 1 &&
         "PHIs with non-vector integer types should have a single VT.");
  EVT IntVT = ValueVTs[0];

  if (TLI->getNumRegisters(PN->getContext(), IntVT) != 1)
    return;
  IntVT = TLI->getTypeToTransformTo(PN->getContext(), IntVT);
  unsigned BitWidth = IntVT.getSizeInBits();

  unsigned DestReg = ValueMap[PN];
  if (!TargetRegisterInfo::isVirtualRegister(DestReg))
    return;
  LiveOutRegInfo.grow(DestReg);
  LiveOutInfo &DestLOI = LiveOutRegInfo[DestReg];

  Value *V = PN->getIncomingValue(0);
  if (isa<UndefValue>(V) || isa<ConstantExpr>(V)) {
    DestLOI.NumSignBits = 1;
    APInt Zero(BitWidth, 0);
    DestLOI.KnownZero = Zero;
    DestLOI.KnownOne = Zero;
    return;
  }

  if (ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
    APInt Val = CI->getValue().zextOrTrunc(BitWidth);
    DestLOI.NumSignBits = Val.getNumSignBits();
    DestLOI.KnownZero = ~Val;
    DestLOI.KnownOne = Val;
  } else {
    assert(ValueMap.count(V) && "V should have been placed in ValueMap when its"
                                "CopyToReg node was created.");
    unsigned SrcReg = ValueMap[V];
    if (!TargetRegisterInfo::isVirtualRegister(SrcReg)) {
      DestLOI.IsValid = false;
      return;
    }
    const LiveOutInfo *SrcLOI = GetLiveOutRegInfo(SrcReg, BitWidth);
    if (!SrcLOI) {
      DestLOI.IsValid = false;
      return;
    }
    DestLOI = *SrcLOI;
  }

  assert(DestLOI.KnownZero.getBitWidth() == BitWidth &&
         DestLOI.KnownOne.getBitWidth() == BitWidth &&
         "Masks should have the same bit width as the type.");

  for (unsigned i = 1, e = PN->getNumIncomingValues(); i != e; ++i) {
    Value *V = PN->getIncomingValue(i);
    if (isa<UndefValue>(V) || isa<ConstantExpr>(V)) {
      DestLOI.NumSignBits = 1;
      APInt Zero(BitWidth, 0);
      DestLOI.KnownZero = Zero;
      DestLOI.KnownOne = Zero;
      return;
    }

    if (ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
      APInt Val = CI->getValue().zextOrTrunc(BitWidth);
      DestLOI.NumSignBits = std::min(DestLOI.NumSignBits, Val.getNumSignBits());
      DestLOI.KnownZero &= ~Val;
      DestLOI.KnownOne &= Val;
      continue;
    }

    assert(ValueMap.count(V) && "V should have been placed in ValueMap when "
                                "its CopyToReg node was created.");
    unsigned SrcReg = ValueMap[V];
    if (!TargetRegisterInfo::isVirtualRegister(SrcReg)) {
      DestLOI.IsValid = false;
      return;
    }
    const LiveOutInfo *SrcLOI = GetLiveOutRegInfo(SrcReg, BitWidth);
    if (!SrcLOI) {
      DestLOI.IsValid = false;
      return;
    }
    DestLOI.NumSignBits = std::min(DestLOI.NumSignBits, SrcLOI->NumSignBits);
    DestLOI.KnownZero &= SrcLOI->KnownZero;
    DestLOI.KnownOne &= SrcLOI->KnownOne;
  }
}

/// setArgumentFrameIndex - Record frame index for the byval
/// argument. This overrides previous frame index entry for this argument,
/// if any.
void FunctionLoweringInfo::setArgumentFrameIndex(const Argument *A,
                                                 int FI) {
  ByValArgFrameIndexMap[A] = FI;
}

/// getArgumentFrameIndex - Get frame index for the byval argument.
/// If the argument does not have any assigned frame index then 0 is
/// returned.
int FunctionLoweringInfo::getArgumentFrameIndex(const Argument *A) {
  DenseMap<const Argument *, int>::iterator I =
    ByValArgFrameIndexMap.find(A);
  if (I != ByValArgFrameIndexMap.end())
    return I->second;
  DEBUG(dbgs() << "Argument does not have assigned frame index!\n");
  return 0;
}

/// ComputeUsesVAFloatArgument - Determine if any floating-point values are
/// being passed to this variadic function, and set the MachineModuleInfo's
/// usesVAFloatArgument flag if so. This flag is used to emit an undefined
/// reference to _fltused on Windows, which will link in MSVCRT's
/// floating-point support.
void llvm::ComputeUsesVAFloatArgument(const CallInst &I,
                                      MachineModuleInfo *MMI)
{
  FunctionType *FT = cast<FunctionType>(
    I.getCalledValue()->getType()->getContainedType(0));
  if (FT->isVarArg() && !MMI->usesVAFloatArgument()) {
    for (unsigned i = 0, e = I.getNumArgOperands(); i != e; ++i) {
      Type* T = I.getArgOperand(i)->getType();
      for (auto i : post_order(T)) {
        if (i->isFloatingPointTy()) {
          MMI->setUsesVAFloatArgument(true);
          return;
        }
      }
    }
  }
}

/// AddLandingPadInfo - Extract the exception handling information from the
/// landingpad instruction and add them to the specified machine module info.
void llvm::AddLandingPadInfo(const LandingPadInst &I, MachineModuleInfo &MMI,
                             MachineBasicBlock *MBB) {
  MMI.addPersonality(
      MBB,
      cast<Function>(
          I.getParent()->getParent()->getPersonalityFn()->stripPointerCasts()));

  if (I.isCleanup())
    MMI.addCleanup(MBB);

  // FIXME: New EH - Add the clauses in reverse order. This isn't 100% correct,
  //        but we need to do it this way because of how the DWARF EH emitter
  //        processes the clauses.
  for (unsigned i = I.getNumClauses(); i != 0; --i) {
    Value *Val = I.getClause(i - 1);
    if (I.isCatch(i - 1)) {
      MMI.addCatchTypeInfo(MBB,
                           dyn_cast<GlobalValue>(Val->stripPointerCasts()));
    } else {
      // Add filters in a list.
      Constant *CVal = cast<Constant>(Val);
      SmallVector<const GlobalValue*, 4> FilterList;
      for (User::op_iterator
             II = CVal->op_begin(), IE = CVal->op_end(); II != IE; ++II)
        FilterList.push_back(cast<GlobalValue>((*II)->stripPointerCasts()));

      MMI.addFilterTypeInfo(MBB, FilterList);
    }
  }
}
