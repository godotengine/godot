//===-- llvm/Target/TargetLowering.h - Target Lowering Info -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes how to lower LLVM code to machine code.  This has two
/// main components:
///
///  1. Which ValueTypes are natively supported by the target.
///  2. Which operations are supported for supported ValueTypes.
///  3. Cost thresholds for alternative implementations of certain operations.
///
/// In addition it has a few other components, like information about FP
/// immediates.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETLOWERING_H
#define LLVM_TARGET_TARGETLOWERING_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/DAGCombine.h"
#include "llvm/CodeGen/RuntimeLibcalls.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Target/TargetCallingConv.h"
#include "llvm/Target/TargetMachine.h"
#include <climits>
#include <map>
#include <vector>

namespace llvm {
  class CallInst;
  class CCState;
  class FastISel;
  class FunctionLoweringInfo;
  class ImmutableCallSite;
  class IntrinsicInst;
  class MachineBasicBlock;
  class MachineFunction;
  class MachineInstr;
  class MachineJumpTableInfo;
  class MachineLoop;
  class Mangler;
  class MCContext;
  class MCExpr;
  class MCSymbol;
  template<typename T> class SmallVectorImpl;
  class DataLayout;
  class TargetRegisterClass;
  class TargetLibraryInfo;
  class TargetLoweringObjectFile;
  class Value;

  namespace Sched {
    enum Preference {
      None,             // No preference
      Source,           // Follow source order.
      RegPressure,      // Scheduling for lowest register pressure.
      Hybrid,           // Scheduling for both latency and register pressure.
      ILP,              // Scheduling for ILP in low register pressure mode.
      VLIW              // Scheduling for VLIW targets.
    };
  }

/// This base class for TargetLowering contains the SelectionDAG-independent
/// parts that can be used from the rest of CodeGen.
class TargetLoweringBase {
  TargetLoweringBase(const TargetLoweringBase&) = delete;
  void operator=(const TargetLoweringBase&) = delete;

public:
  /// This enum indicates whether operations are valid for a target, and if not,
  /// what action should be used to make them valid.
  enum LegalizeAction {
    Legal,      // The target natively supports this operation.
    Promote,    // This operation should be executed in a larger type.
    Expand,     // Try to expand this to other ops, otherwise use a libcall.
    Custom      // Use the LowerOperation hook to implement custom lowering.
  };

  /// This enum indicates whether a types are legal for a target, and if not,
  /// what action should be used to make them valid.
  enum LegalizeTypeAction {
    TypeLegal,           // The target natively supports this type.
    TypePromoteInteger,  // Replace this integer with a larger one.
    TypeExpandInteger,   // Split this integer into two of half the size.
    TypeSoftenFloat,     // Convert this float to a same size integer type.
    TypeExpandFloat,     // Split this float into two of half the size.
    TypeScalarizeVector, // Replace this one-element vector with its element.
    TypeSplitVector,     // Split this vector into two of half the size.
    TypeWidenVector,     // This vector should be widened into a larger vector.
    TypePromoteFloat     // Replace this float with a larger one.
  };

  /// LegalizeKind holds the legalization kind that needs to happen to EVT
  /// in order to type-legalize it.
  typedef std::pair<LegalizeTypeAction, EVT> LegalizeKind;

  /// Enum that describes how the target represents true/false values.
  enum BooleanContent {
    UndefinedBooleanContent,    // Only bit 0 counts, the rest can hold garbage.
    ZeroOrOneBooleanContent,        // All bits zero except for bit 0.
    ZeroOrNegativeOneBooleanContent // All bits equal to bit 0.
  };

  /// Enum that describes what type of support for selects the target has.
  enum SelectSupportKind {
    ScalarValSelect,      // The target supports scalar selects (ex: cmov).
    ScalarCondVectorVal,  // The target supports selects with a scalar condition
                          // and vector values (ex: cmov).
    VectorMaskSelect      // The target supports vector selects with a vector
                          // mask (ex: x86 blends).
  };

  /// Enum that specifies what a AtomicRMWInst is expanded to, if at all. Exists
  /// because different targets have different levels of support for these
  /// atomic RMW instructions, and also have different options w.r.t. what they
  /// should expand to.
  enum class AtomicRMWExpansionKind {
    None,      // Don't expand the instruction.
    LLSC,      // Expand the instruction into loadlinked/storeconditional; used
               // by ARM/AArch64. Implies `hasLoadLinkedStoreConditional`
               // returns true.
    CmpXChg,   // Expand the instruction into cmpxchg; used by at least X86.
  };

  static ISD::NodeType getExtendForContent(BooleanContent Content) {
    switch (Content) {
    case UndefinedBooleanContent:
      // Extend by adding rubbish bits.
      return ISD::ANY_EXTEND;
    case ZeroOrOneBooleanContent:
      // Extend by adding zero bits.
      return ISD::ZERO_EXTEND;
    case ZeroOrNegativeOneBooleanContent:
      // Extend by copying the sign bit.
      return ISD::SIGN_EXTEND;
    }
    llvm_unreachable("Invalid content kind");
  }

  /// NOTE: The TargetMachine owns TLOF.
  explicit TargetLoweringBase(const TargetMachine &TM);
  virtual ~TargetLoweringBase() {}

protected:
  /// \brief Initialize all of the actions to default values.
  void initActions();

public:
  const TargetMachine &getTargetMachine() const { return TM; }

  virtual bool useSoftFloat() const { return false; }

  /// Return the pointer type for the given address space, defaults to
  /// the pointer type from the data layout.
  /// FIXME: The default needs to be removed once all the code is updated.
  MVT getPointerTy(const DataLayout &DL, uint32_t AS = 0) const {
    return MVT::getIntegerVT(DL.getPointerSizeInBits(AS));
  }

  /// EVT is not used in-tree, but is used by out-of-tree target.
  /// A documentation for this function would be nice...
  virtual MVT getScalarShiftAmountTy(const DataLayout &, EVT) const;

  EVT getShiftAmountTy(EVT LHSTy, const DataLayout &DL) const;

  /// Returns the type to be used for the index operand of:
  /// ISD::INSERT_VECTOR_ELT, ISD::EXTRACT_VECTOR_ELT,
  /// ISD::INSERT_SUBVECTOR, and ISD::EXTRACT_SUBVECTOR
  virtual MVT getVectorIdxTy(const DataLayout &DL) const {
    return getPointerTy(DL);
  }

  /// Return true if the select operation is expensive for this target.
  bool isSelectExpensive() const { return SelectIsExpensive; }

  virtual bool isSelectSupported(SelectSupportKind /*kind*/) const {
    return true;
  }

  /// Return true if multiple condition registers are available.
  bool hasMultipleConditionRegisters() const {
    return HasMultipleConditionRegisters;
  }

  /// Return true if the target has BitExtract instructions.
  bool hasExtractBitsInsn() const { return HasExtractBitsInsn; }

  /// Return the preferred vector type legalization action.
  virtual TargetLoweringBase::LegalizeTypeAction
  getPreferredVectorAction(EVT VT) const {
    // The default action for one element vectors is to scalarize
    if (VT.getVectorNumElements() == 1)
      return TypeScalarizeVector;
    // The default action for other vectors is to promote
    return TypePromoteInteger;
  }

  // There are two general methods for expanding a BUILD_VECTOR node:
  //  1. Use SCALAR_TO_VECTOR on the defined scalar values and then shuffle
  //     them together.
  //  2. Build the vector on the stack and then load it.
  // If this function returns true, then method (1) will be used, subject to
  // the constraint that all of the necessary shuffles are legal (as determined
  // by isShuffleMaskLegal). If this function returns false, then method (2) is
  // always used. The vector type, and the number of defined values, are
  // provided.
  virtual bool
  shouldExpandBuildVectorWithShuffles(EVT /* VT */,
                                      unsigned DefinedValues) const {
    return DefinedValues < 3;
  }

  /// Return true if integer divide is usually cheaper than a sequence of
  /// several shifts, adds, and multiplies for this target.
  bool isIntDivCheap() const { return IntDivIsCheap; }

  /// Return true if sqrt(x) is as cheap or cheaper than 1 / rsqrt(x)
  bool isFsqrtCheap() const {
    return FsqrtIsCheap;
  }

  /// Returns true if target has indicated at least one type should be bypassed.
  bool isSlowDivBypassed() const { return !BypassSlowDivWidths.empty(); }

  /// Returns map of slow types for division or remainder with corresponding
  /// fast types
  const DenseMap<unsigned int, unsigned int> &getBypassSlowDivWidths() const {
    return BypassSlowDivWidths;
  }

  /// Return true if pow2 sdiv is cheaper than a chain of sra/srl/add/sra.
  bool isPow2SDivCheap() const { return Pow2SDivIsCheap; }

  /// Return true if Flow Control is an expensive operation that should be
  /// avoided.
  bool isJumpExpensive() const { return JumpIsExpensive; }

  /// Return true if selects are only cheaper than branches if the branch is
  /// unlikely to be predicted right.
  bool isPredictableSelectExpensive() const {
    return PredictableSelectIsExpensive;
  }

  /// isLoadBitCastBeneficial() - Return true if the following transform
  /// is beneficial.
  /// fold (conv (load x)) -> (load (conv*)x)
  /// On architectures that don't natively support some vector loads
  /// efficiently, casting the load to a smaller vector of larger types and
  /// loading is more efficient, however, this can be undone by optimizations in
  /// dag combiner.
  virtual bool isLoadBitCastBeneficial(EVT /* Load */,
                                       EVT /* Bitcast */) const {
    return true;
  }

  /// Return true if it is expected to be cheaper to do a store of a non-zero
  /// vector constant with the given size and type for the address space than to
  /// store the individual scalar element constants.
  virtual bool storeOfVectorConstantIsCheap(EVT MemVT,
                                            unsigned NumElem,
                                            unsigned AddrSpace) const {
    return false;
  }

  /// \brief Return true if it is cheap to speculate a call to intrinsic cttz.
  virtual bool isCheapToSpeculateCttz() const {
    return false;
  }

  /// \brief Return true if it is cheap to speculate a call to intrinsic ctlz.
  virtual bool isCheapToSpeculateCtlz() const {
    return false;
  }

  /// \brief Return if the target supports combining a
  /// chain like:
  /// \code
  ///   %andResult = and %val1, #imm-with-one-bit-set;
  ///   %icmpResult = icmp %andResult, 0
  ///   br i1 %icmpResult, label %dest1, label %dest2
  /// \endcode
  /// into a single machine instruction of a form like:
  /// \code
  ///   brOnBitSet %register, #bitNumber, dest
  /// \endcode
  bool isMaskAndBranchFoldingLegal() const {
    return MaskAndBranchFoldingIsLegal;
  }

  /// \brief Return true if the target wants to use the optimization that
  /// turns ext(promotableInst1(...(promotableInstN(load)))) into
  /// promotedInst1(...(promotedInstN(ext(load)))).
  bool enableExtLdPromotion() const { return EnableExtLdPromotion; }

  /// Return true if the target can combine store(extractelement VectorTy,
  /// Idx).
  /// \p Cost[out] gives the cost of that transformation when this is true.
  virtual bool canCombineStoreAndExtract(Type *VectorTy, Value *Idx,
                                         unsigned &Cost) const {
    return false;
  }

  /// Return true if target supports floating point exceptions.
  bool hasFloatingPointExceptions() const {
    return HasFloatingPointExceptions;
  }

  /// Return true if target always beneficiates from combining into FMA for a
  /// given value type. This must typically return false on targets where FMA
  /// takes more cycles to execute than FADD.
  virtual bool enableAggressiveFMAFusion(EVT VT) const {
    return false;
  }

  /// Return the ValueType of the result of SETCC operations.
  virtual EVT getSetCCResultType(const DataLayout &DL, LLVMContext &Context,
                                 EVT VT) const;

  /// Return the ValueType for comparison libcalls. Comparions libcalls include
  /// floating point comparion calls, and Ordered/Unordered check calls on
  /// floating point numbers.
  virtual
  MVT::SimpleValueType getCmpLibcallReturnType() const;

  /// For targets without i1 registers, this gives the nature of the high-bits
  /// of boolean values held in types wider than i1.
  ///
  /// "Boolean values" are special true/false values produced by nodes like
  /// SETCC and consumed (as the condition) by nodes like SELECT and BRCOND.
  /// Not to be confused with general values promoted from i1.  Some cpus
  /// distinguish between vectors of boolean and scalars; the isVec parameter
  /// selects between the two kinds.  For example on X86 a scalar boolean should
  /// be zero extended from i1, while the elements of a vector of booleans
  /// should be sign extended from i1.
  ///
  /// Some cpus also treat floating point types the same way as they treat
  /// vectors instead of the way they treat scalars.
  BooleanContent getBooleanContents(bool isVec, bool isFloat) const {
    if (isVec)
      return BooleanVectorContents;
    return isFloat ? BooleanFloatContents : BooleanContents;
  }

  BooleanContent getBooleanContents(EVT Type) const {
    return getBooleanContents(Type.isVector(), Type.isFloatingPoint());
  }

  /// Return target scheduling preference.
  Sched::Preference getSchedulingPreference() const {
    return SchedPreferenceInfo;
  }

  /// Some scheduler, e.g. hybrid, can switch to different scheduling heuristics
  /// for different nodes. This function returns the preference (or none) for
  /// the given node.
  virtual Sched::Preference getSchedulingPreference(SDNode *) const {
    return Sched::None;
  }

  /// Return the register class that should be used for the specified value
  /// type.
  virtual const TargetRegisterClass *getRegClassFor(MVT VT) const {
    const TargetRegisterClass *RC = RegClassForVT[VT.SimpleTy];
    assert(RC && "This value type is not natively supported!");
    return RC;
  }

  /// Return the 'representative' register class for the specified value
  /// type.
  ///
  /// The 'representative' register class is the largest legal super-reg
  /// register class for the register class of the value type.  For example, on
  /// i386 the rep register class for i8, i16, and i32 are GR32; while the rep
  /// register class is GR64 on x86_64.
  virtual const TargetRegisterClass *getRepRegClassFor(MVT VT) const {
    const TargetRegisterClass *RC = RepRegClassForVT[VT.SimpleTy];
    return RC;
  }

  /// Return the cost of the 'representative' register class for the specified
  /// value type.
  virtual uint8_t getRepRegClassCostFor(MVT VT) const {
    return RepRegClassCostForVT[VT.SimpleTy];
  }

  /// Return true if the target has native support for the specified value type.
  /// This means that it has a register that directly holds it without
  /// promotions or expansions.
  bool isTypeLegal(EVT VT) const {
    assert(!VT.isSimple() ||
           (unsigned)VT.getSimpleVT().SimpleTy < array_lengthof(RegClassForVT));
    return VT.isSimple() && RegClassForVT[VT.getSimpleVT().SimpleTy] != nullptr;
  }

  class ValueTypeActionImpl {
    /// ValueTypeActions - For each value type, keep a LegalizeTypeAction enum
    /// that indicates how instruction selection should deal with the type.
    uint8_t ValueTypeActions[MVT::LAST_VALUETYPE];

  public:
    ValueTypeActionImpl() {
      std::fill(std::begin(ValueTypeActions), std::end(ValueTypeActions), 0);
    }

    LegalizeTypeAction getTypeAction(MVT VT) const {
      return (LegalizeTypeAction)ValueTypeActions[VT.SimpleTy];
    }

    void setTypeAction(MVT VT, LegalizeTypeAction Action) {
      unsigned I = VT.SimpleTy;
      ValueTypeActions[I] = Action;
    }
  };

  const ValueTypeActionImpl &getValueTypeActions() const {
    return ValueTypeActions;
  }

  /// Return how we should legalize values of this type, either it is already
  /// legal (return 'Legal') or we need to promote it to a larger type (return
  /// 'Promote'), or we need to expand it into multiple registers of smaller
  /// integer type (return 'Expand').  'Custom' is not an option.
  LegalizeTypeAction getTypeAction(LLVMContext &Context, EVT VT) const {
    return getTypeConversion(Context, VT).first;
  }
  LegalizeTypeAction getTypeAction(MVT VT) const {
    return ValueTypeActions.getTypeAction(VT);
  }

  /// For types supported by the target, this is an identity function.  For
  /// types that must be promoted to larger types, this returns the larger type
  /// to promote to.  For integer types that are larger than the largest integer
  /// register, this contains one step in the expansion to get to the smaller
  /// register. For illegal floating point types, this returns the integer type
  /// to transform to.
  EVT getTypeToTransformTo(LLVMContext &Context, EVT VT) const {
    return getTypeConversion(Context, VT).second;
  }

  /// For types supported by the target, this is an identity function.  For
  /// types that must be expanded (i.e. integer types that are larger than the
  /// largest integer register or illegal floating point types), this returns
  /// the largest legal type it will be expanded to.
  EVT getTypeToExpandTo(LLVMContext &Context, EVT VT) const {
    assert(!VT.isVector());
    while (true) {
      switch (getTypeAction(Context, VT)) {
      case TypeLegal:
        return VT;
      case TypeExpandInteger:
        VT = getTypeToTransformTo(Context, VT);
        break;
      default:
        llvm_unreachable("Type is not legal nor is it to be expanded!");
      }
    }
  }

  /// Vector types are broken down into some number of legal first class types.
  /// For example, EVT::v8f32 maps to 2 EVT::v4f32 with Altivec or SSE1, or 8
  /// promoted EVT::f64 values with the X86 FP stack.  Similarly, EVT::v2i64
  /// turns into 4 EVT::i32 values with both PPC and X86.
  ///
  /// This method returns the number of registers needed, and the VT for each
  /// register.  It also returns the VT and quantity of the intermediate values
  /// before they are promoted/expanded.
  unsigned getVectorTypeBreakdown(LLVMContext &Context, EVT VT,
                                  EVT &IntermediateVT,
                                  unsigned &NumIntermediates,
                                  MVT &RegisterVT) const;

  struct IntrinsicInfo {
    unsigned     opc;         // target opcode
    EVT          memVT;       // memory VT
    const Value* ptrVal;      // value representing memory location
    int          offset;      // offset off of ptrVal
    unsigned     size;        // the size of the memory location
                              // (taken from memVT if zero)
    unsigned     align;       // alignment
    bool         vol;         // is volatile?
    bool         readMem;     // reads memory?
    bool         writeMem;    // writes memory?

    IntrinsicInfo() : opc(0), ptrVal(nullptr), offset(0), size(0), align(1),
                      vol(false), readMem(false), writeMem(false) {}
  };

  /// Given an intrinsic, checks if on the target the intrinsic will need to map
  /// to a MemIntrinsicNode (touches memory). If this is the case, it returns
  /// true and store the intrinsic information into the IntrinsicInfo that was
  /// passed to the function.
  virtual bool getTgtMemIntrinsic(IntrinsicInfo &, const CallInst &,
                                  unsigned /*Intrinsic*/) const {
    return false;
  }

  /// Returns true if the target can instruction select the specified FP
  /// immediate natively. If false, the legalizer will materialize the FP
  /// immediate as a load from a constant pool.
  virtual bool isFPImmLegal(const APFloat &/*Imm*/, EVT /*VT*/) const {
    return false;
  }

  /// Targets can use this to indicate that they only support *some*
  /// VECTOR_SHUFFLE operations, those with specific masks.  By default, if a
  /// target supports the VECTOR_SHUFFLE node, all mask values are assumed to be
  /// legal.
  virtual bool isShuffleMaskLegal(const SmallVectorImpl<int> &/*Mask*/,
                                  EVT /*VT*/) const {
    return true;
  }

  /// Returns true if the operation can trap for the value type.
  ///
  /// VT must be a legal type. By default, we optimistically assume most
  /// operations don't trap except for divide and remainder.
  virtual bool canOpTrap(unsigned Op, EVT VT) const;

  /// Similar to isShuffleMaskLegal. This is used by Targets can use this to
  /// indicate if there is a suitable VECTOR_SHUFFLE that can be used to replace
  /// a VAND with a constant pool entry.
  virtual bool isVectorClearMaskLegal(const SmallVectorImpl<int> &/*Mask*/,
                                      EVT /*VT*/) const {
    return false;
  }

  /// Return how this operation should be treated: either it is legal, needs to
  /// be promoted to a larger size, needs to be expanded to some other code
  /// sequence, or the target has a custom expander for it.
  LegalizeAction getOperationAction(unsigned Op, EVT VT) const {
    if (VT.isExtended()) return Expand;
    // If a target-specific SDNode requires legalization, require the target
    // to provide custom legalization for it.
    if (Op > array_lengthof(OpActions[0])) return Custom;
    unsigned I = (unsigned) VT.getSimpleVT().SimpleTy;
    return (LegalizeAction)OpActions[I][Op];
  }

  /// Return true if the specified operation is legal on this target or can be
  /// made legal with custom lowering. This is used to help guide high-level
  /// lowering decisions.
  bool isOperationLegalOrCustom(unsigned Op, EVT VT) const {
    return (VT == MVT::Other || isTypeLegal(VT)) &&
      (getOperationAction(Op, VT) == Legal ||
       getOperationAction(Op, VT) == Custom);
  }

  /// Return true if the specified operation is legal on this target or can be
  /// made legal using promotion. This is used to help guide high-level lowering
  /// decisions.
  bool isOperationLegalOrPromote(unsigned Op, EVT VT) const {
    return (VT == MVT::Other || isTypeLegal(VT)) &&
      (getOperationAction(Op, VT) == Legal ||
       getOperationAction(Op, VT) == Promote);
  }

  /// Return true if the specified operation is illegal on this target or
  /// unlikely to be made legal with custom lowering. This is used to help guide
  /// high-level lowering decisions.
  bool isOperationExpand(unsigned Op, EVT VT) const {
    return (!isTypeLegal(VT) || getOperationAction(Op, VT) == Expand);
  }

  /// Return true if the specified operation is legal on this target.
  bool isOperationLegal(unsigned Op, EVT VT) const {
    return (VT == MVT::Other || isTypeLegal(VT)) &&
           getOperationAction(Op, VT) == Legal;
  }

  /// Return how this load with extension should be treated: either it is legal,
  /// needs to be promoted to a larger size, needs to be expanded to some other
  /// code sequence, or the target has a custom expander for it.
  LegalizeAction getLoadExtAction(unsigned ExtType, EVT ValVT,
                                  EVT MemVT) const {
    if (ValVT.isExtended() || MemVT.isExtended()) return Expand;
    unsigned ValI = (unsigned) ValVT.getSimpleVT().SimpleTy;
    unsigned MemI = (unsigned) MemVT.getSimpleVT().SimpleTy;
    assert(ExtType < ISD::LAST_LOADEXT_TYPE && ValI < MVT::LAST_VALUETYPE &&
           MemI < MVT::LAST_VALUETYPE && "Table isn't big enough!");
    return (LegalizeAction)LoadExtActions[ValI][MemI][ExtType];
  }

  /// Return true if the specified load with extension is legal on this target.
  bool isLoadExtLegal(unsigned ExtType, EVT ValVT, EVT MemVT) const {
    return ValVT.isSimple() && MemVT.isSimple() &&
      getLoadExtAction(ExtType, ValVT, MemVT) == Legal;
  }

  /// Return true if the specified load with extension is legal or custom
  /// on this target.
  bool isLoadExtLegalOrCustom(unsigned ExtType, EVT ValVT, EVT MemVT) const {
    return ValVT.isSimple() && MemVT.isSimple() &&
      (getLoadExtAction(ExtType, ValVT, MemVT) == Legal ||
       getLoadExtAction(ExtType, ValVT, MemVT) == Custom);
  }

  /// Return how this store with truncation should be treated: either it is
  /// legal, needs to be promoted to a larger size, needs to be expanded to some
  /// other code sequence, or the target has a custom expander for it.
  LegalizeAction getTruncStoreAction(EVT ValVT, EVT MemVT) const {
    if (ValVT.isExtended() || MemVT.isExtended()) return Expand;
    unsigned ValI = (unsigned) ValVT.getSimpleVT().SimpleTy;
    unsigned MemI = (unsigned) MemVT.getSimpleVT().SimpleTy;
    assert(ValI < MVT::LAST_VALUETYPE && MemI < MVT::LAST_VALUETYPE &&
           "Table isn't big enough!");
    return (LegalizeAction)TruncStoreActions[ValI][MemI];
  }

  /// Return true if the specified store with truncation is legal on this
  /// target.
  bool isTruncStoreLegal(EVT ValVT, EVT MemVT) const {
    return isTypeLegal(ValVT) && MemVT.isSimple() &&
      getTruncStoreAction(ValVT.getSimpleVT(), MemVT.getSimpleVT()) == Legal;
  }

  /// Return how the indexed load should be treated: either it is legal, needs
  /// to be promoted to a larger size, needs to be expanded to some other code
  /// sequence, or the target has a custom expander for it.
  LegalizeAction
  getIndexedLoadAction(unsigned IdxMode, MVT VT) const {
    assert(IdxMode < ISD::LAST_INDEXED_MODE && VT.isValid() &&
           "Table isn't big enough!");
    unsigned Ty = (unsigned)VT.SimpleTy;
    return (LegalizeAction)((IndexedModeActions[Ty][IdxMode] & 0xf0) >> 4);
  }

  /// Return true if the specified indexed load is legal on this target.
  bool isIndexedLoadLegal(unsigned IdxMode, EVT VT) const {
    return VT.isSimple() &&
      (getIndexedLoadAction(IdxMode, VT.getSimpleVT()) == Legal ||
       getIndexedLoadAction(IdxMode, VT.getSimpleVT()) == Custom);
  }

  /// Return how the indexed store should be treated: either it is legal, needs
  /// to be promoted to a larger size, needs to be expanded to some other code
  /// sequence, or the target has a custom expander for it.
  LegalizeAction
  getIndexedStoreAction(unsigned IdxMode, MVT VT) const {
    assert(IdxMode < ISD::LAST_INDEXED_MODE && VT.isValid() &&
           "Table isn't big enough!");
    unsigned Ty = (unsigned)VT.SimpleTy;
    return (LegalizeAction)(IndexedModeActions[Ty][IdxMode] & 0x0f);
  }

  /// Return true if the specified indexed load is legal on this target.
  bool isIndexedStoreLegal(unsigned IdxMode, EVT VT) const {
    return VT.isSimple() &&
      (getIndexedStoreAction(IdxMode, VT.getSimpleVT()) == Legal ||
       getIndexedStoreAction(IdxMode, VT.getSimpleVT()) == Custom);
  }

  /// Return how the condition code should be treated: either it is legal, needs
  /// to be expanded to some other code sequence, or the target has a custom
  /// expander for it.
  LegalizeAction
  getCondCodeAction(ISD::CondCode CC, MVT VT) const {
    assert((unsigned)CC < array_lengthof(CondCodeActions) &&
           ((unsigned)VT.SimpleTy >> 4) < array_lengthof(CondCodeActions[0]) &&
           "Table isn't big enough!");
    // See setCondCodeAction for how this is encoded.
    uint32_t Shift = 2 * (VT.SimpleTy & 0xF);
    uint32_t Value = CondCodeActions[CC][VT.SimpleTy >> 4];
    LegalizeAction Action = (LegalizeAction) ((Value >> Shift) & 0x3);
    assert(Action != Promote && "Can't promote condition code!");
    return Action;
  }

  /// Return true if the specified condition code is legal on this target.
  bool isCondCodeLegal(ISD::CondCode CC, MVT VT) const {
    return
      getCondCodeAction(CC, VT) == Legal ||
      getCondCodeAction(CC, VT) == Custom;
  }


  /// If the action for this operation is to promote, this method returns the
  /// ValueType to promote to.
  MVT getTypeToPromoteTo(unsigned Op, MVT VT) const {
    assert(getOperationAction(Op, VT) == Promote &&
           "This operation isn't promoted!");

    // See if this has an explicit type specified.
    std::map<std::pair<unsigned, MVT::SimpleValueType>,
             MVT::SimpleValueType>::const_iterator PTTI =
      PromoteToType.find(std::make_pair(Op, VT.SimpleTy));
    if (PTTI != PromoteToType.end()) return PTTI->second;

    assert((VT.isInteger() || VT.isFloatingPoint()) &&
           "Cannot autopromote this type, add it with AddPromotedToType.");

    MVT NVT = VT;
    do {
      NVT = (MVT::SimpleValueType)(NVT.SimpleTy+1);
      assert(NVT.isInteger() == VT.isInteger() && NVT != MVT::isVoid &&
             "Didn't find type to promote to!");
    } while (!isTypeLegal(NVT) ||
              getOperationAction(Op, NVT) == Promote);
    return NVT;
  }

  /// Return the EVT corresponding to this LLVM type.  This is fixed by the LLVM
  /// operations except for the pointer size.  If AllowUnknown is true, this
  /// will return MVT::Other for types with no EVT counterpart (e.g. structs),
  /// otherwise it will assert.
  EVT getValueType(const DataLayout &DL, Type *Ty,
                   bool AllowUnknown = false) const {
    // Lower scalar pointers to native pointer types.
    if (PointerType *PTy = dyn_cast<PointerType>(Ty))
      return getPointerTy(DL, PTy->getAddressSpace());

    if (Ty->isVectorTy()) {
      VectorType *VTy = cast<VectorType>(Ty);
      Type *Elm = VTy->getElementType();
      // Lower vectors of pointers to native pointer types.
      if (PointerType *PT = dyn_cast<PointerType>(Elm)) {
        EVT PointerTy(getPointerTy(DL, PT->getAddressSpace()));
        Elm = PointerTy.getTypeForEVT(Ty->getContext());
      }

      return EVT::getVectorVT(Ty->getContext(), EVT::getEVT(Elm, false),
                       VTy->getNumElements());
    }
    return EVT::getEVT(Ty, AllowUnknown);
  }

  /// Return the MVT corresponding to this LLVM type. See getValueType.
  MVT getSimpleValueType(const DataLayout &DL, Type *Ty,
                         bool AllowUnknown = false) const {
    return getValueType(DL, Ty, AllowUnknown).getSimpleVT();
  }

  /// Return the desired alignment for ByVal or InAlloca aggregate function
  /// arguments in the caller parameter area.  This is the actual alignment, not
  /// its logarithm.
  virtual unsigned getByValTypeAlignment(Type *Ty, const DataLayout &DL) const;

  /// Return the type of registers that this ValueType will eventually require.
  MVT getRegisterType(MVT VT) const {
    assert((unsigned)VT.SimpleTy < array_lengthof(RegisterTypeForVT));
    return RegisterTypeForVT[VT.SimpleTy];
  }

  /// Return the type of registers that this ValueType will eventually require.
  MVT getRegisterType(LLVMContext &Context, EVT VT) const {
    if (VT.isSimple()) {
      assert((unsigned)VT.getSimpleVT().SimpleTy <
                array_lengthof(RegisterTypeForVT));
      return RegisterTypeForVT[VT.getSimpleVT().SimpleTy];
    }
    if (VT.isVector()) {
      EVT VT1;
      MVT RegisterVT;
      unsigned NumIntermediates;
      (void)getVectorTypeBreakdown(Context, VT, VT1,
                                   NumIntermediates, RegisterVT);
      return RegisterVT;
    }
    if (VT.isInteger()) {
      return getRegisterType(Context, getTypeToTransformTo(Context, VT));
    }
    llvm_unreachable("Unsupported extended type!");
  }

  /// Return the number of registers that this ValueType will eventually
  /// require.
  ///
  /// This is one for any types promoted to live in larger registers, but may be
  /// more than one for types (like i64) that are split into pieces.  For types
  /// like i140, which are first promoted then expanded, it is the number of
  /// registers needed to hold all the bits of the original type.  For an i140
  /// on a 32 bit machine this means 5 registers.
  unsigned getNumRegisters(LLVMContext &Context, EVT VT) const {
    if (VT.isSimple()) {
      assert((unsigned)VT.getSimpleVT().SimpleTy <
                array_lengthof(NumRegistersForVT));
      return NumRegistersForVT[VT.getSimpleVT().SimpleTy];
    }
    if (VT.isVector()) {
      EVT VT1;
      MVT VT2;
      unsigned NumIntermediates;
      return getVectorTypeBreakdown(Context, VT, VT1, NumIntermediates, VT2);
    }
    if (VT.isInteger()) {
      unsigned BitWidth = VT.getSizeInBits();
      unsigned RegWidth = getRegisterType(Context, VT).getSizeInBits();
      return (BitWidth + RegWidth - 1) / RegWidth;
    }
    llvm_unreachable("Unsupported extended type!");
  }

  /// If true, then instruction selection should seek to shrink the FP constant
  /// of the specified type to a smaller type in order to save space and / or
  /// reduce runtime.
  virtual bool ShouldShrinkFPConstant(EVT) const { return true; }

  // Return true if it is profitable to reduce the given load node to a smaller
  // type.
  //
  // e.g. (i16 (trunc (i32 (load x))) -> i16 load x should be performed
  virtual bool shouldReduceLoadWidth(SDNode *Load,
                                     ISD::LoadExtType ExtTy,
                                     EVT NewVT) const {
    return true;
  }

  /// When splitting a value of the specified type into parts, does the Lo
  /// or Hi part come first?  This usually follows the endianness, except
  /// for ppcf128, where the Hi part always comes first.
  bool hasBigEndianPartOrdering(EVT VT, const DataLayout &DL) const {
    return DL.isBigEndian() || VT == MVT::ppcf128;
  }

  /// If true, the target has custom DAG combine transformations that it can
  /// perform for the specified node.
  bool hasTargetDAGCombine(ISD::NodeType NT) const {
    assert(unsigned(NT >> 3) < array_lengthof(TargetDAGCombineArray));
    return TargetDAGCombineArray[NT >> 3] & (1 << (NT&7));
  }

  /// \brief Get maximum # of store operations permitted for llvm.memset
  ///
  /// This function returns the maximum number of store operations permitted
  /// to replace a call to llvm.memset. The value is set by the target at the
  /// performance threshold for such a replacement. If OptSize is true,
  /// return the limit for functions that have OptSize attribute.
  unsigned getMaxStoresPerMemset(bool OptSize) const {
    return OptSize ? MaxStoresPerMemsetOptSize : MaxStoresPerMemset;
  }

  /// \brief Get maximum # of store operations permitted for llvm.memcpy
  ///
  /// This function returns the maximum number of store operations permitted
  /// to replace a call to llvm.memcpy. The value is set by the target at the
  /// performance threshold for such a replacement. If OptSize is true,
  /// return the limit for functions that have OptSize attribute.
  unsigned getMaxStoresPerMemcpy(bool OptSize) const {
    return OptSize ? MaxStoresPerMemcpyOptSize : MaxStoresPerMemcpy;
  }

  /// \brief Get maximum # of store operations permitted for llvm.memmove
  ///
  /// This function returns the maximum number of store operations permitted
  /// to replace a call to llvm.memmove. The value is set by the target at the
  /// performance threshold for such a replacement. If OptSize is true,
  /// return the limit for functions that have OptSize attribute.
  unsigned getMaxStoresPerMemmove(bool OptSize) const {
    return OptSize ? MaxStoresPerMemmoveOptSize : MaxStoresPerMemmove;
  }

  /// \brief Determine if the target supports unaligned memory accesses.
  ///
  /// This function returns true if the target allows unaligned memory accesses
  /// of the specified type in the given address space. If true, it also returns
  /// whether the unaligned memory access is "fast" in the last argument by
  /// reference. This is used, for example, in situations where an array
  /// copy/move/set is converted to a sequence of store operations. Its use
  /// helps to ensure that such replacements don't generate code that causes an
  /// alignment error (trap) on the target machine.
  virtual bool allowsMisalignedMemoryAccesses(EVT,
                                              unsigned AddrSpace = 0,
                                              unsigned Align = 1,
                                              bool * /*Fast*/ = nullptr) const {
    return false;
  }

  /// Returns the target specific optimal type for load and store operations as
  /// a result of memset, memcpy, and memmove lowering.
  ///
  /// If DstAlign is zero that means it's safe to destination alignment can
  /// satisfy any constraint. Similarly if SrcAlign is zero it means there isn't
  /// a need to check it against alignment requirement, probably because the
  /// source does not need to be loaded. If 'IsMemset' is true, that means it's
  /// expanding a memset. If 'ZeroMemset' is true, that means it's a memset of
  /// zero. 'MemcpyStrSrc' indicates whether the memcpy source is constant so it
  /// does not need to be loaded.  It returns EVT::Other if the type should be
  /// determined using generic target-independent logic.
  virtual EVT getOptimalMemOpType(uint64_t /*Size*/,
                                  unsigned /*DstAlign*/, unsigned /*SrcAlign*/,
                                  bool /*IsMemset*/,
                                  bool /*ZeroMemset*/,
                                  bool /*MemcpyStrSrc*/,
                                  MachineFunction &/*MF*/) const {
    return MVT::Other;
  }

  /// Returns true if it's safe to use load / store of the specified type to
  /// expand memcpy / memset inline.
  ///
  /// This is mostly true for all types except for some special cases. For
  /// example, on X86 targets without SSE2 f64 load / store are done with fldl /
  /// fstpl which also does type conversion. Note the specified type doesn't
  /// have to be legal as the hook is used before type legalization.
  virtual bool isSafeMemOpType(MVT /*VT*/) const { return true; }

  /// Determine if we should use _setjmp or setjmp to implement llvm.setjmp.
  bool usesUnderscoreSetJmp() const {
    return UseUnderscoreSetJmp;
  }

  /// Determine if we should use _longjmp or longjmp to implement llvm.longjmp.
  bool usesUnderscoreLongJmp() const {
    return UseUnderscoreLongJmp;
  }

  /// Return integer threshold on number of blocks to use jump tables rather
  /// than if sequence.
  int getMinimumJumpTableEntries() const {
    return MinimumJumpTableEntries;
  }

  /// If a physical register, this specifies the register that
  /// llvm.savestack/llvm.restorestack should save and restore.
  unsigned getStackPointerRegisterToSaveRestore() const {
    return StackPointerRegisterToSaveRestore;
  }

  /// If a physical register, this returns the register that receives the
  /// exception address on entry to a landing pad.
  unsigned getExceptionPointerRegister() const {
    return ExceptionPointerRegister;
  }

  /// If a physical register, this returns the register that receives the
  /// exception typeid on entry to a landing pad.
  unsigned getExceptionSelectorRegister() const {
    return ExceptionSelectorRegister;
  }

  /// Returns the target's jmp_buf size in bytes (if never set, the default is
  /// 200)
  unsigned getJumpBufSize() const {
    return JumpBufSize;
  }

  /// Returns the target's jmp_buf alignment in bytes (if never set, the default
  /// is 0)
  unsigned getJumpBufAlignment() const {
    return JumpBufAlignment;
  }

  /// Return the minimum stack alignment of an argument.
  unsigned getMinStackArgumentAlignment() const {
    return MinStackArgumentAlignment;
  }

  /// Return the minimum function alignment.
  unsigned getMinFunctionAlignment() const {
    return MinFunctionAlignment;
  }

  /// Return the preferred function alignment.
  unsigned getPrefFunctionAlignment() const {
    return PrefFunctionAlignment;
  }

  /// Return the preferred loop alignment.
  virtual unsigned getPrefLoopAlignment(MachineLoop *ML = nullptr) const {
    return PrefLoopAlignment;
  }

  /// Return whether the DAG builder should automatically insert fences and
  /// reduce ordering for atomics.
  bool getInsertFencesForAtomic() const {
    return InsertFencesForAtomic;
  }

  /// Return true if the target stores stack protector cookies at a fixed offset
  /// in some non-standard address space, and populates the address space and
  /// offset as appropriate.
  virtual bool getStackCookieLocation(unsigned &/*AddressSpace*/,
                                      unsigned &/*Offset*/) const {
    return false;
  }

  /// Returns true if a cast between SrcAS and DestAS is a noop.
  virtual bool isNoopAddrSpaceCast(unsigned SrcAS, unsigned DestAS) const {
    return false;
  }

  /// Return true if the pointer arguments to CI should be aligned by aligning
  /// the object whose address is being passed. If so then MinSize is set to the
  /// minimum size the object must be to be aligned and PrefAlign is set to the
  /// preferred alignment.
  virtual bool shouldAlignPointerArgs(CallInst * /*CI*/, unsigned & /*MinSize*/,
                                      unsigned & /*PrefAlign*/) const {
    return false;
  }

  //===--------------------------------------------------------------------===//
  /// \name Helpers for TargetTransformInfo implementations
  /// @{

  /// Get the ISD node that corresponds to the Instruction class opcode.
  int InstructionOpcodeToISD(unsigned Opcode) const;

  /// Estimate the cost of type-legalization and the legalized type.
  std::pair<unsigned, MVT> getTypeLegalizationCost(const DataLayout &DL,
                                                   Type *Ty) const;

  /// @}

  //===--------------------------------------------------------------------===//
  /// \name Helpers for atomic expansion.
  /// @{

  /// True if AtomicExpandPass should use emitLoadLinked/emitStoreConditional
  /// and expand AtomicCmpXchgInst.
  virtual bool hasLoadLinkedStoreConditional() const { return false; }

  /// Perform a load-linked operation on Addr, returning a "Value *" with the
  /// corresponding pointee type. This may entail some non-trivial operations to
  /// truncate or reconstruct types that will be illegal in the backend. See
  /// ARMISelLowering for an example implementation.
  virtual Value *emitLoadLinked(IRBuilder<> &Builder, Value *Addr,
                                AtomicOrdering Ord) const {
    llvm_unreachable("Load linked unimplemented on this target");
  }

  /// Perform a store-conditional operation to Addr. Return the status of the
  /// store. This should be 0 if the store succeeded, non-zero otherwise.
  virtual Value *emitStoreConditional(IRBuilder<> &Builder, Value *Val,
                                      Value *Addr, AtomicOrdering Ord) const {
    llvm_unreachable("Store conditional unimplemented on this target");
  }

  /// Inserts in the IR a target-specific intrinsic specifying a fence.
  /// It is called by AtomicExpandPass before expanding an
  ///   AtomicRMW/AtomicCmpXchg/AtomicStore/AtomicLoad.
  /// RMW and CmpXchg set both IsStore and IsLoad to true.
  /// This function should either return a nullptr, or a pointer to an IR-level
  ///   Instruction*. Even complex fence sequences can be represented by a
  ///   single Instruction* through an intrinsic to be lowered later.
  /// Backends with !getInsertFencesForAtomic() should keep a no-op here.
  /// Backends should override this method to produce target-specific intrinsic
  ///   for their fences.
  /// FIXME: Please note that the default implementation here in terms of
  ///   IR-level fences exists for historical/compatibility reasons and is
  ///   *unsound* ! Fences cannot, in general, be used to restore sequential
  ///   consistency. For example, consider the following example:
  /// atomic<int> x = y = 0;
  /// int r1, r2, r3, r4;
  /// Thread 0:
  ///   x.store(1);
  /// Thread 1:
  ///   y.store(1);
  /// Thread 2:
  ///   r1 = x.load();
  ///   r2 = y.load();
  /// Thread 3:
  ///   r3 = y.load();
  ///   r4 = x.load();
  ///  r1 = r3 = 1 and r2 = r4 = 0 is impossible as long as the accesses are all
  ///  seq_cst. But if they are lowered to monotonic accesses, no amount of
  ///  IR-level fences can prevent it.
  /// @{
  virtual Instruction *emitLeadingFence(IRBuilder<> &Builder,
                                        AtomicOrdering Ord, bool IsStore,
                                        bool IsLoad) const {
    if (!getInsertFencesForAtomic())
      return nullptr;

    if (isAtLeastRelease(Ord) && IsStore)
      return Builder.CreateFence(Ord);
    else
      return nullptr;
  }

  virtual Instruction *emitTrailingFence(IRBuilder<> &Builder,
                                         AtomicOrdering Ord, bool IsStore,
                                         bool IsLoad) const {
    if (!getInsertFencesForAtomic())
      return nullptr;

    if (isAtLeastAcquire(Ord))
      return Builder.CreateFence(Ord);
    else
      return nullptr;
  }
  /// @}

  /// Returns true if the given (atomic) store should be expanded by the
  /// IR-level AtomicExpand pass into an "atomic xchg" which ignores its input.
  virtual bool shouldExpandAtomicStoreInIR(StoreInst *SI) const {
    return false;
  }

  /// Returns true if arguments should be sign-extended in lib calls.
  virtual bool shouldSignExtendTypeInLibCall(EVT Type, bool IsSigned) const {
    return IsSigned;
 }

  /// Returns true if the given (atomic) load should be expanded by the
  /// IR-level AtomicExpand pass into a load-linked instruction
  /// (through emitLoadLinked()).
  virtual bool shouldExpandAtomicLoadInIR(LoadInst *LI) const { return false; }

  /// Returns how the IR-level AtomicExpand pass should expand the given
  /// AtomicRMW, if at all. Default is to never expand.
  virtual AtomicRMWExpansionKind
  shouldExpandAtomicRMWInIR(AtomicRMWInst *) const {
    return AtomicRMWExpansionKind::None;
  }

  /// On some platforms, an AtomicRMW that never actually modifies the value
  /// (such as fetch_add of 0) can be turned into a fence followed by an
  /// atomic load. This may sound useless, but it makes it possible for the
  /// processor to keep the cacheline shared, dramatically improving
  /// performance. And such idempotent RMWs are useful for implementing some
  /// kinds of locks, see for example (justification + benchmarks):
  /// http://www.hpl.hp.com/techreports/2012/HPL-2012-68.pdf
  /// This method tries doing that transformation, returning the atomic load if
  /// it succeeds, and nullptr otherwise.
  /// If shouldExpandAtomicLoadInIR returns true on that load, it will undergo
  /// another round of expansion.
  virtual LoadInst *
  lowerIdempotentRMWIntoFencedLoad(AtomicRMWInst *RMWI) const {
    return nullptr;
  }

  /// Returns true if we should normalize
  /// select(N0&N1, X, Y) => select(N0, select(N1, X, Y), Y) and
  /// select(N0|N1, X, Y) => select(N0, select(N1, X, Y, Y)) if it is likely
  /// that it saves us from materializing N0 and N1 in an integer register.
  /// Targets that are able to perform and/or on flags should return false here.
  virtual bool shouldNormalizeToSelectSequence(LLVMContext &Context,
                                               EVT VT) const {
    // If a target has multiple condition registers, then it likely has logical
    // operations on those registers.
    if (hasMultipleConditionRegisters())
      return false;
    // Only do the transform if the value won't be split into multiple
    // registers.
    LegalizeTypeAction Action = getTypeAction(Context, VT);
    return Action != TypeExpandInteger && Action != TypeExpandFloat &&
      Action != TypeSplitVector;
  }

  //===--------------------------------------------------------------------===//
  // TargetLowering Configuration Methods - These methods should be invoked by
  // the derived class constructor to configure this object for the target.
  //
protected:
  /// Specify how the target extends the result of integer and floating point
  /// boolean values from i1 to a wider type.  See getBooleanContents.
  void setBooleanContents(BooleanContent Ty) {
    BooleanContents = Ty;
    BooleanFloatContents = Ty;
  }

  /// Specify how the target extends the result of integer and floating point
  /// boolean values from i1 to a wider type.  See getBooleanContents.
  void setBooleanContents(BooleanContent IntTy, BooleanContent FloatTy) {
    BooleanContents = IntTy;
    BooleanFloatContents = FloatTy;
  }

  /// Specify how the target extends the result of a vector boolean value from a
  /// vector of i1 to a wider type.  See getBooleanContents.
  void setBooleanVectorContents(BooleanContent Ty) {
    BooleanVectorContents = Ty;
  }

  /// Specify the target scheduling preference.
  void setSchedulingPreference(Sched::Preference Pref) {
    SchedPreferenceInfo = Pref;
  }

  /// Indicate whether this target prefers to use _setjmp to implement
  /// llvm.setjmp or the version without _.  Defaults to false.
  void setUseUnderscoreSetJmp(bool Val) {
    UseUnderscoreSetJmp = Val;
  }

  /// Indicate whether this target prefers to use _longjmp to implement
  /// llvm.longjmp or the version without _.  Defaults to false.
  void setUseUnderscoreLongJmp(bool Val) {
    UseUnderscoreLongJmp = Val;
  }

  /// Indicate the number of blocks to generate jump tables rather than if
  /// sequence.
  void setMinimumJumpTableEntries(int Val) {
    MinimumJumpTableEntries = Val;
  }

  /// If set to a physical register, this specifies the register that
  /// llvm.savestack/llvm.restorestack should save and restore.
  void setStackPointerRegisterToSaveRestore(unsigned R) {
    StackPointerRegisterToSaveRestore = R;
  }

  /// If set to a physical register, this sets the register that receives the
  /// exception address on entry to a landing pad.
  void setExceptionPointerRegister(unsigned R) {
    ExceptionPointerRegister = R;
  }

  /// If set to a physical register, this sets the register that receives the
  /// exception typeid on entry to a landing pad.
  void setExceptionSelectorRegister(unsigned R) {
    ExceptionSelectorRegister = R;
  }

  /// Tells the code generator not to expand operations into sequences that use
  /// the select operations if possible.
  void setSelectIsExpensive(bool isExpensive = true) {
    SelectIsExpensive = isExpensive;
  }

  /// Tells the code generator that the target has multiple (allocatable)
  /// condition registers that can be used to store the results of comparisons
  /// for use by selects and conditional branches. With multiple condition
  /// registers, the code generator will not aggressively sink comparisons into
  /// the blocks of their users.
  void setHasMultipleConditionRegisters(bool hasManyRegs = true) {
    HasMultipleConditionRegisters = hasManyRegs;
  }

  /// Tells the code generator that the target has BitExtract instructions.
  /// The code generator will aggressively sink "shift"s into the blocks of
  /// their users if the users will generate "and" instructions which can be
  /// combined with "shift" to BitExtract instructions.
  void setHasExtractBitsInsn(bool hasExtractInsn = true) {
    HasExtractBitsInsn = hasExtractInsn;
  }

  /// Tells the code generator not to expand logic operations on comparison
  /// predicates into separate sequences that increase the amount of flow
  /// control.
  void setJumpIsExpensive(bool isExpensive = true);

  /// Tells the code generator that integer divide is expensive, and if
  /// possible, should be replaced by an alternate sequence of instructions not
  /// containing an integer divide.
  void setIntDivIsCheap(bool isCheap = true) { IntDivIsCheap = isCheap; }

  /// Tells the code generator that fsqrt is cheap, and should not be replaced
  /// with an alternative sequence of instructions.
  void setFsqrtIsCheap(bool isCheap = true) { FsqrtIsCheap = isCheap; }

  /// Tells the code generator that this target supports floating point
  /// exceptions and cares about preserving floating point exception behavior.
  void setHasFloatingPointExceptions(bool FPExceptions = true) {
    HasFloatingPointExceptions = FPExceptions;
  }

  /// Tells the code generator which bitwidths to bypass.
  void addBypassSlowDiv(unsigned int SlowBitWidth, unsigned int FastBitWidth) {
    BypassSlowDivWidths[SlowBitWidth] = FastBitWidth;
  }

  /// Tells the code generator that it shouldn't generate sra/srl/add/sra for a
  /// signed divide by power of two; let the target handle it.
  void setPow2SDivIsCheap(bool isCheap = true) { Pow2SDivIsCheap = isCheap; }

  /// Add the specified register class as an available regclass for the
  /// specified value type. This indicates the selector can handle values of
  /// that class natively.
  void addRegisterClass(MVT VT, const TargetRegisterClass *RC) {
    assert((unsigned)VT.SimpleTy < array_lengthof(RegClassForVT));
    AvailableRegClasses.push_back(std::make_pair(VT, RC));
    RegClassForVT[VT.SimpleTy] = RC;
  }

  /// Remove all register classes.
  void clearRegisterClasses() {
    memset(RegClassForVT, 0,MVT::LAST_VALUETYPE * sizeof(TargetRegisterClass*));

    AvailableRegClasses.clear();
  }

  /// \brief Remove all operation actions.
  void clearOperationActions() {
  }

  /// Return the largest legal super-reg register class of the register class
  /// for the specified type and its associated "cost".
  virtual std::pair<const TargetRegisterClass *, uint8_t>
  findRepresentativeClass(const TargetRegisterInfo *TRI, MVT VT) const;

  /// Once all of the register classes are added, this allows us to compute
  /// derived properties we expose.
  void computeRegisterProperties(const TargetRegisterInfo *TRI);

  /// Indicate that the specified operation does not work with the specified
  /// type and indicate what to do about it.
  void setOperationAction(unsigned Op, MVT VT,
                          LegalizeAction Action) {
    assert(Op < array_lengthof(OpActions[0]) && "Table isn't big enough!");
    OpActions[(unsigned)VT.SimpleTy][Op] = (uint8_t)Action;
  }

  /// Indicate that the specified load with extension does not work with the
  /// specified type and indicate what to do about it.
  void setLoadExtAction(unsigned ExtType, MVT ValVT, MVT MemVT,
                        LegalizeAction Action) {
    assert(ExtType < ISD::LAST_LOADEXT_TYPE && ValVT.isValid() &&
           MemVT.isValid() && "Table isn't big enough!");
    LoadExtActions[ValVT.SimpleTy][MemVT.SimpleTy][ExtType] = (uint8_t)Action;
  }

  /// Indicate that the specified truncating store does not work with the
  /// specified type and indicate what to do about it.
  void setTruncStoreAction(MVT ValVT, MVT MemVT,
                           LegalizeAction Action) {
    assert(ValVT.isValid() && MemVT.isValid() && "Table isn't big enough!");
    TruncStoreActions[ValVT.SimpleTy][MemVT.SimpleTy] = (uint8_t)Action;
  }

  /// Indicate that the specified indexed load does or does not work with the
  /// specified type and indicate what to do abort it.
  ///
  /// NOTE: All indexed mode loads are initialized to Expand in
  /// TargetLowering.cpp
  void setIndexedLoadAction(unsigned IdxMode, MVT VT,
                            LegalizeAction Action) {
    assert(VT.isValid() && IdxMode < ISD::LAST_INDEXED_MODE &&
           (unsigned)Action < 0xf && "Table isn't big enough!");
    // Load action are kept in the upper half.
    IndexedModeActions[(unsigned)VT.SimpleTy][IdxMode] &= ~0xf0;
    IndexedModeActions[(unsigned)VT.SimpleTy][IdxMode] |= ((uint8_t)Action) <<4;
  }

  /// Indicate that the specified indexed store does or does not work with the
  /// specified type and indicate what to do about it.
  ///
  /// NOTE: All indexed mode stores are initialized to Expand in
  /// TargetLowering.cpp
  void setIndexedStoreAction(unsigned IdxMode, MVT VT,
                             LegalizeAction Action) {
    assert(VT.isValid() && IdxMode < ISD::LAST_INDEXED_MODE &&
           (unsigned)Action < 0xf && "Table isn't big enough!");
    // Store action are kept in the lower half.
    IndexedModeActions[(unsigned)VT.SimpleTy][IdxMode] &= ~0x0f;
    IndexedModeActions[(unsigned)VT.SimpleTy][IdxMode] |= ((uint8_t)Action);
  }

  /// Indicate that the specified condition code is or isn't supported on the
  /// target and indicate what to do about it.
  void setCondCodeAction(ISD::CondCode CC, MVT VT,
                         LegalizeAction Action) {
    assert(VT.isValid() && (unsigned)CC < array_lengthof(CondCodeActions) &&
           "Table isn't big enough!");
    /// The lower 5 bits of the SimpleTy index into Nth 2bit set from the 32-bit
    /// value and the upper 27 bits index into the second dimension of the array
    /// to select what 32-bit value to use.
    uint32_t Shift = 2 * (VT.SimpleTy & 0xF);
    CondCodeActions[CC][VT.SimpleTy >> 4] &= ~((uint32_t)0x3 << Shift);
    CondCodeActions[CC][VT.SimpleTy >> 4] |= (uint32_t)Action << Shift;
  }

  /// If Opc/OrigVT is specified as being promoted, the promotion code defaults
  /// to trying a larger integer/fp until it can find one that works. If that
  /// default is insufficient, this method can be used by the target to override
  /// the default.
  void AddPromotedToType(unsigned Opc, MVT OrigVT, MVT DestVT) {
    PromoteToType[std::make_pair(Opc, OrigVT.SimpleTy)] = DestVT.SimpleTy;
  }

  /// Targets should invoke this method for each target independent node that
  /// they want to provide a custom DAG combiner for by implementing the
  /// PerformDAGCombine virtual method.
  void setTargetDAGCombine(ISD::NodeType NT) {
    assert(unsigned(NT >> 3) < array_lengthof(TargetDAGCombineArray));
    TargetDAGCombineArray[NT >> 3] |= 1 << (NT&7);
  }

  /// Set the target's required jmp_buf buffer size (in bytes); default is 200
  void setJumpBufSize(unsigned Size) {
    JumpBufSize = Size;
  }

  /// Set the target's required jmp_buf buffer alignment (in bytes); default is
  /// 0
  void setJumpBufAlignment(unsigned Align) {
    JumpBufAlignment = Align;
  }

  /// Set the target's minimum function alignment (in log2(bytes))
  void setMinFunctionAlignment(unsigned Align) {
    MinFunctionAlignment = Align;
  }

  /// Set the target's preferred function alignment.  This should be set if
  /// there is a performance benefit to higher-than-minimum alignment (in
  /// log2(bytes))
  void setPrefFunctionAlignment(unsigned Align) {
    PrefFunctionAlignment = Align;
  }

  /// Set the target's preferred loop alignment. Default alignment is zero, it
  /// means the target does not care about loop alignment.  The alignment is
  /// specified in log2(bytes). The target may also override
  /// getPrefLoopAlignment to provide per-loop values.
  void setPrefLoopAlignment(unsigned Align) {
    PrefLoopAlignment = Align;
  }

  /// Set the minimum stack alignment of an argument (in log2(bytes)).
  void setMinStackArgumentAlignment(unsigned Align) {
    MinStackArgumentAlignment = Align;
  }

  /// Set if the DAG builder should automatically insert fences and reduce the
  /// order of atomic memory operations to Monotonic.
  void setInsertFencesForAtomic(bool fence) {
    InsertFencesForAtomic = fence;
  }

public:
  //===--------------------------------------------------------------------===//
  // Addressing mode description hooks (used by LSR etc).
  //

  /// CodeGenPrepare sinks address calculations into the same BB as Load/Store
  /// instructions reading the address. This allows as much computation as
  /// possible to be done in the address mode for that operand. This hook lets
  /// targets also pass back when this should be done on intrinsics which
  /// load/store.
  virtual bool GetAddrModeArguments(IntrinsicInst * /*I*/,
                                    SmallVectorImpl<Value*> &/*Ops*/,
                                    Type *&/*AccessTy*/,
                                    unsigned AddrSpace = 0) const {
    return false;
  }

  /// This represents an addressing mode of:
  ///    BaseGV + BaseOffs + BaseReg + Scale*ScaleReg
  /// If BaseGV is null,  there is no BaseGV.
  /// If BaseOffs is zero, there is no base offset.
  /// If HasBaseReg is false, there is no base register.
  /// If Scale is zero, there is no ScaleReg.  Scale of 1 indicates a reg with
  /// no scale.
  struct AddrMode {
    GlobalValue *BaseGV;
    int64_t      BaseOffs;
    bool         HasBaseReg;
    int64_t      Scale;
    AddrMode() : BaseGV(nullptr), BaseOffs(0), HasBaseReg(false), Scale(0) {}
  };

  /// Return true if the addressing mode represented by AM is legal for this
  /// target, for a load/store of the specified type.
  ///
  /// The type may be VoidTy, in which case only return true if the addressing
  /// mode is legal for a load/store of any legal type.  TODO: Handle
  /// pre/postinc as well.
  ///
  /// If the address space cannot be determined, it will be -1.
  ///
  /// TODO: Remove default argument
  virtual bool isLegalAddressingMode(const DataLayout &DL, const AddrMode &AM,
                                     Type *Ty, unsigned AddrSpace) const;

  /// \brief Return the cost of the scaling factor used in the addressing mode
  /// represented by AM for this target, for a load/store of the specified type.
  ///
  /// If the AM is supported, the return value must be >= 0.
  /// If the AM is not supported, it returns a negative value.
  /// TODO: Handle pre/postinc as well.
  /// TODO: Remove default argument
  virtual int getScalingFactorCost(const DataLayout &DL, const AddrMode &AM,
                                   Type *Ty, unsigned AS = 0) const {
    // Default: assume that any scaling factor used in a legal AM is free.
    if (isLegalAddressingMode(DL, AM, Ty, AS))
      return 0;
    return -1;
  }

  /// Return true if the specified immediate is legal icmp immediate, that is
  /// the target has icmp instructions which can compare a register against the
  /// immediate without having to materialize the immediate into a register.
  virtual bool isLegalICmpImmediate(int64_t) const {
    return true;
  }

  /// Return true if the specified immediate is legal add immediate, that is the
  /// target has add instructions which can add a register with the immediate
  /// without having to materialize the immediate into a register.
  virtual bool isLegalAddImmediate(int64_t) const {
    return true;
  }

  /// Return true if it's significantly cheaper to shift a vector by a uniform
  /// scalar than by an amount which will vary across each lane. On x86, for
  /// example, there is a "psllw" instruction for the former case, but no simple
  /// instruction for a general "a << b" operation on vectors.
  virtual bool isVectorShiftByScalarCheap(Type *Ty) const {
    return false;
  }

  /// Return true if it's free to truncate a value of type Ty1 to type
  /// Ty2. e.g. On x86 it's free to truncate a i32 value in register EAX to i16
  /// by referencing its sub-register AX.
  virtual bool isTruncateFree(Type * /*Ty1*/, Type * /*Ty2*/) const {
    return false;
  }

  /// Return true if a truncation from Ty1 to Ty2 is permitted when deciding
  /// whether a call is in tail position. Typically this means that both results
  /// would be assigned to the same register or stack slot, but it could mean
  /// the target performs adequate checks of its own before proceeding with the
  /// tail call.
  virtual bool allowTruncateForTailCall(Type * /*Ty1*/, Type * /*Ty2*/) const {
    return false;
  }

  virtual bool isTruncateFree(EVT /*VT1*/, EVT /*VT2*/) const {
    return false;
  }

  virtual bool isProfitableToHoist(Instruction *I) const { return true; }

  /// Return true if the extension represented by \p I is free.
  /// Unlikely the is[Z|FP]ExtFree family which is based on types,
  /// this method can use the context provided by \p I to decide
  /// whether or not \p I is free.
  /// This method extends the behavior of the is[Z|FP]ExtFree family.
  /// In other words, if is[Z|FP]Free returns true, then this method
  /// returns true as well. The converse is not true.
  /// The target can perform the adequate checks by overriding isExtFreeImpl.
  /// \pre \p I must be a sign, zero, or fp extension.
  bool isExtFree(const Instruction *I) const {
    switch (I->getOpcode()) {
    case Instruction::FPExt:
      if (isFPExtFree(EVT::getEVT(I->getType())))
        return true;
      break;
    case Instruction::ZExt:
      if (isZExtFree(I->getOperand(0)->getType(), I->getType()))
        return true;
      break;
    case Instruction::SExt:
      break;
    default:
      llvm_unreachable("Instruction is not an extension");
    }
    return isExtFreeImpl(I);
  }

  /// Return true if any actual instruction that defines a value of type Ty1
  /// implicitly zero-extends the value to Ty2 in the result register.
  ///
  /// This does not necessarily include registers defined in unknown ways, such
  /// as incoming arguments, or copies from unknown virtual registers. Also, if
  /// isTruncateFree(Ty2, Ty1) is true, this does not necessarily apply to
  /// truncate instructions. e.g. on x86-64, all instructions that define 32-bit
  /// values implicit zero-extend the result out to 64 bits.
  virtual bool isZExtFree(Type * /*Ty1*/, Type * /*Ty2*/) const {
    return false;
  }

  virtual bool isZExtFree(EVT /*VT1*/, EVT /*VT2*/) const {
    return false;
  }

  /// Return true if the target supplies and combines to a paired load
  /// two loaded values of type LoadedType next to each other in memory.
  /// RequiredAlignment gives the minimal alignment constraints that must be met
  /// to be able to select this paired load.
  ///
  /// This information is *not* used to generate actual paired loads, but it is
  /// used to generate a sequence of loads that is easier to combine into a
  /// paired load.
  /// For instance, something like this:
  /// a = load i64* addr
  /// b = trunc i64 a to i32
  /// c = lshr i64 a, 32
  /// d = trunc i64 c to i32
  /// will be optimized into:
  /// b = load i32* addr1
  /// d = load i32* addr2
  /// Where addr1 = addr2 +/- sizeof(i32).
  ///
  /// In other words, unless the target performs a post-isel load combining,
  /// this information should not be provided because it will generate more
  /// loads.
  virtual bool hasPairedLoad(Type * /*LoadedType*/,
                             unsigned & /*RequiredAligment*/) const {
    return false;
  }

  virtual bool hasPairedLoad(EVT /*LoadedType*/,
                             unsigned & /*RequiredAligment*/) const {
    return false;
  }

  /// \brief Get the maximum supported factor for interleaved memory accesses.
  /// Default to be the minimum interleave factor: 2.
  virtual unsigned getMaxSupportedInterleaveFactor() const { return 2; }

  /// \brief Lower an interleaved load to target specific intrinsics. Return
  /// true on success.
  ///
  /// \p LI is the vector load instruction.
  /// \p Shuffles is the shufflevector list to DE-interleave the loaded vector.
  /// \p Indices is the corresponding indices for each shufflevector.
  /// \p Factor is the interleave factor.
  virtual bool lowerInterleavedLoad(LoadInst *LI,
                                    ArrayRef<ShuffleVectorInst *> Shuffles,
                                    ArrayRef<unsigned> Indices,
                                    unsigned Factor) const {
    return false;
  }

  /// \brief Lower an interleaved store to target specific intrinsics. Return
  /// true on success.
  ///
  /// \p SI is the vector store instruction.
  /// \p SVI is the shufflevector to RE-interleave the stored vector.
  /// \p Factor is the interleave factor.
  virtual bool lowerInterleavedStore(StoreInst *SI, ShuffleVectorInst *SVI,
                                     unsigned Factor) const {
    return false;
  }

  /// Return true if zero-extending the specific node Val to type VT2 is free
  /// (either because it's implicitly zero-extended such as ARM ldrb / ldrh or
  /// because it's folded such as X86 zero-extending loads).
  virtual bool isZExtFree(SDValue Val, EVT VT2) const {
    return isZExtFree(Val.getValueType(), VT2);
  }

  /// Return true if an fpext operation is free (for instance, because
  /// single-precision floating-point numbers are implicitly extended to
  /// double-precision).
  virtual bool isFPExtFree(EVT VT) const {
    assert(VT.isFloatingPoint());
    return false;
  }

  /// Return true if folding a vector load into ExtVal (a sign, zero, or any
  /// extend node) is profitable.
  virtual bool isVectorLoadExtDesirable(SDValue ExtVal) const { return false; }

  /// Return true if an fneg operation is free to the point where it is never
  /// worthwhile to replace it with a bitwise operation.
  virtual bool isFNegFree(EVT VT) const {
    assert(VT.isFloatingPoint());
    return false;
  }

  /// Return true if an fabs operation is free to the point where it is never
  /// worthwhile to replace it with a bitwise operation.
  virtual bool isFAbsFree(EVT VT) const {
    assert(VT.isFloatingPoint());
    return false;
  }

  /// Return true if an FMA operation is faster than a pair of fmul and fadd
  /// instructions. fmuladd intrinsics will be expanded to FMAs when this method
  /// returns true, otherwise fmuladd is expanded to fmul + fadd.
  ///
  /// NOTE: This may be called before legalization on types for which FMAs are
  /// not legal, but should return true if those types will eventually legalize
  /// to types that support FMAs. After legalization, it will only be called on
  /// types that support FMAs (via Legal or Custom actions)
  virtual bool isFMAFasterThanFMulAndFAdd(EVT) const {
    return false;
  }

  /// Return true if it's profitable to narrow operations of type VT1 to
  /// VT2. e.g. on x86, it's profitable to narrow from i32 to i8 but not from
  /// i32 to i16.
  virtual bool isNarrowingProfitable(EVT /*VT1*/, EVT /*VT2*/) const {
    return false;
  }

  /// \brief Return true if it is beneficial to convert a load of a constant to
  /// just the constant itself.
  /// On some targets it might be more efficient to use a combination of
  /// arithmetic instructions to materialize the constant instead of loading it
  /// from a constant pool.
  virtual bool shouldConvertConstantLoadToIntImm(const APInt &Imm,
                                                 Type *Ty) const {
    return false;
  }

  /// Return true if EXTRACT_SUBVECTOR is cheap for this result type
  /// with this index. This is needed because EXTRACT_SUBVECTOR usually
  /// has custom lowering that depends on the index of the first element,
  /// and only the target knows which lowering is cheap.
  virtual bool isExtractSubvectorCheap(EVT ResVT, unsigned Index) const {
    return false;
  }

  //===--------------------------------------------------------------------===//
  // Runtime Library hooks
  //

  /// Rename the default libcall routine name for the specified libcall.
  void setLibcallName(RTLIB::Libcall Call, const char *Name) {
    LibcallRoutineNames[Call] = Name;
  }

  /// Get the libcall routine name for the specified libcall.
  const char *getLibcallName(RTLIB::Libcall Call) const {
    return LibcallRoutineNames[Call];
  }

  /// Override the default CondCode to be used to test the result of the
  /// comparison libcall against zero.
  void setCmpLibcallCC(RTLIB::Libcall Call, ISD::CondCode CC) {
    CmpLibcallCCs[Call] = CC;
  }

  /// Get the CondCode that's to be used to test the result of the comparison
  /// libcall against zero.
  ISD::CondCode getCmpLibcallCC(RTLIB::Libcall Call) const {
    return CmpLibcallCCs[Call];
  }

  /// Set the CallingConv that should be used for the specified libcall.
  void setLibcallCallingConv(RTLIB::Libcall Call, CallingConv::ID CC) {
    LibcallCallingConvs[Call] = CC;
  }

  /// Get the CallingConv that should be used for the specified libcall.
  CallingConv::ID getLibcallCallingConv(RTLIB::Libcall Call) const {
    return LibcallCallingConvs[Call];
  }

private:
  const TargetMachine &TM;

  /// Tells the code generator not to expand operations into sequences that use
  /// the select operations if possible.
  bool SelectIsExpensive;

  /// Tells the code generator that the target has multiple (allocatable)
  /// condition registers that can be used to store the results of comparisons
  /// for use by selects and conditional branches. With multiple condition
  /// registers, the code generator will not aggressively sink comparisons into
  /// the blocks of their users.
  bool HasMultipleConditionRegisters;

  /// Tells the code generator that the target has BitExtract instructions.
  /// The code generator will aggressively sink "shift"s into the blocks of
  /// their users if the users will generate "and" instructions which can be
  /// combined with "shift" to BitExtract instructions.
  bool HasExtractBitsInsn;

  /// Tells the code generator not to expand integer divides by constants into a
  /// sequence of muls, adds, and shifts.  This is a hack until a real cost
  /// model is in place.  If we ever optimize for size, this will be set to true
  /// unconditionally.
  bool IntDivIsCheap;

  // Don't expand fsqrt with an approximation based on the inverse sqrt.
  bool FsqrtIsCheap;

  /// Tells the code generator to bypass slow divide or remainder
  /// instructions. For example, BypassSlowDivWidths[32,8] tells the code
  /// generator to bypass 32-bit integer div/rem with an 8-bit unsigned integer
  /// div/rem when the operands are positive and less than 256.
  DenseMap <unsigned int, unsigned int> BypassSlowDivWidths;

  /// Tells the code generator that it shouldn't generate sra/srl/add/sra for a
  /// signed divide by power of two; let the target handle it.
  bool Pow2SDivIsCheap;

  /// Tells the code generator that it shouldn't generate extra flow control
  /// instructions and should attempt to combine flow control instructions via
  /// predication.
  bool JumpIsExpensive;

  /// Whether the target supports or cares about preserving floating point
  /// exception behavior.
  bool HasFloatingPointExceptions;

  /// This target prefers to use _setjmp to implement llvm.setjmp.
  ///
  /// Defaults to false.
  bool UseUnderscoreSetJmp;

  /// This target prefers to use _longjmp to implement llvm.longjmp.
  ///
  /// Defaults to false.
  bool UseUnderscoreLongJmp;

  /// Number of blocks threshold to use jump tables.
  int MinimumJumpTableEntries;

  /// Information about the contents of the high-bits in boolean values held in
  /// a type wider than i1. See getBooleanContents.
  BooleanContent BooleanContents;

  /// Information about the contents of the high-bits in boolean values held in
  /// a type wider than i1. See getBooleanContents.
  BooleanContent BooleanFloatContents;

  /// Information about the contents of the high-bits in boolean vector values
  /// when the element type is wider than i1. See getBooleanContents.
  BooleanContent BooleanVectorContents;

  /// The target scheduling preference: shortest possible total cycles or lowest
  /// register usage.
  Sched::Preference SchedPreferenceInfo;

  /// The size, in bytes, of the target's jmp_buf buffers
  unsigned JumpBufSize;

  /// The alignment, in bytes, of the target's jmp_buf buffers
  unsigned JumpBufAlignment;

  /// The minimum alignment that any argument on the stack needs to have.
  unsigned MinStackArgumentAlignment;

  /// The minimum function alignment (used when optimizing for size, and to
  /// prevent explicitly provided alignment from leading to incorrect code).
  unsigned MinFunctionAlignment;

  /// The preferred function alignment (used when alignment unspecified and
  /// optimizing for speed).
  unsigned PrefFunctionAlignment;

  /// The preferred loop alignment.
  unsigned PrefLoopAlignment;

  /// Whether the DAG builder should automatically insert fences and reduce
  /// ordering for atomics.  (This will be set for for most architectures with
  /// weak memory ordering.)
  bool InsertFencesForAtomic;

  /// If set to a physical register, this specifies the register that
  /// llvm.savestack/llvm.restorestack should save and restore.
  unsigned StackPointerRegisterToSaveRestore;

  /// If set to a physical register, this specifies the register that receives
  /// the exception address on entry to a landing pad.
  unsigned ExceptionPointerRegister;

  /// If set to a physical register, this specifies the register that receives
  /// the exception typeid on entry to a landing pad.
  unsigned ExceptionSelectorRegister;

  /// This indicates the default register class to use for each ValueType the
  /// target supports natively.
  const TargetRegisterClass *RegClassForVT[MVT::LAST_VALUETYPE];
  unsigned char NumRegistersForVT[MVT::LAST_VALUETYPE];
  MVT RegisterTypeForVT[MVT::LAST_VALUETYPE];

  /// This indicates the "representative" register class to use for each
  /// ValueType the target supports natively. This information is used by the
  /// scheduler to track register pressure. By default, the representative
  /// register class is the largest legal super-reg register class of the
  /// register class of the specified type. e.g. On x86, i8, i16, and i32's
  /// representative class would be GR32.
  const TargetRegisterClass *RepRegClassForVT[MVT::LAST_VALUETYPE];

  /// This indicates the "cost" of the "representative" register class for each
  /// ValueType. The cost is used by the scheduler to approximate register
  /// pressure.
  uint8_t RepRegClassCostForVT[MVT::LAST_VALUETYPE];

  /// For any value types we are promoting or expanding, this contains the value
  /// type that we are changing to.  For Expanded types, this contains one step
  /// of the expand (e.g. i64 -> i32), even if there are multiple steps required
  /// (e.g. i64 -> i16).  For types natively supported by the system, this holds
  /// the same type (e.g. i32 -> i32).
  MVT TransformToType[MVT::LAST_VALUETYPE];

  /// For each operation and each value type, keep a LegalizeAction that
  /// indicates how instruction selection should deal with the operation.  Most
  /// operations are Legal (aka, supported natively by the target), but
  /// operations that are not should be described.  Note that operations on
  /// non-legal value types are not described here.
  uint8_t OpActions[MVT::LAST_VALUETYPE][ISD::BUILTIN_OP_END];

  /// For each load extension type and each value type, keep a LegalizeAction
  /// that indicates how instruction selection should deal with a load of a
  /// specific value type and extension type.
  uint8_t LoadExtActions[MVT::LAST_VALUETYPE][MVT::LAST_VALUETYPE]
                        [ISD::LAST_LOADEXT_TYPE];

  /// For each value type pair keep a LegalizeAction that indicates whether a
  /// truncating store of a specific value type and truncating type is legal.
  uint8_t TruncStoreActions[MVT::LAST_VALUETYPE][MVT::LAST_VALUETYPE];

  /// For each indexed mode and each value type, keep a pair of LegalizeAction
  /// that indicates how instruction selection should deal with the load /
  /// store.
  ///
  /// The first dimension is the value_type for the reference. The second
  /// dimension represents the various modes for load store.
  uint8_t IndexedModeActions[MVT::LAST_VALUETYPE][ISD::LAST_INDEXED_MODE];

  /// For each condition code (ISD::CondCode) keep a LegalizeAction that
  /// indicates how instruction selection should deal with the condition code.
  ///
  /// Because each CC action takes up 2 bits, we need to have the array size be
  /// large enough to fit all of the value types. This can be done by rounding
  /// up the MVT::LAST_VALUETYPE value to the next multiple of 16.
  uint32_t CondCodeActions[ISD::SETCC_INVALID][(MVT::LAST_VALUETYPE + 15) / 16];

  ValueTypeActionImpl ValueTypeActions;

private:
  LegalizeKind getTypeConversion(LLVMContext &Context, EVT VT) const;

private:
  std::vector<std::pair<MVT, const TargetRegisterClass*> > AvailableRegClasses;

  /// Targets can specify ISD nodes that they would like PerformDAGCombine
  /// callbacks for by calling setTargetDAGCombine(), which sets a bit in this
  /// array.
  unsigned char
  TargetDAGCombineArray[(ISD::BUILTIN_OP_END+CHAR_BIT-1)/CHAR_BIT];

  /// For operations that must be promoted to a specific type, this holds the
  /// destination type.  This map should be sparse, so don't hold it as an
  /// array.
  ///
  /// Targets add entries to this map with AddPromotedToType(..), clients access
  /// this with getTypeToPromoteTo(..).
  std::map<std::pair<unsigned, MVT::SimpleValueType>, MVT::SimpleValueType>
    PromoteToType;

  /// Stores the name each libcall.
  const char *LibcallRoutineNames[RTLIB::UNKNOWN_LIBCALL];

  /// The ISD::CondCode that should be used to test the result of each of the
  /// comparison libcall against zero.
  ISD::CondCode CmpLibcallCCs[RTLIB::UNKNOWN_LIBCALL];

  /// Stores the CallingConv that should be used for each libcall.
  CallingConv::ID LibcallCallingConvs[RTLIB::UNKNOWN_LIBCALL];

protected:
  /// Return true if the extension represented by \p I is free.
  /// \pre \p I is a sign, zero, or fp extension and
  ///      is[Z|FP]ExtFree of the related types is not true.
  virtual bool isExtFreeImpl(const Instruction *I) const { return false; }

  /// \brief Specify maximum number of store instructions per memset call.
  ///
  /// When lowering \@llvm.memset this field specifies the maximum number of
  /// store operations that may be substituted for the call to memset. Targets
  /// must set this value based on the cost threshold for that target. Targets
  /// should assume that the memset will be done using as many of the largest
  /// store operations first, followed by smaller ones, if necessary, per
  /// alignment restrictions. For example, storing 9 bytes on a 32-bit machine
  /// with 16-bit alignment would result in four 2-byte stores and one 1-byte
  /// store.  This only applies to setting a constant array of a constant size.
  unsigned MaxStoresPerMemset;

  /// Maximum number of stores operations that may be substituted for the call
  /// to memset, used for functions with OptSize attribute.
  unsigned MaxStoresPerMemsetOptSize;

  /// \brief Specify maximum bytes of store instructions per memcpy call.
  ///
  /// When lowering \@llvm.memcpy this field specifies the maximum number of
  /// store operations that may be substituted for a call to memcpy. Targets
  /// must set this value based on the cost threshold for that target. Targets
  /// should assume that the memcpy will be done using as many of the largest
  /// store operations first, followed by smaller ones, if necessary, per
  /// alignment restrictions. For example, storing 7 bytes on a 32-bit machine
  /// with 32-bit alignment would result in one 4-byte store, a one 2-byte store
  /// and one 1-byte store. This only applies to copying a constant array of
  /// constant size.
  unsigned MaxStoresPerMemcpy;

  /// Maximum number of store operations that may be substituted for a call to
  /// memcpy, used for functions with OptSize attribute.
  unsigned MaxStoresPerMemcpyOptSize;

  /// \brief Specify maximum bytes of store instructions per memmove call.
  ///
  /// When lowering \@llvm.memmove this field specifies the maximum number of
  /// store instructions that may be substituted for a call to memmove. Targets
  /// must set this value based on the cost threshold for that target. Targets
  /// should assume that the memmove will be done using as many of the largest
  /// store operations first, followed by smaller ones, if necessary, per
  /// alignment restrictions. For example, moving 9 bytes on a 32-bit machine
  /// with 8-bit alignment would result in nine 1-byte stores.  This only
  /// applies to copying a constant array of constant size.
  unsigned MaxStoresPerMemmove;

  /// Maximum number of store instructions that may be substituted for a call to
  /// memmove, used for functions with OpSize attribute.
  unsigned MaxStoresPerMemmoveOptSize;

  /// Tells the code generator that select is more expensive than a branch if
  /// the branch is usually predicted right.
  bool PredictableSelectIsExpensive;

  /// MaskAndBranchFoldingIsLegal - Indicates if the target supports folding
  /// a mask of a single bit, a compare, and a branch into a single instruction.
  bool MaskAndBranchFoldingIsLegal;

  /// \see enableExtLdPromotion.
  bool EnableExtLdPromotion;

protected:
  /// Return true if the value types that can be represented by the specified
  /// register class are all legal.
  bool isLegalRC(const TargetRegisterClass *RC) const;

  /// Replace/modify any TargetFrameIndex operands with a targte-dependent
  /// sequence of memory operands that is recognized by PrologEpilogInserter.
  MachineBasicBlock *emitPatchPoint(MachineInstr *MI,
                                    MachineBasicBlock *MBB) const;
};

/// This class defines information used to lower LLVM code to legal SelectionDAG
/// operators that the target instruction selector can accept natively.
///
/// This class also defines callbacks that targets must implement to lower
/// target-specific constructs to SelectionDAG operators.
class TargetLowering : public TargetLoweringBase {
  TargetLowering(const TargetLowering&) = delete;
  void operator=(const TargetLowering&) = delete;

public:
  /// NOTE: The TargetMachine owns TLOF.
  explicit TargetLowering(const TargetMachine &TM);

  /// Returns true by value, base pointer and offset pointer and addressing mode
  /// by reference if the node's address can be legally represented as
  /// pre-indexed load / store address.
  virtual bool getPreIndexedAddressParts(SDNode * /*N*/, SDValue &/*Base*/,
                                         SDValue &/*Offset*/,
                                         ISD::MemIndexedMode &/*AM*/,
                                         SelectionDAG &/*DAG*/) const {
    return false;
  }

  /// Returns true by value, base pointer and offset pointer and addressing mode
  /// by reference if this node can be combined with a load / store to form a
  /// post-indexed load / store.
  virtual bool getPostIndexedAddressParts(SDNode * /*N*/, SDNode * /*Op*/,
                                          SDValue &/*Base*/,
                                          SDValue &/*Offset*/,
                                          ISD::MemIndexedMode &/*AM*/,
                                          SelectionDAG &/*DAG*/) const {
    return false;
  }

  /// Return the entry encoding for a jump table in the current function.  The
  /// returned value is a member of the MachineJumpTableInfo::JTEntryKind enum.
  virtual unsigned getJumpTableEncoding() const;

  virtual const MCExpr *
  LowerCustomJumpTableEntry(const MachineJumpTableInfo * /*MJTI*/,
                            const MachineBasicBlock * /*MBB*/, unsigned /*uid*/,
                            MCContext &/*Ctx*/) const {
    llvm_unreachable("Need to implement this hook if target has custom JTIs");
  }

  /// Returns relocation base for the given PIC jumptable.
  virtual SDValue getPICJumpTableRelocBase(SDValue Table,
                                           SelectionDAG &DAG) const;

  /// This returns the relocation base for the given PIC jumptable, the same as
  /// getPICJumpTableRelocBase, but as an MCExpr.
  virtual const MCExpr *
  getPICJumpTableRelocBaseExpr(const MachineFunction *MF,
                               unsigned JTI, MCContext &Ctx) const;

  /// Return true if folding a constant offset with the given GlobalAddress is
  /// legal.  It is frequently not legal in PIC relocation models.
  virtual bool isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const;

  bool isInTailCallPosition(SelectionDAG &DAG, SDNode *Node,
                            SDValue &Chain) const;

  void softenSetCCOperands(SelectionDAG &DAG, EVT VT,
                           SDValue &NewLHS, SDValue &NewRHS,
                           ISD::CondCode &CCCode, SDLoc DL) const;

  /// Returns a pair of (return value, chain).
  /// It is an error to pass RTLIB::UNKNOWN_LIBCALL as \p LC.
  std::pair<SDValue, SDValue> makeLibCall(SelectionDAG &DAG, RTLIB::Libcall LC,
                                          EVT RetVT, const SDValue *Ops,
                                          unsigned NumOps, bool isSigned,
                                          SDLoc dl, bool doesNotReturn = false,
                                          bool isReturnValueUsed = true) const;

  //===--------------------------------------------------------------------===//
  // TargetLowering Optimization Methods
  //

  /// A convenience struct that encapsulates a DAG, and two SDValues for
  /// returning information from TargetLowering to its clients that want to
  /// combine.
  struct TargetLoweringOpt {
    SelectionDAG &DAG;
    bool LegalTys;
    bool LegalOps;
    SDValue Old;
    SDValue New;

    explicit TargetLoweringOpt(SelectionDAG &InDAG,
                               bool LT, bool LO) :
      DAG(InDAG), LegalTys(LT), LegalOps(LO) {}

    bool LegalTypes() const { return LegalTys; }
    bool LegalOperations() const { return LegalOps; }

    bool CombineTo(SDValue O, SDValue N) {
      Old = O;
      New = N;
      return true;
    }

    /// Check to see if the specified operand of the specified instruction is a
    /// constant integer.  If so, check to see if there are any bits set in the
    /// constant that are not demanded.  If so, shrink the constant and return
    /// true.
    bool ShrinkDemandedConstant(SDValue Op, const APInt &Demanded);

    /// Convert x+y to (VT)((SmallVT)x+(SmallVT)y) if the casts are free.  This
    /// uses isZExtFree and ZERO_EXTEND for the widening cast, but it could be
    /// generalized for targets with other types of implicit widening casts.
    bool ShrinkDemandedOp(SDValue Op, unsigned BitWidth, const APInt &Demanded,
                          SDLoc dl);
  };

  /// Look at Op.  At this point, we know that only the DemandedMask bits of the
  /// result of Op are ever used downstream.  If we can use this information to
  /// simplify Op, create a new simplified DAG node and return true, returning
  /// the original and new nodes in Old and New.  Otherwise, analyze the
  /// expression and return a mask of KnownOne and KnownZero bits for the
  /// expression (used to simplify the caller).  The KnownZero/One bits may only
  /// be accurate for those bits in the DemandedMask.
  bool SimplifyDemandedBits(SDValue Op, const APInt &DemandedMask,
                            APInt &KnownZero, APInt &KnownOne,
                            TargetLoweringOpt &TLO, unsigned Depth = 0) const;

  /// Determine which of the bits specified in Mask are known to be either zero
  /// or one and return them in the KnownZero/KnownOne bitsets.
  virtual void computeKnownBitsForTargetNode(const SDValue Op,
                                             APInt &KnownZero,
                                             APInt &KnownOne,
                                             const SelectionDAG &DAG,
                                             unsigned Depth = 0) const;

  /// This method can be implemented by targets that want to expose additional
  /// information about sign bits to the DAG Combiner.
  virtual unsigned ComputeNumSignBitsForTargetNode(SDValue Op,
                                                   const SelectionDAG &DAG,
                                                   unsigned Depth = 0) const;

  struct DAGCombinerInfo {
    void *DC;  // The DAG Combiner object.
    CombineLevel Level;
    bool CalledByLegalizer;
  public:
    SelectionDAG &DAG;

    DAGCombinerInfo(SelectionDAG &dag, CombineLevel level,  bool cl, void *dc)
      : DC(dc), Level(level), CalledByLegalizer(cl), DAG(dag) {}

    bool isBeforeLegalize() const { return Level == BeforeLegalizeTypes; }
    bool isBeforeLegalizeOps() const { return Level < AfterLegalizeVectorOps; }
    bool isAfterLegalizeVectorOps() const {
      return Level == AfterLegalizeDAG;
    }
    CombineLevel getDAGCombineLevel() { return Level; }
    bool isCalledByLegalizer() const { return CalledByLegalizer; }

    void AddToWorklist(SDNode *N);
    void RemoveFromWorklist(SDNode *N);
    SDValue CombineTo(SDNode *N, ArrayRef<SDValue> To, bool AddTo = true);
    SDValue CombineTo(SDNode *N, SDValue Res, bool AddTo = true);
    SDValue CombineTo(SDNode *N, SDValue Res0, SDValue Res1, bool AddTo = true);

    void CommitTargetLoweringOpt(const TargetLoweringOpt &TLO);
  };

  /// Return if the N is a constant or constant vector equal to the true value
  /// from getBooleanContents().
  bool isConstTrueVal(const SDNode *N) const;

  /// Return if the N is a constant or constant vector equal to the false value
  /// from getBooleanContents().
  bool isConstFalseVal(const SDNode *N) const;

  /// Try to simplify a setcc built with the specified operands and cc. If it is
  /// unable to simplify it, return a null SDValue.
  SDValue SimplifySetCC(EVT VT, SDValue N0, SDValue N1,
                          ISD::CondCode Cond, bool foldBooleans,
                          DAGCombinerInfo &DCI, SDLoc dl) const;

  /// Returns true (and the GlobalValue and the offset) if the node is a
  /// GlobalAddress + offset.
  virtual bool
  isGAPlusOffset(SDNode *N, const GlobalValue* &GA, int64_t &Offset) const;

  /// This method will be invoked for all target nodes and for any
  /// target-independent nodes that the target has registered with invoke it
  /// for.
  ///
  /// The semantics are as follows:
  /// Return Value:
  ///   SDValue.Val == 0   - No change was made
  ///   SDValue.Val == N   - N was replaced, is dead, and is already handled.
  ///   otherwise          - N should be replaced by the returned Operand.
  ///
  /// In addition, methods provided by DAGCombinerInfo may be used to perform
  /// more complex transformations.
  ///
  virtual SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const;

  /// Return true if it is profitable to move a following shift through this
  //  node, adjusting any immediate operands as necessary to preserve semantics.
  //  This transformation may not be desirable if it disrupts a particularly
  //  auspicious target-specific tree (e.g. bitfield extraction in AArch64).
  //  By default, it returns true.
  virtual bool isDesirableToCommuteWithShift(const SDNode *N /*Op*/) const {
    return true;
  }

  /// Return true if the target has native support for the specified value type
  /// and it is 'desirable' to use the type for the given node type. e.g. On x86
  /// i16 is legal, but undesirable since i16 instruction encodings are longer
  /// and some i16 instructions are slow.
  virtual bool isTypeDesirableForOp(unsigned /*Opc*/, EVT VT) const {
    // By default, assume all legal types are desirable.
    return isTypeLegal(VT);
  }

  /// Return true if it is profitable for dag combiner to transform a floating
  /// point op of specified opcode to a equivalent op of an integer
  /// type. e.g. f32 load -> i32 load can be profitable on ARM.
  virtual bool isDesirableToTransformToIntegerOp(unsigned /*Opc*/,
                                                 EVT /*VT*/) const {
    return false;
  }

  /// This method query the target whether it is beneficial for dag combiner to
  /// promote the specified node. If true, it should return the desired
  /// promotion type by reference.
  virtual bool IsDesirableToPromoteOp(SDValue /*Op*/, EVT &/*PVT*/) const {
    return false;
  }

  //===--------------------------------------------------------------------===//
  // Lowering methods - These methods must be implemented by targets so that
  // the SelectionDAGBuilder code knows how to lower these.
  //

  /// This hook must be implemented to lower the incoming (formal) arguments,
  /// described by the Ins array, into the specified DAG. The implementation
  /// should fill in the InVals array with legal-type argument values, and
  /// return the resulting token chain value.
  ///
  virtual SDValue
    LowerFormalArguments(SDValue /*Chain*/, CallingConv::ID /*CallConv*/,
                         bool /*isVarArg*/,
                         const SmallVectorImpl<ISD::InputArg> &/*Ins*/,
                         SDLoc /*dl*/, SelectionDAG &/*DAG*/,
                         SmallVectorImpl<SDValue> &/*InVals*/) const {
    llvm_unreachable("Not Implemented");
  }

  struct ArgListEntry {
    SDValue Node;
    Type* Ty;
    bool isSExt     : 1;
    bool isZExt     : 1;
    bool isInReg    : 1;
    bool isSRet     : 1;
    bool isNest     : 1;
    bool isByVal    : 1;
    bool isInAlloca : 1;
    bool isReturned : 1;
    uint16_t Alignment;

    ArgListEntry() : isSExt(false), isZExt(false), isInReg(false),
      isSRet(false), isNest(false), isByVal(false), isInAlloca(false),
      isReturned(false), Alignment(0) { }

    void setAttributes(ImmutableCallSite *CS, unsigned AttrIdx);
  };
  typedef std::vector<ArgListEntry> ArgListTy;

  /// This structure contains all information that is necessary for lowering
  /// calls. It is passed to TLI::LowerCallTo when the SelectionDAG builder
  /// needs to lower a call, and targets will see this struct in their LowerCall
  /// implementation.
  struct CallLoweringInfo {
    SDValue Chain;
    Type *RetTy;
    bool RetSExt           : 1;
    bool RetZExt           : 1;
    bool IsVarArg          : 1;
    bool IsInReg           : 1;
    bool DoesNotReturn     : 1;
    bool IsReturnValueUsed : 1;

    // IsTailCall should be modified by implementations of
    // TargetLowering::LowerCall that perform tail call conversions.
    bool IsTailCall;

    unsigned NumFixedArgs;
    CallingConv::ID CallConv;
    SDValue Callee;
    ArgListTy Args;
    SelectionDAG &DAG;
    SDLoc DL;
    ImmutableCallSite *CS;
    bool IsPatchPoint;
    SmallVector<ISD::OutputArg, 32> Outs;
    SmallVector<SDValue, 32> OutVals;
    SmallVector<ISD::InputArg, 32> Ins;

    CallLoweringInfo(SelectionDAG &DAG)
      : RetTy(nullptr), RetSExt(false), RetZExt(false), IsVarArg(false),
        IsInReg(false), DoesNotReturn(false), IsReturnValueUsed(true),
        IsTailCall(false), NumFixedArgs(-1), CallConv(CallingConv::C),
        DAG(DAG), CS(nullptr), IsPatchPoint(false) {}

    CallLoweringInfo &setDebugLoc(SDLoc dl) {
      DL = dl;
      return *this;
    }

    CallLoweringInfo &setChain(SDValue InChain) {
      Chain = InChain;
      return *this;
    }

    CallLoweringInfo &setCallee(CallingConv::ID CC, Type *ResultType,
                                SDValue Target, ArgListTy &&ArgsList,
                                unsigned FixedArgs = -1) {
      RetTy = ResultType;
      Callee = Target;
      CallConv = CC;
      NumFixedArgs =
        (FixedArgs == static_cast<unsigned>(-1) ? Args.size() : FixedArgs);
      Args = std::move(ArgsList);
      return *this;
    }

    CallLoweringInfo &setCallee(Type *ResultType, FunctionType *FTy,
                                SDValue Target, ArgListTy &&ArgsList,
                                ImmutableCallSite &Call) {
      RetTy = ResultType;

      IsInReg = Call.paramHasAttr(0, Attribute::InReg);
      DoesNotReturn = Call.doesNotReturn();
      IsVarArg = FTy->isVarArg();
      IsReturnValueUsed = !Call.getInstruction()->use_empty();
      RetSExt = Call.paramHasAttr(0, Attribute::SExt);
      RetZExt = Call.paramHasAttr(0, Attribute::ZExt);

      Callee = Target;

      CallConv = Call.getCallingConv();
      NumFixedArgs = FTy->getNumParams();
      Args = std::move(ArgsList);

      CS = &Call;

      return *this;
    }

    CallLoweringInfo &setInRegister(bool Value = true) {
      IsInReg = Value;
      return *this;
    }

    CallLoweringInfo &setNoReturn(bool Value = true) {
      DoesNotReturn = Value;
      return *this;
    }

    CallLoweringInfo &setVarArg(bool Value = true) {
      IsVarArg = Value;
      return *this;
    }

    CallLoweringInfo &setTailCall(bool Value = true) {
      IsTailCall = Value;
      return *this;
    }

    CallLoweringInfo &setDiscardResult(bool Value = true) {
      IsReturnValueUsed = !Value;
      return *this;
    }

    CallLoweringInfo &setSExtResult(bool Value = true) {
      RetSExt = Value;
      return *this;
    }

    CallLoweringInfo &setZExtResult(bool Value = true) {
      RetZExt = Value;
      return *this;
    }

    CallLoweringInfo &setIsPatchPoint(bool Value = true) {
      IsPatchPoint = Value;
      return *this;
    }

    ArgListTy &getArgs() {
      return Args;
    }

  };

  /// This function lowers an abstract call to a function into an actual call.
  /// This returns a pair of operands.  The first element is the return value
  /// for the function (if RetTy is not VoidTy).  The second element is the
  /// outgoing token chain. It calls LowerCall to do the actual lowering.
  std::pair<SDValue, SDValue> LowerCallTo(CallLoweringInfo &CLI) const;

  /// This hook must be implemented to lower calls into the specified
  /// DAG. The outgoing arguments to the call are described by the Outs array,
  /// and the values to be returned by the call are described by the Ins
  /// array. The implementation should fill in the InVals array with legal-type
  /// return values from the call, and return the resulting token chain value.
  virtual SDValue
    LowerCall(CallLoweringInfo &/*CLI*/,
              SmallVectorImpl<SDValue> &/*InVals*/) const {
    llvm_unreachable("Not Implemented");
  }

  /// Target-specific cleanup for formal ByVal parameters.
  virtual void HandleByVal(CCState *, unsigned &, unsigned) const {}

  /// This hook should be implemented to check whether the return values
  /// described by the Outs array can fit into the return registers.  If false
  /// is returned, an sret-demotion is performed.
  virtual bool CanLowerReturn(CallingConv::ID /*CallConv*/,
                              MachineFunction &/*MF*/, bool /*isVarArg*/,
               const SmallVectorImpl<ISD::OutputArg> &/*Outs*/,
               LLVMContext &/*Context*/) const
  {
    // Return true by default to get preexisting behavior.
    return true;
  }

  /// This hook must be implemented to lower outgoing return values, described
  /// by the Outs array, into the specified DAG. The implementation should
  /// return the resulting token chain value.
  virtual SDValue
    LowerReturn(SDValue /*Chain*/, CallingConv::ID /*CallConv*/,
                bool /*isVarArg*/,
                const SmallVectorImpl<ISD::OutputArg> &/*Outs*/,
                const SmallVectorImpl<SDValue> &/*OutVals*/,
                SDLoc /*dl*/, SelectionDAG &/*DAG*/) const {
    llvm_unreachable("Not Implemented");
  }

  /// Return true if result of the specified node is used by a return node
  /// only. It also compute and return the input chain for the tail call.
  ///
  /// This is used to determine whether it is possible to codegen a libcall as
  /// tail call at legalization time.
  virtual bool isUsedByReturnOnly(SDNode *, SDValue &/*Chain*/) const {
    return false;
  }

  /// Return true if the target may be able emit the call instruction as a tail
  /// call. This is used by optimization passes to determine if it's profitable
  /// to duplicate return instructions to enable tailcall optimization.
  virtual bool mayBeEmittedAsTailCall(CallInst *) const {
    return false;
  }

  /// Return the builtin name for the __builtin___clear_cache intrinsic
  /// Default is to invoke the clear cache library call
  virtual const char * getClearCacheBuiltinName() const {
    return "__clear_cache";
  }

  /// Return the register ID of the name passed in. Used by named register
  /// global variables extension. There is no target-independent behaviour
  /// so the default action is to bail.
  virtual unsigned getRegisterByName(const char* RegName, EVT VT,
                                     SelectionDAG &DAG) const {
    report_fatal_error("Named registers not implemented for this target");
  }

  /// Return the type that should be used to zero or sign extend a
  /// zeroext/signext integer argument or return value.  FIXME: Most C calling
  /// convention requires the return type to be promoted, but this is not true
  /// all the time, e.g. i1 on x86-64. It is also not necessary for non-C
  /// calling conventions. The frontend should handle this and include all of
  /// the necessary information.
  virtual EVT getTypeForExtArgOrReturn(LLVMContext &Context, EVT VT,
                                       ISD::NodeType /*ExtendKind*/) const {
    EVT MinVT = getRegisterType(Context, MVT::i32);
    return VT.bitsLT(MinVT) ? MinVT : VT;
  }

  /// For some targets, an LLVM struct type must be broken down into multiple
  /// simple types, but the calling convention specifies that the entire struct
  /// must be passed in a block of consecutive registers.
  virtual bool
  functionArgumentNeedsConsecutiveRegisters(Type *Ty, CallingConv::ID CallConv,
                                            bool isVarArg) const {
    return false;
  }

  /// Returns a 0 terminated array of registers that can be safely used as
  /// scratch registers.
  virtual const MCPhysReg *getScratchRegisters(CallingConv::ID CC) const {
    return nullptr;
  }

  /// This callback is used to prepare for a volatile or atomic load.
  /// It takes a chain node as input and returns the chain for the load itself.
  ///
  /// Having a callback like this is necessary for targets like SystemZ,
  /// which allows a CPU to reuse the result of a previous load indefinitely,
  /// even if a cache-coherent store is performed by another CPU.  The default
  /// implementation does nothing.
  virtual SDValue prepareVolatileOrAtomicLoad(SDValue Chain, SDLoc DL,
                                              SelectionDAG &DAG) const {
    return Chain;
  }

  /// This callback is invoked by the type legalizer to legalize nodes with an
  /// illegal operand type but legal result types.  It replaces the
  /// LowerOperation callback in the type Legalizer.  The reason we can not do
  /// away with LowerOperation entirely is that LegalizeDAG isn't yet ready to
  /// use this callback.
  ///
  /// TODO: Consider merging with ReplaceNodeResults.
  ///
  /// The target places new result values for the node in Results (their number
  /// and types must exactly match those of the original return values of
  /// the node), or leaves Results empty, which indicates that the node is not
  /// to be custom lowered after all.
  /// The default implementation calls LowerOperation.
  virtual void LowerOperationWrapper(SDNode *N,
                                     SmallVectorImpl<SDValue> &Results,
                                     SelectionDAG &DAG) const;

  /// This callback is invoked for operations that are unsupported by the
  /// target, which are registered to use 'custom' lowering, and whose defined
  /// values are all legal.  If the target has no operations that require custom
  /// lowering, it need not implement this.  The default implementation of this
  /// aborts.
  virtual SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const;

  /// This callback is invoked when a node result type is illegal for the
  /// target, and the operation was registered to use 'custom' lowering for that
  /// result type.  The target places new result values for the node in Results
  /// (their number and types must exactly match those of the original return
  /// values of the node), or leaves Results empty, which indicates that the
  /// node is not to be custom lowered after all.
  ///
  /// If the target has no operations that require custom lowering, it need not
  /// implement this.  The default implementation aborts.
  virtual void ReplaceNodeResults(SDNode * /*N*/,
                                  SmallVectorImpl<SDValue> &/*Results*/,
                                  SelectionDAG &/*DAG*/) const {
    llvm_unreachable("ReplaceNodeResults not implemented for this target!");
  }

  /// This method returns the name of a target specific DAG node.
  virtual const char *getTargetNodeName(unsigned Opcode) const;

  /// This method returns a target specific FastISel object, or null if the
  /// target does not support "fast" ISel.
  virtual FastISel *createFastISel(FunctionLoweringInfo &,
                                   const TargetLibraryInfo *) const {
    return nullptr;
  }


  bool verifyReturnAddressArgumentIsConstant(SDValue Op,
                                             SelectionDAG &DAG) const;

  //===--------------------------------------------------------------------===//
  // Inline Asm Support hooks
  //

  /// This hook allows the target to expand an inline asm call to be explicit
  /// llvm code if it wants to.  This is useful for turning simple inline asms
  /// into LLVM intrinsics, which gives the compiler more information about the
  /// behavior of the code.
  virtual bool ExpandInlineAsm(CallInst *) const {
    return false;
  }

  enum ConstraintType {
    C_Register,            // Constraint represents specific register(s).
    C_RegisterClass,       // Constraint represents any of register(s) in class.
    C_Memory,              // Memory constraint.
    C_Other,               // Something else.
    C_Unknown              // Unsupported constraint.
  };

  enum ConstraintWeight {
    // Generic weights.
    CW_Invalid  = -1,     // No match.
    CW_Okay     = 0,      // Acceptable.
    CW_Good     = 1,      // Good weight.
    CW_Better   = 2,      // Better weight.
    CW_Best     = 3,      // Best weight.

    // Well-known weights.
    CW_SpecificReg  = CW_Okay,    // Specific register operands.
    CW_Register     = CW_Good,    // Register operands.
    CW_Memory       = CW_Better,  // Memory operands.
    CW_Constant     = CW_Best,    // Constant operand.
    CW_Default      = CW_Okay     // Default or don't know type.
  };

  /// This contains information for each constraint that we are lowering.
  struct AsmOperandInfo : public InlineAsm::ConstraintInfo {
    /// This contains the actual string for the code, like "m".  TargetLowering
    /// picks the 'best' code from ConstraintInfo::Codes that most closely
    /// matches the operand.
    std::string ConstraintCode;

    /// Information about the constraint code, e.g. Register, RegisterClass,
    /// Memory, Other, Unknown.
    TargetLowering::ConstraintType ConstraintType;

    /// If this is the result output operand or a clobber, this is null,
    /// otherwise it is the incoming operand to the CallInst.  This gets
    /// modified as the asm is processed.
    Value *CallOperandVal;

    /// The ValueType for the operand value.
    MVT ConstraintVT;

    /// Return true of this is an input operand that is a matching constraint
    /// like "4".
    bool isMatchingInputConstraint() const;

    /// If this is an input matching constraint, this method returns the output
    /// operand it matches.
    unsigned getMatchedOperand() const;

    /// Copy constructor for copying from a ConstraintInfo.
    AsmOperandInfo(InlineAsm::ConstraintInfo Info)
        : InlineAsm::ConstraintInfo(std::move(Info)),
          ConstraintType(TargetLowering::C_Unknown), CallOperandVal(nullptr),
          ConstraintVT(MVT::Other) {}
  };

  typedef std::vector<AsmOperandInfo> AsmOperandInfoVector;

  /// Split up the constraint string from the inline assembly value into the
  /// specific constraints and their prefixes, and also tie in the associated
  /// operand values.  If this returns an empty vector, and if the constraint
  /// string itself isn't empty, there was an error parsing.
  virtual AsmOperandInfoVector ParseConstraints(const DataLayout &DL,
                                                const TargetRegisterInfo *TRI,
                                                ImmutableCallSite CS) const;

  /// Examine constraint type and operand type and determine a weight value.
  /// The operand object must already have been set up with the operand type.
  virtual ConstraintWeight getMultipleConstraintMatchWeight(
      AsmOperandInfo &info, int maIndex) const;

  /// Examine constraint string and operand type and determine a weight value.
  /// The operand object must already have been set up with the operand type.
  virtual ConstraintWeight getSingleConstraintMatchWeight(
      AsmOperandInfo &info, const char *constraint) const;

  /// Determines the constraint code and constraint type to use for the specific
  /// AsmOperandInfo, setting OpInfo.ConstraintCode and OpInfo.ConstraintType.
  /// If the actual operand being passed in is available, it can be passed in as
  /// Op, otherwise an empty SDValue can be passed.
  virtual void ComputeConstraintToUse(AsmOperandInfo &OpInfo,
                                      SDValue Op,
                                      SelectionDAG *DAG = nullptr) const;

  /// Given a constraint, return the type of constraint it is for this target.
  virtual ConstraintType getConstraintType(StringRef Constraint) const;

  /// Given a physical register constraint (e.g.  {edx}), return the register
  /// number and the register class for the register.
  ///
  /// Given a register class constraint, like 'r', if this corresponds directly
  /// to an LLVM register class, return a register of 0 and the register class
  /// pointer.
  ///
  /// This should only be used for C_Register constraints.  On error, this
  /// returns a register number of 0 and a null register class pointer.
  virtual std::pair<unsigned, const TargetRegisterClass *>
  getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                               StringRef Constraint, MVT VT) const;

  virtual unsigned getInlineAsmMemConstraint(StringRef ConstraintCode) const {
    if (ConstraintCode == "i")
      return InlineAsm::Constraint_i;
    else if (ConstraintCode == "m")
      return InlineAsm::Constraint_m;
    return InlineAsm::Constraint_Unknown;
  }

  /// Try to replace an X constraint, which matches anything, with another that
  /// has more specific requirements based on the type of the corresponding
  /// operand.  This returns null if there is no replacement to make.
  virtual const char *LowerXConstraint(EVT ConstraintVT) const;

  /// Lower the specified operand into the Ops vector.  If it is invalid, don't
  /// add anything to Ops.
  virtual void LowerAsmOperandForConstraint(SDValue Op, std::string &Constraint,
                                            std::vector<SDValue> &Ops,
                                            SelectionDAG &DAG) const;

  //===--------------------------------------------------------------------===//
  // Div utility functions
  //
  SDValue BuildSDIV(SDNode *N, const APInt &Divisor, SelectionDAG &DAG,
                    bool IsAfterLegalization,
                    std::vector<SDNode *> *Created) const;
  SDValue BuildUDIV(SDNode *N, const APInt &Divisor, SelectionDAG &DAG,
                    bool IsAfterLegalization,
                    std::vector<SDNode *> *Created) const;
  virtual SDValue BuildSDIVPow2(SDNode *N, const APInt &Divisor,
                                SelectionDAG &DAG,
                                std::vector<SDNode *> *Created) const {
    return SDValue();
  }

  /// Indicate whether this target prefers to combine the given number of FDIVs
  /// with the same divisor.
  virtual bool combineRepeatedFPDivisors(unsigned NumUsers) const {
    return false;
  }

  /// Hooks for building estimates in place of slower divisions and square
  /// roots.

  /// Return a reciprocal square root estimate value for the input operand.
  /// The RefinementSteps output is the number of Newton-Raphson refinement
  /// iterations required to generate a sufficient (though not necessarily
  /// IEEE-754 compliant) estimate for the value type.
  /// The boolean UseOneConstNR output is used to select a Newton-Raphson
  /// algorithm implementation that uses one constant or two constants.
  /// A target may choose to implement its own refinement within this function.
  /// If that's true, then return '0' as the number of RefinementSteps to avoid
  /// any further refinement of the estimate.
  /// An empty SDValue return means no estimate sequence can be created.
  virtual SDValue getRsqrtEstimate(SDValue Operand, DAGCombinerInfo &DCI,
                                   unsigned &RefinementSteps,
                                   bool &UseOneConstNR) const {
    return SDValue();
  }

  /// Return a reciprocal estimate value for the input operand.
  /// The RefinementSteps output is the number of Newton-Raphson refinement
  /// iterations required to generate a sufficient (though not necessarily
  /// IEEE-754 compliant) estimate for the value type.
  /// A target may choose to implement its own refinement within this function.
  /// If that's true, then return '0' as the number of RefinementSteps to avoid
  /// any further refinement of the estimate.
  /// An empty SDValue return means no estimate sequence can be created.
  virtual SDValue getRecipEstimate(SDValue Operand, DAGCombinerInfo &DCI,
                                   unsigned &RefinementSteps) const {
    return SDValue();
  }

  //===--------------------------------------------------------------------===//
  // Legalization utility functions
  //

  /// Expand a MUL into two nodes.  One that computes the high bits of
  /// the result and one that computes the low bits.
  /// \param HiLoVT The value type to use for the Lo and Hi nodes.
  /// \param LL Low bits of the LHS of the MUL.  You can use this parameter
  ///        if you want to control how low bits are extracted from the LHS.
  /// \param LH High bits of the LHS of the MUL.  See LL for meaning.
  /// \param RL Low bits of the RHS of the MUL.  See LL for meaning
  /// \param RH High bits of the RHS of the MUL.  See LL for meaning.
  /// \returns true if the node has been expanded. false if it has not
  bool expandMUL(SDNode *N, SDValue &Lo, SDValue &Hi, EVT HiLoVT,
                 SelectionDAG &DAG, SDValue LL = SDValue(),
                 SDValue LH = SDValue(), SDValue RL = SDValue(),
                 SDValue RH = SDValue()) const;

  /// Expand float(f32) to SINT(i64) conversion
  /// \param N Node to expand
  /// \param Result output after conversion
  /// \returns True, if the expansion was successful, false otherwise
  bool expandFP_TO_SINT(SDNode *N, SDValue &Result, SelectionDAG &DAG) const;

  //===--------------------------------------------------------------------===//
  // Instruction Emitting Hooks
  //

  /// This method should be implemented by targets that mark instructions with
  /// the 'usesCustomInserter' flag.  These instructions are special in various
  /// ways, which require special support to insert.  The specified MachineInstr
  /// is created but not inserted into any basic blocks, and this method is
  /// called to expand it into a sequence of instructions, potentially also
  /// creating new basic blocks and control flow.
  /// As long as the returned basic block is different (i.e., we created a new
  /// one), the custom inserter is free to modify the rest of \p MBB.
  virtual MachineBasicBlock *
    EmitInstrWithCustomInserter(MachineInstr *MI, MachineBasicBlock *MBB) const;

  /// This method should be implemented by targets that mark instructions with
  /// the 'hasPostISelHook' flag. These instructions must be adjusted after
  /// instruction selection by target hooks.  e.g. To fill in optional defs for
  /// ARM 's' setting instructions.
  virtual void
  AdjustInstrPostInstrSelection(MachineInstr *MI, SDNode *Node) const;

  /// If this function returns true, SelectionDAGBuilder emits a
  /// LOAD_STACK_GUARD node when it is lowering Intrinsic::stackprotector.
  virtual bool useLoadStackGuardNode() const {
    return false;
  }
};

/// Given an LLVM IR type and return type attributes, compute the return value
/// EVTs and flags, and optionally also the offsets, if the return value is
/// being lowered to memory.
void GetReturnInfo(Type *ReturnType, AttributeSet attr,
                   SmallVectorImpl<ISD::OutputArg> &Outs,
                   const TargetLowering &TLI, const DataLayout &DL);

} // end llvm namespace

#endif
