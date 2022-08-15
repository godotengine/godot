//===- TargetTransformInfo.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This pass exposes codegen information to IR-level passes. Every
/// transformation that uses codegen information is broken into three parts:
/// 1. The IR-level analysis pass.
/// 2. The IR-level transformation interface which provides the needed
///    information.
/// 3. Codegen-level implementation which uses target-specific hooks.
///
/// This file defines #2, which is the interface that IR-level transformations
/// use for querying the codegen.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_TARGETTRANSFORMINFO_H
#define LLVM_ANALYSIS_TARGETTRANSFORMINFO_H

#include "llvm/ADT/Optional.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Pass.h"
#include "llvm/Support/DataTypes.h"
#include <functional>

namespace llvm {

class Function;
class GlobalValue;
class Loop;
class PreservedAnalyses;
class Type;
class User;
class Value;

/// \brief Information about a load/store intrinsic defined by the target.
struct MemIntrinsicInfo {
  MemIntrinsicInfo()
      : ReadMem(false), WriteMem(false), Vol(false), MatchingId(0),
        NumMemRefs(0), PtrVal(nullptr) {}
  bool ReadMem;
  bool WriteMem;
  bool Vol;
  // Same Id is set by the target for corresponding load/store intrinsics.
  unsigned short MatchingId;
  int NumMemRefs;
  Value *PtrVal;
};

/// \brief This pass provides access to the codegen interfaces that are needed
/// for IR-level transformations.
class TargetTransformInfo {
public:
  /// \brief Construct a TTI object using a type implementing the \c Concept
  /// API below.
  ///
  /// This is used by targets to construct a TTI wrapping their target-specific
  /// implementaion that encodes appropriate costs for their target.
  template <typename T> TargetTransformInfo(T Impl);

  /// \brief Construct a baseline TTI object using a minimal implementation of
  /// the \c Concept API below.
  ///
  /// The TTI implementation will reflect the information in the DataLayout
  /// provided if non-null.
  explicit TargetTransformInfo(const DataLayout &DL);

  // Provide move semantics.
  TargetTransformInfo(TargetTransformInfo &&Arg);
  TargetTransformInfo &operator=(TargetTransformInfo &&RHS);

  // We need to define the destructor out-of-line to define our sub-classes
  // out-of-line.
  ~TargetTransformInfo();

  /// \brief Handle the invalidation of this information.
  ///
  /// When used as a result of \c TargetIRAnalysis this method will be called
  /// when the function this was computed for changes. When it returns false,
  /// the information is preserved across those changes.
  bool invalidate(Function &, const PreservedAnalyses &) {
    // FIXME: We should probably in some way ensure that the subtarget
    // information for a function hasn't changed.
    return false;
  }

  /// \name Generic Target Information
  /// @{

  /// \brief Underlying constants for 'cost' values in this interface.
  ///
  /// Many APIs in this interface return a cost. This enum defines the
  /// fundamental values that should be used to interpret (and produce) those
  /// costs. The costs are returned as an unsigned rather than a member of this
  /// enumeration because it is expected that the cost of one IR instruction
  /// may have a multiplicative factor to it or otherwise won't fit directly
  /// into the enum. Moreover, it is common to sum or average costs which works
  /// better as simple integral values. Thus this enum only provides constants.
  ///
  /// Note that these costs should usually reflect the intersection of code-size
  /// cost and execution cost. A free instruction is typically one that folds
  /// into another instruction. For example, reg-to-reg moves can often be
  /// skipped by renaming the registers in the CPU, but they still are encoded
  /// and thus wouldn't be considered 'free' here.
  enum TargetCostConstants {
    TCC_Free = 0,     ///< Expected to fold away in lowering.
    TCC_Basic = 1,    ///< The cost of a typical 'add' instruction.
    TCC_Expensive = 4 ///< The cost of a 'div' instruction on x86.
  };

  /// \brief Estimate the cost of a specific operation when lowered.
  ///
  /// Note that this is designed to work on an arbitrary synthetic opcode, and
  /// thus work for hypothetical queries before an instruction has even been
  /// formed. However, this does *not* work for GEPs, and must not be called
  /// for a GEP instruction. Instead, use the dedicated getGEPCost interface as
  /// analyzing a GEP's cost required more information.
  ///
  /// Typically only the result type is required, and the operand type can be
  /// omitted. However, if the opcode is one of the cast instructions, the
  /// operand type is required.
  ///
  /// The returned cost is defined in terms of \c TargetCostConstants, see its
  /// comments for a detailed explanation of the cost values.
  unsigned getOperationCost(unsigned Opcode, Type *Ty,
                            Type *OpTy = nullptr) const;

  /// \brief Estimate the cost of a GEP operation when lowered.
  ///
  /// The contract for this function is the same as \c getOperationCost except
  /// that it supports an interface that provides extra information specific to
  /// the GEP operation.
  unsigned getGEPCost(const Value *Ptr, ArrayRef<const Value *> Operands) const;

  /// \brief Estimate the cost of a function call when lowered.
  ///
  /// The contract for this is the same as \c getOperationCost except that it
  /// supports an interface that provides extra information specific to call
  /// instructions.
  ///
  /// This is the most basic query for estimating call cost: it only knows the
  /// function type and (potentially) the number of arguments at the call site.
  /// The latter is only interesting for varargs function types.
  unsigned getCallCost(FunctionType *FTy, int NumArgs = -1) const;

  /// \brief Estimate the cost of calling a specific function when lowered.
  ///
  /// This overload adds the ability to reason about the particular function
  /// being called in the event it is a library call with special lowering.
  unsigned getCallCost(const Function *F, int NumArgs = -1) const;

  /// \brief Estimate the cost of calling a specific function when lowered.
  ///
  /// This overload allows specifying a set of candidate argument values.
  unsigned getCallCost(const Function *F,
                       ArrayRef<const Value *> Arguments) const;

  /// \brief Estimate the cost of an intrinsic when lowered.
  ///
  /// Mirrors the \c getCallCost method but uses an intrinsic identifier.
  unsigned getIntrinsicCost(Intrinsic::ID IID, Type *RetTy,
                            ArrayRef<Type *> ParamTys) const;

  /// \brief Estimate the cost of an intrinsic when lowered.
  ///
  /// Mirrors the \c getCallCost method but uses an intrinsic identifier.
  unsigned getIntrinsicCost(Intrinsic::ID IID, Type *RetTy,
                            ArrayRef<const Value *> Arguments) const;

  /// \brief Estimate the cost of a given IR user when lowered.
  ///
  /// This can estimate the cost of either a ConstantExpr or Instruction when
  /// lowered. It has two primary advantages over the \c getOperationCost and
  /// \c getGEPCost above, and one significant disadvantage: it can only be
  /// used when the IR construct has already been formed.
  ///
  /// The advantages are that it can inspect the SSA use graph to reason more
  /// accurately about the cost. For example, all-constant-GEPs can often be
  /// folded into a load or other instruction, but if they are used in some
  /// other context they may not be folded. This routine can distinguish such
  /// cases.
  ///
  /// The returned cost is defined in terms of \c TargetCostConstants, see its
  /// comments for a detailed explanation of the cost values.
  unsigned getUserCost(const User *U) const;

  /// \brief Return true if branch divergence exists.
  ///
  /// Branch divergence has a significantly negative impact on GPU performance
  /// when threads in the same wavefront take different paths due to conditional
  /// branches.
  bool hasBranchDivergence() const;

  /// \brief Returns whether V is a source of divergence.
  ///
  /// This function provides the target-dependent information for
  /// the target-independent DivergenceAnalysis. DivergenceAnalysis first
  /// builds the dependency graph, and then runs the reachability algorithm
  /// starting with the sources of divergence.
  bool isSourceOfDivergence(const Value *V) const;

  /// \brief Test whether calls to a function lower to actual program function
  /// calls.
  ///
  /// The idea is to test whether the program is likely to require a 'call'
  /// instruction or equivalent in order to call the given function.
  ///
  /// FIXME: It's not clear that this is a good or useful query API. Client's
  /// should probably move to simpler cost metrics using the above.
  /// Alternatively, we could split the cost interface into distinct code-size
  /// and execution-speed costs. This would allow modelling the core of this
  /// query more accurately as a call is a single small instruction, but
  /// incurs significant execution cost.
  bool isLoweredToCall(const Function *F) const;

  /// Parameters that control the generic loop unrolling transformation.
  struct UnrollingPreferences {
    /// The cost threshold for the unrolled loop. Should be relative to the
    /// getUserCost values returned by this API, and the expectation is that
    /// the unrolled loop's instructions when run through that interface should
    /// not exceed this cost. However, this is only an estimate. Also, specific
    /// loops may be unrolled even with a cost above this threshold if deemed
    /// profitable. Set this to UINT_MAX to disable the loop body cost
    /// restriction.
    unsigned Threshold;
    /// If complete unrolling will reduce the cost of the loop below its
    /// expected dynamic cost while rolled by this percentage, apply a discount
    /// (below) to its unrolled cost.
    unsigned PercentDynamicCostSavedThreshold;
    /// The discount applied to the unrolled cost when the *dynamic* cost
    /// savings of unrolling exceed the \c PercentDynamicCostSavedThreshold.
    unsigned DynamicCostSavingsDiscount;
    /// The cost threshold for the unrolled loop when optimizing for size (set
    /// to UINT_MAX to disable).
    unsigned OptSizeThreshold;
    /// The cost threshold for the unrolled loop, like Threshold, but used
    /// for partial/runtime unrolling (set to UINT_MAX to disable).
    unsigned PartialThreshold;
    /// The cost threshold for the unrolled loop when optimizing for size, like
    /// OptSizeThreshold, but used for partial/runtime unrolling (set to
    /// UINT_MAX to disable).
    unsigned PartialOptSizeThreshold;
    /// A forced unrolling factor (the number of concatenated bodies of the
    /// original loop in the unrolled loop body). When set to 0, the unrolling
    /// transformation will select an unrolling factor based on the current cost
    /// threshold and other factors.
    unsigned Count;
    // Set the maximum unrolling factor. The unrolling factor may be selected
    // using the appropriate cost threshold, but may not exceed this number
    // (set to UINT_MAX to disable). This does not apply in cases where the
    // loop is being fully unrolled.
    unsigned MaxCount;
    /// Allow partial unrolling (unrolling of loops to expand the size of the
    /// loop body, not only to eliminate small constant-trip-count loops).
    bool Partial;
    /// Allow runtime unrolling (unrolling of loops to expand the size of the
    /// loop body even when the number of loop iterations is not known at
    /// compile time).
    bool Runtime;
    /// Allow emitting expensive instructions (such as divisions) when computing
    /// the trip count of a loop for runtime unrolling.
    bool AllowExpensiveTripCount;
  };

  /// \brief Get target-customized preferences for the generic loop unrolling
  /// transformation. The caller will initialize UP with the current
  /// target-independent defaults.
  void getUnrollingPreferences(Loop *L, UnrollingPreferences &UP) const;

  /// @}

  /// \name Scalar Target Information
  /// @{

  /// \brief Flags indicating the kind of support for population count.
  ///
  /// Compared to the SW implementation, HW support is supposed to
  /// significantly boost the performance when the population is dense, and it
  /// may or may not degrade performance if the population is sparse. A HW
  /// support is considered as "Fast" if it can outperform, or is on a par
  /// with, SW implementation when the population is sparse; otherwise, it is
  /// considered as "Slow".
  enum PopcntSupportKind { PSK_Software, PSK_SlowHardware, PSK_FastHardware };

  /// \brief Return true if the specified immediate is legal add immediate, that
  /// is the target has add instructions which can add a register with the
  /// immediate without having to materialize the immediate into a register.
  bool isLegalAddImmediate(int64_t Imm) const;

  /// \brief Return true if the specified immediate is legal icmp immediate,
  /// that is the target has icmp instructions which can compare a register
  /// against the immediate without having to materialize the immediate into a
  /// register.
  bool isLegalICmpImmediate(int64_t Imm) const;

  /// \brief Return true if the addressing mode represented by AM is legal for
  /// this target, for a load/store of the specified type.
  /// The type may be VoidTy, in which case only return true if the addressing
  /// mode is legal for a load/store of any legal type.
  /// TODO: Handle pre/postinc as well.
  bool isLegalAddressingMode(Type *Ty, GlobalValue *BaseGV, int64_t BaseOffset,
                             bool HasBaseReg, int64_t Scale,
                             unsigned AddrSpace = 0) const;

  /// \brief Return true if the target works with masked instruction
  /// AVX2 allows masks for consecutive load and store for i32 and i64 elements.
  /// AVX-512 architecture will also allow masks for non-consecutive memory
  /// accesses.
  bool isLegalMaskedStore(Type *DataType, int Consecutive) const;
  bool isLegalMaskedLoad(Type *DataType, int Consecutive) const;

  /// \brief Return the cost of the scaling factor used in the addressing
  /// mode represented by AM for this target, for a load/store
  /// of the specified type.
  /// If the AM is supported, the return value must be >= 0.
  /// If the AM is not supported, it returns a negative value.
  /// TODO: Handle pre/postinc as well.
  int getScalingFactorCost(Type *Ty, GlobalValue *BaseGV, int64_t BaseOffset,
                           bool HasBaseReg, int64_t Scale,
                           unsigned AddrSpace = 0) const;

  /// \brief Return true if it's free to truncate a value of type Ty1 to type
  /// Ty2. e.g. On x86 it's free to truncate a i32 value in register EAX to i16
  /// by referencing its sub-register AX.
  bool isTruncateFree(Type *Ty1, Type *Ty2) const;

  /// \brief Return true if it is profitable to hoist instruction in the
  /// then/else to before if.
  bool isProfitableToHoist(Instruction *I) const;

  /// \brief Return true if this type is legal.
  bool isTypeLegal(Type *Ty) const;

  /// \brief Returns the target's jmp_buf alignment in bytes.
  unsigned getJumpBufAlignment() const;

  /// \brief Returns the target's jmp_buf size in bytes.
  unsigned getJumpBufSize() const;

  /// \brief Return true if switches should be turned into lookup tables for the
  /// target.
  bool shouldBuildLookupTables() const;

  /// \brief Don't restrict interleaved unrolling to small loops.
  bool enableAggressiveInterleaving(bool LoopHasReductions) const;

  /// \brief Return hardware support for population count.
  PopcntSupportKind getPopcntSupport(unsigned IntTyWidthInBit) const;

  /// \brief Return true if the hardware has a fast square-root instruction.
  bool haveFastSqrt(Type *Ty) const;

  /// \brief Return the expected cost of supporting the floating point operation
  /// of the specified type.
  unsigned getFPOpCost(Type *Ty) const;

  /// \brief Return the expected cost of materializing for the given integer
  /// immediate of the specified type.
  unsigned getIntImmCost(const APInt &Imm, Type *Ty) const;

  /// \brief Return the expected cost of materialization for the given integer
  /// immediate of the specified type for a given instruction. The cost can be
  /// zero if the immediate can be folded into the specified instruction.
  unsigned getIntImmCost(unsigned Opc, unsigned Idx, const APInt &Imm,
                         Type *Ty) const;
  unsigned getIntImmCost(Intrinsic::ID IID, unsigned Idx, const APInt &Imm,
                         Type *Ty) const;
  /// @}

  /// \name Vector Target Information
  /// @{

  /// \brief The various kinds of shuffle patterns for vector queries.
  enum ShuffleKind {
    SK_Broadcast,       ///< Broadcast element 0 to all other elements.
    SK_Reverse,         ///< Reverse the order of the vector.
    SK_Alternate,       ///< Choose alternate elements from vector.
    SK_InsertSubvector, ///< InsertSubvector. Index indicates start offset.
    SK_ExtractSubvector ///< ExtractSubvector Index indicates start offset.
  };

  /// \brief Additional information about an operand's possible values.
  enum OperandValueKind {
    OK_AnyValue,               // Operand can have any value.
    OK_UniformValue,           // Operand is uniform (splat of a value).
    OK_UniformConstantValue,   // Operand is uniform constant.
    OK_NonUniformConstantValue // Operand is a non uniform constant value.
  };

  /// \brief Additional properties of an operand's values.
  enum OperandValueProperties { OP_None = 0, OP_PowerOf2 = 1 };

  /// \return The number of scalar or vector registers that the target has.
  /// If 'Vectors' is true, it returns the number of vector registers. If it is
  /// set to false, it returns the number of scalar registers.
  unsigned getNumberOfRegisters(bool Vector) const;

  /// \return The width of the largest scalar or vector register type.
  unsigned getRegisterBitWidth(bool Vector) const;

  /// \return The maximum interleave factor that any transform should try to
  /// perform for this target. This number depends on the level of parallelism
  /// and the number of execution units in the CPU.
  unsigned getMaxInterleaveFactor(unsigned VF) const;

  /// \return The expected cost of arithmetic ops, such as mul, xor, fsub, etc.
  unsigned
  getArithmeticInstrCost(unsigned Opcode, Type *Ty,
                         OperandValueKind Opd1Info = OK_AnyValue,
                         OperandValueKind Opd2Info = OK_AnyValue,
                         OperandValueProperties Opd1PropInfo = OP_None,
                         OperandValueProperties Opd2PropInfo = OP_None) const;

  /// \return The cost of a shuffle instruction of kind Kind and of type Tp.
  /// The index and subtype parameters are used by the subvector insertion and
  /// extraction shuffle kinds.
  unsigned getShuffleCost(ShuffleKind Kind, Type *Tp, int Index = 0,
                          Type *SubTp = nullptr) const;

  /// \return The expected cost of cast instructions, such as bitcast, trunc,
  /// zext, etc.
  unsigned getCastInstrCost(unsigned Opcode, Type *Dst, Type *Src) const;

  /// \return The expected cost of control-flow related instructions such as
  /// Phi, Ret, Br.
  unsigned getCFInstrCost(unsigned Opcode) const;

  /// \returns The expected cost of compare and select instructions.
  unsigned getCmpSelInstrCost(unsigned Opcode, Type *ValTy,
                              Type *CondTy = nullptr) const;

  /// \return The expected cost of vector Insert and Extract.
  /// Use -1 to indicate that there is no information on the index value.
  unsigned getVectorInstrCost(unsigned Opcode, Type *Val,
                              unsigned Index = -1) const;

  /// \return The cost of Load and Store instructions.
  unsigned getMemoryOpCost(unsigned Opcode, Type *Src, unsigned Alignment,
                           unsigned AddressSpace) const;

  /// \return The cost of masked Load and Store instructions.
  unsigned getMaskedMemoryOpCost(unsigned Opcode, Type *Src, unsigned Alignment,
                                 unsigned AddressSpace) const;

  /// \return The cost of the interleaved memory operation.
  /// \p Opcode is the memory operation code
  /// \p VecTy is the vector type of the interleaved access.
  /// \p Factor is the interleave factor
  /// \p Indices is the indices for interleaved load members (as interleaved
  ///    load allows gaps)
  /// \p Alignment is the alignment of the memory operation
  /// \p AddressSpace is address space of the pointer.
  unsigned getInterleavedMemoryOpCost(unsigned Opcode, Type *VecTy,
                                      unsigned Factor,
                                      ArrayRef<unsigned> Indices,
                                      unsigned Alignment,
                                      unsigned AddressSpace) const;

  /// \brief Calculate the cost of performing a vector reduction.
  ///
  /// This is the cost of reducing the vector value of type \p Ty to a scalar
  /// value using the operation denoted by \p Opcode. The form of the reduction
  /// can either be a pairwise reduction or a reduction that splits the vector
  /// at every reduction level.
  ///
  /// Pairwise:
  ///  (v0, v1, v2, v3)
  ///  ((v0+v1), (v2, v3), undef, undef)
  /// Split:
  ///  (v0, v1, v2, v3)
  ///  ((v0+v2), (v1+v3), undef, undef)
  unsigned getReductionCost(unsigned Opcode, Type *Ty,
                            bool IsPairwiseForm) const;

  /// \returns The cost of Intrinsic instructions.
  unsigned getIntrinsicInstrCost(Intrinsic::ID ID, Type *RetTy,
                                 ArrayRef<Type *> Tys) const;

  /// \returns The cost of Call instructions.
  unsigned getCallInstrCost(Function *F, Type *RetTy,
                            ArrayRef<Type *> Tys) const;

  /// \returns The number of pieces into which the provided type must be
  /// split during legalization. Zero is returned when the answer is unknown.
  unsigned getNumberOfParts(Type *Tp) const;

  /// \returns The cost of the address computation. For most targets this can be
  /// merged into the instruction indexing mode. Some targets might want to
  /// distinguish between address computation for memory operations on vector
  /// types and scalar types. Such targets should override this function.
  /// The 'IsComplex' parameter is a hint that the address computation is likely
  /// to involve multiple instructions and as such unlikely to be merged into
  /// the address indexing mode.
  unsigned getAddressComputationCost(Type *Ty, bool IsComplex = false) const;

  /// \returns The cost, if any, of keeping values of the given types alive
  /// over a callsite.
  ///
  /// Some types may require the use of register classes that do not have
  /// any callee-saved registers, so would require a spill and fill.
  unsigned getCostOfKeepingLiveOverCall(ArrayRef<Type *> Tys) const;

  /// \returns True if the intrinsic is a supported memory intrinsic.  Info
  /// will contain additional information - whether the intrinsic may write
  /// or read to memory, volatility and the pointer.  Info is undefined
  /// if false is returned.
  bool getTgtMemIntrinsic(IntrinsicInst *Inst, MemIntrinsicInfo &Info) const;

  /// \returns A value which is the result of the given memory intrinsic.  New
  /// instructions may be created to extract the result from the given intrinsic
  /// memory operation.  Returns nullptr if the target cannot create a result
  /// from the given intrinsic.
  Value *getOrCreateResultFromMemIntrinsic(IntrinsicInst *Inst,
                                           Type *ExpectedType) const;

  /// \returns True if the two functions have compatible attributes for inlining
  /// purposes.
  bool hasCompatibleFunctionAttributes(const Function *Caller,
                                       const Function *Callee) const;

  /// @}

private:
  /// \brief The abstract base class used to type erase specific TTI
  /// implementations.
  class Concept;

  /// \brief The template model for the base class which wraps a concrete
  /// implementation in a type erased interface.
  template <typename T> class Model;

  std::unique_ptr<Concept> TTIImpl;
};

class TargetTransformInfo::Concept {
public:
  virtual ~Concept() = 0;
  virtual const DataLayout &getDataLayout() const = 0;
  virtual unsigned getOperationCost(unsigned Opcode, Type *Ty, Type *OpTy) = 0;
  virtual unsigned getGEPCost(const Value *Ptr,
                              ArrayRef<const Value *> Operands) = 0;
  virtual unsigned getCallCost(FunctionType *FTy, int NumArgs) = 0;
  virtual unsigned getCallCost(const Function *F, int NumArgs) = 0;
  virtual unsigned getCallCost(const Function *F,
                               ArrayRef<const Value *> Arguments) = 0;
  virtual unsigned getIntrinsicCost(Intrinsic::ID IID, Type *RetTy,
                                    ArrayRef<Type *> ParamTys) = 0;
  virtual unsigned getIntrinsicCost(Intrinsic::ID IID, Type *RetTy,
                                    ArrayRef<const Value *> Arguments) = 0;
  virtual unsigned getUserCost(const User *U) = 0;
  virtual bool hasBranchDivergence() = 0;
  virtual bool isSourceOfDivergence(const Value *V) = 0;
  virtual bool isLoweredToCall(const Function *F) = 0;
  virtual void getUnrollingPreferences(Loop *L, UnrollingPreferences &UP) = 0;
  virtual bool isLegalAddImmediate(int64_t Imm) = 0;
  virtual bool isLegalICmpImmediate(int64_t Imm) = 0;
  virtual bool isLegalAddressingMode(Type *Ty, GlobalValue *BaseGV,
                                     int64_t BaseOffset, bool HasBaseReg,
                                     int64_t Scale,
                                     unsigned AddrSpace) = 0;
  virtual bool isLegalMaskedStore(Type *DataType, int Consecutive) = 0;
  virtual bool isLegalMaskedLoad(Type *DataType, int Consecutive) = 0;
  virtual int getScalingFactorCost(Type *Ty, GlobalValue *BaseGV,
                                   int64_t BaseOffset, bool HasBaseReg,
                                   int64_t Scale, unsigned AddrSpace) = 0;
  virtual bool isTruncateFree(Type *Ty1, Type *Ty2) = 0;
  virtual bool isProfitableToHoist(Instruction *I) = 0;
  virtual bool isTypeLegal(Type *Ty) = 0;
  virtual unsigned getJumpBufAlignment() = 0;
  virtual unsigned getJumpBufSize() = 0;
  virtual bool shouldBuildLookupTables() = 0;
  virtual bool enableAggressiveInterleaving(bool LoopHasReductions) = 0;
  virtual PopcntSupportKind getPopcntSupport(unsigned IntTyWidthInBit) = 0;
  virtual bool haveFastSqrt(Type *Ty) = 0;
  virtual unsigned getFPOpCost(Type *Ty) = 0;
  virtual unsigned getIntImmCost(const APInt &Imm, Type *Ty) = 0;
  virtual unsigned getIntImmCost(unsigned Opc, unsigned Idx, const APInt &Imm,
                                 Type *Ty) = 0;
  virtual unsigned getIntImmCost(Intrinsic::ID IID, unsigned Idx,
                                 const APInt &Imm, Type *Ty) = 0;
  virtual unsigned getNumberOfRegisters(bool Vector) = 0;
  virtual unsigned getRegisterBitWidth(bool Vector) = 0;
  virtual unsigned getMaxInterleaveFactor(unsigned VF) = 0;
  virtual unsigned
  getArithmeticInstrCost(unsigned Opcode, Type *Ty, OperandValueKind Opd1Info,
                         OperandValueKind Opd2Info,
                         OperandValueProperties Opd1PropInfo,
                         OperandValueProperties Opd2PropInfo) = 0;
  virtual unsigned getShuffleCost(ShuffleKind Kind, Type *Tp, int Index,
                                  Type *SubTp) = 0;
  virtual unsigned getCastInstrCost(unsigned Opcode, Type *Dst, Type *Src) = 0;
  virtual unsigned getCFInstrCost(unsigned Opcode) = 0;
  virtual unsigned getCmpSelInstrCost(unsigned Opcode, Type *ValTy,
                                      Type *CondTy) = 0;
  virtual unsigned getVectorInstrCost(unsigned Opcode, Type *Val,
                                      unsigned Index) = 0;
  virtual unsigned getMemoryOpCost(unsigned Opcode, Type *Src,
                                   unsigned Alignment,
                                   unsigned AddressSpace) = 0;
  virtual unsigned getMaskedMemoryOpCost(unsigned Opcode, Type *Src,
                                         unsigned Alignment,
                                         unsigned AddressSpace) = 0;
  virtual unsigned getInterleavedMemoryOpCost(unsigned Opcode, Type *VecTy,
                                              unsigned Factor,
                                              ArrayRef<unsigned> Indices,
                                              unsigned Alignment,
                                              unsigned AddressSpace) = 0;
  virtual unsigned getReductionCost(unsigned Opcode, Type *Ty,
                                    bool IsPairwiseForm) = 0;
  virtual unsigned getIntrinsicInstrCost(Intrinsic::ID ID, Type *RetTy,
                                         ArrayRef<Type *> Tys) = 0;
  virtual unsigned getCallInstrCost(Function *F, Type *RetTy,
                                    ArrayRef<Type *> Tys) = 0;
  virtual unsigned getNumberOfParts(Type *Tp) = 0;
  virtual unsigned getAddressComputationCost(Type *Ty, bool IsComplex) = 0;
  virtual unsigned getCostOfKeepingLiveOverCall(ArrayRef<Type *> Tys) = 0;
  virtual bool getTgtMemIntrinsic(IntrinsicInst *Inst,
                                  MemIntrinsicInfo &Info) = 0;
  virtual Value *getOrCreateResultFromMemIntrinsic(IntrinsicInst *Inst,
                                                   Type *ExpectedType) = 0;
  virtual bool hasCompatibleFunctionAttributes(const Function *Caller,
                                               const Function *Callee) const = 0;
};

template <typename T>
class TargetTransformInfo::Model final : public TargetTransformInfo::Concept {
  T Impl;

public:
  Model(T Impl) : Impl(std::move(Impl)) {}
  ~Model() override {}

  const DataLayout &getDataLayout() const override {
    return Impl.getDataLayout();
  }

  unsigned getOperationCost(unsigned Opcode, Type *Ty, Type *OpTy) override {
    return Impl.getOperationCost(Opcode, Ty, OpTy);
  }
  unsigned getGEPCost(const Value *Ptr,
                      ArrayRef<const Value *> Operands) override {
    return Impl.getGEPCost(Ptr, Operands);
  }
  unsigned getCallCost(FunctionType *FTy, int NumArgs) override {
    return Impl.getCallCost(FTy, NumArgs);
  }
  unsigned getCallCost(const Function *F, int NumArgs) override {
    return Impl.getCallCost(F, NumArgs);
  }
  unsigned getCallCost(const Function *F,
                       ArrayRef<const Value *> Arguments) override {
    return Impl.getCallCost(F, Arguments);
  }
  unsigned getIntrinsicCost(Intrinsic::ID IID, Type *RetTy,
                            ArrayRef<Type *> ParamTys) override {
    return Impl.getIntrinsicCost(IID, RetTy, ParamTys);
  }
  unsigned getIntrinsicCost(Intrinsic::ID IID, Type *RetTy,
                            ArrayRef<const Value *> Arguments) override {
    return Impl.getIntrinsicCost(IID, RetTy, Arguments);
  }
  unsigned getUserCost(const User *U) override { return Impl.getUserCost(U); }
  bool hasBranchDivergence() override { return Impl.hasBranchDivergence(); }
  bool isSourceOfDivergence(const Value *V) override {
    return Impl.isSourceOfDivergence(V);
  }
  bool isLoweredToCall(const Function *F) override {
    return Impl.isLoweredToCall(F);
  }
  void getUnrollingPreferences(Loop *L, UnrollingPreferences &UP) override {
    return Impl.getUnrollingPreferences(L, UP);
  }
  bool isLegalAddImmediate(int64_t Imm) override {
    return Impl.isLegalAddImmediate(Imm);
  }
  bool isLegalICmpImmediate(int64_t Imm) override {
    return Impl.isLegalICmpImmediate(Imm);
  }
  bool isLegalAddressingMode(Type *Ty, GlobalValue *BaseGV, int64_t BaseOffset,
                             bool HasBaseReg, int64_t Scale,
                             unsigned AddrSpace) override {
    return Impl.isLegalAddressingMode(Ty, BaseGV, BaseOffset, HasBaseReg,
                                      Scale, AddrSpace);
  }
  bool isLegalMaskedStore(Type *DataType, int Consecutive) override {
    return Impl.isLegalMaskedStore(DataType, Consecutive);
  }
  bool isLegalMaskedLoad(Type *DataType, int Consecutive) override {
    return Impl.isLegalMaskedLoad(DataType, Consecutive);
  }
  int getScalingFactorCost(Type *Ty, GlobalValue *BaseGV, int64_t BaseOffset,
                           bool HasBaseReg, int64_t Scale,
                           unsigned AddrSpace) override {
    return Impl.getScalingFactorCost(Ty, BaseGV, BaseOffset, HasBaseReg,
                                     Scale, AddrSpace);
  }
  bool isTruncateFree(Type *Ty1, Type *Ty2) override {
    return Impl.isTruncateFree(Ty1, Ty2);
  }
  bool isProfitableToHoist(Instruction *I) override {
    return Impl.isProfitableToHoist(I);
  }
  bool isTypeLegal(Type *Ty) override { return Impl.isTypeLegal(Ty); }
  unsigned getJumpBufAlignment() override { return Impl.getJumpBufAlignment(); }
  unsigned getJumpBufSize() override { return Impl.getJumpBufSize(); }
  bool shouldBuildLookupTables() override {
    return Impl.shouldBuildLookupTables();
  }
  bool enableAggressiveInterleaving(bool LoopHasReductions) override {
    return Impl.enableAggressiveInterleaving(LoopHasReductions);
  }
  PopcntSupportKind getPopcntSupport(unsigned IntTyWidthInBit) override {
    return Impl.getPopcntSupport(IntTyWidthInBit);
  }
  bool haveFastSqrt(Type *Ty) override { return Impl.haveFastSqrt(Ty); }

  unsigned getFPOpCost(Type *Ty) override {
    return Impl.getFPOpCost(Ty);
  }

  unsigned getIntImmCost(const APInt &Imm, Type *Ty) override {
    return Impl.getIntImmCost(Imm, Ty);
  }
  unsigned getIntImmCost(unsigned Opc, unsigned Idx, const APInt &Imm,
                         Type *Ty) override {
    return Impl.getIntImmCost(Opc, Idx, Imm, Ty);
  }
  unsigned getIntImmCost(Intrinsic::ID IID, unsigned Idx, const APInt &Imm,
                         Type *Ty) override {
    return Impl.getIntImmCost(IID, Idx, Imm, Ty);
  }
  unsigned getNumberOfRegisters(bool Vector) override {
    return Impl.getNumberOfRegisters(Vector);
  }
  unsigned getRegisterBitWidth(bool Vector) override {
    return Impl.getRegisterBitWidth(Vector);
  }
  unsigned getMaxInterleaveFactor(unsigned VF) override {
    return Impl.getMaxInterleaveFactor(VF);
  }
  unsigned
  getArithmeticInstrCost(unsigned Opcode, Type *Ty, OperandValueKind Opd1Info,
                         OperandValueKind Opd2Info,
                         OperandValueProperties Opd1PropInfo,
                         OperandValueProperties Opd2PropInfo) override {
    return Impl.getArithmeticInstrCost(Opcode, Ty, Opd1Info, Opd2Info,
                                       Opd1PropInfo, Opd2PropInfo);
  }
  unsigned getShuffleCost(ShuffleKind Kind, Type *Tp, int Index,
                          Type *SubTp) override {
    return Impl.getShuffleCost(Kind, Tp, Index, SubTp);
  }
  unsigned getCastInstrCost(unsigned Opcode, Type *Dst, Type *Src) override {
    return Impl.getCastInstrCost(Opcode, Dst, Src);
  }
  unsigned getCFInstrCost(unsigned Opcode) override {
    return Impl.getCFInstrCost(Opcode);
  }
  unsigned getCmpSelInstrCost(unsigned Opcode, Type *ValTy,
                              Type *CondTy) override {
    return Impl.getCmpSelInstrCost(Opcode, ValTy, CondTy);
  }
  unsigned getVectorInstrCost(unsigned Opcode, Type *Val,
                              unsigned Index) override {
    return Impl.getVectorInstrCost(Opcode, Val, Index);
  }
  unsigned getMemoryOpCost(unsigned Opcode, Type *Src, unsigned Alignment,
                           unsigned AddressSpace) override {
    return Impl.getMemoryOpCost(Opcode, Src, Alignment, AddressSpace);
  }
  unsigned getMaskedMemoryOpCost(unsigned Opcode, Type *Src, unsigned Alignment,
                                 unsigned AddressSpace) override {
    return Impl.getMaskedMemoryOpCost(Opcode, Src, Alignment, AddressSpace);
  }
  unsigned getInterleavedMemoryOpCost(unsigned Opcode, Type *VecTy,
                                      unsigned Factor,
                                      ArrayRef<unsigned> Indices,
                                      unsigned Alignment,
                                      unsigned AddressSpace) override {
    return Impl.getInterleavedMemoryOpCost(Opcode, VecTy, Factor, Indices,
                                           Alignment, AddressSpace);
  }
  unsigned getReductionCost(unsigned Opcode, Type *Ty,
                            bool IsPairwiseForm) override {
    return Impl.getReductionCost(Opcode, Ty, IsPairwiseForm);
  }
  unsigned getIntrinsicInstrCost(Intrinsic::ID ID, Type *RetTy,
                                 ArrayRef<Type *> Tys) override {
    return Impl.getIntrinsicInstrCost(ID, RetTy, Tys);
  }
  unsigned getCallInstrCost(Function *F, Type *RetTy,
                            ArrayRef<Type *> Tys) override {
    return Impl.getCallInstrCost(F, RetTy, Tys);
  }
  unsigned getNumberOfParts(Type *Tp) override {
    return Impl.getNumberOfParts(Tp);
  }
  unsigned getAddressComputationCost(Type *Ty, bool IsComplex) override {
    return Impl.getAddressComputationCost(Ty, IsComplex);
  }
  unsigned getCostOfKeepingLiveOverCall(ArrayRef<Type *> Tys) override {
    return Impl.getCostOfKeepingLiveOverCall(Tys);
  }
  bool getTgtMemIntrinsic(IntrinsicInst *Inst,
                          MemIntrinsicInfo &Info) override {
    return Impl.getTgtMemIntrinsic(Inst, Info);
  }
  Value *getOrCreateResultFromMemIntrinsic(IntrinsicInst *Inst,
                                           Type *ExpectedType) override {
    return Impl.getOrCreateResultFromMemIntrinsic(Inst, ExpectedType);
  }
  bool hasCompatibleFunctionAttributes(const Function *Caller,
                                       const Function *Callee) const override {
    return Impl.hasCompatibleFunctionAttributes(Caller, Callee);
  }
};

template <typename T>
TargetTransformInfo::TargetTransformInfo(T Impl)
    : TTIImpl(new Model<T>(Impl)) {}

/// \brief Analysis pass providing the \c TargetTransformInfo.
///
/// The core idea of the TargetIRAnalysis is to expose an interface through
/// which LLVM targets can analyze and provide information about the middle
/// end's target-independent IR. This supports use cases such as target-aware
/// cost modeling of IR constructs.
///
/// This is a function analysis because much of the cost modeling for targets
/// is done in a subtarget specific way and LLVM supports compiling different
/// functions targeting different subtargets in order to support runtime
/// dispatch according to the observed subtarget.
class TargetIRAnalysis {
public:
  typedef TargetTransformInfo Result;

  /// \brief Opaque, unique identifier for this analysis pass.
  static void *ID() { return (void *)&PassID; }

  /// \brief Provide access to a name for this pass for debugging purposes.
  static StringRef name() { return "TargetIRAnalysis"; }

  /// \brief Default construct a target IR analysis.
  ///
  /// This will use the module's datalayout to construct a baseline
  /// conservative TTI result.
  TargetIRAnalysis();

  /// \brief Construct an IR analysis pass around a target-provide callback.
  ///
  /// The callback will be called with a particular function for which the TTI
  /// is needed and must return a TTI object for that function.
  TargetIRAnalysis(std::function<Result(Function &)> TTICallback);

  // Value semantics. We spell out the constructors for MSVC.
  TargetIRAnalysis(const TargetIRAnalysis &Arg)
      : TTICallback(Arg.TTICallback) {}
  TargetIRAnalysis(TargetIRAnalysis &&Arg)
      : TTICallback(std::move(Arg.TTICallback)) {}
  TargetIRAnalysis &operator=(const TargetIRAnalysis &RHS) {
    TTICallback = RHS.TTICallback;
    return *this;
  }
  TargetIRAnalysis &operator=(TargetIRAnalysis &&RHS) {
    TTICallback = std::move(RHS.TTICallback);
    return *this;
  }

  Result run(Function &F);

private:
  static char PassID;

  /// \brief The callback used to produce a result.
  ///
  /// We use a completely opaque callback so that targets can provide whatever
  /// mechanism they desire for constructing the TTI for a given function.
  ///
  /// FIXME: Should we really use std::function? It's relatively inefficient.
  /// It might be possible to arrange for even stateful callbacks to outlive
  /// the analysis and thus use a function_ref which would be lighter weight.
  /// This may also be less error prone as the callback is likely to reference
  /// the external TargetMachine, and that reference needs to never dangle.
  std::function<Result(Function &)> TTICallback;

  /// \brief Helper function used as the callback in the default constructor.
  static Result getDefaultTTI(Function &F);
};

/// \brief Wrapper pass for TargetTransformInfo.
///
/// This pass can be constructed from a TTI object which it stores internally
/// and is queried by passes.
class TargetTransformInfoWrapperPass : public ImmutablePass {
  TargetIRAnalysis TIRA;
  Optional<TargetTransformInfo> TTI;

  virtual void anchor();

public:
  static char ID;

  /// \brief We must provide a default constructor for the pass but it should
  /// never be used.
  ///
  /// Use the constructor below or call one of the creation routines.
  TargetTransformInfoWrapperPass();

  explicit TargetTransformInfoWrapperPass(TargetIRAnalysis TIRA);

  TargetTransformInfo &getTTI(Function &F);
};

/// \brief Create an analysis pass wrapper around a TTI object.
///
/// This analysis pass just holds the TTI instance and makes it available to
/// clients.
ImmutablePass *createTargetTransformInfoWrapperPass(TargetIRAnalysis TIRA);

} // End llvm namespace

#endif
