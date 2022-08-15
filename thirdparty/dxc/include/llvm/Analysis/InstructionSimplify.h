//===-- InstructionSimplify.h - Fold instrs into simpler forms --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares routines for folding instructions into simpler forms
// that do not require creating new instructions.  This does constant folding
// ("add i32 1, 1" -> "2") but can also handle non-constant operands, either
// returning a constant ("and i32 %x, 0" -> "0") or an already existing value
// ("and i32 %x, %x" -> "%x").  If the simplification is also an instruction
// then it dominates the original instruction.
//
// These routines implicitly resolve undef uses. The easiest way to be safe when
// using these routines to obtain simplified values for existing instructions is
// to always replace all uses of the instructions with the resulting simplified
// values. This will prevent other code from seeing the same undef uses and
// resolving them to different values.
//
// These routines are designed to tolerate moderately incomplete IR, such as
// instructions that are not connected to basic blocks yet. However, they do
// require that all the IR that they encounter be valid. In particular, they
// require that all non-constant values be defined in the same function, and the
// same call context of that function (and not split between caller and callee
// contexts of a directly recursive call, for example).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_INSTRUCTIONSIMPLIFY_H
#define LLVM_ANALYSIS_INSTRUCTIONSIMPLIFY_H

#include "llvm/IR/User.h"

namespace llvm {
  template<typename T>
  class ArrayRef;
  class AssumptionCache;
  class DominatorTree;
  class Instruction;
  class DataLayout;
  class FastMathFlags;
  class TargetLibraryInfo;
  class Type;
  class Value;

  /// SimplifyAddInst - Given operands for an Add, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyAddInst(Value *LHS, Value *RHS, bool isNSW, bool isNUW,
                         const DataLayout &DL,
                         const TargetLibraryInfo *TLI = nullptr,
                         const DominatorTree *DT = nullptr,
                         AssumptionCache *AC = nullptr,
                         const Instruction *CxtI = nullptr);

  /// SimplifySubInst - Given operands for a Sub, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifySubInst(Value *LHS, Value *RHS, bool isNSW, bool isNUW,
                         const DataLayout &DL,
                         const TargetLibraryInfo *TLI = nullptr,
                         const DominatorTree *DT = nullptr,
                         AssumptionCache *AC = nullptr,
                         const Instruction *CxtI = nullptr);

  /// Given operands for an FAdd, see if we can fold the result.  If not, this
  /// returns null.
  Value *SimplifyFAddInst(Value *LHS, Value *RHS, FastMathFlags FMF,
                          const DataLayout &DL,
                          const TargetLibraryInfo *TLI = nullptr,
                          const DominatorTree *DT = nullptr,
                          AssumptionCache *AC = nullptr,
                          const Instruction *CxtI = nullptr);

  /// Given operands for an FSub, see if we can fold the result.  If not, this
  /// returns null.
  Value *SimplifyFSubInst(Value *LHS, Value *RHS, FastMathFlags FMF,
                          const DataLayout &DL,
                          const TargetLibraryInfo *TLI = nullptr,
                          const DominatorTree *DT = nullptr,
                          AssumptionCache *AC = nullptr,
                          const Instruction *CxtI = nullptr);

  /// Given operands for an FMul, see if we can fold the result.  If not, this
  /// returns null.
  Value *SimplifyFMulInst(Value *LHS, Value *RHS, FastMathFlags FMF,
                          const DataLayout &DL,
                          const TargetLibraryInfo *TLI = nullptr,
                          const DominatorTree *DT = nullptr,
                          AssumptionCache *AC = nullptr,
                          const Instruction *CxtI = nullptr);

  /// SimplifyMulInst - Given operands for a Mul, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyMulInst(Value *LHS, Value *RHS, const DataLayout &DL,
                         const TargetLibraryInfo *TLI = nullptr,
                         const DominatorTree *DT = nullptr,
                         AssumptionCache *AC = nullptr,
                         const Instruction *CxtI = nullptr);

  /// SimplifySDivInst - Given operands for an SDiv, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifySDivInst(Value *LHS, Value *RHS, const DataLayout &DL,
                          const TargetLibraryInfo *TLI = nullptr,
                          const DominatorTree *DT = nullptr,
                          AssumptionCache *AC = nullptr,
                          const Instruction *CxtI = nullptr);

  /// SimplifyUDivInst - Given operands for a UDiv, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyUDivInst(Value *LHS, Value *RHS, const DataLayout &DL,
                          const TargetLibraryInfo *TLI = nullptr,
                          const DominatorTree *DT = nullptr,
                          AssumptionCache *AC = nullptr,
                          const Instruction *CxtI = nullptr);

  /// SimplifyFDivInst - Given operands for an FDiv, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyFDivInst(Value *LHS, Value *RHS, FastMathFlags FMF,
                          const DataLayout &DL,
                          const TargetLibraryInfo *TLI = nullptr,
                          const DominatorTree *DT = nullptr,
                          AssumptionCache *AC = nullptr,
                          const Instruction *CxtI = nullptr);

  /// SimplifySRemInst - Given operands for an SRem, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifySRemInst(Value *LHS, Value *RHS, const DataLayout &DL,
                          const TargetLibraryInfo *TLI = nullptr,
                          const DominatorTree *DT = nullptr,
                          AssumptionCache *AC = nullptr,
                          const Instruction *CxtI = nullptr);

  /// SimplifyURemInst - Given operands for a URem, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyURemInst(Value *LHS, Value *RHS, const DataLayout &DL,
                          const TargetLibraryInfo *TLI = nullptr,
                          const DominatorTree *DT = nullptr,
                          AssumptionCache *AC = nullptr,
                          const Instruction *CxtI = nullptr);

  /// SimplifyFRemInst - Given operands for an FRem, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyFRemInst(Value *LHS, Value *RHS, FastMathFlags FMF,
                          const DataLayout &DL,
                          const TargetLibraryInfo *TLI = nullptr,
                          const DominatorTree *DT = nullptr,
                          AssumptionCache *AC = nullptr,
                          const Instruction *CxtI = nullptr);

  /// SimplifyShlInst - Given operands for a Shl, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyShlInst(Value *Op0, Value *Op1, bool isNSW, bool isNUW,
                         const DataLayout &DL,
                         const TargetLibraryInfo *TLI = nullptr,
                         const DominatorTree *DT = nullptr,
                         AssumptionCache *AC = nullptr,
                         const Instruction *CxtI = nullptr);

  /// SimplifyLShrInst - Given operands for a LShr, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyLShrInst(Value *Op0, Value *Op1, bool isExact,
                          const DataLayout &DL,
                          const TargetLibraryInfo *TLI = nullptr,
                          const DominatorTree *DT = nullptr,
                          AssumptionCache *AC = nullptr,
                          const Instruction *CxtI = nullptr);

  /// SimplifyAShrInst - Given operands for a AShr, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyAShrInst(Value *Op0, Value *Op1, bool isExact,
                          const DataLayout &DL,
                          const TargetLibraryInfo *TLI = nullptr,
                          const DominatorTree *DT = nullptr,
                          AssumptionCache *AC = nullptr,
                          const Instruction *CxtI = nullptr);

  /// SimplifyAndInst - Given operands for an And, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyAndInst(Value *LHS, Value *RHS, const DataLayout &DL,
                         const TargetLibraryInfo *TLI = nullptr,
                         const DominatorTree *DT = nullptr,
                         AssumptionCache *AC = nullptr,
                         const Instruction *CxtI = nullptr);

  /// SimplifyOrInst - Given operands for an Or, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyOrInst(Value *LHS, Value *RHS, const DataLayout &DL,
                        const TargetLibraryInfo *TLI = nullptr,
                        const DominatorTree *DT = nullptr,
                        AssumptionCache *AC = nullptr,
                        const Instruction *CxtI = nullptr);

  /// SimplifyXorInst - Given operands for a Xor, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyXorInst(Value *LHS, Value *RHS, const DataLayout &DL,
                         const TargetLibraryInfo *TLI = nullptr,
                         const DominatorTree *DT = nullptr,
                         AssumptionCache *AC = nullptr,
                         const Instruction *CxtI = nullptr);

  /// SimplifyICmpInst - Given operands for an ICmpInst, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyICmpInst(unsigned Predicate, Value *LHS, Value *RHS,
                          const DataLayout &DL,
                          const TargetLibraryInfo *TLI = nullptr,
                          const DominatorTree *DT = nullptr,
                          AssumptionCache *AC = nullptr,
                          Instruction *CxtI = nullptr);

  /// SimplifyFCmpInst - Given operands for an FCmpInst, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyFCmpInst(unsigned Predicate, Value *LHS, Value *RHS,
                          FastMathFlags FMF, const DataLayout &DL,
                          const TargetLibraryInfo *TLI = nullptr,
                          const DominatorTree *DT = nullptr,
                          AssumptionCache *AC = nullptr,
                          const Instruction *CxtI = nullptr);

  /// SimplifySelectInst - Given operands for a SelectInst, see if we can fold
  /// the result.  If not, this returns null.
  Value *SimplifySelectInst(Value *Cond, Value *TrueVal, Value *FalseVal,
                            const DataLayout &DL,
                            const TargetLibraryInfo *TLI = nullptr,
                            const DominatorTree *DT = nullptr,
                            AssumptionCache *AC = nullptr,
                            const Instruction *CxtI = nullptr);

  /// SimplifyGEPInst - Given operands for an GetElementPtrInst, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyGEPInst(ArrayRef<Value *> Ops, const DataLayout &DL,
                         const TargetLibraryInfo *TLI = nullptr,
                         const DominatorTree *DT = nullptr,
                         AssumptionCache *AC = nullptr,
                         const Instruction *CxtI = nullptr);

  /// SimplifyInsertValueInst - Given operands for an InsertValueInst, see if we
  /// can fold the result.  If not, this returns null.
  Value *SimplifyInsertValueInst(Value *Agg, Value *Val,
                                 ArrayRef<unsigned> Idxs, const DataLayout &DL,
                                 const TargetLibraryInfo *TLI = nullptr,
                                 const DominatorTree *DT = nullptr,
                                 AssumptionCache *AC = nullptr,
                                 const Instruction *CxtI = nullptr);

  /// \brief Given operands for an ExtractValueInst, see if we can fold the
  /// result.  If not, this returns null.
  Value *SimplifyExtractValueInst(Value *Agg, ArrayRef<unsigned> Idxs,
                                  const DataLayout &DL,
                                  const TargetLibraryInfo *TLI = nullptr,
                                  const DominatorTree *DT = nullptr,
                                  AssumptionCache *AC = nullptr,
                                  const Instruction *CxtI = nullptr);

  /// \brief Given operands for an ExtractElementInst, see if we can fold the
  /// result.  If not, this returns null.
  Value *SimplifyExtractElementInst(Value *Vec, Value *Idx,
                                    const DataLayout &DL,
                                    const TargetLibraryInfo *TLI = nullptr,
                                    const DominatorTree *DT = nullptr,
                                    AssumptionCache *AC = nullptr,
                                    const Instruction *CxtI = nullptr);

  /// SimplifyTruncInst - Given operands for an TruncInst, see if we can fold
  /// the result.  If not, this returns null.
  Value *SimplifyTruncInst(Value *Op, Type *Ty, const DataLayout &DL,
                           const TargetLibraryInfo *TLI = nullptr,
                           const DominatorTree *DT = nullptr,
                           AssumptionCache *AC = nullptr,
                           const Instruction *CxtI = nullptr);

  //=== Helper functions for higher up the class hierarchy.


  /// SimplifyCmpInst - Given operands for a CmpInst, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyCmpInst(unsigned Predicate, Value *LHS, Value *RHS,
                         const DataLayout &DL,
                         const TargetLibraryInfo *TLI = nullptr,
                         const DominatorTree *DT = nullptr,
                         AssumptionCache *AC = nullptr,
                         const Instruction *CxtI = nullptr);

  /// SimplifyBinOp - Given operands for a BinaryOperator, see if we can
  /// fold the result.  If not, this returns null.
  Value *SimplifyBinOp(unsigned Opcode, Value *LHS, Value *RHS,
                       const DataLayout &DL,
                       const TargetLibraryInfo *TLI = nullptr,
                       const DominatorTree *DT = nullptr,
                       AssumptionCache *AC = nullptr,
                       const Instruction *CxtI = nullptr);
  /// SimplifyFPBinOp - Given operands for a BinaryOperator, see if we can
  /// fold the result.  If not, this returns null.
  /// In contrast to SimplifyBinOp, try to use FastMathFlag when folding the
  /// result. In case we don't need FastMathFlags, simply fall to SimplifyBinOp.
  Value *SimplifyFPBinOp(unsigned Opcode, Value *LHS, Value *RHS,
                         const FastMathFlags &FMF, const DataLayout &DL,
                         const TargetLibraryInfo *TLI = nullptr,
                         const DominatorTree *DT = nullptr,
                         AssumptionCache *AC = nullptr,
                         const Instruction *CxtI = nullptr);

  /// \brief Given a function and iterators over arguments, see if we can fold
  /// the result.
  ///
  /// If this call could not be simplified returns null.
  Value *SimplifyCall(Value *V, User::op_iterator ArgBegin,
                      User::op_iterator ArgEnd, const DataLayout &DL,
                      const TargetLibraryInfo *TLI = nullptr,
                      const DominatorTree *DT = nullptr,
                      AssumptionCache *AC = nullptr,
                      const Instruction *CxtI = nullptr);

  /// \brief Given a function and set of arguments, see if we can fold the
  /// result.
  ///
  /// If this call could not be simplified returns null.
  Value *SimplifyCall(Value *V, ArrayRef<Value *> Args, const DataLayout &DL,
                      const TargetLibraryInfo *TLI = nullptr,
                      const DominatorTree *DT = nullptr,
                      AssumptionCache *AC = nullptr,
                      const Instruction *CxtI = nullptr);

// HLSL Change - Begin
  Value *SimplifyCastInst(unsigned CastOpc, Value *Op,
                          Type *Ty, const DataLayout &DL);
// HLSL Change - End

  /// SimplifyInstruction - See if we can compute a simplified version of this
  /// instruction.  If not, this returns null.
  Value *SimplifyInstruction(Instruction *I, const DataLayout &DL,
                             const TargetLibraryInfo *TLI = nullptr,
                             const DominatorTree *DT = nullptr,
                             AssumptionCache *AC = nullptr);

  /// \brief Replace all uses of 'I' with 'SimpleV' and simplify the uses
  /// recursively.
  ///
  /// This first performs a normal RAUW of I with SimpleV. It then recursively
  /// attempts to simplify those users updated by the operation. The 'I'
  /// instruction must not be equal to the simplified value 'SimpleV'.
  ///
  /// The function returns true if any simplifications were performed.
  bool replaceAndRecursivelySimplify(Instruction *I, Value *SimpleV,
                                     const TargetLibraryInfo *TLI = nullptr,
                                     const DominatorTree *DT = nullptr,
                                     AssumptionCache *AC = nullptr);

  /// \brief Recursively attempt to simplify an instruction.
  ///
  /// This routine uses SimplifyInstruction to simplify 'I', and if successful
  /// replaces uses of 'I' with the simplified value. It then recurses on each
  /// of the users impacted. It returns true if any simplifications were
  /// performed.
  bool recursivelySimplifyInstruction(Instruction *I,
                                      const TargetLibraryInfo *TLI = nullptr,
                                      const DominatorTree *DT = nullptr,
                                      AssumptionCache *AC = nullptr);
} // end namespace llvm

#endif

