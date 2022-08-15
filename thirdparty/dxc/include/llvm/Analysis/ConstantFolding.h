//===-- ConstantFolding.h - Fold instructions into constants ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares routines for folding instructions into constants when all
// operands are constants, for example "sub i32 1, 0" -> "1".
//
// Also, to supplement the basic VMCore ConstantExpr simplifications,
// this file declares some additional folding routines that can make use of
// DataLayout information. These functions cannot go in VMCore due to library
// dependency issues.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CONSTANTFOLDING_H
#define LLVM_ANALYSIS_CONSTANTFOLDING_H

#include "dxc/Support/WinAdapter.h" // HLSL Change

namespace llvm {
  class Constant;
  class ConstantFP;
  class ConstantExpr;
  class Instruction;
  class DataLayout;
  class TargetLibraryInfo;
  class Function;
  class Type;
  template<typename T>
  class ArrayRef;

/// ConstantFoldInstruction - Try to constant fold the specified instruction.
/// If successful, the constant result is returned, if not, null is returned.
/// Note that this fails if not all of the operands are constant.  Otherwise,
/// this function can only fail when attempting to fold instructions like loads
/// and stores, which have no constant expression form.
  Constant *ConstantFoldInstruction(Instruction *I, const DataLayout &DL,
                                    const TargetLibraryInfo *TLI = nullptr);

/// ConstantFoldConstantExpression - Attempt to fold the constant expression
/// using the specified DataLayout.  If successful, the constant result is
/// result is returned, if not, null is returned.
  Constant *
  ConstantFoldConstantExpression(const ConstantExpr *CE, const DataLayout &DL,
                                 const TargetLibraryInfo *TLI = nullptr);

/// ConstantFoldInstOperands - Attempt to constant fold an instruction with the
/// specified operands.  If successful, the constant result is returned, if not,
/// null is returned.  Note that this function can fail when attempting to
/// fold instructions like loads and stores, which have no constant expression
/// form.
///
  Constant *ConstantFoldInstOperands(unsigned Opcode, Type *DestTy,
                                     ArrayRef<Constant *> Ops,
                                     const DataLayout &DL,
                                     const TargetLibraryInfo *TLI = nullptr);

/// ConstantFoldCompareInstOperands - Attempt to constant fold a compare
/// instruction (icmp/fcmp) with the specified operands.  If it fails, it
/// returns a constant expression of the specified operands.
///
  Constant *
  ConstantFoldCompareInstOperands(unsigned Predicate, Constant *LHS,
                                  Constant *RHS, const DataLayout &DL,
                                  const TargetLibraryInfo *TLI = nullptr);

/// ConstantFoldInsertValueInstruction - Attempt to constant fold an insertvalue
/// instruction with the specified operands and indices.  The constant result is
/// returned if successful; if not, null is returned.
Constant *ConstantFoldInsertValueInstruction(Constant *Agg, Constant *Val,
                                             ArrayRef<unsigned> Idxs);

/// \brief Attempt to constant fold an extractvalue instruction with the
/// specified operands and indices.  The constant result is returned if
/// successful; if not, null is returned.
Constant *ConstantFoldExtractValueInstruction(Constant *Agg,
                                              ArrayRef<unsigned> Idxs);

/// \brief Attempt to constant fold an extractelement instruction with the
/// specified operands and indices.  The constant result is returned if
/// successful; if not, null is returned.
Constant *ConstantFoldExtractElementInstruction(Constant *Val, Constant *Idx);

/// ConstantFoldLoadFromConstPtr - Return the value that a load from C would
/// produce if it is constant and determinable.  If this is not determinable,
/// return null.
Constant *ConstantFoldLoadFromConstPtr(Constant *C, const DataLayout &DL);

/// ConstantFoldLoadThroughGEPConstantExpr - Given a constant and a
/// getelementptr constantexpr, return the constant value being addressed by the
/// constant expression, or null if something is funny and we can't decide.
Constant *ConstantFoldLoadThroughGEPConstantExpr(Constant *C, ConstantExpr *CE);

/// ConstantFoldLoadThroughGEPIndices - Given a constant and getelementptr
/// indices (with an *implied* zero pointer index that is not in the list),
/// return the constant value being addressed by a virtual load, or null if
/// something is funny and we can't decide.
Constant *ConstantFoldLoadThroughGEPIndices(Constant *C,
                                            ArrayRef<Constant*> Indices);

/// canConstantFoldCallTo - Return true if its even possible to fold a call to
/// the specified function.
bool canConstantFoldCallTo(const Function *F);

/// ConstantFoldCall - Attempt to constant fold a call to the specified function
/// with the specified arguments, returning null if unsuccessful.
Constant *ConstantFoldCall(Function *F, ArrayRef<Constant *> Operands,
                           const TargetLibraryInfo *TLI = nullptr);

/// HLSL Change - make these functions external so we can call them from 
/// DxilConstantFolding.cpp.
Constant *ConstantFoldFP(double(__cdecl *NativeFP)(double), double V, Type *Ty);
double getValueAsDouble(ConstantFP *Op);
}

#endif
