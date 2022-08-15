//===-- llvm/CodeGen/SDNodeDbgValue.h - SelectionDAG dbg_value --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the SDDbgValue class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_SELECTIONDAG_SDNODEDBGVALUE_H
#define LLVM_LIB_CODEGEN_SELECTIONDAG_SDNODEDBGVALUE_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {

class MDNode;
class SDNode;
class Value;

/// SDDbgValue - Holds the information from a dbg_value node through SDISel.
/// We do not use SDValue here to avoid including its header.

class SDDbgValue {
public:
  enum DbgValueKind {
    SDNODE = 0,             // value is the result of an expression
    CONST = 1,              // value is a constant
    FRAMEIX = 2             // value is contents of a stack location
  };
private:
  union {
    struct {
      SDNode *Node;         // valid for expressions
      unsigned ResNo;       // valid for expressions
    } s;
    const Value *Const;     // valid for constants
    unsigned FrameIx;       // valid for stack objects
  } u;
  MDNode *Var;
  MDNode *Expr;
  uint64_t Offset;
  DebugLoc DL;
  unsigned Order;
  enum DbgValueKind kind;
  bool IsIndirect;
  bool Invalid = false;

public:
  // Constructor for non-constants.
  SDDbgValue(MDNode *Var, MDNode *Expr, SDNode *N, unsigned R, bool indir,
             uint64_t off, DebugLoc dl, unsigned O)
      : Var(Var), Expr(Expr), Offset(off), DL(dl), Order(O), IsIndirect(indir) {
    kind = SDNODE;
    u.s.Node = N;
    u.s.ResNo = R;
  }

  // Constructor for constants.
  SDDbgValue(MDNode *Var, MDNode *Expr, const Value *C, uint64_t off,
             DebugLoc dl, unsigned O)
      : Var(Var), Expr(Expr), Offset(off), DL(dl), Order(O), IsIndirect(false) {
    kind = CONST;
    u.Const = C;
  }

  // Constructor for frame indices.
  SDDbgValue(MDNode *Var, MDNode *Expr, unsigned FI, uint64_t off, DebugLoc dl,
             unsigned O)
      : Var(Var), Expr(Expr), Offset(off), DL(dl), Order(O), IsIndirect(false) {
    kind = FRAMEIX;
    u.FrameIx = FI;
  }

  // Returns the kind.
  DbgValueKind getKind() const { return kind; }

  // Returns the MDNode pointer for the variable.
  MDNode *getVariable() const { return Var; }

  // Returns the MDNode pointer for the expression.
  MDNode *getExpression() const { return Expr; }

  // Returns the SDNode* for a register ref
  SDNode *getSDNode() const { assert (kind==SDNODE); return u.s.Node; }

  // Returns the ResNo for a register ref
  unsigned getResNo() const { assert (kind==SDNODE); return u.s.ResNo; }

  // Returns the Value* for a constant
  const Value *getConst() const { assert (kind==CONST); return u.Const; }

  // Returns the FrameIx for a stack object
  unsigned getFrameIx() const { assert (kind==FRAMEIX); return u.FrameIx; }

  // Returns whether this is an indirect value.
  bool isIndirect() const { return IsIndirect; }

  // Returns the offset.
  uint64_t getOffset() const { return Offset; }

  // Returns the DebugLoc.
  DebugLoc getDebugLoc() const { return DL; }

  // Returns the SDNodeOrder.  This is the order of the preceding node in the
  // input.
  unsigned getOrder() const { return Order; }

  // setIsInvalidated / isInvalidated - Setter / getter of the "Invalidated"
  // property. A SDDbgValue is invalid if the SDNode that produces the value is
  // deleted.
  void setIsInvalidated() { Invalid = true; }
  bool isInvalidated() const { return Invalid; }
};

} // end llvm namespace

#endif
