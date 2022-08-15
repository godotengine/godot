//==-- llvm/Target/TargetSelectionDAGInfo.h - SelectionDAG Info --*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the TargetSelectionDAGInfo class, which targets can
// subclass to parameterize the SelectionDAG lowering and instruction
// selection process.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETSELECTIONDAGINFO_H
#define LLVM_TARGET_TARGETSELECTIONDAGINFO_H

#include "llvm/CodeGen/SelectionDAGNodes.h"

namespace llvm {
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
/// TargetSelectionDAGInfo - Targets can subclass this to parameterize the
/// SelectionDAG lowering and instruction selection process.
///
class TargetSelectionDAGInfo {
  TargetSelectionDAGInfo(const TargetSelectionDAGInfo &) = delete;
  void operator=(const TargetSelectionDAGInfo &) = delete;

public:
  explicit TargetSelectionDAGInfo() = default;
  virtual ~TargetSelectionDAGInfo();

  /// EmitTargetCodeForMemcpy - Emit target-specific code that performs a
  /// memcpy. This can be used by targets to provide code sequences for cases
  /// that don't fit the target's parameters for simple loads/stores and can be
  /// more efficient than using a library call. This function can return a null
  /// SDValue if the target declines to use custom code and a different
  /// lowering strategy should be used.
  ///
  /// If AlwaysInline is true, the size is constant and the target should not
  /// emit any calls and is strongly encouraged to attempt to emit inline code
  /// even if it is beyond the usual threshold because this intrinsic is being
  /// expanded in a place where calls are not feasible (e.g. within the prologue
  /// for another call). If the target chooses to decline an AlwaysInline
  /// request here, legalize will resort to using simple loads and stores.
  virtual SDValue
  EmitTargetCodeForMemcpy(SelectionDAG &DAG, SDLoc dl,
                          SDValue Chain,
                          SDValue Op1, SDValue Op2,
                          SDValue Op3, unsigned Align, bool isVolatile,
                          bool AlwaysInline,
                          MachinePointerInfo DstPtrInfo,
                          MachinePointerInfo SrcPtrInfo) const {
    return SDValue();
  }

  /// EmitTargetCodeForMemmove - Emit target-specific code that performs a
  /// memmove. This can be used by targets to provide code sequences for cases
  /// that don't fit the target's parameters for simple loads/stores and can be
  /// more efficient than using a library call. This function can return a null
  /// SDValue if the target declines to use custom code and a different
  /// lowering strategy should be used.
  virtual SDValue
  EmitTargetCodeForMemmove(SelectionDAG &DAG, SDLoc dl,
                           SDValue Chain,
                           SDValue Op1, SDValue Op2,
                           SDValue Op3, unsigned Align, bool isVolatile,
                           MachinePointerInfo DstPtrInfo,
                           MachinePointerInfo SrcPtrInfo) const {
    return SDValue();
  }

  /// EmitTargetCodeForMemset - Emit target-specific code that performs a
  /// memset. This can be used by targets to provide code sequences for cases
  /// that don't fit the target's parameters for simple stores and can be more
  /// efficient than using a library call. This function can return a null
  /// SDValue if the target declines to use custom code and a different
  /// lowering strategy should be used.
  virtual SDValue
  EmitTargetCodeForMemset(SelectionDAG &DAG, SDLoc dl,
                          SDValue Chain,
                          SDValue Op1, SDValue Op2,
                          SDValue Op3, unsigned Align, bool isVolatile,
                          MachinePointerInfo DstPtrInfo) const {
    return SDValue();
  }

  /// EmitTargetCodeForMemcmp - Emit target-specific code that performs a
  /// memcmp, in cases where that is faster than a libcall.  The first
  /// returned SDValue is the result of the memcmp and the second is
  /// the chain.  Both SDValues can be null if a normal libcall should
  /// be used.
  virtual std::pair<SDValue, SDValue>
  EmitTargetCodeForMemcmp(SelectionDAG &DAG, SDLoc dl,
                          SDValue Chain,
                          SDValue Op1, SDValue Op2,
                          SDValue Op3, MachinePointerInfo Op1PtrInfo,
                          MachinePointerInfo Op2PtrInfo) const {
    return std::make_pair(SDValue(), SDValue());
  }

  /// EmitTargetCodeForMemchr - Emit target-specific code that performs a
  /// memchr, in cases where that is faster than a libcall.  The first
  /// returned SDValue is the result of the memchr and the second is
  /// the chain.  Both SDValues can be null if a normal libcall should
  /// be used.
  virtual std::pair<SDValue, SDValue>
  EmitTargetCodeForMemchr(SelectionDAG &DAG, SDLoc dl, SDValue Chain,
                          SDValue Src, SDValue Char, SDValue Length,
                          MachinePointerInfo SrcPtrInfo) const {
    return std::make_pair(SDValue(), SDValue());
  }

  /// EmitTargetCodeForStrcpy - Emit target-specific code that performs a
  /// strcpy or stpcpy, in cases where that is faster than a libcall.
  /// The first returned SDValue is the result of the copy (the start
  /// of the destination string for strcpy, a pointer to the null terminator
  /// for stpcpy) and the second is the chain.  Both SDValues can be null
  /// if a normal libcall should be used.
  virtual std::pair<SDValue, SDValue>
  EmitTargetCodeForStrcpy(SelectionDAG &DAG, SDLoc DL, SDValue Chain,
                          SDValue Dest, SDValue Src,
                          MachinePointerInfo DestPtrInfo,
                          MachinePointerInfo SrcPtrInfo,
                          bool isStpcpy) const {
    return std::make_pair(SDValue(), SDValue());
  }

  /// EmitTargetCodeForStrcmp - Emit target-specific code that performs a
  /// strcmp, in cases where that is faster than a libcall.  The first
  /// returned SDValue is the result of the strcmp and the second is
  /// the chain.  Both SDValues can be null if a normal libcall should
  /// be used.
  virtual std::pair<SDValue, SDValue>
  EmitTargetCodeForStrcmp(SelectionDAG &DAG, SDLoc dl,
                          SDValue Chain,
                          SDValue Op1, SDValue Op2,
                          MachinePointerInfo Op1PtrInfo,
                          MachinePointerInfo Op2PtrInfo) const {
    return std::make_pair(SDValue(), SDValue());
  }

  virtual std::pair<SDValue, SDValue>
  EmitTargetCodeForStrlen(SelectionDAG &DAG, SDLoc DL, SDValue Chain,
                          SDValue Src, MachinePointerInfo SrcPtrInfo) const {
    return std::make_pair(SDValue(), SDValue());
  }

  virtual std::pair<SDValue, SDValue>
  EmitTargetCodeForStrnlen(SelectionDAG &DAG, SDLoc DL, SDValue Chain,
                           SDValue Src, SDValue MaxLength,
                           MachinePointerInfo SrcPtrInfo) const {
    return std::make_pair(SDValue(), SDValue());
  }
};

} // end llvm namespace

#endif
