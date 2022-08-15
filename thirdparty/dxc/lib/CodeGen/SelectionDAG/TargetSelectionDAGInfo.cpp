//===-- TargetSelectionDAGInfo.cpp - SelectionDAG Info --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the TargetSelectionDAGInfo class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetSelectionDAGInfo.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;

TargetSelectionDAGInfo::~TargetSelectionDAGInfo() {
}
