//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// RemoveSwitchFallThrough.h: Remove fall-through from switch statements.
// Note that it is unsafe to do further AST transformations on the AST generated
// by this function. It leaves duplicate nodes in the AST making replacements
// unreliable.

#ifndef COMPILER_TRANSLATOR_TREEOPS_REMOVESWITCHFALLTHROUGH_H_
#define COMPILER_TRANSLATOR_TREEOPS_REMOVESWITCHFALLTHROUGH_H_

namespace sh
{

class TIntermBlock;
class PerformanceDiagnostics;

// When given a statementList from a switch AST node, return an updated
// statementList that has fall-through removed.
TIntermBlock *RemoveSwitchFallThrough(TIntermBlock *statementList,
                                      PerformanceDiagnostics *perfDiagnostics);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEOPS_REMOVESWITCHFALLTHROUGH_H_
