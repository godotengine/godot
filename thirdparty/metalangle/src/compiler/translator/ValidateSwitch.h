//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_TRANSLATOR_VALIDATESWITCH_H_
#define COMPILER_TRANSLATOR_VALIDATESWITCH_H_

#include "compiler/translator/BaseTypes.h"
#include "compiler/translator/Common.h"

namespace sh
{
class TDiagnostics;
class TIntermBlock;

// Check for errors and output error messages on the context.
// Returns true if there are no errors.
bool ValidateSwitchStatementList(TBasicType switchType,
                                 TDiagnostics *diagnostics,
                                 TIntermBlock *statementList,
                                 const TSourceLoc &loc);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_VALIDATESWITCH_H_
