//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// FindMain.h: Adds functions to get the main function definition and its body.

#ifndef COMPILER_TRANSLATOR_TREEUTIL_FINDMAIN_H_
#define COMPILER_TRANSLATOR_TREEUTIL_FINDMAIN_H_

#include <cstddef>

namespace sh
{
class TIntermBlock;
class TIntermFunctionDefinition;

size_t FindMainIndex(TIntermBlock *root);
TIntermFunctionDefinition *FindMain(TIntermBlock *root);
TIntermBlock *FindMainBody(TIntermBlock *root);
}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEUTIL_FINDMAIN_H_
