//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// FindFunction.h: Adds functions to find functions

#ifndef COMPILER_TRANSLATOR_TREEUTIL_FINDFUNCTION_H_
#define COMPILER_TRANSLATOR_TREEUTIL_FINDFUNCTION_H_

#include <cstddef>

namespace sh
{
class TIntermBlock;

size_t FindFirstFunctionDefinitionIndex(TIntermBlock *root);
}  // namespace sh

#endif  // COMPILER_TRANSLATOR_TREEUTIL_FINDFUNCTION_H_