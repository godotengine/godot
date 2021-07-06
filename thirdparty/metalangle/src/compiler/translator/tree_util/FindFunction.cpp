//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// FindFunction.cpp: Find functions.

#include "compiler/translator/tree_util/FindFunction.h"

#include "compiler/translator/IntermNode.h"
#include "compiler/translator/Symbol.h"

namespace sh
{

size_t FindFirstFunctionDefinitionIndex(TIntermBlock *root)
{
    const TIntermSequence &sequence = *root->getSequence();
    for (size_t index = 0; index < sequence.size(); ++index)
    {
        TIntermNode *node                       = sequence[index];
        TIntermFunctionDefinition *nodeFunction = node->getAsFunctionDefinition();
        if (nodeFunction != nullptr)
        {
            return index;
        }
    }
    return std::numeric_limits<size_t>::max();
}

}  // namespace sh
