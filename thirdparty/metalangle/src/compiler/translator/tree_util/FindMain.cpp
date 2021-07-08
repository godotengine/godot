//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// FindMain.cpp: Find the main() function definition in a given AST.

#include "compiler/translator/tree_util/FindMain.h"

#include "compiler/translator/IntermNode.h"
#include "compiler/translator/Symbol.h"

namespace sh
{

size_t FindMainIndex(TIntermBlock *root)
{
    const TIntermSequence &sequence = *root->getSequence();
    for (size_t index = 0; index < sequence.size(); ++index)
    {
        TIntermNode *node                       = sequence[index];
        TIntermFunctionDefinition *nodeFunction = node->getAsFunctionDefinition();
        if (nodeFunction != nullptr && nodeFunction->getFunction()->isMain())
        {
            return index;
        }
    }
    return std::numeric_limits<size_t>::max();
}

TIntermFunctionDefinition *FindMain(TIntermBlock *root)
{
    for (TIntermNode *node : *root->getSequence())
    {
        TIntermFunctionDefinition *nodeFunction = node->getAsFunctionDefinition();
        if (nodeFunction != nullptr && nodeFunction->getFunction()->isMain())
        {
            return nodeFunction;
        }
    }
    return nullptr;
}

TIntermBlock *FindMainBody(TIntermBlock *root)
{
    TIntermFunctionDefinition *main = FindMain(root);
    ASSERT(main != nullptr);
    TIntermBlock *mainBody = main->getBody();
    ASSERT(mainBody != nullptr);
    return mainBody;
}

}  // namespace sh
