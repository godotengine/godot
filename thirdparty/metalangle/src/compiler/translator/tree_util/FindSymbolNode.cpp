//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// FindSymbol.cpp:
//     Utility for finding a symbol node inside an AST tree.

#include "compiler/translator/tree_util/FindSymbolNode.h"

#include "compiler/translator/ImmutableString.h"
#include "compiler/translator/Symbol.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

class SymbolFinder : public TIntermTraverser
{
  public:
    SymbolFinder(const ImmutableString &symbolName)
        : TIntermTraverser(true, false, false), mSymbolName(symbolName), mNodeFound(nullptr)
    {}

    void visitSymbol(TIntermSymbol *node)
    {
        if (node->variable().symbolType() != SymbolType::Empty && node->getName() == mSymbolName)
        {
            mNodeFound = node;
        }
    }

    bool isFound() const { return mNodeFound != nullptr; }
    const TIntermSymbol *getNode() const { return mNodeFound; }

  private:
    ImmutableString mSymbolName;
    TIntermSymbol *mNodeFound;
};

}  // anonymous namespace

const TIntermSymbol *FindSymbolNode(TIntermNode *root, const ImmutableString &symbolName)
{
    SymbolFinder finder(symbolName);
    root->traverse(&finder);
    return finder.getNode();
}

}  // namespace sh
