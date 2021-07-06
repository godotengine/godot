//
// Copyright 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// FlagStd140Structs.cpp: Find structs in std140 blocks, where the padding added in the translator
// conflicts with the "natural" unpadded type.

#include "compiler/translator/FlagStd140Structs.h"

#include "compiler/translator/SymbolTable.h"
#include "compiler/translator/tree_util/IntermTraverse.h"

namespace sh
{

namespace
{

class FlagStd140StructsTraverser : public TIntermTraverser
{
  public:
    FlagStd140StructsTraverser() : TIntermTraverser(true, false, false) {}

    const std::vector<MappedStruct> getMappedStructs() const { return mMappedStructs; }

  protected:
    bool visitDeclaration(Visit visit, TIntermDeclaration *node) override;

  private:
    void mapBlockStructMembers(TIntermSymbol *blockDeclarator, const TInterfaceBlock *block);

    std::vector<MappedStruct> mMappedStructs;
};

void FlagStd140StructsTraverser::mapBlockStructMembers(TIntermSymbol *blockDeclarator,
                                                       const TInterfaceBlock *block)
{
    for (auto *field : block->fields())
    {
        if (field->type()->getBasicType() == EbtStruct)
        {
            MappedStruct mappedStruct;
            mappedStruct.blockDeclarator = blockDeclarator;
            mappedStruct.field           = field;
            mMappedStructs.push_back(mappedStruct);
        }
    }
}

bool FlagStd140StructsTraverser::visitDeclaration(Visit visit, TIntermDeclaration *node)
{
    TIntermTyped *declarator = node->getSequence()->back()->getAsTyped();
    if (declarator->getBasicType() == EbtInterfaceBlock)
    {
        const TInterfaceBlock *block = declarator->getType().getInterfaceBlock();
        if (block->blockStorage() == EbsStd140)
        {
            mapBlockStructMembers(declarator->getAsSymbolNode(), block);
        }
    }
    return false;
}

}  // anonymous namespace

std::vector<MappedStruct> FlagStd140Structs(TIntermNode *node)
{
    FlagStd140StructsTraverser flaggingTraversal;

    node->traverse(&flaggingTraversal);

    return flaggingTraversal.getMappedStructs();
}

}  // namespace sh
