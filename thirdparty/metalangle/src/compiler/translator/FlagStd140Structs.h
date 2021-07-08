//
// Copyright 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// FlagStd140Structs.h: Find structs in std140 blocks, where the padding added in the translator
// conflicts with the "natural" unpadded type.

#ifndef COMPILER_TRANSLATOR_FLAGSTD140STRUCTS_H_
#define COMPILER_TRANSLATOR_FLAGSTD140STRUCTS_H_

#include <vector>

namespace sh
{

class TField;
class TIntermNode;
class TIntermSymbol;

struct MappedStruct
{
    TIntermSymbol *blockDeclarator;
    TField *field;
};

std::vector<MappedStruct> FlagStd140Structs(TIntermNode *node);
}  // namespace sh

#endif  // COMPILER_TRANSLATOR_FLAGSTD140STRUCTS_H_
