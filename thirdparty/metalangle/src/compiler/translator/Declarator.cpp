//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Declarator.cpp:
//   Declarator type for parsing structure field declarators.

#include "compiler/translator/Declarator.h"

namespace sh
{

TDeclarator::TDeclarator(const ImmutableString &name, const TSourceLoc &line)
    : mName(name), mArraySizes(nullptr), mLine(line)
{
    ASSERT(mName != "");
}

TDeclarator::TDeclarator(const ImmutableString &name,
                         const TVector<unsigned int> *arraySizes,
                         const TSourceLoc &line)
    : mName(name), mArraySizes(arraySizes), mLine(line)
{
    ASSERT(mArraySizes);
}

bool TDeclarator::isArray() const
{
    return mArraySizes != nullptr && mArraySizes->size() > 0;
}

}  // namespace sh
