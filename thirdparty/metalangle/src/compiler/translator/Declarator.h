//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Declarator.h:
//   Declarator type for parsing structure field declarators.

#ifndef COMPILER_TRANSLATOR_DECLARATOR_H_
#define COMPILER_TRANSLATOR_DECLARATOR_H_

#include "compiler/translator/Common.h"
#include "compiler/translator/ImmutableString.h"

namespace sh
{

// Declarator like "a[2][4]". Only used for parsing structure field declarators.
class TDeclarator : angle::NonCopyable
{
  public:
    POOL_ALLOCATOR_NEW_DELETE
    TDeclarator(const ImmutableString &name, const TSourceLoc &line);

    TDeclarator(const ImmutableString &name,
                const TVector<unsigned int> *arraySizes,
                const TSourceLoc &line);

    const ImmutableString &name() const { return mName; }

    bool isArray() const;
    const TVector<unsigned int> *arraySizes() const { return mArraySizes; }

    const TSourceLoc &line() const { return mLine; }

  private:
    const ImmutableString mName;

    // Outermost array size is stored at the end of the vector.
    const TVector<unsigned int> *const mArraySizes;

    const TSourceLoc mLine;
};

using TDeclaratorList = TVector<TDeclarator *>;

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_DECLARATOR_H_
