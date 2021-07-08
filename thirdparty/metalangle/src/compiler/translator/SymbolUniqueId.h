//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// SymbolUniqueId.h: Encapsulates a unique id for a symbol.

#ifndef COMPILER_TRANSLATOR_SYMBOLUNIQUEID_H_
#define COMPILER_TRANSLATOR_SYMBOLUNIQUEID_H_

#include "compiler/translator/Common.h"

namespace sh
{

class TSymbolTable;
class TSymbol;

class TSymbolUniqueId
{
  public:
    POOL_ALLOCATOR_NEW_DELETE
    explicit TSymbolUniqueId(const TSymbol &symbol);
    constexpr TSymbolUniqueId(const TSymbolUniqueId &) = default;
    TSymbolUniqueId &operator                          =(const TSymbolUniqueId &);
    bool operator==(const TSymbolUniqueId &) const;

    constexpr int get() const { return mId; }

  private:
    friend class TSymbolTable;
    explicit TSymbolUniqueId(TSymbolTable *symbolTable);

    friend class BuiltInId;
    constexpr TSymbolUniqueId(int staticId) : mId(staticId) {}

    int mId;
};

enum class SymbolType
{
    BuiltIn,
    UserDefined,
    AngleInternal,
    Empty  // Meaning symbol without a name.
};

enum class SymbolClass
{
    Function,
    Variable,
    Struct,
    InterfaceBlock
};

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_SYMBOLUNIQUEID_H_
