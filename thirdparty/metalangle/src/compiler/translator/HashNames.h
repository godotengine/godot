//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_TRANSLATOR_HASHNAMES_H_
#define COMPILER_TRANSLATOR_HASHNAMES_H_

#include <map>

#include "GLSLANG/ShaderLang.h"
#include "compiler/translator/Common.h"

namespace sh
{

typedef std::map<TPersistString, TPersistString> NameMap;

class ImmutableString;
class TSymbol;

ImmutableString HashName(const ImmutableString &name,
                         ShHashFunction64 hashFunction,
                         NameMap *nameMap);

// Hash user-defined name for GLSL output, with special handling for internal names.
// The nameMap parameter is optional and is used to cache hashed names if set.
ImmutableString HashName(const TSymbol *symbol, ShHashFunction64 hashFunction, NameMap *nameMap);

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_HASHNAMES_H_
