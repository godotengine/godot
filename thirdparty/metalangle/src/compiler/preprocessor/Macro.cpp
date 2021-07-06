//
// Copyright 2011 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "compiler/preprocessor/Macro.h"

#include "common/angleutils.h"
#include "compiler/preprocessor/Token.h"

namespace angle
{

namespace pp
{

Macro::Macro() : predefined(false), disabled(false), expansionCount(0), type(kTypeObj) {}

Macro::~Macro() {}

bool Macro::equals(const Macro &other) const
{
    return (type == other.type) && (name == other.name) && (parameters == other.parameters) &&
           (replacements == other.replacements);
}

void PredefineMacro(MacroSet *macroSet, const char *name, int value)
{
    Token token;
    token.type = Token::CONST_INT;
    token.text = ToString(value);

    std::shared_ptr<Macro> macro = std::make_shared<Macro>();
    macro->predefined            = true;
    macro->type                  = Macro::kTypeObj;
    macro->name                  = name;
    macro->replacements.push_back(token);

    (*macroSet)[name] = macro;
}

}  // namespace pp

}  // namespace angle
