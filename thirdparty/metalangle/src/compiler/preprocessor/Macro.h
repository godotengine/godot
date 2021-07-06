//
// Copyright 2012 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_PREPROCESSOR_MACRO_H_
#define COMPILER_PREPROCESSOR_MACRO_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace angle
{

namespace pp
{

struct Token;

struct Macro
{
    enum Type
    {
        kTypeObj,
        kTypeFunc
    };
    typedef std::vector<std::string> Parameters;
    typedef std::vector<Token> Replacements;

    Macro();
    ~Macro();
    bool equals(const Macro &other) const;

    bool predefined;
    mutable bool disabled;
    mutable int expansionCount;

    Type type;
    std::string name;
    Parameters parameters;
    Replacements replacements;
};

typedef std::map<std::string, std::shared_ptr<Macro>> MacroSet;

void PredefineMacro(MacroSet *macroSet, const char *name, int value);

}  // namespace pp

}  // namespace angle

#endif  // COMPILER_PREPROCESSOR_MACRO_H_
