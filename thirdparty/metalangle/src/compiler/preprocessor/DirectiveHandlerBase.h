//
// Copyright 2012 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_PREPROCESSOR_DIRECTIVEHANDLERBASE_H_
#define COMPILER_PREPROCESSOR_DIRECTIVEHANDLERBASE_H_

#include <string>
#include "GLSLANG/ShaderLang.h"

namespace angle
{

namespace pp
{

struct SourceLocation;

// Base class for handling directives.
// Preprocessor uses this class to notify the clients about certain
// preprocessor directives. Derived classes are responsible for
// handling them in an appropriate manner.
class DirectiveHandler
{
  public:
    virtual ~DirectiveHandler();

    virtual void handleError(const SourceLocation &loc, const std::string &msg) = 0;

    // Handle pragma of form: #pragma name[(value)]
    virtual void handlePragma(const SourceLocation &loc,
                              const std::string &name,
                              const std::string &value,
                              bool stdgl) = 0;

    virtual void handleExtension(const SourceLocation &loc,
                                 const std::string &name,
                                 const std::string &behavior) = 0;

    virtual void handleVersion(const SourceLocation &loc, int version, ShShaderSpec spec) = 0;
};

}  // namespace pp

}  // namespace angle

#endif  // COMPILER_PREPROCESSOR_DIRECTIVEHANDLERBASE_H_
