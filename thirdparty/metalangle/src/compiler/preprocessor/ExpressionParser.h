//
// Copyright 2012 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_PREPROCESSOR_EXPRESSIONPARSER_H_
#define COMPILER_PREPROCESSOR_EXPRESSIONPARSER_H_

#include "common/angleutils.h"
#include "compiler/preprocessor/DiagnosticsBase.h"

namespace angle
{

namespace pp
{

class Lexer;
struct Token;

class ExpressionParser : angle::NonCopyable
{
  public:
    struct ErrorSettings
    {
        Diagnostics::ID unexpectedIdentifier;
        bool integerLiteralsMustFit32BitSignedRange;
    };

    ExpressionParser(Lexer *lexer, Diagnostics *diagnostics);

    bool parse(Token *token,
               int *result,
               bool parsePresetToken,
               const ErrorSettings &errorSettings,
               bool *valid);

  private:
    Lexer *mLexer;
    Diagnostics *mDiagnostics;
};

}  // namespace pp

}  // namespace angle

#endif  // COMPILER_PREPROCESSOR_EXPRESSIONPARSER_H_
