//
// Copyright 2012 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_PREPROCESSOR_DIAGNOSTICSBASE_H_
#define COMPILER_PREPROCESSOR_DIAGNOSTICSBASE_H_

#include <string>

namespace angle
{

namespace pp
{

struct SourceLocation;

// Base class for reporting diagnostic messages.
// Derived classes are responsible for formatting and printing the messages.
class Diagnostics
{
  public:
    enum ID
    {
        PP_ERROR_BEGIN,
        PP_INTERNAL_ERROR,
        PP_OUT_OF_MEMORY,
        PP_INVALID_CHARACTER,
        PP_INVALID_NUMBER,
        PP_INTEGER_OVERFLOW,
        PP_FLOAT_OVERFLOW,
        PP_TOKEN_TOO_LONG,
        PP_INVALID_EXPRESSION,
        PP_DIVISION_BY_ZERO,
        PP_EOF_IN_COMMENT,
        PP_UNEXPECTED_TOKEN,
        PP_DIRECTIVE_INVALID_NAME,
        PP_MACRO_NAME_RESERVED,
        PP_MACRO_REDEFINED,
        PP_MACRO_PREDEFINED_REDEFINED,
        PP_MACRO_PREDEFINED_UNDEFINED,
        PP_MACRO_UNTERMINATED_INVOCATION,
        PP_MACRO_UNDEFINED_WHILE_INVOKED,
        PP_MACRO_TOO_FEW_ARGS,
        PP_MACRO_TOO_MANY_ARGS,
        PP_MACRO_DUPLICATE_PARAMETER_NAMES,
        PP_MACRO_INVOCATION_CHAIN_TOO_DEEP,
        PP_CONDITIONAL_ENDIF_WITHOUT_IF,
        PP_CONDITIONAL_ELSE_WITHOUT_IF,
        PP_CONDITIONAL_ELSE_AFTER_ELSE,
        PP_CONDITIONAL_ELIF_WITHOUT_IF,
        PP_CONDITIONAL_ELIF_AFTER_ELSE,
        PP_CONDITIONAL_UNTERMINATED,
        PP_CONDITIONAL_UNEXPECTED_TOKEN,
        PP_INVALID_EXTENSION_NAME,
        PP_INVALID_EXTENSION_BEHAVIOR,
        PP_INVALID_EXTENSION_DIRECTIVE,
        PP_INVALID_VERSION_NUMBER,
        PP_INVALID_VERSION_DIRECTIVE,
        PP_VERSION_NOT_FIRST_STATEMENT,
        PP_VERSION_NOT_FIRST_LINE_ESSL3,
        PP_INVALID_LINE_NUMBER,
        PP_INVALID_FILE_NUMBER,
        PP_INVALID_LINE_DIRECTIVE,
        PP_NON_PP_TOKEN_BEFORE_EXTENSION_ESSL,
        PP_UNDEFINED_SHIFT,
        PP_TOKENIZER_ERROR,
        PP_ERROR_END,

        PP_WARNING_BEGIN,
        PP_EOF_IN_DIRECTIVE,
        PP_UNRECOGNIZED_PRAGMA,
        PP_NON_PP_TOKEN_BEFORE_EXTENSION_WEBGL,
        PP_WARNING_MACRO_NAME_RESERVED,
        PP_WARNING_END
    };

    virtual ~Diagnostics();

    void report(ID id, const SourceLocation &loc, const std::string &text);

  protected:
    bool isError(ID id);
    const char *message(ID id);

    virtual void print(ID id, const SourceLocation &loc, const std::string &text) = 0;
};

}  // namespace pp

}  // namespace angle

#endif  // COMPILER_PREPROCESSOR_DIAGNOSTICSBASE_H_
