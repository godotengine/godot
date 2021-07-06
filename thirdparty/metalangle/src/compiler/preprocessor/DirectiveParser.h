//
// Copyright 2012 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_PREPROCESSOR_DIRECTIVEPARSER_H_
#define COMPILER_PREPROCESSOR_DIRECTIVEPARSER_H_

#include "compiler/preprocessor/Lexer.h"
#include "compiler/preprocessor/Macro.h"
#include "compiler/preprocessor/Preprocessor.h"
#include "compiler/preprocessor/SourceLocation.h"

namespace angle
{

namespace pp
{

class Diagnostics;
class DirectiveHandler;
class Tokenizer;

class DirectiveParser : public Lexer
{
  public:
    DirectiveParser(Tokenizer *tokenizer,
                    MacroSet *macroSet,
                    Diagnostics *diagnostics,
                    DirectiveHandler *directiveHandler,
                    const PreprocessorSettings &settings);
    ~DirectiveParser() override;

    void lex(Token *token) override;

  private:
    void parseDirective(Token *token);
    void parseDefine(Token *token);
    void parseUndef(Token *token);
    void parseIf(Token *token);
    void parseIfdef(Token *token);
    void parseIfndef(Token *token);
    void parseElse(Token *token);
    void parseElif(Token *token);
    void parseEndif(Token *token);
    void parseError(Token *token);
    void parsePragma(Token *token);
    void parseExtension(Token *token);
    void parseVersion(Token *token);
    void parseLine(Token *token);

    bool skipping() const;
    void parseConditionalIf(Token *token);
    int parseExpressionIf(Token *token);
    int parseExpressionIfdef(Token *token);

    struct ConditionalBlock
    {
        std::string type;
        SourceLocation location;
        bool skipBlock;
        bool skipGroup;
        bool foundValidGroup;
        bool foundElseGroup;

        ConditionalBlock()
            : skipBlock(false), skipGroup(false), foundValidGroup(false), foundElseGroup(false)
        {}
    };
    bool mPastFirstStatement;
    bool mSeenNonPreprocessorToken;  // Tracks if a non-preprocessor token has been seen yet.  Some
                                     // macros, such as
                                     // #extension must be declared before all shader code.
    std::vector<ConditionalBlock> mConditionalStack;
    Tokenizer *mTokenizer;
    MacroSet *mMacroSet;
    Diagnostics *mDiagnostics;
    DirectiveHandler *mDirectiveHandler;
    int mShaderVersion;
    const PreprocessorSettings mSettings;
};

}  // namespace pp

}  // namespace angle

#endif  // COMPILER_PREPROCESSOR_DIRECTIVEPARSER_H_
