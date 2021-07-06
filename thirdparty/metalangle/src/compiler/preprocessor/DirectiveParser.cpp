//
// Copyright 2011 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "compiler/preprocessor/DirectiveParser.h"

#include <algorithm>
#include <cstdlib>
#include <sstream>

#include "GLSLANG/ShaderLang.h"
#include "common/debug.h"
#include "compiler/preprocessor/DiagnosticsBase.h"
#include "compiler/preprocessor/DirectiveHandlerBase.h"
#include "compiler/preprocessor/ExpressionParser.h"
#include "compiler/preprocessor/MacroExpander.h"
#include "compiler/preprocessor/Token.h"
#include "compiler/preprocessor/Tokenizer.h"

namespace angle
{

namespace
{
enum DirectiveType
{
    DIRECTIVE_NONE,
    DIRECTIVE_DEFINE,
    DIRECTIVE_UNDEF,
    DIRECTIVE_IF,
    DIRECTIVE_IFDEF,
    DIRECTIVE_IFNDEF,
    DIRECTIVE_ELSE,
    DIRECTIVE_ELIF,
    DIRECTIVE_ENDIF,
    DIRECTIVE_ERROR,
    DIRECTIVE_PRAGMA,
    DIRECTIVE_EXTENSION,
    DIRECTIVE_VERSION,
    DIRECTIVE_LINE
};

DirectiveType getDirective(const pp::Token *token)
{
    const char kDirectiveDefine[]    = "define";
    const char kDirectiveUndef[]     = "undef";
    const char kDirectiveIf[]        = "if";
    const char kDirectiveIfdef[]     = "ifdef";
    const char kDirectiveIfndef[]    = "ifndef";
    const char kDirectiveElse[]      = "else";
    const char kDirectiveElif[]      = "elif";
    const char kDirectiveEndif[]     = "endif";
    const char kDirectiveError[]     = "error";
    const char kDirectivePragma[]    = "pragma";
    const char kDirectiveExtension[] = "extension";
    const char kDirectiveVersion[]   = "version";
    const char kDirectiveLine[]      = "line";

    if (token->type != pp::Token::IDENTIFIER)
        return DIRECTIVE_NONE;

    if (token->text == kDirectiveDefine)
        return DIRECTIVE_DEFINE;
    if (token->text == kDirectiveUndef)
        return DIRECTIVE_UNDEF;
    if (token->text == kDirectiveIf)
        return DIRECTIVE_IF;
    if (token->text == kDirectiveIfdef)
        return DIRECTIVE_IFDEF;
    if (token->text == kDirectiveIfndef)
        return DIRECTIVE_IFNDEF;
    if (token->text == kDirectiveElse)
        return DIRECTIVE_ELSE;
    if (token->text == kDirectiveElif)
        return DIRECTIVE_ELIF;
    if (token->text == kDirectiveEndif)
        return DIRECTIVE_ENDIF;
    if (token->text == kDirectiveError)
        return DIRECTIVE_ERROR;
    if (token->text == kDirectivePragma)
        return DIRECTIVE_PRAGMA;
    if (token->text == kDirectiveExtension)
        return DIRECTIVE_EXTENSION;
    if (token->text == kDirectiveVersion)
        return DIRECTIVE_VERSION;
    if (token->text == kDirectiveLine)
        return DIRECTIVE_LINE;

    return DIRECTIVE_NONE;
}

bool isConditionalDirective(DirectiveType directive)
{
    switch (directive)
    {
        case DIRECTIVE_IF:
        case DIRECTIVE_IFDEF:
        case DIRECTIVE_IFNDEF:
        case DIRECTIVE_ELSE:
        case DIRECTIVE_ELIF:
        case DIRECTIVE_ENDIF:
            return true;
        default:
            return false;
    }
}

// Returns true if the token represents End Of Directive.
bool isEOD(const pp::Token *token)
{
    return (token->type == '\n') || (token->type == pp::Token::LAST);
}

void skipUntilEOD(pp::Lexer *lexer, pp::Token *token)
{
    while (!isEOD(token))
    {
        lexer->lex(token);
    }
}

bool isMacroNameReserved(const std::string &name)
{
    // Names prefixed with "GL_" and the name "defined" are reserved.
    return name == "defined" || (name.substr(0, 3) == "GL_");
}

bool hasDoubleUnderscores(const std::string &name)
{
    return (name.find("__") != std::string::npos);
}

bool isMacroPredefined(const std::string &name, const pp::MacroSet &macroSet)
{
    pp::MacroSet::const_iterator iter = macroSet.find(name);
    return iter != macroSet.end() ? iter->second->predefined : false;
}

}  // namespace

namespace pp
{
DirectiveParser::DirectiveParser(Tokenizer *tokenizer,
                                 MacroSet *macroSet,
                                 Diagnostics *diagnostics,
                                 DirectiveHandler *directiveHandler,
                                 const PreprocessorSettings &settings)
    : mPastFirstStatement(false),
      mSeenNonPreprocessorToken(false),
      mTokenizer(tokenizer),
      mMacroSet(macroSet),
      mDiagnostics(diagnostics),
      mDirectiveHandler(directiveHandler),
      mShaderVersion(100),
      mSettings(settings)
{}

DirectiveParser::~DirectiveParser() {}

void DirectiveParser::lex(Token *token)
{
    do
    {
        mTokenizer->lex(token);

        if (token->type == Token::PP_HASH)
        {
            parseDirective(token);
            mPastFirstStatement = true;
        }
        else if (!isEOD(token))
        {
            mSeenNonPreprocessorToken = true;
        }

        if (token->type == Token::LAST)
        {
            if (!mConditionalStack.empty())
            {
                const ConditionalBlock &block = mConditionalStack.back();
                mDiagnostics->report(Diagnostics::PP_CONDITIONAL_UNTERMINATED, block.location,
                                     block.type);
            }
            break;
        }

    } while (skipping() || (token->type == '\n'));

    mPastFirstStatement = true;
}

void DirectiveParser::parseDirective(Token *token)
{
    ASSERT(token->type == Token::PP_HASH);

    mTokenizer->lex(token);
    if (isEOD(token))
    {
        // Empty Directive.
        return;
    }

    DirectiveType directive = getDirective(token);

    // While in an excluded conditional block/group,
    // we only parse conditional directives.
    if (skipping() && !isConditionalDirective(directive))
    {
        skipUntilEOD(mTokenizer, token);
        return;
    }

    switch (directive)
    {
        case DIRECTIVE_NONE:
            mDiagnostics->report(Diagnostics::PP_DIRECTIVE_INVALID_NAME, token->location,
                                 token->text);
            skipUntilEOD(mTokenizer, token);
            break;
        case DIRECTIVE_DEFINE:
            parseDefine(token);
            break;
        case DIRECTIVE_UNDEF:
            parseUndef(token);
            break;
        case DIRECTIVE_IF:
            parseIf(token);
            break;
        case DIRECTIVE_IFDEF:
            parseIfdef(token);
            break;
        case DIRECTIVE_IFNDEF:
            parseIfndef(token);
            break;
        case DIRECTIVE_ELSE:
            parseElse(token);
            break;
        case DIRECTIVE_ELIF:
            parseElif(token);
            break;
        case DIRECTIVE_ENDIF:
            parseEndif(token);
            break;
        case DIRECTIVE_ERROR:
            parseError(token);
            break;
        case DIRECTIVE_PRAGMA:
            parsePragma(token);
            break;
        case DIRECTIVE_EXTENSION:
            parseExtension(token);
            break;
        case DIRECTIVE_VERSION:
            parseVersion(token);
            break;
        case DIRECTIVE_LINE:
            parseLine(token);
            break;
        default:
            UNREACHABLE();
            break;
    }

    skipUntilEOD(mTokenizer, token);
    if (token->type == Token::LAST)
    {
        mDiagnostics->report(Diagnostics::PP_EOF_IN_DIRECTIVE, token->location, token->text);
    }
}

void DirectiveParser::parseDefine(Token *token)
{
    ASSERT(getDirective(token) == DIRECTIVE_DEFINE);

    mTokenizer->lex(token);
    if (token->type != Token::IDENTIFIER)
    {
        mDiagnostics->report(Diagnostics::PP_UNEXPECTED_TOKEN, token->location, token->text);
        return;
    }
    if (isMacroPredefined(token->text, *mMacroSet))
    {
        mDiagnostics->report(Diagnostics::PP_MACRO_PREDEFINED_REDEFINED, token->location,
                             token->text);
        return;
    }
    if (isMacroNameReserved(token->text))
    {
        mDiagnostics->report(Diagnostics::PP_MACRO_NAME_RESERVED, token->location, token->text);
        return;
    }
    // Using double underscores is allowed, but may result in unintended
    // behavior, so a warning is issued. At the time of writing this was
    // specified in ESSL 3.10, but the intent judging from Khronos
    // discussions and dEQP tests was that double underscores should be
    // allowed in earlier ESSL versions too.
    if (hasDoubleUnderscores(token->text))
    {
        mDiagnostics->report(Diagnostics::PP_WARNING_MACRO_NAME_RESERVED, token->location,
                             token->text);
    }

    std::shared_ptr<Macro> macro = std::make_shared<Macro>();
    macro->type                  = Macro::kTypeObj;
    macro->name                  = token->text;

    mTokenizer->lex(token);
    if (token->type == '(' && !token->hasLeadingSpace())
    {
        // Function-like macro. Collect arguments.
        macro->type = Macro::kTypeFunc;
        do
        {
            mTokenizer->lex(token);
            if (token->type != Token::IDENTIFIER)
                break;

            if (std::find(macro->parameters.begin(), macro->parameters.end(), token->text) !=
                macro->parameters.end())
            {
                mDiagnostics->report(Diagnostics::PP_MACRO_DUPLICATE_PARAMETER_NAMES,
                                     token->location, token->text);
                return;
            }

            macro->parameters.push_back(token->text);

            mTokenizer->lex(token);  // Get ','.
        } while (token->type == ',');

        if (token->type != ')')
        {
            mDiagnostics->report(Diagnostics::PP_UNEXPECTED_TOKEN, token->location, token->text);
            return;
        }
        mTokenizer->lex(token);  // Get ')'.
    }

    while ((token->type != '\n') && (token->type != Token::LAST))
    {
        // Reset the token location because it is unnecessary in replacement
        // list. Resetting it also allows us to reuse Token::equals() to
        // compare macros.
        token->location = SourceLocation();
        macro->replacements.push_back(*token);
        mTokenizer->lex(token);
    }
    if (!macro->replacements.empty())
    {
        // Whitespace preceding the replacement list is not considered part of
        // the replacement list for either form of macro.
        macro->replacements.front().setHasLeadingSpace(false);
    }

    // Check for macro redefinition.
    MacroSet::const_iterator iter = mMacroSet->find(macro->name);
    if (iter != mMacroSet->end() && !macro->equals(*iter->second))
    {
        mDiagnostics->report(Diagnostics::PP_MACRO_REDEFINED, token->location, macro->name);
        return;
    }
    mMacroSet->insert(std::make_pair(macro->name, macro));
}

void DirectiveParser::parseUndef(Token *token)
{
    ASSERT(getDirective(token) == DIRECTIVE_UNDEF);

    mTokenizer->lex(token);
    if (token->type != Token::IDENTIFIER)
    {
        mDiagnostics->report(Diagnostics::PP_UNEXPECTED_TOKEN, token->location, token->text);
        return;
    }

    MacroSet::iterator iter = mMacroSet->find(token->text);
    if (iter != mMacroSet->end())
    {
        if (iter->second->predefined)
        {
            mDiagnostics->report(Diagnostics::PP_MACRO_PREDEFINED_UNDEFINED, token->location,
                                 token->text);
            return;
        }
        else if (iter->second->expansionCount > 0)
        {
            mDiagnostics->report(Diagnostics::PP_MACRO_UNDEFINED_WHILE_INVOKED, token->location,
                                 token->text);
            return;
        }
        else
        {
            mMacroSet->erase(iter);
        }
    }

    mTokenizer->lex(token);
    if (!isEOD(token))
    {
        mDiagnostics->report(Diagnostics::PP_UNEXPECTED_TOKEN, token->location, token->text);
        skipUntilEOD(mTokenizer, token);
    }
}

void DirectiveParser::parseIf(Token *token)
{
    ASSERT(getDirective(token) == DIRECTIVE_IF);
    parseConditionalIf(token);
}

void DirectiveParser::parseIfdef(Token *token)
{
    ASSERT(getDirective(token) == DIRECTIVE_IFDEF);
    parseConditionalIf(token);
}

void DirectiveParser::parseIfndef(Token *token)
{
    ASSERT(getDirective(token) == DIRECTIVE_IFNDEF);
    parseConditionalIf(token);
}

void DirectiveParser::parseElse(Token *token)
{
    ASSERT(getDirective(token) == DIRECTIVE_ELSE);

    if (mConditionalStack.empty())
    {
        mDiagnostics->report(Diagnostics::PP_CONDITIONAL_ELSE_WITHOUT_IF, token->location,
                             token->text);
        skipUntilEOD(mTokenizer, token);
        return;
    }

    ConditionalBlock &block = mConditionalStack.back();
    if (block.skipBlock)
    {
        // No diagnostics. Just skip the whole line.
        skipUntilEOD(mTokenizer, token);
        return;
    }
    if (block.foundElseGroup)
    {
        mDiagnostics->report(Diagnostics::PP_CONDITIONAL_ELSE_AFTER_ELSE, token->location,
                             token->text);
        skipUntilEOD(mTokenizer, token);
        return;
    }

    block.foundElseGroup  = true;
    block.skipGroup       = block.foundValidGroup;
    block.foundValidGroup = true;

    // Check if there are extra tokens after #else.
    mTokenizer->lex(token);
    if (!isEOD(token))
    {
        mDiagnostics->report(Diagnostics::PP_CONDITIONAL_UNEXPECTED_TOKEN, token->location,
                             token->text);
        skipUntilEOD(mTokenizer, token);
    }
}

void DirectiveParser::parseElif(Token *token)
{
    ASSERT(getDirective(token) == DIRECTIVE_ELIF);

    if (mConditionalStack.empty())
    {
        mDiagnostics->report(Diagnostics::PP_CONDITIONAL_ELIF_WITHOUT_IF, token->location,
                             token->text);
        skipUntilEOD(mTokenizer, token);
        return;
    }

    ConditionalBlock &block = mConditionalStack.back();
    if (block.skipBlock)
    {
        // No diagnostics. Just skip the whole line.
        skipUntilEOD(mTokenizer, token);
        return;
    }
    if (block.foundElseGroup)
    {
        mDiagnostics->report(Diagnostics::PP_CONDITIONAL_ELIF_AFTER_ELSE, token->location,
                             token->text);
        skipUntilEOD(mTokenizer, token);
        return;
    }
    if (block.foundValidGroup)
    {
        // Do not parse the expression.
        // Also be careful not to emit a diagnostic.
        block.skipGroup = true;
        skipUntilEOD(mTokenizer, token);
        return;
    }

    int expression        = parseExpressionIf(token);
    block.skipGroup       = expression == 0;
    block.foundValidGroup = expression != 0;
}

void DirectiveParser::parseEndif(Token *token)
{
    ASSERT(getDirective(token) == DIRECTIVE_ENDIF);

    if (mConditionalStack.empty())
    {
        mDiagnostics->report(Diagnostics::PP_CONDITIONAL_ENDIF_WITHOUT_IF, token->location,
                             token->text);
        skipUntilEOD(mTokenizer, token);
        return;
    }

    mConditionalStack.pop_back();

    // Check if there are tokens after #endif.
    mTokenizer->lex(token);
    if (!isEOD(token))
    {
        mDiagnostics->report(Diagnostics::PP_CONDITIONAL_UNEXPECTED_TOKEN, token->location,
                             token->text);
        skipUntilEOD(mTokenizer, token);
    }
}

void DirectiveParser::parseError(Token *token)
{
    ASSERT(getDirective(token) == DIRECTIVE_ERROR);

    std::ostringstream stream;
    mTokenizer->lex(token);
    while ((token->type != '\n') && (token->type != Token::LAST))
    {
        stream << *token;
        mTokenizer->lex(token);
    }
    mDirectiveHandler->handleError(token->location, stream.str());
}

// Parses pragma of form: #pragma name[(value)].
void DirectiveParser::parsePragma(Token *token)
{
    ASSERT(getDirective(token) == DIRECTIVE_PRAGMA);

    enum State
    {
        PRAGMA_NAME,
        LEFT_PAREN,
        PRAGMA_VALUE,
        RIGHT_PAREN
    };

    bool valid = true;
    std::string name, value;
    int state = PRAGMA_NAME;

    mTokenizer->lex(token);
    bool stdgl = token->text == "STDGL";
    if (stdgl)
    {
        mTokenizer->lex(token);
    }
    while ((token->type != '\n') && (token->type != Token::LAST))
    {
        switch (state++)
        {
            case PRAGMA_NAME:
                name  = token->text;
                valid = valid && (token->type == Token::IDENTIFIER);
                break;
            case LEFT_PAREN:
                valid = valid && (token->type == '(');
                break;
            case PRAGMA_VALUE:
                value = token->text;
                valid = valid && (token->type == Token::IDENTIFIER);
                break;
            case RIGHT_PAREN:
                valid = valid && (token->type == ')');
                break;
            default:
                valid = false;
                break;
        }
        mTokenizer->lex(token);
    }

    valid = valid && ((state == PRAGMA_NAME) ||     // Empty pragma.
                      (state == LEFT_PAREN) ||      // Without value.
                      (state == RIGHT_PAREN + 1));  // With value.
    if (!valid)
    {
        mDiagnostics->report(Diagnostics::PP_UNRECOGNIZED_PRAGMA, token->location, name);
    }
    else if (state > PRAGMA_NAME)  // Do not notify for empty pragma.
    {
        mDirectiveHandler->handlePragma(token->location, name, value, stdgl);
    }
}

void DirectiveParser::parseExtension(Token *token)
{
    ASSERT(getDirective(token) == DIRECTIVE_EXTENSION);

    enum State
    {
        EXT_NAME,
        COLON,
        EXT_BEHAVIOR
    };

    bool valid = true;
    std::string name, behavior;
    int state = EXT_NAME;

    mTokenizer->lex(token);
    while ((token->type != '\n') && (token->type != Token::LAST))
    {
        switch (state++)
        {
            case EXT_NAME:
                if (valid && (token->type != Token::IDENTIFIER))
                {
                    mDiagnostics->report(Diagnostics::PP_INVALID_EXTENSION_NAME, token->location,
                                         token->text);
                    valid = false;
                }
                if (valid)
                    name = token->text;
                break;
            case COLON:
                if (valid && (token->type != ':'))
                {
                    mDiagnostics->report(Diagnostics::PP_UNEXPECTED_TOKEN, token->location,
                                         token->text);
                    valid = false;
                }
                break;
            case EXT_BEHAVIOR:
                if (valid && (token->type != Token::IDENTIFIER))
                {
                    mDiagnostics->report(Diagnostics::PP_INVALID_EXTENSION_BEHAVIOR,
                                         token->location, token->text);
                    valid = false;
                }
                if (valid)
                    behavior = token->text;
                break;
            default:
                if (valid)
                {
                    mDiagnostics->report(Diagnostics::PP_UNEXPECTED_TOKEN, token->location,
                                         token->text);
                    valid = false;
                }
                break;
        }
        mTokenizer->lex(token);
    }
    if (valid && (state != EXT_BEHAVIOR + 1))
    {
        mDiagnostics->report(Diagnostics::PP_INVALID_EXTENSION_DIRECTIVE, token->location,
                             token->text);
        valid = false;
    }
    if (valid && mSeenNonPreprocessorToken)
    {
        if (mSettings.shaderSpec == SH_WEBGL_SPEC && mShaderVersion < 300)
        {
            mDiagnostics->report(Diagnostics::PP_NON_PP_TOKEN_BEFORE_EXTENSION_WEBGL,
                                 token->location, token->text);
        }
        else
        {
            mDiagnostics->report(Diagnostics::PP_NON_PP_TOKEN_BEFORE_EXTENSION_ESSL,
                                 token->location, token->text);
            valid = false;
        }
    }
    if (valid)
        mDirectiveHandler->handleExtension(token->location, name, behavior);
}

void DirectiveParser::parseVersion(Token *token)
{
    ASSERT(getDirective(token) == DIRECTIVE_VERSION);

    if (mPastFirstStatement)
    {
        mDiagnostics->report(Diagnostics::PP_VERSION_NOT_FIRST_STATEMENT, token->location,
                             token->text);
        skipUntilEOD(mTokenizer, token);
        return;
    }

    enum State
    {
        VERSION_NUMBER,
        VERSION_PROFILE_ES,
        VERSION_PROFILE_GL,
        VERSION_ENDLINE
    };

    bool valid  = true;
    int version = 0;
    int state   = VERSION_NUMBER;

    mTokenizer->lex(token);
    while (valid && (token->type != '\n') && (token->type != Token::LAST))
    {
        switch (state)
        {
            case VERSION_NUMBER:
                if (token->type != Token::CONST_INT)
                {
                    mDiagnostics->report(Diagnostics::PP_INVALID_VERSION_NUMBER, token->location,
                                         token->text);
                    valid = false;
                }
                if (valid && !token->iValue(&version))
                {
                    mDiagnostics->report(Diagnostics::PP_INTEGER_OVERFLOW, token->location,
                                         token->text);
                    valid = false;
                }
                if (valid)
                {
                    if (sh::IsDesktopGLSpec(mSettings.shaderSpec))
                    {
                        state = VERSION_PROFILE_GL;
                    }
                    else if (version < 300)
                    {
                        state = VERSION_ENDLINE;
                    }
                    else
                    {
                        state = VERSION_PROFILE_ES;
                    }
                }
                break;
            case VERSION_PROFILE_ES:
                ASSERT(!sh::IsDesktopGLSpec(mSettings.shaderSpec));
                if (token->type != Token::IDENTIFIER || token->text != "es")
                {
                    mDiagnostics->report(Diagnostics::PP_INVALID_VERSION_DIRECTIVE, token->location,
                                         token->text);
                    valid = false;
                }
                state = VERSION_ENDLINE;
                break;
            case VERSION_PROFILE_GL:
                ASSERT(sh::IsDesktopGLSpec(mSettings.shaderSpec));
                if (token->type != Token::IDENTIFIER || token->text != "core")
                {
                    mDiagnostics->report(Diagnostics::PP_INVALID_VERSION_DIRECTIVE, token->location,
                                         token->text);
                    valid = false;
                }
                state = VERSION_ENDLINE;
                break;
            default:
                mDiagnostics->report(Diagnostics::PP_UNEXPECTED_TOKEN, token->location,
                                     token->text);
                valid = false;
                break;
        }

        mTokenizer->lex(token);

        if (token->type == '\n' && state == VERSION_PROFILE_GL)
        {
            state = VERSION_ENDLINE;
        }
    }

    if (valid && (state != VERSION_ENDLINE))
    {
        mDiagnostics->report(Diagnostics::PP_INVALID_VERSION_DIRECTIVE, token->location,
                             token->text);
        valid = false;
    }

    if (valid && version >= 300 && token->location.line > 1)
    {
        mDiagnostics->report(Diagnostics::PP_VERSION_NOT_FIRST_LINE_ESSL3, token->location,
                             token->text);
        valid = false;
    }

    if (valid)
    {
        mDirectiveHandler->handleVersion(token->location, version, mSettings.shaderSpec);
        mShaderVersion = version;
        PredefineMacro(mMacroSet, "__VERSION__", version);
    }
}

void DirectiveParser::parseLine(Token *token)
{
    ASSERT(getDirective(token) == DIRECTIVE_LINE);

    bool valid            = true;
    bool parsedFileNumber = false;
    int line = 0, file = 0;

    MacroExpander macroExpander(mTokenizer, mMacroSet, mDiagnostics, mSettings, false);

    // Lex the first token after "#line" so we can check it for EOD.
    macroExpander.lex(token);

    if (isEOD(token))
    {
        mDiagnostics->report(Diagnostics::PP_INVALID_LINE_DIRECTIVE, token->location, token->text);
        valid = false;
    }
    else
    {
        ExpressionParser expressionParser(&macroExpander, mDiagnostics);
        ExpressionParser::ErrorSettings errorSettings;

        // See GLES3 section 12.42
        errorSettings.integerLiteralsMustFit32BitSignedRange = true;

        errorSettings.unexpectedIdentifier = Diagnostics::PP_INVALID_LINE_NUMBER;
        // The first token was lexed earlier to check if it was EOD. Include
        // the token in parsing for a second time by setting the
        // parsePresetToken flag to true.
        expressionParser.parse(token, &line, true, errorSettings, &valid);
        if (!isEOD(token) && valid)
        {
            errorSettings.unexpectedIdentifier = Diagnostics::PP_INVALID_FILE_NUMBER;
            // After parsing the line expression expressionParser has also
            // advanced to the first token of the file expression - this is the
            // token that makes the parser reduce the "input" rule for the line
            // expression and stop. So we're using parsePresetToken = true here
            // as well.
            expressionParser.parse(token, &file, true, errorSettings, &valid);
            parsedFileNumber = true;
        }
        if (!isEOD(token))
        {
            if (valid)
            {
                mDiagnostics->report(Diagnostics::PP_UNEXPECTED_TOKEN, token->location,
                                     token->text);
                valid = false;
            }
            skipUntilEOD(mTokenizer, token);
        }
    }

    if (valid)
    {
        mTokenizer->setLineNumber(line);
        if (parsedFileNumber)
            mTokenizer->setFileNumber(file);
    }
}

bool DirectiveParser::skipping() const
{
    if (mConditionalStack.empty())
        return false;

    const ConditionalBlock &block = mConditionalStack.back();
    return block.skipBlock || block.skipGroup;
}

void DirectiveParser::parseConditionalIf(Token *token)
{
    ConditionalBlock block;
    block.type     = token->text;
    block.location = token->location;

    if (skipping())
    {
        // This conditional block is inside another conditional group
        // which is skipped. As a consequence this whole block is skipped.
        // Be careful not to parse the conditional expression that might
        // emit a diagnostic.
        skipUntilEOD(mTokenizer, token);
        block.skipBlock = true;
    }
    else
    {
        DirectiveType directive = getDirective(token);

        int expression = 0;
        switch (directive)
        {
            case DIRECTIVE_IF:
                expression = parseExpressionIf(token);
                break;
            case DIRECTIVE_IFDEF:
                expression = parseExpressionIfdef(token);
                break;
            case DIRECTIVE_IFNDEF:
                expression = parseExpressionIfdef(token) == 0 ? 1 : 0;
                break;
            default:
                UNREACHABLE();
                break;
        }
        block.skipGroup       = expression == 0;
        block.foundValidGroup = expression != 0;
    }
    mConditionalStack.push_back(block);
}

int DirectiveParser::parseExpressionIf(Token *token)
{
    ASSERT((getDirective(token) == DIRECTIVE_IF) || (getDirective(token) == DIRECTIVE_ELIF));

    MacroExpander macroExpander(mTokenizer, mMacroSet, mDiagnostics, mSettings, true);
    ExpressionParser expressionParser(&macroExpander, mDiagnostics);

    int expression = 0;
    ExpressionParser::ErrorSettings errorSettings;
    errorSettings.integerLiteralsMustFit32BitSignedRange = false;
    errorSettings.unexpectedIdentifier = Diagnostics::PP_CONDITIONAL_UNEXPECTED_TOKEN;

    bool valid = true;
    expressionParser.parse(token, &expression, false, errorSettings, &valid);

    // Check if there are tokens after #if expression.
    if (!isEOD(token))
    {
        mDiagnostics->report(Diagnostics::PP_CONDITIONAL_UNEXPECTED_TOKEN, token->location,
                             token->text);
        skipUntilEOD(mTokenizer, token);
    }

    return expression;
}

int DirectiveParser::parseExpressionIfdef(Token *token)
{
    ASSERT((getDirective(token) == DIRECTIVE_IFDEF) || (getDirective(token) == DIRECTIVE_IFNDEF));

    mTokenizer->lex(token);
    if (token->type != Token::IDENTIFIER)
    {
        mDiagnostics->report(Diagnostics::PP_UNEXPECTED_TOKEN, token->location, token->text);
        skipUntilEOD(mTokenizer, token);
        return 0;
    }

    MacroSet::const_iterator iter = mMacroSet->find(token->text);
    int expression                = iter != mMacroSet->end() ? 1 : 0;

    // Check if there are tokens after #ifdef expression.
    mTokenizer->lex(token);
    if (!isEOD(token))
    {
        mDiagnostics->report(Diagnostics::PP_CONDITIONAL_UNEXPECTED_TOKEN, token->location,
                             token->text);
        skipUntilEOD(mTokenizer, token);
    }
    return expression;
}

}  // namespace pp

}  // namespace angle
