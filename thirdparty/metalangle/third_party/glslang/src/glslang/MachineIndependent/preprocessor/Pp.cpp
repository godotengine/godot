//
// Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
// Copyright (C) 2013 LunarG, Inc.
// Copyright (C) 2015-2018 Google, Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//    Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//    Neither the name of 3Dlabs Inc. Ltd. nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
/****************************************************************************\
Copyright (c) 2002, NVIDIA Corporation.

NVIDIA Corporation("NVIDIA") supplies this software to you in
consideration of your agreement to the following terms, and your use,
installation, modification or redistribution of this NVIDIA software
constitutes acceptance of these terms.  If you do not agree with these
terms, please do not use, install, modify or redistribute this NVIDIA
software.

In consideration of your agreement to abide by the following terms, and
subject to these terms, NVIDIA grants you a personal, non-exclusive
license, under NVIDIA's copyrights in this original NVIDIA software (the
"NVIDIA Software"), to use, reproduce, modify and redistribute the
NVIDIA Software, with or without modifications, in source and/or binary
forms; provided that if you redistribute the NVIDIA Software, you must
retain the copyright notice of NVIDIA, this notice and the following
text and disclaimers in all such redistributions of the NVIDIA Software.
Neither the name, trademarks, service marks nor logos of NVIDIA
Corporation may be used to endorse or promote products derived from the
NVIDIA Software without specific prior written permission from NVIDIA.
Except as expressly stated in this notice, no other rights or licenses
express or implied, are granted by NVIDIA herein, including but not
limited to any patent rights that may be infringed by your derivative
works or by other works in which the NVIDIA Software may be
incorporated. No hardware is licensed hereunder.

THE NVIDIA SOFTWARE IS BEING PROVIDED ON AN "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING WITHOUT LIMITATION, WARRANTIES OR CONDITIONS OF TITLE,
NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR
ITS USE AND OPERATION EITHER ALONE OR IN COMBINATION WITH OTHER
PRODUCTS.

IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT,
INCIDENTAL, EXEMPLARY, CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, LOST PROFITS; PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) OR ARISING IN ANY WAY
OUT OF THE USE, REPRODUCTION, MODIFICATION AND/OR DISTRIBUTION OF THE
NVIDIA SOFTWARE, HOWEVER CAUSED AND WHETHER UNDER THEORY OF CONTRACT,
TORT (INCLUDING NEGLIGENCE), STRICT LIABILITY OR OTHERWISE, EVEN IF
NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
\****************************************************************************/

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <sstream>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <climits>

#include "PpContext.h"
#include "PpTokens.h"

namespace glslang {

// Handle #define
int TPpContext::CPPdefine(TPpToken* ppToken)
{
    MacroSymbol mac;

    // get the macro name
    int token = scanToken(ppToken);
    if (token != PpAtomIdentifier) {
        parseContext.ppError(ppToken->loc, "must be followed by macro name", "#define", "");
        return token;
    }
    if (ppToken->loc.string >= 0) {
        // We are in user code; check for reserved name use:
        parseContext.reservedPpErrorCheck(ppToken->loc, ppToken->name, "#define");
    }

    // save the macro name
    const int defAtom = atomStrings.getAddAtom(ppToken->name);
    TSourceLoc defineLoc = ppToken->loc; // because ppToken might go to the next line before we report errors

    // gather parameters to the macro, between (...)
    token = scanToken(ppToken);
    if (token == '(' && !ppToken->space) {
        mac.functionLike = 1;
        do {
            token = scanToken(ppToken);
            if (mac.args.size() == 0 && token == ')')
                break;
            if (token != PpAtomIdentifier) {
                parseContext.ppError(ppToken->loc, "bad argument", "#define", "");

                return token;
            }
            const int argAtom = atomStrings.getAddAtom(ppToken->name);

            // check for duplication of parameter name
            bool duplicate = false;
            for (size_t a = 0; a < mac.args.size(); ++a) {
                if (mac.args[a] == argAtom) {
                    parseContext.ppError(ppToken->loc, "duplicate macro parameter", "#define", "");
                    duplicate = true;
                    break;
                }
            }
            if (! duplicate)
                mac.args.push_back(argAtom);
            token = scanToken(ppToken);
        } while (token == ',');
        if (token != ')') {
            parseContext.ppError(ppToken->loc, "missing parenthesis", "#define", "");

            return token;
        }

        token = scanToken(ppToken);
    } else if (token != '\n' && token != EndOfInput && !ppToken->space) {
        parseContext.ppWarn(ppToken->loc, "missing space after macro name", "#define", "");

        return token;
    }

    // record the definition of the macro
    while (token != '\n' && token != EndOfInput) {
        mac.body.putToken(token, ppToken);
        token = scanToken(ppToken);
        if (token != '\n' && ppToken->space)
            mac.body.putToken(' ', ppToken);
    }

    // check for duplicate definition
    MacroSymbol* existing = lookupMacroDef(defAtom);
    if (existing != nullptr) {
        if (! existing->undef) {
            // Already defined -- need to make sure they are identical:
            // "Two replacement lists are identical if and only if the
            // preprocessing tokens in both have the same number,
            // ordering, spelling, and white-space separation, where all
            // white-space separations are considered identical."
            if (existing->functionLike != mac.functionLike) {
                parseContext.ppError(defineLoc, "Macro redefined; function-like versus object-like:", "#define",
                    atomStrings.getString(defAtom));
            } else if (existing->args.size() != mac.args.size()) {
                parseContext.ppError(defineLoc, "Macro redefined; different number of arguments:", "#define",
                    atomStrings.getString(defAtom));
            } else {
                if (existing->args != mac.args) {
                    parseContext.ppError(defineLoc, "Macro redefined; different argument names:", "#define",
                       atomStrings.getString(defAtom));
                }
                // set up to compare the two
                existing->body.reset();
                mac.body.reset();
                int newToken;
                bool firstToken = true;
                do {
                    int oldToken;
                    TPpToken oldPpToken;
                    TPpToken newPpToken;
                    oldToken = existing->body.getToken(parseContext, &oldPpToken);
                    newToken = mac.body.getToken(parseContext, &newPpToken);
                    // for the first token, preceding spaces don't matter
                    if (firstToken) {
                        newPpToken.space = oldPpToken.space;
                        firstToken = false;
                    }
                    if (oldToken != newToken || oldPpToken != newPpToken) {
                        parseContext.ppError(defineLoc, "Macro redefined; different substitutions:", "#define",
                            atomStrings.getString(defAtom));
                        break;
                    }
                } while (newToken != EndOfInput);
            }
        }
        *existing = mac;
    } else
        addMacroDef(defAtom, mac);

    return '\n';
}

// Handle #undef
int TPpContext::CPPundef(TPpToken* ppToken)
{
    int token = scanToken(ppToken);
    if (token != PpAtomIdentifier) {
        parseContext.ppError(ppToken->loc, "must be followed by macro name", "#undef", "");

        return token;
    }

    parseContext.reservedPpErrorCheck(ppToken->loc, ppToken->name, "#undef");

    MacroSymbol* macro = lookupMacroDef(atomStrings.getAtom(ppToken->name));
    if (macro != nullptr)
        macro->undef = 1;
    token = scanToken(ppToken);
    if (token != '\n')
        parseContext.ppError(ppToken->loc, "can only be followed by a single macro name", "#undef", "");

    return token;
}

// Handle #else
/* Skip forward to appropriate spot.  This is used both
** to skip to a #endif after seeing an #else, AND to skip to a #else,
** #elif, or #endif after a #if/#ifdef/#ifndef/#elif test was false.
*/
int TPpContext::CPPelse(int matchelse, TPpToken* ppToken)
{
    int depth = 0;
    int token = scanToken(ppToken);

    while (token != EndOfInput) {
        if (token != '#') {
            while (token != '\n' && token != EndOfInput)
                token = scanToken(ppToken);

            if (token == EndOfInput)
                return token;

            token = scanToken(ppToken);
            continue;
        }

        if ((token = scanToken(ppToken)) != PpAtomIdentifier)
            continue;

        int nextAtom = atomStrings.getAtom(ppToken->name);
        if (nextAtom == PpAtomIf || nextAtom == PpAtomIfdef || nextAtom == PpAtomIfndef) {
            depth++;
            if (ifdepth >= maxIfNesting || elsetracker >= maxIfNesting) {
                parseContext.ppError(ppToken->loc, "maximum nesting depth exceeded", "#if/#ifdef/#ifndef", "");
                return EndOfInput;
            } else {
                ifdepth++;
                elsetracker++;
            }
        } else if (nextAtom == PpAtomEndif) {
            token = extraTokenCheck(nextAtom, ppToken, scanToken(ppToken));
            elseSeen[elsetracker] = false;
            --elsetracker;
            if (depth == 0) {
                // found the #endif we are looking for
                if (ifdepth > 0)
                    --ifdepth;
                break;
            }
            --depth;
            --ifdepth;
        } else if (matchelse && depth == 0) {
            if (nextAtom == PpAtomElse) {
                elseSeen[elsetracker] = true;
                token = extraTokenCheck(nextAtom, ppToken, scanToken(ppToken));
                // found the #else we are looking for
                break;
            } else if (nextAtom == PpAtomElif) {
                if (elseSeen[elsetracker])
                    parseContext.ppError(ppToken->loc, "#elif after #else", "#elif", "");
                /* we decrement ifdepth here, because CPPif will increment
                * it and we really want to leave it alone */
                if (ifdepth > 0) {
                    --ifdepth;
                    elseSeen[elsetracker] = false;
                    --elsetracker;
                }

                return CPPif(ppToken);
            }
        } else if (nextAtom == PpAtomElse) {
            if (elseSeen[elsetracker])
                parseContext.ppError(ppToken->loc, "#else after #else", "#else", "");
            else
                elseSeen[elsetracker] = true;
            token = extraTokenCheck(nextAtom, ppToken, scanToken(ppToken));
        } else if (nextAtom == PpAtomElif) {
            if (elseSeen[elsetracker])
                parseContext.ppError(ppToken->loc, "#elif after #else", "#elif", "");
        }
    }

    return token;
}

// Call when there should be no more tokens left on a line.
int TPpContext::extraTokenCheck(int contextAtom, TPpToken* ppToken, int token)
{
    if (token != '\n' && token != EndOfInput) {
        static const char* message = "unexpected tokens following directive";

        const char* label;
        if (contextAtom == PpAtomElse)
            label = "#else";
        else if (contextAtom == PpAtomElif)
            label = "#elif";
        else if (contextAtom == PpAtomEndif)
            label = "#endif";
        else if (contextAtom == PpAtomIf)
            label = "#if";
        else if (contextAtom == PpAtomLine)
            label = "#line";
        else
            label = "";

        if (parseContext.relaxedErrors())
            parseContext.ppWarn(ppToken->loc, message, label, "");
        else
            parseContext.ppError(ppToken->loc, message, label, "");

        while (token != '\n' && token != EndOfInput)
            token = scanToken(ppToken);
    }

    return token;
}

enum eval_prec {
    MIN_PRECEDENCE,
    COND, LOGOR, LOGAND, OR, XOR, AND, EQUAL, RELATION, SHIFT, ADD, MUL, UNARY,
    MAX_PRECEDENCE
};

namespace {

    int op_logor(int a, int b) { return a || b; }
    int op_logand(int a, int b) { return a && b; }
    int op_or(int a, int b) { return a | b; }
    int op_xor(int a, int b) { return a ^ b; }
    int op_and(int a, int b) { return a & b; }
    int op_eq(int a, int b) { return a == b; }
    int op_ne(int a, int b) { return a != b; }
    int op_ge(int a, int b) { return a >= b; }
    int op_le(int a, int b) { return a <= b; }
    int op_gt(int a, int b) { return a > b; }
    int op_lt(int a, int b) { return a < b; }
    int op_shl(int a, int b) { return a << b; }
    int op_shr(int a, int b) { return a >> b; }
    int op_add(int a, int b) { return a + b; }
    int op_sub(int a, int b) { return a - b; }
    int op_mul(int a, int b) { return a * b; }
    int op_div(int a, int b) { return a == INT_MIN && b == -1 ? 0 : a / b; }
    int op_mod(int a, int b) { return a == INT_MIN && b == -1 ? 0 : a % b; }
    int op_pos(int a) { return a; }
    int op_neg(int a) { return -a; }
    int op_cmpl(int a) { return ~a; }
    int op_not(int a) { return !a; }

};

struct TBinop {
    int token, precedence, (*op)(int, int);
} binop[] = {
    { PpAtomOr, LOGOR, op_logor },
    { PpAtomAnd, LOGAND, op_logand },
    { '|', OR, op_or },
    { '^', XOR, op_xor },
    { '&', AND, op_and },
    { PpAtomEQ, EQUAL, op_eq },
    { PpAtomNE, EQUAL, op_ne },
    { '>', RELATION, op_gt },
    { PpAtomGE, RELATION, op_ge },
    { '<', RELATION, op_lt },
    { PpAtomLE, RELATION, op_le },
    { PpAtomLeft, SHIFT, op_shl },
    { PpAtomRight, SHIFT, op_shr },
    { '+', ADD, op_add },
    { '-', ADD, op_sub },
    { '*', MUL, op_mul },
    { '/', MUL, op_div },
    { '%', MUL, op_mod },
};

struct TUnop {
    int token, (*op)(int);
} unop[] = {
    { '+', op_pos },
    { '-', op_neg },
    { '~', op_cmpl },
    { '!', op_not },
};

#define NUM_ELEMENTS(A) (sizeof(A) / sizeof(A[0]))

int TPpContext::eval(int token, int precedence, bool shortCircuit, int& res, bool& err, TPpToken* ppToken)
{
    TSourceLoc loc = ppToken->loc;  // because we sometimes read the newline before reporting the error
    if (token == PpAtomIdentifier) {
        if (strcmp("defined", ppToken->name) == 0) {
            if (! parseContext.isReadingHLSL() && isMacroInput()) {
                if (parseContext.relaxedErrors())
                    parseContext.ppWarn(ppToken->loc, "nonportable when expanded from macros for preprocessor expression",
                                                      "defined", "");
                else
                    parseContext.ppError(ppToken->loc, "cannot use in preprocessor expression when expanded from macros",
                                                       "defined", "");
            }
            bool needclose = 0;
            token = scanToken(ppToken);
            if (token == '(') {
                needclose = true;
                token = scanToken(ppToken);
            }
            if (token != PpAtomIdentifier) {
                parseContext.ppError(loc, "incorrect directive, expected identifier", "preprocessor evaluation", "");
                err = true;
                res = 0;

                return token;
            }

            MacroSymbol* macro = lookupMacroDef(atomStrings.getAtom(ppToken->name));
            res = macro != nullptr ? !macro->undef : 0;
            token = scanToken(ppToken);
            if (needclose) {
                if (token != ')') {
                    parseContext.ppError(loc, "expected ')'", "preprocessor evaluation", "");
                    err = true;
                    res = 0;

                    return token;
                }
                token = scanToken(ppToken);
            }
        } else {
            token = evalToToken(token, shortCircuit, res, err, ppToken);
            return eval(token, precedence, shortCircuit, res, err, ppToken);
        }
    } else if (token == PpAtomConstInt) {
        res = ppToken->ival;
        token = scanToken(ppToken);
    } else if (token == '(') {
        token = scanToken(ppToken);
        token = eval(token, MIN_PRECEDENCE, shortCircuit, res, err, ppToken);
        if (! err) {
            if (token != ')') {
                parseContext.ppError(loc, "expected ')'", "preprocessor evaluation", "");
                err = true;
                res = 0;

                return token;
            }
            token = scanToken(ppToken);
        }
    } else {
        int op = NUM_ELEMENTS(unop) - 1;
        for (; op >= 0; op--) {
            if (unop[op].token == token)
                break;
        }
        if (op >= 0) {
            token = scanToken(ppToken);
            token = eval(token, UNARY, shortCircuit, res, err, ppToken);
            res = unop[op].op(res);
        } else {
            parseContext.ppError(loc, "bad expression", "preprocessor evaluation", "");
            err = true;
            res = 0;

            return token;
        }
    }

    token = evalToToken(token, shortCircuit, res, err, ppToken);

    // Perform evaluation of binary operation, if there is one, otherwise we are done.
    while (! err) {
        if (token == ')' || token == '\n')
            break;
        int op;
        for (op = NUM_ELEMENTS(binop) - 1; op >= 0; op--) {
            if (binop[op].token == token)
                break;
        }
        if (op < 0 || binop[op].precedence <= precedence)
            break;
        int leftSide = res;

        // Setup short-circuiting, needed for ES, unless already in a short circuit.
        // (Once in a short-circuit, can't turn off again, until that whole subexpression is done.
        if (! shortCircuit) {
            if ((token == PpAtomOr  && leftSide == 1) ||
                (token == PpAtomAnd && leftSide == 0))
                shortCircuit = true;
        }

        token = scanToken(ppToken);
        token = eval(token, binop[op].precedence, shortCircuit, res, err, ppToken);

        if (binop[op].op == op_div || binop[op].op == op_mod) {
            if (res == 0) {
                parseContext.ppError(loc, "division by 0", "preprocessor evaluation", "");
                res = 1;
            }
        }
        res = binop[op].op(leftSide, res);
    }

    return token;
}

// Expand macros, skipping empty expansions, to get to the first real token in those expansions.
int TPpContext::evalToToken(int token, bool shortCircuit, int& res, bool& err, TPpToken* ppToken)
{
    while (token == PpAtomIdentifier && strcmp("defined", ppToken->name) != 0) {
        switch (MacroExpand(ppToken, true, false)) {
        case MacroExpandNotStarted:
        case MacroExpandError:
            parseContext.ppError(ppToken->loc, "can't evaluate expression", "preprocessor evaluation", "");
            err = true;
            res = 0;
            break;
        case MacroExpandStarted:
            break;
        case MacroExpandUndef:
            if (! shortCircuit && parseContext.isEsProfile()) {
                const char* message = "undefined macro in expression not allowed in es profile";
                if (parseContext.relaxedErrors())
                    parseContext.ppWarn(ppToken->loc, message, "preprocessor evaluation", ppToken->name);
                else
                    parseContext.ppError(ppToken->loc, message, "preprocessor evaluation", ppToken->name);
            }
            break;
        }
        token = scanToken(ppToken);
        if (err)
            break;
    }

    return token;
}

// Handle #if
int TPpContext::CPPif(TPpToken* ppToken)
{
    int token = scanToken(ppToken);
    if (ifdepth >= maxIfNesting || elsetracker >= maxIfNesting) {
        parseContext.ppError(ppToken->loc, "maximum nesting depth exceeded", "#if", "");
        return EndOfInput;
    } else {
        elsetracker++;
        ifdepth++;
    }
    int res = 0;
    bool err = false;
    token = eval(token, MIN_PRECEDENCE, false, res, err, ppToken);
    token = extraTokenCheck(PpAtomIf, ppToken, token);
    if (!res && !err)
        token = CPPelse(1, ppToken);

    return token;
}

// Handle #ifdef
int TPpContext::CPPifdef(int defined, TPpToken* ppToken)
{
    int token = scanToken(ppToken);
    if (ifdepth > maxIfNesting || elsetracker > maxIfNesting) {
        parseContext.ppError(ppToken->loc, "maximum nesting depth exceeded", "#ifdef", "");
        return EndOfInput;
    } else {
        elsetracker++;
        ifdepth++;
    }

    if (token != PpAtomIdentifier) {
        if (defined)
            parseContext.ppError(ppToken->loc, "must be followed by macro name", "#ifdef", "");
        else
            parseContext.ppError(ppToken->loc, "must be followed by macro name", "#ifndef", "");
    } else {
        MacroSymbol* macro = lookupMacroDef(atomStrings.getAtom(ppToken->name));
        token = scanToken(ppToken);
        if (token != '\n') {
            parseContext.ppError(ppToken->loc, "unexpected tokens following #ifdef directive - expected a newline", "#ifdef", "");
            while (token != '\n' && token != EndOfInput)
                token = scanToken(ppToken);
        }
        if (((macro != nullptr && !macro->undef) ? 1 : 0) != defined)
            token = CPPelse(1, ppToken);
    }

    return token;
}

// Handle #include ...
// TODO: Handle macro expansions for the header name
int TPpContext::CPPinclude(TPpToken* ppToken)
{
    const TSourceLoc directiveLoc = ppToken->loc;
    bool startWithLocalSearch = true; // to additionally include the extra "" paths
    int token;

    // Find the first non-whitespace char after #include
    int ch = getChar();
    while (ch == ' ' || ch == '\t') {
        ch = getChar();
    }
    if (ch == '<') {
        // <header-name> style
        startWithLocalSearch = false;
        token = scanHeaderName(ppToken, '>');
    } else if (ch == '"') {
        // "header-name" style
        token = scanHeaderName(ppToken, '"');
    } else {
        // unexpected, get the full token to generate the error
        ungetChar();
        token = scanToken(ppToken);
    }

    if (token != PpAtomConstString) {
        parseContext.ppError(directiveLoc, "must be followed by a header name", "#include", "");
        return token;
    }

    // Make a copy of the name because it will be overwritten by the next token scan.
    const std::string filename = ppToken->name;

    // See if the directive was well formed
    token = scanToken(ppToken);
    if (token != '\n') {
        if (token == EndOfInput)
            parseContext.ppError(ppToken->loc, "expected newline after header name:", "#include", "%s", filename.c_str());
        else
            parseContext.ppError(ppToken->loc, "extra content after header name:", "#include", "%s", filename.c_str());
        return token;
    }

    // Process well-formed directive

    // Find the inclusion, first look in "Local" ("") paths, if requested,
    // otherwise, only search the "System" (<>) paths.
    TShader::Includer::IncludeResult* res = nullptr;
    if (startWithLocalSearch)
        res = includer.includeLocal(filename.c_str(), currentSourceFile.c_str(), includeStack.size() + 1);
    if (res == nullptr || res->headerName.empty()) {
        includer.releaseInclude(res);
        res = includer.includeSystem(filename.c_str(), currentSourceFile.c_str(), includeStack.size() + 1);
    }

    // Process the results
    if (res != nullptr && !res->headerName.empty()) {
        if (res->headerData != nullptr && res->headerLength > 0) {
            // path for processing one or more tokens from an included header, hand off 'res'
            const bool forNextLine = parseContext.lineDirectiveShouldSetNextLine();
            std::ostringstream prologue;
            std::ostringstream epilogue;
            prologue << "#line " << forNextLine << " " << "\"" << res->headerName << "\"\n";
            epilogue << (res->headerData[res->headerLength - 1] == '\n'? "" : "\n") <<
                "#line " << directiveLoc.line + forNextLine << " " << directiveLoc.getStringNameOrNum() << "\n";
            pushInput(new TokenizableIncludeFile(directiveLoc, prologue.str(), res, epilogue.str(), this));
            parseContext.intermediate.addIncludeText(res->headerName.c_str(), res->headerData, res->headerLength);
            // There's no "current" location anymore.
            parseContext.setCurrentColumn(0);
        } else {
            // things are okay, but there is nothing to process
            includer.releaseInclude(res);
        }
    } else {
        // error path, clean up
        std::string message =
            res != nullptr ? std::string(res->headerData, res->headerLength)
                           : std::string("Could not process include directive");
        parseContext.ppError(directiveLoc, message.c_str(), "#include", "for header name: %s", filename.c_str());
        includer.releaseInclude(res);
    }

    return token;
}

// Handle #line
int TPpContext::CPPline(TPpToken* ppToken)
{
    // "#line must have, after macro substitution, one of the following forms:
    // "#line line
    // "#line line source-string-number"

    int token = scanToken(ppToken);
    const TSourceLoc directiveLoc = ppToken->loc;
    if (token == '\n') {
        parseContext.ppError(ppToken->loc, "must by followed by an integral literal", "#line", "");
        return token;
    }

    int lineRes = 0; // Line number after macro expansion.
    int lineToken = 0;
    bool hasFile = false;
    int fileRes = 0; // Source file number after macro expansion.
    const char* sourceName = nullptr; // Optional source file name.
    bool lineErr = false;
    bool fileErr = false;
    disableEscapeSequences = true;
    token = eval(token, MIN_PRECEDENCE, false, lineRes, lineErr, ppToken);
    disableEscapeSequences = false;
    if (! lineErr) {
        lineToken = lineRes;
        if (token == '\n')
            ++lineRes;

        if (parseContext.lineDirectiveShouldSetNextLine())
            --lineRes;
        parseContext.setCurrentLine(lineRes);

        if (token != '\n') {
#ifndef GLSLANG_WEB
            if (token == PpAtomConstString) {
                parseContext.ppRequireExtensions(directiveLoc, 1, &E_GL_GOOGLE_cpp_style_line_directive, "filename-based #line");
                // We need to save a copy of the string instead of pointing
                // to the name field of the token since the name field
                // will likely be overwritten by the next token scan.
                sourceName = atomStrings.getString(atomStrings.getAddAtom(ppToken->name));
                parseContext.setCurrentSourceName(sourceName);
                hasFile = true;
                token = scanToken(ppToken);
            } else
#endif
            {
                token = eval(token, MIN_PRECEDENCE, false, fileRes, fileErr, ppToken);
                if (! fileErr) {
                    parseContext.setCurrentString(fileRes);
                    hasFile = true;
                }
            }
        }
    }
    if (!fileErr && !lineErr) {
        parseContext.notifyLineDirective(directiveLoc.line, lineToken, hasFile, fileRes, sourceName);
    }
    token = extraTokenCheck(PpAtomLine, ppToken, token);

    return token;
}

// Handle #error
int TPpContext::CPPerror(TPpToken* ppToken)
{
    disableEscapeSequences = true;
    int token = scanToken(ppToken);
    disableEscapeSequences = false;
    std::string message;
    TSourceLoc loc = ppToken->loc;

    while (token != '\n' && token != EndOfInput) {
        if (token == PpAtomConstInt16 || token == PpAtomConstUint16 ||
            token == PpAtomConstInt   || token == PpAtomConstUint   ||
            token == PpAtomConstInt64 || token == PpAtomConstUint64 ||
            token == PpAtomConstFloat16 ||
            token == PpAtomConstFloat || token == PpAtomConstDouble) {
                message.append(ppToken->name);
        } else if (token == PpAtomIdentifier || token == PpAtomConstString) {
            message.append(ppToken->name);
        } else {
            message.append(atomStrings.getString(token));
        }
        message.append(" ");
        token = scanToken(ppToken);
    }
    parseContext.notifyErrorDirective(loc.line, message.c_str());
    // store this msg into the shader's information log..set the Compile Error flag!!!!
    parseContext.ppError(loc, message.c_str(), "#error", "");

    return '\n';
}

// Handle #pragma
int TPpContext::CPPpragma(TPpToken* ppToken)
{
    char SrcStrName[2];
    TVector<TString> tokens;

    TSourceLoc loc = ppToken->loc;  // because we go to the next line before processing
    int token = scanToken(ppToken);
    while (token != '\n' && token != EndOfInput) {
        switch (token) {
        case PpAtomIdentifier:
        case PpAtomConstInt:
        case PpAtomConstUint:
        case PpAtomConstInt64:
        case PpAtomConstUint64:
        case PpAtomConstInt16:
        case PpAtomConstUint16:
        case PpAtomConstFloat:
        case PpAtomConstDouble:
        case PpAtomConstFloat16:
            tokens.push_back(ppToken->name);
            break;
        default:
            SrcStrName[0] = (char)token;
            SrcStrName[1] = '\0';
            tokens.push_back(SrcStrName);
        }
        token = scanToken(ppToken);
    }

    if (token == EndOfInput)
        parseContext.ppError(loc, "directive must end with a newline", "#pragma", "");
    else
        parseContext.handlePragma(loc, tokens);

    return token;
}

// #version: This is just for error checking: the version and profile are decided before preprocessing starts
int TPpContext::CPPversion(TPpToken* ppToken)
{
    int token = scanToken(ppToken);

    if (errorOnVersion || versionSeen) {
        if (parseContext.isReadingHLSL())
            parseContext.ppError(ppToken->loc, "invalid preprocessor command", "#version", "");
        else
            parseContext.ppError(ppToken->loc, "must occur first in shader", "#version", "");
    }
    versionSeen = true;

    if (token == '\n') {
        parseContext.ppError(ppToken->loc, "must be followed by version number", "#version", "");

        return token;
    }

    if (token != PpAtomConstInt)
        parseContext.ppError(ppToken->loc, "must be followed by version number", "#version", "");

    ppToken->ival = atoi(ppToken->name);
    int versionNumber = ppToken->ival;
    int line = ppToken->loc.line;
    token = scanToken(ppToken);

    if (token == '\n') {
        parseContext.notifyVersion(line, versionNumber, nullptr);
        return token;
    } else {
        int profileAtom = atomStrings.getAtom(ppToken->name);
        if (profileAtom != PpAtomCore &&
            profileAtom != PpAtomCompatibility &&
            profileAtom != PpAtomEs)
            parseContext.ppError(ppToken->loc, "bad profile name; use es, core, or compatibility", "#version", "");
        parseContext.notifyVersion(line, versionNumber, ppToken->name);
        token = scanToken(ppToken);

        if (token == '\n')
            return token;
        else
            parseContext.ppError(ppToken->loc, "bad tokens following profile -- expected newline", "#version", "");
    }

    return token;
}

// Handle #extension
int TPpContext::CPPextension(TPpToken* ppToken)
{
    int line = ppToken->loc.line;
    int token = scanToken(ppToken);
    char extensionName[MaxTokenLength + 1];

    if (token=='\n') {
        parseContext.ppError(ppToken->loc, "extension name not specified", "#extension", "");
        return token;
    }

    if (token != PpAtomIdentifier)
        parseContext.ppError(ppToken->loc, "extension name expected", "#extension", "");

    snprintf(extensionName, sizeof(extensionName), "%s", ppToken->name);

    token = scanToken(ppToken);
    if (token != ':') {
        parseContext.ppError(ppToken->loc, "':' missing after extension name", "#extension", "");
        return token;
    }

    token = scanToken(ppToken);
    if (token != PpAtomIdentifier) {
        parseContext.ppError(ppToken->loc, "behavior for extension not specified", "#extension", "");
        return token;
    }

    parseContext.updateExtensionBehavior(line, extensionName, ppToken->name);
    parseContext.notifyExtensionDirective(line, extensionName, ppToken->name);

    token = scanToken(ppToken);
    if (token == '\n')
        return token;
    else
        parseContext.ppError(ppToken->loc,  "extra tokens -- expected newline", "#extension","");

    return token;
}

int TPpContext::readCPPline(TPpToken* ppToken)
{
    int token = scanToken(ppToken);

    if (token == PpAtomIdentifier) {
        switch (atomStrings.getAtom(ppToken->name)) {
        case PpAtomDefine:
            token = CPPdefine(ppToken);
            break;
        case PpAtomElse:
            if (elseSeen[elsetracker])
                parseContext.ppError(ppToken->loc, "#else after #else", "#else", "");
            elseSeen[elsetracker] = true;
            if (ifdepth == 0)
                parseContext.ppError(ppToken->loc, "mismatched statements", "#else", "");
            token = extraTokenCheck(PpAtomElse, ppToken, scanToken(ppToken));
            token = CPPelse(0, ppToken);
            break;
        case PpAtomElif:
            if (ifdepth == 0)
                parseContext.ppError(ppToken->loc, "mismatched statements", "#elif", "");
            if (elseSeen[elsetracker])
                parseContext.ppError(ppToken->loc, "#elif after #else", "#elif", "");
            // this token is really a dont care, but we still need to eat the tokens
            token = scanToken(ppToken);
            while (token != '\n' && token != EndOfInput)
                token = scanToken(ppToken);
            token = CPPelse(0, ppToken);
            break;
        case PpAtomEndif:
            if (ifdepth == 0)
                parseContext.ppError(ppToken->loc, "mismatched statements", "#endif", "");
            else {
                elseSeen[elsetracker] = false;
                --elsetracker;
                --ifdepth;
            }
            token = extraTokenCheck(PpAtomEndif, ppToken, scanToken(ppToken));
            break;
        case PpAtomIf:
            token = CPPif(ppToken);
            break;
        case PpAtomIfdef:
            token = CPPifdef(1, ppToken);
            break;
        case PpAtomIfndef:
            token = CPPifdef(0, ppToken);
            break;
        case PpAtomLine:
            token = CPPline(ppToken);
            break;
#ifndef GLSLANG_WEB
        case PpAtomInclude:
            if(!parseContext.isReadingHLSL()) {
                parseContext.ppRequireExtensions(ppToken->loc, 1, &E_GL_GOOGLE_include_directive, "#include");
            }
            token = CPPinclude(ppToken);
            break;
        case PpAtomPragma:
            token = CPPpragma(ppToken);
            break;
#endif
        case PpAtomUndef:
            token = CPPundef(ppToken);
            break;
        case PpAtomError:
            token = CPPerror(ppToken);
            break;
        case PpAtomVersion:
            token = CPPversion(ppToken);
            break;
        case PpAtomExtension:
            token = CPPextension(ppToken);
            break;
        default:
            parseContext.ppError(ppToken->loc, "invalid directive:", "#", ppToken->name);
            break;
        }
    } else if (token != '\n' && token != EndOfInput)
        parseContext.ppError(ppToken->loc, "invalid directive", "#", "");

    while (token != '\n' && token != EndOfInput)
        token = scanToken(ppToken);

    return token;
}

// Context-dependent parsing of a #include <header-name>.
// Assumes no macro expansions etc. are being done; the name is just on the current input.
// Always creates a name and returns PpAtomicConstString, unless we run out of input.
int TPpContext::scanHeaderName(TPpToken* ppToken, char delimit)
{
    bool tooLong = false;

    if (inputStack.empty())
        return EndOfInput;

    int len = 0;
    ppToken->name[0] = '\0';
    do {
        int ch = inputStack.back()->getch();

        // done yet?
        if (ch == delimit) {
            ppToken->name[len] = '\0';
            if (tooLong)
                parseContext.ppError(ppToken->loc, "header name too long", "", "");
            return PpAtomConstString;
        } else if (ch == EndOfInput)
            return EndOfInput;

        // found a character to expand the name with
        if (len < MaxTokenLength)
            ppToken->name[len++] = (char)ch;
        else
            tooLong = true;
    } while (true);
}

// Macro-expand a macro argument 'arg' to create 'expandedArg'.
// Does not replace 'arg'.
// Returns nullptr if no expanded argument is created.
TPpContext::TokenStream* TPpContext::PrescanMacroArg(TokenStream& arg, TPpToken* ppToken, bool newLineOkay)
{
    // expand the argument
    TokenStream* expandedArg = new TokenStream;
    pushInput(new tMarkerInput(this));
    pushTokenStreamInput(arg);
    int token;
    while ((token = scanToken(ppToken)) != tMarkerInput::marker && token != EndOfInput) {
        token = tokenPaste(token, *ppToken);
        if (token == PpAtomIdentifier) {
            switch (MacroExpand(ppToken, false, newLineOkay)) {
            case MacroExpandNotStarted:
                break;
            case MacroExpandError:
                // toss the rest of the pushed-input argument by scanning until tMarkerInput
                while ((token = scanToken(ppToken)) != tMarkerInput::marker && token != EndOfInput)
                    ;
                break;
            case MacroExpandStarted:
            case MacroExpandUndef:
                continue;
            }
        }
        if (token == tMarkerInput::marker || token == EndOfInput)
            break;
        expandedArg->putToken(token, ppToken);
    }

    if (token != tMarkerInput::marker) {
        // Error, or MacroExpand ate the marker, so had bad input, recover
        delete expandedArg;
        expandedArg = nullptr;
    }

    return expandedArg;
}

//
// Return the next token for a macro expansion, handling macro arguments,
// whose semantics are dependent on being adjacent to ##.
//
int TPpContext::tMacroInput::scan(TPpToken* ppToken)
{
    int token;
    do {
        token = mac->body.getToken(pp->parseContext, ppToken);
    } while (token == ' ');  // handle white space in macro

    // Hash operators basically turn off a round of macro substitution
    // (the round done on the argument before the round done on the RHS of the
    // macro definition):
    //
    // "A parameter in the replacement list, unless preceded by a # or ##
    // preprocessing token or followed by a ## preprocessing token (see below),
    // is replaced by the corresponding argument after all macros contained
    // therein have been expanded."
    //
    // "If, in the replacement list, a parameter is immediately preceded or
    // followed by a ## preprocessing token, the parameter is replaced by the
    // corresponding argument's preprocessing token sequence."

    bool pasting = false;
    if (postpaste) {
        // don't expand next token
        pasting = true;
        postpaste = false;
    }

    if (prepaste) {
        // already know we should be on a ##, verify
        assert(token == PpAtomPaste);
        prepaste = false;
        postpaste = true;
    }

    // see if are preceding a ##
    if (mac->body.peekUntokenizedPasting()) {
        prepaste = true;
        pasting = true;
    }

    // HLSL does expand macros before concatenation
    if (pasting && pp->parseContext.isReadingHLSL())
        pasting = false;

    // TODO: preprocessor:  properly handle whitespace (or lack of it) between tokens when expanding
    if (token == PpAtomIdentifier) {
        int i;
        for (i = (int)mac->args.size() - 1; i >= 0; i--)
            if (strcmp(pp->atomStrings.getString(mac->args[i]), ppToken->name) == 0)
                break;
        if (i >= 0) {
            TokenStream* arg = expandedArgs[i];
            if (arg == nullptr || pasting)
                arg = args[i];
            pp->pushTokenStreamInput(*arg, prepaste);

            return pp->scanToken(ppToken);
        }
    }

    if (token == EndOfInput)
        mac->busy = 0;

    return token;
}

// return a textual zero, for scanning a macro that was never defined
int TPpContext::tZeroInput::scan(TPpToken* ppToken)
{
    if (done)
        return EndOfInput;

    ppToken->name[0] = '0';
    ppToken->name[1] = 0;
    ppToken->ival = 0;
    ppToken->space = false;
    done = true;

    return PpAtomConstInt;
}

//
// Check a token to see if it is a macro that should be expanded:
// - If it is, and defined, push a tInput that will produce the appropriate
//   expansion and return MacroExpandStarted.
// - If it is, but undefined, and expandUndef is requested, push a tInput
//   that will expand to 0 and return MacroExpandUndef.
// - Otherwise, there is no expansion, and there are two cases:
//   * It might be okay there is no expansion, and no specific error was
//     detected. Returns MacroExpandNotStarted.
//   * The expansion was started, but could not be completed, due to an error
//     that cannot be recovered from. Returns MacroExpandError.
//
MacroExpandResult TPpContext::MacroExpand(TPpToken* ppToken, bool expandUndef, bool newLineOkay)
{
    ppToken->space = false;
    int macroAtom = atomStrings.getAtom(ppToken->name);
    switch (macroAtom) {
    case PpAtomLineMacro:
        ppToken->ival = parseContext.getCurrentLoc().line;
        snprintf(ppToken->name, sizeof(ppToken->name), "%d", ppToken->ival);
        UngetToken(PpAtomConstInt, ppToken);
        return MacroExpandStarted;

    case PpAtomFileMacro: {
        if (parseContext.getCurrentLoc().name)
            parseContext.ppRequireExtensions(ppToken->loc, 1, &E_GL_GOOGLE_cpp_style_line_directive, "filename-based __FILE__");
        ppToken->ival = parseContext.getCurrentLoc().string;
        snprintf(ppToken->name, sizeof(ppToken->name), "%s", ppToken->loc.getStringNameOrNum().c_str());
        UngetToken(PpAtomConstInt, ppToken);
        return MacroExpandStarted;
    }

    case PpAtomVersionMacro:
        ppToken->ival = parseContext.version;
        snprintf(ppToken->name, sizeof(ppToken->name), "%d", ppToken->ival);
        UngetToken(PpAtomConstInt, ppToken);
        return MacroExpandStarted;

    default:
        break;
    }

    MacroSymbol* macro = macroAtom == 0 ? nullptr : lookupMacroDef(macroAtom);

    // no recursive expansions
    if (macro != nullptr && macro->busy)
        return MacroExpandNotStarted;

    // not expanding undefined macros
    if ((macro == nullptr || macro->undef) && ! expandUndef)
        return MacroExpandNotStarted;

    // 0 is the value of an undefined macro
    if ((macro == nullptr || macro->undef) && expandUndef) {
        pushInput(new tZeroInput(this));
        return MacroExpandUndef;
    }

    tMacroInput *in = new tMacroInput(this);

    TSourceLoc loc = ppToken->loc;  // in case we go to the next line before discovering the error
    in->mac = macro;
    if (macro->functionLike) {
        // We don't know yet if this will be a successful call of a
        // function-like macro; need to look for a '(', but without trashing
        // the passed in ppToken, until we know we are no longer speculative.
        TPpToken parenToken;
        int token = scanToken(&parenToken);
        if (newLineOkay) {
            while (token == '\n')
                token = scanToken(&parenToken);
        }
        if (token != '(') {
            // Function-like macro called with object-like syntax: okay, don't expand.
            // (We ate exactly one token that might not be white space; put it back.
            UngetToken(token, &parenToken);
            delete in;
            return MacroExpandNotStarted;
        }
        in->args.resize(in->mac->args.size());
        for (size_t i = 0; i < in->mac->args.size(); i++)
            in->args[i] = new TokenStream;
        in->expandedArgs.resize(in->mac->args.size());
        for (size_t i = 0; i < in->mac->args.size(); i++)
            in->expandedArgs[i] = nullptr;
        size_t arg = 0;
        bool tokenRecorded = false;
        do {
            TVector<char> nestStack;
            while (true) {
                token = scanToken(ppToken);
                if (token == EndOfInput || token == tMarkerInput::marker) {
                    parseContext.ppError(loc, "End of input in macro", "macro expansion", atomStrings.getString(macroAtom));
                    delete in;
                    return MacroExpandError;
                }
                if (token == '\n') {
                    if (! newLineOkay) {
                        parseContext.ppError(loc, "End of line in macro substitution:", "macro expansion", atomStrings.getString(macroAtom));
                        delete in;
                        return MacroExpandError;
                    }
                    continue;
                }
                if (token == '#') {
                    parseContext.ppError(ppToken->loc, "unexpected '#'", "macro expansion", atomStrings.getString(macroAtom));
                    delete in;
                    return MacroExpandError;
                }
                if (in->mac->args.size() == 0 && token != ')')
                    break;
                if (nestStack.size() == 0 && (token == ',' || token == ')'))
                    break;
                if (token == '(')
                    nestStack.push_back(')');
                else if (token == '{' && parseContext.isReadingHLSL())
                    nestStack.push_back('}');
                else if (nestStack.size() > 0 && token == nestStack.back())
                    nestStack.pop_back();
                in->args[arg]->putToken(token, ppToken);
                tokenRecorded = true;
            }
            // end of single argument scan

            if (token == ')') {
                // closing paren of call
                if (in->mac->args.size() == 1 && !tokenRecorded)
                    break;
                arg++;
                break;
            }
            arg++;
        } while (arg < in->mac->args.size());
        // end of all arguments scan

        if (arg < in->mac->args.size())
            parseContext.ppError(loc, "Too few args in Macro", "macro expansion", atomStrings.getString(macroAtom));
        else if (token != ')') {
            // Error recover code; find end of call, if possible
            int depth = 0;
            while (token != EndOfInput && (depth > 0 || token != ')')) {
                if (token == ')' || token == '}')
                    depth--;
                token = scanToken(ppToken);
                if (token == '(' || token == '{')
                    depth++;
            }

            if (token == EndOfInput) {
                parseContext.ppError(loc, "End of input in macro", "macro expansion", atomStrings.getString(macroAtom));
                delete in;
                return MacroExpandError;
            }
            parseContext.ppError(loc, "Too many args in macro", "macro expansion", atomStrings.getString(macroAtom));
        }

        // We need both expanded and non-expanded forms of the argument, for whether or
        // not token pasting will be applied later when the argument is consumed next to ##.
        for (size_t i = 0; i < in->mac->args.size(); i++)
            in->expandedArgs[i] = PrescanMacroArg(*in->args[i], ppToken, newLineOkay);
    }

    pushInput(in);
    macro->busy = 1;
    macro->body.reset();

    return MacroExpandStarted;
}

} // end namespace glslang
