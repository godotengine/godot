//
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

#ifndef PPCONTEXT_H
#define PPCONTEXT_H

#include <stack>
#include <unordered_map>
#include <sstream>

#include "../ParseHelper.h"
#include "PpTokens.h"

namespace glslang {

class TPpToken {
public:
    TPpToken() { clear(); }
    void clear()
    {
        space = false;
        i64val = 0;
        loc.init();
        name[0] = 0;
        fullyExpanded = false;
    }

    // Used for comparing macro definitions, so checks what is relevant for that.
    bool operator==(const TPpToken& right) const
    {
        return space == right.space &&
               ival == right.ival && dval == right.dval && i64val == right.i64val &&
               strncmp(name, right.name, MaxTokenLength) == 0;
    }
    bool operator!=(const TPpToken& right) const { return ! operator==(right); }

    TSourceLoc loc;
    // True if a space (for white space or a removed comment) should also be
    // recognized, in front of the token returned:
    bool space;

    bool fullyExpanded;
    // Numeric value of the token:
    union {
        int ival;
        double dval;
        long long i64val;
    };
    // Text string of the token:
    char name[MaxTokenLength + 1];
};

class TStringAtomMap {
//
// Implementation is in PpAtom.cpp
//
// Maintain a bi-directional mapping between relevant preprocessor strings and
// "atoms" which a unique integers (small, contiguous, not hash-like) per string.
//
public:
    TStringAtomMap();

    // Map string -> atom.
    // Return 0 if no existing string.
    int getAtom(const char* s) const
    {
        auto it = atomMap.find(s);
        return it == atomMap.end() ? 0 : it->second;
    }

    // Map a new or existing string -> atom, inventing a new atom if necessary.
    int getAddAtom(const char* s)
    {
        int atom = getAtom(s);
        if (atom == 0) {
            atom = nextAtom++;
            addAtomFixed(s, atom);
        }
        return atom;
    }

    // Map atom -> string.
    const char* getString(int atom) const { return stringMap[atom]->c_str(); }

protected:
    TStringAtomMap(TStringAtomMap&);
    TStringAtomMap& operator=(TStringAtomMap&);

    TUnorderedMap<TString, int> atomMap;
    TVector<const TString*> stringMap;    // these point into the TString in atomMap
    int nextAtom;

    // Bad source characters can lead to bad atoms, so gracefully handle those by
    // pre-filling the table with them (to avoid if tests later).
    TString badToken;

    // Add bi-directional mappings:
    //  - string -> atom
    //  - atom -> string
    void addAtomFixed(const char* s, int atom)
    {
        auto it = atomMap.insert(std::pair<TString, int>(s, atom)).first;
        if (stringMap.size() < (size_t)atom + 1)
            stringMap.resize(atom + 100, &badToken);
        stringMap[atom] = &it->first;
    }
};

class TInputScanner;

enum MacroExpandResult {
    MacroExpandNotStarted, // macro not expanded, which might not be an error
    MacroExpandError,      // a clear error occurred while expanding, no expansion
    MacroExpandStarted,    // macro expansion process has started
    MacroExpandUndef       // macro is undefined and will be expanded
};

// This class is the result of turning a huge pile of C code communicating through globals
// into a class.  This was done to allowing instancing to attain thread safety.
// Don't expect too much in terms of OO design.
class TPpContext {
public:
    TPpContext(TParseContextBase&, const std::string& rootFileName, TShader::Includer&);
    virtual ~TPpContext();

    void setPreamble(const char* preamble, size_t length);

    int tokenize(TPpToken& ppToken);
    int tokenPaste(int token, TPpToken&);

    class tInput {
    public:
        tInput(TPpContext* p) : done(false), pp(p) { }
        virtual ~tInput() { }

        virtual int scan(TPpToken*) = 0;
        virtual int getch() = 0;
        virtual void ungetch() = 0;
        virtual bool peekPasting() { return false; }             // true when about to see ##
        virtual bool peekContinuedPasting(int) { return false; } // true when non-spaced tokens can paste
        virtual bool endOfReplacementList() { return false; } // true when at the end of a macro replacement list (RHS of #define)
        virtual bool isMacroInput() { return false; }
        virtual bool isStringInput() { return false; }

        // Will be called when we start reading tokens from this instance
        virtual void notifyActivated() {}
        // Will be called when we do not read tokens from this instance anymore
        virtual void notifyDeleted() {}
    protected:
        bool done;
        TPpContext* pp;
    };

    void setInput(TInputScanner& input, bool versionWillBeError);

    void pushInput(tInput* in)
    {
        inputStack.push_back(in);
        in->notifyActivated();
    }
    void popInput()
    {
        inputStack.back()->notifyDeleted();
        delete inputStack.back();
        inputStack.pop_back();
    }

    //
    // From PpTokens.cpp
    //

    // Capture the needed parts of a token stream for macro recording/playback.
    class TokenStream {
    public:
        // Manage a stream of these 'Token', which capture the relevant parts
        // of a TPpToken, plus its atom.
        class Token {
        public:
            Token(int atom, const TPpToken& ppToken) : 
                atom(atom),
                space(ppToken.space),
                i64val(ppToken.i64val),
                name(ppToken.name) { }
            int get(TPpToken& ppToken)
            {
                ppToken.clear();
                ppToken.space = space;
                ppToken.i64val = i64val;
                snprintf(ppToken.name, sizeof(ppToken.name), "%s", name.c_str());
                return atom;
            }
            bool isAtom(int a) const { return atom == a; }
            int getAtom() const { return atom; }
            bool nonSpaced() const { return !space; }
        protected:
            Token() {}
            int atom;
            bool space;        // did a space precede the token?
            long long i64val;
            TString name;
        };

        TokenStream() : currentPos(0) { }

        void putToken(int token, TPpToken* ppToken);
        bool peekToken(int atom) { return !atEnd() && stream[currentPos].isAtom(atom); }
        bool peekContinuedPasting(int atom)
        {
            // This is basically necessary because, for example, the PP
            // tokenizer only accepts valid numeric-literals plus suffixes, so
            // separates numeric-literals plus bad suffix into two tokens, which
            // should get both pasted together as one token when token pasting.
            //
            // The following code is a bit more generalized than the above example.
            if (!atEnd() && atom == PpAtomIdentifier && stream[currentPos].nonSpaced()) {
                switch(stream[currentPos].getAtom()) {
                    case PpAtomConstInt:
                    case PpAtomConstUint:
                    case PpAtomConstInt64:
                    case PpAtomConstUint64:
                    case PpAtomConstInt16:
                    case PpAtomConstUint16:
                    case PpAtomConstFloat:
                    case PpAtomConstDouble:
                    case PpAtomConstFloat16:
                    case PpAtomConstString:
                    case PpAtomIdentifier:
                        return true;
                    default:
                        break;
                }
            }

            return false;
        }
        int getToken(TParseContextBase&, TPpToken*);
        bool atEnd() { return currentPos >= stream.size(); }
        bool peekTokenizedPasting(bool lastTokenPastes);
        void reset() { currentPos = 0; }

    protected:
        TVector<Token> stream;
        size_t currentPos;
    };

    //
    // From Pp.cpp
    //

    struct MacroSymbol {
        MacroSymbol() : functionLike(0), busy(0), undef(0) { }
        TVector<int> args;
        TokenStream body;
        unsigned functionLike : 1;  // 0 means object-like, 1 means function-like
        unsigned busy         : 1;
        unsigned undef        : 1;
    };

    typedef TMap<int, MacroSymbol> TSymbolMap;
    TSymbolMap macroDefs;  // map atoms to macro definitions
    MacroSymbol* lookupMacroDef(int atom)
    {
        auto existingMacroIt = macroDefs.find(atom);
        return (existingMacroIt == macroDefs.end()) ? nullptr : &(existingMacroIt->second);
    }
    void addMacroDef(int atom, MacroSymbol& macroDef) { macroDefs[atom] = macroDef; }

protected:
    TPpContext(TPpContext&);
    TPpContext& operator=(TPpContext&);

    TStringAtomMap atomStrings;
    char*   preamble;               // string to parse, all before line 1 of string 0, it is 0 if no preamble
    int     preambleLength;
    char**  strings;                // official strings of shader, starting a string 0 line 1
    size_t* lengths;
    int     numStrings;             // how many official strings there are
    int     currentString;          // which string we're currently parsing  (-1 for preamble)

    // Scanner data:
    int previous_token;
    TParseContextBase& parseContext;
    std::vector<int> lastLineTokens;
    std::vector<TSourceLoc> lastLineTokenLocs;
    // Get the next token from *stack* of input sources, popping input sources
    // that are out of tokens, down until an input source is found that has a token.
    // Return EndOfInput when there are no more tokens to be found by doing this.
    int scanToken(TPpToken* ppToken)
    {
        int token = EndOfInput;

        while (! inputStack.empty()) {
            token = inputStack.back()->scan(ppToken);
            if (token != EndOfInput || inputStack.empty())
                break;
            popInput();
        }
        if (!inputStack.empty() && inputStack.back()->isStringInput() && !inElseSkip) {
            if (token == '\n') {
                lastLineTokens.clear();
                lastLineTokenLocs.clear();
            } else {
                lastLineTokens.push_back(token);
                lastLineTokenLocs.push_back(ppToken->loc);
            }
        }
        return token;
    }
    int  getChar() { return inputStack.back()->getch(); }
    void ungetChar() { inputStack.back()->ungetch(); }
    bool peekPasting() { return !inputStack.empty() && inputStack.back()->peekPasting(); }
    bool peekContinuedPasting(int a)
    {
        return !inputStack.empty() && inputStack.back()->peekContinuedPasting(a);
    }
    bool endOfReplacementList() { return inputStack.empty() || inputStack.back()->endOfReplacementList(); }
    bool isMacroInput() { return inputStack.size() > 0 && inputStack.back()->isMacroInput(); }

    static const int maxIfNesting = 65;

    int ifdepth;                  // current #if-#else-#endif nesting in the cpp.c file (pre-processor)
    bool elseSeen[maxIfNesting];  // Keep a track of whether an else has been seen at a particular depth
    int elsetracker;              // #if-#else and #endif constructs...Counter.

    class tMacroInput : public tInput {
    public:
        tMacroInput(TPpContext* pp) : tInput(pp), prepaste(false), postpaste(false) { }
        virtual ~tMacroInput()
        {
            for (size_t i = 0; i < args.size(); ++i)
                delete args[i];
            for (size_t i = 0; i < expandedArgs.size(); ++i)
                delete expandedArgs[i];
        }

        virtual int scan(TPpToken*) override;
        virtual int getch() override { assert(0); return EndOfInput; }
        virtual void ungetch() override { assert(0); }
        bool peekPasting() override { return prepaste; }
        bool peekContinuedPasting(int a) override { return mac->body.peekContinuedPasting(a); }
        bool endOfReplacementList() override { return mac->body.atEnd(); }
        bool isMacroInput() override { return true; }

        MacroSymbol *mac;
        TVector<TokenStream*> args;
        TVector<TokenStream*> expandedArgs;

    protected:
        bool prepaste;         // true if we are just before ##
        bool postpaste;        // true if we are right after ##
    };

    class tMarkerInput : public tInput {
    public:
        tMarkerInput(TPpContext* pp) : tInput(pp) { }
        virtual int scan(TPpToken*) override
        {
            if (done)
                return EndOfInput;
            done = true;

            return marker;
        }
        virtual int getch() override { assert(0); return EndOfInput; }
        virtual void ungetch() override { assert(0); }
        static const int marker = -3;
    };

    class tStringifyLevelInput : public tInput {
        int what;
        tStringifyLevelInput(TPpContext* pp) : tInput(pp) { }
    public:
        static tStringifyLevelInput popMarker(TPpContext* pp)
        {
            tStringifyLevelInput sl(pp);
            sl.what = POP;
            return sl;
        }

        static tStringifyLevelInput pushMarker(TPpContext* pp)
        {
            tStringifyLevelInput sl(pp);
            sl.what = PUSH;
            return sl;
        }

        int scan(TPpToken*) override
        {
            if (done)
                return EndOfInput;
            done = true;

            return what;
        }
        virtual int getch() override { assert(0); return EndOfInput; }
        virtual void ungetch() override { assert(0); }
        static const int PUSH = -4;
        static const int POP = -5;
    };

    class tZeroInput : public tInput {
    public:
        tZeroInput(TPpContext* pp) : tInput(pp) { }
        virtual int scan(TPpToken*) override;
        virtual int getch() override { assert(0); return EndOfInput; }
        virtual void ungetch() override { assert(0); }
    };

    std::vector<tInput*> inputStack;
    bool errorOnVersion;
    bool versionSeen;

    //
    // from Pp.cpp
    //

    // Used to obtain #include content.
    TShader::Includer& includer;

    int CPPdefine(TPpToken * ppToken);
    int CPPundef(TPpToken * ppToken);
    int CPPelse(int matchelse, TPpToken * ppToken);
    int extraTokenCheck(int atom, TPpToken* ppToken, int token);
    int eval(int token, int precedence, bool shortCircuit, int& res, bool& err, TPpToken * ppToken);
    int evalToToken(int token, bool shortCircuit, int& res, bool& err, TPpToken * ppToken);
    int CPPif (TPpToken * ppToken);
    int CPPifdef(int defined, TPpToken * ppToken);
    int CPPinclude(TPpToken * ppToken);
    int CPPline(TPpToken * ppToken);
    int CPPerror(TPpToken * ppToken);
    int CPPpragma(TPpToken * ppToken);
    int CPPversion(TPpToken * ppToken);
    int CPPextension(TPpToken * ppToken);
    int readCPPline(TPpToken * ppToken);
    int scanHeaderName(TPpToken* ppToken, char delimit);
    TokenStream* PrescanMacroArg(TokenStream&, TPpToken*, bool newLineOkay);
    MacroExpandResult MacroExpand(TPpToken* ppToken, bool expandUndef, bool newLineOkay);

    //
    // From PpTokens.cpp
    //
    void pushTokenStreamInput(TokenStream&, bool pasting = false, bool expanded = false);
    void UngetToken(int token, TPpToken*);

    class tTokenInput : public tInput {
    public:
        tTokenInput(TPpContext* pp, TokenStream* t, bool prepasting, bool expanded) :
            tInput(pp),
            tokens(t),
            lastTokenPastes(prepasting),
            preExpanded(expanded) { }
        virtual int scan(TPpToken *ppToken) override {
            int token = tokens->getToken(pp->parseContext, ppToken);
            ppToken->fullyExpanded = preExpanded;
            if (tokens->atEnd() && token == PpAtomIdentifier) {
                int macroAtom = pp->atomStrings.getAtom(ppToken->name);
                MacroSymbol* macro = macroAtom == 0 ? nullptr : pp->lookupMacroDef(macroAtom);
                if (macro && macro->functionLike)
                    ppToken->fullyExpanded = false;
            }
            return token;
        }
        virtual int getch() override { assert(0); return EndOfInput; }
        virtual void ungetch() override { assert(0); }
        virtual bool peekPasting() override { return tokens->peekTokenizedPasting(lastTokenPastes); }
        bool peekContinuedPasting(int a) override { return tokens->peekContinuedPasting(a); }
    protected:
        TokenStream* tokens;
        bool lastTokenPastes; // true if the last token in the input is to be pasted, rather than consumed as a token
        bool preExpanded;
    };

    class tUngotTokenInput : public tInput {
    public:
        tUngotTokenInput(TPpContext* pp, int t, TPpToken* p) : tInput(pp), token(t), lval(*p) { }
        virtual int scan(TPpToken *) override;
        virtual int getch() override { assert(0); return EndOfInput; }
        virtual void ungetch() override { assert(0); }
    protected:
        int token;
        TPpToken lval;
    };

    //
    // From PpScanner.cpp
    //
    class tStringInput : public tInput {
    public:
        tStringInput(TPpContext* pp, TInputScanner& i) : tInput(pp), input(&i) { }
        virtual int scan(TPpToken*) override;
        bool isStringInput() override { return true; }
        // Scanner used to get source stream characters.
        //  - Escaped newlines are handled here, invisibly to the caller.
        //  - All forms of newline are handled, and turned into just a '\n'.
        int getch() override
        {
            int ch = input->get();

            if (ch == '\\') {
                // Move past escaped newlines, as many as sequentially exist
                do {
                    if (input->peek() == '\r' || input->peek() == '\n') {
                        bool allowed = pp->parseContext.lineContinuationCheck(input->getSourceLoc(), pp->inComment);
                        if (! allowed && pp->inComment)
                            return '\\';

                        // escape one newline now
                        ch = input->get();
                        int nextch = input->get();
                        if (ch == '\r' && nextch == '\n')
                            ch = input->get();
                        else
                            ch = nextch;
                    } else
                        return '\\';
                } while (ch == '\\');
            }

            // handle any non-escaped newline
            if (ch == '\r' || ch == '\n') {
                if (ch == '\r' && input->peek() == '\n')
                    input->get();
                return '\n';
            }

            return ch;
        }

        // Scanner used to backup the source stream characters.  Newlines are
        // handled here, invisibly to the caller, meaning have to undo exactly
        // what getch() above does (e.g., don't leave things in the middle of a
        // sequence of escaped newlines).
        void ungetch() override
        {
            input->unget();

            do {
                int ch = input->peek();
                if (ch == '\r' || ch == '\n') {
                    if (ch == '\n') {
                        // correct for two-character newline
                        input->unget();
                        if (input->peek() != '\r')
                            input->get();
                    }
                    // now in front of a complete newline, move past an escape character
                    input->unget();
                    if (input->peek() == '\\')
                        input->unget();
                    else {
                        input->get();
                        break;
                    }
                } else
                    break;
            } while (true);
        }

    protected:
        TInputScanner* input;
    };

    // Holds a reference to included file data, as well as a
    // prologue and an epilogue string. This can be scanned using the tInput
    // interface and acts as a single source string.
    class TokenizableIncludeFile : public tInput {
    public:
        // Copies prologue and epilogue. The includedFile must remain valid
        // until this TokenizableIncludeFile is no longer used.
        TokenizableIncludeFile(const TSourceLoc& startLoc,
                          const std::string& prologue,
                          TShader::Includer::IncludeResult* includedFile,
                          const std::string& epilogue,
                          TPpContext* pp)
            : tInput(pp),
              prologue_(prologue),
              epilogue_(epilogue),
              includedFile_(includedFile),
              scanner(3, strings, lengths, nullptr, 0, 0, true),
              prevScanner(nullptr),
              stringInput(pp, scanner)
        {
              strings[0] = prologue_.data();
              strings[1] = includedFile_->headerData;
              strings[2] = epilogue_.data();

              lengths[0] = prologue_.size();
              lengths[1] = includedFile_->headerLength;
              lengths[2] = epilogue_.size();

              scanner.setLine(startLoc.line);
              scanner.setString(startLoc.string);

              scanner.setFile(startLoc.getFilenameStr(), 0);
              scanner.setFile(startLoc.getFilenameStr(), 1);
              scanner.setFile(startLoc.getFilenameStr(), 2);
        }

        // tInput methods:
        int scan(TPpToken* t) override { return stringInput.scan(t); }
        int getch() override { return stringInput.getch(); }
        void ungetch() override { stringInput.ungetch(); }

        void notifyActivated() override
        {
            prevScanner = pp->parseContext.getScanner();
            pp->parseContext.setScanner(&scanner);
            pp->push_include(includedFile_);
        }

        void notifyDeleted() override
        {
            pp->parseContext.setScanner(prevScanner);
            pp->pop_include();
        }

    private:
        TokenizableIncludeFile& operator=(const TokenizableIncludeFile&);

        // Stores the prologue for this string.
        const std::string prologue_;

        // Stores the epilogue for this string.
        const std::string epilogue_;

        // Points to the IncludeResult that this TokenizableIncludeFile represents.
        TShader::Includer::IncludeResult* includedFile_;

        // Will point to prologue_, includedFile_->headerData and epilogue_
        // This is passed to scanner constructor.
        // These do not own the storage and it must remain valid until this
        // object has been destroyed.
        const char* strings[3];
        // Length of str_, passed to scanner constructor.
        size_t lengths[3];
        // Scans over str_.
        TInputScanner scanner;
        // The previous effective scanner before the scanner in this instance
        // has been activated.
        TInputScanner* prevScanner;
        // Delegate object implementing the tInput interface.
        tStringInput stringInput;
    };

    int ScanFromString(char* s);
    void missingEndifCheck();
    int lFloatConst(int len, int ch, TPpToken* ppToken);
    int characterLiteral(TPpToken* ppToken);

    void push_include(TShader::Includer::IncludeResult* result)
    {
        currentSourceFile = result->headerName;
        includeStack.push(result);
    }

    void pop_include()
    {
        TShader::Includer::IncludeResult* include = includeStack.top();
        includeStack.pop();
        includer.releaseInclude(include);
        if (includeStack.empty()) {
            currentSourceFile = rootFileName;
        } else {
            currentSourceFile = includeStack.top()->headerName;
        }
    }

    bool inComment;
    std::string rootFileName;
    std::stack<TShader::Includer::IncludeResult*> includeStack;
    std::string currentSourceFile;

    std::istringstream strtodStream;
    bool disableEscapeSequences;
    // True if we're skipping a section enclosed by #if/#ifdef/#elif/#else which was evaluated to
    // be inactive, e.g. #if 0
    bool inElseSkip;
};

} // end namespace glslang

#endif  // PPCONTEXT_H
