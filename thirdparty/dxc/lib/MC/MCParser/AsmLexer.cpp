//===- AsmLexer.cpp - Lexer for Assembly Files ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class implements the lexer for assembly files.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCParser/AsmLexer.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SMLoc.h"
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
using namespace llvm;

AsmLexer::AsmLexer(const MCAsmInfo &MAI) : MAI(MAI) {
  CurPtr = nullptr;
  isAtStartOfLine = true;
  AllowAtInIdentifier = !StringRef(MAI.getCommentString()).startswith("@");
}

AsmLexer::~AsmLexer() {
}

void AsmLexer::setBuffer(StringRef Buf, const char *ptr) {
  CurBuf = Buf;

  if (ptr)
    CurPtr = ptr;
  else
    CurPtr = CurBuf.begin();

  TokStart = nullptr;
}

/// ReturnError - Set the error to the specified string at the specified
/// location.  This is defined to always return AsmToken::Error.
AsmToken AsmLexer::ReturnError(const char *Loc, const std::string &Msg) {
  SetError(SMLoc::getFromPointer(Loc), Msg);

  return AsmToken(AsmToken::Error, StringRef(Loc, 0));
}

int AsmLexer::getNextChar() {
  char CurChar = *CurPtr++;
  switch (CurChar) {
  default:
    return (unsigned char)CurChar;
  case 0:
    // A nul character in the stream is either the end of the current buffer or
    // a random nul in the file.  Disambiguate that here.
    if (CurPtr - 1 != CurBuf.end())
      return 0;  // Just whitespace.

    // Otherwise, return end of file.
    --CurPtr;  // Another call to lex will return EOF again.
    return EOF;
  }
}

/// LexFloatLiteral: [0-9]*[.][0-9]*([eE][+-]?[0-9]*)?
///
/// The leading integral digit sequence and dot should have already been
/// consumed, some or all of the fractional digit sequence *can* have been
/// consumed.
AsmToken AsmLexer::LexFloatLiteral() {
  // Skip the fractional digit sequence.
  while (isdigit(*CurPtr))
    ++CurPtr;

  // Check for exponent; we intentionally accept a slighlty wider set of
  // literals here and rely on the upstream client to reject invalid ones (e.g.,
  // "1e+").
  if (*CurPtr == 'e' || *CurPtr == 'E') {
    ++CurPtr;
    if (*CurPtr == '-' || *CurPtr == '+')
      ++CurPtr;
    while (isdigit(*CurPtr))
      ++CurPtr;
  }

  return AsmToken(AsmToken::Real,
                  StringRef(TokStart, CurPtr - TokStart));
}

/// LexHexFloatLiteral matches essentially (.[0-9a-fA-F]*)?[pP][+-]?[0-9a-fA-F]+
/// while making sure there are enough actual digits around for the constant to
/// be valid.
///
/// The leading "0x[0-9a-fA-F]*" (i.e. integer part) has already been consumed
/// before we get here.
AsmToken AsmLexer::LexHexFloatLiteral(bool NoIntDigits) {
  assert((*CurPtr == 'p' || *CurPtr == 'P' || *CurPtr == '.') &&
         "unexpected parse state in floating hex");
  bool NoFracDigits = true;

  // Skip the fractional part if there is one
  if (*CurPtr == '.') {
    ++CurPtr;

    const char *FracStart = CurPtr;
    while (isxdigit(*CurPtr))
      ++CurPtr;

    NoFracDigits = CurPtr == FracStart;
  }

  if (NoIntDigits && NoFracDigits)
    return ReturnError(TokStart, "invalid hexadecimal floating-point constant: "
                                 "expected at least one significand digit");

  // Make sure we do have some kind of proper exponent part
  if (*CurPtr != 'p' && *CurPtr != 'P')
    return ReturnError(TokStart, "invalid hexadecimal floating-point constant: "
                                 "expected exponent part 'p'");
  ++CurPtr;

  if (*CurPtr == '+' || *CurPtr == '-')
    ++CurPtr;

  // N.b. exponent digits are *not* hex
  const char *ExpStart = CurPtr;
  while (isdigit(*CurPtr))
    ++CurPtr;

  if (CurPtr == ExpStart)
    return ReturnError(TokStart, "invalid hexadecimal floating-point constant: "
                                 "expected at least one exponent digit");

  return AsmToken(AsmToken::Real, StringRef(TokStart, CurPtr - TokStart));
}

/// LexIdentifier: [a-zA-Z_.][a-zA-Z0-9_$.@?]*
static bool IsIdentifierChar(char c, bool AllowAt) {
  return isalnum(c) || c == '_' || c == '$' || c == '.' ||
         (c == '@' && AllowAt) || c == '?';
}
AsmToken AsmLexer::LexIdentifier() {
  // Check for floating point literals.
  if (CurPtr[-1] == '.' && isdigit(*CurPtr)) {
    // Disambiguate a .1243foo identifier from a floating literal.
    while (isdigit(*CurPtr))
      ++CurPtr;
    if (*CurPtr == 'e' || *CurPtr == 'E' ||
        !IsIdentifierChar(*CurPtr, AllowAtInIdentifier))
      return LexFloatLiteral();
  }

  while (IsIdentifierChar(*CurPtr, AllowAtInIdentifier))
    ++CurPtr;

  // Handle . as a special case.
  if (CurPtr == TokStart+1 && TokStart[0] == '.')
    return AsmToken(AsmToken::Dot, StringRef(TokStart, 1));

  return AsmToken(AsmToken::Identifier, StringRef(TokStart, CurPtr - TokStart));
}

/// LexSlash: Slash: /
///           C-Style Comment: /* ... */
AsmToken AsmLexer::LexSlash() {
  switch (*CurPtr) {
  case '*': break; // C style comment.
  case '/': return ++CurPtr, LexLineComment();
  default:  return AsmToken(AsmToken::Slash, StringRef(CurPtr-1, 1));
  }

  // C Style comment.
  ++CurPtr;  // skip the star.
  while (1) {
    int CurChar = getNextChar();
    switch (CurChar) {
    case EOF:
      return ReturnError(TokStart, "unterminated comment");
    case '*':
      // End of the comment?
      if (CurPtr[0] != '/') break;

      ++CurPtr;   // End the */.
      return LexToken();
    }
  }
}

/// LexLineComment: Comment: #[^\n]*
///                        : //[^\n]*
AsmToken AsmLexer::LexLineComment() {
  // FIXME: This is broken if we happen to a comment at the end of a file, which
  // was .included, and which doesn't end with a newline.
  int CurChar = getNextChar();
  while (CurChar != '\n' && CurChar != '\r' && CurChar != EOF)
    CurChar = getNextChar();

  if (CurChar == EOF)
    return AsmToken(AsmToken::Eof, StringRef(TokStart, 0));
  return AsmToken(AsmToken::EndOfStatement, StringRef(TokStart, 0));
}

static void SkipIgnoredIntegerSuffix(const char *&CurPtr) {
  // Skip ULL, UL, U, L and LL suffices.
  if (CurPtr[0] == 'U')
    ++CurPtr;
  if (CurPtr[0] == 'L')
    ++CurPtr;
  if (CurPtr[0] == 'L')
    ++CurPtr;
}

// Look ahead to search for first non-hex digit, if it's [hH], then we treat the
// integer as a hexadecimal, possibly with leading zeroes.
static unsigned doLookAhead(const char *&CurPtr, unsigned DefaultRadix) {
  const char *FirstHex = nullptr;
  const char *LookAhead = CurPtr;
  while (1) {
    if (isdigit(*LookAhead)) {
      ++LookAhead;
    } else if (isxdigit(*LookAhead)) {
      if (!FirstHex)
        FirstHex = LookAhead;
      ++LookAhead;
    } else {
      break;
    }
  }
  bool isHex = *LookAhead == 'h' || *LookAhead == 'H';
  CurPtr = isHex || !FirstHex ? LookAhead : FirstHex;
  if (isHex)
    return 16;
  return DefaultRadix;
}

static AsmToken intToken(StringRef Ref, APInt &Value)
{
  if (Value.isIntN(64))
    return AsmToken(AsmToken::Integer, Ref, Value);
  return AsmToken(AsmToken::BigNum, Ref, Value);
}

/// LexDigit: First character is [0-9].
///   Local Label: [0-9][:]
///   Forward/Backward Label: [0-9][fb]
///   Binary integer: 0b[01]+
///   Octal integer: 0[0-7]+
///   Hex integer: 0x[0-9a-fA-F]+ or [0x]?[0-9][0-9a-fA-F]*[hH]
///   Decimal integer: [1-9][0-9]*
AsmToken AsmLexer::LexDigit() {
  // Decimal integer: [1-9][0-9]*
  if (CurPtr[-1] != '0' || CurPtr[0] == '.') {
    unsigned Radix = doLookAhead(CurPtr, 10);
    bool isHex = Radix == 16;
    // Check for floating point literals.
    if (!isHex && (*CurPtr == '.' || *CurPtr == 'e')) {
      ++CurPtr;
      return LexFloatLiteral();
    }

    StringRef Result(TokStart, CurPtr - TokStart);

    APInt Value(128, 0, true);
    if (Result.getAsInteger(Radix, Value))
      return ReturnError(TokStart, !isHex ? "invalid decimal number" :
                           "invalid hexdecimal number");

    // Consume the [bB][hH].
    if (Radix == 2 || Radix == 16)
      ++CurPtr;

    // The darwin/x86 (and x86-64) assembler accepts and ignores type
    // suffices on integer literals.
    SkipIgnoredIntegerSuffix(CurPtr);

    return intToken(Result, Value);
  }

  if (*CurPtr == 'b') {
    ++CurPtr;
    // See if we actually have "0b" as part of something like "jmp 0b\n"
    if (!isdigit(CurPtr[0])) {
      --CurPtr;
      StringRef Result(TokStart, CurPtr - TokStart);
      return AsmToken(AsmToken::Integer, Result, 0);
    }
    const char *NumStart = CurPtr;
    while (CurPtr[0] == '0' || CurPtr[0] == '1')
      ++CurPtr;

    // Requires at least one binary digit.
    if (CurPtr == NumStart)
      return ReturnError(TokStart, "invalid binary number");

    StringRef Result(TokStart, CurPtr - TokStart);

    APInt Value(128, 0, true);
    if (Result.substr(2).getAsInteger(2, Value))
      return ReturnError(TokStart, "invalid binary number");

    // The darwin/x86 (and x86-64) assembler accepts and ignores ULL and LL
    // suffixes on integer literals.
    SkipIgnoredIntegerSuffix(CurPtr);

    return intToken(Result, Value);
  }

  if (*CurPtr == 'x') {
    ++CurPtr;
    const char *NumStart = CurPtr;
    while (isxdigit(CurPtr[0]))
      ++CurPtr;

    // "0x.0p0" is valid, and "0x0p0" (but not "0xp0" for example, which will be
    // diagnosed by LexHexFloatLiteral).
    if (CurPtr[0] == '.' || CurPtr[0] == 'p' || CurPtr[0] == 'P')
      return LexHexFloatLiteral(NumStart == CurPtr);

    // Otherwise requires at least one hex digit.
    if (CurPtr == NumStart)
      return ReturnError(CurPtr-2, "invalid hexadecimal number");

    APInt Result(128, 0);
    if (StringRef(TokStart, CurPtr - TokStart).getAsInteger(0, Result))
      return ReturnError(TokStart, "invalid hexadecimal number");

    // Consume the optional [hH].
    if (*CurPtr == 'h' || *CurPtr == 'H')
      ++CurPtr;

    // The darwin/x86 (and x86-64) assembler accepts and ignores ULL and LL
    // suffixes on integer literals.
    SkipIgnoredIntegerSuffix(CurPtr);

    return intToken(StringRef(TokStart, CurPtr - TokStart), Result);
  }

  // Either octal or hexadecimal.
  APInt Value(128, 0, true);
  unsigned Radix = doLookAhead(CurPtr, 8);
  bool isHex = Radix == 16;
  StringRef Result(TokStart, CurPtr - TokStart);
  if (Result.getAsInteger(Radix, Value))
    return ReturnError(TokStart, !isHex ? "invalid octal number" :
                       "invalid hexdecimal number");

  // Consume the [hH].
  if (Radix == 16)
    ++CurPtr;

  // The darwin/x86 (and x86-64) assembler accepts and ignores ULL and LL
  // suffixes on integer literals.
  SkipIgnoredIntegerSuffix(CurPtr);

  return intToken(Result, Value);
}

/// LexSingleQuote: Integer: 'b'
AsmToken AsmLexer::LexSingleQuote() {
  int CurChar = getNextChar();

  if (CurChar == '\\')
    CurChar = getNextChar();

  if (CurChar == EOF)
    return ReturnError(TokStart, "unterminated single quote");

  CurChar = getNextChar();

  if (CurChar != '\'')
    return ReturnError(TokStart, "single quote way too long");

  // The idea here being that 'c' is basically just an integral
  // constant.
  StringRef Res = StringRef(TokStart,CurPtr - TokStart);
  long long Value;

  if (Res.startswith("\'\\")) {
    char theChar = Res[2];
    switch (theChar) {
      default: Value = theChar; break;
      case '\'': Value = '\''; break;
      case 't': Value = '\t'; break;
      case 'n': Value = '\n'; break;
      case 'b': Value = '\b'; break;
    }
  } else
    Value = TokStart[1];

  return AsmToken(AsmToken::Integer, Res, Value);
}


/// LexQuote: String: "..."
AsmToken AsmLexer::LexQuote() {
  int CurChar = getNextChar();
  // TODO: does gas allow multiline string constants?
  while (CurChar != '"') {
    if (CurChar == '\\') {
      // Allow \", etc.
      CurChar = getNextChar();
    }

    if (CurChar == EOF)
      return ReturnError(TokStart, "unterminated string constant");

    CurChar = getNextChar();
  }

  return AsmToken(AsmToken::String, StringRef(TokStart, CurPtr - TokStart));
}

StringRef AsmLexer::LexUntilEndOfStatement() {
  TokStart = CurPtr;

  while (!isAtStartOfComment(CurPtr) &&     // Start of line comment.
         !isAtStatementSeparator(CurPtr) && // End of statement marker.
         *CurPtr != '\n' && *CurPtr != '\r' &&
         (*CurPtr != 0 || CurPtr != CurBuf.end())) {
    ++CurPtr;
  }
  return StringRef(TokStart, CurPtr-TokStart);
}

StringRef AsmLexer::LexUntilEndOfLine() {
  TokStart = CurPtr;

  while (*CurPtr != '\n' && *CurPtr != '\r' &&
         (*CurPtr != 0 || CurPtr != CurBuf.end())) {
    ++CurPtr;
  }
  return StringRef(TokStart, CurPtr-TokStart);
}

const AsmToken AsmLexer::peekTok(bool ShouldSkipSpace) {
  const char *SavedTokStart = TokStart;
  const char *SavedCurPtr = CurPtr;
  bool SavedAtStartOfLine = isAtStartOfLine;
  bool SavedSkipSpace = SkipSpace;

  std::string SavedErr = getErr();
  SMLoc SavedErrLoc = getErrLoc();

  SkipSpace = ShouldSkipSpace;
  AsmToken Token = LexToken();

  SetError(SavedErrLoc, SavedErr);

  SkipSpace = SavedSkipSpace;
  isAtStartOfLine = SavedAtStartOfLine;
  CurPtr = SavedCurPtr;
  TokStart = SavedTokStart;

  return Token;
}

bool AsmLexer::isAtStartOfComment(const char *Ptr) {
  const char *CommentString = MAI.getCommentString();

  if (CommentString[1] == '\0')
    return CommentString[0] == Ptr[0];

  // FIXME: special case for the bogus "##" comment string in X86MCAsmInfoDarwin
  if (CommentString[1] == '#')
    return CommentString[0] == Ptr[0];

  return strncmp(Ptr, CommentString, strlen(CommentString)) == 0;
}

bool AsmLexer::isAtStatementSeparator(const char *Ptr) {
  return strncmp(Ptr, MAI.getSeparatorString(),
                 strlen(MAI.getSeparatorString())) == 0;
}

AsmToken AsmLexer::LexToken() {
  TokStart = CurPtr;
  // This always consumes at least one character.
  int CurChar = getNextChar();

  if (isAtStartOfComment(TokStart)) {
    // If this comment starts with a '#', then return the Hash token and let
    // the assembler parser see if it can be parsed as a cpp line filename
    // comment. We do this only if we are at the start of a line.
    if (CurChar == '#' && isAtStartOfLine)
      return AsmToken(AsmToken::Hash, StringRef(TokStart, 1));
    isAtStartOfLine = true;
    return LexLineComment();
  }
  if (isAtStatementSeparator(TokStart)) {
    CurPtr += strlen(MAI.getSeparatorString()) - 1;
    return AsmToken(AsmToken::EndOfStatement,
                    StringRef(TokStart, strlen(MAI.getSeparatorString())));
  }

  // If we're missing a newline at EOF, make sure we still get an
  // EndOfStatement token before the Eof token.
  if (CurChar == EOF && !isAtStartOfLine) {
    isAtStartOfLine = true;
    return AsmToken(AsmToken::EndOfStatement, StringRef(TokStart, 1));
  }

  isAtStartOfLine = false;
  switch (CurChar) {
  default:
    // Handle identifier: [a-zA-Z_.][a-zA-Z0-9_$.@]*
    if (isalpha(CurChar) || CurChar == '_' || CurChar == '.')
      return LexIdentifier();

    // Unknown character, emit an error.
    return ReturnError(TokStart, "invalid character in input");
  case EOF: return AsmToken(AsmToken::Eof, StringRef(TokStart, 0));
  case 0:
  case ' ':
  case '\t':
    if (SkipSpace) {
      // Ignore whitespace.
      return LexToken();
    } else {
      int len = 1;
      while (*CurPtr==' ' || *CurPtr=='\t') {
        CurPtr++;
        len++;
      }
      return AsmToken(AsmToken::Space, StringRef(TokStart, len));
    }
  case '\n': // FALL THROUGH.
  case '\r':
    isAtStartOfLine = true;
    return AsmToken(AsmToken::EndOfStatement, StringRef(TokStart, 1));
  case ':': return AsmToken(AsmToken::Colon, StringRef(TokStart, 1));
  case '+': return AsmToken(AsmToken::Plus, StringRef(TokStart, 1));
  case '-': return AsmToken(AsmToken::Minus, StringRef(TokStart, 1));
  case '~': return AsmToken(AsmToken::Tilde, StringRef(TokStart, 1));
  case '(': return AsmToken(AsmToken::LParen, StringRef(TokStart, 1));
  case ')': return AsmToken(AsmToken::RParen, StringRef(TokStart, 1));
  case '[': return AsmToken(AsmToken::LBrac, StringRef(TokStart, 1));
  case ']': return AsmToken(AsmToken::RBrac, StringRef(TokStart, 1));
  case '{': return AsmToken(AsmToken::LCurly, StringRef(TokStart, 1));
  case '}': return AsmToken(AsmToken::RCurly, StringRef(TokStart, 1));
  case '*': return AsmToken(AsmToken::Star, StringRef(TokStart, 1));
  case ',': return AsmToken(AsmToken::Comma, StringRef(TokStart, 1));
  case '$': return AsmToken(AsmToken::Dollar, StringRef(TokStart, 1));
  case '@': return AsmToken(AsmToken::At, StringRef(TokStart, 1));
  case '\\': return AsmToken(AsmToken::BackSlash, StringRef(TokStart, 1));
  case '=':
    if (*CurPtr == '=')
      return ++CurPtr, AsmToken(AsmToken::EqualEqual, StringRef(TokStart, 2));
    return AsmToken(AsmToken::Equal, StringRef(TokStart, 1));
  case '|':
    if (*CurPtr == '|')
      return ++CurPtr, AsmToken(AsmToken::PipePipe, StringRef(TokStart, 2));
    return AsmToken(AsmToken::Pipe, StringRef(TokStart, 1));
  case '^': return AsmToken(AsmToken::Caret, StringRef(TokStart, 1));
  case '&':
    if (*CurPtr == '&')
      return ++CurPtr, AsmToken(AsmToken::AmpAmp, StringRef(TokStart, 2));
    return AsmToken(AsmToken::Amp, StringRef(TokStart, 1));
  case '!':
    if (*CurPtr == '=')
      return ++CurPtr, AsmToken(AsmToken::ExclaimEqual, StringRef(TokStart, 2));
    return AsmToken(AsmToken::Exclaim, StringRef(TokStart, 1));
  case '%': return AsmToken(AsmToken::Percent, StringRef(TokStart, 1));
  case '/': return LexSlash();
  case '#': return AsmToken(AsmToken::Hash, StringRef(TokStart, 1));
  case '\'': return LexSingleQuote();
  case '"': return LexQuote();
  case '0': case '1': case '2': case '3': case '4':
  case '5': case '6': case '7': case '8': case '9':
    return LexDigit();
  case '<':
    switch (*CurPtr) {
    case '<': return ++CurPtr, AsmToken(AsmToken::LessLess,
                                        StringRef(TokStart, 2));
    case '=': return ++CurPtr, AsmToken(AsmToken::LessEqual,
                                        StringRef(TokStart, 2));
    case '>': return ++CurPtr, AsmToken(AsmToken::LessGreater,
                                        StringRef(TokStart, 2));
    default: return AsmToken(AsmToken::Less, StringRef(TokStart, 1));
    }
  case '>':
    switch (*CurPtr) {
    case '>': return ++CurPtr, AsmToken(AsmToken::GreaterGreater,
                                        StringRef(TokStart, 2));
    case '=': return ++CurPtr, AsmToken(AsmToken::GreaterEqual,
                                        StringRef(TokStart, 2));
    default: return AsmToken(AsmToken::Greater, StringRef(TokStart, 1));
    }

  // TODO: Quoted identifiers (objc methods etc)
  // local labels: [0-9][:]
  // Forward/backward labels: [0-9][fb]
  // Integers, fp constants, character constants.
  }
}
