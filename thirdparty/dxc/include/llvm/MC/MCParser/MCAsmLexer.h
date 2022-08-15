//===-- llvm/MC/MCAsmLexer.h - Abstract Asm Lexer Interface -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCPARSER_MCASMLEXER_H
#define LLVM_MC_MCPARSER_MCASMLEXER_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/SMLoc.h"

namespace llvm {

/// Target independent representation for an assembler token.
class AsmToken {
public:
  enum TokenKind {
    // Markers
    Eof, Error,

    // String values.
    Identifier,
    String,

    // Integer values.
    Integer,
    BigNum, // larger than 64 bits

    // Real values.
    Real,

    // No-value.
    EndOfStatement,
    Colon,
    Space,
    Plus, Minus, Tilde,
    Slash,    // '/'
    BackSlash, // '\'
    LParen, RParen, LBrac, RBrac, LCurly, RCurly,
    Star, Dot, Comma, Dollar, Equal, EqualEqual,

    Pipe, PipePipe, Caret,
    Amp, AmpAmp, Exclaim, ExclaimEqual, Percent, Hash,
    Less, LessEqual, LessLess, LessGreater,
    Greater, GreaterEqual, GreaterGreater, At
  };

private:
  TokenKind Kind;

  /// A reference to the entire token contents; this is always a pointer into
  /// a memory buffer owned by the source manager.
  StringRef Str;

  APInt IntVal;

public:
  AsmToken() {}
  AsmToken(TokenKind Kind, StringRef Str, APInt IntVal)
      : Kind(Kind), Str(Str), IntVal(IntVal) {}
  AsmToken(TokenKind Kind, StringRef Str, int64_t IntVal = 0)
      : Kind(Kind), Str(Str), IntVal(64, IntVal, true) {}

  TokenKind getKind() const { return Kind; }
  bool is(TokenKind K) const { return Kind == K; }
  bool isNot(TokenKind K) const { return Kind != K; }

  SMLoc getLoc() const;
  SMLoc getEndLoc() const;
  SMRange getLocRange() const;

  /// Get the contents of a string token (without quotes).
  StringRef getStringContents() const {
    assert(Kind == String && "This token isn't a string!");
    return Str.slice(1, Str.size() - 1);
  }

  /// Get the identifier string for the current token, which should be an
  /// identifier or a string. This gets the portion of the string which should
  /// be used as the identifier, e.g., it does not include the quotes on
  /// strings.
  StringRef getIdentifier() const {
    if (Kind == Identifier)
      return getString();
    return getStringContents();
  }

  /// Get the string for the current token, this includes all characters (for
  /// example, the quotes on strings) in the token.
  ///
  /// The returned StringRef points into the source manager's memory buffer, and
  /// is safe to store across calls to Lex().
  StringRef getString() const { return Str; }

  // FIXME: Don't compute this in advance, it makes every token larger, and is
  // also not generally what we want (it is nicer for recovery etc. to lex 123br
  // as a single token, then diagnose as an invalid number).
  int64_t getIntVal() const {
    assert(Kind == Integer && "This token isn't an integer!");
    return IntVal.getZExtValue();
  }

  APInt getAPIntVal() const {
    assert((Kind == Integer || Kind == BigNum) &&
           "This token isn't an integer!");
    return IntVal;
  }
};

/// Generic assembler lexer interface, for use by target specific assembly
/// lexers.
class MCAsmLexer {
  /// The current token, stored in the base class for faster access.
  AsmToken CurTok;

  /// The location and description of the current error
  SMLoc ErrLoc;
  std::string Err;

  MCAsmLexer(const MCAsmLexer &) = delete;
  void operator=(const MCAsmLexer &) = delete;
protected: // Can only create subclasses.
  const char *TokStart;
  bool SkipSpace;
  bool AllowAtInIdentifier;

  MCAsmLexer();

  virtual AsmToken LexToken() = 0;

  void SetError(const SMLoc &errLoc, const std::string &err) {
    ErrLoc = errLoc;
    Err = err;
  }

public:
  virtual ~MCAsmLexer();

  /// Consume the next token from the input stream and return it.
  ///
  /// The lexer will continuosly return the end-of-file token once the end of
  /// the main input file has been reached.
  const AsmToken &Lex() {
    return CurTok = LexToken();
  }

  virtual StringRef LexUntilEndOfStatement() = 0;

  /// Get the current source location.
  SMLoc getLoc() const;

  /// Get the current (last) lexed token.
  const AsmToken &getTok() const {
    return CurTok;
  }

  /// Look ahead at the next token to be lexed.
  virtual const AsmToken peekTok(bool ShouldSkipSpace = true) = 0;

  /// Get the current error location
  const SMLoc &getErrLoc() {
    return ErrLoc;
  }

  /// Get the current error string
  const std::string &getErr() {
    return Err;
  }

  /// Get the kind of current token.
  AsmToken::TokenKind getKind() const { return CurTok.getKind(); }

  /// Check if the current token has kind \p K.
  bool is(AsmToken::TokenKind K) const { return CurTok.is(K); }

  /// Check if the current token has kind \p K.
  bool isNot(AsmToken::TokenKind K) const { return CurTok.isNot(K); }

  /// Set whether spaces should be ignored by the lexer
  void setSkipSpace(bool val) { SkipSpace = val; }

  bool getAllowAtInIdentifier() { return AllowAtInIdentifier; }
  void setAllowAtInIdentifier(bool v) { AllowAtInIdentifier = v; }
};

} // End llvm namespace

#endif
