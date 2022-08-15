//===- MILexer.cpp - Machine instructions lexer implementation ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the lexing of machine instructions.
//
//===----------------------------------------------------------------------===//

#include "MILexer.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include <cctype>

using namespace llvm;

namespace {

/// This class provides a way to iterate and get characters from the source
/// string.
class Cursor {
  const char *Ptr;
  const char *End;

public:
  Cursor(NoneType) : Ptr(nullptr), End(nullptr) {}

  explicit Cursor(StringRef Str) {
    Ptr = Str.data();
    End = Ptr + Str.size();
  }

  bool isEOF() const { return Ptr == End; }

  char peek(int I = 0) const { return End - Ptr <= I ? 0 : Ptr[I]; }

  void advance(unsigned I = 1) { Ptr += I; }

  StringRef remaining() const { return StringRef(Ptr, End - Ptr); }

  StringRef upto(Cursor C) const {
    assert(C.Ptr >= Ptr && C.Ptr <= End);
    return StringRef(Ptr, C.Ptr - Ptr);
  }

  StringRef::iterator location() const { return Ptr; }

  operator bool() const { return Ptr != nullptr; }
};

} // end anonymous namespace

/// Skip the leading whitespace characters and return the updated cursor.
static Cursor skipWhitespace(Cursor C) {
  while (isspace(C.peek()))
    C.advance();
  return C;
}

static bool isIdentifierChar(char C) {
  return isalpha(C) || isdigit(C) || C == '_' || C == '-' || C == '.';
}

static MIToken::TokenKind getIdentifierKind(StringRef Identifier) {
  return StringSwitch<MIToken::TokenKind>(Identifier)
      .Case("_", MIToken::underscore)
      .Case("implicit", MIToken::kw_implicit)
      .Case("implicit-def", MIToken::kw_implicit_define)
      .Case("dead", MIToken::kw_dead)
      .Case("killed", MIToken::kw_killed)
      .Case("undef", MIToken::kw_undef)
      .Default(MIToken::Identifier);
}

static Cursor maybeLexIdentifier(Cursor C, MIToken &Token) {
  if (!isalpha(C.peek()) && C.peek() != '_')
    return None;
  auto Range = C;
  while (isIdentifierChar(C.peek()))
    C.advance();
  auto Identifier = Range.upto(C);
  Token = MIToken(getIdentifierKind(Identifier), Identifier);
  return C;
}

static Cursor maybeLexMachineBasicBlock(
    Cursor C, MIToken &Token,
    function_ref<void(StringRef::iterator Loc, const Twine &)> ErrorCallback) {
  if (!C.remaining().startswith("%bb."))
    return None;
  auto Range = C;
  C.advance(4); // Skip '%bb.'
  if (!isdigit(C.peek())) {
    Token = MIToken(MIToken::Error, C.remaining());
    ErrorCallback(C.location(), "expected a number after '%bb.'");
    return C;
  }
  auto NumberRange = C;
  while (isdigit(C.peek()))
    C.advance();
  StringRef Number = NumberRange.upto(C);
  unsigned StringOffset = 4 + Number.size(); // Drop '%bb.<id>'
  if (C.peek() == '.') {
    C.advance(); // Skip '.'
    ++StringOffset;
    while (isIdentifierChar(C.peek()))
      C.advance();
  }
  Token = MIToken(MIToken::MachineBasicBlock, Range.upto(C), APSInt(Number),
                  StringOffset);
  return C;
}

static Cursor lexVirtualRegister(Cursor C, MIToken &Token) {
  auto Range = C;
  C.advance(); // Skip '%'
  auto NumberRange = C;
  while (isdigit(C.peek()))
    C.advance();
  Token = MIToken(MIToken::VirtualRegister, Range.upto(C),
                  APSInt(NumberRange.upto(C)));
  return C;
}

static Cursor maybeLexRegister(Cursor C, MIToken &Token) {
  if (C.peek() != '%')
    return None;
  if (isdigit(C.peek(1)))
    return lexVirtualRegister(C, Token);
  auto Range = C;
  C.advance(); // Skip '%'
  while (isIdentifierChar(C.peek()))
    C.advance();
  Token = MIToken(MIToken::NamedRegister, Range.upto(C),
                  /*StringOffset=*/1); // Drop the '%'
  return C;
}

static Cursor maybeLexGlobalValue(Cursor C, MIToken &Token) {
  if (C.peek() != '@')
    return None;
  auto Range = C;
  C.advance(); // Skip the '@'
  // TODO: add support for quoted names.
  if (!isdigit(C.peek())) {
    while (isIdentifierChar(C.peek()))
      C.advance();
    Token = MIToken(MIToken::NamedGlobalValue, Range.upto(C),
                    /*StringOffset=*/1); // Drop the '@'
    return C;
  }
  auto NumberRange = C;
  while (isdigit(C.peek()))
    C.advance();
  Token =
      MIToken(MIToken::GlobalValue, Range.upto(C), APSInt(NumberRange.upto(C)));
  return C;
}

static Cursor maybeLexIntegerLiteral(Cursor C, MIToken &Token) {
  if (!isdigit(C.peek()) && (C.peek() != '-' || !isdigit(C.peek(1))))
    return None;
  auto Range = C;
  C.advance();
  while (isdigit(C.peek()))
    C.advance();
  StringRef StrVal = Range.upto(C);
  Token = MIToken(MIToken::IntegerLiteral, StrVal, APSInt(StrVal));
  return C;
}

static MIToken::TokenKind symbolToken(char C) {
  switch (C) {
  case ',':
    return MIToken::comma;
  case '=':
    return MIToken::equal;
  case ':':
    return MIToken::colon;
  default:
    return MIToken::Error;
  }
}

static Cursor maybeLexSymbol(Cursor C, MIToken &Token) {
  auto Kind = symbolToken(C.peek());
  if (Kind == MIToken::Error)
    return None;
  auto Range = C;
  C.advance();
  Token = MIToken(Kind, Range.upto(C));
  return C;
}

StringRef llvm::lexMIToken(
    StringRef Source, MIToken &Token,
    function_ref<void(StringRef::iterator Loc, const Twine &)> ErrorCallback) {
  auto C = skipWhitespace(Cursor(Source));
  if (C.isEOF()) {
    Token = MIToken(MIToken::Eof, C.remaining());
    return C.remaining();
  }

  if (Cursor R = maybeLexIdentifier(C, Token))
    return R.remaining();
  if (Cursor R = maybeLexMachineBasicBlock(C, Token, ErrorCallback))
    return R.remaining();
  if (Cursor R = maybeLexRegister(C, Token))
    return R.remaining();
  if (Cursor R = maybeLexGlobalValue(C, Token))
    return R.remaining();
  if (Cursor R = maybeLexIntegerLiteral(C, Token))
    return R.remaining();
  if (Cursor R = maybeLexSymbol(C, Token))
    return R.remaining();

  Token = MIToken(MIToken::Error, C.remaining());
  ErrorCallback(C.location(),
                Twine("unexpected character '") + Twine(C.peek()) + "'");
  return C.remaining();
}
