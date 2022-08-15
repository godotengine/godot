//===- MILexer.h - Lexer for machine instructions -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the function that lexes the machine instruction source
// string.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_MIRPARSER_MILEXER_H
#define LLVM_LIB_CODEGEN_MIRPARSER_MILEXER_H

#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/STLExtras.h"
#include <functional>

namespace llvm {

class Twine;

/// A token produced by the machine instruction lexer.
struct MIToken {
  enum TokenKind {
    // Markers
    Eof,
    Error,

    // Tokens with no info.
    comma,
    equal,
    underscore,
    colon,

    // Keywords
    kw_implicit,
    kw_implicit_define,
    kw_dead,
    kw_killed,
    kw_undef,

    // Identifier tokens
    Identifier,
    NamedRegister,
    MachineBasicBlock,
    NamedGlobalValue,
    GlobalValue,

    // Other tokens
    IntegerLiteral,
    VirtualRegister
  };

private:
  TokenKind Kind;
  unsigned StringOffset;
  StringRef Range;
  APSInt IntVal;

public:
  MIToken(TokenKind Kind, StringRef Range, unsigned StringOffset = 0)
      : Kind(Kind), StringOffset(StringOffset), Range(Range) {}

  MIToken(TokenKind Kind, StringRef Range, const APSInt &IntVal,
          unsigned StringOffset = 0)
      : Kind(Kind), StringOffset(StringOffset), Range(Range), IntVal(IntVal) {}

  TokenKind kind() const { return Kind; }

  bool isError() const { return Kind == Error; }

  bool isRegister() const {
    return Kind == NamedRegister || Kind == underscore ||
           Kind == VirtualRegister;
  }

  bool isRegisterFlag() const {
    return Kind == kw_implicit || Kind == kw_implicit_define ||
           Kind == kw_dead || Kind == kw_killed || Kind == kw_undef;
  }

  bool is(TokenKind K) const { return Kind == K; }

  bool isNot(TokenKind K) const { return Kind != K; }

  StringRef::iterator location() const { return Range.begin(); }

  StringRef stringValue() const { return Range.drop_front(StringOffset); }

  const APSInt &integerValue() const { return IntVal; }

  bool hasIntegerValue() const {
    return Kind == IntegerLiteral || Kind == MachineBasicBlock ||
           Kind == GlobalValue || Kind == VirtualRegister;
  }
};

/// Consume a single machine instruction token in the given source and return
/// the remaining source string.
StringRef lexMIToken(
    StringRef Source, MIToken &Token,
    function_ref<void(StringRef::iterator, const Twine &)> ErrorCallback);

} // end namespace llvm

#endif
