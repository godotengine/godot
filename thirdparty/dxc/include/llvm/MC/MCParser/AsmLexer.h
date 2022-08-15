//===- AsmLexer.h - Lexer for Assembly Files --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class declares the lexer for assembly files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCPARSER_ASMLEXER_H
#define LLVM_MC_MCPARSER_ASMLEXER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/Support/DataTypes.h"
#include <string>

namespace llvm {
class MemoryBuffer;
class MCAsmInfo;

/// AsmLexer - Lexer class for assembly files.
class AsmLexer : public MCAsmLexer {
  const MCAsmInfo &MAI;

  const char *CurPtr;
  StringRef CurBuf;
  bool isAtStartOfLine;

  void operator=(const AsmLexer&) = delete;
  AsmLexer(const AsmLexer&) = delete;

protected:
  /// LexToken - Read the next token and return its code.
  AsmToken LexToken() override;

public:
  AsmLexer(const MCAsmInfo &MAI);
  ~AsmLexer() override;

  void setBuffer(StringRef Buf, const char *ptr = nullptr);

  StringRef LexUntilEndOfStatement() override;
  StringRef LexUntilEndOfLine();

  const AsmToken peekTok(bool ShouldSkipSpace = true) override;

  bool isAtStartOfComment(const char *Ptr);
  bool isAtStatementSeparator(const char *Ptr);

  const MCAsmInfo &getMAI() const { return MAI; }

private:
  int getNextChar();
  AsmToken ReturnError(const char *Loc, const std::string &Msg);

  AsmToken LexIdentifier();
  AsmToken LexSlash();
  AsmToken LexLineComment();
  AsmToken LexDigit();
  AsmToken LexSingleQuote();
  AsmToken LexQuote();
  AsmToken LexFloatLiteral();
  AsmToken LexHexFloatLiteral(bool NoIntDigits);
};

} // end namespace llvm

#endif
