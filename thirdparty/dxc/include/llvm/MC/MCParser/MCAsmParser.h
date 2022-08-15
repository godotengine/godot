//===-- llvm/MC/MCAsmParser.h - Abstract Asm Parser Interface ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCPARSER_MCASMPARSER_H
#define LLVM_MC_MCPARSER_MCASMPARSER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCParser/AsmLexer.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {
class MCAsmInfo;
class MCAsmLexer;
class MCAsmParserExtension;
class MCContext;
class MCExpr;
class MCInstPrinter;
class MCInstrInfo;
class MCStreamer;
class MCTargetAsmParser;
class SMLoc;
class SMRange;
class SourceMgr;
class Twine;

class InlineAsmIdentifierInfo {
public:
  void *OpDecl;
  bool IsVarDecl;
  unsigned Length, Size, Type;

  void clear() {
    OpDecl = nullptr;
    IsVarDecl = false;
    Length = 1;
    Size = 0;
    Type = 0;
  }
};

/// \brief Generic Sema callback for assembly parser.
class MCAsmParserSemaCallback {
public:
  virtual ~MCAsmParserSemaCallback();
  virtual void *LookupInlineAsmIdentifier(StringRef &LineBuf,
                                          InlineAsmIdentifierInfo &Info,
                                          bool IsUnevaluatedContext) = 0;
  virtual StringRef LookupInlineAsmLabel(StringRef Identifier, SourceMgr &SM,
                                         SMLoc Location, bool Create) = 0;

  virtual bool LookupInlineAsmField(StringRef Base, StringRef Member,
                                    unsigned &Offset) = 0;
};

/// \brief Generic assembler parser interface, for use by target specific
/// assembly parsers.
class MCAsmParser {
public:
  typedef bool (*DirectiveHandler)(MCAsmParserExtension*, StringRef, SMLoc);
  typedef std::pair<MCAsmParserExtension*, DirectiveHandler>
    ExtensionDirectiveHandler;

private:
  MCAsmParser(const MCAsmParser &) = delete;
  void operator=(const MCAsmParser &) = delete;

  MCTargetAsmParser *TargetParser;

  unsigned ShowParsedOperands : 1;

protected: // Can only create subclasses.
  MCAsmParser();

public:
  virtual ~MCAsmParser();

  virtual void addDirectiveHandler(StringRef Directive,
                                   ExtensionDirectiveHandler Handler) = 0;

  virtual void addAliasForDirective(StringRef Directive, StringRef Alias) = 0;

  virtual SourceMgr &getSourceManager() = 0;

  virtual MCAsmLexer &getLexer() = 0;
  const MCAsmLexer &getLexer() const {
    return const_cast<MCAsmParser*>(this)->getLexer();
  }

  virtual MCContext &getContext() = 0;

  /// \brief Return the output streamer for the assembler.
  virtual MCStreamer &getStreamer() = 0;

  MCTargetAsmParser &getTargetParser() const { return *TargetParser; }
  void setTargetParser(MCTargetAsmParser &P);

  virtual unsigned getAssemblerDialect() { return 0;}
  virtual void setAssemblerDialect(unsigned i) { }

  bool getShowParsedOperands() const { return ShowParsedOperands; }
  void setShowParsedOperands(bool Value) { ShowParsedOperands = Value; }

  /// \brief Run the parser on the input source buffer.
  virtual bool Run(bool NoInitialTextSection, bool NoFinalize = false) = 0;

  virtual void setParsingInlineAsm(bool V) = 0;
  virtual bool isParsingInlineAsm() = 0;

  /// \brief Parse MS-style inline assembly.
  virtual bool parseMSInlineAsm(
      void *AsmLoc, std::string &AsmString, unsigned &NumOutputs,
      unsigned &NumInputs, SmallVectorImpl<std::pair<void *, bool>> &OpDecls,
      SmallVectorImpl<std::string> &Constraints,
      SmallVectorImpl<std::string> &Clobbers, const MCInstrInfo *MII,
      const MCInstPrinter *IP, MCAsmParserSemaCallback &SI) = 0;

  /// \brief Emit a note at the location \p L, with the message \p Msg.
  virtual void Note(SMLoc L, const Twine &Msg,
                    ArrayRef<SMRange> Ranges = None) = 0;

  /// \brief Emit a warning at the location \p L, with the message \p Msg.
  ///
  /// \return The return value is true, if warnings are fatal.
  virtual bool Warning(SMLoc L, const Twine &Msg,
                       ArrayRef<SMRange> Ranges = None) = 0;

  /// \brief Emit an error at the location \p L, with the message \p Msg.
  ///
  /// \return The return value is always true, as an idiomatic convenience to
  /// clients.
  virtual bool Error(SMLoc L, const Twine &Msg,
                     ArrayRef<SMRange> Ranges = None) = 0;

  /// \brief Get the next AsmToken in the stream, possibly handling file
  /// inclusion first.
  virtual const AsmToken &Lex() = 0;

  /// \brief Get the current AsmToken from the stream.
  const AsmToken &getTok() const;

  /// \brief Report an error at the current lexer location.
  bool TokError(const Twine &Msg, ArrayRef<SMRange> Ranges = None);

  /// \brief Parse an identifier or string (as a quoted identifier) and set \p
  /// Res to the identifier contents.
  virtual bool parseIdentifier(StringRef &Res) = 0;

  /// \brief Parse up to the end of statement and return the contents from the
  /// current token until the end of the statement; the current token on exit
  /// will be either the EndOfStatement or EOF.
  virtual StringRef parseStringToEndOfStatement() = 0;

  /// \brief Parse the current token as a string which may include escaped
  /// characters and return the string contents.
  virtual bool parseEscapedString(std::string &Data) = 0;

  /// \brief Skip to the end of the current statement, for error recovery.
  virtual void eatToEndOfStatement() = 0;

  /// \brief Parse an arbitrary expression.
  ///
  /// \param Res - The value of the expression. The result is undefined
  /// on error.
  /// \return - False on success.
  virtual bool parseExpression(const MCExpr *&Res, SMLoc &EndLoc) = 0;
  bool parseExpression(const MCExpr *&Res);

  /// \brief Parse a primary expression.
  ///
  /// \param Res - The value of the expression. The result is undefined
  /// on error.
  /// \return - False on success.
  virtual bool parsePrimaryExpr(const MCExpr *&Res, SMLoc &EndLoc) = 0;

  /// \brief Parse an arbitrary expression, assuming that an initial '(' has
  /// already been consumed.
  ///
  /// \param Res - The value of the expression. The result is undefined
  /// on error.
  /// \return - False on success.
  virtual bool parseParenExpression(const MCExpr *&Res, SMLoc &EndLoc) = 0;

  /// \brief Parse an expression which must evaluate to an absolute value.
  ///
  /// \param Res - The value of the absolute expression. The result is undefined
  /// on error.
  /// \return - False on success.
  virtual bool parseAbsoluteExpression(int64_t &Res) = 0;

  /// \brief Ensure that we have a valid section set in the streamer. Otherwise,
  /// report an error and switch to .text.
  virtual void checkForValidSection() = 0;

  /// \brief Parse an arbitrary expression of a specified parenthesis depth,
  /// assuming that the initial '(' characters have already been consumed.
  ///
  /// \param ParenDepth - Specifies how many trailing expressions outside the
  /// current parentheses we have to parse.
  /// \param Res - The value of the expression. The result is undefined
  /// on error.
  /// \return - False on success.
  virtual bool parseParenExprOfDepth(unsigned ParenDepth, const MCExpr *&Res,
                                     SMLoc &EndLoc) = 0;
};

/// \brief Create an MCAsmParser instance.
MCAsmParser *createMCAsmParser(SourceMgr &, MCContext &, MCStreamer &,
                               const MCAsmInfo &);

} // End llvm namespace

#endif
