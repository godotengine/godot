//===-- llvm/MC/MCTargetAsmParser.h - Target Assembly Parser ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCTARGETASMPARSER_H
#define LLVM_MC_MCTARGETASMPARSER_H

#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCParser/MCAsmParserExtension.h"
#include "llvm/MC/MCTargetOptions.h"
#include <memory>

namespace llvm {
class AsmToken;
class MCInst;
class MCParsedAsmOperand;
class MCStreamer;
class SMLoc;
class StringRef;
template <typename T> class SmallVectorImpl;

typedef SmallVectorImpl<std::unique_ptr<MCParsedAsmOperand>> OperandVector;

enum AsmRewriteKind {
  AOK_Delete = 0,     // Rewrite should be ignored.
  AOK_Align,          // Rewrite align as .align.
  AOK_DotOperator,    // Rewrite a dot operator expression as an immediate.
                      // E.g., [eax].foo.bar -> [eax].8
  AOK_Emit,           // Rewrite _emit as .byte.
  AOK_Imm,            // Rewrite as $$N.
  AOK_ImmPrefix,      // Add $$ before a parsed Imm.
  AOK_Input,          // Rewrite in terms of $N.
  AOK_Output,         // Rewrite in terms of $N.
  AOK_SizeDirective,  // Add a sizing directive (e.g., dword ptr).
  AOK_Label,          // Rewrite local labels.
  AOK_Skip            // Skip emission (e.g., offset/type operators).
};

const char AsmRewritePrecedence [] = {
  0, // AOK_Delete
  2, // AOK_Align
  2, // AOK_DotOperator
  2, // AOK_Emit
  4, // AOK_Imm
  4, // AOK_ImmPrefix
  3, // AOK_Input
  3, // AOK_Output
  5, // AOK_SizeDirective
  1, // AOK_Label
  2  // AOK_Skip
};

struct AsmRewrite {
  AsmRewriteKind Kind;
  SMLoc Loc;
  unsigned Len;
  unsigned Val;
  StringRef Label;
public:
  AsmRewrite(AsmRewriteKind kind, SMLoc loc, unsigned len = 0, unsigned val = 0)
    : Kind(kind), Loc(loc), Len(len), Val(val) {}
  AsmRewrite(AsmRewriteKind kind, SMLoc loc, unsigned len, StringRef label)
    : Kind(kind), Loc(loc), Len(len), Val(0), Label(label) {}
};

struct ParseInstructionInfo {

  SmallVectorImpl<AsmRewrite> *AsmRewrites;

  ParseInstructionInfo() : AsmRewrites(nullptr) {}
  ParseInstructionInfo(SmallVectorImpl<AsmRewrite> *rewrites)
    : AsmRewrites(rewrites) {}
};

/// MCTargetAsmParser - Generic interface to target specific assembly parsers.
class MCTargetAsmParser : public MCAsmParserExtension {
public:
  enum MatchResultTy {
    Match_InvalidOperand,
    Match_MissingFeature,
    Match_MnemonicFail,
    Match_Success,
    FIRST_TARGET_MATCH_RESULT_TY
  };

private:
  MCTargetAsmParser(const MCTargetAsmParser &) = delete;
  void operator=(const MCTargetAsmParser &) = delete;
protected: // Can only create subclasses.
  MCTargetAsmParser();

  /// AvailableFeatures - The current set of available features.
  uint64_t AvailableFeatures;

  /// ParsingInlineAsm - Are we parsing ms-style inline assembly?
  bool ParsingInlineAsm;

  /// SemaCallback - The Sema callback implementation.  Must be set when parsing
  /// ms-style inline assembly.
  MCAsmParserSemaCallback *SemaCallback;

  /// Set of options which affects instrumentation of inline assembly.
  MCTargetOptions MCOptions;

public:
  ~MCTargetAsmParser() override;

  uint64_t getAvailableFeatures() const { return AvailableFeatures; }
  void setAvailableFeatures(uint64_t Value) { AvailableFeatures = Value; }

  bool isParsingInlineAsm () { return ParsingInlineAsm; }
  void setParsingInlineAsm (bool Value) { ParsingInlineAsm = Value; }

  MCTargetOptions getTargetOptions() const { return MCOptions; }

  void setSemaCallback(MCAsmParserSemaCallback *Callback) {
    SemaCallback = Callback;
  }

  virtual bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc,
                             SMLoc &EndLoc) = 0;

  /// Sets frame register corresponding to the current MachineFunction.
  virtual void SetFrameRegister(unsigned RegNo) {}

  /// ParseInstruction - Parse one assembly instruction.
  ///
  /// The parser is positioned following the instruction name. The target
  /// specific instruction parser should parse the entire instruction and
  /// construct the appropriate MCInst, or emit an error. On success, the entire
  /// line should be parsed up to and including the end-of-statement token. On
  /// failure, the parser is not required to read to the end of the line.
  //
  /// \param Name - The instruction name.
  /// \param NameLoc - The source location of the name.
  /// \param Operands [out] - The list of parsed operands, this returns
  ///        ownership of them to the caller.
  /// \return True on failure.
  virtual bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                                SMLoc NameLoc, OperandVector &Operands) = 0;

  /// ParseDirective - Parse a target specific assembler directive
  ///
  /// The parser is positioned following the directive name.  The target
  /// specific directive parser should parse the entire directive doing or
  /// recording any target specific work, or return true and do nothing if the
  /// directive is not target specific. If the directive is specific for
  /// the target, the entire line is parsed up to and including the
  /// end-of-statement token and false is returned.
  ///
  /// \param DirectiveID - the identifier token of the directive.
  virtual bool ParseDirective(AsmToken DirectiveID) = 0;

  /// mnemonicIsValid - This returns true if this is a valid mnemonic and false
  /// otherwise.
  virtual bool mnemonicIsValid(StringRef Mnemonic, unsigned VariantID) = 0;

  /// MatchAndEmitInstruction - Recognize a series of operands of a parsed
  /// instruction as an actual MCInst and emit it to the specified MCStreamer.
  /// This returns false on success and returns true on failure to match.
  ///
  /// On failure, the target parser is responsible for emitting a diagnostic
  /// explaining the match failure.
  virtual bool MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                                       OperandVector &Operands, MCStreamer &Out,
                                       uint64_t &ErrorInfo,
                                       bool MatchingInlineAsm) = 0;

  /// Allows targets to let registers opt out of clobber lists.
  virtual bool OmitRegisterFromClobberLists(unsigned RegNo) { return false; }

  /// Allow a target to add special case operand matching for things that
  /// tblgen doesn't/can't handle effectively. For example, literal
  /// immediates on ARM. TableGen expects a token operand, but the parser
  /// will recognize them as immediates.
  virtual unsigned validateTargetOperandClass(MCParsedAsmOperand &Op,
                                              unsigned Kind) {
    return Match_InvalidOperand;
  }

  /// checkTargetMatchPredicate - Validate the instruction match against
  /// any complex target predicates not expressible via match classes.
  virtual unsigned checkTargetMatchPredicate(MCInst &Inst) {
    return Match_Success;
  }

  virtual void convertToMapAndConstraints(unsigned Kind,
                                          const OperandVector &Operands) = 0;

  virtual const MCExpr *applyModifierToExpr(const MCExpr *E,
                                            MCSymbolRefExpr::VariantKind,
                                            MCContext &Ctx) {
    return nullptr;
  }

  virtual void onLabelParsed(MCSymbol *Symbol) { };
};

} // End llvm namespace

#endif
