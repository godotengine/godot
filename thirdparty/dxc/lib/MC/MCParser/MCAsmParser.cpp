//===-- MCAsmParser.cpp - Abstract Asm Parser Interface -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCTargetAsmParser.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

MCAsmParser::MCAsmParser() : TargetParser(nullptr), ShowParsedOperands(0) {
}

MCAsmParser::~MCAsmParser() {
}

void MCAsmParser::setTargetParser(MCTargetAsmParser &P) {
  assert(!TargetParser && "Target parser is already initialized!");
  TargetParser = &P;
  TargetParser->Initialize(*this);
}

const AsmToken &MCAsmParser::getTok() const {
  return getLexer().getTok();
}

bool MCAsmParser::TokError(const Twine &Msg, ArrayRef<SMRange> Ranges) {
  Error(getLexer().getLoc(), Msg, Ranges);
  return true;
}

bool MCAsmParser::parseExpression(const MCExpr *&Res) {
  SMLoc L;
  return parseExpression(Res, L);
}

void MCParsedAsmOperand::dump() const {
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  dbgs() << "  " << *this;
#endif
}
