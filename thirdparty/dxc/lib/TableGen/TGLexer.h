//===- TGLexer.h - Lexer for TableGen Files ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class represents the Lexer for tablegen files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TABLEGEN_TGLEXER_H
#define LLVM_LIB_TABLEGEN_TGLEXER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/SMLoc.h"
#include <cassert>
#include <map>
#include <string>

namespace llvm {
class SourceMgr;
class SMLoc;
class Twine;

namespace tgtok {
  enum TokKind {
    // Markers
    Eof, Error,
    
    // Tokens with no info.
    minus, plus,        // - +
    l_square, r_square, // [ ]
    l_brace, r_brace,   // { }
    l_paren, r_paren,   // ( )
    less, greater,      // < >
    colon, semi,        // : ;
    comma, period,      // , .
    equal, question,    // = ?
    paste,              // #

    // Keywords.
    Bit, Bits, Class, Code, Dag, Def, Foreach, Defm, Field, In, Int, Let, List,
    MultiClass, String,
    
    // !keywords.
    XConcat, XADD, XAND, XSRA, XSRL, XSHL, XListConcat, XStrConcat, XCast,
    XSubst, XForEach, XHead, XTail, XEmpty, XIf, XEq,

    // Integer value.
    IntVal,

    // Binary constant.  Note that these are sized according to the number of
    // bits given.
    BinaryIntVal,
    
    // String valued tokens.
    Id, StrVal, VarName, CodeFragment
  };
}

/// TGLexer - TableGen Lexer class.
class TGLexer {
  SourceMgr &SrcMgr;
  
  const char *CurPtr;
  StringRef CurBuf;

  // Information about the current token.
  const char *TokStart;
  tgtok::TokKind CurCode;
  std::string CurStrVal;  // This is valid for ID, STRVAL, VARNAME, CODEFRAGMENT
  int64_t CurIntVal;      // This is valid for INTVAL.

  /// CurBuffer - This is the current buffer index we're lexing from as managed
  /// by the SourceMgr object.
  unsigned CurBuffer;

public:
  typedef std::map<std::string, SMLoc> DependenciesMapTy;
private:
  /// Dependencies - This is the list of all included files.
  DependenciesMapTy Dependencies;

public:
  TGLexer(SourceMgr &SrcMgr);

  tgtok::TokKind Lex() {
    return CurCode = LexToken();
  }

  const DependenciesMapTy &getDependencies() const {
    return Dependencies;
  }
  
  tgtok::TokKind getCode() const { return CurCode; }

  const std::string &getCurStrVal() const {
    assert((CurCode == tgtok::Id || CurCode == tgtok::StrVal || 
            CurCode == tgtok::VarName || CurCode == tgtok::CodeFragment) &&
           "This token doesn't have a string value");
    return CurStrVal;
  }
  int64_t getCurIntVal() const {
    assert(CurCode == tgtok::IntVal && "This token isn't an integer");
    return CurIntVal;
  }
  std::pair<int64_t, unsigned> getCurBinaryIntVal() const {
    assert(CurCode == tgtok::BinaryIntVal &&
           "This token isn't a binary integer");
    return std::make_pair(CurIntVal, (CurPtr - TokStart)-2);
  }

  SMLoc getLoc() const;
  
private:
  /// LexToken - Read the next token and return its code.
  tgtok::TokKind LexToken();
  
  tgtok::TokKind ReturnError(const char *Loc, const Twine &Msg);
  
  int getNextChar();
  int peekNextChar(int Index);
  void SkipBCPLComment();
  bool SkipCComment();
  tgtok::TokKind LexIdentifier();
  bool LexInclude();
  tgtok::TokKind LexString();
  tgtok::TokKind LexVarName();
  tgtok::TokKind LexNumber();
  tgtok::TokKind LexBracket();
  tgtok::TokKind LexExclaim();
};
  
} // end namespace llvm

#endif
