//===- TGLexer.cpp - Lexer for TableGen -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implement the Lexer for TableGen.
//
//===----------------------------------------------------------------------===//

#include "TGLexer.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Config/config.h" // for strtoull()/strtoll() define
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/TableGen/Error.h"
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace llvm;

TGLexer::TGLexer(SourceMgr &SM) : SrcMgr(SM) {
  CurBuffer = SrcMgr.getMainFileID();
  CurBuf = SrcMgr.getMemoryBuffer(CurBuffer)->getBuffer();
  CurPtr = CurBuf.begin();
  TokStart = nullptr;
}

SMLoc TGLexer::getLoc() const {
  return SMLoc::getFromPointer(TokStart);
}

/// ReturnError - Set the error to the specified string at the specified
/// location.  This is defined to always return tgtok::Error.
tgtok::TokKind TGLexer::ReturnError(const char *Loc, const Twine &Msg) {
  PrintError(Loc, Msg);
  return tgtok::Error;
}

int TGLexer::getNextChar() {
  char CurChar = *CurPtr++;
  switch (CurChar) {
  default:
    return (unsigned char)CurChar;
  case 0: {
    // A nul character in the stream is either the end of the current buffer or
    // a random nul in the file.  Disambiguate that here.
    if (CurPtr-1 != CurBuf.end())
      return 0;  // Just whitespace.
    
    // If this is the end of an included file, pop the parent file off the
    // include stack.
    SMLoc ParentIncludeLoc = SrcMgr.getParentIncludeLoc(CurBuffer);
    if (ParentIncludeLoc != SMLoc()) {
      CurBuffer = SrcMgr.FindBufferContainingLoc(ParentIncludeLoc);
      CurBuf = SrcMgr.getMemoryBuffer(CurBuffer)->getBuffer();
      CurPtr = ParentIncludeLoc.getPointer();
      return getNextChar();
    }
    
    // Otherwise, return end of file.
    --CurPtr;  // Another call to lex will return EOF again.  
    return EOF;
  }
  case '\n':
  case '\r':
    // Handle the newline character by ignoring it and incrementing the line
    // count.  However, be careful about 'dos style' files with \n\r in them.
    // Only treat a \n\r or \r\n as a single line.
    if ((*CurPtr == '\n' || (*CurPtr == '\r')) &&
        *CurPtr != CurChar)
      ++CurPtr;  // Eat the two char newline sequence.
    return '\n';
  }  
}

int TGLexer::peekNextChar(int Index) {
  return *(CurPtr + Index);
}

tgtok::TokKind TGLexer::LexToken() {
  TokStart = CurPtr;
  // This always consumes at least one character.
  int CurChar = getNextChar();

  switch (CurChar) {
  default:
    // Handle letters: [a-zA-Z_]
    if (isalpha(CurChar) || CurChar == '_')
      return LexIdentifier();

    // Unknown character, emit an error.
    return ReturnError(TokStart, "Unexpected character");
  case EOF: return tgtok::Eof;
  case ':': return tgtok::colon;
  case ';': return tgtok::semi;
  case '.': return tgtok::period;
  case ',': return tgtok::comma;
  case '<': return tgtok::less;
  case '>': return tgtok::greater;
  case ']': return tgtok::r_square;
  case '{': return tgtok::l_brace;
  case '}': return tgtok::r_brace;
  case '(': return tgtok::l_paren;
  case ')': return tgtok::r_paren;
  case '=': return tgtok::equal;
  case '?': return tgtok::question;
  case '#': return tgtok::paste;
      
  case 0:
  case ' ':
  case '\t':
  case '\n':
  case '\r':
    // Ignore whitespace.
    return LexToken();
  case '/':
    // If this is the start of a // comment, skip until the end of the line or
    // the end of the buffer.
    if (*CurPtr == '/')
      SkipBCPLComment();
    else if (*CurPtr == '*') {
      if (SkipCComment())
        return tgtok::Error;
    } else // Otherwise, this is an error.
      return ReturnError(TokStart, "Unexpected character");
    return LexToken();
  case '-': case '+':
  case '0': case '1': case '2': case '3': case '4': case '5': case '6':
  case '7': case '8': case '9': {
    int NextChar = 0;
    if (isdigit(CurChar)) {
      // Allow identifiers to start with a number if it is followed by
      // an identifier.  This can happen with paste operations like
      // foo#8i.
      int i = 0;
      do {
        NextChar = peekNextChar(i++);
      } while (isdigit(NextChar));

      if (NextChar == 'x' || NextChar == 'b') {
        // If this is [0-9]b[01] or [0-9]x[0-9A-fa-f] this is most
        // likely a number.
        int NextNextChar = peekNextChar(i);
        switch (NextNextChar) {
        default:
          break;
        case '0': case '1': 
          if (NextChar == 'b')
            return LexNumber();
          // Fallthrough
        case '2': case '3': case '4': case '5':
        case '6': case '7': case '8': case '9':
        case 'a': case 'b': case 'c': case 'd': case 'e': case 'f':
        case 'A': case 'B': case 'C': case 'D': case 'E': case 'F':
          if (NextChar == 'x')
            return LexNumber();
          break;
        }
      }
    }

    if (isalpha(NextChar) || NextChar == '_')
      return LexIdentifier();

    return LexNumber();
  }
  case '"': return LexString();
  case '$': return LexVarName();
  case '[': return LexBracket();
  case '!': return LexExclaim();
  }
}

/// LexString - Lex "[^"]*"
tgtok::TokKind TGLexer::LexString() {
  const char *StrStart = CurPtr;
  
  CurStrVal = "";
  
  while (*CurPtr != '"') {
    // If we hit the end of the buffer, report an error.
    if (*CurPtr == 0 && CurPtr == CurBuf.end())
      return ReturnError(StrStart, "End of file in string literal");
    
    if (*CurPtr == '\n' || *CurPtr == '\r')
      return ReturnError(StrStart, "End of line in string literal");
    
    if (*CurPtr != '\\') {
      CurStrVal += *CurPtr++;
      continue;
    }

    ++CurPtr;
    
    switch (*CurPtr) {
    case '\\': case '\'': case '"':
      // These turn into their literal character.
      CurStrVal += *CurPtr++;
      break;
    case 't':
      CurStrVal += '\t';
      ++CurPtr;
      break;
    case 'n':
      CurStrVal += '\n';
      ++CurPtr;
      break;
        
    case '\n':
    case '\r':
      return ReturnError(CurPtr, "escaped newlines not supported in tblgen");

    // If we hit the end of the buffer, report an error.
    case '\0':
      if (CurPtr == CurBuf.end())
        return ReturnError(StrStart, "End of file in string literal");
      // FALL THROUGH
    default:
      return ReturnError(CurPtr, "invalid escape in string literal");
    }
  }
  
  ++CurPtr;
  return tgtok::StrVal;
}

tgtok::TokKind TGLexer::LexVarName() {
  if (!isalpha(CurPtr[0]) && CurPtr[0] != '_')
    return ReturnError(TokStart, "Invalid variable name");
  
  // Otherwise, we're ok, consume the rest of the characters.
  const char *VarNameStart = CurPtr++;
  
  while (isalpha(*CurPtr) || isdigit(*CurPtr) || *CurPtr == '_')
    ++CurPtr;

  CurStrVal.assign(VarNameStart, CurPtr);
  return tgtok::VarName;
}


tgtok::TokKind TGLexer::LexIdentifier() {
  // The first letter is [a-zA-Z_#].
  const char *IdentStart = TokStart;

  // Match the rest of the identifier regex: [0-9a-zA-Z_#]*
  while (isalpha(*CurPtr) || isdigit(*CurPtr) || *CurPtr == '_')
    ++CurPtr;

  // Check to see if this identifier is a keyword.
  StringRef Str(IdentStart, CurPtr-IdentStart);

  if (Str == "include") {
    if (LexInclude()) return tgtok::Error;
    return Lex();
  }

  tgtok::TokKind Kind = StringSwitch<tgtok::TokKind>(Str)
    .Case("int", tgtok::Int)
    .Case("bit", tgtok::Bit)
    .Case("bits", tgtok::Bits)
    .Case("string", tgtok::String)
    .Case("list", tgtok::List)
    .Case("code", tgtok::Code)
    .Case("dag", tgtok::Dag)
    .Case("class", tgtok::Class)
    .Case("def", tgtok::Def)
    .Case("foreach", tgtok::Foreach)
    .Case("defm", tgtok::Defm)
    .Case("multiclass", tgtok::MultiClass)
    .Case("field", tgtok::Field)
    .Case("let", tgtok::Let)
    .Case("in", tgtok::In)
    .Default(tgtok::Id);

  if (Kind == tgtok::Id)
    CurStrVal.assign(Str.begin(), Str.end());
  return Kind;
}

/// LexInclude - We just read the "include" token.  Get the string token that
/// comes next and enter the include.
bool TGLexer::LexInclude() {
  // The token after the include must be a string.
  tgtok::TokKind Tok = LexToken();
  if (Tok == tgtok::Error) return true;
  if (Tok != tgtok::StrVal) {
    PrintError(getLoc(), "Expected filename after include");
    return true;
  }

  // Get the string.
  std::string Filename = CurStrVal;
  std::string IncludedFile;

  
  CurBuffer = SrcMgr.AddIncludeFile(Filename, SMLoc::getFromPointer(CurPtr),
                                    IncludedFile);
  if (!CurBuffer) {
    PrintError(getLoc(), "Could not find include file '" + Filename + "'");
    return true;
  }
  
  DependenciesMapTy::const_iterator Found = Dependencies.find(IncludedFile);
  if (Found != Dependencies.end()) {
    PrintError(getLoc(),
               "File '" + IncludedFile + "' has already been included.");
    SrcMgr.PrintMessage(Found->second, SourceMgr::DK_Note,
                        "previously included here");
    return true;
  }
  Dependencies.insert(std::make_pair(IncludedFile, getLoc()));
  // Save the line number and lex buffer of the includer.
  CurBuf = SrcMgr.getMemoryBuffer(CurBuffer)->getBuffer();
  CurPtr = CurBuf.begin();
  return false;
}

void TGLexer::SkipBCPLComment() {
  ++CurPtr;  // skip the second slash.
  while (1) {
    switch (*CurPtr) {
    case '\n':
    case '\r':
      return;  // Newline is end of comment.
    case 0:
      // If this is the end of the buffer, end the comment.
      if (CurPtr == CurBuf.end())
        return;
      break;
    }
    // Otherwise, skip the character.
    ++CurPtr;
  }
}

/// SkipCComment - This skips C-style /**/ comments.  The only difference from C
/// is that we allow nesting.
bool TGLexer::SkipCComment() {
  ++CurPtr;  // skip the star.
  unsigned CommentDepth = 1;
  
  while (1) {
    int CurChar = getNextChar();
    switch (CurChar) {
    case EOF:
      PrintError(TokStart, "Unterminated comment!");
      return true;
    case '*':
      // End of the comment?
      if (CurPtr[0] != '/') break;
      
      ++CurPtr;   // End the */.
      if (--CommentDepth == 0)
        return false;
      break;
    case '/':
      // Start of a nested comment?
      if (CurPtr[0] != '*') break;
      ++CurPtr;
      ++CommentDepth;
      break;
    }
  }
}

/// LexNumber - Lex:
///    [-+]?[0-9]+
///    0x[0-9a-fA-F]+
///    0b[01]+
tgtok::TokKind TGLexer::LexNumber() {
  if (CurPtr[-1] == '0') {
    if (CurPtr[0] == 'x') {
      ++CurPtr;
      const char *NumStart = CurPtr;
      while (isxdigit(CurPtr[0]))
        ++CurPtr;
      
      // Requires at least one hex digit.
      if (CurPtr == NumStart)
        return ReturnError(TokStart, "Invalid hexadecimal number");

      errno = 0;
      CurIntVal = strtoll(NumStart, nullptr, 16);
      if (errno == EINVAL)
        return ReturnError(TokStart, "Invalid hexadecimal number");
      if (errno == ERANGE) {
        errno = 0;
        CurIntVal = (int64_t)strtoull(NumStart, nullptr, 16);
        if (errno == EINVAL)
          return ReturnError(TokStart, "Invalid hexadecimal number");
        if (errno == ERANGE)
          return ReturnError(TokStart, "Hexadecimal number out of range");
      }
      return tgtok::IntVal;
    } else if (CurPtr[0] == 'b') {
      ++CurPtr;
      const char *NumStart = CurPtr;
      while (CurPtr[0] == '0' || CurPtr[0] == '1')
        ++CurPtr;

      // Requires at least one binary digit.
      if (CurPtr == NumStart)
        return ReturnError(CurPtr-2, "Invalid binary number");
      CurIntVal = strtoll(NumStart, nullptr, 2);
      return tgtok::BinaryIntVal;
    }
  }

  // Check for a sign without a digit.
  if (!isdigit(CurPtr[0])) {
    if (CurPtr[-1] == '-')
      return tgtok::minus;
    else if (CurPtr[-1] == '+')
      return tgtok::plus;
  }
  
  while (isdigit(CurPtr[0]))
    ++CurPtr;
  CurIntVal = strtoll(TokStart, nullptr, 10);
  return tgtok::IntVal;
}

/// LexBracket - We just read '['.  If this is a code block, return it,
/// otherwise return the bracket.  Match: '[' and '[{ ( [^}]+ | }[^]] )* }]'
tgtok::TokKind TGLexer::LexBracket() {
  if (CurPtr[0] != '{')
    return tgtok::l_square;
  ++CurPtr;
  const char *CodeStart = CurPtr;
  while (1) {
    int Char = getNextChar();
    if (Char == EOF) break;
    
    if (Char != '}') continue;
    
    Char = getNextChar();
    if (Char == EOF) break;
    if (Char == ']') {
      CurStrVal.assign(CodeStart, CurPtr-2);
      return tgtok::CodeFragment;
    }
  }
  
  return ReturnError(CodeStart-2, "Unterminated Code Block");
}

/// LexExclaim - Lex '!' and '![a-zA-Z]+'.
tgtok::TokKind TGLexer::LexExclaim() {
  if (!isalpha(*CurPtr))
    return ReturnError(CurPtr - 1, "Invalid \"!operator\"");
  
  const char *Start = CurPtr++;
  while (isalpha(*CurPtr))
    ++CurPtr;
  
  // Check to see which operator this is.
  tgtok::TokKind Kind =
    StringSwitch<tgtok::TokKind>(StringRef(Start, CurPtr - Start))
    .Case("eq", tgtok::XEq)
    .Case("if", tgtok::XIf)
    .Case("head", tgtok::XHead)
    .Case("tail", tgtok::XTail)
    .Case("con", tgtok::XConcat)
    .Case("add", tgtok::XADD)
    .Case("and", tgtok::XAND)
    .Case("shl", tgtok::XSHL)
    .Case("sra", tgtok::XSRA)
    .Case("srl", tgtok::XSRL)
    .Case("cast", tgtok::XCast)
    .Case("empty", tgtok::XEmpty)
    .Case("subst", tgtok::XSubst)
    .Case("foreach", tgtok::XForEach)
    .Case("listconcat", tgtok::XListConcat)
    .Case("strconcat", tgtok::XStrConcat)
    .Default(tgtok::Error);

  return Kind != tgtok::Error ? Kind : ReturnError(Start-1, "Unknown operator");
}

