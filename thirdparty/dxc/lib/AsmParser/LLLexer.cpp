//===- LLLexer.cpp - Lexer for .ll Files ----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implement the Lexer for .ll files.
//
//===----------------------------------------------------------------------===//

#include "LLLexer.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
using namespace llvm;

bool LLLexer::Error(LocTy ErrorLoc, const Twine &Msg) const {
  ErrorInfo = SM.GetMessage(ErrorLoc, SourceMgr::DK_Error, Msg);
  return true;
}

void LLLexer::Warning(LocTy WarningLoc, const Twine &Msg) const {
  SM.PrintMessage(WarningLoc, SourceMgr::DK_Warning, Msg);
}

//===----------------------------------------------------------------------===//
// Helper functions.
//===----------------------------------------------------------------------===//

// atoull - Convert an ascii string of decimal digits into the unsigned long
// long representation... this does not have to do input error checking,
// because we know that the input will be matched by a suitable regex...
//
uint64_t LLLexer::atoull(const char *Buffer, const char *End) {
  uint64_t Result = 0;
  for (; Buffer != End; Buffer++) {
    uint64_t OldRes = Result;
    Result *= 10;
    Result += *Buffer-'0';
    if (Result < OldRes) {  // Uh, oh, overflow detected!!!
      Error("constant bigger than 64 bits detected!");
      return 0;
    }
  }
  return Result;
}

uint64_t LLLexer::HexIntToVal(const char *Buffer, const char *End) {
  uint64_t Result = 0;
  for (; Buffer != End; ++Buffer) {
    uint64_t OldRes = Result;
    Result *= 16;
    Result += hexDigitValue(*Buffer);

    if (Result < OldRes) {   // Uh, oh, overflow detected!!!
      Error("constant bigger than 64 bits detected!");
      return 0;
    }
  }
  return Result;
}

void LLLexer::HexToIntPair(const char *Buffer, const char *End,
                           uint64_t Pair[2]) {
  Pair[0] = 0;
  if (End - Buffer >= 16) {
    for (int i = 0; i < 16; i++, Buffer++) {
      assert(Buffer != End);
      Pair[0] *= 16;
      Pair[0] += hexDigitValue(*Buffer);
    }
  }
  Pair[1] = 0;
  for (int i = 0; i < 16 && Buffer != End; i++, Buffer++) {
    Pair[1] *= 16;
    Pair[1] += hexDigitValue(*Buffer);
  }
  if (Buffer != End)
    Error("constant bigger than 128 bits detected!");
}

/// FP80HexToIntPair - translate an 80 bit FP80 number (20 hexits) into
/// { low64, high16 } as usual for an APInt.
void LLLexer::FP80HexToIntPair(const char *Buffer, const char *End,
                           uint64_t Pair[2]) {
  Pair[1] = 0;
  for (int i=0; i<4 && Buffer != End; i++, Buffer++) {
    assert(Buffer != End);
    Pair[1] *= 16;
    Pair[1] += hexDigitValue(*Buffer);
  }
  Pair[0] = 0;
  for (int i=0; i<16; i++, Buffer++) {
    Pair[0] *= 16;
    Pair[0] += hexDigitValue(*Buffer);
  }
  if (Buffer != End)
    Error("constant bigger than 128 bits detected!");
}

// UnEscapeLexed - Run through the specified buffer and change \xx codes to the
// appropriate character.
static void UnEscapeLexed(std::string &Str) {
  if (Str.empty()) return;

  char *Buffer = &Str[0], *EndBuffer = Buffer+Str.size();
  char *BOut = Buffer;
  for (char *BIn = Buffer; BIn != EndBuffer; ) {
    if (BIn[0] == '\\') {
      if (BIn < EndBuffer-1 && BIn[1] == '\\') {
        *BOut++ = '\\'; // Two \ becomes one
        BIn += 2;
      } else if (BIn < EndBuffer-2 &&
                 isxdigit(static_cast<unsigned char>(BIn[1])) &&
                 isxdigit(static_cast<unsigned char>(BIn[2]))) {
        *BOut = hexDigitValue(BIn[1]) * 16 + hexDigitValue(BIn[2]);
        BIn += 3;                           // Skip over handled chars
        ++BOut;
      } else {
        *BOut++ = *BIn++;
      }
    } else {
      *BOut++ = *BIn++;
    }
  }
  Str.resize(BOut-Buffer);
}

/// isLabelChar - Return true for [-a-zA-Z$._0-9].
static bool isLabelChar(char C) {
  return isalnum(static_cast<unsigned char>(C)) || C == '-' || C == '$' ||
         C == '.' || C == '_';
}


/// isLabelTail - Return true if this pointer points to a valid end of a label.
static const char *isLabelTail(const char *CurPtr) {
  while (1) {
    if (CurPtr[0] == ':') return CurPtr+1;
    if (!isLabelChar(CurPtr[0])) return nullptr;
    ++CurPtr;
  }
}



//===----------------------------------------------------------------------===//
// Lexer definition.
//===----------------------------------------------------------------------===//

LLLexer::LLLexer(StringRef StartBuf, SourceMgr &sm, SMDiagnostic &Err,
                 LLVMContext &C)
  : CurBuf(StartBuf), ErrorInfo(Err), SM(sm), Context(C), APFloatVal(0.0) {
  CurPtr = CurBuf.begin();
}

int LLLexer::getNextChar() {
  char CurChar = *CurPtr++;
  switch (CurChar) {
  default: return (unsigned char)CurChar;
  case 0:
    // A nul character in the stream is either the end of the current buffer or
    // a random nul in the file.  Disambiguate that here.
    if (CurPtr-1 != CurBuf.end())
      return 0;  // Just whitespace.

    // Otherwise, return end of file.
    --CurPtr;  // Another call to lex will return EOF again.
    return EOF;
  }
}


lltok::Kind LLLexer::LexToken() {
  TokStart = CurPtr;

  int CurChar = getNextChar();
  switch (CurChar) {
  default:
    // Handle letters: [a-zA-Z_]
    if (isalpha(static_cast<unsigned char>(CurChar)) || CurChar == '_')
      return LexIdentifier();

    return lltok::Error;
  case EOF: return lltok::Eof;
  case 0:
  case ' ':
  case '\t':
  case '\n':
  case '\r':
    // Ignore whitespace.
    return LexToken();
  case '+': return LexPositive();
  case '@': return LexAt();
  case '$': return LexDollar();
  case '%': return LexPercent();
  case '"': return LexQuote();
  case '.':
    if (const char *Ptr = isLabelTail(CurPtr)) {
      CurPtr = Ptr;
      StrVal.assign(TokStart, CurPtr-1);
      return lltok::LabelStr;
    }
    if (CurPtr[0] == '.' && CurPtr[1] == '.') {
      CurPtr += 2;
      return lltok::dotdotdot;
    }
    return lltok::Error;
  case ';':
    SkipLineComment();
    return LexToken();
  case '!': return LexExclaim();
  case '#': return LexHash();
  case '0': case '1': case '2': case '3': case '4':
  case '5': case '6': case '7': case '8': case '9':
  case '-':
    return LexDigitOrNegative();
  case '=': return lltok::equal;
  case '[': return lltok::lsquare;
  case ']': return lltok::rsquare;
  case '{': return lltok::lbrace;
  case '}': return lltok::rbrace;
  case '<': return lltok::less;
  case '>': return lltok::greater;
  case '(': return lltok::lparen;
  case ')': return lltok::rparen;
  case ',': return lltok::comma;
  case '*': return lltok::star;
  case '|': return lltok::bar;
  }
}

void LLLexer::SkipLineComment() {
  while (1) {
    if (CurPtr[0] == '\n' || CurPtr[0] == '\r' || getNextChar() == EOF)
      return;
  }
}

/// Lex all tokens that start with an @ character.
///   GlobalVar   @\"[^\"]*\"
///   GlobalVar   @[-a-zA-Z$._][-a-zA-Z$._0-9]*
///   GlobalVarID @[0-9]+
lltok::Kind LLLexer::LexAt() {
  return LexVar(lltok::GlobalVar, lltok::GlobalID);
}

lltok::Kind LLLexer::LexDollar() {
  if (const char *Ptr = isLabelTail(TokStart)) {
    CurPtr = Ptr;
    StrVal.assign(TokStart, CurPtr - 1);
    return lltok::LabelStr;
  }

  // Handle DollarStringConstant: $\"[^\"]*\"
  if (CurPtr[0] == '"') {
    ++CurPtr;

    while (1) {
      int CurChar = getNextChar();

      if (CurChar == EOF) {
        Error("end of file in COMDAT variable name");
        return lltok::Error;
      }
      if (CurChar == '"') {
        StrVal.assign(TokStart + 2, CurPtr - 1);
        UnEscapeLexed(StrVal);
        if (StringRef(StrVal).find_first_of(0) != StringRef::npos) {
          Error("Null bytes are not allowed in names");
          return lltok::Error;
        }
        return lltok::ComdatVar;
      }
    }
  }

  // Handle ComdatVarName: $[-a-zA-Z$._][-a-zA-Z$._0-9]*
  if (ReadVarName())
    return lltok::ComdatVar;

  return lltok::Error;
}

/// ReadString - Read a string until the closing quote.
lltok::Kind LLLexer::ReadString(lltok::Kind kind) {
  const char *Start = CurPtr;
  while (1) {
    int CurChar = getNextChar();

    if (CurChar == EOF) {
      Error("end of file in string constant");
      return lltok::Error;
    }
    if (CurChar == '"') {
      StrVal.assign(Start, CurPtr-1);
      UnEscapeLexed(StrVal);
      return kind;
    }
  }
}

/// ReadVarName - Read the rest of a token containing a variable name.
bool LLLexer::ReadVarName() {
  const char *NameStart = CurPtr;
  if (isalpha(static_cast<unsigned char>(CurPtr[0])) ||
      CurPtr[0] == '-' || CurPtr[0] == '$' ||
      CurPtr[0] == '.' || CurPtr[0] == '_') {
    ++CurPtr;
    while (isalnum(static_cast<unsigned char>(CurPtr[0])) ||
           CurPtr[0] == '-' || CurPtr[0] == '$' ||
           CurPtr[0] == '.' || CurPtr[0] == '_')
      ++CurPtr;

    StrVal.assign(NameStart, CurPtr);
    return true;
  }
  return false;
}

lltok::Kind LLLexer::LexVar(lltok::Kind Var, lltok::Kind VarID) {
  // Handle StringConstant: \"[^\"]*\"
  if (CurPtr[0] == '"') {
    ++CurPtr;

    while (1) {
      int CurChar = getNextChar();

      if (CurChar == EOF) {
        Error("end of file in global variable name");
        return lltok::Error;
      }
      if (CurChar == '"') {
        StrVal.assign(TokStart+2, CurPtr-1);
        UnEscapeLexed(StrVal);
        if (StringRef(StrVal).find_first_of(0) != StringRef::npos) {
          Error("Null bytes are not allowed in names");
          return lltok::Error;
        }
        return Var;
      }
    }
  }

  // Handle VarName: [-a-zA-Z$._][-a-zA-Z$._0-9]*
  if (ReadVarName())
    return Var;

  // Handle VarID: [0-9]+
  if (isdigit(static_cast<unsigned char>(CurPtr[0]))) {
    for (++CurPtr; isdigit(static_cast<unsigned char>(CurPtr[0])); ++CurPtr)
      /*empty*/;

    uint64_t Val = atoull(TokStart+1, CurPtr);
    if ((unsigned)Val != Val)
      Error("invalid value number (too large)!");
    UIntVal = unsigned(Val);
    return VarID;
  }
  return lltok::Error;
}

/// Lex all tokens that start with a % character.
///   LocalVar   ::= %\"[^\"]*\"
///   LocalVar   ::= %[-a-zA-Z$._][-a-zA-Z$._0-9]*
///   LocalVarID ::= %[0-9]+
lltok::Kind LLLexer::LexPercent() {
  return LexVar(lltok::LocalVar, lltok::LocalVarID);
}

/// Lex all tokens that start with a " character.
///   QuoteLabel        "[^"]+":
///   StringConstant    "[^"]*"
lltok::Kind LLLexer::LexQuote() {
  lltok::Kind kind = ReadString(lltok::StringConstant);
  if (kind == lltok::Error || kind == lltok::Eof)
    return kind;

  if (CurPtr[0] == ':') {
    ++CurPtr;
    if (StringRef(StrVal).find_first_of(0) != StringRef::npos) {
      Error("Null bytes are not allowed in names");
      kind = lltok::Error;
    } else {
      kind = lltok::LabelStr;
    }
  }

  return kind;
}

/// Lex all tokens that start with a ! character.
///    !foo
///    !
lltok::Kind LLLexer::LexExclaim() {
  // Lex a metadata name as a MetadataVar.
  if (isalpha(static_cast<unsigned char>(CurPtr[0])) ||
      CurPtr[0] == '-' || CurPtr[0] == '$' ||
      CurPtr[0] == '.' || CurPtr[0] == '_' || CurPtr[0] == '\\') {
    ++CurPtr;
    while (isalnum(static_cast<unsigned char>(CurPtr[0])) ||
           CurPtr[0] == '-' || CurPtr[0] == '$' ||
           CurPtr[0] == '.' || CurPtr[0] == '_' || CurPtr[0] == '\\')
      ++CurPtr;

    StrVal.assign(TokStart+1, CurPtr);   // Skip !
    UnEscapeLexed(StrVal);
    return lltok::MetadataVar;
  }
  return lltok::exclaim;
}

/// Lex all tokens that start with a # character.
///    AttrGrpID ::= #[0-9]+
lltok::Kind LLLexer::LexHash() {
  // Handle AttrGrpID: #[0-9]+
  if (isdigit(static_cast<unsigned char>(CurPtr[0]))) {
    for (++CurPtr; isdigit(static_cast<unsigned char>(CurPtr[0])); ++CurPtr)
      /*empty*/;

    uint64_t Val = atoull(TokStart+1, CurPtr);
    if ((unsigned)Val != Val)
      Error("invalid value number (too large)!");
    UIntVal = unsigned(Val);
    return lltok::AttrGrpID;
  }

  return lltok::Error;
}

/// Lex a label, integer type, keyword, or hexadecimal integer constant.
///    Label           [-a-zA-Z$._0-9]+:
///    IntegerType     i[0-9]+
///    Keyword         sdiv, float, ...
///    HexIntConstant  [us]0x[0-9A-Fa-f]+
lltok::Kind LLLexer::LexIdentifier() {
  const char *StartChar = CurPtr;
  const char *IntEnd = CurPtr[-1] == 'i' ? nullptr : StartChar;
  const char *KeywordEnd = nullptr;

  for (; isLabelChar(*CurPtr); ++CurPtr) {
    // If we decide this is an integer, remember the end of the sequence.
    if (!IntEnd && !isdigit(static_cast<unsigned char>(*CurPtr)))
      IntEnd = CurPtr;
    if (!KeywordEnd && !isalnum(static_cast<unsigned char>(*CurPtr)) &&
        *CurPtr != '_')
      KeywordEnd = CurPtr;
  }

  // If we stopped due to a colon, this really is a label.
  if (*CurPtr == ':') {
    StrVal.assign(StartChar-1, CurPtr++);
    return lltok::LabelStr;
  }

  // Otherwise, this wasn't a label.  If this was valid as an integer type,
  // return it.
  if (!IntEnd) IntEnd = CurPtr;
  if (IntEnd != StartChar) {
    CurPtr = IntEnd;
    uint64_t NumBits = atoull(StartChar, CurPtr);
    if (NumBits < IntegerType::MIN_INT_BITS ||
        NumBits > IntegerType::MAX_INT_BITS) {
      Error("bitwidth for integer type out of range!");
      return lltok::Error;
    }
    TyVal = IntegerType::get(Context, NumBits);
    return lltok::Type;
  }

  // Otherwise, this was a letter sequence.  See which keyword this is.
  if (!KeywordEnd) KeywordEnd = CurPtr;
  CurPtr = KeywordEnd;
  --StartChar;
  StringRef Keyword(StartChar, CurPtr - StartChar);
#define KEYWORD(STR)                                                           \
  do {                                                                         \
    if (Keyword == #STR)                                                       \
      return lltok::kw_##STR;                                                  \
  } while (0)

  KEYWORD(true);    KEYWORD(false);
  KEYWORD(declare); KEYWORD(define);
  KEYWORD(global);  KEYWORD(constant);

  KEYWORD(private);
  KEYWORD(internal);
  KEYWORD(available_externally);
  KEYWORD(linkonce);
  KEYWORD(linkonce_odr);
  KEYWORD(weak); // Use as a linkage, and a modifier for "cmpxchg".
  KEYWORD(weak_odr);
  KEYWORD(appending);
  KEYWORD(dllimport);
  KEYWORD(dllexport);
  KEYWORD(common);
  KEYWORD(default);
  KEYWORD(hidden);
  KEYWORD(protected);
  KEYWORD(unnamed_addr);
  KEYWORD(externally_initialized);
  KEYWORD(extern_weak);
  KEYWORD(external);
  KEYWORD(thread_local);
  KEYWORD(localdynamic);
  KEYWORD(initialexec);
  KEYWORD(localexec);
  KEYWORD(zeroinitializer);
  KEYWORD(undef);
  KEYWORD(null);
  KEYWORD(to);
  KEYWORD(tail);
  KEYWORD(musttail);
  KEYWORD(target);
  KEYWORD(triple);
  KEYWORD(unwind);
  KEYWORD(deplibs);             // FIXME: Remove in 4.0.
  KEYWORD(datalayout);
  KEYWORD(volatile);
  KEYWORD(atomic);
  KEYWORD(unordered);
  KEYWORD(monotonic);
  KEYWORD(acquire);
  KEYWORD(release);
  KEYWORD(acq_rel);
  KEYWORD(seq_cst);
  KEYWORD(singlethread);

  KEYWORD(nnan);
  KEYWORD(ninf);
  KEYWORD(nsz);
  KEYWORD(arcp);
  KEYWORD(fast);
  KEYWORD(nuw);
  KEYWORD(nsw);
  KEYWORD(exact);
  KEYWORD(inbounds);
  KEYWORD(align);
  KEYWORD(addrspace);
  KEYWORD(section);
  KEYWORD(alias);
  KEYWORD(module);
  KEYWORD(asm);
  KEYWORD(sideeffect);
  KEYWORD(alignstack);
  KEYWORD(inteldialect);
  KEYWORD(gc);
  KEYWORD(prefix);
  KEYWORD(prologue);

  KEYWORD(ccc);
  KEYWORD(fastcc);
  KEYWORD(coldcc);
  KEYWORD(x86_stdcallcc);
  KEYWORD(x86_fastcallcc);
  KEYWORD(x86_thiscallcc);
  KEYWORD(x86_vectorcallcc);
  KEYWORD(arm_apcscc);
  KEYWORD(arm_aapcscc);
  KEYWORD(arm_aapcs_vfpcc);
  KEYWORD(msp430_intrcc);
  KEYWORD(ptx_kernel);
  KEYWORD(ptx_device);
  KEYWORD(spir_kernel);
  KEYWORD(spir_func);
  KEYWORD(intel_ocl_bicc);
  KEYWORD(x86_64_sysvcc);
  KEYWORD(x86_64_win64cc);
  KEYWORD(webkit_jscc);
  KEYWORD(anyregcc);
  KEYWORD(preserve_mostcc);
  KEYWORD(preserve_allcc);
  KEYWORD(ghccc);

  KEYWORD(cc);
  KEYWORD(c);

  KEYWORD(attributes);

  KEYWORD(alwaysinline);
  KEYWORD(argmemonly);
  KEYWORD(builtin);
  KEYWORD(byval);
  KEYWORD(inalloca);
  KEYWORD(cold);
  KEYWORD(convergent);
  KEYWORD(dereferenceable);
  KEYWORD(dereferenceable_or_null);
  KEYWORD(inlinehint);
  KEYWORD(inreg);
  KEYWORD(jumptable);
  KEYWORD(minsize);
  KEYWORD(naked);
  KEYWORD(nest);
  KEYWORD(noalias);
  KEYWORD(nobuiltin);
  KEYWORD(nocapture);
  KEYWORD(noduplicate);
  KEYWORD(noimplicitfloat);
  KEYWORD(noinline);
  KEYWORD(nonlazybind);
  KEYWORD(nonnull);
  KEYWORD(noredzone);
  KEYWORD(noreturn);
  KEYWORD(nounwind);
  KEYWORD(optnone);
  KEYWORD(optsize);
  KEYWORD(readnone);
  KEYWORD(readonly);
  KEYWORD(returned);
  KEYWORD(returns_twice);
  KEYWORD(signext);
  KEYWORD(sret);
  KEYWORD(ssp);
  KEYWORD(sspreq);
  KEYWORD(sspstrong);
  KEYWORD(safestack);
  KEYWORD(sanitize_address);
  KEYWORD(sanitize_thread);
  KEYWORD(sanitize_memory);
  KEYWORD(uwtable);
  KEYWORD(zeroext);

  KEYWORD(type);
  KEYWORD(opaque);

  KEYWORD(comdat);

  // Comdat types
  KEYWORD(any);
  KEYWORD(exactmatch);
  KEYWORD(largest);
  KEYWORD(noduplicates);
  KEYWORD(samesize);

  KEYWORD(eq); KEYWORD(ne); KEYWORD(slt); KEYWORD(sgt); KEYWORD(sle);
  KEYWORD(sge); KEYWORD(ult); KEYWORD(ugt); KEYWORD(ule); KEYWORD(uge);
  KEYWORD(oeq); KEYWORD(one); KEYWORD(olt); KEYWORD(ogt); KEYWORD(ole);
  KEYWORD(oge); KEYWORD(ord); KEYWORD(uno); KEYWORD(ueq); KEYWORD(une);

  KEYWORD(xchg); KEYWORD(nand); KEYWORD(max); KEYWORD(min); KEYWORD(umax);
  KEYWORD(umin);

  KEYWORD(x);
  KEYWORD(blockaddress);

  // Metadata types.
  KEYWORD(distinct);

  // Use-list order directives.
  KEYWORD(uselistorder);
  KEYWORD(uselistorder_bb);

  KEYWORD(personality);
  KEYWORD(cleanup);
  KEYWORD(catch);
  KEYWORD(filter);
#undef KEYWORD

  // Keywords for types.
#define TYPEKEYWORD(STR, LLVMTY)                                               \
  do {                                                                         \
    if (Keyword == STR) {                                                      \
      TyVal = LLVMTY;                                                          \
      return lltok::Type;                                                      \
    }                                                                          \
  } while (false)
  TYPEKEYWORD("void",      Type::getVoidTy(Context));
  TYPEKEYWORD("half",      Type::getHalfTy(Context));
  TYPEKEYWORD("float",     Type::getFloatTy(Context));
  TYPEKEYWORD("double",    Type::getDoubleTy(Context));
  TYPEKEYWORD("x86_fp80",  Type::getX86_FP80Ty(Context));
  TYPEKEYWORD("fp128",     Type::getFP128Ty(Context));
  TYPEKEYWORD("ppc_fp128", Type::getPPC_FP128Ty(Context));
  TYPEKEYWORD("label",     Type::getLabelTy(Context));
  TYPEKEYWORD("metadata",  Type::getMetadataTy(Context));
  TYPEKEYWORD("x86_mmx",   Type::getX86_MMXTy(Context));
#undef TYPEKEYWORD

  // Keywords for instructions.
#define INSTKEYWORD(STR, Enum)                                                 \
  do {                                                                         \
    if (Keyword == #STR) {                                                     \
      UIntVal = Instruction::Enum;                                             \
      return lltok::kw_##STR;                                                  \
    }                                                                          \
  } while (false)

  INSTKEYWORD(add,   Add);  INSTKEYWORD(fadd,   FAdd);
  INSTKEYWORD(sub,   Sub);  INSTKEYWORD(fsub,   FSub);
  INSTKEYWORD(mul,   Mul);  INSTKEYWORD(fmul,   FMul);
  INSTKEYWORD(udiv,  UDiv); INSTKEYWORD(sdiv,  SDiv); INSTKEYWORD(fdiv,  FDiv);
  INSTKEYWORD(urem,  URem); INSTKEYWORD(srem,  SRem); INSTKEYWORD(frem,  FRem);
  INSTKEYWORD(shl,   Shl);  INSTKEYWORD(lshr,  LShr); INSTKEYWORD(ashr,  AShr);
  INSTKEYWORD(and,   And);  INSTKEYWORD(or,    Or);   INSTKEYWORD(xor,   Xor);
  INSTKEYWORD(icmp,  ICmp); INSTKEYWORD(fcmp,  FCmp);

  INSTKEYWORD(phi,         PHI);
  INSTKEYWORD(call,        Call);
  INSTKEYWORD(trunc,       Trunc);
  INSTKEYWORD(zext,        ZExt);
  INSTKEYWORD(sext,        SExt);
  INSTKEYWORD(fptrunc,     FPTrunc);
  INSTKEYWORD(fpext,       FPExt);
  INSTKEYWORD(uitofp,      UIToFP);
  INSTKEYWORD(sitofp,      SIToFP);
  INSTKEYWORD(fptoui,      FPToUI);
  INSTKEYWORD(fptosi,      FPToSI);
  INSTKEYWORD(inttoptr,    IntToPtr);
  INSTKEYWORD(ptrtoint,    PtrToInt);
  INSTKEYWORD(bitcast,     BitCast);
  INSTKEYWORD(addrspacecast, AddrSpaceCast);
  INSTKEYWORD(select,      Select);
  INSTKEYWORD(va_arg,      VAArg);
  INSTKEYWORD(ret,         Ret);
  INSTKEYWORD(br,          Br);
  INSTKEYWORD(switch,      Switch);
  INSTKEYWORD(indirectbr,  IndirectBr);
  INSTKEYWORD(invoke,      Invoke);
  INSTKEYWORD(resume,      Resume);
  INSTKEYWORD(unreachable, Unreachable);

  INSTKEYWORD(alloca,      Alloca);
  INSTKEYWORD(load,        Load);
  INSTKEYWORD(store,       Store);
  INSTKEYWORD(cmpxchg,     AtomicCmpXchg);
  INSTKEYWORD(atomicrmw,   AtomicRMW);
  INSTKEYWORD(fence,       Fence);
  INSTKEYWORD(getelementptr, GetElementPtr);

  INSTKEYWORD(extractelement, ExtractElement);
  INSTKEYWORD(insertelement,  InsertElement);
  INSTKEYWORD(shufflevector,  ShuffleVector);
  INSTKEYWORD(extractvalue,   ExtractValue);
  INSTKEYWORD(insertvalue,    InsertValue);
  INSTKEYWORD(landingpad,     LandingPad);
#undef INSTKEYWORD

#define DWKEYWORD(TYPE, TOKEN)                                                 \
  do {                                                                         \
    if (Keyword.startswith("DW_" #TYPE "_")) {                                 \
      StrVal.assign(Keyword.begin(), Keyword.end());                           \
      return lltok::TOKEN;                                                     \
    }                                                                          \
  } while (false)
  DWKEYWORD(TAG, DwarfTag);
  DWKEYWORD(ATE, DwarfAttEncoding);
  DWKEYWORD(VIRTUALITY, DwarfVirtuality);
  DWKEYWORD(LANG, DwarfLang);
  DWKEYWORD(OP, DwarfOp);
#undef DWKEYWORD

  if (Keyword.startswith("DIFlag")) {
    StrVal.assign(Keyword.begin(), Keyword.end());
    return lltok::DIFlag;
  }

  // Check for [us]0x[0-9A-Fa-f]+ which are Hexadecimal constant generated by
  // the CFE to avoid forcing it to deal with 64-bit numbers.
  if ((TokStart[0] == 'u' || TokStart[0] == 's') &&
      TokStart[1] == '0' && TokStart[2] == 'x' &&
      isxdigit(static_cast<unsigned char>(TokStart[3]))) {
    int len = CurPtr-TokStart-3;
    uint32_t bits = len * 4;
    StringRef HexStr(TokStart + 3, len);
    if (!std::all_of(HexStr.begin(), HexStr.end(), isxdigit)) {
      // Bad token, return it as an error.
      CurPtr = TokStart+3;
      return lltok::Error;
    }
    APInt Tmp(bits, HexStr, 16);
    uint32_t activeBits = Tmp.getActiveBits();
    if (activeBits > 0 && activeBits < bits)
      Tmp = Tmp.trunc(activeBits);
    APSIntVal = APSInt(Tmp, TokStart[0] == 'u');
    return lltok::APSInt;
  }

  // If this is "cc1234", return this as just "cc".
  if (TokStart[0] == 'c' && TokStart[1] == 'c') {
    CurPtr = TokStart+2;
    return lltok::kw_cc;
  }

  // Finally, if this isn't known, return an error.
  CurPtr = TokStart+1;
  return lltok::Error;
}

/// Lex all tokens that start with a 0x prefix, knowing they match and are not
/// labels.
///    HexFPConstant     0x[0-9A-Fa-f]+
///    HexFP80Constant   0xK[0-9A-Fa-f]+
///    HexFP128Constant  0xL[0-9A-Fa-f]+
///    HexPPC128Constant 0xM[0-9A-Fa-f]+
///    HexHalfConstant   0xH[0-9A-Fa-f]+
lltok::Kind LLLexer::Lex0x() {
  CurPtr = TokStart + 2;

  char Kind;
  if ((CurPtr[0] >= 'K' && CurPtr[0] <= 'M') || CurPtr[0] == 'H') {
    Kind = *CurPtr++;
  } else {
    Kind = 'J';
  }

  if (!isxdigit(static_cast<unsigned char>(CurPtr[0]))) {
    // Bad token, return it as an error.
    CurPtr = TokStart+1;
    return lltok::Error;
  }

  while (isxdigit(static_cast<unsigned char>(CurPtr[0])))
    ++CurPtr;

  if (Kind == 'J') {
    // HexFPConstant - Floating point constant represented in IEEE format as a
    // hexadecimal number for when exponential notation is not precise enough.
    // Half, Float, and double only.
    APFloatVal = APFloat(BitsToDouble(HexIntToVal(TokStart+2, CurPtr)));
    return lltok::APFloat;
  }

  uint64_t Pair[2];
  switch (Kind) {
  default: llvm_unreachable("Unknown kind!");
  case 'K':
    // F80HexFPConstant - x87 long double in hexadecimal format (10 bytes)
    FP80HexToIntPair(TokStart+3, CurPtr, Pair);
    APFloatVal = APFloat(APFloat::x87DoubleExtended, APInt(80, Pair));
    return lltok::APFloat;
  case 'L':
    // F128HexFPConstant - IEEE 128-bit in hexadecimal format (16 bytes)
    HexToIntPair(TokStart+3, CurPtr, Pair);
    APFloatVal = APFloat(APFloat::IEEEquad, APInt(128, Pair));
    return lltok::APFloat;
  case 'M':
    // PPC128HexFPConstant - PowerPC 128-bit in hexadecimal format (16 bytes)
    HexToIntPair(TokStart+3, CurPtr, Pair);
    APFloatVal = APFloat(APFloat::PPCDoubleDouble, APInt(128, Pair));
    return lltok::APFloat;
  case 'H':
    APFloatVal = APFloat(APFloat::IEEEhalf,
                         APInt(16,HexIntToVal(TokStart+3, CurPtr)));
    return lltok::APFloat;
  }
}

/// Lex tokens for a label or a numeric constant, possibly starting with -.
///    Label             [-a-zA-Z$._0-9]+:
///    NInteger          -[0-9]+
///    FPConstant        [-+]?[0-9]+[.][0-9]*([eE][-+]?[0-9]+)?
///    PInteger          [0-9]+
///    HexFPConstant     0x[0-9A-Fa-f]+
///    HexFP80Constant   0xK[0-9A-Fa-f]+
///    HexFP128Constant  0xL[0-9A-Fa-f]+
///    HexPPC128Constant 0xM[0-9A-Fa-f]+
lltok::Kind LLLexer::LexDigitOrNegative() {
  // If the letter after the negative is not a number, this is probably a label.
  if (!isdigit(static_cast<unsigned char>(TokStart[0])) &&
      !isdigit(static_cast<unsigned char>(CurPtr[0]))) {
    // Okay, this is not a number after the -, it's probably a label.
    if (const char *End = isLabelTail(CurPtr)) {
      StrVal.assign(TokStart, End-1);
      CurPtr = End;
      return lltok::LabelStr;
    }

    return lltok::Error;
  }

  // At this point, it is either a label, int or fp constant.

  // Skip digits, we have at least one.
  for (; isdigit(static_cast<unsigned char>(CurPtr[0])); ++CurPtr)
    /*empty*/;

  // Check to see if this really is a label afterall, e.g. "-1:".
  if (isLabelChar(CurPtr[0]) || CurPtr[0] == ':') {
    if (const char *End = isLabelTail(CurPtr)) {
      StrVal.assign(TokStart, End-1);
      CurPtr = End;
      return lltok::LabelStr;
    }
  }

  // If the next character is a '.', then it is a fp value, otherwise its
  // integer.
  if (CurPtr[0] != '.') {
    if (TokStart[0] == '0' && TokStart[1] == 'x')
      return Lex0x();
    APSIntVal = APSInt(StringRef(TokStart, CurPtr - TokStart));
    return lltok::APSInt;
  }

  ++CurPtr;

  // Skip over [0-9]*([eE][-+]?[0-9]+)?
  while (isdigit(static_cast<unsigned char>(CurPtr[0]))) ++CurPtr;

  if (CurPtr[0] == 'e' || CurPtr[0] == 'E') {
    if (isdigit(static_cast<unsigned char>(CurPtr[1])) ||
        ((CurPtr[1] == '-' || CurPtr[1] == '+') &&
          isdigit(static_cast<unsigned char>(CurPtr[2])))) {
      CurPtr += 2;
      while (isdigit(static_cast<unsigned char>(CurPtr[0]))) ++CurPtr;
    }
  }

  APFloatVal = APFloat(std::atof(TokStart));
  return lltok::APFloat;
}

/// Lex a floating point constant starting with +.
///    FPConstant  [-+]?[0-9]+[.][0-9]*([eE][-+]?[0-9]+)?
lltok::Kind LLLexer::LexPositive() {
  // If the letter after the negative is a number, this is probably not a
  // label.
  if (!isdigit(static_cast<unsigned char>(CurPtr[0])))
    return lltok::Error;

  // Skip digits.
  for (++CurPtr; isdigit(static_cast<unsigned char>(CurPtr[0])); ++CurPtr)
    /*empty*/;

  // At this point, we need a '.'.
  if (CurPtr[0] != '.') {
    CurPtr = TokStart+1;
    return lltok::Error;
  }

  ++CurPtr;

  // Skip over [0-9]*([eE][-+]?[0-9]+)?
  while (isdigit(static_cast<unsigned char>(CurPtr[0]))) ++CurPtr;

  if (CurPtr[0] == 'e' || CurPtr[0] == 'E') {
    if (isdigit(static_cast<unsigned char>(CurPtr[1])) ||
        ((CurPtr[1] == '-' || CurPtr[1] == '+') &&
        isdigit(static_cast<unsigned char>(CurPtr[2])))) {
      CurPtr += 2;
      while (isdigit(static_cast<unsigned char>(CurPtr[0]))) ++CurPtr;
    }
  }

  APFloatVal = APFloat(std::atof(TokStart));
  return lltok::APFloat;
}
