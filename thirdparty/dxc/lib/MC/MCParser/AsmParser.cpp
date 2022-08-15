//===- AsmParser.cpp - Parser for Assembly Files --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class implements the parser for assembly files.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCParser/AsmCond.h"
#include "llvm/MC/MCParser/AsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCAsmParserUtils.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCTargetAsmParser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <cctype>
#include <deque>
#include <set>
#include <string>
#include <vector>
using namespace llvm;

MCAsmParserSemaCallback::~MCAsmParserSemaCallback() {}

namespace {
/// \brief Helper types for tracking macro definitions.
typedef std::vector<AsmToken> MCAsmMacroArgument;
typedef std::vector<MCAsmMacroArgument> MCAsmMacroArguments;

struct MCAsmMacroParameter {
  StringRef Name;
  MCAsmMacroArgument Value;
  bool Required;
  bool Vararg;

  MCAsmMacroParameter() : Required(false), Vararg(false) {}
};

typedef std::vector<MCAsmMacroParameter> MCAsmMacroParameters;

struct MCAsmMacro {
  StringRef Name;
  StringRef Body;
  MCAsmMacroParameters Parameters;

public:
  MCAsmMacro(StringRef N, StringRef B, MCAsmMacroParameters P)
      : Name(N), Body(B), Parameters(std::move(P)) {}
};

/// \brief Helper class for storing information about an active macro
/// instantiation.
struct MacroInstantiation {
  /// The location of the instantiation.
  SMLoc InstantiationLoc;

  /// The buffer where parsing should resume upon instantiation completion.
  int ExitBuffer;

  /// The location where parsing should resume upon instantiation completion.
  SMLoc ExitLoc;

  /// The depth of TheCondStack at the start of the instantiation.
  size_t CondStackDepth;

public:
  MacroInstantiation(SMLoc IL, int EB, SMLoc EL, size_t CondStackDepth);
};

struct ParseStatementInfo {
  /// \brief The parsed operands from the last parsed statement.
  SmallVector<std::unique_ptr<MCParsedAsmOperand>, 8> ParsedOperands;

  /// \brief The opcode from the last parsed instruction.
  unsigned Opcode;

  /// \brief Was there an error parsing the inline assembly?
  bool ParseError;

  SmallVectorImpl<AsmRewrite> *AsmRewrites;

  ParseStatementInfo() : Opcode(~0U), ParseError(false), AsmRewrites(nullptr) {}
  ParseStatementInfo(SmallVectorImpl<AsmRewrite> *rewrites)
    : Opcode(~0), ParseError(false), AsmRewrites(rewrites) {}
};

/// \brief The concrete assembly parser instance.
class AsmParser : public MCAsmParser {
  AsmParser(const AsmParser &) = delete;
  void operator=(const AsmParser &) = delete;
private:
  AsmLexer Lexer;
  MCContext &Ctx;
  MCStreamer &Out;
  const MCAsmInfo &MAI;
  SourceMgr &SrcMgr;
  SourceMgr::DiagHandlerTy SavedDiagHandler;
  void *SavedDiagContext;
  std::unique_ptr<MCAsmParserExtension> PlatformParser;

  /// This is the current buffer index we're lexing from as managed by the
  /// SourceMgr object.
  unsigned CurBuffer;

  AsmCond TheCondState;
  std::vector<AsmCond> TheCondStack;

  /// \brief maps directive names to handler methods in parser
  /// extensions. Extensions register themselves in this map by calling
  /// addDirectiveHandler.
  StringMap<ExtensionDirectiveHandler> ExtensionDirectiveMap;

  /// \brief Map of currently defined macros.
  StringMap<MCAsmMacro> MacroMap;

  /// \brief Stack of active macro instantiations.
  std::vector<MacroInstantiation*> ActiveMacros;

  /// \brief List of bodies of anonymous macros.
  std::deque<MCAsmMacro> MacroLikeBodies;

  /// Boolean tracking whether macro substitution is enabled.
  unsigned MacrosEnabledFlag : 1;

  /// \brief Keeps track of how many .macro's have been instantiated.
  unsigned NumOfMacroInstantiations;

  /// Flag tracking whether any errors have been encountered.
  unsigned HadError : 1;

  /// The values from the last parsed cpp hash file line comment if any.
  StringRef CppHashFilename;
  int64_t CppHashLineNumber;
  SMLoc CppHashLoc;
  unsigned CppHashBuf;
  /// When generating dwarf for assembly source files we need to calculate the
  /// logical line number based on the last parsed cpp hash file line comment
  /// and current line. Since this is slow and messes up the SourceMgr's
  /// cache we save the last info we queried with SrcMgr.FindLineNumber().
  SMLoc LastQueryIDLoc;
  unsigned LastQueryBuffer;
  unsigned LastQueryLine;

  /// AssemblerDialect. ~OU means unset value and use value provided by MAI.
  unsigned AssemblerDialect;

  /// \brief is Darwin compatibility enabled?
  bool IsDarwin;

  /// \brief Are we parsing ms-style inline assembly?
  bool ParsingInlineAsm;

public:
  AsmParser(SourceMgr &SM, MCContext &Ctx, MCStreamer &Out,
            const MCAsmInfo &MAI);
  ~AsmParser() override;

  bool Run(bool NoInitialTextSection, bool NoFinalize = false) override;

  void addDirectiveHandler(StringRef Directive,
                           ExtensionDirectiveHandler Handler) override {
    ExtensionDirectiveMap[Directive] = Handler;
  }

  void addAliasForDirective(StringRef Directive, StringRef Alias) override {
    DirectiveKindMap[Directive] = DirectiveKindMap[Alias];
  }

public:
  /// @name MCAsmParser Interface
  /// {

  SourceMgr &getSourceManager() override { return SrcMgr; }
  MCAsmLexer &getLexer() override { return Lexer; }
  MCContext &getContext() override { return Ctx; }
  MCStreamer &getStreamer() override { return Out; }
  unsigned getAssemblerDialect() override {
    if (AssemblerDialect == ~0U)
      return MAI.getAssemblerDialect();
    else
      return AssemblerDialect;
  }
  void setAssemblerDialect(unsigned i) override {
    AssemblerDialect = i;
  }

  void Note(SMLoc L, const Twine &Msg,
            ArrayRef<SMRange> Ranges = None) override;
  bool Warning(SMLoc L, const Twine &Msg,
               ArrayRef<SMRange> Ranges = None) override;
  bool Error(SMLoc L, const Twine &Msg,
             ArrayRef<SMRange> Ranges = None) override;

  const AsmToken &Lex() override;

  void setParsingInlineAsm(bool V) override { ParsingInlineAsm = V; }
  bool isParsingInlineAsm() override { return ParsingInlineAsm; }

  bool parseMSInlineAsm(void *AsmLoc, std::string &AsmString,
                        unsigned &NumOutputs, unsigned &NumInputs,
                        SmallVectorImpl<std::pair<void *,bool> > &OpDecls,
                        SmallVectorImpl<std::string> &Constraints,
                        SmallVectorImpl<std::string> &Clobbers,
                        const MCInstrInfo *MII, const MCInstPrinter *IP,
                        MCAsmParserSemaCallback &SI) override;

  bool parseExpression(const MCExpr *&Res);
  bool parseExpression(const MCExpr *&Res, SMLoc &EndLoc) override;
  bool parsePrimaryExpr(const MCExpr *&Res, SMLoc &EndLoc) override;
  bool parseParenExpression(const MCExpr *&Res, SMLoc &EndLoc) override;
  bool parseParenExprOfDepth(unsigned ParenDepth, const MCExpr *&Res,
                             SMLoc &EndLoc) override;
  bool parseAbsoluteExpression(int64_t &Res) override;

  /// \brief Parse an identifier or string (as a quoted identifier)
  /// and set \p Res to the identifier contents.
  bool parseIdentifier(StringRef &Res) override;
  void eatToEndOfStatement() override;

  void checkForValidSection() override;
  /// }

private:

  bool parseStatement(ParseStatementInfo &Info,
                      MCAsmParserSemaCallback *SI);
  void eatToEndOfLine();
  bool parseCppHashLineFilenameComment(const SMLoc &L);

  void checkForBadMacro(SMLoc DirectiveLoc, StringRef Name, StringRef Body,
                        ArrayRef<MCAsmMacroParameter> Parameters);
  bool expandMacro(raw_svector_ostream &OS, StringRef Body,
                   ArrayRef<MCAsmMacroParameter> Parameters,
                   ArrayRef<MCAsmMacroArgument> A, bool EnableAtPseudoVariable,
                   const SMLoc &L);

  /// \brief Are macros enabled in the parser?
  bool areMacrosEnabled() {return MacrosEnabledFlag;}

  /// \brief Control a flag in the parser that enables or disables macros.
  void setMacrosEnabled(bool Flag) {MacrosEnabledFlag = Flag;}

  /// \brief Lookup a previously defined macro.
  /// \param Name Macro name.
  /// \returns Pointer to macro. NULL if no such macro was defined.
  const MCAsmMacro* lookupMacro(StringRef Name);

  /// \brief Define a new macro with the given name and information.
  void defineMacro(StringRef Name, MCAsmMacro Macro);

  /// \brief Undefine a macro. If no such macro was defined, it's a no-op.
  void undefineMacro(StringRef Name);

  /// \brief Are we inside a macro instantiation?
  bool isInsideMacroInstantiation() {return !ActiveMacros.empty();}

  /// \brief Handle entry to macro instantiation.
  ///
  /// \param M The macro.
  /// \param NameLoc Instantiation location.
  bool handleMacroEntry(const MCAsmMacro *M, SMLoc NameLoc);

  /// \brief Handle exit from macro instantiation.
  void handleMacroExit();

  /// \brief Extract AsmTokens for a macro argument.
  bool parseMacroArgument(MCAsmMacroArgument &MA, bool Vararg);

  /// \brief Parse all macro arguments for a given macro.
  bool parseMacroArguments(const MCAsmMacro *M, MCAsmMacroArguments &A);

  void printMacroInstantiations();
  void printMessage(SMLoc Loc, SourceMgr::DiagKind Kind, const Twine &Msg,
                    ArrayRef<SMRange> Ranges = None) const {
    SrcMgr.PrintMessage(Loc, Kind, Msg, Ranges);
  }
  static void DiagHandler(const SMDiagnostic &Diag, void *Context);

  /// \brief Enter the specified file. This returns true on failure.
  bool enterIncludeFile(const std::string &Filename);

  /// \brief Process the specified file for the .incbin directive.
  /// This returns true on failure.
  bool processIncbinFile(const std::string &Filename);

  /// \brief Reset the current lexer position to that given by \p Loc. The
  /// current token is not set; clients should ensure Lex() is called
  /// subsequently.
  ///
  /// \param InBuffer If not 0, should be the known buffer id that contains the
  /// location.
  void jumpToLoc(SMLoc Loc, unsigned InBuffer = 0);

  /// \brief Parse up to the end of statement and a return the contents from the
  /// current token until the end of the statement; the current token on exit
  /// will be either the EndOfStatement or EOF.
  StringRef parseStringToEndOfStatement() override;

  /// \brief Parse until the end of a statement or a comma is encountered,
  /// return the contents from the current token up to the end or comma.
  StringRef parseStringToComma();

  bool parseAssignment(StringRef Name, bool allow_redef,
                       bool NoDeadStrip = false);

  unsigned getBinOpPrecedence(AsmToken::TokenKind K,
                              MCBinaryExpr::Opcode &Kind);

  bool parseBinOpRHS(unsigned Precedence, const MCExpr *&Res, SMLoc &EndLoc);
  bool parseParenExpr(const MCExpr *&Res, SMLoc &EndLoc);
  bool parseBracketExpr(const MCExpr *&Res, SMLoc &EndLoc);

  bool parseRegisterOrRegisterNumber(int64_t &Register, SMLoc DirectiveLoc);

  // Generic (target and platform independent) directive parsing.
  enum DirectiveKind {
    DK_NO_DIRECTIVE, // Placeholder
    DK_SET, DK_EQU, DK_EQUIV, DK_ASCII, DK_ASCIZ, DK_STRING, DK_BYTE, DK_SHORT,
    DK_VALUE, DK_2BYTE, DK_LONG, DK_INT, DK_4BYTE, DK_QUAD, DK_8BYTE, DK_OCTA,
    DK_SINGLE, DK_FLOAT, DK_DOUBLE, DK_ALIGN, DK_ALIGN32, DK_BALIGN, DK_BALIGNW,
    DK_BALIGNL, DK_P2ALIGN, DK_P2ALIGNW, DK_P2ALIGNL, DK_ORG, DK_FILL, DK_ENDR,
    DK_BUNDLE_ALIGN_MODE, DK_BUNDLE_LOCK, DK_BUNDLE_UNLOCK,
    DK_ZERO, DK_EXTERN, DK_GLOBL, DK_GLOBAL,
    DK_LAZY_REFERENCE, DK_NO_DEAD_STRIP, DK_SYMBOL_RESOLVER, DK_PRIVATE_EXTERN,
    DK_REFERENCE, DK_WEAK_DEFINITION, DK_WEAK_REFERENCE,
    DK_WEAK_DEF_CAN_BE_HIDDEN, DK_COMM, DK_COMMON, DK_LCOMM, DK_ABORT,
    DK_INCLUDE, DK_INCBIN, DK_CODE16, DK_CODE16GCC, DK_REPT, DK_IRP, DK_IRPC,
    DK_IF, DK_IFEQ, DK_IFGE, DK_IFGT, DK_IFLE, DK_IFLT, DK_IFNE, DK_IFB,
    DK_IFNB, DK_IFC, DK_IFEQS, DK_IFNC, DK_IFNES, DK_IFDEF, DK_IFNDEF,
    DK_IFNOTDEF, DK_ELSEIF, DK_ELSE, DK_ENDIF,
    DK_SPACE, DK_SKIP, DK_FILE, DK_LINE, DK_LOC, DK_STABS,
    DK_CFI_SECTIONS, DK_CFI_STARTPROC, DK_CFI_ENDPROC, DK_CFI_DEF_CFA,
    DK_CFI_DEF_CFA_OFFSET, DK_CFI_ADJUST_CFA_OFFSET, DK_CFI_DEF_CFA_REGISTER,
    DK_CFI_OFFSET, DK_CFI_REL_OFFSET, DK_CFI_PERSONALITY, DK_CFI_LSDA,
    DK_CFI_REMEMBER_STATE, DK_CFI_RESTORE_STATE, DK_CFI_SAME_VALUE,
    DK_CFI_RESTORE, DK_CFI_ESCAPE, DK_CFI_SIGNAL_FRAME, DK_CFI_UNDEFINED,
    DK_CFI_REGISTER, DK_CFI_WINDOW_SAVE,
    DK_MACROS_ON, DK_MACROS_OFF,
    DK_MACRO, DK_EXITM, DK_ENDM, DK_ENDMACRO, DK_PURGEM,
    DK_SLEB128, DK_ULEB128,
    DK_ERR, DK_ERROR, DK_WARNING,
    DK_END
  };

  /// \brief Maps directive name --> DirectiveKind enum, for
  /// directives parsed by this class.
  StringMap<DirectiveKind> DirectiveKindMap;

  // ".ascii", ".asciz", ".string"
  bool parseDirectiveAscii(StringRef IDVal, bool ZeroTerminated);
  bool parseDirectiveValue(unsigned Size); // ".byte", ".long", ...
  bool parseDirectiveOctaValue(); // ".octa"
  bool parseDirectiveRealValue(const fltSemantics &); // ".single", ...
  bool parseDirectiveFill(); // ".fill"
  bool parseDirectiveZero(); // ".zero"
  // ".set", ".equ", ".equiv"
  bool parseDirectiveSet(StringRef IDVal, bool allow_redef);
  bool parseDirectiveOrg(); // ".org"
  // ".align{,32}", ".p2align{,w,l}"
  bool parseDirectiveAlign(bool IsPow2, unsigned ValueSize);

  // ".file", ".line", ".loc", ".stabs"
  bool parseDirectiveFile(SMLoc DirectiveLoc);
  bool parseDirectiveLine();
  bool parseDirectiveLoc();
  bool parseDirectiveStabs();

  // .cfi directives
  bool parseDirectiveCFIRegister(SMLoc DirectiveLoc);
  bool parseDirectiveCFIWindowSave();
  bool parseDirectiveCFISections();
  bool parseDirectiveCFIStartProc();
  bool parseDirectiveCFIEndProc();
  bool parseDirectiveCFIDefCfaOffset();
  bool parseDirectiveCFIDefCfa(SMLoc DirectiveLoc);
  bool parseDirectiveCFIAdjustCfaOffset();
  bool parseDirectiveCFIDefCfaRegister(SMLoc DirectiveLoc);
  bool parseDirectiveCFIOffset(SMLoc DirectiveLoc);
  bool parseDirectiveCFIRelOffset(SMLoc DirectiveLoc);
  bool parseDirectiveCFIPersonalityOrLsda(bool IsPersonality);
  bool parseDirectiveCFIRememberState();
  bool parseDirectiveCFIRestoreState();
  bool parseDirectiveCFISameValue(SMLoc DirectiveLoc);
  bool parseDirectiveCFIRestore(SMLoc DirectiveLoc);
  bool parseDirectiveCFIEscape();
  bool parseDirectiveCFISignalFrame();
  bool parseDirectiveCFIUndefined(SMLoc DirectiveLoc);

  // macro directives
  bool parseDirectivePurgeMacro(SMLoc DirectiveLoc);
  bool parseDirectiveExitMacro(StringRef Directive);
  bool parseDirectiveEndMacro(StringRef Directive);
  bool parseDirectiveMacro(SMLoc DirectiveLoc);
  bool parseDirectiveMacrosOnOff(StringRef Directive);

  // ".bundle_align_mode"
  bool parseDirectiveBundleAlignMode();
  // ".bundle_lock"
  bool parseDirectiveBundleLock();
  // ".bundle_unlock"
  bool parseDirectiveBundleUnlock();

  // ".space", ".skip"
  bool parseDirectiveSpace(StringRef IDVal);

  // .sleb128 (Signed=true) and .uleb128 (Signed=false)
  bool parseDirectiveLEB128(bool Signed);

  /// \brief Parse a directive like ".globl" which
  /// accepts a single symbol (which should be a label or an external).
  bool parseDirectiveSymbolAttribute(MCSymbolAttr Attr);

  bool parseDirectiveComm(bool IsLocal); // ".comm" and ".lcomm"

  bool parseDirectiveAbort(); // ".abort"
  bool parseDirectiveInclude(); // ".include"
  bool parseDirectiveIncbin(); // ".incbin"

  // ".if", ".ifeq", ".ifge", ".ifgt" , ".ifle", ".iflt" or ".ifne"
  bool parseDirectiveIf(SMLoc DirectiveLoc, DirectiveKind DirKind);
  // ".ifb" or ".ifnb", depending on ExpectBlank.
  bool parseDirectiveIfb(SMLoc DirectiveLoc, bool ExpectBlank);
  // ".ifc" or ".ifnc", depending on ExpectEqual.
  bool parseDirectiveIfc(SMLoc DirectiveLoc, bool ExpectEqual);
  // ".ifeqs" or ".ifnes", depending on ExpectEqual.
  bool parseDirectiveIfeqs(SMLoc DirectiveLoc, bool ExpectEqual);
  // ".ifdef" or ".ifndef", depending on expect_defined
  bool parseDirectiveIfdef(SMLoc DirectiveLoc, bool expect_defined);
  bool parseDirectiveElseIf(SMLoc DirectiveLoc); // ".elseif"
  bool parseDirectiveElse(SMLoc DirectiveLoc); // ".else"
  bool parseDirectiveEndIf(SMLoc DirectiveLoc); // .endif
  bool parseEscapedString(std::string &Data) override;

  const MCExpr *applyModifierToExpr(const MCExpr *E,
                                    MCSymbolRefExpr::VariantKind Variant);

  // Macro-like directives
  MCAsmMacro *parseMacroLikeBody(SMLoc DirectiveLoc);
  void instantiateMacroLikeBody(MCAsmMacro *M, SMLoc DirectiveLoc,
                                raw_svector_ostream &OS);
  bool parseDirectiveRept(SMLoc DirectiveLoc, StringRef Directive);
  bool parseDirectiveIrp(SMLoc DirectiveLoc);  // ".irp"
  bool parseDirectiveIrpc(SMLoc DirectiveLoc); // ".irpc"
  bool parseDirectiveEndr(SMLoc DirectiveLoc); // ".endr"

  // "_emit" or "__emit"
  bool parseDirectiveMSEmit(SMLoc DirectiveLoc, ParseStatementInfo &Info,
                            size_t Len);

  // "align"
  bool parseDirectiveMSAlign(SMLoc DirectiveLoc, ParseStatementInfo &Info);

  // "end"
  bool parseDirectiveEnd(SMLoc DirectiveLoc);

  // ".err" or ".error"
  bool parseDirectiveError(SMLoc DirectiveLoc, bool WithMessage);

  // ".warning"
  bool parseDirectiveWarning(SMLoc DirectiveLoc);

  void initializeDirectiveKindMap();
};
}

namespace llvm {

extern MCAsmParserExtension *createDarwinAsmParser();
extern MCAsmParserExtension *createELFAsmParser();
extern MCAsmParserExtension *createCOFFAsmParser();

}

enum { DEFAULT_ADDRSPACE = 0 };

AsmParser::AsmParser(SourceMgr &SM, MCContext &Ctx, MCStreamer &Out,
                     const MCAsmInfo &MAI)
    : Lexer(MAI), Ctx(Ctx), Out(Out), MAI(MAI), SrcMgr(SM),
      PlatformParser(nullptr), CurBuffer(SM.getMainFileID()),
      MacrosEnabledFlag(true), HadError(false), CppHashLineNumber(0),
      AssemblerDialect(~0U), IsDarwin(false), ParsingInlineAsm(false) {
  // Save the old handler.
  SavedDiagHandler = SrcMgr.getDiagHandler();
  SavedDiagContext = SrcMgr.getDiagContext();
  // Set our own handler which calls the saved handler.
  SrcMgr.setDiagHandler(DiagHandler, this);
  Lexer.setBuffer(SrcMgr.getMemoryBuffer(CurBuffer)->getBuffer());

  // Initialize the platform / file format parser.
  switch (Ctx.getObjectFileInfo()->getObjectFileType()) {
  case MCObjectFileInfo::IsCOFF:
    PlatformParser.reset(createCOFFAsmParser());
    break;
  case MCObjectFileInfo::IsMachO:
    PlatformParser.reset(createDarwinAsmParser());
    IsDarwin = true;
    break;
  case MCObjectFileInfo::IsELF:
    PlatformParser.reset(createELFAsmParser());
    break;
  }

  PlatformParser->Initialize(*this);
  initializeDirectiveKindMap();

  NumOfMacroInstantiations = 0;
}

AsmParser::~AsmParser() {
  assert((HadError || ActiveMacros.empty()) &&
         "Unexpected active macro instantiation!");
}

void AsmParser::printMacroInstantiations() {
  // Print the active macro instantiation stack.
  for (std::vector<MacroInstantiation *>::const_reverse_iterator
           it = ActiveMacros.rbegin(),
           ie = ActiveMacros.rend();
       it != ie; ++it)
    printMessage((*it)->InstantiationLoc, SourceMgr::DK_Note,
                 "while in macro instantiation");
}

void AsmParser::Note(SMLoc L, const Twine &Msg, ArrayRef<SMRange> Ranges) {
  printMessage(L, SourceMgr::DK_Note, Msg, Ranges);
  printMacroInstantiations();
}

bool AsmParser::Warning(SMLoc L, const Twine &Msg, ArrayRef<SMRange> Ranges) {
  if (getTargetParser().getTargetOptions().MCFatalWarnings)
    return Error(L, Msg, Ranges);
  printMessage(L, SourceMgr::DK_Warning, Msg, Ranges);
  printMacroInstantiations();
  return false;
}

bool AsmParser::Error(SMLoc L, const Twine &Msg, ArrayRef<SMRange> Ranges) {
  HadError = true;
  printMessage(L, SourceMgr::DK_Error, Msg, Ranges);
  printMacroInstantiations();
  return true;
}

bool AsmParser::enterIncludeFile(const std::string &Filename) {
  std::string IncludedFile;
  unsigned NewBuf =
      SrcMgr.AddIncludeFile(Filename, Lexer.getLoc(), IncludedFile);
  if (!NewBuf)
    return true;

  CurBuffer = NewBuf;
  Lexer.setBuffer(SrcMgr.getMemoryBuffer(CurBuffer)->getBuffer());
  return false;
}

/// Process the specified .incbin file by searching for it in the include paths
/// then just emitting the byte contents of the file to the streamer. This
/// returns true on failure.
bool AsmParser::processIncbinFile(const std::string &Filename) {
  std::string IncludedFile;
  unsigned NewBuf =
      SrcMgr.AddIncludeFile(Filename, Lexer.getLoc(), IncludedFile);
  if (!NewBuf)
    return true;

  // Pick up the bytes from the file and emit them.
  getStreamer().EmitBytes(SrcMgr.getMemoryBuffer(NewBuf)->getBuffer());
  return false;
}

void AsmParser::jumpToLoc(SMLoc Loc, unsigned InBuffer) {
  CurBuffer = InBuffer ? InBuffer : SrcMgr.FindBufferContainingLoc(Loc);
  Lexer.setBuffer(SrcMgr.getMemoryBuffer(CurBuffer)->getBuffer(),
                  Loc.getPointer());
}

const AsmToken &AsmParser::Lex() {
  const AsmToken *tok = &Lexer.Lex();

  if (tok->is(AsmToken::Eof)) {
    // If this is the end of an included file, pop the parent file off the
    // include stack.
    SMLoc ParentIncludeLoc = SrcMgr.getParentIncludeLoc(CurBuffer);
    if (ParentIncludeLoc != SMLoc()) {
      jumpToLoc(ParentIncludeLoc);
      tok = &Lexer.Lex();
    }
  }

  if (tok->is(AsmToken::Error))
    Error(Lexer.getErrLoc(), Lexer.getErr());

  return *tok;
}

bool AsmParser::Run(bool NoInitialTextSection, bool NoFinalize) {
  // Create the initial section, if requested.
  if (!NoInitialTextSection)
    Out.InitSections(false);

  // Prime the lexer.
  Lex();

  HadError = false;
  AsmCond StartingCondState = TheCondState;

  // If we are generating dwarf for assembly source files save the initial text
  // section and generate a .file directive.
  if (getContext().getGenDwarfForAssembly()) {
    MCSection *Sec = getStreamer().getCurrentSection().first;
    if (!Sec->getBeginSymbol()) {
      MCSymbol *SectionStartSym = getContext().createTempSymbol();
      getStreamer().EmitLabel(SectionStartSym);
      Sec->setBeginSymbol(SectionStartSym);
    }
    bool InsertResult = getContext().addGenDwarfSection(Sec);
    assert(InsertResult && ".text section should not have debug info yet");
    (void)InsertResult;
    getContext().setGenDwarfFileNumber(getStreamer().EmitDwarfFileDirective(
        0, StringRef(), getContext().getMainFileName()));
  }

  // While we have input, parse each statement.
  while (Lexer.isNot(AsmToken::Eof)) {
    ParseStatementInfo Info;
    if (!parseStatement(Info, nullptr))
      continue;

    // We had an error, validate that one was emitted and recover by skipping to
    // the next line.
    assert(HadError && "Parse statement returned an error, but none emitted!");
    eatToEndOfStatement();
  }

  if (TheCondState.TheCond != StartingCondState.TheCond ||
      TheCondState.Ignore != StartingCondState.Ignore)
    return TokError("unmatched .ifs or .elses");

  // Check to see there are no empty DwarfFile slots.
  const auto &LineTables = getContext().getMCDwarfLineTables();
  if (!LineTables.empty()) {
    unsigned Index = 0;
    for (const auto &File : LineTables.begin()->second.getMCDwarfFiles()) {
      if (File.Name.empty() && Index != 0)
        TokError("unassigned file number: " + Twine(Index) +
                 " for .file directives");
      ++Index;
    }
  }

  // Check to see that all assembler local symbols were actually defined.
  // Targets that don't do subsections via symbols may not want this, though,
  // so conservatively exclude them. Only do this if we're finalizing, though,
  // as otherwise we won't necessarilly have seen everything yet.
  if (!NoFinalize && MAI.hasSubsectionsViaSymbols()) {
    const MCContext::SymbolTable &Symbols = getContext().getSymbols();
    for (MCContext::SymbolTable::const_iterator i = Symbols.begin(),
                                                e = Symbols.end();
         i != e; ++i) {
      MCSymbol *Sym = i->getValue();
      // Variable symbols may not be marked as defined, so check those
      // explicitly. If we know it's a variable, we have a definition for
      // the purposes of this check.
      if (Sym->isTemporary() && !Sym->isVariable() && !Sym->isDefined())
        // FIXME: We would really like to refer back to where the symbol was
        // first referenced for a source location. We need to add something
        // to track that. Currently, we just point to the end of the file.
        printMessage(
            getLexer().getLoc(), SourceMgr::DK_Error,
            "assembler local symbol '" + Sym->getName() + "' not defined");
    }
  }

  // Finalize the output stream if there are no errors and if the client wants
  // us to.
  if (!HadError && !NoFinalize)
    Out.Finish();

  return HadError;
}

void AsmParser::checkForValidSection() {
  if (!ParsingInlineAsm && !getStreamer().getCurrentSection().first) {
    TokError("expected section directive before assembly directive");
    Out.InitSections(false);
  }
}

/// \brief Throw away the rest of the line for testing purposes.
void AsmParser::eatToEndOfStatement() {
  while (Lexer.isNot(AsmToken::EndOfStatement) && Lexer.isNot(AsmToken::Eof))
    Lex();

  // Eat EOL.
  if (Lexer.is(AsmToken::EndOfStatement))
    Lex();
}

StringRef AsmParser::parseStringToEndOfStatement() {
  const char *Start = getTok().getLoc().getPointer();

  while (Lexer.isNot(AsmToken::EndOfStatement) && Lexer.isNot(AsmToken::Eof))
    Lex();

  const char *End = getTok().getLoc().getPointer();
  return StringRef(Start, End - Start);
}

StringRef AsmParser::parseStringToComma() {
  const char *Start = getTok().getLoc().getPointer();

  while (Lexer.isNot(AsmToken::EndOfStatement) &&
         Lexer.isNot(AsmToken::Comma) && Lexer.isNot(AsmToken::Eof))
    Lex();

  const char *End = getTok().getLoc().getPointer();
  return StringRef(Start, End - Start);
}

/// \brief Parse a paren expression and return it.
/// NOTE: This assumes the leading '(' has already been consumed.
///
/// parenexpr ::= expr)
///
bool AsmParser::parseParenExpr(const MCExpr *&Res, SMLoc &EndLoc) {
  if (parseExpression(Res))
    return true;
  if (Lexer.isNot(AsmToken::RParen))
    return TokError("expected ')' in parentheses expression");
  EndLoc = Lexer.getTok().getEndLoc();
  Lex();
  return false;
}

/// \brief Parse a bracket expression and return it.
/// NOTE: This assumes the leading '[' has already been consumed.
///
/// bracketexpr ::= expr]
///
bool AsmParser::parseBracketExpr(const MCExpr *&Res, SMLoc &EndLoc) {
  if (parseExpression(Res))
    return true;
  if (Lexer.isNot(AsmToken::RBrac))
    return TokError("expected ']' in brackets expression");
  EndLoc = Lexer.getTok().getEndLoc();
  Lex();
  return false;
}

/// \brief Parse a primary expression and return it.
///  primaryexpr ::= (parenexpr
///  primaryexpr ::= symbol
///  primaryexpr ::= number
///  primaryexpr ::= '.'
///  primaryexpr ::= ~,+,- primaryexpr
bool AsmParser::parsePrimaryExpr(const MCExpr *&Res, SMLoc &EndLoc) {
  SMLoc FirstTokenLoc = getLexer().getLoc();
  AsmToken::TokenKind FirstTokenKind = Lexer.getKind();
  switch (FirstTokenKind) {
  default:
    return TokError("unknown token in expression");
  // If we have an error assume that we've already handled it.
  case AsmToken::Error:
    return true;
  case AsmToken::Exclaim:
    Lex(); // Eat the operator.
    if (parsePrimaryExpr(Res, EndLoc))
      return true;
    Res = MCUnaryExpr::createLNot(Res, getContext());
    return false;
  case AsmToken::Dollar:
  case AsmToken::At:
  case AsmToken::String:
  case AsmToken::Identifier: {
    StringRef Identifier;
    if (parseIdentifier(Identifier)) {
      if (FirstTokenKind == AsmToken::Dollar) {
        if (Lexer.getMAI().getDollarIsPC()) {
          // This is a '$' reference, which references the current PC.  Emit a
          // temporary label to the streamer and refer to it.
          MCSymbol *Sym = Ctx.createTempSymbol();
          Out.EmitLabel(Sym);
          Res = MCSymbolRefExpr::create(Sym, MCSymbolRefExpr::VK_None,
                                        getContext());
          EndLoc = FirstTokenLoc;
          return false;
        }
        return Error(FirstTokenLoc, "invalid token in expression");
      }
    }
    // Parse symbol variant
    std::pair<StringRef, StringRef> Split;
    if (!MAI.useParensForSymbolVariant()) {
      if (FirstTokenKind == AsmToken::String) {
        if (Lexer.is(AsmToken::At)) {
          Lexer.Lex(); // eat @
          SMLoc AtLoc = getLexer().getLoc();
          StringRef VName;
          if (parseIdentifier(VName))
            return Error(AtLoc, "expected symbol variant after '@'");

          Split = std::make_pair(Identifier, VName);
        }
      } else {
        Split = Identifier.split('@');
      }
    } else if (Lexer.is(AsmToken::LParen)) {
      Lexer.Lex(); // eat (
      StringRef VName;
      parseIdentifier(VName);
      if (Lexer.isNot(AsmToken::RParen)) {
          return Error(Lexer.getTok().getLoc(),
                       "unexpected token in variant, expected ')'");
      }
      Lexer.Lex(); // eat )
      Split = std::make_pair(Identifier, VName);
    }

    EndLoc = SMLoc::getFromPointer(Identifier.end());

    // This is a symbol reference.
    StringRef SymbolName = Identifier;
    MCSymbolRefExpr::VariantKind Variant = MCSymbolRefExpr::VK_None;

    // Lookup the symbol variant if used.
    if (Split.second.size()) {
      Variant = MCSymbolRefExpr::getVariantKindForName(Split.second);
      if (Variant != MCSymbolRefExpr::VK_Invalid) {
        SymbolName = Split.first;
      } else if (MAI.doesAllowAtInName() && !MAI.useParensForSymbolVariant()) {
        Variant = MCSymbolRefExpr::VK_None;
      } else {
        return Error(SMLoc::getFromPointer(Split.second.begin()),
                     "invalid variant '" + Split.second + "'");
      }
    }

    MCSymbol *Sym = getContext().getOrCreateSymbol(SymbolName);

    // If this is an absolute variable reference, substitute it now to preserve
    // semantics in the face of reassignment.
    if (Sym->isVariable() && isa<MCConstantExpr>(Sym->getVariableValue())) {
      if (Variant)
        return Error(EndLoc, "unexpected modifier on variable reference");

      Res = Sym->getVariableValue();
      return false;
    }

    // Otherwise create a symbol ref.
    Res = MCSymbolRefExpr::create(Sym, Variant, getContext());
    return false;
  }
  case AsmToken::BigNum:
    return TokError("literal value out of range for directive");
  case AsmToken::Integer: {
    SMLoc Loc = getTok().getLoc();
    int64_t IntVal = getTok().getIntVal();
    Res = MCConstantExpr::create(IntVal, getContext());
    EndLoc = Lexer.getTok().getEndLoc();
    Lex(); // Eat token.
    // Look for 'b' or 'f' following an Integer as a directional label
    if (Lexer.getKind() == AsmToken::Identifier) {
      StringRef IDVal = getTok().getString();
      // Lookup the symbol variant if used.
      std::pair<StringRef, StringRef> Split = IDVal.split('@');
      MCSymbolRefExpr::VariantKind Variant = MCSymbolRefExpr::VK_None;
      if (Split.first.size() != IDVal.size()) {
        Variant = MCSymbolRefExpr::getVariantKindForName(Split.second);
        if (Variant == MCSymbolRefExpr::VK_Invalid)
          return TokError("invalid variant '" + Split.second + "'");
        IDVal = Split.first;
      }
      if (IDVal == "f" || IDVal == "b") {
        MCSymbol *Sym =
            Ctx.getDirectionalLocalSymbol(IntVal, IDVal == "b");
        Res = MCSymbolRefExpr::create(Sym, Variant, getContext());
        if (IDVal == "b" && Sym->isUndefined())
          return Error(Loc, "invalid reference to undefined symbol");
        EndLoc = Lexer.getTok().getEndLoc();
        Lex(); // Eat identifier.
      }
    }
    return false;
  }
  case AsmToken::Real: {
    APFloat RealVal(APFloat::IEEEdouble, getTok().getString());
    uint64_t IntVal = RealVal.bitcastToAPInt().getZExtValue();
    Res = MCConstantExpr::create(IntVal, getContext());
    EndLoc = Lexer.getTok().getEndLoc();
    Lex(); // Eat token.
    return false;
  }
  case AsmToken::Dot: {
    // This is a '.' reference, which references the current PC.  Emit a
    // temporary label to the streamer and refer to it.
    MCSymbol *Sym = Ctx.createTempSymbol();
    Out.EmitLabel(Sym);
    Res = MCSymbolRefExpr::create(Sym, MCSymbolRefExpr::VK_None, getContext());
    EndLoc = Lexer.getTok().getEndLoc();
    Lex(); // Eat identifier.
    return false;
  }
  case AsmToken::LParen:
    Lex(); // Eat the '('.
    return parseParenExpr(Res, EndLoc);
  case AsmToken::LBrac:
    if (!PlatformParser->HasBracketExpressions())
      return TokError("brackets expression not supported on this target");
    Lex(); // Eat the '['.
    return parseBracketExpr(Res, EndLoc);
  case AsmToken::Minus:
    Lex(); // Eat the operator.
    if (parsePrimaryExpr(Res, EndLoc))
      return true;
    Res = MCUnaryExpr::createMinus(Res, getContext());
    return false;
  case AsmToken::Plus:
    Lex(); // Eat the operator.
    if (parsePrimaryExpr(Res, EndLoc))
      return true;
    Res = MCUnaryExpr::createPlus(Res, getContext());
    return false;
  case AsmToken::Tilde:
    Lex(); // Eat the operator.
    if (parsePrimaryExpr(Res, EndLoc))
      return true;
    Res = MCUnaryExpr::createNot(Res, getContext());
    return false;
  }
}

bool AsmParser::parseExpression(const MCExpr *&Res) {
  SMLoc EndLoc;
  return parseExpression(Res, EndLoc);
}

const MCExpr *
AsmParser::applyModifierToExpr(const MCExpr *E,
                               MCSymbolRefExpr::VariantKind Variant) {
  // Ask the target implementation about this expression first.
  const MCExpr *NewE = getTargetParser().applyModifierToExpr(E, Variant, Ctx);
  if (NewE)
    return NewE;
  // Recurse over the given expression, rebuilding it to apply the given variant
  // if there is exactly one symbol.
  switch (E->getKind()) {
  case MCExpr::Target:
  case MCExpr::Constant:
    return nullptr;

  case MCExpr::SymbolRef: {
    const MCSymbolRefExpr *SRE = cast<MCSymbolRefExpr>(E);

    if (SRE->getKind() != MCSymbolRefExpr::VK_None) {
      TokError("invalid variant on expression '" + getTok().getIdentifier() +
               "' (already modified)");
      return E;
    }

    return MCSymbolRefExpr::create(&SRE->getSymbol(), Variant, getContext());
  }

  case MCExpr::Unary: {
    const MCUnaryExpr *UE = cast<MCUnaryExpr>(E);
    const MCExpr *Sub = applyModifierToExpr(UE->getSubExpr(), Variant);
    if (!Sub)
      return nullptr;
    return MCUnaryExpr::create(UE->getOpcode(), Sub, getContext());
  }

  case MCExpr::Binary: {
    const MCBinaryExpr *BE = cast<MCBinaryExpr>(E);
    const MCExpr *LHS = applyModifierToExpr(BE->getLHS(), Variant);
    const MCExpr *RHS = applyModifierToExpr(BE->getRHS(), Variant);

    if (!LHS && !RHS)
      return nullptr;

    if (!LHS)
      LHS = BE->getLHS();
    if (!RHS)
      RHS = BE->getRHS();

    return MCBinaryExpr::create(BE->getOpcode(), LHS, RHS, getContext());
  }
  }

  llvm_unreachable("Invalid expression kind!");
}

/// \brief Parse an expression and return it.
///
///  expr ::= expr &&,|| expr               -> lowest.
///  expr ::= expr |,^,&,! expr
///  expr ::= expr ==,!=,<>,<,<=,>,>= expr
///  expr ::= expr <<,>> expr
///  expr ::= expr +,- expr
///  expr ::= expr *,/,% expr               -> highest.
///  expr ::= primaryexpr
///
bool AsmParser::parseExpression(const MCExpr *&Res, SMLoc &EndLoc) {
  // Parse the expression.
  Res = nullptr;
  if (parsePrimaryExpr(Res, EndLoc) || parseBinOpRHS(1, Res, EndLoc))
    return true;

  // As a special case, we support 'a op b @ modifier' by rewriting the
  // expression to include the modifier. This is inefficient, but in general we
  // expect users to use 'a@modifier op b'.
  if (Lexer.getKind() == AsmToken::At) {
    Lex();

    if (Lexer.isNot(AsmToken::Identifier))
      return TokError("unexpected symbol modifier following '@'");

    MCSymbolRefExpr::VariantKind Variant =
        MCSymbolRefExpr::getVariantKindForName(getTok().getIdentifier());
    if (Variant == MCSymbolRefExpr::VK_Invalid)
      return TokError("invalid variant '" + getTok().getIdentifier() + "'");

    const MCExpr *ModifiedRes = applyModifierToExpr(Res, Variant);
    if (!ModifiedRes) {
      return TokError("invalid modifier '" + getTok().getIdentifier() +
                      "' (no symbols present)");
    }

    Res = ModifiedRes;
    Lex();
  }

  // Try to constant fold it up front, if possible.
  int64_t Value;
  if (Res->evaluateAsAbsolute(Value))
    Res = MCConstantExpr::create(Value, getContext());

  return false;
}

bool AsmParser::parseParenExpression(const MCExpr *&Res, SMLoc &EndLoc) {
  Res = nullptr;
  return parseParenExpr(Res, EndLoc) || parseBinOpRHS(1, Res, EndLoc);
}

bool AsmParser::parseParenExprOfDepth(unsigned ParenDepth, const MCExpr *&Res,
                                      SMLoc &EndLoc) {
  if (parseParenExpr(Res, EndLoc))
    return true;

  for (; ParenDepth > 0; --ParenDepth) {
    if (parseBinOpRHS(1, Res, EndLoc))
      return true;

    // We don't Lex() the last RParen.
    // This is the same behavior as parseParenExpression().
    if (ParenDepth - 1 > 0) {
      if (Lexer.isNot(AsmToken::RParen))
        return TokError("expected ')' in parentheses expression");
      EndLoc = Lexer.getTok().getEndLoc();
      Lex();
    }
  }
  return false;
}

bool AsmParser::parseAbsoluteExpression(int64_t &Res) {
  const MCExpr *Expr;

  SMLoc StartLoc = Lexer.getLoc();
  if (parseExpression(Expr))
    return true;

  if (!Expr->evaluateAsAbsolute(Res))
    return Error(StartLoc, "expected absolute expression");

  return false;
}

unsigned AsmParser::getBinOpPrecedence(AsmToken::TokenKind K,
                                       MCBinaryExpr::Opcode &Kind) {
  switch (K) {
  default:
    return 0; // not a binop.

  // Lowest Precedence: &&, ||
  case AsmToken::AmpAmp:
    Kind = MCBinaryExpr::LAnd;
    return 1;
  case AsmToken::PipePipe:
    Kind = MCBinaryExpr::LOr;
    return 1;

  // Low Precedence: |, &, ^
  //
  // FIXME: gas seems to support '!' as an infix operator?
  case AsmToken::Pipe:
    Kind = MCBinaryExpr::Or;
    return 2;
  case AsmToken::Caret:
    Kind = MCBinaryExpr::Xor;
    return 2;
  case AsmToken::Amp:
    Kind = MCBinaryExpr::And;
    return 2;

  // Low Intermediate Precedence: ==, !=, <>, <, <=, >, >=
  case AsmToken::EqualEqual:
    Kind = MCBinaryExpr::EQ;
    return 3;
  case AsmToken::ExclaimEqual:
  case AsmToken::LessGreater:
    Kind = MCBinaryExpr::NE;
    return 3;
  case AsmToken::Less:
    Kind = MCBinaryExpr::LT;
    return 3;
  case AsmToken::LessEqual:
    Kind = MCBinaryExpr::LTE;
    return 3;
  case AsmToken::Greater:
    Kind = MCBinaryExpr::GT;
    return 3;
  case AsmToken::GreaterEqual:
    Kind = MCBinaryExpr::GTE;
    return 3;

  // Intermediate Precedence: <<, >>
  case AsmToken::LessLess:
    Kind = MCBinaryExpr::Shl;
    return 4;
  case AsmToken::GreaterGreater:
    Kind = MAI.shouldUseLogicalShr() ? MCBinaryExpr::LShr : MCBinaryExpr::AShr;
    return 4;

  // High Intermediate Precedence: +, -
  case AsmToken::Plus:
    Kind = MCBinaryExpr::Add;
    return 5;
  case AsmToken::Minus:
    Kind = MCBinaryExpr::Sub;
    return 5;

  // Highest Precedence: *, /, %
  case AsmToken::Star:
    Kind = MCBinaryExpr::Mul;
    return 6;
  case AsmToken::Slash:
    Kind = MCBinaryExpr::Div;
    return 6;
  case AsmToken::Percent:
    Kind = MCBinaryExpr::Mod;
    return 6;
  }
}

/// \brief Parse all binary operators with precedence >= 'Precedence'.
/// Res contains the LHS of the expression on input.
bool AsmParser::parseBinOpRHS(unsigned Precedence, const MCExpr *&Res,
                              SMLoc &EndLoc) {
  while (1) {
    MCBinaryExpr::Opcode Kind = MCBinaryExpr::Add;
    unsigned TokPrec = getBinOpPrecedence(Lexer.getKind(), Kind);

    // If the next token is lower precedence than we are allowed to eat, return
    // successfully with what we ate already.
    if (TokPrec < Precedence)
      return false;

    Lex();

    // Eat the next primary expression.
    const MCExpr *RHS;
    if (parsePrimaryExpr(RHS, EndLoc))
      return true;

    // If BinOp binds less tightly with RHS than the operator after RHS, let
    // the pending operator take RHS as its LHS.
    MCBinaryExpr::Opcode Dummy;
    unsigned NextTokPrec = getBinOpPrecedence(Lexer.getKind(), Dummy);
    if (TokPrec < NextTokPrec && parseBinOpRHS(TokPrec + 1, RHS, EndLoc))
      return true;

    // Merge LHS and RHS according to operator.
    Res = MCBinaryExpr::create(Kind, Res, RHS, getContext());
  }
}

/// ParseStatement:
///   ::= EndOfStatement
///   ::= Label* Directive ...Operands... EndOfStatement
///   ::= Label* Identifier OperandList* EndOfStatement
bool AsmParser::parseStatement(ParseStatementInfo &Info,
                               MCAsmParserSemaCallback *SI) {
  if (Lexer.is(AsmToken::EndOfStatement)) {
    Out.AddBlankLine();
    Lex();
    return false;
  }

  // Statements always start with an identifier or are a full line comment.
  AsmToken ID = getTok();
  SMLoc IDLoc = ID.getLoc();
  StringRef IDVal;
  int64_t LocalLabelVal = -1;
  // A full line comment is a '#' as the first token.
  if (Lexer.is(AsmToken::Hash))
    return parseCppHashLineFilenameComment(IDLoc);

  // Allow an integer followed by a ':' as a directional local label.
  if (Lexer.is(AsmToken::Integer)) {
    LocalLabelVal = getTok().getIntVal();
    if (LocalLabelVal < 0) {
      if (!TheCondState.Ignore)
        return TokError("unexpected token at start of statement");
      IDVal = "";
    } else {
      IDVal = getTok().getString();
      Lex(); // Consume the integer token to be used as an identifier token.
      if (Lexer.getKind() != AsmToken::Colon) {
        if (!TheCondState.Ignore)
          return TokError("unexpected token at start of statement");
      }
    }
  } else if (Lexer.is(AsmToken::Dot)) {
    // Treat '.' as a valid identifier in this context.
    Lex();
    IDVal = ".";
  } else if (parseIdentifier(IDVal)) {
    if (!TheCondState.Ignore)
      return TokError("unexpected token at start of statement");
    IDVal = "";
  }

  // Handle conditional assembly here before checking for skipping.  We
  // have to do this so that .endif isn't skipped in a ".if 0" block for
  // example.
  StringMap<DirectiveKind>::const_iterator DirKindIt =
      DirectiveKindMap.find(IDVal);
  DirectiveKind DirKind = (DirKindIt == DirectiveKindMap.end())
                              ? DK_NO_DIRECTIVE
                              : DirKindIt->getValue();
  switch (DirKind) {
  default:
    break;
  case DK_IF:
  case DK_IFEQ:
  case DK_IFGE:
  case DK_IFGT:
  case DK_IFLE:
  case DK_IFLT:
  case DK_IFNE:
    return parseDirectiveIf(IDLoc, DirKind);
  case DK_IFB:
    return parseDirectiveIfb(IDLoc, true);
  case DK_IFNB:
    return parseDirectiveIfb(IDLoc, false);
  case DK_IFC:
    return parseDirectiveIfc(IDLoc, true);
  case DK_IFEQS:
    return parseDirectiveIfeqs(IDLoc, true);
  case DK_IFNC:
    return parseDirectiveIfc(IDLoc, false);
  case DK_IFNES:
    return parseDirectiveIfeqs(IDLoc, false);
  case DK_IFDEF:
    return parseDirectiveIfdef(IDLoc, true);
  case DK_IFNDEF:
  case DK_IFNOTDEF:
    return parseDirectiveIfdef(IDLoc, false);
  case DK_ELSEIF:
    return parseDirectiveElseIf(IDLoc);
  case DK_ELSE:
    return parseDirectiveElse(IDLoc);
  case DK_ENDIF:
    return parseDirectiveEndIf(IDLoc);
  }

  // Ignore the statement if in the middle of inactive conditional
  // (e.g. ".if 0").
  if (TheCondState.Ignore) {
    eatToEndOfStatement();
    return false;
  }

  // FIXME: Recurse on local labels?

  // See what kind of statement we have.
  switch (Lexer.getKind()) {
  case AsmToken::Colon: {
    checkForValidSection();

    // identifier ':'   -> Label.
    Lex();

    // Diagnose attempt to use '.' as a label.
    if (IDVal == ".")
      return Error(IDLoc, "invalid use of pseudo-symbol '.' as a label");

    // Diagnose attempt to use a variable as a label.
    //
    // FIXME: Diagnostics. Note the location of the definition as a label.
    // FIXME: This doesn't diagnose assignment to a symbol which has been
    // implicitly marked as external.
    MCSymbol *Sym;
    if (LocalLabelVal == -1) {
      if (ParsingInlineAsm && SI) {
        StringRef RewrittenLabel =
            SI->LookupInlineAsmLabel(IDVal, getSourceManager(), IDLoc, true);
        assert(RewrittenLabel.size() &&
               "We should have an internal name here.");
        Info.AsmRewrites->push_back(AsmRewrite(AOK_Label, IDLoc,
                                               IDVal.size(), RewrittenLabel));
        IDVal = RewrittenLabel;
      }
      Sym = getContext().getOrCreateSymbol(IDVal);
    } else
      Sym = Ctx.createDirectionalLocalSymbol(LocalLabelVal);

    Sym->redefineIfPossible();

    if (!Sym->isUndefined() || Sym->isVariable())
      return Error(IDLoc, "invalid symbol redefinition");

    // Emit the label.
    if (!ParsingInlineAsm)
      Out.EmitLabel(Sym);

    // If we are generating dwarf for assembly source files then gather the
    // info to make a dwarf label entry for this label if needed.
    if (getContext().getGenDwarfForAssembly())
      MCGenDwarfLabelEntry::Make(Sym, &getStreamer(), getSourceManager(),
                                 IDLoc);

    getTargetParser().onLabelParsed(Sym);

    // Consume any end of statement token, if present, to avoid spurious
    // AddBlankLine calls().
    if (Lexer.is(AsmToken::EndOfStatement)) {
      Lex();
      if (Lexer.is(AsmToken::Eof))
        return false;
    }

    return false;
  }

  case AsmToken::Equal:
    // identifier '=' ... -> assignment statement
    Lex();

    return parseAssignment(IDVal, true);

  default: // Normal instruction or directive.
    break;
  }

  // If macros are enabled, check to see if this is a macro instantiation.
  if (areMacrosEnabled())
    if (const MCAsmMacro *M = lookupMacro(IDVal)) {
      return handleMacroEntry(M, IDLoc);
    }

  // Otherwise, we have a normal instruction or directive.

  // Directives start with "."
  if (IDVal[0] == '.' && IDVal != ".") {
    // There are several entities interested in parsing directives:
    //
    // 1. The target-specific assembly parser. Some directives are target
    //    specific or may potentially behave differently on certain targets.
    // 2. Asm parser extensions. For example, platform-specific parsers
    //    (like the ELF parser) register themselves as extensions.
    // 3. The generic directive parser implemented by this class. These are
    //    all the directives that behave in a target and platform independent
    //    manner, or at least have a default behavior that's shared between
    //    all targets and platforms.

    // First query the target-specific parser. It will return 'true' if it
    // isn't interested in this directive.
    if (!getTargetParser().ParseDirective(ID))
      return false;

    // Next, check the extension directive map to see if any extension has
    // registered itself to parse this directive.
    std::pair<MCAsmParserExtension *, DirectiveHandler> Handler =
        ExtensionDirectiveMap.lookup(IDVal);
    if (Handler.first)
      return (*Handler.second)(Handler.first, IDVal, IDLoc);

    // Finally, if no one else is interested in this directive, it must be
    // generic and familiar to this class.
    switch (DirKind) {
    default:
      break;
    case DK_SET:
    case DK_EQU:
      return parseDirectiveSet(IDVal, true);
    case DK_EQUIV:
      return parseDirectiveSet(IDVal, false);
    case DK_ASCII:
      return parseDirectiveAscii(IDVal, false);
    case DK_ASCIZ:
    case DK_STRING:
      return parseDirectiveAscii(IDVal, true);
    case DK_BYTE:
      return parseDirectiveValue(1);
    case DK_SHORT:
    case DK_VALUE:
    case DK_2BYTE:
      return parseDirectiveValue(2);
    case DK_LONG:
    case DK_INT:
    case DK_4BYTE:
      return parseDirectiveValue(4);
    case DK_QUAD:
    case DK_8BYTE:
      return parseDirectiveValue(8);
    case DK_OCTA:
      return parseDirectiveOctaValue();
    case DK_SINGLE:
    case DK_FLOAT:
      return parseDirectiveRealValue(APFloat::IEEEsingle);
    case DK_DOUBLE:
      return parseDirectiveRealValue(APFloat::IEEEdouble);
    case DK_ALIGN: {
      bool IsPow2 = !getContext().getAsmInfo()->getAlignmentIsInBytes();
      return parseDirectiveAlign(IsPow2, /*ExprSize=*/1);
    }
    case DK_ALIGN32: {
      bool IsPow2 = !getContext().getAsmInfo()->getAlignmentIsInBytes();
      return parseDirectiveAlign(IsPow2, /*ExprSize=*/4);
    }
    case DK_BALIGN:
      return parseDirectiveAlign(/*IsPow2=*/false, /*ExprSize=*/1);
    case DK_BALIGNW:
      return parseDirectiveAlign(/*IsPow2=*/false, /*ExprSize=*/2);
    case DK_BALIGNL:
      return parseDirectiveAlign(/*IsPow2=*/false, /*ExprSize=*/4);
    case DK_P2ALIGN:
      return parseDirectiveAlign(/*IsPow2=*/true, /*ExprSize=*/1);
    case DK_P2ALIGNW:
      return parseDirectiveAlign(/*IsPow2=*/true, /*ExprSize=*/2);
    case DK_P2ALIGNL:
      return parseDirectiveAlign(/*IsPow2=*/true, /*ExprSize=*/4);
    case DK_ORG:
      return parseDirectiveOrg();
    case DK_FILL:
      return parseDirectiveFill();
    case DK_ZERO:
      return parseDirectiveZero();
    case DK_EXTERN:
      eatToEndOfStatement(); // .extern is the default, ignore it.
      return false;
    case DK_GLOBL:
    case DK_GLOBAL:
      return parseDirectiveSymbolAttribute(MCSA_Global);
    case DK_LAZY_REFERENCE:
      return parseDirectiveSymbolAttribute(MCSA_LazyReference);
    case DK_NO_DEAD_STRIP:
      return parseDirectiveSymbolAttribute(MCSA_NoDeadStrip);
    case DK_SYMBOL_RESOLVER:
      return parseDirectiveSymbolAttribute(MCSA_SymbolResolver);
    case DK_PRIVATE_EXTERN:
      return parseDirectiveSymbolAttribute(MCSA_PrivateExtern);
    case DK_REFERENCE:
      return parseDirectiveSymbolAttribute(MCSA_Reference);
    case DK_WEAK_DEFINITION:
      return parseDirectiveSymbolAttribute(MCSA_WeakDefinition);
    case DK_WEAK_REFERENCE:
      return parseDirectiveSymbolAttribute(MCSA_WeakReference);
    case DK_WEAK_DEF_CAN_BE_HIDDEN:
      return parseDirectiveSymbolAttribute(MCSA_WeakDefAutoPrivate);
    case DK_COMM:
    case DK_COMMON:
      return parseDirectiveComm(/*IsLocal=*/false);
    case DK_LCOMM:
      return parseDirectiveComm(/*IsLocal=*/true);
    case DK_ABORT:
      return parseDirectiveAbort();
    case DK_INCLUDE:
      return parseDirectiveInclude();
    case DK_INCBIN:
      return parseDirectiveIncbin();
    case DK_CODE16:
    case DK_CODE16GCC:
      return TokError(Twine(IDVal) + " not supported yet");
    case DK_REPT:
      return parseDirectiveRept(IDLoc, IDVal);
    case DK_IRP:
      return parseDirectiveIrp(IDLoc);
    case DK_IRPC:
      return parseDirectiveIrpc(IDLoc);
    case DK_ENDR:
      return parseDirectiveEndr(IDLoc);
    case DK_BUNDLE_ALIGN_MODE:
      return parseDirectiveBundleAlignMode();
    case DK_BUNDLE_LOCK:
      return parseDirectiveBundleLock();
    case DK_BUNDLE_UNLOCK:
      return parseDirectiveBundleUnlock();
    case DK_SLEB128:
      return parseDirectiveLEB128(true);
    case DK_ULEB128:
      return parseDirectiveLEB128(false);
    case DK_SPACE:
    case DK_SKIP:
      return parseDirectiveSpace(IDVal);
    case DK_FILE:
      return parseDirectiveFile(IDLoc);
    case DK_LINE:
      return parseDirectiveLine();
    case DK_LOC:
      return parseDirectiveLoc();
    case DK_STABS:
      return parseDirectiveStabs();
    case DK_CFI_SECTIONS:
      return parseDirectiveCFISections();
    case DK_CFI_STARTPROC:
      return parseDirectiveCFIStartProc();
    case DK_CFI_ENDPROC:
      return parseDirectiveCFIEndProc();
    case DK_CFI_DEF_CFA:
      return parseDirectiveCFIDefCfa(IDLoc);
    case DK_CFI_DEF_CFA_OFFSET:
      return parseDirectiveCFIDefCfaOffset();
    case DK_CFI_ADJUST_CFA_OFFSET:
      return parseDirectiveCFIAdjustCfaOffset();
    case DK_CFI_DEF_CFA_REGISTER:
      return parseDirectiveCFIDefCfaRegister(IDLoc);
    case DK_CFI_OFFSET:
      return parseDirectiveCFIOffset(IDLoc);
    case DK_CFI_REL_OFFSET:
      return parseDirectiveCFIRelOffset(IDLoc);
    case DK_CFI_PERSONALITY:
      return parseDirectiveCFIPersonalityOrLsda(true);
    case DK_CFI_LSDA:
      return parseDirectiveCFIPersonalityOrLsda(false);
    case DK_CFI_REMEMBER_STATE:
      return parseDirectiveCFIRememberState();
    case DK_CFI_RESTORE_STATE:
      return parseDirectiveCFIRestoreState();
    case DK_CFI_SAME_VALUE:
      return parseDirectiveCFISameValue(IDLoc);
    case DK_CFI_RESTORE:
      return parseDirectiveCFIRestore(IDLoc);
    case DK_CFI_ESCAPE:
      return parseDirectiveCFIEscape();
    case DK_CFI_SIGNAL_FRAME:
      return parseDirectiveCFISignalFrame();
    case DK_CFI_UNDEFINED:
      return parseDirectiveCFIUndefined(IDLoc);
    case DK_CFI_REGISTER:
      return parseDirectiveCFIRegister(IDLoc);
    case DK_CFI_WINDOW_SAVE:
      return parseDirectiveCFIWindowSave();
    case DK_MACROS_ON:
    case DK_MACROS_OFF:
      return parseDirectiveMacrosOnOff(IDVal);
    case DK_MACRO:
      return parseDirectiveMacro(IDLoc);
    case DK_EXITM:
      return parseDirectiveExitMacro(IDVal);
    case DK_ENDM:
    case DK_ENDMACRO:
      return parseDirectiveEndMacro(IDVal);
    case DK_PURGEM:
      return parseDirectivePurgeMacro(IDLoc);
    case DK_END:
      return parseDirectiveEnd(IDLoc);
    case DK_ERR:
      return parseDirectiveError(IDLoc, false);
    case DK_ERROR:
      return parseDirectiveError(IDLoc, true);
    case DK_WARNING:
      return parseDirectiveWarning(IDLoc);
    }

    return Error(IDLoc, "unknown directive");
  }

  // __asm _emit or __asm __emit
  if (ParsingInlineAsm && (IDVal == "_emit" || IDVal == "__emit" ||
                           IDVal == "_EMIT" || IDVal == "__EMIT"))
    return parseDirectiveMSEmit(IDLoc, Info, IDVal.size());

  // __asm align
  if (ParsingInlineAsm && (IDVal == "align" || IDVal == "ALIGN"))
    return parseDirectiveMSAlign(IDLoc, Info);

  checkForValidSection();

  // Canonicalize the opcode to lower case.
  std::string OpcodeStr = IDVal.lower();
  ParseInstructionInfo IInfo(Info.AsmRewrites);
  bool HadError = getTargetParser().ParseInstruction(IInfo, OpcodeStr, IDLoc,
                                                     Info.ParsedOperands);
  Info.ParseError = HadError;

  // Dump the parsed representation, if requested.
  if (getShowParsedOperands()) {
    SmallString<256> Str;
    raw_svector_ostream OS(Str);
    OS << "parsed instruction: [";
    for (unsigned i = 0; i != Info.ParsedOperands.size(); ++i) {
      if (i != 0)
        OS << ", ";
      Info.ParsedOperands[i]->print(OS);
    }
    OS << "]";

    printMessage(IDLoc, SourceMgr::DK_Note, OS.str());
  }

  // If we are generating dwarf for the current section then generate a .loc
  // directive for the instruction.
  if (!HadError && getContext().getGenDwarfForAssembly() &&
      getContext().getGenDwarfSectionSyms().count(
          getStreamer().getCurrentSection().first)) {
    unsigned Line;
    if (ActiveMacros.empty())
      Line = SrcMgr.FindLineNumber(IDLoc, CurBuffer);
    else
      Line = SrcMgr.FindLineNumber(ActiveMacros.front()->InstantiationLoc,
                                   ActiveMacros.front()->ExitBuffer);

    // If we previously parsed a cpp hash file line comment then make sure the
    // current Dwarf File is for the CppHashFilename if not then emit the
    // Dwarf File table for it and adjust the line number for the .loc.
    if (CppHashFilename.size()) {
      unsigned FileNumber = getStreamer().EmitDwarfFileDirective(
          0, StringRef(), CppHashFilename);
      getContext().setGenDwarfFileNumber(FileNumber);

      // Since SrcMgr.FindLineNumber() is slow and messes up the SourceMgr's
      // cache with the different Loc from the call above we save the last
      // info we queried here with SrcMgr.FindLineNumber().
      unsigned CppHashLocLineNo;
      if (LastQueryIDLoc == CppHashLoc && LastQueryBuffer == CppHashBuf)
        CppHashLocLineNo = LastQueryLine;
      else {
        CppHashLocLineNo = SrcMgr.FindLineNumber(CppHashLoc, CppHashBuf);
        LastQueryLine = CppHashLocLineNo;
        LastQueryIDLoc = CppHashLoc;
        LastQueryBuffer = CppHashBuf;
      }
      Line = CppHashLineNumber - 1 + (Line - CppHashLocLineNo);
    }

    getStreamer().EmitDwarfLocDirective(
        getContext().getGenDwarfFileNumber(), Line, 0,
        DWARF2_LINE_DEFAULT_IS_STMT ? DWARF2_FLAG_IS_STMT : 0, 0, 0,
        StringRef());
  }

  // If parsing succeeded, match the instruction.
  if (!HadError) {
    uint64_t ErrorInfo;
    getTargetParser().MatchAndEmitInstruction(IDLoc, Info.Opcode,
                                              Info.ParsedOperands, Out,
                                              ErrorInfo, ParsingInlineAsm);
  }

  // Don't skip the rest of the line, the instruction parser is responsible for
  // that.
  return false;
}

/// eatToEndOfLine uses the Lexer to eat the characters to the end of the line
/// since they may not be able to be tokenized to get to the end of line token.
void AsmParser::eatToEndOfLine() {
  if (!Lexer.is(AsmToken::EndOfStatement))
    Lexer.LexUntilEndOfLine();
  // Eat EOL.
  Lex();
}

/// parseCppHashLineFilenameComment as this:
///   ::= # number "filename"
/// or just as a full line comment if it doesn't have a number and a string.
bool AsmParser::parseCppHashLineFilenameComment(const SMLoc &L) {
  Lex(); // Eat the hash token.

  if (getLexer().isNot(AsmToken::Integer)) {
    // Consume the line since in cases it is not a well-formed line directive,
    // as if were simply a full line comment.
    eatToEndOfLine();
    return false;
  }

  int64_t LineNumber = getTok().getIntVal();
  Lex();

  if (getLexer().isNot(AsmToken::String)) {
    eatToEndOfLine();
    return false;
  }

  StringRef Filename = getTok().getString();
  // Get rid of the enclosing quotes.
  Filename = Filename.substr(1, Filename.size() - 2);

  // Save the SMLoc, Filename and LineNumber for later use by diagnostics.
  CppHashLoc = L;
  CppHashFilename = Filename;
  CppHashLineNumber = LineNumber;
  CppHashBuf = CurBuffer;

  // Ignore any trailing characters, they're just comment.
  eatToEndOfLine();
  return false;
}

/// \brief will use the last parsed cpp hash line filename comment
/// for the Filename and LineNo if any in the diagnostic.
void AsmParser::DiagHandler(const SMDiagnostic &Diag, void *Context) {
  const AsmParser *Parser = static_cast<const AsmParser *>(Context);
  raw_ostream &OS = errs();

  const SourceMgr &DiagSrcMgr = *Diag.getSourceMgr();
  const SMLoc &DiagLoc = Diag.getLoc();
  unsigned DiagBuf = DiagSrcMgr.FindBufferContainingLoc(DiagLoc);
  unsigned CppHashBuf =
      Parser->SrcMgr.FindBufferContainingLoc(Parser->CppHashLoc);

  // Like SourceMgr::printMessage() we need to print the include stack if any
  // before printing the message.
  unsigned DiagCurBuffer = DiagSrcMgr.FindBufferContainingLoc(DiagLoc);
  if (!Parser->SavedDiagHandler && DiagCurBuffer &&
      DiagCurBuffer != DiagSrcMgr.getMainFileID()) {
    SMLoc ParentIncludeLoc = DiagSrcMgr.getParentIncludeLoc(DiagCurBuffer);
    DiagSrcMgr.PrintIncludeStack(ParentIncludeLoc, OS);
  }

  // If we have not parsed a cpp hash line filename comment or the source
  // manager changed or buffer changed (like in a nested include) then just
  // print the normal diagnostic using its Filename and LineNo.
  if (!Parser->CppHashLineNumber || &DiagSrcMgr != &Parser->SrcMgr ||
      DiagBuf != CppHashBuf) {
    if (Parser->SavedDiagHandler)
      Parser->SavedDiagHandler(Diag, Parser->SavedDiagContext);
    else
      Diag.print(nullptr, OS);
    return;
  }

  // Use the CppHashFilename and calculate a line number based on the
  // CppHashLoc and CppHashLineNumber relative to this Diag's SMLoc for
  // the diagnostic.
  const std::string &Filename = Parser->CppHashFilename;

  int DiagLocLineNo = DiagSrcMgr.FindLineNumber(DiagLoc, DiagBuf);
  int CppHashLocLineNo =
      Parser->SrcMgr.FindLineNumber(Parser->CppHashLoc, CppHashBuf);
  int LineNo =
      Parser->CppHashLineNumber - 1 + (DiagLocLineNo - CppHashLocLineNo);

  SMDiagnostic NewDiag(*Diag.getSourceMgr(), Diag.getLoc(), Filename, LineNo,
                       Diag.getColumnNo(), Diag.getKind(), Diag.getMessage(),
                       Diag.getLineContents(), Diag.getRanges());

  if (Parser->SavedDiagHandler)
    Parser->SavedDiagHandler(NewDiag, Parser->SavedDiagContext);
  else
    NewDiag.print(nullptr, OS);
}

// FIXME: This is mostly duplicated from the function in AsmLexer.cpp. The
// difference being that that function accepts '@' as part of identifiers and
// we can't do that. AsmLexer.cpp should probably be changed to handle
// '@' as a special case when needed.
static bool isIdentifierChar(char c) {
  return isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '$' ||
         c == '.';
}

bool AsmParser::expandMacro(raw_svector_ostream &OS, StringRef Body,
                            ArrayRef<MCAsmMacroParameter> Parameters,
                            ArrayRef<MCAsmMacroArgument> A,
                            bool EnableAtPseudoVariable, const SMLoc &L) {
  unsigned NParameters = Parameters.size();
  bool HasVararg = NParameters ? Parameters.back().Vararg : false;
  if ((!IsDarwin || NParameters != 0) && NParameters != A.size())
    return Error(L, "Wrong number of arguments");

  // A macro without parameters is handled differently on Darwin:
  // gas accepts no arguments and does no substitutions
  while (!Body.empty()) {
    // Scan for the next substitution.
    std::size_t End = Body.size(), Pos = 0;
    for (; Pos != End; ++Pos) {
      // Check for a substitution or escape.
      if (IsDarwin && !NParameters) {
        // This macro has no parameters, look for $0, $1, etc.
        if (Body[Pos] != '$' || Pos + 1 == End)
          continue;

        char Next = Body[Pos + 1];
        if (Next == '$' || Next == 'n' ||
            isdigit(static_cast<unsigned char>(Next)))
          break;
      } else {
        // This macro has parameters, look for \foo, \bar, etc.
        if (Body[Pos] == '\\' && Pos + 1 != End)
          break;
      }
    }

    // Add the prefix.
    OS << Body.slice(0, Pos);

    // Check if we reached the end.
    if (Pos == End)
      break;

    if (IsDarwin && !NParameters) {
      switch (Body[Pos + 1]) {
      // $$ => $
      case '$':
        OS << '$';
        break;

      // $n => number of arguments
      case 'n':
        OS << A.size();
        break;

      // $[0-9] => argument
      default: {
        // Missing arguments are ignored.
        unsigned Index = Body[Pos + 1] - '0';
        if (Index >= A.size())
          break;

        // Otherwise substitute with the token values, with spaces eliminated.
        for (MCAsmMacroArgument::const_iterator it = A[Index].begin(),
                                                ie = A[Index].end();
             it != ie; ++it)
          OS << it->getString();
        break;
      }
      }
      Pos += 2;
    } else {
      unsigned I = Pos + 1;

      // Check for the \@ pseudo-variable.
      if (EnableAtPseudoVariable && Body[I] == '@' && I + 1 != End)
        ++I;
      else
        while (isIdentifierChar(Body[I]) && I + 1 != End)
          ++I;

      const char *Begin = Body.data() + Pos + 1;
      StringRef Argument(Begin, I - (Pos + 1));
      unsigned Index = 0;

      if (Argument == "@") {
        OS << NumOfMacroInstantiations;
        Pos += 2;
      } else {
        for (; Index < NParameters; ++Index)
          if (Parameters[Index].Name == Argument)
            break;

        if (Index == NParameters) {
          if (Body[Pos + 1] == '(' && Body[Pos + 2] == ')')
            Pos += 3;
          else {
            OS << '\\' << Argument;
            Pos = I;
          }
        } else {
          bool VarargParameter = HasVararg && Index == (NParameters - 1);
          for (MCAsmMacroArgument::const_iterator it = A[Index].begin(),
                                                  ie = A[Index].end();
               it != ie; ++it)
            // We expect no quotes around the string's contents when
            // parsing for varargs.
            if (it->getKind() != AsmToken::String || VarargParameter)
              OS << it->getString();
            else
              OS << it->getStringContents();

          Pos += 1 + Argument.size();
        }
      }
    }
    // Update the scan point.
    Body = Body.substr(Pos);
  }

  return false;
}

MacroInstantiation::MacroInstantiation(SMLoc IL, int EB, SMLoc EL,
                                       size_t CondStackDepth)
    : InstantiationLoc(IL), ExitBuffer(EB), ExitLoc(EL),
      CondStackDepth(CondStackDepth) {}

static bool isOperator(AsmToken::TokenKind kind) {
  switch (kind) {
  default:
    return false;
  case AsmToken::Plus:
  case AsmToken::Minus:
  case AsmToken::Tilde:
  case AsmToken::Slash:
  case AsmToken::Star:
  case AsmToken::Dot:
  case AsmToken::Equal:
  case AsmToken::EqualEqual:
  case AsmToken::Pipe:
  case AsmToken::PipePipe:
  case AsmToken::Caret:
  case AsmToken::Amp:
  case AsmToken::AmpAmp:
  case AsmToken::Exclaim:
  case AsmToken::ExclaimEqual:
  case AsmToken::Percent:
  case AsmToken::Less:
  case AsmToken::LessEqual:
  case AsmToken::LessLess:
  case AsmToken::LessGreater:
  case AsmToken::Greater:
  case AsmToken::GreaterEqual:
  case AsmToken::GreaterGreater:
    return true;
  }
}

namespace {
class AsmLexerSkipSpaceRAII {
public:
  AsmLexerSkipSpaceRAII(AsmLexer &Lexer, bool SkipSpace) : Lexer(Lexer) {
    Lexer.setSkipSpace(SkipSpace);
  }

  ~AsmLexerSkipSpaceRAII() {
    Lexer.setSkipSpace(true);
  }

private:
  AsmLexer &Lexer;
};
}

bool AsmParser::parseMacroArgument(MCAsmMacroArgument &MA, bool Vararg) {

  if (Vararg) {
    if (Lexer.isNot(AsmToken::EndOfStatement)) {
      StringRef Str = parseStringToEndOfStatement();
      MA.emplace_back(AsmToken::String, Str);
    }
    return false;
  }

  unsigned ParenLevel = 0;
  unsigned AddTokens = 0;

  // Darwin doesn't use spaces to delmit arguments.
  AsmLexerSkipSpaceRAII ScopedSkipSpace(Lexer, IsDarwin);

  for (;;) {
    if (Lexer.is(AsmToken::Eof) || Lexer.is(AsmToken::Equal))
      return TokError("unexpected token in macro instantiation");

    if (ParenLevel == 0 && Lexer.is(AsmToken::Comma))
      break;

    if (Lexer.is(AsmToken::Space)) {
      Lex(); // Eat spaces

      // Spaces can delimit parameters, but could also be part an expression.
      // If the token after a space is an operator, add the token and the next
      // one into this argument
      if (!IsDarwin) {
        if (isOperator(Lexer.getKind())) {
          // Check to see whether the token is used as an operator,
          // or part of an identifier
          const char *NextChar = getTok().getEndLoc().getPointer();
          if (*NextChar == ' ')
            AddTokens = 2;
        }

        if (!AddTokens && ParenLevel == 0) {
          break;
        }
      }
    }

    // handleMacroEntry relies on not advancing the lexer here
    // to be able to fill in the remaining default parameter values
    if (Lexer.is(AsmToken::EndOfStatement))
      break;

    // Adjust the current parentheses level.
    if (Lexer.is(AsmToken::LParen))
      ++ParenLevel;
    else if (Lexer.is(AsmToken::RParen) && ParenLevel)
      --ParenLevel;

    // Append the token to the current argument list.
    MA.push_back(getTok());
    if (AddTokens)
      AddTokens--;
    Lex();
  }

  if (ParenLevel != 0)
    return TokError("unbalanced parentheses in macro argument");
  return false;
}

// Parse the macro instantiation arguments.
bool AsmParser::parseMacroArguments(const MCAsmMacro *M,
                                    MCAsmMacroArguments &A) {
  const unsigned NParameters = M ? M->Parameters.size() : 0;
  bool NamedParametersFound = false;
  SmallVector<SMLoc, 4> FALocs;

  A.resize(NParameters);
  FALocs.resize(NParameters);

  // Parse two kinds of macro invocations:
  // - macros defined without any parameters accept an arbitrary number of them
  // - macros defined with parameters accept at most that many of them
  bool HasVararg = NParameters ? M->Parameters.back().Vararg : false;
  for (unsigned Parameter = 0; !NParameters || Parameter < NParameters;
       ++Parameter) {
    SMLoc IDLoc = Lexer.getLoc();
    MCAsmMacroParameter FA;

    if (Lexer.is(AsmToken::Identifier) && Lexer.peekTok().is(AsmToken::Equal)) {
      if (parseIdentifier(FA.Name)) {
        Error(IDLoc, "invalid argument identifier for formal argument");
        eatToEndOfStatement();
        return true;
      }

      if (!Lexer.is(AsmToken::Equal)) {
        TokError("expected '=' after formal parameter identifier");
        eatToEndOfStatement();
        return true;
      }
      Lex();

      NamedParametersFound = true;
    }

    if (NamedParametersFound && FA.Name.empty()) {
      Error(IDLoc, "cannot mix positional and keyword arguments");
      eatToEndOfStatement();
      return true;
    }

    bool Vararg = HasVararg && Parameter == (NParameters - 1);
    if (parseMacroArgument(FA.Value, Vararg))
      return true;

    unsigned PI = Parameter;
    if (!FA.Name.empty()) {
      unsigned FAI = 0;
      for (FAI = 0; FAI < NParameters; ++FAI)
        if (M->Parameters[FAI].Name == FA.Name)
          break;

      if (FAI >= NParameters) {
    assert(M && "expected macro to be defined");
        Error(IDLoc,
              "parameter named '" + FA.Name + "' does not exist for macro '" +
              M->Name + "'");
        return true;
      }
      PI = FAI;
    }

    if (!FA.Value.empty()) {
      if (A.size() <= PI)
        A.resize(PI + 1);
      A[PI] = FA.Value;

      if (FALocs.size() <= PI)
        FALocs.resize(PI + 1);

      FALocs[PI] = Lexer.getLoc();
    }

    // At the end of the statement, fill in remaining arguments that have
    // default values. If there aren't any, then the next argument is
    // required but missing
    if (Lexer.is(AsmToken::EndOfStatement)) {
      bool Failure = false;
      for (unsigned FAI = 0; FAI < NParameters; ++FAI) {
        if (A[FAI].empty()) {
          if (M->Parameters[FAI].Required) {
            Error(FALocs[FAI].isValid() ? FALocs[FAI] : Lexer.getLoc(),
                  "missing value for required parameter "
                  "'" + M->Parameters[FAI].Name + "' in macro '" + M->Name + "'");
            Failure = true;
          }

          if (!M->Parameters[FAI].Value.empty())
            A[FAI] = M->Parameters[FAI].Value;
        }
      }
      return Failure;
    }

    if (Lexer.is(AsmToken::Comma))
      Lex();
  }

  return TokError("too many positional arguments");
}

const MCAsmMacro *AsmParser::lookupMacro(StringRef Name) {
  StringMap<MCAsmMacro>::iterator I = MacroMap.find(Name);
  return (I == MacroMap.end()) ? nullptr : &I->getValue();
}

void AsmParser::defineMacro(StringRef Name, MCAsmMacro Macro) {
  MacroMap.insert(std::make_pair(Name, std::move(Macro)));
}

void AsmParser::undefineMacro(StringRef Name) { MacroMap.erase(Name); }

bool AsmParser::handleMacroEntry(const MCAsmMacro *M, SMLoc NameLoc) {
  // Arbitrarily limit macro nesting depth, to match 'as'. We can eliminate
  // this, although we should protect against infinite loops.
  if (ActiveMacros.size() == 20)
    return TokError("macros cannot be nested more than 20 levels deep");

  MCAsmMacroArguments A;
  if (parseMacroArguments(M, A))
    return true;

  // Macro instantiation is lexical, unfortunately. We construct a new buffer
  // to hold the macro body with substitutions.
  SmallString<256> Buf;
  StringRef Body = M->Body;
  raw_svector_ostream OS(Buf);

  if (expandMacro(OS, Body, M->Parameters, A, true, getTok().getLoc()))
    return true;

  // We include the .endmacro in the buffer as our cue to exit the macro
  // instantiation.
  OS << ".endmacro\n";

  std::unique_ptr<MemoryBuffer> Instantiation =
      MemoryBuffer::getMemBufferCopy(OS.str(), "<instantiation>");

  // Create the macro instantiation object and add to the current macro
  // instantiation stack.
  MacroInstantiation *MI = new MacroInstantiation(
      NameLoc, CurBuffer, getTok().getLoc(), TheCondStack.size());
  ActiveMacros.push_back(MI);

  ++NumOfMacroInstantiations;

  // Jump to the macro instantiation and prime the lexer.
  CurBuffer = SrcMgr.AddNewSourceBuffer(std::move(Instantiation), SMLoc());
  Lexer.setBuffer(SrcMgr.getMemoryBuffer(CurBuffer)->getBuffer());
  Lex();

  return false;
}

void AsmParser::handleMacroExit() {
  // Jump to the EndOfStatement we should return to, and consume it.
  jumpToLoc(ActiveMacros.back()->ExitLoc, ActiveMacros.back()->ExitBuffer);
  Lex();

  // Pop the instantiation entry.
  delete ActiveMacros.back();
  ActiveMacros.pop_back();
}

bool AsmParser::parseAssignment(StringRef Name, bool allow_redef,
                                bool NoDeadStrip) {
  MCSymbol *Sym;
  const MCExpr *Value;
  if (MCParserUtils::parseAssignmentExpression(Name, allow_redef, *this, Sym,
                                               Value))
    return true;

  if (!Sym) {
    // In the case where we parse an expression starting with a '.', we will
    // not generate an error, nor will we create a symbol.  In this case we
    // should just return out.
    return false;
  }

  // Do the assignment.
  Out.EmitAssignment(Sym, Value);
  if (NoDeadStrip)
    Out.EmitSymbolAttribute(Sym, MCSA_NoDeadStrip);

  return false;
}

/// parseIdentifier:
///   ::= identifier
///   ::= string
bool AsmParser::parseIdentifier(StringRef &Res) {
  // The assembler has relaxed rules for accepting identifiers, in particular we
  // allow things like '.globl $foo' and '.def @feat.00', which would normally be
  // separate tokens. At this level, we have already lexed so we cannot (currently)
  // handle this as a context dependent token, instead we detect adjacent tokens
  // and return the combined identifier.
  if (Lexer.is(AsmToken::Dollar) || Lexer.is(AsmToken::At)) {
    SMLoc PrefixLoc = getLexer().getLoc();

    // Consume the prefix character, and check for a following identifier.
    Lex();
    if (Lexer.isNot(AsmToken::Identifier))
      return true;

    // We have a '$' or '@' followed by an identifier, make sure they are adjacent.
    if (PrefixLoc.getPointer() + 1 != getTok().getLoc().getPointer())
      return true;

    // Construct the joined identifier and consume the token.
    Res =
        StringRef(PrefixLoc.getPointer(), getTok().getIdentifier().size() + 1);
    Lex();
    return false;
  }

  if (Lexer.isNot(AsmToken::Identifier) && Lexer.isNot(AsmToken::String))
    return true;

  Res = getTok().getIdentifier();

  Lex(); // Consume the identifier token.

  return false;
}

/// parseDirectiveSet:
///   ::= .equ identifier ',' expression
///   ::= .equiv identifier ',' expression
///   ::= .set identifier ',' expression
bool AsmParser::parseDirectiveSet(StringRef IDVal, bool allow_redef) {
  StringRef Name;

  if (parseIdentifier(Name))
    return TokError("expected identifier after '" + Twine(IDVal) + "'");

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in '" + Twine(IDVal) + "'");
  Lex();

  return parseAssignment(Name, allow_redef, true);
}

bool AsmParser::parseEscapedString(std::string &Data) {
  assert(getLexer().is(AsmToken::String) && "Unexpected current token!");

  Data = "";
  StringRef Str = getTok().getStringContents();
  for (unsigned i = 0, e = Str.size(); i != e; ++i) {
    if (Str[i] != '\\') {
      Data += Str[i];
      continue;
    }

    // Recognize escaped characters. Note that this escape semantics currently
    // loosely follows Darwin 'as'. Notably, it doesn't support hex escapes.
    ++i;
    if (i == e)
      return TokError("unexpected backslash at end of string");

    // Recognize octal sequences.
    if ((unsigned)(Str[i] - '0') <= 7) {
      // Consume up to three octal characters.
      unsigned Value = Str[i] - '0';

      if (i + 1 != e && ((unsigned)(Str[i + 1] - '0')) <= 7) {
        ++i;
        Value = Value * 8 + (Str[i] - '0');

        if (i + 1 != e && ((unsigned)(Str[i + 1] - '0')) <= 7) {
          ++i;
          Value = Value * 8 + (Str[i] - '0');
        }
      }

      if (Value > 255)
        return TokError("invalid octal escape sequence (out of range)");

      Data += (unsigned char)Value;
      continue;
    }

    // Otherwise recognize individual escapes.
    switch (Str[i]) {
    default:
      // Just reject invalid escape sequences for now.
      return TokError("invalid escape sequence (unrecognized character)");

    case 'b': Data += '\b'; break;
    case 'f': Data += '\f'; break;
    case 'n': Data += '\n'; break;
    case 'r': Data += '\r'; break;
    case 't': Data += '\t'; break;
    case '"': Data += '"'; break;
    case '\\': Data += '\\'; break;
    }
  }

  return false;
}

/// parseDirectiveAscii:
///   ::= ( .ascii | .asciz | .string ) [ "string" ( , "string" )* ]
bool AsmParser::parseDirectiveAscii(StringRef IDVal, bool ZeroTerminated) {
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    checkForValidSection();

    for (;;) {
      if (getLexer().isNot(AsmToken::String))
        return TokError("expected string in '" + Twine(IDVal) + "' directive");

      std::string Data;
      if (parseEscapedString(Data))
        return true;

      getStreamer().EmitBytes(Data);
      if (ZeroTerminated)
        getStreamer().EmitBytes(StringRef("\0", 1));

      Lex();

      if (getLexer().is(AsmToken::EndOfStatement))
        break;

      if (getLexer().isNot(AsmToken::Comma))
        return TokError("unexpected token in '" + Twine(IDVal) + "' directive");
      Lex();
    }
  }

  Lex();
  return false;
}

/// parseDirectiveValue
///  ::= (.byte | .short | ... ) [ expression (, expression)* ]
bool AsmParser::parseDirectiveValue(unsigned Size) {
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    checkForValidSection();

    for (;;) {
      const MCExpr *Value;
      SMLoc ExprLoc = getLexer().getLoc();
      if (parseExpression(Value))
        return true;

      // Special case constant expressions to match code generator.
      if (const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(Value)) {
        assert(Size <= 8 && "Invalid size");
        uint64_t IntValue = MCE->getValue();
        if (!isUIntN(8 * Size, IntValue) && !isIntN(8 * Size, IntValue))
          return Error(ExprLoc, "literal value out of range for directive");
        getStreamer().EmitIntValue(IntValue, Size);
      } else
        getStreamer().EmitValue(Value, Size, ExprLoc);

      if (getLexer().is(AsmToken::EndOfStatement))
        break;

      // FIXME: Improve diagnostic.
      if (getLexer().isNot(AsmToken::Comma))
        return TokError("unexpected token in directive");
      Lex();
    }
  }

  Lex();
  return false;
}

/// ParseDirectiveOctaValue
///  ::= .octa [ hexconstant (, hexconstant)* ]
bool AsmParser::parseDirectiveOctaValue() {
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    checkForValidSection();

    for (;;) {
      if (Lexer.getKind() == AsmToken::Error)
        return true;
      if (Lexer.getKind() != AsmToken::Integer &&
          Lexer.getKind() != AsmToken::BigNum)
        return TokError("unknown token in expression");

      SMLoc ExprLoc = getLexer().getLoc();
      APInt IntValue = getTok().getAPIntVal();
      Lex();

      uint64_t hi, lo;
      if (IntValue.isIntN(64)) {
        hi = 0;
        lo = IntValue.getZExtValue();
      } else if (IntValue.isIntN(128)) {
        // It might actually have more than 128 bits, but the top ones are zero.
        hi = IntValue.getHiBits(IntValue.getBitWidth() - 64).getZExtValue();
        lo = IntValue.getLoBits(64).getZExtValue();
      } else
        return Error(ExprLoc, "literal value out of range for directive");

      if (MAI.isLittleEndian()) {
        getStreamer().EmitIntValue(lo, 8);
        getStreamer().EmitIntValue(hi, 8);
      } else {
        getStreamer().EmitIntValue(hi, 8);
        getStreamer().EmitIntValue(lo, 8);
      }

      if (getLexer().is(AsmToken::EndOfStatement))
        break;

      // FIXME: Improve diagnostic.
      if (getLexer().isNot(AsmToken::Comma))
        return TokError("unexpected token in directive");
      Lex();
    }
  }

  Lex();
  return false;
}

/// parseDirectiveRealValue
///  ::= (.single | .double) [ expression (, expression)* ]
bool AsmParser::parseDirectiveRealValue(const fltSemantics &Semantics) {
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    checkForValidSection();

    for (;;) {
      // We don't truly support arithmetic on floating point expressions, so we
      // have to manually parse unary prefixes.
      bool IsNeg = false;
      if (getLexer().is(AsmToken::Minus)) {
        Lex();
        IsNeg = true;
      } else if (getLexer().is(AsmToken::Plus))
        Lex();

      if (getLexer().isNot(AsmToken::Integer) &&
          getLexer().isNot(AsmToken::Real) &&
          getLexer().isNot(AsmToken::Identifier))
        return TokError("unexpected token in directive");

      // Convert to an APFloat.
      APFloat Value(Semantics);
      StringRef IDVal = getTok().getString();
      if (getLexer().is(AsmToken::Identifier)) {
        if (!IDVal.compare_lower("infinity") || !IDVal.compare_lower("inf"))
          Value = APFloat::getInf(Semantics);
        else if (!IDVal.compare_lower("nan"))
          Value = APFloat::getNaN(Semantics, false, ~0);
        else
          return TokError("invalid floating point literal");
      } else if (Value.convertFromString(IDVal, APFloat::rmNearestTiesToEven) ==
                 APFloat::opInvalidOp)
        return TokError("invalid floating point literal");
      if (IsNeg)
        Value.changeSign();

      // Consume the numeric token.
      Lex();

      // Emit the value as an integer.
      APInt AsInt = Value.bitcastToAPInt();
      getStreamer().EmitIntValue(AsInt.getLimitedValue(),
                                 AsInt.getBitWidth() / 8);

      if (getLexer().is(AsmToken::EndOfStatement))
        break;

      if (getLexer().isNot(AsmToken::Comma))
        return TokError("unexpected token in directive");
      Lex();
    }
  }

  Lex();
  return false;
}

/// parseDirectiveZero
///  ::= .zero expression
bool AsmParser::parseDirectiveZero() {
  checkForValidSection();

  int64_t NumBytes;
  if (parseAbsoluteExpression(NumBytes))
    return true;

  int64_t Val = 0;
  if (getLexer().is(AsmToken::Comma)) {
    Lex();
    if (parseAbsoluteExpression(Val))
      return true;
  }

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.zero' directive");

  Lex();

  getStreamer().EmitFill(NumBytes, Val);

  return false;
}

/// parseDirectiveFill
///  ::= .fill expression [ , expression [ , expression ] ]
bool AsmParser::parseDirectiveFill() {
  checkForValidSection();

  SMLoc RepeatLoc = getLexer().getLoc();
  int64_t NumValues;
  if (parseAbsoluteExpression(NumValues))
    return true;

  if (NumValues < 0) {
    Warning(RepeatLoc,
            "'.fill' directive with negative repeat count has no effect");
    NumValues = 0;
  }

  int64_t FillSize = 1;
  int64_t FillExpr = 0;

  SMLoc SizeLoc, ExprLoc;
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    if (getLexer().isNot(AsmToken::Comma))
      return TokError("unexpected token in '.fill' directive");
    Lex();

    SizeLoc = getLexer().getLoc();
    if (parseAbsoluteExpression(FillSize))
      return true;

    if (getLexer().isNot(AsmToken::EndOfStatement)) {
      if (getLexer().isNot(AsmToken::Comma))
        return TokError("unexpected token in '.fill' directive");
      Lex();

      ExprLoc = getLexer().getLoc();
      if (parseAbsoluteExpression(FillExpr))
        return true;

      if (getLexer().isNot(AsmToken::EndOfStatement))
        return TokError("unexpected token in '.fill' directive");

      Lex();
    }
  }

  if (FillSize < 0) {
    Warning(SizeLoc, "'.fill' directive with negative size has no effect");
    NumValues = 0;
  }
  if (FillSize > 8) {
    Warning(SizeLoc, "'.fill' directive with size greater than 8 has been truncated to 8");
    FillSize = 8;
  }

  if (!isUInt<32>(FillExpr) && FillSize > 4)
    Warning(ExprLoc, "'.fill' directive pattern has been truncated to 32-bits");

  if (NumValues > 0) {
    int64_t NonZeroFillSize = FillSize > 4 ? 4 : FillSize;
    FillExpr &= ~0ULL >> (64 - NonZeroFillSize * 8);
    for (uint64_t i = 0, e = NumValues; i != e; ++i) {
      getStreamer().EmitIntValue(FillExpr, NonZeroFillSize);
      if (NonZeroFillSize < FillSize)
        getStreamer().EmitIntValue(0, FillSize - NonZeroFillSize);
    }
  }

  return false;
}

/// parseDirectiveOrg
///  ::= .org expression [ , expression ]
bool AsmParser::parseDirectiveOrg() {
  checkForValidSection();

  const MCExpr *Offset;
  SMLoc Loc = getTok().getLoc();
  if (parseExpression(Offset))
    return true;

  // Parse optional fill expression.
  int64_t FillExpr = 0;
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    if (getLexer().isNot(AsmToken::Comma))
      return TokError("unexpected token in '.org' directive");
    Lex();

    if (parseAbsoluteExpression(FillExpr))
      return true;

    if (getLexer().isNot(AsmToken::EndOfStatement))
      return TokError("unexpected token in '.org' directive");
  }

  Lex();

  // Only limited forms of relocatable expressions are accepted here, it
  // has to be relative to the current section. The streamer will return
  // 'true' if the expression wasn't evaluatable.
  if (getStreamer().EmitValueToOffset(Offset, FillExpr))
    return Error(Loc, "expected assembly-time absolute expression");

  return false;
}

/// parseDirectiveAlign
///  ::= {.align, ...} expression [ , expression [ , expression ]]
bool AsmParser::parseDirectiveAlign(bool IsPow2, unsigned ValueSize) {
  checkForValidSection();

  SMLoc AlignmentLoc = getLexer().getLoc();
  int64_t Alignment;
  if (parseAbsoluteExpression(Alignment))
    return true;

  SMLoc MaxBytesLoc;
  bool HasFillExpr = false;
  int64_t FillExpr = 0;
  int64_t MaxBytesToFill = 0;
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    if (getLexer().isNot(AsmToken::Comma))
      return TokError("unexpected token in directive");
    Lex();

    // The fill expression can be omitted while specifying a maximum number of
    // alignment bytes, e.g:
    //  .align 3,,4
    if (getLexer().isNot(AsmToken::Comma)) {
      HasFillExpr = true;
      if (parseAbsoluteExpression(FillExpr))
        return true;
    }

    if (getLexer().isNot(AsmToken::EndOfStatement)) {
      if (getLexer().isNot(AsmToken::Comma))
        return TokError("unexpected token in directive");
      Lex();

      MaxBytesLoc = getLexer().getLoc();
      if (parseAbsoluteExpression(MaxBytesToFill))
        return true;

      if (getLexer().isNot(AsmToken::EndOfStatement))
        return TokError("unexpected token in directive");
    }
  }

  Lex();

  if (!HasFillExpr)
    FillExpr = 0;

  // Compute alignment in bytes.
  if (IsPow2) {
    // FIXME: Diagnose overflow.
    if (Alignment >= 32) {
      Error(AlignmentLoc, "invalid alignment value");
      Alignment = 31;
    }

    Alignment = 1ULL << Alignment;
  } else {
    // Reject alignments that aren't a power of two, for gas compatibility.
    if (!isPowerOf2_64(Alignment))
      Error(AlignmentLoc, "alignment must be a power of 2");
  }

  // Diagnose non-sensical max bytes to align.
  if (MaxBytesLoc.isValid()) {
    if (MaxBytesToFill < 1) {
      Error(MaxBytesLoc, "alignment directive can never be satisfied in this "
                         "many bytes, ignoring maximum bytes expression");
      MaxBytesToFill = 0;
    }

    if (MaxBytesToFill >= Alignment) {
      Warning(MaxBytesLoc, "maximum bytes expression exceeds alignment and "
                           "has no effect");
      MaxBytesToFill = 0;
    }
  }

  // Check whether we should use optimal code alignment for this .align
  // directive.
  const MCSection *Section = getStreamer().getCurrentSection().first;
  assert(Section && "must have section to emit alignment");
  bool UseCodeAlign = Section->UseCodeAlign();
  if ((!HasFillExpr || Lexer.getMAI().getTextAlignFillValue() == FillExpr) &&
      ValueSize == 1 && UseCodeAlign) {
    getStreamer().EmitCodeAlignment(Alignment, MaxBytesToFill);
  } else {
    // FIXME: Target specific behavior about how the "extra" bytes are filled.
    getStreamer().EmitValueToAlignment(Alignment, FillExpr, ValueSize,
                                       MaxBytesToFill);
  }

  return false;
}

/// parseDirectiveFile
/// ::= .file [number] filename
/// ::= .file number directory filename
bool AsmParser::parseDirectiveFile(SMLoc DirectiveLoc) {
  // FIXME: I'm not sure what this is.
  int64_t FileNumber = -1;
  SMLoc FileNumberLoc = getLexer().getLoc();
  if (getLexer().is(AsmToken::Integer)) {
    FileNumber = getTok().getIntVal();
    Lex();

    if (FileNumber < 1)
      return TokError("file number less than one");
  }

  if (getLexer().isNot(AsmToken::String))
    return TokError("unexpected token in '.file' directive");

  // Usually the directory and filename together, otherwise just the directory.
  // Allow the strings to have escaped octal character sequence.
  std::string Path = getTok().getString();
  if (parseEscapedString(Path))
    return true;
  Lex();

  StringRef Directory;
  StringRef Filename;
  std::string FilenameData;
  if (getLexer().is(AsmToken::String)) {
    if (FileNumber == -1)
      return TokError("explicit path specified, but no file number");
    if (parseEscapedString(FilenameData))
      return true;
    Filename = FilenameData;
    Directory = Path;
    Lex();
  } else {
    Filename = Path;
  }

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.file' directive");

  if (FileNumber == -1)
    getStreamer().EmitFileDirective(Filename);
  else {
    if (getContext().getGenDwarfForAssembly())
      Error(DirectiveLoc,
            "input can't have .file dwarf directives when -g is "
            "used to generate dwarf debug info for assembly code");

    if (getStreamer().EmitDwarfFileDirective(FileNumber, Directory, Filename) ==
        0)
      Error(FileNumberLoc, "file number already allocated");
  }

  return false;
}

/// parseDirectiveLine
/// ::= .line [number]
bool AsmParser::parseDirectiveLine() {
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    if (getLexer().isNot(AsmToken::Integer))
      return TokError("unexpected token in '.line' directive");

    int64_t LineNumber = getTok().getIntVal();
    (void)LineNumber;
    Lex();

    // FIXME: Do something with the .line.
  }

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.line' directive");

  return false;
}

/// parseDirectiveLoc
/// ::= .loc FileNumber [LineNumber] [ColumnPos] [basic_block] [prologue_end]
///                                [epilogue_begin] [is_stmt VALUE] [isa VALUE]
/// The first number is a file number, must have been previously assigned with
/// a .file directive, the second number is the line number and optionally the
/// third number is a column position (zero if not specified).  The remaining
/// optional items are .loc sub-directives.
bool AsmParser::parseDirectiveLoc() {
  if (getLexer().isNot(AsmToken::Integer))
    return TokError("unexpected token in '.loc' directive");
  int64_t FileNumber = getTok().getIntVal();
  if (FileNumber < 1)
    return TokError("file number less than one in '.loc' directive");
  if (!getContext().isValidDwarfFileNumber(FileNumber))
    return TokError("unassigned file number in '.loc' directive");
  Lex();

  int64_t LineNumber = 0;
  if (getLexer().is(AsmToken::Integer)) {
    LineNumber = getTok().getIntVal();
    if (LineNumber < 0)
      return TokError("line number less than zero in '.loc' directive");
    Lex();
  }

  int64_t ColumnPos = 0;
  if (getLexer().is(AsmToken::Integer)) {
    ColumnPos = getTok().getIntVal();
    if (ColumnPos < 0)
      return TokError("column position less than zero in '.loc' directive");
    Lex();
  }

  unsigned Flags = DWARF2_LINE_DEFAULT_IS_STMT ? DWARF2_FLAG_IS_STMT : 0;
  unsigned Isa = 0;
  int64_t Discriminator = 0;
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    for (;;) {
      if (getLexer().is(AsmToken::EndOfStatement))
        break;

      StringRef Name;
      SMLoc Loc = getTok().getLoc();
      if (parseIdentifier(Name))
        return TokError("unexpected token in '.loc' directive");

      if (Name == "basic_block")
        Flags |= DWARF2_FLAG_BASIC_BLOCK;
      else if (Name == "prologue_end")
        Flags |= DWARF2_FLAG_PROLOGUE_END;
      else if (Name == "epilogue_begin")
        Flags |= DWARF2_FLAG_EPILOGUE_BEGIN;
      else if (Name == "is_stmt") {
        Loc = getTok().getLoc();
        const MCExpr *Value;
        if (parseExpression(Value))
          return true;
        // The expression must be the constant 0 or 1.
        if (const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(Value)) {
          int Value = MCE->getValue();
          if (Value == 0)
            Flags &= ~DWARF2_FLAG_IS_STMT;
          else if (Value == 1)
            Flags |= DWARF2_FLAG_IS_STMT;
          else
            return Error(Loc, "is_stmt value not 0 or 1");
        } else {
          return Error(Loc, "is_stmt value not the constant value of 0 or 1");
        }
      } else if (Name == "isa") {
        Loc = getTok().getLoc();
        const MCExpr *Value;
        if (parseExpression(Value))
          return true;
        // The expression must be a constant greater or equal to 0.
        if (const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(Value)) {
          int Value = MCE->getValue();
          if (Value < 0)
            return Error(Loc, "isa number less than zero");
          Isa = Value;
        } else {
          return Error(Loc, "isa number not a constant value");
        }
      } else if (Name == "discriminator") {
        if (parseAbsoluteExpression(Discriminator))
          return true;
      } else {
        return Error(Loc, "unknown sub-directive in '.loc' directive");
      }

      if (getLexer().is(AsmToken::EndOfStatement))
        break;
    }
  }

  getStreamer().EmitDwarfLocDirective(FileNumber, LineNumber, ColumnPos, Flags,
                                      Isa, Discriminator, StringRef());

  return false;
}

/// parseDirectiveStabs
/// ::= .stabs string, number, number, number
bool AsmParser::parseDirectiveStabs() {
  return TokError("unsupported directive '.stabs'");
}

/// parseDirectiveCFISections
/// ::= .cfi_sections section [, section]
bool AsmParser::parseDirectiveCFISections() {
  StringRef Name;
  bool EH = false;
  bool Debug = false;

  if (parseIdentifier(Name))
    return TokError("Expected an identifier");

  if (Name == ".eh_frame")
    EH = true;
  else if (Name == ".debug_frame")
    Debug = true;

  if (getLexer().is(AsmToken::Comma)) {
    Lex();

    if (parseIdentifier(Name))
      return TokError("Expected an identifier");

    if (Name == ".eh_frame")
      EH = true;
    else if (Name == ".debug_frame")
      Debug = true;
  }

  getStreamer().EmitCFISections(EH, Debug);
  return false;
}

/// parseDirectiveCFIStartProc
/// ::= .cfi_startproc [simple]
bool AsmParser::parseDirectiveCFIStartProc() {
  StringRef Simple;
  if (getLexer().isNot(AsmToken::EndOfStatement))
    if (parseIdentifier(Simple) || Simple != "simple")
      return TokError("unexpected token in .cfi_startproc directive");

  getStreamer().EmitCFIStartProc(!Simple.empty());
  return false;
}

/// parseDirectiveCFIEndProc
/// ::= .cfi_endproc
bool AsmParser::parseDirectiveCFIEndProc() {
  getStreamer().EmitCFIEndProc();
  return false;
}

/// \brief parse register name or number.
bool AsmParser::parseRegisterOrRegisterNumber(int64_t &Register,
                                              SMLoc DirectiveLoc) {
  unsigned RegNo;

  if (getLexer().isNot(AsmToken::Integer)) {
    if (getTargetParser().ParseRegister(RegNo, DirectiveLoc, DirectiveLoc))
      return true;
    Register = getContext().getRegisterInfo()->getDwarfRegNum(RegNo, true);
  } else
    return parseAbsoluteExpression(Register);

  return false;
}

/// parseDirectiveCFIDefCfa
/// ::= .cfi_def_cfa register,  offset
bool AsmParser::parseDirectiveCFIDefCfa(SMLoc DirectiveLoc) {
  int64_t Register = 0;
  if (parseRegisterOrRegisterNumber(Register, DirectiveLoc))
    return true;

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in directive");
  Lex();

  int64_t Offset = 0;
  if (parseAbsoluteExpression(Offset))
    return true;

  getStreamer().EmitCFIDefCfa(Register, Offset);
  return false;
}

/// parseDirectiveCFIDefCfaOffset
/// ::= .cfi_def_cfa_offset offset
bool AsmParser::parseDirectiveCFIDefCfaOffset() {
  int64_t Offset = 0;
  if (parseAbsoluteExpression(Offset))
    return true;

  getStreamer().EmitCFIDefCfaOffset(Offset);
  return false;
}

/// parseDirectiveCFIRegister
/// ::= .cfi_register register, register
bool AsmParser::parseDirectiveCFIRegister(SMLoc DirectiveLoc) {
  int64_t Register1 = 0;
  if (parseRegisterOrRegisterNumber(Register1, DirectiveLoc))
    return true;

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in directive");
  Lex();

  int64_t Register2 = 0;
  if (parseRegisterOrRegisterNumber(Register2, DirectiveLoc))
    return true;

  getStreamer().EmitCFIRegister(Register1, Register2);
  return false;
}

/// parseDirectiveCFIWindowSave
/// ::= .cfi_window_save
bool AsmParser::parseDirectiveCFIWindowSave() {
  getStreamer().EmitCFIWindowSave();
  return false;
}

/// parseDirectiveCFIAdjustCfaOffset
/// ::= .cfi_adjust_cfa_offset adjustment
bool AsmParser::parseDirectiveCFIAdjustCfaOffset() {
  int64_t Adjustment = 0;
  if (parseAbsoluteExpression(Adjustment))
    return true;

  getStreamer().EmitCFIAdjustCfaOffset(Adjustment);
  return false;
}

/// parseDirectiveCFIDefCfaRegister
/// ::= .cfi_def_cfa_register register
bool AsmParser::parseDirectiveCFIDefCfaRegister(SMLoc DirectiveLoc) {
  int64_t Register = 0;
  if (parseRegisterOrRegisterNumber(Register, DirectiveLoc))
    return true;

  getStreamer().EmitCFIDefCfaRegister(Register);
  return false;
}

/// parseDirectiveCFIOffset
/// ::= .cfi_offset register, offset
bool AsmParser::parseDirectiveCFIOffset(SMLoc DirectiveLoc) {
  int64_t Register = 0;
  int64_t Offset = 0;

  if (parseRegisterOrRegisterNumber(Register, DirectiveLoc))
    return true;

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in directive");
  Lex();

  if (parseAbsoluteExpression(Offset))
    return true;

  getStreamer().EmitCFIOffset(Register, Offset);
  return false;
}

/// parseDirectiveCFIRelOffset
/// ::= .cfi_rel_offset register, offset
bool AsmParser::parseDirectiveCFIRelOffset(SMLoc DirectiveLoc) {
  int64_t Register = 0;

  if (parseRegisterOrRegisterNumber(Register, DirectiveLoc))
    return true;

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in directive");
  Lex();

  int64_t Offset = 0;
  if (parseAbsoluteExpression(Offset))
    return true;

  getStreamer().EmitCFIRelOffset(Register, Offset);
  return false;
}

static bool isValidEncoding(int64_t Encoding) {
  if (Encoding & ~0xff)
    return false;

  if (Encoding == dwarf::DW_EH_PE_omit)
    return true;

  const unsigned Format = Encoding & 0xf;
  if (Format != dwarf::DW_EH_PE_absptr && Format != dwarf::DW_EH_PE_udata2 &&
      Format != dwarf::DW_EH_PE_udata4 && Format != dwarf::DW_EH_PE_udata8 &&
      Format != dwarf::DW_EH_PE_sdata2 && Format != dwarf::DW_EH_PE_sdata4 &&
      Format != dwarf::DW_EH_PE_sdata8 && Format != dwarf::DW_EH_PE_signed)
    return false;

  const unsigned Application = Encoding & 0x70;
  if (Application != dwarf::DW_EH_PE_absptr &&
      Application != dwarf::DW_EH_PE_pcrel)
    return false;

  return true;
}

/// parseDirectiveCFIPersonalityOrLsda
/// IsPersonality true for cfi_personality, false for cfi_lsda
/// ::= .cfi_personality encoding, [symbol_name]
/// ::= .cfi_lsda encoding, [symbol_name]
bool AsmParser::parseDirectiveCFIPersonalityOrLsda(bool IsPersonality) {
  int64_t Encoding = 0;
  if (parseAbsoluteExpression(Encoding))
    return true;
  if (Encoding == dwarf::DW_EH_PE_omit)
    return false;

  if (!isValidEncoding(Encoding))
    return TokError("unsupported encoding.");

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in directive");
  Lex();

  StringRef Name;
  if (parseIdentifier(Name))
    return TokError("expected identifier in directive");

  MCSymbol *Sym = getContext().getOrCreateSymbol(Name);

  if (IsPersonality)
    getStreamer().EmitCFIPersonality(Sym, Encoding);
  else
    getStreamer().EmitCFILsda(Sym, Encoding);
  return false;
}

/// parseDirectiveCFIRememberState
/// ::= .cfi_remember_state
bool AsmParser::parseDirectiveCFIRememberState() {
  getStreamer().EmitCFIRememberState();
  return false;
}

/// parseDirectiveCFIRestoreState
/// ::= .cfi_remember_state
bool AsmParser::parseDirectiveCFIRestoreState() {
  getStreamer().EmitCFIRestoreState();
  return false;
}

/// parseDirectiveCFISameValue
/// ::= .cfi_same_value register
bool AsmParser::parseDirectiveCFISameValue(SMLoc DirectiveLoc) {
  int64_t Register = 0;

  if (parseRegisterOrRegisterNumber(Register, DirectiveLoc))
    return true;

  getStreamer().EmitCFISameValue(Register);
  return false;
}

/// parseDirectiveCFIRestore
/// ::= .cfi_restore register
bool AsmParser::parseDirectiveCFIRestore(SMLoc DirectiveLoc) {
  int64_t Register = 0;
  if (parseRegisterOrRegisterNumber(Register, DirectiveLoc))
    return true;

  getStreamer().EmitCFIRestore(Register);
  return false;
}

/// parseDirectiveCFIEscape
/// ::= .cfi_escape expression[,...]
bool AsmParser::parseDirectiveCFIEscape() {
  std::string Values;
  int64_t CurrValue;
  if (parseAbsoluteExpression(CurrValue))
    return true;

  Values.push_back((uint8_t)CurrValue);

  while (getLexer().is(AsmToken::Comma)) {
    Lex();

    if (parseAbsoluteExpression(CurrValue))
      return true;

    Values.push_back((uint8_t)CurrValue);
  }

  getStreamer().EmitCFIEscape(Values);
  return false;
}

/// parseDirectiveCFISignalFrame
/// ::= .cfi_signal_frame
bool AsmParser::parseDirectiveCFISignalFrame() {
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return Error(getLexer().getLoc(),
                 "unexpected token in '.cfi_signal_frame'");

  getStreamer().EmitCFISignalFrame();
  return false;
}

/// parseDirectiveCFIUndefined
/// ::= .cfi_undefined register
bool AsmParser::parseDirectiveCFIUndefined(SMLoc DirectiveLoc) {
  int64_t Register = 0;

  if (parseRegisterOrRegisterNumber(Register, DirectiveLoc))
    return true;

  getStreamer().EmitCFIUndefined(Register);
  return false;
}

/// parseDirectiveMacrosOnOff
/// ::= .macros_on
/// ::= .macros_off
bool AsmParser::parseDirectiveMacrosOnOff(StringRef Directive) {
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return Error(getLexer().getLoc(),
                 "unexpected token in '" + Directive + "' directive");

  setMacrosEnabled(Directive == ".macros_on");
  return false;
}

/// parseDirectiveMacro
/// ::= .macro name[,] [parameters]
bool AsmParser::parseDirectiveMacro(SMLoc DirectiveLoc) {
  StringRef Name;
  if (parseIdentifier(Name))
    return TokError("expected identifier in '.macro' directive");

  if (getLexer().is(AsmToken::Comma))
    Lex();

  MCAsmMacroParameters Parameters;
  while (getLexer().isNot(AsmToken::EndOfStatement)) {

    if (!Parameters.empty() && Parameters.back().Vararg)
      return Error(Lexer.getLoc(),
                   "Vararg parameter '" + Parameters.back().Name +
                   "' should be last one in the list of parameters.");

    MCAsmMacroParameter Parameter;
    if (parseIdentifier(Parameter.Name))
      return TokError("expected identifier in '.macro' directive");

    if (Lexer.is(AsmToken::Colon)) {
      Lex();  // consume ':'

      SMLoc QualLoc;
      StringRef Qualifier;

      QualLoc = Lexer.getLoc();
      if (parseIdentifier(Qualifier))
        return Error(QualLoc, "missing parameter qualifier for "
                     "'" + Parameter.Name + "' in macro '" + Name + "'");

      if (Qualifier == "req")
        Parameter.Required = true;
      else if (Qualifier == "vararg")
        Parameter.Vararg = true;
      else
        return Error(QualLoc, Qualifier + " is not a valid parameter qualifier "
                     "for '" + Parameter.Name + "' in macro '" + Name + "'");
    }

    if (getLexer().is(AsmToken::Equal)) {
      Lex();

      SMLoc ParamLoc;

      ParamLoc = Lexer.getLoc();
      if (parseMacroArgument(Parameter.Value, /*Vararg=*/false ))
        return true;

      if (Parameter.Required)
        Warning(ParamLoc, "pointless default value for required parameter "
                "'" + Parameter.Name + "' in macro '" + Name + "'");
    }

    Parameters.push_back(std::move(Parameter));

    if (getLexer().is(AsmToken::Comma))
      Lex();
  }

  // Eat the end of statement.
  Lex();

  AsmToken EndToken, StartToken = getTok();
  unsigned MacroDepth = 0;

  // Lex the macro definition.
  for (;;) {
    // Check whether we have reached the end of the file.
    if (getLexer().is(AsmToken::Eof))
      return Error(DirectiveLoc, "no matching '.endmacro' in definition");

    // Otherwise, check whether we have reach the .endmacro.
    if (getLexer().is(AsmToken::Identifier)) {
      if (getTok().getIdentifier() == ".endm" ||
          getTok().getIdentifier() == ".endmacro") {
        if (MacroDepth == 0) { // Outermost macro.
          EndToken = getTok();
          Lex();
          if (getLexer().isNot(AsmToken::EndOfStatement))
            return TokError("unexpected token in '" + EndToken.getIdentifier() +
                            "' directive");
          break;
        } else {
          // Otherwise we just found the end of an inner macro.
          --MacroDepth;
        }
      } else if (getTok().getIdentifier() == ".macro") {
        // We allow nested macros. Those aren't instantiated until the outermost
        // macro is expanded so just ignore them for now.
        ++MacroDepth;
      }
    }

    // Otherwise, scan til the end of the statement.
    eatToEndOfStatement();
  }

  if (lookupMacro(Name)) {
    return Error(DirectiveLoc, "macro '" + Name + "' is already defined");
  }

  const char *BodyStart = StartToken.getLoc().getPointer();
  const char *BodyEnd = EndToken.getLoc().getPointer();
  StringRef Body = StringRef(BodyStart, BodyEnd - BodyStart);
  checkForBadMacro(DirectiveLoc, Name, Body, Parameters);
  defineMacro(Name, MCAsmMacro(Name, Body, std::move(Parameters)));
  return false;
}

/// checkForBadMacro
///
/// With the support added for named parameters there may be code out there that
/// is transitioning from positional parameters.  In versions of gas that did
/// not support named parameters they would be ignored on the macro definition.
/// But to support both styles of parameters this is not possible so if a macro
/// definition has named parameters but does not use them and has what appears
/// to be positional parameters, strings like $1, $2, ... and $n, then issue a
/// warning that the positional parameter found in body which have no effect.
/// Hoping the developer will either remove the named parameters from the macro
/// definition so the positional parameters get used if that was what was
/// intended or change the macro to use the named parameters.  It is possible
/// this warning will trigger when the none of the named parameters are used
/// and the strings like $1 are infact to simply to be passed trough unchanged.
void AsmParser::checkForBadMacro(SMLoc DirectiveLoc, StringRef Name,
                                 StringRef Body,
                                 ArrayRef<MCAsmMacroParameter> Parameters) {
  // If this macro is not defined with named parameters the warning we are
  // checking for here doesn't apply.
  unsigned NParameters = Parameters.size();
  if (NParameters == 0)
    return;

  bool NamedParametersFound = false;
  bool PositionalParametersFound = false;

  // Look at the body of the macro for use of both the named parameters and what
  // are likely to be positional parameters.  This is what expandMacro() is
  // doing when it finds the parameters in the body.
  while (!Body.empty()) {
    // Scan for the next possible parameter.
    std::size_t End = Body.size(), Pos = 0;
    for (; Pos != End; ++Pos) {
      // Check for a substitution or escape.
      // This macro is defined with parameters, look for \foo, \bar, etc.
      if (Body[Pos] == '\\' && Pos + 1 != End)
        break;

      // This macro should have parameters, but look for $0, $1, ..., $n too.
      if (Body[Pos] != '$' || Pos + 1 == End)
        continue;
      char Next = Body[Pos + 1];
      if (Next == '$' || Next == 'n' ||
          isdigit(static_cast<unsigned char>(Next)))
        break;
    }

    // Check if we reached the end.
    if (Pos == End)
      break;

    if (Body[Pos] == '$') {
      switch (Body[Pos + 1]) {
      // $$ => $
      case '$':
        break;

      // $n => number of arguments
      case 'n':
        PositionalParametersFound = true;
        break;

      // $[0-9] => argument
      default: {
        PositionalParametersFound = true;
        break;
      }
      }
      Pos += 2;
    } else {
      unsigned I = Pos + 1;
      while (isIdentifierChar(Body[I]) && I + 1 != End)
        ++I;

      const char *Begin = Body.data() + Pos + 1;
      StringRef Argument(Begin, I - (Pos + 1));
      unsigned Index = 0;
      for (; Index < NParameters; ++Index)
        if (Parameters[Index].Name == Argument)
          break;

      if (Index == NParameters) {
        if (Body[Pos + 1] == '(' && Body[Pos + 2] == ')')
          Pos += 3;
        else {
          Pos = I;
        }
      } else {
        NamedParametersFound = true;
        Pos += 1 + Argument.size();
      }
    }
    // Update the scan point.
    Body = Body.substr(Pos);
  }

  if (!NamedParametersFound && PositionalParametersFound)
    Warning(DirectiveLoc, "macro defined with named parameters which are not "
                          "used in macro body, possible positional parameter "
                          "found in body which will have no effect");
}

/// parseDirectiveExitMacro
/// ::= .exitm
bool AsmParser::parseDirectiveExitMacro(StringRef Directive) {
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '" + Directive + "' directive");

  if (!isInsideMacroInstantiation())
    return TokError("unexpected '" + Directive + "' in file, "
                                                 "no current macro definition");

  // Exit all conditionals that are active in the current macro.
  while (TheCondStack.size() != ActiveMacros.back()->CondStackDepth) {
    TheCondState = TheCondStack.back();
    TheCondStack.pop_back();
  }

  handleMacroExit();
  return false;
}

/// parseDirectiveEndMacro
/// ::= .endm
/// ::= .endmacro
bool AsmParser::parseDirectiveEndMacro(StringRef Directive) {
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '" + Directive + "' directive");

  // If we are inside a macro instantiation, terminate the current
  // instantiation.
  if (isInsideMacroInstantiation()) {
    handleMacroExit();
    return false;
  }

  // Otherwise, this .endmacro is a stray entry in the file; well formed
  // .endmacro directives are handled during the macro definition parsing.
  return TokError("unexpected '" + Directive + "' in file, "
                                               "no current macro definition");
}

/// parseDirectivePurgeMacro
/// ::= .purgem
bool AsmParser::parseDirectivePurgeMacro(SMLoc DirectiveLoc) {
  StringRef Name;
  if (parseIdentifier(Name))
    return TokError("expected identifier in '.purgem' directive");

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.purgem' directive");

  if (!lookupMacro(Name))
    return Error(DirectiveLoc, "macro '" + Name + "' is not defined");

  undefineMacro(Name);
  return false;
}

/// parseDirectiveBundleAlignMode
/// ::= {.bundle_align_mode} expression
bool AsmParser::parseDirectiveBundleAlignMode() {
  checkForValidSection();

  // Expect a single argument: an expression that evaluates to a constant
  // in the inclusive range 0-30.
  SMLoc ExprLoc = getLexer().getLoc();
  int64_t AlignSizePow2;
  if (parseAbsoluteExpression(AlignSizePow2))
    return true;
  else if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token after expression in"
                    " '.bundle_align_mode' directive");
  else if (AlignSizePow2 < 0 || AlignSizePow2 > 30)
    return Error(ExprLoc,
                 "invalid bundle alignment size (expected between 0 and 30)");

  Lex();

  // Because of AlignSizePow2's verified range we can safely truncate it to
  // unsigned.
  getStreamer().EmitBundleAlignMode(static_cast<unsigned>(AlignSizePow2));
  return false;
}

/// parseDirectiveBundleLock
/// ::= {.bundle_lock} [align_to_end]
bool AsmParser::parseDirectiveBundleLock() {
  checkForValidSection();
  bool AlignToEnd = false;

  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    StringRef Option;
    SMLoc Loc = getTok().getLoc();
    const char *kInvalidOptionError =
        "invalid option for '.bundle_lock' directive";

    if (parseIdentifier(Option))
      return Error(Loc, kInvalidOptionError);

    if (Option != "align_to_end")
      return Error(Loc, kInvalidOptionError);
    else if (getLexer().isNot(AsmToken::EndOfStatement))
      return Error(Loc,
                   "unexpected token after '.bundle_lock' directive option");
    AlignToEnd = true;
  }

  Lex();

  getStreamer().EmitBundleLock(AlignToEnd);
  return false;
}

/// parseDirectiveBundleLock
/// ::= {.bundle_lock}
bool AsmParser::parseDirectiveBundleUnlock() {
  checkForValidSection();

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.bundle_unlock' directive");
  Lex();

  getStreamer().EmitBundleUnlock();
  return false;
}

/// parseDirectiveSpace
/// ::= (.skip | .space) expression [ , expression ]
bool AsmParser::parseDirectiveSpace(StringRef IDVal) {
  checkForValidSection();

  int64_t NumBytes;
  if (parseAbsoluteExpression(NumBytes))
    return true;

  int64_t FillExpr = 0;
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    if (getLexer().isNot(AsmToken::Comma))
      return TokError("unexpected token in '" + Twine(IDVal) + "' directive");
    Lex();

    if (parseAbsoluteExpression(FillExpr))
      return true;

    if (getLexer().isNot(AsmToken::EndOfStatement))
      return TokError("unexpected token in '" + Twine(IDVal) + "' directive");
  }

  Lex();

  if (NumBytes <= 0)
    return TokError("invalid number of bytes in '" + Twine(IDVal) +
                    "' directive");

  // FIXME: Sometimes the fill expr is 'nop' if it isn't supplied, instead of 0.
  getStreamer().EmitFill(NumBytes, FillExpr);

  return false;
}

/// parseDirectiveLEB128
/// ::= (.sleb128 | .uleb128) [ expression (, expression)* ]
bool AsmParser::parseDirectiveLEB128(bool Signed) {
  checkForValidSection();
  const MCExpr *Value;

  for (;;) {
    if (parseExpression(Value))
      return true;

    if (Signed)
      getStreamer().EmitSLEB128Value(Value);
    else
      getStreamer().EmitULEB128Value(Value);

    if (getLexer().is(AsmToken::EndOfStatement))
      break;

    if (getLexer().isNot(AsmToken::Comma))
      return TokError("unexpected token in directive");
    Lex();
  }

  return false;
}

/// parseDirectiveSymbolAttribute
///  ::= { ".globl", ".weak", ... } [ identifier ( , identifier )* ]
bool AsmParser::parseDirectiveSymbolAttribute(MCSymbolAttr Attr) {
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    for (;;) {
      StringRef Name;
      SMLoc Loc = getTok().getLoc();

      if (parseIdentifier(Name))
        return Error(Loc, "expected identifier in directive");

      MCSymbol *Sym = getContext().getOrCreateSymbol(Name);

      // Assembler local symbols don't make any sense here. Complain loudly.
      if (Sym->isTemporary())
        return Error(Loc, "non-local symbol required in directive");

      if (!getStreamer().EmitSymbolAttribute(Sym, Attr))
        return Error(Loc, "unable to emit symbol attribute");

      if (getLexer().is(AsmToken::EndOfStatement))
        break;

      if (getLexer().isNot(AsmToken::Comma))
        return TokError("unexpected token in directive");
      Lex();
    }
  }

  Lex();
  return false;
}

/// parseDirectiveComm
///  ::= ( .comm | .lcomm ) identifier , size_expression [ , align_expression ]
bool AsmParser::parseDirectiveComm(bool IsLocal) {
  checkForValidSection();

  SMLoc IDLoc = getLexer().getLoc();
  StringRef Name;
  if (parseIdentifier(Name))
    return TokError("expected identifier in directive");

  // Handle the identifier as the key symbol.
  MCSymbol *Sym = getContext().getOrCreateSymbol(Name);

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in directive");
  Lex();

  int64_t Size;
  SMLoc SizeLoc = getLexer().getLoc();
  if (parseAbsoluteExpression(Size))
    return true;

  int64_t Pow2Alignment = 0;
  SMLoc Pow2AlignmentLoc;
  if (getLexer().is(AsmToken::Comma)) {
    Lex();
    Pow2AlignmentLoc = getLexer().getLoc();
    if (parseAbsoluteExpression(Pow2Alignment))
      return true;

    LCOMM::LCOMMType LCOMM = Lexer.getMAI().getLCOMMDirectiveAlignmentType();
    if (IsLocal && LCOMM == LCOMM::NoAlignment)
      return Error(Pow2AlignmentLoc, "alignment not supported on this target");

    // If this target takes alignments in bytes (not log) validate and convert.
    if ((!IsLocal && Lexer.getMAI().getCOMMDirectiveAlignmentIsInBytes()) ||
        (IsLocal && LCOMM == LCOMM::ByteAlignment)) {
      if (!isPowerOf2_64(Pow2Alignment))
        return Error(Pow2AlignmentLoc, "alignment must be a power of 2");
      Pow2Alignment = Log2_64(Pow2Alignment);
    }
  }

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.comm' or '.lcomm' directive");

  Lex();

  // NOTE: a size of zero for a .comm should create a undefined symbol
  // but a size of .lcomm creates a bss symbol of size zero.
  if (Size < 0)
    return Error(SizeLoc, "invalid '.comm' or '.lcomm' directive size, can't "
                          "be less than zero");

  // NOTE: The alignment in the directive is a power of 2 value, the assembler
  // may internally end up wanting an alignment in bytes.
  // FIXME: Diagnose overflow.
  if (Pow2Alignment < 0)
    return Error(Pow2AlignmentLoc, "invalid '.comm' or '.lcomm' directive "
                                   "alignment, can't be less than zero");

  if (!Sym->isUndefined())
    return Error(IDLoc, "invalid symbol redefinition");

  // Create the Symbol as a common or local common with Size and Pow2Alignment
  if (IsLocal) {
    getStreamer().EmitLocalCommonSymbol(Sym, Size, 1 << Pow2Alignment);
    return false;
  }

  getStreamer().EmitCommonSymbol(Sym, Size, 1 << Pow2Alignment);
  return false;
}

/// parseDirectiveAbort
///  ::= .abort [... message ...]
bool AsmParser::parseDirectiveAbort() {
  // FIXME: Use loc from directive.
  SMLoc Loc = getLexer().getLoc();

  StringRef Str = parseStringToEndOfStatement();
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.abort' directive");

  Lex();

  if (Str.empty())
    Error(Loc, ".abort detected. Assembly stopping.");
  else
    Error(Loc, ".abort '" + Str + "' detected. Assembly stopping.");
  // FIXME: Actually abort assembly here.

  return false;
}

/// parseDirectiveInclude
///  ::= .include "filename"
bool AsmParser::parseDirectiveInclude() {
  if (getLexer().isNot(AsmToken::String))
    return TokError("expected string in '.include' directive");

  // Allow the strings to have escaped octal character sequence.
  std::string Filename;
  if (parseEscapedString(Filename))
    return true;
  SMLoc IncludeLoc = getLexer().getLoc();
  Lex();

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.include' directive");

  // Attempt to switch the lexer to the included file before consuming the end
  // of statement to avoid losing it when we switch.
  if (enterIncludeFile(Filename)) {
    Error(IncludeLoc, "Could not find include file '" + Filename + "'");
    return true;
  }

  return false;
}

/// parseDirectiveIncbin
///  ::= .incbin "filename"
bool AsmParser::parseDirectiveIncbin() {
  if (getLexer().isNot(AsmToken::String))
    return TokError("expected string in '.incbin' directive");

  // Allow the strings to have escaped octal character sequence.
  std::string Filename;
  if (parseEscapedString(Filename))
    return true;
  SMLoc IncbinLoc = getLexer().getLoc();
  Lex();

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.incbin' directive");

  // Attempt to process the included file.
  if (processIncbinFile(Filename)) {
    Error(IncbinLoc, "Could not find incbin file '" + Filename + "'");
    return true;
  }

  return false;
}

/// parseDirectiveIf
/// ::= .if{,eq,ge,gt,le,lt,ne} expression
bool AsmParser::parseDirectiveIf(SMLoc DirectiveLoc, DirectiveKind DirKind) {
  TheCondStack.push_back(TheCondState);
  TheCondState.TheCond = AsmCond::IfCond;
  if (TheCondState.Ignore) {
    eatToEndOfStatement();
  } else {
    int64_t ExprValue;
    if (parseAbsoluteExpression(ExprValue))
      return true;

    if (getLexer().isNot(AsmToken::EndOfStatement))
      return TokError("unexpected token in '.if' directive");

    Lex();

    switch (DirKind) {
    default:
      llvm_unreachable("unsupported directive");
    case DK_IF:
    case DK_IFNE:
      break;
    case DK_IFEQ:
      ExprValue = ExprValue == 0;
      break;
    case DK_IFGE:
      ExprValue = ExprValue >= 0;
      break;
    case DK_IFGT:
      ExprValue = ExprValue > 0;
      break;
    case DK_IFLE:
      ExprValue = ExprValue <= 0;
      break;
    case DK_IFLT:
      ExprValue = ExprValue < 0;
      break;
    }

    TheCondState.CondMet = ExprValue;
    TheCondState.Ignore = !TheCondState.CondMet;
  }

  return false;
}

/// parseDirectiveIfb
/// ::= .ifb string
bool AsmParser::parseDirectiveIfb(SMLoc DirectiveLoc, bool ExpectBlank) {
  TheCondStack.push_back(TheCondState);
  TheCondState.TheCond = AsmCond::IfCond;

  if (TheCondState.Ignore) {
    eatToEndOfStatement();
  } else {
    StringRef Str = parseStringToEndOfStatement();

    if (getLexer().isNot(AsmToken::EndOfStatement))
      return TokError("unexpected token in '.ifb' directive");

    Lex();

    TheCondState.CondMet = ExpectBlank == Str.empty();
    TheCondState.Ignore = !TheCondState.CondMet;
  }

  return false;
}

/// parseDirectiveIfc
/// ::= .ifc string1, string2
/// ::= .ifnc string1, string2
bool AsmParser::parseDirectiveIfc(SMLoc DirectiveLoc, bool ExpectEqual) {
  TheCondStack.push_back(TheCondState);
  TheCondState.TheCond = AsmCond::IfCond;

  if (TheCondState.Ignore) {
    eatToEndOfStatement();
  } else {
    StringRef Str1 = parseStringToComma();

    if (getLexer().isNot(AsmToken::Comma))
      return TokError("unexpected token in '.ifc' directive");

    Lex();

    StringRef Str2 = parseStringToEndOfStatement();

    if (getLexer().isNot(AsmToken::EndOfStatement))
      return TokError("unexpected token in '.ifc' directive");

    Lex();

    TheCondState.CondMet = ExpectEqual == (Str1.trim() == Str2.trim());
    TheCondState.Ignore = !TheCondState.CondMet;
  }

  return false;
}

/// parseDirectiveIfeqs
///   ::= .ifeqs string1, string2
bool AsmParser::parseDirectiveIfeqs(SMLoc DirectiveLoc, bool ExpectEqual) {
  if (Lexer.isNot(AsmToken::String)) {
    if (ExpectEqual)
      TokError("expected string parameter for '.ifeqs' directive");
    else
      TokError("expected string parameter for '.ifnes' directive");
    eatToEndOfStatement();
    return true;
  }

  StringRef String1 = getTok().getStringContents();
  Lex();

  if (Lexer.isNot(AsmToken::Comma)) {
    if (ExpectEqual)
      TokError("expected comma after first string for '.ifeqs' directive");
    else
      TokError("expected comma after first string for '.ifnes' directive");
    eatToEndOfStatement();
    return true;
  }

  Lex();

  if (Lexer.isNot(AsmToken::String)) {
    if (ExpectEqual)
      TokError("expected string parameter for '.ifeqs' directive");
    else
      TokError("expected string parameter for '.ifnes' directive");
    eatToEndOfStatement();
    return true;
  }

  StringRef String2 = getTok().getStringContents();
  Lex();

  TheCondStack.push_back(TheCondState);
  TheCondState.TheCond = AsmCond::IfCond;
  TheCondState.CondMet = ExpectEqual == (String1 == String2);
  TheCondState.Ignore = !TheCondState.CondMet;

  return false;
}

/// parseDirectiveIfdef
/// ::= .ifdef symbol
bool AsmParser::parseDirectiveIfdef(SMLoc DirectiveLoc, bool expect_defined) {
  StringRef Name;
  TheCondStack.push_back(TheCondState);
  TheCondState.TheCond = AsmCond::IfCond;

  if (TheCondState.Ignore) {
    eatToEndOfStatement();
  } else {
    if (parseIdentifier(Name))
      return TokError("expected identifier after '.ifdef'");

    Lex();

    MCSymbol *Sym = getContext().lookupSymbol(Name);

    if (expect_defined)
      TheCondState.CondMet = (Sym && !Sym->isUndefined());
    else
      TheCondState.CondMet = (!Sym || Sym->isUndefined());
    TheCondState.Ignore = !TheCondState.CondMet;
  }

  return false;
}

/// parseDirectiveElseIf
/// ::= .elseif expression
bool AsmParser::parseDirectiveElseIf(SMLoc DirectiveLoc) {
  if (TheCondState.TheCond != AsmCond::IfCond &&
      TheCondState.TheCond != AsmCond::ElseIfCond)
    Error(DirectiveLoc, "Encountered a .elseif that doesn't follow a .if or "
                        " an .elseif");
  TheCondState.TheCond = AsmCond::ElseIfCond;

  bool LastIgnoreState = false;
  if (!TheCondStack.empty())
    LastIgnoreState = TheCondStack.back().Ignore;
  if (LastIgnoreState || TheCondState.CondMet) {
    TheCondState.Ignore = true;
    eatToEndOfStatement();
  } else {
    int64_t ExprValue;
    if (parseAbsoluteExpression(ExprValue))
      return true;

    if (getLexer().isNot(AsmToken::EndOfStatement))
      return TokError("unexpected token in '.elseif' directive");

    Lex();
    TheCondState.CondMet = ExprValue;
    TheCondState.Ignore = !TheCondState.CondMet;
  }

  return false;
}

/// parseDirectiveElse
/// ::= .else
bool AsmParser::parseDirectiveElse(SMLoc DirectiveLoc) {
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.else' directive");

  Lex();

  if (TheCondState.TheCond != AsmCond::IfCond &&
      TheCondState.TheCond != AsmCond::ElseIfCond)
    Error(DirectiveLoc, "Encountered a .else that doesn't follow a .if or an "
                        ".elseif");
  TheCondState.TheCond = AsmCond::ElseCond;
  bool LastIgnoreState = false;
  if (!TheCondStack.empty())
    LastIgnoreState = TheCondStack.back().Ignore;
  if (LastIgnoreState || TheCondState.CondMet)
    TheCondState.Ignore = true;
  else
    TheCondState.Ignore = false;

  return false;
}

/// parseDirectiveEnd
/// ::= .end
bool AsmParser::parseDirectiveEnd(SMLoc DirectiveLoc) {
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.end' directive");

  Lex();

  while (Lexer.isNot(AsmToken::Eof))
    Lex();

  return false;
}

/// parseDirectiveError
///   ::= .err
///   ::= .error [string]
bool AsmParser::parseDirectiveError(SMLoc L, bool WithMessage) {
  if (!TheCondStack.empty()) {
    if (TheCondStack.back().Ignore) {
      eatToEndOfStatement();
      return false;
    }
  }

  if (!WithMessage)
    return Error(L, ".err encountered");

  StringRef Message = ".error directive invoked in source file";
  if (Lexer.isNot(AsmToken::EndOfStatement)) {
    if (Lexer.isNot(AsmToken::String)) {
      TokError(".error argument must be a string");
      eatToEndOfStatement();
      return true;
    }

    Message = getTok().getStringContents();
    Lex();
  }

  Error(L, Message);
  return true;
}

/// parseDirectiveWarning
///   ::= .warning [string]
bool AsmParser::parseDirectiveWarning(SMLoc L) {
  if (!TheCondStack.empty()) {
    if (TheCondStack.back().Ignore) {
      eatToEndOfStatement();
      return false;
    }
  }

  StringRef Message = ".warning directive invoked in source file";
  if (Lexer.isNot(AsmToken::EndOfStatement)) {
    if (Lexer.isNot(AsmToken::String)) {
      TokError(".warning argument must be a string");
      eatToEndOfStatement();
      return true;
    }

    Message = getTok().getStringContents();
    Lex();
  }

  Warning(L, Message);
  return false;
}

/// parseDirectiveEndIf
/// ::= .endif
bool AsmParser::parseDirectiveEndIf(SMLoc DirectiveLoc) {
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.endif' directive");

  Lex();

  if ((TheCondState.TheCond == AsmCond::NoCond) || TheCondStack.empty())
    Error(DirectiveLoc, "Encountered a .endif that doesn't follow a .if or "
                        ".else");
  if (!TheCondStack.empty()) {
    TheCondState = TheCondStack.back();
    TheCondStack.pop_back();
  }

  return false;
}

void AsmParser::initializeDirectiveKindMap() {
  DirectiveKindMap[".set"] = DK_SET;
  DirectiveKindMap[".equ"] = DK_EQU;
  DirectiveKindMap[".equiv"] = DK_EQUIV;
  DirectiveKindMap[".ascii"] = DK_ASCII;
  DirectiveKindMap[".asciz"] = DK_ASCIZ;
  DirectiveKindMap[".string"] = DK_STRING;
  DirectiveKindMap[".byte"] = DK_BYTE;
  DirectiveKindMap[".short"] = DK_SHORT;
  DirectiveKindMap[".value"] = DK_VALUE;
  DirectiveKindMap[".2byte"] = DK_2BYTE;
  DirectiveKindMap[".long"] = DK_LONG;
  DirectiveKindMap[".int"] = DK_INT;
  DirectiveKindMap[".4byte"] = DK_4BYTE;
  DirectiveKindMap[".quad"] = DK_QUAD;
  DirectiveKindMap[".8byte"] = DK_8BYTE;
  DirectiveKindMap[".octa"] = DK_OCTA;
  DirectiveKindMap[".single"] = DK_SINGLE;
  DirectiveKindMap[".float"] = DK_FLOAT;
  DirectiveKindMap[".double"] = DK_DOUBLE;
  DirectiveKindMap[".align"] = DK_ALIGN;
  DirectiveKindMap[".align32"] = DK_ALIGN32;
  DirectiveKindMap[".balign"] = DK_BALIGN;
  DirectiveKindMap[".balignw"] = DK_BALIGNW;
  DirectiveKindMap[".balignl"] = DK_BALIGNL;
  DirectiveKindMap[".p2align"] = DK_P2ALIGN;
  DirectiveKindMap[".p2alignw"] = DK_P2ALIGNW;
  DirectiveKindMap[".p2alignl"] = DK_P2ALIGNL;
  DirectiveKindMap[".org"] = DK_ORG;
  DirectiveKindMap[".fill"] = DK_FILL;
  DirectiveKindMap[".zero"] = DK_ZERO;
  DirectiveKindMap[".extern"] = DK_EXTERN;
  DirectiveKindMap[".globl"] = DK_GLOBL;
  DirectiveKindMap[".global"] = DK_GLOBAL;
  DirectiveKindMap[".lazy_reference"] = DK_LAZY_REFERENCE;
  DirectiveKindMap[".no_dead_strip"] = DK_NO_DEAD_STRIP;
  DirectiveKindMap[".symbol_resolver"] = DK_SYMBOL_RESOLVER;
  DirectiveKindMap[".private_extern"] = DK_PRIVATE_EXTERN;
  DirectiveKindMap[".reference"] = DK_REFERENCE;
  DirectiveKindMap[".weak_definition"] = DK_WEAK_DEFINITION;
  DirectiveKindMap[".weak_reference"] = DK_WEAK_REFERENCE;
  DirectiveKindMap[".weak_def_can_be_hidden"] = DK_WEAK_DEF_CAN_BE_HIDDEN;
  DirectiveKindMap[".comm"] = DK_COMM;
  DirectiveKindMap[".common"] = DK_COMMON;
  DirectiveKindMap[".lcomm"] = DK_LCOMM;
  DirectiveKindMap[".abort"] = DK_ABORT;
  DirectiveKindMap[".include"] = DK_INCLUDE;
  DirectiveKindMap[".incbin"] = DK_INCBIN;
  DirectiveKindMap[".code16"] = DK_CODE16;
  DirectiveKindMap[".code16gcc"] = DK_CODE16GCC;
  DirectiveKindMap[".rept"] = DK_REPT;
  DirectiveKindMap[".rep"] = DK_REPT;
  DirectiveKindMap[".irp"] = DK_IRP;
  DirectiveKindMap[".irpc"] = DK_IRPC;
  DirectiveKindMap[".endr"] = DK_ENDR;
  DirectiveKindMap[".bundle_align_mode"] = DK_BUNDLE_ALIGN_MODE;
  DirectiveKindMap[".bundle_lock"] = DK_BUNDLE_LOCK;
  DirectiveKindMap[".bundle_unlock"] = DK_BUNDLE_UNLOCK;
  DirectiveKindMap[".if"] = DK_IF;
  DirectiveKindMap[".ifeq"] = DK_IFEQ;
  DirectiveKindMap[".ifge"] = DK_IFGE;
  DirectiveKindMap[".ifgt"] = DK_IFGT;
  DirectiveKindMap[".ifle"] = DK_IFLE;
  DirectiveKindMap[".iflt"] = DK_IFLT;
  DirectiveKindMap[".ifne"] = DK_IFNE;
  DirectiveKindMap[".ifb"] = DK_IFB;
  DirectiveKindMap[".ifnb"] = DK_IFNB;
  DirectiveKindMap[".ifc"] = DK_IFC;
  DirectiveKindMap[".ifeqs"] = DK_IFEQS;
  DirectiveKindMap[".ifnc"] = DK_IFNC;
  DirectiveKindMap[".ifnes"] = DK_IFNES;
  DirectiveKindMap[".ifdef"] = DK_IFDEF;
  DirectiveKindMap[".ifndef"] = DK_IFNDEF;
  DirectiveKindMap[".ifnotdef"] = DK_IFNOTDEF;
  DirectiveKindMap[".elseif"] = DK_ELSEIF;
  DirectiveKindMap[".else"] = DK_ELSE;
  DirectiveKindMap[".end"] = DK_END;
  DirectiveKindMap[".endif"] = DK_ENDIF;
  DirectiveKindMap[".skip"] = DK_SKIP;
  DirectiveKindMap[".space"] = DK_SPACE;
  DirectiveKindMap[".file"] = DK_FILE;
  DirectiveKindMap[".line"] = DK_LINE;
  DirectiveKindMap[".loc"] = DK_LOC;
  DirectiveKindMap[".stabs"] = DK_STABS;
  DirectiveKindMap[".sleb128"] = DK_SLEB128;
  DirectiveKindMap[".uleb128"] = DK_ULEB128;
  DirectiveKindMap[".cfi_sections"] = DK_CFI_SECTIONS;
  DirectiveKindMap[".cfi_startproc"] = DK_CFI_STARTPROC;
  DirectiveKindMap[".cfi_endproc"] = DK_CFI_ENDPROC;
  DirectiveKindMap[".cfi_def_cfa"] = DK_CFI_DEF_CFA;
  DirectiveKindMap[".cfi_def_cfa_offset"] = DK_CFI_DEF_CFA_OFFSET;
  DirectiveKindMap[".cfi_adjust_cfa_offset"] = DK_CFI_ADJUST_CFA_OFFSET;
  DirectiveKindMap[".cfi_def_cfa_register"] = DK_CFI_DEF_CFA_REGISTER;
  DirectiveKindMap[".cfi_offset"] = DK_CFI_OFFSET;
  DirectiveKindMap[".cfi_rel_offset"] = DK_CFI_REL_OFFSET;
  DirectiveKindMap[".cfi_personality"] = DK_CFI_PERSONALITY;
  DirectiveKindMap[".cfi_lsda"] = DK_CFI_LSDA;
  DirectiveKindMap[".cfi_remember_state"] = DK_CFI_REMEMBER_STATE;
  DirectiveKindMap[".cfi_restore_state"] = DK_CFI_RESTORE_STATE;
  DirectiveKindMap[".cfi_same_value"] = DK_CFI_SAME_VALUE;
  DirectiveKindMap[".cfi_restore"] = DK_CFI_RESTORE;
  DirectiveKindMap[".cfi_escape"] = DK_CFI_ESCAPE;
  DirectiveKindMap[".cfi_signal_frame"] = DK_CFI_SIGNAL_FRAME;
  DirectiveKindMap[".cfi_undefined"] = DK_CFI_UNDEFINED;
  DirectiveKindMap[".cfi_register"] = DK_CFI_REGISTER;
  DirectiveKindMap[".cfi_window_save"] = DK_CFI_WINDOW_SAVE;
  DirectiveKindMap[".macros_on"] = DK_MACROS_ON;
  DirectiveKindMap[".macros_off"] = DK_MACROS_OFF;
  DirectiveKindMap[".macro"] = DK_MACRO;
  DirectiveKindMap[".exitm"] = DK_EXITM;
  DirectiveKindMap[".endm"] = DK_ENDM;
  DirectiveKindMap[".endmacro"] = DK_ENDMACRO;
  DirectiveKindMap[".purgem"] = DK_PURGEM;
  DirectiveKindMap[".err"] = DK_ERR;
  DirectiveKindMap[".error"] = DK_ERROR;
  DirectiveKindMap[".warning"] = DK_WARNING;
}

MCAsmMacro *AsmParser::parseMacroLikeBody(SMLoc DirectiveLoc) {
  AsmToken EndToken, StartToken = getTok();

  unsigned NestLevel = 0;
  for (;;) {
    // Check whether we have reached the end of the file.
    if (getLexer().is(AsmToken::Eof)) {
      Error(DirectiveLoc, "no matching '.endr' in definition");
      return nullptr;
    }

    if (Lexer.is(AsmToken::Identifier) &&
        (getTok().getIdentifier() == ".rept")) {
      ++NestLevel;
    }

    // Otherwise, check whether we have reached the .endr.
    if (Lexer.is(AsmToken::Identifier) && getTok().getIdentifier() == ".endr") {
      if (NestLevel == 0) {
        EndToken = getTok();
        Lex();
        if (Lexer.isNot(AsmToken::EndOfStatement)) {
          TokError("unexpected token in '.endr' directive");
          return nullptr;
        }
        break;
      }
      --NestLevel;
    }

    // Otherwise, scan till the end of the statement.
    eatToEndOfStatement();
  }

  const char *BodyStart = StartToken.getLoc().getPointer();
  const char *BodyEnd = EndToken.getLoc().getPointer();
  StringRef Body = StringRef(BodyStart, BodyEnd - BodyStart);

  // We Are Anonymous.
  MacroLikeBodies.emplace_back(StringRef(), Body, MCAsmMacroParameters());
  return &MacroLikeBodies.back();
}

void AsmParser::instantiateMacroLikeBody(MCAsmMacro *M, SMLoc DirectiveLoc,
                                         raw_svector_ostream &OS) {
  OS << ".endr\n";

  std::unique_ptr<MemoryBuffer> Instantiation =
      MemoryBuffer::getMemBufferCopy(OS.str(), "<instantiation>");

  // Create the macro instantiation object and add to the current macro
  // instantiation stack.
  MacroInstantiation *MI = new MacroInstantiation(
      DirectiveLoc, CurBuffer, getTok().getLoc(), TheCondStack.size());
  ActiveMacros.push_back(MI);

  // Jump to the macro instantiation and prime the lexer.
  CurBuffer = SrcMgr.AddNewSourceBuffer(std::move(Instantiation), SMLoc());
  Lexer.setBuffer(SrcMgr.getMemoryBuffer(CurBuffer)->getBuffer());
  Lex();
}

/// parseDirectiveRept
///   ::= .rep | .rept count
bool AsmParser::parseDirectiveRept(SMLoc DirectiveLoc, StringRef Dir) {
  const MCExpr *CountExpr;
  SMLoc CountLoc = getTok().getLoc();
  if (parseExpression(CountExpr))
    return true;

  int64_t Count;
  if (!CountExpr->evaluateAsAbsolute(Count)) {
    eatToEndOfStatement();
    return Error(CountLoc, "unexpected token in '" + Dir + "' directive");
  }

  if (Count < 0)
    return Error(CountLoc, "Count is negative");

  if (Lexer.isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '" + Dir + "' directive");

  // Eat the end of statement.
  Lex();

  // Lex the rept definition.
  MCAsmMacro *M = parseMacroLikeBody(DirectiveLoc);
  if (!M)
    return true;

  // Macro instantiation is lexical, unfortunately. We construct a new buffer
  // to hold the macro body with substitutions.
  SmallString<256> Buf;
  raw_svector_ostream OS(Buf);
  while (Count--) {
    // Note that the AtPseudoVariable is disabled for instantiations of .rep(t).
    if (expandMacro(OS, M->Body, None, None, false, getTok().getLoc()))
      return true;
  }
  instantiateMacroLikeBody(M, DirectiveLoc, OS);

  return false;
}

/// parseDirectiveIrp
/// ::= .irp symbol,values
bool AsmParser::parseDirectiveIrp(SMLoc DirectiveLoc) {
  MCAsmMacroParameter Parameter;

  if (parseIdentifier(Parameter.Name))
    return TokError("expected identifier in '.irp' directive");

  if (Lexer.isNot(AsmToken::Comma))
    return TokError("expected comma in '.irp' directive");

  Lex();

  MCAsmMacroArguments A;
  if (parseMacroArguments(nullptr, A))
    return true;

  // Eat the end of statement.
  Lex();

  // Lex the irp definition.
  MCAsmMacro *M = parseMacroLikeBody(DirectiveLoc);
  if (!M)
    return true;

  // Macro instantiation is lexical, unfortunately. We construct a new buffer
  // to hold the macro body with substitutions.
  SmallString<256> Buf;
  raw_svector_ostream OS(Buf);

  for (MCAsmMacroArguments::iterator i = A.begin(), e = A.end(); i != e; ++i) {
    // Note that the AtPseudoVariable is enabled for instantiations of .irp.
    // This is undocumented, but GAS seems to support it.
    if (expandMacro(OS, M->Body, Parameter, *i, true, getTok().getLoc()))
      return true;
  }

  instantiateMacroLikeBody(M, DirectiveLoc, OS);

  return false;
}

/// parseDirectiveIrpc
/// ::= .irpc symbol,values
bool AsmParser::parseDirectiveIrpc(SMLoc DirectiveLoc) {
  MCAsmMacroParameter Parameter;

  if (parseIdentifier(Parameter.Name))
    return TokError("expected identifier in '.irpc' directive");

  if (Lexer.isNot(AsmToken::Comma))
    return TokError("expected comma in '.irpc' directive");

  Lex();

  MCAsmMacroArguments A;
  if (parseMacroArguments(nullptr, A))
    return true;

  if (A.size() != 1 || A.front().size() != 1)
    return TokError("unexpected token in '.irpc' directive");

  // Eat the end of statement.
  Lex();

  // Lex the irpc definition.
  MCAsmMacro *M = parseMacroLikeBody(DirectiveLoc);
  if (!M)
    return true;

  // Macro instantiation is lexical, unfortunately. We construct a new buffer
  // to hold the macro body with substitutions.
  SmallString<256> Buf;
  raw_svector_ostream OS(Buf);

  StringRef Values = A.front().front().getString();
  for (std::size_t I = 0, End = Values.size(); I != End; ++I) {
    MCAsmMacroArgument Arg;
    Arg.emplace_back(AsmToken::Identifier, Values.slice(I, I + 1));

    // Note that the AtPseudoVariable is enabled for instantiations of .irpc.
    // This is undocumented, but GAS seems to support it.
    if (expandMacro(OS, M->Body, Parameter, Arg, true, getTok().getLoc()))
      return true;
  }

  instantiateMacroLikeBody(M, DirectiveLoc, OS);

  return false;
}

bool AsmParser::parseDirectiveEndr(SMLoc DirectiveLoc) {
  if (ActiveMacros.empty())
    return TokError("unmatched '.endr' directive");

  // The only .repl that should get here are the ones created by
  // instantiateMacroLikeBody.
  assert(getLexer().is(AsmToken::EndOfStatement));

  handleMacroExit();
  return false;
}

bool AsmParser::parseDirectiveMSEmit(SMLoc IDLoc, ParseStatementInfo &Info,
                                     size_t Len) {
  const MCExpr *Value;
  SMLoc ExprLoc = getLexer().getLoc();
  if (parseExpression(Value))
    return true;
  const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(Value);
  if (!MCE)
    return Error(ExprLoc, "unexpected expression in _emit");
  uint64_t IntValue = MCE->getValue();
  if (!isUIntN(8, IntValue) && !isIntN(8, IntValue))
    return Error(ExprLoc, "literal value out of range for directive");

  Info.AsmRewrites->push_back(AsmRewrite(AOK_Emit, IDLoc, Len));
  return false;
}

bool AsmParser::parseDirectiveMSAlign(SMLoc IDLoc, ParseStatementInfo &Info) {
  const MCExpr *Value;
  SMLoc ExprLoc = getLexer().getLoc();
  if (parseExpression(Value))
    return true;
  const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(Value);
  if (!MCE)
    return Error(ExprLoc, "unexpected expression in align");
  uint64_t IntValue = MCE->getValue();
  if (!isPowerOf2_64(IntValue))
    return Error(ExprLoc, "literal value not a power of two greater then zero");

  Info.AsmRewrites->push_back(
      AsmRewrite(AOK_Align, IDLoc, 5, Log2_64(IntValue)));
  return false;
}

// We are comparing pointers, but the pointers are relative to a single string.
// Thus, this should always be deterministic.
// HLSL Change: changed calling convention to __cdecl
static int __cdecl rewritesSort(const AsmRewrite *AsmRewriteA,
                        const AsmRewrite *AsmRewriteB) {
  if (AsmRewriteA->Loc.getPointer() < AsmRewriteB->Loc.getPointer())
    return -1;
  if (AsmRewriteB->Loc.getPointer() < AsmRewriteA->Loc.getPointer())
    return 1;

  // It's possible to have a SizeDirective, Imm/ImmPrefix and an Input/Output
  // rewrite to the same location.  Make sure the SizeDirective rewrite is
  // performed first, then the Imm/ImmPrefix and finally the Input/Output.  This
  // ensures the sort algorithm is stable.
  if (AsmRewritePrecedence[AsmRewriteA->Kind] >
      AsmRewritePrecedence[AsmRewriteB->Kind])
    return -1;

  if (AsmRewritePrecedence[AsmRewriteA->Kind] <
      AsmRewritePrecedence[AsmRewriteB->Kind])
    return 1;
  llvm_unreachable("Unstable rewrite sort.");
}

bool AsmParser::parseMSInlineAsm(
    void *AsmLoc, std::string &AsmString, unsigned &NumOutputs,
    unsigned &NumInputs, SmallVectorImpl<std::pair<void *, bool> > &OpDecls,
    SmallVectorImpl<std::string> &Constraints,
    SmallVectorImpl<std::string> &Clobbers, const MCInstrInfo *MII,
    const MCInstPrinter *IP, MCAsmParserSemaCallback &SI) {
  SmallVector<void *, 4> InputDecls;
  SmallVector<void *, 4> OutputDecls;
  SmallVector<bool, 4> InputDeclsAddressOf;
  SmallVector<bool, 4> OutputDeclsAddressOf;
  SmallVector<std::string, 4> InputConstraints;
  SmallVector<std::string, 4> OutputConstraints;
  SmallVector<unsigned, 4> ClobberRegs;

  SmallVector<AsmRewrite, 4> AsmStrRewrites;

  // Prime the lexer.
  Lex();

  // While we have input, parse each statement.
  unsigned InputIdx = 0;
  unsigned OutputIdx = 0;
  while (getLexer().isNot(AsmToken::Eof)) {
    ParseStatementInfo Info(&AsmStrRewrites);
    if (parseStatement(Info, &SI))
      return true;

    if (Info.ParseError)
      return true;

    if (Info.Opcode == ~0U)
      continue;

    const MCInstrDesc &Desc = MII->get(Info.Opcode);

    // Build the list of clobbers, outputs and inputs.
    for (unsigned i = 1, e = Info.ParsedOperands.size(); i != e; ++i) {
      MCParsedAsmOperand &Operand = *Info.ParsedOperands[i];

      // Immediate.
      if (Operand.isImm())
        continue;

      // Register operand.
      if (Operand.isReg() && !Operand.needAddressOf() &&
          !getTargetParser().OmitRegisterFromClobberLists(Operand.getReg())) {
        unsigned NumDefs = Desc.getNumDefs();
        // Clobber.
        if (NumDefs && Operand.getMCOperandNum() < NumDefs)
          ClobberRegs.push_back(Operand.getReg());
        continue;
      }

      // Expr/Input or Output.
      StringRef SymName = Operand.getSymName();
      if (SymName.empty())
        continue;

      void *OpDecl = Operand.getOpDecl();
      if (!OpDecl)
        continue;

      bool isOutput = (i == 1) && Desc.mayStore();
      SMLoc Start = SMLoc::getFromPointer(SymName.data());
      if (isOutput) {
        ++InputIdx;
        OutputDecls.push_back(OpDecl);
        OutputDeclsAddressOf.push_back(Operand.needAddressOf());
        OutputConstraints.push_back(("=" + Operand.getConstraint()).str());
        AsmStrRewrites.push_back(AsmRewrite(AOK_Output, Start, SymName.size()));
      } else {
        InputDecls.push_back(OpDecl);
        InputDeclsAddressOf.push_back(Operand.needAddressOf());
        InputConstraints.push_back(Operand.getConstraint().str());
        AsmStrRewrites.push_back(AsmRewrite(AOK_Input, Start, SymName.size()));
      }
    }

    // Consider implicit defs to be clobbers.  Think of cpuid and push.
    ArrayRef<uint16_t> ImpDefs(Desc.getImplicitDefs(),
                               Desc.getNumImplicitDefs());
    ClobberRegs.insert(ClobberRegs.end(), ImpDefs.begin(), ImpDefs.end());
  }

  // Set the number of Outputs and Inputs.
  NumOutputs = OutputDecls.size();
  NumInputs = InputDecls.size();

  // Set the unique clobbers.
  array_pod_sort(ClobberRegs.begin(), ClobberRegs.end());
  ClobberRegs.erase(std::unique(ClobberRegs.begin(), ClobberRegs.end()),
                    ClobberRegs.end());
  Clobbers.assign(ClobberRegs.size(), std::string());
  for (unsigned I = 0, E = ClobberRegs.size(); I != E; ++I) {
    raw_string_ostream OS(Clobbers[I]);
    IP->printRegName(OS, ClobberRegs[I]);
  }

  // Merge the various outputs and inputs.  Output are expected first.
  if (NumOutputs || NumInputs) {
    unsigned NumExprs = NumOutputs + NumInputs;
    OpDecls.resize(NumExprs);
    Constraints.resize(NumExprs);
    for (unsigned i = 0; i < NumOutputs; ++i) {
      OpDecls[i] = std::make_pair(OutputDecls[i], OutputDeclsAddressOf[i]);
      Constraints[i] = OutputConstraints[i];
    }
    for (unsigned i = 0, j = NumOutputs; i < NumInputs; ++i, ++j) {
      OpDecls[j] = std::make_pair(InputDecls[i], InputDeclsAddressOf[i]);
      Constraints[j] = InputConstraints[i];
    }
  }

  // Build the IR assembly string.
  std::string AsmStringIR;
  raw_string_ostream OS(AsmStringIR);
  StringRef ASMString =
      SrcMgr.getMemoryBuffer(SrcMgr.getMainFileID())->getBuffer();
  const char *AsmStart = ASMString.begin();
  const char *AsmEnd = ASMString.end();
  array_pod_sort(AsmStrRewrites.begin(), AsmStrRewrites.end(), rewritesSort);
  for (const AsmRewrite &AR : AsmStrRewrites) {
    AsmRewriteKind Kind = AR.Kind;
    if (Kind == AOK_Delete)
      continue;

    const char *Loc = AR.Loc.getPointer();
    assert(Loc >= AsmStart && "Expected Loc to be at or after Start!");

    // Emit everything up to the immediate/expression.
    if (unsigned Len = Loc - AsmStart)
      OS << StringRef(AsmStart, Len);

    // Skip the original expression.
    if (Kind == AOK_Skip) {
      AsmStart = Loc + AR.Len;
      continue;
    }

    unsigned AdditionalSkip = 0;
    // Rewrite expressions in $N notation.
    switch (Kind) {
    default:
      break;
    case AOK_Imm:
      OS << "$$" << AR.Val;
      break;
    case AOK_ImmPrefix:
      OS << "$$";
      break;
    case AOK_Label:
      OS << Ctx.getAsmInfo()->getPrivateLabelPrefix() << AR.Label;
      break;
    case AOK_Input:
      OS << '$' << InputIdx++;
      break;
    case AOK_Output:
      OS << '$' << OutputIdx++;
      break;
    case AOK_SizeDirective:
      switch (AR.Val) {
      default: break;
      case 8:  OS << "byte ptr "; break;
      case 16: OS << "word ptr "; break;
      case 32: OS << "dword ptr "; break;
      case 64: OS << "qword ptr "; break;
      case 80: OS << "xword ptr "; break;
      case 128: OS << "xmmword ptr "; break;
      case 256: OS << "ymmword ptr "; break;
      }
      break;
    case AOK_Emit:
      OS << ".byte";
      break;
    case AOK_Align: {
      unsigned Val = AR.Val;
      OS << ".align " << Val;

      // Skip the original immediate.
      assert(Val < 10 && "Expected alignment less then 2^10.");
      AdditionalSkip = (Val < 4) ? 2 : Val < 7 ? 3 : 4;
      break;
    }
    case AOK_DotOperator:
      // Insert the dot if the user omitted it.
      OS.flush();
      if (AsmStringIR.back() != '.')
        OS << '.';
      OS << AR.Val;
      break;
    }

    // Skip the original expression.
    AsmStart = Loc + AR.Len + AdditionalSkip;
  }

  // Emit the remainder of the asm string.
  if (AsmStart != AsmEnd)
    OS << StringRef(AsmStart, AsmEnd - AsmStart);

  AsmString = OS.str();
  return false;
}

namespace llvm {
namespace MCParserUtils {

/// Returns whether the given symbol is used anywhere in the given expression,
/// or subexpressions.
static bool isSymbolUsedInExpression(const MCSymbol *Sym, const MCExpr *Value) {
  switch (Value->getKind()) {
  case MCExpr::Binary: {
    const MCBinaryExpr *BE = static_cast<const MCBinaryExpr *>(Value);
    return isSymbolUsedInExpression(Sym, BE->getLHS()) ||
           isSymbolUsedInExpression(Sym, BE->getRHS());
  }
  case MCExpr::Target:
  case MCExpr::Constant:
    return false;
  case MCExpr::SymbolRef: {
    const MCSymbol &S =
        static_cast<const MCSymbolRefExpr *>(Value)->getSymbol();
    if (S.isVariable())
      return isSymbolUsedInExpression(Sym, S.getVariableValue());
    return &S == Sym;
  }
  case MCExpr::Unary:
    return isSymbolUsedInExpression(
        Sym, static_cast<const MCUnaryExpr *>(Value)->getSubExpr());
  }

  llvm_unreachable("Unknown expr kind!");
}

bool parseAssignmentExpression(StringRef Name, bool allow_redef,
                               MCAsmParser &Parser, MCSymbol *&Sym,
                               const MCExpr *&Value) {
  MCAsmLexer &Lexer = Parser.getLexer();

  // FIXME: Use better location, we should use proper tokens.
  SMLoc EqualLoc = Lexer.getLoc();

  if (Parser.parseExpression(Value)) {
    Parser.TokError("missing expression");
    Parser.eatToEndOfStatement();
    return true;
  }

  // Note: we don't count b as used in "a = b". This is to allow
  // a = b
  // b = c

  if (Lexer.isNot(AsmToken::EndOfStatement))
    return Parser.TokError("unexpected token in assignment");

  // Eat the end of statement marker.
  Parser.Lex();

  // Validate that the LHS is allowed to be a variable (either it has not been
  // used as a symbol, or it is an absolute symbol).
  Sym = Parser.getContext().lookupSymbol(Name);
  if (Sym) {
    // Diagnose assignment to a label.
    //
    // FIXME: Diagnostics. Note the location of the definition as a label.
    // FIXME: Diagnose assignment to protected identifier (e.g., register name).
    if (isSymbolUsedInExpression(Sym, Value))
      return Parser.Error(EqualLoc, "Recursive use of '" + Name + "'");
    else if (Sym->isUndefined() && !Sym->isUsed() && !Sym->isVariable())
      ; // Allow redefinitions of undefined symbols only used in directives.
    else if (Sym->isVariable() && !Sym->isUsed() && allow_redef)
      ; // Allow redefinitions of variables that haven't yet been used.
    else if (!Sym->isUndefined() && (!Sym->isVariable() || !allow_redef))
      return Parser.Error(EqualLoc, "redefinition of '" + Name + "'");
    else if (!Sym->isVariable())
      return Parser.Error(EqualLoc, "invalid assignment to '" + Name + "'");
    else if (!isa<MCConstantExpr>(Sym->getVariableValue()))
      return Parser.Error(EqualLoc,
                          "invalid reassignment of non-absolute variable '" +
                              Name + "'");

    // Don't count these checks as uses.
    Sym->setUsed(false);
  } else if (Name == ".") {
    if (Parser.getStreamer().EmitValueToOffset(Value, 0)) {
      Parser.Error(EqualLoc, "expected absolute expression");
      Parser.eatToEndOfStatement();
      return true;
    }
    return false;
  } else
    Sym = Parser.getContext().getOrCreateSymbol(Name);

  Sym->setRedefinable(allow_redef);

  return false;
}

} // namespace MCParserUtils
} // namespace llvm

/// \brief Create an MCAsmParser instance.
MCAsmParser *llvm::createMCAsmParser(SourceMgr &SM, MCContext &C,
                                     MCStreamer &Out, const MCAsmInfo &MAI) {
  return new AsmParser(SM, C, Out, MAI);
}
