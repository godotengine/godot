//===- ELFAsmParser.cpp - ELF Assembly Parser -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCParser/MCAsmParserExtension.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/Support/ELF.h"
using namespace llvm;

namespace {

class ELFAsmParser : public MCAsmParserExtension {
  template<bool (ELFAsmParser::*HandlerMethod)(StringRef, SMLoc)>
  void addDirectiveHandler(StringRef Directive) {
    MCAsmParser::ExtensionDirectiveHandler Handler = std::make_pair(
        this, HandleDirective<ELFAsmParser, HandlerMethod>);

    getParser().addDirectiveHandler(Directive, Handler);
  }

  bool ParseSectionSwitch(StringRef Section, unsigned Type, unsigned Flags,
                          SectionKind Kind);

public:
  ELFAsmParser() { BracketExpressionsSupported = true; }

  void Initialize(MCAsmParser &Parser) override {
    // Call the base implementation.
    this->MCAsmParserExtension::Initialize(Parser);

    addDirectiveHandler<&ELFAsmParser::ParseSectionDirectiveData>(".data");
    addDirectiveHandler<&ELFAsmParser::ParseSectionDirectiveText>(".text");
    addDirectiveHandler<&ELFAsmParser::ParseSectionDirectiveBSS>(".bss");
    addDirectiveHandler<&ELFAsmParser::ParseSectionDirectiveRoData>(".rodata");
    addDirectiveHandler<&ELFAsmParser::ParseSectionDirectiveTData>(".tdata");
    addDirectiveHandler<&ELFAsmParser::ParseSectionDirectiveTBSS>(".tbss");
    addDirectiveHandler<
      &ELFAsmParser::ParseSectionDirectiveDataRel>(".data.rel");
    addDirectiveHandler<
      &ELFAsmParser::ParseSectionDirectiveDataRelRo>(".data.rel.ro");
    addDirectiveHandler<
      &ELFAsmParser::ParseSectionDirectiveDataRelRoLocal>(".data.rel.ro.local");
    addDirectiveHandler<
      &ELFAsmParser::ParseSectionDirectiveEhFrame>(".eh_frame");
    addDirectiveHandler<&ELFAsmParser::ParseDirectiveSection>(".section");
    addDirectiveHandler<
      &ELFAsmParser::ParseDirectivePushSection>(".pushsection");
    addDirectiveHandler<&ELFAsmParser::ParseDirectivePopSection>(".popsection");
    addDirectiveHandler<&ELFAsmParser::ParseDirectiveSize>(".size");
    addDirectiveHandler<&ELFAsmParser::ParseDirectivePrevious>(".previous");
    addDirectiveHandler<&ELFAsmParser::ParseDirectiveType>(".type");
    addDirectiveHandler<&ELFAsmParser::ParseDirectiveIdent>(".ident");
    addDirectiveHandler<&ELFAsmParser::ParseDirectiveSymver>(".symver");
    addDirectiveHandler<&ELFAsmParser::ParseDirectiveVersion>(".version");
    addDirectiveHandler<&ELFAsmParser::ParseDirectiveWeakref>(".weakref");
    addDirectiveHandler<&ELFAsmParser::ParseDirectiveSymbolAttribute>(".weak");
    addDirectiveHandler<&ELFAsmParser::ParseDirectiveSymbolAttribute>(".local");
    addDirectiveHandler<
      &ELFAsmParser::ParseDirectiveSymbolAttribute>(".protected");
    addDirectiveHandler<
      &ELFAsmParser::ParseDirectiveSymbolAttribute>(".internal");
    addDirectiveHandler<
      &ELFAsmParser::ParseDirectiveSymbolAttribute>(".hidden");
    addDirectiveHandler<&ELFAsmParser::ParseDirectiveSubsection>(".subsection");
  }

  // FIXME: Part of this logic is duplicated in the MCELFStreamer. What is
  // the best way for us to get access to it?
  bool ParseSectionDirectiveData(StringRef, SMLoc) {
    return ParseSectionSwitch(".data", ELF::SHT_PROGBITS,
                              ELF::SHF_WRITE |ELF::SHF_ALLOC,
                              SectionKind::getDataRel());
  }
  bool ParseSectionDirectiveText(StringRef, SMLoc) {
    return ParseSectionSwitch(".text", ELF::SHT_PROGBITS,
                              ELF::SHF_EXECINSTR |
                              ELF::SHF_ALLOC, SectionKind::getText());
  }
  bool ParseSectionDirectiveBSS(StringRef, SMLoc) {
    return ParseSectionSwitch(".bss", ELF::SHT_NOBITS,
                              ELF::SHF_WRITE |
                              ELF::SHF_ALLOC, SectionKind::getBSS());
  }
  bool ParseSectionDirectiveRoData(StringRef, SMLoc) {
    return ParseSectionSwitch(".rodata", ELF::SHT_PROGBITS,
                              ELF::SHF_ALLOC,
                              SectionKind::getReadOnly());
  }
  bool ParseSectionDirectiveTData(StringRef, SMLoc) {
    return ParseSectionSwitch(".tdata", ELF::SHT_PROGBITS,
                              ELF::SHF_ALLOC |
                              ELF::SHF_TLS | ELF::SHF_WRITE,
                              SectionKind::getThreadData());
  }
  bool ParseSectionDirectiveTBSS(StringRef, SMLoc) {
    return ParseSectionSwitch(".tbss", ELF::SHT_NOBITS,
                              ELF::SHF_ALLOC |
                              ELF::SHF_TLS | ELF::SHF_WRITE,
                              SectionKind::getThreadBSS());
  }
  bool ParseSectionDirectiveDataRel(StringRef, SMLoc) {
    return ParseSectionSwitch(".data.rel", ELF::SHT_PROGBITS,
                              ELF::SHF_ALLOC |
                              ELF::SHF_WRITE,
                              SectionKind::getDataRel());
  }
  bool ParseSectionDirectiveDataRelRo(StringRef, SMLoc) {
    return ParseSectionSwitch(".data.rel.ro", ELF::SHT_PROGBITS,
                              ELF::SHF_ALLOC |
                              ELF::SHF_WRITE,
                              SectionKind::getReadOnlyWithRel());
  }
  bool ParseSectionDirectiveDataRelRoLocal(StringRef, SMLoc) {
    return ParseSectionSwitch(".data.rel.ro.local", ELF::SHT_PROGBITS,
                              ELF::SHF_ALLOC |
                              ELF::SHF_WRITE,
                              SectionKind::getReadOnlyWithRelLocal());
  }
  bool ParseSectionDirectiveEhFrame(StringRef, SMLoc) {
    return ParseSectionSwitch(".eh_frame", ELF::SHT_PROGBITS,
                              ELF::SHF_ALLOC |
                              ELF::SHF_WRITE,
                              SectionKind::getDataRel());
  }
  bool ParseDirectivePushSection(StringRef, SMLoc);
  bool ParseDirectivePopSection(StringRef, SMLoc);
  bool ParseDirectiveSection(StringRef, SMLoc);
  bool ParseDirectiveSize(StringRef, SMLoc);
  bool ParseDirectivePrevious(StringRef, SMLoc);
  bool ParseDirectiveType(StringRef, SMLoc);
  bool ParseDirectiveIdent(StringRef, SMLoc);
  bool ParseDirectiveSymver(StringRef, SMLoc);
  bool ParseDirectiveVersion(StringRef, SMLoc);
  bool ParseDirectiveWeakref(StringRef, SMLoc);
  bool ParseDirectiveSymbolAttribute(StringRef, SMLoc);
  bool ParseDirectiveSubsection(StringRef, SMLoc);

private:
  bool ParseSectionName(StringRef &SectionName);
  bool ParseSectionArguments(bool IsPush, SMLoc loc);
  unsigned parseSunStyleSectionFlags();
};

}

/// ParseDirectiveSymbolAttribute
///  ::= { ".local", ".weak", ... } [ identifier ( , identifier )* ]
bool ELFAsmParser::ParseDirectiveSymbolAttribute(StringRef Directive, SMLoc) {
  MCSymbolAttr Attr = StringSwitch<MCSymbolAttr>(Directive)
    .Case(".weak", MCSA_Weak)
    .Case(".local", MCSA_Local)
    .Case(".hidden", MCSA_Hidden)
    .Case(".internal", MCSA_Internal)
    .Case(".protected", MCSA_Protected)
    .Default(MCSA_Invalid);
  assert(Attr != MCSA_Invalid && "unexpected symbol attribute directive!");
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    for (;;) {
      StringRef Name;

      if (getParser().parseIdentifier(Name))
        return TokError("expected identifier in directive");

      MCSymbol *Sym = getContext().getOrCreateSymbol(Name);

      getStreamer().EmitSymbolAttribute(Sym, Attr);

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

bool ELFAsmParser::ParseSectionSwitch(StringRef Section, unsigned Type,
                                      unsigned Flags, SectionKind Kind) {
  const MCExpr *Subsection = nullptr;
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    if (getParser().parseExpression(Subsection))
      return true;
  }

  getStreamer().SwitchSection(getContext().getELFSection(Section, Type, Flags),
                              Subsection);

  return false;
}

bool ELFAsmParser::ParseDirectiveSize(StringRef, SMLoc) {
  StringRef Name;
  if (getParser().parseIdentifier(Name))
    return TokError("expected identifier in directive");
  MCSymbolELF *Sym = cast<MCSymbolELF>(getContext().getOrCreateSymbol(Name));

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in directive");
  Lex();

  const MCExpr *Expr;
  if (getParser().parseExpression(Expr))
    return true;

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in directive");

  getStreamer().emitELFSize(Sym, Expr);
  return false;
}

bool ELFAsmParser::ParseSectionName(StringRef &SectionName) {
  // A section name can contain -, so we cannot just use
  // parseIdentifier.
  SMLoc FirstLoc = getLexer().getLoc();
  unsigned Size = 0;

  if (getLexer().is(AsmToken::String)) {
    SectionName = getTok().getIdentifier();
    Lex();
    return false;
  }

  for (;;) {
    unsigned CurSize;

    SMLoc PrevLoc = getLexer().getLoc();
    if (getLexer().is(AsmToken::Minus)) {
      CurSize = 1;
      Lex(); // Consume the "-".
    } else if (getLexer().is(AsmToken::String)) {
      CurSize = getTok().getIdentifier().size() + 2;
      Lex();
    } else if (getLexer().is(AsmToken::Identifier)) {
      CurSize = getTok().getIdentifier().size();
      Lex();
    } else {
      break;
    }

    Size += CurSize;
    SectionName = StringRef(FirstLoc.getPointer(), Size);

    // Make sure the following token is adjacent.
    if (PrevLoc.getPointer() + CurSize != getTok().getLoc().getPointer())
      break;
  }
  if (Size == 0)
    return true;

  return false;
}

static unsigned parseSectionFlags(StringRef flagsStr, bool *UseLastGroup) {
  unsigned flags = 0;

  for (unsigned i = 0; i < flagsStr.size(); i++) {
    switch (flagsStr[i]) {
    case 'a':
      flags |= ELF::SHF_ALLOC;
      break;
    case 'e':
      flags |= ELF::SHF_EXCLUDE;
      break;
    case 'x':
      flags |= ELF::SHF_EXECINSTR;
      break;
    case 'w':
      flags |= ELF::SHF_WRITE;
      break;
    case 'M':
      flags |= ELF::SHF_MERGE;
      break;
    case 'S':
      flags |= ELF::SHF_STRINGS;
      break;
    case 'T':
      flags |= ELF::SHF_TLS;
      break;
    case 'c':
      flags |= ELF::XCORE_SHF_CP_SECTION;
      break;
    case 'd':
      flags |= ELF::XCORE_SHF_DP_SECTION;
      break;
    case 'G':
      flags |= ELF::SHF_GROUP;
      break;
    case '?':
      *UseLastGroup = true;
      break;
    default:
      return -1U;
    }
  }

  return flags;
}

unsigned ELFAsmParser::parseSunStyleSectionFlags() {
  unsigned flags = 0;
  while (getLexer().is(AsmToken::Hash)) {
    Lex(); // Eat the #.

    if (!getLexer().is(AsmToken::Identifier))
      return -1U;

    StringRef flagId = getTok().getIdentifier();
    if (flagId == "alloc")
      flags |= ELF::SHF_ALLOC;
    else if (flagId == "execinstr")
      flags |= ELF::SHF_EXECINSTR;
    else if (flagId == "write")
      flags |= ELF::SHF_WRITE;
    else if (flagId == "tls")
      flags |= ELF::SHF_TLS;
    else
      return -1U;

    Lex(); // Eat the flag.

    if (!getLexer().is(AsmToken::Comma))
        break;
    Lex(); // Eat the comma.
  }
  return flags;
}


bool ELFAsmParser::ParseDirectivePushSection(StringRef s, SMLoc loc) {
  getStreamer().PushSection();

  if (ParseSectionArguments(/*IsPush=*/true, loc)) {
    getStreamer().PopSection();
    return true;
  }

  return false;
}

bool ELFAsmParser::ParseDirectivePopSection(StringRef, SMLoc) {
  if (!getStreamer().PopSection())
    return TokError(".popsection without corresponding .pushsection");
  return false;
}

// FIXME: This is a work in progress.
bool ELFAsmParser::ParseDirectiveSection(StringRef, SMLoc loc) {
  return ParseSectionArguments(/*IsPush=*/false, loc);
}

bool ELFAsmParser::ParseSectionArguments(bool IsPush, SMLoc loc) {
  StringRef SectionName;

  if (ParseSectionName(SectionName))
    return TokError("expected identifier in directive");

  StringRef TypeName;
  int64_t Size = 0;
  StringRef GroupName;
  unsigned Flags = 0;
  const MCExpr *Subsection = nullptr;
  bool UseLastGroup = false;
  StringRef UniqueStr;
  int64_t UniqueID = ~0;

  // Set the defaults first.
  if (SectionName == ".fini" || SectionName == ".init" ||
      SectionName == ".rodata")
    Flags |= ELF::SHF_ALLOC;
  if (SectionName == ".fini" || SectionName == ".init")
    Flags |= ELF::SHF_EXECINSTR;

  if (getLexer().is(AsmToken::Comma)) {
    Lex();

    if (IsPush && getLexer().isNot(AsmToken::String)) {
      if (getParser().parseExpression(Subsection))
        return true;
      if (getLexer().isNot(AsmToken::Comma))
        goto EndStmt;
      Lex();
    }

    unsigned extraFlags;

    if (getLexer().isNot(AsmToken::String)) {
      if (!getContext().getAsmInfo()->usesSunStyleELFSectionSwitchSyntax()
          || getLexer().isNot(AsmToken::Hash))
        return TokError("expected string in directive");
      extraFlags = parseSunStyleSectionFlags();
    } else {
      StringRef FlagsStr = getTok().getStringContents();
      Lex();
      extraFlags = parseSectionFlags(FlagsStr, &UseLastGroup);
    }

    if (extraFlags == -1U)
      return TokError("unknown flag");
    Flags |= extraFlags;

    bool Mergeable = Flags & ELF::SHF_MERGE;
    bool Group = Flags & ELF::SHF_GROUP;
    if (Group && UseLastGroup)
      return TokError("Section cannot specifiy a group name while also acting "
                      "as a member of the last group");

    if (getLexer().isNot(AsmToken::Comma)) {
      if (Mergeable)
        return TokError("Mergeable section must specify the type");
      if (Group)
        return TokError("Group section must specify the type");
    } else {
      Lex();
      if (getLexer().is(AsmToken::At) || getLexer().is(AsmToken::Percent) ||
          getLexer().is(AsmToken::String)) {
        if (!getLexer().is(AsmToken::String))
          Lex();
      } else
        return TokError("expected '@<type>', '%<type>' or \"<type>\"");

      if (getParser().parseIdentifier(TypeName))
        return TokError("expected identifier in directive");

      if (Mergeable) {
        if (getLexer().isNot(AsmToken::Comma))
          return TokError("expected the entry size");
        Lex();
        if (getParser().parseAbsoluteExpression(Size))
          return true;
        if (Size <= 0)
          return TokError("entry size must be positive");
      }

      if (Group) {
        if (getLexer().isNot(AsmToken::Comma))
          return TokError("expected group name");
        Lex();
        if (getParser().parseIdentifier(GroupName))
          return true;
        if (getLexer().is(AsmToken::Comma)) {
          Lex();
          StringRef Linkage;
          if (getParser().parseIdentifier(Linkage))
            return true;
          if (Linkage != "comdat")
            return TokError("Linkage must be 'comdat'");
        }
      }
      if (getLexer().is(AsmToken::Comma)) {
        Lex();
        if (getParser().parseIdentifier(UniqueStr))
          return TokError("expected identifier in directive");
        if (UniqueStr != "unique")
          return TokError("expected 'unique'");
        if (getLexer().isNot(AsmToken::Comma))
          return TokError("expected commma");
        Lex();
        if (getParser().parseAbsoluteExpression(UniqueID))
          return true;
        if (UniqueID < 0)
          return TokError("unique id must be positive");
        if (!isUInt<32>(UniqueID) || UniqueID == ~0U)
          return TokError("unique id is too large");
      }
    }
  }

EndStmt:
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in directive");

  unsigned Type = ELF::SHT_PROGBITS;

  if (TypeName.empty()) {
    if (SectionName.startswith(".note"))
      Type = ELF::SHT_NOTE;
    else if (SectionName == ".init_array")
      Type = ELF::SHT_INIT_ARRAY;
    else if (SectionName == ".fini_array")
      Type = ELF::SHT_FINI_ARRAY;
    else if (SectionName == ".preinit_array")
      Type = ELF::SHT_PREINIT_ARRAY;
  } else {
    if (TypeName == "init_array")
      Type = ELF::SHT_INIT_ARRAY;
    else if (TypeName == "fini_array")
      Type = ELF::SHT_FINI_ARRAY;
    else if (TypeName == "preinit_array")
      Type = ELF::SHT_PREINIT_ARRAY;
    else if (TypeName == "nobits")
      Type = ELF::SHT_NOBITS;
    else if (TypeName == "progbits")
      Type = ELF::SHT_PROGBITS;
    else if (TypeName == "note")
      Type = ELF::SHT_NOTE;
    else if (TypeName == "unwind")
      Type = ELF::SHT_X86_64_UNWIND;
    else
      return TokError("unknown section type");
  }

  if (UseLastGroup) {
    MCSectionSubPair CurrentSection = getStreamer().getCurrentSection();
    if (const MCSectionELF *Section =
            cast_or_null<MCSectionELF>(CurrentSection.first))
      if (const MCSymbol *Group = Section->getGroup()) {
        GroupName = Group->getName();
        Flags |= ELF::SHF_GROUP;
      }
  }

  MCSection *ELFSection = getContext().getELFSection(SectionName, Type, Flags,
                                                     Size, GroupName, UniqueID);
  getStreamer().SwitchSection(ELFSection, Subsection);

  if (getContext().getGenDwarfForAssembly()) {
    bool InsertResult = getContext().addGenDwarfSection(ELFSection);
    if (InsertResult) {
      if (getContext().getDwarfVersion() <= 2)
        Warning(loc, "DWARF2 only supports one section per compilation unit");

      if (!ELFSection->getBeginSymbol()) {
        MCSymbol *SectionStartSymbol = getContext().createTempSymbol();
        getStreamer().EmitLabel(SectionStartSymbol);
        ELFSection->setBeginSymbol(SectionStartSymbol);
      }
    }
  }

  return false;
}

bool ELFAsmParser::ParseDirectivePrevious(StringRef DirName, SMLoc) {
  MCSectionSubPair PreviousSection = getStreamer().getPreviousSection();
  if (PreviousSection.first == nullptr)
      return TokError(".previous without corresponding .section");
  getStreamer().SwitchSection(PreviousSection.first, PreviousSection.second);

  return false;
}

static MCSymbolAttr MCAttrForString(StringRef Type) {
  return StringSwitch<MCSymbolAttr>(Type)
          .Cases("STT_FUNC", "function", MCSA_ELF_TypeFunction)
          .Cases("STT_OBJECT", "object", MCSA_ELF_TypeObject)
          .Cases("STT_TLS", "tls_object", MCSA_ELF_TypeTLS)
          .Cases("STT_COMMON", "common", MCSA_ELF_TypeCommon)
          .Cases("STT_NOTYPE", "notype", MCSA_ELF_TypeNoType)
          .Cases("STT_GNU_IFUNC", "gnu_indirect_function",
                 MCSA_ELF_TypeIndFunction)
          .Case("gnu_unique_object", MCSA_ELF_TypeGnuUniqueObject)
          .Default(MCSA_Invalid);
}

/// ParseDirectiveELFType
///  ::= .type identifier , STT_<TYPE_IN_UPPER_CASE>
///  ::= .type identifier , #attribute
///  ::= .type identifier , @attribute
///  ::= .type identifier , %attribute
///  ::= .type identifier , "attribute"
bool ELFAsmParser::ParseDirectiveType(StringRef, SMLoc) {
  StringRef Name;
  if (getParser().parseIdentifier(Name))
    return TokError("expected identifier in directive");

  // Handle the identifier as the key symbol.
  MCSymbol *Sym = getContext().getOrCreateSymbol(Name);

  // NOTE the comma is optional in all cases.  It is only documented as being
  // optional in the first case, however, GAS will silently treat the comma as
  // optional in all cases.  Furthermore, although the documentation states that
  // the first form only accepts STT_<TYPE_IN_UPPER_CASE>, in reality, GAS
  // accepts both the upper case name as well as the lower case aliases.
  if (getLexer().is(AsmToken::Comma))
    Lex();

  if (getLexer().isNot(AsmToken::Identifier) &&
      getLexer().isNot(AsmToken::Hash) &&
      getLexer().isNot(AsmToken::Percent) &&
      getLexer().isNot(AsmToken::String)) {
    if (!getLexer().getAllowAtInIdentifier())
      return TokError("expected STT_<TYPE_IN_UPPER_CASE>, '#<type>', "
                      "'%<type>' or \"<type>\"");
    else if (getLexer().isNot(AsmToken::At))
      return TokError("expected STT_<TYPE_IN_UPPER_CASE>, '#<type>', '@<type>', "
                      "'%<type>' or \"<type>\"");
  }

  if (getLexer().isNot(AsmToken::String) &&
      getLexer().isNot(AsmToken::Identifier))
    Lex();

  SMLoc TypeLoc = getLexer().getLoc();

  StringRef Type;
  if (getParser().parseIdentifier(Type))
    return TokError("expected symbol type in directive");

  MCSymbolAttr Attr = MCAttrForString(Type);
  if (Attr == MCSA_Invalid)
    return Error(TypeLoc, "unsupported attribute in '.type' directive");

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.type' directive");
  Lex();

  getStreamer().EmitSymbolAttribute(Sym, Attr);

  return false;
}

/// ParseDirectiveIdent
///  ::= .ident string
bool ELFAsmParser::ParseDirectiveIdent(StringRef, SMLoc) {
  if (getLexer().isNot(AsmToken::String))
    return TokError("unexpected token in '.ident' directive");

  StringRef Data = getTok().getIdentifier();

  Lex();

  getStreamer().EmitIdent(Data);
  return false;
}

/// ParseDirectiveSymver
///  ::= .symver foo, bar2@zed
bool ELFAsmParser::ParseDirectiveSymver(StringRef, SMLoc) {
  StringRef Name;
  if (getParser().parseIdentifier(Name))
    return TokError("expected identifier in directive");

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("expected a comma");

  // ARM assembly uses @ for a comment...
  // except when parsing the second parameter of the .symver directive.
  // Force the next symbol to allow @ in the identifier, which is
  // required for this directive and then reset it to its initial state.
  const bool AllowAtInIdentifier = getLexer().getAllowAtInIdentifier();
  getLexer().setAllowAtInIdentifier(true);
  Lex();
  getLexer().setAllowAtInIdentifier(AllowAtInIdentifier);

  StringRef AliasName;
  if (getParser().parseIdentifier(AliasName))
    return TokError("expected identifier in directive");

  if (AliasName.find('@') == StringRef::npos)
    return TokError("expected a '@' in the name");

  MCSymbol *Alias = getContext().getOrCreateSymbol(AliasName);
  MCSymbol *Sym = getContext().getOrCreateSymbol(Name);
  const MCExpr *Value = MCSymbolRefExpr::create(Sym, getContext());

  getStreamer().EmitAssignment(Alias, Value);
  return false;
}

/// ParseDirectiveVersion
///  ::= .version string
bool ELFAsmParser::ParseDirectiveVersion(StringRef, SMLoc) {
  if (getLexer().isNot(AsmToken::String))
    return TokError("unexpected token in '.version' directive");

  StringRef Data = getTok().getIdentifier();

  Lex();

  MCSection *Note = getContext().getELFSection(".note", ELF::SHT_NOTE, 0);

  getStreamer().PushSection();
  getStreamer().SwitchSection(Note);
  getStreamer().EmitIntValue(Data.size()+1, 4); // namesz.
  getStreamer().EmitIntValue(0, 4);             // descsz = 0 (no description).
  getStreamer().EmitIntValue(1, 4);             // type = NT_VERSION.
  getStreamer().EmitBytes(Data);                // name.
  getStreamer().EmitIntValue(0, 1);             // terminate the string.
  getStreamer().EmitValueToAlignment(4);        // ensure 4 byte alignment.
  getStreamer().PopSection();
  return false;
}

/// ParseDirectiveWeakref
///  ::= .weakref foo, bar
bool ELFAsmParser::ParseDirectiveWeakref(StringRef, SMLoc) {
  // FIXME: Share code with the other alias building directives.

  StringRef AliasName;
  if (getParser().parseIdentifier(AliasName))
    return TokError("expected identifier in directive");

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("expected a comma");

  Lex();

  StringRef Name;
  if (getParser().parseIdentifier(Name))
    return TokError("expected identifier in directive");

  MCSymbol *Alias = getContext().getOrCreateSymbol(AliasName);

  MCSymbol *Sym = getContext().getOrCreateSymbol(Name);

  getStreamer().EmitWeakReference(Alias, Sym);
  return false;
}

bool ELFAsmParser::ParseDirectiveSubsection(StringRef, SMLoc) {
  const MCExpr *Subsection = nullptr;
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    if (getParser().parseExpression(Subsection))
     return true;
  }

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in directive");

  getStreamer().SubSection(Subsection);
  return false;
}

namespace llvm {

MCAsmParserExtension *createELFAsmParser() {
  return new ELFAsmParser;
}

}
