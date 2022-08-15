//===- TGParser.cpp - Parser for TableGen Files ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implement the Parser for TableGen.
//
//===----------------------------------------------------------------------===//

#include "TGParser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TableGen/Record.h"
#include <algorithm>
#include <sstream>
using namespace llvm;

//===----------------------------------------------------------------------===//
// Support Code for the Semantic Actions.
//===----------------------------------------------------------------------===//

namespace llvm {
struct SubClassReference {
  SMRange RefRange;
  Record *Rec;
  std::vector<Init*> TemplateArgs;
  SubClassReference() : Rec(nullptr) {}

  bool isInvalid() const { return Rec == nullptr; }
};

struct SubMultiClassReference {
  SMRange RefRange;
  MultiClass *MC;
  std::vector<Init*> TemplateArgs;
  SubMultiClassReference() : MC(nullptr) {}

  bool isInvalid() const { return MC == nullptr; }
  void dump() const;
};

void SubMultiClassReference::dump() const {
  errs() << "Multiclass:\n";

  MC->dump();

  errs() << "Template args:\n";
  for (Init *TA : TemplateArgs)
    TA->dump();
}

} // end namespace llvm

bool TGParser::AddValue(Record *CurRec, SMLoc Loc, const RecordVal &RV) {
  if (!CurRec)
    CurRec = &CurMultiClass->Rec;

  if (RecordVal *ERV = CurRec->getValue(RV.getNameInit())) {
    // The value already exists in the class, treat this as a set.
    if (ERV->setValue(RV.getValue()))
      return Error(Loc, "New definition of '" + RV.getName() + "' of type '" +
                   RV.getType()->getAsString() + "' is incompatible with " +
                   "previous definition of type '" +
                   ERV->getType()->getAsString() + "'");
  } else {
    CurRec->addValue(RV);
  }
  return false;
}

/// SetValue -
/// Return true on error, false on success.
bool TGParser::SetValue(Record *CurRec, SMLoc Loc, Init *ValName,
                        const std::vector<unsigned> &BitList, Init *V) {
  if (!V) return false;

  if (!CurRec) CurRec = &CurMultiClass->Rec;

  RecordVal *RV = CurRec->getValue(ValName);
  if (!RV)
    return Error(Loc, "Value '" + ValName->getAsUnquotedString() +
                 "' unknown!");

  // Do not allow assignments like 'X = X'.  This will just cause infinite loops
  // in the resolution machinery.
  if (BitList.empty())
    if (VarInit *VI = dyn_cast<VarInit>(V))
      if (VI->getNameInit() == ValName)
        return false;

  // If we are assigning to a subset of the bits in the value... then we must be
  // assigning to a field of BitsRecTy, which must have a BitsInit
  // initializer.
  //
  if (!BitList.empty()) {
    BitsInit *CurVal = dyn_cast<BitsInit>(RV->getValue());
    if (!CurVal)
      return Error(Loc, "Value '" + ValName->getAsUnquotedString() +
                   "' is not a bits type");

    // Convert the incoming value to a bits type of the appropriate size...
    Init *BI = V->convertInitializerTo(BitsRecTy::get(BitList.size()));
    if (!BI)
      return Error(Loc, "Initializer is not compatible with bit range");

    // We should have a BitsInit type now.
    BitsInit *BInit = cast<BitsInit>(BI);

    SmallVector<Init *, 16> NewBits(CurVal->getNumBits());

    // Loop over bits, assigning values as appropriate.
    for (unsigned i = 0, e = BitList.size(); i != e; ++i) {
      unsigned Bit = BitList[i];
      if (NewBits[Bit])
        return Error(Loc, "Cannot set bit #" + Twine(Bit) + " of value '" +
                     ValName->getAsUnquotedString() + "' more than once");
      NewBits[Bit] = BInit->getBit(i);
    }

    for (unsigned i = 0, e = CurVal->getNumBits(); i != e; ++i)
      if (!NewBits[i])
        NewBits[i] = CurVal->getBit(i);

    V = BitsInit::get(NewBits);
  }

  if (RV->setValue(V)) {
    std::string InitType = "";
    if (BitsInit *BI = dyn_cast<BitsInit>(V))
      InitType = (Twine("' of type bit initializer with length ") +
                  Twine(BI->getNumBits())).str();
    return Error(Loc, "Value '" + ValName->getAsUnquotedString() +
                 "' of type '" + RV->getType()->getAsString() +
                 "' is incompatible with initializer '" + V->getAsString() +
                 InitType + "'");
  }
  return false;
}

/// AddSubClass - Add SubClass as a subclass to CurRec, resolving its template
/// args as SubClass's template arguments.
bool TGParser::AddSubClass(Record *CurRec, SubClassReference &SubClass) {
  Record *SC = SubClass.Rec;
  // Add all of the values in the subclass into the current class.
  for (const RecordVal &Val : SC->getValues())
    if (AddValue(CurRec, SubClass.RefRange.Start, Val))
      return true;

  const std::vector<Init *> &TArgs = SC->getTemplateArgs();

  // Ensure that an appropriate number of template arguments are specified.
  if (TArgs.size() < SubClass.TemplateArgs.size())
    return Error(SubClass.RefRange.Start,
                 "More template args specified than expected");

  // Loop over all of the template arguments, setting them to the specified
  // value or leaving them as the default if necessary.
  for (unsigned i = 0, e = TArgs.size(); i != e; ++i) {
    if (i < SubClass.TemplateArgs.size()) {
      // If a value is specified for this template arg, set it now.
      if (SetValue(CurRec, SubClass.RefRange.Start, TArgs[i],
                   std::vector<unsigned>(), SubClass.TemplateArgs[i]))
        return true;

      // Resolve it next.
      CurRec->resolveReferencesTo(CurRec->getValue(TArgs[i]));

      // Now remove it.
      CurRec->removeValue(TArgs[i]);

    } else if (!CurRec->getValue(TArgs[i])->getValue()->isComplete()) {
      return Error(SubClass.RefRange.Start,
                   "Value not specified for template argument #" +
                   Twine(i) + " (" + TArgs[i]->getAsUnquotedString() +
                   ") of subclass '" + SC->getNameInitAsString() + "'!");
    }
  }

  // Since everything went well, we can now set the "superclass" list for the
  // current record.
  ArrayRef<Record *> SCs = SC->getSuperClasses();
  ArrayRef<SMRange> SCRanges = SC->getSuperClassRanges();
  for (unsigned i = 0, e = SCs.size(); i != e; ++i) {
    if (CurRec->isSubClassOf(SCs[i]))
      return Error(SubClass.RefRange.Start,
                   "Already subclass of '" + SCs[i]->getName() + "'!\n");
    CurRec->addSuperClass(SCs[i], SCRanges[i]);
  }

  if (CurRec->isSubClassOf(SC))
    return Error(SubClass.RefRange.Start,
                 "Already subclass of '" + SC->getName() + "'!\n");
  CurRec->addSuperClass(SC, SubClass.RefRange);
  return false;
}

/// AddSubMultiClass - Add SubMultiClass as a subclass to
/// CurMC, resolving its template args as SubMultiClass's
/// template arguments.
bool TGParser::AddSubMultiClass(MultiClass *CurMC,
                                SubMultiClassReference &SubMultiClass) {
  MultiClass *SMC = SubMultiClass.MC;
  Record *CurRec = &CurMC->Rec;

  // Add all of the values in the subclass into the current class.
  for (const auto &SMCVal : SMC->Rec.getValues())
    if (AddValue(CurRec, SubMultiClass.RefRange.Start, SMCVal))
      return true;

  unsigned newDefStart = CurMC->DefPrototypes.size();

  // Add all of the defs in the subclass into the current multiclass.
  for (const std::unique_ptr<Record> &R : SMC->DefPrototypes) {
    // Clone the def and add it to the current multiclass
    auto NewDef = make_unique<Record>(*R);

    // Add all of the values in the superclass into the current def.
    for (const auto &MCVal : CurRec->getValues())
      if (AddValue(NewDef.get(), SubMultiClass.RefRange.Start, MCVal))
        return true;

    CurMC->DefPrototypes.push_back(std::move(NewDef));
  }

  const std::vector<Init *> &SMCTArgs = SMC->Rec.getTemplateArgs();

  // Ensure that an appropriate number of template arguments are
  // specified.
  if (SMCTArgs.size() < SubMultiClass.TemplateArgs.size())
    return Error(SubMultiClass.RefRange.Start,
                 "More template args specified than expected");

  // Loop over all of the template arguments, setting them to the specified
  // value or leaving them as the default if necessary.
  for (unsigned i = 0, e = SMCTArgs.size(); i != e; ++i) {
    if (i < SubMultiClass.TemplateArgs.size()) {
      // If a value is specified for this template arg, set it in the
      // superclass now.
      if (SetValue(CurRec, SubMultiClass.RefRange.Start, SMCTArgs[i],
                   std::vector<unsigned>(),
                   SubMultiClass.TemplateArgs[i]))
        return true;

      // Resolve it next.
      CurRec->resolveReferencesTo(CurRec->getValue(SMCTArgs[i]));

      // Now remove it.
      CurRec->removeValue(SMCTArgs[i]);

      // If a value is specified for this template arg, set it in the
      // new defs now.
      for (const auto &Def :
             makeArrayRef(CurMC->DefPrototypes).slice(newDefStart)) {
        if (SetValue(Def.get(), SubMultiClass.RefRange.Start, SMCTArgs[i],
                     std::vector<unsigned>(),
                     SubMultiClass.TemplateArgs[i]))
          return true;

        // Resolve it next.
        Def->resolveReferencesTo(Def->getValue(SMCTArgs[i]));

        // Now remove it
        Def->removeValue(SMCTArgs[i]);
      }
    } else if (!CurRec->getValue(SMCTArgs[i])->getValue()->isComplete()) {
      return Error(SubMultiClass.RefRange.Start,
                   "Value not specified for template argument #" +
                   Twine(i) + " (" + SMCTArgs[i]->getAsUnquotedString() +
                   ") of subclass '" + SMC->Rec.getNameInitAsString() + "'!");
    }
  }

  return false;
}

/// ProcessForeachDefs - Given a record, apply all of the variable
/// values in all surrounding foreach loops, creating new records for
/// each combination of values.
bool TGParser::ProcessForeachDefs(Record *CurRec, SMLoc Loc) {
  if (Loops.empty())
    return false;

  // We want to instantiate a new copy of CurRec for each combination
  // of nested loop iterator values.  We don't want top instantiate
  // any copies until we have values for each loop iterator.
  IterSet IterVals;
  return ProcessForeachDefs(CurRec, Loc, IterVals);
}

/// ProcessForeachDefs - Given a record, a loop and a loop iterator,
/// apply each of the variable values in this loop and then process
/// subloops.
bool TGParser::ProcessForeachDefs(Record *CurRec, SMLoc Loc, IterSet &IterVals){
  // Recursively build a tuple of iterator values.
  if (IterVals.size() != Loops.size()) {
    assert(IterVals.size() < Loops.size());
    ForeachLoop &CurLoop = Loops[IterVals.size()];
    ListInit *List = dyn_cast<ListInit>(CurLoop.ListValue);
    if (!List) {
      Error(Loc, "Loop list is not a list");
      return true;
    }

    // Process each value.
    for (unsigned i = 0; i < List->size(); ++i) {
      Init *ItemVal = List->resolveListElementReference(*CurRec, nullptr, i);
      IterVals.push_back(IterRecord(CurLoop.IterVar, ItemVal));
      if (ProcessForeachDefs(CurRec, Loc, IterVals))
        return true;
      IterVals.pop_back();
    }
    return false;
  }

  // This is the bottom of the recursion. We have all of the iterator values
  // for this point in the iteration space.  Instantiate a new record to
  // reflect this combination of values.
  auto IterRec = make_unique<Record>(*CurRec);

  // Set the iterator values now.
  for (IterRecord &IR : IterVals) {
    VarInit *IterVar = IR.IterVar;
    TypedInit *IVal = dyn_cast<TypedInit>(IR.IterValue);
    if (!IVal)
      return Error(Loc, "foreach iterator value is untyped");

    IterRec->addValue(RecordVal(IterVar->getName(), IVal->getType(), false));

    if (SetValue(IterRec.get(), Loc, IterVar->getName(),
                 std::vector<unsigned>(), IVal))
      return Error(Loc, "when instantiating this def");

    // Resolve it next.
    IterRec->resolveReferencesTo(IterRec->getValue(IterVar->getName()));

    // Remove it.
    IterRec->removeValue(IterVar->getName());
  }

  if (Records.getDef(IterRec->getNameInitAsString())) {
    // If this record is anonymous, it's no problem, just generate a new name
    if (!IterRec->isAnonymous())
      return Error(Loc, "def already exists: " +IterRec->getNameInitAsString());

    IterRec->setName(GetNewAnonymousName());
  }

  Record *IterRecSave = IterRec.get(); // Keep a copy before release.
  Records.addDef(std::move(IterRec));
  IterRecSave->resolveReferences();
  return false;
}

//===----------------------------------------------------------------------===//
// Parser Code
//===----------------------------------------------------------------------===//

/// isObjectStart - Return true if this is a valid first token for an Object.
static bool isObjectStart(tgtok::TokKind K) {
  return K == tgtok::Class || K == tgtok::Def ||
         K == tgtok::Defm || K == tgtok::Let ||
         K == tgtok::MultiClass || K == tgtok::Foreach;
}

/// GetNewAnonymousName - Generate a unique anonymous name that can be used as
/// an identifier.
std::string TGParser::GetNewAnonymousName() {
  return "anonymous_" + utostr(AnonCounter++);
}

/// ParseObjectName - If an object name is specified, return it.  Otherwise,
/// return 0.
///   ObjectName ::= Value [ '#' Value ]*
///   ObjectName ::= /*empty*/
///
Init *TGParser::ParseObjectName(MultiClass *CurMultiClass) {
  switch (Lex.getCode()) {
  case tgtok::colon:
  case tgtok::semi:
  case tgtok::l_brace:
    // These are all of the tokens that can begin an object body.
    // Some of these can also begin values but we disallow those cases
    // because they are unlikely to be useful.
    return nullptr;
  default:
    break;
  }

  Record *CurRec = nullptr;
  if (CurMultiClass)
    CurRec = &CurMultiClass->Rec;

  RecTy *Type = nullptr;
  if (CurRec) {
    const TypedInit *CurRecName = dyn_cast<TypedInit>(CurRec->getNameInit());
    if (!CurRecName) {
      TokError("Record name is not typed!");
      return nullptr;
    }
    Type = CurRecName->getType();
  }

  return ParseValue(CurRec, Type, ParseNameMode);
}

/// ParseClassID - Parse and resolve a reference to a class name.  This returns
/// null on error.
///
///    ClassID ::= ID
///
Record *TGParser::ParseClassID() {
  if (Lex.getCode() != tgtok::Id) {
    TokError("expected name for ClassID");
    return nullptr;
  }

  Record *Result = Records.getClass(Lex.getCurStrVal());
  if (!Result)
    TokError("Couldn't find class '" + Lex.getCurStrVal() + "'");

  Lex.Lex();
  return Result;
}

/// ParseMultiClassID - Parse and resolve a reference to a multiclass name.
/// This returns null on error.
///
///    MultiClassID ::= ID
///
MultiClass *TGParser::ParseMultiClassID() {
  if (Lex.getCode() != tgtok::Id) {
    TokError("expected name for MultiClassID");
    return nullptr;
  }

  MultiClass *Result = MultiClasses[Lex.getCurStrVal()].get();
  if (!Result)
    TokError("Couldn't find multiclass '" + Lex.getCurStrVal() + "'");

  Lex.Lex();
  return Result;
}

/// ParseSubClassReference - Parse a reference to a subclass or to a templated
/// subclass.  This returns a SubClassRefTy with a null Record* on error.
///
///  SubClassRef ::= ClassID
///  SubClassRef ::= ClassID '<' ValueList '>'
///
SubClassReference TGParser::
ParseSubClassReference(Record *CurRec, bool isDefm) {
  SubClassReference Result;
  Result.RefRange.Start = Lex.getLoc();

  if (isDefm) {
    if (MultiClass *MC = ParseMultiClassID())
      Result.Rec = &MC->Rec;
  } else {
    Result.Rec = ParseClassID();
  }
  if (!Result.Rec) return Result;

  // If there is no template arg list, we're done.
  if (Lex.getCode() != tgtok::less) {
    Result.RefRange.End = Lex.getLoc();
    return Result;
  }
  Lex.Lex();  // Eat the '<'

  if (Lex.getCode() == tgtok::greater) {
    TokError("subclass reference requires a non-empty list of template values");
    Result.Rec = nullptr;
    return Result;
  }

  Result.TemplateArgs = ParseValueList(CurRec, Result.Rec);
  if (Result.TemplateArgs.empty()) {
    Result.Rec = nullptr;   // Error parsing value list.
    return Result;
  }

  if (Lex.getCode() != tgtok::greater) {
    TokError("expected '>' in template value list");
    Result.Rec = nullptr;
    return Result;
  }
  Lex.Lex();
  Result.RefRange.End = Lex.getLoc();

  return Result;
}

/// ParseSubMultiClassReference - Parse a reference to a subclass or to a
/// templated submulticlass.  This returns a SubMultiClassRefTy with a null
/// Record* on error.
///
///  SubMultiClassRef ::= MultiClassID
///  SubMultiClassRef ::= MultiClassID '<' ValueList '>'
///
SubMultiClassReference TGParser::
ParseSubMultiClassReference(MultiClass *CurMC) {
  SubMultiClassReference Result;
  Result.RefRange.Start = Lex.getLoc();

  Result.MC = ParseMultiClassID();
  if (!Result.MC) return Result;

  // If there is no template arg list, we're done.
  if (Lex.getCode() != tgtok::less) {
    Result.RefRange.End = Lex.getLoc();
    return Result;
  }
  Lex.Lex();  // Eat the '<'

  if (Lex.getCode() == tgtok::greater) {
    TokError("subclass reference requires a non-empty list of template values");
    Result.MC = nullptr;
    return Result;
  }

  Result.TemplateArgs = ParseValueList(&CurMC->Rec, &Result.MC->Rec);
  if (Result.TemplateArgs.empty()) {
    Result.MC = nullptr;   // Error parsing value list.
    return Result;
  }

  if (Lex.getCode() != tgtok::greater) {
    TokError("expected '>' in template value list");
    Result.MC = nullptr;
    return Result;
  }
  Lex.Lex();
  Result.RefRange.End = Lex.getLoc();

  return Result;
}

/// ParseRangePiece - Parse a bit/value range.
///   RangePiece ::= INTVAL
///   RangePiece ::= INTVAL '-' INTVAL
///   RangePiece ::= INTVAL INTVAL
bool TGParser::ParseRangePiece(std::vector<unsigned> &Ranges) {
  if (Lex.getCode() != tgtok::IntVal) {
    TokError("expected integer or bitrange");
    return true;
  }
  int64_t Start = Lex.getCurIntVal();
  int64_t End;

  if (Start < 0)
    return TokError("invalid range, cannot be negative");

  switch (Lex.Lex()) {  // eat first character.
  default:
    Ranges.push_back(Start);
    return false;
  case tgtok::minus:
    if (Lex.Lex() != tgtok::IntVal) {
      TokError("expected integer value as end of range");
      return true;
    }
    End = Lex.getCurIntVal();
    break;
  case tgtok::IntVal:
    End = -Lex.getCurIntVal();
    break;
  }
  if (End < 0)
    return TokError("invalid range, cannot be negative");
  Lex.Lex();

  // Add to the range.
  if (Start < End)
    for (; Start <= End; ++Start)
      Ranges.push_back(Start);
  else
    for (; Start >= End; --Start)
      Ranges.push_back(Start);
  return false;
}

/// ParseRangeList - Parse a list of scalars and ranges into scalar values.
///
///   RangeList ::= RangePiece (',' RangePiece)*
///
std::vector<unsigned> TGParser::ParseRangeList() {
  std::vector<unsigned> Result;

  // Parse the first piece.
  if (ParseRangePiece(Result))
    return std::vector<unsigned>();
  while (Lex.getCode() == tgtok::comma) {
    Lex.Lex();  // Eat the comma.

    // Parse the next range piece.
    if (ParseRangePiece(Result))
      return std::vector<unsigned>();
  }
  return Result;
}

/// ParseOptionalRangeList - Parse either a range list in <>'s or nothing.
///   OptionalRangeList ::= '<' RangeList '>'
///   OptionalRangeList ::= /*empty*/
bool TGParser::ParseOptionalRangeList(std::vector<unsigned> &Ranges) {
  if (Lex.getCode() != tgtok::less)
    return false;

  SMLoc StartLoc = Lex.getLoc();
  Lex.Lex(); // eat the '<'

  // Parse the range list.
  Ranges = ParseRangeList();
  if (Ranges.empty()) return true;

  if (Lex.getCode() != tgtok::greater) {
    TokError("expected '>' at end of range list");
    return Error(StartLoc, "to match this '<'");
  }
  Lex.Lex();   // eat the '>'.
  return false;
}

/// ParseOptionalBitList - Parse either a bit list in {}'s or nothing.
///   OptionalBitList ::= '{' RangeList '}'
///   OptionalBitList ::= /*empty*/
bool TGParser::ParseOptionalBitList(std::vector<unsigned> &Ranges) {
  if (Lex.getCode() != tgtok::l_brace)
    return false;

  SMLoc StartLoc = Lex.getLoc();
  Lex.Lex(); // eat the '{'

  // Parse the range list.
  Ranges = ParseRangeList();
  if (Ranges.empty()) return true;

  if (Lex.getCode() != tgtok::r_brace) {
    TokError("expected '}' at end of bit list");
    return Error(StartLoc, "to match this '{'");
  }
  Lex.Lex();   // eat the '}'.
  return false;
}


/// ParseType - Parse and return a tblgen type.  This returns null on error.
///
///   Type ::= STRING                       // string type
///   Type ::= CODE                         // code type
///   Type ::= BIT                          // bit type
///   Type ::= BITS '<' INTVAL '>'          // bits<x> type
///   Type ::= INT                          // int type
///   Type ::= LIST '<' Type '>'            // list<x> type
///   Type ::= DAG                          // dag type
///   Type ::= ClassID                      // Record Type
///
RecTy *TGParser::ParseType() {
  switch (Lex.getCode()) {
  default: TokError("Unknown token when expecting a type"); return nullptr;
  case tgtok::String: Lex.Lex(); return StringRecTy::get();
  case tgtok::Code:   Lex.Lex(); return StringRecTy::get();
  case tgtok::Bit:    Lex.Lex(); return BitRecTy::get();
  case tgtok::Int:    Lex.Lex(); return IntRecTy::get();
  case tgtok::Dag:    Lex.Lex(); return DagRecTy::get();
  case tgtok::Id:
    if (Record *R = ParseClassID()) return RecordRecTy::get(R);
    return nullptr;
  case tgtok::Bits: {
    if (Lex.Lex() != tgtok::less) { // Eat 'bits'
      TokError("expected '<' after bits type");
      return nullptr;
    }
    if (Lex.Lex() != tgtok::IntVal) {  // Eat '<'
      TokError("expected integer in bits<n> type");
      return nullptr;
    }
    uint64_t Val = Lex.getCurIntVal();
    if (Lex.Lex() != tgtok::greater) {  // Eat count.
      TokError("expected '>' at end of bits<n> type");
      return nullptr;
    }
    Lex.Lex();  // Eat '>'
    return BitsRecTy::get(Val);
  }
  case tgtok::List: {
    if (Lex.Lex() != tgtok::less) { // Eat 'bits'
      TokError("expected '<' after list type");
      return nullptr;
    }
    Lex.Lex();  // Eat '<'
    RecTy *SubType = ParseType();
    if (!SubType) return nullptr;

    if (Lex.getCode() != tgtok::greater) {
      TokError("expected '>' at end of list<ty> type");
      return nullptr;
    }
    Lex.Lex();  // Eat '>'
    return ListRecTy::get(SubType);
  }
  }
}

/// ParseIDValue - This is just like ParseIDValue above, but it assumes the ID
/// has already been read.
Init *TGParser::ParseIDValue(Record *CurRec,
                             const std::string &Name, SMLoc NameLoc,
                             IDParseMode Mode) {
  if (CurRec) {
    if (const RecordVal *RV = CurRec->getValue(Name))
      return VarInit::get(Name, RV->getType());

    Init *TemplateArgName = QualifyName(*CurRec, CurMultiClass, Name, ":");

    if (CurMultiClass)
      TemplateArgName = QualifyName(CurMultiClass->Rec, CurMultiClass, Name,
                                    "::");

    if (CurRec->isTemplateArg(TemplateArgName)) {
      const RecordVal *RV = CurRec->getValue(TemplateArgName);
      assert(RV && "Template arg doesn't exist??");
      return VarInit::get(TemplateArgName, RV->getType());
    }
  }

  if (CurMultiClass) {
    Init *MCName = QualifyName(CurMultiClass->Rec, CurMultiClass, Name,
                               "::");

    if (CurMultiClass->Rec.isTemplateArg(MCName)) {
      const RecordVal *RV = CurMultiClass->Rec.getValue(MCName);
      assert(RV && "Template arg doesn't exist??");
      return VarInit::get(MCName, RV->getType());
    }
  }

  // If this is in a foreach loop, make sure it's not a loop iterator
  for (const auto &L : Loops) {
    VarInit *IterVar = dyn_cast<VarInit>(L.IterVar);
    if (IterVar && IterVar->getName() == Name)
      return IterVar;
  }

  if (Mode == ParseNameMode)
    return StringInit::get(Name);

  if (Record *D = Records.getDef(Name))
    return DefInit::get(D);

  if (Mode == ParseValueMode) {
    Error(NameLoc, "Variable not defined: '" + Name + "'");
    return nullptr;
  }

  return StringInit::get(Name);
}

/// ParseOperation - Parse an operator.  This returns null on error.
///
/// Operation ::= XOperator ['<' Type '>'] '(' Args ')'
///
Init *TGParser::ParseOperation(Record *CurRec, RecTy *ItemType) {
  switch (Lex.getCode()) {
  default:
    TokError("unknown operation");
    return nullptr;
  case tgtok::XHead:
  case tgtok::XTail:
  case tgtok::XEmpty:
  case tgtok::XCast: {  // Value ::= !unop '(' Value ')'
    UnOpInit::UnaryOp Code;
    RecTy *Type = nullptr;

    switch (Lex.getCode()) {
    default: llvm_unreachable("Unhandled code!");
    case tgtok::XCast:
      Lex.Lex();  // eat the operation
      Code = UnOpInit::CAST;

      Type = ParseOperatorType();

      if (!Type) {
        TokError("did not get type for unary operator");
        return nullptr;
      }

      break;
    case tgtok::XHead:
      Lex.Lex();  // eat the operation
      Code = UnOpInit::HEAD;
      break;
    case tgtok::XTail:
      Lex.Lex();  // eat the operation
      Code = UnOpInit::TAIL;
      break;
    case tgtok::XEmpty:
      Lex.Lex();  // eat the operation
      Code = UnOpInit::EMPTY;
      Type = IntRecTy::get();
      break;
    }
    if (Lex.getCode() != tgtok::l_paren) {
      TokError("expected '(' after unary operator");
      return nullptr;
    }
    Lex.Lex();  // eat the '('

    Init *LHS = ParseValue(CurRec);
    if (!LHS) return nullptr;

    if (Code == UnOpInit::HEAD ||
        Code == UnOpInit::TAIL ||
        Code == UnOpInit::EMPTY) {
      ListInit *LHSl = dyn_cast<ListInit>(LHS);
      StringInit *LHSs = dyn_cast<StringInit>(LHS);
      TypedInit *LHSt = dyn_cast<TypedInit>(LHS);
      if (!LHSl && !LHSs && !LHSt) {
        TokError("expected list or string type argument in unary operator");
        return nullptr;
      }
      if (LHSt) {
        ListRecTy *LType = dyn_cast<ListRecTy>(LHSt->getType());
        StringRecTy *SType = dyn_cast<StringRecTy>(LHSt->getType());
        if (!LType && !SType) {
          TokError("expected list or string type argument in unary operator");
          return nullptr;
        }
      }

      if (Code == UnOpInit::HEAD || Code == UnOpInit::TAIL) {
        if (!LHSl && !LHSt) {
          TokError("expected list type argument in unary operator");
          return nullptr;
        }

        if (LHSl && LHSl->empty()) {
          TokError("empty list argument in unary operator");
          return nullptr;
        }
        if (LHSl) {
          Init *Item = LHSl->getElement(0);
          TypedInit *Itemt = dyn_cast<TypedInit>(Item);
          if (!Itemt) {
            TokError("untyped list element in unary operator");
            return nullptr;
          }
          Type = (Code == UnOpInit::HEAD) ? Itemt->getType()
                                          : ListRecTy::get(Itemt->getType());
        } else {
          assert(LHSt && "expected list type argument in unary operator");
          ListRecTy *LType = dyn_cast<ListRecTy>(LHSt->getType());
          if (!LType) {
            TokError("expected list type argument in unary operator");
            return nullptr;
          }
          Type = (Code == UnOpInit::HEAD) ? LType->getElementType() : LType;
        }
      }
    }

    if (Lex.getCode() != tgtok::r_paren) {
      TokError("expected ')' in unary operator");
      return nullptr;
    }
    Lex.Lex();  // eat the ')'
    return (UnOpInit::get(Code, LHS, Type))->Fold(CurRec, CurMultiClass);
  }

  case tgtok::XConcat:
  case tgtok::XADD:
  case tgtok::XAND:
  case tgtok::XSRA:
  case tgtok::XSRL:
  case tgtok::XSHL:
  case tgtok::XEq:
  case tgtok::XListConcat:
  case tgtok::XStrConcat: {  // Value ::= !binop '(' Value ',' Value ')'
    tgtok::TokKind OpTok = Lex.getCode();
    SMLoc OpLoc = Lex.getLoc();
    Lex.Lex();  // eat the operation

    BinOpInit::BinaryOp Code;
    RecTy *Type = nullptr;

    switch (OpTok) {
    default: llvm_unreachable("Unhandled code!");
    case tgtok::XConcat: Code = BinOpInit::CONCAT;Type = DagRecTy::get(); break;
    case tgtok::XADD:    Code = BinOpInit::ADD;   Type = IntRecTy::get(); break;
    case tgtok::XAND:    Code = BinOpInit::AND;   Type = IntRecTy::get(); break;
    case tgtok::XSRA:    Code = BinOpInit::SRA;   Type = IntRecTy::get(); break;
    case tgtok::XSRL:    Code = BinOpInit::SRL;   Type = IntRecTy::get(); break;
    case tgtok::XSHL:    Code = BinOpInit::SHL;   Type = IntRecTy::get(); break;
    case tgtok::XEq:     Code = BinOpInit::EQ;    Type = BitRecTy::get(); break;
    case tgtok::XListConcat:
      Code = BinOpInit::LISTCONCAT;
      // We don't know the list type until we parse the first argument
      break;
    case tgtok::XStrConcat:
      Code = BinOpInit::STRCONCAT;
      Type = StringRecTy::get();
      break;
    }

    if (Lex.getCode() != tgtok::l_paren) {
      TokError("expected '(' after binary operator");
      return nullptr;
    }
    Lex.Lex();  // eat the '('

    SmallVector<Init*, 2> InitList;

    InitList.push_back(ParseValue(CurRec));
    if (!InitList.back()) return nullptr;

    while (Lex.getCode() == tgtok::comma) {
      Lex.Lex();  // eat the ','

      InitList.push_back(ParseValue(CurRec));
      if (!InitList.back()) return nullptr;
    }

    if (Lex.getCode() != tgtok::r_paren) {
      TokError("expected ')' in operator");
      return nullptr;
    }
    Lex.Lex();  // eat the ')'

    // If we are doing !listconcat, we should know the type by now
    if (OpTok == tgtok::XListConcat) {
      if (VarInit *Arg0 = dyn_cast<VarInit>(InitList[0]))
        Type = Arg0->getType();
      else if (ListInit *Arg0 = dyn_cast<ListInit>(InitList[0]))
        Type = Arg0->getType();
      else {
        InitList[0]->dump();
        Error(OpLoc, "expected a list");
        return nullptr;
      }
    }

    // We allow multiple operands to associative operators like !strconcat as
    // shorthand for nesting them.
    if (Code == BinOpInit::STRCONCAT || Code == BinOpInit::LISTCONCAT) {
      while (InitList.size() > 2) {
        Init *RHS = InitList.pop_back_val();
        RHS = (BinOpInit::get(Code, InitList.back(), RHS, Type))
                           ->Fold(CurRec, CurMultiClass);
        InitList.back() = RHS;
      }
    }

    if (InitList.size() == 2)
      return (BinOpInit::get(Code, InitList[0], InitList[1], Type))
        ->Fold(CurRec, CurMultiClass);

    Error(OpLoc, "expected two operands to operator");
    return nullptr;
  }

  case tgtok::XIf:
  case tgtok::XForEach:
  case tgtok::XSubst: {  // Value ::= !ternop '(' Value ',' Value ',' Value ')'
    TernOpInit::TernaryOp Code;
    RecTy *Type = nullptr;

    tgtok::TokKind LexCode = Lex.getCode();
    Lex.Lex();  // eat the operation
    switch (LexCode) {
    default: llvm_unreachable("Unhandled code!");
    case tgtok::XIf:
      Code = TernOpInit::IF;
      break;
    case tgtok::XForEach:
      Code = TernOpInit::FOREACH;
      break;
    case tgtok::XSubst:
      Code = TernOpInit::SUBST;
      break;
    }
    if (Lex.getCode() != tgtok::l_paren) {
      TokError("expected '(' after ternary operator");
      return nullptr;
    }
    Lex.Lex();  // eat the '('

    Init *LHS = ParseValue(CurRec);
    if (!LHS) return nullptr;

    if (Lex.getCode() != tgtok::comma) {
      TokError("expected ',' in ternary operator");
      return nullptr;
    }
    Lex.Lex();  // eat the ','

    Init *MHS = ParseValue(CurRec, ItemType);
    if (!MHS)
      return nullptr;

    if (Lex.getCode() != tgtok::comma) {
      TokError("expected ',' in ternary operator");
      return nullptr;
    }
    Lex.Lex();  // eat the ','

    Init *RHS = ParseValue(CurRec, ItemType);
    if (!RHS)
      return nullptr;

    if (Lex.getCode() != tgtok::r_paren) {
      TokError("expected ')' in binary operator");
      return nullptr;
    }
    Lex.Lex();  // eat the ')'

    switch (LexCode) {
    default: llvm_unreachable("Unhandled code!");
    case tgtok::XIf: {
      RecTy *MHSTy = nullptr;
      RecTy *RHSTy = nullptr;

      if (TypedInit *MHSt = dyn_cast<TypedInit>(MHS))
        MHSTy = MHSt->getType();
      if (BitsInit *MHSbits = dyn_cast<BitsInit>(MHS))
        MHSTy = BitsRecTy::get(MHSbits->getNumBits());
      if (isa<BitInit>(MHS))
        MHSTy = BitRecTy::get();

      if (TypedInit *RHSt = dyn_cast<TypedInit>(RHS))
        RHSTy = RHSt->getType();
      if (BitsInit *RHSbits = dyn_cast<BitsInit>(RHS))
        RHSTy = BitsRecTy::get(RHSbits->getNumBits());
      if (isa<BitInit>(RHS))
        RHSTy = BitRecTy::get();

      // For UnsetInit, it's typed from the other hand.
      if (isa<UnsetInit>(MHS))
        MHSTy = RHSTy;
      if (isa<UnsetInit>(RHS))
        RHSTy = MHSTy;

      if (!MHSTy || !RHSTy) {
        TokError("could not get type for !if");
        return nullptr;
      }

      if (MHSTy->typeIsConvertibleTo(RHSTy)) {
        Type = RHSTy;
      } else if (RHSTy->typeIsConvertibleTo(MHSTy)) {
        Type = MHSTy;
      } else {
        TokError("inconsistent types for !if");
        return nullptr;
      }
      break;
    }
    case tgtok::XForEach: {
      TypedInit *MHSt = dyn_cast<TypedInit>(MHS);
      if (!MHSt) {
        TokError("could not get type for !foreach");
        return nullptr;
      }
      Type = MHSt->getType();
      break;
    }
    case tgtok::XSubst: {
      TypedInit *RHSt = dyn_cast<TypedInit>(RHS);
      if (!RHSt) {
        TokError("could not get type for !subst");
        return nullptr;
      }
      Type = RHSt->getType();
      break;
    }
    }
    return (TernOpInit::get(Code, LHS, MHS, RHS, Type))->Fold(CurRec,
                                                             CurMultiClass);
  }
  }
}

/// ParseOperatorType - Parse a type for an operator.  This returns
/// null on error.
///
/// OperatorType ::= '<' Type '>'
///
RecTy *TGParser::ParseOperatorType() {
  RecTy *Type = nullptr;

  if (Lex.getCode() != tgtok::less) {
    TokError("expected type name for operator");
    return nullptr;
  }
  Lex.Lex();  // eat the <

  Type = ParseType();

  if (!Type) {
    TokError("expected type name for operator");
    return nullptr;
  }

  if (Lex.getCode() != tgtok::greater) {
    TokError("expected type name for operator");
    return nullptr;
  }
  Lex.Lex();  // eat the >

  return Type;
}


/// ParseSimpleValue - Parse a tblgen value.  This returns null on error.
///
///   SimpleValue ::= IDValue
///   SimpleValue ::= INTVAL
///   SimpleValue ::= STRVAL+
///   SimpleValue ::= CODEFRAGMENT
///   SimpleValue ::= '?'
///   SimpleValue ::= '{' ValueList '}'
///   SimpleValue ::= ID '<' ValueListNE '>'
///   SimpleValue ::= '[' ValueList ']'
///   SimpleValue ::= '(' IDValue DagArgList ')'
///   SimpleValue ::= CONCATTOK '(' Value ',' Value ')'
///   SimpleValue ::= ADDTOK '(' Value ',' Value ')'
///   SimpleValue ::= SHLTOK '(' Value ',' Value ')'
///   SimpleValue ::= SRATOK '(' Value ',' Value ')'
///   SimpleValue ::= SRLTOK '(' Value ',' Value ')'
///   SimpleValue ::= LISTCONCATTOK '(' Value ',' Value ')'
///   SimpleValue ::= STRCONCATTOK '(' Value ',' Value ')'
///
Init *TGParser::ParseSimpleValue(Record *CurRec, RecTy *ItemType,
                                 IDParseMode Mode) {
  Init *R = nullptr;
  switch (Lex.getCode()) {
  default: TokError("Unknown token when parsing a value"); break;
  case tgtok::paste:
    // This is a leading paste operation.  This is deprecated but
    // still exists in some .td files.  Ignore it.
    Lex.Lex();  // Skip '#'.
    return ParseSimpleValue(CurRec, ItemType, Mode);
  case tgtok::IntVal: R = IntInit::get(Lex.getCurIntVal()); Lex.Lex(); break;
  case tgtok::BinaryIntVal: {
    auto BinaryVal = Lex.getCurBinaryIntVal();
    SmallVector<Init*, 16> Bits(BinaryVal.second);
    for (unsigned i = 0, e = BinaryVal.second; i != e; ++i)
      Bits[i] = BitInit::get(BinaryVal.first & (1LL << i));
    R = BitsInit::get(Bits);
    Lex.Lex();
    break;
  }
  case tgtok::StrVal: {
    std::string Val = Lex.getCurStrVal();
    Lex.Lex();

    // Handle multiple consecutive concatenated strings.
    while (Lex.getCode() == tgtok::StrVal) {
      Val += Lex.getCurStrVal();
      Lex.Lex();
    }

    R = StringInit::get(Val);
    break;
  }
  case tgtok::CodeFragment:
    R = StringInit::get(Lex.getCurStrVal());
    Lex.Lex();
    break;
  case tgtok::question:
    R = UnsetInit::get();
    Lex.Lex();
    break;
  case tgtok::Id: {
    SMLoc NameLoc = Lex.getLoc();
    std::string Name = Lex.getCurStrVal();
    if (Lex.Lex() != tgtok::less)  // consume the Id.
      return ParseIDValue(CurRec, Name, NameLoc, Mode);    // Value ::= IDValue

    // Value ::= ID '<' ValueListNE '>'
    if (Lex.Lex() == tgtok::greater) {
      TokError("expected non-empty value list");
      return nullptr;
    }

    // This is a CLASS<initvalslist> expression.  This is supposed to synthesize
    // a new anonymous definition, deriving from CLASS<initvalslist> with no
    // body.
    Record *Class = Records.getClass(Name);
    if (!Class) {
      Error(NameLoc, "Expected a class name, got '" + Name + "'");
      return nullptr;
    }

    std::vector<Init*> ValueList = ParseValueList(CurRec, Class);
    if (ValueList.empty()) return nullptr;

    if (Lex.getCode() != tgtok::greater) {
      TokError("expected '>' at end of value list");
      return nullptr;
    }
    Lex.Lex();  // eat the '>'
    SMLoc EndLoc = Lex.getLoc();

    // Create the new record, set it as CurRec temporarily.
    auto NewRecOwner = llvm::make_unique<Record>(GetNewAnonymousName(), NameLoc,
                                                 Records, /*IsAnonymous=*/true);
    Record *NewRec = NewRecOwner.get(); // Keep a copy since we may release.
    SubClassReference SCRef;
    SCRef.RefRange = SMRange(NameLoc, EndLoc);
    SCRef.Rec = Class;
    SCRef.TemplateArgs = ValueList;
    // Add info about the subclass to NewRec.
    if (AddSubClass(NewRec, SCRef))
      return nullptr;

    if (!CurMultiClass) {
      NewRec->resolveReferences();
      Records.addDef(std::move(NewRecOwner));
    } else {
      // This needs to get resolved once the multiclass template arguments are
      // known before any use.
      NewRec->setResolveFirst(true);
      // Otherwise, we're inside a multiclass, add it to the multiclass.
      CurMultiClass->DefPrototypes.push_back(std::move(NewRecOwner));

      // Copy the template arguments for the multiclass into the def.
      for (Init *TArg : CurMultiClass->Rec.getTemplateArgs()) {
        const RecordVal *RV = CurMultiClass->Rec.getValue(TArg);
        assert(RV && "Template arg doesn't exist?");
        NewRec->addValue(*RV);
      }

      // We can't return the prototype def here, instead return:
      // !cast<ItemType>(!strconcat(NAME, AnonName)).
      const RecordVal *MCNameRV = CurMultiClass->Rec.getValue("NAME");
      assert(MCNameRV && "multiclass record must have a NAME");

      return UnOpInit::get(UnOpInit::CAST,
                           BinOpInit::get(BinOpInit::STRCONCAT,
                                          VarInit::get(MCNameRV->getName(),
                                                       MCNameRV->getType()),
                                          NewRec->getNameInit(),
                                          StringRecTy::get()),
                           Class->getDefInit()->getType());
    }

    // The result of the expression is a reference to the new record.
    return DefInit::get(NewRec);
  }
  case tgtok::l_brace: {           // Value ::= '{' ValueList '}'
    SMLoc BraceLoc = Lex.getLoc();
    Lex.Lex(); // eat the '{'
    std::vector<Init*> Vals;

    if (Lex.getCode() != tgtok::r_brace) {
      Vals = ParseValueList(CurRec);
      if (Vals.empty()) return nullptr;
    }
    if (Lex.getCode() != tgtok::r_brace) {
      TokError("expected '}' at end of bit list value");
      return nullptr;
    }
    Lex.Lex();  // eat the '}'

    SmallVector<Init *, 16> NewBits;

    // As we parse { a, b, ... }, 'a' is the highest bit, but we parse it
    // first.  We'll first read everything in to a vector, then we can reverse
    // it to get the bits in the correct order for the BitsInit value.
    for (unsigned i = 0, e = Vals.size(); i != e; ++i) {
      // FIXME: The following two loops would not be duplicated
      //        if the API was a little more orthogonal.

      // bits<n> values are allowed to initialize n bits.
      if (BitsInit *BI = dyn_cast<BitsInit>(Vals[i])) {
        for (unsigned i = 0, e = BI->getNumBits(); i != e; ++i)
          NewBits.push_back(BI->getBit((e - i) - 1));
        continue;
      }
      // bits<n> can also come from variable initializers.
      if (VarInit *VI = dyn_cast<VarInit>(Vals[i])) {
        if (BitsRecTy *BitsRec = dyn_cast<BitsRecTy>(VI->getType())) {
          for (unsigned i = 0, e = BitsRec->getNumBits(); i != e; ++i)
            NewBits.push_back(VI->getBit((e - i) - 1));
          continue;
        }
        // Fallthrough to try convert this to a bit.
      }
      // All other values must be convertible to just a single bit.
      Init *Bit = Vals[i]->convertInitializerTo(BitRecTy::get());
      if (!Bit) {
        Error(BraceLoc, "Element #" + Twine(i) + " (" + Vals[i]->getAsString() +
              ") is not convertable to a bit");
        return nullptr;
      }
      NewBits.push_back(Bit);
    }
    std::reverse(NewBits.begin(), NewBits.end());
    return BitsInit::get(NewBits);
  }
  case tgtok::l_square: {          // Value ::= '[' ValueList ']'
    Lex.Lex(); // eat the '['
    std::vector<Init*> Vals;

    RecTy *DeducedEltTy = nullptr;
    ListRecTy *GivenListTy = nullptr;

    if (ItemType) {
      ListRecTy *ListType = dyn_cast<ListRecTy>(ItemType);
      if (!ListType) {
        TokError(Twine("Type mismatch for list, expected list type, got ") +
                 ItemType->getAsString());
        return nullptr;
      }
      GivenListTy = ListType;
    }

    if (Lex.getCode() != tgtok::r_square) {
      Vals = ParseValueList(CurRec, nullptr,
                            GivenListTy ? GivenListTy->getElementType() : nullptr);
      if (Vals.empty()) return nullptr;
    }
    if (Lex.getCode() != tgtok::r_square) {
      TokError("expected ']' at end of list value");
      return nullptr;
    }
    Lex.Lex();  // eat the ']'

    RecTy *GivenEltTy = nullptr;
    if (Lex.getCode() == tgtok::less) {
      // Optional list element type
      Lex.Lex();  // eat the '<'

      GivenEltTy = ParseType();
      if (!GivenEltTy) {
        // Couldn't parse element type
        return nullptr;
      }

      if (Lex.getCode() != tgtok::greater) {
        TokError("expected '>' at end of list element type");
        return nullptr;
      }
      Lex.Lex();  // eat the '>'
    }

    // Check elements
    RecTy *EltTy = nullptr;
    for (Init *V : Vals) {
      TypedInit *TArg = dyn_cast<TypedInit>(V);
      if (!TArg) {
        TokError("Untyped list element");
        return nullptr;
      }
      if (EltTy) {
        EltTy = resolveTypes(EltTy, TArg->getType());
        if (!EltTy) {
          TokError("Incompatible types in list elements");
          return nullptr;
        }
      } else {
        EltTy = TArg->getType();
      }
    }

    if (GivenEltTy) {
      if (EltTy) {
        // Verify consistency
        if (!EltTy->typeIsConvertibleTo(GivenEltTy)) {
          TokError("Incompatible types in list elements");
          return nullptr;
        }
      }
      EltTy = GivenEltTy;
    }

    if (!EltTy) {
      if (!ItemType) {
        TokError("No type for list");
        return nullptr;
      }
      DeducedEltTy = GivenListTy->getElementType();
    } else {
      // Make sure the deduced type is compatible with the given type
      if (GivenListTy) {
        if (!EltTy->typeIsConvertibleTo(GivenListTy->getElementType())) {
          TokError("Element type mismatch for list");
          return nullptr;
        }
      }
      DeducedEltTy = EltTy;
    }

    return ListInit::get(Vals, DeducedEltTy);
  }
  case tgtok::l_paren: {         // Value ::= '(' IDValue DagArgList ')'
    Lex.Lex();   // eat the '('
    if (Lex.getCode() != tgtok::Id && Lex.getCode() != tgtok::XCast) {
      TokError("expected identifier in dag init");
      return nullptr;
    }

    Init *Operator = ParseValue(CurRec);
    if (!Operator) return nullptr;

    // If the operator name is present, parse it.
    std::string OperatorName;
    if (Lex.getCode() == tgtok::colon) {
      if (Lex.Lex() != tgtok::VarName) { // eat the ':'
        TokError("expected variable name in dag operator");
        return nullptr;
      }
      OperatorName = Lex.getCurStrVal();
      Lex.Lex();  // eat the VarName.
    }

    std::vector<std::pair<llvm::Init*, std::string> > DagArgs;
    if (Lex.getCode() != tgtok::r_paren) {
      DagArgs = ParseDagArgList(CurRec);
      if (DagArgs.empty()) return nullptr;
    }

    if (Lex.getCode() != tgtok::r_paren) {
      TokError("expected ')' in dag init");
      return nullptr;
    }
    Lex.Lex();  // eat the ')'

    return DagInit::get(Operator, OperatorName, DagArgs);
  }

  case tgtok::XHead:
  case tgtok::XTail:
  case tgtok::XEmpty:
  case tgtok::XCast:  // Value ::= !unop '(' Value ')'
  case tgtok::XConcat:
  case tgtok::XADD:
  case tgtok::XAND:
  case tgtok::XSRA:
  case tgtok::XSRL:
  case tgtok::XSHL:
  case tgtok::XEq:
  case tgtok::XListConcat:
  case tgtok::XStrConcat:   // Value ::= !binop '(' Value ',' Value ')'
  case tgtok::XIf:
  case tgtok::XForEach:
  case tgtok::XSubst: {  // Value ::= !ternop '(' Value ',' Value ',' Value ')'
    return ParseOperation(CurRec, ItemType);
  }
  }

  return R;
}

/// ParseValue - Parse a tblgen value.  This returns null on error.
///
///   Value       ::= SimpleValue ValueSuffix*
///   ValueSuffix ::= '{' BitList '}'
///   ValueSuffix ::= '[' BitList ']'
///   ValueSuffix ::= '.' ID
///
Init *TGParser::ParseValue(Record *CurRec, RecTy *ItemType, IDParseMode Mode) {
  Init *Result = ParseSimpleValue(CurRec, ItemType, Mode);
  if (!Result) return nullptr;

  // Parse the suffixes now if present.
  while (1) {
    switch (Lex.getCode()) {
    default: return Result;
    case tgtok::l_brace: {
      if (Mode == ParseNameMode || Mode == ParseForeachMode)
        // This is the beginning of the object body.
        return Result;

      SMLoc CurlyLoc = Lex.getLoc();
      Lex.Lex(); // eat the '{'
      std::vector<unsigned> Ranges = ParseRangeList();
      if (Ranges.empty()) return nullptr;

      // Reverse the bitlist.
      std::reverse(Ranges.begin(), Ranges.end());
      Result = Result->convertInitializerBitRange(Ranges);
      if (!Result) {
        Error(CurlyLoc, "Invalid bit range for value");
        return nullptr;
      }

      // Eat the '}'.
      if (Lex.getCode() != tgtok::r_brace) {
        TokError("expected '}' at end of bit range list");
        return nullptr;
      }
      Lex.Lex();
      break;
    }
    case tgtok::l_square: {
      SMLoc SquareLoc = Lex.getLoc();
      Lex.Lex(); // eat the '['
      std::vector<unsigned> Ranges = ParseRangeList();
      if (Ranges.empty()) return nullptr;

      Result = Result->convertInitListSlice(Ranges);
      if (!Result) {
        Error(SquareLoc, "Invalid range for list slice");
        return nullptr;
      }

      // Eat the ']'.
      if (Lex.getCode() != tgtok::r_square) {
        TokError("expected ']' at end of list slice");
        return nullptr;
      }
      Lex.Lex();
      break;
    }
    case tgtok::period:
      if (Lex.Lex() != tgtok::Id) {  // eat the .
        TokError("expected field identifier after '.'");
        return nullptr;
      }
      if (!Result->getFieldType(Lex.getCurStrVal())) {
        TokError("Cannot access field '" + Lex.getCurStrVal() + "' of value '" +
                 Result->getAsString() + "'");
        return nullptr;
      }
      Result = FieldInit::get(Result, Lex.getCurStrVal());
      Lex.Lex();  // eat field name
      break;

    case tgtok::paste:
      SMLoc PasteLoc = Lex.getLoc();

      // Create a !strconcat() operation, first casting each operand to
      // a string if necessary.

      TypedInit *LHS = dyn_cast<TypedInit>(Result);
      if (!LHS) {
        Error(PasteLoc, "LHS of paste is not typed!");
        return nullptr;
      }

      if (LHS->getType() != StringRecTy::get()) {
        LHS = UnOpInit::get(UnOpInit::CAST, LHS, StringRecTy::get());
      }

      TypedInit *RHS = nullptr;

      Lex.Lex();  // Eat the '#'.
      switch (Lex.getCode()) { 
      case tgtok::colon:
      case tgtok::semi:
      case tgtok::l_brace:
        // These are all of the tokens that can begin an object body.
        // Some of these can also begin values but we disallow those cases
        // because they are unlikely to be useful.

        // Trailing paste, concat with an empty string.
        RHS = StringInit::get("");
        break;

      default:
        Init *RHSResult = ParseValue(CurRec, ItemType, ParseNameMode);
        RHS = dyn_cast<TypedInit>(RHSResult);
        if (!RHS) {
          Error(PasteLoc, "RHS of paste is not typed!");
          return nullptr;
        }

        if (RHS->getType() != StringRecTy::get()) {
          RHS = UnOpInit::get(UnOpInit::CAST, RHS, StringRecTy::get());
        }

        break;
      }

      Result = BinOpInit::get(BinOpInit::STRCONCAT, LHS, RHS,
                              StringRecTy::get())->Fold(CurRec, CurMultiClass);
      break;
    }
  }
}

/// ParseDagArgList - Parse the argument list for a dag literal expression.
///
///    DagArg     ::= Value (':' VARNAME)?
///    DagArg     ::= VARNAME
///    DagArgList ::= DagArg
///    DagArgList ::= DagArgList ',' DagArg
std::vector<std::pair<llvm::Init*, std::string> >
TGParser::ParseDagArgList(Record *CurRec) {
  std::vector<std::pair<llvm::Init*, std::string> > Result;

  while (1) {
    // DagArg ::= VARNAME
    if (Lex.getCode() == tgtok::VarName) {
      // A missing value is treated like '?'.
      Result.emplace_back(UnsetInit::get(), Lex.getCurStrVal());
      Lex.Lex();
    } else {
      // DagArg ::= Value (':' VARNAME)?
      Init *Val = ParseValue(CurRec);
      if (!Val)
        return std::vector<std::pair<llvm::Init*, std::string> >();

      // If the variable name is present, add it.
      std::string VarName;
      if (Lex.getCode() == tgtok::colon) {
        if (Lex.Lex() != tgtok::VarName) { // eat the ':'
          TokError("expected variable name in dag literal");
          return std::vector<std::pair<llvm::Init*, std::string> >();
        }
        VarName = Lex.getCurStrVal();
        Lex.Lex();  // eat the VarName.
      }

      Result.push_back(std::make_pair(Val, VarName));
    }
    if (Lex.getCode() != tgtok::comma) break;
    Lex.Lex(); // eat the ','
  }

  return Result;
}


/// ParseValueList - Parse a comma separated list of values, returning them as a
/// vector.  Note that this always expects to be able to parse at least one
/// value.  It returns an empty list if this is not possible.
///
///   ValueList ::= Value (',' Value)
///
std::vector<Init*> TGParser::ParseValueList(Record *CurRec, Record *ArgsRec,
                                            RecTy *EltTy) {
  std::vector<Init*> Result;
  RecTy *ItemType = EltTy;
  unsigned int ArgN = 0;
  if (ArgsRec && !EltTy) {
    const std::vector<Init *> &TArgs = ArgsRec->getTemplateArgs();
    if (TArgs.empty()) {
      TokError("template argument provided to non-template class");
      return std::vector<Init*>();
    }
    const RecordVal *RV = ArgsRec->getValue(TArgs[ArgN]);
    if (!RV) {
      errs() << "Cannot find template arg " << ArgN << " (" << TArgs[ArgN]
        << ")\n";
    }
    assert(RV && "Template argument record not found??");
    ItemType = RV->getType();
    ++ArgN;
  }
  Result.push_back(ParseValue(CurRec, ItemType));
  if (!Result.back()) return std::vector<Init*>();

  while (Lex.getCode() == tgtok::comma) {
    Lex.Lex();  // Eat the comma

    if (ArgsRec && !EltTy) {
      const std::vector<Init *> &TArgs = ArgsRec->getTemplateArgs();
      if (ArgN >= TArgs.size()) {
        TokError("too many template arguments");
        return std::vector<Init*>();
      }
      const RecordVal *RV = ArgsRec->getValue(TArgs[ArgN]);
      assert(RV && "Template argument record not found??");
      ItemType = RV->getType();
      ++ArgN;
    }
    Result.push_back(ParseValue(CurRec, ItemType));
    if (!Result.back()) return std::vector<Init*>();
  }

  return Result;
}


/// ParseDeclaration - Read a declaration, returning the name of field ID, or an
/// empty string on error.  This can happen in a number of different context's,
/// including within a def or in the template args for a def (which which case
/// CurRec will be non-null) and within the template args for a multiclass (in
/// which case CurRec will be null, but CurMultiClass will be set).  This can
/// also happen within a def that is within a multiclass, which will set both
/// CurRec and CurMultiClass.
///
///  Declaration ::= FIELD? Type ID ('=' Value)?
///
Init *TGParser::ParseDeclaration(Record *CurRec,
                                       bool ParsingTemplateArgs) {
  // Read the field prefix if present.
  bool HasField = Lex.getCode() == tgtok::Field;
  if (HasField) Lex.Lex();

  RecTy *Type = ParseType();
  if (!Type) return nullptr;

  if (Lex.getCode() != tgtok::Id) {
    TokError("Expected identifier in declaration");
    return nullptr;
  }

  SMLoc IdLoc = Lex.getLoc();
  Init *DeclName = StringInit::get(Lex.getCurStrVal());
  Lex.Lex();

  if (ParsingTemplateArgs) {
    if (CurRec)
      DeclName = QualifyName(*CurRec, CurMultiClass, DeclName, ":");
    else
      assert(CurMultiClass);
    if (CurMultiClass)
      DeclName = QualifyName(CurMultiClass->Rec, CurMultiClass, DeclName,
                             "::");
  }

  // Add the value.
  if (AddValue(CurRec, IdLoc, RecordVal(DeclName, Type, HasField)))
    return nullptr;

  // If a value is present, parse it.
  if (Lex.getCode() == tgtok::equal) {
    Lex.Lex();
    SMLoc ValLoc = Lex.getLoc();
    Init *Val = ParseValue(CurRec, Type);
    if (!Val ||
        SetValue(CurRec, ValLoc, DeclName, std::vector<unsigned>(), Val))
      // Return the name, even if an error is thrown.  This is so that we can
      // continue to make some progress, even without the value having been
      // initialized.
      return DeclName;
  }

  return DeclName;
}

/// ParseForeachDeclaration - Read a foreach declaration, returning
/// the name of the declared object or a NULL Init on error.  Return
/// the name of the parsed initializer list through ForeachListName.
///
///  ForeachDeclaration ::= ID '=' '[' ValueList ']'
///  ForeachDeclaration ::= ID '=' '{' RangeList '}'
///  ForeachDeclaration ::= ID '=' RangePiece
///
VarInit *TGParser::ParseForeachDeclaration(ListInit *&ForeachListValue) {
  if (Lex.getCode() != tgtok::Id) {
    TokError("Expected identifier in foreach declaration");
    return nullptr;
  }

  Init *DeclName = StringInit::get(Lex.getCurStrVal());
  Lex.Lex();

  // If a value is present, parse it.
  if (Lex.getCode() != tgtok::equal) {
    TokError("Expected '=' in foreach declaration");
    return nullptr;
  }
  Lex.Lex();  // Eat the '='

  RecTy *IterType = nullptr;
  std::vector<unsigned> Ranges;

  switch (Lex.getCode()) {
  default: TokError("Unknown token when expecting a range list"); return nullptr;
  case tgtok::l_square: { // '[' ValueList ']'
    Init *List = ParseSimpleValue(nullptr, nullptr, ParseForeachMode);
    ForeachListValue = dyn_cast<ListInit>(List);
    if (!ForeachListValue) {
      TokError("Expected a Value list");
      return nullptr;
    }
    RecTy *ValueType = ForeachListValue->getType();
    ListRecTy *ListType = dyn_cast<ListRecTy>(ValueType);
    if (!ListType) {
      TokError("Value list is not of list type");
      return nullptr;
    }
    IterType = ListType->getElementType();
    break;
  }

  case tgtok::IntVal: { // RangePiece.
    if (ParseRangePiece(Ranges))
      return nullptr;
    break;
  }

  case tgtok::l_brace: { // '{' RangeList '}'
    Lex.Lex(); // eat the '{'
    Ranges = ParseRangeList();
    if (Lex.getCode() != tgtok::r_brace) {
      TokError("expected '}' at end of bit range list");
      return nullptr;
    }
    Lex.Lex();
    break;
  }
  }

  if (!Ranges.empty()) {
    assert(!IterType && "Type already initialized?");
    IterType = IntRecTy::get();
    std::vector<Init*> Values;
    for (unsigned R : Ranges)
      Values.push_back(IntInit::get(R));
    ForeachListValue = ListInit::get(Values, IterType);
  }

  if (!IterType)
    return nullptr;

  return VarInit::get(DeclName, IterType);
}

/// ParseTemplateArgList - Read a template argument list, which is a non-empty
/// sequence of template-declarations in <>'s.  If CurRec is non-null, these are
/// template args for a def, which may or may not be in a multiclass.  If null,
/// these are the template args for a multiclass.
///
///    TemplateArgList ::= '<' Declaration (',' Declaration)* '>'
///
bool TGParser::ParseTemplateArgList(Record *CurRec) {
  assert(Lex.getCode() == tgtok::less && "Not a template arg list!");
  Lex.Lex(); // eat the '<'

  Record *TheRecToAddTo = CurRec ? CurRec : &CurMultiClass->Rec;

  // Read the first declaration.
  Init *TemplArg = ParseDeclaration(CurRec, true/*templateargs*/);
  if (!TemplArg)
    return true;

  TheRecToAddTo->addTemplateArg(TemplArg);

  while (Lex.getCode() == tgtok::comma) {
    Lex.Lex(); // eat the ','

    // Read the following declarations.
    TemplArg = ParseDeclaration(CurRec, true/*templateargs*/);
    if (!TemplArg)
      return true;
    TheRecToAddTo->addTemplateArg(TemplArg);
  }

  if (Lex.getCode() != tgtok::greater)
    return TokError("expected '>' at end of template argument list");
  Lex.Lex(); // eat the '>'.
  return false;
}


/// ParseBodyItem - Parse a single item at within the body of a def or class.
///
///   BodyItem ::= Declaration ';'
///   BodyItem ::= LET ID OptionalBitList '=' Value ';'
bool TGParser::ParseBodyItem(Record *CurRec) {
  if (Lex.getCode() != tgtok::Let) {
    if (!ParseDeclaration(CurRec, false))
      return true;

    if (Lex.getCode() != tgtok::semi)
      return TokError("expected ';' after declaration");
    Lex.Lex();
    return false;
  }

  // LET ID OptionalRangeList '=' Value ';'
  if (Lex.Lex() != tgtok::Id)
    return TokError("expected field identifier after let");

  SMLoc IdLoc = Lex.getLoc();
  std::string FieldName = Lex.getCurStrVal();
  Lex.Lex();  // eat the field name.

  std::vector<unsigned> BitList;
  if (ParseOptionalBitList(BitList))
    return true;
  std::reverse(BitList.begin(), BitList.end());

  if (Lex.getCode() != tgtok::equal)
    return TokError("expected '=' in let expression");
  Lex.Lex();  // eat the '='.

  RecordVal *Field = CurRec->getValue(FieldName);
  if (!Field)
    return TokError("Value '" + FieldName + "' unknown!");

  RecTy *Type = Field->getType();

  Init *Val = ParseValue(CurRec, Type);
  if (!Val) return true;

  if (Lex.getCode() != tgtok::semi)
    return TokError("expected ';' after let expression");
  Lex.Lex();

  return SetValue(CurRec, IdLoc, FieldName, BitList, Val);
}

/// ParseBody - Read the body of a class or def.  Return true on error, false on
/// success.
///
///   Body     ::= ';'
///   Body     ::= '{' BodyList '}'
///   BodyList BodyItem*
///
bool TGParser::ParseBody(Record *CurRec) {
  // If this is a null definition, just eat the semi and return.
  if (Lex.getCode() == tgtok::semi) {
    Lex.Lex();
    return false;
  }

  if (Lex.getCode() != tgtok::l_brace)
    return TokError("Expected ';' or '{' to start body");
  // Eat the '{'.
  Lex.Lex();

  while (Lex.getCode() != tgtok::r_brace)
    if (ParseBodyItem(CurRec))
      return true;

  // Eat the '}'.
  Lex.Lex();
  return false;
}

/// \brief Apply the current let bindings to \a CurRec.
/// \returns true on error, false otherwise.
bool TGParser::ApplyLetStack(Record *CurRec) {
  for (std::vector<LetRecord> &LetInfo : LetStack)
    for (LetRecord &LR : LetInfo)
      if (SetValue(CurRec, LR.Loc, LR.Name, LR.Bits, LR.Value))
        return true;
  return false;
}

/// ParseObjectBody - Parse the body of a def or class.  This consists of an
/// optional ClassList followed by a Body.  CurRec is the current def or class
/// that is being parsed.
///
///   ObjectBody      ::= BaseClassList Body
///   BaseClassList   ::= /*empty*/
///   BaseClassList   ::= ':' BaseClassListNE
///   BaseClassListNE ::= SubClassRef (',' SubClassRef)*
///
bool TGParser::ParseObjectBody(Record *CurRec) {
  // If there is a baseclass list, read it.
  if (Lex.getCode() == tgtok::colon) {
    Lex.Lex();

    // Read all of the subclasses.
    SubClassReference SubClass = ParseSubClassReference(CurRec, false);
    while (1) {
      // Check for error.
      if (!SubClass.Rec) return true;

      // Add it.
      if (AddSubClass(CurRec, SubClass))
        return true;

      if (Lex.getCode() != tgtok::comma) break;
      Lex.Lex(); // eat ','.
      SubClass = ParseSubClassReference(CurRec, false);
    }
  }

  if (ApplyLetStack(CurRec))
    return true;

  return ParseBody(CurRec);
}

/// ParseDef - Parse and return a top level or multiclass def, return the record
/// corresponding to it.  This returns null on error.
///
///   DefInst ::= DEF ObjectName ObjectBody
///
bool TGParser::ParseDef(MultiClass *CurMultiClass) {
  SMLoc DefLoc = Lex.getLoc();
  assert(Lex.getCode() == tgtok::Def && "Unknown tok");
  Lex.Lex();  // Eat the 'def' token.

  // Parse ObjectName and make a record for it.
  std::unique_ptr<Record> CurRecOwner;
  Init *Name = ParseObjectName(CurMultiClass);
  if (Name)
    CurRecOwner = make_unique<Record>(Name, DefLoc, Records);
  else
    CurRecOwner = llvm::make_unique<Record>(GetNewAnonymousName(), DefLoc,
                                            Records, /*IsAnonymous=*/true);
  Record *CurRec = CurRecOwner.get(); // Keep a copy since we may release.

  if (!CurMultiClass && Loops.empty()) {
    // Top-level def definition.

    // Ensure redefinition doesn't happen.
    if (Records.getDef(CurRec->getNameInitAsString()))
      return Error(DefLoc, "def '" + CurRec->getNameInitAsString()+
                   "' already defined");
    Records.addDef(std::move(CurRecOwner));

    if (ParseObjectBody(CurRec))
      return true;
  } else if (CurMultiClass) {
    // Parse the body before adding this prototype to the DefPrototypes vector.
    // That way implicit definitions will be added to the DefPrototypes vector
    // before this object, instantiated prior to defs derived from this object,
    // and this available for indirect name resolution when defs derived from
    // this object are instantiated.
    if (ParseObjectBody(CurRec))
      return true;

    // Otherwise, a def inside a multiclass, add it to the multiclass.
    for (const auto &Proto : CurMultiClass->DefPrototypes)
      if (Proto->getNameInit() == CurRec->getNameInit())
        return Error(DefLoc, "def '" + CurRec->getNameInitAsString() +
                     "' already defined in this multiclass!");
    CurMultiClass->DefPrototypes.push_back(std::move(CurRecOwner));
  } else if (ParseObjectBody(CurRec)) {
    return true;
  }

  if (!CurMultiClass)  // Def's in multiclasses aren't really defs.
    // See Record::setName().  This resolve step will see any new name
    // for the def that might have been created when resolving
    // inheritance, values and arguments above.
    CurRec->resolveReferences();

  // If ObjectBody has template arguments, it's an error.
  assert(CurRec->getTemplateArgs().empty() && "How'd this get template args?");

  if (CurMultiClass) {
    // Copy the template arguments for the multiclass into the def.
    for (Init *TArg : CurMultiClass->Rec.getTemplateArgs()) {
      const RecordVal *RV = CurMultiClass->Rec.getValue(TArg);
      assert(RV && "Template arg doesn't exist?");
      CurRec->addValue(*RV);
    }
  }

  if (ProcessForeachDefs(CurRec, DefLoc))
    return Error(DefLoc, "Could not process loops for def" +
                 CurRec->getNameInitAsString());

  return false;
}

/// ParseForeach - Parse a for statement.  Return the record corresponding
/// to it.  This returns true on error.
///
///   Foreach ::= FOREACH Declaration IN '{ ObjectList '}'
///   Foreach ::= FOREACH Declaration IN Object
///
bool TGParser::ParseForeach(MultiClass *CurMultiClass) {
  assert(Lex.getCode() == tgtok::Foreach && "Unknown tok");
  Lex.Lex();  // Eat the 'for' token.

  // Make a temporary object to record items associated with the for
  // loop.
  ListInit *ListValue = nullptr;
  VarInit *IterName = ParseForeachDeclaration(ListValue);
  if (!IterName)
    return TokError("expected declaration in for");

  if (Lex.getCode() != tgtok::In)
    return TokError("Unknown tok");
  Lex.Lex();  // Eat the in

  // Create a loop object and remember it.
  Loops.push_back(ForeachLoop(IterName, ListValue));

  if (Lex.getCode() != tgtok::l_brace) {
    // FOREACH Declaration IN Object
    if (ParseObject(CurMultiClass))
      return true;
  } else {
    SMLoc BraceLoc = Lex.getLoc();
    // Otherwise, this is a group foreach.
    Lex.Lex();  // eat the '{'.

    // Parse the object list.
    if (ParseObjectList(CurMultiClass))
      return true;

    if (Lex.getCode() != tgtok::r_brace) {
      TokError("expected '}' at end of foreach command");
      return Error(BraceLoc, "to match this '{'");
    }
    Lex.Lex();  // Eat the }
  }

  // We've processed everything in this loop.
  Loops.pop_back();

  return false;
}

/// ParseClass - Parse a tblgen class definition.
///
///   ClassInst ::= CLASS ID TemplateArgList? ObjectBody
///
bool TGParser::ParseClass() {
  assert(Lex.getCode() == tgtok::Class && "Unexpected token!");
  Lex.Lex();

  if (Lex.getCode() != tgtok::Id)
    return TokError("expected class name after 'class' keyword");

  Record *CurRec = Records.getClass(Lex.getCurStrVal());
  if (CurRec) {
    // If the body was previously defined, this is an error.
    if (CurRec->getValues().size() > 1 ||  // Account for NAME.
        !CurRec->getSuperClasses().empty() ||
        !CurRec->getTemplateArgs().empty())
      return TokError("Class '" + CurRec->getNameInitAsString() +
                      "' already defined");
  } else {
    // If this is the first reference to this class, create and add it.
    auto NewRec =
        llvm::make_unique<Record>(Lex.getCurStrVal(), Lex.getLoc(), Records);
    CurRec = NewRec.get();
    Records.addClass(std::move(NewRec));
  }
  Lex.Lex(); // eat the name.

  // If there are template args, parse them.
  if (Lex.getCode() == tgtok::less)
    if (ParseTemplateArgList(CurRec))
      return true;

  // Finally, parse the object body.
  return ParseObjectBody(CurRec);
}

/// ParseLetList - Parse a non-empty list of assignment expressions into a list
/// of LetRecords.
///
///   LetList ::= LetItem (',' LetItem)*
///   LetItem ::= ID OptionalRangeList '=' Value
///
std::vector<LetRecord> TGParser::ParseLetList() {
  std::vector<LetRecord> Result;

  while (1) {
    if (Lex.getCode() != tgtok::Id) {
      TokError("expected identifier in let definition");
      return std::vector<LetRecord>();
    }
    std::string Name = Lex.getCurStrVal();
    SMLoc NameLoc = Lex.getLoc();
    Lex.Lex();  // Eat the identifier.

    // Check for an optional RangeList.
    std::vector<unsigned> Bits;
    if (ParseOptionalRangeList(Bits))
      return std::vector<LetRecord>();
    std::reverse(Bits.begin(), Bits.end());

    if (Lex.getCode() != tgtok::equal) {
      TokError("expected '=' in let expression");
      return std::vector<LetRecord>();
    }
    Lex.Lex();  // eat the '='.

    Init *Val = ParseValue(nullptr);
    if (!Val) return std::vector<LetRecord>();

    // Now that we have everything, add the record.
    Result.emplace_back(std::move(Name), std::move(Bits), Val, NameLoc);

    if (Lex.getCode() != tgtok::comma)
      return Result;
    Lex.Lex();  // eat the comma.
  }
}

/// ParseTopLevelLet - Parse a 'let' at top level.  This can be a couple of
/// different related productions. This works inside multiclasses too.
///
///   Object ::= LET LetList IN '{' ObjectList '}'
///   Object ::= LET LetList IN Object
///
bool TGParser::ParseTopLevelLet(MultiClass *CurMultiClass) {
  assert(Lex.getCode() == tgtok::Let && "Unexpected token");
  Lex.Lex();

  // Add this entry to the let stack.
  std::vector<LetRecord> LetInfo = ParseLetList();
  if (LetInfo.empty()) return true;
  LetStack.push_back(std::move(LetInfo));

  if (Lex.getCode() != tgtok::In)
    return TokError("expected 'in' at end of top-level 'let'");
  Lex.Lex();

  // If this is a scalar let, just handle it now
  if (Lex.getCode() != tgtok::l_brace) {
    // LET LetList IN Object
    if (ParseObject(CurMultiClass))
      return true;
  } else {   // Object ::= LETCommand '{' ObjectList '}'
    SMLoc BraceLoc = Lex.getLoc();
    // Otherwise, this is a group let.
    Lex.Lex();  // eat the '{'.

    // Parse the object list.
    if (ParseObjectList(CurMultiClass))
      return true;

    if (Lex.getCode() != tgtok::r_brace) {
      TokError("expected '}' at end of top level let command");
      return Error(BraceLoc, "to match this '{'");
    }
    Lex.Lex();
  }

  // Outside this let scope, this let block is not active.
  LetStack.pop_back();
  return false;
}

/// ParseMultiClass - Parse a multiclass definition.
///
///  MultiClassInst ::= MULTICLASS ID TemplateArgList?
///                     ':' BaseMultiClassList '{' MultiClassObject+ '}'
///  MultiClassObject ::= DefInst
///  MultiClassObject ::= MultiClassInst
///  MultiClassObject ::= DefMInst
///  MultiClassObject ::= LETCommand '{' ObjectList '}'
///  MultiClassObject ::= LETCommand Object
///
bool TGParser::ParseMultiClass() {
  assert(Lex.getCode() == tgtok::MultiClass && "Unexpected token");
  Lex.Lex();  // Eat the multiclass token.

  if (Lex.getCode() != tgtok::Id)
    return TokError("expected identifier after multiclass for name");
  std::string Name = Lex.getCurStrVal();

  auto Result =
    MultiClasses.insert(std::make_pair(Name,
                    llvm::make_unique<MultiClass>(Name, Lex.getLoc(),Records)));

  if (!Result.second)
    return TokError("multiclass '" + Name + "' already defined");

  CurMultiClass = Result.first->second.get();
  Lex.Lex();  // Eat the identifier.

  // If there are template args, parse them.
  if (Lex.getCode() == tgtok::less)
    if (ParseTemplateArgList(nullptr))
      return true;

  bool inherits = false;

  // If there are submulticlasses, parse them.
  if (Lex.getCode() == tgtok::colon) {
    inherits = true;

    Lex.Lex();

    // Read all of the submulticlasses.
    SubMultiClassReference SubMultiClass =
      ParseSubMultiClassReference(CurMultiClass);
    while (1) {
      // Check for error.
      if (!SubMultiClass.MC) return true;

      // Add it.
      if (AddSubMultiClass(CurMultiClass, SubMultiClass))
        return true;

      if (Lex.getCode() != tgtok::comma) break;
      Lex.Lex(); // eat ','.
      SubMultiClass = ParseSubMultiClassReference(CurMultiClass);
    }
  }

  if (Lex.getCode() != tgtok::l_brace) {
    if (!inherits)
      return TokError("expected '{' in multiclass definition");
    if (Lex.getCode() != tgtok::semi)
      return TokError("expected ';' in multiclass definition");
    Lex.Lex();  // eat the ';'.
  } else {
    if (Lex.Lex() == tgtok::r_brace)  // eat the '{'.
      return TokError("multiclass must contain at least one def");

    while (Lex.getCode() != tgtok::r_brace) {
      switch (Lex.getCode()) {
      default:
        return TokError("expected 'let', 'def' or 'defm' in multiclass body");
      case tgtok::Let:
      case tgtok::Def:
      case tgtok::Defm:
      case tgtok::Foreach:
        if (ParseObject(CurMultiClass))
          return true;
        break;
      }
    }
    Lex.Lex();  // eat the '}'.
  }

  CurMultiClass = nullptr;
  return false;
}

Record *TGParser::
InstantiateMulticlassDef(MultiClass &MC,
                         Record *DefProto,
                         Init *&DefmPrefix,
                         SMRange DefmPrefixRange,
                         const std::vector<Init *> &TArgs,
                         std::vector<Init *> &TemplateVals) {
  // We need to preserve DefProto so it can be reused for later
  // instantiations, so create a new Record to inherit from it.

  // Add in the defm name.  If the defm prefix is empty, give each
  // instantiated def a unique name.  Otherwise, if "#NAME#" exists in the
  // name, substitute the prefix for #NAME#.  Otherwise, use the defm name
  // as a prefix.

  bool IsAnonymous = false;
  if (!DefmPrefix) {
    DefmPrefix = StringInit::get(GetNewAnonymousName());
    IsAnonymous = true;
  }

  Init *DefName = DefProto->getNameInit();
  StringInit *DefNameString = dyn_cast<StringInit>(DefName);

  if (DefNameString) {
    // We have a fully expanded string so there are no operators to
    // resolve.  We should concatenate the given prefix and name.
    DefName =
      BinOpInit::get(BinOpInit::STRCONCAT,
                     UnOpInit::get(UnOpInit::CAST, DefmPrefix,
                                   StringRecTy::get())->Fold(DefProto, &MC),
                     DefName, StringRecTy::get())->Fold(DefProto, &MC);
  }

  // Make a trail of SMLocs from the multiclass instantiations.
  SmallVector<SMLoc, 4> Locs(1, DefmPrefixRange.Start);
  Locs.append(DefProto->getLoc().begin(), DefProto->getLoc().end());
  auto CurRec = make_unique<Record>(DefName, Locs, Records, IsAnonymous);

  SubClassReference Ref;
  Ref.RefRange = DefmPrefixRange;
  Ref.Rec = DefProto;
  AddSubClass(CurRec.get(), Ref);

  // Set the value for NAME. We don't resolve references to it 'til later,
  // though, so that uses in nested multiclass names don't get
  // confused.
  if (SetValue(CurRec.get(), Ref.RefRange.Start, "NAME",
               std::vector<unsigned>(), DefmPrefix)) {
    Error(DefmPrefixRange.Start, "Could not resolve " +
          CurRec->getNameInitAsString() + ":NAME to '" +
          DefmPrefix->getAsUnquotedString() + "'");
    return nullptr;
  }

  // If the DefNameString didn't resolve, we probably have a reference to
  // NAME and need to replace it. We need to do at least this much greedily,
  // otherwise nested multiclasses will end up with incorrect NAME expansions.
  if (!DefNameString) {
    RecordVal *DefNameRV = CurRec->getValue("NAME");
    CurRec->resolveReferencesTo(DefNameRV);
  }

  if (!CurMultiClass) {
    // Now that we're at the top level, resolve all NAME references
    // in the resultant defs that weren't in the def names themselves.
    RecordVal *DefNameRV = CurRec->getValue("NAME");
    CurRec->resolveReferencesTo(DefNameRV);

    // Check if the name is a complex pattern.
    // If so, resolve it.
    DefName = CurRec->getNameInit();
    DefNameString = dyn_cast<StringInit>(DefName);

    // OK the pattern is more complex than simply using NAME.
    // Let's use the heavy weaponery.
    if (!DefNameString) {
      ResolveMulticlassDefArgs(MC, CurRec.get(), DefmPrefixRange.Start,
                               Lex.getLoc(), TArgs, TemplateVals,
                               false/*Delete args*/);
      DefName = CurRec->getNameInit();
      DefNameString = dyn_cast<StringInit>(DefName);

      if (!DefNameString)
        DefName = DefName->convertInitializerTo(StringRecTy::get());

      // We ran out of options here...
      DefNameString = dyn_cast<StringInit>(DefName);
      if (!DefNameString) {
        PrintFatalError(CurRec->getLoc()[CurRec->getLoc().size() - 1],
                        DefName->getAsUnquotedString() + " is not a string.");
        return nullptr;
      }

      CurRec->setName(DefName);
    }

    // Now that NAME references are resolved and we're at the top level of
    // any multiclass expansions, add the record to the RecordKeeper. If we are
    // currently in a multiclass, it means this defm appears inside a
    // multiclass and its name won't be fully resolvable until we see
    // the top-level defm. Therefore, we don't add this to the
    // RecordKeeper at this point. If we did we could get duplicate
    // defs as more than one probably refers to NAME or some other
    // common internal placeholder.

    // Ensure redefinition doesn't happen.
    if (Records.getDef(CurRec->getNameInitAsString())) {
      Error(DefmPrefixRange.Start, "def '" + CurRec->getNameInitAsString() +
            "' already defined, instantiating defm with subdef '" + 
            DefProto->getNameInitAsString() + "'");
      return nullptr;
    }

    Record *CurRecSave = CurRec.get(); // Keep a copy before we release.
    Records.addDef(std::move(CurRec));
    return CurRecSave;
  }

  // FIXME This is bad but the ownership transfer to caller is pretty messy.
  // The unique_ptr in this function at least protects the exits above.
  return CurRec.release();
}

bool TGParser::ResolveMulticlassDefArgs(MultiClass &MC,
                                        Record *CurRec,
                                        SMLoc DefmPrefixLoc,
                                        SMLoc SubClassLoc,
                                        const std::vector<Init *> &TArgs,
                                        std::vector<Init *> &TemplateVals,
                                        bool DeleteArgs) {
  // Loop over all of the template arguments, setting them to the specified
  // value or leaving them as the default if necessary.
  for (unsigned i = 0, e = TArgs.size(); i != e; ++i) {
    // Check if a value is specified for this temp-arg.
    if (i < TemplateVals.size()) {
      // Set it now.
      if (SetValue(CurRec, DefmPrefixLoc, TArgs[i], std::vector<unsigned>(),
                   TemplateVals[i]))
        return true;

      // Resolve it next.
      CurRec->resolveReferencesTo(CurRec->getValue(TArgs[i]));

      if (DeleteArgs)
        // Now remove it.
        CurRec->removeValue(TArgs[i]);

    } else if (!CurRec->getValue(TArgs[i])->getValue()->isComplete()) {
      return Error(SubClassLoc, "value not specified for template argument #" +
                   Twine(i) + " (" + TArgs[i]->getAsUnquotedString() +
                   ") of multiclassclass '" + MC.Rec.getNameInitAsString() +
                   "'");
    }
  }
  return false;
}

bool TGParser::ResolveMulticlassDef(MultiClass &MC,
                                    Record *CurRec,
                                    Record *DefProto,
                                    SMLoc DefmPrefixLoc) {
  // If the mdef is inside a 'let' expression, add to each def.
  if (ApplyLetStack(CurRec))
    return Error(DefmPrefixLoc, "when instantiating this defm");

  // Don't create a top level definition for defm inside multiclasses,
  // instead, only update the prototypes and bind the template args
  // with the new created definition.
  if (!CurMultiClass)
    return false;
  for (const auto &Proto : CurMultiClass->DefPrototypes)
    if (Proto->getNameInit() == CurRec->getNameInit())
      return Error(DefmPrefixLoc, "defm '" + CurRec->getNameInitAsString() +
                   "' already defined in this multiclass!");
  CurMultiClass->DefPrototypes.push_back(std::unique_ptr<Record>(CurRec));

  // Copy the template arguments for the multiclass into the new def.
  for (Init * TA : CurMultiClass->Rec.getTemplateArgs()) {
    const RecordVal *RV = CurMultiClass->Rec.getValue(TA);
    assert(RV && "Template arg doesn't exist?");
    CurRec->addValue(*RV);
  }

  return false;
}

/// ParseDefm - Parse the instantiation of a multiclass.
///
///   DefMInst ::= DEFM ID ':' DefmSubClassRef ';'
///
bool TGParser::ParseDefm(MultiClass *CurMultiClass) {
  assert(Lex.getCode() == tgtok::Defm && "Unexpected token!");
  SMLoc DefmLoc = Lex.getLoc();
  Init *DefmPrefix = nullptr;

  if (Lex.Lex() == tgtok::Id) {  // eat the defm.
    DefmPrefix = ParseObjectName(CurMultiClass);
  }

  SMLoc DefmPrefixEndLoc = Lex.getLoc();
  if (Lex.getCode() != tgtok::colon)
    return TokError("expected ':' after defm identifier");

  // Keep track of the new generated record definitions.
  std::vector<Record*> NewRecDefs;

  // This record also inherits from a regular class (non-multiclass)?
  bool InheritFromClass = false;

  // eat the colon.
  Lex.Lex();

  SMLoc SubClassLoc = Lex.getLoc();
  SubClassReference Ref = ParseSubClassReference(nullptr, true);

  while (1) {
    if (!Ref.Rec) return true;

    // To instantiate a multiclass, we need to first get the multiclass, then
    // instantiate each def contained in the multiclass with the SubClassRef
    // template parameters.
    MultiClass *MC = MultiClasses[Ref.Rec->getName()].get();
    assert(MC && "Didn't lookup multiclass correctly?");
    std::vector<Init*> &TemplateVals = Ref.TemplateArgs;

    // Verify that the correct number of template arguments were specified.
    const std::vector<Init *> &TArgs = MC->Rec.getTemplateArgs();
    if (TArgs.size() < TemplateVals.size())
      return Error(SubClassLoc,
                   "more template args specified than multiclass expects");

    // Loop over all the def's in the multiclass, instantiating each one.
    for (const std::unique_ptr<Record> &DefProto : MC->DefPrototypes) {
      // The record name construction goes as follow:
      //  - If the def name is a string, prepend the prefix.
      //  - If the def name is a more complex pattern, use that pattern.
      // As a result, the record is instanciated before resolving
      // arguments, as it would make its name a string.
      Record *CurRec = InstantiateMulticlassDef(*MC, DefProto.get(), DefmPrefix,
                                                SMRange(DefmLoc,
                                                        DefmPrefixEndLoc),
                                                TArgs, TemplateVals);
      if (!CurRec)
        return true;

      // Now that the record is instanciated, we can resolve arguments.
      if (ResolveMulticlassDefArgs(*MC, CurRec, DefmLoc, SubClassLoc,
                                   TArgs, TemplateVals, true/*Delete args*/))
        return Error(SubClassLoc, "could not instantiate def");

      if (ResolveMulticlassDef(*MC, CurRec, DefProto.get(), DefmLoc))
        return Error(SubClassLoc, "could not instantiate def");

      // Defs that can be used by other definitions should be fully resolved
      // before any use.
      if (DefProto->isResolveFirst() && !CurMultiClass) {
        CurRec->resolveReferences();
        CurRec->setResolveFirst(false);
      }
      NewRecDefs.push_back(CurRec);
    }


    if (Lex.getCode() != tgtok::comma) break;
    Lex.Lex(); // eat ','.

    if (Lex.getCode() != tgtok::Id)
      return TokError("expected identifier");

    SubClassLoc = Lex.getLoc();

    // A defm can inherit from regular classes (non-multiclass) as
    // long as they come in the end of the inheritance list.
    InheritFromClass = (Records.getClass(Lex.getCurStrVal()) != nullptr);

    if (InheritFromClass)
      break;

    Ref = ParseSubClassReference(nullptr, true);
  }

  if (InheritFromClass) {
    // Process all the classes to inherit as if they were part of a
    // regular 'def' and inherit all record values.
    SubClassReference SubClass = ParseSubClassReference(nullptr, false);
    while (1) {
      // Check for error.
      if (!SubClass.Rec) return true;

      // Get the expanded definition prototypes and teach them about
      // the record values the current class to inherit has
      for (Record *CurRec : NewRecDefs) {
        // Add it.
        if (AddSubClass(CurRec, SubClass))
          return true;

        if (ApplyLetStack(CurRec))
          return true;
      }

      if (Lex.getCode() != tgtok::comma) break;
      Lex.Lex(); // eat ','.
      SubClass = ParseSubClassReference(nullptr, false);
    }
  }

  if (!CurMultiClass)
    for (Record *CurRec : NewRecDefs)
      // See Record::setName().  This resolve step will see any new
      // name for the def that might have been created when resolving
      // inheritance, values and arguments above.
      CurRec->resolveReferences();

  if (Lex.getCode() != tgtok::semi)
    return TokError("expected ';' at end of defm");
  Lex.Lex();

  return false;
}

/// ParseObject
///   Object ::= ClassInst
///   Object ::= DefInst
///   Object ::= MultiClassInst
///   Object ::= DefMInst
///   Object ::= LETCommand '{' ObjectList '}'
///   Object ::= LETCommand Object
bool TGParser::ParseObject(MultiClass *MC) {
  switch (Lex.getCode()) {
  default:
    return TokError("Expected class, def, defm, multiclass or let definition");
  case tgtok::Let:   return ParseTopLevelLet(MC);
  case tgtok::Def:   return ParseDef(MC);
  case tgtok::Foreach:   return ParseForeach(MC);
  case tgtok::Defm:  return ParseDefm(MC);
  case tgtok::Class: return ParseClass();
  case tgtok::MultiClass: return ParseMultiClass();
  }
}

/// ParseObjectList
///   ObjectList :== Object*
bool TGParser::ParseObjectList(MultiClass *MC) {
  while (isObjectStart(Lex.getCode())) {
    if (ParseObject(MC))
      return true;
  }
  return false;
}

bool TGParser::ParseFile() {
  Lex.Lex(); // Prime the lexer.
  if (ParseObjectList()) return true;

  // If we have unread input at the end of the file, report it.
  if (Lex.getCode() == tgtok::Eof)
    return false;

  return TokError("Unexpected input at top level");
}

