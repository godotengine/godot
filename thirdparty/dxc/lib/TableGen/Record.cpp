//===- Record.cpp - Record implementation ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implement the tablegen record classes.
//
//===----------------------------------------------------------------------===//

#include "llvm/TableGen/Record.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/TableGen/Error.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
//    std::string wrapper for DenseMap purposes
//===----------------------------------------------------------------------===//

namespace llvm {

/// TableGenStringKey - This is a wrapper for std::string suitable for
/// using as a key to a DenseMap.  Because there isn't a particularly
/// good way to indicate tombstone or empty keys for strings, we want
/// to wrap std::string to indicate that this is a "special" string
/// not expected to take on certain values (those of the tombstone and
/// empty keys).  This makes things a little safer as it clarifies
/// that DenseMap is really not appropriate for general strings.

class TableGenStringKey {
public:
  TableGenStringKey(const std::string &str) : data(str) {}
  TableGenStringKey(const char *str) : data(str) {}

  const std::string &str() const { return data; }

  friend hash_code hash_value(const TableGenStringKey &Value) {
    using llvm::hash_value;
    return hash_value(Value.str());
  }
private:
  std::string data;
};

/// Specialize DenseMapInfo for TableGenStringKey.
template<> struct DenseMapInfo<TableGenStringKey> {
  static inline TableGenStringKey getEmptyKey() {
    TableGenStringKey Empty("<<<EMPTY KEY>>>");
    return Empty;
  }
  static inline TableGenStringKey getTombstoneKey() {
    TableGenStringKey Tombstone("<<<TOMBSTONE KEY>>>");
    return Tombstone;
  }
  static unsigned getHashValue(const TableGenStringKey& Val) {
    using llvm::hash_value;
    return hash_value(Val);
  }
  static bool isEqual(const TableGenStringKey& LHS,
                      const TableGenStringKey& RHS) {
    return LHS.str() == RHS.str();
  }
};

} // namespace llvm

//===----------------------------------------------------------------------===//
//    Type implementations
//===----------------------------------------------------------------------===//

BitRecTy BitRecTy::Shared;
IntRecTy IntRecTy::Shared;
StringRecTy StringRecTy::Shared;
DagRecTy DagRecTy::Shared;

void RecTy::dump() const { print(errs()); }

ListRecTy *RecTy::getListTy() {
  if (!ListTy)
    ListTy.reset(new ListRecTy(this));
  return ListTy.get();
}

bool RecTy::typeIsConvertibleTo(const RecTy *RHS) const {
  assert(RHS && "NULL pointer");
  return Kind == RHS->getRecTyKind();
}

bool BitRecTy::typeIsConvertibleTo(const RecTy *RHS) const{
  if (RecTy::typeIsConvertibleTo(RHS) || RHS->getRecTyKind() == IntRecTyKind)
    return true;
  if (const BitsRecTy *BitsTy = dyn_cast<BitsRecTy>(RHS))
    return BitsTy->getNumBits() == 1;
  return false;
}

BitsRecTy *BitsRecTy::get(unsigned Sz) {
  static std::vector<std::unique_ptr<BitsRecTy>> Shared;
  if (Sz >= Shared.size())
    Shared.resize(Sz + 1);
  std::unique_ptr<BitsRecTy> &Ty = Shared[Sz];
  if (!Ty)
    Ty.reset(new BitsRecTy(Sz));
  return Ty.get();
}

std::string BitsRecTy::getAsString() const {
  return "bits<" + utostr(Size) + ">";
}

bool BitsRecTy::typeIsConvertibleTo(const RecTy *RHS) const {
  if (RecTy::typeIsConvertibleTo(RHS)) //argument and the sender are same type
    return cast<BitsRecTy>(RHS)->Size == Size;
  RecTyKind kind = RHS->getRecTyKind();
  return (kind == BitRecTyKind && Size == 1) || (kind == IntRecTyKind);
}

bool IntRecTy::typeIsConvertibleTo(const RecTy *RHS) const {
  RecTyKind kind = RHS->getRecTyKind();
  return kind==BitRecTyKind || kind==BitsRecTyKind || kind==IntRecTyKind;
}

std::string StringRecTy::getAsString() const {
  return "string";
}

std::string ListRecTy::getAsString() const {
  return "list<" + Ty->getAsString() + ">";
}

bool ListRecTy::typeIsConvertibleTo(const RecTy *RHS) const {
  if (const auto *ListTy = dyn_cast<ListRecTy>(RHS))
    return Ty->typeIsConvertibleTo(ListTy->getElementType());
  return false;
}

std::string DagRecTy::getAsString() const {
  return "dag";
}

RecordRecTy *RecordRecTy::get(Record *R) {
  return dyn_cast<RecordRecTy>(R->getDefInit()->getType());
}

std::string RecordRecTy::getAsString() const {
  return Rec->getName();
}

bool RecordRecTy::typeIsConvertibleTo(const RecTy *RHS) const {
  const RecordRecTy *RTy = dyn_cast<RecordRecTy>(RHS);
  if (!RTy)
    return false;

  if (RTy->getRecord() == Rec || Rec->isSubClassOf(RTy->getRecord()))
    return true;

  for (Record *SC : RTy->getRecord()->getSuperClasses())
    if (Rec->isSubClassOf(SC))
      return true;

  return false;
}

/// resolveTypes - Find a common type that T1 and T2 convert to.
/// Return null if no such type exists.
///
RecTy *llvm::resolveTypes(RecTy *T1, RecTy *T2) {
  if (T1->typeIsConvertibleTo(T2))
    return T2;
  if (T2->typeIsConvertibleTo(T1))
    return T1;

  // If one is a Record type, check superclasses
  if (RecordRecTy *RecTy1 = dyn_cast<RecordRecTy>(T1)) {
    // See if T2 inherits from a type T1 also inherits from
    for (Record *SuperRec1 : RecTy1->getRecord()->getSuperClasses()) {
      RecordRecTy *SuperRecTy1 = RecordRecTy::get(SuperRec1);
      RecTy *NewType1 = resolveTypes(SuperRecTy1, T2);
      if (NewType1)
        return NewType1;
    }
  }
  if (RecordRecTy *RecTy2 = dyn_cast<RecordRecTy>(T2)) {
    // See if T1 inherits from a type T2 also inherits from
    for (Record *SuperRec2 : RecTy2->getRecord()->getSuperClasses()) {
      RecordRecTy *SuperRecTy2 = RecordRecTy::get(SuperRec2);
      RecTy *NewType2 = resolveTypes(T1, SuperRecTy2);
      if (NewType2)
        return NewType2;
    }
  }
  return nullptr;
}


//===----------------------------------------------------------------------===//
//    Initializer implementations
//===----------------------------------------------------------------------===//

void Init::anchor() { }
void Init::dump() const { return print(errs()); }

UnsetInit *UnsetInit::get() {
  static UnsetInit TheInit;
  return &TheInit;
}

Init *UnsetInit::convertInitializerTo(RecTy *Ty) const {
  if (auto *BRT = dyn_cast<BitsRecTy>(Ty)) {
    SmallVector<Init *, 16> NewBits(BRT->getNumBits());

    for (unsigned i = 0; i != BRT->getNumBits(); ++i)
      NewBits[i] = UnsetInit::get();

    return BitsInit::get(NewBits);
  }

  // All other types can just be returned.
  return const_cast<UnsetInit *>(this);
}

BitInit *BitInit::get(bool V) {
  static BitInit True(true);
  static BitInit False(false);

  return V ? &True : &False;
}

Init *BitInit::convertInitializerTo(RecTy *Ty) const {
  if (isa<BitRecTy>(Ty))
    return const_cast<BitInit *>(this);

  if (isa<IntRecTy>(Ty))
    return IntInit::get(getValue());

  if (auto *BRT = dyn_cast<BitsRecTy>(Ty)) {
    // Can only convert single bit.
    if (BRT->getNumBits() == 1)
      return BitsInit::get(const_cast<BitInit *>(this));
  }

  return nullptr;
}

static void
ProfileBitsInit(FoldingSetNodeID &ID, ArrayRef<Init *> Range) {
  ID.AddInteger(Range.size());

  for (Init *I : Range)
    ID.AddPointer(I);
}

BitsInit *BitsInit::get(ArrayRef<Init *> Range) {
  static FoldingSet<BitsInit> ThePool;
  static std::vector<std::unique_ptr<BitsInit>> TheActualPool;

  FoldingSetNodeID ID;
  ProfileBitsInit(ID, Range);

  void *IP = nullptr;
  if (BitsInit *I = ThePool.FindNodeOrInsertPos(ID, IP))
    return I;

  BitsInit *I = new BitsInit(Range);
  ThePool.InsertNode(I, IP);
  TheActualPool.push_back(std::unique_ptr<BitsInit>(I));
  return I;
}

void BitsInit::Profile(FoldingSetNodeID &ID) const {
  ProfileBitsInit(ID, Bits);
}

Init *BitsInit::convertInitializerTo(RecTy *Ty) const {
  if (isa<BitRecTy>(Ty)) {
    if (getNumBits() != 1) return nullptr; // Only accept if just one bit!
    return getBit(0);
  }

  if (auto *BRT = dyn_cast<BitsRecTy>(Ty)) {
    // If the number of bits is right, return it.  Otherwise we need to expand
    // or truncate.
    if (getNumBits() != BRT->getNumBits()) return nullptr;
    return const_cast<BitsInit *>(this);
  }

  if (isa<IntRecTy>(Ty)) {
    int64_t Result = 0;
    for (unsigned i = 0, e = getNumBits(); i != e; ++i)
      if (auto *Bit = dyn_cast<BitInit>(getBit(i)))
        Result |= static_cast<int64_t>(Bit->getValue()) << i;
      else
        return nullptr;
    return IntInit::get(Result);
  }

  return nullptr;
}

Init *
BitsInit::convertInitializerBitRange(const std::vector<unsigned> &Bits) const {
  SmallVector<Init *, 16> NewBits(Bits.size());

  for (unsigned i = 0, e = Bits.size(); i != e; ++i) {
    if (Bits[i] >= getNumBits())
      return nullptr;
    NewBits[i] = getBit(Bits[i]);
  }
  return BitsInit::get(NewBits);
}

std::string BitsInit::getAsString() const {
  std::string Result = "{ ";
  for (unsigned i = 0, e = getNumBits(); i != e; ++i) {
    if (i) Result += ", ";
    if (Init *Bit = getBit(e-i-1))
      Result += Bit->getAsString();
    else
      Result += "*";
  }
  return Result + " }";
}

// Fix bit initializer to preserve the behavior that bit reference from a unset
// bits initializer will resolve into VarBitInit to keep the field name and bit
// number used in targets with fixed insn length.
static Init *fixBitInit(const RecordVal *RV, Init *Before, Init *After) {
  if (RV || !isa<UnsetInit>(After))
    return After;
  return Before;
}

// resolveReferences - If there are any field references that refer to fields
// that have been filled in, we can propagate the values now.
//
Init *BitsInit::resolveReferences(Record &R, const RecordVal *RV) const {
  bool Changed = false;
  SmallVector<Init *, 16> NewBits(getNumBits());

  Init *CachedInit = nullptr;
  Init *CachedBitVar = nullptr;
  bool CachedBitVarChanged = false;

  for (unsigned i = 0, e = getNumBits(); i != e; ++i) {
    Init *CurBit = Bits[i];
    Init *CurBitVar = CurBit->getBitVar();

    NewBits[i] = CurBit;

    if (CurBitVar == CachedBitVar) {
      if (CachedBitVarChanged) {
        Init *Bit = CachedInit->getBit(CurBit->getBitNum());
        NewBits[i] = fixBitInit(RV, CurBit, Bit);
      }
      continue;
    }
    CachedBitVar = CurBitVar;
    CachedBitVarChanged = false;

    Init *B;
    do {
      B = CurBitVar;
      CurBitVar = CurBitVar->resolveReferences(R, RV);
      CachedBitVarChanged |= B != CurBitVar;
      Changed |= B != CurBitVar;
    } while (B != CurBitVar);
    CachedInit = CurBitVar;

    if (CachedBitVarChanged) {
      Init *Bit = CurBitVar->getBit(CurBit->getBitNum());
      NewBits[i] = fixBitInit(RV, CurBit, Bit);
    }
  }

  if (Changed)
    return BitsInit::get(NewBits);

  return const_cast<BitsInit *>(this);
}

IntInit *IntInit::get(int64_t V) {
  static DenseMap<int64_t, std::unique_ptr<IntInit>> ThePool;

  std::unique_ptr<IntInit> &I = ThePool[V];
  if (!I) I.reset(new IntInit(V));
  return I.get();
}

std::string IntInit::getAsString() const {
  return itostr(Value);
}

/// canFitInBitfield - Return true if the number of bits is large enough to hold
/// the integer value.
static bool canFitInBitfield(int64_t Value, unsigned NumBits) {
  // For example, with NumBits == 4, we permit Values from [-7 .. 15].
  return (NumBits >= sizeof(Value) * 8) ||
         (Value >> NumBits == 0) || (Value >> (NumBits-1) == -1);
}

Init *IntInit::convertInitializerTo(RecTy *Ty) const {
  if (isa<IntRecTy>(Ty))
    return const_cast<IntInit *>(this);

  if (isa<BitRecTy>(Ty)) {
    int64_t Val = getValue();
    if (Val != 0 && Val != 1) return nullptr;  // Only accept 0 or 1 for a bit!
    return BitInit::get(Val != 0);
  }

  if (auto *BRT = dyn_cast<BitsRecTy>(Ty)) {
    int64_t Value = getValue();
    // Make sure this bitfield is large enough to hold the integer value.
    if (!canFitInBitfield(Value, BRT->getNumBits()))
      return nullptr;

    SmallVector<Init *, 16> NewBits(BRT->getNumBits());
    for (unsigned i = 0; i != BRT->getNumBits(); ++i)
      NewBits[i] = BitInit::get(Value & (1LL << i));

    return BitsInit::get(NewBits);
  }

  return nullptr;
}

Init *
IntInit::convertInitializerBitRange(const std::vector<unsigned> &Bits) const {
  SmallVector<Init *, 16> NewBits(Bits.size());

  for (unsigned i = 0, e = Bits.size(); i != e; ++i) {
    if (Bits[i] >= 64)
      return nullptr;

    NewBits[i] = BitInit::get(Value & (INT64_C(1) << Bits[i]));
  }
  return BitsInit::get(NewBits);
}

StringInit *StringInit::get(StringRef V) {
  static StringMap<std::unique_ptr<StringInit>> ThePool;

  std::unique_ptr<StringInit> &I = ThePool[V];
  if (!I) I.reset(new StringInit(V));
  return I.get();
}

Init *StringInit::convertInitializerTo(RecTy *Ty) const {
  if (isa<StringRecTy>(Ty))
    return const_cast<StringInit *>(this);

  return nullptr;
}

static void ProfileListInit(FoldingSetNodeID &ID,
                            ArrayRef<Init *> Range,
                            RecTy *EltTy) {
  ID.AddInteger(Range.size());
  ID.AddPointer(EltTy);

  for (Init *I : Range)
    ID.AddPointer(I);
}

ListInit *ListInit::get(ArrayRef<Init *> Range, RecTy *EltTy) {
  static FoldingSet<ListInit> ThePool;
  static std::vector<std::unique_ptr<ListInit>> TheActualPool;

  FoldingSetNodeID ID;
  ProfileListInit(ID, Range, EltTy);

  void *IP = nullptr;
  if (ListInit *I = ThePool.FindNodeOrInsertPos(ID, IP))
    return I;

  ListInit *I = new ListInit(Range, EltTy);
  ThePool.InsertNode(I, IP);
  TheActualPool.push_back(std::unique_ptr<ListInit>(I));
  return I;
}

void ListInit::Profile(FoldingSetNodeID &ID) const {
  RecTy *EltTy = cast<ListRecTy>(getType())->getElementType();

  ProfileListInit(ID, Values, EltTy);
}

Init *ListInit::convertInitializerTo(RecTy *Ty) const {
  if (auto *LRT = dyn_cast<ListRecTy>(Ty)) {
    std::vector<Init*> Elements;

    // Verify that all of the elements of the list are subclasses of the
    // appropriate class!
    for (Init *I : getValues())
      if (Init *CI = I->convertInitializerTo(LRT->getElementType()))
        Elements.push_back(CI);
      else
        return nullptr;

    if (isa<ListRecTy>(getType()))
      return ListInit::get(Elements, Ty);
  }

  return nullptr;
}

Init *
ListInit::convertInitListSlice(const std::vector<unsigned> &Elements) const {
  std::vector<Init*> Vals;
  for (unsigned i = 0, e = Elements.size(); i != e; ++i) {
    if (Elements[i] >= size())
      return nullptr;
    Vals.push_back(getElement(Elements[i]));
  }
  return ListInit::get(Vals, getType());
}

Record *ListInit::getElementAsRecord(unsigned i) const {
  assert(i < Values.size() && "List element index out of range!");
  DefInit *DI = dyn_cast<DefInit>(Values[i]);
  if (!DI)
    PrintFatalError("Expected record in list!");
  return DI->getDef();
}

Init *ListInit::resolveReferences(Record &R, const RecordVal *RV) const {
  std::vector<Init*> Resolved;
  Resolved.reserve(size());
  bool Changed = false;

  for (Init *CurElt : getValues()) {
    Init *E;

    do {
      E = CurElt;
      CurElt = CurElt->resolveReferences(R, RV);
      Changed |= E != CurElt;
    } while (E != CurElt);
    Resolved.push_back(E);
  }

  if (Changed)
    return ListInit::get(Resolved, getType());
  return const_cast<ListInit *>(this);
}

Init *ListInit::resolveListElementReference(Record &R, const RecordVal *IRV,
                                            unsigned Elt) const {
  if (Elt >= size())
    return nullptr;  // Out of range reference.
  Init *E = getElement(Elt);
  // If the element is set to some value, or if we are resolving a reference
  // to a specific variable and that variable is explicitly unset, then
  // replace the VarListElementInit with it.
  if (IRV || !isa<UnsetInit>(E))
    return E;
  return nullptr;
}

std::string ListInit::getAsString() const {
  std::string Result = "[";
  for (unsigned i = 0, e = Values.size(); i != e; ++i) {
    if (i) Result += ", ";
    Result += Values[i]->getAsString();
  }
  return Result + "]";
}

Init *OpInit::resolveListElementReference(Record &R, const RecordVal *IRV,
                                          unsigned Elt) const {
  Init *Resolved = resolveReferences(R, IRV);
  OpInit *OResolved = dyn_cast<OpInit>(Resolved);
  if (OResolved) {
    Resolved = OResolved->Fold(&R, nullptr);
  }

  if (Resolved != this) {
    TypedInit *Typed = cast<TypedInit>(Resolved);
    if (Init *New = Typed->resolveListElementReference(R, IRV, Elt))
      return New;
    return VarListElementInit::get(Typed, Elt);
  }

  return nullptr;
}

Init *OpInit::getBit(unsigned Bit) const {
  if (getType() == BitRecTy::get())
    return const_cast<OpInit*>(this);
  return VarBitInit::get(const_cast<OpInit*>(this), Bit);
}

UnOpInit *UnOpInit::get(UnaryOp opc, Init *lhs, RecTy *Type) {
  typedef std::pair<std::pair<unsigned, Init *>, RecTy *> Key;
  static DenseMap<Key, std::unique_ptr<UnOpInit>> ThePool;

  Key TheKey(std::make_pair(std::make_pair(opc, lhs), Type));

  std::unique_ptr<UnOpInit> &I = ThePool[TheKey];
  if (!I) I.reset(new UnOpInit(opc, lhs, Type));
  return I.get();
}

Init *UnOpInit::Fold(Record *CurRec, MultiClass *CurMultiClass) const {
  switch (getOpcode()) {
  case CAST: {
    if (isa<StringRecTy>(getType())) {
      if (StringInit *LHSs = dyn_cast<StringInit>(LHS))
        return LHSs;

      if (DefInit *LHSd = dyn_cast<DefInit>(LHS))
        return StringInit::get(LHSd->getAsString());

      if (IntInit *LHSi = dyn_cast<IntInit>(LHS))
        return StringInit::get(LHSi->getAsString());
    } else {
      if (StringInit *LHSs = dyn_cast<StringInit>(LHS)) {
        std::string Name = LHSs->getValue();

        // From TGParser::ParseIDValue
        if (CurRec) {
          if (const RecordVal *RV = CurRec->getValue(Name)) {
            if (RV->getType() != getType())
              PrintFatalError("type mismatch in cast");
            return VarInit::get(Name, RV->getType());
          }

          Init *TemplateArgName = QualifyName(*CurRec, CurMultiClass, Name,
                                              ":");

          if (CurRec->isTemplateArg(TemplateArgName)) {
            const RecordVal *RV = CurRec->getValue(TemplateArgName);
            assert(RV && "Template arg doesn't exist??");

            if (RV->getType() != getType())
              PrintFatalError("type mismatch in cast");

            return VarInit::get(TemplateArgName, RV->getType());
          }
        }

        if (CurMultiClass) {
          Init *MCName = QualifyName(CurMultiClass->Rec, CurMultiClass, Name,
                                     "::");

          if (CurMultiClass->Rec.isTemplateArg(MCName)) {
            const RecordVal *RV = CurMultiClass->Rec.getValue(MCName);
            assert(RV && "Template arg doesn't exist??");

            if (RV->getType() != getType())
              PrintFatalError("type mismatch in cast");

            return VarInit::get(MCName, RV->getType());
          }
        }
        assert(CurRec && "NULL pointer");
        if (Record *D = (CurRec->getRecords()).getDef(Name))
          return DefInit::get(D);

        PrintFatalError(CurRec->getLoc(),
                        "Undefined reference:'" + Name + "'\n");
      }
    }
    break;
  }
  case HEAD: {
    if (ListInit *LHSl = dyn_cast<ListInit>(LHS)) {
      assert(!LHSl->empty() && "Empty list in head");
      return LHSl->getElement(0);
    }
    break;
  }
  case TAIL: {
    if (ListInit *LHSl = dyn_cast<ListInit>(LHS)) {
      assert(!LHSl->empty() && "Empty list in tail");
      // Note the +1.  We can't just pass the result of getValues()
      // directly.
      return ListInit::get(LHSl->getValues().slice(1), LHSl->getType());
    }
    break;
  }
  case EMPTY: {
    if (ListInit *LHSl = dyn_cast<ListInit>(LHS))
      return IntInit::get(LHSl->empty());
    if (StringInit *LHSs = dyn_cast<StringInit>(LHS))
      return IntInit::get(LHSs->getValue().empty());

    break;
  }
  }
  return const_cast<UnOpInit *>(this);
}

Init *UnOpInit::resolveReferences(Record &R, const RecordVal *RV) const {
  Init *lhs = LHS->resolveReferences(R, RV);

  if (LHS != lhs)
    return (UnOpInit::get(getOpcode(), lhs, getType()))->Fold(&R, nullptr);
  return Fold(&R, nullptr);
}

std::string UnOpInit::getAsString() const {
  std::string Result;
  switch (Opc) {
  case CAST: Result = "!cast<" + getType()->getAsString() + ">"; break;
  case HEAD: Result = "!head"; break;
  case TAIL: Result = "!tail"; break;
  case EMPTY: Result = "!empty"; break;
  }
  return Result + "(" + LHS->getAsString() + ")";
}

BinOpInit *BinOpInit::get(BinaryOp opc, Init *lhs,
                          Init *rhs, RecTy *Type) {
  typedef std::pair<
    std::pair<std::pair<unsigned, Init *>, Init *>,
    RecTy *
    > Key;

  static DenseMap<Key, std::unique_ptr<BinOpInit>> ThePool;

  Key TheKey(std::make_pair(std::make_pair(std::make_pair(opc, lhs), rhs),
                            Type));

  std::unique_ptr<BinOpInit> &I = ThePool[TheKey];
  if (!I) I.reset(new BinOpInit(opc, lhs, rhs, Type));
  return I.get();
}

Init *BinOpInit::Fold(Record *CurRec, MultiClass *CurMultiClass) const {
  switch (getOpcode()) {
  case CONCAT: {
    DagInit *LHSs = dyn_cast<DagInit>(LHS);
    DagInit *RHSs = dyn_cast<DagInit>(RHS);
    if (LHSs && RHSs) {
      DefInit *LOp = dyn_cast<DefInit>(LHSs->getOperator());
      DefInit *ROp = dyn_cast<DefInit>(RHSs->getOperator());
      if (!LOp || !ROp || LOp->getDef() != ROp->getDef())
        PrintFatalError("Concated Dag operators do not match!");
      std::vector<Init*> Args;
      std::vector<std::string> ArgNames;
      for (unsigned i = 0, e = LHSs->getNumArgs(); i != e; ++i) {
        Args.push_back(LHSs->getArg(i));
        ArgNames.push_back(LHSs->getArgName(i));
      }
      for (unsigned i = 0, e = RHSs->getNumArgs(); i != e; ++i) {
        Args.push_back(RHSs->getArg(i));
        ArgNames.push_back(RHSs->getArgName(i));
      }
      return DagInit::get(LHSs->getOperator(), "", Args, ArgNames);
    }
    break;
  }
  case LISTCONCAT: {
    ListInit *LHSs = dyn_cast<ListInit>(LHS);
    ListInit *RHSs = dyn_cast<ListInit>(RHS);
    if (LHSs && RHSs) {
      std::vector<Init *> Args;
      Args.insert(Args.end(), LHSs->begin(), LHSs->end());
      Args.insert(Args.end(), RHSs->begin(), RHSs->end());
      return ListInit::get(
          Args, cast<ListRecTy>(LHSs->getType())->getElementType());
    }
    break;
  }
  case STRCONCAT: {
    StringInit *LHSs = dyn_cast<StringInit>(LHS);
    StringInit *RHSs = dyn_cast<StringInit>(RHS);
    if (LHSs && RHSs)
      return StringInit::get(LHSs->getValue() + RHSs->getValue());
    break;
  }
  case EQ: {
    // try to fold eq comparison for 'bit' and 'int', otherwise fallback
    // to string objects.
    IntInit *L =
      dyn_cast_or_null<IntInit>(LHS->convertInitializerTo(IntRecTy::get()));
    IntInit *R =
      dyn_cast_or_null<IntInit>(RHS->convertInitializerTo(IntRecTy::get()));

    if (L && R)
      return IntInit::get(L->getValue() == R->getValue());

    StringInit *LHSs = dyn_cast<StringInit>(LHS);
    StringInit *RHSs = dyn_cast<StringInit>(RHS);

    // Make sure we've resolved
    if (LHSs && RHSs)
      return IntInit::get(LHSs->getValue() == RHSs->getValue());

    break;
  }
  case ADD:
  case AND:
  case SHL:
  case SRA:
  case SRL: {
    IntInit *LHSi =
      dyn_cast_or_null<IntInit>(LHS->convertInitializerTo(IntRecTy::get()));
    IntInit *RHSi =
      dyn_cast_or_null<IntInit>(RHS->convertInitializerTo(IntRecTy::get()));
    if (LHSi && RHSi) {
      int64_t LHSv = LHSi->getValue(), RHSv = RHSi->getValue();
      int64_t Result;
      switch (getOpcode()) {
      default: llvm_unreachable("Bad opcode!");
      case ADD: Result = LHSv +  RHSv; break;
      case AND: Result = LHSv &  RHSv; break;
      case SHL: Result = LHSv << RHSv; break;
      case SRA: Result = LHSv >> RHSv; break;
      case SRL: Result = (uint64_t)LHSv >> (uint64_t)RHSv; break;
      }
      return IntInit::get(Result);
    }
    break;
  }
  }
  return const_cast<BinOpInit *>(this);
}

Init *BinOpInit::resolveReferences(Record &R, const RecordVal *RV) const {
  Init *lhs = LHS->resolveReferences(R, RV);
  Init *rhs = RHS->resolveReferences(R, RV);

  if (LHS != lhs || RHS != rhs)
    return (BinOpInit::get(getOpcode(), lhs, rhs, getType()))->Fold(&R,nullptr);
  return Fold(&R, nullptr);
}

std::string BinOpInit::getAsString() const {
  std::string Result;
  switch (Opc) {
  case CONCAT: Result = "!con"; break;
  case ADD: Result = "!add"; break;
  case AND: Result = "!and"; break;
  case SHL: Result = "!shl"; break;
  case SRA: Result = "!sra"; break;
  case SRL: Result = "!srl"; break;
  case EQ: Result = "!eq"; break;
  case LISTCONCAT: Result = "!listconcat"; break;
  case STRCONCAT: Result = "!strconcat"; break;
  }
  return Result + "(" + LHS->getAsString() + ", " + RHS->getAsString() + ")";
}

TernOpInit *TernOpInit::get(TernaryOp opc, Init *lhs, Init *mhs, Init *rhs,
                            RecTy *Type) {
  typedef std::pair<
    std::pair<
      std::pair<std::pair<unsigned, RecTy *>, Init *>,
      Init *
      >,
    Init *
    > Key;

  static DenseMap<Key, std::unique_ptr<TernOpInit>> ThePool;

  Key TheKey(std::make_pair(std::make_pair(std::make_pair(std::make_pair(opc,
                                                                         Type),
                                                          lhs),
                                           mhs),
                            rhs));

  std::unique_ptr<TernOpInit> &I = ThePool[TheKey];
  if (!I) I.reset(new TernOpInit(opc, lhs, mhs, rhs, Type));
  return I.get();
}

static Init *ForeachHelper(Init *LHS, Init *MHS, Init *RHS, RecTy *Type,
                           Record *CurRec, MultiClass *CurMultiClass);

static Init *EvaluateOperation(OpInit *RHSo, Init *LHS, Init *Arg,
                               RecTy *Type, Record *CurRec,
                               MultiClass *CurMultiClass) {
  // If this is a dag, recurse
  if (auto *TArg = dyn_cast<TypedInit>(Arg))
    if (isa<DagRecTy>(TArg->getType()))
      return ForeachHelper(LHS, Arg, RHSo, Type, CurRec, CurMultiClass);

  std::vector<Init *> NewOperands;
  for (unsigned i = 0; i < RHSo->getNumOperands(); ++i) {
    if (auto *RHSoo = dyn_cast<OpInit>(RHSo->getOperand(i))) {
      if (Init *Result = EvaluateOperation(RHSoo, LHS, Arg,
                                           Type, CurRec, CurMultiClass))
        NewOperands.push_back(Result);
      else
        NewOperands.push_back(Arg);
    } else if (LHS->getAsString() == RHSo->getOperand(i)->getAsString()) {
      NewOperands.push_back(Arg);
    } else {
      NewOperands.push_back(RHSo->getOperand(i));
    }
  }

  // Now run the operator and use its result as the new leaf
  const OpInit *NewOp = RHSo->clone(NewOperands);
  Init *NewVal = NewOp->Fold(CurRec, CurMultiClass);
  return (NewVal != NewOp) ? NewVal : nullptr;
}

static Init *ForeachHelper(Init *LHS, Init *MHS, Init *RHS, RecTy *Type,
                           Record *CurRec, MultiClass *CurMultiClass) {

  OpInit *RHSo = dyn_cast<OpInit>(RHS);

  if (!RHSo)
    PrintFatalError(CurRec->getLoc(), "!foreach requires an operator\n");

  TypedInit *LHSt = dyn_cast<TypedInit>(LHS);

  if (!LHSt)
    PrintFatalError(CurRec->getLoc(), "!foreach requires typed variable\n");

  DagInit *MHSd = dyn_cast<DagInit>(MHS);
  if (MHSd && isa<DagRecTy>(Type)) {
    Init *Val = MHSd->getOperator();
    if (Init *Result = EvaluateOperation(RHSo, LHS, Val,
                                         Type, CurRec, CurMultiClass))
      Val = Result;

    std::vector<std::pair<Init *, std::string> > args;
    for (unsigned int i = 0; i < MHSd->getNumArgs(); ++i) {
      Init *Arg = MHSd->getArg(i);
      std::string ArgName = MHSd->getArgName(i);

      // Process args
      if (Init *Result = EvaluateOperation(RHSo, LHS, Arg, Type,
                                           CurRec, CurMultiClass))
        Arg = Result;

      // TODO: Process arg names
      args.push_back(std::make_pair(Arg, ArgName));
    }

    return DagInit::get(Val, "", args);
  }

  ListInit *MHSl = dyn_cast<ListInit>(MHS);
  if (MHSl && isa<ListRecTy>(Type)) {
    std::vector<Init *> NewOperands;
    std::vector<Init *> NewList(MHSl->begin(), MHSl->end());

    for (Init *&Item : NewList) {
      NewOperands.clear();
      for(unsigned i = 0; i < RHSo->getNumOperands(); ++i) {
        // First, replace the foreach variable with the list item
        if (LHS->getAsString() == RHSo->getOperand(i)->getAsString())
          NewOperands.push_back(Item);
        else
          NewOperands.push_back(RHSo->getOperand(i));
      }

      // Now run the operator and use its result as the new list item
      const OpInit *NewOp = RHSo->clone(NewOperands);
      Init *NewItem = NewOp->Fold(CurRec, CurMultiClass);
      if (NewItem != NewOp)
        Item = NewItem;
    }
    return ListInit::get(NewList, MHSl->getType());
  }
  return nullptr;
}

Init *TernOpInit::Fold(Record *CurRec, MultiClass *CurMultiClass) const {
  switch (getOpcode()) {
  case SUBST: {
    DefInit *LHSd = dyn_cast<DefInit>(LHS);
    VarInit *LHSv = dyn_cast<VarInit>(LHS);
    StringInit *LHSs = dyn_cast<StringInit>(LHS);

    DefInit *MHSd = dyn_cast<DefInit>(MHS);
    VarInit *MHSv = dyn_cast<VarInit>(MHS);
    StringInit *MHSs = dyn_cast<StringInit>(MHS);

    DefInit *RHSd = dyn_cast<DefInit>(RHS);
    VarInit *RHSv = dyn_cast<VarInit>(RHS);
    StringInit *RHSs = dyn_cast<StringInit>(RHS);

    if (LHSd && MHSd && RHSd) {
      Record *Val = RHSd->getDef();
      if (LHSd->getAsString() == RHSd->getAsString())
        Val = MHSd->getDef();
      return DefInit::get(Val);
    }
    if (LHSv && MHSv && RHSv) {
      std::string Val = RHSv->getName();
      if (LHSv->getAsString() == RHSv->getAsString())
        Val = MHSv->getName();
      return VarInit::get(Val, getType());
    }
    if (LHSs && MHSs && RHSs) {
      std::string Val = RHSs->getValue();

      std::string::size_type found;
      std::string::size_type idx = 0;
      while (true) {
        found = Val.find(LHSs->getValue(), idx);
        if (found == std::string::npos)
          break;
        Val.replace(found, LHSs->getValue().size(), MHSs->getValue());
        idx = found + MHSs->getValue().size();
      }

      return StringInit::get(Val);
    }
    break;
  }

  case FOREACH: {
    if (Init *Result = ForeachHelper(LHS, MHS, RHS, getType(),
                                     CurRec, CurMultiClass))
      return Result;
    break;
  }

  case IF: {
    IntInit *LHSi = dyn_cast<IntInit>(LHS);
    if (Init *I = LHS->convertInitializerTo(IntRecTy::get()))
      LHSi = dyn_cast<IntInit>(I);
    if (LHSi) {
      if (LHSi->getValue())
        return MHS;
      return RHS;
    }
    break;
  }
  }

  return const_cast<TernOpInit *>(this);
}

Init *TernOpInit::resolveReferences(Record &R,
                                    const RecordVal *RV) const {
  Init *lhs = LHS->resolveReferences(R, RV);

  if (Opc == IF && lhs != LHS) {
    IntInit *Value = dyn_cast<IntInit>(lhs);
    if (Init *I = lhs->convertInitializerTo(IntRecTy::get()))
      Value = dyn_cast<IntInit>(I);
    if (Value) {
      // Short-circuit
      if (Value->getValue()) {
        Init *mhs = MHS->resolveReferences(R, RV);
        return (TernOpInit::get(getOpcode(), lhs, mhs,
                                RHS, getType()))->Fold(&R, nullptr);
      }
      Init *rhs = RHS->resolveReferences(R, RV);
      return (TernOpInit::get(getOpcode(), lhs, MHS,
                              rhs, getType()))->Fold(&R, nullptr);
    }
  }

  Init *mhs = MHS->resolveReferences(R, RV);
  Init *rhs = RHS->resolveReferences(R, RV);

  if (LHS != lhs || MHS != mhs || RHS != rhs)
    return (TernOpInit::get(getOpcode(), lhs, mhs, rhs,
                            getType()))->Fold(&R, nullptr);
  return Fold(&R, nullptr);
}

std::string TernOpInit::getAsString() const {
  std::string Result;
  switch (Opc) {
  case SUBST: Result = "!subst"; break;
  case FOREACH: Result = "!foreach"; break;
  case IF: Result = "!if"; break;
  }
  return Result + "(" + LHS->getAsString() + ", " + MHS->getAsString() + ", " +
         RHS->getAsString() + ")";
}

RecTy *TypedInit::getFieldType(const std::string &FieldName) const {
  if (RecordRecTy *RecordType = dyn_cast<RecordRecTy>(getType()))
    if (RecordVal *Field = RecordType->getRecord()->getValue(FieldName))
      return Field->getType();
  return nullptr;
}

Init *
TypedInit::convertInitializerTo(RecTy *Ty) const {
  if (isa<IntRecTy>(Ty)) {
    if (getType()->typeIsConvertibleTo(Ty))
      return const_cast<TypedInit *>(this);
    return nullptr;
  }

  if (isa<StringRecTy>(Ty)) {
    if (isa<StringRecTy>(getType()))
      return const_cast<TypedInit *>(this);
    return nullptr;
  }

  if (isa<BitRecTy>(Ty)) {
    // Accept variable if it is already of bit type!
    if (isa<BitRecTy>(getType()))
      return const_cast<TypedInit *>(this);
    if (auto *BitsTy = dyn_cast<BitsRecTy>(getType())) {
      // Accept only bits<1> expression.
      if (BitsTy->getNumBits() == 1)
        return const_cast<TypedInit *>(this);
      return nullptr;
    }
    // Ternary !if can be converted to bit, but only if both sides are
    // convertible to a bit.
    if (const auto *TOI = dyn_cast<TernOpInit>(this)) {
      if (TOI->getOpcode() == TernOpInit::TernaryOp::IF &&
          TOI->getMHS()->convertInitializerTo(BitRecTy::get()) &&
          TOI->getRHS()->convertInitializerTo(BitRecTy::get()))
        return const_cast<TypedInit *>(this);
      return nullptr;
    }
    return nullptr;
  }

  if (auto *BRT = dyn_cast<BitsRecTy>(Ty)) {
    if (BRT->getNumBits() == 1 && isa<BitRecTy>(getType()))
      return BitsInit::get(const_cast<TypedInit *>(this));

    if (getType()->typeIsConvertibleTo(BRT)) {
      SmallVector<Init *, 16> NewBits(BRT->getNumBits());

      for (unsigned i = 0; i != BRT->getNumBits(); ++i)
        NewBits[i] = VarBitInit::get(const_cast<TypedInit *>(this), i);
      return BitsInit::get(NewBits);
    }

    return nullptr;
  }

  if (auto *DLRT = dyn_cast<ListRecTy>(Ty)) {
    if (auto *SLRT = dyn_cast<ListRecTy>(getType()))
      if (SLRT->getElementType()->typeIsConvertibleTo(DLRT->getElementType()))
        return const_cast<TypedInit *>(this);
    return nullptr;
  }

  if (auto *DRT = dyn_cast<DagRecTy>(Ty)) {
    if (getType()->typeIsConvertibleTo(DRT))
      return const_cast<TypedInit *>(this);
    return nullptr;
  }

  if (auto *SRRT = dyn_cast<RecordRecTy>(Ty)) {
    // Ensure that this is compatible with Rec.
    if (RecordRecTy *DRRT = dyn_cast<RecordRecTy>(getType()))
      if (DRRT->getRecord()->isSubClassOf(SRRT->getRecord()) ||
          DRRT->getRecord() == SRRT->getRecord())
        return const_cast<TypedInit *>(this);
    return nullptr;
  }

  return nullptr;
}

Init *
TypedInit::convertInitializerBitRange(const std::vector<unsigned> &Bits) const {
  BitsRecTy *T = dyn_cast<BitsRecTy>(getType());
  if (!T) return nullptr;  // Cannot subscript a non-bits variable.
  unsigned NumBits = T->getNumBits();

  SmallVector<Init *, 16> NewBits(Bits.size());
  for (unsigned i = 0, e = Bits.size(); i != e; ++i) {
    if (Bits[i] >= NumBits)
      return nullptr;

    NewBits[i] = VarBitInit::get(const_cast<TypedInit *>(this), Bits[i]);
  }
  return BitsInit::get(NewBits);
}

Init *
TypedInit::convertInitListSlice(const std::vector<unsigned> &Elements) const {
  ListRecTy *T = dyn_cast<ListRecTy>(getType());
  if (!T) return nullptr;  // Cannot subscript a non-list variable.

  if (Elements.size() == 1)
    return VarListElementInit::get(const_cast<TypedInit *>(this), Elements[0]);

  std::vector<Init*> ListInits;
  ListInits.reserve(Elements.size());
  for (unsigned i = 0, e = Elements.size(); i != e; ++i)
    ListInits.push_back(VarListElementInit::get(const_cast<TypedInit *>(this),
                                                Elements[i]));
  return ListInit::get(ListInits, T);
}


VarInit *VarInit::get(const std::string &VN, RecTy *T) {
  Init *Value = StringInit::get(VN);
  return VarInit::get(Value, T);
}

VarInit *VarInit::get(Init *VN, RecTy *T) {
  typedef std::pair<RecTy *, Init *> Key;
  static DenseMap<Key, std::unique_ptr<VarInit>> ThePool;

  Key TheKey(std::make_pair(T, VN));

  std::unique_ptr<VarInit> &I = ThePool[TheKey];
  if (!I) I.reset(new VarInit(VN, T));
  return I.get();
}

const std::string &VarInit::getName() const {
  StringInit *NameString = cast<StringInit>(getNameInit());
  return NameString->getValue();
}

Init *VarInit::getBit(unsigned Bit) const {
  if (getType() == BitRecTy::get())
    return const_cast<VarInit*>(this);
  return VarBitInit::get(const_cast<VarInit*>(this), Bit);
}

Init *VarInit::resolveListElementReference(Record &R,
                                           const RecordVal *IRV,
                                           unsigned Elt) const {
  if (R.isTemplateArg(getNameInit())) return nullptr;
  if (IRV && IRV->getNameInit() != getNameInit()) return nullptr;

  RecordVal *RV = R.getValue(getNameInit());
  assert(RV && "Reference to a non-existent variable?");
  ListInit *LI = dyn_cast<ListInit>(RV->getValue());
  if (!LI)
    return VarListElementInit::get(cast<TypedInit>(RV->getValue()), Elt);

  if (Elt >= LI->size())
    return nullptr;  // Out of range reference.
  Init *E = LI->getElement(Elt);
  // If the element is set to some value, or if we are resolving a reference
  // to a specific variable and that variable is explicitly unset, then
  // replace the VarListElementInit with it.
  if (IRV || !isa<UnsetInit>(E))
    return E;
  return nullptr;
}


RecTy *VarInit::getFieldType(const std::string &FieldName) const {
  if (RecordRecTy *RTy = dyn_cast<RecordRecTy>(getType()))
    if (const RecordVal *RV = RTy->getRecord()->getValue(FieldName))
      return RV->getType();
  return nullptr;
}

Init *VarInit::getFieldInit(Record &R, const RecordVal *RV,
                            const std::string &FieldName) const {
  if (isa<RecordRecTy>(getType()))
    if (const RecordVal *Val = R.getValue(VarName)) {
      if (RV != Val && (RV || isa<UnsetInit>(Val->getValue())))
        return nullptr;
      Init *TheInit = Val->getValue();
      assert(TheInit != this && "Infinite loop detected!");
      if (Init *I = TheInit->getFieldInit(R, RV, FieldName))
        return I;
      return nullptr;
    }
  return nullptr;
}

/// resolveReferences - This method is used by classes that refer to other
/// variables which may not be defined at the time the expression is formed.
/// If a value is set for the variable later, this method will be called on
/// users of the value to allow the value to propagate out.
///
Init *VarInit::resolveReferences(Record &R, const RecordVal *RV) const {
  if (RecordVal *Val = R.getValue(VarName))
    if (RV == Val || (!RV && !isa<UnsetInit>(Val->getValue())))
      return Val->getValue();
  return const_cast<VarInit *>(this);
}

VarBitInit *VarBitInit::get(TypedInit *T, unsigned B) {
  typedef std::pair<TypedInit *, unsigned> Key;
  static DenseMap<Key, std::unique_ptr<VarBitInit>> ThePool;

  Key TheKey(std::make_pair(T, B));

  std::unique_ptr<VarBitInit> &I = ThePool[TheKey];
  if (!I) I.reset(new VarBitInit(T, B));
  return I.get();
}

Init *VarBitInit::convertInitializerTo(RecTy *Ty) const {
  if (isa<BitRecTy>(Ty))
    return const_cast<VarBitInit *>(this);

  return nullptr;
}

std::string VarBitInit::getAsString() const {
  return TI->getAsString() + "{" + utostr(Bit) + "}";
}

Init *VarBitInit::resolveReferences(Record &R, const RecordVal *RV) const {
  Init *I = TI->resolveReferences(R, RV);
  if (TI != I)
    return I->getBit(getBitNum());

  return const_cast<VarBitInit*>(this);
}

VarListElementInit *VarListElementInit::get(TypedInit *T,
                                            unsigned E) {
  typedef std::pair<TypedInit *, unsigned> Key;
  static DenseMap<Key, std::unique_ptr<VarListElementInit>> ThePool;

  Key TheKey(std::make_pair(T, E));

  std::unique_ptr<VarListElementInit> &I = ThePool[TheKey];
  if (!I) I.reset(new VarListElementInit(T, E));
  return I.get();
}

std::string VarListElementInit::getAsString() const {
  return TI->getAsString() + "[" + utostr(Element) + "]";
}

Init *
VarListElementInit::resolveReferences(Record &R, const RecordVal *RV) const {
  if (Init *I = getVariable()->resolveListElementReference(R, RV,
                                                           getElementNum()))
    return I;
  return const_cast<VarListElementInit *>(this);
}

Init *VarListElementInit::getBit(unsigned Bit) const {
  if (getType() == BitRecTy::get())
    return const_cast<VarListElementInit*>(this);
  return VarBitInit::get(const_cast<VarListElementInit*>(this), Bit);
}

Init *VarListElementInit:: resolveListElementReference(Record &R,
                                                       const RecordVal *RV,
                                                       unsigned Elt) const {
  if (Init *Result = TI->resolveListElementReference(R, RV, Element)) {
    if (TypedInit *TInit = dyn_cast<TypedInit>(Result)) {
      if (Init *Result2 = TInit->resolveListElementReference(R, RV, Elt))
        return Result2;
      return VarListElementInit::get(TInit, Elt);
    }
    return Result;
  }

  return nullptr;
}

DefInit *DefInit::get(Record *R) {
  return R->getDefInit();
}

Init *DefInit::convertInitializerTo(RecTy *Ty) const {
  if (auto *RRT = dyn_cast<RecordRecTy>(Ty))
    if (getDef()->isSubClassOf(RRT->getRecord()))
      return const_cast<DefInit *>(this);
  return nullptr;
}

RecTy *DefInit::getFieldType(const std::string &FieldName) const {
  if (const RecordVal *RV = Def->getValue(FieldName))
    return RV->getType();
  return nullptr;
}

Init *DefInit::getFieldInit(Record &R, const RecordVal *RV,
                            const std::string &FieldName) const {
  return Def->getValue(FieldName)->getValue();
}


std::string DefInit::getAsString() const {
  return Def->getName();
}

FieldInit *FieldInit::get(Init *R, const std::string &FN) {
  typedef std::pair<Init *, TableGenStringKey> Key;
  static DenseMap<Key, std::unique_ptr<FieldInit>> ThePool;

  Key TheKey(std::make_pair(R, FN));

  std::unique_ptr<FieldInit> &I = ThePool[TheKey];
  if (!I) I.reset(new FieldInit(R, FN));
  return I.get();
}

Init *FieldInit::getBit(unsigned Bit) const {
  if (getType() == BitRecTy::get())
    return const_cast<FieldInit*>(this);
  return VarBitInit::get(const_cast<FieldInit*>(this), Bit);
}

Init *FieldInit::resolveListElementReference(Record &R, const RecordVal *RV,
                                             unsigned Elt) const {
  if (Init *ListVal = Rec->getFieldInit(R, RV, FieldName))
    if (ListInit *LI = dyn_cast<ListInit>(ListVal)) {
      if (Elt >= LI->size()) return nullptr;
      Init *E = LI->getElement(Elt);

      // If the element is set to some value, or if we are resolving a
      // reference to a specific variable and that variable is explicitly
      // unset, then replace the VarListElementInit with it.
      if (RV || !isa<UnsetInit>(E))
        return E;
    }
  return nullptr;
}

Init *FieldInit::resolveReferences(Record &R, const RecordVal *RV) const {
  Init *NewRec = RV ? Rec->resolveReferences(R, RV) : Rec;

  if (Init *BitsVal = NewRec->getFieldInit(R, RV, FieldName)) {
    Init *BVR = BitsVal->resolveReferences(R, RV);
    return BVR->isComplete() ? BVR : const_cast<FieldInit *>(this);
  }

  if (NewRec != Rec)
    return FieldInit::get(NewRec, FieldName);
  return const_cast<FieldInit *>(this);
}

static void ProfileDagInit(FoldingSetNodeID &ID, Init *V, const std::string &VN,
                           ArrayRef<Init *> ArgRange,
                           ArrayRef<std::string> NameRange) {
  ID.AddPointer(V);
  ID.AddString(VN);

  ArrayRef<Init *>::iterator Arg  = ArgRange.begin();
  ArrayRef<std::string>::iterator  Name = NameRange.begin();
  while (Arg != ArgRange.end()) {
    assert(Name != NameRange.end() && "Arg name underflow!");
    ID.AddPointer(*Arg++);
    ID.AddString(*Name++);
  }
  assert(Name == NameRange.end() && "Arg name overflow!");
}

DagInit *
DagInit::get(Init *V, const std::string &VN,
             ArrayRef<Init *> ArgRange,
             ArrayRef<std::string> NameRange) {
  static FoldingSet<DagInit> ThePool;
  static std::vector<std::unique_ptr<DagInit>> TheActualPool;

  FoldingSetNodeID ID;
  ProfileDagInit(ID, V, VN, ArgRange, NameRange);

  void *IP = nullptr;
  if (DagInit *I = ThePool.FindNodeOrInsertPos(ID, IP))
    return I;

  DagInit *I = new DagInit(V, VN, ArgRange, NameRange);
  ThePool.InsertNode(I, IP);
  TheActualPool.push_back(std::unique_ptr<DagInit>(I));
  return I;
}

DagInit *
DagInit::get(Init *V, const std::string &VN,
             const std::vector<std::pair<Init*, std::string> > &args) {
  std::vector<Init *> Args;
  std::vector<std::string> Names;

  for (const auto &Arg : args) {
    Args.push_back(Arg.first);
    Names.push_back(Arg.second);
  }

  return DagInit::get(V, VN, Args, Names);
}

void DagInit::Profile(FoldingSetNodeID &ID) const {
  ProfileDagInit(ID, Val, ValName, Args, ArgNames);
}

Init *DagInit::convertInitializerTo(RecTy *Ty) const {
  if (isa<DagRecTy>(Ty))
    return const_cast<DagInit *>(this);

  return nullptr;
}

Init *DagInit::resolveReferences(Record &R, const RecordVal *RV) const {
  std::vector<Init*> NewArgs;
  for (unsigned i = 0, e = Args.size(); i != e; ++i)
    NewArgs.push_back(Args[i]->resolveReferences(R, RV));

  Init *Op = Val->resolveReferences(R, RV);

  if (Args != NewArgs || Op != Val)
    return DagInit::get(Op, ValName, NewArgs, ArgNames);

  return const_cast<DagInit *>(this);
}


std::string DagInit::getAsString() const {
  std::string Result = "(" + Val->getAsString();
  if (!ValName.empty())
    Result += ":" + ValName;
  if (!Args.empty()) {
    Result += " " + Args[0]->getAsString();
    if (!ArgNames[0].empty()) Result += ":$" + ArgNames[0];
    for (unsigned i = 1, e = Args.size(); i != e; ++i) {
      Result += ", " + Args[i]->getAsString();
      if (!ArgNames[i].empty()) Result += ":$" + ArgNames[i];
    }
  }
  return Result + ")";
}


//===----------------------------------------------------------------------===//
//    Other implementations
//===----------------------------------------------------------------------===//

RecordVal::RecordVal(Init *N, RecTy *T, bool P)
  : NameAndPrefix(N, P), Ty(T) {
  Value = UnsetInit::get()->convertInitializerTo(Ty);
  assert(Value && "Cannot create unset value for current type!");
}

RecordVal::RecordVal(const std::string &N, RecTy *T, bool P)
  : NameAndPrefix(StringInit::get(N), P), Ty(T) {
  Value = UnsetInit::get()->convertInitializerTo(Ty);
  assert(Value && "Cannot create unset value for current type!");
}

const std::string &RecordVal::getName() const {
  return cast<StringInit>(getNameInit())->getValue();
}

void RecordVal::dump() const { errs() << *this; }

void RecordVal::print(raw_ostream &OS, bool PrintSem) const {
  if (getPrefix()) OS << "field ";
  OS << *getType() << " " << getNameInitAsString();

  if (getValue())
    OS << " = " << *getValue();

  if (PrintSem) OS << ";\n";
}

unsigned Record::LastID = 0;

void Record::init() {
  checkName();

  // Every record potentially has a def at the top.  This value is
  // replaced with the top-level def name at instantiation time.
  RecordVal DN("NAME", StringRecTy::get(), 0);
  addValue(DN);
}

void Record::checkName() {
  // Ensure the record name has string type.
  const TypedInit *TypedName = cast<const TypedInit>(Name);
  if (!isa<StringRecTy>(TypedName->getType()))
    PrintFatalError(getLoc(), "Record name is not a string!");
}

DefInit *Record::getDefInit() {
  if (!TheInit)
    TheInit.reset(new DefInit(this, new RecordRecTy(this)));
  return TheInit.get();
}

const std::string &Record::getName() const {
  return cast<StringInit>(Name)->getValue();
}

void Record::setName(Init *NewName) {
  Name = NewName;
  checkName();
  // DO NOT resolve record values to the name at this point because
  // there might be default values for arguments of this def.  Those
  // arguments might not have been resolved yet so we don't want to
  // prematurely assume values for those arguments were not passed to
  // this def.
  //
  // Nonetheless, it may be that some of this Record's values
  // reference the record name.  Indeed, the reason for having the
  // record name be an Init is to provide this flexibility.  The extra
  // resolve steps after completely instantiating defs takes care of
  // this.  See TGParser::ParseDef and TGParser::ParseDefm.
}

void Record::setName(const std::string &Name) {
  setName(StringInit::get(Name));
}

/// resolveReferencesTo - If anything in this record refers to RV, replace the
/// reference to RV with the RHS of RV.  If RV is null, we resolve all possible
/// references.
void Record::resolveReferencesTo(const RecordVal *RV) {
  for (unsigned i = 0, e = Values.size(); i != e; ++i) {
    if (RV == &Values[i]) // Skip resolve the same field as the given one
      continue;
    if (Init *V = Values[i].getValue())
      if (Values[i].setValue(V->resolveReferences(*this, RV)))
        PrintFatalError(getLoc(), "Invalid value is found when setting '" +
                        Values[i].getNameInitAsString() +
                        "' after resolving references" +
                        (RV ? " against '" + RV->getNameInitAsString() +
                              "' of (" + RV->getValue()->getAsUnquotedString() +
                              ")"
                            : "") + "\n");
  }
  Init *OldName = getNameInit();
  Init *NewName = Name->resolveReferences(*this, RV);
  if (NewName != OldName) {
    // Re-register with RecordKeeper.
    setName(NewName);
  }
}

void Record::dump() const { errs() << *this; }

raw_ostream &llvm::operator<<(raw_ostream &OS, const Record &R) {
  OS << R.getNameInitAsString();

  const std::vector<Init *> &TArgs = R.getTemplateArgs();
  if (!TArgs.empty()) {
    OS << "<";
    bool NeedComma = false;
    for (const Init *TA : TArgs) {
      if (NeedComma) OS << ", ";
      NeedComma = true;
      const RecordVal *RV = R.getValue(TA);
      assert(RV && "Template argument record not found??");
      RV->print(OS, false);
    }
    OS << ">";
  }

  OS << " {";
  ArrayRef<Record *> SC = R.getSuperClasses();
  if (!SC.empty()) {
    OS << "\t//";
    for (const Record *Super : SC)
      OS << " " << Super->getNameInitAsString();
  }
  OS << "\n";

  for (const RecordVal &Val : R.getValues())
    if (Val.getPrefix() && !R.isTemplateArg(Val.getName()))
      OS << Val;
  for (const RecordVal &Val : R.getValues())
    if (!Val.getPrefix() && !R.isTemplateArg(Val.getName()))
      OS << Val;

  return OS << "}\n";
}

/// getValueInit - Return the initializer for a value with the specified name,
/// or abort if the field does not exist.
///
Init *Record::getValueInit(StringRef FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (!R || !R->getValue())
    PrintFatalError(getLoc(), "Record `" + getName() +
      "' does not have a field named `" + FieldName + "'!\n");
  return R->getValue();
}


/// getValueAsString - This method looks up the specified field and returns its
/// value as a string, aborts if the field does not exist or if
/// the value is not a string.
///
std::string Record::getValueAsString(StringRef FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (!R || !R->getValue())
    PrintFatalError(getLoc(), "Record `" + getName() +
      "' does not have a field named `" + FieldName + "'!\n");

  if (StringInit *SI = dyn_cast<StringInit>(R->getValue()))
    return SI->getValue();
  PrintFatalError(getLoc(), "Record `" + getName() + "', field `" +
    FieldName + "' does not have a string initializer!");
}

/// getValueAsBitsInit - This method looks up the specified field and returns
/// its value as a BitsInit, aborts if the field does not exist or if
/// the value is not the right type.
///
BitsInit *Record::getValueAsBitsInit(StringRef FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (!R || !R->getValue())
    PrintFatalError(getLoc(), "Record `" + getName() +
      "' does not have a field named `" + FieldName + "'!\n");

  if (BitsInit *BI = dyn_cast<BitsInit>(R->getValue()))
    return BI;
  PrintFatalError(getLoc(), "Record `" + getName() + "', field `" +
    FieldName + "' does not have a BitsInit initializer!");
}

/// getValueAsListInit - This method looks up the specified field and returns
/// its value as a ListInit, aborting if the field does not exist or if
/// the value is not the right type.
///
ListInit *Record::getValueAsListInit(StringRef FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (!R || !R->getValue())
    PrintFatalError(getLoc(), "Record `" + getName() +
      "' does not have a field named `" + FieldName + "'!\n");

  if (ListInit *LI = dyn_cast<ListInit>(R->getValue()))
    return LI;
  PrintFatalError(getLoc(), "Record `" + getName() + "', field `" +
    FieldName + "' does not have a list initializer!");
}

/// getValueAsListOfDefs - This method looks up the specified field and returns
/// its value as a vector of records, aborting if the field does not exist
/// or if the value is not the right type.
///
std::vector<Record*>
Record::getValueAsListOfDefs(StringRef FieldName) const {
  ListInit *List = getValueAsListInit(FieldName);
  std::vector<Record*> Defs;
  for (Init *I : List->getValues()) {
    if (DefInit *DI = dyn_cast<DefInit>(I))
      Defs.push_back(DI->getDef());
    else
      PrintFatalError(getLoc(), "Record `" + getName() + "', field `" +
        FieldName + "' list is not entirely DefInit!");
  }
  return Defs;
}

/// getValueAsInt - This method looks up the specified field and returns its
/// value as an int64_t, aborting if the field does not exist or if the value
/// is not the right type.
///
int64_t Record::getValueAsInt(StringRef FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (!R || !R->getValue())
    PrintFatalError(getLoc(), "Record `" + getName() +
      "' does not have a field named `" + FieldName + "'!\n");

  if (IntInit *II = dyn_cast<IntInit>(R->getValue()))
    return II->getValue();
  PrintFatalError(getLoc(), "Record `" + getName() + "', field `" +
    FieldName + "' does not have an int initializer!");
}

/// getValueAsListOfInts - This method looks up the specified field and returns
/// its value as a vector of integers, aborting if the field does not exist or
/// if the value is not the right type.
///
std::vector<int64_t>
Record::getValueAsListOfInts(StringRef FieldName) const {
  ListInit *List = getValueAsListInit(FieldName);
  std::vector<int64_t> Ints;
  for (Init *I : List->getValues()) {
    if (IntInit *II = dyn_cast<IntInit>(I))
      Ints.push_back(II->getValue());
    else
      PrintFatalError(getLoc(), "Record `" + getName() + "', field `" +
        FieldName + "' does not have a list of ints initializer!");
  }
  return Ints;
}

/// getValueAsListOfStrings - This method looks up the specified field and
/// returns its value as a vector of strings, aborting if the field does not
/// exist or if the value is not the right type.
///
std::vector<std::string>
Record::getValueAsListOfStrings(StringRef FieldName) const {
  ListInit *List = getValueAsListInit(FieldName);
  std::vector<std::string> Strings;
  for (Init *I : List->getValues()) {
    if (StringInit *SI = dyn_cast<StringInit>(I))
      Strings.push_back(SI->getValue());
    else
      PrintFatalError(getLoc(), "Record `" + getName() + "', field `" +
        FieldName + "' does not have a list of strings initializer!");
  }
  return Strings;
}

/// getValueAsDef - This method looks up the specified field and returns its
/// value as a Record, aborting if the field does not exist or if the value
/// is not the right type.
///
Record *Record::getValueAsDef(StringRef FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (!R || !R->getValue())
    PrintFatalError(getLoc(), "Record `" + getName() +
      "' does not have a field named `" + FieldName + "'!\n");

  if (DefInit *DI = dyn_cast<DefInit>(R->getValue()))
    return DI->getDef();
  PrintFatalError(getLoc(), "Record `" + getName() + "', field `" +
    FieldName + "' does not have a def initializer!");
}

/// getValueAsBit - This method looks up the specified field and returns its
/// value as a bit, aborting if the field does not exist or if the value is
/// not the right type.
///
bool Record::getValueAsBit(StringRef FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (!R || !R->getValue())
    PrintFatalError(getLoc(), "Record `" + getName() +
      "' does not have a field named `" + FieldName + "'!\n");

  if (BitInit *BI = dyn_cast<BitInit>(R->getValue()))
    return BI->getValue();
  PrintFatalError(getLoc(), "Record `" + getName() + "', field `" +
    FieldName + "' does not have a bit initializer!");
}

bool Record::getValueAsBitOrUnset(StringRef FieldName, bool &Unset) const {
  const RecordVal *R = getValue(FieldName);
  if (!R || !R->getValue())
    PrintFatalError(getLoc(), "Record `" + getName() +
      "' does not have a field named `" + FieldName.str() + "'!\n");

  if (isa<UnsetInit>(R->getValue())) {
    Unset = true;
    return false;
  }
  Unset = false;
  if (BitInit *BI = dyn_cast<BitInit>(R->getValue()))
    return BI->getValue();
  PrintFatalError(getLoc(), "Record `" + getName() + "', field `" +
    FieldName + "' does not have a bit initializer!");
}

/// getValueAsDag - This method looks up the specified field and returns its
/// value as an Dag, aborting if the field does not exist or if the value is
/// not the right type.
///
DagInit *Record::getValueAsDag(StringRef FieldName) const {
  const RecordVal *R = getValue(FieldName);
  if (!R || !R->getValue())
    PrintFatalError(getLoc(), "Record `" + getName() +
      "' does not have a field named `" + FieldName + "'!\n");

  if (DagInit *DI = dyn_cast<DagInit>(R->getValue()))
    return DI;
  PrintFatalError(getLoc(), "Record `" + getName() + "', field `" +
    FieldName + "' does not have a dag initializer!");
}


void MultiClass::dump() const {
  errs() << "Record:\n";
  Rec.dump();

  errs() << "Defs:\n";
  for (const auto &Proto : DefPrototypes)
    Proto->dump();
}


void RecordKeeper::dump() const { errs() << *this; }

raw_ostream &llvm::operator<<(raw_ostream &OS, const RecordKeeper &RK) {
  OS << "------------- Classes -----------------\n";
  for (const auto &C : RK.getClasses())
    OS << "class " << *C.second;

  OS << "------------- Defs -----------------\n";
  for (const auto &D : RK.getDefs())
    OS << "def " << *D.second;
  return OS;
}


/// getAllDerivedDefinitions - This method returns all concrete definitions
/// that derive from the specified class name.  If a class with the specified
/// name does not exist, an error is printed and true is returned.
std::vector<Record*>
RecordKeeper::getAllDerivedDefinitions(const std::string &ClassName) const {
  Record *Class = getClass(ClassName);
  if (!Class)
    PrintFatalError("ERROR: Couldn't find the `" + ClassName + "' class!\n");

  std::vector<Record*> Defs;
  for (const auto &D : getDefs())
    if (D.second->isSubClassOf(Class))
      Defs.push_back(D.second.get());

  return Defs;
}

/// QualifyName - Return an Init with a qualifier prefix referring
/// to CurRec's name.
Init *llvm::QualifyName(Record &CurRec, MultiClass *CurMultiClass,
                        Init *Name, const std::string &Scoper) {
  RecTy *Type = cast<TypedInit>(Name)->getType();

  BinOpInit *NewName =
    BinOpInit::get(BinOpInit::STRCONCAT,
                   BinOpInit::get(BinOpInit::STRCONCAT,
                                  CurRec.getNameInit(),
                                  StringInit::get(Scoper),
                                  Type)->Fold(&CurRec, CurMultiClass),
                   Name,
                   Type);

  if (CurMultiClass && Scoper != "::") {
    NewName =
      BinOpInit::get(BinOpInit::STRCONCAT,
                     BinOpInit::get(BinOpInit::STRCONCAT,
                                    CurMultiClass->Rec.getNameInit(),
                                    StringInit::get("::"),
                                    Type)->Fold(&CurRec, CurMultiClass),
                     NewName->Fold(&CurRec, CurMultiClass),
                     Type);
  }

  return NewName->Fold(&CurRec, CurMultiClass);
}

/// QualifyName - Return an Init with a qualifier prefix referring
/// to CurRec's name.
Init *llvm::QualifyName(Record &CurRec, MultiClass *CurMultiClass,
                        const std::string &Name,
                        const std::string &Scoper) {
  return QualifyName(CurRec, CurMultiClass, StringInit::get(Name), Scoper);
}
