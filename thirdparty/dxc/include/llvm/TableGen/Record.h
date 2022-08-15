//===- llvm/TableGen/Record.h - Classes for Table Records -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the main TableGen data structures, including the TableGen
// types, values, and high-level data structures.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TABLEGEN_RECORD_H
#define LLVM_TABLEGEN_RECORD_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"
#include <map>

namespace llvm {

class ListRecTy;
struct MultiClass;
class Record;
class RecordVal;
class RecordKeeper;

//===----------------------------------------------------------------------===//
//  Type Classes
//===----------------------------------------------------------------------===//

class RecTy {
public:
  /// \brief Subclass discriminator (for dyn_cast<> et al.)
  enum RecTyKind {
    BitRecTyKind,
    BitsRecTyKind,
    IntRecTyKind,
    StringRecTyKind,
    ListRecTyKind,
    DagRecTyKind,
    RecordRecTyKind
  };

private:
  RecTyKind Kind;
  std::unique_ptr<ListRecTy> ListTy;

public:
  RecTyKind getRecTyKind() const { return Kind; }

  RecTy(RecTyKind K) : Kind(K) {}
  virtual ~RecTy() {}

  virtual std::string getAsString() const = 0;
  void print(raw_ostream &OS) const { OS << getAsString(); }
  void dump() const;

  /// typeIsConvertibleTo - Return true if all values of 'this' type can be
  /// converted to the specified type.
  virtual bool typeIsConvertibleTo(const RecTy *RHS) const;

  /// getListTy - Returns the type representing list<this>.
  ListRecTy *getListTy();
};

inline raw_ostream &operator<<(raw_ostream &OS, const RecTy &Ty) {
  Ty.print(OS);
  return OS;
}

/// BitRecTy - 'bit' - Represent a single bit
///
class BitRecTy : public RecTy {
  static BitRecTy Shared;
  BitRecTy() : RecTy(BitRecTyKind) {}

public:
  static bool classof(const RecTy *RT) {
    return RT->getRecTyKind() == BitRecTyKind;
  }

  static BitRecTy *get() { return &Shared; }

  std::string getAsString() const override { return "bit"; }

  bool typeIsConvertibleTo(const RecTy *RHS) const override;
};

/// BitsRecTy - 'bits<n>' - Represent a fixed number of bits
///
class BitsRecTy : public RecTy {
  unsigned Size;
  explicit BitsRecTy(unsigned Sz) : RecTy(BitsRecTyKind), Size(Sz) {}

public:
  static bool classof(const RecTy *RT) {
    return RT->getRecTyKind() == BitsRecTyKind;
  }

  static BitsRecTy *get(unsigned Sz);

  unsigned getNumBits() const { return Size; }

  std::string getAsString() const override;

  bool typeIsConvertibleTo(const RecTy *RHS) const override;
};

/// IntRecTy - 'int' - Represent an integer value of no particular size
///
class IntRecTy : public RecTy {
  static IntRecTy Shared;
  IntRecTy() : RecTy(IntRecTyKind) {}

public:
  static bool classof(const RecTy *RT) {
    return RT->getRecTyKind() == IntRecTyKind;
  }

  static IntRecTy *get() { return &Shared; }

  std::string getAsString() const override { return "int"; }

  bool typeIsConvertibleTo(const RecTy *RHS) const override;
};

/// StringRecTy - 'string' - Represent an string value
///
class StringRecTy : public RecTy {
  static StringRecTy Shared;
  StringRecTy() : RecTy(StringRecTyKind) {}

public:
  static bool classof(const RecTy *RT) {
    return RT->getRecTyKind() == StringRecTyKind;
  }

  static StringRecTy *get() { return &Shared; }

  std::string getAsString() const override;
};

/// ListRecTy - 'list<Ty>' - Represent a list of values, all of which must be of
/// the specified type.
///
class ListRecTy : public RecTy {
  RecTy *Ty;
  explicit ListRecTy(RecTy *T) : RecTy(ListRecTyKind), Ty(T) {}
  friend ListRecTy *RecTy::getListTy();

public:
  static bool classof(const RecTy *RT) {
    return RT->getRecTyKind() == ListRecTyKind;
  }

  static ListRecTy *get(RecTy *T) { return T->getListTy(); }
  RecTy *getElementType() const { return Ty; }

  std::string getAsString() const override;

  bool typeIsConvertibleTo(const RecTy *RHS) const override;
};

/// DagRecTy - 'dag' - Represent a dag fragment
///
class DagRecTy : public RecTy {
  static DagRecTy Shared;
  DagRecTy() : RecTy(DagRecTyKind) {}

public:
  static bool classof(const RecTy *RT) {
    return RT->getRecTyKind() == DagRecTyKind;
  }

  static DagRecTy *get() { return &Shared; }

  std::string getAsString() const override;
};

/// RecordRecTy - '[classname]' - Represent an instance of a class, such as:
/// (R32 X = EAX).
///
class RecordRecTy : public RecTy {
  Record *Rec;
  explicit RecordRecTy(Record *R) : RecTy(RecordRecTyKind), Rec(R) {}
  friend class Record;

public:
  static bool classof(const RecTy *RT) {
    return RT->getRecTyKind() == RecordRecTyKind;
  }

  static RecordRecTy *get(Record *R);

  Record *getRecord() const { return Rec; }

  std::string getAsString() const override;

  bool typeIsConvertibleTo(const RecTy *RHS) const override;
};

/// resolveTypes - Find a common type that T1 and T2 convert to.
/// Return 0 if no such type exists.
///
RecTy *resolveTypes(RecTy *T1, RecTy *T2);

//===----------------------------------------------------------------------===//
//  Initializer Classes
//===----------------------------------------------------------------------===//

class Init {
protected:
  /// \brief Discriminator enum (for isa<>, dyn_cast<>, et al.)
  ///
  /// This enum is laid out by a preorder traversal of the inheritance
  /// hierarchy, and does not contain an entry for abstract classes, as per
  /// the recommendation in docs/HowToSetUpLLVMStyleRTTI.rst.
  ///
  /// We also explicitly include "first" and "last" values for each
  /// interior node of the inheritance tree, to make it easier to read the
  /// corresponding classof().
  ///
  /// We could pack these a bit tighter by not having the IK_FirstXXXInit
  /// and IK_LastXXXInit be their own values, but that would degrade
  /// readability for really no benefit.
  enum InitKind {
    IK_BitInit,
    IK_FirstTypedInit,
    IK_BitsInit,
    IK_DagInit,
    IK_DefInit,
    IK_FieldInit,
    IK_IntInit,
    IK_ListInit,
    IK_FirstOpInit,
    IK_BinOpInit,
    IK_TernOpInit,
    IK_UnOpInit,
    IK_LastOpInit,
    IK_StringInit,
    IK_VarInit,
    IK_VarListElementInit,
    IK_LastTypedInit,
    IK_UnsetInit,
    IK_VarBitInit
  };

private:
  const InitKind Kind;
  Init(const Init &) = delete;
  Init &operator=(const Init &) = delete;
  virtual void anchor();

public:
  InitKind getKind() const { return Kind; }

protected:
  explicit Init(InitKind K) : Kind(K) {}

public:
  virtual ~Init() {}

  /// isComplete - This virtual method should be overridden by values that may
  /// not be completely specified yet.
  virtual bool isComplete() const { return true; }

  /// print - Print out this value.
  void print(raw_ostream &OS) const { OS << getAsString(); }

  /// getAsString - Convert this value to a string form.
  virtual std::string getAsString() const = 0;
  /// getAsUnquotedString - Convert this value to a string form,
  /// without adding quote markers.  This primaruly affects
  /// StringInits where we will not surround the string value with
  /// quotes.
  virtual std::string getAsUnquotedString() const { return getAsString(); }

  /// dump - Debugging method that may be called through a debugger, just
  /// invokes print on stderr.
  void dump() const;

  /// convertInitializerTo - This virtual function converts to the appropriate
  /// Init based on the passed in type.
  virtual Init *convertInitializerTo(RecTy *Ty) const = 0;

  /// convertInitializerBitRange - This method is used to implement the bitrange
  /// selection operator.  Given an initializer, it selects the specified bits
  /// out, returning them as a new init of bits type.  If it is not legal to use
  /// the bit subscript operator on this initializer, return null.
  ///
  virtual Init *
  convertInitializerBitRange(const std::vector<unsigned> &Bits) const {
    return nullptr;
  }

  /// convertInitListSlice - This method is used to implement the list slice
  /// selection operator.  Given an initializer, it selects the specified list
  /// elements, returning them as a new init of list type.  If it is not legal
  /// to take a slice of this, return null.
  ///
  virtual Init *
  convertInitListSlice(const std::vector<unsigned> &Elements) const {
    return nullptr;
  }

  /// getFieldType - This method is used to implement the FieldInit class.
  /// Implementors of this method should return the type of the named field if
  /// they are of record type.
  ///
  virtual RecTy *getFieldType(const std::string &FieldName) const {
    return nullptr;
  }

  /// getFieldInit - This method complements getFieldType to return the
  /// initializer for the specified field.  If getFieldType returns non-null
  /// this method should return non-null, otherwise it returns null.
  ///
  virtual Init *getFieldInit(Record &R, const RecordVal *RV,
                             const std::string &FieldName) const {
    return nullptr;
  }

  /// resolveReferences - This method is used by classes that refer to other
  /// variables which may not be defined at the time the expression is formed.
  /// If a value is set for the variable later, this method will be called on
  /// users of the value to allow the value to propagate out.
  ///
  virtual Init *resolveReferences(Record &R, const RecordVal *RV) const {
    return const_cast<Init *>(this);
  }

  /// getBit - This method is used to return the initializer for the specified
  /// bit.
  virtual Init *getBit(unsigned Bit) const = 0;

  /// getBitVar - This method is used to retrieve the initializer for bit
  /// reference. For non-VarBitInit, it simply returns itself.
  virtual Init *getBitVar() const { return const_cast<Init*>(this); }

  /// getBitNum - This method is used to retrieve the bit number of a bit
  /// reference. For non-VarBitInit, it simply returns 0.
  virtual unsigned getBitNum() const { return 0; }
};

inline raw_ostream &operator<<(raw_ostream &OS, const Init &I) {
  I.print(OS); return OS;
}

/// TypedInit - This is the common super-class of types that have a specific,
/// explicit, type.
///
class TypedInit : public Init {
  RecTy *Ty;

  TypedInit(const TypedInit &Other) = delete;
  TypedInit &operator=(const TypedInit &Other) = delete;

protected:
  explicit TypedInit(InitKind K, RecTy *T) : Init(K), Ty(T) {}
  ~TypedInit() {
    // If this is a DefInit we need to delete the RecordRecTy.
    if (getKind() == IK_DefInit)
      delete Ty;
  }

public:
  static bool classof(const Init *I) {
    return I->getKind() >= IK_FirstTypedInit &&
           I->getKind() <= IK_LastTypedInit;
  }
  RecTy *getType() const { return Ty; }

  Init *convertInitializerTo(RecTy *Ty) const override;

  Init *
  convertInitializerBitRange(const std::vector<unsigned> &Bits) const override;
  Init *
  convertInitListSlice(const std::vector<unsigned> &Elements) const override;

  /// getFieldType - This method is used to implement the FieldInit class.
  /// Implementors of this method should return the type of the named field if
  /// they are of record type.
  ///
  RecTy *getFieldType(const std::string &FieldName) const override;

  /// resolveListElementReference - This method is used to implement
  /// VarListElementInit::resolveReferences.  If the list element is resolvable
  /// now, we return the resolved value, otherwise we return null.
  virtual Init *resolveListElementReference(Record &R, const RecordVal *RV,
                                            unsigned Elt) const = 0;
};

/// UnsetInit - ? - Represents an uninitialized value
///
class UnsetInit : public Init {
  UnsetInit() : Init(IK_UnsetInit) {}
  UnsetInit(const UnsetInit &) = delete;
  UnsetInit &operator=(const UnsetInit &Other) = delete;

public:
  static bool classof(const Init *I) {
    return I->getKind() == IK_UnsetInit;
  }
  static UnsetInit *get();

  Init *convertInitializerTo(RecTy *Ty) const override;

  Init *getBit(unsigned Bit) const override {
    return const_cast<UnsetInit*>(this);
  }

  bool isComplete() const override { return false; }
  std::string getAsString() const override { return "?"; }
};

/// BitInit - true/false - Represent a concrete initializer for a bit.
///
class BitInit : public Init {
  bool Value;

  explicit BitInit(bool V) : Init(IK_BitInit), Value(V) {}
  BitInit(const BitInit &Other) = delete;
  BitInit &operator=(BitInit &Other) = delete;

public:
  static bool classof(const Init *I) {
    return I->getKind() == IK_BitInit;
  }
  static BitInit *get(bool V);

  bool getValue() const { return Value; }

  Init *convertInitializerTo(RecTy *Ty) const override;

  Init *getBit(unsigned Bit) const override {
    assert(Bit < 1 && "Bit index out of range!");
    return const_cast<BitInit*>(this);
  }

  std::string getAsString() const override { return Value ? "1" : "0"; }
};

/// BitsInit - { a, b, c } - Represents an initializer for a BitsRecTy value.
/// It contains a vector of bits, whose size is determined by the type.
///
class BitsInit : public TypedInit, public FoldingSetNode {
  std::vector<Init*> Bits;

  BitsInit(ArrayRef<Init *> Range)
    : TypedInit(IK_BitsInit, BitsRecTy::get(Range.size())),
      Bits(Range.begin(), Range.end()) {}

  BitsInit(const BitsInit &Other) = delete;
  BitsInit &operator=(const BitsInit &Other) = delete;

public:
  static bool classof(const Init *I) {
    return I->getKind() == IK_BitsInit;
  }
  static BitsInit *get(ArrayRef<Init *> Range);

  void Profile(FoldingSetNodeID &ID) const;

  unsigned getNumBits() const { return Bits.size(); }

  Init *convertInitializerTo(RecTy *Ty) const override;
  Init *
  convertInitializerBitRange(const std::vector<unsigned> &Bits) const override;

  bool isComplete() const override {
    for (unsigned i = 0; i != getNumBits(); ++i)
      if (!getBit(i)->isComplete()) return false;
    return true;
  }
  bool allInComplete() const {
    for (unsigned i = 0; i != getNumBits(); ++i)
      if (getBit(i)->isComplete()) return false;
    return true;
  }
  std::string getAsString() const override;

  /// resolveListElementReference - This method is used to implement
  /// VarListElementInit::resolveReferences.  If the list element is resolvable
  /// now, we return the resolved value, otherwise we return null.
  Init *resolveListElementReference(Record &R, const RecordVal *RV,
                                    unsigned Elt) const override {
    llvm_unreachable("Illegal element reference off bits<n>");
  }

  Init *resolveReferences(Record &R, const RecordVal *RV) const override;

  Init *getBit(unsigned Bit) const override {
    assert(Bit < Bits.size() && "Bit index out of range!");
    return Bits[Bit];
  }
};

/// IntInit - 7 - Represent an initialization by a literal integer value.
///
class IntInit : public TypedInit {
  int64_t Value;

  explicit IntInit(int64_t V)
    : TypedInit(IK_IntInit, IntRecTy::get()), Value(V) {}

  IntInit(const IntInit &Other) = delete;
  IntInit &operator=(const IntInit &Other) = delete;

public:
  static bool classof(const Init *I) {
    return I->getKind() == IK_IntInit;
  }
  static IntInit *get(int64_t V);

  int64_t getValue() const { return Value; }

  Init *convertInitializerTo(RecTy *Ty) const override;
  Init *
  convertInitializerBitRange(const std::vector<unsigned> &Bits) const override;

  std::string getAsString() const override;

  /// resolveListElementReference - This method is used to implement
  /// VarListElementInit::resolveReferences.  If the list element is resolvable
  /// now, we return the resolved value, otherwise we return null.
  Init *resolveListElementReference(Record &R, const RecordVal *RV,
                                    unsigned Elt) const override {
    llvm_unreachable("Illegal element reference off int");
  }

  Init *getBit(unsigned Bit) const override {
    return BitInit::get((Value & (1ULL << Bit)) != 0);
  }
};

/// StringInit - "foo" - Represent an initialization by a string value.
///
class StringInit : public TypedInit {
  std::string Value;

  explicit StringInit(const std::string &V)
    : TypedInit(IK_StringInit, StringRecTy::get()), Value(V) {}

  StringInit(const StringInit &Other) = delete;
  StringInit &operator=(const StringInit &Other) = delete;

public:
  static bool classof(const Init *I) {
    return I->getKind() == IK_StringInit;
  }
  static StringInit *get(StringRef);

  const std::string &getValue() const { return Value; }

  Init *convertInitializerTo(RecTy *Ty) const override;

  std::string getAsString() const override { return "\"" + Value + "\""; }
  std::string getAsUnquotedString() const override { return Value; }

  /// resolveListElementReference - This method is used to implement
  /// VarListElementInit::resolveReferences.  If the list element is resolvable
  /// now, we return the resolved value, otherwise we return null.
  Init *resolveListElementReference(Record &R, const RecordVal *RV,
                                    unsigned Elt) const override {
    llvm_unreachable("Illegal element reference off string");
  }

  Init *getBit(unsigned Bit) const override {
    llvm_unreachable("Illegal bit reference off string");
  }
};

/// ListInit - [AL, AH, CL] - Represent a list of defs
///
class ListInit : public TypedInit, public FoldingSetNode {
  std::vector<Init*> Values;

public:
  typedef std::vector<Init*>::const_iterator const_iterator;

private:
  explicit ListInit(ArrayRef<Init *> Range, RecTy *EltTy)
    : TypedInit(IK_ListInit, ListRecTy::get(EltTy)),
      Values(Range.begin(), Range.end()) {}

  ListInit(const ListInit &Other) = delete;
  ListInit &operator=(const ListInit &Other) = delete;

public:
  static bool classof(const Init *I) {
    return I->getKind() == IK_ListInit;
  }
  static ListInit *get(ArrayRef<Init *> Range, RecTy *EltTy);

  void Profile(FoldingSetNodeID &ID) const;

  Init *getElement(unsigned i) const {
    assert(i < Values.size() && "List element index out of range!");
    return Values[i];
  }

  Record *getElementAsRecord(unsigned i) const;

  Init *
    convertInitListSlice(const std::vector<unsigned> &Elements) const override;

  Init *convertInitializerTo(RecTy *Ty) const override;

  /// resolveReferences - This method is used by classes that refer to other
  /// variables which may not be defined at the time they expression is formed.
  /// If a value is set for the variable later, this method will be called on
  /// users of the value to allow the value to propagate out.
  ///
  Init *resolveReferences(Record &R, const RecordVal *RV) const override;

  std::string getAsString() const override;

  ArrayRef<Init*> getValues() const { return Values; }

  const_iterator begin() const { return Values.begin(); }
  const_iterator end  () const { return Values.end();   }

  size_t         size () const { return Values.size();  }
  bool           empty() const { return Values.empty(); }

  /// resolveListElementReference - This method is used to implement
  /// VarListElementInit::resolveReferences.  If the list element is resolvable
  /// now, we return the resolved value, otherwise we return null.
  Init *resolveListElementReference(Record &R, const RecordVal *RV,
                                    unsigned Elt) const override;

  Init *getBit(unsigned Bit) const override {
    llvm_unreachable("Illegal bit reference off list");
  }
};

/// OpInit - Base class for operators
///
class OpInit : public TypedInit {
  OpInit(const OpInit &Other) = delete;
  OpInit &operator=(OpInit &Other) = delete;

protected:
  explicit OpInit(InitKind K, RecTy *Type) : TypedInit(K, Type) {}

public:
  static bool classof(const Init *I) {
    return I->getKind() >= IK_FirstOpInit &&
           I->getKind() <= IK_LastOpInit;
  }
  // Clone - Clone this operator, replacing arguments with the new list
  virtual OpInit *clone(std::vector<Init *> &Operands) const = 0;

  virtual unsigned getNumOperands() const = 0;
  virtual Init *getOperand(unsigned i) const = 0;

  // Fold - If possible, fold this to a simpler init.  Return this if not
  // possible to fold.
  virtual Init *Fold(Record *CurRec, MultiClass *CurMultiClass) const = 0;

  Init *resolveListElementReference(Record &R, const RecordVal *RV,
                                    unsigned Elt) const override;

  Init *getBit(unsigned Bit) const override;
};

/// UnOpInit - !op (X) - Transform an init.
///
class UnOpInit : public OpInit {
public:
  enum UnaryOp { CAST, HEAD, TAIL, EMPTY };

private:
  UnaryOp Opc;
  Init *LHS;

  UnOpInit(UnaryOp opc, Init *lhs, RecTy *Type)
    : OpInit(IK_UnOpInit, Type), Opc(opc), LHS(lhs) {}

  UnOpInit(const UnOpInit &Other) = delete;
  UnOpInit &operator=(const UnOpInit &Other) = delete;

public:
  static bool classof(const Init *I) {
    return I->getKind() == IK_UnOpInit;
  }
  static UnOpInit *get(UnaryOp opc, Init *lhs, RecTy *Type);

  // Clone - Clone this operator, replacing arguments with the new list
  OpInit *clone(std::vector<Init *> &Operands) const override {
    assert(Operands.size() == 1 &&
           "Wrong number of operands for unary operation");
    return UnOpInit::get(getOpcode(), *Operands.begin(), getType());
  }

  unsigned getNumOperands() const override { return 1; }
  Init *getOperand(unsigned i) const override {
    assert(i == 0 && "Invalid operand id for unary operator");
    return getOperand();
  }

  UnaryOp getOpcode() const { return Opc; }
  Init *getOperand() const { return LHS; }

  // Fold - If possible, fold this to a simpler init.  Return this if not
  // possible to fold.
  Init *Fold(Record *CurRec, MultiClass *CurMultiClass) const override;

  Init *resolveReferences(Record &R, const RecordVal *RV) const override;

  std::string getAsString() const override;
};

/// BinOpInit - !op (X, Y) - Combine two inits.
///
class BinOpInit : public OpInit {
public:
  enum BinaryOp { ADD, AND, SHL, SRA, SRL, LISTCONCAT, STRCONCAT, CONCAT, EQ };

private:
  BinaryOp Opc;
  Init *LHS, *RHS;

  BinOpInit(BinaryOp opc, Init *lhs, Init *rhs, RecTy *Type) :
      OpInit(IK_BinOpInit, Type), Opc(opc), LHS(lhs), RHS(rhs) {}

  BinOpInit(const BinOpInit &Other) = delete;
  BinOpInit &operator=(const BinOpInit &Other) = delete;

public:
  static bool classof(const Init *I) {
    return I->getKind() == IK_BinOpInit;
  }
  static BinOpInit *get(BinaryOp opc, Init *lhs, Init *rhs,
                        RecTy *Type);

  // Clone - Clone this operator, replacing arguments with the new list
  OpInit *clone(std::vector<Init *> &Operands) const override {
    assert(Operands.size() == 2 &&
           "Wrong number of operands for binary operation");
    return BinOpInit::get(getOpcode(), Operands[0], Operands[1], getType());
  }

  unsigned getNumOperands() const override { return 2; }
  Init *getOperand(unsigned i) const override {
    switch (i) {
    default: llvm_unreachable("Invalid operand id for binary operator");
    case 0: return getLHS();
    case 1: return getRHS();
    }
  }

  BinaryOp getOpcode() const { return Opc; }
  Init *getLHS() const { return LHS; }
  Init *getRHS() const { return RHS; }

  // Fold - If possible, fold this to a simpler init.  Return this if not
  // possible to fold.
  Init *Fold(Record *CurRec, MultiClass *CurMultiClass) const override;

  Init *resolveReferences(Record &R, const RecordVal *RV) const override;

  std::string getAsString() const override;
};

/// TernOpInit - !op (X, Y, Z) - Combine two inits.
///
class TernOpInit : public OpInit {
public:
  enum TernaryOp { SUBST, FOREACH, IF };

private:
  TernaryOp Opc;
  Init *LHS, *MHS, *RHS;

  TernOpInit(TernaryOp opc, Init *lhs, Init *mhs, Init *rhs,
             RecTy *Type) :
      OpInit(IK_TernOpInit, Type), Opc(opc), LHS(lhs), MHS(mhs), RHS(rhs) {}

  TernOpInit(const TernOpInit &Other) = delete;
  TernOpInit &operator=(const TernOpInit &Other) = delete;

public:
  static bool classof(const Init *I) {
    return I->getKind() == IK_TernOpInit;
  }
  static TernOpInit *get(TernaryOp opc, Init *lhs,
                         Init *mhs, Init *rhs,
                         RecTy *Type);

  // Clone - Clone this operator, replacing arguments with the new list
  OpInit *clone(std::vector<Init *> &Operands) const override {
    assert(Operands.size() == 3 &&
           "Wrong number of operands for ternary operation");
    return TernOpInit::get(getOpcode(), Operands[0], Operands[1], Operands[2],
                           getType());
  }

  unsigned getNumOperands() const override { return 3; }
  Init *getOperand(unsigned i) const override {
    switch (i) {
    default: llvm_unreachable("Invalid operand id for ternary operator");
    case 0: return getLHS();
    case 1: return getMHS();
    case 2: return getRHS();
    }
  }

  TernaryOp getOpcode() const { return Opc; }
  Init *getLHS() const { return LHS; }
  Init *getMHS() const { return MHS; }
  Init *getRHS() const { return RHS; }

  // Fold - If possible, fold this to a simpler init.  Return this if not
  // possible to fold.
  Init *Fold(Record *CurRec, MultiClass *CurMultiClass) const override;

  bool isComplete() const override { return false; }

  Init *resolveReferences(Record &R, const RecordVal *RV) const override;

  std::string getAsString() const override;
};

/// VarInit - 'Opcode' - Represent a reference to an entire variable object.
///
class VarInit : public TypedInit {
  Init *VarName;

  explicit VarInit(const std::string &VN, RecTy *T)
      : TypedInit(IK_VarInit, T), VarName(StringInit::get(VN)) {}
  explicit VarInit(Init *VN, RecTy *T)
      : TypedInit(IK_VarInit, T), VarName(VN) {}

  VarInit(const VarInit &Other) = delete;
  VarInit &operator=(const VarInit &Other) = delete;

public:
  static bool classof(const Init *I) {
    return I->getKind() == IK_VarInit;
  }
  static VarInit *get(const std::string &VN, RecTy *T);
  static VarInit *get(Init *VN, RecTy *T);

  const std::string &getName() const;
  Init *getNameInit() const { return VarName; }
  std::string getNameInitAsString() const {
    return getNameInit()->getAsUnquotedString();
  }

  Init *resolveListElementReference(Record &R, const RecordVal *RV,
                                    unsigned Elt) const override;

  RecTy *getFieldType(const std::string &FieldName) const override;
  Init *getFieldInit(Record &R, const RecordVal *RV,
                     const std::string &FieldName) const override;

  /// resolveReferences - This method is used by classes that refer to other
  /// variables which may not be defined at the time they expression is formed.
  /// If a value is set for the variable later, this method will be called on
  /// users of the value to allow the value to propagate out.
  ///
  Init *resolveReferences(Record &R, const RecordVal *RV) const override;

  Init *getBit(unsigned Bit) const override;

  std::string getAsString() const override { return getName(); }
};

/// VarBitInit - Opcode{0} - Represent access to one bit of a variable or field.
///
class VarBitInit : public Init {
  TypedInit *TI;
  unsigned Bit;

  VarBitInit(TypedInit *T, unsigned B) : Init(IK_VarBitInit), TI(T), Bit(B) {
    assert(T->getType() &&
           (isa<IntRecTy>(T->getType()) ||
            (isa<BitsRecTy>(T->getType()) &&
             cast<BitsRecTy>(T->getType())->getNumBits() > B)) &&
           "Illegal VarBitInit expression!");
  }

  VarBitInit(const VarBitInit &Other) = delete;
  VarBitInit &operator=(const VarBitInit &Other) = delete;

public:
  static bool classof(const Init *I) {
    return I->getKind() == IK_VarBitInit;
  }
  static VarBitInit *get(TypedInit *T, unsigned B);

  Init *convertInitializerTo(RecTy *Ty) const override;

  Init *getBitVar() const override { return TI; }
  unsigned getBitNum() const override { return Bit; }

  std::string getAsString() const override;
  Init *resolveReferences(Record &R, const RecordVal *RV) const override;

  Init *getBit(unsigned B) const override {
    assert(B < 1 && "Bit index out of range!");
    return const_cast<VarBitInit*>(this);
  }
};

/// VarListElementInit - List[4] - Represent access to one element of a var or
/// field.
class VarListElementInit : public TypedInit {
  TypedInit *TI;
  unsigned Element;

  VarListElementInit(TypedInit *T, unsigned E)
      : TypedInit(IK_VarListElementInit,
                  cast<ListRecTy>(T->getType())->getElementType()),
        TI(T), Element(E) {
    assert(T->getType() && isa<ListRecTy>(T->getType()) &&
           "Illegal VarBitInit expression!");
  }

  VarListElementInit(const VarListElementInit &Other) = delete;
  void operator=(const VarListElementInit &Other) = delete;

public:
  static bool classof(const Init *I) {
    return I->getKind() == IK_VarListElementInit;
  }
  static VarListElementInit *get(TypedInit *T, unsigned E);

  TypedInit *getVariable() const { return TI; }
  unsigned getElementNum() const { return Element; }

  /// resolveListElementReference - This method is used to implement
  /// VarListElementInit::resolveReferences.  If the list element is resolvable
  /// now, we return the resolved value, otherwise we return null.
  Init *resolveListElementReference(Record &R, const RecordVal *RV,
                                    unsigned Elt) const override;

  std::string getAsString() const override;
  Init *resolveReferences(Record &R, const RecordVal *RV) const override;

  Init *getBit(unsigned Bit) const override;
};

/// DefInit - AL - Represent a reference to a 'def' in the description
///
class DefInit : public TypedInit {
  Record *Def;

  DefInit(Record *D, RecordRecTy *T) : TypedInit(IK_DefInit, T), Def(D) {}
  friend class Record;

  DefInit(const DefInit &Other) = delete;
  DefInit &operator=(const DefInit &Other) = delete;

public:
  static bool classof(const Init *I) {
    return I->getKind() == IK_DefInit;
  }
  static DefInit *get(Record*);

  Init *convertInitializerTo(RecTy *Ty) const override;

  Record *getDef() const { return Def; }

  //virtual Init *convertInitializerBitRange(const std::vector<unsigned> &Bits);

  RecTy *getFieldType(const std::string &FieldName) const override;
  Init *getFieldInit(Record &R, const RecordVal *RV,
                     const std::string &FieldName) const override;

  std::string getAsString() const override;

  Init *getBit(unsigned Bit) const override {
    llvm_unreachable("Illegal bit reference off def");
  }

  /// resolveListElementReference - This method is used to implement
  /// VarListElementInit::resolveReferences.  If the list element is resolvable
  /// now, we return the resolved value, otherwise we return null.
  Init *resolveListElementReference(Record &R, const RecordVal *RV,
                                    unsigned Elt) const override {
    llvm_unreachable("Illegal element reference off def");
  }
};

/// FieldInit - X.Y - Represent a reference to a subfield of a variable
///
class FieldInit : public TypedInit {
  Init *Rec;                // Record we are referring to
  std::string FieldName;    // Field we are accessing

  FieldInit(Init *R, const std::string &FN)
      : TypedInit(IK_FieldInit, R->getFieldType(FN)), Rec(R), FieldName(FN) {
    assert(getType() && "FieldInit with non-record type!");
  }

  FieldInit(const FieldInit &Other) = delete;
  FieldInit &operator=(const FieldInit &Other) = delete;

public:
  static bool classof(const Init *I) {
    return I->getKind() == IK_FieldInit;
  }
  static FieldInit *get(Init *R, const std::string &FN);

  Init *getBit(unsigned Bit) const override;

  Init *resolveListElementReference(Record &R, const RecordVal *RV,
                                    unsigned Elt) const override;

  Init *resolveReferences(Record &R, const RecordVal *RV) const override;

  std::string getAsString() const override {
    return Rec->getAsString() + "." + FieldName;
  }
};

/// DagInit - (v a, b) - Represent a DAG tree value.  DAG inits are required
/// to have at least one value then a (possibly empty) list of arguments.  Each
/// argument can have a name associated with it.
///
class DagInit : public TypedInit, public FoldingSetNode {
  Init *Val;
  std::string ValName;
  std::vector<Init*> Args;
  std::vector<std::string> ArgNames;

  DagInit(Init *V, const std::string &VN,
          ArrayRef<Init *> ArgRange,
          ArrayRef<std::string> NameRange)
      : TypedInit(IK_DagInit, DagRecTy::get()), Val(V), ValName(VN),
          Args(ArgRange.begin(), ArgRange.end()),
          ArgNames(NameRange.begin(), NameRange.end()) {}

  DagInit(const DagInit &Other) = delete;
  DagInit &operator=(const DagInit &Other) = delete;

public:
  static bool classof(const Init *I) {
    return I->getKind() == IK_DagInit;
  }
  static DagInit *get(Init *V, const std::string &VN,
                      ArrayRef<Init *> ArgRange,
                      ArrayRef<std::string> NameRange);
  static DagInit *get(Init *V, const std::string &VN,
                      const std::vector<
                        std::pair<Init*, std::string> > &args);

  void Profile(FoldingSetNodeID &ID) const;

  Init *convertInitializerTo(RecTy *Ty) const override;

  Init *getOperator() const { return Val; }

  const std::string &getName() const { return ValName; }

  unsigned getNumArgs() const { return Args.size(); }
  Init *getArg(unsigned Num) const {
    assert(Num < Args.size() && "Arg number out of range!");
    return Args[Num];
  }
  const std::string &getArgName(unsigned Num) const {
    assert(Num < ArgNames.size() && "Arg number out of range!");
    return ArgNames[Num];
  }

  Init *resolveReferences(Record &R, const RecordVal *RV) const override;

  std::string getAsString() const override;

  typedef std::vector<Init*>::const_iterator       const_arg_iterator;
  typedef std::vector<std::string>::const_iterator const_name_iterator;

  inline const_arg_iterator  arg_begin() const { return Args.begin(); }
  inline const_arg_iterator  arg_end  () const { return Args.end();   }

  inline size_t              arg_size () const { return Args.size();  }
  inline bool                arg_empty() const { return Args.empty(); }

  inline const_name_iterator name_begin() const { return ArgNames.begin(); }
  inline const_name_iterator name_end  () const { return ArgNames.end();   }

  inline size_t              name_size () const { return ArgNames.size();  }
  inline bool                name_empty() const { return ArgNames.empty(); }

  Init *getBit(unsigned Bit) const override {
    llvm_unreachable("Illegal bit reference off dag");
  }

  Init *resolveListElementReference(Record &R, const RecordVal *RV,
                                    unsigned Elt) const override {
    llvm_unreachable("Illegal element reference off dag");
  }
};

//===----------------------------------------------------------------------===//
//  High-Level Classes
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

class RecordVal {
  PointerIntPair<Init *, 1, bool> NameAndPrefix;
  RecTy *Ty;
  Init *Value;

public:
  RecordVal(Init *N, RecTy *T, bool P);
  RecordVal(const std::string &N, RecTy *T, bool P);

  const std::string &getName() const;
  const Init *getNameInit() const { return NameAndPrefix.getPointer(); }
  std::string getNameInitAsString() const {
    return getNameInit()->getAsUnquotedString();
  }

  bool getPrefix() const { return NameAndPrefix.getInt(); }
  RecTy *getType() const { return Ty; }
  Init *getValue() const { return Value; }

  bool setValue(Init *V) {
    if (V) {
      Value = V->convertInitializerTo(Ty);
      return Value == nullptr;
    }
    Value = nullptr;
    return false;
  }

  void dump() const;
  void print(raw_ostream &OS, bool PrintSem = true) const;
};

inline raw_ostream &operator<<(raw_ostream &OS, const RecordVal &RV) {
  RV.print(OS << "  ");
  return OS;
}

class Record {
  static unsigned LastID;

  // Unique record ID.
  unsigned ID;
  Init *Name;
  // Location where record was instantiated, followed by the location of
  // multiclass prototypes used.
  SmallVector<SMLoc, 4> Locs;
  std::vector<Init *> TemplateArgs;
  std::vector<RecordVal> Values;
  std::vector<Record *> SuperClasses;
  std::vector<SMRange> SuperClassRanges;

  // Tracks Record instances. Not owned by Record.
  RecordKeeper &TrackedRecords;

  std::unique_ptr<DefInit> TheInit;
  bool IsAnonymous;

  // Class-instance values can be used by other defs.  For example, Struct<i>
  // is used here as a template argument to another class:
  //
  //   multiclass MultiClass<int i> {
  //     def Def : Class<Struct<i>>;
  //
  // These need to get fully resolved before instantiating any other
  // definitions that use them (e.g. Def).  However, inside a multiclass they
  // can't be immediately resolved so we mark them ResolveFirst to fully
  // resolve them later as soon as the multiclass is instantiated.
  bool ResolveFirst;

  void init();
  void checkName();

public:
  // Constructs a record.
  explicit Record(Init *N, ArrayRef<SMLoc> locs, RecordKeeper &records,
                  bool Anonymous = false) :
    ID(LastID++), Name(N), Locs(locs.begin(), locs.end()),
    TrackedRecords(records), IsAnonymous(Anonymous), ResolveFirst(false) {
    init();
  }
  explicit Record(const std::string &N, ArrayRef<SMLoc> locs,
                  RecordKeeper &records, bool Anonymous = false)
    : Record(StringInit::get(N), locs, records, Anonymous) {}


  // When copy-constructing a Record, we must still guarantee a globally unique
  // ID number.  Don't copy TheInit either since it's owned by the original
  // record. All other fields can be copied normally.
  Record(const Record &O) :
    ID(LastID++), Name(O.Name), Locs(O.Locs), TemplateArgs(O.TemplateArgs),
    Values(O.Values), SuperClasses(O.SuperClasses),
    SuperClassRanges(O.SuperClassRanges), TrackedRecords(O.TrackedRecords),
    IsAnonymous(O.IsAnonymous),
    ResolveFirst(O.ResolveFirst) { }

  static unsigned getNewUID() { return LastID++; }

  unsigned getID() const { return ID; }

  const std::string &getName() const;
  Init *getNameInit() const {
    return Name;
  }
  const std::string getNameInitAsString() const {
    return getNameInit()->getAsUnquotedString();
  }

  void setName(Init *Name);               // Also updates RecordKeeper.
  void setName(const std::string &Name);  // Also updates RecordKeeper.

  ArrayRef<SMLoc> getLoc() const { return Locs; }

  /// get the corresponding DefInit.
  DefInit *getDefInit();

  ArrayRef<Init *> getTemplateArgs() const {
    return TemplateArgs;
  }
  ArrayRef<RecordVal> getValues() const { return Values; }
  ArrayRef<Record *>  getSuperClasses() const { return SuperClasses; }
  ArrayRef<SMRange> getSuperClassRanges() const { return SuperClassRanges; }

  bool isTemplateArg(Init *Name) const {
    for (Init *TA : TemplateArgs)
      if (TA == Name) return true;
    return false;
  }
  bool isTemplateArg(StringRef Name) const {
    return isTemplateArg(StringInit::get(Name));
  }

  const RecordVal *getValue(const Init *Name) const {
    for (const RecordVal &Val : Values)
      if (Val.getNameInit() == Name) return &Val;
    return nullptr;
  }
  const RecordVal *getValue(StringRef Name) const {
    return getValue(StringInit::get(Name));
  }
  RecordVal *getValue(const Init *Name) {
    for (RecordVal &Val : Values)
      if (Val.getNameInit() == Name) return &Val;
    return nullptr;
  }
  RecordVal *getValue(StringRef Name) {
    return getValue(StringInit::get(Name));
  }

  void addTemplateArg(Init *Name) {
    assert(!isTemplateArg(Name) && "Template arg already defined!");
    TemplateArgs.push_back(Name);
  }
  void addTemplateArg(StringRef Name) {
    addTemplateArg(StringInit::get(Name));
  }

  void addValue(const RecordVal &RV) {
    assert(getValue(RV.getNameInit()) == nullptr && "Value already added!");
    Values.push_back(RV);
    if (Values.size() > 1)
      // Keep NAME at the end of the list.  It makes record dumps a
      // bit prettier and allows TableGen tests to be written more
      // naturally.  Tests can use CHECK-NEXT to look for Record
      // fields they expect to see after a def.  They can't do that if
      // NAME is the first Record field.
      std::swap(Values[Values.size() - 2], Values[Values.size() - 1]);
  }

  void removeValue(Init *Name) {
    for (unsigned i = 0, e = Values.size(); i != e; ++i)
      if (Values[i].getNameInit() == Name) {
        Values.erase(Values.begin()+i);
        return;
      }
    llvm_unreachable("Cannot remove an entry that does not exist!");
  }

  void removeValue(StringRef Name) {
    removeValue(StringInit::get(Name));
  }

  bool isSubClassOf(const Record *R) const {
    for (const Record *SC : SuperClasses)
      if (SC == R)
        return true;
    return false;
  }

  bool isSubClassOf(StringRef Name) const {
    for (const Record *SC : SuperClasses)
      if (SC->getNameInitAsString() == Name)
        return true;
    return false;
  }

  void addSuperClass(Record *R, SMRange Range) {
    assert(!isSubClassOf(R) && "Already subclassing record!");
    SuperClasses.push_back(R);
    SuperClassRanges.push_back(Range);
  }

  /// resolveReferences - If there are any field references that refer to fields
  /// that have been filled in, we can propagate the values now.
  ///
  void resolveReferences() { resolveReferencesTo(nullptr); }

  /// resolveReferencesTo - If anything in this record refers to RV, replace the
  /// reference to RV with the RHS of RV.  If RV is null, we resolve all
  /// possible references.
  void resolveReferencesTo(const RecordVal *RV);

  RecordKeeper &getRecords() const {
    return TrackedRecords;
  }

  bool isAnonymous() const {
    return IsAnonymous;
  }

  bool isResolveFirst() const {
    return ResolveFirst;
  }

  void setResolveFirst(bool b) {
    ResolveFirst = b;
  }

  void dump() const;

  //===--------------------------------------------------------------------===//
  // High-level methods useful to tablegen back-ends
  //

  /// getValueInit - Return the initializer for a value with the specified name,
  /// or throw an exception if the field does not exist.
  ///
  Init *getValueInit(StringRef FieldName) const;

  /// Return true if the named field is unset.
  bool isValueUnset(StringRef FieldName) const {
    return isa<UnsetInit>(getValueInit(FieldName));
  }

  /// getValueAsString - This method looks up the specified field and returns
  /// its value as a string, throwing an exception if the field does not exist
  /// or if the value is not a string.
  ///
  std::string getValueAsString(StringRef FieldName) const;

  /// getValueAsBitsInit - This method looks up the specified field and returns
  /// its value as a BitsInit, throwing an exception if the field does not exist
  /// or if the value is not the right type.
  ///
  BitsInit *getValueAsBitsInit(StringRef FieldName) const;

  /// getValueAsListInit - This method looks up the specified field and returns
  /// its value as a ListInit, throwing an exception if the field does not exist
  /// or if the value is not the right type.
  ///
  ListInit *getValueAsListInit(StringRef FieldName) const;

  /// getValueAsListOfDefs - This method looks up the specified field and
  /// returns its value as a vector of records, throwing an exception if the
  /// field does not exist or if the value is not the right type.
  ///
  std::vector<Record*> getValueAsListOfDefs(StringRef FieldName) const;

  /// getValueAsListOfInts - This method looks up the specified field and
  /// returns its value as a vector of integers, throwing an exception if the
  /// field does not exist or if the value is not the right type.
  ///
  std::vector<int64_t> getValueAsListOfInts(StringRef FieldName) const;

  /// getValueAsListOfStrings - This method looks up the specified field and
  /// returns its value as a vector of strings, throwing an exception if the
  /// field does not exist or if the value is not the right type.
  ///
  std::vector<std::string> getValueAsListOfStrings(StringRef FieldName) const;

  /// getValueAsDef - This method looks up the specified field and returns its
  /// value as a Record, throwing an exception if the field does not exist or if
  /// the value is not the right type.
  ///
  Record *getValueAsDef(StringRef FieldName) const;

  /// getValueAsBit - This method looks up the specified field and returns its
  /// value as a bit, throwing an exception if the field does not exist or if
  /// the value is not the right type.
  ///
  bool getValueAsBit(StringRef FieldName) const;

  /// getValueAsBitOrUnset - This method looks up the specified field and
  /// returns its value as a bit. If the field is unset, sets Unset to true and
  /// returns false.
  ///
  bool getValueAsBitOrUnset(StringRef FieldName, bool &Unset) const;

  /// getValueAsInt - This method looks up the specified field and returns its
  /// value as an int64_t, throwing an exception if the field does not exist or
  /// if the value is not the right type.
  ///
  int64_t getValueAsInt(StringRef FieldName) const;

  /// getValueAsDag - This method looks up the specified field and returns its
  /// value as an Dag, throwing an exception if the field does not exist or if
  /// the value is not the right type.
  ///
  DagInit *getValueAsDag(StringRef FieldName) const;
};

raw_ostream &operator<<(raw_ostream &OS, const Record &R);

struct MultiClass {
  Record Rec;  // Placeholder for template args and Name.
  typedef std::vector<std::unique_ptr<Record>> RecordVector;
  RecordVector DefPrototypes;

  void dump() const;

  MultiClass(const std::string &Name, SMLoc Loc, RecordKeeper &Records) :
    Rec(Name, Loc, Records) {}
};

class RecordKeeper {
  typedef std::map<std::string, std::unique_ptr<Record>> RecordMap;
  RecordMap Classes, Defs;

public:
  const RecordMap &getClasses() const { return Classes; }
  const RecordMap &getDefs() const { return Defs; }

  Record *getClass(const std::string &Name) const {
    auto I = Classes.find(Name);
    return I == Classes.end() ? nullptr : I->second.get();
  }
  Record *getDef(const std::string &Name) const {
    auto I = Defs.find(Name);
    return I == Defs.end() ? nullptr : I->second.get();
  }
  void addClass(std::unique_ptr<Record> R) {
    bool Ins = Classes.insert(std::make_pair(R->getName(),
                                             std::move(R))).second;
    (void)Ins;
    assert(Ins && "Class already exists");
  }
  void addDef(std::unique_ptr<Record> R) {
    bool Ins = Defs.insert(std::make_pair(R->getName(),
                                          std::move(R))).second;
    (void)Ins;
    assert(Ins && "Record already exists");
  }

  //===--------------------------------------------------------------------===//
  // High-level helper methods, useful for tablegen backends...

  /// getAllDerivedDefinitions - This method returns all concrete definitions
  /// that derive from the specified class name.  If a class with the specified
  /// name does not exist, an exception is thrown.
  std::vector<Record*>
  getAllDerivedDefinitions(const std::string &ClassName) const;

  void dump() const;
};

/// LessRecord - Sorting predicate to sort record pointers by name.
///
struct LessRecord {
  bool operator()(const Record *Rec1, const Record *Rec2) const {
    return StringRef(Rec1->getName()).compare_numeric(Rec2->getName()) < 0;
  }
};

/// LessRecordByID - Sorting predicate to sort record pointers by their
/// unique ID. If you just need a deterministic order, use this, since it
/// just compares two `unsigned`; the other sorting predicates require
/// string manipulation.
struct LessRecordByID {
  bool operator()(const Record *LHS, const Record *RHS) const {
    return LHS->getID() < RHS->getID();
  }
};

/// LessRecordFieldName - Sorting predicate to sort record pointers by their
/// name field.
///
struct LessRecordFieldName {
  bool operator()(const Record *Rec1, const Record *Rec2) const {
    return Rec1->getValueAsString("Name") < Rec2->getValueAsString("Name");
  }
};

struct LessRecordRegister {
  static bool ascii_isdigit(char x) { return x >= '0' && x <= '9'; }

  struct RecordParts {
    SmallVector<std::pair< bool, StringRef>, 4> Parts;

    RecordParts(StringRef Rec) {
      if (Rec.empty())
        return;

      size_t Len = 0;
      const char *Start = Rec.data();
      const char *Curr = Start;
      bool isDigitPart = ascii_isdigit(Curr[0]);
      for (size_t I = 0, E = Rec.size(); I != E; ++I, ++Len) {
        bool isDigit = ascii_isdigit(Curr[I]);
        if (isDigit != isDigitPart) {
          Parts.push_back(std::make_pair(isDigitPart, StringRef(Start, Len)));
          Len = 0;
          Start = &Curr[I];
          isDigitPart = ascii_isdigit(Curr[I]);
        }
      }
      // Push the last part.
      Parts.push_back(std::make_pair(isDigitPart, StringRef(Start, Len)));
    }

    size_t size() { return Parts.size(); }

    std::pair<bool, StringRef> getPart(size_t i) {
      assert (i < Parts.size() && "Invalid idx!");
      return Parts[i];
    }
  };

  bool operator()(const Record *Rec1, const Record *Rec2) const {
    RecordParts LHSParts(StringRef(Rec1->getName()));
    RecordParts RHSParts(StringRef(Rec2->getName()));

    size_t LHSNumParts = LHSParts.size();
    size_t RHSNumParts = RHSParts.size();
    assert (LHSNumParts && RHSNumParts && "Expected at least one part!");

    if (LHSNumParts != RHSNumParts)
      return LHSNumParts < RHSNumParts;

    // We expect the registers to be of the form [_a-zA-z]+([0-9]*[_a-zA-Z]*)*.
    for (size_t I = 0, E = LHSNumParts; I < E; I+=2) {
      std::pair<bool, StringRef> LHSPart = LHSParts.getPart(I);
      std::pair<bool, StringRef> RHSPart = RHSParts.getPart(I);
      // Expect even part to always be alpha.
      assert (LHSPart.first == false && RHSPart.first == false &&
              "Expected both parts to be alpha.");
      if (int Res = LHSPart.second.compare(RHSPart.second))
        return Res < 0;
    }
    for (size_t I = 1, E = LHSNumParts; I < E; I+=2) {
      std::pair<bool, StringRef> LHSPart = LHSParts.getPart(I);
      std::pair<bool, StringRef> RHSPart = RHSParts.getPart(I);
      // Expect odd part to always be numeric.
      assert (LHSPart.first == true && RHSPart.first == true &&
              "Expected both parts to be numeric.");
      if (LHSPart.second.size() != RHSPart.second.size())
        return LHSPart.second.size() < RHSPart.second.size();

      unsigned LHSVal, RHSVal;

      bool LHSFailed = LHSPart.second.getAsInteger(10, LHSVal); (void)LHSFailed;
      assert(!LHSFailed && "Unable to convert LHS to integer.");
      bool RHSFailed = RHSPart.second.getAsInteger(10, RHSVal); (void)RHSFailed;
      assert(!RHSFailed && "Unable to convert RHS to integer.");

      if (LHSVal != RHSVal)
        return LHSVal < RHSVal;
    }
    return LHSNumParts < RHSNumParts;
  }
};

raw_ostream &operator<<(raw_ostream &OS, const RecordKeeper &RK);

/// QualifyName - Return an Init with a qualifier prefix referring
/// to CurRec's name.
Init *QualifyName(Record &CurRec, MultiClass *CurMultiClass,
                  Init *Name, const std::string &Scoper);

/// QualifyName - Return an Init with a qualifier prefix referring
/// to CurRec's name.
Init *QualifyName(Record &CurRec, MultiClass *CurMultiClass,
                  const std::string &Name, const std::string &Scoper);

} // End llvm namespace

#endif
