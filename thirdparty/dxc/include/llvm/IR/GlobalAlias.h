//===-------- llvm/GlobalAlias.h - GlobalAlias class ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the GlobalAlias class, which
// represents a single function or variable alias in the IR.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_GLOBALALIAS_H
#define LLVM_IR_GLOBALALIAS_H

#include "llvm/ADT/Twine.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/OperandTraits.h"

namespace llvm {

class Module;
template<typename ValueSubClass, typename ItemParentClass>
  class SymbolTableListTraits;

class GlobalAlias : public GlobalValue, public ilist_node<GlobalAlias> {
  friend class SymbolTableListTraits<GlobalAlias, Module>;
  void operator=(const GlobalAlias &) = delete;
  GlobalAlias(const GlobalAlias &) = delete;

  void setParent(Module *parent);

  GlobalAlias(PointerType *Ty, LinkageTypes Linkage, const Twine &Name,
              Constant *Aliasee, Module *Parent);

public:
  // allocate space for exactly one operand
  void *operator new(size_t s) {
    return User::operator new(s, 1);
  }

  /// If a parent module is specified, the alias is automatically inserted into
  /// the end of the specified module's alias list.
  static GlobalAlias *create(PointerType *Ty, LinkageTypes Linkage,
                             const Twine &Name, Constant *Aliasee,
                             Module *Parent);

  // Without the Aliasee.
  static GlobalAlias *create(PointerType *Ty, LinkageTypes Linkage,
                             const Twine &Name, Module *Parent);

  // The module is taken from the Aliasee.
  static GlobalAlias *create(PointerType *Ty, LinkageTypes Linkage,
                             const Twine &Name, GlobalValue *Aliasee);

  // Type, Parent and AddressSpace taken from the Aliasee.
  static GlobalAlias *create(LinkageTypes Linkage, const Twine &Name,
                             GlobalValue *Aliasee);

  // Linkage, Type, Parent and AddressSpace taken from the Aliasee.
  static GlobalAlias *create(const Twine &Name, GlobalValue *Aliasee);

  /// Provide fast operand accessors
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Constant);

  /// removeFromParent - This method unlinks 'this' from the containing module,
  /// but does not delete it.
  ///
  void removeFromParent() override;

  /// eraseFromParent - This method unlinks 'this' from the containing module
  /// and deletes it.
  ///
  void eraseFromParent() override;

  /// These methods retrive and set alias target.
  void setAliasee(Constant *Aliasee);
  const Constant *getAliasee() const {
    return const_cast<GlobalAlias *>(this)->getAliasee();
  }
  Constant *getAliasee() {
    return getOperand(0);
  }

  const GlobalObject *getBaseObject() const {
    return const_cast<GlobalAlias *>(this)->getBaseObject();
  }
  GlobalObject *getBaseObject() {
    return dyn_cast<GlobalObject>(getAliasee()->stripInBoundsOffsets());
  }

  const GlobalObject *getBaseObject(const DataLayout &DL, APInt &Offset) const {
    return const_cast<GlobalAlias *>(this)->getBaseObject(DL, Offset);
  }
  GlobalObject *getBaseObject(const DataLayout &DL, APInt &Offset) {
    return dyn_cast<GlobalObject>(
        getAliasee()->stripAndAccumulateInBoundsConstantOffsets(DL, Offset));
  }

  static bool isValidLinkage(LinkageTypes L) {
    return isExternalLinkage(L) || isLocalLinkage(L) ||
      isWeakLinkage(L) || isLinkOnceLinkage(L);
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Value *V) {
    return V->getValueID() == Value::GlobalAliasVal;
  }
};

template <>
struct OperandTraits<GlobalAlias> :
  public FixedNumOperandTraits<GlobalAlias, 1> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(GlobalAlias, Constant)

} // End llvm namespace

#endif
