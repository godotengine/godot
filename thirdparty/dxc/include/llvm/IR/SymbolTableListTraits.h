//===-- llvm/SymbolTableListTraits.h - Traits for iplist --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a generic class that is used to implement the automatic
// symbol table manipulation that occurs when you put (for example) a named
// instruction into a basic block.
//
// The way that this is implemented is by using a special traits class with the
// intrusive list that makes up the list of instructions in a basic block.  When
// a new element is added to the list of instructions, the traits class is
// notified, allowing the symbol table to be updated.
//
// This generic class implements the traits class.  It must be generic so that
// it can work for all uses it, which include lists of instructions, basic
// blocks, arguments, functions, global variables, etc...
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_SYMBOLTABLELISTTRAITS_H
#define LLVM_IR_SYMBOLTABLELISTTRAITS_H

#include "llvm/ADT/ilist.h"

namespace llvm {
class ValueSymbolTable;
  
template<typename NodeTy> class ilist_iterator;
template<typename NodeTy, typename Traits> class iplist;
template<typename Ty> struct ilist_traits;

// ValueSubClass   - The type of objects that I hold, e.g. Instruction.
// ItemParentClass - The type of object that owns the list, e.g. BasicBlock.
//
template<typename ValueSubClass, typename ItemParentClass>
class SymbolTableListTraits : public ilist_default_traits<ValueSubClass> {
  typedef ilist_traits<ValueSubClass> TraitsClass;
public:
  SymbolTableListTraits() {}

  /// getListOwner - Return the object that owns this list.  If this is a list
  /// of instructions, it returns the BasicBlock that owns them.
  ItemParentClass *getListOwner() {
    size_t Offset(size_t(&((ItemParentClass*)nullptr->*ItemParentClass::
                           getSublistAccess(static_cast<ValueSubClass*>(nullptr)))));
    iplist<ValueSubClass>* Anchor(static_cast<iplist<ValueSubClass>*>(this));
    return reinterpret_cast<ItemParentClass*>(reinterpret_cast<char*>(Anchor)-
                                              Offset);
  }

  static iplist<ValueSubClass> &getList(ItemParentClass *Par) {
    return Par->*(Par->getSublistAccess((ValueSubClass*)nullptr));
  }

  static ValueSymbolTable *getSymTab(ItemParentClass *Par) {
    return Par ? toPtr(Par->getValueSymbolTable()) : nullptr;
  }

  void addNodeToList(ValueSubClass *V);
  void removeNodeFromList(ValueSubClass *V);
  void transferNodesFromList(ilist_traits<ValueSubClass> &L2,
                             ilist_iterator<ValueSubClass> first,
                             ilist_iterator<ValueSubClass> last);
//private:
  template<typename TPtr>
  void setSymTabObject(TPtr *, TPtr);
  static ValueSymbolTable *toPtr(ValueSymbolTable *P) { return P; }
  static ValueSymbolTable *toPtr(ValueSymbolTable &R) { return &R; }
};

} // End llvm namespace

#endif
