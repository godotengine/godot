//===- GetElementPtrTypeIterator.h ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements an iterator for walking through the types indexed by
// getelementptr instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_GETELEMENTPTRTYPEITERATOR_H
#define LLVM_IR_GETELEMENTPTRTYPEITERATOR_H

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/User.h"
#include "llvm/ADT/PointerIntPair.h"

namespace llvm {
  template<typename ItTy = User::const_op_iterator>
  class generic_gep_type_iterator
    : public std::iterator<std::forward_iterator_tag, Type *, ptrdiff_t> {
    typedef std::iterator<std::forward_iterator_tag,
                          Type *, ptrdiff_t> super;

    ItTy OpIt;
    PointerIntPair<Type *, 1> CurTy;
    unsigned AddrSpace;
    generic_gep_type_iterator() {}
  public:

    static generic_gep_type_iterator begin(Type *Ty, ItTy It) {
      generic_gep_type_iterator I;
      I.CurTy.setPointer(Ty);
      I.OpIt = It;
      return I;
    }
    static generic_gep_type_iterator begin(Type *Ty, unsigned AddrSpace,
                                           ItTy It) {
      generic_gep_type_iterator I;
      I.CurTy.setPointer(Ty);
      I.CurTy.setInt(true);
      I.AddrSpace = AddrSpace;
      I.OpIt = It;
      return I;
    }
    static generic_gep_type_iterator end(ItTy It) {
      generic_gep_type_iterator I;
      I.OpIt = It;
      return I;
    }

    bool operator==(const generic_gep_type_iterator& x) const {
      return OpIt == x.OpIt;
    }
    bool operator!=(const generic_gep_type_iterator& x) const {
      return !operator==(x);
    }

    Type *operator*() const {
      if (CurTy.getInt())
        return CurTy.getPointer()->getPointerTo(AddrSpace);
      return CurTy.getPointer();
    }

    Type *getIndexedType() const {
      if (CurTy.getInt())
        return CurTy.getPointer();
      CompositeType *CT = cast<CompositeType>(CurTy.getPointer());
      return CT->getTypeAtIndex(getOperand());
    }

    // This is a non-standard operator->.  It allows you to call methods on the
    // current type directly.
    Type *operator->() const { return operator*(); }

    Value *getOperand() const { return *OpIt; }

    generic_gep_type_iterator& operator++() {   // Preincrement
      if (CurTy.getInt()) {
        CurTy.setInt(false);
      } else if (CompositeType *CT =
                     dyn_cast<CompositeType>(CurTy.getPointer())) {
        CurTy.setPointer(CT->getTypeAtIndex(getOperand()));
      } else {
        CurTy.setPointer(nullptr);
      }
      ++OpIt;
      return *this;
    }

    generic_gep_type_iterator operator++(int) { // Postincrement
      generic_gep_type_iterator tmp = *this; ++*this; return tmp;
    }
  };

  typedef generic_gep_type_iterator<> gep_type_iterator;

  inline gep_type_iterator gep_type_begin(const User *GEP) {
    auto *GEPOp = cast<GEPOperator>(GEP);
    return gep_type_iterator::begin(
        GEPOp->getSourceElementType(),
        cast<PointerType>(GEPOp->getPointerOperandType()->getScalarType())
            ->getAddressSpace(),
        GEP->op_begin() + 1);
  }
  inline gep_type_iterator gep_type_end(const User *GEP) {
    return gep_type_iterator::end(GEP->op_end());
  }
  inline gep_type_iterator gep_type_begin(const User &GEP) {
    auto &GEPOp = cast<GEPOperator>(GEP);
    return gep_type_iterator::begin(
        GEPOp.getSourceElementType(),
        cast<PointerType>(GEPOp.getPointerOperandType()->getScalarType())
            ->getAddressSpace(),
        GEP.op_begin() + 1);
  }
  inline gep_type_iterator gep_type_end(const User &GEP) {
    return gep_type_iterator::end(GEP.op_end());
  }

  template<typename T>
  inline generic_gep_type_iterator<const T *>
  gep_type_begin(Type *Op0, ArrayRef<T> A) {
    return generic_gep_type_iterator<const T *>::begin(Op0, A.begin());
  }

  template<typename T>
  inline generic_gep_type_iterator<const T *>
  gep_type_end(Type * /*Op0*/, ArrayRef<T> A) {
    return generic_gep_type_iterator<const T *>::end(A.end());
  }
} // end namespace llvm

#endif
