//===-- CostTable.h - Instruction Cost Table handling -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Cost tables and simple lookup functions
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_COSTTABLE_H_
#define LLVM_TARGET_COSTTABLE_H_

namespace llvm {

/// Cost Table Entry
template <class TypeTy>
struct CostTblEntry {
  int ISD;
  TypeTy Type;
  unsigned Cost;
};

/// Find in cost table, TypeTy must be comparable to CompareTy by ==
template <class TypeTy, class CompareTy>
int CostTableLookup(const CostTblEntry<TypeTy> *Tbl, unsigned len, int ISD,
                    CompareTy Ty) {
  for (unsigned int i = 0; i < len; ++i)
    if (ISD == Tbl[i].ISD && Ty == Tbl[i].Type)
      return i;

  // Could not find an entry.
  return -1;
}

/// Find in cost table, TypeTy must be comparable to CompareTy by ==
template <class TypeTy, class CompareTy, unsigned N>
int CostTableLookup(const CostTblEntry<TypeTy>(&Tbl)[N], int ISD,
                    CompareTy Ty) {
  return CostTableLookup(Tbl, N, ISD, Ty);
}

/// Type Conversion Cost Table
template <class TypeTy>
struct TypeConversionCostTblEntry {
  int ISD;
  TypeTy Dst;
  TypeTy Src;
  unsigned Cost;
};

/// Find in type conversion cost table, TypeTy must be comparable to CompareTy
/// by ==
template <class TypeTy, class CompareTy>
int ConvertCostTableLookup(const TypeConversionCostTblEntry<TypeTy> *Tbl,
                           unsigned len, int ISD, CompareTy Dst,
                           CompareTy Src) {
  for (unsigned int i = 0; i < len; ++i)
    if (ISD == Tbl[i].ISD && Src == Tbl[i].Src && Dst == Tbl[i].Dst)
      return i;

  // Could not find an entry.
  return -1;
}

/// Find in type conversion cost table, TypeTy must be comparable to CompareTy
/// by ==
template <class TypeTy, class CompareTy, unsigned N>
int ConvertCostTableLookup(const TypeConversionCostTblEntry<TypeTy>(&Tbl)[N],
                           int ISD, CompareTy Dst, CompareTy Src) {
  return ConvertCostTableLookup(Tbl, N, ISD, Dst, Src);
}

} // namespace llvm


#endif /* LLVM_TARGET_COSTTABLE_H_ */
