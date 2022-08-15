//===- iterator_range.h - A range adaptor for iterators ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This provides a very simple, boring adaptor for a begin and end iterator
/// into a range type. This should be used to build range views that work well
/// with range based for loops and range based constructors.
///
/// Note that code here follows more standards-based coding conventions as it
/// is mirroring proposed interfaces for standardization.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_ITERATOR_RANGE_H
#define LLVM_ADT_ITERATOR_RANGE_H

#include <utility>

namespace llvm {

/// \brief A range adaptor for a pair of iterators.
///
/// This just wraps two iterators into a range-compatible interface. Nothing
/// fancy at all.
template <typename IteratorT>
class iterator_range {
  IteratorT begin_iterator, end_iterator;

public:
  iterator_range(IteratorT begin_iterator, IteratorT end_iterator)
      : begin_iterator(std::move(begin_iterator)),
        end_iterator(std::move(end_iterator)) {}

  IteratorT begin() const { return begin_iterator; }
  IteratorT end() const { return end_iterator; }
};

/// \brief Convenience function for iterating over sub-ranges.
///
/// This provides a bit of syntactic sugar to make using sub-ranges
/// in for loops a bit easier. Analogous to std::make_pair().
template <class T> iterator_range<T> make_range(T x, T y) {
  return iterator_range<T>(std::move(x), std::move(y));
}

template <typename T> iterator_range<T> make_range(std::pair<T, T> p) {
  return iterator_range<T>(std::move(p.first), std::move(p.second));
}
}

#endif
