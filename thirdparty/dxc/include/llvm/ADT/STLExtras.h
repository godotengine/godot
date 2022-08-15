//===- llvm/ADT/STLExtras.h - Useful STL related functions ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains some templates that are useful if you are working with the
// STL at all.
//
// No library is required when using these functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_STLEXTRAS_H
#define LLVM_ADT_STLEXTRAS_H

#include "dxc/Support/WinAdapter.h" // HLSL Change
#include "llvm/Support/Compiler.h"
#include <algorithm> // for std::all_of
#include <cassert>
#include <cstddef> // for std::size_t
#include <cstdlib> // for qsort
#include <functional>
#include <iterator>
#include <memory>
#include <utility> // for std::pair

namespace llvm {

//===----------------------------------------------------------------------===//
//     Extra additions to <functional>
//===----------------------------------------------------------------------===//

template<class Ty>
struct identity {
  using argument_type = Ty;

  Ty &operator()(Ty &self) const {
    return self;
  }
  const Ty &operator()(const Ty &self) const {
    return self;
  }
};

template<class Ty>
struct less_ptr {
  bool operator()(const Ty* left, const Ty* right) const {
    return *left < *right;
  }
};

template<class Ty>
struct greater_ptr {
  bool operator()(const Ty* left, const Ty* right) const {
    return *right < *left;
  }
};

/// An efficient, type-erasing, non-owning reference to a callable. This is
/// intended for use as the type of a function parameter that is not used
/// after the function in question returns.
///
/// This class does not own the callable, so it is not in general safe to store
/// a function_ref.
template<typename Fn> class function_ref;

template<typename Ret, typename ...Params>
class function_ref<Ret(Params...)> {
  Ret (*callback)(intptr_t callable, Params ...params);
  intptr_t callable;

  template<typename Callable>
  static Ret callback_fn(intptr_t callable, Params ...params) {
    return (*reinterpret_cast<Callable*>(callable))(
        std::forward<Params>(params)...);
  }

public:
  template <typename Callable>
  function_ref(Callable &&callable,
               typename std::enable_if<
                   !std::is_same<typename std::remove_reference<Callable>::type,
                                 function_ref>::value>::type * = nullptr)
      : callback(callback_fn<typename std::remove_reference<Callable>::type>),
        callable(reinterpret_cast<intptr_t>(&callable)) {}
  Ret operator()(Params ...params) const {
    return callback(callable, std::forward<Params>(params)...);
  }
};

// deleter - Very very very simple method that is used to invoke operator
// delete on something.  It is used like this:
//
//   for_each(V.begin(), B.end(), deleter<Interval>);
//
template <class T>
inline void deleter(T *Ptr) {
  delete Ptr;
}



//===----------------------------------------------------------------------===//
//     Extra additions to <iterator>
//===----------------------------------------------------------------------===//

// mapped_iterator - This is a simple iterator adapter that causes a function to
// be dereferenced whenever operator* is invoked on the iterator.
//
template <class RootIt, class UnaryFunc>
class mapped_iterator {
  RootIt current;
  UnaryFunc Fn;
public:
  typedef typename std::iterator_traits<RootIt>::iterator_category
          iterator_category;
  typedef typename std::iterator_traits<RootIt>::difference_type
          difference_type;
  typedef decltype(std::declval<UnaryFunc>()(*std::declval<RootIt>()))
          value_type;

  typedef void pointer;
  //typedef typename UnaryFunc::result_type *pointer;
  typedef void reference;        // Can't modify value returned by fn

  typedef RootIt iterator_type;

  inline const RootIt &getCurrent() const { return current; }
  inline const UnaryFunc &getFunc() const { return Fn; }

  inline explicit mapped_iterator(const RootIt &I, UnaryFunc F)
    : current(I), Fn(F) {}

  inline value_type operator*() const {   // All this work to do this
    return Fn(*current);         // little change
  }

  mapped_iterator &operator++() {
    ++current;
    return *this;
  }
  mapped_iterator &operator--() {
    --current;
    return *this;
  }
  mapped_iterator operator++(int) {
    mapped_iterator __tmp = *this;
    ++current;
    return __tmp;
  }
  mapped_iterator operator--(int) {
    mapped_iterator __tmp = *this;
    --current;
    return __tmp;
  }
  mapped_iterator operator+(difference_type n) const {
    return mapped_iterator(current + n, Fn);
  }
  mapped_iterator &operator+=(difference_type n) {
    current += n;
    return *this;
  }
  mapped_iterator operator-(difference_type n) const {
    return mapped_iterator(current - n, Fn);
  }
  mapped_iterator &operator-=(difference_type n) {
    current -= n;
    return *this;
  }
  reference operator[](difference_type n) const { return *(*this + n); }

  bool operator!=(const mapped_iterator &X) const { return !operator==(X); }
  bool operator==(const mapped_iterator &X) const {
    return current == X.current;
  }
  bool operator<(const mapped_iterator &X) const { return current < X.current; }

  difference_type operator-(const mapped_iterator &X) const {
    return current - X.current;
  }
};

template <class Iterator, class Func>
inline mapped_iterator<Iterator, Func>
operator+(typename mapped_iterator<Iterator, Func>::difference_type N,
          const mapped_iterator<Iterator, Func> &X) {
  return mapped_iterator<Iterator, Func>(X.getCurrent() - N, X.getFunc());
}


// map_iterator - Provide a convenient way to create mapped_iterators, just like
// make_pair is useful for creating pairs...
//
template <class ItTy, class FuncTy>
inline mapped_iterator<ItTy, FuncTy> map_iterator(const ItTy &I, FuncTy F) {
  return mapped_iterator<ItTy, FuncTy>(I, F);
}

//===----------------------------------------------------------------------===//
//     Extra additions to <utility>
//===----------------------------------------------------------------------===//

/// \brief Function object to check whether the first component of a std::pair
/// compares less than the first component of another std::pair.
struct less_first {
  template <typename T> bool operator()(const T &lhs, const T &rhs) const {
    return lhs.first < rhs.first;
  }
};

/// \brief Function object to check whether the second component of a std::pair
/// compares less than the second component of another std::pair.
struct less_second {
  template <typename T> bool operator()(const T &lhs, const T &rhs) const {
    return lhs.second < rhs.second;
  }
};

// A subset of N3658. More stuff can be added as-needed.

/// \brief Represents a compile-time sequence of integers.
template <class T, T... I> struct integer_sequence {
  typedef T value_type;

  static LLVM_CONSTEXPR size_t size() { return sizeof...(I); }
};

/// \brief Alias for the common case of a sequence of size_ts.
template <size_t... I>
struct index_sequence : integer_sequence<std::size_t, I...> {};

template <std::size_t N, std::size_t... I>
struct build_index_impl : build_index_impl<N - 1, N - 1, I...> {};
template <std::size_t... I>
struct build_index_impl<0, I...> : index_sequence<I...> {};

/// \brief Creates a compile-time integer sequence for a parameter pack.
template <class... Ts>
struct index_sequence_for : build_index_impl<sizeof...(Ts)> {};

//===----------------------------------------------------------------------===//
//     Extra additions for arrays
//===----------------------------------------------------------------------===//

/// Find the length of an array.
template <class T, std::size_t N>
LLVM_CONSTEXPR inline size_t array_lengthof(T (&)[N]) {
  return N;
}

/// Adapt std::less<T> for array_pod_sort.
// HLSL Change: changed calling convention to __cdecl
template<typename T>
inline int __cdecl array_pod_sort_comparator(const void *P1, const void *P2) {
  if (std::less<T>()(*reinterpret_cast<const T*>(P1),
                     *reinterpret_cast<const T*>(P2)))
    return -1;
  if (std::less<T>()(*reinterpret_cast<const T*>(P2),
                     *reinterpret_cast<const T*>(P1)))
    return 1;
  return 0;
}

/// get_array_pod_sort_comparator - This is an internal helper function used to
/// get type deduction of T right.
//
// HLSL Change: changed calling convention to __cdecl
// HLSL Change: pulled this out into a typdef to make it easier to make change
typedef int(__cdecl *llvm_cmp_func)(const void *, const void *);

template<typename T>
inline llvm_cmp_func get_array_pod_sort_comparator(const T &) {
  return array_pod_sort_comparator<T>;
}


/// array_pod_sort - This sorts an array with the specified start and end
/// extent.  This is just like std::sort, except that it calls qsort instead of
/// using an inlined template.  qsort is slightly slower than std::sort, but
/// most sorts are not performance critical in LLVM and std::sort has to be
/// template instantiated for each type, leading to significant measured code
/// bloat.  This function should generally be used instead of std::sort where
/// possible.
///
/// This function assumes that you have simple POD-like types that can be
/// compared with std::less and can be moved with memcpy.  If this isn't true,
/// you should use std::sort.
///
/// NOTE: If qsort_r were portable, we could allow a custom comparator and
/// default to std::less.
template<class IteratorTy>
inline void array_pod_sort(IteratorTy Start, IteratorTy End) {
  // Don't inefficiently call qsort with one element or trigger undefined
  // behavior with an empty sequence.
  auto NElts = End - Start;
  if (NElts <= 1) return;
  qsort(&*Start, NElts, sizeof(*Start), get_array_pod_sort_comparator(*Start));
}

// HLSL Change: changed calling convention of Compare to __cdecl
template <class IteratorTy>
inline void array_pod_sort(
    IteratorTy Start, IteratorTy End,
    int (__cdecl *Compare)(
        const typename std::iterator_traits<IteratorTy>::value_type *,
        const typename std::iterator_traits<IteratorTy>::value_type *)) {
  // Don't inefficiently call qsort with one element or trigger undefined
  // behavior with an empty sequence.
  auto NElts = End - Start;
  if (NElts <= 1) return;
  qsort(&*Start, NElts, sizeof(*Start),
        reinterpret_cast<int (__cdecl *)(const void *, const void *)>(Compare));  // HLSL Change - __cdecl
}

//===----------------------------------------------------------------------===//
//     Extra additions to <algorithm>
//===----------------------------------------------------------------------===//

/// For a container of pointers, deletes the pointers and then clears the
/// container.
template<typename Container>
void DeleteContainerPointers(Container &C) {
  for (typename Container::iterator I = C.begin(), E = C.end(); I != E; ++I)
    delete *I;
  C.clear();
}

/// In a container of pairs (usually a map) whose second element is a pointer,
/// deletes the second elements and then clears the container.
template<typename Container>
void DeleteContainerSeconds(Container &C) {
  for (typename Container::iterator I = C.begin(), E = C.end(); I != E; ++I)
    delete I->second;
  C.clear();
}

/// Provide wrappers to std::all_of which take ranges instead of having to pass
/// being/end explicitly.
template<typename R, class UnaryPredicate>
bool all_of(R &&Range, UnaryPredicate &&P) {
  return std::all_of(Range.begin(), Range.end(),
                     std::forward<UnaryPredicate>(P));
}

//===----------------------------------------------------------------------===//
//     Extra additions to <memory>
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

// Implement make_unique according to N3656.

/// \brief Constructs a `new T()` with the given args and returns a
///        `unique_ptr<T>` which owns the object.
///
/// Example:
///
///     auto p = make_unique<int>();
///     auto p = make_unique<std::tuple<int, int>>(0, 1);
template <class T, class... Args>
typename std::enable_if<!std::is_array<T>::value, std::unique_ptr<T>>::type
make_unique(Args &&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

/// \brief Constructs a `new T[n]` with the given args and returns a
///        `unique_ptr<T[]>` which owns the object.
///
/// \param n size of the new array.
///
/// Example:
///
///     auto p = make_unique<int[]>(2); // value-initializes the array with 0's.
template <class T>
typename std::enable_if<std::is_array<T>::value && std::extent<T>::value == 0,
                        std::unique_ptr<T>>::type
make_unique(size_t n) {
  return std::unique_ptr<T>(new typename std::remove_extent<T>::type[n]());
}

/// This function isn't used and is only here to provide better compile errors.
template <class T, class... Args>
typename std::enable_if<std::extent<T>::value != 0>::type
make_unique(Args &&...) = delete;

struct FreeDeleter {
  void operator()(void* v) {
    ::free(v);
  }
};

template<typename First, typename Second>
struct pair_hash {
  size_t operator()(const std::pair<First, Second> &P) const {
    return std::hash<First>()(P.first) * 31 + std::hash<Second>()(P.second);
  }
};

/// A functor like C++14's std::less<void> in its absence.
struct less {
  template <typename A, typename B> bool operator()(A &&a, B &&b) const {
    return std::forward<A>(a) < std::forward<B>(b);
  }
};

/// A functor like C++14's std::equal<void> in its absence.
struct equal {
  template <typename A, typename B> bool operator()(A &&a, B &&b) const {
    return std::forward<A>(a) == std::forward<B>(b);
  }
};

/// Binary functor that adapts to any other binary functor after dereferencing
/// operands.
template <typename T> struct deref {
  T func;
  // Could be further improved to cope with non-derivable functors and
  // non-binary functors (should be a variadic template member function
  // operator()).
  template <typename A, typename B>
  auto operator()(A &lhs, B &rhs) const -> decltype(func(*lhs, *rhs)) {
    assert(lhs);
    assert(rhs);
    return func(*lhs, *rhs);
  }
};

} // End llvm namespace

#endif
