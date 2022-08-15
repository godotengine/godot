//===- llvm/ADT/SmallVector.h - 'Normally small' vectors --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the SmallVector class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_SMALLVECTOR_H
#define LLVM_ADT_SMALLVECTOR_H

#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/AlignOf.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/type_traits.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <memory>

namespace llvm {

/// This is all the non-templated stuff common to all SmallVectors.
class SmallVectorBase {
protected:
  void *BeginX, *EndX, *CapacityX;

protected:
  SmallVectorBase(void *FirstEl, size_t Size)
    : BeginX(FirstEl), EndX(FirstEl), CapacityX((char*)FirstEl+Size) {}

  /// This is an implementation of the grow() method which only works
  /// on POD-like data types and is out of line to reduce code duplication.
  void grow_pod(void *FirstEl, size_t MinSizeInBytes, size_t TSize);

public:
  /// This returns size()*sizeof(T).
  size_t size_in_bytes() const {
    return size_t((char*)EndX - (char*)BeginX);
  }

  /// capacity_in_bytes - This returns capacity()*sizeof(T).
  size_t capacity_in_bytes() const {
    return size_t((char*)CapacityX - (char*)BeginX);
  }

  bool LLVM_ATTRIBUTE_UNUSED_RESULT empty() const { return BeginX == EndX; }
};

template <typename T, unsigned N> struct SmallVectorStorage;

/// This is the part of SmallVectorTemplateBase which does not depend on whether
/// the type T is a POD. The extra dummy template argument is used by ArrayRef
/// to avoid unnecessarily requiring T to be complete.
template <typename T, typename = void>
class SmallVectorTemplateCommon : public SmallVectorBase {
private:
  template <typename, unsigned> friend struct SmallVectorStorage;

  // Allocate raw space for N elements of type T.  If T has a ctor or dtor, we
  // don't want it to be automatically run, so we need to represent the space as
  // something else.  Use an array of char of sufficient alignment.
  typedef llvm::AlignedCharArrayUnion<T> U;
  U FirstEl;
  // Space after 'FirstEl' is clobbered, do not add any instance vars after it.

protected:
  SmallVectorTemplateCommon(size_t Size) : SmallVectorBase(&FirstEl, Size) {}

  void grow_pod(size_t MinSizeInBytes, size_t TSize) {
    SmallVectorBase::grow_pod(&FirstEl, MinSizeInBytes, TSize);
  }

  /// Return true if this is a smallvector which has not had dynamic
  /// memory allocated for it.
  bool isSmall() const {
    return BeginX == static_cast<const void*>(&FirstEl);
  }

  /// Put this vector in a state of being small.
  void resetToSmall() {
    BeginX = EndX = CapacityX = &FirstEl;
  }

  void setEnd(T *P) { this->EndX = P; }
public:
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef T value_type;
  typedef T *iterator;
  typedef const T *const_iterator;

  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
  typedef std::reverse_iterator<iterator> reverse_iterator;

  typedef T &reference;
  typedef const T &const_reference;
  typedef T *pointer;
  typedef const T *const_pointer;

  // forward iterator creation methods.
  iterator begin() { return (iterator)this->BeginX; }
  const_iterator begin() const { return (const_iterator)this->BeginX; }
  iterator end() { return (iterator)this->EndX; }
  const_iterator end() const { return (const_iterator)this->EndX; }
protected:
  iterator capacity_ptr() { return (iterator)this->CapacityX; }
  const_iterator capacity_ptr() const { return (const_iterator)this->CapacityX;}
public:

  // reverse iterator creation methods.
  reverse_iterator rbegin()            { return reverse_iterator(end()); }
  const_reverse_iterator rbegin() const{ return const_reverse_iterator(end()); }
  reverse_iterator rend()              { return reverse_iterator(begin()); }
  const_reverse_iterator rend() const { return const_reverse_iterator(begin());}

  size_type size() const { return end()-begin(); }
  size_type max_size() const { return size_type(-1) / sizeof(T); }

  /// Return the total number of elements in the currently allocated buffer.
  size_t capacity() const { return capacity_ptr() - begin(); }

  /// Return a pointer to the vector's buffer, even if empty().
  pointer data() { return pointer(begin()); }
  /// Return a pointer to the vector's buffer, even if empty().
  const_pointer data() const { return const_pointer(begin()); }

  reference operator[](size_type idx) {
    assert(idx < size());
    return begin()[idx];
  }
  const_reference operator[](size_type idx) const {
    assert(idx < size());
    return begin()[idx];
  }

  reference front() {
    assert(!empty());
    return begin()[0];
  }
  const_reference front() const {
    assert(!empty());
    return begin()[0];
  }

  reference back() {
    assert(!empty());
    return end()[-1];
  }
  const_reference back() const {
    assert(!empty());
    return end()[-1];
  }
};

/// SmallVectorTemplateBase<isPodLike = false> - This is where we put method
/// implementations that are designed to work with non-POD-like T's.
template <typename T, bool isPodLike>
class SmallVectorTemplateBase : public SmallVectorTemplateCommon<T> {
protected:
  SmallVectorTemplateBase(size_t Size) : SmallVectorTemplateCommon<T>(Size) {}

  static void destroy_range(T *S, T *E) {
    while (S != E) {
      --E;
      E->~T();
    }
  }

  /// Use move-assignment to move the range [I, E) onto the
  /// objects starting with "Dest".  This is just <memory>'s
  /// std::move, but not all stdlibs actually provide that.
  template<typename It1, typename It2>
  static It2 move(It1 I, It1 E, It2 Dest) {
    for (; I != E; ++I, ++Dest)
      *Dest = ::std::move(*I);
    return Dest;
  }

  /// Use move-assignment to move the range
  /// [I, E) onto the objects ending at "Dest", moving objects
  /// in reverse order.  This is just <algorithm>'s
  /// std::move_backward, but not all stdlibs actually provide that.
  template<typename It1, typename It2>
  static It2 move_backward(It1 I, It1 E, It2 Dest) {
    while (I != E)
      *--Dest = ::std::move(*--E);
    return Dest;
  }

  /// Move the range [I, E) into the uninitialized memory starting with "Dest",
  /// constructing elements as needed.
  template<typename It1, typename It2>
  static void uninitialized_move(It1 I, It1 E, It2 Dest) {
    for (; I != E; ++I, ++Dest)
      ::new ((void*) &*Dest) T(::std::move(*I));
  }

  /// Copy the range [I, E) onto the uninitialized memory starting with "Dest",
  /// constructing elements as needed.
  template<typename It1, typename It2>
  static void uninitialized_copy(It1 I, It1 E, It2 Dest) {
    std::uninitialized_copy(I, E, Dest);
  }

  /// Grow the allocated memory (without initializing new elements), doubling
  /// the size of the allocated memory. Guarantees space for at least one more
  /// element, or MinSize more elements if specified.
  void grow(size_t MinSize = 0);

public:
  void push_back(const T &Elt) {
    if (LLVM_UNLIKELY(this->EndX >= this->CapacityX))
      this->grow();
    ::new ((void*) this->end()) T(Elt);
    this->setEnd(this->end()+1);
  }

  void push_back(T &&Elt) {
    if (LLVM_UNLIKELY(this->EndX >= this->CapacityX))
      this->grow();
    ::new ((void*) this->end()) T(::std::move(Elt));
    this->setEnd(this->end()+1);
  }

  void pop_back() {
    this->setEnd(this->end()-1);
    this->end()->~T();
  }
};

// Define this out-of-line to dissuade the C++ compiler from inlining it.
template <typename T, bool isPodLike>
void SmallVectorTemplateBase<T, isPodLike>::grow(size_t MinSize) {
  size_t CurCapacity = this->capacity();
  size_t CurSize = this->size();
  // Always grow, even from zero.
  size_t NewCapacity = size_t(NextPowerOf2(CurCapacity+2));
  if (NewCapacity < MinSize)
    NewCapacity = MinSize;
  T *NewElts = (T*)new char[NewCapacity*sizeof(T)]; // HLSL Change: Use overridable operator new

  // Move the elements over.
  this->uninitialized_move(this->begin(), this->end(), NewElts);

  // Destroy the original elements.
  destroy_range(this->begin(), this->end());

  // If this wasn't grown from the inline copy, deallocate the old space.
  if (!this->isSmall())
    delete[] (char*)this->begin(); // HLSL Change: Use overridable operator delete

  this->setEnd(NewElts+CurSize);
  this->BeginX = NewElts;
  this->CapacityX = this->begin()+NewCapacity;
}


/// SmallVectorTemplateBase<isPodLike = true> - This is where we put method
/// implementations that are designed to work with POD-like T's.
template <typename T>
class SmallVectorTemplateBase<T, true> : public SmallVectorTemplateCommon<T> {
protected:
  SmallVectorTemplateBase(size_t Size) : SmallVectorTemplateCommon<T>(Size) {}

  // No need to do a destroy loop for POD's.
  static void destroy_range(T *, T *) {}

  /// Use move-assignment to move the range [I, E) onto the
  /// objects starting with "Dest".  For PODs, this is just memcpy.
  template<typename It1, typename It2>
  static It2 move(It1 I, It1 E, It2 Dest) {
    return ::std::copy(I, E, Dest);
  }

  /// Use move-assignment to move the range [I, E) onto the objects ending at
  /// "Dest", moving objects in reverse order.
  template<typename It1, typename It2>
  static It2 move_backward(It1 I, It1 E, It2 Dest) {
    return ::std::copy_backward(I, E, Dest);
  }

  /// Move the range [I, E) onto the uninitialized memory
  /// starting with "Dest", constructing elements into it as needed.
  template<typename It1, typename It2>
  static void uninitialized_move(It1 I, It1 E, It2 Dest) {
    // Just do a copy.
    uninitialized_copy(I, E, Dest);
  }

  /// Copy the range [I, E) onto the uninitialized memory
  /// starting with "Dest", constructing elements into it as needed.
  template<typename It1, typename It2>
  static void uninitialized_copy(It1 I, It1 E, It2 Dest) {
    // Arbitrary iterator types; just use the basic implementation.
    std::uninitialized_copy(I, E, Dest);
  }

  /// Copy the range [I, E) onto the uninitialized memory
  /// starting with "Dest", constructing elements into it as needed.
  template <typename T1, typename T2>
  static void uninitialized_copy(
      T1 *I, T1 *E, T2 *Dest,
      typename std::enable_if<std::is_same<typename std::remove_const<T1>::type,
                                           T2>::value>::type * = nullptr) {
    // Use memcpy for PODs iterated by pointers (which includes SmallVector
    // iterators): std::uninitialized_copy optimizes to memmove, but we can
    // use memcpy here. Note that I and E are iterators and thus might be
    // invalid for memcpy if they are equal.
    if (I != E)
      memcpy(reinterpret_cast<void *>(Dest), I, (E - I) * sizeof(T));
  }

  /// Double the size of the allocated memory, guaranteeing space for at
  /// least one more element or MinSize if specified.
  void grow(size_t MinSize = 0) {
    this->grow_pod(MinSize*sizeof(T), sizeof(T));
  }
public:
  void push_back(const T &Elt) {
    if (LLVM_UNLIKELY(this->EndX >= this->CapacityX))
      this->grow();
    memcpy(reinterpret_cast<void *>(this->end()), &Elt, sizeof(T));
    this->setEnd(this->end()+1);
  }

  void pop_back() {
    this->setEnd(this->end()-1);
  }
};


/// This class consists of common code factored out of the SmallVector class to
/// reduce code duplication based on the SmallVector 'N' template parameter.
template <typename T>
class SmallVectorImpl : public SmallVectorTemplateBase<T, isPodLike<T>::value> {
  typedef SmallVectorTemplateBase<T, isPodLike<T>::value > SuperClass;

  SmallVectorImpl(const SmallVectorImpl&) = delete;
public:
  typedef typename SuperClass::iterator iterator;
  typedef typename SuperClass::size_type size_type;

protected:
  // Default ctor - Initialize to empty.
  explicit SmallVectorImpl(unsigned N)
    : SmallVectorTemplateBase<T, isPodLike<T>::value>(N*sizeof(T)) {
  }

public:
  ~SmallVectorImpl() {
    // Destroy the constructed elements in the vector.
    this->destroy_range(this->begin(), this->end());

    // If this wasn't grown from the inline copy, deallocate the old space.
    if (!this->isSmall())
      delete[] (char*)this->begin(); // HLSL Change: Use overridable operator delete
  }


  void clear() {
    this->destroy_range(this->begin(), this->end());
    this->EndX = this->BeginX;
  }

  void resize(size_type N) {
    if (N < this->size()) {
      this->destroy_range(this->begin()+N, this->end());
      this->setEnd(this->begin()+N);
    } else if (N > this->size()) {
      if (this->capacity() < N)
        this->grow(N);
      for (auto I = this->end(), E = this->begin() + N; I != E; ++I)
        new (&*I) T();
      this->setEnd(this->begin()+N);
    }
  }

  void resize(size_type N, const T &NV) {
    if (N < this->size()) {
      this->destroy_range(this->begin()+N, this->end());
      this->setEnd(this->begin()+N);
    } else if (N > this->size()) {
      if (this->capacity() < N)
        this->grow(N);
      std::uninitialized_fill(this->end(), this->begin()+N, NV);
      this->setEnd(this->begin()+N);
    }
  }

  void reserve(size_type N) {
    if (this->capacity() < N)
      this->grow(N);
  }

  T LLVM_ATTRIBUTE_UNUSED_RESULT pop_back_val() {
    T Result = ::std::move(this->back());
    this->pop_back();
    return Result;
  }

  void swap(SmallVectorImpl &RHS);

  /// Add the specified range to the end of the SmallVector.
  template<typename in_iter>
  void append(in_iter in_start, in_iter in_end) {
    size_type NumInputs = std::distance(in_start, in_end);
    // Grow allocated space if needed.
    if (NumInputs > size_type(this->capacity_ptr()-this->end()))
      this->grow(this->size()+NumInputs);

    // Copy the new elements over.
    this->uninitialized_copy(in_start, in_end, this->end());
    this->setEnd(this->end() + NumInputs);
  }

  /// Add the specified range to the end of the SmallVector.
  void append(size_type NumInputs, const T &Elt) {
    // Grow allocated space if needed.
    if (NumInputs > size_type(this->capacity_ptr()-this->end()))
      this->grow(this->size()+NumInputs);

    // Copy the new elements over.
    std::uninitialized_fill_n(this->end(), NumInputs, Elt);
    this->setEnd(this->end() + NumInputs);
  }

  void append(std::initializer_list<T> IL) {
    append(IL.begin(), IL.end());
  }

  void assign(size_type NumElts, const T &Elt) {
    clear();
    if (this->capacity() < NumElts)
      this->grow(NumElts);
    this->setEnd(this->begin()+NumElts);
    std::uninitialized_fill(this->begin(), this->end(), Elt);
  }

  void assign(std::initializer_list<T> IL) {
    clear();
    append(IL);
  }

  iterator erase(iterator I) {
    assert(I >= this->begin() && "Iterator to erase is out of bounds.");
    assert(I < this->end() && "Erasing at past-the-end iterator.");

    iterator N = I;
    // Shift all elts down one.
    this->move(I+1, this->end(), I);
    // Drop the last elt.
    this->pop_back();
    return(N);
  }

  iterator erase(iterator S, iterator E) {
    assert(S >= this->begin() && "Range to erase is out of bounds.");
    assert(S <= E && "Trying to erase invalid range.");
    assert(E <= this->end() && "Trying to erase past the end.");

    iterator N = S;
    // Shift all elts down.
    iterator I = this->move(E, this->end(), S);
    // Drop the last elts.
    this->destroy_range(I, this->end());
    this->setEnd(I);
    return(N);
  }

  iterator insert(iterator I, T &&Elt) {
    if (I == this->end()) {  // Important special case for empty vector.
      this->push_back(::std::move(Elt));
      return this->end()-1;
    }

    assert(I >= this->begin() && "Insertion iterator is out of bounds.");
    assert(I <= this->end() && "Inserting past the end of the vector.");

    if (this->EndX >= this->CapacityX) {
      size_t EltNo = I-this->begin();
      this->grow();
      I = this->begin()+EltNo;
    }

    ::new ((void*) this->end()) T(::std::move(this->back()));
    // Push everything else over.
    this->move_backward(I, this->end()-1, this->end());
    this->setEnd(this->end()+1);

    // If we just moved the element we're inserting, be sure to update
    // the reference.
    T *EltPtr = &Elt;
    if (I <= EltPtr && EltPtr < this->EndX)
      ++EltPtr;

    *I = ::std::move(*EltPtr);
    return I;
  }

  iterator insert(iterator I, const T &Elt) {
    if (I == this->end()) {  // Important special case for empty vector.
      this->push_back(Elt);
      return this->end()-1;
    }

    assert(I >= this->begin() && "Insertion iterator is out of bounds.");
    assert(I <= this->end() && "Inserting past the end of the vector.");

    if (this->EndX >= this->CapacityX) {
      size_t EltNo = I-this->begin();
      this->grow();
      I = this->begin()+EltNo;
    }
    ::new ((void*) this->end()) T(std::move(this->back()));
    // Push everything else over.
    this->move_backward(I, this->end()-1, this->end());
    this->setEnd(this->end()+1);

    // If we just moved the element we're inserting, be sure to update
    // the reference.
    const T *EltPtr = &Elt;
    if (I <= EltPtr && EltPtr < this->EndX)
      ++EltPtr;

    *I = *EltPtr;
    return I;
  }

  iterator insert(iterator I, size_type NumToInsert, const T &Elt) {
    // Convert iterator to elt# to avoid invalidating iterator when we reserve()
    size_t InsertElt = I - this->begin();

    if (I == this->end()) {  // Important special case for empty vector.
      append(NumToInsert, Elt);
      return this->begin()+InsertElt;
    }

    assert(I >= this->begin() && "Insertion iterator is out of bounds.");
    assert(I <= this->end() && "Inserting past the end of the vector.");

    // Ensure there is enough space.
    reserve(this->size() + NumToInsert);

    // Uninvalidate the iterator.
    I = this->begin()+InsertElt;

    // If there are more elements between the insertion point and the end of the
    // range than there are being inserted, we can use a simple approach to
    // insertion.  Since we already reserved space, we know that this won't
    // reallocate the vector.
    if (size_t(this->end()-I) >= NumToInsert) {
      T *OldEnd = this->end();
      append(std::move_iterator<iterator>(this->end() - NumToInsert),
             std::move_iterator<iterator>(this->end()));

      // Copy the existing elements that get replaced.
      this->move_backward(I, OldEnd-NumToInsert, OldEnd);

      std::fill_n(I, NumToInsert, Elt);
      return I;
    }

    // Otherwise, we're inserting more elements than exist already, and we're
    // not inserting at the end.

    // Move over the elements that we're about to overwrite.
    T *OldEnd = this->end();
    this->setEnd(this->end() + NumToInsert);
    size_t NumOverwritten = OldEnd-I;
    this->uninitialized_move(I, OldEnd, this->end()-NumOverwritten);

    // Replace the overwritten part.
    std::fill_n(I, NumOverwritten, Elt);

    // Insert the non-overwritten middle part.
    std::uninitialized_fill_n(OldEnd, NumToInsert-NumOverwritten, Elt);
    return I;
  }

  template<typename ItTy>
  iterator insert(iterator I, ItTy From, ItTy To) {
    // Convert iterator to elt# to avoid invalidating iterator when we reserve()
    size_t InsertElt = I - this->begin();

    if (I == this->end()) {  // Important special case for empty vector.
      append(From, To);
      return this->begin()+InsertElt;
    }

    assert(I >= this->begin() && "Insertion iterator is out of bounds.");
    assert(I <= this->end() && "Inserting past the end of the vector.");

    size_t NumToInsert = std::distance(From, To);

    // Ensure there is enough space.
    reserve(this->size() + NumToInsert);

    // Uninvalidate the iterator.
    I = this->begin()+InsertElt;

    // If there are more elements between the insertion point and the end of the
    // range than there are being inserted, we can use a simple approach to
    // insertion.  Since we already reserved space, we know that this won't
    // reallocate the vector.
    if (size_t(this->end()-I) >= NumToInsert) {
      T *OldEnd = this->end();
      append(std::move_iterator<iterator>(this->end() - NumToInsert),
             std::move_iterator<iterator>(this->end()));

      // Copy the existing elements that get replaced.
      this->move_backward(I, OldEnd-NumToInsert, OldEnd);

      std::copy(From, To, I);
      return I;
    }

    // Otherwise, we're inserting more elements than exist already, and we're
    // not inserting at the end.

    // Move over the elements that we're about to overwrite.
    T *OldEnd = this->end();
    this->setEnd(this->end() + NumToInsert);
    size_t NumOverwritten = OldEnd-I;
    this->uninitialized_move(I, OldEnd, this->end()-NumOverwritten);

    // Replace the overwritten part.
    for (T *J = I; NumOverwritten > 0; --NumOverwritten) {
      *J = *From;
      ++J; ++From;
    }

    // Insert the non-overwritten middle part.
    this->uninitialized_copy(From, To, OldEnd);
    return I;
  }

  void insert(iterator I, std::initializer_list<T> IL) {
    insert(I, IL.begin(), IL.end());
  }

  template <typename... ArgTypes> void emplace_back(ArgTypes &&... Args) {
    if (LLVM_UNLIKELY(this->EndX >= this->CapacityX))
      this->grow();
    ::new ((void *)this->end()) T(std::forward<ArgTypes>(Args)...);
    this->setEnd(this->end() + 1);
  }

  SmallVectorImpl &operator=(const SmallVectorImpl &RHS);

  SmallVectorImpl &operator=(SmallVectorImpl &&RHS);

  bool operator==(const SmallVectorImpl &RHS) const {
    if (this->size() != RHS.size()) return false;
    return std::equal(this->begin(), this->end(), RHS.begin());
  }
  bool operator!=(const SmallVectorImpl &RHS) const {
    return !(*this == RHS);
  }

  bool operator<(const SmallVectorImpl &RHS) const {
    return std::lexicographical_compare(this->begin(), this->end(),
                                        RHS.begin(), RHS.end());
  }

  /// Set the array size to \p N, which the current array must have enough
  /// capacity for.
  ///
  /// This does not construct or destroy any elements in the vector.
  ///
  /// Clients can use this in conjunction with capacity() to write past the end
  /// of the buffer when they know that more elements are available, and only
  /// update the size later. This avoids the cost of value initializing elements
  /// which will only be overwritten.
  void set_size(size_type N) {
    assert(N <= this->capacity());
    this->setEnd(this->begin() + N);
  }
};


template <typename T>
void SmallVectorImpl<T>::swap(SmallVectorImpl<T> &RHS) {
  if (this == &RHS) return;

  // We can only avoid copying elements if neither vector is small.
  if (!this->isSmall() && !RHS.isSmall()) {
    std::swap(this->BeginX, RHS.BeginX);
    std::swap(this->EndX, RHS.EndX);
    std::swap(this->CapacityX, RHS.CapacityX);
    return;
  }
  if (RHS.size() > this->capacity())
    this->grow(RHS.size());
  if (this->size() > RHS.capacity())
    RHS.grow(this->size());

  // Swap the shared elements.
  size_t NumShared = this->size();
  if (NumShared > RHS.size()) NumShared = RHS.size();
  for (size_type i = 0; i != NumShared; ++i)
    std::swap((*this)[i], RHS[i]);

  // Copy over the extra elts.
  if (this->size() > RHS.size()) {
    size_t EltDiff = this->size() - RHS.size();
    this->uninitialized_copy(this->begin()+NumShared, this->end(), RHS.end());
    RHS.setEnd(RHS.end()+EltDiff);
    this->destroy_range(this->begin()+NumShared, this->end());
    this->setEnd(this->begin()+NumShared);
  } else if (RHS.size() > this->size()) {
    size_t EltDiff = RHS.size() - this->size();
    this->uninitialized_copy(RHS.begin()+NumShared, RHS.end(), this->end());
    this->setEnd(this->end() + EltDiff);
    this->destroy_range(RHS.begin()+NumShared, RHS.end());
    RHS.setEnd(RHS.begin()+NumShared);
  }
}

template <typename T>
SmallVectorImpl<T> &SmallVectorImpl<T>::
  operator=(const SmallVectorImpl<T> &RHS) {
  // Avoid self-assignment.
  if (this == &RHS) return *this;

  // If we already have sufficient space, assign the common elements, then
  // destroy any excess.
  size_t RHSSize = RHS.size();
  size_t CurSize = this->size();
  if (CurSize >= RHSSize) {
    // Assign common elements.
    iterator NewEnd;
    if (RHSSize)
      NewEnd = std::copy(RHS.begin(), RHS.begin()+RHSSize, this->begin());
    else
      NewEnd = this->begin();

    // Destroy excess elements.
    this->destroy_range(NewEnd, this->end());

    // Trim.
    this->setEnd(NewEnd);
    return *this;
  }

  // If we have to grow to have enough elements, destroy the current elements.
  // This allows us to avoid copying them during the grow.
  // FIXME: don't do this if they're efficiently moveable.
  if (this->capacity() < RHSSize) {
    // Destroy current elements.
    this->destroy_range(this->begin(), this->end());
    this->setEnd(this->begin());
    CurSize = 0;
    this->grow(RHSSize);
  } else if (CurSize) {
    // Otherwise, use assignment for the already-constructed elements.
    std::copy(RHS.begin(), RHS.begin()+CurSize, this->begin());
  }

  // Copy construct the new elements in place.
  this->uninitialized_copy(RHS.begin()+CurSize, RHS.end(),
                           this->begin()+CurSize);

  // Set end.
  this->setEnd(this->begin()+RHSSize);
  return *this;
}

template <typename T>
SmallVectorImpl<T> &SmallVectorImpl<T>::operator=(SmallVectorImpl<T> &&RHS) {
  // Avoid self-assignment.
  if (this == &RHS) return *this;

  // If the RHS isn't small, clear this vector and then steal its buffer.
  if (!RHS.isSmall()) {
    this->destroy_range(this->begin(), this->end());
    if (!this->isSmall()) delete[] (char*)this->begin(); // HLSL Change: Use overridable operator delete
    this->BeginX = RHS.BeginX;
    this->EndX = RHS.EndX;
    this->CapacityX = RHS.CapacityX;
    RHS.resetToSmall();
    return *this;
  }

  // If we already have sufficient space, assign the common elements, then
  // destroy any excess.
  size_t RHSSize = RHS.size();
  size_t CurSize = this->size();
  if (CurSize >= RHSSize) {
    // Assign common elements.
    iterator NewEnd = this->begin();
    if (RHSSize)
      NewEnd = this->move(RHS.begin(), RHS.end(), NewEnd);

    // Destroy excess elements and trim the bounds.
    this->destroy_range(NewEnd, this->end());
    this->setEnd(NewEnd);

    // Clear the RHS.
    RHS.clear();

    return *this;
  }

  // If we have to grow to have enough elements, destroy the current elements.
  // This allows us to avoid copying them during the grow.
  // FIXME: this may not actually make any sense if we can efficiently move
  // elements.
  if (this->capacity() < RHSSize) {
    // Destroy current elements.
    this->destroy_range(this->begin(), this->end());
    this->setEnd(this->begin());
    CurSize = 0;
    this->grow(RHSSize);
  } else if (CurSize) {
    // Otherwise, use assignment for the already-constructed elements.
    this->move(RHS.begin(), RHS.begin()+CurSize, this->begin());
  }

  // Move-construct the new elements in place.
  this->uninitialized_move(RHS.begin()+CurSize, RHS.end(),
                           this->begin()+CurSize);

  // Set end.
  this->setEnd(this->begin()+RHSSize);

  RHS.clear();
  return *this;
}

/// Storage for the SmallVector elements which aren't contained in
/// SmallVectorTemplateCommon. There are 'N-1' elements here. The remaining '1'
/// element is in the base class. This is specialized for the N=1 and N=0 cases
/// to avoid allocating unnecessary storage.
template <typename T, unsigned N>
struct SmallVectorStorage {
  typename SmallVectorTemplateCommon<T>::U InlineElts[N - 1];
};
template <typename T> struct SmallVectorStorage<T, 1> {};
template <typename T> struct SmallVectorStorage<T, 0> {};

/// This is a 'vector' (really, a variable-sized array), optimized
/// for the case when the array is small.  It contains some number of elements
/// in-place, which allows it to avoid heap allocation when the actual number of
/// elements is below that threshold.  This allows normal "small" cases to be
/// fast without losing generality for large inputs.
///
/// Note that this does not attempt to be exception safe.
///
template <typename T, unsigned N>
class SmallVector : public SmallVectorImpl<T> {
  /// Inline space for elements which aren't stored in the base class.
  SmallVectorStorage<T, N> Storage;
public:
  SmallVector() : SmallVectorImpl<T>(N) {
  }

  explicit SmallVector(size_t Size, const T &Value = T())
    : SmallVectorImpl<T>(N) {
    this->assign(Size, Value);
  }

  template<typename ItTy>
  SmallVector(ItTy S, ItTy E) : SmallVectorImpl<T>(N) {
    this->append(S, E);
  }

  template <typename RangeTy>
  explicit SmallVector(const llvm::iterator_range<RangeTy> R)
      : SmallVectorImpl<T>(N) {
    this->append(R.begin(), R.end());
  }

  SmallVector(std::initializer_list<T> IL) : SmallVectorImpl<T>(N) {
    this->assign(IL);
  }

  SmallVector(const SmallVector &RHS) : SmallVectorImpl<T>(N) {
    if (!RHS.empty())
      SmallVectorImpl<T>::operator=(RHS);
  }

  const SmallVector &operator=(const SmallVector &RHS) {
    SmallVectorImpl<T>::operator=(RHS);
    return *this;
  }

  SmallVector(SmallVector &&RHS) : SmallVectorImpl<T>(N) {
    if (!RHS.empty())
      SmallVectorImpl<T>::operator=(::std::move(RHS));
  }

  const SmallVector &operator=(SmallVector &&RHS) {
    SmallVectorImpl<T>::operator=(::std::move(RHS));
    return *this;
  }

  SmallVector(SmallVectorImpl<T> &&RHS) : SmallVectorImpl<T>(N) {
    if (!RHS.empty())
      SmallVectorImpl<T>::operator=(::std::move(RHS));
  }

  const SmallVector &operator=(SmallVectorImpl<T> &&RHS) {
    SmallVectorImpl<T>::operator=(::std::move(RHS));
    return *this;
  }

  const SmallVector &operator=(std::initializer_list<T> IL) {
    this->assign(IL);
    return *this;
  }
};

template<typename T, unsigned N>
static inline size_t capacity_in_bytes(const SmallVector<T, N> &X) {
  return X.capacity_in_bytes();
}

} // End llvm namespace

namespace std {
  /// Implement std::swap in terms of SmallVector swap.
  template<typename T>
  inline void
  swap(llvm::SmallVectorImpl<T> &LHS, llvm::SmallVectorImpl<T> &RHS) {
    LHS.swap(RHS);
  }

  /// Implement std::swap in terms of SmallVector swap.
  template<typename T, unsigned N>
  inline void
  swap(llvm::SmallVector<T, N> &LHS, llvm::SmallVector<T, N> &RHS) {
    LHS.swap(RHS);
  }
}

#endif
