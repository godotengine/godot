//===- llvm/Support/ErrorOr.h - Error Smart Pointer -----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
///
/// Provides ErrorOr<T> smart pointer.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_ERROROR_H
#define LLVM_SUPPORT_ERROROR_H

#include "llvm/ADT/PointerIntPair.h"
#include "llvm/Support/AlignOf.h"
#include <cassert>
#include <system_error>
#include <type_traits>

namespace llvm {
template<class T, class V>
typename std::enable_if< std::is_constructible<T, V>::value
                       , typename std::remove_reference<V>::type>::type &&
 moveIfMoveConstructible(V &Val) {
  return std::move(Val);
}

template<class T, class V>
typename std::enable_if< !std::is_constructible<T, V>::value
                       , typename std::remove_reference<V>::type>::type &
moveIfMoveConstructible(V &Val) {
  return Val;
}

/// \brief Stores a reference that can be changed.
template <typename T>
class ReferenceStorage {
  T *Storage;

public:
  ReferenceStorage(T &Ref) : Storage(&Ref) {}

  operator T &() const { return *Storage; }
  T &get() const { return *Storage; }
};

/// \brief Represents either an error or a value T.
///
/// ErrorOr<T> is a pointer-like class that represents the result of an
/// operation. The result is either an error, or a value of type T. This is
/// designed to emulate the usage of returning a pointer where nullptr indicates
/// failure. However instead of just knowing that the operation failed, we also
/// have an error_code and optional user data that describes why it failed.
///
/// It is used like the following.
/// \code
///   ErrorOr<Buffer> getBuffer();
///
///   auto buffer = getBuffer();
///   if (error_code ec = buffer.getError())
///     return ec;
///   buffer->write("adena");
/// \endcode
///
///
/// Implicit conversion to bool returns true if there is a usable value. The
/// unary * and -> operators provide pointer like access to the value. Accessing
/// the value when there is an error has undefined behavior.
///
/// When T is a reference type the behaivor is slightly different. The reference
/// is held in a std::reference_wrapper<std::remove_reference<T>::type>, and
/// there is special handling to make operator -> work as if T was not a
/// reference.
///
/// T cannot be a rvalue reference.
template<class T>
class ErrorOr {
  template <class OtherT> friend class ErrorOr;
  static const bool isRef = std::is_reference<T>::value;
  typedef ReferenceStorage<typename std::remove_reference<T>::type> wrap;

public:
  typedef typename std::conditional<isRef, wrap, T>::type storage_type;

private:
  typedef typename std::remove_reference<T>::type &reference;
  typedef const typename std::remove_reference<T>::type &const_reference;
  typedef typename std::remove_reference<T>::type *pointer;

public:
  template <class E>
  ErrorOr(E ErrorCode,
          typename std::enable_if<std::is_error_code_enum<E>::value ||
                                      std::is_error_condition_enum<E>::value,
                                  void *>::type = 0)
      : HasError(true) {
    new (getErrorStorage()) std::error_code(make_error_code(ErrorCode));
  }

  ErrorOr(std::error_code EC) : HasError(true) {
    new (getErrorStorage()) std::error_code(EC);
  }

  ErrorOr(T Val) : HasError(false) {
    new (getStorage()) storage_type(moveIfMoveConstructible<storage_type>(Val));
  }

  ErrorOr(const ErrorOr &Other) {
    copyConstruct(Other);
  }

  template <class OtherT>
  ErrorOr(
      const ErrorOr<OtherT> &Other,
      typename std::enable_if<std::is_convertible<OtherT, T>::value>::type * =
          nullptr) {
    copyConstruct(Other);
  }

  template <class OtherT>
  explicit ErrorOr(
      const ErrorOr<OtherT> &Other,
      typename std::enable_if<
          !std::is_convertible<OtherT, const T &>::value>::type * = nullptr) {
    copyConstruct(Other);
  }

  ErrorOr(ErrorOr &&Other) {
    moveConstruct(std::move(Other));
  }

  template <class OtherT>
  ErrorOr(
      ErrorOr<OtherT> &&Other,
      typename std::enable_if<std::is_convertible<OtherT, T>::value>::type * =
          nullptr) {
    moveConstruct(std::move(Other));
  }

  // This might eventually need SFINAE but it's more complex than is_convertible
  // & I'm too lazy to write it right now.
  template <class OtherT>
  explicit ErrorOr(
      ErrorOr<OtherT> &&Other,
      typename std::enable_if<!std::is_convertible<OtherT, T>::value>::type * =
          nullptr) {
    moveConstruct(std::move(Other));
  }

  ErrorOr &operator=(const ErrorOr &Other) {
    copyAssign(Other);
    return *this;
  }

  ErrorOr &operator=(ErrorOr &&Other) {
    moveAssign(std::move(Other));
    return *this;
  }

  ~ErrorOr() {
    if (!HasError)
      getStorage()->~storage_type();
  }

  /// \brief Return false if there is an error.
  explicit operator bool() const {
    return !HasError;
  }

  reference get() { return *getStorage(); }
  const_reference get() const { return const_cast<ErrorOr<T> *>(this)->get(); }

  std::error_code getError() const {
    return HasError ? *getErrorStorage() : std::error_code();
  }

  pointer operator ->() {
    return toPointer(getStorage());
  }

  reference operator *() {
    return *getStorage();
  }

private:
  template <class OtherT>
  void copyConstruct(const ErrorOr<OtherT> &Other) {
    if (!Other.HasError) {
      // Get the other value.
      HasError = false;
      new (getStorage()) storage_type(*Other.getStorage());
    } else {
      // Get other's error.
      HasError = true;
      new (getErrorStorage()) std::error_code(Other.getError());
    }
  }

  template <class T1>
  static bool compareThisIfSameType(const T1 &a, const T1 &b) {
    return &a == &b;
  }

  template <class T1, class T2>
  static bool compareThisIfSameType(const T1 &a, const T2 &b) {
    return false;
  }

  template <class OtherT>
  void copyAssign(const ErrorOr<OtherT> &Other) {
    if (compareThisIfSameType(*this, Other))
      return;

    this->~ErrorOr();
    new (this) ErrorOr(Other);
  }

  template <class OtherT>
  void moveConstruct(ErrorOr<OtherT> &&Other) {
    if (!Other.HasError) {
      // Get the other value.
      HasError = false;
      new (getStorage()) storage_type(std::move(*Other.getStorage()));
    } else {
      // Get other's error.
      HasError = true;
      new (getErrorStorage()) std::error_code(Other.getError());
    }
  }

  template <class OtherT>
  void moveAssign(ErrorOr<OtherT> &&Other) {
    if (compareThisIfSameType(*this, Other))
      return;

    this->~ErrorOr();
    new (this) ErrorOr(std::move(Other));
  }

  pointer toPointer(pointer Val) {
    return Val;
  }

  pointer toPointer(wrap *Val) {
    return &Val->get();
  }

  storage_type *getStorage() {
    assert(!HasError && "Cannot get value when an error exists!");
    return reinterpret_cast<storage_type*>(TStorage.buffer);
  }

  const storage_type *getStorage() const {
    assert(!HasError && "Cannot get value when an error exists!");
    return reinterpret_cast<const storage_type*>(TStorage.buffer);
  }

  std::error_code *getErrorStorage() {
    assert(HasError && "Cannot get error when a value exists!");
    return reinterpret_cast<std::error_code *>(ErrorStorage.buffer);
  }

  const std::error_code *getErrorStorage() const {
    return const_cast<ErrorOr<T> *>(this)->getErrorStorage();
  }


  union {
    AlignedCharArrayUnion<storage_type> TStorage;
    AlignedCharArrayUnion<std::error_code> ErrorStorage;
  };
  bool HasError : 1;
};

template <class T, class E>
typename std::enable_if<std::is_error_code_enum<E>::value ||
                            std::is_error_condition_enum<E>::value,
                        bool>::type
operator==(const ErrorOr<T> &Err, E Code) {
  return Err.getError() == Code;
}
} // end namespace llvm

#endif
