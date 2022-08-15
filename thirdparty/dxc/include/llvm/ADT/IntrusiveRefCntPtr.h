//== llvm/ADT/IntrusiveRefCntPtr.h - Smart Refcounting Pointer ---*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines IntrusiveRefCntPtr, a template class that
// implements a "smart" pointer for objects that maintain their own
// internal reference count, and RefCountedBase/RefCountedBaseVPTR, two
// generic base classes for objects that wish to have their lifetimes
// managed using reference counting.
//
// IntrusiveRefCntPtr is similar to Boost's intrusive_ptr with added
// LLVM-style casting.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_INTRUSIVEREFCNTPTR_H
#define LLVM_ADT_INTRUSIVEREFCNTPTR_H

#include <atomic>
#include <cassert>
#include <cstddef>

namespace llvm {

  template <class T>
  class IntrusiveRefCntPtr;

//===----------------------------------------------------------------------===//
/// RefCountedBase - A generic base class for objects that wish to
///  have their lifetimes managed using reference counts. Classes
///  subclass RefCountedBase to obtain such functionality, and are
///  typically handled with IntrusiveRefCntPtr "smart pointers" (see below)
///  which automatically handle the management of reference counts.
///  Objects that subclass RefCountedBase should not be allocated on
///  the stack, as invoking "delete" (which is called when the
///  reference count hits 0) on such objects is an error.
//===----------------------------------------------------------------------===//
  template <class Derived>
  class RefCountedBase {
    mutable unsigned ref_cnt;

  public:
    RefCountedBase() : ref_cnt(0) {}
    RefCountedBase(const RefCountedBase &) : ref_cnt(0) {}

    void Retain() const { ++ref_cnt; }
    void Release() const {
      assert (ref_cnt > 0 && "Reference count is already zero.");
      if (--ref_cnt == 0) delete static_cast<const Derived*>(this);
    }
  };

//===----------------------------------------------------------------------===//
/// RefCountedBaseVPTR - A class that has the same function as
///  RefCountedBase, but with a virtual destructor. Should be used
///  instead of RefCountedBase for classes that already have virtual
///  methods to enforce dynamic allocation via 'new'. Classes that
///  inherit from RefCountedBaseVPTR can't be allocated on stack -
///  attempting to do this will produce a compile error.
//===----------------------------------------------------------------------===//
  class RefCountedBaseVPTR {
    mutable unsigned ref_cnt;
    virtual void anchor();

  protected:
    RefCountedBaseVPTR() : ref_cnt(0) {}
    RefCountedBaseVPTR(const RefCountedBaseVPTR &) : ref_cnt(0) {}

    virtual ~RefCountedBaseVPTR() {}

    void Retain() const { ++ref_cnt; }
    void Release() const {
      assert (ref_cnt > 0 && "Reference count is already zero.");
      if (--ref_cnt == 0) delete this;
    }

    template <typename T>
    friend struct IntrusiveRefCntPtrInfo;
  };


  template <typename T> struct IntrusiveRefCntPtrInfo {
    static void retain(T *obj) { obj->Retain(); }
    static void release(T *obj) { obj->Release(); }
  };

/// \brief A thread-safe version of \c llvm::RefCountedBase.
///
/// A generic base class for objects that wish to have their lifetimes managed
/// using reference counts. Classes subclass \c ThreadSafeRefCountedBase to
/// obtain such functionality, and are typically handled with
/// \c IntrusiveRefCntPtr "smart pointers" which automatically handle the
/// management of reference counts.
template <class Derived>
class ThreadSafeRefCountedBase {
  mutable std::atomic<int> RefCount;

protected:
  ThreadSafeRefCountedBase() : RefCount(0) {}

public:
  void Retain() const { ++RefCount; }

  void Release() const {
    int NewRefCount = --RefCount;
    assert(NewRefCount >= 0 && "Reference count was already zero.");
    if (NewRefCount == 0)
      delete static_cast<const Derived*>(this);
  }
};

//===----------------------------------------------------------------------===//
/// IntrusiveRefCntPtr - A template class that implements a "smart pointer"
///  that assumes the wrapped object has a reference count associated
///  with it that can be managed via calls to
///  IntrusivePtrAddRef/IntrusivePtrRelease.  The smart pointers
///  manage reference counts via the RAII idiom: upon creation of
///  smart pointer the reference count of the wrapped object is
///  incremented and upon destruction of the smart pointer the
///  reference count is decremented.  This class also safely handles
///  wrapping NULL pointers.
///
/// Reference counting is implemented via calls to
///  Obj->Retain()/Obj->Release(). Release() is required to destroy
///  the object when the reference count reaches zero. Inheriting from
///  RefCountedBase/RefCountedBaseVPTR takes care of this
///  automatically.
//===----------------------------------------------------------------------===//
  template <typename T>
  class IntrusiveRefCntPtr {
    T* Obj;

  public:
    typedef T element_type;

    explicit IntrusiveRefCntPtr() : Obj(nullptr) {}

    IntrusiveRefCntPtr(T* obj) : Obj(obj) {
      retain();
    }

    IntrusiveRefCntPtr(const IntrusiveRefCntPtr& S) : Obj(S.Obj) {
      retain();
    }

    IntrusiveRefCntPtr(IntrusiveRefCntPtr&& S) : Obj(S.Obj) {
      S.Obj = nullptr;
    }

    template <class X>
    IntrusiveRefCntPtr(IntrusiveRefCntPtr<X>&& S) : Obj(S.get()) {
      S.Obj = 0;
    }

    template <class X>
    IntrusiveRefCntPtr(const IntrusiveRefCntPtr<X>& S)
      : Obj(S.get()) {
      retain();
    }

    IntrusiveRefCntPtr& operator=(IntrusiveRefCntPtr S) {
      swap(S);
      return *this;
    }

    ~IntrusiveRefCntPtr() { release(); }

    T& operator*() const { return *Obj; }

    T* operator->() const { return Obj; }

    T* get() const { return Obj; }

    explicit operator bool() const { return Obj != nullptr; } // HLSL Change

    void swap(IntrusiveRefCntPtr& other) {
      T* tmp = other.Obj;
      other.Obj = Obj;
      Obj = tmp;
    }

    void reset() {
      release();
      Obj = nullptr;
    }

    void resetWithoutRelease() {
      Obj = 0;
    }

  private:
    void retain() { if (Obj) IntrusiveRefCntPtrInfo<T>::retain(Obj); }
    void release() { if (Obj) IntrusiveRefCntPtrInfo<T>::release(Obj); }

    template <typename X>
    friend class IntrusiveRefCntPtr;
  };

  template<class T, class U>
  inline bool operator==(const IntrusiveRefCntPtr<T>& A,
                         const IntrusiveRefCntPtr<U>& B)
  {
    return A.get() == B.get();
  }

  template<class T, class U>
  inline bool operator!=(const IntrusiveRefCntPtr<T>& A,
                         const IntrusiveRefCntPtr<U>& B)
  {
    return A.get() != B.get();
  }

  template<class T, class U>
  inline bool operator==(const IntrusiveRefCntPtr<T>& A,
                         U* B)
  {
    return A.get() == B;
  }

  template<class T, class U>
  inline bool operator!=(const IntrusiveRefCntPtr<T>& A,
                         U* B)
  {
    return A.get() != B;
  }

  template<class T, class U>
  inline bool operator==(T* A,
                         const IntrusiveRefCntPtr<U>& B)
  {
    return A == B.get();
  }

  template<class T, class U>
  inline bool operator!=(T* A,
                         const IntrusiveRefCntPtr<U>& B)
  {
    return A != B.get();
  }

  template <class T>
  bool operator==(std::nullptr_t A, const IntrusiveRefCntPtr<T> &B) {
    return !B;
  }

  template <class T>
  bool operator==(const IntrusiveRefCntPtr<T> &A, std::nullptr_t B) {
    return B == A;
  }

  template <class T>
  bool operator!=(std::nullptr_t A, const IntrusiveRefCntPtr<T> &B) {
    return !(A == B);
  }

  template <class T>
  bool operator!=(const IntrusiveRefCntPtr<T> &A, std::nullptr_t B) {
    return !(A == B);
  }

//===----------------------------------------------------------------------===//
// LLVM-style downcasting support for IntrusiveRefCntPtr objects
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

  template <typename From> struct simplify_type;

  template<class T> struct simplify_type<IntrusiveRefCntPtr<T> > {
    typedef T* SimpleType;
    static SimpleType getSimplifiedValue(IntrusiveRefCntPtr<T>& Val) {
      return Val.get();
    }
  };

  template<class T> struct simplify_type<const IntrusiveRefCntPtr<T> > {
    typedef /*const*/ T* SimpleType;
    static SimpleType getSimplifiedValue(const IntrusiveRefCntPtr<T>& Val) {
      return Val.get();
    }
  };

} // end namespace llvm

#endif // LLVM_ADT_INTRUSIVEREFCNTPTR_H
