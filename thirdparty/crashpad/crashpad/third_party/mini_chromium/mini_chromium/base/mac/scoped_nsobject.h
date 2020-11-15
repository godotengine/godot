// Copyright 2006-2008 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef MINI_CHROMIUM_BASE_MAC_SCOPED_NSOBJECT_H_
#define MINI_CHROMIUM_BASE_MAC_SCOPED_NSOBJECT_H_

#import <Foundation/Foundation.h>

#include <type_traits>

#include "base/compiler_specific.h"
#include "base/mac/scoped_typeref.h"

namespace base {

namespace internal {

template <typename NST>
struct ScopedNSProtocolTraits {
  static NST InvalidValue() { return nil; }
  static NST Retain(NST nst) { return [nst retain]; }
  static void Release(NST nst) { [nst release]; }
};

}  // namespace internal

template <typename NST>
class scoped_nsprotocol
    : public ScopedTypeRef<NST, internal::ScopedNSProtocolTraits<NST>> {
 public:
  using ScopedTypeRef<NST,
                      internal::ScopedNSProtocolTraits<NST>>::ScopedTypeRef;

  NST autorelease() { return [this->release() autorelease]; }
};

template <class C>
void swap(scoped_nsprotocol<C>& p1, scoped_nsprotocol<C>& p2) {
  p1.swap(p2);
}

template <class C>
bool operator==(C p1, const scoped_nsprotocol<C>& p2) {
  return p1 == p2.get();
}

template <class C>
bool operator!=(C p1, const scoped_nsprotocol<C>& p2) {
  return p1 != p2.get();
}

template <typename NST>
class scoped_nsobject : public scoped_nsprotocol<NST*> {
 public:
  using scoped_nsprotocol<NST*>::scoped_nsprotocol;

  static_assert(std::is_same<NST, NSAutoreleasePool>::value == false,
                "Use ScopedNSAutoreleasePool instead");
};

template<>
class scoped_nsobject<id> : public scoped_nsprotocol<id> {
 public:
  using scoped_nsprotocol<id>::scoped_nsprotocol;
};

}  // namespace base

#endif  // MINI_CHROMIUM_BASE_MAC_SCOPED_NSOBJECT_H_
