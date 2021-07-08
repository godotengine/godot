//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// macros.h: Compatiblity hacks for importing Chromium's MRUCache.

#ifndef ANGLEBASE_MACROS_H_
#define ANGLEBASE_MACROS_H_

// A macro to disallow the copy constructor and operator= functions.
// This should be used in the private: declarations for a class.
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
    TypeName(const TypeName &) = delete;   \
    void operator=(const TypeName &) = delete

#endif  // ANGLEBASE_MACROS_H_
