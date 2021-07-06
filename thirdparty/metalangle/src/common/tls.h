//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// tls.h: Simple cross-platform interface for thread local storage.

#ifndef COMMON_TLS_H_
#define COMMON_TLS_H_

#include "common/platform.h"

#ifdef ANGLE_PLATFORM_WINDOWS

// TLS does not exist for Windows Store and needs to be emulated
#    ifdef ANGLE_ENABLE_WINDOWS_UWP
#        ifndef TLS_OUT_OF_INDEXES
#            define TLS_OUT_OF_INDEXES static_cast<DWORD>(0xFFFFFFFF)
#        endif
#        ifndef CREATE_SUSPENDED
#            define CREATE_SUSPENDED 0x00000004
#        endif
#    endif
typedef DWORD TLSIndex;
#    define TLS_INVALID_INDEX (TLS_OUT_OF_INDEXES)
#elif defined(ANGLE_PLATFORM_POSIX)
#    include <errno.h>
#    include <pthread.h>
#    include <semaphore.h>
typedef pthread_key_t TLSIndex;
#    define TLS_INVALID_INDEX (static_cast<TLSIndex>(-1))
#else
#    error Unsupported platform.
#endif

// TODO(kbr): for POSIX platforms this will have to be changed to take
// in a destructor function pointer, to allow the thread-local storage
// to be properly deallocated upon thread exit.
TLSIndex CreateTLSIndex();
bool DestroyTLSIndex(TLSIndex index);

bool SetTLSValue(TLSIndex index, void *value);
void *GetTLSValue(TLSIndex index);

#endif  // COMMON_TLS_H_
