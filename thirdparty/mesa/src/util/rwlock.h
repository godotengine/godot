/**************************************************************************
 *
 * Copyright 2020 Lag Free Games, LLC
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/

#ifndef RWLOCK_H
#define RWLOCK_H

#if defined(HAVE_PTHREAD)
#include <pthread.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct u_rwlock
{
#if defined(_WIN32) && !defined(HAVE_PTHREAD)
   struct {
      void *Ptr;
   } rwlock;
#else
   pthread_rwlock_t rwlock;
#endif
};

int u_rwlock_init(struct u_rwlock *rwlock);
int u_rwlock_destroy(struct u_rwlock *rwlock);
int u_rwlock_rdlock(struct u_rwlock *rwlock);
int u_rwlock_rdunlock(struct u_rwlock *rwlock);
int u_rwlock_wrlock(struct u_rwlock *rwlock);
int u_rwlock_wrunlock(struct u_rwlock *rwlock);

#ifdef __cplusplus
}
#endif

#endif
