// Copyright 2007 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef UTIL_MUTEX_H_
#define UTIL_MUTEX_H_

/*
 * A simple mutex wrapper, supporting locks and read-write locks.
 * You should assume the locks are *not* re-entrant.
 */

#if !defined(_WIN32)
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include <unistd.h>
#if defined(_POSIX_READER_WRITER_LOCKS) && _POSIX_READER_WRITER_LOCKS > 0
#define MUTEX_IS_PTHREAD_RWLOCK
#endif
#endif

#if defined(MUTEX_IS_PTHREAD_RWLOCK)
#include <pthread.h>
#include <stdlib.h>
typedef pthread_rwlock_t MutexType;
#else
#include <mutex>
typedef std::mutex MutexType;
#endif

namespace re2 {

class Mutex {
 public:
  inline Mutex();
  inline ~Mutex();
  inline void Lock();    // Block if needed until free then acquire exclusively
  inline void Unlock();  // Release a lock acquired via Lock()
  // Note that on systems that don't support read-write locks, these may
  // be implemented as synonyms to Lock() and Unlock().  So you can use
  // these for efficiency, but don't use them anyplace where being able
  // to do shared reads is necessary to avoid deadlock.
  inline void ReaderLock();   // Block until free or shared then acquire a share
  inline void ReaderUnlock(); // Release a read share of this Mutex
  inline void WriterLock() { Lock(); }     // Acquire an exclusive lock
  inline void WriterUnlock() { Unlock(); } // Release a lock from WriterLock()

 private:
  MutexType mutex_;

  // Catch the error of writing Mutex when intending MutexLock.
  Mutex(Mutex *ignored);

  Mutex(const Mutex&) = delete;
  Mutex& operator=(const Mutex&) = delete;
};

#if defined(MUTEX_IS_PTHREAD_RWLOCK)

#define SAFE_PTHREAD(fncall)    \
  do {                          \
    if ((fncall) != 0) abort(); \
  } while (0)

Mutex::Mutex()             { SAFE_PTHREAD(pthread_rwlock_init(&mutex_, NULL)); }
Mutex::~Mutex()            { SAFE_PTHREAD(pthread_rwlock_destroy(&mutex_)); }
void Mutex::Lock()         { SAFE_PTHREAD(pthread_rwlock_wrlock(&mutex_)); }
void Mutex::Unlock()       { SAFE_PTHREAD(pthread_rwlock_unlock(&mutex_)); }
void Mutex::ReaderLock()   { SAFE_PTHREAD(pthread_rwlock_rdlock(&mutex_)); }
void Mutex::ReaderUnlock() { SAFE_PTHREAD(pthread_rwlock_unlock(&mutex_)); }

#undef SAFE_PTHREAD

#else

Mutex::Mutex()             { }
Mutex::~Mutex()            { }
void Mutex::Lock()         { mutex_.lock(); }
void Mutex::Unlock()       { mutex_.unlock(); }
void Mutex::ReaderLock()   { Lock(); }  // C++11 doesn't have std::shared_mutex.
void Mutex::ReaderUnlock() { Unlock(); }

#endif

// --------------------------------------------------------------------------
// Some helper classes

// MutexLock(mu) acquires mu when constructed and releases it when destroyed.
class MutexLock {
 public:
  explicit MutexLock(Mutex *mu) : mu_(mu) { mu_->Lock(); }
  ~MutexLock() { mu_->Unlock(); }
 private:
  Mutex * const mu_;

  MutexLock(const MutexLock&) = delete;
  MutexLock& operator=(const MutexLock&) = delete;
};

// ReaderMutexLock and WriterMutexLock do the same, for rwlocks
class ReaderMutexLock {
 public:
  explicit ReaderMutexLock(Mutex *mu) : mu_(mu) { mu_->ReaderLock(); }
  ~ReaderMutexLock() { mu_->ReaderUnlock(); }
 private:
  Mutex * const mu_;

  ReaderMutexLock(const ReaderMutexLock&) = delete;
  ReaderMutexLock& operator=(const ReaderMutexLock&) = delete;
};

class WriterMutexLock {
 public:
  explicit WriterMutexLock(Mutex *mu) : mu_(mu) { mu_->WriterLock(); }
  ~WriterMutexLock() { mu_->WriterUnlock(); }
 private:
  Mutex * const mu_;

  WriterMutexLock(const WriterMutexLock&) = delete;
  WriterMutexLock& operator=(const WriterMutexLock&) = delete;
};

// Catch bug where variable name is omitted, e.g. MutexLock (&mu);
#define MutexLock(x) static_assert(false, "MutexLock declaration missing variable name")
#define ReaderMutexLock(x) static_assert(false, "ReaderMutexLock declaration missing variable name")
#define WriterMutexLock(x) static_assert(false, "WriterMutexLock declaration missing variable name")

}  // namespace re2

#endif  // UTIL_MUTEX_H_
