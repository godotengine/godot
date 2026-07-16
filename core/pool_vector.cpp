/**************************************************************************/
/*  pool_vector.cpp                                                       */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "pool_vector.h"

uint32_t MemoryPool::allocs_used = 0;

size_t MemoryPool::total_memory = 0;
size_t MemoryPool::max_memory = 0;

// ===================================================================
// GENERAL IMPLEMENTATION DETAILS
// ===================================================================
// PoolVector is a lightweight wrapper on LocalVector. The historic
// approach of compacting locked memory pools is obsolete on modern
// devices with virtual memory and 64-bit address spaces. Backing by
// LocalVector avoids high performance overhead, hardcoded limits,
// and multiple longstanding bugs.
//
// Element access via operator[] and get() return elements by value.
// This is the safest way to read data that may mutate (e.g. via COW).
// For large or complex structures, value copying can be slow; in
// such cases, consider using a scoped `Read` lock.

// ===================================================================
// MUTATION LOCKS AND THE "COW FOOTGUN"
// ===================================================================
// Historically, mutating functions (like resize) were completely
// blocked whenever any active lock existed. While this successfully
// prevented pointer invalidation within the same thread, it created
// a severe issue when PoolVectors were shared across threads (e.g.,
// passing data from the scene side to a server thread).
//
// If a server thread held a `Read` lock on a shared vector, the scene
// thread was blocked from calling `resize()`, resulting in the
// infamous and unpredictable "Can't resize PoolVector if locked" bug.
//
// Here we fix this problem by changing the locking paradigm to be
// thread local only. This prevents users from shooting themselves in
// the foot (by COWing a PoolVector that they have locked with an
// Access), but still allows PoolVectors on other threads (especially
// servers) to COW whenever they need.

// ===================================================================
// USAGE & THREAD SAFETY GUIDELINES
// ===================================================================
//
// 1. Thread Safety (Single vs. Multiple Handles)
// ----------------------------------------------
// * Single Handle: A single PoolVector instance is not thread-safe.
//   Do not read or write to the same instance concurrently from
//   multiple threads without external synchronization (e.g. Mutex).
// * Multiple Handles (Copies): It is safe to pass PoolVectors across
//   threads by making copies of the handle. Copying is a cheap,
//   O(1) operation that only copies a pointer and increments an
//   atomic reference count (SafeRefCount).
// * Concurrent access on copies:
//   - Concurrent reads across copies are completely safe.
//   - Writes on copies safely perform COW, decoupling the modified
//     handle.
//
// 2. Copy-on-Write (COW) Performance
// ----------------------------------
// * Modifying operations (set, push_back, remove, resize, etc.)
//   will clone the entire underlying LocalVector if shared
//   (refcount > 1).
// * Avoid maintaining unnecessary copies of the handle if you
//   intend to perform frequent modifications, as this will trigger
//   unwanted copy overhead on large datasets.
//
// 3. Element Access & Locking (Read and Write Scopes)
// ---------------------------------------------------
// * Operator[] and get() return elements by value. Avoid using
//   them in loops for large or complex structures.
// * To bypass value copy overhead, request a Read or Write lock:
//
//       {
//           PoolVector<MyStruct>::Read r = my_vector.read();
//           const MyStruct &element = r[index]; // Fast reference
//       } // Lock is automatically released here.
//
// * Keep Read and Write scopes as brief as possible. Do not store
//   Access locks for long-term use.
//
// 4. Pointer / Reference Invalidation
// -----------------------------------
// * Never store raw pointers (T*) or references (T&) beyond the
//   lifetime of the active Read or Write lock.
//   Modifications, resizing, or COW events can reallocate the
//   underlying buffer, immediately invalidating old pointers.
