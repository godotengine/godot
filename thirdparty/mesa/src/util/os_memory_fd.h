/**************************************************************************
 *
 * Copyright 2021 Snap Inc.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/


/*
 * Memory fd wrappers.
 */


#ifndef _OS_MEMORY_H_
#error "Must not be included directly. Include os_memory.h instead"
#endif

/**
 * Imports memory from a file descriptor
 */
bool
os_import_memory_fd(int fd, void **ptr, uint64_t *size, char const *driver_id);

/**
 * Return memory on given byte alignment
 */
void *
os_malloc_aligned_fd(size_t size, size_t alignment, int *fd, char const *fd_name, char const *driver_id);

/**
 * Free memory returned by os_malloc_aligned_fd().
 */
void
os_free_fd(void *ptr);
