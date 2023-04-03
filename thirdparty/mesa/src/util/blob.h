/*
 * Copyright © 2014 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef BLOB_H
#define BLOB_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The blob functions implement a simple, low-level API for serializing and
 * deserializing.
 *
 * All objects written to a blob will be serialized directly, (without any
 * additional meta-data to describe the data written). Therefore, it is the
 * caller's responsibility to ensure that any data can be read later, (either
 * by knowing exactly what data is expected, or by writing to the blob
 * sufficient meta-data to describe what has been written).
 *
 * A blob is efficient in that it dynamically grows by doubling in size, so
 * allocation costs are logarithmic.
 */

struct blob {
   /* The data actually written to the blob. Never read or write this directly
    * when serializing, use blob_reserve_* and blob_overwrite_* instead which
    * check for out_of_memory and handle fixed-size blobs correctly.
    */
   uint8_t *data;

   /** Number of bytes that have been allocated for \c data. */
   size_t allocated;

   /** The number of bytes that have actual data written to them. */
   size_t size;

   /** True if \c data a fixed allocation that we cannot resize
    *
    * \see blob_init_fixed
    */
   bool fixed_allocation;

   /**
    * True if we've ever failed to realloc or if we go pas the end of a fixed
    * allocation blob.
    */
   bool out_of_memory;
};

/* When done reading, the caller can ensure that everything was consumed by
 * checking the following:
 *
 *   1. blob->current should be equal to blob->end, (if not, too little was
 *      read).
 *
 *   2. blob->overrun should be false, (otherwise, too much was read).
 */
struct blob_reader {
   const uint8_t *data;
   const uint8_t *end;
   const uint8_t *current;
   bool overrun;
};

/**
 * Init a new, empty blob.
 */
void
blob_init(struct blob *blob);

/**
 * Init a new, fixed-size blob.
 *
 * A fixed-size blob has a fixed block of data that will not be freed on
 * blob_finish and will never be grown.  If we hit the end, we simply start
 * returning false from the write functions.
 *
 * If a fixed-size blob has a NULL data pointer then the data is written but
 * it otherwise operates normally.  This can be used to determine the size
 * that will be required to write a given data structure.
 */
void
blob_init_fixed(struct blob *blob, void *data, size_t size);

/**
 * Finish a blob and free its memory.
 *
 * If \blob was initialized with blob_init_fixed, the data pointer is
 * considered to be owned by the user and will not be freed.
 */
static inline void
blob_finish(struct blob *blob)
{
   if (!blob->fixed_allocation)
      free(blob->data);
}

void
blob_finish_get_buffer(struct blob *blob, void **buffer, size_t *size);

/**
 * Aligns the blob to the given alignment.
 *
 * \see blob_reader_align
 *
 * \return True unless allocation fails
 */
bool
blob_align(struct blob *blob, size_t alignment);

/**
 * Add some unstructured, fixed-size data to a blob.
 *
 * \return True unless allocation failed.
 */
bool
blob_write_bytes(struct blob *blob, const void *bytes, size_t to_write);

/**
 * Reserve space in \blob for a number of bytes.
 *
 * Space will be allocated within the blob for these byes, but the bytes will
 * be left uninitialized. The caller is expected to use \sa
 * blob_overwrite_bytes to write to these bytes.
 *
 * \return An offset to space allocated within \blob to which \to_write bytes
 * can be written, (or -1 in case of any allocation error).
 */
intptr_t
blob_reserve_bytes(struct blob *blob, size_t to_write);

/**
 * Similar to \sa blob_reserve_bytes, but only reserves an uint32_t worth of
 * space. Note that this must be used if later reading with \sa
 * blob_read_uint32, since it aligns the offset correctly.
 */
intptr_t
blob_reserve_uint32(struct blob *blob);

/**
 * Similar to \sa blob_reserve_bytes, but only reserves an intptr_t worth of
 * space. Note that this must be used if later reading with \sa
 * blob_read_intptr, since it aligns the offset correctly.
 */
intptr_t
blob_reserve_intptr(struct blob *blob);

/**
 * Overwrite some data previously written to the blob.
 *
 * Writes data to an existing portion of the blob at an offset of \offset.
 * This data range must have previously been written to the blob by one of the
 * blob_write_* calls.
 *
 * For example usage, see blob_overwrite_uint32
 *
 * \return True unless the requested offset or offset+to_write lie outside
 * the current blob's size.
 */
bool
blob_overwrite_bytes(struct blob *blob,
                     size_t offset,
                     const void *bytes,
                     size_t to_write);

/**
 * Add a uint8_t to a blob.
 *
 * \return True unless allocation failed.
 */
bool
blob_write_uint8(struct blob *blob, uint8_t value);

/**
 * Overwrite a uint8_t previously written to the blob.
 *
 * Writes a uint8_t value to an existing portion of the blob at an offset of
 * \offset.  This data range must have previously been written to the blob by
 * one of the blob_write_* calls.
 *
 * \return True unless the requested position or position+to_write lie outside
 * the current blob's size.
 */
bool
blob_overwrite_uint8(struct blob *blob,
                     size_t offset,
                     uint8_t value);

/**
 * Add a uint16_t to a blob.
 *
 * \note This function will only write to a uint16_t-aligned offset from the
 * beginning of the blob's data, so some padding bytes may be added to the
 * blob if this write follows some unaligned write (such as
 * blob_write_string).
 *
 * \return True unless allocation failed.
 */
bool
blob_write_uint16(struct blob *blob, uint16_t value);

/**
 * Add a uint32_t to a blob.
 *
 * \note This function will only write to a uint32_t-aligned offset from the
 * beginning of the blob's data, so some padding bytes may be added to the
 * blob if this write follows some unaligned write (such as
 * blob_write_string).
 *
 * \return True unless allocation failed.
 */
bool
blob_write_uint32(struct blob *blob, uint32_t value);

/**
 * Overwrite a uint32_t previously written to the blob.
 *
 * Writes a uint32_t value to an existing portion of the blob at an offset of
 * \offset.  This data range must have previously been written to the blob by
 * one of the blob_write_* calls.
 *
 *
 * The expected usage is something like the following pattern:
 *
 *	size_t offset;
 *
 *	offset = blob_reserve_uint32(blob);
 *	... various blob write calls, writing N items ...
 *	blob_overwrite_uint32 (blob, offset, N);
 *
 * \return True unless the requested position or position+to_write lie outside
 * the current blob's size.
 */
bool
blob_overwrite_uint32(struct blob *blob,
                      size_t offset,
                      uint32_t value);

/**
 * Add a uint64_t to a blob.
 *
 * \note This function will only write to a uint64_t-aligned offset from the
 * beginning of the blob's data, so some padding bytes may be added to the
 * blob if this write follows some unaligned write (such as
 * blob_write_string).
 *
 * \return True unless allocation failed.
 */
bool
blob_write_uint64(struct blob *blob, uint64_t value);

/**
 * Add an intptr_t to a blob.
 *
 * \note This function will only write to an intptr_t-aligned offset from the
 * beginning of the blob's data, so some padding bytes may be added to the
 * blob if this write follows some unaligned write (such as
 * blob_write_string).
 *
 * \return True unless allocation failed.
 */
bool
blob_write_intptr(struct blob *blob, intptr_t value);

/**
 * Overwrite an intptr_t previously written to the blob.
 *
 * Writes a intptr_t value to an existing portion of the blob at an offset of
 * \offset.  This data range must have previously been written to the blob by
 * one of the blob_write_* calls.
 *
 * For example usage, see blob_overwrite_uint32
 *
 * \return True unless the requested position or position+to_write lie outside
 * the current blob's size.
 */
bool
blob_overwrite_intptr(struct blob *blob,
                      size_t offset,
                      intptr_t value);

/**
 * Add a NULL-terminated string to a blob, (including the NULL terminator).
 *
 * \return True unless allocation failed.
 */
bool
blob_write_string(struct blob *blob, const char *str);

/**
 * Start reading a blob, (initializing the contents of \blob for reading).
 *
 * After this call, the caller can use the various blob_read_* functions to
 * read elements from the data array.
 *
 * For all of the blob_read_* functions, if there is insufficient data
 * remaining, the functions will do nothing, (perhaps returning default values
 * such as 0). The caller can detect this by noting that the blob_reader's
 * current value is unchanged before and after the call.
 */
void
blob_reader_init(struct blob_reader *blob, const void *data, size_t size);

/**
 * Align the current offset of the blob reader to the given alignment.
 *
 * This may be useful if you need the result of blob_read_bytes to have a
 * particular alignment.  Note that this only aligns relative to blob->data
 * and the alignment of the resulting pointer is only guaranteed if blob->data
 * is also aligned to the requested alignment.
 */
void
blob_reader_align(struct blob_reader *blob, size_t alignment);

/**
 * Read some unstructured, fixed-size data from the current location, (and
 * update the current location to just past this data).
 *
 * \note The memory returned belongs to the data underlying the blob reader. The
 * caller must copy the data in order to use it after the lifetime of the data
 * underlying the blob reader.
 *
 * \return The bytes read (see note above about memory lifetime).
 */
const void *
blob_read_bytes(struct blob_reader *blob, size_t size);

/**
 * Read some unstructured, fixed-size data from the current location, copying
 * it to \dest (and update the current location to just past this data)
 */
void
blob_copy_bytes(struct blob_reader *blob, void *dest, size_t size);

/**
 * Skip \size bytes within the blob.
 */
void
blob_skip_bytes(struct blob_reader *blob, size_t size);

/**
 * Read a uint8_t from the current location, (and update the current location
 * to just past this uint8_t).
 *
 * \return The uint8_t read
 */
uint8_t
blob_read_uint8(struct blob_reader *blob);

/**
 * Read a uint16_t from the current location, (and update the current location
 * to just past this uint16_t).
 *
 * \note This function will only read from a uint16_t-aligned offset from the
 * beginning of the blob's data, so some padding bytes may be skipped.
 *
 * \return The uint16_t read
 */
uint16_t
blob_read_uint16(struct blob_reader *blob);

/**
 * Read a uint32_t from the current location, (and update the current location
 * to just past this uint32_t).
 *
 * \note This function will only read from a uint32_t-aligned offset from the
 * beginning of the blob's data, so some padding bytes may be skipped.
 *
 * \return The uint32_t read
 */
uint32_t
blob_read_uint32(struct blob_reader *blob);

/**
 * Read a uint64_t from the current location, (and update the current location
 * to just past this uint64_t).
 *
 * \note This function will only read from a uint64_t-aligned offset from the
 * beginning of the blob's data, so some padding bytes may be skipped.
 *
 * \return The uint64_t read
 */
uint64_t
blob_read_uint64(struct blob_reader *blob);

/**
 * Read an intptr_t value from the current location, (and update the
 * current location to just past this intptr_t).
 *
 * \note This function will only read from an intptr_t-aligned offset from the
 * beginning of the blob's data, so some padding bytes may be skipped.
 *
 * \return The intptr_t read
 */
intptr_t
blob_read_intptr(struct blob_reader *blob);

/**
 * Read a NULL-terminated string from the current location, (and update the
 * current location to just past this string).
 *
 * \note The memory returned belongs to the data underlying the blob reader. The
 * caller must copy the string in order to use the string after the lifetime
 * of the data underlying the blob reader.
 *
 * \return The string read (see note above about memory lifetime). However, if
 * there is no NULL byte remaining within the blob, this function returns
 * NULL.
 */
char *
blob_read_string(struct blob_reader *blob);

#ifdef __cplusplus
}
#endif

#endif /* BLOB_H */
