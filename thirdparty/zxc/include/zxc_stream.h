/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file zxc_stream.h
 * @brief @c FILE*-flavored variants of the ZXC API.
 *
 * Groups the public entry points that depend on @c <stdio.h> so they can be
 * cleanly excluded from kernel / freestanding builds (which include
 * @c zxc_buffer.h, @c zxc_pstream.h, and the storage-agnostic part of
 * @c zxc_seekable.h instead).
 *
 * Two subsystems live here:
 *
 * 1. **Multi-threaded streaming driver**: reads from a @c FILE* input and
 *    writes compressed (or decompressed) output to a @c FILE*.  Internally
 *    the driver uses an asynchronous Producer-Consumer pipeline via a ring
 *    buffer to separate I/O from CPU-intensive work:
 *      - Reader thread: reads chunks from @c f_in.
 *      - Worker threads: compress/decompress chunks in parallel.
 *      - Writer thread: orders the results and writes them to @c f_out.
 *    Functions: @ref zxc_stream_compress, @ref zxc_stream_decompress,
 *    @ref zxc_stream_get_decompressed_size.
 *
 * 2. **Seekable @c FILE* open helper**: thin wrapper that adapts a
 *    @c FILE* into a thread-safe @c pread / @c ReadFile-backed
 *    @ref zxc_reader_t and delegates to @ref zxc_seekable_open_reader.
 *    Function: @ref zxc_seekable_open_file.
 *
 * @see zxc_buffer.h   for the simple one-shot buffer API.
 * @see zxc_pstream.h  for single-threaded push-based streaming.
 * @see zxc_seekable.h for the storage-agnostic seekable reader.
 */

#ifndef ZXC_STREAM_H
#define ZXC_STREAM_H

#include <stdint.h>
#include <stdio.h>

#include "zxc_export.h"
#include "zxc_opts.h"
#include "zxc_seekable.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup stream_api Streaming API
 * @brief Multi-threaded, FILE*-based compression and decompression.
 * @{
 */

/**
 * @brief Compresses data from an input stream to an output stream.
 *
 * This function sets up a multi-threaded pipeline:
 * 1. Reader Thread: Reads chunks from f_in.
 * 2. Worker Threads: Compress chunks in parallel (LZ77 + Bitpacking).
 * 3. Writer Thread: Orders the processed chunks and writes them to f_out.
 *
 * @param[in] f_in   Input file stream (must be opened in "rb" mode).
 * @param[out] f_out  Output file stream (must be opened in "wb" mode).
 * @param[in] opts   Compression options (NULL uses all defaults).
 *
 * @return Total compressed bytes written, or a negative zxc_error_t code (e.g.,
 * ZXC_ERROR_IO) if an error occurred.
 */
ZXC_EXPORT int64_t zxc_stream_compress(FILE* f_in, FILE* f_out, const zxc_compress_opts_t* opts);

/**
 * @brief Decompresses data from an input stream to an output stream.
 *
 * Uses the same pipeline architecture as compression to maximize throughput.
 *
 * @param[in] f_in   Input file stream (must be opened in "rb" mode).
 * @param[out] f_out  Output file stream (must be opened in "wb" mode).
 * @param[in] opts   Decompression options (NULL uses all defaults).
 *
 * @return Total decompressed bytes written, or a negative zxc_error_t code (e.g.,
 * ZXC_ERROR_BAD_HEADER) if an error occurred.
 */
ZXC_EXPORT int64_t zxc_stream_decompress(FILE* f_in, FILE* f_out,
                                         const zxc_decompress_opts_t* opts);

/**
 * @brief Returns the decompressed size stored in a ZXC compressed file.
 *
 * This function reads the file footer to extract the original uncompressed size
 * without performing any decompression. The file position is restored after reading.
 *
 * @param[in] f_in  Input file stream (must be opened in "rb" mode).
 *
 * @return The original uncompressed size in bytes, or a negative zxc_error_t code (e.g.,
 * ZXC_ERROR_BAD_MAGIC) if the file is invalid or an I/O error occurred.
 */
ZXC_EXPORT int64_t zxc_stream_get_decompressed_size(FILE* f_in);

/* ========================================================================= */
/*  Seekable FILE* open helper                                               */
/* ========================================================================= */

/**
 * @brief Opens a seekable archive from a @c FILE*.
 *
 * Internally builds a @ref zxc_reader_t that performs thread-safe positional
 * reads (@c pread on POSIX, @c ReadFile + @c OVERLAPPED on Windows) on the
 * file descriptor extracted from @p f, then delegates to
 * @ref zxc_seekable_open_reader.  The current file position is saved and
 * restored.  The @c FILE* must remain open for the lifetime of the handle.
 *
 * Lives here (next to the other @c FILE*-based entry points) rather than in
 * @c zxc_seekable.h so the latter remains freestanding (kernel-includable).
 *
 * @param[in] f  File opened in @c "rb" mode (must be seekable, not a pipe).
 * @return Handle on success (free with @ref zxc_seekable_free), or @c NULL
 *         on error.
 */
ZXC_EXPORT zxc_seekable* zxc_seekable_open_file(FILE* f);

/** @} */ /* end of stream_api */

#ifdef __cplusplus
}
#endif

#endif  // ZXC_STREAM_H