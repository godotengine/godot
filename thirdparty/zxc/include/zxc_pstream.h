/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file zxc_pstream.h
 * @brief Push-based, single-threaded streaming compression and decompression.
 *
 * This header exposes a non-blocking, caller-driven streaming API, the
 * counterpart of the @c FILE*-based @ref zxc_stream_compress / @ref
 * zxc_stream_decompress.  Where the @c FILE* API takes ownership of the
 * pipeline (it reads until EOF and writes until done), the push API hands the
 * control back to the caller: feed input chunks when available, drain output
 * chunks when ready, finalise on demand.
 *
 * Use this API when you need to integrate ZXC into:
 * - a callback-driven library;
 * - an asynchronous event loop;
 * - a network protocol that streams data without seeking (HTTP chunked
 *   transfer, gRPC, custom binary protocols);
 * - any pipeline where you cannot block on a @c FILE*.
 *
 * The API is single-threaded: one context is processed by one thread at a
 * time.  For multi-threaded compression of a single file end-to-end, use
 * @ref zxc_stream_compress instead.
 *
 * @par Compression usage
 * @code
 * zxc_compress_opts_t opts = { .level = 3, .checksum_enabled = 1 };
 * zxc_cstream* cs = zxc_cstream_create(&opts);
 *
 * uint8_t in_buf[64*1024], out_buf[64*1024];
 * zxc_outbuf_t out = { out_buf, sizeof out_buf, 0 };
 *
 * ssize_t n;
 * while ((n = read_some(in_buf, sizeof in_buf)) > 0) {
 *     zxc_inbuf_t in = { in_buf, (size_t)n, 0 };
 *     while (in.pos < in.size) {
 *         int64_t r = zxc_cstream_compress(cs, &out, &in);
 *         if (r < 0) goto fatal;
 *         if (out.pos > 0) { write_to_sink(out_buf, out.pos); out.pos = 0; }
 *     }
 * }
 *
 * int64_t pending;
 * do {
 *     pending = zxc_cstream_end(cs, &out);
 *     if (pending < 0) goto fatal;
 *     if (out.pos > 0) { write_to_sink(out_buf, out.pos); out.pos = 0; }
 * } while (pending > 0);
 *
 * zxc_cstream_free(cs);
 * @endcode
 *
 * @see zxc_stream.h  for the multi-threaded @c FILE*-based pipeline.
 * @see zxc_buffer.h  for one-shot in-memory compression.
 */

#ifndef ZXC_PSTREAM_H
#define ZXC_PSTREAM_H

#include <stddef.h>
#include <stdint.h>

#include "zxc_export.h"
#include "zxc_opts.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup pstream Push Streaming API
 * @brief Caller-driven, single-threaded streaming compression and decompression.
 * @{
 */

/**
 * @brief Input buffer descriptor for push streaming.
 *
 * The caller fills @c src with bytes to feed in and sets @c size to their
 * count.  The library advances @c pos as it consumes input; the caller must
 * not modify @c pos between calls.
 */
typedef struct {
    const void* src; /**< Caller-owned input bytes. */
    size_t size;     /**< Total bytes available in @c src. */
    size_t pos;      /**< Bytes already consumed by the library (in/out). */
} zxc_inbuf_t;

/**
 * @brief Output buffer descriptor for push streaming.
 *
 * The caller provides a writable region of capacity @c size starting at
 * @c dst.  The library writes starting at @c dst+pos and advances @c pos by
 * the number of bytes produced.  The caller drains @c [dst, dst+pos) and
 * resets @c pos to 0 between rounds (or grows @c size).
 */
typedef struct {
    void* dst;   /**< Caller-owned output region. */
    size_t size; /**< Total capacity available at @c dst. */
    size_t pos;  /**< Bytes already produced by the library (in/out). */
} zxc_outbuf_t;

/* Opaque streaming contexts. */
/** @brief Opaque push-model compression stream (see @ref zxc_cstream_create). */
typedef struct zxc_cstream_s zxc_cstream;
/** @brief Opaque push-model decompression stream (see @ref zxc_dstream_create). */
typedef struct zxc_dstream_s zxc_dstream;

/* ===== Compression =================================================== */

/**
 * @brief Creates a push compression stream.
 *
 * All settings from @p opts are copied into the context.  After this call,
 * the @p opts struct may be freed or reused.
 *
 * Only @c level, @c block_size, and @c checksum_enabled are honoured.
 * @c n_threads is ignored (this API is single-threaded; use
 * @ref zxc_stream_compress for the multi-threaded @c FILE* pipeline).
 *
 * If @p opts is not @c NULL, the honoured fields must contain valid values.
 * Invalid option values (for example an unsupported @c block_size) cause
 * stream creation to fail.
 *
 * @param[in] opts  Compression options, or @c NULL for all defaults.
 * @return Allocated context to be released with @ref zxc_cstream_free,
 *         or @c NULL if stream creation fails due to memory allocation
 *         failure or invalid option values in @p opts.
 */
ZXC_EXPORT zxc_cstream* zxc_cstream_create(const zxc_compress_opts_t* opts);

/**
 * @brief Releases a compression stream and all internal buffers.
 *
 * Safe to call with @c NULL (no-op).
 *
 * @param[in] cs  Stream returned by @ref zxc_cstream_create.
 */
ZXC_EXPORT void zxc_cstream_free(zxc_cstream* cs);

/**
 * @brief Pushes input bytes into the stream and drains compressed output.
 *
 * Reads from @c in->src starting at @c in->pos, writes to @c out->dst
 * starting at @c out->pos, advancing both as data flows.  Each call makes as
 * much progress as either buffer allows in a single visit:
 *
 * - emits the file header on the first invocation (16 B);
 * - copies input into the internal block accumulator;
 * - whenever the accumulator is full, compresses one block and writes it
 *   into @p out (up to @c out->size);
 * - returns when @p in is fully consumed *and* no more compressed bytes are
 *   pending, or when @p out has no room left.
 *
 * The function is fully reentrant: if @p out fills mid-block, the next call
 * resumes draining from where the previous left off.  Safe to call with
 * @c in->size == in->pos (drain-only mode).
 *
 * @par Errors
 * On failure the context becomes errored (sticky): every subsequent call to
 * @ref zxc_cstream_compress / @ref zxc_cstream_end returns the same negative
 * code without doing further work.  Only @ref zxc_cstream_free is safe.
 *
 * @param[in,out] cs   Compression stream.
 * @param[in,out] out  Output descriptor; @c pos is advanced by produced bytes.
 * @param[in,out] in   Input descriptor;  @c pos is advanced by consumed bytes.
 *
 * @return @c 0 if @p in was fully consumed and no compressed bytes remain
 *         pending in the internal staging area;
 *         @c >0 number of bytes still pending, drain @p out and call again
 *         with the same (or new) input;
 *         @c <0 a @ref zxc_error_t code.
 */
ZXC_EXPORT int64_t zxc_cstream_compress(zxc_cstream* cs, zxc_outbuf_t* out, zxc_inbuf_t* in);

/**
 * @brief Finalises the stream: flushes pending data, writes EOF block + footer.
 *
 * Must be called after the last @ref zxc_cstream_compress invocation to
 * produce a valid ZXC file.  Like @ref zxc_cstream_compress, this function
 * is reentrant: if @p out fills before everything is drained, it returns a
 * positive count and the caller drains and calls again.
 *
 * After @ref zxc_cstream_end returns @c 0, the stream is in DONE state and
 * any further call returns @c ZXC_ERROR_NULL_INPUT (use @ref
 * zxc_cstream_free to release).
 *
 * @param[in,out] cs   Compression stream.
 * @param[in,out] out  Output descriptor.
 *
 * @return @c 0 finalisation complete (file is now valid);
 *         @c >0 bytes still pending, drain @p out and call again;
 *         @c <0 a @ref zxc_error_t code.
 */
ZXC_EXPORT int64_t zxc_cstream_end(zxc_cstream* cs, zxc_outbuf_t* out);

/**
 * @brief Suggested input chunk size for best throughput.
 *
 * Equal to the configured block size (default 512 KB).  The caller may
 * supply any input chunk; this is purely a performance hint.
 *
 * @param[in] cs  Compression stream.
 * @return Suggested @c in_buf capacity in bytes, or 0 if @p cs is @c NULL.
 */
ZXC_EXPORT size_t zxc_cstream_in_size(const zxc_cstream* cs);

/**
 * @brief Suggested output chunk size to never trigger a partial drain.
 *
 * Sized to hold one full compressed block plus framing overhead.  Smaller
 * outputs work but may force the caller into an extra drain loop.
 *
 * @param[in] cs  Compression stream.
 * @return Suggested @c out_buf capacity in bytes, or 0 if @p cs is @c NULL.
 */
ZXC_EXPORT size_t zxc_cstream_out_size(const zxc_cstream* cs);

/* ===== Decompression ================================================= */

/**
 * @brief Creates a push decompression stream.
 *
 * All settings from @p opts are copied into the context.  Only
 * @c checksum_enabled is honoured (controls whether per-block and global
 * checksums are verified when present).
 *
 * @param[in] opts  Decompression options, or @c NULL for defaults.
 * @return Allocated context to be released with @ref zxc_dstream_free,
 *         or @c NULL on allocation failure.
 */
ZXC_EXPORT zxc_dstream* zxc_dstream_create(const zxc_decompress_opts_t* opts);

/**
 * @brief Releases a decompression stream.  Safe to call with @c NULL.
 *
 * @param[in] ds  Stream returned by @ref zxc_dstream_create.
 */
ZXC_EXPORT void zxc_dstream_free(zxc_dstream* ds);

/**
 * @brief Pushes compressed input and drains decompressed output.
 *
 * Internally runs a parser state machine: file header -> per-block
 * (header + payload + optional checksum) -> EOF block -> optional SEK block ->
 * file footer.  Each call makes as much progress as @p in and @p out allow.
 *
 * @par End of stream
 * When the decoder reaches the file footer and validates it, the stream
 * enters DONE state.  Subsequent calls return @c 0 without producing more
 * output, even if extra bytes remain in @p in (those trailing bytes are
 * silently ignored, the caller may use the residual @c in->pos to detect
 * how much real data was consumed).
 *
 * @par Errors
 * Sticky: once a negative code is returned, further calls keep returning it.
 *
 * @param[in,out] ds   Decompression stream.
 * @param[in,out] out  Output descriptor; @c pos advanced by produced bytes.
 * @param[in,out] in   Input descriptor;  @c pos advanced by consumed bytes.
 *
 * @return @c >0 number of decompressed bytes written into @p out this call;
 *         @c 0 stream complete (DONE) or no progress possible (caller should
 *         feed more input);
 *         @c <0 a @ref zxc_error_t code.
 */
ZXC_EXPORT int64_t zxc_dstream_decompress(zxc_dstream* ds, zxc_outbuf_t* out, zxc_inbuf_t* in);

/**
 * @brief Reports whether the decoder has fully consumed a valid stream.
 *
 * Returns @c 1 iff the parser has reached the file footer **and** validated
 * it.  Callers that have finished feeding input use this to detect truncated
 * streams: if @ref zxc_dstream_decompress returns @c 0 with no output and
 * @ref zxc_dstream_finished returns @c 0, the input ended prematurely.
 *
 * @param[in] ds  Decompression stream.
 * @return @c 1 if DONE, @c 0 otherwise (including errored).
 */
ZXC_EXPORT int zxc_dstream_finished(const zxc_dstream* ds);

/**
 * @brief Suggested input chunk size for the decompressor.
 *
 * @param[in] ds  Decompression stream.
 * @return Suggested @c in_buf capacity in bytes, or 0 if @p ds is @c NULL.
 */
ZXC_EXPORT size_t zxc_dstream_in_size(const zxc_dstream* ds);

/**
 * @brief Suggested output chunk size for the decompressor.
 *
 * Sized to hold at least one full decompressed block.
 *
 * @param[in] ds  Decompression stream.
 * @return Suggested @c out_buf capacity in bytes, or 0 if @p ds is @c NULL.
 */
ZXC_EXPORT size_t zxc_dstream_out_size(const zxc_dstream* ds);

/** @} */ /* end of pstream */

#ifdef __cplusplus
}
#endif

#endif /* ZXC_PSTREAM_H */
