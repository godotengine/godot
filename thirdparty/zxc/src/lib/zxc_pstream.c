/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file zxc_pstream.c
 * @brief Push-based, single-threaded streaming driver implementation.
 *
 * See zxc_pstream.h for the public contract.  The implementation composes
 * the public block API (@ref zxc_compress_block / @ref zxc_decompress_block)
 * with the public sans-IO header helpers (@ref zxc_write_file_header /
 * footer, @c zxc_read_*); the only internal dependency is on shared
 * constants and the global-hash combine inline, pulled from zxc_internal.h.
 *
 * Both compression and decompression are structured as resumable state
 * machines driven by caller-provided input/output buffers
 * (@ref zxc_inbuf_t / @ref zxc_outbuf_t).  Each call advances as much as
 * possible without blocking and returns a status indicating whether the
 * caller should drain @p out, supply more @p in, or finalise the stream.
 */

#include "../../include/zxc_pstream.h"

#include "../../include/zxc_buffer.h"
#include "../../include/zxc_constants.h"
#include "../../include/zxc_error.h"
#include "zxc_internal.h"

/* ===================================================================== */
/*  Compression                                                          */
/* ===================================================================== */

/**
 * @enum cstream_state_t
 * @brief Lifecycle states of the push compression stream.
 *
 * The compression state machine alternates between *staging* a chunk of
 * output bytes into the @c pending buffer and *draining* those bytes into
 * the caller's @ref zxc_outbuf_t.  Forward progress is therefore always
 * either consuming from @c in or producing into @c out.
 *
 * @var cstream_state_t::CS_INIT
 *      Initial state; nothing has been emitted yet.
 * @var cstream_state_t::CS_DRAIN_HEADER
 *      File header staged in @c pending; draining to @p out, then transitions
 *      to @c CS_ACCUMULATE.
 * @var cstream_state_t::CS_ACCUMULATE
 *      Copying input bytes into the internal block accumulator until it is
 *      full (or the stream is finalised).
 * @var cstream_state_t::CS_DRAIN_BLOCK
 *      A full data block was just compressed; draining it to @p out, then
 *      back to @c CS_ACCUMULATE.
 * @var cstream_state_t::CS_DRAIN_LAST
 *      Draining the final partial block produced inside @ref zxc_cstream_end;
 *      transitions to @c CS_DRAIN_EOF.
 * @var cstream_state_t::CS_DRAIN_EOF
 *      Draining the EOF block; transitions to @c CS_DRAIN_FOOTER.
 * @var cstream_state_t::CS_DRAIN_FOOTER
 *      Draining the file footer; transitions to @c CS_DONE.
 * @var cstream_state_t::CS_DONE
 *      Finalisation complete; further @c _compress / @c _end calls are
 *      rejected.
 * @var cstream_state_t::CS_ERRORED
 *      Sticky error state; subsequent calls return the latched error code.
 */
typedef enum {
    CS_INIT = 0,
    CS_DRAIN_HEADER,
    CS_ACCUMULATE,
    CS_DRAIN_BLOCK,
    CS_DRAIN_LAST,
    CS_DRAIN_EOF,
    CS_DRAIN_FOOTER,
    CS_DONE,
    CS_ERRORED
} cstream_state_t;

/**
 * @struct zxc_cstream_s
 * @brief Internal state of a push compression stream.
 *
 * Owns three buffers: a fixed-size input accumulator (@c in_block, sized
 * to one block), a variable-size output staging area (@c pending, holding
 * the file header, one compressed block, the EOF block, or the file
 * footer), and the underlying compression context (@c cctx).
 *
 * @var zxc_cstream_s::opts
 *      Compression options (copied from the caller at creation time).
 * @var zxc_cstream_s::cctx
 *      Underlying single-block compression context.
 * @var zxc_cstream_s::block_size
 *      Target block size in bytes; cached from @c opts.block_size.
 * @var zxc_cstream_s::in_block
 *      Heap buffer of capacity @c block_size used to accumulate one full
 *      uncompressed block before invoking the block compressor.
 * @var zxc_cstream_s::in_used
 *      Number of valid bytes currently held in @c in_block (in
 *      [0, @c block_size]).
 * @var zxc_cstream_s::pending
 *      Heap buffer holding the next chunk of output bytes to emit
 *      (header, compressed block, EOF marker or footer).
 * @var zxc_cstream_s::pending_cap
 *      Allocated capacity of @c pending.
 * @var zxc_cstream_s::pending_len
 *      Total valid bytes currently staged in @c pending.
 * @var zxc_cstream_s::pending_pos
 *      Bytes already copied from @c pending to the caller's output buffer.
 *      Drain is complete when @c pending_pos == @c pending_len.
 * @var zxc_cstream_s::total_in
 *      Running count of uncompressed bytes consumed; written into the file
 *      footer.
 * @var zxc_cstream_s::global_hash
 *      Rolling per-block-trailer hash; written into the file footer when
 *      checksums are enabled.
 * @var zxc_cstream_s::state
 *      Current state machine position (see @ref cstream_state_t).
 * @var zxc_cstream_s::error_code
 *      Sticky error code; valid only when @c state is @c CS_ERRORED.
 */
struct zxc_cstream_s {
    zxc_compress_opts_t opts;
    zxc_cctx* cctx;
    size_t block_size;

    uint8_t* in_block;
    size_t in_used;

    uint8_t* pending;
    size_t pending_cap;
    size_t pending_len;
    size_t pending_pos;

    uint64_t total_in;
    uint32_t global_hash;

    cstream_state_t state;
    int error_code;
};

/**
 * @brief Latches a sticky error on the compression stream.
 *
 * Stores @p code in @c cs->error_code, transitions @c cs->state to
 * @c CS_ERRORED, and returns @p code.  Once errored, subsequent
 * @ref zxc_cstream_compress / @ref zxc_cstream_end calls return the same
 * code without performing further work.
 *
 * @param[in,out] cs   Compression stream.
 * @param[in]     code Negative @ref zxc_error_t value to latch.
 * @return @p code (always negative).
 */
static int cs_set_error(zxc_cstream* cs, const int code) {
    cs->error_code = code;
    cs->state = CS_ERRORED;
    return code;
}

/**
 * @brief Compresses one full or partial accumulated block.
 *
 * Compresses the contents of @c cs->in_block into @c cs->pending, growing
 * the latter to @ref zxc_compress_block_bound if needed, and updates
 * bookkeeping (@c total_in, @c global_hash, @c in_used reset to 0).  When
 * file-level checksums are enabled, folds the block trailer into the
 * rolling @c global_hash.
 *
 * @pre @c cs->in_used > 0.
 *
 * @param[in,out] cs Compression stream.
 * @return @ref ZXC_OK on success, negative @ref zxc_error_t on failure.
 */
static int cs_compress_one_block(zxc_cstream* cs) {
    const uint64_t bound = zxc_compress_block_bound(cs->in_used);
    // LCOV_EXCL_START
    if (UNLIKELY(bound == 0 || bound > SIZE_MAX)) return ZXC_ERROR_OVERFLOW;
    if (UNLIKELY(bound > cs->pending_cap)) {
        uint8_t* nb = (uint8_t*)ZXC_REALLOC(cs->pending, (size_t)bound);
        if (UNLIKELY(!nb)) return ZXC_ERROR_MEMORY;
        cs->pending = nb;
        cs->pending_cap = (size_t)bound;
    }
    // LCOV_EXCL_STOP
    const int64_t csize = zxc_compress_block(cs->cctx, cs->in_block, cs->in_used, cs->pending,
                                             cs->pending_cap, &cs->opts);
    if (UNLIKELY(csize < 0)) return (int)csize;  // LCOV_EXCL_LINE

    cs->pending_len = (size_t)csize;
    cs->pending_pos = 0;
    cs->total_in += cs->in_used;
    cs->in_used = 0;

    /* If checksums are on, the block trailer is the last ZXC_BLOCK_CHECKSUM_SIZE
     * bytes of pending; fold it into the rolling global hash. */
    if (cs->opts.checksum_enabled && cs->pending_len >= ZXC_BLOCK_CHECKSUM_SIZE) {
        const uint32_t bh = zxc_le32(cs->pending + cs->pending_len - ZXC_BLOCK_CHECKSUM_SIZE);
        cs->global_hash = zxc_hash_combine_rotate(cs->global_hash, bh);
    }
    return ZXC_OK;
}

/**
 * @brief Drains staged output bytes into the caller's output buffer.
 *
 * Copies as many bytes as possible from
 * @c cs->pending[pending_pos..pending_len) into
 * @c out->dst[pos..size), advancing both cursors.
 *
 * @param[in,out] cs  Compression stream.
 * @param[in,out] out Caller output buffer.
 * @return Non-zero once the @c pending buffer has been fully drained
 *         (@c pending_pos == @c pending_len), zero otherwise.
 */
static int cs_drain_pending(zxc_cstream* cs, zxc_outbuf_t* out) {
    const size_t avail_out = out->size - out->pos;
    const size_t avail_pen = cs->pending_len - cs->pending_pos;
    const size_t n = avail_out < avail_pen ? avail_out : avail_pen;
    if (n) {
        ZXC_MEMCPY((uint8_t*)out->dst + out->pos, cs->pending + cs->pending_pos, n);
        out->pos += n;
        cs->pending_pos += n;
    }
    return cs->pending_pos == cs->pending_len;
}

/**
 * @brief Allocates and initialises a push compression stream.
 *
 * Copies @p opts into the new context, applying defaults for any zero-valued
 * field (@c level -> @ref ZXC_LEVEL_DEFAULT, @c block_size ->
 * @ref ZXC_BLOCK_SIZE_DEFAULT).  Forces single-threaded operation
 * (@c n_threads = 0), disables progress callbacks and seekable framing
 * (those modes belong to the @c FILE*-based pipeline).  Pre-sizes the
 * @c pending buffer so that the file header / footer paths never need a
 * realloc.
 *
 * @param[in] opts Compression options, or @c NULL for full defaults.
 * @return New stream owned by the caller, or @c NULL on allocation
 *         failure / invalid option values.
 */
zxc_cstream* zxc_cstream_create(const zxc_compress_opts_t* opts) {
    zxc_cstream* cs = (zxc_cstream*)ZXC_CALLOC(1, sizeof(*cs));
    if (UNLIKELY(!cs)) return NULL;  // LCOV_EXCL_LINE

    if (opts) cs->opts = *opts;
    if (cs->opts.level == 0) cs->opts.level = ZXC_LEVEL_DEFAULT;
    if (cs->opts.block_size == 0) cs->opts.block_size = ZXC_BLOCK_SIZE_DEFAULT;
    /* n_threads is ignored on this single-threaded path. */
    cs->opts.n_threads = 0;
    cs->opts.progress_cb = NULL;
    cs->opts.user_data = NULL;
    cs->opts.seekable = 0;
    cs->block_size = cs->opts.block_size;

    cs->cctx = zxc_create_cctx(&cs->opts);
    // LCOV_EXCL_START
    if (UNLIKELY(!cs->cctx)) {
        ZXC_FREE(cs);
        return NULL;
    }
    cs->in_block = (uint8_t*)ZXC_MALLOC(cs->block_size);
    if (UNLIKELY(!cs->in_block)) {
        zxc_free_cctx(cs->cctx);
        ZXC_FREE(cs);
        return NULL;
    }
    // LCOV_EXCL_STOP
    /* Pre-size pending so the file header path never needs realloc. */
    cs->pending_cap =
        ZXC_FILE_HEADER_SIZE > ZXC_FILE_FOOTER_SIZE ? ZXC_FILE_HEADER_SIZE : ZXC_FILE_FOOTER_SIZE;
    cs->pending = (uint8_t*)ZXC_MALLOC(cs->pending_cap);
    // LCOV_EXCL_START
    if (UNLIKELY(!cs->pending)) {
        ZXC_FREE(cs->in_block);
        zxc_free_cctx(cs->cctx);
        ZXC_FREE(cs);
        return NULL;
    }
    // LCOV_EXCL_STOP
    cs->state = CS_INIT;
    return cs;
}

/**
 * @brief Stages the 16-byte file header into the @c pending buffer.
 *
 * @param[in,out] cs Compression stream.
 * @return @ref ZXC_OK on success, negative @ref zxc_error_t on failure.
 */
static int cs_stage_file_header(zxc_cstream* cs) {
    const int w = zxc_write_file_header(cs->pending, cs->pending_cap, cs->block_size,
                                        cs->opts.checksum_enabled, 0);
    if (UNLIKELY(w < 0)) return w;  // LCOV_EXCL_LINE
    cs->pending_len = (size_t)w;
    cs->pending_pos = 0;
    return ZXC_OK;
}

/**
 * @brief Stages the 8-byte EOF block into the @c pending buffer.
 *
 * The EOF block is a regular block header with @c block_type set to
 * @ref ZXC_BLOCK_EOF and @c comp_size = 0; it carries no payload.
 *
 * @param[in,out] cs Compression stream.
 * @return @ref ZXC_OK on success, negative @ref zxc_error_t on failure.
 */
static int cs_stage_eof(zxc_cstream* cs) {
    // LCOV_EXCL_START
    if (UNLIKELY(ZXC_BLOCK_HEADER_SIZE > cs->pending_cap)) {
        uint8_t* nb = (uint8_t*)ZXC_REALLOC(cs->pending, ZXC_BLOCK_HEADER_SIZE);
        if (UNLIKELY(!nb)) return ZXC_ERROR_MEMORY;
        cs->pending = nb;
        cs->pending_cap = ZXC_BLOCK_HEADER_SIZE;
    }
    // LCOV_EXCL_STOP
    const zxc_block_header_t eof = {
        .block_type = (uint8_t)ZXC_BLOCK_EOF,
        .block_flags = 0,
        .reserved = 0,
        .header_crc = 0,
        .comp_size = 0,
    };
    const int w = zxc_write_block_header(cs->pending, cs->pending_cap, &eof);
    if (UNLIKELY(w < 0)) return w;  // LCOV_EXCL_LINE
    cs->pending_len = (size_t)w;
    cs->pending_pos = 0;
    return ZXC_OK;
}

/**
 * @brief Stages the 12-byte file footer into the @c pending buffer.
 *
 * The footer carries the total uncompressed input size and (when checksums
 * are enabled) the global rolling hash accumulated across all data blocks.
 *
 * @param[in,out] cs Compression stream.
 * @return @ref ZXC_OK on success, negative @ref zxc_error_t on failure.
 */
static int cs_stage_footer(zxc_cstream* cs) {
    // LCOV_EXCL_START
    if (UNLIKELY(ZXC_FILE_FOOTER_SIZE > cs->pending_cap)) {
        uint8_t* nb = (uint8_t*)ZXC_REALLOC(cs->pending, ZXC_FILE_FOOTER_SIZE);
        if (UNLIKELY(!nb)) return ZXC_ERROR_MEMORY;
        cs->pending = nb;
        cs->pending_cap = ZXC_FILE_FOOTER_SIZE;
    }
    // LCOV_EXCL_STOP
    const int w = zxc_write_file_footer(cs->pending, cs->pending_cap, cs->total_in, cs->global_hash,
                                        cs->opts.checksum_enabled);
    if (UNLIKELY(w < 0)) return w;  // LCOV_EXCL_LINE
    cs->pending_len = (size_t)w;
    cs->pending_pos = 0;
    return ZXC_OK;
}

/**
 * @brief Releases a compression stream and all internal buffers.
 *
 * Safe to call with @c NULL.
 *
 * @param[in,out] cs Stream returned by @ref zxc_cstream_create.
 */
void zxc_cstream_free(zxc_cstream* cs) {
    if (!cs) return;
    ZXC_FREE(cs->pending);
    ZXC_FREE(cs->in_block);
    zxc_free_cctx(cs->cctx);
    ZXC_FREE(cs);
}

/**
 * @brief Returns the suggested input chunk size (configured block size).
 *
 * @param[in] cs Compression stream.
 * @return Block size in bytes, or 0 if @p cs is @c NULL.
 */
size_t zxc_cstream_in_size(const zxc_cstream* cs) { return cs ? cs->block_size : 0; }

/**
 * @brief Returns the suggested output chunk size.
 *
 * Sized to hold one full compressed block plus framing overhead, i.e.
 * @ref zxc_compress_block_bound applied to the configured block size.
 * Falls back to @c block_size when the bound overflows @c size_t.
 *
 * @param[in] cs Compression stream.
 * @return Suggested output buffer capacity in bytes, or 0 if @p cs is @c NULL.
 */
size_t zxc_cstream_out_size(const zxc_cstream* cs) {
    if (!cs) return 0;
    const uint64_t b = zxc_compress_block_bound(cs->block_size);
    return (b == 0 || b > SIZE_MAX) ? cs->block_size : (size_t)b;
}

/**
 * @brief Push-side entry point: feeds input and drains compressed output.
 *
 * Drives the @ref cstream_state_t machine: emits the file header on the
 * first call, then accumulates input until a block is full, compresses it,
 * and drains the result into @p out.  Each call makes as much progress as
 * either buffer allows; the function is fully reentrant.  See the public
 * contract in @ref zxc_cstream_compress for full semantics.
 *
 * The terminal states (@c CS_DRAIN_LAST, @c CS_DRAIN_EOF, @c CS_DRAIN_FOOTER,
 * @c CS_DONE, @c CS_ERRORED) are owned by @ref zxc_cstream_end; reaching
 * them here yields @ref ZXC_ERROR_NULL_INPUT.
 *
 * @param[in,out] cs  Compression stream.
 * @param[in,out] out Caller output buffer.
 * @param[in,out] in  Caller input buffer.
 * @return @c 0 if @p in fully consumed and nothing pending,
 *         @c >0 number of bytes still pending (drain @p out then call again),
 *         @c <0 a @ref zxc_error_t code.
 */
int64_t zxc_cstream_compress(zxc_cstream* cs, zxc_outbuf_t* out, zxc_inbuf_t* in) {
    if (UNLIKELY(!cs || !out || !in || in->pos > in->size || out->pos > out->size ||
                 (in->size > in->pos && !in->src) || (out->size > out->pos && !out->dst) ||
                 cs->state == CS_DONE)) {
        return ZXC_ERROR_NULL_INPUT;
    }
    if (UNLIKELY(cs->state == CS_ERRORED)) return cs->error_code;

    for (;;) {
        switch (cs->state) {
            case CS_INIT: {
                const int rc = cs_stage_file_header(cs);
                if (UNLIKELY(rc < 0)) return cs_set_error(cs, rc);  // LCOV_EXCL_LINE
                cs->state = CS_DRAIN_HEADER;
                break;
            }

            case CS_DRAIN_HEADER:
            case CS_DRAIN_BLOCK: {
                if (!cs_drain_pending(cs, out)) return (int64_t)(cs->pending_len - cs->pending_pos);
                cs->state = CS_ACCUMULATE;
                break;
            }

            case CS_ACCUMULATE: {
                const size_t avail_in = in->size - in->pos;
                const size_t room = cs->block_size - cs->in_used;
                const size_t n = avail_in < room ? avail_in : room;
                if (n) {
                    ZXC_MEMCPY(cs->in_block + cs->in_used, (const uint8_t*)in->src + in->pos, n);
                    in->pos += n;
                    cs->in_used += n;
                }

                if (cs->in_used == cs->block_size) {
                    const int rc = cs_compress_one_block(cs);
                    if (UNLIKELY(rc < 0)) return cs_set_error(cs, rc);  // LCOV_EXCL_LINE
                    cs->state = CS_DRAIN_BLOCK;
                    break;
                }
                /* Block not yet full either in is empty or we made no progress. */
                return 0;
            }

            case CS_DRAIN_LAST:
            case CS_DRAIN_EOF:
            case CS_DRAIN_FOOTER:
            case CS_DONE:
            case CS_ERRORED:
                /* These states are owned by _end(). */
                return ZXC_ERROR_NULL_INPUT;
        }
    }
}

/**
 * @brief Finalises the stream: residual block (if any), EOF, and footer.
 *
 * Continues the same state machine as @ref zxc_cstream_compress through the
 * terminal states (@c CS_DRAIN_LAST -> @c CS_DRAIN_EOF -> @c CS_DRAIN_FOOTER
 * -> @c CS_DONE).  Reentrant: when @p out fills mid-drain, returns the
 * number of bytes still pending and resumes from where it left off on the
 * next call.  See the public contract in @ref zxc_cstream_end.
 *
 * @param[in,out] cs  Compression stream.
 * @param[in,out] out Caller output buffer.
 * @return @c 0 once finalisation is complete (stream is now in DONE state),
 *         @c >0 number of bytes still pending (drain and call again),
 *         @c <0 a @ref zxc_error_t code.
 */
int64_t zxc_cstream_end(zxc_cstream* cs, zxc_outbuf_t* out) {
    if (UNLIKELY(!cs || !out || cs->state == CS_DONE)) return ZXC_ERROR_NULL_INPUT;
    if (UNLIKELY(cs->state == CS_ERRORED)) return cs->error_code;

    for (;;) {
        switch (cs->state) {
            case CS_INIT: {
                /* _end before any input, still need to emit file header. */
                const int rc = cs_stage_file_header(cs);
                if (UNLIKELY(rc < 0)) return cs_set_error(cs, rc);  // LCOV_EXCL_LINE
                cs->state = CS_DRAIN_HEADER;
                break;
            }

            case CS_DRAIN_HEADER: {
                if (!cs_drain_pending(cs, out)) return (int64_t)(cs->pending_len - cs->pending_pos);
                cs->state = CS_ACCUMULATE;
                break;
            }

            case CS_DRAIN_BLOCK: {
                /* This drain came from a full block compressed during _compress. */
                if (!cs_drain_pending(cs, out)) return (int64_t)(cs->pending_len - cs->pending_pos);
                cs->state = CS_ACCUMULATE;
                break;
            }

            case CS_ACCUMULATE: {
                /* Compress the residual partial block (if any), then EOF + footer. */
                if (cs->in_used > 0) {
                    const int rc = cs_compress_one_block(cs);
                    if (UNLIKELY(rc < 0)) return cs_set_error(cs, rc);  // LCOV_EXCL_LINE
                    cs->state = CS_DRAIN_LAST;
                    break;
                }
                /* No residual data: go straight to EOF. */
                {
                    const int rc = cs_stage_eof(cs);
                    if (UNLIKELY(rc < 0)) return cs_set_error(cs, rc);  // LCOV_EXCL_LINE
                    cs->state = CS_DRAIN_EOF;
                    break;
                }
            }

            case CS_DRAIN_LAST: {
                if (!cs_drain_pending(cs, out)) return (int64_t)(cs->pending_len - cs->pending_pos);
                /* After last data block -> EOF. */
                const int rc = cs_stage_eof(cs);
                if (UNLIKELY(rc < 0)) return cs_set_error(cs, rc);  // LCOV_EXCL_LINE
                cs->state = CS_DRAIN_EOF;
                break;
            }

            case CS_DRAIN_EOF: {
                if (!cs_drain_pending(cs, out)) return (int64_t)(cs->pending_len - cs->pending_pos);
                const int rc = cs_stage_footer(cs);
                if (UNLIKELY(rc < 0)) return cs_set_error(cs, rc);  // LCOV_EXCL_LINE
                cs->state = CS_DRAIN_FOOTER;
                break;
            }

            case CS_DRAIN_FOOTER: {
                if (!cs_drain_pending(cs, out)) return (int64_t)(cs->pending_len - cs->pending_pos);
                cs->state = CS_DONE;
                return 0;
            }

            case CS_DONE:
            case CS_ERRORED:
                return cs->state == CS_ERRORED ? cs->error_code : 0;
        }
    }
}

/* ===================================================================== */
/*  Decompression                                                        */
/* ===================================================================== */

/**
 * @enum dstream_state_t
 * @brief Lifecycle states of the push decompression stream.
 *
 * The decompressor implements a frame-aware parser: file header -> N
 * (data block header + payload [+ optional checksum]) -> EOF block ->
 * optional SEK index block -> file footer.  The states alternate between
 * *pulling* fixed- or variable-sized chunks from the caller's input, and
 * *emitting* the corresponding decoded output.
 *
 * @var dstream_state_t::DS_NEED_FILE_HEADER
 *      Pulling the 16-byte file header into @c scratch.
 * @var dstream_state_t::DS_NEED_BLOCK_HEADER
 *      Pulling an 8-byte block header into @c scratch.
 * @var dstream_state_t::DS_NEED_BLOCK_PAYLOAD
 *      Pulling a data block payload (and optional checksum) into @c payload.
 * @var dstream_state_t::DS_DECODE_BLOCK
 *      Calling the underlying block decoder on the accumulated payload.
 * @var dstream_state_t::DS_EMIT_DECODED
 *      Draining decoded bytes from @c decoded into @p out.
 * @var dstream_state_t::DS_PEEK_TAIL
 *      Just past the EOF block: read 8 bytes and peek to disambiguate
 *      between an optional SEK index block and the file footer.
 * @var dstream_state_t::DS_DRAIN_SEK_PAYLOAD
 *      The peeked 8 bytes were a SEK header; skipping its payload bytes.
 * @var dstream_state_t::DS_NEED_FOOTER_FULL
 *      Pulling the full 12-byte file footer (used after a SEK block).
 * @var dstream_state_t::DS_NEED_FOOTER_REST
 *      The 8 peeked bytes were the head of the footer; pulling the
 *      remaining 4 bytes.
 * @var dstream_state_t::DS_VALIDATE_FOOTER
 *      Validating @c total_out and the optional global hash.
 * @var dstream_state_t::DS_DONE
 *      Stream fully consumed and validated.
 * @var dstream_state_t::DS_ERRORED
 *      Sticky error state; subsequent calls return the latched code.
 */
typedef enum {
    DS_NEED_FILE_HEADER = 0,
    DS_NEED_BLOCK_HEADER,
    DS_NEED_BLOCK_PAYLOAD,
    DS_DECODE_BLOCK,
    DS_EMIT_DECODED,
    DS_PEEK_TAIL,
    DS_DRAIN_SEK_PAYLOAD,
    DS_NEED_FOOTER_FULL,
    DS_NEED_FOOTER_REST,
    DS_VALIDATE_FOOTER,
    DS_DONE,
    DS_ERRORED
} dstream_state_t;

/**
 * @struct zxc_dstream_s
 * @brief Internal state of a push decompression stream.
 *
 * Owns three accumulator regions: a fixed-size on-stack-style @c scratch
 * buffer for headers/footers/peeks, a heap @c payload buffer for variable
 * block payloads, and a heap @c decoded buffer that holds one decompressed
 * block before it is emitted to the caller.
 *
 * @var zxc_dstream_s::opts
 *      Decompression options (copied at creation time).
 * @var zxc_dstream_s::inner
 *      Underlying single-block decompression context.
 * @var zxc_dstream_s::inner_initialized
 *      Non-zero once @c inner has been initialised; gates the matching
 *      @ref zxc_cctx_free call at teardown.
 * @var zxc_dstream_s::block_size
 *      Block size declared by the file header; 0 until the header is parsed.
 * @var zxc_dstream_s::file_has_checksum
 *      File-level checksum flag declared by the file header.
 * @var zxc_dstream_s::scratch
 *      Generic 32-byte accumulator for fixed-size frames (file header, block
 *      header, footer); comfortably holds the largest (16-byte file header).
 * @var zxc_dstream_s::scratch_used
 *      Number of bytes currently held in @c scratch.
 * @var zxc_dstream_s::scratch_need
 *      Target number of bytes for the current accumulation phase.
 * @var zxc_dstream_s::payload
 *      Heap buffer holding one block: header + compressed payload + optional
 *      checksum.
 * @var zxc_dstream_s::payload_cap
 *      Allocated capacity of @c payload.
 * @var zxc_dstream_s::payload_used
 *      Number of valid bytes currently in @c payload.
 * @var zxc_dstream_s::payload_need
 *      Target byte count for the current payload phase
 *      (= header size + comp_size + checksum size).
 * @var zxc_dstream_s::decoded
 *      Heap buffer holding the decoded output of one block (sized for the
 *      wild-copy fast path: @c block_size + @ref ZXC_DECOMPRESS_TAIL_PAD).
 * @var zxc_dstream_s::decoded_cap
 *      Allocated capacity of @c decoded.
 * @var zxc_dstream_s::decoded_size
 *      Real number of decoded bytes in @c decoded after the last decode.
 * @var zxc_dstream_s::decoded_pos
 *      Bytes already emitted from @c decoded; drain complete when
 *      @c decoded_pos == @c decoded_size.
 * @var zxc_dstream_s::cur_bh
 *      Parsed block header for the block currently being processed.
 * @var zxc_dstream_s::sek_remaining
 *      Bytes left to skip from a SEK block payload (only used in
 *      @c DS_DRAIN_SEK_PAYLOAD).
 * @var zxc_dstream_s::total_out
 *      Cumulative decompressed output size; cross-checked against the
 *      file footer.
 * @var zxc_dstream_s::global_hash
 *      Rolling per-block-trailer hash; cross-checked against the file
 *      footer when checksums are enabled.
 * @var zxc_dstream_s::state
 *      Current state machine position (see @ref dstream_state_t).
 * @var zxc_dstream_s::error_code
 *      Sticky error code; valid only when @c state is @c DS_ERRORED.
 */
struct zxc_dstream_s {
    zxc_decompress_opts_t opts;
    zxc_cctx_t inner;
    int inner_initialized;
    size_t block_size;
    int file_has_checksum;

    uint8_t scratch[32];
    size_t scratch_used;
    size_t scratch_need;

    uint8_t* payload;
    size_t payload_cap;
    size_t payload_used;
    size_t payload_need;

    uint8_t* decoded;
    size_t decoded_cap;
    size_t decoded_size;
    size_t decoded_pos;

    zxc_block_header_t cur_bh;
    size_t sek_remaining;

    uint64_t total_out;
    uint32_t global_hash;

    dstream_state_t state;
    int error_code;
};

/**
 * @brief Latches a sticky error on the decompression stream.
 *
 * Stores @p code in @c ds->error_code, transitions @c ds->state to
 * @c DS_ERRORED, and returns @p code.  Once errored, subsequent
 * @ref zxc_dstream_decompress calls return the same code without
 * performing further work.
 *
 * @param[in,out] ds   Decompression stream.
 * @param[in]     code Negative @ref zxc_error_t value to latch.
 * @return @p code (always negative).
 */
static int ds_set_error(zxc_dstream* ds, const int code) {
    ds->error_code = code;
    ds->state = DS_ERRORED;
    return code;
}

/**
 * @brief Pulls up to @c (scratch_need - scratch_used) bytes from @p in.
 *
 * Used to accumulate fixed-size frames (file header, block header, footer,
 * EOF tail peek) into the inline @c scratch buffer.
 *
 * @param[in,out] ds Decompression stream.
 * @param[in,out] in Caller input buffer; @c pos is advanced.
 * @return @c 1 once @c scratch holds exactly @c scratch_need bytes,
 *         @c 0 otherwise (need more input).
 */
static int ds_pull_scratch(zxc_dstream* ds, zxc_inbuf_t* in) {
    const size_t want = ds->scratch_need - ds->scratch_used;
    const size_t avail = in->size - in->pos;
    const size_t n = want < avail ? want : avail;
    if (n) {
        ZXC_MEMCPY(ds->scratch + ds->scratch_used, (const uint8_t*)in->src + in->pos, n);
        in->pos += n;
        ds->scratch_used += n;
    }
    return ds->scratch_used == ds->scratch_need;
}

/**
 * @brief Same as @ref ds_pull_scratch but pulls into the heap @c payload buffer.
 *
 * Used to accumulate the variable-size compressed block payload
 * (header + comp_size [+ checksum]).
 *
 * @param[in,out] ds Decompression stream.
 * @param[in,out] in Caller input buffer; @c pos is advanced.
 * @return @c 1 once @c payload holds exactly @c payload_need bytes,
 *         @c 0 otherwise (need more input).
 */
static int ds_pull_payload(zxc_dstream* ds, zxc_inbuf_t* in) {
    const size_t want = ds->payload_need - ds->payload_used;
    const size_t avail = in->size - in->pos;
    const size_t n = want < avail ? want : avail;
    if (n) {
        ZXC_MEMCPY(ds->payload + ds->payload_used, (const uint8_t*)in->src + in->pos, n);
        in->pos += n;
        ds->payload_used += n;
    }
    return ds->payload_used == ds->payload_need;
}

/**
 * @brief Allocates and initialises a push decompression stream.
 *
 * Copies @p opts into the new context.  The internal multi-threading,
 * progress-callback, and seekable knobs are forced off (those modes belong
 * to the @c FILE*-based pipeline).  @c block_size, @c file_has_checksum,
 * and the @c payload / @c decoded buffers are filled in lazily once the
 * file header is parsed.
 *
 * @param[in] opts Decompression options, or @c NULL for full defaults.
 * @return New stream owned by the caller, or @c NULL on allocation failure.
 */
zxc_dstream* zxc_dstream_create(const zxc_decompress_opts_t* opts) {
    zxc_dstream* ds = (zxc_dstream*)ZXC_CALLOC(1, sizeof(*ds));
    if (UNLIKELY(!ds)) return NULL;  // LCOV_EXCL_LINE
    if (opts) ds->opts = *opts;
    ds->opts.n_threads = 0;
    ds->opts.progress_cb = NULL;
    ds->opts.user_data = NULL;
    ds->state = DS_NEED_FILE_HEADER;
    ds->scratch_need = ZXC_FILE_HEADER_SIZE;
    return ds;
}

/**
 * @brief Releases a decompression stream and all internal buffers.
 *
 * Safe to call with @c NULL.
 *
 * @param[in,out] ds Stream returned by @ref zxc_dstream_create.
 */
void zxc_dstream_free(zxc_dstream* ds) {
    if (!ds) return;
    ZXC_FREE(ds->payload);
    ZXC_FREE(ds->decoded);
    if (ds->inner_initialized) zxc_cctx_free(&ds->inner);
    ZXC_FREE(ds);
}

/**
 * @brief Returns 1 iff the stream has reached @c DS_DONE.
 *
 * @param[in] ds Decompression stream.
 * @return @c 1 if DONE, @c 0 otherwise (including errored).
 */
int zxc_dstream_finished(const zxc_dstream* ds) { return (ds && ds->state == DS_DONE) ? 1 : 0; }

/**
 * @brief Returns the suggested input chunk size for the decompressor.
 *
 * Before the file header is parsed the call returns
 * @ref ZXC_BLOCK_SIZE_DEFAULT; afterwards it returns the maximal compressed
 * block size derived from the negotiated @c block_size.
 *
 * @param[in] ds Decompression stream.
 * @return Suggested input buffer capacity in bytes, or 0 if @p ds is @c NULL.
 */
size_t zxc_dstream_in_size(const zxc_dstream* ds) {
    if (!ds) return 0;
    if (ds->block_size == 0) return ZXC_BLOCK_SIZE_DEFAULT;
    const uint64_t b = zxc_compress_block_bound(ds->block_size);
    return (b == 0 || b > SIZE_MAX) ? ds->block_size : (size_t)b;
}

/**
 * @brief Returns the suggested output chunk size for the decompressor.
 *
 * Equals the negotiated @c block_size; before the file header is parsed,
 * returns @ref ZXC_BLOCK_SIZE_DEFAULT.
 *
 * @param[in] ds Decompression stream.
 * @return Suggested output buffer capacity in bytes, or 0 if @p ds is @c NULL.
 */
size_t zxc_dstream_out_size(const zxc_dstream* ds) {
    if (!ds) return 0;
    return ds->block_size == 0 ? ZXC_BLOCK_SIZE_DEFAULT : ds->block_size;
}

/**
 * @brief Drains @c ds->decoded[decoded_pos..decoded_size) into @p out.
 *
 * Updates @c ds->total_out by the number of bytes copied and, when
 * @p produced is non-NULL, accumulates the same count into @c *produced
 * (used by the outer state machine to compute the per-call return value).
 *
 * @param[in,out] ds       Decompression stream.
 * @param[in,out] out      Caller output buffer.
 * @param[in,out] produced Optional running count of bytes produced this call.
 * @return @c 1 once @c decoded is fully drained, @c 0 otherwise.
 */
static int ds_drain_decoded(zxc_dstream* ds, zxc_outbuf_t* out, size_t* produced) {
    const size_t avail_out = out->size - out->pos;
    const size_t avail_dec = ds->decoded_size - ds->decoded_pos;
    const size_t n = avail_out < avail_dec ? avail_out : avail_dec;
    if (n) {
        ZXC_MEMCPY((uint8_t*)out->dst + out->pos, ds->decoded + ds->decoded_pos, n);
        out->pos += n;
        ds->decoded_pos += n;
        ds->total_out += n;
        if (produced) *produced += n;
    }
    return ds->decoded_pos == ds->decoded_size;
}

/**
 * @brief Handles the @c DS_NEED_FILE_HEADER state.
 *
 * Pulls the 16-byte file header into @c scratch, parses it via
 * @ref zxc_read_file_header, and lazily allocates the @c payload and
 * @c decoded buffers (sized from the negotiated @c block_size).  The
 * @c decoded buffer is over-allocated by @ref ZXC_DECOMPRESS_TAIL_PAD bytes to
 * absorb wild-copy overflow and give the decoder's 4x ML bounds checks their
 * required tail headroom.  Initialises the underlying
 * decompression context and transitions to @c DS_NEED_BLOCK_HEADER.
 *
 * @param[in,out] ds Decompression stream.
 * @param[in,out] in Caller input buffer.
 * @return @c 1 if more input is needed, @c 0 to continue the outer loop,
 *         negative @ref zxc_error_t on validation/allocation failure.
 */
static int ds_handle_need_file_header(zxc_dstream* ds, zxc_inbuf_t* in) {
    if (!ds_pull_scratch(ds, in)) return 1;

    size_t bs = 0;
    int has_csum = 0;
    const int rc = zxc_read_file_header(ds->scratch, ds->scratch_used, &bs, &has_csum, NULL);
    if (UNLIKELY(rc != ZXC_OK)) return ds_set_error(ds, rc);  // LCOV_EXCL_LINE
    ds->block_size = bs;
    ds->file_has_checksum = has_csum;

    /* Allocate payload + decoded buffers now that block_size is known. */
    const uint64_t pb = zxc_compress_block_bound(ds->block_size);
    // LCOV_EXCL_START
    if (UNLIKELY(pb == 0 || pb > SIZE_MAX)) return ds_set_error(ds, ZXC_ERROR_OVERFLOW);
    // LCOV_EXCL_STOP
    ds->payload_cap = (size_t)pb;
    ds->payload = (uint8_t*)ZXC_MALLOC(ds->payload_cap);

    ds->decoded_cap = ds->block_size + ZXC_DECOMPRESS_TAIL_PAD;
    ds->decoded = (uint8_t*)ZXC_MALLOC(ds->decoded_cap);
    // LCOV_EXCL_START
    if (UNLIKELY(!ds->payload || !ds->decoded)) return ds_set_error(ds, ZXC_ERROR_MEMORY);

    if (UNLIKELY(zxc_cctx_init(&ds->inner, ds->block_size, 0, 0,
                               ds->file_has_checksum && ds->opts.checksum_enabled, 0) != ZXC_OK)) {
        return ds_set_error(ds, ZXC_ERROR_MEMORY);
    }
    // LCOV_EXCL_STOP
    ds->inner_initialized = 1;

    ds->state = DS_NEED_BLOCK_HEADER;
    ds->scratch_used = 0;
    ds->scratch_need = ZXC_BLOCK_HEADER_SIZE;
    return 0;
}

/**
 * @brief Handles the @c DS_NEED_BLOCK_HEADER state.
 *
 * Pulls 8 bytes into @c scratch and parses them as a block header.  If the
 * block is an EOF block, transitions to @c DS_PEEK_TAIL to disambiguate
 * between an optional SEK index and the file footer.  Otherwise, validates
 * the announced @c comp_size against the @c payload buffer capacity, copies
 * the parsed header into @c payload (the underlying decoder expects header
 * + body + optional checksum as a single contiguous frame), and transitions
 * to @c DS_NEED_BLOCK_PAYLOAD.
 *
 * @param[in,out] ds Decompression stream.
 * @param[in,out] in Caller input buffer.
 * @return @c 1 if more input is needed, @c 0 to continue the outer loop,
 *         negative @ref zxc_error_t on validation/allocation failure.
 */
static int ds_handle_need_block_header(zxc_dstream* ds, zxc_inbuf_t* in) {
    if (!ds_pull_scratch(ds, in)) return 1;

    const int rc = zxc_read_block_header(ds->scratch, ds->scratch_used, &ds->cur_bh);
    if (UNLIKELY(rc != ZXC_OK)) return ds_set_error(ds, rc);  // LCOV_EXCL_LINE

    if (ds->cur_bh.block_type == (uint8_t)ZXC_BLOCK_EOF) {
        /* EOF block: comp_size must be 0; no payload, no checksum. */
        if (UNLIKELY(ds->cur_bh.comp_size != 0)) return ds_set_error(ds, ZXC_ERROR_BAD_BLOCK_SIZE);
        ds->state = DS_PEEK_TAIL;
        ds->scratch_used = 0;
        ds->scratch_need = ZXC_BLOCK_HEADER_SIZE; /* sniff */
        return 0;
    }

    /* Normal data block: read comp_size [+ ZXC_BLOCK_CHECKSUM_SIZE if file-level checksums]. */
    const uint64_t need = (uint64_t)ds->cur_bh.comp_size +
                          (ds->file_has_checksum ? (uint64_t)ZXC_BLOCK_CHECKSUM_SIZE : 0u);
    if (UNLIKELY(need > ds->payload_cap)) return ds_set_error(ds, ZXC_ERROR_BAD_BLOCK_SIZE);

    /* Feed the full block (header + payload + opt csum) to zxc_decompress_block,
     * so prefix with the 8-byte header we just parsed. */
    ZXC_MEMCPY(ds->payload, ds->scratch, ZXC_BLOCK_HEADER_SIZE);
    ds->payload_used = ZXC_BLOCK_HEADER_SIZE;
    ds->payload_need = (size_t)need + ZXC_BLOCK_HEADER_SIZE;
    // LCOV_EXCL_START
    if (UNLIKELY(ds->payload_need > ds->payload_cap)) {
        /* grow */
        uint8_t* nb = (uint8_t*)ZXC_REALLOC(ds->payload, ds->payload_need);
        if (UNLIKELY(!nb)) return ds_set_error(ds, ZXC_ERROR_MEMORY);
        ds->payload = nb;
        ds->payload_cap = ds->payload_need;
    }
    // LCOV_EXCL_STOP
    ds->state = DS_NEED_BLOCK_PAYLOAD;
    return 0;
}

/**
 * @brief Push-side entry point: feeds compressed input and drains decoded output.
 *
 * Drives the @ref dstream_state_t machine: file header, then per-block
 * (header + payload + optional checksum) -> decode -> emit, repeated until
 * the EOF block, optional SEK index block, and file footer have been parsed
 * and validated.  Each call makes as much progress as @p in and @p out
 * allow; the function is fully reentrant.  See the public contract in
 * @ref zxc_dstream_decompress for full semantics.
 *
 * @par End of stream
 * Once @c DS_VALIDATE_FOOTER succeeds, the stream is in @c DS_DONE; further
 * calls return @c 0 without consuming input.
 *
 * @par Errors
 * On any negative return the stream becomes errored (sticky); subsequent
 * calls keep returning the same code until @ref zxc_dstream_free.
 *
 * @param[in,out] ds  Decompression stream.
 * @param[in,out] out Caller output buffer.
 * @param[in,out] in  Caller input buffer.
 * @return @c >0 number of decompressed bytes written into @p out this call,
 *         @c 0 stream complete (DONE) or no progress possible,
 *         @c <0 a @ref zxc_error_t code.
 */
int64_t zxc_dstream_decompress(zxc_dstream* ds, zxc_outbuf_t* out, zxc_inbuf_t* in) {
    if (UNLIKELY(!ds || !out || !in || in->pos > in->size || out->pos > out->size ||
                 (in->size > in->pos && !in->src) || (out->size > out->pos && !out->dst))) {
        return ZXC_ERROR_NULL_INPUT;
    }
    if (UNLIKELY(ds->state == DS_ERRORED)) return ds->error_code;
    if (UNLIKELY(ds->state == DS_DONE)) return 0;

    size_t produced = 0;

    for (;;) {
        switch (ds->state) {
            case DS_NEED_FILE_HEADER: {
                const int rc = ds_handle_need_file_header(ds, in);
                if (rc == 1) return (int64_t)produced;
                if (rc < 0) return rc;
                break;
            }

            case DS_NEED_BLOCK_HEADER: {
                const int rc = ds_handle_need_block_header(ds, in);
                if (rc == 1) return (int64_t)produced;
                if (rc < 0) return rc;
                break;
            }

            case DS_NEED_BLOCK_PAYLOAD: {
                if (!ds_pull_payload(ds, in)) return (int64_t)produced;
                ds->state = DS_DECODE_BLOCK;
                break;
            }

            case DS_DECODE_BLOCK: {
                const int dsz = zxc_decompress_chunk_wrapper(
                    &ds->inner, ds->payload, ds->payload_used, ds->decoded, ds->decoded_cap);
                if (UNLIKELY(dsz < 0)) return ds_set_error(ds, dsz);
                ds->decoded_size = (size_t)dsz;
                ds->decoded_pos = 0;

                /* If file-level checksum verification is enabled, fold this
                 * block's trailer into the rolling global hash (last
                 * ZXC_BLOCK_CHECKSUM_SIZE bytes of the *raw* block). */
                if (ds->opts.checksum_enabled && ds->file_has_checksum &&
                    ds->payload_used >= ZXC_BLOCK_CHECKSUM_SIZE) {
                    const uint32_t bh =
                        zxc_le32(ds->payload + ds->payload_used - ZXC_BLOCK_CHECKSUM_SIZE);
                    ds->global_hash = zxc_hash_combine_rotate(ds->global_hash, bh);
                }
                ds->state = DS_EMIT_DECODED;
                break;
            }

            case DS_EMIT_DECODED: {
                const int done = ds_drain_decoded(ds, out, &produced);
                if (!done) return (int64_t)produced;
                ds->state = DS_NEED_BLOCK_HEADER;
                ds->scratch_used = 0;
                ds->scratch_need = ZXC_BLOCK_HEADER_SIZE;
                break;
            }

            case DS_PEEK_TAIL: {
                if (!ds_pull_scratch(ds, in)) return (int64_t)produced;
                /* Try to interpret as a block header (SEK). */
                zxc_block_header_t peek;
                const int sek_rc = zxc_read_block_header(ds->scratch, ds->scratch_used, &peek);
                if (sek_rc == ZXC_OK && peek.block_type == (uint8_t)ZXC_BLOCK_SEK) {
                    /* SEK block: skip its payload (peek.comp_size bytes). */
                    ds->sek_remaining = (size_t)peek.comp_size;
                    ds->state = DS_DRAIN_SEK_PAYLOAD;
                    break;
                }
                /* Not SEK -> these 8 bytes are the first 8 of the 12-byte footer. */
                ds->state = DS_NEED_FOOTER_REST;
                ds->scratch_need = ZXC_FILE_FOOTER_SIZE; /* keep first 8, want 4 more */
                break;
            }

            case DS_DRAIN_SEK_PAYLOAD: {
                const size_t avail = in->size - in->pos;
                const size_t n = avail < ds->sek_remaining ? avail : ds->sek_remaining;
                in->pos += n;
                ds->sek_remaining -= n;
                if (ds->sek_remaining > 0) return (int64_t)produced;
                ds->state = DS_NEED_FOOTER_FULL;
                ds->scratch_used = 0;
                ds->scratch_need = ZXC_FILE_FOOTER_SIZE;
                break;
            }

            case DS_NEED_FOOTER_REST:
            case DS_NEED_FOOTER_FULL: {
                if (!ds_pull_scratch(ds, in)) return (int64_t)produced;
                ds->state = DS_VALIDATE_FOOTER;
                break;
            }

            case DS_VALIDATE_FOOTER: {
                const uint64_t declared = zxc_le64(ds->scratch);
                if (UNLIKELY(declared != ds->total_out))
                    return ds_set_error(ds, ZXC_ERROR_CORRUPT_DATA);
                if (ds->opts.checksum_enabled && ds->file_has_checksum) {
                    const uint32_t fh = zxc_le32(ds->scratch + sizeof(uint64_t));
                    if (UNLIKELY(fh != ds->global_hash))
                        return ds_set_error(ds, ZXC_ERROR_BAD_CHECKSUM);
                }
                ds->state = DS_DONE;
                return (int64_t)produced;
            }

            case DS_DONE:
            case DS_ERRORED:
                return ds->state == DS_ERRORED ? ds->error_code : (int64_t)produced;
        }
    }
}
