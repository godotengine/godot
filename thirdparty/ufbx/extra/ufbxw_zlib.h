#ifndef UFBXW_ZLIB_H_INCLUDED
#define UFBXW_ZLIB_H_INCLUDED

#if !defined(ufbxw_zlib_abi)
	#if defined(UFBXW_ZLIB_STATIC)
		#define ufbxw_zlib_abi static
	#else
		#define ufbxw_zlib_abi
	#endif
#endif

typedef void* ufbxw_zlib_alloc_fn(void *user, unsigned int items, unsigned int size);
typedef void ufbxw_zlib_free_fn(void *user, void *address);

typedef struct ufbxw_zlib_allocator {
	ufbxw_zlib_alloc_fn *alloc_fn;
	ufbxw_zlib_free_fn *free_fn;
	void *user;
} ufbxw_zlib_allocator;

typedef struct ufbxw_zlib_opts {
	ufbxw_zlib_allocator allocator;
} ufbxw_zlib_opts;

#if defined(__cplusplus)
extern "C" {
#endif

ufbxw_zlib_abi void ufbxw_zlib_setup(struct ufbxw_deflate *deflate, const ufbxw_zlib_opts *opts);

#if defined(__cplusplus)
}
#endif

#endif

#ifdef UFBXW_ZLIB_IMPLEMENTATION
#ifndef UFBXW_ZLIB_H_IMPLEMENTED
#define UFBXW_ZLIB_H_IMPLEMENTED

#if !defined(UFBXW_VERSION)
	#error "Please include ufbx_write.h before implementing ufbxw_zlib.h"
#endif

#if !defined(ZLIB_VERNUM)
	#error "Please include zlib.h before implementing ufbxw_zlib.h"
#endif

#include <stdlib.h>
#include <string.h>

static size_t ufbxw_zlib_begin(void *user, size_t input_size)
{
	z_stream *zs = (z_stream*)user;
	deflateReset(zs);
	return 0;
}

static ufbxw_deflate_advance_result ufbxw_zlib_advance(void *user, ufbxw_deflate_advance_status *status, void *dst, size_t dst_size, const void *src, size_t src_size, uint32_t flags)
{
	if (src_size > UINT_MAX) src_size = UINT_MAX;
	if (dst_size > UINT_MAX) dst_size = UINT_MAX;

	z_stream *zs = (z_stream*)user;

    zs->next_in = (z_const Bytef*)src;
    zs->avail_in = (uInt)src_size;

    zs->next_out = (Bytef*)dst;
	zs->avail_out = (uInt)dst_size;

	int flush = Z_NO_FLUSH;
	if (flags & UFBXW_DEFLATE_ADVANCE_FLAG_FINISH) {
		flush = Z_FINISH;
	} else if (flags & UFBXW_DEFLATE_ADVANCE_FLAG_FLUSH) {
		flush = Z_SYNC_FLUSH;
	}

	int res = deflate(zs, flush);

	status->bytes_read += (size_t)(zs->next_in - (z_const Bytef*)src);
	status->bytes_written += (size_t)(zs->next_out - (Bytef*)dst);

	if (res == Z_STREAM_END) {
		return UFBXW_DEFLATE_ADVANCE_RESULT_COMPLETED;
	} else if (res == Z_OK || res == Z_BUF_ERROR) {
		return UFBXW_DEFLATE_ADVANCE_RESULT_INCOMPLETE;
	} else {
		return UFBXW_DEFLATE_ADVANCE_RESULT_ERROR;
	}
}

static void ufbxw_zlib_free(void *user)
{
	z_stream *zs = (z_stream*)user;
	deflateEnd(zs);
	free(zs);
}

static bool ufbxw_zlib_init(void *user, ufbxw_deflate_compressor *compressor, int32_t compression_level)
{
	z_stream *zs = (z_stream*)malloc(sizeof(z_stream));
	if (!zs) return false;
	memset(zs, 0, sizeof(z_stream));

	if (deflateInit(zs, compression_level) != Z_OK) return false;

	const ufbxw_zlib_opts *opts = (const ufbxw_zlib_opts*)user;
	if (opts) {
		zs->zalloc = opts->allocator.alloc_fn;
		zs->zfree = opts->allocator.free_fn;
		zs->opaque = opts->allocator.user;
	}

	compressor->begin_fn = &ufbxw_zlib_begin;
	compressor->advance_fn = &ufbxw_zlib_advance;
	compressor->free_fn = &ufbxw_zlib_free;
	compressor->user = zs;

	return true;
}

#if defined(__cplusplus)
extern "C" {
#endif

ufbxw_zlib_abi void ufbxw_zlib_setup(struct ufbxw_deflate *deflate, const ufbxw_zlib_opts *opts)
{
	deflate->create_cb.fn = &ufbxw_zlib_init;
	deflate->create_cb.user = (void*)opts;
	deflate->streaming_input = true;
	deflate->streaming_output = true;
}

#if defined(__cplusplus)
}
#endif

#endif
#endif
