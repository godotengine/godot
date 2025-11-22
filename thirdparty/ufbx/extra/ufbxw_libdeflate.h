#ifndef UFBXW_LIBDEFLATE_H_INCLUDED
#define UFBXW_LIBDEFLATE_H_INCLUDED

#include <stddef.h>

#if !defined(ufbxw_libdeflate_abi)
	#if defined(UFBXW_LIBDEFLATE_STATIC)
		#define ufbxw_libdeflate_abi static
	#else
		#define ufbxw_libdeflate_abi
	#endif
#endif

typedef void *ufbxw_libdeflate_alloc_fn(size_t size);
typedef void ufbxw_libdeflate_free_fn(void *address);

typedef struct ufbxw_libdeflate_allocator {
	ufbxw_libdeflate_alloc_fn *alloc_fn;
	ufbxw_libdeflate_free_fn *free_fn;
	void *user;
} ufbxw_libdeflate_allocator;

typedef struct ufbxw_libdeflate_opts {
	ufbxw_libdeflate_allocator allocator;
} ufbxw_libdeflate_opts;

#if defined(__cplusplus)
extern "C" {
#endif

ufbxw_libdeflate_abi void ufbxw_libdeflate_setup(struct ufbxw_deflate *deflate, const ufbxw_libdeflate_opts *opts);

#if defined(__cplusplus)
}
#endif

#endif

#ifdef UFBXW_LIBDEFLATE_IMPLEMENTATION
#ifndef UFBXW_LIBDEFLATE_H_IMPLEMENTED
#define UFBXW_LIBDEFLATE_H_IMPLEMENTED

#if !defined(UFBXW_VERSION)
	#error "Please include ufbx_write.h before implementing ufbxw_libdeflate.h"
#endif

#if !defined(LIBDEFLATE_VERSION_MAJOR)
	#error "Please include libdeflate.h before implementing ufbxw_libdeflate.h"
#endif

static size_t ufbxw_libdeflate_begin(void *user, size_t input_size)
{
	struct libdeflate_compressor *c = (struct libdeflate_compressor*)user;
	return libdeflate_zlib_compress_bound(c, input_size);
}

static ufbxw_deflate_advance_result ufbxw_libdeflate_advance(void *user, ufbxw_deflate_advance_status *status, void *dst, size_t dst_size, const void *src, size_t src_size, uint32_t flags)
{
	struct libdeflate_compressor *c = (struct libdeflate_compressor*)user;
	size_t dst_written = libdeflate_zlib_compress(c, src, src_size, dst, dst_size);
	if (dst_written == 0) return UFBXW_DEFLATE_ADVANCE_RESULT_ERROR;

	status->bytes_read = src_size;
	status->bytes_written = dst_written;
	return UFBXW_DEFLATE_ADVANCE_RESULT_COMPLETED;
}

static void ufbxw_libdeflate_free(void *user)
{
	struct libdeflate_compressor *c = (struct libdeflate_compressor*)user;
	libdeflate_free_compressor(c);
}

static bool ufbxw_libdeflate_init(void *user, ufbxw_deflate_compressor *compressor, int32_t compression_level)
{
	const ufbxw_libdeflate_opts *opts = (const ufbxw_libdeflate_opts*)user;

	struct libdeflate_compressor *c = NULL;
	if (opts) {
		struct libdeflate_options options = { sizeof(struct libdeflate_options) };
		options.malloc_func = opts->allocator.alloc_fn;
		options.free_func = opts->allocator.free_fn;
		c = libdeflate_alloc_compressor_ex(compression_level, &options);
	} else {
		c = libdeflate_alloc_compressor(compression_level);
	}
	if (!c) return false;

	compressor->begin_fn = &ufbxw_libdeflate_begin;
	compressor->advance_fn = &ufbxw_libdeflate_advance;
	compressor->free_fn = &ufbxw_libdeflate_free;
	compressor->user = c;

	return true;
}

#if defined(__cplusplus)
extern "C" {
#endif

ufbxw_libdeflate_abi void ufbxw_libdeflate_setup(struct ufbxw_deflate *deflate, const ufbxw_libdeflate_opts *opts)
{
	deflate->create_cb.fn = &ufbxw_libdeflate_init;
	deflate->create_cb.user = (void*)opts;
	deflate->streaming_input = false;
	deflate->streaming_output = false;
}

#if defined(__cplusplus)
}
#endif

#endif
#endif

