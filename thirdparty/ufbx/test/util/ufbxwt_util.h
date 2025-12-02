#ifndef UFBXWT_UTIL_H_INCLUDED
#define UFBXWT_UTIL_H_INCLUDED

typedef enum {
	UFBXWT_DEFLATE_IMPL_NONE,
	UFBXWT_DEFLATE_IMPL_LIBDEFLATE,
	UFBXWT_DEFLATE_IMPL_ZLIB,

	UFBXWT_DEFLATE_IMPL_COUNT,
} ufbxwt_deflate_impl;

typedef enum {
	UFBXWT_ASCII_FORMAT_IMPL_DEFAULT,
	UFBXWT_ASCII_FORMAT_IMPL_FMTLIB,
	UFBXWT_ASCII_FORMAT_IMPL_TO_CHARS,

	UFBXWT_ASCII_FORMAT_IMPL_COUNT,
} ufbxwt_ascii_format_impl;

typedef enum {
	UFBXWT_THREAD_IMPL_NONE,
	UFBXWT_THREAD_IMPL_CPP_THREADS,

	UFBXWT_THREAD_IMPL_COUNT,
} ufbxwt_thread_impl;

#include "../../ufbx_write.h"

bool ufbxwt_deflate_setup(ufbxw_deflate *deflate, ufbxwt_deflate_impl impl);
const char *ufbxwt_deflate_impl_name(ufbxwt_deflate_impl impl);

bool ufbxwt_ascii_format_setup(ufbxw_ascii_formatter *formatter, ufbxwt_ascii_format_impl impl);
const char *ufbxwt_ascii_format_name(ufbxwt_ascii_format_impl impl);

bool ufbxwt_thread_setup(ufbxw_thread_sync *sync, ufbxw_thread_pool *pool, ufbxwt_thread_impl impl);
const char *ufbxwt_thread_impl_name(ufbxwt_thread_impl impl);

#endif

