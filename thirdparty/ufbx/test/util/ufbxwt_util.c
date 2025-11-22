#include "ufbxwt_util.h"

#ifdef UFBXWT_HAS_LIBDEFLATE
	#include "../../extra/ufbxw_libdeflate.h"
#endif

#ifdef UFBXWT_HAS_ZLIB
	#include "../../extra/ufbxw_zlib.h"
#endif

#ifdef UFBXWT_HAS_FMTLIB
	#include "../../extra/ufbxw_fmtlib.h"
#endif

#ifdef UFBXWT_HAS_TO_CHARS
	#include "../../extra/ufbxw_to_chars.h"
#endif

#ifdef UFBXWT_HAS_CPP_THREADS
	#include "../../extra/ufbxw_cpp_threads.h"
#endif

#include <assert.h>

#define ufbxwt_assert(cond) assert(cond)

bool ufbxwt_deflate_setup(ufbxw_deflate *deflate, ufbxwt_deflate_impl impl)
{
	switch (impl) {
	case UFBXWT_DEFLATE_IMPL_NONE:
		return true;

	case UFBXWT_DEFLATE_IMPL_LIBDEFLATE:
		#if UFBXWT_HAS_LIBDEFLATE
			ufbxw_libdeflate_setup(deflate, NULL);
			return true;
		#endif
		return false;

	case UFBXWT_DEFLATE_IMPL_ZLIB:
		#if UFBXWT_HAS_LIBDEFLATE
			ufbxw_zlib_setup(deflate, NULL);
			return true;
		#endif
		return false;

	default:
		ufbxwt_assert(false);
		break;
	}
	return false;
}

const char *ufbxwt_deflate_impl_name(ufbxwt_deflate_impl impl)
{
	switch (impl) {
	case UFBXWT_DEFLATE_IMPL_NONE: return "none";
	case UFBXWT_DEFLATE_IMPL_LIBDEFLATE: return "libdeflate";
	case UFBXWT_DEFLATE_IMPL_ZLIB: return "zlib";
	default: return "";
	}
}

bool ufbxwt_ascii_format_setup(ufbxw_ascii_formatter *formatter, ufbxwt_ascii_format_impl impl)
{
	switch (impl) {
	case UFBXWT_ASCII_FORMAT_IMPL_DEFAULT:
		return true;

	case UFBXWT_ASCII_FORMAT_IMPL_FMTLIB:
		#if UFBXWT_HAS_FMTLIB
			ufbxw_fmtlib_setup(formatter);
			return true;
		#endif
		return false;

	case UFBXWT_ASCII_FORMAT_IMPL_TO_CHARS:
		#if UFBXWT_HAS_TO_CHARS
			ufbxw_to_chars_setup(formatter);
			return true;
		#endif
		return false;

	default:
		ufbxwt_assert(false);
		break;
	}
	return false;
}

const char *ufbxwt_ascii_format_name(ufbxwt_ascii_format_impl impl)
{
	switch (impl) {
	case UFBXWT_ASCII_FORMAT_IMPL_DEFAULT: return "default";
	case UFBXWT_ASCII_FORMAT_IMPL_FMTLIB: return "fmtlib";
	case UFBXWT_ASCII_FORMAT_IMPL_TO_CHARS: return "to_chars";
	default: return "";
	}
}

bool ufbxwt_thread_setup(ufbxw_thread_sync *sync, ufbxw_thread_pool *pool, ufbxwt_thread_impl impl)
{
	switch (impl) {
	case UFBXWT_THREAD_IMPL_NONE:
		return true;

	case UFBXWT_THREAD_IMPL_CPP_THREADS:
		#if UFBXWT_HAS_LIBDEFLATE
			ufbxw_cpp_threads_setup_sync(sync);
			ufbxw_cpp_threads_setup_pool(pool);
			return true;
		#endif
		return false;

	default:
		ufbxwt_assert(false);
		break;
	}
	return false;
}

const char *ufbxwt_thread_impl_name(ufbxwt_thread_impl impl)
{
	switch (impl) {
	case UFBXWT_THREAD_IMPL_NONE: return "none";
	case UFBXWT_THREAD_IMPL_CPP_THREADS: return "cpp_threads";
	default: return "";
	}
}

