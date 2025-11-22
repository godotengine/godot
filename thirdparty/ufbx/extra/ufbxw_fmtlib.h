#ifndef UFBXW_FMTLIB_H_INCLUDED
#define UFBXW_FMTLIB_H_INCLUDED

#include <stddef.h>

#if !defined(ufbxw_fmtlib_abi)
	#if defined(UFBXW_FMTLIB_STATIC)
		#define ufbxw_fmtlib_abi static
	#else
		#define ufbxw_fmtlib_abi
	#endif
#endif

#if defined(__cplusplus)
extern "C" {
#endif

ufbxw_fmtlib_abi void ufbxw_fmtlib_setup(struct ufbxw_ascii_formatter *formatter);

#if defined(__cplusplus)
}
#endif

#endif

#ifdef UFBXW_FMTLIB_IMPLEMENTATION
#ifndef UFBXW_FMTLIB_H_IMPLEMENTED
#define UFBXW_FMTLIB_H_IMPLEMENTED

#if !defined(__cplusplus)
	#error "ufbxw_fmtlib.h should be implemented in a C++ file, though it can be used in another C file"
#endif

#if !defined(UFBXW_VERSION)
	#error "Please include ufbx_write.h before implementing ufbxw_fmtlib.h"
#endif

#if !defined(FMT_VERSION)
	#error "Please include fmt/format.h before implementing ufbxw_fmtlib.h"
#endif

#if defined(FMT_COMPILE)
	#define UFBXWI_FMT(fmt) FMT_COMPILE(fmt)
#else
	#define UFBXWI_FMT(fmt) fmt
#endif

static size_t ufbxw_fmtlib_format_int(void *user, char *dst, size_t dst_size, const int32_t *src, size_t src_count)
{
	char *d = dst;

	while (src_count >= 4) {
		d = fmt::format_to(d, UFBXWI_FMT("{},{},{},{},"), src[0], src[1], src[2], src[3]);
		src += 4;
		src_count -= 4;
	}
	while (src_count > 0) {
		d = fmt::format_to(d, UFBXWI_FMT("{},"), src[0]);
		src += 1;
		src_count -= 1;
	}

	return (size_t)(d - dst);
}

static size_t ufbxw_fmtlib_format_long(void *user, char *dst, size_t dst_size, const int64_t *src, size_t src_count)
{
	char *d = dst;

	while (src_count >= 4) {
		d = fmt::format_to(d, UFBXWI_FMT("{},{},{},{},"), src[0], src[1], src[2], src[3]);
		src += 4;
		src_count -= 4;
	}
	while (src_count > 0) {
		d = fmt::format_to(d, UFBXWI_FMT("{},"), src[0]);
		src += 1;
		src_count -= 1;
	}

	return (size_t)(d - dst);
}

static size_t ufbxw_fmtlib_format_float(void *user, char *dst, size_t dst_size, const float *src, size_t src_count, ufbxw_ascii_float_format format)
{
	char *d = dst;

	if (format == UFBXW_ASCII_FLOAT_FORMAT_FIXED_PRECISION) {
		while (src_count >= 4) {
			d = fmt::format_to(d, UFBXWI_FMT("{:.7g},{:.7g},{:.7g},{:.7g},"), src[0], src[1], src[2], src[3]);
			src += 4;
			src_count -= 4;
		}
		while (src_count > 0) {
			d = fmt::format_to(d, UFBXWI_FMT("{:.7g},"), src[0]);
			src += 1;
			src_count -= 1;
		}
	} else if (format == UFBXW_ASCII_FLOAT_FORMAT_ROUND_TRIP) {
		while (src_count >= 4) {
			d = fmt::format_to(d, UFBXWI_FMT("{:g},{:g},{:g},{:g},"), src[0], src[1], src[2], src[3]);
			src += 4;
			src_count -= 4;
		}
		while (src_count > 0) {
			d = fmt::format_to(d, UFBXWI_FMT("{:g},"), src[0]);
			src += 1;
			src_count -= 1;
		}
	}

	return (size_t)(d - dst);
}

static size_t ufbxw_fmtlib_format_double(void *user, char *dst, size_t dst_size, const double *src, size_t src_count, ufbxw_ascii_float_format format)
{
	char *d = dst;

	if (format == UFBXW_ASCII_FLOAT_FORMAT_FIXED_PRECISION) {
		while (src_count >= 4) {
			d = fmt::format_to(d, UFBXWI_FMT("{:.16g},{:.16g},{:.16g},{:.16g},"), src[0], src[1], src[2], src[3]);
			src += 4;
			src_count -= 4;
		}
		while (src_count > 0) {
			d = fmt::format_to(d, UFBXWI_FMT("{:.16g},"), src[0]);
			src += 1;
			src_count -= 1;
		}
	} else if (format == UFBXW_ASCII_FLOAT_FORMAT_ROUND_TRIP) {
		while (src_count >= 4) {
			d = fmt::format_to(d, UFBXWI_FMT("{:g},{:g},{:g},{:g},"), src[0], src[1], src[2], src[3]);
			src += 4;
			src_count -= 4;
		}
		while (src_count > 0) {
			d = fmt::format_to(d, UFBXWI_FMT("{:g},"), src[0]);
			src += 1;
			src_count -= 1;
		}
	}

	return (size_t)(d - dst);
}

#if defined(__cplusplus)
extern "C" {
#endif

ufbxw_fmtlib_abi void ufbxw_fmtlib_setup(struct ufbxw_ascii_formatter *formatter)
{
	formatter->format_int_fn = &ufbxw_fmtlib_format_int;
	formatter->format_long_fn = &ufbxw_fmtlib_format_long;
	formatter->format_float_fn = &ufbxw_fmtlib_format_float;
	formatter->format_double_fn = &ufbxw_fmtlib_format_double;
	formatter->user = nullptr;
}

#if defined(__cplusplus)
}
#endif

#endif
#endif

