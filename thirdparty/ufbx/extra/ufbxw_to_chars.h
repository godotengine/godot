#ifndef UFBXW_TO_CHARS_H_INCLUDED
#define UFBXW_TO_CHARS_H_INCLUDED

#include <stddef.h>

#if !defined(ufbxw_to_chars_abi)
	#if defined(UFBXW_TO_CHARS_STATIC)
		#define ufbxw_to_chars_abi static
	#else
		#define ufbxw_to_chars_abi
	#endif
#endif

#if defined(__cplusplus)
extern "C" {
#endif

ufbxw_to_chars_abi void ufbxw_to_chars_setup(struct ufbxw_ascii_formatter *formatter);

#if defined(__cplusplus)
}
#endif

#endif

#ifdef UFBXW_TO_CHARS_IMPLEMENTATION
#ifndef UFBXW_TO_CHARS_H_IMPLEMENTED
#define UFBXW_TO_CHARS_H_IMPLEMENTED

#if !defined(__cplusplus)
	#error "ufbxw_to_chars.h should be implemented in a C++ file, though it can be used in another C file"
#endif

#if !defined(UFBXW_VERSION)
	#error "Please include ufbx_write.h before implementing ufbxw_to_chars.h"
#endif

#include <charconv>

static size_t ufbxw_to_chars_format_int(void *user, char *dst, size_t dst_size, const int32_t *src, size_t src_count)
{
	char *d = dst, *end = dst + dst_size;

	for (size_t i = 0; i < src_count; i++) {
		d = std::to_chars(d, end, src[i]).ptr;
		*d++ = ',';
	}

	return (size_t)(d - dst);
}

static size_t ufbxw_to_chars_format_long(void *user, char *dst, size_t dst_size, const int64_t *src, size_t src_count)
{
	char *d = dst, *end = dst + dst_size;

	for (size_t i = 0; i < src_count; i++) {
		d = std::to_chars(d, end, src[i]).ptr;
		*d++ = ',';
	}

	return (size_t)(d - dst);
}

static size_t ufbxw_to_chars_format_float(void *user, char *dst, size_t dst_size, const float *src, size_t src_count, ufbxw_ascii_float_format format)
{
	char *d = dst, *end = dst + dst_size;

	if (format == UFBXW_ASCII_FLOAT_FORMAT_FIXED_PRECISION) {
		for (size_t i = 0; i < src_count; i++) {
			d = std::to_chars(d, end, src[i], std::chars_format::general, 7).ptr;
			*d++ = ',';
		}
	} else if (format == UFBXW_ASCII_FLOAT_FORMAT_ROUND_TRIP) {
		for (size_t i = 0; i < src_count; i++) {
			d = std::to_chars(d, end, src[i]).ptr;
			*d++ = ',';
		}
	}

	return (size_t)(d - dst);
}

static size_t ufbxw_to_chars_format_double(void *user, char *dst, size_t dst_size, const double *src, size_t src_count, ufbxw_ascii_float_format format)
{
	char *d = dst, *end = dst + dst_size;

	if (format == UFBXW_ASCII_FLOAT_FORMAT_FIXED_PRECISION) {
		for (size_t i = 0; i < src_count; i++) {
			d = std::to_chars(d, end, src[i], std::chars_format::general, 15).ptr;
			*d++ = ',';
		}
	} else if (format == UFBXW_ASCII_FLOAT_FORMAT_ROUND_TRIP) {
		for (size_t i = 0; i < src_count; i++) {
			d = std::to_chars(d, end, src[i]).ptr;
			*d++ = ',';
		}
	}

	return (size_t)(d - dst);
}

#if defined(__cplusplus)
extern "C" {
#endif

ufbxw_to_chars_abi void ufbxw_to_chars_setup(struct ufbxw_ascii_formatter *formatter)
{
	formatter->format_int_fn = &ufbxw_to_chars_format_int;
	formatter->format_long_fn = &ufbxw_to_chars_format_long;
	formatter->format_float_fn = &ufbxw_to_chars_format_float;
	formatter->format_double_fn = &ufbxw_to_chars_format_double;
	formatter->user = nullptr;
}

#if defined(__cplusplus)
}
#endif

#endif
#endif


