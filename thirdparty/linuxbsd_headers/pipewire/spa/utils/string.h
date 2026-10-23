/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2021 Red Hat, Inc. */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_UTILS_STRING_H
#define SPA_UTILS_STRING_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdarg.h>
#include <stdbool.h>
#include <errno.h>
#include <stdlib.h>
#include <locale.h>

#include <spa/utils/defs.h>

/**
 * \defgroup spa_string String handling
 * String handling utilities
 */

/**
 * \addtogroup spa_string
 * \{
 */

/**
 * \return true if the two strings are equal, false otherwise
 *
 * If both \a a and \a b are NULL, the two are considered equal.
 *
 */
static inline bool spa_streq(const char *s1, const char *s2)
{
	return SPA_LIKELY(s1 && s2) ? strcmp(s1, s2) == 0 : s1 == s2;
}

/**
 * \return true if the two strings are equal, false otherwise
 *
 * If both \a a and \a b are NULL, the two are considered equal.
 */
static inline bool spa_strneq(const char *s1, const char *s2, size_t len)
{
	return SPA_LIKELY(s1 && s2) ? strncmp(s1, s2, len) == 0 : s1 == s2;
}


/**
 * \return true if \a s starts with the \a prefix or false otherwise.
 * A \a s is NULL, it never starts with the given \a prefix. A \a prefix of
 * NULL is a bug in the caller.
 */
static inline bool spa_strstartswith(const char *s, const char *prefix)
{
	if (SPA_UNLIKELY(s == NULL))
		return false;

	spa_assert_se(prefix);

	return strncmp(s, prefix, strlen(prefix)) == 0;
}


/**
 * \return true if \a s ends with the \a suffix or false otherwise.
 * A \a s is NULL, it never ends with the given \a suffix. A \a suffix of
 * NULL is a bug in the caller.
 */
static inline bool spa_strendswith(const char *s, const char *suffix)
{
	size_t l1, l2;

	if (SPA_UNLIKELY(s == NULL))
	    return false;

	spa_assert_se(suffix);

	l1 = strlen(s);
	l2 = strlen(suffix);
	return l1 >= l2 && spa_streq(s + l1 - l2, suffix);
}

/**
 * Convert \a str to an int32_t with the given \a base and store the
 * result in \a val.
 *
 * On failure, the value of \a val is unmodified.
 *
 * \return true on success, false otherwise
 */
static inline bool spa_atoi32(const char *str, int32_t *val, int base)
{
	char *endptr;
	long v;

	if (!str || *str =='\0')
		return false;

	errno = 0;
	v = strtol(str, &endptr, base);
	if (errno != 0 || *endptr != '\0')
		return false;

	if (v != (int32_t)v)
		return false;

	*val = v;
	return true;
}

/**
 * Convert \a str to an uint32_t with the given \a base and store the
 * result in \a val.
 *
 * On failure, the value of \a val is unmodified.
 *
 * \return true on success, false otherwise
 */
static inline bool spa_atou32(const char *str, uint32_t *val, int base)
{
	char *endptr;
	unsigned long long v;

	if (!str || *str =='\0')
		return false;

	errno = 0;
	v = strtoull(str, &endptr, base);
	if (errno != 0 || *endptr != '\0')
		return false;

	if (v != (uint32_t)v)
		return false;

	*val = v;
	return true;
}

/**
 * Convert \a str to an int64_t with the given \a base and store the
 * result in \a val.
 *
 * On failure, the value of \a val is unmodified.
 *
 * \return true on success, false otherwise
 */
static inline bool spa_atoi64(const char *str, int64_t *val, int base)
{
	char *endptr;
	long long v;

	if (!str || *str =='\0')
		return false;

	errno = 0;
	v = strtoll(str, &endptr, base);
	if (errno != 0 || *endptr != '\0')
		return false;

	*val = v;
	return true;
}

/**
 * Convert \a str to an uint64_t with the given \a base and store the
 * result in \a val.
 *
 * On failure, the value of \a val is unmodified.
 *
 * \return true on success, false otherwise
 */
static inline bool spa_atou64(const char *str, uint64_t *val, int base)
{
	char *endptr;
	unsigned long long v;

	if (!str || *str =='\0')
		return false;

	errno = 0;
	v = strtoull(str, &endptr, base);
	if (errno != 0 || *endptr != '\0')
		return false;

	*val = v;
	return true;
}

/**
 * Convert \a str to a boolean. Allowed boolean values are "true" and a
 * literal "1", anything else is false.
 *
 * \return true on success, false otherwise
 */
static inline bool spa_atob(const char *str)
{
	return spa_streq(str, "true") || spa_streq(str, "1");
}

/**
 * "Safe" version of vsnprintf. Exactly the same as vsnprintf but the
 * returned value is clipped to `size - 1` and a negative or zero size
 * will abort() the program.
 *
 * \return The number of bytes printed, capped to `size-1`, or a negative
 * number on error.
 */
SPA_PRINTF_FUNC(3, 0)
static inline int spa_vscnprintf(char *buffer, size_t size, const char *format, va_list args)
{
	int r;

	spa_assert_se((ssize_t)size > 0);

	r = vsnprintf(buffer, size, format, args);
	if (SPA_UNLIKELY(r < 0))
		buffer[0] = '\0';
	if (SPA_LIKELY(r < (ssize_t)size))
		return r;
	return size - 1;
}

/**
 * "Safe" version of snprintf. Exactly the same as snprintf but the
 * returned value is clipped to `size - 1` and a negative or zero size
 * will abort() the program.
 *
 * \return The number of bytes printed, capped to `size-1`, or a negative
 * number on error.
 */
SPA_PRINTF_FUNC(3, 4)
static inline int spa_scnprintf(char *buffer, size_t size, const char *format, ...)
{
	int r;
	va_list args;

	va_start(args, format);
	r = spa_vscnprintf(buffer, size, format, args);
	va_end(args);

	return r;
}

/**
 * Convert \a str to a float in the C locale.
 *
 * If \a endptr is not NULL, a pointer to the character after the last character
 * used in the conversion is stored in the location referenced by endptr.
 *
 * \return the result float.
 */
static inline float spa_strtof(const char *str, char **endptr)
{
#ifndef __LOCALE_C_ONLY
	static locale_t locale = NULL;
	locale_t prev;
#endif
	float v;
#ifndef __LOCALE_C_ONLY
	if (SPA_UNLIKELY(locale == NULL))
		locale = newlocale(LC_ALL_MASK, "C", NULL);
	prev = uselocale(locale);
#endif
	v = strtof(str, endptr);
#ifndef __LOCALE_C_ONLY
	uselocale(prev);
#endif
	return v;
}

/**
 * Convert \a str to a float and store the result in \a val.
 *
 * On failure, the value of \a val is unmodified.
 *
 * \return true on success, false otherwise
 */
static inline bool spa_atof(const char *str, float *val)
{
	char *endptr;
	float v;

	if (!str || *str =='\0')
		return false;
	errno = 0;
	v = spa_strtof(str, &endptr);
	if (errno != 0 || *endptr != '\0')
		return false;

	*val = v;
	return true;
}

/**
 * Convert \a str to a double in the C locale.
 *
 * If \a endptr is not NULL, a pointer to the character after the last character
 * used in the conversion is stored in the location referenced by endptr.
 *
 * \return the result float.
 */
static inline double spa_strtod(const char *str, char **endptr)
{
#ifndef __LOCALE_C_ONLY
	static locale_t locale = NULL;
	locale_t prev;
#endif
	double v;
#ifndef __LOCALE_C_ONLY
	if (SPA_UNLIKELY(locale == NULL))
		locale = newlocale(LC_ALL_MASK, "C", NULL);
	prev = uselocale(locale);
#endif
	v = strtod(str, endptr);
#ifndef __LOCALE_C_ONLY
	uselocale(prev);
#endif
	return v;
}

/**
 * Convert \a str to a double and store the result in \a val.
 *
 * On failure, the value of \a val is unmodified.
 *
 * \return true on success, false otherwise
 */
static inline bool spa_atod(const char *str, double *val)
{
	char *endptr;
	double v;

	if (!str || *str =='\0')
		return false;

	errno = 0;
	v = spa_strtod(str, &endptr);
	if (errno != 0 || *endptr != '\0')
		return false;

	*val = v;
	return true;
}

static inline char *spa_dtoa(char *str, size_t size, double val)
{
	int i, l;
	l = spa_scnprintf(str, size, "%f", val);
	for (i = 0; i < l; i++)
		if (str[i] == ',')
			str[i] = '.';
	return str;
}

struct spa_strbuf {
	char *buffer;
	size_t maxsize;
	size_t pos;
};

static inline void spa_strbuf_init(struct spa_strbuf *buf, char *buffer, size_t maxsize)
{
	buf->buffer = buffer;
	buf->maxsize = maxsize;
	buf->pos = 0;
}

SPA_PRINTF_FUNC(2, 3)
static inline int spa_strbuf_append(struct spa_strbuf *buf, const char *fmt, ...)
{
	size_t remain = buf->maxsize - buf->pos;
	ssize_t written;
	va_list args;
	va_start(args, fmt);
	written = vsnprintf(&buf->buffer[buf->pos], remain, fmt, args);
	va_end(args);
	if (written > 0)
		buf->pos += SPA_MIN(remain, (size_t)written);
	return written;
}

/**
 * \}
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SPA_UTILS_STRING_H */
