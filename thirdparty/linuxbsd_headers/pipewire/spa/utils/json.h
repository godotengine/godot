/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2020 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_UTILS_JSON_H
#define SPA_UTILS_JSON_H

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include <spa/utils/defs.h>
#include <spa/utils/string.h>

/** \defgroup spa_json JSON
 * Relaxed JSON variant parsing
 */

/**
 * \addtogroup spa_json
 * \{
 */

/* a simple JSON compatible tokenizer */
struct spa_json {
	const char *cur;
	const char *end;
	struct spa_json *parent;
	uint32_t state;
	uint32_t depth;
};

#define SPA_JSON_INIT(data,size) ((struct spa_json) { (data), (data)+(size), 0, 0, 0 })

static inline void spa_json_init(struct spa_json * iter, const char *data, size_t size)
{
	*iter =  SPA_JSON_INIT(data, size);
}
#define SPA_JSON_ENTER(iter) ((struct spa_json) { (iter)->cur, (iter)->end, (iter), 0, 0 })

static inline void spa_json_enter(struct spa_json * iter, struct spa_json * sub)
{
	*sub = SPA_JSON_ENTER(iter);
}

#define SPA_JSON_SAVE(iter) ((struct spa_json) { (iter)->cur, (iter)->end, })

/** Get the next token. \a value points to the token and the return value
 * is the length. */
static inline int spa_json_next(struct spa_json * iter, const char **value)
{
	int utf8_remain = 0;
	enum { __NONE, __STRUCT, __BARE, __STRING, __UTF8, __ESC, __COMMENT };

	*value = iter->cur;
	for (; iter->cur < iter->end; iter->cur++) {
		unsigned char cur = (unsigned char)*iter->cur;
 again:
		switch (iter->state) {
		case __NONE:
			iter->state = __STRUCT;
			iter->depth = 0;
			goto again;
		case __STRUCT:
			switch (cur) {
			case '\0': case '\t': case ' ': case '\r': case '\n': case ':': case '=': case ',':
				continue;
			case '#':
				iter->state = __COMMENT;
				continue;
			case '"':
				*value = iter->cur;
				iter->state = __STRING;
				continue;
			case '[': case '{':
				*value = iter->cur;
				if (++iter->depth > 1)
					continue;
				iter->cur++;
				return 1;
			case '}': case ']':
				if (iter->depth == 0) {
					if (iter->parent)
						iter->parent->cur = iter->cur;
					return 0;
				}
				--iter->depth;
				continue;
			default:
				*value = iter->cur;
				iter->state = __BARE;
			}
			continue;
		case __BARE:
			switch (cur) {
			case '\t': case ' ': case '\r': case '\n':
			case ':': case ',': case '=': case ']': case '}':
				iter->state = __STRUCT;
				if (iter->depth > 0)
					goto again;
				return iter->cur - *value;
			}
			continue;
		case __STRING:
			switch (cur) {
			case '\\':
				iter->state = __ESC;
				continue;
			case '"':
				iter->state = __STRUCT;
				if (iter->depth > 0)
					continue;
				return ++iter->cur - *value;
			case 240 ... 247:
				utf8_remain++;
				SPA_FALLTHROUGH;
			case 224 ... 239:
				utf8_remain++;
				SPA_FALLTHROUGH;
			case 192 ... 223:
				utf8_remain++;
				iter->state = __UTF8;
				continue;
			default:
				if (cur >= 32 && cur <= 126)
					continue;
			}
			return -1;
		case __UTF8:
			switch (cur) {
			case 128 ... 191:
				if (--utf8_remain == 0)
					iter->state = __STRING;
				continue;
			}
			return -1;
		case __ESC:
			switch (cur) {
			case '"': case '\\': case '/': case 'b': case 'f':
			case 'n': case 'r': case 't': case 'u':
				iter->state = __STRING;
				continue;
			}
			return -1;
		case __COMMENT:
			switch (cur) {
			case '\n': case '\r':
				iter->state = __STRUCT;
			}
		}

	}
	if (iter->depth != 0)
		return -1;
	if (iter->state != __STRUCT) {
		iter->state = __STRUCT;
		return iter->cur - *value;
	}
	return 0;
}

static inline int spa_json_enter_container(struct spa_json *iter, struct spa_json *sub, char type)
{
	const char *value;
	if (spa_json_next(iter, &value) <= 0 || *value != type)
		return -1;
	spa_json_enter(iter, sub);
	return 1;
}

static inline int spa_json_is_container(const char *val, int len)
{
	return len > 0 && (*val == '{'  || *val == '[');
}

static inline int spa_json_container_len(struct spa_json *iter, const char *value, int len SPA_UNUSED)
{
	const char *val;
	struct spa_json sub;
	spa_json_enter(iter, &sub);
	while (spa_json_next(&sub, &val) > 0);
	return sub.cur + 1 - value;
}

/* object */
static inline int spa_json_is_object(const char *val, int len)
{
	return len > 0 && *val == '{';
}
static inline int spa_json_enter_object(struct spa_json *iter, struct spa_json *sub)
{
	return spa_json_enter_container(iter, sub, '{');
}

/* array */
static inline bool spa_json_is_array(const char *val, int len)
{
	return len > 0 && *val == '[';
}
static inline int spa_json_enter_array(struct spa_json *iter, struct spa_json *sub)
{
	return spa_json_enter_container(iter, sub, '[');
}

/* null */
static inline bool spa_json_is_null(const char *val, int len)
{
	return len == 4 && strncmp(val, "null", 4) == 0;
}

/* float */
static inline int spa_json_parse_float(const char *val, int len, float *result)
{
	char *end;
	if (strspn(val, "+-0123456789.Ee") < (size_t)len)
		return 0;
	*result = spa_strtof(val, &end);
	return len > 0 && end == val + len;
}

static inline bool spa_json_is_float(const char *val, int len)
{
	float dummy;
	return spa_json_parse_float(val, len, &dummy);
}
static inline int spa_json_get_float(struct spa_json *iter, float *res)
{
	const char *value;
	int len;
	if ((len = spa_json_next(iter, &value)) <= 0)
		return -1;
	return spa_json_parse_float(value, len, res);
}

static inline char *spa_json_format_float(char *str, int size, float val)
{
	if (SPA_UNLIKELY(!isnormal(val))) {
		if (val == INFINITY)
			val = FLT_MAX;
		else if (val == -INFINITY)
			val = FLT_MIN;
		else
			val = 0.0f;
	}
	return spa_dtoa(str, size, val);
}

/* int */
static inline int spa_json_parse_int(const char *val, int len, int *result)
{
	char *end;
	*result = strtol(val, &end, 0);
	return len > 0 && end == val + len;
}
static inline bool spa_json_is_int(const char *val, int len)
{
	int dummy;
	return spa_json_parse_int(val, len, &dummy);
}
static inline int spa_json_get_int(struct spa_json *iter, int *res)
{
	const char *value;
	int len;
	if ((len = spa_json_next(iter, &value)) <= 0)
		return -1;
	return spa_json_parse_int(value, len, res);
}

/* bool */
static inline bool spa_json_is_true(const char *val, int len)
{
	return len == 4 && strncmp(val, "true", 4) == 0;
}

static inline bool spa_json_is_false(const char *val, int len)
{
	return len == 5 && strncmp(val, "false", 5) == 0;
}

static inline bool spa_json_is_bool(const char *val, int len)
{
	return spa_json_is_true(val, len) || spa_json_is_false(val, len);
}

static inline int spa_json_parse_bool(const char *val, int len, bool *result)
{
	if ((*result = spa_json_is_true(val, len)))
		return 1;
	if (!(*result = !spa_json_is_false(val, len)))
		return 1;
	return -1;
}
static inline int spa_json_get_bool(struct spa_json *iter, bool *res)
{
	const char *value;
	int len;
	if ((len = spa_json_next(iter, &value)) <= 0)
		return -1;
	return spa_json_parse_bool(value, len, res);
}

/* string */
static inline bool spa_json_is_string(const char *val, int len)
{
	return len > 1 && *val == '"';
}

static inline int spa_json_parse_hex(const char *p, int num, uint32_t *res)
{
	int i;
	*res = 0;
	for (i = 0; i < num; i++) {
		char v = p[i];
		if (v >= '0' && v <= '9')
			v = v - '0';
		else if (v >= 'a' && v <= 'f')
			v = v - 'a' + 10;
		else if (v >= 'A' && v <= 'F')
			v = v - 'A' + 10;
		else
			return -1;
		*res = (*res << 4) | v;
	}
	return 1;
}

static inline int spa_json_parse_stringn(const char *val, int len, char *result, int maxlen)
{
	const char *p;
	if (maxlen <= len)
		return -1;
	if (!spa_json_is_string(val, len)) {
		if (result != val)
			strncpy(result, val, len);
		result += len;
	} else {
		for (p = val+1; p < val + len; p++) {
			if (*p == '\\') {
				p++;
				if (*p == 'n')
					*result++ = '\n';
				else if (*p == 'r')
					*result++ = '\r';
				else if (*p == 'b')
					*result++ = '\b';
				else if (*p == 't')
					*result++ = '\t';
				else if (*p == 'f')
					*result++ = '\f';
				else if (*p == 'u') {
					uint8_t prefix[] = { 0, 0xc0, 0xe0, 0xf0 };
					uint32_t idx, n, v, cp, enc[] = { 0x80, 0x800, 0x10000 };
					if (val + len - p < 5 ||
					    spa_json_parse_hex(p+1, 4, &cp) < 0) {
						*result++ = *p;
						continue;
					}
					p += 4;

					if (cp >= 0xd800 && cp <= 0xdbff) {
						if (val + len - p < 7 ||
						    p[1] != '\\' || p[2] != 'u' ||
						    spa_json_parse_hex(p+3, 4, &v) < 0 ||
						    v < 0xdc00 || v > 0xdfff)
							continue;
						p += 6;
						cp = 0x010000 | ((cp & 0x3ff) << 10) | (v & 0x3ff);
					} else if (cp >= 0xdc00 && cp <= 0xdfff)
						continue;

					for (idx = 0; idx < 3; idx++)
						if (cp < enc[idx])
							break;
					for (n = idx; n > 0; n--, cp >>= 6)
						result[n] = (cp | 0x80) & 0xbf;
					*result++ = (cp | prefix[idx]) & 0xff;
					result += idx;
				} else
					*result++ = *p;
			} else if (*p == '\"') {
				break;
			} else
				*result++ = *p;
		}
	}
	*result = '\0';
	return 1;
}

static inline int spa_json_parse_string(const char *val, int len, char *result)
{
	return spa_json_parse_stringn(val, len, result, len+1);
}

static inline int spa_json_get_string(struct spa_json *iter, char *res, int maxlen)
{
	const char *value;
	int len;
	if ((len = spa_json_next(iter, &value)) <= 0)
		return -1;
	return spa_json_parse_stringn(value, len, res, maxlen);
}

static inline int spa_json_encode_string(char *str, int size, const char *val)
{
	int len = 0;
	static const char hex[] = { "0123456789abcdef" };
#define __PUT(c) { if (len < size) *str++ = c; len++; }
	__PUT('"');
	while (*val) {
		switch (*val) {
		case '\n':
			__PUT('\\'); __PUT('n');
			break;
		case '\r':
			__PUT('\\'); __PUT('r');
			break;
		case '\b':
			__PUT('\\'); __PUT('b');
			break;
		case '\t':
			__PUT('\\'); __PUT('t');
			break;
		case '\f':
			__PUT('\\'); __PUT('f');
			break;
		case '\\':
		case '"':
			__PUT('\\'); __PUT(*val);
			break;
		default:
			if (*val > 0 && *val < 0x20) {
				__PUT('\\'); __PUT('u');
				__PUT('0'); __PUT('0');
				__PUT(hex[((*val)>>4)&0xf]); __PUT(hex[(*val)&0xf]);
			} else {
				__PUT(*val);
			}
			break;
		}
		val++;
	}
	__PUT('"');
	__PUT('\0');
#undef __PUT
	return len-1;
}

/**
 * \}
 */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SPA_UTILS_JSON_H */
