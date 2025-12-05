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
#define SPA_JSON_ERROR_FLAG	0x100
	uint32_t state;
	uint32_t depth;
};

#define SPA_JSON_INIT(data,size) ((struct spa_json) { (data), (data)+(size), 0, 0, 0 })

static inline void spa_json_init(struct spa_json * iter, const char *data, size_t size)
{
	*iter =  SPA_JSON_INIT(data, size);
}
#define SPA_JSON_ENTER(iter) ((struct spa_json) { (iter)->cur, (iter)->end, (iter), (iter)->state & 0xff0, 0 })

static inline void spa_json_enter(struct spa_json * iter, struct spa_json * sub)
{
	*sub = SPA_JSON_ENTER(iter);
}

#define SPA_JSON_SAVE(iter) ((struct spa_json) { (iter)->cur, (iter)->end, NULL, (iter)->state, 0 })

/** Get the next token. \a value points to the token and the return value
 * is the length. Returns -1 on parse error, 0 on end of input. */
static inline int spa_json_next(struct spa_json * iter, const char **value)
{
	int utf8_remain = 0, err = 0;
	enum {
		__NONE, __STRUCT, __BARE, __STRING, __UTF8, __ESC, __COMMENT,
		__ARRAY_FLAG = 0x10,		/* in array context */
		__PREV_ARRAY_FLAG = 0x20,	/* depth=0 array context flag */
		__KEY_FLAG = 0x40,		/* inside object key */
		__SUB_FLAG = 0x80,		/* not at top-level */
		__FLAGS = 0xff0,
		__ERROR_SYSTEM = SPA_JSON_ERROR_FLAG,
		__ERROR_INVALID_ARRAY_SEPARATOR,
		__ERROR_EXPECTED_OBJECT_KEY,
		__ERROR_EXPECTED_OBJECT_VALUE,
		__ERROR_TOO_DEEP_NESTING,
		__ERROR_EXPECTED_ARRAY_CLOSE,
		__ERROR_EXPECTED_OBJECT_CLOSE,
		__ERROR_MISMATCHED_BRACKET,
		__ERROR_ESCAPE_NOT_ALLOWED,
		__ERROR_CHARACTERS_NOT_ALLOWED,
		__ERROR_INVALID_ESCAPE,
		__ERROR_INVALID_STATE,
		__ERROR_UNFINISHED_STRING,
	};
	uint64_t array_stack[8] = {0};		/* array context flags of depths 1...512 */

	*value = iter->cur;

	if (iter->state & SPA_JSON_ERROR_FLAG)
		return -1;

	for (; iter->cur < iter->end; iter->cur++) {
		unsigned char cur = (unsigned char)*iter->cur;
		uint32_t flag;

#define _SPA_ERROR(reason)	{ err = __ERROR_ ## reason; goto error; }
 again:
		flag = iter->state & __FLAGS;
		switch (iter->state & ~__FLAGS) {
		case __NONE:
			flag &= ~(__KEY_FLAG | __PREV_ARRAY_FLAG);
			iter->state = __STRUCT | flag;
			iter->depth = 0;
			goto again;
		case __STRUCT:
			switch (cur) {
			case '\0': case '\t': case ' ': case '\r': case '\n': case ',':
				continue;
			case ':': case '=':
				if (flag & __ARRAY_FLAG)
					_SPA_ERROR(INVALID_ARRAY_SEPARATOR);
				if (!(flag & __KEY_FLAG))
					_SPA_ERROR(EXPECTED_OBJECT_KEY);
				iter->state |= __SUB_FLAG;
				continue;
			case '#':
				iter->state = __COMMENT | flag;
				continue;
			case '"':
				if (flag & __KEY_FLAG)
					flag |= __SUB_FLAG;
				if (!(flag & __ARRAY_FLAG))
					SPA_FLAG_UPDATE(flag, __KEY_FLAG, !(flag & __KEY_FLAG));
				*value = iter->cur;
				iter->state = __STRING | flag;
				continue;
			case '[': case '{':
				if (!(flag & __ARRAY_FLAG)) {
					/* At top-level we may be either in object context
					 * or in single-item context, and then we need to
					 * accept array/object here.
					 */
					if ((iter->state & __SUB_FLAG) && !(flag & __KEY_FLAG))
						_SPA_ERROR(EXPECTED_OBJECT_KEY);
					SPA_FLAG_CLEAR(flag, __KEY_FLAG);
				}
				iter->state = __STRUCT | __SUB_FLAG | flag;
				SPA_FLAG_UPDATE(iter->state, __ARRAY_FLAG, cur == '[');

				/* We need to remember previous array state across calls
				 * for depth=0, so store that in state. Others bits go to
				 * temporary stack.
				 */
				if (iter->depth == 0) {
					SPA_FLAG_UPDATE(iter->state, __PREV_ARRAY_FLAG, flag & __ARRAY_FLAG);
				} else if (((iter->depth-1) >> 6) < SPA_N_ELEMENTS(array_stack)) {
					uint64_t mask = 1ULL << ((iter->depth-1) & 0x3f);
					SPA_FLAG_UPDATE(array_stack[(iter->depth-1) >> 6], mask, flag & __ARRAY_FLAG);
				} else {
					/* too deep */
					_SPA_ERROR(TOO_DEEP_NESTING);
				}

				*value = iter->cur;
				if (++iter->depth > 1)
					continue;
				iter->cur++;
				return 1;
			case '}': case ']':
				if ((flag & __ARRAY_FLAG) && cur != ']')
					_SPA_ERROR(EXPECTED_ARRAY_CLOSE);
				if (!(flag & __ARRAY_FLAG) && cur != '}')
					_SPA_ERROR(EXPECTED_OBJECT_CLOSE);
				if (flag & __KEY_FLAG) {
					/* incomplete key-value pair */
					_SPA_ERROR(EXPECTED_OBJECT_VALUE);
				}
				iter->state = __STRUCT | __SUB_FLAG | flag;
				if (iter->depth == 0) {
					if (iter->parent)
						iter->parent->cur = iter->cur;
					else
						_SPA_ERROR(MISMATCHED_BRACKET);
					return 0;
				}
				--iter->depth;
				if (iter->depth == 0) {
					SPA_FLAG_UPDATE(iter->state, __ARRAY_FLAG, flag & __PREV_ARRAY_FLAG);
				} else if (((iter->depth-1) >> 6) < SPA_N_ELEMENTS(array_stack)) {
					uint64_t mask = 1ULL << ((iter->depth-1) & 0x3f);
					SPA_FLAG_UPDATE(iter->state, __ARRAY_FLAG,
							SPA_FLAG_IS_SET(array_stack[(iter->depth-1) >> 6], mask));
				} else {
					/* too deep */
					_SPA_ERROR(TOO_DEEP_NESTING);
				}
				continue;
			case '\\':
				/* disallow bare escape */
				_SPA_ERROR(ESCAPE_NOT_ALLOWED);
			default:
				/* allow bare ascii */
				if (!(cur >= 32 && cur <= 126))
					_SPA_ERROR(CHARACTERS_NOT_ALLOWED);
				if (flag & __KEY_FLAG)
					flag |= __SUB_FLAG;
				if (!(flag & __ARRAY_FLAG))
					SPA_FLAG_UPDATE(flag, __KEY_FLAG, !(flag & __KEY_FLAG));
				*value = iter->cur;
				iter->state = __BARE | flag;
			}
			continue;
		case __BARE:
			switch (cur) {
			case '\0':
			case '\t': case ' ': case '\r': case '\n':
			case '"': case '#':
			case ':': case ',': case '=': case ']': case '}':
				iter->state = __STRUCT | flag;
				if (iter->depth > 0)
					goto again;
				return iter->cur - *value;
			case '\\':
				/* disallow bare escape */
				_SPA_ERROR(ESCAPE_NOT_ALLOWED);
			default:
				/* allow bare ascii */
				if (cur >= 32 && cur <= 126)
					continue;
			}
			_SPA_ERROR(CHARACTERS_NOT_ALLOWED);
		case __STRING:
			switch (cur) {
			case '\\':
				iter->state = __ESC | flag;
				continue;
			case '"':
				iter->state = __STRUCT | flag;
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
				iter->state = __UTF8 | flag;
				continue;
			default:
				if (cur >= 32 && cur <= 127)
					continue;
			}
			_SPA_ERROR(CHARACTERS_NOT_ALLOWED);
		case __UTF8:
			switch (cur) {
			case 128 ... 191:
				if (--utf8_remain == 0)
					iter->state = __STRING | flag;
				continue;
			}
			_SPA_ERROR(CHARACTERS_NOT_ALLOWED);
		case __ESC:
			switch (cur) {
			case '"': case '\\': case '/': case 'b': case 'f':
			case 'n': case 'r': case 't': case 'u':
				iter->state = __STRING | flag;
				continue;
			}
			_SPA_ERROR(INVALID_ESCAPE);
		case __COMMENT:
			switch (cur) {
			case '\n': case '\r':
				iter->state = __STRUCT | flag;
			}
			break;
		default:
			_SPA_ERROR(INVALID_STATE);
		}

	}
	if (iter->depth != 0 || iter->parent)
		_SPA_ERROR(MISMATCHED_BRACKET);

	switch (iter->state & ~__FLAGS) {
	case __STRING: case __UTF8: case __ESC:
		/* string/escape not closed */
		_SPA_ERROR(UNFINISHED_STRING);
	case __COMMENT:
		/* trailing comment */
		return 0;
	}

	if ((iter->state & __SUB_FLAG) && (iter->state & __KEY_FLAG)) {
		/* incomplete key-value pair */
		_SPA_ERROR(EXPECTED_OBJECT_VALUE);
	}

	if ((iter->state & ~__FLAGS) != __STRUCT) {
		iter->state = __STRUCT | (iter->state & __FLAGS);
		return iter->cur - *value;
	}
	return 0;
#undef _SPA_ERROR

error:
	iter->state = err;
	while (iter->parent) {
		if (iter->parent->state & SPA_JSON_ERROR_FLAG)
			break;
		iter->parent->state = err;
		iter->parent->cur = iter->cur;
		iter = iter->parent;
	}
	return -1;
}

/**
 * Return it there was a parse error, and its possible location.
 *
 * \since 1.1.0
 */
static inline bool spa_json_get_error(struct spa_json *iter, const char *start,
		struct spa_error_location *loc)
{
	static const char *reasons[] = {
		"System error",
		"Invalid array separator",
		"Expected object key",
		"Expected object value",
		"Too deep nesting",
		"Expected array close bracket",
		"Expected object close brace",
		"Mismatched bracket",
		"Escape not allowed",
		"Character not allowed",
		"Invalid escape",
		"Invalid state",
		"Unfinished string",
		"Expected key separator",
	};

	if (!(iter->state & SPA_JSON_ERROR_FLAG))
		return false;

	if (loc) {
		int linepos = 1, colpos = 1, code;
		const char *p, *l;

		for (l = p = start; p && p != iter->cur; ++p) {
			if (*p == '\n') {
				linepos++;
				colpos = 1;
				l = p+1;
			} else {
				colpos++;
			}
		}
		code = SPA_CLAMP(iter->state & 0xff, 0u, SPA_N_ELEMENTS(reasons)-1);
		loc->line = linepos;
		loc->col = colpos;
		loc->location = l;
		loc->len = SPA_PTRDIFF(iter->end, loc->location) / sizeof(char);
		loc->reason = code == 0 ? strerror(errno) : reasons[code];
	}
	return true;
}

static inline int spa_json_enter_container(struct spa_json *iter, struct spa_json *sub, char type)
{
	const char *value;
	int len;
	if ((len = spa_json_next(iter, &value)) <= 0)
		return len;
	if (*value != type)
		return -1;
	spa_json_enter(iter, sub);
	return 1;
}

static inline int spa_json_is_container(const char *val, int len)
{
	return len > 0 && (*val == '{'  || *val == '[');
}

/**
 * Return length of container at current position, starting at \a value.
 *
 * \return Length of container including {} or [], or 0 on error.
 */
static inline int spa_json_container_len(struct spa_json *iter, const char *value, int len SPA_UNUSED)
{
	const char *val;
	struct spa_json sub;
	int res;
	spa_json_enter(iter, &sub);
	while ((res = spa_json_next(&sub, &val)) > 0);
	if (res < 0)
		return 0;
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
	char buf[96];
	char *end;
	int pos;

	if (len <= 0 || len >= (int)sizeof(buf))
		return 0;

	for (pos = 0; pos < len; ++pos) {
		switch (val[pos]) {
		case '+': case '-': case '0' ... '9': case '.': case 'e': case 'E': break;
		default: return 0;
		}
	}

	memcpy(buf, val, len);
	buf[len] = '\0';

	*result = spa_strtof(buf, &end);
	return len > 0 && end == buf + len;
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
		return len;
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
	char buf[64];
	char *end;

	if (len <= 0 || len >= (int)sizeof(buf))
		return 0;

	memcpy(buf, val, len);
	buf[len] = '\0';

	*result = strtol(buf, &end, 0);
	return len > 0 && end == buf + len;
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
		return len;
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
		return len;
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
			memmove(result, val, len);
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
						cp = 0x010000 + (((cp & 0x3ff) << 10) | (v & 0x3ff));
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
		return len;
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
