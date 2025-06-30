#ifndef PLUTOVG_UTILS_H
#define PLUTOVG_UTILS_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <float.h>
#include <math.h>

#define PLUTOVG_IS_NUM(c) ((c) >= '0' && (c) <= '9')
#define PLUTOVG_IS_ALPHA(c) (((c) >= 'a' && (c) <= 'z') || ((c) >= 'A' && (c) <= 'Z'))
#define PLUTOVG_IS_ALNUM(c) (PLUTOVG_IS_ALPHA(c) || PLUTOVG_IS_NUM(c))
#define PLUTOVG_IS_WS(c) ((c) == ' ' || (c) == '\t' || (c) == '\n' || (c) == '\r')

#define plutovg_min(a, b) ((a) < (b) ? (a) : (b))
#define plutovg_max(a, b) ((a) > (b) ? (a) : (b))
#define plutovg_clamp(v, lo, hi) ((v) < (lo) ? (lo) : (hi) < (v) ? (hi) : (v))

#define plutovg_alpha(c) (((c) >> 24) & 0xff)
#define plutovg_red(c) (((c) >> 16) & 0xff)
#define plutovg_green(c) (((c) >> 8) & 0xff)
#define plutovg_blue(c) (((c) >> 0) & 0xff)

#define plutovg_array_init(array) \
    do { \
        (array).data = NULL; \
        (array).size = 0; \
        (array).capacity = 0; \
    } while(0)

#define plutovg_array_ensure(array, count) \
    do { \
        if((array).size + (count) > (array).capacity) { \
            int capacity = (array).size + (count); \
            int newcapacity = (array).capacity == 0 ? 8 : (array).capacity; \
            while(newcapacity < capacity) { newcapacity *= 2; } \
            (array).data = realloc((array).data, newcapacity * sizeof((array).data[0])); \
            (array).capacity = newcapacity; \
        } \
    } while(0)

#define plutovg_array_append_data(array, newdata, count) \
    do { \
        if(newdata && count > 0) { \
            plutovg_array_ensure(array, count); \
            memcpy((array).data + (array).size, newdata, (count) * sizeof((newdata)[0])); \
            (array).size += count; \
        } \
    } while(0)

#define plutovg_array_append(array, other) plutovg_array_append_data(array, (other).data, (other).size)
#define plutovg_array_clear(array) ((array).size = 0)
#define plutovg_array_destroy(array) free((array).data)

static inline uint32_t plutovg_div255(uint32_t x)
{
    return (((x) + ((x) >> 8) + 0x80) >> 8);
}

static inline uint32_t plutovg_premultiply_argb(uint32_t color)
{
    uint32_t a = plutovg_alpha(color);
    uint32_t r = plutovg_red(color);
    uint32_t g = plutovg_green(color);
    uint32_t b = plutovg_blue(color);
    if(a != 255) {
        r = plutovg_div255(r * a);
        g = plutovg_div255(g * a);
        b = plutovg_div255(b * a);
    }

    return (a << 24) | (r << 16) | (g << 8) | (b);
}

static inline bool plutovg_parse_number(const char** begin, const char* end, float* number)
{
    const char* it = *begin;
    float fraction = 0;
    float integer = 0;
    float exponent = 0;
    int sign = 1;
    int expsign = 1;

    if(it < end && *it == '+') {
        ++it;
    } else if(it < end && *it == '-') {
        ++it;
        sign = -1;
    }

    if(it >= end || (*it != '.' && !PLUTOVG_IS_NUM(*it)))
        return false;
    if(PLUTOVG_IS_NUM(*it)) {
        do {
            integer = 10.f * integer + (*it++ - '0');
        } while(it < end && PLUTOVG_IS_NUM(*it));
    }

    if(it < end && *it == '.') {
        ++it;
        if(it >= end || !PLUTOVG_IS_NUM(*it))
            return false;
        float divisor = 1.f;
        do {
            fraction = 10.f * fraction + (*it++ - '0');
            divisor *= 10.f;
        } while(it < end && PLUTOVG_IS_NUM(*it));
        fraction /= divisor;
    }

    if(it < end && (*it == 'e' || *it == 'E')) {
        ++it;
        if(it < end && *it == '+') {
            ++it;
        } else if(it < end && *it == '-') {
            ++it;
            expsign = -1;
        }

        if(it >= end || !PLUTOVG_IS_NUM(*it))
            return false;
        do {
            exponent = 10 * exponent + (*it++ - '0');
        } while(it < end && PLUTOVG_IS_NUM(*it));
    }

    *begin = it;
    *number = sign * (integer + fraction);
    if(exponent)
        *number *= powf(10.f, expsign * exponent);
    return *number >= -FLT_MAX && *number <= FLT_MAX;
}

static inline bool plutovg_skip_delim(const char** begin, const char* end, const char delim)
{
    const char* it = *begin;
    if(it < end && *it == delim) {
        *begin = it + 1;
        return true;
    }

    return false;
}

static inline bool plutovg_skip_string(const char** begin, const char* end, const char* data)
{
    const char* it = *begin;
    while(it < end && *data && *it == *data) {
        ++data;
        ++it;
    }

    if(*data == '\0') {
        *begin = it;
        return true;
    }

    return false;
}

static inline bool plutovg_skip_ws(const char** begin, const char* end)
{
    const char* it = *begin;
    while(it < end && PLUTOVG_IS_WS(*it))
        ++it;
    *begin = it;
    return it < end;
}

static inline bool plutovg_skip_ws_or_delim(const char** begin, const char* end, char delim)
{
    const char* it = *begin;
    if(it < end && *it != delim && !PLUTOVG_IS_WS(*it))
        return false;
    if(plutovg_skip_ws(&it, end)) {
        if(plutovg_skip_delim(&it, end, delim)) {
            plutovg_skip_ws(&it, end);
        }
    }

    *begin = it;
    return it < end;
}

static inline bool plutovg_skip_ws_or_comma(const char** begin, const char* end)
{
    return plutovg_skip_ws_or_delim(begin, end, ',');
}

static inline bool plutovg_skip_ws_and_delim(const char** begin, const char* end, char delim)
{
    const char* it = *begin;
    if(plutovg_skip_ws(&it, end)) {
        if(!plutovg_skip_delim(&it, end, delim))
            return false;
        plutovg_skip_ws(&it, end);
    }

    *begin = it;
    return it < end;
}

static inline bool plutovg_skip_ws_and_comma(const char** begin, const char* end)
{
    return plutovg_skip_ws_and_delim(begin, end, ',');
}

#endif // PLUTOVG_UTILS_H
