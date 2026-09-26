/* zmemory.h -- Private inline functions used internally in zlib-ng
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#ifndef _ZMEMORY_H
#define _ZMEMORY_H

#if defined(__GNUC__) && (__GNUC__ >= 4)
#  define HAVE_MAY_ALIAS
#endif

static inline uint16_t zng_memread_2(const void *ptr) {
#if defined(HAVE_MAY_ALIAS)
    typedef struct { uint16_t val; } __attribute__ ((__packed__, __may_alias__)) unaligned_uint16_t;
    return ((const unaligned_uint16_t *)ptr)->val;
#else
    uint16_t val;
    memcpy(&val, ptr, sizeof(val));
    return val;
#endif
}

static inline uint32_t zng_memread_4(const void *ptr) {
#if defined(HAVE_MAY_ALIAS)
    typedef struct { uint32_t val; } __attribute__ ((__packed__, __may_alias__)) unaligned_uint32_t;
    return ((const unaligned_uint32_t *)ptr)->val;
#else
    uint32_t val;
    memcpy(&val, ptr, sizeof(val));
    return val;
#endif
}

static inline uint64_t zng_memread_8(const void *ptr) {
#if defined(HAVE_MAY_ALIAS)
    typedef struct { uint64_t val; } __attribute__ ((__packed__, __may_alias__)) unaligned_uint64_t;
    return ((const unaligned_uint64_t *)ptr)->val;
#else
    uint64_t val;
    memcpy(&val, ptr, sizeof(val));
    return val;
#endif
}

static inline void zng_memwrite_2(void *ptr, uint16_t val) {
#if defined(HAVE_MAY_ALIAS)
    typedef struct { uint16_t val; } __attribute__ ((__packed__, __may_alias__)) unaligned_uint16_t;
    ((unaligned_uint16_t *)ptr)->val = val;
#else
    memcpy(ptr, &val, sizeof(val));
#endif
}

static inline void zng_memwrite_4(void *ptr, uint32_t val) {
#if defined(HAVE_MAY_ALIAS)
    typedef struct { uint32_t val; } __attribute__ ((__packed__, __may_alias__)) unaligned_uint32_t;
    ((unaligned_uint32_t *)ptr)->val = val;
#else
    memcpy(ptr, &val, sizeof(val));
#endif
}

static inline void zng_memwrite_8(void *ptr, uint64_t val) {
#if defined(HAVE_MAY_ALIAS)
    typedef struct { uint64_t val; } __attribute__ ((__packed__, __may_alias__)) unaligned_uint64_t;
    ((unaligned_uint64_t *)ptr)->val = val;
#else
    memcpy(ptr, &val, sizeof(val));
#endif
}

/* Use zng_memread_* instead of memcmp to avoid older compilers not converting memcmp
   calls to unaligned comparisons when unaligned access is supported. Use memcmp only when
   unaligned support is not available to avoid an extra call to memcpy. */
static inline int32_t zng_memcmp_2(const void *src0, const void *src1) {
#if defined(HAVE_MAY_ALIAS)
    return zng_memread_2(src0) != zng_memread_2(src1);
#else
    return memcmp(src0, src1, 2);
#endif
}

static inline int32_t zng_memcmp_4(const void *src0, const void *src1) {
#if defined(HAVE_MAY_ALIAS)
    return zng_memread_4(src0) != zng_memread_4(src1);
#else
    return memcmp(src0, src1, 4);
#endif
}

static inline int32_t zng_memcmp_8(const void *src0, const void *src1) {
#if defined(HAVE_MAY_ALIAS)
    return zng_memread_8(src0) != zng_memread_8(src1);
#else
    return memcmp(src0, src1, 8);
#endif
}

#endif
