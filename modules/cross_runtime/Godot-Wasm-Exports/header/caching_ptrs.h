// caching_ptrs.h
// cache mapping for use by signal infrastructure.
// since .NET expects only identifies ptrs, this is used to make sure we can extract a ptr of an id and send the ptr instead which .NET will recognize
#ifndef CACHING_PTRS_H
#define CACHING_PTRS_H

#include "core/templates/hash_map.h"
#include <cstdint>

extern HashMap<uint64_t, uintptr_t> ptr_cache;   // defined once, somewhere

inline void ptr_caching(uint64_t _id_, uintptr_t _ptr_) {
    ptr_cache[_id_] = _ptr_;
}

inline uintptr_t extract_ptr(uint64_t _id_) {
    return ptr_cache[_id_];
}

#endif