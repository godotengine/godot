/**************************************************************************/
/*  chunked_sparse_array.h                                                */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "core/templates/paged_allocator.h"
#include "core/typedefs.h"

#ifdef _MSC_VER
#include <intrin.h>
#pragma intrinsic(_BitScanForward64)

_FORCE_INLINE_ int ctzll(uint64_t value) {
	unsigned long index;
	if (_BitScanForward64(&index, value)) {
		return static_cast<int>(index);
	}
	return 64; // Should never happen if value != 0
}
#else
_FORCE_INLINE_ int ctzll(uint64_t value) {
	return __builtin_ctzll(value);
}
#endif

template <typename T>
class ChunkedSparseArray {
public:
	static constexpr int MAX_CHUNKS = 128;

	uint64_t bitmask_low = 0;
	uint64_t bitmask_high = 0;

	struct Chunk {
		uint64_t used_mask = 0;
		T entries[64];

		_FORCE_INLINE_ bool is_entry_used(uint64_t offset) const {
			return used_mask & (1ULL << offset);
		}

		_FORCE_INLINE_ void mark_entry_used(uint64_t offset) {
			used_mask |= (1ULL << offset);
		}

		_FORCE_INLINE_ void mark_entry_free(uint64_t offset) {
			used_mask &= ~(1ULL << offset);
		}
	};

	PagedAllocator<Chunk> &allocator;
	Chunk *chunks[MAX_CHUNKS];

	_FORCE_INLINE_ bool is_chunk_used(uint64_t chunk_id) const {
		if (chunk_id < 64) {
			return bitmask_low & (1ULL << chunk_id);
		} else if (chunk_id < MAX_CHUNKS) {
			return bitmask_high & (1ULL << (chunk_id - 64));
		}
		return false;
	}

	_FORCE_INLINE_ void mark_chunk_used(uint64_t chunk_id) {
		if (chunk_id < 64) {
			bitmask_low |= (1ULL << chunk_id);
		} else if (chunk_id < MAX_CHUNKS) {
			bitmask_high |= (1ULL << (chunk_id - 64));
		}
	}

	_FORCE_INLINE_ void mark_chunk_free(uint64_t chunk_id) {
		if (chunk_id < 64) {
			bitmask_low &= ~(1ULL << chunk_id);
		} else if (chunk_id < MAX_CHUNKS) {
			bitmask_high &= ~(1ULL << (chunk_id - 64));
		}
	}

	_FORCE_INLINE_ void free_chunk(uint64_t zidx) {
		uint64_t chunk_id = zidx >> 6;
		if (chunk_id >= MAX_CHUNKS) {
			return;
		}
		Chunk *chunk = chunks[chunk_id];
		if (chunk) {
			chunk->used_mask = 0;
			allocator.free(chunk);
			chunks[chunk_id] = nullptr;
			mark_chunk_free(chunk_id);
		}
	}

	_FORCE_INLINE_ Chunk *get_chunk(uint64_t zidx) {
		uint64_t chunk_id = zidx >> 6;
		if (chunk_id >= MAX_CHUNKS) {
			return nullptr;
		}
		if (!is_chunk_used(chunk_id)) {
			chunks[chunk_id] = allocator.alloc();
			if (!chunks[chunk_id]) {
				return nullptr;
			}
			chunks[chunk_id]->used_mask = 0;
			mark_chunk_used(chunk_id);
		}
		return chunks[chunk_id];
	}

	_FORCE_INLINE_ Chunk *get_chunk_unsafe(uint64_t zidx) {
		uint64_t chunk_id = zidx >> 6;
		if (chunk_id >= MAX_CHUNKS) {
			return nullptr;
		}
		return chunks[chunk_id];
	}

	_FORCE_INLINE_ T *get_entry(uint64_t zidx) {
		uint64_t chunk_id = zidx >> 6;
		if (chunk_id >= MAX_CHUNKS) {
			return nullptr;
		}
		if (!is_chunk_used(chunk_id)) {
			chunks[chunk_id] = allocator.alloc();
			if (!chunks[chunk_id]) {
				return nullptr;
			}
			chunks[chunk_id]->used_mask = 0;
			mark_chunk_used(chunk_id);
		}
		Chunk *chunk = chunks[chunk_id];
		uint64_t offset = zidx & 63;
		if (!chunk->is_entry_used(offset)) {
			T *entry = &chunk->entries[offset];
			entry->reset();
			chunk->mark_entry_used(offset);
			return entry;
		}
		return &chunk->entries[offset];
	}

	_FORCE_INLINE_ T *get_entry_unsafe(uint64_t zidx) {
		uint64_t chunk_id = zidx >> 6;
		if (chunk_id >= MAX_CHUNKS) {
			return nullptr;
		}
		return &chunks[chunk_id]->entries[(zidx & 63)];
	}

	_FORCE_INLINE_ void clear() {
		uint64_t mask = bitmask_low;
		while (mask) {
			int chunk_id = ctzll(mask);
			if (chunk_id >= 64) {
				break;
			}

			mask &= mask - 1;
			Chunk *chunk = chunks[chunk_id];
			if (chunk) {
				chunk->used_mask = 0;
				allocator.free(chunk);
				chunks[chunk_id] = nullptr;
			}
		}
		bitmask_low = 0;

		mask = bitmask_high;
		while (mask) {
			int chunk_id = ctzll(mask) + 64;
			if (chunk_id >= MAX_CHUNKS) {
				break;
			}

			mask &= mask - 1;
			Chunk *chunk = chunks[chunk_id];
			if (chunk) {
				chunk->used_mask = 0;
				allocator.free(chunk);
				chunks[chunk_id] = nullptr;
			}
		}
		bitmask_high = 0;
	}

	_FORCE_INLINE_ ChunkedSparseArray(PagedAllocator<Chunk> &p_allocator) : allocator(p_allocator) {}

	_FORCE_INLINE_ ~ChunkedSparseArray() {
		clear();
	}
};
