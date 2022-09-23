/*************************************************************************/
/*  pool_allocator.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "pool_allocator.h"

#include "core/error_macros.h"
#include "core/os/memory.h"
#include "core/os/os.h"
#include "core/print_string.h"

#define COMPACT_CHUNK(m_entry, m_to_pos)                      \
	do {                                                      \
		void *_dst = &((unsigned char *)pool)[m_to_pos];      \
		void *_src = &((unsigned char *)pool)[(m_entry).pos]; \
		memmove(_dst, _src, aligned((m_entry).len));          \
		(m_entry).pos = m_to_pos;                             \
	} while (0);

void PoolAllocator::mt_lock() const {
}

void PoolAllocator::mt_unlock() const {
}

bool PoolAllocator::get_free_entry(EntryArrayPos *p_pos) {
	if (entry_count == entry_max) {
		return false;
	}

	for (int i = 0; i < entry_max; i++) {
		if (entry_array[i].len == 0) {
			*p_pos = i;
			return true;
		}
	}

	ERR_PRINT("Out of memory Chunks!");

	return false; //
}

/**
 * Find a hole
 * @param p_pos The hole is behind the block pointed by this variable upon return. if pos==entry_count, then allocate at end
 * @param p_for_size hole size
 * @return false if hole found, true if no hole found
 */
bool PoolAllocator::find_hole(EntryArrayPos *p_pos, int p_for_size) {
	/* position where previous entry ends. Defaults to zero (begin of pool) */

	int prev_entry_end_pos = 0;

	for (int i = 0; i < entry_count; i++) {
		Entry &entry = entry_array[entry_indices[i]];

		/* determine hole size to previous entry */

		int hole_size = entry.pos - prev_entry_end_pos;

		/* determine if what we want fits in that hole */
		if (hole_size >= p_for_size) {
			*p_pos = i;
			return true;
		}

		/* prepare for next one */
		prev_entry_end_pos = entry_end(entry);
	}

	/* No holes between entries, check at the end..*/

	if ((pool_size - prev_entry_end_pos) >= p_for_size) {
		*p_pos = entry_count;
		return true;
	}

	return false;
}

void PoolAllocator::compact(int p_up_to) {
	uint32_t prev_entry_end_pos = 0;

	if (p_up_to < 0) {
		p_up_to = entry_count;
	}
	for (int i = 0; i < p_up_to; i++) {
		Entry &entry = entry_array[entry_indices[i]];

		/* determine hole size to previous entry */

		int hole_size = entry.pos - prev_entry_end_pos;

		/* if we can compact, do it */
		if (hole_size > 0 && !entry.lock) {
			COMPACT_CHUNK(entry, prev_entry_end_pos);
		}

		/* prepare for next one */
		prev_entry_end_pos = entry_end(entry);
	}
}

void PoolAllocator::compact_up(int p_from) {
	uint32_t next_entry_end_pos = pool_size; // - static_area_size;

	for (int i = entry_count - 1; i >= p_from; i--) {
		Entry &entry = entry_array[entry_indices[i]];

		/* determine hole size to nextious entry */

		int hole_size = next_entry_end_pos - (entry.pos + aligned(entry.len));

		/* if we can compact, do it */
		if (hole_size > 0 && !entry.lock) {
			COMPACT_CHUNK(entry, (next_entry_end_pos - aligned(entry.len)));
		}

		/* prepare for next one */
		next_entry_end_pos = entry.pos;
	}
}

bool PoolAllocator::find_entry_index(EntryIndicesPos *p_map_pos, Entry *p_entry) {
	EntryArrayPos entry_pos = entry_max;

	for (int i = 0; i < entry_count; i++) {
		if (&entry_array[entry_indices[i]] == p_entry) {
			entry_pos = i;
			break;
		}
	}

	if (entry_pos == entry_max) {
		return false;
	}

	*p_map_pos = entry_pos;
	return true;
}

PoolAllocator::ID PoolAllocator::alloc(int p_size) {
	ERR_FAIL_COND_V(p_size < 1, POOL_ALLOCATOR_INVALID_ID);
	ERR_FAIL_COND_V(p_size > free_mem, POOL_ALLOCATOR_INVALID_ID);

	mt_lock();

	if (entry_count == entry_max) {
		mt_unlock();
		ERR_PRINT("entry_count==entry_max");
		return POOL_ALLOCATOR_INVALID_ID;
	}

	int size_to_alloc = aligned(p_size);

	EntryIndicesPos new_entry_indices_pos;

	if (!find_hole(&new_entry_indices_pos, size_to_alloc)) {
		/* No hole could be found, try compacting mem */
		compact();
		/* Then search again */

		if (!find_hole(&new_entry_indices_pos, size_to_alloc)) {
			mt_unlock();
			ERR_FAIL_V_MSG(POOL_ALLOCATOR_INVALID_ID, "Memory can't be compacted further.");
		}
	}

	EntryArrayPos new_entry_array_pos;

	bool found_free_entry = get_free_entry(&new_entry_array_pos);

	if (!found_free_entry) {
		mt_unlock();
		ERR_FAIL_V_MSG(POOL_ALLOCATOR_INVALID_ID, "No free entry found in PoolAllocator.");
	}

	/* move all entry indices up, make room for this one */
	for (int i = entry_count; i > new_entry_indices_pos; i--) {
		entry_indices[i] = entry_indices[i - 1];
	}

	entry_indices[new_entry_indices_pos] = new_entry_array_pos;

	entry_count++;

	Entry &entry = entry_array[entry_indices[new_entry_indices_pos]];

	entry.len = p_size;
	entry.pos = (new_entry_indices_pos == 0) ? 0 : entry_end(entry_array[entry_indices[new_entry_indices_pos - 1]]); //alloc either at beginning or end of previous
	entry.lock = 0;
	entry.check = (check_count++) & CHECK_MASK;
	free_mem -= size_to_alloc;
	if (free_mem < free_mem_peak) {
		free_mem_peak = free_mem;
	}

	ID retval = (entry_indices[new_entry_indices_pos] << CHECK_BITS) | entry.check;
	mt_unlock();

	//ERR_FAIL_COND_V( (uintptr_t)get(retval)%align != 0, retval );

	return retval;
}

PoolAllocator::Entry *PoolAllocator::get_entry(ID p_mem) {
	unsigned int check = p_mem & CHECK_MASK;
	int entry = p_mem >> CHECK_BITS;
	ERR_FAIL_INDEX_V(entry, entry_max, nullptr);
	ERR_FAIL_COND_V(entry_array[entry].check != check, nullptr);
	ERR_FAIL_COND_V(entry_array[entry].len == 0, nullptr);

	return &entry_array[entry];
}

const PoolAllocator::Entry *PoolAllocator::get_entry(ID p_mem) const {
	unsigned int check = p_mem & CHECK_MASK;
	int entry = p_mem >> CHECK_BITS;
	ERR_FAIL_INDEX_V(entry, entry_max, nullptr);
	ERR_FAIL_COND_V(entry_array[entry].check != check, nullptr);
	ERR_FAIL_COND_V(entry_array[entry].len == 0, nullptr);

	return &entry_array[entry];
}

void PoolAllocator::free(ID p_mem) {
	mt_lock();
	Entry *e = get_entry(p_mem);
	if (!e) {
		mt_unlock();
		ERR_PRINT("!e");
		return;
	}
	if (e->lock) {
		mt_unlock();
		ERR_PRINT("e->lock");
		return;
	}

	EntryIndicesPos entry_indices_pos;

	bool index_found = find_entry_index(&entry_indices_pos, e);
	if (!index_found) {
		mt_unlock();
		ERR_FAIL_COND(!index_found);
	}

	for (int i = entry_indices_pos; i < (entry_count - 1); i++) {
		entry_indices[i] = entry_indices[i + 1];
	}

	entry_count--;
	free_mem += aligned(e->len);
	e->clear();
	mt_unlock();
}

int PoolAllocator::get_size(ID p_mem) const {
	int size;
	mt_lock();

	const Entry *e = get_entry(p_mem);
	if (!e) {
		mt_unlock();
		ERR_PRINT("!e");
		return 0;
	}

	size = e->len;

	mt_unlock();

	return size;
}

Error PoolAllocator::resize(ID p_mem, int p_new_size) {
	mt_lock();
	Entry *e = get_entry(p_mem);

	if (!e) {
		mt_unlock();
		ERR_FAIL_COND_V(!e, ERR_INVALID_PARAMETER);
	}

	if (needs_locking && e->lock) {
		mt_unlock();
		ERR_FAIL_COND_V(e->lock, ERR_ALREADY_IN_USE);
	}

	uint32_t alloc_size = aligned(p_new_size);

	if ((uint32_t)aligned(e->len) == alloc_size) {
		e->len = p_new_size;
		mt_unlock();
		return OK;
	} else if (e->len > (uint32_t)p_new_size) {
		free_mem += aligned(e->len);
		free_mem -= alloc_size;
		e->len = p_new_size;
		mt_unlock();
		return OK;
	}

	//p_new_size = align(p_new_size)
	int _free = free_mem; // - static_area_size;

	if (uint32_t(_free + aligned(e->len)) < alloc_size) {
		mt_unlock();
		ERR_FAIL_V(ERR_OUT_OF_MEMORY);
	};

	EntryIndicesPos entry_indices_pos;

	bool index_found = find_entry_index(&entry_indices_pos, e);

	if (!index_found) {
		mt_unlock();
		ERR_FAIL_COND_V(!index_found, ERR_BUG);
	}

	//no need to move stuff around, it fits before the next block
	uint32_t next_pos;
	if (entry_indices_pos + 1 == entry_count) {
		next_pos = pool_size; // - static_area_size;
	} else {
		next_pos = entry_array[entry_indices[entry_indices_pos + 1]].pos;
	};

	if ((next_pos - e->pos) > alloc_size) {
		free_mem += aligned(e->len);
		e->len = p_new_size;
		free_mem -= alloc_size;
		mt_unlock();
		return OK;
	}
	//it doesn't fit, compact around BEFORE current index (make room behind)

	compact(entry_indices_pos + 1);

	if ((next_pos - e->pos) > alloc_size) {
		//now fits! hooray!
		free_mem += aligned(e->len);
		e->len = p_new_size;
		free_mem -= alloc_size;
		mt_unlock();
		if (free_mem < free_mem_peak) {
			free_mem_peak = free_mem;
		}
		return OK;
	}

	//STILL doesn't fit, compact around AFTER current index (make room after)

	compact_up(entry_indices_pos + 1);

	if ((entry_array[entry_indices[entry_indices_pos + 1]].pos - e->pos) > alloc_size) {
		//now fits! hooray!
		free_mem += aligned(e->len);
		e->len = p_new_size;
		free_mem -= alloc_size;
		mt_unlock();
		if (free_mem < free_mem_peak) {
			free_mem_peak = free_mem;
		}
		return OK;
	}

	mt_unlock();
	ERR_FAIL_V(ERR_OUT_OF_MEMORY);
}

Error PoolAllocator::lock(ID p_mem) {
	if (!needs_locking) {
		return OK;
	}
	mt_lock();
	Entry *e = get_entry(p_mem);
	if (!e) {
		mt_unlock();
		ERR_PRINT("!e");
		return ERR_INVALID_PARAMETER;
	}
	e->lock++;
	mt_unlock();
	return OK;
}

bool PoolAllocator::is_locked(ID p_mem) const {
	if (!needs_locking) {
		return false;
	}

	mt_lock();
	const Entry *e = ((PoolAllocator *)(this))->get_entry(p_mem);
	if (!e) {
		mt_unlock();
		ERR_PRINT("!e");
		return false;
	}
	bool locked = e->lock;
	mt_unlock();
	return locked;
}

const void *PoolAllocator::get(ID p_mem) const {
	if (!needs_locking) {
		const Entry *e = get_entry(p_mem);
		ERR_FAIL_COND_V(!e, nullptr);
		return &pool[e->pos];
	}

	mt_lock();
	const Entry *e = get_entry(p_mem);

	if (!e) {
		mt_unlock();
		ERR_FAIL_COND_V(!e, nullptr);
	}
	if (e->lock == 0) {
		mt_unlock();
		ERR_PRINT("e->lock == 0");
		return nullptr;
	}

	if ((int)e->pos >= pool_size) {
		mt_unlock();
		ERR_PRINT("e->pos<0 || e->pos>=pool_size");
		return nullptr;
	}
	const void *ptr = &pool[e->pos];

	mt_unlock();

	return ptr;
}

void *PoolAllocator::get(ID p_mem) {
	if (!needs_locking) {
		Entry *e = get_entry(p_mem);
		ERR_FAIL_COND_V(!e, nullptr);
		return &pool[e->pos];
	}

	mt_lock();
	Entry *e = get_entry(p_mem);

	if (!e) {
		mt_unlock();
		ERR_FAIL_COND_V(!e, nullptr);
	}
	if (e->lock == 0) {
		mt_unlock();
		ERR_PRINT("e->lock == 0");
		return nullptr;
	}

	if ((int)e->pos >= pool_size) {
		mt_unlock();
		ERR_PRINT("e->pos<0 || e->pos>=pool_size");
		return nullptr;
	}
	void *ptr = &pool[e->pos];

	mt_unlock();

	return ptr;
}
void PoolAllocator::unlock(ID p_mem) {
	if (!needs_locking) {
		return;
	}
	mt_lock();
	Entry *e = get_entry(p_mem);
	if (!e) {
		mt_unlock();
		ERR_FAIL_COND(!e);
	}
	if (e->lock == 0) {
		mt_unlock();
		ERR_PRINT("e->lock == 0");
		return;
	}
	e->lock--;
	mt_unlock();
}

int PoolAllocator::get_used_mem() const {
	return pool_size - free_mem;
}

int PoolAllocator::get_free_peak() {
	return free_mem_peak;
}

int PoolAllocator::get_free_mem() {
	return free_mem;
}

void PoolAllocator::create_pool(void *p_mem, int p_size, int p_max_entries) {
	pool = (uint8_t *)p_mem;
	pool_size = p_size;

	entry_array = memnew_arr(Entry, p_max_entries);
	entry_indices = memnew_arr(int, p_max_entries);
	entry_max = p_max_entries;
	entry_count = 0;

	free_mem = p_size;
	free_mem_peak = p_size;

	check_count = 0;
}

PoolAllocator::PoolAllocator(int p_size, bool p_needs_locking, int p_max_entries) {
	mem_ptr = memalloc(p_size);
	ERR_FAIL_COND(!mem_ptr);
	align = 1;
	create_pool(mem_ptr, p_size, p_max_entries);
	needs_locking = p_needs_locking;
}

PoolAllocator::PoolAllocator(void *p_mem, int p_size, int p_align, bool p_needs_locking, int p_max_entries) {
	if (p_align > 1) {
		uint8_t *mem8 = (uint8_t *)p_mem;
		uint64_t ofs = (uint64_t)mem8;
		if (ofs % p_align) {
			int dif = p_align - (ofs % p_align);
			mem8 += p_align - (ofs % p_align);
			p_size -= dif;
			p_mem = (void *)mem8;
		};
	};

	create_pool(p_mem, p_size, p_max_entries);
	needs_locking = p_needs_locking;
	align = p_align;
	mem_ptr = nullptr;
}

PoolAllocator::PoolAllocator(int p_align, int p_size, bool p_needs_locking, int p_max_entries) {
	ERR_FAIL_COND(p_align < 1);
	mem_ptr = Memory::alloc_static(p_size + p_align, true);
	uint8_t *mem8 = (uint8_t *)mem_ptr;
	uint64_t ofs = (uint64_t)mem8;
	if (ofs % p_align) {
		mem8 += p_align - (ofs % p_align);
	}
	create_pool(mem8, p_size, p_max_entries);
	needs_locking = p_needs_locking;
	align = p_align;
}

PoolAllocator::~PoolAllocator() {
	if (mem_ptr) {
		memfree(mem_ptr);
	}

	memdelete_arr(entry_array);
	memdelete_arr(entry_indices);
}
