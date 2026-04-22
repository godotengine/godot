/**************************************************************************/
/*  a_hash_map.h                                                          */
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

#include "core/os/memory.h"
#include "core/string/print_string.h" // IWYU pragma: keep. `WARN_VERBOSE` macro.
#include "core/templates/hashfuncs.h"
#include "core/templates/pair.h"
#include "core/templates/swiss_table_simd.h"

#include <initializer_list>

class String;
class StringName;
class Variant;

/**
 * An array-based implementation of a hash map. Very efficient in performance
 * and memory usage. Uses a SwissTable-style control-byte index (SIMD-scanned)
 * pointing into a contiguous packed entries array. Insertion order is NOT
 * preserved across erases: erase swaps the removed element with the last one
 * in the entries array (constant-time erase, dense storage).
 *
 * Element pointers and indices are NOT stable across erases that move the
 * tail. Pointers/iterators may also be invalidated by inserts that grow the
 * table. If you need pointer stability or insertion-order preservation across
 * erases, use HashMap instead.
 *
 * Public API mirrors the older Robin-Hood implementation:
 *   - get / getptr / has / find
 *   - insert / insert_new / operator[]
 *   - erase / erase_by_index / replace_key
 *   - get_index (key -> entry index) / get_by_index (entry index -> KeyValue)
 *   - get_elements_ptr (raw pointer to dense entries array)
 *   - reserve / clear / reset
 *
 * Use RBMap if you need to iterate over sorted elements.
 *
 * Use HashMap if:
 *   - You need to keep an iterator or const pointer to Key and you intend to
 *     add/remove elements in the meantime.
 *   - You need to preserve the insertion order when using erase.
 */
template <typename TKey, typename TValue,
		typename Hasher = HashMapHasherDefault,
		typename Comparator = HashMapComparatorDefault<TKey>>
class AHashMap {
public:
	// Power of two; must be at least kGroupWidth (16 for SSE2/WASM, 8 otherwise).
	static constexpr uint32_t INITIAL_CAPACITY = 16;
	using KV = KeyValue<TKey, TValue>;

private:
	using Mask = typename SwissTable::Group::Mask;
	static constexpr uint32_t kGroupWidth = SwissTable::Group::kWidth;

	// Index table:
	//   _ctrl: capacity + kGroupWidth control bytes (the trailing bytes mirror
	//          the leading ones so group probing can wrap-overshoot safely).
	//   _slots: capacity entry indices (uint32_t each), parallel to _ctrl[0..capacity).
	uint8_t *_ctrl = nullptr;
	uint32_t *_slots = nullptr;

	// Dense entries store (insertion order is NOT preserved on erase).
	KV *_elements = nullptr;
	// Parallel cached 32-bit hashes, sized identically to _elements. Stored
	// at insert time so probes can short-circuit before a (potentially
	// expensive) key compare and so rehashing on grow doesn't have to re-run
	// Hasher on the key.
	uint32_t *_hashes = nullptr;
	uint32_t _elements_capacity = 0;

	uint32_t _capacity = 0; // Power of two, or 0 if unallocated.
	uint32_t _capacity_mask = 0; // _capacity - 1.
	uint32_t _size = 0;
	uint32_t _growth_left = 0; // Number of inserts allowed before next grow/rehash.
	uint32_t _deleted = 0; // Tombstones currently in the index table.

	static _FORCE_INLINE_ uint32_t _hash(const TKey &p_key) {
		// Apply a SwissTable-friendly bit mixer at the boundary so callers
		// don't have to. Several Godot hashers (notably String/DJB2) leave
		// the high bits weakly mixed; splitting them into h1/h2 directly
		// would cluster fingerprints and inflate probe chains. The same
		// mixed value is what we cache in _hashes[i], so that grow and the
		// hash short-circuit in _lookup keep working consistently.
		return SwissTable::mix(Hasher::hash(p_key));
	}

	// Round up to a power of two no smaller than min, and at least kGroupWidth.
	static _FORCE_INLINE_ uint32_t _round_up_capacity(uint32_t p_min) {
		uint32_t cap = kGroupWidth;
		while (cap < p_min) {
			cap <<= 1;
		}
		return cap;
	}

	// Find the slot containing p_key. Returns true if found, with r_slot_idx
	// set to the slot index in _ctrl/_slots and r_element_idx to the index in
	// _elements. Uses the cached 32-bit hash on each candidate to short-
	// circuit before the (potentially expensive) key compare.
	bool _lookup(const TKey &p_key, uint32_t p_hash, uint32_t &r_slot_idx, uint32_t &r_element_idx) const {
		if (_capacity == 0) {
			return false;
		}
		const uint8_t h2 = SwissTable::h2(p_hash);
		uint32_t group_idx = SwissTable::h1(p_hash) & _capacity_mask;
		uint32_t probe_dist = 0;

		while (true) {
			SwissTable::Group g(_ctrl + group_idx);
			Mask m = g.match(h2);
			auto it = m.iter();
			while (it.has_next()) {
				uint32_t i = it.next();
				uint32_t slot_idx = (group_idx + i) & _capacity_mask;
				uint32_t elem_idx = _slots[slot_idx];
				if (_hashes[elem_idx] == p_hash && Comparator::compare(_elements[elem_idx].key, p_key)) {
					r_slot_idx = slot_idx;
					r_element_idx = elem_idx;
					return true;
				}
			}
			// If this group has any empty slot, the key cannot be further along
			// the probe sequence (deletes are tombstones, empties are not).
			if (g.match_empty()) {
				return false;
			}
			probe_dist += kGroupWidth;
			// Quadratic probing across groups; bounded by capacity (always succeeds
			// because we maintain at least one empty slot in the table).
			group_idx = (group_idx + probe_dist) & _capacity_mask;
		}
	}

	// Combined "lookup if present, otherwise pick insert slot" probe used
	// by insert and operator[]. Returns true if the key is already in the
	// table (with r_slot_idx / r_element_idx set to the existing entry);
	// returns false with r_slot_idx pointing at a kEmpty/kDeleted slot
	// where the new entry should be placed.
	//
	// Caller must ensure _capacity > 0 and _growth_left > 0.
	bool _find_or_prepare_insert(const TKey &p_key, uint32_t p_hash,
			uint32_t &r_slot_idx, uint32_t &r_element_idx) const {
		const uint8_t h2 = SwissTable::h2(p_hash);
		uint32_t group_idx = SwissTable::h1(p_hash) & _capacity_mask;
		uint32_t probe_dist = 0;
		uint32_t insert_slot = 0;
		bool have_insert_slot = false;

		while (true) {
			SwissTable::Group g(_ctrl + group_idx);
			Mask m = g.match(h2);
			auto it = m.iter();
			while (it.has_next()) {
				const uint32_t i = it.next();
				const uint32_t slot_idx = (group_idx + i) & _capacity_mask;
				const uint32_t elem_idx = _slots[slot_idx];
				if (_hashes[elem_idx] == p_hash && Comparator::compare(_elements[elem_idx].key, p_key)) {
					r_slot_idx = slot_idx;
					r_element_idx = elem_idx;
					return true;
				}
			}
			if (!have_insert_slot) {
				Mask ed = g.match_empty_or_deleted();
				if (ed) {
					const uint32_t i = ed.lowest_set_bit();
					insert_slot = (group_idx + i) & _capacity_mask;
					have_insert_slot = true;
				}
			}
			if (g.match_empty()) {
				r_slot_idx = insert_slot;
				return false;
			}
			probe_dist += kGroupWidth;
			group_idx = (group_idx + probe_dist) & _capacity_mask;
		}
	}

	// Find the first empty-or-deleted slot where a new element with p_hash can
	// be inserted. Caller is responsible for ensuring growth_left > 0.
	uint32_t _find_insert_slot(uint32_t p_hash) const {
		uint32_t group_idx = SwissTable::h1(p_hash) & _capacity_mask;
		uint32_t probe_dist = 0;
		while (true) {
			SwissTable::Group g(_ctrl + group_idx);
			Mask m = g.match_empty_or_deleted();
			if (m) {
				uint32_t i = m.lowest_set_bit();
				return (group_idx + i) & _capacity_mask;
			}
			probe_dist += kGroupWidth;
			group_idx = (group_idx + probe_dist) & _capacity_mask;
		}
	}

	// Decide whether an erased slot can become kEmpty (so it doesn't
	// permanently consume probe-chain length) or must remain a kDeleted
	// tombstone. With unaligned probing, the promotion to kEmpty is only
	// safe when no probe sequence could ever have been forced to skip past
	// this slot while it was full -- approximated by requiring an empty in
	// both the kGroupWidth-window immediately after AND immediately before
	// the slot, with no run of kGroupWidth full bytes in between. This
	// matches absl::container_internal::WasNeverFull.
	_FORCE_INLINE_ uint8_t _ctrl_after_erase(uint32_t p_slot_idx) const {
		// Single-group tables: no probe sequence ever leaves the group, so
		// converting an erased slot to empty is always safe.
		if (_capacity <= kGroupWidth) {
			return SwissTable::kEmpty;
		}
		// Trailing: number of consecutive non-empty bytes starting AT the
		// slot (looking forward). We scan up to kGroupWidth bytes; the
		// trailing-mirror region of _ctrl makes the read safe even if the
		// slot is near the end of the index table.
		uint32_t trailing = 0;
		while (trailing < kGroupWidth && _ctrl[p_slot_idx + trailing] != SwissTable::kEmpty) {
			trailing++;
		}
		if (trailing == kGroupWidth) {
			return SwissTable::kDeleted;
		}
		// Leading: number of consecutive non-empty bytes ending JUST BEFORE
		// the slot (looking backward, with wrap).
		uint32_t leading = 0;
		while (leading < kGroupWidth) {
			const uint32_t idx = (p_slot_idx + _capacity - 1 - leading) & _capacity_mask;
			if (_ctrl[idx] == SwissTable::kEmpty) {
				break;
			}
			leading++;
		}
		// Promote to empty only when the run of non-empties surrounding the
		// slot is shorter than one full group window -- meaning no probe
		// could have been forced to scan a full group through this slot.
		if (trailing + leading < kGroupWidth) {
			return SwissTable::kEmpty;
		}
		return SwissTable::kDeleted;
	}

	// Set the control byte and the trailing mirror byte (if applicable).
	_FORCE_INLINE_ void _set_ctrl(uint32_t p_slot_idx, uint8_t p_value) {
		_ctrl[p_slot_idx] = p_value;
		// Mirror the first (kGroupWidth - 1) control bytes at the tail so that
		// group probing can read past the end without bounds checks.
		const uint32_t mirrored = ((p_slot_idx - (kGroupWidth - 1)) & _capacity_mask) + (kGroupWidth - 1);
		_ctrl[mirrored] = p_value;
	}

	// (Re)allocate the index table to the given capacity (power of two), then
	// reinsert all existing elements.
	void _resize_table(uint32_t p_new_capacity) {
		const uint32_t new_capacity = MAX(p_new_capacity, kGroupWidth);
		// Allocate ctrl with kGroupWidth-1 trailing mirror bytes.
		uint8_t *new_ctrl = reinterpret_cast<uint8_t *>(
				Memory::alloc_static(sizeof(uint8_t) * (new_capacity + kGroupWidth)));
		uint32_t *new_slots = reinterpret_cast<uint32_t *>(
				Memory::alloc_static(sizeof(uint32_t) * new_capacity));
		memset(new_ctrl, SwissTable::kEmpty, new_capacity + kGroupWidth);

		uint8_t *old_ctrl = _ctrl;
		uint32_t *old_slots = _slots;

		_ctrl = new_ctrl;
		_slots = new_slots;
		_capacity = new_capacity;
		_capacity_mask = new_capacity - 1;
		_growth_left = SwissTable::capacity_to_growth(new_capacity);
		_deleted = 0;

		// Reinsert all live elements (entries themselves don't move). We
		// reuse the cached hashes so grow doesn't have to re-run Hasher.
		for (uint32_t i = 0; i < _size; i++) {
			const uint32_t hash = _hashes[i];
			const uint32_t slot = _find_insert_slot(hash);
			_set_ctrl(slot, SwissTable::h2(hash));
			_slots[slot] = i;
			_growth_left--;
		}

		if (old_ctrl != nullptr) {
			Memory::free_static(old_ctrl);
			Memory::free_static(old_slots);
		}
	}

	void _grow_entries(uint32_t p_new_capacity) {
		if (p_new_capacity <= _elements_capacity) {
			return;
		}
		_elements = reinterpret_cast<KV *>(
				Memory::realloc_static(_elements, sizeof(KV) * p_new_capacity));
		_hashes = reinterpret_cast<uint32_t *>(
				Memory::realloc_static(_hashes, sizeof(uint32_t) * p_new_capacity));
		_elements_capacity = p_new_capacity;
	}

	// Compute the entry-array capacity needed to support a given index-table
	// capacity (we hold up to growth-left elements).
	static _FORCE_INLINE_ uint32_t _entries_capacity_for(uint32_t p_index_capacity) {
		return SwissTable::capacity_to_growth(p_index_capacity);
	}

	void _ensure_allocated() {
		if (_capacity == 0) {
			_resize_table(MAX((uint32_t)INITIAL_CAPACITY, kGroupWidth));
			_grow_entries(_entries_capacity_for(_capacity));
		}
	}

	void _grow() {
		const uint32_t new_capacity = _capacity == 0 ? MAX((uint32_t)INITIAL_CAPACITY, kGroupWidth) : (_capacity * 2);
		_resize_table(new_capacity);
		_grow_entries(_entries_capacity_for(_capacity));
	}

	// Place a new element at the given (already chosen) slot. The slot must
	// currently be kEmpty or kDeleted, and the caller must have already
	// guaranteed _growth_left > 0 and that the entries array has room.
	_FORCE_INLINE_ uint32_t _emplace_at_slot(const TKey &p_key, const TValue &p_value,
			uint32_t p_hash, uint32_t p_slot) {
		const bool was_deleted = SwissTable::is_deleted(_ctrl[p_slot]);
		_set_ctrl(p_slot, SwissTable::h2(p_hash));
		const uint32_t element_idx = _size;
		_slots[p_slot] = element_idx;
		memnew_placement(&_elements[element_idx], KV(p_key, p_value));
		_hashes[element_idx] = p_hash;
		_size++;
		if (was_deleted) {
			_deleted--;
		}
		_growth_left--;
		return element_idx;
	}

	// Insert with the assumption that p_key does not already exist.
	uint32_t _insert_new(const TKey &p_key, const TValue &p_value, uint32_t p_hash) {
		if (_growth_left == 0) {
			_grow();
		}
		const uint32_t slot = _find_insert_slot(p_hash);
		return _emplace_at_slot(p_key, p_value, p_hash, slot);
	}

	void _init_from(const AHashMap &p_other) {
		if (p_other._size == 0) {
			return;
		}
		_resize_table(p_other._capacity);
		// Match the source's full entries-array capacity, not just its current
		// size. The index table copied below was sized for `p_other._capacity`
		// and still has growth_left slots available without rehashing; if we
		// only allocate `p_other._size` entries here, subsequent inserts will
		// happily write past the entries array before the next grow.
		_grow_entries(MAX(p_other._size, _entries_capacity_for(_capacity)));
		// Copy entries.
		if constexpr (std::is_trivially_copyable_v<TKey> && std::is_trivially_copyable_v<TValue>) {
			memcpy(static_cast<void *>(_elements), static_cast<const void *>(p_other._elements), sizeof(KV) * p_other._size);
		} else {
			for (uint32_t i = 0; i < p_other._size; i++) {
				memnew_placement(&_elements[i], KV(p_other._elements[i]));
			}
		}
		// Copy parallel hash array.
		memcpy(_hashes, p_other._hashes, sizeof(uint32_t) * p_other._size);
		_size = p_other._size;
		// Copy ctrl table verbatim, including mirror bytes.
		memcpy(_ctrl, p_other._ctrl, p_other._capacity + kGroupWidth);
		memcpy(_slots, p_other._slots, sizeof(uint32_t) * p_other._capacity);
		_growth_left = p_other._growth_left;
		_deleted = p_other._deleted;
	}

public:
	/* Standard Godot Container API */

	_FORCE_INLINE_ uint32_t get_capacity() const { return _capacity; }
	_FORCE_INLINE_ uint32_t size() const { return _size; }

	_FORCE_INLINE_ bool is_empty() const {
		return _size == 0;
	}

	void clear() {
		if (_capacity == 0 || _size == 0) {
			return;
		}
		if constexpr (!(std::is_trivially_destructible_v<TKey> && std::is_trivially_destructible_v<TValue>)) {
			for (uint32_t i = 0; i < _size; i++) {
				_elements[i].key.~TKey();
				_elements[i].value.~TValue();
			}
		}
		memset(_ctrl, SwissTable::kEmpty, _capacity + kGroupWidth);
		_size = 0;
		_deleted = 0;
		_growth_left = SwissTable::capacity_to_growth(_capacity);
	}

	TValue &get(const TKey &p_key) {
		uint32_t slot = 0;
		uint32_t element_idx = 0;
		bool exists = _lookup(p_key, _hash(p_key), slot, element_idx);
		CRASH_COND_MSG(!exists, "AHashMap key not found.");
		return _elements[element_idx].value;
	}

	const TValue &get(const TKey &p_key) const {
		uint32_t slot = 0;
		uint32_t element_idx = 0;
		bool exists = _lookup(p_key, _hash(p_key), slot, element_idx);
		CRASH_COND_MSG(!exists, "AHashMap key not found.");
		return _elements[element_idx].value;
	}

	const TValue *getptr(const TKey &p_key) const {
		uint32_t slot = 0;
		uint32_t element_idx = 0;
		if (_lookup(p_key, _hash(p_key), slot, element_idx)) {
			return &_elements[element_idx].value;
		}
		return nullptr;
	}

	TValue *getptr(const TKey &p_key) {
		uint32_t slot = 0;
		uint32_t element_idx = 0;
		if (_lookup(p_key, _hash(p_key), slot, element_idx)) {
			return &_elements[element_idx].value;
		}
		return nullptr;
	}

	bool has(const TKey &p_key) const {
		uint32_t slot = 0;
		uint32_t element_idx = 0;
		return _lookup(p_key, _hash(p_key), slot, element_idx);
	}

	bool erase(const TKey &p_key) {
		uint32_t slot = 0;
		uint32_t element_idx = 0;
		if (!_lookup(p_key, _hash(p_key), slot, element_idx)) {
			return false;
		}
		const uint8_t new_ctrl = _ctrl_after_erase(slot);
		if (new_ctrl == SwissTable::kEmpty) {
			_growth_left++;
		} else {
			_deleted++;
		}
		_set_ctrl(slot, new_ctrl);

		// Destroy the element.
		_elements[element_idx].key.~TKey();
		_elements[element_idx].value.~TValue();
		_size--;

		// Swap-with-tail to keep entries dense. We need to update the moved
		// entry's slot mapping BEFORE doing the actual move, because a
		// post-move lookup would find the slot but read the (now-garbage)
		// data at the old tail index when comparing keys. We use the cached
		// hash on the tail entry to drive the lookup so we don't re-run
		// Hasher on the moved key.
		if (element_idx < _size) {
			uint32_t moved_slot = 0;
			uint32_t moved_idx = 0;
			bool found = _lookup(_elements[_size].key, _hashes[_size], moved_slot, moved_idx);
			(void)found;
			DEV_ASSERT(found && moved_idx == _size);
			_slots[moved_slot] = element_idx;
			memcpy(static_cast<void *>(&_elements[element_idx]), static_cast<const void *>(&_elements[_size]), sizeof(KV));
			_hashes[element_idx] = _hashes[_size];
		}

		return true;
	}

	// Replace the key of an entry in-place, without invalidating its index in
	// the entries array. p_old_key must exist; p_new_key must not unless equal
	// to p_old_key.
	bool replace_key(const TKey &p_old_key, const TKey &p_new_key) {
		if (p_old_key == p_new_key) {
			return true;
		}
		uint32_t old_slot = 0;
		uint32_t element_idx = 0;
		ERR_FAIL_COND_V(!_lookup(p_old_key, _hash(p_old_key), old_slot, element_idx), false);
		uint32_t new_slot_check = 0;
		uint32_t check_idx = 0;
		ERR_FAIL_COND_V(_lookup(p_new_key, _hash(p_new_key), new_slot_check, check_idx), false);

		const uint8_t new_ctrl = _ctrl_after_erase(old_slot);
		if (new_ctrl == SwissTable::kEmpty) {
			_growth_left++;
		} else {
			_deleted++;
		}
		_set_ctrl(old_slot, new_ctrl);

		// Mutate the key in place and update the cached hash.
		const_cast<TKey &>(_elements[element_idx].key) = p_new_key;
		const uint32_t new_hash = _hash(p_new_key);
		_hashes[element_idx] = new_hash;

		// Insert into the new position. We may need to grow if the deleted slot
		// converted to empty consumed our growth budget; check growth_left.
		if (_growth_left == 0) {
			_grow();
		}
		const uint32_t slot = _find_insert_slot(new_hash);
		const bool was_deleted = SwissTable::is_deleted(_ctrl[slot]);
		_set_ctrl(slot, SwissTable::h2(new_hash));
		_slots[slot] = element_idx;
		if (was_deleted) {
			_deleted--;
		}
		_growth_left--;

		return true;
	}

	void reserve(uint32_t p_new_capacity) {
		if (p_new_capacity <= _entries_capacity_for(_capacity)) {
			if (_capacity != 0 && p_new_capacity < _size) {
				WARN_VERBOSE("reserve() called with a capacity smaller than the current size. This is likely a mistake.");
			}
			return;
		}
		// Choose smallest power-of-two index capacity that fits p_new_capacity
		// elements at 7/8 load factor.
		uint32_t needed_index_capacity = SwissTable::growth_to_lower_bound_capacity(p_new_capacity);
		uint32_t cap = _round_up_capacity(needed_index_capacity);
		_resize_table(cap);
		_grow_entries(MAX(p_new_capacity, _entries_capacity_for(_capacity)));
	}

	/** Iterator API **/

	struct ConstIterator {
		_FORCE_INLINE_ const KV &operator*() const { return *pair; }
		_FORCE_INLINE_ const KV *operator->() const { return pair; }
		_FORCE_INLINE_ ConstIterator &operator++() {
			pair++;
			return *this;
		}
		_FORCE_INLINE_ ConstIterator &operator--() {
			pair--;
			if (pair < begin_ptr) {
				pair = end_ptr;
			}
			return *this;
		}
		_FORCE_INLINE_ bool operator==(const ConstIterator &b) const { return pair == b.pair; }
		_FORCE_INLINE_ bool operator!=(const ConstIterator &b) const { return pair != b.pair; }
		_FORCE_INLINE_ explicit operator bool() const { return pair != end_ptr; }

		_FORCE_INLINE_ ConstIterator(const KV *p_pair, const KV *p_begin, const KV *p_end) :
				pair(p_pair), begin_ptr(p_begin), end_ptr(p_end) {}
		_FORCE_INLINE_ ConstIterator() {}
		_FORCE_INLINE_ ConstIterator(const ConstIterator &p_it) = default;
		_FORCE_INLINE_ ConstIterator &operator=(const ConstIterator &p_it) = default;

	private:
		const KV *pair = nullptr;
		const KV *begin_ptr = nullptr;
		const KV *end_ptr = nullptr;
	};

	struct Iterator {
		_FORCE_INLINE_ KV &operator*() const { return *pair; }
		_FORCE_INLINE_ KV *operator->() const { return pair; }
		_FORCE_INLINE_ Iterator &operator++() {
			pair++;
			return *this;
		}
		_FORCE_INLINE_ Iterator &operator--() {
			pair--;
			if (pair < begin_ptr) {
				pair = end_ptr;
			}
			return *this;
		}
		_FORCE_INLINE_ bool operator==(const Iterator &b) const { return pair == b.pair; }
		_FORCE_INLINE_ bool operator!=(const Iterator &b) const { return pair != b.pair; }
		_FORCE_INLINE_ explicit operator bool() const { return pair != end_ptr; }

		_FORCE_INLINE_ Iterator(KV *p_pair, KV *p_begin, KV *p_end) :
				pair(p_pair), begin_ptr(p_begin), end_ptr(p_end) {}
		_FORCE_INLINE_ Iterator() {}
		_FORCE_INLINE_ Iterator(const Iterator &p_it) = default;
		_FORCE_INLINE_ Iterator &operator=(const Iterator &p_it) = default;

		operator ConstIterator() const {
			return ConstIterator(pair, begin_ptr, end_ptr);
		}

	private:
		KV *pair = nullptr;
		KV *begin_ptr = nullptr;
		KV *end_ptr = nullptr;
	};

	_FORCE_INLINE_ Iterator begin() { return Iterator(_elements, _elements, _elements + _size); }
	_FORCE_INLINE_ Iterator end() { return Iterator(_elements + _size, _elements, _elements + _size); }
	_FORCE_INLINE_ Iterator last() {
		if (unlikely(_size == 0)) {
			return Iterator(nullptr, nullptr, nullptr);
		}
		return Iterator(_elements + _size - 1, _elements, _elements + _size);
	}

	Iterator find(const TKey &p_key) {
		uint32_t slot = 0;
		uint32_t element_idx = 0;
		if (!_lookup(p_key, _hash(p_key), slot, element_idx)) {
			return end();
		}
		return Iterator(_elements + element_idx, _elements, _elements + _size);
	}

	void remove(const Iterator &p_iter) {
		if (p_iter) {
			erase(p_iter->key);
		}
	}

	_FORCE_INLINE_ ConstIterator begin() const { return ConstIterator(_elements, _elements, _elements + _size); }
	_FORCE_INLINE_ ConstIterator end() const { return ConstIterator(_elements + _size, _elements, _elements + _size); }
	_FORCE_INLINE_ ConstIterator last() const {
		if (unlikely(_size == 0)) {
			return ConstIterator(nullptr, nullptr, nullptr);
		}
		return ConstIterator(_elements + _size - 1, _elements, _elements + _size);
	}

	ConstIterator find(const TKey &p_key) const {
		uint32_t slot = 0;
		uint32_t element_idx = 0;
		if (!_lookup(p_key, _hash(p_key), slot, element_idx)) {
			return end();
		}
		return ConstIterator(_elements + element_idx, _elements, _elements + _size);
	}

	/* Indexing */

	const TValue &operator[](const TKey &p_key) const {
		uint32_t slot = 0;
		uint32_t element_idx = 0;
		bool exists = _lookup(p_key, _hash(p_key), slot, element_idx);
		CRASH_COND(!exists);
		return _elements[element_idx].value;
	}

	TValue &operator[](const TKey &p_key) {
		const uint32_t hash = _hash(p_key);
		_ensure_allocated();
		if (_growth_left == 0) {
			_grow();
		}
		uint32_t slot = 0;
		uint32_t element_idx = 0;
		if (_size > 0 && _find_or_prepare_insert(p_key, hash, slot, element_idx)) {
			return _elements[element_idx].value;
		}
		if (_size == 0) {
			slot = _find_insert_slot(hash);
		}
		element_idx = _emplace_at_slot(p_key, TValue(), hash, slot);
		return _elements[element_idx].value;
	}

	/* Insert */

	Iterator insert(const TKey &p_key, const TValue &p_value) {
		const uint32_t hash = _hash(p_key);
		_ensure_allocated();
		if (_growth_left == 0) {
			_grow();
		}
		uint32_t slot = 0;
		uint32_t element_idx = 0;
		if (_size > 0 && _find_or_prepare_insert(p_key, hash, slot, element_idx)) {
			_elements[element_idx].value = p_value;
			return Iterator(_elements + element_idx, _elements, _elements + _size);
		}
		if (_size == 0) {
			slot = _find_insert_slot(hash);
		}
		element_idx = _emplace_at_slot(p_key, p_value, hash, slot);
		return Iterator(_elements + element_idx, _elements, _elements + _size);
	}

	// Inserts an element without checking if it already exists.
	Iterator insert_new(const TKey &p_key, const TValue &p_value) {
		DEV_ASSERT(!has(p_key));
		_ensure_allocated();
		const uint32_t element_idx = _insert_new(p_key, p_value, _hash(p_key));
		return Iterator(_elements + element_idx, _elements, _elements + _size);
	}

	/* Array methods. */

	// Unsafe. Pointer is invalidated by any reserve/insert that grows the
	// entries array, and by erases that swap with the tail.
	KV *get_elements_ptr() {
		return _elements;
	}

	// Returns the element index. If not found, returns -1.
	int get_index(const TKey &p_key) {
		uint32_t slot = 0;
		uint32_t element_idx = 0;
		if (!_lookup(p_key, _hash(p_key), slot, element_idx)) {
			return -1;
		}
		return static_cast<int>(element_idx);
	}

	KV &get_by_index(uint32_t p_index) {
		CRASH_BAD_UNSIGNED_INDEX(p_index, _size);
		return _elements[p_index];
	}

	bool erase_by_index(uint32_t p_index) {
		if (p_index >= _size) {
			return false;
		}
		return erase(_elements[p_index].key);
	}

	/* Constructors */

	AHashMap(AHashMap &&p_other) {
		_ctrl = p_other._ctrl;
		_slots = p_other._slots;
		_elements = p_other._elements;
		_hashes = p_other._hashes;
		_elements_capacity = p_other._elements_capacity;
		_capacity = p_other._capacity;
		_capacity_mask = p_other._capacity_mask;
		_size = p_other._size;
		_growth_left = p_other._growth_left;
		_deleted = p_other._deleted;

		p_other._ctrl = nullptr;
		p_other._slots = nullptr;
		p_other._elements = nullptr;
		p_other._hashes = nullptr;
		p_other._elements_capacity = 0;
		p_other._capacity = 0;
		p_other._capacity_mask = 0;
		p_other._size = 0;
		p_other._growth_left = 0;
		p_other._deleted = 0;
	}

	explicit AHashMap(const AHashMap &p_other) {
		_init_from(p_other);
	}

	void operator=(const AHashMap &p_other) {
		if (this == &p_other) {
			return;
		}
		reset();
		_init_from(p_other);
	}

	AHashMap(uint32_t p_initial_capacity) {
		reserve(p_initial_capacity);
	}

	AHashMap() {}

	AHashMap(std::initializer_list<KV> p_init) {
		reserve(p_init.size());
		for (const KV &E : p_init) {
			insert(E.key, E.value);
		}
	}

	void reset() {
		if (_elements != nullptr) {
			if constexpr (!(std::is_trivially_destructible_v<TKey> && std::is_trivially_destructible_v<TValue>)) {
				for (uint32_t i = 0; i < _size; i++) {
					_elements[i].key.~TKey();
					_elements[i].value.~TValue();
				}
			}
			Memory::free_static(_elements);
			_elements = nullptr;
			_elements_capacity = 0;
		}
		if (_hashes != nullptr) {
			Memory::free_static(_hashes);
			_hashes = nullptr;
		}
		if (_ctrl != nullptr) {
			Memory::free_static(_ctrl);
			Memory::free_static(_slots);
			_ctrl = nullptr;
			_slots = nullptr;
		}
		_capacity = 0;
		_capacity_mask = 0;
		_size = 0;
		_growth_left = 0;
		_deleted = 0;
	}

	~AHashMap() {
		reset();
	}
};

extern template class AHashMap<int, int>;
extern template class AHashMap<String, int>;
extern template class AHashMap<StringName, StringName>;
extern template class AHashMap<StringName, Variant>;
extern template class AHashMap<StringName, int>;
