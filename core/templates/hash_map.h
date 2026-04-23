/**************************************************************************/
/*  hash_map.h                                                            */
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
#include "core/templates/sort_list.h"
#include "core/templates/swiss_table_simd.h"

#include <initializer_list>

/**
 * An insertion-ordered, pointer-stable hash map. Uses a SwissTable-style
 * SIMD-scanned control-byte index over a separately allocated entries store.
 *
 * Layout:
 *   - _ctrl[capacity + kGroupWidth]: control bytes (h2 / kEmpty / kDeleted),
 *     with the leading kGroupWidth bytes mirrored at the tail so groups can
 *     overshoot safely.
 *   - _slots[capacity]: pointer per slot to the owning HashMapElement (or null
 *     if empty/deleted).
 *   - HashMapElement holds key/value plus prev/next pointers for the
 *     intrusive insertion-order doubly linked list (rooted at _head_element /
 *     _tail_element). The physical placement of HashMapElement in memory is
 *     independent of its position in the index table.
 *
 * Properties:
 *   - Pointer stability: once inserted, KeyValue addresses (and iterators
 *     pointing to them) are valid until that specific entry is erased or the
 *     entire map is destroyed. Inserts/erases of OTHER keys never invalidate
 *     them, even when the index table is rehashed.
 *   - Insertion order is preserved across erases, and iteration follows the
 *     intrusive linked list. `front_insert` prepends instead of appending.
 *   - Lookups SIMD-scan a group at a time, then run a full key compare only
 *     on slots whose 7-bit fingerprint matches.
 *
 * The Allocator template parameter manages HashMapElement allocation. The
 * default uses the heap; PagedAllocator may be passed for pool-based storage.
 *
 * Use AHashMap if you don't need order preservation or pointer stability and
 * want the densest possible memory layout.
 */

template <typename TKey, typename TValue>
struct HashMapElement {
	HashMapElement *next = nullptr;
	HashMapElement *prev = nullptr;
	// Cached full 32-bit hash. Stored at insert time so probes can
	// short-circuit before a (potentially expensive) key compare, and so
	// rehashing on grow doesn't have to re-run Hasher on the key.
	uint32_t hash = 0;
	KeyValue<TKey, TValue> data;
	HashMapElement() {}
	HashMapElement(const TKey &p_key, const TValue &p_value, uint32_t p_hash) :
			hash(p_hash), data(p_key, p_value) {}
};

template <typename TKey, typename TValue,
		typename Hasher = HashMapHasherDefault,
		typename Comparator = HashMapComparatorDefault<TKey>,
		typename Allocator = DefaultTypedAllocator<HashMapElement<TKey, TValue>>>
class HashMap : private Allocator {
public:
	static constexpr uint32_t INITIAL_CAPACITY = 16;
	using KV = KeyValue<TKey, TValue>;
	using Element = HashMapElement<TKey, TValue>;

private:
	using Mask = typename SwissTable::Group::Mask;
	static constexpr uint32_t kGroupWidth = SwissTable::Group::kWidth;

	// Index table.
	uint8_t *_ctrl = nullptr;
	Element **_slots = nullptr;

	// Insertion-order linked list head/tail.
	Element *_head_element = nullptr;
	Element *_tail_element = nullptr;

	uint32_t _capacity = 0; // Power of two, or 0 if unallocated.
	uint32_t _capacity_mask = 0; // _capacity - 1.
	uint32_t _size = 0;
	uint32_t _growth_left = 0;
	uint32_t _deleted = 0;

	// Initial capacity hint requested via the (uint32_t) constructor or
	// reserve() before any actual allocation has happened. Once _ctrl is
	// allocated this is no longer consulted.
	uint32_t _pending_capacity = 0;

	_FORCE_INLINE_ static uint32_t _hash(const TKey &p_key) {
		// Apply a SwissTable-friendly bit mixer at the boundary so callers
		// don't have to. Several Godot hashers (notably String/DJB2) leave
		// the high bits weakly mixed; splitting them into h1/h2 directly
		// would cluster fingerprints and inflate probe chains. The same
		// mixed value is what we cache on Element::hash, so that grow and
		// the hash short-circuit in _lookup keep working consistently.
		return SwissTable::mix(Hasher::hash(p_key));
	}

	static _FORCE_INLINE_ uint32_t _round_up_capacity(uint32_t p_min) {
		uint32_t cap = kGroupWidth;
		while (cap < p_min) {
			cap <<= 1;
		}
		return cap;
	}

	// Find the slot containing p_key. Returns true if found, with r_slot_idx
	// set to the slot index in _ctrl/_slots. Uses the cached 32-bit hash on
	// each candidate element to short-circuit before the (potentially
	// expensive) key compare; with a 7-bit fingerprint we expect a false
	// match every ~128 slots, and with a stored 32-bit hash false matches
	// almost never escalate to a real key compare.
	bool _lookup(const TKey &p_key, uint32_t p_hash, uint32_t &r_slot_idx) const {
		if (_capacity == 0 || _size == 0) {
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
				const uint32_t i = it.next();
				const uint32_t slot_idx = (group_idx + i) & _capacity_mask;
				Element *elem = _slots[slot_idx];
				if (elem->hash == p_hash && Comparator::compare(elem->data.key, p_key)) {
					r_slot_idx = slot_idx;
					return true;
				}
			}
			if (g.match_empty()) {
				return false;
			}
			probe_dist += kGroupWidth;
			group_idx = (group_idx + probe_dist) & _capacity_mask;
		}
	}

	// Find the slot containing p_key OR the first empty/deleted slot the
	// probe sequence visits. Single-pass replacement for "_lookup then
	// _find_insert_slot"; saves one probe walk per insert miss.
	//
	// Returns true if the key was found at r_slot_idx; r_slot_idx is the
	// existing slot. Returns false if the key was not found; r_slot_idx
	// is the slot where a new entry should be placed (always either kEmpty
	// or kDeleted -- caller must pass through to _set_ctrl / accounting).
	//
	// Caller must ensure _capacity > 0 and _growth_left > 0 before calling
	// (an empty/full table cannot satisfy both halves of the contract).
	bool _find_or_prepare_insert(const TKey &p_key, uint32_t p_hash, uint32_t &r_slot_idx) const {
		const uint8_t h2 = SwissTable::h2(p_hash);
		uint32_t group_idx = SwissTable::h1(p_hash) & _capacity_mask;
		uint32_t probe_dist = 0;
		uint32_t insert_slot = 0;
		bool have_insert_slot = false;

		while (true) {
			SwissTable::Group g(_ctrl + group_idx);
			// Look for a matching fingerprint first; if any candidate's full
			// hash matches and key compares equal, we've found the entry.
			Mask m = g.match(h2);
			auto it = m.iter();
			while (it.has_next()) {
				const uint32_t i = it.next();
				const uint32_t slot_idx = (group_idx + i) & _capacity_mask;
				Element *elem = _slots[slot_idx];
				if (elem->hash == p_hash && Comparator::compare(elem->data.key, p_key)) {
					r_slot_idx = slot_idx;
					return true;
				}
			}
			// Record the first kEmpty/kDeleted slot we encounter so we have
			// somewhere to put the new entry once we conclude the key isn't
			// present anywhere along the probe chain.
			if (!have_insert_slot) {
				Mask ed = g.match_empty_or_deleted();
				if (ed) {
					const uint32_t i = ed.lowest_set_bit();
					insert_slot = (group_idx + i) & _capacity_mask;
					have_insert_slot = true;
				}
			}
			// kEmpty terminates the probe chain: the key cannot appear past
			// this group, so the recorded insert slot (which exists, since
			// this group has at least one empty byte) is the answer.
			if (g.match_empty()) {
				r_slot_idx = insert_slot;
				return false;
			}
			probe_dist += kGroupWidth;
			group_idx = (group_idx + probe_dist) & _capacity_mask;
		}
	}

	uint32_t _find_insert_slot(uint32_t p_hash) const {
		uint32_t group_idx = SwissTable::h1(p_hash) & _capacity_mask;
		uint32_t probe_dist = 0;
		while (true) {
			SwissTable::Group g(_ctrl + group_idx);
			Mask m = g.match_empty_or_deleted();
			if (m) {
				const uint32_t i = m.lowest_set_bit();
				return (group_idx + i) & _capacity_mask;
			}
			probe_dist += kGroupWidth;
			group_idx = (group_idx + probe_dist) & _capacity_mask;
		}
	}

	_FORCE_INLINE_ void _set_ctrl(uint32_t p_slot_idx, uint8_t p_value) {
		_ctrl[p_slot_idx] = p_value;
		const uint32_t mirrored = ((p_slot_idx - (kGroupWidth - 1)) & _capacity_mask) + (kGroupWidth - 1);
		_ctrl[mirrored] = p_value;
	}

	void _resize_table(uint32_t p_new_capacity) {
		const uint32_t new_capacity = MAX(p_new_capacity, kGroupWidth);
		uint8_t *new_ctrl = reinterpret_cast<uint8_t *>(
				Memory::alloc_static(sizeof(uint8_t) * (new_capacity + kGroupWidth)));
		Element **new_slots = reinterpret_cast<Element **>(
				Memory::alloc_static(sizeof(Element *) * new_capacity));
		memset(new_ctrl, SwissTable::kEmpty, new_capacity + kGroupWidth);

		uint8_t *old_ctrl = _ctrl;
		Element **old_slots = _slots;

		_ctrl = new_ctrl;
		_slots = new_slots;
		_capacity = new_capacity;
		_capacity_mask = new_capacity - 1;
		_growth_left = SwissTable::capacity_to_growth(new_capacity);
		_deleted = 0;

		// Reinsert by walking the existing insertion-order linked list. This
		// preserves order while letting us drop the old index table. We
		// reuse the cached hash on each Element so grow doesn't have to
		// re-run Hasher (a real win for non-trivial keys like String).
		for (Element *e = _head_element; e != nullptr; e = e->next) {
			const uint32_t hash = e->hash;
			const uint32_t slot = _find_insert_slot(hash);
			_set_ctrl(slot, SwissTable::h2(hash));
			_slots[slot] = e;
			_growth_left--;
		}

		if (old_ctrl != nullptr) {
			Memory::free_static(old_ctrl);
			Memory::free_static(old_slots);
		}
	}

	void _ensure_allocated() {
		if (_capacity == 0) {
			const uint32_t initial = MAX(_pending_capacity, MAX((uint32_t)INITIAL_CAPACITY, kGroupWidth));
			_resize_table(_round_up_capacity(initial));
			_pending_capacity = 0;
		}
	}

	void _grow() {
		const uint32_t new_capacity = _capacity == 0 ? MAX((uint32_t)INITIAL_CAPACITY, kGroupWidth) : (_capacity * 2);
		_resize_table(new_capacity);
	}

	// Decide whether an erased slot can become kEmpty (so it doesn't
	// permanently consume probe-chain length) or must remain a kDeleted
	// tombstone. With unaligned probing, the promotion to kEmpty is only
	// safe when no probe sequence could ever have been forced to skip past
	// this slot while it was full -- approximated by requiring an empty in
	// both the kGroupWidth-window immediately after AND immediately before
	// the slot, with no run of kGroupWidth full bytes in between. This
	// matches absl::container_internal::WasNeverFull.
	_FORCE_INLINE_ uint8_t _ctrl_after_erase(uint32_t p_slot) const {
		if (_capacity <= kGroupWidth) {
			return SwissTable::kEmpty;
		}
		// SIMD-scan the kGroupWidth-byte windows AFTER and BEFORE p_slot for
		// kEmpty. Both reads are single unaligned loads -- the trailing read
		// stays in-bounds because of the mirror tail past _capacity, and the
		// leading read of the kGroupWidth bytes ending at p_slot - 1 is also
		// in-bounds for any p_slot since the wrap case (p_slot < kGroupWidth)
		// lands its tail in that same mirror region.
		const SwissTable::Group g_after(_ctrl + p_slot);
		const Mask after_empty = g_after.match_empty();
		if (!after_empty) {
			return SwissTable::kDeleted;
		}
		const uint32_t trailing = after_empty.lowest_set_bit();

		const uint32_t lead_start = (p_slot - kGroupWidth) & _capacity_mask;
		const SwissTable::Group g_before(_ctrl + lead_start);
		const Mask before_empty = g_before.match_empty();
		// Slot at position k within g_before is (kGroupWidth - 1 - k) bytes
		// before p_slot, so the closest kEmpty corresponds to the highest set
		// bit. No empty -> full kGroupWidth-byte run, force kDeleted.
		const uint32_t leading = before_empty
				? ((kGroupWidth - 1u) - before_empty.highest_set_bit())
				: kGroupWidth;
		if (trailing + leading < kGroupWidth) {
			return SwissTable::kEmpty;
		}
		return SwissTable::kDeleted;
	}

	void _link_after_alloc(Element *p_elem, bool p_front_insert) {
		if (_tail_element == nullptr) {
			_head_element = p_elem;
			_tail_element = p_elem;
		} else if (p_front_insert) {
			_head_element->prev = p_elem;
			p_elem->next = _head_element;
			_head_element = p_elem;
		} else {
			_tail_element->next = p_elem;
			p_elem->prev = _tail_element;
			_tail_element = p_elem;
		}
	}

	// Remove p_elem from the insertion-order list. Caller is expected to
	// either delete p_elem immediately afterwards or re-link it before any
	// further iteration -- we deliberately leave p_elem->{prev,next} dangling
	// to avoid two pointless writes on the erase hot path.
	void _unlink(Element *p_elem) {
		Element *const prev = p_elem->prev;
		Element *const next = p_elem->next;
		if (prev) {
			prev->next = next;
		} else {
			// p_elem was the head.
			_head_element = next;
		}
		if (next) {
			next->prev = prev;
		} else {
			// p_elem was the tail.
			_tail_element = prev;
		}
	}

	// Place a brand-new element at the given (already chosen) slot. The slot
	// must currently be kEmpty or kDeleted, and the caller must have already
	// guaranteed _growth_left > 0.
	_FORCE_INLINE_ Element *_emplace_at_slot(const TKey &p_key, const TValue &p_value,
			uint32_t p_hash, uint32_t p_slot, bool p_front_insert) {
		const bool was_deleted = SwissTable::is_deleted(_ctrl[p_slot]);
		Element *elem = Allocator::new_allocation(Element(p_key, p_value, p_hash));
		_set_ctrl(p_slot, SwissTable::h2(p_hash));
		_slots[p_slot] = elem;
		_link_after_alloc(elem, p_front_insert);
		_size++;
		if (was_deleted) {
			_deleted--;
		}
		_growth_left--;
		return elem;
	}

	Element *_insert_internal(const TKey &p_key, const TValue &p_value, uint32_t p_hash, bool p_front_insert) {
		_ensure_allocated();
		if (_growth_left == 0) {
			_grow();
		}
		const uint32_t slot = _find_insert_slot(p_hash);
		return _emplace_at_slot(p_key, p_value, p_hash, slot, p_front_insert);
	}

	void _clear_data() {
		Element *current = _tail_element;
		while (current != nullptr) {
			Element *prev = current->prev;
			Allocator::delete_allocation(current);
			current = prev;
		}
		_head_element = nullptr;
		_tail_element = nullptr;
		_size = 0;
	}

public:
	_FORCE_INLINE_ uint32_t get_capacity() const { return _capacity; }
	_FORCE_INLINE_ uint32_t size() const { return _size; }

	bool is_empty() const {
		return _size == 0;
	}

	void clear() {
		if (_size == 0 && _deleted == 0) {
			return;
		}
		_clear_data();
		if (_ctrl != nullptr) {
			memset(_ctrl, SwissTable::kEmpty, _capacity + kGroupWidth);
		}
		_deleted = 0;
		_growth_left = (_capacity > 0) ? SwissTable::capacity_to_growth(_capacity) : 0;
	}

	void sort() {
		sort_custom<KeyValueSort<TKey, TValue>>();
	}

	template <typename C>
	void sort_custom() {
		if (size() < 2) {
			return;
		}
		SortList<Element, KeyValue<TKey, TValue>, &Element::data, &Element::prev, &Element::next, C> sorter;
		sorter.sort(_head_element, _tail_element);
	}

	TValue &get(const TKey &p_key) {
		uint32_t slot = 0;
		const bool exists = _lookup(p_key, _hash(p_key), slot);
		CRASH_COND_MSG(!exists, "HashMap key not found.");
		return _slots[slot]->data.value;
	}

	const TValue &get(const TKey &p_key) const {
		uint32_t slot = 0;
		const bool exists = _lookup(p_key, _hash(p_key), slot);
		CRASH_COND_MSG(!exists, "HashMap key not found.");
		return _slots[slot]->data.value;
	}

	const TValue *getptr(const TKey &p_key) const {
		uint32_t slot = 0;
		if (_lookup(p_key, _hash(p_key), slot)) {
			return &_slots[slot]->data.value;
		}
		return nullptr;
	}

	TValue *getptr(const TKey &p_key) {
		uint32_t slot = 0;
		if (_lookup(p_key, _hash(p_key), slot)) {
			return &_slots[slot]->data.value;
		}
		return nullptr;
	}

	_FORCE_INLINE_ bool has(const TKey &p_key) const {
		uint32_t slot = 0;
		return _lookup(p_key, _hash(p_key), slot);
	}

	bool erase(const TKey &p_key) {
		uint32_t slot = 0;
		if (!_lookup(p_key, _hash(p_key), slot)) {
			return false;
		}
		Element *elem = _slots[slot];
		const uint8_t new_ctrl = _ctrl_after_erase(slot);
		if (new_ctrl == SwissTable::kEmpty) {
			_growth_left++;
		} else {
			_deleted++;
		}
		_set_ctrl(slot, new_ctrl);
		// Leave _slots[slot] dangling: any subsequent reader (lookup,
		// resize, iteration) gates on _ctrl[slot] being kEmpty/kDeleted
		// before consulting _slots, so this write is dead and skipping it
		// shaves a small amount off the erase hot path.
		_unlink(elem);
		Allocator::delete_allocation(elem);
		_size--;
		return true;
	}

	// Replace the key of an entry in-place, without invalidating iterators or
	// changing the entry's position in the insertion-order list. p_old_key
	// must exist; p_new_key must not, unless equal to p_old_key.
	bool replace_key(const TKey &p_old_key, const TKey &p_new_key) {
		ERR_FAIL_COND_V(_capacity == 0 || _size == 0, false);
		if (p_old_key == p_new_key) {
			return true;
		}
		const uint32_t new_hash = _hash(p_new_key);
		uint32_t check_slot = 0;
		ERR_FAIL_COND_V(_lookup(p_new_key, new_hash, check_slot), false);
		uint32_t old_slot = 0;
		ERR_FAIL_COND_V(!_lookup(p_old_key, _hash(p_old_key), old_slot), false);
		Element *elem = _slots[old_slot];

		// Vacate the old slot.
		const uint8_t new_ctrl = _ctrl_after_erase(old_slot);
		if (new_ctrl == SwissTable::kEmpty) {
			_growth_left++;
		} else {
			_deleted++;
		}
		_set_ctrl(old_slot, new_ctrl);
		_slots[old_slot] = nullptr;

		// Mutate the key in place. KeyValue::key is `const TKey`; the cast is
		// safe because the entry is uniquely owned and no caller may hold a
		// concurrent reference to it. Update the cached hash to match.
		const_cast<TKey &>(elem->data.key) = p_new_key;
		elem->hash = new_hash;

		// Reinsert into a fresh slot under the new hash.
		const uint32_t target = _find_insert_slot(new_hash);
		const bool was_deleted = SwissTable::is_deleted(_ctrl[target]);
		_set_ctrl(target, SwissTable::h2(new_hash));
		_slots[target] = elem;
		if (was_deleted) {
			_deleted--;
		} else {
			_growth_left--;
		}
		return true;
	}

	// Reserve at least p_new_capacity *entries*. Sized so the index table
	// can host that many keys without triggering another grow.
	void reserve(uint32_t p_new_capacity) {
		const uint32_t needed_index_capacity = _round_up_capacity(
				MAX(SwissTable::growth_to_lower_bound_capacity(p_new_capacity), kGroupWidth));
		if (_capacity == 0) {
			_pending_capacity = MAX(_pending_capacity, needed_index_capacity);
			return;
		}
		if (needed_index_capacity <= _capacity) {
			if (p_new_capacity < _size) {
				WARN_VERBOSE("reserve() called with a capacity smaller than the current size. This is likely a mistake.");
			}
			return;
		}
		_resize_table(needed_index_capacity);
	}

	/** Iterator API **/

	struct ConstIterator {
		_FORCE_INLINE_ const KeyValue<TKey, TValue> &operator*() const { return E->data; }
		_FORCE_INLINE_ const KeyValue<TKey, TValue> *operator->() const { return &E->data; }
		_FORCE_INLINE_ ConstIterator &operator++() {
			if (E) {
				E = E->next;
			}
			return *this;
		}
		_FORCE_INLINE_ ConstIterator &operator--() {
			if (E) {
				E = E->prev;
			}
			return *this;
		}

		_FORCE_INLINE_ bool operator==(const ConstIterator &b) const { return E == b.E; }
		_FORCE_INLINE_ bool operator!=(const ConstIterator &b) const { return E != b.E; }

		_FORCE_INLINE_ explicit operator bool() const { return E != nullptr; }

		_FORCE_INLINE_ ConstIterator(const Element *p_E) { E = p_E; }
		_FORCE_INLINE_ ConstIterator() {}
		_FORCE_INLINE_ ConstIterator(const ConstIterator &p_it) { E = p_it.E; }
		_FORCE_INLINE_ void operator=(const ConstIterator &p_it) {
			E = p_it.E;
		}

	private:
		const Element *E = nullptr;
	};

	struct Iterator {
		_FORCE_INLINE_ KeyValue<TKey, TValue> &operator*() const { return E->data; }
		_FORCE_INLINE_ KeyValue<TKey, TValue> *operator->() const { return &E->data; }
		_FORCE_INLINE_ Iterator &operator++() {
			if (E) {
				E = E->next;
			}
			return *this;
		}
		_FORCE_INLINE_ Iterator &operator--() {
			if (E) {
				E = E->prev;
			}
			return *this;
		}

		_FORCE_INLINE_ bool operator==(const Iterator &b) const { return E == b.E; }
		_FORCE_INLINE_ bool operator!=(const Iterator &b) const { return E != b.E; }

		_FORCE_INLINE_ explicit operator bool() const { return E != nullptr; }

		_FORCE_INLINE_ Iterator(Element *p_E) { E = p_E; }
		_FORCE_INLINE_ Iterator() {}
		_FORCE_INLINE_ Iterator(const Iterator &p_it) { E = p_it.E; }
		_FORCE_INLINE_ void operator=(const Iterator &p_it) {
			E = p_it.E;
		}

		operator ConstIterator() const {
			return ConstIterator(E);
		}

	private:
		Element *E = nullptr;
	};

	_FORCE_INLINE_ Iterator begin() {
		return Iterator(_head_element);
	}
	_FORCE_INLINE_ Iterator end() {
		return Iterator(nullptr);
	}
	_FORCE_INLINE_ Iterator last() {
		return Iterator(_tail_element);
	}

	_FORCE_INLINE_ Iterator find(const TKey &p_key) {
		uint32_t slot = 0;
		if (!_lookup(p_key, _hash(p_key), slot)) {
			return end();
		}
		return Iterator(_slots[slot]);
	}

	_FORCE_INLINE_ void remove(const Iterator &p_iter) {
		if (p_iter) {
			erase(p_iter->key);
		}
	}

	_FORCE_INLINE_ ConstIterator begin() const {
		return ConstIterator(_head_element);
	}
	_FORCE_INLINE_ ConstIterator end() const {
		return ConstIterator(nullptr);
	}
	_FORCE_INLINE_ ConstIterator last() const {
		return ConstIterator(_tail_element);
	}

	_FORCE_INLINE_ ConstIterator find(const TKey &p_key) const {
		uint32_t slot = 0;
		if (!_lookup(p_key, _hash(p_key), slot)) {
			return end();
		}
		return ConstIterator(_slots[slot]);
	}

	/* Indexing */

	const TValue &operator[](const TKey &p_key) const {
		uint32_t slot = 0;
		const bool exists = _lookup(p_key, _hash(p_key), slot);
		CRASH_COND(!exists);
		return _slots[slot]->data.value;
	}

	TValue &operator[](const TKey &p_key) {
		const uint32_t hash = _hash(p_key);
		_ensure_allocated();
		if (_growth_left == 0) {
			_grow();
		}
		uint32_t slot = 0;
		if (_size > 0 && _find_or_prepare_insert(p_key, hash, slot)) {
			return _slots[slot]->data.value;
		}
		// Single-pass path: when the table is empty we still need to pick a
		// slot. _find_or_prepare_insert handles that, but only when called;
		// when _size == 0 we skip it and fall back to the simpler probe.
		if (_size == 0) {
			slot = _find_insert_slot(hash);
		}
		return _emplace_at_slot(p_key, TValue(), hash, slot, false)->data.value;
	}

	/* Insert */

	Iterator insert(const TKey &p_key, const TValue &p_value, bool p_front_insert = false) {
		const uint32_t hash = _hash(p_key);
		_ensure_allocated();
		if (_growth_left == 0) {
			_grow();
		}
		uint32_t slot = 0;
		if (_size > 0 && _find_or_prepare_insert(p_key, hash, slot)) {
			_slots[slot]->data.value = p_value;
			return Iterator(_slots[slot]);
		}
		if (_size == 0) {
			slot = _find_insert_slot(hash);
		}
		return Iterator(_emplace_at_slot(p_key, p_value, hash, slot, p_front_insert));
	}

	/* Constructors */

	HashMap() {}

	explicit HashMap(uint32_t p_initial_capacity) {
		reserve(p_initial_capacity);
	}

	HashMap(const HashMap &p_other) {
		if (p_other._size == 0) {
			return;
		}
		reserve(p_other._size);
		for (const KeyValue<TKey, TValue> &E : p_other) {
			insert(E.key, E.value);
		}
	}

	HashMap(HashMap &&p_other) {
		_ctrl = p_other._ctrl;
		_slots = p_other._slots;
		_head_element = p_other._head_element;
		_tail_element = p_other._tail_element;
		_capacity = p_other._capacity;
		_capacity_mask = p_other._capacity_mask;
		_size = p_other._size;
		_growth_left = p_other._growth_left;
		_deleted = p_other._deleted;
		_pending_capacity = p_other._pending_capacity;

		p_other._ctrl = nullptr;
		p_other._slots = nullptr;
		p_other._head_element = nullptr;
		p_other._tail_element = nullptr;
		p_other._capacity = 0;
		p_other._capacity_mask = 0;
		p_other._size = 0;
		p_other._growth_left = 0;
		p_other._deleted = 0;
		p_other._pending_capacity = 0;
	}

	void operator=(const HashMap &p_other) {
		if (this == &p_other) {
			return;
		}
		clear();
		if (p_other._size == 0) {
			return;
		}
		reserve(p_other._size);
		for (const KeyValue<TKey, TValue> &E : p_other) {
			insert(E.key, E.value);
		}
	}

	HashMap &operator=(HashMap &&p_other) {
		if (this == &p_other) {
			return *this;
		}
		_clear_data();
		if (_ctrl != nullptr) {
			Memory::free_static(_ctrl);
			Memory::free_static(_slots);
		}

		_ctrl = p_other._ctrl;
		_slots = p_other._slots;
		_head_element = p_other._head_element;
		_tail_element = p_other._tail_element;
		_capacity = p_other._capacity;
		_capacity_mask = p_other._capacity_mask;
		_size = p_other._size;
		_growth_left = p_other._growth_left;
		_deleted = p_other._deleted;
		_pending_capacity = p_other._pending_capacity;

		p_other._ctrl = nullptr;
		p_other._slots = nullptr;
		p_other._head_element = nullptr;
		p_other._tail_element = nullptr;
		p_other._capacity = 0;
		p_other._capacity_mask = 0;
		p_other._size = 0;
		p_other._growth_left = 0;
		p_other._deleted = 0;
		p_other._pending_capacity = 0;

		return *this;
	}

	HashMap(std::initializer_list<KeyValue<TKey, TValue>> p_init) {
		reserve(p_init.size());
		for (const KeyValue<TKey, TValue> &E : p_init) {
			insert(E.key, E.value);
		}
	}

	uint32_t debug_get_hash(uint32_t p_idx) {
		if (_size == 0 || _capacity == 0) {
			return 0;
		}
		ERR_FAIL_INDEX_V(p_idx, _capacity, 0);
		const uint8_t c = _ctrl[p_idx];
		if (c == SwissTable::kEmpty || c == SwissTable::kDeleted) {
			return 0;
		}
		Element *e = _slots[p_idx];
		return e != nullptr ? e->hash : 0;
	}
	Iterator debug_get_element(uint32_t p_idx) {
		if (_size == 0 || _capacity == 0) {
			return Iterator();
		}
		ERR_FAIL_INDEX_V(p_idx, _capacity, Iterator());
		Element *e = _slots[p_idx];
		if (e == nullptr) {
			return Iterator();
		}
		return Iterator(e);
	}

	~HashMap() {
		_clear_data();
		if (_ctrl != nullptr) {
			Memory::free_static(_ctrl);
			Memory::free_static(_slots);
		}
	}
};
