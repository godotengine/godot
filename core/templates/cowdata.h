/**************************************************************************/
/*  cowdata.h                                                             */
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

#include "core/error/error_macros.h"
#include "core/os/memory.h"
#include "core/string/print_string.h"
#include "core/templates/safe_refcount.h"
#include "core/templates/span.h"

#include <initializer_list>
#include <type_traits>

static_assert(std::is_trivially_destructible_v<std::atomic<uint64_t>>);

// Silences false-positive warnings.
GODOT_GCC_WARNING_PUSH
GODOT_GCC_WARNING_IGNORE("-Wplacement-new") // Silence a false positive warning (see GH-52119).
GODOT_GCC_WARNING_IGNORE("-Wmaybe-uninitialized") // False positive raised when using constexpr.
GODOT_GCC_WARNING_IGNORE("-Warray-bounds")
GODOT_GCC_WARNING_IGNORE("-Wrestrict")
GODOT_GCC_PRAGMA(GCC diagnostic warning "-Wstringop-overflow=0") // Can't "ignore" this for some reason.
#ifdef WINDOWS_ENABLED
GODOT_GCC_PRAGMA(GCC diagnostic warning "-Wdangling-pointer=0") // Can't "ignore" this for some reason.
#endif

template <typename T>
class CowData {
public:
	typedef int64_t Size;
	typedef uint64_t USize;
	static constexpr USize MAX_INT = INT64_MAX;

private:
	// Alignment:  ↓ max_align_t           ↓ USize          ↓ USize            ↓ MAX_ALIGN
	//             ┌────────────────────┬──┬───────────────┬──┬─────────────┬──┬───────────...
	//             │ SafeNumeric<USize> │░░│ USize         │░░│ USize       │░░│ T[]
	//             │ ref. count         │░░│ data capacity │░░│ data size   │░░│ data
	//             └────────────────────┴──┴───────────────┴──┴─────────────┴──┴───────────...
	// Offset:     ↑ REF_COUNT_OFFSET      ↑ CAPACITY_OFFSET  ↑ SIZE_OFFSET    ↑ DATA_OFFSET

	static constexpr size_t REF_COUNT_OFFSET = 0;
	static constexpr size_t CAPACITY_OFFSET = Memory::get_aligned_address(REF_COUNT_OFFSET + sizeof(SafeNumeric<USize>), alignof(USize));
	static constexpr size_t SIZE_OFFSET = Memory::get_aligned_address(CAPACITY_OFFSET + sizeof(USize), alignof(USize));
	static constexpr size_t DATA_OFFSET = Memory::get_aligned_address(SIZE_OFFSET + sizeof(USize), Memory::MAX_ALIGN);

	mutable T *_ptr = nullptr;

	// internal helpers

	static constexpr _FORCE_INLINE_ USize grow_capacity(USize p_previous_capacity) {
		// 1.5x the given size.
		// This ratio was chosen because it is close to the ideal growth rate of the golden ratio.
		// See https://archive.ph/Z2R8w for details.
		return MAX((USize)2, p_previous_capacity + ((1 + p_previous_capacity) >> 1));
	}

	static constexpr _FORCE_INLINE_ USize next_capacity(USize p_previous_capacity, USize p_size) {
		if (p_previous_capacity < p_size) {
			return MAX(grow_capacity(p_previous_capacity), p_size);
		}
		return p_previous_capacity;
	}

	static constexpr _FORCE_INLINE_ USize smaller_capacity(USize p_previous_capacity, USize p_size) {
		if (p_size < p_previous_capacity >> 2) {
			return grow_capacity(p_size);
		}
		return p_previous_capacity;
	}

	static _FORCE_INLINE_ T *_get_data_ptr(uint8_t *p_ptr) {
		return (T *)(p_ptr + DATA_OFFSET);
	}

	/// Note: Assumes _ptr != nullptr.
	_FORCE_INLINE_ SafeNumeric<USize> *_get_refcount() const {
		return (SafeNumeric<USize> *)((uint8_t *)_ptr - DATA_OFFSET + REF_COUNT_OFFSET);
	}

	/// Note: Assumes _ptr != nullptr.
	_FORCE_INLINE_ USize *_get_size() const {
		return (USize *)((uint8_t *)_ptr - DATA_OFFSET + SIZE_OFFSET);
	}

	/// Note: Assumes _ptr != nullptr.
	_FORCE_INLINE_ USize *_get_capacity() const {
		return (USize *)((uint8_t *)_ptr - DATA_OFFSET + CAPACITY_OFFSET);
	}

	// Decrements the reference count. Deallocates the backing buffer if needed.
	// After this function, _ptr is guaranteed to be NULL.
	void _unref();
	void _ref(const CowData *p_from);
	void _ref(const CowData &p_from);

	/// Allocates a backing array of the given capacity. The reference count is initialized to 1, size to 0.
	/// It is the responsibility of the caller to:
	/// - Ensure _ptr == nullptr
	/// - Ensure p_capacity > 0
	Error _alloc_exact(USize p_capacity);

	/// Re-allocates the backing array to the given capacity.
	/// It is the responsibility of the caller to:
	/// - Ensure we are the only owner of the backing array
	/// - Ensure p_capacity > 0
	Error _realloc_exact(USize p_capacity);

	/// Create a new buffer and copies over elements from the old buffer.
	/// Elements are inserted first from the start, then a gap is left uninitialized, and then elements are inserted from the back.
	/// It is the responsibility of the caller to:
	/// - Construct elements in the gap.
	/// - Ensure size() >= p_size_from_start and size() >= p_size_from_back.
	/// - Ensure p_capacity is enough to hold all elements.
	[[nodiscard]] Error _copy_to_new_buffer_exact(USize p_capacity, USize p_size_from_start, USize p_gap, USize p_size_from_back);

	/// Ensure we are the only owners of the backing buffer.
	[[nodiscard]] Error _copy_on_write();

public:
	void operator=(const CowData<T> &p_from) { _ref(p_from); }
	void operator=(CowData<T> &&p_from) {
		if (_ptr == p_from._ptr) {
			return;
		}

		_unref();
		_ptr = p_from._ptr;
		p_from._ptr = nullptr;
	}

	_FORCE_INLINE_ T *ptrw() {
		// If forking fails, we can only crash.
		CRASH_COND(_copy_on_write());
		return _ptr;
	}

	_FORCE_INLINE_ const T *ptr() const {
		return _ptr;
	}

	_FORCE_INLINE_ Size size() const { return !_ptr ? 0 : *_get_size(); }
	_FORCE_INLINE_ USize capacity() const { return !_ptr ? 0 : *_get_capacity(); }
	_FORCE_INLINE_ USize refcount() const { return !_ptr ? 0 : *_get_refcount(); }

	_FORCE_INLINE_ void clear() { _unref(); }
	_FORCE_INLINE_ bool is_empty() const { return size() == 0; }

	_FORCE_INLINE_ void set(Size p_index, const T &p_elem) {
		ERR_FAIL_INDEX(p_index, size());
		// TODO Returning the error would be more appropriate.
		CRASH_COND(_copy_on_write());
		_ptr[p_index] = p_elem;
	}

	_FORCE_INLINE_ T &get_m(Size p_index) {
		CRASH_BAD_INDEX(p_index, size());
		// If we fail to fork, all we can do is crash,
		// since the caller may write incorrectly to the unforked array.
		CRASH_COND(_copy_on_write());
		return _ptr[p_index];
	}

	_FORCE_INLINE_ const T &get(Size p_index) const {
		CRASH_BAD_INDEX(p_index, size());

		return _ptr[p_index];
	}

	template <bool p_init = false>
	Error resize(Size p_size);

	template <bool p_exact = false>
	Error reserve(USize p_min_capacity);
	_FORCE_INLINE_ Error reserve_exact(USize p_capacity) {
		return reserve<true>(p_capacity);
	}

	_FORCE_INLINE_ void remove_at(Size p_index);

	Error insert(Size p_pos, T &&p_val);
	Error push_back(T &&p_val);

	_FORCE_INLINE_ operator Span<T>() const { return Span<T>(ptr(), size()); }
	_FORCE_INLINE_ Span<T> span() const { return operator Span<T>(); }

	_FORCE_INLINE_ CowData() {}
	_FORCE_INLINE_ ~CowData() { _unref(); }
	_FORCE_INLINE_ CowData(std::initializer_list<T> p_init);
	_FORCE_INLINE_ CowData(const CowData<T> &p_from) { _ref(p_from); }
	_FORCE_INLINE_ CowData(CowData<T> &&p_from) {
		_ptr = p_from._ptr;
		p_from._ptr = nullptr;
	}
};

template <typename T>
void CowData<T>::_unref() {
	if (!_ptr) {
		return;
	}

	if (_get_refcount()->decrement() > 0) {
		// Data is still in use elsewhere.
		_ptr = nullptr;
		return;
	}
	// We had the only reference; destroy the data.

	// First, invalidate our own reference.
	// NOTE: It is required to do so immediately because it must not be observable outside of this
	//       function after refcount has already been reduced to 0.
	// WARNING: It must be done before calling the destructors, because one of them may otherwise
	//          observe it through a reference to us. In this case, it may try to access the buffer,
	//          which is illegal after some of the elements in it have already been destructed, and
	//          may lead to a segmentation fault.
	USize current_size = size();
	T *prev_ptr = _ptr;
	_ptr = nullptr;

	destruct_arr_placement(prev_ptr, current_size);

	// Safety check; none of the destructors should have added elements during destruction.
	DEV_ASSERT(!_ptr);

	// Free Memory.
	Memory::free_static((uint8_t *)prev_ptr - DATA_OFFSET, false);

#ifdef DEBUG_ENABLED
	// If any destructors access us through pointers, it is a bug.
	// We can't really test for that, but we can at least check no items have been added.
	ERR_FAIL_COND_MSG(_ptr != nullptr, "Internal bug, please report: CowData was modified during destruction.");
#endif
}

template <typename T>
void CowData<T>::remove_at(Size p_index) {
	const Size prev_size = size();
	ERR_FAIL_INDEX(p_index, prev_size);

	if (prev_size == 1) {
		// Removing the only element.
		_unref();
		return;
	}

	const USize new_size = prev_size - 1;

	if (_get_refcount()->get() == 1) {
		// We're the only owner; remove in-place.

		// Destruct the element, then relocate the rest one down.
		_ptr[p_index].~T();
		memmove((void *)(_ptr + p_index), (void *)(_ptr + p_index + 1), (new_size - p_index) * sizeof(T));

		// Shrink to fit if necessary.
		const USize new_capacity = smaller_capacity(capacity(), new_size);
		if (new_capacity < capacity()) {
			Error err = _realloc_exact(new_capacity);
			CRASH_COND(err);
		}
		*_get_size() = new_size;
	} else {
		// Remove by forking.
		Error err = _copy_to_new_buffer_exact(smaller_capacity(capacity(), new_size), p_index, 0, new_size - p_index);
		CRASH_COND(err);
	}
}

template <typename T>
Error CowData<T>::insert(Size p_pos, T &&p_val) {
	const Size new_size = size() + 1;
	ERR_FAIL_INDEX_V(p_pos, new_size, ERR_INVALID_PARAMETER);

	if (!_ptr) {
		_alloc_exact(next_capacity(0, 1));
		*_get_size() = 1;
	} else if (_get_refcount()->get() == 1) {
		if ((USize)new_size > capacity()) {
			// Need to grow.
			const Error error = _realloc_exact(grow_capacity(capacity()));
			if (error) {
				return error;
			}
		}

		// Relocate elements one position up.
		memmove((void *)(_ptr + p_pos + 1), (void *)(_ptr + p_pos), (size() - p_pos) * sizeof(T));
		*_get_size() = new_size;
	} else {
		// Insert new element by forking.
		// Use the max of capacity and new_size, to ensure we don't accidentally shrink after reserve.
		const USize new_capacity = next_capacity(capacity(), new_size);
		const Error error = _copy_to_new_buffer_exact(new_capacity, p_pos, 1, size() - p_pos);
		if (error) {
			return error;
		}
	}

	// Create the new element at the given index.
	memnew_placement(_ptr + p_pos, T(std::move(p_val)));

	return OK;
}

template <typename T>
Error CowData<T>::push_back(T &&p_val) {
	const Size new_size = size() + 1;

	if (!_ptr) {
		// Grow by allocating.
		_alloc_exact(next_capacity(0, 1));
		*_get_size() = 1;
	} else if (_get_refcount()->get() == 1) {
		// Grow in-place.
		if ((USize)new_size > capacity()) {
			// Need to grow.
			const Error error = _realloc_exact(grow_capacity(capacity()));
			if (error) {
				return error;
			}
		}

		*_get_size() = new_size;
	} else {
		// Grow by forking.
		// Use the max of capacity and new_size, to ensure we don't accidentally shrink after reserve.
		const USize new_capacity = next_capacity(capacity(), new_size);
		const Error error = _copy_to_new_buffer_exact(new_capacity, size(), 1, 0);
		if (error) {
			return error;
		}
	}

	// Create the new element at the given index.
	memnew_placement(_ptr + new_size - 1, T(std::move(p_val)));

	return OK;
}

template <typename T>
template <bool p_exact>
Error CowData<T>::reserve(USize p_min_capacity) {
	USize new_capacity = p_exact ? p_min_capacity : next_capacity(capacity(), p_min_capacity);
	if (new_capacity <= capacity()) {
		if (p_min_capacity < (USize)size()) {
			WARN_VERBOSE("reserve() called with a capacity smaller than the current size. This is likely a mistake.");
		}
		// No need to reserve more, we already have (at least) the right size.
		return OK;
	}

	if (!_ptr) {
		// Initial allocation.
		return _alloc_exact(new_capacity);
	} else if (_get_refcount()->get() == 1) {
		// Grow in-place.
		return _realloc_exact(new_capacity);
	} else {
		// Grow by forking.
		return _copy_to_new_buffer_exact(new_capacity, size(), 0, 0);
	}
}

template <typename T>
template <bool p_initialize>
Error CowData<T>::resize(Size p_size) {
	ERR_FAIL_COND_V(p_size < 0, ERR_INVALID_PARAMETER);

	const Size prev_size = size();
	if (p_size > prev_size) {
		// Caller wants to grow.

		if (!_ptr) {
			// Grow by allocating.
			const Error error = _alloc_exact(next_capacity(0, p_size));
			if (error) {
				return error;
			}
		} else if (_get_refcount()->get() == 1) {
			// Grow in-place.
			if ((USize)p_size > capacity()) {
				const Error error = _realloc_exact(next_capacity(capacity(), p_size));
				if (error) {
					return error;
				}
			}
		} else {
			// Grow by forking.
			const Error error = _copy_to_new_buffer_exact(next_capacity(capacity(), p_size), prev_size, 0, 0);
			if (error) {
				return error;
			}
		}

		// Construct new elements.
		if constexpr (p_initialize) {
			memnew_arr_placement(_ptr + prev_size, p_size - prev_size);
		}
		*_get_size() = p_size;

		return OK;
	} else {
		// Caller wants to shrink.

		if (p_size == 0) {
			_unref();
			return OK;
		} else if (_get_refcount()->get() == 1) {
			// Shrink in-place.
			destruct_arr_placement(_ptr + p_size, prev_size - p_size);

			// Shrink buffer if necessary.
			const USize new_capacity = smaller_capacity(capacity(), p_size);
			if (new_capacity < capacity()) {
				Error err = _realloc_exact(new_capacity);
				CRASH_COND(err);
			}

			*_get_size() = p_size;
			return OK;
		} else {
			// Shrink by forking.
			const USize new_capacity = smaller_capacity(capacity(), p_size);
			return _copy_to_new_buffer_exact(new_capacity, p_size, 0, 0);
		}
	}
}

template <typename T>
Error CowData<T>::_alloc_exact(USize p_capacity) {
	DEV_ASSERT(!_ptr);

	uint8_t *mem_new = (uint8_t *)Memory::alloc_static(p_capacity * sizeof(T) + DATA_OFFSET, false);
	ERR_FAIL_NULL_V(mem_new, ERR_OUT_OF_MEMORY);

	_ptr = _get_data_ptr(mem_new);

	// If we alloc, we're guaranteed to be the only reference.
	new (_get_refcount()) SafeNumeric<USize>(1);
	*_get_size() = 0;
	// The actual capacity is whatever we can stuff into the alloc_size.
	*_get_capacity() = p_capacity;

	return OK;
}

template <typename T>
Error CowData<T>::_realloc_exact(USize p_capacity) {
	DEV_ASSERT(_ptr);

	uint8_t *mem_new = (uint8_t *)Memory::realloc_static(((uint8_t *)_ptr) - DATA_OFFSET, p_capacity * sizeof(T) + DATA_OFFSET, false);
	ERR_FAIL_NULL_V(mem_new, ERR_OUT_OF_MEMORY);

	_ptr = _get_data_ptr(mem_new);

	// If we realloc, we're guaranteed to be the only reference.
	// So the reference was 1 and was copied to be 1 again.
	DEV_ASSERT(_get_refcount()->get() == 1);
	// The size was also copied from the previous allocation.
	// The actual capacity is whatever we can stuff into the alloc_size.
	*_get_capacity() = p_capacity;

	return OK;
}

template <typename T>
Error CowData<T>::_copy_to_new_buffer_exact(USize p_capacity, USize p_size_from_start, USize p_gap, USize p_size_from_back) {
	DEV_ASSERT(p_capacity >= p_size_from_start + p_size_from_back + p_gap);
	DEV_ASSERT((USize)size() >= p_size_from_start && (USize)size() >= p_size_from_back);

	// Create a temporary CowData to hold ownership over our _ptr.
	// It will be used to copy elements from the old buffer over to our new buffer.
	// At the end of the block, it will be automatically destructed by going out of scope.
	const CowData prev_data;
	prev_data._ptr = _ptr;
	_ptr = nullptr;

	const Error error = _alloc_exact(p_capacity);
	if (error) {
		// On failure to allocate, recover the old data and return the error.
		_ptr = prev_data._ptr;
		prev_data._ptr = nullptr;
		return error;
	}

	// Copy over elements.
	copy_arr_placement(_ptr, prev_data._ptr, p_size_from_start);
	copy_arr_placement(
			_ptr + p_size_from_start + p_gap,
			prev_data._ptr + prev_data.size() - p_size_from_back,
			p_size_from_back);
	*_get_size() = p_size_from_start + p_gap + p_size_from_back;

	return OK;
}

template <typename T>
Error CowData<T>::_copy_on_write() {
	if (!_ptr || _get_refcount()->get() == 1) {
		// Nothing to do.
		return OK;
	}

	// Fork to become the only reference.
	return _copy_to_new_buffer_exact(capacity(), size(), 0, 0);
}

template <typename T>
void CowData<T>::_ref(const CowData *p_from) {
	_ref(*p_from);
}

template <typename T>
void CowData<T>::_ref(const CowData &p_from) {
	if (_ptr == p_from._ptr) {
		return; // self assign, do nothing.
	}

	_unref(); // Resets _ptr to nullptr.

	if (!p_from._ptr) {
		return; //nothing to do
	}

	if (p_from._get_refcount()->conditional_increment() > 0) { // could reference
		_ptr = p_from._ptr;
	}
}

template <typename T>
CowData<T>::CowData(std::initializer_list<T> p_init) {
	CRASH_COND(_alloc_exact(p_init.size()));

	copy_arr_placement(_ptr, p_init.begin(), p_init.size());
	*_get_size() = p_init.size();
}

GODOT_GCC_WARNING_POP

// Zero-constructing CowData initializes _ptr to nullptr (and thus empty).
template <typename T>
struct is_zero_constructible<CowData<T>> : std::true_type {};
