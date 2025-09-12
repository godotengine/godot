/**************************************************************************/
/*  heap_object.h                                                         */
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

#include <initializer_list>

// (dev-note)
// (I wrote these implementations myself - I am unsure if there is a well-known name for these patterns; if there is, these should be renamed to match)
// (it may also be appropriate to not leave this as a template, and just move it directly into variant_struct.h)

/////////////////////////

// VarHeapPointer is useful for storing a known type with an unknown type in the heap
// Behaves a lot like a pointer to the known type, which can be accessed as usual using the `->` operator
// While the unknown object data can be accessed with the `->*` operator, `get_data()`, or `get_heap()`

// VarHeapPointer behaves similar to any other pointer, and is compatible with many of the same operations
// But cannot be compared with any void pointer, only to other similar VarHeapPointers

template <class CommonT>
class VarHeapPointer {
	union {
		uintptr_t _ptr;
		CommonT *_c_ptr;
	};

	constexpr static size_t calc_start(const size_t fixed_size, const size_t heap_align) {
		const size_t over = fixed_size % heap_align;
		return (over == 0) ? fixed_size : (fixed_size + heap_align - over);
	}
	constexpr static size_t calc_total(const size_t fixed_align, const size_t fixed_size, const size_t heap_align, const size_t heap_size) {
		const size_t approx_end = calc_start(fixed_size, heap_align) + heap_size;
		const size_t over = approx_end % fixed_align;
		return (over == 0) ? approx_end : (approx_end + fixed_align - over);
	}
	using data_ptr_t = uint32_t;

public:
	// MemberDataPointers can be used to directly reference a part of the "unknown" heap data
	// They are automatically adjusted at time of creation to account for the memory requirements of both CommonT and HeapT
	// which should result in fewer operations, and means we don't need to keep a reference to the memory requirements of HeapT

	struct MemberDataPointer {
		data_ptr_t _val;
		// _ALWAYS_INLINE_ MemberDataPointer() = default;
		// _ALWAYS_INLINE_ MemberDataPointer() {}
		template <class HeapT, typename MemberT>
		_ALWAYS_INLINE_ constexpr MemberDataPointer(MemberT HeapT::*p_member_pointer) {
			// This constructor makes use of template type deduction to determine the memory requirements of given data-member-pointers
			static_assert(sizeof(p_member_pointer) == sizeof(data_ptr_t), "Member Pointer size mismatch");
			constexpr size_t offset = calc_start(sizeof(CommonT), alignof(HeapT));
			_val = offset + *reinterpret_cast<const data_ptr_t *>(&p_member_pointer);
		}
		constexpr MemberDataPointer(const size_t &p_heap_align, const data_ptr_t &p_member_byte_position) {
			// Otherwise, if the alignment requirements and byte-position of the member-data are known, you can explicitly give those instead
			size_t offset = calc_start(sizeof(CommonT), p_heap_align);
			_val = offset + p_member_byte_position;
		}
	};

	// -- CommonT access

	// Use the -> operator to to access the known common part of the data

	_ALWAYS_INLINE_ CommonT *operator->() {
		return _c_ptr;
	}
	_ALWAYS_INLINE_ const CommonT *operator->() const {
		return _c_ptr;
	}

	// Is implicitly castable to a pointer to CommonT

	// -- HeapT access

	// Use `get_heap` to get a pointer to the "unknown" data
	// For this to work, we need to know its alignment requirements

	_ALWAYS_INLINE_ void *get_heap(const size_t heap_align) {
		size_t offset = calc_start(sizeof(CommonT), heap_align);
		return reinterpret_cast<void *>(_ptr + offset);
	}
	_ALWAYS_INLINE_ const void *get_heap(const size_t heap_align) const {
		size_t offset = calc_start(sizeof(CommonT), heap_align);
		return reinterpret_cast<void *>(_ptr + offset);
	}
	template <class HeapT>
	_ALWAYS_INLINE_ HeapT *get_heap() {
		constexpr size_t offset = calc_start(sizeof(CommonT), alignof(HeapT));
		return reinterpret_cast<HeapT *>(_ptr + offset);
	}
	template <class HeapT>
	_ALWAYS_INLINE_ const HeapT *get_heap() const {
		constexpr size_t offset = calc_start(sizeof(CommonT), alignof(HeapT));
		return reinterpret_cast<HeapT *>(_ptr + offset);
	}
	// (only enabled if we don't need to know the align requirements of HeapT)
	template <typename = std::enable_if_t<sizeof(CommonT) % alignof(max_align_t) == 0>>
	_ALWAYS_INLINE_ void *get_heap_() {
		return reinterpret_cast<void *>(_ptr + sizeof(CommonT));
	}
	template <typename = std::enable_if_t<sizeof(CommonT) % alignof(max_align_t) == 0>>
	_ALWAYS_INLINE_ const void *get_heap_() const {
		return reinterpret_cast<void *>(_ptr + sizeof(CommonT));
	}

	// Use the ->* operator to get a pointer to a specific part of the "unknown" data

	_ALWAYS_INLINE_ void *operator->*(const MemberDataPointer &p_ptr) {
		return reinterpret_cast<void *>(_ptr + p_ptr._val);
	}
	_ALWAYS_INLINE_ const void *operator->*(const MemberDataPointer &p_ptr) const {
		return reinterpret_cast<void *>(_ptr + p_ptr._val);
	}

	// Use `get_data` to get a typed reference to a part of the "unknown" data
	// (equivalent to ->* except that the return is a typed reference instead of a pointer, which may simplify code)

	template <typename T>
	_ALWAYS_INLINE_ T &get_data(const MemberDataPointer &p_ptr) {
		return *reinterpret_cast<T *>(_ptr + p_ptr._val);
	}
	template <typename T>
	_ALWAYS_INLINE_ const T &get_data(const MemberDataPointer &p_ptr) const {
		return *reinterpret_cast<const T *>(_ptr + p_ptr._val);
	}

	// -- Memory functions

	// Use `allocate` to change the internal pointer to a new block of memory large enough to fit both the common known and given "unknown" data
	// Does not perform any operations on that memory (such as copying or initialization; it is up to the caller to know what to do with that memory)

	_ALWAYS_INLINE_ void allocate(const size_t heap_align, const size_t heap_size) {
		size_t totalsize = calc_total(alignof(CommonT), sizeof(CommonT), alignof(max_align_t), heap_size);
		_ptr = (uintptr_t)Memory::alloc_static(totalsize);
	}
	template <class HeapT>
	_ALWAYS_INLINE_ void allocate() {
		constexpr size_t totalsize = calc_total(alignof(CommonT), sizeof(CommonT), alignof(HeapT), sizeof(HeapT));
		_ptr = (uintptr_t)Memory::alloc_static(totalsize);
	}
	// (only enabled if we don't need to know the align requirements of HeapT)
	template <typename = std::enable_if_t<sizeof(CommonT) % alignof(max_align_t) == 0>>
	_ALWAYS_INLINE_ void allocate_(const size_t heap_size) {
		size_t totalsize = calc_total(alignof(CommonT), sizeof(CommonT), alignof(max_align_t), heap_size);
		_ptr = (uintptr_t)Memory::alloc_static(totalsize);
	}

	// Frees the allocated memory and set the internal pointer to nullptr

	_ALWAYS_INLINE_ void free() {
		Memory::free_static((void *)_ptr, false);
		_ptr = 0;
	}

	// Automatic memory allocation constructors

	explicit VarHeapPointer(const size_t heap_align, const size_t heap_size) {
		allocate(heap_align, heap_size);
	}
	// (only enabled if we don't need to know the align requirements of HeapT)
	template <typename = std::enable_if_t<sizeof(CommonT) % alignof(max_align_t) == 0>>
	explicit VarHeapPointer(const size_t heap_size) {
		allocate_(heap_size);
	}

	// -- Pointer functions

	// Can be compared to nullptr (to detect if cleared or not) or another correctly aligned VarHeapPointer

	_ALWAYS_INLINE_ bool operator!=(const std::nullptr_t) const {
		return _ptr != 0;
	}
	_ALWAYS_INLINE_ bool operator==(const std::nullptr_t) const {
		return _ptr == 0;
	}
	_ALWAYS_INLINE_ bool operator!=(const VarHeapPointer<CommonT> p_other) const {
		return _ptr != p_other._ptr;
	}
	_ALWAYS_INLINE_ bool operator==(const VarHeapPointer<CommonT> p_other) const {
		return _ptr == p_other._ptr;
	}

	// Can be casted to bool for checking if not nullptr (like any other pointer)

	_ALWAYS_INLINE_ operator bool() const {
		return _ptr;
	}

	// Can be constructed with nullptr; or assigned to nullptr (which clears the pointer without freeing)

	constexpr VarHeapPointer(const std::nullptr_t) {
		_ptr = 0;
	}
	_ALWAYS_INLINE_ VarHeapPointer<CommonT> &operator=(const std::nullptr_t) {
		_ptr = 0;
		return *this;
	}

	// Ensure that VarHeapPointer is trivial (like any pointer)

	VarHeapPointer() = default;
	VarHeapPointer(const VarHeapPointer &) = default;
	VarHeapPointer(VarHeapPointer &&) = default;
	VarHeapPointer<CommonT> &operator=(const VarHeapPointer<CommonT> &) = default;
	VarHeapPointer<CommonT> &operator=(VarHeapPointer<CommonT> &&) = default;
};

/////////////////////////

// VarHeapObject is useful for allowing an object to store an array of data near to itself in the heap
// Only useful if the array of data doesn't change size after the object is initialized, but is an unknown at compile time or a variable size
// Storing them near to each other in memory can reduce cache misses and improve efficiency

// NOTE: for VarHeapObject to work, derived classes must use a VarHeapData property, and this property should be the last property

struct VarHeapObject {
	using data_ptr_t = uint32_t;

	// VarHeapObject are exclusively stored on the heap
	// They should only ever be passed by pointer
	// Should never be assigned to or copy constructed
	VarHeapObject(const VarHeapObject &) = delete;
	VarHeapObject(VarHeapObject &&) = delete;
	VarHeapObject &operator=(const VarHeapObject &) = delete;
	VarHeapObject &operator=(VarHeapObject &&) = delete;

protected:
	template <class HeapT>
	class VarHeapData {
		friend struct VarHeapObject;

		uint64_t element_count;
		// Prevent automatic construction and deconstruction for HeapT types
		// Ensures that at least enough space for 1 instance is allocated
		union {
			HeapT _first_element;
			uint8_t _data[sizeof(HeapT)];
		} alignas(alignof(HeapT));

	public:
		_FORCE_INLINE_ HeapT &operator[](int p_index) {
			return (&_first_element)[p_index];
		}
		_FORCE_INLINE_ const HeapT &operator[](int p_index) const {
			return (&_first_element)[p_index];
		}
		_FORCE_INLINE_ bool has(const HeapT &p_val) const {
			return &p_val >= begin() && &p_val <= end();
		}
		_FORCE_INLINE_ size_t size() const {
			return element_count;
		}
		_FORCE_INLINE_ const HeapT *begin() const {
			return &_first_element;
		}
		_FORCE_INLINE_ const HeapT *end() const {
			return &(operator[](element_count - 1));
		}

		VarHeapData(const VarHeapData &) = delete;
		VarHeapData(VarHeapData &&) = delete;
		VarHeapData &operator=(const VarHeapData &) = delete;
		VarHeapData &operator=(VarHeapData &&) = delete;

		VarHeapData() {}
		~VarHeapData() {
			if constexpr (!std::is_trivially_destructible_v<HeapT>) {
				const HeapT *en = end();
				HeapT *E = reinterpret_cast<HeapT *>(_data);
				while (E <= en) {
					E->~HeapT();
					E++;
				}
			}
		}
	};

	VarHeapObject() {}

	// Forces initialization of the heap data member through this implementation
	// This ensures that there is a VarHeapData property on any and all derived types and that they are implemented correctly
	template <class HeapT, class DerivedT>
	_FORCE_INLINE_ static void *heap_allocate(VarHeapData<HeapT> DerivedT::*p_member_pointer, std::initializer_list<HeapT> list) {
		static_assert(sizeof(p_member_pointer) == sizeof(data_ptr_t), "Member Pointer size mismatch");
		const data_ptr_t heap_data_position = *reinterpret_cast<data_ptr_t *>(&p_member_pointer);

#if DEV_ENABLED
		const size_t heap_data_size = sizeof(VarHeapData<HeapT>);
		const size_t derived_class_size = sizeof(DerivedT);
		DEV_ASSERT(derived_class_size == heap_data_position + heap_data_size);
#endif

		// Allocate memory for derived type + additional heap data
		size_t to_allocate = sizeof(DerivedT) + sizeof(HeapT) * (list.size() > 1 ? (list.size() - 1) : 1);
		void *ptr = Memory::alloc_static(to_allocate);

		// Initialize the heap data and copy from the initialiser list
		uintptr_t hd_ptr = *reinterpret_cast<uintptr_t *>(&ptr) + heap_data_position;
		VarHeapData<HeapT> &hd = *reinterpret_cast<VarHeapData<HeapT> *>(hd_ptr);
		hd.element_count = list.size();
		int i = 0;
		for (const HeapT &E : list) {
			memnew_placement(&hd[i], HeapT(E));
			i++;
		}

		return ptr;
	}
};
