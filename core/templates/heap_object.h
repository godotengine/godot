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

namespace {
constexpr size_t calc_start(const size_t fixed_size, const size_t heap_align) {
	const size_t over = fixed_size % heap_align;
	return (over == 0) ? fixed_size : (fixed_size + heap_align - over);
}
constexpr size_t calc_total(const size_t fixed_size, const size_t fixed_align, const size_t heap_size, const size_t heap_align) {
	const size_t approx_end = calc_start(fixed_size, heap_align) + heap_size;
	const size_t over = approx_end % fixed_align;
	return (over == 0) ? approx_end : (approx_end + fixed_align - over);
}
using data_ptr_t = uint32_t;
}; //namespace

// (dev-note: I wrote these implementations myself - I am unsure if there is a well-known name for these patterns; if there is, these should be renamed to match)

/////////////////////////

// VarHeapPointer is useful for storing a known type with an unknown type in the heap
// Behaves a lot like a void pointer to the known type (and is for -> operator)
// While the unknown type can be accessed with the ->* operator

template <class CommonT>
class VarHeapPointer {
	uintptr_t _ptr;

public:
	struct MemberDataPointer {
		data_ptr_t _val;
		_ALWAYS_INLINE_ MemberDataPointer() {}
		template <typename HeapT, typename MemberT>
		_ALWAYS_INLINE_ MemberDataPointer(const MemberT HeapT::*const &p_member_pointer) {
			static_assert(sizeof(p_member_pointer) == sizeof(data_ptr_t), "Member Pointer size mismatch");
			constexpr size_t offset = calc_start(sizeof(CommonT), alignof(HeapT));
			_val = offset + *reinterpret_cast<const data_ptr_t *>(&p_member_pointer);
		}
	};

	_ALWAYS_INLINE_ void *get_heap(const size_t heap_align) {
		return reinterpret_cast<void *>(_ptr + calc_start(sizeof(CommonT), heap_align));
	}
	template <class HeapT>
	_ALWAYS_INLINE_ HeapT *get_heap() {
		constexpr size_t offset = calc_start(sizeof(CommonT), alignof(HeapT));
		return reinterpret_cast<HeapT *>(_ptr + offset);
	}
	template <class HeapT>
	_ALWAYS_INLINE_ HeapT *get_heap(const size_t heap_align) {
		return reinterpret_cast<HeapT *>(_ptr + calc_start(sizeof(CommonT), heap_align));
	}

	_ALWAYS_INLINE_ VarHeapPointer<CommonT> &operator=(void *p_ptr) {
		_ptr = *reinterpret_cast<uintptr_t *>(&p_ptr);
		return *this;
	}
	_ALWAYS_INLINE_ bool operator==(void *p_ptr) {
		return _ptr == *reinterpret_cast<uintptr_t *>(p_ptr);
	}

	_ALWAYS_INLINE_ CommonT *operator->() {
		return reinterpret_cast<CommonT *>(_ptr);
	}
	_ALWAYS_INLINE_ void *operator->*(const MemberDataPointer &p_ptr) {
		return reinterpret_cast<void *>(_ptr + p_ptr._val);
	}
	_ALWAYS_INLINE_ const void *operator->*(const MemberDataPointer &p_ptr) const {
		return reinterpret_cast<void *>(_ptr + p_ptr._val);
	}
	_ALWAYS_INLINE_ operator bool() const {
		return _ptr;
	}
	_ALWAYS_INLINE_ operator void *() const {
		return reinterpret_cast<void *>(_ptr);
	}

	_ALWAYS_INLINE_ void allocate(const size_t heap_align, const size_t heap_size) {
		*this = Memory::alloc_static(calc_total(sizeof(CommonT), alignof(CommonT), heap_align, heap_size));
	}
	template <class HeapT>
	_ALWAYS_INLINE_ void allocate() {
		constexpr size_t totalsize = calc_total(sizeof(CommonT), alignof(CommonT), alignof(HeapT), sizeof(HeapT));
		*this = Memory::alloc_static(totalsize);
	}
	_ALWAYS_INLINE_ void free() {
		Memory::free_static(*this, false);
		*this = nullptr;
	}

	VarHeapPointer() {}
	VarHeapPointer(void *p_ptr) {
		*this = p_ptr;
	}
};

/////////////////////////

// VarHeapObject is useful for allowing an object to store an array of data near to itself in the heap
// Only useful if the array of data doesn't change size after the object is initialised, but is an unknown at compile time or a variable size
// Storing them near to each other in memory can reduce cache misses and improve efficiency

// NOTE: for VarHeapObject to work, derived classes must use a VarHeapData property, and this property should be the last property

struct VarHeapObject {
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

	// Forces initialisation of the heap data member through this implementation
	// This ensures that there is a VarHeapData property on any and all derived types and that they are implemented correctly
	template <class HeapT, class DerivedT>
	_FORCE_INLINE_ static void *heap_allocate(const VarHeapData<HeapT> DerivedT::*const &p_member_pointer, std::initializer_list<HeapT> list) {
		static_assert(sizeof(p_member_pointer) == sizeof(data_ptr_t), "Member Pointer size mismatch");
		const data_ptr_t heap_data_position = *reinterpret_cast<const data_ptr_t *>(&p_member_pointer);
#if DEV_ENABLED
		const size_t heap_data_size = sizeof(VarHeapData<HeapT>);
		const size_t derived_class_size = sizeof(DerivedT);
		DEV_ASSERT(derived_class_size == heap_data_position + heap_data_size);
#endif

		// Allocate memory for derived type + additional heap data
		size_t to_allocate = sizeof(DerivedT) + sizeof(HeapT) * (list.size() > 1 ? (list.size() - 1) : 1);
		void *ptr = Memory::alloc_static(to_allocate);

		// Initialise the heap data and copy from the initialiser list
		uintptr_t hd_ptr = *reinterpret_cast<uintptr_t *>(&ptr) + heap_data_position;
		VarHeapData<HeapT> &hd = *reinterpret_cast<VarHeapData<HeapT> *>(hd_ptr);
		hd.element_count = list.size();
		int i = 0;
		for (const HeapT &E : list) {
			new (&hd[i]) HeapT(E);
			i++;
		}

		return ptr;
	}
};