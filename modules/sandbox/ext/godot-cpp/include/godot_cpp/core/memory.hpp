/**************************************************************************/
/*  memory.hpp                                                            */
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

#include <cstddef>
#include <cstdint>

#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/core/error_macros.hpp>
#include <godot_cpp/godot.hpp>

#include <type_traits>

// p_dummy argument is added to avoid conflicts with the engine functions when both engine and GDExtension are built as a static library on iOS.
void *operator new(size_t p_size, const char *p_dummy, const char *p_description); ///< operator new that takes a description and uses MemoryStaticPool
void *operator new(size_t p_size, const char *p_dummy, void *(*p_allocfunc)(size_t p_size)); ///< operator new that takes a description and uses MemoryStaticPool
void *operator new(size_t p_size, const char *p_dummy, void *p_pointer, size_t check, const char *p_description); ///< operator new that takes a description and uses a pointer to the preallocated memory

_ALWAYS_INLINE_ void *operator new(size_t p_size, const char *p_dummy, void *p_pointer, size_t check, const char *p_description) {
	return p_pointer;
}

#ifdef _MSC_VER
// When compiling with VC++ 2017, the above declarations of placement new generate many irrelevant warnings (C4291).
// The purpose of the following definitions is to muffle these warnings, not to provide a usable implementation of placement delete.
void operator delete(void *p_mem, const char *p_dummy, const char *p_description);
void operator delete(void *p_mem, const char *p_dummy, void *(*p_allocfunc)(size_t p_size));
void operator delete(void *p_mem, const char *p_dummy, void *p_pointer, size_t check, const char *p_description);
#endif

namespace godot {

class Wrapped;

class Memory {
	Memory();

public:
	// Alignment:  ↓ max_align_t        ↓ uint64_t          ↓ max_align_t
	//             ┌─────────────────┬──┬────────────────┬──┬───────────...
	//             │ uint64_t        │░░│ uint64_t       │░░│ T[]
	//             │ alloc size      │░░│ element count  │░░│ data
	//             └─────────────────┴──┴────────────────┴──┴───────────...
	// Offset:     ↑ SIZE_OFFSET        ↑ ELEMENT_OFFSET    ↑ DATA_OFFSET
	// Note: "alloc size" is used and set by the engine and is never accessed or changed for the extension.

	static constexpr size_t SIZE_OFFSET = 0;
	static constexpr size_t ELEMENT_OFFSET = ((SIZE_OFFSET + sizeof(uint64_t)) % alignof(uint64_t) == 0) ? (SIZE_OFFSET + sizeof(uint64_t)) : ((SIZE_OFFSET + sizeof(uint64_t)) + alignof(uint64_t) - ((SIZE_OFFSET + sizeof(uint64_t)) % alignof(uint64_t)));
	static constexpr size_t DATA_OFFSET = ((ELEMENT_OFFSET + sizeof(uint64_t)) % alignof(max_align_t) == 0) ? (ELEMENT_OFFSET + sizeof(uint64_t)) : ((ELEMENT_OFFSET + sizeof(uint64_t)) + alignof(max_align_t) - ((ELEMENT_OFFSET + sizeof(uint64_t)) % alignof(max_align_t)));

	static void *alloc_static(size_t p_bytes, bool p_pad_align = false);
	static void *realloc_static(void *p_memory, size_t p_bytes, bool p_pad_align = false);
	static void free_static(void *p_ptr, bool p_pad_align = false);
};

template <typename T, std::enable_if_t<!std::is_base_of<::godot::Wrapped, T>::value, bool> = true>
_ALWAYS_INLINE_ void _pre_initialize() {}

_ALWAYS_INLINE_ void postinitialize_handler(void *) {}

template <typename T>
_ALWAYS_INLINE_ T *_post_initialize(T *p_obj) {
	postinitialize_handler(p_obj);
	return p_obj;
}

#define memalloc(m_size) ::godot::Memory::alloc_static(m_size)
#define memrealloc(m_mem, m_size) ::godot::Memory::realloc_static(m_mem, m_size)
#define memfree(m_mem) ::godot::Memory::free_static(m_mem)

#define memnew(m_class) (::godot::_pre_initialize<std::remove_pointer_t<decltype(new ("", "") m_class)>>(), ::godot::_post_initialize(new ("", "") m_class))

#define memnew_allocator(m_class, m_allocator) (::godot::_pre_initialize<std::remove_pointer_t<decltype(new ("", "") m_class)>>(), ::godot::_post_initialize(new ("", m_allocator::alloc) m_class))
#define memnew_placement(m_placement, m_class) (::godot::_pre_initialize<std::remove_pointer_t<decltype(new ("", "") m_class)>>(), ::godot::_post_initialize(new ("", m_placement, sizeof(m_class), "") m_class))

template <typename T>
void memdelete(T *p_class, typename std::enable_if<!std::is_base_of_v<godot::Wrapped, T>>::type * = nullptr) {
	if constexpr (!std::is_trivially_destructible_v<T>) {
		p_class->~T();
	}

	Memory::free_static(p_class);
}

template <typename T, std::enable_if_t<std::is_base_of_v<godot::Wrapped, T>, bool> = true>
void memdelete(T *p_class) {
	godot::internal::gdextension_interface_object_destroy(p_class->_owner);
}

template <typename T, typename A>
void memdelete_allocator(T *p_class) {
	if constexpr (!std::is_trivially_destructible_v<T>) {
		p_class->~T();
	}

	A::free(p_class);
}

class DefaultAllocator {
public:
	_ALWAYS_INLINE_ static void *alloc(size_t p_memory) { return Memory::alloc_static(p_memory); }
	_ALWAYS_INLINE_ static void free(void *p_ptr) { Memory::free_static(p_ptr); }
};

template <typename T>
class DefaultTypedAllocator {
public:
	template <typename... Args>
	_ALWAYS_INLINE_ T *new_allocation(const Args &&...p_args) { return memnew(T(p_args...)); }
	_ALWAYS_INLINE_ void delete_allocation(T *p_allocation) { memdelete(p_allocation); }
};

#define memnew_arr(m_class, m_count) memnew_arr_template<m_class>(m_count)

_FORCE_INLINE_ uint64_t *_get_element_count_ptr(uint8_t *p_ptr) {
	return (uint64_t *)(p_ptr - Memory::DATA_OFFSET + Memory::ELEMENT_OFFSET);
}

template <typename T>
T *memnew_arr_template(size_t p_elements, const char *p_descr = "") {
	if (p_elements == 0) {
		return nullptr;
	}
	/** overloading operator new[] cannot be done , because it may not return the real allocated address (it may pad the 'element count' before the actual array). Because of that, it must be done by hand. This is the
	same strategy used by std::vector, and the Vector class, so it should be safe.*/

	size_t len = sizeof(T) * p_elements;
	uint8_t *mem = (uint8_t *)Memory::alloc_static(len, true);
	T *failptr = nullptr; // Get rid of a warning.
	ERR_FAIL_NULL_V(mem, failptr);

	uint64_t *_elem_count_ptr = _get_element_count_ptr(mem);
	*(_elem_count_ptr) = p_elements;

	if constexpr (!std::is_trivially_destructible_v<T>) {
		T *elems = (T *)mem;

		/* call operator new */
		for (size_t i = 0; i < p_elements; i++) {
			new ("", &elems[i], sizeof(T), p_descr) T;
		}
	}

	return (T *)mem;
}

template <typename T>
size_t memarr_len(const T *p_class) {
	uint8_t *ptr = (uint8_t *)p_class;
	uint64_t *_elem_count_ptr = _get_element_count_ptr(ptr);
	return *(_elem_count_ptr);
}

template <typename T>
void memdelete_arr(T *p_class) {
	uint8_t *ptr = (uint8_t *)p_class;

	if constexpr (!std::is_trivially_destructible_v<T>) {
		uint64_t *_elem_count_ptr = _get_element_count_ptr(ptr);
		uint64_t elem_count = *(_elem_count_ptr);

		for (uint64_t i = 0; i < elem_count; i++) {
			p_class[i].~T();
		}
	}

	Memory::free_static(ptr, true);
}

struct _GlobalNil {
	int color = 1;
	_GlobalNil *right;
	_GlobalNil *left;
	_GlobalNil *parent;

	_GlobalNil();
};

struct _GlobalNilClass {
	static _GlobalNil _nil;
};

} // namespace godot
