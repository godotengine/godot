/**************************************************************************/
/*  memory.cpp                                                            */
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

#include <godot_cpp/core/memory.hpp>

#include <godot_cpp/godot.hpp>

namespace godot {

void *Memory::alloc_static(size_t p_bytes, bool p_pad_align) {
#ifdef DEBUG_ENABLED
	bool prepad = false; // Already pre paded in the engine.
#else
	bool prepad = p_pad_align;
#endif

	void *mem = internal::gdextension_interface_mem_alloc(p_bytes + (prepad ? DATA_OFFSET : 0));
	ERR_FAIL_NULL_V(mem, nullptr);

	if (prepad) {
		uint8_t *s8 = (uint8_t *)mem;
		return s8 + DATA_OFFSET;
	} else {
		return mem;
	}
}

void *Memory::realloc_static(void *p_memory, size_t p_bytes, bool p_pad_align) {
	if (p_memory == nullptr) {
		return alloc_static(p_bytes, p_pad_align);
	} else if (p_bytes == 0) {
		free_static(p_memory, p_pad_align);
		return nullptr;
	}

	uint8_t *mem = (uint8_t *)p_memory;

#ifdef DEBUG_ENABLED
	bool prepad = false; // Already pre paded in the engine.
#else
	bool prepad = p_pad_align;
#endif

	if (prepad) {
		mem -= DATA_OFFSET;
		mem = (uint8_t *)internal::gdextension_interface_mem_realloc(mem, p_bytes + DATA_OFFSET);
		ERR_FAIL_NULL_V(mem, nullptr);
		return mem + DATA_OFFSET;
	} else {
		return (uint8_t *)internal::gdextension_interface_mem_realloc(mem, p_bytes);
	}
}

void Memory::free_static(void *p_ptr, bool p_pad_align) {
	uint8_t *mem = (uint8_t *)p_ptr;

#ifdef DEBUG_ENABLED
	bool prepad = false; // Already pre paded in the engine.
#else
	bool prepad = p_pad_align;
#endif

	if (prepad) {
		mem -= DATA_OFFSET;
	}
	internal::gdextension_interface_mem_free(mem);
}

_GlobalNil::_GlobalNil() {
	left = this;
	right = this;
	parent = this;
}

_GlobalNil _GlobalNilClass::_nil;

} // namespace godot

// p_dummy argument is added to avoid conflicts with the engine functions when both engine and GDExtension are built as a static library on iOS.
void *operator new(size_t p_size, const char *p_dummy, const char *p_description) {
	return godot::Memory::alloc_static(p_size);
}

void *operator new(size_t p_size, const char *p_dummy, void *(*p_allocfunc)(size_t p_size)) {
	return p_allocfunc(p_size);
}

using namespace godot;

#ifdef _MSC_VER
void operator delete(void *p_mem, const char *p_dummy, const char *p_description) {
	ERR_PRINT("Call to placement delete should not happen.");
	CRASH_NOW();
}

void operator delete(void *p_mem, const char *p_dummy, void *(*p_allocfunc)(size_t p_size)) {
	ERR_PRINT("Call to placement delete should not happen.");
	CRASH_NOW();
}

void operator delete(void *p_mem, const char *p_dummy, void *p_pointer, size_t check, const char *p_description) {
	ERR_PRINT("Call to placement delete should not happen.");
	CRASH_NOW();
}

#endif
