/**************************************************************************/
/*  openxr_structure.cpp                                                  */
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

#include "openxr_structure.h"

#include "core/object/class_db.h"

void OpenXRStructureBase::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_structure_type"), &OpenXRStructureBase::_get_structure_type);

	ClassDB::bind_method(D_METHOD("set_next", "entity"), &OpenXRStructureBase::set_next);
	ClassDB::bind_method(D_METHOD("get_next"), &OpenXRStructureBase::get_next);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "next", PROPERTY_HINT_RESOURCE_TYPE, "OpenXRStructureBase"), "set_next", "get_next");

	GDVIRTUAL_BIND(_get_header, "next");
}

void OpenXRStructureBase::set_next(const Ref<OpenXRStructureBase> p_next) {
	next = p_next;
}

Ref<OpenXRStructureBase> OpenXRStructureBase::get_next() const {
	return next;
}

void *OpenXRStructureBase::get_header(void *p_next) {
	void *n = p_next;
	if (get_next().is_valid()) {
		n = get_next()->get_header(p_next);
	}

	uint64_t pointer = 0;

	if (GDVIRTUAL_CALL(_get_header, (uint64_t)n, pointer)) {
		return reinterpret_cast<void *>(pointer);
	}

	return n;
}

XrStructureType OpenXRStructureBase::get_structure_type() {
	// By default we call get_header to get the structure type so we have a guaranteed implementation.

	// The first member of our header is always the structure type, so this works:
	XrStructureType *header = (XrStructureType *)get_header();
	if (header == nullptr) {
		// Header can return nullptr for valid reasons, so we do not error here!
		return XR_TYPE_UNKNOWN;
	} else {
		return *header;
	}
}

// Return structure type as uint64_t to GDScript
uint64_t OpenXRStructureBase::_get_structure_type() {
	return (uint64_t)get_structure_type();
}
