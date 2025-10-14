/**************************************************************************/
/*  openxr_structure.h                                                    */
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

#include "core/object/gdvirtual.gen.inc"
#include "core/object/ref_counted.h"
#include "core/variant/binder_common.h"
#include "openxr_util.h"
#include "util.h"

// Base class for XrStructureType based headers
class OpenXRStructureBase : public RefCounted {
	GDCLASS(OpenXRStructureBase, RefCounted);

public:
	/*
	 * get_header should return a pointer to a proper XrStructureType structure.
	 * The pointer should remain valid as long as this object is not destructed.
	 *
	 * This function should be implemented based on the following template:
	 * void *get_header(void *p_next = nullptr) {
	 *     my_xr_struct.type = XR_TYPE_XYZ;
	 *     if (get_next().is_valid()) {
	 *         my_xr_struct.next = get_next()->get_header(p_next);
	 *     } else {
	 *         my_xr_struct.next = p_next
	 *     }
	 *
	 *     // add further setup of the struct here
	 *
	 *     return &my_xr_struct;
	 * }
	 */
	virtual void *get_header(void *p_next = nullptr);
	virtual XrStructureType get_structure_type();

	void set_next(const Ref<OpenXRStructureBase> p_next);
	Ref<OpenXRStructureBase> get_next() const;

	GDVIRTUAL1R(uint64_t, _get_header, uint64_t);

protected:
	static void _bind_methods();

private:
	Ref<OpenXRStructureBase> next;

	uint64_t _get_structure_type();
};
