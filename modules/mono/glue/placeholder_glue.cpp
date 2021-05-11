/*************************************************************************/
/*  placeholder_glue.cpp                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/object/object.h"

#include "../mono_gd/gd_mono_internals.h"
#include "../mono_gd/gd_mono_utils.h"

MonoObject *godot_icall_InteropUtils_unmanaged_get_managed(Object *unmanaged) {
	return GDMonoUtils::unmanaged_get_managed(unmanaged);
}

void godot_icall_InteropUtils_tie_managed_to_unmanaged(MonoObject *managed, Object *unmanaged) {
	GDMonoInternals::tie_managed_to_unmanaged(managed, unmanaged);
}

void godot_register_placeholder_icalls() {
	GDMonoUtils::add_internal_call(
			"Godot.NativeInterop.InteropUtils::internal_unmanaged_get_managed",
			godot_icall_InteropUtils_unmanaged_get_managed);
	GDMonoUtils::add_internal_call(
			"Godot.NativeInterop.InteropUtils::internal_tie_managed_to_unmanaged",
			godot_icall_InteropUtils_tie_managed_to_unmanaged);
}
