/**************************************************************************/
/*  aspect_ratio_container.cpp                                            */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#include <godot_cpp/classes/aspect_ratio_container.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void AspectRatioContainer::set_ratio(float p_ratio) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AspectRatioContainer::get_class_static()._native_ptr(), StringName("set_ratio")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_ratio_encoded;
	PtrToArg<double>::encode(p_ratio, &p_ratio_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_ratio_encoded);
}

float AspectRatioContainer::get_ratio() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AspectRatioContainer::get_class_static()._native_ptr(), StringName("get_ratio")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void AspectRatioContainer::set_stretch_mode(AspectRatioContainer::StretchMode p_stretch_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AspectRatioContainer::get_class_static()._native_ptr(), StringName("set_stretch_mode")._native_ptr(), 1876743467);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_stretch_mode_encoded;
	PtrToArg<int64_t>::encode(p_stretch_mode, &p_stretch_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_stretch_mode_encoded);
}

AspectRatioContainer::StretchMode AspectRatioContainer::get_stretch_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AspectRatioContainer::get_class_static()._native_ptr(), StringName("get_stretch_mode")._native_ptr(), 3416449033);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AspectRatioContainer::StretchMode(0)));
	return (AspectRatioContainer::StretchMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void AspectRatioContainer::set_alignment_horizontal(AspectRatioContainer::AlignmentMode p_alignment_horizontal) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AspectRatioContainer::get_class_static()._native_ptr(), StringName("set_alignment_horizontal")._native_ptr(), 2147829016);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_alignment_horizontal_encoded;
	PtrToArg<int64_t>::encode(p_alignment_horizontal, &p_alignment_horizontal_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_alignment_horizontal_encoded);
}

AspectRatioContainer::AlignmentMode AspectRatioContainer::get_alignment_horizontal() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AspectRatioContainer::get_class_static()._native_ptr(), StringName("get_alignment_horizontal")._native_ptr(), 3838875429);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AspectRatioContainer::AlignmentMode(0)));
	return (AspectRatioContainer::AlignmentMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void AspectRatioContainer::set_alignment_vertical(AspectRatioContainer::AlignmentMode p_alignment_vertical) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AspectRatioContainer::get_class_static()._native_ptr(), StringName("set_alignment_vertical")._native_ptr(), 2147829016);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_alignment_vertical_encoded;
	PtrToArg<int64_t>::encode(p_alignment_vertical, &p_alignment_vertical_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_alignment_vertical_encoded);
}

AspectRatioContainer::AlignmentMode AspectRatioContainer::get_alignment_vertical() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AspectRatioContainer::get_class_static()._native_ptr(), StringName("get_alignment_vertical")._native_ptr(), 3838875429);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AspectRatioContainer::AlignmentMode(0)));
	return (AspectRatioContainer::AlignmentMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

} // namespace godot
