/**************************************************************************/
/*  color_picker.cpp                                                      */
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

#include <godot_cpp/classes/color_picker.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void ColorPicker::set_pick_color(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ColorPicker::get_class_static()._native_ptr(), StringName("set_pick_color")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

Color ColorPicker::get_pick_color() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ColorPicker::get_class_static()._native_ptr(), StringName("get_pick_color")._native_ptr(), 3444240500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void ColorPicker::set_deferred_mode(bool p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ColorPicker::get_class_static()._native_ptr(), StringName("set_deferred_mode")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_mode_encoded;
	PtrToArg<bool>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

bool ColorPicker::is_deferred_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ColorPicker::get_class_static()._native_ptr(), StringName("is_deferred_mode")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void ColorPicker::set_color_mode(ColorPicker::ColorModeType p_color_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ColorPicker::get_class_static()._native_ptr(), StringName("set_color_mode")._native_ptr(), 1579114136);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_color_mode_encoded;
	PtrToArg<int64_t>::encode(p_color_mode, &p_color_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color_mode_encoded);
}

ColorPicker::ColorModeType ColorPicker::get_color_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ColorPicker::get_class_static()._native_ptr(), StringName("get_color_mode")._native_ptr(), 392907674);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (ColorPicker::ColorModeType(0)));
	return (ColorPicker::ColorModeType)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void ColorPicker::set_edit_alpha(bool p_show) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ColorPicker::get_class_static()._native_ptr(), StringName("set_edit_alpha")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_show_encoded;
	PtrToArg<bool>::encode(p_show, &p_show_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_show_encoded);
}

bool ColorPicker::is_editing_alpha() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ColorPicker::get_class_static()._native_ptr(), StringName("is_editing_alpha")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void ColorPicker::set_edit_intensity(bool p_show) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ColorPicker::get_class_static()._native_ptr(), StringName("set_edit_intensity")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_show_encoded;
	PtrToArg<bool>::encode(p_show, &p_show_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_show_encoded);
}

bool ColorPicker::is_editing_intensity() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ColorPicker::get_class_static()._native_ptr(), StringName("is_editing_intensity")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void ColorPicker::set_can_add_swatches(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ColorPicker::get_class_static()._native_ptr(), StringName("set_can_add_swatches")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool ColorPicker::are_swatches_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ColorPicker::get_class_static()._native_ptr(), StringName("are_swatches_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void ColorPicker::set_presets_visible(bool p_visible) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ColorPicker::get_class_static()._native_ptr(), StringName("set_presets_visible")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_visible_encoded;
	PtrToArg<bool>::encode(p_visible, &p_visible_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_visible_encoded);
}

bool ColorPicker::are_presets_visible() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ColorPicker::get_class_static()._native_ptr(), StringName("are_presets_visible")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void ColorPicker::set_modes_visible(bool p_visible) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ColorPicker::get_class_static()._native_ptr(), StringName("set_modes_visible")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_visible_encoded;
	PtrToArg<bool>::encode(p_visible, &p_visible_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_visible_encoded);
}

bool ColorPicker::are_modes_visible() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ColorPicker::get_class_static()._native_ptr(), StringName("are_modes_visible")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void ColorPicker::set_sampler_visible(bool p_visible) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ColorPicker::get_class_static()._native_ptr(), StringName("set_sampler_visible")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_visible_encoded;
	PtrToArg<bool>::encode(p_visible, &p_visible_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_visible_encoded);
}

bool ColorPicker::is_sampler_visible() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ColorPicker::get_class_static()._native_ptr(), StringName("is_sampler_visible")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void ColorPicker::set_sliders_visible(bool p_visible) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ColorPicker::get_class_static()._native_ptr(), StringName("set_sliders_visible")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_visible_encoded;
	PtrToArg<bool>::encode(p_visible, &p_visible_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_visible_encoded);
}

bool ColorPicker::are_sliders_visible() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ColorPicker::get_class_static()._native_ptr(), StringName("are_sliders_visible")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void ColorPicker::set_hex_visible(bool p_visible) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ColorPicker::get_class_static()._native_ptr(), StringName("set_hex_visible")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_visible_encoded;
	PtrToArg<bool>::encode(p_visible, &p_visible_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_visible_encoded);
}

bool ColorPicker::is_hex_visible() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ColorPicker::get_class_static()._native_ptr(), StringName("is_hex_visible")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void ColorPicker::add_preset(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ColorPicker::get_class_static()._native_ptr(), StringName("add_preset")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

void ColorPicker::erase_preset(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ColorPicker::get_class_static()._native_ptr(), StringName("erase_preset")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

PackedColorArray ColorPicker::get_presets() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ColorPicker::get_class_static()._native_ptr(), StringName("get_presets")._native_ptr(), 1392750486);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedColorArray()));
	return ::godot::internal::_call_native_mb_ret<PackedColorArray>(_gde_method_bind, _owner);
}

void ColorPicker::add_recent_preset(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ColorPicker::get_class_static()._native_ptr(), StringName("add_recent_preset")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

void ColorPicker::erase_recent_preset(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ColorPicker::get_class_static()._native_ptr(), StringName("erase_recent_preset")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

PackedColorArray ColorPicker::get_recent_presets() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ColorPicker::get_class_static()._native_ptr(), StringName("get_recent_presets")._native_ptr(), 1392750486);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedColorArray()));
	return ::godot::internal::_call_native_mb_ret<PackedColorArray>(_gde_method_bind, _owner);
}

void ColorPicker::set_picker_shape(ColorPicker::PickerShapeType p_shape) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ColorPicker::get_class_static()._native_ptr(), StringName("set_picker_shape")._native_ptr(), 3981373861);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_shape_encoded;
	PtrToArg<int64_t>::encode(p_shape, &p_shape_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shape_encoded);
}

ColorPicker::PickerShapeType ColorPicker::get_picker_shape() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(ColorPicker::get_class_static()._native_ptr(), StringName("get_picker_shape")._native_ptr(), 1143229889);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (ColorPicker::PickerShapeType(0)));
	return (ColorPicker::PickerShapeType)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

} // namespace godot
