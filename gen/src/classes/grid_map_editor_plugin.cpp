/**************************************************************************/
/*  grid_map_editor_plugin.cpp                                            */
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

#include <godot_cpp/classes/grid_map_editor_plugin.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/grid_map.hpp>
#include <godot_cpp/variant/vector3i.hpp>

namespace godot {

GridMap *GridMapEditorPlugin::get_current_grid_map() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GridMapEditorPlugin::get_class_static()._native_ptr(), StringName("get_current_grid_map")._native_ptr(), 1184264483);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<GridMap>(_gde_method_bind, _owner);
}

void GridMapEditorPlugin::set_selection(const Vector3i &p_begin, const Vector3i &p_end) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GridMapEditorPlugin::get_class_static()._native_ptr(), StringName("set_selection")._native_ptr(), 3659408297);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_begin, &p_end);
}

void GridMapEditorPlugin::clear_selection() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GridMapEditorPlugin::get_class_static()._native_ptr(), StringName("clear_selection")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

AABB GridMapEditorPlugin::get_selection() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GridMapEditorPlugin::get_class_static()._native_ptr(), StringName("get_selection")._native_ptr(), 1068685055);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AABB()));
	return ::godot::internal::_call_native_mb_ret<AABB>(_gde_method_bind, _owner);
}

bool GridMapEditorPlugin::has_selection() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GridMapEditorPlugin::get_class_static()._native_ptr(), StringName("has_selection")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

Array GridMapEditorPlugin::get_selected_cells() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GridMapEditorPlugin::get_class_static()._native_ptr(), StringName("get_selected_cells")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Array()));
	return ::godot::internal::_call_native_mb_ret<Array>(_gde_method_bind, _owner);
}

void GridMapEditorPlugin::set_selected_palette_item(int32_t p_item) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GridMapEditorPlugin::get_class_static()._native_ptr(), StringName("set_selected_palette_item")._native_ptr(), 998575451);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_item_encoded;
	PtrToArg<int64_t>::encode(p_item, &p_item_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item_encoded);
}

int32_t GridMapEditorPlugin::get_selected_palette_item() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GridMapEditorPlugin::get_class_static()._native_ptr(), StringName("get_selected_palette_item")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

} // namespace godot
