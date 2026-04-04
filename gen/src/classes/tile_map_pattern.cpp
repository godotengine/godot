/**************************************************************************/
/*  tile_map_pattern.cpp                                                  */
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

#include <godot_cpp/classes/tile_map_pattern.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void TileMapPattern::set_cell(const Vector2i &p_coords, int32_t p_source_id, const Vector2i &p_atlas_coords, int32_t p_alternative_tile) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapPattern::get_class_static()._native_ptr(), StringName("set_cell")._native_ptr(), 2224802556);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_source_id_encoded;
	PtrToArg<int64_t>::encode(p_source_id, &p_source_id_encoded);
	int64_t p_alternative_tile_encoded;
	PtrToArg<int64_t>::encode(p_alternative_tile, &p_alternative_tile_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_coords, &p_source_id_encoded, &p_atlas_coords, &p_alternative_tile_encoded);
}

bool TileMapPattern::has_cell(const Vector2i &p_coords) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapPattern::get_class_static()._native_ptr(), StringName("has_cell")._native_ptr(), 3900751641);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_coords);
}

void TileMapPattern::remove_cell(const Vector2i &p_coords, bool p_update_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapPattern::get_class_static()._native_ptr(), StringName("remove_cell")._native_ptr(), 4153096796);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_update_size_encoded;
	PtrToArg<bool>::encode(p_update_size, &p_update_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_coords, &p_update_size_encoded);
}

int32_t TileMapPattern::get_cell_source_id(const Vector2i &p_coords) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapPattern::get_class_static()._native_ptr(), StringName("get_cell_source_id")._native_ptr(), 2485466453);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_coords);
}

Vector2i TileMapPattern::get_cell_atlas_coords(const Vector2i &p_coords) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapPattern::get_class_static()._native_ptr(), StringName("get_cell_atlas_coords")._native_ptr(), 3050897911);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner, &p_coords);
}

int32_t TileMapPattern::get_cell_alternative_tile(const Vector2i &p_coords) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapPattern::get_class_static()._native_ptr(), StringName("get_cell_alternative_tile")._native_ptr(), 2485466453);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_coords);
}

TypedArray<Vector2i> TileMapPattern::get_used_cells() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapPattern::get_class_static()._native_ptr(), StringName("get_used_cells")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Vector2i>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Vector2i>>(_gde_method_bind, _owner);
}

Vector2i TileMapPattern::get_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapPattern::get_class_static()._native_ptr(), StringName("get_size")._native_ptr(), 3690982128);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner);
}

void TileMapPattern::set_size(const Vector2i &p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapPattern::get_class_static()._native_ptr(), StringName("set_size")._native_ptr(), 1130785943);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size);
}

bool TileMapPattern::is_empty() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileMapPattern::get_class_static()._native_ptr(), StringName("is_empty")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

} // namespace godot
