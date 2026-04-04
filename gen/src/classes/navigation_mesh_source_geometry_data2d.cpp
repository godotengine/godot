/**************************************************************************/
/*  navigation_mesh_source_geometry_data2d.cpp                            */
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

#include <godot_cpp/classes/navigation_mesh_source_geometry_data2d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void NavigationMeshSourceGeometryData2D::clear() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshSourceGeometryData2D::get_class_static()._native_ptr(), StringName("clear")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

bool NavigationMeshSourceGeometryData2D::has_data() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshSourceGeometryData2D::get_class_static()._native_ptr(), StringName("has_data")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void NavigationMeshSourceGeometryData2D::set_traversable_outlines(const TypedArray<PackedVector2Array> &p_traversable_outlines) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshSourceGeometryData2D::get_class_static()._native_ptr(), StringName("set_traversable_outlines")._native_ptr(), 381264803);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_traversable_outlines);
}

TypedArray<PackedVector2Array> NavigationMeshSourceGeometryData2D::get_traversable_outlines() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshSourceGeometryData2D::get_class_static()._native_ptr(), StringName("get_traversable_outlines")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<PackedVector2Array>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<PackedVector2Array>>(_gde_method_bind, _owner);
}

void NavigationMeshSourceGeometryData2D::set_obstruction_outlines(const TypedArray<PackedVector2Array> &p_obstruction_outlines) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshSourceGeometryData2D::get_class_static()._native_ptr(), StringName("set_obstruction_outlines")._native_ptr(), 381264803);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_obstruction_outlines);
}

TypedArray<PackedVector2Array> NavigationMeshSourceGeometryData2D::get_obstruction_outlines() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshSourceGeometryData2D::get_class_static()._native_ptr(), StringName("get_obstruction_outlines")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<PackedVector2Array>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<PackedVector2Array>>(_gde_method_bind, _owner);
}

void NavigationMeshSourceGeometryData2D::append_traversable_outlines(const TypedArray<PackedVector2Array> &p_traversable_outlines) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshSourceGeometryData2D::get_class_static()._native_ptr(), StringName("append_traversable_outlines")._native_ptr(), 381264803);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_traversable_outlines);
}

void NavigationMeshSourceGeometryData2D::append_obstruction_outlines(const TypedArray<PackedVector2Array> &p_obstruction_outlines) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshSourceGeometryData2D::get_class_static()._native_ptr(), StringName("append_obstruction_outlines")._native_ptr(), 381264803);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_obstruction_outlines);
}

void NavigationMeshSourceGeometryData2D::add_traversable_outline(const PackedVector2Array &p_shape_outline) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshSourceGeometryData2D::get_class_static()._native_ptr(), StringName("add_traversable_outline")._native_ptr(), 1509147220);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shape_outline);
}

void NavigationMeshSourceGeometryData2D::add_obstruction_outline(const PackedVector2Array &p_shape_outline) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshSourceGeometryData2D::get_class_static()._native_ptr(), StringName("add_obstruction_outline")._native_ptr(), 1509147220);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shape_outline);
}

void NavigationMeshSourceGeometryData2D::merge(const Ref<NavigationMeshSourceGeometryData2D> &p_other_geometry) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshSourceGeometryData2D::get_class_static()._native_ptr(), StringName("merge")._native_ptr(), 742424872);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_other_geometry != nullptr ? &p_other_geometry->_owner : nullptr));
}

void NavigationMeshSourceGeometryData2D::add_projected_obstruction(const PackedVector2Array &p_vertices, bool p_carve) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshSourceGeometryData2D::get_class_static()._native_ptr(), StringName("add_projected_obstruction")._native_ptr(), 3882407395);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_carve_encoded;
	PtrToArg<bool>::encode(p_carve, &p_carve_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_vertices, &p_carve_encoded);
}

void NavigationMeshSourceGeometryData2D::clear_projected_obstructions() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshSourceGeometryData2D::get_class_static()._native_ptr(), StringName("clear_projected_obstructions")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void NavigationMeshSourceGeometryData2D::set_projected_obstructions(const Array &p_projected_obstructions) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshSourceGeometryData2D::get_class_static()._native_ptr(), StringName("set_projected_obstructions")._native_ptr(), 381264803);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_projected_obstructions);
}

Array NavigationMeshSourceGeometryData2D::get_projected_obstructions() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshSourceGeometryData2D::get_class_static()._native_ptr(), StringName("get_projected_obstructions")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Array()));
	return ::godot::internal::_call_native_mb_ret<Array>(_gde_method_bind, _owner);
}

Rect2 NavigationMeshSourceGeometryData2D::get_bounds() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMeshSourceGeometryData2D::get_class_static()._native_ptr(), StringName("get_bounds")._native_ptr(), 3248174);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Rect2()));
	return ::godot::internal::_call_native_mb_ret<Rect2>(_gde_method_bind, _owner);
}

} // namespace godot
