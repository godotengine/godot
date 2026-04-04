/**************************************************************************/
/*  tile_set.cpp                                                          */
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

#include <godot_cpp/classes/tile_set.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/physics_material.hpp>
#include <godot_cpp/classes/tile_map_pattern.hpp>
#include <godot_cpp/classes/tile_set_source.hpp>

namespace godot {

int32_t TileSet::get_next_source_id() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("get_next_source_id")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t TileSet::add_source(const Ref<TileSetSource> &p_source, int32_t p_atlas_source_id_override) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("add_source")._native_ptr(), 1059186179);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_atlas_source_id_override_encoded;
	PtrToArg<int64_t>::encode(p_atlas_source_id_override, &p_atlas_source_id_override_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, (p_source != nullptr ? &p_source->_owner : nullptr), &p_atlas_source_id_override_encoded);
}

void TileSet::remove_source(int32_t p_source_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("remove_source")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_source_id_encoded;
	PtrToArg<int64_t>::encode(p_source_id, &p_source_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_source_id_encoded);
}

void TileSet::set_source_id(int32_t p_source_id, int32_t p_new_source_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("set_source_id")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_source_id_encoded;
	PtrToArg<int64_t>::encode(p_source_id, &p_source_id_encoded);
	int64_t p_new_source_id_encoded;
	PtrToArg<int64_t>::encode(p_new_source_id, &p_new_source_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_source_id_encoded, &p_new_source_id_encoded);
}

int32_t TileSet::get_source_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("get_source_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t TileSet::get_source_id(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("get_source_id")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded);
}

bool TileSet::has_source(int32_t p_source_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("has_source")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_source_id_encoded;
	PtrToArg<int64_t>::encode(p_source_id, &p_source_id_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_source_id_encoded);
}

Ref<TileSetSource> TileSet::get_source(int32_t p_source_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("get_source")._native_ptr(), 1763540252);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<TileSetSource>()));
	int64_t p_source_id_encoded;
	PtrToArg<int64_t>::encode(p_source_id, &p_source_id_encoded);
	return Ref<TileSetSource>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<TileSetSource>(_gde_method_bind, _owner, &p_source_id_encoded));
}

void TileSet::set_tile_shape(TileSet::TileShape p_shape) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("set_tile_shape")._native_ptr(), 2131427112);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_shape_encoded;
	PtrToArg<int64_t>::encode(p_shape, &p_shape_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shape_encoded);
}

TileSet::TileShape TileSet::get_tile_shape() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("get_tile_shape")._native_ptr(), 716918169);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TileSet::TileShape(0)));
	return (TileSet::TileShape)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TileSet::set_tile_layout(TileSet::TileLayout p_layout) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("set_tile_layout")._native_ptr(), 1071216679);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layout_encoded;
	PtrToArg<int64_t>::encode(p_layout, &p_layout_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layout_encoded);
}

TileSet::TileLayout TileSet::get_tile_layout() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("get_tile_layout")._native_ptr(), 194628839);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TileSet::TileLayout(0)));
	return (TileSet::TileLayout)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TileSet::set_tile_offset_axis(TileSet::TileOffsetAxis p_alignment) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("set_tile_offset_axis")._native_ptr(), 3300198521);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_alignment_encoded;
	PtrToArg<int64_t>::encode(p_alignment, &p_alignment_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_alignment_encoded);
}

TileSet::TileOffsetAxis TileSet::get_tile_offset_axis() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("get_tile_offset_axis")._native_ptr(), 762494114);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TileSet::TileOffsetAxis(0)));
	return (TileSet::TileOffsetAxis)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TileSet::set_tile_size(const Vector2i &p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("set_tile_size")._native_ptr(), 1130785943);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size);
}

Vector2i TileSet::get_tile_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("get_tile_size")._native_ptr(), 3690982128);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner);
}

void TileSet::set_uv_clipping(bool p_uv_clipping) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("set_uv_clipping")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_uv_clipping_encoded;
	PtrToArg<bool>::encode(p_uv_clipping, &p_uv_clipping_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_uv_clipping_encoded);
}

bool TileSet::is_uv_clipping() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("is_uv_clipping")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

int32_t TileSet::get_occlusion_layers_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("get_occlusion_layers_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TileSet::add_occlusion_layer(int32_t p_to_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("add_occlusion_layer")._native_ptr(), 1025054187);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_to_position_encoded;
	PtrToArg<int64_t>::encode(p_to_position, &p_to_position_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_to_position_encoded);
}

void TileSet::move_occlusion_layer(int32_t p_layer_index, int32_t p_to_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("move_occlusion_layer")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_index_encoded;
	PtrToArg<int64_t>::encode(p_layer_index, &p_layer_index_encoded);
	int64_t p_to_position_encoded;
	PtrToArg<int64_t>::encode(p_to_position, &p_to_position_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_index_encoded, &p_to_position_encoded);
}

void TileSet::remove_occlusion_layer(int32_t p_layer_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("remove_occlusion_layer")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_index_encoded;
	PtrToArg<int64_t>::encode(p_layer_index, &p_layer_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_index_encoded);
}

void TileSet::set_occlusion_layer_light_mask(int32_t p_layer_index, int32_t p_light_mask) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("set_occlusion_layer_light_mask")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_index_encoded;
	PtrToArg<int64_t>::encode(p_layer_index, &p_layer_index_encoded);
	int64_t p_light_mask_encoded;
	PtrToArg<int64_t>::encode(p_light_mask, &p_light_mask_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_index_encoded, &p_light_mask_encoded);
}

int32_t TileSet::get_occlusion_layer_light_mask(int32_t p_layer_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("get_occlusion_layer_light_mask")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_layer_index_encoded;
	PtrToArg<int64_t>::encode(p_layer_index, &p_layer_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_layer_index_encoded);
}

void TileSet::set_occlusion_layer_sdf_collision(int32_t p_layer_index, bool p_sdf_collision) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("set_occlusion_layer_sdf_collision")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_index_encoded;
	PtrToArg<int64_t>::encode(p_layer_index, &p_layer_index_encoded);
	int8_t p_sdf_collision_encoded;
	PtrToArg<bool>::encode(p_sdf_collision, &p_sdf_collision_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_index_encoded, &p_sdf_collision_encoded);
}

bool TileSet::get_occlusion_layer_sdf_collision(int32_t p_layer_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("get_occlusion_layer_sdf_collision")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_layer_index_encoded;
	PtrToArg<int64_t>::encode(p_layer_index, &p_layer_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_layer_index_encoded);
}

int32_t TileSet::get_physics_layers_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("get_physics_layers_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TileSet::add_physics_layer(int32_t p_to_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("add_physics_layer")._native_ptr(), 1025054187);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_to_position_encoded;
	PtrToArg<int64_t>::encode(p_to_position, &p_to_position_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_to_position_encoded);
}

void TileSet::move_physics_layer(int32_t p_layer_index, int32_t p_to_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("move_physics_layer")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_index_encoded;
	PtrToArg<int64_t>::encode(p_layer_index, &p_layer_index_encoded);
	int64_t p_to_position_encoded;
	PtrToArg<int64_t>::encode(p_to_position, &p_to_position_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_index_encoded, &p_to_position_encoded);
}

void TileSet::remove_physics_layer(int32_t p_layer_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("remove_physics_layer")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_index_encoded;
	PtrToArg<int64_t>::encode(p_layer_index, &p_layer_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_index_encoded);
}

void TileSet::set_physics_layer_collision_layer(int32_t p_layer_index, uint32_t p_layer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("set_physics_layer_collision_layer")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_index_encoded;
	PtrToArg<int64_t>::encode(p_layer_index, &p_layer_index_encoded);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_index_encoded, &p_layer_encoded);
}

uint32_t TileSet::get_physics_layer_collision_layer(int32_t p_layer_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("get_physics_layer_collision_layer")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_layer_index_encoded;
	PtrToArg<int64_t>::encode(p_layer_index, &p_layer_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_layer_index_encoded);
}

void TileSet::set_physics_layer_collision_mask(int32_t p_layer_index, uint32_t p_mask) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("set_physics_layer_collision_mask")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_index_encoded;
	PtrToArg<int64_t>::encode(p_layer_index, &p_layer_index_encoded);
	int64_t p_mask_encoded;
	PtrToArg<int64_t>::encode(p_mask, &p_mask_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_index_encoded, &p_mask_encoded);
}

uint32_t TileSet::get_physics_layer_collision_mask(int32_t p_layer_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("get_physics_layer_collision_mask")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_layer_index_encoded;
	PtrToArg<int64_t>::encode(p_layer_index, &p_layer_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_layer_index_encoded);
}

void TileSet::set_physics_layer_collision_priority(int32_t p_layer_index, float p_priority) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("set_physics_layer_collision_priority")._native_ptr(), 1602489585);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_index_encoded;
	PtrToArg<int64_t>::encode(p_layer_index, &p_layer_index_encoded);
	double p_priority_encoded;
	PtrToArg<double>::encode(p_priority, &p_priority_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_index_encoded, &p_priority_encoded);
}

float TileSet::get_physics_layer_collision_priority(int32_t p_layer_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("get_physics_layer_collision_priority")._native_ptr(), 2339986948);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_layer_index_encoded;
	PtrToArg<int64_t>::encode(p_layer_index, &p_layer_index_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_layer_index_encoded);
}

void TileSet::set_physics_layer_physics_material(int32_t p_layer_index, const Ref<PhysicsMaterial> &p_physics_material) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("set_physics_layer_physics_material")._native_ptr(), 1018687357);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_index_encoded;
	PtrToArg<int64_t>::encode(p_layer_index, &p_layer_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_index_encoded, (p_physics_material != nullptr ? &p_physics_material->_owner : nullptr));
}

Ref<PhysicsMaterial> TileSet::get_physics_layer_physics_material(int32_t p_layer_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("get_physics_layer_physics_material")._native_ptr(), 788318639);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<PhysicsMaterial>()));
	int64_t p_layer_index_encoded;
	PtrToArg<int64_t>::encode(p_layer_index, &p_layer_index_encoded);
	return Ref<PhysicsMaterial>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<PhysicsMaterial>(_gde_method_bind, _owner, &p_layer_index_encoded));
}

int32_t TileSet::get_terrain_sets_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("get_terrain_sets_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TileSet::add_terrain_set(int32_t p_to_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("add_terrain_set")._native_ptr(), 1025054187);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_to_position_encoded;
	PtrToArg<int64_t>::encode(p_to_position, &p_to_position_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_to_position_encoded);
}

void TileSet::move_terrain_set(int32_t p_terrain_set, int32_t p_to_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("move_terrain_set")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_terrain_set_encoded;
	PtrToArg<int64_t>::encode(p_terrain_set, &p_terrain_set_encoded);
	int64_t p_to_position_encoded;
	PtrToArg<int64_t>::encode(p_to_position, &p_to_position_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_terrain_set_encoded, &p_to_position_encoded);
}

void TileSet::remove_terrain_set(int32_t p_terrain_set) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("remove_terrain_set")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_terrain_set_encoded;
	PtrToArg<int64_t>::encode(p_terrain_set, &p_terrain_set_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_terrain_set_encoded);
}

void TileSet::set_terrain_set_mode(int32_t p_terrain_set, TileSet::TerrainMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("set_terrain_set_mode")._native_ptr(), 3943003916);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_terrain_set_encoded;
	PtrToArg<int64_t>::encode(p_terrain_set, &p_terrain_set_encoded);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_terrain_set_encoded, &p_mode_encoded);
}

TileSet::TerrainMode TileSet::get_terrain_set_mode(int32_t p_terrain_set) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("get_terrain_set_mode")._native_ptr(), 2084469411);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TileSet::TerrainMode(0)));
	int64_t p_terrain_set_encoded;
	PtrToArg<int64_t>::encode(p_terrain_set, &p_terrain_set_encoded);
	return (TileSet::TerrainMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_terrain_set_encoded);
}

int32_t TileSet::get_terrains_count(int32_t p_terrain_set) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("get_terrains_count")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_terrain_set_encoded;
	PtrToArg<int64_t>::encode(p_terrain_set, &p_terrain_set_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_terrain_set_encoded);
}

void TileSet::add_terrain(int32_t p_terrain_set, int32_t p_to_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("add_terrain")._native_ptr(), 1230568737);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_terrain_set_encoded;
	PtrToArg<int64_t>::encode(p_terrain_set, &p_terrain_set_encoded);
	int64_t p_to_position_encoded;
	PtrToArg<int64_t>::encode(p_to_position, &p_to_position_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_terrain_set_encoded, &p_to_position_encoded);
}

void TileSet::move_terrain(int32_t p_terrain_set, int32_t p_terrain_index, int32_t p_to_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("move_terrain")._native_ptr(), 1649997291);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_terrain_set_encoded;
	PtrToArg<int64_t>::encode(p_terrain_set, &p_terrain_set_encoded);
	int64_t p_terrain_index_encoded;
	PtrToArg<int64_t>::encode(p_terrain_index, &p_terrain_index_encoded);
	int64_t p_to_position_encoded;
	PtrToArg<int64_t>::encode(p_to_position, &p_to_position_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_terrain_set_encoded, &p_terrain_index_encoded, &p_to_position_encoded);
}

void TileSet::remove_terrain(int32_t p_terrain_set, int32_t p_terrain_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("remove_terrain")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_terrain_set_encoded;
	PtrToArg<int64_t>::encode(p_terrain_set, &p_terrain_set_encoded);
	int64_t p_terrain_index_encoded;
	PtrToArg<int64_t>::encode(p_terrain_index, &p_terrain_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_terrain_set_encoded, &p_terrain_index_encoded);
}

void TileSet::set_terrain_name(int32_t p_terrain_set, int32_t p_terrain_index, const String &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("set_terrain_name")._native_ptr(), 2285447957);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_terrain_set_encoded;
	PtrToArg<int64_t>::encode(p_terrain_set, &p_terrain_set_encoded);
	int64_t p_terrain_index_encoded;
	PtrToArg<int64_t>::encode(p_terrain_index, &p_terrain_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_terrain_set_encoded, &p_terrain_index_encoded, &p_name);
}

String TileSet::get_terrain_name(int32_t p_terrain_set, int32_t p_terrain_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("get_terrain_name")._native_ptr(), 1391810591);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_terrain_set_encoded;
	PtrToArg<int64_t>::encode(p_terrain_set, &p_terrain_set_encoded);
	int64_t p_terrain_index_encoded;
	PtrToArg<int64_t>::encode(p_terrain_index, &p_terrain_index_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_terrain_set_encoded, &p_terrain_index_encoded);
}

void TileSet::set_terrain_color(int32_t p_terrain_set, int32_t p_terrain_index, const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("set_terrain_color")._native_ptr(), 3733378741);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_terrain_set_encoded;
	PtrToArg<int64_t>::encode(p_terrain_set, &p_terrain_set_encoded);
	int64_t p_terrain_index_encoded;
	PtrToArg<int64_t>::encode(p_terrain_index, &p_terrain_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_terrain_set_encoded, &p_terrain_index_encoded, &p_color);
}

Color TileSet::get_terrain_color(int32_t p_terrain_set, int32_t p_terrain_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("get_terrain_color")._native_ptr(), 2165839948);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	int64_t p_terrain_set_encoded;
	PtrToArg<int64_t>::encode(p_terrain_set, &p_terrain_set_encoded);
	int64_t p_terrain_index_encoded;
	PtrToArg<int64_t>::encode(p_terrain_index, &p_terrain_index_encoded);
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner, &p_terrain_set_encoded, &p_terrain_index_encoded);
}

int32_t TileSet::get_navigation_layers_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("get_navigation_layers_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TileSet::add_navigation_layer(int32_t p_to_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("add_navigation_layer")._native_ptr(), 1025054187);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_to_position_encoded;
	PtrToArg<int64_t>::encode(p_to_position, &p_to_position_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_to_position_encoded);
}

void TileSet::move_navigation_layer(int32_t p_layer_index, int32_t p_to_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("move_navigation_layer")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_index_encoded;
	PtrToArg<int64_t>::encode(p_layer_index, &p_layer_index_encoded);
	int64_t p_to_position_encoded;
	PtrToArg<int64_t>::encode(p_to_position, &p_to_position_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_index_encoded, &p_to_position_encoded);
}

void TileSet::remove_navigation_layer(int32_t p_layer_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("remove_navigation_layer")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_index_encoded;
	PtrToArg<int64_t>::encode(p_layer_index, &p_layer_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_index_encoded);
}

void TileSet::set_navigation_layer_layers(int32_t p_layer_index, uint32_t p_layers) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("set_navigation_layer_layers")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_index_encoded;
	PtrToArg<int64_t>::encode(p_layer_index, &p_layer_index_encoded);
	int64_t p_layers_encoded;
	PtrToArg<int64_t>::encode(p_layers, &p_layers_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_index_encoded, &p_layers_encoded);
}

uint32_t TileSet::get_navigation_layer_layers(int32_t p_layer_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("get_navigation_layer_layers")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_layer_index_encoded;
	PtrToArg<int64_t>::encode(p_layer_index, &p_layer_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_layer_index_encoded);
}

void TileSet::set_navigation_layer_layer_value(int32_t p_layer_index, int32_t p_layer_number, bool p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("set_navigation_layer_layer_value")._native_ptr(), 1383440665);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_index_encoded;
	PtrToArg<int64_t>::encode(p_layer_index, &p_layer_index_encoded);
	int64_t p_layer_number_encoded;
	PtrToArg<int64_t>::encode(p_layer_number, &p_layer_number_encoded);
	int8_t p_value_encoded;
	PtrToArg<bool>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_index_encoded, &p_layer_number_encoded, &p_value_encoded);
}

bool TileSet::get_navigation_layer_layer_value(int32_t p_layer_index, int32_t p_layer_number) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("get_navigation_layer_layer_value")._native_ptr(), 2522259332);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_layer_index_encoded;
	PtrToArg<int64_t>::encode(p_layer_index, &p_layer_index_encoded);
	int64_t p_layer_number_encoded;
	PtrToArg<int64_t>::encode(p_layer_number, &p_layer_number_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_layer_index_encoded, &p_layer_number_encoded);
}

int32_t TileSet::get_custom_data_layers_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("get_custom_data_layers_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void TileSet::add_custom_data_layer(int32_t p_to_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("add_custom_data_layer")._native_ptr(), 1025054187);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_to_position_encoded;
	PtrToArg<int64_t>::encode(p_to_position, &p_to_position_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_to_position_encoded);
}

void TileSet::move_custom_data_layer(int32_t p_layer_index, int32_t p_to_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("move_custom_data_layer")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_index_encoded;
	PtrToArg<int64_t>::encode(p_layer_index, &p_layer_index_encoded);
	int64_t p_to_position_encoded;
	PtrToArg<int64_t>::encode(p_to_position, &p_to_position_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_index_encoded, &p_to_position_encoded);
}

void TileSet::remove_custom_data_layer(int32_t p_layer_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("remove_custom_data_layer")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_index_encoded;
	PtrToArg<int64_t>::encode(p_layer_index, &p_layer_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_index_encoded);
}

int32_t TileSet::get_custom_data_layer_by_name(const String &p_layer_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("get_custom_data_layer_by_name")._native_ptr(), 1321353865);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_layer_name);
}

void TileSet::set_custom_data_layer_name(int32_t p_layer_index, const String &p_layer_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("set_custom_data_layer_name")._native_ptr(), 501894301);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_index_encoded;
	PtrToArg<int64_t>::encode(p_layer_index, &p_layer_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_index_encoded, &p_layer_name);
}

bool TileSet::has_custom_data_layer_by_name(const String &p_layer_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("has_custom_data_layer_by_name")._native_ptr(), 3927539163);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_layer_name);
}

String TileSet::get_custom_data_layer_name(int32_t p_layer_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("get_custom_data_layer_name")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_layer_index_encoded;
	PtrToArg<int64_t>::encode(p_layer_index, &p_layer_index_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_layer_index_encoded);
}

void TileSet::set_custom_data_layer_type(int32_t p_layer_index, Variant::Type p_layer_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("set_custom_data_layer_type")._native_ptr(), 3492912874);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_index_encoded;
	PtrToArg<int64_t>::encode(p_layer_index, &p_layer_index_encoded);
	int64_t p_layer_type_encoded;
	PtrToArg<int64_t>::encode(p_layer_type, &p_layer_type_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_index_encoded, &p_layer_type_encoded);
}

Variant::Type TileSet::get_custom_data_layer_type(int32_t p_layer_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("get_custom_data_layer_type")._native_ptr(), 2990820875);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant::Type(0)));
	int64_t p_layer_index_encoded;
	PtrToArg<int64_t>::encode(p_layer_index, &p_layer_index_encoded);
	return (Variant::Type)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_layer_index_encoded);
}

void TileSet::set_source_level_tile_proxy(int32_t p_source_from, int32_t p_source_to) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("set_source_level_tile_proxy")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_source_from_encoded;
	PtrToArg<int64_t>::encode(p_source_from, &p_source_from_encoded);
	int64_t p_source_to_encoded;
	PtrToArg<int64_t>::encode(p_source_to, &p_source_to_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_source_from_encoded, &p_source_to_encoded);
}

int32_t TileSet::get_source_level_tile_proxy(int32_t p_source_from) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("get_source_level_tile_proxy")._native_ptr(), 3744713108);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_source_from_encoded;
	PtrToArg<int64_t>::encode(p_source_from, &p_source_from_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_source_from_encoded);
}

bool TileSet::has_source_level_tile_proxy(int32_t p_source_from) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("has_source_level_tile_proxy")._native_ptr(), 3067735520);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_source_from_encoded;
	PtrToArg<int64_t>::encode(p_source_from, &p_source_from_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_source_from_encoded);
}

void TileSet::remove_source_level_tile_proxy(int32_t p_source_from) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("remove_source_level_tile_proxy")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_source_from_encoded;
	PtrToArg<int64_t>::encode(p_source_from, &p_source_from_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_source_from_encoded);
}

void TileSet::set_coords_level_tile_proxy(int32_t p_source_from, const Vector2i &p_coords_from, int32_t p_source_to, const Vector2i &p_coords_to) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("set_coords_level_tile_proxy")._native_ptr(), 1769939278);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_source_from_encoded;
	PtrToArg<int64_t>::encode(p_source_from, &p_source_from_encoded);
	int64_t p_source_to_encoded;
	PtrToArg<int64_t>::encode(p_source_to, &p_source_to_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_source_from_encoded, &p_coords_from, &p_source_to_encoded, &p_coords_to);
}

Array TileSet::get_coords_level_tile_proxy(int32_t p_source_from, const Vector2i &p_coords_from) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("get_coords_level_tile_proxy")._native_ptr(), 2856536371);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Array()));
	int64_t p_source_from_encoded;
	PtrToArg<int64_t>::encode(p_source_from, &p_source_from_encoded);
	return ::godot::internal::_call_native_mb_ret<Array>(_gde_method_bind, _owner, &p_source_from_encoded, &p_coords_from);
}

bool TileSet::has_coords_level_tile_proxy(int32_t p_source_from, const Vector2i &p_coords_from) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("has_coords_level_tile_proxy")._native_ptr(), 3957903770);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_source_from_encoded;
	PtrToArg<int64_t>::encode(p_source_from, &p_source_from_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_source_from_encoded, &p_coords_from);
}

void TileSet::remove_coords_level_tile_proxy(int32_t p_source_from, const Vector2i &p_coords_from) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("remove_coords_level_tile_proxy")._native_ptr(), 2311374912);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_source_from_encoded;
	PtrToArg<int64_t>::encode(p_source_from, &p_source_from_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_source_from_encoded, &p_coords_from);
}

void TileSet::set_alternative_level_tile_proxy(int32_t p_source_from, const Vector2i &p_coords_from, int32_t p_alternative_from, int32_t p_source_to, const Vector2i &p_coords_to, int32_t p_alternative_to) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("set_alternative_level_tile_proxy")._native_ptr(), 3862385460);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_source_from_encoded;
	PtrToArg<int64_t>::encode(p_source_from, &p_source_from_encoded);
	int64_t p_alternative_from_encoded;
	PtrToArg<int64_t>::encode(p_alternative_from, &p_alternative_from_encoded);
	int64_t p_source_to_encoded;
	PtrToArg<int64_t>::encode(p_source_to, &p_source_to_encoded);
	int64_t p_alternative_to_encoded;
	PtrToArg<int64_t>::encode(p_alternative_to, &p_alternative_to_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_source_from_encoded, &p_coords_from, &p_alternative_from_encoded, &p_source_to_encoded, &p_coords_to, &p_alternative_to_encoded);
}

Array TileSet::get_alternative_level_tile_proxy(int32_t p_source_from, const Vector2i &p_coords_from, int32_t p_alternative_from) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("get_alternative_level_tile_proxy")._native_ptr(), 2303761075);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Array()));
	int64_t p_source_from_encoded;
	PtrToArg<int64_t>::encode(p_source_from, &p_source_from_encoded);
	int64_t p_alternative_from_encoded;
	PtrToArg<int64_t>::encode(p_alternative_from, &p_alternative_from_encoded);
	return ::godot::internal::_call_native_mb_ret<Array>(_gde_method_bind, _owner, &p_source_from_encoded, &p_coords_from, &p_alternative_from_encoded);
}

bool TileSet::has_alternative_level_tile_proxy(int32_t p_source_from, const Vector2i &p_coords_from, int32_t p_alternative_from) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("has_alternative_level_tile_proxy")._native_ptr(), 180086755);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_source_from_encoded;
	PtrToArg<int64_t>::encode(p_source_from, &p_source_from_encoded);
	int64_t p_alternative_from_encoded;
	PtrToArg<int64_t>::encode(p_alternative_from, &p_alternative_from_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_source_from_encoded, &p_coords_from, &p_alternative_from_encoded);
}

void TileSet::remove_alternative_level_tile_proxy(int32_t p_source_from, const Vector2i &p_coords_from, int32_t p_alternative_from) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("remove_alternative_level_tile_proxy")._native_ptr(), 2328951467);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_source_from_encoded;
	PtrToArg<int64_t>::encode(p_source_from, &p_source_from_encoded);
	int64_t p_alternative_from_encoded;
	PtrToArg<int64_t>::encode(p_alternative_from, &p_alternative_from_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_source_from_encoded, &p_coords_from, &p_alternative_from_encoded);
}

Array TileSet::map_tile_proxy(int32_t p_source_from, const Vector2i &p_coords_from, int32_t p_alternative_from) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("map_tile_proxy")._native_ptr(), 4267935328);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Array()));
	int64_t p_source_from_encoded;
	PtrToArg<int64_t>::encode(p_source_from, &p_source_from_encoded);
	int64_t p_alternative_from_encoded;
	PtrToArg<int64_t>::encode(p_alternative_from, &p_alternative_from_encoded);
	return ::godot::internal::_call_native_mb_ret<Array>(_gde_method_bind, _owner, &p_source_from_encoded, &p_coords_from, &p_alternative_from_encoded);
}

void TileSet::cleanup_invalid_tile_proxies() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("cleanup_invalid_tile_proxies")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void TileSet::clear_tile_proxies() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("clear_tile_proxies")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

int32_t TileSet::add_pattern(const Ref<TileMapPattern> &p_pattern, int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("add_pattern")._native_ptr(), 763712015);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, (p_pattern != nullptr ? &p_pattern->_owner : nullptr), &p_index_encoded);
}

Ref<TileMapPattern> TileSet::get_pattern(int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("get_pattern")._native_ptr(), 4207737510);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<TileMapPattern>()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return Ref<TileMapPattern>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<TileMapPattern>(_gde_method_bind, _owner, &p_index_encoded));
}

void TileSet::remove_pattern(int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("remove_pattern")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded);
}

int32_t TileSet::get_patterns_count() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(TileSet::get_class_static()._native_ptr(), StringName("get_patterns_count")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

} // namespace godot
