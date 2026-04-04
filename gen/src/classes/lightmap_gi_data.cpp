/**************************************************************************/
/*  lightmap_gi_data.cpp                                                  */
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

#include <godot_cpp/classes/lightmap_gi_data.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/texture_layered.hpp>
#include <godot_cpp/variant/rect2.hpp>

namespace godot {

void LightmapGIData::set_lightmap_textures(const TypedArray<Ref<TextureLayered>> &p_light_textures) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGIData::get_class_static()._native_ptr(), StringName("set_lightmap_textures")._native_ptr(), 381264803);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light_textures);
}

TypedArray<Ref<TextureLayered>> LightmapGIData::get_lightmap_textures() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGIData::get_class_static()._native_ptr(), StringName("get_lightmap_textures")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Ref<TextureLayered>>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Ref<TextureLayered>>>(_gde_method_bind, _owner);
}

void LightmapGIData::set_shadowmask_textures(const TypedArray<Ref<TextureLayered>> &p_shadowmask_textures) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGIData::get_class_static()._native_ptr(), StringName("set_shadowmask_textures")._native_ptr(), 381264803);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shadowmask_textures);
}

TypedArray<Ref<TextureLayered>> LightmapGIData::get_shadowmask_textures() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGIData::get_class_static()._native_ptr(), StringName("get_shadowmask_textures")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Ref<TextureLayered>>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Ref<TextureLayered>>>(_gde_method_bind, _owner);
}

void LightmapGIData::set_uses_spherical_harmonics(bool p_uses_spherical_harmonics) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGIData::get_class_static()._native_ptr(), StringName("set_uses_spherical_harmonics")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_uses_spherical_harmonics_encoded;
	PtrToArg<bool>::encode(p_uses_spherical_harmonics, &p_uses_spherical_harmonics_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_uses_spherical_harmonics_encoded);
}

bool LightmapGIData::is_using_spherical_harmonics() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGIData::get_class_static()._native_ptr(), StringName("is_using_spherical_harmonics")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void LightmapGIData::add_user(const NodePath &p_path, const Rect2 &p_uv_scale, int32_t p_slice_index, int32_t p_sub_instance) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGIData::get_class_static()._native_ptr(), StringName("add_user")._native_ptr(), 4272570515);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_slice_index_encoded;
	PtrToArg<int64_t>::encode(p_slice_index, &p_slice_index_encoded);
	int64_t p_sub_instance_encoded;
	PtrToArg<int64_t>::encode(p_sub_instance, &p_sub_instance_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_path, &p_uv_scale, &p_slice_index_encoded, &p_sub_instance_encoded);
}

int32_t LightmapGIData::get_user_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGIData::get_class_static()._native_ptr(), StringName("get_user_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

NodePath LightmapGIData::get_user_path(int32_t p_user_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGIData::get_class_static()._native_ptr(), StringName("get_user_path")._native_ptr(), 408788394);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	int64_t p_user_idx_encoded;
	PtrToArg<int64_t>::encode(p_user_idx, &p_user_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner, &p_user_idx_encoded);
}

void LightmapGIData::clear_users() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGIData::get_class_static()._native_ptr(), StringName("clear_users")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void LightmapGIData::set_light_texture(const Ref<TextureLayered> &p_light_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGIData::get_class_static()._native_ptr(), StringName("set_light_texture")._native_ptr(), 1278366092);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_light_texture != nullptr ? &p_light_texture->_owner : nullptr));
}

Ref<TextureLayered> LightmapGIData::get_light_texture() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(LightmapGIData::get_class_static()._native_ptr(), StringName("get_light_texture")._native_ptr(), 3984243839);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<TextureLayered>()));
	return Ref<TextureLayered>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<TextureLayered>(_gde_method_bind, _owner));
}

} // namespace godot
