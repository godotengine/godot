/**************************************************************************/
/*  rendering_server.cpp                                                  */
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

#include <godot_cpp/classes/rendering_server.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/variant/basis.hpp>
#include <godot_cpp/variant/callable.hpp>
#include <godot_cpp/variant/plane.hpp>
#include <godot_cpp/variant/vector2.hpp>
#include <godot_cpp/variant/vector2i.hpp>
#include <godot_cpp/variant/vector3.hpp>

namespace godot {

RenderingServer *RenderingServer::singleton = nullptr;

RenderingServer *RenderingServer::get_singleton() {
	if (unlikely(singleton == nullptr)) {
		GDExtensionObjectPtr singleton_obj = ::godot::gdextension_interface::global_get_singleton(RenderingServer::get_class_static()._native_ptr());
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton_obj, nullptr);
#endif // DEBUG_ENABLED
		singleton = reinterpret_cast<RenderingServer *>(::godot::gdextension_interface::object_get_instance_binding(singleton_obj, ::godot::gdextension_interface::token, &RenderingServer::_gde_binding_callbacks));
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton, nullptr);
#endif // DEBUG_ENABLED
		if (likely(singleton)) {
			ClassDB::_register_engine_singleton(RenderingServer::get_class_static(), singleton);
		}
	}
	return singleton;
}

RenderingServer::~RenderingServer() {
	if (singleton == this) {
		ClassDB::_unregister_engine_singleton(RenderingServer::get_class_static());
		singleton = nullptr;
	}
}

RID RenderingServer::texture_2d_create(const Ref<Image> &p_image) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("texture_2d_create")._native_ptr(), 2010018390);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, (p_image != nullptr ? &p_image->_owner : nullptr));
}

RID RenderingServer::texture_2d_layered_create(const TypedArray<Ref<Image>> &p_layers, RenderingServer::TextureLayeredType p_layered_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("texture_2d_layered_create")._native_ptr(), 913689023);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_layered_type_encoded;
	PtrToArg<int64_t>::encode(p_layered_type, &p_layered_type_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_layers, &p_layered_type_encoded);
}

RID RenderingServer::texture_3d_create(Image::Format p_format, int32_t p_width, int32_t p_height, int32_t p_depth, bool p_mipmaps, const TypedArray<Ref<Image>> &p_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("texture_3d_create")._native_ptr(), 4036838706);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_format_encoded;
	PtrToArg<int64_t>::encode(p_format, &p_format_encoded);
	int64_t p_width_encoded;
	PtrToArg<int64_t>::encode(p_width, &p_width_encoded);
	int64_t p_height_encoded;
	PtrToArg<int64_t>::encode(p_height, &p_height_encoded);
	int64_t p_depth_encoded;
	PtrToArg<int64_t>::encode(p_depth, &p_depth_encoded);
	int8_t p_mipmaps_encoded;
	PtrToArg<bool>::encode(p_mipmaps, &p_mipmaps_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_format_encoded, &p_width_encoded, &p_height_encoded, &p_depth_encoded, &p_mipmaps_encoded, &p_data);
}

RID RenderingServer::texture_proxy_create(const RID &p_base) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("texture_proxy_create")._native_ptr(), 41030802);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_base);
}

RID RenderingServer::texture_create_from_native_handle(RenderingServer::TextureType p_type, Image::Format p_format, uint64_t p_native_handle, int32_t p_width, int32_t p_height, int32_t p_depth, int32_t p_layers, RenderingServer::TextureLayeredType p_layered_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("texture_create_from_native_handle")._native_ptr(), 1682977582);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	int64_t p_format_encoded;
	PtrToArg<int64_t>::encode(p_format, &p_format_encoded);
	int64_t p_native_handle_encoded;
	PtrToArg<int64_t>::encode(p_native_handle, &p_native_handle_encoded);
	int64_t p_width_encoded;
	PtrToArg<int64_t>::encode(p_width, &p_width_encoded);
	int64_t p_height_encoded;
	PtrToArg<int64_t>::encode(p_height, &p_height_encoded);
	int64_t p_depth_encoded;
	PtrToArg<int64_t>::encode(p_depth, &p_depth_encoded);
	int64_t p_layers_encoded;
	PtrToArg<int64_t>::encode(p_layers, &p_layers_encoded);
	int64_t p_layered_type_encoded;
	PtrToArg<int64_t>::encode(p_layered_type, &p_layered_type_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_type_encoded, &p_format_encoded, &p_native_handle_encoded, &p_width_encoded, &p_height_encoded, &p_depth_encoded, &p_layers_encoded, &p_layered_type_encoded);
}

void RenderingServer::texture_2d_update(const RID &p_texture, const Ref<Image> &p_image, int32_t p_layer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("texture_2d_update")._native_ptr(), 999539803);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_texture, (p_image != nullptr ? &p_image->_owner : nullptr), &p_layer_encoded);
}

void RenderingServer::texture_3d_update(const RID &p_texture, const TypedArray<Ref<Image>> &p_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("texture_3d_update")._native_ptr(), 684822712);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_texture, &p_data);
}

void RenderingServer::texture_proxy_update(const RID &p_texture, const RID &p_proxy_to) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("texture_proxy_update")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_texture, &p_proxy_to);
}

RID RenderingServer::texture_2d_placeholder_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("texture_2d_placeholder_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

RID RenderingServer::texture_2d_layered_placeholder_create(RenderingServer::TextureLayeredType p_layered_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("texture_2d_layered_placeholder_create")._native_ptr(), 1394585590);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_layered_type_encoded;
	PtrToArg<int64_t>::encode(p_layered_type, &p_layered_type_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_layered_type_encoded);
}

RID RenderingServer::texture_3d_placeholder_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("texture_3d_placeholder_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

Ref<Image> RenderingServer::texture_2d_get(const RID &p_texture) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("texture_2d_get")._native_ptr(), 4206205781);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Image>()));
	return Ref<Image>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Image>(_gde_method_bind, _owner, &p_texture));
}

Ref<Image> RenderingServer::texture_2d_layer_get(const RID &p_texture, int32_t p_layer) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("texture_2d_layer_get")._native_ptr(), 2705440895);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Image>()));
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	return Ref<Image>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Image>(_gde_method_bind, _owner, &p_texture, &p_layer_encoded));
}

TypedArray<Ref<Image>> RenderingServer::texture_3d_get(const RID &p_texture) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("texture_3d_get")._native_ptr(), 2684255073);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Ref<Image>>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Ref<Image>>>(_gde_method_bind, _owner, &p_texture);
}

void RenderingServer::texture_replace(const RID &p_texture, const RID &p_by_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("texture_replace")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_texture, &p_by_texture);
}

void RenderingServer::texture_set_size_override(const RID &p_texture, int32_t p_width, int32_t p_height) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("texture_set_size_override")._native_ptr(), 4288446313);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_width_encoded;
	PtrToArg<int64_t>::encode(p_width, &p_width_encoded);
	int64_t p_height_encoded;
	PtrToArg<int64_t>::encode(p_height, &p_height_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_texture, &p_width_encoded, &p_height_encoded);
}

void RenderingServer::texture_set_path(const RID &p_texture, const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("texture_set_path")._native_ptr(), 2726140452);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_texture, &p_path);
}

String RenderingServer::texture_get_path(const RID &p_texture) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("texture_get_path")._native_ptr(), 642473191);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_texture);
}

Image::Format RenderingServer::texture_get_format(const RID &p_texture) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("texture_get_format")._native_ptr(), 1932918979);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Image::Format(0)));
	return (Image::Format)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_texture);
}

void RenderingServer::texture_set_force_redraw_if_visible(const RID &p_texture, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("texture_set_force_redraw_if_visible")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_texture, &p_enable_encoded);
}

RID RenderingServer::texture_rd_create(const RID &p_rd_texture, RenderingServer::TextureLayeredType p_layer_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("texture_rd_create")._native_ptr(), 1434128712);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_layer_type_encoded;
	PtrToArg<int64_t>::encode(p_layer_type, &p_layer_type_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_rd_texture, &p_layer_type_encoded);
}

RID RenderingServer::texture_get_rd_texture(const RID &p_texture, bool p_srgb) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("texture_get_rd_texture")._native_ptr(), 2790148051);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int8_t p_srgb_encoded;
	PtrToArg<bool>::encode(p_srgb, &p_srgb_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_texture, &p_srgb_encoded);
}

uint64_t RenderingServer::texture_get_native_handle(const RID &p_texture, bool p_srgb) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("texture_get_native_handle")._native_ptr(), 1834114100);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int8_t p_srgb_encoded;
	PtrToArg<bool>::encode(p_srgb, &p_srgb_encoded);
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_texture, &p_srgb_encoded);
}

RID RenderingServer::shader_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("shader_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void RenderingServer::shader_set_code(const RID &p_shader, const String &p_code) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("shader_set_code")._native_ptr(), 2726140452);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shader, &p_code);
}

void RenderingServer::shader_set_path_hint(const RID &p_shader, const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("shader_set_path_hint")._native_ptr(), 2726140452);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shader, &p_path);
}

String RenderingServer::shader_get_code(const RID &p_shader) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("shader_get_code")._native_ptr(), 642473191);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_shader);
}

TypedArray<Dictionary> RenderingServer::get_shader_parameter_list(const RID &p_shader) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("get_shader_parameter_list")._native_ptr(), 2684255073);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Dictionary>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Dictionary>>(_gde_method_bind, _owner, &p_shader);
}

Variant RenderingServer::shader_get_parameter_default(const RID &p_shader, const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("shader_get_parameter_default")._native_ptr(), 2621281810);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_shader, &p_name);
}

void RenderingServer::shader_set_default_texture_parameter(const RID &p_shader, const StringName &p_name, const RID &p_texture, int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("shader_set_default_texture_parameter")._native_ptr(), 4094001817);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shader, &p_name, &p_texture, &p_index_encoded);
}

RID RenderingServer::shader_get_default_texture_parameter(const RID &p_shader, const StringName &p_name, int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("shader_get_default_texture_parameter")._native_ptr(), 1464608890);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_shader, &p_name, &p_index_encoded);
}

RID RenderingServer::material_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("material_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void RenderingServer::material_set_shader(const RID &p_shader_material, const RID &p_shader) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("material_set_shader")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shader_material, &p_shader);
}

void RenderingServer::material_set_param(const RID &p_material, const StringName &p_parameter, const Variant &p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("material_set_param")._native_ptr(), 3477296213);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_material, &p_parameter, &p_value);
}

Variant RenderingServer::material_get_param(const RID &p_material, const StringName &p_parameter) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("material_get_param")._native_ptr(), 2621281810);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_material, &p_parameter);
}

void RenderingServer::material_set_render_priority(const RID &p_material, int32_t p_priority) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("material_set_render_priority")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_priority_encoded;
	PtrToArg<int64_t>::encode(p_priority, &p_priority_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_material, &p_priority_encoded);
}

void RenderingServer::material_set_next_pass(const RID &p_material, const RID &p_next_material) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("material_set_next_pass")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_material, &p_next_material);
}

void RenderingServer::material_set_use_debanding(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("material_set_use_debanding")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

RID RenderingServer::mesh_create_from_surfaces(const TypedArray<Dictionary> &p_surfaces, int32_t p_blend_shape_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("mesh_create_from_surfaces")._native_ptr(), 4291747531);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_blend_shape_count_encoded;
	PtrToArg<int64_t>::encode(p_blend_shape_count, &p_blend_shape_count_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_surfaces, &p_blend_shape_count_encoded);
}

RID RenderingServer::mesh_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("mesh_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

uint32_t RenderingServer::mesh_surface_get_format_offset(BitField<RenderingServer::ArrayFormat> p_format, int32_t p_vertex_count, int32_t p_array_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("mesh_surface_get_format_offset")._native_ptr(), 2981368685);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_vertex_count_encoded;
	PtrToArg<int64_t>::encode(p_vertex_count, &p_vertex_count_encoded);
	int64_t p_array_index_encoded;
	PtrToArg<int64_t>::encode(p_array_index, &p_array_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_format, &p_vertex_count_encoded, &p_array_index_encoded);
}

uint32_t RenderingServer::mesh_surface_get_format_vertex_stride(BitField<RenderingServer::ArrayFormat> p_format, int32_t p_vertex_count) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("mesh_surface_get_format_vertex_stride")._native_ptr(), 3188363337);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_vertex_count_encoded;
	PtrToArg<int64_t>::encode(p_vertex_count, &p_vertex_count_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_format, &p_vertex_count_encoded);
}

uint32_t RenderingServer::mesh_surface_get_format_normal_tangent_stride(BitField<RenderingServer::ArrayFormat> p_format, int32_t p_vertex_count) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("mesh_surface_get_format_normal_tangent_stride")._native_ptr(), 3188363337);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_vertex_count_encoded;
	PtrToArg<int64_t>::encode(p_vertex_count, &p_vertex_count_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_format, &p_vertex_count_encoded);
}

uint32_t RenderingServer::mesh_surface_get_format_attribute_stride(BitField<RenderingServer::ArrayFormat> p_format, int32_t p_vertex_count) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("mesh_surface_get_format_attribute_stride")._native_ptr(), 3188363337);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_vertex_count_encoded;
	PtrToArg<int64_t>::encode(p_vertex_count, &p_vertex_count_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_format, &p_vertex_count_encoded);
}

uint32_t RenderingServer::mesh_surface_get_format_skin_stride(BitField<RenderingServer::ArrayFormat> p_format, int32_t p_vertex_count) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("mesh_surface_get_format_skin_stride")._native_ptr(), 3188363337);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_vertex_count_encoded;
	PtrToArg<int64_t>::encode(p_vertex_count, &p_vertex_count_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_format, &p_vertex_count_encoded);
}

uint32_t RenderingServer::mesh_surface_get_format_index_stride(BitField<RenderingServer::ArrayFormat> p_format, int32_t p_vertex_count) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("mesh_surface_get_format_index_stride")._native_ptr(), 3188363337);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_vertex_count_encoded;
	PtrToArg<int64_t>::encode(p_vertex_count, &p_vertex_count_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_format, &p_vertex_count_encoded);
}

void RenderingServer::mesh_add_surface(const RID &p_mesh, const Dictionary &p_surface) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("mesh_add_surface")._native_ptr(), 1217542888);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mesh, &p_surface);
}

void RenderingServer::mesh_add_surface_from_arrays(const RID &p_mesh, RenderingServer::PrimitiveType p_primitive, const Array &p_arrays, const Array &p_blend_shapes, const Dictionary &p_lods, BitField<RenderingServer::ArrayFormat> p_compress_format) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("mesh_add_surface_from_arrays")._native_ptr(), 2342446560);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_primitive_encoded;
	PtrToArg<int64_t>::encode(p_primitive, &p_primitive_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mesh, &p_primitive_encoded, &p_arrays, &p_blend_shapes, &p_lods, &p_compress_format);
}

int32_t RenderingServer::mesh_get_blend_shape_count(const RID &p_mesh) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("mesh_get_blend_shape_count")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_mesh);
}

void RenderingServer::mesh_set_blend_shape_mode(const RID &p_mesh, RenderingServer::BlendShapeMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("mesh_set_blend_shape_mode")._native_ptr(), 1294662092);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mesh, &p_mode_encoded);
}

RenderingServer::BlendShapeMode RenderingServer::mesh_get_blend_shape_mode(const RID &p_mesh) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("mesh_get_blend_shape_mode")._native_ptr(), 4282291819);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingServer::BlendShapeMode(0)));
	return (RenderingServer::BlendShapeMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_mesh);
}

void RenderingServer::mesh_surface_set_material(const RID &p_mesh, int32_t p_surface, const RID &p_material) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("mesh_surface_set_material")._native_ptr(), 2310537182);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_surface_encoded;
	PtrToArg<int64_t>::encode(p_surface, &p_surface_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mesh, &p_surface_encoded, &p_material);
}

RID RenderingServer::mesh_surface_get_material(const RID &p_mesh, int32_t p_surface) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("mesh_surface_get_material")._native_ptr(), 1066463050);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_surface_encoded;
	PtrToArg<int64_t>::encode(p_surface, &p_surface_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_mesh, &p_surface_encoded);
}

Dictionary RenderingServer::mesh_get_surface(const RID &p_mesh, int32_t p_surface) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("mesh_get_surface")._native_ptr(), 186674697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	int64_t p_surface_encoded;
	PtrToArg<int64_t>::encode(p_surface, &p_surface_encoded);
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, &p_mesh, &p_surface_encoded);
}

Array RenderingServer::mesh_surface_get_arrays(const RID &p_mesh, int32_t p_surface) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("mesh_surface_get_arrays")._native_ptr(), 1778388067);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Array()));
	int64_t p_surface_encoded;
	PtrToArg<int64_t>::encode(p_surface, &p_surface_encoded);
	return ::godot::internal::_call_native_mb_ret<Array>(_gde_method_bind, _owner, &p_mesh, &p_surface_encoded);
}

TypedArray<Array> RenderingServer::mesh_surface_get_blend_shape_arrays(const RID &p_mesh, int32_t p_surface) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("mesh_surface_get_blend_shape_arrays")._native_ptr(), 1778388067);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Array>()));
	int64_t p_surface_encoded;
	PtrToArg<int64_t>::encode(p_surface, &p_surface_encoded);
	return ::godot::internal::_call_native_mb_ret<TypedArray<Array>>(_gde_method_bind, _owner, &p_mesh, &p_surface_encoded);
}

int32_t RenderingServer::mesh_get_surface_count(const RID &p_mesh) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("mesh_get_surface_count")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_mesh);
}

void RenderingServer::mesh_set_custom_aabb(const RID &p_mesh, const AABB &p_aabb) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("mesh_set_custom_aabb")._native_ptr(), 3696536120);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mesh, &p_aabb);
}

AABB RenderingServer::mesh_get_custom_aabb(const RID &p_mesh) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("mesh_get_custom_aabb")._native_ptr(), 974181306);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AABB()));
	return ::godot::internal::_call_native_mb_ret<AABB>(_gde_method_bind, _owner, &p_mesh);
}

void RenderingServer::mesh_surface_remove(const RID &p_mesh, int32_t p_surface) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("mesh_surface_remove")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_surface_encoded;
	PtrToArg<int64_t>::encode(p_surface, &p_surface_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mesh, &p_surface_encoded);
}

void RenderingServer::mesh_clear(const RID &p_mesh) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("mesh_clear")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mesh);
}

void RenderingServer::mesh_surface_update_vertex_region(const RID &p_mesh, int32_t p_surface, int32_t p_offset, const PackedByteArray &p_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("mesh_surface_update_vertex_region")._native_ptr(), 2900195149);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_surface_encoded;
	PtrToArg<int64_t>::encode(p_surface, &p_surface_encoded);
	int64_t p_offset_encoded;
	PtrToArg<int64_t>::encode(p_offset, &p_offset_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mesh, &p_surface_encoded, &p_offset_encoded, &p_data);
}

void RenderingServer::mesh_surface_update_attribute_region(const RID &p_mesh, int32_t p_surface, int32_t p_offset, const PackedByteArray &p_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("mesh_surface_update_attribute_region")._native_ptr(), 2900195149);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_surface_encoded;
	PtrToArg<int64_t>::encode(p_surface, &p_surface_encoded);
	int64_t p_offset_encoded;
	PtrToArg<int64_t>::encode(p_offset, &p_offset_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mesh, &p_surface_encoded, &p_offset_encoded, &p_data);
}

void RenderingServer::mesh_surface_update_skin_region(const RID &p_mesh, int32_t p_surface, int32_t p_offset, const PackedByteArray &p_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("mesh_surface_update_skin_region")._native_ptr(), 2900195149);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_surface_encoded;
	PtrToArg<int64_t>::encode(p_surface, &p_surface_encoded);
	int64_t p_offset_encoded;
	PtrToArg<int64_t>::encode(p_offset, &p_offset_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mesh, &p_surface_encoded, &p_offset_encoded, &p_data);
}

void RenderingServer::mesh_surface_update_index_region(const RID &p_mesh, int32_t p_surface, int32_t p_offset, const PackedByteArray &p_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("mesh_surface_update_index_region")._native_ptr(), 2900195149);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_surface_encoded;
	PtrToArg<int64_t>::encode(p_surface, &p_surface_encoded);
	int64_t p_offset_encoded;
	PtrToArg<int64_t>::encode(p_offset, &p_offset_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mesh, &p_surface_encoded, &p_offset_encoded, &p_data);
}

void RenderingServer::mesh_set_shadow_mesh(const RID &p_mesh, const RID &p_shadow_mesh) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("mesh_set_shadow_mesh")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mesh, &p_shadow_mesh);
}

RID RenderingServer::multimesh_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("multimesh_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void RenderingServer::multimesh_allocate_data(const RID &p_multimesh, int32_t p_instances, RenderingServer::MultimeshTransformFormat p_transform_format, bool p_color_format, bool p_custom_data_format, bool p_use_indirect) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("multimesh_allocate_data")._native_ptr(), 557240154);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_instances_encoded;
	PtrToArg<int64_t>::encode(p_instances, &p_instances_encoded);
	int64_t p_transform_format_encoded;
	PtrToArg<int64_t>::encode(p_transform_format, &p_transform_format_encoded);
	int8_t p_color_format_encoded;
	PtrToArg<bool>::encode(p_color_format, &p_color_format_encoded);
	int8_t p_custom_data_format_encoded;
	PtrToArg<bool>::encode(p_custom_data_format, &p_custom_data_format_encoded);
	int8_t p_use_indirect_encoded;
	PtrToArg<bool>::encode(p_use_indirect, &p_use_indirect_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_multimesh, &p_instances_encoded, &p_transform_format_encoded, &p_color_format_encoded, &p_custom_data_format_encoded, &p_use_indirect_encoded);
}

int32_t RenderingServer::multimesh_get_instance_count(const RID &p_multimesh) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("multimesh_get_instance_count")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_multimesh);
}

void RenderingServer::multimesh_set_mesh(const RID &p_multimesh, const RID &p_mesh) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("multimesh_set_mesh")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_multimesh, &p_mesh);
}

void RenderingServer::multimesh_instance_set_transform(const RID &p_multimesh, int32_t p_index, const Transform3D &p_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("multimesh_instance_set_transform")._native_ptr(), 675327471);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_multimesh, &p_index_encoded, &p_transform);
}

void RenderingServer::multimesh_instance_set_transform_2d(const RID &p_multimesh, int32_t p_index, const Transform2D &p_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("multimesh_instance_set_transform_2d")._native_ptr(), 736082694);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_multimesh, &p_index_encoded, &p_transform);
}

void RenderingServer::multimesh_instance_set_color(const RID &p_multimesh, int32_t p_index, const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("multimesh_instance_set_color")._native_ptr(), 176975443);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_multimesh, &p_index_encoded, &p_color);
}

void RenderingServer::multimesh_instance_set_custom_data(const RID &p_multimesh, int32_t p_index, const Color &p_custom_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("multimesh_instance_set_custom_data")._native_ptr(), 176975443);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_multimesh, &p_index_encoded, &p_custom_data);
}

RID RenderingServer::multimesh_get_mesh(const RID &p_multimesh) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("multimesh_get_mesh")._native_ptr(), 3814569979);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_multimesh);
}

AABB RenderingServer::multimesh_get_aabb(const RID &p_multimesh) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("multimesh_get_aabb")._native_ptr(), 974181306);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AABB()));
	return ::godot::internal::_call_native_mb_ret<AABB>(_gde_method_bind, _owner, &p_multimesh);
}

void RenderingServer::multimesh_set_custom_aabb(const RID &p_multimesh, const AABB &p_aabb) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("multimesh_set_custom_aabb")._native_ptr(), 3696536120);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_multimesh, &p_aabb);
}

AABB RenderingServer::multimesh_get_custom_aabb(const RID &p_multimesh) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("multimesh_get_custom_aabb")._native_ptr(), 974181306);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AABB()));
	return ::godot::internal::_call_native_mb_ret<AABB>(_gde_method_bind, _owner, &p_multimesh);
}

Transform3D RenderingServer::multimesh_instance_get_transform(const RID &p_multimesh, int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("multimesh_instance_get_transform")._native_ptr(), 1050775521);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform3D()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<Transform3D>(_gde_method_bind, _owner, &p_multimesh, &p_index_encoded);
}

Transform2D RenderingServer::multimesh_instance_get_transform_2d(const RID &p_multimesh, int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("multimesh_instance_get_transform_2d")._native_ptr(), 1324854622);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform2D()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<Transform2D>(_gde_method_bind, _owner, &p_multimesh, &p_index_encoded);
}

Color RenderingServer::multimesh_instance_get_color(const RID &p_multimesh, int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("multimesh_instance_get_color")._native_ptr(), 2946315076);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner, &p_multimesh, &p_index_encoded);
}

Color RenderingServer::multimesh_instance_get_custom_data(const RID &p_multimesh, int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("multimesh_instance_get_custom_data")._native_ptr(), 2946315076);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner, &p_multimesh, &p_index_encoded);
}

void RenderingServer::multimesh_set_visible_instances(const RID &p_multimesh, int32_t p_visible) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("multimesh_set_visible_instances")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_visible_encoded;
	PtrToArg<int64_t>::encode(p_visible, &p_visible_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_multimesh, &p_visible_encoded);
}

int32_t RenderingServer::multimesh_get_visible_instances(const RID &p_multimesh) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("multimesh_get_visible_instances")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_multimesh);
}

void RenderingServer::multimesh_set_buffer(const RID &p_multimesh, const PackedFloat32Array &p_buffer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("multimesh_set_buffer")._native_ptr(), 2960552364);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_multimesh, &p_buffer);
}

RID RenderingServer::multimesh_get_command_buffer_rd_rid(const RID &p_multimesh) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("multimesh_get_command_buffer_rd_rid")._native_ptr(), 3814569979);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_multimesh);
}

RID RenderingServer::multimesh_get_buffer_rd_rid(const RID &p_multimesh) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("multimesh_get_buffer_rd_rid")._native_ptr(), 3814569979);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_multimesh);
}

PackedFloat32Array RenderingServer::multimesh_get_buffer(const RID &p_multimesh) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("multimesh_get_buffer")._native_ptr(), 3964669176);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedFloat32Array()));
	return ::godot::internal::_call_native_mb_ret<PackedFloat32Array>(_gde_method_bind, _owner, &p_multimesh);
}

void RenderingServer::multimesh_set_buffer_interpolated(const RID &p_multimesh, const PackedFloat32Array &p_buffer, const PackedFloat32Array &p_buffer_previous) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("multimesh_set_buffer_interpolated")._native_ptr(), 659844711);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_multimesh, &p_buffer, &p_buffer_previous);
}

void RenderingServer::multimesh_set_physics_interpolated(const RID &p_multimesh, bool p_interpolated) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("multimesh_set_physics_interpolated")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_interpolated_encoded;
	PtrToArg<bool>::encode(p_interpolated, &p_interpolated_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_multimesh, &p_interpolated_encoded);
}

void RenderingServer::multimesh_set_physics_interpolation_quality(const RID &p_multimesh, RenderingServer::MultimeshPhysicsInterpolationQuality p_quality) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("multimesh_set_physics_interpolation_quality")._native_ptr(), 3934808223);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_quality_encoded;
	PtrToArg<int64_t>::encode(p_quality, &p_quality_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_multimesh, &p_quality_encoded);
}

void RenderingServer::multimesh_instance_reset_physics_interpolation(const RID &p_multimesh, int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("multimesh_instance_reset_physics_interpolation")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_multimesh, &p_index_encoded);
}

void RenderingServer::multimesh_instances_reset_physics_interpolation(const RID &p_multimesh) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("multimesh_instances_reset_physics_interpolation")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_multimesh);
}

RID RenderingServer::skeleton_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("skeleton_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void RenderingServer::skeleton_allocate_data(const RID &p_skeleton, int32_t p_bones, bool p_is_2d_skeleton) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("skeleton_allocate_data")._native_ptr(), 1904426712);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_bones_encoded;
	PtrToArg<int64_t>::encode(p_bones, &p_bones_encoded);
	int8_t p_is_2d_skeleton_encoded;
	PtrToArg<bool>::encode(p_is_2d_skeleton, &p_is_2d_skeleton_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_skeleton, &p_bones_encoded, &p_is_2d_skeleton_encoded);
}

int32_t RenderingServer::skeleton_get_bone_count(const RID &p_skeleton) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("skeleton_get_bone_count")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_skeleton);
}

void RenderingServer::skeleton_bone_set_transform(const RID &p_skeleton, int32_t p_bone, const Transform3D &p_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("skeleton_bone_set_transform")._native_ptr(), 675327471);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_bone_encoded;
	PtrToArg<int64_t>::encode(p_bone, &p_bone_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_skeleton, &p_bone_encoded, &p_transform);
}

Transform3D RenderingServer::skeleton_bone_get_transform(const RID &p_skeleton, int32_t p_bone) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("skeleton_bone_get_transform")._native_ptr(), 1050775521);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform3D()));
	int64_t p_bone_encoded;
	PtrToArg<int64_t>::encode(p_bone, &p_bone_encoded);
	return ::godot::internal::_call_native_mb_ret<Transform3D>(_gde_method_bind, _owner, &p_skeleton, &p_bone_encoded);
}

void RenderingServer::skeleton_bone_set_transform_2d(const RID &p_skeleton, int32_t p_bone, const Transform2D &p_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("skeleton_bone_set_transform_2d")._native_ptr(), 736082694);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_bone_encoded;
	PtrToArg<int64_t>::encode(p_bone, &p_bone_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_skeleton, &p_bone_encoded, &p_transform);
}

Transform2D RenderingServer::skeleton_bone_get_transform_2d(const RID &p_skeleton, int32_t p_bone) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("skeleton_bone_get_transform_2d")._native_ptr(), 1324854622);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform2D()));
	int64_t p_bone_encoded;
	PtrToArg<int64_t>::encode(p_bone, &p_bone_encoded);
	return ::godot::internal::_call_native_mb_ret<Transform2D>(_gde_method_bind, _owner, &p_skeleton, &p_bone_encoded);
}

void RenderingServer::skeleton_set_base_transform_2d(const RID &p_skeleton, const Transform2D &p_base_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("skeleton_set_base_transform_2d")._native_ptr(), 1246044741);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_skeleton, &p_base_transform);
}

RID RenderingServer::directional_light_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("directional_light_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

RID RenderingServer::omni_light_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("omni_light_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

RID RenderingServer::spot_light_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("spot_light_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void RenderingServer::light_set_color(const RID &p_light, const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("light_set_color")._native_ptr(), 2948539648);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light, &p_color);
}

void RenderingServer::light_set_param(const RID &p_light, RenderingServer::LightParam p_param, float p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("light_set_param")._native_ptr(), 501936875);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_param_encoded;
	PtrToArg<int64_t>::encode(p_param, &p_param_encoded);
	double p_value_encoded;
	PtrToArg<double>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light, &p_param_encoded, &p_value_encoded);
}

void RenderingServer::light_set_shadow(const RID &p_light, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("light_set_shadow")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light, &p_enabled_encoded);
}

void RenderingServer::light_set_projector(const RID &p_light, const RID &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("light_set_projector")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light, &p_texture);
}

void RenderingServer::light_set_negative(const RID &p_light, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("light_set_negative")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light, &p_enable_encoded);
}

void RenderingServer::light_set_cull_mask(const RID &p_light, uint32_t p_mask) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("light_set_cull_mask")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mask_encoded;
	PtrToArg<int64_t>::encode(p_mask, &p_mask_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light, &p_mask_encoded);
}

void RenderingServer::light_set_distance_fade(const RID &p_decal, bool p_enabled, float p_begin, float p_shadow, float p_length) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("light_set_distance_fade")._native_ptr(), 1622292572);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	double p_begin_encoded;
	PtrToArg<double>::encode(p_begin, &p_begin_encoded);
	double p_shadow_encoded;
	PtrToArg<double>::encode(p_shadow, &p_shadow_encoded);
	double p_length_encoded;
	PtrToArg<double>::encode(p_length, &p_length_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_decal, &p_enabled_encoded, &p_begin_encoded, &p_shadow_encoded, &p_length_encoded);
}

void RenderingServer::light_set_reverse_cull_face_mode(const RID &p_light, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("light_set_reverse_cull_face_mode")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light, &p_enabled_encoded);
}

void RenderingServer::light_set_shadow_caster_mask(const RID &p_light, uint32_t p_mask) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("light_set_shadow_caster_mask")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mask_encoded;
	PtrToArg<int64_t>::encode(p_mask, &p_mask_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light, &p_mask_encoded);
}

void RenderingServer::light_set_bake_mode(const RID &p_light, RenderingServer::LightBakeMode p_bake_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("light_set_bake_mode")._native_ptr(), 1048525260);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_bake_mode_encoded;
	PtrToArg<int64_t>::encode(p_bake_mode, &p_bake_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light, &p_bake_mode_encoded);
}

void RenderingServer::light_set_max_sdfgi_cascade(const RID &p_light, uint32_t p_cascade) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("light_set_max_sdfgi_cascade")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_cascade_encoded;
	PtrToArg<int64_t>::encode(p_cascade, &p_cascade_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light, &p_cascade_encoded);
}

void RenderingServer::light_omni_set_shadow_mode(const RID &p_light, RenderingServer::LightOmniShadowMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("light_omni_set_shadow_mode")._native_ptr(), 2552677200);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light, &p_mode_encoded);
}

void RenderingServer::light_directional_set_shadow_mode(const RID &p_light, RenderingServer::LightDirectionalShadowMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("light_directional_set_shadow_mode")._native_ptr(), 380462970);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light, &p_mode_encoded);
}

void RenderingServer::light_directional_set_blend_splits(const RID &p_light, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("light_directional_set_blend_splits")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light, &p_enable_encoded);
}

void RenderingServer::light_directional_set_sky_mode(const RID &p_light, RenderingServer::LightDirectionalSkyMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("light_directional_set_sky_mode")._native_ptr(), 2559740754);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light, &p_mode_encoded);
}

void RenderingServer::light_projectors_set_filter(RenderingServer::LightProjectorFilter p_filter) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("light_projectors_set_filter")._native_ptr(), 43944325);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_filter_encoded;
	PtrToArg<int64_t>::encode(p_filter, &p_filter_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_filter_encoded);
}

void RenderingServer::lightmaps_set_bicubic_filter(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("lightmaps_set_bicubic_filter")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

void RenderingServer::positional_soft_shadow_filter_set_quality(RenderingServer::ShadowQuality p_quality) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("positional_soft_shadow_filter_set_quality")._native_ptr(), 3613045266);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_quality_encoded;
	PtrToArg<int64_t>::encode(p_quality, &p_quality_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_quality_encoded);
}

void RenderingServer::directional_soft_shadow_filter_set_quality(RenderingServer::ShadowQuality p_quality) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("directional_soft_shadow_filter_set_quality")._native_ptr(), 3613045266);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_quality_encoded;
	PtrToArg<int64_t>::encode(p_quality, &p_quality_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_quality_encoded);
}

void RenderingServer::directional_shadow_atlas_set_size(int32_t p_size, bool p_is_16bits) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("directional_shadow_atlas_set_size")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	int8_t p_is_16bits_encoded;
	PtrToArg<bool>::encode(p_is_16bits, &p_is_16bits_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size_encoded, &p_is_16bits_encoded);
}

RID RenderingServer::reflection_probe_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("reflection_probe_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void RenderingServer::reflection_probe_set_update_mode(const RID &p_probe, RenderingServer::ReflectionProbeUpdateMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("reflection_probe_set_update_mode")._native_ptr(), 3853670147);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_probe, &p_mode_encoded);
}

void RenderingServer::reflection_probe_set_intensity(const RID &p_probe, float p_intensity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("reflection_probe_set_intensity")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_intensity_encoded;
	PtrToArg<double>::encode(p_intensity, &p_intensity_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_probe, &p_intensity_encoded);
}

void RenderingServer::reflection_probe_set_blend_distance(const RID &p_probe, float p_blend_distance) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("reflection_probe_set_blend_distance")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_blend_distance_encoded;
	PtrToArg<double>::encode(p_blend_distance, &p_blend_distance_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_probe, &p_blend_distance_encoded);
}

void RenderingServer::reflection_probe_set_ambient_mode(const RID &p_probe, RenderingServer::ReflectionProbeAmbientMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("reflection_probe_set_ambient_mode")._native_ptr(), 184163074);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_probe, &p_mode_encoded);
}

void RenderingServer::reflection_probe_set_ambient_color(const RID &p_probe, const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("reflection_probe_set_ambient_color")._native_ptr(), 2948539648);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_probe, &p_color);
}

void RenderingServer::reflection_probe_set_ambient_energy(const RID &p_probe, float p_energy) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("reflection_probe_set_ambient_energy")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_energy_encoded;
	PtrToArg<double>::encode(p_energy, &p_energy_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_probe, &p_energy_encoded);
}

void RenderingServer::reflection_probe_set_max_distance(const RID &p_probe, float p_distance) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("reflection_probe_set_max_distance")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_distance_encoded;
	PtrToArg<double>::encode(p_distance, &p_distance_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_probe, &p_distance_encoded);
}

void RenderingServer::reflection_probe_set_size(const RID &p_probe, const Vector3 &p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("reflection_probe_set_size")._native_ptr(), 3227306858);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_probe, &p_size);
}

void RenderingServer::reflection_probe_set_origin_offset(const RID &p_probe, const Vector3 &p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("reflection_probe_set_origin_offset")._native_ptr(), 3227306858);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_probe, &p_offset);
}

void RenderingServer::reflection_probe_set_as_interior(const RID &p_probe, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("reflection_probe_set_as_interior")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_probe, &p_enable_encoded);
}

void RenderingServer::reflection_probe_set_enable_box_projection(const RID &p_probe, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("reflection_probe_set_enable_box_projection")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_probe, &p_enable_encoded);
}

void RenderingServer::reflection_probe_set_enable_shadows(const RID &p_probe, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("reflection_probe_set_enable_shadows")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_probe, &p_enable_encoded);
}

void RenderingServer::reflection_probe_set_cull_mask(const RID &p_probe, uint32_t p_layers) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("reflection_probe_set_cull_mask")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layers_encoded;
	PtrToArg<int64_t>::encode(p_layers, &p_layers_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_probe, &p_layers_encoded);
}

void RenderingServer::reflection_probe_set_reflection_mask(const RID &p_probe, uint32_t p_layers) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("reflection_probe_set_reflection_mask")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layers_encoded;
	PtrToArg<int64_t>::encode(p_layers, &p_layers_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_probe, &p_layers_encoded);
}

void RenderingServer::reflection_probe_set_resolution(const RID &p_probe, int32_t p_resolution) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("reflection_probe_set_resolution")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_resolution_encoded;
	PtrToArg<int64_t>::encode(p_resolution, &p_resolution_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_probe, &p_resolution_encoded);
}

void RenderingServer::reflection_probe_set_mesh_lod_threshold(const RID &p_probe, float p_pixels) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("reflection_probe_set_mesh_lod_threshold")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_pixels_encoded;
	PtrToArg<double>::encode(p_pixels, &p_pixels_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_probe, &p_pixels_encoded);
}

RID RenderingServer::decal_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("decal_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void RenderingServer::decal_set_size(const RID &p_decal, const Vector3 &p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("decal_set_size")._native_ptr(), 3227306858);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_decal, &p_size);
}

void RenderingServer::decal_set_texture(const RID &p_decal, RenderingServer::DecalTexture p_type, const RID &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("decal_set_texture")._native_ptr(), 3953344054);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_decal, &p_type_encoded, &p_texture);
}

void RenderingServer::decal_set_emission_energy(const RID &p_decal, float p_energy) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("decal_set_emission_energy")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_energy_encoded;
	PtrToArg<double>::encode(p_energy, &p_energy_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_decal, &p_energy_encoded);
}

void RenderingServer::decal_set_albedo_mix(const RID &p_decal, float p_albedo_mix) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("decal_set_albedo_mix")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_albedo_mix_encoded;
	PtrToArg<double>::encode(p_albedo_mix, &p_albedo_mix_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_decal, &p_albedo_mix_encoded);
}

void RenderingServer::decal_set_modulate(const RID &p_decal, const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("decal_set_modulate")._native_ptr(), 2948539648);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_decal, &p_color);
}

void RenderingServer::decal_set_cull_mask(const RID &p_decal, uint32_t p_mask) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("decal_set_cull_mask")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mask_encoded;
	PtrToArg<int64_t>::encode(p_mask, &p_mask_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_decal, &p_mask_encoded);
}

void RenderingServer::decal_set_distance_fade(const RID &p_decal, bool p_enabled, float p_begin, float p_length) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("decal_set_distance_fade")._native_ptr(), 2972769666);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	double p_begin_encoded;
	PtrToArg<double>::encode(p_begin, &p_begin_encoded);
	double p_length_encoded;
	PtrToArg<double>::encode(p_length, &p_length_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_decal, &p_enabled_encoded, &p_begin_encoded, &p_length_encoded);
}

void RenderingServer::decal_set_fade(const RID &p_decal, float p_above, float p_below) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("decal_set_fade")._native_ptr(), 2513314492);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_above_encoded;
	PtrToArg<double>::encode(p_above, &p_above_encoded);
	double p_below_encoded;
	PtrToArg<double>::encode(p_below, &p_below_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_decal, &p_above_encoded, &p_below_encoded);
}

void RenderingServer::decal_set_normal_fade(const RID &p_decal, float p_fade) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("decal_set_normal_fade")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_fade_encoded;
	PtrToArg<double>::encode(p_fade, &p_fade_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_decal, &p_fade_encoded);
}

void RenderingServer::decals_set_filter(RenderingServer::DecalFilter p_filter) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("decals_set_filter")._native_ptr(), 3519875702);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_filter_encoded;
	PtrToArg<int64_t>::encode(p_filter, &p_filter_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_filter_encoded);
}

void RenderingServer::gi_set_use_half_resolution(bool p_half_resolution) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("gi_set_use_half_resolution")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_half_resolution_encoded;
	PtrToArg<bool>::encode(p_half_resolution, &p_half_resolution_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_half_resolution_encoded);
}

RID RenderingServer::voxel_gi_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("voxel_gi_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void RenderingServer::voxel_gi_allocate_data(const RID &p_voxel_gi, const Transform3D &p_to_cell_xform, const AABB &p_aabb, const Vector3i &p_octree_size, const PackedByteArray &p_octree_cells, const PackedByteArray &p_data_cells, const PackedByteArray &p_distance_field, const PackedInt32Array &p_level_counts) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("voxel_gi_allocate_data")._native_ptr(), 4108223027);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_voxel_gi, &p_to_cell_xform, &p_aabb, &p_octree_size, &p_octree_cells, &p_data_cells, &p_distance_field, &p_level_counts);
}

Vector3i RenderingServer::voxel_gi_get_octree_size(const RID &p_voxel_gi) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("voxel_gi_get_octree_size")._native_ptr(), 2607699645);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3i()));
	return ::godot::internal::_call_native_mb_ret<Vector3i>(_gde_method_bind, _owner, &p_voxel_gi);
}

PackedByteArray RenderingServer::voxel_gi_get_octree_cells(const RID &p_voxel_gi) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("voxel_gi_get_octree_cells")._native_ptr(), 3348040486);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedByteArray()));
	return ::godot::internal::_call_native_mb_ret<PackedByteArray>(_gde_method_bind, _owner, &p_voxel_gi);
}

PackedByteArray RenderingServer::voxel_gi_get_data_cells(const RID &p_voxel_gi) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("voxel_gi_get_data_cells")._native_ptr(), 3348040486);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedByteArray()));
	return ::godot::internal::_call_native_mb_ret<PackedByteArray>(_gde_method_bind, _owner, &p_voxel_gi);
}

PackedByteArray RenderingServer::voxel_gi_get_distance_field(const RID &p_voxel_gi) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("voxel_gi_get_distance_field")._native_ptr(), 3348040486);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedByteArray()));
	return ::godot::internal::_call_native_mb_ret<PackedByteArray>(_gde_method_bind, _owner, &p_voxel_gi);
}

PackedInt32Array RenderingServer::voxel_gi_get_level_counts(const RID &p_voxel_gi) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("voxel_gi_get_level_counts")._native_ptr(), 788230395);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner, &p_voxel_gi);
}

Transform3D RenderingServer::voxel_gi_get_to_cell_xform(const RID &p_voxel_gi) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("voxel_gi_get_to_cell_xform")._native_ptr(), 1128465797);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform3D()));
	return ::godot::internal::_call_native_mb_ret<Transform3D>(_gde_method_bind, _owner, &p_voxel_gi);
}

void RenderingServer::voxel_gi_set_dynamic_range(const RID &p_voxel_gi, float p_range) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("voxel_gi_set_dynamic_range")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_range_encoded;
	PtrToArg<double>::encode(p_range, &p_range_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_voxel_gi, &p_range_encoded);
}

void RenderingServer::voxel_gi_set_propagation(const RID &p_voxel_gi, float p_amount) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("voxel_gi_set_propagation")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_amount_encoded;
	PtrToArg<double>::encode(p_amount, &p_amount_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_voxel_gi, &p_amount_encoded);
}

void RenderingServer::voxel_gi_set_energy(const RID &p_voxel_gi, float p_energy) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("voxel_gi_set_energy")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_energy_encoded;
	PtrToArg<double>::encode(p_energy, &p_energy_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_voxel_gi, &p_energy_encoded);
}

void RenderingServer::voxel_gi_set_baked_exposure_normalization(const RID &p_voxel_gi, float p_baked_exposure) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("voxel_gi_set_baked_exposure_normalization")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_baked_exposure_encoded;
	PtrToArg<double>::encode(p_baked_exposure, &p_baked_exposure_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_voxel_gi, &p_baked_exposure_encoded);
}

void RenderingServer::voxel_gi_set_bias(const RID &p_voxel_gi, float p_bias) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("voxel_gi_set_bias")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_bias_encoded;
	PtrToArg<double>::encode(p_bias, &p_bias_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_voxel_gi, &p_bias_encoded);
}

void RenderingServer::voxel_gi_set_normal_bias(const RID &p_voxel_gi, float p_bias) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("voxel_gi_set_normal_bias")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_bias_encoded;
	PtrToArg<double>::encode(p_bias, &p_bias_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_voxel_gi, &p_bias_encoded);
}

void RenderingServer::voxel_gi_set_interior(const RID &p_voxel_gi, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("voxel_gi_set_interior")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_voxel_gi, &p_enable_encoded);
}

void RenderingServer::voxel_gi_set_use_two_bounces(const RID &p_voxel_gi, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("voxel_gi_set_use_two_bounces")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_voxel_gi, &p_enable_encoded);
}

void RenderingServer::voxel_gi_set_quality(RenderingServer::VoxelGIQuality p_quality) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("voxel_gi_set_quality")._native_ptr(), 1538689978);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_quality_encoded;
	PtrToArg<int64_t>::encode(p_quality, &p_quality_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_quality_encoded);
}

RID RenderingServer::lightmap_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("lightmap_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void RenderingServer::lightmap_set_textures(const RID &p_lightmap, const RID &p_light, bool p_uses_sh) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("lightmap_set_textures")._native_ptr(), 2646464759);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_uses_sh_encoded;
	PtrToArg<bool>::encode(p_uses_sh, &p_uses_sh_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_lightmap, &p_light, &p_uses_sh_encoded);
}

void RenderingServer::lightmap_set_probe_bounds(const RID &p_lightmap, const AABB &p_bounds) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("lightmap_set_probe_bounds")._native_ptr(), 3696536120);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_lightmap, &p_bounds);
}

void RenderingServer::lightmap_set_probe_interior(const RID &p_lightmap, bool p_interior) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("lightmap_set_probe_interior")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_interior_encoded;
	PtrToArg<bool>::encode(p_interior, &p_interior_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_lightmap, &p_interior_encoded);
}

void RenderingServer::lightmap_set_probe_capture_data(const RID &p_lightmap, const PackedVector3Array &p_points, const PackedColorArray &p_point_sh, const PackedInt32Array &p_tetrahedra, const PackedInt32Array &p_bsp_tree) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("lightmap_set_probe_capture_data")._native_ptr(), 3217845880);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_lightmap, &p_points, &p_point_sh, &p_tetrahedra, &p_bsp_tree);
}

PackedVector3Array RenderingServer::lightmap_get_probe_capture_points(const RID &p_lightmap) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("lightmap_get_probe_capture_points")._native_ptr(), 808965560);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedVector3Array()));
	return ::godot::internal::_call_native_mb_ret<PackedVector3Array>(_gde_method_bind, _owner, &p_lightmap);
}

PackedColorArray RenderingServer::lightmap_get_probe_capture_sh(const RID &p_lightmap) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("lightmap_get_probe_capture_sh")._native_ptr(), 1569415609);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedColorArray()));
	return ::godot::internal::_call_native_mb_ret<PackedColorArray>(_gde_method_bind, _owner, &p_lightmap);
}

PackedInt32Array RenderingServer::lightmap_get_probe_capture_tetrahedra(const RID &p_lightmap) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("lightmap_get_probe_capture_tetrahedra")._native_ptr(), 788230395);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner, &p_lightmap);
}

PackedInt32Array RenderingServer::lightmap_get_probe_capture_bsp_tree(const RID &p_lightmap) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("lightmap_get_probe_capture_bsp_tree")._native_ptr(), 788230395);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner, &p_lightmap);
}

void RenderingServer::lightmap_set_baked_exposure_normalization(const RID &p_lightmap, float p_baked_exposure) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("lightmap_set_baked_exposure_normalization")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_baked_exposure_encoded;
	PtrToArg<double>::encode(p_baked_exposure, &p_baked_exposure_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_lightmap, &p_baked_exposure_encoded);
}

void RenderingServer::lightmap_set_probe_capture_update_speed(float p_speed) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("lightmap_set_probe_capture_update_speed")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_speed_encoded;
	PtrToArg<double>::encode(p_speed, &p_speed_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_speed_encoded);
}

RID RenderingServer::particles_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void RenderingServer::particles_set_mode(const RID &p_particles, RenderingServer::ParticlesMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_set_mode")._native_ptr(), 3492270028);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles, &p_mode_encoded);
}

void RenderingServer::particles_set_emitting(const RID &p_particles, bool p_emitting) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_set_emitting")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_emitting_encoded;
	PtrToArg<bool>::encode(p_emitting, &p_emitting_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles, &p_emitting_encoded);
}

bool RenderingServer::particles_get_emitting(const RID &p_particles) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_get_emitting")._native_ptr(), 3521089500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_particles);
}

void RenderingServer::particles_set_amount(const RID &p_particles, int32_t p_amount) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_set_amount")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_amount_encoded;
	PtrToArg<int64_t>::encode(p_amount, &p_amount_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles, &p_amount_encoded);
}

void RenderingServer::particles_set_amount_ratio(const RID &p_particles, float p_ratio) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_set_amount_ratio")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_ratio_encoded;
	PtrToArg<double>::encode(p_ratio, &p_ratio_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles, &p_ratio_encoded);
}

void RenderingServer::particles_set_lifetime(const RID &p_particles, double p_lifetime) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_set_lifetime")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_lifetime_encoded;
	PtrToArg<double>::encode(p_lifetime, &p_lifetime_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles, &p_lifetime_encoded);
}

void RenderingServer::particles_set_one_shot(const RID &p_particles, bool p_one_shot) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_set_one_shot")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_one_shot_encoded;
	PtrToArg<bool>::encode(p_one_shot, &p_one_shot_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles, &p_one_shot_encoded);
}

void RenderingServer::particles_set_pre_process_time(const RID &p_particles, double p_time) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_set_pre_process_time")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_time_encoded;
	PtrToArg<double>::encode(p_time, &p_time_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles, &p_time_encoded);
}

void RenderingServer::particles_request_process_time(const RID &p_particles, float p_time) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_request_process_time")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_time_encoded;
	PtrToArg<double>::encode(p_time, &p_time_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles, &p_time_encoded);
}

void RenderingServer::particles_set_explosiveness_ratio(const RID &p_particles, float p_ratio) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_set_explosiveness_ratio")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_ratio_encoded;
	PtrToArg<double>::encode(p_ratio, &p_ratio_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles, &p_ratio_encoded);
}

void RenderingServer::particles_set_randomness_ratio(const RID &p_particles, float p_ratio) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_set_randomness_ratio")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_ratio_encoded;
	PtrToArg<double>::encode(p_ratio, &p_ratio_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles, &p_ratio_encoded);
}

void RenderingServer::particles_set_interp_to_end(const RID &p_particles, float p_factor) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_set_interp_to_end")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_factor_encoded;
	PtrToArg<double>::encode(p_factor, &p_factor_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles, &p_factor_encoded);
}

void RenderingServer::particles_set_emitter_velocity(const RID &p_particles, const Vector3 &p_velocity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_set_emitter_velocity")._native_ptr(), 3227306858);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles, &p_velocity);
}

void RenderingServer::particles_set_custom_aabb(const RID &p_particles, const AABB &p_aabb) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_set_custom_aabb")._native_ptr(), 3696536120);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles, &p_aabb);
}

void RenderingServer::particles_set_speed_scale(const RID &p_particles, double p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_set_speed_scale")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_scale_encoded;
	PtrToArg<double>::encode(p_scale, &p_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles, &p_scale_encoded);
}

void RenderingServer::particles_set_use_local_coordinates(const RID &p_particles, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_set_use_local_coordinates")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles, &p_enable_encoded);
}

void RenderingServer::particles_set_process_material(const RID &p_particles, const RID &p_material) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_set_process_material")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles, &p_material);
}

void RenderingServer::particles_set_fixed_fps(const RID &p_particles, int32_t p_fps) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_set_fixed_fps")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_fps_encoded;
	PtrToArg<int64_t>::encode(p_fps, &p_fps_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles, &p_fps_encoded);
}

void RenderingServer::particles_set_interpolate(const RID &p_particles, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_set_interpolate")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles, &p_enable_encoded);
}

void RenderingServer::particles_set_fractional_delta(const RID &p_particles, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_set_fractional_delta")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles, &p_enable_encoded);
}

void RenderingServer::particles_set_collision_base_size(const RID &p_particles, float p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_set_collision_base_size")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_size_encoded;
	PtrToArg<double>::encode(p_size, &p_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles, &p_size_encoded);
}

void RenderingServer::particles_set_transform_align(const RID &p_particles, RenderingServer::ParticlesTransformAlign p_align) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_set_transform_align")._native_ptr(), 3264971368);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_align_encoded;
	PtrToArg<int64_t>::encode(p_align, &p_align_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles, &p_align_encoded);
}

void RenderingServer::particles_set_trails(const RID &p_particles, bool p_enable, float p_length_sec) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_set_trails")._native_ptr(), 2010054925);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	double p_length_sec_encoded;
	PtrToArg<double>::encode(p_length_sec, &p_length_sec_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles, &p_enable_encoded, &p_length_sec_encoded);
}

void RenderingServer::particles_set_trail_bind_poses(const RID &p_particles, const TypedArray<Transform3D> &p_bind_poses) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_set_trail_bind_poses")._native_ptr(), 684822712);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles, &p_bind_poses);
}

bool RenderingServer::particles_is_inactive(const RID &p_particles) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_is_inactive")._native_ptr(), 3521089500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_particles);
}

void RenderingServer::particles_request_process(const RID &p_particles) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_request_process")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles);
}

void RenderingServer::particles_restart(const RID &p_particles) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_restart")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles);
}

void RenderingServer::particles_set_subemitter(const RID &p_particles, const RID &p_subemitter_particles) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_set_subemitter")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles, &p_subemitter_particles);
}

void RenderingServer::particles_emit(const RID &p_particles, const Transform3D &p_transform, const Vector3 &p_velocity, const Color &p_color, const Color &p_custom, uint32_t p_emit_flags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_emit")._native_ptr(), 4043136117);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_emit_flags_encoded;
	PtrToArg<int64_t>::encode(p_emit_flags, &p_emit_flags_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles, &p_transform, &p_velocity, &p_color, &p_custom, &p_emit_flags_encoded);
}

void RenderingServer::particles_set_draw_order(const RID &p_particles, RenderingServer::ParticlesDrawOrder p_order) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_set_draw_order")._native_ptr(), 935028487);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_order_encoded;
	PtrToArg<int64_t>::encode(p_order, &p_order_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles, &p_order_encoded);
}

void RenderingServer::particles_set_draw_passes(const RID &p_particles, int32_t p_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_set_draw_passes")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_count_encoded;
	PtrToArg<int64_t>::encode(p_count, &p_count_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles, &p_count_encoded);
}

void RenderingServer::particles_set_draw_pass_mesh(const RID &p_particles, int32_t p_pass, const RID &p_mesh) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_set_draw_pass_mesh")._native_ptr(), 2310537182);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_pass_encoded;
	PtrToArg<int64_t>::encode(p_pass, &p_pass_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles, &p_pass_encoded, &p_mesh);
}

AABB RenderingServer::particles_get_current_aabb(const RID &p_particles) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_get_current_aabb")._native_ptr(), 3952830260);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AABB()));
	return ::godot::internal::_call_native_mb_ret<AABB>(_gde_method_bind, _owner, &p_particles);
}

void RenderingServer::particles_set_emission_transform(const RID &p_particles, const Transform3D &p_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_set_emission_transform")._native_ptr(), 3935195649);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles, &p_transform);
}

RID RenderingServer::particles_collision_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_collision_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void RenderingServer::particles_collision_set_collision_type(const RID &p_particles_collision, RenderingServer::ParticlesCollisionType p_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_collision_set_collision_type")._native_ptr(), 1497044930);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles_collision, &p_type_encoded);
}

void RenderingServer::particles_collision_set_cull_mask(const RID &p_particles_collision, uint32_t p_mask) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_collision_set_cull_mask")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mask_encoded;
	PtrToArg<int64_t>::encode(p_mask, &p_mask_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles_collision, &p_mask_encoded);
}

void RenderingServer::particles_collision_set_sphere_radius(const RID &p_particles_collision, float p_radius) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_collision_set_sphere_radius")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_radius_encoded;
	PtrToArg<double>::encode(p_radius, &p_radius_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles_collision, &p_radius_encoded);
}

void RenderingServer::particles_collision_set_box_extents(const RID &p_particles_collision, const Vector3 &p_extents) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_collision_set_box_extents")._native_ptr(), 3227306858);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles_collision, &p_extents);
}

void RenderingServer::particles_collision_set_attractor_strength(const RID &p_particles_collision, float p_strength) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_collision_set_attractor_strength")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_strength_encoded;
	PtrToArg<double>::encode(p_strength, &p_strength_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles_collision, &p_strength_encoded);
}

void RenderingServer::particles_collision_set_attractor_directionality(const RID &p_particles_collision, float p_amount) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_collision_set_attractor_directionality")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_amount_encoded;
	PtrToArg<double>::encode(p_amount, &p_amount_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles_collision, &p_amount_encoded);
}

void RenderingServer::particles_collision_set_attractor_attenuation(const RID &p_particles_collision, float p_curve) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_collision_set_attractor_attenuation")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_curve_encoded;
	PtrToArg<double>::encode(p_curve, &p_curve_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles_collision, &p_curve_encoded);
}

void RenderingServer::particles_collision_set_field_texture(const RID &p_particles_collision, const RID &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_collision_set_field_texture")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles_collision, &p_texture);
}

void RenderingServer::particles_collision_height_field_update(const RID &p_particles_collision) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_collision_height_field_update")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles_collision);
}

void RenderingServer::particles_collision_set_height_field_resolution(const RID &p_particles_collision, RenderingServer::ParticlesCollisionHeightfieldResolution p_resolution) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_collision_set_height_field_resolution")._native_ptr(), 962977297);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_resolution_encoded;
	PtrToArg<int64_t>::encode(p_resolution, &p_resolution_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles_collision, &p_resolution_encoded);
}

void RenderingServer::particles_collision_set_height_field_mask(const RID &p_particles_collision, uint32_t p_mask) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("particles_collision_set_height_field_mask")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mask_encoded;
	PtrToArg<int64_t>::encode(p_mask, &p_mask_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles_collision, &p_mask_encoded);
}

RID RenderingServer::fog_volume_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("fog_volume_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void RenderingServer::fog_volume_set_shape(const RID &p_fog_volume, RenderingServer::FogVolumeShape p_shape) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("fog_volume_set_shape")._native_ptr(), 3818703106);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_shape_encoded;
	PtrToArg<int64_t>::encode(p_shape, &p_shape_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_fog_volume, &p_shape_encoded);
}

void RenderingServer::fog_volume_set_size(const RID &p_fog_volume, const Vector3 &p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("fog_volume_set_size")._native_ptr(), 3227306858);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_fog_volume, &p_size);
}

void RenderingServer::fog_volume_set_material(const RID &p_fog_volume, const RID &p_material) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("fog_volume_set_material")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_fog_volume, &p_material);
}

RID RenderingServer::visibility_notifier_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("visibility_notifier_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void RenderingServer::visibility_notifier_set_aabb(const RID &p_notifier, const AABB &p_aabb) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("visibility_notifier_set_aabb")._native_ptr(), 3696536120);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_notifier, &p_aabb);
}

void RenderingServer::visibility_notifier_set_callbacks(const RID &p_notifier, const Callable &p_enter_callable, const Callable &p_exit_callable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("visibility_notifier_set_callbacks")._native_ptr(), 2689735388);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_notifier, &p_enter_callable, &p_exit_callable);
}

RID RenderingServer::occluder_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("occluder_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void RenderingServer::occluder_set_mesh(const RID &p_occluder, const PackedVector3Array &p_vertices, const PackedInt32Array &p_indices) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("occluder_set_mesh")._native_ptr(), 3854404263);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_occluder, &p_vertices, &p_indices);
}

RID RenderingServer::camera_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("camera_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void RenderingServer::camera_set_perspective(const RID &p_camera, float p_fovy_degrees, float p_z_near, float p_z_far) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("camera_set_perspective")._native_ptr(), 157498339);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_fovy_degrees_encoded;
	PtrToArg<double>::encode(p_fovy_degrees, &p_fovy_degrees_encoded);
	double p_z_near_encoded;
	PtrToArg<double>::encode(p_z_near, &p_z_near_encoded);
	double p_z_far_encoded;
	PtrToArg<double>::encode(p_z_far, &p_z_far_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_camera, &p_fovy_degrees_encoded, &p_z_near_encoded, &p_z_far_encoded);
}

void RenderingServer::camera_set_orthogonal(const RID &p_camera, float p_size, float p_z_near, float p_z_far) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("camera_set_orthogonal")._native_ptr(), 157498339);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_size_encoded;
	PtrToArg<double>::encode(p_size, &p_size_encoded);
	double p_z_near_encoded;
	PtrToArg<double>::encode(p_z_near, &p_z_near_encoded);
	double p_z_far_encoded;
	PtrToArg<double>::encode(p_z_far, &p_z_far_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_camera, &p_size_encoded, &p_z_near_encoded, &p_z_far_encoded);
}

void RenderingServer::camera_set_frustum(const RID &p_camera, float p_size, const Vector2 &p_offset, float p_z_near, float p_z_far) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("camera_set_frustum")._native_ptr(), 1889878953);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_size_encoded;
	PtrToArg<double>::encode(p_size, &p_size_encoded);
	double p_z_near_encoded;
	PtrToArg<double>::encode(p_z_near, &p_z_near_encoded);
	double p_z_far_encoded;
	PtrToArg<double>::encode(p_z_far, &p_z_far_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_camera, &p_size_encoded, &p_offset, &p_z_near_encoded, &p_z_far_encoded);
}

void RenderingServer::camera_set_transform(const RID &p_camera, const Transform3D &p_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("camera_set_transform")._native_ptr(), 3935195649);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_camera, &p_transform);
}

void RenderingServer::camera_set_cull_mask(const RID &p_camera, uint32_t p_layers) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("camera_set_cull_mask")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layers_encoded;
	PtrToArg<int64_t>::encode(p_layers, &p_layers_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_camera, &p_layers_encoded);
}

void RenderingServer::camera_set_environment(const RID &p_camera, const RID &p_env) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("camera_set_environment")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_camera, &p_env);
}

void RenderingServer::camera_set_camera_attributes(const RID &p_camera, const RID &p_effects) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("camera_set_camera_attributes")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_camera, &p_effects);
}

void RenderingServer::camera_set_compositor(const RID &p_camera, const RID &p_compositor) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("camera_set_compositor")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_camera, &p_compositor);
}

void RenderingServer::camera_set_use_vertical_aspect(const RID &p_camera, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("camera_set_use_vertical_aspect")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_camera, &p_enable_encoded);
}

RID RenderingServer::viewport_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void RenderingServer::viewport_set_use_xr(const RID &p_viewport, bool p_use_xr) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_use_xr")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_use_xr_encoded;
	PtrToArg<bool>::encode(p_use_xr, &p_use_xr_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_use_xr_encoded);
}

void RenderingServer::viewport_set_size(const RID &p_viewport, int32_t p_width, int32_t p_height) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_size")._native_ptr(), 4288446313);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_width_encoded;
	PtrToArg<int64_t>::encode(p_width, &p_width_encoded);
	int64_t p_height_encoded;
	PtrToArg<int64_t>::encode(p_height, &p_height_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_width_encoded, &p_height_encoded);
}

void RenderingServer::viewport_set_active(const RID &p_viewport, bool p_active) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_active")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_active_encoded;
	PtrToArg<bool>::encode(p_active, &p_active_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_active_encoded);
}

void RenderingServer::viewport_set_parent_viewport(const RID &p_viewport, const RID &p_parent_viewport) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_parent_viewport")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_parent_viewport);
}

void RenderingServer::viewport_attach_to_screen(const RID &p_viewport, const Rect2 &p_rect, int32_t p_screen) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_attach_to_screen")._native_ptr(), 1062245816);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_screen_encoded;
	PtrToArg<int64_t>::encode(p_screen, &p_screen_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_rect, &p_screen_encoded);
}

void RenderingServer::viewport_set_render_direct_to_screen(const RID &p_viewport, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_render_direct_to_screen")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_enabled_encoded);
}

void RenderingServer::viewport_set_canvas_cull_mask(const RID &p_viewport, uint32_t p_canvas_cull_mask) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_canvas_cull_mask")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_canvas_cull_mask_encoded;
	PtrToArg<int64_t>::encode(p_canvas_cull_mask, &p_canvas_cull_mask_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_canvas_cull_mask_encoded);
}

void RenderingServer::viewport_set_scaling_3d_mode(const RID &p_viewport, RenderingServer::ViewportScaling3DMode p_scaling_3d_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_scaling_3d_mode")._native_ptr(), 2386524376);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_scaling_3d_mode_encoded;
	PtrToArg<int64_t>::encode(p_scaling_3d_mode, &p_scaling_3d_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_scaling_3d_mode_encoded);
}

void RenderingServer::viewport_set_scaling_3d_scale(const RID &p_viewport, float p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_scaling_3d_scale")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_scale_encoded;
	PtrToArg<double>::encode(p_scale, &p_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_scale_encoded);
}

void RenderingServer::viewport_set_fsr_sharpness(const RID &p_viewport, float p_sharpness) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_fsr_sharpness")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_sharpness_encoded;
	PtrToArg<double>::encode(p_sharpness, &p_sharpness_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_sharpness_encoded);
}

void RenderingServer::viewport_set_texture_mipmap_bias(const RID &p_viewport, float p_mipmap_bias) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_texture_mipmap_bias")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_mipmap_bias_encoded;
	PtrToArg<double>::encode(p_mipmap_bias, &p_mipmap_bias_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_mipmap_bias_encoded);
}

void RenderingServer::viewport_set_anisotropic_filtering_level(const RID &p_viewport, RenderingServer::ViewportAnisotropicFiltering p_anisotropic_filtering_level) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_anisotropic_filtering_level")._native_ptr(), 3953214029);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_anisotropic_filtering_level_encoded;
	PtrToArg<int64_t>::encode(p_anisotropic_filtering_level, &p_anisotropic_filtering_level_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_anisotropic_filtering_level_encoded);
}

void RenderingServer::viewport_set_update_mode(const RID &p_viewport, RenderingServer::ViewportUpdateMode p_update_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_update_mode")._native_ptr(), 3161116010);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_update_mode_encoded;
	PtrToArg<int64_t>::encode(p_update_mode, &p_update_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_update_mode_encoded);
}

RenderingServer::ViewportUpdateMode RenderingServer::viewport_get_update_mode(const RID &p_viewport) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_get_update_mode")._native_ptr(), 3803901472);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingServer::ViewportUpdateMode(0)));
	return (RenderingServer::ViewportUpdateMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_viewport);
}

void RenderingServer::viewport_set_clear_mode(const RID &p_viewport, RenderingServer::ViewportClearMode p_clear_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_clear_mode")._native_ptr(), 3628367896);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_clear_mode_encoded;
	PtrToArg<int64_t>::encode(p_clear_mode, &p_clear_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_clear_mode_encoded);
}

RID RenderingServer::viewport_get_render_target(const RID &p_viewport) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_get_render_target")._native_ptr(), 3814569979);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_viewport);
}

RID RenderingServer::viewport_get_texture(const RID &p_viewport) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_get_texture")._native_ptr(), 3814569979);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_viewport);
}

void RenderingServer::viewport_set_disable_3d(const RID &p_viewport, bool p_disable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_disable_3d")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_disable_encoded;
	PtrToArg<bool>::encode(p_disable, &p_disable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_disable_encoded);
}

void RenderingServer::viewport_set_disable_2d(const RID &p_viewport, bool p_disable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_disable_2d")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_disable_encoded;
	PtrToArg<bool>::encode(p_disable, &p_disable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_disable_encoded);
}

void RenderingServer::viewport_set_environment_mode(const RID &p_viewport, RenderingServer::ViewportEnvironmentMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_environment_mode")._native_ptr(), 2196892182);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_mode_encoded);
}

void RenderingServer::viewport_attach_camera(const RID &p_viewport, const RID &p_camera) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_attach_camera")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_camera);
}

void RenderingServer::viewport_set_scenario(const RID &p_viewport, const RID &p_scenario) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_scenario")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_scenario);
}

void RenderingServer::viewport_attach_canvas(const RID &p_viewport, const RID &p_canvas) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_attach_canvas")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_canvas);
}

void RenderingServer::viewport_remove_canvas(const RID &p_viewport, const RID &p_canvas) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_remove_canvas")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_canvas);
}

void RenderingServer::viewport_set_snap_2d_transforms_to_pixel(const RID &p_viewport, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_snap_2d_transforms_to_pixel")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_enabled_encoded);
}

void RenderingServer::viewport_set_snap_2d_vertices_to_pixel(const RID &p_viewport, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_snap_2d_vertices_to_pixel")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_enabled_encoded);
}

void RenderingServer::viewport_set_default_canvas_item_texture_filter(const RID &p_viewport, RenderingServer::CanvasItemTextureFilter p_filter) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_default_canvas_item_texture_filter")._native_ptr(), 1155129294);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_filter_encoded;
	PtrToArg<int64_t>::encode(p_filter, &p_filter_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_filter_encoded);
}

void RenderingServer::viewport_set_default_canvas_item_texture_repeat(const RID &p_viewport, RenderingServer::CanvasItemTextureRepeat p_repeat) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_default_canvas_item_texture_repeat")._native_ptr(), 1652956681);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_repeat_encoded;
	PtrToArg<int64_t>::encode(p_repeat, &p_repeat_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_repeat_encoded);
}

void RenderingServer::viewport_set_canvas_transform(const RID &p_viewport, const RID &p_canvas, const Transform2D &p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_canvas_transform")._native_ptr(), 3608606053);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_canvas, &p_offset);
}

void RenderingServer::viewport_set_canvas_stacking(const RID &p_viewport, const RID &p_canvas, int32_t p_layer, int32_t p_sublayer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_canvas_stacking")._native_ptr(), 3713930247);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	int64_t p_sublayer_encoded;
	PtrToArg<int64_t>::encode(p_sublayer, &p_sublayer_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_canvas, &p_layer_encoded, &p_sublayer_encoded);
}

void RenderingServer::viewport_set_transparent_background(const RID &p_viewport, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_transparent_background")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_enabled_encoded);
}

void RenderingServer::viewport_set_global_canvas_transform(const RID &p_viewport, const Transform2D &p_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_global_canvas_transform")._native_ptr(), 1246044741);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_transform);
}

void RenderingServer::viewport_set_sdf_oversize_and_scale(const RID &p_viewport, RenderingServer::ViewportSDFOversize p_oversize, RenderingServer::ViewportSDFScale p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_sdf_oversize_and_scale")._native_ptr(), 1329198632);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_oversize_encoded;
	PtrToArg<int64_t>::encode(p_oversize, &p_oversize_encoded);
	int64_t p_scale_encoded;
	PtrToArg<int64_t>::encode(p_scale, &p_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_oversize_encoded, &p_scale_encoded);
}

void RenderingServer::viewport_set_positional_shadow_atlas_size(const RID &p_viewport, int32_t p_size, bool p_use_16_bits) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_positional_shadow_atlas_size")._native_ptr(), 1904426712);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	int8_t p_use_16_bits_encoded;
	PtrToArg<bool>::encode(p_use_16_bits, &p_use_16_bits_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_size_encoded, &p_use_16_bits_encoded);
}

void RenderingServer::viewport_set_positional_shadow_atlas_quadrant_subdivision(const RID &p_viewport, int32_t p_quadrant, int32_t p_subdivision) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_positional_shadow_atlas_quadrant_subdivision")._native_ptr(), 4288446313);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_quadrant_encoded;
	PtrToArg<int64_t>::encode(p_quadrant, &p_quadrant_encoded);
	int64_t p_subdivision_encoded;
	PtrToArg<int64_t>::encode(p_subdivision, &p_subdivision_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_quadrant_encoded, &p_subdivision_encoded);
}

void RenderingServer::viewport_set_msaa_3d(const RID &p_viewport, RenderingServer::ViewportMSAA p_msaa) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_msaa_3d")._native_ptr(), 3764433340);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_msaa_encoded;
	PtrToArg<int64_t>::encode(p_msaa, &p_msaa_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_msaa_encoded);
}

void RenderingServer::viewport_set_msaa_2d(const RID &p_viewport, RenderingServer::ViewportMSAA p_msaa) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_msaa_2d")._native_ptr(), 3764433340);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_msaa_encoded;
	PtrToArg<int64_t>::encode(p_msaa, &p_msaa_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_msaa_encoded);
}

void RenderingServer::viewport_set_use_hdr_2d(const RID &p_viewport, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_use_hdr_2d")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_enabled_encoded);
}

void RenderingServer::viewport_set_screen_space_aa(const RID &p_viewport, RenderingServer::ViewportScreenSpaceAA p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_screen_space_aa")._native_ptr(), 1447279591);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_mode_encoded);
}

void RenderingServer::viewport_set_use_taa(const RID &p_viewport, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_use_taa")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_enable_encoded);
}

void RenderingServer::viewport_set_use_debanding(const RID &p_viewport, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_use_debanding")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_enable_encoded);
}

void RenderingServer::viewport_set_use_occlusion_culling(const RID &p_viewport, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_use_occlusion_culling")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_enable_encoded);
}

void RenderingServer::viewport_set_occlusion_rays_per_thread(int32_t p_rays_per_thread) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_occlusion_rays_per_thread")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_rays_per_thread_encoded;
	PtrToArg<int64_t>::encode(p_rays_per_thread, &p_rays_per_thread_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rays_per_thread_encoded);
}

void RenderingServer::viewport_set_occlusion_culling_build_quality(RenderingServer::ViewportOcclusionCullingBuildQuality p_quality) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_occlusion_culling_build_quality")._native_ptr(), 2069725696);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_quality_encoded;
	PtrToArg<int64_t>::encode(p_quality, &p_quality_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_quality_encoded);
}

int32_t RenderingServer::viewport_get_render_info(const RID &p_viewport, RenderingServer::ViewportRenderInfoType p_type, RenderingServer::ViewportRenderInfo p_info) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_get_render_info")._native_ptr(), 2041262392);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	int64_t p_info_encoded;
	PtrToArg<int64_t>::encode(p_info, &p_info_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_viewport, &p_type_encoded, &p_info_encoded);
}

void RenderingServer::viewport_set_debug_draw(const RID &p_viewport, RenderingServer::ViewportDebugDraw p_draw) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_debug_draw")._native_ptr(), 2089420930);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_draw_encoded;
	PtrToArg<int64_t>::encode(p_draw, &p_draw_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_draw_encoded);
}

void RenderingServer::viewport_set_measure_render_time(const RID &p_viewport, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_measure_render_time")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_enable_encoded);
}

double RenderingServer::viewport_get_measured_render_time_cpu(const RID &p_viewport) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_get_measured_render_time_cpu")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_viewport);
}

double RenderingServer::viewport_get_measured_render_time_gpu(const RID &p_viewport) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_get_measured_render_time_gpu")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_viewport);
}

void RenderingServer::viewport_set_vrs_mode(const RID &p_viewport, RenderingServer::ViewportVRSMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_vrs_mode")._native_ptr(), 398809874);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_mode_encoded);
}

void RenderingServer::viewport_set_vrs_update_mode(const RID &p_viewport, RenderingServer::ViewportVRSUpdateMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_vrs_update_mode")._native_ptr(), 2696154815);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_mode_encoded);
}

void RenderingServer::viewport_set_vrs_texture(const RID &p_viewport, const RID &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("viewport_set_vrs_texture")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_viewport, &p_texture);
}

RID RenderingServer::sky_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("sky_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void RenderingServer::sky_set_radiance_size(const RID &p_sky, int32_t p_radiance_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("sky_set_radiance_size")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_radiance_size_encoded;
	PtrToArg<int64_t>::encode(p_radiance_size, &p_radiance_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_sky, &p_radiance_size_encoded);
}

void RenderingServer::sky_set_mode(const RID &p_sky, RenderingServer::SkyMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("sky_set_mode")._native_ptr(), 3279019937);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_sky, &p_mode_encoded);
}

void RenderingServer::sky_set_material(const RID &p_sky, const RID &p_material) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("sky_set_material")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_sky, &p_material);
}

Ref<Image> RenderingServer::sky_bake_panorama(const RID &p_sky, float p_energy, bool p_bake_irradiance, const Vector2i &p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("sky_bake_panorama")._native_ptr(), 3875285818);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Image>()));
	double p_energy_encoded;
	PtrToArg<double>::encode(p_energy, &p_energy_encoded);
	int8_t p_bake_irradiance_encoded;
	PtrToArg<bool>::encode(p_bake_irradiance, &p_bake_irradiance_encoded);
	return Ref<Image>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Image>(_gde_method_bind, _owner, &p_sky, &p_energy_encoded, &p_bake_irradiance_encoded, &p_size));
}

RID RenderingServer::compositor_effect_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("compositor_effect_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void RenderingServer::compositor_effect_set_enabled(const RID &p_effect, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("compositor_effect_set_enabled")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_effect, &p_enabled_encoded);
}

void RenderingServer::compositor_effect_set_callback(const RID &p_effect, RenderingServer::CompositorEffectCallbackType p_callback_type, const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("compositor_effect_set_callback")._native_ptr(), 487412485);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_callback_type_encoded;
	PtrToArg<int64_t>::encode(p_callback_type, &p_callback_type_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_effect, &p_callback_type_encoded, &p_callback);
}

void RenderingServer::compositor_effect_set_flag(const RID &p_effect, RenderingServer::CompositorEffectFlags p_flag, bool p_set) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("compositor_effect_set_flag")._native_ptr(), 3659527075);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_flag_encoded;
	PtrToArg<int64_t>::encode(p_flag, &p_flag_encoded);
	int8_t p_set_encoded;
	PtrToArg<bool>::encode(p_set, &p_set_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_effect, &p_flag_encoded, &p_set_encoded);
}

RID RenderingServer::compositor_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("compositor_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void RenderingServer::compositor_set_compositor_effects(const RID &p_compositor, const TypedArray<RID> &p_effects) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("compositor_set_compositor_effects")._native_ptr(), 684822712);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_compositor, &p_effects);
}

RID RenderingServer::environment_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("environment_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void RenderingServer::environment_set_background(const RID &p_env, RenderingServer::EnvironmentBG p_bg) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("environment_set_background")._native_ptr(), 3937328877);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_bg_encoded;
	PtrToArg<int64_t>::encode(p_bg, &p_bg_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_env, &p_bg_encoded);
}

void RenderingServer::environment_set_camera_id(const RID &p_env, int32_t p_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("environment_set_camera_id")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_env, &p_id_encoded);
}

void RenderingServer::environment_set_sky(const RID &p_env, const RID &p_sky) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("environment_set_sky")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_env, &p_sky);
}

void RenderingServer::environment_set_sky_custom_fov(const RID &p_env, float p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("environment_set_sky_custom_fov")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_scale_encoded;
	PtrToArg<double>::encode(p_scale, &p_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_env, &p_scale_encoded);
}

void RenderingServer::environment_set_sky_orientation(const RID &p_env, const Basis &p_orientation) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("environment_set_sky_orientation")._native_ptr(), 1735850857);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_env, &p_orientation);
}

void RenderingServer::environment_set_bg_color(const RID &p_env, const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("environment_set_bg_color")._native_ptr(), 2948539648);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_env, &p_color);
}

void RenderingServer::environment_set_bg_energy(const RID &p_env, float p_multiplier, float p_exposure_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("environment_set_bg_energy")._native_ptr(), 2513314492);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_multiplier_encoded;
	PtrToArg<double>::encode(p_multiplier, &p_multiplier_encoded);
	double p_exposure_value_encoded;
	PtrToArg<double>::encode(p_exposure_value, &p_exposure_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_env, &p_multiplier_encoded, &p_exposure_value_encoded);
}

void RenderingServer::environment_set_canvas_max_layer(const RID &p_env, int32_t p_max_layer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("environment_set_canvas_max_layer")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_max_layer_encoded;
	PtrToArg<int64_t>::encode(p_max_layer, &p_max_layer_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_env, &p_max_layer_encoded);
}

void RenderingServer::environment_set_ambient_light(const RID &p_env, const Color &p_color, RenderingServer::EnvironmentAmbientSource p_ambient, float p_energy, float p_sky_contribution, RenderingServer::EnvironmentReflectionSource p_reflection_source) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("environment_set_ambient_light")._native_ptr(), 1214961493);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_ambient_encoded;
	PtrToArg<int64_t>::encode(p_ambient, &p_ambient_encoded);
	double p_energy_encoded;
	PtrToArg<double>::encode(p_energy, &p_energy_encoded);
	double p_sky_contribution_encoded;
	PtrToArg<double>::encode(p_sky_contribution, &p_sky_contribution_encoded);
	int64_t p_reflection_source_encoded;
	PtrToArg<int64_t>::encode(p_reflection_source, &p_reflection_source_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_env, &p_color, &p_ambient_encoded, &p_energy_encoded, &p_sky_contribution_encoded, &p_reflection_source_encoded);
}

void RenderingServer::environment_set_glow(const RID &p_env, bool p_enable, const PackedFloat32Array &p_levels, float p_intensity, float p_strength, float p_mix, float p_bloom_threshold, RenderingServer::EnvironmentGlowBlendMode p_blend_mode, float p_hdr_bleed_threshold, float p_hdr_bleed_scale, float p_hdr_luminance_cap, float p_glow_map_strength, const RID &p_glow_map) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("environment_set_glow")._native_ptr(), 2421724940);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	double p_intensity_encoded;
	PtrToArg<double>::encode(p_intensity, &p_intensity_encoded);
	double p_strength_encoded;
	PtrToArg<double>::encode(p_strength, &p_strength_encoded);
	double p_mix_encoded;
	PtrToArg<double>::encode(p_mix, &p_mix_encoded);
	double p_bloom_threshold_encoded;
	PtrToArg<double>::encode(p_bloom_threshold, &p_bloom_threshold_encoded);
	int64_t p_blend_mode_encoded;
	PtrToArg<int64_t>::encode(p_blend_mode, &p_blend_mode_encoded);
	double p_hdr_bleed_threshold_encoded;
	PtrToArg<double>::encode(p_hdr_bleed_threshold, &p_hdr_bleed_threshold_encoded);
	double p_hdr_bleed_scale_encoded;
	PtrToArg<double>::encode(p_hdr_bleed_scale, &p_hdr_bleed_scale_encoded);
	double p_hdr_luminance_cap_encoded;
	PtrToArg<double>::encode(p_hdr_luminance_cap, &p_hdr_luminance_cap_encoded);
	double p_glow_map_strength_encoded;
	PtrToArg<double>::encode(p_glow_map_strength, &p_glow_map_strength_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_env, &p_enable_encoded, &p_levels, &p_intensity_encoded, &p_strength_encoded, &p_mix_encoded, &p_bloom_threshold_encoded, &p_blend_mode_encoded, &p_hdr_bleed_threshold_encoded, &p_hdr_bleed_scale_encoded, &p_hdr_luminance_cap_encoded, &p_glow_map_strength_encoded, &p_glow_map);
}

void RenderingServer::environment_set_tonemap(const RID &p_env, RenderingServer::EnvironmentToneMapper p_tone_mapper, float p_exposure, float p_white) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("environment_set_tonemap")._native_ptr(), 2914312638);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_tone_mapper_encoded;
	PtrToArg<int64_t>::encode(p_tone_mapper, &p_tone_mapper_encoded);
	double p_exposure_encoded;
	PtrToArg<double>::encode(p_exposure, &p_exposure_encoded);
	double p_white_encoded;
	PtrToArg<double>::encode(p_white, &p_white_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_env, &p_tone_mapper_encoded, &p_exposure_encoded, &p_white_encoded);
}

void RenderingServer::environment_set_tonemap_agx_contrast(const RID &p_env, float p_agx_contrast) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("environment_set_tonemap_agx_contrast")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_agx_contrast_encoded;
	PtrToArg<double>::encode(p_agx_contrast, &p_agx_contrast_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_env, &p_agx_contrast_encoded);
}

void RenderingServer::environment_set_adjustment(const RID &p_env, bool p_enable, float p_brightness, float p_contrast, float p_saturation, bool p_use_1d_color_correction, const RID &p_color_correction) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("environment_set_adjustment")._native_ptr(), 876799838);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	double p_brightness_encoded;
	PtrToArg<double>::encode(p_brightness, &p_brightness_encoded);
	double p_contrast_encoded;
	PtrToArg<double>::encode(p_contrast, &p_contrast_encoded);
	double p_saturation_encoded;
	PtrToArg<double>::encode(p_saturation, &p_saturation_encoded);
	int8_t p_use_1d_color_correction_encoded;
	PtrToArg<bool>::encode(p_use_1d_color_correction, &p_use_1d_color_correction_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_env, &p_enable_encoded, &p_brightness_encoded, &p_contrast_encoded, &p_saturation_encoded, &p_use_1d_color_correction_encoded, &p_color_correction);
}

void RenderingServer::environment_set_ssr(const RID &p_env, bool p_enable, int32_t p_max_steps, float p_fade_in, float p_fade_out, float p_depth_tolerance) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("environment_set_ssr")._native_ptr(), 3607294374);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	int64_t p_max_steps_encoded;
	PtrToArg<int64_t>::encode(p_max_steps, &p_max_steps_encoded);
	double p_fade_in_encoded;
	PtrToArg<double>::encode(p_fade_in, &p_fade_in_encoded);
	double p_fade_out_encoded;
	PtrToArg<double>::encode(p_fade_out, &p_fade_out_encoded);
	double p_depth_tolerance_encoded;
	PtrToArg<double>::encode(p_depth_tolerance, &p_depth_tolerance_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_env, &p_enable_encoded, &p_max_steps_encoded, &p_fade_in_encoded, &p_fade_out_encoded, &p_depth_tolerance_encoded);
}

void RenderingServer::environment_set_ssao(const RID &p_env, bool p_enable, float p_radius, float p_intensity, float p_power, float p_detail, float p_horizon, float p_sharpness, float p_light_affect, float p_ao_channel_affect) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("environment_set_ssao")._native_ptr(), 3994732740);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	double p_radius_encoded;
	PtrToArg<double>::encode(p_radius, &p_radius_encoded);
	double p_intensity_encoded;
	PtrToArg<double>::encode(p_intensity, &p_intensity_encoded);
	double p_power_encoded;
	PtrToArg<double>::encode(p_power, &p_power_encoded);
	double p_detail_encoded;
	PtrToArg<double>::encode(p_detail, &p_detail_encoded);
	double p_horizon_encoded;
	PtrToArg<double>::encode(p_horizon, &p_horizon_encoded);
	double p_sharpness_encoded;
	PtrToArg<double>::encode(p_sharpness, &p_sharpness_encoded);
	double p_light_affect_encoded;
	PtrToArg<double>::encode(p_light_affect, &p_light_affect_encoded);
	double p_ao_channel_affect_encoded;
	PtrToArg<double>::encode(p_ao_channel_affect, &p_ao_channel_affect_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_env, &p_enable_encoded, &p_radius_encoded, &p_intensity_encoded, &p_power_encoded, &p_detail_encoded, &p_horizon_encoded, &p_sharpness_encoded, &p_light_affect_encoded, &p_ao_channel_affect_encoded);
}

void RenderingServer::environment_set_fog(const RID &p_env, bool p_enable, const Color &p_light_color, float p_light_energy, float p_sun_scatter, float p_density, float p_height, float p_height_density, float p_aerial_perspective, float p_sky_affect, RenderingServer::EnvironmentFogMode p_fog_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("environment_set_fog")._native_ptr(), 105051629);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	double p_light_energy_encoded;
	PtrToArg<double>::encode(p_light_energy, &p_light_energy_encoded);
	double p_sun_scatter_encoded;
	PtrToArg<double>::encode(p_sun_scatter, &p_sun_scatter_encoded);
	double p_density_encoded;
	PtrToArg<double>::encode(p_density, &p_density_encoded);
	double p_height_encoded;
	PtrToArg<double>::encode(p_height, &p_height_encoded);
	double p_height_density_encoded;
	PtrToArg<double>::encode(p_height_density, &p_height_density_encoded);
	double p_aerial_perspective_encoded;
	PtrToArg<double>::encode(p_aerial_perspective, &p_aerial_perspective_encoded);
	double p_sky_affect_encoded;
	PtrToArg<double>::encode(p_sky_affect, &p_sky_affect_encoded);
	int64_t p_fog_mode_encoded;
	PtrToArg<int64_t>::encode(p_fog_mode, &p_fog_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_env, &p_enable_encoded, &p_light_color, &p_light_energy_encoded, &p_sun_scatter_encoded, &p_density_encoded, &p_height_encoded, &p_height_density_encoded, &p_aerial_perspective_encoded, &p_sky_affect_encoded, &p_fog_mode_encoded);
}

void RenderingServer::environment_set_fog_depth(const RID &p_env, float p_curve, float p_begin, float p_end) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("environment_set_fog_depth")._native_ptr(), 157498339);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_curve_encoded;
	PtrToArg<double>::encode(p_curve, &p_curve_encoded);
	double p_begin_encoded;
	PtrToArg<double>::encode(p_begin, &p_begin_encoded);
	double p_end_encoded;
	PtrToArg<double>::encode(p_end, &p_end_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_env, &p_curve_encoded, &p_begin_encoded, &p_end_encoded);
}

void RenderingServer::environment_set_sdfgi(const RID &p_env, bool p_enable, int32_t p_cascades, float p_min_cell_size, RenderingServer::EnvironmentSDFGIYScale p_y_scale, bool p_use_occlusion, float p_bounce_feedback, bool p_read_sky, float p_energy, float p_normal_bias, float p_probe_bias) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("environment_set_sdfgi")._native_ptr(), 3519144388);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	int64_t p_cascades_encoded;
	PtrToArg<int64_t>::encode(p_cascades, &p_cascades_encoded);
	double p_min_cell_size_encoded;
	PtrToArg<double>::encode(p_min_cell_size, &p_min_cell_size_encoded);
	int64_t p_y_scale_encoded;
	PtrToArg<int64_t>::encode(p_y_scale, &p_y_scale_encoded);
	int8_t p_use_occlusion_encoded;
	PtrToArg<bool>::encode(p_use_occlusion, &p_use_occlusion_encoded);
	double p_bounce_feedback_encoded;
	PtrToArg<double>::encode(p_bounce_feedback, &p_bounce_feedback_encoded);
	int8_t p_read_sky_encoded;
	PtrToArg<bool>::encode(p_read_sky, &p_read_sky_encoded);
	double p_energy_encoded;
	PtrToArg<double>::encode(p_energy, &p_energy_encoded);
	double p_normal_bias_encoded;
	PtrToArg<double>::encode(p_normal_bias, &p_normal_bias_encoded);
	double p_probe_bias_encoded;
	PtrToArg<double>::encode(p_probe_bias, &p_probe_bias_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_env, &p_enable_encoded, &p_cascades_encoded, &p_min_cell_size_encoded, &p_y_scale_encoded, &p_use_occlusion_encoded, &p_bounce_feedback_encoded, &p_read_sky_encoded, &p_energy_encoded, &p_normal_bias_encoded, &p_probe_bias_encoded);
}

void RenderingServer::environment_set_volumetric_fog(const RID &p_env, bool p_enable, float p_density, const Color &p_albedo, const Color &p_emission, float p_emission_energy, float p_anisotropy, float p_length, float p_detail_spread, float p_gi_inject, bool p_temporal_reprojection, float p_temporal_reprojection_amount, float p_ambient_inject, float p_sky_affect) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("environment_set_volumetric_fog")._native_ptr(), 1553633833);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	double p_density_encoded;
	PtrToArg<double>::encode(p_density, &p_density_encoded);
	double p_emission_energy_encoded;
	PtrToArg<double>::encode(p_emission_energy, &p_emission_energy_encoded);
	double p_anisotropy_encoded;
	PtrToArg<double>::encode(p_anisotropy, &p_anisotropy_encoded);
	double p_length_encoded;
	PtrToArg<double>::encode(p_length, &p_length_encoded);
	double p_detail_spread_encoded;
	PtrToArg<double>::encode(p_detail_spread, &p_detail_spread_encoded);
	double p_gi_inject_encoded;
	PtrToArg<double>::encode(p_gi_inject, &p_gi_inject_encoded);
	int8_t p_temporal_reprojection_encoded;
	PtrToArg<bool>::encode(p_temporal_reprojection, &p_temporal_reprojection_encoded);
	double p_temporal_reprojection_amount_encoded;
	PtrToArg<double>::encode(p_temporal_reprojection_amount, &p_temporal_reprojection_amount_encoded);
	double p_ambient_inject_encoded;
	PtrToArg<double>::encode(p_ambient_inject, &p_ambient_inject_encoded);
	double p_sky_affect_encoded;
	PtrToArg<double>::encode(p_sky_affect, &p_sky_affect_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_env, &p_enable_encoded, &p_density_encoded, &p_albedo, &p_emission, &p_emission_energy_encoded, &p_anisotropy_encoded, &p_length_encoded, &p_detail_spread_encoded, &p_gi_inject_encoded, &p_temporal_reprojection_encoded, &p_temporal_reprojection_amount_encoded, &p_ambient_inject_encoded, &p_sky_affect_encoded);
}

void RenderingServer::environment_glow_set_use_bicubic_upscale(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("environment_glow_set_use_bicubic_upscale")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

void RenderingServer::environment_set_ssr_half_size(bool p_half_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("environment_set_ssr_half_size")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_half_size_encoded;
	PtrToArg<bool>::encode(p_half_size, &p_half_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_half_size_encoded);
}

void RenderingServer::environment_set_ssr_roughness_quality(RenderingServer::EnvironmentSSRRoughnessQuality p_quality) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("environment_set_ssr_roughness_quality")._native_ptr(), 1190026788);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_quality_encoded;
	PtrToArg<int64_t>::encode(p_quality, &p_quality_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_quality_encoded);
}

void RenderingServer::environment_set_ssao_quality(RenderingServer::EnvironmentSSAOQuality p_quality, bool p_half_size, float p_adaptive_target, int32_t p_blur_passes, float p_fadeout_from, float p_fadeout_to) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("environment_set_ssao_quality")._native_ptr(), 189753569);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_quality_encoded;
	PtrToArg<int64_t>::encode(p_quality, &p_quality_encoded);
	int8_t p_half_size_encoded;
	PtrToArg<bool>::encode(p_half_size, &p_half_size_encoded);
	double p_adaptive_target_encoded;
	PtrToArg<double>::encode(p_adaptive_target, &p_adaptive_target_encoded);
	int64_t p_blur_passes_encoded;
	PtrToArg<int64_t>::encode(p_blur_passes, &p_blur_passes_encoded);
	double p_fadeout_from_encoded;
	PtrToArg<double>::encode(p_fadeout_from, &p_fadeout_from_encoded);
	double p_fadeout_to_encoded;
	PtrToArg<double>::encode(p_fadeout_to, &p_fadeout_to_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_quality_encoded, &p_half_size_encoded, &p_adaptive_target_encoded, &p_blur_passes_encoded, &p_fadeout_from_encoded, &p_fadeout_to_encoded);
}

void RenderingServer::environment_set_ssil_quality(RenderingServer::EnvironmentSSILQuality p_quality, bool p_half_size, float p_adaptive_target, int32_t p_blur_passes, float p_fadeout_from, float p_fadeout_to) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("environment_set_ssil_quality")._native_ptr(), 1713836683);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_quality_encoded;
	PtrToArg<int64_t>::encode(p_quality, &p_quality_encoded);
	int8_t p_half_size_encoded;
	PtrToArg<bool>::encode(p_half_size, &p_half_size_encoded);
	double p_adaptive_target_encoded;
	PtrToArg<double>::encode(p_adaptive_target, &p_adaptive_target_encoded);
	int64_t p_blur_passes_encoded;
	PtrToArg<int64_t>::encode(p_blur_passes, &p_blur_passes_encoded);
	double p_fadeout_from_encoded;
	PtrToArg<double>::encode(p_fadeout_from, &p_fadeout_from_encoded);
	double p_fadeout_to_encoded;
	PtrToArg<double>::encode(p_fadeout_to, &p_fadeout_to_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_quality_encoded, &p_half_size_encoded, &p_adaptive_target_encoded, &p_blur_passes_encoded, &p_fadeout_from_encoded, &p_fadeout_to_encoded);
}

void RenderingServer::environment_set_sdfgi_ray_count(RenderingServer::EnvironmentSDFGIRayCount p_ray_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("environment_set_sdfgi_ray_count")._native_ptr(), 340137951);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_ray_count_encoded;
	PtrToArg<int64_t>::encode(p_ray_count, &p_ray_count_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_ray_count_encoded);
}

void RenderingServer::environment_set_sdfgi_frames_to_converge(RenderingServer::EnvironmentSDFGIFramesToConverge p_frames) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("environment_set_sdfgi_frames_to_converge")._native_ptr(), 2182444374);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_frames_encoded;
	PtrToArg<int64_t>::encode(p_frames, &p_frames_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_frames_encoded);
}

void RenderingServer::environment_set_sdfgi_frames_to_update_light(RenderingServer::EnvironmentSDFGIFramesToUpdateLight p_frames) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("environment_set_sdfgi_frames_to_update_light")._native_ptr(), 1251144068);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_frames_encoded;
	PtrToArg<int64_t>::encode(p_frames, &p_frames_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_frames_encoded);
}

void RenderingServer::environment_set_volumetric_fog_volume_size(int32_t p_size, int32_t p_depth) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("environment_set_volumetric_fog_volume_size")._native_ptr(), 3937882851);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	int64_t p_depth_encoded;
	PtrToArg<int64_t>::encode(p_depth, &p_depth_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size_encoded, &p_depth_encoded);
}

void RenderingServer::environment_set_volumetric_fog_filter_active(bool p_active) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("environment_set_volumetric_fog_filter_active")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_active_encoded;
	PtrToArg<bool>::encode(p_active, &p_active_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_active_encoded);
}

Ref<Image> RenderingServer::environment_bake_panorama(const RID &p_environment, bool p_bake_irradiance, const Vector2i &p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("environment_bake_panorama")._native_ptr(), 2452908646);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Image>()));
	int8_t p_bake_irradiance_encoded;
	PtrToArg<bool>::encode(p_bake_irradiance, &p_bake_irradiance_encoded);
	return Ref<Image>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Image>(_gde_method_bind, _owner, &p_environment, &p_bake_irradiance_encoded, &p_size));
}

void RenderingServer::screen_space_roughness_limiter_set_active(bool p_enable, float p_amount, float p_limit) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("screen_space_roughness_limiter_set_active")._native_ptr(), 916716790);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	double p_amount_encoded;
	PtrToArg<double>::encode(p_amount, &p_amount_encoded);
	double p_limit_encoded;
	PtrToArg<double>::encode(p_limit, &p_limit_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded, &p_amount_encoded, &p_limit_encoded);
}

void RenderingServer::sub_surface_scattering_set_quality(RenderingServer::SubSurfaceScatteringQuality p_quality) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("sub_surface_scattering_set_quality")._native_ptr(), 64571803);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_quality_encoded;
	PtrToArg<int64_t>::encode(p_quality, &p_quality_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_quality_encoded);
}

void RenderingServer::sub_surface_scattering_set_scale(float p_scale, float p_depth_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("sub_surface_scattering_set_scale")._native_ptr(), 1017552074);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_scale_encoded;
	PtrToArg<double>::encode(p_scale, &p_scale_encoded);
	double p_depth_scale_encoded;
	PtrToArg<double>::encode(p_depth_scale, &p_depth_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scale_encoded, &p_depth_scale_encoded);
}

RID RenderingServer::camera_attributes_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("camera_attributes_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void RenderingServer::camera_attributes_set_dof_blur_quality(RenderingServer::DOFBlurQuality p_quality, bool p_use_jitter) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("camera_attributes_set_dof_blur_quality")._native_ptr(), 2220136795);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_quality_encoded;
	PtrToArg<int64_t>::encode(p_quality, &p_quality_encoded);
	int8_t p_use_jitter_encoded;
	PtrToArg<bool>::encode(p_use_jitter, &p_use_jitter_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_quality_encoded, &p_use_jitter_encoded);
}

void RenderingServer::camera_attributes_set_dof_blur_bokeh_shape(RenderingServer::DOFBokehShape p_shape) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("camera_attributes_set_dof_blur_bokeh_shape")._native_ptr(), 1205058394);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_shape_encoded;
	PtrToArg<int64_t>::encode(p_shape, &p_shape_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_shape_encoded);
}

void RenderingServer::camera_attributes_set_dof_blur(const RID &p_camera_attributes, bool p_far_enable, float p_far_distance, float p_far_transition, bool p_near_enable, float p_near_distance, float p_near_transition, float p_amount) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("camera_attributes_set_dof_blur")._native_ptr(), 316272616);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_far_enable_encoded;
	PtrToArg<bool>::encode(p_far_enable, &p_far_enable_encoded);
	double p_far_distance_encoded;
	PtrToArg<double>::encode(p_far_distance, &p_far_distance_encoded);
	double p_far_transition_encoded;
	PtrToArg<double>::encode(p_far_transition, &p_far_transition_encoded);
	int8_t p_near_enable_encoded;
	PtrToArg<bool>::encode(p_near_enable, &p_near_enable_encoded);
	double p_near_distance_encoded;
	PtrToArg<double>::encode(p_near_distance, &p_near_distance_encoded);
	double p_near_transition_encoded;
	PtrToArg<double>::encode(p_near_transition, &p_near_transition_encoded);
	double p_amount_encoded;
	PtrToArg<double>::encode(p_amount, &p_amount_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_camera_attributes, &p_far_enable_encoded, &p_far_distance_encoded, &p_far_transition_encoded, &p_near_enable_encoded, &p_near_distance_encoded, &p_near_transition_encoded, &p_amount_encoded);
}

void RenderingServer::camera_attributes_set_exposure(const RID &p_camera_attributes, float p_multiplier, float p_normalization) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("camera_attributes_set_exposure")._native_ptr(), 2513314492);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_multiplier_encoded;
	PtrToArg<double>::encode(p_multiplier, &p_multiplier_encoded);
	double p_normalization_encoded;
	PtrToArg<double>::encode(p_normalization, &p_normalization_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_camera_attributes, &p_multiplier_encoded, &p_normalization_encoded);
}

void RenderingServer::camera_attributes_set_auto_exposure(const RID &p_camera_attributes, bool p_enable, float p_min_sensitivity, float p_max_sensitivity, float p_speed, float p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("camera_attributes_set_auto_exposure")._native_ptr(), 4266986332);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	double p_min_sensitivity_encoded;
	PtrToArg<double>::encode(p_min_sensitivity, &p_min_sensitivity_encoded);
	double p_max_sensitivity_encoded;
	PtrToArg<double>::encode(p_max_sensitivity, &p_max_sensitivity_encoded);
	double p_speed_encoded;
	PtrToArg<double>::encode(p_speed, &p_speed_encoded);
	double p_scale_encoded;
	PtrToArg<double>::encode(p_scale, &p_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_camera_attributes, &p_enable_encoded, &p_min_sensitivity_encoded, &p_max_sensitivity_encoded, &p_speed_encoded, &p_scale_encoded);
}

RID RenderingServer::scenario_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("scenario_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void RenderingServer::scenario_set_environment(const RID &p_scenario, const RID &p_environment) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("scenario_set_environment")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scenario, &p_environment);
}

void RenderingServer::scenario_set_fallback_environment(const RID &p_scenario, const RID &p_environment) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("scenario_set_fallback_environment")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scenario, &p_environment);
}

void RenderingServer::scenario_set_camera_attributes(const RID &p_scenario, const RID &p_effects) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("scenario_set_camera_attributes")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scenario, &p_effects);
}

void RenderingServer::scenario_set_compositor(const RID &p_scenario, const RID &p_compositor) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("scenario_set_compositor")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scenario, &p_compositor);
}

RID RenderingServer::instance_create2(const RID &p_base, const RID &p_scenario) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("instance_create2")._native_ptr(), 746547085);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_base, &p_scenario);
}

RID RenderingServer::instance_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("instance_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void RenderingServer::instance_set_base(const RID &p_instance, const RID &p_base) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("instance_set_base")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_instance, &p_base);
}

void RenderingServer::instance_set_scenario(const RID &p_instance, const RID &p_scenario) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("instance_set_scenario")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_instance, &p_scenario);
}

void RenderingServer::instance_set_layer_mask(const RID &p_instance, uint32_t p_mask) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("instance_set_layer_mask")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mask_encoded;
	PtrToArg<int64_t>::encode(p_mask, &p_mask_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_instance, &p_mask_encoded);
}

void RenderingServer::instance_set_pivot_data(const RID &p_instance, float p_sorting_offset, bool p_use_aabb_center) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("instance_set_pivot_data")._native_ptr(), 1280615259);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_sorting_offset_encoded;
	PtrToArg<double>::encode(p_sorting_offset, &p_sorting_offset_encoded);
	int8_t p_use_aabb_center_encoded;
	PtrToArg<bool>::encode(p_use_aabb_center, &p_use_aabb_center_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_instance, &p_sorting_offset_encoded, &p_use_aabb_center_encoded);
}

void RenderingServer::instance_set_transform(const RID &p_instance, const Transform3D &p_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("instance_set_transform")._native_ptr(), 3935195649);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_instance, &p_transform);
}

void RenderingServer::instance_attach_object_instance_id(const RID &p_instance, uint64_t p_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("instance_attach_object_instance_id")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_instance, &p_id_encoded);
}

void RenderingServer::instance_set_blend_shape_weight(const RID &p_instance, int32_t p_shape, float p_weight) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("instance_set_blend_shape_weight")._native_ptr(), 1892459533);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_shape_encoded;
	PtrToArg<int64_t>::encode(p_shape, &p_shape_encoded);
	double p_weight_encoded;
	PtrToArg<double>::encode(p_weight, &p_weight_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_instance, &p_shape_encoded, &p_weight_encoded);
}

void RenderingServer::instance_set_surface_override_material(const RID &p_instance, int32_t p_surface, const RID &p_material) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("instance_set_surface_override_material")._native_ptr(), 2310537182);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_surface_encoded;
	PtrToArg<int64_t>::encode(p_surface, &p_surface_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_instance, &p_surface_encoded, &p_material);
}

void RenderingServer::instance_set_visible(const RID &p_instance, bool p_visible) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("instance_set_visible")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_visible_encoded;
	PtrToArg<bool>::encode(p_visible, &p_visible_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_instance, &p_visible_encoded);
}

void RenderingServer::instance_geometry_set_transparency(const RID &p_instance, float p_transparency) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("instance_geometry_set_transparency")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_transparency_encoded;
	PtrToArg<double>::encode(p_transparency, &p_transparency_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_instance, &p_transparency_encoded);
}

void RenderingServer::instance_teleport(const RID &p_instance) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("instance_teleport")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_instance);
}

void RenderingServer::instance_set_custom_aabb(const RID &p_instance, const AABB &p_aabb) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("instance_set_custom_aabb")._native_ptr(), 3696536120);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_instance, &p_aabb);
}

void RenderingServer::instance_attach_skeleton(const RID &p_instance, const RID &p_skeleton) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("instance_attach_skeleton")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_instance, &p_skeleton);
}

void RenderingServer::instance_set_extra_visibility_margin(const RID &p_instance, float p_margin) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("instance_set_extra_visibility_margin")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_margin_encoded;
	PtrToArg<double>::encode(p_margin, &p_margin_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_instance, &p_margin_encoded);
}

void RenderingServer::instance_set_visibility_parent(const RID &p_instance, const RID &p_parent) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("instance_set_visibility_parent")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_instance, &p_parent);
}

void RenderingServer::instance_set_ignore_culling(const RID &p_instance, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("instance_set_ignore_culling")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_instance, &p_enabled_encoded);
}

void RenderingServer::instance_geometry_set_flag(const RID &p_instance, RenderingServer::InstanceFlags p_flag, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("instance_geometry_set_flag")._native_ptr(), 1014989537);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_flag_encoded;
	PtrToArg<int64_t>::encode(p_flag, &p_flag_encoded);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_instance, &p_flag_encoded, &p_enabled_encoded);
}

void RenderingServer::instance_geometry_set_cast_shadows_setting(const RID &p_instance, RenderingServer::ShadowCastingSetting p_shadow_casting_setting) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("instance_geometry_set_cast_shadows_setting")._native_ptr(), 3768836020);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_shadow_casting_setting_encoded;
	PtrToArg<int64_t>::encode(p_shadow_casting_setting, &p_shadow_casting_setting_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_instance, &p_shadow_casting_setting_encoded);
}

void RenderingServer::instance_geometry_set_material_override(const RID &p_instance, const RID &p_material) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("instance_geometry_set_material_override")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_instance, &p_material);
}

void RenderingServer::instance_geometry_set_material_overlay(const RID &p_instance, const RID &p_material) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("instance_geometry_set_material_overlay")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_instance, &p_material);
}

void RenderingServer::instance_geometry_set_visibility_range(const RID &p_instance, float p_min, float p_max, float p_min_margin, float p_max_margin, RenderingServer::VisibilityRangeFadeMode p_fade_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("instance_geometry_set_visibility_range")._native_ptr(), 4263925858);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_min_encoded;
	PtrToArg<double>::encode(p_min, &p_min_encoded);
	double p_max_encoded;
	PtrToArg<double>::encode(p_max, &p_max_encoded);
	double p_min_margin_encoded;
	PtrToArg<double>::encode(p_min_margin, &p_min_margin_encoded);
	double p_max_margin_encoded;
	PtrToArg<double>::encode(p_max_margin, &p_max_margin_encoded);
	int64_t p_fade_mode_encoded;
	PtrToArg<int64_t>::encode(p_fade_mode, &p_fade_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_instance, &p_min_encoded, &p_max_encoded, &p_min_margin_encoded, &p_max_margin_encoded, &p_fade_mode_encoded);
}

void RenderingServer::instance_geometry_set_lightmap(const RID &p_instance, const RID &p_lightmap, const Rect2 &p_lightmap_uv_scale, int32_t p_lightmap_slice) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("instance_geometry_set_lightmap")._native_ptr(), 536974962);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_lightmap_slice_encoded;
	PtrToArg<int64_t>::encode(p_lightmap_slice, &p_lightmap_slice_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_instance, &p_lightmap, &p_lightmap_uv_scale, &p_lightmap_slice_encoded);
}

void RenderingServer::instance_geometry_set_lod_bias(const RID &p_instance, float p_lod_bias) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("instance_geometry_set_lod_bias")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_lod_bias_encoded;
	PtrToArg<double>::encode(p_lod_bias, &p_lod_bias_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_instance, &p_lod_bias_encoded);
}

void RenderingServer::instance_geometry_set_shader_parameter(const RID &p_instance, const StringName &p_parameter, const Variant &p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("instance_geometry_set_shader_parameter")._native_ptr(), 3477296213);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_instance, &p_parameter, &p_value);
}

Variant RenderingServer::instance_geometry_get_shader_parameter(const RID &p_instance, const StringName &p_parameter) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("instance_geometry_get_shader_parameter")._native_ptr(), 2621281810);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_instance, &p_parameter);
}

Variant RenderingServer::instance_geometry_get_shader_parameter_default_value(const RID &p_instance, const StringName &p_parameter) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("instance_geometry_get_shader_parameter_default_value")._native_ptr(), 2621281810);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_instance, &p_parameter);
}

TypedArray<Dictionary> RenderingServer::instance_geometry_get_shader_parameter_list(const RID &p_instance) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("instance_geometry_get_shader_parameter_list")._native_ptr(), 2684255073);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Dictionary>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Dictionary>>(_gde_method_bind, _owner, &p_instance);
}

PackedInt64Array RenderingServer::instances_cull_aabb(const AABB &p_aabb, const RID &p_scenario) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("instances_cull_aabb")._native_ptr(), 2570105777);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt64Array()));
	return ::godot::internal::_call_native_mb_ret<PackedInt64Array>(_gde_method_bind, _owner, &p_aabb, &p_scenario);
}

PackedInt64Array RenderingServer::instances_cull_ray(const Vector3 &p_from, const Vector3 &p_to, const RID &p_scenario) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("instances_cull_ray")._native_ptr(), 2208759584);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt64Array()));
	return ::godot::internal::_call_native_mb_ret<PackedInt64Array>(_gde_method_bind, _owner, &p_from, &p_to, &p_scenario);
}

PackedInt64Array RenderingServer::instances_cull_convex(const TypedArray<Plane> &p_convex, const RID &p_scenario) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("instances_cull_convex")._native_ptr(), 2488539944);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt64Array()));
	return ::godot::internal::_call_native_mb_ret<PackedInt64Array>(_gde_method_bind, _owner, &p_convex, &p_scenario);
}

TypedArray<Ref<Image>> RenderingServer::bake_render_uv2(const RID &p_base, const TypedArray<RID> &p_material_overrides, const Vector2i &p_image_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("bake_render_uv2")._native_ptr(), 1904608558);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Ref<Image>>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Ref<Image>>>(_gde_method_bind, _owner, &p_base, &p_material_overrides, &p_image_size);
}

RID RenderingServer::canvas_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void RenderingServer::canvas_set_item_mirroring(const RID &p_canvas, const RID &p_item, const Vector2 &p_mirroring) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_set_item_mirroring")._native_ptr(), 2343975398);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_canvas, &p_item, &p_mirroring);
}

void RenderingServer::canvas_set_item_repeat(const RID &p_item, const Vector2 &p_repeat_size, int32_t p_repeat_times) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_set_item_repeat")._native_ptr(), 1739512717);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_repeat_times_encoded;
	PtrToArg<int64_t>::encode(p_repeat_times, &p_repeat_times_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_repeat_size, &p_repeat_times_encoded);
}

void RenderingServer::canvas_set_modulate(const RID &p_canvas, const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_set_modulate")._native_ptr(), 2948539648);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_canvas, &p_color);
}

void RenderingServer::canvas_set_disable_scale(bool p_disable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_set_disable_scale")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_disable_encoded;
	PtrToArg<bool>::encode(p_disable, &p_disable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_disable_encoded);
}

RID RenderingServer::canvas_texture_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_texture_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void RenderingServer::canvas_texture_set_channel(const RID &p_canvas_texture, RenderingServer::CanvasTextureChannel p_channel, const RID &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_texture_set_channel")._native_ptr(), 3822119138);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_channel_encoded;
	PtrToArg<int64_t>::encode(p_channel, &p_channel_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_canvas_texture, &p_channel_encoded, &p_texture);
}

void RenderingServer::canvas_texture_set_shading_parameters(const RID &p_canvas_texture, const Color &p_base_color, float p_shininess) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_texture_set_shading_parameters")._native_ptr(), 2124967469);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_shininess_encoded;
	PtrToArg<double>::encode(p_shininess, &p_shininess_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_canvas_texture, &p_base_color, &p_shininess_encoded);
}

void RenderingServer::canvas_texture_set_texture_filter(const RID &p_canvas_texture, RenderingServer::CanvasItemTextureFilter p_filter) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_texture_set_texture_filter")._native_ptr(), 1155129294);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_filter_encoded;
	PtrToArg<int64_t>::encode(p_filter, &p_filter_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_canvas_texture, &p_filter_encoded);
}

void RenderingServer::canvas_texture_set_texture_repeat(const RID &p_canvas_texture, RenderingServer::CanvasItemTextureRepeat p_repeat) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_texture_set_texture_repeat")._native_ptr(), 1652956681);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_repeat_encoded;
	PtrToArg<int64_t>::encode(p_repeat, &p_repeat_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_canvas_texture, &p_repeat_encoded);
}

RID RenderingServer::canvas_item_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void RenderingServer::canvas_item_set_parent(const RID &p_item, const RID &p_parent) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_set_parent")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_parent);
}

void RenderingServer::canvas_item_set_default_texture_filter(const RID &p_item, RenderingServer::CanvasItemTextureFilter p_filter) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_set_default_texture_filter")._native_ptr(), 1155129294);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_filter_encoded;
	PtrToArg<int64_t>::encode(p_filter, &p_filter_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_filter_encoded);
}

void RenderingServer::canvas_item_set_default_texture_repeat(const RID &p_item, RenderingServer::CanvasItemTextureRepeat p_repeat) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_set_default_texture_repeat")._native_ptr(), 1652956681);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_repeat_encoded;
	PtrToArg<int64_t>::encode(p_repeat, &p_repeat_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_repeat_encoded);
}

void RenderingServer::canvas_item_set_visible(const RID &p_item, bool p_visible) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_set_visible")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_visible_encoded;
	PtrToArg<bool>::encode(p_visible, &p_visible_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_visible_encoded);
}

void RenderingServer::canvas_item_set_light_mask(const RID &p_item, int32_t p_mask) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_set_light_mask")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mask_encoded;
	PtrToArg<int64_t>::encode(p_mask, &p_mask_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_mask_encoded);
}

void RenderingServer::canvas_item_set_visibility_layer(const RID &p_item, uint32_t p_visibility_layer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_set_visibility_layer")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_visibility_layer_encoded;
	PtrToArg<int64_t>::encode(p_visibility_layer, &p_visibility_layer_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_visibility_layer_encoded);
}

void RenderingServer::canvas_item_set_transform(const RID &p_item, const Transform2D &p_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_set_transform")._native_ptr(), 1246044741);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_transform);
}

void RenderingServer::canvas_item_set_clip(const RID &p_item, bool p_clip) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_set_clip")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_clip_encoded;
	PtrToArg<bool>::encode(p_clip, &p_clip_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_clip_encoded);
}

void RenderingServer::canvas_item_set_distance_field_mode(const RID &p_item, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_set_distance_field_mode")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_enabled_encoded);
}

void RenderingServer::canvas_item_set_custom_rect(const RID &p_item, bool p_use_custom_rect, const Rect2 &p_rect) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_set_custom_rect")._native_ptr(), 1333997032);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_use_custom_rect_encoded;
	PtrToArg<bool>::encode(p_use_custom_rect, &p_use_custom_rect_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_use_custom_rect_encoded, &p_rect);
}

void RenderingServer::canvas_item_set_modulate(const RID &p_item, const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_set_modulate")._native_ptr(), 2948539648);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_color);
}

void RenderingServer::canvas_item_set_self_modulate(const RID &p_item, const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_set_self_modulate")._native_ptr(), 2948539648);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_color);
}

void RenderingServer::canvas_item_set_draw_behind_parent(const RID &p_item, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_set_draw_behind_parent")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_enabled_encoded);
}

void RenderingServer::canvas_item_set_interpolated(const RID &p_item, bool p_interpolated) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_set_interpolated")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_interpolated_encoded;
	PtrToArg<bool>::encode(p_interpolated, &p_interpolated_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_interpolated_encoded);
}

void RenderingServer::canvas_item_reset_physics_interpolation(const RID &p_item) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_reset_physics_interpolation")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item);
}

void RenderingServer::canvas_item_transform_physics_interpolation(const RID &p_item, const Transform2D &p_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_transform_physics_interpolation")._native_ptr(), 1246044741);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_transform);
}

void RenderingServer::canvas_item_add_line(const RID &p_item, const Vector2 &p_from, const Vector2 &p_to, const Color &p_color, float p_width, bool p_antialiased) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_add_line")._native_ptr(), 1819681853);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_width_encoded;
	PtrToArg<double>::encode(p_width, &p_width_encoded);
	int8_t p_antialiased_encoded;
	PtrToArg<bool>::encode(p_antialiased, &p_antialiased_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_from, &p_to, &p_color, &p_width_encoded, &p_antialiased_encoded);
}

void RenderingServer::canvas_item_add_polyline(const RID &p_item, const PackedVector2Array &p_points, const PackedColorArray &p_colors, float p_width, bool p_antialiased) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_add_polyline")._native_ptr(), 3098767073);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_width_encoded;
	PtrToArg<double>::encode(p_width, &p_width_encoded);
	int8_t p_antialiased_encoded;
	PtrToArg<bool>::encode(p_antialiased, &p_antialiased_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_points, &p_colors, &p_width_encoded, &p_antialiased_encoded);
}

void RenderingServer::canvas_item_add_multiline(const RID &p_item, const PackedVector2Array &p_points, const PackedColorArray &p_colors, float p_width, bool p_antialiased) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_add_multiline")._native_ptr(), 3098767073);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_width_encoded;
	PtrToArg<double>::encode(p_width, &p_width_encoded);
	int8_t p_antialiased_encoded;
	PtrToArg<bool>::encode(p_antialiased, &p_antialiased_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_points, &p_colors, &p_width_encoded, &p_antialiased_encoded);
}

void RenderingServer::canvas_item_add_rect(const RID &p_item, const Rect2 &p_rect, const Color &p_color, bool p_antialiased) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_add_rect")._native_ptr(), 3523446176);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_antialiased_encoded;
	PtrToArg<bool>::encode(p_antialiased, &p_antialiased_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_rect, &p_color, &p_antialiased_encoded);
}

void RenderingServer::canvas_item_add_circle(const RID &p_item, const Vector2 &p_pos, float p_radius, const Color &p_color, bool p_antialiased) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_add_circle")._native_ptr(), 333077949);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_radius_encoded;
	PtrToArg<double>::encode(p_radius, &p_radius_encoded);
	int8_t p_antialiased_encoded;
	PtrToArg<bool>::encode(p_antialiased, &p_antialiased_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_pos, &p_radius_encoded, &p_color, &p_antialiased_encoded);
}

void RenderingServer::canvas_item_add_ellipse(const RID &p_item, const Vector2 &p_pos, float p_major, float p_minor, const Color &p_color, bool p_antialiased) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_add_ellipse")._native_ptr(), 4188642757);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_major_encoded;
	PtrToArg<double>::encode(p_major, &p_major_encoded);
	double p_minor_encoded;
	PtrToArg<double>::encode(p_minor, &p_minor_encoded);
	int8_t p_antialiased_encoded;
	PtrToArg<bool>::encode(p_antialiased, &p_antialiased_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_pos, &p_major_encoded, &p_minor_encoded, &p_color, &p_antialiased_encoded);
}

void RenderingServer::canvas_item_add_texture_rect(const RID &p_item, const Rect2 &p_rect, const RID &p_texture, bool p_tile, const Color &p_modulate, bool p_transpose) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_add_texture_rect")._native_ptr(), 324864032);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_tile_encoded;
	PtrToArg<bool>::encode(p_tile, &p_tile_encoded);
	int8_t p_transpose_encoded;
	PtrToArg<bool>::encode(p_transpose, &p_transpose_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_rect, &p_texture, &p_tile_encoded, &p_modulate, &p_transpose_encoded);
}

void RenderingServer::canvas_item_add_msdf_texture_rect_region(const RID &p_item, const Rect2 &p_rect, const RID &p_texture, const Rect2 &p_src_rect, const Color &p_modulate, int32_t p_outline_size, float p_px_range, float p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_add_msdf_texture_rect_region")._native_ptr(), 97408773);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_outline_size_encoded;
	PtrToArg<int64_t>::encode(p_outline_size, &p_outline_size_encoded);
	double p_px_range_encoded;
	PtrToArg<double>::encode(p_px_range, &p_px_range_encoded);
	double p_scale_encoded;
	PtrToArg<double>::encode(p_scale, &p_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_rect, &p_texture, &p_src_rect, &p_modulate, &p_outline_size_encoded, &p_px_range_encoded, &p_scale_encoded);
}

void RenderingServer::canvas_item_add_lcd_texture_rect_region(const RID &p_item, const Rect2 &p_rect, const RID &p_texture, const Rect2 &p_src_rect, const Color &p_modulate) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_add_lcd_texture_rect_region")._native_ptr(), 359793297);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_rect, &p_texture, &p_src_rect, &p_modulate);
}

void RenderingServer::canvas_item_add_texture_rect_region(const RID &p_item, const Rect2 &p_rect, const RID &p_texture, const Rect2 &p_src_rect, const Color &p_modulate, bool p_transpose, bool p_clip_uv) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_add_texture_rect_region")._native_ptr(), 485157892);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_transpose_encoded;
	PtrToArg<bool>::encode(p_transpose, &p_transpose_encoded);
	int8_t p_clip_uv_encoded;
	PtrToArg<bool>::encode(p_clip_uv, &p_clip_uv_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_rect, &p_texture, &p_src_rect, &p_modulate, &p_transpose_encoded, &p_clip_uv_encoded);
}

void RenderingServer::canvas_item_add_nine_patch(const RID &p_item, const Rect2 &p_rect, const Rect2 &p_source, const RID &p_texture, const Vector2 &p_topleft, const Vector2 &p_bottomright, RenderingServer::NinePatchAxisMode p_x_axis_mode, RenderingServer::NinePatchAxisMode p_y_axis_mode, bool p_draw_center, const Color &p_modulate) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_add_nine_patch")._native_ptr(), 389957886);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_x_axis_mode_encoded;
	PtrToArg<int64_t>::encode(p_x_axis_mode, &p_x_axis_mode_encoded);
	int64_t p_y_axis_mode_encoded;
	PtrToArg<int64_t>::encode(p_y_axis_mode, &p_y_axis_mode_encoded);
	int8_t p_draw_center_encoded;
	PtrToArg<bool>::encode(p_draw_center, &p_draw_center_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_rect, &p_source, &p_texture, &p_topleft, &p_bottomright, &p_x_axis_mode_encoded, &p_y_axis_mode_encoded, &p_draw_center_encoded, &p_modulate);
}

void RenderingServer::canvas_item_add_primitive(const RID &p_item, const PackedVector2Array &p_points, const PackedColorArray &p_colors, const PackedVector2Array &p_uvs, const RID &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_add_primitive")._native_ptr(), 3731601077);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_points, &p_colors, &p_uvs, &p_texture);
}

void RenderingServer::canvas_item_add_polygon(const RID &p_item, const PackedVector2Array &p_points, const PackedColorArray &p_colors, const PackedVector2Array &p_uvs, const RID &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_add_polygon")._native_ptr(), 3580000528);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_points, &p_colors, &p_uvs, &p_texture);
}

void RenderingServer::canvas_item_add_triangle_array(const RID &p_item, const PackedInt32Array &p_indices, const PackedVector2Array &p_points, const PackedColorArray &p_colors, const PackedVector2Array &p_uvs, const PackedInt32Array &p_bones, const PackedFloat32Array &p_weights, const RID &p_texture, int32_t p_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_add_triangle_array")._native_ptr(), 660261329);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_count_encoded;
	PtrToArg<int64_t>::encode(p_count, &p_count_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_indices, &p_points, &p_colors, &p_uvs, &p_bones, &p_weights, &p_texture, &p_count_encoded);
}

void RenderingServer::canvas_item_add_mesh(const RID &p_item, const RID &p_mesh, const Transform2D &p_transform, const Color &p_modulate, const RID &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_add_mesh")._native_ptr(), 316450961);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_mesh, &p_transform, &p_modulate, &p_texture);
}

void RenderingServer::canvas_item_add_multimesh(const RID &p_item, const RID &p_mesh, const RID &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_add_multimesh")._native_ptr(), 2131855138);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_mesh, &p_texture);
}

void RenderingServer::canvas_item_add_particles(const RID &p_item, const RID &p_particles, const RID &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_add_particles")._native_ptr(), 2575754278);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_particles, &p_texture);
}

void RenderingServer::canvas_item_add_set_transform(const RID &p_item, const Transform2D &p_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_add_set_transform")._native_ptr(), 1246044741);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_transform);
}

void RenderingServer::canvas_item_add_clip_ignore(const RID &p_item, bool p_ignore) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_add_clip_ignore")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_ignore_encoded;
	PtrToArg<bool>::encode(p_ignore, &p_ignore_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_ignore_encoded);
}

void RenderingServer::canvas_item_add_animation_slice(const RID &p_item, double p_animation_length, double p_slice_begin, double p_slice_end, double p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_add_animation_slice")._native_ptr(), 2646834499);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_animation_length_encoded;
	PtrToArg<double>::encode(p_animation_length, &p_animation_length_encoded);
	double p_slice_begin_encoded;
	PtrToArg<double>::encode(p_slice_begin, &p_slice_begin_encoded);
	double p_slice_end_encoded;
	PtrToArg<double>::encode(p_slice_end, &p_slice_end_encoded);
	double p_offset_encoded;
	PtrToArg<double>::encode(p_offset, &p_offset_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_animation_length_encoded, &p_slice_begin_encoded, &p_slice_end_encoded, &p_offset_encoded);
}

void RenderingServer::canvas_item_set_sort_children_by_y(const RID &p_item, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_set_sort_children_by_y")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_enabled_encoded);
}

void RenderingServer::canvas_item_set_z_index(const RID &p_item, int32_t p_z_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_set_z_index")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_z_index_encoded;
	PtrToArg<int64_t>::encode(p_z_index, &p_z_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_z_index_encoded);
}

void RenderingServer::canvas_item_set_z_as_relative_to_parent(const RID &p_item, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_set_z_as_relative_to_parent")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_enabled_encoded);
}

void RenderingServer::canvas_item_set_copy_to_backbuffer(const RID &p_item, bool p_enabled, const Rect2 &p_rect) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_set_copy_to_backbuffer")._native_ptr(), 2429202503);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_enabled_encoded, &p_rect);
}

void RenderingServer::canvas_item_attach_skeleton(const RID &p_item, const RID &p_skeleton) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_attach_skeleton")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_skeleton);
}

void RenderingServer::canvas_item_clear(const RID &p_item) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_clear")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item);
}

void RenderingServer::canvas_item_set_draw_index(const RID &p_item, int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_set_draw_index")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_index_encoded);
}

void RenderingServer::canvas_item_set_material(const RID &p_item, const RID &p_material) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_set_material")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_material);
}

void RenderingServer::canvas_item_set_use_parent_material(const RID &p_item, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_set_use_parent_material")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_enabled_encoded);
}

void RenderingServer::canvas_item_set_instance_shader_parameter(const RID &p_instance, const StringName &p_parameter, const Variant &p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_set_instance_shader_parameter")._native_ptr(), 3477296213);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_instance, &p_parameter, &p_value);
}

Variant RenderingServer::canvas_item_get_instance_shader_parameter(const RID &p_instance, const StringName &p_parameter) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_get_instance_shader_parameter")._native_ptr(), 2621281810);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_instance, &p_parameter);
}

Variant RenderingServer::canvas_item_get_instance_shader_parameter_default_value(const RID &p_instance, const StringName &p_parameter) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_get_instance_shader_parameter_default_value")._native_ptr(), 2621281810);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_instance, &p_parameter);
}

TypedArray<Dictionary> RenderingServer::canvas_item_get_instance_shader_parameter_list(const RID &p_instance) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_get_instance_shader_parameter_list")._native_ptr(), 2684255073);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Dictionary>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Dictionary>>(_gde_method_bind, _owner, &p_instance);
}

void RenderingServer::canvas_item_set_visibility_notifier(const RID &p_item, bool p_enable, const Rect2 &p_area, const Callable &p_enter_callable, const Callable &p_exit_callable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_set_visibility_notifier")._native_ptr(), 3568945579);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_enable_encoded, &p_area, &p_enter_callable, &p_exit_callable);
}

void RenderingServer::canvas_item_set_canvas_group_mode(const RID &p_item, RenderingServer::CanvasGroupMode p_mode, float p_clear_margin, bool p_fit_empty, float p_fit_margin, bool p_blur_mipmaps) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_item_set_canvas_group_mode")._native_ptr(), 3973586316);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	double p_clear_margin_encoded;
	PtrToArg<double>::encode(p_clear_margin, &p_clear_margin_encoded);
	int8_t p_fit_empty_encoded;
	PtrToArg<bool>::encode(p_fit_empty, &p_fit_empty_encoded);
	double p_fit_margin_encoded;
	PtrToArg<double>::encode(p_fit_margin, &p_fit_margin_encoded);
	int8_t p_blur_mipmaps_encoded;
	PtrToArg<bool>::encode(p_blur_mipmaps, &p_blur_mipmaps_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_item, &p_mode_encoded, &p_clear_margin_encoded, &p_fit_empty_encoded, &p_fit_margin_encoded, &p_blur_mipmaps_encoded);
}

Rect2 RenderingServer::debug_canvas_item_get_rect(const RID &p_item) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("debug_canvas_item_get_rect")._native_ptr(), 624227424);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Rect2()));
	return ::godot::internal::_call_native_mb_ret<Rect2>(_gde_method_bind, _owner, &p_item);
}

RID RenderingServer::canvas_light_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_light_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void RenderingServer::canvas_light_attach_to_canvas(const RID &p_light, const RID &p_canvas) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_light_attach_to_canvas")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light, &p_canvas);
}

void RenderingServer::canvas_light_set_enabled(const RID &p_light, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_light_set_enabled")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light, &p_enabled_encoded);
}

void RenderingServer::canvas_light_set_texture_scale(const RID &p_light, float p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_light_set_texture_scale")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_scale_encoded;
	PtrToArg<double>::encode(p_scale, &p_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light, &p_scale_encoded);
}

void RenderingServer::canvas_light_set_transform(const RID &p_light, const Transform2D &p_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_light_set_transform")._native_ptr(), 1246044741);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light, &p_transform);
}

void RenderingServer::canvas_light_set_texture(const RID &p_light, const RID &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_light_set_texture")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light, &p_texture);
}

void RenderingServer::canvas_light_set_texture_offset(const RID &p_light, const Vector2 &p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_light_set_texture_offset")._native_ptr(), 3201125042);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light, &p_offset);
}

void RenderingServer::canvas_light_set_color(const RID &p_light, const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_light_set_color")._native_ptr(), 2948539648);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light, &p_color);
}

void RenderingServer::canvas_light_set_height(const RID &p_light, float p_height) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_light_set_height")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_height_encoded;
	PtrToArg<double>::encode(p_height, &p_height_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light, &p_height_encoded);
}

void RenderingServer::canvas_light_set_energy(const RID &p_light, float p_energy) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_light_set_energy")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_energy_encoded;
	PtrToArg<double>::encode(p_energy, &p_energy_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light, &p_energy_encoded);
}

void RenderingServer::canvas_light_set_z_range(const RID &p_light, int32_t p_min_z, int32_t p_max_z) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_light_set_z_range")._native_ptr(), 4288446313);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_min_z_encoded;
	PtrToArg<int64_t>::encode(p_min_z, &p_min_z_encoded);
	int64_t p_max_z_encoded;
	PtrToArg<int64_t>::encode(p_max_z, &p_max_z_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light, &p_min_z_encoded, &p_max_z_encoded);
}

void RenderingServer::canvas_light_set_layer_range(const RID &p_light, int32_t p_min_layer, int32_t p_max_layer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_light_set_layer_range")._native_ptr(), 4288446313);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_min_layer_encoded;
	PtrToArg<int64_t>::encode(p_min_layer, &p_min_layer_encoded);
	int64_t p_max_layer_encoded;
	PtrToArg<int64_t>::encode(p_max_layer, &p_max_layer_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light, &p_min_layer_encoded, &p_max_layer_encoded);
}

void RenderingServer::canvas_light_set_item_cull_mask(const RID &p_light, int32_t p_mask) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_light_set_item_cull_mask")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mask_encoded;
	PtrToArg<int64_t>::encode(p_mask, &p_mask_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light, &p_mask_encoded);
}

void RenderingServer::canvas_light_set_item_shadow_cull_mask(const RID &p_light, int32_t p_mask) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_light_set_item_shadow_cull_mask")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mask_encoded;
	PtrToArg<int64_t>::encode(p_mask, &p_mask_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light, &p_mask_encoded);
}

void RenderingServer::canvas_light_set_mode(const RID &p_light, RenderingServer::CanvasLightMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_light_set_mode")._native_ptr(), 2957564891);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light, &p_mode_encoded);
}

void RenderingServer::canvas_light_set_shadow_enabled(const RID &p_light, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_light_set_shadow_enabled")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light, &p_enabled_encoded);
}

void RenderingServer::canvas_light_set_shadow_filter(const RID &p_light, RenderingServer::CanvasLightShadowFilter p_filter) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_light_set_shadow_filter")._native_ptr(), 393119659);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_filter_encoded;
	PtrToArg<int64_t>::encode(p_filter, &p_filter_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light, &p_filter_encoded);
}

void RenderingServer::canvas_light_set_shadow_color(const RID &p_light, const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_light_set_shadow_color")._native_ptr(), 2948539648);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light, &p_color);
}

void RenderingServer::canvas_light_set_shadow_smooth(const RID &p_light, float p_smooth) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_light_set_shadow_smooth")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_smooth_encoded;
	PtrToArg<double>::encode(p_smooth, &p_smooth_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light, &p_smooth_encoded);
}

void RenderingServer::canvas_light_set_blend_mode(const RID &p_light, RenderingServer::CanvasLightBlendMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_light_set_blend_mode")._native_ptr(), 804895945);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light, &p_mode_encoded);
}

void RenderingServer::canvas_light_set_interpolated(const RID &p_light, bool p_interpolated) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_light_set_interpolated")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_interpolated_encoded;
	PtrToArg<bool>::encode(p_interpolated, &p_interpolated_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light, &p_interpolated_encoded);
}

void RenderingServer::canvas_light_reset_physics_interpolation(const RID &p_light) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_light_reset_physics_interpolation")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light);
}

void RenderingServer::canvas_light_transform_physics_interpolation(const RID &p_light, const Transform2D &p_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_light_transform_physics_interpolation")._native_ptr(), 1246044741);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light, &p_transform);
}

RID RenderingServer::canvas_light_occluder_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_light_occluder_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void RenderingServer::canvas_light_occluder_attach_to_canvas(const RID &p_occluder, const RID &p_canvas) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_light_occluder_attach_to_canvas")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_occluder, &p_canvas);
}

void RenderingServer::canvas_light_occluder_set_enabled(const RID &p_occluder, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_light_occluder_set_enabled")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_occluder, &p_enabled_encoded);
}

void RenderingServer::canvas_light_occluder_set_polygon(const RID &p_occluder, const RID &p_polygon) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_light_occluder_set_polygon")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_occluder, &p_polygon);
}

void RenderingServer::canvas_light_occluder_set_as_sdf_collision(const RID &p_occluder, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_light_occluder_set_as_sdf_collision")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_occluder, &p_enable_encoded);
}

void RenderingServer::canvas_light_occluder_set_transform(const RID &p_occluder, const Transform2D &p_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_light_occluder_set_transform")._native_ptr(), 1246044741);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_occluder, &p_transform);
}

void RenderingServer::canvas_light_occluder_set_light_mask(const RID &p_occluder, int32_t p_mask) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_light_occluder_set_light_mask")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mask_encoded;
	PtrToArg<int64_t>::encode(p_mask, &p_mask_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_occluder, &p_mask_encoded);
}

void RenderingServer::canvas_light_occluder_set_interpolated(const RID &p_occluder, bool p_interpolated) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_light_occluder_set_interpolated")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_interpolated_encoded;
	PtrToArg<bool>::encode(p_interpolated, &p_interpolated_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_occluder, &p_interpolated_encoded);
}

void RenderingServer::canvas_light_occluder_reset_physics_interpolation(const RID &p_occluder) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_light_occluder_reset_physics_interpolation")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_occluder);
}

void RenderingServer::canvas_light_occluder_transform_physics_interpolation(const RID &p_occluder, const Transform2D &p_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_light_occluder_transform_physics_interpolation")._native_ptr(), 1246044741);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_occluder, &p_transform);
}

RID RenderingServer::canvas_occluder_polygon_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_occluder_polygon_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void RenderingServer::canvas_occluder_polygon_set_shape(const RID &p_occluder_polygon, const PackedVector2Array &p_shape, bool p_closed) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_occluder_polygon_set_shape")._native_ptr(), 2103882027);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_closed_encoded;
	PtrToArg<bool>::encode(p_closed, &p_closed_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_occluder_polygon, &p_shape, &p_closed_encoded);
}

void RenderingServer::canvas_occluder_polygon_set_cull_mode(const RID &p_occluder_polygon, RenderingServer::CanvasOccluderPolygonCullMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_occluder_polygon_set_cull_mode")._native_ptr(), 1839404663);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_occluder_polygon, &p_mode_encoded);
}

void RenderingServer::canvas_set_shadow_texture_size(int32_t p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("canvas_set_shadow_texture_size")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size_encoded);
}

void RenderingServer::global_shader_parameter_add(const StringName &p_name, RenderingServer::GlobalShaderParameterType p_type, const Variant &p_default_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("global_shader_parameter_add")._native_ptr(), 463390080);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_type_encoded, &p_default_value);
}

void RenderingServer::global_shader_parameter_remove(const StringName &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("global_shader_parameter_remove")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name);
}

TypedArray<StringName> RenderingServer::global_shader_parameter_get_list() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("global_shader_parameter_get_list")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<StringName>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<StringName>>(_gde_method_bind, _owner);
}

void RenderingServer::global_shader_parameter_set(const StringName &p_name, const Variant &p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("global_shader_parameter_set")._native_ptr(), 3776071444);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_value);
}

void RenderingServer::global_shader_parameter_set_override(const StringName &p_name, const Variant &p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("global_shader_parameter_set_override")._native_ptr(), 3776071444);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_value);
}

Variant RenderingServer::global_shader_parameter_get(const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("global_shader_parameter_get")._native_ptr(), 2760726917);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_name);
}

RenderingServer::GlobalShaderParameterType RenderingServer::global_shader_parameter_get_type(const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("global_shader_parameter_get_type")._native_ptr(), 1601414142);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingServer::GlobalShaderParameterType(0)));
	return (RenderingServer::GlobalShaderParameterType)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_name);
}

void RenderingServer::free_rid(const RID &p_rid) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("free_rid")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rid);
}

void RenderingServer::request_frame_drawn_callback(const Callable &p_callable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("request_frame_drawn_callback")._native_ptr(), 1611583062);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_callable);
}

bool RenderingServer::has_changed() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("has_changed")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

uint64_t RenderingServer::get_rendering_info(RenderingServer::RenderingInfo p_info) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("get_rendering_info")._native_ptr(), 3763192241);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_info_encoded;
	PtrToArg<int64_t>::encode(p_info, &p_info_encoded);
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_info_encoded);
}

String RenderingServer::get_video_adapter_name() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("get_video_adapter_name")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String RenderingServer::get_video_adapter_vendor() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("get_video_adapter_vendor")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

RenderingDevice::DeviceType RenderingServer::get_video_adapter_type() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("get_video_adapter_type")._native_ptr(), 3099547011);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingDevice::DeviceType(0)));
	return (RenderingDevice::DeviceType)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

String RenderingServer::get_video_adapter_api_version() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("get_video_adapter_api_version")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String RenderingServer::get_current_rendering_driver_name() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("get_current_rendering_driver_name")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String RenderingServer::get_current_rendering_method() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("get_current_rendering_method")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

RID RenderingServer::make_sphere_mesh(int32_t p_latitudes, int32_t p_longitudes, float p_radius) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("make_sphere_mesh")._native_ptr(), 2251015897);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_latitudes_encoded;
	PtrToArg<int64_t>::encode(p_latitudes, &p_latitudes_encoded);
	int64_t p_longitudes_encoded;
	PtrToArg<int64_t>::encode(p_longitudes, &p_longitudes_encoded);
	double p_radius_encoded;
	PtrToArg<double>::encode(p_radius, &p_radius_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_latitudes_encoded, &p_longitudes_encoded, &p_radius_encoded);
}

RID RenderingServer::get_test_cube() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("get_test_cube")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

RID RenderingServer::get_test_texture() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("get_test_texture")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

RID RenderingServer::get_white_texture() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("get_white_texture")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void RenderingServer::set_boot_image_with_stretch(const Ref<Image> &p_image, const Color &p_color, RenderingServer::SplashStretchMode p_stretch_mode, bool p_use_filter) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("set_boot_image_with_stretch")._native_ptr(), 1104470771);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_stretch_mode_encoded;
	PtrToArg<int64_t>::encode(p_stretch_mode, &p_stretch_mode_encoded);
	int8_t p_use_filter_encoded;
	PtrToArg<bool>::encode(p_use_filter, &p_use_filter_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_image != nullptr ? &p_image->_owner : nullptr), &p_color, &p_stretch_mode_encoded, &p_use_filter_encoded);
}

void RenderingServer::set_boot_image(const Ref<Image> &p_image, const Color &p_color, bool p_scale, bool p_use_filter) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("set_boot_image")._native_ptr(), 3759744527);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_scale_encoded;
	PtrToArg<bool>::encode(p_scale, &p_scale_encoded);
	int8_t p_use_filter_encoded;
	PtrToArg<bool>::encode(p_use_filter, &p_use_filter_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_image != nullptr ? &p_image->_owner : nullptr), &p_color, &p_scale_encoded, &p_use_filter_encoded);
}

Color RenderingServer::get_default_clear_color() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("get_default_clear_color")._native_ptr(), 3200896285);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Color()));
	return ::godot::internal::_call_native_mb_ret<Color>(_gde_method_bind, _owner);
}

void RenderingServer::set_default_clear_color(const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("set_default_clear_color")._native_ptr(), 2920490490);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color);
}

bool RenderingServer::has_os_feature(const String &p_feature) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("has_os_feature")._native_ptr(), 3927539163);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_feature);
}

void RenderingServer::set_debug_generate_wireframes(bool p_generate) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("set_debug_generate_wireframes")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_generate_encoded;
	PtrToArg<bool>::encode(p_generate, &p_generate_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_generate_encoded);
}

bool RenderingServer::is_render_loop_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("is_render_loop_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RenderingServer::set_render_loop_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("set_render_loop_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

double RenderingServer::get_frame_setup_time_cpu() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("get_frame_setup_time_cpu")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void RenderingServer::force_sync() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("force_sync")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void RenderingServer::force_draw(bool p_swap_buffers, double p_frame_step) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("force_draw")._native_ptr(), 1076185472);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_swap_buffers_encoded;
	PtrToArg<bool>::encode(p_swap_buffers, &p_swap_buffers_encoded);
	double p_frame_step_encoded;
	PtrToArg<double>::encode(p_frame_step, &p_frame_step_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_swap_buffers_encoded, &p_frame_step_encoded);
}

RenderingDevice *RenderingServer::get_rendering_device() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("get_rendering_device")._native_ptr(), 1405107940);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<RenderingDevice>(_gde_method_bind, _owner);
}

RenderingDevice *RenderingServer::create_local_rendering_device() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("create_local_rendering_device")._native_ptr(), 1405107940);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<RenderingDevice>(_gde_method_bind, _owner);
}

bool RenderingServer::is_on_render_thread() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("is_on_render_thread")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void RenderingServer::call_on_render_thread(const Callable &p_callable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("call_on_render_thread")._native_ptr(), 1611583062);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_callable);
}

bool RenderingServer::has_feature(RenderingServer::Features p_feature) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingServer::get_class_static()._native_ptr(), StringName("has_feature")._native_ptr(), 598462696);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_feature_encoded;
	PtrToArg<int64_t>::encode(p_feature, &p_feature_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_feature_encoded);
}

} // namespace godot
