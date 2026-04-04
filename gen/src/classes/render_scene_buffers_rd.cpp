/**************************************************************************/
/*  render_scene_buffers_rd.cpp                                           */
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

#include <godot_cpp/classes/render_scene_buffers_rd.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/rd_texture_format.hpp>
#include <godot_cpp/classes/rd_texture_view.hpp>
#include <godot_cpp/variant/string_name.hpp>

namespace godot {

bool RenderSceneBuffersRD::has_texture(const StringName &p_context, const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersRD::get_class_static()._native_ptr(), StringName("has_texture")._native_ptr(), 471820014);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_context, &p_name);
}

RID RenderSceneBuffersRD::create_texture(const StringName &p_context, const StringName &p_name, RenderingDevice::DataFormat p_data_format, uint32_t p_usage_bits, RenderingDevice::TextureSamples p_texture_samples, const Vector2i &p_size, uint32_t p_layers, uint32_t p_mipmaps, bool p_unique, bool p_discardable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersRD::get_class_static()._native_ptr(), StringName("create_texture")._native_ptr(), 2950875024);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_data_format_encoded;
	PtrToArg<int64_t>::encode(p_data_format, &p_data_format_encoded);
	int64_t p_usage_bits_encoded;
	PtrToArg<int64_t>::encode(p_usage_bits, &p_usage_bits_encoded);
	int64_t p_texture_samples_encoded;
	PtrToArg<int64_t>::encode(p_texture_samples, &p_texture_samples_encoded);
	int64_t p_layers_encoded;
	PtrToArg<int64_t>::encode(p_layers, &p_layers_encoded);
	int64_t p_mipmaps_encoded;
	PtrToArg<int64_t>::encode(p_mipmaps, &p_mipmaps_encoded);
	int8_t p_unique_encoded;
	PtrToArg<bool>::encode(p_unique, &p_unique_encoded);
	int8_t p_discardable_encoded;
	PtrToArg<bool>::encode(p_discardable, &p_discardable_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_context, &p_name, &p_data_format_encoded, &p_usage_bits_encoded, &p_texture_samples_encoded, &p_size, &p_layers_encoded, &p_mipmaps_encoded, &p_unique_encoded, &p_discardable_encoded);
}

RID RenderSceneBuffersRD::create_texture_from_format(const StringName &p_context, const StringName &p_name, const Ref<RDTextureFormat> &p_format, const Ref<RDTextureView> &p_view, bool p_unique) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersRD::get_class_static()._native_ptr(), StringName("create_texture_from_format")._native_ptr(), 3344669382);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int8_t p_unique_encoded;
	PtrToArg<bool>::encode(p_unique, &p_unique_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_context, &p_name, (p_format != nullptr ? &p_format->_owner : nullptr), (p_view != nullptr ? &p_view->_owner : nullptr), &p_unique_encoded);
}

RID RenderSceneBuffersRD::create_texture_view(const StringName &p_context, const StringName &p_name, const StringName &p_view_name, const Ref<RDTextureView> &p_view) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersRD::get_class_static()._native_ptr(), StringName("create_texture_view")._native_ptr(), 283055834);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_context, &p_name, &p_view_name, (p_view != nullptr ? &p_view->_owner : nullptr));
}

RID RenderSceneBuffersRD::get_texture(const StringName &p_context, const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersRD::get_class_static()._native_ptr(), StringName("get_texture")._native_ptr(), 750006389);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_context, &p_name);
}

Ref<RDTextureFormat> RenderSceneBuffersRD::get_texture_format(const StringName &p_context, const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersRD::get_class_static()._native_ptr(), StringName("get_texture_format")._native_ptr(), 371461758);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<RDTextureFormat>()));
	return Ref<RDTextureFormat>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<RDTextureFormat>(_gde_method_bind, _owner, &p_context, &p_name));
}

RID RenderSceneBuffersRD::get_texture_slice(const StringName &p_context, const StringName &p_name, uint32_t p_layer, uint32_t p_mipmap, uint32_t p_layers, uint32_t p_mipmaps) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersRD::get_class_static()._native_ptr(), StringName("get_texture_slice")._native_ptr(), 588440706);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	int64_t p_mipmap_encoded;
	PtrToArg<int64_t>::encode(p_mipmap, &p_mipmap_encoded);
	int64_t p_layers_encoded;
	PtrToArg<int64_t>::encode(p_layers, &p_layers_encoded);
	int64_t p_mipmaps_encoded;
	PtrToArg<int64_t>::encode(p_mipmaps, &p_mipmaps_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_context, &p_name, &p_layer_encoded, &p_mipmap_encoded, &p_layers_encoded, &p_mipmaps_encoded);
}

RID RenderSceneBuffersRD::get_texture_slice_view(const StringName &p_context, const StringName &p_name, uint32_t p_layer, uint32_t p_mipmap, uint32_t p_layers, uint32_t p_mipmaps, const Ref<RDTextureView> &p_view) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersRD::get_class_static()._native_ptr(), StringName("get_texture_slice_view")._native_ptr(), 682451778);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	int64_t p_mipmap_encoded;
	PtrToArg<int64_t>::encode(p_mipmap, &p_mipmap_encoded);
	int64_t p_layers_encoded;
	PtrToArg<int64_t>::encode(p_layers, &p_layers_encoded);
	int64_t p_mipmaps_encoded;
	PtrToArg<int64_t>::encode(p_mipmaps, &p_mipmaps_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_context, &p_name, &p_layer_encoded, &p_mipmap_encoded, &p_layers_encoded, &p_mipmaps_encoded, (p_view != nullptr ? &p_view->_owner : nullptr));
}

Vector2i RenderSceneBuffersRD::get_texture_slice_size(const StringName &p_context, const StringName &p_name, uint32_t p_mipmap) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersRD::get_class_static()._native_ptr(), StringName("get_texture_slice_size")._native_ptr(), 2617625368);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	int64_t p_mipmap_encoded;
	PtrToArg<int64_t>::encode(p_mipmap, &p_mipmap_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner, &p_context, &p_name, &p_mipmap_encoded);
}

void RenderSceneBuffersRD::clear_context(const StringName &p_context) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersRD::get_class_static()._native_ptr(), StringName("clear_context")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_context);
}

RID RenderSceneBuffersRD::get_color_texture(bool p_msaa) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersRD::get_class_static()._native_ptr(), StringName("get_color_texture")._native_ptr(), 3050822880);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int8_t p_msaa_encoded;
	PtrToArg<bool>::encode(p_msaa, &p_msaa_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_msaa_encoded);
}

RID RenderSceneBuffersRD::get_color_layer(uint32_t p_layer, bool p_msaa) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersRD::get_class_static()._native_ptr(), StringName("get_color_layer")._native_ptr(), 3087988589);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	int8_t p_msaa_encoded;
	PtrToArg<bool>::encode(p_msaa, &p_msaa_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_layer_encoded, &p_msaa_encoded);
}

RID RenderSceneBuffersRD::get_depth_texture(bool p_msaa) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersRD::get_class_static()._native_ptr(), StringName("get_depth_texture")._native_ptr(), 3050822880);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int8_t p_msaa_encoded;
	PtrToArg<bool>::encode(p_msaa, &p_msaa_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_msaa_encoded);
}

RID RenderSceneBuffersRD::get_depth_layer(uint32_t p_layer, bool p_msaa) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersRD::get_class_static()._native_ptr(), StringName("get_depth_layer")._native_ptr(), 3087988589);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	int8_t p_msaa_encoded;
	PtrToArg<bool>::encode(p_msaa, &p_msaa_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_layer_encoded, &p_msaa_encoded);
}

RID RenderSceneBuffersRD::get_velocity_texture(bool p_msaa) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersRD::get_class_static()._native_ptr(), StringName("get_velocity_texture")._native_ptr(), 3050822880);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int8_t p_msaa_encoded;
	PtrToArg<bool>::encode(p_msaa, &p_msaa_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_msaa_encoded);
}

RID RenderSceneBuffersRD::get_velocity_layer(uint32_t p_layer, bool p_msaa) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersRD::get_class_static()._native_ptr(), StringName("get_velocity_layer")._native_ptr(), 3087988589);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	int8_t p_msaa_encoded;
	PtrToArg<bool>::encode(p_msaa, &p_msaa_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_layer_encoded, &p_msaa_encoded);
}

RID RenderSceneBuffersRD::get_render_target() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersRD::get_class_static()._native_ptr(), StringName("get_render_target")._native_ptr(), 2944877500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

uint32_t RenderSceneBuffersRD::get_view_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersRD::get_class_static()._native_ptr(), StringName("get_view_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Vector2i RenderSceneBuffersRD::get_internal_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersRD::get_class_static()._native_ptr(), StringName("get_internal_size")._native_ptr(), 3690982128);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner);
}

Vector2i RenderSceneBuffersRD::get_target_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersRD::get_class_static()._native_ptr(), StringName("get_target_size")._native_ptr(), 3690982128);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2i()));
	return ::godot::internal::_call_native_mb_ret<Vector2i>(_gde_method_bind, _owner);
}

RenderingServer::ViewportScaling3DMode RenderSceneBuffersRD::get_scaling_3d_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersRD::get_class_static()._native_ptr(), StringName("get_scaling_3d_mode")._native_ptr(), 976778074);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingServer::ViewportScaling3DMode(0)));
	return (RenderingServer::ViewportScaling3DMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

float RenderSceneBuffersRD::get_fsr_sharpness() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersRD::get_class_static()._native_ptr(), StringName("get_fsr_sharpness")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

RenderingServer::ViewportMSAA RenderSceneBuffersRD::get_msaa_3d() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersRD::get_class_static()._native_ptr(), StringName("get_msaa_3d")._native_ptr(), 3109158617);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingServer::ViewportMSAA(0)));
	return (RenderingServer::ViewportMSAA)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

RenderingDevice::TextureSamples RenderSceneBuffersRD::get_texture_samples() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersRD::get_class_static()._native_ptr(), StringName("get_texture_samples")._native_ptr(), 407791724);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingDevice::TextureSamples(0)));
	return (RenderingDevice::TextureSamples)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

RenderingServer::ViewportScreenSpaceAA RenderSceneBuffersRD::get_screen_space_aa() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersRD::get_class_static()._native_ptr(), StringName("get_screen_space_aa")._native_ptr(), 641513172);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingServer::ViewportScreenSpaceAA(0)));
	return (RenderingServer::ViewportScreenSpaceAA)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

bool RenderSceneBuffersRD::get_use_taa() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersRD::get_class_static()._native_ptr(), StringName("get_use_taa")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool RenderSceneBuffersRD::get_use_debanding() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneBuffersRD::get_class_static()._native_ptr(), StringName("get_use_debanding")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

} // namespace godot
