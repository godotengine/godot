/**************************************************************************/
/*  rendering_device.cpp                                                  */
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

#include <godot_cpp/classes/rendering_device.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/rd_attachment_format.hpp>
#include <godot_cpp/classes/rd_framebuffer_pass.hpp>
#include <godot_cpp/classes/rd_pipeline_color_blend_state.hpp>
#include <godot_cpp/classes/rd_pipeline_depth_stencil_state.hpp>
#include <godot_cpp/classes/rd_pipeline_multisample_state.hpp>
#include <godot_cpp/classes/rd_pipeline_rasterization_state.hpp>
#include <godot_cpp/classes/rd_sampler_state.hpp>
#include <godot_cpp/classes/rd_shader_source.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/rd_texture_format.hpp>
#include <godot_cpp/classes/rd_texture_view.hpp>
#include <godot_cpp/classes/rd_uniform.hpp>
#include <godot_cpp/classes/rd_vertex_attribute.hpp>
#include <godot_cpp/variant/callable.hpp>
#include <godot_cpp/variant/vector2i.hpp>
#include <godot_cpp/variant/vector3.hpp>

namespace godot {

RID RenderingDevice::texture_create(const Ref<RDTextureFormat> &p_format, const Ref<RDTextureView> &p_view, const TypedArray<PackedByteArray> &p_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("texture_create")._native_ptr(), 3709173589);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, (p_format != nullptr ? &p_format->_owner : nullptr), (p_view != nullptr ? &p_view->_owner : nullptr), &p_data);
}

RID RenderingDevice::texture_create_shared(const Ref<RDTextureView> &p_view, const RID &p_with_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("texture_create_shared")._native_ptr(), 3178156134);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, (p_view != nullptr ? &p_view->_owner : nullptr), &p_with_texture);
}

RID RenderingDevice::texture_create_shared_from_slice(const Ref<RDTextureView> &p_view, const RID &p_with_texture, uint32_t p_layer, uint32_t p_mipmap, uint32_t p_mipmaps, RenderingDevice::TextureSliceType p_slice_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("texture_create_shared_from_slice")._native_ptr(), 1808971279);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	int64_t p_mipmap_encoded;
	PtrToArg<int64_t>::encode(p_mipmap, &p_mipmap_encoded);
	int64_t p_mipmaps_encoded;
	PtrToArg<int64_t>::encode(p_mipmaps, &p_mipmaps_encoded);
	int64_t p_slice_type_encoded;
	PtrToArg<int64_t>::encode(p_slice_type, &p_slice_type_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, (p_view != nullptr ? &p_view->_owner : nullptr), &p_with_texture, &p_layer_encoded, &p_mipmap_encoded, &p_mipmaps_encoded, &p_slice_type_encoded);
}

RID RenderingDevice::texture_create_from_extension(RenderingDevice::TextureType p_type, RenderingDevice::DataFormat p_format, RenderingDevice::TextureSamples p_samples, BitField<RenderingDevice::TextureUsageBits> p_usage_flags, uint64_t p_image, uint64_t p_width, uint64_t p_height, uint64_t p_depth, uint64_t p_layers, uint64_t p_mipmaps) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("texture_create_from_extension")._native_ptr(), 3732868568);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	int64_t p_format_encoded;
	PtrToArg<int64_t>::encode(p_format, &p_format_encoded);
	int64_t p_samples_encoded;
	PtrToArg<int64_t>::encode(p_samples, &p_samples_encoded);
	int64_t p_image_encoded;
	PtrToArg<int64_t>::encode(p_image, &p_image_encoded);
	int64_t p_width_encoded;
	PtrToArg<int64_t>::encode(p_width, &p_width_encoded);
	int64_t p_height_encoded;
	PtrToArg<int64_t>::encode(p_height, &p_height_encoded);
	int64_t p_depth_encoded;
	PtrToArg<int64_t>::encode(p_depth, &p_depth_encoded);
	int64_t p_layers_encoded;
	PtrToArg<int64_t>::encode(p_layers, &p_layers_encoded);
	int64_t p_mipmaps_encoded;
	PtrToArg<int64_t>::encode(p_mipmaps, &p_mipmaps_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_type_encoded, &p_format_encoded, &p_samples_encoded, &p_usage_flags, &p_image_encoded, &p_width_encoded, &p_height_encoded, &p_depth_encoded, &p_layers_encoded, &p_mipmaps_encoded);
}

Error RenderingDevice::texture_update(const RID &p_texture, uint32_t p_layer, const PackedByteArray &p_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("texture_update")._native_ptr(), 1349464008);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_texture, &p_layer_encoded, &p_data);
}

PackedByteArray RenderingDevice::texture_get_data(const RID &p_texture, uint32_t p_layer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("texture_get_data")._native_ptr(), 1859412099);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedByteArray()));
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedByteArray>(_gde_method_bind, _owner, &p_texture, &p_layer_encoded);
}

Error RenderingDevice::texture_get_data_async(const RID &p_texture, uint32_t p_layer, const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("texture_get_data_async")._native_ptr(), 498832090);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_texture, &p_layer_encoded, &p_callback);
}

bool RenderingDevice::texture_is_format_supported_for_usage(RenderingDevice::DataFormat p_format, BitField<RenderingDevice::TextureUsageBits> p_usage_flags) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("texture_is_format_supported_for_usage")._native_ptr(), 2592520478);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_format_encoded;
	PtrToArg<int64_t>::encode(p_format, &p_format_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_format_encoded, &p_usage_flags);
}

bool RenderingDevice::texture_is_shared(const RID &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("texture_is_shared")._native_ptr(), 3521089500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_texture);
}

bool RenderingDevice::texture_is_valid(const RID &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("texture_is_valid")._native_ptr(), 3521089500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_texture);
}

void RenderingDevice::texture_set_discardable(const RID &p_texture, bool p_discardable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("texture_set_discardable")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_discardable_encoded;
	PtrToArg<bool>::encode(p_discardable, &p_discardable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_texture, &p_discardable_encoded);
}

bool RenderingDevice::texture_is_discardable(const RID &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("texture_is_discardable")._native_ptr(), 3521089500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_texture);
}

Error RenderingDevice::texture_copy(const RID &p_from_texture, const RID &p_to_texture, const Vector3 &p_from_pos, const Vector3 &p_to_pos, const Vector3 &p_size, uint32_t p_src_mipmap, uint32_t p_dst_mipmap, uint32_t p_src_layer, uint32_t p_dst_layer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("texture_copy")._native_ptr(), 2859522160);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_src_mipmap_encoded;
	PtrToArg<int64_t>::encode(p_src_mipmap, &p_src_mipmap_encoded);
	int64_t p_dst_mipmap_encoded;
	PtrToArg<int64_t>::encode(p_dst_mipmap, &p_dst_mipmap_encoded);
	int64_t p_src_layer_encoded;
	PtrToArg<int64_t>::encode(p_src_layer, &p_src_layer_encoded);
	int64_t p_dst_layer_encoded;
	PtrToArg<int64_t>::encode(p_dst_layer, &p_dst_layer_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_from_texture, &p_to_texture, &p_from_pos, &p_to_pos, &p_size, &p_src_mipmap_encoded, &p_dst_mipmap_encoded, &p_src_layer_encoded, &p_dst_layer_encoded);
}

Error RenderingDevice::texture_clear(const RID &p_texture, const Color &p_color, uint32_t p_base_mipmap, uint32_t p_mipmap_count, uint32_t p_base_layer, uint32_t p_layer_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("texture_clear")._native_ptr(), 3477703247);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_base_mipmap_encoded;
	PtrToArg<int64_t>::encode(p_base_mipmap, &p_base_mipmap_encoded);
	int64_t p_mipmap_count_encoded;
	PtrToArg<int64_t>::encode(p_mipmap_count, &p_mipmap_count_encoded);
	int64_t p_base_layer_encoded;
	PtrToArg<int64_t>::encode(p_base_layer, &p_base_layer_encoded);
	int64_t p_layer_count_encoded;
	PtrToArg<int64_t>::encode(p_layer_count, &p_layer_count_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_texture, &p_color, &p_base_mipmap_encoded, &p_mipmap_count_encoded, &p_base_layer_encoded, &p_layer_count_encoded);
}

Error RenderingDevice::texture_resolve_multisample(const RID &p_from_texture, const RID &p_to_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("texture_resolve_multisample")._native_ptr(), 3181288260);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_from_texture, &p_to_texture);
}

Ref<RDTextureFormat> RenderingDevice::texture_get_format(const RID &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("texture_get_format")._native_ptr(), 1374471690);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<RDTextureFormat>()));
	return Ref<RDTextureFormat>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<RDTextureFormat>(_gde_method_bind, _owner, &p_texture));
}

uint64_t RenderingDevice::texture_get_native_handle(const RID &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("texture_get_native_handle")._native_ptr(), 3917799429);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_texture);
}

int64_t RenderingDevice::framebuffer_format_create(const TypedArray<Ref<RDAttachmentFormat>> &p_attachments, uint32_t p_view_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("framebuffer_format_create")._native_ptr(), 697032759);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_view_count_encoded;
	PtrToArg<int64_t>::encode(p_view_count, &p_view_count_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_attachments, &p_view_count_encoded);
}

int64_t RenderingDevice::framebuffer_format_create_multipass(const TypedArray<Ref<RDAttachmentFormat>> &p_attachments, const TypedArray<Ref<RDFramebufferPass>> &p_passes, uint32_t p_view_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("framebuffer_format_create_multipass")._native_ptr(), 2647479094);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_view_count_encoded;
	PtrToArg<int64_t>::encode(p_view_count, &p_view_count_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_attachments, &p_passes, &p_view_count_encoded);
}

int64_t RenderingDevice::framebuffer_format_create_empty(RenderingDevice::TextureSamples p_samples) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("framebuffer_format_create_empty")._native_ptr(), 555930169);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_samples_encoded;
	PtrToArg<int64_t>::encode(p_samples, &p_samples_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_samples_encoded);
}

RenderingDevice::TextureSamples RenderingDevice::framebuffer_format_get_texture_samples(int64_t p_format, uint32_t p_render_pass) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("framebuffer_format_get_texture_samples")._native_ptr(), 4223391010);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingDevice::TextureSamples(0)));
	int64_t p_format_encoded;
	PtrToArg<int64_t>::encode(p_format, &p_format_encoded);
	int64_t p_render_pass_encoded;
	PtrToArg<int64_t>::encode(p_render_pass, &p_render_pass_encoded);
	return (RenderingDevice::TextureSamples)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_format_encoded, &p_render_pass_encoded);
}

RID RenderingDevice::framebuffer_create(const TypedArray<RID> &p_textures, int64_t p_validate_with_format, uint32_t p_view_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("framebuffer_create")._native_ptr(), 3284231055);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_validate_with_format_encoded;
	PtrToArg<int64_t>::encode(p_validate_with_format, &p_validate_with_format_encoded);
	int64_t p_view_count_encoded;
	PtrToArg<int64_t>::encode(p_view_count, &p_view_count_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_textures, &p_validate_with_format_encoded, &p_view_count_encoded);
}

RID RenderingDevice::framebuffer_create_multipass(const TypedArray<RID> &p_textures, const TypedArray<Ref<RDFramebufferPass>> &p_passes, int64_t p_validate_with_format, uint32_t p_view_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("framebuffer_create_multipass")._native_ptr(), 1750306695);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_validate_with_format_encoded;
	PtrToArg<int64_t>::encode(p_validate_with_format, &p_validate_with_format_encoded);
	int64_t p_view_count_encoded;
	PtrToArg<int64_t>::encode(p_view_count, &p_view_count_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_textures, &p_passes, &p_validate_with_format_encoded, &p_view_count_encoded);
}

RID RenderingDevice::framebuffer_create_empty(const Vector2i &p_size, RenderingDevice::TextureSamples p_samples, int64_t p_validate_with_format) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("framebuffer_create_empty")._native_ptr(), 3058360618);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_samples_encoded;
	PtrToArg<int64_t>::encode(p_samples, &p_samples_encoded);
	int64_t p_validate_with_format_encoded;
	PtrToArg<int64_t>::encode(p_validate_with_format, &p_validate_with_format_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_size, &p_samples_encoded, &p_validate_with_format_encoded);
}

int64_t RenderingDevice::framebuffer_get_format(const RID &p_framebuffer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("framebuffer_get_format")._native_ptr(), 3917799429);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_framebuffer);
}

bool RenderingDevice::framebuffer_is_valid(const RID &p_framebuffer) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("framebuffer_is_valid")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_framebuffer);
}

RID RenderingDevice::sampler_create(const Ref<RDSamplerState> &p_state) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("sampler_create")._native_ptr(), 2327892535);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, (p_state != nullptr ? &p_state->_owner : nullptr));
}

bool RenderingDevice::sampler_is_format_supported_for_filter(RenderingDevice::DataFormat p_format, RenderingDevice::SamplerFilter p_sampler_filter) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("sampler_is_format_supported_for_filter")._native_ptr(), 2247922238);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_format_encoded;
	PtrToArg<int64_t>::encode(p_format, &p_format_encoded);
	int64_t p_sampler_filter_encoded;
	PtrToArg<int64_t>::encode(p_sampler_filter, &p_sampler_filter_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_format_encoded, &p_sampler_filter_encoded);
}

RID RenderingDevice::vertex_buffer_create(uint32_t p_size_bytes, const PackedByteArray &p_data, BitField<RenderingDevice::BufferCreationBits> p_creation_bits) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("vertex_buffer_create")._native_ptr(), 2089548973);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_size_bytes_encoded;
	PtrToArg<int64_t>::encode(p_size_bytes, &p_size_bytes_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_size_bytes_encoded, &p_data, &p_creation_bits);
}

int64_t RenderingDevice::vertex_format_create(const TypedArray<Ref<RDVertexAttribute>> &p_vertex_descriptions) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("vertex_format_create")._native_ptr(), 1242678479);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_vertex_descriptions);
}

RID RenderingDevice::vertex_array_create(uint32_t p_vertex_count, int64_t p_vertex_format, const TypedArray<RID> &p_src_buffers, const PackedInt64Array &p_offsets) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("vertex_array_create")._native_ptr(), 3799816279);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_vertex_count_encoded;
	PtrToArg<int64_t>::encode(p_vertex_count, &p_vertex_count_encoded);
	int64_t p_vertex_format_encoded;
	PtrToArg<int64_t>::encode(p_vertex_format, &p_vertex_format_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_vertex_count_encoded, &p_vertex_format_encoded, &p_src_buffers, &p_offsets);
}

RID RenderingDevice::index_buffer_create(uint32_t p_size_indices, RenderingDevice::IndexBufferFormat p_format, const PackedByteArray &p_data, bool p_use_restart_indices, BitField<RenderingDevice::BufferCreationBits> p_creation_bits) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("index_buffer_create")._native_ptr(), 2368684885);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_size_indices_encoded;
	PtrToArg<int64_t>::encode(p_size_indices, &p_size_indices_encoded);
	int64_t p_format_encoded;
	PtrToArg<int64_t>::encode(p_format, &p_format_encoded);
	int8_t p_use_restart_indices_encoded;
	PtrToArg<bool>::encode(p_use_restart_indices, &p_use_restart_indices_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_size_indices_encoded, &p_format_encoded, &p_data, &p_use_restart_indices_encoded, &p_creation_bits);
}

RID RenderingDevice::index_array_create(const RID &p_index_buffer, uint32_t p_index_offset, uint32_t p_index_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("index_array_create")._native_ptr(), 2256026069);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_index_offset_encoded;
	PtrToArg<int64_t>::encode(p_index_offset, &p_index_offset_encoded);
	int64_t p_index_count_encoded;
	PtrToArg<int64_t>::encode(p_index_count, &p_index_count_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_index_buffer, &p_index_offset_encoded, &p_index_count_encoded);
}

Ref<RDShaderSPIRV> RenderingDevice::shader_compile_spirv_from_source(const Ref<RDShaderSource> &p_shader_source, bool p_allow_cache) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("shader_compile_spirv_from_source")._native_ptr(), 1178973306);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<RDShaderSPIRV>()));
	int8_t p_allow_cache_encoded;
	PtrToArg<bool>::encode(p_allow_cache, &p_allow_cache_encoded);
	return Ref<RDShaderSPIRV>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<RDShaderSPIRV>(_gde_method_bind, _owner, (p_shader_source != nullptr ? &p_shader_source->_owner : nullptr), &p_allow_cache_encoded));
}

PackedByteArray RenderingDevice::shader_compile_binary_from_spirv(const Ref<RDShaderSPIRV> &p_spirv_data, const String &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("shader_compile_binary_from_spirv")._native_ptr(), 134910450);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedByteArray()));
	return ::godot::internal::_call_native_mb_ret<PackedByteArray>(_gde_method_bind, _owner, (p_spirv_data != nullptr ? &p_spirv_data->_owner : nullptr), &p_name);
}

RID RenderingDevice::shader_create_from_spirv(const Ref<RDShaderSPIRV> &p_spirv_data, const String &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("shader_create_from_spirv")._native_ptr(), 342949005);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, (p_spirv_data != nullptr ? &p_spirv_data->_owner : nullptr), &p_name);
}

RID RenderingDevice::shader_create_from_bytecode(const PackedByteArray &p_binary_data, const RID &p_placeholder_rid) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("shader_create_from_bytecode")._native_ptr(), 1687031350);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_binary_data, &p_placeholder_rid);
}

RID RenderingDevice::shader_create_placeholder() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("shader_create_placeholder")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

uint64_t RenderingDevice::shader_get_vertex_input_attribute_mask(const RID &p_shader) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("shader_get_vertex_input_attribute_mask")._native_ptr(), 3917799429);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_shader);
}

RID RenderingDevice::uniform_buffer_create(uint32_t p_size_bytes, const PackedByteArray &p_data, BitField<RenderingDevice::BufferCreationBits> p_creation_bits) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("uniform_buffer_create")._native_ptr(), 2089548973);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_size_bytes_encoded;
	PtrToArg<int64_t>::encode(p_size_bytes, &p_size_bytes_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_size_bytes_encoded, &p_data, &p_creation_bits);
}

RID RenderingDevice::storage_buffer_create(uint32_t p_size_bytes, const PackedByteArray &p_data, BitField<RenderingDevice::StorageBufferUsage> p_usage, BitField<RenderingDevice::BufferCreationBits> p_creation_bits) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("storage_buffer_create")._native_ptr(), 1609052553);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_size_bytes_encoded;
	PtrToArg<int64_t>::encode(p_size_bytes, &p_size_bytes_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_size_bytes_encoded, &p_data, &p_usage, &p_creation_bits);
}

RID RenderingDevice::texture_buffer_create(uint32_t p_size_bytes, RenderingDevice::DataFormat p_format, const PackedByteArray &p_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("texture_buffer_create")._native_ptr(), 1470338698);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_size_bytes_encoded;
	PtrToArg<int64_t>::encode(p_size_bytes, &p_size_bytes_encoded);
	int64_t p_format_encoded;
	PtrToArg<int64_t>::encode(p_format, &p_format_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_size_bytes_encoded, &p_format_encoded, &p_data);
}

RID RenderingDevice::uniform_set_create(const TypedArray<Ref<RDUniform>> &p_uniforms, const RID &p_shader, uint32_t p_shader_set) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("uniform_set_create")._native_ptr(), 2280795797);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_shader_set_encoded;
	PtrToArg<int64_t>::encode(p_shader_set, &p_shader_set_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_uniforms, &p_shader, &p_shader_set_encoded);
}

bool RenderingDevice::uniform_set_is_valid(const RID &p_uniform_set) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("uniform_set_is_valid")._native_ptr(), 3521089500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_uniform_set);
}

Error RenderingDevice::buffer_copy(const RID &p_src_buffer, const RID &p_dst_buffer, uint32_t p_src_offset, uint32_t p_dst_offset, uint32_t p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("buffer_copy")._native_ptr(), 864257779);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_src_offset_encoded;
	PtrToArg<int64_t>::encode(p_src_offset, &p_src_offset_encoded);
	int64_t p_dst_offset_encoded;
	PtrToArg<int64_t>::encode(p_dst_offset, &p_dst_offset_encoded);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_src_buffer, &p_dst_buffer, &p_src_offset_encoded, &p_dst_offset_encoded, &p_size_encoded);
}

Error RenderingDevice::buffer_update(const RID &p_buffer, uint32_t p_offset, uint32_t p_size_bytes, const PackedByteArray &p_data) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("buffer_update")._native_ptr(), 3454956949);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_offset_encoded;
	PtrToArg<int64_t>::encode(p_offset, &p_offset_encoded);
	int64_t p_size_bytes_encoded;
	PtrToArg<int64_t>::encode(p_size_bytes, &p_size_bytes_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_buffer, &p_offset_encoded, &p_size_bytes_encoded, &p_data);
}

Error RenderingDevice::buffer_clear(const RID &p_buffer, uint32_t p_offset, uint32_t p_size_bytes) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("buffer_clear")._native_ptr(), 2452320800);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_offset_encoded;
	PtrToArg<int64_t>::encode(p_offset, &p_offset_encoded);
	int64_t p_size_bytes_encoded;
	PtrToArg<int64_t>::encode(p_size_bytes, &p_size_bytes_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_buffer, &p_offset_encoded, &p_size_bytes_encoded);
}

PackedByteArray RenderingDevice::buffer_get_data(const RID &p_buffer, uint32_t p_offset_bytes, uint32_t p_size_bytes) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("buffer_get_data")._native_ptr(), 3101830688);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedByteArray()));
	int64_t p_offset_bytes_encoded;
	PtrToArg<int64_t>::encode(p_offset_bytes, &p_offset_bytes_encoded);
	int64_t p_size_bytes_encoded;
	PtrToArg<int64_t>::encode(p_size_bytes, &p_size_bytes_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedByteArray>(_gde_method_bind, _owner, &p_buffer, &p_offset_bytes_encoded, &p_size_bytes_encoded);
}

Error RenderingDevice::buffer_get_data_async(const RID &p_buffer, const Callable &p_callback, uint32_t p_offset_bytes, uint32_t p_size_bytes) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("buffer_get_data_async")._native_ptr(), 2370287848);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_offset_bytes_encoded;
	PtrToArg<int64_t>::encode(p_offset_bytes, &p_offset_bytes_encoded);
	int64_t p_size_bytes_encoded;
	PtrToArg<int64_t>::encode(p_size_bytes, &p_size_bytes_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_buffer, &p_callback, &p_offset_bytes_encoded, &p_size_bytes_encoded);
}

uint64_t RenderingDevice::buffer_get_device_address(const RID &p_buffer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("buffer_get_device_address")._native_ptr(), 3917799429);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_buffer);
}

RID RenderingDevice::render_pipeline_create(const RID &p_shader, int64_t p_framebuffer_format, int64_t p_vertex_format, RenderingDevice::RenderPrimitive p_primitive, const Ref<RDPipelineRasterizationState> &p_rasterization_state, const Ref<RDPipelineMultisampleState> &p_multisample_state, const Ref<RDPipelineDepthStencilState> &p_stencil_state, const Ref<RDPipelineColorBlendState> &p_color_blend_state, BitField<RenderingDevice::PipelineDynamicStateFlags> p_dynamic_state_flags, uint32_t p_for_render_pass, const TypedArray<Ref<RDPipelineSpecializationConstant>> &p_specialization_constants) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("render_pipeline_create")._native_ptr(), 2385451958);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_framebuffer_format_encoded;
	PtrToArg<int64_t>::encode(p_framebuffer_format, &p_framebuffer_format_encoded);
	int64_t p_vertex_format_encoded;
	PtrToArg<int64_t>::encode(p_vertex_format, &p_vertex_format_encoded);
	int64_t p_primitive_encoded;
	PtrToArg<int64_t>::encode(p_primitive, &p_primitive_encoded);
	int64_t p_for_render_pass_encoded;
	PtrToArg<int64_t>::encode(p_for_render_pass, &p_for_render_pass_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_shader, &p_framebuffer_format_encoded, &p_vertex_format_encoded, &p_primitive_encoded, (p_rasterization_state != nullptr ? &p_rasterization_state->_owner : nullptr), (p_multisample_state != nullptr ? &p_multisample_state->_owner : nullptr), (p_stencil_state != nullptr ? &p_stencil_state->_owner : nullptr), (p_color_blend_state != nullptr ? &p_color_blend_state->_owner : nullptr), &p_dynamic_state_flags, &p_for_render_pass_encoded, &p_specialization_constants);
}

bool RenderingDevice::render_pipeline_is_valid(const RID &p_render_pipeline) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("render_pipeline_is_valid")._native_ptr(), 3521089500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_render_pipeline);
}

RID RenderingDevice::compute_pipeline_create(const RID &p_shader, const TypedArray<Ref<RDPipelineSpecializationConstant>> &p_specialization_constants) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("compute_pipeline_create")._native_ptr(), 1448838280);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_shader, &p_specialization_constants);
}

bool RenderingDevice::compute_pipeline_is_valid(const RID &p_compute_pipeline) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("compute_pipeline_is_valid")._native_ptr(), 3521089500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_compute_pipeline);
}

int32_t RenderingDevice::screen_get_width(int32_t p_screen) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("screen_get_width")._native_ptr(), 1591665591);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_screen_encoded;
	PtrToArg<int64_t>::encode(p_screen, &p_screen_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_screen_encoded);
}

int32_t RenderingDevice::screen_get_height(int32_t p_screen) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("screen_get_height")._native_ptr(), 1591665591);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_screen_encoded;
	PtrToArg<int64_t>::encode(p_screen, &p_screen_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_screen_encoded);
}

int64_t RenderingDevice::screen_get_framebuffer_format(int32_t p_screen) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("screen_get_framebuffer_format")._native_ptr(), 1591665591);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_screen_encoded;
	PtrToArg<int64_t>::encode(p_screen, &p_screen_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_screen_encoded);
}

int64_t RenderingDevice::draw_list_begin_for_screen(int32_t p_screen, const Color &p_clear_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("draw_list_begin_for_screen")._native_ptr(), 3988079995);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_screen_encoded;
	PtrToArg<int64_t>::encode(p_screen, &p_screen_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_screen_encoded, &p_clear_color);
}

int64_t RenderingDevice::draw_list_begin(const RID &p_framebuffer, BitField<RenderingDevice::DrawFlags> p_draw_flags, const PackedColorArray &p_clear_color_values, float p_clear_depth_value, uint32_t p_clear_stencil_value, const Rect2 &p_region, uint32_t p_breadcrumb) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("draw_list_begin")._native_ptr(), 1317926357);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	double p_clear_depth_value_encoded;
	PtrToArg<double>::encode(p_clear_depth_value, &p_clear_depth_value_encoded);
	int64_t p_clear_stencil_value_encoded;
	PtrToArg<int64_t>::encode(p_clear_stencil_value, &p_clear_stencil_value_encoded);
	int64_t p_breadcrumb_encoded;
	PtrToArg<int64_t>::encode(p_breadcrumb, &p_breadcrumb_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_framebuffer, &p_draw_flags, &p_clear_color_values, &p_clear_depth_value_encoded, &p_clear_stencil_value_encoded, &p_region, &p_breadcrumb_encoded);
}

PackedInt64Array RenderingDevice::draw_list_begin_split(const RID &p_framebuffer, uint32_t p_splits, RenderingDevice::InitialAction p_initial_color_action, RenderingDevice::FinalAction p_final_color_action, RenderingDevice::InitialAction p_initial_depth_action, RenderingDevice::FinalAction p_final_depth_action, const PackedColorArray &p_clear_color_values, float p_clear_depth, uint32_t p_clear_stencil, const Rect2 &p_region, const TypedArray<RID> &p_storage_textures) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("draw_list_begin_split")._native_ptr(), 2406300660);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt64Array()));
	int64_t p_splits_encoded;
	PtrToArg<int64_t>::encode(p_splits, &p_splits_encoded);
	int64_t p_initial_color_action_encoded;
	PtrToArg<int64_t>::encode(p_initial_color_action, &p_initial_color_action_encoded);
	int64_t p_final_color_action_encoded;
	PtrToArg<int64_t>::encode(p_final_color_action, &p_final_color_action_encoded);
	int64_t p_initial_depth_action_encoded;
	PtrToArg<int64_t>::encode(p_initial_depth_action, &p_initial_depth_action_encoded);
	int64_t p_final_depth_action_encoded;
	PtrToArg<int64_t>::encode(p_final_depth_action, &p_final_depth_action_encoded);
	double p_clear_depth_encoded;
	PtrToArg<double>::encode(p_clear_depth, &p_clear_depth_encoded);
	int64_t p_clear_stencil_encoded;
	PtrToArg<int64_t>::encode(p_clear_stencil, &p_clear_stencil_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedInt64Array>(_gde_method_bind, _owner, &p_framebuffer, &p_splits_encoded, &p_initial_color_action_encoded, &p_final_color_action_encoded, &p_initial_depth_action_encoded, &p_final_depth_action_encoded, &p_clear_color_values, &p_clear_depth_encoded, &p_clear_stencil_encoded, &p_region, &p_storage_textures);
}

void RenderingDevice::draw_list_set_blend_constants(int64_t p_draw_list, const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("draw_list_set_blend_constants")._native_ptr(), 2878471219);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_draw_list_encoded;
	PtrToArg<int64_t>::encode(p_draw_list, &p_draw_list_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_draw_list_encoded, &p_color);
}

void RenderingDevice::draw_list_bind_render_pipeline(int64_t p_draw_list, const RID &p_render_pipeline) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("draw_list_bind_render_pipeline")._native_ptr(), 4040184819);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_draw_list_encoded;
	PtrToArg<int64_t>::encode(p_draw_list, &p_draw_list_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_draw_list_encoded, &p_render_pipeline);
}

void RenderingDevice::draw_list_bind_uniform_set(int64_t p_draw_list, const RID &p_uniform_set, uint32_t p_set_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("draw_list_bind_uniform_set")._native_ptr(), 749655778);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_draw_list_encoded;
	PtrToArg<int64_t>::encode(p_draw_list, &p_draw_list_encoded);
	int64_t p_set_index_encoded;
	PtrToArg<int64_t>::encode(p_set_index, &p_set_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_draw_list_encoded, &p_uniform_set, &p_set_index_encoded);
}

void RenderingDevice::draw_list_bind_vertex_array(int64_t p_draw_list, const RID &p_vertex_array) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("draw_list_bind_vertex_array")._native_ptr(), 4040184819);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_draw_list_encoded;
	PtrToArg<int64_t>::encode(p_draw_list, &p_draw_list_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_draw_list_encoded, &p_vertex_array);
}

void RenderingDevice::draw_list_bind_vertex_buffers_format(int64_t p_draw_list, int64_t p_vertex_format, uint32_t p_vertex_count, const TypedArray<RID> &p_vertex_buffers, const PackedInt64Array &p_offsets) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("draw_list_bind_vertex_buffers_format")._native_ptr(), 2008628980);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_draw_list_encoded;
	PtrToArg<int64_t>::encode(p_draw_list, &p_draw_list_encoded);
	int64_t p_vertex_format_encoded;
	PtrToArg<int64_t>::encode(p_vertex_format, &p_vertex_format_encoded);
	int64_t p_vertex_count_encoded;
	PtrToArg<int64_t>::encode(p_vertex_count, &p_vertex_count_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_draw_list_encoded, &p_vertex_format_encoded, &p_vertex_count_encoded, &p_vertex_buffers, &p_offsets);
}

void RenderingDevice::draw_list_bind_index_array(int64_t p_draw_list, const RID &p_index_array) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("draw_list_bind_index_array")._native_ptr(), 4040184819);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_draw_list_encoded;
	PtrToArg<int64_t>::encode(p_draw_list, &p_draw_list_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_draw_list_encoded, &p_index_array);
}

void RenderingDevice::draw_list_set_push_constant(int64_t p_draw_list, const PackedByteArray &p_buffer, uint32_t p_size_bytes) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("draw_list_set_push_constant")._native_ptr(), 2772371345);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_draw_list_encoded;
	PtrToArg<int64_t>::encode(p_draw_list, &p_draw_list_encoded);
	int64_t p_size_bytes_encoded;
	PtrToArg<int64_t>::encode(p_size_bytes, &p_size_bytes_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_draw_list_encoded, &p_buffer, &p_size_bytes_encoded);
}

void RenderingDevice::draw_list_draw(int64_t p_draw_list, bool p_use_indices, uint32_t p_instances, uint32_t p_procedural_vertex_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("draw_list_draw")._native_ptr(), 4230067973);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_draw_list_encoded;
	PtrToArg<int64_t>::encode(p_draw_list, &p_draw_list_encoded);
	int8_t p_use_indices_encoded;
	PtrToArg<bool>::encode(p_use_indices, &p_use_indices_encoded);
	int64_t p_instances_encoded;
	PtrToArg<int64_t>::encode(p_instances, &p_instances_encoded);
	int64_t p_procedural_vertex_count_encoded;
	PtrToArg<int64_t>::encode(p_procedural_vertex_count, &p_procedural_vertex_count_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_draw_list_encoded, &p_use_indices_encoded, &p_instances_encoded, &p_procedural_vertex_count_encoded);
}

void RenderingDevice::draw_list_draw_indirect(int64_t p_draw_list, bool p_use_indices, const RID &p_buffer, uint32_t p_offset, uint32_t p_draw_count, uint32_t p_stride) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("draw_list_draw_indirect")._native_ptr(), 1092133571);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_draw_list_encoded;
	PtrToArg<int64_t>::encode(p_draw_list, &p_draw_list_encoded);
	int8_t p_use_indices_encoded;
	PtrToArg<bool>::encode(p_use_indices, &p_use_indices_encoded);
	int64_t p_offset_encoded;
	PtrToArg<int64_t>::encode(p_offset, &p_offset_encoded);
	int64_t p_draw_count_encoded;
	PtrToArg<int64_t>::encode(p_draw_count, &p_draw_count_encoded);
	int64_t p_stride_encoded;
	PtrToArg<int64_t>::encode(p_stride, &p_stride_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_draw_list_encoded, &p_use_indices_encoded, &p_buffer, &p_offset_encoded, &p_draw_count_encoded, &p_stride_encoded);
}

void RenderingDevice::draw_list_enable_scissor(int64_t p_draw_list, const Rect2 &p_rect) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("draw_list_enable_scissor")._native_ptr(), 244650101);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_draw_list_encoded;
	PtrToArg<int64_t>::encode(p_draw_list, &p_draw_list_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_draw_list_encoded, &p_rect);
}

void RenderingDevice::draw_list_disable_scissor(int64_t p_draw_list) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("draw_list_disable_scissor")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_draw_list_encoded;
	PtrToArg<int64_t>::encode(p_draw_list, &p_draw_list_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_draw_list_encoded);
}

int64_t RenderingDevice::draw_list_switch_to_next_pass() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("draw_list_switch_to_next_pass")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

PackedInt64Array RenderingDevice::draw_list_switch_to_next_pass_split(uint32_t p_splits) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("draw_list_switch_to_next_pass_split")._native_ptr(), 2865087369);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt64Array()));
	int64_t p_splits_encoded;
	PtrToArg<int64_t>::encode(p_splits, &p_splits_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedInt64Array>(_gde_method_bind, _owner, &p_splits_encoded);
}

void RenderingDevice::draw_list_end() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("draw_list_end")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

int64_t RenderingDevice::compute_list_begin() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("compute_list_begin")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RenderingDevice::compute_list_bind_compute_pipeline(int64_t p_compute_list, const RID &p_compute_pipeline) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("compute_list_bind_compute_pipeline")._native_ptr(), 4040184819);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_compute_list_encoded;
	PtrToArg<int64_t>::encode(p_compute_list, &p_compute_list_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_compute_list_encoded, &p_compute_pipeline);
}

void RenderingDevice::compute_list_set_push_constant(int64_t p_compute_list, const PackedByteArray &p_buffer, uint32_t p_size_bytes) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("compute_list_set_push_constant")._native_ptr(), 2772371345);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_compute_list_encoded;
	PtrToArg<int64_t>::encode(p_compute_list, &p_compute_list_encoded);
	int64_t p_size_bytes_encoded;
	PtrToArg<int64_t>::encode(p_size_bytes, &p_size_bytes_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_compute_list_encoded, &p_buffer, &p_size_bytes_encoded);
}

void RenderingDevice::compute_list_bind_uniform_set(int64_t p_compute_list, const RID &p_uniform_set, uint32_t p_set_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("compute_list_bind_uniform_set")._native_ptr(), 749655778);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_compute_list_encoded;
	PtrToArg<int64_t>::encode(p_compute_list, &p_compute_list_encoded);
	int64_t p_set_index_encoded;
	PtrToArg<int64_t>::encode(p_set_index, &p_set_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_compute_list_encoded, &p_uniform_set, &p_set_index_encoded);
}

void RenderingDevice::compute_list_dispatch(int64_t p_compute_list, uint32_t p_x_groups, uint32_t p_y_groups, uint32_t p_z_groups) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("compute_list_dispatch")._native_ptr(), 4275841770);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_compute_list_encoded;
	PtrToArg<int64_t>::encode(p_compute_list, &p_compute_list_encoded);
	int64_t p_x_groups_encoded;
	PtrToArg<int64_t>::encode(p_x_groups, &p_x_groups_encoded);
	int64_t p_y_groups_encoded;
	PtrToArg<int64_t>::encode(p_y_groups, &p_y_groups_encoded);
	int64_t p_z_groups_encoded;
	PtrToArg<int64_t>::encode(p_z_groups, &p_z_groups_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_compute_list_encoded, &p_x_groups_encoded, &p_y_groups_encoded, &p_z_groups_encoded);
}

void RenderingDevice::compute_list_dispatch_indirect(int64_t p_compute_list, const RID &p_buffer, uint32_t p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("compute_list_dispatch_indirect")._native_ptr(), 749655778);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_compute_list_encoded;
	PtrToArg<int64_t>::encode(p_compute_list, &p_compute_list_encoded);
	int64_t p_offset_encoded;
	PtrToArg<int64_t>::encode(p_offset, &p_offset_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_compute_list_encoded, &p_buffer, &p_offset_encoded);
}

void RenderingDevice::compute_list_add_barrier(int64_t p_compute_list) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("compute_list_add_barrier")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_compute_list_encoded;
	PtrToArg<int64_t>::encode(p_compute_list, &p_compute_list_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_compute_list_encoded);
}

void RenderingDevice::compute_list_end() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("compute_list_end")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void RenderingDevice::free_rid(const RID &p_rid) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("free_rid")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rid);
}

void RenderingDevice::capture_timestamp(const String &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("capture_timestamp")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name);
}

uint32_t RenderingDevice::get_captured_timestamps_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("get_captured_timestamps_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

uint64_t RenderingDevice::get_captured_timestamps_frame() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("get_captured_timestamps_frame")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner);
}

uint64_t RenderingDevice::get_captured_timestamp_gpu_time(uint32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("get_captured_timestamp_gpu_time")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_index_encoded);
}

uint64_t RenderingDevice::get_captured_timestamp_cpu_time(uint32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("get_captured_timestamp_cpu_time")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_index_encoded);
}

String RenderingDevice::get_captured_timestamp_name(uint32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("get_captured_timestamp_name")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_index_encoded);
}

bool RenderingDevice::has_feature(RenderingDevice::Features p_feature) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("has_feature")._native_ptr(), 1772728326);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_feature_encoded;
	PtrToArg<int64_t>::encode(p_feature, &p_feature_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_feature_encoded);
}

uint64_t RenderingDevice::limit_get(RenderingDevice::Limit p_limit) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("limit_get")._native_ptr(), 1559202131);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_limit_encoded;
	PtrToArg<int64_t>::encode(p_limit, &p_limit_encoded);
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_limit_encoded);
}

uint32_t RenderingDevice::get_frame_delay() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("get_frame_delay")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void RenderingDevice::submit() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("submit")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void RenderingDevice::sync() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("sync")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void RenderingDevice::barrier(BitField<RenderingDevice::BarrierMask> p_from, BitField<RenderingDevice::BarrierMask> p_to) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("barrier")._native_ptr(), 3718155691);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_from, &p_to);
}

void RenderingDevice::full_barrier() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("full_barrier")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

RenderingDevice *RenderingDevice::create_local_device() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("create_local_device")._native_ptr(), 2846302423);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<RenderingDevice>(_gde_method_bind, _owner);
}

void RenderingDevice::set_resource_name(const RID &p_id, const String &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("set_resource_name")._native_ptr(), 2726140452);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_id, &p_name);
}

void RenderingDevice::draw_command_begin_label(const String &p_name, const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("draw_command_begin_label")._native_ptr(), 1636512886);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_color);
}

void RenderingDevice::draw_command_insert_label(const String &p_name, const Color &p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("draw_command_insert_label")._native_ptr(), 1636512886);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_color);
}

void RenderingDevice::draw_command_end_label() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("draw_command_end_label")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

String RenderingDevice::get_device_vendor_name() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("get_device_vendor_name")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String RenderingDevice::get_device_name() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("get_device_name")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String RenderingDevice::get_device_pipeline_cache_uuid() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("get_device_pipeline_cache_uuid")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

uint64_t RenderingDevice::get_memory_usage(RenderingDevice::MemoryType p_type) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("get_memory_usage")._native_ptr(), 251690689);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_type_encoded);
}

uint64_t RenderingDevice::get_driver_resource(RenderingDevice::DriverResource p_resource, const RID &p_rid, uint64_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("get_driver_resource")._native_ptr(), 501815484);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_resource_encoded;
	PtrToArg<int64_t>::encode(p_resource, &p_resource_encoded);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_resource_encoded, &p_rid, &p_index_encoded);
}

String RenderingDevice::get_perf_report() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("get_perf_report")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String RenderingDevice::get_driver_and_device_memory_report() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("get_driver_and_device_memory_report")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

String RenderingDevice::get_tracked_object_name(uint32_t p_type_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("get_tracked_object_name")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_type_index_encoded;
	PtrToArg<int64_t>::encode(p_type_index, &p_type_index_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_type_index_encoded);
}

uint64_t RenderingDevice::get_tracked_object_type_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("get_tracked_object_type_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner);
}

uint64_t RenderingDevice::get_driver_total_memory() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("get_driver_total_memory")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner);
}

uint64_t RenderingDevice::get_driver_allocation_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("get_driver_allocation_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner);
}

uint64_t RenderingDevice::get_driver_memory_by_object_type(uint32_t p_type) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("get_driver_memory_by_object_type")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_type_encoded);
}

uint64_t RenderingDevice::get_driver_allocs_by_object_type(uint32_t p_type) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("get_driver_allocs_by_object_type")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_type_encoded);
}

uint64_t RenderingDevice::get_device_total_memory() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("get_device_total_memory")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner);
}

uint64_t RenderingDevice::get_device_allocation_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("get_device_allocation_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner);
}

uint64_t RenderingDevice::get_device_memory_by_object_type(uint32_t p_type) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("get_device_memory_by_object_type")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_type_encoded);
}

uint64_t RenderingDevice::get_device_allocs_by_object_type(uint32_t p_type) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderingDevice::get_class_static()._native_ptr(), StringName("get_device_allocs_by_object_type")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_type_encoded);
}

} // namespace godot
