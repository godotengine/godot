/**************************************************************************/
/*  rendering_device_binds.h                                              */
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

#ifndef RENDERING_DEVICE_BINDS_H
#define RENDERING_DEVICE_BINDS_H

#include "servers/rendering/rendering_device.h"

#define RD_SETGET(m_member, m_type)                                            \
	void set_##m_member(m_type p_##m_member) { base.m_member = p_##m_member; } \
	m_type get_##m_member() const { return base.m_member; }

#define RD_BIND(m_class, m_member, m_variant_type)                                                          \
	ClassDB::bind_method(D_METHOD("set_" _MKSTR(m_member), "p_" _MKSTR(member)), &m_class::set_##m_member); \
	ClassDB::bind_method(D_METHOD("get_" _MKSTR(m_member)), &m_class::get_##m_member);                      \
	ADD_PROPERTY(PropertyInfo(m_variant_type, #m_member), "set_" _MKSTR(m_member), "get_" _MKSTR(m_member))

#define RD_BIND_ENUM(m_class, m_member, m_enum, m_options)                                                  \
	ClassDB::bind_method(D_METHOD("set_" _MKSTR(m_member), "p_" _MKSTR(member)), &m_class::set_##m_member); \
	ClassDB::bind_method(D_METHOD("get_" _MKSTR(m_member)), &m_class::get_##m_member);                      \
	ADD_PROPERTY(PropertyInfo::make_enum(#m_member, m_enum, m_options),                                     \
			"set_" _MKSTR(m_member), "get_" _MKSTR(m_member))

#define RD_BIND_FLAGS(m_class, m_member, m_enum, m_options)                                                 \
	ClassDB::bind_method(D_METHOD("set_" _MKSTR(m_member), "p_" _MKSTR(member)), &m_class::set_##m_member); \
	ClassDB::bind_method(D_METHOD("get_" _MKSTR(m_member)), &m_class::get_##m_member);                      \
	ADD_PROPERTY(PropertyInfo::make_flags(#m_member, m_enum, m_options),                                    \
			"set_" _MKSTR(m_member), "get_" _MKSTR(m_member))

#define RD_SETGET_SUB(m_sub, m_member, m_type)                                                 \
	void set_##m_sub##_##m_member(m_type p_##m_member) { base.m_sub.m_member = p_##m_member; } \
	m_type get_##m_sub##_##m_member() const { return base.m_sub.m_member; }

#define RD_BIND_SUB(m_class, m_sub, m_member, m_variant_type)                                                      \
	ClassDB::bind_method(D_METHOD("set_" _MKSTR(m_sub) "_" _MKSTR(m_member), "p_" _MKSTR(member)),                 \
			&m_class::set_##m_sub##_##m_member);                                                                   \
	ClassDB::bind_method(D_METHOD("get_" _MKSTR(m_sub) "_" _MKSTR(m_member)), &m_class::get_##m_sub##_##m_member); \
	ADD_PROPERTY(PropertyInfo(m_variant_type, _MKSTR(m_sub) "_" _MKSTR(m_member)),                                 \
			"set_" _MKSTR(m_sub) "_" _MKSTR(m_member), "get_" _MKSTR(m_sub) "_" _MKSTR(m_member))

#define RD_BIND_SUB_ENUM(m_class, m_sub, m_member, m_enum, m_options)                                              \
	ClassDB::bind_method(D_METHOD("set_" _MKSTR(m_sub) "_" _MKSTR(m_member), "p_" _MKSTR(member)),                 \
			&m_class::set_##m_sub##_##m_member);                                                                   \
	ClassDB::bind_method(D_METHOD("get_" _MKSTR(m_sub) "_" _MKSTR(m_member)), &m_class::get_##m_sub##_##m_member); \
	ADD_PROPERTY(PropertyInfo::make_enum(_MKSTR(m_sub) "_" _MKSTR(m_member), m_enum, m_options),                   \
			"set_" _MKSTR(m_sub) "_" _MKSTR(m_member), "get_" _MKSTR(m_sub) "_" _MKSTR(m_member))

#define OPTIONS_COMPARE_OPERATOR "Never,Less,Equal,Less or Equal,Greater,Not Equal,Greater or Equal,Always"
#define OPTIONS_LOGIC_OPERATION "Clear,AND,AND Reverse,Copy,AND Inverted,No-Op,XOR,OR,NOR,Equivalent,Invert,OR Reverse," \
								"Copy Inverted,OR Inverted,NAND,Set"
#define OPTIONS_STENCIL_OPERATION "Keep,Zero,Replace,Increment and Clamp,Decrement and Clamp,Invert,Increment and Wrap," \
								  "Decrement and Wrap"
#define OPTIONS_BLEND_OPERATION "Add,Subtract,Reverse Subtract,Minimum,Maximum"
#define OPTIONS_BLEND_FACTOR "Zero,One,Src Color,One Minus Src Color,Dst Color,One Minus Dst Color,Src Alpha,"             \
							 "One Minus Src Alpha,Dst Alpha,One Minus Dst Alpha,Constant Color,One Minus Constant Color,"  \
							 "Constant Alpha,One Minus Constant Alpha,Src Alpha Saturate,Src1 Color,One Minus Src1 Color," \
							 "Src1 Alpha,One Minus Src1 Alpha"

// There are too many formats and codespell triggers on some constant names.
#define OPTIONS_DATA_FORMAT ""

#define OPTIONS_TEXTURE_TYPE "1D,2D,3D,Cube,1D Array,2D Array,Cube Array"
#define OPTIONS_TEXTURE_SAMPLES "1,2,4,8,16,32,64"
#define OPTIONS_TEXTURE_SWIZZLE "Identity,Zero,One,R,G,B,A"
#define OPTIONS_TEXTURE_USAGE_BITS "Sampling,Color Attachment,Depth Stencil Attachment,Storage,Storage Atomic,CPU Read," \
								   "Can Update,Can Copy From,Can Copy To,Input Attachment"

#define OPTIONS_SAMPLER_FILTER "Nearest,Linear"
#define OPTIONS_SAMPLER_REPEAT_MODE "Repeat,Mirrored Repeat,Clamp to Edge,Clamp to Border,Mirror Clamp to Edge"
#define OPTIONS_SAMPLER_BORDER_COLOR "Float Transparent Black,Int Transparent Black,Float Opaque Black,Int Opaque Black," \
									 "Float Opaque White,Int Opaque White"

#define OPTIONS_POLYGON_CULL_MODE "Disabled,Front,Back"
#define OPTIONS_POLYGON_FRONT_FACE "Clockwise,Counter Clockwise"

#define OPTIONS_UNIFORM_TYPE "Sampler,Sampler with Texture,Texture,Image,Texture Buffer,Sampler with Texture Buffer," \
							 "Image Buffer,Uniform Buffer,Storage Buffer,Input Attachment"

#define OPTIONS_VERTEX_FREQUENCY "Vertex,Instance"

class RDTextureFormat : public RefCounted {
	GDCLASS(RDTextureFormat, RefCounted)

	friend class RenderingDevice;
	friend class RenderSceneBuffersRD;

	RD::TextureFormat base;

public:
	RD_SETGET(format, RD::DataFormat)
	RD_SETGET(width, uint32_t)
	RD_SETGET(height, uint32_t)
	RD_SETGET(depth, uint32_t)
	RD_SETGET(array_layers, uint32_t)
	RD_SETGET(mipmaps, uint32_t)
	RD_SETGET(texture_type, RD::TextureType)
	RD_SETGET(samples, RD::TextureSamples)
	RD_SETGET(usage_bits, BitField<RenderingDevice::TextureUsageBits>)

	void add_shareable_format(RD::DataFormat p_format) { base.shareable_formats.push_back(p_format); }
	void remove_shareable_format(RD::DataFormat p_format) { base.shareable_formats.erase(p_format); }

protected:
	static void _bind_methods() {
		RD_BIND_ENUM(RDTextureFormat, format, "RenderingDevice.DataFormat", OPTIONS_DATA_FORMAT);
		RD_BIND(RDTextureFormat, width, Variant::INT);
		RD_BIND(RDTextureFormat, height, Variant::INT);
		RD_BIND(RDTextureFormat, depth, Variant::INT);
		RD_BIND(RDTextureFormat, array_layers, Variant::INT);
		RD_BIND(RDTextureFormat, mipmaps, Variant::INT);
		RD_BIND_ENUM(RDTextureFormat, texture_type, "RenderingDevice.TextureType", OPTIONS_TEXTURE_TYPE);
		RD_BIND_ENUM(RDTextureFormat, samples, "RenderingDevice.TextureSamples", OPTIONS_TEXTURE_SAMPLES);
		RD_BIND_FLAGS(RDTextureFormat, usage_bits, "RenderingDevice.TextureUsageBits", OPTIONS_TEXTURE_USAGE_BITS);

		ClassDB::bind_method(D_METHOD("add_shareable_format", "format"), &RDTextureFormat::add_shareable_format);
		ClassDB::bind_method(D_METHOD("remove_shareable_format", "format"), &RDTextureFormat::remove_shareable_format);
	}
};

class RDTextureView : public RefCounted {
	GDCLASS(RDTextureView, RefCounted)

	friend class RenderingDevice;
	friend class RenderSceneBuffersRD;

	RD::TextureView base;

public:
	RD_SETGET(format_override, RD::DataFormat)
	RD_SETGET(swizzle_r, RD::TextureSwizzle)
	RD_SETGET(swizzle_g, RD::TextureSwizzle)
	RD_SETGET(swizzle_b, RD::TextureSwizzle)
	RD_SETGET(swizzle_a, RD::TextureSwizzle)

protected:
	static void _bind_methods() {
		RD_BIND_ENUM(RDTextureView, format_override, "RenderingDevice.DataFormat", OPTIONS_DATA_FORMAT);
		RD_BIND_ENUM(RDTextureView, swizzle_r, "RenderingDevice.TextureSwizzle", OPTIONS_TEXTURE_SWIZZLE);
		RD_BIND_ENUM(RDTextureView, swizzle_g, "RenderingDevice.TextureSwizzle", OPTIONS_TEXTURE_SWIZZLE);
		RD_BIND_ENUM(RDTextureView, swizzle_b, "RenderingDevice.TextureSwizzle", OPTIONS_TEXTURE_SWIZZLE);
		RD_BIND_ENUM(RDTextureView, swizzle_a, "RenderingDevice.TextureSwizzle", OPTIONS_TEXTURE_SWIZZLE);
	}
};

class RDAttachmentFormat : public RefCounted {
	GDCLASS(RDAttachmentFormat, RefCounted)
	friend class RenderingDevice;

	RD::AttachmentFormat base;

public:
	RD_SETGET(format, RD::DataFormat)
	RD_SETGET(samples, RD::TextureSamples)
	RD_SETGET(usage_flags, uint32_t)

protected:
	static void _bind_methods() {
		RD_BIND_ENUM(RDAttachmentFormat, format, "RenderingDevice.DataFormat", OPTIONS_DATA_FORMAT);
		RD_BIND_ENUM(RDAttachmentFormat, samples, "RenderingDevice.TextureSamples", OPTIONS_TEXTURE_SAMPLES);
		RD_BIND(RDAttachmentFormat, usage_flags, Variant::INT);
	}
};

class RDFramebufferPass : public RefCounted {
	GDCLASS(RDFramebufferPass, RefCounted)
	friend class RenderingDevice;
	friend class FramebufferCacheRD;

	RD::FramebufferPass base;

public:
	RD_SETGET(color_attachments, PackedInt32Array)
	RD_SETGET(input_attachments, PackedInt32Array)
	RD_SETGET(resolve_attachments, PackedInt32Array)
	RD_SETGET(preserve_attachments, PackedInt32Array)
	RD_SETGET(depth_attachment, int32_t)

protected:
	enum {
		ATTACHMENT_UNUSED = -1
	};

	static void _bind_methods() {
		RD_BIND(RDFramebufferPass, color_attachments, Variant::PACKED_INT32_ARRAY);
		RD_BIND(RDFramebufferPass, input_attachments, Variant::PACKED_INT32_ARRAY);
		RD_BIND(RDFramebufferPass, resolve_attachments, Variant::PACKED_INT32_ARRAY);
		RD_BIND(RDFramebufferPass, preserve_attachments, Variant::PACKED_INT32_ARRAY);
		RD_BIND(RDFramebufferPass, depth_attachment, Variant::INT);

		BIND_CONSTANT(ATTACHMENT_UNUSED);
	}
};

class RDSamplerState : public RefCounted {
	GDCLASS(RDSamplerState, RefCounted)
	friend class RenderingDevice;

	RD::SamplerState base;

public:
	RD_SETGET(mag_filter, RD::SamplerFilter)
	RD_SETGET(min_filter, RD::SamplerFilter)
	RD_SETGET(mip_filter, RD::SamplerFilter)
	RD_SETGET(repeat_u, RD::SamplerRepeatMode)
	RD_SETGET(repeat_v, RD::SamplerRepeatMode)
	RD_SETGET(repeat_w, RD::SamplerRepeatMode)
	RD_SETGET(lod_bias, float)
	RD_SETGET(use_anisotropy, bool)
	RD_SETGET(anisotropy_max, float)
	RD_SETGET(enable_compare, bool)
	RD_SETGET(compare_op, RD::CompareOperator)
	RD_SETGET(min_lod, float)
	RD_SETGET(max_lod, float)
	RD_SETGET(border_color, RD::SamplerBorderColor)
	RD_SETGET(unnormalized_uvw, bool)

protected:
	static void _bind_methods() {
		RD_BIND_ENUM(RDSamplerState, mag_filter, "RenderingDevice.SamplerFilter", OPTIONS_SAMPLER_FILTER);
		RD_BIND_ENUM(RDSamplerState, min_filter, "RenderingDevice.SamplerFilter", OPTIONS_SAMPLER_FILTER);
		RD_BIND_ENUM(RDSamplerState, mip_filter, "RenderingDevice.SamplerFilter", OPTIONS_SAMPLER_FILTER);
		RD_BIND_ENUM(RDSamplerState, repeat_u, "RenderingDevice.SamplerRepeatMode", OPTIONS_SAMPLER_REPEAT_MODE);
		RD_BIND_ENUM(RDSamplerState, repeat_v, "RenderingDevice.SamplerRepeatMode", OPTIONS_SAMPLER_REPEAT_MODE);
		RD_BIND_ENUM(RDSamplerState, repeat_w, "RenderingDevice.SamplerRepeatMode", OPTIONS_SAMPLER_REPEAT_MODE);
		RD_BIND(RDSamplerState, lod_bias, Variant::FLOAT);
		RD_BIND(RDSamplerState, use_anisotropy, Variant::BOOL);
		RD_BIND(RDSamplerState, anisotropy_max, Variant::FLOAT);
		RD_BIND(RDSamplerState, enable_compare, Variant::BOOL);
		RD_BIND_ENUM(RDSamplerState, compare_op, "RenderingDevice.CompareOperator", OPTIONS_COMPARE_OPERATOR);
		RD_BIND(RDSamplerState, min_lod, Variant::FLOAT);
		RD_BIND(RDSamplerState, max_lod, Variant::FLOAT);
		RD_BIND_ENUM(RDSamplerState, border_color, "RenderingDevice.SamplerBorderColor", OPTIONS_SAMPLER_BORDER_COLOR);
		RD_BIND(RDSamplerState, unnormalized_uvw, Variant::BOOL);
	}
};

class RDVertexAttribute : public RefCounted {
	GDCLASS(RDVertexAttribute, RefCounted)
	friend class RenderingDevice;
	RD::VertexAttribute base;

public:
	RD_SETGET(location, uint32_t)
	RD_SETGET(offset, uint32_t)
	RD_SETGET(format, RD::DataFormat)
	RD_SETGET(stride, uint32_t)
	RD_SETGET(frequency, RD::VertexFrequency)

protected:
	static void _bind_methods() {
		RD_BIND(RDVertexAttribute, location, Variant::INT);
		RD_BIND(RDVertexAttribute, offset, Variant::INT);
		RD_BIND_ENUM(RDVertexAttribute, format, "RenderingDevice.DataFormat", OPTIONS_DATA_FORMAT);
		RD_BIND(RDVertexAttribute, stride, Variant::INT);
		RD_BIND_ENUM(RDVertexAttribute, frequency, "RenderingDevice.VertexFrequency", OPTIONS_VERTEX_FREQUENCY);
	}
};

class RDShaderSource : public RefCounted {
	GDCLASS(RDShaderSource, RefCounted)
	String source[RD::SHADER_STAGE_MAX];
	RD::ShaderLanguage language = RD::SHADER_LANGUAGE_GLSL;

public:
	void set_stage_source(RD::ShaderStage p_stage, const String &p_source) {
		ERR_FAIL_INDEX(p_stage, RD::SHADER_STAGE_MAX);
		source[p_stage] = p_source;
	}

	String get_stage_source(RD::ShaderStage p_stage) const {
		ERR_FAIL_INDEX_V(p_stage, RD::SHADER_STAGE_MAX, String());
		return source[p_stage];
	}

	void set_language(RD::ShaderLanguage p_language) {
		language = p_language;
	}

	RD::ShaderLanguage get_language() const {
		return language;
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_stage_source", "stage", "source"), &RDShaderSource::set_stage_source);
		ClassDB::bind_method(D_METHOD("get_stage_source", "stage"), &RDShaderSource::get_stage_source);

		ClassDB::bind_method(D_METHOD("set_language", "language"), &RDShaderSource::set_language);
		ClassDB::bind_method(D_METHOD("get_language"), &RDShaderSource::get_language);

		ADD_GROUP("Source", "source_");
		ADD_PROPERTYI(PropertyInfo(Variant::STRING, "source_vertex"), "set_stage_source", "get_stage_source", RD::SHADER_STAGE_VERTEX);
		ADD_PROPERTYI(PropertyInfo(Variant::STRING, "source_fragment"), "set_stage_source", "get_stage_source", RD::SHADER_STAGE_FRAGMENT);
		ADD_PROPERTYI(PropertyInfo(Variant::STRING, "source_tesselation_control"), "set_stage_source", "get_stage_source", RD::SHADER_STAGE_TESSELATION_CONTROL);
		ADD_PROPERTYI(PropertyInfo(Variant::STRING, "source_tesselation_evaluation"), "set_stage_source", "get_stage_source", RD::SHADER_STAGE_TESSELATION_EVALUATION);
		ADD_PROPERTYI(PropertyInfo(Variant::STRING, "source_compute"), "set_stage_source", "get_stage_source", RD::SHADER_STAGE_COMPUTE);
		ADD_GROUP("Syntax", "source_");
		ADD_PROPERTY(PropertyInfo::make_enum("language", "RenderingDevice.ShaderLanguage", "GLSL,HLSL"), "set_language", "get_language");
	}
};

class RDShaderSPIRV : public Resource {
	GDCLASS(RDShaderSPIRV, Resource)

	Vector<uint8_t> bytecode[RD::SHADER_STAGE_MAX];
	String compile_error[RD::SHADER_STAGE_MAX];

public:
	void set_stage_bytecode(RD::ShaderStage p_stage, const Vector<uint8_t> &p_bytecode) {
		ERR_FAIL_INDEX(p_stage, RD::SHADER_STAGE_MAX);
		bytecode[p_stage] = p_bytecode;
	}

	Vector<uint8_t> get_stage_bytecode(RD::ShaderStage p_stage) const {
		ERR_FAIL_INDEX_V(p_stage, RD::SHADER_STAGE_MAX, Vector<uint8_t>());
		return bytecode[p_stage];
	}

	Vector<RD::ShaderStageSPIRVData> get_stages() const {
		Vector<RD::ShaderStageSPIRVData> stages;
		for (int i = 0; i < RD::SHADER_STAGE_MAX; i++) {
			if (bytecode[i].size()) {
				RD::ShaderStageSPIRVData stage;
				stage.shader_stage = RD::ShaderStage(i);
				stage.spirv = bytecode[i];
				stages.push_back(stage);
			}
		}
		return stages;
	}

	void set_stage_compile_error(RD::ShaderStage p_stage, const String &p_compile_error) {
		ERR_FAIL_INDEX(p_stage, RD::SHADER_STAGE_MAX);
		compile_error[p_stage] = p_compile_error;
	}

	String get_stage_compile_error(RD::ShaderStage p_stage) const {
		ERR_FAIL_INDEX_V(p_stage, RD::SHADER_STAGE_MAX, String());
		return compile_error[p_stage];
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_stage_bytecode", "stage", "bytecode"), &RDShaderSPIRV::set_stage_bytecode);
		ClassDB::bind_method(D_METHOD("get_stage_bytecode", "stage"), &RDShaderSPIRV::get_stage_bytecode);

		ClassDB::bind_method(D_METHOD("set_stage_compile_error", "stage", "compile_error"), &RDShaderSPIRV::set_stage_compile_error);
		ClassDB::bind_method(D_METHOD("get_stage_compile_error", "stage"), &RDShaderSPIRV::get_stage_compile_error);

		ADD_GROUP("Bytecode", "bytecode_");
		ADD_PROPERTYI(PropertyInfo(Variant::PACKED_BYTE_ARRAY, "bytecode_vertex"), "set_stage_bytecode", "get_stage_bytecode", RD::SHADER_STAGE_VERTEX);
		ADD_PROPERTYI(PropertyInfo(Variant::PACKED_BYTE_ARRAY, "bytecode_fragment"), "set_stage_bytecode", "get_stage_bytecode", RD::SHADER_STAGE_FRAGMENT);
		ADD_PROPERTYI(PropertyInfo(Variant::PACKED_BYTE_ARRAY, "bytecode_tesselation_control"), "set_stage_bytecode", "get_stage_bytecode", RD::SHADER_STAGE_TESSELATION_CONTROL);
		ADD_PROPERTYI(PropertyInfo(Variant::PACKED_BYTE_ARRAY, "bytecode_tesselation_evaluation"), "set_stage_bytecode", "get_stage_bytecode", RD::SHADER_STAGE_TESSELATION_EVALUATION);
		ADD_PROPERTYI(PropertyInfo(Variant::PACKED_BYTE_ARRAY, "bytecode_compute"), "set_stage_bytecode", "get_stage_bytecode", RD::SHADER_STAGE_COMPUTE);
		ADD_GROUP("Compile Error", "compile_error_");
		ADD_PROPERTYI(PropertyInfo(Variant::STRING, "compile_error_vertex"), "set_stage_compile_error", "get_stage_compile_error", RD::SHADER_STAGE_VERTEX);
		ADD_PROPERTYI(PropertyInfo(Variant::STRING, "compile_error_fragment"), "set_stage_compile_error", "get_stage_compile_error", RD::SHADER_STAGE_FRAGMENT);
		ADD_PROPERTYI(PropertyInfo(Variant::STRING, "compile_error_tesselation_control"), "set_stage_compile_error", "get_stage_compile_error", RD::SHADER_STAGE_TESSELATION_CONTROL);
		ADD_PROPERTYI(PropertyInfo(Variant::STRING, "compile_error_tesselation_evaluation"), "set_stage_compile_error", "get_stage_compile_error", RD::SHADER_STAGE_TESSELATION_EVALUATION);
		ADD_PROPERTYI(PropertyInfo(Variant::STRING, "compile_error_compute"), "set_stage_compile_error", "get_stage_compile_error", RD::SHADER_STAGE_COMPUTE);
	}
};

class RDShaderFile : public Resource {
	GDCLASS(RDShaderFile, Resource)

	HashMap<StringName, Ref<RDShaderSPIRV>> versions;
	String base_error;

public:
	void set_bytecode(const Ref<RDShaderSPIRV> &p_bytecode, const StringName &p_version = StringName()) {
		ERR_FAIL_COND(p_bytecode.is_null());
		versions[p_version] = p_bytecode;
		emit_changed();
	}

	Ref<RDShaderSPIRV> get_spirv(const StringName &p_version = StringName()) const {
		ERR_FAIL_COND_V(!versions.has(p_version), Ref<RDShaderSPIRV>());
		return versions[p_version];
	}

	Vector<RD::ShaderStageSPIRVData> get_spirv_stages(const StringName &p_version = StringName()) const {
		ERR_FAIL_COND_V(!versions.has(p_version), Vector<RD::ShaderStageSPIRVData>());
		return versions[p_version]->get_stages();
	}

	TypedArray<StringName> get_version_list() const {
		Vector<StringName> vnames;
		for (const KeyValue<StringName, Ref<RDShaderSPIRV>> &E : versions) {
			vnames.push_back(E.key);
		}
		vnames.sort_custom<StringName::AlphCompare>();
		TypedArray<StringName> ret;
		ret.resize(vnames.size());
		for (int i = 0; i < vnames.size(); i++) {
			ret[i] = vnames[i];
		}
		return ret;
	}

	void set_base_error(const String &p_error) {
		base_error = p_error;
		emit_changed();
	}

	String get_base_error() const {
		return base_error;
	}

	void print_errors(const String &p_file) {
		if (!base_error.is_empty()) {
			ERR_PRINT("Error parsing shader '" + p_file + "':\n\n" + base_error);
		} else {
			for (KeyValue<StringName, Ref<RDShaderSPIRV>> &E : versions) {
				for (int i = 0; i < RD::SHADER_STAGE_MAX; i++) {
					String error = E.value->get_stage_compile_error(RD::ShaderStage(i));
					if (!error.is_empty()) {
						static const char *stage_str[RD::SHADER_STAGE_MAX] = {
							"vertex",
							"fragment",
							"tesselation_control",
							"tesselation_evaluation",
							"compute"
						};

						ERR_PRINT("Error parsing shader '" + p_file + "', version '" + String(E.key) + "', stage '" + stage_str[i] + "':\n\n" + error);
					}
				}
			}
		}
	}

	typedef String (*OpenIncludeFunction)(const String &, void *userdata);
	Error parse_versions_from_text(const String &p_text, const String p_defines = String(), OpenIncludeFunction p_include_func = nullptr, void *p_include_func_userdata = nullptr);

protected:
	Dictionary _get_versions() const {
		TypedArray<StringName> vnames = get_version_list();
		Dictionary ret;
		for (int i = 0; i < vnames.size(); i++) {
			ret[vnames[i]] = versions[vnames[i]];
		}
		return ret;
	}
	void _set_versions(const Dictionary &p_versions) {
		versions.clear();
		List<Variant> keys;
		p_versions.get_key_list(&keys);
		for (const Variant &E : keys) {
			StringName vname = E;
			Ref<RDShaderSPIRV> bc = p_versions[E];
			ERR_CONTINUE(bc.is_null());
			versions[vname] = bc;
		}

		emit_changed();
	}

	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_bytecode", "bytecode", "version"), &RDShaderFile::set_bytecode, DEFVAL(StringName()));
		ClassDB::bind_method(D_METHOD("get_spirv", "version"), &RDShaderFile::get_spirv, DEFVAL(StringName()));
		ClassDB::bind_method(D_METHOD("get_version_list"), &RDShaderFile::get_version_list);

		ClassDB::bind_method(D_METHOD("set_base_error", "error"), &RDShaderFile::set_base_error);
		ClassDB::bind_method(D_METHOD("get_base_error"), &RDShaderFile::get_base_error);

		ClassDB::bind_method(D_METHOD("_set_versions", "versions"), &RDShaderFile::_set_versions);
		ClassDB::bind_method(D_METHOD("_get_versions"), &RDShaderFile::_get_versions);

		ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "_versions", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL), "_set_versions", "_get_versions");
		ADD_PROPERTY(PropertyInfo(Variant::STRING, "base_error"), "set_base_error", "get_base_error");
	}
};

class RDUniform : public RefCounted {
	GDCLASS(RDUniform, RefCounted)
	friend class RenderingDevice;
	friend class UniformSetCacheRD;
	RD::Uniform base;

public:
	RD_SETGET(uniform_type, RD::UniformType)
	RD_SETGET(binding, int32_t)

	void add_id(const RID &p_id) { base.append_id(p_id); }
	void clear_ids() { base.clear_ids(); }
	TypedArray<RID> get_ids() const {
		TypedArray<RID> ids;
		for (uint32_t i = 0; i < base.get_id_count(); i++) {
			ids.push_back(base.get_id(i));
		}
		return ids;
	}

protected:
	void _set_ids(const TypedArray<RID> &p_ids) {
		base.clear_ids();
		for (int i = 0; i < p_ids.size(); i++) {
			RID id = p_ids[i];
			ERR_FAIL_COND(id.is_null());
			base.append_id(id);
		}
	}
	static void _bind_methods() {
		RD_BIND_ENUM(RDUniform, uniform_type, "RenderingDevice.UniformType", OPTIONS_UNIFORM_TYPE);
		RD_BIND(RDUniform, binding, Variant::INT);

		ClassDB::bind_method(D_METHOD("add_id", "id"), &RDUniform::add_id);
		ClassDB::bind_method(D_METHOD("clear_ids"), &RDUniform::clear_ids);
		ClassDB::bind_method(D_METHOD("_set_ids", "ids"), &RDUniform::_set_ids);
		ClassDB::bind_method(D_METHOD("get_ids"), &RDUniform::get_ids);

		ADD_PROPERTY(PropertyInfo::make_typed_array("_ids", "RID", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_INTERNAL), "_set_ids", "get_ids");
	}
};

class RDPipelineSpecializationConstant : public RefCounted {
	GDCLASS(RDPipelineSpecializationConstant, RefCounted)
	friend class RenderingDevice;

	Variant value = false;
	uint32_t constant_id = 0;

public:
	void set_value(const Variant &p_value) {
		ERR_FAIL_COND(p_value.get_type() != Variant::BOOL && p_value.get_type() != Variant::INT && p_value.get_type() != Variant::FLOAT);
		value = p_value;
	}
	Variant get_value() const { return value; }

	void set_constant_id(uint32_t p_id) {
		constant_id = p_id;
	}
	uint32_t get_constant_id() const {
		return constant_id;
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_value", "value"), &RDPipelineSpecializationConstant::set_value);
		ClassDB::bind_method(D_METHOD("get_value"), &RDPipelineSpecializationConstant::get_value);

		ClassDB::bind_method(D_METHOD("set_constant_id", "constant_id"), &RDPipelineSpecializationConstant::set_constant_id);
		ClassDB::bind_method(D_METHOD("get_constant_id"), &RDPipelineSpecializationConstant::get_constant_id);

		ADD_PROPERTY(PropertyInfo(Variant::NIL, "value", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NIL_IS_VARIANT), "set_value", "get_value");
		ADD_PROPERTY(PropertyInfo(Variant::INT, "constant_id", PROPERTY_HINT_RANGE, "0,65535,0"), "set_constant_id", "get_constant_id");
	}
};

class RDPipelineRasterizationState : public RefCounted {
	GDCLASS(RDPipelineRasterizationState, RefCounted)
	friend class RenderingDevice;

	RD::PipelineRasterizationState base;

public:
	RD_SETGET(enable_depth_clamp, bool)
	RD_SETGET(discard_primitives, bool)
	RD_SETGET(wireframe, bool)
	RD_SETGET(cull_mode, RD::PolygonCullMode)
	RD_SETGET(front_face, RD::PolygonFrontFace)
	RD_SETGET(depth_bias_enabled, bool)
	RD_SETGET(depth_bias_constant_factor, float)
	RD_SETGET(depth_bias_clamp, float)
	RD_SETGET(depth_bias_slope_factor, float)
	RD_SETGET(line_width, float)
	RD_SETGET(patch_control_points, uint32_t)

protected:
	static void _bind_methods() {
		RD_BIND(RDPipelineRasterizationState, enable_depth_clamp, Variant::BOOL);
		RD_BIND(RDPipelineRasterizationState, discard_primitives, Variant::BOOL);
		RD_BIND(RDPipelineRasterizationState, wireframe, Variant::BOOL);
		RD_BIND_ENUM(RDPipelineRasterizationState, cull_mode, "RenderingDevice.PolygonCullMode", OPTIONS_POLYGON_CULL_MODE);
		RD_BIND_ENUM(RDPipelineRasterizationState, front_face, "RenderingDevice.PolygonFrontFace", OPTIONS_POLYGON_FRONT_FACE);
		RD_BIND(RDPipelineRasterizationState, depth_bias_enabled, Variant::BOOL);
		RD_BIND(RDPipelineRasterizationState, depth_bias_constant_factor, Variant::FLOAT);
		RD_BIND(RDPipelineRasterizationState, depth_bias_clamp, Variant::FLOAT);
		RD_BIND(RDPipelineRasterizationState, depth_bias_slope_factor, Variant::FLOAT);
		RD_BIND(RDPipelineRasterizationState, line_width, Variant::FLOAT);
		RD_BIND(RDPipelineRasterizationState, patch_control_points, Variant::INT);
	}
};

class RDPipelineMultisampleState : public RefCounted {
	GDCLASS(RDPipelineMultisampleState, RefCounted)
	friend class RenderingDevice;

	RD::PipelineMultisampleState base;
	TypedArray<int64_t> sample_masks;

public:
	RD_SETGET(sample_count, RD::TextureSamples)
	RD_SETGET(enable_sample_shading, bool)
	RD_SETGET(min_sample_shading, float)
	RD_SETGET(enable_alpha_to_coverage, bool)
	RD_SETGET(enable_alpha_to_one, bool)

	void set_sample_masks(const TypedArray<int64_t> &p_masks) { sample_masks = p_masks; }
	TypedArray<int64_t> get_sample_masks() const { return sample_masks; }

protected:
	static void _bind_methods() {
		RD_BIND_ENUM(RDPipelineMultisampleState, sample_count, "RenderingDevice.TextureSamples", OPTIONS_TEXTURE_SAMPLES);
		RD_BIND(RDPipelineMultisampleState, enable_sample_shading, Variant::BOOL);
		RD_BIND(RDPipelineMultisampleState, min_sample_shading, Variant::FLOAT);
		RD_BIND(RDPipelineMultisampleState, enable_alpha_to_coverage, Variant::BOOL);
		RD_BIND(RDPipelineMultisampleState, enable_alpha_to_one, Variant::BOOL);

		ClassDB::bind_method(D_METHOD("set_sample_masks", "masks"), &RDPipelineMultisampleState::set_sample_masks);
		ClassDB::bind_method(D_METHOD("get_sample_masks"), &RDPipelineMultisampleState::get_sample_masks);
		ADD_PROPERTY(PropertyInfo::make_typed_array("sample_masks", "int"), "set_sample_masks", "get_sample_masks");
	}
};

class RDPipelineDepthStencilState : public RefCounted {
	GDCLASS(RDPipelineDepthStencilState, RefCounted)
	friend class RenderingDevice;

	RD::PipelineDepthStencilState base;

public:
	RD_SETGET(enable_depth_test, bool)
	RD_SETGET(enable_depth_write, bool)
	RD_SETGET(depth_compare_operator, RD::CompareOperator)
	RD_SETGET(enable_depth_range, bool)
	RD_SETGET(depth_range_min, float)
	RD_SETGET(depth_range_max, float)
	RD_SETGET(enable_stencil, bool)

	RD_SETGET_SUB(front_op, fail, RD::StencilOperation)
	RD_SETGET_SUB(front_op, pass, RD::StencilOperation)
	RD_SETGET_SUB(front_op, depth_fail, RD::StencilOperation)
	RD_SETGET_SUB(front_op, compare, RD::CompareOperator)
	RD_SETGET_SUB(front_op, compare_mask, uint32_t)
	RD_SETGET_SUB(front_op, write_mask, uint32_t)
	RD_SETGET_SUB(front_op, reference, uint32_t)

	RD_SETGET_SUB(back_op, fail, RD::StencilOperation)
	RD_SETGET_SUB(back_op, pass, RD::StencilOperation)
	RD_SETGET_SUB(back_op, depth_fail, RD::StencilOperation)
	RD_SETGET_SUB(back_op, compare, RD::CompareOperator)
	RD_SETGET_SUB(back_op, compare_mask, uint32_t)
	RD_SETGET_SUB(back_op, write_mask, uint32_t)
	RD_SETGET_SUB(back_op, reference, uint32_t)

protected:
	static void _bind_methods() {
		RD_BIND(RDPipelineDepthStencilState, enable_depth_test, Variant::BOOL);
		RD_BIND(RDPipelineDepthStencilState, enable_depth_write, Variant::BOOL);
		RD_BIND_ENUM(RDPipelineDepthStencilState, depth_compare_operator, "RenderingDevice.CompareOperator", OPTIONS_COMPARE_OPERATOR);
		RD_BIND(RDPipelineDepthStencilState, enable_depth_range, Variant::BOOL);
		RD_BIND(RDPipelineDepthStencilState, depth_range_min, Variant::FLOAT);
		RD_BIND(RDPipelineDepthStencilState, depth_range_max, Variant::FLOAT);
		RD_BIND(RDPipelineDepthStencilState, enable_stencil, Variant::BOOL);

		RD_BIND_SUB_ENUM(RDPipelineDepthStencilState, front_op, fail, "RenderingDevice.StencilOperation", OPTIONS_STENCIL_OPERATION);
		RD_BIND_SUB_ENUM(RDPipelineDepthStencilState, front_op, pass, "RenderingDevice.StencilOperation", OPTIONS_STENCIL_OPERATION);
		RD_BIND_SUB_ENUM(RDPipelineDepthStencilState, front_op, depth_fail, "RenderingDevice.StencilOperation", OPTIONS_STENCIL_OPERATION);
		RD_BIND_SUB_ENUM(RDPipelineDepthStencilState, front_op, compare, "RenderingDevice.CompareOperator", OPTIONS_COMPARE_OPERATOR);
		RD_BIND_SUB(RDPipelineDepthStencilState, front_op, compare_mask, Variant::INT);
		RD_BIND_SUB(RDPipelineDepthStencilState, front_op, write_mask, Variant::INT);
		RD_BIND_SUB(RDPipelineDepthStencilState, front_op, reference, Variant::INT);

		RD_BIND_SUB_ENUM(RDPipelineDepthStencilState, back_op, fail, "RenderingDevice.StencilOperation", OPTIONS_STENCIL_OPERATION);
		RD_BIND_SUB_ENUM(RDPipelineDepthStencilState, back_op, pass, "RenderingDevice.StencilOperation", OPTIONS_STENCIL_OPERATION);
		RD_BIND_SUB_ENUM(RDPipelineDepthStencilState, back_op, depth_fail, "RenderingDevice.StencilOperation", OPTIONS_STENCIL_OPERATION);
		RD_BIND_SUB_ENUM(RDPipelineDepthStencilState, back_op, compare, "RenderingDevice.CompareOperator", OPTIONS_COMPARE_OPERATOR);
		RD_BIND_SUB(RDPipelineDepthStencilState, back_op, compare_mask, Variant::INT);
		RD_BIND_SUB(RDPipelineDepthStencilState, back_op, write_mask, Variant::INT);
		RD_BIND_SUB(RDPipelineDepthStencilState, back_op, reference, Variant::INT);
	}
};

class RDPipelineColorBlendStateAttachment : public RefCounted {
	GDCLASS(RDPipelineColorBlendStateAttachment, RefCounted)
	friend class RenderingDevice;
	RD::PipelineColorBlendState::Attachment base;

public:
	RD_SETGET(enable_blend, bool)
	RD_SETGET(src_color_blend_factor, RD::BlendFactor)
	RD_SETGET(dst_color_blend_factor, RD::BlendFactor)
	RD_SETGET(color_blend_op, RD::BlendOperation)
	RD_SETGET(src_alpha_blend_factor, RD::BlendFactor)
	RD_SETGET(dst_alpha_blend_factor, RD::BlendFactor)
	RD_SETGET(alpha_blend_op, RD::BlendOperation)
	RD_SETGET(write_r, bool)
	RD_SETGET(write_g, bool)
	RD_SETGET(write_b, bool)
	RD_SETGET(write_a, bool)

	void set_as_mix() {
		base = RD::PipelineColorBlendState::Attachment();
		base.enable_blend = true;
		base.src_color_blend_factor = RD::BLEND_FACTOR_SRC_ALPHA;
		base.dst_color_blend_factor = RD::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		base.src_alpha_blend_factor = RD::BLEND_FACTOR_SRC_ALPHA;
		base.dst_alpha_blend_factor = RD::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	}

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_as_mix"), &RDPipelineColorBlendStateAttachment::set_as_mix);

		RD_BIND(RDPipelineColorBlendStateAttachment, enable_blend, Variant::BOOL);
		RD_BIND_ENUM(RDPipelineColorBlendStateAttachment, src_color_blend_factor, "RenderingDevice.BlendFactor", OPTIONS_BLEND_FACTOR);
		RD_BIND_ENUM(RDPipelineColorBlendStateAttachment, dst_color_blend_factor, "RenderingDevice.BlendFactor", OPTIONS_BLEND_FACTOR);
		RD_BIND_ENUM(RDPipelineColorBlendStateAttachment, color_blend_op, "RenderingDevice.BlendOperation", OPTIONS_BLEND_OPERATION);
		RD_BIND_ENUM(RDPipelineColorBlendStateAttachment, src_alpha_blend_factor, "RenderingDevice.BlendFactor", OPTIONS_BLEND_FACTOR);
		RD_BIND_ENUM(RDPipelineColorBlendStateAttachment, dst_alpha_blend_factor, "RenderingDevice.BlendFactor", OPTIONS_BLEND_FACTOR);
		RD_BIND_ENUM(RDPipelineColorBlendStateAttachment, alpha_blend_op, "RenderingDevice.BlendOperation", OPTIONS_BLEND_OPERATION);
		RD_BIND(RDPipelineColorBlendStateAttachment, write_r, Variant::BOOL);
		RD_BIND(RDPipelineColorBlendStateAttachment, write_g, Variant::BOOL);
		RD_BIND(RDPipelineColorBlendStateAttachment, write_b, Variant::BOOL);
		RD_BIND(RDPipelineColorBlendStateAttachment, write_a, Variant::BOOL);
	}
};

class RDPipelineColorBlendState : public RefCounted {
	GDCLASS(RDPipelineColorBlendState, RefCounted)
	friend class RenderingDevice;
	RD::PipelineColorBlendState base;

	TypedArray<RDPipelineColorBlendStateAttachment> attachments;

public:
	RD_SETGET(enable_logic_op, bool)
	RD_SETGET(logic_op, RD::LogicOperation)
	RD_SETGET(blend_constant, Color)

	void set_attachments(const TypedArray<RDPipelineColorBlendStateAttachment> &p_attachments) {
		attachments = p_attachments;
	}

	TypedArray<RDPipelineColorBlendStateAttachment> get_attachments() const {
		return attachments;
	}

protected:
	static void _bind_methods() {
		RD_BIND(RDPipelineColorBlendState, enable_logic_op, Variant::BOOL);
		RD_BIND_ENUM(RDPipelineColorBlendState, logic_op, "RenderingDevice.LogicOperation", OPTIONS_LOGIC_OPERATION);
		RD_BIND(RDPipelineColorBlendState, blend_constant, Variant::COLOR);

		ClassDB::bind_method(D_METHOD("set_attachments", "attachments"), &RDPipelineColorBlendState::set_attachments);
		ClassDB::bind_method(D_METHOD("get_attachments"), &RDPipelineColorBlendState::get_attachments);
		ADD_PROPERTY(PropertyInfo::make_typed_array("attachments", "RDPipelineColorBlendStateAttachment"), "set_attachments", "get_attachments");
	}
};

#endif // RENDERING_DEVICE_BINDS_H
