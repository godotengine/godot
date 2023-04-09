/**************************************************************************/
/*  rendering_device.h                                                    */
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

#ifndef RENDERING_DEVICE_H
#define RENDERING_DEVICE_H

#include "core/object/class_db.h"
#include "core/variant/typed_array.h"
#include "servers/display_server.h"

class RDTextureFormat;
class RDTextureView;
class RDAttachmentFormat;
class RDSamplerState;
class RDVertexAttribute;
class RDShaderSource;
class RDShaderSPIRV;
class RDUniform;
class RDPipelineRasterizationState;
class RDPipelineMultisampleState;
class RDPipelineDepthStencilState;
class RDPipelineColorBlendState;
class RDFramebufferPass;
class RDPipelineSpecializationConstant;

class RenderingDevice : public Object {
	GDCLASS(RenderingDevice, Object)
public:
	enum DeviceFamily {
		DEVICE_UNKNOWN,
		DEVICE_OPENGL,
		DEVICE_VULKAN,
		DEVICE_DIRECTX
	};

	// This enum matches VkPhysicalDeviceType (except for `DEVICE_TYPE_MAX`).
	// Unlike VkPhysicalDeviceType, DeviceType is exposed to the scripting API.
	enum DeviceType {
		DEVICE_TYPE_OTHER,
		DEVICE_TYPE_INTEGRATED_GPU,
		DEVICE_TYPE_DISCRETE_GPU,
		DEVICE_TYPE_VIRTUAL_GPU,
		DEVICE_TYPE_CPU,
		DEVICE_TYPE_MAX,
	};

	enum DriverResource {
		DRIVER_RESOURCE_VULKAN_DEVICE = 0,
		DRIVER_RESOURCE_VULKAN_PHYSICAL_DEVICE,
		DRIVER_RESOURCE_VULKAN_INSTANCE,
		DRIVER_RESOURCE_VULKAN_QUEUE,
		DRIVER_RESOURCE_VULKAN_QUEUE_FAMILY_INDEX,
		DRIVER_RESOURCE_VULKAN_IMAGE,
		DRIVER_RESOURCE_VULKAN_IMAGE_VIEW,
		DRIVER_RESOURCE_VULKAN_IMAGE_NATIVE_TEXTURE_FORMAT,
		DRIVER_RESOURCE_VULKAN_SAMPLER,
		DRIVER_RESOURCE_VULKAN_DESCRIPTOR_SET,
		DRIVER_RESOURCE_VULKAN_BUFFER,
		DRIVER_RESOURCE_VULKAN_COMPUTE_PIPELINE,
		DRIVER_RESOURCE_VULKAN_RENDER_PIPELINE,
		//next driver continue enum from 1000 to keep order
	};

	enum ShaderStage {
		SHADER_STAGE_VERTEX,
		SHADER_STAGE_FRAGMENT,
		SHADER_STAGE_TESSELATION_CONTROL,
		SHADER_STAGE_TESSELATION_EVALUATION,
		SHADER_STAGE_COMPUTE,
		SHADER_STAGE_MAX,
		SHADER_STAGE_VERTEX_BIT = (1 << SHADER_STAGE_VERTEX),
		SHADER_STAGE_FRAGMENT_BIT = (1 << SHADER_STAGE_FRAGMENT),
		SHADER_STAGE_TESSELATION_CONTROL_BIT = (1 << SHADER_STAGE_TESSELATION_CONTROL),
		SHADER_STAGE_TESSELATION_EVALUATION_BIT = (1 << SHADER_STAGE_TESSELATION_EVALUATION),
		SHADER_STAGE_COMPUTE_BIT = (1 << SHADER_STAGE_COMPUTE),
	};

	enum ShaderLanguage {
		SHADER_LANGUAGE_GLSL,
		SHADER_LANGUAGE_HLSL
	};

	enum SubgroupOperations {
		SUBGROUP_BASIC_BIT = 1,
		SUBGROUP_VOTE_BIT = 2,
		SUBGROUP_ARITHMETIC_BIT = 4,
		SUBGROUP_BALLOT_BIT = 8,
		SUBGROUP_SHUFFLE_BIT = 16,
		SUBGROUP_SHUFFLE_RELATIVE_BIT = 32,
		SUBGROUP_CLUSTERED_BIT = 64,
		SUBGROUP_QUAD_BIT = 128,
	};

	struct Capabilities {
		// main device info
		DeviceFamily device_family = DEVICE_UNKNOWN;
		uint32_t version_major = 1.0;
		uint32_t version_minor = 0.0;
	};

	typedef String (*ShaderSPIRVGetCacheKeyFunction)(const RenderingDevice *p_render_device);
	typedef Vector<uint8_t> (*ShaderCompileToSPIRVFunction)(ShaderStage p_stage, const String &p_source_code, ShaderLanguage p_language, String *r_error, const RenderingDevice *p_render_device);
	typedef Vector<uint8_t> (*ShaderCacheFunction)(ShaderStage p_stage, const String &p_source_code, ShaderLanguage p_language);

	typedef void (*InvalidationCallback)(void *);

private:
	static ShaderCompileToSPIRVFunction compile_to_spirv_function;
	static ShaderCacheFunction cache_function;
	static ShaderSPIRVGetCacheKeyFunction get_spirv_cache_key_function;

	static RenderingDevice *singleton;

protected:
	static void _bind_methods();

	Capabilities device_capabilities;

public:
	//base numeric ID for all types
	enum {
		INVALID_ID = -1,
		INVALID_FORMAT_ID = -1
	};

	/*****************/
	/**** GENERIC ****/
	/*****************/

	enum CompareOperator {
		COMPARE_OP_NEVER,
		COMPARE_OP_LESS,
		COMPARE_OP_EQUAL,
		COMPARE_OP_LESS_OR_EQUAL,
		COMPARE_OP_GREATER,
		COMPARE_OP_NOT_EQUAL,
		COMPARE_OP_GREATER_OR_EQUAL,
		COMPARE_OP_ALWAYS,
		COMPARE_OP_MAX //not an actual operator, just the amount of operators :D
	};

	enum DataFormat {
		DATA_FORMAT_R4G4_UNORM_PACK8,
		DATA_FORMAT_R4G4B4A4_UNORM_PACK16,
		DATA_FORMAT_B4G4R4A4_UNORM_PACK16,
		DATA_FORMAT_R5G6B5_UNORM_PACK16,
		DATA_FORMAT_B5G6R5_UNORM_PACK16,
		DATA_FORMAT_R5G5B5A1_UNORM_PACK16,
		DATA_FORMAT_B5G5R5A1_UNORM_PACK16,
		DATA_FORMAT_A1R5G5B5_UNORM_PACK16,
		DATA_FORMAT_R8_UNORM,
		DATA_FORMAT_R8_SNORM,
		DATA_FORMAT_R8_USCALED,
		DATA_FORMAT_R8_SSCALED,
		DATA_FORMAT_R8_UINT,
		DATA_FORMAT_R8_SINT,
		DATA_FORMAT_R8_SRGB,
		DATA_FORMAT_R8G8_UNORM,
		DATA_FORMAT_R8G8_SNORM,
		DATA_FORMAT_R8G8_USCALED,
		DATA_FORMAT_R8G8_SSCALED,
		DATA_FORMAT_R8G8_UINT,
		DATA_FORMAT_R8G8_SINT,
		DATA_FORMAT_R8G8_SRGB,
		DATA_FORMAT_R8G8B8_UNORM,
		DATA_FORMAT_R8G8B8_SNORM,
		DATA_FORMAT_R8G8B8_USCALED,
		DATA_FORMAT_R8G8B8_SSCALED,
		DATA_FORMAT_R8G8B8_UINT,
		DATA_FORMAT_R8G8B8_SINT,
		DATA_FORMAT_R8G8B8_SRGB,
		DATA_FORMAT_B8G8R8_UNORM,
		DATA_FORMAT_B8G8R8_SNORM,
		DATA_FORMAT_B8G8R8_USCALED,
		DATA_FORMAT_B8G8R8_SSCALED,
		DATA_FORMAT_B8G8R8_UINT,
		DATA_FORMAT_B8G8R8_SINT,
		DATA_FORMAT_B8G8R8_SRGB,
		DATA_FORMAT_R8G8B8A8_UNORM,
		DATA_FORMAT_R8G8B8A8_SNORM,
		DATA_FORMAT_R8G8B8A8_USCALED,
		DATA_FORMAT_R8G8B8A8_SSCALED,
		DATA_FORMAT_R8G8B8A8_UINT,
		DATA_FORMAT_R8G8B8A8_SINT,
		DATA_FORMAT_R8G8B8A8_SRGB,
		DATA_FORMAT_B8G8R8A8_UNORM,
		DATA_FORMAT_B8G8R8A8_SNORM,
		DATA_FORMAT_B8G8R8A8_USCALED,
		DATA_FORMAT_B8G8R8A8_SSCALED,
		DATA_FORMAT_B8G8R8A8_UINT,
		DATA_FORMAT_B8G8R8A8_SINT,
		DATA_FORMAT_B8G8R8A8_SRGB,
		DATA_FORMAT_A8B8G8R8_UNORM_PACK32,
		DATA_FORMAT_A8B8G8R8_SNORM_PACK32,
		DATA_FORMAT_A8B8G8R8_USCALED_PACK32,
		DATA_FORMAT_A8B8G8R8_SSCALED_PACK32,
		DATA_FORMAT_A8B8G8R8_UINT_PACK32,
		DATA_FORMAT_A8B8G8R8_SINT_PACK32,
		DATA_FORMAT_A8B8G8R8_SRGB_PACK32,
		DATA_FORMAT_A2R10G10B10_UNORM_PACK32,
		DATA_FORMAT_A2R10G10B10_SNORM_PACK32,
		DATA_FORMAT_A2R10G10B10_USCALED_PACK32,
		DATA_FORMAT_A2R10G10B10_SSCALED_PACK32,
		DATA_FORMAT_A2R10G10B10_UINT_PACK32,
		DATA_FORMAT_A2R10G10B10_SINT_PACK32,
		DATA_FORMAT_A2B10G10R10_UNORM_PACK32,
		DATA_FORMAT_A2B10G10R10_SNORM_PACK32,
		DATA_FORMAT_A2B10G10R10_USCALED_PACK32,
		DATA_FORMAT_A2B10G10R10_SSCALED_PACK32,
		DATA_FORMAT_A2B10G10R10_UINT_PACK32,
		DATA_FORMAT_A2B10G10R10_SINT_PACK32,
		DATA_FORMAT_R16_UNORM,
		DATA_FORMAT_R16_SNORM,
		DATA_FORMAT_R16_USCALED,
		DATA_FORMAT_R16_SSCALED,
		DATA_FORMAT_R16_UINT,
		DATA_FORMAT_R16_SINT,
		DATA_FORMAT_R16_SFLOAT,
		DATA_FORMAT_R16G16_UNORM,
		DATA_FORMAT_R16G16_SNORM,
		DATA_FORMAT_R16G16_USCALED,
		DATA_FORMAT_R16G16_SSCALED,
		DATA_FORMAT_R16G16_UINT,
		DATA_FORMAT_R16G16_SINT,
		DATA_FORMAT_R16G16_SFLOAT,
		DATA_FORMAT_R16G16B16_UNORM,
		DATA_FORMAT_R16G16B16_SNORM,
		DATA_FORMAT_R16G16B16_USCALED,
		DATA_FORMAT_R16G16B16_SSCALED,
		DATA_FORMAT_R16G16B16_UINT,
		DATA_FORMAT_R16G16B16_SINT,
		DATA_FORMAT_R16G16B16_SFLOAT,
		DATA_FORMAT_R16G16B16A16_UNORM,
		DATA_FORMAT_R16G16B16A16_SNORM,
		DATA_FORMAT_R16G16B16A16_USCALED,
		DATA_FORMAT_R16G16B16A16_SSCALED,
		DATA_FORMAT_R16G16B16A16_UINT,
		DATA_FORMAT_R16G16B16A16_SINT,
		DATA_FORMAT_R16G16B16A16_SFLOAT,
		DATA_FORMAT_R32_UINT,
		DATA_FORMAT_R32_SINT,
		DATA_FORMAT_R32_SFLOAT,
		DATA_FORMAT_R32G32_UINT,
		DATA_FORMAT_R32G32_SINT,
		DATA_FORMAT_R32G32_SFLOAT,
		DATA_FORMAT_R32G32B32_UINT,
		DATA_FORMAT_R32G32B32_SINT,
		DATA_FORMAT_R32G32B32_SFLOAT,
		DATA_FORMAT_R32G32B32A32_UINT,
		DATA_FORMAT_R32G32B32A32_SINT,
		DATA_FORMAT_R32G32B32A32_SFLOAT,
		DATA_FORMAT_R64_UINT,
		DATA_FORMAT_R64_SINT,
		DATA_FORMAT_R64_SFLOAT,
		DATA_FORMAT_R64G64_UINT,
		DATA_FORMAT_R64G64_SINT,
		DATA_FORMAT_R64G64_SFLOAT,
		DATA_FORMAT_R64G64B64_UINT,
		DATA_FORMAT_R64G64B64_SINT,
		DATA_FORMAT_R64G64B64_SFLOAT,
		DATA_FORMAT_R64G64B64A64_UINT,
		DATA_FORMAT_R64G64B64A64_SINT,
		DATA_FORMAT_R64G64B64A64_SFLOAT,
		DATA_FORMAT_B10G11R11_UFLOAT_PACK32,
		DATA_FORMAT_E5B9G9R9_UFLOAT_PACK32,
		DATA_FORMAT_D16_UNORM,
		DATA_FORMAT_X8_D24_UNORM_PACK32,
		DATA_FORMAT_D32_SFLOAT,
		DATA_FORMAT_S8_UINT,
		DATA_FORMAT_D16_UNORM_S8_UINT,
		DATA_FORMAT_D24_UNORM_S8_UINT,
		DATA_FORMAT_D32_SFLOAT_S8_UINT,
		DATA_FORMAT_BC1_RGB_UNORM_BLOCK,
		DATA_FORMAT_BC1_RGB_SRGB_BLOCK,
		DATA_FORMAT_BC1_RGBA_UNORM_BLOCK,
		DATA_FORMAT_BC1_RGBA_SRGB_BLOCK,
		DATA_FORMAT_BC2_UNORM_BLOCK,
		DATA_FORMAT_BC2_SRGB_BLOCK,
		DATA_FORMAT_BC3_UNORM_BLOCK,
		DATA_FORMAT_BC3_SRGB_BLOCK,
		DATA_FORMAT_BC4_UNORM_BLOCK,
		DATA_FORMAT_BC4_SNORM_BLOCK,
		DATA_FORMAT_BC5_UNORM_BLOCK,
		DATA_FORMAT_BC5_SNORM_BLOCK,
		DATA_FORMAT_BC6H_UFLOAT_BLOCK,
		DATA_FORMAT_BC6H_SFLOAT_BLOCK,
		DATA_FORMAT_BC7_UNORM_BLOCK,
		DATA_FORMAT_BC7_SRGB_BLOCK,
		DATA_FORMAT_ETC2_R8G8B8_UNORM_BLOCK,
		DATA_FORMAT_ETC2_R8G8B8_SRGB_BLOCK,
		DATA_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK,
		DATA_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK,
		DATA_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK,
		DATA_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK,
		DATA_FORMAT_EAC_R11_UNORM_BLOCK,
		DATA_FORMAT_EAC_R11_SNORM_BLOCK,
		DATA_FORMAT_EAC_R11G11_UNORM_BLOCK,
		DATA_FORMAT_EAC_R11G11_SNORM_BLOCK,
		DATA_FORMAT_ASTC_4x4_UNORM_BLOCK,
		DATA_FORMAT_ASTC_4x4_SRGB_BLOCK,
		DATA_FORMAT_ASTC_5x4_UNORM_BLOCK,
		DATA_FORMAT_ASTC_5x4_SRGB_BLOCK,
		DATA_FORMAT_ASTC_5x5_UNORM_BLOCK,
		DATA_FORMAT_ASTC_5x5_SRGB_BLOCK,
		DATA_FORMAT_ASTC_6x5_UNORM_BLOCK,
		DATA_FORMAT_ASTC_6x5_SRGB_BLOCK,
		DATA_FORMAT_ASTC_6x6_UNORM_BLOCK,
		DATA_FORMAT_ASTC_6x6_SRGB_BLOCK,
		DATA_FORMAT_ASTC_8x5_UNORM_BLOCK,
		DATA_FORMAT_ASTC_8x5_SRGB_BLOCK,
		DATA_FORMAT_ASTC_8x6_UNORM_BLOCK,
		DATA_FORMAT_ASTC_8x6_SRGB_BLOCK,
		DATA_FORMAT_ASTC_8x8_UNORM_BLOCK,
		DATA_FORMAT_ASTC_8x8_SRGB_BLOCK,
		DATA_FORMAT_ASTC_10x5_UNORM_BLOCK,
		DATA_FORMAT_ASTC_10x5_SRGB_BLOCK,
		DATA_FORMAT_ASTC_10x6_UNORM_BLOCK,
		DATA_FORMAT_ASTC_10x6_SRGB_BLOCK,
		DATA_FORMAT_ASTC_10x8_UNORM_BLOCK,
		DATA_FORMAT_ASTC_10x8_SRGB_BLOCK,
		DATA_FORMAT_ASTC_10x10_UNORM_BLOCK,
		DATA_FORMAT_ASTC_10x10_SRGB_BLOCK,
		DATA_FORMAT_ASTC_12x10_UNORM_BLOCK,
		DATA_FORMAT_ASTC_12x10_SRGB_BLOCK,
		DATA_FORMAT_ASTC_12x12_UNORM_BLOCK,
		DATA_FORMAT_ASTC_12x12_SRGB_BLOCK,
		DATA_FORMAT_G8B8G8R8_422_UNORM,
		DATA_FORMAT_B8G8R8G8_422_UNORM,
		DATA_FORMAT_G8_B8_R8_3PLANE_420_UNORM,
		DATA_FORMAT_G8_B8R8_2PLANE_420_UNORM,
		DATA_FORMAT_G8_B8_R8_3PLANE_422_UNORM,
		DATA_FORMAT_G8_B8R8_2PLANE_422_UNORM,
		DATA_FORMAT_G8_B8_R8_3PLANE_444_UNORM,
		DATA_FORMAT_R10X6_UNORM_PACK16,
		DATA_FORMAT_R10X6G10X6_UNORM_2PACK16,
		DATA_FORMAT_R10X6G10X6B10X6A10X6_UNORM_4PACK16,
		DATA_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16,
		DATA_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16,
		DATA_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16,
		DATA_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16,
		DATA_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16,
		DATA_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16,
		DATA_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16,
		DATA_FORMAT_R12X4_UNORM_PACK16,
		DATA_FORMAT_R12X4G12X4_UNORM_2PACK16,
		DATA_FORMAT_R12X4G12X4B12X4A12X4_UNORM_4PACK16,
		DATA_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16,
		DATA_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16,
		DATA_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16,
		DATA_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16,
		DATA_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16,
		DATA_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16,
		DATA_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16,
		DATA_FORMAT_G16B16G16R16_422_UNORM,
		DATA_FORMAT_B16G16R16G16_422_UNORM,
		DATA_FORMAT_G16_B16_R16_3PLANE_420_UNORM,
		DATA_FORMAT_G16_B16R16_2PLANE_420_UNORM,
		DATA_FORMAT_G16_B16_R16_3PLANE_422_UNORM,
		DATA_FORMAT_G16_B16R16_2PLANE_422_UNORM,
		DATA_FORMAT_G16_B16_R16_3PLANE_444_UNORM,
		DATA_FORMAT_MAX
	};

	/*****************/
	/**** BARRIER ****/
	/*****************/

	enum BarrierMask {
		BARRIER_MASK_RASTER = 1,
		BARRIER_MASK_COMPUTE = 2,
		BARRIER_MASK_TRANSFER = 4,
		BARRIER_MASK_ALL_BARRIERS = BARRIER_MASK_RASTER | BARRIER_MASK_COMPUTE | BARRIER_MASK_TRANSFER, // 7
		BARRIER_MASK_NO_BARRIER = 8,
	};

	/*****************/
	/**** TEXTURE ****/
	/*****************/

	enum TextureType {
		TEXTURE_TYPE_1D,
		TEXTURE_TYPE_2D,
		TEXTURE_TYPE_3D,
		TEXTURE_TYPE_CUBE,
		TEXTURE_TYPE_1D_ARRAY,
		TEXTURE_TYPE_2D_ARRAY,
		TEXTURE_TYPE_CUBE_ARRAY,
		TEXTURE_TYPE_MAX
	};

	enum TextureSamples {
		TEXTURE_SAMPLES_1,
		TEXTURE_SAMPLES_2,
		TEXTURE_SAMPLES_4,
		TEXTURE_SAMPLES_8,
		TEXTURE_SAMPLES_16,
		TEXTURE_SAMPLES_32,
		TEXTURE_SAMPLES_64,
		TEXTURE_SAMPLES_MAX
	};

	enum TextureUsageBits {
		TEXTURE_USAGE_SAMPLING_BIT = (1 << 0),
		TEXTURE_USAGE_COLOR_ATTACHMENT_BIT = (1 << 1),
		TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT = (1 << 2),
		TEXTURE_USAGE_STORAGE_BIT = (1 << 3),
		TEXTURE_USAGE_STORAGE_ATOMIC_BIT = (1 << 4),
		TEXTURE_USAGE_CPU_READ_BIT = (1 << 5),
		TEXTURE_USAGE_CAN_UPDATE_BIT = (1 << 6),
		TEXTURE_USAGE_CAN_COPY_FROM_BIT = (1 << 7),
		TEXTURE_USAGE_CAN_COPY_TO_BIT = (1 << 8),
		TEXTURE_USAGE_INPUT_ATTACHMENT_BIT = (1 << 9),
		TEXTURE_USAGE_VRS_ATTACHMENT_BIT = (1 << 10),
	};

	enum TextureSwizzle {
		TEXTURE_SWIZZLE_IDENTITY,
		TEXTURE_SWIZZLE_ZERO,
		TEXTURE_SWIZZLE_ONE,
		TEXTURE_SWIZZLE_R,
		TEXTURE_SWIZZLE_G,
		TEXTURE_SWIZZLE_B,
		TEXTURE_SWIZZLE_A,
		TEXTURE_SWIZZLE_MAX
	};

	struct TextureFormat {
		DataFormat format;
		uint32_t width;
		uint32_t height;
		uint32_t depth;
		uint32_t array_layers;
		uint32_t mipmaps;
		TextureType texture_type;
		TextureSamples samples;
		uint32_t usage_bits;
		Vector<DataFormat> shareable_formats;
		bool is_resolve_buffer = false;

		bool operator==(const TextureFormat &b) const {
			if (format != b.format) {
				return false;
			} else if (width != b.width) {
				return false;
			} else if (height != b.height) {
				return false;
			} else if (depth != b.depth) {
				return false;
			} else if (array_layers != b.array_layers) {
				return false;
			} else if (mipmaps != b.mipmaps) {
				return false;
			} else if (texture_type != b.texture_type) {
				return false;
			} else if (samples != b.samples) {
				return false;
			} else if (usage_bits != b.usage_bits) {
				return false;
			} else if (shareable_formats != b.shareable_formats) {
				return false;
			} else {
				return true;
			}
		}

		TextureFormat() {
			format = DATA_FORMAT_R8_UNORM;
			width = 1;
			height = 1;
			depth = 1;
			array_layers = 1;
			mipmaps = 1;
			texture_type = TEXTURE_TYPE_2D;
			samples = TEXTURE_SAMPLES_1;
			usage_bits = 0;
		}
	};

	struct TextureView {
		DataFormat format_override;
		TextureSwizzle swizzle_r;
		TextureSwizzle swizzle_g;
		TextureSwizzle swizzle_b;
		TextureSwizzle swizzle_a;

		TextureView() {
			format_override = DATA_FORMAT_MAX; //means, use same as format
			swizzle_r = TEXTURE_SWIZZLE_R;
			swizzle_g = TEXTURE_SWIZZLE_G;
			swizzle_b = TEXTURE_SWIZZLE_B;
			swizzle_a = TEXTURE_SWIZZLE_A;
		}
	};

	virtual RID texture_create(const TextureFormat &p_format, const TextureView &p_view, const Vector<Vector<uint8_t>> &p_data = Vector<Vector<uint8_t>>()) = 0;
	virtual RID texture_create_shared(const TextureView &p_view, RID p_with_texture) = 0;
	virtual RID texture_create_from_extension(TextureType p_type, DataFormat p_format, TextureSamples p_samples, uint64_t p_flags, uint64_t p_image, uint64_t p_width, uint64_t p_height, uint64_t p_depth, uint64_t p_layers) = 0;

	enum TextureSliceType {
		TEXTURE_SLICE_2D,
		TEXTURE_SLICE_CUBEMAP,
		TEXTURE_SLICE_3D,
		TEXTURE_SLICE_2D_ARRAY,
	};

	virtual RID texture_create_shared_from_slice(const TextureView &p_view, RID p_with_texture, uint32_t p_layer, uint32_t p_mipmap, uint32_t p_mipmaps = 1, TextureSliceType p_slice_type = TEXTURE_SLICE_2D, uint32_t p_layers = 0) = 0;

	virtual Error texture_update(RID p_texture, uint32_t p_layer, const Vector<uint8_t> &p_data, BitField<BarrierMask> p_post_barrier = BARRIER_MASK_ALL_BARRIERS) = 0;
	virtual Error texture_update_partial(RID p_texture, const Ref<Image> &p_image, int p_dst_x, int p_dst_y, uint32_t p_mipmap = 0, uint32_t p_layer = 0, BitField<BarrierMask> p_post_barrier = BARRIER_MASK_ALL_BARRIERS) = 0;
	virtual Vector<uint8_t> texture_get_data(RID p_texture, uint32_t p_layer) = 0; // CPU textures will return immediately, while GPU textures will most likely force a flush

	virtual bool texture_is_format_supported_for_usage(DataFormat p_format, BitField<RenderingDevice::TextureUsageBits> p_usage) const = 0;
	virtual bool texture_is_shared(RID p_texture) = 0;
	virtual bool texture_is_valid(RID p_texture) = 0;
	virtual Size2i texture_size(RID p_texture) = 0;

	virtual Error texture_copy(RID p_from_texture, RID p_to_texture, const Vector3 &p_from, const Vector3 &p_to, const Vector3 &p_size, uint32_t p_src_mipmap, uint32_t p_dst_mipmap, uint32_t p_src_layer, uint32_t p_dst_layer, BitField<BarrierMask> p_post_barrier = BARRIER_MASK_ALL_BARRIERS) = 0;
	virtual Error texture_clear(RID p_texture, const Color &p_color, uint32_t p_base_mipmap, uint32_t p_mipmaps, uint32_t p_base_layer, uint32_t p_layers, BitField<BarrierMask> p_post_barrier = BARRIER_MASK_ALL_BARRIERS) = 0;
	virtual Error texture_resolve_multisample(RID p_from_texture, RID p_to_texture, BitField<BarrierMask> p_post_barrier = BARRIER_MASK_ALL_BARRIERS) = 0;

	/*********************/
	/**** FRAMEBUFFER ****/
	/*********************/

	struct AttachmentFormat {
		enum { UNUSED_ATTACHMENT = 0xFFFFFFFF };
		DataFormat format;
		TextureSamples samples;
		uint32_t usage_flags;
		AttachmentFormat() {
			format = DATA_FORMAT_R8G8B8A8_UNORM;
			samples = TEXTURE_SAMPLES_1;
			usage_flags = 0;
		}
	};

	typedef int64_t FramebufferFormatID;

	// This ID is warranted to be unique for the same formats, does not need to be freed
	virtual FramebufferFormatID framebuffer_format_create(const Vector<AttachmentFormat> &p_format, uint32_t p_view_count = 1) = 0;
	struct FramebufferPass {
		enum {
			ATTACHMENT_UNUSED = -1
		};
		Vector<int32_t> color_attachments;
		Vector<int32_t> input_attachments;
		Vector<int32_t> resolve_attachments;
		Vector<int32_t> preserve_attachments;
		int32_t depth_attachment = ATTACHMENT_UNUSED;
		int32_t vrs_attachment = ATTACHMENT_UNUSED; // density map for VRS, only used if supported
	};

	virtual FramebufferFormatID framebuffer_format_create_multipass(const Vector<AttachmentFormat> &p_attachments, const Vector<FramebufferPass> &p_passes, uint32_t p_view_count = 1) = 0;
	virtual FramebufferFormatID framebuffer_format_create_empty(TextureSamples p_samples = TEXTURE_SAMPLES_1) = 0;
	virtual TextureSamples framebuffer_format_get_texture_samples(FramebufferFormatID p_format, uint32_t p_pass = 0) = 0;

	virtual RID framebuffer_create(const Vector<RID> &p_texture_attachments, FramebufferFormatID p_format_check = INVALID_ID, uint32_t p_view_count = 1) = 0;
	virtual RID framebuffer_create_multipass(const Vector<RID> &p_texture_attachments, const Vector<FramebufferPass> &p_passes, FramebufferFormatID p_format_check = INVALID_ID, uint32_t p_view_count = 1) = 0;
	virtual RID framebuffer_create_empty(const Size2i &p_size, TextureSamples p_samples = TEXTURE_SAMPLES_1, FramebufferFormatID p_format_check = INVALID_ID) = 0;
	virtual bool framebuffer_is_valid(RID p_framebuffer) const = 0;
	virtual void framebuffer_set_invalidation_callback(RID p_framebuffer, InvalidationCallback p_callback, void *p_userdata) = 0;

	virtual FramebufferFormatID framebuffer_get_format(RID p_framebuffer) = 0;

	/*****************/
	/**** SAMPLER ****/
	/*****************/

	enum SamplerFilter {
		SAMPLER_FILTER_NEAREST,
		SAMPLER_FILTER_LINEAR,
	};

	enum SamplerRepeatMode {
		SAMPLER_REPEAT_MODE_REPEAT,
		SAMPLER_REPEAT_MODE_MIRRORED_REPEAT,
		SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE,
		SAMPLER_REPEAT_MODE_CLAMP_TO_BORDER,
		SAMPLER_REPEAT_MODE_MIRROR_CLAMP_TO_EDGE,
		SAMPLER_REPEAT_MODE_MAX
	};

	enum SamplerBorderColor {
		SAMPLER_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
		SAMPLER_BORDER_COLOR_INT_TRANSPARENT_BLACK,
		SAMPLER_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
		SAMPLER_BORDER_COLOR_INT_OPAQUE_BLACK,
		SAMPLER_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
		SAMPLER_BORDER_COLOR_INT_OPAQUE_WHITE,
		SAMPLER_BORDER_COLOR_MAX
	};

	struct SamplerState {
		SamplerFilter mag_filter;
		SamplerFilter min_filter;
		SamplerFilter mip_filter;
		SamplerRepeatMode repeat_u;
		SamplerRepeatMode repeat_v;
		SamplerRepeatMode repeat_w;
		float lod_bias;
		bool use_anisotropy;
		float anisotropy_max;
		bool enable_compare;
		CompareOperator compare_op;
		float min_lod;
		float max_lod;
		SamplerBorderColor border_color;
		bool unnormalized_uvw;

		SamplerState() {
			mag_filter = SAMPLER_FILTER_NEAREST;
			min_filter = SAMPLER_FILTER_NEAREST;
			mip_filter = SAMPLER_FILTER_NEAREST;
			repeat_u = SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
			repeat_v = SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
			repeat_w = SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
			lod_bias = 0;
			use_anisotropy = false;
			anisotropy_max = 1.0;
			enable_compare = false;
			compare_op = COMPARE_OP_ALWAYS;
			min_lod = 0;
			max_lod = 1e20; //something very large should do
			border_color = SAMPLER_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
			unnormalized_uvw = false;
		}
	};

	virtual RID sampler_create(const SamplerState &p_state) = 0;

	/**********************/
	/**** VERTEX ARRAY ****/
	/**********************/

	enum VertexFrequency {
		VERTEX_FREQUENCY_VERTEX,
		VERTEX_FREQUENCY_INSTANCE,
	};

	struct VertexAttribute {
		uint32_t location; //shader location
		uint32_t offset;
		DataFormat format;
		uint32_t stride;
		VertexFrequency frequency;
		VertexAttribute() {
			location = 0;
			offset = 0;
			stride = 0;
			format = DATA_FORMAT_MAX;
			frequency = VERTEX_FREQUENCY_VERTEX;
		}
	};
	virtual RID vertex_buffer_create(uint32_t p_size_bytes, const Vector<uint8_t> &p_data = Vector<uint8_t>(), bool p_use_as_storage = false) = 0;

	typedef int64_t VertexFormatID;

	// This ID is warranted to be unique for the same formats, does not need to be freed
	virtual VertexFormatID vertex_format_create(const Vector<VertexAttribute> &p_vertex_formats) = 0;
	virtual RID vertex_array_create(uint32_t p_vertex_count, VertexFormatID p_vertex_format, const Vector<RID> &p_src_buffers, const Vector<uint64_t> &p_offsets = Vector<uint64_t>()) = 0;

	enum IndexBufferFormat {
		INDEX_BUFFER_FORMAT_UINT16,
		INDEX_BUFFER_FORMAT_UINT32,
	};

	virtual RID index_buffer_create(uint32_t p_size_indices, IndexBufferFormat p_format, const Vector<uint8_t> &p_data = Vector<uint8_t>(), bool p_use_restart_indices = false) = 0;
	virtual RID index_array_create(RID p_index_buffer, uint32_t p_index_offset, uint32_t p_index_count) = 0;

	/****************/
	/**** SHADER ****/
	/****************/

	const Capabilities *get_device_capabilities() const { return &device_capabilities; };

	enum Features {
		SUPPORTS_MULTIVIEW,
		SUPPORTS_FSR_HALF_FLOAT,
		SUPPORTS_ATTACHMENT_VRS,
	};
	virtual bool has_feature(const Features p_feature) const = 0;

	virtual Vector<uint8_t> shader_compile_spirv_from_source(ShaderStage p_stage, const String &p_source_code, ShaderLanguage p_language = SHADER_LANGUAGE_GLSL, String *r_error = nullptr, bool p_allow_cache = true);
	virtual String shader_get_spirv_cache_key() const;

	static void shader_set_compile_to_spirv_function(ShaderCompileToSPIRVFunction p_function);
	static void shader_set_spirv_cache_function(ShaderCacheFunction p_function);
	static void shader_set_get_cache_key_function(ShaderSPIRVGetCacheKeyFunction p_function);

	struct ShaderStageSPIRVData {
		ShaderStage shader_stage;
		Vector<uint8_t> spir_v;

		ShaderStageSPIRVData() {
			shader_stage = SHADER_STAGE_VERTEX;
		}
	};

	virtual String shader_get_binary_cache_key() const = 0;
	virtual Vector<uint8_t> shader_compile_binary_from_spirv(const Vector<ShaderStageSPIRVData> &p_spirv, const String &p_shader_name = "") = 0;

	virtual RID shader_create_from_spirv(const Vector<ShaderStageSPIRVData> &p_spirv, const String &p_shader_name = "");
	virtual RID shader_create_from_bytecode(const Vector<uint8_t> &p_shader_binary) = 0;

	virtual uint32_t shader_get_vertex_input_attribute_mask(RID p_shader) = 0;

	/******************/
	/**** UNIFORMS ****/
	/******************/

	enum UniformType {
		UNIFORM_TYPE_SAMPLER, //for sampling only (sampler GLSL type)
		UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, // for sampling only, but includes a texture, (samplerXX GLSL type), first a sampler then a texture
		UNIFORM_TYPE_TEXTURE, //only texture, (textureXX GLSL type)
		UNIFORM_TYPE_IMAGE, // storage image (imageXX GLSL type), for compute mostly
		UNIFORM_TYPE_TEXTURE_BUFFER, // buffer texture (or TBO, textureBuffer type)
		UNIFORM_TYPE_SAMPLER_WITH_TEXTURE_BUFFER, // buffer texture with a sampler(or TBO, samplerBuffer type)
		UNIFORM_TYPE_IMAGE_BUFFER, //texel buffer, (imageBuffer type), for compute mostly
		UNIFORM_TYPE_UNIFORM_BUFFER, //regular uniform buffer (or UBO).
		UNIFORM_TYPE_STORAGE_BUFFER, //storage buffer ("buffer" qualifier) like UBO, but supports storage, for compute mostly
		UNIFORM_TYPE_INPUT_ATTACHMENT, //used for sub-pass read/write, for mobile mostly
		UNIFORM_TYPE_MAX
	};

	enum StorageBufferUsage {
		STORAGE_BUFFER_USAGE_DISPATCH_INDIRECT = 1,
	};

	virtual RID uniform_buffer_create(uint32_t p_size_bytes, const Vector<uint8_t> &p_data = Vector<uint8_t>()) = 0;
	virtual RID storage_buffer_create(uint32_t p_size, const Vector<uint8_t> &p_data = Vector<uint8_t>(), BitField<StorageBufferUsage> p_usage = 0) = 0;
	virtual RID texture_buffer_create(uint32_t p_size_elements, DataFormat p_format, const Vector<uint8_t> &p_data = Vector<uint8_t>()) = 0;

	struct Uniform {
		UniformType uniform_type;
		int binding; // Binding index as specified in shader.

	private:
		// In most cases only one ID is provided per binding, so avoid allocating memory unnecessarily for performance.
		RID id; // If only one is provided, this is used.
		Vector<RID> ids; // If multiple ones are provided, this is used instead.

	public:
		_FORCE_INLINE_ uint32_t get_id_count() const {
			return (id.is_valid() ? 1 : ids.size());
		}

		_FORCE_INLINE_ RID get_id(uint32_t p_idx) const {
			if (id.is_valid()) {
				ERR_FAIL_COND_V(p_idx != 0, RID());
				return id;
			} else {
				return ids[p_idx];
			}
		}
		_FORCE_INLINE_ void set_id(uint32_t p_idx, RID p_id) {
			if (id.is_valid()) {
				ERR_FAIL_COND(p_idx != 0);
				id = p_id;
			} else {
				ids.write[p_idx] = p_id;
			}
		}

		_FORCE_INLINE_ void append_id(RID p_id) {
			if (ids.is_empty()) {
				if (id == RID()) {
					id = p_id;
				} else {
					ids.push_back(id);
					ids.push_back(p_id);
					id = RID();
				}
			} else {
				ids.push_back(p_id);
			}
		}

		_FORCE_INLINE_ void clear_ids() {
			id = RID();
			ids.clear();
		}

		_FORCE_INLINE_ Uniform(UniformType p_type, int p_binding, RID p_id) {
			uniform_type = p_type;
			binding = p_binding;
			id = p_id;
		}
		_FORCE_INLINE_ Uniform(UniformType p_type, int p_binding, const Vector<RID> &p_ids) {
			uniform_type = p_type;
			binding = p_binding;
			ids = p_ids;
		}
		_FORCE_INLINE_ Uniform() {
			uniform_type = UNIFORM_TYPE_IMAGE;
			binding = 0;
		}
	};

	virtual RID uniform_set_create(const Vector<Uniform> &p_uniforms, RID p_shader, uint32_t p_shader_set) = 0;
	virtual bool uniform_set_is_valid(RID p_uniform_set) = 0;
	virtual void uniform_set_set_invalidation_callback(RID p_uniform_set, InvalidationCallback p_callback, void *p_userdata) = 0;

	virtual Error buffer_update(RID p_buffer, uint32_t p_offset, uint32_t p_size, const void *p_data, BitField<BarrierMask> p_post_barrier = BARRIER_MASK_ALL_BARRIERS) = 0;
	virtual Error buffer_clear(RID p_buffer, uint32_t p_offset, uint32_t p_size, BitField<BarrierMask> p_post_barrier = BARRIER_MASK_ALL_BARRIERS) = 0;
	virtual Vector<uint8_t> buffer_get_data(RID p_buffer, uint32_t p_offset = 0, uint32_t p_size = 0) = 0; // This causes stall, only use to retrieve large buffers for saving.

	/******************************************/
	/**** PIPELINE SPECIALIZATION CONSTANT ****/
	/******************************************/

	enum PipelineSpecializationConstantType {
		PIPELINE_SPECIALIZATION_CONSTANT_TYPE_BOOL,
		PIPELINE_SPECIALIZATION_CONSTANT_TYPE_INT,
		PIPELINE_SPECIALIZATION_CONSTANT_TYPE_FLOAT,
	};

	struct PipelineSpecializationConstant {
		PipelineSpecializationConstantType type;
		uint32_t constant_id;
		union {
			uint32_t int_value;
			float float_value;
			bool bool_value;
		};

		PipelineSpecializationConstant() {
			type = PIPELINE_SPECIALIZATION_CONSTANT_TYPE_BOOL;
			constant_id = 0;
			int_value = 0;
		}
	};

	/*************************/
	/**** RENDER PIPELINE ****/
	/*************************/

	enum RenderPrimitive {
		RENDER_PRIMITIVE_POINTS,
		RENDER_PRIMITIVE_LINES,
		RENDER_PRIMITIVE_LINES_WITH_ADJACENCY,
		RENDER_PRIMITIVE_LINESTRIPS,
		RENDER_PRIMITIVE_LINESTRIPS_WITH_ADJACENCY,
		RENDER_PRIMITIVE_TRIANGLES,
		RENDER_PRIMITIVE_TRIANGLES_WITH_ADJACENCY,
		RENDER_PRIMITIVE_TRIANGLE_STRIPS,
		RENDER_PRIMITIVE_TRIANGLE_STRIPS_WITH_AJACENCY,
		RENDER_PRIMITIVE_TRIANGLE_STRIPS_WITH_RESTART_INDEX,
		RENDER_PRIMITIVE_TESSELATION_PATCH,
		RENDER_PRIMITIVE_MAX
	};

	//disable optimization, tessellate control points

	enum PolygonCullMode {
		POLYGON_CULL_DISABLED,
		POLYGON_CULL_FRONT,
		POLYGON_CULL_BACK,
	};

	enum PolygonFrontFace {
		POLYGON_FRONT_FACE_CLOCKWISE,
		POLYGON_FRONT_FACE_COUNTER_CLOCKWISE,
	};

	enum StencilOperation {
		STENCIL_OP_KEEP,
		STENCIL_OP_ZERO,
		STENCIL_OP_REPLACE,
		STENCIL_OP_INCREMENT_AND_CLAMP,
		STENCIL_OP_DECREMENT_AND_CLAMP,
		STENCIL_OP_INVERT,
		STENCIL_OP_INCREMENT_AND_WRAP,
		STENCIL_OP_DECREMENT_AND_WRAP,
		STENCIL_OP_MAX //not an actual operator, just the amount of operators :D
	};

	enum LogicOperation {
		LOGIC_OP_CLEAR,
		LOGIC_OP_AND,
		LOGIC_OP_AND_REVERSE,
		LOGIC_OP_COPY,
		LOGIC_OP_AND_INVERTED,
		LOGIC_OP_NO_OP,
		LOGIC_OP_XOR,
		LOGIC_OP_OR,
		LOGIC_OP_NOR,
		LOGIC_OP_EQUIVALENT,
		LOGIC_OP_INVERT,
		LOGIC_OP_OR_REVERSE,
		LOGIC_OP_COPY_INVERTED,
		LOGIC_OP_OR_INVERTED,
		LOGIC_OP_NAND,
		LOGIC_OP_SET,
		LOGIC_OP_MAX //not an actual operator, just the amount of operators :D
	};

	enum BlendFactor {
		BLEND_FACTOR_ZERO,
		BLEND_FACTOR_ONE,
		BLEND_FACTOR_SRC_COLOR,
		BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
		BLEND_FACTOR_DST_COLOR,
		BLEND_FACTOR_ONE_MINUS_DST_COLOR,
		BLEND_FACTOR_SRC_ALPHA,
		BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
		BLEND_FACTOR_DST_ALPHA,
		BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
		BLEND_FACTOR_CONSTANT_COLOR,
		BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR,
		BLEND_FACTOR_CONSTANT_ALPHA,
		BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA,
		BLEND_FACTOR_SRC_ALPHA_SATURATE,
		BLEND_FACTOR_SRC1_COLOR,
		BLEND_FACTOR_ONE_MINUS_SRC1_COLOR,
		BLEND_FACTOR_SRC1_ALPHA,
		BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA,
		BLEND_FACTOR_MAX
	};

	enum BlendOperation {
		BLEND_OP_ADD,
		BLEND_OP_SUBTRACT,
		BLEND_OP_REVERSE_SUBTRACT,
		BLEND_OP_MINIMUM,
		BLEND_OP_MAXIMUM, //yes this one is an actual operator
		BLEND_OP_MAX //not an actual operator, just the amount of operators :D
	};

	struct PipelineRasterizationState {
		bool enable_depth_clamp;
		bool discard_primitives;
		bool wireframe;
		PolygonCullMode cull_mode;
		PolygonFrontFace front_face;
		bool depth_bias_enabled;
		float depth_bias_constant_factor;
		float depth_bias_clamp;
		float depth_bias_slope_factor;
		float line_width;
		uint32_t patch_control_points;
		PipelineRasterizationState() {
			enable_depth_clamp = false;
			discard_primitives = false;
			wireframe = false;
			cull_mode = POLYGON_CULL_DISABLED;
			front_face = POLYGON_FRONT_FACE_CLOCKWISE;
			depth_bias_enabled = false;
			depth_bias_constant_factor = 0;
			depth_bias_clamp = 0;
			depth_bias_slope_factor = 0;
			line_width = 1.0;
			patch_control_points = 1;
		}
	};

	struct PipelineMultisampleState {
		TextureSamples sample_count;
		bool enable_sample_shading;
		float min_sample_shading;
		Vector<uint32_t> sample_mask;
		bool enable_alpha_to_coverage;
		bool enable_alpha_to_one;

		PipelineMultisampleState() {
			sample_count = TEXTURE_SAMPLES_1;
			enable_sample_shading = false;
			min_sample_shading = 0;
			enable_alpha_to_coverage = false;
			enable_alpha_to_one = false;
		}
	};

	struct PipelineDepthStencilState {
		bool enable_depth_test;
		bool enable_depth_write;
		CompareOperator depth_compare_operator;
		bool enable_depth_range;
		float depth_range_min;
		float depth_range_max;
		bool enable_stencil;

		struct StencilOperationState {
			StencilOperation fail;
			StencilOperation pass;
			StencilOperation depth_fail;
			CompareOperator compare;
			uint32_t compare_mask;
			uint32_t write_mask;
			uint32_t reference;

			StencilOperationState() {
				fail = STENCIL_OP_ZERO;
				pass = STENCIL_OP_ZERO;
				depth_fail = STENCIL_OP_ZERO;
				compare = COMPARE_OP_ALWAYS;
				compare_mask = 0;
				write_mask = 0;
				reference = 0;
			}
		};

		StencilOperationState front_op;
		StencilOperationState back_op;

		PipelineDepthStencilState() {
			enable_depth_test = false;
			enable_depth_write = false;
			depth_compare_operator = COMPARE_OP_ALWAYS;
			enable_depth_range = false;
			depth_range_min = 0;
			depth_range_max = 0;
			enable_stencil = false;
		}
	};

	struct PipelineColorBlendState {
		bool enable_logic_op;
		LogicOperation logic_op;
		struct Attachment {
			bool enable_blend;
			BlendFactor src_color_blend_factor;
			BlendFactor dst_color_blend_factor;
			BlendOperation color_blend_op;
			BlendFactor src_alpha_blend_factor;
			BlendFactor dst_alpha_blend_factor;
			BlendOperation alpha_blend_op;
			bool write_r;
			bool write_g;
			bool write_b;
			bool write_a;
			Attachment() {
				enable_blend = false;
				src_color_blend_factor = BLEND_FACTOR_ZERO;
				dst_color_blend_factor = BLEND_FACTOR_ZERO;
				color_blend_op = BLEND_OP_ADD;
				src_alpha_blend_factor = BLEND_FACTOR_ZERO;
				dst_alpha_blend_factor = BLEND_FACTOR_ZERO;
				alpha_blend_op = BLEND_OP_ADD;
				write_r = true;
				write_g = true;
				write_b = true;
				write_a = true;
			}
		};

		static PipelineColorBlendState create_disabled(int p_attachments = 1) {
			PipelineColorBlendState bs;
			for (int i = 0; i < p_attachments; i++) {
				bs.attachments.push_back(Attachment());
			}
			return bs;
		}

		static PipelineColorBlendState create_blend(int p_attachments = 1) {
			PipelineColorBlendState bs;
			for (int i = 0; i < p_attachments; i++) {
				Attachment ba;
				ba.enable_blend = true;
				ba.src_color_blend_factor = BLEND_FACTOR_SRC_ALPHA;
				ba.dst_color_blend_factor = BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
				ba.src_alpha_blend_factor = BLEND_FACTOR_SRC_ALPHA;
				ba.dst_alpha_blend_factor = BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;

				bs.attachments.push_back(ba);
			}
			return bs;
		}

		Vector<Attachment> attachments; //one per render target texture
		Color blend_constant;

		PipelineColorBlendState() {
			enable_logic_op = false;
			logic_op = LOGIC_OP_CLEAR;
		}
	};

	enum PipelineDynamicStateFlags {
		DYNAMIC_STATE_LINE_WIDTH = (1 << 0),
		DYNAMIC_STATE_DEPTH_BIAS = (1 << 1),
		DYNAMIC_STATE_BLEND_CONSTANTS = (1 << 2),
		DYNAMIC_STATE_DEPTH_BOUNDS = (1 << 3),
		DYNAMIC_STATE_STENCIL_COMPARE_MASK = (1 << 4),
		DYNAMIC_STATE_STENCIL_WRITE_MASK = (1 << 5),
		DYNAMIC_STATE_STENCIL_REFERENCE = (1 << 6),
	};

	virtual bool render_pipeline_is_valid(RID p_pipeline) = 0;
	virtual RID render_pipeline_create(RID p_shader, FramebufferFormatID p_framebuffer_format, VertexFormatID p_vertex_format, RenderPrimitive p_render_primitive, const PipelineRasterizationState &p_rasterization_state, const PipelineMultisampleState &p_multisample_state, const PipelineDepthStencilState &p_depth_stencil_state, const PipelineColorBlendState &p_blend_state, BitField<PipelineDynamicStateFlags> p_dynamic_state_flags = 0, uint32_t p_for_render_pass = 0, const Vector<PipelineSpecializationConstant> &p_specialization_constants = Vector<PipelineSpecializationConstant>()) = 0;

	/**************************/
	/**** COMPUTE PIPELINE ****/
	/**************************/

	virtual RID compute_pipeline_create(RID p_shader, const Vector<PipelineSpecializationConstant> &p_specialization_constants = Vector<PipelineSpecializationConstant>()) = 0;
	virtual bool compute_pipeline_is_valid(RID p_pipeline) = 0;

	/****************/
	/**** SCREEN ****/
	/****************/

	virtual int screen_get_width(DisplayServer::WindowID p_screen = 0) const = 0;
	virtual int screen_get_height(DisplayServer::WindowID p_screen = 0) const = 0;
	virtual FramebufferFormatID screen_get_framebuffer_format() const = 0;

	/********************/
	/**** DRAW LISTS ****/
	/********************/

	enum InitialAction {
		INITIAL_ACTION_CLEAR, //start rendering and clear the whole framebuffer (region or not) (supply params)
		INITIAL_ACTION_CLEAR_REGION, //start rendering and clear the framebuffer in the specified region (supply params)
		INITIAL_ACTION_CLEAR_REGION_CONTINUE, //continue rendering and clear the framebuffer in the specified region (supply params)
		INITIAL_ACTION_KEEP, //start rendering, but keep attached color texture contents (depth will be cleared)
		INITIAL_ACTION_DROP, //start rendering, ignore what is there, just write above it
		INITIAL_ACTION_CONTINUE, //continue rendering (framebuffer must have been left in "continue" state as final action previously)
		INITIAL_ACTION_MAX
	};

	enum FinalAction {
		FINAL_ACTION_READ, //will no longer render to it, allows attached textures to be read again, but depth buffer contents will be dropped (Can't be read from)
		FINAL_ACTION_DISCARD, // discard contents after rendering
		FINAL_ACTION_CONTINUE, //will continue rendering later, attached textures can't be read until re-bound with "finish"
		FINAL_ACTION_MAX
	};

	typedef int64_t DrawListID;

	virtual DrawListID draw_list_begin_for_screen(DisplayServer::WindowID p_screen = 0, const Color &p_clear_color = Color()) = 0;
	virtual DrawListID draw_list_begin(RID p_framebuffer, InitialAction p_initial_color_action, FinalAction p_final_color_action, InitialAction p_initial_depth_action, FinalAction p_final_depth_action, const Vector<Color> &p_clear_color_values = Vector<Color>(), float p_clear_depth = 1.0, uint32_t p_clear_stencil = 0, const Rect2 &p_region = Rect2(), const Vector<RID> &p_storage_textures = Vector<RID>()) = 0;
	virtual Error draw_list_begin_split(RID p_framebuffer, uint32_t p_splits, DrawListID *r_split_ids, InitialAction p_initial_color_action, FinalAction p_final_color_action, InitialAction p_initial_depth_action, FinalAction p_final_depth_action, const Vector<Color> &p_clear_color_values = Vector<Color>(), float p_clear_depth = 1.0, uint32_t p_clear_stencil = 0, const Rect2 &p_region = Rect2(), const Vector<RID> &p_storage_textures = Vector<RID>()) = 0;

	virtual void draw_list_set_blend_constants(DrawListID p_list, const Color &p_color) = 0;
	virtual void draw_list_bind_render_pipeline(DrawListID p_list, RID p_render_pipeline) = 0;
	virtual void draw_list_bind_uniform_set(DrawListID p_list, RID p_uniform_set, uint32_t p_index) = 0;
	virtual void draw_list_bind_vertex_array(DrawListID p_list, RID p_vertex_array) = 0;
	virtual void draw_list_bind_index_array(DrawListID p_list, RID p_index_array) = 0;
	virtual void draw_list_set_line_width(DrawListID p_list, float p_width) = 0;
	virtual void draw_list_set_push_constant(DrawListID p_list, const void *p_data, uint32_t p_data_size) = 0;

	virtual void draw_list_draw(DrawListID p_list, bool p_use_indices, uint32_t p_instances = 1, uint32_t p_procedural_vertices = 0) = 0;

	virtual void draw_list_enable_scissor(DrawListID p_list, const Rect2 &p_rect) = 0;
	virtual void draw_list_disable_scissor(DrawListID p_list) = 0;

	virtual uint32_t draw_list_get_current_pass() = 0;
	virtual DrawListID draw_list_switch_to_next_pass() = 0;
	virtual Error draw_list_switch_to_next_pass_split(uint32_t p_splits, DrawListID *r_split_ids) = 0;

	virtual void draw_list_end(BitField<BarrierMask> p_post_barrier = BARRIER_MASK_ALL_BARRIERS) = 0;

	/***********************/
	/**** COMPUTE LISTS ****/
	/***********************/

	typedef int64_t ComputeListID;

	virtual ComputeListID compute_list_begin(bool p_allow_draw_overlap = false) = 0;
	virtual void compute_list_bind_compute_pipeline(ComputeListID p_list, RID p_compute_pipeline) = 0;
	virtual void compute_list_bind_uniform_set(ComputeListID p_list, RID p_uniform_set, uint32_t p_index) = 0;
	virtual void compute_list_set_push_constant(ComputeListID p_list, const void *p_data, uint32_t p_data_size) = 0;
	virtual void compute_list_dispatch(ComputeListID p_list, uint32_t p_x_groups, uint32_t p_y_groups, uint32_t p_z_groups) = 0;
	virtual void compute_list_dispatch_threads(ComputeListID p_list, uint32_t p_x_threads, uint32_t p_y_threads, uint32_t p_z_threads) = 0;
	virtual void compute_list_dispatch_indirect(ComputeListID p_list, RID p_buffer, uint32_t p_offset) = 0;
	virtual void compute_list_add_barrier(ComputeListID p_list) = 0;

	virtual void compute_list_end(BitField<BarrierMask> p_post_barrier = BARRIER_MASK_ALL_BARRIERS) = 0;

	virtual void barrier(BitField<BarrierMask> p_from = BARRIER_MASK_ALL_BARRIERS, BitField<BarrierMask> p_to = BARRIER_MASK_ALL_BARRIERS) = 0;
	virtual void full_barrier() = 0;

	/***************/
	/**** FREE! ****/
	/***************/

	virtual void free(RID p_id) = 0;

	/****************/
	/**** Timing ****/
	/****************/

	virtual void capture_timestamp(const String &p_name) = 0;
	virtual uint32_t get_captured_timestamps_count() const = 0;
	virtual uint64_t get_captured_timestamps_frame() const = 0;
	virtual uint64_t get_captured_timestamp_gpu_time(uint32_t p_index) const = 0;
	virtual uint64_t get_captured_timestamp_cpu_time(uint32_t p_index) const = 0;
	virtual String get_captured_timestamp_name(uint32_t p_index) const = 0;

	/****************/
	/**** LIMITS ****/
	/****************/

	enum Limit {
		LIMIT_MAX_BOUND_UNIFORM_SETS,
		LIMIT_MAX_FRAMEBUFFER_COLOR_ATTACHMENTS,
		LIMIT_MAX_TEXTURES_PER_UNIFORM_SET,
		LIMIT_MAX_SAMPLERS_PER_UNIFORM_SET,
		LIMIT_MAX_STORAGE_BUFFERS_PER_UNIFORM_SET,
		LIMIT_MAX_STORAGE_IMAGES_PER_UNIFORM_SET,
		LIMIT_MAX_UNIFORM_BUFFERS_PER_UNIFORM_SET,
		LIMIT_MAX_DRAW_INDEXED_INDEX,
		LIMIT_MAX_FRAMEBUFFER_HEIGHT,
		LIMIT_MAX_FRAMEBUFFER_WIDTH,
		LIMIT_MAX_TEXTURE_ARRAY_LAYERS,
		LIMIT_MAX_TEXTURE_SIZE_1D,
		LIMIT_MAX_TEXTURE_SIZE_2D,
		LIMIT_MAX_TEXTURE_SIZE_3D,
		LIMIT_MAX_TEXTURE_SIZE_CUBE,
		LIMIT_MAX_TEXTURES_PER_SHADER_STAGE,
		LIMIT_MAX_SAMPLERS_PER_SHADER_STAGE,
		LIMIT_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
		LIMIT_MAX_STORAGE_IMAGES_PER_SHADER_STAGE,
		LIMIT_MAX_UNIFORM_BUFFERS_PER_SHADER_STAGE,
		LIMIT_MAX_PUSH_CONSTANT_SIZE,
		LIMIT_MAX_UNIFORM_BUFFER_SIZE,
		LIMIT_MAX_VERTEX_INPUT_ATTRIBUTE_OFFSET,
		LIMIT_MAX_VERTEX_INPUT_ATTRIBUTES,
		LIMIT_MAX_VERTEX_INPUT_BINDINGS,
		LIMIT_MAX_VERTEX_INPUT_BINDING_STRIDE,
		LIMIT_MIN_UNIFORM_BUFFER_OFFSET_ALIGNMENT,
		LIMIT_MAX_COMPUTE_SHARED_MEMORY_SIZE,
		LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_X,
		LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_Y,
		LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_Z,
		LIMIT_MAX_COMPUTE_WORKGROUP_INVOCATIONS,
		LIMIT_MAX_COMPUTE_WORKGROUP_SIZE_X,
		LIMIT_MAX_COMPUTE_WORKGROUP_SIZE_Y,
		LIMIT_MAX_COMPUTE_WORKGROUP_SIZE_Z,
		LIMIT_MAX_VIEWPORT_DIMENSIONS_X,
		LIMIT_MAX_VIEWPORT_DIMENSIONS_Y,
		LIMIT_SUBGROUP_SIZE,
		LIMIT_SUBGROUP_IN_SHADERS, // Set flags using SHADER_STAGE_VERTEX_BIT, SHADER_STAGE_FRAGMENT_BIT, etc.
		LIMIT_SUBGROUP_OPERATIONS,
		LIMIT_VRS_TEXEL_WIDTH,
		LIMIT_VRS_TEXEL_HEIGHT,
	};

	virtual uint64_t limit_get(Limit p_limit) const = 0;

	//methods below not exposed, used by RenderingDeviceRD
	virtual void prepare_screen_for_drawing() = 0;

	virtual void swap_buffers() = 0;

	virtual uint32_t get_frame_delay() const = 0;

	virtual void submit() = 0;
	virtual void sync() = 0;

	enum MemoryType {
		MEMORY_TEXTURES,
		MEMORY_BUFFERS,
		MEMORY_TOTAL
	};

	virtual uint64_t get_memory_usage(MemoryType p_type) const = 0;

	virtual RenderingDevice *create_local_device() = 0;

	virtual void set_resource_name(RID p_id, const String p_name) = 0;

	virtual void draw_command_begin_label(String p_label_name, const Color p_color = Color(1, 1, 1, 1)) = 0;
	virtual void draw_command_insert_label(String p_label_name, const Color p_color = Color(1, 1, 1, 1)) = 0;
	virtual void draw_command_end_label() = 0;

	virtual String get_device_vendor_name() const = 0;
	virtual String get_device_name() const = 0;
	virtual RenderingDevice::DeviceType get_device_type() const = 0;
	virtual String get_device_api_version() const = 0;
	virtual String get_device_pipeline_cache_uuid() const = 0;

	virtual uint64_t get_driver_resource(DriverResource p_resource, RID p_rid = RID(), uint64_t p_index = 0) = 0;

	static RenderingDevice *get_singleton();
	RenderingDevice();

protected:
	static const char *shader_stage_names[RenderingDevice::SHADER_STAGE_MAX];

	static const uint32_t MAX_UNIFORM_SETS = 16;

	//binders to script API
	RID _texture_create(const Ref<RDTextureFormat> &p_format, const Ref<RDTextureView> &p_view, const TypedArray<PackedByteArray> &p_data = Array());
	RID _texture_create_shared(const Ref<RDTextureView> &p_view, RID p_with_texture);
	RID _texture_create_shared_from_slice(const Ref<RDTextureView> &p_view, RID p_with_texture, uint32_t p_layer, uint32_t p_mipmap, uint32_t p_mipmaps = 1, TextureSliceType p_slice_type = TEXTURE_SLICE_2D);

	FramebufferFormatID _framebuffer_format_create(const TypedArray<RDAttachmentFormat> &p_attachments, uint32_t p_view_count);
	FramebufferFormatID _framebuffer_format_create_multipass(const TypedArray<RDAttachmentFormat> &p_attachments, const TypedArray<RDFramebufferPass> &p_passes, uint32_t p_view_count);
	RID _framebuffer_create(const TypedArray<RID> &p_textures, FramebufferFormatID p_format_check = INVALID_ID, uint32_t p_view_count = 1);
	RID _framebuffer_create_multipass(const TypedArray<RID> &p_textures, const TypedArray<RDFramebufferPass> &p_passes, FramebufferFormatID p_format_check = INVALID_ID, uint32_t p_view_count = 1);
	RID _sampler_create(const Ref<RDSamplerState> &p_state);
	VertexFormatID _vertex_format_create(const TypedArray<RDVertexAttribute> &p_vertex_formats);
	RID _vertex_array_create(uint32_t p_vertex_count, VertexFormatID p_vertex_format, const TypedArray<RID> &p_src_buffers, const Vector<int64_t> &p_offsets = Vector<int64_t>());

	Ref<RDShaderSPIRV> _shader_compile_spirv_from_source(const Ref<RDShaderSource> &p_source, bool p_allow_cache = true);
	Vector<uint8_t> _shader_compile_binary_from_spirv(const Ref<RDShaderSPIRV> &p_bytecode, const String &p_shader_name = "");
	RID _shader_create_from_spirv(const Ref<RDShaderSPIRV> &p_spirv, const String &p_shader_name = "");

	RID _uniform_set_create(const TypedArray<RDUniform> &p_uniforms, RID p_shader, uint32_t p_shader_set);

	Error _buffer_update(RID p_buffer, uint32_t p_offset, uint32_t p_size, const Vector<uint8_t> &p_data, BitField<BarrierMask> p_post_barrier = BARRIER_MASK_ALL_BARRIERS);

	RID _render_pipeline_create(RID p_shader, FramebufferFormatID p_framebuffer_format, VertexFormatID p_vertex_format, RenderPrimitive p_render_primitive, const Ref<RDPipelineRasterizationState> &p_rasterization_state, const Ref<RDPipelineMultisampleState> &p_multisample_state, const Ref<RDPipelineDepthStencilState> &p_depth_stencil_state, const Ref<RDPipelineColorBlendState> &p_blend_state, BitField<PipelineDynamicStateFlags> p_dynamic_state_flags, uint32_t p_for_render_pass, const TypedArray<RDPipelineSpecializationConstant> &p_specialization_constants);
	RID _compute_pipeline_create(RID p_shader, const TypedArray<RDPipelineSpecializationConstant> &p_specialization_constants);

	Vector<int64_t> _draw_list_begin_split(RID p_framebuffer, uint32_t p_splits, InitialAction p_initial_color_action, FinalAction p_final_color_action, InitialAction p_initial_depth_action, FinalAction p_final_depth_action, const Vector<Color> &p_clear_color_values = Vector<Color>(), float p_clear_depth = 1.0, uint32_t p_clear_stencil = 0, const Rect2 &p_region = Rect2(), const TypedArray<RID> &p_storage_textures = TypedArray<RID>());
	void _draw_list_set_push_constant(DrawListID p_list, const Vector<uint8_t> &p_data, uint32_t p_data_size);
	void _compute_list_set_push_constant(ComputeListID p_list, const Vector<uint8_t> &p_data, uint32_t p_data_size);
	Vector<int64_t> _draw_list_switch_to_next_pass_split(uint32_t p_splits);

	struct SpirvReflectionData {
		BitField<ShaderStage> stages_mask;
		uint32_t vertex_input_mask;
		uint32_t fragment_output_mask;
		bool is_compute;
		uint32_t compute_local_size[3];
		uint32_t push_constant_size;
		BitField<ShaderStage> push_constant_stages_mask;

		struct Uniform {
			UniformType type;
			uint32_t binding;
			BitField<ShaderStage> stages_mask;
			uint32_t length; // Size of arrays (in total elements), or ubos (in bytes * total elements).
			bool writable;
		};
		Vector<Vector<Uniform>> uniforms;

		struct SpecializationConstant {
			PipelineSpecializationConstantType type;
			uint32_t constant_id;
			union {
				uint32_t int_value;
				float float_value;
				bool bool_value;
			};
			BitField<ShaderStage> stages_mask;
		};
		Vector<SpecializationConstant> specialization_constants;
	};

	Error _reflect_spirv(const Vector<ShaderStageSPIRVData> &p_spirv, SpirvReflectionData &r_reflection_data);
};

VARIANT_ENUM_CAST(RenderingDevice::DeviceType)
VARIANT_ENUM_CAST(RenderingDevice::DriverResource)
VARIANT_ENUM_CAST(RenderingDevice::ShaderStage)
VARIANT_ENUM_CAST(RenderingDevice::ShaderLanguage)
VARIANT_ENUM_CAST(RenderingDevice::CompareOperator)
VARIANT_ENUM_CAST(RenderingDevice::DataFormat)
VARIANT_BITFIELD_CAST(RenderingDevice::BarrierMask);
VARIANT_ENUM_CAST(RenderingDevice::TextureType)
VARIANT_ENUM_CAST(RenderingDevice::TextureSamples)
VARIANT_BITFIELD_CAST(RenderingDevice::TextureUsageBits)
VARIANT_ENUM_CAST(RenderingDevice::TextureSwizzle)
VARIANT_ENUM_CAST(RenderingDevice::TextureSliceType)
VARIANT_ENUM_CAST(RenderingDevice::SamplerFilter)
VARIANT_ENUM_CAST(RenderingDevice::SamplerRepeatMode)
VARIANT_ENUM_CAST(RenderingDevice::SamplerBorderColor)
VARIANT_ENUM_CAST(RenderingDevice::VertexFrequency)
VARIANT_ENUM_CAST(RenderingDevice::IndexBufferFormat)
VARIANT_BITFIELD_CAST(RenderingDevice::StorageBufferUsage)
VARIANT_ENUM_CAST(RenderingDevice::UniformType)
VARIANT_ENUM_CAST(RenderingDevice::RenderPrimitive)
VARIANT_ENUM_CAST(RenderingDevice::PolygonCullMode)
VARIANT_ENUM_CAST(RenderingDevice::PolygonFrontFace)
VARIANT_ENUM_CAST(RenderingDevice::StencilOperation)
VARIANT_ENUM_CAST(RenderingDevice::LogicOperation)
VARIANT_ENUM_CAST(RenderingDevice::BlendFactor)
VARIANT_ENUM_CAST(RenderingDevice::BlendOperation)
VARIANT_BITFIELD_CAST(RenderingDevice::PipelineDynamicStateFlags)
VARIANT_ENUM_CAST(RenderingDevice::PipelineSpecializationConstantType)
VARIANT_ENUM_CAST(RenderingDevice::InitialAction)
VARIANT_ENUM_CAST(RenderingDevice::FinalAction)
VARIANT_ENUM_CAST(RenderingDevice::Limit)
VARIANT_ENUM_CAST(RenderingDevice::MemoryType)
VARIANT_ENUM_CAST(RenderingDevice::Features)

typedef RenderingDevice RD;

#endif // RENDERING_DEVICE_H
