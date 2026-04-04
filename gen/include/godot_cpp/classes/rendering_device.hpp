/**************************************************************************/
/*  rendering_device.hpp                                                  */
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

#pragma once

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/rd_pipeline_specialization_constant.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/packed_color_array.hpp>
#include <godot_cpp/variant/packed_int64_array.hpp>
#include <godot_cpp/variant/rect2.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/typed_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Callable;
class RDAttachmentFormat;
class RDFramebufferPass;
class RDPipelineColorBlendState;
class RDPipelineDepthStencilState;
class RDPipelineMultisampleState;
class RDPipelineRasterizationState;
class RDSamplerState;
class RDShaderSPIRV;
class RDShaderSource;
class RDTextureFormat;
class RDTextureView;
class RDUniform;
class RDVertexAttribute;
struct Vector2i;
struct Vector3;

class RenderingDevice : public Object {
	GDEXTENSION_CLASS(RenderingDevice, Object)

public:
	enum DeviceType {
		DEVICE_TYPE_OTHER = 0,
		DEVICE_TYPE_INTEGRATED_GPU = 1,
		DEVICE_TYPE_DISCRETE_GPU = 2,
		DEVICE_TYPE_VIRTUAL_GPU = 3,
		DEVICE_TYPE_CPU = 4,
		DEVICE_TYPE_MAX = 5,
	};

	enum DriverResource {
		DRIVER_RESOURCE_LOGICAL_DEVICE = 0,
		DRIVER_RESOURCE_PHYSICAL_DEVICE = 1,
		DRIVER_RESOURCE_TOPMOST_OBJECT = 2,
		DRIVER_RESOURCE_COMMAND_QUEUE = 3,
		DRIVER_RESOURCE_QUEUE_FAMILY = 4,
		DRIVER_RESOURCE_TEXTURE = 5,
		DRIVER_RESOURCE_TEXTURE_VIEW = 6,
		DRIVER_RESOURCE_TEXTURE_DATA_FORMAT = 7,
		DRIVER_RESOURCE_SAMPLER = 8,
		DRIVER_RESOURCE_UNIFORM_SET = 9,
		DRIVER_RESOURCE_BUFFER = 10,
		DRIVER_RESOURCE_COMPUTE_PIPELINE = 11,
		DRIVER_RESOURCE_RENDER_PIPELINE = 12,
		DRIVER_RESOURCE_VULKAN_DEVICE = 0,
		DRIVER_RESOURCE_VULKAN_PHYSICAL_DEVICE = 1,
		DRIVER_RESOURCE_VULKAN_INSTANCE = 2,
		DRIVER_RESOURCE_VULKAN_QUEUE = 3,
		DRIVER_RESOURCE_VULKAN_QUEUE_FAMILY_INDEX = 4,
		DRIVER_RESOURCE_VULKAN_IMAGE = 5,
		DRIVER_RESOURCE_VULKAN_IMAGE_VIEW = 6,
		DRIVER_RESOURCE_VULKAN_IMAGE_NATIVE_TEXTURE_FORMAT = 7,
		DRIVER_RESOURCE_VULKAN_SAMPLER = 8,
		DRIVER_RESOURCE_VULKAN_DESCRIPTOR_SET = 9,
		DRIVER_RESOURCE_VULKAN_BUFFER = 10,
		DRIVER_RESOURCE_VULKAN_COMPUTE_PIPELINE = 11,
		DRIVER_RESOURCE_VULKAN_RENDER_PIPELINE = 12,
	};

	enum DataFormat {
		DATA_FORMAT_R4G4_UNORM_PACK8 = 0,
		DATA_FORMAT_R4G4B4A4_UNORM_PACK16 = 1,
		DATA_FORMAT_B4G4R4A4_UNORM_PACK16 = 2,
		DATA_FORMAT_R5G6B5_UNORM_PACK16 = 3,
		DATA_FORMAT_B5G6R5_UNORM_PACK16 = 4,
		DATA_FORMAT_R5G5B5A1_UNORM_PACK16 = 5,
		DATA_FORMAT_B5G5R5A1_UNORM_PACK16 = 6,
		DATA_FORMAT_A1R5G5B5_UNORM_PACK16 = 7,
		DATA_FORMAT_R8_UNORM = 8,
		DATA_FORMAT_R8_SNORM = 9,
		DATA_FORMAT_R8_USCALED = 10,
		DATA_FORMAT_R8_SSCALED = 11,
		DATA_FORMAT_R8_UINT = 12,
		DATA_FORMAT_R8_SINT = 13,
		DATA_FORMAT_R8_SRGB = 14,
		DATA_FORMAT_R8G8_UNORM = 15,
		DATA_FORMAT_R8G8_SNORM = 16,
		DATA_FORMAT_R8G8_USCALED = 17,
		DATA_FORMAT_R8G8_SSCALED = 18,
		DATA_FORMAT_R8G8_UINT = 19,
		DATA_FORMAT_R8G8_SINT = 20,
		DATA_FORMAT_R8G8_SRGB = 21,
		DATA_FORMAT_R8G8B8_UNORM = 22,
		DATA_FORMAT_R8G8B8_SNORM = 23,
		DATA_FORMAT_R8G8B8_USCALED = 24,
		DATA_FORMAT_R8G8B8_SSCALED = 25,
		DATA_FORMAT_R8G8B8_UINT = 26,
		DATA_FORMAT_R8G8B8_SINT = 27,
		DATA_FORMAT_R8G8B8_SRGB = 28,
		DATA_FORMAT_B8G8R8_UNORM = 29,
		DATA_FORMAT_B8G8R8_SNORM = 30,
		DATA_FORMAT_B8G8R8_USCALED = 31,
		DATA_FORMAT_B8G8R8_SSCALED = 32,
		DATA_FORMAT_B8G8R8_UINT = 33,
		DATA_FORMAT_B8G8R8_SINT = 34,
		DATA_FORMAT_B8G8R8_SRGB = 35,
		DATA_FORMAT_R8G8B8A8_UNORM = 36,
		DATA_FORMAT_R8G8B8A8_SNORM = 37,
		DATA_FORMAT_R8G8B8A8_USCALED = 38,
		DATA_FORMAT_R8G8B8A8_SSCALED = 39,
		DATA_FORMAT_R8G8B8A8_UINT = 40,
		DATA_FORMAT_R8G8B8A8_SINT = 41,
		DATA_FORMAT_R8G8B8A8_SRGB = 42,
		DATA_FORMAT_B8G8R8A8_UNORM = 43,
		DATA_FORMAT_B8G8R8A8_SNORM = 44,
		DATA_FORMAT_B8G8R8A8_USCALED = 45,
		DATA_FORMAT_B8G8R8A8_SSCALED = 46,
		DATA_FORMAT_B8G8R8A8_UINT = 47,
		DATA_FORMAT_B8G8R8A8_SINT = 48,
		DATA_FORMAT_B8G8R8A8_SRGB = 49,
		DATA_FORMAT_A8B8G8R8_UNORM_PACK32 = 50,
		DATA_FORMAT_A8B8G8R8_SNORM_PACK32 = 51,
		DATA_FORMAT_A8B8G8R8_USCALED_PACK32 = 52,
		DATA_FORMAT_A8B8G8R8_SSCALED_PACK32 = 53,
		DATA_FORMAT_A8B8G8R8_UINT_PACK32 = 54,
		DATA_FORMAT_A8B8G8R8_SINT_PACK32 = 55,
		DATA_FORMAT_A8B8G8R8_SRGB_PACK32 = 56,
		DATA_FORMAT_A2R10G10B10_UNORM_PACK32 = 57,
		DATA_FORMAT_A2R10G10B10_SNORM_PACK32 = 58,
		DATA_FORMAT_A2R10G10B10_USCALED_PACK32 = 59,
		DATA_FORMAT_A2R10G10B10_SSCALED_PACK32 = 60,
		DATA_FORMAT_A2R10G10B10_UINT_PACK32 = 61,
		DATA_FORMAT_A2R10G10B10_SINT_PACK32 = 62,
		DATA_FORMAT_A2B10G10R10_UNORM_PACK32 = 63,
		DATA_FORMAT_A2B10G10R10_SNORM_PACK32 = 64,
		DATA_FORMAT_A2B10G10R10_USCALED_PACK32 = 65,
		DATA_FORMAT_A2B10G10R10_SSCALED_PACK32 = 66,
		DATA_FORMAT_A2B10G10R10_UINT_PACK32 = 67,
		DATA_FORMAT_A2B10G10R10_SINT_PACK32 = 68,
		DATA_FORMAT_R16_UNORM = 69,
		DATA_FORMAT_R16_SNORM = 70,
		DATA_FORMAT_R16_USCALED = 71,
		DATA_FORMAT_R16_SSCALED = 72,
		DATA_FORMAT_R16_UINT = 73,
		DATA_FORMAT_R16_SINT = 74,
		DATA_FORMAT_R16_SFLOAT = 75,
		DATA_FORMAT_R16G16_UNORM = 76,
		DATA_FORMAT_R16G16_SNORM = 77,
		DATA_FORMAT_R16G16_USCALED = 78,
		DATA_FORMAT_R16G16_SSCALED = 79,
		DATA_FORMAT_R16G16_UINT = 80,
		DATA_FORMAT_R16G16_SINT = 81,
		DATA_FORMAT_R16G16_SFLOAT = 82,
		DATA_FORMAT_R16G16B16_UNORM = 83,
		DATA_FORMAT_R16G16B16_SNORM = 84,
		DATA_FORMAT_R16G16B16_USCALED = 85,
		DATA_FORMAT_R16G16B16_SSCALED = 86,
		DATA_FORMAT_R16G16B16_UINT = 87,
		DATA_FORMAT_R16G16B16_SINT = 88,
		DATA_FORMAT_R16G16B16_SFLOAT = 89,
		DATA_FORMAT_R16G16B16A16_UNORM = 90,
		DATA_FORMAT_R16G16B16A16_SNORM = 91,
		DATA_FORMAT_R16G16B16A16_USCALED = 92,
		DATA_FORMAT_R16G16B16A16_SSCALED = 93,
		DATA_FORMAT_R16G16B16A16_UINT = 94,
		DATA_FORMAT_R16G16B16A16_SINT = 95,
		DATA_FORMAT_R16G16B16A16_SFLOAT = 96,
		DATA_FORMAT_R32_UINT = 97,
		DATA_FORMAT_R32_SINT = 98,
		DATA_FORMAT_R32_SFLOAT = 99,
		DATA_FORMAT_R32G32_UINT = 100,
		DATA_FORMAT_R32G32_SINT = 101,
		DATA_FORMAT_R32G32_SFLOAT = 102,
		DATA_FORMAT_R32G32B32_UINT = 103,
		DATA_FORMAT_R32G32B32_SINT = 104,
		DATA_FORMAT_R32G32B32_SFLOAT = 105,
		DATA_FORMAT_R32G32B32A32_UINT = 106,
		DATA_FORMAT_R32G32B32A32_SINT = 107,
		DATA_FORMAT_R32G32B32A32_SFLOAT = 108,
		DATA_FORMAT_R64_UINT = 109,
		DATA_FORMAT_R64_SINT = 110,
		DATA_FORMAT_R64_SFLOAT = 111,
		DATA_FORMAT_R64G64_UINT = 112,
		DATA_FORMAT_R64G64_SINT = 113,
		DATA_FORMAT_R64G64_SFLOAT = 114,
		DATA_FORMAT_R64G64B64_UINT = 115,
		DATA_FORMAT_R64G64B64_SINT = 116,
		DATA_FORMAT_R64G64B64_SFLOAT = 117,
		DATA_FORMAT_R64G64B64A64_UINT = 118,
		DATA_FORMAT_R64G64B64A64_SINT = 119,
		DATA_FORMAT_R64G64B64A64_SFLOAT = 120,
		DATA_FORMAT_B10G11R11_UFLOAT_PACK32 = 121,
		DATA_FORMAT_E5B9G9R9_UFLOAT_PACK32 = 122,
		DATA_FORMAT_D16_UNORM = 123,
		DATA_FORMAT_X8_D24_UNORM_PACK32 = 124,
		DATA_FORMAT_D32_SFLOAT = 125,
		DATA_FORMAT_S8_UINT = 126,
		DATA_FORMAT_D16_UNORM_S8_UINT = 127,
		DATA_FORMAT_D24_UNORM_S8_UINT = 128,
		DATA_FORMAT_D32_SFLOAT_S8_UINT = 129,
		DATA_FORMAT_BC1_RGB_UNORM_BLOCK = 130,
		DATA_FORMAT_BC1_RGB_SRGB_BLOCK = 131,
		DATA_FORMAT_BC1_RGBA_UNORM_BLOCK = 132,
		DATA_FORMAT_BC1_RGBA_SRGB_BLOCK = 133,
		DATA_FORMAT_BC2_UNORM_BLOCK = 134,
		DATA_FORMAT_BC2_SRGB_BLOCK = 135,
		DATA_FORMAT_BC3_UNORM_BLOCK = 136,
		DATA_FORMAT_BC3_SRGB_BLOCK = 137,
		DATA_FORMAT_BC4_UNORM_BLOCK = 138,
		DATA_FORMAT_BC4_SNORM_BLOCK = 139,
		DATA_FORMAT_BC5_UNORM_BLOCK = 140,
		DATA_FORMAT_BC5_SNORM_BLOCK = 141,
		DATA_FORMAT_BC6H_UFLOAT_BLOCK = 142,
		DATA_FORMAT_BC6H_SFLOAT_BLOCK = 143,
		DATA_FORMAT_BC7_UNORM_BLOCK = 144,
		DATA_FORMAT_BC7_SRGB_BLOCK = 145,
		DATA_FORMAT_ETC2_R8G8B8_UNORM_BLOCK = 146,
		DATA_FORMAT_ETC2_R8G8B8_SRGB_BLOCK = 147,
		DATA_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK = 148,
		DATA_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK = 149,
		DATA_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK = 150,
		DATA_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK = 151,
		DATA_FORMAT_EAC_R11_UNORM_BLOCK = 152,
		DATA_FORMAT_EAC_R11_SNORM_BLOCK = 153,
		DATA_FORMAT_EAC_R11G11_UNORM_BLOCK = 154,
		DATA_FORMAT_EAC_R11G11_SNORM_BLOCK = 155,
		DATA_FORMAT_ASTC_4x4_UNORM_BLOCK = 156,
		DATA_FORMAT_ASTC_4x4_SRGB_BLOCK = 157,
		DATA_FORMAT_ASTC_5x4_UNORM_BLOCK = 158,
		DATA_FORMAT_ASTC_5x4_SRGB_BLOCK = 159,
		DATA_FORMAT_ASTC_5x5_UNORM_BLOCK = 160,
		DATA_FORMAT_ASTC_5x5_SRGB_BLOCK = 161,
		DATA_FORMAT_ASTC_6x5_UNORM_BLOCK = 162,
		DATA_FORMAT_ASTC_6x5_SRGB_BLOCK = 163,
		DATA_FORMAT_ASTC_6x6_UNORM_BLOCK = 164,
		DATA_FORMAT_ASTC_6x6_SRGB_BLOCK = 165,
		DATA_FORMAT_ASTC_8x5_UNORM_BLOCK = 166,
		DATA_FORMAT_ASTC_8x5_SRGB_BLOCK = 167,
		DATA_FORMAT_ASTC_8x6_UNORM_BLOCK = 168,
		DATA_FORMAT_ASTC_8x6_SRGB_BLOCK = 169,
		DATA_FORMAT_ASTC_8x8_UNORM_BLOCK = 170,
		DATA_FORMAT_ASTC_8x8_SRGB_BLOCK = 171,
		DATA_FORMAT_ASTC_10x5_UNORM_BLOCK = 172,
		DATA_FORMAT_ASTC_10x5_SRGB_BLOCK = 173,
		DATA_FORMAT_ASTC_10x6_UNORM_BLOCK = 174,
		DATA_FORMAT_ASTC_10x6_SRGB_BLOCK = 175,
		DATA_FORMAT_ASTC_10x8_UNORM_BLOCK = 176,
		DATA_FORMAT_ASTC_10x8_SRGB_BLOCK = 177,
		DATA_FORMAT_ASTC_10x10_UNORM_BLOCK = 178,
		DATA_FORMAT_ASTC_10x10_SRGB_BLOCK = 179,
		DATA_FORMAT_ASTC_12x10_UNORM_BLOCK = 180,
		DATA_FORMAT_ASTC_12x10_SRGB_BLOCK = 181,
		DATA_FORMAT_ASTC_12x12_UNORM_BLOCK = 182,
		DATA_FORMAT_ASTC_12x12_SRGB_BLOCK = 183,
		DATA_FORMAT_G8B8G8R8_422_UNORM = 184,
		DATA_FORMAT_B8G8R8G8_422_UNORM = 185,
		DATA_FORMAT_G8_B8_R8_3PLANE_420_UNORM = 186,
		DATA_FORMAT_G8_B8R8_2PLANE_420_UNORM = 187,
		DATA_FORMAT_G8_B8_R8_3PLANE_422_UNORM = 188,
		DATA_FORMAT_G8_B8R8_2PLANE_422_UNORM = 189,
		DATA_FORMAT_G8_B8_R8_3PLANE_444_UNORM = 190,
		DATA_FORMAT_R10X6_UNORM_PACK16 = 191,
		DATA_FORMAT_R10X6G10X6_UNORM_2PACK16 = 192,
		DATA_FORMAT_R10X6G10X6B10X6A10X6_UNORM_4PACK16 = 193,
		DATA_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16 = 194,
		DATA_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16 = 195,
		DATA_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16 = 196,
		DATA_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16 = 197,
		DATA_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16 = 198,
		DATA_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16 = 199,
		DATA_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16 = 200,
		DATA_FORMAT_R12X4_UNORM_PACK16 = 201,
		DATA_FORMAT_R12X4G12X4_UNORM_2PACK16 = 202,
		DATA_FORMAT_R12X4G12X4B12X4A12X4_UNORM_4PACK16 = 203,
		DATA_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16 = 204,
		DATA_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16 = 205,
		DATA_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16 = 206,
		DATA_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16 = 207,
		DATA_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16 = 208,
		DATA_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16 = 209,
		DATA_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16 = 210,
		DATA_FORMAT_G16B16G16R16_422_UNORM = 211,
		DATA_FORMAT_B16G16R16G16_422_UNORM = 212,
		DATA_FORMAT_G16_B16_R16_3PLANE_420_UNORM = 213,
		DATA_FORMAT_G16_B16R16_2PLANE_420_UNORM = 214,
		DATA_FORMAT_G16_B16_R16_3PLANE_422_UNORM = 215,
		DATA_FORMAT_G16_B16R16_2PLANE_422_UNORM = 216,
		DATA_FORMAT_G16_B16_R16_3PLANE_444_UNORM = 217,
		DATA_FORMAT_ASTC_4x4_SFLOAT_BLOCK = 218,
		DATA_FORMAT_ASTC_5x4_SFLOAT_BLOCK = 219,
		DATA_FORMAT_ASTC_5x5_SFLOAT_BLOCK = 220,
		DATA_FORMAT_ASTC_6x5_SFLOAT_BLOCK = 221,
		DATA_FORMAT_ASTC_6x6_SFLOAT_BLOCK = 222,
		DATA_FORMAT_ASTC_8x5_SFLOAT_BLOCK = 223,
		DATA_FORMAT_ASTC_8x6_SFLOAT_BLOCK = 224,
		DATA_FORMAT_ASTC_8x8_SFLOAT_BLOCK = 225,
		DATA_FORMAT_ASTC_10x5_SFLOAT_BLOCK = 226,
		DATA_FORMAT_ASTC_10x6_SFLOAT_BLOCK = 227,
		DATA_FORMAT_ASTC_10x8_SFLOAT_BLOCK = 228,
		DATA_FORMAT_ASTC_10x10_SFLOAT_BLOCK = 229,
		DATA_FORMAT_ASTC_12x10_SFLOAT_BLOCK = 230,
		DATA_FORMAT_ASTC_12x12_SFLOAT_BLOCK = 231,
		DATA_FORMAT_MAX = 232,
	};

	enum BarrierMask : uint64_t {
		BARRIER_MASK_VERTEX = 1,
		BARRIER_MASK_FRAGMENT = 8,
		BARRIER_MASK_COMPUTE = 2,
		BARRIER_MASK_TRANSFER = 4,
		BARRIER_MASK_RASTER = 9,
		BARRIER_MASK_ALL_BARRIERS = 32767,
		BARRIER_MASK_NO_BARRIER = 32768,
	};

	enum TextureType {
		TEXTURE_TYPE_1D = 0,
		TEXTURE_TYPE_2D = 1,
		TEXTURE_TYPE_3D = 2,
		TEXTURE_TYPE_CUBE = 3,
		TEXTURE_TYPE_1D_ARRAY = 4,
		TEXTURE_TYPE_2D_ARRAY = 5,
		TEXTURE_TYPE_CUBE_ARRAY = 6,
		TEXTURE_TYPE_MAX = 7,
	};

	enum TextureSamples {
		TEXTURE_SAMPLES_1 = 0,
		TEXTURE_SAMPLES_2 = 1,
		TEXTURE_SAMPLES_4 = 2,
		TEXTURE_SAMPLES_8 = 3,
		TEXTURE_SAMPLES_16 = 4,
		TEXTURE_SAMPLES_32 = 5,
		TEXTURE_SAMPLES_64 = 6,
		TEXTURE_SAMPLES_MAX = 7,
	};

	enum TextureUsageBits : uint64_t {
		TEXTURE_USAGE_SAMPLING_BIT = 1,
		TEXTURE_USAGE_COLOR_ATTACHMENT_BIT = 2,
		TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT = 4,
		TEXTURE_USAGE_DEPTH_RESOLVE_ATTACHMENT_BIT = 4096,
		TEXTURE_USAGE_STORAGE_BIT = 8,
		TEXTURE_USAGE_STORAGE_ATOMIC_BIT = 16,
		TEXTURE_USAGE_CPU_READ_BIT = 32,
		TEXTURE_USAGE_CAN_UPDATE_BIT = 64,
		TEXTURE_USAGE_CAN_COPY_FROM_BIT = 128,
		TEXTURE_USAGE_CAN_COPY_TO_BIT = 256,
		TEXTURE_USAGE_INPUT_ATTACHMENT_BIT = 512,
	};

	enum TextureSwizzle {
		TEXTURE_SWIZZLE_IDENTITY = 0,
		TEXTURE_SWIZZLE_ZERO = 1,
		TEXTURE_SWIZZLE_ONE = 2,
		TEXTURE_SWIZZLE_R = 3,
		TEXTURE_SWIZZLE_G = 4,
		TEXTURE_SWIZZLE_B = 5,
		TEXTURE_SWIZZLE_A = 6,
		TEXTURE_SWIZZLE_MAX = 7,
	};

	enum TextureSliceType {
		TEXTURE_SLICE_2D = 0,
		TEXTURE_SLICE_CUBEMAP = 1,
		TEXTURE_SLICE_3D = 2,
	};

	enum SamplerFilter {
		SAMPLER_FILTER_NEAREST = 0,
		SAMPLER_FILTER_LINEAR = 1,
	};

	enum SamplerRepeatMode {
		SAMPLER_REPEAT_MODE_REPEAT = 0,
		SAMPLER_REPEAT_MODE_MIRRORED_REPEAT = 1,
		SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE = 2,
		SAMPLER_REPEAT_MODE_CLAMP_TO_BORDER = 3,
		SAMPLER_REPEAT_MODE_MIRROR_CLAMP_TO_EDGE = 4,
		SAMPLER_REPEAT_MODE_MAX = 5,
	};

	enum SamplerBorderColor {
		SAMPLER_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK = 0,
		SAMPLER_BORDER_COLOR_INT_TRANSPARENT_BLACK = 1,
		SAMPLER_BORDER_COLOR_FLOAT_OPAQUE_BLACK = 2,
		SAMPLER_BORDER_COLOR_INT_OPAQUE_BLACK = 3,
		SAMPLER_BORDER_COLOR_FLOAT_OPAQUE_WHITE = 4,
		SAMPLER_BORDER_COLOR_INT_OPAQUE_WHITE = 5,
		SAMPLER_BORDER_COLOR_MAX = 6,
	};

	enum VertexFrequency {
		VERTEX_FREQUENCY_VERTEX = 0,
		VERTEX_FREQUENCY_INSTANCE = 1,
	};

	enum IndexBufferFormat {
		INDEX_BUFFER_FORMAT_UINT16 = 0,
		INDEX_BUFFER_FORMAT_UINT32 = 1,
	};

	enum StorageBufferUsage : uint64_t {
		STORAGE_BUFFER_USAGE_DISPATCH_INDIRECT = 1,
	};

	enum BufferCreationBits : uint64_t {
		BUFFER_CREATION_DEVICE_ADDRESS_BIT = 1,
		BUFFER_CREATION_AS_STORAGE_BIT = 2,
	};

	enum UniformType {
		UNIFORM_TYPE_SAMPLER = 0,
		UNIFORM_TYPE_SAMPLER_WITH_TEXTURE = 1,
		UNIFORM_TYPE_TEXTURE = 2,
		UNIFORM_TYPE_IMAGE = 3,
		UNIFORM_TYPE_TEXTURE_BUFFER = 4,
		UNIFORM_TYPE_SAMPLER_WITH_TEXTURE_BUFFER = 5,
		UNIFORM_TYPE_IMAGE_BUFFER = 6,
		UNIFORM_TYPE_UNIFORM_BUFFER = 7,
		UNIFORM_TYPE_STORAGE_BUFFER = 8,
		UNIFORM_TYPE_INPUT_ATTACHMENT = 9,
		UNIFORM_TYPE_UNIFORM_BUFFER_DYNAMIC = 10,
		UNIFORM_TYPE_STORAGE_BUFFER_DYNAMIC = 11,
		UNIFORM_TYPE_MAX = 12,
	};

	enum RenderPrimitive {
		RENDER_PRIMITIVE_POINTS = 0,
		RENDER_PRIMITIVE_LINES = 1,
		RENDER_PRIMITIVE_LINES_WITH_ADJACENCY = 2,
		RENDER_PRIMITIVE_LINESTRIPS = 3,
		RENDER_PRIMITIVE_LINESTRIPS_WITH_ADJACENCY = 4,
		RENDER_PRIMITIVE_TRIANGLES = 5,
		RENDER_PRIMITIVE_TRIANGLES_WITH_ADJACENCY = 6,
		RENDER_PRIMITIVE_TRIANGLE_STRIPS = 7,
		RENDER_PRIMITIVE_TRIANGLE_STRIPS_WITH_AJACENCY = 8,
		RENDER_PRIMITIVE_TRIANGLE_STRIPS_WITH_RESTART_INDEX = 9,
		RENDER_PRIMITIVE_TESSELATION_PATCH = 10,
		RENDER_PRIMITIVE_MAX = 11,
	};

	enum PolygonCullMode {
		POLYGON_CULL_DISABLED = 0,
		POLYGON_CULL_FRONT = 1,
		POLYGON_CULL_BACK = 2,
	};

	enum PolygonFrontFace {
		POLYGON_FRONT_FACE_CLOCKWISE = 0,
		POLYGON_FRONT_FACE_COUNTER_CLOCKWISE = 1,
	};

	enum StencilOperation {
		STENCIL_OP_KEEP = 0,
		STENCIL_OP_ZERO = 1,
		STENCIL_OP_REPLACE = 2,
		STENCIL_OP_INCREMENT_AND_CLAMP = 3,
		STENCIL_OP_DECREMENT_AND_CLAMP = 4,
		STENCIL_OP_INVERT = 5,
		STENCIL_OP_INCREMENT_AND_WRAP = 6,
		STENCIL_OP_DECREMENT_AND_WRAP = 7,
		STENCIL_OP_MAX = 8,
	};

	enum CompareOperator {
		COMPARE_OP_NEVER = 0,
		COMPARE_OP_LESS = 1,
		COMPARE_OP_EQUAL = 2,
		COMPARE_OP_LESS_OR_EQUAL = 3,
		COMPARE_OP_GREATER = 4,
		COMPARE_OP_NOT_EQUAL = 5,
		COMPARE_OP_GREATER_OR_EQUAL = 6,
		COMPARE_OP_ALWAYS = 7,
		COMPARE_OP_MAX = 8,
	};

	enum LogicOperation {
		LOGIC_OP_CLEAR = 0,
		LOGIC_OP_AND = 1,
		LOGIC_OP_AND_REVERSE = 2,
		LOGIC_OP_COPY = 3,
		LOGIC_OP_AND_INVERTED = 4,
		LOGIC_OP_NO_OP = 5,
		LOGIC_OP_XOR = 6,
		LOGIC_OP_OR = 7,
		LOGIC_OP_NOR = 8,
		LOGIC_OP_EQUIVALENT = 9,
		LOGIC_OP_INVERT = 10,
		LOGIC_OP_OR_REVERSE = 11,
		LOGIC_OP_COPY_INVERTED = 12,
		LOGIC_OP_OR_INVERTED = 13,
		LOGIC_OP_NAND = 14,
		LOGIC_OP_SET = 15,
		LOGIC_OP_MAX = 16,
	};

	enum BlendFactor {
		BLEND_FACTOR_ZERO = 0,
		BLEND_FACTOR_ONE = 1,
		BLEND_FACTOR_SRC_COLOR = 2,
		BLEND_FACTOR_ONE_MINUS_SRC_COLOR = 3,
		BLEND_FACTOR_DST_COLOR = 4,
		BLEND_FACTOR_ONE_MINUS_DST_COLOR = 5,
		BLEND_FACTOR_SRC_ALPHA = 6,
		BLEND_FACTOR_ONE_MINUS_SRC_ALPHA = 7,
		BLEND_FACTOR_DST_ALPHA = 8,
		BLEND_FACTOR_ONE_MINUS_DST_ALPHA = 9,
		BLEND_FACTOR_CONSTANT_COLOR = 10,
		BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR = 11,
		BLEND_FACTOR_CONSTANT_ALPHA = 12,
		BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA = 13,
		BLEND_FACTOR_SRC_ALPHA_SATURATE = 14,
		BLEND_FACTOR_SRC1_COLOR = 15,
		BLEND_FACTOR_ONE_MINUS_SRC1_COLOR = 16,
		BLEND_FACTOR_SRC1_ALPHA = 17,
		BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA = 18,
		BLEND_FACTOR_MAX = 19,
	};

	enum BlendOperation {
		BLEND_OP_ADD = 0,
		BLEND_OP_SUBTRACT = 1,
		BLEND_OP_REVERSE_SUBTRACT = 2,
		BLEND_OP_MINIMUM = 3,
		BLEND_OP_MAXIMUM = 4,
		BLEND_OP_MAX = 5,
	};

	enum PipelineDynamicStateFlags : uint64_t {
		DYNAMIC_STATE_LINE_WIDTH = 1,
		DYNAMIC_STATE_DEPTH_BIAS = 2,
		DYNAMIC_STATE_BLEND_CONSTANTS = 4,
		DYNAMIC_STATE_DEPTH_BOUNDS = 8,
		DYNAMIC_STATE_STENCIL_COMPARE_MASK = 16,
		DYNAMIC_STATE_STENCIL_WRITE_MASK = 32,
		DYNAMIC_STATE_STENCIL_REFERENCE = 64,
	};

	enum InitialAction {
		INITIAL_ACTION_LOAD = 0,
		INITIAL_ACTION_CLEAR = 1,
		INITIAL_ACTION_DISCARD = 2,
		INITIAL_ACTION_MAX = 3,
		INITIAL_ACTION_CLEAR_REGION = 1,
		INITIAL_ACTION_CLEAR_REGION_CONTINUE = 1,
		INITIAL_ACTION_KEEP = 0,
		INITIAL_ACTION_DROP = 2,
		INITIAL_ACTION_CONTINUE = 0,
	};

	enum FinalAction {
		FINAL_ACTION_STORE = 0,
		FINAL_ACTION_DISCARD = 1,
		FINAL_ACTION_MAX = 2,
		FINAL_ACTION_READ = 0,
		FINAL_ACTION_CONTINUE = 0,
	};

	enum ShaderStage {
		SHADER_STAGE_VERTEX = 0,
		SHADER_STAGE_FRAGMENT = 1,
		SHADER_STAGE_TESSELATION_CONTROL = 2,
		SHADER_STAGE_TESSELATION_EVALUATION = 3,
		SHADER_STAGE_COMPUTE = 4,
		SHADER_STAGE_MAX = 5,
		SHADER_STAGE_VERTEX_BIT = 1,
		SHADER_STAGE_FRAGMENT_BIT = 2,
		SHADER_STAGE_TESSELATION_CONTROL_BIT = 4,
		SHADER_STAGE_TESSELATION_EVALUATION_BIT = 8,
		SHADER_STAGE_COMPUTE_BIT = 16,
	};

	enum ShaderLanguage {
		SHADER_LANGUAGE_GLSL = 0,
		SHADER_LANGUAGE_HLSL = 1,
	};

	enum PipelineSpecializationConstantType {
		PIPELINE_SPECIALIZATION_CONSTANT_TYPE_BOOL = 0,
		PIPELINE_SPECIALIZATION_CONSTANT_TYPE_INT = 1,
		PIPELINE_SPECIALIZATION_CONSTANT_TYPE_FLOAT = 2,
	};

	enum Features {
		SUPPORTS_METALFX_SPATIAL = 3,
		SUPPORTS_METALFX_TEMPORAL = 4,
		SUPPORTS_BUFFER_DEVICE_ADDRESS = 6,
		SUPPORTS_IMAGE_ATOMIC_32_BIT = 7,
	};

	enum Limit {
		LIMIT_MAX_BOUND_UNIFORM_SETS = 0,
		LIMIT_MAX_FRAMEBUFFER_COLOR_ATTACHMENTS = 1,
		LIMIT_MAX_TEXTURES_PER_UNIFORM_SET = 2,
		LIMIT_MAX_SAMPLERS_PER_UNIFORM_SET = 3,
		LIMIT_MAX_STORAGE_BUFFERS_PER_UNIFORM_SET = 4,
		LIMIT_MAX_STORAGE_IMAGES_PER_UNIFORM_SET = 5,
		LIMIT_MAX_UNIFORM_BUFFERS_PER_UNIFORM_SET = 6,
		LIMIT_MAX_DRAW_INDEXED_INDEX = 7,
		LIMIT_MAX_FRAMEBUFFER_HEIGHT = 8,
		LIMIT_MAX_FRAMEBUFFER_WIDTH = 9,
		LIMIT_MAX_TEXTURE_ARRAY_LAYERS = 10,
		LIMIT_MAX_TEXTURE_SIZE_1D = 11,
		LIMIT_MAX_TEXTURE_SIZE_2D = 12,
		LIMIT_MAX_TEXTURE_SIZE_3D = 13,
		LIMIT_MAX_TEXTURE_SIZE_CUBE = 14,
		LIMIT_MAX_TEXTURES_PER_SHADER_STAGE = 15,
		LIMIT_MAX_SAMPLERS_PER_SHADER_STAGE = 16,
		LIMIT_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE = 17,
		LIMIT_MAX_STORAGE_IMAGES_PER_SHADER_STAGE = 18,
		LIMIT_MAX_UNIFORM_BUFFERS_PER_SHADER_STAGE = 19,
		LIMIT_MAX_PUSH_CONSTANT_SIZE = 20,
		LIMIT_MAX_UNIFORM_BUFFER_SIZE = 21,
		LIMIT_MAX_VERTEX_INPUT_ATTRIBUTE_OFFSET = 22,
		LIMIT_MAX_VERTEX_INPUT_ATTRIBUTES = 23,
		LIMIT_MAX_VERTEX_INPUT_BINDINGS = 24,
		LIMIT_MAX_VERTEX_INPUT_BINDING_STRIDE = 25,
		LIMIT_MIN_UNIFORM_BUFFER_OFFSET_ALIGNMENT = 26,
		LIMIT_MAX_COMPUTE_SHARED_MEMORY_SIZE = 27,
		LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_X = 28,
		LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_Y = 29,
		LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_Z = 30,
		LIMIT_MAX_COMPUTE_WORKGROUP_INVOCATIONS = 31,
		LIMIT_MAX_COMPUTE_WORKGROUP_SIZE_X = 32,
		LIMIT_MAX_COMPUTE_WORKGROUP_SIZE_Y = 33,
		LIMIT_MAX_COMPUTE_WORKGROUP_SIZE_Z = 34,
		LIMIT_MAX_VIEWPORT_DIMENSIONS_X = 35,
		LIMIT_MAX_VIEWPORT_DIMENSIONS_Y = 36,
		LIMIT_METALFX_TEMPORAL_SCALER_MIN_SCALE = 46,
		LIMIT_METALFX_TEMPORAL_SCALER_MAX_SCALE = 47,
	};

	enum MemoryType {
		MEMORY_TEXTURES = 0,
		MEMORY_BUFFERS = 1,
		MEMORY_TOTAL = 2,
	};

	enum BreadcrumbMarker {
		NONE = 0,
		REFLECTION_PROBES = 65536,
		SKY_PASS = 131072,
		LIGHTMAPPER_PASS = 196608,
		SHADOW_PASS_DIRECTIONAL = 262144,
		SHADOW_PASS_CUBE = 327680,
		OPAQUE_PASS = 393216,
		ALPHA_PASS = 458752,
		TRANSPARENT_PASS = 524288,
		POST_PROCESSING_PASS = 589824,
		BLIT_PASS = 655360,
		UI_PASS = 720896,
		DEBUG_PASS = 786432,
	};

	enum DrawFlags : uint64_t {
		DRAW_DEFAULT_ALL = 0,
		DRAW_CLEAR_COLOR_0 = 1,
		DRAW_CLEAR_COLOR_1 = 2,
		DRAW_CLEAR_COLOR_2 = 4,
		DRAW_CLEAR_COLOR_3 = 8,
		DRAW_CLEAR_COLOR_4 = 16,
		DRAW_CLEAR_COLOR_5 = 32,
		DRAW_CLEAR_COLOR_6 = 64,
		DRAW_CLEAR_COLOR_7 = 128,
		DRAW_CLEAR_COLOR_MASK = 255,
		DRAW_CLEAR_COLOR_ALL = 255,
		DRAW_IGNORE_COLOR_0 = 256,
		DRAW_IGNORE_COLOR_1 = 512,
		DRAW_IGNORE_COLOR_2 = 1024,
		DRAW_IGNORE_COLOR_3 = 2048,
		DRAW_IGNORE_COLOR_4 = 4096,
		DRAW_IGNORE_COLOR_5 = 8192,
		DRAW_IGNORE_COLOR_6 = 16384,
		DRAW_IGNORE_COLOR_7 = 32768,
		DRAW_IGNORE_COLOR_MASK = 65280,
		DRAW_IGNORE_COLOR_ALL = 65280,
		DRAW_CLEAR_DEPTH = 65536,
		DRAW_IGNORE_DEPTH = 131072,
		DRAW_CLEAR_STENCIL = 262144,
		DRAW_IGNORE_STENCIL = 524288,
		DRAW_CLEAR_ALL = 327935,
		DRAW_IGNORE_ALL = 720640,
	};

	static const int INVALID_ID = -1;
	static const int INVALID_FORMAT_ID = -1;

	RID texture_create(const Ref<RDTextureFormat> &p_format, const Ref<RDTextureView> &p_view, const TypedArray<PackedByteArray> &p_data = Array());
	RID texture_create_shared(const Ref<RDTextureView> &p_view, const RID &p_with_texture);
	RID texture_create_shared_from_slice(const Ref<RDTextureView> &p_view, const RID &p_with_texture, uint32_t p_layer, uint32_t p_mipmap, uint32_t p_mipmaps = 1, RenderingDevice::TextureSliceType p_slice_type = (RenderingDevice::TextureSliceType)0);
	RID texture_create_from_extension(RenderingDevice::TextureType p_type, RenderingDevice::DataFormat p_format, RenderingDevice::TextureSamples p_samples, BitField<RenderingDevice::TextureUsageBits> p_usage_flags, uint64_t p_image, uint64_t p_width, uint64_t p_height, uint64_t p_depth, uint64_t p_layers, uint64_t p_mipmaps = 1);
	Error texture_update(const RID &p_texture, uint32_t p_layer, const PackedByteArray &p_data);
	PackedByteArray texture_get_data(const RID &p_texture, uint32_t p_layer);
	Error texture_get_data_async(const RID &p_texture, uint32_t p_layer, const Callable &p_callback);
	bool texture_is_format_supported_for_usage(RenderingDevice::DataFormat p_format, BitField<RenderingDevice::TextureUsageBits> p_usage_flags) const;
	bool texture_is_shared(const RID &p_texture);
	bool texture_is_valid(const RID &p_texture);
	void texture_set_discardable(const RID &p_texture, bool p_discardable);
	bool texture_is_discardable(const RID &p_texture);
	Error texture_copy(const RID &p_from_texture, const RID &p_to_texture, const Vector3 &p_from_pos, const Vector3 &p_to_pos, const Vector3 &p_size, uint32_t p_src_mipmap, uint32_t p_dst_mipmap, uint32_t p_src_layer, uint32_t p_dst_layer);
	Error texture_clear(const RID &p_texture, const Color &p_color, uint32_t p_base_mipmap, uint32_t p_mipmap_count, uint32_t p_base_layer, uint32_t p_layer_count);
	Error texture_resolve_multisample(const RID &p_from_texture, const RID &p_to_texture);
	Ref<RDTextureFormat> texture_get_format(const RID &p_texture);
	uint64_t texture_get_native_handle(const RID &p_texture);
	int64_t framebuffer_format_create(const TypedArray<Ref<RDAttachmentFormat>> &p_attachments, uint32_t p_view_count = 1);
	int64_t framebuffer_format_create_multipass(const TypedArray<Ref<RDAttachmentFormat>> &p_attachments, const TypedArray<Ref<RDFramebufferPass>> &p_passes, uint32_t p_view_count = 1);
	int64_t framebuffer_format_create_empty(RenderingDevice::TextureSamples p_samples = (RenderingDevice::TextureSamples)0);
	RenderingDevice::TextureSamples framebuffer_format_get_texture_samples(int64_t p_format, uint32_t p_render_pass = 0);
	RID framebuffer_create(const TypedArray<RID> &p_textures, int64_t p_validate_with_format = -1, uint32_t p_view_count = 1);
	RID framebuffer_create_multipass(const TypedArray<RID> &p_textures, const TypedArray<Ref<RDFramebufferPass>> &p_passes, int64_t p_validate_with_format = -1, uint32_t p_view_count = 1);
	RID framebuffer_create_empty(const Vector2i &p_size, RenderingDevice::TextureSamples p_samples = (RenderingDevice::TextureSamples)0, int64_t p_validate_with_format = -1);
	int64_t framebuffer_get_format(const RID &p_framebuffer);
	bool framebuffer_is_valid(const RID &p_framebuffer) const;
	RID sampler_create(const Ref<RDSamplerState> &p_state);
	bool sampler_is_format_supported_for_filter(RenderingDevice::DataFormat p_format, RenderingDevice::SamplerFilter p_sampler_filter) const;
	RID vertex_buffer_create(uint32_t p_size_bytes, const PackedByteArray &p_data = PackedByteArray(), BitField<RenderingDevice::BufferCreationBits> p_creation_bits = (BitField<RenderingDevice::BufferCreationBits>)0);
	int64_t vertex_format_create(const TypedArray<Ref<RDVertexAttribute>> &p_vertex_descriptions);
	RID vertex_array_create(uint32_t p_vertex_count, int64_t p_vertex_format, const TypedArray<RID> &p_src_buffers, const PackedInt64Array &p_offsets = PackedInt64Array());
	RID index_buffer_create(uint32_t p_size_indices, RenderingDevice::IndexBufferFormat p_format, const PackedByteArray &p_data = PackedByteArray(), bool p_use_restart_indices = false, BitField<RenderingDevice::BufferCreationBits> p_creation_bits = (BitField<RenderingDevice::BufferCreationBits>)0);
	RID index_array_create(const RID &p_index_buffer, uint32_t p_index_offset, uint32_t p_index_count);
	Ref<RDShaderSPIRV> shader_compile_spirv_from_source(const Ref<RDShaderSource> &p_shader_source, bool p_allow_cache = true);
	PackedByteArray shader_compile_binary_from_spirv(const Ref<RDShaderSPIRV> &p_spirv_data, const String &p_name = String());
	RID shader_create_from_spirv(const Ref<RDShaderSPIRV> &p_spirv_data, const String &p_name = String());
	RID shader_create_from_bytecode(const PackedByteArray &p_binary_data, const RID &p_placeholder_rid = RID());
	RID shader_create_placeholder();
	uint64_t shader_get_vertex_input_attribute_mask(const RID &p_shader);
	RID uniform_buffer_create(uint32_t p_size_bytes, const PackedByteArray &p_data = PackedByteArray(), BitField<RenderingDevice::BufferCreationBits> p_creation_bits = (BitField<RenderingDevice::BufferCreationBits>)0);
	RID storage_buffer_create(uint32_t p_size_bytes, const PackedByteArray &p_data = PackedByteArray(), BitField<RenderingDevice::StorageBufferUsage> p_usage = (BitField<RenderingDevice::StorageBufferUsage>)0, BitField<RenderingDevice::BufferCreationBits> p_creation_bits = (BitField<RenderingDevice::BufferCreationBits>)0);
	RID texture_buffer_create(uint32_t p_size_bytes, RenderingDevice::DataFormat p_format, const PackedByteArray &p_data = PackedByteArray());
	RID uniform_set_create(const TypedArray<Ref<RDUniform>> &p_uniforms, const RID &p_shader, uint32_t p_shader_set);
	bool uniform_set_is_valid(const RID &p_uniform_set);
	Error buffer_copy(const RID &p_src_buffer, const RID &p_dst_buffer, uint32_t p_src_offset, uint32_t p_dst_offset, uint32_t p_size);
	Error buffer_update(const RID &p_buffer, uint32_t p_offset, uint32_t p_size_bytes, const PackedByteArray &p_data);
	Error buffer_clear(const RID &p_buffer, uint32_t p_offset, uint32_t p_size_bytes);
	PackedByteArray buffer_get_data(const RID &p_buffer, uint32_t p_offset_bytes = 0, uint32_t p_size_bytes = 0);
	Error buffer_get_data_async(const RID &p_buffer, const Callable &p_callback, uint32_t p_offset_bytes = 0, uint32_t p_size_bytes = 0);
	uint64_t buffer_get_device_address(const RID &p_buffer);
	RID render_pipeline_create(const RID &p_shader, int64_t p_framebuffer_format, int64_t p_vertex_format, RenderingDevice::RenderPrimitive p_primitive, const Ref<RDPipelineRasterizationState> &p_rasterization_state, const Ref<RDPipelineMultisampleState> &p_multisample_state, const Ref<RDPipelineDepthStencilState> &p_stencil_state, const Ref<RDPipelineColorBlendState> &p_color_blend_state, BitField<RenderingDevice::PipelineDynamicStateFlags> p_dynamic_state_flags = (BitField<RenderingDevice::PipelineDynamicStateFlags>)0, uint32_t p_for_render_pass = 0, const TypedArray<Ref<RDPipelineSpecializationConstant>> &p_specialization_constants = {});
	bool render_pipeline_is_valid(const RID &p_render_pipeline);
	RID compute_pipeline_create(const RID &p_shader, const TypedArray<Ref<RDPipelineSpecializationConstant>> &p_specialization_constants = {});
	bool compute_pipeline_is_valid(const RID &p_compute_pipeline);
	int32_t screen_get_width(int32_t p_screen = 0) const;
	int32_t screen_get_height(int32_t p_screen = 0) const;
	int64_t screen_get_framebuffer_format(int32_t p_screen = 0) const;
	int64_t draw_list_begin_for_screen(int32_t p_screen = 0, const Color &p_clear_color = Color(0, 0, 0, 1));
	int64_t draw_list_begin(const RID &p_framebuffer, BitField<RenderingDevice::DrawFlags> p_draw_flags = (BitField<RenderingDevice::DrawFlags>)0, const PackedColorArray &p_clear_color_values = PackedColorArray(), float p_clear_depth_value = 1.0, uint32_t p_clear_stencil_value = 0, const Rect2 &p_region = Rect2(0, 0, 0, 0), uint32_t p_breadcrumb = 0);
	PackedInt64Array draw_list_begin_split(const RID &p_framebuffer, uint32_t p_splits, RenderingDevice::InitialAction p_initial_color_action, RenderingDevice::FinalAction p_final_color_action, RenderingDevice::InitialAction p_initial_depth_action, RenderingDevice::FinalAction p_final_depth_action, const PackedColorArray &p_clear_color_values = PackedColorArray(), float p_clear_depth = 1.0, uint32_t p_clear_stencil = 0, const Rect2 &p_region = Rect2(0, 0, 0, 0), const TypedArray<RID> &p_storage_textures = {});
	void draw_list_set_blend_constants(int64_t p_draw_list, const Color &p_color);
	void draw_list_bind_render_pipeline(int64_t p_draw_list, const RID &p_render_pipeline);
	void draw_list_bind_uniform_set(int64_t p_draw_list, const RID &p_uniform_set, uint32_t p_set_index);
	void draw_list_bind_vertex_array(int64_t p_draw_list, const RID &p_vertex_array);
	void draw_list_bind_vertex_buffers_format(int64_t p_draw_list, int64_t p_vertex_format, uint32_t p_vertex_count, const TypedArray<RID> &p_vertex_buffers, const PackedInt64Array &p_offsets = PackedInt64Array());
	void draw_list_bind_index_array(int64_t p_draw_list, const RID &p_index_array);
	void draw_list_set_push_constant(int64_t p_draw_list, const PackedByteArray &p_buffer, uint32_t p_size_bytes);
	void draw_list_draw(int64_t p_draw_list, bool p_use_indices, uint32_t p_instances, uint32_t p_procedural_vertex_count = 0);
	void draw_list_draw_indirect(int64_t p_draw_list, bool p_use_indices, const RID &p_buffer, uint32_t p_offset = 0, uint32_t p_draw_count = 1, uint32_t p_stride = 0);
	void draw_list_enable_scissor(int64_t p_draw_list, const Rect2 &p_rect = Rect2(0, 0, 0, 0));
	void draw_list_disable_scissor(int64_t p_draw_list);
	int64_t draw_list_switch_to_next_pass();
	PackedInt64Array draw_list_switch_to_next_pass_split(uint32_t p_splits);
	void draw_list_end();
	int64_t compute_list_begin();
	void compute_list_bind_compute_pipeline(int64_t p_compute_list, const RID &p_compute_pipeline);
	void compute_list_set_push_constant(int64_t p_compute_list, const PackedByteArray &p_buffer, uint32_t p_size_bytes);
	void compute_list_bind_uniform_set(int64_t p_compute_list, const RID &p_uniform_set, uint32_t p_set_index);
	void compute_list_dispatch(int64_t p_compute_list, uint32_t p_x_groups, uint32_t p_y_groups, uint32_t p_z_groups);
	void compute_list_dispatch_indirect(int64_t p_compute_list, const RID &p_buffer, uint32_t p_offset);
	void compute_list_add_barrier(int64_t p_compute_list);
	void compute_list_end();
	void free_rid(const RID &p_rid);
	void capture_timestamp(const String &p_name);
	uint32_t get_captured_timestamps_count() const;
	uint64_t get_captured_timestamps_frame() const;
	uint64_t get_captured_timestamp_gpu_time(uint32_t p_index) const;
	uint64_t get_captured_timestamp_cpu_time(uint32_t p_index) const;
	String get_captured_timestamp_name(uint32_t p_index) const;
	bool has_feature(RenderingDevice::Features p_feature) const;
	uint64_t limit_get(RenderingDevice::Limit p_limit) const;
	uint32_t get_frame_delay() const;
	void submit();
	void sync();
	void barrier(BitField<RenderingDevice::BarrierMask> p_from = (BitField<RenderingDevice::BarrierMask>)32767, BitField<RenderingDevice::BarrierMask> p_to = (BitField<RenderingDevice::BarrierMask>)32767);
	void full_barrier();
	RenderingDevice *create_local_device();
	void set_resource_name(const RID &p_id, const String &p_name);
	void draw_command_begin_label(const String &p_name, const Color &p_color);
	void draw_command_insert_label(const String &p_name, const Color &p_color);
	void draw_command_end_label();
	String get_device_vendor_name() const;
	String get_device_name() const;
	String get_device_pipeline_cache_uuid() const;
	uint64_t get_memory_usage(RenderingDevice::MemoryType p_type) const;
	uint64_t get_driver_resource(RenderingDevice::DriverResource p_resource, const RID &p_rid, uint64_t p_index);
	String get_perf_report() const;
	String get_driver_and_device_memory_report() const;
	String get_tracked_object_name(uint32_t p_type_index) const;
	uint64_t get_tracked_object_type_count() const;
	uint64_t get_driver_total_memory() const;
	uint64_t get_driver_allocation_count() const;
	uint64_t get_driver_memory_by_object_type(uint32_t p_type) const;
	uint64_t get_driver_allocs_by_object_type(uint32_t p_type) const;
	uint64_t get_device_total_memory() const;
	uint64_t get_device_allocation_count() const;
	uint64_t get_device_memory_by_object_type(uint32_t p_type) const;
	uint64_t get_device_allocs_by_object_type(uint32_t p_type) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Object::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(RenderingDevice::DeviceType);
VARIANT_ENUM_CAST(RenderingDevice::DriverResource);
VARIANT_ENUM_CAST(RenderingDevice::DataFormat);
VARIANT_BITFIELD_CAST(RenderingDevice::BarrierMask);
VARIANT_ENUM_CAST(RenderingDevice::TextureType);
VARIANT_ENUM_CAST(RenderingDevice::TextureSamples);
VARIANT_BITFIELD_CAST(RenderingDevice::TextureUsageBits);
VARIANT_ENUM_CAST(RenderingDevice::TextureSwizzle);
VARIANT_ENUM_CAST(RenderingDevice::TextureSliceType);
VARIANT_ENUM_CAST(RenderingDevice::SamplerFilter);
VARIANT_ENUM_CAST(RenderingDevice::SamplerRepeatMode);
VARIANT_ENUM_CAST(RenderingDevice::SamplerBorderColor);
VARIANT_ENUM_CAST(RenderingDevice::VertexFrequency);
VARIANT_ENUM_CAST(RenderingDevice::IndexBufferFormat);
VARIANT_BITFIELD_CAST(RenderingDevice::StorageBufferUsage);
VARIANT_BITFIELD_CAST(RenderingDevice::BufferCreationBits);
VARIANT_ENUM_CAST(RenderingDevice::UniformType);
VARIANT_ENUM_CAST(RenderingDevice::RenderPrimitive);
VARIANT_ENUM_CAST(RenderingDevice::PolygonCullMode);
VARIANT_ENUM_CAST(RenderingDevice::PolygonFrontFace);
VARIANT_ENUM_CAST(RenderingDevice::StencilOperation);
VARIANT_ENUM_CAST(RenderingDevice::CompareOperator);
VARIANT_ENUM_CAST(RenderingDevice::LogicOperation);
VARIANT_ENUM_CAST(RenderingDevice::BlendFactor);
VARIANT_ENUM_CAST(RenderingDevice::BlendOperation);
VARIANT_BITFIELD_CAST(RenderingDevice::PipelineDynamicStateFlags);
VARIANT_ENUM_CAST(RenderingDevice::InitialAction);
VARIANT_ENUM_CAST(RenderingDevice::FinalAction);
VARIANT_ENUM_CAST(RenderingDevice::ShaderStage);
VARIANT_ENUM_CAST(RenderingDevice::ShaderLanguage);
VARIANT_ENUM_CAST(RenderingDevice::PipelineSpecializationConstantType);
VARIANT_ENUM_CAST(RenderingDevice::Features);
VARIANT_ENUM_CAST(RenderingDevice::Limit);
VARIANT_ENUM_CAST(RenderingDevice::MemoryType);
VARIANT_ENUM_CAST(RenderingDevice::BreadcrumbMarker);
VARIANT_BITFIELD_CAST(RenderingDevice::DrawFlags);

