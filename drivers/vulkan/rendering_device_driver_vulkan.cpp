/**************************************************************************/
/*  rendering_device_driver_vulkan.cpp                                    */
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

#include "rendering_device_driver_vulkan.h"

#include "core/config/project_settings.h"
#include "core/io/marshalls.h"
#include "thirdparty/misc/smolv.h"
#include "vulkan_context.h"

#define PRINT_NATIVE_COMMANDS 0

/*****************/
/**** GENERIC ****/
/*****************/

static const VkFormat RD_TO_VK_FORMAT[RDD::DATA_FORMAT_MAX] = {
	VK_FORMAT_R4G4_UNORM_PACK8,
	VK_FORMAT_R4G4B4A4_UNORM_PACK16,
	VK_FORMAT_B4G4R4A4_UNORM_PACK16,
	VK_FORMAT_R5G6B5_UNORM_PACK16,
	VK_FORMAT_B5G6R5_UNORM_PACK16,
	VK_FORMAT_R5G5B5A1_UNORM_PACK16,
	VK_FORMAT_B5G5R5A1_UNORM_PACK16,
	VK_FORMAT_A1R5G5B5_UNORM_PACK16,
	VK_FORMAT_R8_UNORM,
	VK_FORMAT_R8_SNORM,
	VK_FORMAT_R8_USCALED,
	VK_FORMAT_R8_SSCALED,
	VK_FORMAT_R8_UINT,
	VK_FORMAT_R8_SINT,
	VK_FORMAT_R8_SRGB,
	VK_FORMAT_R8G8_UNORM,
	VK_FORMAT_R8G8_SNORM,
	VK_FORMAT_R8G8_USCALED,
	VK_FORMAT_R8G8_SSCALED,
	VK_FORMAT_R8G8_UINT,
	VK_FORMAT_R8G8_SINT,
	VK_FORMAT_R8G8_SRGB,
	VK_FORMAT_R8G8B8_UNORM,
	VK_FORMAT_R8G8B8_SNORM,
	VK_FORMAT_R8G8B8_USCALED,
	VK_FORMAT_R8G8B8_SSCALED,
	VK_FORMAT_R8G8B8_UINT,
	VK_FORMAT_R8G8B8_SINT,
	VK_FORMAT_R8G8B8_SRGB,
	VK_FORMAT_B8G8R8_UNORM,
	VK_FORMAT_B8G8R8_SNORM,
	VK_FORMAT_B8G8R8_USCALED,
	VK_FORMAT_B8G8R8_SSCALED,
	VK_FORMAT_B8G8R8_UINT,
	VK_FORMAT_B8G8R8_SINT,
	VK_FORMAT_B8G8R8_SRGB,
	VK_FORMAT_R8G8B8A8_UNORM,
	VK_FORMAT_R8G8B8A8_SNORM,
	VK_FORMAT_R8G8B8A8_USCALED,
	VK_FORMAT_R8G8B8A8_SSCALED,
	VK_FORMAT_R8G8B8A8_UINT,
	VK_FORMAT_R8G8B8A8_SINT,
	VK_FORMAT_R8G8B8A8_SRGB,
	VK_FORMAT_B8G8R8A8_UNORM,
	VK_FORMAT_B8G8R8A8_SNORM,
	VK_FORMAT_B8G8R8A8_USCALED,
	VK_FORMAT_B8G8R8A8_SSCALED,
	VK_FORMAT_B8G8R8A8_UINT,
	VK_FORMAT_B8G8R8A8_SINT,
	VK_FORMAT_B8G8R8A8_SRGB,
	VK_FORMAT_A8B8G8R8_UNORM_PACK32,
	VK_FORMAT_A8B8G8R8_SNORM_PACK32,
	VK_FORMAT_A8B8G8R8_USCALED_PACK32,
	VK_FORMAT_A8B8G8R8_SSCALED_PACK32,
	VK_FORMAT_A8B8G8R8_UINT_PACK32,
	VK_FORMAT_A8B8G8R8_SINT_PACK32,
	VK_FORMAT_A8B8G8R8_SRGB_PACK32,
	VK_FORMAT_A2R10G10B10_UNORM_PACK32,
	VK_FORMAT_A2R10G10B10_SNORM_PACK32,
	VK_FORMAT_A2R10G10B10_USCALED_PACK32,
	VK_FORMAT_A2R10G10B10_SSCALED_PACK32,
	VK_FORMAT_A2R10G10B10_UINT_PACK32,
	VK_FORMAT_A2R10G10B10_SINT_PACK32,
	VK_FORMAT_A2B10G10R10_UNORM_PACK32,
	VK_FORMAT_A2B10G10R10_SNORM_PACK32,
	VK_FORMAT_A2B10G10R10_USCALED_PACK32,
	VK_FORMAT_A2B10G10R10_SSCALED_PACK32,
	VK_FORMAT_A2B10G10R10_UINT_PACK32,
	VK_FORMAT_A2B10G10R10_SINT_PACK32,
	VK_FORMAT_R16_UNORM,
	VK_FORMAT_R16_SNORM,
	VK_FORMAT_R16_USCALED,
	VK_FORMAT_R16_SSCALED,
	VK_FORMAT_R16_UINT,
	VK_FORMAT_R16_SINT,
	VK_FORMAT_R16_SFLOAT,
	VK_FORMAT_R16G16_UNORM,
	VK_FORMAT_R16G16_SNORM,
	VK_FORMAT_R16G16_USCALED,
	VK_FORMAT_R16G16_SSCALED,
	VK_FORMAT_R16G16_UINT,
	VK_FORMAT_R16G16_SINT,
	VK_FORMAT_R16G16_SFLOAT,
	VK_FORMAT_R16G16B16_UNORM,
	VK_FORMAT_R16G16B16_SNORM,
	VK_FORMAT_R16G16B16_USCALED,
	VK_FORMAT_R16G16B16_SSCALED,
	VK_FORMAT_R16G16B16_UINT,
	VK_FORMAT_R16G16B16_SINT,
	VK_FORMAT_R16G16B16_SFLOAT,
	VK_FORMAT_R16G16B16A16_UNORM,
	VK_FORMAT_R16G16B16A16_SNORM,
	VK_FORMAT_R16G16B16A16_USCALED,
	VK_FORMAT_R16G16B16A16_SSCALED,
	VK_FORMAT_R16G16B16A16_UINT,
	VK_FORMAT_R16G16B16A16_SINT,
	VK_FORMAT_R16G16B16A16_SFLOAT,
	VK_FORMAT_R32_UINT,
	VK_FORMAT_R32_SINT,
	VK_FORMAT_R32_SFLOAT,
	VK_FORMAT_R32G32_UINT,
	VK_FORMAT_R32G32_SINT,
	VK_FORMAT_R32G32_SFLOAT,
	VK_FORMAT_R32G32B32_UINT,
	VK_FORMAT_R32G32B32_SINT,
	VK_FORMAT_R32G32B32_SFLOAT,
	VK_FORMAT_R32G32B32A32_UINT,
	VK_FORMAT_R32G32B32A32_SINT,
	VK_FORMAT_R32G32B32A32_SFLOAT,
	VK_FORMAT_R64_UINT,
	VK_FORMAT_R64_SINT,
	VK_FORMAT_R64_SFLOAT,
	VK_FORMAT_R64G64_UINT,
	VK_FORMAT_R64G64_SINT,
	VK_FORMAT_R64G64_SFLOAT,
	VK_FORMAT_R64G64B64_UINT,
	VK_FORMAT_R64G64B64_SINT,
	VK_FORMAT_R64G64B64_SFLOAT,
	VK_FORMAT_R64G64B64A64_UINT,
	VK_FORMAT_R64G64B64A64_SINT,
	VK_FORMAT_R64G64B64A64_SFLOAT,
	VK_FORMAT_B10G11R11_UFLOAT_PACK32,
	VK_FORMAT_E5B9G9R9_UFLOAT_PACK32,
	VK_FORMAT_D16_UNORM,
	VK_FORMAT_X8_D24_UNORM_PACK32,
	VK_FORMAT_D32_SFLOAT,
	VK_FORMAT_S8_UINT,
	VK_FORMAT_D16_UNORM_S8_UINT,
	VK_FORMAT_D24_UNORM_S8_UINT,
	VK_FORMAT_D32_SFLOAT_S8_UINT,
	VK_FORMAT_BC1_RGB_UNORM_BLOCK,
	VK_FORMAT_BC1_RGB_SRGB_BLOCK,
	VK_FORMAT_BC1_RGBA_UNORM_BLOCK,
	VK_FORMAT_BC1_RGBA_SRGB_BLOCK,
	VK_FORMAT_BC2_UNORM_BLOCK,
	VK_FORMAT_BC2_SRGB_BLOCK,
	VK_FORMAT_BC3_UNORM_BLOCK,
	VK_FORMAT_BC3_SRGB_BLOCK,
	VK_FORMAT_BC4_UNORM_BLOCK,
	VK_FORMAT_BC4_SNORM_BLOCK,
	VK_FORMAT_BC5_UNORM_BLOCK,
	VK_FORMAT_BC5_SNORM_BLOCK,
	VK_FORMAT_BC6H_UFLOAT_BLOCK,
	VK_FORMAT_BC6H_SFLOAT_BLOCK,
	VK_FORMAT_BC7_UNORM_BLOCK,
	VK_FORMAT_BC7_SRGB_BLOCK,
	VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK,
	VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK,
	VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK,
	VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK,
	VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK,
	VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK,
	VK_FORMAT_EAC_R11_UNORM_BLOCK,
	VK_FORMAT_EAC_R11_SNORM_BLOCK,
	VK_FORMAT_EAC_R11G11_UNORM_BLOCK,
	VK_FORMAT_EAC_R11G11_SNORM_BLOCK,
	VK_FORMAT_ASTC_4x4_UNORM_BLOCK,
	VK_FORMAT_ASTC_4x4_SRGB_BLOCK,
	VK_FORMAT_ASTC_5x4_UNORM_BLOCK,
	VK_FORMAT_ASTC_5x4_SRGB_BLOCK,
	VK_FORMAT_ASTC_5x5_UNORM_BLOCK,
	VK_FORMAT_ASTC_5x5_SRGB_BLOCK,
	VK_FORMAT_ASTC_6x5_UNORM_BLOCK,
	VK_FORMAT_ASTC_6x5_SRGB_BLOCK,
	VK_FORMAT_ASTC_6x6_UNORM_BLOCK,
	VK_FORMAT_ASTC_6x6_SRGB_BLOCK,
	VK_FORMAT_ASTC_8x5_UNORM_BLOCK,
	VK_FORMAT_ASTC_8x5_SRGB_BLOCK,
	VK_FORMAT_ASTC_8x6_UNORM_BLOCK,
	VK_FORMAT_ASTC_8x6_SRGB_BLOCK,
	VK_FORMAT_ASTC_8x8_UNORM_BLOCK,
	VK_FORMAT_ASTC_8x8_SRGB_BLOCK,
	VK_FORMAT_ASTC_10x5_UNORM_BLOCK,
	VK_FORMAT_ASTC_10x5_SRGB_BLOCK,
	VK_FORMAT_ASTC_10x6_UNORM_BLOCK,
	VK_FORMAT_ASTC_10x6_SRGB_BLOCK,
	VK_FORMAT_ASTC_10x8_UNORM_BLOCK,
	VK_FORMAT_ASTC_10x8_SRGB_BLOCK,
	VK_FORMAT_ASTC_10x10_UNORM_BLOCK,
	VK_FORMAT_ASTC_10x10_SRGB_BLOCK,
	VK_FORMAT_ASTC_12x10_UNORM_BLOCK,
	VK_FORMAT_ASTC_12x10_SRGB_BLOCK,
	VK_FORMAT_ASTC_12x12_UNORM_BLOCK,
	VK_FORMAT_ASTC_12x12_SRGB_BLOCK,
	VK_FORMAT_G8B8G8R8_422_UNORM,
	VK_FORMAT_B8G8R8G8_422_UNORM,
	VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM,
	VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
	VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM,
	VK_FORMAT_G8_B8R8_2PLANE_422_UNORM,
	VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM,
	VK_FORMAT_R10X6_UNORM_PACK16,
	VK_FORMAT_R10X6G10X6_UNORM_2PACK16,
	VK_FORMAT_R10X6G10X6B10X6A10X6_UNORM_4PACK16,
	VK_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16,
	VK_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16,
	VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16,
	VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16,
	VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16,
	VK_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16,
	VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16,
	VK_FORMAT_R12X4_UNORM_PACK16,
	VK_FORMAT_R12X4G12X4_UNORM_2PACK16,
	VK_FORMAT_R12X4G12X4B12X4A12X4_UNORM_4PACK16,
	VK_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16,
	VK_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16,
	VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16,
	VK_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16,
	VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16,
	VK_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16,
	VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16,
	VK_FORMAT_G16B16G16R16_422_UNORM,
	VK_FORMAT_B16G16R16G16_422_UNORM,
	VK_FORMAT_G16_B16_R16_3PLANE_420_UNORM,
	VK_FORMAT_G16_B16R16_2PLANE_420_UNORM,
	VK_FORMAT_G16_B16_R16_3PLANE_422_UNORM,
	VK_FORMAT_G16_B16R16_2PLANE_422_UNORM,
	VK_FORMAT_G16_B16_R16_3PLANE_444_UNORM,
};

// RDD::CompareOperator == VkCompareOp.
static_assert(ENUM_MEMBERS_EQUAL(RDD::COMPARE_OP_NEVER, VK_COMPARE_OP_NEVER));
static_assert(ENUM_MEMBERS_EQUAL(RDD::COMPARE_OP_LESS, VK_COMPARE_OP_LESS));
static_assert(ENUM_MEMBERS_EQUAL(RDD::COMPARE_OP_EQUAL, VK_COMPARE_OP_EQUAL));
static_assert(ENUM_MEMBERS_EQUAL(RDD::COMPARE_OP_LESS_OR_EQUAL, VK_COMPARE_OP_LESS_OR_EQUAL));
static_assert(ENUM_MEMBERS_EQUAL(RDD::COMPARE_OP_GREATER, VK_COMPARE_OP_GREATER));
static_assert(ENUM_MEMBERS_EQUAL(RDD::COMPARE_OP_NOT_EQUAL, VK_COMPARE_OP_NOT_EQUAL));
static_assert(ENUM_MEMBERS_EQUAL(RDD::COMPARE_OP_GREATER_OR_EQUAL, VK_COMPARE_OP_GREATER_OR_EQUAL));
static_assert(ENUM_MEMBERS_EQUAL(RDD::COMPARE_OP_ALWAYS, VK_COMPARE_OP_ALWAYS));

static_assert(ARRAYS_COMPATIBLE_FIELDWISE(Rect2i, VkRect2D));

/****************/
/**** MEMORY ****/
/****************/

static const uint32_t SMALL_ALLOCATION_MAX_SIZE = 4096;

VmaPool RenderingDeviceDriverVulkan::_find_or_create_small_allocs_pool(uint32_t p_mem_type_index) {
	if (small_allocs_pools.has(p_mem_type_index)) {
		return small_allocs_pools[p_mem_type_index];
	}

	print_verbose("Creating VMA small objects pool for memory type index " + itos(p_mem_type_index));

	VmaPoolCreateInfo pci = {};
	pci.memoryTypeIndex = p_mem_type_index;
	pci.flags = 0;
	pci.blockSize = 0;
	pci.minBlockCount = 0;
	pci.maxBlockCount = SIZE_MAX;
	pci.priority = 0.5f;
	pci.minAllocationAlignment = 0;
	pci.pMemoryAllocateNext = nullptr;
	VmaPool pool = VK_NULL_HANDLE;
	VkResult res = vmaCreatePool(allocator, &pci, &pool);
	small_allocs_pools[p_mem_type_index] = pool; // Don't try to create it again if failed the first time.
	ERR_FAIL_COND_V_MSG(res, pool, "vmaCreatePool failed with error " + itos(res) + ".");

	return pool;
}

/*****************/
/**** BUFFERS ****/
/*****************/

// RDD::BufferUsageBits == VkBufferUsageFlagBits.
static_assert(ENUM_MEMBERS_EQUAL(RDD::BUFFER_USAGE_TRANSFER_FROM_BIT, VK_BUFFER_USAGE_TRANSFER_SRC_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BUFFER_USAGE_TRANSFER_TO_BIT, VK_BUFFER_USAGE_TRANSFER_DST_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BUFFER_USAGE_TEXEL_BIT, VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BUFFER_USAGE_UNIFORM_BIT, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BUFFER_USAGE_STORAGE_BIT, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BUFFER_USAGE_INDEX_BIT, VK_BUFFER_USAGE_INDEX_BUFFER_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BUFFER_USAGE_VERTEX_BIT, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BUFFER_USAGE_INDIRECT_BIT, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT));

RDD::BufferID RenderingDeviceDriverVulkan::buffer_create(uint64_t p_size, BitField<BufferUsageBits> p_usage, MemoryAllocationType p_allocation_type) {
	VkBufferCreateInfo create_info = {};
	create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	create_info.size = p_size;
	create_info.usage = p_usage;
	create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VmaAllocationCreateInfo alloc_create_info = {};
	switch (p_allocation_type) {
		case MEMORY_ALLOCATION_TYPE_CPU: {
			bool is_src = p_usage.has_flag(BUFFER_USAGE_TRANSFER_FROM_BIT);
			bool is_dst = p_usage.has_flag(BUFFER_USAGE_TRANSFER_TO_BIT);
			if (is_src && !is_dst) {
				// Looks like a staging buffer: CPU maps, writes sequentially, then GPU copies to VRAM.
				alloc_create_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
			}
			if (is_dst && !is_src) {
				// Looks like a readback buffer: GPU copies from VRAM, then CPU maps and reads.
				alloc_create_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
			}
			alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
			alloc_create_info.requiredFlags = (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
		} break;
		case MEMORY_ALLOCATION_TYPE_GPU: {
			alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
			if (p_size <= SMALL_ALLOCATION_MAX_SIZE) {
				uint32_t mem_type_index = 0;
				vmaFindMemoryTypeIndexForBufferInfo(allocator, &create_info, &alloc_create_info, &mem_type_index);
				alloc_create_info.pool = _find_or_create_small_allocs_pool(mem_type_index);
			}
		} break;
	}

	VkBuffer vk_buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = nullptr;
	VmaAllocationInfo alloc_info = {};
	VkResult err = vmaCreateBuffer(allocator, &create_info, &alloc_create_info, &vk_buffer, &allocation, &alloc_info);
	ERR_FAIL_COND_V_MSG(err, BufferID(), "Can't create buffer of size: " + itos(p_size) + ", error " + itos(err) + ".");

	// Bookkeep.

	BufferInfo *buf_info = VersatileResource::allocate<BufferInfo>(resources_allocator);
	buf_info->vk_buffer = vk_buffer;
	buf_info->allocation.handle = allocation;
	buf_info->allocation.size = alloc_info.size;
	buf_info->size = p_size;

	return BufferID(buf_info);
}

bool RenderingDeviceDriverVulkan::buffer_set_texel_format(BufferID p_buffer, DataFormat p_format) {
	BufferInfo *buf_info = (BufferInfo *)p_buffer.id;

	DEV_ASSERT(!buf_info->vk_view);

	VkBufferViewCreateInfo view_create_info = {};
	view_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO;
	view_create_info.buffer = buf_info->vk_buffer;
	view_create_info.format = RD_TO_VK_FORMAT[p_format];
	view_create_info.range = buf_info->allocation.size;

	VkResult res = vkCreateBufferView(vk_device, &view_create_info, nullptr, &buf_info->vk_view);
	ERR_FAIL_COND_V_MSG(res, false, "Unable to create buffer view, error " + itos(res) + ".");

	return true;
}

void RenderingDeviceDriverVulkan::buffer_free(BufferID p_buffer) {
	BufferInfo *buf_info = (BufferInfo *)p_buffer.id;
	if (buf_info->vk_view) {
		vkDestroyBufferView(vk_device, buf_info->vk_view, nullptr);
	}
	vmaDestroyBuffer(allocator, buf_info->vk_buffer, buf_info->allocation.handle);
	VersatileResource::free(resources_allocator, buf_info);
}

uint64_t RenderingDeviceDriverVulkan::buffer_get_allocation_size(BufferID p_buffer) {
	const BufferInfo *buf_info = (const BufferInfo *)p_buffer.id;
	return buf_info->allocation.size;
}

uint8_t *RenderingDeviceDriverVulkan::buffer_map(BufferID p_buffer) {
	const BufferInfo *buf_info = (const BufferInfo *)p_buffer.id;
	void *data_ptr = nullptr;
	VkResult err = vmaMapMemory(allocator, buf_info->allocation.handle, &data_ptr);
	ERR_FAIL_COND_V_MSG(err, nullptr, "vmaMapMemory failed with error " + itos(err) + ".");
	return (uint8_t *)data_ptr;
}

void RenderingDeviceDriverVulkan::buffer_unmap(BufferID p_buffer) {
	const BufferInfo *buf_info = (const BufferInfo *)p_buffer.id;
	vmaUnmapMemory(allocator, buf_info->allocation.handle);
}

/*****************/
/**** TEXTURE ****/
/*****************/

static const VkImageType RD_TEX_TYPE_TO_VK_IMG_TYPE[RDD::TEXTURE_TYPE_MAX] = {
	VK_IMAGE_TYPE_1D,
	VK_IMAGE_TYPE_2D,
	VK_IMAGE_TYPE_3D,
	VK_IMAGE_TYPE_2D,
	VK_IMAGE_TYPE_1D,
	VK_IMAGE_TYPE_2D,
	VK_IMAGE_TYPE_2D,
};

static const VkSampleCountFlagBits RD_TO_VK_SAMPLE_COUNT[RDD::TEXTURE_SAMPLES_MAX] = {
	VK_SAMPLE_COUNT_1_BIT,
	VK_SAMPLE_COUNT_2_BIT,
	VK_SAMPLE_COUNT_4_BIT,
	VK_SAMPLE_COUNT_8_BIT,
	VK_SAMPLE_COUNT_16_BIT,
	VK_SAMPLE_COUNT_32_BIT,
	VK_SAMPLE_COUNT_64_BIT,
};

// RDD::TextureType == VkImageViewType.
static_assert(ENUM_MEMBERS_EQUAL(RDD::TEXTURE_TYPE_1D, VK_IMAGE_VIEW_TYPE_1D));
static_assert(ENUM_MEMBERS_EQUAL(RDD::TEXTURE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D));
static_assert(ENUM_MEMBERS_EQUAL(RDD::TEXTURE_TYPE_3D, VK_IMAGE_VIEW_TYPE_3D));
static_assert(ENUM_MEMBERS_EQUAL(RDD::TEXTURE_TYPE_CUBE, VK_IMAGE_VIEW_TYPE_CUBE));
static_assert(ENUM_MEMBERS_EQUAL(RDD::TEXTURE_TYPE_1D_ARRAY, VK_IMAGE_VIEW_TYPE_1D_ARRAY));
static_assert(ENUM_MEMBERS_EQUAL(RDD::TEXTURE_TYPE_2D_ARRAY, VK_IMAGE_VIEW_TYPE_2D_ARRAY));
static_assert(ENUM_MEMBERS_EQUAL(RDD::TEXTURE_TYPE_CUBE_ARRAY, VK_IMAGE_VIEW_TYPE_CUBE_ARRAY));

// RDD::TextureSwizzle == VkComponentSwizzle.
static_assert(ENUM_MEMBERS_EQUAL(RDD::TEXTURE_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY));
static_assert(ENUM_MEMBERS_EQUAL(RDD::TEXTURE_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ZERO));
static_assert(ENUM_MEMBERS_EQUAL(RDD::TEXTURE_SWIZZLE_ONE, VK_COMPONENT_SWIZZLE_ONE));
static_assert(ENUM_MEMBERS_EQUAL(RDD::TEXTURE_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R));
static_assert(ENUM_MEMBERS_EQUAL(RDD::TEXTURE_SWIZZLE_G, VK_COMPONENT_SWIZZLE_G));
static_assert(ENUM_MEMBERS_EQUAL(RDD::TEXTURE_SWIZZLE_B, VK_COMPONENT_SWIZZLE_B));
static_assert(ENUM_MEMBERS_EQUAL(RDD::TEXTURE_SWIZZLE_A, VK_COMPONENT_SWIZZLE_A));

// RDD::TextureLayout == VkImageLayout.
static_assert(ENUM_MEMBERS_EQUAL(RDD::TEXTURE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED));
static_assert(ENUM_MEMBERS_EQUAL(RDD::TEXTURE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL));
static_assert(ENUM_MEMBERS_EQUAL(RDD::TEXTURE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL));
static_assert(ENUM_MEMBERS_EQUAL(RDD::TEXTURE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL));
static_assert(ENUM_MEMBERS_EQUAL(RDD::TEXTURE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL));
static_assert(ENUM_MEMBERS_EQUAL(RDD::TEXTURE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
static_assert(ENUM_MEMBERS_EQUAL(RDD::TEXTURE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL));
static_assert(ENUM_MEMBERS_EQUAL(RDD::TEXTURE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL));
static_assert(ENUM_MEMBERS_EQUAL(RDD::TEXTURE_LAYOUT_PREINITIALIZED, VK_IMAGE_LAYOUT_PREINITIALIZED));
static_assert(ENUM_MEMBERS_EQUAL(RDD::TEXTURE_LAYOUT_VRS_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR));

// RDD::TextureAspectBits == VkImageAspectFlagBits.
static_assert(ENUM_MEMBERS_EQUAL(RDD::TEXTURE_ASPECT_COLOR_BIT, VK_IMAGE_ASPECT_COLOR_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::TEXTURE_ASPECT_DEPTH_BIT, VK_IMAGE_ASPECT_DEPTH_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::TEXTURE_ASPECT_STENCIL_BIT, VK_IMAGE_ASPECT_STENCIL_BIT));

VkSampleCountFlagBits RenderingDeviceDriverVulkan::_ensure_supported_sample_count(TextureSamples p_requested_sample_count) {
	VkSampleCountFlags sample_count_flags = (context->get_device_limits().framebufferColorSampleCounts & limits.framebufferDepthSampleCounts);

	if ((sample_count_flags & RD_TO_VK_SAMPLE_COUNT[p_requested_sample_count])) {
		// The requested sample count is supported.
		return RD_TO_VK_SAMPLE_COUNT[p_requested_sample_count];
	} else {
		// Find the closest lower supported sample count.
		VkSampleCountFlagBits sample_count = RD_TO_VK_SAMPLE_COUNT[p_requested_sample_count];
		while (sample_count > VK_SAMPLE_COUNT_1_BIT) {
			if (sample_count_flags & sample_count) {
				return sample_count;
			}
			sample_count = (VkSampleCountFlagBits)(sample_count >> 1);
		}
	}
	return VK_SAMPLE_COUNT_1_BIT;
}

RDD::TextureID RenderingDeviceDriverVulkan::texture_create(const TextureFormat &p_format, const TextureView &p_view) {
	VkImageCreateInfo create_info = {};
	create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;

	if (p_format.shareable_formats.size()) {
		create_info.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;

		if (context->is_device_extension_enabled(VK_KHR_IMAGE_FORMAT_LIST_EXTENSION_NAME)) {
			VkFormat *vk_allowed_formats = ALLOCA_ARRAY(VkFormat, p_format.shareable_formats.size());
			for (int i = 0; i < p_format.shareable_formats.size(); i++) {
				vk_allowed_formats[i] = RD_TO_VK_FORMAT[p_format.shareable_formats[i]];
			}

			VkImageFormatListCreateInfoKHR *format_list_create_info = ALLOCA_SINGLE(VkImageFormatListCreateInfoKHR);
			*format_list_create_info = {};
			format_list_create_info->sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO_KHR;
			format_list_create_info->viewFormatCount = p_format.shareable_formats.size();
			format_list_create_info->pViewFormats = vk_allowed_formats;

			create_info.pNext = format_list_create_info;
		}
	}

	if (p_format.texture_type == TEXTURE_TYPE_CUBE || p_format.texture_type == TEXTURE_TYPE_CUBE_ARRAY) {
		create_info.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
	}
	/*if (p_format.texture_type == TEXTURE_TYPE_2D || p_format.texture_type == TEXTURE_TYPE_2D_ARRAY) {
		create_info.flags |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
	}*/

	create_info.imageType = RD_TEX_TYPE_TO_VK_IMG_TYPE[p_format.texture_type];

	create_info.format = RD_TO_VK_FORMAT[p_format.format];

	create_info.extent.width = p_format.width;
	create_info.extent.height = p_format.height;
	create_info.extent.depth = p_format.depth;

	create_info.mipLevels = p_format.mipmaps;
	create_info.arrayLayers = p_format.array_layers;

	create_info.samples = _ensure_supported_sample_count(p_format.samples);
	create_info.tiling = (p_format.usage_bits & TEXTURE_USAGE_CPU_READ_BIT) ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL;

	// Usage.
	if ((p_format.usage_bits & TEXTURE_USAGE_SAMPLING_BIT)) {
		create_info.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
	}
	if ((p_format.usage_bits & TEXTURE_USAGE_STORAGE_BIT)) {
		create_info.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
	}
	if ((p_format.usage_bits & TEXTURE_USAGE_COLOR_ATTACHMENT_BIT)) {
		create_info.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	}
	if ((p_format.usage_bits & TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) {
		create_info.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
	}
	if ((p_format.usage_bits & TEXTURE_USAGE_INPUT_ATTACHMENT_BIT)) {
		create_info.usage |= VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
	}
	if ((p_format.usage_bits & TEXTURE_USAGE_VRS_ATTACHMENT_BIT)) {
		create_info.usage |= VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR;
	}
	if ((p_format.usage_bits & TEXTURE_USAGE_CAN_UPDATE_BIT)) {
		create_info.usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	}
	if ((p_format.usage_bits & TEXTURE_USAGE_CAN_COPY_FROM_BIT)) {
		create_info.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	}
	if ((p_format.usage_bits & TEXTURE_USAGE_CAN_COPY_TO_BIT)) {
		create_info.usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	}

	create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	// Allocate memory.

	uint32_t width = 0, height = 0;
	uint32_t image_size = get_image_format_required_size(p_format.format, p_format.width, p_format.height, p_format.depth, p_format.mipmaps, &width, &height);

	VmaAllocationCreateInfo alloc_create_info = {};
	alloc_create_info.flags = (p_format.usage_bits & TEXTURE_USAGE_CPU_READ_BIT) ? VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT : 0;
	alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	if (image_size <= SMALL_ALLOCATION_MAX_SIZE) {
		uint32_t mem_type_index = 0;
		vmaFindMemoryTypeIndexForImageInfo(allocator, &create_info, &alloc_create_info, &mem_type_index);
		alloc_create_info.pool = _find_or_create_small_allocs_pool(mem_type_index);
	}

	// Create.

	VkImage vk_image = VK_NULL_HANDLE;
	VmaAllocation allocation = nullptr;
	VmaAllocationInfo alloc_info = {};
	VkResult err = vmaCreateImage(allocator, &create_info, &alloc_create_info, &vk_image, &allocation, &alloc_info);
	ERR_FAIL_COND_V_MSG(err, TextureID(), "vmaCreateImage failed with error " + itos(err) + ".");

	// Create view.

	VkImageViewCreateInfo image_view_create_info = {};
	image_view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	image_view_create_info.image = vk_image;
	image_view_create_info.viewType = (VkImageViewType)p_format.texture_type;
	image_view_create_info.format = RD_TO_VK_FORMAT[p_view.format];
	image_view_create_info.components.r = (VkComponentSwizzle)p_view.swizzle_r;
	image_view_create_info.components.g = (VkComponentSwizzle)p_view.swizzle_g;
	image_view_create_info.components.b = (VkComponentSwizzle)p_view.swizzle_b;
	image_view_create_info.components.a = (VkComponentSwizzle)p_view.swizzle_a;
	image_view_create_info.subresourceRange.levelCount = create_info.mipLevels;
	image_view_create_info.subresourceRange.layerCount = create_info.arrayLayers;
	if ((p_format.usage_bits & TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) {
		image_view_create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
	} else {
		image_view_create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	}

	VkImageView vk_image_view = VK_NULL_HANDLE;
	err = vkCreateImageView(vk_device, &image_view_create_info, nullptr, &vk_image_view);
	if (err) {
		vmaDestroyImage(allocator, vk_image, allocation);
		ERR_FAIL_COND_V_MSG(err, TextureID(), "vkCreateImageView failed with error " + itos(err) + ".");
	}

	// Bookkeep.

	TextureInfo *tex_info = VersatileResource::allocate<TextureInfo>(resources_allocator);
	tex_info->vk_view = vk_image_view;
	tex_info->rd_format = p_format.format;
	tex_info->vk_create_info = create_info;
	tex_info->vk_view_create_info = image_view_create_info;
	tex_info->allocation.handle = allocation;
	vmaGetAllocationInfo(allocator, tex_info->allocation.handle, &tex_info->allocation.info);

#if PRINT_NATIVE_COMMANDS
	print_line(vformat("vkCreateImageView: 0x%uX for 0x%uX", uint64_t(vk_image_view), uint64_t(vk_image)));
#endif

	return TextureID(tex_info);
}

RDD::TextureID RenderingDeviceDriverVulkan::texture_create_from_extension(uint64_t p_native_texture, TextureType p_type, DataFormat p_format, uint32_t p_array_layers, bool p_depth_stencil) {
	VkImage vk_image = (VkImage)p_native_texture;

	// We only need to create a view into the already existing natively-provided texture.

	VkImageViewCreateInfo image_view_create_info = {};
	image_view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	image_view_create_info.image = vk_image;
	image_view_create_info.viewType = (VkImageViewType)p_type;
	image_view_create_info.format = RD_TO_VK_FORMAT[p_format];
	image_view_create_info.components.r = VK_COMPONENT_SWIZZLE_R;
	image_view_create_info.components.g = VK_COMPONENT_SWIZZLE_G;
	image_view_create_info.components.b = VK_COMPONENT_SWIZZLE_B;
	image_view_create_info.components.a = VK_COMPONENT_SWIZZLE_A;
	image_view_create_info.subresourceRange.levelCount = 1;
	image_view_create_info.subresourceRange.layerCount = p_array_layers;
	image_view_create_info.subresourceRange.aspectMask = p_depth_stencil ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;

	VkImageView vk_image_view = VK_NULL_HANDLE;
	VkResult err = vkCreateImageView(vk_device, &image_view_create_info, nullptr, &vk_image_view);
	if (err) {
		ERR_FAIL_COND_V_MSG(err, TextureID(), "vkCreateImageView failed with error " + itos(err) + ".");
	}

	// Bookkeep.

	TextureInfo *tex_info = VersatileResource::allocate<TextureInfo>(resources_allocator);
	tex_info->vk_view = vk_image_view;
	tex_info->rd_format = p_format;
	tex_info->vk_view_create_info = image_view_create_info;

	return TextureID(tex_info);
}

RDD::TextureID RenderingDeviceDriverVulkan::texture_create_shared(TextureID p_original_texture, const TextureView &p_view) {
	const TextureInfo *owner_tex_info = (const TextureInfo *)p_original_texture.id;
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND_V(!owner_tex_info->allocation.handle, TextureID());
#endif

	VkImageViewCreateInfo image_view_create_info = owner_tex_info->vk_view_create_info;
	image_view_create_info.format = RD_TO_VK_FORMAT[p_view.format];
	image_view_create_info.components.r = (VkComponentSwizzle)p_view.swizzle_r;
	image_view_create_info.components.g = (VkComponentSwizzle)p_view.swizzle_g;
	image_view_create_info.components.b = (VkComponentSwizzle)p_view.swizzle_b;
	image_view_create_info.components.a = (VkComponentSwizzle)p_view.swizzle_a;

	if (context->is_device_extension_enabled(VK_KHR_MAINTENANCE_2_EXTENSION_NAME)) {
		// May need to make VK_KHR_maintenance2 mandatory and thus has Vulkan 1.1 be our minimum supported version
		// if we require setting this information. Vulkan 1.0 may simply not care.
		if (image_view_create_info.format != owner_tex_info->vk_view_create_info.format) {
			VkImageViewUsageCreateInfo *usage_info = ALLOCA_SINGLE(VkImageViewUsageCreateInfo);
			*usage_info = {};
			usage_info->sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO;
			usage_info->usage = owner_tex_info->vk_create_info.usage;

			// Certain features may not be available for the format of the view.
			{
				VkFormatProperties properties = {};
				vkGetPhysicalDeviceFormatProperties(context->get_physical_device(), RD_TO_VK_FORMAT[p_view.format], &properties);
				const VkFormatFeatureFlags &supported_flags = owner_tex_info->vk_create_info.tiling == VK_IMAGE_TILING_LINEAR ? properties.linearTilingFeatures : properties.optimalTilingFeatures;
				if ((usage_info->usage & VK_IMAGE_USAGE_STORAGE_BIT) && !(supported_flags & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) {
					usage_info->usage &= ~VK_IMAGE_USAGE_STORAGE_BIT;
				}
				if ((usage_info->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) && !(supported_flags & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)) {
					usage_info->usage &= ~VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
				}
			}

			image_view_create_info.pNext = usage_info;
		}
	}

	VkImageView new_vk_image_view = VK_NULL_HANDLE;
	VkResult err = vkCreateImageView(vk_device, &image_view_create_info, nullptr, &new_vk_image_view);
	ERR_FAIL_COND_V_MSG(err, TextureID(), "vkCreateImageView failed with error " + itos(err) + ".");

	// Bookkeep.

	TextureInfo *tex_info = VersatileResource::allocate<TextureInfo>(resources_allocator);
	*tex_info = *owner_tex_info;
	tex_info->vk_view = new_vk_image_view;
	tex_info->vk_view_create_info = image_view_create_info;
	tex_info->allocation = {};

#if PRINT_NATIVE_COMMANDS
	print_line(vformat("vkCreateImageView: 0x%uX for 0x%uX", uint64_t(new_vk_image_view), uint64_t(owner_tex_info->vk_view_create_info.image)));
#endif

	return TextureID(tex_info);
}

RDD::TextureID RenderingDeviceDriverVulkan::texture_create_shared_from_slice(TextureID p_original_texture, const TextureView &p_view, TextureSliceType p_slice_type, uint32_t p_layer, uint32_t p_layers, uint32_t p_mipmap, uint32_t p_mipmaps) {
	const TextureInfo *owner_tex_info = (const TextureInfo *)p_original_texture.id;
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND_V(!owner_tex_info->allocation.handle, TextureID());
#endif

	VkImageViewCreateInfo image_view_create_info = owner_tex_info->vk_view_create_info;
	switch (p_slice_type) {
		case TEXTURE_SLICE_2D: {
			image_view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
		} break;
		case TEXTURE_SLICE_3D: {
			image_view_create_info.viewType = VK_IMAGE_VIEW_TYPE_3D;
		} break;
		case TEXTURE_SLICE_CUBEMAP: {
			image_view_create_info.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
		} break;
		case TEXTURE_SLICE_2D_ARRAY: {
			image_view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
		} break;
		default: {
			return TextureID(nullptr);
		}
	}
	image_view_create_info.format = RD_TO_VK_FORMAT[p_view.format];
	image_view_create_info.components.r = (VkComponentSwizzle)p_view.swizzle_r;
	image_view_create_info.components.g = (VkComponentSwizzle)p_view.swizzle_g;
	image_view_create_info.components.b = (VkComponentSwizzle)p_view.swizzle_b;
	image_view_create_info.components.a = (VkComponentSwizzle)p_view.swizzle_a;
	image_view_create_info.subresourceRange.baseMipLevel = p_mipmap;
	image_view_create_info.subresourceRange.levelCount = p_mipmaps;
	image_view_create_info.subresourceRange.baseArrayLayer = p_layer;
	image_view_create_info.subresourceRange.layerCount = p_layers;

	VkImageView new_vk_image_view = VK_NULL_HANDLE;
	VkResult err = vkCreateImageView(vk_device, &image_view_create_info, nullptr, &new_vk_image_view);
	ERR_FAIL_COND_V_MSG(err, TextureID(), "vkCreateImageView failed with error " + itos(err) + ".");

	// Bookkeep.

	TextureInfo *tex_info = VersatileResource::allocate<TextureInfo>(resources_allocator);
	*tex_info = *owner_tex_info;
	tex_info->vk_view = new_vk_image_view;
	tex_info->vk_view_create_info = image_view_create_info;
	tex_info->allocation = {};

#if PRINT_NATIVE_COMMANDS
	print_line(vformat("vkCreateImageView: 0x%uX for 0x%uX (%d %d %d %d)", uint64_t(new_vk_image_view), uint64_t(owner_tex_info->vk_view_create_info.image), p_mipmap, p_mipmaps, p_layer, p_layers));
#endif

	return TextureID(tex_info);
}

void RenderingDeviceDriverVulkan::texture_free(TextureID p_texture) {
	TextureInfo *tex_info = (TextureInfo *)p_texture.id;
	vkDestroyImageView(vk_device, tex_info->vk_view, nullptr);
	if (tex_info->allocation.handle) {
		vmaDestroyImage(allocator, tex_info->vk_view_create_info.image, tex_info->allocation.handle);
	}
	VersatileResource::free(resources_allocator, tex_info);
}

uint64_t RenderingDeviceDriverVulkan::texture_get_allocation_size(TextureID p_texture) {
	const TextureInfo *tex_info = (const TextureInfo *)p_texture.id;
	return tex_info->allocation.info.size;
}

void RenderingDeviceDriverVulkan::texture_get_copyable_layout(TextureID p_texture, const TextureSubresource &p_subresource, TextureCopyableLayout *r_layout) {
	const TextureInfo *tex_info = (const TextureInfo *)p_texture.id;

	*r_layout = {};

	if (tex_info->vk_create_info.tiling == VK_IMAGE_TILING_LINEAR) {
		VkImageSubresource vk_subres = {};
		vk_subres.aspectMask = (VkImageAspectFlags)(1 << p_subresource.aspect);
		vk_subres.arrayLayer = p_subresource.layer;
		vk_subres.mipLevel = p_subresource.mipmap;

		VkSubresourceLayout vk_layout = {};
		vkGetImageSubresourceLayout(vk_device, tex_info->vk_view_create_info.image, &vk_subres, &vk_layout);

		r_layout->offset = vk_layout.offset;
		r_layout->size = vk_layout.size;
		r_layout->row_pitch = vk_layout.rowPitch;
		r_layout->depth_pitch = vk_layout.depthPitch;
		r_layout->layer_pitch = vk_layout.arrayPitch;
	} else {
		// Tight.
		uint32_t w = tex_info->vk_create_info.extent.width;
		uint32_t h = tex_info->vk_create_info.extent.height;
		uint32_t d = tex_info->vk_create_info.extent.depth;
		if (p_subresource.mipmap > 0) {
			r_layout->offset = get_image_format_required_size(tex_info->rd_format, w, h, d, p_subresource.mipmap);
		}
		for (uint32_t i = 0; i < p_subresource.mipmap; i++) {
			w = MAX(1u, w >> 1);
			h = MAX(1u, h >> 1);
			d = MAX(1u, d >> 1);
		}
		uint32_t bw = 0, bh = 0;
		get_compressed_image_format_block_dimensions(tex_info->rd_format, bw, bh);
		uint32_t sbw = 0, sbh = 0;
		r_layout->size = get_image_format_required_size(tex_info->rd_format, w, h, d, 1, &sbw, &sbh);
		r_layout->row_pitch = r_layout->size / ((sbh / bh) * d);
		r_layout->depth_pitch = r_layout->size / d;
		r_layout->layer_pitch = r_layout->size / tex_info->vk_create_info.arrayLayers;
	}
}

uint8_t *RenderingDeviceDriverVulkan::texture_map(TextureID p_texture, const TextureSubresource &p_subresource) {
	const TextureInfo *tex_info = (const TextureInfo *)p_texture.id;

	VkImageSubresource vk_subres = {};
	vk_subres.aspectMask = (VkImageAspectFlags)(1 << p_subresource.aspect);
	vk_subres.arrayLayer = p_subresource.layer;
	vk_subres.mipLevel = p_subresource.mipmap;

	VkSubresourceLayout vk_layout = {};
	vkGetImageSubresourceLayout(vk_device, tex_info->vk_view_create_info.image, &vk_subres, &vk_layout);

	void *data_ptr = nullptr;
	VkResult err = vkMapMemory(
			vk_device,
			tex_info->allocation.info.deviceMemory,
			tex_info->allocation.info.offset + vk_layout.offset,
			vk_layout.size,
			0,
			&data_ptr);

	vmaMapMemory(allocator, tex_info->allocation.handle, &data_ptr);
	ERR_FAIL_COND_V_MSG(err, nullptr, "vkMapMemory failed with error " + itos(err) + ".");
	return (uint8_t *)data_ptr;
}

void RenderingDeviceDriverVulkan::texture_unmap(TextureID p_texture) {
	const TextureInfo *tex_info = (const TextureInfo *)p_texture.id;
	vkUnmapMemory(vk_device, tex_info->allocation.info.deviceMemory);
}

BitField<RDD::TextureUsageBits> RenderingDeviceDriverVulkan::texture_get_usages_supported_by_format(DataFormat p_format, bool p_cpu_readable) {
	VkFormatProperties properties = {};
	vkGetPhysicalDeviceFormatProperties(context->get_physical_device(), RD_TO_VK_FORMAT[p_format], &properties);

	const VkFormatFeatureFlags &flags = p_cpu_readable ? properties.linearTilingFeatures : properties.optimalTilingFeatures;

	// Everything supported by default makes an all-or-nothing check easier for the caller.
	BitField<RDD::TextureUsageBits> supported = INT64_MAX;

	if (!(flags & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) {
		supported.clear_flag(TEXTURE_USAGE_SAMPLING_BIT);
	}
	if (!(flags & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)) {
		supported.clear_flag(TEXTURE_USAGE_COLOR_ATTACHMENT_BIT);
	}
	if (!(flags & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) {
		supported.clear_flag(TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
	}
	if (!(flags & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) {
		supported.clear_flag(TEXTURE_USAGE_STORAGE_BIT);
	}
	if (!(flags & VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT)) {
		supported.clear_flag(TEXTURE_USAGE_STORAGE_ATOMIC_BIT);
	}
	// Validation via VK_FORMAT_FEATURE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR fails if VRS attachment is not supported.
	if (p_format != DATA_FORMAT_R8_UINT) {
		supported.clear_flag(TEXTURE_USAGE_VRS_ATTACHMENT_BIT);
	}

	return supported;
}

/*****************/
/**** SAMPLER ****/
/*****************/

// RDD::SamplerRepeatMode == VkSamplerAddressMode.
static_assert(ENUM_MEMBERS_EQUAL(RDD::SAMPLER_REPEAT_MODE_REPEAT, VK_SAMPLER_ADDRESS_MODE_REPEAT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::SAMPLER_REPEAT_MODE_MIRRORED_REPEAT, VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE));
static_assert(ENUM_MEMBERS_EQUAL(RDD::SAMPLER_REPEAT_MODE_CLAMP_TO_BORDER, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER));
static_assert(ENUM_MEMBERS_EQUAL(RDD::SAMPLER_REPEAT_MODE_MIRROR_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE));

// RDD::SamplerBorderColor == VkBorderColor.
static_assert(ENUM_MEMBERS_EQUAL(RDD::SAMPLER_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK, VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK));
static_assert(ENUM_MEMBERS_EQUAL(RDD::SAMPLER_BORDER_COLOR_INT_TRANSPARENT_BLACK, VK_BORDER_COLOR_INT_TRANSPARENT_BLACK));
static_assert(ENUM_MEMBERS_EQUAL(RDD::SAMPLER_BORDER_COLOR_FLOAT_OPAQUE_BLACK, VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK));
static_assert(ENUM_MEMBERS_EQUAL(RDD::SAMPLER_BORDER_COLOR_INT_OPAQUE_BLACK, VK_BORDER_COLOR_INT_OPAQUE_BLACK));
static_assert(ENUM_MEMBERS_EQUAL(RDD::SAMPLER_BORDER_COLOR_FLOAT_OPAQUE_WHITE, VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE));
static_assert(ENUM_MEMBERS_EQUAL(RDD::SAMPLER_BORDER_COLOR_INT_OPAQUE_WHITE, VK_BORDER_COLOR_INT_OPAQUE_WHITE));

RDD::SamplerID RenderingDeviceDriverVulkan::sampler_create(const SamplerState &p_state) {
	VkSamplerCreateInfo sampler_create_info = {};
	sampler_create_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	sampler_create_info.pNext = nullptr;
	sampler_create_info.flags = 0;
	sampler_create_info.magFilter = p_state.mag_filter == SAMPLER_FILTER_LINEAR ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
	sampler_create_info.minFilter = p_state.min_filter == SAMPLER_FILTER_LINEAR ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
	sampler_create_info.mipmapMode = p_state.mip_filter == SAMPLER_FILTER_LINEAR ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
	sampler_create_info.addressModeU = (VkSamplerAddressMode)p_state.repeat_u;
	sampler_create_info.addressModeV = (VkSamplerAddressMode)p_state.repeat_v;
	sampler_create_info.addressModeW = (VkSamplerAddressMode)p_state.repeat_w;
	sampler_create_info.mipLodBias = p_state.lod_bias;
	sampler_create_info.anisotropyEnable = p_state.use_anisotropy && context->get_physical_device_features().samplerAnisotropy;
	sampler_create_info.maxAnisotropy = p_state.anisotropy_max;
	sampler_create_info.compareEnable = p_state.enable_compare;
	sampler_create_info.compareOp = (VkCompareOp)p_state.compare_op;
	sampler_create_info.minLod = p_state.min_lod;
	sampler_create_info.maxLod = p_state.max_lod;
	sampler_create_info.borderColor = (VkBorderColor)p_state.border_color;
	sampler_create_info.unnormalizedCoordinates = p_state.unnormalized_uvw;

	VkSampler vk_sampler = VK_NULL_HANDLE;
	VkResult res = vkCreateSampler(vk_device, &sampler_create_info, nullptr, &vk_sampler);
	ERR_FAIL_COND_V_MSG(res, SamplerID(), "vkCreateSampler failed with error " + itos(res) + ".");

	return SamplerID(vk_sampler);
}

void RenderingDeviceDriverVulkan::sampler_free(SamplerID p_sampler) {
	vkDestroySampler(vk_device, (VkSampler)p_sampler.id, nullptr);
}

bool RenderingDeviceDriverVulkan::sampler_is_format_supported_for_filter(DataFormat p_format, SamplerFilter p_filter) {
	switch (p_filter) {
		case RD::SAMPLER_FILTER_NEAREST: {
			return true;
		}
		case RD::SAMPLER_FILTER_LINEAR: {
			VkFormatProperties properties = {};
			vkGetPhysicalDeviceFormatProperties(context->get_physical_device(), RD_TO_VK_FORMAT[p_format], &properties);
			return (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT);
		}
	}
	return false;
}

/**********************/
/**** VERTEX ARRAY ****/
/**********************/

RDD::VertexFormatID RenderingDeviceDriverVulkan::vertex_format_create(VectorView<VertexAttribute> p_vertex_attribs) {
	// Pre-bookkeep.
	VertexFormatInfo *vf_info = VersatileResource::allocate<VertexFormatInfo>(resources_allocator);

	vf_info->vk_bindings.resize(p_vertex_attribs.size());
	vf_info->vk_attributes.resize(p_vertex_attribs.size());
	for (uint32_t i = 0; i < p_vertex_attribs.size(); i++) {
		vf_info->vk_bindings[i] = {};
		vf_info->vk_bindings[i].binding = i;
		vf_info->vk_bindings[i].stride = p_vertex_attribs[i].stride;
		vf_info->vk_bindings[i].inputRate = p_vertex_attribs[i].frequency == VERTEX_FREQUENCY_INSTANCE ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX;
		vf_info->vk_attributes[i] = {};
		vf_info->vk_attributes[i].binding = i;
		vf_info->vk_attributes[i].location = p_vertex_attribs[i].location;
		vf_info->vk_attributes[i].format = RD_TO_VK_FORMAT[p_vertex_attribs[i].format];
		vf_info->vk_attributes[i].offset = p_vertex_attribs[i].offset;
	}

	vf_info->vk_create_info = {};
	vf_info->vk_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vf_info->vk_create_info.vertexBindingDescriptionCount = vf_info->vk_bindings.size();
	vf_info->vk_create_info.pVertexBindingDescriptions = vf_info->vk_bindings.ptr();
	vf_info->vk_create_info.vertexAttributeDescriptionCount = vf_info->vk_attributes.size();
	vf_info->vk_create_info.pVertexAttributeDescriptions = vf_info->vk_attributes.ptr();

	return VertexFormatID(vf_info);
}

void RenderingDeviceDriverVulkan::vertex_format_free(VertexFormatID p_vertex_format) {
	VertexFormatInfo *vf_info = (VertexFormatInfo *)p_vertex_format.id;
	VersatileResource::free(resources_allocator, vf_info);
}

/******************/
/**** BARRIERS ****/
/******************/

// RDD::PipelineStageBits == VkPipelineStageFlagBits.
static_assert(ENUM_MEMBERS_EQUAL(RDD::PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::PIPELINE_STAGE_DRAW_INDIRECT_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::PIPELINE_STAGE_VERTEX_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT, VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT, VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::PIPELINE_STAGE_GEOMETRY_SHADER_BIT, VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::PIPELINE_STAGE_ALL_GRAPHICS_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT));

// RDD::BarrierAccessBits == VkAccessFlagBits.
static_assert(ENUM_MEMBERS_EQUAL(RDD::BARRIER_ACCESS_INDIRECT_COMMAND_READ_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BARRIER_ACCESS_INDEX_READ_BIT, VK_ACCESS_INDEX_READ_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BARRIER_ACCESS_VERTEX_ATTRIBUTE_READ_BIT, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BARRIER_ACCESS_UNIFORM_READ_BIT, VK_ACCESS_UNIFORM_READ_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BARRIER_ACCESS_INPUT_ATTACHMENT_READ_BIT, VK_ACCESS_INPUT_ATTACHMENT_READ_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BARRIER_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BARRIER_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BARRIER_ACCESS_COLOR_ATTACHMENT_READ_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BARRIER_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BARRIER_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BARRIER_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BARRIER_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BARRIER_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BARRIER_ACCESS_HOST_READ_BIT, VK_ACCESS_HOST_READ_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BARRIER_ACCESS_HOST_WRITE_BIT, VK_ACCESS_HOST_WRITE_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BARRIER_ACCESS_MEMORY_READ_BIT, VK_ACCESS_MEMORY_READ_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BARRIER_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_WRITE_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BARRIER_ACCESS_FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT, VK_ACCESS_FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT_KHR));

void RenderingDeviceDriverVulkan::command_pipeline_barrier(
		CommandBufferID p_cmd_buffer,
		BitField<PipelineStageBits> p_src_stages,
		BitField<PipelineStageBits> p_dst_stages,
		VectorView<MemoryBarrier> p_memory_barriers,
		VectorView<BufferBarrier> p_buffer_barriers,
		VectorView<TextureBarrier> p_texture_barriers) {
	VkMemoryBarrier *vk_memory_barriers = ALLOCA_ARRAY(VkMemoryBarrier, p_memory_barriers.size());
	for (uint32_t i = 0; i < p_memory_barriers.size(); i++) {
		vk_memory_barriers[i] = {};
		vk_memory_barriers[i].sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		vk_memory_barriers[i].srcAccessMask = (VkPipelineStageFlags)p_memory_barriers[i].src_access;
		vk_memory_barriers[i].dstAccessMask = (VkAccessFlags)p_memory_barriers[i].dst_access;
	}

	VkBufferMemoryBarrier *vk_buffer_barriers = ALLOCA_ARRAY(VkBufferMemoryBarrier, p_buffer_barriers.size());
	for (uint32_t i = 0; i < p_buffer_barriers.size(); i++) {
		vk_buffer_barriers[i] = {};
		vk_buffer_barriers[i].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
		vk_buffer_barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		vk_buffer_barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		vk_buffer_barriers[i].srcAccessMask = (VkAccessFlags)p_buffer_barriers[i].src_access;
		vk_buffer_barriers[i].dstAccessMask = (VkAccessFlags)p_buffer_barriers[i].dst_access;
		vk_buffer_barriers[i].buffer = ((const BufferInfo *)p_buffer_barriers[i].buffer.id)->vk_buffer;
		vk_buffer_barriers[i].offset = p_buffer_barriers[i].offset;
		vk_buffer_barriers[i].size = p_buffer_barriers[i].size;
	}

	VkImageMemoryBarrier *vk_image_barriers = ALLOCA_ARRAY(VkImageMemoryBarrier, p_texture_barriers.size());
	for (uint32_t i = 0; i < p_texture_barriers.size(); i++) {
		const TextureInfo *tex_info = (const TextureInfo *)p_texture_barriers[i].texture.id;
		vk_image_barriers[i] = {};
		vk_image_barriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		vk_image_barriers[i].srcAccessMask = (VkAccessFlags)p_texture_barriers[i].src_access;
		vk_image_barriers[i].dstAccessMask = (VkAccessFlags)p_texture_barriers[i].dst_access;
		vk_image_barriers[i].oldLayout = (VkImageLayout)p_texture_barriers[i].prev_layout;
		vk_image_barriers[i].newLayout = (VkImageLayout)p_texture_barriers[i].next_layout;
		vk_image_barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		vk_image_barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		vk_image_barriers[i].image = tex_info->vk_view_create_info.image;
		vk_image_barriers[i].subresourceRange.aspectMask = (VkImageAspectFlags)p_texture_barriers[i].subresources.aspect;
		vk_image_barriers[i].subresourceRange.baseMipLevel = p_texture_barriers[i].subresources.base_mipmap;
		vk_image_barriers[i].subresourceRange.levelCount = p_texture_barriers[i].subresources.mipmap_count;
		vk_image_barriers[i].subresourceRange.baseArrayLayer = p_texture_barriers[i].subresources.base_layer;
		vk_image_barriers[i].subresourceRange.layerCount = p_texture_barriers[i].subresources.layer_count;
	}

#if PRINT_NATIVE_COMMANDS
	print_line(vformat("vkCmdPipelineBarrier MEMORY %d BUFFER %d TEXTURE %d", p_memory_barriers.size(), p_buffer_barriers.size(), p_texture_barriers.size()));
	for (uint32_t i = 0; i < p_memory_barriers.size(); i++) {
		print_line(vformat("  VkMemoryBarrier #%d src 0x%uX dst 0x%uX", i, vk_memory_barriers[i].srcAccessMask, vk_memory_barriers[i].dstAccessMask));
	}

	for (uint32_t i = 0; i < p_buffer_barriers.size(); i++) {
		print_line(vformat("  VkBufferMemoryBarrier #%d src 0x%uX dst 0x%uX buffer 0x%ux", i, vk_buffer_barriers[i].srcAccessMask, vk_buffer_barriers[i].dstAccessMask, uint64_t(vk_buffer_barriers[i].buffer)));
	}

	for (uint32_t i = 0; i < p_texture_barriers.size(); i++) {
		print_line(vformat("  VkImageMemoryBarrier #%d src 0x%uX dst 0x%uX image 0x%ux old %d new %d (%d %d %d %d)", i, vk_image_barriers[i].srcAccessMask, vk_image_barriers[i].dstAccessMask,
				uint64_t(vk_image_barriers[i].image), vk_image_barriers[i].oldLayout, vk_image_barriers[i].newLayout, vk_image_barriers[i].subresourceRange.baseMipLevel, vk_image_barriers[i].subresourceRange.levelCount,
				vk_image_barriers[i].subresourceRange.baseArrayLayer, vk_image_barriers[i].subresourceRange.layerCount));
	}
#endif

	vkCmdPipelineBarrier(
			(VkCommandBuffer)p_cmd_buffer.id,
			(VkPipelineStageFlags)p_src_stages,
			(VkPipelineStageFlags)p_dst_stages,
			0,
			p_memory_barriers.size(), vk_memory_barriers,
			p_buffer_barriers.size(), vk_buffer_barriers,
			p_texture_barriers.size(), vk_image_barriers);
}

/*************************/
/**** COMMAND BUFFERS ****/
/*************************/

// ----- POOL -----

RDD::CommandPoolID RenderingDeviceDriverVulkan::command_pool_create(CommandBufferType p_cmd_buffer_type) {
	VkCommandPoolCreateInfo cmd_pool_info = {};
	cmd_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	cmd_pool_info.queueFamilyIndex = context->get_graphics_queue_family_index();
	cmd_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

	VkCommandPool vk_cmd_pool = VK_NULL_HANDLE;
	VkResult res = vkCreateCommandPool(vk_device, &cmd_pool_info, nullptr, &vk_cmd_pool);
	ERR_FAIL_COND_V_MSG(res, CommandPoolID(), "vkCreateCommandPool failed with error " + itos(res) + ".");

#ifdef DEBUG_ENABLED
	if (p_cmd_buffer_type == COMMAND_BUFFER_TYPE_SECONDARY) {
		secondary_cmd_pools.insert(CommandPoolID(vk_cmd_pool));
	}
#endif

	return CommandPoolID(vk_cmd_pool);
}

void RenderingDeviceDriverVulkan::command_pool_free(CommandPoolID p_cmd_pool) {
	vkDestroyCommandPool(vk_device, (VkCommandPool)p_cmd_pool.id, nullptr);

#ifdef DEBUG_ENABLED
	secondary_cmd_pools.erase(p_cmd_pool);
#endif
}

// ----- BUFFER -----

RDD::CommandBufferID RenderingDeviceDriverVulkan::command_buffer_create(CommandBufferType p_cmd_buffer_type, CommandPoolID p_cmd_pool) {
#ifdef DEBUG_ENABLED
	if (p_cmd_buffer_type == COMMAND_BUFFER_TYPE_PRIMARY) {
		ERR_FAIL_COND_V(secondary_cmd_pools.has(p_cmd_pool), CommandBufferID());
	} else {
		ERR_FAIL_COND_V(!secondary_cmd_pools.has(p_cmd_pool), CommandBufferID());
	}
#endif

	VkCommandBufferAllocateInfo cmd_buf_info = {};
	cmd_buf_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	cmd_buf_info.commandPool = (VkCommandPool)p_cmd_pool.id;
	cmd_buf_info.level = p_cmd_buffer_type == COMMAND_BUFFER_TYPE_PRIMARY ? VK_COMMAND_BUFFER_LEVEL_PRIMARY : VK_COMMAND_BUFFER_LEVEL_SECONDARY;
	cmd_buf_info.commandBufferCount = 1;

	VkCommandBuffer vk_cmd_buffer = VK_NULL_HANDLE;
	VkResult err = vkAllocateCommandBuffers(vk_device, &cmd_buf_info, &vk_cmd_buffer);
	ERR_FAIL_COND_V_MSG(err, CommandBufferID(), "vkAllocateCommandBuffers failed with error " + itos(err) + ".");

	CommandBufferID cmd_buffer_id = CommandBufferID(vk_cmd_buffer);
#ifdef DEBUG_ENABLED
	// Erase first because Vulkan may reuse a handle.
	secondary_cmd_buffers.erase(cmd_buffer_id);
	if (p_cmd_buffer_type == COMMAND_BUFFER_TYPE_SECONDARY) {
		secondary_cmd_buffers.insert(cmd_buffer_id);
	}
#endif
	return cmd_buffer_id;
}

bool RenderingDeviceDriverVulkan::command_buffer_begin(CommandBufferID p_cmd_buffer) {
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND_V(secondary_cmd_buffers.has(p_cmd_buffer), false);
#endif

	// Reset is implicit (VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT).

	VkCommandBufferBeginInfo cmd_buf_begin_info = {};
	cmd_buf_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	cmd_buf_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	VkResult err = vkBeginCommandBuffer((VkCommandBuffer)p_cmd_buffer.id, &cmd_buf_begin_info);
	ERR_FAIL_COND_V_MSG(err, false, "vkBeginCommandBuffer failed with error " + itos(err) + ".");

	return true;
}

bool RenderingDeviceDriverVulkan::command_buffer_begin_secondary(CommandBufferID p_cmd_buffer, RenderPassID p_render_pass, uint32_t p_subpass, FramebufferID p_framebuffer) {
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND_V(!secondary_cmd_buffers.has(p_cmd_buffer), false);
#endif

	// Reset is implicit (VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT).

	VkCommandBufferInheritanceInfo inheritance_info = {};
	inheritance_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
	inheritance_info.renderPass = (VkRenderPass)p_render_pass.id;
	inheritance_info.subpass = p_subpass;
	inheritance_info.framebuffer = (VkFramebuffer)p_framebuffer.id;

	VkCommandBufferBeginInfo cmd_buf_begin_info = {};
	cmd_buf_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	cmd_buf_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT | VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
	cmd_buf_begin_info.pInheritanceInfo = &inheritance_info;

	VkResult err = vkBeginCommandBuffer((VkCommandBuffer)p_cmd_buffer.id, &cmd_buf_begin_info);
	ERR_FAIL_COND_V_MSG(err, false, "vkBeginCommandBuffer failed with error " + itos(err) + ".");

	return true;
}

void RenderingDeviceDriverVulkan::command_buffer_end(CommandBufferID p_cmd_buffer) {
	vkEndCommandBuffer((VkCommandBuffer)p_cmd_buffer.id);
}

void RenderingDeviceDriverVulkan::command_buffer_execute_secondary(CommandBufferID p_cmd_buffer, VectorView<CommandBufferID> p_secondary_cmd_buffers) {
#ifdef DEBUG_ENABLED
	ERR_FAIL_COND(secondary_cmd_buffers.has(p_cmd_buffer));
	for (uint32_t i = 0; i < p_secondary_cmd_buffers.size(); i++) {
		ERR_FAIL_COND(!secondary_cmd_buffers.has(p_secondary_cmd_buffers[i]));
	}
#endif

	vkCmdExecuteCommands((VkCommandBuffer)p_cmd_buffer.id, p_secondary_cmd_buffers.size(), (const VkCommandBuffer *)p_secondary_cmd_buffers.ptr());
}

/*********************/
/**** FRAMEBUFFER ****/
/*********************/

RDD::FramebufferID RenderingDeviceDriverVulkan::framebuffer_create(RenderPassID p_render_pass, VectorView<TextureID> p_attachments, uint32_t p_width, uint32_t p_height) {
	VkImageView *vk_img_views = ALLOCA_ARRAY(VkImageView, p_attachments.size());
	for (uint32_t i = 0; i < p_attachments.size(); i++) {
		vk_img_views[i] = ((const TextureInfo *)p_attachments[i].id)->vk_view;
	}

	VkFramebufferCreateInfo framebuffer_create_info = {};
	framebuffer_create_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
	framebuffer_create_info.renderPass = (VkRenderPass)p_render_pass.id;
	framebuffer_create_info.attachmentCount = p_attachments.size();
	framebuffer_create_info.pAttachments = vk_img_views;
	framebuffer_create_info.width = p_width;
	framebuffer_create_info.height = p_height;
	framebuffer_create_info.layers = 1;

	VkFramebuffer vk_framebuffer = VK_NULL_HANDLE;
	VkResult err = vkCreateFramebuffer(vk_device, &framebuffer_create_info, nullptr, &vk_framebuffer);
	ERR_FAIL_COND_V_MSG(err, FramebufferID(), "vkCreateFramebuffer failed with error " + itos(err) + ".");

#if PRINT_NATIVE_COMMANDS
	print_line(vformat("vkCreateFramebuffer 0x%uX with %d attachments", uint64_t(vk_framebuffer), p_attachments.size()));
	for (uint32_t i = 0; i < p_attachments.size(); i++) {
		const TextureInfo *attachment_info = (const TextureInfo *)p_attachments[i].id;
		print_line(vformat("  Attachment #%d: IMAGE 0x%uX VIEW 0x%uX", i, uint64_t(attachment_info->vk_view_create_info.image), uint64_t(attachment_info->vk_view)));
	}
#endif

	return FramebufferID(vk_framebuffer);
}

void RenderingDeviceDriverVulkan::framebuffer_free(FramebufferID p_framebuffer) {
	vkDestroyFramebuffer(vk_device, (VkFramebuffer)p_framebuffer.id, nullptr);
}

/****************/
/**** SHADER ****/
/****************/

static VkShaderStageFlagBits RD_STAGE_TO_VK_SHADER_STAGE_BITS[RDD::SHADER_STAGE_MAX] = {
	VK_SHADER_STAGE_VERTEX_BIT,
	VK_SHADER_STAGE_FRAGMENT_BIT,
	VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
	VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
	VK_SHADER_STAGE_COMPUTE_BIT,
};

String RenderingDeviceDriverVulkan::shader_get_binary_cache_key() {
	return "Vulkan-SV" + uitos(ShaderBinary::VERSION);
}

Vector<uint8_t> RenderingDeviceDriverVulkan::shader_compile_binary_from_spirv(VectorView<ShaderStageSPIRVData> p_spirv, const String &p_shader_name) {
	ShaderReflection shader_refl;
	if (_reflect_spirv(p_spirv, shader_refl) != OK) {
		return Vector<uint8_t>();
	}

	ERR_FAIL_COND_V_MSG((uint32_t)shader_refl.uniform_sets.size() > limits.maxBoundDescriptorSets, Vector<uint8_t>(),
			"Number of uniform sets is larger than what is supported by the hardware (" + itos(limits.maxBoundDescriptorSets) + ").");

	// Collect reflection data into binary data.
	ShaderBinary::Data binary_data;
	Vector<Vector<ShaderBinary::DataBinding>> uniforms; // Set bindings.
	Vector<ShaderBinary::SpecializationConstant> specialization_constants;
	{
		binary_data.vertex_input_mask = shader_refl.vertex_input_mask;
		binary_data.fragment_output_mask = shader_refl.fragment_output_mask;
		binary_data.specialization_constants_count = shader_refl.specialization_constants.size();
		binary_data.is_compute = shader_refl.is_compute;
		binary_data.compute_local_size[0] = shader_refl.compute_local_size[0];
		binary_data.compute_local_size[1] = shader_refl.compute_local_size[1];
		binary_data.compute_local_size[2] = shader_refl.compute_local_size[2];
		binary_data.set_count = shader_refl.uniform_sets.size();
		binary_data.push_constant_size = shader_refl.push_constant_size;
		for (uint32_t i = 0; i < SHADER_STAGE_MAX; i++) {
			if (shader_refl.push_constant_stages.has_flag((ShaderStage)(1 << i))) {
				binary_data.vk_push_constant_stages_mask |= RD_STAGE_TO_VK_SHADER_STAGE_BITS[i];
			}
		}

		for (const Vector<ShaderUniform> &set_refl : shader_refl.uniform_sets) {
			Vector<ShaderBinary::DataBinding> set_bindings;
			for (const ShaderUniform &uniform_refl : set_refl) {
				ShaderBinary::DataBinding binding;
				binding.type = (uint32_t)uniform_refl.type;
				binding.binding = uniform_refl.binding;
				binding.stages = (uint32_t)uniform_refl.stages;
				binding.length = uniform_refl.length;
				binding.writable = (uint32_t)uniform_refl.writable;
				set_bindings.push_back(binding);
			}
			uniforms.push_back(set_bindings);
		}

		for (const ShaderSpecializationConstant &refl_sc : shader_refl.specialization_constants) {
			ShaderBinary::SpecializationConstant spec_constant;
			spec_constant.type = (uint32_t)refl_sc.type;
			spec_constant.constant_id = refl_sc.constant_id;
			spec_constant.int_value = refl_sc.int_value;
			spec_constant.stage_flags = (uint32_t)refl_sc.stages;
			specialization_constants.push_back(spec_constant);
		}
	}

	Vector<Vector<uint8_t>> compressed_stages;
	Vector<uint32_t> smolv_size;
	Vector<uint32_t> zstd_size; // If 0, zstd not used.

	uint32_t stages_binary_size = 0;

	bool strip_debug = false;

	for (uint32_t i = 0; i < p_spirv.size(); i++) {
		smolv::ByteArray smolv;
		if (!smolv::Encode(p_spirv[i].spirv.ptr(), p_spirv[i].spirv.size(), smolv, strip_debug ? smolv::kEncodeFlagStripDebugInfo : 0)) {
			ERR_FAIL_V_MSG(Vector<uint8_t>(), "Error compressing shader stage :" + String(SHADER_STAGE_NAMES[p_spirv[i].shader_stage]));
		} else {
			smolv_size.push_back(smolv.size());
			{ // zstd.
				Vector<uint8_t> zstd;
				zstd.resize(Compression::get_max_compressed_buffer_size(smolv.size(), Compression::MODE_ZSTD));
				int dst_size = Compression::compress(zstd.ptrw(), &smolv[0], smolv.size(), Compression::MODE_ZSTD);

				if (dst_size > 0 && (uint32_t)dst_size < smolv.size()) {
					zstd_size.push_back(dst_size);
					zstd.resize(dst_size);
					compressed_stages.push_back(zstd);
				} else {
					Vector<uint8_t> smv;
					smv.resize(smolv.size());
					memcpy(smv.ptrw(), &smolv[0], smolv.size());
					zstd_size.push_back(0); // Not using zstd.
					compressed_stages.push_back(smv);
				}
			}
		}
		uint32_t s = compressed_stages[i].size();
		if (s % 4 != 0) {
			s += 4 - (s % 4);
		}
		stages_binary_size += s;
	}

	binary_data.specialization_constants_count = specialization_constants.size();
	binary_data.set_count = uniforms.size();
	binary_data.stage_count = p_spirv.size();

	CharString shader_name_utf = p_shader_name.utf8();

	binary_data.shader_name_len = shader_name_utf.length();

	uint32_t total_size = sizeof(uint32_t) * 3; // Header + version + main datasize;.
	total_size += sizeof(ShaderBinary::Data);

	total_size += binary_data.shader_name_len;

	if ((binary_data.shader_name_len % 4) != 0) { // Alignment rules are really strange.
		total_size += 4 - (binary_data.shader_name_len % 4);
	}

	for (int i = 0; i < uniforms.size(); i++) {
		total_size += sizeof(uint32_t);
		total_size += uniforms[i].size() * sizeof(ShaderBinary::DataBinding);
	}

	total_size += sizeof(ShaderBinary::SpecializationConstant) * specialization_constants.size();

	total_size += compressed_stages.size() * sizeof(uint32_t) * 3; // Sizes.
	total_size += stages_binary_size;

	Vector<uint8_t> ret;
	ret.resize(total_size);
	{
		uint32_t offset = 0;
		uint8_t *binptr = ret.ptrw();
		binptr[0] = 'G';
		binptr[1] = 'S';
		binptr[2] = 'B';
		binptr[3] = 'D'; // Godot Shader Binary Data.
		offset += 4;
		encode_uint32(ShaderBinary::VERSION, binptr + offset);
		offset += sizeof(uint32_t);
		encode_uint32(sizeof(ShaderBinary::Data), binptr + offset);
		offset += sizeof(uint32_t);
		memcpy(binptr + offset, &binary_data, sizeof(ShaderBinary::Data));
		offset += sizeof(ShaderBinary::Data);

		if (binary_data.shader_name_len > 0) {
			memcpy(binptr + offset, shader_name_utf.ptr(), binary_data.shader_name_len);
			offset += binary_data.shader_name_len;

			if ((binary_data.shader_name_len % 4) != 0) { // Alignment rules are really strange.
				offset += 4 - (binary_data.shader_name_len % 4);
			}
		}

		for (int i = 0; i < uniforms.size(); i++) {
			int count = uniforms[i].size();
			encode_uint32(count, binptr + offset);
			offset += sizeof(uint32_t);
			if (count > 0) {
				memcpy(binptr + offset, uniforms[i].ptr(), sizeof(ShaderBinary::DataBinding) * count);
				offset += sizeof(ShaderBinary::DataBinding) * count;
			}
		}

		if (specialization_constants.size()) {
			memcpy(binptr + offset, specialization_constants.ptr(), sizeof(ShaderBinary::SpecializationConstant) * specialization_constants.size());
			offset += sizeof(ShaderBinary::SpecializationConstant) * specialization_constants.size();
		}

		for (int i = 0; i < compressed_stages.size(); i++) {
			encode_uint32(p_spirv[i].shader_stage, binptr + offset);
			offset += sizeof(uint32_t);
			encode_uint32(smolv_size[i], binptr + offset);
			offset += sizeof(uint32_t);
			encode_uint32(zstd_size[i], binptr + offset);
			offset += sizeof(uint32_t);
			memcpy(binptr + offset, compressed_stages[i].ptr(), compressed_stages[i].size());

			uint32_t s = compressed_stages[i].size();

			if (s % 4 != 0) {
				s += 4 - (s % 4);
			}

			offset += s;
		}

		DEV_ASSERT(offset == (uint32_t)ret.size());
	}

	return ret;
}

RDD::ShaderID RenderingDeviceDriverVulkan::shader_create_from_bytecode(const Vector<uint8_t> &p_shader_binary, ShaderDescription &r_shader_desc, String &r_name) {
	r_shader_desc = {}; // Driver-agnostic.
	ShaderInfo shader_info; // Driver-specific.

	const uint8_t *binptr = p_shader_binary.ptr();
	uint32_t binsize = p_shader_binary.size();

	uint32_t read_offset = 0;

	// Consistency check.
	ERR_FAIL_COND_V(binsize < sizeof(uint32_t) * 3 + sizeof(ShaderBinary::Data), ShaderID());
	ERR_FAIL_COND_V(binptr[0] != 'G' || binptr[1] != 'S' || binptr[2] != 'B' || binptr[3] != 'D', ShaderID());

	uint32_t bin_version = decode_uint32(binptr + 4);
	ERR_FAIL_COND_V(bin_version != ShaderBinary::VERSION, ShaderID());

	uint32_t bin_data_size = decode_uint32(binptr + 8);

	const ShaderBinary::Data &binary_data = *(reinterpret_cast<const ShaderBinary::Data *>(binptr + 12));

	r_shader_desc.push_constant_size = binary_data.push_constant_size;
	shader_info.vk_push_constant_stages = binary_data.vk_push_constant_stages_mask;

	r_shader_desc.vertex_input_mask = binary_data.vertex_input_mask;
	r_shader_desc.fragment_output_mask = binary_data.fragment_output_mask;

	r_shader_desc.is_compute = binary_data.is_compute;
	r_shader_desc.compute_local_size[0] = binary_data.compute_local_size[0];
	r_shader_desc.compute_local_size[1] = binary_data.compute_local_size[1];
	r_shader_desc.compute_local_size[2] = binary_data.compute_local_size[2];

	read_offset += sizeof(uint32_t) * 3 + bin_data_size;

	if (binary_data.shader_name_len) {
		r_name.parse_utf8((const char *)(binptr + read_offset), binary_data.shader_name_len);
		read_offset += binary_data.shader_name_len;
		if ((binary_data.shader_name_len % 4) != 0) { // Alignment rules are really strange.
			read_offset += 4 - (binary_data.shader_name_len % 4);
		}
	}

	Vector<Vector<VkDescriptorSetLayoutBinding>> vk_set_bindings;

	r_shader_desc.uniform_sets.resize(binary_data.set_count);
	vk_set_bindings.resize(binary_data.set_count);

	for (uint32_t i = 0; i < binary_data.set_count; i++) {
		ERR_FAIL_COND_V(read_offset + sizeof(uint32_t) >= binsize, ShaderID());
		uint32_t set_count = decode_uint32(binptr + read_offset);
		read_offset += sizeof(uint32_t);
		const ShaderBinary::DataBinding *set_ptr = reinterpret_cast<const ShaderBinary::DataBinding *>(binptr + read_offset);
		uint32_t set_size = set_count * sizeof(ShaderBinary::DataBinding);
		ERR_FAIL_COND_V(read_offset + set_size >= binsize, ShaderID());

		for (uint32_t j = 0; j < set_count; j++) {
			ShaderUniform info;
			info.type = UniformType(set_ptr[j].type);
			info.writable = set_ptr[j].writable;
			info.length = set_ptr[j].length;
			info.binding = set_ptr[j].binding;
			info.stages = set_ptr[j].stages;

			VkDescriptorSetLayoutBinding layout_binding = {};
			layout_binding.binding = set_ptr[j].binding;
			layout_binding.descriptorCount = 1;
			for (uint32_t k = 0; k < SHADER_STAGE_MAX; k++) {
				if ((set_ptr[j].stages & (1 << k))) {
					layout_binding.stageFlags |= RD_STAGE_TO_VK_SHADER_STAGE_BITS[k];
				}
			}

			switch (info.type) {
				case UNIFORM_TYPE_SAMPLER: {
					layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
					layout_binding.descriptorCount = set_ptr[j].length;
				} break;
				case UNIFORM_TYPE_SAMPLER_WITH_TEXTURE: {
					layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
					layout_binding.descriptorCount = set_ptr[j].length;
				} break;
				case UNIFORM_TYPE_TEXTURE: {
					layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
					layout_binding.descriptorCount = set_ptr[j].length;
				} break;
				case UNIFORM_TYPE_IMAGE: {
					layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
					layout_binding.descriptorCount = set_ptr[j].length;
				} break;
				case UNIFORM_TYPE_TEXTURE_BUFFER: {
					layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
					layout_binding.descriptorCount = set_ptr[j].length;
				} break;
				case UNIFORM_TYPE_IMAGE_BUFFER: {
					layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
				} break;
				case UNIFORM_TYPE_UNIFORM_BUFFER: {
					layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				} break;
				case UNIFORM_TYPE_STORAGE_BUFFER: {
					layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				} break;
				case UNIFORM_TYPE_INPUT_ATTACHMENT: {
					layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
				} break;
				default: {
					DEV_ASSERT(false);
				}
			}

			r_shader_desc.uniform_sets.write[i].push_back(info);
			vk_set_bindings.write[i].push_back(layout_binding);
		}

		read_offset += set_size;
	}

	ERR_FAIL_COND_V(read_offset + binary_data.specialization_constants_count * sizeof(ShaderBinary::SpecializationConstant) >= binsize, ShaderID());

	r_shader_desc.specialization_constants.resize(binary_data.specialization_constants_count);
	for (uint32_t i = 0; i < binary_data.specialization_constants_count; i++) {
		const ShaderBinary::SpecializationConstant &src_sc = *(reinterpret_cast<const ShaderBinary::SpecializationConstant *>(binptr + read_offset));
		ShaderSpecializationConstant sc;
		sc.type = PipelineSpecializationConstantType(src_sc.type);
		sc.constant_id = src_sc.constant_id;
		sc.int_value = src_sc.int_value;
		sc.stages = src_sc.stage_flags;
		r_shader_desc.specialization_constants.write[i] = sc;

		read_offset += sizeof(ShaderBinary::SpecializationConstant);
	}

	Vector<Vector<uint8_t>> stages_spirv;
	stages_spirv.resize(binary_data.stage_count);
	r_shader_desc.stages.resize(binary_data.stage_count);

	for (uint32_t i = 0; i < binary_data.stage_count; i++) {
		ERR_FAIL_COND_V(read_offset + sizeof(uint32_t) * 3 >= binsize, ShaderID());
		uint32_t stage = decode_uint32(binptr + read_offset);
		read_offset += sizeof(uint32_t);
		uint32_t smolv_size = decode_uint32(binptr + read_offset);
		read_offset += sizeof(uint32_t);
		uint32_t zstd_size = decode_uint32(binptr + read_offset);
		read_offset += sizeof(uint32_t);

		uint32_t buf_size = (zstd_size > 0) ? zstd_size : smolv_size;

		Vector<uint8_t> smolv;
		const uint8_t *src_smolv = nullptr;

		if (zstd_size > 0) {
			// Decompress to smolv.
			smolv.resize(smolv_size);
			int dec_smolv_size = Compression::decompress(smolv.ptrw(), smolv.size(), binptr + read_offset, zstd_size, Compression::MODE_ZSTD);
			ERR_FAIL_COND_V(dec_smolv_size != (int32_t)smolv_size, ShaderID());
			src_smolv = smolv.ptr();
		} else {
			src_smolv = binptr + read_offset;
		}

		Vector<uint8_t> &spirv = stages_spirv.ptrw()[i];
		uint32_t spirv_size = smolv::GetDecodedBufferSize(src_smolv, smolv_size);
		spirv.resize(spirv_size);
		if (!smolv::Decode(src_smolv, smolv_size, spirv.ptrw(), spirv_size)) {
			ERR_FAIL_V_MSG(ShaderID(), "Malformed smolv input uncompressing shader stage:" + String(SHADER_STAGE_NAMES[stage]));
		}

		r_shader_desc.stages.set(i, ShaderStage(stage));

		if (buf_size % 4 != 0) {
			buf_size += 4 - (buf_size % 4);
		}

		DEV_ASSERT(read_offset + buf_size <= binsize);

		read_offset += buf_size;
	}

	DEV_ASSERT(read_offset == binsize);

	// Modules.

	String error_text;

	for (int i = 0; i < r_shader_desc.stages.size(); i++) {
		VkShaderModuleCreateInfo shader_module_create_info = {};
		shader_module_create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		shader_module_create_info.codeSize = stages_spirv[i].size();
		shader_module_create_info.pCode = (const uint32_t *)stages_spirv[i].ptr();

		VkShaderModule vk_module = VK_NULL_HANDLE;
		VkResult res = vkCreateShaderModule(vk_device, &shader_module_create_info, nullptr, &vk_module);
		if (res) {
			error_text = "Error (" + itos(res) + ") creating shader module for stage: " + String(SHADER_STAGE_NAMES[r_shader_desc.stages[i]]);
			break;
		}

		VkPipelineShaderStageCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		create_info.stage = RD_STAGE_TO_VK_SHADER_STAGE_BITS[r_shader_desc.stages[i]];
		create_info.module = vk_module;
		create_info.pName = "main";

		shader_info.vk_stages_create_info.push_back(create_info);
	}

	// Descriptor sets.

	if (error_text.is_empty()) {
		DEV_ASSERT((uint32_t)vk_set_bindings.size() == binary_data.set_count);
		for (uint32_t i = 0; i < binary_data.set_count; i++) {
			// Empty ones are fine if they were not used according to spec (binding count will be 0).
			VkDescriptorSetLayoutCreateInfo layout_create_info = {};
			layout_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
			layout_create_info.bindingCount = vk_set_bindings[i].size();
			layout_create_info.pBindings = vk_set_bindings[i].ptr();

			VkDescriptorSetLayout layout = VK_NULL_HANDLE;
			VkResult res = vkCreateDescriptorSetLayout(vk_device, &layout_create_info, nullptr, &layout);
			if (res) {
				error_text = "Error (" + itos(res) + ") creating descriptor set layout for set " + itos(i);
				break;
			}

			shader_info.vk_descriptor_set_layouts.push_back(layout);
		}
	}

	if (error_text.is_empty()) {
		// Pipeline layout.

		VkPipelineLayoutCreateInfo pipeline_layout_create_info = {};
		pipeline_layout_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipeline_layout_create_info.setLayoutCount = binary_data.set_count;
		pipeline_layout_create_info.pSetLayouts = shader_info.vk_descriptor_set_layouts.ptr();

		if (binary_data.push_constant_size) {
			VkPushConstantRange *push_constant_range = ALLOCA_SINGLE(VkPushConstantRange);
			*push_constant_range = {};
			push_constant_range->stageFlags = binary_data.vk_push_constant_stages_mask;
			push_constant_range->size = binary_data.push_constant_size;
			pipeline_layout_create_info.pushConstantRangeCount = 1;
			pipeline_layout_create_info.pPushConstantRanges = push_constant_range;
		}

		VkResult err = vkCreatePipelineLayout(vk_device, &pipeline_layout_create_info, nullptr, &shader_info.vk_pipeline_layout);
		if (err) {
			error_text = "Error (" + itos(err) + ") creating pipeline layout.";
		}
	}

	if (!error_text.is_empty()) {
		// Clean up if failed.
		for (uint32_t i = 0; i < shader_info.vk_stages_create_info.size(); i++) {
			vkDestroyShaderModule(vk_device, shader_info.vk_stages_create_info[i].module, nullptr);
		}
		for (uint32_t i = 0; i < binary_data.set_count; i++) {
			vkDestroyDescriptorSetLayout(vk_device, shader_info.vk_descriptor_set_layouts[i], nullptr);
		}

		ERR_FAIL_V_MSG(ShaderID(), error_text);
	}

	// Bookkeep.

	ShaderInfo *shader_info_ptr = VersatileResource::allocate<ShaderInfo>(resources_allocator);
	*shader_info_ptr = shader_info;
	return ShaderID(shader_info_ptr);
}

void RenderingDeviceDriverVulkan::shader_free(ShaderID p_shader) {
	ShaderInfo *shader_info = (ShaderInfo *)p_shader.id;

	for (uint32_t i = 0; i < shader_info->vk_descriptor_set_layouts.size(); i++) {
		vkDestroyDescriptorSetLayout(vk_device, shader_info->vk_descriptor_set_layouts[i], nullptr);
	}

	vkDestroyPipelineLayout(vk_device, shader_info->vk_pipeline_layout, nullptr);

	for (uint32_t i = 0; i < shader_info->vk_stages_create_info.size(); i++) {
		vkDestroyShaderModule(vk_device, shader_info->vk_stages_create_info[i].module, nullptr);
	}

	VersatileResource::free(resources_allocator, shader_info);
}

/*********************/
/**** UNIFORM SET ****/
/*********************/

VkDescriptorPool RenderingDeviceDriverVulkan::_descriptor_set_pool_find_or_create(const DescriptorSetPoolKey &p_key, DescriptorSetPools::Iterator *r_pool_sets_it) {
	DescriptorSetPools::Iterator pool_sets_it = descriptor_set_pools.find(p_key);

	if (pool_sets_it) {
		for (KeyValue<VkDescriptorPool, uint32_t> &E : pool_sets_it->value) {
			if (E.value < max_descriptor_sets_per_pool) {
				*r_pool_sets_it = pool_sets_it;
				return E.key;
			}
		}
	}

	// Create a new one.

	// Here comes more vulkan API strangeness.
	VkDescriptorPoolSize *vk_sizes = ALLOCA_ARRAY(VkDescriptorPoolSize, UNIFORM_TYPE_MAX);
	uint32_t vk_sizes_count = 0;
	{
		VkDescriptorPoolSize *curr_vk_size = vk_sizes;
		if (p_key.uniform_type[UNIFORM_TYPE_SAMPLER]) {
			*curr_vk_size = {};
			curr_vk_size->type = VK_DESCRIPTOR_TYPE_SAMPLER;
			curr_vk_size->descriptorCount = p_key.uniform_type[UNIFORM_TYPE_SAMPLER] * max_descriptor_sets_per_pool;
			curr_vk_size++;
			vk_sizes_count++;
		}
		if (p_key.uniform_type[UNIFORM_TYPE_SAMPLER_WITH_TEXTURE]) {
			*curr_vk_size = {};
			curr_vk_size->type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			curr_vk_size->descriptorCount = p_key.uniform_type[UNIFORM_TYPE_SAMPLER_WITH_TEXTURE] * max_descriptor_sets_per_pool;
			curr_vk_size++;
			vk_sizes_count++;
		}
		if (p_key.uniform_type[UNIFORM_TYPE_TEXTURE]) {
			*curr_vk_size = {};
			curr_vk_size->type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
			curr_vk_size->descriptorCount = p_key.uniform_type[UNIFORM_TYPE_TEXTURE] * max_descriptor_sets_per_pool;
			curr_vk_size++;
			vk_sizes_count++;
		}
		if (p_key.uniform_type[UNIFORM_TYPE_IMAGE]) {
			*curr_vk_size = {};
			curr_vk_size->type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			curr_vk_size->descriptorCount = p_key.uniform_type[UNIFORM_TYPE_IMAGE] * max_descriptor_sets_per_pool;
			curr_vk_size++;
			vk_sizes_count++;
		}
		if (p_key.uniform_type[UNIFORM_TYPE_TEXTURE_BUFFER] || p_key.uniform_type[UNIFORM_TYPE_SAMPLER_WITH_TEXTURE_BUFFER]) {
			*curr_vk_size = {};
			curr_vk_size->type = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
			curr_vk_size->descriptorCount = (p_key.uniform_type[UNIFORM_TYPE_TEXTURE_BUFFER] + p_key.uniform_type[UNIFORM_TYPE_SAMPLER_WITH_TEXTURE_BUFFER]) * max_descriptor_sets_per_pool;
			curr_vk_size++;
			vk_sizes_count++;
		}
		if (p_key.uniform_type[UNIFORM_TYPE_IMAGE_BUFFER]) {
			*curr_vk_size = {};
			curr_vk_size->type = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
			curr_vk_size->descriptorCount = p_key.uniform_type[UNIFORM_TYPE_IMAGE_BUFFER] * max_descriptor_sets_per_pool;
			curr_vk_size++;
			vk_sizes_count++;
		}
		if (p_key.uniform_type[UNIFORM_TYPE_UNIFORM_BUFFER]) {
			*curr_vk_size = {};
			curr_vk_size->type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			curr_vk_size->descriptorCount = p_key.uniform_type[UNIFORM_TYPE_UNIFORM_BUFFER] * max_descriptor_sets_per_pool;
			curr_vk_size++;
			vk_sizes_count++;
		}
		if (p_key.uniform_type[UNIFORM_TYPE_STORAGE_BUFFER]) {
			*curr_vk_size = {};
			curr_vk_size->type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			curr_vk_size->descriptorCount = p_key.uniform_type[UNIFORM_TYPE_STORAGE_BUFFER] * max_descriptor_sets_per_pool;
			curr_vk_size++;
			vk_sizes_count++;
		}
		if (p_key.uniform_type[UNIFORM_TYPE_INPUT_ATTACHMENT]) {
			*curr_vk_size = {};
			curr_vk_size->type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
			curr_vk_size->descriptorCount = p_key.uniform_type[UNIFORM_TYPE_INPUT_ATTACHMENT] * max_descriptor_sets_per_pool;
			curr_vk_size++;
			vk_sizes_count++;
		}
		DEV_ASSERT(vk_sizes_count <= UNIFORM_TYPE_MAX);
	}

	VkDescriptorPoolCreateInfo descriptor_set_pool_create_info = {};
	descriptor_set_pool_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	descriptor_set_pool_create_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT; // Can't think how somebody may NOT need this flag.
	descriptor_set_pool_create_info.maxSets = max_descriptor_sets_per_pool;
	descriptor_set_pool_create_info.poolSizeCount = vk_sizes_count;
	descriptor_set_pool_create_info.pPoolSizes = vk_sizes;

	VkDescriptorPool vk_pool = VK_NULL_HANDLE;
	VkResult res = vkCreateDescriptorPool(vk_device, &descriptor_set_pool_create_info, nullptr, &vk_pool);
	if (res) {
		ERR_FAIL_COND_V_MSG(res, VK_NULL_HANDLE, "vkCreateDescriptorPool failed with error " + itos(res) + ".");
	}

	// Bookkeep.

	if (!pool_sets_it) {
		pool_sets_it = descriptor_set_pools.insert(p_key, HashMap<VkDescriptorPool, uint32_t>());
	}
	HashMap<VkDescriptorPool, uint32_t> &pool_rcs = pool_sets_it->value;
	pool_rcs.insert(vk_pool, 0);
	*r_pool_sets_it = pool_sets_it;
	return vk_pool;
}

void RenderingDeviceDriverVulkan::_descriptor_set_pool_unreference(DescriptorSetPools::Iterator p_pool_sets_it, VkDescriptorPool p_vk_descriptor_pool) {
	HashMap<VkDescriptorPool, uint32_t>::Iterator pool_rcs_it = p_pool_sets_it->value.find(p_vk_descriptor_pool);
	pool_rcs_it->value--;
	if (pool_rcs_it->value == 0) {
		vkDestroyDescriptorPool(vk_device, p_vk_descriptor_pool, nullptr);
		p_pool_sets_it->value.erase(p_vk_descriptor_pool);
		if (p_pool_sets_it->value.is_empty()) {
			descriptor_set_pools.remove(p_pool_sets_it);
		}
	}
}

RDD::UniformSetID RenderingDeviceDriverVulkan::uniform_set_create(VectorView<BoundUniform> p_uniforms, ShaderID p_shader, uint32_t p_set_index) {
	DescriptorSetPoolKey pool_key;

	VkWriteDescriptorSet *vk_writes = ALLOCA_ARRAY(VkWriteDescriptorSet, p_uniforms.size());
	for (uint32_t i = 0; i < p_uniforms.size(); i++) {
		const BoundUniform &uniform = p_uniforms[i];

		vk_writes[i] = {};
		vk_writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		vk_writes[i].dstBinding = uniform.binding;
		vk_writes[i].descriptorType = VK_DESCRIPTOR_TYPE_MAX_ENUM; // Invalid value.

		uint32_t num_descriptors = 1;

		switch (uniform.type) {
			case UNIFORM_TYPE_SAMPLER: {
				num_descriptors = uniform.ids.size();
				VkDescriptorImageInfo *vk_img_infos = ALLOCA_ARRAY(VkDescriptorImageInfo, num_descriptors);

				for (uint32_t j = 0; j < num_descriptors; j++) {
					vk_img_infos[j] = {};
					vk_img_infos[j].sampler = (VkSampler)uniform.ids[j].id;
					vk_img_infos[j].imageView = VK_NULL_HANDLE;
					vk_img_infos[j].imageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
				}

				vk_writes[i].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
				vk_writes[i].pImageInfo = vk_img_infos;
			} break;
			case UNIFORM_TYPE_SAMPLER_WITH_TEXTURE: {
				num_descriptors = uniform.ids.size() / 2;
				VkDescriptorImageInfo *vk_img_infos = ALLOCA_ARRAY(VkDescriptorImageInfo, num_descriptors);

				for (uint32_t j = 0; j < num_descriptors; j++) {
					vk_img_infos[j] = {};
					vk_img_infos[j].sampler = (VkSampler)uniform.ids[j * 2 + 0].id;
					vk_img_infos[j].imageView = ((const TextureInfo *)uniform.ids[j * 2 + 1].id)->vk_view;
					vk_img_infos[j].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				}

				vk_writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				vk_writes[i].pImageInfo = vk_img_infos;
			} break;
			case UNIFORM_TYPE_TEXTURE: {
				num_descriptors = uniform.ids.size();
				VkDescriptorImageInfo *vk_img_infos = ALLOCA_ARRAY(VkDescriptorImageInfo, num_descriptors);

				for (uint32_t j = 0; j < num_descriptors; j++) {
					vk_img_infos[j] = {};
					vk_img_infos[j].imageView = ((const TextureInfo *)uniform.ids[j].id)->vk_view;
					vk_img_infos[j].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				}

				vk_writes[i].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
				vk_writes[i].pImageInfo = vk_img_infos;
			} break;
			case UNIFORM_TYPE_IMAGE: {
				num_descriptors = uniform.ids.size();
				VkDescriptorImageInfo *vk_img_infos = ALLOCA_ARRAY(VkDescriptorImageInfo, num_descriptors);

				for (uint32_t j = 0; j < num_descriptors; j++) {
					vk_img_infos[j] = {};
					vk_img_infos[j].imageView = ((const TextureInfo *)uniform.ids[j].id)->vk_view;
					vk_img_infos[j].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
				}

				vk_writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
				vk_writes[i].pImageInfo = vk_img_infos;
			} break;
			case UNIFORM_TYPE_TEXTURE_BUFFER: {
				num_descriptors = uniform.ids.size();
				VkDescriptorBufferInfo *vk_buf_infos = ALLOCA_ARRAY(VkDescriptorBufferInfo, num_descriptors);
				VkBufferView *vk_buf_views = ALLOCA_ARRAY(VkBufferView, num_descriptors);

				for (uint32_t j = 0; j < num_descriptors; j++) {
					const BufferInfo *buf_info = (const BufferInfo *)uniform.ids[j].id;
					vk_buf_infos[j] = {};
					vk_buf_infos[j].buffer = buf_info->vk_buffer;
					vk_buf_infos[j].range = buf_info->size;

					vk_buf_views[j] = buf_info->vk_view;
				}

				vk_writes[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
				vk_writes[i].pBufferInfo = vk_buf_infos;
				vk_writes[i].pTexelBufferView = vk_buf_views;
			} break;
			case UNIFORM_TYPE_SAMPLER_WITH_TEXTURE_BUFFER: {
				num_descriptors = uniform.ids.size() / 2;
				VkDescriptorImageInfo *vk_img_infos = ALLOCA_ARRAY(VkDescriptorImageInfo, num_descriptors);
				VkDescriptorBufferInfo *vk_buf_infos = ALLOCA_ARRAY(VkDescriptorBufferInfo, num_descriptors);
				VkBufferView *vk_buf_views = ALLOCA_ARRAY(VkBufferView, num_descriptors);

				for (uint32_t j = 0; j < num_descriptors; j++) {
					vk_img_infos[j] = {};
					vk_img_infos[j].sampler = (VkSampler)uniform.ids[j * 2 + 0].id;

					const BufferInfo *buf_info = (const BufferInfo *)uniform.ids[j * 2 + 1].id;
					vk_buf_infos[j] = {};
					vk_buf_infos[j].buffer = buf_info->vk_buffer;
					vk_buf_infos[j].range = buf_info->size;

					vk_buf_views[j] = buf_info->vk_view;
				}

				vk_writes[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
				vk_writes[i].pImageInfo = vk_img_infos;
				vk_writes[i].pBufferInfo = vk_buf_infos;
				vk_writes[i].pTexelBufferView = vk_buf_views;
			} break;
			case UNIFORM_TYPE_IMAGE_BUFFER: {
				CRASH_NOW_MSG("Unimplemented!"); // TODO.
			} break;
			case UNIFORM_TYPE_UNIFORM_BUFFER: {
				const BufferInfo *buf_info = (const BufferInfo *)uniform.ids[0].id;
				VkDescriptorBufferInfo *vk_buf_info = ALLOCA_SINGLE(VkDescriptorBufferInfo);
				*vk_buf_info = {};
				vk_buf_info->buffer = buf_info->vk_buffer;
				vk_buf_info->range = buf_info->size;

				vk_writes[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				vk_writes[i].pBufferInfo = vk_buf_info;
			} break;
			case UNIFORM_TYPE_STORAGE_BUFFER: {
				const BufferInfo *buf_info = (const BufferInfo *)uniform.ids[0].id;
				VkDescriptorBufferInfo *vk_buf_info = ALLOCA_SINGLE(VkDescriptorBufferInfo);
				*vk_buf_info = {};
				vk_buf_info->buffer = buf_info->vk_buffer;
				vk_buf_info->range = buf_info->size;

				vk_writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				vk_writes[i].pBufferInfo = vk_buf_info;
			} break;
			case UNIFORM_TYPE_INPUT_ATTACHMENT: {
				num_descriptors = uniform.ids.size();
				VkDescriptorImageInfo *vk_img_infos = ALLOCA_ARRAY(VkDescriptorImageInfo, num_descriptors);

				for (uint32_t j = 0; j < uniform.ids.size(); j++) {
					vk_img_infos[j] = {};
					vk_img_infos[j].imageView = ((const TextureInfo *)uniform.ids[j].id)->vk_view;
					vk_img_infos[j].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				}

				vk_writes[i].descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
				vk_writes[i].pImageInfo = vk_img_infos;
			} break;
			default: {
				DEV_ASSERT(false);
			}
		}

		vk_writes[i].descriptorCount = num_descriptors;

		ERR_FAIL_COND_V_MSG(pool_key.uniform_type[uniform.type] == MAX_UNIFORM_POOL_ELEMENT, UniformSetID(),
				"Uniform set reached the limit of bindings for the same type (" + itos(MAX_UNIFORM_POOL_ELEMENT) + ").");
		pool_key.uniform_type[uniform.type] += num_descriptors;
	}

	// Need a descriptor pool.
	DescriptorSetPools::Iterator pool_sets_it = {};
	VkDescriptorPool vk_pool = _descriptor_set_pool_find_or_create(pool_key, &pool_sets_it);
	DEV_ASSERT(vk_pool);
	pool_sets_it->value[vk_pool]++;

	VkDescriptorSetAllocateInfo descriptor_set_allocate_info = {};
	descriptor_set_allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	descriptor_set_allocate_info.descriptorPool = vk_pool;
	descriptor_set_allocate_info.descriptorSetCount = 1;
	const ShaderInfo *shader_info = (const ShaderInfo *)p_shader.id;
	descriptor_set_allocate_info.pSetLayouts = &shader_info->vk_descriptor_set_layouts[p_set_index];

	VkDescriptorSet vk_descriptor_set = VK_NULL_HANDLE;
	VkResult res = vkAllocateDescriptorSets(vk_device, &descriptor_set_allocate_info, &vk_descriptor_set);
	if (res) {
		_descriptor_set_pool_unreference(pool_sets_it, vk_pool);
		ERR_FAIL_V_MSG(UniformSetID(), "Cannot allocate descriptor sets, error " + itos(res) + ".");
	}

	for (uint32_t i = 0; i < p_uniforms.size(); i++) {
		vk_writes[i].dstSet = vk_descriptor_set;
	}
	vkUpdateDescriptorSets(vk_device, p_uniforms.size(), vk_writes, 0, nullptr);

	// Bookkeep.

	UniformSetInfo *usi = VersatileResource::allocate<UniformSetInfo>(resources_allocator);
	usi->vk_descriptor_set = vk_descriptor_set;
	usi->vk_descriptor_pool = vk_pool;
	usi->pool_sets_it = pool_sets_it;

	return UniformSetID(usi);
}

void RenderingDeviceDriverVulkan::uniform_set_free(UniformSetID p_uniform_set) {
	UniformSetInfo *usi = (UniformSetInfo *)p_uniform_set.id;
	vkFreeDescriptorSets(vk_device, usi->vk_descriptor_pool, 1, &usi->vk_descriptor_set);

	_descriptor_set_pool_unreference(usi->pool_sets_it, usi->vk_descriptor_pool);

	VersatileResource::free(resources_allocator, usi);
}

// ----- COMMANDS -----

void RenderingDeviceDriverVulkan::command_uniform_set_prepare_for_use(CommandBufferID p_cmd_buffer, UniformSetID p_uniform_set, ShaderID p_shader, uint32_t p_set_index) {
}

/******************/
/**** TRANSFER ****/
/******************/

static_assert(ARRAYS_COMPATIBLE_FIELDWISE(RDD::BufferCopyRegion, VkBufferCopy));

static void _texture_subresource_range_to_vk(const RDD::TextureSubresourceRange &p_subresources, VkImageSubresourceRange *r_vk_subreources) {
	*r_vk_subreources = {};
	r_vk_subreources->aspectMask = (VkImageAspectFlags)p_subresources.aspect;
	r_vk_subreources->baseMipLevel = p_subresources.base_mipmap;
	r_vk_subreources->levelCount = p_subresources.mipmap_count;
	r_vk_subreources->baseArrayLayer = p_subresources.base_layer;
	r_vk_subreources->layerCount = p_subresources.layer_count;
}

static void _texture_subresource_layers_to_vk(const RDD::TextureSubresourceLayers &p_subresources, VkImageSubresourceLayers *r_vk_subreources) {
	*r_vk_subreources = {};
	r_vk_subreources->aspectMask = (VkImageAspectFlags)p_subresources.aspect;
	r_vk_subreources->mipLevel = p_subresources.mipmap;
	r_vk_subreources->baseArrayLayer = p_subresources.base_layer;
	r_vk_subreources->layerCount = p_subresources.layer_count;
}

static void _buffer_texture_copy_region_to_vk(const RDD::BufferTextureCopyRegion &p_copy_region, VkBufferImageCopy *r_vk_copy_region) {
	*r_vk_copy_region = {};
	r_vk_copy_region->bufferOffset = p_copy_region.buffer_offset;
	_texture_subresource_layers_to_vk(p_copy_region.texture_subresources, &r_vk_copy_region->imageSubresource);
	r_vk_copy_region->imageOffset.x = p_copy_region.texture_offset.x;
	r_vk_copy_region->imageOffset.y = p_copy_region.texture_offset.y;
	r_vk_copy_region->imageOffset.z = p_copy_region.texture_offset.z;
	r_vk_copy_region->imageExtent.width = p_copy_region.texture_region_size.x;
	r_vk_copy_region->imageExtent.height = p_copy_region.texture_region_size.y;
	r_vk_copy_region->imageExtent.depth = p_copy_region.texture_region_size.z;
}

static void _texture_copy_region_to_vk(const RDD::TextureCopyRegion &p_copy_region, VkImageCopy *r_vk_copy_region) {
	*r_vk_copy_region = {};
	_texture_subresource_layers_to_vk(p_copy_region.src_subresources, &r_vk_copy_region->srcSubresource);
	r_vk_copy_region->srcOffset.x = p_copy_region.src_offset.x;
	r_vk_copy_region->srcOffset.y = p_copy_region.src_offset.y;
	r_vk_copy_region->srcOffset.z = p_copy_region.src_offset.z;
	_texture_subresource_layers_to_vk(p_copy_region.dst_subresources, &r_vk_copy_region->dstSubresource);
	r_vk_copy_region->dstOffset.x = p_copy_region.dst_offset.x;
	r_vk_copy_region->dstOffset.y = p_copy_region.dst_offset.y;
	r_vk_copy_region->dstOffset.z = p_copy_region.dst_offset.z;
	r_vk_copy_region->extent.width = p_copy_region.size.x;
	r_vk_copy_region->extent.height = p_copy_region.size.y;
	r_vk_copy_region->extent.depth = p_copy_region.size.z;
}

void RenderingDeviceDriverVulkan::command_clear_buffer(CommandBufferID p_cmd_buffer, BufferID p_buffer, uint64_t p_offset, uint64_t p_size) {
	const BufferInfo *buf_info = (const BufferInfo *)p_buffer.id;
	vkCmdFillBuffer((VkCommandBuffer)p_cmd_buffer.id, buf_info->vk_buffer, p_offset, p_size, 0);
}

void RenderingDeviceDriverVulkan::command_copy_buffer(CommandBufferID p_cmd_buffer, BufferID p_src_buffer, BufferID p_dst_buffer, VectorView<BufferCopyRegion> p_regions) {
	const BufferInfo *src_buf_info = (const BufferInfo *)p_src_buffer.id;
	const BufferInfo *dst_buf_info = (const BufferInfo *)p_dst_buffer.id;
	vkCmdCopyBuffer((VkCommandBuffer)p_cmd_buffer.id, src_buf_info->vk_buffer, dst_buf_info->vk_buffer, p_regions.size(), (const VkBufferCopy *)p_regions.ptr());
}

void RenderingDeviceDriverVulkan::command_copy_texture(CommandBufferID p_cmd_buffer, TextureID p_src_texture, TextureLayout p_src_texture_layout, TextureID p_dst_texture, TextureLayout p_dst_texture_layout, VectorView<TextureCopyRegion> p_regions) {
	VkImageCopy *vk_copy_regions = ALLOCA_ARRAY(VkImageCopy, p_regions.size());
	for (uint32_t i = 0; i < p_regions.size(); i++) {
		_texture_copy_region_to_vk(p_regions[i], &vk_copy_regions[i]);
	}

	const TextureInfo *src_tex_info = (const TextureInfo *)p_src_texture.id;
	const TextureInfo *dst_tex_info = (const TextureInfo *)p_dst_texture.id;
	vkCmdCopyImage((VkCommandBuffer)p_cmd_buffer.id, src_tex_info->vk_view_create_info.image, (VkImageLayout)p_src_texture_layout, dst_tex_info->vk_view_create_info.image, (VkImageLayout)p_dst_texture_layout, p_regions.size(), vk_copy_regions);
}

void RenderingDeviceDriverVulkan::command_resolve_texture(CommandBufferID p_cmd_buffer, TextureID p_src_texture, TextureLayout p_src_texture_layout, uint32_t p_src_layer, uint32_t p_src_mipmap, TextureID p_dst_texture, TextureLayout p_dst_texture_layout, uint32_t p_dst_layer, uint32_t p_dst_mipmap) {
	const TextureInfo *src_tex_info = (const TextureInfo *)p_src_texture.id;
	const TextureInfo *dst_tex_info = (const TextureInfo *)p_dst_texture.id;

	VkImageResolve vk_resolve = {};
	vk_resolve.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	vk_resolve.srcSubresource.mipLevel = p_src_mipmap;
	vk_resolve.srcSubresource.baseArrayLayer = p_src_layer;
	vk_resolve.srcSubresource.layerCount = 1;
	vk_resolve.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	vk_resolve.dstSubresource.mipLevel = p_dst_mipmap;
	vk_resolve.dstSubresource.baseArrayLayer = p_dst_layer;
	vk_resolve.dstSubresource.layerCount = 1;
	vk_resolve.extent.width = MAX(1u, src_tex_info->vk_create_info.extent.width >> p_src_mipmap);
	vk_resolve.extent.height = MAX(1u, src_tex_info->vk_create_info.extent.height >> p_src_mipmap);
	vk_resolve.extent.depth = MAX(1u, src_tex_info->vk_create_info.extent.depth >> p_src_mipmap);

	vkCmdResolveImage((VkCommandBuffer)p_cmd_buffer.id, src_tex_info->vk_view_create_info.image, (VkImageLayout)p_src_texture_layout, dst_tex_info->vk_view_create_info.image, (VkImageLayout)p_dst_texture_layout, 1, &vk_resolve);
}

void RenderingDeviceDriverVulkan::command_clear_color_texture(CommandBufferID p_cmd_buffer, TextureID p_texture, TextureLayout p_texture_layout, const Color &p_color, const TextureSubresourceRange &p_subresources) {
	VkClearColorValue vk_color = {};
	memcpy(&vk_color.float32, p_color.components, sizeof(VkClearColorValue::float32));

	VkImageSubresourceRange vk_subresources = {};
	_texture_subresource_range_to_vk(p_subresources, &vk_subresources);

	const TextureInfo *tex_info = (const TextureInfo *)p_texture.id;
	vkCmdClearColorImage((VkCommandBuffer)p_cmd_buffer.id, tex_info->vk_view_create_info.image, (VkImageLayout)p_texture_layout, &vk_color, 1, &vk_subresources);
}

void RenderingDeviceDriverVulkan::command_copy_buffer_to_texture(CommandBufferID p_cmd_buffer, BufferID p_src_buffer, TextureID p_dst_texture, TextureLayout p_dst_texture_layout, VectorView<BufferTextureCopyRegion> p_regions) {
	VkBufferImageCopy *vk_copy_regions = ALLOCA_ARRAY(VkBufferImageCopy, p_regions.size());
	for (uint32_t i = 0; i < p_regions.size(); i++) {
		_buffer_texture_copy_region_to_vk(p_regions[i], &vk_copy_regions[i]);
	}

	const BufferInfo *buf_info = (const BufferInfo *)p_src_buffer.id;
	const TextureInfo *tex_info = (const TextureInfo *)p_dst_texture.id;
	vkCmdCopyBufferToImage((VkCommandBuffer)p_cmd_buffer.id, buf_info->vk_buffer, tex_info->vk_view_create_info.image, (VkImageLayout)p_dst_texture_layout, p_regions.size(), vk_copy_regions);
}

void RenderingDeviceDriverVulkan::command_copy_texture_to_buffer(CommandBufferID p_cmd_buffer, TextureID p_src_texture, TextureLayout p_src_texture_layout, BufferID p_dst_buffer, VectorView<BufferTextureCopyRegion> p_regions) {
	VkBufferImageCopy *vk_copy_regions = ALLOCA_ARRAY(VkBufferImageCopy, p_regions.size());
	for (uint32_t i = 0; i < p_regions.size(); i++) {
		_buffer_texture_copy_region_to_vk(p_regions[i], &vk_copy_regions[i]);
	}

	const TextureInfo *tex_info = (const TextureInfo *)p_src_texture.id;
	const BufferInfo *buf_info = (const BufferInfo *)p_dst_buffer.id;
	vkCmdCopyImageToBuffer((VkCommandBuffer)p_cmd_buffer.id, tex_info->vk_view_create_info.image, (VkImageLayout)p_src_texture_layout, buf_info->vk_buffer, p_regions.size(), vk_copy_regions);
}

/******************/
/**** PIPELINE ****/
/******************/

void RenderingDeviceDriverVulkan::pipeline_free(PipelineID p_pipeline) {
	vkDestroyPipeline(vk_device, (VkPipeline)p_pipeline.id, nullptr);
}

// ----- BINDING -----

void RenderingDeviceDriverVulkan::command_bind_push_constants(CommandBufferID p_cmd_buffer, ShaderID p_shader, uint32_t p_dst_first_index, VectorView<uint32_t> p_data) {
	const ShaderInfo *shader_info = (const ShaderInfo *)p_shader.id;
	vkCmdPushConstants((VkCommandBuffer)p_cmd_buffer.id, shader_info->vk_pipeline_layout, shader_info->vk_push_constant_stages, p_dst_first_index * sizeof(uint32_t), p_data.size() * sizeof(uint32_t), p_data.ptr());
}

// ----- CACHE -----

int RenderingDeviceDriverVulkan::caching_instance_count = 0;

bool RenderingDeviceDriverVulkan::pipeline_cache_create(const Vector<uint8_t> &p_data) {
	if (caching_instance_count) {
		WARN_PRINT("There's already a RenderingDeviceDriverVulkan instance doing PSO caching. Only one can at the same time. This one won't.");
		return false;
	}
	caching_instance_count++;

	pipelines_cache.current_size = 0;
	pipelines_cache.buffer.resize(sizeof(PipelineCacheHeader));

	// Parse.
	{
		if (p_data.size() <= (int)sizeof(PipelineCacheHeader)) {
			WARN_PRINT("Invalid/corrupt pipelines cache.");
		} else {
			const PipelineCacheHeader *loaded_header = reinterpret_cast<const PipelineCacheHeader *>(p_data.ptr());
			if (loaded_header->magic != 868 + VK_PIPELINE_CACHE_HEADER_VERSION_ONE) {
				WARN_PRINT("Invalid pipelines cache magic number.");
			} else {
				const uint8_t *loaded_buffer_start = p_data.ptr() + sizeof(PipelineCacheHeader);
				uint32_t loaded_buffer_size = p_data.size() - sizeof(PipelineCacheHeader);
				const PipelineCacheHeader *current_header = (PipelineCacheHeader *)pipelines_cache.buffer.ptr();
				if (loaded_header->data_hash != hash_murmur3_buffer(loaded_buffer_start, loaded_buffer_size) ||
						loaded_header->data_size != loaded_buffer_size ||
						loaded_header->vendor_id != current_header->vendor_id ||
						loaded_header->device_id != current_header->device_id ||
						loaded_header->driver_version != current_header->driver_version ||
						memcmp(loaded_header->uuid, current_header->uuid, VK_UUID_SIZE) != 0 ||
						loaded_header->driver_abi != current_header->driver_abi) {
					WARN_PRINT("Invalid pipelines cache header.");
				} else {
					pipelines_cache.current_size = loaded_buffer_size;
					pipelines_cache.buffer = p_data;
				}
			}
		}
	}

	// Create.
	{
		VkPipelineCacheCreateInfo cache_info = {};
		cache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
		if (context->get_pipeline_cache_control_support()) {
			cache_info.flags = VK_PIPELINE_CACHE_CREATE_EXTERNALLY_SYNCHRONIZED_BIT;
		}
		cache_info.initialDataSize = pipelines_cache.buffer.size() - sizeof(PipelineCacheHeader);
		cache_info.pInitialData = pipelines_cache.buffer.ptr() + sizeof(PipelineCacheHeader);

		VkResult err = vkCreatePipelineCache(vk_device, &cache_info, nullptr, &pipelines_cache.vk_cache);
		if (err != VK_SUCCESS) {
			WARN_PRINT("vkCreatePipelinecache failed with error " + itos(err) + ".");
			return false;
		}
	}

	return true;
}

void RenderingDeviceDriverVulkan::pipeline_cache_free() {
	DEV_ASSERT(pipelines_cache.vk_cache);

	vkDestroyPipelineCache(vk_device, pipelines_cache.vk_cache, nullptr);

	DEV_ASSERT(caching_instance_count > 0);
	caching_instance_count--;
}

size_t RenderingDeviceDriverVulkan::pipeline_cache_query_size() {
	DEV_ASSERT(pipelines_cache.vk_cache);

	// FIXME:
	// We're letting the cache grow unboundedly. We may want to set at limit and see if implementations use LRU or the like.
	// If we do, we won't be able to assume any longer that the cache is dirty if, and only if, it has grown.
	VkResult err = vkGetPipelineCacheData(vk_device, pipelines_cache.vk_cache, &pipelines_cache.current_size, nullptr);
	ERR_FAIL_COND_V_MSG(err, 0, "vkGetPipelineCacheData failed with error " + itos(err) + ".");

	return pipelines_cache.current_size;
}

Vector<uint8_t> RenderingDeviceDriverVulkan::pipeline_cache_serialize() {
	DEV_ASSERT(pipelines_cache.vk_cache);

	pipelines_cache.buffer.resize(pipelines_cache.current_size + sizeof(PipelineCacheHeader));

	VkResult err = vkGetPipelineCacheData(vk_device, pipelines_cache.vk_cache, &pipelines_cache.current_size, pipelines_cache.buffer.ptrw() + sizeof(PipelineCacheHeader));
	ERR_FAIL_COND_V(err != VK_SUCCESS && err != VK_INCOMPLETE, Vector<uint8_t>()); // Incomplete is OK because the cache may have grown since the size was queried (unless when exiting).

	// The real buffer size may now be bigger than the updated current_size.
	// We take into account the new size but keep the buffer resized in a worst-case fashion.

	PipelineCacheHeader *header = (PipelineCacheHeader *)pipelines_cache.buffer.ptrw();
	header->data_size = pipelines_cache.current_size;
	header->data_hash = hash_murmur3_buffer(pipelines_cache.buffer.ptr() + sizeof(PipelineCacheHeader), pipelines_cache.current_size);

	return pipelines_cache.buffer;
}

/*******************/
/**** RENDERING ****/
/*******************/

// ----- SUBPASS -----

// RDD::AttachmentLoadOp == VkAttachmentLoadOp.
static_assert(ENUM_MEMBERS_EQUAL(RDD::ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_LOAD_OP_LOAD));
static_assert(ENUM_MEMBERS_EQUAL(RDD::ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_LOAD_OP_CLEAR));
static_assert(ENUM_MEMBERS_EQUAL(RDD::ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_LOAD_OP_DONT_CARE));

// RDD::AttachmentStoreOp == VkAttachmentStoreOp.
static_assert(ENUM_MEMBERS_EQUAL(RDD::ATTACHMENT_STORE_OP_STORE, VK_ATTACHMENT_STORE_OP_STORE));
static_assert(ENUM_MEMBERS_EQUAL(RDD::ATTACHMENT_STORE_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE));

// Assuming Vulkan and RDD's are backed by uint32_t in:
// - VkSubpassDescription2::pPreserveAttachments and RDD::Subpass::preserve_attachments.
// - VkRenderPassCreateInfo2KHR::pCorrelatedViewMasks and p_view_correlation_mask.

static void _attachment_reference_to_vk(const RDD::AttachmentReference &p_attachment_reference, VkAttachmentReference2KHR *r_vk_attachment_reference) {
	*r_vk_attachment_reference = {};
	r_vk_attachment_reference->sType = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2_KHR;
	r_vk_attachment_reference->attachment = p_attachment_reference.attachment;
	r_vk_attachment_reference->layout = (VkImageLayout)p_attachment_reference.layout;
	r_vk_attachment_reference->aspectMask = (VkImageAspectFlags)p_attachment_reference.aspect;
}

RDD::RenderPassID RenderingDeviceDriverVulkan::render_pass_create(VectorView<Attachment> p_attachments, VectorView<Subpass> p_subpasses, VectorView<SubpassDependency> p_subpass_dependencies, uint32_t p_view_count) {
	// These are only used if we use multiview but we need to define them in scope.
	const uint32_t view_mask = (1 << p_view_count) - 1;
	const uint32_t correlation_mask = (1 << p_view_count) - 1;

	VkAttachmentDescription2KHR *vk_attachments = ALLOCA_ARRAY(VkAttachmentDescription2KHR, p_attachments.size());
	for (uint32_t i = 0; i < p_attachments.size(); i++) {
		vk_attachments[i] = {};
		vk_attachments[i].sType = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2_KHR;
		vk_attachments[i].format = RD_TO_VK_FORMAT[p_attachments[i].format];
		vk_attachments[i].samples = _ensure_supported_sample_count(p_attachments[i].samples);
		vk_attachments[i].loadOp = (VkAttachmentLoadOp)p_attachments[i].load_op;
		vk_attachments[i].storeOp = (VkAttachmentStoreOp)p_attachments[i].store_op;
		vk_attachments[i].stencilLoadOp = (VkAttachmentLoadOp)p_attachments[i].stencil_load_op;
		vk_attachments[i].stencilStoreOp = (VkAttachmentStoreOp)p_attachments[i].stencil_store_op;
		vk_attachments[i].initialLayout = (VkImageLayout)p_attachments[i].initial_layout;
		vk_attachments[i].finalLayout = (VkImageLayout)p_attachments[i].final_layout;
	}

	VkSubpassDescription2KHR *vk_subpasses = ALLOCA_ARRAY(VkSubpassDescription2KHR, p_subpasses.size());
	for (uint32_t i = 0; i < p_subpasses.size(); i++) {
		VkAttachmentReference2KHR *vk_subpass_input_attachments = ALLOCA_ARRAY(VkAttachmentReference2KHR, p_subpasses[i].input_references.size());
		for (uint32_t j = 0; j < p_subpasses[i].input_references.size(); j++) {
			_attachment_reference_to_vk(p_subpasses[i].input_references[j], &vk_subpass_input_attachments[j]);
		}

		VkAttachmentReference2KHR *vk_subpass_color_attachments = ALLOCA_ARRAY(VkAttachmentReference2KHR, p_subpasses[i].color_references.size());
		for (uint32_t j = 0; j < p_subpasses[i].color_references.size(); j++) {
			_attachment_reference_to_vk(p_subpasses[i].color_references[j], &vk_subpass_color_attachments[j]);
		}

		VkAttachmentReference2KHR *vk_subpass_resolve_attachments = ALLOCA_ARRAY(VkAttachmentReference2KHR, p_subpasses[i].resolve_references.size());
		for (uint32_t j = 0; j < p_subpasses[i].resolve_references.size(); j++) {
			_attachment_reference_to_vk(p_subpasses[i].resolve_references[j], &vk_subpass_resolve_attachments[j]);
		}

		VkAttachmentReference2KHR *vk_subpass_depth_stencil_attachment = nullptr;
		if (p_subpasses[i].depth_stencil_reference.attachment != AttachmentReference::UNUSED) {
			vk_subpass_depth_stencil_attachment = ALLOCA_SINGLE(VkAttachmentReference2KHR);
			_attachment_reference_to_vk(p_subpasses[i].depth_stencil_reference, vk_subpass_depth_stencil_attachment);
		}

		vk_subpasses[i] = {};
		vk_subpasses[i].sType = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2_KHR;
		vk_subpasses[i].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		vk_subpasses[i].viewMask = p_view_count == 1 ? 0 : view_mask;
		vk_subpasses[i].inputAttachmentCount = p_subpasses[i].input_references.size();
		vk_subpasses[i].pInputAttachments = vk_subpass_input_attachments;
		vk_subpasses[i].colorAttachmentCount = p_subpasses[i].color_references.size();
		vk_subpasses[i].pColorAttachments = vk_subpass_color_attachments;
		vk_subpasses[i].pResolveAttachments = vk_subpass_resolve_attachments;
		vk_subpasses[i].pDepthStencilAttachment = vk_subpass_depth_stencil_attachment;
		vk_subpasses[i].preserveAttachmentCount = p_subpasses[i].preserve_attachments.size();
		vk_subpasses[i].pPreserveAttachments = p_subpasses[i].preserve_attachments.ptr();

		// VRS.
		if (context->get_vrs_capabilities().attachment_vrs_supported && p_subpasses[i].vrs_reference.attachment != AttachmentReference::UNUSED) {
			VkAttachmentReference2KHR *vk_subpass_vrs_attachment = ALLOCA_SINGLE(VkAttachmentReference2KHR);
			*vk_subpass_vrs_attachment = {};
			vk_subpass_vrs_attachment->sType = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2_KHR;
			vk_subpass_vrs_attachment->attachment = p_subpasses[i].vrs_reference.attachment;
			vk_subpass_vrs_attachment->layout = VK_IMAGE_LAYOUT_FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR;

			VkFragmentShadingRateAttachmentInfoKHR *vk_vrs_info = ALLOCA_SINGLE(VkFragmentShadingRateAttachmentInfoKHR);
			*vk_vrs_info = {};
			vk_vrs_info->sType = VK_STRUCTURE_TYPE_FRAGMENT_SHADING_RATE_ATTACHMENT_INFO_KHR;
			vk_vrs_info->pFragmentShadingRateAttachment = vk_subpass_vrs_attachment;
			vk_vrs_info->shadingRateAttachmentTexelSize.width = context->get_vrs_capabilities().texel_size.x;
			vk_vrs_info->shadingRateAttachmentTexelSize.height = context->get_vrs_capabilities().texel_size.y;

			vk_subpasses[i].pNext = vk_vrs_info;
		}
	}

	VkSubpassDependency2KHR *vk_subpass_dependencies = ALLOCA_ARRAY(VkSubpassDependency2KHR, p_subpass_dependencies.size());
	for (uint32_t i = 0; i < p_subpass_dependencies.size(); i++) {
		vk_subpass_dependencies[i] = {};
		vk_subpass_dependencies[i].sType = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2;
		vk_subpass_dependencies[i].srcSubpass = p_subpass_dependencies[i].src_subpass;
		vk_subpass_dependencies[i].dstSubpass = p_subpass_dependencies[i].dst_subpass;
		vk_subpass_dependencies[i].srcStageMask = (VkPipelineStageFlags)p_subpass_dependencies[i].src_stages;
		vk_subpass_dependencies[i].dstStageMask = (VkPipelineStageFlags)p_subpass_dependencies[i].dst_stages;
		vk_subpass_dependencies[i].srcAccessMask = (VkAccessFlags)p_subpass_dependencies[i].src_access;
		vk_subpass_dependencies[i].dstAccessMask = (VkAccessFlags)p_subpass_dependencies[i].dst_access;
	}

	VkRenderPassCreateInfo2KHR create_info = {};
	create_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2_KHR;
	create_info.attachmentCount = p_attachments.size();
	create_info.pAttachments = vk_attachments;
	create_info.subpassCount = p_subpasses.size();
	create_info.pSubpasses = vk_subpasses;
	create_info.dependencyCount = p_subpass_dependencies.size();
	create_info.pDependencies = vk_subpass_dependencies;
	create_info.correlatedViewMaskCount = p_view_count == 1 ? 0 : 1;
	create_info.pCorrelatedViewMasks = p_view_count == 1 ? nullptr : &correlation_mask;

	// Multiview.
	if (p_view_count > 1 && !context->supports_renderpass2()) {
		// This is only required when using vkCreateRenderPass.
		// We add it if vkCreateRenderPass2KHR is not supported,
		// resulting this in being passed to our vkCreateRenderPass fallback.

		uint32_t *vk_view_masks = ALLOCA_ARRAY(uint32_t, p_subpasses.size());
		for (uint32_t i = 0; i < p_subpasses.size(); i++) {
			vk_view_masks[i] = view_mask;
		}

		VkRenderPassMultiviewCreateInfo *multiview_create_info = ALLOCA_SINGLE(VkRenderPassMultiviewCreateInfo);
		*multiview_create_info = {};
		multiview_create_info->sType = VK_STRUCTURE_TYPE_RENDER_PASS_MULTIVIEW_CREATE_INFO;
		multiview_create_info->subpassCount = p_subpasses.size();
		multiview_create_info->pViewMasks = vk_view_masks;
		multiview_create_info->correlationMaskCount = 1;
		multiview_create_info->pCorrelationMasks = &correlation_mask;

		create_info.pNext = multiview_create_info;
	}

	VkRenderPass vk_render_pass = VK_NULL_HANDLE;
	VkResult res = context->vkCreateRenderPass2KHR(vk_device, &create_info, nullptr, &vk_render_pass);
	ERR_FAIL_COND_V_MSG(res, RenderPassID(), "vkCreateRenderPass2KHR failed with error " + itos(res) + ".");

	return RenderPassID(vk_render_pass);
}

void RenderingDeviceDriverVulkan::render_pass_free(RenderPassID p_render_pass) {
	vkDestroyRenderPass(vk_device, (VkRenderPass)p_render_pass.id, nullptr);
}

// ----- COMMANDS -----

static_assert(ARRAYS_COMPATIBLE_FIELDWISE(RDD::RenderPassClearValue, VkClearValue));

void RenderingDeviceDriverVulkan::command_begin_render_pass(CommandBufferID p_cmd_buffer, RenderPassID p_render_pass, FramebufferID p_framebuffer, CommandBufferType p_cmd_buffer_type, const Rect2i &p_rect, VectorView<RenderPassClearValue> p_clear_values) {
	VkRenderPassBeginInfo render_pass_begin = {};
	render_pass_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	render_pass_begin.renderPass = (VkRenderPass)p_render_pass.id;
	render_pass_begin.framebuffer = (VkFramebuffer)p_framebuffer.id;

	render_pass_begin.renderArea.offset.x = p_rect.position.x;
	render_pass_begin.renderArea.offset.y = p_rect.position.y;
	render_pass_begin.renderArea.extent.width = p_rect.size.x;
	render_pass_begin.renderArea.extent.height = p_rect.size.y;

	render_pass_begin.clearValueCount = p_clear_values.size();
	render_pass_begin.pClearValues = (const VkClearValue *)p_clear_values.ptr();

	VkSubpassContents vk_subpass_contents = p_cmd_buffer_type == COMMAND_BUFFER_TYPE_PRIMARY ? VK_SUBPASS_CONTENTS_INLINE : VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS;
	vkCmdBeginRenderPass((VkCommandBuffer)p_cmd_buffer.id, &render_pass_begin, vk_subpass_contents);

#if PRINT_NATIVE_COMMANDS
	print_line(vformat("vkCmdBeginRenderPass Pass 0x%uX Framebuffer 0x%uX", p_render_pass.id, p_framebuffer.id));
#endif
}

void RenderingDeviceDriverVulkan::command_end_render_pass(CommandBufferID p_cmd_buffer) {
	vkCmdEndRenderPass((VkCommandBuffer)p_cmd_buffer.id);

#if PRINT_NATIVE_COMMANDS
	print_line("vkCmdEndRenderPass");
#endif
}

void RenderingDeviceDriverVulkan::command_next_render_subpass(CommandBufferID p_cmd_buffer, CommandBufferType p_cmd_buffer_type) {
	VkSubpassContents vk_subpass_contents = p_cmd_buffer_type == COMMAND_BUFFER_TYPE_PRIMARY ? VK_SUBPASS_CONTENTS_INLINE : VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS;
	vkCmdNextSubpass((VkCommandBuffer)p_cmd_buffer.id, vk_subpass_contents);
}

void RenderingDeviceDriverVulkan::command_render_set_viewport(CommandBufferID p_cmd_buffer, VectorView<Rect2i> p_viewports) {
	VkViewport *vk_viewports = ALLOCA_ARRAY(VkViewport, p_viewports.size());
	for (uint32_t i = 0; i < p_viewports.size(); i++) {
		vk_viewports[i] = {};
		vk_viewports[i].x = p_viewports[i].position.x;
		vk_viewports[i].y = p_viewports[i].position.y;
		vk_viewports[i].width = p_viewports[i].size.x;
		vk_viewports[i].height = p_viewports[i].size.y;
		vk_viewports[i].minDepth = 0.0f;
		vk_viewports[i].maxDepth = 1.0f;
	}
	vkCmdSetViewport((VkCommandBuffer)p_cmd_buffer.id, 0, p_viewports.size(), vk_viewports);
}

void RenderingDeviceDriverVulkan::command_render_set_scissor(CommandBufferID p_cmd_buffer, VectorView<Rect2i> p_scissors) {
	vkCmdSetScissor((VkCommandBuffer)p_cmd_buffer.id, 0, p_scissors.size(), (VkRect2D *)p_scissors.ptr());
}

void RenderingDeviceDriverVulkan::command_render_clear_attachments(CommandBufferID p_cmd_buffer, VectorView<AttachmentClear> p_attachment_clears, VectorView<Rect2i> p_rects) {
	VkClearAttachment *vk_clears = ALLOCA_ARRAY(VkClearAttachment, p_attachment_clears.size());
	for (uint32_t i = 0; i < p_attachment_clears.size(); i++) {
		vk_clears[i] = {};
		memcpy(&vk_clears[i].clearValue, &p_attachment_clears[i].value, sizeof(VkClearValue));
		vk_clears[i].colorAttachment = p_attachment_clears[i].color_attachment;
		vk_clears[i].aspectMask = p_attachment_clears[i].aspect;
	}

	VkClearRect *vk_rects = ALLOCA_ARRAY(VkClearRect, p_rects.size());
	for (uint32_t i = 0; i < p_rects.size(); i++) {
		vk_rects[i] = {};
		vk_rects[i].rect.offset.x = p_rects[i].position.x;
		vk_rects[i].rect.offset.y = p_rects[i].position.y;
		vk_rects[i].rect.extent.width = p_rects[i].size.x;
		vk_rects[i].rect.extent.height = p_rects[i].size.y;
		vk_rects[i].baseArrayLayer = 0;
		vk_rects[i].layerCount = 1;
	}

	vkCmdClearAttachments((VkCommandBuffer)p_cmd_buffer.id, p_attachment_clears.size(), vk_clears, p_rects.size(), vk_rects);
}

void RenderingDeviceDriverVulkan::command_bind_render_pipeline(CommandBufferID p_cmd_buffer, PipelineID p_pipeline) {
	vkCmdBindPipeline((VkCommandBuffer)p_cmd_buffer.id, VK_PIPELINE_BIND_POINT_GRAPHICS, (VkPipeline)p_pipeline.id);
}

void RenderingDeviceDriverVulkan::command_bind_render_uniform_set(CommandBufferID p_cmd_buffer, UniformSetID p_uniform_set, ShaderID p_shader, uint32_t p_set_index) {
	const ShaderInfo *shader_info = (const ShaderInfo *)p_shader.id;
	const UniformSetInfo *usi = (const UniformSetInfo *)p_uniform_set.id;
	vkCmdBindDescriptorSets((VkCommandBuffer)p_cmd_buffer.id, VK_PIPELINE_BIND_POINT_GRAPHICS, shader_info->vk_pipeline_layout, p_set_index, 1, &usi->vk_descriptor_set, 0, nullptr);
}

void RenderingDeviceDriverVulkan::command_render_draw(CommandBufferID p_cmd_buffer, uint32_t p_vertex_count, uint32_t p_instance_count, uint32_t p_base_vertex, uint32_t p_first_instance) {
	vkCmdDraw((VkCommandBuffer)p_cmd_buffer.id, p_vertex_count, p_instance_count, p_base_vertex, p_first_instance);
}

void RenderingDeviceDriverVulkan::command_render_draw_indexed(CommandBufferID p_cmd_buffer, uint32_t p_index_count, uint32_t p_instance_count, uint32_t p_first_index, int32_t p_vertex_offset, uint32_t p_first_instance) {
	vkCmdDrawIndexed((VkCommandBuffer)p_cmd_buffer.id, p_index_count, p_instance_count, p_first_index, p_vertex_offset, p_first_instance);
}

void RenderingDeviceDriverVulkan::command_render_draw_indexed_indirect(CommandBufferID p_cmd_buffer, BufferID p_indirect_buffer, uint64_t p_offset, uint32_t p_draw_count, uint32_t p_stride) {
	const BufferInfo *buf_info = (const BufferInfo *)p_indirect_buffer.id;
	vkCmdDrawIndexedIndirect((VkCommandBuffer)p_cmd_buffer.id, buf_info->vk_buffer, p_offset, p_draw_count, p_stride);
}

void RenderingDeviceDriverVulkan::command_render_draw_indexed_indirect_count(CommandBufferID p_cmd_buffer, BufferID p_indirect_buffer, uint64_t p_offset, BufferID p_count_buffer, uint64_t p_count_buffer_offset, uint32_t p_max_draw_count, uint32_t p_stride) {
	const BufferInfo *indirect_buf_info = (const BufferInfo *)p_indirect_buffer.id;
	const BufferInfo *count_buf_info = (const BufferInfo *)p_count_buffer.id;
	vkCmdDrawIndexedIndirectCount((VkCommandBuffer)p_cmd_buffer.id, indirect_buf_info->vk_buffer, p_offset, count_buf_info->vk_buffer, p_count_buffer_offset, p_max_draw_count, p_stride);
}

void RenderingDeviceDriverVulkan::command_render_draw_indirect(CommandBufferID p_cmd_buffer, BufferID p_indirect_buffer, uint64_t p_offset, uint32_t p_draw_count, uint32_t p_stride) {
	const BufferInfo *buf_info = (const BufferInfo *)p_indirect_buffer.id;
	vkCmdDrawIndirect((VkCommandBuffer)p_cmd_buffer.id, buf_info->vk_buffer, p_offset, p_draw_count, p_stride);
}

void RenderingDeviceDriverVulkan::command_render_draw_indirect_count(CommandBufferID p_cmd_buffer, BufferID p_indirect_buffer, uint64_t p_offset, BufferID p_count_buffer, uint64_t p_count_buffer_offset, uint32_t p_max_draw_count, uint32_t p_stride) {
	const BufferInfo *indirect_buf_info = (const BufferInfo *)p_indirect_buffer.id;
	const BufferInfo *count_buf_info = (const BufferInfo *)p_count_buffer.id;
	vkCmdDrawIndirectCount((VkCommandBuffer)p_cmd_buffer.id, indirect_buf_info->vk_buffer, p_offset, count_buf_info->vk_buffer, p_count_buffer_offset, p_max_draw_count, p_stride);
}

void RenderingDeviceDriverVulkan::command_render_bind_vertex_buffers(CommandBufferID p_cmd_buffer, uint32_t p_binding_count, const BufferID *p_buffers, const uint64_t *p_offsets) {
	VkBuffer *vk_buffers = ALLOCA_ARRAY(VkBuffer, p_binding_count);
	for (uint32_t i = 0; i < p_binding_count; i++) {
		vk_buffers[i] = ((const BufferInfo *)p_buffers[i].id)->vk_buffer;
	}
	vkCmdBindVertexBuffers((VkCommandBuffer)p_cmd_buffer.id, 0, p_binding_count, vk_buffers, p_offsets);
}

void RenderingDeviceDriverVulkan::command_render_bind_index_buffer(CommandBufferID p_cmd_buffer, BufferID p_buffer, IndexBufferFormat p_format, uint64_t p_offset) {
	const BufferInfo *buf_info = (const BufferInfo *)p_buffer.id;
	vkCmdBindIndexBuffer((VkCommandBuffer)p_cmd_buffer.id, buf_info->vk_buffer, p_offset, p_format == INDEX_BUFFER_FORMAT_UINT16 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
}

void RenderingDeviceDriverVulkan::command_render_set_blend_constants(CommandBufferID p_cmd_buffer, const Color &p_constants) {
	vkCmdSetBlendConstants((VkCommandBuffer)p_cmd_buffer.id, p_constants.components);
}

void RenderingDeviceDriverVulkan::command_render_set_line_width(CommandBufferID p_cmd_buffer, float p_width) {
	vkCmdSetLineWidth((VkCommandBuffer)p_cmd_buffer.id, p_width);
}

// ----- PIPELINE -----

static const VkPrimitiveTopology RD_TO_VK_PRIMITIVE[RDD::RENDER_PRIMITIVE_MAX] = {
	VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
	VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
	VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY,
	VK_PRIMITIVE_TOPOLOGY_LINE_STRIP,
	VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY,
	VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY,
	VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
	VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY,
	VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
	VK_PRIMITIVE_TOPOLOGY_PATCH_LIST,
};

// RDD::PolygonCullMode == VkCullModeFlagBits.
static_assert(ENUM_MEMBERS_EQUAL(RDD::POLYGON_CULL_DISABLED, VK_CULL_MODE_NONE));
static_assert(ENUM_MEMBERS_EQUAL(RDD::POLYGON_CULL_FRONT, VK_CULL_MODE_FRONT_BIT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::POLYGON_CULL_BACK, VK_CULL_MODE_BACK_BIT));

// RDD::StencilOperation == VkStencilOp.
static_assert(ENUM_MEMBERS_EQUAL(RDD::STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP));
static_assert(ENUM_MEMBERS_EQUAL(RDD::STENCIL_OP_ZERO, VK_STENCIL_OP_ZERO));
static_assert(ENUM_MEMBERS_EQUAL(RDD::STENCIL_OP_REPLACE, VK_STENCIL_OP_REPLACE));
static_assert(ENUM_MEMBERS_EQUAL(RDD::STENCIL_OP_INCREMENT_AND_CLAMP, VK_STENCIL_OP_INCREMENT_AND_CLAMP));
static_assert(ENUM_MEMBERS_EQUAL(RDD::STENCIL_OP_DECREMENT_AND_CLAMP, VK_STENCIL_OP_DECREMENT_AND_CLAMP));
static_assert(ENUM_MEMBERS_EQUAL(RDD::STENCIL_OP_INVERT, VK_STENCIL_OP_INVERT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::STENCIL_OP_INCREMENT_AND_WRAP, VK_STENCIL_OP_INCREMENT_AND_WRAP));
static_assert(ENUM_MEMBERS_EQUAL(RDD::STENCIL_OP_DECREMENT_AND_WRAP, VK_STENCIL_OP_DECREMENT_AND_WRAP));

// RDD::LogicOperation == VkLogicOp.
static_assert(ENUM_MEMBERS_EQUAL(RDD::LOGIC_OP_CLEAR, VK_LOGIC_OP_CLEAR));
static_assert(ENUM_MEMBERS_EQUAL(RDD::LOGIC_OP_AND, VK_LOGIC_OP_AND));
static_assert(ENUM_MEMBERS_EQUAL(RDD::LOGIC_OP_AND_REVERSE, VK_LOGIC_OP_AND_REVERSE));
static_assert(ENUM_MEMBERS_EQUAL(RDD::LOGIC_OP_COPY, VK_LOGIC_OP_COPY));
static_assert(ENUM_MEMBERS_EQUAL(RDD::LOGIC_OP_AND_INVERTED, VK_LOGIC_OP_AND_INVERTED));
static_assert(ENUM_MEMBERS_EQUAL(RDD::LOGIC_OP_NO_OP, VK_LOGIC_OP_NO_OP));
static_assert(ENUM_MEMBERS_EQUAL(RDD::LOGIC_OP_XOR, VK_LOGIC_OP_XOR));
static_assert(ENUM_MEMBERS_EQUAL(RDD::LOGIC_OP_OR, VK_LOGIC_OP_OR));
static_assert(ENUM_MEMBERS_EQUAL(RDD::LOGIC_OP_NOR, VK_LOGIC_OP_NOR));
static_assert(ENUM_MEMBERS_EQUAL(RDD::LOGIC_OP_EQUIVALENT, VK_LOGIC_OP_EQUIVALENT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::LOGIC_OP_INVERT, VK_LOGIC_OP_INVERT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::LOGIC_OP_OR_REVERSE, VK_LOGIC_OP_OR_REVERSE));
static_assert(ENUM_MEMBERS_EQUAL(RDD::LOGIC_OP_COPY_INVERTED, VK_LOGIC_OP_COPY_INVERTED));
static_assert(ENUM_MEMBERS_EQUAL(RDD::LOGIC_OP_OR_INVERTED, VK_LOGIC_OP_OR_INVERTED));
static_assert(ENUM_MEMBERS_EQUAL(RDD::LOGIC_OP_NAND, VK_LOGIC_OP_NAND));
static_assert(ENUM_MEMBERS_EQUAL(RDD::LOGIC_OP_SET, VK_LOGIC_OP_SET));

// RDD::BlendFactor == VkBlendFactor.
static_assert(ENUM_MEMBERS_EQUAL(RDD::BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ZERO));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BLEND_FACTOR_SRC_COLOR, VK_BLEND_FACTOR_SRC_COLOR));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BLEND_FACTOR_ONE_MINUS_SRC_COLOR, VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BLEND_FACTOR_DST_COLOR, VK_BLEND_FACTOR_DST_COLOR));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BLEND_FACTOR_ONE_MINUS_DST_COLOR, VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_SRC_ALPHA));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BLEND_FACTOR_DST_ALPHA, VK_BLEND_FACTOR_DST_ALPHA));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BLEND_FACTOR_ONE_MINUS_DST_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BLEND_FACTOR_CONSTANT_COLOR, VK_BLEND_FACTOR_CONSTANT_COLOR));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR, VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BLEND_FACTOR_CONSTANT_ALPHA, VK_BLEND_FACTOR_CONSTANT_ALPHA));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BLEND_FACTOR_SRC_ALPHA_SATURATE, VK_BLEND_FACTOR_SRC_ALPHA_SATURATE));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BLEND_FACTOR_SRC1_COLOR, VK_BLEND_FACTOR_SRC1_COLOR));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BLEND_FACTOR_ONE_MINUS_SRC1_COLOR, VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BLEND_FACTOR_SRC1_ALPHA, VK_BLEND_FACTOR_SRC1_ALPHA));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA));

// RDD::BlendOperation == VkBlendOp.
static_assert(ENUM_MEMBERS_EQUAL(RDD::BLEND_OP_ADD, VK_BLEND_OP_ADD));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BLEND_OP_SUBTRACT, VK_BLEND_OP_SUBTRACT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BLEND_OP_REVERSE_SUBTRACT, VK_BLEND_OP_REVERSE_SUBTRACT));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BLEND_OP_MINIMUM, VK_BLEND_OP_MIN));
static_assert(ENUM_MEMBERS_EQUAL(RDD::BLEND_OP_MAXIMUM, VK_BLEND_OP_MAX));

RDD::PipelineID RenderingDeviceDriverVulkan::render_pipeline_create(
		ShaderID p_shader,
		VertexFormatID p_vertex_format,
		RenderPrimitive p_render_primitive,
		PipelineRasterizationState p_rasterization_state,
		PipelineMultisampleState p_multisample_state,
		PipelineDepthStencilState p_depth_stencil_state,
		PipelineColorBlendState p_blend_state,
		VectorView<int32_t> p_color_attachments,
		BitField<PipelineDynamicStateFlags> p_dynamic_state,
		RenderPassID p_render_pass,
		uint32_t p_render_subpass,
		VectorView<PipelineSpecializationConstant> p_specialization_constants) {
	// Vertex.
	const VkPipelineVertexInputStateCreateInfo *vertex_input_state_create_info = nullptr;
	if (p_vertex_format.id) {
		const VertexFormatInfo *vf_info = (const VertexFormatInfo *)p_vertex_format.id;
		vertex_input_state_create_info = &vf_info->vk_create_info;
	} else {
		VkPipelineVertexInputStateCreateInfo *null_vertex_input_state = ALLOCA_SINGLE(VkPipelineVertexInputStateCreateInfo);
		*null_vertex_input_state = {};
		null_vertex_input_state->sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertex_input_state_create_info = null_vertex_input_state;
	}

	// Input assembly.
	VkPipelineInputAssemblyStateCreateInfo input_assembly_create_info = {};
	input_assembly_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	input_assembly_create_info.topology = RD_TO_VK_PRIMITIVE[p_render_primitive];
	input_assembly_create_info.primitiveRestartEnable = (p_render_primitive == RENDER_PRIMITIVE_TRIANGLE_STRIPS_WITH_RESTART_INDEX);

	// Tessellation.
	VkPipelineTessellationStateCreateInfo tessellation_create_info = {};
	tessellation_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO;
	ERR_FAIL_COND_V(limits.maxTessellationPatchSize > 0 && (p_rasterization_state.patch_control_points < 1 || p_rasterization_state.patch_control_points > limits.maxTessellationPatchSize), PipelineID());
	tessellation_create_info.patchControlPoints = p_rasterization_state.patch_control_points;

	// Viewport.
	VkPipelineViewportStateCreateInfo viewport_state_create_info = {};
	viewport_state_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewport_state_create_info.viewportCount = 1; // If VR extensions are supported at some point, this will have to be customizable in the framebuffer format.
	viewport_state_create_info.scissorCount = 1;

	// Rasterization.
	VkPipelineRasterizationStateCreateInfo rasterization_state_create_info = {};
	rasterization_state_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterization_state_create_info.depthClampEnable = p_rasterization_state.enable_depth_clamp;
	rasterization_state_create_info.rasterizerDiscardEnable = p_rasterization_state.discard_primitives;
	rasterization_state_create_info.polygonMode = p_rasterization_state.wireframe ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
	rasterization_state_create_info.cullMode = (PolygonCullMode)p_rasterization_state.cull_mode;
	rasterization_state_create_info.frontFace = (p_rasterization_state.front_face == POLYGON_FRONT_FACE_CLOCKWISE ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE);
	rasterization_state_create_info.depthBiasEnable = p_rasterization_state.depth_bias_enabled;
	rasterization_state_create_info.depthBiasConstantFactor = p_rasterization_state.depth_bias_constant_factor;
	rasterization_state_create_info.depthBiasClamp = p_rasterization_state.depth_bias_clamp;
	rasterization_state_create_info.depthBiasSlopeFactor = p_rasterization_state.depth_bias_slope_factor;
	rasterization_state_create_info.lineWidth = p_rasterization_state.line_width;

	// Multisample.
	VkPipelineMultisampleStateCreateInfo multisample_state_create_info = {};
	multisample_state_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisample_state_create_info.rasterizationSamples = _ensure_supported_sample_count(p_multisample_state.sample_count);
	multisample_state_create_info.sampleShadingEnable = p_multisample_state.enable_sample_shading;
	multisample_state_create_info.minSampleShading = p_multisample_state.min_sample_shading;
	if (p_multisample_state.sample_mask.size()) {
		static_assert(ARRAYS_COMPATIBLE(uint32_t, VkSampleMask));
		multisample_state_create_info.pSampleMask = p_multisample_state.sample_mask.ptr();
	} else {
		multisample_state_create_info.pSampleMask = nullptr;
	}
	multisample_state_create_info.alphaToCoverageEnable = p_multisample_state.enable_alpha_to_coverage;
	multisample_state_create_info.alphaToOneEnable = p_multisample_state.enable_alpha_to_one;

	// Depth stencil.

	VkPipelineDepthStencilStateCreateInfo depth_stencil_state_create_info = {};
	depth_stencil_state_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depth_stencil_state_create_info.depthTestEnable = p_depth_stencil_state.enable_depth_test;
	depth_stencil_state_create_info.depthWriteEnable = p_depth_stencil_state.enable_depth_write;
	depth_stencil_state_create_info.depthCompareOp = (VkCompareOp)p_depth_stencil_state.depth_compare_operator;
	depth_stencil_state_create_info.depthBoundsTestEnable = p_depth_stencil_state.enable_depth_range;
	depth_stencil_state_create_info.stencilTestEnable = p_depth_stencil_state.enable_stencil;

	depth_stencil_state_create_info.front.failOp = (VkStencilOp)p_depth_stencil_state.front_op.fail;
	depth_stencil_state_create_info.front.passOp = (VkStencilOp)p_depth_stencil_state.front_op.pass;
	depth_stencil_state_create_info.front.depthFailOp = (VkStencilOp)p_depth_stencil_state.front_op.depth_fail;
	depth_stencil_state_create_info.front.compareOp = (VkCompareOp)p_depth_stencil_state.front_op.compare;
	depth_stencil_state_create_info.front.compareMask = p_depth_stencil_state.front_op.compare_mask;
	depth_stencil_state_create_info.front.writeMask = p_depth_stencil_state.front_op.write_mask;
	depth_stencil_state_create_info.front.reference = p_depth_stencil_state.front_op.reference;

	depth_stencil_state_create_info.back.failOp = (VkStencilOp)p_depth_stencil_state.back_op.fail;
	depth_stencil_state_create_info.back.passOp = (VkStencilOp)p_depth_stencil_state.back_op.pass;
	depth_stencil_state_create_info.back.depthFailOp = (VkStencilOp)p_depth_stencil_state.back_op.depth_fail;
	depth_stencil_state_create_info.back.compareOp = (VkCompareOp)p_depth_stencil_state.back_op.compare;
	depth_stencil_state_create_info.back.compareMask = p_depth_stencil_state.back_op.compare_mask;
	depth_stencil_state_create_info.back.writeMask = p_depth_stencil_state.back_op.write_mask;
	depth_stencil_state_create_info.back.reference = p_depth_stencil_state.back_op.reference;

	depth_stencil_state_create_info.minDepthBounds = p_depth_stencil_state.depth_range_min;
	depth_stencil_state_create_info.maxDepthBounds = p_depth_stencil_state.depth_range_max;

	// Blend state.

	VkPipelineColorBlendStateCreateInfo color_blend_state_create_info = {};
	color_blend_state_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	color_blend_state_create_info.logicOpEnable = p_blend_state.enable_logic_op;
	color_blend_state_create_info.logicOp = (VkLogicOp)p_blend_state.logic_op;

	VkPipelineColorBlendAttachmentState *vk_attachment_states = ALLOCA_ARRAY(VkPipelineColorBlendAttachmentState, p_color_attachments.size());
	{
		for (uint32_t i = 0; i < p_color_attachments.size(); i++) {
			vk_attachment_states[i] = {};
			if (p_color_attachments[i] != ATTACHMENT_UNUSED) {
				vk_attachment_states[i].blendEnable = p_blend_state.attachments[i].enable_blend;

				vk_attachment_states[i].srcColorBlendFactor = (VkBlendFactor)p_blend_state.attachments[i].src_color_blend_factor;
				vk_attachment_states[i].dstColorBlendFactor = (VkBlendFactor)p_blend_state.attachments[i].dst_color_blend_factor;
				vk_attachment_states[i].colorBlendOp = (VkBlendOp)p_blend_state.attachments[i].color_blend_op;

				vk_attachment_states[i].srcAlphaBlendFactor = (VkBlendFactor)p_blend_state.attachments[i].src_alpha_blend_factor;
				vk_attachment_states[i].dstAlphaBlendFactor = (VkBlendFactor)p_blend_state.attachments[i].dst_alpha_blend_factor;
				vk_attachment_states[i].alphaBlendOp = (VkBlendOp)p_blend_state.attachments[i].alpha_blend_op;

				if (p_blend_state.attachments[i].write_r) {
					vk_attachment_states[i].colorWriteMask |= VK_COLOR_COMPONENT_R_BIT;
				}
				if (p_blend_state.attachments[i].write_g) {
					vk_attachment_states[i].colorWriteMask |= VK_COLOR_COMPONENT_G_BIT;
				}
				if (p_blend_state.attachments[i].write_b) {
					vk_attachment_states[i].colorWriteMask |= VK_COLOR_COMPONENT_B_BIT;
				}
				if (p_blend_state.attachments[i].write_a) {
					vk_attachment_states[i].colorWriteMask |= VK_COLOR_COMPONENT_A_BIT;
				}
			}
		}
	}
	color_blend_state_create_info.attachmentCount = p_color_attachments.size();
	color_blend_state_create_info.pAttachments = vk_attachment_states;

	color_blend_state_create_info.blendConstants[0] = p_blend_state.blend_constant.r;
	color_blend_state_create_info.blendConstants[1] = p_blend_state.blend_constant.g;
	color_blend_state_create_info.blendConstants[2] = p_blend_state.blend_constant.b;
	color_blend_state_create_info.blendConstants[3] = p_blend_state.blend_constant.a;

	// Dynamic state.

	VkPipelineDynamicStateCreateInfo dynamic_state_create_info = {};
	dynamic_state_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;

	static const uint32_t MAX_DYN_STATE_COUNT = 9;
	VkDynamicState *vk_dynamic_states = ALLOCA_ARRAY(VkDynamicState, MAX_DYN_STATE_COUNT);
	uint32_t vk_dynamic_states_count = 0;

	vk_dynamic_states[vk_dynamic_states_count] = VK_DYNAMIC_STATE_VIEWPORT; // Viewport and scissor are always dynamic.
	vk_dynamic_states_count++;
	vk_dynamic_states[vk_dynamic_states_count] = VK_DYNAMIC_STATE_SCISSOR;
	vk_dynamic_states_count++;
	if (p_dynamic_state.has_flag(DYNAMIC_STATE_LINE_WIDTH)) {
		vk_dynamic_states[vk_dynamic_states_count] = VK_DYNAMIC_STATE_LINE_WIDTH;
		vk_dynamic_states_count++;
	}
	if (p_dynamic_state.has_flag(DYNAMIC_STATE_DEPTH_BIAS)) {
		vk_dynamic_states[vk_dynamic_states_count] = VK_DYNAMIC_STATE_DEPTH_BIAS;
		vk_dynamic_states_count++;
	}
	if (p_dynamic_state.has_flag(DYNAMIC_STATE_BLEND_CONSTANTS)) {
		vk_dynamic_states[vk_dynamic_states_count] = VK_DYNAMIC_STATE_BLEND_CONSTANTS;
		vk_dynamic_states_count++;
	}
	if (p_dynamic_state.has_flag(DYNAMIC_STATE_DEPTH_BOUNDS)) {
		vk_dynamic_states[vk_dynamic_states_count] = VK_DYNAMIC_STATE_DEPTH_BOUNDS;
		vk_dynamic_states_count++;
	}
	if (p_dynamic_state.has_flag(DYNAMIC_STATE_STENCIL_COMPARE_MASK)) {
		vk_dynamic_states[vk_dynamic_states_count] = VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK;
		vk_dynamic_states_count++;
	}
	if (p_dynamic_state.has_flag(DYNAMIC_STATE_STENCIL_WRITE_MASK)) {
		vk_dynamic_states[vk_dynamic_states_count] = VK_DYNAMIC_STATE_STENCIL_WRITE_MASK;
		vk_dynamic_states_count++;
	}
	if (p_dynamic_state.has_flag(DYNAMIC_STATE_STENCIL_REFERENCE)) {
		vk_dynamic_states[vk_dynamic_states_count] = VK_DYNAMIC_STATE_STENCIL_REFERENCE;
		vk_dynamic_states_count++;
	}
	DEV_ASSERT(vk_dynamic_states_count <= MAX_DYN_STATE_COUNT);

	dynamic_state_create_info.dynamicStateCount = vk_dynamic_states_count;
	dynamic_state_create_info.pDynamicStates = vk_dynamic_states;

	// VRS.

	void *graphics_pipeline_nextptr = nullptr;

	if (context->get_vrs_capabilities().attachment_vrs_supported) {
		// If VRS is used, this defines how the different VRS types are combined.
		// combinerOps[0] decides how we use the output of pipeline and primitive (drawcall) VRS.
		// combinerOps[1] decides how we use the output of combinerOps[0] and our attachment VRS.

		VkPipelineFragmentShadingRateStateCreateInfoKHR *vrs_create_info = ALLOCA_SINGLE(VkPipelineFragmentShadingRateStateCreateInfoKHR);
		*vrs_create_info = {};
		vrs_create_info->sType = VK_STRUCTURE_TYPE_PIPELINE_FRAGMENT_SHADING_RATE_STATE_CREATE_INFO_KHR;
		vrs_create_info->fragmentSize = { 4, 4 };
		vrs_create_info->combinerOps[0] = VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR; // We don't use pipeline/primitive VRS so this really doesn't matter.
		vrs_create_info->combinerOps[1] = VK_FRAGMENT_SHADING_RATE_COMBINER_OP_REPLACE_KHR; // Always use the outcome of attachment VRS if enabled.

		graphics_pipeline_nextptr = vrs_create_info;
	}

	// Finally, pipeline create info.

	const ShaderInfo *shader_info = (const ShaderInfo *)p_shader.id;

	VkGraphicsPipelineCreateInfo pipeline_create_info = {};

	pipeline_create_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipeline_create_info.pNext = graphics_pipeline_nextptr;
	pipeline_create_info.stageCount = shader_info->vk_stages_create_info.size();

	VkPipelineShaderStageCreateInfo *vk_pipeline_stages = ALLOCA_ARRAY(VkPipelineShaderStageCreateInfo, shader_info->vk_stages_create_info.size());

	for (uint32_t i = 0; i < shader_info->vk_stages_create_info.size(); i++) {
		vk_pipeline_stages[i] = shader_info->vk_stages_create_info[i];

		if (p_specialization_constants.size()) {
			VkSpecializationMapEntry *specialization_map_entries = ALLOCA_ARRAY(VkSpecializationMapEntry, p_specialization_constants.size());
			for (uint32_t j = 0; j < p_specialization_constants.size(); j++) {
				specialization_map_entries[j] = {};
				specialization_map_entries[j].constantID = p_specialization_constants[j].constant_id;
				specialization_map_entries[j].offset = (const char *)&p_specialization_constants[j].int_value - (const char *)p_specialization_constants.ptr();
				specialization_map_entries[j].size = sizeof(uint32_t);
			}

			VkSpecializationInfo *specialization_info = ALLOCA_SINGLE(VkSpecializationInfo);
			*specialization_info = {};
			specialization_info->dataSize = p_specialization_constants.size() * sizeof(PipelineSpecializationConstant);
			specialization_info->pData = p_specialization_constants.ptr();
			specialization_info->mapEntryCount = p_specialization_constants.size();
			specialization_info->pMapEntries = specialization_map_entries;

			vk_pipeline_stages[i].pSpecializationInfo = specialization_info;
		}
	}

	pipeline_create_info.pStages = vk_pipeline_stages;
	pipeline_create_info.pVertexInputState = vertex_input_state_create_info;
	pipeline_create_info.pInputAssemblyState = &input_assembly_create_info;
	pipeline_create_info.pTessellationState = &tessellation_create_info;
	pipeline_create_info.pViewportState = &viewport_state_create_info;
	pipeline_create_info.pRasterizationState = &rasterization_state_create_info;
	pipeline_create_info.pMultisampleState = &multisample_state_create_info;
	pipeline_create_info.pDepthStencilState = &depth_stencil_state_create_info;
	pipeline_create_info.pColorBlendState = &color_blend_state_create_info;
	pipeline_create_info.pDynamicState = &dynamic_state_create_info;
	pipeline_create_info.layout = shader_info->vk_pipeline_layout;
	pipeline_create_info.renderPass = (VkRenderPass)p_render_pass.id;
	pipeline_create_info.subpass = p_render_subpass;

	// ---

	VkPipeline vk_pipeline = VK_NULL_HANDLE;
	VkResult err = vkCreateGraphicsPipelines(vk_device, pipelines_cache.vk_cache, 1, &pipeline_create_info, nullptr, &vk_pipeline);
	ERR_FAIL_COND_V_MSG(err, PipelineID(), "vkCreateGraphicsPipelines failed with error " + itos(err) + ".");

	return PipelineID(vk_pipeline);
}

/*****************/
/**** COMPUTE ****/
/*****************/

// ----- COMMANDS -----

void RenderingDeviceDriverVulkan::command_bind_compute_pipeline(CommandBufferID p_cmd_buffer, PipelineID p_pipeline) {
	vkCmdBindPipeline((VkCommandBuffer)p_cmd_buffer.id, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)p_pipeline.id);
}

void RenderingDeviceDriverVulkan::command_bind_compute_uniform_set(CommandBufferID p_cmd_buffer, UniformSetID p_uniform_set, ShaderID p_shader, uint32_t p_set_index) {
	const ShaderInfo *shader_info = (const ShaderInfo *)p_shader.id;
	const UniformSetInfo *usi = (const UniformSetInfo *)p_uniform_set.id;
	vkCmdBindDescriptorSets((VkCommandBuffer)p_cmd_buffer.id, VK_PIPELINE_BIND_POINT_COMPUTE, shader_info->vk_pipeline_layout, p_set_index, 1, &usi->vk_descriptor_set, 0, nullptr);
}

void RenderingDeviceDriverVulkan::command_compute_dispatch(CommandBufferID p_cmd_buffer, uint32_t p_x_groups, uint32_t p_y_groups, uint32_t p_z_groups) {
	vkCmdDispatch((VkCommandBuffer)p_cmd_buffer.id, p_x_groups, p_y_groups, p_z_groups);
}

void RenderingDeviceDriverVulkan::command_compute_dispatch_indirect(CommandBufferID p_cmd_buffer, BufferID p_indirect_buffer, uint64_t p_offset) {
	const BufferInfo *buf_info = (const BufferInfo *)p_indirect_buffer.id;
	vkCmdDispatchIndirect((VkCommandBuffer)p_cmd_buffer.id, buf_info->vk_buffer, p_offset);
}

// ----- PIPELINE -----

RDD::PipelineID RenderingDeviceDriverVulkan::compute_pipeline_create(ShaderID p_shader, VectorView<PipelineSpecializationConstant> p_specialization_constants) {
	const ShaderInfo *shader_info = (const ShaderInfo *)p_shader.id;

	VkComputePipelineCreateInfo pipeline_create_info = {};
	pipeline_create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	pipeline_create_info.stage = shader_info->vk_stages_create_info[0];
	pipeline_create_info.layout = shader_info->vk_pipeline_layout;

	if (p_specialization_constants.size()) {
		VkSpecializationMapEntry *specialization_map_entries = ALLOCA_ARRAY(VkSpecializationMapEntry, p_specialization_constants.size());
		for (uint32_t i = 0; i < p_specialization_constants.size(); i++) {
			specialization_map_entries[i] = {};
			specialization_map_entries[i].constantID = p_specialization_constants[i].constant_id;
			specialization_map_entries[i].offset = (const char *)&p_specialization_constants[i].int_value - (const char *)p_specialization_constants.ptr();
			specialization_map_entries[i].size = sizeof(uint32_t);
		}

		VkSpecializationInfo *specialization_info = ALLOCA_SINGLE(VkSpecializationInfo);
		*specialization_info = {};
		specialization_info->dataSize = p_specialization_constants.size() * sizeof(PipelineSpecializationConstant);
		specialization_info->pData = p_specialization_constants.ptr();
		specialization_info->mapEntryCount = p_specialization_constants.size();
		specialization_info->pMapEntries = specialization_map_entries;

		pipeline_create_info.stage.pSpecializationInfo = specialization_info;
	}

	VkPipeline vk_pipeline = VK_NULL_HANDLE;
	VkResult err = vkCreateComputePipelines(vk_device, pipelines_cache.vk_cache, 1, &pipeline_create_info, nullptr, &vk_pipeline);
	ERR_FAIL_COND_V_MSG(err, PipelineID(), "vkCreateComputePipelines failed with error " + itos(err) + ".");

	return PipelineID(vk_pipeline);
}

/*****************/
/**** QUERIES ****/
/*****************/

// ----- TIMESTAMP -----

RDD::QueryPoolID RenderingDeviceDriverVulkan::timestamp_query_pool_create(uint32_t p_query_count) {
	VkQueryPoolCreateInfo query_pool_create_info = {};
	query_pool_create_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
	query_pool_create_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
	query_pool_create_info.queryCount = p_query_count;

	VkQueryPool vk_query_pool = VK_NULL_HANDLE;
	vkCreateQueryPool(vk_device, &query_pool_create_info, nullptr, &vk_query_pool);
	return RDD::QueryPoolID(vk_query_pool);
}

void RenderingDeviceDriverVulkan::timestamp_query_pool_free(QueryPoolID p_pool_id) {
	vkDestroyQueryPool(vk_device, (VkQueryPool)p_pool_id.id, nullptr);
}

void RenderingDeviceDriverVulkan::timestamp_query_pool_get_results(QueryPoolID p_pool_id, uint32_t p_query_count, uint64_t *r_results) {
	vkGetQueryPoolResults(vk_device, (VkQueryPool)p_pool_id.id, 0, p_query_count, sizeof(uint64_t) * p_query_count, r_results, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
}

uint64_t RenderingDeviceDriverVulkan::timestamp_query_result_to_time(uint64_t p_result) {
	// This sucks because timestampPeriod multiplier is a float, while the timestamp is 64 bits nanosecs.
	// So, in cases like nvidia which give you enormous numbers and 1 as multiplier, multiplying is next to impossible.
	// Need to do 128 bits fixed point multiplication to get the right value.

	auto mult64to128 = [](uint64_t u, uint64_t v, uint64_t &h, uint64_t &l) {
		uint64_t u1 = (u & 0xffffffff);
		uint64_t v1 = (v & 0xffffffff);
		uint64_t t = (u1 * v1);
		uint64_t w3 = (t & 0xffffffff);
		uint64_t k = (t >> 32);

		u >>= 32;
		t = (u * v1) + k;
		k = (t & 0xffffffff);
		uint64_t w1 = (t >> 32);

		v >>= 32;
		t = (u1 * v) + k;
		k = (t >> 32);

		h = (u * v) + w1 + k;
		l = (t << 32) + w3;
	};

	uint64_t shift_bits = 16;
	uint64_t h = 0, l = 0;
	mult64to128(p_result, uint64_t(double(context->get_device_limits().timestampPeriod) * double(1 << shift_bits)), h, l);
	l >>= shift_bits;
	l |= h << (64 - shift_bits);

	return l;
}

void RenderingDeviceDriverVulkan::command_timestamp_query_pool_reset(CommandBufferID p_cmd_buffer, QueryPoolID p_pool_id, uint32_t p_query_count) {
	vkCmdResetQueryPool((VkCommandBuffer)p_cmd_buffer.id, (VkQueryPool)p_pool_id.id, 0, p_query_count);
}

void RenderingDeviceDriverVulkan::command_timestamp_write(CommandBufferID p_cmd_buffer, QueryPoolID p_pool_id, uint32_t p_index) {
	vkCmdWriteTimestamp((VkCommandBuffer)p_cmd_buffer.id, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, (VkQueryPool)p_pool_id.id, p_index);
}

/****************/
/**** LABELS ****/
/****************/

void RenderingDeviceDriverVulkan::command_begin_label(CommandBufferID p_cmd_buffer, const char *p_label_name, const Color &p_color) {
	VkDebugUtilsLabelEXT label;
	label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
	label.pNext = nullptr;
	label.pLabelName = p_label_name;
	label.color[0] = p_color[0];
	label.color[1] = p_color[1];
	label.color[2] = p_color[2];
	label.color[3] = p_color[3];
	vkCmdBeginDebugUtilsLabelEXT((VkCommandBuffer)p_cmd_buffer.id, &label);
}

void RenderingDeviceDriverVulkan::command_end_label(CommandBufferID p_cmd_buffer) {
	vkCmdEndDebugUtilsLabelEXT((VkCommandBuffer)p_cmd_buffer.id);
}

/****************/
/**** SCREEN ****/
/****************/

RDD::DataFormat RenderingDeviceDriverVulkan::screen_get_format() {
	// Very hacky, but not used often per frame so I guess ok.
	VkFormat vk_format = context->get_screen_format();
	DataFormat format = DATA_FORMAT_MAX;
	for (int i = 0; i < DATA_FORMAT_MAX; i++) {
		if (vk_format == RD_TO_VK_FORMAT[i]) {
			format = DataFormat(i);
			break;
		}
	}
	return format;
}

/********************/
/**** SUBMISSION ****/
/********************/

void RenderingDeviceDriverVulkan::begin_segment(CommandBufferID p_cmd_buffer, uint32_t p_frame_index, uint32_t p_frames_drawn) {
}

void RenderingDeviceDriverVulkan::end_segment() {
}

/**************/
/**** MISC ****/
/**************/

void RenderingDeviceDriverVulkan::set_object_name(ObjectType p_type, ID p_driver_id, const String &p_name) {
	switch (p_type) {
		case OBJECT_TYPE_TEXTURE: {
			const TextureInfo *tex_info = (const TextureInfo *)p_driver_id.id;
			if (tex_info->allocation.handle) {
				context->set_object_name(VK_OBJECT_TYPE_IMAGE, (uint64_t)tex_info->vk_view_create_info.image, p_name);
			}
			context->set_object_name(VK_OBJECT_TYPE_IMAGE_VIEW, (uint64_t)tex_info->vk_view, p_name + " View");
		} break;
		case OBJECT_TYPE_SAMPLER: {
			context->set_object_name(VK_OBJECT_TYPE_SAMPLER, p_driver_id.id, p_name);
		} break;
		case OBJECT_TYPE_BUFFER: {
			const BufferInfo *buf_info = (const BufferInfo *)p_driver_id.id;
			context->set_object_name(VK_OBJECT_TYPE_BUFFER, (uint64_t)buf_info->vk_buffer, p_name);
			if (buf_info->vk_view) {
				context->set_object_name(VK_OBJECT_TYPE_BUFFER_VIEW, (uint64_t)buf_info->vk_view, p_name + " View");
			}
		} break;
		case OBJECT_TYPE_SHADER: {
			const ShaderInfo *shader_info = (const ShaderInfo *)p_driver_id.id;
			for (uint32_t i = 0; i < shader_info->vk_descriptor_set_layouts.size(); i++) {
				context->set_object_name(VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, (uint64_t)shader_info->vk_descriptor_set_layouts[i], p_name);
			}
			context->set_object_name(VK_OBJECT_TYPE_PIPELINE_LAYOUT, (uint64_t)shader_info->vk_pipeline_layout, p_name + " Pipeline Layout");
		} break;
		case OBJECT_TYPE_UNIFORM_SET: {
			const UniformSetInfo *usi = (const UniformSetInfo *)p_driver_id.id;
			context->set_object_name(VK_OBJECT_TYPE_DESCRIPTOR_SET, (uint64_t)usi->vk_descriptor_set, p_name);
		} break;
		case OBJECT_TYPE_PIPELINE: {
			context->set_object_name(VK_OBJECT_TYPE_PIPELINE, (uint64_t)p_driver_id.id, p_name);
		} break;
		default: {
			DEV_ASSERT(false);
		}
	}
}

uint64_t RenderingDeviceDriverVulkan::get_resource_native_handle(DriverResource p_type, ID p_driver_id) {
	switch (p_type) {
		case DRIVER_RESOURCE_LOGICAL_DEVICE: {
			return (uint64_t)vk_device;
		}
		case DRIVER_RESOURCE_PHYSICAL_DEVICE: {
			return (uint64_t)context->get_physical_device();
		}
		case DRIVER_RESOURCE_TOPMOST_OBJECT: {
			return (uint64_t)context->get_instance();
		}
		case DRIVER_RESOURCE_COMMAND_QUEUE: {
			return (uint64_t)context->get_graphics_queue();
		}
		case DRIVER_RESOURCE_QUEUE_FAMILY: {
			return context->get_graphics_queue_family_index();
		}
		case DRIVER_RESOURCE_TEXTURE: {
			const TextureInfo *tex_info = (const TextureInfo *)p_driver_id.id;
			return (uint64_t)tex_info->vk_view_create_info.image;
		}
		case DRIVER_RESOURCE_TEXTURE_VIEW: {
			const TextureInfo *tex_info = (const TextureInfo *)p_driver_id.id;
			return (uint64_t)tex_info->vk_view;
		}
		case DRIVER_RESOURCE_TEXTURE_DATA_FORMAT: {
			const TextureInfo *tex_info = (const TextureInfo *)p_driver_id.id;
			return (uint64_t)tex_info->vk_view_create_info.format;
		}
		case DRIVER_RESOURCE_SAMPLER:
		case DRIVER_RESOURCE_UNIFORM_SET:
		case DRIVER_RESOURCE_BUFFER:
		case DRIVER_RESOURCE_COMPUTE_PIPELINE:
		case DRIVER_RESOURCE_RENDER_PIPELINE: {
			return p_driver_id.id;
		}
		default: {
			return 0;
		}
	}
}

uint64_t RenderingDeviceDriverVulkan::get_total_memory_used() {
	VmaTotalStatistics stats = {};
	vmaCalculateStatistics(allocator, &stats);
	return stats.total.statistics.allocationBytes;
}

uint64_t RenderingDeviceDriverVulkan::limit_get(Limit p_limit) {
	switch (p_limit) {
		case LIMIT_MAX_BOUND_UNIFORM_SETS:
			return limits.maxBoundDescriptorSets;
		case LIMIT_MAX_FRAMEBUFFER_COLOR_ATTACHMENTS:
			return limits.maxColorAttachments;
		case LIMIT_MAX_TEXTURES_PER_UNIFORM_SET:
			return limits.maxDescriptorSetSampledImages;
		case LIMIT_MAX_SAMPLERS_PER_UNIFORM_SET:
			return limits.maxDescriptorSetSamplers;
		case LIMIT_MAX_STORAGE_BUFFERS_PER_UNIFORM_SET:
			return limits.maxDescriptorSetStorageBuffers;
		case LIMIT_MAX_STORAGE_IMAGES_PER_UNIFORM_SET:
			return limits.maxDescriptorSetStorageImages;
		case LIMIT_MAX_UNIFORM_BUFFERS_PER_UNIFORM_SET:
			return limits.maxDescriptorSetUniformBuffers;
		case LIMIT_MAX_DRAW_INDEXED_INDEX:
			return limits.maxDrawIndexedIndexValue;
		case LIMIT_MAX_FRAMEBUFFER_HEIGHT:
			return limits.maxFramebufferHeight;
		case LIMIT_MAX_FRAMEBUFFER_WIDTH:
			return limits.maxFramebufferWidth;
		case LIMIT_MAX_TEXTURE_ARRAY_LAYERS:
			return limits.maxImageArrayLayers;
		case LIMIT_MAX_TEXTURE_SIZE_1D:
			return limits.maxImageDimension1D;
		case LIMIT_MAX_TEXTURE_SIZE_2D:
			return limits.maxImageDimension2D;
		case LIMIT_MAX_TEXTURE_SIZE_3D:
			return limits.maxImageDimension3D;
		case LIMIT_MAX_TEXTURE_SIZE_CUBE:
			return limits.maxImageDimensionCube;
		case LIMIT_MAX_TEXTURES_PER_SHADER_STAGE:
			return limits.maxPerStageDescriptorSampledImages;
		case LIMIT_MAX_SAMPLERS_PER_SHADER_STAGE:
			return limits.maxPerStageDescriptorSamplers;
		case LIMIT_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE:
			return limits.maxPerStageDescriptorStorageBuffers;
		case LIMIT_MAX_STORAGE_IMAGES_PER_SHADER_STAGE:
			return limits.maxPerStageDescriptorStorageImages;
		case LIMIT_MAX_UNIFORM_BUFFERS_PER_SHADER_STAGE:
			return limits.maxPerStageDescriptorUniformBuffers;
		case LIMIT_MAX_PUSH_CONSTANT_SIZE:
			return limits.maxPushConstantsSize;
		case LIMIT_MAX_UNIFORM_BUFFER_SIZE:
			return limits.maxUniformBufferRange;
		case LIMIT_MAX_VERTEX_INPUT_ATTRIBUTE_OFFSET:
			return limits.maxVertexInputAttributeOffset;
		case LIMIT_MAX_VERTEX_INPUT_ATTRIBUTES:
			return limits.maxVertexInputAttributes;
		case LIMIT_MAX_VERTEX_INPUT_BINDINGS:
			return limits.maxVertexInputBindings;
		case LIMIT_MAX_VERTEX_INPUT_BINDING_STRIDE:
			return limits.maxVertexInputBindingStride;
		case LIMIT_MIN_UNIFORM_BUFFER_OFFSET_ALIGNMENT:
			return limits.minUniformBufferOffsetAlignment;
		case LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_X:
			return limits.maxComputeWorkGroupCount[0];
		case LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_Y:
			return limits.maxComputeWorkGroupCount[1];
		case LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_Z:
			return limits.maxComputeWorkGroupCount[2];
		case LIMIT_MAX_COMPUTE_WORKGROUP_INVOCATIONS:
			return limits.maxComputeWorkGroupInvocations;
		case LIMIT_MAX_COMPUTE_WORKGROUP_SIZE_X:
			return limits.maxComputeWorkGroupSize[0];
		case LIMIT_MAX_COMPUTE_WORKGROUP_SIZE_Y:
			return limits.maxComputeWorkGroupSize[1];
		case LIMIT_MAX_COMPUTE_WORKGROUP_SIZE_Z:
			return limits.maxComputeWorkGroupSize[2];
		case LIMIT_MAX_VIEWPORT_DIMENSIONS_X:
			return limits.maxViewportDimensions[0];
		case LIMIT_MAX_VIEWPORT_DIMENSIONS_Y:
			return limits.maxViewportDimensions[1];
		case LIMIT_SUBGROUP_SIZE: {
			VulkanContext::SubgroupCapabilities subgroup_capabilities = context->get_subgroup_capabilities();
			return subgroup_capabilities.size;
		}
		case LIMIT_SUBGROUP_MIN_SIZE: {
			VulkanContext::SubgroupCapabilities subgroup_capabilities = context->get_subgroup_capabilities();
			return subgroup_capabilities.min_size;
		}
		case LIMIT_SUBGROUP_MAX_SIZE: {
			VulkanContext::SubgroupCapabilities subgroup_capabilities = context->get_subgroup_capabilities();
			return subgroup_capabilities.max_size;
		}
		case LIMIT_SUBGROUP_IN_SHADERS: {
			VulkanContext::SubgroupCapabilities subgroup_capabilities = context->get_subgroup_capabilities();
			return subgroup_capabilities.supported_stages_flags_rd();
		}
		case LIMIT_SUBGROUP_OPERATIONS: {
			VulkanContext::SubgroupCapabilities subgroup_capabilities = context->get_subgroup_capabilities();
			return subgroup_capabilities.supported_operations_flags_rd();
		}
		case LIMIT_VRS_TEXEL_WIDTH:
			return context->get_vrs_capabilities().texel_size.x;
		case LIMIT_VRS_TEXEL_HEIGHT:
			return context->get_vrs_capabilities().texel_size.y;
		default:
			ERR_FAIL_V(0);
	}
}

uint64_t RenderingDeviceDriverVulkan::api_trait_get(ApiTrait p_trait) {
	switch (p_trait) {
		case API_TRAIT_TEXTURE_TRANSFER_ALIGNMENT:
			return (uint64_t)MAX((uint64_t)16, limits.optimalBufferCopyOffsetAlignment);
		case API_TRAIT_SHADER_CHANGE_INVALIDATION:
			return (uint64_t)SHADER_CHANGE_INVALIDATION_INCOMPATIBLE_SETS_PLUS_CASCADE;
		default:
			return RenderingDeviceDriver::api_trait_get(p_trait);
	}
}

bool RenderingDeviceDriverVulkan::has_feature(Features p_feature) {
	switch (p_feature) {
		case SUPPORTS_MULTIVIEW: {
			MultiviewCapabilities multiview_capabilies = context->get_multiview_capabilities();
			return multiview_capabilies.is_supported && multiview_capabilies.max_view_count > 1;
		} break;
		case SUPPORTS_FSR_HALF_FLOAT: {
			return context->get_shader_capabilities().shader_float16_is_supported && context->get_physical_device_features().shaderInt16 && context->get_storage_buffer_capabilities().storage_buffer_16_bit_access_is_supported;
		} break;
		case SUPPORTS_ATTACHMENT_VRS: {
			VulkanContext::VRSCapabilities vrs_capabilities = context->get_vrs_capabilities();
			return vrs_capabilities.attachment_vrs_supported && context->get_physical_device_features().shaderStorageImageExtendedFormats;
		} break;
		case SUPPORTS_FRAGMENT_SHADER_WITH_ONLY_SIDE_EFFECTS: {
			return true;
		} break;
		default: {
			return false;
		}
	}
}

const RDD::MultiviewCapabilities &RenderingDeviceDriverVulkan::get_multiview_capabilities() {
	return context->get_multiview_capabilities();
}

/******************/

RenderingDeviceDriverVulkan::RenderingDeviceDriverVulkan(VulkanContext *p_context, VkDevice p_vk_device) :
		context(p_context),
		vk_device(p_vk_device) {
	VmaAllocatorCreateInfo allocator_info = {};
	allocator_info.physicalDevice = context->get_physical_device();
	allocator_info.device = vk_device;
	allocator_info.instance = context->get_instance();
	VkResult err = vmaCreateAllocator(&allocator_info, &allocator);
	ERR_FAIL_COND_MSG(err, "vmaCreateAllocator failed with error " + itos(err) + ".");

	max_descriptor_sets_per_pool = GLOBAL_GET("rendering/rendering_device/vulkan/max_descriptors_per_pool");

	VkPhysicalDeviceProperties props = {};
	vkGetPhysicalDeviceProperties(context->get_physical_device(), &props);
	pipelines_cache.buffer.resize(sizeof(PipelineCacheHeader));
	PipelineCacheHeader *header = (PipelineCacheHeader *)pipelines_cache.buffer.ptrw();
	*header = {};
	header->magic = 868 + VK_PIPELINE_CACHE_HEADER_VERSION_ONE;
	header->device_id = props.deviceID;
	header->vendor_id = props.vendorID;
	header->driver_version = props.driverVersion;
	memcpy(header->uuid, props.pipelineCacheUUID, VK_UUID_SIZE);
	header->driver_abi = sizeof(void *);

	limits = context->get_device_limits();
}

RenderingDeviceDriverVulkan::~RenderingDeviceDriverVulkan() {
	while (small_allocs_pools.size()) {
		HashMap<uint32_t, VmaPool>::Iterator E = small_allocs_pools.begin();
		vmaDestroyPool(allocator, E->value);
		small_allocs_pools.remove(E);
	}
	vmaDestroyAllocator(allocator);
}
