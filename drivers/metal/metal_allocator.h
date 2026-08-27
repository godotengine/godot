/**************************************************************************/
/*  metal_allocator.h                                                     */
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

#pragma once

#include "core/templates/local_vector.h"

#include <Metal/Metal.hpp>

/// Opaque allocation handle. POOL entries reference a pool block plus the
/// OffsetAllocator range (in 256-byte units); DEDICATED entries own a
/// single-resource placement heap. INVALID means there is nothing to return
/// to the allocator (passthrough allocations, committed resources, texture
/// views, external imports).
struct MetalAllocation {
	enum class Kind : uint8_t {
		INVALID,
		POOL,
		DEDICATED,
	};

	Kind kind = Kind::INVALID;

	_FORCE_INLINE_ bool is_valid() const { return kind != Kind::INVALID; }
};

struct MetalBuffer {
	NS::SharedPtr<MTL::Buffer> buffer;
	MetalAllocation allocation;
};

struct MetalTexture {
	NS::SharedPtr<MTL::Texture> texture;
	MetalAllocation allocation;
};

#ifdef DEBUG_ENABLED
struct MetalAllocatorStats {
	/// Live statistics for the associated pool.
	struct Pool {
		uint64_t reserved_bytes = 0;
		uint64_t used_bytes = 0;
		// The number of blocks.
		uint32_t block_count = 0;
		// Total number of allocations.
		uint32_t allocation_count = 0;
		// Total number of dedicated heaps.
		uint32_t dedicated_count = 0;
		// Total size of dedicated heaps.
		uint64_t dedicated_bytes = 0;
	};
	Pool pools[3];
};
#endif

/// Abstract allocation interface for the Metal driver. The driver holds a
/// single instance selected by sync mode.
class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0)) MetalAllocator {
public:
	static MetalAllocator *create(MTL::Device *p_device, bool p_use_heaps);
	virtual ~MetalAllocator() = default;

	virtual MetalBuffer new_buffer(NS::UInteger p_length, MTL::ResourceOptions p_options) = 0;
	virtual MetalTexture new_texture(const MTL::TextureDescriptor *p_desc) = 0;
	virtual void free_buffer(MetalBuffer &p_buffer) = 0;
	virtual void free_texture(MetalTexture &p_texture) = 0;

	/// Appends all live heaps to r_heaps and returns the generation the
	/// snapshot corresponds to. The generation increments whenever a heap is
	/// created or destroyed; callers cache their heap arrays against it.
	virtual uint64_t get_heaps(LocalVector<MTL::Heap *> &r_heaps) = 0;
	virtual uint64_t get_heap_generation() const = 0;

#ifdef DEBUG_ENABLED
	virtual void get_stats(MetalAllocatorStats &r_stats) = 0;
#endif
};

/// Passthrough: forwards to MTL::Device, returns empty allocation handles.
/// Used in hazard tracking mode. Behavior identical to the pre-allocator
/// driver.
class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0)) MetalDeviceAllocator final : public MetalAllocator {
	MTL::Device *device = nullptr;

public:
	explicit MetalDeviceAllocator(MTL::Device *p_device) :
			device(p_device) {}

	MetalBuffer new_buffer(NS::UInteger p_length, MTL::ResourceOptions p_options) override;
	MetalTexture new_texture(const MTL::TextureDescriptor *p_desc) override;
	void free_buffer(MetalBuffer &p_buffer) override;
	void free_texture(MetalTexture &p_texture) override;
	uint64_t get_heaps(LocalVector<MTL::Heap *> &r_heaps) override;
	uint64_t get_heap_generation() const override;
#ifdef DEBUG_ENABLED
	void get_stats(MetalAllocatorStats &r_stats) override;
#endif
};
