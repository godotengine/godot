/**************************************************************************/
/*  metal_allocator.cpp                                                   */
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

#include "drivers/metal/metal_allocator.h"

namespace {

class SpinLockGuard {
	const SpinLock &lock;

public:
	explicit SpinLockGuard(const SpinLock &p_lock) :
			lock(p_lock) {
		lock.lock();
	}
	~SpinLockGuard() {
		lock.unlock();
	}
};

} // namespace

#pragma mark - MetalAllocator

MetalAllocator *MetalAllocator::create(MTL::Device *p_device, bool p_use_heaps) {
	if (p_use_heaps) {
		return memnew(MetalHeapAllocator(p_device));
	}
	return memnew(MetalDeviceAllocator(p_device));
}

#pragma mark - MetalDeviceAllocator

MetalBuffer MetalDeviceAllocator::new_buffer(NS::UInteger p_length, MTL::ResourceOptions p_options) {
	MetalBuffer result;
	result.buffer = NS::TransferPtr(device->newBuffer(p_length, p_options));
	return result;
}

MetalTexture MetalDeviceAllocator::new_texture(const MTL::TextureDescriptor *p_desc) {
	MetalTexture result;
	result.texture = NS::TransferPtr(device->newTexture(p_desc));
	return result;
}

void MetalDeviceAllocator::free_buffer(MetalBuffer &p_buffer) {
	DEV_ASSERT(!p_buffer.allocation.is_valid());
	p_buffer.buffer.reset();
}

void MetalDeviceAllocator::free_texture(MetalTexture &p_texture) {
	DEV_ASSERT(!p_texture.allocation.is_valid());
	p_texture.texture.reset();
}

uint64_t MetalDeviceAllocator::get_heaps(LocalVector<MTL::Heap *> &r_heaps) {
	return 0;
}

uint64_t MetalDeviceAllocator::get_heap_generation() const {
	return 0;
}

#ifdef DEBUG_ENABLED
void MetalDeviceAllocator::get_stats(MetalAllocatorStats &r_stats) {
	r_stats = MetalAllocatorStats();
}
#endif

#pragma mark - MetalHeapAllocator

namespace {
constexpr NS::UInteger RESOURCE_CPU_CACHE_MODE_SHIFT = 0;
constexpr NS::UInteger RESOURCE_CPU_CACHE_MODE_MASK = 0xFull << RESOURCE_CPU_CACHE_MODE_SHIFT;
constexpr NS::UInteger RESOURCE_STORAGE_MODE_SHIFT = 4;
constexpr NS::UInteger RESOURCE_STORAGE_MODE_MASK = 0xFull << RESOURCE_STORAGE_MODE_SHIFT;
} // namespace

uint64_t MetalHeapAllocator::_preferred_block_size(uint32_t p_pool) {
	switch (p_pool) {
		case POOL_PRIVATE:
			return 128ull * 1024 * 1024;
		case POOL_SHARED:
			return 32ull * 1024 * 1024;
		case POOL_SHARED_WC:
		default:
			return 16ull * 1024 * 1024;
	}
}

uint32_t MetalHeapAllocator::_pool_for_options(MTL::ResourceOptions p_options) {
	MTL::StorageMode storage = (MTL::StorageMode)((p_options & RESOURCE_STORAGE_MODE_MASK) >> RESOURCE_STORAGE_MODE_SHIFT);
	MTL::CPUCacheMode cache = (MTL::CPUCacheMode)((p_options & RESOURCE_CPU_CACHE_MODE_MASK) >> RESOURCE_CPU_CACHE_MODE_SHIFT);
	if (storage == MTL::StorageModeShared) {
		return cache == MTL::CPUCacheModeWriteCombined ? POOL_SHARED_WC : POOL_SHARED;
	}
	// Private and Memoryless both land in the private pool.
	return POOL_PRIVATE;
}

uint32_t MetalHeapAllocator::_pool_for_texture(const MTL::TextureDescriptor *p_desc) {
	// Memoryless textures may be placed into any heap:
	// newTextureWithDescriptor:offset: permits StorageModeMemoryless
	// regardless of the heap's storage mode.
	return p_desc->storageMode() == MTL::StorageModeShared ? POOL_SHARED : POOL_PRIVATE;
}

NS::SharedPtr<MTL::Heap> MetalHeapAllocator::_create_heap(uint32_t p_pool, uint64_t p_size) {
	NS::SharedPtr<MTL::HeapDescriptor> desc = NS::TransferPtr(MTL::HeapDescriptor::alloc()->init());
	desc->setType(MTL::HeapTypePlacement);
	desc->setStorageMode(p_pool == POOL_PRIVATE ? MTL::StorageModePrivate : MTL::StorageModeShared);
	desc->setCpuCacheMode(p_pool == POOL_SHARED_WC ? MTL::CPUCacheModeWriteCombined : MTL::CPUCacheModeDefaultCache);
	desc->setHazardTrackingMode(MTL::HazardTrackingModeUntracked);
	desc->setSize(p_size);
	return NS::TransferPtr(device->newHeap(desc.get()));
}

bool MetalHeapAllocator::_pool_allocate(uint32_t p_pool, uint64_t p_size, uint64_t p_align, MetalAllocation &r_allocation, MTL::Heap *&r_heap, uint64_t &r_offset) {
	Pool &pool = pools[p_pool];
	uint64_t padded = p_size + (p_align > UNIT ? p_align - 1 : 0);
	uint32_t units = (uint32_t)((padded + UNIT - 1) / UNIT);

	SpinLockGuard lock(pool.mutex);

	uint32_t block_index = UINT32_MAX;
	OffsetAllocator::Allocation alloc{};
	for (uint32_t i = 0; i < pool.blocks.size(); i++) {
		if (pool.blocks[i].metadata == nullptr) {
			continue; // Reclaimed slot.
		}
		alloc = pool.blocks[i].metadata->allocate(units);
		if (alloc.offset != OffsetAllocator::Allocation::NO_SPACE) {
			block_index = i;
			break;
		}
	}

	if (block_index == UINT32_MAX) {
		uint64_t block_size = MAX(_preferred_block_size(p_pool), (padded + UNIT - 1) & ~(UNIT - 1));
		NS::SharedPtr<MTL::Heap> heap = _create_heap(p_pool, block_size);
		ERR_FAIL_NULL_V_MSG(heap.get(), false, "Unable to create placement heap.");

		Block block;
		block.heap = heap;
		block.metadata = memnew(OffsetAllocator::Allocator((uint32_t)(block_size / UNIT), MAX_ALLOCS_PER_BLOCK));

		// Reuse a reclaimed slot if one exists, else append. Block indices
		// stored in live handles must stay stable, so blocks are never erased.
		block_index = pool.blocks.size();
		for (uint32_t i = 0; i < pool.blocks.size(); i++) {
			if (pool.blocks[i].metadata == nullptr && pool.blocks[i].heap.get() == nullptr) {
				block_index = i;
				break;
			}
		}
		if (block_index == pool.blocks.size()) {
			pool.blocks.push_back(block);
		} else {
			pool.blocks[block_index] = block;
		}
#ifdef DEBUG_ENABLED
		pool.stats.block_count++;
		pool.stats.reserved_bytes += heap->size();
#endif
		generation.fetch_add(1, std::memory_order_relaxed);

		alloc = pool.blocks[block_index].metadata->allocate(units);
		ERR_FAIL_COND_V(alloc.offset == OffsetAllocator::Allocation::NO_SPACE, false);
	}

	Block &block = pool.blocks[block_index];
	if (block.live == 0 && pool.empty_blocks > 0) {
		pool.empty_blocks--;
	}
	block.live++;
#ifdef DEBUG_ENABLED
	pool.stats.allocation_count++;
	pool.stats.used_bytes += (uint64_t)units * UNIT;
#endif

	uint64_t raw_offset = (uint64_t)alloc.offset * UNIT;
	r_offset = (raw_offset + p_align - 1) & ~(p_align - 1);
	r_heap = block.heap.get();
	r_allocation.kind = MetalAllocation::Kind::POOL;
	r_allocation.pool = (uint8_t)p_pool;
	r_allocation.block = block_index;
	r_allocation.alloc = alloc;
	return true;
}

bool MetalHeapAllocator::_dedicated_allocate(uint32_t p_pool, uint64_t p_size, MetalAllocation &r_allocation, MTL::Heap *&r_heap) {
	NS::SharedPtr<MTL::Heap> heap = _create_heap(p_pool, p_size);
	ERR_FAIL_NULL_V_MSG(heap.get(), false, "Unable to create dedicated placement heap.");

	Pool &pool = pools[p_pool];
	SpinLockGuard lock(pool.mutex);
	pool.dedicated_heaps.push_back(heap.get());
	heap->retain(); // dedicated_heaps holds an owned reference.
#ifdef DEBUG_ENABLED
	pool.stats.dedicated_count++;
	pool.stats.dedicated_bytes += heap->size();
#endif
	generation.fetch_add(1, std::memory_order_relaxed);

	r_heap = heap.get();
	r_allocation.kind = MetalAllocation::Kind::DEDICATED;
	r_allocation.pool = (uint8_t)p_pool;
	r_allocation.heap = heap.get();
	r_allocation.heap->retain(); // Handle's owned reference, released in _free_allocation.
	return true;
}

void MetalHeapAllocator::_free_allocation(MetalAllocation &p_allocation) {
	// old_heap takes ownership of the MTL::Heap, so it is released outside any locks.
	NS::SharedPtr<MTL::Heap> old_heap;

	switch (p_allocation.kind) {
		case MetalAllocation::Kind::INVALID:
			break;
		case MetalAllocation::Kind::POOL: {
			Pool &pool = pools[p_allocation.pool];
			SpinLockGuard lock(pool.mutex);
			Block &block = pool.blocks[p_allocation.block];
#ifdef DEBUG_ENABLED
			uint64_t freed_bytes = (uint64_t)block.metadata->allocationSize(p_allocation.alloc) * UNIT;
			pool.stats.allocation_count--;
			pool.stats.used_bytes -= freed_bytes;
#endif
			block.metadata->free(p_allocation.alloc);
			block.live--;
			if (block.live == 0) {
				pool.empty_blocks++;
				// Keep one empty block cached; reclaim beyond that.
				if (pool.empty_blocks > 1) {
#ifdef DEBUG_ENABLED
					pool.stats.block_count--;
					pool.stats.reserved_bytes -= block.heap->size();
#endif
					memdelete(block.metadata);
					block.metadata = nullptr;
					old_heap = std::move(block.heap);
					pool.empty_blocks--;
					generation.fetch_add(1, std::memory_order_relaxed);
				}
			}
		} break;
		case MetalAllocation::Kind::DEDICATED: {
			Pool &pool = pools[p_allocation.pool];
			SpinLockGuard lock(pool.mutex);
			int64_t idx = pool.dedicated_heaps.find(p_allocation.heap);
			DEV_ASSERT(idx >= 0);
			if (likely(idx >= 0)) {
#ifdef DEBUG_ENABLED
				pool.stats.dedicated_count--;
				pool.stats.dedicated_bytes -= p_allocation.heap->size();
#endif
				pool.dedicated_heaps[idx]->release(); // not released here, as p_allocation also has a reference
				pool.dedicated_heaps.remove_at_unordered(idx);
			}
			old_heap = NS::TransferPtr(p_allocation.heap);
			p_allocation.heap = nullptr;
			generation.fetch_add(1, std::memory_order_relaxed);
		} break;
	}
	p_allocation.invalidate();
}

MetalBuffer MetalHeapAllocator::new_buffer(NS::UInteger p_length, MTL::ResourceOptions p_options) {
	uint32_t pool_index = _pool_for_options(p_options);
	MTL::SizeAndAlign sa = device->heapBufferSizeAndAlign(p_length, p_options);
	// Placement-heap resources inherit the heap's hazard tracking; strip
	// hazard bits and pass storage/cache + untracked only.
	MTL::ResourceOptions heap_options = (p_options & (RESOURCE_STORAGE_MODE_MASK | RESOURCE_CPU_CACHE_MODE_MASK)) | MTL::ResourceHazardTrackingModeUntracked;

	MetalBuffer result;
	MTL::Heap *heap = nullptr;
	if (sa.size > _preferred_block_size(pool_index) / 2) {
		if (!_dedicated_allocate(pool_index, sa.size, result.allocation, heap)) {
			return result;
		}
		result.buffer = NS::TransferPtr(heap->newBuffer(p_length, heap_options, 0));
	} else {
		uint64_t offset = 0;
		if (!_pool_allocate(pool_index, sa.size, sa.align, result.allocation, heap, offset)) {
			return result;
		}
		result.buffer = NS::TransferPtr(heap->newBuffer(p_length, heap_options, offset));
	}
	if (!result.buffer) {
		_free_allocation(result.allocation);
	}
	return result;
}

MetalTexture MetalHeapAllocator::new_texture(const MTL::TextureDescriptor *p_desc) {
	uint32_t pool_index = _pool_for_texture(p_desc);
	MTL::SizeAndAlign sa = device->heapTextureSizeAndAlign(p_desc);

	MetalTexture result;
	MTL::Heap *heap = nullptr;
	if (sa.size > _preferred_block_size(pool_index) / 2) {
		if (!_dedicated_allocate(pool_index, sa.size, result.allocation, heap)) {
			return result;
		}
		result.texture = NS::TransferPtr(heap->newTexture(p_desc, 0));
	} else {
		uint64_t offset = 0;
		if (!_pool_allocate(pool_index, sa.size, sa.align, result.allocation, heap, offset)) {
			return result;
		}
		result.texture = NS::TransferPtr(heap->newTexture(p_desc, offset));
	}
	if (!result.texture) {
		_free_allocation(result.allocation);
	}
	return result;
}

void MetalHeapAllocator::free_buffer(MetalBuffer &p_buffer) {
	p_buffer.buffer.reset();
	_free_allocation(p_buffer.allocation);
}

void MetalHeapAllocator::free_texture(MetalTexture &p_texture) {
	p_texture.texture.reset();
	_free_allocation(p_texture.allocation);
}

uint64_t MetalHeapAllocator::get_heaps(LocalVector<MTL::Heap *> &r_heaps) {
	uint64_t gen = generation.load(std::memory_order_relaxed);
	for (uint32_t p = 0; p < POOL_COUNT; p++) {
		Pool &pool = pools[p];
		SpinLockGuard lock(pool.mutex);
		for (const Block &block : pool.blocks) {
			if (block.heap.get() != nullptr) {
				r_heaps.push_back(block.heap.get());
			}
		}
		for (MTL::Heap *heap : pool.dedicated_heaps) {
			r_heaps.push_back(heap);
		}
	}
	return gen;
}

uint64_t MetalHeapAllocator::get_heap_generation() const {
	return generation.load(std::memory_order_relaxed);
}

#ifdef DEBUG_ENABLED
void MetalHeapAllocator::get_stats(MetalAllocatorStats &r_stats) {
	for (uint32_t p = 0; p < POOL_COUNT; p++) {
		SpinLockGuard lock(pools[p].mutex);
		r_stats.pools[p] = pools[p].stats;
	}
}
#endif

MetalHeapAllocator::~MetalHeapAllocator() {
	for (uint32_t p = 0; p < POOL_COUNT; p++) {
		Pool &pool = pools[p];
		for (Block &block : pool.blocks) {
			if (block.metadata != nullptr) {
				memdelete(block.metadata);
			}
		}
		for (MTL::Heap *heap : pool.dedicated_heaps) {
			heap->release();
		}
	}
}
