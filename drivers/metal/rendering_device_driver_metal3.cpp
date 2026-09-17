/**************************************************************************/
/*  rendering_device_driver_metal3.cpp                                    */
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

#include "rendering_device_driver_metal3.h"

#include "core/config/project_settings.h"
#include "core/os/os.h"
#include "core/string/ustring.h"
#include "drivers/metal/pixel_formats.h"
#include "drivers/metal/rendering_context_driver_metal.h"

namespace MTL3 {

#pragma mark - Fence

void RenderingDeviceDriverMetal::Fence::signal(MTL::CommandBuffer *p_cb) {
	if (p_cb) {
		value++;
		p_cb->encodeSignalEvent(event.get(), value);
	}
}

Error RenderingDeviceDriverMetal::Fence::wait(uint32_t p_timeout_ms) {
	if (unlikely(value == 0)) {
		WARN_PRINT_ONCE("Never signaled fence.");
		return OK;
	}
	bool signaled = event->waitUntilSignaledValue(value, p_timeout_ms);
	if (!signaled) {
#ifdef DEBUG_ENABLED
		ERR_PRINT("timeout waiting for fence");
#endif
		return ERR_TIMEOUT;
	}
	return OK;
}

#pragma mark - Constructor / Destructor

RenderingDeviceDriverMetal::RenderingDeviceDriverMetal(RenderingContextDriverMetal *p_context_driver) :
		::RenderingDeviceDriverMetal(p_context_driver) {
}

RenderingDeviceDriverMetal::~RenderingDeviceDriverMetal() {
	for (MDCommandBuffer *cb : command_buffers) {
		memdelete(cb);
	}
}

#pragma mark - Initialization

Error RenderingDeviceDriverMetal::_create_device() {
	Error err = ::RenderingDeviceDriverMetal::_create_device();
	ERR_FAIL_COND_V(err, err);

	device_queue = NS::TransferPtr(device->newCommandQueue());
	ERR_FAIL_NULL_V(device_queue.get(), ERR_CANT_CREATE);
	device_queue->setLabel(MTLSTR("Godot Main Command Queue"));

	return OK;
}

void RenderingDeviceDriverMetal::_resolve_sync_mode() {
	if (sync_mode == Barriers) {
		print_verbose("Metal 3: Barrier synchronization enabled.");
		base_hazard_tracking = MTL::ResourceHazardTrackingModeUntracked;
		// Apple GPUs only sample counters at stage boundaries, which is all the
		// timestamp encoders rely on.
		timestamp_queries_supported = device->supportsCounterSampling(MTL::CounterSamplingPointAtStageBoundary);
	} else {
		print_verbose("Metal 3: Native hazard tracking enabled.");
	}
}

Error RenderingDeviceDriverMetal::initialize(uint32_t p_device_index, uint32_t p_frame_count) {
	Error err = _initialize(p_device_index, p_frame_count);
	ERR_FAIL_COND_V(err, err);

	return OK;
}

#pragma mark - Residency

void RenderingDeviceDriverMetal::add_residency_set_to_main_queue(MTL::ResidencySet *p_set) {
}

void RenderingDeviceDriverMetal::remove_residency_set_to_main_queue(MTL::ResidencySet *p_set) {
}

#pragma mark - Fences

RDD::FenceID RenderingDeviceDriverMetal::fence_create() {
	Fence *fence = memnew(Fence(NS::TransferPtr(device->newSharedEvent())));
	return FenceID(fence);
}

Error RenderingDeviceDriverMetal::fence_wait(FenceID p_fence) {
	Fence *fence = (Fence *)(p_fence.id);
	return fence->wait(2000);
}

void RenderingDeviceDriverMetal::fence_free(FenceID p_fence) {
	Fence *fence = (Fence *)(p_fence.id);
	memdelete(fence);
}

#pragma mark - Semaphores

RDD::SemaphoreID RenderingDeviceDriverMetal::semaphore_create() {
	if (sync_mode != HazardTracking) {
		Semaphore *sem = memnew(Semaphore(NS::TransferPtr(device->newEvent())));
		return SemaphoreID(sem);
	}
	return SemaphoreID(1);
}

void RenderingDeviceDriverMetal::semaphore_free(SemaphoreID p_semaphore) {
	if (sync_mode != HazardTracking) {
		Semaphore *sem = (Semaphore *)(p_semaphore.id);
		memdelete(sem);
	}
}

#pragma mark - Command Queues

RDD::CommandQueueID RenderingDeviceDriverMetal::command_queue_create(CommandQueueFamilyID p_cmd_queue_family, bool p_identify_as_main_queue) {
	return CommandQueueID(1);
}

Error RenderingDeviceDriverMetal::_execute_and_present_barriers(CommandQueueID p_cmd_queue, VectorView<SemaphoreID> p_wait_sem, VectorView<CommandBufferID> p_cmd_buffers, VectorView<SemaphoreID> p_cmd_sem, FenceID p_cmd_fence, VectorView<SwapChainID> p_swap_chains) {
	uint32_t size = p_cmd_buffers.size();
	if (size == 0) {
		return OK;
	}

	if (p_wait_sem.size() > 0) {
		MTL::CommandBuffer *cb = device_queue->commandBuffer();
#ifdef DEV_ENABLED
		cb->setLabel(MTLSTR("Wait Command Buffer"));
#endif
		for (uint32_t i = 0; i < p_wait_sem.size(); i++) {
			Semaphore *sem = (Semaphore *)p_wait_sem[i].id;
			cb->encodeWait(sem->event.get(), sem->value);
		}
		cb->commit();
	}

	for (uint32_t i = 0; i < size - 1; i++) {
		MDCommandBuffer *cmd_buffer = (MDCommandBuffer *)(p_cmd_buffers[i].id);
		cmd_buffer->commit();
	}

	// The last command buffer will signal the fence and semaphores.
	MDCommandBuffer *cmd_buffer = (MDCommandBuffer *)(p_cmd_buffers[size - 1].id);
	Fence *fence = (Fence *)(p_cmd_fence.id);
	if (fence != nullptr) {
		cmd_buffer->end();
		MTL::CommandBuffer *cb = cmd_buffer->get_command_buffer();
		fence->signal(cb);
	}

	for (uint32_t i = 0; i < p_swap_chains.size(); i++) {
		SwapChain *swap_chain = (SwapChain *)(p_swap_chains[i].id);
		RenderingContextDriverMetal::Surface *metal_surface = (RenderingContextDriverMetal::Surface *)(swap_chain->surface);
		metal_surface->present(cmd_buffer);
	}

	cmd_buffer->commit();

	if (p_cmd_sem.size() > 0) {
		MTL::CommandBuffer *cb = device_queue->commandBuffer();
		for (uint32_t i = 0; i < p_cmd_sem.size(); i++) {
			Semaphore *sem = (Semaphore *)p_cmd_sem[i].id;
			sem->value++;
			cb->encodeSignalEvent(sem->event.get(), sem->value);
		}
		cb->commit();
	}

	return OK;
}

Error RenderingDeviceDriverMetal::_execute_and_present(CommandQueueID p_cmd_queue, VectorView<SemaphoreID> p_wait_sem, VectorView<CommandBufferID> p_cmd_buffers, VectorView<SemaphoreID> p_cmd_sem, FenceID p_cmd_fence, VectorView<SwapChainID> p_swap_chains) {
	uint32_t size = p_cmd_buffers.size();
	if (size == 0) {
		return OK;
	}

	for (uint32_t i = 0; i < size - 1; i++) {
		MDCommandBuffer *cmd_buffer = (MDCommandBuffer *)(p_cmd_buffers[i].id);
		cmd_buffer->commit();
	}

	// The last command buffer will signal the fence and semaphores.
	MDCommandBuffer *cmd_buffer = (MDCommandBuffer *)(p_cmd_buffers[size - 1].id);
	Fence *fence = (Fence *)(p_cmd_fence.id);
	if (fence != nullptr) {
		cmd_buffer->end();
		MTL::CommandBuffer *cb = cmd_buffer->get_command_buffer();
		fence->signal(cb);
	}

	for (uint32_t i = 0; i < p_swap_chains.size(); i++) {
		SwapChain *swap_chain = (SwapChain *)(p_swap_chains[i].id);
		RenderingContextDriverMetal::Surface *metal_surface = (RenderingContextDriverMetal::Surface *)(swap_chain->surface);
		metal_surface->present(cmd_buffer);
	}

	cmd_buffer->commit();

	return OK;
}

Error RenderingDeviceDriverMetal::command_queue_execute_and_present(CommandQueueID p_cmd_queue, VectorView<SemaphoreID> p_wait_sem, VectorView<CommandBufferID> p_cmd_buffers, VectorView<SemaphoreID> p_cmd_sem, FenceID p_cmd_fence, VectorView<SwapChainID> p_swap_chains) {
	Error res;
	if (sync_mode != HazardTracking) {
		res = _execute_and_present_barriers(p_cmd_queue, p_wait_sem, p_cmd_buffers, p_cmd_sem, p_cmd_fence, p_swap_chains);
	} else {
		res = _execute_and_present(p_cmd_queue, p_wait_sem, p_cmd_buffers, p_cmd_sem, p_cmd_fence, p_swap_chains);
	}
	ERR_FAIL_COND_V(res != OK, res);

	if (p_swap_chains.size() > 0) {
		// Used as a signal that we're presenting, so this is the end of a frame.
		MTL::CaptureScope *scope = device_scope.get();
		scope->endScope();
		scope->beginScope();
	}

	return OK;
}

void RenderingDeviceDriverMetal::command_queue_free(CommandQueueID p_cmd_queue) {
}

#pragma mark - Command Pools

RDD::CommandPoolID RenderingDeviceDriverMetal::command_pool_create(CommandQueueFamilyID p_cmd_queue_family, CommandBufferType p_cmd_buffer_type) {
	DEV_ASSERT(p_cmd_buffer_type == COMMAND_BUFFER_TYPE_PRIMARY);
	return CommandPoolID(reinterpret_cast<uint64_t>(device_queue.get()));
}

bool RenderingDeviceDriverMetal::command_pool_reset(CommandPoolID p_cmd_pool) {
	return true;
}

void RenderingDeviceDriverMetal::command_pool_free(CommandPoolID p_cmd_pool) {
	// Nothing to free - the device_queue is managed by SharedPtr.
}

#pragma mark - Timestamp

RDD::QueryPoolID RenderingDeviceDriverMetal::timestamp_query_pool_create(uint32_t p_query_count) {
	if (timestamp_queries_supported) {
		QueryPool *pool = QueryPool::create(device, p_query_count);
		if (pool != nullptr) {
			return QueryPoolID(pool);
		}
		WARN_PRINT_ONCE("Metal 3: The device does not expose the timestamp counter set; GPU timestamps are unavailable.");
		timestamp_queries_supported = false;
	}
	return ::RenderingDeviceDriverMetal::timestamp_query_pool_create(p_query_count);
}

void RenderingDeviceDriverMetal::timestamp_query_pool_free(QueryPoolID p_pool_id) {
	if (!timestamp_queries_supported) {
		return ::RenderingDeviceDriverMetal::timestamp_query_pool_free(p_pool_id);
	}
	QueryPool *pool = (QueryPool *)(p_pool_id.id);
	memdelete(pool);
}

void RenderingDeviceDriverMetal::timestamp_query_pool_get_results(QueryPoolID p_pool_id, uint32_t p_query_count, uint64_t *r_results) {
	if (!timestamp_queries_supported) {
		return ::RenderingDeviceDriverMetal::timestamp_query_pool_get_results(p_pool_id, p_query_count, r_results);
	}
	QueryPool *pool = (QueryPool *)(p_pool_id.id);
	pool->get_results(p_query_count, r_results);
}

uint64_t RenderingDeviceDriverMetal::timestamp_query_result_to_time(uint64_t p_result) {
	return p_result; // Already converted to nanoseconds in get_results.
}

void RenderingDeviceDriverMetal::command_timestamp_write(CommandBufferID p_cmd_buffer, QueryPoolID p_pool_id, uint32_t p_index) {
	if (!timestamp_queries_supported) {
		return ::RenderingDeviceDriverMetal::command_timestamp_write(p_cmd_buffer, p_pool_id, p_index);
	}
	MDCommandBuffer *cmd = (MDCommandBuffer *)(p_cmd_buffer.id);
	QueryPool *pool = (QueryPool *)(p_pool_id.id);
	cmd->timestamp_write(pool, p_index);
}

#pragma mark - Command Buffers

RDD::CommandBufferID RenderingDeviceDriverMetal::command_buffer_create(CommandPoolID p_cmd_pool) {
	MTL::CommandQueue *queue = reinterpret_cast<MTL::CommandQueue *>(p_cmd_pool.id);
	MDCommandBuffer *obj = memnew(MDCommandBuffer(queue, this));
	command_buffers.push_back(obj);
	return CommandBufferID(obj);
}

} // namespace MTL3
