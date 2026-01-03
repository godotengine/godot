/**************************************************************************/
/*  rendering_device_driver_metal3.mm                                     */
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

#import "rendering_device_driver_metal3.h"

#import "pixel_formats.h"
#import "rendering_context_driver_metal.h"

#import "core/config/project_settings.h"
#import "core/string/ustring.h"
#import "drivers/apple/foundation_helpers.h"

#import <Metal/Metal.h>
#import <os/log.h>

#pragma mark - Logging

extern os_log_t LOG_DRIVER;

namespace MTL3 {

#pragma mark - FenceEvent / FenceSemaphore

void RenderingDeviceDriverMetal::FenceEvent::signal(id<MTLCommandBuffer> p_cb) {
	if (p_cb) {
		value++;
		[p_cb encodeSignalEvent:event value:value];
	}
}

Error RenderingDeviceDriverMetal::FenceEvent::wait(uint32_t p_timeout_ms) {
	GODOT_CLANG_WARNING_PUSH_AND_IGNORE("-Wunguarded-availability")
	BOOL signaled = [event waitUntilSignaledValue:value timeoutMS:p_timeout_ms];
	GODOT_CLANG_WARNING_POP
	if (!signaled) {
#ifdef DEBUG_ENABLED
		ERR_PRINT("timeout waiting for fence");
#endif
		return ERR_TIMEOUT;
	}
	return OK;
}

void RenderingDeviceDriverMetal::FenceSemaphore::signal(id<MTLCommandBuffer> p_cb) {
	if (p_cb) {
		[p_cb addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
			dispatch_semaphore_signal(semaphore);
		}];
	} else {
		dispatch_semaphore_signal(semaphore);
	}
}

Error RenderingDeviceDriverMetal::FenceSemaphore::wait(uint32_t p_timeout_ms) {
	dispatch_time_t timeout = dispatch_time(DISPATCH_TIME_NOW, static_cast<int64_t>(p_timeout_ms) * 1000000);
	long result = dispatch_semaphore_wait(semaphore, timeout);
	if (result != 0) {
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

	device_queue = [device newCommandQueue];
	device_queue.label = @"Godot Main Command Queue";
	ERR_FAIL_NULL_V(device_queue, ERR_CANT_CREATE);

	return OK;
}

Error RenderingDeviceDriverMetal::initialize(uint32_t p_device_index, uint32_t p_frame_count) {
	Error err = _initialize(p_device_index, p_frame_count);
	ERR_FAIL_COND_V(err, err);

	if (@available(macOS 26.0, iOS 26.0, tvOS 26.0, *)) {
		// Check if the user has explicitly enabled resource barriers.
		bool barriers_enabled = GLOBAL_GET("rendering/rendering_device/metal3/enable_pipeline_barriers");
		barriers_enabled |= OS::get_singleton()->get_environment("GODOT_MTL_FORCE_BARRIERS") == "1";
		if (barriers_enabled) {
			print_line("Metal3: Resource barriers enabled.");
			GODOT_CLANG_WARNING_PUSH_AND_IGNORE("-Wunguarded-availability")
			MTLResidencySetDescriptor *rs_desc = [MTLResidencySetDescriptor new];
			[rs_desc setInitialCapacity:250];
			rs_desc.label = @"Main Residency Set";
			NSError *error;
			main_residency_set = [device newResidencySetWithDescriptor:rs_desc error:&error];
			if (main_residency_set == nil) {
				String error_msg = conv::to_string(error.localizedDescription);
				print_error(vformat("Resource barriers unavailable. Failed to create main residency set for explicit resource barriers: %s", error_msg));
			} else {
				use_barriers = true;
				base_hazard_tracking = MTLResourceHazardTrackingModeUntracked;
				[device_queue addResidencySet:main_residency_set];
			}
			GODOT_CLANG_WARNING_POP;
		} else {
			print_verbose("Metal3: Resource barriers are disabled.");
		}
	}

	return OK;
}

#pragma mark - Fences

RDD::FenceID RenderingDeviceDriverMetal::fence_create() {
	Fence *fence = nullptr;
	if (@available(macOS 10.14, iOS 12.0, tvOS 12.0, visionOS 1.0, *)) {
		fence = memnew(FenceEvent([device newSharedEvent]));
	} else {
		fence = memnew(FenceSemaphore());
	}
	return FenceID(fence);
}

Error RenderingDeviceDriverMetal::fence_wait(FenceID p_fence) {
	Fence *fence = (Fence *)(p_fence.id);
	return fence->wait(1000);
}

void RenderingDeviceDriverMetal::fence_free(FenceID p_fence) {
	Fence *fence = (Fence *)(p_fence.id);
	memdelete(fence);
}

#pragma mark - Semaphores

RDD::SemaphoreID RenderingDeviceDriverMetal::semaphore_create() {
	if (use_barriers) {
		Semaphore *sem = memnew(Semaphore(device.newEvent));
		return SemaphoreID(sem);
	}
	return SemaphoreID(1);
}

void RenderingDeviceDriverMetal::semaphore_free(SemaphoreID p_semaphore) {
	if (use_barriers) {
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

	bool changed = false;
	if (!_residency_add.is_empty()) {
		[main_residency_set addAllocations:(id<MTLAllocation> *)_residency_add.ptr() count:_residency_add.size()];
		_residency_add.clear();
		changed = true;
	}
	if (!_residency_del.is_empty()) {
		[main_residency_set removeAllocations:(id<MTLAllocation> *)_residency_del.ptr() count:_residency_del.size()];
		_residency_del.clear();
		changed = true;
	}
	if (changed) {
		[main_residency_set commit];
	}

	if (p_wait_sem.size() > 0) {
		id<MTLCommandBuffer> cb = [device_queue commandBuffer];
#ifdef DEV_ENABLED
		cb.label = @"Wait Command Buffer";
#endif
		for (uint32_t i = 0; i < p_wait_sem.size(); i++) {
			Semaphore *sem = (Semaphore *)p_wait_sem[i].id;
			[cb encodeWaitForEvent:sem->event value:sem->value];
		}
		[cb commit];
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
		id<MTLCommandBuffer> cb = cmd_buffer->get_command_buffer();
		fence->signal(cb);
	}

	struct DrawRequest {
		id<MTLDrawable> drawable;
		DisplayServer::VSyncMode vsync_mode;
		double duration;
	};

	id<MTLCommandBuffer> cb = nil;

	if (p_swap_chains.size() > 0) {
		Vector<DrawRequest> drawables;
		drawables.reserve(p_swap_chains.size());

		for (uint32_t i = 0; i < p_swap_chains.size(); i++) {
			SwapChain *swap_chain = (SwapChain *)(p_swap_chains[i].id);
			RenderingContextDriverMetal::Surface *metal_surface = (RenderingContextDriverMetal::Surface *)(swap_chain->surface);
			id<MTLDrawable> drawable = metal_surface->next_drawable();
			if (drawable) {
				drawables.push_back(DrawRequest{
						.drawable = drawable,
						.vsync_mode = metal_surface->vsync_mode,
						.duration = metal_surface->present_minimum_duration,
				});
			}
		}

		cb = cmd_buffer->get_command_buffer();
		[cb addCompletedHandler:^(id<MTLCommandBuffer>) {
			for (const DrawRequest &dr : drawables) {
				switch (dr.vsync_mode) {
					case DisplayServer::VSYNC_DISABLED: {
						[dr.drawable present];
					} break;
					default: {
						[dr.drawable presentAfterMinimumDuration:dr.duration];
					} break;
				}
			}
		}];
	}

	cmd_buffer->commit();

	if (p_cmd_sem.size() > 0) {
		id<MTLCommandBuffer> cb = [device_queue commandBuffer];
		for (uint32_t i = 0; i < p_cmd_sem.size(); i++) {
			Semaphore *sem = (Semaphore *)p_cmd_sem[i].id;
			sem->value++;
			[cb encodeSignalEvent:sem->event value:sem->value];
		}
		[cb commit];
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
		id<MTLCommandBuffer> cb = cmd_buffer->get_command_buffer();
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
	if (use_barriers) {
		res = _execute_and_present_barriers(p_cmd_queue, p_wait_sem, p_cmd_buffers, p_cmd_sem, p_cmd_fence, p_swap_chains);
	} else {
		res = _execute_and_present(p_cmd_queue, p_wait_sem, p_cmd_buffers, p_cmd_sem, p_cmd_fence, p_swap_chains);
	}
	ERR_FAIL_COND_V(res != OK, res);

	if (p_swap_chains.size() > 0) {
		// Used as a signal that we're presenting, so this is the end of a frame.
		[device_scope endScope];
		[device_scope beginScope];
	}

	return OK;
}

void RenderingDeviceDriverMetal::command_queue_free(CommandQueueID p_cmd_queue) {
}

#pragma mark - Command Pools

RDD::CommandPoolID RenderingDeviceDriverMetal::command_pool_create(CommandQueueFamilyID p_cmd_queue_family, CommandBufferType p_cmd_buffer_type) {
	DEV_ASSERT(p_cmd_buffer_type == COMMAND_BUFFER_TYPE_PRIMARY);
	return rid::make(device_queue);
}

bool RenderingDeviceDriverMetal::command_pool_reset(CommandPoolID p_cmd_pool) {
	return true;
}

void RenderingDeviceDriverMetal::command_pool_free(CommandPoolID p_cmd_pool) {
	rid::release(p_cmd_pool);
}

#pragma mark - Command Buffers

RDD::CommandBufferID RenderingDeviceDriverMetal::command_buffer_create(CommandPoolID p_cmd_pool) {
	id<MTLCommandQueue> queue = rid::get(p_cmd_pool);
	MDCommandBuffer *obj = memnew(MDCommandBuffer(queue, this));
	command_buffers.push_back(obj);
	return CommandBufferID(obj);
}

} // namespace MTL3
