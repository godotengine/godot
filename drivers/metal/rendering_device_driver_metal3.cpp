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

#include "pixel_formats.h"
#include "rendering_context_driver_metal.h"

#include "core/config/project_settings.h"
#include "core/string/ustring.h"

namespace MTL3 {

#pragma mark - FenceEvent / FenceSemaphore

void RenderingDeviceDriverMetal::FenceEvent::signal(MTL::CommandBuffer *p_cb) {
	if (p_cb) {
		value++;
		p_cb->encodeSignalEvent(event.get(), value);
	}
}

Error RenderingDeviceDriverMetal::FenceEvent::wait(uint32_t p_timeout_ms) {
	bool signaled = event->waitUntilSignaledValue(value, p_timeout_ms);
	if (!signaled) {
#ifdef DEBUG_ENABLED
		ERR_PRINT("timeout waiting for fence");
#endif
		return ERR_TIMEOUT;
	}
	return OK;
}

void RenderingDeviceDriverMetal::FenceSemaphore::signal(MTL::CommandBuffer *p_cb) {
	if (p_cb) {
		p_cb->addCompletedHandler([this](MTL::CommandBuffer *) {
			dispatch_semaphore_signal(semaphore);
		});
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

	device_queue = NS::TransferPtr(device->newCommandQueue());
	ERR_FAIL_NULL_V(device_queue.get(), ERR_CANT_CREATE);
	device_queue->setLabel(MTLSTR("Godot Main Command Queue"));

	return OK;
}

Error RenderingDeviceDriverMetal::initialize(uint32_t p_device_index, uint32_t p_frame_count) {
	Error err = _initialize(p_device_index, p_frame_count);
	ERR_FAIL_COND_V(err, err);

	// Barriers are still experimental in Metal 3, so they are disabled by default
	// and can only be enabled via an environment variable.
	bool barriers_enabled = OS::get_singleton()->get_environment("GODOT_MTL_FORCE_BARRIERS") == "1";
	if (__builtin_available(macos 26.0, ios 26.0, tvos 26.0, visionos 26.0, *)) {
		if (barriers_enabled) {
			print_line("Metal 3: Resource barriers enabled.");
			NS::SharedPtr<MTL::ResidencySetDescriptor> rs_desc = NS::TransferPtr(MTL::ResidencySetDescriptor::alloc()->init());
			rs_desc->setInitialCapacity(250);
			rs_desc->setLabel(MTLSTR("Main Residency Set"));
			NS::Error *error = nullptr;
			NS::SharedPtr<MTL::ResidencySet> mrs = NS::TransferPtr(device->newResidencySet(rs_desc.get(), &error));
			if (!mrs) {
				String error_msg = error ? String(error->localizedDescription()->utf8String()) : "Unknown error";
				print_error(vformat("Resource barriers unavailable. Failed to create main residency set for explicit resource barriers: %s", error_msg));
			} else {
				use_barriers = true;
				base_hazard_tracking = MTL::ResourceHazardTrackingModeUntracked;
				main_residency_set = mrs;
				device_queue->addResidencySet(mrs.get());
			}
		}
	} else {
		if (barriers_enabled) {
			// Application or user has requested barriers, but the OS doesn't support them.
			print_verbose("Metal 3: Resource barriers are not supported on this OS version.");
			barriers_enabled = false;
		}
	}

	return OK;
}

#pragma mark - Residency

void RenderingDeviceDriverMetal::add_residency_set_to_main_queue(MTL::ResidencySet *p_set) {
	device_queue->addResidencySet(p_set);
}

void RenderingDeviceDriverMetal::remove_residency_set_to_main_queue(MTL::ResidencySet *p_set) {
	device_queue->removeResidencySet(p_set);
}

#pragma mark - Fences

RDD::FenceID RenderingDeviceDriverMetal::fence_create() {
	Fence *fence = memnew(FenceEvent(NS::TransferPtr(device->newSharedEvent())));
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
		Semaphore *sem = memnew(Semaphore(NS::TransferPtr(device->newEvent())));
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
	MTL::ResidencySet *mrs = main_residency_set.get();
	if (!_residency_add.is_empty()) {
		mrs->addAllocations(reinterpret_cast<const MTL::Allocation *const *>(_residency_add.ptr()), _residency_add.size());
		_residency_add.clear();
		changed = true;
	}
	if (!_residency_del.is_empty()) {
		mrs->removeAllocations(reinterpret_cast<const MTL::Allocation *const *>(_residency_del.ptr()), _residency_del.size());
		_residency_del.clear();
		changed = true;
	}
	if (changed) {
		mrs->commit();
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

	struct DrawRequest {
		NS::SharedPtr<MTL::Drawable> drawable;
		DisplayServer::VSyncMode vsync_mode;
		double duration;
	};

	if (p_swap_chains.size() > 0) {
		Vector<DrawRequest> drawables;
		drawables.reserve(p_swap_chains.size());

		for (uint32_t i = 0; i < p_swap_chains.size(); i++) {
			SwapChain *swap_chain = (SwapChain *)(p_swap_chains[i].id);
			RenderingContextDriverMetal::Surface *metal_surface = (RenderingContextDriverMetal::Surface *)(swap_chain->surface);
			MTL::Drawable *drawable = metal_surface->next_drawable();
			if (drawable) {
				drawables.push_back(DrawRequest{
						.drawable = NS::RetainPtr(drawable),
						.vsync_mode = metal_surface->vsync_mode,
						.duration = metal_surface->present_minimum_duration,
				});
			}
		}

		MTL::CommandBuffer *cb = cmd_buffer->get_command_buffer();
		cb->addCompletedHandler([drawables = std::move(drawables)](MTL::CommandBuffer *) {
			for (const DrawRequest &dr : drawables) {
				switch (dr.vsync_mode) {
					case DisplayServer::VSYNC_DISABLED: {
						dr.drawable->present();
					} break;
					default: {
						dr.drawable->presentAfterMinimumDuration(dr.duration);
					} break;
				}
			}
		});
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
	if (use_barriers) {
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

#pragma mark - Command Buffers

RDD::CommandBufferID RenderingDeviceDriverMetal::command_buffer_create(CommandPoolID p_cmd_pool) {
	MTL::CommandQueue *queue = reinterpret_cast<MTL::CommandQueue *>(p_cmd_pool.id);
	MDCommandBuffer *obj = memnew(MDCommandBuffer(queue, this));
	command_buffers.push_back(obj);
	return CommandBufferID(obj);
}

} // namespace MTL3
