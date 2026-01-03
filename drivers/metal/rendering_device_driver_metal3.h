/**************************************************************************/
/*  rendering_device_driver_metal3.h                                      */
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

#import "metal3_objects.h"
#import "rendering_device_driver_metal.h"

#import <Metal/Metal.h>

namespace MTL3 {

class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0)) RenderingDeviceDriverMetal final : public ::RenderingDeviceDriverMetal {
	friend class MDCommandBuffer;
#pragma mark - Generic

	id<MTLCommandQueue> device_queue = nil;

	struct Fence {
		virtual void signal(id<MTLCommandBuffer> p_cmd_buffer) = 0;
		virtual Error wait(uint32_t p_timeout_ms) = 0;
		virtual ~Fence() = default;
	};

	struct FenceEvent : Fence {
		id<MTLSharedEvent> event;
		uint64_t value;
		FenceEvent(id<MTLSharedEvent> p_event) :
				event(p_event), value(0) {}
		void signal(id<MTLCommandBuffer> p_cb) override;
		Error wait(uint32_t p_timeout_ms) override;
	};

	struct FenceSemaphore : Fence {
		dispatch_semaphore_t semaphore;
		FenceSemaphore() :
				semaphore(dispatch_semaphore_create(0)) {}
		void signal(id<MTLCommandBuffer> p_cb) override;
		Error wait(uint32_t p_timeout_ms) override;
	};

	struct Semaphore {
		id<MTLEvent> event;
		uint64_t value;
		Semaphore(id<MTLEvent> p_event) :
				event(p_event), value(0) {}
	};

	Vector<MDCommandBuffer *> command_buffers;

	Error _create_device() override;
	Error _execute_and_present_barriers(CommandQueueID p_cmd_queue, VectorView<SemaphoreID> p_wait_semaphores, VectorView<CommandBufferID> p_cmd_buffers, VectorView<SemaphoreID> p_cmd_semaphores, FenceID p_cmd_fence, VectorView<SwapChainID> p_swap_chains);
	Error _execute_and_present(CommandQueueID p_cmd_queue, VectorView<SemaphoreID> p_wait_semaphores, VectorView<CommandBufferID> p_cmd_buffers, VectorView<SemaphoreID> p_cmd_semaphores, FenceID p_cmd_fence, VectorView<SwapChainID> p_swap_chains);

protected:
	id get_command_queue() const override { return device_queue; }
	GODOT_CLANG_WARNING_PUSH_AND_IGNORE("-Wunguarded-availability")
	void add_residency_set_to_main_queue(id<MTLResidencySet> p_set) override {
		[device_queue addResidencySet:p_set];
	}
	void remove_residency_set_to_main_queue(id<MTLResidencySet> p_set) override {
		[device_queue removeResidencySet:p_set];
	}
	GODOT_CLANG_WARNING_POP
public:
	Error initialize(uint32_t p_device_index, uint32_t p_frame_count) override;

	FenceID fence_create() override;
	Error fence_wait(FenceID p_fence) override;
	void fence_free(FenceID p_fence) override;

	SemaphoreID semaphore_create() override;
	void semaphore_free(SemaphoreID p_semaphore) override;

	CommandQueueID command_queue_create(CommandQueueFamilyID p_cmd_queue_family, bool p_identify_as_main_queue = false) override;
	Error command_queue_execute_and_present(CommandQueueID p_cmd_queue, VectorView<SemaphoreID> p_wait_semaphores, VectorView<CommandBufferID> p_cmd_buffers, VectorView<SemaphoreID> p_cmd_semaphores, FenceID p_cmd_fence, VectorView<SwapChainID> p_swap_chains) override;
	void command_queue_free(CommandQueueID p_cmd_queue) override;

	CommandPoolID command_pool_create(CommandQueueFamilyID p_cmd_queue_family, CommandBufferType p_cmd_buffer_type) override;
	bool command_pool_reset(CommandPoolID p_cmd_pool) override;
	void command_pool_free(CommandPoolID p_cmd_pool) override;

	CommandBufferID command_buffer_create(CommandPoolID p_cmd_pool) override;

#pragma mark - Miscellaneous

	String get_api_name() const override { return "Metal"; }

	RenderingDeviceDriverMetal(RenderingContextDriverMetal *p_context_driver);
	~RenderingDeviceDriverMetal();
};

} // namespace MTL3
