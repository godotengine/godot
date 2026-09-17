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

#include "drivers/metal/metal3_objects.h"
#include "drivers/metal/rendering_device_driver_metal.h"

#include <Metal/Metal.hpp>

namespace MTL3 {

class API_AVAILABLE(macos(11.0), ios(14.0), tvos(14.0)) RenderingDeviceDriverMetal final : public ::RenderingDeviceDriverMetal {
	friend class MDCommandBuffer;
#pragma mark - Generic

	NS::SharedPtr<MTL::CommandQueue> device_queue;

	struct Fence {
		NS::SharedPtr<MTL::SharedEvent> event;
		uint64_t value = 0;
		Fence(NS::SharedPtr<MTL::SharedEvent> p_event) :
				event(p_event) {}
		void signal(MTL::CommandBuffer *p_cb);
		Error wait(uint32_t p_timeout_ms);
	};

	struct Semaphore {
		NS::SharedPtr<MTL::Event> event;
		uint64_t value = 0;
		Semaphore(NS::SharedPtr<MTL::Event> p_event) :
				event(p_event) {}
	};

	Vector<MDCommandBuffer *> command_buffers;

	// GPU timestamps require barrier synchronization, so the level fences order the
	// timestamp encoders against the work they bracket.
	bool timestamp_queries_supported = false;

	Error _execute_and_present_barriers(CommandQueueID p_cmd_queue, VectorView<SemaphoreID> p_wait_semaphores, VectorView<CommandBufferID> p_cmd_buffers, VectorView<SemaphoreID> p_cmd_semaphores, FenceID p_cmd_fence, VectorView<SwapChainID> p_swap_chains);
	Error _execute_and_present(CommandQueueID p_cmd_queue, VectorView<SemaphoreID> p_wait_semaphores, VectorView<CommandBufferID> p_cmd_buffers, VectorView<SemaphoreID> p_cmd_semaphores, FenceID p_cmd_fence, VectorView<SwapChainID> p_swap_chains);

protected:
	Error _create_device() override;
	void _resolve_sync_mode() override;
	MTL::CommandQueue *get_command_queue() const override { return device_queue.get(); }
	void add_residency_set_to_main_queue(MTL::ResidencySet *p_set) override API_AVAILABLE(macos(15.0), ios(18.0), tvos(18.0), visionos(2.0));
	void remove_residency_set_to_main_queue(MTL::ResidencySet *p_set) override API_AVAILABLE(macos(15.0), ios(18.0), tvos(18.0), visionos(2.0));

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

#pragma mark - Timestamp

	QueryPoolID timestamp_query_pool_create(uint32_t p_query_count) override;
	void timestamp_query_pool_free(QueryPoolID p_pool_id) override;
	void timestamp_query_pool_get_results(QueryPoolID p_pool_id, uint32_t p_query_count, uint64_t *r_results) override;
	uint64_t timestamp_query_result_to_time(uint64_t p_result) override;
	void command_timestamp_write(CommandBufferID p_cmd_buffer, QueryPoolID p_pool_id, uint32_t p_index) override;

#pragma mark - Miscellaneous

	String get_api_name() const override { return "Metal"; }

	RenderingDeviceDriverMetal(RenderingContextDriverMetal *p_context_driver);
	~RenderingDeviceDriverMetal();
};

} // namespace MTL3
