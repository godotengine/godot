/**************************************************************************/
/*  thread_safe_frame_buffer.cpp                                         */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#include "thread_safe_frame_buffer.h"
#include "core/os/os.h"

ThreadSafeFrameBuffer::ThreadSafeFrameBuffer() {
	// Initialize atomic variables
	has_new_data.store(false);
	last_sequence.store(0);
	total_updates.store(0);
	buffer_switches.store(0);
}

ThreadSafeFrameBuffer::~ThreadSafeFrameBuffer() {
	// Clean up resources
	reset();
}

void ThreadSafeFrameBuffer::update_frame(const Ref<Image> &new_frame, uint64_t timestamp, uint32_t sequence) {
	// Parameter validation
	if (new_frame.is_null()) {
		return;
	}
	
	total_updates.fetch_add(1);
	
	// Fast path: get a reference to the write buffer without locking
	FrameData *write_buffer = writing_to_a ? &buffer_a : &buffer_b;
	
	// Update write buffer (non-critical section)
	write_buffer->image = new_frame;
	write_buffer->game_timestamp = timestamp;
	write_buffer->frame_sequence = sequence;
	write_buffer->is_new_frame = true;
	
	// Critical section: switch buffer pointers
	{
		MutexLock lock(buffer_mutex);
		writing_to_a = !writing_to_a;
		buffer_switches.fetch_add(1);
	}
	
	// Update status flags
	has_new_data.store(true);
	last_sequence.store(sequence);
}

ThreadSafeFrameBuffer::FrameData ThreadSafeFrameBuffer::get_current_frame() const {
	MutexLock lock(buffer_mutex);
	
	// Get the stable read buffer (not the write buffer)
	const FrameData *read_buffer = writing_to_a ? &buffer_b : &buffer_a;
	
	// Copy data (within the lock to ensure consistency)
	FrameData result = *read_buffer;
	
	return result;
}

void ThreadSafeFrameBuffer::reset() {
	MutexLock lock(buffer_mutex);
	
	// Clear buffers
	buffer_a = FrameData();
	buffer_b = FrameData();
	writing_to_a = true;
	
	// Reset status
	has_new_data.store(false);
	last_sequence.store(0);
	total_updates.store(0);
	buffer_switches.store(0);
} 