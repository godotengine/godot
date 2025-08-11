/**************************************************************************/
/*  thread_safe_frame_buffer.h                                           */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#ifndef THREAD_SAFE_FRAME_BUFFER_H
#define THREAD_SAFE_FRAME_BUFFER_H

#include "core/io/image.h"
#include "core/os/mutex.h"
#include "core/os/os.h"
#include <atomic>

/**
 * Thread-safe double-buffered frame data manager.
 * Implements safe data exchange between the game's main thread and the recording thread.
 */
class ThreadSafeFrameBuffer {
public:
	struct FrameData {
		Ref<Image> image;           // Image data
		uint64_t game_timestamp;    // Game timestamp (microseconds)
		uint32_t frame_sequence;    // Game frame sequence number
		bool is_new_frame = false;  // Whether it is a new frame
		
		FrameData() : game_timestamp(0), frame_sequence(0), is_new_frame(false) {}
		
		FrameData(const FrameData &other) 
			: image(other.image)
			, game_timestamp(other.game_timestamp)
			, frame_sequence(other.frame_sequence)
			, is_new_frame(other.is_new_frame) {}
		
		FrameData &operator=(const FrameData &other) {
			if (this != &other) {
				image = other.image;
				game_timestamp = other.game_timestamp;
				frame_sequence = other.frame_sequence;
				is_new_frame = other.is_new_frame;
			}
			return *this;
		}
	};

private:
	// Double buffer
	FrameData buffer_a;
	FrameData buffer_b;
	bool writing_to_a = true;           // Game thread writing flag
	
	// Synchronization control
	mutable Mutex buffer_mutex;         // Buffer switching protection
	std::atomic<bool> has_new_data{false};    // New data flag
	std::atomic<uint32_t> last_sequence{0};   // Last processed sequence number
	
	// Statistics
	std::atomic<uint64_t> total_updates{0};
	std::atomic<uint64_t> buffer_switches{0};

public:
	ThreadSafeFrameBuffer();
	~ThreadSafeFrameBuffer();
	
	/**
	 * Called by the game thread: update the frame.
	 * @param new_frame New image frame
	 * @param timestamp Game timestamp
	 * @param sequence Game frame sequence number
	 */
	void update_frame(const Ref<Image> &new_frame, uint64_t timestamp, uint32_t sequence);
	
	/**
	 * Called by the recording thread: get the stable frame.
	 * @return Current stable frame data
	 */
	FrameData get_current_frame() const;
	
	/**
	 * Check if there is new data.
	 */
	bool has_new_frame() const { return has_new_data.load(); }
	
	/**
	 * Get the last processed sequence number.
	 */
	uint32_t get_last_sequence() const { return last_sequence.load(); }
	
	/**
	 * Get statistics.
	 */
	uint64_t get_total_updates() const { return total_updates.load(); }
	uint64_t get_buffer_switches() const { return buffer_switches.load(); }
	
	/**
	 * Reset the buffer.
	 */
	void reset();
};

#endif // THREAD_SAFE_FRAME_BUFFER_H 