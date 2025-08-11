/**************************************************************************/
/*  independent_video_recorder.h                                         */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#ifndef INDEPENDENT_VIDEO_RECORDER_H
#define INDEPENDENT_VIDEO_RECORDER_H

#include "thread_safe_frame_buffer.h"
#include "simple_video_writer.h"
#include "core/os/thread.h"
#include "core/os/mutex.h"
#include "core/os/os.h"
#include <atomic>
#include <chrono>

/**
 * Independent video recording thread.
 * Runs at a fixed 30fps, completely independent of the game's main thread.
 * Reads frame data from a double buffer to generate a recorded video with a standard timeline.
 */
class IndependentVideoRecorder {
public:
	// Recording configuration
	struct RecordingConfig {
		uint32_t target_fps; // Target recording frame rate
		uint32_t video_width; // Video width
		uint32_t video_height; // Video height
		float jpeg_quality; // JPEG quality
		bool enable_timestamp_chunks; // Enable timestamp recording
		bool enable_repeat_frame_marking; // Enable repeat frame marking

		RecordingConfig() :
			target_fps(30),
			video_width(1920),
			video_height(1080),
			jpeg_quality(0.85f),
			enable_timestamp_chunks(true),
			enable_repeat_frame_marking(true) {}
	};

	// Recording statistics
	struct RecordingStats {
		uint32_t total_recorded_frames = 0; // Total recorded frames
		uint32_t new_frames_count = 0; // Number of new frames
		uint32_t repeated_frames_count = 0; // Number of repeated frames
		uint64_t recording_duration_us = 0; // Recording duration (microseconds)
		uint64_t avg_frame_process_time_us = 0; // Average frame processing time
		uint32_t last_game_frame_sequence = 0; // Last processed game frame sequence number
	};

private:
	// Recording parameters (fixed)
	static const uint64_t FRAME_INTERVAL_USEC = 1000000 / 30; // 33333 microseconds (30fps)

	// Thread control
	Thread recording_thread;
	std::atomic<bool> recording_active{false};
	std::atomic<bool> thread_started{false};

	// Data source and output
	ThreadSafeFrameBuffer *frame_buffer = nullptr;
	Ref<SimpleVideoWriter> video_writer;

	// Recording configuration
	RecordingConfig config;

	// State tracking
	uint32_t last_game_frame_sequence = 0;
	ThreadSafeFrameBuffer::FrameData last_valid_frame;
	bool has_valid_frame = false;

	// Statistics
	RecordingStats stats;
	uint64_t recording_start_time = 0;
	uint64_t last_stats_update_time = 0;

	// Synchronization protection
	mutable Mutex stats_mutex;

	// Thread main loop
	static void recording_thread_func(void *p_userdata);
	void recording_loop();

	// Internal processing methods
	bool process_frame(uint64_t current_recording_time);
	void update_statistics(uint64_t frame_process_start_time);
	uint8_t determine_frame_flags(const ThreadSafeFrameBuffer::FrameData &frame_data);

public:
	IndependentVideoRecorder();
	~IndependentVideoRecorder();

	/**
	 * Initialize recorder
	 */
	Error initialize(ThreadSafeFrameBuffer *p_frame_buffer,
			const String &p_video_path,
			const RecordingConfig &p_config = RecordingConfig());

	/**
	 * Start recording
	 */
	Error start_recording();

	/**
	 * Stop recording
	 */
	void stop_recording();

	/**
	 * Check recording status
	 */
	bool is_recording() const { return recording_active.load(); }
	bool is_thread_running() const { return thread_started.load(); }

	/**
	 * Get recording statistics
	 */
	RecordingStats get_statistics() const;

	/**
	 * Get repeat frame ratio
	 */
	float get_repeat_frame_ratio() const;

	/**
	 * Update configuration
	 */
	void update_config(const RecordingConfig &p_config);

	/**
	 * Reset statistics
	 */
	void reset_statistics();

	/**
	 * Get recording progress information (for debugging)
	 */
	String get_debug_info() const;
};

#endif // INDEPENDENT_VIDEO_RECORDER_H 