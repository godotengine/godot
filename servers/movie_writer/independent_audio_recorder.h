/**************************************************************************/
/*  independent_audio_recorder.h                                         */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#ifndef INDEPENDENT_AUDIO_RECORDER_H
#define INDEPENDENT_AUDIO_RECORDER_H

#include "simple_audio_writer.h"
#include "core/os/thread.h"
#include "core/os/mutex.h"
#include "core/os/os.h"
#include "core/templates/ring_buffer.h"
#include <atomic>

// Forward declaration
class HybridAudioDriver;

/**
 * Independent audio recording thread
 * Runs continuously at a fixed sample rate, completely independent of the main game thread
 * Gets output data from the audio driver to generate a continuous audio stream
 */
class IndependentAudioRecorder {
public:
	// Audio recording configuration
	struct AudioConfig {
		uint32_t sample_rate; // Sample rate
		uint32_t channels; // Number of channels
		uint32_t chunk_size; // Number of samples to process at a time (10ms at 48kHz)
		uint32_t buffer_size_seconds; // Ring buffer size in seconds
		bool enable_audio_monitoring; // Enable audio monitoring

		AudioConfig() :
			sample_rate(48000),
			channels(2),
			chunk_size(480),
			buffer_size_seconds(2),
			enable_audio_monitoring(false) {}
	};

	// Audio statistics
	struct AudioStats {
		uint64_t total_chunks_recorded = 0; // Total number of recorded audio chunks
		uint64_t total_samples_recorded = 0; // Total number of recorded samples
		uint64_t buffer_overruns = 0; // Number of buffer overflows
		uint64_t buffer_underruns = 0; // Number of buffer underflows
		uint64_t recording_duration_us = 0; // Recording duration (microseconds)
		uint32_t current_buffer_level = 0; // Current buffer usage (0-100)
		uint64_t avg_chunk_process_time_us = 0; // Average chunk processing time
	};

private:
	// Recording parameters (fixed)
	static const uint64_t CHUNK_INTERVAL_USEC = 10000; // 10ms interval

	// Thread control
	Thread recording_thread;
	std::atomic<bool> recording_active{false};
	std::atomic<bool> thread_started{false};

	// Audio configuration
	AudioConfig config;

	// Data source and output
	HybridAudioDriver *audio_driver = nullptr;
	Ref<SimpleAudioWriter> audio_writer;

	// Audio ring buffer
	RingBuffer<int32_t> audio_ring_buffer;
	mutable Mutex buffer_mutex;
	std::atomic<uint32_t> buffer_read_pos{0};
	std::atomic<uint32_t> buffer_write_pos{0};
	uint32_t buffer_size = 0;

	// Temporary buffers
	Vector<int32_t> temp_audio_buffer;
	Vector<int32_t> chunk_buffer;

	// Statistics
	AudioStats stats;
	uint64_t recording_start_time = 0;
	mutable Mutex stats_mutex;

	// Thread main loop
	static void recording_thread_func(void *p_userdata);
	void recording_loop();

	// Internal processing methods
	bool process_audio_chunk(uint64_t current_recording_time);
	void update_statistics(uint64_t chunk_process_start_time, uint32_t samples_processed);
	void update_buffer_level();

	// Buffer management
	bool read_audio_chunk(Vector<int32_t> &output_buffer, uint32_t requested_samples);
	void handle_buffer_underrun();
	void handle_buffer_overrun();

public:
	IndependentAudioRecorder();
	~IndependentAudioRecorder();

	/**
	 * Initialize recorder
	 */
	Error initialize(HybridAudioDriver *p_audio_driver,
			const String &p_audio_path,
			const AudioConfig &p_config = AudioConfig());

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
	 * Mark recording as inactive (for external cleanup)
	 */
	void mark_inactive() { recording_active.store(false); }

	/**
	 * Audio data input interface (called by HybridAudioDriver)
	 */
	void on_audio_output(const int32_t *p_buffer, int p_frame_count);

	/**
	 * Get audio statistics
	 */
	AudioStats get_statistics() const;

	/**
	 * Get buffer status
	 */
	uint32_t get_available_samples() const;
	bool has_audio_data() const;
	float get_buffer_usage_ratio() const;

	/**
	 * Update configuration
	 */
	void update_config(const AudioConfig &p_config);

	/**
	 * Reset statistics
	 */
	void reset_statistics();

	/**
	 * Get debug information
	 */
	String get_debug_info() const;

	/**
	 * Get audio configuration
	 */
	const AudioConfig &get_config() const { return config; }
};

#endif // INDEPENDENT_AUDIO_RECORDER_H 