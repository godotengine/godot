/**************************************************************************/
/*  independent_audio_recorder.cpp                                       */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#include "independent_audio_recorder.h"
#include "servers/audio/audio_driver_hybrid.h"
#include "core/string/print_string.h"
#include "core/os/os.h"
#include "movie_utils.h"

IndependentAudioRecorder::IndependentAudioRecorder() {
    recording_active.store(false);
    thread_started.store(false);
    buffer_read_pos.store(0);
    buffer_write_pos.store(0);
    buffer_size = 0;
    recording_start_time = 0;
}

IndependentAudioRecorder::~IndependentAudioRecorder() {
    if (is_recording()) {
        stop_recording();
    }
}

Error IndependentAudioRecorder::initialize(HybridAudioDriver *p_audio_driver, 
                                          const String &p_audio_path,
                                          const AudioConfig &p_config) {
    if (!p_audio_driver || p_audio_path.is_empty()) {
        ERR_PRINT("IndependentAudioRecorder: Invalid audio_driver or audio_path");
        return ERR_INVALID_PARAMETER;
    }
    
    audio_driver = p_audio_driver;
    config = p_config;
    
    // Create a simple audio writer
    audio_writer.instantiate();
    
    Error open_result = audio_writer->open(p_audio_path, config.sample_rate, config.channels);
    if (open_result != OK) {
        ERR_PRINT("IndependentAudioRecorder: Failed to open audio writer");
        return open_result;
    }
    
    // Calculate buffer size
    buffer_size = config.sample_rate * config.channels * config.buffer_size_seconds;
    
    // Calculate the power for the RingBuffer
    int power = 0;
    while ((1 << power) < (int)buffer_size) {
        power++;
    }
    int actual_buffer_size = 1 << power;  // Actual buffer size
    
    // Initialize the ring buffer
    audio_ring_buffer = RingBuffer<int32_t>(power);
    buffer_size = actual_buffer_size;  // Update to actual size
    
    // Initialize temporary buffers
    temp_audio_buffer.resize(config.chunk_size * config.channels);
    chunk_buffer.resize(config.chunk_size * config.channels);
    
    // Reset statistics
    reset_statistics();
    
    if (MovieDebugUtils::is_stdout_verbose()) {
        print_line("IndependentAudioRecorder initialization completed");
        print_line(String("Audio path: ") + p_audio_path);
        print_line(String("Sample rate: ") + String::num_int64(config.sample_rate) + "Hz");
        print_line(String("Channels: ") + String::num_int64(config.channels));
        print_line(String("Chunk size: ") + String::num_int64(config.chunk_size) + " samples");
        print_line(String("Buffer size: ") + String::num_int64(buffer_size) + " samples (" + 
                  String::num_real((float)config.buffer_size_seconds) + " seconds)");
    }
    
    return OK;
}

Error IndependentAudioRecorder::start_recording() {
    if (recording_active.load()) {
        ERR_PRINT("IndependentAudioRecorder: Recording is already in progress");
        return ERR_ALREADY_IN_USE;
    }
    
    if (!audio_driver || audio_writer.is_null()) {
        ERR_PRINT("IndependentAudioRecorder: Not initialized");
        return ERR_UNCONFIGURED;
    }
    
    // Clear buffer
    audio_ring_buffer.clear();
    buffer_read_pos.store(0);
    buffer_write_pos.store(0);
    
    // Set recording status
    recording_active.store(true);
    recording_start_time = OS::get_singleton()->get_ticks_usec();
    
    // Start recording thread
    recording_thread.start(recording_thread_func, this);
    thread_started.store(true);
    
    if (MovieDebugUtils::is_stdout_verbose()) {
        print_line("IndependentAudioRecorder: Start recording");
    }
    
    return OK;
}

void IndependentAudioRecorder::stop_recording() {
    // Atomic check-and-set to prevent double cleanup
    bool expected = true;
    if (!recording_active.compare_exchange_strong(expected, false)) {
        // Already stopped or stopping
        return;
    }
    
    // Wait for the thread to finish
    if (thread_started.load()) {
        recording_thread.wait_to_finish();
        thread_started.store(false);
    }
    
    // Close the audio writer
    if (audio_writer.is_valid()) {
        audio_writer->close();
    }
    
    // Output final statistics
    if (MovieDebugUtils::is_stdout_verbose()) {
        AudioStats final_stats = get_statistics();
        print_line(String("Audio recording completed - Total chunks: ") + String::num_int64(final_stats.total_chunks_recorded));
        print_line(String("Total samples: ") + String::num_int64(final_stats.total_samples_recorded));
        print_line(String("Recording duration: ") + String::num_real(final_stats.recording_duration_us / 1000000.0) + " seconds");
        print_line(String("Buffer overruns: ") + String::num_int64(final_stats.buffer_overruns));
        print_line(String("Buffer underruns: ") + String::num_int64(final_stats.buffer_underruns));
    }
}

void IndependentAudioRecorder::recording_thread_func(void *p_userdata) {
    IndependentAudioRecorder *recorder = static_cast<IndependentAudioRecorder *>(p_userdata);
    recorder->recording_loop();
}

void IndependentAudioRecorder::recording_loop() {
    
    uint64_t next_chunk_time = recording_start_time;
    uint64_t chunk_count = 0;
    
    while (recording_active.load()) {
        uint64_t current_time = OS::get_singleton()->get_ticks_usec();
        
        if (current_time >= next_chunk_time) {
            uint64_t chunk_process_start = OS::get_singleton()->get_ticks_usec();
            
            // Process audio chunk
            bool chunk_processed = process_audio_chunk(next_chunk_time - recording_start_time);
            
            if (chunk_processed) {
                chunk_count++;
                update_statistics(chunk_process_start, config.chunk_size);
                
                // Output debug info every 1000 chunks
                if (MovieDebugUtils::is_stdout_verbose() && chunk_count % 1000 == 0) {
                    AudioStats current_stats = get_statistics();
                    print_line(String("Audio recording progress: ") + String::num_int64(chunk_count) + " chunks, " +
                              String("Buffer usage: ") + String::num_int64(current_stats.current_buffer_level) + "%");
                }
            }
            
            // Calculate next chunk time
            next_chunk_time += CHUNK_INTERVAL_USEC;
        }
        
        // Precise sleep control
        current_time = OS::get_singleton()->get_ticks_usec();
        if (next_chunk_time > current_time) {
            uint64_t sleep_time = next_chunk_time - current_time;
            if (sleep_time > 1000) { // If we need to wait more than 1ms
                OS::get_singleton()->delay_usec(sleep_time - 500); // Leave a 500 microsecond buffer
            }
        }
    }
}

bool IndependentAudioRecorder::process_audio_chunk(uint64_t current_recording_time) {
    // Read audio data from the ring buffer
    if (!read_audio_chunk(chunk_buffer, config.chunk_size * config.channels)) {
        handle_buffer_underrun();
        return false;
    }
    // Write to audio file
    Error write_result = audio_writer->write_audio_chunk(
        chunk_buffer.ptr(),
        config.chunk_size
    );
    
    if (write_result != OK) {
        ERR_PRINT("IndependentAudioRecorder: Failed to write audio chunk");
        return false;
    }
    // Update statistics
    {
        MutexLock lock(stats_mutex);
        stats.total_chunks_recorded++;
        stats.total_samples_recorded += config.chunk_size;
    }
    
    return true;
}

bool IndependentAudioRecorder::read_audio_chunk(Vector<int32_t> &output_buffer, uint32_t requested_samples) {
    uint32_t available = get_available_samples();
    
    if (available < requested_samples) {
        return false; // Not enough data
    }
    
    MutexLock lock(buffer_mutex);
    
    // Use RingBuffer's read method directly
    int samples_read = audio_ring_buffer.read(output_buffer.ptrw(), requested_samples);
    
    if (samples_read < (int)requested_samples) {
        // Fill the remaining samples with 0
        for (int i = samples_read; i < (int)requested_samples; i++) {
            output_buffer.write[i] = 0;
        }
    }
    
    return true;
}

void IndependentAudioRecorder::on_audio_output(const int32_t *p_buffer, int p_frame_count) {
    if (!recording_active.load() || !p_buffer || p_frame_count <= 0) {
        return;
    }
    
    MutexLock lock(buffer_mutex);
    
    uint32_t samples_to_write = p_frame_count * config.channels;
    
    // Check buffer space
    int available_space = audio_ring_buffer.space_left();
    if ((int)samples_to_write > available_space) {
        handle_buffer_overrun();
        return;
    }
    
    // Write data
    audio_ring_buffer.write(p_buffer, samples_to_write);
    

}

void IndependentAudioRecorder::update_statistics(uint64_t chunk_process_start_time, uint32_t samples_processed) {
    uint64_t current_time = OS::get_singleton()->get_ticks_usec();
    uint64_t process_time = current_time - chunk_process_start_time;
    
    MutexLock lock(stats_mutex);
    
    // Update recording duration
    stats.recording_duration_us = current_time - recording_start_time;
    
    // Update average processing time (using moving average)
    if (stats.avg_chunk_process_time_us == 0) {
        stats.avg_chunk_process_time_us = process_time;
    } else {
        // Use a moving average of 90% old value + 10% new value
        stats.avg_chunk_process_time_us = (stats.avg_chunk_process_time_us * 9 + process_time) / 10;
    }
    
    // Update buffer usage
    update_buffer_level();
}

void IndependentAudioRecorder::update_buffer_level() {
    uint32_t available = get_available_samples();
    stats.current_buffer_level = (available * 100) / buffer_size;
}

void IndependentAudioRecorder::handle_buffer_underrun() {
    MutexLock lock(stats_mutex);
    stats.buffer_underruns++;
    
    // Fill with silence data
    chunk_buffer.fill(0);
}

void IndependentAudioRecorder::handle_buffer_overrun() {
    MutexLock lock(stats_mutex);
    stats.buffer_overruns++;
    
    // Skip the oldest data to make space
    uint32_t samples_to_skip = config.chunk_size * config.channels;
    uint32_t read_pos = buffer_read_pos.load();
    read_pos = (read_pos + samples_to_skip) % buffer_size;
    buffer_read_pos.store(read_pos);
}

uint32_t IndependentAudioRecorder::get_available_samples() const {
    return audio_ring_buffer.data_left();
}

bool IndependentAudioRecorder::has_audio_data() const {
    return get_available_samples() >= (config.chunk_size * config.channels);
}

float IndependentAudioRecorder::get_buffer_usage_ratio() const {
    if (buffer_size == 0) {
        return 0.0f;
    }
    return (float)get_available_samples() / (float)buffer_size;
}

IndependentAudioRecorder::AudioStats IndependentAudioRecorder::get_statistics() const {
    MutexLock lock(stats_mutex);
    return stats;
}

void IndependentAudioRecorder::update_config(const AudioConfig &p_config) {
    if (is_recording()) {
        ERR_PRINT("IndependentAudioRecorder: Cannot update configuration while recording");
        return;
    }
    
    config.sample_rate = p_config.sample_rate;
    config.channels = p_config.channels;
    config.chunk_size = p_config.chunk_size;
    config.buffer_size_seconds = p_config.buffer_size_seconds;
    config.enable_audio_monitoring = p_config.enable_audio_monitoring;
    
    // Recalculate buffer size
    buffer_size = config.sample_rate * config.channels * config.buffer_size_seconds;
    
    // Calculate the power for the RingBuffer
    int power = 0;
    while ((1 << power) < (int)buffer_size) {
        power++;
    }
    int actual_buffer_size = 1 << power;  // Actual buffer size
    
    // Reinitialize the ring buffer
    audio_ring_buffer = RingBuffer<int32_t>(power);
    buffer_size = actual_buffer_size;  // Update to actual size
    
    temp_audio_buffer.resize(config.chunk_size * config.channels);
    chunk_buffer.resize(config.chunk_size * config.channels);
}

void IndependentAudioRecorder::reset_statistics() {
    MutexLock lock(stats_mutex);
    
    stats = AudioStats();
    buffer_read_pos.store(0);
    buffer_write_pos.store(0);
    recording_start_time = 0;
}

String IndependentAudioRecorder::get_debug_info() const {
    AudioStats current_stats = get_statistics();
    
    String info;
    info += "=== IndependentAudioRecorder Debug Info ===\n";
    info += String("Recording status: ") + (is_recording() ? "Running" : "Stopped") + "\n";
    info += String("Thread status: ") + (is_thread_running() ? "Running" : "Stopped") + "\n";
    info += String("Total audio chunks: ") + String::num_int64(current_stats.total_chunks_recorded) + "\n";
    info += String("Total samples: ") + String::num_int64(current_stats.total_samples_recorded) + "\n";
    info += String("Recording duration: ") + String::num_real(current_stats.recording_duration_us / 1000000.0) + " seconds\n";
    info += String("Buffer usage: ") + String::num_int64(current_stats.current_buffer_level) + "%\n";
    info += String("Available samples: ") + String::num_int64(get_available_samples()) + "\n";
    info += String("Buffer overruns: ") + String::num_int64(current_stats.buffer_overruns) + "\n";
    info += String("Buffer underruns: ") + String::num_int64(current_stats.buffer_underruns) + "\n";
    info += String("Avg chunk process time: ") + String::num_int64(current_stats.avg_chunk_process_time_us) + " microseconds\n";
    info += String("Config sample rate: ") + String::num_int64(config.sample_rate) + "Hz\n";
    info += String("Config channels: ") + String::num_int64(config.channels) + "\n";
    info += String("Config chunk size: ") + String::num_int64(config.chunk_size) + " samples\n";
    info += "==========================================";
    
    return info;
} 