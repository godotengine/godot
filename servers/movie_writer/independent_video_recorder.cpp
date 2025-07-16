/**************************************************************************/
/*  independent_video_recorder.cpp                                       */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#include "independent_video_recorder.h"
#include "core/string/print_string.h"
#include "core/os/os.h"
#include "movie_utils.h"

IndependentVideoRecorder::IndependentVideoRecorder() {
    recording_active.store(false);
    thread_started.store(false);
    last_game_frame_sequence = 0;
    has_valid_frame = false;
    recording_start_time = 0;
    last_stats_update_time = 0;
}

IndependentVideoRecorder::~IndependentVideoRecorder() {
    if (is_recording()) {
        stop_recording();
    }
}

Error IndependentVideoRecorder::initialize(ThreadSafeFrameBuffer *p_frame_buffer, 
                                          const String &p_video_path,
                                          const RecordingConfig &p_config) {
    if (!p_frame_buffer || p_video_path.is_empty()) {
        ERR_PRINT("IndependentVideoRecorder: Invalid frame_buffer or video_path");
        return ERR_INVALID_PARAMETER;
    }
    
    frame_buffer = p_frame_buffer;
    config = p_config;
    
    // create simple video writer
    video_writer.instantiate();
    
    Size2i movie_size(config.video_width, config.video_height);
    Error open_result = video_writer->open(p_video_path, movie_size, config.target_fps, config.jpeg_quality);
    if (open_result != OK) {
        ERR_PRINT("IndependentVideoRecorder: Failed to open video writer");
        return open_result;
    }
    
    // reset statistics
    reset_statistics();
    
    if (MovieDebugUtils::is_stdout_verbose()) {
        print_line(String("IndependentVideoRecorder initialized"));
        print_line(String("Video path: ") + p_video_path);
        print_line(String("Resolution: ") + String::num_int64(config.video_width) + "x" + String::num_int64(config.video_height));
        print_line(String("Target FPS: ") + String::num_int64(config.target_fps));
        print_line(String("JPEG quality: ") + String::num_real(config.jpeg_quality));
    }
    
    return OK;
}

Error IndependentVideoRecorder::start_recording() {
    if (recording_active.load()) {
        ERR_PRINT("IndependentVideoRecorder: Recording already in progress");
        return ERR_ALREADY_IN_USE;
    }
    
    if (!frame_buffer || video_writer.is_null()) {
        ERR_PRINT("IndependentVideoRecorder: Not initialized");
        return ERR_UNCONFIGURED;
    }
    
    // set recording status
    recording_active.store(true);
    recording_start_time = OS::get_singleton()->get_ticks_usec();
    
    // start recording thread
    recording_thread.start(recording_thread_func, this);
    thread_started.store(true);
    
    if (MovieDebugUtils::is_stdout_verbose()) {
        print_line("IndependentVideoRecorder: Start recording");
    }
    
    return OK;
}

void IndependentVideoRecorder::stop_recording() {
    // Atomic check-and-set to prevent double cleanup
    bool expected = true;
    if (!recording_active.compare_exchange_strong(expected, false)) {
        // Already stopped or stopping
        return;
    }
    
    
    // wait for thread to finish
    if (thread_started.load()) {
        recording_thread.wait_to_finish();
        thread_started.store(false);
    }
    
    // close video writer
    if (video_writer.is_valid()) {
        video_writer->close();
    }
    
    // output final statistics
    if (MovieDebugUtils::is_stdout_verbose()) {
        RecordingStats final_stats = get_statistics();
        print_line(String("Recording completed - Total frames: ") + String::num_int64(final_stats.total_recorded_frames));
        print_line(String("New frames: ") + String::num_int64(final_stats.new_frames_count));
        print_line(String("Repeated frames: ") + String::num_int64(final_stats.repeated_frames_count));
        print_line(String("Repeated frame ratio: ") + String::num_real(get_repeat_frame_ratio() * 100.0f) + "%");
        print_line(String("Recording duration: ") + String::num_real(final_stats.recording_duration_us / 1000000.0) + " seconds");
    }
}

void IndependentVideoRecorder::recording_thread_func(void *p_userdata) {
    IndependentVideoRecorder *recorder = static_cast<IndependentVideoRecorder *>(p_userdata);
    recorder->recording_loop();
}

void IndependentVideoRecorder::recording_loop() {
    
    uint64_t next_record_time = recording_start_time;
    uint64_t frame_count = 0;
    
    while (recording_active.load()) {
        uint64_t current_time = OS::get_singleton()->get_ticks_usec();
        
        if (current_time >= next_record_time) {
            uint64_t frame_process_start = OS::get_singleton()->get_ticks_usec();
            
            // Process frame
            bool frame_processed = process_frame(next_record_time - recording_start_time);
            
            if (frame_processed) {
                frame_count++;
                update_statistics(frame_process_start);
                
                // Output debug information every 30 frames
                if (MovieDebugUtils::is_stdout_verbose() && frame_count % 30 == 0) {
                    print_line(String("Recording progress: ") + String::num_int64(frame_count) + " frames, " +
                              String("Repeated frame ratio: ") + String::num_real(get_repeat_frame_ratio() * 100.0f) + "%");
                }
            }
            
            // Calculate next frame time
            next_record_time += FRAME_INTERVAL_USEC;
        }
        
        // Precise sleep control
        current_time = OS::get_singleton()->get_ticks_usec();
        if (next_record_time > current_time) {
            uint64_t sleep_time = next_record_time - current_time;
            if (sleep_time > 1000) { // If waiting for more than 1ms
                OS::get_singleton()->delay_usec(sleep_time - 500); // Leave a 500 microsecond buffer
            }
        }
    }
}

bool IndependentVideoRecorder::process_frame(uint64_t current_recording_time) {
    // Get current frame data
    ThreadSafeFrameBuffer::FrameData frame_data = frame_buffer->get_current_frame();
    
    ThreadSafeFrameBuffer::FrameData frame_to_write;
    
    if (frame_data.frame_sequence == last_game_frame_sequence || frame_data.image.is_null()) {
        // No new frame from the game or invalid frame, use a repeated frame
        if (!has_valid_frame) {
            // No valid frame yet, skip
            return false;
        }
        frame_to_write = last_valid_frame;
        
        // Update timestamp for repeated frames but keep the game timestamp unchanged
        frame_to_write.is_new_frame = false;
        
        MutexLock lock(stats_mutex);
        stats.repeated_frames_count++;
    } else {
        // New frame, update records
        frame_to_write = frame_data;
        frame_to_write.is_new_frame = true;
        
        last_valid_frame = frame_data;
        last_game_frame_sequence = frame_data.frame_sequence;
        has_valid_frame = true;
        
        MutexLock lock(stats_mutex);
        stats.new_frames_count++;
        stats.last_game_frame_sequence = frame_data.frame_sequence;
    }
    
    // Write to video file
    Error write_result = video_writer->write_frame(frame_to_write.image);
    
    if (write_result != OK) {
        ERR_PRINT("IndependentVideoRecorder: write video frame failed");
        return false;
    }
    
    // Update statistics
    {
        MutexLock lock(stats_mutex);
        stats.total_recorded_frames++;
    }
    
    return true;
}

void IndependentVideoRecorder::update_statistics(uint64_t frame_process_start_time) {
    uint64_t current_time = OS::get_singleton()->get_ticks_usec();
    uint64_t process_time = current_time - frame_process_start_time;
    
    MutexLock lock(stats_mutex);
    
    // Update recording duration
    stats.recording_duration_us = current_time - recording_start_time;
    
    // Update average processing time (using a moving average)
    if (stats.avg_frame_process_time_us == 0) {
        stats.avg_frame_process_time_us = process_time;
    } else {
        // Moving average with 90% old value + 10% new value
        stats.avg_frame_process_time_us = (stats.avg_frame_process_time_us * 9 + process_time) / 10;
    }
}

// Simplified version, no longer uses complex frame flags
uint8_t IndependentVideoRecorder::determine_frame_flags(const ThreadSafeFrameBuffer::FrameData &frame_data) {
    // Simplified version, no longer uses complex frame flags
    return 0;
}

IndependentVideoRecorder::RecordingStats IndependentVideoRecorder::get_statistics() const {
    MutexLock lock(stats_mutex);
    return stats;
}

float IndependentVideoRecorder::get_repeat_frame_ratio() const {
    MutexLock lock(stats_mutex);
    
    if (stats.total_recorded_frames == 0) {
        return 0.0f;
    }
    
    return (float)stats.repeated_frames_count / (float)stats.total_recorded_frames;
}

void IndependentVideoRecorder::update_config(const RecordingConfig &p_config) {
    config.target_fps = p_config.target_fps;
    config.video_width = p_config.video_width;
    config.video_height = p_config.video_height;
    config.jpeg_quality = p_config.jpeg_quality;
    config.enable_timestamp_chunks = p_config.enable_timestamp_chunks;
    config.enable_repeat_frame_marking = p_config.enable_repeat_frame_marking;
    
    if (video_writer.is_valid()) {
        video_writer->set_quality(config.jpeg_quality);
    }
}

void IndependentVideoRecorder::reset_statistics() {
    MutexLock lock(stats_mutex);
    
    stats = RecordingStats();
    last_game_frame_sequence = 0;
    has_valid_frame = false;
    recording_start_time = 0;
    last_stats_update_time = 0;
}

String IndependentVideoRecorder::get_debug_info() const {
    RecordingStats current_stats = get_statistics();
    
    String info;
    info += "=== IndependentVideoRecorder Debug Info ===\n";
    info += String("Recording status: ") + (is_recording() ? "Running" : "Stopped") + "\n";
    info += String("Thread status: ") + (is_thread_running() ? "Running" : "Stopped") + "\n";
    info += String("Total recorded frames: ") + String::num_int64(current_stats.total_recorded_frames) + "\n";
    info += String("New frames: ") + String::num_int64(current_stats.new_frames_count) + "\n";
    info += String("Repeated frames: ") + String::num_int64(current_stats.repeated_frames_count) + "\n";
    info += String("Repeated frame ratio: ") + String::num_real(get_repeat_frame_ratio() * 100.0f) + "%\n";
    info += String("Recording duration: ") + String::num_real(current_stats.recording_duration_us / 1000000.0) + " seconds\n";
    info += String("Avg frame process time: ") + String::num_int64(current_stats.avg_frame_process_time_us) + " microseconds\n";
    info += String("Last game frame sequence: ") + String::num_int64(current_stats.last_game_frame_sequence) + "\n";
    info += String("Config FPS: ") + String::num_int64(config.target_fps) + "\n";
    info += String("Config resolution: ") + String::num_int64(config.video_width) + "x" + String::num_int64(config.video_height) + "\n";
    info += "==========================================";
    
    return info;
} 