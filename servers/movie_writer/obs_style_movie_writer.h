/**************************************************************************/
/*  obs_style_movie_writer.h                                             */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#ifndef OBS_STYLE_MOVIE_WRITER_H
#define OBS_STYLE_MOVIE_WRITER_H

#include "movie_writer.h"
#include "thread_safe_frame_buffer.h"
#include "independent_video_recorder.h"
#include "independent_audio_recorder.h"
#include "post_merge_processor.h"
#include "servers/audio/audio_driver_hybrid.h"
#include "core/os/os.h"
#include "core/os/thread.h"
#include "core/os/mutex.h"
#include <atomic>

/**
 * OBS-style independent thread recorder.
 * Implements recording effects similar to OBS:
 * - Fixed timeline recording (e.g., 30fps)
 * - Game stutter is realistically reflected in the video
 * - Audio is recorded continuously, unaffected by game frame rate
 * - Complete timing information is saved
 */
class ObsStyleMovieWriter : public MovieWriter {
    GDCLASS(ObsStyleMovieWriter, MovieWriter);

public:
    // OBS recording configuration
    struct ObsRecordingConfig {
        // Video parameters
        uint32_t video_fps = 30;            // Fixed recording frame rate
        uint32_t video_width = 1920;        // Video width
        uint32_t video_height = 1080;       // Video height
        float jpeg_quality = 0.85f;         // JPEG quality
        
        // Audio parameters
        uint32_t audio_sample_rate = 48000; // Audio sample rate
        uint32_t audio_channels = 2;        // Audio channels
        uint32_t audio_chunk_ms = 10;       // Audio chunk duration (milliseconds)
        uint32_t audio_buffer_seconds = 2;  // Audio buffer duration (seconds)
        
        // Feature switches
        bool enable_timestamp_chunks = true;      // Enable timestamp recording
        bool enable_repeat_frame_marking = true;  // Enable repeat frame marking
        bool enable_audio_monitoring = false;     // Enable audio monitoring
        bool enable_debug_output = false;          // Enable debug output
        
        // Post-merge configuration
        bool enable_post_merge = true;            // Enable post-recording file merge
        bool keep_intermediate_files = false;    // Keep separate video/audio files after merge
        String ffmpeg_path = "ffmpeg";           // FFmpeg executable path
        
        // Performance parameters
        uint32_t max_frame_buffer_size = 4;       // Maximum frame buffer size
    };
    
    // Recording state
    enum RecordingState {
        STATE_UNINITIALIZED,    // Uninitialized
        STATE_INITIALIZED,      // Initialized
        STATE_RECORDING,        // Recording
        STATE_STOPPING,         // Stopping
        STATE_ERROR            // Error state
    };

private:
    // Recording components
    ThreadSafeFrameBuffer *frame_buffer = nullptr;
    IndependentVideoRecorder *video_recorder = nullptr;
    IndependentAudioRecorder *audio_recorder = nullptr;
    HybridAudioDriver *hybrid_audio_driver = nullptr;
    PostMergeProcessor *post_merge_processor = nullptr;
    
    // Recording configuration
    ObsRecordingConfig obs_config;
    
    // Recording state
    RecordingState current_state = STATE_UNINITIALIZED;
    String output_file_path;
    uint32_t game_frame_sequence = 0;
    uint64_t recording_start_time = 0;
    
    // Audio driver management
    AudioDriver *original_audio_driver = nullptr;
    bool audio_driver_replaced = false;
    
    // Performance monitoring
    uint64_t last_add_frame_time = 0;
    uint32_t frames_added_count = 0;
    
    
    // Internal methods
    Error setup_components();
    void cleanup_components(IndependentAudioRecorder* temp_audio_recorder = nullptr);
    Error setup_audio_capture();
    void restore_audio_driver();
    void update_recording_state(RecordingState new_state);
    
    // Post-merge processing
    void perform_post_merge();
    
    
    // Config validation
    Error validate_config() const;
    void apply_config_to_components();

protected:
    // MovieWriter interface implementation
    virtual uint32_t get_audio_mix_rate() const override;
    virtual AudioServer::SpeakerMode get_audio_speaker_mode() const override;
    virtual Error write_begin(const Size2i &p_movie_size, uint32_t p_fps, const String &p_base_path) override;
    virtual Error write_frame(const Ref<Image> &p_image, const int32_t *p_audio_data) override;
    virtual void write_end() override;

public:
    ObsStyleMovieWriter();
    ~ObsStyleMovieWriter();
    
    // MovieWriter interface
    virtual bool handles_file(const String &p_path) const override;
    virtual void get_supported_extensions(List<String> *r_extensions) const override;
    
    /**
     * Configuration Management
     */
    void set_recording_config(const ObsRecordingConfig &p_config);
    const ObsRecordingConfig &get_recording_config() const { return obs_config; }
    
    /**
     * Status Query
     */
    RecordingState get_recording_state() const { return current_state; }
    String get_state_name() const;
    bool is_recording_active() const { return current_state == STATE_RECORDING; }
    
    /**
     * Statistics
     */
    struct CombinedStats {
        IndependentVideoRecorder::RecordingStats video_stats;
        IndependentAudioRecorder::AudioStats audio_stats;
        uint32_t game_frames_added = 0;
        uint64_t total_recording_duration_us = 0;
        float overall_repeat_frame_ratio = 0.0f;
    };
    
    CombinedStats get_combined_statistics() const;
    
    /**
     * Debugging features
     */
    String get_comprehensive_debug_info() const;
    void print_recording_summary() const;
    
    /**
     * Advanced controls
     */
    Error pause_recording();
    Error resume_recording();
    bool is_paused() const;
    
    /**
     * Configuration presets
     */
    static ObsRecordingConfig get_high_quality_config();
    static ObsRecordingConfig get_standard_config();
    static ObsRecordingConfig get_performance_config();
    
    /**
     * File format support
     */
    static bool is_supported_format(const String &p_extension);
};

#endif // OBS_STYLE_MOVIE_WRITER_H 