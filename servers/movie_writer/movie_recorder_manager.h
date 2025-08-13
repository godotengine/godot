#ifndef MOVIE_RECORDER_MANAGER_H
#define MOVIE_RECORDER_MANAGER_H

#include "movie_writer.h"
#include "core/string/ustring.h"
#include "core/math/vector2i.h"

class MovieRecorderManager {
public:
    // Recording configuration structure
    struct RecordingConfig {
        String output_path;
        uint32_t video_fps = 30;
        uint32_t video_width = 0;  // 0 = use current window size
        uint32_t video_height = 0;
        float video_quality = 0.85f;
        uint32_t audio_sample_rate = 48000;
        uint32_t audio_channels = 2;
        bool enable_audio = true;
        bool realtime_mode = false;
        
        RecordingConfig() = default;
        RecordingConfig(const String &path) : output_path(path) {}
    };

    // Static interfaces called by main.cpp
    static void onInit();        // Save initialization parameters, check command line recording requirements
    static void onStart();       // If recording needed, create and start instance
    static void onUpdate();      // If active instance exists, call add_frame()
    static void onCleanup();     // Cleanup instance and resources
    
    // Parameter setting interface (called by main.cpp)
    static void set_fixed_fps(int fps);

    // Runtime recording control API
    static Error start_recording(const RecordingConfig &config);
    static Error stop_recording();
    static Error pause_recording();
    static Error resume_recording();

    // Status queries
    static bool is_recording();
    static bool is_initialized();
    static bool has_active_instance();
    static String get_current_output_path();
    static float get_recording_duration();

private:
    // Instance state
    enum InstanceState {
        NONE,           // No instance
        CREATED,        // Instance created, but begin() not called
        STARTED,        // Instance called begin(), currently recording
        PAUSED,         // Recording paused
        ENDED           // Instance called end(), waiting for destruction
    };

    // Initialization parameter structure
    struct InitParams {
        Size2i window_size;
        uint32_t fixed_fps = 60;
        bool initialized = false;
        
        void capture_current_params();
    };

    // Static member variables
    static MovieWriter* instance;
    static InstanceState state;
    static bool needs_recording;         // Mark if recording is needed (from command line or runtime)
    static bool cmdline_recording;       // Mark if recording is started from command line
    static InitParams saved_params;      // Parameters saved during onInit
    static RecordingConfig current_config; // Current recording configuration
    static uint64_t recording_start_time; // Recording start time

    // Internal methods
    static Error create_instance();
    static Error start_instance();
    static void destroy_instance();
    static void update_config_from_cmdline();
    static void log_state_change(InstanceState old_state, InstanceState new_state);
    static const char* state_to_string(InstanceState state);
};

#endif // MOVIE_RECORDER_MANAGER_H