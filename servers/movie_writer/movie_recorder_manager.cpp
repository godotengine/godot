#include "movie_recorder_manager.h"
#include "core/config/engine.h"
#include "core/config/project_settings.h"
#include "core/string/print_string.h"
#include "core/os/time.h"
#include "core/os/os.h"
#include "servers/display_server.h"
#include "movie_utils.h"

// Static member variable definitions
MovieWriter* MovieRecorderManager::instance = nullptr;
MovieRecorderManager::InstanceState MovieRecorderManager::state = NONE;
bool MovieRecorderManager::needs_recording = false;
bool MovieRecorderManager::cmdline_recording = false;
MovieRecorderManager::InitParams MovieRecorderManager::saved_params;
MovieRecorderManager::RecordingConfig MovieRecorderManager::current_config;
uint64_t MovieRecorderManager::recording_start_time = 0;

void MovieRecorderManager::InitParams::capture_current_params() {
    if (DisplayServer::get_singleton()) {
        window_size = DisplayServer::get_singleton()->window_get_size();
    }
    // fixed_fps will be set through set_fixed_fps()
    initialized = true;
}

void MovieRecorderManager::set_fixed_fps(int fps) {
    saved_params.fixed_fps = fps > 0 ? (uint32_t)fps : 60;  // Default 60fps
    current_config.video_fps = saved_params.fixed_fps;
}

void MovieRecorderManager::onInit() {
    if (MovieDebugUtils::is_stdout_verbose()) {
        print_line("MovieRecorderManager::onInit() called");
    }

    // Save current system parameters
    saved_params.capture_current_params();
    
    // Check command line recording requirements
    String movie_path = Engine::get_singleton()->get_write_movie_path();
    if (!movie_path.is_empty()) {
        cmdline_recording = true;
        needs_recording = true;
        current_config.output_path = movie_path;
        
        update_config_from_cmdline();
        
        if (MovieDebugUtils::is_stdout_verbose()) {
            print_line("MovieRecorderManager: Command line recording requested - " + movie_path);
        }
    }

    if (MovieDebugUtils::is_stdout_verbose()) {
        print_line(String("MovieRecorderManager initialized - needs_recording: ") + 
                   (needs_recording ? "true" : "false"));
    }
}

void MovieRecorderManager::onStart() {
    if (!needs_recording || state == STARTED) {
        return;
    }

    if (MovieDebugUtils::is_stdout_verbose()) {
        print_line("MovieRecorderManager::onStart() called");
    }

    InstanceState old_state = state;

    // If no instance exists, create instance
    if (state == NONE) {
        Error err = create_instance();
        if (err != OK) {
            ERR_PRINT("MovieRecorderManager: Failed to create recorder instance");
            needs_recording = false;
            return;
        }
    }

    // If instance is created but not started, begin recording
    if (state == CREATED) {
        Error err = start_instance();
        if (err != OK) {
            ERR_PRINT("MovieRecorderManager: Failed to start recording");
            destroy_instance();
            needs_recording = false;
            return;
        }
    }

    log_state_change(old_state, state);
}

void MovieRecorderManager::onUpdate() {
    // Skip if no active instance or recording not started
    if (state != STARTED || !instance) {
        return;
    }

    // Call add_frame on the instance
    instance->add_frame();
}

void MovieRecorderManager::onCleanup() {
    if (state == NONE || !instance) {
        return;
    }

    if (MovieDebugUtils::is_stdout_verbose()) {
        print_line("MovieRecorderManager::onCleanup() called");
    }

    InstanceState old_state = state;

    // If currently recording, end recording first
    if (state == STARTED || state == PAUSED) {
        instance->end();
        state = ENDED;
        
        if (MovieDebugUtils::is_stdout_verbose()) {
            print_line("MovieRecorderManager: Recording ended");
        }
    }

    // Destroy instance
    if (state == ENDED) {
        destroy_instance();
    }

    log_state_change(old_state, state);
}

Error MovieRecorderManager::start_recording(const RecordingConfig &config) {
    if (state == STARTED) {
        ERR_PRINT("MovieRecorderManager: Recording already in progress");
        return ERR_ALREADY_IN_USE;
    }

    if (config.output_path.is_empty()) {
        ERR_PRINT("MovieRecorderManager: Output path is required");
        return ERR_INVALID_PARAMETER;
    }

    // Stop current recording (if any)
    if (state == STARTED || state == PAUSED) {
        stop_recording();
    }

    // Save new configuration
    current_config = config;
    needs_recording = true;
    cmdline_recording = false; // This is runtime recording, not command line recording

    if (MovieDebugUtils::is_stdout_verbose()) {
        print_line("MovieRecorderManager: Runtime recording requested - " + config.output_path);
    }

    // Immediately try to start recording
#ifdef WEB_ENABLED
    // web platform only update status
    state = STARTED;
    recording_start_time = OS::get_singleton()->get_ticks_msec();
    if (MovieDebugUtils::is_stdout_verbose()) {
        print_line("MovieRecorderManager: Web recording started - " + config.output_path);
    }
    return OK;
#else
    onStart();
    return state == STARTED ? OK : ERR_CANT_CREATE;
#endif
}

Error MovieRecorderManager::stop_recording() {
    if (state != STARTED && state != PAUSED) {
        return ERR_INVALID_PARAMETER;
    }

    if (MovieDebugUtils::is_stdout_verbose()) {
        print_line("MovieRecorderManager: Stopping recording");
    }

    needs_recording = false;

#ifdef WEB_ENABLED
    // Web Platform, only update state    
    InstanceState old_state = state;
    state = ENDED;
    uint64_t duration = OS::get_singleton()->get_ticks_msec() - recording_start_time;
    if (MovieDebugUtils::is_stdout_verbose()) {
        print_line("MovieRecorderManager: Web recording stopped, duration: " + String::num_int64(duration) + "ms");
    }
    
    log_state_change(old_state, state);
    return OK;
#else
    if (instance) {
        instance->end();
        state = ENDED;
    }
    return OK;
#endif
}


bool MovieRecorderManager::is_recording() {
    return state == STARTED;
}

bool MovieRecorderManager::is_initialized() {
    return saved_params.initialized;
}

bool MovieRecorderManager::has_active_instance() {
    return instance != nullptr && state != NONE;
}

String MovieRecorderManager::get_current_output_path() {
    return current_config.output_path;
}

float MovieRecorderManager::get_recording_duration() {
    if (state != STARTED || recording_start_time == 0) {
        return 0.0f;
    }

    uint64_t current_time = Time::get_singleton()->get_ticks_usec();
    return (current_time - recording_start_time) / 1000000.0f;
}

Error MovieRecorderManager::create_instance() {
    if (instance != nullptr) {
        ERR_PRINT("MovieRecorderManager: Instance already exists");
        return ERR_ALREADY_EXISTS;
    }

    // Find suitable MovieWriter
    instance = MovieWriter::find_writer_for_file(current_config.output_path);
    if (!instance) {
        ERR_PRINT("MovieRecorderManager: Can't find movie writer for file type - " + current_config.output_path);
        return ERR_FILE_UNRECOGNIZED;
    }

    // Configure recording mode
    if (current_config.realtime_mode) {
        instance->set_realtime_mode(true);
    }

    state = CREATED;

    if (MovieDebugUtils::is_stdout_verbose()) {
        print_line("MovieRecorderManager: Instance created for " + current_config.output_path);
    }

    return OK;
}

Error MovieRecorderManager::start_instance() {
    if (!instance || state != CREATED) {
        ERR_PRINT("MovieRecorderManager: Invalid state for starting instance");
        return ERR_UNCONFIGURED;
    }

    // Determine window size
    Size2i movie_size;
    if (current_config.video_width > 0 && current_config.video_height > 0) {
        movie_size = Size2i(current_config.video_width, current_config.video_height);
    } else {
        movie_size = saved_params.window_size;
        if (DisplayServer::get_singleton()) {
            movie_size = DisplayServer::get_singleton()->window_get_size();
        }
    }

    // Start recording
    instance->begin(movie_size, current_config.video_fps, current_config.output_path);
    
    state = STARTED;
    recording_start_time = Time::get_singleton()->get_ticks_usec();

    if (MovieDebugUtils::is_stdout_verbose()) {
        print_line(String("MovieRecorderManager: Recording started - ") + 
                   String::num_int64(movie_size.width) + "x" + String::num_int64(movie_size.height) + 
                   " @ " + String::num_int64(current_config.video_fps) + "fps");
    }

    return OK;
}

void MovieRecorderManager::destroy_instance() {
    if (!instance) {
        return;
    }

    // MovieWriter instance returned by MovieWriter::find_writer_for_file is a statically registered instance
    // No need to manually delete, just clear reference
    instance = nullptr;
    state = NONE;
    recording_start_time = 0;

    if (MovieDebugUtils::is_stdout_verbose()) {
        print_line("MovieRecorderManager: Instance destroyed");
    }
}

void MovieRecorderManager::update_config_from_cmdline() {
    // Check real-time recording mode
    if (ProjectSettings::get_singleton()->has_setting("movie_writer/realtime_mode")) {
        current_config.realtime_mode = (bool)ProjectSettings::get_singleton()->get_setting("movie_writer/realtime_mode");
    }

    // Other configurations can be read from ProjectSettings
    if (ProjectSettings::get_singleton()->has_setting("editor/movie_writer/fps")) {
        current_config.video_fps = (uint32_t)ProjectSettings::get_singleton()->get_setting("editor/movie_writer/fps");
    }

    // Update fps in saved parameters
    if (ProjectSettings::get_singleton()->has_setting("editor/movie_writer/fps")) {
        saved_params.fixed_fps = (uint32_t)ProjectSettings::get_singleton()->get_setting("editor/movie_writer/fps");
    }
}

void MovieRecorderManager::log_state_change(InstanceState old_state, InstanceState new_state) {
    if (old_state != new_state && MovieDebugUtils::is_stdout_verbose()) {
        print_line(String("MovieRecorderManager: State changed from ") + 
                   state_to_string(old_state) + " to " + state_to_string(new_state));
    }
}

const char* MovieRecorderManager::state_to_string(InstanceState state) {
    switch (state) {
        case NONE: return "NONE";
        case CREATED: return "CREATED";
        case STARTED: return "STARTED";
        case PAUSED: return "PAUSED";
        case ENDED: return "ENDED";
        default: return "UNKNOWN";
    }
}