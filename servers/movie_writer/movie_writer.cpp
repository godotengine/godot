/**************************************************************************/
/*  movie_writer.cpp                                                      */
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

#include "movie_writer.h"
#include "core/config/project_settings.h"
#include "core/io/dir_access.h"
#include "core/os/os.h"
#include "core/os/time.h"
#include "servers/audio/audio_driver_dummy.h"
#include "servers/audio/audio_driver_hybrid.h"
#include "servers/display_server.h"
#include "servers/rendering_server.h"

#ifdef WEB_ENABLED
#include "platform/web/godot_audio.h"
#include "core/io/file_access.h"
#endif

#include "movie_utils.h"

MovieWriter *MovieWriter::writers[MovieWriter::MAX_WRITERS];
uint32_t MovieWriter::writer_count = 0;
HybridAudioDriver *MovieWriter::hybrid_driver = nullptr;

void MovieWriter::add_writer(MovieWriter *p_writer) {
	ERR_FAIL_COND(writer_count == MAX_WRITERS);
	writers[writer_count++] = p_writer;
}

MovieWriter *MovieWriter::find_writer_for_file(const String &p_file) {
	for (int32_t i = writer_count - 1; i >= 0; i--) { // More recent last, to have override ability.
		if (writers[i]->handles_file(p_file)) {
			return writers[i];
		}
	}
	return nullptr;
}

uint32_t MovieWriter::get_audio_mix_rate() const {
	uint32_t ret = 48000;
	GDVIRTUAL_CALL(_get_audio_mix_rate, ret);
	return ret;
}
AudioServer::SpeakerMode MovieWriter::get_audio_speaker_mode() const {
	AudioServer::SpeakerMode ret = AudioServer::SPEAKER_MODE_STEREO;
	GDVIRTUAL_CALL(_get_audio_speaker_mode, ret);
	return ret;
}

Error MovieWriter::write_begin(const Size2i &p_movie_size, uint32_t p_fps, const String &p_base_path) {
	Error ret = ERR_UNCONFIGURED;
	GDVIRTUAL_CALL(_write_begin, p_movie_size, p_fps, p_base_path, ret);
	return ret;
}

Error MovieWriter::write_frame(const Ref<Image> &p_image, const int32_t *p_audio_data) {
	Error ret = ERR_UNCONFIGURED;
	GDVIRTUAL_CALL(_write_frame, p_image, p_audio_data, ret);
	return ret;
}

void MovieWriter::write_end() {
	GDVIRTUAL_CALL(_write_end);
}

bool MovieWriter::handles_file(const String &p_path) const {
	bool ret = false;
	GDVIRTUAL_CALL(_handles_file, p_path, ret);
	return ret;
}

void MovieWriter::get_supported_extensions(List<String> *r_extensions) const {
	Vector<String> exts;
	GDVIRTUAL_CALL(_get_supported_extensions, exts);
	for (int i = 0; i < exts.size(); i++) {
		r_extensions->push_back(exts[i]);
	}
}

void MovieWriter::begin(const Size2i &p_movie_size, uint32_t p_fps, const String &p_base_path) {
	project_name = GLOBAL_GET("application/config/name");

	if (MovieDebugUtils::is_stdout_verbose()) {
		if (realtime_mode) {
			print_line(vformat("Movie Maker mode enabled (REALTIME), recording movie at %d FPS...", p_fps));
		} else {
			print_line(vformat("Movie Maker mode enabled, recording movie at %d FPS...", p_fps));
		}
	}

	// Check for available disk space and warn the user if needed.
	Ref<DirAccess> dir = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	String path = p_base_path.get_basename();
	if (path.is_relative_path()) {
		path = "res://" + path;
	}
	dir->open(path);
	if (dir->get_space_left() < 10 * Math::pow(1024.0, 3.0)) {
		// Less than 10 GiB available.
		WARN_PRINT(vformat("Current available space on disk is low (%s). MovieWriter will fail during movie recording if the disk runs out of available space.", String::humanize_size(dir->get_space_left())));
	}

	cpu_time = 0.0f;
	gpu_time = 0.0f;

	mix_rate = get_audio_mix_rate();
	fps = p_fps;
	
	if (realtime_mode) {
#ifdef WEB_ENABLED
		// Web platform: Prioritize Canvas video recording (if available)
		if (godot_video_recorder_init(p_fps) == 1) {
			// Canvas video recording initialization successful
			web_video_recorder_initialized = true;
			int start_result = godot_video_recorder_start();
			if (start_result == 1) {
				web_video_recording_active = true;
				audio_channels = 2; // Web platform default stereo
				if (MovieDebugUtils::is_stdout_verbose()) {
					print_line("MovieWriter: Web realtime recording mode - Canvas video + audio recording started");
					print_line("  Using Canvas.captureStream() for video");
					print_line("  Using MediaRecorder API for audio+video combined recording");
					print_line("  Frame rate: " + itos(p_fps) + " FPS");
				}
			} else {
				ERR_PRINT("MovieWriter: Failed to start Canvas video recording, falling back to audio-only");
				// Fallback to pure audio recording
				setup_web_audio_recorder();
				if (web_audio_recorder_initialized) {
					int audio_start_result = godot_audio_recorder_start();
					if (audio_start_result == 1) {
						web_audio_recording_active = true;
						audio_channels = 2;
						if (MovieDebugUtils::is_stdout_verbose()) {
							print_line("MovieWriter: Fallback to audio-only recording mode");
						}
					} else {
						realtime_mode = false;
					}
				} else {
					realtime_mode = false;
				}
			}
		} else {
			ERR_PRINT("MovieWriter: Canvas video recording not supported, falling back to audio-only");
			// Web platform: Use MediaRecorder API (pure audio)
			setup_web_audio_recorder();
			if (web_audio_recorder_initialized) {
				// Start Web audio recording
				int start_result = godot_audio_recorder_start();
				if (start_result == 1) {
					web_audio_recording_active = true;
					audio_channels = 2; // Web platform default stereo
					if (MovieDebugUtils::is_stdout_verbose()) {
						print_line("MovieWriter: Web realtime recording mode - MediaRecorder started (audio only)");
					}
				} else {
					ERR_PRINT("MovieWriter: Failed to start web audio recording, falling back to offline mode");
					realtime_mode = false;
				}
			} else {
				ERR_PRINT("MovieWriter: Failed to initialize web audio recorder, falling back to offline mode");
				realtime_mode = false;
			}
		}
#else
		// PC platform: Use HybridAudioDriver
		// Ensure HybridAudioDriver is initialized (if it wasn't initialized due to timing issues)
		if (!MovieWriter::hybrid_driver) {
			setup_hybrid_audio_driver();
		}
		
		if (MovieWriter::hybrid_driver) {
			// Real-time recording mode: enable audio capture
			MovieWriter::hybrid_driver->enable_recording(true);
			audio_channels = MovieWriter::hybrid_driver->get_channels();
			if (MovieDebugUtils::is_stdout_verbose()) {
				print_line("MovieWriter: Realtime recording mode - audio capture enabled");
			}
		} else {
			ERR_PRINT("MovieWriter: Failed to initialize HybridAudioDriver, falling back to offline mode");
			realtime_mode = false;
		}
#endif
	}
	
	if (!realtime_mode) {
		// Traditional offline recording mode: use dummy driver
		AudioDriverDummy::get_dummy_singleton()->set_mix_rate(mix_rate);
		AudioDriverDummy::get_dummy_singleton()->set_speaker_mode(AudioDriver::SpeakerMode(get_audio_speaker_mode()));
		audio_channels = AudioDriverDummy::get_dummy_singleton()->get_channels();
		if (MovieDebugUtils::is_stdout_verbose()) {
			print_line("MovieWriter: Offline recording mode - using dummy driver");
		}
	}
	
	if ((mix_rate % fps) != 0) {
		WARN_PRINT("MovieWriter's audio mix rate (" + itos(mix_rate) + ") can not be divided by the recording FPS (" + itos(fps) + "). Audio may go out of sync over time.");
	}

	audio_mix_buffer.resize(mix_rate * audio_channels / fps);

#ifdef WEB_ENABLED
	// Web platform: If canvas recording or audio recording is successful, skip the traditional write_begin process.
	if (realtime_mode && (web_video_recording_active || web_audio_recording_active)) {
		if (MovieDebugUtils::is_stdout_verbose()) {
			print_line("MovieWriter: Using Web native recording, skipping traditional MovieWriter pipeline");
		}
		return; // Skip write_begin call
	}
#endif

	write_begin(p_movie_size, p_fps, p_base_path);
}

void MovieWriter::_bind_methods() {
	ClassDB::bind_static_method("MovieWriter", D_METHOD("add_writer", "writer"), &MovieWriter::add_writer);

	GDVIRTUAL_BIND(_get_audio_mix_rate)
	GDVIRTUAL_BIND(_get_audio_speaker_mode)

	GDVIRTUAL_BIND(_handles_file, "path")

	GDVIRTUAL_BIND(_write_begin, "movie_size", "fps", "base_path")
	GDVIRTUAL_BIND(_write_frame, "frame_image", "audio_frame_block")
	GDVIRTUAL_BIND(_write_end)

	GLOBAL_DEF(PropertyInfo(Variant::INT, "editor/movie_writer/mix_rate", PROPERTY_HINT_RANGE, "8000,192000,1,suffix:Hz"), 48000);
	GLOBAL_DEF(PropertyInfo(Variant::INT, "editor/movie_writer/speaker_mode", PROPERTY_HINT_ENUM, "Stereo,3.1,5.1,7.1"), 0);
	GLOBAL_DEF(PropertyInfo(Variant::FLOAT, "editor/movie_writer/mjpeg_quality", PROPERTY_HINT_RANGE, "0.01,1.0,0.01"), 0.75);
	// Used by the editor.
	GLOBAL_DEF_BASIC("editor/movie_writer/movie_file", "");
	GLOBAL_DEF_BASIC("editor/movie_writer/disable_vsync", false);
	GLOBAL_DEF_BASIC(PropertyInfo(Variant::INT, "editor/movie_writer/fps", PROPERTY_HINT_RANGE, "1,300,1,suffix:FPS"), 60);
	// Real-time recording configuration (realtime_mode is defined in main.cpp)
	GLOBAL_DEF_BASIC("movie_writer/enable_audio_playback", true);
	
	// OBS-style recording configuration
	GLOBAL_DEF("movie_writer/obs_mode", false);
	GLOBAL_DEF(PropertyInfo(Variant::INT, "movie_writer/obs_video_fps", PROPERTY_HINT_RANGE, "10,120,1,suffix:FPS"), 30);
	GLOBAL_DEF(PropertyInfo(Variant::FLOAT, "movie_writer/obs_video_quality", PROPERTY_HINT_RANGE, "0.1,1.0,0.01"), 0.85);
	GLOBAL_DEF(PropertyInfo(Variant::INT, "movie_writer/obs_audio_sample_rate", PROPERTY_HINT_RANGE, "8000,192000,1,suffix:Hz"), 48000);
	GLOBAL_DEF(PropertyInfo(Variant::INT, "movie_writer/obs_audio_channels", PROPERTY_HINT_RANGE, "1,8,1"), 2);
	GLOBAL_DEF("movie_writer/obs_enable_timestamp_chunks", true);
	GLOBAL_DEF("movie_writer/obs_enable_repeat_frame_marking", true);
	GLOBAL_DEF("movie_writer/obs_enable_debug_output", true);
}

void MovieWriter::set_extensions_hint() {
	RBSet<String> found;
	for (uint32_t i = 0; i < writer_count; i++) {
		List<String> extensions;
		writers[i]->get_supported_extensions(&extensions);
		for (const String &ext : extensions) {
			found.insert(ext);
		}
	}

	String ext_hint;

	for (const String &S : found) {
		if (ext_hint != "") {
			ext_hint += ",";
		}
		ext_hint += "*." + S;
	}
	ProjectSettings::get_singleton()->set_custom_property_info(PropertyInfo(Variant::STRING, "editor/movie_writer/movie_file", PROPERTY_HINT_GLOBAL_SAVE_FILE, ext_hint));
}

void MovieWriter::add_frame() {
	const int movie_time_seconds = Engine::get_singleton()->get_frames_drawn() / fps;
	const String movie_time = vformat("%s:%s:%s",
			String::num(movie_time_seconds / 3600).pad_zeros(2),
			String::num((movie_time_seconds % 3600) / 60).pad_zeros(2),
			String::num(movie_time_seconds % 60).pad_zeros(2));

#ifdef DEBUG_ENABLED
	DisplayServer::get_singleton()->window_set_title(vformat("MovieWriter: Frame %d (time: %s) - %s (DEBUG)", Engine::get_singleton()->get_frames_drawn(), movie_time, project_name));
#else
	DisplayServer::get_singleton()->window_set_title(vformat("MovieWriter: Frame %d (time: %s) - %s", Engine::get_singleton()->get_frames_drawn(), movie_time, project_name));
#endif

	RID main_vp_rid = RenderingServer::get_singleton()->viewport_find_from_screen_attachment(DisplayServer::MAIN_WINDOW_ID);
	RID main_vp_texture = RenderingServer::get_singleton()->viewport_get_texture(main_vp_rid);
	Ref<Image> vp_tex = RenderingServer::get_singleton()->texture_2d_get(main_vp_texture);
	if (RenderingServer::get_singleton()->viewport_is_using_hdr_2d(main_vp_rid)) {
		vp_tex->convert(Image::FORMAT_RGBA8);
		vp_tex->linear_to_srgb();
	}

	RenderingServer::get_singleton()->viewport_set_measure_render_time(main_vp_rid, true);
	cpu_time += RenderingServer::get_singleton()->viewport_get_measured_render_time_cpu(main_vp_rid);
	cpu_time += RenderingServer::get_singleton()->get_frame_setup_time_cpu();
	gpu_time += RenderingServer::get_singleton()->viewport_get_measured_render_time_gpu(main_vp_rid);

	if (realtime_mode) {
#ifdef WEB_ENABLED
		// Web platform: Check Canvas video recording status
		if (web_video_recording_active) {
			// Canvas video recording is in progress, audio+video are handled by MediaRecorder
		} else if (web_audio_recording_active) {
			// Only audio recording is in progress
			bool has_audio_data = process_web_audio_data();
			if (!has_audio_data) {
				// If no audio data, fill with silence
				int requested_frames = mix_rate / fps;
				for (int i = 0; i < requested_frames * audio_channels; i++) {
					audio_mix_buffer[i] = 0;
				}
			}
		} else {
			// Web platform: fallback to offline mode
			AudioDriverDummy::get_dummy_singleton()->mix_audio(mix_rate / fps, audio_mix_buffer.ptr());
		}
#else
		// PC platform: Get captured audio data from HybridAudioDriver
		if (MovieWriter::hybrid_driver) {
			int requested_frames = mix_rate / fps;
			
			// Get audio data
			MovieWriter::hybrid_driver->get_captured_audio_data(audio_mix_buffer.ptr(), requested_frames);

		} else {
			// PC platform: fallback to offline mode
			AudioDriverDummy::get_dummy_singleton()->mix_audio(mix_rate / fps, audio_mix_buffer.ptr());
		}
#endif
	} else {
		// Traditional offline recording mode: get audio data from dummy driver
		AudioDriverDummy::get_dummy_singleton()->mix_audio(mix_rate / fps, audio_mix_buffer.ptr());
	}
	
#ifdef WEB_ENABLED
	// Web platform: If canvas recording or audio recording is successful, skip the traditional write_frame process.
	if (realtime_mode && (web_video_recording_active || web_audio_recording_active)) {
		// Canvas recording or Web audio recording is in progress, no need to call traditional write_frame.
		return;
	}
#endif
	
	write_frame(vp_tex, audio_mix_buffer.ptr());
}

void MovieWriter::end() {
	if (realtime_mode) {
#ifdef WEB_ENABLED
		// Web platform: Cleanup
		if (web_video_recording_active) {
			cleanup_web_video_recorder();
		} else if (web_audio_recording_active) {
			cleanup_web_audio_recorder();
		}
#else
		// PC platform: Cleanup using HybridAudioDriver
		restore_original_audio_driver();
#endif
	}
	
	// Call the write_end method of the subclass
	write_end();

	// Print a report with various statistics.
	String movie_path = Engine::get_singleton()->get_write_movie_path();
	if (movie_path.is_relative_path()) {
		// Print absolute path to make finding the file easier,
		// and to make it clickable in terminal emulators that support this.
		movie_path = ProjectSettings::get_singleton()->globalize_path("res://").path_join(movie_path);
	}

	const int movie_time_seconds = Engine::get_singleton()->get_frames_drawn() / fps;
	const String movie_time = vformat("%s:%s:%s",
			String::num(movie_time_seconds / 3600).pad_zeros(2),
			String::num((movie_time_seconds % 3600) / 60).pad_zeros(2),
			String::num(movie_time_seconds % 60).pad_zeros(2));

	const int real_time_seconds = Time::get_singleton()->get_ticks_msec() / 1000;
	const String real_time = vformat("%s:%s:%s",
			String::num(real_time_seconds / 3600).pad_zeros(2),
			String::num((real_time_seconds % 3600) / 60).pad_zeros(2),
			String::num(real_time_seconds % 60).pad_zeros(2));

	if(MovieDebugUtils::is_stdout_verbose()) {
		print_line("----------------");
		print_line(vformat("Done recording movie at path: %s", movie_path));
		print_line(vformat("%d frames at %d FPS (movie length: %s), recorded in %s (%d%% of real-time speed).", Engine::get_singleton()->get_frames_drawn(), fps, movie_time, real_time, (float(movie_time_seconds) / real_time_seconds) * 100));
		print_line(vformat("CPU time: %.2f seconds (average: %.2f ms/frame)", cpu_time / 1000, cpu_time / Engine::get_singleton()->get_frames_drawn()));
		print_line(vformat("GPU time: %.2f seconds (average: %.2f ms/frame)", gpu_time / 1000, gpu_time / Engine::get_singleton()->get_frames_drawn()));
		print_line("----------------");
	}
}

void MovieWriter::set_realtime_mode(bool p_enable) {
	realtime_mode = p_enable;
	if (p_enable) {
		setup_hybrid_audio_driver();
	} else {
		restore_original_audio_driver();
	}
}

void MovieWriter::setup_hybrid_audio_driver() {
	// Check if AudioServer is initialized
	if (!AudioServer::get_singleton()) {
		if (MovieDebugUtils::is_stdout_verbose()) {
			print_line("MovieWriter: AudioServer not yet initialized, deferring HybridAudioDriver setup");
		}
		return;
	}
	
	if (!MovieWriter::hybrid_driver) {
		MovieWriter::hybrid_driver = memnew(HybridAudioDriver);
		
		// Initialize with current AudioServer parameters
		int current_mix_rate = AudioServer::get_singleton()->get_mix_rate();
		AudioServer::SpeakerMode current_speaker_mode = AudioServer::get_singleton()->get_speaker_mode();
		
		if (MovieDebugUtils::is_stdout_verbose()) {
			print_line(vformat("MovieWriter: Initializing HybridAudioDriver - Mix rate: %d Hz, Speaker mode: %d", 
					   current_mix_rate, current_speaker_mode));
		}
		
		// Initialize mixed driver
		Error err = MovieWriter::hybrid_driver->init(current_mix_rate, AudioDriver::SpeakerMode(current_speaker_mode));
		if (err != OK) {
			ERR_PRINT("Failed to initialize HybridAudioDriver");
			memdelete(MovieWriter::hybrid_driver);
			MovieWriter::hybrid_driver = nullptr;
			return;
		}
		
		// Start mixed driver
		MovieWriter::hybrid_driver->start();
		
		// Register with AudioServer for audio capture
		AudioServer::get_singleton()->set_audio_capture_interface(MovieWriter::hybrid_driver);
		
		// Verify audio parameter synchronization
		if (MovieWriter::hybrid_driver->get_mix_rate() != current_mix_rate) {
			WARN_PRINT(vformat("HybridAudioDriver mix rate mismatch: expected %d, got %d", 
					   current_mix_rate, MovieWriter::hybrid_driver->get_mix_rate()));
		}
		
		if (MovieDebugUtils::is_stdout_verbose()) {
			print_line(vformat("HybridAudioDriver initialized successfully - %d Hz, %d channels, Buffer: %d frames (%.1fms)", 
					   MovieWriter::hybrid_driver->get_mix_rate(), MovieWriter::hybrid_driver->get_channels(), 
					   int(MovieWriter::hybrid_driver->get_mix_rate() * 0.2f), 200.0f)); // 200ms double buffer
		}
	}
}

void MovieWriter::restore_original_audio_driver() {
	if (MovieWriter::hybrid_driver) {
		// Remove capture interface from AudioServer (ensure AudioServer still exists)
		if (AudioServer::get_singleton()) {
			AudioServer::get_singleton()->remove_audio_capture_interface();
		}
		
		MovieWriter::hybrid_driver->enable_recording(false);
		MovieWriter::hybrid_driver->finish();
		memdelete(MovieWriter::hybrid_driver);
		MovieWriter::hybrid_driver = nullptr;
		if (MovieDebugUtils::is_stdout_verbose()) {
			print_line("HybridAudioDriver restored");
		}
	}
}

#ifdef WEB_ENABLED
// Web platform audio recording methods implementation

void MovieWriter::setup_web_audio_recorder() {
	if (web_audio_recorder_initialized) {
		if (MovieDebugUtils::is_stdout_verbose()) {
			print_line("MovieWriter: Web audio recorder already initialized");
		}
		return;
	}

	// Initialize Web audio recorder
	int result = godot_audio_recorder_init();
	if (result == 1) {
		web_audio_recorder_initialized = true;
		if (MovieDebugUtils::is_stdout_verbose()) {
			print_line("MovieWriter: Web audio recorder initialized successfully");
		}
		
		// Get supported MIME types
		int mime_type_ptr = godot_audio_recorder_get_mime_type();
		if (mime_type_ptr != 0) {
			// Note: String pointer needs to be handled appropriately in actual use, memory needs to be freed
			if (MovieDebugUtils::is_stdout_verbose()) {
				print_line("MovieWriter: Web audio recorder MIME type initialized");
			}
		}
	} else {
		ERR_PRINT("MovieWriter: Failed to initialize web audio recorder");
		web_audio_recorder_initialized = false;
	}
}

void MovieWriter::cleanup_web_audio_recorder() {
	if (!web_audio_recorder_initialized) {
		return;
	}

	// Stop recording
	if (web_audio_recording_active) {
		godot_audio_recorder_stop();
		web_audio_recording_active = false;
	}

	// Clean up recorder resources
	godot_audio_recorder_cleanup();
	web_audio_recorder_initialized = false;
	web_audio_buffer.clear();
	
	if (MovieDebugUtils::is_stdout_verbose()) {
		print_line("MovieWriter: Web audio recorder cleaned up");
	}
}

bool MovieWriter::process_web_audio_data() {
	if (!web_audio_recorder_initialized || !web_audio_recording_active) {
		return false;
	}

	return true;
}

#endif // WEB_ENABLED

HybridAudioDriver *MovieWriter::get_hybrid_audio_driver() {
	return MovieWriter::hybrid_driver;
}

#ifdef WEB_ENABLED

void MovieWriter::setup_web_video_recorder(uint32_t p_fps) {
	if (godot_video_recorder_init(p_fps) == 1) {
		web_video_recorder_initialized = true;
		if (MovieDebugUtils::is_stdout_verbose()) {
			print_line("MovieWriter: Canvas video recorder initialized successfully");
		}
	} else {
		web_video_recorder_initialized = false;
		ERR_PRINT("MovieWriter: Failed to initialize Canvas video recorder");
	}
}

void MovieWriter::cleanup_web_video_recorder() {
	
	if (web_video_recording_active) {
		godot_video_recorder_stop();
		web_video_recording_active = false;
		
		// Check if there is recorded data to download
		if (godot_video_recorder_has_data() == 1 && enable_web_auto_download) {
			int data_size = godot_video_recorder_get_data_size();
			if (MovieDebugUtils::is_stdout_verbose()) {
				print_line(String("MovieWriter: Canvas video recording completed. Data size: ") + String::humanize_size(data_size));
			}
			
			// Automatically download the recorded video file
			String filename = String("spx_recording_") + Time::get_singleton()->get_datetime_string_from_system(false, true) + ".webm";
			godot_video_recorder_download_data(filename.utf8().get_data());
			if (MovieDebugUtils::is_stdout_verbose()) {
				print_line("MovieWriter: Canvas video file download initiated: " + filename);
			}
		} else {
			if (MovieDebugUtils::is_stdout_verbose()) {
				print_line("MovieWriter: No Canvas video data to download");
			}
		}
	}
	
	if (web_video_recorder_initialized && enable_web_auto_download) {
		godot_video_recorder_cleanup();
	}
	
	if (web_video_recorder_initialized){
		web_video_recorder_initialized = false;
		if (MovieDebugUtils::is_stdout_verbose()) {
			print_line("MovieWriter: Canvas video recorder cleaned up");
		}
	}
}

bool MovieWriter::process_web_video_data() {
	if (!web_video_recording_active) {
		return false;
	}
	
	// Check for new recorded data
	return godot_video_recorder_has_new_data() == 1;
}

#endif // WEB_ENABLED
