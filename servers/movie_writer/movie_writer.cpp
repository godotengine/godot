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
		// PC platform: Get captured audio data from HybridAudioDriver
		if (MovieWriter::hybrid_driver) {
			int requested_frames = mix_rate / fps;
			
			// Get audio data
			MovieWriter::hybrid_driver->get_captured_audio_data(audio_mix_buffer.ptr(), requested_frames);

		} else {
			// PC platform: fallback to offline mode
			AudioDriverDummy::get_dummy_singleton()->mix_audio(mix_rate / fps, audio_mix_buffer.ptr());
		}
	} else {
		// Traditional offline recording mode: get audio data from dummy driver
		AudioDriverDummy::get_dummy_singleton()->mix_audio(mix_rate / fps, audio_mix_buffer.ptr());
	}
	
	write_frame(vp_tex, audio_mix_buffer.ptr());
}

void MovieWriter::end() {
	if (realtime_mode) {
		// PC platform: Cleanup using HybridAudioDriver
		restore_original_audio_driver();
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


HybridAudioDriver *MovieWriter::get_hybrid_audio_driver() {
	return MovieWriter::hybrid_driver;
}
