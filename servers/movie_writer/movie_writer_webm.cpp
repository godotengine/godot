/**************************************************************************/
/*  movie_writer_webm.cpp                                                */
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

#include "movie_writer_webm.h"
#include "core/config/project_settings.h"
#include "core/os/time.h"
#include "movie_utils.h"
#include "platform/web/godot_audio.h"
#include "core/io/file_access.h"
#include "movie_recorder_manager.h"

#ifdef WEB_ENABLED
#include <emscripten.h>
#endif


MovieWriterWebM::MovieWriterWebM() {
    if (ProjectSettings::get_singleton()->has_setting("movie_writer/enable_web_auto_download")) {
        enable_auto_download = GLOBAL_GET("movie_writer/enable_web_auto_download");
    }else{
		enable_auto_download = true;
	}
}

bool MovieWriterWebM::handles_file(const String &p_path) const {
	// web only support this movie writer
#ifndef WEB_ENABLED
	return false;
#else 
	return true;
#endif 
}

uint32_t MovieWriterWebM::get_audio_mix_rate() const {
	return 48000;
}

AudioServer::SpeakerMode MovieWriterWebM::get_audio_speaker_mode() const {
	return AudioServer::SPEAKER_MODE_STEREO;;
}

void MovieWriterWebM::get_supported_extensions(List<String> *r_extensions) const {
	r_extensions->push_back("webm");
}

#ifndef WEB_ENABLED //  give a empty implement

Error MovieWriterWebM::write_begin(const Size2i &p_movie_size, uint32_t p_fps, const String &p_base_path) {
	return ERR_UNAVAILABLE;
}
Error MovieWriterWebM::write_frame(const Ref<Image> &p_image, const int32_t *p_audio_data) {
	return ERR_UNAVAILABLE;
}
void MovieWriterWebM::write_end() {
}

#else //WEB_ENABLED
Error MovieWriterWebM::write_begin(const Size2i &p_movie_size, uint32_t p_fps, const String &p_base_path) {
	base_path = p_base_path.get_basename();
	if (base_path.is_relative_path()) {
		base_path = "res://" + base_path;
	}
	base_path += ".webm";
	fps = p_fps;


	if (is_realtime_mode()) {
		if (MovieDebugUtils::is_stdout_verbose()) {
			print_line("MovieWriterWebM: Starting web realtime recording");
		}
		
		if (godot_video_recorder_init(p_fps) == 1) {
			web_video_recorder_initialized = true;
			int start_result = godot_video_recorder_start();
			if (start_result == 1) {
				web_video_recording_active = true;
				if (MovieDebugUtils::is_stdout_verbose()) {
					print_line("MovieWriterWebM: Canvas video + audio recording started");
					print_line("  Using Canvas.captureStream() for video");
					print_line("  Using MediaRecorder API for audio+video combined recording");
					print_line("  Frame rate: " + itos(p_fps) + " FPS");
				}
				return OK;
			} else {
				ERR_PRINT("MovieWriterWebM: Failed to start Canvas video recording, falling back to audio-only");
				setup_web_audio_recorder();
				if (web_audio_recorder_initialized) {
					int audio_start_result = godot_audio_recorder_start();
					if (audio_start_result == 1) {
						web_audio_recording_active = true;
						if (MovieDebugUtils::is_stdout_verbose()) {
							print_line("MovieWriterWebM: Fallback to audio-only recording mode");
						}
						return OK;
					}
				}
				return ERR_CANT_CREATE;
			}
		} else {
			ERR_PRINT("MovieWriterWebM: Canvas video recording not supported, falling back to audio-only");
			setup_web_audio_recorder();
			if (web_audio_recorder_initialized) {
				int start_result = godot_audio_recorder_start();
				if (start_result == 1) {
					web_audio_recording_active = true;
					if (MovieDebugUtils::is_stdout_verbose()) {
						print_line("MovieWriterWebM: Web audio-only recording started");
					}
					return OK;
				}
			}
			return ERR_CANT_CREATE;
		}
	}


	ERR_PRINT("MovieWriterWebM: This writer is designed for web platform realtime recording only");
	return ERR_UNAVAILABLE;
}

Error MovieWriterWebM::write_frame(const Ref<Image> &p_image, const int32_t *p_audio_data) {
	if (web_video_recording_active) {
		return OK;
	} else if (web_audio_recording_active) {
		bool has_audio_data = process_web_audio_data();
		return has_audio_data ? OK : ERR_CANT_ACQUIRE_RESOURCE;
	}


	return ERR_UNAVAILABLE;
}

void MovieWriterWebM::write_end() {
	if (web_video_recording_active) {
		cleanup_web_video_recorder();
	} else if (web_audio_recording_active) {
		cleanup_web_audio_recorder();
	}
}



void MovieWriterWebM::setup_web_audio_recorder() {
	if (web_audio_recorder_initialized) {
		if (MovieDebugUtils::is_stdout_verbose()) {
			print_line("MovieWriterWebM: Web audio recorder already initialized");
		}
		return;
	}

	int result = godot_audio_recorder_init();
	if (result == 1) {
		web_audio_recorder_initialized = true;
		if (MovieDebugUtils::is_stdout_verbose()) {
			print_line("MovieWriterWebM: Web audio recorder initialized successfully");
		}
		
		int mime_type_ptr = godot_audio_recorder_get_mime_type();
		if (mime_type_ptr != 0) {
			if (MovieDebugUtils::is_stdout_verbose()) {
				print_line("MovieWriterWebM: Web audio recorder MIME type initialized");
			}
		}
	} else {
		ERR_PRINT("MovieWriterWebM: Failed to initialize web audio recorder");
		web_audio_recorder_initialized = false;
	}
}

void MovieWriterWebM::cleanup_web_audio_recorder() {
	if (!web_audio_recorder_initialized) {
		return;
	}

	if (web_audio_recording_active) {
		godot_audio_recorder_stop();
		web_audio_recording_active = false;
	}

	godot_audio_recorder_cleanup();
	web_audio_recorder_initialized = false;
	web_audio_buffer.clear();
	
	if (MovieDebugUtils::is_stdout_verbose()) {
		print_line("MovieWriterWebM: Web audio recorder cleaned up");
	}
}

bool MovieWriterWebM::process_web_audio_data() {
	if (!web_audio_recorder_initialized || !web_audio_recording_active) {
		return false;
	}
	return true;
}

void MovieWriterWebM::setup_web_video_recorder(uint32_t p_fps) {
	if (godot_video_recorder_init(p_fps) == 1) {
		web_video_recorder_initialized = true;
		if (MovieDebugUtils::is_stdout_verbose()) {
			print_line("MovieWriterWebM: Canvas video recorder initialized successfully");
		}
	} else {
		web_video_recorder_initialized = false;
		ERR_PRINT("MovieWriterWebM: Failed to initialize Canvas video recorder");
	}
}

void MovieWriterWebM::cleanup_web_video_recorder() {
	if (web_video_recording_active) {
		godot_video_recorder_stop();
		web_video_recording_active = false;
		
		if (godot_video_recorder_has_data() == 1 && enable_auto_download) {
			int data_size = godot_video_recorder_get_data_size();
			if (MovieDebugUtils::is_stdout_verbose()) {
				print_line(String("MovieWriterWebM: Canvas video recording completed. Data size: ") + String::humanize_size(data_size));
			}
			
			String filename = String("spx_recording_") + Time::get_singleton()->get_datetime_string_from_system(false, true) + ".webm";
			godot_video_recorder_download_data(filename.utf8().get_data());
			if (MovieDebugUtils::is_stdout_verbose()) {
				print_line("MovieWriterWebM: Canvas video file download initiated: " + filename);
			}
		} else {
			if (MovieDebugUtils::is_stdout_verbose()) {
				print_line("MovieWriterWebM: No Canvas video data to download");
			}
		}
	}
	
	if (web_video_recorder_initialized && enable_auto_download) {
		godot_video_recorder_cleanup();
	}
	
	if (web_video_recorder_initialized) {
		web_video_recorder_initialized = false;
		if (MovieDebugUtils::is_stdout_verbose()) {
			print_line("MovieWriterWebM: Canvas video recorder cleaned up");
		}
	}
}

bool MovieWriterWebM::process_web_video_data() {
	if (!web_video_recording_active) {
		return false;
	}
	
	return godot_video_recorder_has_new_data() == 1;
}


extern "C" {
    EMSCRIPTEN_KEEPALIVE
    int godot_web_recording_request_start(const char* filename) {
        String godot_filename = String::utf8(filename ? filename : "recording");
        
        MovieRecorderManager::RecordingConfig config(godot_filename);
        Error result = MovieRecorderManager::start_recording(config);
        
        return result == OK ? 1 : 0;
    }
    
    EMSCRIPTEN_KEEPALIVE
    int godot_web_recording_request_stop() {
        Error result = MovieRecorderManager::stop_recording();
        return result == OK ? 1 : 0;
    }
    
    EMSCRIPTEN_KEEPALIVE
    int godot_web_recording_is_active() {
        return MovieRecorderManager::is_recording() ? 1 : 0;
    }
}

#endif   //WEB_ENABLED