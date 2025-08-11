/**************************************************************************/
/*  movie_writer.h                                                        */
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

#ifndef MOVIE_WRITER_H
#define MOVIE_WRITER_H

#include "core/io/image.h"
#include "core/templates/local_vector.h"
#include "servers/audio_server.h"

class HybridAudioDriver;
class AudioDriver;



class MovieWriter : public Object {
	GDCLASS(MovieWriter, Object);

	uint64_t fps = 0;
	uint64_t mix_rate = 0;
	uint32_t audio_channels = 0;

	float cpu_time = 0.0f;
	float gpu_time = 0.0f;

	String project_name;

	LocalVector<int32_t> audio_mix_buffer;

	// Real-time recording support
	bool realtime_mode = false;
	static class HybridAudioDriver *hybrid_driver;
	class AudioDriver *original_driver = nullptr;
#ifdef WEB_ENABLED
	// Web platform audio recording support (Option 1: MediaRecorder API)
	bool web_audio_recorder_initialized = false;
	bool web_audio_recording_active = false;
	Vector<uint8_t> web_audio_buffer; // Stores audio data recorded on the web platform

	// Web platform Canvas video recording support (Option 2: Canvas.captureStream + MediaRecorder)
	bool web_video_recorder_initialized = false;
	bool web_video_recording_active = false;

#endif

	enum {
		MAX_WRITERS = 8
	};
	static MovieWriter *writers[];
	static uint32_t writer_count;
protected:
	// Web platform configuration
	bool enable_web_auto_download = false;    // Enable automatic file download on web platform
	
	virtual uint32_t get_audio_mix_rate() const;
	virtual AudioServer::SpeakerMode get_audio_speaker_mode() const;

	virtual Error write_begin(const Size2i &p_movie_size, uint32_t p_fps, const String &p_base_path);
	virtual Error write_frame(const Ref<Image> &p_image, const int32_t *p_audio_data);
	virtual void write_end();

	GDVIRTUAL0RC_REQUIRED(uint32_t, _get_audio_mix_rate)
	GDVIRTUAL0RC_REQUIRED(AudioServer::SpeakerMode, _get_audio_speaker_mode)

	GDVIRTUAL1RC_REQUIRED(bool, _handles_file, const String &)
	GDVIRTUAL0RC_REQUIRED(Vector<String>, _get_supported_extensions)

	GDVIRTUAL3R_REQUIRED(Error, _write_begin, const Size2i &, uint32_t, const String &)
	GDVIRTUAL2R_REQUIRED(Error, _write_frame, const Ref<Image> &, GDExtensionConstPtr<int32_t>)
	GDVIRTUAL0_REQUIRED(_write_end)

	static void _bind_methods();

public:
	MovieWriter() {} // 确保成员变量正确初始化
	
	virtual bool handles_file(const String &p_path) const;
	virtual void get_supported_extensions(List<String> *r_extensions) const;

	static void add_writer(MovieWriter *p_writer);
	static MovieWriter *find_writer_for_file(const String &p_file);

	void begin(const Size2i &p_movie_size, uint32_t p_fps, const String &p_base_path);
	void add_frame();

	static void set_extensions_hint();

	void end();

	// Real-time recording control
	void set_realtime_mode(bool p_enable);
	bool is_realtime_mode() const { return realtime_mode; }

	// Get HybridAudioDriver instance (for other recorders to use)
	static class HybridAudioDriver *get_hybrid_audio_driver();

private:
	void setup_hybrid_audio_driver();
	void restore_original_audio_driver();

#ifdef WEB_ENABLED
	// Web platform audio recording specific methods
	void setup_web_audio_recorder();
	void cleanup_web_audio_recorder();
	bool process_web_audio_data();

	// Web platform Canvas video recording specific methods
	void setup_web_video_recorder(uint32_t p_fps);
	void cleanup_web_video_recorder();
	bool process_web_video_data();
#endif
};

#endif // MOVIE_WRITER_H
