/**************************************************************************/
/*  movie_writer_webm.h                                                  */
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

#ifndef MOVIE_WRITER_WEBM_H
#define MOVIE_WRITER_WEBM_H

#include "servers/movie_writer/movie_writer.h"
#include "core/string/ustring.h"
#include "core/math/vector2i.h"
#include "servers/audio_server.h"
#include "core/error/error_macros.h"
#include "core/io/image.h"
#include "core/templates/list.h"

class MovieWriterWebM : public MovieWriter {
	GDCLASS(MovieWriterWebM, MovieWriter)


	bool enable_auto_download;
	String base_path;
	uint32_t fps = 30;
public:
	MovieWriterWebM();

protected:
	virtual uint32_t get_audio_mix_rate() const override;
	virtual AudioServer::SpeakerMode get_audio_speaker_mode() const override;
	virtual void get_supported_extensions(List<String> *r_extensions) const override;
	virtual Error write_begin(const Size2i &p_movie_size, uint32_t p_fps, const String &p_base_path) override;
	virtual Error write_frame(const Ref<Image> &p_image, const int32_t *p_audio_data) override;
	virtual void write_end() override;

	virtual bool handles_file(const String &p_path) const override;

#ifdef WEB_ENABLED

private:

	bool web_audio_recorder_initialized = false;
	bool web_audio_recording_active = false;
	Vector<uint8_t> web_audio_buffer;

	bool web_video_recorder_initialized = false;
	bool web_video_recording_active = false;
	
	void setup_web_audio_recorder();
	void cleanup_web_audio_recorder();
	bool process_web_audio_data();

	void setup_web_video_recorder(uint32_t p_fps);
	void cleanup_web_video_recorder();
	bool process_web_video_data();
#endif

	
};

#endif // MOVIE_WRITER_WEBM_H