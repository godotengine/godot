/**************************************************************************/
/*  movie_writer_ffmpeg.cpp                                               */
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

#include "movie_writer_ffmpeg.h"

#include "core/config/project_settings.h"
#include "core/os/os.h"

// libnut FileAccess handler.
int nut_write_to_pipe(void *p_priv, size_t p_size, const uint8_t *p_buf) {
	FileAccess *pipe = static_cast<FileAccess *>(p_priv);
	if (!pipe->store_buffer(p_buf, p_size)) {
		return -1;
	}
	return p_size;
}

// Allocation callbacks.
static void *nut_malloc(size_t size) {
	return memalloc(size);
}

static void *nut_realloc(void *ptr, size_t size) {
	return memrealloc(ptr, size);
}

static void nut_free(void *ptr) {
	if (ptr) {
		memfree(ptr);
	}
}

uint32_t MovieWriterFFmpeg::get_audio_mix_rate() const {
	return mix_rate;
}

AuSE::SpeakerMode MovieWriterFFmpeg::get_audio_speaker_mode() const {
	return speaker_mode;
}

void MovieWriterFFmpeg::get_supported_extensions(List<String> *r_extensions) const {
	r_extensions->push_back("mkv");
	r_extensions->push_back("mov");
	r_extensions->push_back("mp4");
	r_extensions->push_back("webm");
}

bool MovieWriterFFmpeg::handles_file(const String &p_path) const {
	return p_path.has_extension("mkv") || p_path.has_extension("mov") || p_path.has_extension("mp4") || p_path.has_extension("webm");
}

Error MovieWriterFFmpeg::write_begin(const Size2i &p_movie_size, uint32_t p_fps, const String &p_base_path) {
	if (binary_path.is_empty()) {
		binary_path = "ffmpeg"; // Try from PATH
	} else {
		ERR_FAIL_COND_V_MSG(!FileAccess::exists(binary_path), ERR_FILE_NOT_FOUND, "Invalid FFmpeg binary.");
	}

	base_path = p_base_path.get_basename();
	String base_extension = p_base_path.get_extension();

	List<String> ffargs = {
		"-hide_banner",
		"-y", // Always overwrite without confirmation.
		"-f",
		"nut", // Input format is NUT.
		"-i",
		"-", // Input from stdin.
	};
	if (encoder_args.is_empty()) { // Custom CPU encoder defaults tuned for speed.
		ffargs.push_back("-c:v");
		if (base_extension == "webm") {
			ffargs.push_back("libvpx-vp9");
			ffargs.push_back("-pix_fmt");
			ffargs.push_back("yuv420p");
			ffargs.push_back("-b:v");
			ffargs.push_back("0"); // Needs to be explicit for CRF.
			ffargs.push_back("-deadline");
			ffargs.push_back("realtime");
			ffargs.push_back("-cpu-used");
			ffargs.push_back("8");
			ffargs.push_back("-row-mt");
			ffargs.push_back("1");
		} else {
			ffargs.push_back("libx264");
			ffargs.push_back("-pix_fmt");
			ffargs.push_back("yuv420p");
			ffargs.push_back("-preset");
			ffargs.push_back("veryfast");
		}
		ffargs.push_back("-crf");
		ffargs.push_back(itos(base_extension == "webm" ? 43 : 31));
		ffargs.push_back("-g");
		ffargs.push_back(itos(2 * p_fps)); // One keyframe every two seconds.
		ffargs.push_back("-c:a");
		ffargs.push_back(base_extension == "webm" ? "libopus" : "aac");
		ffargs.push_back("-b:a");
		ffargs.push_back(base_extension == "webm" ? "128k" : "192k");
	} else {
		const PackedStringArray video_args_split = encoder_args.split(" ");
		for (const String &s : video_args_split) {
			if (!s.is_empty()) {
				ffargs.push_back(s);
			}
		}
		PackedStringArray hardware_args_split = hardware_args.split(" ");
		hardware_args_split.reverse();
		for (const String &s : hardware_args_split) {
			if (!s.is_empty()) {
				ffargs.push_front(s);
			}
		}
	}
	ffargs.push_back(base_path + "." + base_extension);

	Dictionary video_dict = OS::get_singleton()->execute_with_pipe(binary_path, ffargs);
	ERR_FAIL_COND_V_MSG(video_dict.size() == 0, ERR_CANT_CREATE, "Could not execute FFmpeg for video writing.");
	video_pipe = video_dict["stdio"];
	video_out = video_dict["stderr"];
	video_pid = video_dict["pid"];

	int channels = 2;
	switch (speaker_mode) {
		case AuSE::SPEAKER_MODE_STEREO:
			channels = 2;
			break;
		case AuSE::SPEAKER_SURROUND_31:
			channels = 4;
			break;
		case AuSE::SPEAKER_SURROUND_51:
			channels = 6;
			break;
		case AuSE::SPEAKER_SURROUND_71:
			channels = 8;
			break;
	}

	fps = p_fps;

	nut_stream_header_tt streams[3] = {};
	memset(streams, 0, sizeof(streams));

	// Fourcc reference: https://github.com/lu-zero/nut/blob/master/docs/nut4cc.txt
	static uint8_t v4cc[4] = { 'R', 'G', 'B', 'A' };
	static uint8_t a4cc[4] = { 'P', 'S', 'D', 32 }; // PCM, Signed Integer, Default, 32-bit.

	// Video stream.
	streams[0].type = NUT_VIDEO_CLASS;
	streams[0].fourcc_len = 4;
	streams[0].fourcc = v4cc;
	streams[0].width = p_movie_size.width;
	streams[0].height = p_movie_size.height;
	streams[0].time_base.num = 1;
	streams[0].time_base.den = fps;
	streams[0].fixed_fps = 1;

	// Audio stream.
	streams[1].type = NUT_AUDIO_CLASS;
	streams[1].fourcc_len = 4;
	streams[1].fourcc = a4cc;
	streams[1].samplerate_num = mix_rate;
	streams[1].samplerate_denom = 1;
	streams[1].channel_count = channels;
	streams[1].time_base.num = 1;
	streams[1].time_base.den = mix_rate;

	// End of streams.
	streams[2].type = -1;

	nut_muxer_opts_tt opts = {};
	opts.output.priv = video_pipe.ptr();
	opts.output.write = nut_write_to_pipe;
	opts.alloc.malloc = nut_malloc;
	opts.alloc.realloc = nut_realloc;
	opts.alloc.free = nut_free;
	opts.realtime_stream = 1;
	opts.max_distance = 32768;

	// Write NUT header.
	nut_ctx = nut_muxer_init(&opts, streams, nullptr);
	Error write_error = video_pipe->get_error();
	ERR_FAIL_COND_V_MSG(write_error != OK, write_error, "Could not write NUT header to FFmpeg pipe.");

	const int blockalign = 4 * channels;
	audio_block_size = (mix_rate / fps) * blockalign;

	return OK;
}

Error MovieWriterFFmpeg::write_frame(const Ref<Image> &p_image, const int32_t *p_audio_data) {
	if (unlikely(!OS::get_singleton()->is_process_running(video_pid) || video_pipe->get_error() != OK)) {
		ERR_PRINT_ONCE(video_out->get_as_text());
		ERR_FAIL_V_MSG(ERR_FILE_CANT_WRITE, "Failed to write to FFmpeg: Broken pipe.");
	}

	const Vector<uint8_t> &v_data = p_image->get_data();
	nut_packet_tt vp;
	vp.stream = 0;
	vp.len = v_data.size();
	vp.pts = frame_count;
	vp.flags = NUT_FLAG_KEY;
	nut_write_frame(nut_ctx, &vp, v_data.ptr());

	nut_packet_tt ap;
	ap.stream = 1;
	ap.len = audio_block_size;
	ap.pts = (uint64_t)frame_count * (mix_rate / fps);
	ap.flags = 0;
	nut_write_frame(nut_ctx, &ap, (const uint8_t *)p_audio_data);

	frame_count++;
	return OK;
}

void MovieWriterFFmpeg::write_end() {
	if (nut_ctx) {
		nut_muxer_uninit(nut_ctx);
		nut_ctx = nullptr;
	}
	if (video_pipe.is_valid()) {
		video_pipe->close();
	}
}

MovieWriterFFmpeg::MovieWriterFFmpeg() {
	binary_path = GLOBAL_GET("editor/movie_writer/ffmpeg/binary_path");
	mix_rate = GLOBAL_GET("editor/movie_writer/mix_rate");
	speaker_mode = AuSE::SpeakerMode(int(GLOBAL_GET("editor/movie_writer/speaker_mode")));
	hardware_args = GLOBAL_GET("editor/movie_writer/ffmpeg/hardware_arguments");
	encoder_args = GLOBAL_GET("editor/movie_writer/ffmpeg/encoder_arguments");
}
