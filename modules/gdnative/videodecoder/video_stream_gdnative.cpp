/*************************************************************************/
/*  video_stream_gdnative.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "video_stream_gdnative.h"

#include "core/project_settings.h"
#include "servers/audio_server.h"

VideoDecoderServer *VideoDecoderServer::instance = nullptr;

static VideoDecoderServer decoder_server;

const int AUX_BUFFER_SIZE = 1024; // Buffer 1024 samples.

// NOTE: Callbacks for the GDNative libraries.
extern "C" {
godot_int GDAPI godot_videodecoder_file_read(void *ptr, uint8_t *buf, int buf_size) {
	// ptr is a FileAccess
	FileAccess *file = reinterpret_cast<FileAccess *>(ptr);

	// if file exists
	if (file) {
		int64_t bytes_read = file->get_buffer(buf, buf_size);
		return bytes_read;
	}
	return -1;
}

int64_t GDAPI godot_videodecoder_file_seek(void *ptr, int64_t pos, int whence) {
	// file
	FileAccess *file = reinterpret_cast<FileAccess *>(ptr);

	if (file) {
		int64_t len = file->get_len();
		switch (whence) {
			case SEEK_SET: {
				if (pos > len) {
					return -1;
				}
				file->seek(pos);
				return file->get_position();
			} break;
			case SEEK_CUR: {
				// Just in case it doesn't exist
				if (pos < 0 && -pos > (int64_t)file->get_position()) {
					return -1;
				}
				file->seek(file->get_position() + pos);
				return file->get_position();
			} break;
			case SEEK_END: {
				// Just in case something goes wrong
				if (-pos > len) {
					return -1;
				}
				file->seek_end(pos);
				return file->get_position();
			} break;
			default: {
				// Only 4 possible options, hence default = AVSEEK_SIZE
				// Asks to return the length of file
				return len;
			} break;
		}
	}
	// In case nothing works out.
	return -1;
}

void GDAPI godot_videodecoder_register_decoder(const godot_videodecoder_interface_gdnative *p_interface) {
	decoder_server.register_decoder_interface(p_interface);
}
}

// VideoStreamPlaybackGDNative starts here.

bool VideoStreamPlaybackGDNative::open_file(const String &p_file) {
	ERR_FAIL_COND_V(interface == nullptr, false);
	file = FileAccess::open(p_file, FileAccess::READ);
	bool file_opened = interface->open_file(data_struct, file);

	if (file_opened) {
		num_channels = interface->get_channels(data_struct);
		mix_rate = interface->get_mix_rate(data_struct);

		godot_vector2 vec = interface->get_texture_size(data_struct);
		texture_size = *(Vector2 *)&vec;
		// Only do memset if num_channels > 0 otherwise it will crash.
		if (num_channels > 0) {
			pcm = (float *)memalloc(num_channels * AUX_BUFFER_SIZE * sizeof(float));
			memset(pcm, 0, num_channels * AUX_BUFFER_SIZE * sizeof(float));
		}

		pcm_write_idx = -1;
		samples_decoded = 0;

		texture->create((int)texture_size.width, (int)texture_size.height, Image::FORMAT_RGBA8, Texture::FLAG_FILTER | Texture::FLAG_VIDEO_SURFACE);
	}

	return file_opened;
}

void VideoStreamPlaybackGDNative::update(float p_delta) {
	if (!playing || paused) {
		return;
	}
	if (!file) {
		return;
	}
	time += p_delta;
	ERR_FAIL_COND(interface == nullptr);
	interface->update(data_struct, p_delta);

	// Don't mix if there's no audio (num_channels == 0).
	if (mix_callback && num_channels > 0) {
		if (pcm_write_idx >= 0) {
			// Previous remains
			int mixed = mix_callback(mix_udata, pcm + pcm_write_idx * num_channels, samples_decoded);
			if (mixed == samples_decoded) {
				pcm_write_idx = -1;
			} else {
				samples_decoded -= mixed;
				pcm_write_idx += mixed;
			}
		}
		if (pcm_write_idx < 0) {
			samples_decoded = interface->get_audioframe(data_struct, pcm, AUX_BUFFER_SIZE);
			pcm_write_idx = mix_callback(mix_udata, pcm, samples_decoded);
			if (pcm_write_idx == samples_decoded) {
				pcm_write_idx = -1;
			} else {
				samples_decoded -= pcm_write_idx;
			}
		}
	}

	if (seek_backward) {
		update_texture();
		seek_backward = false;
	}

	while (interface->get_playback_position(data_struct) < time && playing) {
		update_texture();
	}
}

void VideoStreamPlaybackGDNative::update_texture() {
	PoolByteArray *pba = (PoolByteArray *)interface->get_videoframe(data_struct);

	if (pba == nullptr) {
		playing = false;
		return;
	}

	Ref<Image> img = memnew(Image(texture_size.width, texture_size.height, 0, Image::FORMAT_RGBA8, *pba));

	texture->set_data(img);
}

// ctor and dtor

VideoStreamPlaybackGDNative::VideoStreamPlaybackGDNative() :
		texture(Ref<ImageTexture>(memnew(ImageTexture))),
		playing(false),
		paused(false),
		mix_udata(nullptr),
		mix_callback(nullptr),
		num_channels(-1),
		time(0),
		seek_backward(false),
		mix_rate(0),
		delay_compensation(0),
		pcm(nullptr),
		pcm_write_idx(0),
		samples_decoded(0),
		file(nullptr),
		interface(nullptr),
		data_struct(nullptr) {}

VideoStreamPlaybackGDNative::~VideoStreamPlaybackGDNative() {
	cleanup();
}

void VideoStreamPlaybackGDNative::cleanup() {
	if (data_struct) {
		interface->destructor(data_struct);
	}
	if (pcm) {
		memfree(pcm);
	}
	if (file) {
		file->close();
		memdelete(file);
		file = nullptr;
	}
	pcm = nullptr;
	time = 0;
	num_channels = -1;
	interface = nullptr;
	data_struct = nullptr;
}

void VideoStreamPlaybackGDNative::set_interface(const godot_videodecoder_interface_gdnative *p_interface) {
	ERR_FAIL_COND(p_interface == nullptr);
	if (interface != nullptr) {
		cleanup();
	}
	interface = p_interface;
	data_struct = interface->constructor((godot_object *)this);
}

// controls

bool VideoStreamPlaybackGDNative::is_playing() const {
	return playing;
}

bool VideoStreamPlaybackGDNative::is_paused() const {
	return paused;
}

void VideoStreamPlaybackGDNative::play() {
	stop();

	playing = true;

	delay_compensation = ProjectSettings::get_singleton()->get("audio/video_delay_compensation_ms");
	delay_compensation /= 1000.0;
}

void VideoStreamPlaybackGDNative::stop() {
	if (playing) {
		seek(0);
	}
	playing = false;
}

void VideoStreamPlaybackGDNative::seek(float p_time) {
	ERR_FAIL_COND(interface == nullptr);
	interface->seek(data_struct, p_time);
	if (p_time < time) {
		seek_backward = true;
	}
	time = p_time;
	// reset audio buffers
	memset(pcm, 0, num_channels * AUX_BUFFER_SIZE * sizeof(float));
	pcm_write_idx = -1;
	samples_decoded = 0;
}

void VideoStreamPlaybackGDNative::set_paused(bool p_paused) {
	paused = p_paused;
}

Ref<Texture> VideoStreamPlaybackGDNative::get_texture() const {
	return texture;
}

float VideoStreamPlaybackGDNative::get_length() const {
	ERR_FAIL_COND_V(interface == nullptr, 0);
	return interface->get_length(data_struct);
}

float VideoStreamPlaybackGDNative::get_playback_position() const {
	ERR_FAIL_COND_V(interface == nullptr, 0);
	return interface->get_playback_position(data_struct);
}

bool VideoStreamPlaybackGDNative::has_loop() const {
	// TODO: Implement looping?
	return false;
}

void VideoStreamPlaybackGDNative::set_loop(bool p_enable) {
	// Do nothing
}

void VideoStreamPlaybackGDNative::set_audio_track(int p_idx) {
	ERR_FAIL_COND(interface == nullptr);
	interface->set_audio_track(data_struct, p_idx);
}

void VideoStreamPlaybackGDNative::set_mix_callback(AudioMixCallback p_callback, void *p_userdata) {
	mix_udata = p_userdata;
	mix_callback = p_callback;
}

int VideoStreamPlaybackGDNative::get_channels() const {
	ERR_FAIL_COND_V(interface == nullptr, 0);

	return (num_channels > 0) ? num_channels : 0;
}

int VideoStreamPlaybackGDNative::get_mix_rate() const {
	ERR_FAIL_COND_V(interface == nullptr, 0);

	return mix_rate;
}

/* --- NOTE VideoStreamGDNative starts here. ----- */

Ref<VideoStreamPlayback> VideoStreamGDNative::instance_playback() {
	Ref<VideoStreamPlaybackGDNative> pb = memnew(VideoStreamPlaybackGDNative);
	VideoDecoderGDNative *decoder = decoder_server.get_decoder(file.get_extension().to_lower());
	if (decoder == nullptr) {
		return nullptr;
	}
	pb->set_interface(decoder->interface);
	pb->set_audio_track(audio_track);
	if (pb->open_file(file)) {
		return pb;
	}
	return nullptr;
}

void VideoStreamGDNative::set_file(const String &p_file) {
	file = p_file;
}

String VideoStreamGDNative::get_file() {
	return file;
}

void VideoStreamGDNative::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_file", "file"), &VideoStreamGDNative::set_file);
	ClassDB::bind_method(D_METHOD("get_file"), &VideoStreamGDNative::get_file);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "file", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL), "set_file", "get_file");
}

void VideoStreamGDNative::set_audio_track(int p_track) {
	audio_track = p_track;
}

/* --- NOTE ResourceFormatLoaderVideoStreamGDNative starts here. ----- */

RES ResourceFormatLoaderVideoStreamGDNative::load(const String &p_path, const String &p_original_path, Error *r_error) {
	FileAccess *f = FileAccess::open(p_path, FileAccess::READ);
	if (!f) {
		if (r_error) {
			*r_error = ERR_CANT_OPEN;
		}
		return RES();
	}
	memdelete(f);
	VideoStreamGDNative *stream = memnew(VideoStreamGDNative);
	stream->set_file(p_path);
	Ref<VideoStreamGDNative> ogv_stream = Ref<VideoStreamGDNative>(stream);
	if (r_error) {
		*r_error = OK;
	}
	return ogv_stream;
}

void ResourceFormatLoaderVideoStreamGDNative::get_recognized_extensions(List<String> *p_extensions) const {
	Map<String, int>::Element *el = VideoDecoderServer::get_instance()->get_extensions().front();
	while (el) {
		p_extensions->push_back(el->key());
		el = el->next();
	}
}

bool ResourceFormatLoaderVideoStreamGDNative::handles_type(const String &p_type) const {
	return ClassDB::is_parent_class(p_type, "VideoStream");
}

String ResourceFormatLoaderVideoStreamGDNative::get_resource_type(const String &p_path) const {
	String el = p_path.get_extension().to_lower();
	if (VideoDecoderServer::get_instance()->get_extensions().has(el)) {
		return "VideoStreamGDNative";
	}
	return "";
}
