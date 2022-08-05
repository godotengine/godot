/*************************************************************************/
/*  video_stream_gdnative.h                                              */
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

#ifndef VIDEO_STREAM_GDNATIVE_H
#define VIDEO_STREAM_GDNATIVE_H

#include "../gdnative.h"
#include "core/os/file_access.h"
#include "scene/resources/texture.h"
#include "scene/resources/video_stream.h"

struct VideoDecoderGDNative {
	const godot_videodecoder_interface_gdnative *interface;
	String plugin_name;
	Vector<String> supported_extensions;

	VideoDecoderGDNative() :
			interface(nullptr),
			plugin_name("none") {}

	VideoDecoderGDNative(const godot_videodecoder_interface_gdnative *p_interface) :
			interface(p_interface),
			plugin_name(p_interface->get_plugin_name()) {
		_get_supported_extensions();
	}

private:
	void _get_supported_extensions() {
		supported_extensions.clear();
		int num_ext;
		const char **supported_ext = interface->get_supported_extensions(&num_ext);
		for (int i = 0; i < num_ext; i++) {
			supported_extensions.push_back(supported_ext[i]);
		}
	}
};

class VideoDecoderServer {
private:
	Vector<VideoDecoderGDNative *> decoders;
	Map<String, int> extensions;

	static VideoDecoderServer *instance;

public:
	static VideoDecoderServer *get_instance() {
		return instance;
	}

	const Map<String, int> &get_extensions() {
		return extensions;
	}

	void register_decoder_interface(const godot_videodecoder_interface_gdnative *p_interface) {
		VideoDecoderGDNative *decoder = memnew(VideoDecoderGDNative(p_interface));
		int index = decoders.size();
		for (int i = 0; i < decoder->supported_extensions.size(); i++) {
			extensions[decoder->supported_extensions[i]] = index;
		}
		decoders.push_back(decoder);
	}

	VideoDecoderGDNative *get_decoder(const String &extension) {
		if (extensions.size() == 0 || !extensions.has(extension)) {
			return nullptr;
		}
		return decoders[extensions[extension]];
	}

	VideoDecoderServer() {
		instance = this;
	}

	~VideoDecoderServer() {
		for (int i = 0; i < decoders.size(); i++) {
			memdelete(decoders[i]);
		}
		decoders.clear();
		instance = nullptr;
	}
};

class VideoStreamPlaybackGDNative : public VideoStreamPlayback {
	GDCLASS(VideoStreamPlaybackGDNative, VideoStreamPlayback);

	Ref<ImageTexture> texture;
	bool playing;
	bool paused;

	Vector2 texture_size;

	void *mix_udata;
	AudioMixCallback mix_callback;

	int num_channels;
	float time;
	bool seek_backward;
	int mix_rate;
	double delay_compensation;

	float *pcm;
	int pcm_write_idx;
	int samples_decoded;

	void cleanup();
	void update_texture();

protected:
	String file_name;

	FileAccess *file;

	const godot_videodecoder_interface_gdnative *interface;
	void *data_struct;

public:
	VideoStreamPlaybackGDNative();
	~VideoStreamPlaybackGDNative();

	void set_interface(const godot_videodecoder_interface_gdnative *p_interface);

	bool open_file(const String &p_file);

	virtual void stop();
	virtual void play();

	virtual bool is_playing() const;

	virtual void set_paused(bool p_paused);
	virtual bool is_paused() const;

	virtual void set_loop(bool p_enable);
	virtual bool has_loop() const;

	virtual float get_length() const;

	virtual float get_playback_position() const;
	virtual void seek(float p_time);

	virtual void set_audio_track(int p_idx);

	//virtual int mix(int16_t* p_buffer,int p_frames)=0;

	virtual Ref<Texture> get_texture() const;
	virtual void update(float p_delta);

	virtual void set_mix_callback(AudioMixCallback p_callback, void *p_userdata);
	virtual int get_channels() const;
	virtual int get_mix_rate() const;
};

class VideoStreamGDNative : public VideoStream {
	GDCLASS(VideoStreamGDNative, VideoStream);

	String file;
	int audio_track;

protected:
	static void
	_bind_methods();

public:
	void set_file(const String &p_file);
	String get_file();

	virtual void set_audio_track(int p_track);
	virtual Ref<VideoStreamPlayback> instance_playback();

	VideoStreamGDNative() {}
};

class ResourceFormatLoaderVideoStreamGDNative : public ResourceFormatLoader {
public:
	virtual RES load(const String &p_path, const String &p_original_path = "", Error *r_error = nullptr, bool p_no_subresource_cache = false);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String &p_type) const;
	virtual String get_resource_type(const String &p_path) const;
};

#endif // VIDEO_STREAM_GDNATIVE_H
