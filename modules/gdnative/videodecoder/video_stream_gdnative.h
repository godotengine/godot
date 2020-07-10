/*************************************************************************/
/*  video_stream_gdnative.h                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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
	const godot_videodecoder_interface_gdnative *interface = nullptr;
	String plugin_name = "none";
	Vector<String> supported_extensions;

	VideoDecoderGDNative() {}

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
	bool playing = false;
	bool paused = false;

	Vector2 texture_size;

	void *mix_udata = nullptr;
	AudioMixCallback mix_callback = nullptr;

	int num_channels = -1;
	float time = 0;
	bool seek_backward = false;
	int mix_rate = 0;
	double delay_compensation = 0;

	float *pcm = nullptr;
	int pcm_write_idx = 0;
	int samples_decoded = 0;

	void cleanup();
	void update_texture();

protected:
	String file_name;

	FileAccess *file = nullptr;

	const godot_videodecoder_interface_gdnative *interface = nullptr;
	void *data_struct = nullptr;

public:
	VideoStreamPlaybackGDNative();
	~VideoStreamPlaybackGDNative();

	void set_interface(const godot_videodecoder_interface_gdnative *p_interface);

	bool open_file(const String &p_file);

	virtual void stop() override;
	virtual void play() override;

	virtual bool is_playing() const override;

	virtual void set_paused(bool p_paused) override;
	virtual bool is_paused() const override;

	virtual void set_loop(bool p_enable) override;
	virtual bool has_loop() const override;

	virtual float get_length() const override;

	virtual float get_playback_position() const override;
	virtual void seek(float p_time) override;

	virtual void set_audio_track(int p_idx) override;

	//virtual int mix(int16_t* p_buffer,int p_frames)=0;

	virtual Ref<Texture2D> get_texture() const override;
	virtual void update(float p_delta) override;

	virtual void set_mix_callback(AudioMixCallback p_callback, void *p_userdata) override;
	virtual int get_channels() const override;
	virtual int get_mix_rate() const override;
};

class VideoStreamGDNative : public VideoStream {
	GDCLASS(VideoStreamGDNative, VideoStream);

	String file;
	int audio_track = 0;

protected:
	static void
	_bind_methods();

public:
	void set_file(const String &p_file);
	String get_file();

	virtual void set_audio_track(int p_track) override;
	virtual Ref<VideoStreamPlayback> instance_playback() override;

	VideoStreamGDNative() {}
};

class ResourceFormatLoaderVideoStreamGDNative : public ResourceFormatLoader {
public:
	virtual RES load(const String &p_path, const String &p_original_path = "", Error *r_error = nullptr, bool p_use_sub_threads = false, float *r_progress = nullptr, bool p_no_cache = false);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String &p_type) const;
	virtual String get_resource_type(const String &p_path) const;
};

#endif
