/**************************************************************************/
/*  audio_stream_player.h                                                 */
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

#pragma once

#include "scene/main/node.h"
#include "servers/audio/audio_server.h"

struct AudioFrame;
class AudioStream;
class AudioStreamPlayback;
class AudioStreamPlayerInternal;

class AudioStreamPlayer : public Node {
	GDCLASS(AudioStreamPlayer, Node);

public:
	enum ChannelFlags {
		AUDIO_CHANNEL_LEFT = 1 << 0,
		AUDIO_CHANNEL_RIGHT = 1 << 1,
		AUDIO_CHANNEL_CENTER = 1 << 2,
		AUDIO_CHANNEL_LFE = 1 << 3,
		AUDIO_CHANNEL_REAR_LEFT = 1 << 4,
		AUDIO_CHANNEL_REAR_RIGHT = 1 << 5,
		AUDIO_CHANNEL_SIDE_LEFT = 1 << 6,
		AUDIO_CHANNEL_SIDE_RIGHT = 1 << 7,
	};

private:
	AudioStreamPlayerInternal *internal = nullptr;

	BitField<ChannelFlags> output_channels = AUDIO_CHANNEL_LEFT | AUDIO_CHANNEL_RIGHT; // Default to stereo

	bool downmix = true;

	void _set_playing(bool p_enable);
	bool _is_active() const;

	Vector<AudioFrame> _get_volume_vector();

protected:
	void _validate_property(PropertyInfo &p_property) const;
	void _notification(int p_what);
	static void _bind_methods();

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

#ifndef DISABLE_DEPRECATED
	bool _is_autoplay_enabled_bind_compat_86907();
	static void _bind_compatibility_methods();
#endif // DISABLE_DEPRECATED

public:
	void set_stream(Ref<AudioStream> p_stream);
	Ref<AudioStream> get_stream() const;

	void set_volume_db(float p_volume);
	float get_volume_db() const;

	void set_volume_linear(float p_volume);
	float get_volume_linear() const;

	void set_pitch_scale(float p_pitch_scale);
	float get_pitch_scale() const;

	void set_max_polyphony(int p_max_polyphony);
	int get_max_polyphony() const;

	void play(float p_from_pos = 0.0);
	void seek(float p_seconds);
	void stop();
	bool is_playing() const;
	float get_playback_position();

	void set_bus(const StringName &p_bus);
	StringName get_bus() const;

	void set_autoplay(bool p_enable);
	bool is_autoplay_enabled() const;

	void set_output_channels(BitField<ChannelFlags> p_channels);
	BitField<ChannelFlags> get_output_channels() const;

	void set_downmix(bool p_enable);
	bool is_downmix_enabled() const;

	void set_stream_paused(bool p_pause);
	bool get_stream_paused() const;

	bool has_stream_playback();
	Ref<AudioStreamPlayback> get_stream_playback();

	AudioServer::PlaybackType get_playback_type() const;
	void set_playback_type(AudioServer::PlaybackType p_playback_type);

	AudioStreamPlayer();
	~AudioStreamPlayer();
};

VARIANT_BITFIELD_CAST(AudioStreamPlayer::ChannelFlags);
