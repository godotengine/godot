/**************************************************************************/
/*  audio_server.h                                                        */
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

#include "core/templates/safe_list.h"
#include "core/variant/variant.h"
#include "servers/audio/audio_bus_layout.h"
#include "servers/audio/audio_effect.h"
#include "servers/audio/audio_filter_sw.h"
#include "servers/audio/audio_frame.h"
#include "servers/audio/audio_server_constants.h"
#include "servers/audio/audio_server_enums.h"
#include "servers/audio/audio_server_types.h" // IWYU pragma: keep. Included to have a dedicated file to move stuff over.

class AudioSample;
class AudioStream;
class AudioStreamPlayback;
class AudioSamplePlayback;
class AudioBusLayout;

class AudioServer : public Object {
	GDCLASS(AudioServer, Object);

public:
	typedef void (*AudioCallback)(void *p_userdata);
	bool cached_volume_db_affects_3d_attenuation = true;

private:
	uint64_t mix_time = 0;
	int mix_size = 0;

	uint32_t buffer_size = 0;
	uint64_t mix_count = 0;
	uint64_t mix_frames = 0;
#ifdef DEBUG_ENABLED
	SafeNumeric<uint64_t> prof_time;
#endif

	float channel_disable_threshold_db = 0.0f;
	uint32_t channel_disable_frames = 0;

	int channel_count = 0;
	int to_mix = 0;

	float playback_speed_scale = 1.0f;

	bool tag_used_audio_streams = false;

#ifdef DEBUG_ENABLED
	bool debug_mute = false;
#endif // DEBUG_ENABLED

	bool input_device_active = false;
	int input_buffer_ofs = 0;

	struct Bus {
		StringName name;
		bool solo = false;
		bool mute = false;
		bool bypass = false;

		bool soloed = false;

		// Each channel is a stereo pair.
		struct Channel {
			bool used = false;
			bool active = false;
			AudioFrame peak_volume = AudioFrame(AuSC::AUDIO_MIN_PEAK_DB, AuSC::AUDIO_MIN_PEAK_DB);
			Vector<AudioFrame> buffer;
			Vector<Ref<AudioEffectInstance>> effect_instances;
			uint64_t last_mix_with_audio = 0;
			Channel() {}
		};

		Vector<Channel> channels;

		struct Effect {
			Ref<AudioEffect> effect;
			bool enabled = false;
#ifdef DEBUG_ENABLED
			uint64_t prof_time = 0;
#endif
		};

		Vector<Effect> effects;
		float volume_db = 0.0f;
		StringName send;
		int index_cache = 0;
	};

	struct AudioStreamPlaybackBusDetails {
		bool bus_active[AuSC::MAX_BUSES_PER_PLAYBACK] = {};
		StringName bus[AuSC::MAX_BUSES_PER_PLAYBACK];
		AudioFrame volume[AuSC::MAX_BUSES_PER_PLAYBACK][AuSC::MAX_CHANNELS_PER_BUS];
	};

	struct AudioStreamPlaybackListNode {
		// The state machine for audio stream playbacks is as follows:
		// 1. The playback is created and added to the playback list in the playing state.
		// 2. The playback is (maybe) paused, and the state is set to FADE_OUT_TO_PAUSE.
		// 2.1. The playback is mixed after being paused, and the audio server thread atomically sets the state to PAUSED after performing a brief fade-out.
		// 3. The playback is (maybe) deleted, and the state is set to FADE_OUT_TO_DELETION.
		// 3.1. The playback is mixed after being deleted, and the audio server thread atomically sets the state to AWAITING_DELETION after performing a brief fade-out.
		// 		NOTE: The playback is not deallocated at this time because allocation and deallocation are not realtime-safe.
		// 4. The playback is removed and deallocated on the main thread using the SafeList maybe_cleanup method.
		enum PlaybackState {
			PAUSED = 0, // Paused. Keep this stream playback around though so it can be restarted.
			PLAYING = 1, // Playing. Fading may still be necessary if volume changes!
			FADE_OUT_TO_PAUSE = 2, // About to pause.
			FADE_OUT_TO_DELETION = 3, // About to stop.
			AWAITING_DELETION = 4,
		};
		// If zero or positive, a place in the stream to seek to during the next mix.
		SafeNumeric<float> setseek;
		SafeNumeric<float> pitch_scale;
		SafeNumeric<float> highshelf_gain;
		SafeNumeric<float> attenuation_filter_cutoff_hz; // This isn't used unless highshelf_gain is nonzero.
		AudioFilterSW::Processor filter_process[8];
		// Updating this ref after the list node is created breaks consistency guarantees, don't do it!
		Ref<AudioStreamPlayback> stream_playback;
		// Playback state determines the fate of a particular AudioStreamListNode during the mix step. Must be atomically replaced.
		std::atomic<PlaybackState> state = AWAITING_DELETION;
		// This data should only ever be modified by an atomic replacement of the pointer.
		std::atomic<AudioStreamPlaybackBusDetails *> bus_details = nullptr;
		// Previous bus details should only be accessed on the audio thread.
		AudioStreamPlaybackBusDetails *prev_bus_details = nullptr;
		// The next few samples are stored here so we have some time to fade audio out if it ends abruptly at the beginning of the next mix.
		AudioFrame lookahead[AuSC::LOOKAHEAD_BUFFER_SIZE];
	};

	SafeList<AudioStreamPlaybackListNode *> playback_list;
	SafeList<AudioStreamPlaybackBusDetails *> bus_details_graveyard;
	void _delete_stream_playback(Ref<AudioStreamPlayback> p_playback);
	void _delete_stream_playback_list_node(AudioStreamPlaybackListNode *p_node);

	void _cleanup_lists();

	// TODO document if this is necessary.
	SafeList<AudioStreamPlaybackBusDetails *> bus_details_graveyard_frame_old;

	Vector<Vector<AudioFrame>> temp_buffer; //temp_buffer for each level
	Vector<AudioFrame> mix_buffer;
	Vector<Bus *> buses;
	HashMap<StringName, Bus *> bus_map;

	void _update_bus_effects(int p_bus);

	static AudioServer *singleton;

	void init_channels_and_buffers();

	void _mix_step();
	void _mix_step_for_channel(AudioFrame *p_out_buf, AudioFrame *p_source_buf, AudioFrame p_vol_start, AudioFrame p_vol_final, float p_attenuation_filter_cutoff_hz, float p_highshelf_gain, AudioFilterSW::Processor *p_processor_l, AudioFilterSW::Processor *p_processor_r);

	// Should only be called on the main thread.
	AudioStreamPlaybackListNode *_find_playback_list_node(Ref<AudioStreamPlayback> p_playback);

	struct CallbackItem {
		AudioCallback callback;
		void *userdata = nullptr;
	};

	SafeList<CallbackItem *> update_callback_list;
	SafeList<CallbackItem *> mix_callback_list;
	SafeList<CallbackItem *> listener_changed_callback_list;

	friend class AudioDriver;
	void _driver_process(int p_frames, int32_t *p_buffer);

	LocalVector<Ref<AudioSamplePlayback>> sample_playback_list;

protected:
	static void _bind_methods();

public:
	_FORCE_INLINE_ int get_channel_count() const {
		switch (get_speaker_mode()) {
			case AuSE::SPEAKER_MODE_STEREO:
				return 1;
			case AuSE::SPEAKER_SURROUND_31:
				return 2;
			case AuSE::SPEAKER_SURROUND_51:
				return 3;
			case AuSE::SPEAKER_SURROUND_71:
				return 4;
		}
		ERR_FAIL_V(1);
	}

	// Do not use from outside audio thread.
	bool thread_has_channel_mix_buffer(int p_bus, int p_buffer) const;
	AudioFrame *thread_get_channel_mix_buffer(int p_bus, int p_buffer);
	int thread_get_mix_buffer_size() const;
	int thread_find_bus_index(const StringName &p_name);

#ifdef DEBUG_ENABLED
	void set_debug_mute(bool p_mute);
	bool get_debug_mute() const;
#endif // DEBUG_ENABLED

	void set_bus_count(int p_count);
	int get_bus_count() const;

	void remove_bus(int p_index);
	void add_bus(int p_at_pos = -1);

	void move_bus(int p_bus, int p_to_pos);

	void set_bus_name(int p_bus, const String &p_name);
	String get_bus_name(int p_bus) const;
	int get_bus_index(const StringName &p_bus_name) const;

	int get_bus_channels(int p_bus) const;

	void set_bus_volume_db(int p_bus, float p_volume_db);
	float get_bus_volume_db(int p_bus) const;

	void set_bus_volume_linear(int p_bus, float p_volume_linear);
	float get_bus_volume_linear(int p_bus) const;

	void set_bus_send(int p_bus, const StringName &p_send);
	StringName get_bus_send(int p_bus) const;

	void set_bus_solo(int p_bus, bool p_enable);
	bool is_bus_solo(int p_bus) const;

	void set_bus_mute(int p_bus, bool p_enable);
	bool is_bus_mute(int p_bus) const;

	void set_bus_bypass_effects(int p_bus, bool p_enable);
	bool is_bus_bypassing_effects(int p_bus) const;

	void add_bus_effect(int p_bus, const Ref<AudioEffect> &p_effect, int p_at_pos = -1);
	void remove_bus_effect(int p_bus, int p_effect);

	int get_bus_effect_count(int p_bus);
	Ref<AudioEffect> get_bus_effect(int p_bus, int p_effect);
	Ref<AudioEffectInstance> get_bus_effect_instance(int p_bus, int p_effect, int p_channel = 0);

	void swap_bus_effects(int p_bus, int p_effect, int p_by_effect);

	void set_bus_effect_enabled(int p_bus, int p_effect, bool p_enabled);
	bool is_bus_effect_enabled(int p_bus, int p_effect) const;

	float get_bus_peak_volume_left_db(int p_bus, int p_channel) const;
	float get_bus_peak_volume_right_db(int p_bus, int p_channel) const;

	bool is_bus_channel_active(int p_bus, int p_channel) const;

	void set_playback_speed_scale(float p_scale);
	float get_playback_speed_scale() const;

	// Convenience method.
	void start_playback_stream(Ref<AudioStreamPlayback> p_playback, const StringName &p_bus, Vector<AudioFrame> p_volume_db_vector, float p_start_time = 0, float p_pitch_scale = 1);
	// Expose all parameters.
	void start_playback_stream(Ref<AudioStreamPlayback> p_playback, const HashMap<StringName, Vector<AudioFrame>> &p_bus_volumes, float p_start_time = 0, float p_pitch_scale = 1, float p_highshelf_gain = 0, float p_attenuation_cutoff_hz = 0);
	void stop_playback_stream(Ref<AudioStreamPlayback> p_playback);

	void set_playback_bus_exclusive(Ref<AudioStreamPlayback> p_playback, const StringName &p_bus, Vector<AudioFrame> p_volumes);
	void set_playback_bus_volumes_linear(Ref<AudioStreamPlayback> p_playback, const HashMap<StringName, Vector<AudioFrame>> &p_bus_volumes);
	void set_playback_all_bus_volumes_linear(Ref<AudioStreamPlayback> p_playback, Vector<AudioFrame> p_volumes);
	void set_playback_pitch_scale(Ref<AudioStreamPlayback> p_playback, float p_pitch_scale);
	void set_playback_paused(Ref<AudioStreamPlayback> p_playback, bool p_paused);
	void set_playback_highshelf_params(Ref<AudioStreamPlayback> p_playback, float p_gain, float p_attenuation_cutoff_hz);

	bool is_playback_active(Ref<AudioStreamPlayback> p_playback);
	float get_playback_position(Ref<AudioStreamPlayback> p_playback);
	bool is_playback_paused(Ref<AudioStreamPlayback> p_playback);

	uint64_t get_mix_count() const;
	uint64_t get_mixed_frames() const;

	String get_driver_name() const;

	void notify_listener_changed();

	virtual void init();
	virtual void finish();
	virtual void update();
	virtual void load_default_bus_layout();

	/* MISC config */

	virtual void lock();
	virtual void unlock();

	virtual AuSE::SpeakerMode get_speaker_mode() const;
	virtual float get_mix_rate() const;
	virtual float get_input_mix_rate() const;

	virtual float read_output_peak_db() const;

	static AudioServer *get_singleton();

	virtual double get_output_latency() const;
	virtual double get_time_to_next_mix() const;
	virtual double get_time_since_last_mix() const;

	void add_listener_changed_callback(AudioCallback p_callback, void *p_userdata);
	void remove_listener_changed_callback(AudioCallback p_callback, void *p_userdata);

	void add_update_callback(AudioCallback p_callback, void *p_userdata);
	void remove_update_callback(AudioCallback p_callback, void *p_userdata);

	void add_mix_callback(AudioCallback p_callback, void *p_userdata);
	void remove_mix_callback(AudioCallback p_callback, void *p_userdata);

	void set_bus_layout(const Ref<AudioBusLayout> &p_bus_layout);
	Ref<AudioBusLayout> generate_bus_layout() const;

	PackedStringArray get_output_device_list();
	String get_output_device();
	void set_output_device(const String &p_name);

	PackedStringArray get_input_device_list();
	String get_input_device();
	void set_input_device(const String &p_name);
	Error set_input_device_active(bool p_is_active);
	int get_input_frames_available();
	int get_input_buffer_length_frames();
	PackedVector2Array get_input_frames(int p_frames);

	void set_enable_tagging_used_audio_streams(bool p_enable);

#ifdef TOOLS_ENABLED
	virtual void get_argument_options(const StringName &p_function, int p_idx, List<String> *r_options) const override;
#endif

	AuSE::PlaybackType get_default_playback_type() const;

	bool is_stream_registered_as_sample(const Ref<AudioStream> &p_stream);
	void register_stream_as_sample(const Ref<AudioStream> &p_stream);
	void unregister_stream_as_sample(const Ref<AudioStream> &p_stream);
	void register_sample(const Ref<AudioSample> &p_sample);
	void unregister_sample(const Ref<AudioSample> &p_sample);
	void start_sample_playback(const Ref<AudioSamplePlayback> &p_playback);
	void stop_sample_playback(const Ref<AudioSamplePlayback> &p_playback);
	void set_sample_playback_pause(const Ref<AudioSamplePlayback> &p_playback, bool p_paused);
	bool is_sample_playback_active(const Ref<AudioSamplePlayback> &p_playback);
	double get_sample_playback_position(const Ref<AudioSamplePlayback> &p_playback);
	void update_sample_playback_pitch_scale(const Ref<AudioSamplePlayback> &p_playback, float p_pitch_scale = 0.0f);

	AudioServer();
	virtual ~AudioServer();
};

VARIANT_ENUM_CAST_EXT(AuSE::SpeakerMode, AudioServer::SpeakerMode);
VARIANT_ENUM_CAST_EXT(AuSE::PlaybackType, AudioServer::PlaybackType);
