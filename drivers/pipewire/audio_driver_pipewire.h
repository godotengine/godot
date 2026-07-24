/**************************************************************************/
/*  audio_driver_pipewire.h                                               */
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

#ifdef PIPEWIRE_ENABLED

#include "servers/audio/audio_server.h"

#ifdef SOWRAP_ENABLED
#include "drivers/pipewire/pipewire-so_wrap.h" // IWYU pragma: keep. For PipeWire transitive includes.
#else
#include <pipewire/pipewire.h>
#endif
#include <spa/param/audio/raw.h>

class AudioDriverPipeWire : public AudioDriver {
	struct PipeWireNode {
		const uint32_t id = 0;
		const CharString media_class;
		const CharString node_name;
	};

	static constexpr const char *DEFAULT_DEVICE = "Default";

	static constexpr const char *MEDIA_CLASS_SINK = "Audio/Sink";
	static constexpr const char *MEDIA_CLASS_SOURCE = "Audio/Source";

	static constexpr const spa_audio_channel OUTPUT_POSITION[] = {
		SPA_AUDIO_CHANNEL_FL, SPA_AUDIO_CHANNEL_FR,
		SPA_AUDIO_CHANNEL_FC, SPA_AUDIO_CHANNEL_LFE,
		SPA_AUDIO_CHANNEL_RL, SPA_AUDIO_CHANNEL_RR,
		SPA_AUDIO_CHANNEL_SL, SPA_AUDIO_CHANNEL_SR
	};
	static constexpr const uint32_t OUTPUT_CHANNELS = sizeof(OUTPUT_POSITION) / sizeof(spa_audio_channel);

	static constexpr const spa_audio_channel INPUT_POSITION[] = {
		SPA_AUDIO_CHANNEL_FL, SPA_AUDIO_CHANNEL_FR
	};
	static constexpr const uint32_t INPUT_CHANNELS = sizeof(INPUT_POSITION) / sizeof(spa_audio_channel);

	static const struct pw_core_events core_events;
	static const struct pw_registry_events registry_events;
	static const struct pw_stream_events output_stream_events;
	static const struct pw_stream_events input_stream_events;

	pw_thread_loop *loop = nullptr;
	pw_context *context = nullptr;
	pw_core *core = nullptr;
	pw_registry *registry = nullptr;
	pw_stream *output_stream = nullptr;
	pw_stream *input_stream = nullptr;

	spa_hook core_listener = {};
	spa_hook registry_listener = {};
	spa_hook output_stream_listener = {};
	spa_hook input_stream_listener = {};

	uint32_t pending_id = 0;
	int pending_seq = 0;

	unsigned int mix_rate = 0;
	unsigned int buffer_frames = 0;

	Vector<PipeWireNode> pw_nodes;

	uint32_t output_channels = 0;
	uint32_t output_buffer_channels = 0;
	Vector<int32_t> output_buffer;

	uint32_t input_channels = 0;

	static void on_core_done(void *data, uint32_t id, int seq);
	static void on_registry_event_global(void *data, uint32_t id, uint32_t permissions, const char *type, uint32_t version, const struct spa_dict *props);
	static void on_registry_event_global_remove(void *data, uint32_t id);

	static void on_output_stream_destroy(void *data);
	static void on_output_stream_state_changed(void *data, enum pw_stream_state old, enum pw_stream_state state, const char *error);
	static void on_output_stream_param_changed(void *data, uint32_t id, const struct spa_pod *param);
	static void on_output_stream_process(void *data);

	static void on_input_stream_destroy(void *data);
	static void on_input_stream_state_changed(void *data, enum pw_stream_state old, enum pw_stream_state state, const char *error);
	static void on_input_stream_param_changed(void *data, uint32_t id, const struct spa_pod *param);
	static void on_input_stream_process(void *data);

	void sync_wait();

	const PipeWireNode *get_pw_node(const String &p_name) const;

	void init_output_stream();
	void init_input_stream();

public:
	virtual const char *get_name() const override {
		return "PipeWire";
	}

	virtual Error init() override;
	virtual void start() override;
	virtual int get_mix_rate() const override;
	virtual SpeakerMode get_speaker_mode() const override;
	virtual float get_latency() override;

	virtual void lock() override;
	virtual void unlock() override;
	virtual void finish() override;

	virtual PackedStringArray get_output_device_list() override;
	virtual String get_output_device() override;
	virtual void set_output_device(const String &p_name) override;

	virtual Error input_start() override;
	virtual Error input_stop() override;

	virtual PackedStringArray get_input_device_list() override;
	virtual String get_input_device() override;
	virtual void set_input_device(const String &p_name) override;

	AudioDriverPipeWire() {}
	~AudioDriverPipeWire() {}
};

#endif // PIPEWIRE_ENABLED
