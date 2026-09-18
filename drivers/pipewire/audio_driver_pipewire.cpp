/**************************************************************************/
/*  audio_driver_pipewire.cpp                                             */
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

#include "audio_driver_pipewire.h"

#ifdef PIPEWIRE_ENABLED

#include "core/config/engine.h"
#include "core/config/project_settings.h"
#include "core/version.h"

GODOT_GCC_WARNING_PUSH
GODOT_GCC_WARNING_IGNORE("-Wmissing-field-initializers")
GODOT_CLANG_WARNING_PUSH
GODOT_CLANG_WARNING_IGNORE("-Wmissing-field-initializers")

#include <spa/param/format.h>
#include <spa/pod/builder.h>
#include <spa/pod/parser.h>
#include <spa/utils/keys.h>

const struct pw_core_events AudioDriverPipeWire::core_events = {
	.version = PW_VERSION_CORE_EVENTS,
	.done = on_core_done,
};

const struct pw_registry_events AudioDriverPipeWire::registry_events = {
	.version = PW_VERSION_REGISTRY_EVENTS,
	.global = on_registry_event_global,
	.global_remove = on_registry_event_global_remove,
};

const struct pw_stream_events AudioDriverPipeWire::output_stream_events = {
	.version = PW_VERSION_STREAM_EVENTS,
	.destroy = on_output_stream_destroy,
	.state_changed = on_output_stream_state_changed,
	.param_changed = on_output_stream_param_changed,
	.process = on_output_stream_process,
};

const struct pw_stream_events AudioDriverPipeWire::input_stream_events = {
	.version = PW_VERSION_STREAM_EVENTS,
	.destroy = on_input_stream_destroy,
	.state_changed = on_input_stream_state_changed,
	.param_changed = on_input_stream_param_changed,
	.process = on_input_stream_process,
};

void AudioDriverPipeWire::on_core_done(void *data, uint32_t id, int seq) {
	AudioDriverPipeWire *ad = static_cast<AudioDriverPipeWire *>(data);

	if (id == ad->pending_id && seq == ad->pending_seq) {
		pw_thread_loop_signal(ad->loop, false);
	}
}

void AudioDriverPipeWire::on_registry_event_global(void *data, uint32_t id, uint32_t permissions, const char *type, uint32_t version, const struct spa_dict *props) {
	AudioDriverPipeWire *ad = static_cast<AudioDriverPipeWire *>(data);

	if (strcmp(type, PW_TYPE_INTERFACE_Node) != 0) {
		return;
	}

	const char *media_class = spa_dict_lookup(props, SPA_KEY_MEDIA_CLASS);
	if (media_class == nullptr) {
		return;
	}
	if (strcmp(media_class, MEDIA_CLASS_SINK) && strcmp(media_class, MEDIA_CLASS_SOURCE)) {
		return;
	}

	const char *node_name = spa_dict_lookup(props, PW_KEY_NODE_NAME);
	struct PipeWireNode node = {
		.id = id,
		.media_class = media_class,
		.node_name = node_name,
	};
	ad->pw_nodes.push_back(node);

	ad->sync_wait();
}

void AudioDriverPipeWire::on_registry_event_global_remove(void *data, uint32_t id) {
	AudioDriverPipeWire *ad = static_cast<AudioDriverPipeWire *>(data);

	for (int i = 0; i < ad->pw_nodes.size(); i++) {
		const struct PipeWireNode &node = ad->pw_nodes[i];
		if (node.id != id) {
			continue;
		}
		if (node.media_class == MEDIA_CLASS_SINK) {
			if (ad->get_output_device() == node.node_name.get_data()) {
				ad->set_output_device(DEFAULT_DEVICE);
			}
		} else if (node.media_class == MEDIA_CLASS_SOURCE) {
			if (ad->get_input_device() == node.node_name.get_data()) {
				ad->set_input_device(DEFAULT_DEVICE);
			}
		}
		ad->pw_nodes.remove_at(i);
		return;
	}
}

void AudioDriverPipeWire::on_output_stream_destroy(void *data) {
	AudioDriverPipeWire *ad = static_cast<AudioDriverPipeWire *>(data);
	ad->output_stream = nullptr;
}

void AudioDriverPipeWire::on_output_stream_state_changed(void *data, enum pw_stream_state old, enum pw_stream_state state, const char *error) {
	if (error) {
		ERR_PRINT(vformat("PipeWire output stream error: %s", String::utf8(error)));
	}
	print_verbose(vformat("PipeWire output stream state: %s->%s", pw_stream_state_as_string(old), pw_stream_state_as_string(state)));
}

void AudioDriverPipeWire::on_output_stream_param_changed(void *data, uint32_t id, const struct spa_pod *param) {
	AudioDriverPipeWire *ad = static_cast<AudioDriverPipeWire *>(data);

	if (id == SPA_PARAM_Invalid || param == nullptr) {
		return;
	}

	if (id == SPA_PARAM_Format) {
		if (spa_pod_parse_object(param,
					SPA_TYPE_OBJECT_Format, nullptr,
					SPA_FORMAT_AUDIO_channels, SPA_POD_OPT_Int(&ad->output_channels)) < 0) {
			ERR_PRINT("PipeWire: Failed to get output channel count.");
			ad->output_channels = 2;
		}
		ad->output_buffer_channels = ad->get_total_channels_by_speaker_mode(ad->get_speaker_mode());
		uint32_t buffer_size = ad->buffer_frames * ad->output_buffer_channels * sizeof(int16_t);
		ad->output_buffer.resize(buffer_size);
	}
}

void AudioDriverPipeWire::on_output_stream_process(void *data) {
	struct pw_buffer *b;
	struct spa_data *dst_data;
	AudioDriverPipeWire *ad = static_cast<AudioDriverPipeWire *>(data);

	ad->start_counting_ticks();

	ERR_FAIL_NULL_MSG((b = pw_stream_dequeue_buffer(ad->output_stream)), "Out of buffer.");
	dst_data = &b->buffer->datas[0]; // codespell:ignore datas

	uint32_t stride = sizeof(int16_t) * ad->output_channels;
	uint32_t n_frames = dst_data->maxsize / stride;

	if (b->requested) {
		n_frames = SPA_MIN(b->requested, n_frames);
	}

	n_frames = SPA_MIN(ad->buffer_frames, n_frames);
	ad->audio_server_process(n_frames, ad->output_buffer.ptrw());

	if (ad->output_channels == 1) {
		const int32_t *src = ad->output_buffer.ptr();
		int16_t *dst = (int16_t *)dst_data->data;
		for (uint32_t i = 0; i < n_frames; i++) {
			int32_t l = *src++ >> 16;
			int32_t r = *src++ >> 16;
			*dst++ = (l + r) / 2;
		}
	} else {
		for (uint32_t i = 0; i < n_frames; i++) {
			const int32_t *src = ad->output_buffer.ptr() + i * ad->output_buffer_channels;
			int16_t *dst = (int16_t *)dst_data->data + i * ad->output_channels;
			for (uint32_t j = 0; j < ad->output_buffer_channels; j++) {
				if (j < ad->output_channels) {
					dst[j] = src[j] >> 16;
				}
			}
		}
	}

	dst_data->chunk->offset = 0;
	dst_data->chunk->stride = stride;
	dst_data->chunk->size = n_frames * stride;

	b->size = n_frames;
	pw_stream_queue_buffer(ad->output_stream, b);

	ad->stop_counting_ticks();
}

void AudioDriverPipeWire::on_input_stream_destroy(void *data) {
	AudioDriverPipeWire *ad = static_cast<AudioDriverPipeWire *>(data);
	ad->input_stream = nullptr;
}

void AudioDriverPipeWire::on_input_stream_state_changed(void *data, enum pw_stream_state old, enum pw_stream_state state, const char *error) {
	if (error) {
		ERR_PRINT(vformat("PipeWire input stream error: %s", String::utf8(error)));
	}
	print_verbose(vformat("PipeWire input stream state: %s->%s", pw_stream_state_as_string(old), pw_stream_state_as_string(state)));
}

void AudioDriverPipeWire::on_input_stream_param_changed(void *data, uint32_t id, const struct spa_pod *param) {
	AudioDriverPipeWire *ad = static_cast<AudioDriverPipeWire *>(data);

	if (id == SPA_PARAM_Invalid || param == nullptr) {
		return;
	}

	if (id == SPA_PARAM_Format) {
		if (spa_pod_parse_object(param,
					SPA_TYPE_OBJECT_Format, nullptr,
					SPA_FORMAT_AUDIO_channels, SPA_POD_OPT_Int(&ad->input_channels)) < 0) {
			ERR_PRINT("PipeWire: Failed to get input channel count.");
			ad->input_channels = 2;
		}
	}
}

void AudioDriverPipeWire::on_input_stream_process(void *data) {
	struct pw_buffer *b;
	struct spa_data *src_data;
	AudioDriverPipeWire *ad = static_cast<AudioDriverPipeWire *>(data);

	ERR_FAIL_NULL_MSG((b = pw_stream_dequeue_buffer(ad->input_stream)), "Out of buffer.");
	src_data = &b->buffer->datas[0]; // codespell:ignore datas

	uint32_t stride = src_data->chunk->stride;
	uint32_t n_frames = src_data->chunk->size / stride;

	uint32_t max_samples = src_data->maxsize / sizeof(int16_t);
	if (ad->input_buffer.size() != max_samples) {
		ad->input_buffer.resize(max_samples);
		ad->input_position = 0;
		ad->input_size = 0;
	}

	for (uint32_t i = 0; i < n_frames; i++) {
		int16_t *src = (int16_t *)src_data->data + i * ad->input_channels;
		for (uint32_t j = 0; j < INPUT_CHANNELS; j++) {
			int32_t sample = int32_t(*src++) << 16;
			ad->input_buffer_write(sample);
			if (ad->input_channels == 1) {
				ad->input_buffer_write(sample);
				break;
			}
		}
	}

	pw_stream_queue_buffer(ad->input_stream, b);
}

void AudioDriverPipeWire::sync_wait() {
	ERR_FAIL_NULL(loop);
	pending_id = PW_ID_CORE;
	pending_seq = pw_core_sync(core, pending_id, pending_seq);
	if (!pw_thread_loop_in_thread(loop)) {
		pw_thread_loop_wait(loop);
	}
}

const AudioDriverPipeWire::PipeWireNode *AudioDriverPipeWire::get_pw_node(const String &p_name) const {
	for (int i = 0; i < pw_nodes.size(); i++) {
		const PipeWireNode *node = &pw_nodes.ptr()[i];
		if (p_name == node->node_name.get_data()) {
			return node;
		}
	}
	return nullptr;
}

void AudioDriverPipeWire::init_output_stream() {
	struct pw_properties *props = pw_properties_new(
			PW_KEY_MEDIA_TYPE, "Audio",
			PW_KEY_MEDIA_CATEGORY, "Playback",
			PW_KEY_MEDIA_ROLE, "Music",
			NULL);
	pw_properties_setf(props, PW_KEY_NODE_LATENCY, "%u/%u", buffer_frames, mix_rate);
	output_stream = pw_stream_new(core, "Sound", props);
	pw_stream_add_listener(output_stream, &output_stream_listener, &output_stream_events, this);
}

void AudioDriverPipeWire::init_input_stream() {
	struct pw_properties *props = pw_properties_new(
			PW_KEY_MEDIA_TYPE, "Audio",
			PW_KEY_MEDIA_CATEGORY, "Capture",
			PW_KEY_MEDIA_ROLE, "Music",
			NULL);
	input_stream = pw_stream_new(core, "Record", props);
	pw_stream_add_listener(input_stream, &input_stream_listener, &input_stream_events, this);
}

Error AudioDriverPipeWire::init() {
#ifdef SOWRAP_ENABLED
#ifdef DEBUG_ENABLED
	int dylibloader_verbose = 1;
#else
	int dylibloader_verbose = 0;
#endif // DEBUG_ENABLED
	if (initialize_pipewire(dylibloader_verbose)) {
		return ERR_CANT_OPEN;
	}

	if (pw_check_library_version_dylibloader_wrapper_pipewire) {
		if (!pw_check_library_version(PW_MAJOR, PW_MINOR, PW_MICRO)) {
			print_verbose("Unsupported PipeWire library version!");
			return ERR_CANT_OPEN;
		}
	} else {
		print_verbose("Unable to check PipeWire library version!");
		return ERR_CANT_OPEN;
	}
#endif // SOWRAP_ENABLED

	mix_rate = _get_configured_mix_rate();
	int latency = Engine::get_singleton()->get_audio_output_latency();
	buffer_frames = Math::closest_power_of_2(mix_rate * latency / 1000);

	pw_init(nullptr, nullptr);
	loop = pw_thread_loop_new("", nullptr);
	ERR_FAIL_NULL_V(loop, ERR_CANT_OPEN);

	context = pw_context_new(pw_thread_loop_get_loop(loop), nullptr, 0);
	ERR_FAIL_NULL_V(context, ERR_CANT_OPEN);

	String context_name;
	if (Engine::get_singleton()->is_editor_hint()) {
		context_name = GODOT_VERSION_NAME " Editor";
	} else {
		context_name = GLOBAL_GET("application/config/name");
		if (context_name.is_empty()) {
			context_name = GODOT_VERSION_NAME " Project";
		}
	}
	pw_properties *props = pw_properties_new(PW_KEY_APP_NAME, context_name.utf8().ptr(), NULL);
	ERR_FAIL_NULL_V(props, ERR_CANT_OPEN);

	core = pw_context_connect(context, props, 0);
	if (core == nullptr) {
		print_verbose("PipeWire: Failed to connect to PipeWire instance.");
		return ERR_CANT_OPEN;
	}
	pw_core_add_listener(core, &core_listener, &core_events, this);

	registry = pw_core_get_registry(core, PW_VERSION_REGISTRY, 0);
	ERR_FAIL_NULL_V(registry, ERR_CANT_OPEN);
	pw_registry_add_listener(registry, &registry_listener, &registry_events, this);

	ERR_FAIL_COND_V(pw_thread_loop_start(loop) < 0, ERR_CANT_OPEN);

	pw_thread_loop_lock(loop);
	sync_wait();
	init_output_stream();
	init_input_stream();
	pw_thread_loop_unlock(loop);

	return OK;
}

void AudioDriverPipeWire::start() {
	ERR_FAIL_NULL(output_stream);
	lock();
	pw_stream_flags flags = pw_stream_flags(PW_STREAM_FLAG_AUTOCONNECT | PW_STREAM_FLAG_MAP_BUFFERS | PW_STREAM_FLAG_RT_PROCESS);
	const struct spa_pod *params[1];
	uint32_t n_params = 0;
	uint8_t buffer[1024];
	struct spa_pod_builder b = SPA_POD_BUILDER_INIT(buffer, sizeof(buffer));
	params[n_params++] = (spa_pod *)spa_pod_builder_add_object(&b,
			SPA_TYPE_OBJECT_Format, SPA_PARAM_EnumFormat,
			SPA_FORMAT_mediaType, SPA_POD_Id(SPA_MEDIA_TYPE_audio),
			SPA_FORMAT_mediaSubtype, SPA_POD_Id(SPA_MEDIA_SUBTYPE_raw),
			SPA_FORMAT_AUDIO_format, SPA_POD_Id(SPA_AUDIO_FORMAT_S16),
			SPA_FORMAT_AUDIO_rate, SPA_POD_Int(mix_rate),
			SPA_FORMAT_AUDIO_position, SPA_POD_Array(sizeof(uint32_t), SPA_TYPE_Id, OUTPUT_CHANNELS, OUTPUT_POSITION));
	pw_stream_connect(output_stream, PW_DIRECTION_OUTPUT, PW_ID_ANY, flags, params, n_params);
	unlock();
}

int AudioDriverPipeWire::get_mix_rate() const {
	return mix_rate;
}

AudioDriver::SpeakerMode AudioDriverPipeWire::get_speaker_mode() const {
	uint32_t channels = output_channels;
	if (channels % 2) {
		channels += 1;
	}
	channels = SPA_CLAMP(channels, (uint32_t)2, OUTPUT_CHANNELS);
	return get_speaker_mode_by_total_channels(channels);
}

float AudioDriverPipeWire::get_latency() {
	float latency = 0.0;
	if (output_stream == nullptr) {
		return latency;
	}
	if (pw_stream_get_state(output_stream, nullptr) != PW_STREAM_STATE_STREAMING) {
		return latency;
	}
	pw_time time;
	pw_stream_get_time_n(output_stream, &time, sizeof(time));
	std::chrono::time_point<std::chrono::steady_clock> now = std::chrono::steady_clock::now();
	// Elapsed time in nanoseconds.
	int64_t elapsed = now.time_since_epoch().count() - time.now;
	// Graph delay in seconds.
	float delay = time.delay * time.rate.num / (float)time.rate.denom;
	latency += delay - elapsed / (float)SPA_NSEC_PER_SEC; // graph latency
	latency += time.buffered / (float)mix_rate; // buffered latency
	latency += time.queued / (float)mix_rate; // queued latency
	return latency;
}

void AudioDriverPipeWire::lock() {
	ERR_FAIL_NULL(loop);
	pw_thread_loop_lock(loop);
}

void AudioDriverPipeWire::unlock() {
	ERR_FAIL_NULL(loop);
	pw_thread_loop_unlock(loop);
}

void AudioDriverPipeWire::finish() {
	if (loop) {
		pw_thread_loop_lock(loop);
	}
	if (input_stream) {
		pw_stream_disconnect(input_stream);
		spa_hook_remove(&input_stream_listener);
		input_stream = nullptr;
	}
	if (output_stream) {
		pw_stream_disconnect(output_stream);
		spa_hook_remove(&output_stream_listener);
		output_stream = nullptr;
	}
	if (registry) {
		pw_proxy_destroy((pw_proxy *)registry);
		spa_hook_remove(&registry_listener);
		registry = nullptr;
	}
	if (core) {
		pw_core_disconnect(core);
		spa_hook_remove(&core_listener);
		core = nullptr;
	}
	if (context) {
		pw_context_destroy(context);
		context = nullptr;
	}
	if (loop) {
		pw_thread_loop_unlock(loop);
		pw_thread_loop_destroy(loop);
		loop = nullptr;
		pw_deinit();
	}
}

PackedStringArray AudioDriverPipeWire::get_output_device_list() {
	PackedStringArray devices;
	lock();
	devices.push_back(DEFAULT_DEVICE);
	for (const PipeWireNode &node : pw_nodes) {
		if (node.media_class == MEDIA_CLASS_SINK) {
			devices.push_back(node.node_name.get_data());
		}
	}
	unlock();
	return devices;
}

String AudioDriverPipeWire::get_output_device() {
	const char *device = DEFAULT_DEVICE;
	ERR_FAIL_NULL_V(output_stream, device);
	lock();
	const struct pw_properties *props = pw_stream_get_properties(output_stream);
	const char *target_object = pw_properties_get(props, PW_KEY_TARGET_OBJECT);
	if (target_object) {
		device = target_object;
	}
	unlock();
	return device;
}

void AudioDriverPipeWire::set_output_device(const String &p_name) {
	const char *target_object = nullptr;
	ERR_FAIL_NULL(output_stream);
	lock();
	if (p_name != DEFAULT_DEVICE) {
		const PipeWireNode *node = get_pw_node(p_name);
		if (node && node->media_class == MEDIA_CLASS_SINK) {
			target_object = node->node_name.ptr();
		}
	}
	struct spa_dict_item items[1];
	items[0] = SPA_DICT_ITEM_INIT(PW_KEY_TARGET_OBJECT, target_object);
	spa_dict dict = SPA_DICT_INIT(items, 1);
	pw_stream_update_properties(output_stream, &dict);
	if (pw_stream_get_state(output_stream, nullptr) == PW_STREAM_STATE_STREAMING) {
		pw_stream_disconnect(output_stream);
		start();
	}
	unlock();
}

Error AudioDriverPipeWire::input_start() {
	ERR_FAIL_NULL_V(input_stream, ERR_CANT_OPEN);
	lock();
	pw_stream_flags flags = pw_stream_flags(PW_STREAM_FLAG_AUTOCONNECT | PW_STREAM_FLAG_MAP_BUFFERS | PW_STREAM_FLAG_RT_PROCESS);
	const struct spa_pod *params[1];
	uint32_t n_params = 0;
	uint8_t buffer[1024];
	struct spa_pod_builder b = SPA_POD_BUILDER_INIT(buffer, sizeof(buffer));
	params[n_params++] = (spa_pod *)spa_pod_builder_add_object(&b,
			SPA_TYPE_OBJECT_Format, SPA_PARAM_EnumFormat,
			SPA_FORMAT_mediaType, SPA_POD_Id(SPA_MEDIA_TYPE_audio),
			SPA_FORMAT_mediaSubtype, SPA_POD_Id(SPA_MEDIA_SUBTYPE_raw),
			SPA_FORMAT_AUDIO_format, SPA_POD_Id(SPA_AUDIO_FORMAT_S16),
			SPA_FORMAT_AUDIO_rate, SPA_POD_Int(mix_rate),
			SPA_FORMAT_AUDIO_position, SPA_POD_Array(sizeof(uint32_t), SPA_TYPE_Id, INPUT_CHANNELS, INPUT_POSITION));
	pw_stream_connect(input_stream, PW_DIRECTION_INPUT, PW_ID_ANY, flags, params, n_params);
	unlock();
	return OK;
}

Error AudioDriverPipeWire::input_stop() {
	if (input_stream) {
		lock();
		pw_stream_disconnect(input_stream);
		unlock();
	}
	return OK;
}

PackedStringArray AudioDriverPipeWire::get_input_device_list() {
	PackedStringArray devices;
	lock();
	devices.push_back(DEFAULT_DEVICE);
	for (const PipeWireNode &node : pw_nodes) {
		if (node.media_class == MEDIA_CLASS_SOURCE) {
			devices.push_back(node.node_name.get_data());
		}
	}
	unlock();
	return devices;
}

String AudioDriverPipeWire::get_input_device() {
	const char *device = DEFAULT_DEVICE;
	ERR_FAIL_NULL_V(input_stream, device);
	lock();
	const struct pw_properties *props = pw_stream_get_properties(input_stream);
	const char *target_object = pw_properties_get(props, PW_KEY_TARGET_OBJECT);
	if (target_object) {
		device = target_object;
	}
	unlock();
	return device;
}

void AudioDriverPipeWire::set_input_device(const String &p_name) {
	const char *target_object = nullptr;
	ERR_FAIL_NULL(input_stream);
	lock();
	if (p_name != DEFAULT_DEVICE) {
		const PipeWireNode *node = get_pw_node(p_name);
		if (node && node->media_class == MEDIA_CLASS_SOURCE) {
			target_object = node->node_name.ptr();
		}
	}
	struct spa_dict_item items[1];
	items[0] = SPA_DICT_ITEM_INIT(PW_KEY_TARGET_OBJECT, target_object);
	spa_dict dict = SPA_DICT_INIT(items, 1);
	pw_stream_update_properties(input_stream, &dict);
	if (pw_stream_get_state(input_stream, nullptr) == PW_STREAM_STATE_STREAMING) {
		pw_stream_disconnect(input_stream);
		input_start();
	}
	unlock();
}

#endif // PIPEWIRE_ENABLED
