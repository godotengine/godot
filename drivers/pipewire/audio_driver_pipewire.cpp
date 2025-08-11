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

#include "core/config/project_settings.h"
#include "core/io/json.h"
#include "core/version.h"

GODOT_GCC_WARNING_PUSH
GODOT_GCC_WARNING_IGNORE("-Wmissing-field-initializers")
GODOT_CLANG_WARNING_PUSH
GODOT_CLANG_WARNING_IGNORE("-Wmissing-field-initializers")

#include <spa/param/audio/raw-utils.h>
#include <spa/param/audio/raw.h>
#include <spa/utils/keys.h>

const struct pw_core_events AudioDriverPipeWire::core_events = {
	.version = PW_VERSION_CORE_EVENTS,
	.info = nullptr,
	.done = on_core_done,
	.ping = nullptr,
	.error = nullptr,
	.remove_id = nullptr,
	.bound_id = nullptr,
	.add_mem = nullptr,
	.remove_mem = nullptr,
	.bound_props = nullptr,
};

const struct pw_registry_events AudioDriverPipeWire::registry_events = {
	.version = PW_VERSION_REGISTRY_EVENTS,
	.global = on_registry_event_global,
	.global_remove = on_registry_event_global_remove,
};

const struct pw_node_events AudioDriverPipeWire::node_events = {
	.version = PW_VERSION_NODE_EVENTS,
	.info = on_node_info,
	.param = nullptr,
};

const struct pw_metadata_events AudioDriverPipeWire::metadata_events = {
	.version = PW_VERSION_METADATA_EVENTS,
	.property = on_metadata_property,
};

const struct pw_stream_events AudioDriverPipeWire::output_stream_events = {
	.version = PW_VERSION_STREAM_EVENTS,
	.destroy = on_output_stream_destroy,
	.state_changed = nullptr,
	.control_info = nullptr,
	.io_changed = nullptr,
	.param_changed = nullptr,
	.add_buffer = nullptr,
	.remove_buffer = nullptr,
	.process = on_output_stream_process,
	.drained = nullptr,
	.command = nullptr,
	.trigger_done = nullptr,
};

const struct pw_stream_events AudioDriverPipeWire::input_stream_events = {
	.version = PW_VERSION_STREAM_EVENTS,
	.destroy = on_input_stream_destroy,
	.state_changed = nullptr,
	.control_info = nullptr,
	.io_changed = nullptr,
	.param_changed = nullptr,
	.add_buffer = nullptr,
	.remove_buffer = nullptr,
	.process = on_input_stream_process,
	.drained = nullptr,
	.command = nullptr,
	.trigger_done = nullptr,
};

void AudioDriverPipeWire::on_core_done(void *data, uint32_t id, int seq) {
	AudioDriverPipeWire *ad = static_cast<AudioDriverPipeWire *>(data);

	if (id == ad->pending_id && seq == ad->pending_seq) {
		pw_thread_loop_signal(ad->loop, false);
	}
}

void AudioDriverPipeWire::on_registry_event_global(void *data, uint32_t id, uint32_t permissions, const char *type, uint32_t version, const struct spa_dict *props) {
	AudioDriverPipeWire *ad = static_cast<AudioDriverPipeWire *>(data);

	if (spa_streq(type, PW_TYPE_INTERFACE_Metadata)) {
		const char *metadata_name = spa_dict_lookup(props, PW_KEY_METADATA_NAME);
		if (spa_streq(metadata_name, "default")) {
			ad->metadata = (struct pw_metadata *)pw_registry_bind(ad->registry, id, type, version, 0);
			pw_metadata_add_listener(ad->metadata, &ad->metadata_listener, &metadata_events, ad);
			ad->sync_wait();
			return;
		}
	}

	if (strcmp(type, PW_TYPE_INTERFACE_Node) != 0) {
		return;
	}

	const char *media_class = spa_dict_lookup(props, SPA_KEY_MEDIA_CLASS);
	if (media_class == nullptr) {
		return;
	}
	if (strcmp(media_class, "Audio/Sink") && strcmp(media_class, "Audio/Source")) {
		return;
	}

	const char *node_name = spa_dict_lookup(props, PW_KEY_NODE_NAME);

	struct pw_proxy *proxy = (struct pw_proxy *)pw_registry_bind(ad->registry, id, type, version, 0);
	struct spa_hook *listener = (spa_hook *)malloc(sizeof(spa_hook));
	pw_node_add_listener((pw_node *)proxy, listener, &node_events, ad);

	struct PipeWireNode node = {
		.id = id,
		.proxy = proxy,
		.listener = listener,
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
		if (node.id == id) {
			if (node.media_class == "Audio/Sink" && ad->get_output_device() == node.node_name) {
				ad->set_output_device("Default");
			} else if (node.media_class == "Audio/Source" && ad->get_input_device() == node.node_name) {
				ad->set_input_device("Default");
			}
			if (node.proxy) {
				pw_proxy_destroy(node.proxy);
			}
			if (node.listener) {
				spa_hook_remove(node.listener);
			}
			ad->pw_nodes.remove_at(i);
			return;
		}
	}
}

int AudioDriverPipeWire::on_metadata_property(void *data, uint32_t subject, const char *key, const char *type, const char *value) {
	AudioDriverPipeWire *ad = static_cast<AudioDriverPipeWire *>(data);

	if (spa_streq(key, "default.audio.sink") || spa_streq(key, "default.audio.source")) {
		JSON json;
		ERR_FAIL_COND_V(json.parse(value) != OK, 0);
		Dictionary dict = json.get_data();
		ERR_FAIL_COND_V(!dict.has("name"), 0);
		if (spa_streq(key, "default.audio.sink")) {
			ad->default_output_device = dict["name"];
			if (ad->output_stream && ad->get_output_device() == "Default") {
				ad->set_output_device("Default");
			}
		} else if (spa_streq(key, "default.audio.source")) {
			ad->default_input_device = dict["name"];
			if (ad->input_stream && ad->get_input_device() == "Default") {
				ad->set_input_device("Default");
			}
		}
	}
	return 0;
}

void AudioDriverPipeWire::on_node_info(void *data, const struct pw_node_info *info) {
	AudioDriverPipeWire *ad = static_cast<AudioDriverPipeWire *>(data);

	if (info == nullptr || !(info->change_mask & PW_NODE_CHANGE_MASK_PARAMS)) {
		return;
	}

	for (int i = 0; i < ad->pw_nodes.size(); i++) {
		PipeWireNode &node = ad->pw_nodes.ptrw()[i];
		if (node.id != info->id) {
			continue;
		}
		if (node.media_class == "Audio/Sink") {
			node.channels = info->n_input_ports;
		} else if (node.media_class == "Audio/Source") {
			node.channels = info->n_output_ports;
		}
	}
}

void AudioDriverPipeWire::on_output_stream_destroy(void *data) {
	AudioDriverPipeWire *ad = static_cast<AudioDriverPipeWire *>(data);
	ad->output_stream = nullptr;
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

	if (!ad->active.is_set()) {
		ad->output_buffer.fill(0);
	} else {
		if (ad->output_buffer.size() != dst_data->maxsize) {
			ad->output_buffer.resize(dst_data->maxsize);
		}
		ad->audio_server_process(n_frames, ad->output_buffer.ptrw());
	}

	int16_t *dst = (int16_t *)dst_data->data;
	for (uint32_t i = 0; i < n_frames * ad->output_channels; i++) {
		*dst++ = ad->output_buffer[i] >> 16;
	}

	dst_data->chunk->offset = 0;
	dst_data->chunk->stride = stride;
	dst_data->chunk->size = n_frames * stride;

	pw_stream_queue_buffer(ad->output_stream, b);

	ad->stop_counting_ticks();
}

void AudioDriverPipeWire::on_input_stream_destroy(void *data) {
	AudioDriverPipeWire *ad = static_cast<AudioDriverPipeWire *>(data);
	ad->input_stream = nullptr;
}

void AudioDriverPipeWire::on_input_stream_process(void *data) {
	struct pw_buffer *b;
	struct spa_data *src_data;
	AudioDriverPipeWire *ad = static_cast<AudioDriverPipeWire *>(data);

	ERR_FAIL_NULL_MSG((b = pw_stream_dequeue_buffer(ad->input_stream)), "Out of buffer.");
	src_data = &b->buffer->datas[0]; // codespell:ignore datas

	uint32_t n_samples = src_data->chunk->size / sizeof(int16_t);
	int16_t *src = (int16_t *)src_data->data;

	if (ad->input_buffer.size() != src_data->maxsize) {
		ad->input_buffer.resize(src_data->maxsize);
		ad->input_position = 0;
		ad->input_size = 0;
	}

	for (uint32_t i = 0; i < n_samples; i++) {
		int32_t sample = int32_t(*src++) << 16;
		ad->input_buffer_write(sample);
		if (ad->input_channels == 1) {
			ad->input_buffer_write(sample);
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
		if (node->node_name == p_name) {
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
	output_stream = pw_stream_new(core, "Sound", props);
	pw_stream_add_listener(output_stream, &output_stream_listener, &output_stream_events, this);
	set_output_device("Default");
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

	active.clear();
	mix_rate = _get_configured_mix_rate();

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
	ERR_FAIL_NULL_V(core, ERR_CANT_OPEN);
	pw_core_add_listener(core, &core_listener, &core_events, this);

	registry = pw_core_get_registry(core, PW_VERSION_REGISTRY, 0);
	ERR_FAIL_NULL_V(registry, ERR_CANT_OPEN);
	pw_registry_add_listener(registry, &registry_listener, &registry_events, this);

	if (pw_thread_loop_start(loop) < 0) {
		finish();
		return ERR_CANT_OPEN;
	}

	pw_thread_loop_lock(loop);
	sync_wait();
	init_output_stream();
	init_input_stream();
	pw_thread_loop_unlock(loop);

	return OK;
}

void AudioDriverPipeWire::start() {
	active.set();
}

int AudioDriverPipeWire::get_mix_rate() const {
	return mix_rate;
}

AudioDriver::SpeakerMode AudioDriverPipeWire::get_speaker_mode() const {
	return get_speaker_mode_by_total_channels(output_channels);
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
	if (metadata) {
		pw_proxy_destroy((struct pw_proxy *)metadata);
		spa_hook_remove(&metadata_listener);
		metadata = nullptr;
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
	devices.push_back("Default");
	for (const PipeWireNode &node : pw_nodes) {
		if (node.media_class == "Audio/Sink") {
			devices.push_back(node.node_name);
		}
	}
	unlock();
	return devices;
}

String AudioDriverPipeWire::get_output_device() {
	const char *device = "Default";
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
	const PipeWireNode *node = nullptr;
	const char *target_object = nullptr;
	ERR_FAIL_NULL(output_stream);
	lock();
	if (p_name == "Default" || !get_output_device_list().has(p_name)) {
		node = get_pw_node(default_output_device);
	} else {
		node = get_pw_node(p_name);
	}
	if (node) {
		if (p_name == node->node_name) {
			target_object = node->node_name.utf8().ptr();
		}
		output_channels = SPA_CLAMP((int)node->channels, 2, 8);
		if (output_channels % 2) {
			output_channels += 1;
		}
	} else {
		ERR_PRINT("PipeWire: Failed to get default audio sink.");
		output_channels = 2;
	}
	pw_stream_disconnect(output_stream);
	struct spa_dict_item items[1];
	items[0] = SPA_DICT_ITEM_INIT(PW_KEY_TARGET_OBJECT, target_object);
	spa_dict dict = SPA_DICT_INIT(items, 1);
	pw_stream_update_properties(output_stream, &dict);
	pw_stream_flags flags = pw_stream_flags(PW_STREAM_FLAG_AUTOCONNECT | PW_STREAM_FLAG_MAP_BUFFERS | PW_STREAM_FLAG_RT_PROCESS);
	const struct spa_pod *param[1];
	uint8_t buffer[1024];
	struct spa_pod_builder b = SPA_POD_BUILDER_INIT(buffer, sizeof(buffer));
	struct spa_audio_info_raw info = {
		.format = SPA_AUDIO_FORMAT_S16,
		.flags = SPA_AUDIO_FLAG_NONE,
		.rate = mix_rate,
		.channels = output_channels,
		.position = {
				SPA_AUDIO_CHANNEL_FL, SPA_AUDIO_CHANNEL_FR,
				SPA_AUDIO_CHANNEL_FC, SPA_AUDIO_CHANNEL_LFE,
				SPA_AUDIO_CHANNEL_RL, SPA_AUDIO_CHANNEL_RR,
				SPA_AUDIO_CHANNEL_SL, SPA_AUDIO_CHANNEL_SR },
	};
	param[0] = spa_format_audio_raw_build(&b, SPA_PARAM_EnumFormat, &info);
	pw_stream_connect(output_stream, PW_DIRECTION_OUTPUT, PW_ID_ANY, flags, param, 1);
	unlock();
}

Error AudioDriverPipeWire::input_start() {
	ERR_FAIL_NULL_V(input_stream, ERR_CANT_OPEN);
	lock();
	pw_stream_flags flags = pw_stream_flags(PW_STREAM_FLAG_AUTOCONNECT | PW_STREAM_FLAG_MAP_BUFFERS | PW_STREAM_FLAG_RT_PROCESS);
	const struct spa_pod *param[1];
	uint8_t buffer[1024];
	struct spa_pod_builder builder = {};
	builder.data = buffer;
	builder.size = sizeof(buffer);
	struct spa_audio_info_raw info = {
		.format = SPA_AUDIO_FORMAT_S16,
		.flags = SPA_AUDIO_FLAG_NONE,
		.rate = mix_rate,
		.channels = input_channels,
		.position = {},
	};
	param[0] = spa_format_audio_raw_build(&builder, SPA_PARAM_EnumFormat, &info);
	pw_stream_connect(input_stream, PW_DIRECTION_INPUT, PW_ID_ANY, flags, param, 1);
	unlock();
	return OK;
}

Error AudioDriverPipeWire::input_stop() {
	ERR_FAIL_NULL_V(input_stream, ERR_CANT_OPEN);
	lock();
	pw_stream_disconnect(input_stream);
	unlock();
	return OK;
}

PackedStringArray AudioDriverPipeWire::get_input_device_list() {
	PackedStringArray devices;
	lock();
	devices.push_back("Default");
	for (const PipeWireNode &node : pw_nodes) {
		if (node.media_class == "Audio/Source") {
			devices.push_back(node.node_name);
		}
	}
	unlock();
	return devices;
}

String AudioDriverPipeWire::get_input_device() {
	const char *device = "Default";
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
	const PipeWireNode *node = nullptr;
	const char *target_object = nullptr;
	ERR_FAIL_NULL(input_stream);
	lock();
	if (p_name == "Default" || !get_input_device_list().has(p_name)) {
		node = get_pw_node(default_input_device);
	} else {
		node = get_pw_node(p_name);
	}
	if (node) {
		if (p_name == node->node_name) {
			target_object = node->node_name.utf8().ptr();
		}
		input_channels = (SPA_CLAMP((int)node->channels, 1, 2));
	} else {
		ERR_PRINT("PipeWire: Failed to get default audio source.");
		input_channels = 2;
	}
	struct spa_dict_item items[1];
	items[0] = SPA_DICT_ITEM_INIT(PW_KEY_TARGET_OBJECT, target_object);
	spa_dict dict = SPA_DICT_INIT(items, 1);
	pw_stream_update_properties(input_stream, &dict);
	if (pw_stream_get_state(input_stream, nullptr) == PW_STREAM_STATE_STREAMING) {
		input_stop();
		input_start();
	}
	unlock();
}

#endif // PIPEWIRE_ENABLED
