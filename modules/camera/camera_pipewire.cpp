/**************************************************************************/
/*  camera_pipewire.cpp                                                   */
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

#include "camera_pipewire.h"

#include "camera_feed_pipewire.h"
#include "core/object/callable_method_pointer.h"
#include "platform/linuxbsd/os_linuxbsd.h"

#include <spa/utils/keys.h>

void CameraPipeWire::on_registry_event_global(void *data, uint32_t id, uint32_t permissions, const char *type, uint32_t version, const struct spa_dict *props) {
	CameraPipeWire *server = (CameraPipeWire *)data;

	if (strcmp(type, PW_TYPE_INTERFACE_Node) != 0) {
		return;
	}

	const char *media_class = spa_dict_lookup(props, SPA_KEY_MEDIA_CLASS);
	const char *media_role = spa_dict_lookup(props, SPA_KEY_MEDIA_ROLE);
	if (media_class == nullptr || media_role == nullptr) {
		return;
	}
	if (strcmp(media_class, "Video/Source") || strcmp(media_role, "Camera")) {
		return;
	}

	for (Ref<CameraFeedPipeWire> feed : server->feeds) {
		if (feed.is_null()) {
			continue;
		}
		if (feed->get_object_id() == id) {
			return;
		}
	}

	struct pw_properties *feed_props = pw_properties_new_dict(props);
	struct pw_properties *stream_props = pw_properties_new(
			PW_KEY_MEDIA_TYPE, "Video",
			PW_KEY_MEDIA_CATEGORY, "Capture",
			PW_KEY_MEDIA_ROLE, "Camera",
			PW_KEY_TARGET_OBJECT, pw_properties_get(feed_props, PW_KEY_NODE_NAME),
			NULL);
	pw_stream *stream = pw_stream_new(server->core, "", stream_props);
	struct pw_proxy *proxy = (struct pw_proxy *)pw_registry_bind(server->registry, id, type, version, 0);
	const char *feed_name = pw_properties_get(feed_props, PW_KEY_NODE_DESCRIPTION);
	Ref<CameraFeedPipeWire> feed;
	feed.instantiate(id, stream, proxy, feed_name);
	server->add_feed(feed);
	pw_properties_free(feed_props);
}

void CameraPipeWire::on_registry_event_global_remove(void *data, uint32_t id) {
	CameraPipeWire *server = (CameraPipeWire *)data;

	for (int i = server->feeds.size() - 1; i >= 0; i--) {
		Ref<CameraFeedPipeWire> feed = server->feeds[i];
		if (feed.is_null()) {
			continue;
		}
		if (feed->get_object_id() == id) {
			server->remove_feed(feed);
			break;
		}
	}
}

void CameraPipeWire::on_core_done(void *data, uint32_t id, int seq) {
	CameraPipeWire *server = (CameraPipeWire *)data;
	if (server->pending_id == id && server->pending_seq == seq) {
		pw_thread_loop_signal(server->loop, false);
	}
}

const struct pw_registry_events CameraPipeWire::registry_events = {
	.version = PW_VERSION_REGISTRY_EVENTS,
	.global = on_registry_event_global,
	.global_remove = on_registry_event_global_remove,
};

const struct pw_core_events CameraPipeWire::core_events = {
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

CameraPipeWire::CameraPipeWire() {
	pw_init(nullptr, nullptr);
#ifdef DBUS_ENABLED
	portal = ((OS_LinuxBSD *)OS::get_singleton())->get_portal_desktop();
#endif
}

CameraPipeWire::~CameraPipeWire() {
	pipewire_disconnect();
	pw_deinit();
}

void CameraPipeWire::on_access_camera_response(int p_resp_code) {
	if (p_resp_code == 0) {
		CameraServer::set_monitoring_feeds(true);
		int fd = portal->open_pipewire_remote();
		if (pipewire_connect(fd)) {
			emit_signal(SNAME(CameraServer::feeds_updated_signal_name));
		} else {
			CameraServer::set_monitoring_feeds(false);
		}
	}
}

bool CameraPipeWire::pipewire_connect(int p_fd) {
	if (loop == nullptr) {
		loop = pw_thread_loop_new("", nullptr);
		ERR_FAIL_NULL_V(loop, false);
	}
	if (context == nullptr) {
		context = pw_context_new(pw_thread_loop_get_loop(loop), nullptr, 0);
		ERR_FAIL_NULL_V(context, false);
	}
	if (core == nullptr) {
		core = pw_context_connect(context, nullptr, 0);
#ifdef DBUS_ENABLED
		if (core == nullptr) {
			if (p_fd != -1) {
				core = pw_context_connect_fd(context, p_fd, nullptr, 0);
			} else {
				Callable access_camera_cb = callable_mp(this, &CameraPipeWire::on_access_camera_response);
				if (portal->access_camera(access_camera_cb)) {
					return false;
				}
			}
		}
#endif
		ERR_FAIL_NULL_V(core, false);
		pw_core_add_listener(core, &core_listener, &core_events, this);
	}
	if (registry == nullptr) {
		registry = pw_core_get_registry(core, PW_VERSION_REGISTRY, 0);
		ERR_FAIL_NULL_V(registry, false);
		pw_registry_add_listener(registry, &registry_listener, &registry_events, this);
	}
	ERR_FAIL_COND_V(pw_thread_loop_start(loop) < 0, false);
	pw_thread_loop_lock(loop);
	pending_id = PW_ID_CORE;
	pending_seq = pw_core_sync(core, PW_ID_CORE, 0);
	if (!pw_thread_loop_in_thread(loop)) {
		pw_thread_loop_wait(loop);
	}
	pw_thread_loop_unlock(loop);
	return true;
}

void CameraPipeWire::pipewire_disconnect() {
	for (int i = feeds.size() - 1; i >= 0; i--) {
		Ref<CameraFeedPipeWire> feed = feeds[i];
		if (feed.is_null()) {
			continue;
		}
		remove_feed(feed);
	}
	if (loop) {
		pw_thread_loop_lock(loop);
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
	}
}

void CameraPipeWire::thread_lock() {
	ERR_FAIL_NULL(loop);
	pw_thread_loop_lock(loop);
}

void CameraPipeWire::thread_unlock() {
	ERR_FAIL_NULL(loop);
	pw_thread_loop_unlock(loop);
}

void CameraPipeWire::sync_wait(pw_proxy *p_proxy) {
	ERR_FAIL_NULL(p_proxy);
	ERR_FAIL_NULL(loop);
	pending_id = pw_proxy_get_id(p_proxy);
	pending_seq = pw_proxy_sync(p_proxy, pending_seq);
	if (!pw_thread_loop_in_thread(loop)) {
		pw_thread_loop_wait(loop);
	}
}

void CameraPipeWire::set_monitoring_feeds(bool p_monitoring_feeds) {
	if (p_monitoring_feeds == monitoring_feeds) {
		return;
	}

	CameraServer::set_monitoring_feeds(p_monitoring_feeds);
	if (p_monitoring_feeds) {
		if (!pipewire_connect()) {
			CameraServer::set_monitoring_feeds(false);
			return;
		}
		emit_signal(SNAME(CameraServer::feeds_updated_signal_name));
	} else {
		pipewire_disconnect();
	}
}
