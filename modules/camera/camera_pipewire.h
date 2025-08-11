/**************************************************************************/
/*  camera_pipewire.h                                                     */
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

#include "servers/camera/camera_server.h"

#ifdef DBUS_ENABLED
#include "platform/linuxbsd/freedesktop_portal_desktop.h"
#endif

#ifdef SOWRAP_ENABLED
#include "drivers/pipewire/pipewire-so_wrap.h"
#else
#include <pipewire/pipewire.h>
#endif

class CameraPipeWire : public CameraServer {
	static void on_registry_event_global(void *data, uint32_t id, uint32_t permissions, const char *type, uint32_t version, const struct spa_dict *props);
	static void on_registry_event_global_remove(void *data, uint32_t id);
	static void on_core_done(void *data, uint32_t id, int seq);

	static const struct pw_registry_events registry_events;
	static const struct pw_core_events core_events;

	pw_thread_loop *loop = nullptr;
	pw_core *core = nullptr;
	pw_context *context = nullptr;
	pw_registry *registry = nullptr;
	spa_hook registry_listener = {};
	spa_hook core_listener = {};
	uint32_t pending_id = PW_ID_ANY;
	int pending_seq = 0;
#ifdef DBUS_ENABLED
	FreeDesktopPortalDesktop *portal = nullptr;
#endif

	void on_access_camera_response(int p_resp_code);
	bool pipewire_connect(int p_fd = -1);
	void pipewire_disconnect();

public:
	CameraPipeWire();
	~CameraPipeWire();

	void thread_lock();
	void thread_unlock();
	void sync_wait(pw_proxy *p_proxy);

	virtual void set_monitoring_feeds(bool p_monitoring_feeds) override;
};
