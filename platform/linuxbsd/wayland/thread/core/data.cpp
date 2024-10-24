/**************************************************************************/
/*  data.cpp                                                              */
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

#include "wayland/wayland_thread.h"

#include <unistd.h>

// NOTE: Don't forget to `memfree` the offer's state.
void WaylandThread::_wl_data_device_on_data_offer(void *data, struct wl_data_device *wl_data_device, struct wl_data_offer *id) {
	wl_proxy_tag_godot((struct wl_proxy *)id);
	wl_data_offer_add_listener(id, &wl_data_offer_listener, memnew(OfferState));
}

void WaylandThread::_wl_data_device_on_enter(void *data, struct wl_data_device *wl_data_device, uint32_t serial, struct wl_surface *surface, wl_fixed_t x, wl_fixed_t y, struct wl_data_offer *id) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	ss->dnd_enter_serial = serial;
	ss->wl_data_offer_dnd = id;

	// Godot only supports DnD file copying for now.
	wl_data_offer_accept(id, serial, "text/uri-list");
	wl_data_offer_set_actions(id, WL_DATA_DEVICE_MANAGER_DND_ACTION_COPY, WL_DATA_DEVICE_MANAGER_DND_ACTION_COPY);
}

void WaylandThread::_wl_data_device_on_leave(void *data, struct wl_data_device *wl_data_device) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	if (ss->wl_data_offer_dnd) {
		memdelete(wl_data_offer_get_offer_state(ss->wl_data_offer_dnd));
		wl_data_offer_destroy(ss->wl_data_offer_dnd);
		ss->wl_data_offer_dnd = nullptr;
	}
}

void WaylandThread::_wl_data_device_on_motion(void *data, struct wl_data_device *wl_data_device, uint32_t time, wl_fixed_t x, wl_fixed_t y) {
}

void WaylandThread::_wl_data_device_on_drop(void *data, struct wl_data_device *wl_data_device) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	WaylandThread *wayland_thread = ss->wayland_thread;
	ERR_FAIL_NULL(wayland_thread);

	OfferState *os = wl_data_offer_get_offer_state(ss->wl_data_offer_dnd);
	ERR_FAIL_NULL(os);

	if (os) {
		Ref<DropFilesEventMessage> msg;
		msg.instantiate();

		Vector<uint8_t> list_data = _wl_data_offer_read(wayland_thread->wl_display, "text/uri-list", ss->wl_data_offer_dnd);

		msg->files = String::utf8((const char *)list_data.ptr(), list_data.size()).split("\r\n", false);
		for (int i = 0; i < msg->files.size(); i++) {
			msg->files.write[i] = msg->files[i].replace("file://", "").uri_decode();
		}

		wayland_thread->push_message(msg);

		wl_data_offer_finish(ss->wl_data_offer_dnd);
	}

	memdelete(wl_data_offer_get_offer_state(ss->wl_data_offer_dnd));
	wl_data_offer_destroy(ss->wl_data_offer_dnd);
	ss->wl_data_offer_dnd = nullptr;
}

void WaylandThread::_wl_data_device_on_selection(void *data, struct wl_data_device *wl_data_device, struct wl_data_offer *id) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	if (ss->wl_data_offer_selection) {
		memdelete(wl_data_offer_get_offer_state(ss->wl_data_offer_selection));
		wl_data_offer_destroy(ss->wl_data_offer_selection);
	}

	ss->wl_data_offer_selection = id;
}

void WaylandThread::_wl_data_offer_on_offer(void *data, struct wl_data_offer *wl_data_offer, const char *mime_type) {
	OfferState *os = (OfferState *)data;
	ERR_FAIL_NULL(os);

	if (os) {
		os->mime_types.insert(String::utf8(mime_type));
	}
}

void WaylandThread::_wl_data_offer_on_source_actions(void *data, struct wl_data_offer *wl_data_offer, uint32_t source_actions) {
}

void WaylandThread::_wl_data_offer_on_action(void *data, struct wl_data_offer *wl_data_offer, uint32_t dnd_action) {
}

void WaylandThread::_wl_data_source_on_target(void *data, struct wl_data_source *wl_data_source, const char *mime_type) {
}

void WaylandThread::_wl_data_source_on_send(void *data, struct wl_data_source *wl_data_source, const char *mime_type, int32_t fd) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	Vector<uint8_t> *data_to_send = nullptr;

	if (wl_data_source == ss->wl_data_source_selection) {
		data_to_send = &ss->selection_data;
		DEBUG_LOG_WAYLAND_THREAD("Clipboard: requested selection.");
	}

	if (data_to_send) {
		ssize_t written_bytes = 0;

		bool valid_mime = false;

		if (strcmp(mime_type, "text/plain;charset=utf-8") == 0) {
			valid_mime = true;
		} else if (strcmp(mime_type, "text/plain") == 0) {
			valid_mime = true;
		}

		if (valid_mime) {
			written_bytes = write(fd, data_to_send->ptr(), data_to_send->size());
		}

		if (written_bytes > 0) {
			DEBUG_LOG_WAYLAND_THREAD(vformat("Clipboard: sent %d bytes.", written_bytes));
		} else if (written_bytes == 0) {
			DEBUG_LOG_WAYLAND_THREAD("Clipboard: no bytes sent.");
		} else {
			ERR_PRINT(vformat("Clipboard: write error %d.", errno));
		}
	}

	close(fd);
}

void WaylandThread::_wl_data_source_on_cancelled(void *data, struct wl_data_source *wl_data_source) {
	SeatState *ss = (SeatState *)data;
	ERR_FAIL_NULL(ss);

	wl_data_source_destroy(wl_data_source);

	if (wl_data_source == ss->wl_data_source_selection) {
		ss->wl_data_source_selection = nullptr;
		ss->selection_data.clear();

		DEBUG_LOG_WAYLAND_THREAD("Clipboard: selection set by another program.");
		return;
	}
}

void WaylandThread::_wl_data_source_on_dnd_drop_performed(void *data, struct wl_data_source *wl_data_source) {
}

void WaylandThread::_wl_data_source_on_dnd_finished(void *data, struct wl_data_source *wl_data_source) {
}

void WaylandThread::_wl_data_source_on_action(void *data, struct wl_data_source *wl_data_source, uint32_t dnd_action) {
}
