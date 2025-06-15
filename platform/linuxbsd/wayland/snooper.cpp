/**************************************************************************/
/*  snooper.cpp                                                           */
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

#include "snooper.h"

// Rough general to do list:
//
//  - (Maybe?) Report pointer position somehow for main window when over
//    embedded window
//
//  - Implement custom focusing logic for tablet
//
//  - Track and report state for all non-destructible objects on instancing
//  (luckily there are few of them)
//
//  - Keep cleaning up this mess (code still sucks)
//
//  - Do the mario (swing your arms from side to side)

#include <sys/stat.h>

#ifdef __FreeBSD__
#include <dev/evdev/input-event-codes.h>
#else
// Assume Linux.
#include <linux/input-event-codes.h>
#endif

#include "core/os/os.h"

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#define WAYLAND_SNOOPER_DEBUG_LOGS_ENABLED
#ifdef WAYLAND_SNOOPER_DEBUG_LOGS_ENABLED

// Gotta flush as we're doing this mess from a thread without any
// synchronization. It's awful, I know, but the `print_*` utilities hang for
// some reason during editor startup and I need some quick and dirty debugging.
#define DEBUG_LOG_WAYLAND_SNOOPER(...)                             \
	if (1) {                                                       \
		printf("[PROXY] %s\n", vformat(__VA_ARGS__).utf8().ptr()); \
		fflush(stdout);                                            \
	} else                                                         \
		((void)0)

#else
#define DEBUG_LOG_WAYLAND_SNOOPER(...)
#endif

// Wayland messages are structured with 32-bit words.
#define WORD_SIZE (sizeof(uint32_t))

// Event opcodes. Request opcodes are defined in the generated client headers.
// We could generate server headers but they would clash (without modifications)
// and we use just a few constants anyways.

#define WL_DISPLAY_ERROR 0
#define WL_DISPLAY_DELETE_ID 1

#define WL_REGISTRY_GLOBAL 0

#define WL_CALLBACK_DONE 0

#define WL_KEYBOARD_ENTER 1
#define WL_KEYBOARD_LEAVE 2
#define WL_KEYBOARD_KEY 3

#define WL_POINTER_ENTER 0
#define WL_POINTER_LEAVE 1
#define WL_POINTER_BUTTON 3

#define WL_SHM_FORMAT 0

#define WL_DRM_DEVICE 0
#define WL_DRM_FORMAT 1
#define WL_DRM_AUTHENTICATED 2
#define WL_DRM_CAPABILITIES 3

size_t WaylandEmbedderProxy::wl_array_word_offset(uint32_t p_size) {
	constexpr size_t word_size = sizeof(uint32_t);

	uint32_t pad = (word_size - (p_size % word_size)) % word_size;
	return (p_size + pad) / word_size;
}

const struct wl_interface *WaylandEmbedderProxy::wl_interface_from_string(const char *name, size_t size) {
	for (size_t i = 0; i < (sizeof interfaces / sizeof *interfaces); ++i) {
		if (strncmp(name, interfaces[i]->name, size) == 0) {
			return interfaces[i];
		}
	}

	return nullptr;
}

struct WaylandEmbedderProxy::WaylandObject *WaylandEmbedderProxy::get_object(uint32_t p_global_id) {
	if (p_global_id == 0) {
		return nullptr;
	}

	// Server-allocated stuff starts at 0xff000000.
	bool is_server = p_global_id & 0xff000000;
	if (is_server) {
		p_global_id &= ~(0xff000000);
	}

	CRASH_COND_MSG(p_global_id >= SNOOP_ID_MAX, "max id reached");

	if (is_server) {
		return &server_objects[p_global_id];
	} else {
		return &objects[p_global_id];
	}
}

Error WaylandEmbedderProxy::delete_object(uint32_t p_global_id) {
	WaylandObject *object = get_object(p_global_id);
	ERR_FAIL_NULL_V(object, ERR_DOES_NOT_EXIST);

	if (shared_objects.has(object->interface)) {
		ERR_PRINT(vformat("Tried to delete shared object g0x%x.", p_global_id));
	}

	DEBUG_LOG_WAYLAND_SNOOPER(vformat("Deleting object %s g0x%x", object->interface ? object->interface->name : "UNKNOWN", p_global_id));

	object->interface = nullptr;
	object->version = 0;
	if (object->data) {
		memdelete(object->data);
		object->data = nullptr;
	}

	registry_globals_names.erase(p_global_id);

	return OK;
}

uint32_t WaylandEmbedderProxy::Client::allocate_server_id() {
	for (uint32_t test_id = 0; test_id < server_id_map.size(); ++test_id) {
		if (server_id_map[test_id] == false) {
			server_id_map[test_id] = true;

			uint32_t id = test_id | 0xff000000;
			DEBUG_LOG_WAYLAND_SNOOPER(vformat("allocated server-side id 0x%x", id));

			return id;
		}
	}

	CRASH_NOW_MSG("Out of server-side IDs.");
}

struct WaylandEmbedderProxy::WaylandObject *WaylandEmbedderProxy::Client::get_object(uint32_t p_local_id) {
	if (p_local_id == INVALID_ID) {
		return nullptr;
	}

	if (global_instances.has(p_local_id)) {
		return &global_instances[p_local_id];
	}

	if (fake_objects.has(p_local_id)) {
		return &fake_objects[p_local_id];
	}

	if (!global_ids.has(p_local_id)) {
		return nullptr;
	}

	ERR_FAIL_COND_V(snooper == nullptr, nullptr);
	return snooper->get_object(get_global_id(p_local_id));
}

Error WaylandEmbedderProxy::Client::delete_object(uint32_t p_local_id) {
	if (global_instances.has(p_local_id)) {
#ifdef WAYLAND_SNOOPER_DEBUG_LOGS_ENABLED
		WaylandObject *object = &global_instances[p_local_id];
		DEBUG_LOG_WAYLAND_SNOOPER(vformat("Deleting global instance %s l0x%x", object->interface ? object->interface->name : "UNKNOWN", p_local_id));
#endif

		// wl_display::delete_id
		send_wayland_message(socket, DISPLAY_ID, 1, { p_local_id });

		// We don't want to delete the global object tied to this instance, so we'll only get rid of the local stuff.
		global_instances.erase(p_local_id);
		global_ids.erase(p_local_id);

		// We're done here.
		return OK;
	}
	if (fake_objects.has(p_local_id)) {
#ifdef WAYLAND_SNOOPER_DEBUG_LOGS_ENABLED
		WaylandObject *object = &fake_objects[p_local_id];
		DEBUG_LOG_WAYLAND_SNOOPER(vformat("Deleting fake object %s l0x%x", object->interface ? object->interface->name : "UNKNOWN", p_local_id));
#endif

		if (!(p_local_id & 0xff000000)) {
			// wl_display::delete_id
			send_wayland_message(socket, DISPLAY_ID, 1, { p_local_id });
		}

		fake_objects.erase(p_local_id);

		// We can skip everything else below, as fake objects don't have a global id.
		return OK;
	}

	ERR_FAIL_COND_V(!global_ids.has(p_local_id), ERR_DOES_NOT_EXIST);
	uint32_t global_id = global_ids[p_local_id];

	WaylandObject *object = snooper->get_object(global_id);
	ERR_FAIL_NULL_V(object, ERR_DOES_NOT_EXIST);

	if (snooper->shared_objects.has(object->interface)) {
		ERR_PRINT(vformat("Tried to delete shared object g0x%x.", global_id));
		return ERR_INVALID_PARAMETER;
	}

	global_ids.erase(p_local_id);
	local_ids.erase(global_id);

	if (p_local_id & 0xff000000) {
		server_id_map[p_local_id & ~(0xff000000)] = false;
	}

	uint32_t *global_name = snooper->registry_globals_names.getptr(global_id);
	if (global_name) {
		registry_globals_instances.erase(*global_name);
	}

	return snooper->delete_object(global_id);
}

uint32_t WaylandEmbedderProxy::Client::new_object(uint32_t p_local_id, const struct wl_interface *p_interface, int p_version, WaylandObjectData *p_data) {
	CRASH_COND(snooper == nullptr);
	CRASH_COND_MSG(get_object(p_local_id) != nullptr, vformat("Tried to create %s l0x%x but it already exists as %s", p_interface->name, p_local_id, get_object(p_local_id)->interface->name));

	uint32_t new_global_id = snooper->new_object(p_interface, p_version, p_data);

	global_ids[p_local_id] = new_global_id;
	local_ids[new_global_id] = p_local_id;

	return new_global_id;
}

WaylandEmbedderProxy::WaylandObject *WaylandEmbedderProxy::Client::new_fake_object(uint32_t p_local_id, const struct wl_interface *p_interface, int p_version, WaylandObjectData *p_data) {
	CRASH_COND(snooper == nullptr);
	CRASH_COND_MSG(get_object(p_local_id) != nullptr, vformat("Object l0x%x already exists", p_local_id));

	WaylandObject &new_object = fake_objects[p_local_id];
	new_object.interface = p_interface;
	new_object.version = p_version;
	new_object.data = p_data;

	return &new_object;
}

WaylandEmbedderProxy::WaylandObject *WaylandEmbedderProxy::Client::new_global_instance(uint32_t p_local_id, const struct wl_interface *p_interface, int p_version, WaylandObjectData *p_data) {
	CRASH_COND(snooper == nullptr);
	CRASH_COND_MSG(get_object(p_local_id) != nullptr, vformat("Object l0x%x already exists", p_local_id));

	WaylandObject &new_object = global_instances[p_local_id];
	new_object.interface = p_interface;
	new_object.version = p_version;
	new_object.data = p_data;

	return &new_object;
}

void WaylandEmbedderProxy::Client::send_wl_drm_state(uint32_t p_id, WaylandDrmGlobalData *p_state) {
	CRASH_COND(p_state == nullptr);

	if (p_state->device.is_empty()) {
		// Not yet initialized.
		return;
	}

	{
		// FIXME: Make string marshaling at least somewhat bearable.
		size_t str_words = wl_array_word_offset(p_state->device.length());
		size_t arg_words = str_words + 1;

		CharString device_name = p_state->device.utf8();

		uint32_t *arg_buf = (uint32_t *)memalloc(arg_words);
		char *str_buf = (char *)arg_buf + 1;

		arg_buf[0] = device_name.length();
		strncpy(str_buf, device_name.get_data(), device_name.length());

		send_wayland_message(socket, p_id, WL_DRM_DEVICE, arg_buf, arg_words);

		memfree(arg_buf);
	}

	for (uint32_t format : p_state->formats) {
		send_wayland_message(socket, p_id, WL_DRM_FORMAT, { format });
	}

	if (p_state->authenticated) {
		send_wayland_message(socket, p_id, WL_DRM_AUTHENTICATED, {});
	}

	send_wayland_message(socket, p_id, WL_DRM_CAPABILITIES, { p_state->capabilities });
}

void WaylandEmbedderProxy::client_disconnect(size_t p_client_id) {
	ERR_FAIL_UNSIGNED_INDEX(p_client_id, clients.size());
	Client &client = clients[p_client_id];

	close(client.socket);

	for (size_t i = 0; i < pollfds.size(); ++i) {
		if (pollfds[i].fd == client.socket) {
			pollfds.remove_at_unordered(i);
			break;
		}
	}

	for (KeyValue<uint32_t, WaylandObject> &pair : client.fake_objects) {
		WaylandObject &object = pair.value;

		if (object.interface == &xdg_toplevel_interface) {
			XdgToplevelData *data = (XdgToplevelData *)object.data;
			CRASH_COND(data == nullptr);

			if (data->wl_subsurface_id != INVALID_ID) {
				// wl_subsurface::destroy() - xdg_toplevels are mapped to subsurfaces.
				send_wayland_message(compositor_socket, data->wl_subsurface_id, 0, {});
			}
		}
	}

	for (KeyValue<uint32_t, uint32_t> &pair : client.global_ids) {
		uint32_t global_id = pair.value;

		WaylandObject *object = get_object(global_id);
		if (object == nullptr) {
			continue;
		}

		if (shared_objects.has(object->interface)) {
			continue;
		}

		if (object->interface == &wl_callback_interface) {
			// Those things self-destruct.
			continue;
		}

		// FIXME: Ensure proper destruction order for stuff like surfaces.
		bool destroyed = false;
		for (int opcode = 0; opcode < object->interface->method_count; ++opcode) {
			const struct wl_message &message = object->interface->methods[opcode];

			int destructor_version = String::to_int(message.signature);

			DEBUG_LOG_WAYLAND_SNOOPER(vformat("!!!!!TEST iface %s msg %s parsed_ver %d obj_ver %d", object->interface->name, message.signature, destructor_version, object->version));

			// FIXME: Find a better way of destroying all relevant non-fake objects. The
			// XML files have a "type" field, which can have a "destructor" value but it
			// is currently not exposed to the generated C file.
			if (object->version >= destructor_version && (strcmp(message.name, "destroy") == 0 || strcmp(message.name, "release") == 0)) {
				if (global_id & 0xff000000) {
					delete_object(global_id);
				}

				if (object->interface == &wl_surface_interface) {
					WaylandSeatGlobalData *global_seat_data = (WaylandSeatGlobalData *)registry_globals[wl_seat_name].data;
					if (global_seat_data) {
						if (global_seat_data->pointed_surface_id == global_id) {
							global_seat_data->pointed_surface_id = INVALID_ID;
						}

						if (global_seat_data->focused_surface_id == global_id) {
							global_seat_data->focused_surface_id = INVALID_ID;
						}
					}
				}

				// ??????::destroy() / ??????::release() - yes this is not ideal.
				send_wayland_message(compositor_socket, global_id, opcode, {});
				destroyed = true;
				break;
			}
		}

		if (!destroyed) {
			ERR_PRINT(vformat("Unreferenced object %s g0x%x (leak!)", object->interface->name, global_id));
		}
	}

	uint32_t eclient_id = client.embedded_client_id;

	client = Client();

	WaylandObject *eclient = clients[0].get_object(eclient_id);
	ERR_FAIL_NULL(eclient);

	EmbeddedClientData *eclient_data = (EmbeddedClientData *)eclient->data;
	ERR_FAIL_NULL(eclient_data);

	if (!eclient_data->disconnected) {
		// godot_embedded_client::disconnected
		send_wayland_message(clients[0].socket, eclient_id, 0, {});
	}

	eclient_data->disconnected = true;
}

void WaylandEmbedderProxy::poll_sockets() {
	if (poll(pollfds.ptr(), pollfds.size(), -1) == -1) {
		CRASH_NOW_MSG(vformat("poll() failed, errno %d.", errno));
	}

	for (struct pollfd &pollfd : pollfds) {
		handle_fd(pollfd.fd, pollfd.revents);
	}
}

void WaylandEmbedderProxy::send_raw_message(int p_socket, std::initializer_list<struct iovec> p_vecs, const LocalVector<int> &p_fds) {
	struct msghdr msg = {};
	msg.msg_iov = (struct iovec *)p_vecs.begin();
	msg.msg_iovlen = p_vecs.size();

	if (!p_fds.is_empty()) {
		size_t data_size = p_fds.size() * sizeof(int);

		msg.msg_control = Memory::alloc_aligned_static(CMSG_SPACE(data_size), CMSG_ALIGN(1));
		msg.msg_controllen = CMSG_SPACE(data_size);

		struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
		cmsg->cmsg_level = SOL_SOCKET;
		cmsg->cmsg_type = SCM_RIGHTS;
		cmsg->cmsg_len = CMSG_LEN(data_size);

		// NOTE: According to the linux man page cmsg(5), we shall not access the
		// pointer returned CMSG_DATA directly, due to alignment concerns. We should
		// copy data from a suitably aligned object instead.
		memcpy(CMSG_DATA(cmsg), p_fds.ptr(), data_size);
	}

#ifdef WAYLAND_SNOOPER_DEBUG_LOGS_ENABLED
	printf("[PROXY] Sending: ");

	for (const struct iovec &vec : p_vecs) {
		for (size_t i = 0; i < vec.iov_len; ++i) {
			printf("%.2x", ((const uint8_t *)vec.iov_base)[i]);
		}
	}
	printf("\n");
#endif

	sendmsg(p_socket, &msg, MSG_NOSIGNAL);

	if (msg.msg_control) {
		Memory::free_aligned_static(msg.msg_control);
	}
}

void WaylandEmbedderProxy::send_wayland_message(int p_socket, uint32_t p_id, uint32_t p_opcode, const uint32_t *p_args, const size_t p_args_words) {
	CRASH_COND(p_socket < 0);
	CRASH_COND(p_id == INVALID_ID);

	uint32_t args_size = p_args_words * sizeof *p_args;

	// Header is always 8 bytes long.
	uint32_t total_size = 8 + (args_size);

	uint32_t header[2] = { p_id, (total_size << 16) + p_opcode };

	struct iovec vecs[2] = {
		{ header, 8 },
		// According to the manual, these buffers should are never written, so this
		// cast should be safe.
		{ (void *)p_args, args_size },
	};

	struct msghdr msg = {};
	msg.msg_iov = vecs;
	msg.msg_iovlen = std::size(vecs);

#ifdef WAYLAND_SNOOPER_DEBUG_LOGS_ENABLED
	printf("[PROXY] Sending: ");

	for (struct iovec &vec : vecs) {
		for (size_t i = 0; i < vec.iov_len; ++i) {
			printf("%.2x", ((const uint8_t *)vec.iov_base)[i]);
		}
	}
	printf("\n");
#endif

	sendmsg(p_socket, &msg, MSG_NOSIGNAL);
}

void WaylandEmbedderProxy::send_wayland_message(int p_socket, uint32_t p_id, uint32_t p_opcode, std::initializer_list<uint32_t> p_args) {
	send_wayland_message(p_socket, p_id, p_opcode, p_args.begin(), p_args.size());
}

uint32_t WaylandEmbedderProxy::new_object(const struct wl_interface *p_interface, int p_version, WaylandObjectData *p_data) {
	uint32_t new_global_id = next_global_id();

	WaylandObject *new_object = get_object(new_global_id);
	new_object->interface = p_interface;
	new_object->version = p_version;
	new_object->data = p_data;

	return new_global_id;
}

void WaylandEmbedderProxy::sync() {
	CRASH_COND_MSG(sync_callback_id, "sync already in progress");

	sync_callback_id = next_global_id();
	get_object(sync_callback_id)->interface = &wl_callback_interface;
	get_object(sync_callback_id)->version = 1;
	send_wayland_message(compositor_socket, DISPLAY_ID, 0, { sync_callback_id });

	DEBUG_LOG_WAYLAND_SNOOPER("synchronizing");

	while (true) {
		poll_sockets();

		if (!sync_callback_id) {
			// Obj got deleted - sync is done.
			return;
		}
	}
}

void WaylandEmbedderProxy::seat_name_enter_surface(uint32_t p_seat_name, uint32_t p_wl_surface_id) {
	WaylandSurfaceData *surf_data = (WaylandSurfaceData *)get_object(p_wl_surface_id)->data;
	CRASH_COND(surf_data == nullptr);

	Client *client = surf_data->client;
	CRASH_COND(client == nullptr);

	if (!client->local_ids.has(p_wl_surface_id)) {
		DEBUG_LOG_WAYLAND_SNOOPER("Called seat_name_enter_surface with an unknown surface");
		return;
	}

	uint32_t local_surface_id = client->get_local_id(p_wl_surface_id);

	DEBUG_LOG_WAYLAND_SNOOPER(vformat("KB: Entering surface g0x%x", p_wl_surface_id));

	for (uint32_t local_seat_id : client->registry_globals_instances[p_seat_name]) {
		WaylandSeatInstanceData *seat_data = (WaylandSeatInstanceData *)client->get_object(local_seat_id)->data;
		CRASH_COND(seat_data == nullptr);

		uint32_t local_keyboard_id = client->get_local_id(seat_data->wl_keyboard_id);

		if (local_keyboard_id != INVALID_ID) {
			// TODO: track keys. Not super important at the time of writing, since we
			// don't use that in the engine, although we should.

			// wl_keyboard::enter(serial, surface, keys) - keys will be empty for now
			send_wayland_message(client->socket, local_keyboard_id, 1, { serial_counter++, local_surface_id, 0 });
		}
	}

	if (client != &clients[0]) {
		// godot_embedded_client::window_focus_in
		send_wayland_message(clients[0].socket, client->embedded_client_id, 2, {});
	}
}

void WaylandEmbedderProxy::seat_name_leave_surface(uint32_t p_seat_name, uint32_t p_wl_surface_id) {
	WaylandSurfaceData *surf_data = (WaylandSurfaceData *)get_object(p_wl_surface_id)->data;
	CRASH_COND(surf_data == nullptr);

	Client *client = surf_data->client;
	CRASH_COND(client == nullptr);

	if (!client->local_ids.has(p_wl_surface_id)) {
		DEBUG_LOG_WAYLAND_SNOOPER("Called seat_name_leave_surface with an unknown surface");
		return;
	}

	uint32_t local_surface_id = client->get_local_id(p_wl_surface_id);

	DEBUG_LOG_WAYLAND_SNOOPER(vformat("KB: Leaving surface g0x%x", p_wl_surface_id));

	for (uint32_t local_seat_id : client->registry_globals_instances[p_seat_name]) {
		WaylandSeatInstanceData *seat_data = (WaylandSeatInstanceData *)client->get_object(local_seat_id)->data;
		CRASH_COND(seat_data == nullptr);

		uint32_t local_keyboard_id = client->get_local_id(seat_data->wl_keyboard_id);

		if (local_keyboard_id != INVALID_ID) {
			// wl_keyboard::enter(serial, surface, keys) - keys will be empty for now
			send_wayland_message(client->socket, local_keyboard_id, 2, { serial_counter++, local_surface_id });
		}
	}

	if (client != &clients[0]) {
		// godot_embedded_client::window_focus_out
		send_wayland_message(clients[0].socket, client->embedded_client_id, 3, {});
	}
}

int WaylandEmbedderProxy::next_global_id() {
	for (uint32_t id = REGISTRY_ID; id < objects.size(); ++id) {
		if (objects[id].interface == nullptr) {
			DEBUG_LOG_WAYLAND_SNOOPER(vformat("Allocated new global id g0x%x", id));
			return id;
		}
	}

	// Oh no. Time for debug info!

#ifdef WAYLAND_THREAD_DEBUG_LOGS_ENABLED
	for (uint32_t id = 1; id < objects.size(); ++id) {
		WaylandObject &object = objects[id];
		DEBUG_LOG_WAYLAND_SNOOPER(vformat(" - g0x%x (#%d): %s version %d, data 0x%x", id, id, object.interface->name, object.version, (uintptr_t)object.data));
	}
#endif

	// FIXME: Resize or something.
	CRASH_NOW_MSG("FIXME Out of ids. FIXME");
}

int WaylandEmbedderProxy::next_global_server_id() {
	for (uint32_t id = 2; id < server_objects.size(); ++id) {
		if (server_objects[id].interface == nullptr) {
			return id;
		}
	}

	// FIXME: Resize or something.
	CRASH_NOW_MSG("FIXME Out of ids. FIXME");
}

bool WaylandEmbedderProxy::global_surface_is_window(uint32_t p_wl_surface_id) {
	WaylandObject *surface_object = get_object(p_wl_surface_id);
	ERR_FAIL_NULL_V(surface_object, false);
	if (surface_object->interface != &wl_surface_interface || surface_object->data == nullptr) {
		return false;
	}

	WaylandSurfaceData *surface_data = (WaylandSurfaceData *)surface_object->data;
	if (!surface_data->role_object_handle.is_valid()) {
		return false;
	}

	WaylandObject *role_object = surface_data->role_object_handle.get();

	return (role_object && role_object->interface == &xdg_toplevel_interface);
}

bool WaylandEmbedderProxy::handle_generic_msg(Client *client, const struct wl_interface *interface, const struct wl_message *message, const struct msg_info *info, uint32_t *buf, uint32_t instance_id) {
	bool valid = true;

	if (info->direction == ProxyDirection::COMPOSITOR) {
		DEBUG_LOG_WAYLAND_SNOOPER(vformat("Generic request %s::%s(%s) g0x%x", interface ? interface->name : "UNKNOWN", message ? message->name : "UNKNOWN", message ? message->signature : "UNKNOWN", info->raw_id));
	} else {
		DEBUG_LOG_WAYLAND_SNOOPER(vformat("Generic event %s::%s(%s) l0x%x", interface ? interface->name : "UNKNOWN", message ? message->name : "UNKNOWN", message ? message->signature : "UNKNOWN", info->raw_id));
	}

	// Let's strip the header.
	uint32_t *body = buf + 2;

	size_t arg_idx = 0;
	size_t buf_idx = 0;
	size_t last_str_buf_idx = -1;
	uint32_t last_str_len = 0;
	for (size_t i = 0; i < strlen(message->signature); ++i) {
		ERR_FAIL_COND_V(buf_idx > (info->size / sizeof *body), false);

		char sym = message->signature[i];
		if (sym >= '0' && sym <= '?') {
			// We don't care about version notices and nullability symbols. We can skip
			// those.
			continue;
		}

		switch (sym) {
			case 'a': {
				uint32_t array_len = body[buf_idx];

				// We can't obviously go forward by just one byte. Let's skip to the end of
				// the array.
				buf_idx += wl_array_word_offset(array_len);
			} break;

			case 's': {
				uint32_t string_len = body[buf_idx];

				last_str_buf_idx = buf_idx;
				last_str_len = string_len;

				// Same as the array.
				buf_idx += wl_array_word_offset(string_len);
			} break;

			case 'n': {
				uint32_t arg = body[buf_idx];

				uint32_t new_local_id = 0;
				uint32_t new_global_id = 0;

				if (info->direction == ProxyDirection::COMPOSITOR) {
					new_local_id = arg;

					// Try to allocate a suitable global id.
					new_global_id = next_global_id();

					body[buf_idx] = new_global_id;
				} else if (info->direction == ProxyDirection::CLIENT) {
					new_global_id = arg;

					if (client) {
						new_local_id = client->allocate_server_id();
						CRASH_COND_MSG(new_local_id == 0, "Out of server-side IDs.");

						body[buf_idx] = new_local_id;
					}
				}

				if (client) {
					client->global_ids[new_local_id] = new_global_id;
					client->local_ids[new_global_id] = new_local_id;
				}

				const struct wl_interface *new_interface = message->types[arg_idx];
				uint32_t new_version = interface->version;

				if (!new_interface && last_str_len != 0) {
					// When the protocol definition does not define an interface it reports a
					// string and an unsigned integer representing the interface and the
					// version requested.
					new_interface = wl_interface_from_string((char *)(body + last_str_buf_idx + 1), last_str_len);
					new_version = body[arg_idx - 1];
				}

				if (new_interface) {
					if (client) {
						DEBUG_LOG_WAYLAND_SNOOPER(vformat("new id l0x%x g0x%x interface %s", new_local_id, new_global_id, new_interface->name));
						client->get_object(new_local_id)->interface = new_interface;
						client->get_object(new_local_id)->version = new_version;
					} else {
						DEBUG_LOG_WAYLAND_SNOOPER(vformat("new id lNOCLIENT g0x%x interface %s", new_global_id, new_interface->name));
						get_object(new_global_id)->interface = new_interface;
						get_object(new_global_id)->version = new_version;
					}
				} else {
					if (last_str_len > 0) {
						DEBUG_LOG_WAYLAND_SNOOPER(vformat("Unknown interface %s, marking packet as invalid.", (char *)(body + last_str_buf_idx + 1)));
					} else {
						DEBUG_LOG_WAYLAND_SNOOPER("Unknown interface, marking packet as invalid.");
					}
					valid = false;
					break;
				}
			} break;

			case 'o': {
				uint32_t obj_id = body[buf_idx];
				if (obj_id == 0) {
					// Object arguments can be nil.
					break;
				}

				if (!client) {
					// Doesn't matter. We're not going to send the thing anyways.
					break;
				}

				if (info->direction == ProxyDirection::CLIENT) {
					// FIXME: Make sure that this code is still useful. It's originates from a
					// very old prototype of this code.
					uint32_t *global_name = registry_globals_names.getptr(obj_id);
					if (global_name) {
						if (!client->registry_globals_instances.has(*global_name)) {
							DEBUG_LOG_WAYLAND_SNOOPER(vformat("Object argument g0x%x points to global but client does not have it. Marking packet as invalid.", obj_id));
							valid = false;
							break;
						}
						uint32_t new_local_id = *client->registry_globals_instances[*global_name].begin();
						DEBUG_LOG_WAYLAND_SNOOPER(vformat("Redirecting object argument g0x%x to local registry object l0x%x.", obj_id, new_local_id));
						body[buf_idx] = new_local_id;
						break;
					}

					if (!client->local_ids.has(obj_id)) {
						DEBUG_LOG_WAYLAND_SNOOPER(vformat("Object argument g0x%x not found, marking packet as invalid.", obj_id));
						valid = false;
						break;
					}
					body[buf_idx] = instance_id != INVALID_ID ? instance_id : client->get_local_id(obj_id);
				} else if (info->direction == ProxyDirection::COMPOSITOR) {
					if (!client->global_ids.has(obj_id)) {
						DEBUG_LOG_WAYLAND_SNOOPER(vformat("Object argument l0x%x not found, marking packet as invalid.", obj_id));
						valid = false;
						break;
					}
					body[buf_idx] = client->get_global_id(obj_id);
				}
			} break;
		}

		++arg_idx;
		++buf_idx;
	}

	return valid;
}

// Returns whether handled.
bool WaylandEmbedderProxy::handle_request(LocalObjectHandle p_object, uint32_t p_opcode, uint32_t *msg_data, size_t msg_len) {
	WaylandObject *object = p_object.get();
	Client *client = p_object.get_client();

	ERR_FAIL_NULL_V(object, true);

	ERR_FAIL_COND_V(!p_object.is_valid(), true);

	// NOTE: Global ID may be null.
	uint32_t global_id = p_object.get_global_id();
	uint32_t local_id = p_object.get_local_id();

	CRASH_COND(object->interface == nullptr);
	const struct wl_interface *interface = object->interface;

	CRASH_COND((int)p_opcode >= interface->method_count);
	const struct wl_message message = interface->methods[p_opcode];

	DEBUG_LOG_WAYLAND_SNOOPER(vformat("Request %s::%s(%s) l0x%x -> g0x%x", interface->name, message.name, message.signature, local_id, global_id));

	uint32_t *body = msg_data + 2;
	size_t body_len = msg_len - (WORD_SIZE * 2);

	if (object->interface == &wl_display_interface && p_opcode == WL_DISPLAY_GET_REGISTRY) {
		// The gist of this is that the registry is a global and the compositor can
		// quite simply take for granted that a single client can access any global
		// bound from any registry. Let's remove all doubts by using a single
		// registry (also for efficiency) and doing fancy remaps.
		uint32_t local_registry_id = body[0];

		// Note that the registry has already been allocated in the initialization
		// routine.

		// TODO: Consider something more efficient, I think.
		uint32_t *args_buf = (uint32_t *)calloc(msg_buf.size(), sizeof *args_buf);

		// FIXME: Cleanup.
		for (size_t global_name = 0; global_name < registry_globals.size(); ++global_name) {
			RegistryGlobalInfo &global_info = registry_globals[global_name];
			const struct wl_interface *global_interface = global_info.interface;

			if (client != &clients[0] && (global_interface == &zxdg_decoration_manager_v1_interface || global_interface == &zxdg_exporter_v1_interface || global_interface == &zxdg_exporter_v2_interface || global_interface == &godot_embedding_compositor_interface)) {
				DEBUG_LOG_WAYLAND_SNOOPER(vformat("Skipped global announcement %s for embedded client.", global_interface->name));
				continue;
			}

			size_t interface_name_len = strlen(global_interface->name) + 1;

			uint32_t *write_head = args_buf;

			*write_head = global_name;
			++write_head;

			*write_head = interface_name_len;
			++write_head;

			strncpy((char *)write_head, global_interface->name, interface_name_len);
			write_head += wl_array_word_offset(interface_name_len);

			*write_head = global_info.version;

			size_t arg_buf_words = write_head - args_buf + 1;

			send_wayland_message(client->socket, local_registry_id, 0, args_buf, arg_buf_words);
		}
		free(args_buf);

		// FIXME: Multiple associations.
		client->local_ids[REGISTRY_ID] = local_registry_id;
		client->global_ids[local_registry_id] = REGISTRY_ID;
		client->get_object(local_registry_id)->interface = &wl_registry_interface;
		client->get_object(local_registry_id)->version = 1;

		return true;
	}

	if (object->interface == &wl_registry_interface) {
		if (p_opcode == WL_REGISTRY_BIND) {
			// [Request] wl_registry::bind(usun)
			uint32_t global_name = body[0];
			uint32_t interface_name_len = body[1];
			uint32_t version = body[2 + wl_array_word_offset(interface_name_len)];
			uint32_t new_local_id_idx = 2 + wl_array_word_offset(interface_name_len) + 1;
			uint32_t new_local_id = body[new_local_id_idx];

			RegistryGlobalInfo &global_info = registry_globals[global_name];
			CRASH_COND(global_info.interface == nullptr);

			version = MIN(global_info.version, version);

			if (global_info.interface == &godot_embedding_compositor_interface) {
				if (!client->registry_globals_instances.has(global_name)) {
					client->registry_globals_instances[global_name] = {};
				}

				client->registry_globals_instances[global_name].insert(new_local_id);
				DEBUG_LOG_WAYLAND_SNOOPER("Bound embedded compositor interface.");
				client->new_fake_object(new_local_id, &godot_embedding_compositor_interface, 1);
				return true;
			}

			bool can_destroy = false;

			for (int i = 0; i < global_info.interface->method_count; ++i) {
				const struct wl_message &m = global_info.interface->methods[i];
				uint32_t destructor_version = String::to_int(m.signature);
				if ((strcmp(m.name, "destroy") == 0 || strcmp(m.name, "release") == 0) && destructor_version <= version) {
					can_destroy = true;
					break;
				}
			}

			WaylandObject *instance = nullptr;

			if (can_destroy) {
				DEBUG_LOG_WAYLAND_SNOOPER(vformat("Passthrough global bind #%d iface %s ver %d", global_name, global_info.interface->name, version));

				if (!client->registry_globals_instances.has(global_name)) {
					client->registry_globals_instances[global_name] = {};
				}

				client->registry_globals_instances[global_name].insert(new_local_id);

				// Passthrough.
				uint32_t instance_gid = client->new_object(new_local_id, global_info.interface, version);
				instance = get_object(instance_gid);

				registry_globals_names[instance_gid] = global_name;

				uint32_t *args_buf = (uint32_t *)calloc(msg_buf.size(), sizeof *args_buf);

				uint32_t *write_head = args_buf;

				*write_head = global_info.compositor_name;
				++write_head;

				*write_head = interface_name_len;
				++write_head;

				strncpy((char *)write_head, (char *)(body + 2), interface_name_len);
				write_head += wl_array_word_offset(interface_name_len);

				*write_head = global_info.version;
				++write_head;

				*write_head = instance_gid;

				size_t arg_buf_words = write_head - args_buf + 1;

				send_wayland_message(compositor_socket, REGISTRY_ID, WL_REGISTRY_BIND, args_buf, arg_buf_words);

				free(args_buf);
			} else {
				// Instance of a reusable object. For interfaces without a destructor
				// method.

				if (!client->registry_globals_instances.has(global_name)) {
					client->registry_globals_instances[global_name] = {};
				}

				client->registry_globals_instances[global_name].insert(new_local_id);

				if (global_info.global_id == INVALID_ID) {
					uint32_t header[2] = { REGISTRY_ID, (uint32_t)(msg_len << 16) };

					body[0] = global_info.compositor_name;

					DEBUG_LOG_WAYLAND_SNOOPER(vformat("Binding new global #%d iface %s ver %d", global_name, global_info.interface->name, version));

					global_info.global_id = new_object(global_info.interface, version);
					registry_globals_names[global_info.global_id] = global_name;

					send_raw_message(compositor_socket, { { header, sizeof header }, { body, body_len - WORD_SIZE }, { &global_info.global_id, sizeof global_info.global_id } });
				}

				CRASH_COND(global_info.global_id == INVALID_ID);

				shared_objects[global_info.interface] = global_info.global_id;

				instance = client->new_global_instance(new_local_id, global_info.interface, version);

				DEBUG_LOG_WAYLAND_SNOOPER(vformat("Instancing global #%d iface %s ver %d new id l0x%x g0x%x", global_name, global_info.interface->name, version, new_local_id, global_info.global_id));

				// Some interfaces report their state as soon as they're bound. Since
				// instances are handled by us, we need to track and report the relevant
				// data ourselves.
				if (global_info.interface == &wl_drm_interface) {
					client->send_wl_drm_state(new_local_id, (WaylandDrmGlobalData *)global_info.data);
				} else if (global_info.interface == &wl_shm_interface) {
					WaylandShmGlobalData *global_data = (WaylandShmGlobalData *)global_info.data;
					CRASH_COND(global_data == nullptr);

					for (uint32_t format : global_data->formats) {
						send_wayland_message(client->socket, new_local_id, WL_SHM_FORMAT, { format });
					}
				}

				// FIXME: Multiple associations?
				client->global_ids[new_local_id] = global_info.global_id;
				client->local_ids[global_info.global_id] = new_local_id;
			}

			ERR_FAIL_NULL_V(instance, false);

			if (global_info.interface == &wl_seat_interface) {
				WaylandSeatInstanceData *new_data = memnew(WaylandSeatInstanceData);
				instance->data = new_data;
			}

			return true;
		}
	}

	if (object->interface == &wl_compositor_interface && p_opcode == WL_COMPOSITOR_CREATE_SURFACE) {
		uint32_t new_local_id = body[0];

		WaylandSurfaceData *data = memnew(WaylandSurfaceData);
		data->client = client;

		uint32_t new_global_id = client->new_object(new_local_id, &wl_surface_interface, object->version, data);
		DEBUG_LOG_WAYLAND_SNOOPER(vformat("Keeping track of surface l0x%x g0x%x.", new_local_id, new_global_id));

		send_wayland_message(compositor_socket, global_id, p_opcode, { new_global_id });
		return true;
	}

	if (object->interface == &wl_surface_interface) {
		WaylandSurfaceData *surface_data = (WaylandSurfaceData *)object->data;
		CRASH_COND(surface_data == nullptr);

		if (p_opcode == WL_SURFACE_DESTROY) {
			WaylandSeatGlobalData *global_seat_data = (WaylandSeatGlobalData *)registry_globals[wl_seat_name].data;
			CRASH_COND(global_seat_data == nullptr);

			if (global_seat_data->pointed_surface_id == global_id) {
				global_seat_data->pointed_surface_id = INVALID_ID;
			}

			if (global_seat_data->focused_surface_id == global_id) {
				global_seat_data->focused_surface_id = INVALID_ID;
			}
		} else if (p_opcode == WL_SURFACE_COMMIT) {
			if (surface_data->role_object_handle.is_valid()) {
				WaylandObject *role_object = surface_data->role_object_handle.get();
				if (role_object && role_object->interface) {
					DEBUG_LOG_WAYLAND_SNOOPER(vformat("!!!!! Committed surface g0x%x with role object %s id l0x%x", global_id, role_object->interface->name, surface_data->role_object_handle.get_local_id()));
				}

				if (role_object && role_object->interface == &xdg_toplevel_interface) {
					XdgToplevelData *toplevel_data = (XdgToplevelData *)role_object->data;
					CRASH_COND(toplevel_data == nullptr);
					// xdg shell spec requires clients to first send data and then commit the
					// surface.

					if (toplevel_data->is_embedded() && !toplevel_data->configured) {
						toplevel_data->configured = true;
						// xdg_surface::configure
						send_wayland_message(client->socket, toplevel_data->xdg_surface_handle.get_local_id(), 0, { serial_counter++ });
					}
				}
			}

			send_wayland_message(compositor_socket, global_id, p_opcode, {});
			return true;
		}
	}

	if (object->interface == &wl_seat_interface) {
		uint32_t global_seat_name = registry_globals_names[global_id];

		RegistryGlobalInfo &seat_global_info = registry_globals[global_seat_name];
		WaylandSeatGlobalData *global_data = (WaylandSeatGlobalData *)seat_global_info.data;
		CRASH_COND(global_data == nullptr);

		WaylandSeatInstanceData *instance_data = (WaylandSeatInstanceData *)object->data;
		CRASH_COND(instance_data == nullptr);

		if (p_opcode == WL_SEAT_GET_POINTER) {
			CRASH_COND(global_id == INVALID_ID);
			// [Request] wl_seat::get_pointer(n);
			uint32_t new_local_id = body[0];

			WaylandPointerData *new_data = memnew(WaylandPointerData);
			new_data->wl_seat_id = global_id;

			uint32_t new_global_id = client->new_object(new_local_id, &wl_pointer_interface, object->version, new_data);
			instance_data->wl_pointer_id = new_global_id;

			send_wayland_message(compositor_socket, global_id, p_opcode, { new_global_id });

			return true;
		}

		if (p_opcode == WL_SEAT_GET_KEYBOARD) {
			CRASH_COND(global_id == INVALID_ID);
			// [Request] wl_seat::get_pointer(n);
			uint32_t new_local_id = body[0];

			WaylandKeyboardData *new_data = memnew(WaylandKeyboardData);
			new_data->wl_seat_id = global_id;

			uint32_t new_global_id = client->new_object(new_local_id, &wl_keyboard_interface, object->version, new_data);
			instance_data->wl_keyboard_id = new_global_id;

			send_wayland_message(compositor_socket, global_id, p_opcode, { new_global_id });

			return true;
		}
	}

	if (object->interface == &xdg_wm_base_interface) {
		if (p_opcode == XDG_WM_BASE_CREATE_POSITIONER) {
			uint32_t new_local_id = body[0];
			uint32_t new_global_id = client->new_object(new_local_id, &xdg_positioner_interface, object->version, memnew(XdgPositionerData));

			send_wayland_message(compositor_socket, global_id, p_opcode, { new_global_id });
			return true;
		}

		if (p_opcode == XDG_WM_BASE_GET_XDG_SURFACE) {
			// [Request] xdg_wm_base::get_xdg_surface(no).
			uint32_t new_local_id = body[0];
			uint32_t surface_id = body[1];

			uint32_t global_surface_id = client->get_global_id(surface_id);

			bool fake = (client != &clients[0]);

			XdgSurfaceData *data = memnew(XdgSurfaceData);
			data->wl_surface_id = global_surface_id;

			if (fake) {
				client->new_fake_object(new_local_id, &xdg_surface_interface, object->version, data);
				DEBUG_LOG_WAYLAND_SNOOPER(vformat("Created fake xdg_surface l0x%x for surface l0x%x", new_local_id, surface_id));
			} else {
				uint32_t new_global_id = client->new_object(new_local_id, &xdg_surface_interface, object->version, data);
				DEBUG_LOG_WAYLAND_SNOOPER(vformat("Created real xdg_surface l0x%x g0x%x for surface l0x%x", new_local_id, new_global_id, surface_id));

				send_wayland_message(compositor_socket, global_id, p_opcode, { new_global_id, global_surface_id });
			}

			return true;
		}
	}

	if (object->interface == &xdg_surface_interface) {
		XdgSurfaceData *xdg_surf_data = (XdgSurfaceData *)object->data;
		CRASH_COND(xdg_surf_data == nullptr);

		WaylandSurfaceData *surface_data = (WaylandSurfaceData *)get_object(xdg_surf_data->wl_surface_id)->data;
		CRASH_COND(surface_data == nullptr);

		bool is_embedded = client->fake_objects.has(local_id);

		// FIXME: Adjust positioner for embedded windows.
		if (p_opcode == XDG_SURFACE_GET_POPUP && is_embedded) {
			// [Request] xdg_surface::get_popup(no?o).

			uint32_t new_local_id = body[0];
			uint32_t local_parent_id = body[1];
			uint32_t local_positioner_id = body[2];

			surface_data->role_object_handle = LocalObjectHandle(client, new_local_id);

			{
				// Popups are real, time to actually instantiate an xdg_surface.
				WaylandObject copy = *object;
				client->fake_objects.erase(local_id);

				global_id = client->new_object(local_id, copy.interface, copy.version, copy.data);
				object = get_object(new_local_id);

				// xdg_wm_base::get_xdg_surface(no);
				send_wayland_message(compositor_socket, xdg_wm_base_id, 2, { global_id, xdg_surf_data->wl_surface_id });
			}

			uint32_t new_global_id = client->new_object(new_local_id, &xdg_popup_interface, object->version);

			uint32_t global_parent_id = INVALID_ID;
			if (local_parent_id != INVALID_ID) {
				XdgSurfaceData *parent_xdg_surf_data = (XdgSurfaceData *)client->get_object(local_parent_id)->data;
				CRASH_COND(parent_xdg_surf_data == nullptr);

				WaylandSurfaceData *parent_surface_data = (WaylandSurfaceData *)get_object(parent_xdg_surf_data->wl_surface_id)->data;
				CRASH_COND(parent_surface_data == nullptr);

				WaylandObject *parent_role_obj = parent_surface_data->role_object_handle.get();
				CRASH_COND(parent_role_obj == nullptr);

				if (parent_role_obj->interface == &xdg_toplevel_interface) {
					XdgToplevelData *parent_toplevel_data = (XdgToplevelData *)parent_role_obj->data;
					CRASH_COND(parent_toplevel_data == nullptr);

					if (parent_toplevel_data->is_embedded()) {
						// Embedded windows are subsurfaces of a parent window. We need to
						// "redirect" the popup request on the parent window and adjust the
						// positioner properly if needed.

						XdgToplevelData *main_parent_toplevel_data = (XdgToplevelData *)parent_toplevel_data->parent_handle.get()->data;
						CRASH_COND(main_parent_toplevel_data == nullptr);

						global_parent_id = main_parent_toplevel_data->xdg_surface_handle.get_global_id();

						WaylandSubsurfaceData *subsurf_data = (WaylandSubsurfaceData *)get_object(parent_toplevel_data->wl_subsurface_id)->data;
						CRASH_COND(subsurf_data == nullptr);

						XdgPositionerData *pos_data = (XdgPositionerData *)client->get_object(local_positioner_id)->data;
						CRASH_COND(pos_data == nullptr);

						// FIXME: Clients receive skewed data on popup configure. This is helpful
						// for getting child popups already in place "for free" but this is
						// obviously wrong. Fixing this will require some state tracking for popups
						// but first I'd really, /really/ like to tidy up state handling.
						Point2i adj_pos = subsurf_data->position + pos_data->anchor_rect.position;

						// xdg_positioner::set_anchor_rect
						send_wayland_message(compositor_socket, client->get_global_id(local_positioner_id), 2, { (uint32_t)adj_pos.x, (uint32_t)adj_pos.y, (uint32_t)pos_data->anchor_rect.size.width, (uint32_t)pos_data->anchor_rect.size.height });
					}
				} else {
					global_parent_id = client->get_global_id(local_parent_id);
				}
			}

			send_wayland_message(compositor_socket, global_id, p_opcode, { new_global_id, global_parent_id, client->get_global_id(local_positioner_id) });
			return true;
		}

		if (p_opcode == XDG_SURFACE_GET_TOPLEVEL) {
			// [Request] xdg_surface::get_toplevel(n).
			uint32_t new_local_id = body[0];

			surface_data->role_object_handle = LocalObjectHandle(client, new_local_id);

			XdgToplevelData *data = memnew(XdgToplevelData);
			data->xdg_surface_handle = LocalObjectHandle(client, local_id);

			if (is_embedded) {
				client->new_fake_object(new_local_id, &xdg_toplevel_interface, object->version, data);
				client->embedded_window_id = new_local_id;

				// godot_embedded_client::window_embedded()
				send_wayland_message(clients[0].socket, client->embedded_client_id, 1, {});
			} else {
				uint32_t new_global_id = client->new_object(new_local_id, &xdg_toplevel_interface, object->version, data);

				if (main_toplevel_id == 0) {
					main_toplevel_id = new_global_id;
					DEBUG_LOG_WAYLAND_SNOOPER(vformat("main toplevel set to gx0%x.", main_toplevel_id));
				}

				send_wayland_message(compositor_socket, global_id, p_opcode, { new_global_id });
			}

			return true;
		}
	}

	if (object->interface == &xdg_positioner_interface) {
		XdgPositionerData *pos_data = (XdgPositionerData *)object->data;
		CRASH_COND(pos_data == nullptr);

		if (p_opcode == XDG_POSITIONER_SET_ANCHOR_RECT) {
			// Args: int x, int y, int width, int height.
			pos_data->anchor_rect = Rect2i(body[0], body[1], body[2], body[3]);

			send_wayland_message(compositor_socket, global_id, p_opcode, body, body_len);
			return true;
		}
	}

	if (object->interface == &xdg_toplevel_interface && p_opcode == XDG_TOPLEVEL_DESTROY) {
		if (client->fake_objects.has(local_id)) {
			XdgToplevelData *data = (XdgToplevelData *)object->data;
			CRASH_COND(data == nullptr);

			// wl_display::delete_id
			send_wayland_message(client->socket, local_id, p_opcode, {});

			if (local_id == client->embedded_window_id) {
				client->embedded_window_id = 0;
			}

			send_wayland_message(compositor_socket, data->wl_subsurface_id, WL_SUBSURFACE_DESTROY, {});

			client->delete_object(local_id);

			return true;
		}
	}

	if (interface == &zwp_pointer_constraints_v1_interface) {
		// FIXME: This implementation leaves no way of unlocking the pointer when
		// embedded into the main window. We might need to be a bit more invasive.
		if (p_opcode == ZWP_POINTER_CONSTRAINTS_V1_LOCK_POINTER) {
			// [Request] zwp_pointer_constraints_v1::lock_pointer(nooou).

			uint32_t new_local_id = body[0];
			uint32_t local_surface_id = body[1];
			uint32_t local_pointer_id = body[2];
			uint32_t lifetime = body[4];

			WaylandSurfaceData *surf_data = (WaylandSurfaceData *)client->get_object(local_surface_id)->data;
			CRASH_COND(surf_data == nullptr);

			WaylandObject *role_obj = surf_data->role_object_handle.get();
			CRASH_COND(role_obj == nullptr);

			if (role_obj->interface == &xdg_toplevel_interface) {
				XdgToplevelData *toplevel_data = (XdgToplevelData *)role_obj->data;
				CRASH_COND(toplevel_data == nullptr);

				if (!toplevel_data->is_embedded()) {
					// Passthrough.
					return true;
				}

				// Subsurfaces don't normally work, at least on sway, as the locking
				// condition might rely on focus, which they don't get. We can remap them to
				// the parent surface and set a region though.

				XdgToplevelData *parent_data = (XdgToplevelData *)toplevel_data->parent_handle.get()->data;
				CRASH_COND(parent_data == nullptr);

				XdgSurfaceData *parent_xdg_surf_data = (XdgSurfaceData *)parent_data->xdg_surface_handle.get()->data;
				CRASH_COND(parent_xdg_surf_data == nullptr);

				WaylandSubsurfaceData *subsurf_data = (WaylandSubsurfaceData *)get_object(toplevel_data->wl_subsurface_id)->data;
				CRASH_COND(subsurf_data == nullptr);

				uint32_t new_global_id = client->new_object(new_local_id, &zwp_locked_pointer_v1_interface, object->version);

				uint32_t x = subsurf_data->position.x;
				uint32_t y = subsurf_data->position.y;
				uint32_t width = toplevel_data->size.width;
				uint32_t height = toplevel_data->size.height;

				// NOTE: At least on sway I can't seem to be able to get this region
				// working but the calls check out.
				DEBUG_LOG_WAYLAND_SNOOPER(vformat("creating custom region x%d y%d width%d height%d", x, y, width, height));

				uint32_t new_region_id = next_global_id();
				get_object(new_region_id)->interface = &wl_region_interface;
				get_object(new_region_id)->version = get_object(wl_compositor_id)->version;

				// wl_compostor::create_region(n).
				send_wayland_message(compositor_socket, wl_compositor_id, 1, { new_region_id });

				// wl_region::add(iiii).
				send_wayland_message(compositor_socket, new_region_id, 1, { x, y, width, height });

				send_wayland_message(compositor_socket, global_id, p_opcode, { new_global_id, parent_xdg_surf_data->wl_surface_id, client->get_global_id(local_pointer_id), new_region_id, lifetime });

				// wl_region::destroy().
				send_wayland_message(compositor_socket, new_region_id, 0, {});

				return true;
			}
		}
	}

	if (interface == &godot_embedded_client_interface) {
		EmbeddedClientData *eclient_data = (EmbeddedClientData *)object->data;
		CRASH_COND(eclient_data == nullptr);

		Client *eclient = eclient_data->client;
		CRASH_COND(eclient == nullptr);

		if (p_opcode == GODOT_EMBEDDED_CLIENT_DESTROY) {
			if (!eclient_data->disconnected) {
				// FIXME: Switch to index-based client addressing for everything else.
				for (size_t i = 0; i < clients.size(); ++i) {
					if (clients[i].pid == eclient->pid) {
						client_disconnect(i);
					}
				}
			}

			client->delete_object(local_id);

			return true;
		}

		if (eclient_data->disconnected) {
			// Object is inert.
			return true;
		}

		CRASH_COND(eclient->embedded_window_id == 0);

		XdgToplevelData *toplevel_data = (XdgToplevelData *)eclient->get_object(eclient->embedded_window_id)->data;
		CRASH_COND(toplevel_data == nullptr);

		if (p_opcode == GODOT_EMBEDDED_CLIENT_SET_EMBEDDED_WINDOW_RECT && toplevel_data->wl_subsurface_id != INVALID_ID) {
			uint32_t x = body[0];
			uint32_t y = body[1];
			uint32_t width = body[2];
			uint32_t height = body[3];

			DEBUG_LOG_WAYLAND_SNOOPER("Received?");

			WaylandSubsurfaceData *subsurf_data = (WaylandSubsurfaceData *)get_object(toplevel_data->wl_subsurface_id)->data;
			CRASH_COND(subsurf_data == nullptr);

			toplevel_data->size.width = width;
			toplevel_data->size.height = height;

			subsurf_data->position.x = x;
			subsurf_data->position.y = y;

			// wl_subsurface::set_position
			send_wayland_message(compositor_socket, toplevel_data->wl_subsurface_id, 1, { x, y });

			// xdg_toplevel::configure
			send_wayland_message(eclient->socket, eclient->embedded_window_id, 0, { width, height, 0 });

			return true;
		} else if (p_opcode == GODOT_EMBEDDED_CLIENT_SET_EMBEDDED_WINDOW_PARENT) {
			uint32_t main_client_parent_id = body[0];

			if (toplevel_data->parent_handle.get_local_id() == main_client_parent_id) {
				return true;
			}

			if (main_client_parent_id == INVALID_ID && toplevel_data->wl_subsurface_id != INVALID_ID) {
				// Window hiding logic.

				// wl_subsurface::destroy()
				send_wayland_message(compositor_socket, toplevel_data->wl_subsurface_id, 0, {});

				toplevel_data->parent_handle.invalidate();
				toplevel_data->wl_subsurface_id = INVALID_ID;

				return true;
			}

			XdgToplevelData *parent_toplevel_data = (XdgToplevelData *)client->get_object(main_client_parent_id)->data;
			CRASH_COND(parent_toplevel_data == nullptr);
			XdgSurfaceData *parent_xdg_surf_data = (XdgSurfaceData *)parent_toplevel_data->xdg_surface_handle.get()->data;
			CRASH_COND(parent_xdg_surf_data == nullptr);

			XdgSurfaceData *xdg_surf_data = (XdgSurfaceData *)toplevel_data->xdg_surface_handle.get()->data;
			CRASH_COND(xdg_surf_data == nullptr);

			if (toplevel_data->wl_subsurface_id != INVALID_ID) {
				// wl_subsurface::destroy()
				send_wayland_message(compositor_socket, toplevel_data->wl_subsurface_id, 0, {});
			}

			uint32_t new_sub_id = next_global_id();
			WaylandObject *new_sub_object = get_object(new_sub_id);
			new_sub_object->interface = &wl_subsurface_interface;
			new_sub_object->data = memnew(WaylandSubsurfaceData);
			new_sub_object->version = get_object(wl_subcompositor_id)->version;

			toplevel_data->wl_subsurface_id = new_sub_id;
			toplevel_data->parent_handle = LocalObjectHandle(&clients[0], main_client_parent_id);

			DEBUG_LOG_WAYLAND_SNOOPER(vformat("Binding subsurface g0x%x.", new_sub_id));

			// wl_subcompositor::get_subsurface
			send_wayland_message(compositor_socket, wl_subcompositor_id, 1, { new_sub_id, xdg_surf_data->wl_surface_id, parent_xdg_surf_data->wl_surface_id });

			// wl_subsurface::set_desync
			send_wayland_message(compositor_socket, new_sub_id, 5, {});

			return true;
		} else if (p_opcode == GODOT_EMBEDDED_CLIENT_FOCUS_WINDOW) {
			XdgSurfaceData *xdg_surf_data = (XdgSurfaceData *)toplevel_data->xdg_surface_handle.get()->data;
			CRASH_COND(xdg_surf_data == nullptr);

			RegistryGlobalInfo &global_seat_info = registry_globals[wl_seat_name];
			WaylandSeatGlobalData *global_seat_data = (WaylandSeatGlobalData *)global_seat_info.data;

			if (global_seat_data->focused_surface_id != INVALID_ID) {
				seat_name_leave_surface(wl_seat_name, global_seat_data->focused_surface_id);
			}

			global_seat_data->focused_surface_id = xdg_surf_data->wl_surface_id;

			seat_name_enter_surface(wl_seat_name, xdg_surf_data->wl_surface_id);
		} else if (p_opcode == GODOT_EMBEDDED_CLIENT_EMBEDDED_WINDOW_REQUEST_CLOSE) {
			// xdg_toplevel::close
			send_wayland_message(eclient->socket, eclient->embedded_window_id, 1, {});

			return true;
		}
	}

	// Server-allocated objects are a bit annoying to handle for us. Right now we
	// use an heuristic. See: https://ppaalanen.blogspot.com/2014/07/wayland-protocol-design-object-lifespan.html
	if (strcmp(message.name, "destroy") == 0 || strcmp(message.name, "release") == 0) {
		// TODO: More robust server-side destruction heuristic?
		if (local_id & 0xff000000) {
			DEBUG_LOG_WAYLAND_SNOOPER(vformat("!!!!!! Deallocating server object l0x%x", local_id));
			client->delete_object(local_id);
		}

		if (shared_objects.has(object->interface)) {
			// We must not delete shared objects.
			client->delete_object(local_id);
			return true;
		}

		if (global_id != INVALID_ID) {
			send_wayland_message(compositor_socket, global_id, p_opcode, {});
		}

		return true;
	}

	if (client->fake_objects.has(local_id)) {
		// Object is fake, we're done.
		DEBUG_LOG_WAYLAND_SNOOPER("Dropping unhandled request for fake object.");
		return true;
	}

	if (global_id == INVALID_ID) {
		DEBUG_LOG_WAYLAND_SNOOPER("Dropping request with invalid global object id");
		return true;
	}

	return false;
}

bool WaylandEmbedderProxy::handle_event(uint32_t p_global_id, LocalObjectHandle p_local_handle, uint32_t p_opcode, uint32_t *msg_data, size_t msg_len) {
	WaylandObject *global_object = get_object(p_global_id);
	CRASH_COND_MSG(global_object == nullptr, "Compositor messages must always have a global object.");

	uint32_t *body = msg_data + 2;
	size_t body_len = msg_len - (WORD_SIZE * 2);

	// FIXME: Make sure that it makes sense to track this protocol. Not only it is
	// old and getting deprecated, but I can't even get this code branch to hit
	// probably because, at the time of writing, we only get the "main" display
	// through the proxy.
	if (global_object->interface == &wl_drm_interface) {
		// wl_drm can't ever be destroyed, so we need to track its state as it's going
		// to be instanced at least few times.
		uint32_t global_name = registry_globals_names[p_global_id];
		WaylandDrmGlobalData *global_data = (WaylandDrmGlobalData *)registry_globals[global_name].data;
		CRASH_COND(global_data == nullptr);

		if (p_opcode == WL_DRM_DEVICE) {
			// signature: s
			uint32_t name_len = body[0];
			uint8_t *name = (uint8_t *)(body + 1);
			global_data->device = String::utf8((const char *)name, name_len);

			return false;
		}

		if (p_opcode == WL_DRM_FORMAT) {
			// signature: u
			uint32_t format = body[0];
			global_data->formats.push_back(format);

			return false;
		}

		if (p_opcode == WL_DRM_AUTHENTICATED) {
			// signature: N/A
			global_data->authenticated = true;

			return false;
		}

		if (p_opcode == WL_DRM_CAPABILITIES) {
			// signature: u
			uint32_t capabilities = body[0];
			global_data->capabilities = capabilities;
		}

		return false;
	}

	if (global_object->interface == &wl_shm_interface) {
		uint32_t global_name = registry_globals_names[p_global_id];
		WaylandShmGlobalData *global_data = (WaylandShmGlobalData *)registry_globals[global_name].data;
		CRASH_COND(global_data == nullptr);

		if (p_opcode == WL_SHM_FORMAT) {
			// Signature: u
			uint32_t format = body[0];
			global_data->formats.push_back(format);
		}
	}

	if (!p_local_handle.is_valid()) {
		// Some requests might not have a valid local object handle for various
		// reasons, such as when certain events are directed to this proxy or when the
		// destination client of a message disconnected in the meantime.

		if (global_object->interface == &wl_display_interface) {
			if (p_opcode == WL_DISPLAY_DELETE_ID) {
				// [Event] wl_display::delete_id(u)
				uint32_t global_delete_id = body[0];
				DEBUG_LOG_WAYLAND_SNOOPER(vformat("Delete ID event g0x%x (no client)", global_delete_id));

				delete_object(global_delete_id);
			} else if (p_opcode == WL_DISPLAY_ERROR) {
				// [Event] wl_display::error(ous)
				uint32_t obj_id = body[0];
				uint32_t err_code = body[1];

				CRASH_NOW_MSG(vformat("Error obj g0x%x code %d: %s", obj_id, err_code, (const char *)(body + 3)));
			}
		}

		if (global_object->interface == &wl_callback_interface && p_opcode == WL_CALLBACK_DONE) {
			delete_object(p_global_id);

			if (sync_callback_id != INVALID_ID && p_global_id == sync_callback_id) {
				sync_callback_id = 0;
				DEBUG_LOG_WAYLAND_SNOOPER("Sync response received");
				return true;
			}
		}

		// TODO: wl_registry::global_remove(u)
		if (global_object->interface == &wl_registry_interface && p_opcode == WL_REGISTRY_GLOBAL) {
			// [Event] wl_registry::global(usu).

			uint32_t global_name = body[0];
			uint32_t interface_name_len = body[1];
			const char *interface_name = (const char *)(body + 2);
			uint32_t global_version = body[2 + wl_array_word_offset(interface_name_len)];

			DEBUG_LOG_WAYLAND_SNOOPER("Global %s %d", interface_name, global_version);

			const struct wl_interface *global_interface = wl_interface_from_string(interface_name, interface_name_len);
			if (global_interface) {
				RegistryGlobalInfo global_info = {};
				global_info.interface = global_interface;
				global_info.version = MIN(global_version, (uint32_t)global_interface->version);
				DEBUG_LOG_WAYLAND_SNOOPER("Clamped global %s %d", interface_name, global_info.version);
				global_info.compositor_name = global_name;

				if (global_info.interface == &wl_shm_interface) {
					// FIXME: Cleanup.
					DEBUG_LOG_WAYLAND_SNOOPER("Allocating global wl_shm data.");
					global_info.data = memnew(WaylandShmGlobalData);
				}

				if (global_info.interface == &wl_seat_interface) {
					// FIXME: Cleanup.
					DEBUG_LOG_WAYLAND_SNOOPER("Allocating global wl_seat data.");
					global_info.data = memnew(WaylandSeatGlobalData);
					wl_seat_name = registry_globals.size();
				}

				if (global_info.interface == &wl_drm_interface) {
					// FIXME: Cleanup.
					DEBUG_LOG_WAYLAND_SNOOPER("Allocating global wl_drm data.");
					global_info.data = memnew(WaylandDrmGlobalData);
				}

				// FIXME: Ensure that no duplicate entries get added (VSet?)
				registry_globals.push_back(global_info);

				int new_global_name = registry_globals.size() - 1;

				// We need some interfaces directly. It's better to bind a "copy" ourselves
				// than to wait for the client to ask one. Since I'm lazy, we can exploit
				// the fact that a bind request is the global event with the new id tacked on.
				if (global_interface == &xdg_wm_base_interface && xdg_wm_base_id == 0) {
					xdg_wm_base_id = new_object(&xdg_wm_base_interface, global_info.version);
					DEBUG_LOG_WAYLAND_SNOOPER(vformat("Binding global xdg_wm_base as g0x%x version %d", xdg_wm_base_id, global_info.version));

					uint32_t header[2] = { p_global_id, 0 };
					header[1] = (sizeof header + body_len + sizeof xdg_wm_base_id) << 16; // opcode is 0.

					registry_globals[new_global_name].global_id = xdg_wm_base_id;
					registry_globals_names[xdg_wm_base_id] = new_global_name;

					send_raw_message(compositor_socket, { { header, 8 }, { body, body_len }, { &xdg_wm_base_id, sizeof xdg_wm_base_id } });

					return true;
				}

				if (global_interface == &wl_compositor_interface && wl_compositor_id == 0) {
					wl_compositor_id = new_object(&wl_compositor_interface, global_info.version);
					DEBUG_LOG_WAYLAND_SNOOPER(vformat("Binding global wl_compositor as g0x%x version %d", wl_compositor_id, global_info.version));

					uint32_t header[2] = { p_global_id, 0 };
					header[1] = (sizeof header + body_len + sizeof wl_compositor_id) << 16; // opcode is 0.

					registry_globals[new_global_name].global_id = wl_compositor_id;
					registry_globals_names[wl_compositor_id] = new_global_name;

					send_raw_message(compositor_socket, { { header, 8 }, { body, body_len }, { &wl_compositor_id, sizeof wl_compositor_id } });

					return true;
				}

				if (global_interface == &wl_subcompositor_interface && wl_subcompositor_id == 0) {
					wl_subcompositor_id = new_object(&wl_subcompositor_interface, global_info.version);
					DEBUG_LOG_WAYLAND_SNOOPER(vformat("Binding global wl_subcompositor as g0x%x version %d", wl_subcompositor_id, global_info.version));

					uint32_t header[2] = { p_global_id, 0 };
					header[1] = (sizeof header + body_len + sizeof wl_subcompositor_id) << 16; // opcode is 0.

					registry_globals[new_global_name].global_id = wl_subcompositor_id;
					registry_globals_names[wl_subcompositor_id] = new_global_name;

					send_raw_message(compositor_socket, { { header, 8 }, { body, body_len }, { &wl_subcompositor_id, sizeof wl_subcompositor_id } });

					return true;
				}
			} else {
				DEBUG_LOG_WAYLAND_SNOOPER("Skipping unknown global %s %d.", interface_name, global_version);
				return true;
			}
		}

		DEBUG_LOG_WAYLAND_SNOOPER("No valid local object handle, falling back to generic handler.");
		return false;
	}

	Client *client = p_local_handle.get_client();

	CRASH_COND(client == nullptr);

	WaylandObject *object = p_local_handle.get();

	if (global_object->interface == &wl_display_interface) {
		if (p_opcode == WL_DISPLAY_DELETE_ID) {
			// [Event] wl_display::delete_id(u)
			uint32_t global_delete_id = body[0];
			uint32_t local_delete_id = client->get_local_id(global_delete_id);
			DEBUG_LOG_WAYLAND_SNOOPER(vformat("Delete ID event g0x%x l0x%x", global_delete_id, local_delete_id));
			if (local_delete_id == INVALID_ID) {
				// No idea what this object is, might be of the other client.
				return true;
			}

			client->delete_object(local_delete_id);

			send_wayland_message(client->socket, DISPLAY_ID, WL_DISPLAY_DELETE_ID, { local_delete_id });

			return true;
		}

		return false;
	}

	if (object->interface == &wl_keyboard_interface) {
		WaylandKeyboardData *data = (WaylandKeyboardData *)object->data;
		CRASH_COND(data == nullptr);

		uint32_t global_seat_name = registry_globals_names[data->wl_seat_id];
		RegistryGlobalInfo &global_seat_info = registry_globals[global_seat_name];
		WaylandSeatGlobalData *global_seat_data = (WaylandSeatGlobalData *)global_seat_info.data;
		CRASH_COND(global_seat_data == nullptr);

		if (p_opcode == WL_KEYBOARD_ENTER) {
			// [Event] wl_keyboard::enter(uoa)
			uint32_t surface = body[1];

			if (global_seat_data->focused_surface_id != surface) {
				DEBUG_LOG_WAYLAND_SNOOPER(vformat("Focused g0x%x", surface));
				global_seat_data->focused_surface_id = surface;
			}
		} else if (p_opcode == WL_KEYBOARD_LEAVE) {
			// [Event] wl_keyboard::leave(uo)
			uint32_t surface = body[1];

			if (global_seat_data->focused_surface_id == surface) {
				global_seat_data->focused_surface_id = INVALID_ID;
			}
		} else if (p_opcode == WL_KEYBOARD_KEY) {
			// NOTE: modifiers event can be sent even without focus, according to the
			// spec, so there's no need to skip it.
			if (global_seat_data->focused_surface_id != INVALID_ID && !client->local_ids.has(global_seat_data->focused_surface_id)) {
				DEBUG_LOG_WAYLAND_SNOOPER(vformat("skipped wl_keyboard event due to unfocused surface 0x%x", global_seat_data->focused_surface_id));
				return true;
			}
		}

		return false;
	}

	if (object->interface == &wl_pointer_interface) {
		WaylandPointerData *data = (WaylandPointerData *)object->data;
		CRASH_COND(data == nullptr);

		uint32_t global_seat_name = registry_globals_names[data->wl_seat_id];
		RegistryGlobalInfo &global_seat_info = registry_globals[global_seat_name];
		WaylandSeatGlobalData *global_seat_data = (WaylandSeatGlobalData *)global_seat_info.data;
		CRASH_COND(global_seat_data == nullptr);

		WaylandSeatInstanceData *seat_data = (WaylandSeatInstanceData *)object->data;
		CRASH_COND(seat_data == nullptr);

		if (p_opcode == WL_POINTER_BUTTON && global_seat_data->pointed_surface_id != INVALID_ID) {
			// [Event] wl_pointer::button(uuuu);
			uint32_t button = body[2];
			uint32_t state = body[3];

			DEBUG_LOG_WAYLAND_SNOOPER(vformat("Button %d state %d on surface g0x%x (focused g0x%x)", button, state, global_seat_data->pointed_surface_id, global_seat_data->focused_surface_id));

			bool client_pointed = client->local_ids.has(global_seat_data->pointed_surface_id);

			if (button != BTN_LEFT || state != WL_POINTER_BUTTON_STATE_RELEASED) {
				return !client_pointed;
			}

			if (global_seat_data->focused_surface_id == global_seat_data->pointed_surface_id) {
				return !client_pointed;
			}

			if (!global_surface_is_window(global_seat_data->pointed_surface_id)) {
				return !client_pointed;
			}

			if (global_seat_data->focused_surface_id != INVALID_ID) {
				seat_name_leave_surface(global_seat_name, global_seat_data->focused_surface_id);
			}

			global_seat_data->focused_surface_id = global_seat_data->pointed_surface_id;
			seat_name_enter_surface(global_seat_name, global_seat_data->focused_surface_id);
		} else if (p_opcode == WL_POINTER_ENTER) {
			// [Event] wl_pointer::enter(uoff).
			uint32_t surface = body[1];
			WaylandSurfaceData *surface_data = (WaylandSurfaceData *)get_object(surface)->data;
			CRASH_COND(surface_data == nullptr);

			if (global_seat_data->pointed_surface_id != surface) {
				DEBUG_LOG_WAYLAND_SNOOPER(vformat("Pointer (g0x%x seat g0x%x): pointed surface old g0x%x new g0x%x", p_global_id, data->wl_seat_id, global_seat_data->pointed_surface_id, surface));

				global_seat_data->pointed_surface_id = surface;
			}
		} else if (p_opcode == WL_POINTER_LEAVE) {
			// [Event] wl_pointer::leave(uo).
			uint32_t surface = body[1];
			WaylandSurfaceData *surface_data = (WaylandSurfaceData *)get_object(surface)->data;
			CRASH_COND(surface_data == nullptr);

			if (global_seat_data->pointed_surface_id == surface) {
				DEBUG_LOG_WAYLAND_SNOOPER(vformat("Pointer (g0x%x seat g0x%x): g0x%x -> g0x%x", p_global_id, data->wl_seat_id, global_seat_data->pointed_surface_id, INVALID_ID));
				global_seat_data->pointed_surface_id = INVALID_ID;
			}
		}

		return false;
	}

	return false;
}

bool WaylandEmbedderProxy::handle_msg_info(Client *client, const struct msg_info *info, uint32_t *buf, int *fds_requested) {
	CRASH_COND(info == nullptr);
	CRASH_COND(fds_requested == nullptr);
	CRASH_COND_MSG(info->direction == ProxyDirection::COMPOSITOR && client == nullptr, "Wait, where did this message come from?");

	*fds_requested = 0;

	WaylandObject *object = nullptr;

	uint32_t global_id = INVALID_ID;
	if (info->direction == ProxyDirection::CLIENT) {
		global_id = info->raw_id;
	} else if (info->direction == ProxyDirection::COMPOSITOR) {
		global_id = client->get_global_id(info->raw_id);
	}

	if (global_id != INVALID_ID) {
		object = get_object(global_id);
	} else if (client) {
		object = client->get_object(info->raw_id);
	}

	CRASH_COND_MSG(object == nullptr, vformat("No object found for r0x%x", info->raw_id));

	const struct wl_interface *interface = nullptr;
	interface = object->interface;
	CRASH_COND_MSG(interface == nullptr, vformat("Object r0x%x has no interface", info->raw_id));

	const struct wl_message *message = nullptr;
	if (info->direction == ProxyDirection::CLIENT) {
		CRASH_COND(info->opcode >= interface->event_count);
		message = &interface->events[info->opcode];
	} else {
		CRASH_COND(info->opcode >= interface->method_count);
		message = &interface->methods[info->opcode];
	}
	CRASH_COND(message == nullptr);

	*fds_requested = String(message->signature).count("h");
	LocalVector<int> sent_fds;

	if (*fds_requested > 0) {
		DEBUG_LOG_WAYLAND_SNOOPER(vformat("Requested %d FDs.", *fds_requested));

		List<int> &fd_queue = info->direction == ProxyDirection::COMPOSITOR ? client->fds : compositor_fds;
		for (int i = 0; i < *fds_requested; ++i) {
			CRASH_COND_MSG(fd_queue.is_empty(), "Out of FDs.");
			DEBUG_LOG_WAYLAND_SNOOPER(vformat("Fetching FD %d.", fd_queue.front()->get()));
			sent_fds.push_back(fd_queue.front()->get());
			fd_queue.pop_front();
		}

		DEBUG_LOG_WAYLAND_SNOOPER(vformat("Remaining FDs: %d.", fd_queue.size()));
	}

	if (info->direction == ProxyDirection::COMPOSITOR) {
		if (handle_request(LocalObjectHandle(client, info->raw_id), info->opcode, buf, info->size)) {
			DEBUG_LOG_WAYLAND_SNOOPER("Custom handler success.");
			return false;
		}

		if (global_id != INVALID_ID) {
			buf[0] = global_id;
		}

		DEBUG_LOG_WAYLAND_SNOOPER("Falling back to generic handler.");

		if (handle_generic_msg(client, interface, message, info, buf)) {
			send_raw_message(compositor_socket, { { buf, info->size } }, sent_fds);
		}
	} else {
		uint32_t global_name = 0;

		bool is_global = false;
		if (registry_globals_names.has(global_id)) {
			global_name = registry_globals_names[global_id];
			is_global = true;
		}

		if (shared_objects.has(interface)) {
			handle_event(global_id, LocalObjectHandle(nullptr, INVALID_ID), info->opcode, buf, info->size);

			for (Client &c : clients) {
				if (is_global) {
					if (!c.registry_globals_instances.has(global_name)) {
						continue;
					}

					DEBUG_LOG_WAYLAND_SNOOPER(vformat("Broadcasting to all global instances for client %d", c.pid));
					for (uint32_t instance_id : c.registry_globals_instances[global_name]) {
						DEBUG_LOG_WAYLAND_SNOOPER(vformat("Global instance l0x%x", instance_id));
						if (c.socket < 0) {
							continue;
						}

						if (!c.local_ids.has(global_id)) {
							DEBUG_LOG_WAYLAND_SNOOPER("!!!!!!!!!!! Instance missing?");
							continue;
						}

						LocalObjectHandle local_obj = LocalObjectHandle(&c, instance_id);
						if (handle_event(global_id, local_obj, info->opcode, buf, info->size)) {
							DEBUG_LOG_WAYLAND_SNOOPER("Custom handler success.");
							continue;
						}

						DEBUG_LOG_WAYLAND_SNOOPER("Falling back to generic handler.");

						// Making a working copy so that `handle_generic_msg` does not get confused.
						// TODO: Investigate a better way, I think.
						uint32_t *copy = (uint32_t *)malloc(info->size);
						memcpy((char *)copy, (char *)buf, info->size);

						copy[0] = instance_id;

						if (handle_generic_msg(&c, interface, message, info, copy, instance_id)) {
							send_raw_message(c.socket, { { copy, info->size } }, sent_fds);
						}

						free(copy);
					}
				} else {
					if (c.socket < 0) {
						continue;
					}

					if (!c.local_ids.has(global_id)) {
						DEBUG_LOG_WAYLAND_SNOOPER("!!!!!!!!!!! Instance missing?");
						continue;
					}

					LocalObjectHandle local_obj = LocalObjectHandle(&c, c.get_local_id(global_id));
					if (handle_event(global_id, local_obj, info->opcode, buf, info->size)) {
						DEBUG_LOG_WAYLAND_SNOOPER("Custom handler success.");
						continue;
					}

					DEBUG_LOG_WAYLAND_SNOOPER("Falling back to generic handler.");

					// Making a working copy so that `handle_generic_msg` does not get confused.
					// TODO: Investigate a better way, I think.
					uint32_t *copy = (uint32_t *)malloc(info->size);
					memcpy((char *)copy, (char *)buf, info->size);

					if (handle_generic_msg(&c, interface, message, info, copy)) {
						send_raw_message(c.socket, { { copy, info->size } }, sent_fds);
					}

					free(copy);
				}
			}
		} else {
			LocalObjectHandle local_obj = LocalObjectHandle(client, client ? client->get_local_id(global_id) : INVALID_ID);
			if (handle_event(global_id, local_obj, info->opcode, buf, info->size)) {
				return false;
			}

			if (client) {
				uint32_t local_id = client->get_local_id(global_id);
				ERR_FAIL_COND_V(local_id == INVALID_ID, false);

				DEBUG_LOG_WAYLAND_SNOOPER(vformat("%s::%s(%s) g0x%x -> l0x%x", interface->name, message->name, message->signature, global_id, local_id));
				buf[0] = local_id;

				if (handle_generic_msg(client, interface, message, info, buf)) {
					send_raw_message(client->socket, { { buf, info->size } }, sent_fds);
				}
			}
		}
	}

	for (int fd : sent_fds) {
		DEBUG_LOG_WAYLAND_SNOOPER(vformat("Closing fd %d.", fd));
		close(fd);
	}

	return false;
}

bool WaylandEmbedderProxy::handle_sock(int p_fd, int p_id) {
	ERR_FAIL_COND_V(p_fd < 0, false);

	struct msg_info info = {};

	{
		struct msghdr head_msg = {};
		uint32_t header[2];
		struct iovec vec = { header, sizeof header };

		head_msg.msg_iov = &vec;
		head_msg.msg_iovlen = 1;

		ssize_t head_rec = recvmsg(p_fd, &head_msg, MSG_PEEK);

		ERR_FAIL_COND_V_MSG(head_rec == -1, false, vformat("Can't read message header: %s", strerror(errno)));
		ERR_FAIL_COND_V(((size_t)head_rec) != vec.iov_len, false);

		// Header is two 32-bit words: first is ID, second has size in most significant
		// half and opcode in the other half.
		info.raw_id = header[0];
		info.size = header[1] >> 16;
		info.opcode = header[1] & 0xFFFF;
		info.direction = p_id >= 0 ? ProxyDirection::COMPOSITOR : ProxyDirection::CLIENT;
	}

	if (msg_buf.size() < info.words()) {
		msg_buf.resize(info.words());
	}

	ERR_FAIL_COND_V_MSG(info.size % WORD_SIZE != 0, false, "Invalid message length.");

	struct msghdr full_msg = {};
	struct iovec vec = { msg_buf.ptr(), info.size };
	{
		full_msg.msg_iov = &vec;
		full_msg.msg_iovlen = 1;
		full_msg.msg_control = ancillary_buf.ptr();
		full_msg.msg_controllen = ancillary_buf.size();

		ssize_t full_rec = recvmsg(p_fd, &full_msg, 0);

		ERR_FAIL_COND_V_MSG(full_rec == -1, false, vformat("Can't read full message: %s", strerror(errno)));
		ERR_FAIL_COND_V_MSG(((size_t)full_rec) != info.size, false, "Invalid message length.");

		DEBUG_LOG_WAYLAND_SNOOPER(" === START PACKET === ");

#ifdef WAYLAND_SNOOPER_DEBUG_LOGS_ENABLED
		printf("[PROXY] Received bytes: ");
		for (ssize_t i = 0; i < full_rec; ++i) {
			printf("%.2x", ((const uint8_t *)msg_buf.ptr())[i]);
		}
		printf("\n");
#endif
	}

	String dir_str = info.direction == ProxyDirection::COMPOSITOR ? "compositor" : "client";
	DEBUG_LOG_WAYLAND_SNOOPER(vformat("dir: %s, id: 0x%x, bytes: %d, opcode: %d", dir_str, info.raw_id, info.size, info.opcode));

	if (full_msg.msg_controllen > 0) {
		struct cmsghdr *cmsg = CMSG_FIRSTHDR(&full_msg);
		while (cmsg) {
			// TODO: Sanity-check message fields.
			size_t data_len = cmsg->cmsg_len - sizeof *cmsg;

			if (cmsg->cmsg_type == SCM_RIGHTS) {
				// NOTE: Linux docs say that we can't just cast data to pointer type because
				// of alignment concerns. So we have to memcpy into a new buffer.
				int *cmsg_fds = (int *)malloc(data_len);
				memcpy(cmsg_fds, CMSG_DATA(cmsg), data_len);

				size_t cmsg_fds_count = data_len / sizeof *cmsg_fds;
				for (size_t i = 0; i < cmsg_fds_count; ++i) {
					int fd = cmsg_fds[i];

					if (info.direction == ProxyDirection::COMPOSITOR) {
						clients[p_id].fds.push_back(fd);
					} else {
						compositor_fds.push_back(fd);
					}
				}

#ifdef WAYLAND_SNOOPER_DEBUG_LOGS_ENABLED
				printf("[PROXY] Received %ld file descriptors: ", cmsg_fds_count);
				for (size_t i = 0; i < cmsg_fds_count; ++i) {
					printf("%d ", cmsg_fds[i]);
				}
				printf("\n");
#endif

				free(cmsg_fds);
			}

			cmsg = CMSG_NXTHDR(&full_msg, cmsg);
		}
	}
	full_msg.msg_control = nullptr;
	full_msg.msg_controllen = 0;

	int fds_requested = 0;

	int client_id = p_id;
	Client *client = nullptr;

	if (client_id < 0) {
		// A negative id means that it's coming from the compositor; Let's figure out
		// the recipient of the message.
		for (size_t i = 0; i < clients.size(); ++i) {
			if (clients[i].local_ids.has(info.raw_id)) {
				client_id = i;
				break;
			}
		}
	}

	if (client_id >= 0) {
		client = &clients[client_id];
	}

	if (client_id >= 0) {
		DEBUG_LOG_WAYLAND_SNOOPER(vformat("Client: %d.", client_id));
	} else {
		DEBUG_LOG_WAYLAND_SNOOPER("No client found to forward to.");
	}

	handle_msg_info(client, &info, msg_buf.ptr(), &fds_requested);

	DEBUG_LOG_WAYLAND_SNOOPER(" === END PACKET === ");

	return true;
}

void WaylandEmbedderProxy::_thread_loop(void *p_data) {
	ERR_FAIL_NULL(p_data);
	WaylandEmbedderProxy *proxy = (WaylandEmbedderProxy *)p_data;

	DEBUG_LOG_WAYLAND_SNOOPER("Proxy thread started");

	while (true) {
		proxy->poll_sockets();
	}
}

Error WaylandEmbedderProxy::init() {
	objects.resize(SNOOP_ID_MAX);
	server_objects.resize(SNOOP_ID_MAX);

	ancillary_buf.resize(SNOOP_ANCILLARY_SIZE);

	// FIXME: Resize dynamically.
	clients.resize(SNOOP_MAX_CLIENTS);

	proxy_socket = socket(AF_UNIX, SOCK_STREAM, 0);

	struct sockaddr_un addr = {};
	addr.sun_family = AF_UNIX;

	String runtime_dir_path = OS::get_singleton()->get_environment("XDG_RUNTIME_DIR");
	ERR_FAIL_COND_V_MSG(runtime_dir_path.is_empty(), ERR_DOES_NOT_EXIST, "XDG_RUNTIME_DIR is not set or empty.");

	runtime_dir = DirAccess::create_for_path(runtime_dir_path);
	ERR_FAIL_COND_V(!runtime_dir.is_valid(), ERR_BUG);
	ERR_FAIL_COND_V_MSG(!runtime_dir->is_writable(runtime_dir_path), ERR_FILE_CANT_WRITE, "XDG_RUNTIME_DIR points to an invalid directory.");

	int socket_id = 0;
	while (socket_path.is_empty()) {
		String test_socket_path = runtime_dir_path + "/godot-wayland-" + itos(socket_id);
		String test_socket_lock_path = test_socket_path + ".lock";

		print_verbose(vformat("Trying to get socket %s", test_socket_path));
		print_verbose(vformat("Opening lock %s", test_socket_lock_path));
		int test_lock_fd = open(test_socket_lock_path.utf8().get_data(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);

		if (flock(test_lock_fd, LOCK_EX | LOCK_NB) == -1) {
			print_verbose(vformat("Can't lock %s", test_socket_lock_path));
			close(test_lock_fd);
			++socket_id;
			continue;
		} else {
			lock_fd = test_lock_fd;
			socket_path = test_socket_path;
			socket_lock_path = test_socket_lock_path;

			break;
		}
	}

	DirAccess::remove_absolute(socket_path);
	strncpy(addr.sun_path, socket_path.utf8().get_data(), sizeof(addr.sun_path) - 1);

	if (bind(proxy_socket, (struct sockaddr *)&addr, sizeof(addr)) == -1) {
		ERR_FAIL_V_MSG(ERR_CANT_CREATE, "Can't bind embedding socket.");
	}

	if (listen(proxy_socket, 1) == -1) {
		ERR_FAIL_V_MSG(ERR_CANT_OPEN, "Can't listen embedding socket.");
	}

	struct wl_display *display = wl_display_connect(nullptr);
	ERR_FAIL_NULL_V(display, ERR_CANT_OPEN);
	compositor_socket = wl_display_get_fd(display);

	DEBUG_LOG_WAYLAND_SNOOPER(vformat("proxy %d compositor %d", proxy_socket, compositor_socket));

	pollfds.push_back({ proxy_socket, POLLIN, 0 });
	pollfds.push_back({ compositor_socket, POLLIN, 0 });

	RegistryGlobalInfo control_global_info = {};
	control_global_info.interface = &godot_embedding_compositor_interface;
	control_global_info.version = godot_embedding_compositor_interface.version;

	registry_globals.push_back(control_global_info);
	godot_embedding_compositor_name = registry_globals.size() - 1;

	get_object(DISPLAY_ID)->interface = &wl_display_interface;
	get_object(DISPLAY_ID)->version = 1;
	shared_objects[&wl_display_interface] = DISPLAY_ID;

	get_object(REGISTRY_ID)->interface = &wl_registry_interface;
	get_object(REGISTRY_ID)->version = 1;
	shared_objects[&wl_registry_interface] = REGISTRY_ID;

	// wl_display::get_registry(n)
	send_wayland_message(compositor_socket, DISPLAY_ID, 1, { REGISTRY_ID });

	sync();

	proxy_thread.start(_thread_loop, this);

	return OK;
}

void WaylandEmbedderProxy::handle_fd(int p_fd, int p_revents) {
	if (p_fd == proxy_socket && p_revents & POLLIN) {
		// Client init.
		int new_fd = accept(proxy_socket, nullptr, nullptr);
		ERR_FAIL_COND_MSG(new_fd == -1, "can't accept client");

		struct ucred cred = {};
		socklen_t cred_size = sizeof cred;
		getsockopt(new_fd, SOL_SOCKET, SO_PEERCRED, &cred, &cred_size);

		Client *client = nullptr;
		int client_id = -1;
		for (size_t i = 0; i < clients.size(); ++i) {
			Client &test_client = clients[i];
			if (test_client.socket < 0) {
				client = &test_client;
				client_id = i;
				break;
			}
		}

		if (client == nullptr) {
			WARN_PRINT(vformat("[PROXY] Out of clients. Rejecting client %d.", client_id));
			close(new_fd);
			return;
		}

		pollfds.push_back({ new_fd, POLLIN, 0 });

		client->snooper = this;

		client->pid = cred.pid;
		client->socket = new_fd;

		client->global_ids[DISPLAY_ID] = DISPLAY_ID;
		client->local_ids[DISPLAY_ID] = DISPLAY_ID;

		client->server_id_map.resize(SNOOP_ID_MAX);
		memset(client->server_id_map.ptr(), 0, client->server_id_map.size());

		client->get_object(DISPLAY_ID)->interface = &wl_display_interface;

		Client &main_client = clients[0];
		if (client_id != 0 && main_client.registry_globals_instances.has(godot_embedding_compositor_name)) {
			uint32_t new_local_id = main_client.allocate_server_id();

			client->embedded_client_id = new_local_id;

			for (uint32_t local_id : main_client.registry_globals_instances[godot_embedding_compositor_name]) {
				EmbeddedClientData *eclient_data = memnew(EmbeddedClientData);
				eclient_data->client = client;

				main_client.new_fake_object(new_local_id, &godot_embedded_client_interface, 1, eclient_data);

				// godot_embedding_compositor::client(nu)
				send_wayland_message(main_client.socket, local_id, 0, { new_local_id, (uint32_t)cred.pid });
			}
		}

		DEBUG_LOG_WAYLAND_SNOOPER(vformat("New client #%d (pid %d) initialized.", client_id, cred.pid));
		return;
	}

	if (p_fd == compositor_socket && p_revents & POLLIN) {
		handle_sock(compositor_socket, -1);
		return;
	}

	if (clients[0].socket >= 0 && p_fd == clients[0].socket) {
		// TODO: Handle main client hangup.

		if (p_revents & POLLIN) {
			handle_sock(clients[0].socket, 0);
			return;
		}
	}

	for (size_t i = 1; i < clients.size(); ++i) {
		Client &client = clients[i];

		if (client.socket < 0 || p_fd != client.socket) {
			continue;
		}

		if (p_revents & POLLIN) {
			if (!handle_sock(client.socket, i)) {
				DEBUG_LOG_WAYLAND_SNOOPER("disconnecting");
				client_disconnect(i);
			}
			return;
		} else if (p_revents & (POLLHUP | POLLERR | POLLNVAL)) {
			if (p_revents & POLLHUP) {
				DEBUG_LOG_WAYLAND_SNOOPER(vformat("embedded client %d hangup.", i));
			}
			if (p_revents & POLLERR) {
				DEBUG_LOG_WAYLAND_SNOOPER(vformat("embedded client %d error.", i));
			}
			if (p_revents & POLLNVAL) {
				DEBUG_LOG_WAYLAND_SNOOPER(vformat("embedded client %d invalid fd.", i));
			}

			client_disconnect(i);

			return;
		}
	}
}
