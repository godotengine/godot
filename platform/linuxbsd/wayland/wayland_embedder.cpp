/**************************************************************************/
/*  wayland_embedder.cpp                                                  */
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

#include "wayland_embedder.h"

#ifdef WAYLAND_ENABLED

#ifdef TOOLS_ENABLED

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

#define WAYLAND_EMBED_ID_MAX 1000

//#define WAYLAND_EMBED_DEBUG_LOGS_ENABLED
#ifdef WAYLAND_EMBED_DEBUG_LOGS_ENABLED

// Gotta flush as we're doing this mess from a thread without any
// synchronization. It's awful, I know, but the `print_*` utilities hang for
// some reason during editor startup and I need some quick and dirty debugging.
#define DEBUG_LOG_WAYLAND_EMBED(...)                               \
	if (1) {                                                       \
		printf("[PROXY] %s\n", vformat(__VA_ARGS__).utf8().ptr()); \
		fflush(stdout);                                            \
	} else                                                         \
		((void)0)

#else
#define DEBUG_LOG_WAYLAND_EMBED(...)
#endif

// Wayland messages are structured with 32-bit words.
#define WL_WORD_SIZE (sizeof(uint32_t))

// Event opcodes. Request opcodes are defined in the generated client headers.
// We could generate server headers but they would clash (without modifications)
// and we use just a few constants anyways.

#define WL_DISPLAY_ERROR 0
#define WL_DISPLAY_DELETE_ID 1

#define WL_REGISTRY_GLOBAL 0
#define WL_REGISTRY_GLOBAL_REMOVE 1

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

#define XDG_POPUP_CONFIGURE 0

size_t WaylandEmbedder::wl_array_word_offset(uint32_t p_size) {
	uint32_t pad = (WL_WORD_SIZE - (p_size % WL_WORD_SIZE)) % WL_WORD_SIZE;
	return (p_size + pad) / WL_WORD_SIZE;
}

const struct wl_interface *WaylandEmbedder::wl_interface_from_string(const char *name, size_t size) {
	for (size_t i = 0; i < (sizeof interfaces / sizeof *interfaces); ++i) {
		if (strncmp(name, interfaces[i]->name, size) == 0) {
			return interfaces[i];
		}
	}

	return nullptr;
}

int WaylandEmbedder::wl_interface_get_destructor_opcode(const struct wl_interface *p_iface, uint32_t version) {
	ERR_FAIL_NULL_V(p_iface, -1);

	// FIXME: Figure out how to extract the destructor from the XML files. This
	// value is not currently exposed by wayland-scanner.
	for (int i = 0; i < p_iface->method_count; ++i) {
		const struct wl_message &m = p_iface->methods[i];
		uint32_t destructor_version = String::to_int(m.signature);
		if (destructor_version <= version && (strcmp(m.name, "destroy") == 0 || strcmp(m.name, "release") == 0)) {
			return i;
		}
	}

	return -1;
}

struct WaylandEmbedder::WaylandObject *WaylandEmbedder::get_object(uint32_t p_global_id) {
	if (p_global_id == 0) {
		return nullptr;
	}

	// Server-allocated stuff starts at 0xff000000.
	bool is_server = p_global_id & 0xff000000;
	if (is_server) {
		p_global_id &= ~(0xff000000);
	}

#ifdef DEV_ENABLED
	if (p_global_id >= WAYLAND_EMBED_ID_MAX) {
		// Oh no. Time for debug info!

#ifdef WAYLAND_EMBED_DEBUG_LOGS_ENABLED
		for (uint32_t id = 1; id < objects.reserved_size(); ++id) {
			WaylandObject &object = objects[id];
			DEBUG_LOG_WAYLAND_EMBED(vformat(" - g0x%x (#%d): %s version %d, data 0x%x", id, id, object.interface->name, object.version, (uintptr_t)object.data));
		}
#endif // WAYLAND_EMBED_DEBUG_LOGS_ENABLED

		CRASH_NOW_MSG(vformat("Tried to access ID bigger than debug cap (%d > %d).", p_global_id, WAYLAND_EMBED_ID_MAX));
	}
#endif // DEV_ENABLED

	if (is_server) {
		if (server_objects.size() <= p_global_id) {
			return nullptr;
		}

		return &server_objects[p_global_id];
	} else {
		if (objects.reserved_size() <= p_global_id) {
			return nullptr;
		}

		return &objects[p_global_id];
	}
}

Error WaylandEmbedder::delete_object(uint32_t p_global_id) {
	WaylandObject *object = get_object(p_global_id);
	ERR_FAIL_NULL_V(object, ERR_DOES_NOT_EXIST);

	if (object->shared) {
		ERR_FAIL_V_MSG(FAILED, vformat("Tried to delete shared object g0x%x.", p_global_id));
	}

	DEBUG_LOG_WAYLAND_EMBED(vformat("Deleting object %s g0x%x", object->interface ? object->interface->name : "UNKNOWN", p_global_id));

	if (object->data) {
		memdelete(object->data);
		object->data = nullptr;
	}

	bool is_server = p_global_id & 0xff000000;
	if (is_server) {
		server_objects[p_global_id & ~(0xff000000)] = WaylandObject();
	} else {
		objects.free(p_global_id);
	}

	registry_globals_names.erase(p_global_id);

	return OK;
}

uint32_t WaylandEmbedder::Client::allocate_server_id() {
	uint32_t new_id = INVALID_ID;

	if (free_server_ids.size() > 0) {
		int new_size = free_server_ids.size() - 1;
		new_id = free_server_ids[new_size] | 0xff000000;
		free_server_ids.resize_uninitialized(new_size);
	} else {
		new_id = allocated_server_ids | 0xff000000;

		++allocated_server_ids;
#ifdef DEV_ENABLED
		CRASH_COND_MSG(allocated_server_ids > WAYLAND_EMBED_ID_MAX, "Max server ID reached. This might indicate a leak.");
#endif // DEV_ENABLED
	}

	DEBUG_LOG_WAYLAND_EMBED(vformat("Allocated server-side id 0x%x.", new_id));

	return new_id;
}

struct WaylandEmbedder::WaylandObject *WaylandEmbedder::Client::get_object(uint32_t p_local_id) {
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

	ERR_FAIL_NULL_V(embedder, nullptr);
	return embedder->get_object(get_global_id(p_local_id));
}

Error WaylandEmbedder::Client::bind_global_id(uint32_t p_global_id, uint32_t p_local_id) {
	ERR_FAIL_COND_V(local_ids.has(p_global_id), ERR_ALREADY_EXISTS);
	ERR_FAIL_COND_V(global_ids.has(p_local_id), ERR_ALREADY_EXISTS);

	GlobalIdInfo gid_info;
	gid_info.id = p_global_id;
	DEBUG_LOG_WAYLAND_EMBED(vformat("Pushing g0x%x in the global id history", p_global_id));
	gid_info.history_elem = global_id_history.push_back(p_global_id);
	global_ids[p_local_id] = gid_info;

	local_ids[p_global_id] = p_local_id;

	return OK;
}

Error WaylandEmbedder::Client::delete_object(uint32_t p_local_id) {
	if (fake_objects.has(p_local_id)) {
#ifdef WAYLAND_EMBED_DEBUG_LOGS_ENABLED
		WaylandObject *object = &fake_objects[p_local_id];
		DEBUG_LOG_WAYLAND_EMBED(vformat("Deleting fake object %s l0x%x", object->interface ? object->interface->name : "UNKNOWN", p_local_id));
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
	GlobalIdInfo gid_info = global_ids[p_local_id];
	uint32_t global_id = gid_info.id;

	DEBUG_LOG_WAYLAND_EMBED(vformat("Erasing g0x%x from the global id history", global_id));
	global_id_history.erase(gid_info.history_elem);

	if (global_instances.has(p_local_id)) {
#ifdef WAYLAND_EMBED_DEBUG_LOGS_ENABLED
		WaylandObject *object = &global_instances[p_local_id];
		DEBUG_LOG_WAYLAND_EMBED(vformat("Deleting global instance %s l0x%x", object->interface ? object->interface->name : "UNKNOWN", p_local_id));
#endif

		// wl_display::delete_id
		send_wayland_message(socket, DISPLAY_ID, 1, { p_local_id });

		// We don't want to delete the global object tied to this instance, so we'll only get rid of the local stuff.
		global_instances.erase(p_local_id);
		global_ids.erase(p_local_id);

		if (global_id != INVALID_ID) {
			local_ids.erase(global_id);
		}

		// We're done here.
		return OK;
	}

	if (wl_registry_instances.has(p_local_id)) {
		wl_registry_instances.erase(p_local_id);
	}

	WaylandObject *object = embedder->get_object(global_id);
	ERR_FAIL_NULL_V(object, ERR_DOES_NOT_EXIST);

	ERR_FAIL_COND_V_MSG(object->shared, ERR_INVALID_PARAMETER, vformat("Tried to delete shared object g0x%x.", global_id));

	global_ids.erase(p_local_id);
	local_ids.erase(global_id);

	if (p_local_id & 0xff000000) {
		free_server_ids.push_back(p_local_id & ~(0xff000000));
	}

	uint32_t *global_name = embedder->registry_globals_names.getptr(global_id);
	if (global_name) {
		{
			RegistryGlobalInfo &info = embedder->registry_globals[*global_name];
			ERR_FAIL_COND_V_MSG(info.instance_counter == 0, ERR_BUG, "Instance counter inconsistency.");
			--info.instance_counter;

			if (info.destroyed && info.instance_counter == 0) {
				embedder->registry_globals.erase(*global_name);
			}
		}

		registry_globals_instances[*global_name].erase(p_local_id);
	}

	return embedder->delete_object(global_id);
}

// Returns INVALID_ID if the creation fails. In that case, the user can assume
// that the client got kicked out.
uint32_t WaylandEmbedder::Client::new_object(uint32_t p_local_id, const struct wl_interface *p_interface, int p_version, WaylandObjectData *p_data) {
	if (embedder == nullptr) {
		socket_error(socket, p_local_id, WL_DISPLAY_ERROR_IMPLEMENTATION, "No embedder set.");
		ERR_FAIL_V(INVALID_ID);
	}

	if (get_object(p_local_id) != nullptr) {
		socket_error(socket, p_local_id, WL_DISPLAY_ERROR_IMPLEMENTATION, vformat("Tried to create %s l0x%x but it already exists as %s", p_interface->name, p_local_id, get_object(p_local_id)->interface->name));
		ERR_FAIL_V(INVALID_ID);
	}

	uint32_t new_global_id = embedder->new_object(p_interface, p_version, p_data);

	bind_global_id(new_global_id, p_local_id);

	return new_global_id;
}

uint32_t WaylandEmbedder::Client::new_server_object(uint32_t p_global_id, const struct wl_interface *p_interface, int p_version, WaylandObjectData *p_data) {
	if (embedder == nullptr) {
		socket_error(socket, get_local_id(p_global_id), WL_DISPLAY_ERROR_IMPLEMENTATION, "No embedder set.");
		ERR_FAIL_V(INVALID_ID);
	}

	uint32_t new_local_id = allocate_server_id();

	embedder->new_server_object(p_global_id, p_interface, p_version, p_data);

	bind_global_id(p_global_id, new_local_id);

	return new_local_id;
}

WaylandEmbedder::WaylandObject *WaylandEmbedder::Client::new_fake_object(uint32_t p_local_id, const struct wl_interface *p_interface, int p_version, WaylandObjectData *p_data) {
	if (embedder == nullptr) {
		socket_error(socket, p_local_id, WL_DISPLAY_ERROR_IMPLEMENTATION, "No embedder set.");
		ERR_FAIL_V(nullptr);
	}

	if (get_object(p_local_id) != nullptr) {
		socket_error(socket, p_local_id, WL_DISPLAY_ERROR_IMPLEMENTATION, vformat("Object l0x%x already exists", p_local_id));
		ERR_FAIL_V(nullptr);
	}

	WaylandObject &new_object = fake_objects[p_local_id];
	new_object.interface = p_interface;
	new_object.version = p_version;
	new_object.data = p_data;

	return &new_object;
}

WaylandEmbedder::WaylandObject *WaylandEmbedder::Client::new_global_instance(uint32_t p_local_id, uint32_t p_global_id, const struct wl_interface *p_interface, int p_version, WaylandObjectData *p_data) {
	if (embedder == nullptr) {
		socket_error(socket, p_local_id, WL_DISPLAY_ERROR_IMPLEMENTATION, "No embedder set.");
		ERR_FAIL_V(nullptr);
	}

	if (get_object(p_local_id) != nullptr) {
		socket_error(socket, p_local_id, WL_DISPLAY_ERROR_IMPLEMENTATION, vformat("Object l0x%x already exists", p_local_id));
		ERR_FAIL_V(nullptr);
	}

	WaylandObject &new_object = global_instances[p_local_id];
	new_object.interface = p_interface;
	new_object.version = p_version;
	new_object.data = p_data;

	// FIXME: Track each instance properly. Global instances (the compatibility
	// mechanism) are particular as they're the only case where a global ID might
	// map to multiple local objects. In that case we need to mirror each event
	// which passes a registry object as an argument for each instance.
	GlobalIdInfo gid_info;
	gid_info.id = p_global_id;
	gid_info.history_elem = global_id_history.push_back(p_global_id);
	global_ids[p_local_id] = gid_info;

	// NOTE: Normally, for each client, there's a single local object per global
	// object, but global instances break this expectation. This is technically
	// wrong but should work fine, as we have special logic whenever needed.
	//
	// TODO: it might be nice to enforce that this table is never looked up for
	// global instances or even just log attempts.
	local_ids[p_global_id] = p_local_id;

	return &new_object;
}

Error WaylandEmbedder::Client::send_wl_drm_state(uint32_t p_id, WaylandDrmGlobalData *p_state) {
	ERR_FAIL_NULL_V(p_state, ERR_INVALID_PARAMETER);

	if (p_state->device.is_empty()) {
		// Not yet initialized.
		return OK;
	}

	LocalVector<union wl_argument> args;
	args.push_back(wl_arg_string(p_state->device.utf8().get_data()));
	send_wayland_event(socket, p_id, wl_drm_interface, WL_DRM_DEVICE, args);

	for (uint32_t format : p_state->formats) {
		Error err = send_wayland_message(socket, p_id, WL_DRM_FORMAT, { format });
		ERR_FAIL_COND_V(err != OK, err);
	}

	if (p_state->authenticated) {
		Error err = send_wayland_message(socket, p_id, WL_DRM_AUTHENTICATED, {});
		ERR_FAIL_COND_V(err != OK, err);
	}

	Error err = send_wayland_message(socket, p_id, WL_DRM_CAPABILITIES, { p_state->capabilities });
	ERR_FAIL_COND_V(err != OK, err);

	return OK;
}

void WaylandEmbedder::cleanup_socket(int p_socket) {
	DEBUG_LOG_WAYLAND_EMBED(vformat("Cleaning up socket %d.", p_socket));

	close(p_socket);

	for (size_t i = 0; i < pollfds.size(); ++i) {
		if (pollfds[i].fd == p_socket) {
			pollfds.remove_at_unordered(i);
			break;
		}
	}

	ERR_FAIL_COND(!clients.has(p_socket));

	Client &client = clients[p_socket];

	for (KeyValue<uint32_t, WaylandObject> &pair : client.fake_objects) {
		WaylandObject &object = pair.value;

		if (object.interface == &xdg_toplevel_interface) {
			XdgToplevelData *data = (XdgToplevelData *)object.data;
			CRASH_COND(data == nullptr);

			if (data->wl_subsurface_id != INVALID_ID) {
				// wl_subsurface::destroy() - xdg_toplevels are mapped to subsurfaces.
				send_wayland_message(compositor_socket, data->wl_subsurface_id, 0, {});
			}

			if (!data->xdg_surface_handle.get()) {
				continue;
			}

			XdgSurfaceData *xdg_surf_data = (XdgSurfaceData *)data->xdg_surface_handle.get()->data;
			if (xdg_surf_data == nullptr) {
				continue;
			}

			if (!data->parent_handle.get()) {
				continue;
			}

			XdgToplevelData *parent_data = (XdgToplevelData *)data->parent_handle.get()->data;
			if (parent_data == nullptr) {
				continue;
			}

			if (!parent_data->xdg_surface_handle.get()) {
				continue;
			}

			XdgSurfaceData *parent_xdg_surf_data = (XdgSurfaceData *)parent_data->xdg_surface_handle.get()->data;
			if (parent_xdg_surf_data == nullptr) {
				continue;
			}

			for (uint32_t wl_seat_name : wl_seat_names) {
				WaylandSeatGlobalData *global_seat_data = (WaylandSeatGlobalData *)registry_globals[wl_seat_name].data;
				if (global_seat_data == nullptr) {
					continue;
				}

				if (global_seat_data->focused_surface_id == xdg_surf_data->wl_surface_id) {
					seat_name_leave_surface(wl_seat_name, xdg_surf_data->wl_surface_id);
					seat_name_enter_surface(wl_seat_name, parent_xdg_surf_data->wl_surface_id);
				}
			}
		}
	}

	for (List<uint32_t>::Element *E = client.global_id_history.back(); E;) {
		uint32_t global_id = E->get();
		E = E->prev();

		WaylandObject *object = get_object(global_id);
		if (object == nullptr) {
			DEBUG_LOG_WAYLAND_EMBED(vformat("Skipping deletability check of object g0x%x as it's null.", global_id));
			continue;
		}

		if (object->interface == nullptr) {
			DEBUG_LOG_WAYLAND_EMBED(vformat("Skipping deletability check of object g0x%x as it's invalid.", global_id));
			continue;
		}

		DEBUG_LOG_WAYLAND_EMBED(vformat("Checking deletability of %s#g0x%x version %s", object->interface->name, global_id, object->version));

		if (object->shared) {
			DEBUG_LOG_WAYLAND_EMBED("Shared, skipping.");
			continue;
		}

		if (object->interface == &wl_callback_interface) {
			// Those things self-destruct.
			DEBUG_LOG_WAYLAND_EMBED("wl_callback self destructs.");
			continue;
		}

		if (object->destroyed) {
			DEBUG_LOG_WAYLAND_EMBED("Already destroyed, skipping.");
			continue;
		}

		int destructor = wl_interface_get_destructor_opcode(object->interface, object->version);
		if (destructor >= 0) {
			DEBUG_LOG_WAYLAND_EMBED(vformat("Destroying %s#g0x%x", object->interface->name, global_id));

			if (object->interface == &wl_surface_interface) {
				for (uint32_t wl_seat_name : wl_seat_names) {
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
			}

			send_wayland_message(compositor_socket, global_id, destructor, {});
			object->destroyed = true;

			if (global_id & 0xff000000) {
				delete_object(global_id);
				object = nullptr;
			}
		}

		if (object && !object->destroyed) {
			ERR_PRINT(vformat("Unreferenced object %s g0x%x (leak!)", object->interface->name, global_id));
		}
	}

	uint32_t eclient_id = client.embedded_client_id;

	clients.erase(client.socket);

	WaylandObject *eclient = main_client->get_object(eclient_id);

	if (eclient) {
		EmbeddedClientData *eclient_data = (EmbeddedClientData *)eclient->data;
		ERR_FAIL_NULL(eclient_data);

		if (!eclient_data->disconnected) {
			// godot_embedded_client::disconnected
			send_wayland_message(main_client->socket, eclient_id, 0, {});
		}

		eclient_data->disconnected = true;
	}
}

void WaylandEmbedder::socket_error(int p_socket, uint32_t p_object_id, uint32_t p_code, const String &p_message) {
	const char *err_name = "unknown";
	switch (p_code) {
		case WL_DISPLAY_ERROR_INVALID_OBJECT: {
			err_name = "invalid_object";
		} break;

		case WL_DISPLAY_ERROR_INVALID_METHOD: {
			err_name = "invalid_method";
		} break;

		case WL_DISPLAY_ERROR_NO_MEMORY: {
			err_name = "no_memory";
		} break;

		case WL_DISPLAY_ERROR_IMPLEMENTATION: {
			err_name = "implementation";
		} break;
	}

	ERR_PRINT(vformat("Socket %d %s error: %s", p_socket, err_name, p_message));

	LocalVector<union wl_argument> args;
	args.push_back(wl_arg_object(p_object_id));
	args.push_back(wl_arg_uint(p_code));
	args.push_back(wl_arg_string(vformat("[Godot Embedder] %s", p_message).utf8().get_data()));

	send_wayland_event(p_socket, DISPLAY_ID, wl_display_interface, WL_DISPLAY_ERROR, args);

	// So, here's the deal: from some extensive research I did, there are
	// absolutely zero safeguards for ensuring that the error message ends to the
	// client. It's absolutely tiny and takes _nothing_ to get there (less than
	// 4µs with a debug build on my machine), but still enough to get truncated in
	// the distance between `send_wayland_event` and `close`.
	//
	// Because of this we're going to give the client some slack: we're going to
	// wait for its socket to close (or whatever) or 1s, whichever happens first.
	//
	// Hopefully it's good enough for <1000 bytes :P
	struct pollfd pollfd = {};
	pollfd.fd = p_socket;

	int ret = poll(&pollfd, 1, 1'000);
	if (ret == 0) {
		ERR_PRINT("Client timeout while disconnecting.");
	}
	if (ret < 0) {
		ERR_PRINT(vformat("Client error while disconnecting: %s", strerror(errno)));
	}

	close(p_socket);
}

void WaylandEmbedder::poll_sockets() {
	if (poll(pollfds.ptr(), pollfds.size(), -1) == -1) {
		CRASH_NOW_MSG(vformat("poll() failed, errno %d.", errno));
	}

	// First handle everything but the listening socket (which is always the first
	// element), so that we can cleanup closed sockets before accidentally reusing
	// them (and breaking everything).
	for (size_t i = 1; i < pollfds.size(); ++i) {
		handle_fd(pollfds[i].fd, pollfds[i].revents);
	}

	handle_fd(pollfds[0].fd, pollfds[0].revents);
}

Error WaylandEmbedder::send_raw_message(int p_socket, std::initializer_list<struct iovec> p_vecs, const LocalVector<int> &p_fds) {
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

#ifdef WAYLAND_EMBED_DEBUG_LOGS_ENABLED
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

	return OK;
}

Error WaylandEmbedder::send_wayland_message(int p_socket, uint32_t p_id, uint32_t p_opcode, const uint32_t *p_args, const size_t p_args_words) {
	ERR_FAIL_COND_V(p_socket < 0, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_id == INVALID_ID, ERR_INVALID_PARAMETER);

	uint32_t args_size = p_args_words * sizeof *p_args;

	// Header is always 8 bytes long.
	uint32_t total_size = 8 + (args_size);

	uint32_t header[2] = { p_id, (total_size << 16) + p_opcode };

	struct iovec vecs[2] = {
		{ header, 8 },
		// According to the sendmsg manual, these buffers should never be written to,
		// so this cast should be safe.
		{ (void *)p_args, args_size },
	};

	struct msghdr msg = {};
	msg.msg_iov = vecs;
	msg.msg_iovlen = std_size(vecs);

#ifdef WAYLAND_EMBED_DEBUG_LOGS_ENABLED
	printf("[PROXY] Sending: ");

	for (struct iovec &vec : vecs) {
		for (size_t i = 0; i < vec.iov_len; ++i) {
			printf("%.2x", ((const uint8_t *)vec.iov_base)[i]);
		}
	}
	printf("\n");
#endif

	if (sendmsg(p_socket, &msg, MSG_NOSIGNAL) < 0) {
		return FAILED;
	}

	return OK;
}

Error WaylandEmbedder::send_wayland_message(ProxyDirection p_direction, int p_socket, uint32_t p_id, const struct wl_interface &p_interface, uint32_t p_opcode, const LocalVector<union wl_argument> &p_args) {
	ERR_FAIL_COND_V(p_direction == ProxyDirection::CLIENT && p_opcode >= (uint32_t)p_interface.event_count, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_direction == ProxyDirection::COMPOSITOR && p_opcode >= (uint32_t)p_interface.method_count, ERR_INVALID_PARAMETER);

	const struct wl_message &msg = p_direction == ProxyDirection::CLIENT ? p_interface.events[p_opcode] : p_interface.methods[p_opcode];

	LocalVector<uint32_t> arg_buf;

	size_t arg_idx = 0;
	for (size_t sig_idx = 0; sig_idx < strlen(msg.signature); ++sig_idx) {
		if (arg_idx >= p_args.size()) {
			String err_msg = vformat("Not enough arguments for r0x%d %s.%s(%s) (only got %d)", p_id, p_interface.name, msg.name, msg.signature, p_args.size());
			ERR_FAIL_COND_V_MSG(arg_idx >= p_args.size(), ERR_INVALID_PARAMETER, err_msg);
		}

		char sym = msg.signature[sig_idx];
		if (sym >= '0' && sym <= '?') {
			// We don't care about version notices and nullability symbols. We can skip
			// those.
			continue;
		}

		const union wl_argument &arg = p_args[arg_idx];

		switch (sym) {
			case 'i': {
				arg_buf.push_back((uint32_t)arg.i);
			} break;

			case 'u': {
				arg_buf.push_back(arg.u);
			} break;

			case 'f': {
				arg_buf.push_back((uint32_t)arg.f);
			} break;

			case 'o': {
				// We're encoding object arguments as uints because I don't think we can
				// reuse the whole opaque struct thing.
				arg_buf.push_back(arg.u);
			} break;

			case 'n': {
				arg_buf.push_back(arg.n);
			} break;

			case 's': {
				const char *str = p_args[arg_idx].s;
				// Wayland requires the string length to include the null terminator.
				uint32_t str_len = strlen(str) + 1;

				arg_buf.push_back(str_len);

				size_t data_begin_idx = arg_buf.size();

				uint32_t str_words = wl_array_word_offset(str_len);

				arg_buf.resize(arg_buf.size() + str_words);
				strcpy((char *)(arg_buf.ptr() + data_begin_idx), str);
			} break;

			case 'a': {
				const wl_array *arr = p_args[arg_idx].a;

				arg_buf.push_back(arr->size);

				size_t data_begin_idx = arg_buf.size();

				uint32_t words = wl_array_word_offset(arr->size);

				arg_buf.resize(arg_buf.size() + words);
				memcpy(arg_buf.ptr() + data_begin_idx, arr->data, arr->size);
			} break;

				// FDs (h) are encoded out-of-band.
		}

		++arg_idx;
	}

	send_wayland_message(p_socket, p_id, p_opcode, arg_buf.ptr(), arg_buf.size());

	return OK;
}

uint32_t WaylandEmbedder::new_object(const struct wl_interface *p_interface, int p_version, WaylandObjectData *p_data) {
	uint32_t new_global_id = allocate_global_id();

	DEBUG_LOG_WAYLAND_EMBED(vformat("New object g0x%x %s", new_global_id, p_interface->name));

	WaylandObject *new_object = get_object(new_global_id);
	new_object->interface = p_interface;
	new_object->version = p_version;
	new_object->data = p_data;

	return new_global_id;
}

WaylandEmbedder::WaylandObject *WaylandEmbedder::new_server_object(uint32_t p_global_id, const struct wl_interface *p_interface, int p_version, WaylandObjectData *p_data) {
	// The max ID will never increment more than one at a time, due to the
	// packed nature of IDs. libwayland already does similar assertions so it
	// just makes sense to double-check to avoid messing memory up or
	// allocating a huge buffer for nothing.
	uint32_t stripped_id = p_global_id & ~(0xff000000);

	ERR_FAIL_COND_V_MSG(stripped_id > server_objects.size(), nullptr, "Invalid new server id requested.");
	ERR_FAIL_COND_V_MSG(get_object(p_global_id) && get_object(p_global_id)->interface, nullptr, vformat("Tried to create %s g0x%x but it already exists as %s.", p_interface->name, p_global_id, get_object(p_global_id)->interface->name));

	if (stripped_id == server_objects.size()) {
		server_objects.resize(server_objects.size() + 1);
	}

	DEBUG_LOG_WAYLAND_EMBED(vformat("New server object %s g0x%x", p_interface->name, p_global_id));

	WaylandObject *new_object = get_object(p_global_id);
	new_object->interface = p_interface;
	new_object->version = p_version;
	new_object->data = p_data;

	return new_object;
}

void WaylandEmbedder::sync() {
	CRASH_COND_MSG(sync_callback_id, "Sync already in progress.");

	sync_callback_id = allocate_global_id();
	get_object(sync_callback_id)->interface = &wl_callback_interface;
	get_object(sync_callback_id)->version = 1;
	send_wayland_message(compositor_socket, DISPLAY_ID, 0, { sync_callback_id });

	DEBUG_LOG_WAYLAND_EMBED("Synchronizing");

	while (true) {
		poll_sockets();

		if (!sync_callback_id) {
			// Obj got deleted - sync is done.
			return;
		}
	}
}

// Returns the gid for the newly bound object, or an existing shared object if
// necessary.
uint32_t WaylandEmbedder::wl_registry_bind(uint32_t p_registry_id, uint32_t p_name, int p_version) {
	RegistryGlobalInfo &info = registry_globals[p_name];

	uint32_t id = INVALID_ID;

	if (wl_interface_get_destructor_opcode(info.interface, p_version) < 0) {
		DEBUG_LOG_WAYLAND_EMBED(vformat("Binding instanced global %s %d", info.interface->name, p_version));

		// Reusable object.
		if (info.reusable_objects.has(p_version) && info.reusable_objects[p_version] != INVALID_ID) {
			DEBUG_LOG_WAYLAND_EMBED("Already bound.");
			return info.reusable_objects[p_version];
		}

		id = new_object(info.interface, p_version);
		ERR_FAIL_COND_V(id == INVALID_ID, INVALID_ID);

		info.reusable_objects[p_version] = id;
		get_object(id)->shared = true;
	} else {
		DEBUG_LOG_WAYLAND_EMBED(vformat("Binding global %s as g0x%x version %d", info.interface->name, id, p_version));
		id = new_object(info.interface, p_version);
	}

	ERR_FAIL_COND_V(id == INVALID_ID, INVALID_ID);

	registry_globals_names[id] = p_name;

	LocalVector<union wl_argument> args;
	args.push_back(wl_arg_uint(info.compositor_name));
	args.push_back(wl_arg_string(info.interface->name));
	args.push_back(wl_arg_int(p_version));
	args.push_back(wl_arg_new_id(id));

	Error err = send_wayland_method(compositor_socket, p_registry_id, wl_registry_interface, WL_REGISTRY_BIND, args);
	ERR_FAIL_COND_V_MSG(err != OK, INVALID_ID, "Error while sending bind request.");

	return id;
}

void WaylandEmbedder::seat_name_enter_surface(uint32_t p_seat_name, uint32_t p_wl_surface_id) {
	WaylandSurfaceData *surf_data = (WaylandSurfaceData *)get_object(p_wl_surface_id)->data;
	CRASH_COND(surf_data == nullptr);

	Client *client = surf_data->client;
	CRASH_COND(client == nullptr);

	if (!client->local_ids.has(p_wl_surface_id)) {
		DEBUG_LOG_WAYLAND_EMBED("Called seat_name_enter_surface with an unknown surface");
		return;
	}

	uint32_t local_surface_id = client->get_local_id(p_wl_surface_id);

	DEBUG_LOG_WAYLAND_EMBED(vformat("KB: Entering surface g0x%x", p_wl_surface_id));

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

	if (client->socket != main_client->socket) {
		// godot_embedded_client::window_focus_in
		send_wayland_message(main_client->socket, client->embedded_client_id, 2, {});
	}
}

void WaylandEmbedder::seat_name_leave_surface(uint32_t p_seat_name, uint32_t p_wl_surface_id) {
	WaylandSurfaceData *surf_data = (WaylandSurfaceData *)get_object(p_wl_surface_id)->data;
	CRASH_COND(surf_data == nullptr);

	Client *client = surf_data->client;
	CRASH_COND(client == nullptr);

	if (!client->local_ids.has(p_wl_surface_id)) {
		DEBUG_LOG_WAYLAND_EMBED("Called seat_name_leave_surface with an unknown surface!");
		return;
	}

	uint32_t local_surface_id = client->get_local_id(p_wl_surface_id);

	DEBUG_LOG_WAYLAND_EMBED(vformat("KB: Leaving surface g0x%x", p_wl_surface_id));

	for (uint32_t local_seat_id : client->registry_globals_instances[p_seat_name]) {
		WaylandSeatInstanceData *seat_data = (WaylandSeatInstanceData *)client->get_object(local_seat_id)->data;
		CRASH_COND(seat_data == nullptr);

		uint32_t local_keyboard_id = client->get_local_id(seat_data->wl_keyboard_id);

		if (local_keyboard_id != INVALID_ID) {
			// wl_keyboard::enter(serial, surface, keys) - keys will be empty for now
			send_wayland_message(client->socket, local_keyboard_id, 2, { serial_counter++, local_surface_id });
		}
	}

	if (client != main_client) {
		// godot_embedded_client::window_focus_out
		send_wayland_message(main_client->socket, client->embedded_client_id, 3, {});
	}
}

int WaylandEmbedder::allocate_global_id() {
	uint32_t id = INVALID_ID;
	objects.request(id);
	objects[id] = WaylandObject();

	DEBUG_LOG_WAYLAND_EMBED(vformat("Allocated new global id g0x%x", id));

#ifdef DEV_ENABLED
	if (id > WAYLAND_EMBED_ID_MAX) {
		// Oh no. Time for debug info!

#ifdef WAYLAND_EMBED_DEBUG_LOGS_ENABLED
		for (uint32_t i = 1; i < objects.reserved_size(); ++i) {
			WaylandObject &object = objects[id];
			DEBUG_LOG_WAYLAND_EMBED(vformat(" - g0x%x (#%d): %s version %d, data 0x%x", i, i, object.interface->name, object.version, (uintptr_t)object.data));
		}
#endif // WAYLAND_EMBED_DEBUG_LOGS_ENABLED

		CRASH_NOW_MSG("Max ID reached. This might indicate a leak.");
	}
#endif // DEV_ENABLED

	return id;
}

bool WaylandEmbedder::global_surface_is_window(uint32_t p_wl_surface_id) {
	WaylandObject *surface_object = get_object(p_wl_surface_id);
	ERR_FAIL_NULL_V(surface_object, false);
	if (surface_object->interface != &wl_surface_interface || surface_object->data == nullptr) {
		return false;
	}

	WaylandSurfaceData *surface_data = (WaylandSurfaceData *)surface_object->data;
	if (!surface_data->role_object_handle.get()) {
		return false;
	}

	WaylandObject *role_object = surface_data->role_object_handle.get();

	return (role_object && role_object->interface == &xdg_toplevel_interface);
}

bool WaylandEmbedder::handle_generic_msg(Client *client, const WaylandObject *p_object, const struct wl_message *message, const struct msg_info *info, uint32_t *buf, uint32_t instance_id) {
	// We allow client-less events.
	CRASH_COND(client == nullptr && info->direction == ProxyDirection::COMPOSITOR);

	ERR_FAIL_NULL_V(p_object, false);

	bool valid = true;

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

				const struct wl_interface *new_interface = message->types[arg_idx];
				uint32_t new_version = p_object->version;

				if (!new_interface && last_str_len != 0) {
					// When the protocol definition does not define an interface it reports a
					// string and an unsigned integer representing the interface and the
					// version requested.
					new_interface = wl_interface_from_string((char *)(body + last_str_buf_idx + 1), last_str_len);
					new_version = body[arg_idx - 1];
				}

				if (new_interface == nullptr) {
					if (last_str_len > 0) {
						DEBUG_LOG_WAYLAND_EMBED(vformat("Unknown interface %s, marking packet as invalid.", (char *)(body + last_str_buf_idx + 1)));
					} else {
						DEBUG_LOG_WAYLAND_EMBED("Unknown interface, marking packet as invalid.");
					}
					valid = false;
					break;
				}

				if (info->direction == ProxyDirection::COMPOSITOR) {
					// FIXME: Create objects only if the packet is valid.
					uint32_t new_local_id = arg;
					body[buf_idx] = client->new_object(new_local_id, new_interface, new_version);

					if (body[buf_idx] == INVALID_ID) {
						valid = false;
						break;
					}

				} else if (info->direction == ProxyDirection::CLIENT) {
					uint32_t new_global_id = arg;

					if (client) {
						body[buf_idx] = client->new_server_object(new_global_id, new_interface, new_version);
					} else {
						new_server_object(new_global_id, new_interface, new_version);
					}

					if (body[buf_idx] == INVALID_ID) {
						valid = false;
						break;
					}
				}
			} break;

			case 'o': {
				if (!client) {
					break;
				}

				uint32_t obj_id = body[buf_idx];
				if (obj_id == 0) {
					// Object arguments can be nil.
					break;
				}

				if (info->direction == ProxyDirection::CLIENT) {
					if (!client->local_ids.has(obj_id)) {
						DEBUG_LOG_WAYLAND_EMBED(vformat("Object argument g0x%x not found, marking packet as invalid.", obj_id));
						valid = false;
						break;
					}
					body[buf_idx] = instance_id != INVALID_ID ? instance_id : client->get_local_id(obj_id);
				} else if (info->direction == ProxyDirection::COMPOSITOR) {
					if (!client->global_ids.has(obj_id)) {
						DEBUG_LOG_WAYLAND_EMBED(vformat("Object argument l0x%x not found, marking packet as invalid.", obj_id));
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

WaylandEmbedder::MessageStatus WaylandEmbedder::handle_request(LocalObjectHandle p_object, uint32_t p_opcode, const uint32_t *msg_data, size_t msg_len) {
	ERR_FAIL_COND_V(!p_object.is_valid(), MessageStatus::HANDLED);

	WaylandObject *object = p_object.get();
	Client *client = p_object.get_client();

	ERR_FAIL_NULL_V(object, MessageStatus::HANDLED);

	// NOTE: Global ID may be null.
	uint32_t global_id = p_object.get_global_id();
	uint32_t local_id = p_object.get_local_id();

	ERR_FAIL_NULL_V(object->interface, MessageStatus::ERROR);
	const struct wl_interface *interface = object->interface;

	ERR_FAIL_COND_V((int)p_opcode >= interface->method_count, MessageStatus::ERROR);
	const struct wl_message message = interface->methods[p_opcode];

	DEBUG_LOG_WAYLAND_EMBED(vformat("Client #%d -> %s::%s(%s) l0x%x g0x%x", client->socket, interface->name, message.name, message.signature, local_id, global_id));

	const uint32_t *body = msg_data + 2;

	if (registry_globals_names.has(global_id)) {
		int global_name = registry_globals_names[global_id];
		ERR_FAIL_COND_V(!registry_globals.has(global_name), MessageStatus::ERROR);
		RegistryGlobalInfo &global_info = registry_globals[global_name];

		if (global_info.destroyed) {
			DEBUG_LOG_WAYLAND_EMBED("Skipping request for destroyed global object");
			return MessageStatus::HANDLED;
		}
	}

	if (object->interface == &wl_display_interface && p_opcode == WL_DISPLAY_GET_REGISTRY) {
		// The gist of this is that the registry is a global and the compositor can
		// quite simply take for granted that a single client can access any global
		// bound from any registry. Let's remove all doubts by using a single
		// registry (also for efficiency) and doing fancy remaps.
		uint32_t local_registry_id = body[0];

		// Note that the registry has already been allocated in the initialization
		// routine.

		for (KeyValue<uint32_t, RegistryGlobalInfo> &pair : registry_globals) {
			uint32_t global_name = pair.key;
			RegistryGlobalInfo &global_info = pair.value;

			if (global_info.destroyed) {
				continue;
			}

			const struct wl_interface *global_interface = global_info.interface;

			if (client != main_client && embedded_interface_deny_list.has(global_interface)) {
				DEBUG_LOG_WAYLAND_EMBED(vformat("Skipped global announcement %s for embedded client.", global_interface->name));
				continue;
			}

			LocalVector<union wl_argument> args;
			args.push_back(wl_arg_uint(global_name));
			args.push_back(wl_arg_string(global_interface->name));
			args.push_back(wl_arg_uint(global_info.version));

			send_wayland_event(client->socket, local_registry_id, wl_registry_interface, WL_REGISTRY_GLOBAL, args);
		}

		client->wl_registry_instances.insert(local_registry_id);
		client->new_global_instance(local_registry_id, REGISTRY_ID, &wl_registry_interface, 1);

		return MessageStatus::HANDLED;
	}

	if (object->interface == &wl_registry_interface) {
		if (p_opcode == WL_REGISTRY_BIND) {
			// [Request] wl_registry::bind(usun)
			uint32_t global_name = body[0];
			uint32_t interface_name_len = body[1];
			//const char *interface_name = (const char *)(body + 2);
			uint32_t version = body[2 + wl_array_word_offset(interface_name_len)];
			uint32_t new_local_id_idx = 2 + wl_array_word_offset(interface_name_len) + 1;
			uint32_t new_local_id = body[new_local_id_idx];

			if (!registry_globals.has(global_name)) {
				socket_error(client->socket, local_id, WL_DISPLAY_ERROR_INVALID_METHOD, vformat("Invalid global object #%d", global_name));
				return MessageStatus::HANDLED;
			}

			RegistryGlobalInfo &global_info = registry_globals[global_name];
			ERR_FAIL_NULL_V(global_info.interface, MessageStatus::ERROR);

			version = MIN(global_info.version, version);

			if (global_info.interface == &godot_embedding_compositor_interface) {
				if (!client->registry_globals_instances.has(global_name)) {
					client->registry_globals_instances[global_name] = {};
				}

				client->registry_globals_instances[global_name].insert(new_local_id);
				++global_info.instance_counter;
				DEBUG_LOG_WAYLAND_EMBED("Bound embedded compositor interface.");
				client->new_fake_object(new_local_id, &godot_embedding_compositor_interface, 1);
				return MessageStatus::HANDLED;
			}

			WaylandObject *instance = nullptr;

			client->registry_globals_instances[global_name].insert(new_local_id);
			++global_info.instance_counter;

			if (!client->registry_globals_instances.has(global_name)) {
				client->registry_globals_instances[global_name] = {};
			}

			uint32_t bind_gid = wl_registry_bind(REGISTRY_ID, global_name, version);
			if (bind_gid == INVALID_ID) {
				socket_error(client->socket, local_id, WL_DISPLAY_ERROR_IMPLEMENTATION, "Bind failed.");
				return MessageStatus::HANDLED;
			}

			WaylandObject *bind_obj = get_object(bind_gid);
			if (bind_obj == nullptr) {
				socket_error(client->socket, local_id, WL_DISPLAY_ERROR_IMPLEMENTATION, "Bind failed.");
				return MessageStatus::HANDLED;
			}

			if (!bind_obj->shared) {
				client->bind_global_id(bind_gid, new_local_id);
				instance = bind_obj;
			} else {
				instance = client->new_global_instance(new_local_id, global_info.reusable_objects[version], global_info.interface, version);
				DEBUG_LOG_WAYLAND_EMBED(vformat("Instancing global #%d iface %s ver %d new id l0x%x g0x%x", global_name, global_info.interface->name, version, new_local_id, global_info.reusable_objects[version]));

				// Some interfaces report their state as soon as they're bound. Since
				// instances are handled by us, we need to track and report the relevant
				// data ourselves.
				if (global_info.interface == &wl_drm_interface) {
					Error err = client->send_wl_drm_state(new_local_id, (WaylandDrmGlobalData *)global_info.data);
					if (err != OK) {
						return MessageStatus::ERROR;
					}
				} else if (global_info.interface == &wl_shm_interface) {
					WaylandShmGlobalData *global_data = (WaylandShmGlobalData *)global_info.data;
					ERR_FAIL_NULL_V(global_data, MessageStatus::ERROR);

					for (uint32_t format : global_data->formats) {
						send_wayland_message(client->socket, new_local_id, WL_SHM_FORMAT, { format });
					}
				}
			}

			ERR_FAIL_NULL_V(instance, MessageStatus::UNHANDLED);

			if (global_info.interface == &wl_seat_interface) {
				WaylandSeatInstanceData *new_data = memnew(WaylandSeatInstanceData);
				instance->data = new_data;
			}

			return MessageStatus::HANDLED;
		}
	}

	if (object->interface == &wl_compositor_interface && p_opcode == WL_COMPOSITOR_CREATE_SURFACE) {
		uint32_t new_local_id = body[0];

		WaylandSurfaceData *data = memnew(WaylandSurfaceData);
		data->client = client;

		uint32_t new_global_id = client->new_object(new_local_id, &wl_surface_interface, object->version, data);
		ERR_FAIL_COND_V(new_global_id == INVALID_ID, MessageStatus::HANDLED);

		DEBUG_LOG_WAYLAND_EMBED(vformat("Keeping track of surface l0x%x g0x%x.", new_local_id, new_global_id));

		send_wayland_message(compositor_socket, global_id, p_opcode, { new_global_id });
		return MessageStatus::HANDLED;
	}

	if (object->interface == &wl_surface_interface) {
		WaylandSurfaceData *surface_data = (WaylandSurfaceData *)object->data;
		ERR_FAIL_NULL_V(surface_data, MessageStatus::ERROR);

		if (p_opcode == WL_SURFACE_DESTROY) {
			for (uint32_t wl_seat_name : wl_seat_names) {
				WaylandSeatGlobalData *global_seat_data = (WaylandSeatGlobalData *)registry_globals[wl_seat_name].data;
				ERR_FAIL_NULL_V(global_seat_data, MessageStatus::ERROR);

				if (global_seat_data->pointed_surface_id == global_id) {
					global_seat_data->pointed_surface_id = INVALID_ID;
				}

				if (global_seat_data->focused_surface_id == global_id) {
					global_seat_data->focused_surface_id = INVALID_ID;
				}
			}
		} else if (p_opcode == WL_SURFACE_COMMIT) {
			if (surface_data->role_object_handle.is_valid()) {
				WaylandObject *role_object = surface_data->role_object_handle.get();
				if (role_object && role_object->interface) {
					DEBUG_LOG_WAYLAND_EMBED(vformat("!!!!! Committed surface g0x%x with role object %s id l0x%x", global_id, role_object->interface->name, surface_data->role_object_handle.get_local_id()));
				}

				if (role_object && role_object->interface == &xdg_toplevel_interface) {
					XdgToplevelData *toplevel_data = (XdgToplevelData *)role_object->data;
					ERR_FAIL_NULL_V(toplevel_data, MessageStatus::ERROR);
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
			return MessageStatus::HANDLED;
		}
	}

	if (object->interface == &wl_seat_interface) {
		uint32_t global_seat_name = registry_globals_names[global_id];

		RegistryGlobalInfo &seat_global_info = registry_globals[global_seat_name];
		WaylandSeatGlobalData *global_data = (WaylandSeatGlobalData *)seat_global_info.data;
		ERR_FAIL_NULL_V(global_data, MessageStatus::ERROR);

		WaylandSeatInstanceData *instance_data = (WaylandSeatInstanceData *)object->data;
		ERR_FAIL_NULL_V(instance_data, MessageStatus::ERROR);

		if (p_opcode == WL_SEAT_GET_POINTER) {
			ERR_FAIL_COND_V(global_id == INVALID_ID, MessageStatus::ERROR);
			// [Request] wl_seat::get_pointer(n);
			uint32_t new_local_id = body[0];

			WaylandPointerData *new_data = memnew(WaylandPointerData);
			new_data->wl_seat_id = global_id;

			uint32_t new_global_id = client->new_object(new_local_id, &wl_pointer_interface, object->version, new_data);
			ERR_FAIL_COND_V(new_global_id == INVALID_ID, MessageStatus::HANDLED);

			instance_data->wl_pointer_id = new_global_id;

			send_wayland_message(compositor_socket, global_id, p_opcode, { new_global_id });

			return MessageStatus::HANDLED;
		}

		if (p_opcode == WL_SEAT_GET_KEYBOARD) {
			ERR_FAIL_COND_V(global_id == INVALID_ID, MessageStatus::ERROR);
			// [Request] wl_seat::get_pointer(n);
			uint32_t new_local_id = body[0];

			WaylandKeyboardData *new_data = memnew(WaylandKeyboardData);
			new_data->wl_seat_id = global_id;

			uint32_t new_global_id = client->new_object(new_local_id, &wl_keyboard_interface, object->version, new_data);
			ERR_FAIL_COND_V(new_global_id == INVALID_ID, MessageStatus::HANDLED);

			instance_data->wl_keyboard_id = new_global_id;

			send_wayland_message(compositor_socket, global_id, p_opcode, { new_global_id });

			return MessageStatus::HANDLED;
		}
	}

	if (object->interface == &xdg_wm_base_interface) {
		if (p_opcode == XDG_WM_BASE_CREATE_POSITIONER) {
			uint32_t new_local_id = body[0];
			uint32_t new_global_id = client->new_object(new_local_id, &xdg_positioner_interface, object->version, memnew(XdgPositionerData));
			ERR_FAIL_COND_V(new_global_id == INVALID_ID, MessageStatus::HANDLED);

			send_wayland_message(compositor_socket, global_id, p_opcode, { new_global_id });
			return MessageStatus::HANDLED;
		}

		if (p_opcode == XDG_WM_BASE_GET_XDG_SURFACE) {
			// [Request] xdg_wm_base::get_xdg_surface(no).
			uint32_t new_local_id = body[0];
			uint32_t surface_id = body[1];

			uint32_t global_surface_id = client->get_global_id(surface_id);

			bool fake = (client != main_client);

			XdgSurfaceData *data = memnew(XdgSurfaceData);
			data->wl_surface_id = global_surface_id;

			if (fake) {
				client->new_fake_object(new_local_id, &xdg_surface_interface, object->version, data);
				DEBUG_LOG_WAYLAND_EMBED(vformat("Created fake xdg_surface l0x%x for surface l0x%x", new_local_id, surface_id));
			} else {
				uint32_t new_global_id = client->new_object(new_local_id, &xdg_surface_interface, object->version, data);
				ERR_FAIL_COND_V(new_global_id == INVALID_ID, MessageStatus::HANDLED);

				DEBUG_LOG_WAYLAND_EMBED(vformat("Created real xdg_surface l0x%x g0x%x for surface l0x%x", new_local_id, new_global_id, surface_id));

				send_wayland_message(compositor_socket, global_id, p_opcode, { new_global_id, global_surface_id });
			}

			return MessageStatus::HANDLED;
		}
	}

	if (object->interface == &xdg_surface_interface) {
		XdgSurfaceData *xdg_surf_data = (XdgSurfaceData *)object->data;
		ERR_FAIL_NULL_V(xdg_surf_data, MessageStatus::ERROR);

		WaylandSurfaceData *surface_data = (WaylandSurfaceData *)get_object(xdg_surf_data->wl_surface_id)->data;
		ERR_FAIL_NULL_V(surface_data, MessageStatus::ERROR);

		bool is_embedded = client->fake_objects.has(local_id);

		if (p_opcode == XDG_SURFACE_GET_POPUP) {
			// [Request] xdg_surface::get_popup(no?o).

			uint32_t new_local_id = body[0];
			uint32_t local_parent_id = body[1];
			uint32_t local_positioner_id = body[2];

			surface_data->role_object_handle = LocalObjectHandle(client, new_local_id);

			XdgPopupData *popup_data = memnew(XdgPopupData);
			popup_data->parent_handle = LocalObjectHandle(client, local_parent_id);

			if (!is_embedded) {
				uint32_t new_global_id = client->new_object(new_local_id, &xdg_popup_interface, object->version, popup_data);
				ERR_FAIL_COND_V(new_global_id == INVALID_ID, MessageStatus::HANDLED);

				uint32_t global_parent_id = client->get_global_id(local_parent_id);
				uint32_t global_positioner_id = client->get_global_id(local_positioner_id);
				send_wayland_message(compositor_socket, global_id, p_opcode, { new_global_id, global_parent_id, global_positioner_id });

				return MessageStatus::HANDLED;
			}

			{
				// Popups are real, time to actually instantiate an xdg_surface.
				WaylandObject copy = *object;
				client->fake_objects.erase(local_id);

				global_id = client->new_object(local_id, copy.interface, copy.version, copy.data);
				ERR_FAIL_COND_V(global_id == INVALID_ID, MessageStatus::HANDLED);
				object = get_object(global_id);

				// xdg_wm_base::get_xdg_surface(no);
				send_wayland_message(compositor_socket, xdg_wm_base_id, 2, { global_id, xdg_surf_data->wl_surface_id });
			}

			uint32_t new_global_id = client->new_object(new_local_id, &xdg_popup_interface, object->version, popup_data);
			ERR_FAIL_COND_V(new_global_id == INVALID_ID, MessageStatus::HANDLED);

			uint32_t global_parent_id = INVALID_ID;
			if (local_parent_id != INVALID_ID) {
				XdgSurfaceData *parent_xdg_surf_data = (XdgSurfaceData *)client->get_object(local_parent_id)->data;
				ERR_FAIL_NULL_V(parent_xdg_surf_data, MessageStatus::ERROR);

				WaylandSurfaceData *parent_surface_data = (WaylandSurfaceData *)get_object(parent_xdg_surf_data->wl_surface_id)->data;
				ERR_FAIL_NULL_V(parent_surface_data, MessageStatus::ERROR);

				WaylandObject *parent_role_obj = parent_surface_data->role_object_handle.get();
				ERR_FAIL_NULL_V(parent_role_obj, MessageStatus::ERROR);

				XdgPositionerData *pos_data = (XdgPositionerData *)client->get_object(local_positioner_id)->data;
				ERR_FAIL_NULL_V(pos_data, MessageStatus::ERROR);

				if (parent_role_obj->interface == &xdg_toplevel_interface) {
					XdgToplevelData *parent_toplevel_data = (XdgToplevelData *)parent_role_obj->data;
					ERR_FAIL_NULL_V(parent_toplevel_data, MessageStatus::ERROR);

					if (parent_toplevel_data->is_embedded()) {
						// Embedded windows are subsurfaces of a parent window. We need to
						// "redirect" the popup request on the parent window and adjust the
						// positioner properly if needed.

						XdgToplevelData *main_parent_toplevel_data = (XdgToplevelData *)parent_toplevel_data->parent_handle.get()->data;
						ERR_FAIL_NULL_V(main_parent_toplevel_data, MessageStatus::ERROR);

						global_parent_id = main_parent_toplevel_data->xdg_surface_handle.get_global_id();

						WaylandSubsurfaceData *subsurf_data = (WaylandSubsurfaceData *)get_object(parent_toplevel_data->wl_subsurface_id)->data;
						ERR_FAIL_NULL_V(subsurf_data, MessageStatus::ERROR);

						Point2i adj_pos = subsurf_data->position + pos_data->anchor_rect.position;

						// xdg_positioner::set_anchor_rect
						send_wayland_message(compositor_socket, client->get_global_id(local_positioner_id), 2, { (uint32_t)adj_pos.x, (uint32_t)adj_pos.y, (uint32_t)pos_data->anchor_rect.size.width, (uint32_t)pos_data->anchor_rect.size.height });
					}
				} else {
					global_parent_id = client->get_global_id(local_parent_id);
				}
			}

			send_wayland_message(compositor_socket, global_id, p_opcode, { new_global_id, global_parent_id, client->get_global_id(local_positioner_id) });
			return MessageStatus::HANDLED;
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
				send_wayland_message(main_client->socket, client->embedded_client_id, 1, {});
			} else {
				uint32_t new_global_id = client->new_object(new_local_id, &xdg_toplevel_interface, object->version, data);
				ERR_FAIL_COND_V(new_global_id == INVALID_ID, MessageStatus::HANDLED);

				if (main_toplevel_id == 0) {
					main_toplevel_id = new_global_id;
					DEBUG_LOG_WAYLAND_EMBED(vformat("main toplevel set to gx0%x.", main_toplevel_id));
				}

				send_wayland_message(compositor_socket, global_id, p_opcode, { new_global_id });
			}

			return MessageStatus::HANDLED;
		}
	}

	if (object->interface == &xdg_positioner_interface) {
		XdgPositionerData *pos_data = (XdgPositionerData *)object->data;
		ERR_FAIL_NULL_V(pos_data, MessageStatus::ERROR);

		if (p_opcode == XDG_POSITIONER_SET_ANCHOR_RECT) {
			// Args: int x, int y, int width, int height.
			pos_data->anchor_rect = Rect2i(body[0], body[1], body[2], body[3]);

			send_wayland_message(compositor_socket, global_id, p_opcode, { body[0], body[1], body[2], body[3] });
			return MessageStatus::HANDLED;
		}
	}

	if (object->interface == &xdg_toplevel_interface && p_opcode == XDG_TOPLEVEL_DESTROY) {
		if (client->fake_objects.has(local_id)) {
			XdgToplevelData *data = (XdgToplevelData *)object->data;
			ERR_FAIL_NULL_V(data, MessageStatus::ERROR);

			XdgSurfaceData *xdg_surf_data = nullptr;
			if (data->xdg_surface_handle.is_valid()) {
				xdg_surf_data = (XdgSurfaceData *)data->xdg_surface_handle.get()->data;
				ERR_FAIL_NULL_V(xdg_surf_data, MessageStatus::ERROR);
			}
			ERR_FAIL_NULL_V(xdg_surf_data, MessageStatus::ERROR);

			XdgSurfaceData *parent_xdg_surf_data = nullptr;
			{
				XdgToplevelData *parent_data = nullptr;
				if (data->parent_handle.get()) {
					parent_data = (XdgToplevelData *)data->parent_handle.get()->data;
					ERR_FAIL_NULL_V(parent_data, MessageStatus::ERROR);
				}

				if (parent_data && parent_data->xdg_surface_handle.get()) {
					parent_xdg_surf_data = (XdgSurfaceData *)parent_data->xdg_surface_handle.get()->data;
					ERR_FAIL_NULL_V(parent_xdg_surf_data, MessageStatus::ERROR);
				}
			}

			for (uint32_t wl_seat_name : wl_seat_names) {
				WaylandSeatGlobalData *global_seat_data = (WaylandSeatGlobalData *)registry_globals[wl_seat_name].data;
				ERR_FAIL_NULL_V(global_seat_data, MessageStatus::ERROR);

				if (global_seat_data->focused_surface_id == xdg_surf_data->wl_surface_id) {
					if (xdg_surf_data) {
						seat_name_leave_surface(wl_seat_name, xdg_surf_data->wl_surface_id);
					}

					if (parent_xdg_surf_data) {
						seat_name_enter_surface(wl_seat_name, parent_xdg_surf_data->wl_surface_id);
					}
				}
			}

			// wl_display::delete_id
			send_wayland_message(client->socket, local_id, p_opcode, {});

			if (local_id == client->embedded_window_id) {
				client->embedded_window_id = 0;
			}

			if (data->wl_subsurface_id != INVALID_ID) {
				send_wayland_message(compositor_socket, data->wl_subsurface_id, WL_SUBSURFACE_DESTROY, {});
			}

			client->delete_object(local_id);

			return MessageStatus::HANDLED;
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
			ERR_FAIL_NULL_V(surf_data, MessageStatus::ERROR);

			WaylandObject *role_obj = surf_data->role_object_handle.get();
			ERR_FAIL_NULL_V(role_obj, MessageStatus::ERROR);

			if (role_obj->interface == &xdg_toplevel_interface) {
				XdgToplevelData *toplevel_data = (XdgToplevelData *)role_obj->data;
				ERR_FAIL_NULL_V(toplevel_data, MessageStatus::ERROR);

				if (!toplevel_data->is_embedded()) {
					// Passthrough.
					return MessageStatus::UNHANDLED;
				}

				// Subsurfaces don't normally work, at least on sway, as the locking
				// condition might rely on focus, which they don't get. We can remap them to
				// the parent surface and set a region though.

				XdgToplevelData *parent_data = (XdgToplevelData *)toplevel_data->parent_handle.get()->data;
				ERR_FAIL_NULL_V(parent_data, MessageStatus::ERROR);

				XdgSurfaceData *parent_xdg_surf_data = (XdgSurfaceData *)parent_data->xdg_surface_handle.get()->data;
				ERR_FAIL_NULL_V(parent_xdg_surf_data, MessageStatus::ERROR);

				WaylandSubsurfaceData *subsurf_data = (WaylandSubsurfaceData *)get_object(toplevel_data->wl_subsurface_id)->data;
				ERR_FAIL_NULL_V(subsurf_data, MessageStatus::ERROR);

				uint32_t new_global_id = client->new_object(new_local_id, &zwp_locked_pointer_v1_interface, object->version);
				ERR_FAIL_COND_V(new_global_id == INVALID_ID, MessageStatus::HANDLED);

				uint32_t x = subsurf_data->position.x;
				uint32_t y = subsurf_data->position.y;
				uint32_t width = toplevel_data->size.width;
				uint32_t height = toplevel_data->size.height;

				// NOTE: At least on sway I can't seem to be able to get this region
				// working but the calls check out.
				DEBUG_LOG_WAYLAND_EMBED(vformat("Creating custom region x%d y%d w%d h%d", x, y, width, height));

				uint32_t new_region_id = allocate_global_id();
				get_object(new_region_id)->interface = &wl_region_interface;
				get_object(new_region_id)->version = get_object(wl_compositor_id)->version;

				// wl_compostor::create_region(n).
				send_wayland_message(compositor_socket, wl_compositor_id, 1, { new_region_id });

				// wl_region::add(iiii).
				send_wayland_message(compositor_socket, new_region_id, 1, { x, y, width, height });

				send_wayland_message(compositor_socket, global_id, p_opcode, { new_global_id, parent_xdg_surf_data->wl_surface_id, client->get_global_id(local_pointer_id), new_region_id, lifetime });

				// wl_region::destroy().
				send_wayland_message(compositor_socket, new_region_id, 0, {});

				return MessageStatus::HANDLED;
			}
		}
	}

	if (interface == &godot_embedded_client_interface) {
		EmbeddedClientData *eclient_data = (EmbeddedClientData *)object->data;
		ERR_FAIL_NULL_V(eclient_data, MessageStatus::ERROR);

		Client *eclient = eclient_data->client;
		ERR_FAIL_NULL_V(eclient, MessageStatus::ERROR);

		if (p_opcode == GODOT_EMBEDDED_CLIENT_DESTROY) {
			if (!eclient_data->disconnected) {
				close(eclient->socket);
			}

			client->delete_object(local_id);

			return MessageStatus::HANDLED;
		}

		if (eclient_data->disconnected) {
			// Object is inert.
			return MessageStatus::HANDLED;
		}

		ERR_FAIL_COND_V(eclient->embedded_window_id == 0, MessageStatus::ERROR);

		XdgToplevelData *toplevel_data = (XdgToplevelData *)eclient->get_object(eclient->embedded_window_id)->data;
		ERR_FAIL_NULL_V(toplevel_data, MessageStatus::ERROR);

		if (p_opcode == GODOT_EMBEDDED_CLIENT_SET_EMBEDDED_WINDOW_RECT && toplevel_data->wl_subsurface_id != INVALID_ID) {
			uint32_t x = body[0];
			uint32_t y = body[1];
			uint32_t width = body[2];
			uint32_t height = body[3];

			WaylandSubsurfaceData *subsurf_data = (WaylandSubsurfaceData *)get_object(toplevel_data->wl_subsurface_id)->data;
			ERR_FAIL_NULL_V(subsurf_data, MessageStatus::ERROR);

			toplevel_data->size.width = width;
			toplevel_data->size.height = height;

			subsurf_data->position.x = x;
			subsurf_data->position.y = y;

			// wl_subsurface::set_position
			send_wayland_message(compositor_socket, toplevel_data->wl_subsurface_id, 1, { x, y });

			// xdg_toplevel::configure
			send_wayland_message(eclient->socket, eclient->embedded_window_id, 0, { width, height, 0 });

			// xdg_surface::configure
			send_wayland_message(eclient->socket, toplevel_data->xdg_surface_handle.get_local_id(), 0, { configure_serial_counter++ });

			return MessageStatus::HANDLED;
		} else if (p_opcode == GODOT_EMBEDDED_CLIENT_SET_EMBEDDED_WINDOW_PARENT) {
			uint32_t main_client_parent_id = body[0];

			if (toplevel_data->parent_handle.get_local_id() == main_client_parent_id) {
				return MessageStatus::HANDLED;
			}

			if (main_client_parent_id == INVALID_ID && toplevel_data->wl_subsurface_id != INVALID_ID) {
				// Window hiding logic.

				// wl_subsurface::destroy()
				send_wayland_message(compositor_socket, toplevel_data->wl_subsurface_id, 0, {});

				toplevel_data->parent_handle.invalidate();
				toplevel_data->wl_subsurface_id = INVALID_ID;

				return MessageStatus::HANDLED;
			}

			XdgToplevelData *parent_toplevel_data = (XdgToplevelData *)client->get_object(main_client_parent_id)->data;
			ERR_FAIL_NULL_V(parent_toplevel_data, MessageStatus::ERROR);
			XdgSurfaceData *parent_xdg_surf_data = (XdgSurfaceData *)parent_toplevel_data->xdg_surface_handle.get()->data;
			ERR_FAIL_NULL_V(parent_xdg_surf_data, MessageStatus::ERROR);

			XdgSurfaceData *xdg_surf_data = (XdgSurfaceData *)toplevel_data->xdg_surface_handle.get()->data;
			ERR_FAIL_NULL_V(xdg_surf_data, MessageStatus::ERROR);

			if (toplevel_data->wl_subsurface_id != INVALID_ID) {
				// wl_subsurface::destroy()
				send_wayland_message(compositor_socket, toplevel_data->wl_subsurface_id, 0, {});
			}

			uint32_t new_sub_id = allocate_global_id();
			WaylandObject *new_sub_object = get_object(new_sub_id);
			new_sub_object->interface = &wl_subsurface_interface;
			new_sub_object->data = memnew(WaylandSubsurfaceData);
			new_sub_object->version = get_object(wl_subcompositor_id)->version;

			toplevel_data->wl_subsurface_id = new_sub_id;
			toplevel_data->parent_handle = LocalObjectHandle(main_client, main_client_parent_id);

			DEBUG_LOG_WAYLAND_EMBED(vformat("Binding subsurface g0x%x.", new_sub_id));

			// wl_subcompositor::get_subsurface
			send_wayland_message(compositor_socket, wl_subcompositor_id, 1, { new_sub_id, xdg_surf_data->wl_surface_id, parent_xdg_surf_data->wl_surface_id });

			// wl_subsurface::set_desync
			send_wayland_message(compositor_socket, new_sub_id, 5, {});

			return MessageStatus::HANDLED;
		} else if (p_opcode == GODOT_EMBEDDED_CLIENT_FOCUS_WINDOW) {
			XdgSurfaceData *xdg_surf_data = (XdgSurfaceData *)toplevel_data->xdg_surface_handle.get()->data;
			ERR_FAIL_NULL_V(xdg_surf_data, MessageStatus::ERROR);

			for (uint32_t wl_seat_name : wl_seat_names) {
				RegistryGlobalInfo &global_seat_info = registry_globals[wl_seat_name];
				WaylandSeatGlobalData *global_seat_data = (WaylandSeatGlobalData *)global_seat_info.data;

				if (global_seat_data->focused_surface_id != INVALID_ID) {
					seat_name_leave_surface(wl_seat_name, global_seat_data->focused_surface_id);
				}
				global_seat_data->focused_surface_id = xdg_surf_data->wl_surface_id;

				seat_name_enter_surface(wl_seat_name, xdg_surf_data->wl_surface_id);
			}
		} else if (p_opcode == GODOT_EMBEDDED_CLIENT_EMBEDDED_WINDOW_REQUEST_CLOSE) {
			// xdg_toplevel::close
			send_wayland_message(eclient->socket, eclient->embedded_window_id, 1, {});

			return MessageStatus::HANDLED;
		}
	}

	// Server-allocated objects are a bit annoying to handle for us. Right now we
	// use a heuristic. See: https://ppaalanen.blogspot.com/2014/07/wayland-protocol-design-object-lifespan.html
	if (strcmp(message.name, "destroy") == 0 || strcmp(message.name, "release") == 0) {
		if (object->shared) {
			// We must not delete shared objects.
			client->delete_object(local_id);
			return MessageStatus::HANDLED;
		}

		if (global_id != INVALID_ID) {
			send_wayland_message(compositor_socket, global_id, p_opcode, {});
			object->destroyed = true;
		}

		if (local_id & 0xff000000) {
			DEBUG_LOG_WAYLAND_EMBED(vformat("!!!!!! Deallocating server object l0x%x", local_id));
			client->delete_object(local_id);
		}

		return MessageStatus::HANDLED;
	}

	if (client->fake_objects.has(local_id)) {
		// Object is fake, we're done.
		DEBUG_LOG_WAYLAND_EMBED("Dropping unhandled request for fake object.");
		return MessageStatus::HANDLED;
	}

	if (global_id == INVALID_ID) {
		DEBUG_LOG_WAYLAND_EMBED("Dropping request with invalid global object id");
		return MessageStatus::HANDLED;
	}

	return MessageStatus::UNHANDLED;
}

WaylandEmbedder::MessageStatus WaylandEmbedder::handle_event(uint32_t p_global_id, LocalObjectHandle p_local_handle, uint32_t p_opcode, const uint32_t *msg_data, size_t msg_len) {
	WaylandObject *global_object = get_object(p_global_id);
	ERR_FAIL_NULL_V_MSG(global_object, MessageStatus::ERROR, "Compositor messages must always have a global object.");

#ifdef WAYLAND_EMBED_DEBUG_LOGS_ENABLED
	ERR_FAIL_NULL_V(global_object->interface, MessageStatus::ERROR);
	const struct wl_interface *interface = global_object->interface;

	ERR_FAIL_COND_V((int)p_opcode >= interface->event_count, MessageStatus::ERROR);
	const struct wl_message message = interface->events[p_opcode];

	if (p_local_handle.is_valid()) {
		int socket = p_local_handle.get_client()->socket;
		DEBUG_LOG_WAYLAND_EMBED(vformat("Client #%d <- %s::%s(%s) g0x%x", socket, interface->name, message.name, message.signature, p_global_id));
	} else {
		DEBUG_LOG_WAYLAND_EMBED(vformat("Client N/A <- %s::%s(%s) g0x%x", interface->name, message.name, message.signature, p_global_id));
	}
#endif //WAYLAND_EMBED_DEBUG_LOGS_ENABLED

	const uint32_t *body = msg_data + 2;
	//size_t body_len = msg_len - (WL_WORD_SIZE * 2);

	// FIXME: Make sure that it makes sense to track this protocol. Not only is it
	// old and getting deprecated, but I can't even get this code branch to hit
	// probably because, at the time of writing, we only get the "main" display
	// through the proxy.
	if (global_object->interface == &wl_drm_interface) {
		// wl_drm can't ever be destroyed, so we need to track its state as it's going
		// to be instanced at least few times.
		uint32_t global_name = registry_globals_names[p_global_id];
		WaylandDrmGlobalData *global_data = (WaylandDrmGlobalData *)registry_globals[global_name].data;
		ERR_FAIL_NULL_V(global_data, MessageStatus::ERROR);

		if (p_opcode == WL_DRM_DEVICE) {
			// signature: s
			uint32_t name_len = body[0];
			uint8_t *name = (uint8_t *)(body + 1);
			global_data->device = String::utf8((const char *)name, name_len);

			return MessageStatus::UNHANDLED;
		}

		if (p_opcode == WL_DRM_FORMAT) {
			// signature: u
			uint32_t format = body[0];
			global_data->formats.push_back(format);

			return MessageStatus::UNHANDLED;
		}

		if (p_opcode == WL_DRM_AUTHENTICATED) {
			// signature: N/A
			global_data->authenticated = true;

			return MessageStatus::UNHANDLED;
		}

		if (p_opcode == WL_DRM_CAPABILITIES) {
			// signature: u
			uint32_t capabilities = body[0];
			global_data->capabilities = capabilities;
		}

		return MessageStatus::UNHANDLED;
	}

	if (global_object->interface == &wl_shm_interface) {
		uint32_t global_name = registry_globals_names[p_global_id];
		WaylandShmGlobalData *global_data = (WaylandShmGlobalData *)registry_globals[global_name].data;
		ERR_FAIL_NULL_V(global_data, MessageStatus::ERROR);

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
				DEBUG_LOG_WAYLAND_EMBED(vformat("Compositor requested deletion of g0x%x (no client)", global_delete_id));

				delete_object(global_delete_id);

				return MessageStatus::HANDLED;
			} else if (p_opcode == WL_DISPLAY_ERROR) {
				// [Event] wl_display::error(ous)
				uint32_t obj_id = body[0];
				uint32_t err_code = body[1];

				CRASH_NOW_MSG(vformat("Error obj g0x%x code %d: %s", obj_id, err_code, (const char *)(body + 3)));
			}
		}

		if (global_object->interface == &wl_callback_interface && p_opcode == WL_CALLBACK_DONE) {
			if (sync_callback_id != INVALID_ID && p_global_id == sync_callback_id) {
				sync_callback_id = 0;
				DEBUG_LOG_WAYLAND_EMBED("Sync response received");
				return MessageStatus::HANDLED;
			}
		}

		if (global_object->interface == &wl_registry_interface) {
			if (p_opcode == WL_REGISTRY_GLOBAL) {
				// [Event] wl_registry::global(usu).

				uint32_t global_name = body[0];
				uint32_t interface_name_len = body[1];
				const char *interface_name = (const char *)(body + 2);
				uint32_t global_version = body[2 + wl_array_word_offset(interface_name_len)];

				DEBUG_LOG_WAYLAND_EMBED("Global c#%d %s %d", global_name, interface_name, global_version);

				const struct wl_interface *global_interface = wl_interface_from_string(interface_name, interface_name_len);
				if (global_interface) {
					RegistryGlobalInfo global_info = {};
					global_info.interface = global_interface;
					global_info.version = MIN(global_version, (uint32_t)global_interface->version);
					DEBUG_LOG_WAYLAND_EMBED("Clamped global %s to version %d.", interface_name, global_info.version);
					global_info.compositor_name = global_name;

					int new_global_name = registry_globals_counter++;

					if (global_info.interface == &wl_shm_interface) {
						DEBUG_LOG_WAYLAND_EMBED("Allocating global wl_shm data.");
						global_info.data = memnew(WaylandShmGlobalData);
					}

					if (global_info.interface == &wl_seat_interface) {
						DEBUG_LOG_WAYLAND_EMBED("Allocating global wl_seat data.");
						global_info.data = memnew(WaylandSeatGlobalData);
						wl_seat_names.push_back(new_global_name);
					}

					if (global_info.interface == &wl_drm_interface) {
						DEBUG_LOG_WAYLAND_EMBED("Allocating global wl_drm data.");
						global_info.data = memnew(WaylandDrmGlobalData);
					}

					registry_globals[new_global_name] = global_info;

					// We need some interfaces directly. It's better to bind a "copy" ourselves
					// than to wait for the client to ask one.
					if (global_interface == &xdg_wm_base_interface && xdg_wm_base_id == 0) {
						xdg_wm_base_id = wl_registry_bind(p_global_id, new_global_name, global_info.version);
						ERR_FAIL_COND_V(xdg_wm_base_id == INVALID_ID, MessageStatus::ERROR);
					} else if (global_interface == &wl_compositor_interface && wl_compositor_id == 0) {
						wl_compositor_id = wl_registry_bind(p_global_id, new_global_name, global_info.version);
						ERR_FAIL_COND_V(wl_compositor_id == INVALID_ID, MessageStatus::ERROR);
					} else if (global_interface == &wl_subcompositor_interface && wl_subcompositor_id == 0) {
						wl_subcompositor_id = wl_registry_bind(p_global_id, new_global_name, global_info.version);
						ERR_FAIL_COND_V(wl_subcompositor_id == INVALID_ID, MessageStatus::ERROR);
					}

					DEBUG_LOG_WAYLAND_EMBED(vformat("Local registry object name: l#%d", new_global_name));

					if (clients.is_empty()) {
						// Let's not waste time.
						return MessageStatus::HANDLED;
					}

					// Notify all clients.
					LocalVector<wl_argument> args;
					args.push_back(wl_arg_uint(new_global_name));
					args.push_back(wl_arg_string(interface_name));
					args.push_back(wl_arg_uint(global_info.version));
					for (KeyValue<int, Client> &pair : clients) {
						Client &client = pair.value;
						for (uint32_t local_registry_id : client.wl_registry_instances) {
							send_wayland_event(client.socket, local_registry_id, wl_registry_interface, WL_REGISTRY_GLOBAL, args);
						}
					}

					return MessageStatus::HANDLED;
				} else {
					DEBUG_LOG_WAYLAND_EMBED("Skipping unknown global %s version %d.", interface_name, global_version);

					return MessageStatus::HANDLED;
				}
			} else if (p_opcode == WL_REGISTRY_GLOBAL_REMOVE) {
				uint32_t compositor_name = body[0];
				uint32_t local_name = 0;
				RegistryGlobalInfo *global_info = nullptr;

				// FIXME: Use a map or something.
				for (KeyValue<uint32_t, RegistryGlobalInfo> &pair : registry_globals) {
					uint32_t name = pair.key;
					RegistryGlobalInfo &info = pair.value;

					if (info.compositor_name == compositor_name) {
						local_name = name;
						global_info = &info;
						break;
					}
				}

				ERR_FAIL_NULL_V(global_info, MessageStatus::ERROR);

				if (global_info->instance_counter == 0) {
					memdelete(global_info->data);
					registry_globals.erase(local_name);
				} else {
					global_info->destroyed = true;
				}

				// Notify all clients.
				LocalVector<wl_argument> args;
				args.push_back(wl_arg_uint(local_name));
				for (KeyValue<int, Client> &pair : clients) {
					Client &client = pair.value;
					for (uint32_t local_registry_id : client.wl_registry_instances) {
						send_wayland_event(client.socket, local_registry_id, wl_registry_interface, WL_REGISTRY_GLOBAL_REMOVE, args);
					}
				}

				return MessageStatus::HANDLED;
			}
		}

		DEBUG_LOG_WAYLAND_EMBED("No valid local object handle, falling back to generic handler.");
		return MessageStatus::UNHANDLED;
	}

	Client *client = p_local_handle.get_client();

	ERR_FAIL_NULL_V(client, MessageStatus::ERROR);

	WaylandObject *object = p_local_handle.get();
	uint32_t local_id = p_local_handle.get_local_id();

	if (global_object->interface == &wl_display_interface) {
		if (p_opcode == WL_DISPLAY_DELETE_ID) {
			// [Event] wl_display::delete_id(u)
			uint32_t global_delete_id = body[0];
			uint32_t local_delete_id = client->get_local_id(global_delete_id);
			DEBUG_LOG_WAYLAND_EMBED(vformat("Compositor requested delete of g0x%x l0x%x", global_delete_id, local_delete_id));
			if (local_delete_id == INVALID_ID) {
				// No idea what this object is, might be of the other client. This
				// definitely does not make sense to us, so we're done.
				return MessageStatus::INVALID;
			}

			client->delete_object(local_delete_id);

			send_wayland_message(client->socket, DISPLAY_ID, WL_DISPLAY_DELETE_ID, { local_delete_id });

			return MessageStatus::HANDLED;
		}

		return MessageStatus::UNHANDLED;
	}

	if (object->interface == &wl_keyboard_interface) {
		WaylandKeyboardData *data = (WaylandKeyboardData *)object->data;
		ERR_FAIL_NULL_V(data, MessageStatus::ERROR);

		uint32_t global_seat_name = registry_globals_names[data->wl_seat_id];
		RegistryGlobalInfo &global_seat_info = registry_globals[global_seat_name];
		WaylandSeatGlobalData *global_seat_data = (WaylandSeatGlobalData *)global_seat_info.data;
		ERR_FAIL_NULL_V(global_seat_data, MessageStatus::ERROR);

		if (p_opcode == WL_KEYBOARD_ENTER) {
			// [Event] wl_keyboard::enter(uoa)
			uint32_t surface = body[1];

			if (global_seat_data->focused_surface_id != surface) {
				DEBUG_LOG_WAYLAND_EMBED(vformat("Focused g0x%x", surface));
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
				DEBUG_LOG_WAYLAND_EMBED(vformat("Skipped wl_keyboard event due to unfocused surface 0x%x", global_seat_data->focused_surface_id));
				return MessageStatus::HANDLED;
			}
		}

		return MessageStatus::UNHANDLED;
	}

	if (object->interface == &wl_pointer_interface) {
		WaylandPointerData *data = (WaylandPointerData *)object->data;
		ERR_FAIL_NULL_V(data, MessageStatus::ERROR);

		uint32_t global_seat_name = registry_globals_names[data->wl_seat_id];
		RegistryGlobalInfo &global_seat_info = registry_globals[global_seat_name];
		WaylandSeatGlobalData *global_seat_data = (WaylandSeatGlobalData *)global_seat_info.data;
		ERR_FAIL_NULL_V(global_seat_data, MessageStatus::ERROR);

		WaylandSeatInstanceData *seat_data = (WaylandSeatInstanceData *)object->data;
		ERR_FAIL_NULL_V(seat_data, MessageStatus::ERROR);

		if (p_opcode == WL_POINTER_BUTTON && global_seat_data->pointed_surface_id != INVALID_ID) {
			// [Event] wl_pointer::button(uuuu);
			uint32_t button = body[2];
			uint32_t state = body[3];

			DEBUG_LOG_WAYLAND_EMBED(vformat("Button %d state %d on surface g0x%x (focused g0x%x)", button, state, global_seat_data->pointed_surface_id, global_seat_data->focused_surface_id));

			bool client_pointed = client->local_ids.has(global_seat_data->pointed_surface_id);

			if (button != BTN_LEFT || state != WL_POINTER_BUTTON_STATE_RELEASED) {
				return client_pointed ? MessageStatus::UNHANDLED : MessageStatus::HANDLED;
			}

			if (global_seat_data->focused_surface_id == global_seat_data->pointed_surface_id) {
				return client_pointed ? MessageStatus::UNHANDLED : MessageStatus::HANDLED;
			}

			if (!global_surface_is_window(global_seat_data->pointed_surface_id)) {
				return client_pointed ? MessageStatus::UNHANDLED : MessageStatus::HANDLED;
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
			ERR_FAIL_NULL_V(surface_data, MessageStatus::ERROR);

			if (global_seat_data->pointed_surface_id != surface) {
				DEBUG_LOG_WAYLAND_EMBED(vformat("Pointer (g0x%x seat g0x%x): pointed surface old g0x%x new g0x%x", p_global_id, data->wl_seat_id, global_seat_data->pointed_surface_id, surface));

				global_seat_data->pointed_surface_id = surface;
			}
		} else if (p_opcode == WL_POINTER_LEAVE) {
			// [Event] wl_pointer::leave(uo).
			uint32_t surface = body[1];
			WaylandSurfaceData *surface_data = (WaylandSurfaceData *)get_object(surface)->data;
			ERR_FAIL_NULL_V(surface_data, MessageStatus::ERROR);

			if (global_seat_data->pointed_surface_id == surface) {
				DEBUG_LOG_WAYLAND_EMBED(vformat("Pointer (g0x%x seat g0x%x): g0x%x -> g0x%x", p_global_id, data->wl_seat_id, global_seat_data->pointed_surface_id, INVALID_ID));
				global_seat_data->pointed_surface_id = INVALID_ID;
			}
		}

		return MessageStatus::UNHANDLED;
	}

	if (object->interface == &xdg_popup_interface) {
		if (p_opcode == XDG_POPUP_CONFIGURE) {
			// [Event] xdg_popup::configure(iiii);
			int32_t x = body[0];
			int32_t y = body[1];
			int32_t width = body[2];
			int32_t height = body[3];

			XdgPopupData *data = (XdgPopupData *)object->data;
			ERR_FAIL_NULL_V(data, MessageStatus::ERROR);

			XdgSurfaceData *parent_xdg_surf_data = (XdgSurfaceData *)data->parent_handle.get()->data;
			ERR_FAIL_NULL_V(parent_xdg_surf_data, MessageStatus::ERROR);

			WaylandSurfaceData *parent_surface_data = (WaylandSurfaceData *)get_object(parent_xdg_surf_data->wl_surface_id)->data;
			ERR_FAIL_NULL_V(parent_surface_data, MessageStatus::ERROR);

			WaylandObject *parent_role_obj = parent_surface_data->role_object_handle.get();
			ERR_FAIL_NULL_V(parent_role_obj, MessageStatus::ERROR);

			if (parent_role_obj->interface == &xdg_toplevel_interface) {
				XdgToplevelData *parent_toplevel_data = (XdgToplevelData *)parent_role_obj->data;
				ERR_FAIL_NULL_V(parent_toplevel_data, MessageStatus::ERROR);

				if (parent_toplevel_data->is_embedded()) {
					WaylandSubsurfaceData *subsurf_data = (WaylandSubsurfaceData *)get_object(parent_toplevel_data->wl_subsurface_id)->data;
					ERR_FAIL_NULL_V(subsurf_data, MessageStatus::ERROR);

					// The coordinates passed will be shifted by the embedded window position,
					// so we need to fix them back.
					Point2i fixed_position = Point2i(x, y) - subsurf_data->position;

					DEBUG_LOG_WAYLAND_EMBED(vformat("Correcting popup configure position to %s", fixed_position));

					send_wayland_message(client->socket, local_id, p_opcode, { (uint32_t)fixed_position.x, (uint32_t)fixed_position.y, (uint32_t)width, (uint32_t)height });

					return MessageStatus::HANDLED;
				}
			}
		}
	}

	return MessageStatus::UNHANDLED;
}

void WaylandEmbedder::shutdown() {
	thread_done.set();

	{
		// First making a list of all clients so that we can iteratively delete them.
		LocalVector<int> sockets;
		for (KeyValue<int, Client> &pair : clients) {
			sockets.push_back(pair.key);
		}

		for (int socket : sockets) {
			cleanup_socket(socket);
		}
	}

	close(compositor_socket);
	compositor_socket = -1;

	for (KeyValue<uint32_t, RegistryGlobalInfo> &pair : registry_globals) {
		RegistryGlobalInfo &info = pair.value;
		if (info.data) {
			memdelete(info.data);
			info.data = nullptr;
		}
	}
}

Error WaylandEmbedder::handle_msg_info(Client *client, const struct msg_info *info, uint32_t *buf, int *fds_requested) {
	ERR_FAIL_NULL_V(info, ERR_BUG);
	ERR_FAIL_NULL_V(fds_requested, ERR_BUG);
	ERR_FAIL_NULL_V_MSG(info->direction == ProxyDirection::COMPOSITOR && client, ERR_BUG, "Wait, where did this message come from?");

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

	if (object == nullptr) {
		if (info->direction == ProxyDirection::COMPOSITOR) {
			uint32_t local_id = info->raw_id;
			ERR_PRINT(vformat("Couldn't find requested object l0x%x for client %d, disconnecting.", local_id, client->socket));

			socket_error(client->socket, local_id, WL_DISPLAY_ERROR_INVALID_OBJECT, vformat("Object l0x%x not found.", local_id));
			return OK;
		} else {
			CRASH_NOW_MSG(vformat("No object found for r0x%x", info->raw_id));
		}
	}

	const struct wl_interface *interface = nullptr;
	interface = object->interface;

	if (interface == nullptr && info->raw_id & 0xff000000) {
		// Regular clients have no confirmation about deleted server objects (why
		// should they?) but since we share connections there's the risk of receiving
		// messages about deleted server objects. The simplest solution is to ignore
		// unknown server-side objects. Not the safest thing, I know, but it should do
		// the job.
		DEBUG_LOG_WAYLAND_EMBED(vformat("Ignoring unknown server-side object r0x%x", info->raw_id));
		return OK;
	}

	ERR_FAIL_NULL_V_MSG(interface, ERR_BUG, vformat("Object r0x%x has no interface", info->raw_id));

	const struct wl_message *message = nullptr;
	if (info->direction == ProxyDirection::CLIENT) {
		ERR_FAIL_COND_V(info->opcode >= interface->event_count, ERR_BUG);
		message = &interface->events[info->opcode];
	} else {
		ERR_FAIL_COND_V(info->opcode >= interface->method_count, ERR_BUG);
		message = &interface->methods[info->opcode];
	}
	ERR_FAIL_NULL_V(message, ERR_BUG);

	*fds_requested = String(message->signature).count("h");
	LocalVector<int> sent_fds;

	if (*fds_requested > 0) {
		DEBUG_LOG_WAYLAND_EMBED(vformat("Requested %d FDs.", *fds_requested));

		List<int> &fd_queue = info->direction == ProxyDirection::COMPOSITOR ? client->fds : compositor_fds;
		for (int i = 0; i < *fds_requested; ++i) {
			ERR_FAIL_COND_V_MSG(fd_queue.is_empty(), ERR_BUG, "Out of FDs.");
			DEBUG_LOG_WAYLAND_EMBED(vformat("Fetching FD %d.", fd_queue.front()->get()));
			sent_fds.push_back(fd_queue.front()->get());
			fd_queue.pop_front();
		}

		DEBUG_LOG_WAYLAND_EMBED(vformat("Remaining FDs: %d.", fd_queue.size()));
	}

	if (object->destroyed) {
		DEBUG_LOG_WAYLAND_EMBED("Ignoring message for inert object.");
		// Inert object.
		return OK;
	}

	if (info->direction == ProxyDirection::COMPOSITOR) {
		MessageStatus request_status = handle_request(LocalObjectHandle(client, info->raw_id), info->opcode, buf, info->size);
		if (request_status == MessageStatus::ERROR) {
			return ERR_BUG;
		}

		if (request_status == MessageStatus::HANDLED) {
			DEBUG_LOG_WAYLAND_EMBED("Custom handler success.");
			return OK;
		}

		if (global_id != INVALID_ID) {
			buf[0] = global_id;
		}

		DEBUG_LOG_WAYLAND_EMBED("Falling back to generic handler.");

		if (handle_generic_msg(client, object, message, info, buf)) {
			send_raw_message(compositor_socket, { { buf, info->size } }, sent_fds);
		}
	} else {
		uint32_t global_name = 0;

		bool is_global = false;
		if (registry_globals_names.has(global_id)) {
			global_name = registry_globals_names[global_id];
			is_global = true;
		}

		// FIXME: For compatibility, mirror events with instanced registry globals as
		// object arguments. For example, `wl_surface.enter` returns a `wl_output`. If
		// said `wl_output` has been instanced multiple times, we need to resend the
		// same event with each instance as the argument, or the client might miss the
		// event by looking for the "wrong" instance.
		//
		// Note that this missing behavior is exclusively a compatibility mechanism
		// for old compositors which only implement undestroyable globals. We
		// otherwise passthrough every bind request and then the compositor takes care
		// of everything.
		// See: https://lore.freedesktop.org/wayland-devel/20190326121421.06732fd2@eldfell.localdomain/
		if (object->shared) {
			bool handled = false;

			for (KeyValue<int, Client> &pair : clients) {
				Client &c = pair.value;
				if (c.socket < 0) {
					continue;
				}

				if (!c.local_ids.has(global_id)) {
					DEBUG_LOG_WAYLAND_EMBED("!!!!!!!!!!! Instance missing?");
					continue;
				}

				if (is_global) {
					if (!c.registry_globals_instances.has(global_name)) {
						continue;
					}

					DEBUG_LOG_WAYLAND_EMBED(vformat("Broadcasting to all global instances for client %d (socket %d)", c.pid, c.socket));
					for (uint32_t instance_id : c.registry_globals_instances[global_name]) {
						DEBUG_LOG_WAYLAND_EMBED(vformat("Global instance l0x%x", instance_id));

						LocalObjectHandle local_obj = LocalObjectHandle(&c, instance_id);
						if (!local_obj.is_valid()) {
							continue;
						}

						MessageStatus event_status = handle_event(global_id, local_obj, info->opcode, buf, info->size);

						if (event_status == MessageStatus::ERROR) {
							return ERR_BUG;
						}

						if (event_status == MessageStatus::HANDLED) {
							DEBUG_LOG_WAYLAND_EMBED("Custom handler success.");
							handled = true;
							continue;
						}

						if (event_status == MessageStatus::INVALID) {
							continue;
						}

						DEBUG_LOG_WAYLAND_EMBED("Falling back to generic handler.");

						buf[0] = instance_id;

						if (handle_generic_msg(&c, local_obj.get(), message, info, buf, instance_id)) {
							send_raw_message(c.socket, { { buf, info->size } }, sent_fds);
						}

						handled = true;
					}
				} else if (interface == &wl_display_interface) {
					// NOTE: The only shared non-global objects are `wl_display` and
					// `wl_registry`, both of which require custom handlers. Additionally, of
					// those only `wl_display` has client-specific handlers, which is what this
					// branch manages.

					LocalObjectHandle local_obj = LocalObjectHandle(&c, c.get_local_id(global_id));
					if (!local_obj.is_valid()) {
						continue;
					}

					DEBUG_LOG_WAYLAND_EMBED(vformat("Shared non-global l0x%x g0x%x", c.get_local_id(global_id), global_id));

					MessageStatus event_status = handle_event(global_id, local_obj, info->opcode, buf, info->size);
					if (event_status == MessageStatus::ERROR) {
						return ERR_BUG;
					}

					if (event_status == MessageStatus::HANDLED) {
						DEBUG_LOG_WAYLAND_EMBED("Custom handler success.");
						handled = true;
						continue;
					}

					if (event_status == MessageStatus::INVALID) {
						continue;
					}

					DEBUG_LOG_WAYLAND_EMBED("Falling back to generic handler.");

					if (handle_generic_msg(&c, local_obj.get(), message, info, buf)) {
						send_raw_message(c.socket, { { buf, info->size } }, sent_fds);
					}

					handled = true;
				}
			}

			if (!handled) {
				// No client handled this, it's going to be handled as a client-less event.
				// We do this only at the end to avoid handling certain events (e.g.
				// deletion) twice.
				handle_event(global_id, LocalObjectHandle(nullptr, INVALID_ID), info->opcode, buf, info->size);
			}
		} else {
			LocalObjectHandle local_obj = LocalObjectHandle(client, client ? client->get_local_id(global_id) : INVALID_ID);

			MessageStatus event_status = handle_event(global_id, local_obj, info->opcode, buf, info->size);
			if (event_status == MessageStatus::ERROR) {
				return ERR_BUG;
			}

			if (event_status == MessageStatus::HANDLED || event_status == MessageStatus::INVALID) {
				// We're done.
				return OK;
			}

			// Generic passthrough.

			if (client) {
				uint32_t local_id = client->get_local_id(global_id);
				ERR_FAIL_COND_V(local_id == INVALID_ID, OK);

				DEBUG_LOG_WAYLAND_EMBED(vformat("%s::%s(%s) g0x%x -> l0x%x", interface->name, message->name, message->signature, global_id, local_id));
				buf[0] = local_id;

				if (handle_generic_msg(client, local_obj.get(), message, info, buf)) {
					send_raw_message(client->socket, { { buf, info->size } }, sent_fds);
				}
			} else {
				WARN_PRINT_ONCE(vformat("[Wayland Embedder] Unexpected client-less event from %s#g0x%x. Object has probably leaked.", object->interface->name, global_id));
				handle_generic_msg(nullptr, object, message, info, buf);
			}
		}
	}

	for (int fd : sent_fds) {
		DEBUG_LOG_WAYLAND_EMBED(vformat("Closing fd %d.", fd));
		close(fd);
	}

	return OK;
}

Error WaylandEmbedder::handle_sock(int p_fd) {
	ERR_FAIL_COND_V(p_fd < 0, ERR_INVALID_PARAMETER);

	struct msg_info info = {};

	{
		struct msghdr head_msg = {};
		uint32_t header[2];
		struct iovec vec = { header, sizeof header };

		head_msg.msg_iov = &vec;
		head_msg.msg_iovlen = 1;

		ssize_t head_rec = recvmsg(p_fd, &head_msg, MSG_PEEK);

		if (head_rec == 0) {
			// Client disconnected.
			return ERR_CONNECTION_ERROR;
		}

		if (head_rec == -1) {
			if (errno == ECONNRESET) {
				// No need to print the error, the client forcefully disconnected, that's
				// fine.
				return ERR_CONNECTION_ERROR;
			}

			ERR_FAIL_V_MSG(FAILED, vformat("Can't read message header: %s", strerror(errno)));
		}

		ERR_FAIL_COND_V_MSG(((size_t)head_rec) != vec.iov_len, ERR_CONNECTION_ERROR, vformat("Should've received %d bytes, instead got %d bytes", vec.iov_len, head_rec));

		// Header is two 32-bit words: first is ID, second has size in most significant
		// half and opcode in the other half.
		info.raw_id = header[0];
		info.size = header[1] >> 16;
		info.opcode = header[1] & 0xFFFF;
		info.direction = p_fd != compositor_socket ? ProxyDirection::COMPOSITOR : ProxyDirection::CLIENT;
	}

	if (msg_buf.size() < info.words()) {
		msg_buf.resize(info.words());
	}

	ERR_FAIL_COND_V_MSG(info.size % WL_WORD_SIZE != 0, ERR_CONNECTION_ERROR, "Invalid message length.");

	struct msghdr full_msg = {};
	struct iovec vec = { msg_buf.ptr(), info.size };
	{
		full_msg.msg_iov = &vec;
		full_msg.msg_iovlen = 1;
		full_msg.msg_control = ancillary_buf.ptr();
		full_msg.msg_controllen = ancillary_buf.size();

		ssize_t full_rec = recvmsg(p_fd, &full_msg, 0);

		if (full_rec == -1) {
			if (errno == ECONNRESET) {
				// No need to print the error, the client forcefully disconnected, that's
				// fine.
				return ERR_CONNECTION_ERROR;
			}

			ERR_FAIL_V_MSG(FAILED, vformat("Can't read message: %s", strerror(errno)));
		}

		ERR_FAIL_COND_V_MSG(((size_t)full_rec) != info.size, ERR_CONNECTION_ERROR, "Invalid message length.");

		DEBUG_LOG_WAYLAND_EMBED(" === START PACKET === ");

#ifdef WAYLAND_EMBED_DEBUG_LOGS_ENABLED
		printf("[PROXY] Received bytes: ");
		for (ssize_t i = 0; i < full_rec; ++i) {
			printf("%.2x", ((const uint8_t *)msg_buf.ptr())[i]);
		}
		printf("\n");
#endif
	}

	if (full_msg.msg_controllen > 0) {
		struct cmsghdr *cmsg = CMSG_FIRSTHDR(&full_msg);
		while (cmsg) {
			// TODO: Check for validity of message fields.
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
						clients[p_fd].fds.push_back(fd);
					} else {
						compositor_fds.push_back(fd);
					}
				}

#ifdef WAYLAND_EMBED_DEBUG_LOGS_ENABLED
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

	Client *client = nullptr;
	if (p_fd == compositor_socket) {
		// Let's figure out the recipient of the message.
		for (KeyValue<int, Client> &pair : clients) {
			Client &c = pair.value;

			if (c.local_ids.has(info.raw_id)) {
				client = &c;
			}
		}
	} else {
		CRASH_COND(!clients.has(p_fd));
		client = &clients[p_fd];
	}

	if (handle_msg_info(client, &info, msg_buf.ptr(), &fds_requested) != OK) {
		return ERR_BUG;
	}

	DEBUG_LOG_WAYLAND_EMBED(" === END PACKET === ");

	return OK;
}

void WaylandEmbedder::_thread_loop(void *p_data) {
	Thread::set_name("Wayland Embed");

	ERR_FAIL_NULL(p_data);
	WaylandEmbedder *proxy = (WaylandEmbedder *)p_data;

	DEBUG_LOG_WAYLAND_EMBED("Proxy thread started");

	while (!proxy->thread_done.is_set()) {
		proxy->poll_sockets();
	}
}

Error WaylandEmbedder::init() {
	ancillary_buf.resize(EMBED_ANCILLARY_BUF_SIZE);

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

	pollfds.push_back({ proxy_socket, POLLIN, 0 });
	pollfds.push_back({ compositor_socket, POLLIN, 0 });

	RegistryGlobalInfo control_global_info = {};
	control_global_info.interface = &godot_embedding_compositor_interface;
	control_global_info.version = godot_embedding_compositor_interface.version;

	godot_embedding_compositor_name = registry_globals_counter++;
	registry_globals[godot_embedding_compositor_name] = control_global_info;

	{
		uint32_t invalid_id = INVALID_ID;
		objects.request(invalid_id);

		CRASH_COND(invalid_id != INVALID_ID);
	}

	{
		uint32_t display_id = new_object(&wl_display_interface);
		CRASH_COND(display_id != DISPLAY_ID);

		get_object(DISPLAY_ID)->shared = true;
	}

	{
		uint32_t registry_id = new_object(&wl_registry_interface);
		CRASH_COND(registry_id != REGISTRY_ID);

		get_object(REGISTRY_ID)->shared = true;
	}

	// wl_display::get_registry(n)
	send_wayland_message(compositor_socket, DISPLAY_ID, 1, { REGISTRY_ID });

	sync();

	proxy_thread.start(_thread_loop, this);

	return OK;
}

void WaylandEmbedder::handle_fd(int p_fd, int p_revents) {
	if (p_fd == proxy_socket && p_revents & POLLIN) {
		// Client init.
		int new_fd = accept(proxy_socket, nullptr, nullptr);
		ERR_FAIL_COND_MSG(new_fd == -1, "Failed to accept client.");

		struct ucred cred = {};
		socklen_t cred_size = sizeof cred;
		getsockopt(new_fd, SOL_SOCKET, SO_PEERCRED, &cred, &cred_size);

		Client &client = clients.insert_new(new_fd, {})->value;

		client.embedder = this;
		client.socket = new_fd;
		client.pid = cred.pid;

		client.global_ids[DISPLAY_ID] = Client::GlobalIdInfo(DISPLAY_ID, nullptr);
		client.local_ids[DISPLAY_ID] = DISPLAY_ID;

		pollfds.push_back({ new_fd, POLLIN, 0 });

		if (main_client == nullptr) {
			main_client = &client;
		}

		if (new_fd != main_client->socket && main_client->registry_globals_instances.has(godot_embedding_compositor_name)) {
			uint32_t new_local_id = main_client->allocate_server_id();

			client.embedded_client_id = new_local_id;

			for (uint32_t local_id : main_client->registry_globals_instances[godot_embedding_compositor_name]) {
				EmbeddedClientData *eclient_data = memnew(EmbeddedClientData);
				eclient_data->client = &client;

				main_client->new_fake_object(new_local_id, &godot_embedded_client_interface, 1, eclient_data);

				// godot_embedding_compositor::client(nu)
				send_wayland_message(main_client->socket, local_id, 0, { new_local_id, (uint32_t)cred.pid });
			}
		}

		DEBUG_LOG_WAYLAND_EMBED(vformat("New client %d (pid %d) initialized.", client.socket, cred.pid));
		return;
	}

	if (p_fd == compositor_socket && p_revents & POLLIN) {
		Error err = handle_sock(p_fd);

		if (err == ERR_BUG) {
			ERR_PRINT("Unexpected error while handling socket, shutting down.");
			shutdown();
			return;
		}

		return;
	}

	const Client *client = clients.getptr(p_fd);
	if (client) {
		if (main_client && client == main_client && p_revents & (POLLHUP | POLLERR)) {
			DEBUG_LOG_WAYLAND_EMBED("Main client disconnected, shutting down.");
			shutdown();
			return;
		}

		if (p_revents & POLLIN) {
			Error err = handle_sock(p_fd);
			if (err == ERR_BUG) {
				ERR_PRINT("Unexpected error while handling socket, shutting down.");
				shutdown();
				return;
			}

			if (err != OK) {
				DEBUG_LOG_WAYLAND_EMBED("disconnecting");
				cleanup_socket(p_fd);
				return;
			}

			return;
		} else if (p_revents & (POLLHUP | POLLERR | POLLNVAL)) {
			if (p_revents & POLLHUP) {
				DEBUG_LOG_WAYLAND_EMBED(vformat("Socket %d hangup.", p_fd));
			}
			if (p_revents & POLLERR) {
				DEBUG_LOG_WAYLAND_EMBED(vformat("Socket %d error.", p_fd));
			}
			if (p_revents & POLLNVAL) {
				DEBUG_LOG_WAYLAND_EMBED(vformat("Socket %d invalid FD.", p_fd));
			}

			cleanup_socket(p_fd);

			return;
		}
	}
}

WaylandEmbedder::~WaylandEmbedder() {
	shutdown();
	if (proxy_thread.is_started()) {
		proxy_thread.wait_to_finish();
	}
}

#endif // TOOLS_ENABLED

#endif // WAYLAND_ENABLED
