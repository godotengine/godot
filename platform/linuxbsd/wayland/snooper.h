/**************************************************************************/
/*  snooper.h                                                             */
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

#include "core/error/error_list.h"
#include "core/error/error_macros.h"
#include "core/os/os.h"
#include "core/templates/a_hash_map.h"
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"
#include "core/templates/list.h"

#ifdef SOWRAP_ENABLED
#include "wayland/dynwrappers/wayland-client-core-so_wrap.h"
#else
#include <wayland-client-core.h>
#endif

#include "protocol/wayland.gen.h"

#include "protocol/linux_dmabuf_v1.gen.h"
#include "protocol/xdg_shell.gen.h"

#include "protocol/cursor_shape.gen.h"
#include "protocol/fifo_v1.gen.h"
#include "protocol/fractional_scale.gen.h"
#include "protocol/godot_embedding_compositor.gen.h"
#include "protocol/idle_inhibit.gen.h"
#include "protocol/linux_drm_syncobj_v1.gen.h"
#include "protocol/linux_explicit_synchronization_unstable_v1.gen.h"
#include "protocol/pointer_constraints.gen.h"
#include "protocol/pointer_gestures.gen.h"
#include "protocol/primary_selection.gen.h"
#include "protocol/relative_pointer.gen.h"
#include "protocol/tablet.gen.h"
#include "protocol/text_input.gen.h"
#include "protocol/viewporter.gen.h"
#include "protocol/wayland-drm.gen.h"
#include "protocol/xdg_activation.gen.h"
#include "protocol/xdg_decoration.gen.h"
#include "protocol/xdg_foreign_v1.gen.h"
#include "protocol/xdg_foreign_v2.gen.h"
#include "protocol/xdg_shell.gen.h"
#include "protocol/xdg_system_bell.gen.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>

#include <poll.h>

#include "core/io/dir_access.h"
#include "core/os/thread.h"

// FIXME: I think we might need to resize this dynamically, to a somewhat
// arbitrary limit (if any).
#define SNOOP_ANCILLARY_SIZE 4096

// DEBUG: This value is very small on purpose to discover ID leaks quickly. When
// deploying in production, set this to UINT32_MAX.
#define SNOOP_ID_MAX 250

class WaylandEmbedderProxy {
private:
	enum class ProxyDirection {
		CLIENT,
		COMPOSITOR,
	};

	struct msg_info {
		uint32_t raw_id = 0;
		uint16_t size = 0;
		uint16_t opcode = 0;

		pid_t pid = 0;

		ProxyDirection direction = ProxyDirection::CLIENT;

		constexpr size_t words() const { return (size / sizeof(uint32_t)); }
	};

	struct WaylandObjectData {
		virtual ~WaylandObjectData() = default;
	};

	struct WaylandObject {
		const struct wl_interface *interface = nullptr;
		int version = 0;
		WaylandObjectData *data = nullptr;
	};

	struct WaylandDrmGlobalData : WaylandObjectData {
		String device;
		LocalVector<uint32_t> formats;
		bool authenticated;
		uint32_t capabilities;
	};

	struct WaylandShmGlobalData : WaylandObjectData {
		LocalVector<uint32_t> formats;
	};

	struct Client {
		WaylandEmbedderProxy *snooper = nullptr;

		int socket = -1;
		pid_t pid = 0;

		// FIXME: Names suck.
		AHashMap<uint32_t, HashSet<uint32_t>> registry_globals_instances;

		uint32_t embedded_client_id = 0;

		AHashMap<uint32_t, uint32_t> global_ids;
		AHashMap<uint32_t, uint32_t> local_ids;

		// Objects with no equivalent on the real compositor.
		AHashMap<uint32_t, WaylandObject> fake_objects;

		// Objects which mirror events of a global object.
		AHashMap<uint32_t, WaylandObject> global_instances;

		uint32_t embedded_window_id = 0;

		List<int> fds;

		LocalVector<bool> server_id_map;

		uint32_t get_global_id(uint32_t p_local_id) const { return global_ids.has(p_local_id) ? global_ids[p_local_id] : INVALID_ID; }
		uint32_t get_local_id(uint32_t p_global_id) const { return local_ids.has(p_global_id) ? local_ids[p_global_id] : INVALID_ID; }

		uint32_t allocate_server_id();
		WaylandObject *get_object(uint32_t p_local_id);
		Error delete_object(uint32_t p_local_id);

		uint32_t new_object(uint32_t p_local_id, const struct wl_interface *p_interface, int p_version, WaylandObjectData *p_data = nullptr);
		WaylandObject *new_fake_object(uint32_t p_local_id, const struct wl_interface *p_interface, int p_version, WaylandObjectData *p_data = nullptr);
		WaylandObject *new_global_instance(uint32_t p_local_id, const struct wl_interface *p_interface, int p_version, WaylandObjectData *p_data = nullptr);

		void send_wl_drm_state(uint32_t p_id, WaylandDrmGlobalData *p_state);
	};

	// Local IDs are a mess to handle as they strictly depend on their client of
	// origin. This wrapper helps with that.
	class LocalObjectHandle {
		Client *client = nullptr;
		uint32_t local_id = INVALID_ID;

	public:
		constexpr LocalObjectHandle() = default;

		constexpr LocalObjectHandle(Client *p_client, uint32_t p_id) :
				client(p_client), local_id(p_id) {}

		void invalidate() {
			client = nullptr;
			local_id = INVALID_ID;
		}
		constexpr bool is_valid() const { return client != nullptr && local_id != INVALID_ID; }

		WaylandObject *get() { return is_valid() ? client->get_object(local_id) : nullptr; }
		constexpr Client *get_client() const { return client; }
		constexpr uint32_t get_local_id() const { return local_id; }
		uint32_t get_global_id() const { return (is_valid() && client->global_ids.has(local_id)) ? client->global_ids[local_id] : INVALID_ID; }
	};

	struct WaylandSeatInstanceData : WaylandObjectData {
		uint32_t wl_keyboard_id = INVALID_ID;
		uint32_t wl_pointer_id = INVALID_ID;
	};

	struct WaylandSeatGlobalData : WaylandObjectData {
		uint32_t capabilities = 0;

		uint32_t pointed_surface_id = INVALID_ID;
		uint32_t focused_surface_id = INVALID_ID;
	};

	struct WaylandKeyboardData : WaylandObjectData {
		uint32_t wl_seat_id = INVALID_ID;
	};

	struct WaylandPointerData : WaylandObjectData {
		uint32_t wl_seat_id = INVALID_ID;
	};

	struct WaylandSurfaceData : WaylandObjectData {
		Client *client = nullptr;
		LocalObjectHandle role_object_handle;
	};

	struct XdgSurfaceData : WaylandObjectData {
		uint32_t wl_surface_id = INVALID_ID;
	};

	struct WaylandSubsurfaceData : WaylandObjectData {
		Point2i position;
	};

	struct XdgToplevelData : WaylandObjectData {
		LocalObjectHandle xdg_surface_handle;
		LocalObjectHandle parent_handle;
		uint32_t wl_subsurface_id = INVALID_ID;

		Size2i size;

		bool configured = false;

		constexpr bool is_embedded() const { return wl_subsurface_id != INVALID_ID; }
	};

	struct XdgPositionerData : WaylandObjectData {
		Rect2i anchor_rect;
	};

	struct EmbeddedClientData : WaylandObjectData {
		Client *client = nullptr;
		bool disconnected = false;
	};

	struct RegistryGlobalInfo {
		const struct wl_interface *interface = nullptr;
		uint32_t version = 0;
		uint32_t compositor_name = 0;

		// FIXME: Better name. This is a mapping between versions and their global
		// objects, to be instanced.
		HashMap<uint32_t, uint32_t> global_ids;

		void *data = nullptr;
	};

	const static constexpr struct wl_interface *interfaces[] = {
		// wayland
		&wl_buffer_interface,
		&wl_callback_interface,
		&wl_compositor_interface,
		&wl_data_device_interface,
		&wl_data_device_manager_interface,
		&wl_data_offer_interface,
		&wl_data_source_interface,
		&wl_display_interface,
		&wl_keyboard_interface,
		&wl_output_interface,
		&wl_pointer_interface,
		&wl_region_interface,
		&wl_registry_interface,
		&wl_seat_interface,
		//&wl_shell_interface, // Deprecated.
		//&wl_shell_surface_interface, // Deprecated.
		&wl_shm_interface,
		&wl_shm_pool_interface,
		&wl_subcompositor_interface,
		&wl_subsurface_interface,
		&wl_surface_interface,
		//&wl_touch_interface, // Unused (at the moment).

		// xdg-shell
		&xdg_wm_base_interface,
		&xdg_positioner_interface,
		&xdg_surface_interface,
		&xdg_toplevel_interface,
		&xdg_popup_interface,

		// linux-dmabuf-v1
		&zwp_linux_dmabuf_v1_interface,
		&zwp_linux_buffer_params_v1_interface,
		&zwp_linux_dmabuf_feedback_v1_interface,

		// linux-explicit-synchronization-unstable-v1
		&zwp_linux_explicit_synchronization_v1_interface,
		&zwp_linux_surface_synchronization_v1_interface,
		&zwp_linux_buffer_release_v1_interface,

		// fractional-scale
		&wp_fractional_scale_manager_v1_interface,
		&wp_fractional_scale_v1_interface,

		// idle-inhibit
		&zwp_idle_inhibit_manager_v1_interface,
		&zwp_idle_inhibitor_v1_interface,

		// pointer-constraints
		&zwp_pointer_constraints_v1_interface,
		&zwp_locked_pointer_v1_interface,
		&zwp_confined_pointer_v1_interface,

		// pointer-gestures
		&zwp_pointer_gestures_v1_interface,
		&zwp_pointer_gesture_swipe_v1_interface,
		&zwp_pointer_gesture_pinch_v1_interface,
		&zwp_pointer_gesture_hold_v1_interface,

		// primary-selection
		&zwp_primary_selection_device_manager_v1_interface,
		&zwp_primary_selection_device_v1_interface,
		&zwp_primary_selection_offer_v1_interface,
		&zwp_primary_selection_source_v1_interface,

		// relative-pointer
		&zwp_relative_pointer_manager_v1_interface,
		&zwp_relative_pointer_v1_interface,

		// tablet
		&zwp_tablet_manager_v2_interface,
		&zwp_tablet_seat_v2_interface,
		&zwp_tablet_tool_v2_interface,
		&zwp_tablet_v2_interface,
		&zwp_tablet_pad_ring_v2_interface,
		&zwp_tablet_pad_strip_v2_interface,
		&zwp_tablet_pad_group_v2_interface,
		&zwp_tablet_pad_v2_interface,

		// text-input
		&zwp_text_input_v3_interface,
		&zwp_text_input_manager_v3_interface,

		// viewporter
		&wp_viewporter_interface,
		&wp_viewport_interface,

		// xdg-activation
		&xdg_activation_v1_interface,
		&xdg_activation_token_v1_interface,

		// xdg-decoration
		&zxdg_decoration_manager_v1_interface,
		&zxdg_toplevel_decoration_v1_interface,

		// xdg-foreign
		&zxdg_exporter_v1_interface,
		&zxdg_importer_v1_interface,

		// xdg-foreign-v1
		&zxdg_exporter_v1_interface,
		&zxdg_importer_v1_interface,

		// xdg-foreign-v2
		&zxdg_exporter_v2_interface,
		&zxdg_importer_v2_interface,

		// xdg-shell
		&xdg_wm_base_interface,
		&xdg_positioner_interface,
		&xdg_surface_interface,
		&xdg_toplevel_interface,
		&xdg_popup_interface,

		// xdg-system-bell
		&xdg_system_bell_v1_interface,

		// wp-cursor-shape-v1
		&wp_cursor_shape_manager_v1_interface,

		// wayland-drm
		&wl_drm_interface,

		// linux-drm-syncobj-v1
		&wp_linux_drm_syncobj_manager_v1_interface,
		&wp_linux_drm_syncobj_surface_v1_interface,
		&wp_linux_drm_syncobj_timeline_v1_interface,

		// fifo-v1
		&wp_fifo_v1_interface,

		// Our custom things.
		&godot_embedding_compositor_interface,
		&godot_embedded_client_interface,
	};

	static constexpr uint32_t INVALID_ID = 0;
	static constexpr uint32_t DISPLAY_ID = 1;
	static constexpr uint32_t REGISTRY_ID = 2;

	int proxy_socket = -1;
	int compositor_socket = -1;

	LocalVector<struct pollfd> pollfds;
	LocalVector<Client> clients;

	LocalVector<WaylandObject> objects;
	// Proxies allocated by the compositor. Their ID starts from 0xff000000.
	LocalVector<WaylandObject> server_objects;

	uint32_t wl_compositor_id = 0;
	uint32_t wl_subcompositor_id = 0;
	uint32_t main_toplevel_id = 0;
	uint32_t xdg_wm_base_id = 0;

	// FIXME: Add all shareable undestroyable objects (depending on version) to
	// this map. They aren't many, most core stuff from old versions.
	HashMap<const struct wl_interface *, uint32_t> shared_objects;

	// Global id to name
	HashMap<uint32_t, uint32_t> registry_globals_names;

	// FIXME: Ensure that no duplicate entries get added (VSet?)
	LocalVector<RegistryGlobalInfo> registry_globals;
	uint32_t godot_embedding_compositor_name = 0;
	uint32_t wl_seat_name = 0;

	Thread proxy_thread;

	List<int> client_fds;
	List<int> compositor_fds;

	uint32_t serial_counter = 0;

	uint32_t sync_callback_id = 0;

	Ref<DirAccess> runtime_dir;
	int lock_fd = -1;
	String socket_path;
	String socket_lock_path;

	LocalVector<uint32_t> msg_buf;
	LocalVector<uint8_t> ancillary_buf;

	static size_t wl_array_word_offset(uint32_t p_size);
	const static struct wl_interface *wl_interface_from_string(const char *name, size_t size);

	static void send_raw_message(int p_socket, std::initializer_list<struct iovec> p_vecs, const LocalVector<int> &p_fds = LocalVector<int>());

	static void send_wayland_message(int p_socket, uint32_t p_id, uint32_t p_opcode, const uint32_t *p_args, const size_t p_args_words);
	static void send_wayland_message(int p_socket, uint32_t p_id, uint32_t p_opcode, std::initializer_list<uint32_t> p_args);

	uint32_t new_object(const struct wl_interface *p_interface, int p_version, WaylandObjectData *p_data = nullptr);

	void poll_sockets();

	int next_global_id();
	int next_global_server_id();

	bool global_surface_is_window(uint32_t p_global_surface_id);

	WaylandObject *get_object(uint32_t id);
	Error delete_object(uint32_t id);

	void client_disconnect(size_t p_client_id);

	void sync();

	void seat_name_enter_surface(uint32_t p_seat_name, uint32_t p_global_surface_id);
	void seat_name_leave_surface(uint32_t p_seat_name, uint32_t p_global_surface_id);

	bool handle_request(LocalObjectHandle p_object, uint32_t p_opcode, uint32_t *msg_data, size_t msg_len);
	bool handle_event(uint32_t p_global_id, LocalObjectHandle p_local_handle, uint32_t p_opcode, uint32_t *msg_data, size_t msg_len);

	bool handle_generic_msg(Client *client, const WaylandObject *p_object, const struct wl_message *message, const struct msg_info *info, uint32_t *buf, uint32_t instance_id = INVALID_ID);
	bool handle_msg_info(Client *client, const struct msg_info *info, uint32_t *buf, int *fds_requested);
	bool handle_sock(int p_fd, int p_id);
	void handle_fd(int p_fd, int p_revents);

	static void _thread_loop(void *p_data);

public:
	// Returns path to socket.
	Error init();
	void uninit();

	String get_socket_path() const { return socket_path; }
};
