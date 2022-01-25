/*************************************************************************/
/*  portal_gameplay_monitor.h                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef PORTAL_GAMEPLAY_MONITOR_H
#define PORTAL_GAMEPLAY_MONITOR_H

#include "core/local_vector.h"
#include "servers/visual_server_callbacks.h"

#include <stdint.h>

class PortalRenderer;
struct VSRoom;

class PortalGameplayMonitor {
public:
	PortalGameplayMonitor();

	void unload(PortalRenderer &p_portal_renderer);

	// entering and exiting gameplay notifications (requires PVS)
	void update_gameplay(PortalRenderer &p_portal_renderer, const int *p_source_room_ids, int p_num_source_rooms);
	void set_params(bool p_use_secondary_pvs, bool p_use_signals);

private:
	void _update_gameplay_room(PortalRenderer &p_portal_renderer, int p_room_id, bool p_source_rooms_changed);
	bool _source_rooms_changed(const int *p_source_room_ids, int p_num_source_rooms);
	void _swap(bool p_source_rooms_changed);

	// gameplay ticks happen every physics tick
	uint32_t _gameplay_tick = 1;

	// Room ticks only happen when the rooms the cameras are within change.
	// This is an optimization. This tick needs to be maintained separately from _gameplay_tick
	// because testing against the previous tick is used to determine whether to send enter or exit
	// gameplay notifications, and this must be synchronized differently for rooms, roomgroups and static ghosts.
	uint32_t _room_tick = 1;

	// we need two version, current and previous
	LocalVector<uint32_t, int32_t> _active_moving_pool_ids[2];
	LocalVector<uint32_t, int32_t> *_active_moving_pool_ids_curr;
	LocalVector<uint32_t, int32_t> *_active_moving_pool_ids_prev;

	LocalVector<uint32_t, int32_t> _active_rghost_pool_ids[2];
	LocalVector<uint32_t, int32_t> *_active_rghost_pool_ids_curr;
	LocalVector<uint32_t, int32_t> *_active_rghost_pool_ids_prev;

	LocalVector<uint32_t, int32_t> _active_room_ids[2];
	LocalVector<uint32_t, int32_t> *_active_room_ids_curr;
	LocalVector<uint32_t, int32_t> *_active_room_ids_prev;

	LocalVector<uint32_t, int32_t> _active_roomgroup_ids[2];
	LocalVector<uint32_t, int32_t> *_active_roomgroup_ids_curr;
	LocalVector<uint32_t, int32_t> *_active_roomgroup_ids_prev;

	LocalVector<uint32_t, int32_t> _active_sghost_ids[2];
	LocalVector<uint32_t, int32_t> *_active_sghost_ids_curr;
	LocalVector<uint32_t, int32_t> *_active_sghost_ids_prev;

	LocalVector<uint32_t, int32_t> _source_rooms_prev;

	VisualServerCallbacks::CallbackType _enter_callback_type = VisualServerCallbacks::CALLBACK_NOTIFICATION_ENTER_GAMEPLAY;
	VisualServerCallbacks::CallbackType _exit_callback_type = VisualServerCallbacks::CALLBACK_NOTIFICATION_EXIT_GAMEPLAY;

	bool _use_secondary_pvs = false;
	bool _use_signals = false;
};

#endif
