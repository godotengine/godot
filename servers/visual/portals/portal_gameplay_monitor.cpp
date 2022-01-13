/*************************************************************************/
/*  portal_gameplay_monitor.cpp                                          */
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

#include "portal_gameplay_monitor.h"

#include "portal_renderer.h"
#include "portal_types.h"
#include "servers/visual/visual_server_globals.h"
#include "servers/visual/visual_server_scene.h"

PortalGameplayMonitor::PortalGameplayMonitor() {
	_active_moving_pool_ids_prev = &_active_moving_pool_ids[0];
	_active_moving_pool_ids_curr = &_active_moving_pool_ids[1];

	_active_rghost_pool_ids_curr = &_active_rghost_pool_ids[0];
	_active_rghost_pool_ids_prev = &_active_rghost_pool_ids[1];

	_active_room_ids_prev = &_active_room_ids[0];
	_active_room_ids_curr = &_active_room_ids[1];

	_active_roomgroup_ids_prev = &_active_roomgroup_ids[0];
	_active_roomgroup_ids_curr = &_active_roomgroup_ids[1];

	_active_sghost_ids_prev = &_active_sghost_ids[0];
	_active_sghost_ids_curr = &_active_sghost_ids[1];
}

bool PortalGameplayMonitor::_source_rooms_changed(const int *p_source_room_ids, int p_num_source_rooms) {
	bool source_rooms_changed = false;
	if (p_num_source_rooms == _source_rooms_prev.size()) {
		for (int n = 0; n < p_num_source_rooms; n++) {
			if (p_source_room_ids[n] != (int)_source_rooms_prev[n]) {
				source_rooms_changed = true;
				break;
			}
		}
	} else {
		source_rooms_changed = true;
	}
	if (source_rooms_changed) {
		_source_rooms_prev.clear();
		for (int n = 0; n < p_num_source_rooms; n++) {
			_source_rooms_prev.push_back(p_source_room_ids[n]);
		}
	}

	return source_rooms_changed;
}

void PortalGameplayMonitor::set_params(bool p_use_secondary_pvs, bool p_use_signals) {
	_use_secondary_pvs = p_use_secondary_pvs;
	_use_signals = p_use_signals;

	if (_use_signals) {
		_enter_callback_type = VisualServerCallbacks::CALLBACK_SIGNAL_ENTER_GAMEPLAY;
		_exit_callback_type = VisualServerCallbacks::CALLBACK_SIGNAL_EXIT_GAMEPLAY;
	} else {
		_enter_callback_type = VisualServerCallbacks::CALLBACK_NOTIFICATION_ENTER_GAMEPLAY;
		_exit_callback_type = VisualServerCallbacks::CALLBACK_NOTIFICATION_EXIT_GAMEPLAY;
	}
}

// can work with 1 or multiple cameras
void PortalGameplayMonitor::update_gameplay(PortalRenderer &p_portal_renderer, const int *p_source_room_ids, int p_num_source_rooms) {
	const PVS &pvs = p_portal_renderer.get_pvs();

	_gameplay_tick++;

	// if there is no change in the source room IDs, then we can optimize out a lot of the checks
	// (anything not to do with roamers)
	bool source_rooms_changed = _source_rooms_changed(p_source_room_ids, p_num_source_rooms);

	// lock output
	VisualServerCallbacks *callbacks = VSG::scene->get_callbacks();
	callbacks->lock();

	for (int n = 0; n < p_num_source_rooms; n++) {
		const VSRoom &source_room = p_portal_renderer.get_room(p_source_room_ids[n]);

		if (_use_secondary_pvs) {
			int pvs_size = source_room._secondary_pvs_size;
			int pvs_first = source_room._secondary_pvs_first;

			for (int r = 0; r < pvs_size; r++) {
				int room_id = pvs.get_secondary_pvs_room_id(pvs_first + r);
				_update_gameplay_room(p_portal_renderer, room_id, source_rooms_changed);
			} // for r through the rooms hit in the pvs
		} else {
			int pvs_size = source_room._pvs_size;
			int pvs_first = source_room._pvs_first;

			for (int r = 0; r < pvs_size; r++) {
				int room_id = pvs.get_pvs_room_id(pvs_first + r);
				_update_gameplay_room(p_portal_renderer, room_id, source_rooms_changed);
			} // for r through the rooms hit in the pvs
		}
	} // for n through source rooms

	// find any moving that were active last tick that are no longer active, and send notifications
	for (int n = 0; n < _active_moving_pool_ids_prev->size(); n++) {
		int pool_id = (*_active_moving_pool_ids_prev)[n];
		PortalRenderer::Moving &moving = p_portal_renderer.get_pool_moving(pool_id);

		// gone out of view
		if (moving.last_gameplay_tick_hit != _gameplay_tick) {
			VisualServerCallbacks::Message msg;
			msg.object_id = VSG::scene->_instance_get_object_ID(moving.instance);
			msg.type = _exit_callback_type;

			callbacks->push_message(msg);
		}
	}

	// find any roaming ghosts that were active last tick that are no longer active, and send notifications
	for (int n = 0; n < _active_rghost_pool_ids_prev->size(); n++) {
		int pool_id = (*_active_rghost_pool_ids_prev)[n];
		PortalRenderer::RGhost &moving = p_portal_renderer.get_pool_rghost(pool_id);

		// gone out of view
		if (moving.last_gameplay_tick_hit != _gameplay_tick) {
			VisualServerCallbacks::Message msg;
			msg.object_id = moving.object_id;
			msg.type = VisualServerCallbacks::CALLBACK_NOTIFICATION_EXIT_GAMEPLAY;

			callbacks->push_message(msg);
		}
	}

	if (source_rooms_changed) {
		// find any rooms that were active last tick that are no longer active, and send notifications
		for (int n = 0; n < _active_room_ids_prev->size(); n++) {
			int room_id = (*_active_room_ids_prev)[n];
			const VSRoom &room = p_portal_renderer.get_room(room_id);

			// gone out of view
			if (room.last_gameplay_tick_hit != _gameplay_tick) {
				VisualServerCallbacks::Message msg;
				msg.object_id = room._godot_instance_ID;
				msg.type = _exit_callback_type;

				callbacks->push_message(msg);
			}
		}

		// find any roomgroups that were active last tick that are no longer active, and send notifications
		for (int n = 0; n < _active_roomgroup_ids_prev->size(); n++) {
			int roomgroup_id = (*_active_roomgroup_ids_prev)[n];
			const VSRoomGroup &roomgroup = p_portal_renderer.get_roomgroup(roomgroup_id);

			// gone out of view
			if (roomgroup.last_gameplay_tick_hit != _gameplay_tick) {
				VisualServerCallbacks::Message msg;
				msg.object_id = roomgroup._godot_instance_ID;
				msg.type = _exit_callback_type;

				callbacks->push_message(msg);
			}
		}

		// find any static ghosts that were active last tick that are no longer active, and send notifications
		for (int n = 0; n < _active_sghost_ids_prev->size(); n++) {
			int id = (*_active_sghost_ids_prev)[n];
			VSStaticGhost &ghost = p_portal_renderer.get_static_ghost(id);

			// gone out of view
			if (ghost.last_gameplay_tick_hit != _gameplay_tick) {
				VisualServerCallbacks::Message msg;
				msg.object_id = ghost.object_id;
				msg.type = VisualServerCallbacks::CALLBACK_NOTIFICATION_EXIT_GAMEPLAY;

				callbacks->push_message(msg);
			}
		}
	} // only need to check these if the source rooms changed

	// unlock
	callbacks->unlock();

	// swap the current and previous lists
	_swap();
}

void PortalGameplayMonitor::_update_gameplay_room(PortalRenderer &p_portal_renderer, int p_room_id, bool p_source_rooms_changed) {
	// get the room
	VSRoom &room = p_portal_renderer.get_room(p_room_id);

	int num_roamers = room._roamer_pool_ids.size();

	VisualServerCallbacks *callbacks = VSG::scene->get_callbacks();

	for (int n = 0; n < num_roamers; n++) {
		uint32_t pool_id = room._roamer_pool_ids[n];

		PortalRenderer::Moving &moving = p_portal_renderer.get_pool_moving(pool_id);

		// done already?
		if (moving.last_gameplay_tick_hit == _gameplay_tick)
			continue;

		// add to the active list
		_active_moving_pool_ids_curr->push_back(pool_id);

		// if wasn't present in the tick before, add the notification to enter
		if (moving.last_gameplay_tick_hit != (_gameplay_tick - 1)) {
			VisualServerCallbacks::Message msg;
			msg.object_id = VSG::scene->_instance_get_object_ID(moving.instance);
			msg.type = _enter_callback_type;

			callbacks->push_message(msg);
		}

		// mark as done
		moving.last_gameplay_tick_hit = _gameplay_tick;
	}

	// roaming ghosts
	int num_rghosts = room._rghost_pool_ids.size();

	for (int n = 0; n < num_rghosts; n++) {
		uint32_t pool_id = room._rghost_pool_ids[n];

		PortalRenderer::RGhost &moving = p_portal_renderer.get_pool_rghost(pool_id);

		// done already?
		if (moving.last_gameplay_tick_hit == _gameplay_tick)
			continue;

		// add to the active list
		_active_rghost_pool_ids_curr->push_back(pool_id);

		// if wasn't present in the tick before, add the notification to enter
		if (moving.last_gameplay_tick_hit != (_gameplay_tick - 1)) {
			VisualServerCallbacks::Message msg;
			msg.object_id = moving.object_id;
			msg.type = VisualServerCallbacks::CALLBACK_NOTIFICATION_ENTER_GAMEPLAY;

			callbacks->push_message(msg);
		}

		// mark as done
		moving.last_gameplay_tick_hit = _gameplay_tick;
	}

	// no need to progress from here
	if (!p_source_rooms_changed) {
		return;
	}

	// has the room come into gameplay?

	// later tests only relevant if a room has just come into play
	bool room_came_into_play = false;

	if (room.last_gameplay_tick_hit != _gameplay_tick) {
		room_came_into_play = true;

		// add the room to the active list
		_active_room_ids_curr->push_back(p_room_id);

		// if wasn't present in the tick before, add the notification to enter
		if (room.last_gameplay_tick_hit != (_gameplay_tick - 1)) {
			VisualServerCallbacks::Message msg;
			msg.object_id = room._godot_instance_ID;
			msg.type = _enter_callback_type;

			callbacks->push_message(msg);
		}

		// mark as done
		room.last_gameplay_tick_hit = _gameplay_tick;
	}

	// no need to do later tests
	if (!room_came_into_play) {
		return;
	}
	///////////////////////////////////////////////////////////////////

	// has the roomgroup come into gameplay?
	for (int n = 0; n < room._roomgroup_ids.size(); n++) {
		int roomgroup_id = room._roomgroup_ids[n];

		VSRoomGroup &roomgroup = p_portal_renderer.get_roomgroup(roomgroup_id);

		if (roomgroup.last_gameplay_tick_hit != _gameplay_tick) {
			// add the room to the active list
			_active_roomgroup_ids_curr->push_back(roomgroup_id);

			// if wasn't present in the tick before, add the notification to enter
			if (roomgroup.last_gameplay_tick_hit != (_gameplay_tick - 1)) {
				VisualServerCallbacks::Message msg;
				msg.object_id = roomgroup._godot_instance_ID;
				msg.type = _enter_callback_type;

				callbacks->push_message(msg);
			}

			// mark as done
			roomgroup.last_gameplay_tick_hit = _gameplay_tick;
		}
	} // for through roomgroups

	// static ghosts
	int num_sghosts = room._static_ghost_ids.size();

	for (int n = 0; n < num_sghosts; n++) {
		uint32_t id = room._static_ghost_ids[n];

		VSStaticGhost &ghost = p_portal_renderer.get_static_ghost(id);

		// done already?
		if (ghost.last_gameplay_tick_hit == _gameplay_tick)
			continue;

		// add to the active list
		_active_sghost_ids_curr->push_back(id);

		// if wasn't present in the tick before, add the notification to enter
		if (ghost.last_gameplay_tick_hit != (_gameplay_tick - 1)) {
			VisualServerCallbacks::Message msg;
			msg.object_id = ghost.object_id;
			msg.type = VisualServerCallbacks::CALLBACK_NOTIFICATION_ENTER_GAMEPLAY;

			callbacks->push_message(msg);
		}

		// mark as done
		ghost.last_gameplay_tick_hit = _gameplay_tick;
	}
}

void PortalGameplayMonitor::_swap() {
	LocalVector<uint32_t, int32_t> *temp = _active_moving_pool_ids_curr;
	_active_moving_pool_ids_curr = _active_moving_pool_ids_prev;
	_active_moving_pool_ids_prev = temp;
	_active_moving_pool_ids_curr->clear();

	temp = _active_rghost_pool_ids_curr;
	_active_rghost_pool_ids_curr = _active_rghost_pool_ids_prev;
	_active_rghost_pool_ids_prev = temp;
	_active_rghost_pool_ids_curr->clear();

	temp = _active_room_ids_curr;
	_active_room_ids_curr = _active_room_ids_prev;
	_active_room_ids_prev = temp;
	_active_room_ids_curr->clear();

	temp = _active_roomgroup_ids_curr;
	_active_roomgroup_ids_curr = _active_roomgroup_ids_prev;
	_active_roomgroup_ids_prev = temp;
	_active_roomgroup_ids_curr->clear();

	temp = _active_sghost_ids_curr;
	_active_sghost_ids_curr = _active_sghost_ids_prev;
	_active_sghost_ids_prev = temp;
	_active_sghost_ids_curr->clear();
}
