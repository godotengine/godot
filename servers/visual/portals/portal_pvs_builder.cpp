/**************************************************************************/
/*  portal_pvs_builder.cpp                                                */
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

#include "portal_pvs_builder.h"

#include "core/os/file_access.h"
#include "core/os/os.h"
#include "core/print_string.h"
#include "portal_renderer.h"

bool PVSBuilder::_log_active = true;

void PVSBuilder::find_neighbors(LocalVector<Neighbours> &r_neighbors) {
	// first find the neighbors
	int num_rooms = _portal_renderer->get_num_rooms();

	for (int n = 0; n < num_rooms; n++) {
		const VSRoom &room = _portal_renderer->get_room(n);

		// go through each portal
		int num_portals = room._portal_ids.size();

		for (int p = 0; p < num_portals; p++) {
			int portal_id = room._portal_ids[p];
			const VSPortal &portal = _portal_renderer->get_portal(portal_id);

			// everything depends on whether the portal is incoming or outgoing.
			// if incoming we reverse the logic.
			int outgoing = 1;

			int room_a_id = portal._linkedroom_ID[0];
			if (room_a_id != n) {
				outgoing = 0;
				DEV_ASSERT(portal._linkedroom_ID[1] == n);
			}

			// trace through this portal to the next room
			int linked_room_id = portal._linkedroom_ID[outgoing];

			// not relevant, portal doesn't go anywhere
			if (linked_room_id == -1)
				continue;

			r_neighbors[n].room_ids.push_back(linked_room_id);
		} // for p through portals

	} // for n through rooms

	// the secondary PVS is the primary PVS plus the neighbors
}

void PVSBuilder::create_secondary_pvs(int p_room_id, const LocalVector<Neighbours> &p_neighbors, BitFieldDynamic &r_bitfield_rooms) {
	VSRoom &room = _portal_renderer->get_room(p_room_id);
	room._secondary_pvs_first = _pvs->get_secondary_pvs_size();

	// go through each primary PVS room, and add the neighbors in the secondary pvs
	for (int r = 0; r < room._pvs_size; r++) {
		int pvs_entry = room._pvs_first + r;
		int pvs_room_id = _pvs->get_pvs_room_id(pvs_entry);

		// add the visible rooms first
		_pvs->add_to_secondary_pvs(pvs_room_id);
		room._secondary_pvs_size += 1;

		// now any neighbors of this that are not already added
		const Neighbours &neigh = p_neighbors[pvs_room_id];
		for (int n = 0; n < neigh.room_ids.size(); n++) {
			int neigh_room_id = neigh.room_ids[n];

			//log("\tconsidering neigh " + itos(neigh_room_id));

			if (r_bitfield_rooms.check_and_set(neigh_room_id)) {
				// add to the secondary pvs for this room
				_pvs->add_to_secondary_pvs(neigh_room_id);
				room._secondary_pvs_size += 1;
			} // neighbor room has not been added yet
		} // go through the neighbors
	} // go through each room in the primary pvs
}

#ifdef GODOT_PVS_SUPPORT_SAVE_FILE

bool PVSBuilder::load_pvs(String p_filename) {
	if (p_filename == "") {
		return false;
	}

	Error err;
	FileAccess *file = FileAccess::open(p_filename, FileAccess::READ, &err);

	if (err || !file) {
		if (file) {
			memdelete(file);
		}
		return false;
	}

	// goto needs vars declaring ahead of time
	int32_t num_rooms;
	int32_t pvs_size;

	if (!((file->get_8() == 'p') &&
				(file->get_8() == 'v') &&
				(file->get_8() == 's') &&
				(file->get_8() == ' '))) {
		goto failed;
	}

	num_rooms = file->get_32();
	if (num_rooms != _portal_renderer->get_num_rooms()) {
		goto failed;
	}

	for (int n = 0; n < num_rooms; n++) {
		if (file->eof_reached())
			goto failed;

		VSRoom &room = _portal_renderer->get_room(n);
		room._pvs_first = file->get_32();
		room._pvs_size = file->get_32();
		room._secondary_pvs_first = file->get_32();
		room._secondary_pvs_size = file->get_32();
	}

	pvs_size = file->get_32();

	for (int n = 0; n < pvs_size; n++) {
		_pvs->add_to_pvs(file->get_16());
	}

	// secondary pvs
	pvs_size = file->get_32();

	for (int n = 0; n < pvs_size; n++) {
		_pvs->add_to_secondary_pvs(file->get_16());
	}

	if (file) {
		memdelete(file);
	}

	return true;

failed:
	if (file) {
		memdelete(file);
	}

	return false;
}

void PVSBuilder::save_pvs(String p_filename) {
	if (p_filename == "") {
		p_filename = "res://test.pvs";
	}

	Error err;
	FileAccess *file = FileAccess::open(p_filename, FileAccess::WRITE, &err);

	if (err || !file) {
		if (file) {
			memdelete(file);
		}
		return;
	}

	file->store_8('p');
	file->store_8('v');
	file->store_8('s');
	file->store_8(' ');

	// hash? NYI

	// first save the room indices into the pvs
	int num_rooms = _portal_renderer->get_num_rooms();
	file->store_32(num_rooms);

	for (int n = 0; n < num_rooms; n++) {
		VSRoom &room = _portal_renderer->get_room(n);
		file->store_32(room._pvs_first);
		file->store_32(room._pvs_size);
		file->store_32(room._secondary_pvs_first);
		file->store_32(room._secondary_pvs_size);
	}

	int32_t pvs_size = _pvs->get_pvs_size();
	file->store_32(pvs_size);

	for (int n = 0; n < pvs_size; n++) {
		int16_t room_id = _pvs->get_pvs_room_id(n);
		file->store_16(room_id);
	}

	pvs_size = _pvs->get_secondary_pvs_size();
	file->store_32(pvs_size);

	for (int n = 0; n < pvs_size; n++) {
		int16_t room_id = _pvs->get_secondary_pvs_room_id(n);
		file->store_16(room_id);
	}

	if (file) {
		memdelete(file);
	}
}

#endif

void PVSBuilder::calculate_pvs(PortalRenderer &p_portal_renderer, String p_filename, int p_depth_limit, bool p_use_simple_pvs, bool p_log_pvs_generation) {
	_portal_renderer = &p_portal_renderer;
	_pvs = &p_portal_renderer.get_pvs();
	_depth_limit = p_depth_limit;

	_log_active = p_log_pvs_generation;

	// attempt to load from file rather than create each time
#ifdef GODOT_PVS_SUPPORT_SAVE_FILE
	if (load_pvs(p_filename)) {
		print_line("loaded pvs successfully from file " + p_filename);
		_pvs->set_loaded(true);
		return;
	}
#endif

	uint32_t time_before = OS::get_singleton()->get_ticks_msec();

	int num_rooms = _portal_renderer->get_num_rooms();
	BitFieldDynamic bf;
	bf.create(num_rooms);

	LocalVector<Neighbours> neighbors;
	neighbors.resize(num_rooms);

	// find the immediate neighbors of each room -
	// this is needed to create the secondary pvs
	find_neighbors(neighbors);

	for (int n = 0; n < num_rooms; n++) {
		bf.blank();

		//_visible_rooms.clear();

		LocalVector<Plane, int32_t> dummy_planes;

		VSRoom &room = _portal_renderer->get_room(n);
		room._pvs_first = _pvs->get_pvs_size();

		log("pvs from room : " + itos(n));

		if (p_use_simple_pvs) {
			trace_rooms_recursive_simple(0, n, n, -1, false, -1, dummy_planes, bf);
		} else {
			trace_rooms_recursive(0, n, n, -1, false, -1, dummy_planes, bf);
		}

		create_secondary_pvs(n, neighbors, bf);

		if (_log_active) {
			String string = "";
			for (int i = 0; i < room._pvs_size; i++) {
				int visible_room = _pvs->get_pvs_room_id(room._pvs_first + i);
				string += itos(visible_room);
				string += ", ";
			}

			log("\t" + string);

			string = "secondary : ";
			for (int i = 0; i < room._secondary_pvs_size; i++) {
				int visible_room = _pvs->get_secondary_pvs_room_id(room._secondary_pvs_first + i);
				string += itos(visible_room);
				string += ", ";
			}

			log("\t" + string);
		}
	}

	_pvs->set_loaded(true);

	uint32_t time_after = OS::get_singleton()->get_ticks_msec();

	print_verbose("calculated PVS in " + itos(time_after - time_before) + " ms.");

#ifdef GODOT_PVS_SUPPORT_SAVE_FILE
	save_pvs(p_filename);
#endif
}

void PVSBuilder::logd(int p_depth, String p_string) {
	if (!_log_active) {
		return;
	}

	String string_long;
	for (int n = 0; n < p_depth; n++) {
		string_long += "\t";
	}
	string_long += p_string;
	log(string_long);
}

void PVSBuilder::log(String p_string) {
	if (_log_active) {
		print_line(p_string);
	}
}

// The full routine deals with re-entrant rooms. I.e. more than one portal path can lead into a room.
// This makes the logic more complex, because we cannot terminate on the second entry to a room,
// and have to account for internal rooms, and the possibility of portal paths going back on themselves.
void PVSBuilder::trace_rooms_recursive(int p_depth, int p_source_room_id, int p_room_id, int p_first_portal_id, bool p_first_portal_outgoing, int p_previous_portal_id, const LocalVector<Plane, int32_t> &p_planes, BitFieldDynamic &r_bitfield_rooms, int p_from_external_room_id) {
	// prevent too much depth
	if (p_depth > _depth_limit) {
		WARN_PRINT_ONCE("PVS Depth Limit reached (seeing through too many portals)");
		return;
	}

	// is this room hit first time?
	if (r_bitfield_rooms.check_and_set(p_room_id)) {
		// only add to the room PVS of the source room once
		VSRoom &source_room = _portal_renderer->get_room(p_source_room_id);
		_pvs->add_to_pvs(p_room_id);
		source_room._pvs_size += 1;
	}

	logd(p_depth, "trace_rooms_recursive room " + itos(p_room_id));

	// get the room
	const VSRoom &room = _portal_renderer->get_room(p_room_id);

	// go through each portal
	int num_portals = room._portal_ids.size();

	for (int p = 0; p < num_portals; p++) {
		int portal_id = room._portal_ids[p];
		const VSPortal &portal = _portal_renderer->get_portal(portal_id);

		// everything depends on whether the portal is incoming or outgoing.
		// if incoming we reverse the logic.
		int outgoing = 1;

		int room_a_id = portal._linkedroom_ID[0];
		if (room_a_id != p_room_id) {
			outgoing = 0;
			DEV_ASSERT(portal._linkedroom_ID[1] == p_room_id);
		}

		// trace through this portal to the next room
		int linked_room_id = portal._linkedroom_ID[outgoing];

		// not relevant, portal doesn't go anywhere
		if (linked_room_id == -1)
			continue;

		// For pvs there is no real start point, but we will use the centre of the first portal.
		// This is used for checking portals are pointing outward from start point.
		if (p_source_room_id == p_room_id) {
			_trace_start_point = portal._pt_center;

			// We will use a small epsilon because we don't want to trace out
			// to coplanar portals for the first to second portals, before planes
			// have been added. So we will place the trace start point slightly
			// behind the first portal plane (e.g. slightly in the source room).
			// The epsilon must balance being enough in not to cause numerical error
			// at large distances from the origin, but too large and this will also
			// prevent the PVS entering portals that are very closely aligned
			// to the portal in.
			// Closely aligned portals should not happen in normal level design,
			// and will usually be a design error.
			// Watch for bugs here though, caused by closely aligned portals.

			// The epsilon should be BEHIND the way we are going through the portal,
			// so depends whether it is outgoing or not
			if (outgoing) {
				_trace_start_point -= portal._plane.normal * 0.1;
			} else {
				_trace_start_point += portal._plane.normal * 0.1;
			}

		} else {
			// much better way of culling portals by direction to camera...
			// instead of using dot product with a varying view direction, we simply find which side of the portal
			// plane the camera is on! If it is behind, the portal can be seen through, if in front, it can't
			real_t dist_cam = portal._plane.distance_to(_trace_start_point);

			if (!outgoing) {
				dist_cam = -dist_cam;
			}

			if (dist_cam >= 0.0) {
				// logd(p_depth + 2, "portal WRONG DIRECTION");
				continue;
			}
		}

		logd(p_depth + 1, "portal to room " + itos(linked_room_id));

		// is it culled by the planes?
		VSPortal::ClipResult overall_res = VSPortal::ClipResult::CLIP_INSIDE;

		// while clipping to the planes we maintain a list of partial planes, so we can add them to the
		// recursive next iteration of planes to check
		static LocalVector<int> partial_planes;
		partial_planes.clear();

		for (int32_t l = 0; l < p_planes.size(); l++) {
			VSPortal::ClipResult res = portal.clip_with_plane(p_planes[l]);

			switch (res) {
				case VSPortal::ClipResult::CLIP_OUTSIDE: {
					overall_res = res;
				} break;
				case VSPortal::ClipResult::CLIP_PARTIAL: {
					// if the portal intersects one of the planes, we should take this plane into account
					// in the next call of this recursive trace, because it can be used to cull out more objects
					overall_res = res;
					partial_planes.push_back(l);
				} break;
				default: // suppress warning
					break;
			}

			// if the portal was totally outside the 'frustum' then we can ignore it
			if (overall_res == VSPortal::ClipResult::CLIP_OUTSIDE)
				break;
		}

		// this portal is culled
		if (overall_res == VSPortal::ClipResult::CLIP_OUTSIDE) {
			logd(p_depth + 2, "portal CLIP_OUTSIDE");
			continue;
		}

		// Don't allow portals from internal to external room to be followed
		// if the external room has already been processed in this trace stack. This prevents
		// unneeded processing, and also prevents recursive feedback where you
		// see into internal room -> external room and back into the same internal room
		// via the same portal.
		if (portal._internal && (linked_room_id != -1)) {
			if (outgoing) {
				if (linked_room_id == p_from_external_room_id) {
					continue;
				}
			} else {
				// We are entering an internal portal from an external room.
				// set the external room id, so we can recognise this when we are
				// later exiting the internal rooms.
				// Note that as we can only store 1 previous external room, this system
				// won't work completely correctly when you have 2 levels of internal room
				// and you can see from roomgroup a -> b -> c. However this should just result
				// in a little slower culling for that particular view, and hopefully will not break
				// with recursive loop looking through the same portal multiple times. (don't think this
				// is possible in this scenario).
				p_from_external_room_id = p_room_id;
			}
		}

		// construct new planes
		LocalVector<Plane, int32_t> planes;

		if (p_first_portal_id != -1) {
			// add new planes
			const VSPortal &first_portal = _portal_renderer->get_portal(p_first_portal_id);
			portal.add_pvs_planes(first_portal, p_first_portal_outgoing, planes, outgoing != 0);

//#define GODOT_PVS_EXTRA_REJECT_TEST
#ifdef GODOT_PVS_EXTRA_REJECT_TEST
			// extra reject test for pvs - was the previous portal points outside the planes formed by the new portal?
			// not fully tested and not yet found a situation where needed, but will leave in in case testers find
			// such a situation.
			if (p_previous_portal_id != -1) {
				const VSPortal &prev_portal = _portal_renderer->get_portal(p_previous_portal_id);
				if (prev_portal._pvs_is_outside_planes(planes)) {
					continue;
				}
			}
#endif
		}

		// if portal is totally inside the planes, don't copy the old planes ..
		// i.e. we can now cull using the portal and forget about the rest of the frustum (yay)
		if (overall_res != VSPortal::ClipResult::CLIP_INSIDE) {
			// if it WASN'T totally inside the existing frustum, we also need to add any existing planes
			// that cut the portal.
			for (uint32_t n = 0; n < partial_planes.size(); n++)
				planes.push_back(p_planes[partial_planes[n]]);
		}

		// hopefully the portal actually leads somewhere...
		if (linked_room_id != -1) {
			// we either pass on the first portal id, or we start
			// it here, because we are looking through the first portal
			int first_portal_id = p_first_portal_id;
			if (first_portal_id == -1) {
				first_portal_id = portal_id;
				p_first_portal_outgoing = outgoing != 0;
			}

			trace_rooms_recursive(p_depth + 1, p_source_room_id, linked_room_id, first_portal_id, p_first_portal_outgoing, portal_id, planes, r_bitfield_rooms, p_from_external_room_id);
		} // linked room is valid
	}
}

// This simpler routine was the first used. It is reliable and no epsilons, and fast.
// But it will not create the correct result where there are multiple portal paths
// through a room when building the PVS.
void PVSBuilder::trace_rooms_recursive_simple(int p_depth, int p_source_room_id, int p_room_id, int p_first_portal_id, bool p_first_portal_outgoing, int p_previous_portal_id, const LocalVector<Plane, int32_t> &p_planes, BitFieldDynamic &r_bitfield_rooms) {
	// has this room been done already?
	if (!r_bitfield_rooms.check_and_set(p_room_id)) {
		return;
	}

	// prevent too much depth
	if (p_depth > _depth_limit) {
		WARN_PRINT_ONCE("Portal Depth Limit reached (seeing through too many portals)");
		return;
	}

	logd(p_depth, "trace_rooms_recursive room " + itos(p_room_id));

	// get the room
	const VSRoom &room = _portal_renderer->get_room(p_room_id);

	// add to the room PVS of the source room
	VSRoom &source_room = _portal_renderer->get_room(p_source_room_id);
	_pvs->add_to_pvs(p_room_id);
	source_room._pvs_size += 1;

	// go through each portal
	int num_portals = room._portal_ids.size();

	for (int p = 0; p < num_portals; p++) {
		int portal_id = room._portal_ids[p];
		const VSPortal &portal = _portal_renderer->get_portal(portal_id);

		// everything depends on whether the portal is incoming or outgoing.
		// if incoming we reverse the logic.
		int outgoing = 1;

		int room_a_id = portal._linkedroom_ID[0];
		if (room_a_id != p_room_id) {
			outgoing = 0;
			DEV_ASSERT(portal._linkedroom_ID[1] == p_room_id);
		}

		// trace through this portal to the next room
		int linked_room_id = portal._linkedroom_ID[outgoing];

		logd(p_depth + 1, "portal to room " + itos(linked_room_id));

		// not relevant, portal doesn't go anywhere
		if (linked_room_id == -1)
			continue;

		// linked room done already?
		if (r_bitfield_rooms.get_bit(linked_room_id))
			continue;

		// is it culled by the planes?
		VSPortal::ClipResult overall_res = VSPortal::ClipResult::CLIP_INSIDE;

		// while clipping to the planes we maintain a list of partial planes, so we can add them to the
		// recursive next iteration of planes to check
		static LocalVector<int> partial_planes;
		partial_planes.clear();

		for (int32_t l = 0; l < p_planes.size(); l++) {
			VSPortal::ClipResult res = portal.clip_with_plane(p_planes[l]);

			switch (res) {
				case VSPortal::ClipResult::CLIP_OUTSIDE: {
					overall_res = res;
				} break;
				case VSPortal::ClipResult::CLIP_PARTIAL: {
					// if the portal intersects one of the planes, we should take this plane into account
					// in the next call of this recursive trace, because it can be used to cull out more objects
					overall_res = res;
					partial_planes.push_back(l);
				} break;
				default: // suppress warning
					break;
			}

			// if the portal was totally outside the 'frustum' then we can ignore it
			if (overall_res == VSPortal::ClipResult::CLIP_OUTSIDE)
				break;
		}

		// this portal is culled
		if (overall_res == VSPortal::ClipResult::CLIP_OUTSIDE) {
			logd(p_depth + 2, "portal CLIP_OUTSIDE");
			continue;
		}

		// construct new planes
		LocalVector<Plane, int32_t> planes;

		if (p_first_portal_id != -1) {
			// add new planes
			const VSPortal &first_portal = _portal_renderer->get_portal(p_first_portal_id);
			portal.add_pvs_planes(first_portal, p_first_portal_outgoing, planes, outgoing != 0);

#ifdef GODOT_PVS_EXTRA_REJECT_TEST
			// extra reject test for pvs - was the previous portal points outside the planes formed by the new portal?
			// not fully tested and not yet found a situation where needed, but will leave in in case testers find
			// such a situation.
			if (p_previous_portal_id != -1) {
				const VSPortal &prev_portal = _portal_renderer->get_portal(p_previous_portal_id);
				if (prev_portal._pvs_is_outside_planes(planes)) {
					continue;
				}
			}
#endif
		}

		// if portal is totally inside the planes, don't copy the old planes ..
		// i.e. we can now cull using the portal and forget about the rest of the frustum (yay)
		if (overall_res != VSPortal::ClipResult::CLIP_INSIDE) {
			// if it WASN'T totally inside the existing frustum, we also need to add any existing planes
			// that cut the portal.
			for (uint32_t n = 0; n < partial_planes.size(); n++)
				planes.push_back(p_planes[partial_planes[n]]);
		}

		// hopefully the portal actually leads somewhere...
		if (linked_room_id != -1) {
			// we either pass on the first portal id, or we start
			// it here, because we are looking through the first portal
			int first_portal_id = p_first_portal_id;
			if (first_portal_id == -1) {
				first_portal_id = portal_id;
				p_first_portal_outgoing = outgoing != 0;
			}

			trace_rooms_recursive(p_depth + 1, p_source_room_id, linked_room_id, first_portal_id, p_first_portal_outgoing, portal_id, planes, r_bitfield_rooms);
		} // linked room is valid
	}
}
