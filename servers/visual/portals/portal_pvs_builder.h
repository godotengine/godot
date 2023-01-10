/**************************************************************************/
/*  portal_pvs_builder.h                                                  */
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

#ifndef PORTAL_PVS_BUILDER_H
#define PORTAL_PVS_BUILDER_H

#include "core/bitfield_dynamic.h"
#include "core/local_vector.h"
#include "core/math/plane.h"

//#define GODOT_PVS_SUPPORT_SAVE_FILE

class PortalRenderer;
class PVS;

class PVSBuilder {
	struct Neighbours {
		LocalVector<int32_t, int32_t> room_ids;
	};

public:
	void calculate_pvs(PortalRenderer &p_portal_renderer, String p_filename, int p_depth_limit, bool p_use_simple_pvs, bool p_log_pvs_generation);

private:
#ifdef GODOT_PVS_SUPPORT_SAVE_FILE
	bool load_pvs(String p_filename);
	void save_pvs(String p_filename);
#endif
	void find_neighbors(LocalVector<Neighbours> &r_neighbors);

	void logd(int p_depth, String p_string);
	void log(String p_string);

	void trace_rooms_recursive(int p_depth, int p_source_room_id, int p_room_id, int p_first_portal_id, bool p_first_portal_outgoing, int p_previous_portal_id, const LocalVector<Plane, int32_t> &p_planes, BitFieldDynamic &r_bitfield_rooms, int p_from_external_room_id = -1);
	void trace_rooms_recursive_simple(int p_depth, int p_source_room_id, int p_room_id, int p_first_portal_id, bool p_first_portal_outgoing, int p_previous_portal_id, const LocalVector<Plane, int32_t> &p_planes, BitFieldDynamic &r_bitfield_rooms);

	void create_secondary_pvs(int p_room_id, const LocalVector<Neighbours> &p_neighbors, BitFieldDynamic &r_bitfield_rooms);

	PortalRenderer *_portal_renderer = nullptr;
	PVS *_pvs = nullptr;
	int _depth_limit = 16;
	Vector3 _trace_start_point;

	static bool _log_active;
};

#endif // PORTAL_PVS_BUILDER_H
