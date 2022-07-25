/*************************************************************************/
/*  portal_pvs.h                                                         */
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

#ifndef PORTAL_PVS_H
#define PORTAL_PVS_H

#include "core/local_vector.h"

class PVS {
public:
	void clear();

	void add_to_pvs(int p_room_id) { _room_pvs.push_back(p_room_id); }
	int32_t get_pvs_size() const { return _room_pvs.size(); }
	int32_t get_pvs_room_id(int32_t p_entry) const { return _room_pvs[p_entry]; }

	void add_to_secondary_pvs(int p_room_id) { _room_secondary_pvs.push_back(p_room_id); }
	int32_t get_secondary_pvs_size() const { return _room_secondary_pvs.size(); }
	int32_t get_secondary_pvs_room_id(int32_t p_entry) const { return _room_secondary_pvs[p_entry]; }

	void set_loaded(bool p_loaded) { _loaded = p_loaded; }
	bool is_loaded() const { return _loaded; }

private:
	// pvs
	LocalVector<uint16_t, int32_t> _room_pvs;
	// secondary pvs is primary plus the immediate neighbors of the primary pvs
	LocalVector<uint16_t, int32_t> _room_secondary_pvs;
	bool _loaded = false;
};

#endif // PORTAL_PVS_H
