/*************************************************************************/
/*  room_group.h                                                         */
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

#ifndef ROOM_GROUP_H
#define ROOM_GROUP_H

#include "core/rid.h"
#include "spatial.h"

class Room;

class RoomGroup : public Spatial {
	GDCLASS(RoomGroup, Spatial);

	friend class RoomManager;

	RID _room_group_rid;

public:
	RoomGroup();
	~RoomGroup();

	void add_room(Room *p_room);

	void set_roomgroup_priority(int p_priority) {
		_settings_priority = p_priority;
		_changed();
	}
	int get_roomgroup_priority() const { return _settings_priority; }

	String get_configuration_warning() const;

private:
	void clear();
	void _changed();

	// roomgroup ID during conversion
	int _roomgroup_ID;

	// the roomgroup can be used to set a number of rooms to a different priority
	// to allow a group of rooms WITHIN another room / rooms.
	// This is for e.g. buildings on landscape.
	int _settings_priority = 0;

	// makes sure lrooms are not converted more than once per
	// call to rooms_convert
	int _conversion_tick = -1;

protected:
	static void _bind_methods();
	void _notification(int p_what);
};

#endif // ROOM_GROUP_H
