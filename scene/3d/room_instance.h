/*************************************************************************/
/*  room_instance.h                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef ROOM_INSTANCE_H
#define ROOM_INSTANCE_H

#include "scene/3d/visual_instance.h"
#include "scene/resources/room.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

/* RoomInstance Logic:
   a) Instances that belong to the room are drawn only if the room is visible (seen through portal, or player inside)
   b) Instances that don't belong to any room are considered to belong to the root room (RID empty)
   c) "dynamic" Instances are assigned to the rooms their AABB touch

*/

class Room : public VisualInstance {

	GDCLASS(Room, VisualInstance);

public:
private:
	Ref<RoomBounds> room;

	int level;
	void _parse_node_faces(PoolVector<Face3> &all_faces, const Node *p_node) const;

	void _bounds_changed();

protected:
	void _notification(int p_what);

	static void _bind_methods();

public:
	enum {
		// used to notify portals that the room in which they are has changed.
		NOTIFICATION_AREA_CHANGED = 60
	};

	virtual Rect3 get_aabb() const;
	virtual PoolVector<Face3> get_faces(uint32_t p_usage_flags) const;

	void set_room(const Ref<RoomBounds> &p_room);
	Ref<RoomBounds> get_room() const;

	Room();
	~Room();
};

#endif // ROOM_INSTANCE_H
