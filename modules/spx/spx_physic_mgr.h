/**************************************************************************/
/*  spx_physic_mgr.h                                                      */
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

#ifndef SPX_PHYSIC_MGR_H
#define SPX_PHYSIC_MGR_H

#include "gdextension_spx_ext.h"
#include "spx_base_mgr.h"


class SpxPhysicDefine{
private:
	static GdFloat global_gravity;
	static GdFloat global_friction;
	static GdFloat global_air_drag;
public:
	static void set_global_gravity(GdFloat gravity);
	static GdFloat get_global_gravity();
	static void set_global_friction(GdFloat friction);
	static GdFloat get_global_friction();
	static void set_global_air_drag(GdFloat air_drag);
	static GdFloat get_global_air_drag();
};

class SpxRaycastInfo{
public:
	GdBool collide;
	GdObj sprite_gid;
	GdVec2 position;
	GdVec2 normal;
public:
    SpxRaycastInfo() = default;
    ~SpxRaycastInfo() = default;
	GdArray ToArray();
};

class SpxPhysicMgr : SpxBaseMgr {
	SPXCLASS(SpxPhysicMgr, SpxBaseMgr)

private:
	GdArray _check_collision(RID shape, GdVec2 pos, GdInt collision_mask);
	SpxRaycastInfo _raycast(GdVec2 from, GdVec2 to,GdArray ignore_sprites,GdInt collision_mask,GdBool collide_with_areas,GdBool collide_with_bodies);

public:
	bool is_collision_by_pixel;
	void on_awake() override;

	static const int COLLIDER_NONE    = 0x00;
	static const int COLLIDER_AUTO    = 0x01;
	static const int COLLIDER_CIRCLE  = 0x02;
	static const int COLLIDER_RECT    = 0x03;
	static const int COLLIDER_CAPSULE = 0x04;
	static const int COLLIDER_POLYGON = 0x05;
public:
	virtual ~SpxPhysicMgr() = default; // Added virtual destructor to fix -Werror=non-virtual-dtor
	GdObj raycast(GdVec2 from, GdVec2 to, GdInt collision_mask);
	GdBool check_collision(GdVec2 from, GdVec2 to, GdInt collision_mask, GdBool collide_with_areas, GdBool collide_with_bodies);
	GdInt check_touched_camera_boundaries(GdObj obj);
	GdBool check_touched_camera_boundary(GdObj obj,GdInt board_type);
	void set_collision_system_type(GdBool is_collision_by_alpha);
	// configs
	void set_global_gravity(GdFloat gravity);
	GdFloat get_global_gravity();
	void set_global_friction(GdFloat friction);
	GdFloat get_global_friction();
	void set_global_air_drag(GdFloat air_drag);
	GdFloat get_global_air_drag();

	// check collision
	GdArray check_collision_rect(GdVec2 pos, GdVec2 size, GdInt collision_mask);
	GdArray check_collision_circle(GdVec2 pos, GdFloat radius, GdInt collision_mask);
	GdArray raycast_with_details(GdVec2 from, GdVec2 to,GdArray ignore_sprites,GdInt collision_mask,GdBool collide_with_areas,GdBool collide_with_bodies);
};

#endif // SPX_PHYSIC_MGR_H
