/**************************************************************************/
/*  navigation_raycast_hit_3d.h                                           */
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

#ifndef NAVIGATION_RAYCAST_HIT_3D_H
#define NAVIGATION_RAYCAST_HIT_3D_H

#include "core/object/ref_counted.h"

class NavigationRaycastHit3D : public RefCounted {
	GDCLASS(NavigationRaycastHit3D, RefCounted);

	bool did_hit;
	Vector3 hit_position;
	Vector3 hit_normal;
	Vector<Vector3> raycast_path;
	real_t raycast_path_cost;

protected:
	static void _bind_methods();

public:
	void set_did_hit(const bool &p_did_hit);
	const bool &get_did_hit() const;

	void set_hit_position(const Vector3 &p_hit_position);
	const Vector3 &get_hit_position();

	void set_hit_normal(const Vector3 &p_hit_normal);
	const Vector3 &get_hit_normal();

	void set_raycast_path(const Vector<Vector3> &p_raycast_path);
	const Vector<Vector3> &get_raycast_path();

	void reset();
};

#endif // NAVIGATION_RAYCAST_HIT_3D_H
