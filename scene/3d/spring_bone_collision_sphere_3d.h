/**************************************************************************/
/*  spring_bone_collision_sphere_3d.h                                     */
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

#ifndef SPRING_BONE_COLLISION_SPHERE_3D_H
#define SPRING_BONE_COLLISION_SPHERE_3D_H

#include "scene/3d/spring_bone_collision_3d.h"

class SpringBoneCollisionCapsule3D;

class SpringBoneCollisionSphere3D : public SpringBoneCollision3D {
	GDCLASS(SpringBoneCollisionSphere3D, SpringBoneCollision3D);

	friend class SpringBoneCollisionCapsule3D;

	float radius = 0.1;
	bool inside = false;

protected:
	static void _bind_methods();

	static Vector3 _collide_sphere(const Vector3 &p_origin, float p_radius, bool p_inside, float p_bone_radius, float p_bone_length, const Vector3 &p_current);
	virtual Vector3 _collide(const Transform3D &p_center, float p_bone_radius, float p_bone_length, const Vector3 &p_current) const override;

public:
	void set_radius(float p_radius);
	float get_radius() const;
	void set_inside(bool p_enabled);
	bool is_inside() const;
};

#endif // SPRING_BONE_COLLISION_SPHERE_3D_H
