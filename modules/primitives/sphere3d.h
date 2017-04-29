/*************************************************************************/
/*  sphere3d.h                                                           */
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
#ifndef SPHERE3D_H
#define SPHERE3D_H

#include "rid.h"
#include "scene/3d/visual_instance.h"

/**
	@author Bastiaan Olij <mux213@gmail.com>
*/
class Sphere3D : public GeometryInstance {

	OBJ_TYPE(Sphere3D, GeometryInstance);

	float radius;
	float height;
	int segments;
	int rings;
	bool is_hemisphere;

	AABB aabb;
	bool configured;
	bool pending_update;
	RID mesh;

	void _update();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_radius(const float p_radius);
	float get_radius() const;

	void set_height(const float p_height);
	float get_height() const;

	void set_segments(const int p_segments);
	int get_segments() const;

	void set_rings(const int p_rings);
	int get_rings() const;

	void set_is_hemisphere(const bool p_is_hemisphere);
	bool get_is_hemisphere() const;

	virtual AABB get_aabb() const;
	virtual DVector<Face3> get_faces(uint32_t p_usage_flags) const;

	Sphere3D();
	~Sphere3D();
};

#endif
