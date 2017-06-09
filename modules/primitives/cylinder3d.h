/*************************************************************************/
/*  cylinder.h                                                           */
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
#ifndef CYLINDER3D_H
#define CYLINDER3D_H

#include "rid.h"
#include "scene/3d/visual_instance.h"

/**
	@author Bastiaan Olij <mux213@gmail.com>
*/
class Cylinder3D : public GeometryInstance {

	OBJ_TYPE(Cylinder3D, GeometryInstance);

	float top_radius;
	float bottom_radius;
	float height;
	int segments;
	int rings;

	AABB aabb;
	bool configured;
	bool pending_update;
	RID mesh;

	void _update();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_top_radius(const float p_radius);
	float get_top_radius() const;

	void set_bottom_radius(const float p_radius);
	float get_bottom_radius() const;

	void set_height(const float p_height);
	float get_height() const;

	void set_segments(const int p_segments);
	int get_segments() const;

	void set_rings(const int p_rings);
	int get_rings() const;

	virtual AABB get_aabb() const;
	virtual DVector<Face3> get_faces(uint32_t p_usage_flags) const;

	Cylinder3D();
	~Cylinder3D();
};

#endif
