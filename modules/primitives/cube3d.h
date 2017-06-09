/*************************************************************************/
/*  cube3d.h                                                             */
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
#ifndef CUBE3D_H
#define CUBE3D_H

#include "rid.h"
#include "scene/3d/visual_instance.h"

/**
	@author Bastiaan Olij <mux213@gmail.com>

	Similar to test cube but with tesselation support and different texture coordinates
*/
class Cube3D : public GeometryInstance {

	OBJ_TYPE(Cube3D, GeometryInstance);

	Vector3 size;
	int segments_w;
	int segments_h;
	int segments_d;

	AABB aabb;
	bool configured;
	bool pending_update;
	RID mesh;

	void _update();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_size(const Vector3 &p_size);
	Vector3 get_size() const;

	void set_segments_width(const int p_segments);
	int get_segments_width() const;

	void set_segments_height(const int p_segments);
	int get_segments_height() const;

	void set_segments_depth(const int p_segments);
	int get_segments_depth() const;

	virtual AABB get_aabb() const;
	virtual DVector<Face3> get_faces(uint32_t p_usage_flags) const;

	Cube3D();
	~Cube3D();
};

#endif
