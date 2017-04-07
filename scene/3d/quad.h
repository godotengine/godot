/*************************************************************************/
/*  quad.h                                                               */
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
#ifndef QUAD_H
#define QUAD_H

#include "rid.h"
#include "scene/3d/visual_instance.h"

class Quad : public GeometryInstance {

	GDCLASS(Quad, GeometryInstance);

	Vector3::Axis axis;
	bool centered;
	Vector2 offset;
	Vector2 size;

	Rect3 aabb;
	bool configured;
	bool pending_update;
	RID mesh;

	void _update();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_axis(Vector3::Axis p_axis);
	Vector3::Axis get_axis() const;

	void set_size(const Vector2 &p_sizze);
	Vector2 get_size() const;

	void set_offset(const Vector2 &p_offset);
	Vector2 get_offset() const;

	void set_centered(bool p_enabled);
	bool is_centered() const;

	virtual PoolVector<Face3> get_faces(uint32_t p_usage_flags) const;
	virtual Rect3 get_aabb() const;

	Quad();
	~Quad();
};

#endif // QUAD_H
