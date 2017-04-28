/*************************************************************************/
/*  light_occluder_2d.h                                                  */
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
#ifndef LIGHTOCCLUDER2D_H
#define LIGHTOCCLUDER2D_H

#include "scene/2d/node_2d.h"

class OccluderPolygon2D : public Resource {

	GDCLASS(OccluderPolygon2D, Resource);

public:
	enum CullMode {
		CULL_DISABLED,
		CULL_CLOCKWISE,
		CULL_COUNTER_CLOCKWISE
	};

private:
	RID occ_polygon;
	PoolVector<Vector2> polygon;
	bool closed;
	CullMode cull;

protected:
	static void _bind_methods();

public:
	void set_polygon(const PoolVector<Vector2> &p_polygon);
	PoolVector<Vector2> get_polygon() const;

	void set_closed(bool p_closed);
	bool is_closed() const;

	void set_cull_mode(CullMode p_mode);
	CullMode get_cull_mode() const;

	virtual RID get_rid() const;
	OccluderPolygon2D();
	~OccluderPolygon2D();
};

VARIANT_ENUM_CAST(OccluderPolygon2D::CullMode);

class LightOccluder2D : public Node2D {
	GDCLASS(LightOccluder2D, Node2D);

	RID occluder;
	bool enabled;
	int mask;
	Ref<OccluderPolygon2D> occluder_polygon;

#ifdef DEBUG_ENABLED
	void _poly_changed();
#endif

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_occluder_polygon(const Ref<OccluderPolygon2D> &p_polygon);
	Ref<OccluderPolygon2D> get_occluder_polygon() const;

	void set_occluder_light_mask(int p_mask);
	int get_occluder_light_mask() const;

	String get_configuration_warning() const;

	LightOccluder2D();
	~LightOccluder2D();
};

#endif // LIGHTOCCLUDER2D_H
