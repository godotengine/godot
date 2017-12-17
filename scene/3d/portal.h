/*************************************************************************/
/*  portal.h                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#ifndef PORTAL_H
#define PORTAL_H

#include "scene/3d/visual_instance.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

/* Portal Logic:
   If a portal is placed next (very close to) a similar, opposing portal, they automatically connect,
   otherwise, a portal connects to the parent room
*/
// FIXME: This will be redone and replaced by area portals, left for reference
// since a new class with this name will have to exist and want to reuse the gizmos
#if 0
class Portal : public VisualInstance {

	GDCLASS(Portal, VisualInstance);

	RID portal;
	Vector<Point2> shape;

	bool enabled;
	float disable_distance;
	Color disabled_color;
	float connect_range;

	AABB aabb;

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	static void _bind_methods();

public:
	virtual AABB get_aabb() const;
	virtual PoolVector<Face3> get_faces(uint32_t p_usage_flags) const;

	void set_enabled(bool p_enabled);
	bool is_enabled() const;

	void set_disable_distance(float p_distance);
	float get_disable_distance() const;

	void set_disabled_color(const Color &p_disabled_color);
	Color get_disabled_color() const;

	void set_shape(const Vector<Point2> &p_shape);
	Vector<Point2> get_shape() const;

	void set_connect_range(float p_range);
	float get_connect_range() const;

	Portal();
	~Portal();
};

#endif
#endif
