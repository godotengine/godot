/*************************************************************************/
/*  proximity_group_3d.h                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef PROXIMITY_GROUP_H
#define PROXIMITY_GROUP_H

#include "node_3d.h"

class ProximityGroup3D : public Node3D {
	GDCLASS(ProximityGroup3D, Node3D);

public:
	enum DispatchMode {
		MODE_PROXY,
		MODE_SIGNAL,
	};

private:
	Map<StringName, uint32_t> groups;

	String group_name;
	DispatchMode dispatch_mode = MODE_PROXY;
	Vector3 grid_radius = Vector3(1, 1, 1);

	float cell_size = 1.0;
	uint32_t group_version = 0;

	void _clear_groups();
	void _update_groups();
	void _add_groups(int *p_cell, String p_base, int p_depth);
	void _new_group(StringName p_name);

	void _proximity_group_broadcast(String p_method, Variant p_parameters);

protected:
	void _notification(int p_what);

	static void _bind_methods();

public:
	void set_group_name(const String &p_group_name);
	String get_group_name() const;

	void set_dispatch_mode(DispatchMode p_mode);
	DispatchMode get_dispatch_mode() const;

	void set_grid_radius(const Vector3 &p_radius);
	Vector3 get_grid_radius() const;

	void broadcast(String p_method, Variant p_parameters);

	ProximityGroup3D();
	~ProximityGroup3D() {}
};

VARIANT_ENUM_CAST(ProximityGroup3D::DispatchMode);

#endif // PROXIMITY_GROUP_H
