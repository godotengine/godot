/**************************************************************************/
/*  visualizer_3d.h                                                       */
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

#pragma once

#include "core/object/object.h"
#include "core/templates/local_vector.h"
#include "scene/resources/material.h"

class Visualizer3D : public Object {
	GDCLASS(Visualizer3D, Object);

	static Visualizer3D *singleton;

	struct DebugLine {
		Vector3 from;
		Vector3 to;
		Color color;
		float width;
		float remaining_time;
	};

	LocalVector<DebugLine> lines;

	RID mesh_rid;
	RID instance_rid;
	RID scenario_rid;
	Ref<ShaderMaterial> debug_material;
	bool rs_initialized = false;

	void _ensure_rs_resources();
	void _free_rs_resources();
	void _rebuild_mesh();
	RID _get_active_scenario() const;

protected:
	static void _bind_methods();

public:
	static Visualizer3D *get_singleton();

	void line(const Vector3 &from, const Vector3 &to, float duration = 0.0f, const Color &color = Color(1, 1, 1), float width = 1.0f);
	void arrow(const Vector3 &from, const Vector3 &to, float duration = 0.0f, const Color &color = Color(1, 1, 1), float width = 1.0f);
	void wire_box(const Vector3 &position, float size, const Vector3 &rotation = Vector3(), float duration = 0.0f, const Color &color = Color(1, 1, 1), float width = 1.0f);
	void wire_sphere(const Vector3 &position, float radius, float duration = 0.0f, const Color &color = Color(1, 1, 1), float width = 1.0f);

	void clear();

	virtual void process(double p_delta_time);

	Visualizer3D();
	~Visualizer3D();
};
