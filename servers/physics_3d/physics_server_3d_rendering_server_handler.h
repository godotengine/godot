/**************************************************************************/
/*  physics_server_3d_rendering_server_handler.h                          */
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

#include "core/object/gdvirtual.gen.h"
#include "core/object/object.h"

struct AABB;
struct Vector3;

class PhysicsServer3DRenderingServerHandler : public Object {
	GDCLASS(PhysicsServer3DRenderingServerHandler, Object)
protected:
	GDVIRTUAL2_REQUIRED(_set_vertex, int, const Vector3 &)
	GDVIRTUAL2_REQUIRED(_set_normal, int, const Vector3 &)
	GDVIRTUAL1_REQUIRED(_set_aabb, const AABB &)

	static void _bind_methods();

public:
	virtual void set_vertex(int p_vertex_id, const Vector3 &p_vertex);
	virtual void set_normal(int p_vertex_id, const Vector3 &p_normal);
	virtual void set_aabb(const AABB &p_aabb);

	virtual ~PhysicsServer3DRenderingServerHandler() {}
};
