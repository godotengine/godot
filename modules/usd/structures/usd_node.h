/**************************************************************************/
/*  usd_node.h                                                            */
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

#include "core/io/resource.h"

class USDNode : public Resource {
	GDCLASS(USDNode, Resource);

private:
	// Scene hierarchy.
	String original_name;
	int parent = -1;
	Vector<int> children;
	Transform3D xform;

	// Indices into USDState arrays (-1 = none).
	int mesh = -1;
	int camera = -1;
	int light = -1;
	int skeleton = -1;
	int skin = -1;

	bool visible = true;

protected:
	static void _bind_methods();

public:
	String get_original_name() const;
	void set_original_name(const String &p_name);

	int get_parent() const;
	void set_parent(int p_parent);

	Vector<int> get_children() const;
	void set_children(const Vector<int> &p_children);
	void append_child_index(int p_child_index);

	Transform3D get_xform() const;
	void set_xform(const Transform3D &p_xform);

	int get_mesh() const;
	void set_mesh(int p_mesh);

	int get_camera() const;
	void set_camera(int p_camera);

	int get_light() const;
	void set_light(int p_light);

	int get_skeleton() const;
	void set_skeleton(int p_skeleton);

	int get_skin() const;
	void set_skin(int p_skin);

	bool get_visible() const;
	void set_visible(bool p_visible);
};
