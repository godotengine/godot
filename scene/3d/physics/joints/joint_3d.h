/**************************************************************************/
/*  joint_3d.h                                                            */
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

#ifndef JOINT_3D_H
#define JOINT_3D_H

#include "scene/3d/node_3d.h"
#include "scene/3d/physics/physics_body_3d.h"

class Joint3D : public Node3D {
	GDCLASS(Joint3D, Node3D);

	RID ba, bb;

	RID joint;

	NodePath a;
	NodePath b;

	int solver_priority = 1;
	bool exclude_from_collision = true;
	String warning;
	bool configured = false;

protected:
	void _disconnect_signals();
	void _body_exit_tree();
	void _update_joint(bool p_only_free = false);

	void _notification(int p_what);

	virtual void _configure_joint(RID p_joint, PhysicsBody3D *body_a, PhysicsBody3D *body_b) = 0;

	static void _bind_methods();

	_FORCE_INLINE_ bool is_configured() const { return configured; }

public:
	virtual PackedStringArray get_configuration_warnings() const override;

	void set_node_a(const NodePath &p_node_a);
	NodePath get_node_a() const;

	void set_node_b(const NodePath &p_node_b);
	NodePath get_node_b() const;

	void set_solver_priority(int p_priority);
	int get_solver_priority() const;

	void set_exclude_nodes_from_collision(bool p_enable);
	bool get_exclude_nodes_from_collision() const;

	RID get_rid() const { return joint; }
	Joint3D();
	~Joint3D();
};

#endif // JOINT_3D_H
