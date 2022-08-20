/*************************************************************************/
/*  joint_2d.h                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef JOINT_2D_H
#define JOINT_2D_H

#include "node_2d.h"

class PhysicsBody2D;

class Joint2D : public Node2D {
	GDCLASS(Joint2D, Node2D);

	RID joint;
	RID ba, bb;

	NodePath a;
	NodePath b;
	real_t bias = 0.0;

	bool exclude_from_collision = true;
	bool configured = false;
	String warning;

protected:
	void _disconnect_signals();
	void _body_exit_tree();
	void _update_joint(bool p_only_free = false);

	void _notification(int p_what);
	virtual void _configure_joint(RID p_joint, PhysicsBody2D *body_a, PhysicsBody2D *body_b) = 0;

	static void _bind_methods();

	_FORCE_INLINE_ bool is_configured() const { return configured; }

public:
	virtual TypedArray<String> get_configuration_warnings() const override;

	void set_node_a(const NodePath &p_node_a);
	NodePath get_node_a() const;

	void set_node_b(const NodePath &p_node_b);
	NodePath get_node_b() const;

	void set_bias(real_t p_bias);
	real_t get_bias() const;

	void set_exclude_nodes_from_collision(bool p_enable);
	bool get_exclude_nodes_from_collision() const;

	RID get_joint() const { return joint; }
	Joint2D();
	~Joint2D();
};

class PinJoint2D : public Joint2D {
	GDCLASS(PinJoint2D, Joint2D);

	real_t softness = 0.0;

protected:
	void _notification(int p_what);
	virtual void _configure_joint(RID p_joint, PhysicsBody2D *body_a, PhysicsBody2D *body_b) override;
	static void _bind_methods();

public:
	void set_softness(real_t p_softness);
	real_t get_softness() const;

	PinJoint2D();
};

class GrooveJoint2D : public Joint2D {
	GDCLASS(GrooveJoint2D, Joint2D);

	real_t length = 50.0;
	real_t initial_offset = 25.0;

protected:
	void _notification(int p_what);
	virtual void _configure_joint(RID p_joint, PhysicsBody2D *body_a, PhysicsBody2D *body_b) override;
	static void _bind_methods();

public:
	void set_length(real_t p_length);
	real_t get_length() const;

	void set_initial_offset(real_t p_initial_offset);
	real_t get_initial_offset() const;

	GrooveJoint2D();
};

class DampedSpringJoint2D : public Joint2D {
	GDCLASS(DampedSpringJoint2D, Joint2D);

	real_t stiffness = 20.0;
	real_t damping = 1.0;
	real_t rest_length = 0.0;
	real_t length = 50.0;

protected:
	void _notification(int p_what);
	virtual void _configure_joint(RID p_joint, PhysicsBody2D *body_a, PhysicsBody2D *body_b) override;
	static void _bind_methods();

public:
	void set_length(real_t p_length);
	real_t get_length() const;

	void set_rest_length(real_t p_rest_length);
	real_t get_rest_length() const;

	void set_damping(real_t p_damping);
	real_t get_damping() const;

	void set_stiffness(real_t p_stiffness);
	real_t get_stiffness() const;

	DampedSpringJoint2D();
};

#endif // JOINT_2D_H
