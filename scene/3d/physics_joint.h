/*************************************************************************/
/*  physics_joint.h                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef PHYSICS_JOINT_H
#define PHYSICS_JOINT_H

#include "scene/3d/spatial.h"
#include "scene/3d/physics_body.h"

#if 0
class PhysicsJoint : public Spatial {

	OBJ_TYPE(PhysicsJoint,Spatial);
	OBJ_CATEGORY("3D Physics Nodes");

	NodePath body_A;
	NodePath body_B;
	bool active;
	bool no_collision;


	RID indicator_instance;

	RID _get_visual_instance_rid() const;
protected:

	RID joint;
	RID indicator;

	bool _set(const StringName& p_name, const Variant& p_value);
	bool _get(const StringName& p_name,Variant &r_ret) const;
	void _get_property_list( List<PropertyInfo> *p_list) const;
	void _notification(int p_what);
	static void _bind_methods();

	virtual RID create(PhysicsBody*A,PhysicsBody*B)=0;
	virtual void _update_indicator()=0;

	void _disconnect();
	void _connect();
public:

	void set_body_A(const NodePath& p_path);
	void set_body_B(const NodePath& p_path);
	NodePath get_body_A() const;
	NodePath get_body_B() const;

	void set_active(bool p_active);
	bool is_active() const;

	void set_disable_collision(bool p_active);
	bool has_disable_collision() const;

	void reconnect();

	RID get_rid();

	PhysicsJoint();
	~PhysicsJoint();
};



class PhysicsJointPin : public PhysicsJoint {

	OBJ_TYPE( PhysicsJointPin, PhysicsJoint );

protected:

	virtual void _update_indicator();
	virtual RID create(PhysicsBody*A,PhysicsBody*B);
public:


	PhysicsJointPin();
};

#endif // PHYSICS_JOINT_H
#endif
