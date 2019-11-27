/*************************************************************************/
/*  physics_bone_joint_3d.h                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef PHYSICS_BONE_JOINT_3D_H
#define PHYSICS_BONE_JOINT_3D_H

#include "scene/3d/physics_bone_3d.h"
#include "scene/3d/physics_joint_3d.h"

/**
 * @author Marios Staikopoulos <marios@staik.net>
 */

class BonePinJoint3D : public PinJoint3D {
	GDCLASS(BonePinJoint3D, PinJoint3D);

protected:
	virtual void compute_body_offsets(Transform &o_offset_a, Transform &o_offset_b, const PhysicsBody3D &p_body_a, const PhysicsBody3D *p_body_b) override;
};

/////////////////////////////////////////////////////////////////////

class BoneHingeJoint3D : public HingeJoint3D {
	GDCLASS(BoneHingeJoint3D, HingeJoint3D);

protected:
	virtual void compute_body_offsets(Transform &o_offset_a, Transform &o_offset_b, const PhysicsBody3D &p_body_a, const PhysicsBody3D *p_body_b) override;
};

/////////////////////////////////////////////////////////////////////

class BoneSliderJoint3D : public SliderJoint3D {
	GDCLASS(BoneSliderJoint3D, SliderJoint3D);

protected:
	virtual void compute_body_offsets(Transform &o_offset_a, Transform &o_offset_b, const PhysicsBody3D &p_body_a, const PhysicsBody3D *p_body_b) override;
};

/////////////////////////////////////////////////////////////////////

class BoneConeTwistJoint3D : public ConeTwistJoint3D {
	GDCLASS(BoneConeTwistJoint3D, ConeTwistJoint3D);

protected:
	virtual void compute_body_offsets(Transform &o_offset_a, Transform &o_offset_b, const PhysicsBody3D &p_body_a, const PhysicsBody3D *p_body_b) override;
};

/////////////////////////////////////////////////////////////////////

class BoneGeneric6DOFJoint3D : public Generic6DOFJoint3D {
	GDCLASS(BoneGeneric6DOFJoint3D, Generic6DOFJoint3D);

protected:
	virtual void compute_body_offsets(Transform &o_offset_a, Transform &o_offset_b, const PhysicsBody3D &p_body_a, const PhysicsBody3D *p_body_b) override;
};

#endif // PHYSICS_BONE_JOINT_3D_H
