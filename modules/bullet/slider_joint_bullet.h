/*************************************************************************/
/*  slider_joint_bullet.h                                                */
/*  Author: AndreaCatania                                                */
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

#ifndef SLIDER_JOINT_BULLET_H
#define SLIDER_JOINT_BULLET_H

#include "joint_bullet.h"

class RigidBodyBullet;

class SliderJointBullet : public JointBullet {
	class btSliderConstraint *sliderConstraint;

public:
	/// Reference frame is A
	SliderJointBullet(RigidBodyBullet *rbA, RigidBodyBullet *rbB, const Transform &frameInA, const Transform &frameInB);

	virtual PhysicsServer::JointType get_type() const { return PhysicsServer::JOINT_SLIDER; }

	const RigidBodyBullet *getRigidBodyA() const;
	const RigidBodyBullet *getRigidBodyB() const;
	const Transform getCalculatedTransformA() const;
	const Transform getCalculatedTransformB() const;
	const Transform getFrameOffsetA() const;
	const Transform getFrameOffsetB() const;
	Transform getFrameOffsetA();
	Transform getFrameOffsetB();
	real_t getLowerLinLimit() const;
	void setLowerLinLimit(real_t lowerLimit);
	real_t getUpperLinLimit() const;
	void setUpperLinLimit(real_t upperLimit);
	real_t getLowerAngLimit() const;
	void setLowerAngLimit(real_t lowerLimit);
	real_t getUpperAngLimit() const;
	void setUpperAngLimit(real_t upperLimit);

	real_t getSoftnessDirLin() const;
	real_t getRestitutionDirLin() const;
	real_t getDampingDirLin() const;
	real_t getSoftnessDirAng() const;
	real_t getRestitutionDirAng() const;
	real_t getDampingDirAng() const;
	real_t getSoftnessLimLin() const;
	real_t getRestitutionLimLin() const;
	real_t getDampingLimLin() const;
	real_t getSoftnessLimAng() const;
	real_t getRestitutionLimAng() const;
	real_t getDampingLimAng() const;
	real_t getSoftnessOrthoLin() const;
	real_t getRestitutionOrthoLin() const;
	real_t getDampingOrthoLin() const;
	real_t getSoftnessOrthoAng() const;
	real_t getRestitutionOrthoAng() const;
	real_t getDampingOrthoAng() const;
	void setSoftnessDirLin(real_t softnessDirLin);
	void setRestitutionDirLin(real_t restitutionDirLin);
	void setDampingDirLin(real_t dampingDirLin);
	void setSoftnessDirAng(real_t softnessDirAng);
	void setRestitutionDirAng(real_t restitutionDirAng);
	void setDampingDirAng(real_t dampingDirAng);
	void setSoftnessLimLin(real_t softnessLimLin);
	void setRestitutionLimLin(real_t restitutionLimLin);
	void setDampingLimLin(real_t dampingLimLin);
	void setSoftnessLimAng(real_t softnessLimAng);
	void setRestitutionLimAng(real_t restitutionLimAng);
	void setDampingLimAng(real_t dampingLimAng);
	void setSoftnessOrthoLin(real_t softnessOrthoLin);
	void setRestitutionOrthoLin(real_t restitutionOrthoLin);
	void setDampingOrthoLin(real_t dampingOrthoLin);
	void setSoftnessOrthoAng(real_t softnessOrthoAng);
	void setRestitutionOrthoAng(real_t restitutionOrthoAng);
	void setDampingOrthoAng(real_t dampingOrthoAng);
	void setPoweredLinMotor(bool onOff);
	bool getPoweredLinMotor();
	void setTargetLinMotorVelocity(real_t targetLinMotorVelocity);
	real_t getTargetLinMotorVelocity();
	void setMaxLinMotorForce(real_t maxLinMotorForce);
	real_t getMaxLinMotorForce();
	void setPoweredAngMotor(bool onOff);
	bool getPoweredAngMotor();
	void setTargetAngMotorVelocity(real_t targetAngMotorVelocity);
	real_t getTargetAngMotorVelocity();
	void setMaxAngMotorForce(real_t maxAngMotorForce);
	real_t getMaxAngMotorForce();
	real_t getLinearPos();

	void set_param(PhysicsServer::SliderJointParam p_param, real_t p_value);
	real_t get_param(PhysicsServer::SliderJointParam p_param) const;
};
#endif
