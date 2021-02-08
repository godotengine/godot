/*************************************************************************/
/*  godot_motion_state.h                                                 */
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

#ifndef GODOT_MOTION_STATE_H
#define GODOT_MOTION_STATE_H

#include "rigid_body_bullet.h"

#include <LinearMath/btMotionState.h>

/**
	@author AndreaCatania
*/

class RigidBodyBullet;

// This class is responsible to move kinematic actor
// and sincronize rendering engine with Bullet
/// DOC:
/// http://www.bulletphysics.org/mediawiki-1.5.8/index.php/MotionStates#What.27s_a_MotionState.3F
class GodotMotionState : public btMotionState {
	/// This data is used to store the new world position for kinematic body
	btTransform bodyKinematicWorldTransf;
	/// This data is used to store last world position
	btTransform bodyCurrentWorldTransform;

	RigidBodyBullet *owner = nullptr;

public:
	GodotMotionState(RigidBodyBullet *p_owner) :
			bodyKinematicWorldTransf(btMatrix3x3(1., 0., 0., 0., 1., 0., 0., 0., 1.), btVector3(0., 0., 0.)),
			bodyCurrentWorldTransform(btMatrix3x3(1., 0., 0., 0., 1., 0., 0., 0., 1.), btVector3(0., 0., 0.)),
			owner(p_owner) {}

	/// IMPORTANT DON'T USE THIS FUNCTION TO KNOW THE CURRENT BODY TRANSFORM
	/// This class is used internally by Bullet
	/// Use GodotMotionState::getCurrentWorldTransform to know current position
	///
	/// This function is used by Bullet to get the position of object in the world
	/// if the body is kinematic Bullet will move the object to this location
	/// if the body is static Bullet doesn't move at all
	virtual void getWorldTransform(btTransform &worldTrans) const {
		worldTrans = bodyKinematicWorldTransf;
	}

	/// IMPORTANT: to move the body use: moveBody
	/// IMPORTANT: DON'T CALL THIS FUNCTION, IT IS CALLED BY BULLET TO UPDATE RENDERING ENGINE
	///
	/// This function is called each time by Bullet and set the current position of body
	/// inside the physics world.
	/// Don't allow Godot rendering scene takes world transform from this object because
	/// the correct transform is set by Bullet only after the last step when there are sub steps
	/// This function must update Godot transform rendering scene for this object.
	virtual void setWorldTransform(const btTransform &worldTrans) {
		bodyCurrentWorldTransform = worldTrans;

		owner->notify_transform_changed();
	}

public:
	/// Use this function to move kinematic body
	/// -- or set initial transform before body creation.
	void moveBody(const btTransform &newWorldTransform) {
		bodyKinematicWorldTransf = newWorldTransform;
	}

	/// It returns the current body transform from last Bullet update
	const btTransform &getCurrentWorldTransform() const {
		return bodyCurrentWorldTransform;
	}
};
#endif
