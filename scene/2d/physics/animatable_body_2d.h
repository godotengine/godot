/**************************************************************************/
/*  animatable_body_2d.h                                                  */
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

#ifndef ANIMATABLE_BODY_2D_H
#define ANIMATABLE_BODY_2D_H

#include "scene/2d/physics/static_body_2d.h"

class AnimatableBody2D : public StaticBody2D {
	GDCLASS(AnimatableBody2D, StaticBody2D);

private:
	bool sync_to_physics = true;

	Transform2D last_valid_transform;

	static void _body_state_changed_callback(void *p_instance, PhysicsDirectBodyState2D *p_state);
	void _body_state_changed(PhysicsDirectBodyState2D *p_state);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	AnimatableBody2D();

private:
	void _update_kinematic_motion();

	void set_sync_to_physics(bool p_enable);
	bool is_sync_to_physics_enabled() const;
};

#endif // ANIMATABLE_BODY_2D_H
