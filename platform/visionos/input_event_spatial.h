/**************************************************************************/
/*  input_event_spatial.h                                                 */
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

#include "core/input/input_event.h"

class InputEventSpatial : public InputEventFromWindow {
	GDCLASS(InputEventSpatial, InputEventFromWindow)
protected:
	static void _bind_methods();

public:
	enum Flags {
		FLAG_HAS_CHIRALITY,
		FLAG_HAS_SELECTION_RAY,
		FLAG_MAX,
	};

	enum Chirality {
		CHIRALITY_LEFT,
		CHIRALITY_RIGHT,
	};

	enum Phase {
		PHASE_ACTIVE,
		PHASE_CANCELLED,
		PHASE_ENDED,
	};

	bool get_flag(Flags p_flag) const { return flags[uint32_t(p_flag)]; }
	void set_flag(Flags p_flag, bool p_value) { flags[uint32_t(p_flag)] = p_value; }

	void set_index(int p_index) { index = p_index; }
	int get_index() const { return index; }

	Phase get_phase() const { return phase; }
	void set_phase(Phase p_phase) { phase = p_phase; }

	Chirality get_chirality() const { return chirality; }
	void set_chirality(Chirality p_chirality) { chirality = p_chirality; }

	Vector3 get_selection_ray_origin() const { return selection_ray_origin; }
	void set_selection_ray_origin(const Vector3 &p_selection_ray_origin) {
		selection_ray_origin = p_selection_ray_origin;
	}

	Vector3 get_selection_ray_direction() const { return selection_ray_direction; }
	void set_selection_ray_direction(const Vector3 &p_selection_ray_direction) {
		selection_ray_direction = p_selection_ray_direction;
	}

	Vector3 get_input_device_pose_position() const { return input_device_pose_position; }
	void set_input_device_pose_position(const Vector3 &p_input_device_pose_position) {
		input_device_pose_position = p_input_device_pose_position;
	}

	Quaternion get_input_device_pose_rotation() const { return input_device_pose_rotation; }
	void set_input_device_pose_rotation(const Quaternion &p_input_device_pose_rotation) {
		input_device_pose_rotation = p_input_device_pose_rotation;
	}

	virtual String as_text() const override;
	virtual String _to_string() override;

private:
	int index = 0;

	Phase phase = PHASE_ACTIVE;
	Chirality chirality = CHIRALITY_LEFT;

	Vector3 selection_ray_origin;
	Vector3 selection_ray_direction;

	Vector3 input_device_pose_position;
	Quaternion input_device_pose_rotation;

	bool flags[FLAG_MAX] = {};
};

VARIANT_ENUM_CAST(InputEventSpatial::Flags);
VARIANT_ENUM_CAST(InputEventSpatial::Phase);
VARIANT_ENUM_CAST(InputEventSpatial::Chirality);
