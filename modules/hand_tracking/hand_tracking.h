/**************************************************************************/
/*  hand_tracking.h                                                       */
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

// C bridge for visionOS hand tracking integration
// This header can be imported from Swift via bridging header

#ifdef __cplusplus
extern "C" {
#endif

#define GODOT_MAX_HAND_JOINTS 32

// Hand type enumeration
typedef enum {
	GODOT_HAND_NONE = 0,
	GODOT_HAND_LEFT = 1,
	GODOT_HAND_RIGHT = 2,
} godot_hand_type;

// Joint enumeration matching ARKit HandSkeleton joints
// Based on Apple's HandSkeleton.JointName
typedef enum {
	GODOT_HAND_JOINT_WRIST = 0,

	// Thumb joints
	GODOT_HAND_JOINT_THUMB_KNUCKLE = 1,
	GODOT_HAND_JOINT_THUMB_INTERMEDIATE = 2,
	GODOT_HAND_JOINT_THUMB_TIP = 3,

	// Index finger joints
	GODOT_HAND_JOINT_INDEX_KNUCKLE = 4,
	GODOT_HAND_JOINT_INDEX_INTERMEDIATE = 5,
	GODOT_HAND_JOINT_INDEX_DISTAL = 6,
	GODOT_HAND_JOINT_INDEX_TIP = 7,

	// Middle finger joints
	GODOT_HAND_JOINT_MIDDLE_KNUCKLE = 8,
	GODOT_HAND_JOINT_MIDDLE_INTERMEDIATE = 9,
	GODOT_HAND_JOINT_MIDDLE_DISTAL = 10,
	GODOT_HAND_JOINT_MIDDLE_TIP = 11,

	// Ring finger joints
	GODOT_HAND_JOINT_RING_KNUCKLE = 12,
	GODOT_HAND_JOINT_RING_INTERMEDIATE = 13,
	GODOT_HAND_JOINT_RING_DISTAL = 14,
	GODOT_HAND_JOINT_RING_TIP = 15,

	// Little finger joints
	GODOT_HAND_JOINT_LITTLE_KNUCKLE = 16,
	GODOT_HAND_JOINT_LITTLE_INTERMEDIATE = 17,
	GODOT_HAND_JOINT_LITTLE_DISTAL = 18,
	GODOT_HAND_JOINT_LITTLE_TIP = 19,

	GODOT_HAND_JOINT_COUNT = 20,
} godot_hand_joint_id;

// Single hand joint with position and orientation
typedef struct {
	float position[3]; // x, y, z (meters in AR session space)
	float orientation[4]; // quaternion (x, y, z, w)
	int joint_id; // godot_hand_joint_id
	int valid; // 0 = invalid, 1 = valid/tracked
} godot_hand_joint;

// Complete hand frame data for both hands
typedef struct {
	double timestamp_s; // timestamp in seconds

	godot_hand_joint left_joints[GODOT_MAX_HAND_JOINTS];
	int left_joint_count; // number of valid joints in left_joints

	godot_hand_joint right_joints[GODOT_MAX_HAND_JOINTS];
	int right_joint_count; // number of valid joints in right_joints
} godot_hand_frame;

/**
 * Called from Swift/visionOS with the latest hand tracking frame.
 * This function is the primary entry point from the native platform layer.
 *
 * @param frame Pointer to hand frame data containing joint positions and orientations
 */
void godot_visionos_set_hand_frame(const godot_hand_frame *frame);

#ifdef __cplusplus
}
#endif
