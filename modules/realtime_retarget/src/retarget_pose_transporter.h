/*************************************************************************/
/*  retarget_pose_transporter.h                                          */
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

#ifndef RETARGET_POSE_TRANSPORTER_H
#define RETARGET_POSE_TRANSPORTER_H

#include "retarget_profile.h"
#include "scene/3d/skeleton_3d.h"

class RetargetPoseTransporter : public Node {
	GDCLASS(RetargetPoseTransporter, Node);

public:
	enum AnimationProcessCallback {
		ANIMATION_PROCESS_PHYSICS,
		ANIMATION_PROCESS_IDLE,
		ANIMATION_PROCESS_MANUAL,
	};

private:
	NodePath source_skeleton_path;
	NodePath target_skeleton_path;
	Ref<RetargetProfile> profile;
	bool active = false;
	bool position_enabled = true;
	bool rotation_enabled = true;
	bool scale_enabled = true;
	AnimationProcessCallback process_callback = ANIMATION_PROCESS_IDLE;

	void _process();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_source_skeleton(const NodePath &p_skeleton);
	NodePath get_source_skeleton() const;

	void set_target_skeleton(const NodePath &p_skeleton);
	NodePath get_target_skeleton() const;

	void set_profile(const Ref<RetargetProfile> &p_profile);
	Ref<RetargetProfile> get_profile() const;

	void set_active(bool p_active);
	bool is_active() const;

	void set_position_enabled(bool p_enabled);
	bool is_position_enabled() const;

	void set_rotation_enabled(bool p_enabled);
	bool is_rotation_enabled() const;

	void set_scale_enabled(bool p_enabled);
	bool is_scale_enabled() const;

	void set_process_callback(AnimationProcessCallback p_mode);
	AnimationProcessCallback get_process_callback() const;

	void advance();

	RetargetPoseTransporter();
	~RetargetPoseTransporter();
};

VARIANT_ENUM_CAST(RetargetPoseTransporter::AnimationProcessCallback)

#endif // RETARGET_POSE_TRANSPORTER_H
