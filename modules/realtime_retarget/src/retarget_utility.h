/*************************************************************************/
/*  retarget_utility.h                                                   */
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

#ifndef RETARGET_UTILITY_H
#define RETARGET_UTILITY_H

#include "scene/3d/skeleton_3d.h"
#define REALTIME_RETARGET_META "_realtime_retarget" // Seems hackey, but static const is not working correctly in custom module.

class RetargetUtility : public Object {
	GDCLASS(RetargetUtility, Object);

public:
	enum TransformType {
		TYPE_ABSOLUTE,
		TYPE_LOCAL,
		TYPE_GLOBAL,
	};

protected:
	static void _bind_methods();

public:
	static Transform3D extract_global_transform(const Skeleton3D *p_skeleton, int p_bone_idx, Transform3D p_transform);
	static Vector3 extract_global_transform_position(const Skeleton3D *p_skeleton, int p_bone_idx, Vector3 p_position);
	static Quaternion extract_global_transform_rotation(const Skeleton3D *p_skeleton, int p_bone_idx, Quaternion p_rotation);
	static Vector3 extract_global_transform_scale(const Skeleton3D *p_skeleton, int p_bone_idx, Vector3 p_scale);
	static Transform3D global_transform_to_bone_pose(const Skeleton3D *p_skeleton, int p_bone_idx, Transform3D p_transform);
	static Vector3 global_transform_position_to_bone_pose(const Skeleton3D *p_skeleton, int p_bone_idx, Vector3 p_position);
	static Quaternion global_transform_rotation_to_bone_pose(const Skeleton3D *p_skeleton, int p_bone_idx, Quaternion p_rotation);
	static Vector3 global_transform_scale_to_bone_pose(const Skeleton3D *p_skeleton, int p_bone_idx, Vector3 p_scale);

	static Transform3D extract_local_transform(const Skeleton3D *p_skeleton, int p_bone_idx, Transform3D p_transform);
	static Vector3 extract_local_transform_position(const Skeleton3D *p_skeleton, int p_bone_idx, Vector3 p_position);
	static Quaternion extract_local_transform_rotation(const Skeleton3D *p_skeleton, int p_bone_idx, Quaternion p_rotation);
	static Vector3 extract_local_transform_scale(const Skeleton3D *p_skeleton, int p_bone_idx, Vector3 p_scale);
	static Transform3D local_transform_to_bone_pose(const Skeleton3D *p_skeleton, int p_bone_idx, Transform3D p_transform);
	static Vector3 local_transform_position_to_bone_pose(const Skeleton3D *p_skeleton, int p_bone_idx, Vector3 p_position);
	static Quaternion local_transform_rotation_to_bone_pose(const Skeleton3D *p_skeleton, int p_bone_idx, Quaternion p_rotation);
	static Vector3 local_transform_scale_to_bone_pose(const Skeleton3D *p_skeleton, int p_bone_idx, Vector3 p_scale);

	RetargetUtility();
	~RetargetUtility();
};

VARIANT_ENUM_CAST(RetargetUtility::TransformType);

#endif // RETARGET_UTILITY_H
