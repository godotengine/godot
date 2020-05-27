/*************************************************************************/
/*  pivot_transform.cpp 	                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "pivot_transform.h"
#include <modules/fbx_importer/tools/import_utils.h>

void PivotTransform::ReadTransformChain() {
	const Assimp::FBX::PropertyTable &props = fbx_model->Props();
	const Assimp::FBX::Model::RotOrder &rot = fbx_model->RotationOrder();
	const Assimp::FBX::TransformInheritance &inheritType = fbx_model->InheritType();
	inherit_type = inheritType; // copy the inherit type we need it in the second step.
	print_verbose("Model: " + String(fbx_model->Name().c_str()) + " Has inherit type: " + itos(fbx_model->InheritType()));
	bool ok = false;
	pre_rotation = Quat();
	post_rotation = Quat();
	rotation = Quat();
	geometric_rotation = Quat();

	raw_pre_rotation = AssimpUtils::safe_import_vector3(Assimp::FBX::PropertyGet<Vector3>(props, "PreRotation", ok));
	if (ok) {
		pre_rotation = AssimpUtils::EulerToQuaternion(Assimp::FBX::Model::RotOrder_EulerXYZ, raw_pre_rotation);
		print_verbose("valid pre_rotation: " + raw_pre_rotation + " euler conversion: " + (pre_rotation.get_euler() * (180 / Math_PI)));
	}
	raw_post_rotation = AssimpUtils::safe_import_vector3(Assimp::FBX::PropertyGet<Vector3>(props, "PostRotation", ok));
	if (ok) {
		post_rotation = AssimpUtils::EulerToQuaternion(Assimp::FBX::Model::RotOrder_EulerXYZ, raw_post_rotation);
		print_verbose("valid post_rotation: " + raw_post_rotation + " euler conversion: " + (pre_rotation.get_euler() * (180 / Math_PI)));
	}
	const Vector3 &RotationPivot = AssimpUtils::safe_import_vector3(Assimp::FBX::PropertyGet<Vector3>(props, "RotationPivot", ok));
	if (ok) {
		rotation_pivot = AssimpUtils::FixAxisConversions(RotationPivot);
	}
	const Vector3 &RotationOffset = AssimpUtils::safe_import_vector3(Assimp::FBX::PropertyGet<Vector3>(props, "RotationOffset", ok));
	if (ok) {
		rotation_offset = AssimpUtils::FixAxisConversions(RotationOffset);
	}
	const Vector3 &ScalingOffset = AssimpUtils::safe_import_vector3(Assimp::FBX::PropertyGet<Vector3>(props, "ScalingOffset", ok));
	if (ok) {
		scaling_offset = AssimpUtils::FixAxisConversions(ScalingOffset);
	}
	const Vector3 &ScalingPivot = AssimpUtils::safe_import_vector3(Assimp::FBX::PropertyGet<Vector3>(props, "ScalingPivot", ok));
	if (ok) {
		scaling_pivot = AssimpUtils::FixAxisConversions(ScalingPivot);
	}
	const Vector3 &Translation = AssimpUtils::safe_import_vector3(Assimp::FBX::PropertyGet<Vector3>(props, "Lcl Translation", ok));
	if (ok) {
		translation = AssimpUtils::FixAxisConversions(Translation);
	}
	raw_rotation = AssimpUtils::safe_import_vector3(Assimp::FBX::PropertyGet<Vector3>(props, "Lcl Rotation", ok));
	if (ok) {
		rotation = AssimpUtils::EulerToQuaternion(rot, raw_rotation);
	}
	else
	{
		rotation = Quat();
	}

	const Vector3 &Scaling = AssimpUtils::safe_import_vector3(Assimp::FBX::PropertyGet<Vector3>(props, "Lcl Scaling", ok));
	if (ok) {
		scaling = Scaling;
	}
	else
	{
		scaling = Vector3(1,1,1);
	}

	const Vector3 &GeometricScaling = AssimpUtils::safe_import_vector3(Assimp::FBX::PropertyGet<Vector3>(props, "GeometricScaling", ok));
	if (ok) {
		geometric_scaling = GeometricScaling;
	}
	const Vector3 &GeometricRotation = AssimpUtils::safe_import_vector3(Assimp::FBX::PropertyGet<Vector3>(props, "GeometricRotation", ok));
	if (ok) {
		geometric_rotation = AssimpUtils::EulerToQuaternion(rot, GeometricRotation);
	}
	const Vector3 &GeometricTranslation = AssimpUtils::safe_import_vector3(Assimp::FBX::PropertyGet<Vector3>(props, "GeometricTranslation", ok));
	if (ok) {
		geometric_translation = AssimpUtils::FixAxisConversions(GeometricTranslation);
	}
}
void PivotTransform::ComputePivotTransform() {
	// in maya we use geometric_rotation too
	// print_verbose("pre_rotation : " + (pre_rotation.get_euler() * (180 / Math_PI)));
	// print_verbose("post_rotation : " + (post_rotation.get_euler() * (180 / Math_PI)));
	// print_verbose("rotation : " + (rotation.get_euler() * (180 / Math_PI)));
	// print_verbose("geometric_rotation : " + (geometric_rotation.get_euler() * (180 / Math_PI)));

	Transform T, Roff, Rp, Soff, Sp, S;

	// Here I assume this is the operation which needs done.
	// Its WorldTransform * V

	// Origin pivots
	T.set_origin(translation);
	Roff.set_origin(rotation_offset);
	Rp.set_origin(rotation_pivot);
	Soff.set_origin(scaling_offset);
	Sp.set_origin(scaling_pivot);

	// Scaling node
	S.scale(scaling);
	Local_Scaling_Matrix = S; // copy for when node / child is looking for the value of this.

	// Rotation pivots
	Transform Rpre = Transform(pre_rotation);
	Transform R = Transform(rotation);
	Transform Rpost = Transform(post_rotation);

	Transform parent_global_xform;
	Transform parent_local_scaling_m;

	if (parent_transform.is_valid()) {
		parent_global_xform = parent_transform->GlobalTransform;
		parent_local_scaling_m = parent_transform->Local_Scaling_Matrix;
	}

	if(inherit_type == Assimp::FBX::Transform_RSrs)	{
		// Local xform data - pivoted
		LocalTransform = T * Roff * Rp * Rpre * R * Rpost.inverse() * Rp.inverse() * Soff * Sp * S * Sp.inverse();

		Transform local_translation = Transform(Basis(), LocalTransform.origin);
		Transform global_translation_pivoted = parent_global_xform * LocalTransform;
		GlobalTransform = Transform();
		GlobalTransform = global_translation_pivoted;
		AssimpUtils::debug_xform("a) local translation" ,local_translation);
		AssimpUtils::debug_xform("b) parent_global_xform" , parent_global_xform);
		AssimpUtils::debug_xform("b&c) global_translation_pivoted", global_translation_pivoted);
		AssimpUtils::debug_xform("result GlobalTransform", GlobalTransform);
	} else {
		// rotation - inherit type shearing handler (pre-rotation, post-rotation handler)
		Transform local_rotation_m, parent_global_rotation_m;
		AssimpUtils::debug_xform("parent global xform", parent_global_xform);
		Quat parent_global_rotation = parent_global_xform.basis.get_rotation_quat();
		parent_global_rotation_m.basis.set_quat(parent_global_rotation);
		local_rotation_m = Rpre * R * Rpost;

		// translation / scaling - Inherit type shearing handler
		Transform local_shear_scaling, parent_shear_scaling, parent_shear_rotation, parent_shear_translation;
		Vector3 parent_translation = parent_global_xform.get_origin();
		parent_shear_translation.origin = parent_translation;
		parent_shear_rotation = parent_shear_translation.inverse() * parent_global_xform;
		parent_shear_scaling = parent_global_rotation_m.inverse() * parent_shear_rotation;
		local_shear_scaling = S;

		// Inherit type handler - we don't care about T here, just reordering RSrs etc.
		Transform global_rotation_scale;
		if (inherit_type == Assimp::FBX::Transform_RrSs) {
			global_rotation_scale = parent_global_rotation_m * local_rotation_m * parent_shear_scaling * local_shear_scaling;
		} else if (inherit_type == Assimp::FBX::Transform_Rrs) {
			Transform parent_global_shear_m_noLocal = parent_shear_scaling * parent_local_scaling_m.inverse();
			global_rotation_scale = parent_global_rotation_m * local_rotation_m * parent_global_shear_m_noLocal * local_shear_scaling;
		}

		// Local xform data - pivoted
		LocalTransform = T * Roff * Rp * Rpre * R * Rpost.inverse() * Rp.inverse() * Soff * Sp * S * Sp.inverse();

		Transform local_translation = Transform(Basis(), LocalTransform.origin);
		Transform global_translation_pivoted = parent_global_xform * local_translation;
		GlobalTransform = Transform();
		GlobalTransform = global_translation_pivoted * global_rotation_scale;

		AssimpUtils::debug_xform("a) local translation" ,local_translation);
		AssimpUtils::debug_xform("b) parent_global_xform" , parent_global_xform);
		AssimpUtils::debug_xform("b&c) global_translation_pivoted", global_translation_pivoted);
		AssimpUtils::debug_xform("c) global_rotation_scale", global_rotation_scale);
		AssimpUtils::debug_xform("result GlobalTransform", GlobalTransform);
	}

	print_verbose("---------------------------------------------------------------");
}
void PivotTransform::Execute() {
	ReadTransformChain();
	ComputePivotTransform();

	AssimpUtils::debug_xform("global xform: ", GlobalTransform);
	computed_global_xform = true;
}
