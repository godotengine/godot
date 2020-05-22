/*************************************************************************/
/*  fbx_mesh_data.cpp			                                         */
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

#include "fbx_mesh_data.h"

void VertexMapping::GetValidatedBoneWeightInfo(Vector<int> &out_bones, Vector<float> &out_weights) {
	ERR_FAIL_COND_MSG(bones.size() != weights.size(), "[doc] error unable to handle incorrect bone weight info");
	ERR_FAIL_COND_MSG(out_bones.size() > 0 && out_weights.size() > 0, "[doc] error input must be empty before using this function, accidental re-use?");
	for (int idx = 0; idx < weights.size(); idx++) {
		Ref<FBXBone> bone = bones[idx];
		float weight = weights[idx];
		if (bone.is_valid()) {
			out_bones.push_back(bone->godot_bone_id);
			out_weights.push_back(weight);
			//print_verbose("[" + itos(idx) + "] valid bone weight: " + itos(bone->godot_bone_id) + " weight: " + rtos(weight));
		} else {
			out_bones.push_back(0);
			out_weights.push_back(0);
			if (bone.is_valid()) {
				ERR_PRINT("skeleton misconfigured");
			} else {
				//print_verbose("[" + itos(idx) + "] fake weight: 0");
			}
		}
	}
}
