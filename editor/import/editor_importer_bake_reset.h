/*************************************************************************/
/*  editor_importer_bake_reset.h                                         */
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

#ifndef RESOURCE_IMPORTER_BAKE_RESET_H
#define RESOURCE_IMPORTER_BAKE_RESET_H

#include "scene/main/node.h"

class Skeleton3D;
class AnimationPlayer;
class BakeReset {
	struct BakeResetRestBone {
		Transform3D rest_local;
		Basis rest_delta;
		Vector3 loc;
		Vector<int> children;
	};

public:
	void _bake_animation_pose(Node *scene, const String &p_bake_anim);

private:
	void _fix_skeleton(Skeleton3D *p_skeleton, Map<StringName, BakeReset::BakeResetRestBone> &r_rest_bones);
	void _align_animations(AnimationPlayer *p_ap, const Map<StringName, BakeResetRestBone> &r_rest_bones);
	void _fetch_reset_animation(AnimationPlayer *p_ap, Map<StringName, BakeResetRestBone> &r_rest_bones, const String &p_bake_anim);
};
#endif
