/*************************************************************************/
/*  fbx_node.h                                                           */
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

#ifndef FBX_NODE_H
#define FBX_NODE_H

#include "fbx_skeleton.h"
#include "model_abstraction.h"
#include "pivot_transform.h"

#include "fbx_parser/FBXDocument.h"

class Node3D;
struct PivotTransform;

struct FBXNode : RefCounted, ModelAbstraction {
	uint64_t current_node_id = 0;
	String node_name = String();
	Node3D *godot_node = nullptr;

	// used to parent the skeleton once the tree is built.
	Ref<FBXSkeleton> skeleton_node = Ref<FBXSkeleton>();

	void set_parent(Ref<FBXNode> p_parent) {
		fbx_parent = p_parent;
	}

	void set_pivot_transform(Ref<PivotTransform> p_pivot_transform) {
		pivot_transform = p_pivot_transform;
	}

	Ref<PivotTransform> pivot_transform = Ref<PivotTransform>(); // local and global xform data
	Ref<FBXNode> fbx_parent = Ref<FBXNode>(); // parent node
};

#endif // FBX_NODE_H
