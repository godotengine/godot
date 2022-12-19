/*************************************************************************/
/*  node_3d_editor_selected_item.h                                       */
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

#ifndef NODE_3D_EDITOR_SELECTED_ITEM_H
#define NODE_3D_EDITOR_SELECTED_ITEM_H

#include "node_3d_editor_gizmos.h"

class Node3DEditorSelectedItem : public Object {
	GDCLASS(Node3DEditorSelectedItem, Object);

public:
	AABB aabb;
	Transform3D original; // original location when moving
	Transform3D original_local;
	Transform3D last_xform; // last transform
	bool last_xform_dirty;
	Node3D *sp = nullptr;
	RID sbox_instance;
	RID sbox_instance_offset;
	RID sbox_instance_xray;
	RID sbox_instance_xray_offset;
	Ref<EditorNode3DGizmo> gizmo;
	HashMap<int, Transform3D> subgizmos; // map ID -> initial transform

	Node3DEditorSelectedItem() {
		sp = nullptr;
		last_xform_dirty = true;
	}
	~Node3DEditorSelectedItem();
};

#endif // NODE_3D_EDITOR_SELECTED_ITEM_H
