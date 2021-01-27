/*************************************************************************/
/*  gltf_node.h                                                          */
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

#ifndef GLTF_NODE_H
#define GLTF_NODE_H

#include "core/io/resource.h"
#include "gltf_document.h"

class GLTFNode : public Resource {
	GDCLASS(GLTFNode, Resource);
	friend class GLTFDocument;
	friend class PackedSceneGLTF;

private:
	// matrices need to be transformed to this
	GLTFNodeIndex parent = -1;
	int height = -1;
	Transform xform;
	GLTFMeshIndex mesh = -1;
	GLTFCameraIndex camera = -1;
	GLTFSkinIndex skin = -1;
	GLTFSkeletonIndex skeleton = -1;
	bool joint = false;
	Vector3 translation;
	Quat rotation;
	Vector3 scale = Vector3(1, 1, 1);
	Vector<int> children;
	GLTFNodeIndex fake_joint_parent = -1;
	GLTFLightIndex light = -1;

protected:
	static void _bind_methods();

public:
	GLTFNodeIndex get_parent();
	void set_parent(GLTFNodeIndex p_parent);

	int get_height();
	void set_height(int p_height);

	Transform get_xform();
	void set_xform(Transform p_xform);

	GLTFMeshIndex get_mesh();
	void set_mesh(GLTFMeshIndex p_mesh);

	GLTFCameraIndex get_camera();
	void set_camera(GLTFCameraIndex p_camera);

	GLTFSkinIndex get_skin();
	void set_skin(GLTFSkinIndex p_skin);

	GLTFSkeletonIndex get_skeleton();
	void set_skeleton(GLTFSkeletonIndex p_skeleton);

	bool get_joint();
	void set_joint(bool p_joint);

	Vector3 get_translation();
	void set_translation(Vector3 p_translation);

	Quat get_rotation();
	void set_rotation(Quat p_rotation);

	Vector3 get_scale();
	void set_scale(Vector3 p_scale);

	Vector<int> get_children();
	void set_children(Vector<int> p_children);

	GLTFNodeIndex get_fake_joint_parent();
	void set_fake_joint_parent(GLTFNodeIndex p_fake_joint_parent);

	GLTFLightIndex get_light();
	void set_light(GLTFLightIndex p_light);
};
#endif // GLTF_NODE_H
