/**************************************************************************/
/*  usd_state.h                                                           */
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

#include "core/io/resource.h"
#include "scene/resources/material.h"
#include "scene/resources/texture.h"
#include "scene/resources/3d/importer_mesh.h"

#include "usd_node.h"
#include "usd_mesh.h"
#include "usd_material.h"
#include "usd_light.h"
#include "usd_camera.h"
#include "usd_skeleton.h"
#include "usd_animation.h"

class ImporterMeshInstance3D;

class USDState : public Resource {
	GDCLASS(USDState, Resource);
	friend class USDDocument;

protected:
	static void _bind_methods();

	// File info.
	String base_path;
	String filename;
	String scene_name;

	// Parsed data arrays.
	Vector<Ref<USDNode>> nodes;
	Vector<int> root_nodes;
	Vector<Ref<USDMesh>> meshes;
	Vector<Ref<USDMaterial>> materials;
	Vector<Ref<USDLight>> lights;
	Vector<Ref<USDCamera>> cameras;
	Vector<Ref<USDSkeleton>> skeletons;
	Vector<Ref<USDAnimation>> animations;
	Vector<Ref<Texture2D>> images;

	// Scene generation mapping.
	HashMap<int, Node *> scene_nodes;
	HashMap<int, ImporterMeshInstance3D *> scene_mesh_instances;

	// Import options.
	double bake_fps = 30.0;
	bool create_animations = true;

	// Stage metadata.
	float meters_per_unit = 0.01f; // USD default: centimeters.
	bool up_axis_is_z = true; // USD default: Z-up.

public:
	String get_base_path() const;
	void set_base_path(const String &p_base_path);

	String get_filename() const;
	void set_filename(const String &p_filename);

	String get_scene_name() const;
	void set_scene_name(const String &p_scene_name);

	PackedInt32Array get_root_nodes() const;
	void set_root_nodes(const PackedInt32Array &p_root_nodes);

	double get_bake_fps() const;
	void set_bake_fps(double p_bake_fps);

	bool get_create_animations() const;
	void set_create_animations(bool p_create_animations);

	float get_meters_per_unit() const;
	void set_meters_per_unit(float p_meters_per_unit);

	bool get_up_axis_is_z() const;
	void set_up_axis_is_z(bool p_up_axis_is_z);

	int get_node_count() const;
	Ref<USDNode> get_node(int p_index) const;

	int get_mesh_count() const;
	Ref<USDMesh> get_mesh(int p_index) const;

	int get_material_count() const;
	Ref<USDMaterial> get_material(int p_index) const;

	int get_light_count() const;
	Ref<USDLight> get_light(int p_index) const;

	int get_camera_count() const;
	Ref<USDCamera> get_camera(int p_index) const;

	int get_skeleton_count() const;
	Ref<USDSkeleton> get_skeleton(int p_index) const;

	int get_animation_count() const;
	Ref<USDAnimation> get_animation(int p_index) const;

	int get_image_count() const;
	Ref<Texture2D> get_image(int p_index) const;

	Node *get_scene_node(int p_node_index) const;
};
