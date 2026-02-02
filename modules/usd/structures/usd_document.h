/**************************************************************************/
/*  usd_document.h                                                        */
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

#include "usd_state.h"

#include "core/io/resource.h"

class Node;
class Node3D;
class ImporterMeshInstance3D;

// Forward-declare tinyusdz types used in the private interface.
// The actual tinyusdz headers are only included in the .cpp file
// to keep compile times low and avoid leaking thirdparty types
// into the Godot header graph.
namespace tinyusdz {
class Prim;
class Stage;
struct GeomMesh;
struct GeomCamera;
struct Material;
struct SphereLight;
struct DistantLight;
struct DomeLight;
struct DiskLight;
struct RectLight;
struct CylinderLight;
namespace value {
struct matrix4d;
} // namespace value
} // namespace tinyusdz

class USDDocument : public Resource {
	GDCLASS(USDDocument, Resource);

protected:
	static void _bind_methods();

public:
	// Main API: parse a USD file and populate USDState.
	Error append_from_file(const String &p_path, Ref<USDState> p_state,
			uint32_t p_flags = 0, const String &p_base_path = "");

	// Generate a Godot scene tree from the parsed USDState.
	Node *generate_scene(Ref<USDState> p_state, float p_bake_fps = 30.0,
			bool p_trimming = false, bool p_remove_immutable_tracks = true);

private:
	// Parsing pipeline: USD Stage -> USDState.
	Error _parse_scene(Ref<USDState> p_state, const tinyusdz::Stage &p_stage);
	void _parse_nodes_recursive(Ref<USDState> p_state, const tinyusdz::Prim &p_prim, int p_parent_idx);
	void _parse_mesh(Ref<USDState> p_state, const tinyusdz::GeomMesh &p_mesh, int p_node_idx);
	void _parse_material(Ref<USDState> p_state, const tinyusdz::Material &p_mat);
	void _parse_light(Ref<USDState> p_state, const tinyusdz::Prim &p_prim, int p_node_idx);
	void _parse_camera(Ref<USDState> p_state, const tinyusdz::GeomCamera &p_cam, int p_node_idx);

	// Scene generation: USDState -> Godot Node tree.
	void _generate_scene_recursive(Ref<USDState> p_state, int p_node_idx, Node *p_parent, Node *p_root);
	Node3D *_generate_node(Ref<USDState> p_state, int p_node_idx, Node *p_parent, Node *p_root);

	// Coordinate system helpers.
	static Transform3D _convert_transform(const tinyusdz::value::matrix4d &p_mat, float p_meters_per_unit, bool p_z_up);
	static Vector3 _convert_position(double p_x, double p_y, double p_z, float p_meters_per_unit, bool p_z_up);
};
