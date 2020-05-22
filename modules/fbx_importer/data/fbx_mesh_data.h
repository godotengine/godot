/*************************************************************************/
/*  fbx_mesh_data.h                                                      */
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

#ifndef EDITOR_SCENE_FBX_MESH_DATA_H
#define EDITOR_SCENE_FBX_MESH_DATA_H

#include "core/bind/core_bind.h"
#include "core/io/resource_importer.h"
#include "core/vector.h"
#include "editor/import/resource_importer_scene.h"
#include "editor/project_settings_editor.h"
#include "fbx_bone.h"
#include "fbx_node.h"
#include "fbx_skeleton.h"
#include "pivot_transform.h"
#include "scene/3d/mesh_instance.h"
#include "scene/3d/skeleton.h"
#include "scene/3d/spatial.h"
#include "scene/animation/animation_player.h"
#include "scene/resources/animation.h"
#include "scene/resources/surface_tool.h"
#include "tools/import_utils.h"

#include <assimp/matrix4x4.h>
#include <assimp/types.h>
#include <code/FBX/FBXDocument.h>
#include <code/FBX/FBXImportSettings.h>
#include <code/FBX/FBXMeshGeometry.h>
#include <code/FBX/FBXParser.h>
#include <code/FBX/FBXTokenizer.h>
#include <code/FBX/FBXUtil.h>



struct FBXMeshVertexData;
struct FBXBone;

struct FBXSplitBySurfaceVertexMapping {
	Vector<size_t> vertex_id;
	Vector<Vector2> uv_0, uv_1;
	Vector<Vector3> normals;
	Vector<Color> colors;

	void add_uv_0( Vector2 vec )
	{
		vec.y = 1.0f - vec.y;
		//print_verbose("added uv_0 " + vec);
		uv_0.push_back(vec);
	}

	void add_uv_1( Vector2 vec )
	{
		vec.y = 1.0f - vec.y;
		uv_1.push_back(vec);
	}

	Vector3 get_normal( int vertex_id, bool& found ) const
    {
		found = false;
		if( vertex_id < normals.size() )
        {
			found = true;
			return normals[vertex_id];
		}
		return Vector3();
	}

	Color get_colors( int vertex_id, bool& found ) const
    {
        found = false;
        if( vertex_id < colors.size() )
        {
            found = true;
            return colors[vertex_id];
        }
        return Color();
	}

	Vector2 get_uv_0( int vertex_id, bool& found) const
	{
		found = false;
		if( vertex_id < uv_0.size() )
		{
			found = true;
			return uv_0[vertex_id];
		}
		return Vector2();
	}

	Vector2 get_uv_1( int vertex_id, bool& found) const
	{
		found = false;
		if( vertex_id < uv_1.size() )
		{
			found = true;
			return uv_1[vertex_id];
		}
		return Vector2();
	}

	void GenerateSurfaceMaterial( Ref<SurfaceTool> st, size_t vertex_id )
	{
		bool uv_0 = false;
		bool uv_1 = false;
		bool normal_found = false;
		bool color_found = false;
		Vector2 uv_0_vec = get_uv_0(vertex_id, uv_0);
		Vector2 uv_1_vec = get_uv_1(vertex_id, uv_1);
		Vector3 normal = get_normal(vertex_id, normal_found);
		Color color = get_colors(vertex_id, color_found);
		if(uv_0) {
			//print_verbose("added uv_0 st " + uv_0_vec);
			st->add_uv(uv_0_vec);
		}
		if(uv_1)
		{
			//print_verbose("added uv_1 st " + uv_1_vec);
			st->add_uv2(uv_1_vec);
		}

		if(normal_found)
        {
			st->add_normal(normal);
		}

		if(color_found)
        {
			st->add_color(color);
		}
	}
};

struct VertexMapping : Reference {
	Vector<float> weights;
	Vector<Ref<FBXBone> > bones;

	/*** Will only add a vertex weight if it has been validated that it exists in godot **/
	void GetValidatedBoneWeightInfo(Vector<int> &out_bones, Vector<float> &out_weights);
};

struct FBXMeshVertexData : Reference {
	// vertex id, Weight Info
	// later: perf we can use array here
	Map<size_t, Ref<VertexMapping> > vertex_weights;

	// translate fbx mesh data from document context to FBX Mesh Geometry Context
	bool valid_weight_indexes = false;

	// basically this gives the correct ID for the vertex specified. so the weight data is correct for the meshes, as they're de-indexed.
	void FixWeightData(const Assimp::FBX::MeshGeometry *mesh_geometry) {
		if (!valid_weight_indexes && mesh_geometry) {
			Map<size_t, Ref<VertexMapping> > fixed_weight_info;
			for (Map<size_t, Ref<VertexMapping> >::Element *element = vertex_weights.front(); element; element = element->next()) {
				unsigned int count;
				const unsigned int *vert = mesh_geometry->ToOutputVertexIndex(element->key(), count);
				//  print_verbose("begin translation of weight information");
				if (vert != nullptr) {
					for (unsigned int x = 0; x < count; x++) {
						//                        print_verbose("input vertex: " + itos(element->key()) + ", output vert data: " + itos(vert[x]) +
						//                                      " count: " + itos(count));

						// write fixed weight info to the new temp array
						fixed_weight_info.insert(vert[x], element->value());
					}
				}
			}

			//    print_verbose("size of fixed weight info:" + itos(fixed_weight_info.size()));

			// destructive part of this operation is done here
			vertex_weights = fixed_weight_info;

			//  print_verbose("completed weight fixup");
			valid_weight_indexes = true;
		}
	}

	// verticies could go here
	// uvs could go here
	// normals could go here

	/* mesh maximum weight count */
	bool valid_weight_count = false;
	int max_weight_count = 0;
	uint64_t mesh_id; // fbx mesh id
	uint64_t armature_id;
	bool valid_armature_id = false;
	MeshInstance * godot_mesh_instance = nullptr;
};

#endif // EDITOR_SCENE_FBX_MESH_DATA_H