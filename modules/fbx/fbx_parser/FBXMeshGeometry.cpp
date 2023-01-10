/**************************************************************************/
/*  FBXMeshGeometry.cpp                                                   */
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

/*
Open Asset Import Library (assimp)
----------------------------------------------------------------------

Copyright (c) 2006-2019, assimp team


All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the
following conditions are met:

* Redistributions of source code must retain the above
  copyright notice, this list of conditions and the
  following disclaimer.

* Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the
  following disclaimer in the documentation and/or other
  materials provided with the distribution.

* Neither the name of the assimp team, nor the names of its
  contributors may be used to endorse or promote products
  derived from this software without specific prior
  written permission of the assimp team.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

----------------------------------------------------------------------
*/

/** @file  FBXMeshGeometry.cpp
 *  @brief Assimp::FBX::MeshGeometry implementation
 */

#include <functional>

#include "FBXDocument.h"
#include "FBXDocumentUtil.h"
#include "FBXImportSettings.h"
#include "FBXMeshGeometry.h"
#include "core/math/vector3.h"

namespace FBXDocParser {

using namespace Util;

// ------------------------------------------------------------------------------------------------
Geometry::Geometry(uint64_t id, const ElementPtr element, const std::string &name, const Document &doc) :
		Object(id, element, name), skin() {
	const std::vector<const Connection *> &conns = doc.GetConnectionsByDestinationSequenced(ID(), "Deformer");
	for (const Connection *con : conns) {
		const Skin *sk = ProcessSimpleConnection<Skin>(*con, false, "Skin -> Geometry", element);
		if (sk) {
			skin = sk;
		}
		const BlendShape *bsp = ProcessSimpleConnection<BlendShape>(*con, false, "BlendShape -> Geometry",
				element);
		if (bsp) {
			blendShapes.push_back(bsp);
		}
	}
}

// ------------------------------------------------------------------------------------------------
Geometry::~Geometry() {
	// empty
}

// ------------------------------------------------------------------------------------------------
const std::vector<const BlendShape *> &Geometry::get_blend_shapes() const {
	return blendShapes;
}

// ------------------------------------------------------------------------------------------------
const Skin *Geometry::DeformerSkin() const {
	return skin;
}

// ------------------------------------------------------------------------------------------------
MeshGeometry::MeshGeometry(uint64_t id, const ElementPtr element, const std::string &name, const Document &doc) :
		Geometry(id, element, name, doc) {
	print_verbose("mesh name: " + String(name.c_str()));

	ScopePtr sc = element->Compound();
	ERR_FAIL_COND_MSG(sc == nullptr, "failed to read geometry, prevented crash");
	ERR_FAIL_COND_MSG(!HasElement(sc, "Vertices"), "Detected mesh with no vertices, didn't populate the mesh");

	// must have Mesh elements:
	const ElementPtr Vertices = GetRequiredElement(sc, "Vertices", element);
	const ElementPtr PolygonVertexIndex = GetRequiredElement(sc, "PolygonVertexIndex", element);

	if (HasElement(sc, "Edges")) {
		const ElementPtr element_edges = GetRequiredElement(sc, "Edges", element);
		ParseVectorDataArray(m_edges, element_edges);
	}

	// read mesh data into arrays
	ParseVectorDataArray(m_vertices, Vertices);
	ParseVectorDataArray(m_face_indices, PolygonVertexIndex);

	ERR_FAIL_COND_MSG(m_vertices.empty(), "mesh with no vertices in FBX file, did you mean to delete it?");
	ERR_FAIL_COND_MSG(m_face_indices.empty(), "mesh has no faces, was this intended?");

	// Retrieve layer elements, for all of the mesh
	const ElementCollection &Layer = sc->GetCollection("Layer");

	// Store all layers
	std::vector<std::tuple<int, std::string>> valid_layers;

	// now read the sub mesh information from the geometry (normals, uvs, etc)
	for (ElementMap::const_iterator it = Layer.first; it != Layer.second; ++it) {
		const ScopePtr layer = GetRequiredScope(it->second);
		const ElementCollection &LayerElement = layer->GetCollection("LayerElement");
		for (ElementMap::const_iterator eit = LayerElement.first; eit != LayerElement.second; ++eit) {
			std::string layer_name = eit->first;
			ElementPtr element_layer = eit->second;
			const ScopePtr layer_element = GetRequiredScope(element_layer);

			// Actual usable 'type' LayerElementUV, LayerElementNormal, etc
			const ElementPtr Type = GetRequiredElement(layer_element, "Type");
			const ElementPtr TypedIndex = GetRequiredElement(layer_element, "TypedIndex");
			const std::string &type = ParseTokenAsString(GetRequiredToken(Type, 0));
			const int typedIndex = ParseTokenAsInt(GetRequiredToken(TypedIndex, 0));

			// we only need the layer name and the typed index.
			valid_layers.push_back(std::tuple<int, std::string>(typedIndex, type));
		}
	}

	// get object / mesh directly from the FBX by the element ID.
	const ScopePtr top = GetRequiredScope(element);

	// iterate over all layers for the mesh (uvs, normals, smoothing groups, colors, etc)
	for (size_t x = 0; x < valid_layers.size(); x++) {
		const int layer_id = std::get<0>(valid_layers[x]);
		const std::string &layer_type_name = std::get<1>(valid_layers[x]);

		// Get collection of elements from the XLayerMap (example: LayerElementUV)
		// this must contain our proper elements.

		// This is stupid, because it means we select them ALL not just the one we want.
		// but it's fine we can match by id.

		const ElementCollection &candidates = top->GetCollection(layer_type_name);

		ElementMap::const_iterator iter;
		for (iter = candidates.first; iter != candidates.second; ++iter) {
			const ScopePtr layer_scope = GetRequiredScope(iter->second);
			TokenPtr layer_token = GetRequiredToken(iter->second, 0);
			const int index = ParseTokenAsInt(layer_token);

			ERR_FAIL_COND_MSG(layer_scope == nullptr, "prevented crash, layer scope is invalid");

			if (index == layer_id) {
				const std::string &MappingInformationType = ParseTokenAsString(GetRequiredToken(
						GetRequiredElement(layer_scope, "MappingInformationType"), 0));

				const std::string &ReferenceInformationType = ParseTokenAsString(GetRequiredToken(
						GetRequiredElement(layer_scope, "ReferenceInformationType"), 0));

				if (layer_type_name == "LayerElementUV") {
					if (index == 0) {
						m_uv_0 = resolve_vertex_data_array<Vector2>(layer_scope, MappingInformationType, ReferenceInformationType, "UV");
					} else if (index == 1) {
						m_uv_1 = resolve_vertex_data_array<Vector2>(layer_scope, MappingInformationType, ReferenceInformationType, "UV");
					}
				} else if (layer_type_name == "LayerElementMaterial") {
					m_material_allocation_ids = resolve_vertex_data_array<int>(layer_scope, MappingInformationType, ReferenceInformationType, "Materials");
				} else if (layer_type_name == "LayerElementNormal") {
					m_normals = resolve_vertex_data_array<Vector3>(layer_scope, MappingInformationType, ReferenceInformationType, "Normals");
				} else if (layer_type_name == "LayerElementColor") {
					m_colors = resolve_vertex_data_array<Color>(layer_scope, MappingInformationType, ReferenceInformationType, "Colors", "ColorIndex");
					// NOTE: this is a useful sanity check to ensure you're getting any color data which is not default.
					//					const Color first_color_check = m_colors.data[0];
					//					bool colors_are_all_the_same = true;
					//					size_t i = 1;
					//					for(i = 1; i < m_colors.data.size(); i++)
					//					{
					//						const Color current_color = m_colors.data[i];
					//						if(current_color.is_equal_approx(first_color_check))
					//						{
					//							continue;
					//						}
					//						else
					//						{
					//							colors_are_all_the_same = false;
					//							break;
					//						}
					//					}
					//
					//					if(colors_are_all_the_same)
					//					{
					//						print_error("Color serialisation is not working for vertex colors some should be different in the test asset.");
					//					}
					//					else
					//					{
					//						print_verbose("Color array has unique colors at index: " + itos(i));
					//					}
				}
			}
		}
	}

	print_verbose("Mesh statistics \nuv_0: " + m_uv_0.debug_info() + "\nuv_1: " + m_uv_1.debug_info() + "\nvertices: " + itos(m_vertices.size()));

	// Compose the edge of the mesh.
	// You can see how the edges are stored into the FBX here: https://gist.github.com/AndreaCatania/da81840f5aa3b2feedf189e26c5a87e6
	for (size_t i = 0; i < m_edges.size(); i += 1) {
		ERR_FAIL_INDEX_MSG((size_t)m_edges[i], m_face_indices.size(), "The edge is pointing to a weird location in the face indices. The FBX is corrupted.");
		int polygon_vertex_0 = m_face_indices[m_edges[i]];
		int polygon_vertex_1;
		if (polygon_vertex_0 < 0) {
			// The polygon_vertex_0 points to the end of a polygon, so it's
			// connected with the beginning of polygon in the edge list.

			// Fist invert the vertex.
			polygon_vertex_0 = ~polygon_vertex_0;

			// Search the start vertex of the polygon.
			// Iterate from the polygon_vertex_index backward till the start of
			// the polygon is found.
			ERR_FAIL_COND_MSG(m_edges[i] - 1 < 0, "The polygon is not yet started and we already need the final vertex. This FBX is corrupted.");
			bool found_it = false;
			for (int x = m_edges[i] - 1; x >= 0; x -= 1) {
				if (x == 0) {
					// This for sure is the start.
					polygon_vertex_1 = m_face_indices[x];
					found_it = true;
					break;
				} else if (m_face_indices[x] < 0) {
					// This is the end of the previous polygon, so the next is
					// the start of the polygon we need.
					polygon_vertex_1 = m_face_indices[x + 1];
					found_it = true;
					break;
				}
			}
			// As the algorithm above, this check is useless. Because the first
			// ever vertex is always considered the beginning of a polygon.
			ERR_FAIL_COND_MSG(found_it == false, "Was not possible to find the first vertex of this polygon. FBX file is corrupted.");

		} else {
			ERR_FAIL_INDEX_MSG((size_t)(m_edges[i] + 1), m_face_indices.size(), "FBX The other FBX edge seems to point to an invalid vertices. This FBX file is corrupted.");
			// Take the next vertex
			polygon_vertex_1 = m_face_indices[m_edges[i] + 1];
		}

		if (polygon_vertex_1 < 0) {
			// We don't care if the `polygon_vertex_1` is the end of the polygon,
			//  for `polygon_vertex_1` so we can just invert it.
			polygon_vertex_1 = ~polygon_vertex_1;
		}

		ERR_FAIL_COND_MSG(polygon_vertex_0 == polygon_vertex_1, "The vertices of this edge can't be the same, Is this a point???. This FBX file is corrupted.");

		// Just create the edge.
		edge_map.push_back({ polygon_vertex_0, polygon_vertex_1 });
	}
}

MeshGeometry::~MeshGeometry() {
	// empty
}

const std::vector<Vector3> &MeshGeometry::get_vertices() const {
	return m_vertices;
}

const std::vector<MeshGeometry::Edge> &MeshGeometry::get_edge_map() const {
	return edge_map;
}

const std::vector<int> &MeshGeometry::get_polygon_indices() const {
	return m_face_indices;
}

const std::vector<int> &MeshGeometry::get_edges() const {
	return m_edges;
}

const MeshGeometry::MappingData<Vector3> &MeshGeometry::get_normals() const {
	return m_normals;
}

const MeshGeometry::MappingData<Vector2> &MeshGeometry::get_uv_0() const {
	//print_verbose("get uv_0 " + m_uv_0.debug_info() );
	return m_uv_0;
}

const MeshGeometry::MappingData<Vector2> &MeshGeometry::get_uv_1() const {
	//print_verbose("get uv_1 " + m_uv_1.debug_info() );
	return m_uv_1;
}

const MeshGeometry::MappingData<Color> &MeshGeometry::get_colors() const {
	return m_colors;
}

const MeshGeometry::MappingData<int> &MeshGeometry::get_material_allocation_id() const {
	return m_material_allocation_ids;
}

int MeshGeometry::get_edge_id(const std::vector<Edge> &p_map, int p_vertex_a, int p_vertex_b) {
	for (size_t i = 0; i < p_map.size(); i += 1) {
		if ((p_map[i].vertex_0 == p_vertex_a && p_map[i].vertex_1 == p_vertex_b) || (p_map[i].vertex_1 == p_vertex_a && p_map[i].vertex_0 == p_vertex_b)) {
			return i;
		}
	}
	return -1;
}

MeshGeometry::Edge MeshGeometry::get_edge(const std::vector<Edge> &p_map, int p_id) {
	ERR_FAIL_INDEX_V_MSG((size_t)p_id, p_map.size(), Edge({ -1, -1 }), "ID not found.");
	return p_map[p_id];
}

template <class T>
MeshGeometry::MappingData<T> MeshGeometry::resolve_vertex_data_array(
		const ScopePtr source,
		const std::string &MappingInformationType,
		const std::string &ReferenceInformationType,
		const std::string &dataElementName,
		const std::string &indexOverride) {
	ERR_FAIL_COND_V_MSG(source == nullptr, MappingData<T>(), "Invalid scope operator preventing memory corruption");

	// UVIndex, MaterialIndex, NormalIndex, etc..
	std::string indexDataElementName;

	if (indexOverride != "") {
		// Colors should become ColorIndex
		indexDataElementName = indexOverride;
	} else {
		// Some indexes will exist.
		indexDataElementName = dataElementName + "Index";
	}

	// goal: expand everything to be per vertex

	ReferenceType l_ref_type = ReferenceType::direct;

	// Read the reference type into the enumeration
	if (ReferenceInformationType == "IndexToDirect") {
		l_ref_type = ReferenceType::index_to_direct;
	} else if (ReferenceInformationType == "Index") {
		// set non legacy index to direct mapping
		l_ref_type = ReferenceType::index;
	} else if (ReferenceInformationType == "Direct") {
		l_ref_type = ReferenceType::direct;
	} else {
		ERR_FAIL_V_MSG(MappingData<T>(), "invalid reference type has the FBX format changed?");
	}

	MapType l_map_type = MapType::none;

	if (MappingInformationType == "None") {
		l_map_type = MapType::none;
	} else if (MappingInformationType == "ByVertice") {
		l_map_type = MapType::vertex;
	} else if (MappingInformationType == "ByPolygonVertex") {
		l_map_type = MapType::polygon_vertex;
	} else if (MappingInformationType == "ByPolygon") {
		l_map_type = MapType::polygon;
	} else if (MappingInformationType == "ByEdge") {
		l_map_type = MapType::edge;
	} else if (MappingInformationType == "AllSame") {
		l_map_type = MapType::all_the_same;
	} else {
		print_error("invalid mapping type: " + String(MappingInformationType.c_str()));
	}

	// create mapping data
	MeshGeometry::MappingData<T> tempData;
	tempData.map_type = l_map_type;
	tempData.ref_type = l_ref_type;

	// parse data into array
	ParseVectorDataArray(tempData.data, GetRequiredElement(source, dataElementName));

	// index array won't always exist
	const ElementPtr element = GetOptionalElement(source, indexDataElementName);
	if (element) {
		ParseVectorDataArray(tempData.index, element);
	}

	return tempData;
}
// ------------------------------------------------------------------------------------------------
ShapeGeometry::ShapeGeometry(uint64_t id, const ElementPtr element, const std::string &name, const Document &doc) :
		Geometry(id, element, name, doc) {
	const ScopePtr sc = element->Compound();
	if (nullptr == sc) {
		DOMError("failed to read Geometry object (class: Shape), no data scope found");
	}
	const ElementPtr Indexes = GetRequiredElement(sc, "Indexes", element);
	const ElementPtr Normals = GetRequiredElement(sc, "Normals", element);
	const ElementPtr Vertices = GetRequiredElement(sc, "Vertices", element);
	ParseVectorDataArray(m_indices, Indexes);
	ParseVectorDataArray(m_vertices, Vertices);
	ParseVectorDataArray(m_normals, Normals);
}

// ------------------------------------------------------------------------------------------------
ShapeGeometry::~ShapeGeometry() {
	// empty
}
// ------------------------------------------------------------------------------------------------
const std::vector<Vector3> &ShapeGeometry::GetVertices() const {
	return m_vertices;
}
// ------------------------------------------------------------------------------------------------
const std::vector<Vector3> &ShapeGeometry::GetNormals() const {
	return m_normals;
}
// ------------------------------------------------------------------------------------------------
const std::vector<unsigned int> &ShapeGeometry::GetIndices() const {
	return m_indices;
}
// ------------------------------------------------------------------------------------------------
LineGeometry::LineGeometry(uint64_t id, const ElementPtr element, const std::string &name, const Document &doc) :
		Geometry(id, element, name, doc) {
	const ScopePtr sc = element->Compound();
	if (!sc) {
		DOMError("failed to read Geometry object (class: Line), no data scope found");
	}
	const ElementPtr Points = GetRequiredElement(sc, "Points", element);
	const ElementPtr PointsIndex = GetRequiredElement(sc, "PointsIndex", element);
	ParseVectorDataArray(m_vertices, Points);
	ParseVectorDataArray(m_indices, PointsIndex);
}

// ------------------------------------------------------------------------------------------------
LineGeometry::~LineGeometry() {
	// empty
}
// ------------------------------------------------------------------------------------------------
const std::vector<Vector3> &LineGeometry::GetVertices() const {
	return m_vertices;
}
// ------------------------------------------------------------------------------------------------
const std::vector<int> &LineGeometry::GetIndices() const {
	return m_indices;
}

} // namespace FBXDocParser
