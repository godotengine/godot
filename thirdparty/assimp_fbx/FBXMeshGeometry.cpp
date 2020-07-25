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

namespace Assimp {
namespace FBX {

using namespace Util;

// ------------------------------------------------------------------------------------------------
Geometry::Geometry(uint64_t id, const Element &element, const std::string &name, const Document &doc) :
		Object(id, element, name), skin() {
	const std::vector<const Connection *> &conns = doc.GetConnectionsByDestinationSequenced(ID(), "Deformer");
	for (const Connection *con : conns) {
		const Skin *const sk = ProcessSimpleConnection<Skin>(*con, false, "Skin -> Geometry", element);
		if (sk) {
			skin = sk;
		}
		const BlendShape *const bsp = ProcessSimpleConnection<BlendShape>(*con, false, "BlendShape -> Geometry",
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
MeshGeometry::MeshGeometry(uint64_t id, const Element &element, const std::string &name, const Document &doc) :
		Geometry(id, element, name, doc) {
	const Scope *sc = element.Compound();
	if (!sc) {
		DOMError("failed to read Geometry object (class: Mesh), no data scope found");
	}

	if (!HasElement(*sc, "Vertices")) {
		return; // this happened!
	}

	// must have Mesh elements:
	const Element &Vertices = GetRequiredElement(*sc, "Vertices", &element);
	const Element &PolygonVertexIndex = GetRequiredElement(*sc, "PolygonVertexIndex", &element);

	if (HasElement(*sc, "Edges")) {
		const Element &element_edges = GetRequiredElement(*sc, "Edges", &element);
		ParseVectorDataArray(m_edges, element_edges);
	}

	// optional Mesh elements:
	const ElementCollection &Layer = sc->GetCollection("Layer");

	// read mesh data into arrays
	// todo: later we can actually store arrays for these :)
	// and not vector3
	ParseVectorDataArray(m_vertices, Vertices);
	ParseVectorDataArray(m_face_indices, PolygonVertexIndex);

	if (m_vertices.empty()) {
		print_error("encountered mesh with no vertices");
	}

	if (m_face_indices.empty()) {
		print_error("encountered mesh with no faces");
	}

	// now read the sub mesh information from the geometry (normals, uvs, etc)
	for (ElementMap::const_iterator it = Layer.first; it != Layer.second; ++it) {
		//const TokenList &tokens = (*it).second->Tokens();
		const Scope &layer = GetRequiredScope(*(*it).second);
		const ElementCollection &LayerElement = layer.GetCollection("LayerElement");
		for (ElementMap::const_iterator eit = LayerElement.first; eit != LayerElement.second; ++eit) {
			std::string layer_name = (*eit).first;
			Element *element_layer = (*eit).second;
			const Scope &layer_element = GetRequiredScope(*element_layer);
			std::cout << "[read layer] " << layer_name << std::endl;
			const Element &Type = GetRequiredElement(layer_element, "Type");
			const Element &TypedIndex = GetRequiredElement(layer_element, "TypedIndex");
			const std::string &type = ParseTokenAsString(GetRequiredToken(Type, 0));
			const int typedIndex = ParseTokenAsInt(GetRequiredToken(TypedIndex, 0));

			std::cout << "[layer element] type: " << type << ", " << typedIndex << std::endl;

			// get object / mesh directly from the FBX by the element ID.
			const Scope &top = GetRequiredScope(element);

			// Get collection of elements from the NormalLayerMap
			// this must contain our proper elements.
			const ElementCollection candidates = top.GetCollection(type);

			/* typedef std::vector< Scope* > ScopeList;
			 * typedef std::fbx_unordered_multimap< std::string, Element* > ElementMap;
			 * typedef std::pair<ElementMap::const_iterator,ElementMap::const_iterator> ElementCollection;
			 */

			for (ElementMap::const_iterator cand_iter = candidates.first; cand_iter != candidates.second; ++cand_iter) {
				std::string val = (*cand_iter).first;
				//Element *element = (*canditer).second;
				std::cout << "key: " << val << std::endl;

				const Scope &layer_scope = GetRequiredScope(*(*cand_iter).second);
				const Token &layer_token = GetRequiredToken(*(*cand_iter).second, 0);
				const int index = ParseTokenAsInt(layer_token);
				if (index == typedIndex) {
					const std::string &MappingInformationType = ParseTokenAsString(GetRequiredToken(
							GetRequiredElement(layer_scope, "MappingInformationType"), 0));

					const std::string &ReferenceInformationType = ParseTokenAsString(GetRequiredToken(
							GetRequiredElement(layer_scope, "ReferenceInformationType"), 0));

					// Not required:
					// LayerElementTangent
					// LayerElementBinormal - perpendicular to tangent.
					if (type == "LayerElementUV") {
						if (index == 0) {
							m_uv_0 = resolve_vertex_data_array<Vector2>(layer_scope, MappingInformationType, ReferenceInformationType, "UV");
						} else if (index == 1) {
							m_uv_1 = resolve_vertex_data_array<Vector2>(layer_scope, MappingInformationType, ReferenceInformationType, "UV");
						}
					} else if (type == "LayerElementMaterial") {
						m_material_allocation_ids = resolve_vertex_data_array<int>(layer_scope, MappingInformationType, ReferenceInformationType, "Materials");
					} else if (type == "LayerElementNormal") {
						m_normals = resolve_vertex_data_array<Vector3>(layer_scope, MappingInformationType, ReferenceInformationType, "Normals");
					} else if (type == "LayerElementColor") {
						m_colors = resolve_vertex_data_array<Color>(layer_scope, MappingInformationType, ReferenceInformationType, "Colors");
					}
				}
			}
		}
	}

	// Compose the edge of the mesh.
	// You can see how the edges are stored into the FBX here: https://gist.github.com/AndreaCatania/da81840f5aa3b2feedf189e26c5a87e6
	for (size_t i = 0; i < m_edges.size(); i += 1) {
		ERR_FAIL_INDEX_MSG((size_t)m_edges[i], m_face_indices.size(), "The edge is pointing to a weird location in the face indices. The FBX is corrupted.");
		int polygon_vertex_0 = m_face_indices[m_edges[i]];
		int polygon_vertex_1;
		if (polygon_vertex_0 < 0) {
			// The polygon_vertex_0 points to the end of a polygon, so it's
			// connected with the begining of polygon in the edge list.

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
			// ever vertex is always considered the begining of a polygon.
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
	return m_uv_0;
}

const MeshGeometry::MappingData<Vector2> &MeshGeometry::get_uv_1() const {
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
		const Scope &source,
		const std::string &MappingInformationType,
		const std::string &ReferenceInformationType,
		const std::string &dataElementName) {

	// UVIndex, MaterialIndex, NormalIndex, etc..
	std::string indexDataElementName = dataElementName + "Index";
	// goal: expand everything to be per vertex

	ReferenceType l_ref_type = ReferenceType::direct;

	// purposefully merging legacy to IndexToDirect
	if (ReferenceInformationType == "IndexToDirect" || ReferenceInformationType == "Index") {
		// set non legacy index to direct mapping
		l_ref_type = ReferenceType::index_to_direct;

		// override invalid files - should not happen but if it does we're safe.
		if (!HasElement(source, indexDataElementName)) {
			l_ref_type = ReferenceType::direct;
		}
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

	// index array wont always exist
	const Element *element = GetOptionalElement(source, indexDataElementName);
	if (element) {
		ParseVectorDataArray(tempData.index, *element);
	}

	return tempData;
}
// ------------------------------------------------------------------------------------------------
ShapeGeometry::ShapeGeometry(uint64_t id, const Element &element, const std::string &name, const Document &doc) :
		Geometry(id, element, name, doc) {
	const Scope *sc = element.Compound();
	if (nullptr == sc) {
		DOMError("failed to read Geometry object (class: Shape), no data scope found");
	}
	const Element &Indexes = GetRequiredElement(*sc, "Indexes", &element);
	const Element &Normals = GetRequiredElement(*sc, "Normals", &element);
	const Element &Vertices = GetRequiredElement(*sc, "Vertices", &element);
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
LineGeometry::LineGeometry(uint64_t id, const Element &element, const std::string &name, const Document &doc) :
		Geometry(id, element, name, doc) {
	const Scope *sc = element.Compound();
	if (!sc) {
		DOMError("failed to read Geometry object (class: Line), no data scope found");
	}
	const Element &Points = GetRequiredElement(*sc, "Points", &element);
	const Element &PointsIndex = GetRequiredElement(*sc, "PointsIndex", &element);
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
} // namespace FBX
} // namespace Assimp