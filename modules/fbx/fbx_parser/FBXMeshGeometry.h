/*************************************************************************/
/*  FBXMeshGeometry.h                                                    */
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

#ifndef FBX_MESH_GEOMETRY_H
#define FBX_MESH_GEOMETRY_H

#include "core/math/color.h"
#include "core/math/vector2.h"
#include "core/math/vector3.h"
#include "core/templates/vector.h"

#include "FBXDocument.h"
#include "FBXParser.h"

#include <iostream>

#define AI_MAX_NUMBER_OF_TEXTURECOORDS 4
#define AI_MAX_NUMBER_OF_COLOR_SETS 8

namespace FBXDocParser {

/*
 * DOM base class for all kinds of FBX geometry
 */
class Geometry : public Object {
public:
	Geometry(uint64_t id, const ElementPtr element, const std::string &name, const Document &doc);
	virtual ~Geometry();

	/** Get the Skin attached to this geometry or nullptr */
	const Skin *DeformerSkin() const;

	const std::vector<const BlendShape *> &get_blend_shapes() const;

	size_t get_blend_shape_count() const {
		return blendShapes.size();
	}

private:
	const Skin *skin = nullptr;
	std::vector<const BlendShape *> blendShapes;
};

typedef std::vector<int> MatIndexArray;

/// Map Geometry stores the FBX file information.
///
/// # FBX doc.
/// ## Reference type declared:
/// 	- Direct (directly related to the mapping information type)
/// 	- IndexToDirect (Map with key value, meaning depends on the MappingInformationType)
///
/// ## Map Type:
/// 	* None The mapping is undetermined.
/// 	* ByVertex There will be one mapping coordinate for each surface control point/vertex (ControlPoint is a vertex).
/// 		* If you have direct reference type vertices[x]
/// 		* If you have IndexToDirect reference type the UV
/// 	* ByPolygonVertex There will be one mapping coordinate for each vertex, for every polygon of which it is a part. This means that a vertex will have as many mapping coordinates as polygons of which it is a part. (Sorted by polygon, referencing vertex)
/// 	* ByPolygon There can be only one mapping coordinate for the whole polygon.
/// 		* One mapping per polygon polygon x has this normal x
/// 		* For each vertex of the polygon then set the normal to x
/// 	* ByEdge There will be one mapping coordinate for each unique edge in the mesh. This is meant to be used with smoothing layer elements. (Mapping is referencing the edge id)
/// 	* AllSame There can be only one mapping coordinate for the whole surface.
class MeshGeometry : public Geometry {
public:
	enum class MapType {
		none = 0, // No mapping type. Stored as "None".
		vertex, // Maps per vertex. Stored as "ByVertice".
		polygon_vertex, // Maps per polygon vertex. Stored as "ByPolygonVertex".
		polygon, // Maps per polygon. Stored as "ByPolygon".
		edge, // Maps per edge. Stored as "ByEdge".
		all_the_same // Uaps to everything. Stored as "AllSame".
	};

	enum class ReferenceType {
		direct = 0,
		index = 1,
		index_to_direct = 2
	};

	template <class T>
	struct MappingData {
		MapType map_type = MapType::none;
		ReferenceType ref_type = ReferenceType::direct;
		std::vector<T> data;
		/// The meaning of the indices depends from the `MapType`.
		/// If `ref_type` is `direct` this map is hollow.
		std::vector<int> index;

		String debug_info() const {
			return "indexes: " + itos(index.size()) + " data: " + itos(data.size());
		}
	};

	struct Edge {
		int vertex_0 = 0, vertex_1 = 0;
		Edge(int v0, int v1) :
				vertex_0(v0), vertex_1(v1) {}
		Edge() {}
	};

public:
	MeshGeometry(uint64_t id, const ElementPtr element, const std::string &name, const Document &doc);

	virtual ~MeshGeometry();

	const std::vector<Vector3> &get_vertices() const;
	const std::vector<Edge> &get_edge_map() const;
	const std::vector<int> &get_polygon_indices() const;
	const std::vector<int> &get_edges() const;
	const MappingData<Vector3> &get_normals() const;
	const MappingData<Vector2> &get_uv_0() const;
	const MappingData<Vector2> &get_uv_1() const;
	const MappingData<Color> &get_colors() const;
	const MappingData<int> &get_material_allocation_id() const;

	/// Returns -1 if the vertices doesn't form an edge. Vertex order, doesn't
	// matter.
	static int get_edge_id(const std::vector<Edge> &p_map, int p_vertex_a, int p_vertex_b);
	// Returns the edge point bu that ID, or the edge with -1 vertices if the
	// id is not valid.
	static Edge get_edge(const std::vector<Edge> &p_map, int p_id);

private:
	// Read directly from the FBX file.
	std::vector<Vector3> m_vertices;
	std::vector<Edge> edge_map;
	std::vector<int> m_face_indices;
	std::vector<int> m_edges;
	MappingData<Vector3> m_normals;
	MappingData<Vector2> m_uv_0; // first uv coordinates
	MappingData<Vector2> m_uv_1; // second uv coordinates
	MappingData<Color> m_colors; // colors for the mesh
	MappingData<int> m_material_allocation_ids; // slot of material used

	template <class T>
	MappingData<T> resolve_vertex_data_array(
			const ScopePtr source,
			const std::string &MappingInformationType,
			const std::string &ReferenceInformationType,
			const std::string &dataElementName,
			const std::string &indexOverride = "");
};

/*
 * DOM class for FBX geometry of type "Shape"
 */
class ShapeGeometry : public Geometry {
public:
	/** The class constructor */
	ShapeGeometry(uint64_t id, const ElementPtr element, const std::string &name, const Document &doc);

	/** The class destructor */
	virtual ~ShapeGeometry();

	/** Get a list of all vertex points, non-unique*/
	const std::vector<Vector3> &GetVertices() const;

	/** Get a list of all vertex normals or an empty array if
    *  no normals are specified. */
	const std::vector<Vector3> &GetNormals() const;

	/** Return list of vertex indices. */
	const std::vector<unsigned int> &GetIndices() const;

private:
	std::vector<Vector3> m_vertices;
	std::vector<Vector3> m_normals;
	std::vector<unsigned int> m_indices;
};
/**
*  DOM class for FBX geometry of type "Line"
*/
class LineGeometry : public Geometry {
public:
	/** The class constructor */
	LineGeometry(uint64_t id, const ElementPtr element, const std::string &name, const Document &doc);

	/** The class destructor */
	virtual ~LineGeometry();

	/** Get a list of all vertex points, non-unique*/
	const std::vector<Vector3> &GetVertices() const;

	/** Return list of vertex indices. */
	const std::vector<int> &GetIndices() const;

private:
	std::vector<Vector3> m_vertices;
	std::vector<int> m_indices;
};
} // namespace FBXDocParser

#endif // FBX_MESH_GEOMETRY_H
