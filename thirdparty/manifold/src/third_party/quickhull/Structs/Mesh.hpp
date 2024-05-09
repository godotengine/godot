#ifndef MESH_HPP_
#define MESH_HPP_

#include <vector>
#include "Vector3.hpp"
#include "Plane.hpp"
#include "Pool.hpp"
#include <array>
#include <cassert>
#include <limits>
#include <memory>
#include "VertexDataSource.hpp"
#include <unordered_map>
#include <cinttypes>

namespace quickhull {

	template <typename T>
	class MeshBuilder {
	public:
		struct HalfEdge {
			size_t m_endVertex;
			size_t m_opp;
			size_t m_face;
			size_t m_next;
			
			void disable() {
				m_endVertex = std::numeric_limits<size_t>::max();
			}
			
			bool isDisabled() const {
				return m_endVertex == std::numeric_limits<size_t>::max();
			}
		};

		struct Face {
			size_t m_he;
			Plane<T> m_P{};
			T m_mostDistantPointDist;
			size_t m_mostDistantPoint;
			size_t m_visibilityCheckedOnIteration;
			std::uint8_t m_isVisibleFaceOnCurrentIteration : 1;
			std::uint8_t m_inFaceStack : 1;
			std::uint8_t m_horizonEdgesOnCurrentIteration : 3; // Bit for each half edge assigned to this face, each being 0 or 1 depending on whether the edge belongs to horizon edge
			std::unique_ptr<std::vector<size_t>> m_pointsOnPositiveSide;

			Face() : m_he(std::numeric_limits<size_t>::max()),
					 m_mostDistantPointDist(0),
					 m_mostDistantPoint(0),
					 m_visibilityCheckedOnIteration(0),
					 m_isVisibleFaceOnCurrentIteration(0),
					 m_inFaceStack(0),
					 m_horizonEdgesOnCurrentIteration(0)
			{

			}
			
			void disable() {
				m_he = std::numeric_limits<size_t>::max();
			}

			bool isDisabled() const {
				return m_he == std::numeric_limits<size_t>::max();
			}
		};

		// Mesh data
		std::vector<Face> m_faces;
		std::vector<HalfEdge> m_halfEdges;
		
		// When the mesh is modified and faces and half edges are removed from it, we do not actually remove them from the container vectors.
		// Insted, they are marked as disabled which means that the indices can be reused when we need to add new faces and half edges to the mesh.
		// We store the free indices in the following vectors.
		std::vector<size_t> m_disabledFaces,m_disabledHalfEdges;
		
		size_t addFace() {
			if (m_disabledFaces.size()) {
				size_t index = m_disabledFaces.back();
				auto& f = m_faces[index];
				assert(f.isDisabled());
				assert(!f.m_pointsOnPositiveSide);
				f.m_mostDistantPointDist = 0;
				m_disabledFaces.pop_back();
				return index;
			}
			m_faces.emplace_back();
			return m_faces.size()-1;
		}

		size_t addHalfEdge()	{
			if (m_disabledHalfEdges.size()) {
				const size_t index = m_disabledHalfEdges.back();
				m_disabledHalfEdges.pop_back();
				return index;
			}
			m_halfEdges.emplace_back();
			return m_halfEdges.size()-1;
		}

		// Mark a face as disabled and return a pointer to the points that were on the positive of it.
		std::unique_ptr<std::vector<size_t>> disableFace(size_t faceIndex) {
			auto& f = m_faces[faceIndex];
			f.disable();
			m_disabledFaces.push_back(faceIndex);
			return std::move(f.m_pointsOnPositiveSide);
		}

		void disableHalfEdge(size_t heIndex) {
			auto& he = m_halfEdges[heIndex];
			he.disable();
			m_disabledHalfEdges.push_back(heIndex);
		}

		MeshBuilder() = default;
		
		// Create a mesh with initial tetrahedron ABCD. Dot product of AB with the normal of triangle ABC should be negative.
		void setup(size_t a, size_t b, size_t c, size_t d) {
			m_faces.clear();
			m_halfEdges.clear();
			m_disabledFaces.clear();
			m_disabledHalfEdges.clear();
			
			m_faces.reserve(4);
			m_halfEdges.reserve(12);
			
			// Create halfedges
			HalfEdge AB;
			AB.m_endVertex = b;
			AB.m_opp = 6;
			AB.m_face = 0;
			AB.m_next = 1;
			m_halfEdges.push_back(AB);

			HalfEdge BC;
			BC.m_endVertex = c;
			BC.m_opp = 9;
			BC.m_face = 0;
			BC.m_next = 2;
			m_halfEdges.push_back(BC);

			HalfEdge CA;
			CA.m_endVertex = a;
			CA.m_opp = 3;
			CA.m_face = 0;
			CA.m_next = 0;
			m_halfEdges.push_back(CA);

			HalfEdge AC;
			AC.m_endVertex = c;
			AC.m_opp = 2;
			AC.m_face = 1;
			AC.m_next = 4;
			m_halfEdges.push_back(AC);

			HalfEdge CD;
			CD.m_endVertex = d;
			CD.m_opp = 11;
			CD.m_face = 1;
			CD.m_next = 5;
			m_halfEdges.push_back(CD);

			HalfEdge DA;
			DA.m_endVertex = a;
			DA.m_opp = 7;
			DA.m_face = 1;
			DA.m_next = 3;
			m_halfEdges.push_back(DA);

			HalfEdge BA;
			BA.m_endVertex = a;
			BA.m_opp = 0;
			BA.m_face = 2;
			BA.m_next = 7;
			m_halfEdges.push_back(BA);

			HalfEdge AD;
			AD.m_endVertex = d;
			AD.m_opp = 5;
			AD.m_face = 2;
			AD.m_next = 8;
			m_halfEdges.push_back(AD);

			HalfEdge DB;
			DB.m_endVertex = b;
			DB.m_opp = 10;
			DB.m_face = 2;
			DB.m_next = 6;
			m_halfEdges.push_back(DB);

			HalfEdge CB;
			CB.m_endVertex = b;
			CB.m_opp = 1;
			CB.m_face = 3;
			CB.m_next = 10;
			m_halfEdges.push_back(CB);

			HalfEdge BD;
			BD.m_endVertex = d;
			BD.m_opp = 8;
			BD.m_face = 3;
			BD.m_next = 11;
			m_halfEdges.push_back(BD);

			HalfEdge DC;
			DC.m_endVertex = c;
			DC.m_opp = 4;
			DC.m_face = 3;
			DC.m_next = 9;
			m_halfEdges.push_back(DC);

			// Create faces
			Face ABC;
			ABC.m_he = 0;
			m_faces.push_back(std::move(ABC));

			Face ACD;
			ACD.m_he = 3;
			m_faces.push_back(std::move(ACD));

			Face BAD;
			BAD.m_he = 6;
			m_faces.push_back(std::move(BAD));

			Face CBD;
			CBD.m_he = 9;
			m_faces.push_back(std::move(CBD));
		}

		std::array<size_t,3> getVertexIndicesOfFace(const Face& f) const {
			std::array<size_t,3> v;
			const HalfEdge* he = &m_halfEdges[f.m_he];
			v[0] = he->m_endVertex;
			he = &m_halfEdges[he->m_next];
			v[1] = he->m_endVertex;
			he = &m_halfEdges[he->m_next];
			v[2] = he->m_endVertex;
			return v;
		}

		std::array<size_t,2> getVertexIndicesOfHalfEdge(const HalfEdge& he) const {
			return {m_halfEdges[he.m_opp].m_endVertex,he.m_endVertex};
		}

		std::array<size_t,3> getHalfEdgeIndicesOfFace(const Face& f) const {
			return {f.m_he,m_halfEdges[f.m_he].m_next,m_halfEdges[m_halfEdges[f.m_he].m_next].m_next};
		}
	};
	


}



#endif 
