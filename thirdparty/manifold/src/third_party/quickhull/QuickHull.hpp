#ifndef QUICKHULL_HPP_
#define QUICKHULL_HPP_
#include <deque>
#include <vector>
#include <array>
#include <limits>
#include "Structs/Vector3.hpp"
#include "Structs/Plane.hpp"
#include "Structs/Pool.hpp"
#include "Structs/Mesh.hpp"
#include "ConvexHull.hpp"
#include "HalfEdgeMesh.hpp"
#include "MathUtils.hpp"

/*
 * Implementation of the 3D QuickHull algorithm by Antti Kuukka
 *
 * No copyrights. What follows is 100% Public Domain.
 *
 *
 *
 * INPUT:  a list of points in 3D space (for example, vertices of a 3D mesh)
 *
 * OUTPUT: a ConvexHull object which provides vertex and index buffers of the generated convex hull as a triangle mesh.
 *
 *
 *
 * The implementation is thread-safe if each thread is using its own QuickHull object.
 *
 *
 * SUMMARY OF THE ALGORITHM:
 *         - Create initial simplex (tetrahedron) using extreme points. We have four faces now and they form a convex mesh M.
 *         - For each point, assign them to the first face for which they are on the positive side of (so each point is assigned to at most
 *           one face). Points inside the initial tetrahedron are left behind now and no longer affect the calculations.
 *         - Add all faces that have points assigned to them to Face Stack.
 *         - Iterate until Face Stack is empty:
 *              - Pop topmost face F from the stack
 *              - From the points assigned to F, pick the point P that is farthest away from the plane defined by F.
 *              - Find all faces of M that have P on their positive side. Let us call these the "visible faces".
 *              - Because of the way M is constructed, these faces are connected. Solve their horizon edge loop.
 *				- "Extrude to P": Create new faces by connecting P with the points belonging to the horizon edge. Add the new faces to M and remove the visible
 *                faces from M.
 *              - Each point that was assigned to visible faces is now assigned to at most one of the newly created faces.
 *              - Those new faces that have points assigned to them are added to the top of Face Stack.
 *          - M is now the convex hull.
 *
 * TO DO:
 *  - Implement a proper 2D QuickHull and use that to solve the degenerate 2D case (when all the points lie on the same plane in 3D space).
 * */

namespace quickhull {
	
	struct DiagnosticsData {
		size_t m_failedHorizonEdges; // How many times QuickHull failed to solve the horizon edge. Failures lead to degenerated convex hulls.
		
		DiagnosticsData() : m_failedHorizonEdges(0) { }
	};

	template<typename FloatType>
	FloatType defaultEps();

	template<typename FloatType>
	class QuickHull {
		using vec3 = Vector3<FloatType>;

		FloatType m_epsilon, m_epsilonSquared, m_scale;
		bool m_planar;
		std::vector<vec3> m_planarPointCloudTemp;
		VertexDataSource<FloatType> m_vertexData;
		MeshBuilder<FloatType> m_mesh;
		std::array<size_t,6> m_extremeValues;
		DiagnosticsData m_diagnostics;

		// Temporary variables used during iteration process
		std::vector<size_t> m_newFaceIndices;
		std::vector<size_t> m_newHalfEdgeIndices;
		std::vector< std::unique_ptr<std::vector<size_t>> > m_disabledFacePointVectors;
		std::vector<size_t> m_visibleFaces;
		std::vector<size_t> m_horizonEdges;
		struct FaceData {
			size_t m_faceIndex;
			size_t m_enteredFromHalfEdge; // If the face turns out not to be visible, this half edge will be marked as horizon edge
			FaceData(size_t fi, size_t he) : m_faceIndex(fi),m_enteredFromHalfEdge(he) {}
		};
		std::vector<FaceData> m_possiblyVisibleFaces;
		std::deque<size_t> m_faceList;

		// Create a half edge mesh representing the base tetrahedron from which the QuickHull iteration proceeds. m_extremeValues must be properly set up when this is called.
		void setupInitialTetrahedron();

		// Given a list of half edges, try to rearrange them so that they form a loop. Return true on success.
		bool reorderHorizonEdges(std::vector<size_t>& horizonEdges);
		
		// Find indices of extreme values (max x, min x, max y, min y, max z, min z) for the given point cloud
		std::array<size_t,6> getExtremeValues();
		
		// Compute scale of the vertex data.
		FloatType getScale(const std::array<size_t,6>& extremeValues);
		
		// Each face contains a unique pointer to a vector of indices. However, many - often most - faces do not have any points on the positive
		// side of them especially at the the end of the iteration. When a face is removed from the mesh, its associated point vector, if such
		// exists, is moved to the index vector pool, and when we need to add new faces with points on the positive side to the mesh,
		// we reuse these vectors. This reduces the amount of std::vectors we have to deal with, and impact on performance is remarkable.
		Pool<std::vector<size_t>> m_indexVectorPool;
		inline std::unique_ptr<std::vector<size_t>> getIndexVectorFromPool();
		inline void reclaimToIndexVectorPool(std::unique_ptr<std::vector<size_t>>& ptr);
		
		// Associates a point with a face if the point resides on the positive side of the plane. Returns true if the points was on the positive side.
		inline bool addPointToFace(typename MeshBuilder<FloatType>::Face& f, size_t pointIndex);
		
		// This will update m_mesh from which we create the ConvexHull object that getConvexHull function returns
		void createConvexHalfEdgeMesh();
		
		// Constructs the convex hull into a MeshBuilder object which can be converted to a ConvexHull or Mesh object
		void buildMesh(const VertexDataSource<FloatType>& pointCloud, bool CCW, bool useOriginalIndices, FloatType eps);
		
		// The public getConvexHull functions will setup a VertexDataSource object and call this
		ConvexHull<FloatType> getConvexHull(const VertexDataSource<FloatType>& pointCloud, bool CCW, bool useOriginalIndices, FloatType eps);
	public:
		// Computes convex hull for a given point cloud.
		// Params:
		//   pointCloud: a vector of of 3D points
		//   CCW: whether the output mesh triangles should have CCW orientation
		//   useOriginalIndices: should the output mesh use same vertex indices as the original point cloud. If this is false,
		//      then we generate a new vertex buffer which contains only the vertices that are part of the convex hull.
		//   eps: minimum distance to a plane to consider a point being on positive of it (for a point cloud with scale 1)
		ConvexHull<FloatType> getConvexHull(const std::vector<Vector3<FloatType>>& pointCloud,
											bool CCW,
											bool useOriginalIndices,
											FloatType eps = defaultEps<FloatType>());
		
		// Computes convex hull for a given point cloud.
		// Params:
		//   vertexData: pointer to the first 3D point of the point cloud
		//   vertexCount: number of vertices in the point cloud
		//   CCW: whether the output mesh triangles should have CCW orientation
		//   useOriginalIndices: should the output mesh use same vertex indices as the original point cloud. If this is false,
		//      then we generate a new vertex buffer which contains only the vertices that are part of the convex hull.
		//   eps: minimum distance to a plane to consider a point being on positive side of it (for a point cloud with scale 1)
		ConvexHull<FloatType> getConvexHull(const Vector3<FloatType>* vertexData,
											size_t vertexCount,
											bool CCW,
											bool useOriginalIndices,
											FloatType eps = defaultEps<FloatType>());
		
		// Computes convex hull for a given point cloud. This function assumes that the vertex data resides in memory
		// in the following format: x_0,y_0,z_0,x_1,y_1,z_1,...
		// Params:
		//   vertexData: pointer to the X component of the first point of the point cloud.
		//   vertexCount: number of vertices in the point cloud
		//   CCW: whether the output mesh triangles should have CCW orientation
		//   useOriginalIndices: should the output mesh use same vertex indices as the original point cloud. If this is false,
		//      then we generate a new vertex buffer which contains only the vertices that are part of the convex hull.
		//   eps: minimum distance to a plane to consider a point being on positive side of it (for a point cloud with scale 1)
		ConvexHull<FloatType> getConvexHull(const FloatType* vertexData,
											size_t vertexCount,
											bool CCW,
											bool useOriginalIndices,
											FloatType eps = defaultEps<FloatType>());
		
		// Computes convex hull for a given point cloud. This function assumes that the vertex data resides in memory
		// in the following format: x_0,y_0,z_0,x_1,y_1,z_1,...
		// Params:
		//   vertexData: pointer to the X component of the first point of the point cloud.
		//   vertexCount: number of vertices in the point cloud
		//   CCW: whether the output mesh triangles should have CCW orientation
		//   eps: minimum distance to a plane to consider a point being on positive side of it (for a point cloud with scale 1)
		// Returns:
		//   Convex hull of the point cloud as a mesh object with half edge structure.
		HalfEdgeMesh<FloatType, size_t> getConvexHullAsMesh(const FloatType* vertexData,
															size_t vertexCount,
															bool CCW,
															FloatType eps = defaultEps<FloatType>());
		
		// Get diagnostics about last generated convex hull
		const DiagnosticsData& getDiagnostics() {
			return m_diagnostics;
		}
	};
	
	/*
	 * Inline function definitions
	 */
	
	template<typename T>
	std::unique_ptr<std::vector<size_t>> QuickHull<T>::getIndexVectorFromPool() {
		auto r = m_indexVectorPool.get();
		r->clear();
		return r;
	}
	
	template<typename T>
	void QuickHull<T>::reclaimToIndexVectorPool(std::unique_ptr<std::vector<size_t>>& ptr) {
		const size_t oldSize = ptr->size();
		if ((oldSize+1)*128 < ptr->capacity()) {
			// Reduce memory usage! Huge vectors are needed at the beginning of iteration when faces have many points on their positive side. Later on, smaller vectors will suffice.
			ptr.reset(nullptr);
			return;
		}
		m_indexVectorPool.reclaim(ptr);
	}

	template<typename T>
	bool QuickHull<T>::addPointToFace(typename MeshBuilder<T>::Face& f, size_t pointIndex) {
		const T D = mathutils::getSignedDistanceToPlane(m_vertexData[ pointIndex ],f.m_P);
		if (D>0 && D*D > m_epsilonSquared*f.m_P.m_sqrNLength) {
			if (!f.m_pointsOnPositiveSide) {
				f.m_pointsOnPositiveSide = std::move(getIndexVectorFromPool());
			}
			f.m_pointsOnPositiveSide->push_back( pointIndex );
			if (D > f.m_mostDistantPointDist) {
				f.m_mostDistantPointDist = D;
				f.m_mostDistantPoint = pointIndex;
			}
			return true;
		}
		return false;
	}

}


#endif /* QUICKHULL_HPP_ */
