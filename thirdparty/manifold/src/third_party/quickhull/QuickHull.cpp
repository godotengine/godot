#include "QuickHull.hpp"
#include "MathUtils.hpp"
#include <cmath>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <limits>
#include "Structs/Mesh.hpp"

namespace quickhull {

	template<>
	float defaultEps() { 
		return 0.0001f;
	}
	
	template<>
	double defaultEps() {
		return 0.0000001; 
	}
	
	/*
	 * Implementation of the algorithm
	 */
	
	template<typename T>
	ConvexHull<T> QuickHull<T>::getConvexHull(const std::vector<Vector3<T>>& pointCloud, bool CCW, bool useOriginalIndices, T epsilon) {
		VertexDataSource<T> vertexDataSource(pointCloud);
		return getConvexHull(vertexDataSource,CCW,useOriginalIndices,epsilon);
	}
	
	template<typename T>
	ConvexHull<T> QuickHull<T>::getConvexHull(const Vector3<T>* vertexData, size_t vertexCount, bool CCW, bool useOriginalIndices, T epsilon) {
		VertexDataSource<T> vertexDataSource(vertexData,vertexCount);
		return getConvexHull(vertexDataSource,CCW,useOriginalIndices,epsilon);
	}
	
	template<typename T>
	ConvexHull<T> QuickHull<T>::getConvexHull(const T* vertexData, size_t vertexCount, bool CCW, bool useOriginalIndices, T epsilon) {
		VertexDataSource<T> vertexDataSource((const vec3*)vertexData,vertexCount);
		return getConvexHull(vertexDataSource,CCW,useOriginalIndices,epsilon);
	}
	
	template<typename FloatType>
	HalfEdgeMesh<FloatType, size_t> QuickHull<FloatType>::getConvexHullAsMesh(const FloatType* vertexData, size_t vertexCount, bool CCW, FloatType epsilon) {
		VertexDataSource<FloatType> vertexDataSource((const vec3*)vertexData,vertexCount);
		buildMesh(vertexDataSource, CCW, false, epsilon);
		return HalfEdgeMesh<FloatType, size_t>(m_mesh, m_vertexData);
	}
	
	template<typename T>
	void QuickHull<T>::buildMesh(const VertexDataSource<T>& pointCloud, bool CCW, bool useOriginalIndices, T epsilon) {
		// CCW is unused for now
		(void)CCW;
		// useOriginalIndices is unused for now
		(void)useOriginalIndices;

		if (pointCloud.size()==0) {
			m_mesh = MeshBuilder<T>();
			return;
		}
		m_vertexData = pointCloud;
		
		// Very first: find extreme values and use them to compute the scale of the point cloud.
		m_extremeValues = getExtremeValues();
		m_scale = getScale(m_extremeValues);
		
		// Epsilon we use depends on the scale
		m_epsilon = epsilon*m_scale;
		m_epsilonSquared = m_epsilon*m_epsilon;
		
		// Reset diagnostics
		m_diagnostics = DiagnosticsData();
		
		m_planar = false; // The planar case happens when all the points appear to lie on a two dimensional subspace of R^3.
		createConvexHalfEdgeMesh();
		if (m_planar) {
			const size_t extraPointIndex = m_planarPointCloudTemp.size()-1;
			for (auto& he : m_mesh.m_halfEdges) {
				if (he.m_endVertex == extraPointIndex) {
					he.m_endVertex = 0;
				}
			}
			m_vertexData = pointCloud;
			m_planarPointCloudTemp.clear();
		}
	}

	template<typename T>
	ConvexHull<T> QuickHull<T>::getConvexHull(const VertexDataSource<T>& pointCloud, bool CCW, bool useOriginalIndices, T epsilon) {
		buildMesh(pointCloud,CCW,useOriginalIndices,epsilon);
		return ConvexHull<T>(m_mesh,m_vertexData, CCW, useOriginalIndices);
	}

	template<typename T>
	void QuickHull<T>::createConvexHalfEdgeMesh() {
		m_visibleFaces.clear();
		m_horizonEdges.clear();
		m_possiblyVisibleFaces.clear();
		
		// Compute base tetrahedron
		setupInitialTetrahedron();
		assert(m_mesh.m_faces.size()==4);

		// Init face stack with those faces that have points assigned to them
		m_faceList.clear();
		for (size_t i=0;i < 4;i++) {
			auto& f = m_mesh.m_faces[i];
			if (f.m_pointsOnPositiveSide && f.m_pointsOnPositiveSide->size()>0) {
				m_faceList.push_back(i);
				f.m_inFaceStack = 1;
			}
		}

		// Process faces until the face list is empty.
		size_t iter = 0;
		while (!m_faceList.empty()) {
			iter++;
			if (iter == std::numeric_limits<size_t>::max()) {
				// Visible face traversal marks visited faces with iteration counter (to mark that the face has been visited on this iteration) and the max value represents unvisited faces. At this point we have to reset iteration counter. This shouldn't be an
				// issue on 64 bit machines.
				iter = 0;
			}
			
			const size_t topFaceIndex = m_faceList.front();
			m_faceList.pop_front();
			
			auto& tf = m_mesh.m_faces[topFaceIndex];
			tf.m_inFaceStack = 0;

			assert(!tf.m_pointsOnPositiveSide || tf.m_pointsOnPositiveSide->size()>0);
			if (!tf.m_pointsOnPositiveSide || tf.isDisabled()) {
				continue;
			}
			
			// Pick the most distant point to this triangle plane as the point to which we extrude
			const vec3& activePoint = m_vertexData[tf.m_mostDistantPoint];
			const size_t activePointIndex = tf.m_mostDistantPoint;

			// Find out the faces that have our active point on their positive side (these are the "visible faces"). The face on top of the stack of course is one of them. At the same time, we create a list of horizon edges.
			m_horizonEdges.clear();
			m_possiblyVisibleFaces.clear();
			m_visibleFaces.clear();
			m_possiblyVisibleFaces.emplace_back(topFaceIndex,std::numeric_limits<size_t>::max());
			while (m_possiblyVisibleFaces.size()) {
				const auto faceData = m_possiblyVisibleFaces.back();
				m_possiblyVisibleFaces.pop_back();
				auto& pvf = m_mesh.m_faces[faceData.m_faceIndex];
				assert(!pvf.isDisabled());
				
				if (pvf.m_visibilityCheckedOnIteration == iter) {
					if (pvf.m_isVisibleFaceOnCurrentIteration) {
						continue;
					}
				}
				else {
					const Plane<T>& P = pvf.m_P;
					pvf.m_visibilityCheckedOnIteration = iter;
					const T d = P.m_N.dotProduct(activePoint)+P.m_D;
					if (d>0) {
						pvf.m_isVisibleFaceOnCurrentIteration = 1;
						pvf.m_horizonEdgesOnCurrentIteration = 0;
						m_visibleFaces.push_back(faceData.m_faceIndex);
						for (auto heIndex : m_mesh.getHalfEdgeIndicesOfFace(pvf)) {
							if (m_mesh.m_halfEdges[heIndex].m_opp != faceData.m_enteredFromHalfEdge) {
								m_possiblyVisibleFaces.emplace_back( m_mesh.m_halfEdges[m_mesh.m_halfEdges[heIndex].m_opp].m_face,heIndex );
							}
						}
						continue;
					}
					assert(faceData.m_faceIndex != topFaceIndex);
				}

				// The face is not visible. Therefore, the halfedge we came from is part of the horizon edge.
				pvf.m_isVisibleFaceOnCurrentIteration = 0;
				m_horizonEdges.push_back(faceData.m_enteredFromHalfEdge);
				// Store which half edge is the horizon edge. The other half edges of the face will not be part of the final mesh so their data slots can by recycled.
				const auto halfEdges = m_mesh.getHalfEdgeIndicesOfFace(m_mesh.m_faces[m_mesh.m_halfEdges[faceData.m_enteredFromHalfEdge].m_face]);
				const std::int8_t ind = (halfEdges[0]==faceData.m_enteredFromHalfEdge) ? 0 : (halfEdges[1]==faceData.m_enteredFromHalfEdge ? 1 : 2);
				m_mesh.m_faces[m_mesh.m_halfEdges[faceData.m_enteredFromHalfEdge].m_face].m_horizonEdgesOnCurrentIteration |= (1<<ind);
			}
			const size_t horizonEdgeCount = m_horizonEdges.size();

			// Order horizon edges so that they form a loop. This may fail due to numerical instability in which case we give up trying to solve horizon edge for this point and accept a minor degeneration in the convex hull.
			if (!reorderHorizonEdges(m_horizonEdges)) {
				m_diagnostics.m_failedHorizonEdges++;
				std::cerr << "Failed to solve horizon edge." << std::endl;
				auto it = std::find(tf.m_pointsOnPositiveSide->begin(),tf.m_pointsOnPositiveSide->end(),activePointIndex);
				tf.m_pointsOnPositiveSide->erase(it);
				if (tf.m_pointsOnPositiveSide->size()==0) {
					reclaimToIndexVectorPool(tf.m_pointsOnPositiveSide);
				}
				continue;
			}

			// Except for the horizon edges, all half edges of the visible faces can be marked as disabled. Their data slots will be reused.
			// The faces will be disabled as well, but we need to remember the points that were on the positive side of them - therefore
			// we save pointers to them.
			m_newFaceIndices.clear();
			m_newHalfEdgeIndices.clear();
			m_disabledFacePointVectors.clear();
			size_t disableCounter = 0;
			for (auto faceIndex : m_visibleFaces) {
				auto& disabledFace = m_mesh.m_faces[faceIndex];
				auto halfEdges = m_mesh.getHalfEdgeIndicesOfFace(disabledFace);
				for (size_t j=0;j<3;j++) {
					if ((disabledFace.m_horizonEdgesOnCurrentIteration & (1<<j)) == 0) {
						if (disableCounter < horizonEdgeCount*2) {
							// Use on this iteration
							m_newHalfEdgeIndices.push_back(halfEdges[j]);
							disableCounter++;
						}
						else {
							// Mark for reusal on later iteration step
							m_mesh.disableHalfEdge(halfEdges[j]);
						}
					}
				}
				// Disable the face, but retain pointer to the points that were on the positive side of it. We need to assign those points
				// to the new faces we create shortly.
				auto t = m_mesh.disableFace(faceIndex);
				if (t) {
					assert(t->size()); // Because we should not assign point vectors to faces unless needed...
					m_disabledFacePointVectors.push_back(std::move(t));
				}
			}
			if (disableCounter < horizonEdgeCount*2) {
				const size_t newHalfEdgesNeeded = horizonEdgeCount*2-disableCounter;
				for (size_t i=0;i<newHalfEdgesNeeded;i++) {
					m_newHalfEdgeIndices.push_back(m_mesh.addHalfEdge());
				}
			}
			
			// Create new faces using the edgeloop
			for (size_t i = 0; i < horizonEdgeCount; i++) {
				const size_t AB = m_horizonEdges[i];

				auto horizonEdgeVertexIndices = m_mesh.getVertexIndicesOfHalfEdge(m_mesh.m_halfEdges[AB]);
				size_t A,B,C;
				A = horizonEdgeVertexIndices[0];
				B = horizonEdgeVertexIndices[1];
				C = activePointIndex;

				const size_t newFaceIndex = m_mesh.addFace();
				m_newFaceIndices.push_back(newFaceIndex);

				const size_t CA = m_newHalfEdgeIndices[2*i+0];
				const size_t BC = m_newHalfEdgeIndices[2*i+1];

				m_mesh.m_halfEdges[AB].m_next = BC;
				m_mesh.m_halfEdges[BC].m_next = CA;
				m_mesh.m_halfEdges[CA].m_next = AB;

				m_mesh.m_halfEdges[BC].m_face = newFaceIndex;
				m_mesh.m_halfEdges[CA].m_face = newFaceIndex;
				m_mesh.m_halfEdges[AB].m_face = newFaceIndex;

				m_mesh.m_halfEdges[CA].m_endVertex = A;
				m_mesh.m_halfEdges[BC].m_endVertex = C;

				auto& newFace = m_mesh.m_faces[newFaceIndex];

				const Vector3<T> planeNormal = mathutils::getTriangleNormal(m_vertexData[A],m_vertexData[B],activePoint);
				newFace.m_P = Plane<T>(planeNormal,activePoint);
				newFace.m_he = AB;

				m_mesh.m_halfEdges[CA].m_opp = m_newHalfEdgeIndices[i>0 ? i*2-1 : 2*horizonEdgeCount-1];
				m_mesh.m_halfEdges[BC].m_opp = m_newHalfEdgeIndices[((i+1)*2) % (horizonEdgeCount*2)];
			}

			// Assign points that were on the positive side of the disabled faces to the new faces.
			for (auto& disabledPoints : m_disabledFacePointVectors) {
				assert(disabledPoints);
				for (const auto& point : *(disabledPoints)) {
					if (point == activePointIndex) {
						continue;
					}
					for (size_t j=0;j<horizonEdgeCount;j++) {
						if (addPointToFace(m_mesh.m_faces[m_newFaceIndices[j]], point)) {
							break;
						}
					}
				}
				// The points are no longer needed: we can move them to the vector pool for reuse.
				reclaimToIndexVectorPool(disabledPoints);
			}

			// Increase face stack size if needed
			for (const auto newFaceIndex : m_newFaceIndices) {
				auto& newFace = m_mesh.m_faces[newFaceIndex];
				if (newFace.m_pointsOnPositiveSide) {
					assert(newFace.m_pointsOnPositiveSide->size()>0);
					if (!newFace.m_inFaceStack) {
						m_faceList.push_back(newFaceIndex);
						newFace.m_inFaceStack = 1;
					}
				}
			}
		}
		
		// Cleanup
		m_indexVectorPool.clear();
	}
	
	/*
	 * Private helper functions
	 */

	template <typename T>
	std::array<size_t,6> QuickHull<T>::getExtremeValues() {
		std::array<size_t,6> outIndices{0,0,0,0,0,0};
		T extremeVals[6] = {m_vertexData[0].x,m_vertexData[0].x,m_vertexData[0].y,m_vertexData[0].y,m_vertexData[0].z,m_vertexData[0].z};
		const size_t vCount = m_vertexData.size();
		for (size_t i=1;i<vCount;i++) {
			const Vector3<T>& pos = m_vertexData[i];
			if (pos.x>extremeVals[0]) {
				extremeVals[0]=pos.x;
				outIndices[0]=i;
			}
			else if (pos.x<extremeVals[1]) {
				extremeVals[1]=pos.x;
				outIndices[1]=i;
			}
			if (pos.y>extremeVals[2]) {
				extremeVals[2]=pos.y;
				outIndices[2]=i;
			}
			else if (pos.y<extremeVals[3]) {
				extremeVals[3]=pos.y;
				outIndices[3]=i;
			}
			if (pos.z>extremeVals[4]) {
				extremeVals[4]=pos.z;
				outIndices[4]=i;
			}
			else if (pos.z<extremeVals[5]) {
				extremeVals[5]=pos.z;
				outIndices[5]=i;
			}
		}
		return outIndices;
	}

	template<typename T>
	bool QuickHull<T>::reorderHorizonEdges(std::vector<size_t>& horizonEdges) {
		const size_t horizonEdgeCount = horizonEdges.size();
		for (size_t i=0;i<horizonEdgeCount-1;i++) {
			const size_t endVertex = m_mesh.m_halfEdges[ horizonEdges[i] ].m_endVertex;
			bool foundNext = false;
			for (size_t j=i+1;j<horizonEdgeCount;j++) {
				const size_t beginVertex = m_mesh.m_halfEdges[ m_mesh.m_halfEdges[horizonEdges[j]].m_opp ].m_endVertex;
				if (beginVertex == endVertex) {
					std::swap(horizonEdges[i+1],horizonEdges[j]);
					foundNext = true;
					break;
				}
			}
			if (!foundNext) {
				return false;
			}
		}
		assert(m_mesh.m_halfEdges[ horizonEdges[horizonEdges.size()-1] ].m_endVertex == m_mesh.m_halfEdges[ m_mesh.m_halfEdges[horizonEdges[0]].m_opp ].m_endVertex);
		return true;
	}
	
	template <typename T>
	T QuickHull<T>::getScale(const std::array<size_t,6>& extremeValues) {
		T s = 0;
		for (size_t i=0;i<6;i++) {
			const T* v = (const T*)(&m_vertexData[extremeValues[i]]);
			v += i/2;
			auto a = std::abs(*v);
			if (a>s) {
				s = a;
			}
		}
		return s;
	}

	template<typename T>
	void QuickHull<T>::setupInitialTetrahedron() {
		const size_t vertexCount = m_vertexData.size();
		
		// If we have at most 4 points, just return a degenerate tetrahedron:
		if (vertexCount <= 4) {
			size_t v[4] = {0,std::min((size_t)1,vertexCount-1),std::min((size_t)2,vertexCount-1),std::min((size_t)3,vertexCount-1)};
			const Vector3<T> N = mathutils::getTriangleNormal(m_vertexData[v[0]],m_vertexData[v[1]],m_vertexData[v[2]]);
			const Plane<T> trianglePlane(N,m_vertexData[v[0]]);
			if (trianglePlane.isPointOnPositiveSide(m_vertexData[v[3]])) {
				std::swap(v[0],v[1]);
			}
			return m_mesh.setup(v[0],v[1],v[2],v[3]);
		}
		
		// Find two most distant extreme points.
		T maxD = m_epsilonSquared;
		std::pair<size_t,size_t> selectedPoints;
		for (size_t i=0;i<6;i++) {
			for (size_t j=i+1;j<6;j++) {
				const T d = m_vertexData[ m_extremeValues[i] ].getSquaredDistanceTo( m_vertexData[ m_extremeValues[j] ] );
				if (d > maxD) {
					maxD=d;
					selectedPoints={m_extremeValues[i],m_extremeValues[j]};
				}
			}
		}
		if (maxD == m_epsilonSquared) {
			// A degenerate case: the point cloud seems to consists of a single point
			return m_mesh.setup(0,std::min((size_t)1,vertexCount-1),std::min((size_t)2,vertexCount-1),std::min((size_t)3,vertexCount-1));
		}
		assert(selectedPoints.first != selectedPoints.second);
		
		// Find the most distant point to the line between the two chosen extreme points.
		const Ray<T> r(m_vertexData[selectedPoints.first], (m_vertexData[selectedPoints.second] - m_vertexData[selectedPoints.first]));
		maxD = m_epsilonSquared;
		size_t maxI=std::numeric_limits<size_t>::max();
		const size_t vCount = m_vertexData.size();
		for (size_t i=0;i<vCount;i++) {
			const T distToRay = mathutils::getSquaredDistanceBetweenPointAndRay(m_vertexData[i],r);
			if (distToRay > maxD) {
				maxD=distToRay;
				maxI=i;
			}
		}
		if (maxD == m_epsilonSquared) {
			// It appears that the point cloud belongs to a 1 dimensional subspace of R^3: convex hull has no volume => return a thin triangle
			// Pick any point other than selectedPoints.first and selectedPoints.second as the third point of the triangle
			auto it = std::find_if(m_vertexData.begin(),m_vertexData.end(),[&](const vec3& ve) {
				return ve != m_vertexData[selectedPoints.first] && ve != m_vertexData[selectedPoints.second];
			});
			const size_t thirdPoint = (it == m_vertexData.end()) ? selectedPoints.first : std::distance(m_vertexData.begin(),it);
			it = std::find_if(m_vertexData.begin(),m_vertexData.end(),[&](const vec3& ve) {
				return ve != m_vertexData[selectedPoints.first] && ve != m_vertexData[selectedPoints.second] && ve != m_vertexData[thirdPoint];
			});
			const size_t fourthPoint = (it == m_vertexData.end()) ? selectedPoints.first : std::distance(m_vertexData.begin(),it);
			return m_mesh.setup(selectedPoints.first,selectedPoints.second,thirdPoint,fourthPoint);
		}

		// These three points form the base triangle for our tetrahedron.
		assert(selectedPoints.first != maxI && selectedPoints.second != maxI);
		std::array<size_t,3> baseTriangle{selectedPoints.first, selectedPoints.second, maxI};
		const Vector3<T> baseTriangleVertices[]={ m_vertexData[baseTriangle[0]], m_vertexData[baseTriangle[1]],  m_vertexData[baseTriangle[2]] };
		
		// Next step is to find the 4th vertex of the tetrahedron. We naturally choose the point farthest away from the triangle plane.
		maxD=m_epsilon;
		maxI=0;
		const Vector3<T> N = mathutils::getTriangleNormal(baseTriangleVertices[0],baseTriangleVertices[1],baseTriangleVertices[2]);
		Plane<T> trianglePlane(N,baseTriangleVertices[0]);
		for (size_t i=0;i<vCount;i++) {
			const T d = std::abs(mathutils::getSignedDistanceToPlane(m_vertexData[i],trianglePlane));
			if (d>maxD) {
				maxD=d;
				maxI=i;
			}
		}
		if (maxD == m_epsilon) {
			// All the points seem to lie on a 2D subspace of R^3. How to handle this? Well, let's add one extra point to the point cloud so that the convex hull will have volume.
			m_planar = true;
			const vec3 N1 = mathutils::getTriangleNormal(baseTriangleVertices[1],baseTriangleVertices[2],baseTriangleVertices[0]);
			m_planarPointCloudTemp.clear();
			m_planarPointCloudTemp.insert(m_planarPointCloudTemp.begin(),m_vertexData.begin(),m_vertexData.end());
			const vec3 extraPoint = N1 + m_vertexData[0];
			m_planarPointCloudTemp.push_back(extraPoint);
			maxI = m_planarPointCloudTemp.size()-1;
			m_vertexData = VertexDataSource<T>(m_planarPointCloudTemp);
		}

		// Enforce CCW orientation (if user prefers clockwise orientation, swap two vertices in each triangle when final mesh is created)
		const Plane<T> triPlane(N,baseTriangleVertices[0]);
		if (triPlane.isPointOnPositiveSide(m_vertexData[maxI])) {
			std::swap(baseTriangle[0],baseTriangle[1]);
		}

		// Create a tetrahedron half edge mesh and compute planes defined by each triangle
		m_mesh.setup(baseTriangle[0],baseTriangle[1],baseTriangle[2],maxI);
		for (auto& f : m_mesh.m_faces) {
			auto v = m_mesh.getVertexIndicesOfFace(f);
			const Vector3<T>& va = m_vertexData[v[0]];
			const Vector3<T>& vb = m_vertexData[v[1]];
			const Vector3<T>& vc = m_vertexData[v[2]];
			const Vector3<T> N1 = mathutils::getTriangleNormal(va, vb, vc);
			const Plane<T> plane(N1,va);
			f.m_P = plane;
		}

		// Finally we assign a face for each vertex outside the tetrahedron (vertices inside the tetrahedron have no role anymore)
		for (size_t i=0;i<vCount;i++) {
			for (auto& face : m_mesh.m_faces) {
				if (addPointToFace(face, i)) {
					break;
				}
			}
		}
	}
	
	/*
	 * Explicit template specifications for float and double
	 */

	template class QuickHull<float>;
	template class QuickHull<double>;
}

