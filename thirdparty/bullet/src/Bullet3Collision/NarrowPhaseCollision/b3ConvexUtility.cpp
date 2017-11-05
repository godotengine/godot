/*
Copyright (c) 2012 Advanced Micro Devices, Inc.  

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/
//Originally written by Erwin Coumans


#include "b3ConvexUtility.h"
#include "Bullet3Geometry/b3ConvexHullComputer.h"
#include "Bullet3Geometry/b3GrahamScan2dConvexHull.h"
#include "Bullet3Common/b3Quaternion.h"
#include "Bullet3Common/b3HashMap.h"





b3ConvexUtility::~b3ConvexUtility()
{
}

bool	b3ConvexUtility::initializePolyhedralFeatures(const b3Vector3* orgVertices, int numPoints, bool mergeCoplanarTriangles)
{
	
	

	b3ConvexHullComputer conv;
	conv.compute(&orgVertices[0].getX(), sizeof(b3Vector3),numPoints,0.f,0.f);

	b3AlignedObjectArray<b3Vector3> faceNormals;
	int numFaces = conv.faces.size();
	faceNormals.resize(numFaces);
	b3ConvexHullComputer* convexUtil = &conv;

	
	b3AlignedObjectArray<b3MyFace>	tmpFaces;
	tmpFaces.resize(numFaces);

	int numVertices = convexUtil->vertices.size();
	m_vertices.resize(numVertices);
	for (int p=0;p<numVertices;p++)
	{
		m_vertices[p] = convexUtil->vertices[p];
	}


	for (int i=0;i<numFaces;i++)
	{
		int face = convexUtil->faces[i];
		//printf("face=%d\n",face);
		const b3ConvexHullComputer::Edge*  firstEdge = &convexUtil->edges[face];
		const b3ConvexHullComputer::Edge*  edge = firstEdge;

		b3Vector3 edges[3];
		int numEdges = 0;
		//compute face normals

		do
		{
			
			int src = edge->getSourceVertex();
			tmpFaces[i].m_indices.push_back(src);
			int targ = edge->getTargetVertex();
			b3Vector3 wa = convexUtil->vertices[src];

			b3Vector3 wb = convexUtil->vertices[targ];
			b3Vector3 newEdge = wb-wa;
			newEdge.normalize();
			if (numEdges<2)
				edges[numEdges++] = newEdge;

			edge = edge->getNextEdgeOfFace();
		} while (edge!=firstEdge);

		b3Scalar planeEq = 1e30f;

		
		if (numEdges==2)
		{
			faceNormals[i] = edges[0].cross(edges[1]);
			faceNormals[i].normalize();
			tmpFaces[i].m_plane[0] = faceNormals[i].getX();
			tmpFaces[i].m_plane[1] = faceNormals[i].getY();
			tmpFaces[i].m_plane[2] = faceNormals[i].getZ();
			tmpFaces[i].m_plane[3] = planeEq;

		}
		else
		{
			b3Assert(0);//degenerate?
			faceNormals[i].setZero();
		}

		for (int v=0;v<tmpFaces[i].m_indices.size();v++)
		{
			b3Scalar eq = m_vertices[tmpFaces[i].m_indices[v]].dot(faceNormals[i]);
			if (planeEq>eq)
			{
				planeEq=eq;
			}
		}
		tmpFaces[i].m_plane[3] = -planeEq;
	}

	//merge coplanar faces and copy them to m_polyhedron

	b3Scalar faceWeldThreshold= 0.999f;
	b3AlignedObjectArray<int> todoFaces;
	for (int i=0;i<tmpFaces.size();i++)
		todoFaces.push_back(i);

	while (todoFaces.size())
	{
		b3AlignedObjectArray<int> coplanarFaceGroup;
		int refFace = todoFaces[todoFaces.size()-1];

		coplanarFaceGroup.push_back(refFace);
		b3MyFace& faceA = tmpFaces[refFace];
		todoFaces.pop_back();

		b3Vector3 faceNormalA = b3MakeVector3(faceA.m_plane[0],faceA.m_plane[1],faceA.m_plane[2]);
		for (int j=todoFaces.size()-1;j>=0;j--)
		{
			int i = todoFaces[j];
			b3MyFace& faceB = tmpFaces[i];
			b3Vector3 faceNormalB = b3MakeVector3(faceB.m_plane[0],faceB.m_plane[1],faceB.m_plane[2]);
			if (faceNormalA.dot(faceNormalB)>faceWeldThreshold)
			{
				coplanarFaceGroup.push_back(i);
				todoFaces.remove(i);
			}
		}


		bool did_merge = false;
		if (coplanarFaceGroup.size()>1)
		{
			//do the merge: use Graham Scan 2d convex hull

			b3AlignedObjectArray<b3GrahamVector3> orgpoints;
			b3Vector3 averageFaceNormal = b3MakeVector3(0,0,0);

			for (int i=0;i<coplanarFaceGroup.size();i++)
			{
//				m_polyhedron->m_faces.push_back(tmpFaces[coplanarFaceGroup[i]]);

				b3MyFace& face = tmpFaces[coplanarFaceGroup[i]];
				b3Vector3 faceNormal = b3MakeVector3(face.m_plane[0],face.m_plane[1],face.m_plane[2]);
				averageFaceNormal+=faceNormal;
				for (int f=0;f<face.m_indices.size();f++)
				{
					int orgIndex = face.m_indices[f];
					b3Vector3 pt = m_vertices[orgIndex];
					
					bool found = false;

					for (int i=0;i<orgpoints.size();i++)
					{
						//if ((orgpoints[i].m_orgIndex == orgIndex) || ((rotatedPt-orgpoints[i]).length2()<0.0001))
						if (orgpoints[i].m_orgIndex == orgIndex)
						{
							found=true;
							break;
						}
					}
					if (!found)
						orgpoints.push_back(b3GrahamVector3(pt,orgIndex));
				}
			}

			

			b3MyFace combinedFace;
			for (int i=0;i<4;i++)
				combinedFace.m_plane[i] = tmpFaces[coplanarFaceGroup[0]].m_plane[i];

			b3AlignedObjectArray<b3GrahamVector3> hull;

			averageFaceNormal.normalize();
			b3GrahamScanConvexHull2D(orgpoints,hull,averageFaceNormal);

			for (int i=0;i<hull.size();i++)
			{
				combinedFace.m_indices.push_back(hull[i].m_orgIndex);
				for(int k = 0; k < orgpoints.size(); k++) 
				{
					if(orgpoints[k].m_orgIndex == hull[i].m_orgIndex) 
					{
						orgpoints[k].m_orgIndex = -1; // invalidate...
						break;
					}
				}
			}

			// are there rejected vertices?
			bool reject_merge = false;
			


			for(int i = 0; i < orgpoints.size(); i++) {
				if(orgpoints[i].m_orgIndex == -1)
					continue; // this is in the hull...
				// this vertex is rejected -- is anybody else using this vertex?
				for(int j = 0; j < tmpFaces.size(); j++) {
					
					b3MyFace& face = tmpFaces[j];
					// is this a face of the current coplanar group?
					bool is_in_current_group = false;
					for(int k = 0; k < coplanarFaceGroup.size(); k++) {
						if(coplanarFaceGroup[k] == j) {
							is_in_current_group = true;
							break;
						}
					}
					if(is_in_current_group) // ignore this face...
						continue;
					// does this face use this rejected vertex?
					for(int v = 0; v < face.m_indices.size(); v++) {
						if(face.m_indices[v] == orgpoints[i].m_orgIndex) {
							// this rejected vertex is used in another face -- reject merge
							reject_merge = true;
							break;
						}
					}
					if(reject_merge)
						break;
				}
				if(reject_merge)
					break;
			}

			if (!reject_merge)
			{
				// do this merge!
				did_merge = true;
				m_faces.push_back(combinedFace);
			}
		}
		if(!did_merge)
		{
			for (int i=0;i<coplanarFaceGroup.size();i++)
			{
				b3MyFace face = tmpFaces[coplanarFaceGroup[i]];
				m_faces.push_back(face);
			}

		} 



	}

	initialize();

	return true;
}






inline bool IsAlmostZero(const b3Vector3& v)
{
	if(fabsf(v.getX())>1e-6 || fabsf(v.getY())>1e-6 || fabsf(v.getZ())>1e-6)	return false;
	return true;
}

struct b3InternalVertexPair
{
	b3InternalVertexPair(short int v0,short int v1)
		:m_v0(v0),
		m_v1(v1)
	{
		if (m_v1>m_v0)
			b3Swap(m_v0,m_v1);
	}
	short int m_v0;
	short int m_v1;
	int getHash() const
	{
		return m_v0+(m_v1<<16);
	}
	bool equals(const b3InternalVertexPair& other) const
	{
		return m_v0==other.m_v0 && m_v1==other.m_v1;
	}
};

struct b3InternalEdge
{
	b3InternalEdge()
		:m_face0(-1),
		m_face1(-1)
	{
	}
	short int m_face0;
	short int m_face1;
};

//

#ifdef TEST_INTERNAL_OBJECTS
bool b3ConvexUtility::testContainment() const
{
	for(int p=0;p<8;p++)
	{
		b3Vector3 LocalPt;
		if(p==0)		LocalPt = m_localCenter + b3Vector3(m_extents[0], m_extents[1], m_extents[2]);
		else if(p==1)	LocalPt = m_localCenter + b3Vector3(m_extents[0], m_extents[1], -m_extents[2]);
		else if(p==2)	LocalPt = m_localCenter + b3Vector3(m_extents[0], -m_extents[1], m_extents[2]);
		else if(p==3)	LocalPt = m_localCenter + b3Vector3(m_extents[0], -m_extents[1], -m_extents[2]);
		else if(p==4)	LocalPt = m_localCenter + b3Vector3(-m_extents[0], m_extents[1], m_extents[2]);
		else if(p==5)	LocalPt = m_localCenter + b3Vector3(-m_extents[0], m_extents[1], -m_extents[2]);
		else if(p==6)	LocalPt = m_localCenter + b3Vector3(-m_extents[0], -m_extents[1], m_extents[2]);
		else if(p==7)	LocalPt = m_localCenter + b3Vector3(-m_extents[0], -m_extents[1], -m_extents[2]);

		for(int i=0;i<m_faces.size();i++)
		{
			const b3Vector3 Normal(m_faces[i].m_plane[0], m_faces[i].m_plane[1], m_faces[i].m_plane[2]);
			const b3Scalar d = LocalPt.dot(Normal) + m_faces[i].m_plane[3];
			if(d>0.0f)
				return false;
		}
	}
	return true;
}
#endif

void	b3ConvexUtility::initialize()
{

	b3HashMap<b3InternalVertexPair,b3InternalEdge> edges;

	b3Scalar TotalArea = 0.0f;
	
	m_localCenter.setValue(0, 0, 0);
	for(int i=0;i<m_faces.size();i++)
	{
		int numVertices = m_faces[i].m_indices.size();
		int NbTris = numVertices;
		for(int j=0;j<NbTris;j++)
		{
			int k = (j+1)%numVertices;
			b3InternalVertexPair vp(m_faces[i].m_indices[j],m_faces[i].m_indices[k]);
			b3InternalEdge* edptr = edges.find(vp);
			b3Vector3 edge = m_vertices[vp.m_v1]-m_vertices[vp.m_v0];
			edge.normalize();

			bool found = false;
			b3Vector3 diff,diff2;

			for (int p=0;p<m_uniqueEdges.size();p++)
			{
				diff = m_uniqueEdges[p]-edge;
				diff2 = m_uniqueEdges[p]+edge;

			//	if ((diff.length2()==0.f) || 
				//	(diff2.length2()==0.f))

				if (IsAlmostZero(diff) || 
				IsAlmostZero(diff2))
				{
					found = true;
					break;
				}
			}

			if (!found)
			{
				m_uniqueEdges.push_back(edge);
			}

			if (edptr)
			{
					//TBD: figure out why I added this assert
//				b3Assert(edptr->m_face0>=0);
	//			b3Assert(edptr->m_face1<0);
				edptr->m_face1 = i;
			} else
			{
				b3InternalEdge ed;
				ed.m_face0 = i;
				edges.insert(vp,ed);
			}
		}
	}

#ifdef USE_CONNECTED_FACES
	for(int i=0;i<m_faces.size();i++)
	{
		int numVertices = m_faces[i].m_indices.size();
		m_faces[i].m_connectedFaces.resize(numVertices);

		for(int j=0;j<numVertices;j++)
		{
			int k = (j+1)%numVertices;
			b3InternalVertexPair vp(m_faces[i].m_indices[j],m_faces[i].m_indices[k]);
			b3InternalEdge* edptr = edges.find(vp);
			b3Assert(edptr);
			b3Assert(edptr->m_face0>=0);
			b3Assert(edptr->m_face1>=0);

			int connectedFace = (edptr->m_face0==i)?edptr->m_face1:edptr->m_face0;
			m_faces[i].m_connectedFaces[j] = connectedFace;
		}
	}
#endif//USE_CONNECTED_FACES

	for(int i=0;i<m_faces.size();i++)
	{
		int numVertices = m_faces[i].m_indices.size();
		int NbTris = numVertices-2;
		
		const b3Vector3& p0 = m_vertices[m_faces[i].m_indices[0]];
		for(int j=1;j<=NbTris;j++)
		{
			int k = (j+1)%numVertices;
			const b3Vector3& p1 = m_vertices[m_faces[i].m_indices[j]];
			const b3Vector3& p2 = m_vertices[m_faces[i].m_indices[k]];
			b3Scalar Area = ((p0 - p1).cross(p0 - p2)).length() * 0.5f;
			b3Vector3 Center = (p0+p1+p2)/3.0f;
			m_localCenter += Area * Center;
			TotalArea += Area;
		}
	}
	m_localCenter /= TotalArea;




#ifdef TEST_INTERNAL_OBJECTS
	if(1)
	{
		m_radius = FLT_MAX;
		for(int i=0;i<m_faces.size();i++)
		{
			const b3Vector3 Normal(m_faces[i].m_plane[0], m_faces[i].m_plane[1], m_faces[i].m_plane[2]);
			const b3Scalar dist = b3Fabs(m_localCenter.dot(Normal) + m_faces[i].m_plane[3]);
			if(dist<m_radius)
				m_radius = dist;
		}

	
		b3Scalar MinX = FLT_MAX;
		b3Scalar MinY = FLT_MAX;
		b3Scalar MinZ = FLT_MAX;
		b3Scalar MaxX = -FLT_MAX;
		b3Scalar MaxY = -FLT_MAX;
		b3Scalar MaxZ = -FLT_MAX;
		for(int i=0; i<m_vertices.size(); i++)
		{
			const b3Vector3& pt = m_vertices[i];
			if(pt.getX()<MinX)	MinX = pt.getX();
			if(pt.getX()>MaxX)	MaxX = pt.getX();
			if(pt.getY()<MinY)	MinY = pt.getY();
			if(pt.getY()>MaxY)	MaxY = pt.getY();
			if(pt.getZ()<MinZ)	MinZ = pt.getZ();
			if(pt.getZ()>MaxZ)	MaxZ = pt.getZ();
		}
		mC.setValue(MaxX+MinX, MaxY+MinY, MaxZ+MinZ);
		mE.setValue(MaxX-MinX, MaxY-MinY, MaxZ-MinZ);



//		const b3Scalar r = m_radius / sqrtf(2.0f);
		const b3Scalar r = m_radius / sqrtf(3.0f);
		const int LargestExtent = mE.maxAxis();
		const b3Scalar Step = (mE[LargestExtent]*0.5f - r)/1024.0f;
		m_extents[0] = m_extents[1] = m_extents[2] = r;
		m_extents[LargestExtent] = mE[LargestExtent]*0.5f;
		bool FoundBox = false;
		for(int j=0;j<1024;j++)
		{
			if(testContainment())
			{
				FoundBox = true;
				break;
			}

			m_extents[LargestExtent] -= Step;
		}
		if(!FoundBox)
		{
			m_extents[0] = m_extents[1] = m_extents[2] = r;
		}
		else
		{
			// Refine the box
			const b3Scalar Step = (m_radius - r)/1024.0f;
			const int e0 = (1<<LargestExtent) & 3;
			const int e1 = (1<<e0) & 3;

			for(int j=0;j<1024;j++)
			{
				const b3Scalar Saved0 = m_extents[e0];
				const b3Scalar Saved1 = m_extents[e1];
				m_extents[e0] += Step;
				m_extents[e1] += Step;

				if(!testContainment())
				{
					m_extents[e0] = Saved0;
					m_extents[e1] = Saved1;
					break;
				}
			}
		}
	}
#endif
}
