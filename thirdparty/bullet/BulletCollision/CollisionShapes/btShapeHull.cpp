/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2009 Erwin Coumans  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

//btShapeHull was implemented by John McCutchan.


#include "btShapeHull.h"
#include "LinearMath/btConvexHull.h"

#define NUM_UNITSPHERE_POINTS 42

btShapeHull::btShapeHull (const btConvexShape* shape)
{
	m_shape = shape;
	m_vertices.clear ();
	m_indices.clear();
	m_numIndices = 0;
}

btShapeHull::~btShapeHull ()
{
	m_indices.clear();	
	m_vertices.clear ();
}

bool
btShapeHull::buildHull (btScalar /*margin*/)
{
	int numSampleDirections = NUM_UNITSPHERE_POINTS;
	{
		int numPDA = m_shape->getNumPreferredPenetrationDirections();
		if (numPDA)
		{
			for (int i=0;i<numPDA;i++)
			{
				btVector3 norm;
				m_shape->getPreferredPenetrationDirection(i,norm);
				getUnitSpherePoints()[numSampleDirections] = norm;
				numSampleDirections++;
			}
		}
	}

	btVector3 supportPoints[NUM_UNITSPHERE_POINTS+MAX_PREFERRED_PENETRATION_DIRECTIONS*2];
	int i;
	for (i = 0; i < numSampleDirections; i++)
	{
		supportPoints[i] = m_shape->localGetSupportingVertex(getUnitSpherePoints()[i]);
	}

	HullDesc hd;
	hd.mFlags = QF_TRIANGLES;
	hd.mVcount = static_cast<unsigned int>(numSampleDirections);

#ifdef BT_USE_DOUBLE_PRECISION
	hd.mVertices = &supportPoints[0];
	hd.mVertexStride = sizeof(btVector3);
#else
	hd.mVertices = &supportPoints[0];
	hd.mVertexStride = sizeof (btVector3);
#endif

	HullLibrary hl;
	HullResult hr;
	if (hl.CreateConvexHull (hd, hr) == QE_FAIL)
	{
		return false;
	}

	m_vertices.resize (static_cast<int>(hr.mNumOutputVertices));


	for (i = 0; i < static_cast<int>(hr.mNumOutputVertices); i++)
	{
		m_vertices[i] = hr.m_OutputVertices[i];
	}
	m_numIndices = hr.mNumIndices;
	m_indices.resize(static_cast<int>(m_numIndices));
	for (i = 0; i < static_cast<int>(m_numIndices); i++)
	{
		m_indices[i] = hr.m_Indices[i];
	}

	// free temporary hull result that we just copied
	hl.ReleaseResult (hr);

	return true;
}

int
btShapeHull::numTriangles () const
{
	return static_cast<int>(m_numIndices / 3);
}

int
btShapeHull::numVertices () const
{
	return m_vertices.size ();
}

int
btShapeHull::numIndices () const
{
	return static_cast<int>(m_numIndices);
}


btVector3* btShapeHull::getUnitSpherePoints()
{
	static btVector3 sUnitSpherePoints[NUM_UNITSPHERE_POINTS+MAX_PREFERRED_PENETRATION_DIRECTIONS*2] = 
	{
		btVector3(btScalar(0.000000) , btScalar(-0.000000),btScalar(-1.000000)),
		btVector3(btScalar(0.723608) , btScalar(-0.525725),btScalar(-0.447219)),
		btVector3(btScalar(-0.276388) , btScalar(-0.850649),btScalar(-0.447219)),
		btVector3(btScalar(-0.894426) , btScalar(-0.000000),btScalar(-0.447216)),
		btVector3(btScalar(-0.276388) , btScalar(0.850649),btScalar(-0.447220)),
		btVector3(btScalar(0.723608) , btScalar(0.525725),btScalar(-0.447219)),
		btVector3(btScalar(0.276388) , btScalar(-0.850649),btScalar(0.447220)),
		btVector3(btScalar(-0.723608) , btScalar(-0.525725),btScalar(0.447219)),
		btVector3(btScalar(-0.723608) , btScalar(0.525725),btScalar(0.447219)),
		btVector3(btScalar(0.276388) , btScalar(0.850649),btScalar(0.447219)),
		btVector3(btScalar(0.894426) , btScalar(0.000000),btScalar(0.447216)),
		btVector3(btScalar(-0.000000) , btScalar(0.000000),btScalar(1.000000)),
		btVector3(btScalar(0.425323) , btScalar(-0.309011),btScalar(-0.850654)),
		btVector3(btScalar(-0.162456) , btScalar(-0.499995),btScalar(-0.850654)),
		btVector3(btScalar(0.262869) , btScalar(-0.809012),btScalar(-0.525738)),
		btVector3(btScalar(0.425323) , btScalar(0.309011),btScalar(-0.850654)),
		btVector3(btScalar(0.850648) , btScalar(-0.000000),btScalar(-0.525736)),
		btVector3(btScalar(-0.525730) , btScalar(-0.000000),btScalar(-0.850652)),
		btVector3(btScalar(-0.688190) , btScalar(-0.499997),btScalar(-0.525736)),
		btVector3(btScalar(-0.162456) , btScalar(0.499995),btScalar(-0.850654)),
		btVector3(btScalar(-0.688190) , btScalar(0.499997),btScalar(-0.525736)),
		btVector3(btScalar(0.262869) , btScalar(0.809012),btScalar(-0.525738)),
		btVector3(btScalar(0.951058) , btScalar(0.309013),btScalar(0.000000)),
		btVector3(btScalar(0.951058) , btScalar(-0.309013),btScalar(0.000000)),
		btVector3(btScalar(0.587786) , btScalar(-0.809017),btScalar(0.000000)),
		btVector3(btScalar(0.000000) , btScalar(-1.000000),btScalar(0.000000)),
		btVector3(btScalar(-0.587786) , btScalar(-0.809017),btScalar(0.000000)),
		btVector3(btScalar(-0.951058) , btScalar(-0.309013),btScalar(-0.000000)),
		btVector3(btScalar(-0.951058) , btScalar(0.309013),btScalar(-0.000000)),
		btVector3(btScalar(-0.587786) , btScalar(0.809017),btScalar(-0.000000)),
		btVector3(btScalar(-0.000000) , btScalar(1.000000),btScalar(-0.000000)),
		btVector3(btScalar(0.587786) , btScalar(0.809017),btScalar(-0.000000)),
		btVector3(btScalar(0.688190) , btScalar(-0.499997),btScalar(0.525736)),
		btVector3(btScalar(-0.262869) , btScalar(-0.809012),btScalar(0.525738)),
		btVector3(btScalar(-0.850648) , btScalar(0.000000),btScalar(0.525736)),
		btVector3(btScalar(-0.262869) , btScalar(0.809012),btScalar(0.525738)),
		btVector3(btScalar(0.688190) , btScalar(0.499997),btScalar(0.525736)),
		btVector3(btScalar(0.525730) , btScalar(0.000000),btScalar(0.850652)),
		btVector3(btScalar(0.162456) , btScalar(-0.499995),btScalar(0.850654)),
		btVector3(btScalar(-0.425323) , btScalar(-0.309011),btScalar(0.850654)),
		btVector3(btScalar(-0.425323) , btScalar(0.309011),btScalar(0.850654)),
		btVector3(btScalar(0.162456) , btScalar(0.499995),btScalar(0.850654))
	};
	return sUnitSpherePoints;
}

