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

#include "btCylinderShape.h"

btCylinderShape::btCylinderShape (const btVector3& halfExtents)
:btConvexInternalShape(),
m_upAxis(1)
{
	btVector3 margin(getMargin(),getMargin(),getMargin());
	m_implicitShapeDimensions = (halfExtents * m_localScaling) - margin;

	setSafeMargin(halfExtents);

	m_shapeType = CYLINDER_SHAPE_PROXYTYPE;
}


btCylinderShapeX::btCylinderShapeX (const btVector3& halfExtents)
:btCylinderShape(halfExtents)
{
	m_upAxis = 0;

}


btCylinderShapeZ::btCylinderShapeZ (const btVector3& halfExtents)
:btCylinderShape(halfExtents)
{
	m_upAxis = 2;

}

void btCylinderShape::getAabb(const btTransform& t,btVector3& aabbMin,btVector3& aabbMax) const
{
	btTransformAabb(getHalfExtentsWithoutMargin(),getMargin(),t,aabbMin,aabbMax);
}

void	btCylinderShape::calculateLocalInertia(btScalar mass,btVector3& inertia) const
{

//Until Bullet 2.77 a box approximation was used, so uncomment this if you need backwards compatibility
//#define USE_BOX_INERTIA_APPROXIMATION 1
#ifndef USE_BOX_INERTIA_APPROXIMATION

	/*
	cylinder is defined as following:
	*
	* - principle axis aligned along y by default, radius in x, z-value not used
	* - for btCylinderShapeX: principle axis aligned along x, radius in y direction, z-value not used
	* - for btCylinderShapeZ: principle axis aligned along z, radius in x direction, y-value not used
	*
	*/

	btScalar radius2;	// square of cylinder radius
	btScalar height2;	// square of cylinder height
	btVector3 halfExtents = getHalfExtentsWithMargin();	// get cylinder dimension
	btScalar div12 = mass / 12.f;
	btScalar div4 = mass / 4.f;
	btScalar div2 = mass / 2.f;
	int idxRadius, idxHeight;

	switch (m_upAxis)	// get indices of radius and height of cylinder
	{
		case 0:		// cylinder is aligned along x
			idxRadius = 1;
			idxHeight = 0;
			break;
		case 2:		// cylinder is aligned along z
			idxRadius = 0;
			idxHeight = 2;
			break;
		default:	// cylinder is aligned along y
			idxRadius = 0;
			idxHeight = 1;
	}

	// calculate squares
	radius2 = halfExtents[idxRadius] * halfExtents[idxRadius];
	height2 = btScalar(4.) * halfExtents[idxHeight] * halfExtents[idxHeight];

	// calculate tensor terms
	btScalar t1 = div12 * height2 + div4 * radius2;
	btScalar t2 = div2 * radius2;

	switch (m_upAxis)	// set diagonal elements of inertia tensor
	{
		case 0:		// cylinder is aligned along x
			inertia.setValue(t2,t1,t1);
			break;
		case 2:		// cylinder is aligned along z
			inertia.setValue(t1,t1,t2);
			break;
		default:	// cylinder is aligned along y
			inertia.setValue(t1,t2,t1);
	}
#else //USE_BOX_INERTIA_APPROXIMATION
	//approximation of box shape
	btVector3 halfExtents = getHalfExtentsWithMargin();

	btScalar lx=btScalar(2.)*(halfExtents.x());
	btScalar ly=btScalar(2.)*(halfExtents.y());
	btScalar lz=btScalar(2.)*(halfExtents.z());

	inertia.setValue(mass/(btScalar(12.0)) * (ly*ly + lz*lz),
					mass/(btScalar(12.0)) * (lx*lx + lz*lz),
					mass/(btScalar(12.0)) * (lx*lx + ly*ly));
#endif //USE_BOX_INERTIA_APPROXIMATION
}


SIMD_FORCE_INLINE btVector3 CylinderLocalSupportX(const btVector3& halfExtents,const btVector3& v) 
{
const int cylinderUpAxis = 0;
const int XX = 1;
const int YY = 0;
const int ZZ = 2;

	//mapping depends on how cylinder local orientation is
	// extents of the cylinder is: X,Y is for radius, and Z for height


	btScalar radius = halfExtents[XX];
	btScalar halfHeight = halfExtents[cylinderUpAxis];


    btVector3 tmp;
	btScalar d ;

    btScalar s = btSqrt(v[XX] * v[XX] + v[ZZ] * v[ZZ]);
    if (s != btScalar(0.0))
	{
        d = radius / s;  
		tmp[XX] = v[XX] * d;
		tmp[YY] = v[YY] < 0.0 ? -halfHeight : halfHeight;
		tmp[ZZ] = v[ZZ] * d;
		return tmp;
	}
    else
	{
	    tmp[XX] = radius;
		tmp[YY] = v[YY] < 0.0 ? -halfHeight : halfHeight;
		tmp[ZZ] = btScalar(0.0);
		return tmp;
    }


}






inline  btVector3 CylinderLocalSupportY(const btVector3& halfExtents,const btVector3& v) 
{

const int cylinderUpAxis = 1;
const int XX = 0;
const int YY = 1;
const int ZZ = 2;


	btScalar radius = halfExtents[XX];
	btScalar halfHeight = halfExtents[cylinderUpAxis];


    btVector3 tmp;
	btScalar d ;

    btScalar s = btSqrt(v[XX] * v[XX] + v[ZZ] * v[ZZ]);
    if (s != btScalar(0.0))
	{
        d = radius / s;  
		tmp[XX] = v[XX] * d;
		tmp[YY] = v[YY] < 0.0 ? -halfHeight : halfHeight;
		tmp[ZZ] = v[ZZ] * d;
		return tmp;
	}
    else
	{
	    tmp[XX] = radius;
		tmp[YY] = v[YY] < 0.0 ? -halfHeight : halfHeight;
		tmp[ZZ] = btScalar(0.0);
		return tmp;
    }

}

inline btVector3 CylinderLocalSupportZ(const btVector3& halfExtents,const btVector3& v) 
{
const int cylinderUpAxis = 2;
const int XX = 0;
const int YY = 2;
const int ZZ = 1;

	//mapping depends on how cylinder local orientation is
	// extents of the cylinder is: X,Y is for radius, and Z for height


	btScalar radius = halfExtents[XX];
	btScalar halfHeight = halfExtents[cylinderUpAxis];


    btVector3 tmp;
	btScalar d ;

    btScalar s = btSqrt(v[XX] * v[XX] + v[ZZ] * v[ZZ]);
    if (s != btScalar(0.0))
	{
        d = radius / s;  
		tmp[XX] = v[XX] * d;
		tmp[YY] = v[YY] < 0.0 ? -halfHeight : halfHeight;
		tmp[ZZ] = v[ZZ] * d;
		return tmp;
	}
    else
	{
	    tmp[XX] = radius;
		tmp[YY] = v[YY] < 0.0 ? -halfHeight : halfHeight;
		tmp[ZZ] = btScalar(0.0);
		return tmp;
    }


}

btVector3	btCylinderShapeX::localGetSupportingVertexWithoutMargin(const btVector3& vec)const
{
	return CylinderLocalSupportX(getHalfExtentsWithoutMargin(),vec);
}


btVector3	btCylinderShapeZ::localGetSupportingVertexWithoutMargin(const btVector3& vec)const
{
	return CylinderLocalSupportZ(getHalfExtentsWithoutMargin(),vec);
}
btVector3	btCylinderShape::localGetSupportingVertexWithoutMargin(const btVector3& vec)const
{
	return CylinderLocalSupportY(getHalfExtentsWithoutMargin(),vec);
}

void	btCylinderShape::batchedUnitVectorGetSupportingVertexWithoutMargin(const btVector3* vectors,btVector3* supportVerticesOut,int numVectors) const
{
	for (int i=0;i<numVectors;i++)
	{
		supportVerticesOut[i] = CylinderLocalSupportY(getHalfExtentsWithoutMargin(),vectors[i]);
	}
}

void	btCylinderShapeZ::batchedUnitVectorGetSupportingVertexWithoutMargin(const btVector3* vectors,btVector3* supportVerticesOut,int numVectors) const
{
	for (int i=0;i<numVectors;i++)
	{
		supportVerticesOut[i] = CylinderLocalSupportZ(getHalfExtentsWithoutMargin(),vectors[i]);
	}
}




void	btCylinderShapeX::batchedUnitVectorGetSupportingVertexWithoutMargin(const btVector3* vectors,btVector3* supportVerticesOut,int numVectors) const
{
	for (int i=0;i<numVectors;i++)
	{
		supportVerticesOut[i] = CylinderLocalSupportX(getHalfExtentsWithoutMargin(),vectors[i]);
	}
}


