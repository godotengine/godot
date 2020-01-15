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

#ifndef BT_HEIGHTFIELD_TERRAIN_SHAPE_H
#define BT_HEIGHTFIELD_TERRAIN_SHAPE_H

#include "btConcaveShape.h"
#include "LinearMath/btAlignedObjectArray.h"

///btHeightfieldTerrainShape simulates a 2D heightfield terrain
/**
  The caller is responsible for maintaining the heightfield array; this
  class does not make a copy.

  The heightfield can be dynamic so long as the min/max height values
  capture the extremes (heights must always be in that range).

  The local origin of the heightfield is assumed to be the exact
  center (as determined by width and length and height, with each
  axis multiplied by the localScaling).

  \b NOTE: be careful with coordinates.  If you have a heightfield with a local
  min height of -100m, and a max height of +500m, you may be tempted to place it
  at the origin (0,0) and expect the heights in world coordinates to be
  -100 to +500 meters.
  Actually, the heights will be -300 to +300m, because bullet will re-center
  the heightfield based on its AABB (which is determined by the min/max
  heights).  So keep in mind that once you create a btHeightfieldTerrainShape
  object, the heights will be adjusted relative to the center of the AABB.  This
  is different to the behavior of many rendering engines, but is useful for
  physics engines.

  Most (but not all) rendering and heightfield libraries assume upAxis = 1
  (that is, the y-axis is "up").  This class allows any of the 3 coordinates
  to be "up".  Make sure your choice of axis is consistent with your rendering
  system.

  The heightfield heights are determined from the data type used for the
  heightfieldData array.  

   - PHY_UCHAR: height at a point is the uchar value at the
       grid point, multipled by heightScale.  uchar isn't recommended
       because of its inability to deal with negative values, and
       low resolution (8-bit).

   - PHY_SHORT: height at a point is the short int value at that grid
       point, multipled by heightScale.

   - PHY_FLOAT: height at a point is the float value at that grid
       point.  heightScale is ignored when using the float heightfield
       data type.

  Whatever the caller specifies as minHeight and maxHeight will be honored.
  The class will not inspect the heightfield to discover the actual minimum
  or maximum heights.  These values are used to determine the heightfield's
  axis-aligned bounding box, multiplied by localScaling.

  For usage and testing see the TerrainDemo.
 */
ATTRIBUTE_ALIGNED16(class)
btHeightfieldTerrainShape : public btConcaveShape
{
public:
	struct Range
	{
		btScalar min;
		btScalar max;
	};

protected:
	btVector3 m_localAabbMin;
	btVector3 m_localAabbMax;
	btVector3 m_localOrigin;

	///terrain data
	int m_heightStickWidth;
	int m_heightStickLength;
	btScalar m_minHeight;
	btScalar m_maxHeight;
	btScalar m_width;
	btScalar m_length;
	btScalar m_heightScale;
	union {
		const unsigned char* m_heightfieldDataUnsignedChar;
		const short* m_heightfieldDataShort;
		const btScalar* m_heightfieldDataFloat;
		const void* m_heightfieldDataUnknown;
	};

	PHY_ScalarType m_heightDataType;
	bool m_flipQuadEdges;
	bool m_useDiamondSubdivision;
	bool m_useZigzagSubdivision;
	bool m_flipTriangleWinding;
	int m_upAxis;

	btVector3 m_localScaling;

	// Accelerator
	btAlignedObjectArray<Range> m_vboundsGrid;
	int m_vboundsGridWidth;
	int m_vboundsGridLength;
	int m_vboundsChunkSize;

	int m_userIndex2;
	btScalar m_userValue3;

	struct btTriangleInfoMap* m_triangleInfoMap;

	virtual btScalar getRawHeightFieldValue(int x, int y) const;
	void quantizeWithClamp(int* out, const btVector3& point, int isMax) const;

	/// protected initialization
	/**
	  Handles the work of constructors so that public constructors can be
	  backwards-compatible without a lot of copy/paste.
	 */
	void initialize(int heightStickWidth, int heightStickLength,
					const void* heightfieldData, btScalar heightScale,
					btScalar minHeight, btScalar maxHeight, int upAxis,
					PHY_ScalarType heightDataType, bool flipQuadEdges);

public:
	BT_DECLARE_ALIGNED_ALLOCATOR();

	/// preferred constructor
	/**
	  This constructor supports a range of heightfield
	  data types, and allows for a non-zero minimum height value.
	  heightScale is needed for any integer-based heightfield data types.
	 */
	btHeightfieldTerrainShape(int heightStickWidth, int heightStickLength,
							  const void* heightfieldData, btScalar heightScale,
							  btScalar minHeight, btScalar maxHeight,
							  int upAxis, PHY_ScalarType heightDataType,
							  bool flipQuadEdges);

	/// legacy constructor
	/**
	  The legacy constructor assumes the heightfield has a minimum height
	  of zero.  Only unsigned char or floats are supported.  For legacy
	  compatibility reasons, heightScale is calculated as maxHeight / 65535 
	  (and is only used when useFloatData = false).
 	 */
	btHeightfieldTerrainShape(int heightStickWidth, int heightStickLength, const void* heightfieldData, btScalar maxHeight, int upAxis, bool useFloatData, bool flipQuadEdges);

	virtual ~btHeightfieldTerrainShape();

	void setUseDiamondSubdivision(bool useDiamondSubdivision = true) { m_useDiamondSubdivision = useDiamondSubdivision; }

	///could help compatibility with Ogre heightfields. See https://code.google.com/p/bullet/issues/detail?id=625
	void setUseZigzagSubdivision(bool useZigzagSubdivision = true) { m_useZigzagSubdivision = useZigzagSubdivision; }

	void setFlipTriangleWinding(bool flipTriangleWinding)
	{
		m_flipTriangleWinding = flipTriangleWinding;
	}
	virtual void getAabb(const btTransform& t, btVector3& aabbMin, btVector3& aabbMax) const;

	virtual void processAllTriangles(btTriangleCallback * callback, const btVector3& aabbMin, const btVector3& aabbMax) const;

	virtual void calculateLocalInertia(btScalar mass, btVector3 & inertia) const;

	virtual void setLocalScaling(const btVector3& scaling);

	virtual const btVector3& getLocalScaling() const;

	void getVertex(int x, int y, btVector3& vertex) const;

	void performRaycast(btTriangleCallback * callback, const btVector3& raySource, const btVector3& rayTarget) const;

	void buildAccelerator(int chunkSize = 16);
	void clearAccelerator();

	int getUpAxis() const
	{
		return m_upAxis;
	}
	//debugging
	virtual const char* getName() const { return "HEIGHTFIELD"; }

	
	void setUserIndex2(int index)
	{
		m_userIndex2 = index;
	}
	int getUserIndex2() const
	{
		return m_userIndex2;
	}
	void setUserValue3(btScalar value)
	{
		m_userValue3 = value;
	}
	btScalar getUserValue3() const
	{
		return m_userValue3;
	}
	const struct btTriangleInfoMap* getTriangleInfoMap() const
	{
		return m_triangleInfoMap;
	}
	struct btTriangleInfoMap* getTriangleInfoMap()
	{
		return m_triangleInfoMap;
	}
	void setTriangleInfoMap(btTriangleInfoMap* map)
	{
		m_triangleInfoMap = map;
	}
	const unsigned char* getHeightfieldRawData() const
	{
		return m_heightfieldDataUnsignedChar;
	}
};

#endif  //BT_HEIGHTFIELD_TERRAIN_SHAPE_H