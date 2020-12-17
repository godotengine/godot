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

#include "btHeightfieldTerrainShape.h"

#include "LinearMath/btTransformUtil.h"

btHeightfieldTerrainShape::btHeightfieldTerrainShape(
	int heightStickWidth, int heightStickLength, const void* heightfieldData,
	btScalar heightScale, btScalar minHeight, btScalar maxHeight, int upAxis,
	PHY_ScalarType hdt, bool flipQuadEdges)
	:m_userValue3(0),
	m_triangleInfoMap(0)
{
	initialize(heightStickWidth, heightStickLength, heightfieldData,
			   heightScale, minHeight, maxHeight, upAxis, hdt,
			   flipQuadEdges);
}

btHeightfieldTerrainShape::btHeightfieldTerrainShape(int heightStickWidth, int heightStickLength, const void* heightfieldData, btScalar maxHeight, int upAxis, bool useFloatData, bool flipQuadEdges)
	:	m_userValue3(0),
	m_triangleInfoMap(0)
{
	// legacy constructor: support only float or unsigned char,
	// 	and min height is zero
	PHY_ScalarType hdt = (useFloatData) ? PHY_FLOAT : PHY_UCHAR;
	btScalar minHeight = 0.0f;

	// previously, height = uchar * maxHeight / 65535.
	// So to preserve legacy behavior, heightScale = maxHeight / 65535
	btScalar heightScale = maxHeight / 65535;

	initialize(heightStickWidth, heightStickLength, heightfieldData,
			   heightScale, minHeight, maxHeight, upAxis, hdt,
			   flipQuadEdges);
}

void btHeightfieldTerrainShape::initialize(
	int heightStickWidth, int heightStickLength, const void* heightfieldData,
	btScalar heightScale, btScalar minHeight, btScalar maxHeight, int upAxis,
	PHY_ScalarType hdt, bool flipQuadEdges)
{
	// validation
	btAssert(heightStickWidth > 1);   // && "bad width");
	btAssert(heightStickLength > 1);  // && "bad length");
	btAssert(heightfieldData);        // && "null heightfield data");
	// btAssert(heightScale) -- do we care?  Trust caller here
	btAssert(minHeight <= maxHeight);                                    // && "bad min/max height");
	btAssert(upAxis >= 0 && upAxis < 3);                                 // && "bad upAxis--should be in range [0,2]");
	btAssert(hdt != PHY_UCHAR || hdt != PHY_FLOAT || hdt != PHY_SHORT);  // && "Bad height data type enum");

	// initialize member variables
	m_shapeType = TERRAIN_SHAPE_PROXYTYPE;
	m_heightStickWidth = heightStickWidth;
	m_heightStickLength = heightStickLength;
	m_minHeight = minHeight;
	m_maxHeight = maxHeight;
	m_width = (btScalar)(heightStickWidth - 1);
	m_length = (btScalar)(heightStickLength - 1);
	m_heightScale = heightScale;
	m_heightfieldDataUnknown = heightfieldData;
	m_heightDataType = hdt;
	m_flipQuadEdges = flipQuadEdges;
	m_useDiamondSubdivision = false;
	m_useZigzagSubdivision = false;
	m_flipTriangleWinding = false;
	m_upAxis = upAxis;
	m_localScaling.setValue(btScalar(1.), btScalar(1.), btScalar(1.));
	
	m_vboundsChunkSize = 0;
	m_vboundsGridWidth = 0;
	m_vboundsGridLength = 0;

	// determine min/max axis-aligned bounding box (aabb) values
	switch (m_upAxis)
	{
		case 0:
		{
			m_localAabbMin.setValue(m_minHeight, 0, 0);
			m_localAabbMax.setValue(m_maxHeight, m_width, m_length);
			break;
		}
		case 1:
		{
			m_localAabbMin.setValue(0, m_minHeight, 0);
			m_localAabbMax.setValue(m_width, m_maxHeight, m_length);
			break;
		};
		case 2:
		{
			m_localAabbMin.setValue(0, 0, m_minHeight);
			m_localAabbMax.setValue(m_width, m_length, m_maxHeight);
			break;
		}
		default:
		{
			//need to get valid m_upAxis
			btAssert(0);  // && "Bad m_upAxis");
		}
	}

	// remember origin (defined as exact middle of aabb)
	m_localOrigin = btScalar(0.5) * (m_localAabbMin + m_localAabbMax);
}

btHeightfieldTerrainShape::~btHeightfieldTerrainShape()
{
	clearAccelerator();
}

void btHeightfieldTerrainShape::getAabb(const btTransform& t, btVector3& aabbMin, btVector3& aabbMax) const
{
	btVector3 halfExtents = (m_localAabbMax - m_localAabbMin) * m_localScaling * btScalar(0.5);

	btVector3 localOrigin(0, 0, 0);
	localOrigin[m_upAxis] = (m_minHeight + m_maxHeight) * btScalar(0.5);
	localOrigin *= m_localScaling;

	btMatrix3x3 abs_b = t.getBasis().absolute();
	btVector3 center = t.getOrigin();
	btVector3 extent = halfExtents.dot3(abs_b[0], abs_b[1], abs_b[2]);
	extent += btVector3(getMargin(), getMargin(), getMargin());

	aabbMin = center - extent;
	aabbMax = center + extent;
}

/// This returns the "raw" (user's initial) height, not the actual height.
/// The actual height needs to be adjusted to be relative to the center
///   of the heightfield's AABB.
btScalar
btHeightfieldTerrainShape::getRawHeightFieldValue(int x, int y) const
{
	btScalar val = 0.f;
	switch (m_heightDataType)
	{
		case PHY_FLOAT:
		{
			val = m_heightfieldDataFloat[(y * m_heightStickWidth) + x];
			break;
		}

		case PHY_UCHAR:
		{
			unsigned char heightFieldValue = m_heightfieldDataUnsignedChar[(y * m_heightStickWidth) + x];
			val = heightFieldValue * m_heightScale;
			break;
		}

		case PHY_SHORT:
		{
			short hfValue = m_heightfieldDataShort[(y * m_heightStickWidth) + x];
			val = hfValue * m_heightScale;
			break;
		}

		default:
		{
			btAssert(!"Bad m_heightDataType");
		}
	}

	return val;
}

/// this returns the vertex in bullet-local coordinates
void btHeightfieldTerrainShape::getVertex(int x, int y, btVector3& vertex) const
{
	btAssert(x >= 0);
	btAssert(y >= 0);
	btAssert(x < m_heightStickWidth);
	btAssert(y < m_heightStickLength);

	btScalar height = getRawHeightFieldValue(x, y);

	switch (m_upAxis)
	{
		case 0:
		{
			vertex.setValue(
				height - m_localOrigin.getX(),
				(-m_width / btScalar(2.0)) + x,
				(-m_length / btScalar(2.0)) + y);
			break;
		}
		case 1:
		{
			vertex.setValue(
				(-m_width / btScalar(2.0)) + x,
				height - m_localOrigin.getY(),
				(-m_length / btScalar(2.0)) + y);
			break;
		};
		case 2:
		{
			vertex.setValue(
				(-m_width / btScalar(2.0)) + x,
				(-m_length / btScalar(2.0)) + y,
				height - m_localOrigin.getZ());
			break;
		}
		default:
		{
			//need to get valid m_upAxis
			btAssert(0);
		}
	}

	vertex *= m_localScaling;
}

static inline int
getQuantized(
	btScalar x)
{
	if (x < 0.0)
	{
		return (int)(x - 0.5);
	}
	return (int)(x + 0.5);
}

/// given input vector, return quantized version
/**
  This routine is basically determining the gridpoint indices for a given
  input vector, answering the question: "which gridpoint is closest to the
  provided point?".

  "with clamp" means that we restrict the point to be in the heightfield's
  axis-aligned bounding box.
 */
void btHeightfieldTerrainShape::quantizeWithClamp(int* out, const btVector3& point, int /*isMax*/) const
{
	btVector3 clampedPoint(point);
	clampedPoint.setMax(m_localAabbMin);
	clampedPoint.setMin(m_localAabbMax);

	out[0] = getQuantized(clampedPoint.getX());
	out[1] = getQuantized(clampedPoint.getY());
	out[2] = getQuantized(clampedPoint.getZ());
}

/// process all triangles within the provided axis-aligned bounding box
/**
  basic algorithm:
    - convert input aabb to local coordinates (scale down and shift for local origin)
    - convert input aabb to a range of heightfield grid points (quantize)
    - iterate over all triangles in that subset of the grid
 */
void btHeightfieldTerrainShape::processAllTriangles(btTriangleCallback* callback, const btVector3& aabbMin, const btVector3& aabbMax) const
{
	// scale down the input aabb's so they are in local (non-scaled) coordinates
	btVector3 localAabbMin = aabbMin * btVector3(1.f / m_localScaling[0], 1.f / m_localScaling[1], 1.f / m_localScaling[2]);
	btVector3 localAabbMax = aabbMax * btVector3(1.f / m_localScaling[0], 1.f / m_localScaling[1], 1.f / m_localScaling[2]);

	// account for local origin
	localAabbMin += m_localOrigin;
	localAabbMax += m_localOrigin;

	//quantize the aabbMin and aabbMax, and adjust the start/end ranges
	int quantizedAabbMin[3];
	int quantizedAabbMax[3];
	quantizeWithClamp(quantizedAabbMin, localAabbMin, 0);
	quantizeWithClamp(quantizedAabbMax, localAabbMax, 1);

	// expand the min/max quantized values
	// this is to catch the case where the input aabb falls between grid points!
	for (int i = 0; i < 3; ++i)
	{
		quantizedAabbMin[i]--;
		quantizedAabbMax[i]++;
	}

	int startX = 0;
	int endX = m_heightStickWidth - 1;
	int startJ = 0;
	int endJ = m_heightStickLength - 1;

	switch (m_upAxis)
	{
		case 0:
		{
			if (quantizedAabbMin[1] > startX)
				startX = quantizedAabbMin[1];
			if (quantizedAabbMax[1] < endX)
				endX = quantizedAabbMax[1];
			if (quantizedAabbMin[2] > startJ)
				startJ = quantizedAabbMin[2];
			if (quantizedAabbMax[2] < endJ)
				endJ = quantizedAabbMax[2];
			break;
		}
		case 1:
		{
			if (quantizedAabbMin[0] > startX)
				startX = quantizedAabbMin[0];
			if (quantizedAabbMax[0] < endX)
				endX = quantizedAabbMax[0];
			if (quantizedAabbMin[2] > startJ)
				startJ = quantizedAabbMin[2];
			if (quantizedAabbMax[2] < endJ)
				endJ = quantizedAabbMax[2];
			break;
		};
		case 2:
		{
			if (quantizedAabbMin[0] > startX)
				startX = quantizedAabbMin[0];
			if (quantizedAabbMax[0] < endX)
				endX = quantizedAabbMax[0];
			if (quantizedAabbMin[1] > startJ)
				startJ = quantizedAabbMin[1];
			if (quantizedAabbMax[1] < endJ)
				endJ = quantizedAabbMax[1];
			break;
		}
		default:
		{
			//need to get valid m_upAxis
			btAssert(0);
		}
	}

	// TODO If m_vboundsGrid is available, use it to determine if we really need to process this area

	for (int j = startJ; j < endJ; j++)
	{
		for (int x = startX; x < endX; x++)
		{
			btVector3 vertices[3];
			int indices[3] = { 0, 1, 2 };
			if (m_flipTriangleWinding)
			{
				indices[0] = 2;
				indices[2] = 0;
			}

			if (m_flipQuadEdges || (m_useDiamondSubdivision && !((j + x) & 1)) || (m_useZigzagSubdivision && !(j & 1)))
			{
				//first triangle
				getVertex(x, j, vertices[indices[0]]);
				getVertex(x, j + 1, vertices[indices[1]]);
				getVertex(x + 1, j + 1, vertices[indices[2]]);
				callback->processTriangle(vertices, 2 * x, j);
				//second triangle
				//  getVertex(x,j,vertices[0]);//already got this vertex before, thanks to Danny Chapman
				getVertex(x + 1, j + 1, vertices[indices[1]]);
				getVertex(x + 1, j, vertices[indices[2]]);
				callback->processTriangle(vertices, 2 * x+1, j);
			}
			else
			{
				//first triangle
				getVertex(x, j, vertices[indices[0]]);
				getVertex(x, j + 1, vertices[indices[1]]);
				getVertex(x + 1, j, vertices[indices[2]]);
				callback->processTriangle(vertices, 2 * x, j);
				//second triangle
				getVertex(x + 1, j, vertices[indices[0]]);
				//getVertex(x,j+1,vertices[1]);
				getVertex(x + 1, j + 1, vertices[indices[2]]);
				callback->processTriangle(vertices, 2 * x+1, j);
			}
		}
	}
}

void btHeightfieldTerrainShape::calculateLocalInertia(btScalar, btVector3& inertia) const
{
	//moving concave objects not supported

	inertia.setValue(btScalar(0.), btScalar(0.), btScalar(0.));
}

void btHeightfieldTerrainShape::setLocalScaling(const btVector3& scaling)
{
	m_localScaling = scaling;
}
const btVector3& btHeightfieldTerrainShape::getLocalScaling() const
{
	return m_localScaling;
}

namespace
{
	struct GridRaycastState
	{
		int x;  // Next quad coords
		int z;
		int prev_x;  // Previous quad coords
		int prev_z;
		btScalar param;      // Exit param for previous quad
		btScalar prevParam;  // Enter param for previous quad
		btScalar maxDistanceFlat;
		btScalar maxDistance3d;
	};
}

// TODO Does it really need to take 3D vectors?
/// Iterates through a virtual 2D grid of unit-sized square cells,
/// and executes an action on each cell intersecting the given segment, ordered from begin to end.
/// Initially inspired by http://www.cse.yorku.ca/~amana/research/grid.pdf
template <typename Action_T>
void gridRaycast(Action_T& quadAction, const btVector3& beginPos, const btVector3& endPos, int indices[3])
{
	GridRaycastState rs;
	rs.maxDistance3d = beginPos.distance(endPos);
	if (rs.maxDistance3d < 0.0001)
	{
		// Consider the ray is too small to hit anything
		return;
	}
	

	btScalar rayDirectionFlatX = endPos[indices[0]] - beginPos[indices[0]];
	btScalar rayDirectionFlatZ = endPos[indices[2]] - beginPos[indices[2]];
	rs.maxDistanceFlat = btSqrt(rayDirectionFlatX * rayDirectionFlatX + rayDirectionFlatZ * rayDirectionFlatZ);

	if (rs.maxDistanceFlat < 0.0001)
	{
		// Consider the ray vertical
		rayDirectionFlatX = 0;
		rayDirectionFlatZ = 0;
	}
	else
	{
		rayDirectionFlatX /= rs.maxDistanceFlat;
		rayDirectionFlatZ /= rs.maxDistanceFlat;
	}

	const int xiStep = rayDirectionFlatX > 0 ? 1 : rayDirectionFlatX < 0 ? -1 : 0;
	const int ziStep = rayDirectionFlatZ > 0 ? 1 : rayDirectionFlatZ < 0 ? -1 : 0;

	const float infinite = 9999999;
	const btScalar paramDeltaX = xiStep != 0 ? 1.f / btFabs(rayDirectionFlatX) : infinite;
	const btScalar paramDeltaZ = ziStep != 0 ? 1.f / btFabs(rayDirectionFlatZ) : infinite;

	// pos = param * dir
	btScalar paramCrossX;  // At which value of `param` we will cross a x-axis lane?
	btScalar paramCrossZ;  // At which value of `param` we will cross a z-axis lane?

	// paramCrossX and paramCrossZ are initialized as being the first cross
	// X initialization
	if (xiStep != 0)
	{
		if (xiStep == 1)
		{
			paramCrossX = (ceil(beginPos[indices[0]]) - beginPos[indices[0]]) * paramDeltaX;
		}
		else
		{
			paramCrossX = (beginPos[indices[0]] - floor(beginPos[indices[0]])) * paramDeltaX;
		}
	}
	else
	{
		paramCrossX = infinite;  // Will never cross on X
	}

	// Z initialization
	if (ziStep != 0)
	{
		if (ziStep == 1)
		{
			paramCrossZ = (ceil(beginPos[indices[2]]) - beginPos[indices[2]]) * paramDeltaZ;
		}
		else
		{
			paramCrossZ = (beginPos[indices[2]] - floor(beginPos[indices[2]])) * paramDeltaZ;
		}
	}
	else
	{
		paramCrossZ = infinite;  // Will never cross on Z
	}

	rs.x = static_cast<int>(floor(beginPos[indices[0]]));
	rs.z = static_cast<int>(floor(beginPos[indices[2]]));

	// Workaround cases where the ray starts at an integer position
	if (paramCrossX == 0.0)
	{
		paramCrossX += paramDeltaX;
		// If going backwards, we should ignore the position we would get by the above flooring,
		// because the ray is not heading in that direction
		if (xiStep == -1)
		{
			rs.x -= 1;
		}
	}

	if (paramCrossZ == 0.0)
	{
		paramCrossZ += paramDeltaZ;
		if (ziStep == -1)
			rs.z -= 1;
	}

	rs.prev_x = rs.x;
	rs.prev_z = rs.z;
	rs.param = 0;

	while (true)
	{
		rs.prev_x = rs.x;
		rs.prev_z = rs.z;
		rs.prevParam = rs.param;

		if (paramCrossX < paramCrossZ)
		{
			// X lane
			rs.x += xiStep;
			// Assign before advancing the param,
			// to be in sync with the initialization step
			rs.param = paramCrossX;
			paramCrossX += paramDeltaX;
		}
		else
		{
			// Z lane
			rs.z += ziStep;
			rs.param = paramCrossZ;
			paramCrossZ += paramDeltaZ;
		}

		if (rs.param > rs.maxDistanceFlat)
		{
			rs.param = rs.maxDistanceFlat;
			quadAction(rs);
			break;
		}
		else
		{
			quadAction(rs);
		}
	}
}

struct ProcessTrianglesAction
{
	const btHeightfieldTerrainShape* shape;
	bool flipQuadEdges;
	bool useDiamondSubdivision;
	int width;
	int length;
	btTriangleCallback* callback;

	void exec(int x, int z) const
	{
		if (x < 0 || z < 0 || x >= width || z >= length)
		{
			return;
		}

		btVector3 vertices[3];

		// TODO Since this is for raycasts, we could greatly benefit from an early exit on the first hit

		// Check quad
		if (flipQuadEdges || (useDiamondSubdivision && (((z + x) & 1) > 0)))
		{
			// First triangle
			shape->getVertex(x, z, vertices[0]);
			shape->getVertex(x + 1, z, vertices[1]);
			shape->getVertex(x + 1, z + 1, vertices[2]);
			callback->processTriangle(vertices, x, z);

			// Second triangle
			shape->getVertex(x, z, vertices[0]);
			shape->getVertex(x + 1, z + 1, vertices[1]);
			shape->getVertex(x, z + 1, vertices[2]);
			callback->processTriangle(vertices, x, z);
		}
		else
		{
			// First triangle
			shape->getVertex(x, z, vertices[0]);
			shape->getVertex(x, z + 1, vertices[1]);
			shape->getVertex(x + 1, z, vertices[2]);
			callback->processTriangle(vertices, x, z);

			// Second triangle
			shape->getVertex(x + 1, z, vertices[0]);
			shape->getVertex(x, z + 1, vertices[1]);
			shape->getVertex(x + 1, z + 1, vertices[2]);
			callback->processTriangle(vertices, x, z);
		}
	}

	void operator()(const GridRaycastState& bs) const
	{
		exec(bs.prev_x, bs.prev_z);
	}
};

struct ProcessVBoundsAction
{
	const btAlignedObjectArray<btHeightfieldTerrainShape::Range>& vbounds;
	int width;
	int length;
	int chunkSize;

	btVector3 rayBegin;
	btVector3 rayEnd;
	btVector3 rayDir;

	int* m_indices;
	ProcessTrianglesAction processTriangles;

	ProcessVBoundsAction(const btAlignedObjectArray<btHeightfieldTerrainShape::Range>& bnd, int* indices)
		: vbounds(bnd),
		m_indices(indices)
	{
	}
	void operator()(const GridRaycastState& rs) const
	{
		int x = rs.prev_x;
		int z = rs.prev_z;

		if (x < 0 || z < 0 || x >= width || z >= length)
		{
			return;
		}

		const btHeightfieldTerrainShape::Range chunk = vbounds[x + z * width];

		btVector3 enterPos;
		btVector3 exitPos;

		if (rs.maxDistanceFlat > 0.0001)
		{
			btScalar flatTo3d = chunkSize * rs.maxDistance3d / rs.maxDistanceFlat;
			btScalar enterParam3d = rs.prevParam * flatTo3d;
			btScalar exitParam3d = rs.param * flatTo3d;
			enterPos = rayBegin + rayDir * enterParam3d;
			exitPos = rayBegin + rayDir * exitParam3d;

			// We did enter the flat projection of the AABB,
			// but we have to check if we intersect it on the vertical axis
			if (enterPos[1] > chunk.max && exitPos[m_indices[1]] > chunk.max)
			{
				return;
			}
			if (enterPos[1] < chunk.min && exitPos[m_indices[1]] < chunk.min)
			{
				return;
			}
		}
		else
		{
			// Consider the ray vertical
			// (though we shouldn't reach this often because there is an early check up-front)
			enterPos = rayBegin;
			exitPos = rayEnd;
		}

		gridRaycast(processTriangles, enterPos, exitPos, m_indices);
		// Note: it could be possible to have more than one grid at different levels,
		// to do this there would be a branch using a pointer to another ProcessVBoundsAction
	}
};

// TODO How do I interrupt the ray when there is a hit? `callback` does not return any result
/// Performs a raycast using a hierarchical Bresenham algorithm.
/// Does not allocate any memory by itself.
void btHeightfieldTerrainShape::performRaycast(btTriangleCallback* callback, const btVector3& raySource, const btVector3& rayTarget) const
{
	// Transform to cell-local
	btVector3 beginPos = raySource / m_localScaling;
	btVector3 endPos = rayTarget / m_localScaling;
	beginPos += m_localOrigin;
	endPos += m_localOrigin;

	ProcessTrianglesAction processTriangles;
	processTriangles.shape = this;
	processTriangles.flipQuadEdges = m_flipQuadEdges;
	processTriangles.useDiamondSubdivision = m_useDiamondSubdivision;
	processTriangles.callback = callback;
	processTriangles.width = m_heightStickWidth - 1;
	processTriangles.length = m_heightStickLength - 1;

	// TODO Transform vectors to account for m_upAxis
	int indices[3] = { 0, 1, 2 };
	if (m_upAxis == 2)
	{
		indices[1] = 2;
		indices[2] = 1;
	}
	int iBeginX = static_cast<int>(floor(beginPos[indices[0]]));
	int iBeginZ = static_cast<int>(floor(beginPos[indices[2]]));
	int iEndX = static_cast<int>(floor(endPos[indices[0]]));
	int iEndZ = static_cast<int>(floor(endPos[indices[2]]));

	if (iBeginX == iEndX && iBeginZ == iEndZ)
	{
		// The ray will never cross quads within the plane,
		// so directly process triangles within one quad
		// (typically, vertical rays should end up here)
		processTriangles.exec(iBeginX, iEndZ);
		return;
	}

	

	if (m_vboundsGrid.size()==0)
	{
		// Process all quads intersecting the flat projection of the ray
		gridRaycast(processTriangles, beginPos, endPos, &indices[0]);
	}
	else
	{
		btVector3 rayDiff = endPos - beginPos;
		btScalar flatDistance2 = rayDiff[indices[0]] * rayDiff[indices[0]] + rayDiff[indices[2]] * rayDiff[indices[2]];
		if (flatDistance2 < m_vboundsChunkSize * m_vboundsChunkSize)
		{
			// Don't use chunks, the ray is too short in the plane
			gridRaycast(processTriangles, beginPos, endPos, &indices[0]);
		}

		ProcessVBoundsAction processVBounds(m_vboundsGrid, &indices[0]);
		processVBounds.width = m_vboundsGridWidth;
		processVBounds.length = m_vboundsGridLength;
		processVBounds.rayBegin = beginPos;
		processVBounds.rayEnd = endPos;
		processVBounds.rayDir = rayDiff.normalized();
		processVBounds.processTriangles = processTriangles;
		processVBounds.chunkSize = m_vboundsChunkSize;
		// The ray is long, run raycast on a higher-level grid
		gridRaycast(processVBounds, beginPos / m_vboundsChunkSize, endPos / m_vboundsChunkSize, indices);
	}
}

/// Builds a grid data structure storing the min and max heights of the terrain in chunks.
/// if chunkSize is zero, that accelerator is removed.
/// If you modify the heights, you need to rebuild this accelerator.
void btHeightfieldTerrainShape::buildAccelerator(int chunkSize)
{
	if (chunkSize <= 0)
	{
		clearAccelerator();
		return;
	}

	m_vboundsChunkSize = chunkSize;
	int nChunksX = m_heightStickWidth / chunkSize;
	int nChunksZ = m_heightStickLength / chunkSize;

	if (m_heightStickWidth % chunkSize > 0)
	{
		++nChunksX;  // In case terrain size isn't dividable by chunk size
	}
	if (m_heightStickLength % chunkSize > 0)
	{
		++nChunksZ;
	}

	if (m_vboundsGridWidth != nChunksX || m_vboundsGridLength != nChunksZ)
	{
		clearAccelerator();
		m_vboundsGridWidth = nChunksX;
		m_vboundsGridLength = nChunksZ;
	}

	if (nChunksX == 0 || nChunksZ == 0)
	{
		return;
	}

	// This data structure is only reallocated if the required size changed
	m_vboundsGrid.resize(nChunksX * nChunksZ);
	
	// Compute min and max height for all chunks
	for (int cz = 0; cz < nChunksZ; ++cz)
	{
		int z0 = cz * chunkSize;

		for (int cx = 0; cx < nChunksX; ++cx)
		{
			int x0 = cx * chunkSize;

			Range r;

			r.min = getRawHeightFieldValue(x0, z0);
			r.max = r.min;

			// Compute min and max height for this chunk.
			// We have to include one extra cell to account for neighbors.
			// Here is why:
			// Say we have a flat terrain, and a plateau that fits a chunk perfectly.
			//
			//   Left        Right
			// 0---0---0---1---1---1
			// |   |   |   |   |   |
			// 0---0---0---1---1---1
			// |   |   |   |   |   |
			// 0---0---0---1---1---1
			//           x
			//
			// If the AABB for the Left chunk did not share vertices with the Right,
			// then we would fail collision tests at x due to a gap.
			//
			for (int z = z0; z < z0 + chunkSize + 1; ++z)
			{
				if (z >= m_heightStickLength)
				{
					continue;
				}

				for (int x = x0; x < x0 + chunkSize + 1; ++x)
				{
					if (x >= m_heightStickWidth)
					{
						continue;
					}

					btScalar height = getRawHeightFieldValue(x, z);

					if (height < r.min)
					{
						r.min = height;
					}
					else if (height > r.max)
					{
						r.max = height;
					}
				}
			}

			m_vboundsGrid[cx + cz * nChunksX] = r;
		}
	}
}

void btHeightfieldTerrainShape::clearAccelerator()
{
	m_vboundsGrid.clear();
}