/*
Copyright (c) 2011 Ole Kniemeyer, MAXON, www.maxon.net

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#ifndef BT_CONVEX_HULL_COMPUTER_H
#define BT_CONVEX_HULL_COMPUTER_H

#include "btAlignedObjectArray.h"
#include "btVector3.h"

namespace VHACD {

/// Convex hull implementation based on Preparata and Hong
/// See http://code.google.com/p/bullet/issues/detail?id=275
/// Ole Kniemeyer, MAXON Computer GmbH
class btConvexHullComputer {
private:
    btScalar compute(const void* coords, bool doubleCoords, int32_t stride, int32_t count, btScalar shrink, btScalar shrinkClamp);

public:
    class Edge {
    private:
        int32_t next;
        int32_t reverse;
        int32_t targetVertex;

        friend class btConvexHullComputer;

    public:
        int32_t getSourceVertex() const
        {
            return (this + reverse)->targetVertex;
        }

        int32_t getTargetVertex() const
        {
            return targetVertex;
        }

        const Edge* getNextEdgeOfVertex() const // clockwise list of all edges of a vertex
        {
            return this + next;
        }

        const Edge* getNextEdgeOfFace() const // counter-clockwise list of all edges of a face
        {
            return (this + reverse)->getNextEdgeOfVertex();
        }

        const Edge* getReverseEdge() const
        {
            return this + reverse;
        }
    };

    // Vertices of the output hull
    btAlignedObjectArray<btVector3> vertices;

    // Edges of the output hull
    btAlignedObjectArray<Edge> edges;

    // Faces of the convex hull. Each entry is an index into the "edges" array pointing to an edge of the face. Faces are planar n-gons
    btAlignedObjectArray<int32_t> faces;

    /*
		Compute convex hull of "count" vertices stored in "coords". "stride" is the difference in bytes
		between the addresses of consecutive vertices. If "shrink" is positive, the convex hull is shrunken
		by that amount (each face is moved by "shrink" length units towards the center along its normal).
		If "shrinkClamp" is positive, "shrink" is clamped to not exceed "shrinkClamp * innerRadius", where "innerRadius"
		is the minimum distance of a face to the center of the convex hull.

		The returned value is the amount by which the hull has been shrunken. If it is negative, the amount was so large
		that the resulting convex hull is empty.

		The output convex hull can be found in the member variables "vertices", "edges", "faces".
		*/
    btScalar compute(const float* coords, int32_t stride, int32_t count, btScalar shrink, btScalar shrinkClamp)
    {
        return compute(coords, false, stride, count, shrink, shrinkClamp);
    }

    // same as above, but double precision
    btScalar compute(const double* coords, int32_t stride, int32_t count, btScalar shrink, btScalar shrinkClamp)
    {
        return compute(coords, true, stride, count, shrink, shrinkClamp);
    }
};

}; // namespace VHACD

#endif //BT_CONVEX_HULL_COMPUTER_H
