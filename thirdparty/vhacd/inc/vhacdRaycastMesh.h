#ifndef RAYCAST_MESH_H

#define RAYCAST_MESH_H

#include <stdint.h>

namespace VHACD
{

    // Very simple brute force raycast against a triangle mesh.  Tests every triangle; no hierachy.
    // Does a deep copy, always does calculations with full double float precision
    class RaycastMesh
    {
    public:
        static RaycastMesh * createRaycastMesh(uint32_t vcount,		// The number of vertices in the source triangle mesh
            const double *vertices,		// The array of vertex positions in the format x1,y1,z1..x2,y2,z2.. etc.
            uint32_t tcount,		// The number of triangles in the source triangle mesh
            const uint32_t *indices); // The triangle indices in the format of i1,i2,i3 ... i4,i5,i6, ...

        static RaycastMesh * createRaycastMesh(uint32_t vcount,		// The number of vertices in the source triangle mesh
            const float *vertices,		// The array of vertex positions in the format x1,y1,z1..x2,y2,z2.. etc.
            uint32_t tcount,		// The number of triangles in the source triangle mesh
            const uint32_t *indices); // The triangle indices in the format of i1,i2,i3 ... i4,i5,i6, ...


        virtual bool raycast(const double *from,			// The starting point of the raycast
            const double *to,				// The ending point of the raycast
            const double *closestToPoint,	// The point to match the nearest hit location (can just be the 'from' location of no specific point)
            double *hitLocation,			// The point where the ray hit nearest to the 'closestToPoint' location
            double *hitDistance) = 0;		// The distance the ray traveled to the hit location

        virtual void release(void) = 0;
    protected:
        virtual ~RaycastMesh(void) { };
    };

} // end of VHACD namespace

#endif
