#include "vhacdRaycastMesh.h"
#include <math.h>
#include <assert.h>

namespace RAYCAST_MESH
{

/* a = b - c */
#define vector(a,b,c) \
	(a)[0] = (b)[0] - (c)[0];	\
	(a)[1] = (b)[1] - (c)[1];	\
	(a)[2] = (b)[2] - (c)[2];

#define innerProduct(v,q) \
	((v)[0] * (q)[0] + \
	(v)[1] * (q)[1] + \
	(v)[2] * (q)[2])

#define crossProduct(a,b,c) \
	(a)[0] = (b)[1] * (c)[2] - (c)[1] * (b)[2]; \
	(a)[1] = (b)[2] * (c)[0] - (c)[2] * (b)[0]; \
	(a)[2] = (b)[0] * (c)[1] - (c)[0] * (b)[1];


static inline bool rayIntersectsTriangle(const double *p,const double *d,const double *v0,const double *v1,const double *v2,double &t)
{
	double e1[3],e2[3],h[3],s[3],q[3];
	double a,f,u,v;

	vector(e1,v1,v0);
	vector(e2,v2,v0);
	crossProduct(h,d,e2);
	a = innerProduct(e1,h);

	if (a > -0.00001 && a < 0.00001)
		return(false);

	f = 1/a;
	vector(s,p,v0);
	u = f * (innerProduct(s,h));

	if (u < 0.0 || u > 1.0)
		return(false);

	crossProduct(q,s,e1);
	v = f * innerProduct(d,q);
	if (v < 0.0 || u + v > 1.0)
		return(false);
	// at this stage we can compute t to find out where
	// the intersection point is on the line
	t = f * innerProduct(e2,q);
	if (t > 0) // ray intersection
		return(true);
	else // this means that there is a line intersection
		// but not a ray intersection
		return (false);
}

static double getPointDistance(const double *p1, const double *p2)
{
	double dx = p1[0] - p2[0];
	double dy = p1[1] - p2[1];
	double dz = p1[2] - p2[2];
	return sqrt(dx*dx + dy*dy + dz*dz);
}

class MyRaycastMesh : public VHACD::RaycastMesh
{
public:

    template <class T>
	MyRaycastMesh(uint32_t vcount,
                  const T *vertices,
                  uint32_t tcount,
                  const uint32_t *indices)
	{
        mVcount = vcount;
        mVertices = new double[mVcount * 3];
        for (uint32_t i = 0; i < mVcount; i++)
        {
            mVertices[i * 3 + 0] = vertices[0];
            mVertices[i * 3 + 1] = vertices[1];
            mVertices[i * 3 + 2] = vertices[2];
            vertices += 3;
        }
        mTcount = tcount;
        mIndices = new uint32_t[mTcount * 3];
        for (uint32_t i = 0; i < mTcount; i++)
        {
            mIndices[i * 3 + 0] = indices[0];
            mIndices[i * 3 + 1] = indices[1];
            mIndices[i * 3 + 2] = indices[2];
            indices += 3;
        }
	}


	~MyRaycastMesh(void)
	{
        delete[]mVertices;
        delete[]mIndices;
	}

	virtual void release(void)
	{
		delete this;
	}

	virtual bool raycast(const double *from,			// The starting point of the raycast
		const double *to,				// The ending point of the raycast
		const double *closestToPoint,	// The point to match the nearest hit location (can just be the 'from' location of no specific point)
		double *hitLocation,			// The point where the ray hit nearest to the 'closestToPoint' location
		double *hitDistance) final		// The distance the ray traveled to the hit location
	{
		bool ret = false;

		double dir[3];

		dir[0] = to[0] - from[0];
		dir[1] = to[1] - from[1];
		dir[2] = to[2] - from[2];

		double distance = sqrt( dir[0]*dir[0] + dir[1]*dir[1]+dir[2]*dir[2] );
		if ( distance < 0.0000000001f ) return false;
		double recipDistance = 1.0f / distance;
		dir[0]*=recipDistance;
		dir[1]*=recipDistance;
		dir[2]*=recipDistance;
		const uint32_t *indices = mIndices;
		const double *vertices = mVertices;
		double nearestDistance = distance;

		for (uint32_t tri=0; tri<mTcount; tri++)
		{
			uint32_t i1 = indices[tri*3+0];
			uint32_t i2 = indices[tri*3+1];
			uint32_t i3 = indices[tri*3+2];

			const double *p1 = &vertices[i1*3];
			const double *p2 = &vertices[i2*3];
			const double *p3 = &vertices[i3*3];

			double t;
			if ( rayIntersectsTriangle(from,dir,p1,p2,p3,t))
			{
				double hitPos[3];

				hitPos[0] = from[0] + dir[0] * t;
				hitPos[1] = from[1] + dir[1] * t;
				hitPos[2] = from[2] + dir[2] * t;

				double pointDistance = getPointDistance(hitPos, closestToPoint);

				if (pointDistance < nearestDistance )
				{
					nearestDistance = pointDistance;
					if ( hitLocation )
					{
						hitLocation[0] = hitPos[0];
						hitLocation[1] = hitPos[1];
						hitLocation[2] = hitPos[2];
					}
					if ( hitDistance )
					{
						*hitDistance = pointDistance;
					}
					ret = true;
				}
			}
		}
		return ret;
	}

	uint32_t		mVcount;
	double	        *mVertices;
	uint32_t		mTcount;
	uint32_t	    *mIndices;
};

};



using namespace RAYCAST_MESH;

namespace VHACD
{

    RaycastMesh * RaycastMesh::createRaycastMesh(uint32_t vcount,		// The number of vertices in the source triangle mesh
        const double *vertices,		// The array of vertex positions in the format x1,y1,z1..x2,y2,z2.. etc.
        uint32_t tcount,		// The number of triangles in the source triangle mesh
        const uint32_t *indices) // The triangle indices in the format of i1,i2,i3 ... i4,i5,i6, ...
    {
        MyRaycastMesh *m = new MyRaycastMesh(vcount, vertices, tcount, indices);
        return static_cast<RaycastMesh *>(m);
    }

    RaycastMesh * RaycastMesh::createRaycastMesh(uint32_t vcount,		// The number of vertices in the source triangle mesh
        const float *vertices,		// The array of vertex positions in the format x1,y1,z1..x2,y2,z2.. etc.
        uint32_t tcount,		// The number of triangles in the source triangle mesh
        const uint32_t *indices) // The triangle indices in the format of i1,i2,i3 ... i4,i5,i6, ...
    {
        MyRaycastMesh *m = new MyRaycastMesh(vcount, vertices, tcount, indices);
        return static_cast<RaycastMesh *>(m);
    }


} // end of VHACD namespace