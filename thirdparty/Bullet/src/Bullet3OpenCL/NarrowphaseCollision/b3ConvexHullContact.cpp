/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2011 Advanced Micro Devices, Inc.  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

bool findSeparatingAxisOnGpu = true;
bool splitSearchSepAxisConcave = false;
bool splitSearchSepAxisConvex = true;
bool useMprGpu = true;//use mpr for edge-edge  (+contact point) or sat. Needs testing on main OpenCL platforms, before enabling...
bool bvhTraversalKernelGPU = true;
bool findConcaveSeparatingAxisKernelGPU = true;
bool clipConcaveFacesAndFindContactsCPU = false;//false;//true;
bool clipConvexFacesAndFindContactsCPU = false;//false;//true;
bool reduceConcaveContactsOnGPU = true;//false;
bool reduceConvexContactsOnGPU = true;//false;
bool findConvexClippingFacesGPU = true;
bool useGjk = true;///option for CPU/host testing, when findSeparatingAxisOnGpu = false
bool useGjkContacts = true;//////option for CPU/host testing when findSeparatingAxisOnGpu = false


static int myframecount=0;///for testing

///This file was written by Erwin Coumans
///Separating axis rest based on work from Pierre Terdiman, see
///And contact clipping based on work from Simon Hobbs

//#define B3_DEBUG_SAT_FACE

//#define CHECK_ON_HOST

#ifdef CHECK_ON_HOST
//#define PERSISTENT_CONTACTS_HOST
#endif

int b3g_actualSATPairTests=0;

#include "b3ConvexHullContact.h"
#include <string.h>//memcpy
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3ConvexPolyhedronData.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3MprPenetration.h"

#include "Bullet3OpenCL/NarrowphaseCollision/b3ContactCache.h"
#include "Bullet3Geometry/b3AabbUtil.h"

typedef b3AlignedObjectArray<b3Vector3> b3VertexArray;


#include <float.h> //for FLT_MAX
#include "Bullet3OpenCL/Initialize/b3OpenCLUtils.h"
#include "Bullet3OpenCL/ParallelPrimitives/b3LauncherCL.h"
//#include "AdlQuaternion.h"

#include "kernels/satKernels.h"
#include "kernels/mprKernels.h"

#include "kernels/satConcaveKernels.h"

#include "kernels/satClipHullContacts.h"
#include "kernels/bvhTraversal.h"
#include "kernels/primitiveContacts.h"


#include "Bullet3Geometry/b3AabbUtil.h"

#define BT_NARROWPHASE_SAT_PATH "src/Bullet3OpenCL/NarrowphaseCollision/kernels/sat.cl"
#define BT_NARROWPHASE_SAT_CONCAVE_PATH "src/Bullet3OpenCL/NarrowphaseCollision/kernels/satConcave.cl"

#define BT_NARROWPHASE_MPR_PATH "src/Bullet3OpenCL/NarrowphaseCollision/kernels/mpr.cl"


#define BT_NARROWPHASE_CLIPHULL_PATH "src/Bullet3OpenCL/NarrowphaseCollision/kernels/satClipHullContacts.cl"
#define BT_NARROWPHASE_BVH_TRAVERSAL_PATH "src/Bullet3OpenCL/NarrowphaseCollision/kernels/bvhTraversal.cl"
#define BT_NARROWPHASE_PRIMITIVE_CONTACT_PATH "src/Bullet3OpenCL/NarrowphaseCollision/kernels/primitiveContacts.cl"


#ifndef __global
#define __global
#endif

#ifndef __kernel
#define __kernel
#endif


#include "Bullet3Collision/NarrowPhaseCollision/shared/b3BvhTraversal.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3FindConcaveSatAxis.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3ClipFaces.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3NewContactReduction.h"



#define dot3F4 b3Dot

GpuSatCollision::GpuSatCollision(cl_context ctx,cl_device_id device, cl_command_queue  q )
:m_context(ctx),
m_device(device),
m_queue(q),

m_findSeparatingAxisKernel(0),
m_findSeparatingAxisVertexFaceKernel(0),
m_findSeparatingAxisEdgeEdgeKernel(0),
m_unitSphereDirections(m_context,m_queue),

m_totalContactsOut(m_context, m_queue),
m_sepNormals(m_context, m_queue),
m_dmins(m_context,m_queue),

m_hasSeparatingNormals(m_context, m_queue),
m_concaveSepNormals(m_context, m_queue),
m_concaveHasSeparatingNormals(m_context,m_queue),
m_numConcavePairsOut(m_context, m_queue),


m_gpuCompoundPairs(m_context, m_queue),


m_gpuCompoundSepNormals(m_context, m_queue),
m_gpuHasCompoundSepNormals(m_context, m_queue),

m_numCompoundPairsOut(m_context, m_queue)
{
	m_totalContactsOut.push_back(0);
	
	cl_int errNum=0;

	if (1)
	{
		const char* mprSrc = mprKernelsCL;
		
		const char* srcConcave = satConcaveKernelsCL;
		char flags[1024]={0};
//#ifdef CL_PLATFORM_INTEL
//		sprintf(flags,"-g -s \"%s\"","C:/develop/bullet3_experiments2/opencl/gpu_narrowphase/kernels/sat.cl");
//#endif
		m_mprPenetrationKernel  = 0;
		m_findSeparatingAxisUnitSphereKernel = 0;

		if (useMprGpu)
		{
			cl_program mprProg = b3OpenCLUtils::compileCLProgramFromString(m_context,m_device,mprSrc,&errNum,flags,BT_NARROWPHASE_MPR_PATH);
			b3Assert(errNum==CL_SUCCESS);
		
			m_mprPenetrationKernel  = b3OpenCLUtils::compileCLKernelFromString(m_context, m_device,mprSrc, "mprPenetrationKernel",&errNum,mprProg );
			b3Assert(m_mprPenetrationKernel);
			b3Assert(errNum==CL_SUCCESS);

			m_findSeparatingAxisUnitSphereKernel =  b3OpenCLUtils::compileCLKernelFromString(m_context, m_device,mprSrc, "findSeparatingAxisUnitSphereKernel",&errNum,mprProg );
			b3Assert(m_findSeparatingAxisUnitSphereKernel);
            b3Assert(errNum==CL_SUCCESS);


			int numDirections = sizeof(unitSphere162)/sizeof(b3Vector3);
			m_unitSphereDirections.resize(numDirections);
			m_unitSphereDirections.copyFromHostPointer(unitSphere162,numDirections,0,true);


		}


		cl_program satProg = b3OpenCLUtils::compileCLProgramFromString(m_context,m_device,satKernelsCL,&errNum,flags,BT_NARROWPHASE_SAT_PATH);
		b3Assert(errNum==CL_SUCCESS);

		cl_program satConcaveProg = b3OpenCLUtils::compileCLProgramFromString(m_context,m_device,srcConcave,&errNum,flags,BT_NARROWPHASE_SAT_CONCAVE_PATH);
		b3Assert(errNum==CL_SUCCESS);

		m_findSeparatingAxisKernel = b3OpenCLUtils::compileCLKernelFromString(m_context, m_device,satKernelsCL, "findSeparatingAxisKernel",&errNum,satProg );
		b3Assert(m_findSeparatingAxisKernel);
		b3Assert(errNum==CL_SUCCESS);


		m_findSeparatingAxisVertexFaceKernel = b3OpenCLUtils::compileCLKernelFromString(m_context, m_device,satKernelsCL, "findSeparatingAxisVertexFaceKernel",&errNum,satProg );
		b3Assert(m_findSeparatingAxisVertexFaceKernel);

		m_findSeparatingAxisEdgeEdgeKernel = b3OpenCLUtils::compileCLKernelFromString(m_context, m_device,satKernelsCL, "findSeparatingAxisEdgeEdgeKernel",&errNum,satProg );
		b3Assert(m_findSeparatingAxisVertexFaceKernel);


		m_findConcaveSeparatingAxisKernel = b3OpenCLUtils::compileCLKernelFromString(m_context, m_device,satKernelsCL, "findConcaveSeparatingAxisKernel",&errNum,satProg );
		b3Assert(m_findConcaveSeparatingAxisKernel);
		b3Assert(errNum==CL_SUCCESS);
        
        m_findConcaveSeparatingAxisVertexFaceKernel = b3OpenCLUtils::compileCLKernelFromString(m_context, m_device,srcConcave, "findConcaveSeparatingAxisVertexFaceKernel",&errNum,satConcaveProg );
		b3Assert(m_findConcaveSeparatingAxisVertexFaceKernel);
		b3Assert(errNum==CL_SUCCESS);
        
        m_findConcaveSeparatingAxisEdgeEdgeKernel = b3OpenCLUtils::compileCLKernelFromString(m_context, m_device,srcConcave, "findConcaveSeparatingAxisEdgeEdgeKernel",&errNum,satConcaveProg );
		b3Assert(m_findConcaveSeparatingAxisEdgeEdgeKernel);
		b3Assert(errNum==CL_SUCCESS);
        
     
        
		
		m_findCompoundPairsKernel = b3OpenCLUtils::compileCLKernelFromString(m_context, m_device,satKernelsCL, "findCompoundPairsKernel",&errNum,satProg );
		b3Assert(m_findCompoundPairsKernel);
		b3Assert(errNum==CL_SUCCESS);
		m_processCompoundPairsKernel = b3OpenCLUtils::compileCLKernelFromString(m_context, m_device,satKernelsCL, "processCompoundPairsKernel",&errNum,satProg );
		b3Assert(m_processCompoundPairsKernel);
		b3Assert(errNum==CL_SUCCESS);
	}

	if (1)
	{
		const char* srcClip = satClipKernelsCL;

		char flags[1024]={0};
//#ifdef CL_PLATFORM_INTEL
//		sprintf(flags,"-g -s \"%s\"","C:/develop/bullet3_experiments2/opencl/gpu_narrowphase/kernels/satClipHullContacts.cl");
//#endif

		cl_program satClipContactsProg = b3OpenCLUtils::compileCLProgramFromString(m_context,m_device,srcClip,&errNum,flags,BT_NARROWPHASE_CLIPHULL_PATH);
		b3Assert(errNum==CL_SUCCESS);

		m_clipHullHullKernel = b3OpenCLUtils::compileCLKernelFromString(m_context, m_device,srcClip, "clipHullHullKernel",&errNum,satClipContactsProg);
		b3Assert(errNum==CL_SUCCESS);

		m_clipCompoundsHullHullKernel = b3OpenCLUtils::compileCLKernelFromString(m_context, m_device,srcClip, "clipCompoundsHullHullKernel",&errNum,satClipContactsProg);
		b3Assert(errNum==CL_SUCCESS);
		

        m_findClippingFacesKernel = b3OpenCLUtils::compileCLKernelFromString(m_context, m_device,srcClip, "findClippingFacesKernel",&errNum,satClipContactsProg);
		b3Assert(errNum==CL_SUCCESS);

        m_clipFacesAndFindContacts = b3OpenCLUtils::compileCLKernelFromString(m_context, m_device,srcClip, "clipFacesAndFindContactsKernel",&errNum,satClipContactsProg);
		b3Assert(errNum==CL_SUCCESS);        

		m_clipHullHullConcaveConvexKernel = b3OpenCLUtils::compileCLKernelFromString(m_context, m_device,srcClip, "clipHullHullConcaveConvexKernel",&errNum,satClipContactsProg);
		b3Assert(errNum==CL_SUCCESS);

//		m_extractManifoldAndAddContactKernel = b3OpenCLUtils::compileCLKernelFromString(m_context, m_device,srcClip, "extractManifoldAndAddContactKernel",&errNum,satClipContactsProg);
	//	b3Assert(errNum==CL_SUCCESS);

        m_newContactReductionKernel = b3OpenCLUtils::compileCLKernelFromString(m_context, m_device,srcClip,
                            "newContactReductionKernel",&errNum,satClipContactsProg);
		b3Assert(errNum==CL_SUCCESS);
	}
   else
	{
		m_clipHullHullKernel=0;
		m_clipCompoundsHullHullKernel = 0;
        m_findClippingFacesKernel = 0;
        m_newContactReductionKernel=0;
        m_clipFacesAndFindContacts = 0;
		m_clipHullHullConcaveConvexKernel = 0;
//		m_extractManifoldAndAddContactKernel = 0;
	}

	 if (1)
	{
		const char* srcBvh = bvhTraversalKernelCL;
		cl_program bvhTraversalProg = b3OpenCLUtils::compileCLProgramFromString(m_context,m_device,srcBvh,&errNum,"",BT_NARROWPHASE_BVH_TRAVERSAL_PATH);
		b3Assert(errNum==CL_SUCCESS);

		m_bvhTraversalKernel = b3OpenCLUtils::compileCLKernelFromString(m_context, m_device,srcBvh, "bvhTraversalKernel",&errNum,bvhTraversalProg,"");
		b3Assert(errNum==CL_SUCCESS);

	}
        
	 {
		 const char* primitiveContactsSrc = primitiveContactsKernelsCL;
		cl_program primitiveContactsProg = b3OpenCLUtils::compileCLProgramFromString(m_context,m_device,primitiveContactsSrc,&errNum,"",BT_NARROWPHASE_PRIMITIVE_CONTACT_PATH);
		b3Assert(errNum==CL_SUCCESS);

		m_primitiveContactsKernel = b3OpenCLUtils::compileCLKernelFromString(m_context, m_device,primitiveContactsSrc, "primitiveContactsKernel",&errNum,primitiveContactsProg,"");
		b3Assert(errNum==CL_SUCCESS);

		m_findConcaveSphereContactsKernel = b3OpenCLUtils::compileCLKernelFromString(m_context, m_device,primitiveContactsSrc, "findConcaveSphereContactsKernel",&errNum,primitiveContactsProg );
		b3Assert(errNum==CL_SUCCESS);
		b3Assert(m_findConcaveSphereContactsKernel);

		m_processCompoundPairsPrimitivesKernel = b3OpenCLUtils::compileCLKernelFromString(m_context, m_device,primitiveContactsSrc, "processCompoundPairsPrimitivesKernel",&errNum,primitiveContactsProg,"");
		b3Assert(errNum==CL_SUCCESS);
		b3Assert(m_processCompoundPairsPrimitivesKernel);
		 
	 }
	

}

GpuSatCollision::~GpuSatCollision()
{
	
	if (m_findSeparatingAxisVertexFaceKernel)
		clReleaseKernel(m_findSeparatingAxisVertexFaceKernel);

	if (m_findSeparatingAxisEdgeEdgeKernel)
		clReleaseKernel(m_findSeparatingAxisEdgeEdgeKernel);

	if (m_findSeparatingAxisUnitSphereKernel)
		clReleaseKernel(m_findSeparatingAxisUnitSphereKernel);

	if (m_mprPenetrationKernel)
		clReleaseKernel(m_mprPenetrationKernel);


	if (m_findSeparatingAxisKernel)
		clReleaseKernel(m_findSeparatingAxisKernel);

    if (m_findConcaveSeparatingAxisVertexFaceKernel)
        clReleaseKernel(m_findConcaveSeparatingAxisVertexFaceKernel);

    
    if (m_findConcaveSeparatingAxisEdgeEdgeKernel)
        clReleaseKernel(m_findConcaveSeparatingAxisEdgeEdgeKernel);
    
	if (m_findConcaveSeparatingAxisKernel)
		clReleaseKernel(m_findConcaveSeparatingAxisKernel);

	if (m_findCompoundPairsKernel)
		clReleaseKernel(m_findCompoundPairsKernel);

	if (m_processCompoundPairsKernel)
		clReleaseKernel(m_processCompoundPairsKernel);
    
    if (m_findClippingFacesKernel)
        clReleaseKernel(m_findClippingFacesKernel);
   
    if (m_clipFacesAndFindContacts)
        clReleaseKernel(m_clipFacesAndFindContacts);
    if (m_newContactReductionKernel)
        clReleaseKernel(m_newContactReductionKernel);
	if (m_primitiveContactsKernel)
		clReleaseKernel(m_primitiveContactsKernel);
    
	if (m_findConcaveSphereContactsKernel)
		clReleaseKernel(m_findConcaveSphereContactsKernel);

	if (m_processCompoundPairsPrimitivesKernel)
		clReleaseKernel(m_processCompoundPairsPrimitivesKernel);

	if (m_clipHullHullKernel)
		clReleaseKernel(m_clipHullHullKernel);
	if (m_clipCompoundsHullHullKernel)
		clReleaseKernel(m_clipCompoundsHullHullKernel);

	if (m_clipHullHullConcaveConvexKernel)
		clReleaseKernel(m_clipHullHullConcaveConvexKernel);
//	if (m_extractManifoldAndAddContactKernel)
	//	clReleaseKernel(m_extractManifoldAndAddContactKernel);

	if (m_bvhTraversalKernel)
		clReleaseKernel(m_bvhTraversalKernel);

}

struct MyTriangleCallback : public b3NodeOverlapCallback
{
	int m_bodyIndexA;
	int m_bodyIndexB;

	virtual void processNode(int subPart, int triangleIndex)
	{
		printf("bodyIndexA %d, bodyIndexB %d\n",m_bodyIndexA,m_bodyIndexB);
		printf("triangleIndex %d\n", triangleIndex);
	}
};


#define float4 b3Vector3
#define make_float4(x,y,z,w) b3MakeVector3(x,y,z,w)

float signedDistanceFromPointToPlane(const float4& point, const float4& planeEqn, float4* closestPointOnFace)
{
	float4 n = planeEqn;
	n[3] = 0.f;
	float dist = dot3F4(n, point) + planeEqn[3];
	*closestPointOnFace = point - dist * n;
	return dist;
}



#define cross3(a,b) (a.cross(b))
b3Vector3 transform(const b3Vector3* v, const b3Vector3* pos, const b3Quaternion* orn)
{
	b3Transform tr;
	tr.setIdentity();
	tr.setOrigin(*pos);
	tr.setRotation(*orn);
	b3Vector3 res = tr(*v);
	return res;
}


inline bool IsPointInPolygon(const float4& p, 
							const b3GpuFace* face,
							 const float4* baseVertex,
							const  int* convexIndices,
							float4* out)
{
    float4 a;
    float4 b;
    float4 ab;
    float4 ap;
    float4 v;

	float4 plane = b3MakeVector3(face->m_plane.x,face->m_plane.y,face->m_plane.z,0.f);
	
	if (face->m_numIndices<2)
		return false;

	
	float4 v0 = baseVertex[convexIndices[face->m_indexOffset + face->m_numIndices-1]];
	b = v0;

    for(unsigned i=0; i != face->m_numIndices; ++i)
    {
		a = b;
		float4 vi = baseVertex[convexIndices[face->m_indexOffset + i]];
		b = vi;
        ab = b-a;
        ap = p-a;
        v = cross3(ab,plane);

        if (b3Dot(ap, v) > 0.f)
        {
            float ab_m2 = b3Dot(ab, ab);
            float rt = ab_m2 != 0.f ? b3Dot(ab, ap) / ab_m2 : 0.f;
            if (rt <= 0.f)
            {
                *out = a;
            }
            else if (rt >= 1.f) 
            {
                *out = b;
            }
            else
            {
            	float s = 1.f - rt;
				out[0].x = s * a.x + rt * b.x;
				out[0].y = s * a.y + rt * b.y;
				out[0].z = s * a.z + rt * b.z;
            }
            return false;
        }
    }
    return true;
}

#define normalize3(a) (a.normalize())


int extractManifoldSequentialGlobal( const float4* p, int nPoints, const float4& nearNormal, b3Int4* contactIdx)
{
	if( nPoints == 0 )
        return 0;
    
    if (nPoints <=4)
        return nPoints;
    
    
    if (nPoints >64)
        nPoints = 64;
    
	float4 center = b3MakeVector3(0,0,0,0);
	{
		
		for (int i=0;i<nPoints;i++)
			center += p[i];
		center /= (float)nPoints;
	}
    
	
    
	//	sample 4 directions
    
    float4 aVector = p[0] - center;
    float4 u = cross3( nearNormal, aVector );
    float4 v = cross3( nearNormal, u );
    u = normalize3( u );
    v = normalize3( v );
    
    
    //keep point with deepest penetration
    float minW= FLT_MAX;
    
    int minIndex=-1;
    
    float4 maxDots;
    maxDots.x = FLT_MIN;
    maxDots.y = FLT_MIN;
    maxDots.z = FLT_MIN;
    maxDots.w = FLT_MIN;
    
    //	idx, distance
    for(int ie = 0; ie<nPoints; ie++ )
    {
        if (p[ie].w<minW)
        {
            minW = p[ie].w;
            minIndex=ie;
        }
        float f;
        float4 r = p[ie]-center;
        f = dot3F4( u, r );
        if (f<maxDots.x)
        {
            maxDots.x = f;
            contactIdx[0].x = ie;
        }
        
        f = dot3F4( -u, r );
        if (f<maxDots.y)
        {
            maxDots.y = f;
            contactIdx[0].y = ie;
        }
        
        
        f = dot3F4( v, r );
        if (f<maxDots.z)
        {
            maxDots.z = f;
            contactIdx[0].z = ie;
        }
        
        f = dot3F4( -v, r );
        if (f<maxDots.w)
        {
            maxDots.w = f;
            contactIdx[0].w = ie;
        }
        
    }
    
    if (contactIdx[0].x != minIndex && contactIdx[0].y != minIndex && contactIdx[0].z != minIndex && contactIdx[0].w != minIndex)
    {
        //replace the first contact with minimum (todo: replace contact with least penetration)
        contactIdx[0].x = minIndex;
    }
    
    return 4;
    
}



#define MAX_VERTS 1024


inline void project(const b3ConvexPolyhedronData& hull,  const float4& pos, const b3Quaternion& orn, const float4& dir, const b3AlignedObjectArray<b3Vector3>& vertices, b3Scalar& min, b3Scalar& max)
{
	min = FLT_MAX;
	max = -FLT_MAX;
	int numVerts = hull.m_numVertices;

	const float4 localDir = b3QuatRotate(orn.inverse(),dir);

	b3Scalar offset = dot3F4(pos,dir);

	for(int i=0;i<numVerts;i++)
	{
		//b3Vector3 pt = trans * vertices[m_vertexOffset+i];
		//b3Scalar dp = pt.dot(dir);
		//b3Vector3 vertex = vertices[hull.m_vertexOffset+i];
		b3Scalar dp = dot3F4((float4&)vertices[hull.m_vertexOffset+i],localDir);
		//b3Assert(dp==dpL);
		if(dp < min)	min = dp;
		if(dp > max)	max = dp;
	}
	if(min>max)
	{
		b3Scalar tmp = min;
		min = max;
		max = tmp;
	}
	min += offset;
	max += offset;
}


static bool TestSepAxis(const b3ConvexPolyhedronData& hullA, const b3ConvexPolyhedronData& hullB, 
	const float4& posA,const b3Quaternion& ornA,
	const float4& posB,const b3Quaternion& ornB,
	const float4& sep_axis, const b3AlignedObjectArray<b3Vector3>& verticesA,const b3AlignedObjectArray<b3Vector3>& verticesB,b3Scalar& depth)
{
	b3Scalar Min0,Max0;
	b3Scalar Min1,Max1;
	project(hullA,posA,ornA,sep_axis,verticesA, Min0, Max0);
	project(hullB,posB,ornB, sep_axis,verticesB, Min1, Max1);

	if(Max0<Min1 || Max1<Min0)
		return false;

	b3Scalar d0 = Max0 - Min1;
	assert(d0>=0.0f);
	b3Scalar d1 = Max1 - Min0;
	assert(d1>=0.0f);
	depth = d0<d1 ? d0:d1;
	return true;
}

inline bool IsAlmostZero(const b3Vector3& v)
{
	if(fabsf(v.x)>1e-6 || fabsf(v.y)>1e-6 || fabsf(v.z)>1e-6)	return false;
	return true;
}


static bool findSeparatingAxis(	const b3ConvexPolyhedronData& hullA, const b3ConvexPolyhedronData& hullB, 
	const float4& posA1,
	const b3Quaternion& ornA,
	const float4& posB1,
	const b3Quaternion& ornB,
	const b3AlignedObjectArray<b3Vector3>& verticesA,
	const b3AlignedObjectArray<b3Vector3>& uniqueEdgesA, 
	const b3AlignedObjectArray<b3GpuFace>& facesA,
	const b3AlignedObjectArray<int>& indicesA,
	const b3AlignedObjectArray<b3Vector3>& verticesB, 
	const b3AlignedObjectArray<b3Vector3>& uniqueEdgesB, 
	const b3AlignedObjectArray<b3GpuFace>& facesB,
	const b3AlignedObjectArray<int>& indicesB,

	b3Vector3& sep)
{
	B3_PROFILE("findSeparatingAxis");

	b3g_actualSATPairTests++;
	float4 posA = posA1;
	posA.w = 0.f;
	float4 posB = posB1;
	posB.w = 0.f;
//#ifdef TEST_INTERNAL_OBJECTS
	float4 c0local = (float4&)hullA.m_localCenter;
	float4 c0 = transform(&c0local, &posA, &ornA);
	float4 c1local = (float4&)hullB.m_localCenter;
	float4 c1 = transform(&c1local,&posB,&ornB);
	const float4 deltaC2 = c0 - c1;
//#endif

	b3Scalar dmin = FLT_MAX;
	int curPlaneTests=0;

	int numFacesA = hullA.m_numFaces;
	// Test normals from hullA
	for(int i=0;i<numFacesA;i++)
	{
		const float4& normal = (float4&)facesA[hullA.m_faceOffset+i].m_plane;
		float4 faceANormalWS = b3QuatRotate(ornA,normal);

		if (dot3F4(deltaC2,faceANormalWS)<0)
			faceANormalWS*=-1.f;

		curPlaneTests++;
#ifdef TEST_INTERNAL_OBJECTS
		gExpectedNbTests++;
		if(gUseInternalObject && !TestInternalObjects(transA,transB, DeltaC2, faceANormalWS, hullA, hullB, dmin))
			continue;
		gActualNbTests++;
#endif

		
		b3Scalar d;
		if(!TestSepAxis( hullA, hullB, posA,ornA,posB,ornB,faceANormalWS, verticesA, verticesB,d))
			return false;

		if(d<dmin)
		{
			dmin = d;
			sep = (b3Vector3&)faceANormalWS;
		}
	}

	int numFacesB = hullB.m_numFaces;
	// Test normals from hullB
	for(int i=0;i<numFacesB;i++)
	{
		float4 normal = (float4&)facesB[hullB.m_faceOffset+i].m_plane;
		float4 WorldNormal = b3QuatRotate(ornB, normal);

		if (dot3F4(deltaC2,WorldNormal)<0)
		{
			WorldNormal*=-1.f;
		}
		curPlaneTests++;
#ifdef TEST_INTERNAL_OBJECTS
		gExpectedNbTests++;
		if(gUseInternalObject && !TestInternalObjects(transA,transB,DeltaC2, WorldNormal, hullA, hullB, dmin))
			continue;
		gActualNbTests++;
#endif

		b3Scalar d;
		if(!TestSepAxis(hullA, hullB,posA,ornA,posB,ornB,WorldNormal,verticesA,verticesB,d))
			return false;

		if(d<dmin)
		{
			dmin = d;
			sep = (b3Vector3&)WorldNormal;
		}
	}

	int curEdgeEdge = 0;
	// Test edges
	for(int e0=0;e0<hullA.m_numUniqueEdges;e0++)
	{
		const float4& edge0 = (float4&) uniqueEdgesA[hullA.m_uniqueEdgesOffset+e0];
		float4 edge0World = b3QuatRotate(ornA,(float4&)edge0);

		for(int e1=0;e1<hullB.m_numUniqueEdges;e1++)
		{
			const b3Vector3 edge1 = uniqueEdgesB[hullB.m_uniqueEdgesOffset+e1];
			float4 edge1World = b3QuatRotate(ornB,(float4&)edge1);


			float4 crossje = cross3(edge0World,edge1World);

			curEdgeEdge++;
			if(!IsAlmostZero((b3Vector3&)crossje))
			{
				crossje = normalize3(crossje);
				if (dot3F4(deltaC2,crossje)<0)
					crossje*=-1.f;


#ifdef TEST_INTERNAL_OBJECTS
				gExpectedNbTests++;
				if(gUseInternalObject && !TestInternalObjects(transA,transB,DeltaC2, Cross, hullA, hullB, dmin))
					continue;
				gActualNbTests++;
#endif

				b3Scalar dist;
				if(!TestSepAxis( hullA, hullB, posA,ornA,posB,ornB,crossje, verticesA,verticesB,dist))
					return false;

				if(dist<dmin)
				{
					dmin = dist;
					sep = (b3Vector3&)crossje;
				}
			}
		}

	}

	
	if((dot3F4(-deltaC2,(float4&)sep))>0.0f)
		sep = -sep;

	return true;
}


bool findSeparatingAxisEdgeEdge(	__global const b3ConvexPolyhedronData* hullA, __global const b3ConvexPolyhedronData* hullB, 
	const b3Float4& posA1,
	const b3Quat& ornA,
	const b3Float4& posB1,
	const b3Quat& ornB,
	const b3Float4& DeltaC2,
	__global const b3AlignedObjectArray<float4>& vertices, 
	__global const b3AlignedObjectArray<float4>& uniqueEdges, 
	__global const b3AlignedObjectArray<b3GpuFace>& faces,
	__global const b3AlignedObjectArray<int>&  indices,
	float4* sep,
	float* dmin)
{
//	int i = get_global_id(0);

	float4 posA = posA1;
	posA.w = 0.f;
	float4 posB = posB1;
	posB.w = 0.f;

	//int curPlaneTests=0;

	int curEdgeEdge = 0;
	// Test edges
	for(int e0=0;e0<hullA->m_numUniqueEdges;e0++)
	{
		const float4 edge0 = uniqueEdges[hullA->m_uniqueEdgesOffset+e0];
		float4 edge0World = b3QuatRotate(ornA,edge0);

		for(int e1=0;e1<hullB->m_numUniqueEdges;e1++)
		{
			const float4 edge1 = uniqueEdges[hullB->m_uniqueEdgesOffset+e1];
			float4 edge1World = b3QuatRotate(ornB,edge1);


			float4 crossje = cross3(edge0World,edge1World);

			curEdgeEdge++;
			if(!IsAlmostZero(crossje))
			{
				crossje = normalize3(crossje);
				if (dot3F4(DeltaC2,crossje)<0)
					crossje*=-1.f;
					
				float dist;
				bool result = true;
				{
					float Min0,Max0;
					float Min1,Max1;
					project(*hullA,posA,ornA,crossje,vertices, Min0, Max0);
					project(*hullB,posB,ornB,crossje,vertices, Min1, Max1);
				
					if(Max0<Min1 || Max1<Min0)
						result = false;
				
					float d0 = Max0 - Min1;
					float d1 = Max1 - Min0;
					dist = d0<d1 ? d0:d1;
					result = true;

				}
				

				if(dist<*dmin)
				{
					*dmin = dist;
					*sep = crossje;
				}
			}
		}

	}

	
	if((dot3F4(-DeltaC2,*sep))>0.0f)
	{
		*sep = -(*sep);
	}
	return true;
}


__inline float4 lerp3(const float4& a,const float4& b, float  t)
{
	return b3MakeVector3(	a.x + (b.x - a.x) * t,
						a.y + (b.y - a.y) * t,
						a.z + (b.z - a.z) * t,
						0.f);
}


// Clips a face to the back of a plane, return the number of vertices out, stored in ppVtxOut
int clipFace(const float4* pVtxIn, int numVertsIn, float4& planeNormalWS,float planeEqWS, float4* ppVtxOut)
{
	
	int ve;
	float ds, de;
	int numVertsOut = 0;
	if (numVertsIn < 2)
		return 0;

	float4 firstVertex=pVtxIn[numVertsIn-1];
	float4 endVertex = pVtxIn[0];
	
	ds = dot3F4(planeNormalWS,firstVertex)+planeEqWS;

	for (ve = 0; ve < numVertsIn; ve++)
	{
		endVertex=pVtxIn[ve];

		de = dot3F4(planeNormalWS,endVertex)+planeEqWS;

		if (ds<0)
		{
			if (de<0)
			{
				// Start < 0, end < 0, so output endVertex
				ppVtxOut[numVertsOut++] = endVertex;
			}
			else
			{
				// Start < 0, end >= 0, so output intersection
				ppVtxOut[numVertsOut++] = lerp3(firstVertex, endVertex,(ds * 1.f/(ds - de)) );
			}
		}
		else
		{
			if (de<0)
			{
				// Start >= 0, end < 0 so output intersection and end
				ppVtxOut[numVertsOut++] = lerp3(firstVertex, endVertex,(ds * 1.f/(ds - de)) );
				ppVtxOut[numVertsOut++] = endVertex;
			}
		}
		firstVertex = endVertex;
		ds = de;
	}
	return numVertsOut;
}


int clipFaceAgainstHull(const float4& separatingNormal, const b3ConvexPolyhedronData* hullA,  
	const float4& posA, const b3Quaternion& ornA, float4* worldVertsB1, int numWorldVertsB1,
	float4* worldVertsB2, int capacityWorldVertsB2,
	const float minDist, float maxDist,
	const b3AlignedObjectArray<float4>& verticesA,	const b3AlignedObjectArray<b3GpuFace>& facesA,	const b3AlignedObjectArray<int>& indicesA,
	//const float4* verticesB,	const b3GpuFace* facesB,	const int* indicesB,
	float4* contactsOut,
	int contactCapacity)
{
	int numContactsOut = 0;

	float4* pVtxIn = worldVertsB1;
	float4* pVtxOut = worldVertsB2;
	
	int numVertsIn = numWorldVertsB1;
	int numVertsOut = 0;

	int closestFaceA=-1;
	{
		float dmin = FLT_MAX;
		for(int face=0;face<hullA->m_numFaces;face++)
		{
			const float4 Normal = b3MakeVector3(
				facesA[hullA->m_faceOffset+face].m_plane.x, 
				facesA[hullA->m_faceOffset+face].m_plane.y, 
				facesA[hullA->m_faceOffset+face].m_plane.z,0.f);
			const float4 faceANormalWS = b3QuatRotate(ornA,Normal);
		
			float d = dot3F4(faceANormalWS,separatingNormal);
			if (d < dmin)
			{
				dmin = d;
				closestFaceA = face;
			}
		}
	}
	if (closestFaceA<0)
		return numContactsOut;

	b3GpuFace polyA = facesA[hullA->m_faceOffset+closestFaceA];

	// clip polygon to back of planes of all faces of hull A that are adjacent to witness face
//	int numContacts = numWorldVertsB1;
	int numVerticesA = polyA.m_numIndices;
	for(int e0=0;e0<numVerticesA;e0++)
	{
		const float4 a = verticesA[hullA->m_vertexOffset+indicesA[polyA.m_indexOffset+e0]];
		const float4 b = verticesA[hullA->m_vertexOffset+indicesA[polyA.m_indexOffset+((e0+1)%numVerticesA)]];
		const float4 edge0 = a - b;
		const float4 WorldEdge0 = b3QuatRotate(ornA,edge0);
		float4 planeNormalA = make_float4(polyA.m_plane.x,polyA.m_plane.y,polyA.m_plane.z,0.f);
		float4 worldPlaneAnormal1 = b3QuatRotate(ornA,planeNormalA);

		float4 planeNormalWS1 = -cross3(WorldEdge0,worldPlaneAnormal1);
		float4 worldA1 = transform(&a,&posA,&ornA);
		float planeEqWS1 = -dot3F4(worldA1,planeNormalWS1);
		
		float4 planeNormalWS = planeNormalWS1;
		float planeEqWS=planeEqWS1;
		
		//clip face
		//clipFace(*pVtxIn, *pVtxOut,planeNormalWS,planeEqWS);
		numVertsOut = clipFace(pVtxIn, numVertsIn, planeNormalWS,planeEqWS, pVtxOut);

		//btSwap(pVtxIn,pVtxOut);
		float4* tmp = pVtxOut;
		pVtxOut = pVtxIn;
		pVtxIn = tmp;
		numVertsIn = numVertsOut;
		numVertsOut = 0;
	}

	
	// only keep points that are behind the witness face
	{
		float4 localPlaneNormal  = make_float4(polyA.m_plane.x,polyA.m_plane.y,polyA.m_plane.z,0.f);
		float localPlaneEq = polyA.m_plane.w;
		float4 planeNormalWS = b3QuatRotate(ornA,localPlaneNormal);
		float planeEqWS=localPlaneEq-dot3F4(planeNormalWS,posA);
		for (int i=0;i<numVertsIn;i++)
		{
			float depth = dot3F4(planeNormalWS,pVtxIn[i])+planeEqWS;
			if (depth <=minDist)
			{
				depth = minDist;
			}
			if (numContactsOut<contactCapacity)
			{
				if (depth <=maxDist)
				{
					float4 pointInWorld = pVtxIn[i];
					//resultOut.addContactPoint(separatingNormal,point,depth);
					contactsOut[numContactsOut++] = b3MakeVector3(pointInWorld.x,pointInWorld.y,pointInWorld.z,depth);
					//printf("depth=%f\n",depth);
				}
			} else
			{
				b3Error("exceeding contact capacity (%d,%df)\n", numContactsOut,contactCapacity);
			}
		}
	}

	return numContactsOut;
}



static int	clipHullAgainstHull(const float4& separatingNormal, 
	const b3ConvexPolyhedronData& hullA, const b3ConvexPolyhedronData& hullB, 
	const float4& posA, const b3Quaternion& ornA,const float4& posB, const b3Quaternion& ornB, 
	float4* worldVertsB1, float4* worldVertsB2, int capacityWorldVerts,
	const float minDist, float maxDist,
	const b3AlignedObjectArray<float4>& verticesA,	const b3AlignedObjectArray<b3GpuFace>& facesA,	const b3AlignedObjectArray<int>& indicesA,
	const b3AlignedObjectArray<float4>& verticesB,	const b3AlignedObjectArray<b3GpuFace>& facesB,	const b3AlignedObjectArray<int>& indicesB,

	float4*	contactsOut,
	int contactCapacity)
{
	int numContactsOut = 0;
	int numWorldVertsB1= 0;
	
	B3_PROFILE("clipHullAgainstHull");

//	float curMaxDist=maxDist;
	int closestFaceB=-1;
	float dmax = -FLT_MAX;

	{
		//B3_PROFILE("closestFaceB");
		if (hullB.m_numFaces!=1)
		{
			//printf("wtf\n");
		}
		static bool once = true;
		//printf("separatingNormal=%f,%f,%f\n",separatingNormal.x,separatingNormal.y,separatingNormal.z);
		
		for(int face=0;face<hullB.m_numFaces;face++)
		{
#ifdef BT_DEBUG_SAT_FACE
			if (once)
				printf("face %d\n",face);
			const b3GpuFace* faceB = &facesB[hullB.m_faceOffset+face];
			if (once)
			{
				for (int i=0;i<faceB->m_numIndices;i++)
				{
					float4 vert = verticesB[hullB.m_vertexOffset+indicesB[faceB->m_indexOffset+i]];
					printf("vert[%d] = %f,%f,%f\n",i,vert.x,vert.y,vert.z);
				}
			}
#endif //BT_DEBUG_SAT_FACE
			//if (facesB[hullB.m_faceOffset+face].m_numIndices>2)
			{
				const float4 Normal = b3MakeVector3(facesB[hullB.m_faceOffset+face].m_plane.x, 
					facesB[hullB.m_faceOffset+face].m_plane.y, facesB[hullB.m_faceOffset+face].m_plane.z,0.f);
				const float4 WorldNormal = b3QuatRotate(ornB, Normal);
#ifdef BT_DEBUG_SAT_FACE
				if (once)
					printf("faceNormal = %f,%f,%f\n",Normal.x,Normal.y,Normal.z);
#endif
				float d = dot3F4(WorldNormal,separatingNormal);
				if (d > dmax)
				{
					dmax = d;
					closestFaceB = face;
				}
			}
		}
		once = false;
	}

	
	b3Assert(closestFaceB>=0);
	{
		//B3_PROFILE("worldVertsB1");
		const b3GpuFace& polyB = facesB[hullB.m_faceOffset+closestFaceB];
		const int numVertices = polyB.m_numIndices;
		for(int e0=0;e0<numVertices;e0++)
		{
			const float4& b = verticesB[hullB.m_vertexOffset+indicesB[polyB.m_indexOffset+e0]];
			worldVertsB1[numWorldVertsB1++] = transform(&b,&posB,&ornB);
		}
	}

	if (closestFaceB>=0)
	{
		//B3_PROFILE("clipFaceAgainstHull");
		numContactsOut = clipFaceAgainstHull((float4&)separatingNormal, &hullA, 
				posA,ornA,
				worldVertsB1,numWorldVertsB1,worldVertsB2,capacityWorldVerts, minDist, maxDist,
				verticesA,				facesA,				indicesA,
				contactsOut,contactCapacity);
	}

	return numContactsOut;
}






#define PARALLEL_SUM(v, n) for(int j=1; j<n; j++) v[0] += v[j];
#define PARALLEL_DO(execution, n) for(int ie=0; ie<n; ie++){execution;}
#define REDUCE_MAX(v, n) {int i=0;\
for(int offset=0; offset<n; offset++) v[i] = (v[i].y > v[i+offset].y)? v[i]: v[i+offset]; }
#define REDUCE_MIN(v, n) {int i=0;\
for(int offset=0; offset<n; offset++) v[i] = (v[i].y < v[i+offset].y)? v[i]: v[i+offset]; }

int extractManifold(const float4* p, int nPoints, const float4& nearNormal, b3Int4* contactIdx)
{
	if( nPoints == 0 )
        return 0;
    
    if (nPoints <=4)
        return nPoints;
    
    
    if (nPoints >64)
        nPoints = 64;
    
	float4 center = make_float4(0,0,0,0);
	{
		
		for (int i=0;i<nPoints;i++)
			center += p[i];
		center /= (float)nPoints;
	}
    
	
    
	//	sample 4 directions
    
    float4 aVector = p[0] - center;
    float4 u = cross3( nearNormal, aVector );
    float4 v = cross3( nearNormal, u );
    u = normalize3( u );
    v = normalize3( v );
    
    
    //keep point with deepest penetration
    float minW= FLT_MAX;
    
    int minIndex=-1;
    
    float4 maxDots;
    maxDots.x = FLT_MIN;
    maxDots.y = FLT_MIN;
    maxDots.z = FLT_MIN;
    maxDots.w = FLT_MIN;
    
    //	idx, distance
    for(int ie = 0; ie<nPoints; ie++ )
    {
        if (p[ie].w<minW)
        {
            minW = p[ie].w;
            minIndex=ie;
        }
        float f;
        float4 r = p[ie]-center;
        f = dot3F4( u, r );
        if (f<maxDots.x)
        {
            maxDots.x = f;
            contactIdx[0].x = ie;
        }
        
        f = dot3F4( -u, r );
        if (f<maxDots.y)
        {
            maxDots.y = f;
            contactIdx[0].y = ie;
        }
        
        
        f = dot3F4( v, r );
        if (f<maxDots.z)
        {
            maxDots.z = f;
            contactIdx[0].z = ie;
        }
        
        f = dot3F4( -v, r );
        if (f<maxDots.w)
        {
            maxDots.w = f;
            contactIdx[0].w = ie;
        }
        
    }
    
    if (contactIdx[0].x != minIndex && contactIdx[0].y != minIndex && contactIdx[0].z != minIndex && contactIdx[0].w != minIndex)
    {
        //replace the first contact with minimum (todo: replace contact with least penetration)
        contactIdx[0].x = minIndex;
    }
    
    return 4;
    
}




int clipHullHullSingle(
			int bodyIndexA, int bodyIndexB,
										 const float4& posA,
										 const b3Quaternion& ornA,
										 const float4& posB,
										 const b3Quaternion& ornB,

			int collidableIndexA, int collidableIndexB,

			const b3AlignedObjectArray<b3RigidBodyData>* bodyBuf, 
			b3AlignedObjectArray<b3Contact4>* globalContactOut, 
			int& nContacts,
			
			const b3AlignedObjectArray<b3ConvexPolyhedronData>& hostConvexDataA,
			const b3AlignedObjectArray<b3ConvexPolyhedronData>& hostConvexDataB,
	
			const b3AlignedObjectArray<b3Vector3>& verticesA, 
			const b3AlignedObjectArray<b3Vector3>& uniqueEdgesA, 
			const b3AlignedObjectArray<b3GpuFace>& facesA,
			const b3AlignedObjectArray<int>& indicesA,
	
			const b3AlignedObjectArray<b3Vector3>& verticesB,
			const b3AlignedObjectArray<b3Vector3>& uniqueEdgesB,
			const b3AlignedObjectArray<b3GpuFace>& facesB,
			const b3AlignedObjectArray<int>& indicesB,

			const b3AlignedObjectArray<b3Collidable>& hostCollidablesA,
			const b3AlignedObjectArray<b3Collidable>& hostCollidablesB,
			const b3Vector3& sepNormalWorldSpace,
			int maxContactCapacity			)
{
	int contactIndex = -1;
	b3ConvexPolyhedronData hullA, hullB;
    
    b3Collidable colA = hostCollidablesA[collidableIndexA];
    hullA = hostConvexDataA[colA.m_shapeIndex];
    //printf("numvertsA = %d\n",hullA.m_numVertices);
    
    
    b3Collidable colB = hostCollidablesB[collidableIndexB];
    hullB = hostConvexDataB[colB.m_shapeIndex];
    //printf("numvertsB = %d\n",hullB.m_numVertices);
    
	
	float4 contactsOut[MAX_VERTS];
	int localContactCapacity = MAX_VERTS;

#ifdef _WIN32
	b3Assert(_finite(bodyBuf->at(bodyIndexA).m_pos.x));
	b3Assert(_finite(bodyBuf->at(bodyIndexB).m_pos.x));
#endif
	
	
	{
		
		float4 worldVertsB1[MAX_VERTS];
		float4 worldVertsB2[MAX_VERTS];
		int capacityWorldVerts = MAX_VERTS;

		float4 hostNormal = make_float4(sepNormalWorldSpace.x,sepNormalWorldSpace.y,sepNormalWorldSpace.z,0.f);
		int shapeA = hostCollidablesA[collidableIndexA].m_shapeIndex;
		int shapeB = hostCollidablesB[collidableIndexB].m_shapeIndex;

		b3Scalar minDist = -1;
		b3Scalar maxDist = 0.;

		        

		b3Transform trA,trB;
		{
		//B3_PROFILE("transform computation");
		//trA.setIdentity();
		trA.setOrigin(b3MakeVector3(posA.x,posA.y,posA.z));
		trA.setRotation(b3Quaternion(ornA.x,ornA.y,ornA.z,ornA.w));
				
		//trB.setIdentity();
		trB.setOrigin(b3MakeVector3(posB.x,posB.y,posB.z));
		trB.setRotation(b3Quaternion(ornB.x,ornB.y,ornB.z,ornB.w));
		}

		b3Quaternion trAorn = trA.getRotation();
        b3Quaternion trBorn = trB.getRotation();
        
		int numContactsOut = clipHullAgainstHull(hostNormal, 
						hostConvexDataA.at(shapeA), 
						hostConvexDataB.at(shapeB),
								(float4&)trA.getOrigin(), (b3Quaternion&)trAorn,
								(float4&)trB.getOrigin(), (b3Quaternion&)trBorn,
								worldVertsB1,worldVertsB2,capacityWorldVerts,
								minDist, maxDist,
								verticesA,	facesA,indicesA,
								verticesB,	facesB,indicesB,
								
								contactsOut,localContactCapacity);

		if (numContactsOut>0)
		{
			B3_PROFILE("overlap");

			float4 normalOnSurfaceB = (float4&)hostNormal;
			
			b3Int4 contactIdx;
			contactIdx.x = 0;
			contactIdx.y = 1;
			contactIdx.z = 2;
			contactIdx.w = 3;
			
			int numPoints = 0;
					
			{
			//	B3_PROFILE("extractManifold");
				numPoints = extractManifold(contactsOut, numContactsOut, normalOnSurfaceB,  &contactIdx);
			}
					
			b3Assert(numPoints);
					
			if (nContacts<maxContactCapacity)
			{
				contactIndex = nContacts;
				globalContactOut->expand();
				b3Contact4& contact = globalContactOut->at(nContacts);
				contact.m_batchIdx = 0;//i;
				contact.m_bodyAPtrAndSignBit = (bodyBuf->at(bodyIndexA).m_invMass==0)? -bodyIndexA:bodyIndexA;
				contact.m_bodyBPtrAndSignBit = (bodyBuf->at(bodyIndexB).m_invMass==0)? -bodyIndexB:bodyIndexB;

				contact.m_frictionCoeffCmp = 45874;
				contact.m_restituitionCoeffCmp = 0;
					
	//			float distance = 0.f;
				for (int p=0;p<numPoints;p++)
				{
					contact.m_worldPosB[p] = contactsOut[contactIdx.s[p]];//check if it is actually on B
					contact.m_worldNormalOnB = normalOnSurfaceB; 
				}
				//printf("bodyIndexA %d,bodyIndexB %d,normal=%f,%f,%f numPoints %d\n",bodyIndexA,bodyIndexB,normalOnSurfaceB.x,normalOnSurfaceB.y,normalOnSurfaceB.z,numPoints);
				contact.m_worldNormalOnB.w = (b3Scalar)numPoints;
				nContacts++;
			} else
			{
				b3Error("Error: exceeding contact capacity (%d/%d)\n", nContacts,maxContactCapacity);
			}
		}
	}
	return contactIndex;
}

	



void computeContactPlaneConvex(int pairIndex,
																int bodyIndexA, int bodyIndexB, 
																int collidableIndexA, int collidableIndexB, 
																const b3RigidBodyData* rigidBodies, 
																const b3Collidable* collidables,
																const b3ConvexPolyhedronData* convexShapes,
																const b3Vector3* convexVertices,
																const int* convexIndices,
																const b3GpuFace* faces,
																b3Contact4* globalContactsOut,
																int& nGlobalContactsOut,
																int maxContactCapacity)
{

		int shapeIndex = collidables[collidableIndexB].m_shapeIndex;
	const b3ConvexPolyhedronData* hullB = &convexShapes[shapeIndex];
	
	b3Vector3 posB = rigidBodies[bodyIndexB].m_pos;
	b3Quaternion ornB = rigidBodies[bodyIndexB].m_quat;
	b3Vector3 posA = rigidBodies[bodyIndexA].m_pos;
	b3Quaternion ornA = rigidBodies[bodyIndexA].m_quat;

//	int numContactsOut = 0;
//	int numWorldVertsB1= 0;

	b3Vector3 planeEq = faces[collidables[collidableIndexA].m_shapeIndex].m_plane;
	b3Vector3 planeNormal=b3MakeVector3(planeEq.x,planeEq.y,planeEq.z);
	b3Vector3 planeNormalWorld = b3QuatRotate(ornA,planeNormal);
	float planeConstant = planeEq.w;
	b3Transform convexWorldTransform;
	convexWorldTransform.setIdentity();
	convexWorldTransform.setOrigin(posB);
	convexWorldTransform.setRotation(ornB);
	b3Transform planeTransform;
	planeTransform.setIdentity();
	planeTransform.setOrigin(posA);
	planeTransform.setRotation(ornA);

	b3Transform planeInConvex;
	planeInConvex= convexWorldTransform.inverse() * planeTransform;
	b3Transform convexInPlane;
	convexInPlane = planeTransform.inverse() * convexWorldTransform;
	
	b3Vector3 planeNormalInConvex = planeInConvex.getBasis()*-planeNormal;
	float maxDot = -1e30;
	int hitVertex=-1;
	b3Vector3 hitVtx;

#define MAX_PLANE_CONVEX_POINTS 64

	b3Vector3 contactPoints[MAX_PLANE_CONVEX_POINTS];
	int numPoints = 0;

	b3Int4 contactIdx;
	contactIdx.s[0] = 0;
	contactIdx.s[1] = 1;
	contactIdx.s[2] = 2;
	contactIdx.s[3] = 3;
	
	for (int i=0;i<hullB->m_numVertices;i++)
	{
		b3Vector3 vtx = convexVertices[hullB->m_vertexOffset+i];
		float curDot = vtx.dot(planeNormalInConvex);


		if (curDot>maxDot)
		{
			hitVertex=i;
			maxDot=curDot;
			hitVtx = vtx;
			//make sure the deepest points is always included
			if (numPoints==MAX_PLANE_CONVEX_POINTS)
				numPoints--;
		}

		if (numPoints<MAX_PLANE_CONVEX_POINTS)
		{
			b3Vector3 vtxWorld = convexWorldTransform*vtx;
			b3Vector3 vtxInPlane = planeTransform.inverse()*vtxWorld;
			float dist = planeNormal.dot(vtxInPlane)-planeConstant;
			if (dist<0.f)
			{
				vtxWorld.w = dist;
				contactPoints[numPoints] = vtxWorld;
				numPoints++;
			}
		}

	}

	int numReducedPoints  = 0;

	numReducedPoints = numPoints;
	
	if (numPoints>4)
	{
		numReducedPoints = extractManifoldSequentialGlobal( contactPoints, numPoints, planeNormalInConvex, &contactIdx);
	}
	int dstIdx;
//    dstIdx = nGlobalContactsOut++;//AppendInc( nGlobalContactsOut, dstIdx );
		
	if (numReducedPoints>0)
	{
		if (nGlobalContactsOut < maxContactCapacity)
		{
			dstIdx=nGlobalContactsOut;
			nGlobalContactsOut++;

			b3Contact4* c = &globalContactsOut[dstIdx];
			c->m_worldNormalOnB = -planeNormalWorld;
			c->setFrictionCoeff(0.7);
			c->setRestituitionCoeff(0.f);

			c->m_batchIdx = pairIndex;
			c->m_bodyAPtrAndSignBit = rigidBodies[bodyIndexA].m_invMass==0?-bodyIndexA:bodyIndexA;
			c->m_bodyBPtrAndSignBit = rigidBodies[bodyIndexB].m_invMass==0?-bodyIndexB:bodyIndexB;
			for (int i=0;i<numReducedPoints;i++)
			{
				b3Vector3 pOnB1 = contactPoints[contactIdx.s[i]];
				c->m_worldPosB[i] = pOnB1;
			}
			c->m_worldNormalOnB.w = (b3Scalar)numReducedPoints;
		}//if (dstIdx < numPairs)
	}	
		


//	printf("computeContactPlaneConvex\n");
}



B3_FORCE_INLINE b3Vector3	MyUnQuantize(const unsigned short* vecIn, const b3Vector3& quantization, const b3Vector3& bvhAabbMin)
	{
			b3Vector3	vecOut;
			vecOut.setValue(
			(b3Scalar)(vecIn[0]) / (quantization.x),
			(b3Scalar)(vecIn[1]) / (quantization.y),
			(b3Scalar)(vecIn[2]) / (quantization.z));
			vecOut += bvhAabbMin;
			return vecOut;
	}

void traverseTreeTree()
{

}

#include "Bullet3Common/shared/b3Mat3x3.h"

int numAabbChecks = 0;
int maxNumAabbChecks = 0;
int maxDepth = 0;

// work-in-progress
__kernel void   findCompoundPairsKernel( 
	int pairIndex,
	int bodyIndexA,
	int bodyIndexB,
	int collidableIndexA,
	int collidableIndexB,
	__global const b3RigidBodyData* rigidBodies, 
	__global const b3Collidable* collidables,
	__global const b3ConvexPolyhedronData* convexShapes, 
	__global const b3AlignedObjectArray<b3Float4>& vertices,
	__global const b3AlignedObjectArray<b3Aabb>& aabbsWorldSpace,
	__global const b3AlignedObjectArray<b3Aabb>& aabbsLocalSpace,
	__global const b3GpuChildShape* gpuChildShapes,
	__global b3Int4* gpuCompoundPairsOut,
	__global  int* numCompoundPairsOut,
	int maxNumCompoundPairsCapacity,
	b3AlignedObjectArray<b3QuantizedBvhNode>&	treeNodesCPU,
	b3AlignedObjectArray<b3BvhSubtreeInfo>&	subTreesCPU,
	b3AlignedObjectArray<b3BvhInfo>&	bvhInfoCPU
	)
{
	numAabbChecks=0;
	maxNumAabbChecks=0;
//	int i = pairIndex;
	{
		

		int shapeIndexA = collidables[collidableIndexA].m_shapeIndex;
		int shapeIndexB = collidables[collidableIndexB].m_shapeIndex;


		//once the broadphase avoids static-static pairs, we can remove this test
		if ((rigidBodies[bodyIndexA].m_invMass==0) &&(rigidBodies[bodyIndexB].m_invMass==0))
		{
			return;
		}

		if ((collidables[collidableIndexA].m_shapeType==SHAPE_COMPOUND_OF_CONVEX_HULLS) &&(collidables[collidableIndexB].m_shapeType==SHAPE_COMPOUND_OF_CONVEX_HULLS))
		{
			int bvhA = collidables[collidableIndexA].m_compoundBvhIndex;
			int bvhB = collidables[collidableIndexB].m_compoundBvhIndex;
			int numSubTreesA = bvhInfoCPU[bvhA].m_numSubTrees;
			int subTreesOffsetA = bvhInfoCPU[bvhA].m_subTreeOffset;
			int subTreesOffsetB = bvhInfoCPU[bvhB].m_subTreeOffset;


			int numSubTreesB = bvhInfoCPU[bvhB].m_numSubTrees;
			
			float4 posA = rigidBodies[bodyIndexA].m_pos;
			b3Quat ornA = rigidBodies[bodyIndexA].m_quat;

			b3Transform transA;
			transA.setIdentity();
			transA.setOrigin(posA);
			transA.setRotation(ornA);

			b3Quat ornB = rigidBodies[bodyIndexB].m_quat;
			float4 posB = rigidBodies[bodyIndexB].m_pos;

			b3Transform transB;
			transB.setIdentity();
			transB.setOrigin(posB);
			transB.setRotation(ornB);



			for (int p=0;p<numSubTreesA;p++)
			{
				b3BvhSubtreeInfo subtreeA = subTreesCPU[subTreesOffsetA+p];
				//bvhInfoCPU[bvhA].m_quantization
				b3Vector3 treeAminLocal = MyUnQuantize(subtreeA.m_quantizedAabbMin,bvhInfoCPU[bvhA].m_quantization,bvhInfoCPU[bvhA].m_aabbMin);
				b3Vector3 treeAmaxLocal = MyUnQuantize(subtreeA.m_quantizedAabbMax,bvhInfoCPU[bvhA].m_quantization,bvhInfoCPU[bvhA].m_aabbMin);

				b3Vector3 aabbAMinOut,aabbAMaxOut;
				float margin=0.f;
				b3TransformAabb2(treeAminLocal,treeAmaxLocal, margin,transA.getOrigin(),transA.getRotation(),&aabbAMinOut,&aabbAMaxOut);

				for (int q=0;q<numSubTreesB;q++)
				{
					b3BvhSubtreeInfo subtreeB = subTreesCPU[subTreesOffsetB+q];

					b3Vector3 treeBminLocal = MyUnQuantize(subtreeB.m_quantizedAabbMin,bvhInfoCPU[bvhB].m_quantization,bvhInfoCPU[bvhB].m_aabbMin);
					b3Vector3 treeBmaxLocal = MyUnQuantize(subtreeB.m_quantizedAabbMax,bvhInfoCPU[bvhB].m_quantization,bvhInfoCPU[bvhB].m_aabbMin);

					b3Vector3 aabbBMinOut,aabbBMaxOut;
					float margin=0.f;
					b3TransformAabb2(treeBminLocal,treeBmaxLocal, margin,transB.getOrigin(),transB.getRotation(),&aabbBMinOut,&aabbBMaxOut);

					
					numAabbChecks=0;
					bool aabbOverlap = b3TestAabbAgainstAabb(aabbAMinOut,aabbAMaxOut,aabbBMinOut,aabbBMaxOut);
					if (aabbOverlap)
					{
						
						int startNodeIndexA = subtreeA.m_rootNodeIndex+bvhInfoCPU[bvhA].m_nodeOffset;
		//				int endNodeIndexA = startNodeIndexA+subtreeA.m_subtreeSize;

						int startNodeIndexB = subtreeB.m_rootNodeIndex+bvhInfoCPU[bvhB].m_nodeOffset;
		//				int endNodeIndexB = startNodeIndexB+subtreeB.m_subtreeSize;

						b3AlignedObjectArray<b3Int2> nodeStack;
						b3Int2 node0;
						node0.x = startNodeIndexA;
						node0.y = startNodeIndexB;

						int maxStackDepth = 1024;
						nodeStack.resize(maxStackDepth);
						int depth=0;
						nodeStack[depth++]=node0;

						do
						{
							if (depth > maxDepth)
							{
								maxDepth=depth;
								printf("maxDepth=%d\n",maxDepth);
							}
							b3Int2 node = nodeStack[--depth];
							
							b3Vector3 aMinLocal = MyUnQuantize(treeNodesCPU[node.x].m_quantizedAabbMin,bvhInfoCPU[bvhA].m_quantization,bvhInfoCPU[bvhA].m_aabbMin);
							b3Vector3 aMaxLocal = MyUnQuantize(treeNodesCPU[node.x].m_quantizedAabbMax,bvhInfoCPU[bvhA].m_quantization,bvhInfoCPU[bvhA].m_aabbMin);

							b3Vector3 bMinLocal = MyUnQuantize(treeNodesCPU[node.y].m_quantizedAabbMin,bvhInfoCPU[bvhB].m_quantization,bvhInfoCPU[bvhB].m_aabbMin);
							b3Vector3 bMaxLocal = MyUnQuantize(treeNodesCPU[node.y].m_quantizedAabbMax,bvhInfoCPU[bvhB].m_quantization,bvhInfoCPU[bvhB].m_aabbMin);

							float margin=0.f;
							b3Vector3 aabbAMinOut,aabbAMaxOut;
							b3TransformAabb2(aMinLocal,aMaxLocal, margin,transA.getOrigin(),transA.getRotation(),&aabbAMinOut,&aabbAMaxOut);

							b3Vector3 aabbBMinOut,aabbBMaxOut;
							b3TransformAabb2(bMinLocal,bMaxLocal, margin,transB.getOrigin(),transB.getRotation(),&aabbBMinOut,&aabbBMaxOut);

							numAabbChecks++;
							bool nodeOverlap = b3TestAabbAgainstAabb(aabbAMinOut,aabbAMaxOut,aabbBMinOut,aabbBMaxOut);
							if (nodeOverlap)
							{
								bool isLeafA = treeNodesCPU[node.x].isLeafNode();
								bool isLeafB = treeNodesCPU[node.y].isLeafNode();
								bool isInternalA = !isLeafA;
								bool isInternalB = !isLeafB;

								//fail, even though it might hit two leaf nodes
								if (depth+4>maxStackDepth && !(isLeafA && isLeafB))
								{
									b3Error("Error: traversal exceeded maxStackDepth\n");
									continue;
								}

								if(isInternalA)
								{
									int nodeAleftChild = node.x+1;
									bool isNodeALeftChildLeaf = treeNodesCPU[node.x+1].isLeafNode();
									int nodeArightChild = isNodeALeftChildLeaf? node.x+2 : node.x+1 + treeNodesCPU[node.x+1].getEscapeIndex();

									if(isInternalB)
									{					
										int nodeBleftChild = node.y+1;
										bool isNodeBLeftChildLeaf = treeNodesCPU[node.y+1].isLeafNode();
										int nodeBrightChild = isNodeBLeftChildLeaf? node.y+2 : node.y+1 + treeNodesCPU[node.y+1].getEscapeIndex();

										nodeStack[depth++] = b3MakeInt2(nodeAleftChild, nodeBleftChild);
										nodeStack[depth++] = b3MakeInt2(nodeArightChild, nodeBleftChild);
										nodeStack[depth++] = b3MakeInt2(nodeAleftChild, nodeBrightChild);
										nodeStack[depth++] = b3MakeInt2(nodeArightChild, nodeBrightChild);
									}
									else
									{
										nodeStack[depth++] = b3MakeInt2(nodeAleftChild,node.y);
										nodeStack[depth++] = b3MakeInt2(nodeArightChild,node.y);
									}
								}
								else
								{
									if(isInternalB)
									{
										int nodeBleftChild = node.y+1;
										bool isNodeBLeftChildLeaf = treeNodesCPU[node.y+1].isLeafNode();
										int nodeBrightChild = isNodeBLeftChildLeaf? node.y+2 : node.y+1 + treeNodesCPU[node.y+1].getEscapeIndex();
										nodeStack[depth++] = b3MakeInt2(node.x,nodeBleftChild);
										nodeStack[depth++] = b3MakeInt2(node.x,nodeBrightChild);
									}
									else
									{
										int compoundPairIdx = b3AtomicInc(numCompoundPairsOut);
										if (compoundPairIdx<maxNumCompoundPairsCapacity)
										{
											int childShapeIndexA = treeNodesCPU[node.x].getTriangleIndex();
											int childShapeIndexB = treeNodesCPU[node.y].getTriangleIndex();
											gpuCompoundPairsOut[compoundPairIdx]  = b3MakeInt4(bodyIndexA,bodyIndexB,childShapeIndexA,childShapeIndexB);
										}
									}
								}
							}
						} while (depth);
						maxNumAabbChecks = b3Max(numAabbChecks,maxNumAabbChecks);
					}
				}
			}
			
			return;
		}

		if ((collidables[collidableIndexA].m_shapeType==SHAPE_COMPOUND_OF_CONVEX_HULLS) ||(collidables[collidableIndexB].m_shapeType==SHAPE_COMPOUND_OF_CONVEX_HULLS))
		{

			if (collidables[collidableIndexA].m_shapeType==SHAPE_COMPOUND_OF_CONVEX_HULLS) 
			{

				int numChildrenA = collidables[collidableIndexA].m_numChildShapes;
				for (int c=0;c<numChildrenA;c++)
				{
					int childShapeIndexA = collidables[collidableIndexA].m_shapeIndex+c;
					int childColIndexA = gpuChildShapes[childShapeIndexA].m_shapeIndex;

					float4 posA = rigidBodies[bodyIndexA].m_pos;
					b3Quat ornA = rigidBodies[bodyIndexA].m_quat;
					float4 childPosA = gpuChildShapes[childShapeIndexA].m_childPosition;
					b3Quat childOrnA = gpuChildShapes[childShapeIndexA].m_childOrientation;
					float4 newPosA = b3QuatRotate(ornA,childPosA)+posA;
					b3Quat newOrnA = b3QuatMul(ornA,childOrnA);
					

					
					b3Aabb aabbA = aabbsLocalSpace[childColIndexA];

					
					b3Transform transA;
					transA.setIdentity();
					transA.setOrigin(newPosA);
					transA.setRotation(newOrnA);
					b3Scalar margin=0.0f;

					b3Vector3 aabbAMinOut,aabbAMaxOut;

					b3TransformAabb2((const b3Float4&)aabbA.m_min,(const b3Float4&)aabbA.m_max, margin,transA.getOrigin(),transA.getRotation(),&aabbAMinOut,&aabbAMaxOut);

					if (collidables[collidableIndexB].m_shapeType==SHAPE_COMPOUND_OF_CONVEX_HULLS)
					{
						int numChildrenB = collidables[collidableIndexB].m_numChildShapes;
						for (int b=0;b<numChildrenB;b++)
						{
							int childShapeIndexB = collidables[collidableIndexB].m_shapeIndex+b;
							int childColIndexB = gpuChildShapes[childShapeIndexB].m_shapeIndex;
							b3Quat ornB = rigidBodies[bodyIndexB].m_quat;
							float4 posB = rigidBodies[bodyIndexB].m_pos;
							float4 childPosB = gpuChildShapes[childShapeIndexB].m_childPosition;
							b3Quat childOrnB = gpuChildShapes[childShapeIndexB].m_childOrientation;
							float4 newPosB = transform(&childPosB,&posB,&ornB);
							b3Quat newOrnB = b3QuatMul(ornB,childOrnB);

							

							b3Aabb aabbB = aabbsLocalSpace[childColIndexB];

							b3Transform transB;
							transB.setIdentity();
							transB.setOrigin(newPosB);
							transB.setRotation(newOrnB);

							b3Vector3 aabbBMinOut,aabbBMaxOut;
							b3TransformAabb2((const b3Float4&)aabbB.m_min,(const b3Float4&)aabbB.m_max, margin,transB.getOrigin(),transB.getRotation(),&aabbBMinOut,&aabbBMaxOut);

							numAabbChecks++;
							bool aabbOverlap = b3TestAabbAgainstAabb(aabbAMinOut,aabbAMaxOut,aabbBMinOut,aabbBMaxOut);
							if (aabbOverlap)
							{
								/*
								int numFacesA = convexShapes[shapeIndexA].m_numFaces;
								float dmin = FLT_MAX;
								float4 posA = newPosA;
								posA.w = 0.f;
								float4 posB = newPosB;
								posB.w = 0.f;
								float4 c0local = convexShapes[shapeIndexA].m_localCenter;
								b3Quat ornA = newOrnA;
								float4 c0 = transform(&c0local, &posA, &ornA);
								float4 c1local = convexShapes[shapeIndexB].m_localCenter;
								b3Quat ornB =newOrnB;
								float4 c1 = transform(&c1local,&posB,&ornB);
								const float4 DeltaC2 = c0 - c1;
								*/
								{//
									int compoundPairIdx = b3AtomicInc(numCompoundPairsOut);
									if (compoundPairIdx<maxNumCompoundPairsCapacity)
									{
										gpuCompoundPairsOut[compoundPairIdx]  = b3MakeInt4(bodyIndexA,bodyIndexB,childShapeIndexA,childShapeIndexB);
									}
								}//
							}//fi(1)
						} //for (int b=0
					}//if (collidables[collidableIndexB].
					else//if (collidables[collidableIndexB].m_shapeType==SHAPE_COMPOUND_OF_CONVEX_HULLS)
					{
						if (1)
						{
						//	int numFacesA = convexShapes[shapeIndexA].m_numFaces;
						//	float dmin = FLT_MAX;
							float4 posA = newPosA;
							posA.w = 0.f;
							float4 posB = rigidBodies[bodyIndexB].m_pos;
							posB.w = 0.f;
							float4 c0local = convexShapes[shapeIndexA].m_localCenter;
							b3Quat ornA = newOrnA;
							float4 c0;
							c0 = transform(&c0local, &posA, &ornA);
							float4 c1local = convexShapes[shapeIndexB].m_localCenter;
							b3Quat ornB = rigidBodies[bodyIndexB].m_quat;
							float4 c1;
							c1 = transform(&c1local,&posB,&ornB);
						//	const float4 DeltaC2 = c0 - c1;

							{
								int compoundPairIdx = b3AtomicInc(numCompoundPairsOut);
								if (compoundPairIdx<maxNumCompoundPairsCapacity)
								{
									gpuCompoundPairsOut[compoundPairIdx] = b3MakeInt4(bodyIndexA,bodyIndexB,childShapeIndexA,-1);
								}//if (compoundPairIdx<maxNumCompoundPairsCapacity)
							}//
						}//fi (1)
					}//if (collidables[collidableIndexB].m_shapeType==SHAPE_COMPOUND_OF_CONVEX_HULLS)
				}//for (int b=0;b<numChildrenB;b++)	
				return;
			}//if (collidables[collidableIndexB].m_shapeType==SHAPE_COMPOUND_OF_CONVEX_HULLS)
			if ((collidables[collidableIndexA].m_shapeType!=SHAPE_CONCAVE_TRIMESH) 
				&& (collidables[collidableIndexB].m_shapeType==SHAPE_COMPOUND_OF_CONVEX_HULLS))
			{
				int numChildrenB = collidables[collidableIndexB].m_numChildShapes;
				for (int b=0;b<numChildrenB;b++)
				{
					int childShapeIndexB = collidables[collidableIndexB].m_shapeIndex+b;
					int childColIndexB = gpuChildShapes[childShapeIndexB].m_shapeIndex;
					b3Quat ornB = rigidBodies[bodyIndexB].m_quat;
					float4 posB = rigidBodies[bodyIndexB].m_pos;
					float4 childPosB = gpuChildShapes[childShapeIndexB].m_childPosition;
					b3Quat childOrnB = gpuChildShapes[childShapeIndexB].m_childOrientation;
					float4 newPosB = b3QuatRotate(ornB,childPosB)+posB;
					b3Quat newOrnB = b3QuatMul(ornB,childOrnB);

					int shapeIndexB = collidables[childColIndexB].m_shapeIndex;


					//////////////////////////////////////

					if (1)
					{
					//	int numFacesA = convexShapes[shapeIndexA].m_numFaces;
					//	float dmin = FLT_MAX;
						float4 posA = rigidBodies[bodyIndexA].m_pos;
						posA.w = 0.f;
						float4 posB = newPosB;
						posB.w = 0.f;
						float4 c0local = convexShapes[shapeIndexA].m_localCenter;
						b3Quat ornA = rigidBodies[bodyIndexA].m_quat;
						float4 c0;
						c0 = transform(&c0local, &posA, &ornA);
						float4 c1local = convexShapes[shapeIndexB].m_localCenter;
						b3Quat ornB =newOrnB;
						float4 c1;
						c1 = transform(&c1local,&posB,&ornB);
					//	const float4 DeltaC2 = c0 - c1;
						{//
							int compoundPairIdx = b3AtomicInc(numCompoundPairsOut);
							if (compoundPairIdx<maxNumCompoundPairsCapacity)
							{
								gpuCompoundPairsOut[compoundPairIdx] = b3MakeInt4(bodyIndexA,bodyIndexB,-1,childShapeIndexB);
							}//fi (compoundPairIdx<maxNumCompoundPairsCapacity)
						}//
					}//fi (1)	
				}//for (int b=0;b<numChildrenB;b++)
				return;
			}//if (collidables[collidableIndexB].m_shapeType==SHAPE_COMPOUND_OF_CONVEX_HULLS)
			return;
		}//fi ((collidables[collidableIndexA].m_shapeType==SHAPE_COMPOUND_OF_CONVEX_HULLS) ||(collidables[collidableIndexB].m_shapeType==SHAPE_COMPOUND_OF_CONVEX_HULLS))
	}//i<numPairs
}



__kernel void   processCompoundPairsKernel( __global const b3Int4* gpuCompoundPairs,
										__global const b3RigidBodyData* rigidBodies, 
										__global const b3Collidable* collidables,
										__global const b3ConvexPolyhedronData* convexShapes, 
										__global const b3AlignedObjectArray<b3Float4>& vertices,
										__global const b3AlignedObjectArray<b3Float4>& uniqueEdges,
										__global const b3AlignedObjectArray<b3GpuFace>& faces,
										__global const b3AlignedObjectArray<int>& indices,
										__global b3Aabb* aabbs,
										__global const b3GpuChildShape* gpuChildShapes,
										__global b3AlignedObjectArray<b3Float4>& gpuCompoundSepNormalsOut,
										__global b3AlignedObjectArray<int>& gpuHasCompoundSepNormalsOut,
										int numCompoundPairs,
										int i
										)
{

//	int i = get_global_id(0);
	if (i<numCompoundPairs)
	{
		int bodyIndexA = gpuCompoundPairs[i].x;
		int bodyIndexB = gpuCompoundPairs[i].y;

		int childShapeIndexA = gpuCompoundPairs[i].z;
		int childShapeIndexB = gpuCompoundPairs[i].w;
		
		int collidableIndexA = -1;
		int collidableIndexB = -1;
		
		b3Quat ornA = rigidBodies[bodyIndexA].m_quat;
		float4 posA = rigidBodies[bodyIndexA].m_pos;
		
		b3Quat ornB = rigidBodies[bodyIndexB].m_quat;
		float4 posB = rigidBodies[bodyIndexB].m_pos;
							
		if (childShapeIndexA >= 0)
		{
			collidableIndexA = gpuChildShapes[childShapeIndexA].m_shapeIndex;
			float4 childPosA = gpuChildShapes[childShapeIndexA].m_childPosition;
			b3Quat	childOrnA = gpuChildShapes[childShapeIndexA].m_childOrientation;
			float4 newPosA = b3QuatRotate(ornA,childPosA)+posA;
			b3Quat newOrnA = b3QuatMul(ornA,childOrnA);
			posA = newPosA;
			ornA = newOrnA;
		} else
		{
			collidableIndexA = rigidBodies[bodyIndexA].m_collidableIdx;
		}
		
		if (childShapeIndexB>=0)
		{
			collidableIndexB = gpuChildShapes[childShapeIndexB].m_shapeIndex;
			float4 childPosB = gpuChildShapes[childShapeIndexB].m_childPosition;
			b3Quat childOrnB = gpuChildShapes[childShapeIndexB].m_childOrientation;
			float4 newPosB = b3QuatRotate(ornB,childPosB)+posB;
			b3Quat newOrnB = b3QuatMul(ornB,childOrnB);
			posB = newPosB;
			ornB = newOrnB;
		} else
		{
			collidableIndexB = rigidBodies[bodyIndexB].m_collidableIdx;	
		}
	
		gpuHasCompoundSepNormalsOut[i] = 0;
	
		int shapeIndexA = collidables[collidableIndexA].m_shapeIndex;
		int shapeIndexB = collidables[collidableIndexB].m_shapeIndex;
	
		int shapeTypeA = collidables[collidableIndexA].m_shapeType;
		int shapeTypeB = collidables[collidableIndexB].m_shapeType;
	

		if ((shapeTypeA != SHAPE_CONVEX_HULL) || (shapeTypeB != SHAPE_CONVEX_HULL))
		{
			return;
		}

		int hasSeparatingAxis = 5;
							
	//	int numFacesA = convexShapes[shapeIndexA].m_numFaces;
		float dmin = FLT_MAX;
		posA.w = 0.f;
		posB.w = 0.f;
		float4 c0local = convexShapes[shapeIndexA].m_localCenter;
		float4 c0 = transform(&c0local, &posA, &ornA);
		float4 c1local = convexShapes[shapeIndexB].m_localCenter;
		float4 c1 = transform(&c1local,&posB,&ornB);
		const float4 DeltaC2 = c0 - c1;
		float4 sepNormal = make_float4(1,0,0,0);
//		bool sepA = findSeparatingAxis(	convexShapes[shapeIndexA], convexShapes[shapeIndexB],posA,ornA,posB,ornB,DeltaC2,vertices,uniqueEdges,faces,indices,&sepNormal,&dmin);
		bool sepA = findSeparatingAxis(	convexShapes[shapeIndexA], convexShapes[shapeIndexB],posA,ornA,posB,ornB,vertices,uniqueEdges,faces,indices,vertices,uniqueEdges,faces,indices,sepNormal);//,&dmin);
	
		hasSeparatingAxis = 4;
		if (!sepA)
		{
			hasSeparatingAxis = 0;
		} else
		{
			bool sepB = findSeparatingAxis(	convexShapes[shapeIndexB],convexShapes[shapeIndexA],posB,ornB,posA,ornA,vertices,uniqueEdges,faces,indices,vertices,uniqueEdges,faces,indices,sepNormal);//,&dmin);

			if (!sepB)
			{
				hasSeparatingAxis = 0;
			} else//(!sepB)
			{
				bool sepEE = findSeparatingAxisEdgeEdge(	&convexShapes[shapeIndexA], &convexShapes[shapeIndexB],posA,ornA,posB,ornB,DeltaC2,vertices,uniqueEdges,faces,indices,&sepNormal,&dmin);
				if (sepEE)
				{
						gpuCompoundSepNormalsOut[i] = sepNormal;//fastNormalize4(sepNormal);
						gpuHasCompoundSepNormalsOut[i] = 1;
				}//sepEE
			}//(!sepB)
		}//(!sepA)
		
		
	}
		
}


__kernel void   clipCompoundsHullHullKernel( __global const b3Int4* gpuCompoundPairs, 
																					__global const b3RigidBodyData* rigidBodies, 
																					__global const b3Collidable* collidables,
																					__global const b3ConvexPolyhedronData* convexShapes, 
																					__global const b3AlignedObjectArray<b3Float4>& vertices,
																					__global const b3AlignedObjectArray<b3Float4>& uniqueEdges,
																					__global const b3AlignedObjectArray<b3GpuFace>& faces,
																					__global const b3AlignedObjectArray<int>& indices,
																					__global const b3GpuChildShape* gpuChildShapes,
																					__global const b3AlignedObjectArray<b3Float4>& gpuCompoundSepNormalsOut,
																					__global const b3AlignedObjectArray<int>& gpuHasCompoundSepNormalsOut,
																					__global struct b3Contact4Data* globalContactsOut,
																					int* nGlobalContactsOut,
																					int numCompoundPairs, int maxContactCapacity, int i)
{

//	int i = get_global_id(0);
	int pairIndex = i;
	
	float4 worldVertsB1[64];
	float4 worldVertsB2[64];
	int capacityWorldVerts = 64;	

	float4 localContactsOut[64];
	int localContactCapacity=64;
	
	float minDist = -1e30f;
	float maxDist = 0.0f;

	if (i<numCompoundPairs)
	{

		if (gpuHasCompoundSepNormalsOut[i])
		{

			int bodyIndexA = gpuCompoundPairs[i].x;
			int bodyIndexB = gpuCompoundPairs[i].y;
			
			int childShapeIndexA = gpuCompoundPairs[i].z;
			int childShapeIndexB = gpuCompoundPairs[i].w;
			
			int collidableIndexA = -1;
			int collidableIndexB = -1;
			
			b3Quat ornA = rigidBodies[bodyIndexA].m_quat;
			float4 posA = rigidBodies[bodyIndexA].m_pos;
			
			b3Quat ornB = rigidBodies[bodyIndexB].m_quat;
			float4 posB = rigidBodies[bodyIndexB].m_pos;
								
			if (childShapeIndexA >= 0)
			{
				collidableIndexA = gpuChildShapes[childShapeIndexA].m_shapeIndex;
				float4 childPosA = gpuChildShapes[childShapeIndexA].m_childPosition;
				b3Quat childOrnA = gpuChildShapes[childShapeIndexA].m_childOrientation;
				float4 newPosA = b3QuatRotate(ornA,childPosA)+posA;
				b3Quat newOrnA = b3QuatMul(ornA,childOrnA);
				posA = newPosA;
				ornA = newOrnA;
			} else
			{
				collidableIndexA = rigidBodies[bodyIndexA].m_collidableIdx;
			}
			
			if (childShapeIndexB>=0)
			{
				collidableIndexB = gpuChildShapes[childShapeIndexB].m_shapeIndex;
				float4 childPosB = gpuChildShapes[childShapeIndexB].m_childPosition;
				b3Quat  childOrnB = gpuChildShapes[childShapeIndexB].m_childOrientation;
				float4 newPosB = b3QuatRotate(ornB,childPosB)+posB;
				b3Quat  newOrnB = b3QuatMul(ornB,childOrnB);
				posB = newPosB;
				ornB = newOrnB;
			} else
			{
				collidableIndexB = rigidBodies[bodyIndexB].m_collidableIdx;	
			}
			
			int shapeIndexA = collidables[collidableIndexA].m_shapeIndex;
			int shapeIndexB = collidables[collidableIndexB].m_shapeIndex;
		
			int numLocalContactsOut = clipHullAgainstHull(gpuCompoundSepNormalsOut[i],
														convexShapes[shapeIndexA], convexShapes[shapeIndexB],
														posA,ornA,
													  posB,ornB,
													  worldVertsB1,worldVertsB2,capacityWorldVerts,
														minDist, maxDist,
														vertices,faces,indices,
														vertices,faces,indices,
														localContactsOut,localContactCapacity);
												
		if (numLocalContactsOut>0)
		{
				float4 normal = -gpuCompoundSepNormalsOut[i];
				int nPoints = numLocalContactsOut;
				float4* pointsIn = localContactsOut;
				b3Int4 contactIdx;// = {-1,-1,-1,-1};

				contactIdx.s[0] = 0;
				contactIdx.s[1] = 1;
				contactIdx.s[2] = 2;
				contactIdx.s[3] = 3;

				int nReducedContacts = extractManifoldSequentialGlobal(pointsIn, nPoints, normal, &contactIdx);
		
				int dstIdx;
				dstIdx = b3AtomicInc( nGlobalContactsOut);
				if ((dstIdx+nReducedContacts) < maxContactCapacity)
				{
					__global struct b3Contact4Data* c = globalContactsOut+ dstIdx;
					c->m_worldNormalOnB = -normal;
					c->m_restituitionCoeffCmp = (0.f*0xffff);c->m_frictionCoeffCmp = (0.7f*0xffff);
					c->m_batchIdx = pairIndex;
					int bodyA = gpuCompoundPairs[pairIndex].x;
					int bodyB = gpuCompoundPairs[pairIndex].y;
					c->m_bodyAPtrAndSignBit = rigidBodies[bodyA].m_invMass==0?-bodyA:bodyA;
					c->m_bodyBPtrAndSignBit = rigidBodies[bodyB].m_invMass==0?-bodyB:bodyB;
					c->m_childIndexA = childShapeIndexA;
					c->m_childIndexB = childShapeIndexB;
					for (int i=0;i<nReducedContacts;i++)
					{
						c->m_worldPosB[i] = pointsIn[contactIdx.s[i]];
					}
					b3Contact4Data_setNumPoints(c,nReducedContacts);
				}
				
			}//		if (numContactsOut>0)
		}//		if (gpuHasCompoundSepNormalsOut[i])
	}//	if (i<numCompoundPairs)

}


void computeContactCompoundCompound(int pairIndex,
																int bodyIndexA, int bodyIndexB, 
																int collidableIndexA, int collidableIndexB, 
																const b3RigidBodyData* rigidBodies, 
																const b3Collidable* collidables,
																const b3ConvexPolyhedronData* convexShapes,
																const b3GpuChildShape* cpuChildShapes,
																const b3AlignedObjectArray<b3Aabb>& hostAabbsWorldSpace,
																const b3AlignedObjectArray<b3Aabb>& hostAabbsLocalSpace,

																const b3AlignedObjectArray<b3Vector3>& convexVertices,
																const b3AlignedObjectArray<b3Vector3>& hostUniqueEdges,
																const b3AlignedObjectArray<int>& convexIndices,
																const b3AlignedObjectArray<b3GpuFace>& faces,
																
																b3Contact4* globalContactsOut,
																int& nGlobalContactsOut,
																int maxContactCapacity,
																b3AlignedObjectArray<b3QuantizedBvhNode>&	treeNodesCPU,
																b3AlignedObjectArray<b3BvhSubtreeInfo>&	subTreesCPU,
																b3AlignedObjectArray<b3BvhInfo>&	bvhInfoCPU
																)
{

	int shapeTypeB = collidables[collidableIndexB].m_shapeType;
	b3Assert(shapeTypeB == SHAPE_COMPOUND_OF_CONVEX_HULLS);

	b3AlignedObjectArray<b3Int4> cpuCompoundPairsOut;
	int numCompoundPairsOut=0;
	int maxNumCompoundPairsCapacity = 8192;//1024;
	cpuCompoundPairsOut.resize(maxNumCompoundPairsCapacity);

	// work-in-progress
	findCompoundPairsKernel( 
							pairIndex,
							bodyIndexA,bodyIndexB,
							collidableIndexA,collidableIndexB,
							rigidBodies, 
							collidables,
							convexShapes, 
							convexVertices,
							hostAabbsWorldSpace,
							hostAabbsLocalSpace,
							cpuChildShapes,
							&cpuCompoundPairsOut[0],
							&numCompoundPairsOut,
							maxNumCompoundPairsCapacity	,
							treeNodesCPU,
							subTreesCPU,
							bvhInfoCPU
							);

	printf("maxNumAabbChecks=%d\n",maxNumAabbChecks);
	if (numCompoundPairsOut>maxNumCompoundPairsCapacity)
	{
		b3Error("numCompoundPairsOut exceeded maxNumCompoundPairsCapacity (%d)\n",maxNumCompoundPairsCapacity);
		numCompoundPairsOut=maxNumCompoundPairsCapacity;
	}
	b3AlignedObjectArray<b3Float4> cpuCompoundSepNormalsOut;
	b3AlignedObjectArray<int> cpuHasCompoundSepNormalsOut;
	cpuCompoundSepNormalsOut.resize(numCompoundPairsOut);
	cpuHasCompoundSepNormalsOut.resize(numCompoundPairsOut);

	for (int i=0;i<numCompoundPairsOut;i++)
	{

		processCompoundPairsKernel(&cpuCompoundPairsOut[0],rigidBodies,collidables,convexShapes,convexVertices,hostUniqueEdges,faces,convexIndices,0,cpuChildShapes,
			cpuCompoundSepNormalsOut,cpuHasCompoundSepNormalsOut,numCompoundPairsOut,i);
	}

	for (int i=0;i<numCompoundPairsOut;i++)
	{
		clipCompoundsHullHullKernel(&cpuCompoundPairsOut[0],rigidBodies,collidables,convexShapes,convexVertices,hostUniqueEdges,faces,convexIndices,cpuChildShapes,
			cpuCompoundSepNormalsOut,cpuHasCompoundSepNormalsOut,globalContactsOut,&nGlobalContactsOut,numCompoundPairsOut,maxContactCapacity,i);
	}
		/*
		int childColIndexA = gpuChildShapes[childShapeIndexA].m_shapeIndex;

					float4 posA = rigidBodies[bodyIndexA].m_pos;
					b3Quat ornA = rigidBodies[bodyIndexA].m_quat;
					float4 childPosA = gpuChildShapes[childShapeIndexA].m_childPosition;
					b3Quat childOrnA = gpuChildShapes[childShapeIndexA].m_childOrientation;
					float4 newPosA = b3QuatRotate(ornA,childPosA)+posA;
					b3Quat newOrnA = b3QuatMul(ornA,childOrnA);

					int shapeIndexA = collidables[childColIndexA].m_shapeIndex;


			bool foundSepAxis = findSeparatingAxis(hullA,hullB,
							posA,
							ornA,
							posB,
							ornB,

							convexVertices,uniqueEdges,faces,convexIndices,
							convexVertices,uniqueEdges,faces,convexIndices,
							
							sepNormalWorldSpace
							);
							*/

	
	/*
	if (foundSepAxis)
	{
		
		
		contactIndex = clipHullHullSingle(
			bodyIndexA, bodyIndexB,
						   posA,ornA,
						   posB,ornB,
			collidableIndexA, collidableIndexB,
			&rigidBodies, 
			&globalContactsOut,
			nGlobalContactsOut,
			
			convexShapes,
			convexShapes,
	
			convexVertices, 
			uniqueEdges, 
			faces,
			convexIndices,
	
			convexVertices,
			uniqueEdges,
			faces,
			convexIndices,

			collidables,
			collidables,
			sepNormalWorldSpace,
			maxContactCapacity);
			
	}
	*/

//	return contactIndex;
	
	/*

	int numChildrenB = collidables[collidableIndexB].m_numChildShapes;
	for (int c=0;c<numChildrenB;c++)
	{
		int childShapeIndexB = collidables[collidableIndexB].m_shapeIndex+c;
		int childColIndexB = cpuChildShapes[childShapeIndexB].m_shapeIndex;

		float4 rootPosB = rigidBodies[bodyIndexB].m_pos;
		b3Quaternion rootOrnB = rigidBodies[bodyIndexB].m_quat;
		b3Vector3 childPosB = cpuChildShapes[childShapeIndexB].m_childPosition;
		b3Quaternion childOrnB = cpuChildShapes[childShapeIndexB].m_childOrientation;
		float4  posB = b3QuatRotate(rootOrnB,childPosB)+rootPosB;
		b3Quaternion ornB = b3QuatMul(rootOrnB,childOrnB);//b3QuatMul(ornB,childOrnB);

		int shapeIndexB = collidables[childColIndexB].m_shapeIndex;

		const b3ConvexPolyhedronData* hullB = &convexShapes[shapeIndexB];

	}
	*/

}

void computeContactPlaneCompound(int pairIndex,
																int bodyIndexA, int bodyIndexB, 
																int collidableIndexA, int collidableIndexB, 
																const b3RigidBodyData* rigidBodies, 
																const b3Collidable* collidables,
																const b3ConvexPolyhedronData* convexShapes,
																const b3GpuChildShape* cpuChildShapes,
																const b3Vector3* convexVertices,
																const int* convexIndices,
																const b3GpuFace* faces,
																
																b3Contact4* globalContactsOut,
																int& nGlobalContactsOut,
																int maxContactCapacity)
{

	int shapeTypeB = collidables[collidableIndexB].m_shapeType;
	b3Assert(shapeTypeB == SHAPE_COMPOUND_OF_CONVEX_HULLS);


	int numChildrenB = collidables[collidableIndexB].m_numChildShapes;
	for (int c=0;c<numChildrenB;c++)
	{
		int childShapeIndexB = collidables[collidableIndexB].m_shapeIndex+c;
		int childColIndexB = cpuChildShapes[childShapeIndexB].m_shapeIndex;

		float4 rootPosB = rigidBodies[bodyIndexB].m_pos;
		b3Quaternion rootOrnB = rigidBodies[bodyIndexB].m_quat;
		b3Vector3 childPosB = cpuChildShapes[childShapeIndexB].m_childPosition;
		b3Quaternion childOrnB = cpuChildShapes[childShapeIndexB].m_childOrientation;
		float4  posB = b3QuatRotate(rootOrnB,childPosB)+rootPosB;
		b3Quaternion ornB = rootOrnB*childOrnB;//b3QuatMul(ornB,childOrnB);

		int shapeIndexB = collidables[childColIndexB].m_shapeIndex;

		const b3ConvexPolyhedronData* hullB = &convexShapes[shapeIndexB];
	
		
		b3Vector3 posA = rigidBodies[bodyIndexA].m_pos;
		b3Quaternion ornA = rigidBodies[bodyIndexA].m_quat;

	//	int numContactsOut = 0;
	//	int numWorldVertsB1= 0;

		b3Vector3 planeEq = faces[collidables[collidableIndexA].m_shapeIndex].m_plane;
		b3Vector3 planeNormal=b3MakeVector3(planeEq.x,planeEq.y,planeEq.z);
		b3Vector3 planeNormalWorld = b3QuatRotate(ornA,planeNormal);
		float planeConstant = planeEq.w;
		b3Transform convexWorldTransform;
		convexWorldTransform.setIdentity();
		convexWorldTransform.setOrigin(posB);
		convexWorldTransform.setRotation(ornB);
		b3Transform planeTransform;
		planeTransform.setIdentity();
		planeTransform.setOrigin(posA);
		planeTransform.setRotation(ornA);

		b3Transform planeInConvex;
		planeInConvex= convexWorldTransform.inverse() * planeTransform;
		b3Transform convexInPlane;
		convexInPlane = planeTransform.inverse() * convexWorldTransform;
	
		b3Vector3 planeNormalInConvex = planeInConvex.getBasis()*-planeNormal;
		float maxDot = -1e30;
		int hitVertex=-1;
		b3Vector3 hitVtx;

	#define MAX_PLANE_CONVEX_POINTS 64

		b3Vector3 contactPoints[MAX_PLANE_CONVEX_POINTS];
		int numPoints = 0;

		b3Int4 contactIdx;
		contactIdx.s[0] = 0;
		contactIdx.s[1] = 1;
		contactIdx.s[2] = 2;
		contactIdx.s[3] = 3;
	
		for (int i=0;i<hullB->m_numVertices;i++)
		{
			b3Vector3 vtx = convexVertices[hullB->m_vertexOffset+i];
			float curDot = vtx.dot(planeNormalInConvex);


			if (curDot>maxDot)
			{
				hitVertex=i;
				maxDot=curDot;
				hitVtx = vtx;
				//make sure the deepest points is always included
				if (numPoints==MAX_PLANE_CONVEX_POINTS)
					numPoints--;
			}

			if (numPoints<MAX_PLANE_CONVEX_POINTS)
			{
				b3Vector3 vtxWorld = convexWorldTransform*vtx;
				b3Vector3 vtxInPlane = planeTransform.inverse()*vtxWorld;
				float dist = planeNormal.dot(vtxInPlane)-planeConstant;
				if (dist<0.f)
				{
					vtxWorld.w = dist;
					contactPoints[numPoints] = vtxWorld;
					numPoints++;
				}
			}

		}

		int numReducedPoints  = 0;

		numReducedPoints = numPoints;
	
		if (numPoints>4)
		{
			numReducedPoints = extractManifoldSequentialGlobal( contactPoints, numPoints, planeNormalInConvex, &contactIdx);
		}
		int dstIdx;
	//    dstIdx = nGlobalContactsOut++;//AppendInc( nGlobalContactsOut, dstIdx );
		
		if (numReducedPoints>0)
		{
			if (nGlobalContactsOut < maxContactCapacity)
			{
				dstIdx=nGlobalContactsOut;
				nGlobalContactsOut++;

				b3Contact4* c = &globalContactsOut[dstIdx];
				c->m_worldNormalOnB = -planeNormalWorld;
				c->setFrictionCoeff(0.7);
				c->setRestituitionCoeff(0.f);

				c->m_batchIdx = pairIndex;
				c->m_bodyAPtrAndSignBit = rigidBodies[bodyIndexA].m_invMass==0?-bodyIndexA:bodyIndexA;
				c->m_bodyBPtrAndSignBit = rigidBodies[bodyIndexB].m_invMass==0?-bodyIndexB:bodyIndexB;
				for (int i=0;i<numReducedPoints;i++)
				{
					b3Vector3 pOnB1 = contactPoints[contactIdx.s[i]];
					c->m_worldPosB[i] = pOnB1;
				}
				c->m_worldNormalOnB.w = (b3Scalar)numReducedPoints;
			}//if (dstIdx < numPairs)
		}	
		
	}

	
}





void	computeContactSphereConvex(int pairIndex,
																int bodyIndexA, int bodyIndexB, 
																int collidableIndexA, int collidableIndexB, 
																const b3RigidBodyData* rigidBodies, 
																const b3Collidable* collidables,
																const b3ConvexPolyhedronData* convexShapes,
																const b3Vector3* convexVertices,
																const int* convexIndices,
																const b3GpuFace* faces,
																b3Contact4* globalContactsOut,
																int& nGlobalContactsOut,
																int maxContactCapacity)
{

	float radius = collidables[collidableIndexA].m_radius;
	float4 spherePos1 = rigidBodies[bodyIndexA].m_pos;
	b3Quaternion sphereOrn = rigidBodies[bodyIndexA].m_quat;



	float4 pos = rigidBodies[bodyIndexB].m_pos;
	

	b3Quaternion quat = rigidBodies[bodyIndexB].m_quat;

	b3Transform tr;
	tr.setIdentity();
	tr.setOrigin(pos);
	tr.setRotation(quat);
	b3Transform trInv = tr.inverse();

	float4 spherePos = trInv(spherePos1);

	int collidableIndex = rigidBodies[bodyIndexB].m_collidableIdx;
	int shapeIndex = collidables[collidableIndex].m_shapeIndex;
	int numFaces = convexShapes[shapeIndex].m_numFaces;
	float4 closestPnt = b3MakeVector3(0, 0, 0, 0);
//	float4 hitNormalWorld = b3MakeVector3(0, 0, 0, 0);
	float minDist = -1000000.f; // TODO: What is the largest/smallest float?
	bool bCollide = true;
	int region = -1;
	float4 localHitNormal;
	for ( int f = 0; f < numFaces; f++ )
	{
		b3GpuFace face = faces[convexShapes[shapeIndex].m_faceOffset+f];
		float4 planeEqn;
		float4 localPlaneNormal = b3MakeVector3(face.m_plane.x,face.m_plane.y,face.m_plane.z,0.f);
		float4 n1 = localPlaneNormal;//quatRotate(quat,localPlaneNormal);
		planeEqn = n1;
		planeEqn[3] = face.m_plane.w;

		float4 pntReturn;
		float dist = signedDistanceFromPointToPlane(spherePos, planeEqn, &pntReturn);

		if ( dist > radius)
		{
			bCollide = false;
			break;
		}

		if ( dist > 0 )
		{
			//might hit an edge or vertex
			b3Vector3 out;

			bool isInPoly = IsPointInPolygon(spherePos,
					&face,
					&convexVertices[convexShapes[shapeIndex].m_vertexOffset],
					convexIndices,
                    &out);
			if (isInPoly)
			{
				if (dist>minDist)
				{
					minDist = dist;
					closestPnt = pntReturn;
					localHitNormal = planeEqn;
					region=1;
				}
			} else
			{
				b3Vector3 tmp = spherePos-out;
				b3Scalar l2 = tmp.length2();
				if (l2<radius*radius)
				{
					dist  = b3Sqrt(l2);
					if (dist>minDist)
					{
						minDist = dist;
						closestPnt = out;
						localHitNormal = tmp/dist;
						region=2;
					}
					
				} else
				{
					bCollide = false;
					break;
				}
			}
		}
		else
		{
			if ( dist > minDist )
			{
				minDist = dist;
				closestPnt = pntReturn;
				localHitNormal = planeEqn;
				region=3;
			}
		}
	}
	static int numChecks = 0;
	numChecks++;

	if (bCollide && minDist > -10000)
	{
		
		float4 normalOnSurfaceB1 = tr.getBasis()*localHitNormal;//-hitNormalWorld;
		float4 pOnB1 = tr(closestPnt);
		//printf("dist ,%f,",minDist);
		float actualDepth = minDist-radius;
		if (actualDepth<0)
		{
		//printf("actualDepth = ,%f,", actualDepth);
		//printf("normalOnSurfaceB1 = ,%f,%f,%f,", normalOnSurfaceB1.x,normalOnSurfaceB1.y,normalOnSurfaceB1.z);
		//printf("region=,%d,\n", region);
		pOnB1[3] = actualDepth;

		int dstIdx;
//    dstIdx = nGlobalContactsOut++;//AppendInc( nGlobalContactsOut, dstIdx );
		
		if (nGlobalContactsOut < maxContactCapacity)
		{
			dstIdx=nGlobalContactsOut;
			nGlobalContactsOut++;

			b3Contact4* c = &globalContactsOut[dstIdx];
			c->m_worldNormalOnB = normalOnSurfaceB1;
			c->setFrictionCoeff(0.7);
			c->setRestituitionCoeff(0.f);

			c->m_batchIdx = pairIndex;
			c->m_bodyAPtrAndSignBit = rigidBodies[bodyIndexA].m_invMass==0?-bodyIndexA:bodyIndexA;
			c->m_bodyBPtrAndSignBit = rigidBodies[bodyIndexB].m_invMass==0?-bodyIndexB:bodyIndexB;
			c->m_worldPosB[0] = pOnB1;
			int numPoints = 1;
			c->m_worldNormalOnB.w = (b3Scalar)numPoints;
		}//if (dstIdx < numPairs)
		}
	}//if (hasCollision)
	
}



#include "b3GjkPairDetector.h"
#include "b3GjkEpa.h"
#include "b3VoronoiSimplexSolver.h"

int computeContactConvexConvex( b3AlignedObjectArray<b3Int4>& pairs,
																int pairIndex,
																int bodyIndexA, int bodyIndexB, 
																int collidableIndexA, int collidableIndexB, 
																const b3AlignedObjectArray<b3RigidBodyData>& rigidBodies, 
																const b3AlignedObjectArray<b3Collidable>& collidables,
																const b3AlignedObjectArray<b3ConvexPolyhedronData>& convexShapes,
																const b3AlignedObjectArray<b3Vector3>& convexVertices,
																const b3AlignedObjectArray<b3Vector3>& uniqueEdges,
																const b3AlignedObjectArray<int>& convexIndices,
																const b3AlignedObjectArray<b3GpuFace>& faces,
																b3AlignedObjectArray<b3Contact4>& globalContactsOut,
																int& nGlobalContactsOut,
																int maxContactCapacity,
																b3AlignedObjectArray<int>&hostHasSepAxis,
																b3AlignedObjectArray<b3Vector3>&hostSepAxis


							   //,const b3AlignedObjectArray<b3Contact4>& oldContacts
																)
{	
	int contactIndex = -1;
	b3VoronoiSimplexSolver simplexSolver;
	b3GjkEpaSolver2 epaSolver;
			
	b3GjkPairDetector gjkDetector(&simplexSolver,&epaSolver);

	b3Transform transA;
	transA.setOrigin(rigidBodies[bodyIndexA].m_pos);
	transA.setRotation(rigidBodies[bodyIndexA].m_quat);

	b3Transform transB;
	transB.setOrigin(rigidBodies[bodyIndexB].m_pos);
	transB.setRotation(rigidBodies[bodyIndexB].m_quat);
	float maximumDistanceSquared = 1e30f;
					
	b3Vector3 resultPointOnBWorld;
	b3Vector3 sepAxis2=b3MakeVector3(0,1,0);
	b3Scalar distance2 = 1e30f;
	
	int shapeIndexA = collidables[collidableIndexA].m_shapeIndex;
	int shapeIndexB = collidables[collidableIndexB].m_shapeIndex;

	//int sz = sizeof(b3Contact4);

	bool result2 = getClosestPoints(&gjkDetector, transA, transB,
		convexShapes[shapeIndexA], convexShapes[shapeIndexB],
		convexVertices,convexVertices,
		maximumDistanceSquared,
		sepAxis2,
		distance2,
		resultPointOnBWorld);
	
	
	if (result2)
	{
		if (nGlobalContactsOut<maxContactCapacity)
		{
			contactIndex = nGlobalContactsOut;
			globalContactsOut.expand();
			b3Contact4& newContact = globalContactsOut.at(nGlobalContactsOut);
			newContact.m_batchIdx = 0;//i;
			newContact.m_bodyAPtrAndSignBit = (rigidBodies.at(bodyIndexA).m_invMass==0)? -bodyIndexA:bodyIndexA;
			newContact.m_bodyBPtrAndSignBit = (rigidBodies.at(bodyIndexB).m_invMass==0)? -bodyIndexB:bodyIndexB;

			newContact.m_frictionCoeffCmp = 45874;
			newContact.m_restituitionCoeffCmp = 0;
					
			
			int numPoints = 0;
			if (pairs[pairIndex].z>=0)
			{
				//printf("add existing points?\n");
				//refresh
				
				int numOldPoints = 0;//oldContacts[pairs[pairIndex].z].getNPoints();
				if (numOldPoints)
				{
					//newContact = oldContacts[pairs[pairIndex].z];
#ifdef PERSISTENT_CONTACTS_HOST
					b3ContactCache::refreshContactPoints(transA,transB,newContact);
#endif //PERSISTENT_CONTACTS_HOST
				}
				numPoints = b3Contact4Data_getNumPoints(&newContact);

			}

			/*
			int insertIndex = m_manifoldPtr->getCacheEntry(newPt);
				if (insertIndex >= 0)
				{
					//const btManifoldPoint& oldPoint = m_manifoldPtr->getContactPoint(insertIndex);
					m_manifoldPtr->replaceContactPoint(newPt,insertIndex);
				} else
				{
					insertIndex = m_manifoldPtr->addManifoldPoint(newPt);
				}
			*/
			
			int p=numPoints;
			if (numPoints<4)
			{
				numPoints++;
			} else
			{
				p=3;
			}

			{
				resultPointOnBWorld.w = distance2;
				newContact.m_worldPosB[p] = resultPointOnBWorld;
#ifdef PERSISTENT_CONTACTS_HOST
				b3Vector3 resultPointOnAWorld = resultPointOnBWorld+distance2*sepAxis2;
				newContact.m_localPosA[p] = transA.inverse()*resultPointOnAWorld;
				newContact.m_localPosB[p] = transB.inverse()*resultPointOnBWorld;
#endif
				newContact.m_worldNormalOnB = sepAxis2;
				
				
				hostHasSepAxis[pairIndex] = 1;
				hostSepAxis[pairIndex] =sepAxis2;
				//printf("sepAxis[%d]=%f,%f,%f,%f\n",pairIndex,sepAxis2.x,sepAxis2.y,sepAxis2.z,sepAxis2.w);
			}
			//printf("bodyIndexA %d,bodyIndexB %d,normal=%f,%f,%f numPoints %d\n",bodyIndexA,bodyIndexB,normalOnSurfaceB.x,normalOnSurfaceB.y,normalOnSurfaceB.z,numPoints);
			newContact.m_worldNormalOnB.w = (b3Scalar)numPoints;

			
			nGlobalContactsOut++;
		} else
		{
			b3Error("Error: exceeding contact capacity (%d/%d)\n", nGlobalContactsOut,maxContactCapacity);
		}
	}
	
	return contactIndex;
}

int computeContactConvexConvex2(
																int pairIndex,
																int bodyIndexA, int bodyIndexB, 
																int collidableIndexA, int collidableIndexB, 
																const b3AlignedObjectArray<b3RigidBodyData>& rigidBodies, 
																const b3AlignedObjectArray<b3Collidable>& collidables,
																const b3AlignedObjectArray<b3ConvexPolyhedronData>& convexShapes,
																const b3AlignedObjectArray<b3Vector3>& convexVertices,
																const b3AlignedObjectArray<b3Vector3>& uniqueEdges,
																const b3AlignedObjectArray<int>& convexIndices,
																const b3AlignedObjectArray<b3GpuFace>& faces,
																b3AlignedObjectArray<b3Contact4>& globalContactsOut,
																int& nGlobalContactsOut,
																int maxContactCapacity,
																const b3AlignedObjectArray<b3Contact4>& oldContacts
																)
{
	int contactIndex = -1;
	b3Vector3 posA = rigidBodies[bodyIndexA].m_pos;
	b3Quaternion ornA = rigidBodies[bodyIndexA].m_quat;
	b3Vector3 posB = rigidBodies[bodyIndexB].m_pos;
	b3Quaternion ornB = rigidBodies[bodyIndexB].m_quat;
	

	b3ConvexPolyhedronData hullA, hullB;
    
	b3Vector3 sepNormalWorldSpace;

	

    b3Collidable colA = collidables[collidableIndexA];
    hullA = convexShapes[colA.m_shapeIndex];
    //printf("numvertsA = %d\n",hullA.m_numVertices);
    
    
    b3Collidable colB = collidables[collidableIndexB];
    hullB = convexShapes[colB.m_shapeIndex];
    //printf("numvertsB = %d\n",hullB.m_numVertices);

//	int contactCapacity = MAX_VERTS;
	//int numContactsOut=0;


#ifdef _WIN32
	b3Assert(_finite(rigidBodies[bodyIndexA].m_pos.x));
	b3Assert(_finite(rigidBodies[bodyIndexB].m_pos.x));
#endif
	
		bool foundSepAxis = findSeparatingAxis(hullA,hullB,
							posA,
							ornA,
							posB,
							ornB,

							convexVertices,uniqueEdges,faces,convexIndices,
							convexVertices,uniqueEdges,faces,convexIndices,
							
							sepNormalWorldSpace
							);

	
	if (foundSepAxis)
	{
		
		
		contactIndex = clipHullHullSingle(
			bodyIndexA, bodyIndexB,
						   posA,ornA,
						   posB,ornB,
			collidableIndexA, collidableIndexB,
			&rigidBodies, 
			&globalContactsOut,
			nGlobalContactsOut,
			
			convexShapes,
			convexShapes,
	
			convexVertices, 
			uniqueEdges, 
			faces,
			convexIndices,
	
			convexVertices,
			uniqueEdges,
			faces,
			convexIndices,

			collidables,
			collidables,
			sepNormalWorldSpace,
			maxContactCapacity);
			
	}

	return contactIndex;
}





																
																
void GpuSatCollision::computeConvexConvexContactsGPUSAT( b3OpenCLArray<b3Int4>* pairs, int nPairs,
			const b3OpenCLArray<b3RigidBodyData>* bodyBuf,
			b3OpenCLArray<b3Contact4>* contactOut, int& nContacts,
			const b3OpenCLArray<b3Contact4>* oldContacts,
			int maxContactCapacity,
			int compoundPairCapacity,
			const b3OpenCLArray<b3ConvexPolyhedronData>& convexData,
			const b3OpenCLArray<b3Vector3>& gpuVertices,
			const b3OpenCLArray<b3Vector3>& gpuUniqueEdges,
			const b3OpenCLArray<b3GpuFace>& gpuFaces,
			const b3OpenCLArray<int>& gpuIndices,
			const b3OpenCLArray<b3Collidable>& gpuCollidables,
			const b3OpenCLArray<b3GpuChildShape>& gpuChildShapes,

			const b3OpenCLArray<b3Aabb>& clAabbsWorldSpace,
			const b3OpenCLArray<b3Aabb>& clAabbsLocalSpace,

            b3OpenCLArray<b3Vector3>& worldVertsB1GPU,
            b3OpenCLArray<b3Int4>& clippingFacesOutGPU,
            b3OpenCLArray<b3Vector3>& worldNormalsAGPU,
            b3OpenCLArray<b3Vector3>& worldVertsA1GPU,
            b3OpenCLArray<b3Vector3>& worldVertsB2GPU,    
			b3AlignedObjectArray<class b3OptimizedBvh*>& bvhDataUnused,
			b3OpenCLArray<b3QuantizedBvhNode>*	treeNodesGPU,
			b3OpenCLArray<b3BvhSubtreeInfo>*	subTreesGPU,
			b3OpenCLArray<b3BvhInfo>*	bvhInfo,

			int numObjects,
			int maxTriConvexPairCapacity,
			b3OpenCLArray<b3Int4>& triangleConvexPairsOut,
			int& numTriConvexPairsOut
			)
{
	myframecount++;

	if (!nPairs)
		return;

#ifdef CHECK_ON_HOST


	b3AlignedObjectArray<b3QuantizedBvhNode>	treeNodesCPU;
	treeNodesGPU->copyToHost(treeNodesCPU);

	b3AlignedObjectArray<b3BvhSubtreeInfo>	subTreesCPU;
	subTreesGPU->copyToHost(subTreesCPU);

	b3AlignedObjectArray<b3BvhInfo>	bvhInfoCPU;
	bvhInfo->copyToHost(bvhInfoCPU);

	b3AlignedObjectArray<b3Aabb> hostAabbsWorldSpace;
	clAabbsWorldSpace.copyToHost(hostAabbsWorldSpace);

	b3AlignedObjectArray<b3Aabb> hostAabbsLocalSpace;
	clAabbsLocalSpace.copyToHost(hostAabbsLocalSpace);

	b3AlignedObjectArray<b3Int4> hostPairs;
	pairs->copyToHost(hostPairs);

	b3AlignedObjectArray<b3RigidBodyData> hostBodyBuf;
	bodyBuf->copyToHost(hostBodyBuf);

	

	b3AlignedObjectArray<b3ConvexPolyhedronData> hostConvexData;
	convexData.copyToHost(hostConvexData);

	b3AlignedObjectArray<b3Vector3> hostVertices;
	gpuVertices.copyToHost(hostVertices);

	b3AlignedObjectArray<b3Vector3> hostUniqueEdges;
	gpuUniqueEdges.copyToHost(hostUniqueEdges);
	b3AlignedObjectArray<b3GpuFace> hostFaces;
	gpuFaces.copyToHost(hostFaces);
	b3AlignedObjectArray<int> hostIndices;
	gpuIndices.copyToHost(hostIndices);
	b3AlignedObjectArray<b3Collidable> hostCollidables;
	gpuCollidables.copyToHost(hostCollidables);
	
	b3AlignedObjectArray<b3GpuChildShape> cpuChildShapes;
	gpuChildShapes.copyToHost(cpuChildShapes);
	

	b3AlignedObjectArray<b3Int4> hostTriangleConvexPairs;

	b3AlignedObjectArray<b3Contact4> hostContacts;
	if (nContacts)
	{
		contactOut->copyToHost(hostContacts);
	}

	b3AlignedObjectArray<b3Contact4> oldHostContacts;
	
	if (oldContacts->size())
	{
		oldContacts->copyToHost(oldHostContacts);
	}


	hostContacts.resize(maxContactCapacity);

	for (int i=0;i<nPairs;i++)
	{
		int bodyIndexA = hostPairs[i].x;
		int bodyIndexB = hostPairs[i].y;
		int collidableIndexA = hostBodyBuf[bodyIndexA].m_collidableIdx;
		int collidableIndexB = hostBodyBuf[bodyIndexB].m_collidableIdx;

		if (hostCollidables[collidableIndexA].m_shapeType == SHAPE_SPHERE &&
			hostCollidables[collidableIndexB].m_shapeType == SHAPE_CONVEX_HULL)
		{
			computeContactSphereConvex(i,bodyIndexA,bodyIndexB,collidableIndexA,collidableIndexB,&hostBodyBuf[0],
				&hostCollidables[0],&hostConvexData[0],&hostVertices[0],&hostIndices[0],&hostFaces[0],&hostContacts[0],nContacts,maxContactCapacity);
		}

		if (hostCollidables[collidableIndexA].m_shapeType == SHAPE_CONVEX_HULL &&
			hostCollidables[collidableIndexB].m_shapeType == SHAPE_SPHERE)
		{
			computeContactSphereConvex(i,bodyIndexB,bodyIndexA,collidableIndexB,collidableIndexA,&hostBodyBuf[0],
				&hostCollidables[0],&hostConvexData[0],&hostVertices[0],&hostIndices[0],&hostFaces[0],&hostContacts[0],nContacts,maxContactCapacity);
			//printf("convex-sphere\n");
			
		}

		if (hostCollidables[collidableIndexA].m_shapeType == SHAPE_CONVEX_HULL &&
			hostCollidables[collidableIndexB].m_shapeType == SHAPE_PLANE)
		{
			computeContactPlaneConvex(i,bodyIndexB,bodyIndexA,collidableIndexB,collidableIndexA,&hostBodyBuf[0],
			&hostCollidables[0],&hostConvexData[0],&hostVertices[0],&hostIndices[0],&hostFaces[0],&hostContacts[0],nContacts,maxContactCapacity);
//			printf("convex-plane\n");
			
		}

		if (hostCollidables[collidableIndexA].m_shapeType == SHAPE_PLANE &&
			hostCollidables[collidableIndexB].m_shapeType == SHAPE_CONVEX_HULL)
		{
			computeContactPlaneConvex(i,bodyIndexA,bodyIndexB,collidableIndexA,collidableIndexB,&hostBodyBuf[0],
			&hostCollidables[0],&hostConvexData[0],&hostVertices[0],&hostIndices[0],&hostFaces[0],&hostContacts[0],nContacts,maxContactCapacity);
//			printf("plane-convex\n");
			
		}

			if (hostCollidables[collidableIndexA].m_shapeType == SHAPE_COMPOUND_OF_CONVEX_HULLS &&
			hostCollidables[collidableIndexB].m_shapeType == SHAPE_COMPOUND_OF_CONVEX_HULLS)
		{
			computeContactCompoundCompound(i,bodyIndexB,bodyIndexA,collidableIndexB,collidableIndexA,&hostBodyBuf[0],
			&hostCollidables[0],&hostConvexData[0],&cpuChildShapes[0], hostAabbsWorldSpace,hostAabbsLocalSpace,hostVertices,hostUniqueEdges,hostIndices,hostFaces,&hostContacts[0],
			nContacts,maxContactCapacity,treeNodesCPU,subTreesCPU,bvhInfoCPU);	
//			printf("convex-plane\n");
			
		}


				if (hostCollidables[collidableIndexA].m_shapeType == SHAPE_COMPOUND_OF_CONVEX_HULLS &&
			hostCollidables[collidableIndexB].m_shapeType == SHAPE_PLANE)
		{
			computeContactPlaneCompound(i,bodyIndexB,bodyIndexA,collidableIndexB,collidableIndexA,&hostBodyBuf[0],
			&hostCollidables[0],&hostConvexData[0],&cpuChildShapes[0], &hostVertices[0],&hostIndices[0],&hostFaces[0],&hostContacts[0],nContacts,maxContactCapacity);
//			printf("convex-plane\n");
			
		}

		if (hostCollidables[collidableIndexA].m_shapeType == SHAPE_PLANE &&
			hostCollidables[collidableIndexB].m_shapeType == SHAPE_COMPOUND_OF_CONVEX_HULLS)
		{
			computeContactPlaneCompound(i,bodyIndexA,bodyIndexB,collidableIndexA,collidableIndexB,&hostBodyBuf[0],
			&hostCollidables[0],&hostConvexData[0],&cpuChildShapes[0],&hostVertices[0],&hostIndices[0],&hostFaces[0],&hostContacts[0],nContacts,maxContactCapacity);
//			printf("plane-convex\n");
			
		}

		if (hostCollidables[collidableIndexA].m_shapeType == SHAPE_CONVEX_HULL &&
			hostCollidables[collidableIndexB].m_shapeType == SHAPE_CONVEX_HULL)
		{
			//printf("hostPairs[i].z=%d\n",hostPairs[i].z);
			//int contactIndex = computeContactConvexConvex2(i,bodyIndexA,bodyIndexB,collidableIndexA,collidableIndexB,hostBodyBuf, hostCollidables,hostConvexData,hostVertices,hostUniqueEdges,hostIndices,hostFaces,hostContacts,nContacts,maxContactCapacity,oldHostContacts);
			int contactIndex = computeContactConvexConvex(hostPairs,i,bodyIndexA,bodyIndexB,collidableIndexA,collidableIndexB,hostBodyBuf,hostCollidables,hostConvexData,hostVertices,hostUniqueEdges,hostIndices,hostFaces,hostContacts,nContacts,maxContactCapacity,oldHostContacts);


			if (contactIndex>=0)
			{
//				printf("convex convex contactIndex = %d\n",contactIndex);
				hostPairs[i].z = contactIndex;
			}
//			printf("plane-convex\n");
			
		}


	}

	if (hostPairs.size())
	{
		pairs->copyFromHost(hostPairs);
	}

	hostContacts.resize(nContacts);
	if (nContacts)
		{
			
			contactOut->copyFromHost(hostContacts);
		} else
	{
		contactOut->resize(0);
		}

		m_totalContactsOut.copyFromHostPointer(&nContacts,1,0,true);
		//printf("(HOST) nContacts = %d\n",nContacts);

#else

	{
		if (nPairs)
		{
			m_totalContactsOut.copyFromHostPointer(&nContacts,1,0,true);

			B3_PROFILE("primitiveContactsKernel");
			b3BufferInfoCL bInfo[] = {
				b3BufferInfoCL( pairs->getBufferCL(), true ), 
				b3BufferInfoCL( bodyBuf->getBufferCL(),true), 
				b3BufferInfoCL( gpuCollidables.getBufferCL(),true), 
				b3BufferInfoCL( convexData.getBufferCL(),true),
				b3BufferInfoCL( gpuVertices.getBufferCL(),true),
				b3BufferInfoCL( gpuUniqueEdges.getBufferCL(),true),
				b3BufferInfoCL( gpuFaces.getBufferCL(),true),
				b3BufferInfoCL( gpuIndices.getBufferCL(),true),
				b3BufferInfoCL( contactOut->getBufferCL()),
				b3BufferInfoCL( m_totalContactsOut.getBufferCL())	
			};
			
			b3LauncherCL launcher(m_queue, m_primitiveContactsKernel,"m_primitiveContactsKernel");
			launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(b3BufferInfoCL) );
			launcher.setConst( nPairs  );
			launcher.setConst(maxContactCapacity);
			int num = nPairs;
			launcher.launch1D( num);
			clFinish(m_queue);
		
			nContacts = m_totalContactsOut.at(0);
			contactOut->resize(nContacts);
		}
	}

	
#endif//CHECK_ON_HOST
	
	B3_PROFILE("computeConvexConvexContactsGPUSAT");
   // printf("nContacts = %d\n",nContacts);
    
	
	m_sepNormals.resize(nPairs);
	m_hasSeparatingNormals.resize(nPairs);
	
	int concaveCapacity=maxTriConvexPairCapacity;
	m_concaveSepNormals.resize(concaveCapacity);
	m_concaveHasSeparatingNormals.resize(concaveCapacity);
	m_numConcavePairsOut.resize(0);
	m_numConcavePairsOut.push_back(0);

	
	m_gpuCompoundPairs.resize(compoundPairCapacity);

	m_gpuCompoundSepNormals.resize(compoundPairCapacity);
	
	
	m_gpuHasCompoundSepNormals.resize(compoundPairCapacity);
	
	m_numCompoundPairsOut.resize(0);
	m_numCompoundPairsOut.push_back(0);

	int numCompoundPairs = 0;

	int numConcavePairs =0;

	{
		clFinish(m_queue);
		if (findSeparatingAxisOnGpu)
		{
			m_dmins.resize(nPairs);
			if (splitSearchSepAxisConvex)
			{
					

				if (useMprGpu)
				{
					nContacts = m_totalContactsOut.at(0);
					{
						B3_PROFILE("mprPenetrationKernel");
						b3BufferInfoCL bInfo[] = { 
							b3BufferInfoCL( pairs->getBufferCL(), true ), 
							b3BufferInfoCL( bodyBuf->getBufferCL(),true), 
							b3BufferInfoCL( gpuCollidables.getBufferCL(),true), 
							b3BufferInfoCL( convexData.getBufferCL(),true),
							b3BufferInfoCL( gpuVertices.getBufferCL(),true),
							b3BufferInfoCL( m_sepNormals.getBufferCL()),
							b3BufferInfoCL( m_hasSeparatingNormals.getBufferCL()),
							b3BufferInfoCL( contactOut->getBufferCL()),
							b3BufferInfoCL( m_totalContactsOut.getBufferCL())
						};

						b3LauncherCL launcher(m_queue, m_mprPenetrationKernel,"mprPenetrationKernel");
						launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(b3BufferInfoCL) );

						launcher.setConst(maxContactCapacity);
						launcher.setConst( nPairs  );

						int num = nPairs;
						launcher.launch1D( num);
						clFinish(m_queue);
						/*
						b3AlignedObjectArray<int>hostHasSepAxis;
						m_hasSeparatingNormals.copyToHost(hostHasSepAxis);
						b3AlignedObjectArray<b3Vector3>hostSepAxis;
						m_sepNormals.copyToHost(hostSepAxis);
						*/
						nContacts = m_totalContactsOut.at(0);
						contactOut->resize(nContacts);
					//	printf("nContacts (after mprPenetrationKernel) = %d\n",nContacts);
						if (nContacts>maxContactCapacity)
						{
                
							b3Error("Error: contacts exceeds capacity (%d/%d)\n", nContacts, maxContactCapacity);
							nContacts = maxContactCapacity;
						}

					}
				}
				
				if (1)
				{

					if (1)
					{
					{
						B3_PROFILE("findSeparatingAxisVertexFaceKernel");
						b3BufferInfoCL bInfo[] = { 
							b3BufferInfoCL( pairs->getBufferCL(), true ), 
							b3BufferInfoCL( bodyBuf->getBufferCL(),true), 
							b3BufferInfoCL( gpuCollidables.getBufferCL(),true), 
							b3BufferInfoCL( convexData.getBufferCL(),true),
							b3BufferInfoCL( gpuVertices.getBufferCL(),true),
							b3BufferInfoCL( gpuUniqueEdges.getBufferCL(),true),
							b3BufferInfoCL( gpuFaces.getBufferCL(),true),
							b3BufferInfoCL( gpuIndices.getBufferCL(),true),
							b3BufferInfoCL( clAabbsWorldSpace.getBufferCL(),true),
							b3BufferInfoCL( m_sepNormals.getBufferCL()),
							b3BufferInfoCL( m_hasSeparatingNormals.getBufferCL()),
							b3BufferInfoCL( m_dmins.getBufferCL())
						};

						b3LauncherCL launcher(m_queue, m_findSeparatingAxisVertexFaceKernel,"findSeparatingAxisVertexFaceKernel");
						launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(b3BufferInfoCL) );
						launcher.setConst( nPairs  );

						int num = nPairs;
						launcher.launch1D( num);
						clFinish(m_queue);
					}


					int numDirections = sizeof(unitSphere162)/sizeof(b3Vector3);
					
					{
						B3_PROFILE("findSeparatingAxisEdgeEdgeKernel");
						b3BufferInfoCL bInfo[] = { 
							b3BufferInfoCL( pairs->getBufferCL(), true ), 
							b3BufferInfoCL( bodyBuf->getBufferCL(),true), 
							b3BufferInfoCL( gpuCollidables.getBufferCL(),true), 
							b3BufferInfoCL( convexData.getBufferCL(),true),
							b3BufferInfoCL( gpuVertices.getBufferCL(),true),
							b3BufferInfoCL( gpuUniqueEdges.getBufferCL(),true),
							b3BufferInfoCL( gpuFaces.getBufferCL(),true),
							b3BufferInfoCL( gpuIndices.getBufferCL(),true),
							b3BufferInfoCL( clAabbsWorldSpace.getBufferCL(),true),
							b3BufferInfoCL( m_sepNormals.getBufferCL()),
							b3BufferInfoCL( m_hasSeparatingNormals.getBufferCL()),
							b3BufferInfoCL( m_dmins.getBufferCL()),
							b3BufferInfoCL( m_unitSphereDirections.getBufferCL(),true)

						};

						b3LauncherCL launcher(m_queue, m_findSeparatingAxisEdgeEdgeKernel,"findSeparatingAxisEdgeEdgeKernel");
						launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(b3BufferInfoCL) );
						launcher.setConst( numDirections);
						launcher.setConst( nPairs  );
						int num = nPairs;
						launcher.launch1D( num);
						clFinish(m_queue);

					}
					}
					if (useMprGpu)
					{
						B3_PROFILE("findSeparatingAxisUnitSphereKernel");
						b3BufferInfoCL bInfo[] = { 
								b3BufferInfoCL( pairs->getBufferCL(), true ), 
								b3BufferInfoCL( bodyBuf->getBufferCL(),true), 
								b3BufferInfoCL( gpuCollidables.getBufferCL(),true), 
								b3BufferInfoCL( convexData.getBufferCL(),true),
								b3BufferInfoCL( gpuVertices.getBufferCL(),true),
								b3BufferInfoCL( m_unitSphereDirections.getBufferCL(),true),
								b3BufferInfoCL( m_sepNormals.getBufferCL()),
								b3BufferInfoCL( m_hasSeparatingNormals.getBufferCL()),
								b3BufferInfoCL( m_dmins.getBufferCL())
						};

						b3LauncherCL launcher(m_queue, m_findSeparatingAxisUnitSphereKernel,"findSeparatingAxisUnitSphereKernel");
						launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(b3BufferInfoCL) );
						int numDirections = sizeof(unitSphere162)/sizeof(b3Vector3);
						launcher.setConst( numDirections);

						launcher.setConst( nPairs  );
                                                
						int num = nPairs;
						launcher.launch1D( num);
						clFinish(m_queue);
					}
			}
				

			} else
			{
				B3_PROFILE("findSeparatingAxisKernel");
				b3BufferInfoCL bInfo[] = { 
					b3BufferInfoCL( pairs->getBufferCL(), true ), 
					b3BufferInfoCL( bodyBuf->getBufferCL(),true), 
					b3BufferInfoCL( gpuCollidables.getBufferCL(),true), 
					b3BufferInfoCL( convexData.getBufferCL(),true),
					b3BufferInfoCL( gpuVertices.getBufferCL(),true),
					b3BufferInfoCL( gpuUniqueEdges.getBufferCL(),true),
					b3BufferInfoCL( gpuFaces.getBufferCL(),true),
					b3BufferInfoCL( gpuIndices.getBufferCL(),true),
					b3BufferInfoCL( clAabbsWorldSpace.getBufferCL(),true),
					b3BufferInfoCL( m_sepNormals.getBufferCL()),
					b3BufferInfoCL( m_hasSeparatingNormals.getBufferCL())
				};

				b3LauncherCL launcher(m_queue, m_findSeparatingAxisKernel,"m_findSeparatingAxisKernel");
				launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(b3BufferInfoCL) );
				launcher.setConst( nPairs  );

				int num = nPairs;
				launcher.launch1D( num);
				clFinish(m_queue);
			}
			
			
		}
        else
        {
            
			B3_PROFILE("findSeparatingAxisKernel CPU");
           
            
            b3AlignedObjectArray<b3Int4> hostPairs;
            pairs->copyToHost(hostPairs);
            b3AlignedObjectArray<b3RigidBodyData> hostBodyBuf;
            bodyBuf->copyToHost(hostBodyBuf);

            b3AlignedObjectArray<b3Collidable> hostCollidables;
            gpuCollidables.copyToHost(hostCollidables);
            
            b3AlignedObjectArray<b3GpuChildShape> cpuChildShapes;
            gpuChildShapes.copyToHost(cpuChildShapes);
            
            b3AlignedObjectArray<b3ConvexPolyhedronData> hostConvexShapeData;
            convexData.copyToHost(hostConvexShapeData);
            
            b3AlignedObjectArray<b3Vector3> hostVertices;
            gpuVertices.copyToHost(hostVertices);
            
            b3AlignedObjectArray<int> hostHasSepAxis;
            hostHasSepAxis.resize(nPairs);
            b3AlignedObjectArray<b3Vector3> hostSepAxis;
            hostSepAxis.resize(nPairs);
            
            b3AlignedObjectArray<b3Vector3> hostUniqueEdges;
            gpuUniqueEdges.copyToHost(hostUniqueEdges);
            b3AlignedObjectArray<b3GpuFace> hostFaces;
            gpuFaces.copyToHost(hostFaces);
            
            b3AlignedObjectArray<int> hostIndices;
            gpuIndices.copyToHost(hostIndices);
			
			b3AlignedObjectArray<b3Contact4> hostContacts;
			if (nContacts)
			{
				contactOut->copyToHost(hostContacts);
			}
			hostContacts.resize(maxContactCapacity);
			int nGlobalContactsOut = nContacts;
			
            
            for (int i=0;i<nPairs;i++)
            {
                
                int bodyIndexA = hostPairs[i].x;
                int bodyIndexB = hostPairs[i].y;
                int collidableIndexA = hostBodyBuf[bodyIndexA].m_collidableIdx;
                int collidableIndexB = hostBodyBuf[bodyIndexB].m_collidableIdx;
                
                int shapeIndexA = hostCollidables[collidableIndexA].m_shapeIndex;
                int shapeIndexB = hostCollidables[collidableIndexB].m_shapeIndex;
                
                hostHasSepAxis[i] = 0;
                
                //once the broadphase avoids static-static pairs, we can remove this test
                if ((hostBodyBuf[bodyIndexA].m_invMass==0) &&(hostBodyBuf[bodyIndexB].m_invMass==0))
                {
                    continue;
                }
                
                
                if ((hostCollidables[collidableIndexA].m_shapeType!=SHAPE_CONVEX_HULL) ||(hostCollidables[collidableIndexB].m_shapeType!=SHAPE_CONVEX_HULL))
                {
                    continue;
                }
                
                float dmin = FLT_MAX;
                
                b3ConvexPolyhedronData* convexShapeA = &hostConvexShapeData[shapeIndexA];
                b3ConvexPolyhedronData* convexShapeB = &hostConvexShapeData[shapeIndexB];
                b3Vector3 posA = hostBodyBuf[bodyIndexA].m_pos;
                b3Vector3 posB = hostBodyBuf[bodyIndexB].m_pos;
                b3Quaternion ornA =hostBodyBuf[bodyIndexA].m_quat;
                b3Quaternion ornB =hostBodyBuf[bodyIndexB].m_quat;
				
				
				if (useGjk)
				{

					//first approximate the separating axis, to 'fail-proof' GJK+EPA or MPR
					{
						b3Vector3 c0local = hostConvexShapeData[shapeIndexA].m_localCenter;
						b3Vector3 c0 = b3TransformPoint(c0local, posA, ornA);
						b3Vector3 c1local = hostConvexShapeData[shapeIndexB].m_localCenter;
						b3Vector3 c1 = b3TransformPoint(c1local,posB,ornB);
						b3Vector3 DeltaC2 = c0 - c1;
                
						b3Vector3 sepAxis;
                
						bool hasSepAxisA = b3FindSeparatingAxis(convexShapeA, convexShapeB, posA, ornA, posB, ornB, DeltaC2,
							&hostVertices.at(0), &hostUniqueEdges.at(0), &hostFaces.at(0), &hostIndices.at(0),
							&hostVertices.at(0), &hostUniqueEdges.at(0), &hostFaces.at(0), &hostIndices.at(0),
											 &sepAxis, &dmin);
                
						if (hasSepAxisA)
						{
							bool hasSepAxisB = b3FindSeparatingAxis(convexShapeB, convexShapeA, posB, ornB, posA, ornA, DeltaC2,
																	&hostVertices.at(0), &hostUniqueEdges.at(0), &hostFaces.at(0), &hostIndices.at(0),
																	&hostVertices.at(0), &hostUniqueEdges.at(0), &hostFaces.at(0), &hostIndices.at(0),
																	&sepAxis, &dmin);
							if (hasSepAxisB)
							{
								bool hasEdgeEdge =b3FindSeparatingAxisEdgeEdge(convexShapeA, convexShapeB, posA, ornA, posB, ornB, DeltaC2,
															 &hostVertices.at(0), &hostUniqueEdges.at(0), &hostFaces.at(0), &hostIndices.at(0),
															 &hostVertices.at(0), &hostUniqueEdges.at(0), &hostFaces.at(0), &hostIndices.at(0),
															 &sepAxis, &dmin,false);
													 
								if (hasEdgeEdge)
								{
									hostHasSepAxis[i] = 1;
									hostSepAxis[i] = sepAxis;
									hostSepAxis[i].w = dmin;
								}
							}
						}
					}

					if (hostHasSepAxis[i])
					{
						int pairIndex = i;
				
						bool useMpr = true;
						if (useMpr)
						{
							int res=0;
							float depth = 0.f;
							b3Vector3 sepAxis2 = b3MakeVector3(1,0,0);
							b3Vector3 resultPointOnBWorld = b3MakeVector3(0,0,0);

						float depthOut;
						b3Vector3 dirOut;
						b3Vector3 posOut;
						

						//res = b3MprPenetration(bodyIndexA,bodyIndexB,hostBodyBuf,hostConvexShapeData,hostCollidables,hostVertices,&mprConfig,&depthOut,&dirOut,&posOut);
						res = b3MprPenetration(pairIndex,bodyIndexA,bodyIndexB,&hostBodyBuf[0],&hostConvexShapeData[0],&hostCollidables[0],&hostVertices[0],&hostSepAxis[0],&hostHasSepAxis[0],&depthOut,&dirOut,&posOut);
						depth = depthOut;
						sepAxis2 =  b3MakeVector3(-dirOut.x,-dirOut.y,-dirOut.z);
						resultPointOnBWorld = posOut;
						//hostHasSepAxis[i] = 0;


						if (res==0)
						{
							//add point?
							//printf("depth = %f\n",depth);
							//printf("normal = %f,%f,%f\n",dir.v[0],dir.v[1],dir.v[2]);
							//qprintf("pos = %f,%f,%f\n",pos.v[0],pos.v[1],pos.v[2]);
						
							

							float dist=0.f;

							const b3ConvexPolyhedronData& hullA = hostConvexShapeData[hostCollidables[hostBodyBuf[bodyIndexA].m_collidableIdx].m_shapeIndex];
							const b3ConvexPolyhedronData& hullB = hostConvexShapeData[hostCollidables[hostBodyBuf[bodyIndexB].m_collidableIdx].m_shapeIndex];

							if(b3TestSepAxis( &hullA, &hullB, posA,ornA,posB,ornB,&sepAxis2, &hostVertices[0], &hostVertices[0],&dist))
							{
								if (depth > dist)
								{
									float diff = depth - dist;
									
									static float maxdiff = 0.f;
									if (maxdiff < diff)
									{
										maxdiff = diff;
										printf("maxdiff = %20.10f\n",maxdiff);
									}
								}
							}
							if (depth > dmin)
							{
								b3Vector3 oldAxis = hostSepAxis[i];
								depth = dmin;
								sepAxis2 = oldAxis;
							}

							

							if(b3TestSepAxis( &hullA, &hullB, posA,ornA,posB,ornB,&sepAxis2, &hostVertices[0], &hostVertices[0],&dist))
							{
								if (depth > dist)
								{
									float diff = depth - dist;
									//printf("?diff  = %f\n",diff );
									static float maxdiff = 0.f;
									if (maxdiff < diff)
									{
										maxdiff = diff;
										printf("maxdiff = %20.10f\n",maxdiff);
									}
								}
								//this is used for SAT
								//hostHasSepAxis[i] = 1;
								//hostSepAxis[i] = sepAxis2;

								//add contact point

								//int contactIndex = nGlobalContactsOut;
								b3Contact4& newContact = hostContacts.at(nGlobalContactsOut);
								nGlobalContactsOut++;
								newContact.m_batchIdx = 0;//i;
								newContact.m_bodyAPtrAndSignBit = (hostBodyBuf.at(bodyIndexA).m_invMass==0)? -bodyIndexA:bodyIndexA;
								newContact.m_bodyBPtrAndSignBit = (hostBodyBuf.at(bodyIndexB).m_invMass==0)? -bodyIndexB:bodyIndexB;

								newContact.m_frictionCoeffCmp = 45874;
								newContact.m_restituitionCoeffCmp = 0;
						
								
								static float maxDepth = 0.f;
							
								if (depth > maxDepth)
								{
									maxDepth  = depth;
									printf("MPR maxdepth = %f\n",maxDepth );
							
								}
							

								resultPointOnBWorld.w = -depth;
								newContact.m_worldPosB[0] = resultPointOnBWorld;
								//b3Vector3 resultPointOnAWorld = resultPointOnBWorld+depth*sepAxis2;
								newContact.m_worldNormalOnB = sepAxis2;
								newContact.m_worldNormalOnB.w = (b3Scalar)1;
							} else
							{
								printf("rejected\n");
							}

			
						}
						} else
						{

			
				
				
							int result;
							result = computeContactConvexConvex( hostPairs,
													   pairIndex,
													bodyIndexA, bodyIndexB,
													   collidableIndexA, collidableIndexB,
													   hostBodyBuf,
													   hostCollidables,
													   hostConvexShapeData,
													   hostVertices,
													   hostUniqueEdges,
													   hostIndices,
													   hostFaces,
													   hostContacts,
													   nGlobalContactsOut,
														maxContactCapacity,
														hostHasSepAxis,
														hostSepAxis
														
																);
						}//mpr
					}//hostHasSepAxis[i] = 1;
					
				} else
				{
				
					b3Vector3 c0local = hostConvexShapeData[shapeIndexA].m_localCenter;
					b3Vector3 c0 = b3TransformPoint(c0local, posA, ornA);
					b3Vector3 c1local = hostConvexShapeData[shapeIndexB].m_localCenter;
					b3Vector3 c1 = b3TransformPoint(c1local,posB,ornB);
					b3Vector3 DeltaC2 = c0 - c1;
                
					b3Vector3 sepAxis;
                
					bool hasSepAxisA = b3FindSeparatingAxis(convexShapeA, convexShapeB, posA, ornA, posB, ornB, DeltaC2,
						&hostVertices.at(0), &hostUniqueEdges.at(0), &hostFaces.at(0), &hostIndices.at(0),
						&hostVertices.at(0), &hostUniqueEdges.at(0), &hostFaces.at(0), &hostIndices.at(0),
										 &sepAxis, &dmin);
                
					if (hasSepAxisA)
					{
						bool hasSepAxisB = b3FindSeparatingAxis(convexShapeB, convexShapeA, posB, ornB, posA, ornA, DeltaC2,
																&hostVertices.at(0), &hostUniqueEdges.at(0), &hostFaces.at(0), &hostIndices.at(0),
																&hostVertices.at(0), &hostUniqueEdges.at(0), &hostFaces.at(0), &hostIndices.at(0),
																&sepAxis, &dmin);
						if (hasSepAxisB)
						{
							bool hasEdgeEdge =b3FindSeparatingAxisEdgeEdge(convexShapeA, convexShapeB, posA, ornA, posB, ornB, DeltaC2,
														 &hostVertices.at(0), &hostUniqueEdges.at(0), &hostFaces.at(0), &hostIndices.at(0),
														 &hostVertices.at(0), &hostUniqueEdges.at(0), &hostFaces.at(0), &hostIndices.at(0),
														 &sepAxis, &dmin,true);
													 
							if (hasEdgeEdge)
							{
								hostHasSepAxis[i] = 1;
								hostSepAxis[i] = sepAxis;
							}
						}
					}
				}
            }
            
			if (useGjkContacts)//nGlobalContactsOut>0)
			{
				//printf("nGlobalContactsOut=%d\n",nGlobalContactsOut);
				nContacts = nGlobalContactsOut;
				contactOut->copyFromHost(hostContacts);
	
				m_totalContactsOut.copyFromHostPointer(&nContacts,1,0,true);
			}
            
            m_hasSeparatingNormals.copyFromHost(hostHasSepAxis);
            m_sepNormals.copyFromHost(hostSepAxis);
        
            /*
             //double-check results from GPU (comment-out the 'else' so both paths are executed
            b3AlignedObjectArray<int> checkHasSepAxis;
            m_hasSeparatingNormals.copyToHost(checkHasSepAxis);
            static int frameCount = 0;
            frameCount++;
            for (int i=0;i<nPairs;i++)
            {
                if (hostHasSepAxis[i] != checkHasSepAxis[i])
                {
                    printf("at frameCount %d hostHasSepAxis[%d] = %d but checkHasSepAxis[i] = %d\n",
                           frameCount,i,hostHasSepAxis[i],checkHasSepAxis[i]);
                }
            }
            //m_hasSeparatingNormals.copyFromHost(hostHasSepAxis);
            //    m_sepNormals.copyFromHost(hostSepAxis);
            */
        }
        
        
        numCompoundPairs = m_numCompoundPairsOut.at(0);
        bool useGpuFindCompoundPairs=true;
        if (useGpuFindCompoundPairs)
        {
            B3_PROFILE("findCompoundPairsKernel");
            b3BufferInfoCL bInfo[] = 
            { 
                b3BufferInfoCL( pairs->getBufferCL(), true ), 
                b3BufferInfoCL( bodyBuf->getBufferCL(),true), 
                b3BufferInfoCL( gpuCollidables.getBufferCL(),true), 
                b3BufferInfoCL( convexData.getBufferCL(),true),
                b3BufferInfoCL( gpuVertices.getBufferCL(),true),
                b3BufferInfoCL( gpuUniqueEdges.getBufferCL(),true),
                b3BufferInfoCL( gpuFaces.getBufferCL(),true),
                b3BufferInfoCL( gpuIndices.getBufferCL(),true),
                b3BufferInfoCL( clAabbsLocalSpace.getBufferCL(),true),
                b3BufferInfoCL( gpuChildShapes.getBufferCL(),true),
                b3BufferInfoCL( m_gpuCompoundPairs.getBufferCL()),
                b3BufferInfoCL( m_numCompoundPairsOut.getBufferCL()),
                b3BufferInfoCL(subTreesGPU->getBufferCL()),
                b3BufferInfoCL(treeNodesGPU->getBufferCL()),
                b3BufferInfoCL(bvhInfo->getBufferCL())
            };

            b3LauncherCL launcher(m_queue, m_findCompoundPairsKernel,"m_findCompoundPairsKernel");
            launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(b3BufferInfoCL) );
            launcher.setConst( nPairs  );
            launcher.setConst( compoundPairCapacity);

            int num = nPairs;
            launcher.launch1D( num);
            clFinish(m_queue);

            numCompoundPairs = m_numCompoundPairsOut.at(0);
            //printf("numCompoundPairs =%d\n",numCompoundPairs );
            if (numCompoundPairs)
            {
                //printf("numCompoundPairs=%d\n",numCompoundPairs);
            }
            

        } else
        {


            b3AlignedObjectArray<b3QuantizedBvhNode>	treeNodesCPU;
            treeNodesGPU->copyToHost(treeNodesCPU);

            b3AlignedObjectArray<b3BvhSubtreeInfo>	subTreesCPU;
            subTreesGPU->copyToHost(subTreesCPU);

            b3AlignedObjectArray<b3BvhInfo>	bvhInfoCPU;
            bvhInfo->copyToHost(bvhInfoCPU);

            b3AlignedObjectArray<b3Aabb> hostAabbsWorldSpace;
            clAabbsWorldSpace.copyToHost(hostAabbsWorldSpace);

            b3AlignedObjectArray<b3Aabb> hostAabbsLocalSpace;
            clAabbsLocalSpace.copyToHost(hostAabbsLocalSpace);

            b3AlignedObjectArray<b3Int4> hostPairs;
            pairs->copyToHost(hostPairs);

            b3AlignedObjectArray<b3RigidBodyData> hostBodyBuf;
            bodyBuf->copyToHost(hostBodyBuf);


            b3AlignedObjectArray<b3Int4> cpuCompoundPairsOut;
            cpuCompoundPairsOut.resize(compoundPairCapacity);

            b3AlignedObjectArray<b3Collidable> hostCollidables;
            gpuCollidables.copyToHost(hostCollidables);

            b3AlignedObjectArray<b3GpuChildShape> cpuChildShapes;
            gpuChildShapes.copyToHost(cpuChildShapes);

            b3AlignedObjectArray<b3ConvexPolyhedronData> hostConvexData;
            convexData.copyToHost(hostConvexData);

            b3AlignedObjectArray<b3Vector3> hostVertices;
            gpuVertices.copyToHost(hostVertices);




            for (int pairIndex=0;pairIndex<nPairs;pairIndex++)
            {
                int bodyIndexA = hostPairs[pairIndex].x;
                int bodyIndexB = hostPairs[pairIndex].y;
                int collidableIndexA = hostBodyBuf[bodyIndexA].m_collidableIdx;
                int collidableIndexB = hostBodyBuf[bodyIndexB].m_collidableIdx;
				if (cpuChildShapes.size())
				{
                findCompoundPairsKernel( 
                            pairIndex,
                            bodyIndexA,
                            bodyIndexB,
                            collidableIndexA,
                            collidableIndexB,
                            &hostBodyBuf[0],
                            &hostCollidables[0],
                            &hostConvexData[0],
                            hostVertices,
                            hostAabbsWorldSpace,
                            hostAabbsLocalSpace,
                            &cpuChildShapes[0],
                            &cpuCompoundPairsOut[0],
                            &numCompoundPairs,
                            compoundPairCapacity,
                            treeNodesCPU,
                            subTreesCPU,
                            bvhInfoCPU
                            );
				}
            }
            

			m_numCompoundPairsOut.copyFromHostPointer(&numCompoundPairs,1,0,true);
			if (numCompoundPairs)
			{
				b3CompoundOverlappingPair* ptr = (b3CompoundOverlappingPair*)&cpuCompoundPairsOut[0];
				m_gpuCompoundPairs.copyFromHostPointer(ptr,numCompoundPairs,0,true);
			}
			//cpuCompoundPairsOut
            
        }
		if (numCompoundPairs)
		{
			printf("numCompoundPairs=%d\n",numCompoundPairs);
		}

        if (numCompoundPairs > compoundPairCapacity)
        {
            b3Error("Exceeded compound pair capacity (%d/%d)\n", numCompoundPairs,  compoundPairCapacity);
            numCompoundPairs = compoundPairCapacity;
        }

        

        m_gpuCompoundPairs.resize(numCompoundPairs);
        m_gpuHasCompoundSepNormals.resize(numCompoundPairs);
        m_gpuCompoundSepNormals.resize(numCompoundPairs);
        

        if (numCompoundPairs)
        {
            B3_PROFILE("processCompoundPairsPrimitivesKernel");
            b3BufferInfoCL bInfo[] = 
            { 
                b3BufferInfoCL( m_gpuCompoundPairs.getBufferCL(), true ), 
                b3BufferInfoCL( bodyBuf->getBufferCL(),true), 
                b3BufferInfoCL( gpuCollidables.getBufferCL(),true), 
                b3BufferInfoCL( convexData.getBufferCL(),true),
                b3BufferInfoCL( gpuVertices.getBufferCL(),true),
                b3BufferInfoCL( gpuUniqueEdges.getBufferCL(),true),
                b3BufferInfoCL( gpuFaces.getBufferCL(),true),
                b3BufferInfoCL( gpuIndices.getBufferCL(),true),
                b3BufferInfoCL( clAabbsWorldSpace.getBufferCL(),true),
                b3BufferInfoCL( gpuChildShapes.getBufferCL(),true),
                b3BufferInfoCL( contactOut->getBufferCL()),
                b3BufferInfoCL( m_totalContactsOut.getBufferCL())	
            };

            b3LauncherCL launcher(m_queue, m_processCompoundPairsPrimitivesKernel,"m_processCompoundPairsPrimitivesKernel");
            launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(b3BufferInfoCL) );
            launcher.setConst( numCompoundPairs  );
            launcher.setConst(maxContactCapacity);

            int num = numCompoundPairs;
            launcher.launch1D( num);
            clFinish(m_queue);
            nContacts = m_totalContactsOut.at(0);
            //printf("nContacts (after processCompoundPairsPrimitivesKernel) = %d\n",nContacts);
            if (nContacts>maxContactCapacity)
            {
                
                b3Error("Error: contacts exceeds capacity (%d/%d)\n", nContacts, maxContactCapacity);
                nContacts = maxContactCapacity;
            }
        }
        

        if (numCompoundPairs)
        {
            B3_PROFILE("processCompoundPairsKernel");
            b3BufferInfoCL bInfo[] = 
            { 
                b3BufferInfoCL( m_gpuCompoundPairs.getBufferCL(), true ), 
                b3BufferInfoCL( bodyBuf->getBufferCL(),true), 
                b3BufferInfoCL( gpuCollidables.getBufferCL(),true), 
                b3BufferInfoCL( convexData.getBufferCL(),true),
                b3BufferInfoCL( gpuVertices.getBufferCL(),true),
                b3BufferInfoCL( gpuUniqueEdges.getBufferCL(),true),
                b3BufferInfoCL( gpuFaces.getBufferCL(),true),
                b3BufferInfoCL( gpuIndices.getBufferCL(),true),
                b3BufferInfoCL( clAabbsWorldSpace.getBufferCL(),true),
                b3BufferInfoCL( gpuChildShapes.getBufferCL(),true),
                b3BufferInfoCL( m_gpuCompoundSepNormals.getBufferCL()),
                b3BufferInfoCL( m_gpuHasCompoundSepNormals.getBufferCL())
            };

            b3LauncherCL launcher(m_queue, m_processCompoundPairsKernel,"m_processCompoundPairsKernel");
            launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(b3BufferInfoCL) );
            launcher.setConst( numCompoundPairs  );

            int num = numCompoundPairs;
            launcher.launch1D( num);
            clFinish(m_queue);
        
        }


        //printf("numConcave  = %d\n",numConcave);

    

//		printf("hostNormals.size()=%d\n",hostNormals.size());
		//int numPairs = pairCount.at(0);
		
		
		
	}
	int vertexFaceCapacity = 64;


		
	{
		//now perform the tree query on GPU
			
					
				
				
		if (treeNodesGPU->size() && treeNodesGPU->size())
		{
			if (bvhTraversalKernelGPU)
			{
						
				B3_PROFILE("m_bvhTraversalKernel");
						
						
				numConcavePairs = m_numConcavePairsOut.at(0);
						
				b3LauncherCL launcher(m_queue, m_bvhTraversalKernel,"m_bvhTraversalKernel");
				launcher.setBuffer( pairs->getBufferCL());
				launcher.setBuffer(  bodyBuf->getBufferCL());
				launcher.setBuffer( gpuCollidables.getBufferCL());
				launcher.setBuffer( clAabbsWorldSpace.getBufferCL());
				launcher.setBuffer( triangleConvexPairsOut.getBufferCL());
				launcher.setBuffer( m_numConcavePairsOut.getBufferCL());
				launcher.setBuffer( subTreesGPU->getBufferCL());
				launcher.setBuffer( treeNodesGPU->getBufferCL());
				launcher.setBuffer( bvhInfo->getBufferCL());
						
				launcher.setConst( nPairs  );
				launcher.setConst( maxTriConvexPairCapacity);
				int num = nPairs;
				launcher.launch1D( num);
				clFinish(m_queue);
				numConcavePairs = m_numConcavePairsOut.at(0);
			} else
			{
					b3AlignedObjectArray<b3Int4> hostPairs;
					pairs->copyToHost(hostPairs);
					b3AlignedObjectArray<b3RigidBodyData> hostBodyBuf;
					bodyBuf->copyToHost(hostBodyBuf);
					b3AlignedObjectArray<b3Collidable> hostCollidables;
					gpuCollidables.copyToHost(hostCollidables);
					b3AlignedObjectArray<b3Aabb> hostAabbsWorldSpace;
					clAabbsWorldSpace.copyToHost(hostAabbsWorldSpace);

					//int maxTriConvexPairCapacity,
					b3AlignedObjectArray<b3Int4> triangleConvexPairsOutHost;
					triangleConvexPairsOutHost.resize(maxTriConvexPairCapacity);

					//int numTriConvexPairsOutHost=0;
					numConcavePairs = 0;
					//m_numConcavePairsOut

					b3AlignedObjectArray<b3QuantizedBvhNode>	treeNodesCPU;
					treeNodesGPU->copyToHost(treeNodesCPU);
					b3AlignedObjectArray<b3BvhSubtreeInfo>	subTreesCPU;
					subTreesGPU->copyToHost(subTreesCPU);
					b3AlignedObjectArray<b3BvhInfo>	bvhInfoCPU;
					bvhInfo->copyToHost(bvhInfoCPU);
					//compute it...

					volatile int hostNumConcavePairsOut=0;

					//
					for (int i=0;i<nPairs;i++)
					{
						b3BvhTraversal( &hostPairs.at(0), 
						&hostBodyBuf.at(0),
						&hostCollidables.at(0),
						&hostAabbsWorldSpace.at(0),
						&triangleConvexPairsOutHost.at(0),
						&hostNumConcavePairsOut,
						&subTreesCPU.at(0),
						&treeNodesCPU.at(0),
						&bvhInfoCPU.at(0),
						nPairs,
						maxTriConvexPairCapacity,
						i);
					}
					numConcavePairs = hostNumConcavePairsOut;

					if (hostNumConcavePairsOut)
					{
						triangleConvexPairsOutHost.resize(hostNumConcavePairsOut);
						triangleConvexPairsOut.copyFromHost(triangleConvexPairsOutHost);
					}
					//

					m_numConcavePairsOut.resize(0);
					m_numConcavePairsOut.push_back(numConcavePairs);
			}

				//printf("numConcavePairs=%d (max = %d\n",numConcavePairs,maxTriConvexPairCapacity);
						
			if (numConcavePairs > maxTriConvexPairCapacity)
			{
				static int exceeded_maxTriConvexPairCapacity_count = 0;
				b3Error("Exceeded the maxTriConvexPairCapacity (found %d but max is %d, it happened %d times)\n",
					numConcavePairs,maxTriConvexPairCapacity,exceeded_maxTriConvexPairCapacity_count++);
				numConcavePairs = maxTriConvexPairCapacity;
			}
			triangleConvexPairsOut.resize(numConcavePairs);
	
			if (numConcavePairs)
			{

				
						
	
				clippingFacesOutGPU.resize(numConcavePairs);
				worldNormalsAGPU.resize(numConcavePairs);
				worldVertsA1GPU.resize(vertexFaceCapacity*(numConcavePairs));
				worldVertsB1GPU.resize(vertexFaceCapacity*(numConcavePairs));


				if (findConcaveSeparatingAxisKernelGPU)
				{

					/*
					m_concaveHasSeparatingNormals.copyFromHost(concaveHasSeparatingNormalsCPU);
						clippingFacesOutGPU.copyFromHost(clippingFacesOutCPU);
						worldVertsA1GPU.copyFromHost(worldVertsA1CPU);
						worldNormalsAGPU.copyFromHost(worldNormalsACPU);
						worldVertsB1GPU.copyFromHost(worldVertsB1CPU);
					*/

					//now perform a SAT test for each triangle-convex element (stored in triangleConvexPairsOut)
                    if (splitSearchSepAxisConcave)
                    {
                        //printf("numConcavePairs = %d\n",numConcavePairs);
                        m_dmins.resize(numConcavePairs);
                        {
                            B3_PROFILE("findConcaveSeparatingAxisVertexFaceKernel");
                            b3BufferInfoCL bInfo[] = {
                                b3BufferInfoCL( triangleConvexPairsOut.getBufferCL() ),
                                b3BufferInfoCL( bodyBuf->getBufferCL(),true),
                                b3BufferInfoCL( gpuCollidables.getBufferCL(),true),
                                b3BufferInfoCL( convexData.getBufferCL(),true),
                                b3BufferInfoCL( gpuVertices.getBufferCL(),true),
                                b3BufferInfoCL( gpuUniqueEdges.getBufferCL(),true),
                                b3BufferInfoCL( gpuFaces.getBufferCL(),true),
                                b3BufferInfoCL( gpuIndices.getBufferCL(),true),
                                b3BufferInfoCL( gpuChildShapes.getBufferCL(),true),
                                b3BufferInfoCL( clAabbsWorldSpace.getBufferCL(),true),
                                b3BufferInfoCL( m_concaveSepNormals.getBufferCL()),
                                b3BufferInfoCL( m_concaveHasSeparatingNormals.getBufferCL()),
                                b3BufferInfoCL( clippingFacesOutGPU.getBufferCL()),
                                b3BufferInfoCL( worldVertsA1GPU.getBufferCL()),
                                b3BufferInfoCL(worldNormalsAGPU.getBufferCL()),
                                b3BufferInfoCL(worldVertsB1GPU.getBufferCL()),
                                b3BufferInfoCL(m_dmins.getBufferCL())
                            };
                            
                            b3LauncherCL launcher(m_queue, m_findConcaveSeparatingAxisVertexFaceKernel,"m_findConcaveSeparatingAxisVertexFaceKernel");
                            launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(b3BufferInfoCL) );
                            launcher.setConst(vertexFaceCapacity);
                            launcher.setConst( numConcavePairs  );
                            
                            int num = numConcavePairs;
                            launcher.launch1D( num);
                            clFinish(m_queue);

                            
                        }
//                        numConcavePairs = 0;
                        if (1)
                        {
                            B3_PROFILE("findConcaveSeparatingAxisEdgeEdgeKernel");
                            b3BufferInfoCL bInfo[] = {
                                b3BufferInfoCL( triangleConvexPairsOut.getBufferCL() ),
                                b3BufferInfoCL( bodyBuf->getBufferCL(),true),
                                b3BufferInfoCL( gpuCollidables.getBufferCL(),true),
                                b3BufferInfoCL( convexData.getBufferCL(),true),
                                b3BufferInfoCL( gpuVertices.getBufferCL(),true),
                                b3BufferInfoCL( gpuUniqueEdges.getBufferCL(),true),
                                b3BufferInfoCL( gpuFaces.getBufferCL(),true),
                                b3BufferInfoCL( gpuIndices.getBufferCL(),true),
                                b3BufferInfoCL( gpuChildShapes.getBufferCL(),true),
                                b3BufferInfoCL( clAabbsWorldSpace.getBufferCL(),true),
                                b3BufferInfoCL( m_concaveSepNormals.getBufferCL()),
                                b3BufferInfoCL( m_concaveHasSeparatingNormals.getBufferCL()),
                                b3BufferInfoCL( clippingFacesOutGPU.getBufferCL()),
                                b3BufferInfoCL( worldVertsA1GPU.getBufferCL()),
                                b3BufferInfoCL(worldNormalsAGPU.getBufferCL()),
                                b3BufferInfoCL(worldVertsB1GPU.getBufferCL()),
                                b3BufferInfoCL(m_dmins.getBufferCL())
                            };
                            
                            b3LauncherCL launcher(m_queue, m_findConcaveSeparatingAxisEdgeEdgeKernel,"m_findConcaveSeparatingAxisEdgeEdgeKernel");
                            launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(b3BufferInfoCL) );
                            launcher.setConst(vertexFaceCapacity);
                            launcher.setConst( numConcavePairs  );
                            
                            int num = numConcavePairs;
                            launcher.launch1D( num);
                            clFinish(m_queue);
                        }
                      
                        
                        // numConcavePairs = 0;
                        
                        
                        
                        
                        
                        
                    } else
                    {
                        B3_PROFILE("findConcaveSeparatingAxisKernel");
                        b3BufferInfoCL bInfo[] = { 
                            b3BufferInfoCL( triangleConvexPairsOut.getBufferCL() ), 
                            b3BufferInfoCL( bodyBuf->getBufferCL(),true), 
                            b3BufferInfoCL( gpuCollidables.getBufferCL(),true), 
                            b3BufferInfoCL( convexData.getBufferCL(),true),
                            b3BufferInfoCL( gpuVertices.getBufferCL(),true),
                            b3BufferInfoCL( gpuUniqueEdges.getBufferCL(),true),
                            b3BufferInfoCL( gpuFaces.getBufferCL(),true),
                            b3BufferInfoCL( gpuIndices.getBufferCL(),true),
                            b3BufferInfoCL( gpuChildShapes.getBufferCL(),true),
                            b3BufferInfoCL( clAabbsWorldSpace.getBufferCL(),true),
                            b3BufferInfoCL( m_concaveSepNormals.getBufferCL()),
                            b3BufferInfoCL( m_concaveHasSeparatingNormals.getBufferCL()),
                            b3BufferInfoCL( clippingFacesOutGPU.getBufferCL()),
                            b3BufferInfoCL( worldVertsA1GPU.getBufferCL()),
                            b3BufferInfoCL(worldNormalsAGPU.getBufferCL()),
                            b3BufferInfoCL(worldVertsB1GPU.getBufferCL())
                        };

                        b3LauncherCL launcher(m_queue, m_findConcaveSeparatingAxisKernel,"m_findConcaveSeparatingAxisKernel");
                        launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(b3BufferInfoCL) );
                        launcher.setConst(vertexFaceCapacity);
                        launcher.setConst( numConcavePairs  );

                        int num = numConcavePairs;
                        launcher.launch1D( num);
                        clFinish(m_queue);
                    }
                    
                    
				} else
				{

						b3AlignedObjectArray<b3Int4> clippingFacesOutCPU;
						b3AlignedObjectArray<b3Vector3> worldVertsA1CPU;
						b3AlignedObjectArray<b3Vector3> worldNormalsACPU;
						b3AlignedObjectArray<b3Vector3> worldVertsB1CPU;
						b3AlignedObjectArray<int>concaveHasSeparatingNormalsCPU;

						b3AlignedObjectArray<b3Int4> triangleConvexPairsOutHost;
						triangleConvexPairsOut.copyToHost(triangleConvexPairsOutHost);
						//triangleConvexPairsOutHost.resize(maxTriConvexPairCapacity);
						b3AlignedObjectArray<b3RigidBodyData> hostBodyBuf;
						bodyBuf->copyToHost(hostBodyBuf);
						b3AlignedObjectArray<b3Collidable> hostCollidables;
						gpuCollidables.copyToHost(hostCollidables);
						b3AlignedObjectArray<b3Aabb> hostAabbsWorldSpace;
						clAabbsWorldSpace.copyToHost(hostAabbsWorldSpace);

						b3AlignedObjectArray<b3ConvexPolyhedronData> hostConvexData;
						convexData.copyToHost(hostConvexData);

						b3AlignedObjectArray<b3Vector3> hostVertices;
						gpuVertices.copyToHost(hostVertices);

						b3AlignedObjectArray<b3Vector3> hostUniqueEdges;
						gpuUniqueEdges.copyToHost(hostUniqueEdges);
						b3AlignedObjectArray<b3GpuFace> hostFaces;
						gpuFaces.copyToHost(hostFaces);
						b3AlignedObjectArray<int> hostIndices;
						gpuIndices.copyToHost(hostIndices);
						b3AlignedObjectArray<b3GpuChildShape> cpuChildShapes;
						gpuChildShapes.copyToHost(cpuChildShapes);


								
						b3AlignedObjectArray<b3Vector3> concaveSepNormalsHost;
						m_concaveSepNormals.copyToHost(concaveSepNormalsHost);
						concaveHasSeparatingNormalsCPU.resize(concaveSepNormalsHost.size());

						b3GpuChildShape* childShapePointerCPU = 0;
						if (cpuChildShapes.size())
							childShapePointerCPU  = &cpuChildShapes.at(0);

						clippingFacesOutCPU.resize(clippingFacesOutGPU.size());
						worldVertsA1CPU.resize(worldVertsA1GPU.size());
    					worldNormalsACPU.resize(worldNormalsAGPU.size());
						worldVertsB1CPU.resize(worldVertsB1GPU.size());

						for (int i=0;i<numConcavePairs;i++)
						{
							b3FindConcaveSeparatingAxisKernel(&triangleConvexPairsOutHost.at(0),
								&hostBodyBuf.at(0),
								&hostCollidables.at(0),
								&hostConvexData.at(0), &hostVertices.at(0),&hostUniqueEdges.at(0),
								&hostFaces.at(0),&hostIndices.at(0),childShapePointerCPU,
								&hostAabbsWorldSpace.at(0),
								&concaveSepNormalsHost.at(0),
								&clippingFacesOutCPU.at(0),
								&worldVertsA1CPU.at(0),
								&worldNormalsACPU.at(0),
								&worldVertsB1CPU.at(0),
								&concaveHasSeparatingNormalsCPU.at(0),
								vertexFaceCapacity,
								numConcavePairs,i);
						};

						m_concaveSepNormals.copyFromHost(concaveSepNormalsHost);
						m_concaveHasSeparatingNormals.copyFromHost(concaveHasSeparatingNormalsCPU);
						clippingFacesOutGPU.copyFromHost(clippingFacesOutCPU);
						worldVertsA1GPU.copyFromHost(worldVertsA1CPU);
						worldNormalsAGPU.copyFromHost(worldNormalsACPU);
						worldVertsB1GPU.copyFromHost(worldVertsB1CPU);



				}
//							b3AlignedObjectArray<b3Vector3> cpuCompoundSepNormals;
//						m_concaveSepNormals.copyToHost(cpuCompoundSepNormals);
//					b3AlignedObjectArray<b3Int4> cpuConcavePairs;
//				triangleConvexPairsOut.copyToHost(cpuConcavePairs);


			}
		}
		
		
	}

	if (numConcavePairs)
	{
			if (numConcavePairs)
		{
			B3_PROFILE("findConcaveSphereContactsKernel");
				nContacts = m_totalContactsOut.at(0);
//				printf("nContacts1 = %d\n",nContacts);
			b3BufferInfoCL bInfo[] = { 
				b3BufferInfoCL( triangleConvexPairsOut.getBufferCL() ), 
				b3BufferInfoCL( bodyBuf->getBufferCL(),true), 
				b3BufferInfoCL( gpuCollidables.getBufferCL(),true), 
				b3BufferInfoCL( convexData.getBufferCL(),true),
				b3BufferInfoCL( gpuVertices.getBufferCL(),true),
				b3BufferInfoCL( gpuUniqueEdges.getBufferCL(),true),
				b3BufferInfoCL( gpuFaces.getBufferCL(),true),
				b3BufferInfoCL( gpuIndices.getBufferCL(),true),
				b3BufferInfoCL( clAabbsWorldSpace.getBufferCL(),true),
				b3BufferInfoCL( contactOut->getBufferCL()),
				b3BufferInfoCL( m_totalContactsOut.getBufferCL())
			};

			b3LauncherCL launcher(m_queue, m_findConcaveSphereContactsKernel,"m_findConcaveSphereContactsKernel");
			launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(b3BufferInfoCL) );

			launcher.setConst( numConcavePairs  );
			launcher.setConst(maxContactCapacity);

			int num = numConcavePairs;
			launcher.launch1D( num);
			clFinish(m_queue);
			nContacts = m_totalContactsOut.at(0);
			//printf("nContacts (after findConcaveSphereContactsKernel) = %d\n",nContacts);

			//printf("nContacts2 = %d\n",nContacts);

			if (nContacts >= maxContactCapacity)
			{
				b3Error("Error: contacts exceeds capacity (%d/%d)\n", nContacts, maxContactCapacity);
				nContacts = maxContactCapacity;
			}
		}
		
	}



#ifdef __APPLE__
	bool contactClippingOnGpu = true;
#else
	bool contactClippingOnGpu = true;
#endif

	if (contactClippingOnGpu)
	{
		m_totalContactsOut.copyFromHostPointer(&nContacts,1,0,true);
//		printf("nContacts3 = %d\n",nContacts);


		//B3_PROFILE("clipHullHullKernel");

		bool breakupConcaveConvexKernel = true;

#ifdef __APPLE__
		//actually, some Apple OpenCL platform/device combinations work fine...
		breakupConcaveConvexKernel = true;
#endif
		//concave-convex contact clipping
		if (numConcavePairs)
		{
			//			printf("numConcavePairs = %d\n", numConcavePairs);
			//		nContacts = m_totalContactsOut.at(0);
			//	printf("nContacts before = %d\n", nContacts);

			if (breakupConcaveConvexKernel)
			{

				worldVertsB2GPU.resize(vertexFaceCapacity*numConcavePairs);


				//clipFacesAndFindContacts

				if (clipConcaveFacesAndFindContactsCPU)
				{

					b3AlignedObjectArray<b3Int4> clippingFacesOutCPU;
					b3AlignedObjectArray<b3Vector3> worldVertsA1CPU;
					b3AlignedObjectArray<b3Vector3> worldNormalsACPU;
					b3AlignedObjectArray<b3Vector3> worldVertsB1CPU;

					clippingFacesOutGPU.copyToHost(clippingFacesOutCPU);
					worldVertsA1GPU.copyToHost(worldVertsA1CPU);
					worldNormalsAGPU.copyToHost(worldNormalsACPU);
					worldVertsB1GPU.copyToHost(worldVertsB1CPU);



					b3AlignedObjectArray<int>concaveHasSeparatingNormalsCPU;
					m_concaveHasSeparatingNormals.copyToHost(concaveHasSeparatingNormalsCPU);

					b3AlignedObjectArray<b3Vector3> concaveSepNormalsHost;
					m_concaveSepNormals.copyToHost(concaveSepNormalsHost);

					b3AlignedObjectArray<b3Vector3> worldVertsB2CPU;  
					worldVertsB2CPU.resize(worldVertsB2GPU.size());


					for (int i=0;i<numConcavePairs;i++)
					{

						clipFacesAndFindContactsKernel(   &concaveSepNormalsHost.at(0),
							&concaveHasSeparatingNormalsCPU.at(0),
							&clippingFacesOutCPU.at(0),
							&worldVertsA1CPU.at(0),
							&worldNormalsACPU.at(0),
							&worldVertsB1CPU.at(0),
							&worldVertsB2CPU.at(0),
							vertexFaceCapacity,
							i);
					}

					clippingFacesOutGPU.copyFromHost(clippingFacesOutCPU);
					worldVertsB2GPU.copyFromHost(worldVertsB2CPU);


				} else
				{

					if (1)
					{



						B3_PROFILE("clipFacesAndFindContacts");
						//nContacts = m_totalContactsOut.at(0);
						//int h = m_hasSeparatingNormals.at(0);
						//int4 p = clippingFacesOutGPU.at(0);
						b3BufferInfoCL bInfo[] = {
							b3BufferInfoCL( m_concaveSepNormals.getBufferCL()),
							b3BufferInfoCL( m_concaveHasSeparatingNormals.getBufferCL()),
							b3BufferInfoCL( clippingFacesOutGPU.getBufferCL()),
							b3BufferInfoCL( worldVertsA1GPU.getBufferCL()),
							b3BufferInfoCL( worldNormalsAGPU.getBufferCL()),
							b3BufferInfoCL( worldVertsB1GPU.getBufferCL()),
							b3BufferInfoCL( worldVertsB2GPU.getBufferCL())
						};
						b3LauncherCL launcher(m_queue, m_clipFacesAndFindContacts,"m_clipFacesAndFindContacts");
						launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(b3BufferInfoCL) );
						launcher.setConst(vertexFaceCapacity);

						launcher.setConst( numConcavePairs  );
						int debugMode = 0;
						launcher.setConst( debugMode);
						int num = numConcavePairs;
						launcher.launch1D( num);
						clFinish(m_queue);
						//int bla = m_totalContactsOut.at(0);
					}
				}
				//contactReduction
				{
					int newContactCapacity=nContacts+numConcavePairs; 
					contactOut->reserve(newContactCapacity);
					if (reduceConcaveContactsOnGPU)
					{
//						printf("newReservation = %d\n",newReservation);
						{
							B3_PROFILE("newContactReductionKernel");
							b3BufferInfoCL bInfo[] =
							{
								b3BufferInfoCL( triangleConvexPairsOut.getBufferCL(), true ),
								b3BufferInfoCL( bodyBuf->getBufferCL(),true),
								b3BufferInfoCL( m_concaveSepNormals.getBufferCL()),
								b3BufferInfoCL( m_concaveHasSeparatingNormals.getBufferCL()),
								b3BufferInfoCL( contactOut->getBufferCL()),
								b3BufferInfoCL( clippingFacesOutGPU.getBufferCL()),
								b3BufferInfoCL( worldVertsB2GPU.getBufferCL()),
								b3BufferInfoCL( m_totalContactsOut.getBufferCL())
							};

							b3LauncherCL launcher(m_queue, m_newContactReductionKernel,"m_newContactReductionKernel");
							launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(b3BufferInfoCL) );
							launcher.setConst(vertexFaceCapacity);
							launcher.setConst(newContactCapacity);
							launcher.setConst( numConcavePairs  );
							int num = numConcavePairs;

							launcher.launch1D( num);
						}
						nContacts = m_totalContactsOut.at(0);
						contactOut->resize(nContacts);

						//printf("contactOut4 (after newContactReductionKernel) = %d\n",nContacts);
					}else
					{
						
						volatile int nGlobalContactsOut = nContacts;
						b3AlignedObjectArray<b3Int4> triangleConvexPairsOutHost;
						triangleConvexPairsOut.copyToHost(triangleConvexPairsOutHost);
						b3AlignedObjectArray<b3RigidBodyData> hostBodyBuf;
						bodyBuf->copyToHost(hostBodyBuf);

						b3AlignedObjectArray<int>concaveHasSeparatingNormalsCPU;
						m_concaveHasSeparatingNormals.copyToHost(concaveHasSeparatingNormalsCPU);

						b3AlignedObjectArray<b3Vector3> concaveSepNormalsHost;
						m_concaveSepNormals.copyToHost(concaveSepNormalsHost);


						b3AlignedObjectArray<b3Contact4> hostContacts;
						if (nContacts)
						{
							contactOut->copyToHost(hostContacts);
						}
						hostContacts.resize(newContactCapacity);

						b3AlignedObjectArray<b3Int4> clippingFacesOutCPU;
						b3AlignedObjectArray<b3Vector3> worldVertsB2CPU;

						clippingFacesOutGPU.copyToHost(clippingFacesOutCPU);
						worldVertsB2GPU.copyToHost(worldVertsB2CPU);



						for (int i=0;i<numConcavePairs;i++)
						{
							b3NewContactReductionKernel( &triangleConvexPairsOutHost.at(0),
                                                   &hostBodyBuf.at(0),
												   &concaveSepNormalsHost.at(0),
												   &concaveHasSeparatingNormalsCPU.at(0),
												   &hostContacts.at(0),
                                                   &clippingFacesOutCPU.at(0),
                                                   &worldVertsB2CPU.at(0),
                                                   &nGlobalContactsOut,
                                                   vertexFaceCapacity,
												   newContactCapacity,
                                                   numConcavePairs,
												   i
                                                   );

						}


						nContacts = nGlobalContactsOut;
						m_totalContactsOut.copyFromHostPointer(&nContacts,1,0,true);
//						nContacts = m_totalContactsOut.at(0);
						//contactOut->resize(nContacts);
						hostContacts.resize(nContacts);
						//printf("contactOut4 (after newContactReductionKernel) = %d\n",nContacts);
						contactOut->copyFromHost(hostContacts);
					}

				}
				//re-use?


			} else
			{
				B3_PROFILE("clipHullHullConcaveConvexKernel");
				nContacts = m_totalContactsOut.at(0);
				int newContactCapacity = contactOut->capacity();

				//printf("contactOut5 = %d\n",nContacts);
				b3BufferInfoCL bInfo[] = { 
					b3BufferInfoCL( triangleConvexPairsOut.getBufferCL(), true ), 
					b3BufferInfoCL( bodyBuf->getBufferCL(),true), 
					b3BufferInfoCL( gpuCollidables.getBufferCL(),true), 
					b3BufferInfoCL( convexData.getBufferCL(),true),
					b3BufferInfoCL( gpuVertices.getBufferCL(),true),
					b3BufferInfoCL( gpuUniqueEdges.getBufferCL(),true),
					b3BufferInfoCL( gpuFaces.getBufferCL(),true),
					b3BufferInfoCL( gpuIndices.getBufferCL(),true),
					b3BufferInfoCL( gpuChildShapes.getBufferCL(),true),
					b3BufferInfoCL( m_concaveSepNormals.getBufferCL()),
					b3BufferInfoCL( contactOut->getBufferCL()),
					b3BufferInfoCL( m_totalContactsOut.getBufferCL())	
				};
				b3LauncherCL launcher(m_queue, m_clipHullHullConcaveConvexKernel,"m_clipHullHullConcaveConvexKernel");
				launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(b3BufferInfoCL) );
				launcher.setConst(newContactCapacity);
				launcher.setConst( numConcavePairs  );
				int num = numConcavePairs;
				launcher.launch1D( num);
				clFinish(m_queue);
				nContacts = m_totalContactsOut.at(0);
				contactOut->resize(nContacts);
				//printf("contactOut6 = %d\n",nContacts);
				b3AlignedObjectArray<b3Contact4> cpuContacts;
				contactOut->copyToHost(cpuContacts);
			}
			//			printf("nContacts after = %d\n", nContacts);
		}//numConcavePairs



		//convex-convex contact clipping
		
		bool breakupKernel = false;

#ifdef __APPLE__
		breakupKernel = true;
#endif

#ifdef CHECK_ON_HOST
	bool computeConvexConvex = false;
#else
	bool computeConvexConvex = true;
#endif//CHECK_ON_HOST
		if (computeConvexConvex)
		{
			B3_PROFILE("clipHullHullKernel");
		if (breakupKernel)
		{




			worldVertsB1GPU.resize(vertexFaceCapacity*nPairs);
			clippingFacesOutGPU.resize(nPairs);
			worldNormalsAGPU.resize(nPairs);
			worldVertsA1GPU.resize(vertexFaceCapacity*nPairs);
			worldVertsB2GPU.resize(vertexFaceCapacity*nPairs);

			if (findConvexClippingFacesGPU)
			{
				B3_PROFILE("findClippingFacesKernel");
				b3BufferInfoCL bInfo[] = {
					b3BufferInfoCL( pairs->getBufferCL(), true ),
					b3BufferInfoCL( bodyBuf->getBufferCL(),true),
					b3BufferInfoCL( gpuCollidables.getBufferCL(),true),
					b3BufferInfoCL( convexData.getBufferCL(),true),
					b3BufferInfoCL( gpuVertices.getBufferCL(),true),
					b3BufferInfoCL( gpuUniqueEdges.getBufferCL(),true),
					b3BufferInfoCL( gpuFaces.getBufferCL(),true), 
					b3BufferInfoCL( gpuIndices.getBufferCL(),true),
					b3BufferInfoCL( m_sepNormals.getBufferCL()),
					b3BufferInfoCL( m_hasSeparatingNormals.getBufferCL()),
					b3BufferInfoCL( clippingFacesOutGPU.getBufferCL()),
					b3BufferInfoCL( worldVertsA1GPU.getBufferCL()),
					b3BufferInfoCL( worldNormalsAGPU.getBufferCL()),
					b3BufferInfoCL( worldVertsB1GPU.getBufferCL())
				};

				b3LauncherCL launcher(m_queue, m_findClippingFacesKernel,"m_findClippingFacesKernel");
				launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(b3BufferInfoCL) );
				launcher.setConst( vertexFaceCapacity);
				launcher.setConst( nPairs  );
				int num = nPairs;
				launcher.launch1D( num);
				clFinish(m_queue);

			} else
			{
				
				float minDist = -1e30f;
				float maxDist = 0.02f;

				b3AlignedObjectArray<b3ConvexPolyhedronData> hostConvexData;
				convexData.copyToHost(hostConvexData);
				b3AlignedObjectArray<b3Collidable> hostCollidables;
				gpuCollidables.copyToHost(hostCollidables);

				b3AlignedObjectArray<int> hostHasSepNormals;
				m_hasSeparatingNormals.copyToHost(hostHasSepNormals);
				b3AlignedObjectArray<b3Vector3> cpuSepNormals;
				m_sepNormals.copyToHost(cpuSepNormals);

				b3AlignedObjectArray<b3Int4> hostPairs;
				pairs->copyToHost(hostPairs);
				b3AlignedObjectArray<b3RigidBodyData> hostBodyBuf;
				bodyBuf->copyToHost(hostBodyBuf);


				//worldVertsB1GPU.resize(vertexFaceCapacity*nPairs);
				b3AlignedObjectArray<b3Vector3> worldVertsB1CPU;
				worldVertsB1GPU.copyToHost(worldVertsB1CPU);

				b3AlignedObjectArray<b3Int4> clippingFacesOutCPU;
				clippingFacesOutGPU.copyToHost(clippingFacesOutCPU);

				b3AlignedObjectArray<b3Vector3> worldNormalsACPU;
				worldNormalsACPU.resize(nPairs);

				b3AlignedObjectArray<b3Vector3> worldVertsA1CPU;
				worldVertsA1CPU.resize(worldVertsA1GPU.size());
			
			
				b3AlignedObjectArray<b3Vector3> hostVertices;
				gpuVertices.copyToHost(hostVertices);
				b3AlignedObjectArray<b3GpuFace> hostFaces;
				gpuFaces.copyToHost(hostFaces);
				b3AlignedObjectArray<int> hostIndices;
				gpuIndices.copyToHost(hostIndices);
				

				for (int i=0;i<nPairs;i++)
				{

					int bodyIndexA = hostPairs[i].x;
					int bodyIndexB = hostPairs[i].y;
			
					int collidableIndexA = hostBodyBuf[bodyIndexA].m_collidableIdx;
					int collidableIndexB = hostBodyBuf[bodyIndexB].m_collidableIdx;
			
					int shapeIndexA = hostCollidables[collidableIndexA].m_shapeIndex;
					int shapeIndexB = hostCollidables[collidableIndexB].m_shapeIndex;
			

					if (hostHasSepNormals[i])
					{
						b3FindClippingFaces(cpuSepNormals[i],
							&hostConvexData[shapeIndexA],
							&hostConvexData[shapeIndexB],
							hostBodyBuf[bodyIndexA].m_pos,hostBodyBuf[bodyIndexA].m_quat,
							hostBodyBuf[bodyIndexB].m_pos,hostBodyBuf[bodyIndexB].m_quat,
							&worldVertsA1CPU.at(0),&worldNormalsACPU.at(0),
							&worldVertsB1CPU.at(0),
							vertexFaceCapacity,minDist,maxDist,
							&hostVertices.at(0),&hostFaces.at(0),
							&hostIndices.at(0),
							&hostVertices.at(0),&hostFaces.at(0),
							&hostIndices.at(0),&clippingFacesOutCPU.at(0),i);
					}
				}

				clippingFacesOutGPU.copyFromHost(clippingFacesOutCPU);
				worldVertsA1GPU.copyFromHost(worldVertsA1CPU);
				worldNormalsAGPU.copyFromHost(worldNormalsACPU);
				worldVertsB1GPU.copyFromHost(worldVertsB1CPU);

			}





			///clip face B against face A, reduce contacts and append them to a global contact array
			if (1)
			{
				if (clipConvexFacesAndFindContactsCPU)
				{

					//b3AlignedObjectArray<b3Int4> hostPairs;
					//pairs->copyToHost(hostPairs);

					b3AlignedObjectArray<b3Vector3> hostSepNormals;
					m_sepNormals.copyToHost(hostSepNormals);
					b3AlignedObjectArray<int> hostHasSepAxis;
					m_hasSeparatingNormals.copyToHost(hostHasSepAxis);

					b3AlignedObjectArray<b3Int4> hostClippingFaces;
					clippingFacesOutGPU.copyToHost(hostClippingFaces);
					b3AlignedObjectArray<b3Vector3> worldVertsB2CPU;
					worldVertsB2CPU.resize(vertexFaceCapacity*nPairs);
					
					b3AlignedObjectArray<b3Vector3>worldVertsA1CPU;
					worldVertsA1GPU.copyToHost(worldVertsA1CPU);
					b3AlignedObjectArray<b3Vector3> worldNormalsACPU;
					worldNormalsAGPU.copyToHost(worldNormalsACPU);

					b3AlignedObjectArray<b3Vector3>  worldVertsB1CPU;
					worldVertsB1GPU.copyToHost(worldVertsB1CPU);

					/*
					  __global const b3Float4* separatingNormals,
                                                   __global const int* hasSeparatingAxis,
                                                   __global b3Int4* clippingFacesOut,
                                                   __global b3Float4* worldVertsA1,
                                                   __global b3Float4* worldNormalsA1,
                                                   __global b3Float4* worldVertsB1,
                                                   __global b3Float4* worldVertsB2,
                                                    int vertexFaceCapacity,
															int pairIndex
					*/
					for (int i=0;i<nPairs;i++)
					{
						clipFacesAndFindContactsKernel(
							&hostSepNormals.at(0),
							&hostHasSepAxis.at(0),
							&hostClippingFaces.at(0),
							&worldVertsA1CPU.at(0),
							&worldNormalsACPU.at(0),
							&worldVertsB1CPU.at(0),
							&worldVertsB2CPU.at(0),

						vertexFaceCapacity,
							i);
					}
					
					clippingFacesOutGPU.copyFromHost(hostClippingFaces);
					worldVertsB2GPU.copyFromHost(worldVertsB2CPU);

				} else
				{
					B3_PROFILE("clipFacesAndFindContacts");
					//nContacts = m_totalContactsOut.at(0);
					//int h = m_hasSeparatingNormals.at(0);
					//int4 p = clippingFacesOutGPU.at(0);
					b3BufferInfoCL bInfo[] = {
						b3BufferInfoCL( m_sepNormals.getBufferCL()),
						b3BufferInfoCL( m_hasSeparatingNormals.getBufferCL()),
						b3BufferInfoCL( clippingFacesOutGPU.getBufferCL()),
						b3BufferInfoCL( worldVertsA1GPU.getBufferCL()),
						b3BufferInfoCL( worldNormalsAGPU.getBufferCL()),
						b3BufferInfoCL( worldVertsB1GPU.getBufferCL()),
						b3BufferInfoCL( worldVertsB2GPU.getBufferCL())
					};

					b3LauncherCL launcher(m_queue, m_clipFacesAndFindContacts,"m_clipFacesAndFindContacts");
					launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(b3BufferInfoCL) );
					launcher.setConst(vertexFaceCapacity);

					launcher.setConst( nPairs  );
					int debugMode = 0;
					launcher.setConst( debugMode);
					int num = nPairs;
					launcher.launch1D( num);
					clFinish(m_queue);
				} 

				{
					nContacts = m_totalContactsOut.at(0);
					//printf("nContacts = %d\n",nContacts);

					int newContactCapacity = nContacts+nPairs;
					contactOut->reserve(newContactCapacity);

					if (reduceConvexContactsOnGPU)
					{
						{
							B3_PROFILE("newContactReductionKernel");
							b3BufferInfoCL bInfo[] =
							{
								b3BufferInfoCL( pairs->getBufferCL(), true ),
								b3BufferInfoCL( bodyBuf->getBufferCL(),true),
								b3BufferInfoCL( m_sepNormals.getBufferCL()),
								b3BufferInfoCL( m_hasSeparatingNormals.getBufferCL()),
								b3BufferInfoCL( contactOut->getBufferCL()),
								b3BufferInfoCL( clippingFacesOutGPU.getBufferCL()),
								b3BufferInfoCL( worldVertsB2GPU.getBufferCL()),
								b3BufferInfoCL( m_totalContactsOut.getBufferCL())
							};

							b3LauncherCL launcher(m_queue, m_newContactReductionKernel,"m_newContactReductionKernel");
							launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(b3BufferInfoCL) );
							launcher.setConst(vertexFaceCapacity);
							launcher.setConst(newContactCapacity);
							launcher.setConst( nPairs  );
							int num = nPairs;

							launcher.launch1D( num);
						}
						nContacts = m_totalContactsOut.at(0);
						contactOut->resize(nContacts);
					} else
					{

						volatile int nGlobalContactsOut = nContacts;
						b3AlignedObjectArray<b3Int4> hostPairs;
						pairs->copyToHost(hostPairs);
						b3AlignedObjectArray<b3RigidBodyData> hostBodyBuf;
						bodyBuf->copyToHost(hostBodyBuf);
						b3AlignedObjectArray<b3Vector3> hostSepNormals;
						m_sepNormals.copyToHost(hostSepNormals);
						b3AlignedObjectArray<int> hostHasSepAxis;
						m_hasSeparatingNormals.copyToHost(hostHasSepAxis);
						b3AlignedObjectArray<b3Contact4> hostContactsOut;
						contactOut->copyToHost(hostContactsOut);
						hostContactsOut.resize(newContactCapacity);

						b3AlignedObjectArray<b3Int4> hostClippingFaces;
						clippingFacesOutGPU.copyToHost(hostClippingFaces);
						b3AlignedObjectArray<b3Vector3> worldVertsB2CPU;
						worldVertsB2GPU.copyToHost(worldVertsB2CPU);

						for (int i=0;i<nPairs;i++)
						{
							b3NewContactReductionKernel(&hostPairs.at(0),
								&hostBodyBuf.at(0),
								&hostSepNormals.at(0),
								&hostHasSepAxis.at(0),
								&hostContactsOut.at(0),
								&hostClippingFaces.at(0),
								&worldVertsB2CPU.at(0),
								&nGlobalContactsOut,
								vertexFaceCapacity,
								newContactCapacity,
								nPairs,
								i);
						}

						nContacts = nGlobalContactsOut;
						m_totalContactsOut.copyFromHostPointer(&nContacts,1,0,true);
						hostContactsOut.resize(nContacts);
						//printf("contactOut4 (after newContactReductionKernel) = %d\n",nContacts);
						contactOut->copyFromHost(hostContactsOut);
					}
					//                    b3Contact4 pt = contactOut->at(0);
					//                  printf("nContacts = %d\n",nContacts);
				}
			}
		}            
		else//breakupKernel
		{

			if (nPairs)
			{
				b3BufferInfoCL bInfo[] = {
					b3BufferInfoCL( pairs->getBufferCL(), true ), 
					b3BufferInfoCL( bodyBuf->getBufferCL(),true), 
					b3BufferInfoCL( gpuCollidables.getBufferCL(),true), 
					b3BufferInfoCL( convexData.getBufferCL(),true),
					b3BufferInfoCL( gpuVertices.getBufferCL(),true),
					b3BufferInfoCL( gpuUniqueEdges.getBufferCL(),true),
					b3BufferInfoCL( gpuFaces.getBufferCL(),true),
					b3BufferInfoCL( gpuIndices.getBufferCL(),true),
					b3BufferInfoCL( m_sepNormals.getBufferCL()),
					b3BufferInfoCL( m_hasSeparatingNormals.getBufferCL()),
					b3BufferInfoCL( contactOut->getBufferCL()),
					b3BufferInfoCL( m_totalContactsOut.getBufferCL())	
				};
				b3LauncherCL launcher(m_queue, m_clipHullHullKernel,"m_clipHullHullKernel");
				launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(b3BufferInfoCL) );
				launcher.setConst( nPairs  );
				launcher.setConst(maxContactCapacity);

				int num = nPairs;
				launcher.launch1D( num);
				clFinish(m_queue);

				nContacts = m_totalContactsOut.at(0);
				if (nContacts >= maxContactCapacity)
				{
					b3Error("Exceeded contact capacity (%d/%d)\n",nContacts,maxContactCapacity);
					nContacts = maxContactCapacity;
				}
				contactOut->resize(nContacts);
			}
		}


		int nCompoundsPairs = m_gpuCompoundPairs.size();

		if (nCompoundsPairs)
		{
			b3BufferInfoCL bInfo[] = {
				b3BufferInfoCL( m_gpuCompoundPairs.getBufferCL(), true ), 
				b3BufferInfoCL( bodyBuf->getBufferCL(),true), 
				b3BufferInfoCL( gpuCollidables.getBufferCL(),true), 
				b3BufferInfoCL( convexData.getBufferCL(),true),
				b3BufferInfoCL( gpuVertices.getBufferCL(),true),
				b3BufferInfoCL( gpuUniqueEdges.getBufferCL(),true),
				b3BufferInfoCL( gpuFaces.getBufferCL(),true),
				b3BufferInfoCL( gpuIndices.getBufferCL(),true),
				b3BufferInfoCL( gpuChildShapes.getBufferCL(),true),
				b3BufferInfoCL( m_gpuCompoundSepNormals.getBufferCL(),true),
				b3BufferInfoCL( m_gpuHasCompoundSepNormals.getBufferCL(),true),
				b3BufferInfoCL( contactOut->getBufferCL()),
				b3BufferInfoCL( m_totalContactsOut.getBufferCL())	
			};
			b3LauncherCL launcher(m_queue, m_clipCompoundsHullHullKernel,"m_clipCompoundsHullHullKernel");
			launcher.setBuffers( bInfo, sizeof(bInfo)/sizeof(b3BufferInfoCL) );
			launcher.setConst( nCompoundsPairs  );
			launcher.setConst(maxContactCapacity);

			int num = nCompoundsPairs;
			launcher.launch1D( num);
			clFinish(m_queue);

			nContacts = m_totalContactsOut.at(0);
			if (nContacts>maxContactCapacity)
			{

				b3Error("Error: contacts exceeds capacity (%d/%d)\n", nContacts, maxContactCapacity);
				nContacts = maxContactCapacity;
			}
			contactOut->resize(nContacts);
		}//if nCompoundsPairs
		}
	}//contactClippingOnGpu

	//printf("nContacts end = %d\n",nContacts);
	
	//printf("frameCount = %d\n",frameCount++);
}
