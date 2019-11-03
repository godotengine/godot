
#define TRIANGLE_NUM_CONVEX_FACES 5



#pragma OPENCL EXTENSION cl_amd_printf : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable

#ifdef cl_ext_atomic_counters_32
#pragma OPENCL EXTENSION cl_ext_atomic_counters_32 : enable
#else
#define counter32_t volatile __global int*
#endif

#define GET_GROUP_IDX get_group_id(0)
#define GET_LOCAL_IDX get_local_id(0)
#define GET_GLOBAL_IDX get_global_id(0)
#define GET_GROUP_SIZE get_local_size(0)
#define GET_NUM_GROUPS get_num_groups(0)
#define GROUP_LDS_BARRIER barrier(CLK_LOCAL_MEM_FENCE)
#define GROUP_MEM_FENCE mem_fence(CLK_LOCAL_MEM_FENCE)
#define AtomInc(x) atom_inc(&(x))
#define AtomInc1(x, out) out = atom_inc(&(x))
#define AppendInc(x, out) out = atomic_inc(x)
#define AtomAdd(x, value) atom_add(&(x), value)
#define AtomCmpxhg(x, cmp, value) atom_cmpxchg( &(x), cmp, value )
#define AtomXhg(x, value) atom_xchg ( &(x), value )

#define max2 max
#define min2 min

typedef unsigned int u32;



#include "Bullet3Collision/NarrowPhaseCollision/shared/b3Contact4Data.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3ConvexPolyhedronData.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3Collidable.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3RigidBodyData.h"



#define GET_NPOINTS(x) (x).m_worldNormalOnB.w



#define SELECT_UINT4( b, a, condition ) select( b,a,condition )

#define make_float4 (float4)
#define make_float2 (float2)
#define make_uint4 (uint4)
#define make_int4 (int4)
#define make_uint2 (uint2)
#define make_int2 (int2)


__inline
float fastDiv(float numerator, float denominator)
{
	return native_divide(numerator, denominator);	
//	return numerator/denominator;	
}

__inline
float4 fastDiv4(float4 numerator, float4 denominator)
{
	return native_divide(numerator, denominator);	
}


__inline
float4 cross3(float4 a, float4 b)
{
	return cross(a,b);
}

//#define dot3F4 dot

__inline
float dot3F4(float4 a, float4 b)
{
	float4 a1 = make_float4(a.xyz,0.f);
	float4 b1 = make_float4(b.xyz,0.f);
	return dot(a1, b1);
}

__inline
float4 fastNormalize4(float4 v)
{
	return fast_normalize(v);
}


///////////////////////////////////////
//	Quaternion
///////////////////////////////////////

typedef float4 Quaternion;

__inline
Quaternion qtMul(Quaternion a, Quaternion b);

__inline
Quaternion qtNormalize(Quaternion in);

__inline
float4 qtRotate(Quaternion q, float4 vec);

__inline
Quaternion qtInvert(Quaternion q);




__inline
Quaternion qtMul(Quaternion a, Quaternion b)
{
	Quaternion ans;
	ans = cross3( a, b );
	ans += a.w*b+b.w*a;
//	ans.w = a.w*b.w - (a.x*b.x+a.y*b.y+a.z*b.z);
	ans.w = a.w*b.w - dot3F4(a, b);
	return ans;
}

__inline
Quaternion qtNormalize(Quaternion in)
{
	return fastNormalize4(in);
//	in /= length( in );
//	return in;
}
__inline
float4 qtRotate(Quaternion q, float4 vec)
{
	Quaternion qInv = qtInvert( q );
	float4 vcpy = vec;
	vcpy.w = 0.f;
	float4 out = qtMul(qtMul(q,vcpy),qInv);
	return out;
}

__inline
Quaternion qtInvert(Quaternion q)
{
	return (Quaternion)(-q.xyz, q.w);
}

__inline
float4 qtInvRotate(const Quaternion q, float4 vec)
{
	return qtRotate( qtInvert( q ), vec );
}

__inline
float4 transform(const float4* p, const float4* translation, const Quaternion* orientation)
{
	return qtRotate( *orientation, *p ) + (*translation);
}



__inline
float4 normalize3(const float4 a)
{
	float4 n = make_float4(a.x, a.y, a.z, 0.f);
	return fastNormalize4( n );
}


__inline float4 lerp3(const float4 a,const float4 b, float  t)
{
	return make_float4(	a.x + (b.x - a.x) * t,
						a.y + (b.y - a.y) * t,
						a.z + (b.z - a.z) * t,
						0.f);
}



// Clips a face to the back of a plane, return the number of vertices out, stored in ppVtxOut
int clipFaceGlobal(__global const float4* pVtxIn, int numVertsIn, float4 planeNormalWS,float planeEqWS, __global float4* ppVtxOut)
{
	
	int ve;
	float ds, de;
	int numVertsOut = 0;
    //double-check next test
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



// Clips a face to the back of a plane, return the number of vertices out, stored in ppVtxOut
int clipFace(const float4* pVtxIn, int numVertsIn, float4 planeNormalWS,float planeEqWS, float4* ppVtxOut)
{
	
	int ve;
	float ds, de;
	int numVertsOut = 0;
//double-check next test
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


int clipFaceAgainstHull(const float4 separatingNormal, __global const b3ConvexPolyhedronData_t* hullA,  
	const float4 posA, const Quaternion ornA, float4* worldVertsB1, int numWorldVertsB1,
	float4* worldVertsB2, int capacityWorldVertsB2,
	const float minDist, float maxDist,
	__global const float4* vertices,
	__global const b3GpuFace_t* faces,
	__global const int* indices,
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
			const float4 Normal = make_float4(
				faces[hullA->m_faceOffset+face].m_plane.x, 
				faces[hullA->m_faceOffset+face].m_plane.y, 
				faces[hullA->m_faceOffset+face].m_plane.z,0.f);
			const float4 faceANormalWS = qtRotate(ornA,Normal);
		
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

	b3GpuFace_t polyA = faces[hullA->m_faceOffset+closestFaceA];

	// clip polygon to back of planes of all faces of hull A that are adjacent to witness face
	int numVerticesA = polyA.m_numIndices;
	for(int e0=0;e0<numVerticesA;e0++)
	{
		const float4 a = vertices[hullA->m_vertexOffset+indices[polyA.m_indexOffset+e0]];
		const float4 b = vertices[hullA->m_vertexOffset+indices[polyA.m_indexOffset+((e0+1)%numVerticesA)]];
		const float4 edge0 = a - b;
		const float4 WorldEdge0 = qtRotate(ornA,edge0);
		float4 planeNormalA = make_float4(polyA.m_plane.x,polyA.m_plane.y,polyA.m_plane.z,0.f);
		float4 worldPlaneAnormal1 = qtRotate(ornA,planeNormalA);

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
		float4 planeNormalWS = qtRotate(ornA,localPlaneNormal);
		float planeEqWS=localPlaneEq-dot3F4(planeNormalWS,posA);
		for (int i=0;i<numVertsIn;i++)
		{
			float depth = dot3F4(planeNormalWS,pVtxIn[i])+planeEqWS;
			if (depth <=minDist)
			{
				depth = minDist;
			}

			if (depth <=maxDist)
			{
				float4 pointInWorld = pVtxIn[i];
				//resultOut.addContactPoint(separatingNormal,point,depth);
				contactsOut[numContactsOut++] = make_float4(pointInWorld.x,pointInWorld.y,pointInWorld.z,depth);
			}
		}
	}

	return numContactsOut;
}



int clipFaceAgainstHullLocalA(const float4 separatingNormal, const b3ConvexPolyhedronData_t* hullA,  
	const float4 posA, const Quaternion ornA, float4* worldVertsB1, int numWorldVertsB1,
	float4* worldVertsB2, int capacityWorldVertsB2,
	const float minDist, float maxDist,
	const float4* verticesA,
	const b3GpuFace_t* facesA,
	const int* indicesA,
	__global const float4* verticesB,
	__global const b3GpuFace_t* facesB,
	__global const int* indicesB,
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
			const float4 Normal = make_float4(
				facesA[hullA->m_faceOffset+face].m_plane.x, 
				facesA[hullA->m_faceOffset+face].m_plane.y, 
				facesA[hullA->m_faceOffset+face].m_plane.z,0.f);
			const float4 faceANormalWS = qtRotate(ornA,Normal);
		
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

	b3GpuFace_t polyA = facesA[hullA->m_faceOffset+closestFaceA];

	// clip polygon to back of planes of all faces of hull A that are adjacent to witness face
	int numVerticesA = polyA.m_numIndices;
	for(int e0=0;e0<numVerticesA;e0++)
	{
		const float4 a = verticesA[hullA->m_vertexOffset+indicesA[polyA.m_indexOffset+e0]];
		const float4 b = verticesA[hullA->m_vertexOffset+indicesA[polyA.m_indexOffset+((e0+1)%numVerticesA)]];
		const float4 edge0 = a - b;
		const float4 WorldEdge0 = qtRotate(ornA,edge0);
		float4 planeNormalA = make_float4(polyA.m_plane.x,polyA.m_plane.y,polyA.m_plane.z,0.f);
		float4 worldPlaneAnormal1 = qtRotate(ornA,planeNormalA);

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
		float4 planeNormalWS = qtRotate(ornA,localPlaneNormal);
		float planeEqWS=localPlaneEq-dot3F4(planeNormalWS,posA);
		for (int i=0;i<numVertsIn;i++)
		{
			float depth = dot3F4(planeNormalWS,pVtxIn[i])+planeEqWS;
			if (depth <=minDist)
			{
				depth = minDist;
			}

			if (depth <=maxDist)
			{
				float4 pointInWorld = pVtxIn[i];
				//resultOut.addContactPoint(separatingNormal,point,depth);
				contactsOut[numContactsOut++] = make_float4(pointInWorld.x,pointInWorld.y,pointInWorld.z,depth);
			}
		}
	}

	return numContactsOut;
}

int	clipHullAgainstHull(const float4 separatingNormal,
	__global const b3ConvexPolyhedronData_t* hullA, __global const b3ConvexPolyhedronData_t* hullB, 
	const float4 posA, const Quaternion ornA,const float4 posB, const Quaternion ornB, 
	float4* worldVertsB1, float4* worldVertsB2, int capacityWorldVerts,
	const float minDist, float maxDist,
	__global const float4* vertices,
	__global const b3GpuFace_t* faces,
	__global const int* indices,
	float4*	localContactsOut,
	int localContactCapacity)
{
	int numContactsOut = 0;
	int numWorldVertsB1= 0;


	int closestFaceB=-1;
	float dmax = -FLT_MAX;

	{
		for(int face=0;face<hullB->m_numFaces;face++)
		{
			const float4 Normal = make_float4(faces[hullB->m_faceOffset+face].m_plane.x, 
				faces[hullB->m_faceOffset+face].m_plane.y, faces[hullB->m_faceOffset+face].m_plane.z,0.f);
			const float4 WorldNormal = qtRotate(ornB, Normal);
			float d = dot3F4(WorldNormal,separatingNormal);
			if (d > dmax)
			{
				dmax = d;
				closestFaceB = face;
			}
		}
	}

	{
		const b3GpuFace_t polyB = faces[hullB->m_faceOffset+closestFaceB];
		const int numVertices = polyB.m_numIndices;
		for(int e0=0;e0<numVertices;e0++)
		{
			const float4 b = vertices[hullB->m_vertexOffset+indices[polyB.m_indexOffset+e0]];
			worldVertsB1[numWorldVertsB1++] = transform(&b,&posB,&ornB);
		}
	}

	if (closestFaceB>=0)
	{
		numContactsOut = clipFaceAgainstHull(separatingNormal, hullA, 
				posA,ornA,
				worldVertsB1,numWorldVertsB1,worldVertsB2,capacityWorldVerts, minDist, maxDist,vertices,
				faces,
				indices,localContactsOut,localContactCapacity);
	}

	return numContactsOut;
}


int	clipHullAgainstHullLocalA(const float4 separatingNormal,
	const b3ConvexPolyhedronData_t* hullA, __global const b3ConvexPolyhedronData_t* hullB, 
	const float4 posA, const Quaternion ornA,const float4 posB, const Quaternion ornB, 
	float4* worldVertsB1, float4* worldVertsB2, int capacityWorldVerts,
	const float minDist, float maxDist,
	const float4* verticesA,
	const b3GpuFace_t* facesA,
	const int* indicesA,
	__global const float4* verticesB,
	__global const b3GpuFace_t* facesB,
	__global const int* indicesB,
	float4*	localContactsOut,
	int localContactCapacity)
{
	int numContactsOut = 0;
	int numWorldVertsB1= 0;


	int closestFaceB=-1;
	float dmax = -FLT_MAX;

	{
		for(int face=0;face<hullB->m_numFaces;face++)
		{
			const float4 Normal = make_float4(facesB[hullB->m_faceOffset+face].m_plane.x, 
				facesB[hullB->m_faceOffset+face].m_plane.y, facesB[hullB->m_faceOffset+face].m_plane.z,0.f);
			const float4 WorldNormal = qtRotate(ornB, Normal);
			float d = dot3F4(WorldNormal,separatingNormal);
			if (d > dmax)
			{
				dmax = d;
				closestFaceB = face;
			}
		}
	}

	{
		const b3GpuFace_t polyB = facesB[hullB->m_faceOffset+closestFaceB];
		const int numVertices = polyB.m_numIndices;
		for(int e0=0;e0<numVertices;e0++)
		{
			const float4 b = verticesB[hullB->m_vertexOffset+indicesB[polyB.m_indexOffset+e0]];
			worldVertsB1[numWorldVertsB1++] = transform(&b,&posB,&ornB);
		}
	}

	if (closestFaceB>=0)
	{
		numContactsOut = clipFaceAgainstHullLocalA(separatingNormal, hullA, 
				posA,ornA,
				worldVertsB1,numWorldVertsB1,worldVertsB2,capacityWorldVerts, minDist, maxDist,
				verticesA,facesA,indicesA,
				verticesB,facesB,indicesB,
				localContactsOut,localContactCapacity);
	}

	return numContactsOut;
}

#define PARALLEL_SUM(v, n) for(int j=1; j<n; j++) v[0] += v[j];
#define PARALLEL_DO(execution, n) for(int ie=0; ie<n; ie++){execution;}
#define REDUCE_MAX(v, n) {int i=0;\
for(int offset=0; offset<n; offset++) v[i] = (v[i].y > v[i+offset].y)? v[i]: v[i+offset]; }
#define REDUCE_MIN(v, n) {int i=0;\
for(int offset=0; offset<n; offset++) v[i] = (v[i].y < v[i+offset].y)? v[i]: v[i+offset]; }

int extractManifoldSequentialGlobal(__global const float4* p, int nPoints, float4 nearNormal, int4* contactIdx)
{
	if( nPoints == 0 )
        return 0;
    
    if (nPoints <=4)
        return nPoints;
    
    
    if (nPoints >64)
        nPoints = 64;
    
	float4 center = make_float4(0.f);
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


int extractManifoldSequentialGlobalFake(__global const float4* p, int nPoints, float4 nearNormal, int* contactIdx)
{
    contactIdx[0] = 0;
    contactIdx[1] = 1;
    contactIdx[2] = 2;
    contactIdx[3] = 3;
    
	if( nPoints == 0 ) return 0;
    
	nPoints = min2( nPoints, 4 );
    return nPoints;
    
}



int extractManifoldSequential(const float4* p, int nPoints, float4 nearNormal, int* contactIdx)
{
	if( nPoints == 0 ) return 0;

	nPoints = min2( nPoints, 64 );

	float4 center = make_float4(0.f);
	{
		float4 v[64];
		for (int i=0;i<nPoints;i++)
			v[i] = p[i];
		//memcpy( v, p, nPoints*sizeof(float4) );
		PARALLEL_SUM( v, nPoints );
		center = v[0]/(float)nPoints;
	}

	

	{	//	sample 4 directions
		if( nPoints < 4 )
		{
			for(int i=0; i<nPoints; i++) 
				contactIdx[i] = i;
			return nPoints;
		}

		float4 aVector = p[0] - center;
		float4 u = cross3( nearNormal, aVector );
		float4 v = cross3( nearNormal, u );
		u = normalize3( u );
		v = normalize3( v );

		int idx[4];

		float2 max00 = make_float2(0,FLT_MAX);
		{
			//	idx, distance
			{
				{
					int4 a[64];
					for(int ie = 0; ie<nPoints; ie++ )
					{
						
						
						float f;
						float4 r = p[ie]-center;
						f = dot3F4( u, r );
						a[ie].x = ((*(u32*)&f) & 0xffffff00) | (0xff & ie);

						f = dot3F4( -u, r );
						a[ie].y = ((*(u32*)&f) & 0xffffff00) | (0xff & ie);

						f = dot3F4( v, r );
						a[ie].z = ((*(u32*)&f) & 0xffffff00) | (0xff & ie);

						f = dot3F4( -v, r );
						a[ie].w = ((*(u32*)&f) & 0xffffff00) | (0xff & ie);
					}

					for(int ie=0; ie<nPoints; ie++)
					{
						a[0].x = (a[0].x > a[ie].x )? a[0].x: a[ie].x;
						a[0].y = (a[0].y > a[ie].y )? a[0].y: a[ie].y;
						a[0].z = (a[0].z > a[ie].z )? a[0].z: a[ie].z;
						a[0].w = (a[0].w > a[ie].w )? a[0].w: a[ie].w;
					}

					idx[0] = (int)a[0].x & 0xff;
					idx[1] = (int)a[0].y & 0xff;
					idx[2] = (int)a[0].z & 0xff;
					idx[3] = (int)a[0].w & 0xff;
				}
			}

			{
				float2 h[64];
				PARALLEL_DO( h[ie] = make_float2((float)ie, p[ie].w), nPoints );
				REDUCE_MIN( h, nPoints );
				max00 = h[0];
			}
		}

		contactIdx[0] = idx[0];
		contactIdx[1] = idx[1];
		contactIdx[2] = idx[2];
		contactIdx[3] = idx[3];


		return 4;
	}
}



__kernel void   extractManifoldAndAddContactKernel(__global const int4* pairs, 
																	__global const b3RigidBodyData_t* rigidBodies, 
																	__global const float4* closestPointsWorld,
																	__global const float4* separatingNormalsWorld,
																	__global const int* contactCounts,
																	__global const int* contactOffsets,
																	__global struct b3Contact4Data* restrict contactsOut,
																	counter32_t nContactsOut,
																	int contactCapacity,
																	int numPairs,
																	int pairIndex
																	)
{
	int idx = get_global_id(0);
	
	if (idx<numPairs)
	{
		float4 normal = separatingNormalsWorld[idx];
		int nPoints = contactCounts[idx];
		__global const float4* pointsIn = &closestPointsWorld[contactOffsets[idx]];
		float4 localPoints[64];
		for (int i=0;i<nPoints;i++)
		{
			localPoints[i] = pointsIn[i];
		}

		int contactIdx[4];// = {-1,-1,-1,-1};
		contactIdx[0] = -1;
		contactIdx[1] = -1;
		contactIdx[2] = -1;
		contactIdx[3] = -1;

		int nContacts = extractManifoldSequential(localPoints, nPoints, normal, contactIdx);

		int dstIdx;
		AppendInc( nContactsOut, dstIdx );
		if (dstIdx<contactCapacity)
		{
			__global struct b3Contact4Data* c = contactsOut + dstIdx;
			c->m_worldNormalOnB = -normal;
			c->m_restituitionCoeffCmp = (0.f*0xffff);c->m_frictionCoeffCmp = (0.7f*0xffff);
			c->m_batchIdx = idx;
			int bodyA = pairs[pairIndex].x;
			int bodyB = pairs[pairIndex].y;
			c->m_bodyAPtrAndSignBit = rigidBodies[bodyA].m_invMass==0 ? -bodyA:bodyA;
			c->m_bodyBPtrAndSignBit = rigidBodies[bodyB].m_invMass==0 ? -bodyB:bodyB;
			c->m_childIndexA = -1;
			c->m_childIndexB = -1;
			for (int i=0;i<nContacts;i++)
			{
				c->m_worldPosB[i] = localPoints[contactIdx[i]];
			}
			GET_NPOINTS(*c) = nContacts;
		}
	}
}


void	trInverse(float4 translationIn, Quaternion orientationIn,
		float4* translationOut, Quaternion* orientationOut)
{
	*orientationOut = qtInvert(orientationIn);
	*translationOut = qtRotate(*orientationOut, -translationIn);
}

void	trMul(float4 translationA, Quaternion orientationA,
						float4 translationB, Quaternion orientationB,
		float4* translationOut, Quaternion* orientationOut)
{
	*orientationOut = qtMul(orientationA,orientationB);
	*translationOut = transform(&translationB,&translationA,&orientationA);
}




__kernel void   clipHullHullKernel( __global int4* pairs, 
																					__global const b3RigidBodyData_t* rigidBodies, 
																					__global const b3Collidable_t* collidables,
																					__global const b3ConvexPolyhedronData_t* convexShapes, 
																					__global const float4* vertices,
																					__global const float4* uniqueEdges,
																					__global const b3GpuFace_t* faces,
																					__global const int* indices,
																					__global const float4* separatingNormals,
																					__global const int* hasSeparatingAxis,
																					__global struct b3Contact4Data* restrict globalContactsOut,
																					counter32_t nGlobalContactsOut,
																					int numPairs,
																					int contactCapacity)
{

	int i = get_global_id(0);
	int pairIndex = i;
	
	float4 worldVertsB1[64];
	float4 worldVertsB2[64];
	int capacityWorldVerts = 64;	

	float4 localContactsOut[64];
	int localContactCapacity=64;
	
	float minDist = -1e30f;
	float maxDist = 0.02f;

	if (i<numPairs)
	{

		int bodyIndexA = pairs[i].x;
		int bodyIndexB = pairs[i].y;
			
		int collidableIndexA = rigidBodies[bodyIndexA].m_collidableIdx;
		int collidableIndexB = rigidBodies[bodyIndexB].m_collidableIdx;

		if (hasSeparatingAxis[i])
		{

			
			int shapeIndexA = collidables[collidableIndexA].m_shapeIndex;
			int shapeIndexB = collidables[collidableIndexB].m_shapeIndex;
			


		
			int numLocalContactsOut = clipHullAgainstHull(separatingNormals[i],
														&convexShapes[shapeIndexA], &convexShapes[shapeIndexB],
														rigidBodies[bodyIndexA].m_pos,rigidBodies[bodyIndexA].m_quat,
													  rigidBodies[bodyIndexB].m_pos,rigidBodies[bodyIndexB].m_quat,
													  worldVertsB1,worldVertsB2,capacityWorldVerts,
														minDist, maxDist,
														vertices,faces,indices,
														localContactsOut,localContactCapacity);
												
		if (numLocalContactsOut>0)
		{
				float4 normal = -separatingNormals[i];
				int nPoints = numLocalContactsOut;
				float4* pointsIn = localContactsOut;
				int contactIdx[4];// = {-1,-1,-1,-1};

				contactIdx[0] = -1;
				contactIdx[1] = -1;
				contactIdx[2] = -1;
				contactIdx[3] = -1;
		
				int nReducedContacts = extractManifoldSequential(pointsIn, nPoints, normal, contactIdx);
		
				
				int mprContactIndex = pairs[pairIndex].z;

				int dstIdx = mprContactIndex;
				if (dstIdx<0)
				{
					AppendInc( nGlobalContactsOut, dstIdx );
				}

				if (dstIdx<contactCapacity)
				{
					pairs[pairIndex].z = dstIdx;

					__global struct b3Contact4Data* c = globalContactsOut+ dstIdx;
					c->m_worldNormalOnB = -normal;
					c->m_restituitionCoeffCmp = (0.f*0xffff);c->m_frictionCoeffCmp = (0.7f*0xffff);
					c->m_batchIdx = pairIndex;
					int bodyA = pairs[pairIndex].x;
					int bodyB = pairs[pairIndex].y;
					c->m_bodyAPtrAndSignBit = rigidBodies[bodyA].m_invMass==0?-bodyA:bodyA;
					c->m_bodyBPtrAndSignBit = rigidBodies[bodyB].m_invMass==0?-bodyB:bodyB;
					c->m_childIndexA = -1;
					c->m_childIndexB = -1;

					for (int i=0;i<nReducedContacts;i++)
					{
					//this condition means: overwrite contact point, unless at index i==0 we have a valid 'mpr' contact
						if (i>0||(mprContactIndex<0))
						{
							c->m_worldPosB[i] = pointsIn[contactIdx[i]];
						}
					}
					GET_NPOINTS(*c) = nReducedContacts;
				}
				
			}//		if (numContactsOut>0)
		}//		if (hasSeparatingAxis[i])
	}//	if (i<numPairs)

}


__kernel void   clipCompoundsHullHullKernel( __global const int4* gpuCompoundPairs, 
																					__global const b3RigidBodyData_t* rigidBodies, 
																					__global const b3Collidable_t* collidables,
																					__global const b3ConvexPolyhedronData_t* convexShapes, 
																					__global const float4* vertices,
																					__global const float4* uniqueEdges,
																					__global const b3GpuFace_t* faces,
																					__global const int* indices,
																					__global const b3GpuChildShape_t* gpuChildShapes,
																					__global const float4* gpuCompoundSepNormalsOut,
																					__global const int* gpuHasCompoundSepNormalsOut,
																					__global struct b3Contact4Data* restrict globalContactsOut,
																					counter32_t nGlobalContactsOut,
																					int numCompoundPairs, int maxContactCapacity)
{

	int i = get_global_id(0);
	int pairIndex = i;
	
	float4 worldVertsB1[64];
	float4 worldVertsB2[64];
	int capacityWorldVerts = 64;	

	float4 localContactsOut[64];
	int localContactCapacity=64;
	
	float minDist = -1e30f;
	float maxDist = 0.02f;

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
			
			float4 ornA = rigidBodies[bodyIndexA].m_quat;
			float4 posA = rigidBodies[bodyIndexA].m_pos;
			
			float4 ornB = rigidBodies[bodyIndexB].m_quat;
			float4 posB = rigidBodies[bodyIndexB].m_pos;
								
			if (childShapeIndexA >= 0)
			{
				collidableIndexA = gpuChildShapes[childShapeIndexA].m_shapeIndex;
				float4 childPosA = gpuChildShapes[childShapeIndexA].m_childPosition;
				float4 childOrnA = gpuChildShapes[childShapeIndexA].m_childOrientation;
				float4 newPosA = qtRotate(ornA,childPosA)+posA;
				float4 newOrnA = qtMul(ornA,childOrnA);
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
				float4 childOrnB = gpuChildShapes[childShapeIndexB].m_childOrientation;
				float4 newPosB = transform(&childPosB,&posB,&ornB);
				float4 newOrnB = qtMul(ornB,childOrnB);
				posB = newPosB;
				ornB = newOrnB;
			} else
			{
				collidableIndexB = rigidBodies[bodyIndexB].m_collidableIdx;	
			}
			
			int shapeIndexA = collidables[collidableIndexA].m_shapeIndex;
			int shapeIndexB = collidables[collidableIndexB].m_shapeIndex;
		
			int numLocalContactsOut = clipHullAgainstHull(gpuCompoundSepNormalsOut[i],
														&convexShapes[shapeIndexA], &convexShapes[shapeIndexB],
														posA,ornA,
													  posB,ornB,
													  worldVertsB1,worldVertsB2,capacityWorldVerts,
														minDist, maxDist,
														vertices,faces,indices,
														localContactsOut,localContactCapacity);
												
		if (numLocalContactsOut>0)
		{
				float4 normal = -gpuCompoundSepNormalsOut[i];
				int nPoints = numLocalContactsOut;
				float4* pointsIn = localContactsOut;
				int contactIdx[4];// = {-1,-1,-1,-1};

				contactIdx[0] = -1;
				contactIdx[1] = -1;
				contactIdx[2] = -1;
				contactIdx[3] = -1;
		
				int nReducedContacts = extractManifoldSequential(pointsIn, nPoints, normal, contactIdx);
		
				int dstIdx;
				AppendInc( nGlobalContactsOut, dstIdx );
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
						c->m_worldPosB[i] = pointsIn[contactIdx[i]];
					}
					GET_NPOINTS(*c) = nReducedContacts;
				}
				
			}//		if (numContactsOut>0)
		}//		if (gpuHasCompoundSepNormalsOut[i])
	}//	if (i<numCompoundPairs)

}



__kernel void   sphereSphereCollisionKernel( __global const int4* pairs, 
																					__global const b3RigidBodyData_t* rigidBodies, 
																					__global const b3Collidable_t* collidables,
																					__global const float4* separatingNormals,
																					__global const int* hasSeparatingAxis,
																					__global struct b3Contact4Data* restrict globalContactsOut,
																					counter32_t nGlobalContactsOut,
																					int contactCapacity,
																					int numPairs)
{

	int i = get_global_id(0);
	int pairIndex = i;
	
	if (i<numPairs)
	{
		int bodyIndexA = pairs[i].x;
		int bodyIndexB = pairs[i].y;
			
		int collidableIndexA = rigidBodies[bodyIndexA].m_collidableIdx;
		int collidableIndexB = rigidBodies[bodyIndexB].m_collidableIdx;

		if (collidables[collidableIndexA].m_shapeType == SHAPE_SPHERE &&
			collidables[collidableIndexB].m_shapeType == SHAPE_SPHERE)
		{
			//sphere-sphere
			float radiusA = collidables[collidableIndexA].m_radius;
			float radiusB = collidables[collidableIndexB].m_radius;
			float4 posA = rigidBodies[bodyIndexA].m_pos;
			float4 posB = rigidBodies[bodyIndexB].m_pos;

			float4 diff = posA-posB;
			float len = length(diff);
			
			///iff distance positive, don't generate a new contact
			if ( len <= (radiusA+radiusB))
			{
				///distance (negative means penetration)
				float dist = len - (radiusA+radiusB);
				float4 normalOnSurfaceB = make_float4(1.f,0.f,0.f,0.f);
				if (len > 0.00001)
				{
					normalOnSurfaceB = diff / len;
				}
				float4 contactPosB = posB + normalOnSurfaceB*radiusB;
				contactPosB.w = dist;
								
				int dstIdx;
				AppendInc( nGlobalContactsOut, dstIdx );
				if (dstIdx < contactCapacity)
				{
					__global struct b3Contact4Data* c = &globalContactsOut[dstIdx];
					c->m_worldNormalOnB = -normalOnSurfaceB;
					c->m_restituitionCoeffCmp = (0.f*0xffff);c->m_frictionCoeffCmp = (0.7f*0xffff);
					c->m_batchIdx = pairIndex;
					int bodyA = pairs[pairIndex].x;
					int bodyB = pairs[pairIndex].y;
					c->m_bodyAPtrAndSignBit = rigidBodies[bodyA].m_invMass==0?-bodyA:bodyA;
					c->m_bodyBPtrAndSignBit = rigidBodies[bodyB].m_invMass==0?-bodyB:bodyB;
					c->m_worldPosB[0] = contactPosB;
					c->m_childIndexA = -1;
					c->m_childIndexB = -1;

					GET_NPOINTS(*c) = 1;
				}//if (dstIdx < numPairs)
			}//if ( len <= (radiusA+radiusB))
		}//SHAPE_SPHERE SHAPE_SPHERE
	}//if (i<numPairs)
}				

__kernel void   clipHullHullConcaveConvexKernel( __global int4* concavePairsIn,
																					__global const b3RigidBodyData_t* rigidBodies, 
																					__global const b3Collidable_t* collidables,
																					__global const b3ConvexPolyhedronData_t* convexShapes, 
																					__global const float4* vertices,
																					__global const float4* uniqueEdges,
																					__global const b3GpuFace_t* faces,
																					__global const int* indices,
																					__global const b3GpuChildShape_t* gpuChildShapes,
																					__global const float4* separatingNormals,
																					__global struct b3Contact4Data* restrict globalContactsOut,
																					counter32_t nGlobalContactsOut,
																					int contactCapacity,
																					int numConcavePairs)
{

	int i = get_global_id(0);
	int pairIndex = i;
	
	float4 worldVertsB1[64];
	float4 worldVertsB2[64];
	int capacityWorldVerts = 64;	

	float4 localContactsOut[64];
	int localContactCapacity=64;
	
	float minDist = -1e30f;
	float maxDist = 0.02f;

	if (i<numConcavePairs)
	{
		//negative value means that the pair is invalid
		if (concavePairsIn[i].w<0)
			return;

		int bodyIndexA = concavePairsIn[i].x;
		int bodyIndexB = concavePairsIn[i].y;
		int f = concavePairsIn[i].z;
		int childShapeIndexA = f;
		
		int collidableIndexA = rigidBodies[bodyIndexA].m_collidableIdx;
		int collidableIndexB = rigidBodies[bodyIndexB].m_collidableIdx;
		
		int shapeIndexA = collidables[collidableIndexA].m_shapeIndex;
		int shapeIndexB = collidables[collidableIndexB].m_shapeIndex;
		
		///////////////////////////////////////////////////////////////
		
	
		bool overlap = false;
		
		b3ConvexPolyhedronData_t convexPolyhedronA;

	//add 3 vertices of the triangle
		convexPolyhedronA.m_numVertices = 3;
		convexPolyhedronA.m_vertexOffset = 0;
		float4	localCenter = make_float4(0.f,0.f,0.f,0.f);

		b3GpuFace_t face = faces[convexShapes[shapeIndexA].m_faceOffset+f];
		
		float4 verticesA[3];
		for (int i=0;i<3;i++)
		{
			int index = indices[face.m_indexOffset+i];
			float4 vert = vertices[convexShapes[shapeIndexA].m_vertexOffset+index];
			verticesA[i] = vert;
			localCenter += vert;
		}

		float dmin = FLT_MAX;

		int localCC=0;

		//a triangle has 3 unique edges
		convexPolyhedronA.m_numUniqueEdges = 3;
		convexPolyhedronA.m_uniqueEdgesOffset = 0;
		float4 uniqueEdgesA[3];
		
		uniqueEdgesA[0] = (verticesA[1]-verticesA[0]);
		uniqueEdgesA[1] = (verticesA[2]-verticesA[1]);
		uniqueEdgesA[2] = (verticesA[0]-verticesA[2]);


		convexPolyhedronA.m_faceOffset = 0;
                                  
		float4 normal = make_float4(face.m_plane.x,face.m_plane.y,face.m_plane.z,0.f);
                             
		b3GpuFace_t facesA[TRIANGLE_NUM_CONVEX_FACES];
		int indicesA[3+3+2+2+2];
		int curUsedIndices=0;
		int fidx=0;

		//front size of triangle
		{
			facesA[fidx].m_indexOffset=curUsedIndices;
			indicesA[0] = 0;
			indicesA[1] = 1;
			indicesA[2] = 2;
			curUsedIndices+=3;
			float c = face.m_plane.w;
			facesA[fidx].m_plane.x = normal.x;
			facesA[fidx].m_plane.y = normal.y;
			facesA[fidx].m_plane.z = normal.z;
			facesA[fidx].m_plane.w = c;
			facesA[fidx].m_numIndices=3;
		}
		fidx++;
		//back size of triangle
		{
			facesA[fidx].m_indexOffset=curUsedIndices;
			indicesA[3]=2;
			indicesA[4]=1;
			indicesA[5]=0;
			curUsedIndices+=3;
			float c = dot3F4(normal,verticesA[0]);
			float c1 = -face.m_plane.w;
			facesA[fidx].m_plane.x = -normal.x;
			facesA[fidx].m_plane.y = -normal.y;
			facesA[fidx].m_plane.z = -normal.z;
			facesA[fidx].m_plane.w = c;
			facesA[fidx].m_numIndices=3;
		}
		fidx++;

		bool addEdgePlanes = true;
		if (addEdgePlanes)
		{
			int numVertices=3;
			int prevVertex = numVertices-1;
			for (int i=0;i<numVertices;i++)
			{
				float4 v0 = verticesA[i];
				float4 v1 = verticesA[prevVertex];
                                            
				float4 edgeNormal = normalize(cross(normal,v1-v0));
				float c = -dot3F4(edgeNormal,v0);

				facesA[fidx].m_numIndices = 2;
				facesA[fidx].m_indexOffset=curUsedIndices;
				indicesA[curUsedIndices++]=i;
				indicesA[curUsedIndices++]=prevVertex;
                                            
				facesA[fidx].m_plane.x = edgeNormal.x;
				facesA[fidx].m_plane.y = edgeNormal.y;
				facesA[fidx].m_plane.z = edgeNormal.z;
				facesA[fidx].m_plane.w = c;
				fidx++;
				prevVertex = i;
			}
		}
		convexPolyhedronA.m_numFaces = TRIANGLE_NUM_CONVEX_FACES;
		convexPolyhedronA.m_localCenter = localCenter*(1.f/3.f);


		float4 posA = rigidBodies[bodyIndexA].m_pos;
		posA.w = 0.f;
		float4 posB = rigidBodies[bodyIndexB].m_pos;
		posB.w = 0.f;
		float4 ornA = rigidBodies[bodyIndexA].m_quat;
		float4 ornB =rigidBodies[bodyIndexB].m_quat;


		float4 sepAxis = separatingNormals[i];
		
		int shapeTypeB = collidables[collidableIndexB].m_shapeType;
		int childShapeIndexB =-1;
		if (shapeTypeB==SHAPE_COMPOUND_OF_CONVEX_HULLS)
		{
			///////////////////
			///compound shape support
			
			childShapeIndexB = concavePairsIn[pairIndex].w;
			int childColIndexB = gpuChildShapes[childShapeIndexB].m_shapeIndex;
			shapeIndexB = collidables[childColIndexB].m_shapeIndex;
			float4 childPosB = gpuChildShapes[childShapeIndexB].m_childPosition;
			float4 childOrnB = gpuChildShapes[childShapeIndexB].m_childOrientation;
			float4 newPosB = transform(&childPosB,&posB,&ornB);
			float4 newOrnB = qtMul(ornB,childOrnB);
			posB = newPosB;
			ornB = newOrnB;
			
		}
		
		////////////////////////////////////////
		
		
		
		int numLocalContactsOut = clipHullAgainstHullLocalA(sepAxis,
														&convexPolyhedronA, &convexShapes[shapeIndexB],
														posA,ornA,
													  posB,ornB,
													  worldVertsB1,worldVertsB2,capacityWorldVerts,
														minDist, maxDist,
														&verticesA,&facesA,&indicesA,
														vertices,faces,indices,
														localContactsOut,localContactCapacity);
												
		if (numLocalContactsOut>0)
		{
			float4 normal = -separatingNormals[i];
			int nPoints = numLocalContactsOut;
			float4* pointsIn = localContactsOut;
			int contactIdx[4];// = {-1,-1,-1,-1};

			contactIdx[0] = -1;
			contactIdx[1] = -1;
			contactIdx[2] = -1;
			contactIdx[3] = -1;
	
			int nReducedContacts = extractManifoldSequential(pointsIn, nPoints, normal, contactIdx);
	
			int dstIdx;
			AppendInc( nGlobalContactsOut, dstIdx );
			if (dstIdx<contactCapacity)
			{
				__global struct b3Contact4Data* c = globalContactsOut+ dstIdx;
				c->m_worldNormalOnB = -normal;
				c->m_restituitionCoeffCmp = (0.f*0xffff);c->m_frictionCoeffCmp = (0.7f*0xffff);
				c->m_batchIdx = pairIndex;
				int bodyA = concavePairsIn[pairIndex].x;
				int bodyB = concavePairsIn[pairIndex].y;
				c->m_bodyAPtrAndSignBit = rigidBodies[bodyA].m_invMass==0?-bodyA:bodyA;
				c->m_bodyBPtrAndSignBit = rigidBodies[bodyB].m_invMass==0?-bodyB:bodyB;
				c->m_childIndexA = childShapeIndexA;
				c->m_childIndexB = childShapeIndexB;
				for (int i=0;i<nReducedContacts;i++)
				{
					c->m_worldPosB[i] = pointsIn[contactIdx[i]];
				}
				GET_NPOINTS(*c) = nReducedContacts;
			}
				
		}//		if (numContactsOut>0)
	}//	if (i<numPairs)
}






int	findClippingFaces(const float4 separatingNormal,
                      __global const b3ConvexPolyhedronData_t* hullA, __global const b3ConvexPolyhedronData_t* hullB,
                      const float4 posA, const Quaternion ornA,const float4 posB, const Quaternion ornB,
                       __global float4* worldVertsA1,
                      __global float4* worldNormalsA1,
                      __global float4* worldVertsB1,
                      int capacityWorldVerts,
                      const float minDist, float maxDist,
                      __global const float4* vertices,
                      __global const b3GpuFace_t* faces,
                      __global const int* indices,
                      __global int4* clippingFaces, int pairIndex)
{
	int numContactsOut = 0;
	int numWorldVertsB1= 0;
    
    
	int closestFaceB=-1;
	float dmax = -FLT_MAX;
    
	{
		for(int face=0;face<hullB->m_numFaces;face++)
		{
			const float4 Normal = make_float4(faces[hullB->m_faceOffset+face].m_plane.x,
                                              faces[hullB->m_faceOffset+face].m_plane.y, faces[hullB->m_faceOffset+face].m_plane.z,0.f);
			const float4 WorldNormal = qtRotate(ornB, Normal);
			float d = dot3F4(WorldNormal,separatingNormal);
			if (d > dmax)
			{
				dmax = d;
				closestFaceB = face;
			}
		}
	}
    
	{
		const b3GpuFace_t polyB = faces[hullB->m_faceOffset+closestFaceB];
		const int numVertices = polyB.m_numIndices;
		for(int e0=0;e0<numVertices;e0++)
		{
			const float4 b = vertices[hullB->m_vertexOffset+indices[polyB.m_indexOffset+e0]];
			worldVertsB1[pairIndex*capacityWorldVerts+numWorldVertsB1++] = transform(&b,&posB,&ornB);
		}
	}
    
    int closestFaceA=-1;
	{
		float dmin = FLT_MAX;
		for(int face=0;face<hullA->m_numFaces;face++)
		{
			const float4 Normal = make_float4(
                                              faces[hullA->m_faceOffset+face].m_plane.x,
                                              faces[hullA->m_faceOffset+face].m_plane.y,
                                              faces[hullA->m_faceOffset+face].m_plane.z,
                                              0.f);
			const float4 faceANormalWS = qtRotate(ornA,Normal);
            
			float d = dot3F4(faceANormalWS,separatingNormal);
			if (d < dmin)
			{
				dmin = d;
				closestFaceA = face;
                worldNormalsA1[pairIndex] = faceANormalWS;
			}
		}
	}
    
    int numVerticesA = faces[hullA->m_faceOffset+closestFaceA].m_numIndices;
	for(int e0=0;e0<numVerticesA;e0++)
	{
        const float4 a = vertices[hullA->m_vertexOffset+indices[faces[hullA->m_faceOffset+closestFaceA].m_indexOffset+e0]];
        worldVertsA1[pairIndex*capacityWorldVerts+e0] = transform(&a, &posA,&ornA);
    }
    
    clippingFaces[pairIndex].x = closestFaceA;
    clippingFaces[pairIndex].y = closestFaceB;
    clippingFaces[pairIndex].z = numVerticesA;
    clippingFaces[pairIndex].w = numWorldVertsB1;
    
    
	return numContactsOut;
}



int clipFaces(__global float4* worldVertsA1,
              __global float4* worldNormalsA1,
              __global float4* worldVertsB1,
              __global float4* worldVertsB2, 
              int capacityWorldVertsB2,
              const float minDist, float maxDist,
              __global int4* clippingFaces,
              int pairIndex)
{
	int numContactsOut = 0;
    
    int closestFaceA = clippingFaces[pairIndex].x;
    int closestFaceB = clippingFaces[pairIndex].y;
	int numVertsInA = clippingFaces[pairIndex].z;
	int numVertsInB = clippingFaces[pairIndex].w;
    
	int numVertsOut = 0;
    
	if (closestFaceA<0)
		return numContactsOut;
    
    __global float4* pVtxIn = &worldVertsB1[pairIndex*capacityWorldVertsB2];
    __global float4* pVtxOut = &worldVertsB2[pairIndex*capacityWorldVertsB2];
    
    
	
	// clip polygon to back of planes of all faces of hull A that are adjacent to witness face
    
	for(int e0=0;e0<numVertsInA;e0++)
	{
		const float4 aw = worldVertsA1[pairIndex*capacityWorldVertsB2+e0];
		const float4 bw = worldVertsA1[pairIndex*capacityWorldVertsB2+((e0+1)%numVertsInA)];
		const float4 WorldEdge0 = aw - bw;
		float4 worldPlaneAnormal1 = worldNormalsA1[pairIndex];
		float4 planeNormalWS1 = -cross3(WorldEdge0,worldPlaneAnormal1);
		float4 worldA1 = aw;
		float planeEqWS1 = -dot3F4(worldA1,planeNormalWS1);
		float4 planeNormalWS = planeNormalWS1;
		float planeEqWS=planeEqWS1;
		numVertsOut = clipFaceGlobal(pVtxIn, numVertsInB, planeNormalWS,planeEqWS, pVtxOut);
		__global float4* tmp = pVtxOut;
		pVtxOut = pVtxIn;
		pVtxIn = tmp;
		numVertsInB = numVertsOut;
		numVertsOut = 0;
	}
    
    //float4 planeNormalWS = worldNormalsA1[pairIndex];
    //float planeEqWS=-dot3F4(planeNormalWS,worldVertsA1[pairIndex*capacityWorldVertsB2]);


    
    /*for (int i=0;i<numVertsInB;i++)
    {
        pVtxOut[i] = pVtxIn[i];
    }*/
    
    
    
    
    //numVertsInB=0;
	
    float4 planeNormalWS = worldNormalsA1[pairIndex];
    float planeEqWS=-dot3F4(planeNormalWS,worldVertsA1[pairIndex*capacityWorldVertsB2]);

    for (int i=0;i<numVertsInB;i++)
    {
        float depth = dot3F4(planeNormalWS,pVtxIn[i])+planeEqWS;
        if (depth <=minDist)
        {
            depth = minDist;
        }
        
        if (depth <=maxDist)
        {
            float4 pointInWorld = pVtxIn[i];
            pVtxOut[numContactsOut++] = make_float4(pointInWorld.x,pointInWorld.y,pointInWorld.z,depth);
        }
    }
   
    clippingFaces[pairIndex].w =numContactsOut;
   
    
	return numContactsOut;

}




__kernel void   findClippingFacesKernel(  __global const int4* pairs,
                                        __global const b3RigidBodyData_t* rigidBodies,
                                        __global const b3Collidable_t* collidables,
                                        __global const b3ConvexPolyhedronData_t* convexShapes,
                                        __global const float4* vertices,
                                        __global const float4* uniqueEdges,
                                        __global const b3GpuFace_t* faces,
                                        __global const int* indices,
                                        __global const float4* separatingNormals,
                                        __global const int* hasSeparatingAxis,
                                        __global int4* clippingFacesOut,
                                        __global float4* worldVertsA1,
                                        __global float4* worldNormalsA1,
                                        __global float4* worldVertsB1,
                                        int capacityWorldVerts,
                                        int numPairs
                                        )
{
    
	int i = get_global_id(0);
	int pairIndex = i;
    
	
	float minDist = -1e30f;
	float maxDist = 0.02f;
    
	if (i<numPairs)
	{
        
		if (hasSeparatingAxis[i])
		{
            
			int bodyIndexA = pairs[i].x;
			int bodyIndexB = pairs[i].y;
			
			int collidableIndexA = rigidBodies[bodyIndexA].m_collidableIdx;
			int collidableIndexB = rigidBodies[bodyIndexB].m_collidableIdx;
			
			int shapeIndexA = collidables[collidableIndexA].m_shapeIndex;
			int shapeIndexB = collidables[collidableIndexB].m_shapeIndex;
			
            
            
			int numLocalContactsOut = findClippingFaces(separatingNormals[i],
                                                        &convexShapes[shapeIndexA], &convexShapes[shapeIndexB],
                                                        rigidBodies[bodyIndexA].m_pos,rigidBodies[bodyIndexA].m_quat,
                                                        rigidBodies[bodyIndexB].m_pos,rigidBodies[bodyIndexB].m_quat,
                                                        worldVertsA1,
                                                        worldNormalsA1,
                                                        worldVertsB1,capacityWorldVerts,
                                                        minDist, maxDist,
                                                        vertices,faces,indices,
                                                        clippingFacesOut,i);
            
            
		}//		if (hasSeparatingAxis[i])
	}//	if (i<numPairs)
    
}




__kernel void   clipFacesAndFindContactsKernel(    __global const float4* separatingNormals,
                                                   __global const int* hasSeparatingAxis,
                                                   __global int4* clippingFacesOut,
                                                   __global float4* worldVertsA1,
                                                   __global float4* worldNormalsA1,
                                                   __global float4* worldVertsB1,
                                                   __global float4* worldVertsB2,
                                                    int vertexFaceCapacity,
                                                   int numPairs,
					                                        int debugMode
                                                   )
{
    int i = get_global_id(0);
	int pairIndex = i;
	
    
	float minDist = -1e30f;
	float maxDist = 0.02f;
    
	if (i<numPairs)
	{
        
		if (hasSeparatingAxis[i])
		{
            
//			int bodyIndexA = pairs[i].x;
	//		int bodyIndexB = pairs[i].y;
		    
            int numLocalContactsOut = 0;

            int capacityWorldVertsB2 = vertexFaceCapacity;
            
            __global float4* pVtxIn = &worldVertsB1[pairIndex*capacityWorldVertsB2];
            __global float4* pVtxOut = &worldVertsB2[pairIndex*capacityWorldVertsB2];
            

            {
                __global int4* clippingFaces = clippingFacesOut;
            
                
                int closestFaceA = clippingFaces[pairIndex].x;
                int closestFaceB = clippingFaces[pairIndex].y;
                int numVertsInA = clippingFaces[pairIndex].z;
                int numVertsInB = clippingFaces[pairIndex].w;
                
                int numVertsOut = 0;
                
                if (closestFaceA>=0)
                {
                    
                    
                    
                    // clip polygon to back of planes of all faces of hull A that are adjacent to witness face
                    
                    for(int e0=0;e0<numVertsInA;e0++)
                    {
                        const float4 aw = worldVertsA1[pairIndex*capacityWorldVertsB2+e0];
                        const float4 bw = worldVertsA1[pairIndex*capacityWorldVertsB2+((e0+1)%numVertsInA)];
                        const float4 WorldEdge0 = aw - bw;
                        float4 worldPlaneAnormal1 = worldNormalsA1[pairIndex];
                        float4 planeNormalWS1 = -cross3(WorldEdge0,worldPlaneAnormal1);
                        float4 worldA1 = aw;
                        float planeEqWS1 = -dot3F4(worldA1,planeNormalWS1);
                        float4 planeNormalWS = planeNormalWS1;
                        float planeEqWS=planeEqWS1;
                        numVertsOut = clipFaceGlobal(pVtxIn, numVertsInB, planeNormalWS,planeEqWS, pVtxOut);
                        __global float4* tmp = pVtxOut;
                        pVtxOut = pVtxIn;
                        pVtxIn = tmp;
                        numVertsInB = numVertsOut;
                        numVertsOut = 0;
                    }
                    
                    float4 planeNormalWS = worldNormalsA1[pairIndex];
                    float planeEqWS=-dot3F4(planeNormalWS,worldVertsA1[pairIndex*capacityWorldVertsB2]);
                    
                    for (int i=0;i<numVertsInB;i++)
                    {
                        float depth = dot3F4(planeNormalWS,pVtxIn[i])+planeEqWS;
                        if (depth <=minDist)
                        {
                            depth = minDist;
                        }
                        
                        if (depth <=maxDist)
                        {
                            float4 pointInWorld = pVtxIn[i];
                            pVtxOut[numLocalContactsOut++] = make_float4(pointInWorld.x,pointInWorld.y,pointInWorld.z,depth);
                        }
                    }
                    
                }
                clippingFaces[pairIndex].w =numLocalContactsOut;
                

            }
            
            for (int i=0;i<numLocalContactsOut;i++)
                pVtxIn[i] = pVtxOut[i];
                
		}//		if (hasSeparatingAxis[i])
	}//	if (i<numPairs)
    
}





__kernel void   newContactReductionKernel( __global int4* pairs,
                                                   __global const b3RigidBodyData_t* rigidBodies,
                                                   __global const float4* separatingNormals,
                                                   __global const int* hasSeparatingAxis,
                                                   __global struct b3Contact4Data* globalContactsOut,
                                                   __global int4* clippingFaces,
                                                   __global float4* worldVertsB2,
                                                   volatile __global int* nGlobalContactsOut,
                                                   int vertexFaceCapacity,
												   int contactCapacity,
                                                   int numPairs
                                                   )
{
    int i = get_global_id(0);
	int pairIndex = i;
	
    int4 contactIdx;
    contactIdx=make_int4(0,1,2,3);
    
	if (i<numPairs)
	{
        
		if (hasSeparatingAxis[i])
		{
            
			
            
            
			int nPoints = clippingFaces[pairIndex].w;
           
            if (nPoints>0)
            {

                 __global float4* pointsIn = &worldVertsB2[pairIndex*vertexFaceCapacity];
                float4 normal = -separatingNormals[i];
                
                int nReducedContacts = extractManifoldSequentialGlobal(pointsIn, nPoints, normal, &contactIdx);
            
				int mprContactIndex = pairs[pairIndex].z;

                int dstIdx = mprContactIndex;

				if (dstIdx<0)
				{
	                AppendInc( nGlobalContactsOut, dstIdx );
				}
//#if 0
                
				if (dstIdx < contactCapacity)
				{

					__global struct b3Contact4Data* c = &globalContactsOut[dstIdx];
					c->m_worldNormalOnB = -normal;
					c->m_restituitionCoeffCmp = (0.f*0xffff);c->m_frictionCoeffCmp = (0.7f*0xffff);
					c->m_batchIdx = pairIndex;
					int bodyA = pairs[pairIndex].x;
					int bodyB = pairs[pairIndex].y;

					pairs[pairIndex].w = dstIdx;

					c->m_bodyAPtrAndSignBit = rigidBodies[bodyA].m_invMass==0?-bodyA:bodyA;
					c->m_bodyBPtrAndSignBit = rigidBodies[bodyB].m_invMass==0?-bodyB:bodyB;
                    c->m_childIndexA =-1;
					c->m_childIndexB =-1;

                    switch (nReducedContacts)
                    {
                        case 4:
                            c->m_worldPosB[3] = pointsIn[contactIdx.w];
                        case 3:
                            c->m_worldPosB[2] = pointsIn[contactIdx.z];
                        case 2:
                            c->m_worldPosB[1] = pointsIn[contactIdx.y];
                        case 1:
							if (mprContactIndex<0)//test
	                            c->m_worldPosB[0] = pointsIn[contactIdx.x];
                        default:
                        {
                        }
                    };
                    
					GET_NPOINTS(*c) = nReducedContacts;
                    
                 }
                 
                
//#endif
				
			}//		if (numContactsOut>0)
		}//		if (hasSeparatingAxis[i])
	}//	if (i<numPairs)

    
    
}
