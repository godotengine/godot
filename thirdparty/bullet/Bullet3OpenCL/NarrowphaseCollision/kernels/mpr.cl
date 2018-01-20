
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3MprPenetration.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3Contact4Data.h"

#define AppendInc(x, out) out = atomic_inc(x)
#define GET_NPOINTS(x) (x).m_worldNormalOnB.w
#ifdef cl_ext_atomic_counters_32
	#pragma OPENCL EXTENSION cl_ext_atomic_counters_32 : enable
#else
	#define counter32_t volatile __global int*
#endif


__kernel void   mprPenetrationKernel( __global int4* pairs,
																					__global const b3RigidBodyData_t* rigidBodies, 
																					__global const b3Collidable_t* collidables,
																					__global const b3ConvexPolyhedronData_t* convexShapes, 
																					__global const float4* vertices,
																					__global float4* separatingNormals,
																					__global int* hasSeparatingAxis,
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
	
		int shapeIndexA = collidables[collidableIndexA].m_shapeIndex;
		int shapeIndexB = collidables[collidableIndexB].m_shapeIndex;
		
		
		//once the broadphase avoids static-static pairs, we can remove this test
		if ((rigidBodies[bodyIndexA].m_invMass==0) &&(rigidBodies[bodyIndexB].m_invMass==0))
		{
			return;
		}
		

		if ((collidables[collidableIndexA].m_shapeType!=SHAPE_CONVEX_HULL) ||(collidables[collidableIndexB].m_shapeType!=SHAPE_CONVEX_HULL))
		{
			return;
		}

		float depthOut;
		b3Float4 dirOut;
		b3Float4 posOut;


		int res = b3MprPenetration(pairIndex, bodyIndexA, bodyIndexB,rigidBodies,convexShapes,collidables,vertices,separatingNormals,hasSeparatingAxis,&depthOut, &dirOut, &posOut);
		
		
		
		

		if (res==0)
		{
			//add a contact

			int dstIdx;
			AppendInc( nGlobalContactsOut, dstIdx );
			if (dstIdx<contactCapacity)
			{
				pairs[pairIndex].z = dstIdx;
				__global struct b3Contact4Data* c = globalContactsOut + dstIdx;
				c->m_worldNormalOnB = -dirOut;//normal;
				c->m_restituitionCoeffCmp = (0.f*0xffff);c->m_frictionCoeffCmp = (0.7f*0xffff);
				c->m_batchIdx = pairIndex;
				int bodyA = pairs[pairIndex].x;
				int bodyB = pairs[pairIndex].y;
				c->m_bodyAPtrAndSignBit = rigidBodies[bodyA].m_invMass==0 ? -bodyA:bodyA;
				c->m_bodyBPtrAndSignBit = rigidBodies[bodyB].m_invMass==0 ? -bodyB:bodyB;
				c->m_childIndexA = -1;
				c->m_childIndexB = -1;
				//for (int i=0;i<nContacts;i++)
				posOut.w = -depthOut;
				c->m_worldPosB[0] = posOut;//localPoints[contactIdx[i]];
				GET_NPOINTS(*c) = 1;//nContacts;
			}
		}

	}
}

typedef float4 Quaternion;
#define make_float4 (float4)

__inline
float dot3F4(float4 a, float4 b)
{
	float4 a1 = make_float4(a.xyz,0.f);
	float4 b1 = make_float4(b.xyz,0.f);
	return dot(a1, b1);
}




__inline
float4 cross3(float4 a, float4 b)
{
	return cross(a,b);
}
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
Quaternion qtInvert(Quaternion q)
{
	return (Quaternion)(-q.xyz, q.w);
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
float4 transform(const float4* p, const float4* translation, const Quaternion* orientation)
{
	return qtRotate( *orientation, *p ) + (*translation);
}


__inline
float4 qtInvRotate(const Quaternion q, float4 vec)
{
	return qtRotate( qtInvert( q ), vec );
}


inline void project(__global const b3ConvexPolyhedronData_t* hull,  const float4 pos, const float4 orn, 
const float4* dir, __global const float4* vertices, float* min, float* max)
{
	min[0] = FLT_MAX;
	max[0] = -FLT_MAX;
	int numVerts = hull->m_numVertices;

	const float4 localDir = qtInvRotate(orn,*dir);
	float offset = dot(pos,*dir);
	for(int i=0;i<numVerts;i++)
	{
		float dp = dot(vertices[hull->m_vertexOffset+i],localDir);
		if(dp < min[0])	
			min[0] = dp;
		if(dp > max[0])	
			max[0] = dp;
	}
	if(min[0]>max[0])
	{
		float tmp = min[0];
		min[0] = max[0];
		max[0] = tmp;
	}
	min[0] += offset;
	max[0] += offset;
}


bool findSeparatingAxisUnitSphere(	__global const b3ConvexPolyhedronData_t* hullA, __global const b3ConvexPolyhedronData_t* hullB, 
	const float4 posA1,
	const float4 ornA,
	const float4 posB1,
	const float4 ornB,
	const float4 DeltaC2,
	__global const float4* vertices,
	__global const float4* unitSphereDirections,
	int numUnitSphereDirections,
	float4* sep,
	float* dmin)
{
	
	float4 posA = posA1;
	posA.w = 0.f;
	float4 posB = posB1;
	posB.w = 0.f;

	int curPlaneTests=0;

	int curEdgeEdge = 0;
	// Test unit sphere directions
	for (int i=0;i<numUnitSphereDirections;i++)
	{

		float4 crossje;
		crossje = unitSphereDirections[i];	

		if (dot3F4(DeltaC2,crossje)>0)
			crossje *= -1.f;
		{
			float dist;
			bool result = true;
			float Min0,Max0;
			float Min1,Max1;
			project(hullA,posA,ornA,&crossje,vertices, &Min0, &Max0);
			project(hullB,posB,ornB,&crossje,vertices, &Min1, &Max1);
		
			if(Max0<Min1 || Max1<Min0)
				return false;
		
			float d0 = Max0 - Min1;
			float d1 = Max1 - Min0;
			dist = d0<d1 ? d0:d1;
			result = true;
	
			if(dist<*dmin)
			{
				*dmin = dist;
				*sep = crossje;
			}
		}
	}

	
	if((dot3F4(-DeltaC2,*sep))>0.0f)
	{
		*sep = -(*sep);
	}
	return true;
}



__kernel void   findSeparatingAxisUnitSphereKernel( __global const int4* pairs, 
																					__global const b3RigidBodyData_t* rigidBodies, 
																					__global const b3Collidable_t* collidables,
																					__global const b3ConvexPolyhedronData_t* convexShapes, 
																					__global const float4* vertices,
																					__global const float4* unitSphereDirections,
																					__global  float4* separatingNormals,
																					__global  int* hasSeparatingAxis,
																					__global  float* dmins,
																					int numUnitSphereDirections,
																					int numPairs
																					)
{

	int i = get_global_id(0);
	
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
			
			
			int numFacesA = convexShapes[shapeIndexA].m_numFaces;
	
			float dmin = dmins[i];
	
			float4 posA = rigidBodies[bodyIndexA].m_pos;
			posA.w = 0.f;
			float4 posB = rigidBodies[bodyIndexB].m_pos;
			posB.w = 0.f;
			float4 c0local = convexShapes[shapeIndexA].m_localCenter;
			float4 ornA = rigidBodies[bodyIndexA].m_quat;
			float4 c0 = transform(&c0local, &posA, &ornA);
			float4 c1local = convexShapes[shapeIndexB].m_localCenter;
			float4 ornB =rigidBodies[bodyIndexB].m_quat;
			float4 c1 = transform(&c1local,&posB,&ornB);
			const float4 DeltaC2 = c0 - c1;
			float4 sepNormal = separatingNormals[i];
			
			int numEdgeEdgeDirections = convexShapes[shapeIndexA].m_numUniqueEdges*convexShapes[shapeIndexB].m_numUniqueEdges;
			if (numEdgeEdgeDirections>numUnitSphereDirections)
			{
				bool sepEE = findSeparatingAxisUnitSphere(	&convexShapes[shapeIndexA], &convexShapes[shapeIndexB],posA,ornA,
																										posB,ornB,
																										DeltaC2,
																										vertices,unitSphereDirections,numUnitSphereDirections,&sepNormal,&dmin);
				if (!sepEE)
				{
					hasSeparatingAxis[i] = 0;
				} else
				{
					hasSeparatingAxis[i] = 1;
					separatingNormals[i] = sepNormal;
				}
			}
		}		//if (hasSeparatingAxis[i])
	}//(i<numPairs)
}
