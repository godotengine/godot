#include "Bullet3Collision/NarrowPhaseCollision/shared/b3Contact4Data.h"

#define SHAPE_CONVEX_HULL 3
#define SHAPE_PLANE 4
#define SHAPE_CONCAVE_TRIMESH 5
#define SHAPE_COMPOUND_OF_CONVEX_HULLS 6
#define SHAPE_SPHERE 7


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




typedef struct 
{
	union
	{
		float4	m_min;
		float   m_minElems[4];
		int			m_minIndices[4];
	};
	union
	{
		float4	m_max;
		float   m_maxElems[4];
		int			m_maxIndices[4];
	};
} btAabbCL;

///keep this in sync with btCollidable.h
typedef struct
{
	int m_numChildShapes;
	float m_radius;
	int m_shapeType;
	int m_shapeIndex;
	
} btCollidableGpu;

typedef struct
{
	float4	m_childPosition;
	float4	m_childOrientation;
	int m_shapeIndex;
	int m_unused0;
	int m_unused1;
	int m_unused2;
} btGpuChildShape;

#define GET_NPOINTS(x) (x).m_worldNormalOnB.w

typedef struct
{
	float4 m_pos;
	float4 m_quat;
	float4 m_linVel;
	float4 m_angVel;

	u32 m_collidableIdx;	
	float m_invMass;
	float m_restituitionCoeff;
	float m_frictionCoeff;
} BodyData;


typedef struct  
{
	float4		m_localCenter;
	float4		m_extents;
	float4		mC;
	float4		mE;
	
	float			m_radius;
	int	m_faceOffset;
	int m_numFaces;
	int	m_numVertices;
	
	int m_vertexOffset;
	int	m_uniqueEdgesOffset;
	int	m_numUniqueEdges;
	int m_unused;

} ConvexPolyhedronCL;

typedef struct
{
	float4 m_plane;
	int m_indexOffset;
	int m_numIndices;
} btGpuFace;

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


float signedDistanceFromPointToPlane(float4 point, float4 planeEqn, float4* closestPointOnFace)
{
	float4 n = (float4)(planeEqn.x, planeEqn.y, planeEqn.z, 0);
	float dist = dot3F4(n, point) + planeEqn.w;
	*closestPointOnFace = point - dist * n;
	return dist;
}



inline bool IsPointInPolygon(float4 p, 
							const btGpuFace* face,
							__global const float4* baseVertex,
							__global const  int* convexIndices,
							float4* out)
{
    float4 a;
    float4 b;
    float4 ab;
    float4 ap;
    float4 v;

	float4 plane = make_float4(face->m_plane.x,face->m_plane.y,face->m_plane.z,0.f);
	
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

        if (dot(ap, v) > 0.f)
        {
            float ab_m2 = dot(ab, ab);
            float rt = ab_m2 != 0.f ? dot(ab, ap) / ab_m2 : 0.f;
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




void	computeContactSphereConvex(int pairIndex,
																int bodyIndexA, int bodyIndexB, 
																int collidableIndexA, int collidableIndexB, 
																__global const BodyData* rigidBodies, 
																__global const btCollidableGpu* collidables,
																__global const ConvexPolyhedronCL* convexShapes,
																__global const float4* convexVertices,
																__global const int* convexIndices,
																__global const btGpuFace* faces,
																__global struct b3Contact4Data* restrict globalContactsOut,
																counter32_t nGlobalContactsOut,
																int maxContactCapacity,
																float4 spherePos2,
																float radius,
																float4 pos,
																float4 quat
																)
{

	float4 invPos;
	float4 invOrn;

	trInverse(pos,quat, &invPos,&invOrn);

	float4 spherePos = transform(&spherePos2,&invPos,&invOrn);

	int shapeIndex = collidables[collidableIndexB].m_shapeIndex;
	int numFaces = convexShapes[shapeIndex].m_numFaces;
	float4 closestPnt = (float4)(0, 0, 0, 0);
	float4 hitNormalWorld = (float4)(0, 0, 0, 0);
	float minDist = -1000000.f;
	bool bCollide = true;

	for ( int f = 0; f < numFaces; f++ )
	{
		btGpuFace face = faces[convexShapes[shapeIndex].m_faceOffset+f];

		// set up a plane equation 
		float4 planeEqn;
		float4 n1 = face.m_plane;
		n1.w = 0.f;
		planeEqn = n1;
		planeEqn.w = face.m_plane.w;
		
	
		// compute a signed distance from the vertex in cloth to the face of rigidbody.
		float4 pntReturn;
		float dist = signedDistanceFromPointToPlane(spherePos, planeEqn, &pntReturn);

		// If the distance is positive, the plane is a separating plane. 
		if ( dist > radius )
		{
			bCollide = false;
			break;
		}


		if (dist>0)
		{
			//might hit an edge or vertex
			float4 out;
			float4 zeroPos = make_float4(0,0,0,0);

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
					hitNormalWorld = planeEqn;
					
				}
			} else
			{
				float4 tmp = spherePos-out;
				float l2 = dot(tmp,tmp);
				if (l2<radius*radius)
				{
					dist  = sqrt(l2);
					if (dist>minDist)
					{
						minDist = dist;
						closestPnt = out;
						hitNormalWorld = tmp/dist;
						
					}
					
				} else
				{
					bCollide = false;
					break;
				}
			}
		} else
		{
			if ( dist > minDist )
			{
				minDist = dist;
				closestPnt = pntReturn;
				hitNormalWorld.xyz = planeEqn.xyz;
			}
		}
		
	}

	

	if (bCollide && minDist > -10000)
	{
		float4 normalOnSurfaceB1 = qtRotate(quat,-hitNormalWorld);
		float4 pOnB1 = transform(&closestPnt,&pos,&quat);
		
		float actualDepth = minDist-radius;
		if (actualDepth<=0.f)
		{
			

			pOnB1.w = actualDepth;

			int dstIdx;
			AppendInc( nGlobalContactsOut, dstIdx );
		
			
			if (1)//dstIdx < maxContactCapacity)
			{
				__global struct b3Contact4Data* c = &globalContactsOut[dstIdx];
				c->m_worldNormalOnB = -normalOnSurfaceB1;
				c->m_restituitionCoeffCmp = (0.f*0xffff);c->m_frictionCoeffCmp = (0.7f*0xffff);
				c->m_batchIdx = pairIndex;
				c->m_bodyAPtrAndSignBit = rigidBodies[bodyIndexA].m_invMass==0?-bodyIndexA:bodyIndexA;
				c->m_bodyBPtrAndSignBit = rigidBodies[bodyIndexB].m_invMass==0?-bodyIndexB:bodyIndexB;
				c->m_worldPosB[0] = pOnB1;
				c->m_childIndexA = -1;
				c->m_childIndexB = -1;

				GET_NPOINTS(*c) = 1;
			} 

		}
	}//if (hasCollision)

}
							


int extractManifoldSequential(const float4* p, int nPoints, float4 nearNormal, int4* contactIdx)
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

#define MAX_PLANE_CONVEX_POINTS 64

int computeContactPlaneConvex(int pairIndex,
								int bodyIndexA, int bodyIndexB, 
								int collidableIndexA, int collidableIndexB, 
								__global const BodyData* rigidBodies, 
								__global const btCollidableGpu*collidables,
								__global const ConvexPolyhedronCL* convexShapes,
								__global const float4* convexVertices,
								__global const int* convexIndices,
								__global const btGpuFace* faces,
								__global struct b3Contact4Data* restrict globalContactsOut,
								counter32_t nGlobalContactsOut,
								int maxContactCapacity,
								float4 posB,
								Quaternion ornB
								)
{
	int resultIndex=-1;

		int shapeIndex = collidables[collidableIndexB].m_shapeIndex;
	__global const ConvexPolyhedronCL* hullB = &convexShapes[shapeIndex];
	
	float4 posA;
	posA = rigidBodies[bodyIndexA].m_pos;
	Quaternion ornA;
	ornA = rigidBodies[bodyIndexA].m_quat;

	int numContactsOut = 0;
	int numWorldVertsB1= 0;

	float4 planeEq;
	 planeEq = faces[collidables[collidableIndexA].m_shapeIndex].m_plane;
	float4 planeNormal = make_float4(planeEq.x,planeEq.y,planeEq.z,0.f);
	float4 planeNormalWorld;
	planeNormalWorld = qtRotate(ornA,planeNormal);
	float planeConstant = planeEq.w;
	
	float4 invPosA;Quaternion invOrnA;
	float4 convexInPlaneTransPos1; Quaternion convexInPlaneTransOrn1;
	{
		
		trInverse(posA,ornA,&invPosA,&invOrnA);
		trMul(invPosA,invOrnA,posB,ornB,&convexInPlaneTransPos1,&convexInPlaneTransOrn1);
	}
	float4 invPosB;Quaternion invOrnB;
	float4 planeInConvexPos1;	Quaternion planeInConvexOrn1;
	{
		
		trInverse(posB,ornB,&invPosB,&invOrnB);
		trMul(invPosB,invOrnB,posA,ornA,&planeInConvexPos1,&planeInConvexOrn1);	
	}

	
	float4 planeNormalInConvex = qtRotate(planeInConvexOrn1,-planeNormal);
	float maxDot = -1e30;
	int hitVertex=-1;
	float4 hitVtx;



	float4 contactPoints[MAX_PLANE_CONVEX_POINTS];
	int numPoints = 0;

	int4 contactIdx;
	contactIdx=make_int4(0,1,2,3);
    
	
	for (int i=0;i<hullB->m_numVertices;i++)
	{
		float4 vtx = convexVertices[hullB->m_vertexOffset+i];
		float curDot = dot(vtx,planeNormalInConvex);


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
			float4 vtxWorld = transform(&vtx, &posB, &ornB);
			float4 vtxInPlane = transform(&vtxWorld, &invPosA, &invOrnA);//oplaneTransform.inverse()*vtxWorld;
			float dist = dot(planeNormal,vtxInPlane)-planeConstant;
			if (dist<0.f)
			{
				vtxWorld.w = dist;
				contactPoints[numPoints] = vtxWorld;
				numPoints++;
			}
		}

	}

	int numReducedPoints  = numPoints;
	if (numPoints>4)
	{
		numReducedPoints = extractManifoldSequential( contactPoints, numPoints, planeNormalInConvex, &contactIdx);
	}

	if (numReducedPoints>0)
	{
		int dstIdx;
	    AppendInc( nGlobalContactsOut, dstIdx );

		if (dstIdx < maxContactCapacity)
		{
			resultIndex = dstIdx;
			__global struct b3Contact4Data* c = &globalContactsOut[dstIdx];
			c->m_worldNormalOnB = -planeNormalWorld;
			//c->setFrictionCoeff(0.7);
			//c->setRestituitionCoeff(0.f);
			c->m_restituitionCoeffCmp = (0.f*0xffff);c->m_frictionCoeffCmp = (0.7f*0xffff);
			c->m_batchIdx = pairIndex;
			c->m_bodyAPtrAndSignBit = rigidBodies[bodyIndexA].m_invMass==0?-bodyIndexA:bodyIndexA;
			c->m_bodyBPtrAndSignBit = rigidBodies[bodyIndexB].m_invMass==0?-bodyIndexB:bodyIndexB;
			c->m_childIndexA = -1;
			c->m_childIndexB = -1;

			switch (numReducedPoints)
            {
                case 4:
                    c->m_worldPosB[3] = contactPoints[contactIdx.w];
                case 3:
                    c->m_worldPosB[2] = contactPoints[contactIdx.z];
                case 2:
                    c->m_worldPosB[1] = contactPoints[contactIdx.y];
                case 1:
                    c->m_worldPosB[0] = contactPoints[contactIdx.x];
                default:
                {
                }
            };
			
			GET_NPOINTS(*c) = numReducedPoints;
		}//if (dstIdx < numPairs)
	}	

	return resultIndex;
}


void	computeContactPlaneSphere(int pairIndex,
																int bodyIndexA, int bodyIndexB, 
																int collidableIndexA, int collidableIndexB, 
																__global const BodyData* rigidBodies, 
																__global const btCollidableGpu* collidables,
																__global const btGpuFace* faces,
																__global struct b3Contact4Data* restrict globalContactsOut,
																counter32_t nGlobalContactsOut,
																int maxContactCapacity)
{
	float4 planeEq = faces[collidables[collidableIndexA].m_shapeIndex].m_plane;
	float radius = collidables[collidableIndexB].m_radius;
	float4 posA1 = rigidBodies[bodyIndexA].m_pos;
	float4 ornA1 = rigidBodies[bodyIndexA].m_quat;
	float4 posB1 = rigidBodies[bodyIndexB].m_pos;
	float4 ornB1 = rigidBodies[bodyIndexB].m_quat;
	
	bool hasCollision = false;
	float4 planeNormal1 = make_float4(planeEq.x,planeEq.y,planeEq.z,0.f);
	float planeConstant = planeEq.w;
	float4 convexInPlaneTransPos1; Quaternion convexInPlaneTransOrn1;
	{
		float4 invPosA;Quaternion invOrnA;
		trInverse(posA1,ornA1,&invPosA,&invOrnA);
		trMul(invPosA,invOrnA,posB1,ornB1,&convexInPlaneTransPos1,&convexInPlaneTransOrn1);
	}
	float4 planeInConvexPos1;	Quaternion planeInConvexOrn1;
	{
		float4 invPosB;Quaternion invOrnB;
		trInverse(posB1,ornB1,&invPosB,&invOrnB);
		trMul(invPosB,invOrnB,posA1,ornA1,&planeInConvexPos1,&planeInConvexOrn1);	
	}
	float4 vtx1 = qtRotate(planeInConvexOrn1,-planeNormal1)*radius;
	float4 vtxInPlane1 = transform(&vtx1,&convexInPlaneTransPos1,&convexInPlaneTransOrn1);
	float distance = dot3F4(planeNormal1,vtxInPlane1) - planeConstant;
	hasCollision = distance < 0.f;//m_manifoldPtr->getContactBreakingThreshold();
	if (hasCollision)
	{
		float4 vtxInPlaneProjected1 = vtxInPlane1 -   distance*planeNormal1;
		float4 vtxInPlaneWorld1 = transform(&vtxInPlaneProjected1,&posA1,&ornA1);
		float4 normalOnSurfaceB1 = qtRotate(ornA1,planeNormal1);
		float4 pOnB1 = vtxInPlaneWorld1+normalOnSurfaceB1*distance;
		pOnB1.w = distance;

		int dstIdx;
    AppendInc( nGlobalContactsOut, dstIdx );
		
		if (dstIdx < maxContactCapacity)
		{
			__global struct b3Contact4Data* c = &globalContactsOut[dstIdx];
			c->m_worldNormalOnB = -normalOnSurfaceB1;
			c->m_restituitionCoeffCmp = (0.f*0xffff);c->m_frictionCoeffCmp = (0.7f*0xffff);
			c->m_batchIdx = pairIndex;
			c->m_bodyAPtrAndSignBit = rigidBodies[bodyIndexA].m_invMass==0?-bodyIndexA:bodyIndexA;
			c->m_bodyBPtrAndSignBit = rigidBodies[bodyIndexB].m_invMass==0?-bodyIndexB:bodyIndexB;
			c->m_worldPosB[0] = pOnB1;
			c->m_childIndexA = -1;
			c->m_childIndexB = -1;
			GET_NPOINTS(*c) = 1;
		}//if (dstIdx < numPairs)
	}//if (hasCollision)
}


__kernel void   primitiveContactsKernel( __global int4* pairs, 
																					__global const BodyData* rigidBodies, 
																					__global const btCollidableGpu* collidables,
																					__global const ConvexPolyhedronCL* convexShapes, 
																					__global const float4* vertices,
																					__global const float4* uniqueEdges,
																					__global const btGpuFace* faces,
																					__global const int* indices,
																					__global struct b3Contact4Data* restrict globalContactsOut,
																					counter32_t nGlobalContactsOut,
																					int numPairs, int maxContactCapacity)
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
	
		if (collidables[collidableIndexA].m_shapeType == SHAPE_PLANE &&
			collidables[collidableIndexB].m_shapeType == SHAPE_CONVEX_HULL)
		{

			float4 posB;
			posB = rigidBodies[bodyIndexB].m_pos;
			Quaternion ornB;
			ornB = rigidBodies[bodyIndexB].m_quat;
			int contactIndex = computeContactPlaneConvex(pairIndex, bodyIndexA, bodyIndexB, collidableIndexA, collidableIndexB, 
																rigidBodies,collidables,convexShapes,vertices,indices,
																faces,	globalContactsOut, nGlobalContactsOut,maxContactCapacity, posB,ornB);
			if (contactIndex>=0)
				pairs[pairIndex].z = contactIndex;

			return;
		}


		if (collidables[collidableIndexA].m_shapeType == SHAPE_CONVEX_HULL &&
			collidables[collidableIndexB].m_shapeType == SHAPE_PLANE)
		{

			float4 posA;
			posA = rigidBodies[bodyIndexA].m_pos;
			Quaternion ornA;
			ornA = rigidBodies[bodyIndexA].m_quat;


			int contactIndex = computeContactPlaneConvex( pairIndex, bodyIndexB,bodyIndexA,  collidableIndexB,collidableIndexA, 
																rigidBodies,collidables,convexShapes,vertices,indices,
																faces,	globalContactsOut, nGlobalContactsOut,maxContactCapacity,posA,ornA);

			if (contactIndex>=0)
				pairs[pairIndex].z = contactIndex;

			return;
		}

		if (collidables[collidableIndexA].m_shapeType == SHAPE_PLANE &&
			collidables[collidableIndexB].m_shapeType == SHAPE_SPHERE)
		{
			computeContactPlaneSphere(pairIndex, bodyIndexA, bodyIndexB, collidableIndexA, collidableIndexB, 
																rigidBodies,collidables,faces,	globalContactsOut, nGlobalContactsOut,maxContactCapacity);
			return;
		}


		if (collidables[collidableIndexA].m_shapeType == SHAPE_SPHERE &&
			collidables[collidableIndexB].m_shapeType == SHAPE_PLANE)
		{


			computeContactPlaneSphere( pairIndex, bodyIndexB,bodyIndexA,  collidableIndexB,collidableIndexA, 
																rigidBodies,collidables,
																faces,	globalContactsOut, nGlobalContactsOut,maxContactCapacity);

			return;
		}

		

	
		if (collidables[collidableIndexA].m_shapeType == SHAPE_SPHERE &&
			collidables[collidableIndexB].m_shapeType == SHAPE_CONVEX_HULL)
		{
		
			float4 spherePos = rigidBodies[bodyIndexA].m_pos;
			float sphereRadius = collidables[collidableIndexA].m_radius;
			float4 convexPos = rigidBodies[bodyIndexB].m_pos;
			float4 convexOrn = rigidBodies[bodyIndexB].m_quat;

			computeContactSphereConvex(pairIndex, bodyIndexA, bodyIndexB, collidableIndexA, collidableIndexB, 
																rigidBodies,collidables,convexShapes,vertices,indices,faces, globalContactsOut, nGlobalContactsOut,maxContactCapacity,
																spherePos,sphereRadius,convexPos,convexOrn);

			return;
		}

		if (collidables[collidableIndexA].m_shapeType == SHAPE_CONVEX_HULL &&
			collidables[collidableIndexB].m_shapeType == SHAPE_SPHERE)
		{
		
			float4 spherePos = rigidBodies[bodyIndexB].m_pos;
			float sphereRadius = collidables[collidableIndexB].m_radius;
			float4 convexPos = rigidBodies[bodyIndexA].m_pos;
			float4 convexOrn = rigidBodies[bodyIndexA].m_quat;

			computeContactSphereConvex(pairIndex, bodyIndexB, bodyIndexA, collidableIndexB, collidableIndexA, 
																rigidBodies,collidables,convexShapes,vertices,indices,faces, globalContactsOut, nGlobalContactsOut,maxContactCapacity,
																spherePos,sphereRadius,convexPos,convexOrn);
			return;
		}
	
	
	
		
	
	
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
				
				if (dstIdx < maxContactCapacity)
				{
					__global struct b3Contact4Data* c = &globalContactsOut[dstIdx];
					c->m_worldNormalOnB = normalOnSurfaceB;
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

			return;
		}//SHAPE_SPHERE SHAPE_SPHERE

	}//	if (i<numPairs)

}


// work-in-progress
__kernel void   processCompoundPairsPrimitivesKernel( __global const int4* gpuCompoundPairs,
													__global const BodyData* rigidBodies, 
													__global const btCollidableGpu* collidables,
													__global const ConvexPolyhedronCL* convexShapes, 
													__global const float4* vertices,
													__global const float4* uniqueEdges,
													__global const btGpuFace* faces,
													__global const int* indices,
													__global btAabbCL* aabbs,
													__global const btGpuChildShape* gpuChildShapes,
													__global struct b3Contact4Data* restrict globalContactsOut,
													counter32_t nGlobalContactsOut,
													int numCompoundPairs, int maxContactCapacity
													)
{

	int i = get_global_id(0);
	if (i<numCompoundPairs)
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
	
		int shapeTypeA = collidables[collidableIndexA].m_shapeType;
		int shapeTypeB = collidables[collidableIndexB].m_shapeType;

		int pairIndex = i;
		if ((shapeTypeA == SHAPE_PLANE) && (shapeTypeB==SHAPE_CONVEX_HULL))
		{

			computeContactPlaneConvex( pairIndex, bodyIndexA,bodyIndexB,  collidableIndexA,collidableIndexB, 
																rigidBodies,collidables,convexShapes,vertices,indices,
																faces,	globalContactsOut, nGlobalContactsOut,maxContactCapacity,posB,ornB);
			return;
		}

		if ((shapeTypeA == SHAPE_CONVEX_HULL) && (shapeTypeB==SHAPE_PLANE))
		{

			computeContactPlaneConvex( pairIndex, bodyIndexB,bodyIndexA,  collidableIndexB,collidableIndexA, 
																rigidBodies,collidables,convexShapes,vertices,indices,
																faces,	globalContactsOut, nGlobalContactsOut,maxContactCapacity,posA,ornA);
			return;
		}

		if ((shapeTypeA == SHAPE_CONVEX_HULL) && (shapeTypeB == SHAPE_SPHERE))
		{
			float4 spherePos = rigidBodies[bodyIndexB].m_pos;
			float sphereRadius = collidables[collidableIndexB].m_radius;
			float4 convexPos = posA;
			float4 convexOrn = ornA;
			
			computeContactSphereConvex(pairIndex, bodyIndexB, bodyIndexA , collidableIndexB,collidableIndexA, 
										rigidBodies,collidables,convexShapes,vertices,indices,faces, globalContactsOut, nGlobalContactsOut,maxContactCapacity,
										spherePos,sphereRadius,convexPos,convexOrn);
	
			return;
		}

		if ((shapeTypeA == SHAPE_SPHERE) && (shapeTypeB == SHAPE_CONVEX_HULL))
		{

			float4 spherePos = rigidBodies[bodyIndexA].m_pos;
			float sphereRadius = collidables[collidableIndexA].m_radius;
			float4 convexPos = posB;
			float4 convexOrn = ornB;

			
			computeContactSphereConvex(pairIndex, bodyIndexA, bodyIndexB, collidableIndexA, collidableIndexB, 
										rigidBodies,collidables,convexShapes,vertices,indices,faces, globalContactsOut, nGlobalContactsOut,maxContactCapacity,
										spherePos,sphereRadius,convexPos,convexOrn);
	
			return;
		}
	}//	if (i<numCompoundPairs)
}


bool pointInTriangle(const float4* vertices, const float4* normal, float4 *p )
{

	const float4* p1 = &vertices[0];
	const float4* p2 = &vertices[1];
	const float4* p3 = &vertices[2];

	float4 edge1;	edge1 = (*p2 - *p1);
	float4 edge2;	edge2 = ( *p3 - *p2 );
	float4 edge3;	edge3 = ( *p1 - *p3 );

	
	float4 p1_to_p; p1_to_p = ( *p - *p1 );
	float4 p2_to_p; p2_to_p = ( *p - *p2 );
	float4 p3_to_p; p3_to_p = ( *p - *p3 );

	float4 edge1_normal; edge1_normal = ( cross(edge1,*normal));
	float4 edge2_normal; edge2_normal = ( cross(edge2,*normal));
	float4 edge3_normal; edge3_normal = ( cross(edge3,*normal));

	
	
	float r1, r2, r3;
	r1 = dot(edge1_normal,p1_to_p );
	r2 = dot(edge2_normal,p2_to_p );
	r3 = dot(edge3_normal,p3_to_p );
	
	if ( r1 > 0 && r2 > 0 && r3 > 0 )
		return true;
    if ( r1 <= 0 && r2 <= 0 && r3 <= 0 ) 
		return true;
	return false;

}


float segmentSqrDistance(float4 from, float4 to,float4 p, float4* nearest) 
{
	float4 diff = p - from;
	float4 v = to - from;
	float t = dot(v,diff);
	
	if (t > 0) 
	{
		float dotVV = dot(v,v);
		if (t < dotVV) 
		{
			t /= dotVV;
			diff -= t*v;
		} else 
		{
			t = 1;
			diff -= v;
		}
	} else
	{
		t = 0;
	}
	*nearest = from + t*v;
	return dot(diff,diff);	
}


void	computeContactSphereTriangle(int pairIndex,
									int bodyIndexA, int bodyIndexB,
									int collidableIndexA, int collidableIndexB, 
									__global const BodyData* rigidBodies, 
									__global const btCollidableGpu* collidables,
									const float4* triangleVertices,
									__global struct b3Contact4Data* restrict globalContactsOut,
									counter32_t nGlobalContactsOut,
									int maxContactCapacity,
									float4 spherePos2,
									float radius,
									float4 pos,
									float4 quat,
									int faceIndex
									)
{

	float4 invPos;
	float4 invOrn;

	trInverse(pos,quat, &invPos,&invOrn);
	float4 spherePos = transform(&spherePos2,&invPos,&invOrn);
	int numFaces = 3;
	float4 closestPnt = (float4)(0, 0, 0, 0);
	float4 hitNormalWorld = (float4)(0, 0, 0, 0);
	float minDist = -1000000.f;
	bool bCollide = false;

	
	//////////////////////////////////////

	float4 sphereCenter;
	sphereCenter = spherePos;

	const float4* vertices = triangleVertices;
	float contactBreakingThreshold = 0.f;//todo?
	float radiusWithThreshold = radius + contactBreakingThreshold;
	float4 edge10;
	edge10 = vertices[1]-vertices[0];
	edge10.w = 0.f;//is this needed?
	float4 edge20;
	edge20 = vertices[2]-vertices[0];
	edge20.w = 0.f;//is this needed?
	float4 normal = cross3(edge10,edge20);
	normal = normalize(normal);
	float4 p1ToCenter;
	p1ToCenter = sphereCenter - vertices[0];
	
	float distanceFromPlane = dot(p1ToCenter,normal);

	if (distanceFromPlane < 0.f)
	{
		//triangle facing the other way
		distanceFromPlane *= -1.f;
		normal *= -1.f;
	}
	hitNormalWorld = normal;

	bool isInsideContactPlane = distanceFromPlane < radiusWithThreshold;
	
	// Check for contact / intersection
	bool hasContact = false;
	float4 contactPoint;
	if (isInsideContactPlane) 
	{
	
		if (pointInTriangle(vertices,&normal, &sphereCenter)) 
		{
			// Inside the contact wedge - touches a point on the shell plane
			hasContact = true;
			contactPoint = sphereCenter - normal*distanceFromPlane;
			
		} else {
			// Could be inside one of the contact capsules
			float contactCapsuleRadiusSqr = radiusWithThreshold*radiusWithThreshold;
			float4 nearestOnEdge;
			int numEdges = 3;
			for (int i = 0; i < numEdges; i++) 
			{
				float4 pa =vertices[i];
				float4 pb = vertices[(i+1)%3];

				float distanceSqr = segmentSqrDistance(pa,pb,sphereCenter, &nearestOnEdge);
				if (distanceSqr < contactCapsuleRadiusSqr) 
				{
					// Yep, we're inside a capsule
					hasContact = true;
					contactPoint = nearestOnEdge;
					
				}
				
			}
		}
	}

	if (hasContact) 
	{

		closestPnt = contactPoint;
		float4 contactToCenter = sphereCenter - contactPoint;
		minDist = length(contactToCenter);
		if (minDist>FLT_EPSILON)
		{
			hitNormalWorld = normalize(contactToCenter);//*(1./minDist);
			bCollide  = true;
		}
		
	}


	/////////////////////////////////////

	if (bCollide && minDist > -10000)
	{
		
		float4 normalOnSurfaceB1 = qtRotate(quat,-hitNormalWorld);
		float4 pOnB1 = transform(&closestPnt,&pos,&quat);
		float actualDepth = minDist-radius;

		
		if (actualDepth<=0.f)
		{
			pOnB1.w = actualDepth;
			int dstIdx;

			
			float lenSqr = dot3F4(normalOnSurfaceB1,normalOnSurfaceB1);
			if (lenSqr>FLT_EPSILON)
			{
				AppendInc( nGlobalContactsOut, dstIdx );
			
				if (dstIdx < maxContactCapacity)
				{
					__global struct b3Contact4Data* c = &globalContactsOut[dstIdx];
					c->m_worldNormalOnB = -normalOnSurfaceB1;
					c->m_restituitionCoeffCmp = (0.f*0xffff);c->m_frictionCoeffCmp = (0.7f*0xffff);
					c->m_batchIdx = pairIndex;
					c->m_bodyAPtrAndSignBit = rigidBodies[bodyIndexA].m_invMass==0?-bodyIndexA:bodyIndexA;
					c->m_bodyBPtrAndSignBit = rigidBodies[bodyIndexB].m_invMass==0?-bodyIndexB:bodyIndexB;
					c->m_worldPosB[0] = pOnB1;

					c->m_childIndexA = -1;
					c->m_childIndexB = faceIndex;

					GET_NPOINTS(*c) = 1;
				} 
			}

		}
	}//if (hasCollision)

}



// work-in-progress
__kernel void   findConcaveSphereContactsKernel( __global int4* concavePairs,
												__global const BodyData* rigidBodies,
												__global const btCollidableGpu* collidables,
												__global const ConvexPolyhedronCL* convexShapes, 
												__global const float4* vertices,
												__global const float4* uniqueEdges,
												__global const btGpuFace* faces,
												__global const int* indices,
												__global btAabbCL* aabbs,
												__global struct b3Contact4Data* restrict globalContactsOut,
												counter32_t nGlobalContactsOut,
													int numConcavePairs, int maxContactCapacity
												)
{

	int i = get_global_id(0);
	if (i>=numConcavePairs)
		return;
	int pairIdx = i;

	int bodyIndexA = concavePairs[i].x;
	int bodyIndexB = concavePairs[i].y;

	int collidableIndexA = rigidBodies[bodyIndexA].m_collidableIdx;
	int collidableIndexB = rigidBodies[bodyIndexB].m_collidableIdx;

	int shapeIndexA = collidables[collidableIndexA].m_shapeIndex;
	int shapeIndexB = collidables[collidableIndexB].m_shapeIndex;

	if (collidables[collidableIndexB].m_shapeType==SHAPE_SPHERE)
	{
		int f = concavePairs[i].z;
		btGpuFace face = faces[convexShapes[shapeIndexA].m_faceOffset+f];
		
		float4 verticesA[3];
		for (int i=0;i<3;i++)
		{
			int index = indices[face.m_indexOffset+i];
			float4 vert = vertices[convexShapes[shapeIndexA].m_vertexOffset+index];
			verticesA[i] = vert;
		}

		float4 spherePos = rigidBodies[bodyIndexB].m_pos;
		float sphereRadius = collidables[collidableIndexB].m_radius;
		float4 convexPos = rigidBodies[bodyIndexA].m_pos;
		float4 convexOrn = rigidBodies[bodyIndexA].m_quat;

		computeContactSphereTriangle(i, bodyIndexB, bodyIndexA, collidableIndexB, collidableIndexA, 
																rigidBodies,collidables,
																verticesA,
																globalContactsOut, nGlobalContactsOut,maxContactCapacity,
																spherePos,sphereRadius,convexPos,convexOrn, f);

		return;
	}
}