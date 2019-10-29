
//keep this enum in sync with the CPU version (in btCollidable.h)
//written by Erwin Coumans


#define SHAPE_CONVEX_HULL 3
#define SHAPE_CONCAVE_TRIMESH 5
#define TRIANGLE_NUM_CONVEX_FACES 5
#define SHAPE_COMPOUND_OF_CONVEX_HULLS 6

#define B3_MAX_STACK_DEPTH 256


typedef unsigned int u32;

///keep this in sync with btCollidable.h
typedef struct
{
	union {
		int m_numChildShapes;
		int m_bvhIndex;
	};
	union
	{
		float m_radius;
		int	m_compoundBvhIndex;
	};
	
	int m_shapeType;
	int m_shapeIndex;
	
} btCollidableGpu;

#define MAX_NUM_PARTS_IN_BITS 10

///b3QuantizedBvhNode is a compressed aabb node, 16 bytes.
///Node can be used for leafnode or internal node. Leafnodes can point to 32-bit triangle index (non-negative range).
typedef struct
{
	//12 bytes
	unsigned short int	m_quantizedAabbMin[3];
	unsigned short int	m_quantizedAabbMax[3];
	//4 bytes
	int	m_escapeIndexOrTriangleIndex;
} b3QuantizedBvhNode;

typedef struct
{
	float4		m_aabbMin;
	float4		m_aabbMax;
	float4		m_quantization;
	int			m_numNodes;
	int			m_numSubTrees;
	int			m_nodeOffset;
	int			m_subTreeOffset;

} b3BvhInfo;


int	getTriangleIndex(const b3QuantizedBvhNode* rootNode)
{
	unsigned int x=0;
	unsigned int y = (~(x&0))<<(31-MAX_NUM_PARTS_IN_BITS);
	// Get only the lower bits where the triangle index is stored
	return (rootNode->m_escapeIndexOrTriangleIndex&~(y));
}

int	getTriangleIndexGlobal(__global const b3QuantizedBvhNode* rootNode)
{
	unsigned int x=0;
	unsigned int y = (~(x&0))<<(31-MAX_NUM_PARTS_IN_BITS);
	// Get only the lower bits where the triangle index is stored
	return (rootNode->m_escapeIndexOrTriangleIndex&~(y));
}

int isLeafNode(const b3QuantizedBvhNode* rootNode)
{
	//skipindex is negative (internal node), triangleindex >=0 (leafnode)
	return (rootNode->m_escapeIndexOrTriangleIndex >= 0)? 1 : 0;
}

int isLeafNodeGlobal(__global const b3QuantizedBvhNode* rootNode)
{
	//skipindex is negative (internal node), triangleindex >=0 (leafnode)
	return (rootNode->m_escapeIndexOrTriangleIndex >= 0)? 1 : 0;
}
	
int getEscapeIndex(const b3QuantizedBvhNode* rootNode)
{
	return -rootNode->m_escapeIndexOrTriangleIndex;
}

int getEscapeIndexGlobal(__global const b3QuantizedBvhNode* rootNode)
{
	return -rootNode->m_escapeIndexOrTriangleIndex;
}


typedef struct
{
	//12 bytes
	unsigned short int	m_quantizedAabbMin[3];
	unsigned short int	m_quantizedAabbMax[3];
	//4 bytes, points to the root of the subtree
	int			m_rootNodeIndex;
	//4 bytes
	int			m_subtreeSize;
	int			m_padding[3];
} b3BvhSubtreeInfo;







typedef struct
{
	float4	m_childPosition;
	float4	m_childOrientation;
	int m_shapeIndex;
	int m_unused0;
	int m_unused1;
	int m_unused2;
} btGpuChildShape;


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

#include "Bullet3Collision/BroadPhaseCollision/shared/b3Aabb.h"
#include "Bullet3Common/shared/b3Int2.h"



typedef struct
{
	float4 m_plane;
	int m_indexOffset;
	int m_numIndices;
} btGpuFace;

#define make_float4 (float4)


__inline
float4 cross3(float4 a, float4 b)
{
	return cross(a,b);

	
//	float4 a1 = make_float4(a.xyz,0.f);
//	float4 b1 = make_float4(b.xyz,0.f);

//	return cross(a1,b1);

//float4 c = make_float4(a.y*b.z - a.z*b.y,a.z*b.x - a.x*b.z,a.x*b.y - a.y*b.x,0.f);
	
	//	float4 c = make_float4(a.y*b.z - a.z*b.y,1.f,a.x*b.y - a.y*b.x,0.f);
	
	//return c;
}

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
	v = make_float4(v.xyz,0.f);
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

inline void projectLocal(const ConvexPolyhedronCL* hull,  const float4 pos, const float4 orn, 
const float4* dir, const float4* vertices, float* min, float* max)
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

inline void project(__global const ConvexPolyhedronCL* hull,  const float4 pos, const float4 orn, 
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

inline bool TestSepAxisLocalA(const ConvexPolyhedronCL* hullA, __global const ConvexPolyhedronCL* hullB, 
	const float4 posA,const float4 ornA,
	const float4 posB,const float4 ornB,
	float4* sep_axis, const float4* verticesA, __global const float4* verticesB,float* depth)
{
	float Min0,Max0;
	float Min1,Max1;
	projectLocal(hullA,posA,ornA,sep_axis,verticesA, &Min0, &Max0);
	project(hullB,posB,ornB, sep_axis,verticesB, &Min1, &Max1);

	if(Max0<Min1 || Max1<Min0)
		return false;

	float d0 = Max0 - Min1;
	float d1 = Max1 - Min0;
	*depth = d0<d1 ? d0:d1;
	return true;
}




inline bool IsAlmostZero(const float4 v)
{
	if(fabs(v.x)>1e-6f || fabs(v.y)>1e-6f || fabs(v.z)>1e-6f)
		return false;
	return true;
}



bool findSeparatingAxisLocalA(	const ConvexPolyhedronCL* hullA, __global const ConvexPolyhedronCL* hullB, 
	const float4 posA1,
	const float4 ornA,
	const float4 posB1,
	const float4 ornB,
	const float4 DeltaC2,
	
	const float4* verticesA, 
	const float4* uniqueEdgesA, 
	const btGpuFace* facesA,
	const int*  indicesA,

	__global const float4* verticesB, 
	__global const float4* uniqueEdgesB, 
	__global const btGpuFace* facesB,
	__global const int*  indicesB,
	float4* sep,
	float* dmin)
{
	

	float4 posA = posA1;
	posA.w = 0.f;
	float4 posB = posB1;
	posB.w = 0.f;
	int curPlaneTests=0;
	{
		int numFacesA = hullA->m_numFaces;
		// Test normals from hullA
		for(int i=0;i<numFacesA;i++)
		{
			const float4 normal = facesA[hullA->m_faceOffset+i].m_plane;
			float4 faceANormalWS = qtRotate(ornA,normal);
			if (dot3F4(DeltaC2,faceANormalWS)<0)
				faceANormalWS*=-1.f;
			curPlaneTests++;
			float d;
			if(!TestSepAxisLocalA( hullA, hullB, posA,ornA,posB,ornB,&faceANormalWS, verticesA, verticesB,&d))
				return false;
			if(d<*dmin)
			{
				*dmin = d;
				*sep = faceANormalWS;
			}
		}
	}
	if((dot3F4(-DeltaC2,*sep))>0.0f)
	{
		*sep = -(*sep);
	}
	return true;
}

bool findSeparatingAxisLocalB(	__global const ConvexPolyhedronCL* hullA,  const ConvexPolyhedronCL* hullB, 
	const float4 posA1,
	const float4 ornA,
	const float4 posB1,
	const float4 ornB,
	const float4 DeltaC2,
	__global const float4* verticesA, 
	__global const float4* uniqueEdgesA, 
	__global const btGpuFace* facesA,
	__global const int*  indicesA,
	const float4* verticesB,
	const float4* uniqueEdgesB, 
	const btGpuFace* facesB,
	const int*  indicesB,
	float4* sep,
	float* dmin)
{


	float4 posA = posA1;
	posA.w = 0.f;
	float4 posB = posB1;
	posB.w = 0.f;
	int curPlaneTests=0;
	{
		int numFacesA = hullA->m_numFaces;
		// Test normals from hullA
		for(int i=0;i<numFacesA;i++)
		{
			const float4 normal = facesA[hullA->m_faceOffset+i].m_plane;
			float4 faceANormalWS = qtRotate(ornA,normal);
			if (dot3F4(DeltaC2,faceANormalWS)<0)
				faceANormalWS *= -1.f;
			curPlaneTests++;
			float d;
			if(!TestSepAxisLocalA( hullB, hullA, posB,ornB,posA,ornA, &faceANormalWS, verticesB,verticesA, &d))
				return false;
			if(d<*dmin)
			{
				*dmin = d;
				*sep = faceANormalWS;
			}
		}
	}
	if((dot3F4(-DeltaC2,*sep))>0.0f)
	{
		*sep = -(*sep);
	}
	return true;
}



bool findSeparatingAxisEdgeEdgeLocalA(	const ConvexPolyhedronCL* hullA, __global const ConvexPolyhedronCL* hullB, 
	const float4 posA1,
	const float4 ornA,
	const float4 posB1,
	const float4 ornB,
	const float4 DeltaC2,
	const float4* verticesA, 
	const float4* uniqueEdgesA, 
	const btGpuFace* facesA,
	const int*  indicesA,
	__global const float4* verticesB, 
	__global const float4* uniqueEdgesB, 
	__global const btGpuFace* facesB,
	__global const int*  indicesB,
		float4* sep,
	float* dmin)
{


	float4 posA = posA1;
	posA.w = 0.f;
	float4 posB = posB1;
	posB.w = 0.f;

	int curPlaneTests=0;

	int curEdgeEdge = 0;
	// Test edges
	for(int e0=0;e0<hullA->m_numUniqueEdges;e0++)
	{
		const float4 edge0 = uniqueEdgesA[hullA->m_uniqueEdgesOffset+e0];
		float4 edge0World = qtRotate(ornA,edge0);

		for(int e1=0;e1<hullB->m_numUniqueEdges;e1++)
		{
			const float4 edge1 = uniqueEdgesB[hullB->m_uniqueEdgesOffset+e1];
			float4 edge1World = qtRotate(ornB,edge1);


			float4 crossje = cross3(edge0World,edge1World);

			curEdgeEdge++;
			if(!IsAlmostZero(crossje))
			{
				crossje = normalize3(crossje);
				if (dot3F4(DeltaC2,crossje)<0)
					crossje *= -1.f;

				float dist;
				bool result = true;
				{
					float Min0,Max0;
					float Min1,Max1;
					projectLocal(hullA,posA,ornA,&crossje,verticesA, &Min0, &Max0);
					project(hullB,posB,ornB,&crossje,verticesB, &Min1, &Max1);
				
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



inline int	findClippingFaces(const float4 separatingNormal,
                      const ConvexPolyhedronCL* hullA, 
					  __global const ConvexPolyhedronCL* hullB,
                      const float4 posA, const Quaternion ornA,const float4 posB, const Quaternion ornB,
                       __global float4* worldVertsA1,
                      __global float4* worldNormalsA1,
                      __global float4* worldVertsB1,
                      int capacityWorldVerts,
                      const float minDist, float maxDist,
					  const float4* verticesA,
                      const btGpuFace* facesA,
                      const int* indicesA,
					  __global const float4* verticesB,
                      __global const btGpuFace* facesB,
                      __global const int* indicesB,
                      __global int4* clippingFaces, int pairIndex)
{
	int numContactsOut = 0;
	int numWorldVertsB1= 0;
    
    
	int closestFaceB=0;
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
		const btGpuFace polyB = facesB[hullB->m_faceOffset+closestFaceB];
		int numVertices = polyB.m_numIndices;
        if (numVertices>capacityWorldVerts)
            numVertices = capacityWorldVerts;
        if (numVertices<0)
            numVertices = 0;
        
		for(int e0=0;e0<numVertices;e0++)
		{
            if (e0<capacityWorldVerts)
            {
                const float4 b = verticesB[hullB->m_vertexOffset+indicesB[polyB.m_indexOffset+e0]];
                worldVertsB1[pairIndex*capacityWorldVerts+numWorldVertsB1++] = transform(&b,&posB,&ornB);
            }
		}
	}
    
    int closestFaceA=0;
	{
		float dmin = FLT_MAX;
		for(int face=0;face<hullA->m_numFaces;face++)
		{
			const float4 Normal = make_float4(
                                              facesA[hullA->m_faceOffset+face].m_plane.x,
                                              facesA[hullA->m_faceOffset+face].m_plane.y,
                                              facesA[hullA->m_faceOffset+face].m_plane.z,
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
    
    int numVerticesA = facesA[hullA->m_faceOffset+closestFaceA].m_numIndices;
    if (numVerticesA>capacityWorldVerts)
       numVerticesA = capacityWorldVerts;
    if (numVerticesA<0)
        numVerticesA=0;
    
	for(int e0=0;e0<numVerticesA;e0++)
	{
        if (e0<capacityWorldVerts)
        {
            const float4 a = verticesA[hullA->m_vertexOffset+indicesA[facesA[hullA->m_faceOffset+closestFaceA].m_indexOffset+e0]];
            worldVertsA1[pairIndex*capacityWorldVerts+e0] = transform(&a, &posA,&ornA);
        }
    }
    
    clippingFaces[pairIndex].x = closestFaceA;
    clippingFaces[pairIndex].y = closestFaceB;
    clippingFaces[pairIndex].z = numVerticesA;
    clippingFaces[pairIndex].w = numWorldVertsB1;
    
    
	return numContactsOut;
}




// work-in-progress
__kernel void   findConcaveSeparatingAxisVertexFaceKernel( __global int4* concavePairs,
                                                __global const BodyData* rigidBodies,
                                                __global const btCollidableGpu* collidables,
                                                __global const ConvexPolyhedronCL* convexShapes,
                                                __global const float4* vertices,
                                                __global const float4* uniqueEdges,
                                                __global const btGpuFace* faces,
                                                __global const int* indices,
                                                __global const btGpuChildShape* gpuChildShapes,
                                                __global btAabbCL* aabbs,
                                                __global float4* concaveSeparatingNormalsOut,
                                                __global int* concaveHasSeparatingNormals,
                                                __global int4* clippingFacesOut,
                                                __global float4* worldVertsA1GPU,
                                                __global float4*  worldNormalsAGPU,
                                                __global float4* worldVertsB1GPU,
                                                __global float* dmins,
                                                int vertexFaceCapacity,
                                                int numConcavePairs
                                                )
{
    
	int i = get_global_id(0);
	if (i>=numConcavePairs)
		return;
    
	concaveHasSeparatingNormals[i] = 0;
    
	int pairIdx = i;
    
	int bodyIndexA = concavePairs[i].x;
	int bodyIndexB = concavePairs[i].y;
    
	int collidableIndexA = rigidBodies[bodyIndexA].m_collidableIdx;
	int collidableIndexB = rigidBodies[bodyIndexB].m_collidableIdx;
    
	int shapeIndexA = collidables[collidableIndexA].m_shapeIndex;
	int shapeIndexB = collidables[collidableIndexB].m_shapeIndex;
    
	if (collidables[collidableIndexB].m_shapeType!=SHAPE_CONVEX_HULL&&
		collidables[collidableIndexB].m_shapeType!=SHAPE_COMPOUND_OF_CONVEX_HULLS)
	{
		concavePairs[pairIdx].w = -1;
		return;
	}
    
    
    
	int numFacesA = convexShapes[shapeIndexA].m_numFaces;
	int numActualConcaveConvexTests = 0;
	
	int f = concavePairs[i].z;
	
	bool overlap = false;
	
	ConvexPolyhedronCL convexPolyhedronA;
    
	//add 3 vertices of the triangle
	convexPolyhedronA.m_numVertices = 3;
	convexPolyhedronA.m_vertexOffset = 0;
	float4	localCenter = make_float4(0.f,0.f,0.f,0.f);
    
	btGpuFace face = faces[convexShapes[shapeIndexA].m_faceOffset+f];
	float4 triMinAabb, triMaxAabb;
	btAabbCL triAabb;
	triAabb.m_min = make_float4(1e30f,1e30f,1e30f,0.f);
	triAabb.m_max = make_float4(-1e30f,-1e30f,-1e30f,0.f);
	
	float4 verticesA[3];
	for (int i=0;i<3;i++)
	{
		int index = indices[face.m_indexOffset+i];
		float4 vert = vertices[convexShapes[shapeIndexA].m_vertexOffset+index];
		verticesA[i] = vert;
		localCenter += vert;
        
		triAabb.m_min = min(triAabb.m_min,vert);
		triAabb.m_max = max(triAabb.m_max,vert);
        
	}
    
	overlap = true;
	overlap = (triAabb.m_min.x > aabbs[bodyIndexB].m_max.x || triAabb.m_max.x < aabbs[bodyIndexB].m_min.x) ? false : overlap;
	overlap = (triAabb.m_min.z > aabbs[bodyIndexB].m_max.z || triAabb.m_max.z < aabbs[bodyIndexB].m_min.z) ? false : overlap;
	overlap = (triAabb.m_min.y > aabbs[bodyIndexB].m_max.y || triAabb.m_max.y < aabbs[bodyIndexB].m_min.y) ? false : overlap;
    
	if (overlap)
	{
		float dmin = FLT_MAX;
		int hasSeparatingAxis=5;
		float4 sepAxis=make_float4(1,2,3,4);
        
		int localCC=0;
		numActualConcaveConvexTests++;
        
		//a triangle has 3 unique edges
		convexPolyhedronA.m_numUniqueEdges = 3;
		convexPolyhedronA.m_uniqueEdgesOffset = 0;
		float4 uniqueEdgesA[3];
		
		uniqueEdgesA[0] = (verticesA[1]-verticesA[0]);
		uniqueEdgesA[1] = (verticesA[2]-verticesA[1]);
		uniqueEdgesA[2] = (verticesA[0]-verticesA[2]);
        
        
		convexPolyhedronA.m_faceOffset = 0;
        
		float4 normal = make_float4(face.m_plane.x,face.m_plane.y,face.m_plane.z,0.f);
        
		btGpuFace facesA[TRIANGLE_NUM_CONVEX_FACES];
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
			float c = dot(normal,verticesA[0]);
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
				float c = -dot(edgeNormal,v0);
                
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
        
		
        
        
		///////////////////
		///compound shape support
        
		if (collidables[collidableIndexB].m_shapeType==SHAPE_COMPOUND_OF_CONVEX_HULLS)
		{
			int compoundChild = concavePairs[pairIdx].w;
			int childShapeIndexB = compoundChild;//collidables[collidableIndexB].m_shapeIndex+compoundChild;
			int childColIndexB = gpuChildShapes[childShapeIndexB].m_shapeIndex;
			float4 childPosB = gpuChildShapes[childShapeIndexB].m_childPosition;
			float4 childOrnB = gpuChildShapes[childShapeIndexB].m_childOrientation;
			float4 newPosB = transform(&childPosB,&posB,&ornB);
			float4 newOrnB = qtMul(ornB,childOrnB);
			posB = newPosB;
			ornB = newOrnB;
			shapeIndexB = collidables[childColIndexB].m_shapeIndex;
		}
		//////////////////
        
		float4 c0local = convexPolyhedronA.m_localCenter;
		float4 c0 = transform(&c0local, &posA, &ornA);
		float4 c1local = convexShapes[shapeIndexB].m_localCenter;
		float4 c1 = transform(&c1local,&posB,&ornB);
		const float4 DeltaC2 = c0 - c1;
        
        
		bool sepA = findSeparatingAxisLocalA(	&convexPolyhedronA, &convexShapes[shapeIndexB],
                                             posA,ornA,
                                             posB,ornB,
                                             DeltaC2,
                                             verticesA,uniqueEdgesA,facesA,indicesA,
                                             vertices,uniqueEdges,faces,indices,
                                             &sepAxis,&dmin);
		hasSeparatingAxis = 4;
		if (!sepA)
		{
			hasSeparatingAxis = 0;
		} else
		{
			bool sepB = findSeparatingAxisLocalB(	&convexShapes[shapeIndexB],&convexPolyhedronA,
                                                 posB,ornB,
                                                 posA,ornA,
                                                 DeltaC2,
                                                 vertices,uniqueEdges,faces,indices,
                                                 verticesA,uniqueEdgesA,facesA,indicesA,
                                                 &sepAxis,&dmin);
            
			if (!sepB)
			{
				hasSeparatingAxis = 0;
			} else
			{
				hasSeparatingAxis = 1;
			}
		}	
		
		if (hasSeparatingAxis)
		{
            dmins[i] = dmin;
			concaveSeparatingNormalsOut[pairIdx]=sepAxis;
			concaveHasSeparatingNormals[i]=1;
            
		} else
		{	
			//mark this pair as in-active
			concavePairs[pairIdx].w = -1;
		}
	}
	else
	{	
		//mark this pair as in-active
		concavePairs[pairIdx].w = -1;
	}
}




// work-in-progress
__kernel void   findConcaveSeparatingAxisEdgeEdgeKernel( __global int4* concavePairs,
                                                          __global const BodyData* rigidBodies,
                                                          __global const btCollidableGpu* collidables,
                                                          __global const ConvexPolyhedronCL* convexShapes,
                                                          __global const float4* vertices,
                                                          __global const float4* uniqueEdges,
                                                          __global const btGpuFace* faces,
                                                          __global const int* indices,
                                                          __global const btGpuChildShape* gpuChildShapes,
                                                          __global btAabbCL* aabbs,
                                                          __global float4* concaveSeparatingNormalsOut,
                                                          __global int* concaveHasSeparatingNormals,
                                                          __global int4* clippingFacesOut,
                                                          __global float4* worldVertsA1GPU,
                                                          __global float4*  worldNormalsAGPU,
                                                          __global float4* worldVertsB1GPU,
                                                          __global float* dmins,
                                                          int vertexFaceCapacity,
                                                          int numConcavePairs
                                                          )
{
    
	int i = get_global_id(0);
	if (i>=numConcavePairs)
		return;
    
	if (!concaveHasSeparatingNormals[i])
        return;
    
	int pairIdx = i;
    
	int bodyIndexA = concavePairs[i].x;
	int bodyIndexB = concavePairs[i].y;
    
	int collidableIndexA = rigidBodies[bodyIndexA].m_collidableIdx;
	int collidableIndexB = rigidBodies[bodyIndexB].m_collidableIdx;
    
	int shapeIndexA = collidables[collidableIndexA].m_shapeIndex;
	int shapeIndexB = collidables[collidableIndexB].m_shapeIndex;
    
    
	int numFacesA = convexShapes[shapeIndexA].m_numFaces;
	int numActualConcaveConvexTests = 0;
	
	int f = concavePairs[i].z;
	
	bool overlap = false;
	
	ConvexPolyhedronCL convexPolyhedronA;
    
	//add 3 vertices of the triangle
	convexPolyhedronA.m_numVertices = 3;
	convexPolyhedronA.m_vertexOffset = 0;
	float4	localCenter = make_float4(0.f,0.f,0.f,0.f);
    
	btGpuFace face = faces[convexShapes[shapeIndexA].m_faceOffset+f];
	float4 triMinAabb, triMaxAabb;
	btAabbCL triAabb;
	triAabb.m_min = make_float4(1e30f,1e30f,1e30f,0.f);
	triAabb.m_max = make_float4(-1e30f,-1e30f,-1e30f,0.f);
	
	float4 verticesA[3];
	for (int i=0;i<3;i++)
	{
		int index = indices[face.m_indexOffset+i];
		float4 vert = vertices[convexShapes[shapeIndexA].m_vertexOffset+index];
		verticesA[i] = vert;
		localCenter += vert;
        
		triAabb.m_min = min(triAabb.m_min,vert);
		triAabb.m_max = max(triAabb.m_max,vert);
        
	}
    
	overlap = true;
	overlap = (triAabb.m_min.x > aabbs[bodyIndexB].m_max.x || triAabb.m_max.x < aabbs[bodyIndexB].m_min.x) ? false : overlap;
	overlap = (triAabb.m_min.z > aabbs[bodyIndexB].m_max.z || triAabb.m_max.z < aabbs[bodyIndexB].m_min.z) ? false : overlap;
	overlap = (triAabb.m_min.y > aabbs[bodyIndexB].m_max.y || triAabb.m_max.y < aabbs[bodyIndexB].m_min.y) ? false : overlap;
    
	if (overlap)
	{
		float dmin = dmins[i];
		int hasSeparatingAxis=5;
		float4 sepAxis=make_float4(1,2,3,4);
        sepAxis = concaveSeparatingNormalsOut[pairIdx];
        
		int localCC=0;
		numActualConcaveConvexTests++;
        
		//a triangle has 3 unique edges
		convexPolyhedronA.m_numUniqueEdges = 3;
		convexPolyhedronA.m_uniqueEdgesOffset = 0;
		float4 uniqueEdgesA[3];
		
		uniqueEdgesA[0] = (verticesA[1]-verticesA[0]);
		uniqueEdgesA[1] = (verticesA[2]-verticesA[1]);
		uniqueEdgesA[2] = (verticesA[0]-verticesA[2]);
        
        
		convexPolyhedronA.m_faceOffset = 0;
        
		float4 normal = make_float4(face.m_plane.x,face.m_plane.y,face.m_plane.z,0.f);
        
		btGpuFace facesA[TRIANGLE_NUM_CONVEX_FACES];
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
			float c = dot(normal,verticesA[0]);
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
				float c = -dot(edgeNormal,v0);
                
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
        
		
        
        
		///////////////////
		///compound shape support
        
		if (collidables[collidableIndexB].m_shapeType==SHAPE_COMPOUND_OF_CONVEX_HULLS)
		{
			int compoundChild = concavePairs[pairIdx].w;
			int childShapeIndexB = compoundChild;//collidables[collidableIndexB].m_shapeIndex+compoundChild;
			int childColIndexB = gpuChildShapes[childShapeIndexB].m_shapeIndex;
			float4 childPosB = gpuChildShapes[childShapeIndexB].m_childPosition;
			float4 childOrnB = gpuChildShapes[childShapeIndexB].m_childOrientation;
			float4 newPosB = transform(&childPosB,&posB,&ornB);
			float4 newOrnB = qtMul(ornB,childOrnB);
			posB = newPosB;
			ornB = newOrnB;
			shapeIndexB = collidables[childColIndexB].m_shapeIndex;
		}
		//////////////////
        
		float4 c0local = convexPolyhedronA.m_localCenter;
		float4 c0 = transform(&c0local, &posA, &ornA);
		float4 c1local = convexShapes[shapeIndexB].m_localCenter;
		float4 c1 = transform(&c1local,&posB,&ornB);
		const float4 DeltaC2 = c0 - c1;
        
        
		{
			bool sepEE = findSeparatingAxisEdgeEdgeLocalA(	&convexPolyhedronA, &convexShapes[shapeIndexB],
                                                              posA,ornA,
                                                              posB,ornB,
                                                              DeltaC2,
                                                              verticesA,uniqueEdgesA,facesA,indicesA,
                                                              vertices,uniqueEdges,faces,indices,
                                                              &sepAxis,&dmin);
                
			if (!sepEE)
			{
				hasSeparatingAxis = 0;
			} else
			{
				hasSeparatingAxis = 1;
			}
		}
		
		
		if (hasSeparatingAxis)
		{
			sepAxis.w = dmin;
            dmins[i] = dmin;
			concaveSeparatingNormalsOut[pairIdx]=sepAxis;
			concaveHasSeparatingNormals[i]=1;
           
 	float minDist = -1e30f;
			float maxDist = 0.02f;

            
            findClippingFaces(sepAxis,
                              &convexPolyhedronA,
                              &convexShapes[shapeIndexB],
                              posA,ornA,
                              posB,ornB,
                              worldVertsA1GPU,
                              worldNormalsAGPU,
                              worldVertsB1GPU,
                              vertexFaceCapacity,
                              minDist, maxDist,
                              verticesA,
                              facesA,
                              indicesA,
                              vertices,
                              faces,
                              indices,
                              clippingFacesOut, pairIdx);
	           
            
		} else
		{	
			//mark this pair as in-active
			concavePairs[pairIdx].w = -1;
		}
	}
	else
	{	
		//mark this pair as in-active
		concavePairs[pairIdx].w = -1;
	}
	
	concavePairs[i].z = -1;//for the next stage, z is used to determine existing contact points
}

