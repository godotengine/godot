/** \file mikktspace/mikktspace.h
 *  \ingroup mikktspace
 */
/**
 *  Copyright (C) 2011 by Morten S. Mikkelsen
 *
 *  This software is provided 'as-is', without any express or implied
 *  warranty.  In no event will the authors be held liable for any damages
 *  arising from the use of this software.
 *
 *  Permission is granted to anyone to use this software for any purpose,
 *  including commercial applications, and to alter it and redistribute it
 *  freely, subject to the following restrictions:
 *
 *  1. The origin of this software must not be misrepresented; you must not
 *     claim that you wrote the original software. If you use this software
 *     in a product, an acknowledgment in the product documentation would be
 *     appreciated but is not required.
 *  2. Altered source versions must be plainly marked as such, and must not be
 *     misrepresented as being the original software.
 *  3. This notice may not be removed or altered from any source distribution.
 */

#ifndef __MIKKTSPACE_H__
#define __MIKKTSPACE_H__


#ifdef __cplusplus
extern "C" {
#endif

/* Author: Morten S. Mikkelsen
 * Version: 1.0
 *
 * The files mikktspace.h and mikktspace.c are designed to be
 * stand-alone files and it is important that they are kept this way.
 * Not having dependencies on structures/classes/libraries specific
 * to the program, in which they are used, allows them to be copied
 * and used as is into any tool, program or plugin.
 * The code is designed to consistently generate the same
 * tangent spaces, for a given mesh, in any tool in which it is used.
 * This is done by performing an internal welding step and subsequently an order-independent evaluation
 * of tangent space for meshes consisting of triangles and quads.
 * This means faces can be received in any order and the same is true for
 * the order of vertices of each face. The generated result will not be affected
 * by such reordering. Additionally, whether degenerate (vertices or texture coordinates)
 * primitives are present or not will not affect the generated results either.
 * Once tangent space calculation is done the vertices of degenerate primitives will simply
 * inherit tangent space from neighboring non degenerate primitives.
 * The analysis behind this implementation can be found in my master's thesis
 * which is available for download --> http://image.diku.dk/projects/media/morten.mikkelsen.08.pdf
 * Note that though the tangent spaces at the vertices are generated in an order-independent way,
 * by this implementation, the interpolated tangent space is still affected by which diagonal is
 * chosen to split each quad. A sensible solution is to have your tools pipeline always
 * split quads by the shortest diagonal. This choice is order-independent and works with mirroring.
 * If these have the same length then compare the diagonals defined by the texture coordinates.
 * XNormal which is a tool for baking normal maps allows you to write your own tangent space plugin
 * and also quad triangulator plugin.
 */


typedef int tbool;
typedef struct SMikkTSpaceContext SMikkTSpaceContext;

typedef struct {
	// Returns the number of faces (triangles/quads) on the mesh to be processed.
	int (*m_getNumFaces)(const SMikkTSpaceContext * pContext);

	// Returns the number of vertices on face number iFace
	// iFace is a number in the range {0, 1, ..., getNumFaces()-1}
	int (*m_getNumVerticesOfFace)(const SMikkTSpaceContext * pContext, const int iFace);

	// returns the position/normal/texcoord of the referenced face of vertex number iVert.
	// iVert is in the range {0,1,2} for triangles and {0,1,2,3} for quads.
	void (*m_getPosition)(const SMikkTSpaceContext * pContext, float fvPosOut[], const int iFace, const int iVert);
	void (*m_getNormal)(const SMikkTSpaceContext * pContext, float fvNormOut[], const int iFace, const int iVert);
	void (*m_getTexCoord)(const SMikkTSpaceContext * pContext, float fvTexcOut[], const int iFace, const int iVert);

	// either (or both) of the two setTSpace callbacks can be set.
	// The call-back m_setTSpaceBasic() is sufficient for basic normal mapping.

	// This function is used to return the tangent and fSign to the application.
	// fvTangent is a unit length vector.
	// For normal maps it is sufficient to use the following simplified version of the bitangent which is generated at pixel/vertex level.
	// bitangent = fSign * cross(vN, tangent);
	// Note that the results are returned unindexed. It is possible to generate a new index list
	// But averaging/overwriting tangent spaces by using an already existing index list WILL produce INCRORRECT results.
	// DO NOT! use an already existing index list.
	void (*m_setTSpaceBasic)(const SMikkTSpaceContext * pContext, const float fvTangent[], const float fSign, const int iFace, const int iVert);

	// This function is used to return tangent space results to the application.
	// fvTangent and fvBiTangent are unit length vectors and fMagS and fMagT are their
	// true magnitudes which can be used for relief mapping effects.
	// fvBiTangent is the "real" bitangent and thus may not be perpendicular to fvTangent.
	// However, both are perpendicular to the vertex normal.
	// For normal maps it is sufficient to use the following simplified version of the bitangent which is generated at pixel/vertex level.
	// fSign = bIsOrientationPreserving ? 1.0f : (-1.0f);
	// bitangent = fSign * cross(vN, tangent);
	// Note that the results are returned unindexed. It is possible to generate a new index list
	// But averaging/overwriting tangent spaces by using an already existing index list WILL produce INCRORRECT results.
	// DO NOT! use an already existing index list.
	void (*m_setTSpace)(const SMikkTSpaceContext * pContext, const float fvTangent[], const float fvBiTangent[], const float fMagS, const float fMagT,
						const tbool bIsOrientationPreserving, const int iFace, const int iVert);
} SMikkTSpaceInterface;

struct SMikkTSpaceContext
{
	SMikkTSpaceInterface * m_pInterface;	// initialized with callback functions
	void * m_pUserData;						// pointer to client side mesh data etc. (passed as the first parameter with every interface call)
};

// these are both thread safe!
tbool genTangSpaceDefault(const SMikkTSpaceContext * pContext);	// Default (recommended) fAngularThreshold is 180 degrees (which means threshold disabled)
tbool genTangSpace(const SMikkTSpaceContext * pContext, const float fAngularThreshold);


// To avoid visual errors (distortions/unwanted hard edges in lighting), when using sampled normal maps, the
// normal map sampler must use the exact inverse of the pixel shader transformation.
// The most efficient transformation we can possibly do in the pixel shader is
// achieved by using, directly, the "unnormalized" interpolated tangent, bitangent and vertex normal: vT, vB and vN.
// pixel shader (fast transform out)
// vNout = normalize( vNt.x * vT + vNt.y * vB + vNt.z * vN );
// where vNt is the tangent space normal. The normal map sampler must likewise use the
// interpolated and "unnormalized" tangent, bitangent and vertex normal to be compliant with the pixel shader.
// sampler does (exact inverse of pixel shader):
// float3 row0 = cross(vB, vN);
// float3 row1 = cross(vN, vT);
// float3 row2 = cross(vT, vB);
// float fSign = dot(vT, row0)<0 ? -1 : 1;
// vNt = normalize( fSign * float3(dot(vNout,row0), dot(vNout,row1), dot(vNout,row2)) );
// where vNout is the sampled normal in some chosen 3D space.
//
// Should you choose to reconstruct the bitangent in the pixel shader instead
// of the vertex shader, as explained earlier, then be sure to do this in the normal map sampler also.
// Finally, beware of quad triangulations. If the normal map sampler doesn't use the same triangulation of
// quads as your renderer then problems will occur since the interpolated tangent spaces will differ
// eventhough the vertex level tangent spaces match. This can be solved either by triangulating before
// sampling/exporting or by using the order-independent choice of diagonal for splitting quads suggested earlier.
// However, this must be used both by the sampler and your tools/rendering pipeline.

#ifdef __cplusplus
}
#endif

#endif
