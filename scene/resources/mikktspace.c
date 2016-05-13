/** \file mikktspace/mikktspace.c
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

#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <stdlib.h>

#include "mikktspace.h"

#define TFALSE		0
#define TTRUE		1

#ifndef M_PI
#define M_PI	3.1415926535897932384626433832795
#endif

#define INTERNAL_RND_SORT_SEED		39871946

// internal structure
typedef struct {
	float x, y, z;
} SVec3;

static tbool			veq( const SVec3 v1, const SVec3 v2 )
{
	return (v1.x == v2.x) && (v1.y == v2.y) && (v1.z == v2.z);
}

static SVec3		vadd( const SVec3 v1, const SVec3 v2 )
{
	SVec3 vRes;

	vRes.x = v1.x + v2.x;
	vRes.y = v1.y + v2.y;
	vRes.z = v1.z + v2.z;

	return vRes;
}


static SVec3		vsub( const SVec3 v1, const SVec3 v2 )
{
	SVec3 vRes;

	vRes.x = v1.x - v2.x;
	vRes.y = v1.y - v2.y;
	vRes.z = v1.z - v2.z;

	return vRes;
}

static SVec3		vscale(const float fS, const SVec3 v)
{
	SVec3 vRes;

	vRes.x = fS * v.x;
	vRes.y = fS * v.y;
	vRes.z = fS * v.z;

	return vRes;
}

static float			LengthSquared( const SVec3 v )
{
	return v.x*v.x + v.y*v.y + v.z*v.z;
}

static float			Length( const SVec3 v )
{
	return sqrtf(LengthSquared(v));
}

static SVec3		Normalize( const SVec3 v )
{
	return vscale(1 / Length(v), v);
}

static float		vdot( const SVec3 v1, const SVec3 v2)
{
	return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}


static tbool NotZero(const float fX)
{
	// could possibly use FLT_EPSILON instead
	return fabsf(fX) > FLT_MIN;
}

static tbool VNotZero(const SVec3 v)
{
	// might change this to an epsilon based test
	return NotZero(v.x) || NotZero(v.y) || NotZero(v.z);
}



typedef struct {
	int iNrFaces;
	int * pTriMembers;
} SSubGroup;

typedef struct {
	int iNrFaces;
	int * pFaceIndices;
	int iVertexRepresentitive;
	tbool bOrientPreservering;
} SGroup;

// 
#define MARK_DEGENERATE				1
#define QUAD_ONE_DEGEN_TRI			2
#define GROUP_WITH_ANY				4
#define ORIENT_PRESERVING			8



typedef struct {
	int FaceNeighbors[3];
	SGroup * AssignedGroup[3];
	
	// normalized first order face derivatives
	SVec3 vOs, vOt;
	float fMagS, fMagT;	// original magnitudes

	// determines if the current and the next triangle are a quad.
	int iOrgFaceNumber;
	int iFlag, iTSpacesOffs;
	unsigned char vert_num[4];
} STriInfo;

typedef struct {
	SVec3 vOs;
	float fMagS;
	SVec3 vOt;
	float fMagT;
	int iCounter;	// this is to average back into quads.
	tbool bOrient;
} STSpace;

static int GenerateInitialVerticesIndexList(STriInfo pTriInfos[], int piTriList_out[], const SMikkTSpaceContext * pContext, const int iNrTrianglesIn);
static void GenerateSharedVerticesIndexList(int piTriList_in_and_out[], const SMikkTSpaceContext * pContext, const int iNrTrianglesIn);
static void InitTriInfo(STriInfo pTriInfos[], const int piTriListIn[], const SMikkTSpaceContext * pContext, const int iNrTrianglesIn);
static int Build4RuleGroups(STriInfo pTriInfos[], SGroup pGroups[], int piGroupTrianglesBuffer[], const int piTriListIn[], const int iNrTrianglesIn);
static tbool GenerateTSpaces(STSpace psTspace[], const STriInfo pTriInfos[], const SGroup pGroups[],
                             const int iNrActiveGroups, const int piTriListIn[], const float fThresCos,
                             const SMikkTSpaceContext * pContext);

static int MakeIndex(const int iFace, const int iVert)
{
	assert(iVert>=0 && iVert<4 && iFace>=0);
	return (iFace<<2) | (iVert&0x3);
}

static void IndexToData(int * piFace, int * piVert, const int iIndexIn)
{
	piVert[0] = iIndexIn&0x3;
	piFace[0] = iIndexIn>>2;
}

static STSpace AvgTSpace(const STSpace * pTS0, const STSpace * pTS1)
{
	STSpace ts_res;

	// this if is important. Due to floating point precision
	// averaging when ts0==ts1 will cause a slight difference
	// which results in tangent space splits later on
	if (pTS0->fMagS==pTS1->fMagS && pTS0->fMagT==pTS1->fMagT &&
	   veq(pTS0->vOs,pTS1->vOs)	&& veq(pTS0->vOt, pTS1->vOt))
	{
		ts_res.fMagS = pTS0->fMagS;
		ts_res.fMagT = pTS0->fMagT;
		ts_res.vOs = pTS0->vOs;
		ts_res.vOt = pTS0->vOt;
	}
	else
	{
		ts_res.fMagS = 0.5f*(pTS0->fMagS+pTS1->fMagS);
		ts_res.fMagT = 0.5f*(pTS0->fMagT+pTS1->fMagT);
		ts_res.vOs = vadd(pTS0->vOs,pTS1->vOs);
		ts_res.vOt = vadd(pTS0->vOt,pTS1->vOt);
		if ( VNotZero(ts_res.vOs) ) ts_res.vOs = Normalize(ts_res.vOs);
		if ( VNotZero(ts_res.vOt) ) ts_res.vOt = Normalize(ts_res.vOt);
	}

	return ts_res;
}



static SVec3 GetPosition(const SMikkTSpaceContext * pContext, const int index);
static SVec3 GetNormal(const SMikkTSpaceContext * pContext, const int index);
static SVec3 GetTexCoord(const SMikkTSpaceContext * pContext, const int index);


// degen triangles
static void DegenPrologue(STriInfo pTriInfos[], int piTriList_out[], const int iNrTrianglesIn, const int iTotTris);
static void DegenEpilogue(STSpace psTspace[], STriInfo pTriInfos[], int piTriListIn[], const SMikkTSpaceContext * pContext, const int iNrTrianglesIn, const int iTotTris);


tbool genTangSpaceDefault(const SMikkTSpaceContext * pContext)
{
	return genTangSpace(pContext, 180.0f);
}

tbool genTangSpace(const SMikkTSpaceContext * pContext, const float fAngularThreshold)
{
	// count nr_triangles
	int * piTriListIn = NULL, * piGroupTrianglesBuffer = NULL;
	STriInfo * pTriInfos = NULL;
	SGroup * pGroups = NULL;
	STSpace * psTspace = NULL;
	int iNrTrianglesIn = 0, f=0, t=0, i=0;
	int iNrTSPaces = 0, iTotTris = 0, iDegenTriangles = 0, iNrMaxGroups = 0;
	int iNrActiveGroups = 0, index = 0;
	const int iNrFaces = pContext->m_pInterface->m_getNumFaces(pContext);
	tbool bRes = TFALSE;
	const float fThresCos = (float) cos((fAngularThreshold*(float)M_PI)/180.0f);

	// verify all call-backs have been set
	if ( pContext->m_pInterface->m_getNumFaces==NULL ||
		pContext->m_pInterface->m_getNumVerticesOfFace==NULL ||
		pContext->m_pInterface->m_getPosition==NULL ||
		pContext->m_pInterface->m_getNormal==NULL ||
		pContext->m_pInterface->m_getTexCoord==NULL )
		return TFALSE;

	// count triangles on supported faces
	for (f=0; f<iNrFaces; f++)
	{
		const int verts = pContext->m_pInterface->m_getNumVerticesOfFace(pContext, f);
		if (verts==3) ++iNrTrianglesIn;
		else if (verts==4) iNrTrianglesIn += 2;
	}
	if (iNrTrianglesIn<=0) return TFALSE;

	// allocate memory for an index list
	piTriListIn = (int *) malloc(sizeof(int)*3*iNrTrianglesIn);
	pTriInfos = (STriInfo *) malloc(sizeof(STriInfo)*iNrTrianglesIn);
	if (piTriListIn==NULL || pTriInfos==NULL)
	{
		if (piTriListIn!=NULL) free(piTriListIn);
		if (pTriInfos!=NULL) free(pTriInfos);
		return TFALSE;
	}

	// make an initial triangle --> face index list
	iNrTSPaces = GenerateInitialVerticesIndexList(pTriInfos, piTriListIn, pContext, iNrTrianglesIn);

	// make a welded index list of identical positions and attributes (pos, norm, texc)
	//printf("gen welded index list begin\n");
	GenerateSharedVerticesIndexList(piTriListIn, pContext, iNrTrianglesIn);
	//printf("gen welded index list end\n");

	// Mark all degenerate triangles
	iTotTris = iNrTrianglesIn;
	iDegenTriangles = 0;
	for (t=0; t<iTotTris; t++)
	{
		const int i0 = piTriListIn[t*3+0];
		const int i1 = piTriListIn[t*3+1];
		const int i2 = piTriListIn[t*3+2];
		const SVec3 p0 = GetPosition(pContext, i0);
		const SVec3 p1 = GetPosition(pContext, i1);
		const SVec3 p2 = GetPosition(pContext, i2);
		if (veq(p0,p1) || veq(p0,p2) || veq(p1,p2))	// degenerate
		{
			pTriInfos[t].iFlag |= MARK_DEGENERATE;
			++iDegenTriangles;
		}
	}
	iNrTrianglesIn = iTotTris - iDegenTriangles;

	// mark all triangle pairs that belong to a quad with only one
	// good triangle. These need special treatment in DegenEpilogue().
	// Additionally, move all good triangles to the start of
	// pTriInfos[] and piTriListIn[] without changing order and
	// put the degenerate triangles last.
	DegenPrologue(pTriInfos, piTriListIn, iNrTrianglesIn, iTotTris);

	
	// evaluate triangle level attributes and neighbor list
	//printf("gen neighbors list begin\n");
	InitTriInfo(pTriInfos, piTriListIn, pContext, iNrTrianglesIn);
	//printf("gen neighbors list end\n");

	
	// based on the 4 rules, identify groups based on connectivity
	iNrMaxGroups = iNrTrianglesIn*3;
	pGroups = (SGroup *) malloc(sizeof(SGroup)*iNrMaxGroups);
	piGroupTrianglesBuffer = (int *) malloc(sizeof(int)*iNrTrianglesIn*3);
	if (pGroups==NULL || piGroupTrianglesBuffer==NULL)
	{
		if (pGroups!=NULL) free(pGroups);
		if (piGroupTrianglesBuffer!=NULL) free(piGroupTrianglesBuffer);
		free(piTriListIn);
		free(pTriInfos);
		return TFALSE;
	}
	//printf("gen 4rule groups begin\n");
	iNrActiveGroups =
		Build4RuleGroups(pTriInfos, pGroups, piGroupTrianglesBuffer, piTriListIn, iNrTrianglesIn);
	//printf("gen 4rule groups end\n");

	//

	psTspace = (STSpace *) malloc(sizeof(STSpace)*iNrTSPaces);
	if (psTspace==NULL)
	{
		free(piTriListIn);
		free(pTriInfos);
		free(pGroups);
		free(piGroupTrianglesBuffer);
		return TFALSE;
	}
	memset(psTspace, 0, sizeof(STSpace)*iNrTSPaces);
	for (t=0; t<iNrTSPaces; t++)
	{
		psTspace[t].vOs.x=1.0f; psTspace[t].vOs.y=0.0f; psTspace[t].vOs.z=0.0f; psTspace[t].fMagS = 1.0f;
		psTspace[t].vOt.x=0.0f; psTspace[t].vOt.y=1.0f; psTspace[t].vOt.z=0.0f; psTspace[t].fMagT = 1.0f;
	}

	// make tspaces, each group is split up into subgroups if necessary
	// based on fAngularThreshold. Finally a tangent space is made for
	// every resulting subgroup
	//printf("gen tspaces begin\n");
	bRes = GenerateTSpaces(psTspace, pTriInfos, pGroups, iNrActiveGroups, piTriListIn, fThresCos, pContext);
	//printf("gen tspaces end\n");
	
	// clean up
	free(pGroups);
	free(piGroupTrianglesBuffer);

	if (!bRes)	// if an allocation in GenerateTSpaces() failed
	{
		// clean up and return false
		free(pTriInfos); free(piTriListIn); free(psTspace);
		return TFALSE;
	}


	// degenerate quads with one good triangle will be fixed by copying a space from
	// the good triangle to the coinciding vertex.
	// all other degenerate triangles will just copy a space from any good triangle
	// with the same welded index in piTriListIn[].
	DegenEpilogue(psTspace, pTriInfos, piTriListIn, pContext, iNrTrianglesIn, iTotTris);

	free(pTriInfos); free(piTriListIn);

	index = 0;
	for (f=0; f<iNrFaces; f++)
	{
		const int verts = pContext->m_pInterface->m_getNumVerticesOfFace(pContext, f);
		if (verts!=3 && verts!=4) continue;
		

		// I've decided to let degenerate triangles and group-with-anythings
		// vary between left/right hand coordinate systems at the vertices.
		// All healthy triangles on the other hand are built to always be either or.

		/*// force the coordinate system orientation to be uniform for every face.
		// (this is already the case for good triangles but not for
		// degenerate ones and those with bGroupWithAnything==true)
		bool bOrient = psTspace[index].bOrient;
		if (psTspace[index].iCounter == 0)	// tspace was not derived from a group
		{
			// look for a space created in GenerateTSpaces() by iCounter>0
			bool bNotFound = true;
			int i=1;
			while (i<verts && bNotFound)
			{
				if (psTspace[index+i].iCounter > 0) bNotFound=false;
				else ++i;
			}
			if (!bNotFound) bOrient = psTspace[index+i].bOrient;
		}*/

		// set data
		for (i=0; i<verts; i++)
		{
			const STSpace * pTSpace = &psTspace[index];
			float tang[] = {pTSpace->vOs.x, pTSpace->vOs.y, pTSpace->vOs.z};
			float bitang[] = {pTSpace->vOt.x, pTSpace->vOt.y, pTSpace->vOt.z};
			if (pContext->m_pInterface->m_setTSpace!=NULL)
				pContext->m_pInterface->m_setTSpace(pContext, tang, bitang, pTSpace->fMagS, pTSpace->fMagT, pTSpace->bOrient, f, i);
			if (pContext->m_pInterface->m_setTSpaceBasic!=NULL)
				pContext->m_pInterface->m_setTSpaceBasic(pContext, tang, pTSpace->bOrient==TTRUE ? 1.0f : (-1.0f), f, i);

			++index;
		}
	}

	free(psTspace);

	
	return TTRUE;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef struct {
	float vert[3];
	int index;
} STmpVert;

static const int g_iCells = 2048;

#ifdef _MSC_VER
	#define NOINLINE __declspec(noinline)
#else
	#define NOINLINE __attribute__ ((noinline))
#endif

// it is IMPORTANT that this function is called to evaluate the hash since
// inlining could potentially reorder instructions and generate different
// results for the same effective input value fVal.
static NOINLINE int FindGridCell(const float fMin, const float fMax, const float fVal)
{
	const float fIndex = g_iCells * ((fVal-fMin)/(fMax-fMin));
	const int iIndex = (int)fIndex;
	return iIndex < g_iCells ? (iIndex >= 0 ? iIndex : 0) : (g_iCells - 1);
}

static void MergeVertsFast(int piTriList_in_and_out[], STmpVert pTmpVert[], const SMikkTSpaceContext * pContext, const int iL_in, const int iR_in);
static void MergeVertsSlow(int piTriList_in_and_out[], const SMikkTSpaceContext * pContext, const int pTable[], const int iEntries);
static void GenerateSharedVerticesIndexListSlow(int piTriList_in_and_out[], const SMikkTSpaceContext * pContext, const int iNrTrianglesIn);

static void GenerateSharedVerticesIndexList(int piTriList_in_and_out[], const SMikkTSpaceContext * pContext, const int iNrTrianglesIn)
{

	// Generate bounding box
	int * piHashTable=NULL, * piHashCount=NULL, * piHashOffsets=NULL, * piHashCount2=NULL;
	STmpVert * pTmpVert = NULL;
	int i=0, iChannel=0, k=0, e=0;
	int iMaxCount=0;
	SVec3 vMin = GetPosition(pContext, 0), vMax = vMin, vDim;
	float fMin, fMax;
	for (i=1; i<(iNrTrianglesIn*3); i++)
	{
		const int index = piTriList_in_and_out[i];

		const SVec3 vP = GetPosition(pContext, index);
		if (vMin.x > vP.x) vMin.x = vP.x;
		else if (vMax.x < vP.x) vMax.x = vP.x;
		if (vMin.y > vP.y) vMin.y = vP.y;
		else if (vMax.y < vP.y) vMax.y = vP.y;
		if (vMin.z > vP.z) vMin.z = vP.z;
		else if (vMax.z < vP.z) vMax.z = vP.z;
	}

	vDim = vsub(vMax,vMin);
	iChannel = 0;
	fMin = vMin.x; fMax=vMax.x;
	if (vDim.y>vDim.x && vDim.y>vDim.z)
	{
		iChannel=1;
		fMin = vMin.y, fMax=vMax.y;
	}
	else if (vDim.z>vDim.x)
	{
		iChannel=2;
		fMin = vMin.z, fMax=vMax.z;
	}

	// make allocations
	piHashTable = (int *) malloc(sizeof(int)*iNrTrianglesIn*3);
	piHashCount = (int *) malloc(sizeof(int)*g_iCells);
	piHashOffsets = (int *) malloc(sizeof(int)*g_iCells);
	piHashCount2 = (int *) malloc(sizeof(int)*g_iCells);

	if (piHashTable==NULL || piHashCount==NULL || piHashOffsets==NULL || piHashCount2==NULL)
	{
		if (piHashTable!=NULL) free(piHashTable);
		if (piHashCount!=NULL) free(piHashCount);
		if (piHashOffsets!=NULL) free(piHashOffsets);
		if (piHashCount2!=NULL) free(piHashCount2);
		GenerateSharedVerticesIndexListSlow(piTriList_in_and_out, pContext, iNrTrianglesIn);
		return;
	}
	memset(piHashCount, 0, sizeof(int)*g_iCells);
	memset(piHashCount2, 0, sizeof(int)*g_iCells);

	// count amount of elements in each cell unit
	for (i=0; i<(iNrTrianglesIn*3); i++)
	{
		const int index = piTriList_in_and_out[i];
		const SVec3 vP = GetPosition(pContext, index);
		const float fVal = iChannel==0 ? vP.x : (iChannel==1 ? vP.y : vP.z);
		const int iCell = FindGridCell(fMin, fMax, fVal);
		++piHashCount[iCell];
	}

	// evaluate start index of each cell.
	piHashOffsets[0]=0;
	for (k=1; k<g_iCells; k++)
		piHashOffsets[k]=piHashOffsets[k-1]+piHashCount[k-1];

	// insert vertices
	for (i=0; i<(iNrTrianglesIn*3); i++)
	{
		const int index = piTriList_in_and_out[i];
		const SVec3 vP = GetPosition(pContext, index);
		const float fVal = iChannel==0 ? vP.x : (iChannel==1 ? vP.y : vP.z);
		const int iCell = FindGridCell(fMin, fMax, fVal);
		int * pTable = NULL;

		assert(piHashCount2[iCell]<piHashCount[iCell]);
		pTable = &piHashTable[piHashOffsets[iCell]];
		pTable[piHashCount2[iCell]] = i;	// vertex i has been inserted.
		++piHashCount2[iCell];
	}
	for (k=0; k<g_iCells; k++)
		assert(piHashCount2[k] == piHashCount[k]);	// verify the count
	free(piHashCount2);

	// find maximum amount of entries in any hash entry
	iMaxCount = piHashCount[0];
	for (k=1; k<g_iCells; k++)
		if (iMaxCount<piHashCount[k])
			iMaxCount=piHashCount[k];
	pTmpVert = (STmpVert *) malloc(sizeof(STmpVert)*iMaxCount);
	

	// complete the merge
	for (k=0; k<g_iCells; k++)
	{
		// extract table of cell k and amount of entries in it
		int * pTable = &piHashTable[piHashOffsets[k]];
		const int iEntries = piHashCount[k];
		if (iEntries < 2) continue;

		if (pTmpVert!=NULL)
		{
			for (e=0; e<iEntries; e++)
			{
				int i = pTable[e];
				const SVec3 vP = GetPosition(pContext, piTriList_in_and_out[i]);
				pTmpVert[e].vert[0] = vP.x; pTmpVert[e].vert[1] = vP.y;
				pTmpVert[e].vert[2] = vP.z; pTmpVert[e].index = i;
			}
			MergeVertsFast(piTriList_in_and_out, pTmpVert, pContext, 0, iEntries-1);
		}
		else
			MergeVertsSlow(piTriList_in_and_out, pContext, pTable, iEntries);
	}

	if (pTmpVert!=NULL) { free(pTmpVert); }
	free(piHashTable);
	free(piHashCount);
	free(piHashOffsets);
}

static void MergeVertsFast(int piTriList_in_and_out[], STmpVert pTmpVert[], const SMikkTSpaceContext * pContext, const int iL_in, const int iR_in)
{
	// make bbox
	int c=0, l=0, channel=0;
	float fvMin[3], fvMax[3];
	float dx=0, dy=0, dz=0, fSep=0;
	for (c=0; c<3; c++)
	{	fvMin[c]=pTmpVert[iL_in].vert[c]; fvMax[c]=fvMin[c];	}
	for (l=(iL_in+1); l<=iR_in; l++)
		for (c=0; c<3; c++)
			if (fvMin[c]>pTmpVert[l].vert[c]) fvMin[c]=pTmpVert[l].vert[c];
			else if (fvMax[c]<pTmpVert[l].vert[c]) fvMax[c]=pTmpVert[l].vert[c];

	dx = fvMax[0]-fvMin[0];
	dy = fvMax[1]-fvMin[1];
	dz = fvMax[2]-fvMin[2];

	channel = 0;
	if (dy>dx && dy>dz) channel=1;
	else if (dz>dx) channel=2;

	fSep = 0.5f*(fvMax[channel]+fvMin[channel]);

	// terminate recursion when the separation/average value
	// is no longer strictly between fMin and fMax values.
	if (fSep>=fvMax[channel] || fSep<=fvMin[channel])
	{
		// complete the weld
		for (l=iL_in; l<=iR_in; l++)
		{
			int i = pTmpVert[l].index;
			const int index = piTriList_in_and_out[i];
			const SVec3 vP = GetPosition(pContext, index);
			const SVec3 vN = GetNormal(pContext, index);
			const SVec3 vT = GetTexCoord(pContext, index);

			tbool bNotFound = TTRUE;
			int l2=iL_in, i2rec=-1;
			while (l2<l && bNotFound)
			{
				const int i2 = pTmpVert[l2].index;
				const int index2 = piTriList_in_and_out[i2];
				const SVec3 vP2 = GetPosition(pContext, index2);
				const SVec3 vN2 = GetNormal(pContext, index2);
				const SVec3 vT2 = GetTexCoord(pContext, index2);
				i2rec=i2;

				//if (vP==vP2 && vN==vN2 && vT==vT2)
				if (vP.x==vP2.x && vP.y==vP2.y && vP.z==vP2.z &&
					vN.x==vN2.x && vN.y==vN2.y && vN.z==vN2.z &&
					vT.x==vT2.x && vT.y==vT2.y && vT.z==vT2.z)
					bNotFound = TFALSE;
				else
					++l2;
			}
			
			// merge if previously found
			if (!bNotFound)
				piTriList_in_and_out[i] = piTriList_in_and_out[i2rec];
		}
	}
	else
	{
		int iL=iL_in, iR=iR_in;
		assert((iR_in-iL_in)>0);	// at least 2 entries

		// separate (by fSep) all points between iL_in and iR_in in pTmpVert[]
		while (iL < iR)
		{
			tbool bReadyLeftSwap = TFALSE, bReadyRightSwap = TFALSE;
			while ((!bReadyLeftSwap) && iL<iR)
			{
				assert(iL>=iL_in && iL<=iR_in);
				bReadyLeftSwap = !(pTmpVert[iL].vert[channel]<fSep);
				if (!bReadyLeftSwap) ++iL;
			}
			while ((!bReadyRightSwap) && iL<iR)
			{
				assert(iR>=iL_in && iR<=iR_in);
				bReadyRightSwap = pTmpVert[iR].vert[channel]<fSep;
				if (!bReadyRightSwap) --iR;
			}
			assert( (iL<iR) || !(bReadyLeftSwap && bReadyRightSwap) );

			if (bReadyLeftSwap && bReadyRightSwap)
			{
				const STmpVert sTmp = pTmpVert[iL];
				assert(iL<iR);
				pTmpVert[iL] = pTmpVert[iR];
				pTmpVert[iR] = sTmp;
				++iL; --iR;
			}
		}

		assert(iL==(iR+1) || (iL==iR));
		if (iL==iR)
		{
			const tbool bReadyRightSwap = pTmpVert[iR].vert[channel]<fSep;
			if (bReadyRightSwap) ++iL;
			else --iR;
		}

		// only need to weld when there is more than 1 instance of the (x,y,z)
		if (iL_in < iR)
			MergeVertsFast(piTriList_in_and_out, pTmpVert, pContext, iL_in, iR);	// weld all left of fSep
		if (iL < iR_in)
			MergeVertsFast(piTriList_in_and_out, pTmpVert, pContext, iL, iR_in);	// weld all right of (or equal to) fSep
	}
}

static void MergeVertsSlow(int piTriList_in_and_out[], const SMikkTSpaceContext * pContext, const int pTable[], const int iEntries)
{
	// this can be optimized further using a tree structure or more hashing.
	int e=0;
	for (e=0; e<iEntries; e++)
	{
		int i = pTable[e];
		const int index = piTriList_in_and_out[i];
		const SVec3 vP = GetPosition(pContext, index);
		const SVec3 vN = GetNormal(pContext, index);
		const SVec3 vT = GetTexCoord(pContext, index);

		tbool bNotFound = TTRUE;
		int e2=0, i2rec=-1;
		while (e2<e && bNotFound)
		{
			const int i2 = pTable[e2];
			const int index2 = piTriList_in_and_out[i2];
			const SVec3 vP2 = GetPosition(pContext, index2);
			const SVec3 vN2 = GetNormal(pContext, index2);
			const SVec3 vT2 = GetTexCoord(pContext, index2);
			i2rec = i2;

			if (veq(vP,vP2) && veq(vN,vN2) && veq(vT,vT2))
				bNotFound = TFALSE;
			else
				++e2;
		}
		
		// merge if previously found
		if (!bNotFound)
			piTriList_in_and_out[i] = piTriList_in_and_out[i2rec];
	}
}

static void GenerateSharedVerticesIndexListSlow(int piTriList_in_and_out[], const SMikkTSpaceContext * pContext, const int iNrTrianglesIn)
{
	int iNumUniqueVerts = 0, t=0, i=0;
	for (t=0; t<iNrTrianglesIn; t++)
	{
		for (i=0; i<3; i++)
		{
			const int offs = t*3 + i;
			const int index = piTriList_in_and_out[offs];

			const SVec3 vP = GetPosition(pContext, index);
			const SVec3 vN = GetNormal(pContext, index);
			const SVec3 vT = GetTexCoord(pContext, index);

			tbool bFound = TFALSE;
			int t2=0, index2rec=-1;
			while (!bFound && t2<=t)
			{
				int j=0;
				while (!bFound && j<3)
				{
					const int index2 = piTriList_in_and_out[t2*3 + j];
					const SVec3 vP2 = GetPosition(pContext, index2);
					const SVec3 vN2 = GetNormal(pContext, index2);
					const SVec3 vT2 = GetTexCoord(pContext, index2);
					
					if (veq(vP,vP2) && veq(vN,vN2) && veq(vT,vT2))
						bFound = TTRUE;
					else
						++j;
				}
				if (!bFound) ++t2;
			}

			assert(bFound);
			// if we found our own
			if (index2rec == index) { ++iNumUniqueVerts; }

			piTriList_in_and_out[offs] = index2rec;
		}
	}
}

static int GenerateInitialVerticesIndexList(STriInfo pTriInfos[], int piTriList_out[], const SMikkTSpaceContext * pContext, const int iNrTrianglesIn)
{
	int iTSpacesOffs = 0, f=0, t=0;
	int iDstTriIndex = 0;
	for (f=0; f<pContext->m_pInterface->m_getNumFaces(pContext); f++)
	{
		const int verts = pContext->m_pInterface->m_getNumVerticesOfFace(pContext, f);
		if (verts!=3 && verts!=4) continue;

		pTriInfos[iDstTriIndex].iOrgFaceNumber = f;
		pTriInfos[iDstTriIndex].iTSpacesOffs = iTSpacesOffs;

		if (verts==3)
		{
			unsigned char * pVerts = pTriInfos[iDstTriIndex].vert_num;
			pVerts[0]=0; pVerts[1]=1; pVerts[2]=2;
			piTriList_out[iDstTriIndex*3+0] = MakeIndex(f, 0);
			piTriList_out[iDstTriIndex*3+1] = MakeIndex(f, 1);
			piTriList_out[iDstTriIndex*3+2] = MakeIndex(f, 2);
			++iDstTriIndex;	// next
		}
		else
		{
			{
				pTriInfos[iDstTriIndex+1].iOrgFaceNumber = f;
				pTriInfos[iDstTriIndex+1].iTSpacesOffs = iTSpacesOffs;
			}

			{
				// need an order independent way to evaluate
				// tspace on quads. This is done by splitting
				// along the shortest diagonal.
				const int i0 = MakeIndex(f, 0);
				const int i1 = MakeIndex(f, 1);
				const int i2 = MakeIndex(f, 2);
				const int i3 = MakeIndex(f, 3);
				const SVec3 T0 = GetTexCoord(pContext, i0);
				const SVec3 T1 = GetTexCoord(pContext, i1);
				const SVec3 T2 = GetTexCoord(pContext, i2);
				const SVec3 T3 = GetTexCoord(pContext, i3);
				const float distSQ_02 = LengthSquared(vsub(T2,T0));
				const float distSQ_13 = LengthSquared(vsub(T3,T1));
				tbool bQuadDiagIs_02;
				if (distSQ_02<distSQ_13)
					bQuadDiagIs_02 = TTRUE;
				else if (distSQ_13<distSQ_02)
					bQuadDiagIs_02 = TFALSE;
				else
				{
					const SVec3 P0 = GetPosition(pContext, i0);
					const SVec3 P1 = GetPosition(pContext, i1);
					const SVec3 P2 = GetPosition(pContext, i2);
					const SVec3 P3 = GetPosition(pContext, i3);
					const float distSQ_02 = LengthSquared(vsub(P2,P0));
					const float distSQ_13 = LengthSquared(vsub(P3,P1));

					bQuadDiagIs_02 = distSQ_13<distSQ_02 ? TFALSE : TTRUE;
				}

				if (bQuadDiagIs_02)
				{
					{
						unsigned char * pVerts_A = pTriInfos[iDstTriIndex].vert_num;
						pVerts_A[0]=0; pVerts_A[1]=1; pVerts_A[2]=2;
					}
					piTriList_out[iDstTriIndex*3+0] = i0;
					piTriList_out[iDstTriIndex*3+1] = i1;
					piTriList_out[iDstTriIndex*3+2] = i2;
					++iDstTriIndex;	// next
					{
						unsigned char * pVerts_B = pTriInfos[iDstTriIndex].vert_num;
						pVerts_B[0]=0; pVerts_B[1]=2; pVerts_B[2]=3;
					}
					piTriList_out[iDstTriIndex*3+0] = i0;
					piTriList_out[iDstTriIndex*3+1] = i2;
					piTriList_out[iDstTriIndex*3+2] = i3;
					++iDstTriIndex;	// next
				}
				else
				{
					{
						unsigned char * pVerts_A = pTriInfos[iDstTriIndex].vert_num;
						pVerts_A[0]=0; pVerts_A[1]=1; pVerts_A[2]=3;
					}
					piTriList_out[iDstTriIndex*3+0] = i0;
					piTriList_out[iDstTriIndex*3+1] = i1;
					piTriList_out[iDstTriIndex*3+2] = i3;
					++iDstTriIndex;	// next
					{
						unsigned char * pVerts_B = pTriInfos[iDstTriIndex].vert_num;
						pVerts_B[0]=1; pVerts_B[1]=2; pVerts_B[2]=3;
					}
					piTriList_out[iDstTriIndex*3+0] = i1;
					piTriList_out[iDstTriIndex*3+1] = i2;
					piTriList_out[iDstTriIndex*3+2] = i3;
					++iDstTriIndex;	// next
				}
			}
		}

		iTSpacesOffs += verts;
		assert(iDstTriIndex<=iNrTrianglesIn);
	}

	for (t=0; t<iNrTrianglesIn; t++)
		pTriInfos[t].iFlag = 0;

	// return total amount of tspaces
	return iTSpacesOffs;
}

static SVec3 GetPosition(const SMikkTSpaceContext * pContext, const int index)
{
	int iF, iI;
	SVec3 res; float pos[3];
	IndexToData(&iF, &iI, index);
	pContext->m_pInterface->m_getPosition(pContext, pos, iF, iI);
	res.x=pos[0]; res.y=pos[1]; res.z=pos[2];
	return res;
}

static SVec3 GetNormal(const SMikkTSpaceContext * pContext, const int index)
{
	int iF, iI;
	SVec3 res; float norm[3];
	IndexToData(&iF, &iI, index);
	pContext->m_pInterface->m_getNormal(pContext, norm, iF, iI);
	res.x=norm[0]; res.y=norm[1]; res.z=norm[2];
	return res;
}

static SVec3 GetTexCoord(const SMikkTSpaceContext * pContext, const int index)
{
	int iF, iI;
	SVec3 res; float texc[2];
	IndexToData(&iF, &iI, index);
	pContext->m_pInterface->m_getTexCoord(pContext, texc, iF, iI);
	res.x=texc[0]; res.y=texc[1]; res.z=1.0f;
	return res;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////

typedef union {
	struct
	{
		int i0, i1, f;
	};
	int array[3];
} SEdge;

static void BuildNeighborsFast(STriInfo pTriInfos[], SEdge * pEdges, const int piTriListIn[], const int iNrTrianglesIn);
static void BuildNeighborsSlow(STriInfo pTriInfos[], const int piTriListIn[], const int iNrTrianglesIn);

// returns the texture area times 2
static float CalcTexArea(const SMikkTSpaceContext * pContext, const int indices[])
{
	const SVec3 t1 = GetTexCoord(pContext, indices[0]);
	const SVec3 t2 = GetTexCoord(pContext, indices[1]);
	const SVec3 t3 = GetTexCoord(pContext, indices[2]);

	const float t21x = t2.x-t1.x;
	const float t21y = t2.y-t1.y;
	const float t31x = t3.x-t1.x;
	const float t31y = t3.y-t1.y;

	const float fSignedAreaSTx2 = t21x*t31y - t21y*t31x;

	return fSignedAreaSTx2<0 ? (-fSignedAreaSTx2) : fSignedAreaSTx2;
}

static void InitTriInfo(STriInfo pTriInfos[], const int piTriListIn[], const SMikkTSpaceContext * pContext, const int iNrTrianglesIn)
{
	int f=0, i=0, t=0;
	// pTriInfos[f].iFlag is cleared in GenerateInitialVerticesIndexList() which is called before this function.

	// generate neighbor info list
	for (f=0; f<iNrTrianglesIn; f++)
		for (i=0; i<3; i++)
		{
			pTriInfos[f].FaceNeighbors[i] = -1;
			pTriInfos[f].AssignedGroup[i] = NULL;

			pTriInfos[f].vOs.x=0.0f; pTriInfos[f].vOs.y=0.0f; pTriInfos[f].vOs.z=0.0f;
			pTriInfos[f].vOt.x=0.0f; pTriInfos[f].vOt.y=0.0f; pTriInfos[f].vOt.z=0.0f;
			pTriInfos[f].fMagS = 0;
			pTriInfos[f].fMagT = 0;

			// assumed bad
			pTriInfos[f].iFlag |= GROUP_WITH_ANY;
		}

	// evaluate first order derivatives
	for (f=0; f<iNrTrianglesIn; f++)
	{
		// initial values
		const SVec3 v1 = GetPosition(pContext, piTriListIn[f*3+0]);
		const SVec3 v2 = GetPosition(pContext, piTriListIn[f*3+1]);
		const SVec3 v3 = GetPosition(pContext, piTriListIn[f*3+2]);
		const SVec3 t1 = GetTexCoord(pContext, piTriListIn[f*3+0]);
		const SVec3 t2 = GetTexCoord(pContext, piTriListIn[f*3+1]);
		const SVec3 t3 = GetTexCoord(pContext, piTriListIn[f*3+2]);

		const float t21x = t2.x-t1.x;
		const float t21y = t2.y-t1.y;
		const float t31x = t3.x-t1.x;
		const float t31y = t3.y-t1.y;
		const SVec3 d1 = vsub(v2,v1);
		const SVec3 d2 = vsub(v3,v1);

		const float fSignedAreaSTx2 = t21x*t31y - t21y*t31x;
		//assert(fSignedAreaSTx2!=0);
		SVec3 vOs = vsub(vscale(t31y,d1), vscale(t21y,d2));	// eq 18
		SVec3 vOt = vadd(vscale(-t31x,d1), vscale(t21x,d2)); // eq 19

		pTriInfos[f].iFlag |= (fSignedAreaSTx2>0 ? ORIENT_PRESERVING : 0);

		if ( NotZero(fSignedAreaSTx2) )
		{
			const float fAbsArea = fabsf(fSignedAreaSTx2);
			const float fLenOs = Length(vOs);
			const float fLenOt = Length(vOt);
			const float fS = (pTriInfos[f].iFlag&ORIENT_PRESERVING)==0 ? (-1.0f) : 1.0f;
			if ( NotZero(fLenOs) ) pTriInfos[f].vOs = vscale(fS/fLenOs, vOs);
			if ( NotZero(fLenOt) ) pTriInfos[f].vOt = vscale(fS/fLenOt, vOt);

			// evaluate magnitudes prior to normalization of vOs and vOt
			pTriInfos[f].fMagS = fLenOs / fAbsArea;
			pTriInfos[f].fMagT = fLenOt / fAbsArea;

			// if this is a good triangle
			if ( NotZero(pTriInfos[f].fMagS) && NotZero(pTriInfos[f].fMagT))
				pTriInfos[f].iFlag &= (~GROUP_WITH_ANY);
		}
	}

	// force otherwise healthy quads to a fixed orientation
	while (t<(iNrTrianglesIn-1))
	{
		const int iFO_a = pTriInfos[t].iOrgFaceNumber;
		const int iFO_b = pTriInfos[t+1].iOrgFaceNumber;
		if (iFO_a==iFO_b)	// this is a quad
		{
			const tbool bIsDeg_a = (pTriInfos[t].iFlag&MARK_DEGENERATE)!=0 ? TTRUE : TFALSE;
			const tbool bIsDeg_b = (pTriInfos[t+1].iFlag&MARK_DEGENERATE)!=0 ? TTRUE : TFALSE;
			
			// bad triangles should already have been removed by
			// DegenPrologue(), but just in case check bIsDeg_a and bIsDeg_a are false
			if ((bIsDeg_a||bIsDeg_b)==TFALSE)
			{
				const tbool bOrientA = (pTriInfos[t].iFlag&ORIENT_PRESERVING)!=0 ? TTRUE : TFALSE;
				const tbool bOrientB = (pTriInfos[t+1].iFlag&ORIENT_PRESERVING)!=0 ? TTRUE : TFALSE;
				// if this happens the quad has extremely bad mapping!!
				if (bOrientA!=bOrientB)
				{
					//printf("found quad with bad mapping\n");
					tbool bChooseOrientFirstTri = TFALSE;
					if ((pTriInfos[t+1].iFlag&GROUP_WITH_ANY)!=0) bChooseOrientFirstTri = TTRUE;
					else if ( CalcTexArea(pContext, &piTriListIn[t*3+0]) >= CalcTexArea(pContext, &piTriListIn[(t+1)*3+0]) )
						bChooseOrientFirstTri = TTRUE;

					// force match
					{
						const int t0 = bChooseOrientFirstTri ? t : (t+1);
						const int t1 = bChooseOrientFirstTri ? (t+1) : t;
						pTriInfos[t1].iFlag &= (~ORIENT_PRESERVING);	// clear first
						pTriInfos[t1].iFlag |= (pTriInfos[t0].iFlag&ORIENT_PRESERVING);	// copy bit
					}
				}
			}
			t += 2;
		}
		else
			++t;
	}
	
	// match up edge pairs
	{
		SEdge * pEdges = (SEdge *) malloc(sizeof(SEdge)*iNrTrianglesIn*3);
		if (pEdges==NULL)
			BuildNeighborsSlow(pTriInfos, piTriListIn, iNrTrianglesIn);
		else
		{
			BuildNeighborsFast(pTriInfos, pEdges, piTriListIn, iNrTrianglesIn);
	
			free(pEdges);
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////

static tbool AssignRecur(const int piTriListIn[], STriInfo psTriInfos[], const int iMyTriIndex, SGroup * pGroup);
static void AddTriToGroup(SGroup * pGroup, const int iTriIndex);

static int Build4RuleGroups(STriInfo pTriInfos[], SGroup pGroups[], int piGroupTrianglesBuffer[], const int piTriListIn[], const int iNrTrianglesIn)
{
	const int iNrMaxGroups = iNrTrianglesIn*3;
	int iNrActiveGroups = 0;
	int iOffset = 0, f=0, i=0;
	(void)iNrMaxGroups;  /* quiet warnings in non debug mode */
	for (f=0; f<iNrTrianglesIn; f++)
	{
		for (i=0; i<3; i++)
		{
			// if not assigned to a group
			if ((pTriInfos[f].iFlag&GROUP_WITH_ANY)==0 && pTriInfos[f].AssignedGroup[i]==NULL)
			{
				tbool bOrPre;
				int neigh_indexL, neigh_indexR;
				const int vert_index = piTriListIn[f*3+i];
				assert(iNrActiveGroups<iNrMaxGroups);
				pTriInfos[f].AssignedGroup[i] = &pGroups[iNrActiveGroups];
				pTriInfos[f].AssignedGroup[i]->iVertexRepresentitive = vert_index;
				pTriInfos[f].AssignedGroup[i]->bOrientPreservering = (pTriInfos[f].iFlag&ORIENT_PRESERVING)!=0;
				pTriInfos[f].AssignedGroup[i]->iNrFaces = 0;
				pTriInfos[f].AssignedGroup[i]->pFaceIndices = &piGroupTrianglesBuffer[iOffset];
				++iNrActiveGroups;

				AddTriToGroup(pTriInfos[f].AssignedGroup[i], f);
				bOrPre = (pTriInfos[f].iFlag&ORIENT_PRESERVING)!=0 ? TTRUE : TFALSE;
				neigh_indexL = pTriInfos[f].FaceNeighbors[i];
				neigh_indexR = pTriInfos[f].FaceNeighbors[i>0?(i-1):2];
				if (neigh_indexL>=0) // neighbor
				{
					const tbool bAnswer =
						AssignRecur(piTriListIn, pTriInfos, neigh_indexL,
									pTriInfos[f].AssignedGroup[i] );
					
					const tbool bOrPre2 = (pTriInfos[neigh_indexL].iFlag&ORIENT_PRESERVING)!=0 ? TTRUE : TFALSE;
					const tbool bDiff = bOrPre!=bOrPre2 ? TTRUE : TFALSE;
					assert(bAnswer || bDiff);
					(void)bAnswer, (void)bDiff;  /* quiet warnings in non debug mode */
				}
				if (neigh_indexR>=0) // neighbor
				{
					const tbool bAnswer =
						AssignRecur(piTriListIn, pTriInfos, neigh_indexR,
									pTriInfos[f].AssignedGroup[i] );

					const tbool bOrPre2 = (pTriInfos[neigh_indexR].iFlag&ORIENT_PRESERVING)!=0 ? TTRUE : TFALSE;
					const tbool bDiff = bOrPre!=bOrPre2 ? TTRUE : TFALSE;
					assert(bAnswer || bDiff);
					(void)bAnswer, (void)bDiff;  /* quiet warnings in non debug mode */
				}

				// update offset
				iOffset += pTriInfos[f].AssignedGroup[i]->iNrFaces;
				// since the groups are disjoint a triangle can never
				// belong to more than 3 groups. Subsequently something
				// is completely screwed if this assertion ever hits.
				assert(iOffset <= iNrMaxGroups);
			}
		}
	}

	return iNrActiveGroups;
}

static void AddTriToGroup(SGroup * pGroup, const int iTriIndex)
{
	pGroup->pFaceIndices[pGroup->iNrFaces] = iTriIndex;
	++pGroup->iNrFaces;
}

static tbool AssignRecur(const int piTriListIn[], STriInfo psTriInfos[],
				 const int iMyTriIndex, SGroup * pGroup)
{
	STriInfo * pMyTriInfo = &psTriInfos[iMyTriIndex];

	// track down vertex
	const int iVertRep = pGroup->iVertexRepresentitive;
	const int * pVerts = &piTriListIn[3*iMyTriIndex+0];
	int i=-1;
	if (pVerts[0]==iVertRep) i=0;
	else if (pVerts[1]==iVertRep) i=1;
	else if (pVerts[2]==iVertRep) i=2;
	assert(i>=0 && i<3);

	// early out
	if (pMyTriInfo->AssignedGroup[i] == pGroup) return TTRUE;
	else if (pMyTriInfo->AssignedGroup[i]!=NULL) return TFALSE;
	if ((pMyTriInfo->iFlag&GROUP_WITH_ANY)!=0)
	{
		// first to group with a group-with-anything triangle
		// determines it's orientation.
		// This is the only existing order dependency in the code!!
		if ( pMyTriInfo->AssignedGroup[0] == NULL &&
			pMyTriInfo->AssignedGroup[1] == NULL &&
			pMyTriInfo->AssignedGroup[2] == NULL )
		{
			pMyTriInfo->iFlag &= (~ORIENT_PRESERVING);
			pMyTriInfo->iFlag |= (pGroup->bOrientPreservering ? ORIENT_PRESERVING : 0);
		}
	}
	{
		const tbool bOrient = (pMyTriInfo->iFlag&ORIENT_PRESERVING)!=0 ? TTRUE : TFALSE;
		if (bOrient != pGroup->bOrientPreservering) return TFALSE;
	}

	AddTriToGroup(pGroup, iMyTriIndex);
	pMyTriInfo->AssignedGroup[i] = pGroup;

	{
		const int neigh_indexL = pMyTriInfo->FaceNeighbors[i];
		const int neigh_indexR = pMyTriInfo->FaceNeighbors[i>0?(i-1):2];
		if (neigh_indexL>=0)
			AssignRecur(piTriListIn, psTriInfos, neigh_indexL, pGroup);
		if (neigh_indexR>=0)
			AssignRecur(piTriListIn, psTriInfos, neigh_indexR, pGroup);
	}



	return TTRUE;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////

static tbool CompareSubGroups(const SSubGroup * pg1, const SSubGroup * pg2);
static void QuickSort(int* pSortBuffer, int iLeft, int iRight, unsigned int uSeed);
static STSpace EvalTspace(int face_indices[], const int iFaces, const int piTriListIn[], const STriInfo pTriInfos[], const SMikkTSpaceContext * pContext, const int iVertexRepresentitive);

static tbool GenerateTSpaces(STSpace psTspace[], const STriInfo pTriInfos[], const SGroup pGroups[],
                             const int iNrActiveGroups, const int piTriListIn[], const float fThresCos,
                             const SMikkTSpaceContext * pContext)
{
	STSpace * pSubGroupTspace = NULL;
	SSubGroup * pUniSubGroups = NULL;
	int * pTmpMembers = NULL;
	int iMaxNrFaces=0, iUniqueTspaces=0, g=0, i=0;
	for (g=0; g<iNrActiveGroups; g++)
		if (iMaxNrFaces < pGroups[g].iNrFaces)
			iMaxNrFaces = pGroups[g].iNrFaces;

	if (iMaxNrFaces == 0) return TTRUE;

	// make initial allocations
	pSubGroupTspace = (STSpace *) malloc(sizeof(STSpace)*iMaxNrFaces);
	pUniSubGroups = (SSubGroup *) malloc(sizeof(SSubGroup)*iMaxNrFaces);
	pTmpMembers = (int *) malloc(sizeof(int)*iMaxNrFaces);
	if (pSubGroupTspace==NULL || pUniSubGroups==NULL || pTmpMembers==NULL)
	{
		if (pSubGroupTspace!=NULL) free(pSubGroupTspace);
		if (pUniSubGroups!=NULL) free(pUniSubGroups);
		if (pTmpMembers!=NULL) free(pTmpMembers);
		return TFALSE;
	}


	iUniqueTspaces = 0;
	for (g=0; g<iNrActiveGroups; g++)
	{
		const SGroup * pGroup = &pGroups[g];
		int iUniqueSubGroups = 0, s=0;

		for (i=0; i<pGroup->iNrFaces; i++)	// triangles
		{
			const int f = pGroup->pFaceIndices[i];	// triangle number
			int index=-1, iVertIndex=-1, iOF_1=-1, iMembers=0, j=0, l=0;
			SSubGroup tmp_group;
			tbool bFound;
			SVec3 n, vOs, vOt;
			if (pTriInfos[f].AssignedGroup[0]==pGroup) index=0;
			else if (pTriInfos[f].AssignedGroup[1]==pGroup) index=1;
			else if (pTriInfos[f].AssignedGroup[2]==pGroup) index=2;
			assert(index>=0 && index<3);

			iVertIndex = piTriListIn[f*3+index];
			assert(iVertIndex==pGroup->iVertexRepresentitive);

			// is normalized already
			n = GetNormal(pContext, iVertIndex);
			
			// project
			vOs = vsub(pTriInfos[f].vOs, vscale(vdot(n,pTriInfos[f].vOs), n));
			vOt = vsub(pTriInfos[f].vOt, vscale(vdot(n,pTriInfos[f].vOt), n));
			if ( VNotZero(vOs) ) vOs = Normalize(vOs);
			if ( VNotZero(vOt) ) vOt = Normalize(vOt);

			// original face number
			iOF_1 = pTriInfos[f].iOrgFaceNumber;
			
			iMembers = 0;
			for (j=0; j<pGroup->iNrFaces; j++)
			{
				const int t = pGroup->pFaceIndices[j];	// triangle number
				const int iOF_2 = pTriInfos[t].iOrgFaceNumber;

				// project
				SVec3 vOs2 = vsub(pTriInfos[t].vOs, vscale(vdot(n,pTriInfos[t].vOs), n));
				SVec3 vOt2 = vsub(pTriInfos[t].vOt, vscale(vdot(n,pTriInfos[t].vOt), n));
				if ( VNotZero(vOs2) ) vOs2 = Normalize(vOs2);
				if ( VNotZero(vOt2) ) vOt2 = Normalize(vOt2);

				{
					const tbool bAny = ( (pTriInfos[f].iFlag | pTriInfos[t].iFlag) & GROUP_WITH_ANY )!=0 ? TTRUE : TFALSE;
					// make sure triangles which belong to the same quad are joined.
					const tbool bSameOrgFace = iOF_1==iOF_2 ? TTRUE : TFALSE;

					const float fCosS = vdot(vOs,vOs2);
					const float fCosT = vdot(vOt,vOt2);

					assert(f!=t || bSameOrgFace);	// sanity check
					if (bAny || bSameOrgFace || (fCosS>fThresCos && fCosT>fThresCos))
						pTmpMembers[iMembers++] = t;
				}
			}

			// sort pTmpMembers
			tmp_group.iNrFaces = iMembers;
			tmp_group.pTriMembers = pTmpMembers;
			if (iMembers>1)
			{
				unsigned int uSeed = INTERNAL_RND_SORT_SEED;	// could replace with a random seed?
				QuickSort(pTmpMembers, 0, iMembers-1, uSeed);
			}

			// look for an existing match
			bFound = TFALSE;
			l=0;
			while (l<iUniqueSubGroups && !bFound)
			{
				bFound = CompareSubGroups(&tmp_group, &pUniSubGroups[l]);
				if (!bFound) ++l;
			}
			
			// assign tangent space index
			assert(bFound || l==iUniqueSubGroups);
			//piTempTangIndices[f*3+index] = iUniqueTspaces+l;

			// if no match was found we allocate a new subgroup
			if (!bFound)
			{
				// insert new subgroup
				int * pIndices = (int *) malloc(sizeof(int)*iMembers);
				if (pIndices==NULL)
				{
					// clean up and return false
					int s=0;
					for (s=0; s<iUniqueSubGroups; s++)
						free(pUniSubGroups[s].pTriMembers);
					free(pUniSubGroups);
					free(pTmpMembers);
					free(pSubGroupTspace);
					return TFALSE;
				}
				pUniSubGroups[iUniqueSubGroups].iNrFaces = iMembers;
				pUniSubGroups[iUniqueSubGroups].pTriMembers = pIndices;
				memcpy(pIndices, tmp_group.pTriMembers, iMembers*sizeof(int));
				pSubGroupTspace[iUniqueSubGroups] =
					EvalTspace(tmp_group.pTriMembers, iMembers, piTriListIn, pTriInfos, pContext, pGroup->iVertexRepresentitive);
				++iUniqueSubGroups;
			}

			// output tspace
			{
				const int iOffs = pTriInfos[f].iTSpacesOffs;
				const int iVert = pTriInfos[f].vert_num[index];
				STSpace * pTS_out = &psTspace[iOffs+iVert];
				assert(pTS_out->iCounter<2);
				assert(((pTriInfos[f].iFlag&ORIENT_PRESERVING)!=0) == pGroup->bOrientPreservering);
				if (pTS_out->iCounter==1)
				{
					*pTS_out = AvgTSpace(pTS_out, &pSubGroupTspace[l]);
					pTS_out->iCounter = 2;	// update counter
					pTS_out->bOrient = pGroup->bOrientPreservering;
				}
				else
				{
					assert(pTS_out->iCounter==0);
					*pTS_out = pSubGroupTspace[l];
					pTS_out->iCounter = 1;	// update counter
					pTS_out->bOrient = pGroup->bOrientPreservering;
				}
			}
		}

		// clean up and offset iUniqueTspaces
		for (s=0; s<iUniqueSubGroups; s++)
			free(pUniSubGroups[s].pTriMembers);
		iUniqueTspaces += iUniqueSubGroups;
	}

	// clean up
	free(pUniSubGroups);
	free(pTmpMembers);
	free(pSubGroupTspace);

	return TTRUE;
}

static STSpace EvalTspace(int face_indices[], const int iFaces, const int piTriListIn[], const STriInfo pTriInfos[],
                          const SMikkTSpaceContext * pContext, const int iVertexRepresentitive)
{
	STSpace res;
	float fAngleSum = 0;
	int face=0;
	res.vOs.x=0.0f; res.vOs.y=0.0f; res.vOs.z=0.0f;
	res.vOt.x=0.0f; res.vOt.y=0.0f; res.vOt.z=0.0f;
	res.fMagS = 0; res.fMagT = 0;

	for (face=0; face<iFaces; face++)
	{
		const int f = face_indices[face];

		// only valid triangles get to add their contribution
		if ( (pTriInfos[f].iFlag&GROUP_WITH_ANY)==0 )
		{
			SVec3 n, vOs, vOt, p0, p1, p2, v1, v2;
			float fCos, fAngle, fMagS, fMagT;
			int i=-1, index=-1, i0=-1, i1=-1, i2=-1;
			if (piTriListIn[3*f+0]==iVertexRepresentitive) i=0;
			else if (piTriListIn[3*f+1]==iVertexRepresentitive) i=1;
			else if (piTriListIn[3*f+2]==iVertexRepresentitive) i=2;
			assert(i>=0 && i<3);

			// project
			index = piTriListIn[3*f+i];
			n = GetNormal(pContext, index);
			vOs = vsub(pTriInfos[f].vOs, vscale(vdot(n,pTriInfos[f].vOs), n));
			vOt = vsub(pTriInfos[f].vOt, vscale(vdot(n,pTriInfos[f].vOt), n));
			if ( VNotZero(vOs) ) vOs = Normalize(vOs);
			if ( VNotZero(vOt) ) vOt = Normalize(vOt);

			i2 = piTriListIn[3*f + (i<2?(i+1):0)];
			i1 = piTriListIn[3*f + i];
			i0 = piTriListIn[3*f + (i>0?(i-1):2)];

			p0 = GetPosition(pContext, i0);
			p1 = GetPosition(pContext, i1);
			p2 = GetPosition(pContext, i2);
			v1 = vsub(p0,p1);
			v2 = vsub(p2,p1);

			// project
			v1 = vsub(v1, vscale(vdot(n,v1),n)); if ( VNotZero(v1) ) v1 = Normalize(v1);
			v2 = vsub(v2, vscale(vdot(n,v2),n)); if ( VNotZero(v2) ) v2 = Normalize(v2);

			// weight contribution by the angle
			// between the two edge vectors
			fCos = vdot(v1,v2); fCos=fCos>1?1:(fCos<(-1) ? (-1) : fCos);
			fAngle = (float) acos(fCos);
			fMagS = pTriInfos[f].fMagS;
			fMagT = pTriInfos[f].fMagT;

			res.vOs=vadd(res.vOs, vscale(fAngle,vOs));
			res.vOt=vadd(res.vOt,vscale(fAngle,vOt));
			res.fMagS+=(fAngle*fMagS);
			res.fMagT+=(fAngle*fMagT);
			fAngleSum += fAngle;
		}
	}

	// normalize
	if ( VNotZero(res.vOs) ) res.vOs = Normalize(res.vOs);
	if ( VNotZero(res.vOt) ) res.vOt = Normalize(res.vOt);
	if (fAngleSum>0)
	{
		res.fMagS /= fAngleSum;
		res.fMagT /= fAngleSum;
	}

	return res;
}

static tbool CompareSubGroups(const SSubGroup * pg1, const SSubGroup * pg2)
{
	tbool bStillSame=TTRUE;
	int i=0;
	if (pg1->iNrFaces!=pg2->iNrFaces) return TFALSE;
	while (i<pg1->iNrFaces && bStillSame)
	{
		bStillSame = pg1->pTriMembers[i]==pg2->pTriMembers[i] ? TTRUE : TFALSE;
		if (bStillSame) ++i;
	}
	return bStillSame;
}

static void QuickSort(int* pSortBuffer, int iLeft, int iRight, unsigned int uSeed)
{
	int iL, iR, n, index, iMid, iTmp;

	// Random
	unsigned int t=uSeed&31;
	t=(uSeed<<t)|(uSeed>>(32-t));
	uSeed=uSeed+t+3;
	// Random end

	iL=iLeft; iR=iRight;
	n = (iR-iL)+1;
	assert(n>=0);
	index = (int) (uSeed%n);

	iMid=pSortBuffer[index + iL];


	do
	{
		while (pSortBuffer[iL] < iMid)
			++iL;
		while (pSortBuffer[iR] > iMid)
			--iR;

		if (iL <= iR)
		{
			iTmp = pSortBuffer[iL];
			pSortBuffer[iL] = pSortBuffer[iR];
			pSortBuffer[iR] = iTmp;
			++iL; --iR;
		}
	}
	while (iL <= iR);

	if (iLeft < iR)
		QuickSort(pSortBuffer, iLeft, iR, uSeed);
	if (iL < iRight)
		QuickSort(pSortBuffer, iL, iRight, uSeed);
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

static void QuickSortEdges(SEdge * pSortBuffer, int iLeft, int iRight, const int channel, unsigned int uSeed);
static void GetEdge(int * i0_out, int * i1_out, int * edgenum_out, const int indices[], const int i0_in, const int i1_in);

static void BuildNeighborsFast(STriInfo pTriInfos[], SEdge * pEdges, const int piTriListIn[], const int iNrTrianglesIn)
{
	// build array of edges
	unsigned int uSeed = INTERNAL_RND_SORT_SEED;				// could replace with a random seed?
	int iEntries=0, iCurStartIndex=-1, f=0, i=0;
	for (f=0; f<iNrTrianglesIn; f++)
		for (i=0; i<3; i++)
		{
			const int i0 = piTriListIn[f*3+i];
			const int i1 = piTriListIn[f*3+(i<2?(i+1):0)];
			pEdges[f*3+i].i0 = i0 < i1 ? i0 : i1;			// put minimum index in i0
			pEdges[f*3+i].i1 = !(i0 < i1) ? i0 : i1;		// put maximum index in i1
			pEdges[f*3+i].f = f;							// record face number
		}

	// sort over all edges by i0, this is the pricy one.
	QuickSortEdges(pEdges, 0, iNrTrianglesIn*3-1, 0, uSeed);	// sort channel 0 which is i0

	// sub sort over i1, should be fast.
	// could replace this with a 64 bit int sort over (i0,i1)
	// with i0 as msb in the quicksort call above.
	iEntries = iNrTrianglesIn*3;
	iCurStartIndex = 0;
	for (i=1; i<iEntries; i++)
	{
		if (pEdges[iCurStartIndex].i0 != pEdges[i].i0)
		{
			const int iL = iCurStartIndex;
			const int iR = i-1;
			//const int iElems = i-iL;
			iCurStartIndex = i;
			QuickSortEdges(pEdges, iL, iR, 1, uSeed);	// sort channel 1 which is i1
		}
	}

	// sub sort over f, which should be fast.
	// this step is to remain compliant with BuildNeighborsSlow() when
	// more than 2 triangles use the same edge (such as a butterfly topology).
	iCurStartIndex = 0;
	for (i=1; i<iEntries; i++)
	{
		if (pEdges[iCurStartIndex].i0 != pEdges[i].i0 || pEdges[iCurStartIndex].i1 != pEdges[i].i1)
		{
			const int iL = iCurStartIndex;
			const int iR = i-1;
			//const int iElems = i-iL;
			iCurStartIndex = i;
			QuickSortEdges(pEdges, iL, iR, 2, uSeed);	// sort channel 2 which is f
		}
	}

	// pair up, adjacent triangles
	for (i=0; i<iEntries; i++)
	{
		const int i0=pEdges[i].i0;
		const int i1=pEdges[i].i1;
		const int f = pEdges[i].f;
		tbool bUnassigned_A;

		int i0_A, i1_A;
		int edgenum_A, edgenum_B=0;	// 0,1 or 2
		GetEdge(&i0_A, &i1_A, &edgenum_A, &piTriListIn[f*3], i0, i1);	// resolve index ordering and edge_num
		bUnassigned_A = pTriInfos[f].FaceNeighbors[edgenum_A] == -1 ? TTRUE : TFALSE;

		if (bUnassigned_A)
		{
			// get true index ordering
			int j=i+1, t;
			tbool bNotFound = TTRUE;
			while (j<iEntries && i0==pEdges[j].i0 && i1==pEdges[j].i1 && bNotFound)
			{
				tbool bUnassigned_B;
				int i0_B, i1_B;
				t = pEdges[j].f;
				// flip i0_B and i1_B
				GetEdge(&i1_B, &i0_B, &edgenum_B, &piTriListIn[t*3], pEdges[j].i0, pEdges[j].i1);	// resolve index ordering and edge_num
				//assert(!(i0_A==i1_B && i1_A==i0_B));
				bUnassigned_B =  pTriInfos[t].FaceNeighbors[edgenum_B]==-1 ? TTRUE : TFALSE;
				if (i0_A==i0_B && i1_A==i1_B && bUnassigned_B)
					bNotFound = TFALSE;
				else
					++j;
			}

			if (!bNotFound)
			{
				int t = pEdges[j].f;
				pTriInfos[f].FaceNeighbors[edgenum_A] = t;
				//assert(pTriInfos[t].FaceNeighbors[edgenum_B]==-1);
				pTriInfos[t].FaceNeighbors[edgenum_B] = f;
			}
		}
	}
}

static void BuildNeighborsSlow(STriInfo pTriInfos[], const int piTriListIn[], const int iNrTrianglesIn)
{
	int f=0, i=0;
	for (f=0; f<iNrTrianglesIn; f++)
	{
		for (i=0; i<3; i++)
		{
			// if unassigned
			if (pTriInfos[f].FaceNeighbors[i] == -1)
			{
				const int i0_A = piTriListIn[f*3+i];
				const int i1_A = piTriListIn[f*3+(i<2?(i+1):0)];

				// search for a neighbor
				tbool bFound = TFALSE;
				int t=0, j=0;
				while (!bFound && t<iNrTrianglesIn)
				{
					if (t!=f)
					{
						j=0;
						while (!bFound && j<3)
						{
							// in rev order
							const int i1_B = piTriListIn[t*3+j];
							const int i0_B = piTriListIn[t*3+(j<2?(j+1):0)];
							//assert(!(i0_A==i1_B && i1_A==i0_B));
							if (i0_A==i0_B && i1_A==i1_B)
								bFound = TTRUE;
							else
								++j;
						}
					}
					
					if (!bFound) ++t;
				}

				// assign neighbors
				if (bFound)
				{
					pTriInfos[f].FaceNeighbors[i] = t;
					//assert(pTriInfos[t].FaceNeighbors[j]==-1);
					pTriInfos[t].FaceNeighbors[j] = f;
				}
			}
		}
	}
}

static void QuickSortEdges(SEdge * pSortBuffer, int iLeft, int iRight, const int channel, unsigned int uSeed)
{
	unsigned int t;
	int iL, iR, n, index, iMid;

	// early out
	SEdge sTmp;
	const int iElems = iRight-iLeft+1;
	if (iElems<2) return;
	else if (iElems==2)
	{
		if (pSortBuffer[iLeft].array[channel] > pSortBuffer[iRight].array[channel])
		{
			sTmp = pSortBuffer[iLeft];
			pSortBuffer[iLeft] = pSortBuffer[iRight];
			pSortBuffer[iRight] = sTmp;
		}
		return;
	}

	// Random
	t=uSeed&31;
	t=(uSeed<<t)|(uSeed>>(32-t));
	uSeed=uSeed+t+3;
	// Random end

	iL=iLeft, iR=iRight;
	n = (iR-iL)+1;
	assert(n>=0);
	index = (int) (uSeed%n);

	iMid=pSortBuffer[index + iL].array[channel];

	do
	{
		while (pSortBuffer[iL].array[channel] < iMid)
			++iL;
		while (pSortBuffer[iR].array[channel] > iMid)
			--iR;

		if (iL <= iR)
		{
			sTmp = pSortBuffer[iL];
			pSortBuffer[iL] = pSortBuffer[iR];
			pSortBuffer[iR] = sTmp;
			++iL; --iR;
		}
	}
	while (iL <= iR);

	if (iLeft < iR)
		QuickSortEdges(pSortBuffer, iLeft, iR, channel, uSeed);
	if (iL < iRight)
		QuickSortEdges(pSortBuffer, iL, iRight, channel, uSeed);
}

// resolve ordering and edge number
static void GetEdge(int * i0_out, int * i1_out, int * edgenum_out, const int indices[], const int i0_in, const int i1_in)
{
	*edgenum_out = -1;
	
	// test if first index is on the edge
	if (indices[0]==i0_in || indices[0]==i1_in)
	{
		// test if second index is on the edge
		if (indices[1]==i0_in || indices[1]==i1_in)
		{
			edgenum_out[0]=0;	// first edge
			i0_out[0]=indices[0];
			i1_out[0]=indices[1];
		}
		else
		{
			edgenum_out[0]=2;	// third edge
			i0_out[0]=indices[2];
			i1_out[0]=indices[0];
		}
	}
	else
	{
		// only second and third index is on the edge
		edgenum_out[0]=1;	// second edge
		i0_out[0]=indices[1];
		i1_out[0]=indices[2];
	}
}


/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// Degenerate triangles ////////////////////////////////////

static void DegenPrologue(STriInfo pTriInfos[], int piTriList_out[], const int iNrTrianglesIn, const int iTotTris)
{
	int iNextGoodTriangleSearchIndex=-1;
	tbool bStillFindingGoodOnes;

	// locate quads with only one good triangle
	int t=0;
	while (t<(iTotTris-1))
	{
		const int iFO_a = pTriInfos[t].iOrgFaceNumber;
		const int iFO_b = pTriInfos[t+1].iOrgFaceNumber;
		if (iFO_a==iFO_b)	// this is a quad
		{
			const tbool bIsDeg_a = (pTriInfos[t].iFlag&MARK_DEGENERATE)!=0 ? TTRUE : TFALSE;
			const tbool bIsDeg_b = (pTriInfos[t+1].iFlag&MARK_DEGENERATE)!=0 ? TTRUE : TFALSE;
			if ((bIsDeg_a^bIsDeg_b)!=0)
			{
				pTriInfos[t].iFlag |= QUAD_ONE_DEGEN_TRI;
				pTriInfos[t+1].iFlag |= QUAD_ONE_DEGEN_TRI;
			}
			t += 2;
		}
		else
			++t;
	}

	// reorder list so all degen triangles are moved to the back
	// without reordering the good triangles
	iNextGoodTriangleSearchIndex = 1;
	t=0;
	bStillFindingGoodOnes = TTRUE;
	while (t<iNrTrianglesIn && bStillFindingGoodOnes)
	{
		const tbool bIsGood = (pTriInfos[t].iFlag&MARK_DEGENERATE)==0 ? TTRUE : TFALSE;
		if (bIsGood)
		{
			if (iNextGoodTriangleSearchIndex < (t+2))
				iNextGoodTriangleSearchIndex = t+2;
		}
		else
		{
			int t0, t1;
			// search for the first good triangle.
			tbool bJustADegenerate = TTRUE;
			while (bJustADegenerate && iNextGoodTriangleSearchIndex<iTotTris)
			{
				const tbool bIsGood = (pTriInfos[iNextGoodTriangleSearchIndex].iFlag&MARK_DEGENERATE)==0 ? TTRUE : TFALSE;
				if (bIsGood) bJustADegenerate=TFALSE;
				else ++iNextGoodTriangleSearchIndex;
			}

			t0 = t;
			t1 = iNextGoodTriangleSearchIndex;
			++iNextGoodTriangleSearchIndex;
			assert(iNextGoodTriangleSearchIndex > (t+1));

			// swap triangle t0 and t1
			if (!bJustADegenerate)
			{
				int i=0;
				for (i=0; i<3; i++)
				{
					const int index = piTriList_out[t0*3+i];
					piTriList_out[t0*3+i] = piTriList_out[t1*3+i];
					piTriList_out[t1*3+i] = index;
				}
				{
					const STriInfo tri_info = pTriInfos[t0];
					pTriInfos[t0] = pTriInfos[t1];
					pTriInfos[t1] = tri_info;
				}
			}
			else
				bStillFindingGoodOnes = TFALSE;	// this is not supposed to happen
		}

		if (bStillFindingGoodOnes) ++t;
	}

	assert(bStillFindingGoodOnes);	// code will still work.
	assert(iNrTrianglesIn == t);
}

static void DegenEpilogue(STSpace psTspace[], STriInfo pTriInfos[], int piTriListIn[], const SMikkTSpaceContext * pContext, const int iNrTrianglesIn, const int iTotTris)
{
	int t=0, i=0;
	// deal with degenerate triangles
	// punishment for degenerate triangles is O(N^2)
	for (t=iNrTrianglesIn; t<iTotTris; t++)
	{
		// degenerate triangles on a quad with one good triangle are skipped
		// here but processed in the next loop
		const tbool bSkip = (pTriInfos[t].iFlag&QUAD_ONE_DEGEN_TRI)!=0 ? TTRUE : TFALSE;

		if (!bSkip)
		{
			for (i=0; i<3; i++)
			{
				const int index1 = piTriListIn[t*3+i];
				// search through the good triangles
				tbool bNotFound = TTRUE;
				int j=0;
				while (bNotFound && j<(3*iNrTrianglesIn))
				{
					const int index2 = piTriListIn[j];
					if (index1==index2) bNotFound=TFALSE;
					else ++j;
				}

				if (!bNotFound)
				{
					const int iTri = j/3;
					const int iVert = j%3;
					const int iSrcVert=pTriInfos[iTri].vert_num[iVert];
					const int iSrcOffs=pTriInfos[iTri].iTSpacesOffs;
					const int iDstVert=pTriInfos[t].vert_num[i];
					const int iDstOffs=pTriInfos[t].iTSpacesOffs;
					
					// copy tspace
					psTspace[iDstOffs+iDstVert] = psTspace[iSrcOffs+iSrcVert];
				}
			}
		}
	}

	// deal with degenerate quads with one good triangle
	for (t=0; t<iNrTrianglesIn; t++)
	{
		// this triangle belongs to a quad where the
		// other triangle is degenerate
		if ( (pTriInfos[t].iFlag&QUAD_ONE_DEGEN_TRI)!=0 )
		{
			SVec3 vDstP;
			int iOrgF=-1, i=0;
			tbool bNotFound;
			unsigned char * pV = pTriInfos[t].vert_num;
			int iFlag = (1<<pV[0]) | (1<<pV[1]) | (1<<pV[2]);
			int iMissingIndex = 0;
			if ((iFlag&2)==0) iMissingIndex=1;
			else if ((iFlag&4)==0) iMissingIndex=2;
			else if ((iFlag&8)==0) iMissingIndex=3;

			iOrgF = pTriInfos[t].iOrgFaceNumber;
			vDstP = GetPosition(pContext, MakeIndex(iOrgF, iMissingIndex));
			bNotFound = TTRUE;
			i=0;
			while (bNotFound && i<3)
			{
				const int iVert = pV[i];
				const SVec3 vSrcP = GetPosition(pContext, MakeIndex(iOrgF, iVert));
				if (veq(vSrcP, vDstP)==TTRUE)
				{
					const int iOffs = pTriInfos[t].iTSpacesOffs;
					psTspace[iOffs+iMissingIndex] = psTspace[iOffs+iVert];
					bNotFound=TFALSE;
				}
				else
					++i;
			}
			assert(!bNotFound);
		}
	}
}
