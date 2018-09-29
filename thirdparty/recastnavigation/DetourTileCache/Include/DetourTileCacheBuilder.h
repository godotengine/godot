//
// Copyright (c) 2009-2010 Mikko Mononen memon@inside.org
//
// This software is provided 'as-is', without any express or implied
// warranty.  In no event will the authors be held liable for any damages
// arising from the use of this software.
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would be
//    appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.
//

#ifndef DETOURTILECACHEBUILDER_H
#define DETOURTILECACHEBUILDER_H

#include "DetourAlloc.h"
#include "DetourStatus.h"

static const int DT_TILECACHE_MAGIC = 'D'<<24 | 'T'<<16 | 'L'<<8 | 'R'; ///< 'DTLR';
static const int DT_TILECACHE_VERSION = 1;

static const unsigned char DT_TILECACHE_NULL_AREA = 0;
static const unsigned char DT_TILECACHE_WALKABLE_AREA = 63;
static const unsigned short DT_TILECACHE_NULL_IDX = 0xffff;

struct dtTileCacheLayerHeader
{
	int magic;								///< Data magic
	int version;							///< Data version
	int tx,ty,tlayer;
	float bmin[3], bmax[3];
	unsigned short hmin, hmax;				///< Height min/max range
	unsigned char width, height;			///< Dimension of the layer.
	unsigned char minx, maxx, miny, maxy;	///< Usable sub-region.
};

struct dtTileCacheLayer
{
	dtTileCacheLayerHeader* header;
	unsigned char regCount;					///< Region count.
	unsigned char* heights;
	unsigned char* areas;
	unsigned char* cons;
	unsigned char* regs;
};

struct dtTileCacheContour
{
	int nverts;
	unsigned char* verts;
	unsigned char reg;
	unsigned char area;
};

struct dtTileCacheContourSet
{
	int nconts;
	dtTileCacheContour* conts;
};

struct dtTileCachePolyMesh
{
	int nvp;
	int nverts;				///< Number of vertices.
	int npolys;				///< Number of polygons.
	unsigned short* verts;	///< Vertices of the mesh, 3 elements per vertex.
	unsigned short* polys;	///< Polygons of the mesh, nvp*2 elements per polygon.
	unsigned short* flags;	///< Per polygon flags.
	unsigned char* areas;	///< Area ID of polygons.
};


struct dtTileCacheAlloc
{
	virtual ~dtTileCacheAlloc() {}

	virtual void reset() {}
	
	virtual void* alloc(const size_t size)
	{
		return dtAlloc(size, DT_ALLOC_TEMP);
	}
	
	virtual void free(void* ptr)
	{
		dtFree(ptr);
	}
};

struct dtTileCacheCompressor
{
	virtual ~dtTileCacheCompressor() { }

	virtual int maxCompressedSize(const int bufferSize) = 0;
	virtual dtStatus compress(const unsigned char* buffer, const int bufferSize,
							  unsigned char* compressed, const int maxCompressedSize, int* compressedSize) = 0;
	virtual dtStatus decompress(const unsigned char* compressed, const int compressedSize,
								unsigned char* buffer, const int maxBufferSize, int* bufferSize) = 0;
};


dtStatus dtBuildTileCacheLayer(dtTileCacheCompressor* comp,
							   dtTileCacheLayerHeader* header,
							   const unsigned char* heights,
							   const unsigned char* areas,
							   const unsigned char* cons,
							   unsigned char** outData, int* outDataSize);

void dtFreeTileCacheLayer(dtTileCacheAlloc* alloc, dtTileCacheLayer* layer);

dtStatus dtDecompressTileCacheLayer(dtTileCacheAlloc* alloc, dtTileCacheCompressor* comp,
									unsigned char* compressed, const int compressedSize,
									dtTileCacheLayer** layerOut);

dtTileCacheContourSet* dtAllocTileCacheContourSet(dtTileCacheAlloc* alloc);
void dtFreeTileCacheContourSet(dtTileCacheAlloc* alloc, dtTileCacheContourSet* cset);

dtTileCachePolyMesh* dtAllocTileCachePolyMesh(dtTileCacheAlloc* alloc);
void dtFreeTileCachePolyMesh(dtTileCacheAlloc* alloc, dtTileCachePolyMesh* lmesh);

dtStatus dtMarkCylinderArea(dtTileCacheLayer& layer, const float* orig, const float cs, const float ch,
							const float* pos, const float radius, const float height, const unsigned char areaId);

dtStatus dtMarkBoxArea(dtTileCacheLayer& layer, const float* orig, const float cs, const float ch,
					   const float* bmin, const float* bmax, const unsigned char areaId);

dtStatus dtMarkBoxArea(dtTileCacheLayer& layer, const float* orig, const float cs, const float ch,
					   const float* center, const float* halfExtents, const float* rotAux, const unsigned char areaId);

dtStatus dtBuildTileCacheRegions(dtTileCacheAlloc* alloc,
								 dtTileCacheLayer& layer,
								 const int walkableClimb);

dtStatus dtBuildTileCacheContours(dtTileCacheAlloc* alloc,
								  dtTileCacheLayer& layer,
								  const int walkableClimb, 	const float maxError,
								  dtTileCacheContourSet& lcset);

dtStatus dtBuildTileCachePolyMesh(dtTileCacheAlloc* alloc,
								  dtTileCacheContourSet& lcset,
								  dtTileCachePolyMesh& mesh);

/// Swaps the endianess of the compressed tile data's header (#dtTileCacheLayerHeader).
/// Tile layer data does not need endian swapping as it consits only of bytes.
///  @param[in,out]	data		The tile data array.
///  @param[in]		dataSize	The size of the data array.
bool dtTileCacheHeaderSwapEndian(unsigned char* data, const int dataSize);


#endif // DETOURTILECACHEBUILDER_H
