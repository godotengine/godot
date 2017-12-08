// This code is in the public domain -- castanyo@yahoo.es

#pragma once
#ifndef NV_MESH_RASTER_H
#define NV_MESH_RASTER_H

/** @file Raster.h
 * @brief Rasterization library.
 *
 * This is just a standard scanline rasterizer that I took from one of my old
 * projects. The perspective correction wasn't necessary so I just removed it.
**/

#include "nvmath/Vector.h"
#include "nvmesh/nvmesh.h"

namespace nv
{

    namespace Raster 
    {
        enum Mode {
            Mode_Nearest,
            Mode_Antialiased,
            //Mode_Conservative
        };


        /// A callback to sample the environment. Return false to terminate rasterization.
        typedef bool (NV_CDECL * SamplingCallback)(void * param, int x, int y, Vector3::Arg bar, Vector3::Arg dx, Vector3::Arg dy, float coverage);

        // Process the given triangle. Returns false if rasterization was interrupted by the callback.
        NVMESH_API bool drawTriangle(Mode mode, Vector2::Arg extents, bool enableScissors, const Vector2 v[3], SamplingCallback cb, void * param);

        // Process the given quad. Returns false if rasterization was interrupted by the callback.
        NVMESH_API bool drawQuad(Mode mode, Vector2::Arg extents, bool enableScissors, const Vector2 v[4], SamplingCallback cb, void * param);

        typedef bool (NV_CDECL * LineSamplingCallback)(void * param, int x, int y, float t, float d);    // t is the position along the segment, d is the distance to the line.

        // Process the given line.
        NVMESH_API void drawLine(bool antialias, Vector2::Arg extents, bool enableScissors, const Vector2 v[2], LineSamplingCallback cb, void * param);

        // Draw vertical or horizontal segments. For degenerate triangles.
        //NVMESH_API void drawSegment(Vector2::Arg extents, bool enableScissors, const Vector2 v[2], SamplingCallback cb, void * param);
    }
}


#endif // NV_MESH_RASTER_H
