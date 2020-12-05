/*
Open Asset Import Library (assimp)
----------------------------------------------------------------------

Copyright (c) 2006-2019, assimp team


All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the
following conditions are met:

* Redistributions of source code must retain the above
  copyright notice, this list of conditions and the
  following disclaimer.

* Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the
  following disclaimer in the documentation and/or other
  materials provided with the distribution.

* Neither the name of the assimp team, nor the names of its
  contributors may be used to endorse or promote products
  derived from this software without specific prior
  written permission of the assimp team.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

----------------------------------------------------------------------
*/

/** Small helper classes to optimize finding vertices close to a given location
 */
#pragma once
#ifndef AI_D3DSSPATIALSORT_H_INC
#define AI_D3DSSPATIALSORT_H_INC

#ifdef __GNUC__
#   pragma GCC system_header
#endif

#include <assimp/types.h>
#include <vector>
#include <stdint.h>

namespace Assimp    {

// ----------------------------------------------------------------------------------
/** Specialized version of SpatialSort to support smoothing groups
 *  This is used in by the 3DS, ASE and LWO loaders. 3DS and ASE share their
 *  normal computation code in SmoothingGroups.inl, the LWO loader has its own
 *  implementation to handle all details of its file format correctly.
 */
// ----------------------------------------------------------------------------------
class ASSIMP_API SGSpatialSort
{
public:

    SGSpatialSort();

    // -------------------------------------------------------------------
    /** Construction from a given face array, handling smoothing groups
     *  properly
     */
    explicit SGSpatialSort(const std::vector<aiVector3D>& vPositions);

    // -------------------------------------------------------------------
    /** Add a vertex to the spatial sort
     * @param vPosition Vertex position to be added
     * @param index Index of the vrtex
     * @param smoothingGroup SmoothingGroup for this vertex
     */
    void Add(const aiVector3D& vPosition, unsigned int index,
        unsigned int smoothingGroup);

    // -------------------------------------------------------------------
    /** Prepare the spatial sorter for use. This step runs in O(logn)
     */
    void Prepare();

    /** Destructor */
    ~SGSpatialSort();

    // -------------------------------------------------------------------
    /** Returns an iterator for all positions close to the given position.
     * @param pPosition The position to look for vertices.
     * @param pSG Only included vertices with at least one shared smooth group
     * @param pRadius Maximal distance from the position a vertex may have
     *   to be counted in.
     * @param poResults The container to store the indices of the found
     *   positions. Will be emptied by the call so it may contain anything.
     * @param exactMatch Specifies whether smoothing groups are bit masks
     *   (false) or integral values (true). In the latter case, a vertex
     *   cannot belong to more than one smoothing group.
     * @return An iterator to iterate over all vertices in the given area.
     */
    // -------------------------------------------------------------------
    void FindPositions( const aiVector3D& pPosition, uint32_t pSG,
        float pRadius, std::vector<unsigned int>& poResults,
        bool exactMatch = false) const;

protected:
    /** Normal of the sorting plane, normalized. The center is always at (0, 0, 0) */
    aiVector3D mPlaneNormal;

    // -------------------------------------------------------------------
    /** An entry in a spatially sorted position array. Consists of a
     *  vertex index, its position and its pre-calculated distance from
     *  the reference plane */
    // -------------------------------------------------------------------
    struct Entry {
        unsigned int mIndex;    ///< The vertex referred by this entry
        aiVector3D mPosition;   ///< Position
        uint32_t mSmoothGroups;
        float mDistance;        ///< Distance of this vertex to the sorting plane

        Entry() AI_NO_EXCEPT
        : mIndex(0)
        , mPosition()
        , mSmoothGroups(0)
        , mDistance(0.0f) {
            // empty
        }

        Entry( unsigned int pIndex, const aiVector3D& pPosition, float pDistance,uint32_t pSG)
        : mIndex( pIndex)
        , mPosition( pPosition)
        , mSmoothGroups(pSG)
        , mDistance( pDistance) {
            // empty
        }

        bool operator < (const Entry& e) const {
            return mDistance < e.mDistance;
        }
    };

    // all positions, sorted by distance to the sorting plane
    std::vector<Entry> mPositions;
};

} // end of namespace Assimp

#endif // AI_SPATIALSORT_H_INC
