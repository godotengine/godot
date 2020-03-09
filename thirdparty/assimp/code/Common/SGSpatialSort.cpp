/*
---------------------------------------------------------------------------
Open Asset Import Library (assimp)
---------------------------------------------------------------------------

Copyright (c) 2006-2019, assimp team



All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the following
conditions are met:

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
---------------------------------------------------------------------------
*/

/** @file Implementation of the helper class to quickly find
vertices close to a given position. Special implementation for
the 3ds loader handling smooth groups correctly  */

#include <assimp/SGSpatialSort.h>

using namespace Assimp;

// ------------------------------------------------------------------------------------------------
SGSpatialSort::SGSpatialSort()
{
    // define the reference plane. We choose some arbitrary vector away from all basic axises
    // in the hope that no model spreads all its vertices along this plane.
    mPlaneNormal.Set( 0.8523f, 0.34321f, 0.5736f);
    mPlaneNormal.Normalize();
}
// ------------------------------------------------------------------------------------------------
// Destructor
SGSpatialSort::~SGSpatialSort()
{
    // nothing to do here, everything destructs automatically
}
// ------------------------------------------------------------------------------------------------
void SGSpatialSort::Add(const aiVector3D& vPosition, unsigned int index,
    unsigned int smoothingGroup)
{
    // store position by index and distance
    float distance = vPosition * mPlaneNormal;
    mPositions.push_back( Entry( index, vPosition,
        distance, smoothingGroup));
}
// ------------------------------------------------------------------------------------------------
void SGSpatialSort::Prepare()
{
    // now sort the array ascending by distance.
    std::sort( this->mPositions.begin(), this->mPositions.end());
}
// ------------------------------------------------------------------------------------------------
// Returns an iterator for all positions close to the given position.
void SGSpatialSort::FindPositions( const aiVector3D& pPosition,
    uint32_t pSG,
    float pRadius,
    std::vector<unsigned int>& poResults,
    bool exactMatch /*= false*/) const
{
    float dist = pPosition * mPlaneNormal;
    float minDist = dist - pRadius, maxDist = dist + pRadius;

    // clear the array
    poResults.clear();

    // quick check for positions outside the range
    if( mPositions.empty() )
        return;
    if( maxDist < mPositions.front().mDistance)
        return;
    if( minDist > mPositions.back().mDistance)
        return;

    // do a binary search for the minimal distance to start the iteration there
    unsigned int index = (unsigned int)mPositions.size() / 2;
    unsigned int binaryStepSize = (unsigned int)mPositions.size() / 4;
    while( binaryStepSize > 1)
    {
        if( mPositions[index].mDistance < minDist)
            index += binaryStepSize;
        else
            index -= binaryStepSize;

        binaryStepSize /= 2;
    }

    // depending on the direction of the last step we need to single step a bit back or forth
    // to find the actual beginning element of the range
    while( index > 0 && mPositions[index].mDistance > minDist)
        index--;
    while( index < (mPositions.size() - 1) && mPositions[index].mDistance < minDist)
        index++;

    // Mow start iterating from there until the first position lays outside of the distance range.
    // Add all positions inside the distance range within the given radius to the result aray

    float squareEpsilon = pRadius * pRadius;
    std::vector<Entry>::const_iterator it  = mPositions.begin() + index;
    std::vector<Entry>::const_iterator end = mPositions.end();

    if (exactMatch)
    {
        while( it->mDistance < maxDist)
        {
            if((it->mPosition - pPosition).SquareLength() < squareEpsilon && it->mSmoothGroups == pSG)
            {
                poResults.push_back( it->mIndex);
            }
            ++it;
            if( end == it )break;
        }
    }
    else
    {
        // if the given smoothing group is 0, we'll return all surrounding vertices
        if (!pSG)
        {
            while( it->mDistance < maxDist)
            {
                if((it->mPosition - pPosition).SquareLength() < squareEpsilon)
                    poResults.push_back( it->mIndex);
                ++it;
                if( end == it)break;
            }
        }
        else while( it->mDistance < maxDist)
        {
            if((it->mPosition - pPosition).SquareLength() < squareEpsilon &&
                (it->mSmoothGroups & pSG || !it->mSmoothGroups))
            {
                poResults.push_back( it->mIndex);
            }
            ++it;
            if( end == it)break;
        }
    }
}


