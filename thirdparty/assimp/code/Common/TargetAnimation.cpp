/*
Open Asset Import Library (assimp)
----------------------------------------------------------------------

Copyright (c) 2006-2020, assimp team


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

#include "TargetAnimation.h"
#include <algorithm>
#include <assimp/ai_assert.h>

using namespace Assimp;


// ------------------------------------------------------------------------------------------------
KeyIterator::KeyIterator(const std::vector<aiVectorKey>* _objPos,
    const std::vector<aiVectorKey>* _targetObjPos,
    const aiVector3D*  defaultObjectPos /*= NULL*/,
    const aiVector3D*  defaultTargetPos /*= NULL*/)

        :   reachedEnd      (false)
        ,   curTime         (-1.)
        ,   objPos          (_objPos)
        ,   targetObjPos    (_targetObjPos)
        ,   nextObjPos      (0)
        ,   nextTargetObjPos(0)
{
    // Generate default transformation tracks if necessary
    if (!objPos || objPos->empty())
    {
        defaultObjPos.resize(1);
        defaultObjPos.front().mTime  = 10e10;

        if (defaultObjectPos)
            defaultObjPos.front().mValue = *defaultObjectPos;

        objPos = & defaultObjPos;
    }
    if (!targetObjPos || targetObjPos->empty())
    {
        defaultTargetObjPos.resize(1);
        defaultTargetObjPos.front().mTime  = 10e10;

        if (defaultTargetPos)
            defaultTargetObjPos.front().mValue = *defaultTargetPos;

        targetObjPos = & defaultTargetObjPos;
    }
}

// ------------------------------------------------------------------------------------------------
template <class T>
inline T Interpolate(const T& one, const T& two, ai_real val)
{
    return one + (two-one)*val;
}

// ------------------------------------------------------------------------------------------------
void KeyIterator::operator ++()
{
    // If we are already at the end of all keyframes, return
    if (reachedEnd) {
        return;
    }

    // Now search in all arrays for the time value closest
    // to our current position on the time line
    double d0,d1;

    d0 = objPos->at      ( std::min ( nextObjPos, static_cast<unsigned int>(objPos->size()-1))             ).mTime;
    d1 = targetObjPos->at( std::min ( nextTargetObjPos, static_cast<unsigned int>(targetObjPos->size()-1)) ).mTime;

    // Easiest case - all are identical. In this
    // case we don't need to interpolate so we can
    // return earlier
    if ( d0 == d1 )
    {
        curTime = d0;
        curPosition = objPos->at(nextObjPos).mValue;
        curTargetPosition = targetObjPos->at(nextTargetObjPos).mValue;

        // increment counters
        if (objPos->size() != nextObjPos-1)
            ++nextObjPos;

        if (targetObjPos->size() != nextTargetObjPos-1)
            ++nextTargetObjPos;
    }

    // An object position key is closest to us
    else if (d0 < d1)
    {
        curTime = d0;

        // interpolate the other
        if (1 == targetObjPos->size() || !nextTargetObjPos) {
            curTargetPosition = targetObjPos->at(0).mValue;
        }
        else
        {
            const aiVectorKey& last  = targetObjPos->at(nextTargetObjPos);
            const aiVectorKey& first = targetObjPos->at(nextTargetObjPos-1);

            curTargetPosition = Interpolate(first.mValue, last.mValue, (ai_real) (
                (curTime-first.mTime) / (last.mTime-first.mTime) ));
        }

        if (objPos->size() != nextObjPos-1)
            ++nextObjPos;
    }
    // A target position key is closest to us
    else
    {
        curTime = d1;

        // interpolate the other
        if (1 == objPos->size() || !nextObjPos) {
            curPosition = objPos->at(0).mValue;
        }
        else
        {
            const aiVectorKey& last  = objPos->at(nextObjPos);
            const aiVectorKey& first = objPos->at(nextObjPos-1);

            curPosition = Interpolate(first.mValue, last.mValue, (ai_real) (
                (curTime-first.mTime) / (last.mTime-first.mTime)));
        }

        if (targetObjPos->size() != nextTargetObjPos-1)
            ++nextTargetObjPos;
    }

    if (nextObjPos >= objPos->size()-1 &&
        nextTargetObjPos >= targetObjPos->size()-1)
    {
        // We reached the very last keyframe
        reachedEnd = true;
    }
}

// ------------------------------------------------------------------------------------------------
void TargetAnimationHelper::SetTargetAnimationChannel (
    const std::vector<aiVectorKey>* _targetPositions)
{
    ai_assert(NULL != _targetPositions);
    targetPositions = _targetPositions;
}

// ------------------------------------------------------------------------------------------------
void TargetAnimationHelper::SetMainAnimationChannel (
    const std::vector<aiVectorKey>* _objectPositions)
{
    ai_assert(NULL != _objectPositions);
    objectPositions = _objectPositions;
}

// ------------------------------------------------------------------------------------------------
void TargetAnimationHelper::SetFixedMainAnimationChannel(
    const aiVector3D& fixed)
{
    objectPositions = NULL; // just to avoid confusion
    fixedMain = fixed;
}

// ------------------------------------------------------------------------------------------------
void TargetAnimationHelper::Process(std::vector<aiVectorKey>* distanceTrack)
{
    ai_assert(NULL != targetPositions && NULL != distanceTrack);

    // TODO: in most cases we won't need the extra array
    std::vector<aiVectorKey>  real;

    std::vector<aiVectorKey>* fill = (distanceTrack == objectPositions ? &real : distanceTrack);
    fill->reserve(std::max( objectPositions->size(), targetPositions->size() ));

    // Iterate through all object keys and interpolate their values if necessary.
    // Then get the corresponding target position, compute the difference
    // vector between object and target position. Then compute a rotation matrix
    // that rotates the base vector of the object coordinate system at that time
    // to match the diff vector.

    KeyIterator iter(objectPositions,targetPositions,&fixedMain);
    for (;!iter.Finished();++iter)
    {
        const aiVector3D&  position  = iter.GetCurPosition();
        const aiVector3D&  tposition = iter.GetCurTargetPosition();

        // diff vector
        aiVector3D diff = tposition - position;
        ai_real f = diff.Length();

        // output distance vector
        if (f)
        {
            fill->push_back(aiVectorKey());
            aiVectorKey& v = fill->back();
            v.mTime  = iter.GetCurTime();
            v.mValue = diff;

            diff /= f;
        }
        else
        {
            // FIXME: handle this
        }

        // diff is now the vector in which our camera is pointing
    }

    if (real.size()) {
        *distanceTrack = real;
    }
}
