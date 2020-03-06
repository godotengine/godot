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

/** @file Defines a helper class for the ASE and 3DS loaders to
 help them compute camera and spot light animation channels */
#ifndef AI_TARGET_ANIMATION_H_INC
#define AI_TARGET_ANIMATION_H_INC

#include <assimp/anim.h>
#include <vector>

namespace Assimp    {



// ---------------------------------------------------------------------------
/** Helper class to iterate through all keys in an animation channel.
 *
 *  Missing tracks are interpolated. This is a helper class for
 *  TargetAnimationHelper, but it can be freely used for other purposes.
*/
class KeyIterator
{
public:


    // ------------------------------------------------------------------
    /** Constructs a new key iterator
     *
     *  @param _objPos Object position track. May be NULL.
     *  @param _targetObjPos Target object position track. May be NULL.
     *  @param defaultObjectPos Default object position to be used if
     *    no animated track is available. May be NULL.
     *  @param defaultTargetPos Default target position to be used if
     *    no animated track is available. May be NULL.
     */
    KeyIterator(const std::vector<aiVectorKey>* _objPos,
        const std::vector<aiVectorKey>* _targetObjPos,
        const aiVector3D*  defaultObjectPos = NULL,
        const aiVector3D*  defaultTargetPos = NULL);

    // ------------------------------------------------------------------
    /** Returns true if all keys have been processed
     */
    bool Finished() const
        {return reachedEnd;}

    // ------------------------------------------------------------------
    /** Increment the iterator
     */
    void operator++();
    inline void operator++(int)
        {return ++(*this);}



    // ------------------------------------------------------------------
    /** Getters to retrieve the current state of the iterator
     */
    inline const aiVector3D& GetCurPosition() const
        {return curPosition;}

    inline const aiVector3D& GetCurTargetPosition() const
        {return curTargetPosition;}

    inline double GetCurTime() const
        {return curTime;}

private:

    //! Did we reach the end?
    bool reachedEnd;

    //! Represents the current position of the iterator
    aiVector3D curPosition, curTargetPosition;

    double curTime;

    //! Input tracks and the next key to process
    const std::vector<aiVectorKey>* objPos,*targetObjPos;

    unsigned int nextObjPos, nextTargetObjPos;
    std::vector<aiVectorKey> defaultObjPos,defaultTargetObjPos;
};

// ---------------------------------------------------------------------------
/** Helper class for the 3DS and ASE loaders to compute camera and spot light
 *  animations.
 *
 * 3DS and ASE store the differently to Assimp - there is an animation
 * channel for the camera/spot light itself and a separate position
 * animation channels specifying the position of the camera/spot light
 * look-at target */
class TargetAnimationHelper
{
public:

    TargetAnimationHelper()
        :   targetPositions     (NULL)
        ,   objectPositions     (NULL)
    {}


    // ------------------------------------------------------------------
    /** Sets the target animation channel
     *
     *  This channel specifies the position of the camera/spot light
     *  target at a specific position.
     *
     *  @param targetPositions Translation channel*/
    void SetTargetAnimationChannel (const
        std::vector<aiVectorKey>* targetPositions);


    // ------------------------------------------------------------------
    /** Sets the main animation channel
     *
     *  @param objectPositions Translation channel */
    void SetMainAnimationChannel ( const
        std::vector<aiVectorKey>* objectPositions);

    // ------------------------------------------------------------------
    /** Sets the main animation channel to a fixed value
     *
     *  @param fixed Fixed value for the main animation channel*/
    void SetFixedMainAnimationChannel(const aiVector3D& fixed);


    // ------------------------------------------------------------------
    /** Computes final animation channels
     * @param distanceTrack Receive camera translation keys ... != NULL. */
    void Process( std::vector<aiVectorKey>* distanceTrack );


private:

    const std::vector<aiVectorKey>* targetPositions,*objectPositions;
    aiVector3D fixedMain;
};


} // ! end namespace Assimp

#endif // include guard
