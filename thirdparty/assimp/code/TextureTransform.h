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

/** @file Definition of a helper step that processes texture transformations */
#ifndef AI_TEXTURE_TRANSFORM_H_INCLUDED
#define AI_TEXTURE_TRANSFORM_H_INCLUDED

#include <assimp/BaseImporter.h>
#include "BaseProcess.h"

#include <assimp/material.h>
#include <list>

struct aiNode;
struct aiMaterial;

namespace Assimp    {

#define AI_TT_UV_IDX_LOCK_TBD   0xffffffff
#define AI_TT_UV_IDX_LOCK_NONE  0xeeeeeeee


#define AI_TT_ROTATION_EPSILON  ((float)AI_DEG_TO_RAD(0.5))

// ---------------------------------------------------------------------------
/** Small helper structure representing a shortcut into the material list
 *  to be able to update some values quickly.
*/
struct TTUpdateInfo {
    TTUpdateInfo() AI_NO_EXCEPT
    : directShortcut(nullptr)
    , mat(nullptr)
    , semantic(0)
    , index(0) {
        // empty
    }

    //! Direct shortcut, if available
    unsigned int* directShortcut;

    //! Material
    aiMaterial *mat;

    //! Texture type and index
    unsigned int semantic, index;
};


// ---------------------------------------------------------------------------
/** Helper class representing texture coordinate transformations
*/
struct STransformVecInfo : public aiUVTransform {
    STransformVecInfo() AI_NO_EXCEPT
    : uvIndex(0)
    , mapU(aiTextureMapMode_Wrap)
    , mapV(aiTextureMapMode_Wrap)
    , lockedPos(AI_TT_UV_IDX_LOCK_NONE) {
        // empty
    }

    //! Source texture coordinate index
    unsigned int uvIndex;

    //! Texture mapping mode in the u, v direction
    aiTextureMapMode mapU,mapV;

    //! Locked destination UV index
    //! AI_TT_UV_IDX_LOCK_TBD - to be determined
    //! AI_TT_UV_IDX_LOCK_NONE - none (default)
    unsigned int lockedPos;

    //! Update info - shortcuts into all materials
    //! that are referencing this transform setup
    std::list<TTUpdateInfo> updateList;


    // -------------------------------------------------------------------
    /** Compare two transform setups
    */
    inline bool operator== (const STransformVecInfo& other) const
    {
        // We use a small epsilon here
        const static float epsilon = 0.05f;

        if (std::fabs( mTranslation.x - other.mTranslation.x ) > epsilon ||
            std::fabs( mTranslation.y - other.mTranslation.y ) > epsilon)
        {
            return false;
        }

        if (std::fabs( mScaling.x - other.mScaling.x ) > epsilon ||
            std::fabs( mScaling.y - other.mScaling.y ) > epsilon)
        {
            return false;
        }

        if (std::fabs( mRotation - other.mRotation) > epsilon)
        {
            return false;
        }
        return true;
    }

    inline bool operator!= (const STransformVecInfo& other) const
    {
            return !(*this == other);
    }


    // -------------------------------------------------------------------
    /** Returns whether this is an untransformed texture coordinate set
    */
    inline bool IsUntransformed() const
    {
        return (1.0f == mScaling.x && 1.f == mScaling.y &&
            !mTranslation.x && !mTranslation.y &&
            mRotation < AI_TT_ROTATION_EPSILON);
    }

    // -------------------------------------------------------------------
    /** Build a 3x3 matrix from the transformations
    */
    inline void GetMatrix(aiMatrix3x3& mOut)
    {
        mOut = aiMatrix3x3();

        if (1.0f != mScaling.x || 1.0f != mScaling.y)
        {
            aiMatrix3x3 mScale;
            mScale.a1 = mScaling.x;
            mScale.b2 = mScaling.y;
            mOut = mScale;
        }
        if (mRotation)
        {
            aiMatrix3x3 mRot;
            mRot.a1 = mRot.b2 = std::cos(mRotation);
            mRot.a2 = mRot.b1 = std::sin(mRotation);
            mRot.a2 = -mRot.a2;
            mOut *= mRot;
        }
        if (mTranslation.x || mTranslation.y)
        {
            aiMatrix3x3 mTrans;
            mTrans.a3 = mTranslation.x;
            mTrans.b3 = mTranslation.y;
            mOut *= mTrans;
        }
    }
};


// ---------------------------------------------------------------------------
/** Helper step to compute final UV coordinate sets if there are scalings
 *  or rotations in the original data read from the file.
*/
class TextureTransformStep : public BaseProcess
{
public:

    TextureTransformStep();
    ~TextureTransformStep();

public:

    // -------------------------------------------------------------------
    bool IsActive( unsigned int pFlags) const;

    // -------------------------------------------------------------------
    void Execute( aiScene* pScene);

    // -------------------------------------------------------------------
    void SetupProperties(const Importer* pImp);


protected:


    // -------------------------------------------------------------------
    /** Preprocess a specific UV transformation setup
     *
     *  @param info Transformation setup to be preprocessed.
    */
    void PreProcessUVTransform(STransformVecInfo& info);

private:

    unsigned int configFlags;
};

}

#endif //! AI_TEXTURE_TRANSFORM_H_INCLUDED
