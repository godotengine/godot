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

/** 
  * @file   anim.h
  * @brief  Defines the data structures in which the imported animations
  *         are returned.
  */
#pragma once
#ifndef AI_ANIM_H_INC
#define AI_ANIM_H_INC

#include <assimp/types.h>
#include <assimp/quaternion.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
/** A time-value pair specifying a certain 3D vector for the given time. */
struct aiVectorKey
{
    /** The time of this key */
    double mTime;

    /** The value of this key */
    C_STRUCT aiVector3D mValue;

#ifdef __cplusplus

    /// @brief  The default constructor.
    aiVectorKey() AI_NO_EXCEPT
    : mTime( 0.0 )
    , mValue() {
        // empty
    }

    /// @brief  Construction from a given time and key value.

    aiVectorKey(double time, const aiVector3D& value)
    : mTime( time )
    , mValue( value ) {
        // empty
    }

    typedef aiVector3D elem_type;

    // Comparison operators. For use with std::find();
    bool operator == (const aiVectorKey& rhs) const {
        return rhs.mValue == this->mValue;
    }
    bool operator != (const aiVectorKey& rhs ) const {
        return rhs.mValue != this->mValue;
    }

    // Relational operators. For use with std::sort();
    bool operator < (const aiVectorKey& rhs ) const {
        return mTime < rhs.mTime;
    }
    bool operator > (const aiVectorKey& rhs ) const {
        return mTime > rhs.mTime;
    }
#endif // __cplusplus
};

// ---------------------------------------------------------------------------
/** A time-value pair specifying a rotation for the given time.
 *  Rotations are expressed with quaternions. */
struct aiQuatKey
{
    /** The time of this key */
    double mTime;

    /** The value of this key */
    C_STRUCT aiQuaternion mValue;

#ifdef __cplusplus
    aiQuatKey() AI_NO_EXCEPT
    : mTime( 0.0 )
    , mValue() {
        // empty
    }

    /** Construction from a given time and key value */
    aiQuatKey(double time, const aiQuaternion& value)
        :   mTime   (time)
        ,   mValue  (value)
    {}

    typedef aiQuaternion elem_type;

    // Comparison operators. For use with std::find();
    bool operator == (const aiQuatKey& rhs ) const {
        return rhs.mValue == this->mValue;
    }
    bool operator != (const aiQuatKey& rhs ) const {
        return rhs.mValue != this->mValue;
    }

    // Relational operators. For use with std::sort();
    bool operator < (const aiQuatKey& rhs ) const {
        return mTime < rhs.mTime;
    }
    bool operator > (const aiQuatKey& rhs ) const {
        return mTime > rhs.mTime;
    }
#endif
};

// ---------------------------------------------------------------------------
/** Binds a anim-mesh to a specific point in time. */
struct aiMeshKey
{
    /** The time of this key */
    double mTime;

    /** Index into the aiMesh::mAnimMeshes array of the
     *  mesh corresponding to the #aiMeshAnim hosting this
     *  key frame. The referenced anim mesh is evaluated
     *  according to the rules defined in the docs for #aiAnimMesh.*/
    unsigned int mValue;

#ifdef __cplusplus

    aiMeshKey() AI_NO_EXCEPT
    : mTime(0.0)
    , mValue(0)
    {
    }

    /** Construction from a given time and key value */
    aiMeshKey(double time, const unsigned int value)
        :   mTime   (time)
        ,   mValue  (value)
    {}

    typedef unsigned int elem_type;

    // Comparison operators. For use with std::find();
    bool operator == (const aiMeshKey& o) const {
        return o.mValue == this->mValue;
    }
    bool operator != (const aiMeshKey& o) const {
        return o.mValue != this->mValue;
    }

    // Relational operators. For use with std::sort();
    bool operator < (const aiMeshKey& o) const {
        return mTime < o.mTime;
    }
    bool operator > (const aiMeshKey& o) const {
        return mTime > o.mTime;
    }

#endif
};

// ---------------------------------------------------------------------------
/** Binds a morph anim mesh to a specific point in time. */
struct aiMeshMorphKey
{
    /** The time of this key */
    double mTime;

    /** The values and weights at the time of this key */
    unsigned int *mValues;
    double *mWeights;

    /** The number of values and weights */
    unsigned int mNumValuesAndWeights;
#ifdef __cplusplus
	aiMeshMorphKey() AI_NO_EXCEPT
		: mTime(0.0)
		, mValues(nullptr)
		, mWeights(nullptr)
		, mNumValuesAndWeights(0)
	{

	}

    ~aiMeshMorphKey()
    {
        if (mNumValuesAndWeights && mValues && mWeights) {
            delete [] mValues;
            delete [] mWeights;
        }
    }
#endif
};

// ---------------------------------------------------------------------------
/** Defines how an animation channel behaves outside the defined time
 *  range. This corresponds to aiNodeAnim::mPreState and
 *  aiNodeAnim::mPostState.*/
enum aiAnimBehaviour
{
    /** The value from the default node transformation is taken*/
    aiAnimBehaviour_DEFAULT  = 0x0,

    /** The nearest key value is used without interpolation */
    aiAnimBehaviour_CONSTANT = 0x1,

    /** The value of the nearest two keys is linearly
     *  extrapolated for the current time value.*/
    aiAnimBehaviour_LINEAR   = 0x2,

    /** The animation is repeated.
     *
     *  If the animation key go from n to m and the current
     *  time is t, use the value at (t-n) % (|m-n|).*/
    aiAnimBehaviour_REPEAT   = 0x3,

    /** This value is not used, it is just here to force the
     *  the compiler to map this enum to a 32 Bit integer  */
#ifndef SWIG
    _aiAnimBehaviour_Force32Bit = INT_MAX
#endif
};

// ---------------------------------------------------------------------------
/** Describes the animation of a single node. The name specifies the
 *  bone/node which is affected by this animation channel. The keyframes
 *  are given in three separate series of values, one each for position,
 *  rotation and scaling. The transformation matrix computed from these
 *  values replaces the node's original transformation matrix at a
 *  specific time.
 *  This means all keys are absolute and not relative to the bone default pose.
 *  The order in which the transformations are applied is
 *  - as usual - scaling, rotation, translation.
 *
 *  @note All keys are returned in their correct, chronological order.
 *  Duplicate keys don't pass the validation step. Most likely there
 *  will be no negative time values, but they are not forbidden also ( so
 *  implementations need to cope with them! ) */
struct aiNodeAnim {
    /** The name of the node affected by this animation. The node
     *  must exist and it must be unique.*/
    C_STRUCT aiString mNodeName;

    /** The number of position keys */
    unsigned int mNumPositionKeys;

    /** The position keys of this animation channel. Positions are
     * specified as 3D vector. The array is mNumPositionKeys in size.
     *
     * If there are position keys, there will also be at least one
     * scaling and one rotation key.*/
    C_STRUCT aiVectorKey* mPositionKeys;

    /** The number of rotation keys */
    unsigned int mNumRotationKeys;

    /** The rotation keys of this animation channel. Rotations are
     *  given as quaternions,  which are 4D vectors. The array is
     *  mNumRotationKeys in size.
     *
     * If there are rotation keys, there will also be at least one
     * scaling and one position key. */
    C_STRUCT aiQuatKey* mRotationKeys;

    /** The number of scaling keys */
    unsigned int mNumScalingKeys;

    /** The scaling keys of this animation channel. Scalings are
     *  specified as 3D vector. The array is mNumScalingKeys in size.
     *
     * If there are scaling keys, there will also be at least one
     * position and one rotation key.*/
    C_STRUCT aiVectorKey* mScalingKeys;

    /** Defines how the animation behaves before the first
     *  key is encountered.
     *
     *  The default value is aiAnimBehaviour_DEFAULT (the original
     *  transformation matrix of the affected node is used).*/
    C_ENUM aiAnimBehaviour mPreState;

    /** Defines how the animation behaves after the last
     *  key was processed.
     *
     *  The default value is aiAnimBehaviour_DEFAULT (the original
     *  transformation matrix of the affected node is taken).*/
    C_ENUM aiAnimBehaviour mPostState;

#ifdef __cplusplus
    aiNodeAnim() AI_NO_EXCEPT
    : mNumPositionKeys( 0 )
    , mPositionKeys( nullptr )
    , mNumRotationKeys( 0 )
    , mRotationKeys( nullptr )
    , mNumScalingKeys( 0 )
    , mScalingKeys( nullptr )
    , mPreState( aiAnimBehaviour_DEFAULT )
    , mPostState( aiAnimBehaviour_DEFAULT ) {
         // empty
    }

    ~aiNodeAnim() {
        delete [] mPositionKeys;
        delete [] mRotationKeys;
        delete [] mScalingKeys;
    }
#endif // __cplusplus
};

// ---------------------------------------------------------------------------
/** Describes vertex-based animations for a single mesh or a group of
 *  meshes. Meshes carry the animation data for each frame in their
 *  aiMesh::mAnimMeshes array. The purpose of aiMeshAnim is to
 *  define keyframes linking each mesh attachment to a particular
 *  point in time. */
struct aiMeshAnim
{
    /** Name of the mesh to be animated. An empty string is not allowed,
     *  animated meshes need to be named (not necessarily uniquely,
     *  the name can basically serve as wild-card to select a group
     *  of meshes with similar animation setup)*/
    C_STRUCT aiString mName;

    /** Size of the #mKeys array. Must be 1, at least. */
    unsigned int mNumKeys;

    /** Key frames of the animation. May not be NULL. */
    C_STRUCT aiMeshKey* mKeys;

#ifdef __cplusplus

    aiMeshAnim() AI_NO_EXCEPT
        : mNumKeys()
        , mKeys()
    {}

    ~aiMeshAnim()
    {
        delete[] mKeys;
    }

#endif
};

// ---------------------------------------------------------------------------
/** Describes a morphing animation of a given mesh. */
struct aiMeshMorphAnim
{
    /** Name of the mesh to be animated. An empty string is not allowed,
     *  animated meshes need to be named (not necessarily uniquely,
     *  the name can basically serve as wildcard to select a group
     *  of meshes with similar animation setup)*/
    C_STRUCT aiString mName;

    /** Size of the #mKeys array. Must be 1, at least. */
    unsigned int mNumKeys;

    /** Key frames of the animation. May not be NULL. */
    C_STRUCT aiMeshMorphKey* mKeys;

#ifdef __cplusplus

    aiMeshMorphAnim() AI_NO_EXCEPT
        : mNumKeys()
        , mKeys()
    {}

    ~aiMeshMorphAnim()
    {
        delete[] mKeys;
    }

#endif
};

// ---------------------------------------------------------------------------
/** An animation consists of key-frame data for a number of nodes. For
 *  each node affected by the animation a separate series of data is given.*/
struct aiAnimation {
    /** The name of the animation. If the modeling package this data was
     *  exported from does support only a single animation channel, this
     *  name is usually empty (length is zero). */
    C_STRUCT aiString mName;

    /** Duration of the animation in ticks.  */
    double mDuration;

    /** Ticks per second. 0 if not specified in the imported file */
    double mTicksPerSecond;

    /** The number of bone animation channels. Each channel affects
     *  a single node. */
    unsigned int mNumChannels;

    /** The node animation channels. Each channel affects a single node.
     *  The array is mNumChannels in size. */
    C_STRUCT aiNodeAnim** mChannels;


    /** The number of mesh animation channels. Each channel affects
     *  a single mesh and defines vertex-based animation. */
    unsigned int mNumMeshChannels;

    /** The mesh animation channels. Each channel affects a single mesh.
     *  The array is mNumMeshChannels in size. */
    C_STRUCT aiMeshAnim** mMeshChannels;

    /** The number of mesh animation channels. Each channel affects
     *  a single mesh and defines morphing animation. */
    unsigned int mNumMorphMeshChannels;

    /** The morph mesh animation channels. Each channel affects a single mesh.
     *  The array is mNumMorphMeshChannels in size. */
    C_STRUCT aiMeshMorphAnim **mMorphMeshChannels;

#ifdef __cplusplus
    aiAnimation() AI_NO_EXCEPT
    : mDuration(-1.)
    , mTicksPerSecond(0.)
    , mNumChannels(0)
    , mChannels(nullptr)
    , mNumMeshChannels(0)
    , mMeshChannels(nullptr)
    , mNumMorphMeshChannels(0)
    , mMorphMeshChannels(nullptr) {
        // empty
    }

    ~aiAnimation() {
        // DO NOT REMOVE THIS ADDITIONAL CHECK
        if ( mNumChannels && mChannels )  {
            for( unsigned int a = 0; a < mNumChannels; a++) {
                delete mChannels[ a ];
            }

            delete [] mChannels;
        }
        if (mNumMeshChannels && mMeshChannels)  {
            for( unsigned int a = 0; a < mNumMeshChannels; a++) {
                delete mMeshChannels[a];
            }

            delete [] mMeshChannels;
        }
        if (mNumMorphMeshChannels && mMorphMeshChannels) {
                for( unsigned int a = 0; a < mNumMorphMeshChannels; a++) {
                        delete mMorphMeshChannels[a];
                }
            
            delete [] mMorphMeshChannels;
        }
    }
#endif // __cplusplus
};

#ifdef __cplusplus

}

/// @brief  Some C++ utilities for inter- and extrapolation
namespace Assimp {

// ---------------------------------------------------------------------------
/** 
  * @brief CPP-API: Utility class to simplify interpolations of various data types.
  *
  *  The type of interpolation is chosen automatically depending on the
  *  types of the arguments. 
  */
template <typename T>
struct Interpolator
{
    // ------------------------------------------------------------------
    /** @brief Get the result of the interpolation between a,b.
     *
     *  The interpolation algorithm depends on the type of the operands.
     *  aiQuaternion's and aiQuatKey's SLERP, the rest does a simple
     *  linear interpolation. */
    void operator () (T& out,const T& a, const T& b, ai_real d) const {
        out = a + (b-a)*d;
    }
}; // ! Interpolator <T>

//! @cond Never

template <>
struct Interpolator <aiQuaternion>  {
    void operator () (aiQuaternion& out,const aiQuaternion& a,
        const aiQuaternion& b, ai_real d) const
    {
        aiQuaternion::Interpolate(out,a,b,d);
    }
}; // ! Interpolator <aiQuaternion>

template <>
struct Interpolator <unsigned int>  {
    void operator () (unsigned int& out,unsigned int a,
        unsigned int b, ai_real d) const
    {
        out = d>0.5f ? b : a;
    }
}; // ! Interpolator <aiQuaternion>

template <>
struct Interpolator<aiVectorKey>  {
    void operator () (aiVector3D& out,const aiVectorKey& a,
        const aiVectorKey& b, ai_real d) const
    {
        Interpolator<aiVector3D> ipl;
        ipl(out,a.mValue,b.mValue,d);
    }
}; // ! Interpolator <aiVectorKey>

template <>
struct Interpolator<aiQuatKey>  {
    void operator () (aiQuaternion& out, const aiQuatKey& a,
        const aiQuatKey& b, ai_real d) const
    {
        Interpolator<aiQuaternion> ipl;
        ipl(out,a.mValue,b.mValue,d);
    }
}; // ! Interpolator <aiQuatKey>

template <>
struct Interpolator<aiMeshKey>     {
    void operator () (unsigned int& out, const aiMeshKey& a,
        const aiMeshKey& b, ai_real d) const
    {
        Interpolator<unsigned int> ipl;
        ipl(out,a.mValue,b.mValue,d);
    }
}; // ! Interpolator <aiQuatKey>

//! @endcond

} //  ! end namespace Assimp

#endif // __cplusplus

#endif // AI_ANIM_H_INC
