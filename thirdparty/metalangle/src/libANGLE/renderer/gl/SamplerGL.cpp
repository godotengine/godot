//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// SamplerGL.cpp: Defines the rx::SamplerGL class, an implementation of SamplerImpl.

#include "libANGLE/renderer/gl/SamplerGL.h"

#include "libANGLE/renderer/gl/FunctionsGL.h"
#include "libANGLE/renderer/gl/StateManagerGL.h"

namespace
{

template <typename T>
inline void SetSamplerParameter(const rx::FunctionsGL *functions,
                                GLuint sampler,
                                GLenum name,
                                const T &value)
{
    functions->samplerParameterf(sampler, name, static_cast<GLfloat>(value));
}

inline void SetSamplerParameter(const rx::FunctionsGL *functions,
                                GLuint sampler,
                                GLenum name,
                                const angle::ColorGeneric &value)
{
    switch (value.type)
    {
        case angle::ColorGeneric::Type::Float:
            functions->samplerParameterfv(sampler, name, &value.colorF.red);
            break;
        case angle::ColorGeneric::Type::Int:
            functions->samplerParameterIiv(sampler, name, &value.colorI.red);
            break;
        case angle::ColorGeneric::Type::UInt:
            functions->samplerParameterIuiv(sampler, name, &value.colorUI.red);
            break;
        default:
            UNREACHABLE();
            break;
    }
}

template <typename Getter, typename Setter>
static inline void SyncSamplerStateMember(const rx::FunctionsGL *functions,
                                          GLuint sampler,
                                          const gl::SamplerState &newState,
                                          gl::SamplerState &curState,
                                          GLenum name,
                                          Getter getter,
                                          Setter setter)
{
    if ((curState.*getter)() != (newState.*getter)())
    {
        (curState.*setter)((newState.*getter)());
        SetSamplerParameter(functions, sampler, name, (newState.*getter)());
    }
}
}  // namespace

namespace rx
{

SamplerGL::SamplerGL(const gl::SamplerState &state,
                     const FunctionsGL *functions,
                     StateManagerGL *stateManager)
    : SamplerImpl(state),
      mFunctions(functions),
      mStateManager(stateManager),
      mAppliedSamplerState(),
      mSamplerID(0)
{
    mFunctions->genSamplers(1, &mSamplerID);
}

SamplerGL::~SamplerGL()
{
    mStateManager->deleteSampler(mSamplerID);
    mSamplerID = 0;
}

angle::Result SamplerGL::syncState(const gl::Context *context, const bool dirty)
{
    if (!dirty)
    {
        return angle::Result::Continue;
    }
    // clang-format off
    SyncSamplerStateMember(mFunctions, mSamplerID, mState, mAppliedSamplerState, GL_TEXTURE_MIN_FILTER, &gl::SamplerState::getMinFilter, &gl::SamplerState::setMinFilter);
    SyncSamplerStateMember(mFunctions, mSamplerID, mState, mAppliedSamplerState, GL_TEXTURE_MAG_FILTER, &gl::SamplerState::getMagFilter, &gl::SamplerState::setMagFilter);
    SyncSamplerStateMember(mFunctions, mSamplerID, mState, mAppliedSamplerState, GL_TEXTURE_WRAP_S, &gl::SamplerState::getWrapS, &gl::SamplerState::setWrapS);
    SyncSamplerStateMember(mFunctions, mSamplerID, mState, mAppliedSamplerState, GL_TEXTURE_WRAP_T, &gl::SamplerState::getWrapT, &gl::SamplerState::setWrapT);
    SyncSamplerStateMember(mFunctions, mSamplerID, mState, mAppliedSamplerState, GL_TEXTURE_WRAP_R, &gl::SamplerState::getWrapR, &gl::SamplerState::setWrapR);
    SyncSamplerStateMember(mFunctions, mSamplerID, mState, mAppliedSamplerState, GL_TEXTURE_MAX_ANISOTROPY_EXT, &gl::SamplerState::getMaxAnisotropy, &gl::SamplerState::setMaxAnisotropy);
    SyncSamplerStateMember(mFunctions, mSamplerID, mState, mAppliedSamplerState, GL_TEXTURE_MIN_LOD, &gl::SamplerState::getMinLod, &gl::SamplerState::setMinLod);
    SyncSamplerStateMember(mFunctions, mSamplerID, mState, mAppliedSamplerState, GL_TEXTURE_MAX_LOD, &gl::SamplerState::getMaxLod, &gl::SamplerState::setMaxLod);
    SyncSamplerStateMember(mFunctions, mSamplerID, mState, mAppliedSamplerState, GL_TEXTURE_COMPARE_MODE, &gl::SamplerState::getCompareMode, &gl::SamplerState::setCompareMode);
    SyncSamplerStateMember(mFunctions, mSamplerID, mState, mAppliedSamplerState, GL_TEXTURE_COMPARE_FUNC, &gl::SamplerState::getCompareFunc, &gl::SamplerState::setCompareFunc);
    SyncSamplerStateMember(mFunctions, mSamplerID, mState, mAppliedSamplerState, GL_TEXTURE_SRGB_DECODE_EXT, &gl::SamplerState::getSRGBDecode, &gl::SamplerState::setSRGBDecode);
    SyncSamplerStateMember(mFunctions, mSamplerID, mState, mAppliedSamplerState, GL_TEXTURE_BORDER_COLOR, &gl::SamplerState::getBorderColor, &gl::SamplerState::setBorderColor);
    // clang-format on
    return angle::Result::Continue;
}

GLuint SamplerGL::getSamplerID() const
{
    return mSamplerID;
}
}  // namespace rx
