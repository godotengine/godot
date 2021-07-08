//
// Copyright 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Sampler.cpp : Implements the Sampler class, which represents a GLES 3
// sampler object. Sampler objects store some state needed to sample textures.

#include "libANGLE/Sampler.h"
#include "libANGLE/angletypes.h"
#include "libANGLE/renderer/GLImplFactory.h"
#include "libANGLE/renderer/SamplerImpl.h"

namespace gl
{

Sampler::Sampler(rx::GLImplFactory *factory, SamplerID id)
    : RefCountObject(id), mState(), mDirty(true), mSampler(factory->createSampler(mState)), mLabel()
{}

Sampler::~Sampler()
{
    SafeDelete(mSampler);
}

void Sampler::onDestroy(const Context *context)
{
    if (mSampler)
    {
        mSampler->onDestroy(context);
    }
}

void Sampler::setLabel(const Context *context, const std::string &label)
{
    mLabel = label;
}

const std::string &Sampler::getLabel() const
{
    return mLabel;
}

void Sampler::setMinFilter(const Context *context, GLenum minFilter)
{
    mState.setMinFilter(minFilter);
    signalDirtyState();
}

GLenum Sampler::getMinFilter() const
{
    return mState.getMinFilter();
}

void Sampler::setMagFilter(const Context *context, GLenum magFilter)
{
    mState.setMagFilter(magFilter);
    signalDirtyState();
}

GLenum Sampler::getMagFilter() const
{
    return mState.getMagFilter();
}

void Sampler::setWrapS(const Context *context, GLenum wrapS)
{
    mState.setWrapS(wrapS);
    signalDirtyState();
}

GLenum Sampler::getWrapS() const
{
    return mState.getWrapS();
}

void Sampler::setWrapT(const Context *context, GLenum wrapT)
{
    mState.setWrapT(wrapT);
    signalDirtyState();
}

GLenum Sampler::getWrapT() const
{
    return mState.getWrapT();
}

void Sampler::setWrapR(const Context *context, GLenum wrapR)
{
    mState.setWrapR(wrapR);
    signalDirtyState();
}

GLenum Sampler::getWrapR() const
{
    return mState.getWrapR();
}

void Sampler::setMaxAnisotropy(const Context *context, float maxAnisotropy)
{
    mState.setMaxAnisotropy(maxAnisotropy);
    signalDirtyState();
}

float Sampler::getMaxAnisotropy() const
{
    return mState.getMaxAnisotropy();
}

void Sampler::setMinLod(const Context *context, GLfloat minLod)
{
    mState.setMinLod(minLod);
    signalDirtyState();
}

GLfloat Sampler::getMinLod() const
{
    return mState.getMinLod();
}

void Sampler::setMaxLod(const Context *context, GLfloat maxLod)
{
    mState.setMaxLod(maxLod);
    signalDirtyState();
}

GLfloat Sampler::getMaxLod() const
{
    return mState.getMaxLod();
}

void Sampler::setCompareMode(const Context *context, GLenum compareMode)
{
    mState.setCompareMode(compareMode);
    signalDirtyState();
}

GLenum Sampler::getCompareMode() const
{
    return mState.getCompareMode();
}

void Sampler::setCompareFunc(const Context *context, GLenum compareFunc)
{
    mState.setCompareFunc(compareFunc);
    signalDirtyState();
}

GLenum Sampler::getCompareFunc() const
{
    return mState.getCompareFunc();
}

void Sampler::setSRGBDecode(const Context *context, GLenum sRGBDecode)
{
    mState.setSRGBDecode(sRGBDecode);
    signalDirtyState();
}

GLenum Sampler::getSRGBDecode() const
{
    return mState.getSRGBDecode();
}

void Sampler::setBorderColor(const Context *context, const ColorGeneric &color)
{
    mState.setBorderColor(color);
    signalDirtyState();
}

const ColorGeneric &Sampler::getBorderColor() const
{
    return mState.getBorderColor();
}

const SamplerState &Sampler::getSamplerState() const
{
    return mState;
}

rx::SamplerImpl *Sampler::getImplementation() const
{
    return mSampler;
}

angle::Result Sampler::syncState(const Context *context)
{
    ASSERT(isDirty());
    angle::Result result = mSampler->syncState(context, mDirty);
    mDirty               = result != angle::Result::Continue;
    return result;
}

void Sampler::signalDirtyState()
{
    mDirty = true;
    onStateChange(angle::SubjectMessage::DirtyBitsFlagged);
}

}  // namespace gl
