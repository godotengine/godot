//
// Copyright 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Sampler.h : Defines the Sampler class, which represents a GLES 3
// sampler object. Sampler objects store some state needed to sample textures.

#ifndef LIBANGLE_SAMPLER_H_
#define LIBANGLE_SAMPLER_H_

#include "libANGLE/Debug.h"
#include "libANGLE/Observer.h"
#include "libANGLE/RefCountObject.h"
#include "libANGLE/angletypes.h"

namespace rx
{
class GLImplFactory;
class SamplerImpl;
}  // namespace rx

namespace gl
{

class Sampler final : public RefCountObject<SamplerID>, public LabeledObject, public angle::Subject
{
  public:
    Sampler(rx::GLImplFactory *factory, SamplerID id);
    ~Sampler() override;

    void onDestroy(const Context *context) override;

    void setLabel(const Context *context, const std::string &label) override;
    const std::string &getLabel() const override;

    void setMinFilter(const Context *context, GLenum minFilter);
    GLenum getMinFilter() const;

    void setMagFilter(const Context *context, GLenum magFilter);
    GLenum getMagFilter() const;

    void setWrapS(const Context *context, GLenum wrapS);
    GLenum getWrapS() const;

    void setWrapT(const Context *context, GLenum wrapT);
    GLenum getWrapT() const;

    void setWrapR(const Context *context, GLenum wrapR);
    GLenum getWrapR() const;

    void setMaxAnisotropy(const Context *context, float maxAnisotropy);
    float getMaxAnisotropy() const;

    void setMinLod(const Context *context, GLfloat minLod);
    GLfloat getMinLod() const;

    void setMaxLod(const Context *context, GLfloat maxLod);
    GLfloat getMaxLod() const;

    void setCompareMode(const Context *context, GLenum compareMode);
    GLenum getCompareMode() const;

    void setCompareFunc(const Context *context, GLenum compareFunc);
    GLenum getCompareFunc() const;

    void setSRGBDecode(const Context *context, GLenum sRGBDecode);
    GLenum getSRGBDecode() const;

    void setBorderColor(const Context *context, const ColorGeneric &color);
    const ColorGeneric &getBorderColor() const;

    const SamplerState &getSamplerState() const;

    rx::SamplerImpl *getImplementation() const;

    angle::Result syncState(const Context *context);
    bool isDirty() const { return mDirty; }

  private:
    void signalDirtyState();
    SamplerState mState;
    bool mDirty;
    rx::SamplerImpl *mSampler;

    std::string mLabel;
};

}  // namespace gl

#endif  // LIBANGLE_SAMPLER_H_
