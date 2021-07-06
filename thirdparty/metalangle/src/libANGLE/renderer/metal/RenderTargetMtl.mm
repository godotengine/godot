//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// RenderTargetMtl.mm:
//    Implements the class methods for RenderTargetMtl.
//

#include "libANGLE/renderer/metal/RenderTargetMtl.h"

#include "libANGLE/renderer/metal/mtl_state_cache.h"

namespace rx
{
RenderTargetMtl::RenderTargetMtl()
{}

RenderTargetMtl::~RenderTargetMtl()
{
    reset();
}

RenderTargetMtl::RenderTargetMtl(RenderTargetMtl &&other)
    : mTextureRenderTargetInfo(std::move(other.mTextureRenderTargetInfo))
{}

void RenderTargetMtl::set(const mtl::TextureRef &texture,
                          uint32_t level,
                          uint32_t layer,
                          const mtl::Format &format)
{
    set(texture, nullptr, level, layer, format);
}

void RenderTargetMtl::set(const mtl::TextureRef &texture,
                          const mtl::TextureRef &implicitMSTexture,
                          uint32_t level,
                          uint32_t layer,
                          const mtl::Format &format)
{
    mTextureRenderTargetInfo.targetTexture           = texture;
    mTextureRenderTargetInfo.targetImplicitMSTexture = implicitMSTexture;
    mTextureRenderTargetInfo.targetLevel             = level;
    mTextureRenderTargetInfo.targetSliceOrDepth      = layer;
    mFormat                                     = &format;
}

void RenderTargetMtl::setTexture(const mtl::TextureRef &texture)
{
    mTextureRenderTargetInfo.targetTexture = texture;
}

void RenderTargetMtl::setImplicitMSTexture(const mtl::TextureRef &implicitMSTexture)
{
    mTextureRenderTargetInfo.targetImplicitMSTexture = implicitMSTexture;
}

void RenderTargetMtl::reset()
{
    mTextureRenderTargetInfo.reset();
    mFormat = nullptr;
}

void RenderTargetMtl::toRenderPassAttachmentDesc(mtl::RenderPassAttachmentDesc *rpaDescOut) const
{
    *static_cast<mtl::RenderPassAttachmentTextureTargetDesc *>(rpaDescOut) =
        mTextureRenderTargetInfo;
}
}
