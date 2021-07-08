//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// ImageIndex.cpp: Implementation for ImageIndex methods.

#include "libANGLE/ImageIndex.h"

#include "common/utilities.h"
#include "libANGLE/Constants.h"
#include "libANGLE/angletypes.h"

#include <tuple>

namespace gl
{
namespace
{
GLint TextureTargetToLayer(TextureTarget target)
{
    switch (target)
    {
        case TextureTarget::CubeMapPositiveX:
            return 0;
        case TextureTarget::CubeMapNegativeX:
            return 1;
        case TextureTarget::CubeMapPositiveY:
            return 2;
        case TextureTarget::CubeMapNegativeY:
            return 3;
        case TextureTarget::CubeMapPositiveZ:
            return 4;
        case TextureTarget::CubeMapNegativeZ:
            return 5;
        case TextureTarget::External:
            return ImageIndex::kEntireLevel;
        case TextureTarget::Rectangle:
            return ImageIndex::kEntireLevel;
        case TextureTarget::_2D:
            return ImageIndex::kEntireLevel;
        case TextureTarget::_2DArray:
            return ImageIndex::kEntireLevel;
        case TextureTarget::_2DMultisample:
            return ImageIndex::kEntireLevel;
        case TextureTarget::_2DMultisampleArray:
            return ImageIndex::kEntireLevel;
        case TextureTarget::_3D:
            return ImageIndex::kEntireLevel;
        default:
            UNREACHABLE();
            return 0;
    }
}

bool IsArrayTarget(TextureTarget target)
{
    switch (target)
    {
        case TextureTarget::_2DArray:
        case TextureTarget::_2DMultisampleArray:
            return true;
        default:
            return false;
    }
}
}  // anonymous namespace

TextureTarget TextureTypeToTarget(TextureType type, GLint layerIndex)
{
    if (type == TextureType::CubeMap)
    {
        // As GL_TEXTURE_CUBE_MAP cannot be a texture target in texImage*D APIs, so we don't allow
        // an entire cube map to have a texture target.
        ASSERT(layerIndex != ImageIndex::kEntireLevel);
        return CubeFaceIndexToTextureTarget(layerIndex);
    }
    else
    {
        return NonCubeTextureTypeToTarget(type);
    }
}

ImageIndex::ImageIndex()
    : mType(TextureType::InvalidEnum), mLevelIndex(0), mLayerIndex(0), mLayerCount(kEntireLevel)
{}

ImageIndex::ImageIndex(const ImageIndex &other) = default;

ImageIndex &ImageIndex::operator=(const ImageIndex &other) = default;

bool ImageIndex::hasLayer() const
{
    return mLayerIndex != kEntireLevel;
}

bool ImageIndex::isLayered() const
{
    switch (mType)
    {
        case TextureType::_2DArray:
        case TextureType::_2DMultisampleArray:
        case TextureType::CubeMap:
        case TextureType::_3D:
            return mLayerIndex == kEntireLevel;
        default:
            return false;
    }
}

bool ImageIndex::has3DLayer() const
{
    // It's quicker to check != CubeMap than calling usesTex3D, which checks multiple types. This
    // ASSERT validates the check gives the same result.
    ASSERT(!hasLayer() || ((mType != TextureType::CubeMap) == usesTex3D()));
    return (hasLayer() && mType != TextureType::CubeMap);
}

bool ImageIndex::usesTex3D() const
{
    return mType == TextureType::_3D || mType == TextureType::_2DArray ||
           mType == TextureType::_2DMultisampleArray;
}

TextureTarget ImageIndex::getTarget() const
{
    return TextureTypeToTarget(mType, mLayerIndex);
}

gl::TextureTarget ImageIndex::getTargetOrFirstCubeFace() const
{
    if (isEntireLevelCubeMap())
    {
        return gl::kCubeMapTextureTargetMin;
    }
    else
    {
        return getTarget();
    }
}

GLint ImageIndex::cubeMapFaceIndex() const
{
    ASSERT(mType == TextureType::CubeMap);
    ASSERT(mLayerIndex == kEntireLevel || mLayerIndex < static_cast<GLint>(kCubeFaceCount));
    return mLayerIndex;
}

bool ImageIndex::valid() const
{
    return mType != TextureType::InvalidEnum;
}

bool ImageIndex::isEntireLevelCubeMap() const
{
    return mType == TextureType::CubeMap && mLayerIndex == ImageIndex::kEntireLevel;
}

ImageIndex ImageIndex::Make2D(GLint levelIndex)
{
    return ImageIndex(TextureType::_2D, levelIndex, kEntireLevel, 1);
}

ImageIndex ImageIndex::MakeRectangle(GLint levelIndex)
{
    return ImageIndex(TextureType::Rectangle, levelIndex, kEntireLevel, 1);
}

ImageIndex ImageIndex::MakeCubeMapFace(TextureTarget target, GLint levelIndex)
{
    ASSERT(IsCubeMapFaceTarget(target));
    return ImageIndex(TextureType::CubeMap, levelIndex, TextureTargetToLayer(target), 1);
}

ImageIndex ImageIndex::Make2DArray(GLint levelIndex, GLint layerIndex)
{
    return ImageIndex(TextureType::_2DArray, levelIndex, layerIndex, 1);
}

ImageIndex ImageIndex::Make2DArrayRange(GLint levelIndex, GLint layerIndex, GLint numLayers)
{
    return ImageIndex(TextureType::_2DArray, levelIndex, layerIndex, numLayers);
}

ImageIndex ImageIndex::Make3D(GLint levelIndex, GLint layerIndex)
{
    return ImageIndex(TextureType::_3D, levelIndex, layerIndex, 1);
}

ImageIndex ImageIndex::MakeFromTarget(TextureTarget target, GLint levelIndex, GLint depth)
{
    return ImageIndex(TextureTargetToType(target), levelIndex, TextureTargetToLayer(target),
                      IsArrayTarget(target) ? depth : 1);
}

ImageIndex ImageIndex::MakeFromType(TextureType type,
                                    GLint levelIndex,
                                    GLint layerIndex,
                                    GLint layerCount)
{
    GLint overrideLayerCount =
        (type == TextureType::CubeMap && layerIndex == kEntireLevel ? kCubeFaceCount : layerCount);
    return ImageIndex(type, levelIndex, layerIndex, overrideLayerCount);
}

ImageIndex ImageIndex::Make2DMultisample()
{
    return ImageIndex(TextureType::_2DMultisample, 0, kEntireLevel, 1);
}

ImageIndex ImageIndex::Make2DMultisampleArray(GLint layerIndex)
{
    return ImageIndex(TextureType::_2DMultisampleArray, 0, layerIndex, 1);
}

ImageIndex ImageIndex::Make2DMultisampleArrayRange(GLint layerIndex, GLint numLayers)
{
    return ImageIndex(TextureType::_2DMultisampleArray, 0, layerIndex, numLayers);
}

bool ImageIndex::operator<(const ImageIndex &b) const
{
    return std::tie(mType, mLevelIndex, mLayerIndex, mLayerCount) <
           std::tie(b.mType, b.mLevelIndex, b.mLayerIndex, b.mLayerCount);
}

bool ImageIndex::operator==(const ImageIndex &b) const
{
    return std::tie(mType, mLevelIndex, mLayerIndex, mLayerCount) ==
           std::tie(b.mType, b.mLevelIndex, b.mLayerIndex, b.mLayerCount);
}

bool ImageIndex::operator!=(const ImageIndex &b) const
{
    return !(*this == b);
}

ImageIndex::ImageIndex(TextureType type, GLint levelIndex, GLint layerIndex, GLint layerCount)
    : mType(type), mLevelIndex(levelIndex), mLayerIndex(layerIndex), mLayerCount(layerCount)
{}

ImageIndexIterator ImageIndex::getLayerIterator(GLint layerCount) const
{
    ASSERT(mType != TextureType::_2D && !hasLayer());
    return ImageIndexIterator::MakeGeneric(mType, mLevelIndex, mLevelIndex + 1, 0, layerCount);
}

ImageIndexIterator::ImageIndexIterator(const ImageIndexIterator &other) = default;

ImageIndexIterator ImageIndexIterator::Make2D(GLint minMip, GLint maxMip)
{
    return ImageIndexIterator(TextureType::_2D, Range<GLint>(minMip, maxMip),
                              Range<GLint>(ImageIndex::kEntireLevel, ImageIndex::kEntireLevel),
                              nullptr);
}

ImageIndexIterator ImageIndexIterator::MakeRectangle(GLint minMip, GLint maxMip)
{
    return ImageIndexIterator(TextureType::Rectangle, Range<GLint>(minMip, maxMip),
                              Range<GLint>(ImageIndex::kEntireLevel, ImageIndex::kEntireLevel),
                              nullptr);
}

ImageIndexIterator ImageIndexIterator::MakeCube(GLint minMip, GLint maxMip)
{
    return ImageIndexIterator(TextureType::CubeMap, Range<GLint>(minMip, maxMip),
                              Range<GLint>(0, 6), nullptr);
}

ImageIndexIterator ImageIndexIterator::Make3D(GLint minMip,
                                              GLint maxMip,
                                              GLint minLayer,
                                              GLint maxLayer)
{
    return ImageIndexIterator(TextureType::_3D, Range<GLint>(minMip, maxMip),
                              Range<GLint>(minLayer, maxLayer), nullptr);
}

ImageIndexIterator ImageIndexIterator::Make2DArray(GLint minMip,
                                                   GLint maxMip,
                                                   const GLsizei *layerCounts)
{
    return ImageIndexIterator(TextureType::_2DArray, Range<GLint>(minMip, maxMip),
                              Range<GLint>(0, IMPLEMENTATION_MAX_2D_ARRAY_TEXTURE_LAYERS),
                              layerCounts);
}

ImageIndexIterator ImageIndexIterator::Make2DMultisample()
{
    return ImageIndexIterator(TextureType::_2DMultisample, Range<GLint>(0, 1),
                              Range<GLint>(ImageIndex::kEntireLevel, ImageIndex::kEntireLevel),
                              nullptr);
}

ImageIndexIterator ImageIndexIterator::Make2DMultisampleArray(const GLsizei *layerCounts)
{
    return ImageIndexIterator(TextureType::_2DMultisampleArray, Range<GLint>(0, 1),
                              Range<GLint>(0, IMPLEMENTATION_MAX_2D_ARRAY_TEXTURE_LAYERS),
                              layerCounts);
}

ImageIndexIterator ImageIndexIterator::MakeGeneric(TextureType type,
                                                   GLint minMip,
                                                   GLint maxMip,
                                                   GLint minLayer,
                                                   GLint maxLayer)
{
    if (type == TextureType::CubeMap)
    {
        return MakeCube(minMip, maxMip);
    }

    return ImageIndexIterator(type, Range<GLint>(minMip, maxMip), Range<GLint>(minLayer, maxLayer),
                              nullptr);
}

ImageIndexIterator::ImageIndexIterator(TextureType type,
                                       const Range<GLint> &mipRange,
                                       const Range<GLint> &layerRange,
                                       const GLsizei *layerCounts)
    : mMipRange(mipRange),
      mLayerRange(layerRange),
      mLayerCounts(layerCounts),
      mCurrentIndex(type, mipRange.low(), layerRange.low(), 1)
{}

GLint ImageIndexIterator::maxLayer() const
{
    if (mLayerCounts)
    {
        ASSERT(mCurrentIndex.hasLayer());
        return (mCurrentIndex.getLevelIndex() < mMipRange.high())
                   ? mLayerCounts[mCurrentIndex.getLevelIndex()]
                   : 0;
    }
    return mLayerRange.high();
}

ImageIndex ImageIndexIterator::next()
{
    ASSERT(hasNext());

    // Make a copy of the current index to return
    ImageIndex previousIndex = mCurrentIndex;

    // Iterate layers in the inner loop for now. We can add switchable
    // layer or mip iteration if we need it.

    if (mCurrentIndex.hasLayer() && mCurrentIndex.getLayerIndex() < maxLayer() - 1)
    {
        mCurrentIndex.mLayerIndex++;
    }
    else if (mCurrentIndex.mLevelIndex < mMipRange.high() - 1)
    {
        mCurrentIndex.mLayerIndex = mLayerRange.low();
        mCurrentIndex.mLevelIndex++;
    }
    else
    {
        mCurrentIndex = ImageIndex();
    }

    return previousIndex;
}

ImageIndex ImageIndexIterator::current() const
{
    return mCurrentIndex;
}

bool ImageIndexIterator::hasNext() const
{
    return mCurrentIndex.valid();
}

}  // namespace gl
