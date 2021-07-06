//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// ImageIndex.h: A helper struct for indexing into an Image array

#ifndef LIBANGLE_IMAGE_INDEX_H_
#define LIBANGLE_IMAGE_INDEX_H_

#include "common/PackedEnums.h"
#include "common/mathutil.h"

#include "angle_gl.h"

namespace gl
{

class ImageIndexIterator;

class ImageIndex
{
  public:
    ImageIndex();
    ImageIndex(const ImageIndex &other);
    ImageIndex &operator=(const ImageIndex &other);

    TextureType getType() const { return mType; }
    GLint getLevelIndex() const { return mLevelIndex; }
    GLint getLayerIndex() const { return mLayerIndex; }
    GLint getLayerCount() const { return mLayerCount; }

    bool hasLayer() const;
    bool has3DLayer() const;
    bool usesTex3D() const;
    GLint cubeMapFaceIndex() const;
    bool valid() const;
    // Note that you cannot use this function when the ImageIndex represents an entire level of cube
    // map.
    TextureTarget getTarget() const;

    TextureTarget getTargetOrFirstCubeFace() const;

    bool isLayered() const;
    bool isEntireLevelCubeMap() const;

    static ImageIndex Make2D(GLint levelIndex);
    static ImageIndex MakeRectangle(GLint levelIndex);
    static ImageIndex MakeCubeMapFace(TextureTarget target, GLint levelIndex);
    static ImageIndex Make2DArray(GLint levelIndex, GLint layerIndex = kEntireLevel);
    static ImageIndex Make2DArrayRange(GLint levelIndex, GLint layerIndex, GLint layerCount);
    static ImageIndex Make3D(GLint levelIndex, GLint layerIndex = kEntireLevel);
    static ImageIndex MakeFromTarget(TextureTarget target, GLint levelIndex, GLint depth);
    static ImageIndex MakeFromType(TextureType type,
                                   GLint levelIndex,
                                   GLint layerIndex = kEntireLevel,
                                   GLint layerCount = 1);
    static ImageIndex Make2DMultisample();
    static ImageIndex Make2DMultisampleArray(GLint layerIndex = kEntireLevel);
    static ImageIndex Make2DMultisampleArrayRange(GLint layerIndex, GLint layerCount);

    static constexpr GLint kEntireLevel = static_cast<GLint>(-1);

    bool operator<(const ImageIndex &b) const;
    bool operator==(const ImageIndex &b) const;
    bool operator!=(const ImageIndex &b) const;

    // Only valid for 3D/Cube textures with layers.
    ImageIndexIterator getLayerIterator(GLint layerCount) const;

  private:
    friend class ImageIndexIterator;

    ImageIndex(TextureType type, GLint leveIndex, GLint layerIndex, GLint layerCount);

    TextureType mType;
    GLint mLevelIndex;
    GLint mLayerIndex;
    GLint mLayerCount;
};

// To be used like this:
//
// ImageIndexIterator it = ...;
// while (it.hasNext())
// {
//     ImageIndex current = it.next();
// }
class ImageIndexIterator
{
  public:
    ImageIndexIterator(const ImageIndexIterator &other);

    static ImageIndexIterator Make2D(GLint minMip, GLint maxMip);
    static ImageIndexIterator MakeRectangle(GLint minMip, GLint maxMip);
    static ImageIndexIterator MakeCube(GLint minMip, GLint maxMip);
    static ImageIndexIterator Make3D(GLint minMip, GLint maxMip, GLint minLayer, GLint maxLayer);
    static ImageIndexIterator Make2DArray(GLint minMip, GLint maxMip, const GLsizei *layerCounts);
    static ImageIndexIterator Make2DMultisample();
    static ImageIndexIterator Make2DMultisampleArray(const GLsizei *layerCounts);
    static ImageIndexIterator MakeGeneric(TextureType type,
                                          GLint minMip,
                                          GLint maxMip,
                                          GLint minLayer,
                                          GLint maxLayer);

    ImageIndex next();
    ImageIndex current() const;
    bool hasNext() const;

  private:
    ImageIndexIterator(TextureType type,
                       const Range<GLint> &mipRange,
                       const Range<GLint> &layerRange,
                       const GLsizei *layerCounts);

    GLint maxLayer() const;

    const Range<GLint> mMipRange;
    const Range<GLint> mLayerRange;
    const GLsizei *const mLayerCounts;

    ImageIndex mCurrentIndex;
};

TextureTarget TextureTypeToTarget(TextureType type, GLint layerIndex);

}  // namespace gl

#endif  // LIBANGLE_IMAGE_INDEX_H_
