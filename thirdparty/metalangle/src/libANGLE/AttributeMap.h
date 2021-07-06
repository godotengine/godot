//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef LIBANGLE_ATTRIBUTEMAP_H_
#define LIBANGLE_ATTRIBUTEMAP_H_

#include "common/PackedEnums.h"

#include <EGL/egl.h>

#include <map>
#include <vector>

namespace egl
{

class AttributeMap final
{
  public:
    AttributeMap();
    AttributeMap(const AttributeMap &other);
    ~AttributeMap();

    void insert(EGLAttrib key, EGLAttrib value);
    bool contains(EGLAttrib key) const;

    EGLAttrib get(EGLAttrib key) const;
    EGLAttrib get(EGLAttrib key, EGLAttrib defaultValue) const;
    EGLint getAsInt(EGLAttrib key) const;
    EGLint getAsInt(EGLAttrib key, EGLint defaultValue) const;

    template <typename PackedEnumT>
    PackedEnumT getAsPackedEnum(EGLAttrib key) const
    {
        return FromEGLenum<PackedEnumT>(static_cast<EGLenum>(get(key)));
    }

    template <typename PackedEnumT>
    PackedEnumT getAsPackedEnum(EGLAttrib key, PackedEnumT defaultValue) const
    {
        auto iter = mAttributes.find(key);
        return (mAttributes.find(key) != mAttributes.end())
                   ? FromEGLenum<PackedEnumT>(static_cast<EGLenum>(iter->second))
                   : defaultValue;
    }

    bool isEmpty() const;
    std::vector<EGLint> toIntVector() const;

    typedef std::map<EGLAttrib, EGLAttrib>::const_iterator const_iterator;

    const_iterator begin() const;
    const_iterator end() const;

    static AttributeMap CreateFromIntArray(const EGLint *attributes);
    static AttributeMap CreateFromAttribArray(const EGLAttrib *attributes);

  private:
    std::map<EGLAttrib, EGLAttrib> mAttributes;
};
}  // namespace egl

#endif  // LIBANGLE_ATTRIBUTEMAP_H_
