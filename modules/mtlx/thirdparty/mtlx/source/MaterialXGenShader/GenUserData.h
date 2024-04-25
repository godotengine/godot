//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_GENUSERDATA_H
#define MATERIALX_GENUSERDATA_H

/// @file
/// User data base class for shader generation

#include <MaterialXGenShader/Export.h>

MATERIALX_NAMESPACE_BEGIN

class GenUserData;

/// Shared pointer to a GenUserData
using GenUserDataPtr = std::shared_ptr<GenUserData>;

/// Shared pointer to a constant GenUserData
using ConstGenUserDataPtr = std::shared_ptr<const GenUserData>;

/// @class GenUserData
/// Base class for custom user data needed during shader generation.
class MX_GENSHADER_API GenUserData : public std::enable_shared_from_this<GenUserData>
{
  public:
    virtual ~GenUserData() { }

    /// Return a shared pointer for this object.
    GenUserDataPtr getSelf()
    {
        return shared_from_this();
    }

    /// Return a shared pointer for this object.
    ConstGenUserDataPtr getSelf() const
    {
        return shared_from_this();
    }

    /// Return this object cast to a templated type.
    template <class T> shared_ptr<T> asA()
    {
        return std::dynamic_pointer_cast<T>(getSelf());
    }

    /// Return this object cast to a templated type.
    template <class T> shared_ptr<const T> asA() const
    {
        return std::dynamic_pointer_cast<const T>(getSelf());
    }

  protected:
    GenUserData() { }
};

MATERIALX_NAMESPACE_END

#endif // MATERIALX_GENCONTEXT_H
