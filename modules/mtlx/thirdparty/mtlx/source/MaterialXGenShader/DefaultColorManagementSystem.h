//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_DEFAULT_COLOR_MANAGEMENT_SYSTEM_H
#define MATERIALX_DEFAULT_COLOR_MANAGEMENT_SYSTEM_H

/// @file
/// Default color management system implementation

#include <MaterialXGenShader/ColorManagementSystem.h>

MATERIALX_NAMESPACE_BEGIN

/// A shared pointer to a DefaultColorManagementSystem
using DefaultColorManagementSystemPtr = shared_ptr<class DefaultColorManagementSystem>;

/// @class DefaultColorManagementSystem
/// Class for a default color management system.
class MX_GENSHADER_API DefaultColorManagementSystem : public ColorManagementSystem
{
  public:
    virtual ~DefaultColorManagementSystem() { }

    /// Create a new DefaultColorManagementSystem
    static DefaultColorManagementSystemPtr create(const string& target);

    /// Return the DefaultColorManagementSystem name
    const string& getName() const override;

  protected:
    /// Returns a nodedef for a given transform
    NodeDefPtr getNodeDef(const ColorSpaceTransform& transform) const override;

    /// Protected constructor
    DefaultColorManagementSystem(const string& target);

  private:
    string _target;
};

MATERIALX_NAMESPACE_END

#endif
