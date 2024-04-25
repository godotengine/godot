//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_COLOR_MANAGEMENT_SYSTEM_H
#define MATERIALX_COLOR_MANAGEMENT_SYSTEM_H

/// @file
/// Color management system classes

#include <MaterialXGenShader/Export.h>

#include <MaterialXGenShader/ShaderNode.h>
#include <MaterialXGenShader/ShaderNodeImpl.h>
#include <MaterialXGenShader/TypeDesc.h>

#include <MaterialXCore/Document.h>

MATERIALX_NAMESPACE_BEGIN

class ShaderGenerator;

/// A shared pointer to a ColorManagementSystem
using ColorManagementSystemPtr = shared_ptr<class ColorManagementSystem>;

/// @struct ColorSpaceTransform
/// Structure that represents color space transform information
struct MX_GENSHADER_API ColorSpaceTransform
{
    ColorSpaceTransform(const string& ss, const string& ts, const TypeDesc* t);

    string sourceSpace;
    string targetSpace;
    const TypeDesc* type;

    /// Comparison operator
    bool operator==(const ColorSpaceTransform& other) const
    {
        return sourceSpace == other.sourceSpace &&
               targetSpace == other.targetSpace &&
               type == other.type;
    }
};

/// @class ColorManagementSystem
/// Abstract base class for color management systems
class MX_GENSHADER_API ColorManagementSystem
{
  public:
    virtual ~ColorManagementSystem() { }

    /// Return the ColorManagementSystem name
    virtual const string& getName() const = 0;

    /// Load a library of implementations from the provided document,
    /// replacing any previously loaded content.
    virtual void loadLibrary(DocumentPtr document);

    /// Returns whether this color management system supports a provided transform
    bool supportsTransform(const ColorSpaceTransform& transform) const;

    /// Create a node to use to perform the given color space transformation.
    ShaderNodePtr createNode(const ShaderGraph* parent, const ColorSpaceTransform& transform, const string& name,
                             GenContext& context) const;

  protected:
    /// Protected constructor
    ColorManagementSystem();

    /// Returns a nodedef for a given transform
    virtual NodeDefPtr getNodeDef(const ColorSpaceTransform& transform) const = 0;

  protected:
    DocumentPtr _document;
};

MATERIALX_NAMESPACE_END

#endif
