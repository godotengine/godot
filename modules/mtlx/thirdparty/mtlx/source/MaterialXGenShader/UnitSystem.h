//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_UNITSYSTEM_H
#define MATERIALX_UNITSYSTEM_H

/// @file
/// Unit system classes

#include <MaterialXGenShader/Export.h>

#include <MaterialXGenShader/ShaderNode.h>
#include <MaterialXGenShader/ShaderNodeImpl.h>
#include <MaterialXGenShader/TypeDesc.h>
#include <MaterialXCore/Unit.h>

#include <MaterialXCore/Document.h>

MATERIALX_NAMESPACE_BEGIN

class ShaderGenerator;

/// A shared pointer to a UnitSystem
using UnitSystemPtr = shared_ptr<class UnitSystem>;

/// @struct @UnitTransform
/// Structure that represents unit transform information
struct MX_GENSHADER_API UnitTransform
{
    UnitTransform(const string& ss, const string& ts, const TypeDesc* t, const string& unittype);

    string sourceUnit;
    string targetUnit;
    const TypeDesc* type;
    string unitType;

    /// Comparison operator
    bool operator==(const UnitTransform& rhs) const
    {
        return sourceUnit == rhs.sourceUnit &&
               targetUnit == rhs.targetUnit &&
               type == rhs.type &&
               unitType == rhs.unitType;
    }
};

/// @class UnitSystem
/// Base unit system support
class MX_GENSHADER_API UnitSystem
{
  public:
    virtual ~UnitSystem() { }

    /// Create a new UnitSystem
    static UnitSystemPtr create(const string& target);

    /// Return the UnitSystem name
    virtual const string& getName() const
    {
        return UnitSystem::UNITSYTEM_NAME;
    }

    /// Assign unit converter registry replacing any previous assignment
    virtual void setUnitConverterRegistry(UnitConverterRegistryPtr registry);

    /// Returns the currently assigned unit converter registry
    virtual UnitConverterRegistryPtr getUnitConverterRegistry() const;

    /// assign document with unit implementations replacing any previously loaded content.
    virtual void loadLibrary(DocumentPtr document);

    /// Returns whether this unit system supports a provided transform
    bool supportsTransform(const UnitTransform& transform) const;

    /// Create a node to use to perform the given unit space transformation.
    ShaderNodePtr createNode(ShaderGraph* parent, const UnitTransform& transform, const string& name,
                             GenContext& context) const;

    /// Returns a nodedef for a given transform
    virtual NodeDefPtr getNodeDef(const UnitTransform& transform) const;

    static const string UNITSYTEM_NAME;

  protected:
    // Protected constructor
    UnitSystem(const string& target);

  protected:
    UnitConverterRegistryPtr _unitRegistry;
    DocumentPtr _document;
    string _target;
};

MATERIALX_NAMESPACE_END

#endif
