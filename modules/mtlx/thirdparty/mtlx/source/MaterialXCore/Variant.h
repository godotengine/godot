//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_VARIANT_H
#define MATERIALX_VARIANT_H

/// @file
/// Variant element subclasses

#include <MaterialXCore/Export.h>

#include <MaterialXCore/Interface.h>

MATERIALX_NAMESPACE_BEGIN

class Variant;
class VariantSet;
class VariantAssign;

/// A shared pointer to a Variant
using VariantPtr = shared_ptr<Variant>;
/// A shared pointer to a const Variant
using ConstVariantPtr = shared_ptr<const Variant>;

/// A shared pointer to a VariantSet
using VariantSetPtr = shared_ptr<VariantSet>;
/// A shared pointer to a const VariantSet
using ConstVariantSetPtr = shared_ptr<const VariantSet>;

/// A shared pointer to a VariantAssign
using VariantAssignPtr = shared_ptr<VariantAssign>;
/// A shared pointer to a const VariantAssign
using ConstVariantAssignPtr = shared_ptr<const VariantAssign>;

/// @class Variant
/// A variant element within a VariantSet
class MX_CORE_API Variant : public InterfaceElement
{
  public:
    Variant(ElementPtr parent, const string& name) :
        InterfaceElement(parent, CATEGORY, name)
    {
    }
    virtual ~Variant() { }

  public:
    static const string CATEGORY;
};

/// @class VariantSet
/// A variant set element within a Document.
class MX_CORE_API VariantSet : public Element
{
  public:
    VariantSet(ElementPtr parent, const string& name) :
        Element(parent, CATEGORY, name)
    {
    }
    virtual ~VariantSet() { }

    /// @name Variant Elements
    /// @{

    /// Add a Variant to the variant set.
    /// @param name The name of the new Variant.
    ///     If no name is specified, then a unique name will automatically be
    ///     generated.
    /// @return A shared pointer to the new Variant.
    VariantPtr addVariant(const string& name = EMPTY_STRING)
    {
        return addChild<Variant>(name);
    }

    /// Return the Variant, if any, with the given name.
    VariantPtr getVariant(const string& name) const
    {
        return getChildOfType<Variant>(name);
    }

    /// Return a vector of all Variant elements in the look.
    vector<VariantPtr> getVariants() const
    {
        return getChildrenOfType<Variant>();
    }

    /// Remove the Variant, if any, with the given name.
    void removeVariant(const string& name)
    {
        removeChildOfType<Variant>(name);
    }

    /// @}

  public:
    static const string CATEGORY;
};

/// @class VariantAssign
/// A variant assignment element within a Look.
/// @todo Add support for variant assignments in graph traversal and
///    string resolution.
class MX_CORE_API VariantAssign : public Element
{
  public:
    VariantAssign(ElementPtr parent, const string& name) :
        Element(parent, CATEGORY, name)
    {
    }
    virtual ~VariantAssign() { }

    /// @name Variant Set String
    /// @{

    /// Set the element's variant set string.
    void setVariantSetString(const string& variantSet)
    {
        setAttribute(VARIANT_SET_ATTRIBUTE, variantSet);
    }

    /// Return true if the given element has a variant set string.
    bool hasVariantSetString() const
    {
        return hasAttribute(VARIANT_SET_ATTRIBUTE);
    }

    /// Return the element's variant set string.
    const string& getVariantSetString() const
    {
        return getAttribute(VARIANT_SET_ATTRIBUTE);
    }

    /// @}
    /// @name Variant String
    /// @{

    /// Set the element's variant string.
    void setVariantString(const string& variant)
    {
        setAttribute(VARIANT_ATTRIBUTE, variant);
    }

    /// Return true if the given element has a variant string.
    bool hasVariantString() const
    {
        return hasAttribute(VARIANT_ATTRIBUTE);
    }

    /// Return the element's variant string.
    const string& getVariantString() const
    {
        return getAttribute(VARIANT_ATTRIBUTE);
    }

    /// @}

  public:
    static const string CATEGORY;
    static const string VARIANT_SET_ATTRIBUTE;
    static const string VARIANT_ATTRIBUTE;
};

MATERIALX_NAMESPACE_END

#endif
