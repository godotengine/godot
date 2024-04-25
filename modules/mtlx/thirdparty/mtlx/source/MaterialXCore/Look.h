//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_LOOK_H
#define MATERIALX_LOOK_H

/// @file
/// Look element subclasses

#include <MaterialXCore/Export.h>

#include <MaterialXCore/Material.h>
#include <MaterialXCore/Property.h>
#include <MaterialXCore/Variant.h>

MATERIALX_NAMESPACE_BEGIN

class Look;
class LookGroup;
class LookInherit;
class MaterialAssign;
class Visibility;

/// A shared pointer to a Look
using LookPtr = shared_ptr<Look>;
/// A shared pointer to a const Look
using ConstLookPtr = shared_ptr<const Look>;

/// A shared pointer to a LookGroup
using LookGroupPtr = shared_ptr<LookGroup>;
/// A shared pointer to a const LookGroup
using ConstLookGroupPtr = shared_ptr<const LookGroup>;

/// A shared pointer to a MaterialAssign
using MaterialAssignPtr = shared_ptr<MaterialAssign>;
/// A shared pointer to a const MaterialAssign
using ConstMaterialAssignPtr = shared_ptr<const MaterialAssign>;

/// A shared pointer to a Visibility
using VisibilityPtr = shared_ptr<Visibility>;
/// A shared pointer to a const Visibility
using ConstVisibilityPtr = shared_ptr<const Visibility>;

/// @class Look
/// A look element within a Document.
class MX_CORE_API Look : public Element
{
  public:
    Look(ElementPtr parent, const string& name) :
        Element(parent, CATEGORY, name)
    {
    }
    virtual ~Look() { }

    /// @name MaterialAssign Elements
    /// @{

    /// Add a MaterialAssign to the look.
    /// @param name The name of the new MaterialAssign.
    ///     If no name is specified, then a unique name will automatically be
    ///     generated.
    /// @param material An optional material string, which should match the
    ///     name of the material node to be assigned.
    /// @return A shared pointer to the new MaterialAssign.
    MaterialAssignPtr addMaterialAssign(const string& name = EMPTY_STRING,
                                        const string& material = EMPTY_STRING);

    /// Return the MaterialAssign, if any, with the given name.
    MaterialAssignPtr getMaterialAssign(const string& name) const
    {
        return getChildOfType<MaterialAssign>(name);
    }

    /// Return a vector of all MaterialAssign elements in the look.
    vector<MaterialAssignPtr> getMaterialAssigns() const
    {
        return getChildrenOfType<MaterialAssign>();
    }

    /// Return a vector of all MaterialAssign elements that belong to this look,
    /// taking look inheritance into account.
    vector<MaterialAssignPtr> getActiveMaterialAssigns() const;

    /// Remove the MaterialAssign, if any, with the given name.
    void removeMaterialAssign(const string& name)
    {
        removeChildOfType<MaterialAssign>(name);
    }

    /// @}
    /// @name PropertyAssign Elements
    /// @{

    /// Add a PropertyAssign to the look.
    /// @param name The name of the new PropertyAssign.
    ///     If no name is specified, then a unique name will automatically be
    ///     generated.
    /// @return A shared pointer to the new PropertyAssign.
    PropertyAssignPtr addPropertyAssign(const string& name = EMPTY_STRING)
    {
        return addChild<PropertyAssign>(name);
    }

    /// Return the PropertyAssign, if any, with the given name.
    PropertyAssignPtr getPropertyAssign(const string& name) const
    {
        return getChildOfType<PropertyAssign>(name);
    }

    /// Return a vector of all PropertyAssign elements in the look.
    vector<PropertyAssignPtr> getPropertyAssigns() const
    {
        return getChildrenOfType<PropertyAssign>();
    }

    /// Return a vector of all PropertyAssign elements that belong to this look,
    /// taking look inheritance into account.
    vector<PropertyAssignPtr> getActivePropertyAssigns() const;

    /// Remove the PropertyAssign, if any, with the given name.
    void removePropertyAssign(const string& name)
    {
        removeChildOfType<PropertyAssign>(name);
    }

    /// @}
    /// @name PropertySetAssign Elements
    /// @{

    /// Add a PropertySetAssign to the look.
    /// @param name The name of the new PropertySetAssign.
    ///     If no name is specified, then a unique name will automatically be
    ///     generated.
    /// @return A shared pointer to the new PropertySetAssign.
    PropertySetAssignPtr addPropertySetAssign(const string& name = EMPTY_STRING)
    {
        return addChild<PropertySetAssign>(name);
    }

    /// Return the PropertySetAssign, if any, with the given name.
    PropertySetAssignPtr getPropertySetAssign(const string& name) const
    {
        return getChildOfType<PropertySetAssign>(name);
    }

    /// Return a vector of all PropertySetAssign elements in the look.
    vector<PropertySetAssignPtr> getPropertySetAssigns() const
    {
        return getChildrenOfType<PropertySetAssign>();
    }

    /// Return a vector of all PropertySetAssign elements that belong to this look,
    /// taking look inheritance into account.
    vector<PropertySetAssignPtr> getActivePropertySetAssigns() const;

    /// Remove the PropertySetAssign, if any, with the given name.
    void removePropertySetAssign(const string& name)
    {
        removeChildOfType<PropertySetAssign>(name);
    }

    /// @}
    /// @name VariantAssign Elements
    /// @{

    /// Add a VariantAssign to the look.
    /// @param name The name of the new VariantAssign.
    ///     If no name is specified, then a unique name will automatically be
    ///     generated.
    /// @return A shared pointer to the new VariantAssign.
    VariantAssignPtr addVariantAssign(const string& name = EMPTY_STRING)
    {
        return addChild<VariantAssign>(name);
    }

    /// Return the VariantAssign, if any, with the given name.
    VariantAssignPtr getVariantAssign(const string& name) const
    {
        return getChildOfType<VariantAssign>(name);
    }

    /// Return a vector of all VariantAssign elements in the look.
    vector<VariantAssignPtr> getVariantAssigns() const
    {
        return getChildrenOfType<VariantAssign>();
    }

    /// Return a vector of all VariantAssign elements that belong to this look,
    /// taking look inheritance into account.
    vector<VariantAssignPtr> getActiveVariantAssigns() const;

    /// Remove the VariantAssign, if any, with the given name.
    void removeVariantAssign(const string& name)
    {
        removeChildOfType<VariantAssign>(name);
    }

    /// @}
    /// @name Visibility Elements
    /// @{

    /// Add a Visibility to the look.
    /// @param name The name of the new Visibility.
    ///     If no name is specified, then a unique name will automatically be
    ///     generated.
    /// @return A shared pointer to the new Visibility.
    VisibilityPtr addVisibility(const string& name = EMPTY_STRING)
    {
        return addChild<Visibility>(name);
    }

    /// Return the Visibility, if any, with the given name.
    VisibilityPtr getVisibility(const string& name) const
    {
        return getChildOfType<Visibility>(name);
    }

    /// Return a vector of all Visibility elements in the look.
    vector<VisibilityPtr> getVisibilities() const
    {
        return getChildrenOfType<Visibility>();
    }

    /// Return a vector of all Visibility elements that belong to this look,
    /// taking look inheritance into account.
    vector<VisibilityPtr> getActiveVisibilities() const;

    /// Remove the Visibility, if any, with the given name.
    void removeVisibility(const string& name)
    {
        removeChildOfType<Visibility>(name);
    }

    /// @}

  public:
    static const string CATEGORY;
};

/// @class LookGroup
/// A look group element within a Document.
class MX_CORE_API LookGroup : public Element
{
  public:
    LookGroup(ElementPtr parent, const string& name) :
        Element(parent, CATEGORY, name)
    {
    }
    virtual ~LookGroup() { }

    /// Set comma-separated list of looks.
    void setLooks(const string& looks)
    {
        setAttribute(LOOKS_ATTRIBUTE, looks);
    }

    /// Get comma-separated list of looks.
    const string& getLooks() const
    {
        return getAttribute(LOOKS_ATTRIBUTE);
    }

    /// Set the active look.
    void setActiveLook(const string& look)
    {
        setAttribute(ACTIVE_ATTRIBUTE, look);
    }

    /// Return the active look, if any.
    const string& getActiveLook() const
    {
        return getAttribute(ACTIVE_ATTRIBUTE);
    }

  public:
    static const string CATEGORY;
    static const string LOOKS_ATTRIBUTE;
    static const string ACTIVE_ATTRIBUTE;
};

/// @class MaterialAssign
/// A material assignment element within a Look.
class MX_CORE_API MaterialAssign : public GeomElement
{
  public:
    MaterialAssign(ElementPtr parent, const string& name) :
        GeomElement(parent, CATEGORY, name)
    {
    }
    virtual ~MaterialAssign() { }

    /// @name Material String
    /// @{

    /// Set the material string for the MaterialAssign.
    void setMaterial(const string& material)
    {
        setAttribute(MATERIAL_ATTRIBUTE, material);
    }

    /// Return true if the given MaterialAssign has a material string.
    bool hasMaterial() const
    {
        return hasAttribute(MATERIAL_ATTRIBUTE);
    }

    /// Return the material string for the MaterialAssign.
    const string& getMaterial() const
    {
        return getAttribute(MATERIAL_ATTRIBUTE);
    }

    ///  Return the outputs on any referenced material
    vector<OutputPtr> getMaterialOutputs() const;

    /// @}
    /// @name Exclusive
    /// @{

    /// Set the exclusive boolean for the MaterialAssign.
    void setExclusive(bool value)
    {
        setTypedAttribute<bool>(EXCLUSIVE_ATTRIBUTE, value);
    }

    /// Return the exclusive boolean for the MaterialAssign.
    bool getExclusive() const
    {
        return getTypedAttribute<bool>(EXCLUSIVE_ATTRIBUTE);
    }

    /// @}
    /// @name Material References
    /// @{

    /// Return the material node, if any, referenced by the MaterialAssign.
    NodePtr getReferencedMaterial() const;

    /// @}
    /// @name VariantAssign Elements
    /// @{

    /// Add a VariantAssign to the look.
    /// @param name The name of the new VariantAssign.
    ///     If no name is specified, then a unique name will automatically be
    ///     generated.
    /// @return A shared pointer to the new VariantAssign.
    VariantAssignPtr addVariantAssign(const string& name = EMPTY_STRING)
    {
        return addChild<VariantAssign>(name);
    }

    /// Return the VariantAssign, if any, with the given name.
    VariantAssignPtr getVariantAssign(const string& name) const
    {
        return getChildOfType<VariantAssign>(name);
    }

    /// Return a vector of all VariantAssign elements in the look.
    vector<VariantAssignPtr> getVariantAssigns() const
    {
        return getChildrenOfType<VariantAssign>();
    }

    /// Return a vector of all VariantAssign elements that belong to this look,
    /// taking look inheritance into account.
    vector<VariantAssignPtr> getActiveVariantAssigns() const;

    /// Remove the VariantAssign, if any, with the given name.
    void removeVariantAssign(const string& name)
    {
        removeChildOfType<VariantAssign>(name);
    }

  public:
    static const string CATEGORY;
    static const string MATERIAL_ATTRIBUTE;
    static const string EXCLUSIVE_ATTRIBUTE;
};

/// @class Visibility
/// A visibility element within a Look.
///
/// A Visibility describes the visibility relationship between two geometries
/// or geometric collections.
///
/// @todo Add a Look::geomIsVisible method that computes the visibility between
///     two geometries in the context of a specific Look.
class MX_CORE_API Visibility : public GeomElement
{
  public:
    Visibility(ElementPtr parent, const string& name) :
        GeomElement(parent, CATEGORY, name)
    {
    }
    virtual ~Visibility() { }

    /// @name Viewer Geom
    /// @{

    /// Set the viewer geom string of the element.
    void setViewerGeom(const string& geom)
    {
        setAttribute(VIEWER_GEOM_ATTRIBUTE, geom);
    }

    /// Return true if the given element has a viewer geom string.
    bool hasViewerGeom() const
    {
        return hasAttribute(VIEWER_GEOM_ATTRIBUTE);
    }

    /// Return the viewer geom string of the element.
    const string& getViewerGeom() const
    {
        return getAttribute(VIEWER_GEOM_ATTRIBUTE);
    }

    /// @}
    /// @name Viewer Collection
    /// @{

    /// Set the viewer geom string of the element.
    void setViewerCollection(const string& collection)
    {
        setAttribute(VIEWER_COLLECTION_ATTRIBUTE, collection);
    }

    /// Return true if the given element has a viewer collection string.
    bool hasViewerCollection() const
    {
        return hasAttribute(VIEWER_COLLECTION_ATTRIBUTE);
    }

    /// Return the viewer collection string of the element.
    const string& getViewerCollection() const
    {
        return getAttribute(VIEWER_COLLECTION_ATTRIBUTE);
    }

    /// @}
    /// @name Visibility Type
    /// @{

    /// Set the visibility type string of the element.
    void setVisibilityType(const string& type)
    {
        setAttribute(VISIBILITY_TYPE_ATTRIBUTE, type);
    }

    /// Return true if the given element has a visibility type string.
    bool hasVisibilityType() const
    {
        return hasAttribute(VISIBILITY_TYPE_ATTRIBUTE);
    }

    /// Return the visibility type string of the element.
    const string& getVisibilityType() const
    {
        return getAttribute(VISIBILITY_TYPE_ATTRIBUTE);
    }

    /// @}
    /// @name Visible
    /// @{

    /// Set the visible boolean of the element.
    void setVisible(bool visible)
    {
        setTypedAttribute<bool>(VISIBLE_ATTRIBUTE, visible);
    }

    /// Return the visible boolean of the element.
    bool getVisible() const
    {
        return getTypedAttribute<bool>(VISIBLE_ATTRIBUTE);
    }

    /// @}

  public:
    static const string CATEGORY;
    static const string VIEWER_GEOM_ATTRIBUTE;
    static const string VIEWER_COLLECTION_ATTRIBUTE;
    static const string VISIBILITY_TYPE_ATTRIBUTE;
    static const string VISIBLE_ATTRIBUTE;
};

/// Return a vector of all MaterialAssign elements that bind this material node
/// to the given geometry string
/// @param materialNode Node to examine
/// @param geom The geometry for which material bindings should be returned.
///             By default, this argument is the universal geometry string "/",
///             and all material bindings are returned.
/// @return Vector of MaterialAssign elements
MX_CORE_API vector<MaterialAssignPtr> getGeometryBindings(ConstNodePtr materialNode, const string& geom = UNIVERSAL_GEOM_NAME);

MATERIALX_NAMESPACE_END

#endif
