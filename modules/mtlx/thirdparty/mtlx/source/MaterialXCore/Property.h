//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_PROPERTY_H
#define MATERIALX_PROPERTY_H

/// @file
/// Property element subclasses

#include <MaterialXCore/Export.h>

#include <MaterialXCore/Geom.h>

MATERIALX_NAMESPACE_BEGIN

class Property;
class PropertyAssign;
class PropertySet;
class PropertySetAssign;

/// A shared pointer to a Property
using PropertyPtr = shared_ptr<Property>;
/// A shared pointer to a const Property
using ConstPropertyPtr = shared_ptr<const Property>;

/// A shared pointer to a PropertyAssign
using PropertyAssignPtr = shared_ptr<PropertyAssign>;
/// A shared pointer to a const PropertyAssign
using ConstPropertyAssignPtr = shared_ptr<const PropertyAssign>;

/// A shared pointer to a PropertySet
using PropertySetPtr = shared_ptr<PropertySet>;
/// A shared pointer to a const PropertySet
using ConstPropertySetPtr = shared_ptr<const PropertySet>;

/// A shared pointer to a PropertySetAssign
using PropertySetAssignPtr = shared_ptr<PropertySetAssign>;
/// A shared pointer to a const PropertySetAssign
using ConstPropertySetAssignPtr = shared_ptr<const PropertySetAssign>;

/// @class Property
/// A property element within a PropertySet.
class MX_CORE_API Property : public ValueElement
{
  public:
    Property(ElementPtr parent, const string& name) :
        ValueElement(parent, CATEGORY, name)
    {
    }
    virtual ~Property() { }

  public:
    static const string CATEGORY;
};

/// @class PropertyAssign
/// A property assignment element within a Look.
class MX_CORE_API PropertyAssign : public ValueElement
{
  public:
    PropertyAssign(ElementPtr parent, const string& name) :
        ValueElement(parent, CATEGORY, name)
    {
    }
    virtual ~PropertyAssign() { }

    /// @name Property
    /// @{

    /// Set the property string of this element.
    void setProperty(const string& property)
    {
        setAttribute(PROPERTY_ATTRIBUTE, property);
    }

    /// Return true if this element has a property string.
    bool hasProperty() const
    {
        return hasAttribute(PROPERTY_ATTRIBUTE);
    }

    /// Return the property string of this element.
    const string& getProperty() const
    {
        return getAttribute(PROPERTY_ATTRIBUTE);
    }

    /// @}
    /// @name Geometry
    /// @{

    /// Set the geometry string of this element.
    void setGeom(const string& geom)
    {
        setAttribute(GEOM_ATTRIBUTE, geom);
    }

    /// Return true if this element has a geometry string.
    bool hasGeom() const
    {
        return hasAttribute(GEOM_ATTRIBUTE);
    }

    /// Return the geometry string of this element.
    const string& getGeom() const
    {
        return getAttribute(GEOM_ATTRIBUTE);
    }

    /// @}
    /// @name Collection
    /// @{

    /// Set the collection string of this element.
    void setCollectionString(const string& collection)
    {
        setAttribute(COLLECTION_ATTRIBUTE, collection);
    }

    /// Return true if this element has a collection string.
    bool hasCollectionString() const
    {
        return hasAttribute(COLLECTION_ATTRIBUTE);
    }

    /// Return the collection string of this element.
    const string& getCollectionString() const
    {
        return getAttribute(COLLECTION_ATTRIBUTE);
    }

    /// Assign a Collection to this element.
    void setCollection(ConstCollectionPtr collection);

    /// Return the Collection that is assigned to this element.
    CollectionPtr getCollection() const;

    /// @}

  public:
    static const string CATEGORY;
    static const string PROPERTY_ATTRIBUTE;
    static const string GEOM_ATTRIBUTE;
    static const string COLLECTION_ATTRIBUTE;
};

/// @class PropertySet
/// A property set element within a Document.
class MX_CORE_API PropertySet : public Element
{
  public:
    PropertySet(ElementPtr parent, const string& name) :
        Element(parent, CATEGORY, name)
    {
    }
    virtual ~PropertySet() { }

    /// @name Properties
    /// @{

    /// Add a Property to the set.
    /// @param name The name of the new Property.
    ///     If no name is specified, then a unique name will automatically be
    ///     generated.
    /// @return A shared pointer to the new Property.
    PropertyPtr addProperty(const string& name)
    {
        return addChild<Property>(name);
    }

    /// Return the Property, if any, with the given name.
    PropertyPtr getProperty(const string& name) const
    {
        return getChildOfType<Property>(name);
    }

    /// Return a vector of all Property elements in the set.
    vector<PropertyPtr> getProperties() const
    {
        return getChildrenOfType<Property>();
    }

    /// Remove the Property with the given name, if present.
    void removeProperty(const string& name)
    {
        removeChildOfType<Property>(name);
    }

    /// @}
    /// @name Values
    /// @{

    /// Set the typed value of a property by its name, creating a child element
    /// to hold the property if needed.
    template <class T> PropertyPtr setPropertyValue(const string& name,
                                                    const T& value,
                                                    const string& type = EMPTY_STRING)
    {
        PropertyPtr property = getChildOfType<Property>(name);
        if (!property)
            property = addProperty(name);
        property->setValue(value, type);
        return property;
    }

    /// Return the typed value of a property by its name.
    /// @param name The name of the property to be evaluated.
    /// @return If the given property is found, then a shared pointer to its
    ///    value is returned; otherwise, an empty shared pointer is returned.
    ValuePtr getPropertyValue(const string& name) const
    {
        PropertyPtr property = getProperty(name);
        return property ? property->getValue() : ValuePtr();
    }

    /// @}

  public:
    static const string CATEGORY;
};

/// @class PropertySetAssign
/// A property set assignment element within a Look.
class MX_CORE_API PropertySetAssign : public GeomElement
{
  public:
    PropertySetAssign(ElementPtr parent, const string& name) :
        GeomElement(parent, CATEGORY, name)
    {
    }
    virtual ~PropertySetAssign() { }

    /// @name Property Set
    /// @{

    /// Set the property set string of this element.
    void setPropertySetString(const string& propertySet)
    {
        setAttribute(PROPERTY_SET_ATTRIBUTE, propertySet);
    }

    /// Return true if this element has a property set string.
    bool hasPropertySetString() const
    {
        return hasAttribute(PROPERTY_SET_ATTRIBUTE);
    }

    /// Return the property set string of this element.
    const string& getPropertySetString() const
    {
        return getAttribute(PROPERTY_SET_ATTRIBUTE);
    }

    /// Assign a property set to this element.
    void setPropertySet(ConstPropertySetPtr propertySet);

    /// Return the property set that is assigned to this element.
    PropertySetPtr getPropertySet() const;

    /// @}

  public:
    static const string CATEGORY;
    static const string PROPERTY_SET_ATTRIBUTE;
};

MATERIALX_NAMESPACE_END

#endif
