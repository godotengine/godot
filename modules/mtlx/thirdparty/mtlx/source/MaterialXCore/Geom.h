//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_GEOM_H
#define MATERIALX_GEOM_H

/// @file
/// Geometric element subclasses

#include <MaterialXCore/Export.h>

#include <MaterialXCore/Element.h>

MATERIALX_NAMESPACE_BEGIN

extern MX_CORE_API const string GEOM_PATH_SEPARATOR;
extern MX_CORE_API const string UNIVERSAL_GEOM_NAME;
extern MX_CORE_API const string UDIM_TOKEN;
extern MX_CORE_API const string UV_TILE_TOKEN;
extern MX_CORE_API const string UDIM_SET_PROPERTY;

class GeomElement;
class GeomInfo;
class GeomProp;
class GeomPropDef;
class Collection;
class CollectionAdd;
class CollectionRemove;

/// A shared pointer to a GeomElement
using GeomElementPtr = shared_ptr<GeomElement>;
/// A shared pointer to a const GeomElement
using ConstGeomElementPtr = shared_ptr<const GeomElement>;

/// A shared pointer to a GeomInfo
using GeomInfoPtr = shared_ptr<GeomInfo>;
/// A shared pointer to a const GeomInfo
using ConstGeomInfoPtr = shared_ptr<const GeomInfo>;

/// A shared pointer to a GeomProp
using GeomPropPtr = shared_ptr<GeomProp>;
/// A shared pointer to a const GeomProp
using ConstGeomPropPtr = shared_ptr<const GeomProp>;

/// A shared pointer to a GeomPropDef
using GeomPropDefPtr = shared_ptr<GeomPropDef>;
/// A shared pointer to a const GeomPropDef
using ConstGeomPropDefPtr = shared_ptr<const GeomPropDef>;

/// A shared pointer to a Collection
using CollectionPtr = shared_ptr<Collection>;
/// A shared pointer to a const Collection
using ConstCollectionPtr = shared_ptr<const Collection>;

/// @class GeomPath
/// A MaterialX geometry path, representing the hierarchical location
/// expressed by a geometry name.
class MX_CORE_API GeomPath
{
  public:
    GeomPath() :
        _empty(true)
    {
    }
    ~GeomPath() { }

    bool operator==(const GeomPath& rhs) const
    {
        return _vec == rhs._vec &&
               _empty == rhs._empty;
    }
    bool operator!=(const GeomPath& rhs) const
    {
        return !(*this == rhs);
    }

    /// Construct a path from a geometry name string.
    explicit GeomPath(const string& geom) :
        _vec(splitString(geom, GEOM_PATH_SEPARATOR)),
        _empty(geom.empty())
    {
    }

    /// Convert a path to a geometry name string.
    operator string() const
    {
        if (_vec.empty())
        {
            return _empty ? EMPTY_STRING : UNIVERSAL_GEOM_NAME;
        }
        return GEOM_PATH_SEPARATOR + joinStrings(_vec, GEOM_PATH_SEPARATOR);
    }

    /// Return true if there is any geometry in common between the two paths.
    /// @param rhs A second geometry path to be compared with this one
    /// @param contains If true, then we require that the first path completely
    ///    contains the second one.
    bool isMatching(const GeomPath& rhs, bool contains = false) const
    {
        if (_empty || rhs._empty)
        {
            return false;
        }
        if (contains && _vec.size() > rhs._vec.size())
        {
            return false;
        }
        size_t minSize = std::min(_vec.size(), rhs._vec.size());
        for (size_t i = 0; i < minSize; i++)
        {
            if (_vec[i] != rhs._vec[i])
            {
                return false;
            }
        }
        return true;
    }

    /// Return true if this geometry path is empty.  An empty path matches
    /// no other geometry paths.
    bool isEmpty() const
    {
        return _empty;
    }

    /// Return true if this geometry path is universal.  A universal path
    /// matches all non-empty geometry paths.
    bool isUniversal() const
    {
        return _vec.empty() && !_empty;
    }

  private:
    StringVec _vec;
    bool _empty;
};

/// @class GeomElement
/// The base class for geometric elements, which support bindings to geometries
/// and geometric collections.
class MX_CORE_API GeomElement : public Element
{
  protected:
    GeomElement(ElementPtr parent, const string& category, const string& name) :
        Element(parent, category, name)
    {
    }

  public:
    virtual ~GeomElement() { }

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

    /// Return the active geometry string of this element, taking all geometry
    /// string substitutions at this scope into account.
    string getActiveGeom() const
    {
        return hasGeom() ?
               createStringResolver()->resolve(getGeom(), GEOMNAME_TYPE_STRING) :
               EMPTY_STRING;
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
    /// @name Validation
    /// @{

    /// Validate that the given element tree, including all descendants, is
    /// consistent with the MaterialX specification.
    bool validate(string* message = nullptr) const override;

    /// @}

  public:
    static const string GEOM_ATTRIBUTE;
    static const string COLLECTION_ATTRIBUTE;
};

/// @class GeomInfo
/// A geometry info element within a Document.
class MX_CORE_API GeomInfo : public GeomElement
{
  public:
    GeomInfo(ElementPtr parent, const string& name) :
        GeomElement(parent, CATEGORY, name)
    {
    }
    virtual ~GeomInfo() { }

    /// @name GeomProp Elements
    /// @{

    /// Add a GeomProp to this element.
    /// @param name The name of the new GeomProp.
    ///     If no name is specified, then a unique name will automatically be
    ///     generated.
    /// @return A shared pointer to the new GeomProp.
    GeomPropPtr addGeomProp(const string& name = EMPTY_STRING)
    {
        return addChild<GeomProp>(name);
    }

    /// Return the GeomProp, if any, with the given name.
    GeomPropPtr getGeomProp(const string& name) const
    {
        return getChildOfType<GeomProp>(name);
    }

    /// Return a vector of all GeomProp elements.
    vector<GeomPropPtr> getGeomProps() const
    {
        return getChildrenOfType<GeomProp>();
    }

    /// Remove the GeomProp, if any, with the given name.
    void removeGeomProp(const string& name)
    {
        removeChildOfType<GeomProp>(name);
    }

    /// @}
    /// @name Tokens
    /// @{

    /// Add a Token to this element.
    /// @param name The name of the new Token.
    ///     If no name is specified, then a unique name will automatically be
    ///     generated.
    /// @return A shared pointer to the new Token.
    TokenPtr addToken(const string& name = EMPTY_STRING)
    {
        return addChild<Token>(name);
    }

    /// Return the Token, if any, with the given name.
    TokenPtr getToken(const string& name) const
    {
        return getChildOfType<Token>(name);
    }

    /// Return a vector of all Token elements.
    vector<TokenPtr> getTokens() const
    {
        return getChildrenOfType<Token>();
    }

    /// Remove the Token, if any, with the given name.
    void removeToken(const string& name)
    {
        removeChildOfType<Token>(name);
    }

    /// @}
    /// @name Values
    /// @{

    /// Set the value of a GeomProp by its name, creating a child element
    /// to hold the GeomProp if needed.
    template <class T> GeomPropPtr setGeomPropValue(const string& name,
                                                    const T& value,
                                                    const string& type = EMPTY_STRING);

    /// Set the string value of a Token by its name, creating a child element
    /// to hold the Token if needed.
    TokenPtr setTokenValue(const string& name, const string& value)
    {
        TokenPtr token = getToken(name);
        if (!token)
            token = addToken(name);
        token->setValue<string>(value);
        return token;
    }

    /// @}

  public:
    static const string CATEGORY;
};

/// @class GeomProp
/// A geometric property element within a GeomInfo.
class MX_CORE_API GeomProp : public ValueElement
{
  public:
    GeomProp(ElementPtr parent, const string& name) :
        ValueElement(parent, CATEGORY, name)
    {
    }
    virtual ~GeomProp() { }

  public:
    static const string CATEGORY;
};

/// @class GeomPropDef
/// An element representing a declaration of geometric property data.
///
/// A GeomPropDef element contains a reference to a geometric node and a set of
/// modifiers for that node.  For example, a world-space normal can be declared
/// as a reference to the "normal" geometric node with a space setting of
/// "world", or a specific set of texture coordinates can be declared as a
/// reference to the "texcoord" geometric node with an index setting of "1".
class MX_CORE_API GeomPropDef : public TypedElement
{
  public:
    GeomPropDef(ElementPtr parent, const string& name) :
        TypedElement(parent, CATEGORY, name)
    {
    }
    virtual ~GeomPropDef() { }

    /// @name Geometric Property
    /// @{

    /// Set the geometric property string of this element.
    void setGeomProp(const string& node)
    {
        setAttribute(GEOM_PROP_ATTRIBUTE, node);
    }

    /// Return true if this element has a geometric property string.
    bool hasGeomProp() const
    {
        return hasAttribute(GEOM_PROP_ATTRIBUTE);
    }

    /// Return the geometric property string of this element.
    const string& getGeomProp() const
    {
        return getAttribute(GEOM_PROP_ATTRIBUTE);
    }

    /// @}
    /// @name Geometric Space
    /// @{

    /// Set the geometric space string of this element.
    void setSpace(const string& space)
    {
        setAttribute(SPACE_ATTRIBUTE, space);
    }

    /// Return true if this element has a geometric space string.
    bool hasSpace() const
    {
        return hasAttribute(SPACE_ATTRIBUTE);
    }

    /// Return the geometric space string of this element.
    const string& getSpace() const
    {
        return getAttribute(SPACE_ATTRIBUTE);
    }

    /// @}
    /// @name Geometric Index
    /// @{

    /// Set the index string of this element.
    void setIndex(const string& space)
    {
        setAttribute(INDEX_ATTRIBUTE, space);
    }

    /// Return true if this element has an index string.
    bool hasIndex() const
    {
        return hasAttribute(INDEX_ATTRIBUTE);
    }

    /// Return the index string of this element.
    const string& getIndex() const
    {
        return getAttribute(INDEX_ATTRIBUTE);
    }

    /// @}

  public:
    static const string CATEGORY;
    static const string GEOM_PROP_ATTRIBUTE;
    static const string SPACE_ATTRIBUTE;
    static const string INDEX_ATTRIBUTE;
};

/// @class Collection
/// A collection element within a Document.
class MX_CORE_API Collection : public Element
{
  public:
    Collection(ElementPtr parent, const string& name) :
        Element(parent, CATEGORY, name)
    {
    }
    virtual ~Collection() { }

    /// @name Include Geometry
    /// @{

    /// Set the include geometry string of this element.
    void setIncludeGeom(const string& geom)
    {
        setAttribute(INCLUDE_GEOM_ATTRIBUTE, geom);
    }

    /// Return true if this element has an include geometry string.
    bool hasIncludeGeom() const
    {
        return hasAttribute(INCLUDE_GEOM_ATTRIBUTE);
    }

    /// Return the include geometry string of this element.
    const string& getIncludeGeom() const
    {
        return getAttribute(INCLUDE_GEOM_ATTRIBUTE);
    }

    /// Return the active include geometry string of this element, taking all
    /// geometry string substitutions at this scope into account.
    string getActiveIncludeGeom() const
    {
        return hasIncludeGeom() ?
               createStringResolver()->resolve(getIncludeGeom(), GEOMNAME_TYPE_STRING) :
               EMPTY_STRING;
    }

    /// @}
    /// @name Exclude Geometry
    /// @{

    /// Set the exclude geometry string of this element.
    void setExcludeGeom(const string& geom)
    {
        setAttribute(EXCLUDE_GEOM_ATTRIBUTE, geom);
    }

    /// Return true if this element has an exclude geometry string.
    bool hasExcludeGeom() const
    {
        return hasAttribute(EXCLUDE_GEOM_ATTRIBUTE);
    }

    /// Return the exclude geometry string of this element.
    const string& getExcludeGeom() const
    {
        return getAttribute(EXCLUDE_GEOM_ATTRIBUTE);
    }

    /// Return the active exclude geometry string of this element, taking all
    /// geometry string substitutions at this scope into account.
    string getActiveExcludeGeom() const
    {
        return hasExcludeGeom() ?
               createStringResolver()->resolve(getExcludeGeom(), GEOMNAME_TYPE_STRING) :
               EMPTY_STRING;
    }

    /// @}
    /// @name Include Collection
    /// @{

    /// Set the include collection string of this element.
    void setIncludeCollectionString(const string& collection)
    {
        setAttribute(INCLUDE_COLLECTION_ATTRIBUTE, collection);
    }

    /// Return true if this element has an include collection string.
    bool hasIncludeCollectionString() const
    {
        return hasAttribute(INCLUDE_COLLECTION_ATTRIBUTE);
    }

    /// Return the include collection string of this element.
    const string& getIncludeCollectionString() const
    {
        return getAttribute(INCLUDE_COLLECTION_ATTRIBUTE);
    }

    /// Set the collection that is directly included by this element.
    void setIncludeCollection(ConstCollectionPtr collection);

    /// Set the vector of collections that are directly included by
    /// this element.
    void setIncludeCollections(const vector<ConstCollectionPtr>& collections);

    /// Return the vector of collections that are directly included by
    /// this element.
    vector<CollectionPtr> getIncludeCollections() const;

    /// Return true if the include chain for this element contains a cycle.
    bool hasIncludeCycle() const;

    /// @}
    /// @name Geometry Matching
    /// @{

    /// Return true if this collection and the given geometry string have any
    /// geometries in common.
    /// @throws ExceptionFoundCycle if a cycle is encountered.
    bool matchesGeomString(const string& geom) const;

    /// @}
    /// @name Validation
    /// @{

    /// Validate that the given element tree, including all descendants, is
    /// consistent with the MaterialX specification.
    bool validate(string* message = nullptr) const override;

    /// @}

  public:
    static const string CATEGORY;
    static const string INCLUDE_GEOM_ATTRIBUTE;
    static const string EXCLUDE_GEOM_ATTRIBUTE;
    static const string INCLUDE_COLLECTION_ATTRIBUTE;
};

template <class T> GeomPropPtr GeomInfo::setGeomPropValue(const string& name,
                                                          const T& value,
                                                          const string& type)
{
    GeomPropPtr geomProp = getChildOfType<GeomProp>(name);
    if (!geomProp)
        geomProp = addGeomProp(name);
    geomProp->setValue(value, type);
    return geomProp;
}

/// Given two geometry strings, each containing an array of geom names, return
/// true if they have any geometries in common.
///
/// An empty geometry string matches no geometries, while the universal geometry
/// string "/" matches all non-empty geometries.
///
/// If the contains argument is set to true, then we require that a geom path
/// in the first string completely contains a geom path in the second string.
///
/// @todo Geometry name expressions are not yet supported.
MX_CORE_API bool geomStringsMatch(const string& geom1, const string& geom2, bool contains = false);

MATERIALX_NAMESPACE_END

#endif
