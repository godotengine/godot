//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXCore/Geom.h>

#include <MaterialXCore/Document.h>

MATERIALX_NAMESPACE_BEGIN

const string GEOM_PATH_SEPARATOR = "/";
const string UNIVERSAL_GEOM_NAME = GEOM_PATH_SEPARATOR;
const string UDIM_TOKEN = "<UDIM>";
const string UV_TILE_TOKEN = "<UVTILE>";
const string UDIM_SET_PROPERTY = "udimset";

const string GeomElement::GEOM_ATTRIBUTE = "geom";
const string GeomElement::COLLECTION_ATTRIBUTE = "collection";
const string GeomPropDef::GEOM_PROP_ATTRIBUTE = "geomprop";
const string GeomPropDef::SPACE_ATTRIBUTE = "space";
const string GeomPropDef::INDEX_ATTRIBUTE = "index";
const string Collection::INCLUDE_GEOM_ATTRIBUTE = "includegeom";
const string Collection::EXCLUDE_GEOM_ATTRIBUTE = "excludegeom";
const string Collection::INCLUDE_COLLECTION_ATTRIBUTE = "includecollection";

bool geomStringsMatch(const string& geom1, const string& geom2, bool contains)
{
    vector<GeomPath> paths1;
    for (const string& name1 : splitString(geom1, ARRAY_VALID_SEPARATORS))
    {
        paths1.emplace_back(name1);
    }
    for (const string& name2 : splitString(geom2, ARRAY_VALID_SEPARATORS))
    {
        GeomPath path2(name2);
        for (const GeomPath& path1 : paths1)
        {
            if (path1.isMatching(path2, contains))
            {
                return true;
            }
        }
    }
    return false;
}

//
// GeomElement methods
//

void GeomElement::setCollection(ConstCollectionPtr collection)
{
    if (collection)
    {
        setCollectionString(collection->getName());
    }
    else
    {
        removeAttribute(COLLECTION_ATTRIBUTE);
    }
}

CollectionPtr GeomElement::getCollection() const
{
    return resolveNameReference<Collection>(getCollectionString());
}

bool GeomElement::validate(string* message) const
{
    bool res = true;
    if (hasCollectionString())
    {
        validateRequire(getCollection() != nullptr, res, message, "Invalid collection string");
    }
    return Element::validate(message) && res;
}

//
// Collection methods
//

void Collection::setIncludeCollection(ConstCollectionPtr collection)
{
    if (collection)
    {
        setIncludeCollectionString(collection->getName());
    }
    else
    {
        removeAttribute(INCLUDE_COLLECTION_ATTRIBUTE);
    }
}

void Collection::setIncludeCollections(const vector<ConstCollectionPtr>& collections)
{
    if (!collections.empty())
    {
        StringVec stringVec;
        for (ConstCollectionPtr collection : collections)
        {
            stringVec.push_back(collection->getName());
        }
        setTypedAttribute(INCLUDE_COLLECTION_ATTRIBUTE, stringVec);
    }
    else
    {
        removeAttribute(INCLUDE_COLLECTION_ATTRIBUTE);
    }
}

vector<CollectionPtr> Collection::getIncludeCollections() const
{
    vector<CollectionPtr> vec;
    for (const string& str : getTypedAttribute<StringVec>(INCLUDE_COLLECTION_ATTRIBUTE))
    {
        CollectionPtr collection = resolveNameReference<Collection>(str);
        if (collection)
        {
            vec.push_back(collection);
        }
    }
    return vec;
}

bool Collection::hasIncludeCycle() const
{
    try
    {
        matchesGeomString(UNIVERSAL_GEOM_NAME);
    }
    catch (ExceptionFoundCycle&)
    {
        return true;
    }
    return false;
}

bool Collection::matchesGeomString(const string& geom) const
{
    if (geomStringsMatch(getActiveExcludeGeom(), geom, true))
    {
        return false;
    }
    if (geomStringsMatch(getActiveIncludeGeom(), geom))
    {
        return true;
    }

    std::set<CollectionPtr> includedSet;
    vector<CollectionPtr> includedVec = getIncludeCollections();
    for (size_t i = 0; i < includedVec.size(); i++)
    {
        CollectionPtr collection = includedVec[i];
        if (includedSet.count(collection))
        {
            throw ExceptionFoundCycle("Encountered a cycle in collection: " + getName());
        }
        includedSet.insert(collection);
        vector<CollectionPtr> appendVec = collection->getIncludeCollections();
        includedVec.insert(includedVec.end(), appendVec.begin(), appendVec.end());
    }
    for (ConstCollectionPtr collection : includedSet)
    {
        if (collection->matchesGeomString(geom))
        {
            return true;
        }
    }

    return false;
}

bool Collection::validate(string* message) const
{
    bool res = true;
    validateRequire(!hasIncludeCycle(), res, message, "Cycle in collection include chain");
    return Element::validate(message) && res;
}

MATERIALX_NAMESPACE_END
