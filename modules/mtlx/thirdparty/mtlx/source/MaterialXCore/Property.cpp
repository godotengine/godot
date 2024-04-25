//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXCore/Property.h>

MATERIALX_NAMESPACE_BEGIN

const string PropertyAssign::PROPERTY_ATTRIBUTE = "property";
const string PropertyAssign::GEOM_ATTRIBUTE = "geom";
const string PropertyAssign::COLLECTION_ATTRIBUTE = "collection";
const string PropertySetAssign::PROPERTY_SET_ATTRIBUTE = "propertyset";

//
// PropertyAssign methods
//

void PropertyAssign::setCollection(ConstCollectionPtr collection)
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

CollectionPtr PropertyAssign::getCollection() const
{
    return resolveNameReference<Collection>(getCollectionString());
}

//
// PropertySetAssign methods
//

void PropertySetAssign::setPropertySet(ConstPropertySetPtr propertySet)
{
    if (propertySet)
    {
        setPropertySetString(propertySet->getName());
    }
    else
    {
        removeAttribute(PROPERTY_SET_ATTRIBUTE);
    }
}

PropertySetPtr PropertySetAssign::getPropertySet() const
{
    return resolveNameReference<PropertySet>(getPropertySetString());
}

MATERIALX_NAMESPACE_END
