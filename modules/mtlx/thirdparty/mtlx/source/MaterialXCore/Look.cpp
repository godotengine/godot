//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXCore/Look.h>

#include <MaterialXCore/Document.h>

MATERIALX_NAMESPACE_BEGIN

const string MaterialAssign::MATERIAL_ATTRIBUTE = "material";
const string MaterialAssign::EXCLUSIVE_ATTRIBUTE = "exclusive";

const string Visibility::VIEWER_GEOM_ATTRIBUTE = "viewergeom";
const string Visibility::VIEWER_COLLECTION_ATTRIBUTE = "viewercollection";
const string Visibility::VISIBILITY_TYPE_ATTRIBUTE = "vistype";
const string Visibility::VISIBLE_ATTRIBUTE = "visible";

const string LookGroup::LOOKS_ATTRIBUTE = "looks";
const string LookGroup::ACTIVE_ATTRIBUTE = "active";

vector<MaterialAssignPtr> getGeometryBindings(ConstNodePtr materialNode, const string& geom)
{
    vector<MaterialAssignPtr> matAssigns;
    for (LookPtr look : materialNode->getDocument()->getLooks())
    {
        for (MaterialAssignPtr matAssign : look->getMaterialAssigns())
        {
            if (matAssign->getReferencedMaterial() == materialNode)
            {
                if (geomStringsMatch(matAssign->getActiveGeom(), geom, true))
                {
                    matAssigns.push_back(matAssign);
                    continue;
                }
                CollectionPtr coll = matAssign->getCollection();
                if (coll && coll->matchesGeomString(geom))
                {
                    matAssigns.push_back(matAssign);
                    continue;
                }
            }
        }
    }
    return matAssigns;
}

//
// Look methods
//

MaterialAssignPtr Look::addMaterialAssign(const string& name, const string& material)
{
    MaterialAssignPtr matAssign = addChild<MaterialAssign>(name);
    if (!material.empty())
    {
        matAssign->setMaterial(material);
    }
    return matAssign;
}

vector<MaterialAssignPtr> Look::getActiveMaterialAssigns() const
{
    vector<MaterialAssignPtr> activeAssigns;
    for (ConstElementPtr elem : traverseInheritance())
    {
        vector<MaterialAssignPtr> assigns = elem->asA<Look>()->getMaterialAssigns();
        activeAssigns.insert(activeAssigns.end(), assigns.begin(), assigns.end());
    }
    return activeAssigns;
}

vector<PropertyAssignPtr> Look::getActivePropertyAssigns() const
{
    vector<PropertyAssignPtr> activeAssigns;
    for (ConstElementPtr elem : traverseInheritance())
    {
        vector<PropertyAssignPtr> assigns = elem->asA<Look>()->getPropertyAssigns();
        activeAssigns.insert(activeAssigns.end(), assigns.begin(), assigns.end());
    }
    return activeAssigns;
}

vector<PropertySetAssignPtr> Look::getActivePropertySetAssigns() const
{
    vector<PropertySetAssignPtr> activeAssigns;
    for (ConstElementPtr elem : traverseInheritance())
    {
        vector<PropertySetAssignPtr> assigns = elem->asA<Look>()->getPropertySetAssigns();
        activeAssigns.insert(activeAssigns.end(), assigns.begin(), assigns.end());
    }
    return activeAssigns;
}

vector<VariantAssignPtr> Look::getActiveVariantAssigns() const
{
    vector<VariantAssignPtr> activeAssigns;
    for (ConstElementPtr elem : traverseInheritance())
    {
        vector<VariantAssignPtr> assigns = elem->asA<Look>()->getVariantAssigns();
        activeAssigns.insert(activeAssigns.end(), assigns.begin(), assigns.end());
    }
    return activeAssigns;
}

vector<VisibilityPtr> Look::getActiveVisibilities() const
{
    vector<VisibilityPtr> activeVisibilities;
    for (ConstElementPtr elem : traverseInheritance())
    {
        vector<VisibilityPtr> visibilities = elem->asA<Look>()->getVisibilities();
        activeVisibilities.insert(activeVisibilities.end(), visibilities.begin(), visibilities.end());
    }
    return activeVisibilities;
}

//
// MaterialAssign methods
//

NodePtr MaterialAssign::getReferencedMaterial() const
{
    return resolveNameReference<Node>(getMaterial());
}

vector<OutputPtr> MaterialAssign::getMaterialOutputs() const
{
    vector<OutputPtr> materialOutputs;
    NodeGraphPtr materialGraph = resolveNameReference<NodeGraph>(getMaterial());
    if (materialGraph)
    {
        return materialGraph->getMaterialOutputs();
    }
    return materialOutputs;
}

vector<VariantAssignPtr> MaterialAssign::getActiveVariantAssigns() const
{
    vector<VariantAssignPtr> activeAssigns;
    for (ConstElementPtr elem : traverseInheritance())
    {
        vector<VariantAssignPtr> assigns = elem->asA<MaterialAssign>()->getVariantAssigns();
        activeAssigns.insert(activeAssigns.end(), assigns.begin(), assigns.end());
    }
    return activeAssigns;
}

MATERIALX_NAMESPACE_END
