//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXTest/External/Catch/catch.hpp>

#include <MaterialXCore/Document.h>

namespace mx = MaterialX;

TEST_CASE("Look", "[look]")
{
    mx::DocumentPtr doc = mx::createDocument();

    // Create a material and look.
    mx::NodePtr shaderNode = doc->addNode("standard_surface", "", mx::SURFACE_SHADER_TYPE_STRING);
    mx::NodePtr materialNode = doc->addMaterialNode("", shaderNode);
    mx::LookPtr look = doc->addLook();

    // Bind the material to a geometry string.
    mx::MaterialAssignPtr matAssign1 = look->addMaterialAssign("matAssign1", materialNode->getName());
    matAssign1->setGeom("/robot1");
    REQUIRE(matAssign1->getReferencedMaterial() == materialNode);
    REQUIRE(getGeometryBindings(materialNode, "/robot1").size() == 1);
    REQUIRE(getGeometryBindings(materialNode, "/robot2").size() == 0);

    // Bind the material to a geometric collection.
    mx::MaterialAssignPtr matAssign2 = look->addMaterialAssign("matAssign2", materialNode->getName());
    mx::CollectionPtr collection = doc->addCollection();
    collection->setIncludeGeom("/robot2");
    collection->setExcludeGeom("/robot2/left_arm");
    matAssign2->setCollection(collection);
    REQUIRE(getGeometryBindings(materialNode, "/robot2").size() == 1);
    REQUIRE(getGeometryBindings(materialNode, "/robot2/right_arm").size() == 1);
    REQUIRE(getGeometryBindings(materialNode, "/robot2/left_arm").size() == 0);

    // Create a property assignment.
    mx::PropertyAssignPtr propertyAssign = look->addPropertyAssign();
	propertyAssign->setProperty("twosided");
    propertyAssign->setGeom("/robot1");
    propertyAssign->setValue(true);
    REQUIRE(propertyAssign->getProperty() == "twosided");
    REQUIRE(propertyAssign->getGeom() == "/robot1");
    REQUIRE(propertyAssign->getValue()->isA<bool>());
    REQUIRE(propertyAssign->getValue()->asA<bool>() == true);

    // Create a property set assignment.
    mx::PropertySetPtr propertySet = doc->addPropertySet();
    propertySet->setPropertyValue("matte", false);
    REQUIRE(propertySet->getPropertyValue("matte")->isA<bool>());
    REQUIRE(propertySet->getPropertyValue("matte")->asA<bool>() == false);
    mx::PropertySetAssignPtr propertySetAssign = look->addPropertySetAssign();
	propertySetAssign->setPropertySet(propertySet);
    propertySetAssign->setGeom("/robot1");
    REQUIRE(propertySetAssign->getPropertySet() == propertySet);
    REQUIRE(propertySetAssign->getGeom() == "/robot1");
    
    // Create a variant set.
    mx::VariantSetPtr variantSet = doc->addVariantSet("damageVars");
    variantSet->addVariant("original");
    variantSet->addVariant("damaged");
    REQUIRE(variantSet->getVariants().size() == 2);

    // Create a visibility element.
    mx::VisibilityPtr visibility = look->addVisibility();
    REQUIRE(visibility->getVisible() == false);
    visibility->setVisible(true);
    REQUIRE(visibility->getVisible() == true);
    visibility->setGeom("/robot2");
    REQUIRE(visibility->getGeom() == "/robot2");
    visibility->setCollection(collection);
    REQUIRE(visibility->getCollection() == collection);

    // Create an inherited look.
    mx::LookPtr look2 = doc->addLook();
    look2->setInheritsFrom(look);
    REQUIRE(look2->getActiveMaterialAssigns().size() == 2);
    REQUIRE(look2->getActivePropertySetAssigns().size() == 1);
    REQUIRE(look2->getActiveVisibilities().size() == 1);

    // Create and detect an inheritance cycle.
    look->setInheritsFrom(look2);
    REQUIRE(!doc->validate());
    look->setInheritsFrom(nullptr);
    REQUIRE(doc->validate());

    // Disconnect the inherited look.
    look2->setInheritsFrom(nullptr);
    REQUIRE(look2->getActiveMaterialAssigns().empty());
    REQUIRE(look2->getActivePropertySetAssigns().empty());
    REQUIRE(look2->getActiveVisibilities().empty());
}

TEST_CASE("LookGroup", "[look]")
{
    mx::DocumentPtr doc = mx::createDocument();

    mx::LookGroupPtr lookGroup = doc->addLookGroup("lookgroup1");
    std::vector<mx::LookGroupPtr> lookGroups = doc->getLookGroups();
    REQUIRE(lookGroups.size() == 1);

    const std::string looks = "look1,look2,look3,look4,look5";
    mx::StringVec looksVec = mx::splitString(looks, ",");
    for (const std::string& lookName : looksVec)
    {
        mx::LookPtr look = doc->addLook(lookName);
        REQUIRE(look != nullptr);
    }
    lookGroup->setLooks(looks);

    const std::string& looks2 = lookGroup->getLooks();
    mx::StringVec looksVec2 = mx::splitString(looks2, ",");
    REQUIRE(looksVec.size() == looksVec2.size());

    REQUIRE(lookGroup->getActiveLook().empty());
    lookGroup->setActiveLook("look1");
    REQUIRE(lookGroup->getActiveLook() == "look1");

    doc->removeLookGroup("lookgroup1");
    lookGroups = doc->getLookGroups();
    REQUIRE(lookGroups.size() == 0);
}
