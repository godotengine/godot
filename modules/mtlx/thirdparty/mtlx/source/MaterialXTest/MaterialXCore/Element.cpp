//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXTest/External/Catch/catch.hpp>

#include <MaterialXCore/Document.h>

namespace mx = MaterialX;

TEST_CASE("Element", "[element]")
{
    // Create a document.
    mx::DocumentPtr doc = mx::createDocument();

    // Create elements
    mx::ElementPtr elem1 = doc->addChildOfCategory("generic");
    mx::ElementPtr elem2 = doc->addChildOfCategory("generic");
    REQUIRE(elem1->getParent() == doc);
    REQUIRE(elem2->getParent() == doc);
    REQUIRE(elem1->getRoot() == doc);
    REQUIRE(elem2->getRoot() == doc);
    REQUIRE(doc->getChildren()[0] == elem1);
    REQUIRE(doc->getChildren()[1] == elem2);

    // Set hierarchical properties
    doc->setFilePrefix("folder/");
    doc->setColorSpace("lin_rec709");
    REQUIRE(elem1->getActiveFilePrefix() == doc->getFilePrefix());
    REQUIRE(elem2->getActiveColorSpace() == doc->getColorSpace());

    // Set typed attributes.
    REQUIRE(elem1->getTypedAttribute<bool>("customFlag") == false);
    REQUIRE(elem1->getTypedAttribute<mx::Color3>("customColor") == mx::Color3(0.0f));
    elem1->setTypedAttribute<bool>("customFlag", true);
    elem1->setTypedAttribute<mx::Color3>("customColor", mx::Color3(1.0f));
    REQUIRE(elem1->getTypedAttribute<bool>("customFlag") == true);
    REQUIRE(elem1->getTypedAttribute<mx::Color3>("customColor") == mx::Color3(1.0f));
    REQUIRE(elem1->getTypedAttribute<bool>("customColor") == false);
    REQUIRE(elem1->getTypedAttribute<mx::Color3>("customFlag") == mx::Color3(0.0f));

    // Modify element names.
    elem1->setName("elem1");
    elem2->setName("elem2");
    REQUIRE(elem1->getName() == "elem1");
    REQUIRE(elem2->getName() == "elem2");
    REQUIRE_THROWS_AS(elem2->setName("elem1"), mx::Exception);

    // Modify element order.
    mx::DocumentPtr doc2 = doc->copy();
    REQUIRE(*doc2 == *doc);
    doc2->setChildIndex("elem1", doc2->getChildIndex("elem2"));
    REQUIRE(*doc2 != *doc);
    doc2->setChildIndex("elem1", doc2->getChildIndex("elem2"));
    REQUIRE(*doc2 == *doc);
    REQUIRE_THROWS_AS(doc2->setChildIndex("elem1", -100), mx::Exception);
    REQUIRE_THROWS_AS(doc2->setChildIndex("elem1", -1), mx::Exception);
    REQUIRE_THROWS_AS(doc2->setChildIndex("elem1", 2), mx::Exception);
    REQUIRE_THROWS_AS(doc2->setChildIndex("elem1", 100), mx::Exception);
    REQUIRE(*doc2 == *doc);

    // Create and test an orphaned element.
    mx::ElementPtr orphan;
    {
        mx::DocumentPtr doc3 = doc->copy();
        orphan = doc3->getChild("elem1");
        REQUIRE(orphan);
    }
    REQUIRE_THROWS_AS(orphan->getDocument(), mx::ExceptionOrphanedElement);
}
