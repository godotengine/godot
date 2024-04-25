//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_DOCUMENT
#define MATERIALX_DOCUMENT

/// @file
/// The top-level Document class

#include <MaterialXCore/Export.h>

#include <MaterialXCore/Look.h>
#include <MaterialXCore/Node.h>

MATERIALX_NAMESPACE_BEGIN

class Document;

/// A shared pointer to a Document
using DocumentPtr = shared_ptr<Document>;
/// A shared pointer to a const Document
using ConstDocumentPtr = shared_ptr<const Document>;

/// @class Document
/// A MaterialX document, which represents the top-level element in the
/// MaterialX ownership hierarchy.
///
/// Use the factory function createDocument() to create a Document instance.
class MX_CORE_API Document : public GraphElement
{
  public:
    Document(ElementPtr parent, const string& name);
    virtual ~Document();

    /// Create a new document of the given subclass.
    template <class T> static shared_ptr<T> createDocument()
    {
        shared_ptr<T> doc = std::make_shared<T>(ElementPtr(), EMPTY_STRING);
        doc->initialize();
        return doc;
    }

    /// Initialize the document, removing any existing content.
    virtual void initialize();

    /// Create a deep copy of the document.
    virtual DocumentPtr copy() const
    {
        DocumentPtr doc = createDocument<Document>();
        doc->copyContentFrom(getSelf());
        return doc;
    }

    /// Import the given document as a library within this document.
    /// The contents of the library document are copied into this one, and
    /// are assigned the source URI of the library.
    /// @param library The library document to be imported.
    void importLibrary(const ConstDocumentPtr& library);

    /// Get a list of source URI's referenced by the document
    StringSet getReferencedSourceUris() const;

    /// @name NodeGraph Elements
    /// @{

    /// Add a NodeGraph to the document.
    /// @param name The name of the new NodeGraph.
    ///     If no name is specified, then a unique name will automatically be
    ///     generated.
    /// @return A shared pointer to the new NodeGraph.
    NodeGraphPtr addNodeGraph(const string& name = EMPTY_STRING)
    {
        return addChild<NodeGraph>(name);
    }

    /// Return the NodeGraph, if any, with the given name.
    NodeGraphPtr getNodeGraph(const string& name) const
    {
        return getChildOfType<NodeGraph>(name);
    }

    /// Return a vector of all NodeGraph elements in the document.
    vector<NodeGraphPtr> getNodeGraphs() const
    {
        return getChildrenOfType<NodeGraph>();
    }

    /// Remove the NodeGraph, if any, with the given name.
    void removeNodeGraph(const string& name)
    {
        removeChildOfType<NodeGraph>(name);
    }

    /// Return a vector of all port elements that match the given node name.
    /// Port elements support spatially-varying upstream connections to
    /// nodes, and include both Input and Output elements.
    vector<PortElementPtr> getMatchingPorts(const string& nodeName) const;

    /// @}
    /// @name GeomInfo Elements
    /// @{

    /// Add a GeomInfo to the document.
    /// @param name The name of the new GeomInfo.
    ///     If no name is specified, then a unique name will automatically be
    ///     generated.
    /// @param geom An optional geometry string for the GeomInfo.
    /// @return A shared pointer to the new GeomInfo.
    GeomInfoPtr addGeomInfo(const string& name = EMPTY_STRING, const string& geom = UNIVERSAL_GEOM_NAME)
    {
        GeomInfoPtr geomInfo = addChild<GeomInfo>(name);
        geomInfo->setGeom(geom);
        return geomInfo;
    }

    /// Return the GeomInfo, if any, with the given name.
    GeomInfoPtr getGeomInfo(const string& name) const
    {
        return getChildOfType<GeomInfo>(name);
    }

    /// Return a vector of all GeomInfo elements in the document.
    vector<GeomInfoPtr> getGeomInfos() const
    {
        return getChildrenOfType<GeomInfo>();
    }

    /// Remove the GeomInfo, if any, with the given name.
    void removeGeomInfo(const string& name)
    {
        removeChildOfType<GeomInfo>(name);
    }

    /// Return the value of a geometric property for the given geometry string.
    ValuePtr getGeomPropValue(const string& geomPropName, const string& geom = UNIVERSAL_GEOM_NAME) const;

    /// @}
    /// @name GeomPropDef Elements
    /// @{

    /// Add a GeomPropDef to the document.
    /// @param name The name of the new GeomPropDef.
    /// @param geomprop The geometric property to use for the GeomPropDef.
    /// @return A shared pointer to the new GeomPropDef.
    GeomPropDefPtr addGeomPropDef(const string& name, const string& geomprop)
    {
        GeomPropDefPtr geomPropDef = addChild<GeomPropDef>(name);
        geomPropDef->setGeomProp(geomprop);
        return geomPropDef;
    }

    /// Return the GeomPropDef, if any, with the given name.
    GeomPropDefPtr getGeomPropDef(const string& name) const
    {
        return getChildOfType<GeomPropDef>(name);
    }

    /// Return a vector of all GeomPropDef elements in the document.
    vector<GeomPropDefPtr> getGeomPropDefs() const
    {
        return getChildrenOfType<GeomPropDef>();
    }

    /// Remove the GeomPropDef, if any, with the given name.
    void removeGeomPropDef(const string& name)
    {
        removeChildOfType<GeomPropDef>(name);
    }

    /// @}
    /// @name Material Outputs
    /// @{

    /// Return material-type outputs for all nodegraphs in the document.
    vector<OutputPtr> getMaterialOutputs() const;

    /// @}
    /// @name Look Elements
    /// @{

    /// Add a Look to the document.
    /// @param name The name of the new Look.
    ///     If no name is specified, then a unique name will automatically be
    ///     generated.
    /// @return A shared pointer to the new Look.
    LookPtr addLook(const string& name = EMPTY_STRING)
    {
        return addChild<Look>(name);
    }

    /// Return the Look, if any, with the given name.
    LookPtr getLook(const string& name) const
    {
        return getChildOfType<Look>(name);
    }

    /// Return a vector of all Look elements in the document.
    vector<LookPtr> getLooks() const
    {
        return getChildrenOfType<Look>();
    }

    /// Remove the Look, if any, with the given name.
    void removeLook(const string& name)
    {
        removeChildOfType<Look>(name);
    }

    /// @}
    /// @name LookGroup Elements
    /// @{

    /// Add a LookGroup to the document.
    /// @param name The name of the new LookGroup.
    ///     If no name is specified, then a unique name will automatically be
    ///     generated.
    /// @return A shared pointer to the new LookGroup.
    LookGroupPtr addLookGroup(const string& name = EMPTY_STRING)
    {
        return addChild<LookGroup>(name);
    }

    /// Return the LookGroup, if any, with the given name.
    LookGroupPtr getLookGroup(const string& name) const
    {
        return getChildOfType<LookGroup>(name);
    }

    /// Return a vector of all LookGroup elements in the document.
    vector<LookGroupPtr> getLookGroups() const
    {
        return getChildrenOfType<LookGroup>();
    }

    /// Remove the LookGroup, if any, with the given name.
    void removeLookGroup(const string& name)
    {
        removeChildOfType<LookGroup>(name);
    }

    /// @}
    /// @name Collection Elements
    /// @{

    /// Add a Collection to the document.
    /// @param name The name of the new Collection.
    ///     If no name is specified, then a unique name will automatically be
    ///     generated.
    /// @return A shared pointer to the new Collection.
    CollectionPtr addCollection(const string& name = EMPTY_STRING)
    {
        return addChild<Collection>(name);
    }

    /// Return the Collection, if any, with the given name.
    CollectionPtr getCollection(const string& name) const
    {
        return getChildOfType<Collection>(name);
    }

    /// Return a vector of all Collection elements in the document.
    vector<CollectionPtr> getCollections() const
    {
        return getChildrenOfType<Collection>();
    }

    /// Remove the Collection, if any, with the given name.
    void removeCollection(const string& name)
    {
        removeChildOfType<Collection>(name);
    }

    /// @}
    /// @name TypeDef Elements
    /// @{

    /// Add a TypeDef to the document.
    /// @param name The name of the new TypeDef.
    ///     If no name is specified, then a unique name will automatically be
    ///     generated.
    /// @return A shared pointer to the new TypeDef.
    TypeDefPtr addTypeDef(const string& name)
    {
        return addChild<TypeDef>(name);
    }

    /// Return the TypeDef, if any, with the given name.
    TypeDefPtr getTypeDef(const string& name) const
    {
        return getChildOfType<TypeDef>(name);
    }

    /// Return a vector of all TypeDef elements in the document.
    vector<TypeDefPtr> getTypeDefs() const
    {
        return getChildrenOfType<TypeDef>();
    }

    /// Remove the TypeDef, if any, with the given name.
    void removeTypeDef(const string& name)
    {
        removeChildOfType<TypeDef>(name);
    }

    /// @}
    /// @name NodeDef Elements
    /// @{

    /// Add a NodeDef to the document.
    /// @param name The name of the new NodeDef.
    ///     If no name is specified, then a unique name will automatically be
    ///     generated.
    /// @param type An optional type string.
    ///     If specified, then the new NodeDef will be assigned an Output of
    ///     the given type.
    /// @param node An optional node string.
    /// @return A shared pointer to the new NodeDef.
    NodeDefPtr addNodeDef(const string& name = EMPTY_STRING,
                          const string& type = DEFAULT_TYPE_STRING,
                          const string& node = EMPTY_STRING)
    {
        NodeDefPtr child = addChild<NodeDef>(name);
        if (!type.empty() && type != MULTI_OUTPUT_TYPE_STRING)
        {
            child->addOutput("out", type);
        }
        if (!node.empty())
        {
            child->setNodeString(node);
        }
        return child;
    }

    /// Create a NodeDef declaration which is based on a NodeGraph.
    /// @param nodeGraph NodeGraph used to create NodeDef
    /// @param nodeDefName Declaration name
    /// @param node Node type for the new declaration
    /// @param version Version for the new declaration
    /// @param isDefaultVersion If a version is specified is thie definition the default version
    /// @param newGraphName Make a copy of this NodeGraph with the given name if a non-empty name is provided. Otherwise
    ///        modify the existing NodeGraph. Default value is an empty string.
    /// @param nodeGroup Optional node group for the new declaration. The Default value is an emptry string.
    /// @return New declaration if successful.
    NodeDefPtr addNodeDefFromGraph(const NodeGraphPtr nodeGraph, const string& nodeDefName, const string& node, const string& version,
                                   bool isDefaultVersion, const string& nodeGroup, const string& newGraphName);

    /// Return the NodeDef, if any, with the given name.
    NodeDefPtr getNodeDef(const string& name) const
    {
        return getChildOfType<NodeDef>(name);
    }

    /// Return a vector of all NodeDef elements in the document.
    vector<NodeDefPtr> getNodeDefs() const
    {
        return getChildrenOfType<NodeDef>();
    }

    /// Remove the NodeDef, if any, with the given name.
    void removeNodeDef(const string& name)
    {
        removeChildOfType<NodeDef>(name);
    }

    /// Return a vector of all NodeDef elements that match the given node name.
    vector<NodeDefPtr> getMatchingNodeDefs(const string& nodeName) const;

    /// @}
    /// @name AttributeDef Elements
    /// @{

    /// Add an AttributeDef to the document.
    /// @param name The name of the new AttributeDef.
    ///     If no name is specified, then a unique name will automatically be
    ///     generated.
    /// @return A shared pointer to the new AttributeDef.
    AttributeDefPtr addAttributeDef(const string& name = EMPTY_STRING)
    {
        return addChild<AttributeDef>(name);
    }

    /// Return the AttributeDef, if any, with the given name.
    AttributeDefPtr getAttributeDef(const string& name) const
    {
        return getChildOfType<AttributeDef>(name);
    }

    /// Return a vector of all AttributeDef elements in the document.
    vector<AttributeDefPtr> getAttributeDefs() const
    {
        return getChildrenOfType<AttributeDef>();
    }

    /// Remove the AttributeDef, if any, with the given name.
    void removeAttributeDef(const string& name)
    {
        removeChildOfType<AttributeDef>(name);
    }

    /// @}
    /// @name TargetDef Elements
    /// @{

    /// Add an TargetDef to the document.
    /// @param name The name of the new TargetDef.
    ///     If no name is specified, then a unique name will automatically be
    ///     generated.
    /// @return A shared pointer to the new TargetDef.
    TargetDefPtr addTargetDef(const string& name = EMPTY_STRING)
    {
        return addChild<TargetDef>(name);
    }

    /// Return the AttributeDef, if any, with the given name.
    TargetDefPtr getTargetDef(const string& name) const
    {
        return getChildOfType<TargetDef>(name);
    }

    /// Return a vector of all TargetDef elements in the document.
    vector<TargetDefPtr> getTargetDefs() const
    {
        return getChildrenOfType<TargetDef>();
    }

    /// Remove the TargetDef, if any, with the given name.
    void removeTargetDef(const string& name)
    {
        removeChildOfType<TargetDef>(name);
    }

    /// @}
    /// @name PropertySet Elements
    /// @{

    /// Add a PropertySet to the document.
    /// @param name The name of the new PropertySet.
    ///     If no name is specified, then a unique name will automatically be
    ///     generated.
    /// @return A shared pointer to the new PropertySet.
    PropertySetPtr addPropertySet(const string& name = EMPTY_STRING)
    {
        return addChild<PropertySet>(name);
    }

    /// Return the PropertySet, if any, with the given name.
    PropertySetPtr getPropertySet(const string& name) const
    {
        return getChildOfType<PropertySet>(name);
    }

    /// Return a vector of all PropertySet elements in the document.
    vector<PropertySetPtr> getPropertySets() const
    {
        return getChildrenOfType<PropertySet>();
    }

    /// Remove the PropertySet, if any, with the given name.
    void removePropertySet(const string& name)
    {
        removeChildOfType<PropertySet>(name);
    }

    /// @}
    /// @name VariantSet Elements
    /// @{

    /// Add a VariantSet to the document.
    /// @param name The name of the new VariantSet.
    ///     If no name is specified, then a unique name will automatically be
    ///     generated.
    /// @return A shared pointer to the new VariantSet.
    VariantSetPtr addVariantSet(const string& name = EMPTY_STRING)
    {
        return addChild<VariantSet>(name);
    }

    /// Return the VariantSet, if any, with the given name.
    VariantSetPtr getVariantSet(const string& name) const
    {
        return getChildOfType<VariantSet>(name);
    }

    /// Return a vector of all VariantSet elements in the document.
    vector<VariantSetPtr> getVariantSets() const
    {
        return getChildrenOfType<VariantSet>();
    }

    /// Remove the VariantSet, if any, with the given name.
    void removeVariantSet(const string& name)
    {
        removeChildOfType<VariantSet>(name);
    }

    /// @}
    /// @name Implementation Elements
    /// @{

    /// Add an Implementation to the document.
    /// @param name The name of the new Implementation.
    ///     If no name is specified, then a unique name will automatically be
    ///     generated.
    /// @return A shared pointer to the new Implementation.
    ImplementationPtr addImplementation(const string& name = EMPTY_STRING)
    {
        return addChild<Implementation>(name);
    }

    /// Return the Implementation, if any, with the given name.
    ImplementationPtr getImplementation(const string& name) const
    {
        return getChildOfType<Implementation>(name);
    }

    /// Return a vector of all Implementation elements in the document.
    vector<ImplementationPtr> getImplementations() const
    {
        return getChildrenOfType<Implementation>();
    }

    /// Remove the Implementation, if any, with the given name.
    void removeImplementation(const string& name)
    {
        removeChildOfType<Implementation>(name);
    }

    /// Return a vector of all node implementations that match the given
    /// NodeDef string.  Note that a node implementation may be either an
    /// Implementation element or NodeGraph element.
    vector<InterfaceElementPtr> getMatchingImplementations(const string& nodeDef) const;

    /// @}
    /// @name UnitDef Elements
    /// @{

    UnitDefPtr addUnitDef(const string& name)
    {
        if (name.empty())
        {
            throw Exception("A unit definition name cannot be empty");
        }
        return addChild<UnitDef>(name);
    }

    /// Return the UnitDef, if any, with the given name.
    UnitDefPtr getUnitDef(const string& name) const
    {
        return getChildOfType<UnitDef>(name);
    }

    /// Return a vector of all Member elements in the TypeDef.
    vector<UnitDefPtr> getUnitDefs() const
    {
        return getChildrenOfType<UnitDef>();
    }

    /// Remove the UnitDef, if any, with the given name.
    void removeUnitDef(const string& name)
    {
        removeChildOfType<UnitDef>(name);
    }

    /// @}
    /// @name UnitTypeDef Elements
    /// @{

    UnitTypeDefPtr addUnitTypeDef(const string& name)
    {
        if (name.empty())
        {
            throw Exception("A unit type definition name cannot be empty");
        }
        return addChild<UnitTypeDef>(name);
    }

    /// Return the UnitTypeDef, if any, with the given name.
    UnitTypeDefPtr getUnitTypeDef(const string& name) const
    {
        return getChildOfType<UnitTypeDef>(name);
    }

    /// Return a vector of all UnitTypeDef elements in the document.
    vector<UnitTypeDefPtr> getUnitTypeDefs() const
    {
        return getChildrenOfType<UnitTypeDef>();
    }

    /// Remove the UnitTypeDef, if any, with the given name.
    void removeUnitTypeDef(const string& name)
    {
        removeChildOfType<UnitTypeDef>(name);
    }

    /// @}
    /// @name Version
    /// @{

    /// Return the major and minor versions as an integer pair.
    std::pair<int, int> getVersionIntegers() const override;

    /// Upgrade the content of this document from earlier supported versions to
    /// the library version.
    void upgradeVersion();

    /// @}
    /// @name Color Management System
    /// @{

    /// Set the color management system string.
    void setColorManagementSystem(const string& cms)
    {
        setAttribute(CMS_ATTRIBUTE, cms);
    }

    /// Return true if a color management system string has been set.
    bool hasColorManagementSystem() const
    {
        return hasAttribute(CMS_ATTRIBUTE);
    }

    /// Return the color management system string.
    const string& getColorManagementSystem() const
    {
        return getAttribute(CMS_ATTRIBUTE);
    }

    /// @}
    /// @name Color Management Config
    /// @{

    /// Set the color management config string.
    void setColorManagementConfig(const string& cmsConfig)
    {
        setAttribute(CMS_CONFIG_ATTRIBUTE, cmsConfig);
    }

    /// Return true if a color management config string has been set.
    bool hasColorManagementConfig() const
    {
        return hasAttribute(CMS_CONFIG_ATTRIBUTE);
    }

    /// Return the color management config string.
    const string& getColorManagementConfig() const
    {
        return getAttribute(CMS_CONFIG_ATTRIBUTE);
    }

    /// @}
    /// @name Validation
    /// @{

    /// Validate that the given document is consistent with the MaterialX
    /// specification.
    /// @param message An optional output string, to which a description of
    ///    each error will be appended.
    /// @return True if the document passes all tests, false otherwise.
    bool validate(string* message = nullptr) const override;

    /// @}
    /// @name Utility
    /// @{

    /// Invalidate cached data for optimized lookups within the given document.
    void invalidateCache();

    /// @}

  public:
    static const string CATEGORY;
    static const string CMS_ATTRIBUTE;
    static const string CMS_CONFIG_ATTRIBUTE;

  private:
    class Cache;
    std::unique_ptr<Cache> _cache;
};

/// Create a new Document.
/// @relates Document
MX_CORE_API DocumentPtr createDocument();

MATERIALX_NAMESPACE_END

#endif
