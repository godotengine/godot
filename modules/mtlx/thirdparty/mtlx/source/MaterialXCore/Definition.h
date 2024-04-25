//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_DEFINITION_H
#define MATERIALX_DEFINITION_H

/// @file
/// Definition element subclasses

#include <MaterialXCore/Export.h>

#include <MaterialXCore/Interface.h>

MATERIALX_NAMESPACE_BEGIN

extern MX_CORE_API const string COLOR_SEMANTIC;
extern MX_CORE_API const string SHADER_SEMANTIC;

class NodeDef;
class Implementation;
class TypeDef;
class TargetDef;
class Member;
class Unit;
class UnitDef;
class UnitTypeDef;
class AttributeDef;

/// A shared pointer to a NodeDef
using NodeDefPtr = shared_ptr<NodeDef>;
/// A shared pointer to a const NodeDef
using ConstNodeDefPtr = shared_ptr<const NodeDef>;

/// A shared pointer to an Implementation
using ImplementationPtr = shared_ptr<Implementation>;
/// A shared pointer to a const Implementation
using ConstImplementationPtr = shared_ptr<const Implementation>;

/// A shared pointer to a TypeDef
using TypeDefPtr = shared_ptr<TypeDef>;
/// A shared pointer to a const TypeDef
using ConstTypeDefPtr = shared_ptr<const TypeDef>;

/// A shared pointer to a TargetDef
using TargetDefPtr = shared_ptr<TargetDef>;
/// A shared pointer to a const TargetDef
using ConstTargetDefPtr = shared_ptr<const TargetDef>;

/// A shared pointer to a Member
using MemberPtr = shared_ptr<Member>;
/// A shared pointer to a const Member
using ConstMemberPtr = shared_ptr<const Member>;

/// A shared pointer to a Unit
using UnitPtr = shared_ptr<Unit>;
/// A shared pointer to a const Unit
using ConstUnitPtr = shared_ptr<const Unit>;

/// A shared pointer to a UnitDef
using UnitDefPtr = shared_ptr<UnitDef>;
/// A shared pointer to a const UnitDef
using ConstUnitDefPtr = shared_ptr<const UnitDef>;

/// A shared pointer to a UnitTypeDef
using UnitTypeDefPtr = shared_ptr<UnitTypeDef>;
/// A shared pointer to a const UnitTypeDef
using ConstUnitTypeDefPtr = shared_ptr<const UnitTypeDef>;

/// A shared pointer to an AttributeDef
using AttributeDefPtr = shared_ptr<AttributeDef>;
/// A shared pointer to a const AttributeDef
using AttributeDefDefPtr = shared_ptr<const AttributeDef>;

/// @class NodeDef
/// A node definition element within a Document.
///
/// A NodeDef provides the declaration of a node interface, which may then
/// be instantiated as a Node.
class MX_CORE_API NodeDef : public InterfaceElement
{
  public:
    NodeDef(ElementPtr parent, const string& name) :
        InterfaceElement(parent, CATEGORY, name)
    {
    }
    virtual ~NodeDef() { }

    /// @name Node String
    /// @{

    /// Set the node string of the NodeDef.
    void setNodeString(const string& node)
    {
        setAttribute(NODE_ATTRIBUTE, node);
    }

    /// Return true if the given NodeDef has a node string.
    bool hasNodeString() const
    {
        return hasAttribute(NODE_ATTRIBUTE);
    }

    /// Return the node string of the NodeDef.
    const string& getNodeString() const
    {
        return getAttribute(NODE_ATTRIBUTE);
    }

    /// Return the element's output type.
    const string& getType() const override;

    /// @}
    /// @name Node Group
    /// @{

    /// Set the node group of the NodeDef.
    void setNodeGroup(const string& category)
    {
        setAttribute(NODE_GROUP_ATTRIBUTE, category);
    }

    /// Return true if the given NodeDef has a node group.
    bool hasNodeGroup() const
    {
        return hasAttribute(NODE_GROUP_ATTRIBUTE);
    }

    /// Return the node group of the NodeDef.
    const string& getNodeGroup() const
    {
        return getAttribute(NODE_GROUP_ATTRIBUTE);
    }

    /// @}
    /// @name Implementation References
    /// @{

    /// Return the first implementation for this nodedef, optionally filtered
    /// by the given target name.
    /// @param target An optional target name, which will be used to filter
    ///    the implementations that are considered.
    /// @return An implementation for this nodedef, or an empty shared pointer
    ///    if none was found.  Note that a node implementation may be either
    ///    an Implementation element or a NodeGraph element.
    InterfaceElementPtr getImplementation(const string& target = EMPTY_STRING) const;

    /// @}
    /// @name Validation
    /// @{

    /// Validate that the given element tree, including all descendants, is
    /// consistent with the MaterialX specification.
    bool validate(string* message = nullptr) const override;

    /// @}
    /// @name Utility
    /// @{

    /// Return true if the given version string is compatible with this
    /// NodeDef.  This may be used to test, for example, whether a NodeDef
    /// and Node may be used together.
    bool isVersionCompatible(const string& version) const;

    /// Return the first declaration of this interface, optionally filtered
    ///    by the given target name.
    ConstInterfaceElementPtr getDeclaration(const string& target = EMPTY_STRING) const override;

    /// @}

  public:
    static const string CATEGORY;
    static const string NODE_ATTRIBUTE;
    static const string NODE_GROUP_ATTRIBUTE;

    static const string TEXTURE_NODE_GROUP;
    static const string PROCEDURAL_NODE_GROUP;
    static const string GEOMETRIC_NODE_GROUP;
    static const string ADJUSTMENT_NODE_GROUP;
    static const string CONDITIONAL_NODE_GROUP;
    static const string ORGANIZATION_NODE_GROUP;
    static const string TRANSLATION_NODE_GROUP;
};

/// @class Implementation
/// An implementation element within a Document.
///
/// An Implementation is used to associate external source code with a specific
/// NodeDef, providing a definition for the node that may either be universal or
/// restricted to a specific target.
class MX_CORE_API Implementation : public InterfaceElement
{
  public:
    Implementation(ElementPtr parent, const string& name) :
        InterfaceElement(parent, CATEGORY, name)
    {
    }
    virtual ~Implementation() { }

    /// @name File String
    /// @{

    /// Set the file string for the Implementation.
    void setFile(const string& file)
    {
        setAttribute(FILE_ATTRIBUTE, file);
    }

    /// Return true if the given Implementation has a file string.
    bool hasFile() const
    {
        return hasAttribute(FILE_ATTRIBUTE);
    }

    /// Return the file string for the Implementation.
    const string& getFile() const
    {
        return getAttribute(FILE_ATTRIBUTE);
    }

    /// @}
    /// @name Function String
    /// @{

    /// Set the function string for the Implementation.
    void setFunction(const string& function)
    {
        setAttribute(FUNCTION_ATTRIBUTE, function);
    }

    /// Return true if the given Implementation has a function string.
    bool hasFunction() const
    {
        return hasAttribute(FUNCTION_ATTRIBUTE);
    }

    /// Return the function string for the Implementation.
    const string& getFunction() const
    {
        return getAttribute(FUNCTION_ATTRIBUTE);
    }

    /// @}
    /// @name Nodegraph String
    /// @{

    /// Set the nodegraph string for the Implementation.
    void setNodeGraph(const string& nodegraph)
    {
        setAttribute(NODE_GRAPH_ATTRIBUTE, nodegraph);
    }

    /// Return true if the given Implementation has a nodegraph string.
    bool hasNodeGraph() const
    {
        return hasAttribute(NODE_GRAPH_ATTRIBUTE);
    }

    /// Return the nodegraph string for the Implementation.
    const string& getNodeGraph() const
    {
        return getAttribute(PortElement::NODE_GRAPH_ATTRIBUTE);
    }

    /// @}
    /// @name NodeDef References
    /// @{

    /// Set the NodeDef element referenced by the Implementation.
    void setNodeDef(ConstNodeDefPtr nodeDef);

    /// Return the NodeDef element referenced by the Implementation.
    NodeDefPtr getNodeDef() const;

    /// @}
    /// @name Validation
    /// @{

    /// Validate that the given element tree, including all descendants, is
    /// consistent with the MaterialX specification.
    bool validate(string* message = nullptr) const override;

    /// @}
    /// @name Utility
    /// @{

    /// Return the first declaration of this interface, optionally filtered
    ///    by the given target name.
    ConstInterfaceElementPtr getDeclaration(const string& target = EMPTY_STRING) const override;

    /// @}

  public:
    static const string CATEGORY;
    static const string FILE_ATTRIBUTE;
    static const string FUNCTION_ATTRIBUTE;
    static const string NODE_GRAPH_ATTRIBUTE;
};

/// @class TypeDef
/// A type definition element within a Document.
class MX_CORE_API TypeDef : public Element
{
  public:
    TypeDef(ElementPtr parent, const string& name) :
        Element(parent, CATEGORY, name)
    {
    }
    virtual ~TypeDef() { }

    /// @name Semantic
    /// @{

    /// Set the semantic string of the TypeDef.
    void setSemantic(const string& semantic)
    {
        setAttribute(SEMANTIC_ATTRIBUTE, semantic);
    }

    /// Return true if the given TypeDef has a semantic string.
    bool hasSemantic() const
    {
        return hasAttribute(SEMANTIC_ATTRIBUTE);
    }

    /// Return the semantic string of the TypeDef.
    const string& getSemantic() const
    {
        return getAttribute(SEMANTIC_ATTRIBUTE);
    }

    /// @}
    /// @name Context
    /// @{

    /// Set the context string of the TypeDef.
    void setContext(const string& context)
    {
        setAttribute(CONTEXT_ATTRIBUTE, context);
    }

    /// Return true if the given TypeDef has a context string.
    bool hasContext() const
    {
        return hasAttribute(CONTEXT_ATTRIBUTE);
    }

    /// Return the context string of the TypeDef.
    const string& getContext() const
    {
        return getAttribute(CONTEXT_ATTRIBUTE);
    }

    /// @}
    /// @name Member Elements
    /// @{

    /// Add a Member to the TypeDef.
    /// @param name The name of the new Member.
    ///     If no name is specified, then a unique name will automatically be
    ///     generated.
    /// @return A shared pointer to the new Member.
    MemberPtr addMember(const string& name = EMPTY_STRING)
    {
        return addChild<Member>(name);
    }

    /// Return the Member, if any, with the given name.
    MemberPtr getMember(const string& name) const
    {
        return getChildOfType<Member>(name);
    }

    /// Return a vector of all Member elements in the TypeDef.
    vector<MemberPtr> getMembers() const
    {
        return getChildrenOfType<Member>();
    }

    /// Remove the Member, if any, with the given name.
    void removeMember(const string& name)
    {
        removeChildOfType<Member>(name);
    }

    /// @}

  public:
    static const string CATEGORY;
    static const string SEMANTIC_ATTRIBUTE;
    static const string CONTEXT_ATTRIBUTE;
};

/// @class TargetDef
/// A definition of an implementation target.
class MX_CORE_API TargetDef : public TypedElement
{
  public:
    TargetDef(ElementPtr parent, const string& name) :
        TypedElement(parent, CATEGORY, name)
    {
    }
    virtual ~TargetDef() { }

    /// Return a vector of target names that is matching this targetdef
    /// either by itself of by its inheritance.
    /// The vector is ordered by priority starting with this targetdef
    /// itself and then upwards in the inheritance hierarchy.
    StringVec getMatchingTargets() const;

  public:
    static const string CATEGORY;
};

/// @class Member
/// A member element within a TypeDef.
class MX_CORE_API Member : public TypedElement
{
  public:
    Member(ElementPtr parent, const string& name) :
        TypedElement(parent, CATEGORY, name)
    {
    }
    virtual ~Member() { }

  public:
    static const string CATEGORY;
};

/// @class Unit
/// A unit declaration within a UnitDef.
class MX_CORE_API Unit : public Element
{
  public:
    Unit(ElementPtr parent, const string& name) :
        Element(parent, CATEGORY, name)
    {
    }
    virtual ~Unit() { }

  public:
    static const string CATEGORY;
};

/// @class UnitDef
/// A unit definition element within a Document.
class MX_CORE_API UnitDef : public Element
{
  public:
    UnitDef(ElementPtr parent, const string& name) :
        Element(parent, CATEGORY, name)
    {
    }
    virtual ~UnitDef() { }

    /// @name Unit Type methods
    /// @{

    /// Set the element's unittype string.
    void setUnitType(const string& type)
    {
        setAttribute(UNITTYPE_ATTRIBUTE, type);
    }

    /// Return true if the given element has a unittype string.
    bool hasUnitType() const
    {
        return hasAttribute(UNITTYPE_ATTRIBUTE);
    }

    /// Return the element's type string.
    const string& getUnitType() const
    {
        return getAttribute(UNITTYPE_ATTRIBUTE);
    }

    /// @}
    /// @name Unit methods
    /// @{

    /// Add a Unit to the UnitDef.
    /// @param name The name of the new Unit. An exception is thrown
    /// if the name provided is an empty string.
    /// @return A shared pointer to the new Unit.
    UnitPtr addUnit(const string& name)
    {
        if (name.empty())
        {
            throw Exception("A unit definition name cannot be empty");
        }
        return addChild<Unit>(name);
    }

    /// Return the Unit, if any, with the given name.
    UnitPtr getUnit(const string& name) const
    {
        return getChildOfType<Unit>(name);
    }

    /// Return a vector of all Unit elements in the UnitDef.
    vector<UnitPtr> getUnits() const
    {
        return getChildrenOfType<Unit>();
    }

    /// Remove the Unit, if any, with the given name.
    void removeUnit(const string& name)
    {
        removeChildOfType<Unit>(name);
    }

    /// @}

  public:
    static const string CATEGORY;
    static const string UNITTYPE_ATTRIBUTE;
};

/// @class UnitTypeDef
/// A unit type definition element within a Document.
class MX_CORE_API UnitTypeDef : public Element
{
  public:
    UnitTypeDef(ElementPtr parent, const string& name) :
        Element(parent, CATEGORY, name)
    {
    }
    virtual ~UnitTypeDef() { }

    /// Find all UnitDefs for the UnitTypeDef
    vector<UnitDefPtr> getUnitDefs() const;

  public:
    static const string CATEGORY;
};

/// @class AttributeDef
/// An attribute definition element within a Document.
class MX_CORE_API AttributeDef : public TypedElement
{
  public:
    AttributeDef(ElementPtr parent, const string& name) :
        TypedElement(parent, CATEGORY, name)
    {
    }
    virtual ~AttributeDef() { }

    /// @name Attribute name
    /// @{

    /// Set the element's attrname string.
    void setAttrName(const string& name)
    {
        setAttribute(ATTRNAME_ATTRIBUTE, name);
    }

    /// Return true if this element has an attrname string.
    bool hasAttrName() const
    {
        return hasAttribute(ATTRNAME_ATTRIBUTE);
    }

    /// Return the element's attrname string.
    const string& getAttrName() const
    {
        return getAttribute(ATTRNAME_ATTRIBUTE);
    }

    /// @}
    /// @name Value String
    /// @{

    /// Set the value string of an element.
    void setValueString(const string& value)
    {
        setAttribute(VALUE_ATTRIBUTE, value);
    }

    /// Return true if the given element has a value string.
    bool hasValueString() const
    {
        return hasAttribute(VALUE_ATTRIBUTE);
    }

    /// Get the value string of a element.
    const string& getValueString() const
    {
        return getAttribute(VALUE_ATTRIBUTE);
    }

    /// @}
    /// @name Typed Value
    /// @{

    /// Set the typed value of an element.
    template <class T> void setValue(const T& value, const string& type = EMPTY_STRING)
    {
        setType(!type.empty() ? type : getTypeString<T>());
        setValueString(toValueString(value));
    }

    /// Set the typed value of an element from a C-style string.
    void setValue(const char* value, const string& type = EMPTY_STRING)
    {
        setValue(value ? string(value) : EMPTY_STRING, type);
    }

    /// Return true if the element possesses a typed value.
    bool hasValue() const
    {
        return hasAttribute(VALUE_ATTRIBUTE);
    }

    /// Return the typed value of an element as a generic value object, which
    /// may be queried to access its data.
    ///
    /// @return A shared pointer to the typed value of this element, or an
    ///    empty shared pointer if no value is present.
    ValuePtr getValue() const
    {
        if (!hasValue())
            return ValuePtr();
        return Value::createValueFromStrings(getValueString(), getType());
    }

    /// @}
    /// @name Elements
    /// @{

    /// Set the element's elements string.
    void setElements(const string& elements)
    {
        setAttribute(ELEMENTS_ATTRIBUTE, elements);
    }

    /// Return true if the element has an elements string.
    bool hasElements() const
    {
        return hasAttribute(ELEMENTS_ATTRIBUTE);
    }

    /// Return the element's elements string.
    const string& getElements() const
    {
        return getAttribute(ELEMENTS_ATTRIBUTE);
    }

    /// @}
    /// @name Exportable
    /// @{

    /// Set the exportable boolean for the element.
    void setExportable(bool value)
    {
        setTypedAttribute<bool>(EXPORTABLE_ATTRIBUTE, value);
    }

    /// Return the exportable boolean for the element.
    /// Defaults to false if exportable is not set.
    bool getExportable() const
    {
        return getTypedAttribute<bool>(EXPORTABLE_ATTRIBUTE);
    }

    /// @}

  public:
    static const string CATEGORY;
    static const string ATTRNAME_ATTRIBUTE;
    static const string VALUE_ATTRIBUTE;
    static const string ELEMENTS_ATTRIBUTE;
    static const string EXPORTABLE_ATTRIBUTE;
};

MATERIALX_NAMESPACE_END

#endif
