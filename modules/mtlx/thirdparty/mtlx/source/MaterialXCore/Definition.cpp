//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXCore/Definition.h>

#include <MaterialXCore/Document.h>

MATERIALX_NAMESPACE_BEGIN

const string COLOR_SEMANTIC = "color";
const string SHADER_SEMANTIC = "shader";

const string NodeDef::TEXTURE_NODE_GROUP = "texture";
const string NodeDef::PROCEDURAL_NODE_GROUP = "procedural";
const string NodeDef::GEOMETRIC_NODE_GROUP = "geometric";
const string NodeDef::ADJUSTMENT_NODE_GROUP = "adjustment";
const string NodeDef::CONDITIONAL_NODE_GROUP = "conditional";
const string NodeDef::ORGANIZATION_NODE_GROUP = "organization";
const string NodeDef::TRANSLATION_NODE_GROUP = "translation";

const string NodeDef::NODE_ATTRIBUTE = "node";
const string NodeDef::NODE_GROUP_ATTRIBUTE = "nodegroup";
const string TypeDef::SEMANTIC_ATTRIBUTE = "semantic";
const string TypeDef::CONTEXT_ATTRIBUTE = "context";
const string Implementation::FILE_ATTRIBUTE = "file";
const string Implementation::FUNCTION_ATTRIBUTE = "function";
const string Implementation::NODE_GRAPH_ATTRIBUTE = "nodegraph";
const string UnitDef::UNITTYPE_ATTRIBUTE = "unittype";
const string AttributeDef::ATTRNAME_ATTRIBUTE = "attrname";
const string AttributeDef::VALUE_ATTRIBUTE = "value";
const string AttributeDef::ELEMENTS_ATTRIBUTE = "elements";
const string AttributeDef::EXPORTABLE_ATTRIBUTE = "exportable";

//
// NodeDef methods
//

const string& NodeDef::getType() const
{
    const vector<OutputPtr>& activeOutputs = getActiveOutputs();

    size_t numActiveOutputs = activeOutputs.size();
    if (numActiveOutputs > 1)
    {
        return MULTI_OUTPUT_TYPE_STRING;
    }
    else if (numActiveOutputs == 1)
    {
        return activeOutputs[0]->getType();
    }
    else
    {
        return DEFAULT_TYPE_STRING;
    }
}

InterfaceElementPtr NodeDef::getImplementation(const string& target) const
{
    vector<InterfaceElementPtr> interfaces = getDocument()->getMatchingImplementations(getQualifiedName(getName()));
    vector<InterfaceElementPtr> secondary = getDocument()->getMatchingImplementations(getName());
    interfaces.insert(interfaces.end(), secondary.begin(), secondary.end());

    if (target.empty())
    {
        return !interfaces.empty() ? interfaces[0] : InterfaceElementPtr();
    }

    // Get all candidate targets matching the given target,
    // taking inheritance into account.
    const TargetDefPtr targetDef = getDocument()->getTargetDef(target);
    const StringVec candidateTargets = targetDef ? targetDef->getMatchingTargets() : StringVec();

    // First, search for a target-specific match.
    for (const string& candidateTarget : candidateTargets)
    {
        for (InterfaceElementPtr interface : interfaces)
        {
            const std::string& interfaceTarget = interface->getTarget();
            if (!interfaceTarget.empty() && targetStringsMatch(interfaceTarget, candidateTarget))
            {
                return interface;
            }
        }
    }

    // Then search for a generic match.
    for (InterfaceElementPtr interface : interfaces)
    {
        // Look for interfaces without targets
        const std::string& interfaceTarget = interface->getTarget();
        if (interfaceTarget.empty())
        {
            return interface;
        }
    }

    return InterfaceElementPtr();
}

bool NodeDef::validate(string* message) const
{
    bool res = true;
    validateRequire(!hasType(), res, message, "Nodedef should not have a type but an explicit output");
    return InterfaceElement::validate(message) && res;
}

bool NodeDef::isVersionCompatible(const string& version) const
{
    if (getVersionString() == version)
    {
        return true;
    }
    if (getDefaultVersion() && version.empty())
    {
        return true;
    }
    return false;
}

ConstInterfaceElementPtr NodeDef::getDeclaration(const string&) const
{
    return getSelf()->asA<InterfaceElement>();
}

//
// Implementation methods
//

void Implementation::setNodeDef(ConstNodeDefPtr nodeDef)
{
    if (nodeDef)
    {
        setNodeDefString(nodeDef->getName());
    }
    else
    {
        removeAttribute(NODE_DEF_ATTRIBUTE);
    }
}

NodeDefPtr Implementation::getNodeDef() const
{
    return resolveNameReference<NodeDef>(getNodeDefString());
}

bool Implementation::validate(string* message) const
{
    bool res = true;
    validateRequire(!hasVersionString(), res, message, "Implementation elements do not support version strings");
    return InterfaceElement::validate(message) && res;
}

ConstInterfaceElementPtr Implementation::getDeclaration(const string&) const
{
    return getNodeDef();
}

StringVec TargetDef::getMatchingTargets() const
{
    StringVec result = { getName() };
    ElementPtr base = getInheritsFrom();
    while (base)
    {
        result.push_back(base->getName());
        base = base->getInheritsFrom();
    }
    return result;
}

vector<UnitDefPtr> UnitTypeDef::getUnitDefs() const
{
    vector<UnitDefPtr> unitDefs;
    for (UnitDefPtr unitDef : getDocument()->getChildrenOfType<UnitDef>())
    {
        if (unitDef->getUnitType() == _name)
        {
            unitDefs.push_back(unitDef);
        }
    }
    return unitDefs;
}

MATERIALX_NAMESPACE_END
