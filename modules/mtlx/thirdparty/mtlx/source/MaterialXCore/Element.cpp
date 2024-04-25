//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXCore/Element.h>

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Util.h>

#include <iterator>

MATERIALX_NAMESPACE_BEGIN

const string Element::NAME_ATTRIBUTE = "name";
const string Element::FILE_PREFIX_ATTRIBUTE = "fileprefix";
const string Element::GEOM_PREFIX_ATTRIBUTE = "geomprefix";
const string Element::COLOR_SPACE_ATTRIBUTE = "colorspace";
const string Element::INHERIT_ATTRIBUTE = "inherit";
const string Element::NAMESPACE_ATTRIBUTE = "namespace";
const string Element::DOC_ATTRIBUTE = "doc";
const string TypedElement::TYPE_ATTRIBUTE = "type";
const string ValueElement::VALUE_ATTRIBUTE = "value";
const string ValueElement::INTERFACE_NAME_ATTRIBUTE = "interfacename";
const string ValueElement::ENUM_ATTRIBUTE = "enum";
const string ValueElement::IMPLEMENTATION_NAME_ATTRIBUTE = "implname";
const string ValueElement::IMPLEMENTATION_TYPE_ATTRIBUTE = "impltype";
const string ValueElement::ENUM_VALUES_ATTRIBUTE = "enumvalues";
const string ValueElement::UI_NAME_ATTRIBUTE = "uiname";
const string ValueElement::UI_FOLDER_ATTRIBUTE = "uifolder";
const string ValueElement::UI_MIN_ATTRIBUTE = "uimin";
const string ValueElement::UI_MAX_ATTRIBUTE = "uimax";
const string ValueElement::UI_SOFT_MIN_ATTRIBUTE = "uisoftmin";
const string ValueElement::UI_SOFT_MAX_ATTRIBUTE = "uisoftmax";
const string ValueElement::UI_STEP_ATTRIBUTE = "uistep";
const string ValueElement::UI_ADVANCED_ATTRIBUTE = "uiadvanced";
const string ValueElement::UNIT_ATTRIBUTE = "unit";
const string ValueElement::UNITTYPE_ATTRIBUTE = "unittype";
const string ValueElement::UNIFORM_ATTRIBUTE = "uniform";

Element::CreatorMap Element::_creatorMap;

//
// Element methods
//

bool Element::operator==(const Element& rhs) const
{
    if (getCategory() != rhs.getCategory() ||
        getName() != rhs.getName())
    {
        return false;
    }

    // Compare attributes.
    if (getAttributeNames() != rhs.getAttributeNames())
        return false;
    for (const string& attr : rhs.getAttributeNames())
    {
        if (getAttribute(attr) != rhs.getAttribute(attr))
            return false;
    }

    // Compare children.
    const vector<ElementPtr>& c1 = getChildren();
    const vector<ElementPtr>& c2 = rhs.getChildren();
    if (c1.size() != c2.size())
        return false;
    for (size_t i = 0; i < c1.size(); i++)
    {
        if (*c1[i] != *c2[i])
            return false;
    }
    return true;
}

bool Element::operator!=(const Element& rhs) const
{
    return !(*this == rhs);
}

void Element::setName(const string& name)
{
    ElementPtr parent = getParent();
    if (parent && parent->_childMap.count(name) && name != getName())
    {
        throw Exception("Element name is not unique at the given scope: " + name);
    }

    getDocument()->invalidateCache();

    if (parent)
    {
        parent->_childMap.erase(getName());
        parent->_childMap[name] = getSelf();
    }
    _name = name;
}

string Element::getNamePath(ConstElementPtr relativeTo) const
{
    if (!relativeTo)
    {
        relativeTo = getDocument();
    }

    string res;
    for (ConstElementPtr elem = getSelf(); elem; elem = elem->getParent())
    {
        if (elem == relativeTo)
        {
            break;
        }
        res = res.empty() ? elem->getName() : elem->getName() + NAME_PATH_SEPARATOR + res;
    }
    return res;
}

ElementPtr Element::getDescendant(const string& namePath) const
{
    const StringVec nameVec = splitString(namePath, NAME_PATH_SEPARATOR);
    ElementPtr elem = getSelfNonConst();
    for (const string& name : nameVec)
    {
        elem = elem->getChild(name);
        if (!elem)
        {
            return ElementPtr();
        }
    }
    return elem;
}

void Element::registerChildElement(ElementPtr child)
{
    getDocument()->invalidateCache();

    _childMap[child->getName()] = child;
    _childOrder.push_back(child);
}

void Element::unregisterChildElement(ElementPtr child)
{
    getDocument()->invalidateCache();

    _childMap.erase(child->getName());
    _childOrder.erase(
        std::find(_childOrder.begin(), _childOrder.end(), child));
}

int Element::getChildIndex(const string& name) const
{
    ElementPtr child = getChild(name);
    vector<ElementPtr>::const_iterator it = std::find(_childOrder.begin(), _childOrder.end(), child);
    if (it == _childOrder.end())
    {
        return -1;
    }
    return (int) std::distance(_childOrder.begin(), it);
}

void Element::setChildIndex(const string& name, int index)
{
    ElementPtr child = getChild(name);
    vector<ElementPtr>::iterator it = std::find(_childOrder.begin(), _childOrder.end(), child);
    if (it == _childOrder.end())
    {
        return;
    }

    if (index < 0 || index >= (int) _childOrder.size())
    {
        throw Exception("Invalid child index");
    }

    _childOrder.erase(it);
    _childOrder.insert(_childOrder.begin() + (size_t) index, child);
}

void Element::removeChild(const string& name)
{
    ElementMap::iterator it = _childMap.find(name);
    if (it == _childMap.end())
    {
        return;
    }

    unregisterChildElement(it->second);
}

void Element::setAttribute(const string& attrib, const string& value)
{
    getDocument()->invalidateCache();

    if (!_attributeMap.count(attrib))
    {
        _attributeOrder.push_back(attrib);
    }
    _attributeMap[attrib] = value;
}

void Element::removeAttribute(const string& attrib)
{
    StringMap::iterator it = _attributeMap.find(attrib);
    if (it != _attributeMap.end())
    {
        getDocument()->invalidateCache();

        _attributeMap.erase(it);
        _attributeOrder.erase(
            std::find(_attributeOrder.begin(), _attributeOrder.end(), attrib));
    }
}

template <class T> shared_ptr<T> Element::asA()
{
    return std::dynamic_pointer_cast<T>(getSelf());
}

template <class T> shared_ptr<const T> Element::asA() const
{
    return std::dynamic_pointer_cast<const T>(getSelf());
}

ElementPtr Element::addChildOfCategory(const string& category, string name)
{
    if (name.empty())
    {
        name = createValidChildName(category + "1");
    }
    if (_childMap.count(name))
    {
        throw Exception("Child name is not unique: " + name);
    }

    ElementPtr child;

    // Check for this category in the creator map.
    CreatorMap::iterator it = _creatorMap.find(category);
    if (it != _creatorMap.end())
    {
        child = it->second(getSelf(), name);
    }

    // Check for a node within a graph.
    if (!child && isA<GraphElement>())
    {
        child = createElement<Node>(getSelf(), name);
        child->setCategory(category);
    }

    // If no match was found, then create a generic element.
    if (!child)
    {
        child = createElement<GenericElement>(getSelf(), name);
        child->setCategory(category);
    }

    registerChildElement(child);

    return child;
}

ElementPtr Element::changeChildCategory(ElementPtr child, const string& category)
{
    int childIndex = getChildIndex(child->getName());
    if (childIndex == -1)
    {
        return nullptr;
    }

    removeChild(child->getName());
    ElementPtr newChild = addChildOfCategory(category, child->getName());
    setChildIndex(child->getName(), childIndex);
    newChild->copyContentFrom(child);
    return newChild;
}

ElementPtr Element::getRoot()
{
    ElementPtr root = _root.lock();
    if (!root)
    {
        throw ExceptionOrphanedElement("Requested root of orphaned element: " + asString());
    }
    return root;
}

ConstElementPtr Element::getRoot() const
{
    ElementPtr root = _root.lock();
    if (!root)
    {
        throw ExceptionOrphanedElement("Requested root of orphaned element: " + asString());
    }
    return root;
}

DocumentPtr Element::getDocument()
{
    return getRoot()->asA<Document>();
}

/// Return the root document of our tree.
ConstDocumentPtr Element::getDocument() const
{
    return getRoot()->asA<Document>();
}

bool Element::hasInheritedBase(ConstElementPtr base) const
{
    for (ConstElementPtr elem : traverseInheritance())
    {
        if (elem == base)
        {
            return true;
        }
    }
    return false;
}

bool Element::hasInheritanceCycle() const
{
    try
    {
        for (ConstElementPtr elem : traverseInheritance()) { }
    }
    catch (ExceptionFoundCycle&)
    {
        return true;
    }
    return false;
}

TreeIterator Element::traverseTree() const
{
    return TreeIterator(getSelfNonConst());
}

GraphIterator Element::traverseGraph() const
{
    return GraphIterator(getSelfNonConst());
}

Edge Element::getUpstreamEdge(size_t) const
{
    return NULL_EDGE;
}

ElementPtr Element::getUpstreamElement(size_t index) const
{
    return getUpstreamEdge(index).getUpstreamElement();
}

InheritanceIterator Element::traverseInheritance() const
{
    return InheritanceIterator(getSelf());
}

void Element::copyContentFrom(const ConstElementPtr& source)
{
    getDocument()->invalidateCache();

    _sourceUri = source->_sourceUri;
    _attributeMap = source->_attributeMap;
    _attributeOrder = source->_attributeOrder;

    for (auto child : source->getChildren())
    {
        const string& name = child->getName();

        // Check for duplicate elements.
        ConstElementPtr previous = getChild(name);
        if (previous)
        {
            continue;
        }

        // Create the copied element.
        ElementPtr childCopy = addChildOfCategory(child->getCategory(), name);
        childCopy->copyContentFrom(child);
    }
}

void Element::clearContent()
{
    getDocument()->invalidateCache();

    _sourceUri.clear();
    _attributeMap.clear();
    _attributeOrder.clear();
    _childMap.clear();
    _childOrder.clear();
}

bool Element::validate(string* message) const
{
    bool res = true;
    validateRequire(isValidName(getName()), res, message, "Invalid element name");
    if (hasInheritString())
    {
        bool validInherit = getInheritsFrom() && getInheritsFrom()->getCategory() == getCategory();
        validateRequire(validInherit, res, message, "Invalid element inheritance");
    }
    for (auto child : getChildren())
    {
        res = child->validate(message) && res;
    }
    validateRequire(!hasInheritanceCycle(), res, message, "Cycle in element inheritance chain");
    return res;
}

StringResolverPtr Element::createStringResolver(const string& geom) const
{
    StringResolverPtr resolver = StringResolver::create();

    // Compute file and geom prefixes as this scope.
    resolver->setFilePrefix(getActiveFilePrefix());
    resolver->setGeomPrefix(getActiveGeomPrefix());

    // If a geometry name is specified, then apply it to the filename map.
    if (!geom.empty())
    {
        for (GeomInfoPtr geomInfo : getDocument()->getGeomInfos())
        {
            if (!geomStringsMatch(geom, geomInfo->getActiveGeom()))
                continue;
            for (TokenPtr token : geomInfo->getTokens())
            {
                string key = "<" + token->getName() + ">";
                string value = token->getResolvedValueString();
                resolver->setFilenameSubstitution(key, value);
            }
        }
    }

    // Add in token substitutions
    resolver->addTokenSubstitutions(getSelf());

    return resolver;
}

string Element::asString() const
{
    string res = "<" + getCategory();
    if (getName() != EMPTY_STRING)
    {
        res += " name=\"" + getName() + "\"";
    }
    for (const string& attrName : getAttributeNames())
    {
        res += " " + attrName + "=\"" + getAttribute(attrName) + "\"";
    }
    res += ">";
    return res;
}

void Element::validateRequire(bool expression, bool& res, string* message, const string& errorDesc) const
{
    if (!expression)
    {
        res = false;
        if (message)
        {
            *message += errorDesc + ": " + asString() + "\n";
        }
    }
}

//
// TypedElement methods
//

TypeDefPtr TypedElement::getTypeDef() const
{
    return resolveNameReference<TypeDef>(getType());
}

//
// ValueElement methods
//

string ValueElement::getResolvedValueString(StringResolverPtr resolver) const
{
    if (!StringResolver::isResolvedType(getType()))
    {
        return getValueString();
    }
    if (!resolver)
    {
        resolver = createStringResolver();
    }
    return resolver->resolve(getValueString(), getType());
}

ValuePtr ValueElement::getDefaultValue() const
{
    ConstElementPtr parent = getParent();
    ConstInterfaceElementPtr interface = parent ? parent->asA<InterfaceElement>() : nullptr;
    if (interface)
    {
        ConstInterfaceElementPtr decl = interface->getDeclaration();
        if (decl)
        {
            ValueElementPtr value = decl->getActiveValueElement(getName());
            if (value)
            {
                return value->getValue();
            }
        }
    }
    return ValuePtr();
}

const string& ValueElement::getActiveUnit() const
{
    // Return the unit, if any, stored in our declaration.
    ConstElementPtr parent = getParent();
    ConstInterfaceElementPtr interface = parent ? parent->asA<InterfaceElement>() : nullptr;
    if (interface)
    {
        ConstInterfaceElementPtr decl = interface->getDeclaration();
        if (decl)
        {
            ValueElementPtr value = decl->getActiveValueElement(getName());
            if (value)
            {
                return value->getUnit();
            }
        }
    }
    return EMPTY_STRING;
}

bool ValueElement::validate(string* message) const
{
    bool res = true;
    if (hasType() && hasValueString())
    {
        validateRequire(getValue() != nullptr, res, message, "Invalid value");
    }

    if (hasInterfaceName())
    {
        validateRequire(isA<Input>() || isA<Token>(), res, message, "Only input and token elements support interface names");
        ConstNodeGraphPtr nodeGraph = getAncestorOfType<NodeGraph>();
        ConstInterfaceElementPtr decl = nodeGraph ? nodeGraph->getDeclaration() : nullptr;
        if (decl)
        {
            ValueElementPtr valueElem = decl->getActiveValueElement(getInterfaceName());
            validateRequire(valueElem != nullptr, res, message, "Interface name not found in referenced declaration");
            if (valueElem)
            {
                ConstPortElementPtr portElem = asA<PortElement>();
                if (portElem && portElem->hasChannels())
                {
                    bool valid = portElem->validChannelsString(portElem->getChannels(), valueElem->getType(), getType());
                    validateRequire(valid, res, message, "Invalid channels string for interface name");
                }
                else
                {
                    validateRequire(getType() == valueElem->getType(), res, message, "Interface name refers to value element of a different type");
                }
            }
        }
    }

    UnitTypeDefPtr unitTypeDef;
    if (hasUnitType())
    {
        const string& unittype = getUnitType();
        if (!unittype.empty())
        {
            unitTypeDef = getDocument()->getUnitTypeDef(unittype);
            validateRequire(unitTypeDef != nullptr, res, message, "Unit type definition does not exist in document");
        }
    }
    if (hasUnit())
    {
        bool foundUnit = false;
        if (unitTypeDef)
        {
            const string& unit = getUnit();
            for (UnitDefPtr unitDef : unitTypeDef->getUnitDefs())
            {
                if (unitDef->getUnit(unit))
                {
                    foundUnit = true;
                    break;
                }
            }
        }
        validateRequire(foundUnit, res, message, "Unit definition does not exist in document");
    }
    return TypedElement::validate(message) && res;
}

//
// StringResolver methods
//

void StringResolver::setUdimString(const string& udim)
{
    setFilenameSubstitution(UDIM_TOKEN, udim);
}

void StringResolver::setUvTileString(const string& uvTile)
{
    setFilenameSubstitution(UV_TILE_TOKEN, uvTile);
}

void StringResolver::addTokenSubstitutions(ConstElementPtr element)
{
    const string DELIMITER_PREFIX = "[";
    const string DELIMITER_POSTFIX = "]";

    // Travese from sibliings up until root is reached.
    // Child tokens override any parent tokens.
    ConstElementPtr parent = element->getParent();
    while (parent)
    {
        ConstInterfaceElementPtr interfaceElem = parent->asA<InterfaceElement>();
        if (interfaceElem)
        {
            vector<TokenPtr> tokens = interfaceElem->getActiveTokens();
            for (auto token : tokens)
            {
                string key = DELIMITER_PREFIX + token->getName() + DELIMITER_POSTFIX;
                string value = token->getResolvedValueString();
                if (!_filenameMap.count(key))
                {
                    setFilenameSubstitution(key, value);
                }
            }
        }
        parent = parent->getParent();
    }
}

string StringResolver::resolve(const string& str, const string& type) const
{
    if (type == FILENAME_TYPE_STRING)
    {
        return _filePrefix + replaceSubstrings(str, _filenameMap);
    }
    if (type == GEOMNAME_TYPE_STRING)
    {
        return _geomPrefix + replaceSubstrings(str, _geomNameMap);
    }
    return str;
}

//
// Global functions
//

bool targetStringsMatch(const string& target1, const string& target2)
{
    if (target1.empty() || target2.empty())
        return true;

    StringVec vec1 = splitString(target1, ARRAY_VALID_SEPARATORS);
    StringVec vec2 = splitString(target2, ARRAY_VALID_SEPARATORS);
    StringSet set1(vec1.begin(), vec1.end());
    StringSet set2(vec2.begin(), vec2.end());

    StringSet matches;
    std::set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(),
                          std::inserter(matches, matches.end()));
    return !matches.empty();
}

string prettyPrint(ConstElementPtr elem)
{
    string text;
    for (TreeIterator it = elem->traverseTree().begin(); it != TreeIterator::end(); ++it)
    {
        string indent(it.getElementDepth() * 2, ' ');
        text += indent + it.getElement()->asString() + "\n";
    }
    return text;
}

//
// Element registry class
//

template <class T> class ElementRegistry
{
  public:
    ElementRegistry()
    {
        Element::_creatorMap[T::CATEGORY] = Element::createElement<T>;
    }
    ~ElementRegistry() { }
};

//
// Template instantiations
//

#define INSTANTIATE_SUBCLASS(T)                           \
    template MX_CORE_API shared_ptr<T> Element::asA<T>(); \
    template MX_CORE_API shared_ptr<const T> Element::asA<T>() const;

INSTANTIATE_SUBCLASS(Element)
INSTANTIATE_SUBCLASS(GeomElement)
INSTANTIATE_SUBCLASS(GraphElement)
INSTANTIATE_SUBCLASS(InterfaceElement)
INSTANTIATE_SUBCLASS(PortElement)
INSTANTIATE_SUBCLASS(TypedElement)
INSTANTIATE_SUBCLASS(ValueElement)

#define INSTANTIATE_CONCRETE_SUBCLASS(T, category) \
    const string T::CATEGORY(category);            \
    ElementRegistry<T> registry##T;                \
    INSTANTIATE_SUBCLASS(T)

INSTANTIATE_CONCRETE_SUBCLASS(AttributeDef, "attributedef")
INSTANTIATE_CONCRETE_SUBCLASS(Backdrop, "backdrop")
INSTANTIATE_CONCRETE_SUBCLASS(Collection, "collection")
INSTANTIATE_CONCRETE_SUBCLASS(CommentElement, "comment")
INSTANTIATE_CONCRETE_SUBCLASS(Document, "materialx")
INSTANTIATE_CONCRETE_SUBCLASS(GenericElement, "generic")
INSTANTIATE_CONCRETE_SUBCLASS(GeomInfo, "geominfo")
INSTANTIATE_CONCRETE_SUBCLASS(GeomProp, "geomprop")
INSTANTIATE_CONCRETE_SUBCLASS(GeomPropDef, "geompropdef")
INSTANTIATE_CONCRETE_SUBCLASS(Implementation, "implementation")
INSTANTIATE_CONCRETE_SUBCLASS(Input, "input")
INSTANTIATE_CONCRETE_SUBCLASS(Look, "look")
INSTANTIATE_CONCRETE_SUBCLASS(LookGroup, "lookgroup")
INSTANTIATE_CONCRETE_SUBCLASS(MaterialAssign, "materialassign")
INSTANTIATE_CONCRETE_SUBCLASS(Member, "member")
INSTANTIATE_CONCRETE_SUBCLASS(NewlineElement, "newline")
INSTANTIATE_CONCRETE_SUBCLASS(Node, "node")
INSTANTIATE_CONCRETE_SUBCLASS(NodeDef, "nodedef")
INSTANTIATE_CONCRETE_SUBCLASS(NodeGraph, "nodegraph")
INSTANTIATE_CONCRETE_SUBCLASS(Output, "output")
INSTANTIATE_CONCRETE_SUBCLASS(Property, "property")
INSTANTIATE_CONCRETE_SUBCLASS(PropertyAssign, "propertyassign")
INSTANTIATE_CONCRETE_SUBCLASS(PropertySet, "propertyset")
INSTANTIATE_CONCRETE_SUBCLASS(PropertySetAssign, "propertysetassign")
INSTANTIATE_CONCRETE_SUBCLASS(TargetDef, "targetdef")
INSTANTIATE_CONCRETE_SUBCLASS(Token, "token")
INSTANTIATE_CONCRETE_SUBCLASS(TypeDef, "typedef")
INSTANTIATE_CONCRETE_SUBCLASS(Unit, "unit")
INSTANTIATE_CONCRETE_SUBCLASS(UnitDef, "unitdef")
INSTANTIATE_CONCRETE_SUBCLASS(UnitTypeDef, "unittypedef")
INSTANTIATE_CONCRETE_SUBCLASS(Variant, "variant")
INSTANTIATE_CONCRETE_SUBCLASS(VariantAssign, "variantassign")
INSTANTIATE_CONCRETE_SUBCLASS(VariantSet, "variantset")
INSTANTIATE_CONCRETE_SUBCLASS(Visibility, "visibility")

MATERIALX_NAMESPACE_END
