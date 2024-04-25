//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenShader/TypeDesc.h>

#include <MaterialXCore/Exception.h>

MATERIALX_NAMESPACE_BEGIN

namespace
{

using TypeDescPtr = std::unique_ptr<TypeDesc>;
using TypeDescMap = std::unordered_map<string, TypeDescPtr>;

// Internal storage of the type descriptor pointers
TypeDescMap& typeMap()
{
    static TypeDescMap map;
    return map;
}

} // anonymous namespace

//
// TypeDesc methods
//

TypeDesc::TypeDesc(const string& name, unsigned char basetype, unsigned char semantic, size_t size,
                   bool editable, const std::unordered_map<char, int>& channelMapping) :
    _name(name),
    _basetype(basetype),
    _semantic(semantic),
    _size(size),
    _editable(editable),
    _channelMapping(channelMapping)
{
}

bool TypeDesc::operator==(const TypeDesc& rhs) const
{
    return (this->_name == rhs._name);
}

bool TypeDesc::operator!=(const TypeDesc& rhs) const
{
    return !(*this == rhs);
}

const TypeDesc* TypeDesc::registerType(const string& name, unsigned char basetype, unsigned char semantic, size_t size,
                                       bool editable, const std::unordered_map<char, int>& channelMapping)
{
    TypeDescMap& map = typeMap();
    auto it = map.find(name);
    if (it != map.end())
    {
        throw Exception("A type with name '" + name + "' is already registered");
    }

    TypeDesc* typeDesc = new TypeDesc(name, basetype, semantic, size, editable, channelMapping);
    map[name] = std::unique_ptr<TypeDesc>(typeDesc);
    return typeDesc;
}

int TypeDesc::getChannelIndex(char channel) const
{
    auto it = _channelMapping.find(channel);
    return it != _channelMapping.end() ? it->second : -1;
}

const TypeDesc* TypeDesc::get(const string& name)
{
    const TypeDescMap& map = typeMap();
    auto it = map.find(name);
    return it != map.end() ? it->second.get() : nullptr;
}

namespace Type
{

// Register all standard types and save their pointers
// for quick access and type comparisons later.
const TypeDesc* NONE                = TypeDesc::registerType("none", TypeDesc::BASETYPE_NONE, TypeDesc::SEMANTIC_NONE, 1, false);
const TypeDesc* MULTIOUTPUT         = TypeDesc::registerType("multioutput", TypeDesc::BASETYPE_NONE, TypeDesc::SEMANTIC_NONE, 1, false);
const TypeDesc* BOOLEAN             = TypeDesc::registerType("boolean", TypeDesc::BASETYPE_BOOLEAN);
const TypeDesc* INTEGER             = TypeDesc::registerType("integer", TypeDesc::BASETYPE_INTEGER);
const TypeDesc* INTEGERARRAY        = TypeDesc::registerType("integerarray", TypeDesc::BASETYPE_INTEGER, TypeDesc::SEMANTIC_NONE, 0);
const TypeDesc* FLOAT               = TypeDesc::registerType("float", TypeDesc::BASETYPE_FLOAT);
const TypeDesc* FLOATARRAY          = TypeDesc::registerType("floatarray", TypeDesc::BASETYPE_FLOAT, TypeDesc::SEMANTIC_NONE, 0);
const TypeDesc* VECTOR2             = TypeDesc::registerType("vector2", TypeDesc::BASETYPE_FLOAT, TypeDesc::SEMANTIC_VECTOR, 2, true, {{'x', 0}, {'y', 1}});
const TypeDesc* VECTOR3             = TypeDesc::registerType("vector3", TypeDesc::BASETYPE_FLOAT, TypeDesc::SEMANTIC_VECTOR, 3, true, {{'x', 0}, {'y', 1}, {'z', 2}});
const TypeDesc* VECTOR4             = TypeDesc::registerType("vector4", TypeDesc::BASETYPE_FLOAT, TypeDesc::SEMANTIC_VECTOR, 4, true, {{'x', 0}, {'y', 1}, {'z', 2}, {'w', 3}});
const TypeDesc* COLOR3              = TypeDesc::registerType("color3", TypeDesc::BASETYPE_FLOAT, TypeDesc::SEMANTIC_COLOR, 3, true, {{'r', 0}, {'g', 1}, {'b', 2}});
const TypeDesc* COLOR4              = TypeDesc::registerType("color4", TypeDesc::BASETYPE_FLOAT, TypeDesc::SEMANTIC_COLOR, 4, true, {{'r', 0}, {'g', 1}, {'b', 2}, {'a', 3}});
const TypeDesc* MATRIX33            = TypeDesc::registerType("matrix33", TypeDesc::BASETYPE_FLOAT, TypeDesc::SEMANTIC_MATRIX, 9);
const TypeDesc* MATRIX44            = TypeDesc::registerType("matrix44", TypeDesc::BASETYPE_FLOAT, TypeDesc::SEMANTIC_MATRIX, 16);
const TypeDesc* STRING              = TypeDesc::registerType("string", TypeDesc::BASETYPE_STRING);
const TypeDesc* FILENAME            = TypeDesc::registerType("filename", TypeDesc::BASETYPE_STRING, TypeDesc::SEMANTIC_FILENAME);
const TypeDesc* BSDF                = TypeDesc::registerType("BSDF", TypeDesc::BASETYPE_NONE, TypeDesc::SEMANTIC_CLOSURE, 1, false);
const TypeDesc* EDF                 = TypeDesc::registerType("EDF", TypeDesc::BASETYPE_NONE, TypeDesc::SEMANTIC_CLOSURE, 1, false);
const TypeDesc* VDF                 = TypeDesc::registerType("VDF", TypeDesc::BASETYPE_NONE, TypeDesc::SEMANTIC_CLOSURE, 1, false);
const TypeDesc* SURFACESHADER       = TypeDesc::registerType("surfaceshader", TypeDesc::BASETYPE_NONE, TypeDesc::SEMANTIC_SHADER, 1, false);
const TypeDesc* VOLUMESHADER        = TypeDesc::registerType("volumeshader", TypeDesc::BASETYPE_NONE, TypeDesc::SEMANTIC_SHADER, 1, false);
const TypeDesc* DISPLACEMENTSHADER  = TypeDesc::registerType("displacementshader", TypeDesc::BASETYPE_NONE, TypeDesc::SEMANTIC_SHADER, 1, false);
const TypeDesc* LIGHTSHADER         = TypeDesc::registerType("lightshader", TypeDesc::BASETYPE_NONE, TypeDesc::SEMANTIC_SHADER, 1, false);
const TypeDesc* MATERIAL            = TypeDesc::registerType("material", TypeDesc::BASETYPE_NONE, TypeDesc::SEMANTIC_MATERIAL, 1, false);

} // namespace Type

MATERIALX_NAMESPACE_END
