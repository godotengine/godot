//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenShader/Syntax.h>
#include <MaterialXGenShader/TypeDesc.h>
#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/GenContext.h>

#include <MaterialXCore/Value.h>

MATERIALX_NAMESPACE_BEGIN

const string Syntax::NEWLINE = "\n";
const string Syntax::SEMICOLON = ";";
const string Syntax::COMMA = ",";
const string Syntax::INDENTATION = "    ";
const string Syntax::STRING_QUOTE = "\"";
const string Syntax::INCLUDE_STATEMENT = "#include";
const string Syntax::SINGLE_LINE_COMMENT = "// ";
const string Syntax::BEGIN_MULTI_LINE_COMMENT = "/* ";
const string Syntax::END_MULTI_LINE_COMMENT = " */";

const std::unordered_map<char, size_t> Syntax::CHANNELS_MAPPING =
{
    { 'r', 0 }, { 'x', 0 },
    { 'g', 1 }, { 'y', 1 },
    { 'b', 2 }, { 'z', 2 },
    { 'a', 3 }, { 'w', 3 }
};

//
// Syntax methods
//

Syntax::Syntax()
{
}

void Syntax::registerTypeSyntax(const TypeDesc* type, TypeSyntaxPtr syntax)
{
    auto it = _typeSyntaxByType.find(type);
    if (it != _typeSyntaxByType.end())
    {
        _typeSyntaxes[it->second] = syntax;
    }
    else
    {
        _typeSyntaxes.push_back(syntax);
        _typeSyntaxByType[type] = _typeSyntaxes.size() - 1;
    }

    // Make this type a restricted name
    registerReservedWords({ syntax->getName() });
}

void Syntax::registerReservedWords(const StringSet& names)
{
    _reservedWords.insert(names.begin(), names.end());
}

void Syntax::registerInvalidTokens(const StringMap& tokens)
{
    _invalidTokens.insert(tokens.begin(), tokens.end());
}

/// Returns the type syntax object for a named type.
/// Throws an exception if a type syntax is not defined for the given type.
const TypeSyntax& Syntax::getTypeSyntax(const TypeDesc* type) const
{
    auto it = _typeSyntaxByType.find(type);
    if (it == _typeSyntaxByType.end())
    {
        string typeName = type ? type->getName() : "nullptr";
        throw ExceptionShaderGenError("No syntax is defined for the given type '" + typeName + "'.");
    }
    return *_typeSyntaxes[it->second];
}

const TypeDesc* Syntax::getTypeDescription(const TypeSyntaxPtr& typeSyntax) const
{
    auto pos = std::find(_typeSyntaxes.begin(), _typeSyntaxes.end(), typeSyntax);
    if (pos == _typeSyntaxes.end())
    {
        throw ExceptionShaderGenError("The syntax'" + typeSyntax->getName() + "' is not registered.");
    }
    const size_t index = static_cast<size_t>(std::distance(_typeSyntaxes.begin(), pos));
    for (auto item : _typeSyntaxByType)
    {
        if (item.second == index)
        {
            return item.first;
        }
    }
    return nullptr;
}

string Syntax::getValue(const ShaderPort* port, bool uniform) const
{
    const TypeSyntax& syntax = getTypeSyntax(port->getType());
    return syntax.getValue(port, uniform);
}

string Syntax::getValue(const TypeDesc* type, const Value& value, bool uniform) const
{
    const TypeSyntax& syntax = getTypeSyntax(type);
    return syntax.getValue(value, uniform);
}

const string& Syntax::getDefaultValue(const TypeDesc* type, bool uniform) const
{
    const TypeSyntax& syntax = getTypeSyntax(type);
    return syntax.getDefaultValue(uniform);
}

const string& Syntax::getTypeName(const TypeDesc* type) const
{
    const TypeSyntax& syntax = getTypeSyntax(type);
    return syntax.getName();
}

string Syntax::getOutputTypeName(const TypeDesc* type) const
{
    const TypeSyntax& syntax = getTypeSyntax(type);
    const string& outputModifier = getOutputQualifier();
    return outputModifier.size() ? outputModifier + " " + syntax.getName() : syntax.getName();
}

const string& Syntax::getTypeAlias(const TypeDesc* type) const
{
    const TypeSyntax& syntax = getTypeSyntax(type);
    return syntax.getTypeAlias();
}

const string& Syntax::getTypeDefinition(const TypeDesc* type) const
{
    const TypeSyntax& syntax = getTypeSyntax(type);
    return syntax.getTypeDefinition();
}

string Syntax::getSwizzledVariable(const string& srcName, const TypeDesc* srcType, const string& channels, const TypeDesc* dstType) const
{
    const TypeSyntax& srcSyntax = getTypeSyntax(srcType);
    const TypeSyntax& dstSyntax = getTypeSyntax(dstType);

    const StringVec& srcMembers = srcSyntax.getMembers();

    StringVec membersSwizzled;

    for (size_t i = 0; i < channels.size(); ++i)
    {
        const char ch = channels[i];
        if (ch == '0' || ch == '1')
        {
            membersSwizzled.push_back(string(1, ch));
            continue;
        }

        auto it = CHANNELS_MAPPING.find(ch);
        if (it == CHANNELS_MAPPING.end())
        {
            throw ExceptionShaderGenError("Invalid channel pattern '" + channels + "'.");
        }

        if (srcMembers.empty())
        {
            membersSwizzled.push_back(srcName);
        }
        else
        {
            int channelIndex = srcType->getChannelIndex(ch);
            if (channelIndex < 0 || channelIndex >= static_cast<int>(srcMembers.size()))
            {
                throw ExceptionShaderGenError("Given channel index: '" + string(1, ch) + "' in channels pattern is incorrect for type '" + srcType->getName() + "'.");
            }
            membersSwizzled.push_back(srcName + srcMembers[channelIndex]);
        }
    }

    return dstSyntax.getValue(membersSwizzled, false);
}

ValuePtr Syntax::getSwizzledValue(ValuePtr value, const TypeDesc* srcType, const string& channels, const TypeDesc* dstType) const
{
    const TypeSyntax& srcSyntax = getTypeSyntax(srcType);
    const vector<string>& srcMembers = srcSyntax.getMembers();

    StringStream ss;
    string delimiter = ", ";

    for (size_t i = 0; i < channels.size(); ++i)
    {
        if (i == channels.size() - 1)
        {
            delimiter.clear();
        }

        const char ch = channels[i];
        if (ch == '0' || ch == '1')
        {
            ss << string(1, ch);
            continue;
        }

        auto it = CHANNELS_MAPPING.find(ch);
        if (it == CHANNELS_MAPPING.end())
        {
            throw ExceptionShaderGenError("Invalid channel pattern '" + channels + "'.");
        }

        if (srcMembers.empty())
        {
            ss << value->getValueString();
        }
        else
        {
            int channelIndex = srcType->getChannelIndex(ch);
            if (channelIndex < 0 || channelIndex >= static_cast<int>(srcMembers.size()))
            {
                throw ExceptionShaderGenError("Given channel index: '" + string(1, ch) + "' in channels pattern is incorrect for type '" + srcType->getName() + "'.");
            }
            if (*srcType == *Type::FLOAT)
            {
                float v = value->asA<float>();
                ss << std::to_string(v);
            }
            else if (*srcType == *Type::INTEGER)
            {
                int v = value->asA<int>();
                ss << std::to_string(v);
            }
            else if (*srcType == *Type::BOOLEAN)
            {
                bool v = value->asA<bool>();
                ss << std::to_string(v);
            }
            else if (*srcType == *Type::COLOR3)
            {
                Color3 v = value->asA<Color3>();
                ss << std::to_string(v[channelIndex]);
            }
            else if (*srcType == *Type::COLOR4)
            {
                Color4 v = value->asA<Color4>();
                ss << std::to_string(v[channelIndex]);
            }
            else if (*srcType == *Type::VECTOR2)
            {
                Vector2 v = value->asA<Vector2>();
                ss << std::to_string(v[channelIndex]);
            }
            else if (*srcType == *Type::VECTOR3)
            {
                Vector3 v = value->asA<Vector3>();
                ss << std::to_string(v[channelIndex]);
            }
            else if (*srcType == *Type::VECTOR4)
            {
                Vector4 v = value->asA<Vector4>();
                ss << std::to_string(v[channelIndex]);
            }
        }
        ss << delimiter;
    }

    return Value::createValueFromStrings(ss.str(), dstType->getName());
}

bool Syntax::typeSupported(const TypeDesc*) const
{
    return true;
}

string Syntax::getArrayVariableSuffix(const TypeDesc* type, const Value& value) const
{
    if (type->isArray())
    {
        if (value.isA<vector<float>>())
        {
            const size_t size = value.asA<vector<float>>().size();
            return "[" + std::to_string(size) + "]";
        }
        else if (value.isA<vector<int>>())
        {
            const size_t size = value.asA<vector<int>>().size();
            return "[" + std::to_string(size) + "]";
        }
    }
    return string();
}

static bool isInvalidChar(char c)
{
    return !isalnum(c) && c != '_';
}

void Syntax::makeValidName(string& name) const
{
    std::replace_if(name.begin(), name.end(), isInvalidChar, '_');
    name = replaceSubstrings(name, _invalidTokens);
}

void Syntax::makeIdentifier(string& name, IdentifierMap& identifiers) const
{
    makeValidName(name);

    auto it = identifiers.find(name);
    if (it != identifiers.end())
    {
        // Name is not unique so append the counter and keep
        // incrementing until a unique name is found.
        string name2;
        do
        {
            name2 = name + std::to_string(it->second++);
        } while (identifiers.count(name2));

        name = name2;
    }

    // Save it among the known identifiers.
    identifiers[name] = 1;
}

string Syntax::getVariableName(const string& name, const TypeDesc* /*type*/, IdentifierMap& identifiers) const
{
    // Default implementation just makes an identifier, but derived
    // classes can override this for custom variable naming.
    string variable = name;
    makeIdentifier(variable, identifiers);
    return variable;
}

bool Syntax::remapEnumeration(const string&, const TypeDesc*, const string&, std::pair<const TypeDesc*, ValuePtr>&) const
{
    return false;
}

const StringVec TypeSyntax::EMPTY_MEMBERS;

TypeSyntax::TypeSyntax(const string& name, const string& defaultValue, const string& uniformDefaultValue,
                       const string& typeAlias, const string& typeDefinition, const StringVec& members) :
    _name(name),
    _defaultValue(defaultValue),
    _uniformDefaultValue(uniformDefaultValue),
    _typeAlias(typeAlias),
    _typeDefinition(typeDefinition),
    _members(members)
{
}

string TypeSyntax::getValue(const ShaderPort* port, bool uniform) const
{
    if (!port || !port->getValue())
    {
        return getDefaultValue(uniform);
    }
    return getValue(*port->getValue(), uniform);
}

ScalarTypeSyntax::ScalarTypeSyntax(const string& name, const string& defaultValue, const string& uniformDefaultValue,
                                   const string& typeAlias, const string& typeDefinition) :
    TypeSyntax(name, defaultValue, uniformDefaultValue, typeAlias, typeDefinition, EMPTY_MEMBERS)
{
}

string ScalarTypeSyntax::getValue(const Value& value, bool /*uniform*/) const
{
    return value.getValueString();
}

string ScalarTypeSyntax::getValue(const StringVec& values, bool /*uniform*/) const
{
    if (values.empty())
    {
        throw ExceptionShaderGenError("No values given to construct a value");
    }
    // Write the value using a stream to maintain any float formatting set
    // using Value::setFloatFormat() and Value::setFloatPrecision()
    StringStream ss;
    ss << values[0];
    return ss.str();
}

StringTypeSyntax::StringTypeSyntax(const string& name, const string& defaultValue, const string& uniformDefaultValue,
                                   const string& typeAlias, const string& typeDefinition) :
    ScalarTypeSyntax(name, defaultValue, uniformDefaultValue, typeAlias, typeDefinition)
{
}

string StringTypeSyntax::getValue(const Value& value, bool /*uniform*/) const
{
    return "\"" + value.getValueString() + "\"";
}

AggregateTypeSyntax::AggregateTypeSyntax(const string& name, const string& defaultValue, const string& uniformDefaultValue,
                                         const string& typeAlias, const string& typeDefinition, const StringVec& members) :
    TypeSyntax(name, defaultValue, uniformDefaultValue, typeAlias, typeDefinition, members)
{
}

string AggregateTypeSyntax::getValue(const Value& value, bool /*uniform*/) const
{
    const string valueString = value.getValueString();
    return valueString.empty() ? valueString : getName() + "(" + valueString + ")";
}

string AggregateTypeSyntax::getValue(const StringVec& values, bool /*uniform*/) const
{
    if (values.empty())
    {
        throw ExceptionShaderGenError("No values given to construct a value");
    }

    // Write the value using a stream to maintain any float formatting set
    // using Value::setFloatFormat() and Value::setFloatPrecision()
    StringStream ss;
    ss << getName() << "(" << values[0];
    for (size_t i = 1; i < values.size(); ++i)
    {
        ss << ", " << values[i];
    }
    ss << ")";

    return ss.str();
}

MATERIALX_NAMESPACE_END
