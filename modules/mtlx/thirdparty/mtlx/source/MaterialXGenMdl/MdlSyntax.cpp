//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenMdl/MdlSyntax.h>

#include <MaterialXGenShader/ShaderGenerator.h>

#include <MaterialXFormat/File.h>

#include <sstream>

MATERIALX_NAMESPACE_BEGIN

// Custom types to handle enumeration output
namespace Type
{

const TypeDesc* MDL_COORDINATESPACE = TypeDesc::registerType("coordinatespace", TypeDesc::BASETYPE_NONE, TypeDesc::SEMANTIC_ENUM, 0);
const TypeDesc* MDL_ADDRESSMODE = TypeDesc::registerType("addressmode", TypeDesc::BASETYPE_NONE, TypeDesc::SEMANTIC_ENUM, 0);
const TypeDesc* MDL_FILTERLOOKUPMODE = TypeDesc::registerType("filterlookup", TypeDesc::BASETYPE_NONE, TypeDesc::SEMANTIC_ENUM, 0);
const TypeDesc* MDL_FILTERTYPE = TypeDesc::registerType("filtertype", TypeDesc::BASETYPE_NONE, TypeDesc::SEMANTIC_ENUM, 0);
const TypeDesc* MDL_DISTRIBUTION = TypeDesc::registerType("distributiontype", TypeDesc::BASETYPE_NONE, TypeDesc::SEMANTIC_ENUM, 0);
const TypeDesc* MDL_SCATTER_MODE = TypeDesc::registerType("scatter_mode", TypeDesc::BASETYPE_NONE, TypeDesc::SEMANTIC_ENUM, 0);

} // namespace Type

namespace
{

class MdlFilenameTypeSyntax : public ScalarTypeSyntax
{
  public:
    MdlFilenameTypeSyntax() :
        ScalarTypeSyntax("texture_2d", "texture_2d()", "texture_2d()")
    {
    }

    string getValue(const Value& value, bool /*uniform*/) const override
    {
        const string outputValue = value.getValueString();
        if (outputValue.empty() || outputValue == "/")
        {
            return getDefaultValue(true);
        }
        // handle the empty texture, the fileprefix is passed
        // assuming it ends with a slash ...
        if (outputValue.back() == '/')
        {
            return getDefaultValue(true);
        }
        // ... or the last segment does not have an extension suffix
        size_t idx_s = outputValue.find_last_of('/');
        size_t idx_d = outputValue.find_last_of('.');
        if (idx_d == std::string::npos || (idx_s != std::string::npos && idx_s > idx_d))
        {
            return getDefaultValue(true);
        }

        // prefix a slash in order to make MDL resource paths absolute i.e. to be found
        // in the root of an MDL search path
        // do not add the slash in case the path is explicitly relative
        string pathSeparator("");
        FilePath path(outputValue);
        size_t len = outputValue.size();
        if (!path.isAbsolute() &&
            !(len > 2 && outputValue[0] == '.' && outputValue[1] == '.' && outputValue[2] == '/') &&
            !(len > 1 && outputValue[0] == '.' && outputValue[1] == '/'))
        {
            pathSeparator = "/";
        }

        // MDL is using leading slashes as separator
        return getName() + "(\"" + pathSeparator + path.asString(FilePath::FormatPosix) + "\", tex::gamma_linear)";
    }
};

class MdlArrayTypeSyntax : public ScalarTypeSyntax
{
  public:
    MdlArrayTypeSyntax(const string& name) :
        ScalarTypeSyntax(name, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING)
    {
    }

    string getValue(const Value& value, bool /*uniform*/) const override
    {
        if (!isEmpty(value))
        {
            return getName() + "[](" + value.getValueString() + ")";
        }
        return EMPTY_STRING;
    }

    string getValue(const StringVec& values, bool /*uniform*/) const override
    {
        if (values.empty())
        {
            throw ExceptionShaderGenError("No values given to construct an array value");
        }

        string result = getName() + "[](" + values[0];
        for (size_t i = 1; i < values.size(); ++i)
        {
            result += ", " + values[i];
        }
        result += ")";

        return result;
    }

  protected:
    virtual bool isEmpty(const Value& value) const = 0;
};

class MdlFloatArrayTypeSyntax : public MdlArrayTypeSyntax
{
  public:
    explicit MdlFloatArrayTypeSyntax(const string& name) :
        MdlArrayTypeSyntax(name)
    {
    }

  protected:
    bool isEmpty(const Value& value) const override
    {
        const vector<float>& valueArray = value.asA<vector<float>>();
        return valueArray.empty();
    }
};

class MdlIntegerArrayTypeSyntax : public MdlArrayTypeSyntax
{
  public:
    explicit MdlIntegerArrayTypeSyntax(const string& name) :
        MdlArrayTypeSyntax(name)
    {
    }

  protected:
    bool isEmpty(const Value& value) const override
    {
        const vector<int>& valueArray = value.asA<vector<int>>();
        return valueArray.empty();
    }
};

// For the color4 type we need even more specialization since it's a struct of a struct:
//
// struct color4 {
//    color rgb;
//    float a;
// }
//
class MdlColor4TypeSyntax : public AggregateTypeSyntax
{
  public:
    MdlColor4TypeSyntax() :
        AggregateTypeSyntax("color4", "mk_color4(0.0)", "mk_color4(0.0)",
                            EMPTY_STRING, EMPTY_STRING, MdlSyntax::COLOR4_MEMBERS)
    {
    }

    string getValue(const Value& value, bool /*uniform*/) const override
    {
        StringStream ss;

        // Set float format and precision for the stream
        const Value::FloatFormat fmt = Value::getFloatFormat();
        ss.setf(std::ios_base::fmtflags(
                    (fmt == Value::FloatFormatFixed ? std::ios_base::fixed : (fmt == Value::FloatFormatScientific ? std::ios_base::scientific : 0))),
                std::ios_base::floatfield);
        ss.precision(Value::getFloatPrecision());

        const Color4 c = value.asA<Color4>();
        ss << "mk_color4(" << c[0] << ", " << c[1] << ", " << c[2] << ", " << c[3] << ")";

        return ss.str();
    }

    string getValue(const StringVec& values, bool /*uniform*/) const override
    {
        size_t valueCount = values.size();
        if (valueCount == 4)
        {
            return "mk_color4(" + values[0] + ", " + values[1] + ", " + values[2] + ", " + values[3] + ")";
        }
        else if (valueCount == 2)
        {
            return "mk_color4(" + values[0] + ", " + values[1] + ")";
        }
        throw ExceptionShaderGenError("Incorrect number of values to construct a color4 value:" + std::to_string(valueCount));
    }
};

class MdlEnumSyntax : public AggregateTypeSyntax
{
  public:
    MdlEnumSyntax(const string& name, const string& defaultValue, const string& defaultUniformValue, const StringVec& members) :
        AggregateTypeSyntax(name, defaultValue, defaultUniformValue, EMPTY_STRING, EMPTY_STRING, members)
    {
    }

    string getValue(const Value& value, bool /*uniform*/) const override
    {
        return _name + "_" + value.getValueString();
    }
};

} // anonymous namespace

const string MdlSyntax::CONST_QUALIFIER = "";
const string MdlSyntax::UNIFORM_QUALIFIER = "uniform";
const string MdlSyntax::SOURCE_FILE_EXTENSION = ".mdl";
const StringVec MdlSyntax::VECTOR2_MEMBERS = { ".x", ".y" };
const StringVec MdlSyntax::VECTOR3_MEMBERS = { ".x", ".y", ".z" };
const StringVec MdlSyntax::VECTOR4_MEMBERS = { ".x", ".y", ".z", ".w" };
const StringVec MdlSyntax::COLOR3_MEMBERS = { ".x", ".y", ".z" };
const StringVec MdlSyntax::COLOR4_MEMBERS = { ".x", ".y", ".z", ".a" };

const StringVec MdlSyntax::ADDRESSMODE_MEMBERS = { "constant", "clamp", "periodic", "mirror" };
const StringVec MdlSyntax::COORDINATESPACE_MEMBERS = { "model", "object", "world" };
const StringVec MdlSyntax::FILTERLOOKUPMODE_MEMBERS = { "closest", "linear", "cubic" };
const StringVec MdlSyntax::FILTERTYPE_MEMBERS = { "box", "gaussian" };
const StringVec MdlSyntax::DISTRIBUTIONTYPE_MEMBERS = { "ggx" };
const StringVec MdlSyntax::SCATTER_MODE_MEMBERS = { "R", "T", "RT" };

//
// MdlSyntax methods
//

MdlSyntax::MdlSyntax()
{
    // Add in all reserved words and keywords in MDL
    registerReservedWords(
        { // Reserved words
          "annotation", "bool", "bool2", "bool3", "bool4", "break", "bsdf", "bsdf_measurement", "case", "cast", "color", "const",
          "continue", "default", "do", "double", "double2", "double2x2", "double2x3", "double3", "double3x2", "double3x3", "double3x4",
          "double4", "double4x3", "double4x4", "double4x2", "double2x4", "edf", "else", "enum", "export", "false", "float", "float2",
          "float2x2", "float2x3", "float3", "float3x2", "float3x3", "float3x4", "float4", "float4x3", "float4x4", "float4x2", "float2x4",
          "for", "hair_bsdf", "if", "import", "in", "int", "int2", "int3", "int4", "intensity_mode", "intensity_power", "intensity_radiant_exitance",
          "let", "light_profile", "material", "material_emission", "material_geometry", "material_surface", "material_volume", "mdl", "module",
          "package", "return", "string", "struct", "switch", "texture_2d", "texture_3d", "texture_cube", "texture_ptex", "true", "typedef", "uniform",
          "using", "varying", "vdf", "while",
          // Reserved for future use
          "auto", "catch", "char", "class", "const_cast", "delete", "dynamic_cast", "explicit", "extern", "external", "foreach", "friend", "goto",
          "graph", "half", "half2", "half2x2", "half2x3", "half3", "half3x2", "half3x3", "half3x4", "half4", "half4x3", "half4x4", "half4x2", "half2x4",
          "inline", "inout", "lambda", "long", "mutable", "namespace", "native", "new", "operator", "out", "phenomenon", "private", "protected", "public",
          "reinterpret_cast", "sampler", "shader", "short", "signed", "sizeof", "static", "static_cast", "technique", "template", "this", "throw", "try",
          "typeid", "typename", "union", "unsigned", "virtual", "void", "volatile", "wchar_t" });

    // Register restricted tokens in MDL
    StringMap tokens;
    tokens["\\b(_)"] = "u"; // Disallow names which begin with underscores
    registerInvalidTokens(tokens);

    //
    // Register type syntax handlers for each data type.
    //

    registerTypeSyntax(
        Type::FLOAT,
        std::make_shared<ScalarTypeSyntax>(
            "float",
            "0.0",
            "0.0"));

    registerTypeSyntax(
        Type::FLOATARRAY,
        std::make_shared<MdlFloatArrayTypeSyntax>(
            "float"));

    registerTypeSyntax(
        Type::INTEGER,
        std::make_shared<ScalarTypeSyntax>(
            "int",
            "0",
            "0"));

    registerTypeSyntax(
        Type::INTEGERARRAY,
        std::make_shared<MdlIntegerArrayTypeSyntax>(
            "int"));

    registerTypeSyntax(
        Type::BOOLEAN,
        std::make_shared<ScalarTypeSyntax>(
            "bool",
            "false",
            "false"));

    registerTypeSyntax(
        Type::COLOR3,
        std::make_shared<AggregateTypeSyntax>(
            "color",
            "color(0.0)",
            "color(0.0)",
            EMPTY_STRING,
            EMPTY_STRING,
            COLOR3_MEMBERS));

    registerTypeSyntax(
        Type::COLOR4,
        std::make_shared<MdlColor4TypeSyntax>());

    registerTypeSyntax(
        Type::VECTOR2,
        std::make_shared<AggregateTypeSyntax>(
            "float2",
            "float2(0.0)",
            "float2(0.0)",
            EMPTY_STRING,
            EMPTY_STRING,
            VECTOR2_MEMBERS));

    registerTypeSyntax(
        Type::VECTOR3,
        std::make_shared<AggregateTypeSyntax>(
            "float3",
            "float3(0.0)",
            "float3(0.0)",
            EMPTY_STRING,
            EMPTY_STRING,
            VECTOR3_MEMBERS));

    registerTypeSyntax(
        Type::VECTOR4,
        std::make_shared<AggregateTypeSyntax>(
            "float4",
            "float4(0.0)",
            "float4(0.0)",
            EMPTY_STRING,
            EMPTY_STRING,
            VECTOR4_MEMBERS));

    registerTypeSyntax(
        Type::MATRIX33,
        std::make_shared<AggregateTypeSyntax>(
            "float3x3",
            "float3x3(1.0)",
            "float3x3(1.0)"));

    registerTypeSyntax(
        Type::MATRIX44,
        std::make_shared<AggregateTypeSyntax>(
            "float4x4",
            "float4x4(1.0)",
            "float4x4(1.0)"));

    registerTypeSyntax(
        Type::STRING,
        std::make_shared<StringTypeSyntax>(
            "string",
            "\"\"",
            "\"\""));

    registerTypeSyntax(
        Type::FILENAME,
        std::make_shared<MdlFilenameTypeSyntax>());

    registerTypeSyntax(
        Type::BSDF,
        std::make_shared<ScalarTypeSyntax>(
            "material",
            "material()",
            "material()"));

    registerTypeSyntax(
        Type::EDF,
        std::make_shared<ScalarTypeSyntax>(
            "material",
            "material()",
            "material()"));

    registerTypeSyntax(
        Type::VDF,
        std::make_shared<ScalarTypeSyntax>(
            "material",
            "material()",
            "material()"));

    registerTypeSyntax(
        Type::SURFACESHADER,
        std::make_shared<ScalarTypeSyntax>(
            "material",
            "material()",
            "material()"));

    registerTypeSyntax(
        Type::VOLUMESHADER,
        std::make_shared<ScalarTypeSyntax>(
            "material",
            "material()",
            "material()"));

    registerTypeSyntax(
        Type::DISPLACEMENTSHADER,
        std::make_shared<ScalarTypeSyntax>(
            "material",
            "material()",
            "material()"));

    registerTypeSyntax(
        Type::LIGHTSHADER,
        std::make_shared<ScalarTypeSyntax>(
            "material",
            "material()",
            "material()"));

    registerTypeSyntax(
        Type::MATERIAL,
        std::make_shared<ScalarTypeSyntax>(
            "material",
            "material()",
            "material()"));

    registerTypeSyntax(
        Type::MDL_ADDRESSMODE,
        std::make_shared<MdlEnumSyntax>(
            "mx_addressmode_type",
            "mx_addressmode_type_periodic",
            "mx_addressmode_type_periodic",
            ADDRESSMODE_MEMBERS));

    registerTypeSyntax(
        Type::MDL_COORDINATESPACE,
        std::make_shared<MdlEnumSyntax>(
            "mx_coordinatespace_type",
            "mx_coordinatespace_type_model",
            "mx_coordinatespace_type_model",
            COORDINATESPACE_MEMBERS));

    registerTypeSyntax(
        Type::MDL_FILTERLOOKUPMODE,
        std::make_shared<MdlEnumSyntax>(
            "mx_filterlookup_type",
            "mx_filterlookup_type_linear",
            "mx_filterlookup_type_linear",
            FILTERLOOKUPMODE_MEMBERS));

    registerTypeSyntax(
        Type::MDL_FILTERTYPE,
        std::make_shared<MdlEnumSyntax>(
            "mx_filter_type",
            "mx_filter_type_gaussian",
            "mx_filter_type_gaussian",
            FILTERTYPE_MEMBERS));

    registerTypeSyntax(
        Type::MDL_DISTRIBUTION,
        std::make_shared<MdlEnumSyntax>(
            "mx_distribution_type",
            "mx_distribution_type_ggx",
            "mx_distribution_type_ggx",
            DISTRIBUTIONTYPE_MEMBERS));

    registerTypeSyntax(
        Type::MDL_SCATTER_MODE,
        std::make_shared<MdlEnumSyntax>(
            "mx_scatter_mode",
            "mx_scatter_mode_R",
            "mx_scatter_mode_R",
            SCATTER_MODE_MEMBERS));
}

const TypeDesc* MdlSyntax::getEnumeratedType(const string& value) const
{
    for (const TypeSyntaxPtr& syntax : getTypeSyntaxes())
    {
        const StringVec& members = syntax->getMembers();
        if (members.size())
        {
            // TODO: This logic assumes the enum values are strictly unique among all enum types.
            // We should find a more safe way to handled this.
            if (std::find(members.begin(), members.end(), value) != members.end())
            {
                return getTypeDescription(syntax);
            }
        }
    }
    return nullptr;
}

const std::unordered_map<char, char> CHANNELS_TO_XYZW =
{
    { 'r', 'x' }, { 'x', 'x' },
    { 'g', 'y' }, { 'y', 'y' },
    { 'b', 'z' }, { 'z', 'z' },
    { 'a', 'w' }, { 'w', 'w' }
};

string MdlSyntax::getSwizzledVariable(const string& srcName, const TypeDesc* srcType, const string& channels, const TypeDesc* dstType) const
{
    if (*srcType == *Type::COLOR3 || *srcType == *Type::COLOR4)
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

            int channelIndex = srcType->getChannelIndex(ch);
            if (channelIndex < 0 || channelIndex >= static_cast<int>(srcMembers.size()))
            {
                throw ExceptionShaderGenError("Given channel index: '" + string(1, ch) + "' in channels pattern is incorrect for type '" + srcType->getName() + "'.");
            }

            string variable = srcName;
            if (*srcType == *Type::COLOR3)
            {
                variable = "float3(" + srcName + ")";
            }
            else if (channelIndex < 3)
            {
                variable = "float3(" + srcName + ".rgb)";
            }

            membersSwizzled.push_back(variable + srcMembers[channelIndex]);
        }

        return dstSyntax.getValue(membersSwizzled, false);
    }
    return Syntax::getSwizzledVariable(srcName, srcType, channels, dstType);
}

string MdlSyntax::getArrayTypeSuffix(const TypeDesc* type, const Value& value) const
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

bool MdlSyntax::remapEnumeration(const string& value, const TypeDesc* type, const string& enumNames, std::pair<const TypeDesc*, ValuePtr>& result) const
{
    // Early out if not an enum input.
    if (enumNames.empty())
    {
        return false;
    }

    // Don't convert filenames or arrays.
    if (*type == *Type::FILENAME || (type && type->isArray()))
    {
        return false;
    }

    // Try remapping to an enum value.
    if (!value.empty())
    {
        result.first = getEnumeratedType(value);
        if (!result.first || (result.first->getSemantic() != TypeDesc::Semantic::SEMANTIC_ENUM))
        {
            return false;
        }

        StringVec valueElemEnumsVec = splitString(enumNames, ",");
        for (size_t i = 0; i < valueElemEnumsVec.size(); i++)
        {
            valueElemEnumsVec[i] = trimSpaces(valueElemEnumsVec[i]);
        }
        auto pos = std::find(valueElemEnumsVec.begin(), valueElemEnumsVec.end(), value);
        if (pos == valueElemEnumsVec.end())
        {
            throw ExceptionShaderGenError("Given value '" + value + "' is not a valid enum value.");
        }
        const int index = static_cast<int>(std::distance(valueElemEnumsVec.begin(), pos));
        result.second = Value::createValue<string>(valueElemEnumsVec[index]);

        return true;
    }

    return false;
}

void MdlSyntax::makeValidName(string& name) const
{
    Syntax::makeValidName(name);

    // MDL variables are not allowed to begin with underscore.
    if (!name.empty() && name[0] == '_')
    {
        name = "v" + name;
    }
}

MATERIALX_NAMESPACE_END
