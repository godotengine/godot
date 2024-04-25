//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenGlsl/GlslSyntax.h>

#include <MaterialXGenShader/ShaderGenerator.h>

MATERIALX_NAMESPACE_BEGIN

namespace
{

// Since GLSL doesn't support strings we use integers instead.
// TODO: Support options strings by converting to a corresponding enum integer
class GlslStringTypeSyntax : public StringTypeSyntax
{
  public:
    GlslStringTypeSyntax() :
        StringTypeSyntax("int", "0", "0") { }

    string getValue(const Value& /*value*/, bool /*uniform*/) const override
    {
        return "0";
    }
};

class GlslArrayTypeSyntax : public ScalarTypeSyntax
{
  public:
    GlslArrayTypeSyntax(const string& name) :
        ScalarTypeSyntax(name, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING)
    {
    }

    string getValue(const Value& value, bool /*uniform*/) const override
    {
        size_t arraySize = getSize(value);
        if (arraySize > 0)
        {
            return _name + "[" + std::to_string(arraySize) + "](" + value.getValueString() + ")";
        }
        return EMPTY_STRING;
    }

    string getValue(const StringVec& values, bool /*uniform*/) const override
    {
        if (values.empty())
        {
            throw ExceptionShaderGenError("No values given to construct an array value");
        }

        string result = _name + "[" + std::to_string(values.size()) + "](" + values[0];
        for (size_t i = 1; i < values.size(); ++i)
        {
            result += ", " + values[i];
        }
        result += ")";

        return result;
    }

  protected:
    virtual size_t getSize(const Value& value) const = 0;
};

class GlslFloatArrayTypeSyntax : public GlslArrayTypeSyntax
{
  public:
    explicit GlslFloatArrayTypeSyntax(const string& name) :
        GlslArrayTypeSyntax(name)
    {
    }

  protected:
    size_t getSize(const Value& value) const override
    {
        vector<float> valueArray = value.asA<vector<float>>();
        return valueArray.size();
    }
};

class GlslIntegerArrayTypeSyntax : public GlslArrayTypeSyntax
{
  public:
    explicit GlslIntegerArrayTypeSyntax(const string& name) :
        GlslArrayTypeSyntax(name)
    {
    }

  protected:
    size_t getSize(const Value& value) const override
    {
        vector<int> valueArray = value.asA<vector<int>>();
        return valueArray.size();
    }
};

} // anonymous namespace

const string GlslSyntax::INPUT_QUALIFIER = "in";
const string GlslSyntax::OUTPUT_QUALIFIER = "out";
const string GlslSyntax::UNIFORM_QUALIFIER = "uniform";
const string GlslSyntax::CONSTANT_QUALIFIER = "const";
const string GlslSyntax::FLAT_QUALIFIER = "flat";
const string GlslSyntax::SOURCE_FILE_EXTENSION = ".glsl";
const StringVec GlslSyntax::VEC2_MEMBERS = { ".x", ".y" };
const StringVec GlslSyntax::VEC3_MEMBERS = { ".x", ".y", ".z" };
const StringVec GlslSyntax::VEC4_MEMBERS = { ".x", ".y", ".z", ".w" };

//
// GlslSyntax methods
//

GlslSyntax::GlslSyntax()
{
    // Add in all reserved words and keywords in GLSL
    registerReservedWords(
        { "centroid", "flat", "smooth", "noperspective", "patch", "sample",
          "break", "continue", "do", "for", "while", "switch", "case", "default",
          "if", "else,", "subroutine", "in", "out", "inout",
          "float", "double", "int", "void", "bool", "true", "false",
          "invariant", "discard", "return",
          "mat2", "mat3", "mat4", "dmat2", "dmat3", "dmat4",
          "mat2x2", "mat2x3", "mat2x4", "dmat2x2", "dmat2x3", "dmat2x4",
          "mat3x2", "mat3x3", "mat3x4", "dmat3x2", "dmat3x3", "dmat3x4",
          "mat4x2", "mat4x3", "mat4x4", "dmat4x2", "dmat4x3", "dmat4x4",
          "vec2", "vec3", "vec4", "ivec2", "ivec3", "ivec4", "bvec2", "bvec3", "bvec4", "dvec2", "dvec3", "dvec4",
          "uint", "uvec2", "uvec3", "uvec4",
          "lowp", "mediump", "highp", "precision",
          "sampler1D", "sampler2D", "sampler3D", "samplerCube",
          "sampler1DShadow", "sampler2DShadow", "samplerCubeShadow",
          "sampler1DArray", "sampler2DArray",
          "sampler1DArrayShadow", "sampler2DArrayShadow",
          "isampler1D", "isampler2D", "isampler3D", "isamplerCube",
          "isampler1DArray", "isampler2DArray",
          "usampler1D", "usampler2D", "usampler3D", "usamplerCube",
          "usampler1DArray", "usampler2DArray",
          "sampler2DRect", "sampler2DRectShadow", "isampler2DRect", "usampler2DRect",
          "samplerBuffer", "isamplerBuffer", "usamplerBuffer",
          "sampler2DMS", "isampler2DMS", "usampler2DMS",
          "sampler2DMSArray", "isampler2DMSArray", "usampler2DMSArray",
          "samplerCubeArray", "samplerCubeArrayShadow", "isamplerCubeArray", "usamplerCubeArray",
          "common", "partition", "active", "asm",
          "struct", "class", "union", "enum", "typedef", "template", "this", "packed", "goto",
          "inline", "noinline", "volatile", "public", "static", "extern", "external", "interface",
          "long", "short", "half", "fixed", "unsigned", "superp", "input", "output",
          "hvec2", "hvec3", "hvec4", "fvec2", "fvec3", "fvec4",
          "sampler3DRect", "filter",
          "image1D", "image2D", "image3D", "imageCube",
          "iimage1D", "iimage2D", "iimage3D", "iimageCube",
          "uimage1D", "uimage2D", "uimage3D", "uimageCube",
          "image1DArray", "image2DArray",
          "iimage1DArray", "iimage2DArray", "uimage1DArray", "uimage2DArray",
          "image1DShadow", "image2DShadow",
          "image1DArrayShadow", "image2DArrayShadow",
          "imageBuffer", "iimageBuffer", "uimageBuffer",
          "sizeof", "cast", "namespace", "using", "row_major",
          "mix", "sampler" });

    // Register restricted tokens in GLSL
    StringMap tokens;
    tokens["__"] = "_";
    tokens["gl_"] = "gll";
    tokens["webgl_"] = "webgll";
    tokens["_webgl"] = "wwebgl";
    registerInvalidTokens(tokens);

    //
    // Register syntax handlers for each data type.
    //

    registerTypeSyntax(
        Type::FLOAT,
        std::make_shared<ScalarTypeSyntax>(
            "float",
            "0.0",
            "0.0"));

    registerTypeSyntax(
        Type::FLOATARRAY,
        std::make_shared<GlslFloatArrayTypeSyntax>(
            "float"));

    registerTypeSyntax(
        Type::INTEGER,
        std::make_shared<ScalarTypeSyntax>(
            "int",
            "0",
            "0"));

    registerTypeSyntax(
        Type::INTEGERARRAY,
        std::make_shared<GlslIntegerArrayTypeSyntax>(
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
            "vec3",
            "vec3(0.0)",
            "vec3(0.0)",
            EMPTY_STRING,
            EMPTY_STRING,
            VEC3_MEMBERS));

    registerTypeSyntax(
        Type::COLOR4,
        std::make_shared<AggregateTypeSyntax>(
            "vec4",
            "vec4(0.0)",
            "vec4(0.0)",
            EMPTY_STRING,
            EMPTY_STRING,
            VEC4_MEMBERS));

    registerTypeSyntax(
        Type::VECTOR2,
        std::make_shared<AggregateTypeSyntax>(
            "vec2",
            "vec2(0.0)",
            "vec2(0.0)",
            EMPTY_STRING,
            EMPTY_STRING,
            VEC2_MEMBERS));

    registerTypeSyntax(
        Type::VECTOR3,
        std::make_shared<AggregateTypeSyntax>(
            "vec3",
            "vec3(0.0)",
            "vec3(0.0)",
            EMPTY_STRING,
            EMPTY_STRING,
            VEC3_MEMBERS));

    registerTypeSyntax(
        Type::VECTOR4,
        std::make_shared<AggregateTypeSyntax>(
            "vec4",
            "vec4(0.0)",
            "vec4(0.0)",
            EMPTY_STRING,
            EMPTY_STRING,
            VEC4_MEMBERS));

    registerTypeSyntax(
        Type::MATRIX33,
        std::make_shared<AggregateTypeSyntax>(
            "mat3",
            "mat3(1.0)",
            "mat3(1.0)"));

    registerTypeSyntax(
        Type::MATRIX44,
        std::make_shared<AggregateTypeSyntax>(
            "mat4",
            "mat4(1.0)",
            "mat4(1.0)"));

    registerTypeSyntax(
        Type::STRING,
        std::make_shared<GlslStringTypeSyntax>());

    registerTypeSyntax(
        Type::FILENAME,
        std::make_shared<ScalarTypeSyntax>(
            "sampler2D",
            EMPTY_STRING,
            EMPTY_STRING));

    registerTypeSyntax(
        Type::BSDF,
        std::make_shared<AggregateTypeSyntax>(
            "BSDF",
            "BSDF(vec3(0.0),vec3(1.0), 0.0, 0.0)",
            EMPTY_STRING,
            EMPTY_STRING,
            "struct BSDF { vec3 response; vec3 throughput; float thickness; float ior; };"));

    registerTypeSyntax(
        Type::EDF,
        std::make_shared<AggregateTypeSyntax>(
            "EDF",
            "EDF(0.0)",
            "EDF(0.0)",
            "vec3",
            "#define EDF vec3"));

    registerTypeSyntax(
        Type::VDF,
        std::make_shared<AggregateTypeSyntax>(
            "BSDF",
            "BSDF(vec3(0.0),vec3(1.0), 0.0, 0.0)",
            EMPTY_STRING));

    registerTypeSyntax(
        Type::SURFACESHADER,
        std::make_shared<AggregateTypeSyntax>(
            "surfaceshader",
            "surfaceshader(vec3(0.0),vec3(0.0))",
            EMPTY_STRING,
            EMPTY_STRING,
            "struct surfaceshader { vec3 color; vec3 transparency; };"));

    registerTypeSyntax(
        Type::VOLUMESHADER,
        std::make_shared<AggregateTypeSyntax>(
            "volumeshader",
            "volumeshader(vec3(0.0),vec3(0.0))",
            EMPTY_STRING,
            EMPTY_STRING,
            "struct volumeshader { vec3 color; vec3 transparency; };"));

    registerTypeSyntax(
        Type::DISPLACEMENTSHADER,
        std::make_shared<AggregateTypeSyntax>(
            "displacementshader",
            "displacementshader(vec3(0.0),1.0)",
            EMPTY_STRING,
            EMPTY_STRING,
            "struct displacementshader { vec3 offset; float scale; };"));

    registerTypeSyntax(
        Type::LIGHTSHADER,
        std::make_shared<AggregateTypeSyntax>(
            "lightshader",
            "lightshader(vec3(0.0),vec3(0.0))",
            EMPTY_STRING,
            EMPTY_STRING,
            "struct lightshader { vec3 intensity; vec3 direction; };"));

    registerTypeSyntax(
        Type::MATERIAL,
        std::make_shared<AggregateTypeSyntax>(
            "material",
            "material(vec3(0.0),vec3(0.0))",
            EMPTY_STRING,
            "surfaceshader",
            "#define material surfaceshader"));
}

bool GlslSyntax::typeSupported(const TypeDesc* type) const
{
    return type != Type::STRING;
}

bool GlslSyntax::remapEnumeration(const string& value, const TypeDesc* type, const string& enumNames, std::pair<const TypeDesc*, ValuePtr>& result) const
{
    // Early out if not an enum input.
    if (enumNames.empty())
    {
        return false;
    }

    // Don't convert already supported types
    // or filenames and arrays.
    if (typeSupported(type) ||
        *type == *Type::FILENAME || (type && type->isArray()))
    {
        return false;
    }

    // For GLSL we always convert to integer,
    // with the integer value being an index into the enumeration.
    result.first = Type::INTEGER;
    result.second = nullptr;

    // Try remapping to an enum value.
    if (!value.empty())
    {
        StringVec valueElemEnumsVec = splitString(enumNames, ",");
        for (size_t i = 0; i < valueElemEnumsVec.size(); i++)
        {
            valueElemEnumsVec[i] = trimSpaces(valueElemEnumsVec[i]);
        }
        auto pos = std::find(valueElemEnumsVec.begin(), valueElemEnumsVec.end(), value);
        if (pos == valueElemEnumsVec.end())
        {
            throw ExceptionShaderGenError("Given value '" + value + "' is not a valid enum value for input.");
        }
        const int index = static_cast<int>(std::distance(valueElemEnumsVec.begin(), pos));
        result.second = Value::createValue<int>(index);
    }

    return true;
}

MATERIALX_NAMESPACE_END
