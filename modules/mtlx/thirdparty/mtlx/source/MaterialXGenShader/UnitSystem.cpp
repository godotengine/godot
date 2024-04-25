//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenShader/UnitSystem.h>

#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/ShaderStage.h>
#include <MaterialXGenShader/Shader.h>
#include <MaterialXGenShader/Nodes/SourceCodeNode.h>

MATERIALX_NAMESPACE_BEGIN

class ScalarUnitNode : public ShaderNodeImpl
{
  public:
    explicit ScalarUnitNode(LinearUnitConverterPtr scalarUnitConverter) :
        _scalarUnitConverter(scalarUnitConverter),
        _unitRatioFunctionName("mx_" + _scalarUnitConverter->getUnitType() + "_unit_ratio")
    {
    }

    static ShaderNodeImplPtr create(LinearUnitConverterPtr scalarUnitConverter);

    void initialize(const InterfaceElement& element, GenContext& context) override;
    void emitFunctionDefinition(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;
    void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;

  protected:
    LinearUnitConverterPtr _scalarUnitConverter;
    const string _unitRatioFunctionName;
};

ShaderNodeImplPtr ScalarUnitNode::create(LinearUnitConverterPtr scalarUnitConverter)
{
    return std::make_shared<ScalarUnitNode>(scalarUnitConverter);
}

void ScalarUnitNode::initialize(const InterfaceElement& element, GenContext& /*context*/)
{
    _name = element.getName();

    // Use the unit ratio function name has hash to make sure this function
    // is shared, and only emitted once, for all units of the same unit type.
    _hash = std::hash<string>{}(_unitRatioFunctionName);
}

void ScalarUnitNode::emitFunctionDefinition(const ShaderNode& node, GenContext& context, ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        // Emit the helper funtion mx_<unittype>_unit_ratio that embeds a look up table for unit scale
        vector<float> unitScales;
        unitScales.reserve(_scalarUnitConverter->getUnitScale().size());
        auto unitScaleMap = _scalarUnitConverter->getUnitScale();
        unitScales.resize(unitScaleMap.size());
        for (auto unitScale : unitScaleMap)
        {
            int location = _scalarUnitConverter->getUnitAsInteger(unitScale.first);
            unitScales[location] = unitScale.second;
        }
        // See stdlib/gen*/mx_<unittype>_unit. This helper function is called by these shaders.
        const string VAR_UNIT_SCALE = "u_" + _scalarUnitConverter->getUnitType() + "_unit_scales";
        VariableBlock unitLUT("unitLUT", EMPTY_STRING);
        ScopedFloatFormatting fmt(Value::FloatFormatFixed, 15);
        unitLUT.add(Type::FLOATARRAY, VAR_UNIT_SCALE, Value::createValue<vector<float>>(unitScales));

        const ShaderGenerator& shadergen = context.getShaderGenerator();
        shadergen.emitLine("float " + _unitRatioFunctionName + "(int unit_from, int unit_to)", stage, false);
        shadergen.emitFunctionBodyBegin(node, context, stage);
        shadergen.emitVariableDeclarations(unitLUT, shadergen.getSyntax().getConstantQualifier(), ";", context, stage, true);
        shadergen.emitLine("return (" + VAR_UNIT_SCALE + "[unit_from] / " + VAR_UNIT_SCALE + "[unit_to])", stage);
        shadergen.emitFunctionBodyEnd(node, context, stage);
    }
}

void ScalarUnitNode::emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const
{
    DEFINE_SHADER_STAGE(stage, Stage::PIXEL)
    {
        const ShaderGenerator& shadergen = context.getShaderGenerator();

        const ShaderInput* in = node.getInput(0);
        const ShaderInput* from = node.getInput(1);
        const ShaderInput* to = node.getInput(2);

        shadergen.emitLineBegin(stage);
        shadergen.emitOutput(node.getOutput(), true, false, context, stage);
        shadergen.emitString(" = ", stage);
        shadergen.emitInput(in, context, stage);
        shadergen.emitString(" * ", stage);
        shadergen.emitString(_unitRatioFunctionName + "(", stage);
        shadergen.emitInput(from, context, stage);
        shadergen.emitString(", ", stage);
        shadergen.emitInput(to, context, stage);
        shadergen.emitString(")", stage);
        shadergen.emitLineEnd(stage);
    }
}

//
// Unit transform methods
//

UnitTransform::UnitTransform(const string& ss, const string& ts, const TypeDesc* t, const string& unittype) :
    sourceUnit(ss),
    targetUnit(ts),
    type(t),
    unitType(unittype)
{
    if (type != Type::FLOAT && type != Type::VECTOR2 && type != Type::VECTOR3 && type != Type::VECTOR4)
    {
        throw ExceptionShaderGenError("Unit space transform can only be a float or vectors");
    }
}

const string UnitSystem::UNITSYTEM_NAME = "default_unit_system";

UnitSystem::UnitSystem(const string& target) :
    _target(createValidName(target))
{
}

void UnitSystem::loadLibrary(DocumentPtr document)
{
    _document = document;
}

void UnitSystem::setUnitConverterRegistry(UnitConverterRegistryPtr registry)
{
    _unitRegistry = registry;
}

UnitConverterRegistryPtr UnitSystem::getUnitConverterRegistry() const
{
    return _unitRegistry;
}

UnitSystemPtr UnitSystem::create(const string& language)
{
    return UnitSystemPtr(new UnitSystem(language));
}

NodeDefPtr UnitSystem::getNodeDef(const UnitTransform& transform) const
{
    if (!_document)
    {
        throw ExceptionShaderGenError("No library loaded for unit system");
    }

    const string MULTIPLY_NODE_NAME = "multiply";
    for (NodeDefPtr nodeDef : _document->getMatchingNodeDefs(MULTIPLY_NODE_NAME))
    {
        for (OutputPtr output : nodeDef->getOutputs())
        {
            vector<InputPtr> nodeInputs = nodeDef->getInputs();
            if (nodeInputs.size() == 2 &&
                nodeInputs[0]->getType() == transform.type->getName() &&
                nodeInputs[1]->getType() == "float")
            {
                return nodeDef;
            }
        }
    }
    return nullptr;
}

bool UnitSystem::supportsTransform(const UnitTransform& transform) const
{
    NodeDefPtr nodeDef = getNodeDef(transform);
    return nodeDef != nullptr;
}

ShaderNodePtr UnitSystem::createNode(ShaderGraph* parent, const UnitTransform& transform, const string& name,
                                     GenContext& context) const
{
    NodeDefPtr nodeDef = getNodeDef(transform);
    if (!nodeDef)
    {
        throw ExceptionShaderGenError("No nodedef found for transform: ('" + transform.sourceUnit + "', '" + transform.targetUnit + "').");
    }

    // Scalar unit conversion
    UnitTypeDefPtr scalarTypeDef = _document->getUnitTypeDef(transform.unitType);
    if (!_unitRegistry || !_unitRegistry->getUnitConverter(scalarTypeDef))
    {
        throw ExceptionTypeError("Unit registry unavaliable or undefined unit converter for: " + transform.unitType);
    }
    LinearUnitConverterPtr scalarConverter = std::dynamic_pointer_cast<LinearUnitConverter>(_unitRegistry->getUnitConverter(scalarTypeDef));

    // Create the node.
    ShaderNodePtr shaderNode = ShaderNode::create(parent, name, *nodeDef, context);

    // Set ports on the node.
    ShaderInput* in2 = shaderNode->getInput("in2");
    if (!in2)
    {
        throw ExceptionShaderGenError("Invalid node signature for unit transform: ('" + transform.sourceUnit + "', '" + transform.targetUnit + "').");
    }
    float conversionRatio = scalarConverter->conversionRatio(transform.sourceUnit, transform.targetUnit);
    in2->setValue(Value::createValue(conversionRatio));

    return shaderNode;
}

MATERIALX_NAMESPACE_END
