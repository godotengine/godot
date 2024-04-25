//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXRender/Util.h>

#include <MaterialXGenShader/ShaderGenerator.h>

MATERIALX_NAMESPACE_BEGIN

const Color3 DEFAULT_SCREEN_COLOR_SRGB(0.3f, 0.3f, 0.32f);
const Color3 DEFAULT_SCREEN_COLOR_LIN_REC709(DEFAULT_SCREEN_COLOR_SRGB.srgbToLinear());

ShaderPtr createShader(const string& shaderName, GenContext& context, ElementPtr elem)
{
    return context.getShaderGenerator().generate(shaderName, elem, context);
}

ShaderPtr createConstantShader(GenContext& context,
                               DocumentPtr stdLib,
                               const string& shaderName,
                               const Color3& color)
{
    // Construct the constant color nodegraph
    DocumentPtr doc = createDocument();
    doc->importLibrary(stdLib);
    NodeGraphPtr nodeGraph = doc->addNodeGraph();
    NodePtr constant = nodeGraph->addNode("constant");
    constant->setInputValue("value", color);
    OutputPtr output = nodeGraph->addOutput();
    output->setConnectedNode(constant);

    // Generate the shader
    return createShader(shaderName, context, output);
}

ShaderPtr createDepthShader(GenContext& context,
                            DocumentPtr stdLib,
                            const string& shaderName)
{
    // Construct a dummy nodegraph.
    DocumentPtr doc = createDocument();
    doc->importLibrary(stdLib);
    NodeGraphPtr nodeGraph = doc->addNodeGraph();
    NodePtr constant = nodeGraph->addNode("constant");
    OutputPtr output = nodeGraph->addOutput();
    output->setConnectedNode(constant);

    // Generate the shader
    GenContext depthContext = context;
    depthContext.getOptions().hwWriteDepthMoments = true;
    ShaderPtr shader = createShader(shaderName, depthContext, output);

    return shader;
}

ShaderPtr createAlbedoTableShader(GenContext& context,
                                  DocumentPtr stdLib,
                                  const string& shaderName)
{
    // Construct a dummy nodegraph.
    DocumentPtr doc = createDocument();
    doc->importLibrary(stdLib);
    NodeGraphPtr nodeGraph = doc->addNodeGraph();
    NodePtr constant = nodeGraph->addNode("constant");
    OutputPtr output = nodeGraph->addOutput();
    output->setConnectedNode(constant);

    // Generate the shader
    GenContext tableContext = context;
    tableContext.getOptions().hwWriteAlbedoTable = true;
    tableContext.getOptions().hwDirectionalAlbedoMethod = DIRECTIONAL_ALBEDO_MONTE_CARLO;
    ShaderPtr shader = createShader(shaderName, tableContext, output);

    return shader;
}

ShaderPtr createEnvPrefilterShader(GenContext& context,
                                        DocumentPtr stdLib,
                                        const string& shaderName)
{
    // Construct a dummy nodegraph.
    DocumentPtr doc = createDocument();
    doc->importLibrary(stdLib);
    NodeGraphPtr nodeGraph = doc->addNodeGraph();
    NodePtr constant = nodeGraph->addNode("constant");
    OutputPtr output = nodeGraph->addOutput();
    output->setConnectedNode(constant);

    // Generate the shader
    GenContext tableContext = context;
    tableContext.getOptions().hwWriteEnvPrefilter = true;
    ShaderPtr shader = createShader(shaderName, tableContext, output);

    return shader;
}

ShaderPtr createBlurShader(GenContext& context,
                           DocumentPtr stdLib,
                           const string& shaderName,
                           const string& filterType,
                           float filterSize)
{
    // Construct the blur nodegraph
    DocumentPtr doc = createDocument();
    doc->importLibrary(stdLib);
    NodeGraphPtr nodeGraph = doc->addNodeGraph();
    NodePtr imageNode = nodeGraph->addNode("image", "image");
    NodePtr blurNode = nodeGraph->addNode("blur", "blur");
    blurNode->setConnectedNode("in", imageNode);
    blurNode->setInputValue("size", filterSize);
    blurNode->setInputValue("filtertype", filterType);
    OutputPtr output = nodeGraph->addOutput();
    output->setConnectedNode(blurNode);

    // Generate the shader
    GenContext blurContext = context;
    blurContext.getOptions().fileTextureVerticalFlip = false;
    return createShader(shaderName, blurContext, output);
}

unsigned int getUIProperties(InputPtr input, const string& target, UIProperties& uiProperties)
{
    if (!input)
    {
        return 0;
    }
    InputPtr nodeDefInput = getNodeDefInput(input, target);
    if (nodeDefInput)
    {
        input = nodeDefInput;
    }

    unsigned int propertyCount = 0;
    uiProperties.uiName = input->getAttribute(ValueElement::UI_NAME_ATTRIBUTE);
    if (!uiProperties.uiName.empty())
    {
        propertyCount++;
    }

    uiProperties.uiFolder = input->getAttribute(ValueElement::UI_FOLDER_ATTRIBUTE);
    if (!uiProperties.uiFolder.empty())
    {
        propertyCount++;
    }

    if (input->getIsUniform())
    {
        uiProperties.enumeration = input->getTypedAttribute<StringVec>(ValueElement::ENUM_ATTRIBUTE);
        if (!uiProperties.enumeration.empty())
        {
            propertyCount++;
        }

        const string& enumValuesAttr = input->getAttribute(ValueElement::ENUM_VALUES_ATTRIBUTE);
        if (!enumValuesAttr.empty())
        {
            const string COMMA_SEPARATOR = ",";
            const TypeDesc* typeDesc = TypeDesc::get(input->getType());
            string valueString;
            size_t index = 0;
            for (const string& val : splitString(enumValuesAttr, COMMA_SEPARATOR))
            {
                if (index < typeDesc->getSize() - 1)
                {
                    valueString += val + COMMA_SEPARATOR;
                    index++;
                }
                else
                {
                    valueString += val;
                    uiProperties.enumerationValues.push_back(Value::createValueFromStrings(valueString, input->getType()));
                    valueString.clear();
                    index = 0;
                }
            }
            if (uiProperties.enumeration.size() != uiProperties.enumerationValues.size())
            {
                throw std::runtime_error("Every enum must have a value!");
            }
            propertyCount++;
        }
    }

    const string& uiMinString = input->getAttribute(ValueElement::UI_MIN_ATTRIBUTE);
    if (!uiMinString.empty())
    {
        ValuePtr value = Value::createValueFromStrings(uiMinString, input->getType());
        if (value)
        {
            uiProperties.uiMin = value;
            propertyCount++;
        }
    }

    const string& uiMaxString = input->getAttribute(ValueElement::UI_MAX_ATTRIBUTE);
    if (!uiMaxString.empty())
    {
        ValuePtr value = Value::createValueFromStrings(uiMaxString, input->getType());
        if (value)
        {
            uiProperties.uiMax = value;
            propertyCount++;
        }
    }

    const string& uiSoftMinString = input->getAttribute(ValueElement::UI_SOFT_MIN_ATTRIBUTE);
    if (!uiSoftMinString.empty())
    {
        ValuePtr value = Value::createValueFromStrings(uiSoftMinString, input->getType());
        if (value)
        {
            uiProperties.uiSoftMin = value;
            propertyCount++;
        }
    }

    const string& uiSoftMaxString = input->getAttribute(ValueElement::UI_SOFT_MAX_ATTRIBUTE);
    if (!uiSoftMaxString.empty())
    {
        ValuePtr value = Value::createValueFromStrings(uiSoftMaxString, input->getType());
        if (value)
        {
            uiProperties.uiSoftMax = value;
            propertyCount++;
        }
    }

    const string& uiStepString = input->getAttribute(ValueElement::UI_STEP_ATTRIBUTE);
    if (!uiStepString.empty())
    {
        ValuePtr value = Value::createValueFromStrings(uiStepString, input->getType());
        if (value)
        {
            uiProperties.uiStep = value;
            propertyCount++;
        }
    }

    const string& uiAdvancedString = input->getAttribute(ValueElement::UI_ADVANCED_ATTRIBUTE);
    uiProperties.uiAdvanced = (uiAdvancedString == "true");
    if (!uiAdvancedString.empty())
    {
        propertyCount++;
    }

    return propertyCount;
}

void createUIPropertyGroups(DocumentPtr doc, const VariableBlock& block, UIPropertyGroup& groups,
                            UIPropertyGroup& unnamedGroups, const string& pathSeparator)
{
    // Assign a depth-first index to each element in the document.
    std::unordered_map<ConstElementPtr, int> indexMap;
    int curIndex = 0;
    for (ConstElementPtr elem : doc->traverseTree())
    {
        indexMap[elem] = curIndex++;
    }

    // Generated an ordered map of shader inputs.
    using ShaderInputPair = std::pair<InputPtr, ShaderPort*>;
    std::map<int, ShaderInputPair> shaderInputMap;
    for (ShaderPort* variable : block.getVariableOrder())
    {
        if (!variable->getValue())
        {
            continue;
        }

        // Get the input associated with this variable.
        ElementPtr pathElement = doc->getDescendant(variable->getPath());
        InputPtr input = pathElement ? pathElement->asA<Input>() : nullptr;

        // Redirect to interface inputs when present.
        if (input)
        {
            InputPtr interfaceInput = input->getInterfaceInput();
            if (interfaceInput)
            {
                input = interfaceInput;
            }
        }

        // Add the shader input if unique.
        if (input)
        {
            int treeIndex = indexMap[input];
            if (shaderInputMap.count(treeIndex))
            {
                continue;
            }
            shaderInputMap[treeIndex] = ShaderInputPair(input, variable);
        }
    }

    // Generate UI properties for each shader input in order.
    for (const auto& it : shaderInputMap)
    {
        // Retrieve the shader input pair.
        ShaderInputPair pair = it.second;

        // Gather the UI properties for this input.
        UIPropertyItem item;
        item.variable = pair.second;
        getUIProperties(pair.first, EMPTY_STRING, item.ui);

        // Generate the item label.
        if (!item.ui.uiName.empty())
        {
            item.label = item.ui.uiName;
        }
        if (item.label.empty())
        {
            item.label = pair.first->getName();
        }

        // Prepend a parent label for unlabeled node inputs.
        ElementPtr parent = pair.first->getParent();
        if (item.ui.uiFolder.empty() && parent && parent->isA<Node>())
        {
            item.label = parent->getName() + pathSeparator + item.label;
        }

        // Add the new item.
        if (!item.ui.uiFolder.empty())
        {
            groups.emplace(item.ui.uiFolder, item);
        }
        else
        {
            unnamedGroups.emplace(EMPTY_STRING, item);
        }
    }
}

MATERIALX_NAMESPACE_END
