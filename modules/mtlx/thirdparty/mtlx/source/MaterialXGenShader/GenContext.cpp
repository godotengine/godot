//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/ShaderGenerator.h>

MATERIALX_NAMESPACE_BEGIN

//
// GenContext methods
//

GenContext::GenContext(ShaderGeneratorPtr sg) :
    _sg(sg)
{
    if (!_sg)
    {
        throw ExceptionShaderGenError("GenContext must have a valid shader generator");
    }

    // Collect and cache reserved words from the shader generator
    StringSet reservedWords;

    // Add reserved words from the syntax
    reservedWords = _sg->getSyntax().getReservedWords();

    // Add token substitution identifiers
    for (const auto& it : _sg->getTokenSubstitutions())
    {
        if (!it.second.empty())
        {
            reservedWords.insert(it.second);
        }
    }

    addReservedWords(reservedWords);

    _applicationVariableHandler = nullptr;
}

void GenContext::addNodeImplementation(const string& name, ShaderNodeImplPtr impl)
{
    _nodeImpls[name] = impl;
}

ShaderNodeImplPtr GenContext::findNodeImplementation(const string& name) const
{
    auto it = _nodeImpls.find(name);
    return it != _nodeImpls.end() ? it->second : nullptr;
}

void GenContext::getNodeImplementationNames(StringSet& names)
{
    for (auto it : _nodeImpls)
    {
        names.insert(it.first);
    }
}

void GenContext::clearNodeImplementations()
{
    _nodeImpls.clear();
}

void GenContext::clearUserData()
{
    _userData.clear();
}

void GenContext::addInputSuffix(const ShaderInput* input, const string& suffix)
{
    _inputSuffix[input] = suffix;
}

void GenContext::removeInputSuffix(const ShaderInput* input)
{
    _inputSuffix.erase(input);
}

void GenContext::getInputSuffix(const ShaderInput* input, string& suffix) const
{
    suffix.clear();
    std::unordered_map<const ShaderInput*, string>::const_iterator iter = _inputSuffix.find(input);
    if (iter != _inputSuffix.end())
    {
        suffix = iter->second;
    }
}

void GenContext::addOutputSuffix(const ShaderOutput* output, const string& suffix)
{
    _outputSuffix[output] = suffix;
}

void GenContext::removeOutputSuffix(const ShaderOutput* output)
{
    _outputSuffix.erase(output);
}

void GenContext::getOutputSuffix(const ShaderOutput* output, string& suffix) const
{
    suffix.clear();
    std::unordered_map<const ShaderOutput*, string>::const_iterator iter = _outputSuffix.find(output);
    if (iter != _outputSuffix.end())
    {
        suffix = iter->second;
    }
}

ScopedSetClosureParams::ScopedSetClosureParams(const ClosureContext::ClosureParams* params, const ShaderNode* node, ClosureContext* cct) :
    _cct(cct),
    _node(node),
    _oldParams(nullptr)
{
    if (_cct)
    {
        _oldParams = _cct->getClosureParams(_node);
        _cct->setClosureParams(_node, params);
    }
}

ScopedSetClosureParams::ScopedSetClosureParams(const ShaderNode* fromNode, const ShaderNode* toNode, ClosureContext* cct) :
    _cct(cct),
    _node(toNode),
    _oldParams(nullptr)
{
    // The class must be safe for the cases where a context is not set
    // so make sure to check for nullptr here.
    if (_cct)
    {
        const ClosureContext::ClosureParams* newParams = _cct->getClosureParams(fromNode);
        if (newParams)
        {
            _oldParams = _cct->getClosureParams(_node);
            _cct->setClosureParams(_node, newParams);
        }
    }
}

ScopedSetClosureParams::~ScopedSetClosureParams()
{
    if (_cct)
    {
        _cct->setClosureParams(_node, _oldParams);
    }
}

ScopedSetVariableName::ScopedSetVariableName(const string& name, ShaderPort* port) :
    _port(port),
    _oldName(port->getVariable())
{
    _port->setVariable(name);
}

ScopedSetVariableName::~ScopedSetVariableName()
{
    _port->setVariable(_oldName);
}

MATERIALX_NAMESPACE_END
