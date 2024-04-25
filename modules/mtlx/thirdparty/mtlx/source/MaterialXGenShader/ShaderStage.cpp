//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenShader/ShaderStage.h>

#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/Syntax.h>
#include <MaterialXGenShader/Util.h>

#include <MaterialXCore/Value.h>

#include <MaterialXFormat/Util.h>

MATERIALX_NAMESPACE_BEGIN

namespace Stage
{

const string PIXEL = "pixel";

} // namespace Stage

//
// VariableBlock methods
//

ShaderPort* VariableBlock::operator[](const string& name)
{
    ShaderPort* v = find(name);
    if (!v)
    {
        throw ExceptionShaderGenError("No variable named '" + name + "' exists for block '" + getName() + "'");
    }
    return v;
}

const ShaderPort* VariableBlock::operator[](const string& name) const
{
    return const_cast<VariableBlock*>(this)->operator[](name);
}

ShaderPort* VariableBlock::find(const string& name)
{
    auto it = _variableMap.find(name);
    return it != _variableMap.end() ? it->second.get() : nullptr;
}

const ShaderPort* VariableBlock::find(const string& name) const
{
    return const_cast<VariableBlock*>(this)->find(name);
}

ShaderPort* VariableBlock::find(const ShaderPortPredicate& predicate)
{
    for (ShaderPort* port : getVariableOrder())
    {
        if (predicate(port))
        {
            return port;
        }
    }
    return nullptr;
}

ShaderPort* VariableBlock::add(const TypeDesc* type, const string& name, ValuePtr value, bool shouldWiden)
{
    auto it = _variableMap.find(name);
    if (it != _variableMap.end())
    {
        if (shouldWiden)
        {
            // Automatically try to widen the type of the shader port if the requested type differs from
            // the existing port's type.
            if (it->second->getType()->getSize() < type->getSize())
            {
                it->second->setType(type);
            }
        }
        else if (type != it->second->getType())
        {
            throw ExceptionShaderGenError("Trying to add shader port '" + name + "' with type '" +
                                          type->getName() + "', but existing shader port with type '" +
                                          it->second->getType()->getName() + "' was found");
        }
        return it->second.get();
    }

    ShaderPortPtr port = std::make_shared<ShaderPort>(nullptr, type, name, value);
    _variableMap[name] = port;
    _variableOrder.push_back(port.get());

    return port.get();
}

void VariableBlock::add(ShaderPortPtr port)
{
    if (!_variableMap.count(port->getName()))
    {
        _variableMap[port->getName()] = port;
        _variableOrder.push_back(port.get());
    }
}

//
// ShaderStage methods
//

ShaderStage::ShaderStage(const string& name, ConstSyntaxPtr syntax) :
    _name(name),
    _syntax(syntax),
    _indentations(0),
    _constants("Constants", "cn")
{
}

VariableBlockPtr ShaderStage::createUniformBlock(const string& name, const string& instance)
{
    auto it = _uniforms.find(name);
    if (it == _uniforms.end())
    {
        VariableBlockPtr b = std::make_shared<VariableBlock>(name, instance);
        _uniforms[name] = b;
        return b;
    }
    return it->second;
}

VariableBlockPtr ShaderStage::createInputBlock(const string& name, const string& instance)
{
    auto it = _inputs.find(name);
    if (it == _inputs.end())
    {
        VariableBlockPtr b = std::make_shared<VariableBlock>(name, instance);
        _inputs[name] = b;
        return b;
    }
    return it->second;
}

VariableBlockPtr ShaderStage::createOutputBlock(const string& name, const string& instance)
{
    auto it = _outputs.find(name);
    if (it == _outputs.end())
    {
        VariableBlockPtr b = std::make_shared<VariableBlock>(name, instance);
        _outputs[name] = b;
        return b;
    }
    return it->second;
}

VariableBlock& ShaderStage::getUniformBlock(const string& name)
{
    auto it = _uniforms.find(name);
    if (it == _uniforms.end())
    {
        throw ExceptionShaderGenError("No uniform block named '" + name + "' exists for shader stage '" + getName() + "'");
    }
    return *it->second;
}

const VariableBlock& ShaderStage::getUniformBlock(const string& name) const
{
    return const_cast<ShaderStage*>(this)->getUniformBlock(name);
}

VariableBlock& ShaderStage::getInputBlock(const string& name)
{
    auto it = _inputs.find(name);
    if (it == _inputs.end())
    {
        throw ExceptionShaderGenError("No input block named '" + name + "' exists for shader stage '" + getName() + "'");
    }
    return *it->second;
}

const VariableBlock& ShaderStage::getInputBlock(const string& name) const
{
    return const_cast<ShaderStage*>(this)->getInputBlock(name);
}

VariableBlock& ShaderStage::getOutputBlock(const string& name)
{
    auto it = _outputs.find(name);
    if (it == _outputs.end())
    {
        throw ExceptionShaderGenError("No output block named '" + name + "' exists for shader stage '" + getName() + "'");
    }
    return *it->second;
}

const VariableBlock& ShaderStage::getOutputBlock(const string& name) const
{
    return const_cast<ShaderStage*>(this)->getOutputBlock(name);
}

VariableBlock& ShaderStage::getConstantBlock()
{
    return _constants;
}

const VariableBlock& ShaderStage::getConstantBlock() const
{
    return _constants;
}

void ShaderStage::beginScope(Syntax::Punctuation punc)
{
    switch (punc)
    {
        case Syntax::CURLY_BRACKETS:
            beginLine();
            _code += "{" + _syntax->getNewline();
            break;
        case Syntax::PARENTHESES:
            beginLine();
            _code += "(" + _syntax->getNewline();
            break;
        case Syntax::SQUARE_BRACKETS:
            beginLine();
            _code += "[" + _syntax->getNewline();
            break;
        case Syntax::DOUBLE_SQUARE_BRACKETS:
            beginLine();
            _code += "[[" + _syntax->getNewline();
            break;
    }

    ++_indentations;
    _scopes.push_back(Scope(punc));
}

void ShaderStage::endScope(bool semicolon, bool newline)
{
    if (_scopes.empty())
    {
        throw ExceptionShaderGenError("End scope called with no scope active, please check your beginScope/endScope calls");
    }

    Syntax::Punctuation punc = _scopes.back().punctuation;
    _scopes.pop_back();
    --_indentations;

    switch (punc)
    {
        case Syntax::CURLY_BRACKETS:
            beginLine();
            _code += "}";
            break;
        case Syntax::PARENTHESES:
            beginLine();
            _code += ")";
            break;
        case Syntax::SQUARE_BRACKETS:
            beginLine();
            _code += "]";
            break;
        case Syntax::DOUBLE_SQUARE_BRACKETS:
            beginLine();
            _code += "]]";
            break;
    }
    if (semicolon)
        _code += ";";
    if (newline)
        _code += _syntax->getNewline();
}

void ShaderStage::beginLine()
{
    for (int i = 0; i < _indentations; ++i)
    {
        _code += _syntax->getIndentation();
    }
}

void ShaderStage::endLine(bool semicolon)
{
    if (semicolon)
    {
        _code += ";";
    }
    newLine();
}

void ShaderStage::newLine()
{
    _code += _syntax->getNewline();
}

void ShaderStage::addString(const string& str)
{
    _code += str;
}

void ShaderStage::addLine(const string& str, bool semicolon)
{
    beginLine();
    addString(str);
    endLine(semicolon);
}

void ShaderStage::addComment(const string& str)
{
    beginLine();
    _code += _syntax->getSingleLineComment() + str;
    endLine(false);
}

void ShaderStage::addBlock(const string& str, const FilePath& sourceFilename, GenContext& context)
{
    const string& INCLUDE = _syntax->getIncludeStatement();
    const string& QUOTE   = _syntax->getStringQuote();

    // Add each line in the block seperately to get correct indentation.
    StringStream stream(str);
    for (string line; std::getline(stream, line);)
    {
        size_t pos = line.find(INCLUDE);
        if (pos != string::npos)
        {
            size_t startQuote = line.find_first_of(QUOTE);
            size_t endQuote = line.find_last_of(QUOTE);
            if (startQuote != string::npos && endQuote != string::npos && endQuote > startQuote)
            {
                size_t length = (endQuote - startQuote) - 1;
                if (length)
                {
                    const string filename = line.substr(startQuote + 1, length);
                    addInclude(filename, sourceFilename, context);
                }
            }
        }
        else
        {
            addLine(line, false);
        }
    }
}

void ShaderStage::addInclude(const FilePath& includeFilename, const FilePath& sourceFilename, GenContext& context)
{
    string modifiedFile = includeFilename;
    tokenSubstitution(context.getShaderGenerator().getTokenSubstitutions(), modifiedFile);
    FilePath resolvedFile = context.resolveSourceFile(modifiedFile, sourceFilename.getParentPath());

    if (!_includes.count(resolvedFile))
    {
        string content = readFile(resolvedFile);
        if (content.empty())
        {
            throw ExceptionShaderGenError("Could not find include file: '" + includeFilename.asString() + "'");
        }
        _includes.insert(resolvedFile);
        addBlock(content, resolvedFile, context);
    }
}

void ShaderStage::addSourceDependency(const FilePath& file)
{
    if (!_sourceDependencies.count(file))
    {
        _sourceDependencies.insert(file);
    }
}

void ShaderStage::addFunctionDefinition(const ShaderNode& node, GenContext& context)
{
    const ShaderNodeImpl& impl = node.getImplementation();
    const size_t id = impl.getHash();

    // Make sure it's not already defined.
    if (!_definedFunctions.count(id))
    {
        _definedFunctions.insert(id);
        impl.emitFunctionDefinition(node, context, *this);
    }
}

void ShaderStage::addFunctionCall(const ShaderNode& node, GenContext& context, bool emitCode)
{
    // Register this function as being called in the current scope.
    const ClosureContext* cct = context.getClosureContext();
    const FunctionCallId id(&node, cct ? cct->getType() : 0);
    _scopes.back().functions.insert(id);

    // Emit code for the function call if not omitted.
    if (emitCode)
    {
        node.getImplementation().emitFunctionCall(node, context, *this);
    }
}

bool ShaderStage::isEmitted(const ShaderNode& node, GenContext& context) const
{
    const ClosureContext* cct = context.getClosureContext();
    const FunctionCallId id(&node, cct ? cct->getType() : 0);

    for (const Scope& s : _scopes)
    {
        if (s.functions.count(id))
        {
            return true;
        }
    }
    return false;
}

MATERIALX_NAMESPACE_END
