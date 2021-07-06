//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// The ValidateVaryingLocations function checks if there exists location conflicts on shader
// varyings.
//

#include "ValidateVaryingLocations.h"

#include "compiler/translator/Diagnostics.h"
#include "compiler/translator/SymbolTable.h"
#include "compiler/translator/tree_util/IntermTraverse.h"
#include "compiler/translator/util.h"

namespace sh
{

namespace
{

void error(const TIntermSymbol &symbol, const char *reason, TDiagnostics *diagnostics)
{
    diagnostics->error(symbol.getLine(), reason, symbol.getName().data());
}

int GetLocationCount(const TIntermSymbol *varying, bool ignoreVaryingArraySize)
{
    const auto &varyingType = varying->getType();
    if (varyingType.getStruct() != nullptr)
    {
        ASSERT(!varyingType.isArray());
        int totalLocation = 0;
        for (const auto *field : varyingType.getStruct()->fields())
        {
            const auto *fieldType = field->type();
            ASSERT(fieldType->getStruct() == nullptr && !fieldType->isArray());

            totalLocation += fieldType->getSecondarySize();
        }
        return totalLocation;
    }
    // [GL_EXT_shader_io_blocks SPEC Chapter 4.4.1]
    // Geometry shader inputs, tessellation control shader inputs and outputs, and tessellation
    // evaluation inputs all have an additional level of arrayness relative to other shader inputs
    // and outputs. This outer array level is removed from the type before considering how many
    // locations the type consumes.
    else if (ignoreVaryingArraySize)
    {
        // Array-of-arrays cannot be inputs or outputs of a geometry shader.
        // (GL_EXT_geometry_shader SPEC issues(5))
        ASSERT(!varyingType.isArrayOfArrays());
        return varyingType.getSecondarySize();
    }
    else
    {
        return varyingType.getSecondarySize() * static_cast<int>(varyingType.getArraySizeProduct());
    }
}

using VaryingVector = std::vector<const TIntermSymbol *>;

void ValidateShaderInterface(TDiagnostics *diagnostics,
                             VaryingVector &varyingVector,
                             bool ignoreVaryingArraySize)
{
    // Location conflicts can only happen when there are two or more varyings in varyingVector.
    if (varyingVector.size() <= 1)
    {
        return;
    }

    std::map<int, const TIntermSymbol *> locationMap;
    for (const TIntermSymbol *varying : varyingVector)
    {
        const int location = varying->getType().getLayoutQualifier().location;
        ASSERT(location >= 0);

        const int elementCount = GetLocationCount(varying, ignoreVaryingArraySize);
        for (int elementIndex = 0; elementIndex < elementCount; ++elementIndex)
        {
            const int offsetLocation = location + elementIndex;
            if (locationMap.find(offsetLocation) != locationMap.end())
            {
                std::stringstream strstr = sh::InitializeStream<std::stringstream>();
                strstr << "'" << varying->getName()
                       << "' conflicting location with previously defined '"
                       << locationMap[offsetLocation]->getName() << "'";
                error(*varying, strstr.str().c_str(), diagnostics);
            }
            else
            {
                locationMap[offsetLocation] = varying;
            }
        }
    }
}

class ValidateVaryingLocationsTraverser : public TIntermTraverser
{
  public:
    ValidateVaryingLocationsTraverser(GLenum shaderType);
    void validate(TDiagnostics *diagnostics);

  private:
    bool visitDeclaration(Visit visit, TIntermDeclaration *node) override;
    bool visitFunctionDefinition(Visit visit, TIntermFunctionDefinition *node) override;

    VaryingVector mInputVaryingsWithLocation;
    VaryingVector mOutputVaryingsWithLocation;
    GLenum mShaderType;
};

ValidateVaryingLocationsTraverser::ValidateVaryingLocationsTraverser(GLenum shaderType)
    : TIntermTraverser(true, false, false), mShaderType(shaderType)
{}

bool ValidateVaryingLocationsTraverser::visitDeclaration(Visit visit, TIntermDeclaration *node)
{
    const TIntermSequence &sequence = *(node->getSequence());
    ASSERT(!sequence.empty());

    const TIntermSymbol *symbol = sequence.front()->getAsSymbolNode();
    if (symbol == nullptr)
    {
        return false;
    }

    if (symbol->variable().symbolType() == SymbolType::Empty)
    {
        return false;
    }

    // Collect varyings that have explicit 'location' qualifiers.
    const TQualifier qualifier = symbol->getQualifier();
    if (symbol->getType().getLayoutQualifier().location != -1)
    {
        if (IsVaryingIn(qualifier))
        {
            mInputVaryingsWithLocation.push_back(symbol);
        }
        else if (IsVaryingOut(qualifier))
        {
            mOutputVaryingsWithLocation.push_back(symbol);
        }
    }

    return false;
}

bool ValidateVaryingLocationsTraverser::visitFunctionDefinition(Visit visit,
                                                                TIntermFunctionDefinition *node)
{
    // We stop traversing function definitions because varyings cannot be defined in a function.
    return false;
}

void ValidateVaryingLocationsTraverser::validate(TDiagnostics *diagnostics)
{
    ASSERT(diagnostics);

    ValidateShaderInterface(diagnostics, mInputVaryingsWithLocation,
                            mShaderType == GL_GEOMETRY_SHADER_EXT);
    ValidateShaderInterface(diagnostics, mOutputVaryingsWithLocation, false);
}

}  // anonymous namespace

bool ValidateVaryingLocations(TIntermBlock *root, TDiagnostics *diagnostics, GLenum shaderType)
{
    ValidateVaryingLocationsTraverser varyingValidator(shaderType);
    root->traverse(&varyingValidator);
    int numErrorsBefore = diagnostics->numErrors();
    varyingValidator.validate(diagnostics);
    return (diagnostics->numErrors() == numErrorsBefore);
}

}  // namespace sh
