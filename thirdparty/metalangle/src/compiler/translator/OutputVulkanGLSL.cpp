//
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// OutputVulkanGLSL:
//   Code that outputs shaders that fit GL_KHR_vulkan_glsl.
//   The shaders are then fed into glslang to spit out SPIR-V (libANGLE-side).
//   See: https://www.khronos.org/registry/vulkan/specs/misc/GL_KHR_vulkan_glsl.txt
//

#include "compiler/translator/OutputVulkanGLSL.h"

#include "compiler/translator/BaseTypes.h"
#include "compiler/translator/Symbol.h"
#include "compiler/translator/util.h"

namespace sh
{

TOutputVulkanGLSL::TOutputVulkanGLSL(TInfoSinkBase &objSink,
                                     ShArrayIndexClampingStrategy clampingStrategy,
                                     ShHashFunction64 hashFunction,
                                     NameMap &nameMap,
                                     TSymbolTable *symbolTable,
                                     sh::GLenum shaderType,
                                     int shaderVersion,
                                     ShShaderOutput output,
                                     ShCompileOptions compileOptions)
    : TOutputGLSL(objSink,
                  clampingStrategy,
                  hashFunction,
                  nameMap,
                  symbolTable,
                  shaderType,
                  shaderVersion,
                  output,
                  compileOptions)
{}

// TODO(jmadill): This is not complete.
void TOutputVulkanGLSL::writeLayoutQualifier(TIntermTyped *variable)
{
    const TType &type = variable->getType();

    bool needsCustomLayout =
        type.getQualifier() == EvqAttribute || type.getQualifier() == EvqFragmentOut ||
        type.getQualifier() == EvqVertexIn || IsVarying(type.getQualifier()) ||
        IsSampler(type.getBasicType()) || type.isInterfaceBlock() || IsImage(type.getBasicType());

    if ((!NeedsToWriteLayoutQualifier(type) && !needsCustomLayout))
    {
        return;
    }

    TInfoSinkBase &out = objSink();

    // This isn't super clean, but it gets the job done.
    // See corresponding code in glslang_wrapper_utils.cpp.
    TIntermSymbol *symbol = variable->getAsSymbolNode();
    ASSERT(symbol);

    ImmutableString name      = symbol->getName();
    const char *blockStorage  = nullptr;
    const char *matrixPacking = nullptr;

    // For interface blocks, use the block name instead.  When the layout qualifier is being
    // replaced in the backend, that would be the name that's available.
    if (type.isInterfaceBlock())
    {
        const TInterfaceBlock *interfaceBlock = type.getInterfaceBlock();
        name                                  = interfaceBlock->name();
        TLayoutBlockStorage storage           = interfaceBlock->blockStorage();

        // Make sure block storage format is specified.
        if (storage != EbsStd430)
        {
            // Change interface block layout qualifiers to std140 for any layout that is not
            // explicitly set to std430.  This is to comply with GL_KHR_vulkan_glsl where shared and
            // packed are not allowed (and std140 could be used instead) and unspecified layouts can
            // assume either std140 or std430 (and we choose std140 as std430 is not yet universally
            // supported).
            storage = EbsStd140;
        }

        if (interfaceBlock->blockStorage() != EbsUnspecified)
        {
            blockStorage = getBlockStorageString(storage);
        }

        // We expect all interface blocks to have been transformed to column major, so we don't
        // specify the packing.  Any remaining interface block qualified with row_major shouldn't
        // have any matrices inside.
        ASSERT(type.getLayoutQualifier().matrixPacking != EmpRowMajor ||
               !interfaceBlock->containsMatrices());
    }

    if (needsCustomLayout)
    {
        out << "@@ LAYOUT-" << name << "(";
    }
    else
    {
        out << "layout(";
    }

    // Output the list of qualifiers already known at this stage, i.e. everything other than
    // `location` and `set`/`binding`.
    std::string otherQualifiers = getCommonLayoutQualifiers(variable);

    const char *separator = "";
    if (blockStorage)
    {
        out << separator << blockStorage;
        separator = ", ";
    }
    if (matrixPacking)
    {
        out << separator << matrixPacking;
        separator = ", ";
    }
    if (!otherQualifiers.empty())
    {
        out << separator << otherQualifiers;
    }

    out << ") ";
    if (needsCustomLayout)
    {
        out << "@@";
    }
}

void TOutputVulkanGLSL::writeFieldLayoutQualifier(const TField *field)
{
    // We expect all interface blocks to have been transformed to column major, as Vulkan GLSL
    // doesn't allow layout qualifiers on interface block fields.  Any remaining interface block
    // qualified with row_major shouldn't have any matrices inside, so the qualifier can be
    // dropped.
}

void TOutputVulkanGLSL::writeQualifier(TQualifier qualifier,
                                       const TType &type,
                                       const TSymbol *symbol)
{
    if (qualifier != EvqUniform && qualifier != EvqBuffer && qualifier != EvqAttribute &&
        qualifier != EvqVertexIn && !sh::IsVarying(qualifier))
    {
        TOutputGLSLBase::writeQualifier(qualifier, type, symbol);
        return;
    }

    if (symbol == nullptr)
    {
        return;
    }

    ImmutableString name = symbol->name();

    // For interface blocks, use the block name instead.  When the qualifier is being replaced in
    // the backend, that would be the name that's available.
    if (type.isInterfaceBlock())
    {
        name = type.getInterfaceBlock()->name();
    }

    TInfoSinkBase &out = objSink();
    if (name == "gl_ClipDistance")
    {
        // gl_ClipDistance can be redeclared by user. In that case, no need to write custom
        // qualifier
        out << (sh::IsVaryingIn(qualifier) ? "in" : "out ");
    }
    else
    {
        out << "@@ QUALIFIER-" << name.data() << "(" << getMemoryQualifiers(type) << ") @@ ";
    }
}

void TOutputVulkanGLSL::writeVariableType(const TType &type, const TSymbol *symbol)
{
    TType overrideType(type);

    // External textures are treated as 2D textures in the vulkan back-end
    if (type.getBasicType() == EbtSamplerExternalOES)
    {
        overrideType.setBasicType(EbtSampler2D);
    }

    TOutputGLSL::writeVariableType(overrideType, symbol);
}

void TOutputVulkanGLSL::writeStructType(const TStructure *structure)
{
    if (!structDeclared(structure))
    {
        declareStruct(structure);
        objSink() << ";\n";
    }
}

}  // namespace sh
