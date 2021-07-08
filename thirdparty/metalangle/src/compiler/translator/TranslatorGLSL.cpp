//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "compiler/translator/TranslatorGLSL.h"

#include "angle_gl.h"
#include "compiler/translator/BuiltInFunctionEmulatorGLSL.h"
#include "compiler/translator/ExtensionGLSL.h"
#include "compiler/translator/OutputGLSL.h"
#include "compiler/translator/VersionGLSL.h"
#include "compiler/translator/tree_ops/EmulatePrecision.h"
#include "compiler/translator/tree_ops/RewriteTexelFetchOffset.h"
#include "compiler/translator/tree_ops/RewriteUnaryMinusOperatorFloat.h"

namespace sh
{

TranslatorGLSL::TranslatorGLSL(sh::GLenum type, ShShaderSpec spec, ShShaderOutput output)
    : TCompiler(type, spec, output)
{}

void TranslatorGLSL::initBuiltInFunctionEmulator(BuiltInFunctionEmulator *emu,
                                                 ShCompileOptions compileOptions)
{
    if (compileOptions & SH_EMULATE_ABS_INT_FUNCTION)
    {
        InitBuiltInAbsFunctionEmulatorForGLSLWorkarounds(emu, getShaderType());
    }

    if (compileOptions & SH_EMULATE_ISNAN_FLOAT_FUNCTION)
    {
        InitBuiltInIsnanFunctionEmulatorForGLSLWorkarounds(emu, getShaderVersion());
    }

    if (compileOptions & SH_EMULATE_ATAN2_FLOAT_FUNCTION)
    {
        InitBuiltInAtanFunctionEmulatorForGLSLWorkarounds(emu);
    }

    int targetGLSLVersion = ShaderOutputTypeToGLSLVersion(getOutputType());
    InitBuiltInFunctionEmulatorForGLSLMissingFunctions(emu, getShaderType(), targetGLSLVersion);
}

bool TranslatorGLSL::translate(TIntermBlock *root,
                               ShCompileOptions compileOptions,
                               PerformanceDiagnostics * /*perfDiagnostics*/)
{
    TInfoSinkBase &sink = getInfoSink().obj;

    // Write GLSL version.
    writeVersion(root);

    // Write extension behaviour as needed
    writeExtensionBehavior(root, compileOptions);

    // Write pragmas after extensions because some drivers consider pragmas
    // like non-preprocessor tokens.
    writePragma(compileOptions);

    // If flattening the global invariant pragma, write invariant declarations for built-in
    // variables. It should be harmless to do this twice in the case that the shader also explicitly
    // did this. However, it's important to emit invariant qualifiers only for those built-in
    // variables that are actually used, to avoid affecting the behavior of the shader.
    if ((compileOptions & SH_FLATTEN_PRAGMA_STDGL_INVARIANT_ALL) != 0 &&
        getPragma().stdgl.invariantAll &&
        !sh::RemoveInvariant(getShaderType(), getShaderVersion(), getOutputType(), compileOptions))
    {
        ASSERT(wereVariablesCollected());

        switch (getShaderType())
        {
            case GL_VERTEX_SHADER:
                sink << "invariant gl_Position;\n";

                // gl_PointSize should be declared invariant in both ESSL 1.00 and 3.00 fragment
                // shaders if it's statically referenced.
                conditionallyOutputInvariantDeclaration("gl_PointSize");
                break;
            case GL_FRAGMENT_SHADER:
                // The preprocessor will reject this pragma if it's used in ESSL 3.00 fragment
                // shaders, so we can use simple logic to determine whether to declare these
                // variables invariant.
                conditionallyOutputInvariantDeclaration("gl_FragCoord");
                conditionallyOutputInvariantDeclaration("gl_PointCoord");
                break;
            default:
                // Currently not reached, but leave this in for future expansion.
                ASSERT(false);
                break;
        }
    }

    if ((compileOptions & SH_REWRITE_TEXELFETCHOFFSET_TO_TEXELFETCH) != 0)
    {
        if (!sh::RewriteTexelFetchOffset(this, root, getSymbolTable(), getShaderVersion()))
        {
            return false;
        }
    }

    if ((compileOptions & SH_REWRITE_FLOAT_UNARY_MINUS_OPERATOR) != 0)
    {
        if (!sh::RewriteUnaryMinusOperatorFloat(this, root))
        {
            return false;
        }
    }

    bool precisionEmulation =
        getResources().WEBGL_debug_shader_precision && getPragma().debugShaderPrecision;

    if (precisionEmulation)
    {
        EmulatePrecision emulatePrecision(&getSymbolTable());
        root->traverse(&emulatePrecision);
        if (!emulatePrecision.updateTree(this, root))
        {
            return false;
        }
        emulatePrecision.writeEmulationHelpers(sink, getShaderVersion(), getOutputType());
    }

    // Write emulated built-in functions if needed.
    if (!getBuiltInFunctionEmulator().isOutputEmpty())
    {
        sink << "// BEGIN: Generated code for built-in function emulation\n\n";
        sink << "#define emu_precision\n\n";
        getBuiltInFunctionEmulator().outputEmulatedFunctions(sink);
        sink << "// END: Generated code for built-in function emulation\n\n";
    }

    // Write array bounds clamping emulation if needed.
    getArrayBoundsClamper().OutputClampingFunctionDefinition(sink);

    // Declare gl_FragColor and glFragData as webgl_FragColor and webgl_FragData
    // if it's core profile shaders and they are used.
    if (getShaderType() == GL_FRAGMENT_SHADER)
    {
        const bool mayHaveESSL1SecondaryOutputs =
            IsExtensionEnabled(getExtensionBehavior(), TExtension::EXT_blend_func_extended) &&
            getShaderVersion() == 100;
        const bool declareGLFragmentOutputs = IsGLSL130OrNewer(getOutputType());

        bool hasGLFragColor          = false;
        bool hasGLFragData           = false;
        bool hasGLSecondaryFragColor = false;
        bool hasGLSecondaryFragData  = false;

        for (const auto &outputVar : mOutputVariables)
        {
            if (declareGLFragmentOutputs)
            {
                if (outputVar.name == "gl_FragColor")
                {
                    ASSERT(!hasGLFragColor);
                    hasGLFragColor = true;
                    continue;
                }
                else if (outputVar.name == "gl_FragData")
                {
                    ASSERT(!hasGLFragData);
                    hasGLFragData = true;
                    continue;
                }
            }
            if (mayHaveESSL1SecondaryOutputs)
            {
                if (outputVar.name == "gl_SecondaryFragColorEXT")
                {
                    ASSERT(!hasGLSecondaryFragColor);
                    hasGLSecondaryFragColor = true;
                    continue;
                }
                else if (outputVar.name == "gl_SecondaryFragDataEXT")
                {
                    ASSERT(!hasGLSecondaryFragData);
                    hasGLSecondaryFragData = true;
                    continue;
                }
            }
        }
        ASSERT(!((hasGLFragColor || hasGLSecondaryFragColor) &&
                 (hasGLFragData || hasGLSecondaryFragData)));
        if (hasGLFragColor)
        {
            sink << "out vec4 webgl_FragColor;\n";
        }
        if (hasGLFragData)
        {
            sink << "out vec4 webgl_FragData[gl_MaxDrawBuffers];\n";
        }
        if (hasGLSecondaryFragColor)
        {
            sink << "out vec4 angle_SecondaryFragColor;\n";
        }
        if (hasGLSecondaryFragData)
        {
            sink << "out vec4 angle_SecondaryFragData[" << getResources().MaxDualSourceDrawBuffers
                 << "];\n";
        }
    }

    if (getShaderType() == GL_COMPUTE_SHADER)
    {
        EmitWorkGroupSizeGLSL(*this, sink);
    }

    if (getShaderType() == GL_GEOMETRY_SHADER_EXT)
    {
        WriteGeometryShaderLayoutQualifiers(
            sink, getGeometryShaderInputPrimitiveType(), getGeometryShaderInvocations(),
            getGeometryShaderOutputPrimitiveType(), getGeometryShaderMaxVertices());
    }

    // Write translated shader.
    TOutputGLSL outputGLSL(sink, getArrayIndexClampingStrategy(), getHashFunction(), getNameMap(),
                           &getSymbolTable(), getShaderType(), getShaderVersion(), getOutputType(),
                           compileOptions);

    root->traverse(&outputGLSL);

    return true;
}

bool TranslatorGLSL::shouldFlattenPragmaStdglInvariantAll()
{
    // Required when outputting to any GLSL version greater than 1.20, but since ANGLE doesn't
    // translate to that version, return true for the next higher version.
    return IsGLSL130OrNewer(getOutputType());
}

bool TranslatorGLSL::shouldCollectVariables(ShCompileOptions compileOptions)
{
    return (compileOptions & SH_FLATTEN_PRAGMA_STDGL_INVARIANT_ALL) ||
           TCompiler::shouldCollectVariables(compileOptions);
}

void TranslatorGLSL::writeVersion(TIntermNode *root)
{
    TVersionGLSL versionGLSL(getShaderType(), getPragma(), getOutputType());
    root->traverse(&versionGLSL);
    int version = versionGLSL.getVersion();
    // We need to write version directive only if it is greater than 110.
    // If there is no version directive in the shader, 110 is implied.
    if (version > 110)
    {
        TInfoSinkBase &sink = getInfoSink().obj;
        sink << "#version " << version << "\n";
    }
}

void TranslatorGLSL::writeExtensionBehavior(TIntermNode *root, ShCompileOptions compileOptions)
{
    TInfoSinkBase &sink                   = getInfoSink().obj;
    const TExtensionBehavior &extBehavior = getExtensionBehavior();
    for (const auto &iter : extBehavior)
    {
        if (iter.second == EBhUndefined)
        {
            continue;
        }

        if (getOutputType() == SH_GLSL_COMPATIBILITY_OUTPUT)
        {
            // For GLSL output, we don't need to emit most extensions explicitly,
            // but some we need to translate in GL compatibility profile.
            if (iter.first == TExtension::EXT_shader_texture_lod)
            {
                sink << "#extension GL_ARB_shader_texture_lod : " << GetBehaviorString(iter.second)
                     << "\n";
            }

            if (iter.first == TExtension::EXT_draw_buffers)
            {
                sink << "#extension GL_ARB_draw_buffers : " << GetBehaviorString(iter.second)
                     << "\n";
            }

            if (iter.first == TExtension::EXT_geometry_shader)
            {
                sink << "#extension GL_ARB_geometry_shader4 : " << GetBehaviorString(iter.second)
                     << "\n";
            }
        }

        const bool isMultiview =
            (iter.first == TExtension::OVR_multiview) || (iter.first == TExtension::OVR_multiview2);
        if (isMultiview)
        {
            EmitMultiviewGLSL(*this, compileOptions, iter.second, sink);
        }

        // Support ANGLE_texture_multisample extension on GLSL300
        if (getShaderVersion() >= 300 && iter.first == TExtension::ANGLE_texture_multisample &&
            getOutputType() < SH_GLSL_330_CORE_OUTPUT)
        {
            sink << "#extension GL_ARB_texture_multisample : " << GetBehaviorString(iter.second)
                 << "\n";
        }
    }

    // GLSL ES 3 explicit location qualifiers need to use an extension before GLSL 330
    if (getShaderVersion() >= 300 && getOutputType() < SH_GLSL_330_CORE_OUTPUT &&
        getShaderType() != GL_COMPUTE_SHADER)
    {
        sink << "#extension GL_ARB_explicit_attrib_location : require\n";
    }

    // Need to enable gpu_shader5 to have index constant sampler array indexing
    if (getOutputType() != SH_ESSL_OUTPUT && getOutputType() < SH_GLSL_400_CORE_OUTPUT &&
        getShaderVersion() == 100)
    {
        // Don't use "require" on to avoid breaking WebGL 1 on drivers that silently
        // support index constant sampler array indexing, but don't have the extension or
        // on drivers that don't have the extension at all as it would break WebGL 1 for
        // some users.
        sink << "#extension GL_ARB_gpu_shader5 : enable\n";
    }

    TExtensionGLSL extensionGLSL(getOutputType());
    root->traverse(&extensionGLSL);

    for (const auto &ext : extensionGLSL.getEnabledExtensions())
    {
        sink << "#extension " << ext << " : enable\n";
    }
    for (const auto &ext : extensionGLSL.getRequiredExtensions())
    {
        sink << "#extension " << ext << " : require\n";
    }
}

void TranslatorGLSL::conditionallyOutputInvariantDeclaration(const char *builtinVaryingName)
{
    if (isVaryingDefined(builtinVaryingName))
    {
        TInfoSinkBase &sink = getInfoSink().obj;
        sink << "invariant " << builtinVaryingName << ";\n";
    }
}

}  // namespace sh
