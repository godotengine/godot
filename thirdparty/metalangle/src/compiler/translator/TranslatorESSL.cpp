//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "compiler/translator/TranslatorESSL.h"

#include "angle_gl.h"
#include "compiler/translator/BuiltInFunctionEmulatorGLSL.h"
#include "compiler/translator/OutputESSL.h"
#include "compiler/translator/tree_ops/EmulatePrecision.h"
#include "compiler/translator/tree_ops/RecordConstantPrecision.h"

namespace sh
{

TranslatorESSL::TranslatorESSL(sh::GLenum type, ShShaderSpec spec)
    : TCompiler(type, spec, SH_ESSL_OUTPUT)
{}

void TranslatorESSL::initBuiltInFunctionEmulator(BuiltInFunctionEmulator *emu,
                                                 ShCompileOptions compileOptions)
{
    if (compileOptions & SH_EMULATE_ATAN2_FLOAT_FUNCTION)
    {
        InitBuiltInAtanFunctionEmulatorForGLSLWorkarounds(emu);
    }
}

bool TranslatorESSL::translate(TIntermBlock *root,
                               ShCompileOptions compileOptions,
                               PerformanceDiagnostics * /*perfDiagnostics*/)
{
    TInfoSinkBase &sink = getInfoSink().obj;

    int shaderVer = getShaderVersion();
    if (shaderVer > 100)
    {
        sink << "#version " << shaderVer << " es\n";
    }

    // Write built-in extension behaviors.
    writeExtensionBehavior(compileOptions);

    // Write pragmas after extensions because some drivers consider pragmas
    // like non-preprocessor tokens.
    writePragma(compileOptions);

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
        emulatePrecision.writeEmulationHelpers(sink, shaderVer, SH_ESSL_OUTPUT);
    }

    if (!RecordConstantPrecision(this, root, &getSymbolTable()))
    {
        return false;
    }

    // Write emulated built-in functions if needed.
    if (!getBuiltInFunctionEmulator().isOutputEmpty())
    {
        sink << "// BEGIN: Generated code for built-in function emulation\n\n";
        if (getShaderType() == GL_FRAGMENT_SHADER)
        {
            sink << "#if defined(GL_FRAGMENT_PRECISION_HIGH)\n"
                 << "#define emu_precision highp\n"
                 << "#else\n"
                 << "#define emu_precision mediump\n"
                 << "#endif\n\n";
        }
        else
        {
            sink << "#define emu_precision highp\n";
        }

        getBuiltInFunctionEmulator().outputEmulatedFunctions(sink);
        sink << "// END: Generated code for built-in function emulation\n\n";
    }

    // Write array bounds clamping emulation if needed.
    getArrayBoundsClamper().OutputClampingFunctionDefinition(sink);

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
    TOutputESSL outputESSL(sink, getArrayIndexClampingStrategy(), getHashFunction(), getNameMap(),
                           &getSymbolTable(), getShaderType(), shaderVer, precisionEmulation,
                           compileOptions);

    root->traverse(&outputESSL);

    return true;
}

bool TranslatorESSL::shouldFlattenPragmaStdglInvariantAll()
{
    // If following the spec to the letter, we should not flatten this pragma.
    // However, the spec's wording means that the pragma applies only to outputs.
    // This contradicts the spirit of using the pragma,
    // because if the pragma is used in a vertex shader,
    // the only way to be able to link it to a fragment shader
    // is to manually qualify each of fragment shader's inputs as invariant.
    // Which defeats the purpose of this pragma - temporarily make all varyings
    // invariant for debugging.
    // Thus, we should be non-conformant to spec's letter here and flatten.
    return true;
}

void TranslatorESSL::writeExtensionBehavior(ShCompileOptions compileOptions)
{
    TInfoSinkBase &sink                   = getInfoSink().obj;
    const TExtensionBehavior &extBehavior = getExtensionBehavior();
    for (TExtensionBehavior::const_iterator iter = extBehavior.begin(); iter != extBehavior.end();
         ++iter)
    {
        if (iter->second != EBhUndefined)
        {
            const bool isMultiview = (iter->first == TExtension::OVR_multiview) ||
                                     (iter->first == TExtension::OVR_multiview2);
            if (getResources().NV_shader_framebuffer_fetch &&
                iter->first == TExtension::EXT_shader_framebuffer_fetch)
            {
                sink << "#extension GL_NV_shader_framebuffer_fetch : "
                     << GetBehaviorString(iter->second) << "\n";
            }
            else if (getResources().NV_draw_buffers && iter->first == TExtension::EXT_draw_buffers)
            {
                sink << "#extension GL_NV_draw_buffers : " << GetBehaviorString(iter->second)
                     << "\n";
            }
            else if (isMultiview)
            {
                EmitMultiviewGLSL(*this, compileOptions, iter->second, sink);
            }
            else if (iter->first == TExtension::EXT_geometry_shader)
            {
                sink << "#ifdef GL_EXT_geometry_shader\n"
                     << "#extension GL_EXT_geometry_shader : " << GetBehaviorString(iter->second)
                     << "\n"
                     << "#elif defined GL_OES_geometry_shader\n"
                     << "#extension GL_OES_geometry_shader : " << GetBehaviorString(iter->second)
                     << "\n";
                if (iter->second == EBhRequire)
                {
                    sink << "#else\n"
                         << "#error \"No geometry shader extensions available.\" // Only generate "
                            "this if the extension is \"required\"\n";
                }
                sink << "#endif\n";
            }
            else if (iter->first == TExtension::ANGLE_multi_draw)
            {
                // Don't emit anything. This extension is emulated
                ASSERT((compileOptions & SH_EMULATE_GL_DRAW_ID) != 0);
                continue;
            }
            else if (iter->first == TExtension::ANGLE_base_vertex_base_instance)
            {
                // Don't emit anything. This extension is emulated
                ASSERT((compileOptions & SH_EMULATE_GL_BASE_VERTEX_BASE_INSTANCE) != 0);
                continue;
            }
            else
            {
                sink << "#extension " << GetExtensionNameString(iter->first) << " : "
                     << GetBehaviorString(iter->second) << "\n";
            }
        }
    }
}

}  // namespace sh
