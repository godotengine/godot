//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGenShader/GenOptions.h>

#include <emscripten/bind.h>

namespace ems = emscripten;
namespace mx = MaterialX;

EMSCRIPTEN_BINDINGS(GenOptions)
{
    ems::enum_<mx::ShaderInterfaceType>("ShaderInterfaceType")
        .value("SHADER_INTERFACE_COMPLETE", mx::ShaderInterfaceType::SHADER_INTERFACE_COMPLETE)
        .value("SHADER_INTERFACE_REDUCED", mx::ShaderInterfaceType::SHADER_INTERFACE_REDUCED)
        ;

    ems::enum_<mx::HwSpecularEnvironmentMethod>("HwSpecularEnvironmentMethod")
        .value("SPECULAR_ENVIRONMENT_NONE",mx::HwSpecularEnvironmentMethod::SPECULAR_ENVIRONMENT_NONE)
        .value("SPECULAR_ENVIRONMENT_FIS", mx::HwSpecularEnvironmentMethod::SPECULAR_ENVIRONMENT_FIS)
        .value("SPECULAR_ENVIRONMENT_PREFILTER", mx::HwSpecularEnvironmentMethod::SPECULAR_ENVIRONMENT_PREFILTER)
        ;

    ems::enum_<mx::HwDirectionalAlbedoMethod>("HwDirectionalAlbedoMethod")
        .value("DIRECTIONAL_ALBEDO_ANALYTIC",mx::HwDirectionalAlbedoMethod::DIRECTIONAL_ALBEDO_ANALYTIC)
        .value("DIRECTIONAL_ALBEDO_TABLE", mx::HwDirectionalAlbedoMethod::DIRECTIONAL_ALBEDO_TABLE)
        .value("DIRECTIONAL_ALBEDO_MONTE_CARLO", mx::HwDirectionalAlbedoMethod::DIRECTIONAL_ALBEDO_MONTE_CARLO)
        ;
        
    ems::class_<mx::GenOptions>("GenOptions")
        .property("shaderInterfaceType", &mx::GenOptions::shaderInterfaceType)
        .property("fileTextureVerticalFlip", &mx::GenOptions::fileTextureVerticalFlip)
        .property("addUpstreamDependencies", &mx::GenOptions::addUpstreamDependencies)
        .property("hwTransparency", &mx::GenOptions::hwTransparency)
        .property("hwSpecularEnvironmentMethod", &mx::GenOptions::hwSpecularEnvironmentMethod)
        .property("hwDirectionalAlbedoMethod", &mx::GenOptions::hwDirectionalAlbedoMethod)
        .property("hwWriteDepthMoments", &mx::GenOptions::hwWriteDepthMoments)
        .property("hwShadowMap", &mx::GenOptions::hwShadowMap)
        .property("hwAmbientOcclusion", &mx::GenOptions::hwAmbientOcclusion)
        .property("hwMaxActiveLightSources", &mx::GenOptions::hwMaxActiveLightSources)
        .property("hwNormalizeUdimTexCoords", &mx::GenOptions::hwNormalizeUdimTexCoords)
        .property("hwWriteAlbedoTable", &mx::GenOptions::hwWriteAlbedoTable)
        .property("hwWriteEnvPrefilter", &mx::GenOptions::hwWriteEnvPrefilter)
        ;
}
