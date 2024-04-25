//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_RENDER_UTIL_H
#define MATERIALX_RENDER_UTIL_H

/// @file
/// Rendering utility methods

#include <MaterialXRender/Export.h>

#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/Util.h>

#include <map>

MATERIALX_NAMESPACE_BEGIN

extern MX_RENDER_API const Color3 DEFAULT_SCREEN_COLOR_SRGB;
extern MX_RENDER_API const Color3 DEFAULT_SCREEN_COLOR_LIN_REC709;

/// @name Shader Utilities
/// @{

/// Create a shader for a given element.
MX_RENDER_API ShaderPtr createShader(const string& shaderName, GenContext& context, ElementPtr elem);

/// Create a shader with a constant color output, using the given standard libraries
/// for code generation.
MX_RENDER_API ShaderPtr createConstantShader(GenContext& context,
                                             DocumentPtr stdLib,
                                             const string& shaderName,
                                             const Color3& color);

/// Create a shader with depth value output, using the given standard libraries
/// for code generation.
MX_RENDER_API ShaderPtr createDepthShader(GenContext& context,
                                          DocumentPtr stdLib,
                                          const string& shaderName);

/// Create a shader that generates a look-up table for directional albedo, using
/// the given standard libraries for code generation.
MX_RENDER_API ShaderPtr createAlbedoTableShader(GenContext& context,
                                                DocumentPtr stdLib,
                                                const string& shaderName);

/// Create a shader that generates a prefiltered environment map.
MX_RENDER_API ShaderPtr createEnvPrefilterShader(GenContext& context,
                                                 DocumentPtr stdLib,
                                                 const string& shaderName);

/// Create a blur shader, using the given standard libraries for code generation.
MX_RENDER_API ShaderPtr createBlurShader(GenContext& context,
                                         DocumentPtr stdLib,
                                         const string& shaderName,
                                         const string& filterType,
                                         float filterSize);

/// @}
/// @name User Interface Utilities
/// @{

/// Set of possible UI properties for an element
struct MX_RENDER_API UIProperties
{
    /// UI name
    string uiName;

    /// UI folder
    string uiFolder;

    /// Enumeration
    StringVec enumeration;

    /// Enumeration Values
    vector<ValuePtr> enumerationValues;

    /// UI minimum value
    ValuePtr uiMin;

    /// UI maximum value
    ValuePtr uiMax;

    /// UI soft minimum value
    ValuePtr uiSoftMin;

    /// UI soft maximum value
    ValuePtr uiSoftMax;

    /// UI step value
    ValuePtr uiStep;

    /// UI advanced element
    bool uiAdvanced = false;
};

/// Get the UI properties for a given input element and target.
/// Returns the number of properties found.
MX_RENDER_API unsigned int getUIProperties(InputPtr input, const string& target, UIProperties& uiProperties);

/// Interface for holding the UI properties associated shader port
struct MX_RENDER_API UIPropertyItem
{
    string label;
    ShaderPort* variable = nullptr;
    UIProperties ui;
};

/// A grouping of property items by name
using UIPropertyGroup = std::multimap<string, UIPropertyItem>;

/// Utility to group UI properties items based on Element group name from a VariableBlock.
/// Returns a list of named and unnamed groups.
MX_RENDER_API void createUIPropertyGroups(DocumentPtr doc, const VariableBlock& block, UIPropertyGroup& groups,
                                          UIPropertyGroup& unnamedGroups, const string& pathSeparator);

/// @}

MATERIALX_NAMESPACE_END

#endif
