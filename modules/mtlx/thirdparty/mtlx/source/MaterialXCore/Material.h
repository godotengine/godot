//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_MATERIAL_H
#define MATERIALX_MATERIAL_H

/// @file
/// Material node helper functions

#include <MaterialXCore/Export.h>

#include <MaterialXCore/Node.h>

MATERIALX_NAMESPACE_BEGIN

/// Return a vector of all shader nodes connected to the given material node's inputs,
/// filtered by the given shader type and target.  By default, all surface shader nodes
/// are returned.
/// @param materialNode The node to examine.
/// @param nodeType THe shader node type to return.  Defaults to the surface shader type.
/// @param target An optional target name, which will be used to filter the returned nodes.
MX_CORE_API vector<NodePtr> getShaderNodes(NodePtr materialNode,
                                           const string& nodeType = SURFACE_SHADER_TYPE_STRING,
                                           const string& target = EMPTY_STRING);

/// Return a vector of all outputs connected to the given node's inputs.
MX_CORE_API vector<OutputPtr> getConnectedOutputs(NodePtr node);

MATERIALX_NAMESPACE_END

#endif
