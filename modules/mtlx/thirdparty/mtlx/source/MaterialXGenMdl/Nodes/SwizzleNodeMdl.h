//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_SWIZZLENODEMDL_H
#define MATERIALX_SWIZZLENODEMDL_H

#include <MaterialXGenMdl/Export.h>

#include <MaterialXGenShader/Nodes/SwizzleNode.h>

MATERIALX_NAMESPACE_BEGIN

/// Compound node implementation
class MX_GENMDL_API SwizzleNodeMdl : public SwizzleNode
{
    using Base = SwizzleNode;

  public:
    static ShaderNodeImplPtr create();

  protected:
    string getVariableName(const ShaderInput* input) const final;
};

MATERIALX_NAMESPACE_END

#endif
