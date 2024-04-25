//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_IMAGENODEMDL_H
#define MATERIALX_IMAGENODEMDL_H

#include <MaterialXGenMdl/Export.h>

#include "SourceCodeNodeMdl.h"

MATERIALX_NAMESPACE_BEGIN

/// Image node implementation for MDL
class MX_GENMDL_API ImageNodeMdl : public SourceCodeNodeMdl
{
    using BASE = SourceCodeNodeMdl;

  public:
    static const string FLIP_V; ///< the empty string ""

    static ShaderNodeImplPtr create();

    void addInputs(ShaderNode& node, GenContext& context) const override;

    bool isEditable(const ShaderInput& input) const override;

    void emitFunctionCall(const ShaderNode& node, GenContext& context, ShaderStage& stage) const override;
};

MATERIALX_NAMESPACE_END

#endif
