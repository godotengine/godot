//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_GENSHADERLIBRARY_H
#define MATERIALX_GENSHADERLIBRARY_H

/// @file
/// Library-wide includes and types.  This file should be the first include for
/// any public header in the MaterialXGenShader library.

#include <MaterialXCore/Exception.h>

MATERIALX_NAMESPACE_BEGIN

class Shader;
class ShaderStage;
class ShaderGenerator;
class ShaderNode;
class ShaderGraph;
class ShaderInput;
class ShaderOutput;
class ShaderNodeImpl;
class GenOptions;
class GenContext;
class ClosureContext;
class TypeDesc;

/// A string stream
using StringStream = std::stringstream;

/// Shared pointer to a Shader
using ShaderPtr = shared_ptr<Shader>;
/// Shared pointer to a ShaderStage
using ShaderStagePtr = shared_ptr<ShaderStage>;
/// Shared pointer to a ShaderGenerator
using ShaderGeneratorPtr = shared_ptr<ShaderGenerator>;
/// Shared pointer to a ShaderNodeImpl
using ShaderNodeImplPtr = shared_ptr<ShaderNodeImpl>;
/// Shared pointer to a GenContext
using GenContextPtr = shared_ptr<GenContext>;

template <class T> using CreatorFunction = shared_ptr<T> (*)();

MATERIALX_NAMESPACE_END

#endif // MATERIALX_GENSHADERLIBRARY_H
