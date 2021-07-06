//
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// EmulateGLDrawID is an AST traverser to convert the gl_DrawID builtin
// to a uniform int
//
// EmulateGLBaseVertex is an AST traverser to convert the gl_BaseVertex builtin
// to a uniform int
//
// EmulateGLBaseInstance is an AST traverser to convert the gl_BaseInstance builtin
// to a uniform int
//

#include "compiler/translator/tree_ops/EmulateMultiDrawShaderBuiltins.h"

#include "angle_gl.h"
#include "compiler/translator/StaticType.h"
#include "compiler/translator/Symbol.h"
#include "compiler/translator/SymbolTable.h"
#include "compiler/translator/tree_util/BuiltIn.h"
#include "compiler/translator/tree_util/IntermTraverse.h"
#include "compiler/translator/tree_util/ReplaceVariable.h"
#include "compiler/translator/util.h"

namespace sh
{

namespace
{

constexpr const ImmutableString kEmulatedGLDrawIDName("angle_DrawID");

class FindGLDrawIDTraverser : public TIntermTraverser
{
  public:
    FindGLDrawIDTraverser() : TIntermTraverser(true, false, false), mVariable(nullptr) {}

    const TVariable *getGLDrawIDBuiltinVariable() { return mVariable; }

  protected:
    void visitSymbol(TIntermSymbol *node) override
    {
        if (&node->variable() == BuiltInVariable::gl_DrawID())
        {
            mVariable = &node->variable();
        }
    }

  private:
    const TVariable *mVariable;
};

class AddBaseVertexToGLVertexIDTraverser : public TIntermTraverser
{
  public:
    AddBaseVertexToGLVertexIDTraverser() : TIntermTraverser(true, false, false) {}

  protected:
    void visitSymbol(TIntermSymbol *node) override
    {
        if (&node->variable() == BuiltInVariable::gl_VertexID())
        {

            TIntermSymbol *baseVertexRef = new TIntermSymbol(BuiltInVariable::gl_BaseVertex());

            TIntermBinary *addBaseVertex = new TIntermBinary(EOpAdd, node, baseVertexRef);
            queueReplacement(addBaseVertex, OriginalNode::BECOMES_CHILD);
        }
    }
};

constexpr const ImmutableString kEmulatedGLBaseVertexName("angle_BaseVertex");

class FindGLBaseVertexTraverser : public TIntermTraverser
{
  public:
    FindGLBaseVertexTraverser() : TIntermTraverser(true, false, false), mVariable(nullptr) {}

    const TVariable *getGLBaseVertexBuiltinVariable() { return mVariable; }

  protected:
    void visitSymbol(TIntermSymbol *node) override
    {
        if (&node->variable() == BuiltInVariable::gl_BaseVertex())
        {
            mVariable = &node->variable();
        }
    }

  private:
    const TVariable *mVariable;
};

constexpr const ImmutableString kEmulatedGLBaseInstanceName("angle_BaseInstance");

class FindGLBaseInstanceTraverser : public TIntermTraverser
{
  public:
    FindGLBaseInstanceTraverser() : TIntermTraverser(true, false, false), mVariable(nullptr) {}

    const TVariable *getGLBaseInstanceBuiltinVariable() { return mVariable; }

  protected:
    void visitSymbol(TIntermSymbol *node) override
    {
        if (&node->variable() == BuiltInVariable::gl_BaseInstance())
        {
            mVariable = &node->variable();
        }
    }

  private:
    const TVariable *mVariable;
};

}  // namespace

bool EmulateGLDrawID(TCompiler *compiler,
                     TIntermBlock *root,
                     TSymbolTable *symbolTable,
                     std::vector<sh::ShaderVariable> *uniforms,
                     bool shouldCollect)
{
    FindGLDrawIDTraverser traverser;
    root->traverse(&traverser);
    const TVariable *builtInVariable = traverser.getGLDrawIDBuiltinVariable();
    if (builtInVariable)
    {
        const TType *type = StaticType::Get<EbtInt, EbpHigh, EvqUniform, 1, 1>();
        const TVariable *drawID =
            new TVariable(symbolTable, kEmulatedGLDrawIDName, type, SymbolType::AngleInternal);
        const TIntermSymbol *drawIDSymbol = new TIntermSymbol(drawID);

        // AngleInternal variables don't get collected
        if (shouldCollect)
        {
            ShaderVariable uniform;
            uniform.name       = kEmulatedGLDrawIDName.data();
            uniform.mappedName = kEmulatedGLDrawIDName.data();
            uniform.type       = GLVariableType(*type);
            uniform.precision  = GLVariablePrecision(*type);
            uniform.staticUse  = symbolTable->isStaticallyUsed(*builtInVariable);
            uniform.active     = true;
            uniform.binding    = type->getLayoutQualifier().binding;
            uniform.location   = type->getLayoutQualifier().location;
            uniform.offset     = type->getLayoutQualifier().offset;
            uniform.readonly   = type->getMemoryQualifier().readonly;
            uniform.writeonly  = type->getMemoryQualifier().writeonly;
            uniforms->push_back(uniform);
        }

        DeclareGlobalVariable(root, drawID);
        if (!ReplaceVariableWithTyped(compiler, root, builtInVariable, drawIDSymbol))
        {
            return false;
        }
    }

    return true;
}

bool EmulateGLBaseVertexBaseInstance(TCompiler *compiler,
                                     TIntermBlock *root,
                                     TSymbolTable *symbolTable,
                                     std::vector<sh::ShaderVariable> *uniforms,
                                     bool shouldCollect,
                                     bool addBaseVertexToVertexID)
{
    bool addBaseVertex = false, addBaseInstance = false;
    ShaderVariable uniformBaseVertex, uniformBaseInstance;

    if (addBaseVertexToVertexID)
    {
        // This is a workaround for Mac AMD GPU
        // Replace gl_VertexID with (gl_VertexID + gl_BaseVertex)
        AddBaseVertexToGLVertexIDTraverser traverserVertexID;
        root->traverse(&traverserVertexID);
        if (!traverserVertexID.updateTree(compiler, root))
        {
            return false;
        }
    }

    FindGLBaseVertexTraverser traverserBaseVertex;
    root->traverse(&traverserBaseVertex);
    const TVariable *builtInVariableBaseVertex =
        traverserBaseVertex.getGLBaseVertexBuiltinVariable();

    if (builtInVariableBaseVertex)
    {
        const TType *type = StaticType::Get<EbtInt, EbpHigh, EvqUniform, 1, 1>();
        const TVariable *baseVertex =
            new TVariable(symbolTable, kEmulatedGLBaseVertexName, type, SymbolType::AngleInternal);
        const TIntermSymbol *baseVertexSymbol = new TIntermSymbol(baseVertex);

        // AngleInternal variables don't get collected
        if (shouldCollect)
        {
            uniformBaseVertex.name       = kEmulatedGLBaseVertexName.data();
            uniformBaseVertex.mappedName = kEmulatedGLBaseVertexName.data();
            uniformBaseVertex.type       = GLVariableType(*type);
            uniformBaseVertex.precision  = GLVariablePrecision(*type);
            uniformBaseVertex.staticUse = symbolTable->isStaticallyUsed(*builtInVariableBaseVertex);
            uniformBaseVertex.active    = true;
            uniformBaseVertex.binding   = type->getLayoutQualifier().binding;
            uniformBaseVertex.location  = type->getLayoutQualifier().location;
            uniformBaseVertex.offset    = type->getLayoutQualifier().offset;
            uniformBaseVertex.readonly  = type->getMemoryQualifier().readonly;
            uniformBaseVertex.writeonly = type->getMemoryQualifier().writeonly;
            addBaseVertex               = true;
        }

        DeclareGlobalVariable(root, baseVertex);
        if (!ReplaceVariableWithTyped(compiler, root, builtInVariableBaseVertex, baseVertexSymbol))
        {
            return false;
        }
    }

    FindGLBaseInstanceTraverser traverserInstance;
    root->traverse(&traverserInstance);
    const TVariable *builtInVariableBaseInstance =
        traverserInstance.getGLBaseInstanceBuiltinVariable();

    if (builtInVariableBaseInstance)
    {
        const TType *type             = StaticType::Get<EbtInt, EbpHigh, EvqUniform, 1, 1>();
        const TVariable *baseInstance = new TVariable(symbolTable, kEmulatedGLBaseInstanceName,
                                                      type, SymbolType::AngleInternal);
        const TIntermSymbol *baseInstanceSymbol = new TIntermSymbol(baseInstance);

        // AngleInternal variables don't get collected
        if (shouldCollect)
        {
            uniformBaseInstance.name       = kEmulatedGLBaseInstanceName.data();
            uniformBaseInstance.mappedName = kEmulatedGLBaseInstanceName.data();
            uniformBaseInstance.type       = GLVariableType(*type);
            uniformBaseInstance.precision  = GLVariablePrecision(*type);
            uniformBaseInstance.staticUse =
                symbolTable->isStaticallyUsed(*builtInVariableBaseInstance);
            uniformBaseInstance.active    = true;
            uniformBaseInstance.binding   = type->getLayoutQualifier().binding;
            uniformBaseInstance.location  = type->getLayoutQualifier().location;
            uniformBaseInstance.offset    = type->getLayoutQualifier().offset;
            uniformBaseInstance.readonly  = type->getMemoryQualifier().readonly;
            uniformBaseInstance.writeonly = type->getMemoryQualifier().writeonly;
            addBaseInstance               = true;
        }

        DeclareGlobalVariable(root, baseInstance);
        if (!ReplaceVariableWithTyped(compiler, root, builtInVariableBaseInstance,
                                      baseInstanceSymbol))
        {
            return false;
        }
    }

    // Make sure the order in uniforms is the same as the traverse order
    if (addBaseInstance)
    {
        uniforms->push_back(uniformBaseInstance);
    }
    if (addBaseVertex)
    {
        uniforms->push_back(uniformBaseVertex);
    }

    return true;
}

}  // namespace sh
