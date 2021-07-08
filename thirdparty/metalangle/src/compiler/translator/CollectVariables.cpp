//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// CollectVariables.cpp: Collect lists of shader interface variables based on the AST.

#include "compiler/translator/CollectVariables.h"

#include "angle_gl.h"
#include "common/utilities.h"
#include "compiler/translator/HashNames.h"
#include "compiler/translator/SymbolTable.h"
#include "compiler/translator/tree_util/IntermTraverse.h"
#include "compiler/translator/util.h"

namespace sh
{

namespace
{

BlockLayoutType GetBlockLayoutType(TLayoutBlockStorage blockStorage)
{
    switch (blockStorage)
    {
        case EbsPacked:
            return BLOCKLAYOUT_PACKED;
        case EbsShared:
            return BLOCKLAYOUT_SHARED;
        case EbsStd140:
            return BLOCKLAYOUT_STD140;
        case EbsStd430:
            return BLOCKLAYOUT_STD430;
        default:
            UNREACHABLE();
            return BLOCKLAYOUT_SHARED;
    }
}

// TODO(jiawei.shao@intel.com): implement GL_EXT_shader_io_blocks.
BlockType GetBlockType(TQualifier qualifier)
{
    switch (qualifier)
    {
        case EvqUniform:
            return BlockType::BLOCK_UNIFORM;
        case EvqBuffer:
            return BlockType::BLOCK_BUFFER;
        case EvqPerVertexIn:
            return BlockType::BLOCK_IN;
        default:
            UNREACHABLE();
            return BlockType::BLOCK_UNIFORM;
    }
}

template <class VarT>
VarT *FindVariable(const ImmutableString &name, std::vector<VarT> *infoList)
{
    // TODO(zmo): optimize this function.
    for (size_t ii = 0; ii < infoList->size(); ++ii)
    {
        if (name == (*infoList)[ii].name)
            return &((*infoList)[ii]);
    }

    return nullptr;
}

void MarkActive(ShaderVariable *variable)
{
    if (!variable->active)
    {
        if (variable->isStruct())
        {
            // Conservatively assume all fields are statically used as well.
            for (auto &field : variable->fields)
            {
                MarkActive(&field);
            }
        }
        variable->staticUse = true;
        variable->active    = true;
    }
}

ShaderVariable *FindVariableInInterfaceBlock(const ImmutableString &name,
                                             const TInterfaceBlock *interfaceBlock,
                                             std::vector<InterfaceBlock> *infoList)
{
    ASSERT(interfaceBlock);
    InterfaceBlock *namedBlock = FindVariable(interfaceBlock->name(), infoList);
    ASSERT(namedBlock);

    // Set static use on the parent interface block here
    namedBlock->staticUse = true;
    namedBlock->active    = true;
    return FindVariable(name, &namedBlock->fields);
}

// Traverses the intermediate tree to collect all attributes, uniforms, varyings, fragment outputs,
// and interface blocks.
class CollectVariablesTraverser : public TIntermTraverser
{
  public:
    CollectVariablesTraverser(std::vector<ShaderVariable> *attribs,
                              std::vector<ShaderVariable> *outputVariables,
                              std::vector<ShaderVariable> *uniforms,
                              std::vector<ShaderVariable> *inputVaryings,
                              std::vector<ShaderVariable> *outputVaryings,
                              std::vector<InterfaceBlock> *uniformBlocks,
                              std::vector<InterfaceBlock> *shaderStorageBlocks,
                              std::vector<InterfaceBlock> *inBlocks,
                              ShHashFunction64 hashFunction,
                              TSymbolTable *symbolTable,
                              GLenum shaderType,
                              const TExtensionBehavior &extensionBehavior);

    bool visitInvariantDeclaration(Visit visit, TIntermInvariantDeclaration *node) override;
    void visitSymbol(TIntermSymbol *symbol) override;
    bool visitDeclaration(Visit, TIntermDeclaration *node) override;
    bool visitBinary(Visit visit, TIntermBinary *binaryNode) override;

  private:
    std::string getMappedName(const TSymbol *symbol) const;

    void setFieldOrVariableProperties(const TType &type,
                                      bool staticUse,
                                      ShaderVariable *variableOut) const;
    void setFieldProperties(const TType &type,
                            const ImmutableString &name,
                            bool staticUse,
                            ShaderVariable *variableOut) const;
    void setCommonVariableProperties(const TType &type,
                                     const TVariable &variable,
                                     ShaderVariable *variableOut) const;

    ShaderVariable recordAttribute(const TIntermSymbol &variable) const;
    ShaderVariable recordOutputVariable(const TIntermSymbol &variable) const;
    ShaderVariable recordVarying(const TIntermSymbol &variable) const;
    void recordInterfaceBlock(const char *instanceName,
                              const TType &interfaceBlockType,
                              InterfaceBlock *interfaceBlock) const;
    ShaderVariable recordUniform(const TIntermSymbol &variable) const;

    void setBuiltInInfoFromSymbol(const TVariable &variable, ShaderVariable *info);

    void recordBuiltInVaryingUsed(const TVariable &variable,
                                  bool *addedFlag,
                                  std::vector<ShaderVariable> *varyings);
    void recordBuiltInFragmentOutputUsed(const TVariable &variable, bool *addedFlag);
    void recordBuiltInAttributeUsed(const TVariable &variable, bool *addedFlag);
    InterfaceBlock *recordGLInUsed(const TType &glInType);
    InterfaceBlock *findNamedInterfaceBlock(const ImmutableString &name) const;

    std::vector<ShaderVariable> *mAttribs;
    std::vector<ShaderVariable> *mOutputVariables;
    std::vector<ShaderVariable> *mUniforms;
    std::vector<ShaderVariable> *mInputVaryings;
    std::vector<ShaderVariable> *mOutputVaryings;
    std::vector<InterfaceBlock> *mUniformBlocks;
    std::vector<InterfaceBlock> *mShaderStorageBlocks;
    std::vector<InterfaceBlock> *mInBlocks;

    std::map<std::string, ShaderVariable *> mInterfaceBlockFields;

    // Shader uniforms
    bool mDepthRangeAdded;

    // Vertex Shader builtins
    bool mInstanceIDAdded;
    bool mVertexIDAdded;
    bool mPointSizeAdded;
    bool mDrawIDAdded;
    bool mBaseVertexAdded;
    bool mBaseInstanceAdded;

    // Vertex Shader and Geometry Shader builtins
    bool mPositionAdded;
    bool mClipDistanceAdded;

    // Fragment Shader builtins
    bool mPointCoordAdded;
    bool mFrontFacingAdded;
    bool mFragCoordAdded;
    bool mLastFragDataAdded;
    bool mFragColorAdded;
    bool mFragDataAdded;
    bool mFragDepthEXTAdded;
    bool mFragDepthAdded;
    bool mSecondaryFragColorEXTAdded;
    bool mSecondaryFragDataEXTAdded;

    // Geometry Shader builtins
    bool mPerVertexInAdded;
    bool mPrimitiveIDInAdded;
    bool mInvocationIDAdded;

    // Geometry Shader and Fragment Shader builtins
    bool mPrimitiveIDAdded;
    bool mLayerAdded;

    ShHashFunction64 mHashFunction;

    GLenum mShaderType;
    const TExtensionBehavior &mExtensionBehavior;
};

CollectVariablesTraverser::CollectVariablesTraverser(
    std::vector<sh::ShaderVariable> *attribs,
    std::vector<sh::ShaderVariable> *outputVariables,
    std::vector<sh::ShaderVariable> *uniforms,
    std::vector<sh::ShaderVariable> *inputVaryings,
    std::vector<sh::ShaderVariable> *outputVaryings,
    std::vector<sh::InterfaceBlock> *uniformBlocks,
    std::vector<sh::InterfaceBlock> *shaderStorageBlocks,
    std::vector<sh::InterfaceBlock> *inBlocks,
    ShHashFunction64 hashFunction,
    TSymbolTable *symbolTable,
    GLenum shaderType,
    const TExtensionBehavior &extensionBehavior)
    : TIntermTraverser(true, false, false, symbolTable),
      mAttribs(attribs),
      mOutputVariables(outputVariables),
      mUniforms(uniforms),
      mInputVaryings(inputVaryings),
      mOutputVaryings(outputVaryings),
      mUniformBlocks(uniformBlocks),
      mShaderStorageBlocks(shaderStorageBlocks),
      mInBlocks(inBlocks),
      mDepthRangeAdded(false),
      mInstanceIDAdded(false),
      mVertexIDAdded(false),
      mPointSizeAdded(false),
      mDrawIDAdded(false),
      mBaseVertexAdded(false),
      mBaseInstanceAdded(false),
      mPositionAdded(false),
      mClipDistanceAdded(false),
      mPointCoordAdded(false),
      mFrontFacingAdded(false),
      mFragCoordAdded(false),
      mLastFragDataAdded(false),
      mFragColorAdded(false),
      mFragDataAdded(false),
      mFragDepthEXTAdded(false),
      mFragDepthAdded(false),
      mSecondaryFragColorEXTAdded(false),
      mSecondaryFragDataEXTAdded(false),
      mPerVertexInAdded(false),
      mPrimitiveIDInAdded(false),
      mInvocationIDAdded(false),
      mPrimitiveIDAdded(false),
      mLayerAdded(false),
      mHashFunction(hashFunction),
      mShaderType(shaderType),
      mExtensionBehavior(extensionBehavior)
{}

std::string CollectVariablesTraverser::getMappedName(const TSymbol *symbol) const
{
    return HashName(symbol, mHashFunction, nullptr).data();
}

void CollectVariablesTraverser::setBuiltInInfoFromSymbol(const TVariable &variable,
                                                         ShaderVariable *info)
{
    const TType &type = variable.getType();

    info->name       = variable.name().data();
    info->mappedName = variable.name().data();
    info->type       = GLVariableType(type);
    info->precision  = GLVariablePrecision(type);
    if (auto *arraySizes = type.getArraySizes())
    {
        info->arraySizes.assign(arraySizes->begin(), arraySizes->end());
    }
}

void CollectVariablesTraverser::recordBuiltInVaryingUsed(const TVariable &variable,
                                                         bool *addedFlag,
                                                         std::vector<ShaderVariable> *varyings)
{
    ASSERT(varyings);
    if (!(*addedFlag))
    {
        ShaderVariable info;
        setBuiltInInfoFromSymbol(variable, &info);
        info.staticUse   = true;
        info.active      = true;
        info.isInvariant = mSymbolTable->isVaryingInvariant(variable);
        varyings->push_back(info);
        (*addedFlag) = true;
    }
}

void CollectVariablesTraverser::recordBuiltInFragmentOutputUsed(const TVariable &variable,
                                                                bool *addedFlag)
{
    if (!(*addedFlag))
    {
        ShaderVariable info;
        setBuiltInInfoFromSymbol(variable, &info);
        info.staticUse = true;
        info.active    = true;
        mOutputVariables->push_back(info);
        (*addedFlag) = true;
    }
}

void CollectVariablesTraverser::recordBuiltInAttributeUsed(const TVariable &variable,
                                                           bool *addedFlag)
{
    if (!(*addedFlag))
    {
        ShaderVariable info;
        setBuiltInInfoFromSymbol(variable, &info);
        info.staticUse = true;
        info.active    = true;
        info.location  = -1;
        mAttribs->push_back(info);
        (*addedFlag) = true;
    }
}

InterfaceBlock *CollectVariablesTraverser::recordGLInUsed(const TType &glInType)
{
    if (!mPerVertexInAdded)
    {
        ASSERT(glInType.getQualifier() == EvqPerVertexIn);
        InterfaceBlock info;
        recordInterfaceBlock("gl_in", glInType, &info);

        mPerVertexInAdded = true;
        mInBlocks->push_back(info);
        return &mInBlocks->back();
    }
    else
    {
        return FindVariable(ImmutableString("gl_PerVertex"), mInBlocks);
    }
}

bool CollectVariablesTraverser::visitInvariantDeclaration(Visit visit,
                                                          TIntermInvariantDeclaration *node)
{
    // We should not mark variables as active just based on an invariant declaration, so we don't
    // traverse the symbols declared invariant.
    return false;
}

// We want to check whether a uniform/varying is active because we need to skip updating inactive
// ones. We also only count the active ones in packing computing. Also, gl_FragCoord, gl_PointCoord,
// and gl_FrontFacing count toward varying counting if they are active in a fragment shader.
void CollectVariablesTraverser::visitSymbol(TIntermSymbol *symbol)
{
    ASSERT(symbol != nullptr);

    if (symbol->variable().symbolType() == SymbolType::AngleInternal ||
        symbol->variable().symbolType() == SymbolType::Empty)
    {
        // Internal variables or nameless variables are not collected.
        return;
    }

    ShaderVariable *var = nullptr;

    const ImmutableString &symbolName = symbol->getName();

    // Check the qualifier from the variable, not from the symbol node. The node may have a
    // different qualifier if it's the result of a folded ternary node.
    TQualifier qualifier = symbol->variable().getType().getQualifier();

    if (IsVaryingIn(qualifier))
    {
        var = FindVariable(symbolName, mInputVaryings);
    }
    else if (IsVaryingOut(qualifier))
    {
        var = FindVariable(symbolName, mOutputVaryings);
    }
    else if (symbol->getType().getBasicType() == EbtInterfaceBlock)
    {
        UNREACHABLE();
    }
    else if (symbolName == "gl_DepthRange")
    {
        ASSERT(qualifier == EvqUniform);

        if (!mDepthRangeAdded)
        {
            ShaderVariable info;
            const char kName[] = "gl_DepthRange";
            info.name          = kName;
            info.mappedName    = kName;
            info.type          = GL_NONE;
            info.precision     = GL_NONE;
            info.staticUse     = true;
            info.active        = true;

            ShaderVariable nearInfo(GL_FLOAT);
            const char kNearName[] = "near";
            nearInfo.name          = kNearName;
            nearInfo.mappedName    = kNearName;
            nearInfo.precision     = GL_HIGH_FLOAT;
            nearInfo.staticUse     = true;
            nearInfo.active        = true;

            ShaderVariable farInfo(GL_FLOAT);
            const char kFarName[] = "far";
            farInfo.name          = kFarName;
            farInfo.mappedName    = kFarName;
            farInfo.precision     = GL_HIGH_FLOAT;
            farInfo.staticUse     = true;
            farInfo.active        = true;

            ShaderVariable diffInfo(GL_FLOAT);
            const char kDiffName[] = "diff";
            diffInfo.name          = kDiffName;
            diffInfo.mappedName    = kDiffName;
            diffInfo.precision     = GL_HIGH_FLOAT;
            diffInfo.staticUse     = true;
            diffInfo.active        = true;

            info.fields.push_back(nearInfo);
            info.fields.push_back(farInfo);
            info.fields.push_back(diffInfo);

            mUniforms->push_back(info);
            mDepthRangeAdded = true;
        }
    }
    else
    {
        switch (qualifier)
        {
            case EvqAttribute:
            case EvqVertexIn:
                var = FindVariable(symbolName, mAttribs);
                break;
            case EvqFragmentOut:
                var = FindVariable(symbolName, mOutputVariables);
                break;
            case EvqUniform:
            {
                const TInterfaceBlock *interfaceBlock = symbol->getType().getInterfaceBlock();
                if (interfaceBlock)
                {
                    var = FindVariableInInterfaceBlock(symbolName, interfaceBlock, mUniformBlocks);
                }
                else
                {
                    var = FindVariable(symbolName, mUniforms);
                }

                // It's an internal error to reference an undefined user uniform
                ASSERT(!symbolName.beginsWith("gl_") || var);
            }
            break;
            case EvqBuffer:
            {
                const TInterfaceBlock *interfaceBlock = symbol->getType().getInterfaceBlock();
                var =
                    FindVariableInInterfaceBlock(symbolName, interfaceBlock, mShaderStorageBlocks);
            }
            break;
            case EvqFragCoord:
                recordBuiltInVaryingUsed(symbol->variable(), &mFragCoordAdded, mInputVaryings);
                return;
            case EvqFrontFacing:
                recordBuiltInVaryingUsed(symbol->variable(), &mFrontFacingAdded, mInputVaryings);
                return;
            case EvqPointCoord:
                recordBuiltInVaryingUsed(symbol->variable(), &mPointCoordAdded, mInputVaryings);
                return;
            case EvqInstanceID:
                // Whenever the SH_INITIALIZE_BUILTINS_FOR_INSTANCED_MULTIVIEW option is set,
                // gl_InstanceID is added inside expressions to initialize ViewID_OVR and
                // InstanceID. Note that gl_InstanceID is not added to the symbol table for ESSL1
                // shaders.
                recordBuiltInAttributeUsed(symbol->variable(), &mInstanceIDAdded);
                return;
            case EvqVertexID:
                recordBuiltInAttributeUsed(symbol->variable(), &mVertexIDAdded);
                return;
            case EvqPosition:
                recordBuiltInVaryingUsed(symbol->variable(), &mPositionAdded, mOutputVaryings);
                return;
            case EvqPointSize:
                recordBuiltInVaryingUsed(symbol->variable(), &mPointSizeAdded, mOutputVaryings);
                return;
            case EvqDrawID:
                recordBuiltInAttributeUsed(symbol->variable(), &mDrawIDAdded);
                return;
            case EvqBaseVertex:
                recordBuiltInAttributeUsed(symbol->variable(), &mBaseVertexAdded);
                return;
            case EvqBaseInstance:
                recordBuiltInAttributeUsed(symbol->variable(), &mBaseInstanceAdded);
                return;
            case EvqLastFragData:
                recordBuiltInVaryingUsed(symbol->variable(), &mLastFragDataAdded, mInputVaryings);
                return;
            case EvqFragColor:
                recordBuiltInFragmentOutputUsed(symbol->variable(), &mFragColorAdded);
                return;
            case EvqFragData:
                if (!mFragDataAdded)
                {
                    ShaderVariable info;
                    setBuiltInInfoFromSymbol(symbol->variable(), &info);
                    if (!IsExtensionEnabled(mExtensionBehavior, TExtension::EXT_draw_buffers))
                    {
                        ASSERT(info.arraySizes.size() == 1u);
                        info.arraySizes.back() = 1u;
                    }
                    info.staticUse = true;
                    info.active    = true;
                    mOutputVariables->push_back(info);
                    mFragDataAdded = true;
                }
                return;
            case EvqFragDepthEXT:
                recordBuiltInFragmentOutputUsed(symbol->variable(), &mFragDepthEXTAdded);
                return;
            case EvqFragDepth:
                recordBuiltInFragmentOutputUsed(symbol->variable(), &mFragDepthAdded);
                return;
            case EvqSecondaryFragColorEXT:
                recordBuiltInFragmentOutputUsed(symbol->variable(), &mSecondaryFragColorEXTAdded);
                return;
            case EvqSecondaryFragDataEXT:
                recordBuiltInFragmentOutputUsed(symbol->variable(), &mSecondaryFragDataEXTAdded);
                return;
            case EvqInvocationID:
                recordBuiltInVaryingUsed(symbol->variable(), &mInvocationIDAdded, mInputVaryings);
                break;
            case EvqPrimitiveIDIn:
                recordBuiltInVaryingUsed(symbol->variable(), &mPrimitiveIDInAdded, mInputVaryings);
                break;
            case EvqPrimitiveID:
                if (mShaderType == GL_GEOMETRY_SHADER_EXT)
                {
                    recordBuiltInVaryingUsed(symbol->variable(), &mPrimitiveIDAdded,
                                             mOutputVaryings);
                }
                else
                {
                    ASSERT(mShaderType == GL_FRAGMENT_SHADER);
                    recordBuiltInVaryingUsed(symbol->variable(), &mPrimitiveIDAdded,
                                             mInputVaryings);
                }
                break;
            case EvqLayer:
                if (mShaderType == GL_GEOMETRY_SHADER_EXT)
                {
                    recordBuiltInVaryingUsed(symbol->variable(), &mLayerAdded, mOutputVaryings);
                }
                else if (mShaderType == GL_FRAGMENT_SHADER)
                {
                    recordBuiltInVaryingUsed(symbol->variable(), &mLayerAdded, mInputVaryings);
                }
                else
                {
                    ASSERT(mShaderType == GL_VERTEX_SHADER &&
                           (IsExtensionEnabled(mExtensionBehavior, TExtension::OVR_multiview2) ||
                            IsExtensionEnabled(mExtensionBehavior, TExtension::OVR_multiview)));
                }
                break;
            case EvqClipDistance:
                recordBuiltInVaryingUsed(symbol->variable(), &mClipDistanceAdded, mOutputVaryings);
                return;
            default:
                break;
        }
    }
    if (var)
    {
        MarkActive(var);
    }
}

void CollectVariablesTraverser::setFieldOrVariableProperties(const TType &type,
                                                             bool staticUse,
                                                             ShaderVariable *variableOut) const
{
    ASSERT(variableOut);

    variableOut->staticUse = staticUse;

    const TStructure *structure = type.getStruct();
    if (!structure)
    {
        variableOut->type      = GLVariableType(type);
        variableOut->precision = GLVariablePrecision(type);
    }
    else
    {
        // Structures use a NONE type that isn't exposed outside ANGLE.
        variableOut->type = GL_NONE;
        if (structure->symbolType() != SymbolType::Empty)
        {
            variableOut->structName = structure->name().data();
        }

        const TFieldList &fields = structure->fields();

        for (const TField *field : fields)
        {
            // Regardless of the variable type (uniform, in/out etc.) its fields are always plain
            // ShaderVariable objects.
            ShaderVariable fieldVariable;
            setFieldProperties(*field->type(), field->name(), staticUse, &fieldVariable);
            variableOut->fields.push_back(fieldVariable);
        }
    }
    if (auto *arraySizes = type.getArraySizes())
    {
        variableOut->arraySizes.assign(arraySizes->begin(), arraySizes->end());
    }
}

void CollectVariablesTraverser::setFieldProperties(const TType &type,
                                                   const ImmutableString &name,
                                                   bool staticUse,
                                                   ShaderVariable *variableOut) const
{
    ASSERT(variableOut);
    setFieldOrVariableProperties(type, staticUse, variableOut);
    variableOut->name.assign(name.data(), name.length());
    variableOut->mappedName = HashName(name, mHashFunction, nullptr).data();
}

void CollectVariablesTraverser::setCommonVariableProperties(const TType &type,
                                                            const TVariable &variable,
                                                            ShaderVariable *variableOut) const
{
    ASSERT(variableOut);

    variableOut->staticUse = mSymbolTable->isStaticallyUsed(variable);
    setFieldOrVariableProperties(type, variableOut->staticUse, variableOut);
    ASSERT(variable.symbolType() != SymbolType::Empty);
    variableOut->name.assign(variable.name().data(), variable.name().length());
    variableOut->mappedName = getMappedName(&variable);
}

ShaderVariable CollectVariablesTraverser::recordAttribute(const TIntermSymbol &variable) const
{
    const TType &type = variable.getType();
    ASSERT(!type.getStruct());

    ShaderVariable attribute;
    setCommonVariableProperties(type, variable.variable(), &attribute);

    attribute.location = type.getLayoutQualifier().location;
    return attribute;
}

ShaderVariable CollectVariablesTraverser::recordOutputVariable(const TIntermSymbol &variable) const
{
    const TType &type = variable.getType();
    ASSERT(!type.getStruct());

    ShaderVariable outputVariable;
    setCommonVariableProperties(type, variable.variable(), &outputVariable);

    outputVariable.location = type.getLayoutQualifier().location;
    outputVariable.index    = type.getLayoutQualifier().index;
    return outputVariable;
}

ShaderVariable CollectVariablesTraverser::recordVarying(const TIntermSymbol &variable) const
{
    const TType &type = variable.getType();

    ShaderVariable varying;
    setCommonVariableProperties(type, variable.variable(), &varying);
    varying.location = type.getLayoutQualifier().location;

    switch (type.getQualifier())
    {
        case EvqVaryingIn:
        case EvqVaryingOut:
        case EvqVertexOut:
        case EvqSmoothOut:
        case EvqFlatOut:
        case EvqCentroidOut:
        case EvqGeometryOut:
            if (mSymbolTable->isVaryingInvariant(variable.variable()) || type.isInvariant())
            {
                varying.isInvariant = true;
            }
            break;
        default:
            break;
    }

    varying.interpolation = GetInterpolationType(type.getQualifier());
    return varying;
}

// TODO(jiawei.shao@intel.com): implement GL_EXT_shader_io_blocks.
void CollectVariablesTraverser::recordInterfaceBlock(const char *instanceName,
                                                     const TType &interfaceBlockType,
                                                     InterfaceBlock *interfaceBlock) const
{
    ASSERT(interfaceBlockType.getBasicType() == EbtInterfaceBlock);
    ASSERT(interfaceBlock);

    const TInterfaceBlock *blockType = interfaceBlockType.getInterfaceBlock();
    ASSERT(blockType);

    interfaceBlock->name       = blockType->name().data();
    interfaceBlock->mappedName = getMappedName(blockType);
    if (instanceName != nullptr)
    {
        interfaceBlock->instanceName = instanceName;
        const TSymbol *blockSymbol   = nullptr;
        if (strncmp(instanceName, "gl_in", 5u) == 0)
        {
            blockSymbol = mSymbolTable->getGlInVariableWithArraySize();
        }
        else
        {
            blockSymbol = mSymbolTable->findGlobal(ImmutableString(instanceName));
        }
        ASSERT(blockSymbol && blockSymbol->isVariable());
        interfaceBlock->staticUse =
            mSymbolTable->isStaticallyUsed(*static_cast<const TVariable *>(blockSymbol));
    }
    ASSERT(!interfaceBlockType.isArrayOfArrays());  // Disallowed by GLSL ES 3.10 section 4.3.9
    interfaceBlock->arraySize =
        interfaceBlockType.isArray() ? interfaceBlockType.getOutermostArraySize() : 0;

    interfaceBlock->blockType = GetBlockType(interfaceBlockType.getQualifier());
    if (interfaceBlock->blockType == BlockType::BLOCK_UNIFORM ||
        interfaceBlock->blockType == BlockType::BLOCK_BUFFER)
    {
        // TODO(oetuaho): Remove setting isRowMajorLayout.
        interfaceBlock->isRowMajorLayout = false;
        interfaceBlock->binding          = blockType->blockBinding();
        interfaceBlock->layout           = GetBlockLayoutType(blockType->blockStorage());
    }

    // Gather field information
    bool anyFieldStaticallyUsed = false;
    for (const TField *field : blockType->fields())
    {
        const TType &fieldType = *field->type();

        bool staticUse = false;
        if (instanceName == nullptr)
        {
            // Static use of individual fields has been recorded, since they are present in the
            // symbol table as variables.
            const TSymbol *fieldSymbol = mSymbolTable->findGlobal(field->name());
            ASSERT(fieldSymbol && fieldSymbol->isVariable());
            staticUse =
                mSymbolTable->isStaticallyUsed(*static_cast<const TVariable *>(fieldSymbol));
            if (staticUse)
            {
                anyFieldStaticallyUsed = true;
            }
        }

        ShaderVariable fieldVariable;
        setFieldProperties(fieldType, field->name(), staticUse, &fieldVariable);
        fieldVariable.isRowMajorLayout =
            (fieldType.getLayoutQualifier().matrixPacking == EmpRowMajor);
        interfaceBlock->fields.push_back(fieldVariable);
    }
    if (anyFieldStaticallyUsed)
    {
        interfaceBlock->staticUse = true;
    }
}

ShaderVariable CollectVariablesTraverser::recordUniform(const TIntermSymbol &variable) const
{
    ShaderVariable uniform;
    setCommonVariableProperties(variable.getType(), variable.variable(), &uniform);
    uniform.binding = variable.getType().getLayoutQualifier().binding;
    uniform.imageUnitFormat =
        GetImageInternalFormatType(variable.getType().getLayoutQualifier().imageInternalFormat);
    uniform.location  = variable.getType().getLayoutQualifier().location;
    uniform.offset    = variable.getType().getLayoutQualifier().offset;
    uniform.readonly  = variable.getType().getMemoryQualifier().readonly;
    uniform.writeonly = variable.getType().getMemoryQualifier().writeonly;
    return uniform;
}

bool CollectVariablesTraverser::visitDeclaration(Visit, TIntermDeclaration *node)
{
    const TIntermSequence &sequence = *(node->getSequence());
    ASSERT(!sequence.empty());

    const TIntermTyped &typedNode = *(sequence.front()->getAsTyped());
    TQualifier qualifier          = typedNode.getQualifier();

    bool isShaderVariable = qualifier == EvqAttribute || qualifier == EvqVertexIn ||
                            qualifier == EvqFragmentOut || qualifier == EvqUniform ||
                            IsVarying(qualifier);

    if (typedNode.getBasicType() != EbtInterfaceBlock && !isShaderVariable)
    {
        return true;
    }

    for (TIntermNode *variableNode : sequence)
    {
        // The only case in which the sequence will not contain a TIntermSymbol node is
        // initialization. It will contain a TInterBinary node in that case. Since attributes,
        // uniforms, varyings, outputs and interface blocks cannot be initialized in a shader, we
        // must have only TIntermSymbol nodes in the sequence in the cases we are interested in.
        const TIntermSymbol &variable = *variableNode->getAsSymbolNode();
        if (variable.variable().symbolType() == SymbolType::AngleInternal)
        {
            // Internal variables are not collected.
            continue;
        }

        // TODO(jiawei.shao@intel.com): implement GL_EXT_shader_io_blocks.
        if (typedNode.getBasicType() == EbtInterfaceBlock)
        {
            InterfaceBlock interfaceBlock;
            recordInterfaceBlock(variable.variable().symbolType() != SymbolType::Empty
                                     ? variable.getName().data()
                                     : nullptr,
                                 variable.getType(), &interfaceBlock);

            switch (qualifier)
            {
                case EvqUniform:
                    mUniformBlocks->push_back(interfaceBlock);
                    break;
                case EvqBuffer:
                    mShaderStorageBlocks->push_back(interfaceBlock);
                    break;
                default:
                    UNREACHABLE();
            }
        }
        else
        {
            ASSERT(variable.variable().symbolType() != SymbolType::Empty);
            switch (qualifier)
            {
                case EvqAttribute:
                case EvqVertexIn:
                    mAttribs->push_back(recordAttribute(variable));
                    break;
                case EvqFragmentOut:
                    mOutputVariables->push_back(recordOutputVariable(variable));
                    break;
                case EvqUniform:
                    mUniforms->push_back(recordUniform(variable));
                    break;
                default:
                    if (IsVaryingIn(qualifier))
                    {
                        mInputVaryings->push_back(recordVarying(variable));
                    }
                    else
                    {
                        ASSERT(IsVaryingOut(qualifier));
                        mOutputVaryings->push_back(recordVarying(variable));
                    }
                    break;
            }
        }
    }

    // None of the recorded variables can have initializers, so we don't need to traverse the
    // declarators.
    return false;
}

// TODO(jiawei.shao@intel.com): add search on mInBlocks and mOutBlocks when implementing
// GL_EXT_shader_io_blocks.
InterfaceBlock *CollectVariablesTraverser::findNamedInterfaceBlock(
    const ImmutableString &blockName) const
{
    InterfaceBlock *namedBlock = FindVariable(blockName, mUniformBlocks);
    if (!namedBlock)
    {
        namedBlock = FindVariable(blockName, mShaderStorageBlocks);
    }
    return namedBlock;
}

bool CollectVariablesTraverser::visitBinary(Visit, TIntermBinary *binaryNode)
{
    if (binaryNode->getOp() == EOpIndexDirectInterfaceBlock)
    {
        // NOTE: we do not determine static use / activeness for individual blocks of an array.
        TIntermTyped *blockNode = binaryNode->getLeft()->getAsTyped();
        ASSERT(blockNode);

        TIntermConstantUnion *constantUnion = binaryNode->getRight()->getAsConstantUnion();
        ASSERT(constantUnion);

        InterfaceBlock *namedBlock = nullptr;

        bool traverseIndexExpression         = false;
        TIntermBinary *interfaceIndexingNode = blockNode->getAsBinaryNode();
        if (interfaceIndexingNode)
        {
            TIntermTyped *interfaceNode = interfaceIndexingNode->getLeft()->getAsTyped();
            ASSERT(interfaceNode);

            const TType &interfaceType = interfaceNode->getType();
            if (interfaceType.getQualifier() == EvqPerVertexIn)
            {
                namedBlock = recordGLInUsed(interfaceType);
                ASSERT(namedBlock);

                // We need to continue traversing to collect useful variables in the index
                // expression of gl_in.
                traverseIndexExpression = true;
            }
        }

        const TInterfaceBlock *interfaceBlock = blockNode->getType().getInterfaceBlock();
        if (!namedBlock)
        {
            namedBlock = findNamedInterfaceBlock(interfaceBlock->name());
        }
        ASSERT(namedBlock);
        ASSERT(namedBlock->staticUse);
        namedBlock->active      = true;
        unsigned int fieldIndex = static_cast<unsigned int>(constantUnion->getIConst(0));
        ASSERT(fieldIndex < namedBlock->fields.size());
        // TODO(oetuaho): Would be nicer to record static use of fields of named interface blocks
        // more accurately at parse time - now we only mark the fields statically used if they are
        // active. http://anglebug.com/2440
        // We need to mark this field and all of its sub-fields, as static/active
        MarkActive(&namedBlock->fields[fieldIndex]);

        if (traverseIndexExpression)
        {
            ASSERT(interfaceIndexingNode);
            interfaceIndexingNode->getRight()->traverse(this);
        }
        return false;
    }

    return true;
}

}  // anonymous namespace

void CollectVariables(TIntermBlock *root,
                      std::vector<ShaderVariable> *attributes,
                      std::vector<ShaderVariable> *outputVariables,
                      std::vector<ShaderVariable> *uniforms,
                      std::vector<ShaderVariable> *inputVaryings,
                      std::vector<ShaderVariable> *outputVaryings,
                      std::vector<InterfaceBlock> *uniformBlocks,
                      std::vector<InterfaceBlock> *shaderStorageBlocks,
                      std::vector<InterfaceBlock> *inBlocks,
                      ShHashFunction64 hashFunction,
                      TSymbolTable *symbolTable,
                      GLenum shaderType,
                      const TExtensionBehavior &extensionBehavior)
{
    CollectVariablesTraverser collect(attributes, outputVariables, uniforms, inputVaryings,
                                      outputVaryings, uniformBlocks, shaderStorageBlocks, inBlocks,
                                      hashFunction, symbolTable, shaderType, extensionBehavior);
    root->traverse(&collect);
}

}  // namespace sh
