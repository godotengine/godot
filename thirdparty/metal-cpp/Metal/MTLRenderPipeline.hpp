#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"
#include "MTLAllocation.hpp"

namespace MTL {
    class Device;
    class Function;
    class FunctionHandle;
    class IntersectionFunctionTable;
    class IntersectionFunctionTableDescriptor;
    class LinkedFunctions;
    class PipelineBufferDescriptorArray;
    class VertexDescriptor;
    class VisibleFunctionTable;
    class VisibleFunctionTableDescriptor;
    enum PixelFormat : NS::UInteger;
    using RenderStages = NS::UInteger;
    enum ShaderValidation : NS::Integer;
    enum Winding : NS::UInteger;
}
namespace MTL4 {
    class BinaryFunction;
    class PipelineDescriptor;
    class RenderPipelineBinaryFunctionsDescriptor;
}
namespace NS {
    class Array;
    class Error;
    class String;
}

namespace MTL
{

_MTL_ENUM(NS::UInteger, BlendFactor) {
    BlendFactorZero = 0,
    BlendFactorOne = 1,
    BlendFactorSourceColor = 2,
    BlendFactorOneMinusSourceColor = 3,
    BlendFactorSourceAlpha = 4,
    BlendFactorOneMinusSourceAlpha = 5,
    BlendFactorDestinationColor = 6,
    BlendFactorOneMinusDestinationColor = 7,
    BlendFactorDestinationAlpha = 8,
    BlendFactorOneMinusDestinationAlpha = 9,
    BlendFactorSourceAlphaSaturated = 10,
    BlendFactorBlendColor = 11,
    BlendFactorOneMinusBlendColor = 12,
    BlendFactorBlendAlpha = 13,
    BlendFactorOneMinusBlendAlpha = 14,
    BlendFactorSource1Color = 15,
    BlendFactorOneMinusSource1Color = 16,
    BlendFactorSource1Alpha = 17,
    BlendFactorOneMinusSource1Alpha = 18,
    BlendFactorUnspecialized = 19,
};

_MTL_ENUM(NS::UInteger, BlendOperation) {
    BlendOperationAdd = 0,
    BlendOperationSubtract = 1,
    BlendOperationReverseSubtract = 2,
    BlendOperationMin = 3,
    BlendOperationMax = 4,
    BlendOperationUnspecialized = 5,
};

_MTL_OPTIONS(NS::UInteger, ColorWriteMask) {
    ColorWriteMaskNone = 0,
    ColorWriteMaskRed = 0x1 << 3,
    ColorWriteMaskGreen = 0x1 << 2,
    ColorWriteMaskBlue = 0x1 << 1,
    ColorWriteMaskAlpha = 0x1 << 0,
    ColorWriteMaskAll = 0xf,
    ColorWriteMaskUnspecialized = 0x10,
};

_MTL_ENUM(NS::UInteger, PrimitiveTopologyClass) {
    PrimitiveTopologyClassUnspecified = 0,
    PrimitiveTopologyClassPoint = 1,
    PrimitiveTopologyClassLine = 2,
    PrimitiveTopologyClassTriangle = 3,
};

_MTL_ENUM(NS::UInteger, TessellationPartitionMode) {
    TessellationPartitionModePow2 = 0,
    TessellationPartitionModeInteger = 1,
    TessellationPartitionModeFractionalOdd = 2,
    TessellationPartitionModeFractionalEven = 3,
};

_MTL_ENUM(NS::UInteger, TessellationFactorStepFunction) {
    TessellationFactorStepFunctionConstant = 0,
    TessellationFactorStepFunctionPerPatch = 1,
    TessellationFactorStepFunctionPerInstance = 2,
    TessellationFactorStepFunctionPerPatchAndPerInstance = 3,
};

_MTL_ENUM(NS::UInteger, TessellationFactorFormat) {
    TessellationFactorFormatHalf = 0,
};

_MTL_ENUM(NS::UInteger, TessellationControlPointIndexType) {
    TessellationControlPointIndexTypeNone = 0,
    TessellationControlPointIndexTypeUInt16 = 1,
    TessellationControlPointIndexTypeUInt32 = 2,
};


class RenderPipelineColorAttachmentDescriptor;
class LogicalToPhysicalColorAttachmentMap;
class RenderPipelineReflection;
class RenderPipelineDescriptor;
class RenderPipelineFunctionsDescriptor;
class RenderPipelineState;
class RenderPipelineColorAttachmentDescriptorArray;
class TileRenderPipelineColorAttachmentDescriptor;
class TileRenderPipelineColorAttachmentDescriptorArray;
class TileRenderPipelineDescriptor;
class MeshRenderPipelineDescriptor;

class RenderPipelineColorAttachmentDescriptor : public NS::Copying<RenderPipelineColorAttachmentDescriptor>
{
public:
    static RenderPipelineColorAttachmentDescriptor* alloc();
    RenderPipelineColorAttachmentDescriptor*        init() const;

    MTL::BlendOperation alphaBlendOperation() const;
    bool                blendingEnabled() const;
    MTL::BlendFactor    destinationAlphaBlendFactor() const;
    MTL::BlendFactor    destinationRGBBlendFactor() const;
    bool                isBlendingEnabled();
    MTL::PixelFormat    pixelFormat() const;
    MTL::BlendOperation rgbBlendOperation() const;
    void                setAlphaBlendOperation(MTL::BlendOperation alphaBlendOperation);
    void                setBlendingEnabled(bool blendingEnabled);
    void                setDestinationAlphaBlendFactor(MTL::BlendFactor destinationAlphaBlendFactor);
    void                setDestinationRGBBlendFactor(MTL::BlendFactor destinationRGBBlendFactor);
    void                setPixelFormat(MTL::PixelFormat pixelFormat);
    void                setRgbBlendOperation(MTL::BlendOperation rgbBlendOperation);
    void                setSourceAlphaBlendFactor(MTL::BlendFactor sourceAlphaBlendFactor);
    void                setSourceRGBBlendFactor(MTL::BlendFactor sourceRGBBlendFactor);
    void                setWriteMask(MTL::ColorWriteMask writeMask);
    MTL::BlendFactor    sourceAlphaBlendFactor() const;
    MTL::BlendFactor    sourceRGBBlendFactor() const;
    MTL::ColorWriteMask writeMask() const;

};

class LogicalToPhysicalColorAttachmentMap : public NS::Copying<LogicalToPhysicalColorAttachmentMap>
{
public:
    static LogicalToPhysicalColorAttachmentMap* alloc();
    LogicalToPhysicalColorAttachmentMap*        init() const;

    NS::UInteger getPhysicalIndex(NS::UInteger logicalIndex);
    void         reset();
    void         setPhysicalIndex(NS::UInteger physicalIndex, NS::UInteger logicalIndex);

};

class RenderPipelineReflection : public NS::Referencing<RenderPipelineReflection>
{
public:
    static RenderPipelineReflection* alloc();
    RenderPipelineReflection*        init() const;

    NS::Array* fragmentArguments() const;
    NS::Array* fragmentBindings() const;
    NS::Array* meshBindings() const;
    NS::Array* objectBindings() const;
    NS::Array* tileArguments() const;
    NS::Array* tileBindings() const;
    NS::Array* vertexArguments() const;
    NS::Array* vertexBindings() const;

};

class RenderPipelineDescriptor : public NS::Copying<RenderPipelineDescriptor>
{
public:
    static RenderPipelineDescriptor* alloc();
    RenderPipelineDescriptor*        init() const;

    bool                                               alphaToCoverageEnabled() const;
    bool                                               alphaToOneEnabled() const;
    NS::Array*                                         binaryArchives() const;
    MTL::RenderPipelineColorAttachmentDescriptorArray* colorAttachments() const;
    MTL::PixelFormat                                   depthAttachmentPixelFormat() const;
    MTL::PipelineBufferDescriptorArray*                fragmentBuffers() const;
    MTL::Function*                                     fragmentFunction() const;
    MTL::LinkedFunctions*                              fragmentLinkedFunctions() const;
    NS::Array*                                         fragmentPreloadedLibraries() const;
    MTL::PrimitiveTopologyClass                        inputPrimitiveTopology() const;
    bool                                               isAlphaToCoverageEnabled();
    bool                                               isAlphaToOneEnabled();
    bool                                               isRasterizationEnabled();
    bool                                               isTessellationFactorScaleEnabled();
    NS::String*                                        label() const;
    NS::UInteger                                       maxFragmentCallStackDepth() const;
    NS::UInteger                                       maxTessellationFactor() const;
    NS::UInteger                                       maxVertexAmplificationCount() const;
    NS::UInteger                                       maxVertexCallStackDepth() const;
    NS::UInteger                                       rasterSampleCount() const;
    bool                                               rasterizationEnabled() const;
    void                                               reset();
    NS::UInteger                                       sampleCount() const;
    void                                               setAlphaToCoverageEnabled(bool alphaToCoverageEnabled);
    void                                               setAlphaToOneEnabled(bool alphaToOneEnabled);
    void                                               setBinaryArchives(NS::Array* binaryArchives);
    void                                               setDepthAttachmentPixelFormat(MTL::PixelFormat depthAttachmentPixelFormat);
    void                                               setFragmentFunction(MTL::Function* fragmentFunction);
    void                                               setFragmentLinkedFunctions(MTL::LinkedFunctions* fragmentLinkedFunctions);
    void                                               setFragmentPreloadedLibraries(NS::Array* fragmentPreloadedLibraries);
    void                                               setInputPrimitiveTopology(MTL::PrimitiveTopologyClass inputPrimitiveTopology);
    void                                               setLabel(NS::String* label);
    void                                               setMaxFragmentCallStackDepth(NS::UInteger maxFragmentCallStackDepth);
    void                                               setMaxTessellationFactor(NS::UInteger maxTessellationFactor);
    void                                               setMaxVertexAmplificationCount(NS::UInteger maxVertexAmplificationCount);
    void                                               setMaxVertexCallStackDepth(NS::UInteger maxVertexCallStackDepth);
    void                                               setRasterSampleCount(NS::UInteger rasterSampleCount);
    void                                               setRasterizationEnabled(bool rasterizationEnabled);
    void                                               setSampleCount(NS::UInteger sampleCount);
    void                                               setShaderValidation(MTL::ShaderValidation shaderValidation);
    void                                               setStencilAttachmentPixelFormat(MTL::PixelFormat stencilAttachmentPixelFormat);
    void                                               setSupportAddingFragmentBinaryFunctions(bool supportAddingFragmentBinaryFunctions);
    void                                               setSupportAddingVertexBinaryFunctions(bool supportAddingVertexBinaryFunctions);
    void                                               setSupportIndirectCommandBuffers(bool supportIndirectCommandBuffers);
    void                                               setTessellationControlPointIndexType(MTL::TessellationControlPointIndexType tessellationControlPointIndexType);
    void                                               setTessellationFactorFormat(MTL::TessellationFactorFormat tessellationFactorFormat);
    void                                               setTessellationFactorScaleEnabled(bool tessellationFactorScaleEnabled);
    void                                               setTessellationFactorStepFunction(MTL::TessellationFactorStepFunction tessellationFactorStepFunction);
    void                                               setTessellationOutputWindingOrder(MTL::Winding tessellationOutputWindingOrder);
    void                                               setTessellationPartitionMode(MTL::TessellationPartitionMode tessellationPartitionMode);
    void                                               setVertexDescriptor(MTL::VertexDescriptor* vertexDescriptor);
    void                                               setVertexFunction(MTL::Function* vertexFunction);
    void                                               setVertexLinkedFunctions(MTL::LinkedFunctions* vertexLinkedFunctions);
    void                                               setVertexPreloadedLibraries(NS::Array* vertexPreloadedLibraries);
    MTL::ShaderValidation                              shaderValidation() const;
    MTL::PixelFormat                                   stencilAttachmentPixelFormat() const;
    bool                                               supportAddingFragmentBinaryFunctions() const;
    bool                                               supportAddingVertexBinaryFunctions() const;
    bool                                               supportIndirectCommandBuffers() const;
    MTL::TessellationControlPointIndexType             tessellationControlPointIndexType() const;
    MTL::TessellationFactorFormat                      tessellationFactorFormat() const;
    bool                                               tessellationFactorScaleEnabled() const;
    MTL::TessellationFactorStepFunction                tessellationFactorStepFunction() const;
    MTL::Winding                                       tessellationOutputWindingOrder() const;
    MTL::TessellationPartitionMode                     tessellationPartitionMode() const;
    MTL::PipelineBufferDescriptorArray*                vertexBuffers() const;
    MTL::VertexDescriptor*                             vertexDescriptor() const;
    MTL::Function*                                     vertexFunction() const;
    MTL::LinkedFunctions*                              vertexLinkedFunctions() const;
    NS::Array*                                         vertexPreloadedLibraries() const;

};

class RenderPipelineFunctionsDescriptor : public NS::Copying<RenderPipelineFunctionsDescriptor>
{
public:
    static RenderPipelineFunctionsDescriptor* alloc();
    RenderPipelineFunctionsDescriptor*        init() const;

    NS::Array* fragmentAdditionalBinaryFunctions() const;
    void       setFragmentAdditionalBinaryFunctions(NS::Array* fragmentAdditionalBinaryFunctions);
    void       setTileAdditionalBinaryFunctions(NS::Array* tileAdditionalBinaryFunctions);
    void       setVertexAdditionalBinaryFunctions(NS::Array* vertexAdditionalBinaryFunctions);
    NS::Array* tileAdditionalBinaryFunctions() const;
    NS::Array* vertexAdditionalBinaryFunctions() const;

};

class RenderPipelineState : public NS::Referencing<RenderPipelineState, MTL::Allocation>
{
public:
    MTL::Device*                    device() const;
    MTL::FunctionHandle*            functionHandle(NS::String* name, MTL::RenderStages stage);
    MTL::FunctionHandle*            functionHandle(MTL4::BinaryFunction* function, MTL::RenderStages stage);
    MTL::FunctionHandle*            functionHandle(MTL::Function* function, MTL::RenderStages stage);
    MTL::ResourceID                 gpuResourceID() const;
    NS::UInteger                    imageblockMemoryLength(MTL::Size imageblockDimensions);
    NS::UInteger                    imageblockSampleLength() const;
    NS::String*                     label() const;
    NS::UInteger                    maxTotalThreadgroupsPerMeshGrid() const;
    NS::UInteger                    maxTotalThreadsPerMeshThreadgroup() const;
    NS::UInteger                    maxTotalThreadsPerObjectThreadgroup() const;
    NS::UInteger                    maxTotalThreadsPerThreadgroup() const;
    NS::UInteger                    meshThreadExecutionWidth() const;
    MTL::IntersectionFunctionTable* newIntersectionFunctionTable(MTL::IntersectionFunctionTableDescriptor* descriptor, MTL::RenderStages stage);
    MTL4::PipelineDescriptor*       newRenderPipelineDescriptorForSpecialization();
    MTL::RenderPipelineState*       newRenderPipelineState(MTL4::RenderPipelineBinaryFunctionsDescriptor* binaryFunctionsDescriptor, NS::Error** error);
    MTL::RenderPipelineState*       newRenderPipelineState(MTL::RenderPipelineFunctionsDescriptor* additionalBinaryFunctions, NS::Error** error);
    MTL::VisibleFunctionTable*      newVisibleFunctionTable(MTL::VisibleFunctionTableDescriptor* descriptor, MTL::RenderStages stage);
    NS::UInteger                    objectThreadExecutionWidth() const;
    MTL::RenderPipelineReflection*  reflection() const;
    MTL::Size                       requiredThreadsPerMeshThreadgroup() const;
    MTL::Size                       requiredThreadsPerObjectThreadgroup() const;
    MTL::Size                       requiredThreadsPerTileThreadgroup() const;
    MTL::ShaderValidation           shaderValidation() const;
    bool                            supportIndirectCommandBuffers() const;
    bool                            threadgroupSizeMatchesTileSize() const;

};

class RenderPipelineColorAttachmentDescriptorArray : public NS::Referencing<RenderPipelineColorAttachmentDescriptorArray>
{
public:
    static RenderPipelineColorAttachmentDescriptorArray* alloc();
    RenderPipelineColorAttachmentDescriptorArray*        init() const;

    MTL::RenderPipelineColorAttachmentDescriptor* object(NS::UInteger attachmentIndex);
    void                                          setObject(MTL::RenderPipelineColorAttachmentDescriptor* attachment, NS::UInteger attachmentIndex);

};

class TileRenderPipelineColorAttachmentDescriptor : public NS::Copying<TileRenderPipelineColorAttachmentDescriptor>
{
public:
    static TileRenderPipelineColorAttachmentDescriptor* alloc();
    TileRenderPipelineColorAttachmentDescriptor*        init() const;

    MTL::PixelFormat pixelFormat() const;
    void             setPixelFormat(MTL::PixelFormat pixelFormat);

};

class TileRenderPipelineColorAttachmentDescriptorArray : public NS::Referencing<TileRenderPipelineColorAttachmentDescriptorArray>
{
public:
    static TileRenderPipelineColorAttachmentDescriptorArray* alloc();
    TileRenderPipelineColorAttachmentDescriptorArray*        init() const;

    MTL::TileRenderPipelineColorAttachmentDescriptor* object(NS::UInteger attachmentIndex);
    void                                              setObject(MTL::TileRenderPipelineColorAttachmentDescriptor* attachment, NS::UInteger attachmentIndex);

};

class TileRenderPipelineDescriptor : public NS::Copying<TileRenderPipelineDescriptor>
{
public:
    static TileRenderPipelineDescriptor* alloc();
    TileRenderPipelineDescriptor*        init() const;

    NS::Array*                                             binaryArchives() const;
    MTL::TileRenderPipelineColorAttachmentDescriptorArray* colorAttachments() const;
    NS::String*                                            label() const;
    MTL::LinkedFunctions*                                  linkedFunctions() const;
    NS::UInteger                                           maxCallStackDepth() const;
    NS::UInteger                                           maxTotalThreadsPerThreadgroup() const;
    NS::Array*                                             preloadedLibraries() const;
    NS::UInteger                                           rasterSampleCount() const;
    MTL::Size                                              requiredThreadsPerThreadgroup() const;
    void                                                   reset();
    void                                                   setBinaryArchives(NS::Array* binaryArchives);
    void                                                   setLabel(NS::String* label);
    void                                                   setLinkedFunctions(MTL::LinkedFunctions* linkedFunctions);
    void                                                   setMaxCallStackDepth(NS::UInteger maxCallStackDepth);
    void                                                   setMaxTotalThreadsPerThreadgroup(NS::UInteger maxTotalThreadsPerThreadgroup);
    void                                                   setPreloadedLibraries(NS::Array* preloadedLibraries);
    void                                                   setRasterSampleCount(NS::UInteger rasterSampleCount);
    void                                                   setRequiredThreadsPerThreadgroup(MTL::Size requiredThreadsPerThreadgroup);
    void                                                   setShaderValidation(MTL::ShaderValidation shaderValidation);
    void                                                   setSupportAddingBinaryFunctions(bool supportAddingBinaryFunctions);
    void                                                   setThreadgroupSizeMatchesTileSize(bool threadgroupSizeMatchesTileSize);
    void                                                   setTileFunction(MTL::Function* tileFunction);
    MTL::ShaderValidation                                  shaderValidation() const;
    bool                                                   supportAddingBinaryFunctions() const;
    bool                                                   threadgroupSizeMatchesTileSize() const;
    MTL::PipelineBufferDescriptorArray*                    tileBuffers() const;
    MTL::Function*                                         tileFunction() const;

};

class MeshRenderPipelineDescriptor : public NS::Copying<MeshRenderPipelineDescriptor>
{
public:
    static MeshRenderPipelineDescriptor* alloc();
    MeshRenderPipelineDescriptor*        init() const;

    bool                                               alphaToCoverageEnabled() const;
    bool                                               alphaToOneEnabled() const;
    NS::Array*                                         binaryArchives() const;
    MTL::RenderPipelineColorAttachmentDescriptorArray* colorAttachments() const;
    MTL::PixelFormat                                   depthAttachmentPixelFormat() const;
    MTL::PipelineBufferDescriptorArray*                fragmentBuffers() const;
    MTL::Function*                                     fragmentFunction() const;
    MTL::LinkedFunctions*                              fragmentLinkedFunctions() const;
    bool                                               isAlphaToCoverageEnabled();
    bool                                               isAlphaToOneEnabled();
    bool                                               isRasterizationEnabled();
    NS::String*                                        label() const;
    NS::UInteger                                       maxTotalThreadgroupsPerMeshGrid() const;
    NS::UInteger                                       maxTotalThreadsPerMeshThreadgroup() const;
    NS::UInteger                                       maxTotalThreadsPerObjectThreadgroup() const;
    NS::UInteger                                       maxVertexAmplificationCount() const;
    MTL::PipelineBufferDescriptorArray*                meshBuffers() const;
    MTL::Function*                                     meshFunction() const;
    MTL::LinkedFunctions*                              meshLinkedFunctions() const;
    bool                                               meshThreadgroupSizeIsMultipleOfThreadExecutionWidth() const;
    MTL::PipelineBufferDescriptorArray*                objectBuffers() const;
    MTL::Function*                                     objectFunction() const;
    MTL::LinkedFunctions*                              objectLinkedFunctions() const;
    bool                                               objectThreadgroupSizeIsMultipleOfThreadExecutionWidth() const;
    NS::UInteger                                       payloadMemoryLength() const;
    NS::UInteger                                       rasterSampleCount() const;
    bool                                               rasterizationEnabled() const;
    MTL::Size                                          requiredThreadsPerMeshThreadgroup() const;
    MTL::Size                                          requiredThreadsPerObjectThreadgroup() const;
    void                                               reset();
    void                                               setAlphaToCoverageEnabled(bool alphaToCoverageEnabled);
    void                                               setAlphaToOneEnabled(bool alphaToOneEnabled);
    void                                               setBinaryArchives(NS::Array* binaryArchives);
    void                                               setDepthAttachmentPixelFormat(MTL::PixelFormat depthAttachmentPixelFormat);
    void                                               setFragmentFunction(MTL::Function* fragmentFunction);
    void                                               setFragmentLinkedFunctions(MTL::LinkedFunctions* fragmentLinkedFunctions);
    void                                               setLabel(NS::String* label);
    void                                               setMaxTotalThreadgroupsPerMeshGrid(NS::UInteger maxTotalThreadgroupsPerMeshGrid);
    void                                               setMaxTotalThreadsPerMeshThreadgroup(NS::UInteger maxTotalThreadsPerMeshThreadgroup);
    void                                               setMaxTotalThreadsPerObjectThreadgroup(NS::UInteger maxTotalThreadsPerObjectThreadgroup);
    void                                               setMaxVertexAmplificationCount(NS::UInteger maxVertexAmplificationCount);
    void                                               setMeshFunction(MTL::Function* meshFunction);
    void                                               setMeshLinkedFunctions(MTL::LinkedFunctions* meshLinkedFunctions);
    void                                               setMeshThreadgroupSizeIsMultipleOfThreadExecutionWidth(bool meshThreadgroupSizeIsMultipleOfThreadExecutionWidth);
    void                                               setObjectFunction(MTL::Function* objectFunction);
    void                                               setObjectLinkedFunctions(MTL::LinkedFunctions* objectLinkedFunctions);
    void                                               setObjectThreadgroupSizeIsMultipleOfThreadExecutionWidth(bool objectThreadgroupSizeIsMultipleOfThreadExecutionWidth);
    void                                               setPayloadMemoryLength(NS::UInteger payloadMemoryLength);
    void                                               setRasterSampleCount(NS::UInteger rasterSampleCount);
    void                                               setRasterizationEnabled(bool rasterizationEnabled);
    void                                               setRequiredThreadsPerMeshThreadgroup(MTL::Size requiredThreadsPerMeshThreadgroup);
    void                                               setRequiredThreadsPerObjectThreadgroup(MTL::Size requiredThreadsPerObjectThreadgroup);
    void                                               setShaderValidation(MTL::ShaderValidation shaderValidation);
    void                                               setStencilAttachmentPixelFormat(MTL::PixelFormat stencilAttachmentPixelFormat);
    void                                               setSupportIndirectCommandBuffers(bool supportIndirectCommandBuffers);
    MTL::ShaderValidation                              shaderValidation() const;
    MTL::PixelFormat                                   stencilAttachmentPixelFormat() const;
    bool                                               supportIndirectCommandBuffers() const;

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLRenderPipelineColorAttachmentDescriptor;
extern "C" void *OBJC_CLASS_$_MTLLogicalToPhysicalColorAttachmentMap;
extern "C" void *OBJC_CLASS_$_MTLRenderPipelineReflection;
extern "C" void *OBJC_CLASS_$_MTLRenderPipelineDescriptor;
extern "C" void *OBJC_CLASS_$_MTLRenderPipelineFunctionsDescriptor;
extern "C" void *OBJC_CLASS_$_MTLRenderPipelineState;
extern "C" void *OBJC_CLASS_$_MTLRenderPipelineColorAttachmentDescriptorArray;
extern "C" void *OBJC_CLASS_$_MTLTileRenderPipelineColorAttachmentDescriptor;
extern "C" void *OBJC_CLASS_$_MTLTileRenderPipelineColorAttachmentDescriptorArray;
extern "C" void *OBJC_CLASS_$_MTLTileRenderPipelineDescriptor;
extern "C" void *OBJC_CLASS_$_MTLMeshRenderPipelineDescriptor;

_MTL_INLINE MTL::RenderPipelineColorAttachmentDescriptor* MTL::RenderPipelineColorAttachmentDescriptor::alloc()
{
    return _MTL_msg_MTL__RenderPipelineColorAttachmentDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLRenderPipelineColorAttachmentDescriptor, nullptr);
}

_MTL_INLINE MTL::RenderPipelineColorAttachmentDescriptor* MTL::RenderPipelineColorAttachmentDescriptor::init() const
{
    return _MTL_msg_MTL__RenderPipelineColorAttachmentDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::PixelFormat MTL::RenderPipelineColorAttachmentDescriptor::pixelFormat() const
{
    return _MTL_msg_MTL__PixelFormat_pixelFormat((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineColorAttachmentDescriptor::setPixelFormat(MTL::PixelFormat pixelFormat)
{
    _MTL_msg_v_setPixelFormat__MTL__PixelFormat((const void*)this, nullptr, pixelFormat);
}

_MTL_INLINE bool MTL::RenderPipelineColorAttachmentDescriptor::blendingEnabled() const
{
    return _MTL_msg_bool_blendingEnabled((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineColorAttachmentDescriptor::setBlendingEnabled(bool blendingEnabled)
{
    _MTL_msg_v_setBlendingEnabled__bool((const void*)this, nullptr, blendingEnabled);
}

_MTL_INLINE MTL::BlendFactor MTL::RenderPipelineColorAttachmentDescriptor::sourceRGBBlendFactor() const
{
    return _MTL_msg_MTL__BlendFactor_sourceRGBBlendFactor((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineColorAttachmentDescriptor::setSourceRGBBlendFactor(MTL::BlendFactor sourceRGBBlendFactor)
{
    _MTL_msg_v_setSourceRGBBlendFactor__MTL__BlendFactor((const void*)this, nullptr, sourceRGBBlendFactor);
}

_MTL_INLINE MTL::BlendFactor MTL::RenderPipelineColorAttachmentDescriptor::destinationRGBBlendFactor() const
{
    return _MTL_msg_MTL__BlendFactor_destinationRGBBlendFactor((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineColorAttachmentDescriptor::setDestinationRGBBlendFactor(MTL::BlendFactor destinationRGBBlendFactor)
{
    _MTL_msg_v_setDestinationRGBBlendFactor__MTL__BlendFactor((const void*)this, nullptr, destinationRGBBlendFactor);
}

_MTL_INLINE MTL::BlendOperation MTL::RenderPipelineColorAttachmentDescriptor::rgbBlendOperation() const
{
    return _MTL_msg_MTL__BlendOperation_rgbBlendOperation((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineColorAttachmentDescriptor::setRgbBlendOperation(MTL::BlendOperation rgbBlendOperation)
{
    _MTL_msg_v_setRgbBlendOperation__MTL__BlendOperation((const void*)this, nullptr, rgbBlendOperation);
}

_MTL_INLINE MTL::BlendFactor MTL::RenderPipelineColorAttachmentDescriptor::sourceAlphaBlendFactor() const
{
    return _MTL_msg_MTL__BlendFactor_sourceAlphaBlendFactor((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineColorAttachmentDescriptor::setSourceAlphaBlendFactor(MTL::BlendFactor sourceAlphaBlendFactor)
{
    _MTL_msg_v_setSourceAlphaBlendFactor__MTL__BlendFactor((const void*)this, nullptr, sourceAlphaBlendFactor);
}

_MTL_INLINE MTL::BlendFactor MTL::RenderPipelineColorAttachmentDescriptor::destinationAlphaBlendFactor() const
{
    return _MTL_msg_MTL__BlendFactor_destinationAlphaBlendFactor((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineColorAttachmentDescriptor::setDestinationAlphaBlendFactor(MTL::BlendFactor destinationAlphaBlendFactor)
{
    _MTL_msg_v_setDestinationAlphaBlendFactor__MTL__BlendFactor((const void*)this, nullptr, destinationAlphaBlendFactor);
}

_MTL_INLINE MTL::BlendOperation MTL::RenderPipelineColorAttachmentDescriptor::alphaBlendOperation() const
{
    return _MTL_msg_MTL__BlendOperation_alphaBlendOperation((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineColorAttachmentDescriptor::setAlphaBlendOperation(MTL::BlendOperation alphaBlendOperation)
{
    _MTL_msg_v_setAlphaBlendOperation__MTL__BlendOperation((const void*)this, nullptr, alphaBlendOperation);
}

_MTL_INLINE MTL::ColorWriteMask MTL::RenderPipelineColorAttachmentDescriptor::writeMask() const
{
    return _MTL_msg_MTL__ColorWriteMask_writeMask((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineColorAttachmentDescriptor::setWriteMask(MTL::ColorWriteMask writeMask)
{
    _MTL_msg_v_setWriteMask__MTL__ColorWriteMask((const void*)this, nullptr, writeMask);
}

_MTL_INLINE bool MTL::RenderPipelineColorAttachmentDescriptor::isBlendingEnabled()
{
    return _MTL_msg_bool_isBlendingEnabled((const void*)this, nullptr);
}

_MTL_INLINE MTL::LogicalToPhysicalColorAttachmentMap* MTL::LogicalToPhysicalColorAttachmentMap::alloc()
{
    return _MTL_msg_MTL__LogicalToPhysicalColorAttachmentMapp_alloc((const void*)&OBJC_CLASS_$_MTLLogicalToPhysicalColorAttachmentMap, nullptr);
}

_MTL_INLINE MTL::LogicalToPhysicalColorAttachmentMap* MTL::LogicalToPhysicalColorAttachmentMap::init() const
{
    return _MTL_msg_MTL__LogicalToPhysicalColorAttachmentMapp_init((const void*)this, nullptr);
}

_MTL_INLINE void MTL::LogicalToPhysicalColorAttachmentMap::setPhysicalIndex(NS::UInteger physicalIndex, NS::UInteger logicalIndex)
{
    _MTL_msg_v_setPhysicalIndex_forLogicalIndex__NS__UInteger_NS__UInteger((const void*)this, nullptr, physicalIndex, logicalIndex);
}

_MTL_INLINE NS::UInteger MTL::LogicalToPhysicalColorAttachmentMap::getPhysicalIndex(NS::UInteger logicalIndex)
{
    return _MTL_msg_NS__UInteger_getPhysicalIndexForLogicalIndex__NS__UInteger((const void*)this, nullptr, logicalIndex);
}

_MTL_INLINE void MTL::LogicalToPhysicalColorAttachmentMap::reset()
{
    _MTL_msg_v_reset((const void*)this, nullptr);
}

_MTL_INLINE MTL::RenderPipelineReflection* MTL::RenderPipelineReflection::alloc()
{
    return _MTL_msg_MTL__RenderPipelineReflectionp_alloc((const void*)&OBJC_CLASS_$_MTLRenderPipelineReflection, nullptr);
}

_MTL_INLINE MTL::RenderPipelineReflection* MTL::RenderPipelineReflection::init() const
{
    return _MTL_msg_MTL__RenderPipelineReflectionp_init((const void*)this, nullptr);
}

_MTL_INLINE NS::Array* MTL::RenderPipelineReflection::vertexBindings() const
{
    return _MTL_msg_NS__Arrayp_vertexBindings((const void*)this, nullptr);
}

_MTL_INLINE NS::Array* MTL::RenderPipelineReflection::fragmentBindings() const
{
    return _MTL_msg_NS__Arrayp_fragmentBindings((const void*)this, nullptr);
}

_MTL_INLINE NS::Array* MTL::RenderPipelineReflection::tileBindings() const
{
    return _MTL_msg_NS__Arrayp_tileBindings((const void*)this, nullptr);
}

_MTL_INLINE NS::Array* MTL::RenderPipelineReflection::objectBindings() const
{
    return _MTL_msg_NS__Arrayp_objectBindings((const void*)this, nullptr);
}

_MTL_INLINE NS::Array* MTL::RenderPipelineReflection::meshBindings() const
{
    return _MTL_msg_NS__Arrayp_meshBindings((const void*)this, nullptr);
}

_MTL_INLINE NS::Array* MTL::RenderPipelineReflection::vertexArguments() const
{
    return _MTL_msg_NS__Arrayp_vertexArguments((const void*)this, nullptr);
}

_MTL_INLINE NS::Array* MTL::RenderPipelineReflection::fragmentArguments() const
{
    return _MTL_msg_NS__Arrayp_fragmentArguments((const void*)this, nullptr);
}

_MTL_INLINE NS::Array* MTL::RenderPipelineReflection::tileArguments() const
{
    return _MTL_msg_NS__Arrayp_tileArguments((const void*)this, nullptr);
}

_MTL_INLINE MTL::RenderPipelineDescriptor* MTL::RenderPipelineDescriptor::alloc()
{
    return _MTL_msg_MTL__RenderPipelineDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLRenderPipelineDescriptor, nullptr);
}

_MTL_INLINE MTL::RenderPipelineDescriptor* MTL::RenderPipelineDescriptor::init() const
{
    return _MTL_msg_MTL__RenderPipelineDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE NS::String* MTL::RenderPipelineDescriptor::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setLabel(NS::String* label)
{
    _MTL_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL_INLINE MTL::Function* MTL::RenderPipelineDescriptor::vertexFunction() const
{
    return _MTL_msg_MTL__Functionp_vertexFunction((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setVertexFunction(MTL::Function* vertexFunction)
{
    _MTL_msg_v_setVertexFunction__MTL__Functionp((const void*)this, nullptr, vertexFunction);
}

_MTL_INLINE MTL::Function* MTL::RenderPipelineDescriptor::fragmentFunction() const
{
    return _MTL_msg_MTL__Functionp_fragmentFunction((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setFragmentFunction(MTL::Function* fragmentFunction)
{
    _MTL_msg_v_setFragmentFunction__MTL__Functionp((const void*)this, nullptr, fragmentFunction);
}

_MTL_INLINE MTL::VertexDescriptor* MTL::RenderPipelineDescriptor::vertexDescriptor() const
{
    return _MTL_msg_MTL__VertexDescriptorp_vertexDescriptor((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setVertexDescriptor(MTL::VertexDescriptor* vertexDescriptor)
{
    _MTL_msg_v_setVertexDescriptor__MTL__VertexDescriptorp((const void*)this, nullptr, vertexDescriptor);
}

_MTL_INLINE NS::UInteger MTL::RenderPipelineDescriptor::sampleCount() const
{
    return _MTL_msg_NS__UInteger_sampleCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setSampleCount(NS::UInteger sampleCount)
{
    _MTL_msg_v_setSampleCount__NS__UInteger((const void*)this, nullptr, sampleCount);
}

_MTL_INLINE NS::UInteger MTL::RenderPipelineDescriptor::rasterSampleCount() const
{
    return _MTL_msg_NS__UInteger_rasterSampleCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setRasterSampleCount(NS::UInteger rasterSampleCount)
{
    _MTL_msg_v_setRasterSampleCount__NS__UInteger((const void*)this, nullptr, rasterSampleCount);
}

_MTL_INLINE bool MTL::RenderPipelineDescriptor::alphaToCoverageEnabled() const
{
    return _MTL_msg_bool_alphaToCoverageEnabled((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setAlphaToCoverageEnabled(bool alphaToCoverageEnabled)
{
    _MTL_msg_v_setAlphaToCoverageEnabled__bool((const void*)this, nullptr, alphaToCoverageEnabled);
}

_MTL_INLINE bool MTL::RenderPipelineDescriptor::alphaToOneEnabled() const
{
    return _MTL_msg_bool_alphaToOneEnabled((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setAlphaToOneEnabled(bool alphaToOneEnabled)
{
    _MTL_msg_v_setAlphaToOneEnabled__bool((const void*)this, nullptr, alphaToOneEnabled);
}

_MTL_INLINE bool MTL::RenderPipelineDescriptor::rasterizationEnabled() const
{
    return _MTL_msg_bool_rasterizationEnabled((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setRasterizationEnabled(bool rasterizationEnabled)
{
    _MTL_msg_v_setRasterizationEnabled__bool((const void*)this, nullptr, rasterizationEnabled);
}

_MTL_INLINE NS::UInteger MTL::RenderPipelineDescriptor::maxVertexAmplificationCount() const
{
    return _MTL_msg_NS__UInteger_maxVertexAmplificationCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setMaxVertexAmplificationCount(NS::UInteger maxVertexAmplificationCount)
{
    _MTL_msg_v_setMaxVertexAmplificationCount__NS__UInteger((const void*)this, nullptr, maxVertexAmplificationCount);
}

_MTL_INLINE MTL::RenderPipelineColorAttachmentDescriptorArray* MTL::RenderPipelineDescriptor::colorAttachments() const
{
    return _MTL_msg_MTL__RenderPipelineColorAttachmentDescriptorArrayp_colorAttachments((const void*)this, nullptr);
}

_MTL_INLINE MTL::PixelFormat MTL::RenderPipelineDescriptor::depthAttachmentPixelFormat() const
{
    return _MTL_msg_MTL__PixelFormat_depthAttachmentPixelFormat((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setDepthAttachmentPixelFormat(MTL::PixelFormat depthAttachmentPixelFormat)
{
    _MTL_msg_v_setDepthAttachmentPixelFormat__MTL__PixelFormat((const void*)this, nullptr, depthAttachmentPixelFormat);
}

_MTL_INLINE MTL::PixelFormat MTL::RenderPipelineDescriptor::stencilAttachmentPixelFormat() const
{
    return _MTL_msg_MTL__PixelFormat_stencilAttachmentPixelFormat((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setStencilAttachmentPixelFormat(MTL::PixelFormat stencilAttachmentPixelFormat)
{
    _MTL_msg_v_setStencilAttachmentPixelFormat__MTL__PixelFormat((const void*)this, nullptr, stencilAttachmentPixelFormat);
}

_MTL_INLINE MTL::PrimitiveTopologyClass MTL::RenderPipelineDescriptor::inputPrimitiveTopology() const
{
    return _MTL_msg_MTL__PrimitiveTopologyClass_inputPrimitiveTopology((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setInputPrimitiveTopology(MTL::PrimitiveTopologyClass inputPrimitiveTopology)
{
    _MTL_msg_v_setInputPrimitiveTopology__MTL__PrimitiveTopologyClass((const void*)this, nullptr, inputPrimitiveTopology);
}

_MTL_INLINE MTL::TessellationPartitionMode MTL::RenderPipelineDescriptor::tessellationPartitionMode() const
{
    return _MTL_msg_MTL__TessellationPartitionMode_tessellationPartitionMode((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setTessellationPartitionMode(MTL::TessellationPartitionMode tessellationPartitionMode)
{
    _MTL_msg_v_setTessellationPartitionMode__MTL__TessellationPartitionMode((const void*)this, nullptr, tessellationPartitionMode);
}

_MTL_INLINE NS::UInteger MTL::RenderPipelineDescriptor::maxTessellationFactor() const
{
    return _MTL_msg_NS__UInteger_maxTessellationFactor((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setMaxTessellationFactor(NS::UInteger maxTessellationFactor)
{
    _MTL_msg_v_setMaxTessellationFactor__NS__UInteger((const void*)this, nullptr, maxTessellationFactor);
}

_MTL_INLINE bool MTL::RenderPipelineDescriptor::tessellationFactorScaleEnabled() const
{
    return _MTL_msg_bool_tessellationFactorScaleEnabled((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setTessellationFactorScaleEnabled(bool tessellationFactorScaleEnabled)
{
    _MTL_msg_v_setTessellationFactorScaleEnabled__bool((const void*)this, nullptr, tessellationFactorScaleEnabled);
}

_MTL_INLINE MTL::TessellationFactorFormat MTL::RenderPipelineDescriptor::tessellationFactorFormat() const
{
    return _MTL_msg_MTL__TessellationFactorFormat_tessellationFactorFormat((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setTessellationFactorFormat(MTL::TessellationFactorFormat tessellationFactorFormat)
{
    _MTL_msg_v_setTessellationFactorFormat__MTL__TessellationFactorFormat((const void*)this, nullptr, tessellationFactorFormat);
}

_MTL_INLINE MTL::TessellationControlPointIndexType MTL::RenderPipelineDescriptor::tessellationControlPointIndexType() const
{
    return _MTL_msg_MTL__TessellationControlPointIndexType_tessellationControlPointIndexType((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setTessellationControlPointIndexType(MTL::TessellationControlPointIndexType tessellationControlPointIndexType)
{
    _MTL_msg_v_setTessellationControlPointIndexType__MTL__TessellationControlPointIndexType((const void*)this, nullptr, tessellationControlPointIndexType);
}

_MTL_INLINE MTL::TessellationFactorStepFunction MTL::RenderPipelineDescriptor::tessellationFactorStepFunction() const
{
    return _MTL_msg_MTL__TessellationFactorStepFunction_tessellationFactorStepFunction((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setTessellationFactorStepFunction(MTL::TessellationFactorStepFunction tessellationFactorStepFunction)
{
    _MTL_msg_v_setTessellationFactorStepFunction__MTL__TessellationFactorStepFunction((const void*)this, nullptr, tessellationFactorStepFunction);
}

_MTL_INLINE MTL::Winding MTL::RenderPipelineDescriptor::tessellationOutputWindingOrder() const
{
    return _MTL_msg_MTL__Winding_tessellationOutputWindingOrder((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setTessellationOutputWindingOrder(MTL::Winding tessellationOutputWindingOrder)
{
    _MTL_msg_v_setTessellationOutputWindingOrder__MTL__Winding((const void*)this, nullptr, tessellationOutputWindingOrder);
}

_MTL_INLINE MTL::PipelineBufferDescriptorArray* MTL::RenderPipelineDescriptor::vertexBuffers() const
{
    return _MTL_msg_MTL__PipelineBufferDescriptorArrayp_vertexBuffers((const void*)this, nullptr);
}

_MTL_INLINE MTL::PipelineBufferDescriptorArray* MTL::RenderPipelineDescriptor::fragmentBuffers() const
{
    return _MTL_msg_MTL__PipelineBufferDescriptorArrayp_fragmentBuffers((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::RenderPipelineDescriptor::supportIndirectCommandBuffers() const
{
    return _MTL_msg_bool_supportIndirectCommandBuffers((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setSupportIndirectCommandBuffers(bool supportIndirectCommandBuffers)
{
    _MTL_msg_v_setSupportIndirectCommandBuffers__bool((const void*)this, nullptr, supportIndirectCommandBuffers);
}

_MTL_INLINE NS::Array* MTL::RenderPipelineDescriptor::binaryArchives() const
{
    return _MTL_msg_NS__Arrayp_binaryArchives((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setBinaryArchives(NS::Array* binaryArchives)
{
    _MTL_msg_v_setBinaryArchives__NS__Arrayp((const void*)this, nullptr, binaryArchives);
}

_MTL_INLINE NS::Array* MTL::RenderPipelineDescriptor::vertexPreloadedLibraries() const
{
    return _MTL_msg_NS__Arrayp_vertexPreloadedLibraries((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setVertexPreloadedLibraries(NS::Array* vertexPreloadedLibraries)
{
    _MTL_msg_v_setVertexPreloadedLibraries__NS__Arrayp((const void*)this, nullptr, vertexPreloadedLibraries);
}

_MTL_INLINE NS::Array* MTL::RenderPipelineDescriptor::fragmentPreloadedLibraries() const
{
    return _MTL_msg_NS__Arrayp_fragmentPreloadedLibraries((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setFragmentPreloadedLibraries(NS::Array* fragmentPreloadedLibraries)
{
    _MTL_msg_v_setFragmentPreloadedLibraries__NS__Arrayp((const void*)this, nullptr, fragmentPreloadedLibraries);
}

_MTL_INLINE MTL::LinkedFunctions* MTL::RenderPipelineDescriptor::vertexLinkedFunctions() const
{
    return _MTL_msg_MTL__LinkedFunctionsp_vertexLinkedFunctions((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setVertexLinkedFunctions(MTL::LinkedFunctions* vertexLinkedFunctions)
{
    _MTL_msg_v_setVertexLinkedFunctions__MTL__LinkedFunctionsp((const void*)this, nullptr, vertexLinkedFunctions);
}

_MTL_INLINE MTL::LinkedFunctions* MTL::RenderPipelineDescriptor::fragmentLinkedFunctions() const
{
    return _MTL_msg_MTL__LinkedFunctionsp_fragmentLinkedFunctions((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setFragmentLinkedFunctions(MTL::LinkedFunctions* fragmentLinkedFunctions)
{
    _MTL_msg_v_setFragmentLinkedFunctions__MTL__LinkedFunctionsp((const void*)this, nullptr, fragmentLinkedFunctions);
}

_MTL_INLINE bool MTL::RenderPipelineDescriptor::supportAddingVertexBinaryFunctions() const
{
    return _MTL_msg_bool_supportAddingVertexBinaryFunctions((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setSupportAddingVertexBinaryFunctions(bool supportAddingVertexBinaryFunctions)
{
    _MTL_msg_v_setSupportAddingVertexBinaryFunctions__bool((const void*)this, nullptr, supportAddingVertexBinaryFunctions);
}

_MTL_INLINE bool MTL::RenderPipelineDescriptor::supportAddingFragmentBinaryFunctions() const
{
    return _MTL_msg_bool_supportAddingFragmentBinaryFunctions((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setSupportAddingFragmentBinaryFunctions(bool supportAddingFragmentBinaryFunctions)
{
    _MTL_msg_v_setSupportAddingFragmentBinaryFunctions__bool((const void*)this, nullptr, supportAddingFragmentBinaryFunctions);
}

_MTL_INLINE NS::UInteger MTL::RenderPipelineDescriptor::maxVertexCallStackDepth() const
{
    return _MTL_msg_NS__UInteger_maxVertexCallStackDepth((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setMaxVertexCallStackDepth(NS::UInteger maxVertexCallStackDepth)
{
    _MTL_msg_v_setMaxVertexCallStackDepth__NS__UInteger((const void*)this, nullptr, maxVertexCallStackDepth);
}

_MTL_INLINE NS::UInteger MTL::RenderPipelineDescriptor::maxFragmentCallStackDepth() const
{
    return _MTL_msg_NS__UInteger_maxFragmentCallStackDepth((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setMaxFragmentCallStackDepth(NS::UInteger maxFragmentCallStackDepth)
{
    _MTL_msg_v_setMaxFragmentCallStackDepth__NS__UInteger((const void*)this, nullptr, maxFragmentCallStackDepth);
}

_MTL_INLINE MTL::ShaderValidation MTL::RenderPipelineDescriptor::shaderValidation() const
{
    return _MTL_msg_MTL__ShaderValidation_shaderValidation((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::setShaderValidation(MTL::ShaderValidation shaderValidation)
{
    _MTL_msg_v_setShaderValidation__MTL__ShaderValidation((const void*)this, nullptr, shaderValidation);
}

_MTL_INLINE void MTL::RenderPipelineDescriptor::reset()
{
    _MTL_msg_v_reset((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::RenderPipelineDescriptor::isAlphaToCoverageEnabled()
{
    return _MTL_msg_bool_isAlphaToCoverageEnabled((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::RenderPipelineDescriptor::isAlphaToOneEnabled()
{
    return _MTL_msg_bool_isAlphaToOneEnabled((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::RenderPipelineDescriptor::isRasterizationEnabled()
{
    return _MTL_msg_bool_isRasterizationEnabled((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::RenderPipelineDescriptor::isTessellationFactorScaleEnabled()
{
    return _MTL_msg_bool_isTessellationFactorScaleEnabled((const void*)this, nullptr);
}

_MTL_INLINE MTL::RenderPipelineFunctionsDescriptor* MTL::RenderPipelineFunctionsDescriptor::alloc()
{
    return _MTL_msg_MTL__RenderPipelineFunctionsDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLRenderPipelineFunctionsDescriptor, nullptr);
}

_MTL_INLINE MTL::RenderPipelineFunctionsDescriptor* MTL::RenderPipelineFunctionsDescriptor::init() const
{
    return _MTL_msg_MTL__RenderPipelineFunctionsDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE NS::Array* MTL::RenderPipelineFunctionsDescriptor::vertexAdditionalBinaryFunctions() const
{
    return _MTL_msg_NS__Arrayp_vertexAdditionalBinaryFunctions((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineFunctionsDescriptor::setVertexAdditionalBinaryFunctions(NS::Array* vertexAdditionalBinaryFunctions)
{
    _MTL_msg_v_setVertexAdditionalBinaryFunctions__NS__Arrayp((const void*)this, nullptr, vertexAdditionalBinaryFunctions);
}

_MTL_INLINE NS::Array* MTL::RenderPipelineFunctionsDescriptor::fragmentAdditionalBinaryFunctions() const
{
    return _MTL_msg_NS__Arrayp_fragmentAdditionalBinaryFunctions((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineFunctionsDescriptor::setFragmentAdditionalBinaryFunctions(NS::Array* fragmentAdditionalBinaryFunctions)
{
    _MTL_msg_v_setFragmentAdditionalBinaryFunctions__NS__Arrayp((const void*)this, nullptr, fragmentAdditionalBinaryFunctions);
}

_MTL_INLINE NS::Array* MTL::RenderPipelineFunctionsDescriptor::tileAdditionalBinaryFunctions() const
{
    return _MTL_msg_NS__Arrayp_tileAdditionalBinaryFunctions((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RenderPipelineFunctionsDescriptor::setTileAdditionalBinaryFunctions(NS::Array* tileAdditionalBinaryFunctions)
{
    _MTL_msg_v_setTileAdditionalBinaryFunctions__NS__Arrayp((const void*)this, nullptr, tileAdditionalBinaryFunctions);
}

_MTL_INLINE NS::String* MTL::RenderPipelineState::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE MTL::Device* MTL::RenderPipelineState::device() const
{
    return _MTL_msg_MTL__Devicep_device((const void*)this, nullptr);
}

_MTL_INLINE MTL::RenderPipelineReflection* MTL::RenderPipelineState::reflection() const
{
    return _MTL_msg_MTL__RenderPipelineReflectionp_reflection((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::RenderPipelineState::maxTotalThreadsPerThreadgroup() const
{
    return _MTL_msg_NS__UInteger_maxTotalThreadsPerThreadgroup((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::RenderPipelineState::threadgroupSizeMatchesTileSize() const
{
    return _MTL_msg_bool_threadgroupSizeMatchesTileSize((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::RenderPipelineState::imageblockSampleLength() const
{
    return _MTL_msg_NS__UInteger_imageblockSampleLength((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::RenderPipelineState::supportIndirectCommandBuffers() const
{
    return _MTL_msg_bool_supportIndirectCommandBuffers((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::RenderPipelineState::maxTotalThreadsPerObjectThreadgroup() const
{
    return _MTL_msg_NS__UInteger_maxTotalThreadsPerObjectThreadgroup((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::RenderPipelineState::maxTotalThreadsPerMeshThreadgroup() const
{
    return _MTL_msg_NS__UInteger_maxTotalThreadsPerMeshThreadgroup((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::RenderPipelineState::objectThreadExecutionWidth() const
{
    return _MTL_msg_NS__UInteger_objectThreadExecutionWidth((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::RenderPipelineState::meshThreadExecutionWidth() const
{
    return _MTL_msg_NS__UInteger_meshThreadExecutionWidth((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::RenderPipelineState::maxTotalThreadgroupsPerMeshGrid() const
{
    return _MTL_msg_NS__UInteger_maxTotalThreadgroupsPerMeshGrid((const void*)this, nullptr);
}

_MTL_INLINE MTL::ResourceID MTL::RenderPipelineState::gpuResourceID() const
{
    return _MTL_msg_MTL__ResourceID_gpuResourceID((const void*)this, nullptr);
}

_MTL_INLINE MTL::ShaderValidation MTL::RenderPipelineState::shaderValidation() const
{
    return _MTL_msg_MTL__ShaderValidation_shaderValidation((const void*)this, nullptr);
}

_MTL_INLINE MTL::Size MTL::RenderPipelineState::requiredThreadsPerTileThreadgroup() const
{
    return _MTL_msg_MTL__Size_requiredThreadsPerTileThreadgroup((const void*)this, nullptr);
}

_MTL_INLINE MTL::Size MTL::RenderPipelineState::requiredThreadsPerObjectThreadgroup() const
{
    return _MTL_msg_MTL__Size_requiredThreadsPerObjectThreadgroup((const void*)this, nullptr);
}

_MTL_INLINE MTL::Size MTL::RenderPipelineState::requiredThreadsPerMeshThreadgroup() const
{
    return _MTL_msg_MTL__Size_requiredThreadsPerMeshThreadgroup((const void*)this, nullptr);
}

_MTL_INLINE MTL::FunctionHandle* MTL::RenderPipelineState::functionHandle(NS::String* name, MTL::RenderStages stage)
{
    return _MTL_msg_MTL__FunctionHandlep_functionHandleWithName_stage__NS__Stringp_MTL__RenderStages((const void*)this, nullptr, name, stage);
}

_MTL_INLINE MTL::FunctionHandle* MTL::RenderPipelineState::functionHandle(MTL4::BinaryFunction* function, MTL::RenderStages stage)
{
    return _MTL_msg_MTL__FunctionHandlep_functionHandleWithBinaryFunction_stage__MTL4__BinaryFunctionp_MTL__RenderStages((const void*)this, nullptr, function, stage);
}

_MTL_INLINE MTL::RenderPipelineState* MTL::RenderPipelineState::newRenderPipelineState(MTL4::RenderPipelineBinaryFunctionsDescriptor* binaryFunctionsDescriptor, NS::Error** error)
{
    return _MTL_msg_MTL__RenderPipelineStatep_newRenderPipelineStateWithBinaryFunctions_error__MTL4__RenderPipelineBinaryFunctionsDescriptorp_NS__Errorpp((const void*)this, nullptr, binaryFunctionsDescriptor, error);
}

_MTL_INLINE MTL4::PipelineDescriptor* MTL::RenderPipelineState::newRenderPipelineDescriptorForSpecialization()
{
    return _MTL_msg_MTL4__PipelineDescriptorp_newRenderPipelineDescriptorForSpecialization((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::RenderPipelineState::imageblockMemoryLength(MTL::Size imageblockDimensions)
{
    return _MTL_msg_NS__UInteger_imageblockMemoryLengthForDimensions__MTL__Size((const void*)this, nullptr, imageblockDimensions);
}

_MTL_INLINE MTL::FunctionHandle* MTL::RenderPipelineState::functionHandle(MTL::Function* function, MTL::RenderStages stage)
{
    return _MTL_msg_MTL__FunctionHandlep_functionHandleWithFunction_stage__MTL__Functionp_MTL__RenderStages((const void*)this, nullptr, function, stage);
}

_MTL_INLINE MTL::VisibleFunctionTable* MTL::RenderPipelineState::newVisibleFunctionTable(MTL::VisibleFunctionTableDescriptor* descriptor, MTL::RenderStages stage)
{
    return _MTL_msg_MTL__VisibleFunctionTablep_newVisibleFunctionTableWithDescriptor_stage__MTL__VisibleFunctionTableDescriptorp_MTL__RenderStages((const void*)this, nullptr, descriptor, stage);
}

_MTL_INLINE MTL::IntersectionFunctionTable* MTL::RenderPipelineState::newIntersectionFunctionTable(MTL::IntersectionFunctionTableDescriptor* descriptor, MTL::RenderStages stage)
{
    return _MTL_msg_MTL__IntersectionFunctionTablep_newIntersectionFunctionTableWithDescriptor_stage__MTL__IntersectionFunctionTableDescriptorp_MTL__RenderStages((const void*)this, nullptr, descriptor, stage);
}

_MTL_INLINE MTL::RenderPipelineState* MTL::RenderPipelineState::newRenderPipelineState(MTL::RenderPipelineFunctionsDescriptor* additionalBinaryFunctions, NS::Error** error)
{
    return _MTL_msg_MTL__RenderPipelineStatep_newRenderPipelineStateWithAdditionalBinaryFunctions_error__MTL__RenderPipelineFunctionsDescriptorp_NS__Errorpp((const void*)this, nullptr, additionalBinaryFunctions, error);
}

_MTL_INLINE MTL::RenderPipelineColorAttachmentDescriptorArray* MTL::RenderPipelineColorAttachmentDescriptorArray::alloc()
{
    return _MTL_msg_MTL__RenderPipelineColorAttachmentDescriptorArrayp_alloc((const void*)&OBJC_CLASS_$_MTLRenderPipelineColorAttachmentDescriptorArray, nullptr);
}

_MTL_INLINE MTL::RenderPipelineColorAttachmentDescriptorArray* MTL::RenderPipelineColorAttachmentDescriptorArray::init() const
{
    return _MTL_msg_MTL__RenderPipelineColorAttachmentDescriptorArrayp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::RenderPipelineColorAttachmentDescriptor* MTL::RenderPipelineColorAttachmentDescriptorArray::object(NS::UInteger attachmentIndex)
{
    return _MTL_msg_MTL__RenderPipelineColorAttachmentDescriptorp_objectAtIndexedSubscript__NS__UInteger((const void*)this, nullptr, attachmentIndex);
}

_MTL_INLINE void MTL::RenderPipelineColorAttachmentDescriptorArray::setObject(MTL::RenderPipelineColorAttachmentDescriptor* attachment, NS::UInteger attachmentIndex)
{
    _MTL_msg_v_setObject_atIndexedSubscript__MTL__RenderPipelineColorAttachmentDescriptorp_NS__UInteger((const void*)this, nullptr, attachment, attachmentIndex);
}

_MTL_INLINE MTL::TileRenderPipelineColorAttachmentDescriptor* MTL::TileRenderPipelineColorAttachmentDescriptor::alloc()
{
    return _MTL_msg_MTL__TileRenderPipelineColorAttachmentDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLTileRenderPipelineColorAttachmentDescriptor, nullptr);
}

_MTL_INLINE MTL::TileRenderPipelineColorAttachmentDescriptor* MTL::TileRenderPipelineColorAttachmentDescriptor::init() const
{
    return _MTL_msg_MTL__TileRenderPipelineColorAttachmentDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::PixelFormat MTL::TileRenderPipelineColorAttachmentDescriptor::pixelFormat() const
{
    return _MTL_msg_MTL__PixelFormat_pixelFormat((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TileRenderPipelineColorAttachmentDescriptor::setPixelFormat(MTL::PixelFormat pixelFormat)
{
    _MTL_msg_v_setPixelFormat__MTL__PixelFormat((const void*)this, nullptr, pixelFormat);
}

_MTL_INLINE MTL::TileRenderPipelineColorAttachmentDescriptorArray* MTL::TileRenderPipelineColorAttachmentDescriptorArray::alloc()
{
    return _MTL_msg_MTL__TileRenderPipelineColorAttachmentDescriptorArrayp_alloc((const void*)&OBJC_CLASS_$_MTLTileRenderPipelineColorAttachmentDescriptorArray, nullptr);
}

_MTL_INLINE MTL::TileRenderPipelineColorAttachmentDescriptorArray* MTL::TileRenderPipelineColorAttachmentDescriptorArray::init() const
{
    return _MTL_msg_MTL__TileRenderPipelineColorAttachmentDescriptorArrayp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::TileRenderPipelineColorAttachmentDescriptor* MTL::TileRenderPipelineColorAttachmentDescriptorArray::object(NS::UInteger attachmentIndex)
{
    return _MTL_msg_MTL__TileRenderPipelineColorAttachmentDescriptorp_objectAtIndexedSubscript__NS__UInteger((const void*)this, nullptr, attachmentIndex);
}

_MTL_INLINE void MTL::TileRenderPipelineColorAttachmentDescriptorArray::setObject(MTL::TileRenderPipelineColorAttachmentDescriptor* attachment, NS::UInteger attachmentIndex)
{
    _MTL_msg_v_setObject_atIndexedSubscript__MTL__TileRenderPipelineColorAttachmentDescriptorp_NS__UInteger((const void*)this, nullptr, attachment, attachmentIndex);
}

_MTL_INLINE MTL::TileRenderPipelineDescriptor* MTL::TileRenderPipelineDescriptor::alloc()
{
    return _MTL_msg_MTL__TileRenderPipelineDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLTileRenderPipelineDescriptor, nullptr);
}

_MTL_INLINE MTL::TileRenderPipelineDescriptor* MTL::TileRenderPipelineDescriptor::init() const
{
    return _MTL_msg_MTL__TileRenderPipelineDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE NS::String* MTL::TileRenderPipelineDescriptor::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TileRenderPipelineDescriptor::setLabel(NS::String* label)
{
    _MTL_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL_INLINE MTL::Function* MTL::TileRenderPipelineDescriptor::tileFunction() const
{
    return _MTL_msg_MTL__Functionp_tileFunction((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TileRenderPipelineDescriptor::setTileFunction(MTL::Function* tileFunction)
{
    _MTL_msg_v_setTileFunction__MTL__Functionp((const void*)this, nullptr, tileFunction);
}

_MTL_INLINE NS::UInteger MTL::TileRenderPipelineDescriptor::rasterSampleCount() const
{
    return _MTL_msg_NS__UInteger_rasterSampleCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TileRenderPipelineDescriptor::setRasterSampleCount(NS::UInteger rasterSampleCount)
{
    _MTL_msg_v_setRasterSampleCount__NS__UInteger((const void*)this, nullptr, rasterSampleCount);
}

_MTL_INLINE MTL::TileRenderPipelineColorAttachmentDescriptorArray* MTL::TileRenderPipelineDescriptor::colorAttachments() const
{
    return _MTL_msg_MTL__TileRenderPipelineColorAttachmentDescriptorArrayp_colorAttachments((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::TileRenderPipelineDescriptor::threadgroupSizeMatchesTileSize() const
{
    return _MTL_msg_bool_threadgroupSizeMatchesTileSize((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TileRenderPipelineDescriptor::setThreadgroupSizeMatchesTileSize(bool threadgroupSizeMatchesTileSize)
{
    _MTL_msg_v_setThreadgroupSizeMatchesTileSize__bool((const void*)this, nullptr, threadgroupSizeMatchesTileSize);
}

_MTL_INLINE MTL::PipelineBufferDescriptorArray* MTL::TileRenderPipelineDescriptor::tileBuffers() const
{
    return _MTL_msg_MTL__PipelineBufferDescriptorArrayp_tileBuffers((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::TileRenderPipelineDescriptor::maxTotalThreadsPerThreadgroup() const
{
    return _MTL_msg_NS__UInteger_maxTotalThreadsPerThreadgroup((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TileRenderPipelineDescriptor::setMaxTotalThreadsPerThreadgroup(NS::UInteger maxTotalThreadsPerThreadgroup)
{
    _MTL_msg_v_setMaxTotalThreadsPerThreadgroup__NS__UInteger((const void*)this, nullptr, maxTotalThreadsPerThreadgroup);
}

_MTL_INLINE NS::Array* MTL::TileRenderPipelineDescriptor::binaryArchives() const
{
    return _MTL_msg_NS__Arrayp_binaryArchives((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TileRenderPipelineDescriptor::setBinaryArchives(NS::Array* binaryArchives)
{
    _MTL_msg_v_setBinaryArchives__NS__Arrayp((const void*)this, nullptr, binaryArchives);
}

_MTL_INLINE NS::Array* MTL::TileRenderPipelineDescriptor::preloadedLibraries() const
{
    return _MTL_msg_NS__Arrayp_preloadedLibraries((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TileRenderPipelineDescriptor::setPreloadedLibraries(NS::Array* preloadedLibraries)
{
    _MTL_msg_v_setPreloadedLibraries__NS__Arrayp((const void*)this, nullptr, preloadedLibraries);
}

_MTL_INLINE MTL::LinkedFunctions* MTL::TileRenderPipelineDescriptor::linkedFunctions() const
{
    return _MTL_msg_MTL__LinkedFunctionsp_linkedFunctions((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TileRenderPipelineDescriptor::setLinkedFunctions(MTL::LinkedFunctions* linkedFunctions)
{
    _MTL_msg_v_setLinkedFunctions__MTL__LinkedFunctionsp((const void*)this, nullptr, linkedFunctions);
}

_MTL_INLINE bool MTL::TileRenderPipelineDescriptor::supportAddingBinaryFunctions() const
{
    return _MTL_msg_bool_supportAddingBinaryFunctions((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TileRenderPipelineDescriptor::setSupportAddingBinaryFunctions(bool supportAddingBinaryFunctions)
{
    _MTL_msg_v_setSupportAddingBinaryFunctions__bool((const void*)this, nullptr, supportAddingBinaryFunctions);
}

_MTL_INLINE NS::UInteger MTL::TileRenderPipelineDescriptor::maxCallStackDepth() const
{
    return _MTL_msg_NS__UInteger_maxCallStackDepth((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TileRenderPipelineDescriptor::setMaxCallStackDepth(NS::UInteger maxCallStackDepth)
{
    _MTL_msg_v_setMaxCallStackDepth__NS__UInteger((const void*)this, nullptr, maxCallStackDepth);
}

_MTL_INLINE MTL::ShaderValidation MTL::TileRenderPipelineDescriptor::shaderValidation() const
{
    return _MTL_msg_MTL__ShaderValidation_shaderValidation((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TileRenderPipelineDescriptor::setShaderValidation(MTL::ShaderValidation shaderValidation)
{
    _MTL_msg_v_setShaderValidation__MTL__ShaderValidation((const void*)this, nullptr, shaderValidation);
}

_MTL_INLINE MTL::Size MTL::TileRenderPipelineDescriptor::requiredThreadsPerThreadgroup() const
{
    return _MTL_msg_MTL__Size_requiredThreadsPerThreadgroup((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TileRenderPipelineDescriptor::setRequiredThreadsPerThreadgroup(MTL::Size requiredThreadsPerThreadgroup)
{
    _MTL_msg_v_setRequiredThreadsPerThreadgroup__MTL__Size((const void*)this, nullptr, requiredThreadsPerThreadgroup);
}

_MTL_INLINE void MTL::TileRenderPipelineDescriptor::reset()
{
    _MTL_msg_v_reset((const void*)this, nullptr);
}

_MTL_INLINE MTL::MeshRenderPipelineDescriptor* MTL::MeshRenderPipelineDescriptor::alloc()
{
    return _MTL_msg_MTL__MeshRenderPipelineDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLMeshRenderPipelineDescriptor, nullptr);
}

_MTL_INLINE MTL::MeshRenderPipelineDescriptor* MTL::MeshRenderPipelineDescriptor::init() const
{
    return _MTL_msg_MTL__MeshRenderPipelineDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE NS::String* MTL::MeshRenderPipelineDescriptor::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setLabel(NS::String* label)
{
    _MTL_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL_INLINE MTL::Function* MTL::MeshRenderPipelineDescriptor::objectFunction() const
{
    return _MTL_msg_MTL__Functionp_objectFunction((const void*)this, nullptr);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setObjectFunction(MTL::Function* objectFunction)
{
    _MTL_msg_v_setObjectFunction__MTL__Functionp((const void*)this, nullptr, objectFunction);
}

_MTL_INLINE MTL::Function* MTL::MeshRenderPipelineDescriptor::meshFunction() const
{
    return _MTL_msg_MTL__Functionp_meshFunction((const void*)this, nullptr);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setMeshFunction(MTL::Function* meshFunction)
{
    _MTL_msg_v_setMeshFunction__MTL__Functionp((const void*)this, nullptr, meshFunction);
}

_MTL_INLINE MTL::Function* MTL::MeshRenderPipelineDescriptor::fragmentFunction() const
{
    return _MTL_msg_MTL__Functionp_fragmentFunction((const void*)this, nullptr);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setFragmentFunction(MTL::Function* fragmentFunction)
{
    _MTL_msg_v_setFragmentFunction__MTL__Functionp((const void*)this, nullptr, fragmentFunction);
}

_MTL_INLINE NS::UInteger MTL::MeshRenderPipelineDescriptor::maxTotalThreadsPerObjectThreadgroup() const
{
    return _MTL_msg_NS__UInteger_maxTotalThreadsPerObjectThreadgroup((const void*)this, nullptr);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setMaxTotalThreadsPerObjectThreadgroup(NS::UInteger maxTotalThreadsPerObjectThreadgroup)
{
    _MTL_msg_v_setMaxTotalThreadsPerObjectThreadgroup__NS__UInteger((const void*)this, nullptr, maxTotalThreadsPerObjectThreadgroup);
}

_MTL_INLINE NS::UInteger MTL::MeshRenderPipelineDescriptor::maxTotalThreadsPerMeshThreadgroup() const
{
    return _MTL_msg_NS__UInteger_maxTotalThreadsPerMeshThreadgroup((const void*)this, nullptr);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setMaxTotalThreadsPerMeshThreadgroup(NS::UInteger maxTotalThreadsPerMeshThreadgroup)
{
    _MTL_msg_v_setMaxTotalThreadsPerMeshThreadgroup__NS__UInteger((const void*)this, nullptr, maxTotalThreadsPerMeshThreadgroup);
}

_MTL_INLINE bool MTL::MeshRenderPipelineDescriptor::objectThreadgroupSizeIsMultipleOfThreadExecutionWidth() const
{
    return _MTL_msg_bool_objectThreadgroupSizeIsMultipleOfThreadExecutionWidth((const void*)this, nullptr);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setObjectThreadgroupSizeIsMultipleOfThreadExecutionWidth(bool objectThreadgroupSizeIsMultipleOfThreadExecutionWidth)
{
    _MTL_msg_v_setObjectThreadgroupSizeIsMultipleOfThreadExecutionWidth__bool((const void*)this, nullptr, objectThreadgroupSizeIsMultipleOfThreadExecutionWidth);
}

_MTL_INLINE bool MTL::MeshRenderPipelineDescriptor::meshThreadgroupSizeIsMultipleOfThreadExecutionWidth() const
{
    return _MTL_msg_bool_meshThreadgroupSizeIsMultipleOfThreadExecutionWidth((const void*)this, nullptr);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setMeshThreadgroupSizeIsMultipleOfThreadExecutionWidth(bool meshThreadgroupSizeIsMultipleOfThreadExecutionWidth)
{
    _MTL_msg_v_setMeshThreadgroupSizeIsMultipleOfThreadExecutionWidth__bool((const void*)this, nullptr, meshThreadgroupSizeIsMultipleOfThreadExecutionWidth);
}

_MTL_INLINE NS::UInteger MTL::MeshRenderPipelineDescriptor::payloadMemoryLength() const
{
    return _MTL_msg_NS__UInteger_payloadMemoryLength((const void*)this, nullptr);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setPayloadMemoryLength(NS::UInteger payloadMemoryLength)
{
    _MTL_msg_v_setPayloadMemoryLength__NS__UInteger((const void*)this, nullptr, payloadMemoryLength);
}

_MTL_INLINE NS::UInteger MTL::MeshRenderPipelineDescriptor::maxTotalThreadgroupsPerMeshGrid() const
{
    return _MTL_msg_NS__UInteger_maxTotalThreadgroupsPerMeshGrid((const void*)this, nullptr);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setMaxTotalThreadgroupsPerMeshGrid(NS::UInteger maxTotalThreadgroupsPerMeshGrid)
{
    _MTL_msg_v_setMaxTotalThreadgroupsPerMeshGrid__NS__UInteger((const void*)this, nullptr, maxTotalThreadgroupsPerMeshGrid);
}

_MTL_INLINE MTL::PipelineBufferDescriptorArray* MTL::MeshRenderPipelineDescriptor::objectBuffers() const
{
    return _MTL_msg_MTL__PipelineBufferDescriptorArrayp_objectBuffers((const void*)this, nullptr);
}

_MTL_INLINE MTL::PipelineBufferDescriptorArray* MTL::MeshRenderPipelineDescriptor::meshBuffers() const
{
    return _MTL_msg_MTL__PipelineBufferDescriptorArrayp_meshBuffers((const void*)this, nullptr);
}

_MTL_INLINE MTL::PipelineBufferDescriptorArray* MTL::MeshRenderPipelineDescriptor::fragmentBuffers() const
{
    return _MTL_msg_MTL__PipelineBufferDescriptorArrayp_fragmentBuffers((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::MeshRenderPipelineDescriptor::rasterSampleCount() const
{
    return _MTL_msg_NS__UInteger_rasterSampleCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setRasterSampleCount(NS::UInteger rasterSampleCount)
{
    _MTL_msg_v_setRasterSampleCount__NS__UInteger((const void*)this, nullptr, rasterSampleCount);
}

_MTL_INLINE bool MTL::MeshRenderPipelineDescriptor::alphaToCoverageEnabled() const
{
    return _MTL_msg_bool_alphaToCoverageEnabled((const void*)this, nullptr);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setAlphaToCoverageEnabled(bool alphaToCoverageEnabled)
{
    _MTL_msg_v_setAlphaToCoverageEnabled__bool((const void*)this, nullptr, alphaToCoverageEnabled);
}

_MTL_INLINE bool MTL::MeshRenderPipelineDescriptor::alphaToOneEnabled() const
{
    return _MTL_msg_bool_alphaToOneEnabled((const void*)this, nullptr);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setAlphaToOneEnabled(bool alphaToOneEnabled)
{
    _MTL_msg_v_setAlphaToOneEnabled__bool((const void*)this, nullptr, alphaToOneEnabled);
}

_MTL_INLINE bool MTL::MeshRenderPipelineDescriptor::rasterizationEnabled() const
{
    return _MTL_msg_bool_rasterizationEnabled((const void*)this, nullptr);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setRasterizationEnabled(bool rasterizationEnabled)
{
    _MTL_msg_v_setRasterizationEnabled__bool((const void*)this, nullptr, rasterizationEnabled);
}

_MTL_INLINE NS::UInteger MTL::MeshRenderPipelineDescriptor::maxVertexAmplificationCount() const
{
    return _MTL_msg_NS__UInteger_maxVertexAmplificationCount((const void*)this, nullptr);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setMaxVertexAmplificationCount(NS::UInteger maxVertexAmplificationCount)
{
    _MTL_msg_v_setMaxVertexAmplificationCount__NS__UInteger((const void*)this, nullptr, maxVertexAmplificationCount);
}

_MTL_INLINE MTL::RenderPipelineColorAttachmentDescriptorArray* MTL::MeshRenderPipelineDescriptor::colorAttachments() const
{
    return _MTL_msg_MTL__RenderPipelineColorAttachmentDescriptorArrayp_colorAttachments((const void*)this, nullptr);
}

_MTL_INLINE MTL::PixelFormat MTL::MeshRenderPipelineDescriptor::depthAttachmentPixelFormat() const
{
    return _MTL_msg_MTL__PixelFormat_depthAttachmentPixelFormat((const void*)this, nullptr);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setDepthAttachmentPixelFormat(MTL::PixelFormat depthAttachmentPixelFormat)
{
    _MTL_msg_v_setDepthAttachmentPixelFormat__MTL__PixelFormat((const void*)this, nullptr, depthAttachmentPixelFormat);
}

_MTL_INLINE MTL::PixelFormat MTL::MeshRenderPipelineDescriptor::stencilAttachmentPixelFormat() const
{
    return _MTL_msg_MTL__PixelFormat_stencilAttachmentPixelFormat((const void*)this, nullptr);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setStencilAttachmentPixelFormat(MTL::PixelFormat stencilAttachmentPixelFormat)
{
    _MTL_msg_v_setStencilAttachmentPixelFormat__MTL__PixelFormat((const void*)this, nullptr, stencilAttachmentPixelFormat);
}

_MTL_INLINE bool MTL::MeshRenderPipelineDescriptor::supportIndirectCommandBuffers() const
{
    return _MTL_msg_bool_supportIndirectCommandBuffers((const void*)this, nullptr);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setSupportIndirectCommandBuffers(bool supportIndirectCommandBuffers)
{
    _MTL_msg_v_setSupportIndirectCommandBuffers__bool((const void*)this, nullptr, supportIndirectCommandBuffers);
}

_MTL_INLINE NS::Array* MTL::MeshRenderPipelineDescriptor::binaryArchives() const
{
    return _MTL_msg_NS__Arrayp_binaryArchives((const void*)this, nullptr);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setBinaryArchives(NS::Array* binaryArchives)
{
    _MTL_msg_v_setBinaryArchives__NS__Arrayp((const void*)this, nullptr, binaryArchives);
}

_MTL_INLINE MTL::LinkedFunctions* MTL::MeshRenderPipelineDescriptor::objectLinkedFunctions() const
{
    return _MTL_msg_MTL__LinkedFunctionsp_objectLinkedFunctions((const void*)this, nullptr);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setObjectLinkedFunctions(MTL::LinkedFunctions* objectLinkedFunctions)
{
    _MTL_msg_v_setObjectLinkedFunctions__MTL__LinkedFunctionsp((const void*)this, nullptr, objectLinkedFunctions);
}

_MTL_INLINE MTL::LinkedFunctions* MTL::MeshRenderPipelineDescriptor::meshLinkedFunctions() const
{
    return _MTL_msg_MTL__LinkedFunctionsp_meshLinkedFunctions((const void*)this, nullptr);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setMeshLinkedFunctions(MTL::LinkedFunctions* meshLinkedFunctions)
{
    _MTL_msg_v_setMeshLinkedFunctions__MTL__LinkedFunctionsp((const void*)this, nullptr, meshLinkedFunctions);
}

_MTL_INLINE MTL::LinkedFunctions* MTL::MeshRenderPipelineDescriptor::fragmentLinkedFunctions() const
{
    return _MTL_msg_MTL__LinkedFunctionsp_fragmentLinkedFunctions((const void*)this, nullptr);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setFragmentLinkedFunctions(MTL::LinkedFunctions* fragmentLinkedFunctions)
{
    _MTL_msg_v_setFragmentLinkedFunctions__MTL__LinkedFunctionsp((const void*)this, nullptr, fragmentLinkedFunctions);
}

_MTL_INLINE MTL::ShaderValidation MTL::MeshRenderPipelineDescriptor::shaderValidation() const
{
    return _MTL_msg_MTL__ShaderValidation_shaderValidation((const void*)this, nullptr);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setShaderValidation(MTL::ShaderValidation shaderValidation)
{
    _MTL_msg_v_setShaderValidation__MTL__ShaderValidation((const void*)this, nullptr, shaderValidation);
}

_MTL_INLINE MTL::Size MTL::MeshRenderPipelineDescriptor::requiredThreadsPerObjectThreadgroup() const
{
    return _MTL_msg_MTL__Size_requiredThreadsPerObjectThreadgroup((const void*)this, nullptr);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setRequiredThreadsPerObjectThreadgroup(MTL::Size requiredThreadsPerObjectThreadgroup)
{
    _MTL_msg_v_setRequiredThreadsPerObjectThreadgroup__MTL__Size((const void*)this, nullptr, requiredThreadsPerObjectThreadgroup);
}

_MTL_INLINE MTL::Size MTL::MeshRenderPipelineDescriptor::requiredThreadsPerMeshThreadgroup() const
{
    return _MTL_msg_MTL__Size_requiredThreadsPerMeshThreadgroup((const void*)this, nullptr);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::setRequiredThreadsPerMeshThreadgroup(MTL::Size requiredThreadsPerMeshThreadgroup)
{
    _MTL_msg_v_setRequiredThreadsPerMeshThreadgroup__MTL__Size((const void*)this, nullptr, requiredThreadsPerMeshThreadgroup);
}

_MTL_INLINE void MTL::MeshRenderPipelineDescriptor::reset()
{
    _MTL_msg_v_reset((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::MeshRenderPipelineDescriptor::isAlphaToCoverageEnabled()
{
    return _MTL_msg_bool_isAlphaToCoverageEnabled((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::MeshRenderPipelineDescriptor::isAlphaToOneEnabled()
{
    return _MTL_msg_bool_isAlphaToOneEnabled((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::MeshRenderPipelineDescriptor::isRasterizationEnabled()
{
    return _MTL_msg_bool_isRasterizationEnabled((const void*)this, nullptr);
}
