#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"
#include "MTLResource.hpp"

namespace MTL {
    class Buffer;
    enum CPUCacheMode : NS::UInteger;
    enum HazardTrackingMode : NS::UInteger;
    using ResourceOptions = NS::UInteger;
    enum StorageMode : NS::UInteger;
}

namespace MTL
{

extern NS::ErrorDomain const TensorDomain __asm__("_MTLTensorDomain");
_MTL_ENUM(NS::Integer, TensorDataType) {
    TensorDataTypeNone = DataTypeNone,
    TensorDataTypeFloat32 = DataTypeFloat,
    TensorDataTypeFloat16 = DataTypeHalf,
    TensorDataTypeBFloat16 = DataTypeBFloat,
    TensorDataTypeInt8 = DataTypeChar,
    TensorDataTypeUInt8 = DataTypeUChar,
    TensorDataTypeInt16 = DataTypeShort,
    TensorDataTypeUInt16 = DataTypeUShort,
    TensorDataTypeInt32 = DataTypeInt,
    TensorDataTypeUInt32 = DataTypeUInt,
    TensorDataTypeInt4 = 143,
    TensorDataTypeUInt4 = 144,
};

_MTL_ENUM(NS::Integer, TensorError) {
    TensorErrorNone = 0,
    TensorErrorInternalError = 1,
    TensorErrorInvalidDescriptor = 2,
};

_MTL_OPTIONS(NS::UInteger, TensorUsage) {
    TensorUsageCompute = 1 << 0,
    TensorUsageRender = 1 << 1,
    TensorUsageMachineLearning = 1 << 2,
};


class TensorExtents;
class TensorDescriptor;
class Tensor;

class TensorExtents : public NS::Referencing<TensorExtents>
{
public:
    static TensorExtents* alloc();
    TensorExtents*        init() const;

    NS::Integer         extent(NS::UInteger dimensionIndex);
    MTL::TensorExtents* init(NS::UInteger rank, const NS::Integer * values);
    NS::UInteger        rank() const;

};

class TensorDescriptor : public NS::Copying<TensorDescriptor>
{
public:
    static TensorDescriptor* alloc();
    TensorDescriptor*        init() const;

    MTL::CPUCacheMode       cpuCacheMode() const;
    MTL::TensorDataType     dataType() const;
    MTL::TensorExtents*     dimensions() const;
    MTL::HazardTrackingMode hazardTrackingMode() const;
    MTL::ResourceOptions    resourceOptions() const;
    void                    setCpuCacheMode(MTL::CPUCacheMode cpuCacheMode);
    void                    setDataType(MTL::TensorDataType dataType);
    void                    setDimensions(MTL::TensorExtents* dimensions);
    void                    setHazardTrackingMode(MTL::HazardTrackingMode hazardTrackingMode);
    void                    setResourceOptions(MTL::ResourceOptions resourceOptions);
    void                    setStorageMode(MTL::StorageMode storageMode);
    void                    setStrides(MTL::TensorExtents* strides);
    void                    setUsage(MTL::TensorUsage usage);
    MTL::StorageMode        storageMode() const;
    MTL::TensorExtents*     strides() const;
    MTL::TensorUsage        usage() const;

};

class Tensor : public NS::Referencing<Tensor, MTL::Resource>
{
public:
    MTL::Buffer*        buffer() const;
    NS::UInteger        bufferOffset() const;
    MTL::TensorDataType dataType() const;
    MTL::TensorExtents* dimensions() const;
    void                getBytes(void * bytes, MTL::TensorExtents* strides, MTL::TensorExtents* sliceOrigin, MTL::TensorExtents* sliceDimensions);
    MTL::ResourceID     gpuResourceID() const;
    void                replace(MTL::TensorExtents* sliceOrigin, MTL::TensorExtents* sliceDimensions, const void * bytes, MTL::TensorExtents* strides);
    MTL::TensorExtents* strides() const;
    MTL::TensorUsage    usage() const;

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLTensorExtents;
extern "C" void *OBJC_CLASS_$_MTLTensorDescriptor;
extern "C" void *OBJC_CLASS_$_MTLTensor;

_MTL_INLINE MTL::TensorExtents* MTL::TensorExtents::alloc()
{
    return _MTL_msg_MTL__TensorExtentsp_alloc((const void*)&OBJC_CLASS_$_MTLTensorExtents, nullptr);
}

_MTL_INLINE MTL::TensorExtents* MTL::TensorExtents::init() const
{
    return _MTL_msg_MTL__TensorExtentsp_init((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::TensorExtents::rank() const
{
    return _MTL_msg_NS__UInteger_rank((const void*)this, nullptr);
}

_MTL_INLINE MTL::TensorExtents* MTL::TensorExtents::init(NS::UInteger rank, const NS::Integer * values)
{
    return _MTL_msg_MTL__TensorExtentsp_initWithRank_values__NS__UInteger_constNS__Integerp((const void*)this, nullptr, rank, values);
}

_MTL_INLINE NS::Integer MTL::TensorExtents::extent(NS::UInteger dimensionIndex)
{
    return _MTL_msg_NS__Integer_extentAtDimensionIndex__NS__UInteger((const void*)this, nullptr, dimensionIndex);
}

_MTL_INLINE MTL::TensorDescriptor* MTL::TensorDescriptor::alloc()
{
    return _MTL_msg_MTL__TensorDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLTensorDescriptor, nullptr);
}

_MTL_INLINE MTL::TensorDescriptor* MTL::TensorDescriptor::init() const
{
    return _MTL_msg_MTL__TensorDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::TensorExtents* MTL::TensorDescriptor::dimensions() const
{
    return _MTL_msg_MTL__TensorExtentsp_dimensions((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TensorDescriptor::setDimensions(MTL::TensorExtents* dimensions)
{
    _MTL_msg_v_setDimensions__MTL__TensorExtentsp((const void*)this, nullptr, dimensions);
}

_MTL_INLINE MTL::TensorExtents* MTL::TensorDescriptor::strides() const
{
    return _MTL_msg_MTL__TensorExtentsp_strides((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TensorDescriptor::setStrides(MTL::TensorExtents* strides)
{
    _MTL_msg_v_setStrides__MTL__TensorExtentsp((const void*)this, nullptr, strides);
}

_MTL_INLINE MTL::TensorDataType MTL::TensorDescriptor::dataType() const
{
    return _MTL_msg_MTL__TensorDataType_dataType((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TensorDescriptor::setDataType(MTL::TensorDataType dataType)
{
    _MTL_msg_v_setDataType__MTL__TensorDataType((const void*)this, nullptr, dataType);
}

_MTL_INLINE MTL::TensorUsage MTL::TensorDescriptor::usage() const
{
    return _MTL_msg_MTL__TensorUsage_usage((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TensorDescriptor::setUsage(MTL::TensorUsage usage)
{
    _MTL_msg_v_setUsage__MTL__TensorUsage((const void*)this, nullptr, usage);
}

_MTL_INLINE MTL::ResourceOptions MTL::TensorDescriptor::resourceOptions() const
{
    return _MTL_msg_MTL__ResourceOptions_resourceOptions((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TensorDescriptor::setResourceOptions(MTL::ResourceOptions resourceOptions)
{
    _MTL_msg_v_setResourceOptions__MTL__ResourceOptions((const void*)this, nullptr, resourceOptions);
}

_MTL_INLINE MTL::CPUCacheMode MTL::TensorDescriptor::cpuCacheMode() const
{
    return _MTL_msg_MTL__CPUCacheMode_cpuCacheMode((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TensorDescriptor::setCpuCacheMode(MTL::CPUCacheMode cpuCacheMode)
{
    _MTL_msg_v_setCpuCacheMode__MTL__CPUCacheMode((const void*)this, nullptr, cpuCacheMode);
}

_MTL_INLINE MTL::StorageMode MTL::TensorDescriptor::storageMode() const
{
    return _MTL_msg_MTL__StorageMode_storageMode((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TensorDescriptor::setStorageMode(MTL::StorageMode storageMode)
{
    _MTL_msg_v_setStorageMode__MTL__StorageMode((const void*)this, nullptr, storageMode);
}

_MTL_INLINE MTL::HazardTrackingMode MTL::TensorDescriptor::hazardTrackingMode() const
{
    return _MTL_msg_MTL__HazardTrackingMode_hazardTrackingMode((const void*)this, nullptr);
}

_MTL_INLINE void MTL::TensorDescriptor::setHazardTrackingMode(MTL::HazardTrackingMode hazardTrackingMode)
{
    _MTL_msg_v_setHazardTrackingMode__MTL__HazardTrackingMode((const void*)this, nullptr, hazardTrackingMode);
}

_MTL_INLINE MTL::ResourceID MTL::Tensor::gpuResourceID() const
{
    return _MTL_msg_MTL__ResourceID_gpuResourceID((const void*)this, nullptr);
}

_MTL_INLINE MTL::Buffer* MTL::Tensor::buffer() const
{
    return _MTL_msg_MTL__Bufferp_buffer((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::Tensor::bufferOffset() const
{
    return _MTL_msg_NS__UInteger_bufferOffset((const void*)this, nullptr);
}

_MTL_INLINE MTL::TensorExtents* MTL::Tensor::strides() const
{
    return _MTL_msg_MTL__TensorExtentsp_strides((const void*)this, nullptr);
}

_MTL_INLINE MTL::TensorExtents* MTL::Tensor::dimensions() const
{
    return _MTL_msg_MTL__TensorExtentsp_dimensions((const void*)this, nullptr);
}

_MTL_INLINE MTL::TensorDataType MTL::Tensor::dataType() const
{
    return _MTL_msg_MTL__TensorDataType_dataType((const void*)this, nullptr);
}

_MTL_INLINE MTL::TensorUsage MTL::Tensor::usage() const
{
    return _MTL_msg_MTL__TensorUsage_usage((const void*)this, nullptr);
}

_MTL_INLINE void MTL::Tensor::replace(MTL::TensorExtents* sliceOrigin, MTL::TensorExtents* sliceDimensions, const void * bytes, MTL::TensorExtents* strides)
{
    _MTL_msg_v_replaceSliceOrigin_sliceDimensions_withBytes_strides__MTL__TensorExtentsp_MTL__TensorExtentsp_constvoidp_MTL__TensorExtentsp((const void*)this, nullptr, sliceOrigin, sliceDimensions, bytes, strides);
}

_MTL_INLINE void MTL::Tensor::getBytes(void * bytes, MTL::TensorExtents* strides, MTL::TensorExtents* sliceOrigin, MTL::TensorExtents* sliceDimensions)
{
    _MTL_msg_v_getBytes_strides_fromSliceOrigin_sliceDimensions__voidp_MTL__TensorExtentsp_MTL__TensorExtentsp_MTL__TensorExtentsp((const void*)this, nullptr, bytes, strides, sliceOrigin, sliceDimensions);
}
