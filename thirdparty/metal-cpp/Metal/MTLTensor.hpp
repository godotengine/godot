//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLTensor.hpp
//
// Copyright 2020-2025 Apple Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#pragma once

#include "../Foundation/Foundation.hpp"
#include "MTLDefines.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"
#include "MTLResource.hpp"
#include "MTLTypes.hpp"

namespace MTL
{
class Buffer;
class TensorDescriptor;
class TensorExtents;

_MTL_CONST(NS::ErrorDomain, TensorDomain);

_MTL_ENUM(NS::Integer, TensorDataType) {
    TensorDataTypeNone = 0,
    TensorDataTypeFloat32 = 3,
    TensorDataTypeFloat16 = 16,
    TensorDataTypeBFloat16 = 121,
    TensorDataTypeInt8 = 45,
    TensorDataTypeUInt8 = 49,
    TensorDataTypeInt16 = 37,
    TensorDataTypeUInt16 = 41,
    TensorDataTypeInt32 = 29,
    TensorDataTypeUInt32 = 33,
};

_MTL_ENUM(NS::Integer, TensorError) {
    TensorErrorNone = 0,
    TensorErrorInternalError = 1,
    TensorErrorInvalidDescriptor = 2,
};

_MTL_OPTIONS(NS::UInteger, TensorUsage) {
    TensorUsageCompute = 1,
    TensorUsageRender = 1 << 1,
    TensorUsageMachineLearning = 1 << 2,
};

class TensorExtents : public NS::Referencing<TensorExtents>
{
public:
    static TensorExtents* alloc();

    NS::Integer           extentAtDimensionIndex(NS::UInteger dimensionIndex);

    TensorExtents*        init();
    TensorExtents*        init(NS::UInteger rank, const NS::Integer* values);

    NS::UInteger          rank() const;
};
class TensorDescriptor : public NS::Copying<TensorDescriptor>
{
public:
    static TensorDescriptor* alloc();

    CPUCacheMode             cpuCacheMode() const;

    TensorDataType           dataType() const;

    TensorExtents*           dimensions() const;

    HazardTrackingMode       hazardTrackingMode() const;

    TensorDescriptor*        init();

    ResourceOptions          resourceOptions() const;

    void                     setCpuCacheMode(MTL::CPUCacheMode cpuCacheMode);

    void                     setDataType(MTL::TensorDataType dataType);

    void                     setDimensions(const MTL::TensorExtents* dimensions);

    void                     setHazardTrackingMode(MTL::HazardTrackingMode hazardTrackingMode);

    void                     setResourceOptions(MTL::ResourceOptions resourceOptions);

    void                     setStorageMode(MTL::StorageMode storageMode);

    void                     setStrides(const MTL::TensorExtents* strides);

    void                     setUsage(MTL::TensorUsage usage);

    StorageMode              storageMode() const;

    TensorExtents*           strides() const;

    TensorUsage              usage() const;
};
class Tensor : public NS::Referencing<Tensor, Resource>
{
public:
    Buffer*        buffer() const;
    NS::UInteger   bufferOffset() const;

    TensorDataType dataType() const;

    TensorExtents* dimensions() const;

    void           getBytes(void* bytes, const MTL::TensorExtents* strides, const MTL::TensorExtents* sliceOrigin, const MTL::TensorExtents* sliceDimensions);

    ResourceID     gpuResourceID() const;

    void           replaceSliceOrigin(const MTL::TensorExtents* sliceOrigin, const MTL::TensorExtents* sliceDimensions, const void* bytes, const MTL::TensorExtents* strides);

    TensorExtents* strides() const;

    TensorUsage    usage() const;
};

}

_MTL_PRIVATE_DEF_CONST(NS::ErrorDomain, TensorDomain);

_MTL_INLINE MTL::TensorExtents* MTL::TensorExtents::alloc()
{
    return NS::Object::alloc<MTL::TensorExtents>(_MTL_PRIVATE_CLS(MTLTensorExtents));
}

_MTL_INLINE NS::Integer MTL::TensorExtents::extentAtDimensionIndex(NS::UInteger dimensionIndex)
{
    return Object::sendMessage<NS::Integer>(this, _MTL_PRIVATE_SEL(extentAtDimensionIndex_), dimensionIndex);
}

_MTL_INLINE MTL::TensorExtents* MTL::TensorExtents::init()
{
    return NS::Object::init<MTL::TensorExtents>();
}

_MTL_INLINE MTL::TensorExtents* MTL::TensorExtents::init(NS::UInteger rank, const NS::Integer* values)
{
    return Object::sendMessage<MTL::TensorExtents*>(this, _MTL_PRIVATE_SEL(initWithRank_values_), rank, values);
}

_MTL_INLINE NS::UInteger MTL::TensorExtents::rank() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(rank));
}

_MTL_INLINE MTL::TensorDescriptor* MTL::TensorDescriptor::alloc()
{
    return NS::Object::alloc<MTL::TensorDescriptor>(_MTL_PRIVATE_CLS(MTLTensorDescriptor));
}

_MTL_INLINE MTL::CPUCacheMode MTL::TensorDescriptor::cpuCacheMode() const
{
    return Object::sendMessage<MTL::CPUCacheMode>(this, _MTL_PRIVATE_SEL(cpuCacheMode));
}

_MTL_INLINE MTL::TensorDataType MTL::TensorDescriptor::dataType() const
{
    return Object::sendMessage<MTL::TensorDataType>(this, _MTL_PRIVATE_SEL(dataType));
}

_MTL_INLINE MTL::TensorExtents* MTL::TensorDescriptor::dimensions() const
{
    return Object::sendMessage<MTL::TensorExtents*>(this, _MTL_PRIVATE_SEL(dimensions));
}

_MTL_INLINE MTL::HazardTrackingMode MTL::TensorDescriptor::hazardTrackingMode() const
{
    return Object::sendMessage<MTL::HazardTrackingMode>(this, _MTL_PRIVATE_SEL(hazardTrackingMode));
}

_MTL_INLINE MTL::TensorDescriptor* MTL::TensorDescriptor::init()
{
    return NS::Object::init<MTL::TensorDescriptor>();
}

_MTL_INLINE MTL::ResourceOptions MTL::TensorDescriptor::resourceOptions() const
{
    return Object::sendMessage<MTL::ResourceOptions>(this, _MTL_PRIVATE_SEL(resourceOptions));
}

_MTL_INLINE void MTL::TensorDescriptor::setCpuCacheMode(MTL::CPUCacheMode cpuCacheMode)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setCpuCacheMode_), cpuCacheMode);
}

_MTL_INLINE void MTL::TensorDescriptor::setDataType(MTL::TensorDataType dataType)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDataType_), dataType);
}

_MTL_INLINE void MTL::TensorDescriptor::setDimensions(const MTL::TensorExtents* dimensions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDimensions_), dimensions);
}

_MTL_INLINE void MTL::TensorDescriptor::setHazardTrackingMode(MTL::HazardTrackingMode hazardTrackingMode)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setHazardTrackingMode_), hazardTrackingMode);
}

_MTL_INLINE void MTL::TensorDescriptor::setResourceOptions(MTL::ResourceOptions resourceOptions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setResourceOptions_), resourceOptions);
}

_MTL_INLINE void MTL::TensorDescriptor::setStorageMode(MTL::StorageMode storageMode)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStorageMode_), storageMode);
}

_MTL_INLINE void MTL::TensorDescriptor::setStrides(const MTL::TensorExtents* strides)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setStrides_), strides);
}

_MTL_INLINE void MTL::TensorDescriptor::setUsage(MTL::TensorUsage usage)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setUsage_), usage);
}

_MTL_INLINE MTL::StorageMode MTL::TensorDescriptor::storageMode() const
{
    return Object::sendMessage<MTL::StorageMode>(this, _MTL_PRIVATE_SEL(storageMode));
}

_MTL_INLINE MTL::TensorExtents* MTL::TensorDescriptor::strides() const
{
    return Object::sendMessage<MTL::TensorExtents*>(this, _MTL_PRIVATE_SEL(strides));
}

_MTL_INLINE MTL::TensorUsage MTL::TensorDescriptor::usage() const
{
    return Object::sendMessage<MTL::TensorUsage>(this, _MTL_PRIVATE_SEL(usage));
}

_MTL_INLINE MTL::Buffer* MTL::Tensor::buffer() const
{
    return Object::sendMessage<MTL::Buffer*>(this, _MTL_PRIVATE_SEL(buffer));
}

_MTL_INLINE NS::UInteger MTL::Tensor::bufferOffset() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(bufferOffset));
}

_MTL_INLINE MTL::TensorDataType MTL::Tensor::dataType() const
{
    return Object::sendMessage<MTL::TensorDataType>(this, _MTL_PRIVATE_SEL(dataType));
}

_MTL_INLINE MTL::TensorExtents* MTL::Tensor::dimensions() const
{
    return Object::sendMessage<MTL::TensorExtents*>(this, _MTL_PRIVATE_SEL(dimensions));
}

_MTL_INLINE void MTL::Tensor::getBytes(void* bytes, const MTL::TensorExtents* strides, const MTL::TensorExtents* sliceOrigin, const MTL::TensorExtents* sliceDimensions)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(getBytes_strides_fromSliceOrigin_sliceDimensions_), bytes, strides, sliceOrigin, sliceDimensions);
}

_MTL_INLINE MTL::ResourceID MTL::Tensor::gpuResourceID() const
{
    return Object::sendMessage<MTL::ResourceID>(this, _MTL_PRIVATE_SEL(gpuResourceID));
}

_MTL_INLINE void MTL::Tensor::replaceSliceOrigin(const MTL::TensorExtents* sliceOrigin, const MTL::TensorExtents* sliceDimensions, const void* bytes, const MTL::TensorExtents* strides)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(replaceSliceOrigin_sliceDimensions_withBytes_strides_), sliceOrigin, sliceDimensions, bytes, strides);
}

_MTL_INLINE MTL::TensorExtents* MTL::Tensor::strides() const
{
    return Object::sendMessage<MTL::TensorExtents*>(this, _MTL_PRIVATE_SEL(strides));
}

_MTL_INLINE MTL::TensorUsage MTL::Tensor::usage() const
{
    return Object::sendMessage<MTL::TensorUsage>(this, _MTL_PRIVATE_SEL(usage));
}
