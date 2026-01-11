//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLRasterizationRate.hpp
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
#include "MTLDevice.hpp"
#include "MTLHeaderBridge.hpp"
#include "MTLPrivate.hpp"
#include "MTLTypes.hpp"

namespace MTL
{
class Buffer;
class Device;
class RasterizationRateLayerArray;
class RasterizationRateLayerDescriptor;
class RasterizationRateMapDescriptor;
class RasterizationRateSampleArray;

class RasterizationRateSampleArray : public NS::Referencing<RasterizationRateSampleArray>
{
public:
    static RasterizationRateSampleArray* alloc();

    RasterizationRateSampleArray*        init();

    NS::Number*                          object(NS::UInteger index);
    void                                 setObject(const NS::Number* value, NS::UInteger index);
};
class RasterizationRateLayerDescriptor : public NS::Copying<RasterizationRateLayerDescriptor>
{
public:
    static RasterizationRateLayerDescriptor* alloc();

    RasterizationRateSampleArray*            horizontal() const;
    float*                                   horizontalSampleStorage() const;

    RasterizationRateLayerDescriptor*        init();
    RasterizationRateLayerDescriptor*        init(MTL::Size sampleCount);
    RasterizationRateLayerDescriptor*        init(MTL::Size sampleCount, const float* horizontal, const float* vertical);

    Size                                     maxSampleCount() const;
    Size                                     sampleCount() const;
    void                                     setSampleCount(MTL::Size sampleCount);

    RasterizationRateSampleArray*            vertical() const;
    float*                                   verticalSampleStorage() const;
};
class RasterizationRateLayerArray : public NS::Referencing<RasterizationRateLayerArray>
{
public:
    static RasterizationRateLayerArray* alloc();

    RasterizationRateLayerArray*        init();

    RasterizationRateLayerDescriptor*   object(NS::UInteger layerIndex);
    void                                setObject(const MTL::RasterizationRateLayerDescriptor* layer, NS::UInteger layerIndex);
};
class RasterizationRateMapDescriptor : public NS::Copying<RasterizationRateMapDescriptor>
{
public:
    static RasterizationRateMapDescriptor* alloc();

    RasterizationRateMapDescriptor*        init();

    NS::String*                            label() const;

    RasterizationRateLayerDescriptor*      layer(NS::UInteger layerIndex);
    NS::UInteger                           layerCount() const;

    RasterizationRateLayerArray*           layers() const;

    static RasterizationRateMapDescriptor* rasterizationRateMapDescriptor(MTL::Size screenSize);
    static RasterizationRateMapDescriptor* rasterizationRateMapDescriptor(MTL::Size screenSize, const MTL::RasterizationRateLayerDescriptor* layer);
    static RasterizationRateMapDescriptor* rasterizationRateMapDescriptor(MTL::Size screenSize, NS::UInteger layerCount, const MTL::RasterizationRateLayerDescriptor* const* layers);

    Size                                   screenSize() const;

    void                                   setLabel(const NS::String* label);

    void                                   setLayer(const MTL::RasterizationRateLayerDescriptor* layer, NS::UInteger layerIndex);

    void                                   setScreenSize(MTL::Size screenSize);
};
class RasterizationRateMap : public NS::Referencing<RasterizationRateMap>
{
public:
    void         copyParameterDataToBuffer(const MTL::Buffer* buffer, NS::UInteger offset);

    Device*      device() const;

    NS::String*  label() const;

    NS::UInteger layerCount() const;

    Coordinate2D mapPhysicalToScreenCoordinates(MTL::Coordinate2D physicalCoordinates, NS::UInteger layerIndex);

    Coordinate2D mapScreenToPhysicalCoordinates(MTL::Coordinate2D screenCoordinates, NS::UInteger layerIndex);

    SizeAndAlign parameterBufferSizeAndAlign() const;

    Size         physicalGranularity() const;

    Size         physicalSize(NS::UInteger layerIndex);

    Size         screenSize() const;
};

}
_MTL_INLINE MTL::RasterizationRateSampleArray* MTL::RasterizationRateSampleArray::alloc()
{
    return NS::Object::alloc<MTL::RasterizationRateSampleArray>(_MTL_PRIVATE_CLS(MTLRasterizationRateSampleArray));
}

_MTL_INLINE MTL::RasterizationRateSampleArray* MTL::RasterizationRateSampleArray::init()
{
    return NS::Object::init<MTL::RasterizationRateSampleArray>();
}

_MTL_INLINE NS::Number* MTL::RasterizationRateSampleArray::object(NS::UInteger index)
{
    return Object::sendMessage<NS::Number*>(this, _MTL_PRIVATE_SEL(objectAtIndexedSubscript_), index);
}

_MTL_INLINE void MTL::RasterizationRateSampleArray::setObject(const NS::Number* value, NS::UInteger index)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setObject_atIndexedSubscript_), value, index);
}

_MTL_INLINE MTL::RasterizationRateLayerDescriptor* MTL::RasterizationRateLayerDescriptor::alloc()
{
    return NS::Object::alloc<MTL::RasterizationRateLayerDescriptor>(_MTL_PRIVATE_CLS(MTLRasterizationRateLayerDescriptor));
}

_MTL_INLINE MTL::RasterizationRateSampleArray* MTL::RasterizationRateLayerDescriptor::horizontal() const
{
    return Object::sendMessage<MTL::RasterizationRateSampleArray*>(this, _MTL_PRIVATE_SEL(horizontal));
}

_MTL_INLINE float* MTL::RasterizationRateLayerDescriptor::horizontalSampleStorage() const
{
    return Object::sendMessage<float*>(this, _MTL_PRIVATE_SEL(horizontalSampleStorage));
}

_MTL_INLINE MTL::RasterizationRateLayerDescriptor* MTL::RasterizationRateLayerDescriptor::init()
{
    return NS::Object::init<MTL::RasterizationRateLayerDescriptor>();
}

_MTL_INLINE MTL::RasterizationRateLayerDescriptor* MTL::RasterizationRateLayerDescriptor::init(MTL::Size sampleCount)
{
    return Object::sendMessage<MTL::RasterizationRateLayerDescriptor*>(this, _MTL_PRIVATE_SEL(initWithSampleCount_), sampleCount);
}

_MTL_INLINE MTL::RasterizationRateLayerDescriptor* MTL::RasterizationRateLayerDescriptor::init(MTL::Size sampleCount, const float* horizontal, const float* vertical)
{
    return Object::sendMessage<MTL::RasterizationRateLayerDescriptor*>(this, _MTL_PRIVATE_SEL(initWithSampleCount_horizontal_vertical_), sampleCount, horizontal, vertical);
}

_MTL_INLINE MTL::Size MTL::RasterizationRateLayerDescriptor::maxSampleCount() const
{
    return Object::sendMessage<MTL::Size>(this, _MTL_PRIVATE_SEL(maxSampleCount));
}

_MTL_INLINE MTL::Size MTL::RasterizationRateLayerDescriptor::sampleCount() const
{
    return Object::sendMessage<MTL::Size>(this, _MTL_PRIVATE_SEL(sampleCount));
}

_MTL_INLINE void MTL::RasterizationRateLayerDescriptor::setSampleCount(MTL::Size sampleCount)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setSampleCount_), sampleCount);
}

_MTL_INLINE MTL::RasterizationRateSampleArray* MTL::RasterizationRateLayerDescriptor::vertical() const
{
    return Object::sendMessage<MTL::RasterizationRateSampleArray*>(this, _MTL_PRIVATE_SEL(vertical));
}

_MTL_INLINE float* MTL::RasterizationRateLayerDescriptor::verticalSampleStorage() const
{
    return Object::sendMessage<float*>(this, _MTL_PRIVATE_SEL(verticalSampleStorage));
}

_MTL_INLINE MTL::RasterizationRateLayerArray* MTL::RasterizationRateLayerArray::alloc()
{
    return NS::Object::alloc<MTL::RasterizationRateLayerArray>(_MTL_PRIVATE_CLS(MTLRasterizationRateLayerArray));
}

_MTL_INLINE MTL::RasterizationRateLayerArray* MTL::RasterizationRateLayerArray::init()
{
    return NS::Object::init<MTL::RasterizationRateLayerArray>();
}

_MTL_INLINE MTL::RasterizationRateLayerDescriptor* MTL::RasterizationRateLayerArray::object(NS::UInteger layerIndex)
{
    return Object::sendMessage<MTL::RasterizationRateLayerDescriptor*>(this, _MTL_PRIVATE_SEL(objectAtIndexedSubscript_), layerIndex);
}

_MTL_INLINE void MTL::RasterizationRateLayerArray::setObject(const MTL::RasterizationRateLayerDescriptor* layer, NS::UInteger layerIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setObject_atIndexedSubscript_), layer, layerIndex);
}

_MTL_INLINE MTL::RasterizationRateMapDescriptor* MTL::RasterizationRateMapDescriptor::alloc()
{
    return NS::Object::alloc<MTL::RasterizationRateMapDescriptor>(_MTL_PRIVATE_CLS(MTLRasterizationRateMapDescriptor));
}

_MTL_INLINE MTL::RasterizationRateMapDescriptor* MTL::RasterizationRateMapDescriptor::init()
{
    return NS::Object::init<MTL::RasterizationRateMapDescriptor>();
}

_MTL_INLINE NS::String* MTL::RasterizationRateMapDescriptor::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE MTL::RasterizationRateLayerDescriptor* MTL::RasterizationRateMapDescriptor::layer(NS::UInteger layerIndex)
{
    return Object::sendMessage<MTL::RasterizationRateLayerDescriptor*>(this, _MTL_PRIVATE_SEL(layerAtIndex_), layerIndex);
}

_MTL_INLINE NS::UInteger MTL::RasterizationRateMapDescriptor::layerCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(layerCount));
}

_MTL_INLINE MTL::RasterizationRateLayerArray* MTL::RasterizationRateMapDescriptor::layers() const
{
    return Object::sendMessage<MTL::RasterizationRateLayerArray*>(this, _MTL_PRIVATE_SEL(layers));
}

_MTL_INLINE MTL::RasterizationRateMapDescriptor* MTL::RasterizationRateMapDescriptor::rasterizationRateMapDescriptor(MTL::Size screenSize)
{
    return Object::sendMessage<MTL::RasterizationRateMapDescriptor*>(_MTL_PRIVATE_CLS(MTLRasterizationRateMapDescriptor), _MTL_PRIVATE_SEL(rasterizationRateMapDescriptorWithScreenSize_), screenSize);
}

_MTL_INLINE MTL::RasterizationRateMapDescriptor* MTL::RasterizationRateMapDescriptor::rasterizationRateMapDescriptor(MTL::Size screenSize, const MTL::RasterizationRateLayerDescriptor* layer)
{
    return Object::sendMessage<MTL::RasterizationRateMapDescriptor*>(_MTL_PRIVATE_CLS(MTLRasterizationRateMapDescriptor), _MTL_PRIVATE_SEL(rasterizationRateMapDescriptorWithScreenSize_layer_), screenSize, layer);
}

_MTL_INLINE MTL::RasterizationRateMapDescriptor* MTL::RasterizationRateMapDescriptor::rasterizationRateMapDescriptor(MTL::Size screenSize, NS::UInteger layerCount, const MTL::RasterizationRateLayerDescriptor* const* layers)
{
    return Object::sendMessage<MTL::RasterizationRateMapDescriptor*>(_MTL_PRIVATE_CLS(MTLRasterizationRateMapDescriptor), _MTL_PRIVATE_SEL(rasterizationRateMapDescriptorWithScreenSize_layerCount_layers_), screenSize, layerCount, layers);
}

_MTL_INLINE MTL::Size MTL::RasterizationRateMapDescriptor::screenSize() const
{
    return Object::sendMessage<MTL::Size>(this, _MTL_PRIVATE_SEL(screenSize));
}

_MTL_INLINE void MTL::RasterizationRateMapDescriptor::setLabel(const NS::String* label)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLabel_), label);
}

_MTL_INLINE void MTL::RasterizationRateMapDescriptor::setLayer(const MTL::RasterizationRateLayerDescriptor* layer, NS::UInteger layerIndex)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLayer_atIndex_), layer, layerIndex);
}

_MTL_INLINE void MTL::RasterizationRateMapDescriptor::setScreenSize(MTL::Size screenSize)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setScreenSize_), screenSize);
}

_MTL_INLINE void MTL::RasterizationRateMap::copyParameterDataToBuffer(const MTL::Buffer* buffer, NS::UInteger offset)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(copyParameterDataToBuffer_offset_), buffer, offset);
}

_MTL_INLINE MTL::Device* MTL::RasterizationRateMap::device() const
{
    return Object::sendMessage<MTL::Device*>(this, _MTL_PRIVATE_SEL(device));
}

_MTL_INLINE NS::String* MTL::RasterizationRateMap::label() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(label));
}

_MTL_INLINE NS::UInteger MTL::RasterizationRateMap::layerCount() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(layerCount));
}

_MTL_INLINE MTL::Coordinate2D MTL::RasterizationRateMap::mapPhysicalToScreenCoordinates(MTL::Coordinate2D physicalCoordinates, NS::UInteger layerIndex)
{
    return Object::sendMessage<MTL::Coordinate2D>(this, _MTL_PRIVATE_SEL(mapPhysicalToScreenCoordinates_forLayer_), physicalCoordinates, layerIndex);
}

_MTL_INLINE MTL::Coordinate2D MTL::RasterizationRateMap::mapScreenToPhysicalCoordinates(MTL::Coordinate2D screenCoordinates, NS::UInteger layerIndex)
{
    return Object::sendMessage<MTL::Coordinate2D>(this, _MTL_PRIVATE_SEL(mapScreenToPhysicalCoordinates_forLayer_), screenCoordinates, layerIndex);
}

_MTL_INLINE MTL::SizeAndAlign MTL::RasterizationRateMap::parameterBufferSizeAndAlign() const
{
    return Object::sendMessage<MTL::SizeAndAlign>(this, _MTL_PRIVATE_SEL(parameterBufferSizeAndAlign));
}

_MTL_INLINE MTL::Size MTL::RasterizationRateMap::physicalGranularity() const
{
    return Object::sendMessage<MTL::Size>(this, _MTL_PRIVATE_SEL(physicalGranularity));
}

_MTL_INLINE MTL::Size MTL::RasterizationRateMap::physicalSize(NS::UInteger layerIndex)
{
    return Object::sendMessage<MTL::Size>(this, _MTL_PRIVATE_SEL(physicalSizeForLayer_), layerIndex);
}

_MTL_INLINE MTL::Size MTL::RasterizationRateMap::screenSize() const
{
    return Object::sendMessage<MTL::Size>(this, _MTL_PRIVATE_SEL(screenSize));
}
