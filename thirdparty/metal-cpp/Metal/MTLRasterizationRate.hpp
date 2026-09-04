#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL {
    class Buffer;
    class Device;
}
namespace NS {
    class Number;
    class String;
}

namespace MTL
{

class RasterizationRateSampleArray;
class RasterizationRateLayerDescriptor;
class RasterizationRateLayerArray;
class RasterizationRateMapDescriptor;
class RasterizationRateMap;

class RasterizationRateSampleArray : public NS::Referencing<RasterizationRateSampleArray>
{
public:
    static RasterizationRateSampleArray* alloc();
    RasterizationRateSampleArray*        init() const;

    NS::Number* object(NS::UInteger index);
    void        setObject(NS::Number* value, NS::UInteger index);

};

class RasterizationRateLayerDescriptor : public NS::Copying<RasterizationRateLayerDescriptor>
{
public:
    static RasterizationRateLayerDescriptor* alloc();
    RasterizationRateLayerDescriptor*        init() const;

    MTL::RasterizationRateSampleArray*     horizontal() const;
    float *                                horizontalSampleStorage() const;
    MTL::RasterizationRateLayerDescriptor* init(MTL::Size sampleCount);
    MTL::RasterizationRateLayerDescriptor* init(MTL::Size sampleCount, const float * horizontal, const float * vertical);
    MTL::Size                              maxSampleCount() const;
    MTL::Size                              sampleCount() const;
    MTL::RasterizationRateSampleArray*     vertical() const;
    float *                                verticalSampleStorage() const;

};

class RasterizationRateLayerArray : public NS::Referencing<RasterizationRateLayerArray>
{
public:
    static RasterizationRateLayerArray* alloc();
    RasterizationRateLayerArray*        init() const;

    MTL::RasterizationRateLayerDescriptor* object(NS::UInteger layerIndex);
    void                                   setObject(MTL::RasterizationRateLayerDescriptor* layer, NS::UInteger layerIndex);

};

class RasterizationRateMapDescriptor : public NS::Copying<RasterizationRateMapDescriptor>
{
public:
    static RasterizationRateMapDescriptor* alloc();
    RasterizationRateMapDescriptor*        init() const;

    static MTL::RasterizationRateMapDescriptor* rasterizationRateMapDescriptor(MTL::Size screenSize);
    static MTL::RasterizationRateMapDescriptor* rasterizationRateMapDescriptor(MTL::Size screenSize, MTL::RasterizationRateLayerDescriptor* layer);
    static MTL::RasterizationRateMapDescriptor* rasterizationRateMapDescriptor(MTL::Size screenSize, NS::UInteger layerCount, const MTL::RasterizationRateLayerDescriptor* const * layers);

    NS::String*                            label() const;
    MTL::RasterizationRateLayerDescriptor* layer(NS::UInteger layerIndex);
    NS::UInteger                           layerCount() const;
    MTL::RasterizationRateLayerArray*      layers() const;
    MTL::Size                              screenSize() const;
    void                                   setLabel(NS::String* label);
    void                                   setLayer(MTL::RasterizationRateLayerDescriptor* layer, NS::UInteger layerIndex);
    void                                   setScreenSize(MTL::Size screenSize);

};

class RasterizationRateMap : public NS::Referencing<RasterizationRateMap>
{
public:
    void              copyParameterDataToBuffer(MTL::Buffer* buffer, NS::UInteger offset);
    MTL::Device*      device() const;
    NS::String*       label() const;
    NS::UInteger      layerCount() const;
    MTL::Coordinate2D mapPhysicalToScreenCoordinates(MTL::Coordinate2D physicalCoordinates, NS::UInteger layerIndex);
    MTL::Coordinate2D mapScreenToPhysicalCoordinates(MTL::Coordinate2D screenCoordinates, NS::UInteger layerIndex);
    MTL::SizeAndAlign parameterBufferSizeAndAlign() const;
    MTL::Size         physicalGranularity() const;
    MTL::Size         physicalSize(NS::UInteger layerIndex);
    MTL::Size         screenSize() const;

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLRasterizationRateSampleArray;
extern "C" void *OBJC_CLASS_$_MTLRasterizationRateLayerDescriptor;
extern "C" void *OBJC_CLASS_$_MTLRasterizationRateLayerArray;
extern "C" void *OBJC_CLASS_$_MTLRasterizationRateMapDescriptor;
extern "C" void *OBJC_CLASS_$_MTLRasterizationRateMap;

_MTL_INLINE MTL::RasterizationRateSampleArray* MTL::RasterizationRateSampleArray::alloc()
{
    return _MTL_msg_MTL__RasterizationRateSampleArrayp_alloc((const void*)&OBJC_CLASS_$_MTLRasterizationRateSampleArray, nullptr);
}

_MTL_INLINE MTL::RasterizationRateSampleArray* MTL::RasterizationRateSampleArray::init() const
{
    return _MTL_msg_MTL__RasterizationRateSampleArrayp_init((const void*)this, nullptr);
}

_MTL_INLINE NS::Number* MTL::RasterizationRateSampleArray::object(NS::UInteger index)
{
    return _MTL_msg_NS__Numberp_objectAtIndexedSubscript__NS__UInteger((const void*)this, nullptr, index);
}

_MTL_INLINE void MTL::RasterizationRateSampleArray::setObject(NS::Number* value, NS::UInteger index)
{
    _MTL_msg_v_setObject_atIndexedSubscript__NS__Numberp_NS__UInteger((const void*)this, nullptr, value, index);
}

_MTL_INLINE MTL::RasterizationRateLayerDescriptor* MTL::RasterizationRateLayerDescriptor::alloc()
{
    return _MTL_msg_MTL__RasterizationRateLayerDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLRasterizationRateLayerDescriptor, nullptr);
}

_MTL_INLINE MTL::RasterizationRateLayerDescriptor* MTL::RasterizationRateLayerDescriptor::init() const
{
    return _MTL_msg_MTL__RasterizationRateLayerDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::Size MTL::RasterizationRateLayerDescriptor::sampleCount() const
{
    return _MTL_msg_MTL__Size_sampleCount((const void*)this, nullptr);
}

_MTL_INLINE MTL::Size MTL::RasterizationRateLayerDescriptor::maxSampleCount() const
{
    return _MTL_msg_MTL__Size_maxSampleCount((const void*)this, nullptr);
}

_MTL_INLINE float * MTL::RasterizationRateLayerDescriptor::horizontalSampleStorage() const
{
    return _MTL_msg_floatp_horizontalSampleStorage((const void*)this, nullptr);
}

_MTL_INLINE float * MTL::RasterizationRateLayerDescriptor::verticalSampleStorage() const
{
    return _MTL_msg_floatp_verticalSampleStorage((const void*)this, nullptr);
}

_MTL_INLINE MTL::RasterizationRateSampleArray* MTL::RasterizationRateLayerDescriptor::horizontal() const
{
    return _MTL_msg_MTL__RasterizationRateSampleArrayp_horizontal((const void*)this, nullptr);
}

_MTL_INLINE MTL::RasterizationRateSampleArray* MTL::RasterizationRateLayerDescriptor::vertical() const
{
    return _MTL_msg_MTL__RasterizationRateSampleArrayp_vertical((const void*)this, nullptr);
}

_MTL_INLINE MTL::RasterizationRateLayerDescriptor* MTL::RasterizationRateLayerDescriptor::init(MTL::Size sampleCount)
{
    return _MTL_msg_MTL__RasterizationRateLayerDescriptorp_initWithSampleCount__MTL__Size((const void*)this, nullptr, sampleCount);
}

_MTL_INLINE MTL::RasterizationRateLayerDescriptor* MTL::RasterizationRateLayerDescriptor::init(MTL::Size sampleCount, const float * horizontal, const float * vertical)
{
    return _MTL_msg_MTL__RasterizationRateLayerDescriptorp_initWithSampleCount_horizontal_vertical__MTL__Size_constfloatp_constfloatp((const void*)this, nullptr, sampleCount, horizontal, vertical);
}

_MTL_INLINE MTL::RasterizationRateLayerArray* MTL::RasterizationRateLayerArray::alloc()
{
    return _MTL_msg_MTL__RasterizationRateLayerArrayp_alloc((const void*)&OBJC_CLASS_$_MTLRasterizationRateLayerArray, nullptr);
}

_MTL_INLINE MTL::RasterizationRateLayerArray* MTL::RasterizationRateLayerArray::init() const
{
    return _MTL_msg_MTL__RasterizationRateLayerArrayp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::RasterizationRateLayerDescriptor* MTL::RasterizationRateLayerArray::object(NS::UInteger layerIndex)
{
    return _MTL_msg_MTL__RasterizationRateLayerDescriptorp_objectAtIndexedSubscript__NS__UInteger((const void*)this, nullptr, layerIndex);
}

_MTL_INLINE void MTL::RasterizationRateLayerArray::setObject(MTL::RasterizationRateLayerDescriptor* layer, NS::UInteger layerIndex)
{
    _MTL_msg_v_setObject_atIndexedSubscript__MTL__RasterizationRateLayerDescriptorp_NS__UInteger((const void*)this, nullptr, layer, layerIndex);
}

_MTL_INLINE MTL::RasterizationRateMapDescriptor* MTL::RasterizationRateMapDescriptor::alloc()
{
    return _MTL_msg_MTL__RasterizationRateMapDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLRasterizationRateMapDescriptor, nullptr);
}

_MTL_INLINE MTL::RasterizationRateMapDescriptor* MTL::RasterizationRateMapDescriptor::init() const
{
    return _MTL_msg_MTL__RasterizationRateMapDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::RasterizationRateMapDescriptor* MTL::RasterizationRateMapDescriptor::rasterizationRateMapDescriptor(MTL::Size screenSize)
{
    return _MTL_msg_MTL__RasterizationRateMapDescriptorp_rasterizationRateMapDescriptorWithScreenSize__MTL__Size((const void*)&OBJC_CLASS_$_MTLRasterizationRateMapDescriptor, nullptr, screenSize);
}

_MTL_INLINE MTL::RasterizationRateMapDescriptor* MTL::RasterizationRateMapDescriptor::rasterizationRateMapDescriptor(MTL::Size screenSize, MTL::RasterizationRateLayerDescriptor* layer)
{
    return _MTL_msg_MTL__RasterizationRateMapDescriptorp_rasterizationRateMapDescriptorWithScreenSize_layer__MTL__Size_MTL__RasterizationRateLayerDescriptorp((const void*)&OBJC_CLASS_$_MTLRasterizationRateMapDescriptor, nullptr, screenSize, layer);
}

_MTL_INLINE MTL::RasterizationRateMapDescriptor* MTL::RasterizationRateMapDescriptor::rasterizationRateMapDescriptor(MTL::Size screenSize, NS::UInteger layerCount, const MTL::RasterizationRateLayerDescriptor* const * layers)
{
    return _MTL_msg_MTL__RasterizationRateMapDescriptorp_rasterizationRateMapDescriptorWithScreenSize_layerCount_layers__MTL__Size_NS__UInteger_constMTL__RasterizationRateLayerDescriptorpconstp((const void*)&OBJC_CLASS_$_MTLRasterizationRateMapDescriptor, nullptr, screenSize, layerCount, layers);
}

_MTL_INLINE MTL::RasterizationRateLayerArray* MTL::RasterizationRateMapDescriptor::layers() const
{
    return _MTL_msg_MTL__RasterizationRateLayerArrayp_layers((const void*)this, nullptr);
}

_MTL_INLINE MTL::Size MTL::RasterizationRateMapDescriptor::screenSize() const
{
    return _MTL_msg_MTL__Size_screenSize((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RasterizationRateMapDescriptor::setScreenSize(MTL::Size screenSize)
{
    _MTL_msg_v_setScreenSize__MTL__Size((const void*)this, nullptr, screenSize);
}

_MTL_INLINE NS::String* MTL::RasterizationRateMapDescriptor::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RasterizationRateMapDescriptor::setLabel(NS::String* label)
{
    _MTL_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL_INLINE NS::UInteger MTL::RasterizationRateMapDescriptor::layerCount() const
{
    return _MTL_msg_NS__UInteger_layerCount((const void*)this, nullptr);
}

_MTL_INLINE MTL::RasterizationRateLayerDescriptor* MTL::RasterizationRateMapDescriptor::layer(NS::UInteger layerIndex)
{
    return _MTL_msg_MTL__RasterizationRateLayerDescriptorp_layerAtIndex__NS__UInteger((const void*)this, nullptr, layerIndex);
}

_MTL_INLINE void MTL::RasterizationRateMapDescriptor::setLayer(MTL::RasterizationRateLayerDescriptor* layer, NS::UInteger layerIndex)
{
    _MTL_msg_v_setLayer_atIndex__MTL__RasterizationRateLayerDescriptorp_NS__UInteger((const void*)this, nullptr, layer, layerIndex);
}

_MTL_INLINE MTL::Device* MTL::RasterizationRateMap::device() const
{
    return _MTL_msg_MTL__Devicep_device((const void*)this, nullptr);
}

_MTL_INLINE NS::String* MTL::RasterizationRateMap::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE MTL::Size MTL::RasterizationRateMap::screenSize() const
{
    return _MTL_msg_MTL__Size_screenSize((const void*)this, nullptr);
}

_MTL_INLINE MTL::Size MTL::RasterizationRateMap::physicalGranularity() const
{
    return _MTL_msg_MTL__Size_physicalGranularity((const void*)this, nullptr);
}

_MTL_INLINE NS::UInteger MTL::RasterizationRateMap::layerCount() const
{
    return _MTL_msg_NS__UInteger_layerCount((const void*)this, nullptr);
}

_MTL_INLINE MTL::SizeAndAlign MTL::RasterizationRateMap::parameterBufferSizeAndAlign() const
{
    return _MTL_msg_MTL__SizeAndAlign_parameterBufferSizeAndAlign((const void*)this, nullptr);
}

_MTL_INLINE void MTL::RasterizationRateMap::copyParameterDataToBuffer(MTL::Buffer* buffer, NS::UInteger offset)
{
    _MTL_msg_v_copyParameterDataToBuffer_offset__MTL__Bufferp_NS__UInteger((const void*)this, nullptr, buffer, offset);
}

_MTL_INLINE MTL::Size MTL::RasterizationRateMap::physicalSize(NS::UInteger layerIndex)
{
    return _MTL_msg_MTL__Size_physicalSizeForLayer__NS__UInteger((const void*)this, nullptr, layerIndex);
}

_MTL_INLINE MTL::Coordinate2D MTL::RasterizationRateMap::mapScreenToPhysicalCoordinates(MTL::Coordinate2D screenCoordinates, NS::UInteger layerIndex)
{
    return _MTL_msg_MTL__SamplePosition_mapScreenToPhysicalCoordinates_forLayer__MTL__SamplePosition_NS__UInteger((const void*)this, nullptr, screenCoordinates, layerIndex);
}

_MTL_INLINE MTL::Coordinate2D MTL::RasterizationRateMap::mapPhysicalToScreenCoordinates(MTL::Coordinate2D physicalCoordinates, NS::UInteger layerIndex)
{
    return _MTL_msg_MTL__SamplePosition_mapPhysicalToScreenCoordinates_forLayer__MTL__SamplePosition_NS__UInteger((const void*)this, nullptr, physicalCoordinates, layerIndex);
}
