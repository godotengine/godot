#pragma once

#include "MTL4Defines.hpp"
#include "MTL4Blocks.hpp"
#include "MTL4Structs.hpp"
#include "MTL4Bridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace NS {
    class Data;
    class Error;
    class URL;
}

namespace MTL4
{

_MTL4_OPTIONS(NS::UInteger, PipelineDataSetSerializerConfiguration) {
    PipelineDataSetSerializerConfigurationCaptureDescriptors = (1 << 0),
    PipelineDataSetSerializerConfigurationCaptureBinaries = (1 << 1),
};


class PipelineDataSetSerializerDescriptor;
class PipelineDataSetSerializer;

class PipelineDataSetSerializerDescriptor : public NS::Copying<PipelineDataSetSerializerDescriptor>
{
public:
    static PipelineDataSetSerializerDescriptor* alloc();
    PipelineDataSetSerializerDescriptor*        init() const;

    MTL4::PipelineDataSetSerializerConfiguration configuration() const;
    void                                         setConfiguration(MTL4::PipelineDataSetSerializerConfiguration configuration);

};

class PipelineDataSetSerializer : public NS::Referencing<PipelineDataSetSerializer>
{
public:
    bool      serializeAsArchiveAndFlushToURL(NS::URL* url, NS::Error** error);
    NS::Data* serializeAsPipelinesScript(NS::Error** error);

};

} // namespace MTL4

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTL4PipelineDataSetSerializerDescriptor;
extern "C" void *OBJC_CLASS_$_MTL4PipelineDataSetSerializer;

_MTL4_INLINE MTL4::PipelineDataSetSerializerDescriptor* MTL4::PipelineDataSetSerializerDescriptor::alloc()
{
    return _MTL4_msg_MTL4__PipelineDataSetSerializerDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTL4PipelineDataSetSerializerDescriptor, nullptr);
}

_MTL4_INLINE MTL4::PipelineDataSetSerializerDescriptor* MTL4::PipelineDataSetSerializerDescriptor::init() const
{
    return _MTL4_msg_MTL4__PipelineDataSetSerializerDescriptorp_init((const void*)this, nullptr);
}

_MTL4_INLINE MTL4::PipelineDataSetSerializerConfiguration MTL4::PipelineDataSetSerializerDescriptor::configuration() const
{
    return _MTL4_msg_MTL4__PipelineDataSetSerializerConfiguration_configuration((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::PipelineDataSetSerializerDescriptor::setConfiguration(MTL4::PipelineDataSetSerializerConfiguration configuration)
{
    _MTL4_msg_v_setConfiguration__MTL4__PipelineDataSetSerializerConfiguration((const void*)this, nullptr, configuration);
}

_MTL4_INLINE bool MTL4::PipelineDataSetSerializer::serializeAsArchiveAndFlushToURL(NS::URL* url, NS::Error** error)
{
    return _MTL4_msg_bool_serializeAsArchiveAndFlushToURL_error__NS__URLp_NS__Errorpp((const void*)this, nullptr, url, error);
}

_MTL4_INLINE NS::Data* MTL4::PipelineDataSetSerializer::serializeAsPipelinesScript(NS::Error** error)
{
    return _MTL4_msg_NS__Datap_serializeAsPipelinesScriptWithError__NS__Errorpp((const void*)this, nullptr, error);
}
