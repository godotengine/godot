#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL {
    class ComputePipelineDescriptor;
    class Device;
    class FunctionDescriptor;
    class Library;
    class MeshRenderPipelineDescriptor;
    class RenderPipelineDescriptor;
    class StitchedLibraryDescriptor;
    class TileRenderPipelineDescriptor;
}
namespace NS {
    class Error;
    class String;
    class URL;
}

namespace MTL
{

extern NS::ErrorDomain const BinaryArchiveDomain __asm__("_MTLBinaryArchiveDomain");
_MTL_ENUM(NS::UInteger, BinaryArchiveError) {
    BinaryArchiveErrorNone = 0,
    BinaryArchiveErrorInvalidFile = 1,
    BinaryArchiveErrorUnexpectedElement = 2,
    BinaryArchiveErrorCompilationFailure = 3,
    BinaryArchiveErrorInternalError = 4,
};


class BinaryArchiveDescriptor;
class BinaryArchive;

class BinaryArchiveDescriptor : public NS::Copying<BinaryArchiveDescriptor>
{
public:
    static BinaryArchiveDescriptor* alloc();
    BinaryArchiveDescriptor*        init() const;

    void     setUrl(NS::URL* url);
    NS::URL* url() const;

};

class BinaryArchive : public NS::Referencing<BinaryArchive>
{
public:
    bool         addComputePipelineFunctions(MTL::ComputePipelineDescriptor* descriptor, NS::Error** error);
    bool         addFunction(MTL::FunctionDescriptor* descriptor, MTL::Library* library, NS::Error** error);
    bool         addLibrary(MTL::StitchedLibraryDescriptor* descriptor, NS::Error** error);
    bool         addMeshRenderPipelineFunctions(MTL::MeshRenderPipelineDescriptor* descriptor, NS::Error** error);
    bool         addRenderPipelineFunctions(MTL::RenderPipelineDescriptor* descriptor, NS::Error** error);
    bool         addTileRenderPipelineFunctions(MTL::TileRenderPipelineDescriptor* descriptor, NS::Error** error);
    MTL::Device* device() const;
    NS::String*  label() const;
    bool         serializeToURL(NS::URL* url, NS::Error** error);
    void         setLabel(NS::String* label);

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLBinaryArchiveDescriptor;
extern "C" void *OBJC_CLASS_$_MTLBinaryArchive;

_MTL_INLINE MTL::BinaryArchiveDescriptor* MTL::BinaryArchiveDescriptor::alloc()
{
    return _MTL_msg_MTL__BinaryArchiveDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLBinaryArchiveDescriptor, nullptr);
}

_MTL_INLINE MTL::BinaryArchiveDescriptor* MTL::BinaryArchiveDescriptor::init() const
{
    return _MTL_msg_MTL__BinaryArchiveDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE NS::URL* MTL::BinaryArchiveDescriptor::url() const
{
    return _MTL_msg_NS__URLp_url((const void*)this, nullptr);
}

_MTL_INLINE void MTL::BinaryArchiveDescriptor::setUrl(NS::URL* url)
{
    _MTL_msg_v_setUrl__NS__URLp((const void*)this, nullptr, url);
}

_MTL_INLINE NS::String* MTL::BinaryArchive::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE void MTL::BinaryArchive::setLabel(NS::String* label)
{
    _MTL_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL_INLINE MTL::Device* MTL::BinaryArchive::device() const
{
    return _MTL_msg_MTL__Devicep_device((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::BinaryArchive::addComputePipelineFunctions(MTL::ComputePipelineDescriptor* descriptor, NS::Error** error)
{
    return _MTL_msg_bool_addComputePipelineFunctionsWithDescriptor_error__MTL__ComputePipelineDescriptorp_NS__Errorpp((const void*)this, nullptr, descriptor, error);
}

_MTL_INLINE bool MTL::BinaryArchive::addRenderPipelineFunctions(MTL::RenderPipelineDescriptor* descriptor, NS::Error** error)
{
    return _MTL_msg_bool_addRenderPipelineFunctionsWithDescriptor_error__MTL__RenderPipelineDescriptorp_NS__Errorpp((const void*)this, nullptr, descriptor, error);
}

_MTL_INLINE bool MTL::BinaryArchive::addTileRenderPipelineFunctions(MTL::TileRenderPipelineDescriptor* descriptor, NS::Error** error)
{
    return _MTL_msg_bool_addTileRenderPipelineFunctionsWithDescriptor_error__MTL__TileRenderPipelineDescriptorp_NS__Errorpp((const void*)this, nullptr, descriptor, error);
}

_MTL_INLINE bool MTL::BinaryArchive::addMeshRenderPipelineFunctions(MTL::MeshRenderPipelineDescriptor* descriptor, NS::Error** error)
{
    return _MTL_msg_bool_addMeshRenderPipelineFunctionsWithDescriptor_error__MTL__MeshRenderPipelineDescriptorp_NS__Errorpp((const void*)this, nullptr, descriptor, error);
}

_MTL_INLINE bool MTL::BinaryArchive::addLibrary(MTL::StitchedLibraryDescriptor* descriptor, NS::Error** error)
{
    return _MTL_msg_bool_addLibraryWithDescriptor_error__MTL__StitchedLibraryDescriptorp_NS__Errorpp((const void*)this, nullptr, descriptor, error);
}

_MTL_INLINE bool MTL::BinaryArchive::serializeToURL(NS::URL* url, NS::Error** error)
{
    return _MTL_msg_bool_serializeToURL_error__NS__URLp_NS__Errorpp((const void*)this, nullptr, url, error);
}

_MTL_INLINE bool MTL::BinaryArchive::addFunction(MTL::FunctionDescriptor* descriptor, MTL::Library* library, NS::Error** error)
{
    return _MTL_msg_bool_addFunctionWithDescriptor_library_error__MTL__FunctionDescriptorp_MTL__Libraryp_NS__Errorpp((const void*)this, nullptr, descriptor, library, error);
}
