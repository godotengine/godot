#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL {
    class Device;
}
namespace NS {
    class Error;
    class String;
    class URL;
}

namespace MTL
{

extern NS::ErrorDomain const DynamicLibraryDomain __asm__("_MTLDynamicLibraryDomain");
_MTL_ENUM(NS::UInteger, DynamicLibraryError) {
    DynamicLibraryErrorNone = 0,
    DynamicLibraryErrorInvalidFile = 1,
    DynamicLibraryErrorCompilationFailure = 2,
    DynamicLibraryErrorUnresolvedInstallName = 3,
    DynamicLibraryErrorDependencyLoadFailure = 4,
    DynamicLibraryErrorUnsupported = 5,
};


class DynamicLibrary : public NS::Referencing<DynamicLibrary>
{
public:
    MTL::Device* device() const;
    NS::String*  installName() const;
    NS::String*  label() const;
    bool         serializeToURL(NS::URL* url, NS::Error** error);
    void         setLabel(NS::String* label);

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLDynamicLibrary;

_MTL_INLINE NS::String* MTL::DynamicLibrary::label() const
{
    return _MTL_msg_NS__Stringp_label((const void*)this, nullptr);
}

_MTL_INLINE void MTL::DynamicLibrary::setLabel(NS::String* label)
{
    _MTL_msg_v_setLabel__NS__Stringp((const void*)this, nullptr, label);
}

_MTL_INLINE MTL::Device* MTL::DynamicLibrary::device() const
{
    return _MTL_msg_MTL__Devicep_device((const void*)this, nullptr);
}

_MTL_INLINE NS::String* MTL::DynamicLibrary::installName() const
{
    return _MTL_msg_NS__Stringp_installName((const void*)this, nullptr);
}

_MTL_INLINE bool MTL::DynamicLibrary::serializeToURL(NS::URL* url, NS::Error** error)
{
    return _MTL_msg_bool_serializeToURL_error__NS__URLp_NS__Errorpp((const void*)this, nullptr, url, error);
}
