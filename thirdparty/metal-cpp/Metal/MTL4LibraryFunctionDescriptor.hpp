#pragma once

#include "MTL4Defines.hpp"
#include "MTL4Blocks.hpp"
#include "MTL4Structs.hpp"
#include "MTL4Bridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"
#include "MTL4FunctionDescriptor.hpp"

namespace MTL {
    class Library;
}
namespace NS {
    class String;
}

namespace MTL4
{

class LibraryFunctionDescriptor : public NS::Referencing<LibraryFunctionDescriptor, MTL4::FunctionDescriptor>
{
public:
    static LibraryFunctionDescriptor* alloc();
    LibraryFunctionDescriptor*        init() const;

    MTL::Library* library() const;
    NS::String*   name() const;
    void          setLibrary(MTL::Library* library);
    void          setName(NS::String* name);

};

} // namespace MTL4

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTL4LibraryFunctionDescriptor;

_MTL4_INLINE MTL4::LibraryFunctionDescriptor* MTL4::LibraryFunctionDescriptor::alloc()
{
    return _MTL4_msg_MTL4__LibraryFunctionDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTL4LibraryFunctionDescriptor, nullptr);
}

_MTL4_INLINE MTL4::LibraryFunctionDescriptor* MTL4::LibraryFunctionDescriptor::init() const
{
    return _MTL4_msg_MTL4__LibraryFunctionDescriptorp_init((const void*)this, nullptr);
}

_MTL4_INLINE NS::String* MTL4::LibraryFunctionDescriptor::name() const
{
    return _MTL4_msg_NS__Stringp_name((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::LibraryFunctionDescriptor::setName(NS::String* name)
{
    _MTL4_msg_v_setName__NS__Stringp((const void*)this, nullptr, name);
}

_MTL4_INLINE MTL::Library* MTL4::LibraryFunctionDescriptor::library() const
{
    return _MTL4_msg_MTL__Libraryp_library((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::LibraryFunctionDescriptor::setLibrary(MTL::Library* library)
{
    _MTL4_msg_v_setLibrary__MTL__Libraryp((const void*)this, nullptr, library);
}
