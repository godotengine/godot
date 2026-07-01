#pragma once

#include "MTL4Defines.hpp"
#include "MTL4Blocks.hpp"
#include "MTL4Structs.hpp"
#include "MTL4Bridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL {
    class CompileOptions;
}
namespace NS {
    class String;
}

namespace MTL4
{

class LibraryDescriptor : public NS::Copying<LibraryDescriptor>
{
public:
    static LibraryDescriptor* alloc();
    LibraryDescriptor*        init() const;

    NS::String*          name() const;
    MTL::CompileOptions* options() const;
    void                 setName(NS::String* name);
    void                 setOptions(MTL::CompileOptions* options);
    void                 setSource(NS::String* source);
    NS::String*          source() const;

};

} // namespace MTL4

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTL4LibraryDescriptor;

_MTL4_INLINE MTL4::LibraryDescriptor* MTL4::LibraryDescriptor::alloc()
{
    return _MTL4_msg_MTL4__LibraryDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTL4LibraryDescriptor, nullptr);
}

_MTL4_INLINE MTL4::LibraryDescriptor* MTL4::LibraryDescriptor::init() const
{
    return _MTL4_msg_MTL4__LibraryDescriptorp_init((const void*)this, nullptr);
}

_MTL4_INLINE NS::String* MTL4::LibraryDescriptor::source() const
{
    return _MTL4_msg_NS__Stringp_source((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::LibraryDescriptor::setSource(NS::String* source)
{
    _MTL4_msg_v_setSource__NS__Stringp((const void*)this, nullptr, source);
}

_MTL4_INLINE MTL::CompileOptions* MTL4::LibraryDescriptor::options() const
{
    return _MTL4_msg_MTL__CompileOptionsp_options((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::LibraryDescriptor::setOptions(MTL::CompileOptions* options)
{
    _MTL4_msg_v_setOptions__MTL__CompileOptionsp((const void*)this, nullptr, options);
}

_MTL4_INLINE NS::String* MTL4::LibraryDescriptor::name() const
{
    return _MTL4_msg_NS__Stringp_name((const void*)this, nullptr);
}

_MTL4_INLINE void MTL4::LibraryDescriptor::setName(NS::String* name)
{
    _MTL4_msg_v_setName__NS__Stringp((const void*)this, nullptr, name);
}
