#pragma once

#include "MTLDefines.hpp"
#include "MTLBlocks.hpp"
#include "MTLStructs.hpp"
#include "MTLBridge.hpp"
#include "../Foundation/NSObject.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

namespace MTL {
    class CaptureScope;
    class CommandQueue;
    class Device;
}
namespace MTL4 {
    class CommandQueue;
}
namespace NS {
    class Error;
    class Object;
    class URL;
}

namespace MTL
{

extern NS::ErrorDomain const CaptureErrorDomain __asm__("_MTLCaptureErrorDomain");
_MTL_ENUM(NS::Integer, CaptureError) {
    CaptureErrorNotSupported = 1,
    CaptureErrorAlreadyCapturing = 2,
    CaptureErrorInvalidDescriptor = 3,
};

_MTL_ENUM(NS::Integer, CaptureDestination) {
    CaptureDestinationDeveloperTools = 1,
    CaptureDestinationGPUTraceDocument = 2,
};


class CaptureDescriptor;
class CaptureManager;

class CaptureDescriptor : public NS::Copying<CaptureDescriptor>
{
public:
    static CaptureDescriptor* alloc();
    CaptureDescriptor*        init() const;

    NS::Object*             captureObject() const;
    MTL::CaptureDestination destination() const;
    NS::URL*                outputURL() const;
    void                    setCaptureObject(NS::Object* captureObject);
    void                    setDestination(MTL::CaptureDestination destination);
    void                    setOutputURL(NS::URL* outputURL);

};

class CaptureManager : public NS::Referencing<CaptureManager>
{
public:
    static CaptureManager* alloc();
    CaptureManager*        init() const;

    static MTL::CaptureManager* sharedCaptureManager();

    MTL::CaptureScope* defaultCaptureScope() const;
    bool               isCapturing() const;
    MTL::CaptureScope* newCaptureScope(MTL::Device* device);
    MTL::CaptureScope* newCaptureScope(MTL::CommandQueue* commandQueue);
    MTL::CaptureScope* newCaptureScope(MTL4::CommandQueue* commandQueue);
    void               setDefaultCaptureScope(MTL::CaptureScope* defaultCaptureScope);
    bool               startCapture(MTL::CaptureDescriptor* descriptor, NS::Error** error);
    void               startCapture(MTL::Device* device);
    void               startCapture(MTL::CommandQueue* commandQueue);
    void               startCapture(MTL::CaptureScope* captureScope);
    void               stopCapture();
    bool               supportsDestination(MTL::CaptureDestination destination);

};

} // namespace MTL

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_MTLCaptureDescriptor;
extern "C" void *OBJC_CLASS_$_MTLCaptureManager;

_MTL_INLINE MTL::CaptureDescriptor* MTL::CaptureDescriptor::alloc()
{
    return _MTL_msg_MTL__CaptureDescriptorp_alloc((const void*)&OBJC_CLASS_$_MTLCaptureDescriptor, nullptr);
}

_MTL_INLINE MTL::CaptureDescriptor* MTL::CaptureDescriptor::init() const
{
    return _MTL_msg_MTL__CaptureDescriptorp_init((const void*)this, nullptr);
}

_MTL_INLINE NS::Object* MTL::CaptureDescriptor::captureObject() const
{
    return _MTL_msg_NS__Objectp_captureObject((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CaptureDescriptor::setCaptureObject(NS::Object* captureObject)
{
    _MTL_msg_v_setCaptureObject__NS__Objectp((const void*)this, nullptr, captureObject);
}

_MTL_INLINE MTL::CaptureDestination MTL::CaptureDescriptor::destination() const
{
    return _MTL_msg_MTL__CaptureDestination_destination((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CaptureDescriptor::setDestination(MTL::CaptureDestination destination)
{
    _MTL_msg_v_setDestination__MTL__CaptureDestination((const void*)this, nullptr, destination);
}

_MTL_INLINE NS::URL* MTL::CaptureDescriptor::outputURL() const
{
    return _MTL_msg_NS__URLp_outputURL((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CaptureDescriptor::setOutputURL(NS::URL* outputURL)
{
    _MTL_msg_v_setOutputURL__NS__URLp((const void*)this, nullptr, outputURL);
}

_MTL_INLINE MTL::CaptureManager* MTL::CaptureManager::alloc()
{
    return _MTL_msg_MTL__CaptureManagerp_alloc((const void*)&OBJC_CLASS_$_MTLCaptureManager, nullptr);
}

_MTL_INLINE MTL::CaptureManager* MTL::CaptureManager::init() const
{
    return _MTL_msg_MTL__CaptureManagerp_init((const void*)this, nullptr);
}

_MTL_INLINE MTL::CaptureManager* MTL::CaptureManager::sharedCaptureManager()
{
    return _MTL_msg_MTL__CaptureManagerp_sharedCaptureManager((const void*)&OBJC_CLASS_$_MTLCaptureManager, nullptr);
}

_MTL_INLINE MTL::CaptureScope* MTL::CaptureManager::defaultCaptureScope() const
{
    return _MTL_msg_MTL__CaptureScopep_defaultCaptureScope((const void*)this, nullptr);
}

_MTL_INLINE void MTL::CaptureManager::setDefaultCaptureScope(MTL::CaptureScope* defaultCaptureScope)
{
    _MTL_msg_v_setDefaultCaptureScope__MTL__CaptureScopep((const void*)this, nullptr, defaultCaptureScope);
}

_MTL_INLINE bool MTL::CaptureManager::isCapturing() const
{
    return _MTL_msg_bool_isCapturing((const void*)this, nullptr);
}

_MTL_INLINE MTL::CaptureScope* MTL::CaptureManager::newCaptureScope(MTL::Device* device)
{
    return _MTL_msg_MTL__CaptureScopep_newCaptureScopeWithDevice__MTL__Devicep((const void*)this, nullptr, device);
}

_MTL_INLINE MTL::CaptureScope* MTL::CaptureManager::newCaptureScope(MTL::CommandQueue* commandQueue)
{
    return _MTL_msg_MTL__CaptureScopep_newCaptureScopeWithCommandQueue__MTL__CommandQueuep((const void*)this, nullptr, commandQueue);
}

_MTL_INLINE MTL::CaptureScope* MTL::CaptureManager::newCaptureScope(MTL4::CommandQueue* commandQueue)
{
    return _MTL_msg_MTL__CaptureScopep_newCaptureScopeWithMTL4CommandQueue__MTL4__CommandQueuep((const void*)this, nullptr, commandQueue);
}

_MTL_INLINE bool MTL::CaptureManager::supportsDestination(MTL::CaptureDestination destination)
{
    return _MTL_msg_bool_supportsDestination__MTL__CaptureDestination((const void*)this, nullptr, destination);
}

_MTL_INLINE bool MTL::CaptureManager::startCapture(MTL::CaptureDescriptor* descriptor, NS::Error** error)
{
    return _MTL_msg_bool_startCaptureWithDescriptor_error__MTL__CaptureDescriptorp_NS__Errorpp((const void*)this, nullptr, descriptor, error);
}

_MTL_INLINE void MTL::CaptureManager::startCapture(MTL::Device* device)
{
    _MTL_msg_v_startCaptureWithDevice__MTL__Devicep((const void*)this, nullptr, device);
}

_MTL_INLINE void MTL::CaptureManager::startCapture(MTL::CommandQueue* commandQueue)
{
    _MTL_msg_v_startCaptureWithCommandQueue__MTL__CommandQueuep((const void*)this, nullptr, commandQueue);
}

_MTL_INLINE void MTL::CaptureManager::startCapture(MTL::CaptureScope* captureScope)
{
    _MTL_msg_v_startCaptureWithScope__MTL__CaptureScopep((const void*)this, nullptr, captureScope);
}

_MTL_INLINE void MTL::CaptureManager::stopCapture()
{
    _MTL_msg_v_stopCapture((const void*)this, nullptr);
}
