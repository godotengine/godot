//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLCaptureManager.hpp
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

namespace MTL
{
class CaptureDescriptor;
class CaptureManager;
class CaptureScope;
class CommandQueue;
class Device;
}

namespace MTL4
{
class CommandQueue;
}

namespace MTL
{
_MTL_ENUM(NS::Integer, CaptureError) {
    CaptureErrorNotSupported = 1,
    CaptureErrorAlreadyCapturing = 2,
    CaptureErrorInvalidDescriptor = 3,
};

_MTL_ENUM(NS::Integer, CaptureDestination) {
    CaptureDestinationDeveloperTools = 1,
    CaptureDestinationGPUTraceDocument = 2,
};

class CaptureDescriptor : public NS::Copying<CaptureDescriptor>
{
public:
    static CaptureDescriptor* alloc();

    NS::Object*               captureObject() const;

    CaptureDestination        destination() const;

    CaptureDescriptor*        init();

    NS::URL*                  outputURL() const;

    void                      setCaptureObject(NS::Object* captureObject);

    void                      setDestination(MTL::CaptureDestination destination);

    void                      setOutputURL(const NS::URL* outputURL);
};
class CaptureManager : public NS::Referencing<CaptureManager>
{
public:
    static CaptureManager* alloc();

    CaptureScope*          defaultCaptureScope() const;

    CaptureManager*        init();

    bool                   isCapturing() const;

    CaptureScope*          newCaptureScope(const MTL::Device* device);
    CaptureScope*          newCaptureScope(const MTL::CommandQueue* commandQueue);
    CaptureScope*          newCaptureScope(const MTL4::CommandQueue* commandQueue);

    void                   setDefaultCaptureScope(const MTL::CaptureScope* defaultCaptureScope);

    static CaptureManager* sharedCaptureManager();

    bool                   startCapture(const MTL::CaptureDescriptor* descriptor, NS::Error** error);
    void                   startCapture(const MTL::Device* device);
    void                   startCapture(const MTL::CommandQueue* commandQueue);
    void                   startCapture(const MTL::CaptureScope* captureScope);

    void                   stopCapture();

    bool                   supportsDestination(MTL::CaptureDestination destination);
};

}
_MTL_INLINE MTL::CaptureDescriptor* MTL::CaptureDescriptor::alloc()
{
    return NS::Object::alloc<MTL::CaptureDescriptor>(_MTL_PRIVATE_CLS(MTLCaptureDescriptor));
}

_MTL_INLINE NS::Object* MTL::CaptureDescriptor::captureObject() const
{
    return Object::sendMessage<NS::Object*>(this, _MTL_PRIVATE_SEL(captureObject));
}

_MTL_INLINE MTL::CaptureDestination MTL::CaptureDescriptor::destination() const
{
    return Object::sendMessage<MTL::CaptureDestination>(this, _MTL_PRIVATE_SEL(destination));
}

_MTL_INLINE MTL::CaptureDescriptor* MTL::CaptureDescriptor::init()
{
    return NS::Object::init<MTL::CaptureDescriptor>();
}

_MTL_INLINE NS::URL* MTL::CaptureDescriptor::outputURL() const
{
    return Object::sendMessage<NS::URL*>(this, _MTL_PRIVATE_SEL(outputURL));
}

_MTL_INLINE void MTL::CaptureDescriptor::setCaptureObject(NS::Object* captureObject)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setCaptureObject_), captureObject);
}

_MTL_INLINE void MTL::CaptureDescriptor::setDestination(MTL::CaptureDestination destination)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDestination_), destination);
}

_MTL_INLINE void MTL::CaptureDescriptor::setOutputURL(const NS::URL* outputURL)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setOutputURL_), outputURL);
}

_MTL_INLINE MTL::CaptureManager* MTL::CaptureManager::alloc()
{
    return NS::Object::alloc<MTL::CaptureManager>(_MTL_PRIVATE_CLS(MTLCaptureManager));
}

_MTL_INLINE MTL::CaptureScope* MTL::CaptureManager::defaultCaptureScope() const
{
    return Object::sendMessage<MTL::CaptureScope*>(this, _MTL_PRIVATE_SEL(defaultCaptureScope));
}

_MTL_INLINE MTL::CaptureManager* MTL::CaptureManager::init()
{
    return NS::Object::init<MTL::CaptureManager>();
}

_MTL_INLINE bool MTL::CaptureManager::isCapturing() const
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(isCapturing));
}

_MTL_INLINE MTL::CaptureScope* MTL::CaptureManager::newCaptureScope(const MTL::Device* device)
{
    return Object::sendMessage<MTL::CaptureScope*>(this, _MTL_PRIVATE_SEL(newCaptureScopeWithDevice_), device);
}

_MTL_INLINE MTL::CaptureScope* MTL::CaptureManager::newCaptureScope(const MTL::CommandQueue* commandQueue)
{
    return Object::sendMessage<MTL::CaptureScope*>(this, _MTL_PRIVATE_SEL(newCaptureScopeWithCommandQueue_), commandQueue);
}

_MTL_INLINE MTL::CaptureScope* MTL::CaptureManager::newCaptureScope(const MTL4::CommandQueue* commandQueue)
{
    return Object::sendMessage<MTL::CaptureScope*>(this, _MTL_PRIVATE_SEL(newCaptureScopeWithMTL4CommandQueue_), commandQueue);
}

_MTL_INLINE void MTL::CaptureManager::setDefaultCaptureScope(const MTL::CaptureScope* defaultCaptureScope)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setDefaultCaptureScope_), defaultCaptureScope);
}

_MTL_INLINE MTL::CaptureManager* MTL::CaptureManager::sharedCaptureManager()
{
    return Object::sendMessage<MTL::CaptureManager*>(_MTL_PRIVATE_CLS(MTLCaptureManager), _MTL_PRIVATE_SEL(sharedCaptureManager));
}

_MTL_INLINE bool MTL::CaptureManager::startCapture(const MTL::CaptureDescriptor* descriptor, NS::Error** error)
{
    return Object::sendMessage<bool>(this, _MTL_PRIVATE_SEL(startCaptureWithDescriptor_error_), descriptor, error);
}

_MTL_INLINE void MTL::CaptureManager::startCapture(const MTL::Device* device)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(startCaptureWithDevice_), device);
}

_MTL_INLINE void MTL::CaptureManager::startCapture(const MTL::CommandQueue* commandQueue)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(startCaptureWithCommandQueue_), commandQueue);
}

_MTL_INLINE void MTL::CaptureManager::startCapture(const MTL::CaptureScope* captureScope)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(startCaptureWithScope_), captureScope);
}

_MTL_INLINE void MTL::CaptureManager::stopCapture()
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(stopCapture));
}

_MTL_INLINE bool MTL::CaptureManager::supportsDestination(MTL::CaptureDestination destination)
{
    return Object::sendMessageSafe<bool>(this, _MTL_PRIVATE_SEL(supportsDestination_), destination);
}
