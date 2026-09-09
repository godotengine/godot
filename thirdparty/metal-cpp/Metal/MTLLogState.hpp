//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLLogState.hpp
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
class LogStateDescriptor;
_MTL_ENUM(NS::Integer, LogLevel) {
    LogLevelUndefined = 0,
    LogLevelDebug = 1,
    LogLevelInfo = 2,
    LogLevelNotice = 3,
    LogLevelError = 4,
    LogLevelFault = 5,
};

_MTL_ENUM(NS::UInteger, LogStateError) {
    LogStateErrorInvalidSize = 1,
    LogStateErrorInvalid = 2,
};

using LogHandlerFunction = std::function<void(NS::String* subsystem, NS::String* category, MTL::LogLevel logLevel, NS::String* message)>;

_MTL_CONST(NS::ErrorDomain, LogStateErrorDomain);
class LogState : public NS::Referencing<LogState>
{
public:
    void addLogHandler(void (^block)(NS::String*, NS::String*, MTL::LogLevel, NS::String*));
    void addLogHandler(const MTL::LogHandlerFunction& handler);
};
class LogStateDescriptor : public NS::Copying<LogStateDescriptor>
{
public:
    static LogStateDescriptor* alloc();

    NS::Integer                bufferSize() const;

    LogStateDescriptor*        init();

    LogLevel                   level() const;

    void                       setBufferSize(NS::Integer bufferSize);

    void                       setLevel(MTL::LogLevel level);
};

}
_MTL_PRIVATE_DEF_CONST(NS::ErrorDomain, LogStateErrorDomain);
_MTL_INLINE void MTL::LogState::addLogHandler(void (^block)(NS::String*, NS::String*, MTL::LogLevel, NS::String*))
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(addLogHandler_), block);
}

_MTL_INLINE void MTL::LogState::addLogHandler(const MTL::LogHandlerFunction& handler)
{
    __block LogHandlerFunction function = handler;
    addLogHandler(^void(NS::String* subsystem, NS::String* category, MTL::LogLevel logLevel, NS::String* message) { function(subsystem, category, logLevel, message); });
}

_MTL_INLINE MTL::LogStateDescriptor* MTL::LogStateDescriptor::alloc()
{
    return NS::Object::alloc<MTL::LogStateDescriptor>(_MTL_PRIVATE_CLS(MTLLogStateDescriptor));
}

_MTL_INLINE NS::Integer MTL::LogStateDescriptor::bufferSize() const
{
    return Object::sendMessage<NS::Integer>(this, _MTL_PRIVATE_SEL(bufferSize));
}

_MTL_INLINE MTL::LogStateDescriptor* MTL::LogStateDescriptor::init()
{
    return NS::Object::init<MTL::LogStateDescriptor>();
}

_MTL_INLINE MTL::LogLevel MTL::LogStateDescriptor::level() const
{
    return Object::sendMessage<MTL::LogLevel>(this, _MTL_PRIVATE_SEL(level));
}

_MTL_INLINE void MTL::LogStateDescriptor::setBufferSize(NS::Integer bufferSize)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setBufferSize_), bufferSize);
}

_MTL_INLINE void MTL::LogStateDescriptor::setLevel(MTL::LogLevel level)
{
    Object::sendMessage<void>(this, _MTL_PRIVATE_SEL(setLevel_), level);
}
