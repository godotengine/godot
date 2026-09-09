//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTLFunctionLog.hpp
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
class Function;
class FunctionLogDebugLocation;
_MTL_ENUM(NS::UInteger, FunctionLogType) {
    FunctionLogTypeValidation = 0,
};

class LogContainer : public NS::Referencing<LogContainer, NS::FastEnumeration>
{
};
class FunctionLogDebugLocation : public NS::Referencing<FunctionLogDebugLocation>
{
public:
    NS::URL*     URL() const;

    NS::UInteger column() const;

    NS::String*  functionName() const;

    NS::UInteger line() const;
};
class FunctionLog : public NS::Referencing<FunctionLog>
{
public:
    FunctionLogDebugLocation* debugLocation() const;

    NS::String*               encoderLabel() const;

    Function*                 function() const;

    FunctionLogType           type() const;
};

}
_MTL_INLINE NS::URL* MTL::FunctionLogDebugLocation::URL() const
{
    return Object::sendMessage<NS::URL*>(this, _MTL_PRIVATE_SEL(URL));
}

_MTL_INLINE NS::UInteger MTL::FunctionLogDebugLocation::column() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(column));
}

_MTL_INLINE NS::String* MTL::FunctionLogDebugLocation::functionName() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(functionName));
}

_MTL_INLINE NS::UInteger MTL::FunctionLogDebugLocation::line() const
{
    return Object::sendMessage<NS::UInteger>(this, _MTL_PRIVATE_SEL(line));
}

_MTL_INLINE MTL::FunctionLogDebugLocation* MTL::FunctionLog::debugLocation() const
{
    return Object::sendMessage<MTL::FunctionLogDebugLocation*>(this, _MTL_PRIVATE_SEL(debugLocation));
}

_MTL_INLINE NS::String* MTL::FunctionLog::encoderLabel() const
{
    return Object::sendMessage<NS::String*>(this, _MTL_PRIVATE_SEL(encoderLabel));
}

_MTL_INLINE MTL::Function* MTL::FunctionLog::function() const
{
    return Object::sendMessage<MTL::Function*>(this, _MTL_PRIVATE_SEL(function));
}

_MTL_INLINE MTL::FunctionLogType MTL::FunctionLog::type() const
{
    return Object::sendMessage<MTL::FunctionLogType>(this, _MTL_PRIVATE_SEL(type));
}
