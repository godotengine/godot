//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Metal/MTL4CommitFeedback.hpp
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
#include <CoreFoundation/CoreFoundation.h>

#include <functional>

namespace MTL4
{
class CommitFeedback;

using CommitFeedbackHandler = void (^)(MTL4::CommitFeedback*);
using CommitFeedbackHandlerFunction = std::function<void(MTL4::CommitFeedback*)>;

class CommitFeedback : public NS::Referencing<CommitFeedback>
{
public:
    CFTimeInterval GPUEndTime() const;

    CFTimeInterval GPUStartTime() const;

    NS::Error*     error() const;
};

}
_MTL_INLINE CFTimeInterval MTL4::CommitFeedback::GPUEndTime() const
{
    return Object::sendMessage<CFTimeInterval>(this, _MTL_PRIVATE_SEL(GPUEndTime));
}

_MTL_INLINE CFTimeInterval MTL4::CommitFeedback::GPUStartTime() const
{
    return Object::sendMessage<CFTimeInterval>(this, _MTL_PRIVATE_SEL(GPUStartTime));
}

_MTL_INLINE NS::Error* MTL4::CommitFeedback::error() const
{
    return Object::sendMessage<NS::Error*>(this, _MTL_PRIVATE_SEL(error));
}
