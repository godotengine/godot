#pragma once

#include "MTL4Defines.hpp"
#include "../Foundation/NSObjCRuntime.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

#include <functional>

namespace MTL4 {

class BinaryFunction;
class CommitFeedback;
class MachineLearningPipelineState;

} namespace NS {
class Error;
} namespace MTL4 {

using CommitFeedbackHandler = void (^)(MTL4::CommitFeedback*);
using CommitFeedbackHandlerFunction = std::function<void(MTL4::CommitFeedback*)>;

using NewBinaryFunctionCompletionHandler = void (^)(MTL4::BinaryFunction*, NS::Error*);
using NewBinaryFunctionCompletionHandlerFunction = std::function<void(MTL4::BinaryFunction*, NS::Error*)>;

using NewMachineLearningPipelineStateCompletionHandler = void (^)(MTL4::MachineLearningPipelineState*, NS::Error*);
using NewMachineLearningPipelineStateCompletionHandlerFunction = std::function<void(MTL4::MachineLearningPipelineState*, NS::Error*)>;

} // MTL4
