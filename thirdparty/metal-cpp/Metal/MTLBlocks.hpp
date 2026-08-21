#pragma once

#include "MTLDefines.hpp"
#include "../Foundation/NSObjCRuntime.hpp"
#include "../Foundation/NSTypes.hpp"
#include "../Foundation/NSRange.hpp"

#include <functional>

namespace MTL {

class CommandBuffer;
class ComputePipelineState;
class Device;
class Drawable;
class DynamicLibrary;
class Function;
class IOCommandBuffer;
class Library;
class RenderPipelineState;
class SharedEvent;
enum LogLevel : NS::Integer;

} namespace NS {
class Error;
class String;
} namespace MTL {

using CommandBufferHandler = void (^)(MTL::CommandBuffer*);
using CommandBufferHandlerFunction = std::function<void(MTL::CommandBuffer*)>;

using DeviceNotificationHandler = void (^)(MTL::Device*, NS::String*);
using DeviceNotificationHandlerFunction = std::function<void(MTL::Device*, NS::String*)>;

using NewBufferBlock = void (^)(void *, NS::UInteger);
using NewBufferFunction = std::function<void(void *, NS::UInteger)>;

using DrawablePresentedHandler = void (^)(MTL::Drawable*);
using DrawablePresentedHandlerFunction = std::function<void(MTL::Drawable*)>;

using SharedEventNotificationBlock = void (^)(MTL::SharedEvent*, uint64_t);
using SharedEventNotificationFunction = std::function<void(MTL::SharedEvent*, uint64_t)>;

using IOCommandBufferHandler = void (^)(MTL::IOCommandBuffer*);
using IOCommandBufferHandlerFunction = std::function<void(MTL::IOCommandBuffer*)>;

using NewLibraryCompletionHandler = void (^)(MTL::Library*, NS::Error*);
using NewLibraryCompletionHandlerFunction = std::function<void(MTL::Library*, NS::Error*)>;

using NewRenderPipelineStateCompletionHandler = void (^)(MTL::RenderPipelineState*, NS::Error*);
using NewRenderPipelineStateCompletionHandlerFunction = std::function<void(MTL::RenderPipelineState*, NS::Error*)>;

using NewRenderPipelineStateWithReflectionCompletionHandler = void (^)(MTL::RenderPipelineState*, void*, NS::Error*);
using NewRenderPipelineStateWithReflectionCompletionHandlerFunction = std::function<void(MTL::RenderPipelineState*, void*, NS::Error*)>;

using NewComputePipelineStateCompletionHandler = void (^)(MTL::ComputePipelineState*, NS::Error*);
using NewComputePipelineStateCompletionHandlerFunction = std::function<void(MTL::ComputePipelineState*, NS::Error*)>;

using NewComputePipelineStateWithReflectionCompletionHandler = void (^)(MTL::ComputePipelineState*, void*, NS::Error*);
using NewComputePipelineStateWithReflectionCompletionHandlerFunction = std::function<void(MTL::ComputePipelineState*, void*, NS::Error*)>;

using NewDynamicLibraryCompletionHandler = void (^)(MTL::DynamicLibrary*, NS::Error*);
using NewDynamicLibraryCompletionHandlerFunction = std::function<void(MTL::DynamicLibrary*, NS::Error*)>;

using NewFunctionBlock = void (^)(MTL::Function*, NS::Error*);
using NewFunctionFunction = std::function<void(MTL::Function*, NS::Error*)>;

using LogHandlerBlock = void (^)(NS::String*, NS::String*, MTL::LogLevel, NS::String*);
using LogHandlerFunction = std::function<void(NS::String*, NS::String*, MTL::LogLevel, NS::String*)>;

} // MTL
