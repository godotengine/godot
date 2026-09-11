#pragma once

#include "NSDefines.hpp"
#include "NSObjCRuntime.hpp"
#include "NSTypes.hpp"
#include "NSRange.hpp"

#include <functional>

namespace NS {

class Error;
class Notification;
class Object;
class String;

using EnumerateObjectsBlock = void (^)(NS::Object*, NS::UInteger, bool*);
using EnumerateObjectsFunction = std::function<void(NS::Object*, NS::UInteger, bool*)>;

using IndexOfObjectPassingTestBlock = bool (^)(NS::Object*, NS::UInteger, bool*);
using IndexOfObjectPassingTestFunction = std::function<bool(NS::Object*, NS::UInteger, bool*)>;

using DifferenceFromArrayBlock = bool (^)(NS::Object*, NS::Object*);
using DifferenceFromArrayFunction = std::function<bool(NS::Object*, NS::Object*)>;

using BeginAccessingResourcesBlock = void (^)(NS::Error*);
using BeginAccessingResourcesFunction = std::function<void(NS::Error*)>;

using ConditionallyBeginAccessingResourcesBlock = void (^)(bool);
using ConditionallyBeginAccessingResourcesFunction = std::function<void(bool)>;

using EnumerateByteRangesBlock = void (^)(const void *, NS::Range, bool*);
using EnumerateByteRangesFunction = std::function<void(const void *, NS::Range, bool*)>;

using InitBlock = void (^)(void *, NS::UInteger);
using InitFunction = std::function<void(void *, NS::UInteger)>;

using EnumerateKeysAndObjectsBlock = void (^)(NS::Object*, NS::Object*, bool*);
using EnumerateKeysAndObjectsFunction = std::function<void(NS::Object*, NS::Object*, bool*)>;

using KeysOfEntriesPassingTestBlock = bool (^)(NS::Object*, NS::Object*, bool*);
using KeysOfEntriesPassingTestFunction = std::function<bool(NS::Object*, NS::Object*, bool*)>;

using UserInfoValueProviderBlock = NS::Object* (^)(NS::Error*, NS::String*);
using UserInfoValueProviderFunction = std::function<NS::Object*(NS::Error*, NS::String*)>;

using ObserverBlock = void (^)(NS::Notification*);
using ObserverFunction = std::function<void(NS::Notification*)>;

using PerformActivityBlock = void (^)();
using PerformActivityFunction = std::function<void()>;

using PerformExpiringActivityBlock = void (^)(bool);
using PerformExpiringActivityFunction = std::function<void(bool)>;

using ObjectsPassingTestBlock = bool (^)(NS::Object*, bool*);
using ObjectsPassingTestFunction = std::function<bool(NS::Object*, bool*)>;

using EnumerateSubstringsInRangeBlock = void (^)(NS::String*, NS::Range, NS::Range, bool*);
using EnumerateSubstringsInRangeFunction = std::function<void(NS::String*, NS::Range, NS::Range, bool*)>;

using EnumerateLinesBlock = void (^)(NS::String*, bool*);
using EnumerateLinesFunction = std::function<void(NS::String*, bool*)>;

using InitBlock2 = void (^)(void *, NS::UInteger);
using InitBlock2Function = std::function<void(void *, NS::UInteger)>;

} // NS
