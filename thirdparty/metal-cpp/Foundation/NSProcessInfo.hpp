//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Foundation/NSProcessInfo.hpp
//
// Copyright 2020-2024 Apple Inc.
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

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#include "NSDefines.hpp"
#include "NSNotification.hpp"
#include "NSObject.hpp"
#include "NSPrivate.hpp"
#include "NSTypes.hpp"

#include <functional>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace NS
{
_NS_CONST(NotificationName, ProcessInfoThermalStateDidChangeNotification);
_NS_CONST(NotificationName, ProcessInfoPowerStateDidChangeNotification);
_NS_CONST(NotificationName, ProcessInfoPerformanceProfileDidChangeNotification);

_NS_ENUM(NS::Integer, ProcessInfoThermalState) {
    ProcessInfoThermalStateNominal = 0,
    ProcessInfoThermalStateFair = 1,
    ProcessInfoThermalStateSerious = 2,
    ProcessInfoThermalStateCritical = 3
};

_NS_OPTIONS(std::uint64_t, ActivityOptions) {
    ActivityIdleDisplaySleepDisabled = (1ULL << 40),
    ActivityIdleSystemSleepDisabled = (1ULL << 20),
    ActivitySuddenTerminationDisabled = (1ULL << 14),
    ActivityAutomaticTerminationDisabled = (1ULL << 15),
    ActivityUserInitiated = (0x00FFFFFFULL | ActivityIdleSystemSleepDisabled),
    ActivityUserInitiatedAllowingIdleSystemSleep = (ActivityUserInitiated & ~ActivityIdleSystemSleepDisabled),
    ActivityBackground = 0x000000FFULL,
    ActivityLatencyCritical = 0xFF00000000ULL,
};

typedef NS::Integer DeviceCertification;
_NS_CONST(DeviceCertification, DeviceCertificationiPhonePerformanceGaming);

typedef NS::Integer ProcessPerformanceProfile;
_NS_CONST(ProcessPerformanceProfile, ProcessPerformanceProfileDefault);
_NS_CONST(ProcessPerformanceProfile, ProcessPerformanceProfileSustained);

class ProcessInfo : public Referencing<ProcessInfo>
{
public:
    static ProcessInfo*     processInfo();

    class Array*            arguments() const;
    class Dictionary*       environment() const;
    class String*           hostName() const;
    class String*           processName() const;
    void                    setProcessName(const String* pString);
    int                     processIdentifier() const;
    class String*           globallyUniqueString() const;

    class String*           userName() const;
    class String*           fullUserName() const;

    UInteger                operatingSystem() const;
    OperatingSystemVersion  operatingSystemVersion() const;
    class String*           operatingSystemVersionString() const;
    bool                    isOperatingSystemAtLeastVersion(OperatingSystemVersion version) const;

    UInteger                processorCount() const;
    UInteger                activeProcessorCount() const;
    unsigned long long      physicalMemory() const;
    TimeInterval            systemUptime() const;

    void                    disableSuddenTermination();
    void                    enableSuddenTermination();

    void                    disableAutomaticTermination(const class String* pReason);
    void                    enableAutomaticTermination(const class String* pReason);
    bool                    automaticTerminationSupportEnabled() const;
    void                    setAutomaticTerminationSupportEnabled(bool enabled);

    class Object*           beginActivity(ActivityOptions options, const class String* pReason);
    void                    endActivity(class Object* pActivity);
    void                    performActivity(ActivityOptions options, const class String* pReason, void (^block)(void));
    void                    performActivity(ActivityOptions options, const class String* pReason, const std::function<void()>& func);
    void                    performExpiringActivity(const class String* pReason, void (^block)(bool expired));
    void                    performExpiringActivity(const class String* pReason, const std::function<void(bool expired)>& func);

    ProcessInfoThermalState thermalState() const;
    bool                    isLowPowerModeEnabled() const;

    bool                    isiOSAppOnMac() const;
    bool                    isMacCatalystApp() const;

    bool                    isDeviceCertified(DeviceCertification performanceTier) const;
    bool                    hasPerformanceProfile(ProcessPerformanceProfile performanceProfile) const;

};
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_PRIVATE_DEF_CONST(NS::NotificationName, ProcessInfoThermalStateDidChangeNotification);
_NS_PRIVATE_DEF_CONST(NS::NotificationName, ProcessInfoPowerStateDidChangeNotification);

// The linker searches for these symbols in the Metal framework, be sure to link it in as well:
_NS_PRIVATE_DEF_CONST(NS::NotificationName, ProcessInfoPerformanceProfileDidChangeNotification);
_NS_PRIVATE_DEF_CONST(NS::DeviceCertification, DeviceCertificationiPhonePerformanceGaming);
_NS_PRIVATE_DEF_CONST(NS::ProcessPerformanceProfile, ProcessPerformanceProfileDefault);
_NS_PRIVATE_DEF_CONST(NS::ProcessPerformanceProfile, ProcessPerformanceProfileSustained);

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::ProcessInfo* NS::ProcessInfo::processInfo()
{
    return Object::sendMessage<ProcessInfo*>(_NS_PRIVATE_CLS(NSProcessInfo), _NS_PRIVATE_SEL(processInfo));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Array* NS::ProcessInfo::arguments() const
{
    return Object::sendMessage<Array*>(this, _NS_PRIVATE_SEL(arguments));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Dictionary* NS::ProcessInfo::environment() const
{
    return Object::sendMessage<Dictionary*>(this, _NS_PRIVATE_SEL(environment));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::String* NS::ProcessInfo::hostName() const
{
    return Object::sendMessage<String*>(this, _NS_PRIVATE_SEL(hostName));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::String* NS::ProcessInfo::processName() const
{
    return Object::sendMessage<String*>(this, _NS_PRIVATE_SEL(processName));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE void NS::ProcessInfo::setProcessName(const String* pString)
{
    Object::sendMessage<void>(this, _NS_PRIVATE_SEL(setProcessName_), pString);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE int NS::ProcessInfo::processIdentifier() const
{
    return Object::sendMessage<int>(this, _NS_PRIVATE_SEL(processIdentifier));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::String* NS::ProcessInfo::globallyUniqueString() const
{
    return Object::sendMessage<String*>(this, _NS_PRIVATE_SEL(globallyUniqueString));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::String* NS::ProcessInfo::userName() const
{
    return Object::sendMessageSafe<String*>(this, _NS_PRIVATE_SEL(userName));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::String* NS::ProcessInfo::fullUserName() const
{
    return Object::sendMessageSafe<String*>(this, _NS_PRIVATE_SEL(fullUserName));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::UInteger NS::ProcessInfo::operatingSystem() const
{
    return Object::sendMessage<UInteger>(this, _NS_PRIVATE_SEL(operatingSystem));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::OperatingSystemVersion NS::ProcessInfo::operatingSystemVersion() const
{
    return Object::sendMessage<OperatingSystemVersion>(this, _NS_PRIVATE_SEL(operatingSystemVersion));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::String* NS::ProcessInfo::operatingSystemVersionString() const
{
    return Object::sendMessage<String*>(this, _NS_PRIVATE_SEL(operatingSystemVersionString));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE bool NS::ProcessInfo::isOperatingSystemAtLeastVersion(OperatingSystemVersion version) const
{
    return Object::sendMessage<bool>(this, _NS_PRIVATE_SEL(isOperatingSystemAtLeastVersion_), version);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::UInteger NS::ProcessInfo::processorCount() const
{
    return Object::sendMessage<UInteger>(this, _NS_PRIVATE_SEL(processorCount));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::UInteger NS::ProcessInfo::activeProcessorCount() const
{
    return Object::sendMessage<UInteger>(this, _NS_PRIVATE_SEL(activeProcessorCount));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE unsigned long long NS::ProcessInfo::physicalMemory() const
{
    return Object::sendMessage<unsigned long long>(this, _NS_PRIVATE_SEL(physicalMemory));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::TimeInterval NS::ProcessInfo::systemUptime() const
{
    return Object::sendMessage<TimeInterval>(this, _NS_PRIVATE_SEL(systemUptime));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE void NS::ProcessInfo::disableSuddenTermination()
{
    Object::sendMessageSafe<void>(this, _NS_PRIVATE_SEL(disableSuddenTermination));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE void NS::ProcessInfo::enableSuddenTermination()
{
    Object::sendMessageSafe<void>(this, _NS_PRIVATE_SEL(enableSuddenTermination));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE void NS::ProcessInfo::disableAutomaticTermination(const String* pReason)
{
    Object::sendMessageSafe<void>(this, _NS_PRIVATE_SEL(disableAutomaticTermination_), pReason);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE void NS::ProcessInfo::enableAutomaticTermination(const String* pReason)
{
    Object::sendMessageSafe<void>(this, _NS_PRIVATE_SEL(enableAutomaticTermination_), pReason);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE bool NS::ProcessInfo::automaticTerminationSupportEnabled() const
{
    return Object::sendMessageSafe<bool>(this, _NS_PRIVATE_SEL(automaticTerminationSupportEnabled));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE void NS::ProcessInfo::setAutomaticTerminationSupportEnabled(bool enabled)
{
    Object::sendMessageSafe<void>(this, _NS_PRIVATE_SEL(setAutomaticTerminationSupportEnabled_), enabled);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Object* NS::ProcessInfo::beginActivity(ActivityOptions options, const String* pReason)
{
    return Object::sendMessage<Object*>(this, _NS_PRIVATE_SEL(beginActivityWithOptions_reason_), options, pReason);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE void NS::ProcessInfo::endActivity(Object* pActivity)
{
    Object::sendMessage<void>(this, _NS_PRIVATE_SEL(endActivity_), pActivity);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE void NS::ProcessInfo::performActivity(ActivityOptions options, const String* pReason, void (^block)(void))
{
    Object::sendMessage<void>(this, _NS_PRIVATE_SEL(performActivityWithOptions_reason_usingBlock_), options, pReason, block);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE void NS::ProcessInfo::performActivity(ActivityOptions options, const String* pReason, const std::function<void()>& function)
{
    __block std::function<void()> blockFunction = function;

    performActivity(options, pReason, ^() { blockFunction(); });
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE void NS::ProcessInfo::performExpiringActivity(const String* pReason, void (^block)(bool expired))
{
    Object::sendMessageSafe<void>(this, _NS_PRIVATE_SEL(performExpiringActivityWithReason_usingBlock_), pReason, block);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE void NS::ProcessInfo::performExpiringActivity(const String* pReason, const std::function<void(bool expired)>& function)
{
    __block std::function<void(bool expired)> blockFunction = function;

    performExpiringActivity(pReason, ^(bool expired) { blockFunction(expired); });
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::ProcessInfoThermalState NS::ProcessInfo::thermalState() const
{
    return Object::sendMessage<ProcessInfoThermalState>(this, _NS_PRIVATE_SEL(thermalState));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE bool NS::ProcessInfo::isLowPowerModeEnabled() const
{
    return Object::sendMessageSafe<bool>(this, _NS_PRIVATE_SEL(isLowPowerModeEnabled));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE bool NS::ProcessInfo::isiOSAppOnMac() const
{
    return Object::sendMessageSafe<bool>(this, _NS_PRIVATE_SEL(isiOSAppOnMac));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE bool NS::ProcessInfo::isMacCatalystApp() const
{
    return Object::sendMessageSafe<bool>(this, _NS_PRIVATE_SEL(isMacCatalystApp));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE bool NS::ProcessInfo::isDeviceCertified(DeviceCertification performanceTier) const
{
    return Object::sendMessageSafe<bool>(this, _NS_PRIVATE_SEL(isDeviceCertified_), performanceTier);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE bool NS::ProcessInfo::hasPerformanceProfile(ProcessPerformanceProfile performanceProfile) const
{
    return Object::sendMessageSafe<bool>(this, _NS_PRIVATE_SEL(hasPerformanceProfile_), performanceProfile);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------
