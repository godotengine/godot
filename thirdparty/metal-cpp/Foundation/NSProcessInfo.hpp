#pragma once

#include "NSDefines.hpp"
#include "NSBlocks.hpp"
#include "NSStructs.hpp"
#include "NSBridge.hpp"
#include "NSObject.hpp"
#include "NSTypes.hpp"
#include "NSRange.hpp"

namespace NS {
    class Array;
    class Dictionary;
    class Object;
    class String;
}

namespace NS
{

extern NS::NotificationName const ProcessInfoThermalStateDidChangeNotification __asm__("_NSProcessInfoThermalStateDidChangeNotification");
extern NS::NotificationName const ProcessInfoPowerStateDidChangeNotification __asm__("_NSProcessInfoPowerStateDidChangeNotification");
inline constexpr unsigned int WindowsNTOperatingSystem = 1;
inline constexpr unsigned int Windows95OperatingSystem = 2;
inline constexpr unsigned int SolarisOperatingSystem = 3;
inline constexpr unsigned int HPUXOperatingSystem = 4;
inline constexpr unsigned int MACHOperatingSystem = 5;
inline constexpr unsigned int SunOSOperatingSystem = 6;
inline constexpr unsigned int OSF1OperatingSystem = 7;

_NS_OPTIONS(uint64_t, ActivityOptions) {
    ActivityIdleDisplaySleepDisabled = (1ULL << 40),
    ActivityIdleSystemSleepDisabled = (1ULL << 20),
    ActivitySuddenTerminationDisabled = (1ULL << 14),
    ActivityAutomaticTerminationDisabled = (1ULL << 15),
    ActivityAnimationTrackingEnabled = (1ULL << 45),
    ActivityTrackingEnabled = (1ULL << 46),
    ActivityUserInitiated = (0x00FFFFFFULL | ActivityIdleSystemSleepDisabled),
    ActivityUserInitiatedAllowingIdleSystemSleep = (ActivityUserInitiated & ~ActivityIdleSystemSleepDisabled),
    ActivityBackground = 0x000000FFULL,
    ActivityLatencyCritical = 0xFF00000000ULL,
    ActivityUserInteractive = (ActivityUserInitiated | ActivityLatencyCritical),
};

_NS_ENUM(NS::Integer, ProcessInfoThermalState) {
    ProcessInfoThermalStateNominal = 0,
    ProcessInfoThermalStateFair = 1,
    ProcessInfoThermalStateSerious = 2,
    ProcessInfoThermalStateCritical = 3,
};


class ProcessInfo : public NS::Referencing<ProcessInfo>
{
public:
    static ProcessInfo* alloc();
    ProcessInfo*        init() const;

    static NS::ProcessInfo* processInfo();

    NS::UInteger                activeProcessorCount() const;
    NS::Array*                  arguments() const;
    bool                        automaticTerminationSupportEnabled() const;
    NS::Object*                 beginActivity(NS::ActivityOptions options, NS::String* reason);
    void                        disableAutomaticTermination(NS::String* reason);
    void                        disableSuddenTermination();
    void                        enableAutomaticTermination(NS::String* reason);
    void                        enableSuddenTermination();
    void                        endActivity(NS::Object* activity);
    NS::Dictionary*             environment() const;
    NS::String*                 fullUserName() const;
    NS::String*                 globallyUniqueString() const;
    NS::String*                 hostName() const;
    bool                        isLowPowerModeEnabled();
    bool                        isMacCatalystApp();
    bool                        isOperatingSystem(NS::OperatingSystemVersion version);
    bool                        isiOSAppOnMac();
    bool                        lowPowerModeEnabled() const;
    bool                        macCatalystApp() const;
    NS::UInteger                operatingSystem();
    NS::OperatingSystemVersion  operatingSystemVersion() const;
    NS::String*                 operatingSystemVersionString() const;
    void                        performActivity(NS::ActivityOptions options, NS::String* reason, NS::PerformActivityBlock block);
    void                        performActivity(NS::ActivityOptions options, NS::String* reason, const NS::PerformActivityFunction& block);
    void                        performExpiringActivity(NS::String* reason, NS::PerformExpiringActivityBlock block);
    void                        performExpiringActivity(NS::String* reason, const NS::PerformExpiringActivityFunction& block);
    unsigned long long          physicalMemory() const;
    int                         processIdentifier() const;
    NS::String*                 processName() const;
    NS::UInteger                processorCount() const;
    void                        setAutomaticTerminationSupportEnabled(bool automaticTerminationSupportEnabled);
    void                        setProcessName(NS::String* processName);
    NS::TimeInterval            systemUptime() const;
    NS::ProcessInfoThermalState thermalState() const;
    NS::String*                 userName() const;

};

} // namespace NS

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_NSProcessInfo;

_NS_INLINE NS::ProcessInfo* NS::ProcessInfo::alloc()
{
    return _NS_msg_NS__ProcessInfop_alloc((const void*)&OBJC_CLASS_$_NSProcessInfo, nullptr);
}

_NS_INLINE NS::ProcessInfo* NS::ProcessInfo::init() const
{
    return _NS_msg_NS__ProcessInfop_init((const void*)this, nullptr);
}

_NS_INLINE NS::ProcessInfo* NS::ProcessInfo::processInfo()
{
    return _NS_msg_NS__ProcessInfop_processInfo((const void*)&OBJC_CLASS_$_NSProcessInfo, nullptr);
}

_NS_INLINE NS::Dictionary* NS::ProcessInfo::environment() const
{
    return _NS_msg_NS__Dictionaryp_environment((const void*)this, nullptr);
}

_NS_INLINE NS::Array* NS::ProcessInfo::arguments() const
{
    return _NS_msg_NS__Arrayp_arguments((const void*)this, nullptr);
}

_NS_INLINE NS::String* NS::ProcessInfo::hostName() const
{
    return _NS_msg_NS__Stringp_hostName((const void*)this, nullptr);
}

_NS_INLINE NS::String* NS::ProcessInfo::processName() const
{
    return _NS_msg_NS__Stringp_processName((const void*)this, nullptr);
}

_NS_INLINE void NS::ProcessInfo::setProcessName(NS::String* processName)
{
    _NS_msg_v_setProcessName__NS__Stringp((const void*)this, nullptr, processName);
}

_NS_INLINE int NS::ProcessInfo::processIdentifier() const
{
    return _NS_msg_int_processIdentifier((const void*)this, nullptr);
}

_NS_INLINE NS::String* NS::ProcessInfo::globallyUniqueString() const
{
    return _NS_msg_NS__Stringp_globallyUniqueString((const void*)this, nullptr);
}

_NS_INLINE NS::String* NS::ProcessInfo::operatingSystemVersionString() const
{
    return _NS_msg_NS__Stringp_operatingSystemVersionString((const void*)this, nullptr);
}

_NS_INLINE NS::OperatingSystemVersion NS::ProcessInfo::operatingSystemVersion() const
{
    return _NS_msg_NS__OperatingSystemVersion_operatingSystemVersion((const void*)this, nullptr);
}

_NS_INLINE NS::UInteger NS::ProcessInfo::processorCount() const
{
    return _NS_msg_NS__UInteger_processorCount((const void*)this, nullptr);
}

_NS_INLINE NS::UInteger NS::ProcessInfo::activeProcessorCount() const
{
    return _NS_msg_NS__UInteger_activeProcessorCount((const void*)this, nullptr);
}

_NS_INLINE unsigned long long NS::ProcessInfo::physicalMemory() const
{
    return _NS_msg_unsignedlonglong_physicalMemory((const void*)this, nullptr);
}

_NS_INLINE NS::TimeInterval NS::ProcessInfo::systemUptime() const
{
    return _NS_msg_double_systemUptime((const void*)this, nullptr);
}

_NS_INLINE bool NS::ProcessInfo::automaticTerminationSupportEnabled() const
{
    return _NS_msg_bool_automaticTerminationSupportEnabled((const void*)this, nullptr);
}

_NS_INLINE void NS::ProcessInfo::setAutomaticTerminationSupportEnabled(bool automaticTerminationSupportEnabled)
{
    _NS_msg_v_setAutomaticTerminationSupportEnabled__bool((const void*)this, nullptr, automaticTerminationSupportEnabled);
}

_NS_INLINE NS::String* NS::ProcessInfo::userName() const
{
    return _NS_msg_NS__Stringp_userName((const void*)this, nullptr);
}

_NS_INLINE NS::String* NS::ProcessInfo::fullUserName() const
{
    return _NS_msg_NS__Stringp_fullUserName((const void*)this, nullptr);
}

_NS_INLINE NS::ProcessInfoThermalState NS::ProcessInfo::thermalState() const
{
    return _NS_msg_NS__ProcessInfoThermalState_thermalState((const void*)this, nullptr);
}

_NS_INLINE bool NS::ProcessInfo::lowPowerModeEnabled() const
{
    return _NS_msg_bool_lowPowerModeEnabled((const void*)this, nullptr);
}

_NS_INLINE bool NS::ProcessInfo::macCatalystApp() const
{
    return _NS_msg_bool_macCatalystApp((const void*)this, nullptr);
}

_NS_INLINE NS::UInteger NS::ProcessInfo::operatingSystem()
{
    return _NS_msg_NS__UInteger_operatingSystem((const void*)this, nullptr);
}

_NS_INLINE bool NS::ProcessInfo::isOperatingSystem(NS::OperatingSystemVersion version)
{
    return _NS_msg_bool_isOperatingSystemAtLeastVersion__NS__OperatingSystemVersion((const void*)this, nullptr, version);
}

_NS_INLINE void NS::ProcessInfo::disableSuddenTermination()
{
    _NS_msg_v_disableSuddenTermination((const void*)this, nullptr);
}

_NS_INLINE void NS::ProcessInfo::enableSuddenTermination()
{
    _NS_msg_v_enableSuddenTermination((const void*)this, nullptr);
}

_NS_INLINE void NS::ProcessInfo::disableAutomaticTermination(NS::String* reason)
{
    _NS_msg_v_disableAutomaticTermination__NS__Stringp((const void*)this, nullptr, reason);
}

_NS_INLINE void NS::ProcessInfo::enableAutomaticTermination(NS::String* reason)
{
    _NS_msg_v_enableAutomaticTermination__NS__Stringp((const void*)this, nullptr, reason);
}

_NS_INLINE NS::Object* NS::ProcessInfo::beginActivity(NS::ActivityOptions options, NS::String* reason)
{
    return _NS_msg_NS__Objectp_beginActivityWithOptions_reason__NS__ActivityOptions_NS__Stringp((const void*)this, nullptr, options, reason);
}

_NS_INLINE void NS::ProcessInfo::endActivity(NS::Object* activity)
{
    _NS_msg_v_endActivity__NS__Objectp((const void*)this, nullptr, activity);
}

_NS_INLINE void NS::ProcessInfo::performActivity(NS::ActivityOptions options, NS::String* reason, NS::PerformActivityBlock block)
{
    _NS_msg_v_performActivityWithOptions_reason_usingBlock__NS__ActivityOptions_NS__Stringp_NS__PerformActivityBlock((const void*)this, nullptr, options, reason, block);
}

_NS_INLINE void NS::ProcessInfo::performActivity(NS::ActivityOptions options, NS::String* reason, const NS::PerformActivityFunction& block)
{
    __block NS::PerformActivityFunction blockFunction = block;
    performActivity(options, reason, ^() { blockFunction(); });
}

_NS_INLINE void NS::ProcessInfo::performExpiringActivity(NS::String* reason, NS::PerformExpiringActivityBlock block)
{
    _NS_msg_v_performExpiringActivityWithReason_usingBlock__NS__Stringp_NS__PerformExpiringActivityBlock((const void*)this, nullptr, reason, block);
}

_NS_INLINE void NS::ProcessInfo::performExpiringActivity(NS::String* reason, const NS::PerformExpiringActivityFunction& block)
{
    __block NS::PerformExpiringActivityFunction blockFunction = block;
    performExpiringActivity(reason, ^(bool x0) { blockFunction(x0); });
}

_NS_INLINE bool NS::ProcessInfo::isLowPowerModeEnabled()
{
    return _NS_msg_bool_isLowPowerModeEnabled((const void*)this, nullptr);
}

_NS_INLINE bool NS::ProcessInfo::isMacCatalystApp()
{
    return _NS_msg_bool_isMacCatalystApp((const void*)this, nullptr);
}

_NS_INLINE bool NS::ProcessInfo::isiOSAppOnMac()
{
    return _NS_msg_bool_isiOSAppOnMac((const void*)this, nullptr);
}
