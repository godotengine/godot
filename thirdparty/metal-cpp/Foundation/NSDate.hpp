#pragma once

#include "NSDefines.hpp"
#include "NSBlocks.hpp"
#include "NSStructs.hpp"
#include "NSBridge.hpp"
#include "NSObject.hpp"
#include "NSTypes.hpp"
#include "NSRange.hpp"

namespace NS
{

extern NS::NotificationName const SystemClockDidChangeNotification __asm__("_NSSystemClockDidChangeNotification");

class Date : public NS::SecureCoding<Date>
{
public:
    static Date* alloc();
    Date*        init() const;

    static NS::Date* date(NS::TimeInterval secs);

};

} // namespace NS

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_NSDate;

_NS_INLINE NS::Date* NS::Date::alloc()
{
    return _NS_msg_NS__Datep_alloc((const void*)&OBJC_CLASS_$_NSDate, nullptr);
}

_NS_INLINE NS::Date* NS::Date::init() const
{
    return _NS_msg_NS__Datep_init((const void*)this, nullptr);
}

_NS_INLINE NS::Date* NS::Date::date(NS::TimeInterval secs)
{
    return _NS_msg_NS__Datep_dateWithTimeIntervalSinceNow__double((const void*)&OBJC_CLASS_$_NSDate, nullptr, secs);
}
