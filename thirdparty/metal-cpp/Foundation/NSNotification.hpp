#pragma once

#include "NSDefines.hpp"
#include "NSBlocks.hpp"
#include "NSStructs.hpp"
#include "NSBridge.hpp"
#include "NSObject.hpp"
#include "NSTypes.hpp"
#include "NSRange.hpp"

namespace NS {
    class Dictionary;
    class Object;
}

namespace NS
{

using NotificationName = NS::String*;

class Notification;
class NotificationCenter;

class Notification : public NS::Copying<Notification>
{
public:
    static Notification* alloc();
    Notification*        init() const;

    NS::NotificationName name() const;
    NS::Object*          object() const;
    NS::Dictionary*      userInfo() const;

};

class NotificationCenter : public NS::Referencing<NotificationCenter>
{
public:
    static NotificationCenter* alloc();
    NotificationCenter*        init() const;

    static NS::NotificationCenter* defaultCenter();

    void removeObserver(NS::Object* observer);

};

} // namespace NS

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_NSNotification;
extern "C" void *OBJC_CLASS_$_NSNotificationCenter;

_NS_INLINE NS::Notification* NS::Notification::alloc()
{
    return _NS_msg_NS__Notificationp_alloc((const void*)&OBJC_CLASS_$_NSNotification, nullptr);
}

_NS_INLINE NS::Notification* NS::Notification::init() const
{
    return _NS_msg_NS__Notificationp_init((const void*)this, nullptr);
}

_NS_INLINE NS::NotificationName NS::Notification::name() const
{
    return _NS_msg_NS__Stringp_name((const void*)this, nullptr);
}

_NS_INLINE NS::Object* NS::Notification::object() const
{
    return _NS_msg_NS__Objectp_object((const void*)this, nullptr);
}

_NS_INLINE NS::Dictionary* NS::Notification::userInfo() const
{
    return _NS_msg_NS__Dictionaryp_userInfo((const void*)this, nullptr);
}

_NS_INLINE NS::NotificationCenter* NS::NotificationCenter::alloc()
{
    return _NS_msg_NS__NotificationCenterp_alloc((const void*)&OBJC_CLASS_$_NSNotificationCenter, nullptr);
}

_NS_INLINE NS::NotificationCenter* NS::NotificationCenter::init() const
{
    return _NS_msg_NS__NotificationCenterp_init((const void*)this, nullptr);
}

_NS_INLINE NS::NotificationCenter* NS::NotificationCenter::defaultCenter()
{
    return _NS_msg_NS__NotificationCenterp_defaultCenter((const void*)&OBJC_CLASS_$_NSNotificationCenter, nullptr);
}

_NS_INLINE void NS::NotificationCenter::removeObserver(NS::Object* observer)
{
    _NS_msg_v_removeObserver__NS__Objectp((const void*)this, nullptr, observer);
}
