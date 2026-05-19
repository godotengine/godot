#pragma once

#include "NSDefines.hpp"
#include "NSBlocks.hpp"
#include "NSStructs.hpp"
#include "NSBridge.hpp"
#include "NSObject.hpp"
#include "NSTypes.hpp"
#include "NSRange.hpp"

namespace NS {
    class Date;
}

namespace NS
{

class Condition : public NS::Referencing<Condition>
{
public:
    static Condition* alloc();
    Condition*        init() const;

    void broadcast();
    void signal();
    void wait();
    bool waitUntilDate(NS::Date* limit);

};

} // namespace NS

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_NSCondition;

_NS_INLINE NS::Condition* NS::Condition::alloc()
{
    return _NS_msg_NS__Conditionp_alloc((const void*)&OBJC_CLASS_$_NSCondition, nullptr);
}

_NS_INLINE NS::Condition* NS::Condition::init() const
{
    return _NS_msg_NS__Conditionp_init((const void*)this, nullptr);
}

_NS_INLINE void NS::Condition::wait()
{
    _NS_msg_v_wait((const void*)this, nullptr);
}

_NS_INLINE bool NS::Condition::waitUntilDate(NS::Date* limit)
{
    return _NS_msg_bool_waitUntilDate__NS__Datep((const void*)this, nullptr, limit);
}

_NS_INLINE void NS::Condition::signal()
{
    _NS_msg_v_signal((const void*)this, nullptr);
}

_NS_INLINE void NS::Condition::broadcast()
{
    _NS_msg_v_broadcast((const void*)this, nullptr);
}
