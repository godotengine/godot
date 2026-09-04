#pragma once

#include "NSDefines.hpp"
#include "NSBlocks.hpp"
#include "NSStructs.hpp"
#include "NSBridge.hpp"
#include "NSObject.hpp"
#include "NSTypes.hpp"
#include "NSRange.hpp"

namespace NS {
    class Object;
    class String;
}

namespace NS
{

class Value;
class Number;

class Value : public NS::SecureCoding<Value>
{
public:
    static Value* alloc();
    Value*        init() const;

    static NS::Value* value(const void * value, const char * type);
    static NS::Value* value(const void * pointer);

    void         getValue(void * value, NS::UInteger size);
    NS::Value*   init(const void * value, const char * type);
    NS::Value*   init(void* coder);
    bool         isEqualToValue(NS::Value* value);
    const char * objCType() const;
    void *       pointerValue() const;

};

class Number : public NS::Referencing<Number, NS::Value>
{
public:
    static Number* alloc();
    Number*        init() const;

    static NS::Number* number(char value);
    static NS::Number* number(unsigned char value);
    static NS::Number* number(short value);
    static NS::Number* number(unsigned short value);
    static NS::Number* number(int value);
    static NS::Number* number(unsigned int value);
    static NS::Number* number(long value);
    static NS::Number* number(unsigned long value);
    static NS::Number* number(long long value);
    static NS::Number* number(unsigned long long value);
    static NS::Number* number(float value);
    static NS::Number* number(double value);
    static NS::Number* number(bool value);

    bool               boolValue() const;
    char               charValue() const;
    long               compare(NS::Number* otherNumber);
    NS::String*        description(NS::Object* locale);
    double             doubleValue() const;
    float              floatValue() const;
    NS::Number*        init(void* coder);
    NS::Number*        init(char value);
    NS::Number*        init(unsigned char value);
    NS::Number*        init(short value);
    NS::Number*        init(unsigned short value);
    NS::Number*        init(int value);
    NS::Number*        init(unsigned int value);
    NS::Number*        init(long value);
    NS::Number*        init(unsigned long value);
    NS::Number*        init(long long value);
    NS::Number*        init(unsigned long long value);
    NS::Number*        init(float value);
    NS::Number*        init(double value);
    NS::Number*        init(bool value);
    int                intValue() const;
    NS::Integer        integerValue() const;
    bool               isEqualToNumber(NS::Number* number);
    long long          longLongValue() const;
    long               longValue() const;
    short              shortValue() const;
    NS::String*        stringValue() const;
    unsigned char      unsignedCharValue() const;
    unsigned int       unsignedIntValue() const;
    NS::UInteger       unsignedIntegerValue() const;
    unsigned long long unsignedLongLongValue() const;
    unsigned long      unsignedLongValue() const;
    unsigned short     unsignedShortValue() const;

};

} // namespace NS

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_NSValue;
extern "C" void *OBJC_CLASS_$_NSNumber;

_NS_INLINE NS::Value* NS::Value::alloc()
{
    return _NS_msg_NS__Valuep_alloc((const void*)&OBJC_CLASS_$_NSValue, nullptr);
}

_NS_INLINE NS::Value* NS::Value::init() const
{
    return _NS_msg_NS__Valuep_init((const void*)this, nullptr);
}

_NS_INLINE NS::Value* NS::Value::value(const void * value, const char * type)
{
    return _NS_msg_NS__Valuep_valueWithBytes_objCType__constvoidp_constcharp((const void*)&OBJC_CLASS_$_NSValue, nullptr, value, type);
}

_NS_INLINE NS::Value* NS::Value::value(const void * pointer)
{
    return _NS_msg_NS__Valuep_valueWithPointer__constvoidp((const void*)&OBJC_CLASS_$_NSValue, nullptr, pointer);
}

_NS_INLINE const char * NS::Value::objCType() const
{
    return _NS_msg_constcharp_objCType((const void*)this, nullptr);
}

_NS_INLINE void * NS::Value::pointerValue() const
{
    return _NS_msg_voidp_pointerValue((const void*)this, nullptr);
}

_NS_INLINE void NS::Value::getValue(void * value, NS::UInteger size)
{
    _NS_msg_v_getValue_size__voidp_NS__UInteger((const void*)this, nullptr, value, size);
}

_NS_INLINE NS::Value* NS::Value::init(const void * value, const char * type)
{
    return _NS_msg_NS__Valuep_initWithBytes_objCType__constvoidp_constcharp((const void*)this, nullptr, value, type);
}

_NS_INLINE NS::Value* NS::Value::init(void* coder)
{
    return _NS_msg_NS__Valuep_initWithCoder__voidp((const void*)this, nullptr, coder);
}

_NS_INLINE bool NS::Value::isEqualToValue(NS::Value* value)
{
    return _NS_msg_bool_isEqualToValue__NS__Valuep((const void*)this, nullptr, value);
}

_NS_INLINE NS::Number* NS::Number::alloc()
{
    return _NS_msg_NS__Numberp_alloc((const void*)&OBJC_CLASS_$_NSNumber, nullptr);
}

_NS_INLINE NS::Number* NS::Number::init() const
{
    return _NS_msg_NS__Numberp_init((const void*)this, nullptr);
}

_NS_INLINE NS::Number* NS::Number::number(char value)
{
    return _NS_msg_NS__Numberp_numberWithChar__char((const void*)&OBJC_CLASS_$_NSNumber, nullptr, value);
}

_NS_INLINE NS::Number* NS::Number::number(unsigned char value)
{
    return _NS_msg_NS__Numberp_numberWithUnsignedChar__unsignedchar((const void*)&OBJC_CLASS_$_NSNumber, nullptr, value);
}

_NS_INLINE NS::Number* NS::Number::number(short value)
{
    return _NS_msg_NS__Numberp_numberWithShort__short((const void*)&OBJC_CLASS_$_NSNumber, nullptr, value);
}

_NS_INLINE NS::Number* NS::Number::number(unsigned short value)
{
    return _NS_msg_NS__Numberp_numberWithUnsignedShort__unsignedshort((const void*)&OBJC_CLASS_$_NSNumber, nullptr, value);
}

_NS_INLINE NS::Number* NS::Number::number(int value)
{
    return _NS_msg_NS__Numberp_numberWithInt__int((const void*)&OBJC_CLASS_$_NSNumber, nullptr, value);
}

_NS_INLINE NS::Number* NS::Number::number(unsigned int value)
{
    return _NS_msg_NS__Numberp_numberWithUnsignedInt__unsignedint((const void*)&OBJC_CLASS_$_NSNumber, nullptr, value);
}

_NS_INLINE NS::Number* NS::Number::number(long value)
{
    return _NS_msg_NS__Numberp_numberWithLong__long((const void*)&OBJC_CLASS_$_NSNumber, nullptr, value);
}

_NS_INLINE NS::Number* NS::Number::number(unsigned long value)
{
    return _NS_msg_NS__Numberp_numberWithUnsignedLong__unsignedlong((const void*)&OBJC_CLASS_$_NSNumber, nullptr, value);
}

_NS_INLINE NS::Number* NS::Number::number(long long value)
{
    return _NS_msg_NS__Numberp_numberWithLongLong__longlong((const void*)&OBJC_CLASS_$_NSNumber, nullptr, value);
}

_NS_INLINE NS::Number* NS::Number::number(unsigned long long value)
{
    return _NS_msg_NS__Numberp_numberWithUnsignedLongLong__unsignedlonglong((const void*)&OBJC_CLASS_$_NSNumber, nullptr, value);
}

_NS_INLINE NS::Number* NS::Number::number(float value)
{
    return _NS_msg_NS__Numberp_numberWithFloat__float((const void*)&OBJC_CLASS_$_NSNumber, nullptr, value);
}

_NS_INLINE NS::Number* NS::Number::number(double value)
{
    return _NS_msg_NS__Numberp_numberWithDouble__double((const void*)&OBJC_CLASS_$_NSNumber, nullptr, value);
}

_NS_INLINE NS::Number* NS::Number::number(bool value)
{
    return _NS_msg_NS__Numberp_numberWithBool__bool((const void*)&OBJC_CLASS_$_NSNumber, nullptr, value);
}

_NS_INLINE char NS::Number::charValue() const
{
    return _NS_msg_char_charValue((const void*)this, nullptr);
}

_NS_INLINE unsigned char NS::Number::unsignedCharValue() const
{
    return _NS_msg_unsignedchar_unsignedCharValue((const void*)this, nullptr);
}

_NS_INLINE short NS::Number::shortValue() const
{
    return _NS_msg_short_shortValue((const void*)this, nullptr);
}

_NS_INLINE unsigned short NS::Number::unsignedShortValue() const
{
    return _NS_msg_unsignedshort_unsignedShortValue((const void*)this, nullptr);
}

_NS_INLINE int NS::Number::intValue() const
{
    return _NS_msg_int_intValue((const void*)this, nullptr);
}

_NS_INLINE unsigned int NS::Number::unsignedIntValue() const
{
    return _NS_msg_unsignedint_unsignedIntValue((const void*)this, nullptr);
}

_NS_INLINE long NS::Number::longValue() const
{
    return _NS_msg_long_longValue((const void*)this, nullptr);
}

_NS_INLINE unsigned long NS::Number::unsignedLongValue() const
{
    return _NS_msg_unsignedlong_unsignedLongValue((const void*)this, nullptr);
}

_NS_INLINE long long NS::Number::longLongValue() const
{
    return _NS_msg_longlong_longLongValue((const void*)this, nullptr);
}

_NS_INLINE unsigned long long NS::Number::unsignedLongLongValue() const
{
    return _NS_msg_unsignedlonglong_unsignedLongLongValue((const void*)this, nullptr);
}

_NS_INLINE float NS::Number::floatValue() const
{
    return _NS_msg_float_floatValue((const void*)this, nullptr);
}

_NS_INLINE double NS::Number::doubleValue() const
{
    return _NS_msg_double_doubleValue((const void*)this, nullptr);
}

_NS_INLINE bool NS::Number::boolValue() const
{
    return _NS_msg_bool_boolValue((const void*)this, nullptr);
}

_NS_INLINE NS::Integer NS::Number::integerValue() const
{
    return _NS_msg_NS__Integer_integerValue((const void*)this, nullptr);
}

_NS_INLINE NS::UInteger NS::Number::unsignedIntegerValue() const
{
    return _NS_msg_NS__UInteger_unsignedIntegerValue((const void*)this, nullptr);
}

_NS_INLINE NS::String* NS::Number::stringValue() const
{
    return _NS_msg_NS__Stringp_stringValue((const void*)this, nullptr);
}

_NS_INLINE NS::Number* NS::Number::init(void* coder)
{
    return _NS_msg_NS__Numberp_initWithCoder__voidp((const void*)this, nullptr, coder);
}

_NS_INLINE NS::Number* NS::Number::init(char value)
{
    return _NS_msg_NS__Numberp_initWithChar__char((const void*)this, nullptr, value);
}

_NS_INLINE NS::Number* NS::Number::init(unsigned char value)
{
    return _NS_msg_NS__Numberp_initWithUnsignedChar__unsignedchar((const void*)this, nullptr, value);
}

_NS_INLINE NS::Number* NS::Number::init(short value)
{
    return _NS_msg_NS__Numberp_initWithShort__short((const void*)this, nullptr, value);
}

_NS_INLINE NS::Number* NS::Number::init(unsigned short value)
{
    return _NS_msg_NS__Numberp_initWithUnsignedShort__unsignedshort((const void*)this, nullptr, value);
}

_NS_INLINE NS::Number* NS::Number::init(int value)
{
    return _NS_msg_NS__Numberp_initWithInt__int((const void*)this, nullptr, value);
}

_NS_INLINE NS::Number* NS::Number::init(unsigned int value)
{
    return _NS_msg_NS__Numberp_initWithUnsignedInt__unsignedint((const void*)this, nullptr, value);
}

_NS_INLINE NS::Number* NS::Number::init(long value)
{
    return _NS_msg_NS__Numberp_initWithLong__long((const void*)this, nullptr, value);
}

_NS_INLINE NS::Number* NS::Number::init(unsigned long value)
{
    return _NS_msg_NS__Numberp_initWithUnsignedLong__unsignedlong((const void*)this, nullptr, value);
}

_NS_INLINE NS::Number* NS::Number::init(long long value)
{
    return _NS_msg_NS__Numberp_initWithLongLong__longlong((const void*)this, nullptr, value);
}

_NS_INLINE NS::Number* NS::Number::init(unsigned long long value)
{
    return _NS_msg_NS__Numberp_initWithUnsignedLongLong__unsignedlonglong((const void*)this, nullptr, value);
}

_NS_INLINE NS::Number* NS::Number::init(float value)
{
    return _NS_msg_NS__Numberp_initWithFloat__float((const void*)this, nullptr, value);
}

_NS_INLINE NS::Number* NS::Number::init(double value)
{
    return _NS_msg_NS__Numberp_initWithDouble__double((const void*)this, nullptr, value);
}

_NS_INLINE NS::Number* NS::Number::init(bool value)
{
    return _NS_msg_NS__Numberp_initWithBool__bool((const void*)this, nullptr, value);
}

_NS_INLINE long NS::Number::compare(NS::Number* otherNumber)
{
    return _NS_msg_long_compare__NS__Numberp((const void*)this, nullptr, otherNumber);
}

_NS_INLINE bool NS::Number::isEqualToNumber(NS::Number* number)
{
    return _NS_msg_bool_isEqualToNumber__NS__Numberp((const void*)this, nullptr, number);
}

_NS_INLINE NS::String* NS::Number::description(NS::Object* locale)
{
    return _NS_msg_NS__Stringp_descriptionWithLocale__NS__Objectp((const void*)this, nullptr, locale);
}
