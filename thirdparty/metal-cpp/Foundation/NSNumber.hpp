//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Foundation/NSNumber.hpp
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

#include "NSObjCRuntime.hpp"
#include "NSObject.hpp"
#include "NSTypes.hpp"

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace NS
{
class Value : public Copying<Value>
{
public:
    static Value* value(const void* pValue, const char* pType);
    static Value* value(const void* pPointer);

    static Value* alloc();

    Value*        init(const void* pValue, const char* pType);
    Value*        init(const class Coder* pCoder);

    void          getValue(void* pValue, UInteger size) const;
    const char*   objCType() const;

    bool          isEqualToValue(Value* pValue) const;
    void*         pointerValue() const;
};

class Number : public Copying<Number, Value>
{
public:
    static Number*     number(char value);
    static Number*     number(unsigned char value);
    static Number*     number(short value);
    static Number*     number(unsigned short value);
    static Number*     number(int value);
    static Number*     number(unsigned int value);
    static Number*     number(long value);
    static Number*     number(unsigned long value);
    static Number*     number(long long value);
    static Number*     number(unsigned long long value);
    static Number*     number(float value);
    static Number*     number(double value);
    static Number*     number(bool value);

    static Number*     alloc();

    Number*            init(const class Coder* pCoder);
    Number*            init(char value);
    Number*            init(unsigned char value);
    Number*            init(short value);
    Number*            init(unsigned short value);
    Number*            init(int value);
    Number*            init(unsigned int value);
    Number*            init(long value);
    Number*            init(unsigned long value);
    Number*            init(long long value);
    Number*            init(unsigned long long value);
    Number*            init(float value);
    Number*            init(double value);
    Number*            init(bool value);

    char               charValue() const;
    unsigned char      unsignedCharValue() const;
    short              shortValue() const;
    unsigned short     unsignedShortValue() const;
    int                intValue() const;
    unsigned int       unsignedIntValue() const;
    long               longValue() const;
    unsigned long      unsignedLongValue() const;
    long long          longLongValue() const;
    unsigned long long unsignedLongLongValue() const;
    float              floatValue() const;
    double             doubleValue() const;
    bool               boolValue() const;
    Integer            integerValue() const;
    UInteger           unsignedIntegerValue() const;
    class String*      stringValue() const;

    ComparisonResult   compare(const Number* pOtherNumber) const;
    bool               isEqualToNumber(const Number* pNumber) const;

    class String*      descriptionWithLocale(const Object* pLocale) const;
};
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Value* NS::Value::value(const void* pValue, const char* pType)
{
    return Object::sendMessage<Value*>(_NS_PRIVATE_CLS(NSValue), _NS_PRIVATE_SEL(valueWithBytes_objCType_), pValue, pType);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Value* NS::Value::value(const void* pPointer)
{
    return Object::sendMessage<Value*>(_NS_PRIVATE_CLS(NSValue), _NS_PRIVATE_SEL(valueWithPointer_), pPointer);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Value* NS::Value::alloc()
{
    return NS::Object::alloc<Value>(_NS_PRIVATE_CLS(NSValue));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Value* NS::Value::init(const void* pValue, const char* pType)
{
    return Object::sendMessage<Value*>(this, _NS_PRIVATE_SEL(initWithBytes_objCType_), pValue, pType);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Value* NS::Value::init(const class Coder* pCoder)
{
    return Object::sendMessage<Value*>(this, _NS_PRIVATE_SEL(initWithCoder_), pCoder);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE void NS::Value::getValue(void* pValue, UInteger size) const
{
    Object::sendMessage<void>(this, _NS_PRIVATE_SEL(getValue_size_), pValue, size);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE const char* NS::Value::objCType() const
{
    return Object::sendMessage<const char*>(this, _NS_PRIVATE_SEL(objCType));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE bool NS::Value::isEqualToValue(Value* pValue) const
{
    return Object::sendMessage<bool>(this, _NS_PRIVATE_SEL(isEqualToValue_), pValue);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE void* NS::Value::pointerValue() const
{
    return Object::sendMessage<void*>(this, _NS_PRIVATE_SEL(pointerValue));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Number* NS::Number::number(char value)
{
    return Object::sendMessage<Number*>(_NS_PRIVATE_CLS(NSNumber), _NS_PRIVATE_SEL(numberWithChar_), value);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Number* NS::Number::number(unsigned char value)
{
    return Object::sendMessage<Number*>(_NS_PRIVATE_CLS(NSNumber), _NS_PRIVATE_SEL(numberWithUnsignedChar_), value);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Number* NS::Number::number(short value)
{
    return Object::sendMessage<Number*>(_NS_PRIVATE_CLS(NSNumber), _NS_PRIVATE_SEL(numberWithShort_), value);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Number* NS::Number::number(unsigned short value)
{
    return Object::sendMessage<Number*>(_NS_PRIVATE_CLS(NSNumber), _NS_PRIVATE_SEL(numberWithUnsignedShort_), value);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Number* NS::Number::number(int value)
{
    return Object::sendMessage<Number*>(_NS_PRIVATE_CLS(NSNumber), _NS_PRIVATE_SEL(numberWithInt_), value);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Number* NS::Number::number(unsigned int value)
{
    return Object::sendMessage<Number*>(_NS_PRIVATE_CLS(NSNumber), _NS_PRIVATE_SEL(numberWithUnsignedInt_), value);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Number* NS::Number::number(long value)
{
    return Object::sendMessage<Number*>(_NS_PRIVATE_CLS(NSNumber), _NS_PRIVATE_SEL(numberWithLong_), value);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Number* NS::Number::number(unsigned long value)
{
    return Object::sendMessage<Number*>(_NS_PRIVATE_CLS(NSNumber), _NS_PRIVATE_SEL(numberWithUnsignedLong_), value);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Number* NS::Number::number(long long value)
{
    return Object::sendMessage<Number*>(_NS_PRIVATE_CLS(NSNumber), _NS_PRIVATE_SEL(numberWithLongLong_), value);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Number* NS::Number::number(unsigned long long value)
{
    return Object::sendMessage<Number*>(_NS_PRIVATE_CLS(NSNumber), _NS_PRIVATE_SEL(numberWithUnsignedLongLong_), value);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Number* NS::Number::number(float value)
{
    return Object::sendMessage<Number*>(_NS_PRIVATE_CLS(NSNumber), _NS_PRIVATE_SEL(numberWithFloat_), value);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Number* NS::Number::number(double value)
{
    return Object::sendMessage<Number*>(_NS_PRIVATE_CLS(NSNumber), _NS_PRIVATE_SEL(numberWithDouble_), value);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Number* NS::Number::number(bool value)
{
    return Object::sendMessage<Number*>(_NS_PRIVATE_CLS(NSNumber), _NS_PRIVATE_SEL(numberWithBool_), value);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Number* NS::Number::alloc()
{
    return NS::Object::alloc<Number>(_NS_PRIVATE_CLS(NSNumber));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Number* NS::Number::init(const Coder* pCoder)
{
    return Object::sendMessage<Number*>(this, _NS_PRIVATE_SEL(initWithCoder_), pCoder);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Number* NS::Number::init(char value)
{
    return Object::sendMessage<Number*>(this, _NS_PRIVATE_SEL(initWithChar_), value);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Number* NS::Number::init(unsigned char value)
{
    return Object::sendMessage<Number*>(this, _NS_PRIVATE_SEL(initWithUnsignedChar_), value);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Number* NS::Number::init(short value)
{
    return Object::sendMessage<Number*>(this, _NS_PRIVATE_SEL(initWithShort_), value);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Number* NS::Number::init(unsigned short value)
{
    return Object::sendMessage<Number*>(this, _NS_PRIVATE_SEL(initWithUnsignedShort_), value);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Number* NS::Number::init(int value)
{
    return Object::sendMessage<Number*>(this, _NS_PRIVATE_SEL(initWithInt_), value);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Number* NS::Number::init(unsigned int value)
{
    return Object::sendMessage<Number*>(this, _NS_PRIVATE_SEL(initWithUnsignedInt_), value);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Number* NS::Number::init(long value)
{
    return Object::sendMessage<Number*>(this, _NS_PRIVATE_SEL(initWithLong_), value);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Number* NS::Number::init(unsigned long value)
{
    return Object::sendMessage<Number*>(this, _NS_PRIVATE_SEL(initWithUnsignedLong_), value);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Number* NS::Number::init(long long value)
{
    return Object::sendMessage<Number*>(this, _NS_PRIVATE_SEL(initWithLongLong_), value);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Number* NS::Number::init(unsigned long long value)
{
    return Object::sendMessage<Number*>(this, _NS_PRIVATE_SEL(initWithUnsignedLongLong_), value);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Number* NS::Number::init(float value)
{
    return Object::sendMessage<Number*>(this, _NS_PRIVATE_SEL(initWithFloat_), value);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Number* NS::Number::init(double value)
{
    return Object::sendMessage<Number*>(this, _NS_PRIVATE_SEL(initWithDouble_), value);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Number* NS::Number::init(bool value)
{
    return Object::sendMessage<Number*>(this, _NS_PRIVATE_SEL(initWithBool_), value);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE char NS::Number::charValue() const
{
    return Object::sendMessage<char>(this, _NS_PRIVATE_SEL(charValue));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE unsigned char NS::Number::unsignedCharValue() const
{
    return Object::sendMessage<unsigned char>(this, _NS_PRIVATE_SEL(unsignedCharValue));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE short NS::Number::shortValue() const
{
    return Object::sendMessage<short>(this, _NS_PRIVATE_SEL(shortValue));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE unsigned short NS::Number::unsignedShortValue() const
{
    return Object::sendMessage<unsigned short>(this, _NS_PRIVATE_SEL(unsignedShortValue));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE int NS::Number::intValue() const
{
    return Object::sendMessage<int>(this, _NS_PRIVATE_SEL(intValue));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE unsigned int NS::Number::unsignedIntValue() const
{
    return Object::sendMessage<unsigned int>(this, _NS_PRIVATE_SEL(unsignedIntValue));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE long NS::Number::longValue() const
{
    return Object::sendMessage<long>(this, _NS_PRIVATE_SEL(longValue));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE unsigned long NS::Number::unsignedLongValue() const
{
    return Object::sendMessage<unsigned long>(this, _NS_PRIVATE_SEL(unsignedLongValue));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE long long NS::Number::longLongValue() const
{
    return Object::sendMessage<long long>(this, _NS_PRIVATE_SEL(longLongValue));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE unsigned long long NS::Number::unsignedLongLongValue() const
{
    return Object::sendMessage<unsigned long long>(this, _NS_PRIVATE_SEL(unsignedLongLongValue));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE float NS::Number::floatValue() const
{
    return Object::sendMessage<float>(this, _NS_PRIVATE_SEL(floatValue));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE double NS::Number::doubleValue() const
{
    return Object::sendMessage<double>(this, _NS_PRIVATE_SEL(doubleValue));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE bool NS::Number::boolValue() const
{
    return Object::sendMessage<bool>(this, _NS_PRIVATE_SEL(boolValue));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Integer NS::Number::integerValue() const
{
    return Object::sendMessage<Integer>(this, _NS_PRIVATE_SEL(integerValue));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::UInteger NS::Number::unsignedIntegerValue() const
{
    return Object::sendMessage<UInteger>(this, _NS_PRIVATE_SEL(unsignedIntegerValue));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::String* NS::Number::stringValue() const
{
    return Object::sendMessage<String*>(this, _NS_PRIVATE_SEL(stringValue));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::ComparisonResult NS::Number::compare(const Number* pOtherNumber) const
{
    return Object::sendMessage<ComparisonResult>(this, _NS_PRIVATE_SEL(compare_), pOtherNumber);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE bool NS::Number::isEqualToNumber(const Number* pNumber) const
{
    return Object::sendMessage<bool>(this, _NS_PRIVATE_SEL(isEqualToNumber_), pNumber);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::String* NS::Number::descriptionWithLocale(const Object* pLocale) const
{
    return Object::sendMessage<String*>(this, _NS_PRIVATE_SEL(descriptionWithLocale_), pLocale);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------
