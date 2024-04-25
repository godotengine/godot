//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXTest/External/Catch/catch.hpp>

#include <MaterialXCore/Util.h>
#include <MaterialXCore/Value.h>

namespace mx = MaterialX;

template<class T> void testTypedValue(const T& v1, const T& v2)
{
    T v0{};

    // Constructor and assignment
    mx::ValuePtr value0 = mx::Value::createValue(v0);
    mx::ValuePtr value1 = mx::Value::createValue(v1);
    mx::ValuePtr value2 = mx::Value::createValue(v2);
    REQUIRE(value0->isA<T>());
    REQUIRE(value1->isA<T>());
    REQUIRE(value2->isA<T>());
    REQUIRE(value0->asA<T>() == v0);
    REQUIRE(value1->asA<T>() == v1);
    REQUIRE(value2->asA<T>() == v2);

    // Equality and inequality
    REQUIRE(value0->copy()->asA<T>() == value0->asA<T>());
    REQUIRE(value1->copy()->asA<T>() == value1->asA<T>());
    REQUIRE(value2->copy()->asA<T>() == value2->asA<T>());
    REQUIRE(value1->asA<T>() != value2->asA<T>());

    // Serialization and deserialization
    mx::ValuePtr newValue0 = mx::TypedValue<T>::createFromString(value0->getValueString());
    mx::ValuePtr newValue1 = mx::TypedValue<T>::createFromString(value1->getValueString());
    mx::ValuePtr newValue2 = mx::TypedValue<T>::createFromString(value2->getValueString());
    REQUIRE(newValue0->asA<T>() == v0);
    REQUIRE(newValue1->asA<T>() == v1);
    REQUIRE(newValue2->asA<T>() == v2);
}

TEST_CASE("Value strings", "[value]")
{
    // Convert from data values to value strings.
    REQUIRE(mx::toValueString(1) == "1");
    REQUIRE(mx::toValueString(1.0f) == "1");
    REQUIRE(mx::toValueString(true) == "true");
    REQUIRE(mx::toValueString(false) == "false");
    REQUIRE(mx::toValueString(mx::Color3(1.0f)) == "1, 1, 1");
    REQUIRE(mx::toValueString(std::string("text")) == "text");

    // Convert from floats to value strings with custom formatting.
    {
        mx::ScopedFloatFormatting fmt(mx::Value::FloatFormatFixed, 3);
        REQUIRE(mx::toValueString(0.1234f) == "0.123");
        REQUIRE(mx::toValueString(mx::Color3(1.0f)) == "1.000, 1.000, 1.000");
    }
    {
        mx::ScopedFloatFormatting fmt(mx::Value::FloatFormatScientific, 2);
        REQUIRE(mx::toValueString(0.1234f) == "1.23e-01");
    }
    {
        mx::ScopedFloatFormatting fmt(mx::Value::FloatFormatDefault, 2);
        REQUIRE(mx::toValueString(0.1234f) == "0.12");
    }

    // Convert from value strings to data values.
    REQUIRE(mx::fromValueString<int>("1") == 1);
    REQUIRE(mx::fromValueString<float>("1") == 1.0f);
    REQUIRE(mx::fromValueString<bool>("true") == true);
    REQUIRE(mx::fromValueString<bool>("false") == false);
    REQUIRE(mx::fromValueString<mx::Color3>("1, 1, 1") == mx::Color3(1.0f));
    REQUIRE(mx::fromValueString<std::string>("text") == "text");

    // Verify that invalid conversions throw exceptions.
    REQUIRE_THROWS_AS(mx::fromValueString<int>("text"), mx::ExceptionTypeError);
    REQUIRE_THROWS_AS(mx::fromValueString<float>("text"), mx::ExceptionTypeError);
    REQUIRE_THROWS_AS(mx::fromValueString<bool>("1"), mx::ExceptionTypeError);
    REQUIRE_THROWS_AS(mx::fromValueString<mx::Color3>("1"), mx::ExceptionTypeError);
}

TEST_CASE("Typed values", "[value]")
{
    // Base types
    testTypedValue<int>(1, 2);
    testTypedValue<bool>(false, true);
    testTypedValue<float>(1.0f, 2.0f);
    testTypedValue(mx::Color3(0.1f, 0.2f, 0.3f),
                   mx::Color3(0.5f, 0.6f, 0.7f));
    testTypedValue(mx::Color4(0.1f, 0.2f, 0.3f, 0.4f),
                   mx::Color4(0.5f, 0.6f, 0.7f, 0.8f));
    testTypedValue(mx::Vector2(1.0f, 2.0f),
                   mx::Vector2(1.5f, 2.5f));
    testTypedValue(mx::Vector3(1.0f, 2.0f, 3.0f),
                   mx::Vector3(1.5f, 2.5f, 3.5f));
    testTypedValue(mx::Vector4(1.0f, 2.0f, 3.0f, 4.0f),
                   mx::Vector4(1.5f, 2.5f, 3.5f, 4.5f));
    testTypedValue(mx::Matrix33(0.0f),
                   mx::Matrix33(1.0f));
    testTypedValue(mx::Matrix44(0.0f),
                   mx::Matrix44(1.0f));
    testTypedValue(std::string("first_value"),
                   std::string("second_value"));

    // Array types
    testTypedValue(mx::IntVec{1, 2, 3},
                   mx::IntVec{4, 5, 6});
    testTypedValue(mx::BoolVec{false, false, false},
                   mx::BoolVec{true, true, true});
    testTypedValue(mx::FloatVec{1.0f, 2.0f, 3.0f},
                   mx::FloatVec{4.0f, 5.0f, 6.0f});
    testTypedValue(mx::StringVec{"Item A", "Item B", "Item C"},
                   mx::StringVec{"Item D", "Item E", "Item F"});

    // Alias types
    testTypedValue<long>(1l, 2l);
    testTypedValue<double>(1.0, 2.0);

    // Construct a string value from a string literal
    mx::ValuePtr value = mx::Value::createValue("text");
    REQUIRE(value->isA<std::string>());
    REQUIRE(value->asA<std::string>() == "text");
}
