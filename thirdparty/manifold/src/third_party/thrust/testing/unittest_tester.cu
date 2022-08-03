#include <unittest/unittest.h>

void TestAssertEqual(void)
{
    ASSERT_EQUAL(0, 0);
    ASSERT_EQUAL(1, 1);
    ASSERT_EQUAL(-15.0f, -15.0f);
}
DECLARE_UNITTEST(TestAssertEqual);

void TestAssertLEqual(void)
{
    ASSERT_LEQUAL(0, 1);
    ASSERT_LEQUAL(0, 0);
}
DECLARE_UNITTEST(TestAssertLEqual);

void TestAssertGEqual(void)
{
    ASSERT_GEQUAL(1, 0);
    ASSERT_GEQUAL(0, 0);
}
DECLARE_UNITTEST(TestAssertGEqual);

void TestAssertLess(void)
{
    ASSERT_LESS(0, 1);
}
DECLARE_UNITTEST(TestAssertLess);

void TestAssertGreater(void)
{
    ASSERT_GREATER(1, 0);
}
DECLARE_UNITTEST(TestAssertGreater);

void TestTypeName(void)
{
    ASSERT_EQUAL(unittest::type_name<char>(),          "char");
    ASSERT_EQUAL(unittest::type_name<signed char>(),   "signed char");
    ASSERT_EQUAL(unittest::type_name<unsigned char>(), "unsigned char");
    ASSERT_EQUAL(unittest::type_name<int>(),           "int");
    ASSERT_EQUAL(unittest::type_name<float>(),         "float");
    ASSERT_EQUAL(unittest::type_name<double>(),        "double");
}
DECLARE_UNITTEST(TestTypeName);

