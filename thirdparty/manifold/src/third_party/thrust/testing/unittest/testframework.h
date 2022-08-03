#pragma once

#include <cstdio>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

#include "meta.h"
#include "util.h"

#include <thrust/limits.h>
#include <thrust/detail/config.h>
#include <thrust/detail/integer_traits.h>
#include <thrust/mr/host_memory_resource.h>
#include <thrust/mr/device_memory_resource.h>
#include <thrust/mr/universal_memory_resource.h>
#include <thrust/mr/allocator.h>

// define some common lists of types
typedef unittest::type_list<int,
                            unsigned int,
                            float> ThirtyTwoBitTypes;

typedef unittest::type_list<long long,
                            unsigned long long,
                            double> SixtyFourBitTypes;

typedef unittest::type_list<char,
                            signed char,
                            unsigned char,
                            short,
                            unsigned short,
                            int,
                            unsigned int,
                            long,
                            unsigned long,
                            long long,
                            unsigned long long> IntegralTypes;

typedef unittest::type_list<signed char,
                            signed short,
                            signed int,
                            signed long,
                            signed long long> SignedIntegralTypes;

typedef unittest::type_list<unsigned char,
                            unsigned short,
                            unsigned int,
                            unsigned long,
                            unsigned long long> UnsignedIntegralTypes;

typedef unittest::type_list<char,
                            signed char,
                            unsigned char> ByteTypes;

typedef unittest::type_list<char,
                            signed char,
                            unsigned char,
                            short,
                            unsigned short> SmallIntegralTypes;

typedef unittest::type_list<long long,
                            unsigned long long> LargeIntegralTypes;

typedef unittest::type_list<float,
                            double> FloatingPointTypes;

// A type that behaves as if it was a normal numeric type,
// so it can be used in the same tests as "normal" numeric types.
// NOTE: This is explicitly NOT proclaimed trivially reloctable.
class custom_numeric
{
public:
    __host__ __device__
    custom_numeric()
    {
        fill(0);
    }

    // Allow construction from any integral numeric.
    template <typename T,
              typename = typename std::enable_if<std::is_integral<T>::value>::type>
    __host__ __device__
    custom_numeric(const T& i)
    {
        fill(static_cast<int>(i));
    }

    __host__ __device__
    custom_numeric(const custom_numeric & other)
    {
        fill(other.value[0]);
    }

    __host__ __device__
    custom_numeric & operator=(int val)
    {
        fill(val);
        return *this;
    }

    __host__ __device__
    custom_numeric & operator=(const custom_numeric & other)
    {
        fill(other.value[0]);
        return *this;
    }

    // cast to void * instead of bool to fool overload resolution
    // WTB C++11 explicit conversion operators
    __host__ __device__
    operator void *() const
    {
        // static cast first to avoid MSVC warning C4312
        return reinterpret_cast<void *>(static_cast<std::size_t>(value[0]));
    }

#define DEFINE_OPERATOR(op)                                         \
    __host__ __device__                                             \
    custom_numeric & operator op() {                                \
        fill(op value[0]);                                          \
        return *this;                                               \
    }                                                               \
    __host__ __device__                                             \
    custom_numeric operator op(int) const {                         \
        custom_numeric ret(*this);                                  \
        op ret;                                                     \
        return ret;                                                 \
    }

    DEFINE_OPERATOR(++)
    DEFINE_OPERATOR(--)

#undef DEFINE_OPERATOR

#define DEFINE_OPERATOR(op)                                         \
    __host__ __device__                                             \
    custom_numeric operator op () const                             \
    {                                                               \
        return custom_numeric(op value[0]);                         \
    }

    DEFINE_OPERATOR(+)
    DEFINE_OPERATOR(-)
    DEFINE_OPERATOR(~)

#undef DEFINE_OPERATOR

#define DEFINE_OPERATOR(op)                                         \
    __host__ __device__                                             \
    custom_numeric operator op (const custom_numeric & other) const \
    {                                                               \
        return custom_numeric(value[0] op other.value[0]);          \
    }

    DEFINE_OPERATOR(+)
    DEFINE_OPERATOR(-)
    DEFINE_OPERATOR(*)
    DEFINE_OPERATOR(/)
    DEFINE_OPERATOR(%)
    DEFINE_OPERATOR(<<)
    DEFINE_OPERATOR(>>)
    DEFINE_OPERATOR(&)
    DEFINE_OPERATOR(|)
    DEFINE_OPERATOR(^)

#undef DEFINE_OPERATOR

#define CONCAT(X, Y) X ## Y

#define DEFINE_OPERATOR(op)                                         \
    __host__ __device__                                             \
    custom_numeric & operator CONCAT(op, =) (const custom_numeric & other) \
    {                                                               \
        fill(value[0] op other.value[0]);                           \
        return *this;                                               \
    }

    DEFINE_OPERATOR(+)
    DEFINE_OPERATOR(-)
    DEFINE_OPERATOR(*)
    DEFINE_OPERATOR(/)
    DEFINE_OPERATOR(%)
    DEFINE_OPERATOR(<<)
    DEFINE_OPERATOR(>>)
    DEFINE_OPERATOR(&)
    DEFINE_OPERATOR(|)
    DEFINE_OPERATOR(^)

#undef DEFINE_OPERATOR

#define DEFINE_OPERATOR(op)                                         \
    __host__ __device__                                             \
    friend bool operator op (const custom_numeric & lhs, const custom_numeric & rhs) \
    {                                                               \
        return lhs.value[0] op rhs.value[0];                        \
    }

    DEFINE_OPERATOR(==)
    DEFINE_OPERATOR(!=)
    DEFINE_OPERATOR(<)
    DEFINE_OPERATOR(<=)
    DEFINE_OPERATOR(>)
    DEFINE_OPERATOR(>=)
    DEFINE_OPERATOR(&&)
    DEFINE_OPERATOR(||);


#undef DEFINE_OPERATOR

    friend std::ostream & operator<<(std::ostream & os, const custom_numeric & val)
    {
        return os << "custom_numeric{" << val.value[0] << "}";
    }

private:
    int value[5];

    __host__ __device__
    void fill(int val)
    {
        for (int i = 0; i < 5; ++i)
        {
            value[i] = val;
        }
    }
};

THRUST_NAMESPACE_BEGIN

template <>
struct numeric_limits<custom_numeric> : numeric_limits<int> {};

namespace detail
{

// For random number generation
template<>
class integer_traits<custom_numeric>
  : public integer_traits_base<int, INT_MIN, INT_MAX>
{};

} // namespace detail

THRUST_NAMESPACE_END

typedef unittest::type_list<char,
                            signed char,
                            unsigned char,
                            short,
                            unsigned short,
                            int,
                            unsigned int,
                            long,
                            unsigned long,
                            long long,
                            unsigned long long,
                            float,
                            double,
                            custom_numeric> NumericTypes;

typedef unittest::type_list<char,
                            signed char,
                            unsigned char,
                            short,
                            unsigned short,
                            int,
                            unsigned int,
                            long,
                            unsigned long,
                            long long,
                            unsigned long long,
                            float,
                            double> BuiltinNumericTypes;

inline void chop_prefix(std::string& str, const std::string& prefix)
{
    str.replace(str.find(prefix) == 0 ? 0 : str.size(), prefix.size(), "");
}

inline std::string base_class_name(const std::string& name)
{
  std::string result = name;

  // if the name begins with "struct ", chop it off
  chop_prefix(result, "struct ");

  // if the name begins with "class ", chop it off
  chop_prefix(result, "class ");

  const std::size_t first_lt = result.find_first_of("<");

  if (first_lt < result.size())
      // chop everything including and after first "<"
      return result.replace(first_lt, result.size(), "");
  else
      return result;
}

enum TestStatus { Pass = 0, Failure = 1, KnownFailure = 2, Error = 3, UnknownException = 4};

typedef std::set<std::string>              ArgumentSet;
typedef std::map<std::string, std::string> ArgumentMap;

std::vector<size_t> get_test_sizes(void);
void                set_test_sizes(const std::string&);

class UnitTest {
    public:
        std::string name;
        UnitTest() {}
        UnitTest(const char * name);
        virtual ~UnitTest() {}
        virtual void run() {}

        bool operator<(const UnitTest& u) const
        {
            return name < u.name;
        }
};

class UnitTestDriver;

class UnitTestDriver
{
  typedef std::map<std::string, UnitTest*> TestMap;

  TestMap test_map;

  bool run_tests(std::vector<UnitTest *>& tests_to_run, const ArgumentMap& kwargs);

protected:
  // executed immediately after each test
  // \param test The UnitTest of interest
  // \param concise Whether or not to suppress output
  // \return true if all is well; false if the tests must be immediately aborted
  virtual bool post_test_smoke_check(const UnitTest &test, bool concise);

public:
  inline virtual ~UnitTestDriver() {};

  void register_test(UnitTest * test);
  virtual bool run_tests(const ArgumentSet& args, const ArgumentMap& kwargs);
  void list_tests(void);

  static UnitTestDriver &s_driver();
};

// Macro to create a single unittest
#define DECLARE_UNITTEST(TEST)                                   \
class TEST##UnitTest : public UnitTest {                         \
    public:                                                      \
    TEST##UnitTest() : UnitTest(#TEST) {}                        \
    void run(){                                                  \
            TEST();                                              \
    }                                                            \
};                                                               \
TEST##UnitTest TEST##Instance

#define DECLARE_UNITTEST_WITH_NAME(TEST, NAME)                   \
class NAME##UnitTest : public UnitTest {                         \
    public:                                                      \
    NAME##UnitTest() : UnitTest(#NAME) {}                        \
    void run(){                                                  \
        TEST();                                                  \
    }                                                            \
};                                                               \
NAME##UnitTest NAME##Instance

// Macro to create host and device versions of a
// unit test for a bunch of data types
#define DECLARE_VECTOR_UNITTEST(VTEST)                          \
void VTEST##Host(void) {                                        \
    VTEST< thrust::host_vector<signed char> >();                \
    VTEST< thrust::host_vector<short> >();                      \
    VTEST< thrust::host_vector<int> >();                        \
    VTEST< thrust::host_vector<float> >();                      \
    VTEST< thrust::host_vector<custom_numeric> >();             \
    /* MR vectors */                                            \
    VTEST< thrust::host_vector<int,                             \
        thrust::mr::stateless_resource_allocator<int,           \
            thrust::host_memory_resource> > >();                \
}                                                               \
void VTEST##Device(void) {                                      \
    VTEST< thrust::device_vector<signed char> >();              \
    VTEST< thrust::device_vector<short> >();                    \
    VTEST< thrust::device_vector<int> >();                      \
    VTEST< thrust::device_vector<float> >();                    \
    VTEST< thrust::device_vector<custom_numeric> >();           \
    /* MR vectors */                                            \
    VTEST< thrust::device_vector<int,                           \
        thrust::mr::stateless_resource_allocator<int,           \
            thrust::device_memory_resource> > >();              \
}                                                               \
void VTEST##Universal(void) {                                   \
    VTEST< thrust::universal_vector<int> >();                   \
    VTEST< thrust::device_vector<int,                           \
        thrust::mr::stateless_resource_allocator<int,           \
            thrust::universal_host_pinned_memory_resource> > >();\
}                                                               \
DECLARE_UNITTEST(VTEST##Host);                                  \
DECLARE_UNITTEST(VTEST##Device);                                \
DECLARE_UNITTEST(VTEST##Universal);

// Same as above, but only for integral types
#define DECLARE_INTEGRAL_VECTOR_UNITTEST(VTEST)                 \
void VTEST##Host(void) {                                        \
    VTEST< thrust::host_vector<signed char> >();                \
    VTEST< thrust::host_vector<short> >();                      \
    VTEST< thrust::host_vector<int> >();                        \
}                                                               \
void VTEST##Device(void) {                                      \
    VTEST< thrust::device_vector<signed char> >();              \
    VTEST< thrust::device_vector<short> >();                    \
    VTEST< thrust::device_vector<int> >();                      \
}                                                               \
void VTEST##Universal(void) {                                   \
    VTEST< thrust::universal_vector<int> >();                   \
    VTEST< thrust::device_vector<int,                           \
        thrust::mr::stateless_resource_allocator<int,           \
            thrust::universal_host_pinned_memory_resource> > >();\
}                                                               \
DECLARE_UNITTEST(VTEST##Host);                                  \
DECLARE_UNITTEST(VTEST##Device);                                \
DECLARE_UNITTEST(VTEST##Universal);

// Macro to create instances of a test for several data types.
#define DECLARE_GENERIC_UNITTEST(TEST)                           \
class TEST##UnitTest : public UnitTest {                         \
    public:                                                      \
    TEST##UnitTest() : UnitTest(#TEST) {}                        \
    void run()                                                   \
    {                                                            \
        TEST<signed char>();                                     \
        TEST<unsigned char>();                                   \
        TEST<short>();                                           \
        TEST<unsigned short>();                                  \
        TEST<int>();                                             \
        TEST<unsigned int>();                                    \
        TEST<float>();                                           \
    }                                                            \
};                                                               \
TEST##UnitTest TEST##Instance

// Macro to create instances of a test for several array sizes.
#define DECLARE_SIZED_UNITTEST(TEST)                             \
class TEST##UnitTest : public UnitTest {                         \
    public:                                                      \
    TEST##UnitTest() : UnitTest(#TEST) {}                        \
    void run()                                                   \
    {                                                            \
        std::vector<size_t> sizes = get_test_sizes();            \
        for(size_t i = 0; i != sizes.size(); ++i)                \
        {                                                        \
            TEST(sizes[i]);                                      \
        }                                                        \
    }                                                            \
};                                                               \
TEST##UnitTest TEST##Instance

// Macro to create instances of a test for several data types and array sizes
#define DECLARE_VARIABLE_UNITTEST(TEST)                          \
class TEST##UnitTest : public UnitTest {                         \
    public:                                                      \
    TEST##UnitTest() : UnitTest(#TEST) {}                        \
    void run()                                                   \
    {                                                            \
        std::vector<size_t> sizes = get_test_sizes();            \
        for(size_t i = 0; i != sizes.size(); ++i)                \
        {                                                        \
            TEST<signed char>(sizes[i]);                         \
            TEST<unsigned char>(sizes[i]);                       \
            TEST<short>(sizes[i]);                               \
            TEST<unsigned short>(sizes[i]);                      \
            TEST<int>(sizes[i]);                                 \
            TEST<unsigned int>(sizes[i]);                        \
            TEST<float>(sizes[i]);                               \
            TEST<double>(sizes[i]);                              \
        }                                                        \
    }                                                            \
};                                                               \
TEST##UnitTest TEST##Instance

#define DECLARE_INTEGRAL_VARIABLE_UNITTEST(TEST)                 \
class TEST##UnitTest : public UnitTest {                         \
    public:                                                      \
    TEST##UnitTest() : UnitTest(#TEST) {}                        \
    void run()                                                   \
    {                                                            \
        std::vector<size_t> sizes = get_test_sizes();            \
        for(size_t i = 0; i != sizes.size(); ++i)                \
        {                                                        \
            TEST<signed char>(sizes[i]);                         \
            TEST<unsigned char>(sizes[i]);                       \
            TEST<short>(sizes[i]);                               \
            TEST<unsigned short>(sizes[i]);                      \
            TEST<int>(sizes[i]);                                 \
            TEST<unsigned int>(sizes[i]);                        \
        }                                                        \
    }                                                            \
};                                                               \
TEST##UnitTest TEST##Instance

#define DECLARE_GENERIC_UNITTEST_WITH_TYPES_AND_NAME(TEST, TYPES, NAME)       \
  ::SimpleUnitTest<TEST, TYPES> NAME##_instance(#NAME)                        \
  /**/

#define DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(TEST, TYPES, NAME) \
  ::VariableUnitTest<TEST, TYPES> NAME##_instance(#NAME)                      \
  /**/

#define DECLARE_GENERIC_UNITTEST_WITH_TYPES(TEST, TYPES)                      \
  ::SimpleUnitTest<TEST, TYPES> TEST##_instance(#TEST)                        \
  /**/

#define DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES(TEST, TYPES)                \
  ::VariableUnitTest<TEST, TYPES> TEST##_instance(#TEST)                      \
  /**/

template<template <typename> class TestName, typename TypeList>
  class SimpleUnitTest : public UnitTest
{
  public:
    SimpleUnitTest()
      : UnitTest(base_class_name(unittest::type_name<TestName<int> >()).c_str()) {}

    SimpleUnitTest(const char * name)
      : UnitTest(name) {}

    void run()
    {
      // get the first type in the list
      typedef typename unittest::get_type<TypeList,0>::type first_type;

      unittest::for_each_type<TypeList,TestName,first_type,0> for_each;

      // loop over the types
      for_each();
    }
}; // end SimpleUnitTest


template<template <typename> class TestName, typename TypeList>
  class VariableUnitTest : public UnitTest
{
  public:
    VariableUnitTest()
      : UnitTest(base_class_name(unittest::type_name<TestName<int> >()).c_str()) {}

    VariableUnitTest(const char * name)
      : UnitTest(name) {}

    void run()
    {
        std::vector<size_t> sizes = get_test_sizes();
        for(size_t i = 0; i != sizes.size(); ++i)
        {
            // get the first type in the list
            typedef typename unittest::get_type<TypeList,0>::type first_type;

            unittest::for_each_type<TypeList,TestName,first_type,0> loop;

            // loop over the types
            loop(sizes[i]);
        }
    }
}; // end VariableUnitTest

template<template <typename> class TestName,
         typename TypeList,
         template <typename, typename> class Vector,
         template <typename> class Alloc>
  struct VectorUnitTest
    : public UnitTest
{
  VectorUnitTest()
    : UnitTest((base_class_name(unittest::type_name<TestName< Vector<int, Alloc<int> > > >()) + "<" +
                base_class_name(unittest::type_name<Vector<int, Alloc<int> > >()) + ">").c_str())
  { }

  VectorUnitTest(const char * name)
    : UnitTest(name) {}

  void run()
  {
    // zip up the type list with Alloc
    typedef typename unittest::transform1<TypeList, Alloc>::type AllocList;

    // zip up the type list & alloc list with Vector
    typedef typename unittest::transform2<TypeList, AllocList, Vector>::type VectorList;

    // get the first type in the list
    typedef typename unittest::get_type<VectorList,0>::type first_type;

    unittest::for_each_type<VectorList,TestName,first_type,0> loop;

    // loop over the types
    loop(0);
  }
}; // end VectorUnitTest

