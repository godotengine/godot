#pragma once

#include "./defs.h"

// Microsoft is not supporting standard integer type define in C99 <cstdint>
//
#if defined(_MSC_VER) || defined(__GNUC__)

#include <cfloat>
#include <climits>

namespace mecanim
{
    //  8-bit types  -----------------------------------------------------------//
# if UCHAR_MAX == 0xff
    typedef signed char     int8_t;
    typedef unsigned char   uint8_t;

    #define INT8_T_MIN CHAR_MIN
    #define INT8_T_MAX CHAR_MAX
    #define UINT8_T_MIN 0
    #define UINT8_T_MAX UCHAR_MAX
# else
#   error defaults not correct; you must hand modify types.h
# endif

    //  16-bit types  -----------------------------------------------------------//
# if USHRT_MAX == 0xffff
    typedef short           int16_t;
    typedef unsigned short  uint16_t;

    #define INT16_T_MIN SHRT_MIN
    #define INT16_T_MAX SHRT_MAX
    #define UINT16_T_MIN 0
    #define UINT16_T_MAX USHRT_MAX
# else
#   error defaults not correct; you must hand modify types.h
# endif

    //  32-bit types  -----------------------------------------------------------//
#if UINT_MAX == 0xffffffff
    typedef int             int32_t;
    typedef unsigned int    uint32_t;

    #define INT32_T_MIN INT_MIN
    #define INT32_T_MAX INT_MAX
    #define UINT32_T_MIN 0
    #define UINT32_T_MAX UINT_MAX
#elif ULONG_MAX == 0xffffffff
    typedef long            int32_t;
    typedef unsigned long   uint32_t;

    #define INT32_T_MIN LONG_MIN
    #define INT32_T_MAX LONG_MAX
    #define UINT32_T_MIN 0
    #define UINT32_T_MAX ULONG_MAX
# else
#   error defaults not correct; you must hand modify types.h
# endif

    //  64-bit types  -----------------------------------------------------------//
# if ULLONG_MAX == 0xffffffffffffffff
    typedef long long            int64_t;
    typedef unsigned long long   uint64_t;

    #define INT64_T_MIN LLONG_MIN
    #define INT64_T_MAX LLONG_MAX
    #define UINT64_T_MIN 0
    #define UINT64_T_MAX ULLONG_MAX
# else
    //#   error defaults not correct; you must hand modify types.h
    typedef long long            int64_t;
    typedef unsigned long long   uint64_t;
# endif
    //typedef char String[20];
}
#else

#include <cfloat>
#include <climits>
#include <cstdint.h>
namespace mecanim
{
    using std::int8_t;
    using std::uint8_t;
    using std::int16_t;
    using std::uint16_t;
    using std::int32_t;
    using std::uint32_t;
    using std::int64_t;
    using std::uint64_t;

    #define INT8_T_MIN      CHAR_MIN
    #define INT8_T_MAX      CHAR_MAX
    #define UINT8_T_MIN     0
    #define UINT8_T_MAX     UCHAR_MAX

    #define INT16_T_MIN     SHRT_MIN
    #define INT16_T_MAX     SHRT_MAX
    #define UINT16_T_MIN    0
    #define UINT16_T_MAX    USHRT_MAX

# if ULONG_MAX == 0xffffffff
    #define INT32_T_MIN     LONG_MIN
    #define INT32_T_MAX     LONG_MAX
    #define UINT32_T_MIN    0
    #define UINT32_T_MAX    ULONG_MAX
# elif UINT_MAX == 0xffffffff
    #define INT32_T_MIN     INT_MIN
    #define INT32_T_MAX     INT_MAX
    #define UINT32_T_MIN    0
    #define UINT32_T_MAX    UINT_MAX
# else
#   error defaults not correct; you must hand modify types.h
# endif
    #define INT64_T_MIN     LLONG_MIN
    #define INT64_T_MAX     LLONG_MAX
    #define UINT64_T_MIN    0
    #define UINT64_T_MAX    ULLONG_MAX
}

#endif

namespace mecanim
{
    template<typename _Ty> struct BindValue
    {
        typedef _Ty     value_type;

        value_type      mValue;
        uint32_t        mID;
    };

    template<typename _Ty> struct numeric_limits
    {
        typedef _Ty Type;

        // C++ allow us to initialize only a static constant data member if
        // it has an integral or enumeration Type

        // All other specialization cannot have these member, consequently we do provide
        // min/max trait has inline member function for those case.
        static const Type min_value = Type(0);
        static const Type max_value = Type(0);

        static Type min() { return min_value; }
        static Type max() { return max_value; }
    };

    template<> struct numeric_limits<int32_t>
    {
        typedef int32_t Type;

        static const Type min_value = Type(INT32_T_MIN);
        static const Type max_value = Type(INT32_T_MAX);

        static Type min() { return min_value; }
        static Type max() { return max_value; }
    };

    template<> struct numeric_limits<uint32_t>
    {
        typedef uint32_t Type;

        static const Type min_value = Type(UINT32_T_MIN);
        static const Type max_value = Type(UINT32_T_MAX);

        static Type min() { return min_value; }
        static Type max() { return max_value; }
    };

    template<> struct numeric_limits<float>
    {
        typedef float Type;

        static Type min() { return -FLT_MAX; }
        static Type max() { return FLT_MAX; }
    };

    template<std::size_t> struct uint_t;
    template<> struct uint_t<8>  { typedef uint8_t value_type; };
    template<> struct uint_t<16> { typedef uint16_t value_type; };
    template<> struct uint_t<32> { typedef uint32_t value_type; };
    template<> struct uint_t<64> { typedef uint64_t value_type; };

    template<std::size_t> struct int_t;
    template<> struct int_t<8>  { typedef int8_t value_type; };
    template<> struct int_t<16> { typedef int16_t value_type; };
    template<> struct int_t<32> { typedef int32_t value_type; };
    template<> struct int_t<64> { typedef int64_t value_type; };


    template<bool C, typename Ta, typename Tb> class IfThenElse;
    template<typename Ta, typename Tb> class IfThenElse<true, Ta, Tb>
    {
    public:
        typedef Ta result_type;
    };

    template<typename Ta, typename Tb> class IfThenElse<false, Ta, Tb>
    {
    public:
        typedef Tb result_type;
    };

    template<typename T> class is_pointer
    {
    public:
        static const bool value = false;
    };

    template<typename T> class is_pointer<T*>
    {
    public:
        static const bool value = true;
    };
}
