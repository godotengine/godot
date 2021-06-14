#ifndef PARSERUTILS_H
#define PARSERUTILS_H

#include <cstring>
#include <cmath>
#include <limits>
#include <string>
#include <algorithm>

namespace lunasvg {

#define IS_ALPHA(c) (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
#define IS_NUM(c) (c >= '0' && c <= '9')
#define IS_WS(c) (c == ' ' || c == '\t' || c == '\n' || c == '\r')

namespace Utils {

inline const char* rtrim(const char* start, const char* end)
{
    while(end > start && IS_WS(end[-1]))
        --end;

    return end;
}

inline const char* ltrim(const char* start, const char* end)
{
    while(start < end && IS_WS(*start))
        ++start;

    return start;
}

inline bool skipDesc(const char*& ptr, const char* end, const char* data)
{
    int read = 0;
    while(data[read])
    {
        if(ptr >= end || *ptr != data[read])
        {
            ptr -= read;
            return false;
        }

        ++read;
        ++ptr;
    }

    return true;
}

inline bool skipUntil(const char*& ptr, const char* end, char delimiter)
{
    while(ptr < end && *ptr != delimiter)
        ++ptr;

    return ptr < end;
}

inline bool skipUntil(const char*& ptr, const char* end, const char* delimiter)
{
    while(ptr < end)
    {
        auto start = ptr;
        if(skipDesc(start, end, delimiter))
            break;
        ++ptr;
    }

    return ptr < end;
}

inline bool readUntil(const char*& ptr, const char* end, char delimiter, std::string& value)
{
    auto start = ptr;
    if(!skipUntil(ptr, end, delimiter))
        return false;

    value.assign(start, ptr);
    return true;
}

inline bool readUntil(const char*& ptr, const char* end, const char* delimiter, std::string& value)
{
    auto start = ptr;
    if(!skipUntil(ptr, end, delimiter))
        return false;

    value.assign(start, ptr);
    return true;
}

inline bool skipWs(const char*& ptr, const char* end)
{
    while(ptr < end && IS_WS(*ptr))
       ++ptr;

    return ptr < end;
}

inline bool skipWsDelimiter(const char*& ptr, const char* end, char delimiter)
{
    if(ptr < end && !IS_WS(*ptr) && *ptr != delimiter)
        return false;

    if(skipWs(ptr, end))
    {
        if(ptr < end && *ptr == delimiter)
        {
            ++ptr;
            skipWs(ptr, end);
        }
    }

    return ptr < end;
}

inline bool skipWsComma(const char*& ptr, const char* end)
{
    return skipWsDelimiter(ptr, end, ',');
}

inline bool isIntegralDigit(char ch, int base)
{
    if(IS_NUM(ch))
        return ch - '0' < base;

    if(IS_ALPHA(ch))
        return (ch>='a'&&ch<'a'+std::min(base, 36)-10) || (ch>='A'&&ch<'A'+std::min(base, 36)-10);

    return false;
}

template<typename T>
inline bool parseInteger(const char*& ptr, const char* end, T& integer, int base = 10)
{
    bool isNegative = 0;
    T value = 0;

    static const T intMax = std::numeric_limits<T>::max();
    static const bool isSigned = std::numeric_limits<T>::is_signed;
    using signed_t = typename std::make_signed<T>::type;
    const T maxMultiplier = intMax / static_cast<T>(base);

    if(ptr >= end)
        return false;

    if(isSigned && *ptr == '-')
    {
        ++ptr;
        isNegative = true;
    }
    else if(*ptr == '+')
        ++ptr;

    if(ptr >= end || !isIntegralDigit(*ptr, base))
        return false;

    int digitValue;
    while(ptr < end && isIntegralDigit(*ptr, base))
    {
        const char ch = *ptr++;
        if(IS_NUM(ch))
            digitValue = ch - '0';
        else if(ch >= 'a')
            digitValue = ch - 'a' + 10;
        else
            digitValue = ch - 'A' + 10;
        if(value > maxMultiplier || (value == maxMultiplier && static_cast<T>(digitValue) > (intMax % static_cast<T>(base)) + isNegative))
            return false;

        value = static_cast<T>(base) * value + static_cast<T>(digitValue);
    }

    if(isNegative)
        integer = -static_cast<signed_t>(value);
    else
        integer = value;

    return true;
}

template<typename T>
inline bool parseNumber(const char*& ptr, const char* end, T& number)
{
    T integer, fraction;
    int sign, expsign, exponent;

    static const T numberMax = std::numeric_limits<T>::max();
    fraction = 0;
    integer = 0;
    exponent = 0;
    sign = 1;
    expsign = 1;

    if(ptr < end && *ptr == '+')
        ++ptr;
    else if(ptr < end && *ptr == '-')
    {
        ++ptr;
        sign = -1;
    }

    if(ptr >= end || (!IS_NUM(*ptr) && *ptr != '.'))
        return false;

    if(*ptr != '.')
    {
        while(ptr < end && IS_NUM(*ptr))
            integer = static_cast<T>(10) * integer + (*ptr++ - '0');
    }

    if(ptr < end && *ptr == '.')
    {
        ++ptr;
        if(ptr >= end || !IS_NUM(*ptr))
            return false;
        T div = 1;
        while(ptr < end && IS_NUM(*ptr))
        {
            fraction = static_cast<T>(10) * fraction + (*ptr++ - '0');
            div *= static_cast<T>(10);
        }
        fraction /= div;
    }

    if(ptr < end && (*ptr == 'e' || *ptr == 'E')
       && (ptr[1] != 'x' && ptr[1] != 'm'))
    {
        ++ptr;
        if(ptr < end && *ptr == '+')
            ++ptr;
        else if(ptr < end && *ptr == '-')
        {
            ++ptr;
            expsign = -1;
        }

        if(ptr >= end || !IS_NUM(*ptr))
            return false;

        while(ptr < end && IS_NUM(*ptr))
            exponent = 10 * exponent + (*ptr++ - '0');
    }

    number = sign * (integer + fraction);
    if(exponent)
        number *= static_cast<T>(pow(10.0, expsign*exponent));

    return number >= -numberMax && number <= numberMax;
}

} // namespace Utils

} // namespace lunasvg

#endif // PARSERUTILS_H
