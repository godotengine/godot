
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_tostring.hpp>
#include <catch2/interfaces/catch_interfaces_config.hpp>
#include <catch2/interfaces/catch_interfaces_registry_hub.hpp>
#include <catch2/internal/catch_context.hpp>
#include <catch2/internal/catch_polyfills.hpp>

#include <iomanip>

namespace Catch {

namespace Detail {

    namespace {
        const int hexThreshold = 255;

        struct Endianness {
            enum Arch : uint8_t {
                Big,
                Little
            };

            static Arch which() {
                int one = 1;
                // If the lowest byte we read is non-zero, we can assume
                // that little endian format is used.
                auto value = *reinterpret_cast<char*>(&one);
                return value ? Little : Big;
            }
        };

        template<typename T>
        std::string fpToString(T value, int precision) {
            if (Catch::isnan(value)) {
                return "nan";
            }

            ReusableStringStream rss;
            rss << std::setprecision(precision)
                << std::fixed
                << value;
            std::string d = rss.str();
            std::size_t i = d.find_last_not_of('0');
            if (i != std::string::npos && i != d.size() - 1) {
                if (d[i] == '.')
                    i++;
                d = d.substr(0, i + 1);
            }
            return d;
        }
    } // end unnamed namespace

    std::string convertIntoString(StringRef string, bool escapeInvisibles) {
        std::string ret;
        // This is enough for the "don't escape invisibles" case, and a good
        // lower bound on the "escape invisibles" case.
        ret.reserve(string.size() + 2);

        if (!escapeInvisibles) {
            ret += '"';
            ret += string;
            ret += '"';
            return ret;
        }

        ret += '"';
        for (char c : string) {
            switch (c) {
            case '\r':
                ret.append("\\r");
                break;
            case '\n':
                ret.append("\\n");
                break;
            case '\t':
                ret.append("\\t");
                break;
            case '\f':
                ret.append("\\f");
                break;
            default:
                ret.push_back(c);
                break;
            }
        }
        ret += '"';

        return ret;
    }

    std::string convertIntoString(StringRef string) {
        return convertIntoString(string, getCurrentContext().getConfig()->showInvisibles());
    }

    std::string rawMemoryToString( const void *object, std::size_t size ) {
        // Reverse order for little endian architectures
        int i = 0, end = static_cast<int>( size ), inc = 1;
        if( Endianness::which() == Endianness::Little ) {
            i = end-1;
            end = inc = -1;
        }

        unsigned char const *bytes = static_cast<unsigned char const *>(object);
        ReusableStringStream rss;
        rss << "0x" << std::setfill('0') << std::hex;
        for( ; i != end; i += inc )
             rss << std::setw(2) << static_cast<unsigned>(bytes[i]);
       return rss.str();
    }

    std::string makeExceptionHappenedString() {
        return "{ stringification failed with an exception: \"" +
               translateActiveException() + "\" }";

    }

} // end Detail namespace



//// ======================================================= ////
//
//   Out-of-line defs for full specialization of StringMaker
//
//// ======================================================= ////

std::string StringMaker<std::string>::convert(const std::string& str) {
    return Detail::convertIntoString( str );
}

#ifdef CATCH_CONFIG_CPP17_STRING_VIEW
std::string StringMaker<std::string_view>::convert(std::string_view str) {
    return Detail::convertIntoString( StringRef( str.data(), str.size() ) );
}
#endif

std::string StringMaker<char const*>::convert(char const* str) {
    if (str) {
        return Detail::convertIntoString( str );
    } else {
        return{ "{null string}" };
    }
}
std::string StringMaker<char*>::convert(char* str) { // NOLINT(readability-non-const-parameter)
    if (str) {
        return Detail::convertIntoString( str );
    } else {
        return{ "{null string}" };
    }
}

#ifdef CATCH_CONFIG_WCHAR
std::string StringMaker<std::wstring>::convert(const std::wstring& wstr) {
    std::string s;
    s.reserve(wstr.size());
    for (auto c : wstr) {
        s += (c <= 0xff) ? static_cast<char>(c) : '?';
    }
    return ::Catch::Detail::stringify(s);
}

# ifdef CATCH_CONFIG_CPP17_STRING_VIEW
std::string StringMaker<std::wstring_view>::convert(std::wstring_view str) {
    return StringMaker<std::wstring>::convert(std::wstring(str));
}
# endif

std::string StringMaker<wchar_t const*>::convert(wchar_t const * str) {
    if (str) {
        return ::Catch::Detail::stringify(std::wstring{ str });
    } else {
        return{ "{null string}" };
    }
}
std::string StringMaker<wchar_t *>::convert(wchar_t * str) {
    if (str) {
        return ::Catch::Detail::stringify(std::wstring{ str });
    } else {
        return{ "{null string}" };
    }
}
#endif

#if defined(CATCH_CONFIG_CPP17_BYTE)
#include <cstddef>
std::string StringMaker<std::byte>::convert(std::byte value) {
    return ::Catch::Detail::stringify(std::to_integer<unsigned long long>(value));
}
#endif // defined(CATCH_CONFIG_CPP17_BYTE)

std::string StringMaker<int>::convert(int value) {
    return ::Catch::Detail::stringify(static_cast<long long>(value));
}
std::string StringMaker<long>::convert(long value) {
    return ::Catch::Detail::stringify(static_cast<long long>(value));
}
std::string StringMaker<long long>::convert(long long value) {
    ReusableStringStream rss;
    rss << value;
    if (value > Detail::hexThreshold) {
        rss << " (0x" << std::hex << value << ')';
    }
    return rss.str();
}

std::string StringMaker<unsigned int>::convert(unsigned int value) {
    return ::Catch::Detail::stringify(static_cast<unsigned long long>(value));
}
std::string StringMaker<unsigned long>::convert(unsigned long value) {
    return ::Catch::Detail::stringify(static_cast<unsigned long long>(value));
}
std::string StringMaker<unsigned long long>::convert(unsigned long long value) {
    ReusableStringStream rss;
    rss << value;
    if (value > Detail::hexThreshold) {
        rss << " (0x" << std::hex << value << ')';
    }
    return rss.str();
}

std::string StringMaker<signed char>::convert(signed char value) {
    if (value == '\r') {
        return "'\\r'";
    } else if (value == '\f') {
        return "'\\f'";
    } else if (value == '\n') {
        return "'\\n'";
    } else if (value == '\t') {
        return "'\\t'";
    } else if ('\0' <= value && value < ' ') {
        return ::Catch::Detail::stringify(static_cast<unsigned int>(value));
    } else {
        char chstr[] = "' '";
        chstr[1] = value;
        return chstr;
    }
}
std::string StringMaker<char>::convert(char c) {
    return ::Catch::Detail::stringify(static_cast<signed char>(c));
}
std::string StringMaker<unsigned char>::convert(unsigned char value) {
    return ::Catch::Detail::stringify(static_cast<char>(value));
}

int StringMaker<float>::precision = std::numeric_limits<float>::max_digits10;

std::string StringMaker<float>::convert(float value) {
    return Detail::fpToString(value, precision) + 'f';
}

int StringMaker<double>::precision = std::numeric_limits<double>::max_digits10;

std::string StringMaker<double>::convert(double value) {
    return Detail::fpToString(value, precision);
}

} // end namespace Catch
