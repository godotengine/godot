//---------------------------------------------------------------------------------------
//
// ghc::filesystem - A C++17-like filesystem implementation for C++11/C++14/C++17/C++20
//
//---------------------------------------------------------------------------------------
//
// Copyright (c) 2018, Steffen Schümann <s.schuemann@pobox.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//---------------------------------------------------------------------------------------
//
// To dynamically select std::filesystem where available on most platforms,
// you could use:
//
// #if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || (defined(__cplusplus) && __cplusplus >= 201703L)) && defined(__has_include)
// #if __has_include(<filesystem>) && (!defined(__MAC_OS_X_VERSION_MIN_REQUIRED) || __MAC_OS_X_VERSION_MIN_REQUIRED >= 101500)
// #define GHC_USE_STD_FS
// #include <filesystem>
// namespace fs = std::filesystem;
// #endif
// #endif
// #ifndef GHC_USE_STD_FS
// #include <ghc/filesystem.hpp>
// namespace fs = ghc::filesystem;
// #endif
//
//---------------------------------------------------------------------------------------
#ifndef GHC_FILESYSTEM_H
#define GHC_FILESYSTEM_H

// #define BSD manifest constant only in
// sys/param.h
#ifndef _WIN32
#include <sys/param.h>
#endif

#ifndef GHC_OS_DETECTED
#if defined(__APPLE__) && defined(__MACH__)
#define GHC_OS_MACOS
#elif defined(__linux__)
#define GHC_OS_LINUX
#if defined(__ANDROID__)
#define GHC_OS_ANDROID
#endif
#elif defined(_WIN64)
#define GHC_OS_WINDOWS
#define GHC_OS_WIN64
#elif defined(_WIN32)
#define GHC_OS_WINDOWS
#define GHC_OS_WIN32
#elif defined(__CYGWIN__)
#define GHC_OS_CYGWIN
#elif defined(__sun) && defined(__SVR4)
#define GHC_OS_SOLARIS
#elif defined(__svr4__)
#define GHC_OS_SYS5R4
#elif defined(BSD)
#define GHC_OS_BSD
#elif defined(__EMSCRIPTEN__)
#define GHC_OS_WEB
#include <wasi/api.h>
#elif defined(__QNX__)
#define GHC_OS_QNX
#else
#error "Operating system currently not supported!"
#endif
#define GHC_OS_DETECTED
#if (defined(_MSVC_LANG) && _MSVC_LANG >= 201703L)
#if _MSVC_LANG == 201703L
#define GHC_FILESYSTEM_RUNNING_CPP17
#else
#define GHC_FILESYSTEM_RUNNING_CPP20
#endif
#elif (defined(__cplusplus) && __cplusplus >= 201703L)
#if __cplusplus == 201703L
#define GHC_FILESYSTEM_RUNNING_CPP17
#else
#define GHC_FILESYSTEM_RUNNING_CPP20
#endif
#endif
#endif

#if defined(GHC_FILESYSTEM_IMPLEMENTATION)
#define GHC_EXPAND_IMPL
#define GHC_INLINE
#ifdef GHC_OS_WINDOWS
#ifndef GHC_FS_API
#define GHC_FS_API
#endif
#ifndef GHC_FS_API_CLASS
#define GHC_FS_API_CLASS
#endif
#else
#ifndef GHC_FS_API
#define GHC_FS_API __attribute__((visibility("default")))
#endif
#ifndef GHC_FS_API_CLASS
#define GHC_FS_API_CLASS __attribute__((visibility("default")))
#endif
#endif
#elif defined(GHC_FILESYSTEM_FWD)
#define GHC_INLINE
#ifdef GHC_OS_WINDOWS
#ifndef GHC_FS_API
#define GHC_FS_API extern
#endif
#ifndef GHC_FS_API_CLASS
#define GHC_FS_API_CLASS
#endif
#else
#ifndef GHC_FS_API
#define GHC_FS_API extern
#endif
#ifndef GHC_FS_API_CLASS
#define GHC_FS_API_CLASS
#endif
#endif
#else
#define GHC_EXPAND_IMPL
#define GHC_INLINE inline
#ifndef GHC_FS_API
#define GHC_FS_API
#endif
#ifndef GHC_FS_API_CLASS
#define GHC_FS_API_CLASS
#endif
#endif

#ifdef GHC_EXPAND_IMPL

#ifdef GHC_OS_WINDOWS
#include <windows.h>
// additional includes
#include <shellapi.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <wchar.h>
#include <winioctl.h>
#else
#include <dirent.h>
#include <fcntl.h>
#include <limits.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#ifdef GHC_OS_ANDROID
#include <android/api-level.h>
#if __ANDROID_API__ < 12
#include <sys/syscall.h>
#endif
#include <sys/vfs.h>
#define statvfs statfs
#else
#include <sys/statvfs.h>
#endif
#ifdef GHC_OS_CYGWIN
#include <strings.h>
#endif
#if !defined(__ANDROID__) || __ANDROID_API__ >= 26
#include <langinfo.h>
#endif
#endif
#ifdef GHC_OS_MACOS
#include <Availability.h>
#endif

#if defined(__cpp_impl_three_way_comparison) && defined(__has_include)
#if __has_include(<compare>)
#define GHC_HAS_THREEWAY_COMP
#include <compare>
#endif
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <clocale>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <stack>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#else  // GHC_EXPAND_IMPL

#if defined(__cpp_impl_three_way_comparison) && defined(__has_include)
#if __has_include(<compare>)
#define GHC_HAS_THREEWAY_COMP
#include <compare>
#endif
#endif
#include <chrono>
#include <fstream>
#include <memory>
#include <stack>
#include <stdexcept>
#include <string>
#include <system_error>
#ifdef GHC_OS_WINDOWS
#include <vector>
#endif
#endif  // GHC_EXPAND_IMPL

// After standard library includes.
// Standard library support for std::string_view.
#if defined(__cpp_lib_string_view)
#define GHC_HAS_STD_STRING_VIEW
#elif defined(_LIBCPP_VERSION) && (_LIBCPP_VERSION >= 4000) && (__cplusplus >= 201402)
#define GHC_HAS_STD_STRING_VIEW
#elif defined(_GLIBCXX_RELEASE) && (_GLIBCXX_RELEASE >= 7) && (__cplusplus >= 201703)
#define GHC_HAS_STD_STRING_VIEW
#elif defined(_MSC_VER) && (_MSC_VER >= 1910 && _MSVC_LANG >= 201703)
#define GHC_HAS_STD_STRING_VIEW
#endif

// Standard library support for std::experimental::string_view.
#if defined(_LIBCPP_VERSION) && (_LIBCPP_VERSION >= 3700 && _LIBCPP_VERSION < 7000) && (__cplusplus >= 201402)
#define GHC_HAS_STD_EXPERIMENTAL_STRING_VIEW
#elif defined(__GNUC__) && (((__GNUC__ == 4) && (__GNUC_MINOR__ >= 9)) || (__GNUC__ > 4)) && (__cplusplus >= 201402)
#define GHC_HAS_STD_EXPERIMENTAL_STRING_VIEW
#elif defined(__GLIBCXX__) && defined(_GLIBCXX_USE_DUAL_ABI) && (__cplusplus >= 201402)
// macro _GLIBCXX_USE_DUAL_ABI is always defined in libstdc++ from gcc-5 and newer
#define GHC_HAS_STD_EXPERIMENTAL_STRING_VIEW
#endif

#if defined(GHC_HAS_STD_STRING_VIEW)
#include <string_view>
#elif defined(GHC_HAS_STD_EXPERIMENTAL_STRING_VIEW)
#include <experimental/string_view>
#endif

#if !defined(GHC_OS_WINDOWS) && !defined(PATH_MAX)
#define PATH_MAX 4096
#endif

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Behaviour Switches (see README.md, should match the config in test/filesystem_test.cpp):
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Enforce C++17 API where possible when compiling for C++20, handles the following cases:
// * fs::path::u8string() returns std::string instead of std::u8string
// #define GHC_FILESYSTEM_ENFORCE_CPP17_API
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// LWG #2682 disables the since then invalid use of the copy option create_symlinks on directories
// configure LWG conformance ()
#define LWG_2682_BEHAVIOUR
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// LWG #2395 makes crate_directory/create_directories not emit an error if there is a regular
// file with that name, it is superseded by P1164R1, so only activate if really needed
// #define LWG_2935_BEHAVIOUR
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// LWG #2936 enables new element wise (more expensive) path comparison
// * if this->root_name().native().compare(p.root_name().native()) != 0 return result
// * if this->has_root_directory() and !p.has_root_directory() return -1
// * if !this->has_root_directory() and p.has_root_directory() return -1
// * else result of element wise comparison of path iteration where first comparison is != 0 or 0
//   if all comparisons are 0 (on Windows this implementation does case-insensitive root_name()
//   comparison)
#define LWG_2936_BEHAVIOUR
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// LWG #2937 enforces that fs::equivalent emits an error, if !fs::exists(p1)||!exists(p2)
#define LWG_2937_BEHAVIOUR
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// UTF8-Everywhere is the original behaviour of ghc::filesystem. But since v1.5 the Windows
// version defaults to std::wstring storage backend. Still all std::string will be interpreted
// as UTF-8 encoded. With this define you can enforce the old behavior on Windows, using
// std::string as backend and for fs::path::native() and char for fs::path::c_str(). This
// needs more conversions, so it is (and was before v1.5) slower, bot might help keeping source
// homogeneous in a multi-platform project.
// #define GHC_WIN_DISABLE_WSTRING_STORAGE_TYPE
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Raise errors/exceptions when invalid unicode codepoints or UTF-8 sequences are found,
// instead of replacing them with the unicode replacement character (U+FFFD).
// #define GHC_RAISE_UNICODE_ERRORS
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Automatic prefix windows path with "\\?\" if they would break the MAX_PATH length.
// instead of replacing them with the unicode replacement character (U+FFFD).
#ifndef GHC_WIN_DISABLE_AUTO_PREFIXES
#define GHC_WIN_AUTO_PREFIX_LONG_PATH
#endif  // GHC_WIN_DISABLE_AUTO_PREFIXES
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

// ghc::filesystem version in decimal (major * 10000 + minor * 100 + patch)
#define GHC_FILESYSTEM_VERSION 10512L

// TinyUSDZ mod
#ifndef GHC_NO_EXCEPTION
#define GHC_NO_EXCEPTION
#endif

#if defined(GHC_NO_EXCEPTION)

// Should go here.
#if defined(GHC_RAISE_UNICODE_ERRORS)
#error "Can't raise unicode errors with exception support disabled"
#endif

#else

#if !defined(GHC_WITH_EXCEPTIONS) && (defined(__EXCEPTIONS) || defined(__cpp_exceptions) || defined(_CPPUNWIND))
#define GHC_WITH_EXCEPTIONS
#endif
#if !defined(GHC_WITH_EXCEPTIONS) && defined(GHC_RAISE_UNICODE_ERRORS)
#error "Can't raise unicode errors with exception support disabled"
#endif

#endif

namespace ghc {
namespace filesystem {

#if defined(GHC_HAS_CUSTOM_STRING_VIEW)
#define GHC_WITH_STRING_VIEW
#elif defined(GHC_HAS_STD_STRING_VIEW)
#define GHC_WITH_STRING_VIEW
using std::basic_string_view;
#elif defined(GHC_HAS_STD_EXPERIMENTAL_STRING_VIEW)
#define GHC_WITH_STRING_VIEW
using std::experimental::basic_string_view;
#endif

// temporary existing exception type for yet unimplemented parts
class GHC_FS_API_CLASS not_implemented_exception : public std::logic_error
{
public:
    not_implemented_exception()
        : std::logic_error("function not implemented yet.")
    {
    }
};

template <typename char_type>
class path_helper_base
{
public:
    using value_type = char_type;
#ifdef GHC_OS_WINDOWS
    static constexpr value_type preferred_separator = '\\';
#else
    static constexpr value_type preferred_separator = '/';
#endif
};

#if __cplusplus < 201703L
template <typename char_type>
constexpr char_type path_helper_base<char_type>::preferred_separator;
#endif

#ifdef GHC_OS_WINDOWS
class path;
namespace detail {
bool has_executable_extension(const path& p);
}
#endif

// [fs.class.path] class path
class GHC_FS_API_CLASS path
#if defined(GHC_OS_WINDOWS) && !defined(GHC_WIN_DISABLE_WSTRING_STORAGE_TYPE)
#define GHC_USE_WCHAR_T
#define GHC_NATIVEWP(p) p.c_str()
#define GHC_PLATFORM_LITERAL(str) L##str
    : private path_helper_base<std::wstring::value_type>
{
public:
    using path_helper_base<std::wstring::value_type>::value_type;
#else
#define GHC_NATIVEWP(p) p.wstring().c_str()
#define GHC_PLATFORM_LITERAL(str) str
    : private path_helper_base<std::string::value_type>
{
public:
    using path_helper_base<std::string::value_type>::value_type;
#endif
    using string_type = std::basic_string<value_type>;
    using path_helper_base<value_type>::preferred_separator;

    // [fs.enum.path.format] enumeration format
    /// The path format in which the constructor argument is given.
    enum format {
        generic_format,  ///< The generic format, internally used by
                         ///< ghc::filesystem::path with slashes
        native_format,   ///< The format native to the current platform this code
                         ///< is build for
        auto_format,     ///< Try to auto-detect the format, fallback to native
    };

    template <class T>
    struct _is_basic_string : std::false_type
    {
    };
    template <class CharT, class Traits, class Alloc>
    struct _is_basic_string<std::basic_string<CharT, Traits, Alloc>> : std::true_type
    {
    };
    template <class CharT>
    struct _is_basic_string<std::basic_string<CharT, std::char_traits<CharT>, std::allocator<CharT>>> : std::true_type
    {
    };
#ifdef GHC_WITH_STRING_VIEW
    template <class CharT, class Traits>
    struct _is_basic_string<basic_string_view<CharT, Traits>> : std::true_type
    {
    };
    template <class CharT>
    struct _is_basic_string<basic_string_view<CharT, std::char_traits<CharT>>> : std::true_type
    {
    };
#endif

    template <typename T1, typename T2 = void>
    using path_type = typename std::enable_if<!std::is_same<path, T1>::value, path>::type;
    template <typename T>
#if defined(__cpp_lib_char8_t) && !defined(GHC_FILESYSTEM_ENFORCE_CPP17_API)
    using path_from_string =
        typename std::enable_if<_is_basic_string<T>::value || std::is_same<char const*, typename std::decay<T>::type>::value || std::is_same<char*, typename std::decay<T>::type>::value || std::is_same<char8_t const*, typename std::decay<T>::type>::value ||
                                    std::is_same<char8_t*, typename std::decay<T>::type>::value || std::is_same<char16_t const*, typename std::decay<T>::type>::value || std::is_same<char16_t*, typename std::decay<T>::type>::value ||
                                    std::is_same<char32_t const*, typename std::decay<T>::type>::value || std::is_same<char32_t*, typename std::decay<T>::type>::value || std::is_same<wchar_t const*, typename std::decay<T>::type>::value ||
                                    std::is_same<wchar_t*, typename std::decay<T>::type>::value,
                                path>::type;
    template <typename T>
    using path_type_EcharT = typename std::enable_if<std::is_same<T, char>::value || std::is_same<T, char8_t>::value || std::is_same<T, char16_t>::value || std::is_same<T, char32_t>::value || std::is_same<T, wchar_t>::value, path>::type;
#else
    using path_from_string =
        typename std::enable_if<_is_basic_string<T>::value || std::is_same<char const*, typename std::decay<T>::type>::value || std::is_same<char*, typename std::decay<T>::type>::value ||
                                    std::is_same<char16_t const*, typename std::decay<T>::type>::value || std::is_same<char16_t*, typename std::decay<T>::type>::value || std::is_same<char32_t const*, typename std::decay<T>::type>::value ||
                                    std::is_same<char32_t*, typename std::decay<T>::type>::value || std::is_same<wchar_t const*, typename std::decay<T>::type>::value || std::is_same<wchar_t*, typename std::decay<T>::type>::value,
                                path>::type;
    template <typename T>
    using path_type_EcharT = typename std::enable_if<std::is_same<T, char>::value || std::is_same<T, char16_t>::value || std::is_same<T, char32_t>::value || std::is_same<T, wchar_t>::value, path>::type;
#endif
    // [fs.path.construct] constructors and destructor
    path() noexcept;
    path(const path& p);
    path(path&& p) noexcept;
    path(string_type&& source, format fmt = auto_format);
    template <class Source, typename = path_from_string<Source>>
    path(const Source& source, format fmt = auto_format);
    template <class InputIterator>
    path(InputIterator first, InputIterator last, format fmt = auto_format);
#ifdef GHC_WITH_EXCEPTIONS
    template <class Source, typename = path_from_string<Source>>
    path(const Source& source, const std::locale& loc, format fmt = auto_format);
    template <class InputIterator>
    path(InputIterator first, InputIterator last, const std::locale& loc, format fmt = auto_format);
#endif
    ~path();

    // [fs.path.assign] assignments
    path& operator=(const path& p);
    path& operator=(path&& p) noexcept;
    path& operator=(string_type&& source);
    path& assign(string_type&& source);
    template <class Source>
    path& operator=(const Source& source);
    template <class Source>
    path& assign(const Source& source);
    template <class InputIterator>
    path& assign(InputIterator first, InputIterator last);

    // [fs.path.append] appends
    path& operator/=(const path& p);
    template <class Source>
    path& operator/=(const Source& source);
    template <class Source>
    path& append(const Source& source);
    template <class InputIterator>
    path& append(InputIterator first, InputIterator last);

    // [fs.path.concat] concatenation
    path& operator+=(const path& x);
    path& operator+=(const string_type& x);
#ifdef GHC_WITH_STRING_VIEW
    path& operator+=(basic_string_view<value_type> x);
#endif
    path& operator+=(const value_type* x);
    path& operator+=(value_type x);
    template <class Source>
    path_from_string<Source>& operator+=(const Source& x);
    template <class EcharT>
    path_type_EcharT<EcharT>& operator+=(EcharT x);
    template <class Source>
    path& concat(const Source& x);
    template <class InputIterator>
    path& concat(InputIterator first, InputIterator last);

    // [fs.path.modifiers] modifiers
    void clear() noexcept;
    path& make_preferred();
    path& remove_filename();
    path& replace_filename(const path& replacement);
    path& replace_extension(const path& replacement = path());
    void swap(path& rhs) noexcept;

    // [fs.path.native.obs] native format observers
    const string_type& native() const noexcept;
    const value_type* c_str() const noexcept;
    operator string_type() const;
    template <class EcharT, class traits = std::char_traits<EcharT>, class Allocator = std::allocator<EcharT>>
    std::basic_string<EcharT, traits, Allocator> string(const Allocator& a = Allocator()) const;
    std::string string() const;
    std::wstring wstring() const;
#if defined(__cpp_lib_char8_t) && !defined(GHC_FILESYSTEM_ENFORCE_CPP17_API)
    std::u8string u8string() const;
#else
    std::string u8string() const;
#endif
    std::u16string u16string() const;
    std::u32string u32string() const;

    // [fs.path.generic.obs] generic format observers
    template <class EcharT, class traits = std::char_traits<EcharT>, class Allocator = std::allocator<EcharT>>
    std::basic_string<EcharT, traits, Allocator> generic_string(const Allocator& a = Allocator()) const;
    std::string generic_string() const;
    std::wstring generic_wstring() const;
#if defined(__cpp_lib_char8_t) && !defined(GHC_FILESYSTEM_ENFORCE_CPP17_API)
    std::u8string generic_u8string() const;
#else
    std::string generic_u8string() const;
#endif
    std::u16string generic_u16string() const;
    std::u32string generic_u32string() const;

    // [fs.path.compare] compare
    int compare(const path& p) const noexcept;
    int compare(const string_type& s) const;
#ifdef GHC_WITH_STRING_VIEW
    int compare(basic_string_view<value_type> s) const;
#endif
    int compare(const value_type* s) const;

    // [fs.path.decompose] decomposition
    path root_name() const;
    path root_directory() const;
    path root_path() const;
    path relative_path() const;
    path parent_path() const;
    path filename() const;
    path stem() const;
    path extension() const;

    // [fs.path.query] query
    bool empty() const noexcept;
    bool has_root_name() const;
    bool has_root_directory() const;
    bool has_root_path() const;
    bool has_relative_path() const;
    bool has_parent_path() const;
    bool has_filename() const;
    bool has_stem() const;
    bool has_extension() const;
    bool is_absolute() const;
    bool is_relative() const;

    // [fs.path.gen] generation
    path lexically_normal() const;
    path lexically_relative(const path& base) const;
    path lexically_proximate(const path& base) const;

    // [fs.path.itr] iterators
    class iterator;
    using const_iterator = iterator;
    iterator begin() const;
    iterator end() const;

private:
    using impl_value_type = value_type;
    using impl_string_type = std::basic_string<impl_value_type>;
    friend class directory_iterator;
    void append_name(const value_type* name);
    static constexpr impl_value_type generic_separator = '/';
    template <typename InputIterator>
    class input_iterator_range
    {
    public:
        typedef InputIterator iterator;
        typedef InputIterator const_iterator;
        typedef typename InputIterator::difference_type difference_type;

        input_iterator_range(const InputIterator& first, const InputIterator& last)
            : _first(first)
            , _last(last)
        {
        }

        InputIterator begin() const { return _first; }
        InputIterator end() const { return _last; }

    private:
        InputIterator _first;
        InputIterator _last;
    };
    friend void swap(path& lhs, path& rhs) noexcept;
    friend size_t hash_value(const path& p) noexcept;
    friend path canonical(const path& p, std::error_code& ec);
    friend bool create_directories(const path& p, std::error_code& ec) noexcept;
    string_type::size_type root_name_length() const noexcept;
    void postprocess_path_with_format(format fmt);
    void check_long_path();
    impl_string_type _path;
#ifdef GHC_OS_WINDOWS
    void handle_prefixes();
    friend bool detail::has_executable_extension(const path& p);
#ifdef GHC_WIN_AUTO_PREFIX_LONG_PATH
    string_type::size_type _prefixLength{0};
#else   // GHC_WIN_AUTO_PREFIX_LONG_PATH
    static const string_type::size_type _prefixLength{0};
#endif  // GHC_WIN_AUTO_PREFIX_LONG_PATH
#else
    static const string_type::size_type _prefixLength{0};
#endif
};

// [fs.path.nonmember] path non-member functions
GHC_FS_API void swap(path& lhs, path& rhs) noexcept;
GHC_FS_API size_t hash_value(const path& p) noexcept;
#ifdef GHC_HAS_THREEWAY_COMP
GHC_FS_API std::strong_ordering operator<=>(const path& lhs, const path& rhs) noexcept;
#endif
GHC_FS_API bool operator==(const path& lhs, const path& rhs) noexcept;
GHC_FS_API bool operator!=(const path& lhs, const path& rhs) noexcept;
GHC_FS_API bool operator<(const path& lhs, const path& rhs) noexcept;
GHC_FS_API bool operator<=(const path& lhs, const path& rhs) noexcept;
GHC_FS_API bool operator>(const path& lhs, const path& rhs) noexcept;
GHC_FS_API bool operator>=(const path& lhs, const path& rhs) noexcept;
GHC_FS_API path operator/(const path& lhs, const path& rhs);

// [fs.path.io] path inserter and extractor
template <class charT, class traits>
std::basic_ostream<charT, traits>& operator<<(std::basic_ostream<charT, traits>& os, const path& p);
template <class charT, class traits>
std::basic_istream<charT, traits>& operator>>(std::basic_istream<charT, traits>& is, path& p);

// [pfs.path.factory] path factory functions
template <class Source, typename = path::path_from_string<Source>>
#if defined(__cpp_lib_char8_t) && !defined(GHC_FILESYSTEM_ENFORCE_CPP17_API)
[[deprecated("use ghc::filesystem::path::path() with std::u8string instead")]]
#endif
path u8path(const Source& source);
template <class InputIterator>
#if defined(__cpp_lib_char8_t) && !defined(GHC_FILESYSTEM_ENFORCE_CPP17_API)
[[deprecated("use ghc::filesystem::path::path() with std::u8string instead")]]
#endif
path u8path(InputIterator first, InputIterator last);

// [fs.class.filesystem_error] class filesystem_error
class GHC_FS_API_CLASS filesystem_error : public std::system_error
{
public:
    filesystem_error(const std::string& what_arg, std::error_code ec);
    filesystem_error(const std::string& what_arg, const path& p1, std::error_code ec);
    filesystem_error(const std::string& what_arg, const path& p1, const path& p2, std::error_code ec);
    const path& path1() const noexcept;
    const path& path2() const noexcept;
    const char* what() const noexcept override;

private:
    std::string _what_arg;
    std::error_code _ec;
    path _p1, _p2;
};

class GHC_FS_API_CLASS path::iterator
{
public:
    using value_type = const path;
    using difference_type = std::ptrdiff_t;
    using pointer = const path*;
    using reference = const path&;
    using iterator_category = std::bidirectional_iterator_tag;

    iterator();
    iterator(const path& p, const impl_string_type::const_iterator& pos);
    iterator& operator++();
    iterator operator++(int);
    iterator& operator--();
    iterator operator--(int);
    bool operator==(const iterator& other) const;
    bool operator!=(const iterator& other) const;
    reference operator*() const;
    pointer operator->() const;

private:
    friend class path;
    impl_string_type::const_iterator increment(const impl_string_type::const_iterator& pos) const;
    impl_string_type::const_iterator decrement(const impl_string_type::const_iterator& pos) const;
    void updateCurrent();
    impl_string_type::const_iterator _first;
    impl_string_type::const_iterator _last;
    impl_string_type::const_iterator _prefix;
    impl_string_type::const_iterator _root;
    impl_string_type::const_iterator _iter;
    path _current;
};

struct space_info
{
    uintmax_t capacity;
    uintmax_t free;
    uintmax_t available;
};

// [fs.enum] enumerations
// [fs.enum.file_type]
enum class file_type {
    none,
    not_found,
    regular,
    directory,
    symlink,
    block,
    character,
    fifo,
    socket,
    unknown,
};

// [fs.enum.perms]
enum class perms : uint16_t {
    none = 0,

    owner_read = 0400,
    owner_write = 0200,
    owner_exec = 0100,
    owner_all = 0700,

    group_read = 040,
    group_write = 020,
    group_exec = 010,
    group_all = 070,

    others_read = 04,
    others_write = 02,
    others_exec = 01,
    others_all = 07,

    all = 0777,
    set_uid = 04000,
    set_gid = 02000,
    sticky_bit = 01000,

    mask = 07777,
    unknown = 0xffff
};

// [fs.enum.perm.opts]
enum class perm_options : uint16_t {
    replace = 3,
    add = 1,
    remove = 2,
    nofollow = 4,
};

// [fs.enum.copy.opts]
enum class copy_options : uint16_t {
    none = 0,

    skip_existing = 1,
    overwrite_existing = 2,
    update_existing = 4,

    recursive = 8,

    copy_symlinks = 0x10,
    skip_symlinks = 0x20,

    directories_only = 0x40,
    create_symlinks = 0x80,
#ifndef GHC_OS_WEB
    create_hard_links = 0x100
#endif
};

// [fs.enum.dir.opts]
enum class directory_options : uint16_t {
    none = 0,
    follow_directory_symlink = 1,
    skip_permission_denied = 2,
};

// [fs.class.file_status] class file_status
class GHC_FS_API_CLASS file_status
{
public:
    // [fs.file_status.cons] constructors and destructor
    file_status() noexcept;
    explicit file_status(file_type ft, perms prms = perms::unknown) noexcept;
    file_status(const file_status&) noexcept;
    file_status(file_status&&) noexcept;
    ~file_status();
    // assignments:
    file_status& operator=(const file_status&) noexcept;
    file_status& operator=(file_status&&) noexcept;
    // [fs.file_status.mods] modifiers
    void type(file_type ft) noexcept;
    void permissions(perms prms) noexcept;
    // [fs.file_status.obs] observers
    file_type type() const noexcept;
    perms permissions() const noexcept;
    friend bool operator==(const file_status& lhs, const file_status& rhs) noexcept { return lhs.type() == rhs.type() && lhs.permissions() == rhs.permissions(); }

private:
    file_type _type;
    perms _perms;
};

using file_time_type = std::chrono::time_point<std::chrono::system_clock>;

// [fs.class.directory_entry] Class directory_entry
class GHC_FS_API_CLASS directory_entry
{
public:
    // [fs.dir.entry.cons] constructors and destructor
    directory_entry() noexcept = default;
    directory_entry(const directory_entry&) = default;
    directory_entry(directory_entry&&) noexcept = default;
#ifdef GHC_WITH_EXCEPTIONS
    explicit directory_entry(const path& p);
#endif
    directory_entry(const path& p, std::error_code& ec);
    ~directory_entry();

    // assignments:
    directory_entry& operator=(const directory_entry&) = default;
    directory_entry& operator=(directory_entry&&) noexcept = default;

    // [fs.dir.entry.mods] modifiers
#ifdef GHC_WITH_EXCEPTIONS
    void assign(const path& p);
    void replace_filename(const path& p);
    void refresh();
#endif
    void assign(const path& p, std::error_code& ec);
    void replace_filename(const path& p, std::error_code& ec);
    void refresh(std::error_code& ec) noexcept;

    // [fs.dir.entry.obs] observers
    const filesystem::path& path() const noexcept;
    operator const filesystem::path&() const noexcept;
#ifdef GHC_WITH_EXCEPTIONS
    bool exists() const;
    bool is_block_file() const;
    bool is_character_file() const;
    bool is_directory() const;
    bool is_fifo() const;
    bool is_other() const;
    bool is_regular_file() const;
    bool is_socket() const;
    bool is_symlink() const;
    uintmax_t file_size() const;
    file_time_type last_write_time() const;
    file_status status() const;
    file_status symlink_status() const;
#endif
    bool exists(std::error_code& ec) const noexcept;
    bool is_block_file(std::error_code& ec) const noexcept;
    bool is_character_file(std::error_code& ec) const noexcept;
    bool is_directory(std::error_code& ec) const noexcept;
    bool is_fifo(std::error_code& ec) const noexcept;
    bool is_other(std::error_code& ec) const noexcept;
    bool is_regular_file(std::error_code& ec) const noexcept;
    bool is_socket(std::error_code& ec) const noexcept;
    bool is_symlink(std::error_code& ec) const noexcept;
    uintmax_t file_size(std::error_code& ec) const noexcept;
    file_time_type last_write_time(std::error_code& ec) const noexcept;
    file_status status(std::error_code& ec) const noexcept;
    file_status symlink_status(std::error_code& ec) const noexcept;

#ifndef GHC_OS_WEB
#ifdef GHC_WITH_EXCEPTIONS
    uintmax_t hard_link_count() const;
#endif
    uintmax_t hard_link_count(std::error_code& ec) const noexcept;
#endif

#ifdef GHC_HAS_THREEWAY_COMP
    std::strong_ordering operator<=>(const directory_entry& rhs) const noexcept;
#endif
    bool operator<(const directory_entry& rhs) const noexcept;
    bool operator==(const directory_entry& rhs) const noexcept;
    bool operator!=(const directory_entry& rhs) const noexcept;
    bool operator<=(const directory_entry& rhs) const noexcept;
    bool operator>(const directory_entry& rhs) const noexcept;
    bool operator>=(const directory_entry& rhs) const noexcept;

private:
    friend class directory_iterator;
#ifdef GHC_WITH_EXCEPTIONS
    file_type status_file_type() const;
#endif
    file_type status_file_type(std::error_code& ec) const noexcept;
    filesystem::path _path;
    file_status _status;
    file_status _symlink_status;
    uintmax_t _file_size = static_cast<uintmax_t>(-1);
#ifndef GHC_OS_WINDOWS
    uintmax_t _hard_link_count = static_cast<uintmax_t>(-1);
#endif
    time_t _last_write_time = 0;
};

// [fs.class.directory.iterator] Class directory_iterator
class GHC_FS_API_CLASS directory_iterator
{
public:
    class GHC_FS_API_CLASS proxy
    {
    public:
        const directory_entry& operator*() const& noexcept { return _dir_entry; }
        directory_entry operator*() && noexcept { return std::move(_dir_entry); }

    private:
        explicit proxy(const directory_entry& dir_entry)
            : _dir_entry(dir_entry)
        {
        }
        friend class directory_iterator;
        friend class recursive_directory_iterator;
        directory_entry _dir_entry;
    };
    using iterator_category = std::input_iterator_tag;
    using value_type = directory_entry;
    using difference_type = std::ptrdiff_t;
    using pointer = const directory_entry*;
    using reference = const directory_entry&;

    // [fs.dir.itr.members] member functions
    directory_iterator() noexcept;
#ifdef GHC_WITH_EXCEPTIONS
    explicit directory_iterator(const path& p);
    directory_iterator(const path& p, directory_options options);
#endif
    directory_iterator(const path& p, std::error_code& ec) noexcept;
    directory_iterator(const path& p, directory_options options, std::error_code& ec) noexcept;
    directory_iterator(const directory_iterator& rhs);
    directory_iterator(directory_iterator&& rhs) noexcept;
    ~directory_iterator();
    directory_iterator& operator=(const directory_iterator& rhs);
    directory_iterator& operator=(directory_iterator&& rhs) noexcept;
    const directory_entry& operator*() const;
    const directory_entry* operator->() const;
#ifdef GHC_WITH_EXCEPTIONS
    directory_iterator& operator++();
#endif
    directory_iterator& increment(std::error_code& ec) noexcept;

    // other members as required by [input.iterators]
#ifdef GHC_WITH_EXCEPTIONS
    proxy operator++(int)
    {
        proxy p{**this};
        ++*this;
        return p;
    }
#endif
    bool operator==(const directory_iterator& rhs) const;
    bool operator!=(const directory_iterator& rhs) const;

private:
    friend class recursive_directory_iterator;
    class impl;
    std::shared_ptr<impl> _impl;
};

// [fs.dir.itr.nonmembers] directory_iterator non-member functions
GHC_FS_API directory_iterator begin(directory_iterator iter) noexcept;
GHC_FS_API directory_iterator end(const directory_iterator&) noexcept;

// [fs.class.re.dir.itr] class recursive_directory_iterator
class GHC_FS_API_CLASS recursive_directory_iterator
{
public:
    using iterator_category = std::input_iterator_tag;
    using value_type = directory_entry;
    using difference_type = std::ptrdiff_t;
    using pointer = const directory_entry*;
    using reference = const directory_entry&;

    // [fs.rec.dir.itr.members] constructors and destructor
    recursive_directory_iterator() noexcept;
#ifdef GHC_WITH_EXCEPTIONS
    explicit recursive_directory_iterator(const path& p);
    recursive_directory_iterator(const path& p, directory_options options);
#endif
    recursive_directory_iterator(const path& p, directory_options options, std::error_code& ec) noexcept;
    recursive_directory_iterator(const path& p, std::error_code& ec) noexcept;
    recursive_directory_iterator(const recursive_directory_iterator& rhs);
    recursive_directory_iterator(recursive_directory_iterator&& rhs) noexcept;
    ~recursive_directory_iterator();

    // [fs.rec.dir.itr.members] observers
    directory_options options() const;
    int depth() const;
    bool recursion_pending() const;

    const directory_entry& operator*() const;
    const directory_entry* operator->() const;

    // [fs.rec.dir.itr.members] modifiers recursive_directory_iterator&
    recursive_directory_iterator& operator=(const recursive_directory_iterator& rhs);
    recursive_directory_iterator& operator=(recursive_directory_iterator&& rhs) noexcept;
#ifdef GHC_WITH_EXCEPTIONS
    recursive_directory_iterator& operator++();
#endif
    recursive_directory_iterator& increment(std::error_code& ec) noexcept;

#ifdef GHC_WITH_EXCEPTIONS
    void pop();
#endif
    void pop(std::error_code& ec);
    void disable_recursion_pending();

    // other members as required by [input.iterators]
#ifdef GHC_WITH_EXCEPTIONS
    directory_iterator::proxy operator++(int)
    {
        directory_iterator::proxy proxy{**this};
        ++*this;
        return proxy;
    }
#endif
    bool operator==(const recursive_directory_iterator& rhs) const;
    bool operator!=(const recursive_directory_iterator& rhs) const;

private:
    struct recursive_directory_iterator_impl
    {
        directory_options _options;
        bool _recursion_pending;
        std::stack<directory_iterator> _dir_iter_stack;
        recursive_directory_iterator_impl(directory_options options, bool recursion_pending)
            : _options(options)
            , _recursion_pending(recursion_pending)
        {
        }
    };
    std::shared_ptr<recursive_directory_iterator_impl> _impl;
};

// [fs.rec.dir.itr.nonmembers] directory_iterator non-member functions
GHC_FS_API recursive_directory_iterator begin(recursive_directory_iterator iter) noexcept;
GHC_FS_API recursive_directory_iterator end(const recursive_directory_iterator&) noexcept;

// [fs.op.funcs] filesystem operations
#ifdef GHC_WITH_EXCEPTIONS
GHC_FS_API path absolute(const path& p);
GHC_FS_API path canonical(const path& p);
GHC_FS_API void copy(const path& from, const path& to);
GHC_FS_API void copy(const path& from, const path& to, copy_options options);
GHC_FS_API bool copy_file(const path& from, const path& to);
GHC_FS_API bool copy_file(const path& from, const path& to, copy_options option);
GHC_FS_API void copy_symlink(const path& existing_symlink, const path& new_symlink);
GHC_FS_API bool create_directories(const path& p);
GHC_FS_API bool create_directory(const path& p);
GHC_FS_API bool create_directory(const path& p, const path& attributes);
GHC_FS_API void create_directory_symlink(const path& to, const path& new_symlink);
GHC_FS_API void create_symlink(const path& to, const path& new_symlink);
GHC_FS_API path current_path();
GHC_FS_API void current_path(const path& p);
GHC_FS_API bool exists(const path& p);
GHC_FS_API bool equivalent(const path& p1, const path& p2);
GHC_FS_API uintmax_t file_size(const path& p);
GHC_FS_API bool is_block_file(const path& p);
GHC_FS_API bool is_character_file(const path& p);
GHC_FS_API bool is_directory(const path& p);
GHC_FS_API bool is_empty(const path& p);
GHC_FS_API bool is_fifo(const path& p);
GHC_FS_API bool is_other(const path& p);
GHC_FS_API bool is_regular_file(const path& p);
GHC_FS_API bool is_socket(const path& p);
GHC_FS_API bool is_symlink(const path& p);
GHC_FS_API file_time_type last_write_time(const path& p);
GHC_FS_API void last_write_time(const path& p, file_time_type new_time);
GHC_FS_API void permissions(const path& p, perms prms, perm_options opts = perm_options::replace);
GHC_FS_API path proximate(const path& p, const path& base = current_path());
GHC_FS_API path read_symlink(const path& p);
GHC_FS_API path relative(const path& p, const path& base = current_path());
GHC_FS_API bool remove(const path& p);
GHC_FS_API uintmax_t remove_all(const path& p);
GHC_FS_API void rename(const path& from, const path& to);
GHC_FS_API void resize_file(const path& p, uintmax_t size);
GHC_FS_API space_info space(const path& p);
GHC_FS_API file_status status(const path& p);
GHC_FS_API file_status symlink_status(const path& p);
GHC_FS_API path temp_directory_path();
GHC_FS_API path weakly_canonical(const path& p);
#endif
GHC_FS_API path absolute(const path& p, std::error_code& ec);
GHC_FS_API path canonical(const path& p, std::error_code& ec);
GHC_FS_API void copy(const path& from, const path& to, std::error_code& ec) noexcept;
GHC_FS_API void copy(const path& from, const path& to, copy_options options, std::error_code& ec) noexcept;
GHC_FS_API bool copy_file(const path& from, const path& to, std::error_code& ec) noexcept;
GHC_FS_API bool copy_file(const path& from, const path& to, copy_options option, std::error_code& ec) noexcept;
GHC_FS_API void copy_symlink(const path& existing_symlink, const path& new_symlink, std::error_code& ec) noexcept;
GHC_FS_API bool create_directories(const path& p, std::error_code& ec) noexcept;
GHC_FS_API bool create_directory(const path& p, std::error_code& ec) noexcept;
GHC_FS_API bool create_directory(const path& p, const path& attributes, std::error_code& ec) noexcept;
GHC_FS_API void create_directory_symlink(const path& to, const path& new_symlink, std::error_code& ec) noexcept;
GHC_FS_API void create_symlink(const path& to, const path& new_symlink, std::error_code& ec) noexcept;
GHC_FS_API path current_path(std::error_code& ec);
GHC_FS_API void current_path(const path& p, std::error_code& ec) noexcept;
GHC_FS_API bool exists(file_status s) noexcept;
GHC_FS_API bool exists(const path& p, std::error_code& ec) noexcept;
GHC_FS_API bool equivalent(const path& p1, const path& p2, std::error_code& ec) noexcept;
GHC_FS_API uintmax_t file_size(const path& p, std::error_code& ec) noexcept;
GHC_FS_API bool is_block_file(file_status s) noexcept;
GHC_FS_API bool is_block_file(const path& p, std::error_code& ec) noexcept;
GHC_FS_API bool is_character_file(file_status s) noexcept;
GHC_FS_API bool is_character_file(const path& p, std::error_code& ec) noexcept;
GHC_FS_API bool is_directory(file_status s) noexcept;
GHC_FS_API bool is_directory(const path& p, std::error_code& ec) noexcept;
GHC_FS_API bool is_empty(const path& p, std::error_code& ec) noexcept;
GHC_FS_API bool is_fifo(file_status s) noexcept;
GHC_FS_API bool is_fifo(const path& p, std::error_code& ec) noexcept;
GHC_FS_API bool is_other(file_status s) noexcept;
GHC_FS_API bool is_other(const path& p, std::error_code& ec) noexcept;
GHC_FS_API bool is_regular_file(file_status s) noexcept;
GHC_FS_API bool is_regular_file(const path& p, std::error_code& ec) noexcept;
GHC_FS_API bool is_socket(file_status s) noexcept;
GHC_FS_API bool is_socket(const path& p, std::error_code& ec) noexcept;
GHC_FS_API bool is_symlink(file_status s) noexcept;
GHC_FS_API bool is_symlink(const path& p, std::error_code& ec) noexcept;
GHC_FS_API file_time_type last_write_time(const path& p, std::error_code& ec) noexcept;
GHC_FS_API void last_write_time(const path& p, file_time_type new_time, std::error_code& ec) noexcept;
GHC_FS_API void permissions(const path& p, perms prms, std::error_code& ec) noexcept;
GHC_FS_API void permissions(const path& p, perms prms, perm_options opts, std::error_code& ec) noexcept;
GHC_FS_API path proximate(const path& p, std::error_code& ec);
GHC_FS_API path proximate(const path& p, const path& base, std::error_code& ec);
GHC_FS_API path read_symlink(const path& p, std::error_code& ec);
GHC_FS_API path relative(const path& p, std::error_code& ec);
GHC_FS_API path relative(const path& p, const path& base, std::error_code& ec);
GHC_FS_API bool remove(const path& p, std::error_code& ec) noexcept;
GHC_FS_API uintmax_t remove_all(const path& p, std::error_code& ec) noexcept;
GHC_FS_API void rename(const path& from, const path& to, std::error_code& ec) noexcept;
GHC_FS_API void resize_file(const path& p, uintmax_t size, std::error_code& ec) noexcept;
GHC_FS_API space_info space(const path& p, std::error_code& ec) noexcept;
GHC_FS_API file_status status(const path& p, std::error_code& ec) noexcept;
GHC_FS_API bool status_known(file_status s) noexcept;
GHC_FS_API file_status symlink_status(const path& p, std::error_code& ec) noexcept;
GHC_FS_API path temp_directory_path(std::error_code& ec) noexcept;
GHC_FS_API path weakly_canonical(const path& p, std::error_code& ec) noexcept;

#ifndef GHC_OS_WEB
#ifdef GHC_WITH_EXCEPTIONS
GHC_FS_API void create_hard_link(const path& to, const path& new_hard_link);
GHC_FS_API uintmax_t hard_link_count(const path& p);
#endif
GHC_FS_API void create_hard_link(const path& to, const path& new_hard_link, std::error_code& ec) noexcept;
GHC_FS_API uintmax_t hard_link_count(const path& p, std::error_code& ec) noexcept;
#endif

// Non-C++17 add-on std::fstream wrappers with path
template <class charT, class traits = std::char_traits<charT>>
class basic_filebuf : public std::basic_filebuf<charT, traits>
{
public:
    basic_filebuf() {}
    ~basic_filebuf() override {}
    basic_filebuf(const basic_filebuf&) = delete;
    const basic_filebuf& operator=(const basic_filebuf&) = delete;
    basic_filebuf<charT, traits>* open(const path& p, std::ios_base::openmode mode)
    {
#if defined(GHC_OS_WINDOWS) && !defined(__GLIBCXX__)
        return std::basic_filebuf<charT, traits>::open(p.wstring().c_str(), mode) ? this : 0;
#else
        return std::basic_filebuf<charT, traits>::open(p.string().c_str(), mode) ? this : 0;
#endif
    }
};

template <class charT, class traits = std::char_traits<charT>>
class basic_ifstream : public std::basic_ifstream<charT, traits>
{
public:
    basic_ifstream() {}
#if defined(GHC_OS_WINDOWS) && !defined(__GLIBCXX__)
    explicit basic_ifstream(const path& p, std::ios_base::openmode mode = std::ios_base::in)
        : std::basic_ifstream<charT, traits>(p.wstring().c_str(), mode)
    {
    }
    void open(const path& p, std::ios_base::openmode mode = std::ios_base::in) { std::basic_ifstream<charT, traits>::open(p.wstring().c_str(), mode); }
#else
    explicit basic_ifstream(const path& p, std::ios_base::openmode mode = std::ios_base::in)
        : std::basic_ifstream<charT, traits>(p.string().c_str(), mode)
    {
    }
    void open(const path& p, std::ios_base::openmode mode = std::ios_base::in) { std::basic_ifstream<charT, traits>::open(p.string().c_str(), mode); }
#endif
    basic_ifstream(const basic_ifstream&) = delete;
    const basic_ifstream& operator=(const basic_ifstream&) = delete;
    ~basic_ifstream() override {}
};

template <class charT, class traits = std::char_traits<charT>>
class basic_ofstream : public std::basic_ofstream<charT, traits>
{
public:
    basic_ofstream() {}
#if defined(GHC_OS_WINDOWS) && !defined(__GLIBCXX__)
    explicit basic_ofstream(const path& p, std::ios_base::openmode mode = std::ios_base::out)
        : std::basic_ofstream<charT, traits>(p.wstring().c_str(), mode)
    {
    }
    void open(const path& p, std::ios_base::openmode mode = std::ios_base::out) { std::basic_ofstream<charT, traits>::open(p.wstring().c_str(), mode); }
#else
    explicit basic_ofstream(const path& p, std::ios_base::openmode mode = std::ios_base::out)
        : std::basic_ofstream<charT, traits>(p.string().c_str(), mode)
    {
    }
    void open(const path& p, std::ios_base::openmode mode = std::ios_base::out) { std::basic_ofstream<charT, traits>::open(p.string().c_str(), mode); }
#endif
    basic_ofstream(const basic_ofstream&) = delete;
    const basic_ofstream& operator=(const basic_ofstream&) = delete;
    ~basic_ofstream() override {}
};

template <class charT, class traits = std::char_traits<charT>>
class basic_fstream : public std::basic_fstream<charT, traits>
{
public:
    basic_fstream() {}
#if defined(GHC_OS_WINDOWS) && !defined(__GLIBCXX__)
    explicit basic_fstream(const path& p, std::ios_base::openmode mode = std::ios_base::in | std::ios_base::out)
        : std::basic_fstream<charT, traits>(p.wstring().c_str(), mode)
    {
    }
    void open(const path& p, std::ios_base::openmode mode = std::ios_base::in | std::ios_base::out) { std::basic_fstream<charT, traits>::open(p.wstring().c_str(), mode); }
#else
    explicit basic_fstream(const path& p, std::ios_base::openmode mode = std::ios_base::in | std::ios_base::out)
        : std::basic_fstream<charT, traits>(p.string().c_str(), mode)
    {
    }
    void open(const path& p, std::ios_base::openmode mode = std::ios_base::in | std::ios_base::out) { std::basic_fstream<charT, traits>::open(p.string().c_str(), mode); }
#endif
    basic_fstream(const basic_fstream&) = delete;
    const basic_fstream& operator=(const basic_fstream&) = delete;
    ~basic_fstream() override {}
};

typedef basic_filebuf<char> filebuf;
typedef basic_filebuf<wchar_t> wfilebuf;
typedef basic_ifstream<char> ifstream;
typedef basic_ifstream<wchar_t> wifstream;
typedef basic_ofstream<char> ofstream;
typedef basic_ofstream<wchar_t> wofstream;
typedef basic_fstream<char> fstream;
typedef basic_fstream<wchar_t> wfstream;

class GHC_FS_API_CLASS u8arguments
{
public:
    u8arguments(int& argc, char**& argv);
    ~u8arguments()
    {
        _refargc = _argc;
        _refargv = _argv;
    }

    bool valid() const { return _isvalid; }

private:
    int _argc;
    char** _argv;
    int& _refargc;
    char**& _refargv;
    bool _isvalid;
#ifdef GHC_OS_WINDOWS
    std::vector<std::string> _args;
    std::vector<char*> _argp;
#endif
};

//-------------------------------------------------------------------------------------------------
//  Implementation
//-------------------------------------------------------------------------------------------------

namespace detail {
enum utf8_states_t { S_STRT = 0, S_RJCT = 8 };
GHC_FS_API void appendUTF8(std::string& str, uint32_t unicode);
GHC_FS_API bool is_surrogate(uint32_t c);
GHC_FS_API bool is_high_surrogate(uint32_t c);
GHC_FS_API bool is_low_surrogate(uint32_t c);
GHC_FS_API unsigned consumeUtf8Fragment(const unsigned state, const uint8_t fragment, uint32_t& codepoint);
enum class portable_error {
    none = 0,
    exists,
    not_found,
    not_supported,
    not_implemented,
    invalid_argument,
    is_a_directory,
};
GHC_FS_API std::error_code make_error_code(portable_error err);
#ifdef GHC_OS_WINDOWS
GHC_FS_API std::error_code make_system_error(uint32_t err = 0);
#else
GHC_FS_API std::error_code make_system_error(int err = 0);

template <typename T, typename = int>
struct has_d_type : std::false_type{};

template <typename T>
struct has_d_type<T, decltype((void)T::d_type, 0)> : std::true_type {};

template <typename T>
GHC_INLINE file_type file_type_from_dirent_impl(const T&, std::false_type)
{
    return file_type::none;
}

template <typename T>
GHC_INLINE file_type file_type_from_dirent_impl(const T& t, std::true_type)
{
    switch (t.d_type) {
#ifdef DT_BLK
        case DT_BLK:
            return file_type::block;
#endif
#ifdef DT_CHR
        case DT_CHR:
            return file_type::character;
#endif
#ifdef DT_DIR
        case DT_DIR:
            return file_type::directory;
#endif
#ifdef DT_FIFO
        case DT_FIFO:
            return file_type::fifo;
#endif
#ifdef DT_LNK
        case DT_LNK:
            return file_type::symlink;
#endif
#ifdef DT_REG
        case DT_REG:
            return file_type::regular;
#endif
#ifdef DT_SOCK
        case DT_SOCK:
            return file_type::socket;
#endif
#ifdef DT_UNKNOWN
        case DT_UNKNOWN:
            return file_type::none;
#endif
        default:
            return file_type::unknown;
    }
}

template <class T>
GHC_INLINE file_type file_type_from_dirent(const T& t)
{
    return file_type_from_dirent_impl(t, has_d_type<T>{});
}
#endif
}  // namespace detail

namespace detail {

#ifdef GHC_EXPAND_IMPL

GHC_INLINE std::error_code make_error_code(portable_error err)
{
#ifdef GHC_OS_WINDOWS
    switch (err) {
        case portable_error::none:
            return std::error_code();
        case portable_error::exists:
            return std::error_code(ERROR_ALREADY_EXISTS, std::system_category());
        case portable_error::not_found:
            return std::error_code(ERROR_PATH_NOT_FOUND, std::system_category());
        case portable_error::not_supported:
            return std::error_code(ERROR_NOT_SUPPORTED, std::system_category());
        case portable_error::not_implemented:
            return std::error_code(ERROR_CALL_NOT_IMPLEMENTED, std::system_category());
        case portable_error::invalid_argument:
            return std::error_code(ERROR_INVALID_PARAMETER, std::system_category());
        case portable_error::is_a_directory:
#ifdef ERROR_DIRECTORY_NOT_SUPPORTED
            return std::error_code(ERROR_DIRECTORY_NOT_SUPPORTED, std::system_category());
#else
            return std::error_code(ERROR_NOT_SUPPORTED, std::system_category());
#endif
    }
#else
    switch (err) {
        case portable_error::none:
            return std::error_code();
        case portable_error::exists:
            return std::error_code(EEXIST, std::system_category());
        case portable_error::not_found:
            return std::error_code(ENOENT, std::system_category());
        case portable_error::not_supported:
            return std::error_code(ENOTSUP, std::system_category());
        case portable_error::not_implemented:
            return std::error_code(ENOSYS, std::system_category());
        case portable_error::invalid_argument:
            return std::error_code(EINVAL, std::system_category());
        case portable_error::is_a_directory:
            return std::error_code(EISDIR, std::system_category());
    }
#endif
    return std::error_code();
}

#ifdef GHC_OS_WINDOWS
GHC_INLINE std::error_code make_system_error(uint32_t err)
{
    return std::error_code(err ? static_cast<int>(err) : static_cast<int>(::GetLastError()), std::system_category());
}
#else
GHC_INLINE std::error_code make_system_error(int err)
{
    return std::error_code(err ? err : errno, std::system_category());
}
#endif

#endif  // GHC_EXPAND_IMPL

template <typename Enum>
using EnableBitmask = typename std::enable_if<std::is_same<Enum, perms>::value || std::is_same<Enum, perm_options>::value || std::is_same<Enum, copy_options>::value || std::is_same<Enum, directory_options>::value, Enum>::type;
}  // namespace detail

template <typename Enum>
constexpr detail::EnableBitmask<Enum> operator&(Enum X, Enum Y)
{
    using underlying = typename std::underlying_type<Enum>::type;
    return static_cast<Enum>(static_cast<underlying>(X) & static_cast<underlying>(Y));
}

template <typename Enum>
constexpr detail::EnableBitmask<Enum> operator|(Enum X, Enum Y)
{
    using underlying = typename std::underlying_type<Enum>::type;
    return static_cast<Enum>(static_cast<underlying>(X) | static_cast<underlying>(Y));
}

template <typename Enum>
constexpr detail::EnableBitmask<Enum> operator^(Enum X, Enum Y)
{
    using underlying = typename std::underlying_type<Enum>::type;
    return static_cast<Enum>(static_cast<underlying>(X) ^ static_cast<underlying>(Y));
}

template <typename Enum>
constexpr detail::EnableBitmask<Enum> operator~(Enum X)
{
    using underlying = typename std::underlying_type<Enum>::type;
    return static_cast<Enum>(~static_cast<underlying>(X));
}

template <typename Enum>
detail::EnableBitmask<Enum>& operator&=(Enum& X, Enum Y)
{
    X = X & Y;
    return X;
}

template <typename Enum>
detail::EnableBitmask<Enum>& operator|=(Enum& X, Enum Y)
{
    X = X | Y;
    return X;
}

template <typename Enum>
detail::EnableBitmask<Enum>& operator^=(Enum& X, Enum Y)
{
    X = X ^ Y;
    return X;
}

#ifdef GHC_EXPAND_IMPL

namespace detail {

GHC_INLINE bool in_range(uint32_t c, uint32_t lo, uint32_t hi)
{
    return (static_cast<uint32_t>(c - lo) < (hi - lo + 1));
}

GHC_INLINE bool is_surrogate(uint32_t c)
{
    return in_range(c, 0xd800, 0xdfff);
}

GHC_INLINE bool is_high_surrogate(uint32_t c)
{
    return (c & 0xfffffc00) == 0xd800;
}

GHC_INLINE bool is_low_surrogate(uint32_t c)
{
    return (c & 0xfffffc00) == 0xdc00;
}

GHC_INLINE void appendUTF8(std::string& str, uint32_t unicode)
{
    if (unicode <= 0x7f) {
        str.push_back(static_cast<char>(unicode));
    }
    else if (unicode >= 0x80 && unicode <= 0x7ff) {
        str.push_back(static_cast<char>((unicode >> 6) + 192));
        str.push_back(static_cast<char>((unicode & 0x3f) + 128));
    }
    else if ((unicode >= 0x800 && unicode <= 0xd7ff) || (unicode >= 0xe000 && unicode <= 0xffff)) {
        str.push_back(static_cast<char>((unicode >> 12) + 224));
        str.push_back(static_cast<char>(((unicode & 0xfff) >> 6) + 128));
        str.push_back(static_cast<char>((unicode & 0x3f) + 128));
    }
    else if (unicode >= 0x10000 && unicode <= 0x10ffff) {
        str.push_back(static_cast<char>((unicode >> 18) + 240));
        str.push_back(static_cast<char>(((unicode & 0x3ffff) >> 12) + 128));
        str.push_back(static_cast<char>(((unicode & 0xfff) >> 6) + 128));
        str.push_back(static_cast<char>((unicode & 0x3f) + 128));
    }
    else {
#ifdef GHC_RAISE_UNICODE_ERRORS
        throw filesystem_error("Illegal code point for unicode character.", str, std::make_error_code(std::errc::illegal_byte_sequence));
#else
        appendUTF8(str, 0xfffd);
#endif
    }
}

// Thanks to Bjoern Hoehrmann (https://bjoern.hoehrmann.de/utf-8/decoder/dfa/)
// and Taylor R Campbell for the ideas to this DFA approach of UTF-8 decoding;
// Generating debugging and shrinking my own DFA from scratch was a day of fun!
GHC_INLINE unsigned consumeUtf8Fragment(const unsigned state, const uint8_t fragment, uint32_t& codepoint)
{
    static const uint32_t utf8_state_info[] = {
        // encoded states
        0x11111111u, 0x11111111u, 0x77777777u, 0x77777777u, 0x88888888u, 0x88888888u, 0x88888888u, 0x88888888u, 0x22222299u, 0x22222222u, 0x22222222u, 0x22222222u, 0x3333333au, 0x33433333u, 0x9995666bu, 0x99999999u,
        0x88888880u, 0x22818108u, 0x88888881u, 0x88888882u, 0x88888884u, 0x88888887u, 0x88888886u, 0x82218108u, 0x82281108u, 0x88888888u, 0x88888883u, 0x88888885u, 0u,          0u,          0u,          0u,
    };
    uint8_t category = fragment < 128 ? 0 : (utf8_state_info[(fragment >> 3) & 0xf] >> ((fragment & 7) << 2)) & 0xf;
    codepoint = (state ? (codepoint << 6) | (fragment & 0x3fu) : (0xffu >> category) & fragment);
    return state == S_RJCT ? static_cast<unsigned>(S_RJCT) : static_cast<unsigned>((utf8_state_info[category + 16] >> (state << 2)) & 0xf);
}

GHC_INLINE bool validUtf8(const std::string& utf8String)
{
    std::string::const_iterator iter = utf8String.begin();
    unsigned utf8_state = S_STRT;
    std::uint32_t codepoint = 0;
    while (iter < utf8String.end()) {
        if ((utf8_state = consumeUtf8Fragment(utf8_state, static_cast<uint8_t>(*iter++), codepoint)) == S_RJCT) {
            return false;
        }
    }
    if (utf8_state) {
        return false;
    }
    return true;
}

}  // namespace detail

#endif

namespace detail {

template <class StringType, class Utf8String, typename std::enable_if<path::_is_basic_string<Utf8String>::value && (sizeof(typename Utf8String::value_type) == 1) && (sizeof(typename StringType::value_type) == 1)>::type* = nullptr>
inline StringType fromUtf8(const Utf8String& utf8String, const typename StringType::allocator_type& alloc = typename StringType::allocator_type())
{
    return StringType(utf8String.begin(), utf8String.end(), alloc);
}

template <class StringType, class Utf8String, typename std::enable_if<path::_is_basic_string<Utf8String>::value && (sizeof(typename Utf8String::value_type) == 1) && (sizeof(typename StringType::value_type) == 2)>::type* = nullptr>
inline StringType fromUtf8(const Utf8String& utf8String, const typename StringType::allocator_type& alloc = typename StringType::allocator_type())
{
    StringType result(alloc);
    result.reserve(utf8String.length());
    auto iter = utf8String.cbegin();
    unsigned utf8_state = S_STRT;
    std::uint32_t codepoint = 0;
    while (iter < utf8String.cend()) {
        if ((utf8_state = consumeUtf8Fragment(utf8_state, static_cast<uint8_t>(*iter++), codepoint)) == S_STRT) {
            if (codepoint <= 0xffff) {
                result += static_cast<typename StringType::value_type>(codepoint);
            }
            else {
                codepoint -= 0x10000;
                result += static_cast<typename StringType::value_type>((codepoint >> 10) + 0xd800);
                result += static_cast<typename StringType::value_type>((codepoint & 0x3ff) + 0xdc00);
            }
            codepoint = 0;
        }
        else if (utf8_state == S_RJCT) {
#ifdef GHC_RAISE_UNICODE_ERRORS
            throw filesystem_error("Illegal byte sequence for unicode character.", utf8String, std::make_error_code(std::errc::illegal_byte_sequence));
#else
            result += static_cast<typename StringType::value_type>(0xfffd);
            utf8_state = S_STRT;
            codepoint = 0;
#endif
        }
    }
    if (utf8_state) {
#ifdef GHC_RAISE_UNICODE_ERRORS
        throw filesystem_error("Illegal byte sequence for unicode character.", utf8String, std::make_error_code(std::errc::illegal_byte_sequence));
#else
        result += static_cast<typename StringType::value_type>(0xfffd);
#endif
    }
    return result;
}

template <class StringType, class Utf8String, typename std::enable_if<path::_is_basic_string<Utf8String>::value && (sizeof(typename Utf8String::value_type) == 1) && (sizeof(typename StringType::value_type) == 4)>::type* = nullptr>
inline StringType fromUtf8(const Utf8String& utf8String, const typename StringType::allocator_type& alloc = typename StringType::allocator_type())
{
    StringType result(alloc);
    result.reserve(utf8String.length());
    auto iter = utf8String.cbegin();
    unsigned utf8_state = S_STRT;
    std::uint32_t codepoint = 0;
    while (iter < utf8String.cend()) {
        if ((utf8_state = consumeUtf8Fragment(utf8_state, static_cast<uint8_t>(*iter++), codepoint)) == S_STRT) {
            result += static_cast<typename StringType::value_type>(codepoint);
            codepoint = 0;
        }
        else if (utf8_state == S_RJCT) {
#ifdef GHC_RAISE_UNICODE_ERRORS
            throw filesystem_error("Illegal byte sequence for unicode character.", utf8String, std::make_error_code(std::errc::illegal_byte_sequence));
#else
            result += static_cast<typename StringType::value_type>(0xfffd);
            utf8_state = S_STRT;
            codepoint = 0;
#endif
        }
    }
    if (utf8_state) {
#ifdef GHC_RAISE_UNICODE_ERRORS
        throw filesystem_error("Illegal byte sequence for unicode character.", utf8String, std::make_error_code(std::errc::illegal_byte_sequence));
#else
        result += static_cast<typename StringType::value_type>(0xfffd);
#endif
    }
    return result;
}

template <class StringType, typename charT, std::size_t N>
inline StringType fromUtf8(const charT (&utf8String)[N])
{
#ifdef GHC_WITH_STRING_VIEW
    return fromUtf8<StringType>(basic_string_view<charT>(utf8String, N - 1));
#else
    return fromUtf8<StringType>(std::basic_string<charT>(utf8String, N - 1));
#endif
}

template <typename strT, typename std::enable_if<path::_is_basic_string<strT>::value && (sizeof(typename strT::value_type) == 1), int>::type size = 1>
inline std::string toUtf8(const strT& unicodeString)
{
    return std::string(unicodeString.begin(), unicodeString.end());
}

template <typename strT, typename std::enable_if<path::_is_basic_string<strT>::value && (sizeof(typename strT::value_type) == 2), int>::type size = 2>
inline std::string toUtf8(const strT& unicodeString)
{
    std::string result;
    for (auto iter = unicodeString.begin(); iter != unicodeString.end(); ++iter) {
        char32_t c = *iter;
        if (is_surrogate(c)) {
            ++iter;
            if (iter != unicodeString.end() && is_high_surrogate(c) && is_low_surrogate(*iter)) {
                appendUTF8(result, (char32_t(c) << 10) + *iter - 0x35fdc00);
            }
            else {
#ifdef GHC_RAISE_UNICODE_ERRORS
                throw filesystem_error("Illegal code point for unicode character.", result, std::make_error_code(std::errc::illegal_byte_sequence));
#else
                appendUTF8(result, 0xfffd);
                if (iter == unicodeString.end()) {
                    break;
                }
#endif
            }
        }
        else {
            appendUTF8(result, c);
        }
    }
    return result;
}

template <typename strT, typename std::enable_if<path::_is_basic_string<strT>::value && (sizeof(typename strT::value_type) == 4), int>::type size = 4>
inline std::string toUtf8(const strT& unicodeString)
{
    std::string result;
    for (auto c : unicodeString) {
        appendUTF8(result, static_cast<uint32_t>(c));
    }
    return result;
}

template <typename charT>
inline std::string toUtf8(const charT* unicodeString)
{
#ifdef GHC_WITH_STRING_VIEW
    return toUtf8(basic_string_view<charT, std::char_traits<charT>>(unicodeString));
#else
    return toUtf8(std::basic_string<charT, std::char_traits<charT>>(unicodeString));
#endif
}

#ifdef GHC_USE_WCHAR_T
template <class StringType, class WString, typename std::enable_if<path::_is_basic_string<WString>::value && (sizeof(typename WString::value_type) == 2) && (sizeof(typename StringType::value_type) == 1), bool>::type = false>
inline StringType fromWChar(const WString& wString, const typename StringType::allocator_type& alloc = typename StringType::allocator_type())
{
    auto temp = toUtf8(wString);
    return StringType(temp.begin(), temp.end(), alloc);
}

template <class StringType, class WString, typename std::enable_if<path::_is_basic_string<WString>::value && (sizeof(typename WString::value_type) == 2) && (sizeof(typename StringType::value_type) == 2), bool>::type = false>
inline StringType fromWChar(const WString& wString, const typename StringType::allocator_type& alloc = typename StringType::allocator_type())
{
    return StringType(wString.begin(), wString.end(), alloc);
}

template <class StringType, class WString, typename std::enable_if<path::_is_basic_string<WString>::value && (sizeof(typename WString::value_type) == 2) && (sizeof(typename StringType::value_type) == 4), bool>::type = false>
inline StringType fromWChar(const WString& wString, const typename StringType::allocator_type& alloc = typename StringType::allocator_type())
{
    auto temp = toUtf8(wString);
    return fromUtf8<StringType>(temp, alloc);
}

template <typename strT, typename std::enable_if<path::_is_basic_string<strT>::value && (sizeof(typename strT::value_type) == 1), bool>::type = false>
inline std::wstring toWChar(const strT& unicodeString)
{
    return fromUtf8<std::wstring>(unicodeString);
}

template <typename strT, typename std::enable_if<path::_is_basic_string<strT>::value && (sizeof(typename strT::value_type) == 2), bool>::type = false>
inline std::wstring toWChar(const strT& unicodeString)
{
    return std::wstring(unicodeString.begin(), unicodeString.end());
}

template <typename strT, typename std::enable_if<path::_is_basic_string<strT>::value && (sizeof(typename strT::value_type) == 4), bool>::type = false>
inline std::wstring toWChar(const strT& unicodeString)
{
    auto temp = toUtf8(unicodeString);
    return fromUtf8<std::wstring>(temp);
}

template <typename charT>
inline std::wstring toWChar(const charT* unicodeString)
{
#ifdef GHC_WITH_STRING_VIEW
    return toWChar(basic_string_view<charT, std::char_traits<charT>>(unicodeString));
#else
    return toWChar(std::basic_string<charT, std::char_traits<charT>>(unicodeString));
#endif
}
#endif  // GHC_USE_WCHAR_T

}  // namespace detail

#ifdef GHC_EXPAND_IMPL

namespace detail {

template <typename strT, typename std::enable_if<path::_is_basic_string<strT>::value, bool>::type = true>
GHC_INLINE bool startsWith(const strT& what, const strT& with)
{
    return with.length() <= what.length() && equal(with.begin(), with.end(), what.begin());
}

template <typename strT, typename std::enable_if<path::_is_basic_string<strT>::value, bool>::type = true>
GHC_INLINE bool endsWith(const strT& what, const strT& with)
{
    return with.length() <= what.length() && what.compare(what.length() - with.length(), with.size(), with) == 0;
}

}  // namespace detail

GHC_INLINE void path::check_long_path()
{
#if defined(GHC_OS_WINDOWS) && defined(GHC_WIN_AUTO_PREFIX_LONG_PATH)
    if (is_absolute() && _path.length() >= MAX_PATH - 12 && !detail::startsWith(_path, impl_string_type(GHC_PLATFORM_LITERAL("\\\\?\\")))) {
        postprocess_path_with_format(native_format);
    }
#endif
}

GHC_INLINE void path::postprocess_path_with_format(path::format fmt)
{
#ifdef GHC_RAISE_UNICODE_ERRORS
    if (!detail::validUtf8(_path)) {
        path t;
        t._path = _path;
        throw filesystem_error("Illegal byte sequence for unicode character.", t, std::make_error_code(std::errc::illegal_byte_sequence));
    }
#endif
    switch (fmt) {
#ifdef GHC_OS_WINDOWS
        case path::native_format:
        case path::auto_format:
        case path::generic_format:
            for (auto& c : _path) {
                if (c == generic_separator) {
                    c = preferred_separator;
                }
            }
#ifdef GHC_WIN_AUTO_PREFIX_LONG_PATH
            if (is_absolute() && _path.length() >= MAX_PATH - 12 && !detail::startsWith(_path, impl_string_type(GHC_PLATFORM_LITERAL("\\\\?\\")))) {
                _path = GHC_PLATFORM_LITERAL("\\\\?\\") + _path;
            }
#endif
            handle_prefixes();
            break;
#else
        case path::auto_format:
        case path::native_format:
        case path::generic_format:
            // nothing to do
            break;
#endif
    }
    if (_path.length() > _prefixLength + 2 && _path[_prefixLength] == preferred_separator && _path[_prefixLength + 1] == preferred_separator && _path[_prefixLength + 2] != preferred_separator) {
        impl_string_type::iterator new_end = std::unique(_path.begin() + static_cast<string_type::difference_type>(_prefixLength) + 2, _path.end(), [](path::value_type lhs, path::value_type rhs) { return lhs == rhs && lhs == preferred_separator; });
        _path.erase(new_end, _path.end());
    }
    else {
        impl_string_type::iterator new_end = std::unique(_path.begin() + static_cast<string_type::difference_type>(_prefixLength), _path.end(), [](path::value_type lhs, path::value_type rhs) { return lhs == rhs && lhs == preferred_separator; });
        _path.erase(new_end, _path.end());
    }
}

#endif  // GHC_EXPAND_IMPL

template <class Source, typename>
inline path::path(const Source& source, format fmt)
#ifdef GHC_USE_WCHAR_T
    : _path(detail::toWChar(source))
#else
    : _path(detail::toUtf8(source))
#endif
{
    postprocess_path_with_format(fmt);
}

template <class Source, typename>
inline path u8path(const Source& source)
{
    return path(source);
}
template <class InputIterator>
inline path u8path(InputIterator first, InputIterator last)
{
    return path(first, last);
}

template <class InputIterator>
inline path::path(InputIterator first, InputIterator last, format fmt)
    : path(std::basic_string<typename std::iterator_traits<InputIterator>::value_type>(first, last), fmt)
{
    // delegated
}

#ifdef GHC_EXPAND_IMPL

namespace detail {

GHC_INLINE bool equals_simple_insensitive(const path::value_type* str1, const path::value_type* str2)
{
#ifdef GHC_OS_WINDOWS
#ifdef __GNUC__
    while (::tolower((unsigned char)*str1) == ::tolower((unsigned char)*str2++)) {
        if (*str1++ == 0)
            return true;
    }
    return false;
#else  // __GNUC__
#ifdef GHC_USE_WCHAR_T
    return 0 == ::_wcsicmp(str1, str2);
#else   // GHC_USE_WCHAR_T
    return 0 == ::_stricmp(str1, str2);
#endif  // GHC_USE_WCHAR_T
#endif  // __GNUC__
#else   // GHC_OS_WINDOWS
    return 0 == ::strcasecmp(str1, str2);
#endif  // GHC_OS_WINDOWS
}

GHC_INLINE int compare_simple_insensitive(const path::value_type* str1, size_t len1, const path::value_type* str2, size_t len2)
{
    while (len1 > 0 && len2 > 0 && ::tolower(static_cast<unsigned char>(*str1)) == ::tolower(static_cast<unsigned char>(*str2))) {
        --len1;
        --len2;
        ++str1;
        ++str2;
    }
    if (len1 && len2) {
        return *str1 < *str2 ? -1 : 1;
    }
    if (len1 == 0 && len2 == 0) {
        return 0;
    }
    return len1 == 0 ? -1 : 1;
}

GHC_INLINE const char* strerror_adapter(char* gnu, char*)
{
    return gnu;
}

GHC_INLINE const char* strerror_adapter(int posix, char* buffer)
{
    if (posix) {
        return "Error in strerror_r!";
    }
    return buffer;
}

template <typename ErrorNumber>
GHC_INLINE std::string systemErrorText(ErrorNumber code = 0)
{
#if defined(GHC_OS_WINDOWS)
    LPVOID msgBuf;
    DWORD dw = code ? static_cast<DWORD>(code) : ::GetLastError();
    FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, dw, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPWSTR)&msgBuf, 0, NULL);
    std::string msg = toUtf8(std::wstring((LPWSTR)msgBuf));
    LocalFree(msgBuf);
    return msg;
#else
    char buffer[512];
    return strerror_adapter(strerror_r(code ? code : errno, buffer, sizeof(buffer)), buffer);
#endif
}

#ifdef GHC_OS_WINDOWS
using CreateSymbolicLinkW_fp = BOOLEAN(WINAPI*)(LPCWSTR, LPCWSTR, DWORD);
using CreateHardLinkW_fp = BOOLEAN(WINAPI*)(LPCWSTR, LPCWSTR, LPSECURITY_ATTRIBUTES);

GHC_INLINE void create_symlink(const path& target_name, const path& new_symlink, bool to_directory, std::error_code& ec)
{
    std::error_code tec;
    auto fs = status(target_name, tec);
    if ((fs.type() == file_type::directory && !to_directory) || (fs.type() == file_type::regular && to_directory)) {
        ec = detail::make_error_code(detail::portable_error::not_supported);
        return;
    }
#if defined(__GNUC__) && __GNUC__ >= 8
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-function-type"
#endif
    static CreateSymbolicLinkW_fp api_call = reinterpret_cast<CreateSymbolicLinkW_fp>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "CreateSymbolicLinkW"));
#if defined(__GNUC__) && __GNUC__ >= 8
#pragma GCC diagnostic pop
#endif
    if (api_call) {
        if (api_call(GHC_NATIVEWP(new_symlink), GHC_NATIVEWP(target_name), to_directory ? 1 : 0) == 0) {
            auto result = ::GetLastError();
            if (result == ERROR_PRIVILEGE_NOT_HELD && api_call(GHC_NATIVEWP(new_symlink), GHC_NATIVEWP(target_name), to_directory ? 3 : 2) != 0) {
                return;
            }
            ec = detail::make_system_error(result);
        }
    }
    else {
        ec = detail::make_system_error(ERROR_NOT_SUPPORTED);
    }
}

GHC_INLINE void create_hardlink(const path& target_name, const path& new_hardlink, std::error_code& ec)
{
#if defined(__GNUC__) && __GNUC__ >= 8
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-function-type"
#endif
    static CreateHardLinkW_fp api_call = reinterpret_cast<CreateHardLinkW_fp>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "CreateHardLinkW"));
#if defined(__GNUC__) && __GNUC__ >= 8
#pragma GCC diagnostic pop
#endif
    if (api_call) {
        if (api_call(GHC_NATIVEWP(new_hardlink), GHC_NATIVEWP(target_name), NULL) == 0) {
            ec = detail::make_system_error();
        }
    }
    else {
        ec = detail::make_system_error(ERROR_NOT_SUPPORTED);
    }
}

GHC_INLINE path getFullPathName(const wchar_t* p, std::error_code& ec)
{
    ULONG size = ::GetFullPathNameW(p, 0, 0, 0);
    if (size) {
        std::vector<wchar_t> buf(size, 0);
        ULONG s2 = GetFullPathNameW(p, size, buf.data(), nullptr);
        if (s2 && s2 < size) {
            return path(std::wstring(buf.data(), s2));
        }
    }
    ec = detail::make_system_error();
    return path();
}

#else
GHC_INLINE void create_symlink(const path& target_name, const path& new_symlink, bool, std::error_code& ec)
{
    if (::symlink(target_name.c_str(), new_symlink.c_str()) != 0) {
        ec = detail::make_system_error();
    }
}

#ifndef GHC_OS_WEB
GHC_INLINE void create_hardlink(const path& target_name, const path& new_hardlink, std::error_code& ec)
{
    if (::link(target_name.c_str(), new_hardlink.c_str()) != 0) {
        ec = detail::make_system_error();
    }
}
#endif
#endif

template <typename T>
GHC_INLINE file_status file_status_from_st_mode(T mode)
{
#ifdef GHC_OS_WINDOWS
    file_type ft = file_type::unknown;
    if ((mode & _S_IFDIR) == _S_IFDIR) {
        ft = file_type::directory;
    }
    else if ((mode & _S_IFREG) == _S_IFREG) {
        ft = file_type::regular;
    }
    else if ((mode & _S_IFCHR) == _S_IFCHR) {
        ft = file_type::character;
    }
    perms prms = static_cast<perms>(mode & 0xfff);
    return file_status(ft, prms);
#else
    file_type ft = file_type::unknown;
    if (S_ISDIR(mode)) {
        ft = file_type::directory;
    }
    else if (S_ISREG(mode)) {
        ft = file_type::regular;
    }
    else if (S_ISCHR(mode)) {
        ft = file_type::character;
    }
    else if (S_ISBLK(mode)) {
        ft = file_type::block;
    }
    else if (S_ISFIFO(mode)) {
        ft = file_type::fifo;
    }
    else if (S_ISLNK(mode)) {
        ft = file_type::symlink;
    }
    else if (S_ISSOCK(mode)) {
        ft = file_type::socket;
    }
    perms prms = static_cast<perms>(mode & 0xfff);
    return file_status(ft, prms);
#endif
}

#ifdef GHC_OS_WINDOWS

class unique_handle
{
public:
    typedef HANDLE element_type;

    unique_handle() noexcept
        : _handle(INVALID_HANDLE_VALUE)
    {
    }
    explicit unique_handle(element_type h) noexcept
        : _handle(h)
    {
    }
    unique_handle(unique_handle&& u) noexcept
        : _handle(u.release())
    {
    }
    ~unique_handle() { reset(); }
    unique_handle& operator=(unique_handle&& u) noexcept
    {
        reset(u.release());
        return *this;
    }
    element_type get() const noexcept { return _handle; }
    explicit operator bool() const noexcept { return _handle != INVALID_HANDLE_VALUE; }
    element_type release() noexcept
    {
        element_type tmp = _handle;
        _handle = INVALID_HANDLE_VALUE;
        return tmp;
    }
    void reset(element_type h = INVALID_HANDLE_VALUE) noexcept
    {
        element_type tmp = _handle;
        _handle = h;
        if (tmp != INVALID_HANDLE_VALUE) {
            CloseHandle(tmp);
        }
    }
    void swap(unique_handle& u) noexcept { std::swap(_handle, u._handle); }

private:
    element_type _handle;
};

#ifndef REPARSE_DATA_BUFFER_HEADER_SIZE
typedef struct _REPARSE_DATA_BUFFER
{
    ULONG ReparseTag;
    USHORT ReparseDataLength;
    USHORT Reserved;
    union
    {
        struct
        {
            USHORT SubstituteNameOffset;
            USHORT SubstituteNameLength;
            USHORT PrintNameOffset;
            USHORT PrintNameLength;
            ULONG Flags;
            WCHAR PathBuffer[1];
        } SymbolicLinkReparseBuffer;
        struct
        {
            USHORT SubstituteNameOffset;
            USHORT SubstituteNameLength;
            USHORT PrintNameOffset;
            USHORT PrintNameLength;
            WCHAR PathBuffer[1];
        } MountPointReparseBuffer;
        struct
        {
            UCHAR DataBuffer[1];
        } GenericReparseBuffer;
    } DUMMYUNIONNAME;
} REPARSE_DATA_BUFFER;
#ifndef MAXIMUM_REPARSE_DATA_BUFFER_SIZE
#define MAXIMUM_REPARSE_DATA_BUFFER_SIZE (16 * 1024)
#endif
#endif

template <class T>
struct free_deleter
{
    void operator()(T* p) const { std::free(p); }
};

GHC_INLINE std::unique_ptr<REPARSE_DATA_BUFFER, free_deleter<REPARSE_DATA_BUFFER>> getReparseData(const path& p, std::error_code& ec)
{
    unique_handle file(CreateFileW(GHC_NATIVEWP(p), 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, 0, OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS, 0));
    if (!file) {
        ec = detail::make_system_error();
        return nullptr;
    }

    std::unique_ptr<REPARSE_DATA_BUFFER, free_deleter<REPARSE_DATA_BUFFER>> reparseData(reinterpret_cast<REPARSE_DATA_BUFFER*>(std::calloc(1, MAXIMUM_REPARSE_DATA_BUFFER_SIZE)));
    ULONG bufferUsed;
    if (DeviceIoControl(file.get(), FSCTL_GET_REPARSE_POINT, 0, 0, reparseData.get(), MAXIMUM_REPARSE_DATA_BUFFER_SIZE, &bufferUsed, 0)) {
        return reparseData;
    }
    else {
        ec = detail::make_system_error();
    }
    return nullptr;
}
#endif

GHC_INLINE path resolveSymlink(const path& p, std::error_code& ec)
{
#ifdef GHC_OS_WINDOWS
    path result;
    auto reparseData = detail::getReparseData(p, ec);
    if (!ec) {
        if (reparseData && IsReparseTagMicrosoft(reparseData->ReparseTag)) {
            switch (reparseData->ReparseTag) {
                case IO_REPARSE_TAG_SYMLINK: {
                    auto printName = std::wstring(&reparseData->SymbolicLinkReparseBuffer.PathBuffer[reparseData->SymbolicLinkReparseBuffer.PrintNameOffset / sizeof(WCHAR)], reparseData->SymbolicLinkReparseBuffer.PrintNameLength / sizeof(WCHAR));
                    auto substituteName =
                        std::wstring(&reparseData->SymbolicLinkReparseBuffer.PathBuffer[reparseData->SymbolicLinkReparseBuffer.SubstituteNameOffset / sizeof(WCHAR)], reparseData->SymbolicLinkReparseBuffer.SubstituteNameLength / sizeof(WCHAR));
                    if (detail::endsWith(substituteName, printName) && detail::startsWith(substituteName, std::wstring(L"\\??\\"))) {
                        result = printName;
                    }
                    else {
                        result = substituteName;
                    }
                    if (reparseData->SymbolicLinkReparseBuffer.Flags & 0x1 /*SYMLINK_FLAG_RELATIVE*/) {
                        result = p.parent_path() / result;
                    }
                    break;
                }
                case IO_REPARSE_TAG_MOUNT_POINT:
                    result = detail::getFullPathName(GHC_NATIVEWP(p), ec);
                    // result = std::wstring(&reparseData->MountPointReparseBuffer.PathBuffer[reparseData->MountPointReparseBuffer.SubstituteNameOffset / sizeof(WCHAR)], reparseData->MountPointReparseBuffer.SubstituteNameLength / sizeof(WCHAR));
                    break;
                default:
                    break;
            }
        }
    }
    return result;
#else
    size_t bufferSize = 256;
    while (true) {
        std::vector<char> buffer(bufferSize, static_cast<char>(0));
        auto rc = ::readlink(p.c_str(), buffer.data(), buffer.size());
        if (rc < 0) {
            ec = detail::make_system_error();
            return path();
        }
        else if (rc < static_cast<int>(bufferSize)) {
            return path(std::string(buffer.data(), static_cast<std::string::size_type>(rc)));
        }
        bufferSize *= 2;
    }
    return path();
#endif
}

#ifdef GHC_OS_WINDOWS
GHC_INLINE time_t timeFromFILETIME(const FILETIME& ft)
{
    ULARGE_INTEGER ull;
    ull.LowPart = ft.dwLowDateTime;
    ull.HighPart = ft.dwHighDateTime;
    return static_cast<time_t>(ull.QuadPart / 10000000ULL - 11644473600ULL);
}

GHC_INLINE void timeToFILETIME(time_t t, FILETIME& ft)
{
    LONGLONG ll;
    ll = Int32x32To64(t, 10000000) + 116444736000000000;
    ft.dwLowDateTime = static_cast<DWORD>(ll);
    ft.dwHighDateTime = static_cast<DWORD>(ll >> 32);
}

template <typename INFO>
GHC_INLINE uintmax_t hard_links_from_INFO(const INFO* info)
{
    return static_cast<uintmax_t>(-1);
}

template <>
GHC_INLINE uintmax_t hard_links_from_INFO<BY_HANDLE_FILE_INFORMATION>(const BY_HANDLE_FILE_INFORMATION* info)
{
    return info->nNumberOfLinks;
}

template <typename INFO>
GHC_INLINE DWORD reparse_tag_from_INFO(const INFO*)
{
    return 0;
}

template <>
GHC_INLINE DWORD reparse_tag_from_INFO(const WIN32_FIND_DATAW* info)
{
    return info->dwReserved0;
}

template <typename INFO>
GHC_INLINE file_status status_from_INFO(const path& p, const INFO* info, std::error_code& ec, uintmax_t* sz = nullptr, time_t* lwt = nullptr)
{
    file_type ft = file_type::unknown;
    if (sizeof(INFO) == sizeof(WIN32_FIND_DATAW)) {
        if (detail::reparse_tag_from_INFO(info) == IO_REPARSE_TAG_SYMLINK) {
            ft = file_type::symlink;
        }
    }
    else {
        if ((info->dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT)) {
            auto reparseData = detail::getReparseData(p, ec);
            if (!ec && reparseData && IsReparseTagMicrosoft(reparseData->ReparseTag) && reparseData->ReparseTag == IO_REPARSE_TAG_SYMLINK) {
                ft = file_type::symlink;
            }
        }
    }
    if (ft == file_type::unknown) {
        if ((info->dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            ft = file_type::directory;
        }
        else {
            ft = file_type::regular;
        }
    }
    perms prms = perms::owner_read | perms::group_read | perms::others_read;
    if (!(info->dwFileAttributes & FILE_ATTRIBUTE_READONLY)) {
        prms = prms | perms::owner_write | perms::group_write | perms::others_write;
    }
    if (has_executable_extension(p)) {
        prms = prms | perms::owner_exec | perms::group_exec | perms::others_exec;
    }
    if (sz) {
        *sz = static_cast<uintmax_t>(info->nFileSizeHigh) << (sizeof(info->nFileSizeHigh) * 8) | info->nFileSizeLow;
    }
    if (lwt) {
        *lwt = detail::timeFromFILETIME(info->ftLastWriteTime);
    }
    return file_status(ft, prms);
}

#endif

GHC_INLINE bool is_not_found_error(std::error_code& ec)
{
#ifdef GHC_OS_WINDOWS
    return ec.value() == ERROR_FILE_NOT_FOUND || ec.value() == ERROR_PATH_NOT_FOUND || ec.value() == ERROR_INVALID_NAME;
#else
    return ec.value() == ENOENT || ec.value() == ENOTDIR;
#endif
}

GHC_INLINE file_status symlink_status_ex(const path& p, std::error_code& ec, uintmax_t* sz = nullptr, uintmax_t* nhl = nullptr, time_t* lwt = nullptr) noexcept
{
#ifdef GHC_OS_WINDOWS
    file_status fs;
    WIN32_FILE_ATTRIBUTE_DATA attr;
    if (!GetFileAttributesExW(GHC_NATIVEWP(p), GetFileExInfoStandard, &attr)) {
        ec = detail::make_system_error();
    }
    else {
        ec.clear();
        fs = detail::status_from_INFO(p, &attr, ec, sz, lwt);
        if (nhl) {
            *nhl = 0;
        }
    }
    if (detail::is_not_found_error(ec)) {
        return file_status(file_type::not_found);
    }
    return ec ? file_status(file_type::none) : fs;
#else
    (void)sz;
    (void)nhl;
    (void)lwt;
    struct ::stat fs;
    auto result = ::lstat(p.c_str(), &fs);
    if (result == 0) {
        ec.clear();
        file_status f_s = detail::file_status_from_st_mode(fs.st_mode);
        return f_s;
    }
    ec = detail::make_system_error();
    if (detail::is_not_found_error(ec)) {
        return file_status(file_type::not_found, perms::unknown);
    }
    return file_status(file_type::none);
#endif
}

GHC_INLINE file_status status_ex(const path& p, std::error_code& ec, file_status* sls = nullptr, uintmax_t* sz = nullptr, uintmax_t* nhl = nullptr, time_t* lwt = nullptr, int recurse_count = 0) noexcept
{
    ec.clear();
#ifdef GHC_OS_WINDOWS
    if (recurse_count > 16) {
        ec = detail::make_system_error(0x2A9 /*ERROR_STOPPED_ON_SYMLINK*/);
        return file_status(file_type::unknown);
    }
    WIN32_FILE_ATTRIBUTE_DATA attr;
    if (!::GetFileAttributesExW(GHC_NATIVEWP(p), GetFileExInfoStandard, &attr)) {
        ec = detail::make_system_error();
    }
    else if (attr.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) {
        auto reparseData = detail::getReparseData(p, ec);
        if (!ec && reparseData && IsReparseTagMicrosoft(reparseData->ReparseTag) && reparseData->ReparseTag == IO_REPARSE_TAG_SYMLINK) {
            path target = resolveSymlink(p, ec);
            file_status result;
            if (!ec && !target.empty()) {
                if (sls) {
                    *sls = status_from_INFO(p, &attr, ec);
                }
                return detail::status_ex(target, ec, nullptr, sz, nhl, lwt, recurse_count + 1);
            }
            return file_status(file_type::unknown);
        }
    }
    if (ec) {
        if (detail::is_not_found_error(ec)) {
            return file_status(file_type::not_found);
        }
        return file_status(file_type::none);
    }
    if (nhl) {
        *nhl = 0;
    }
    return detail::status_from_INFO(p, &attr, ec, sz, lwt);
#else
    (void)recurse_count;
    struct ::stat st;
    auto result = ::lstat(p.c_str(), &st);
    if (result == 0) {
        ec.clear();
        file_status fs = detail::file_status_from_st_mode(st.st_mode);
        if (sls) {
            *sls = fs;
        }
        if (fs.type() == file_type::symlink) {
            result = ::stat(p.c_str(), &st);
            if (result == 0) {
                fs = detail::file_status_from_st_mode(st.st_mode);
            }
            else {
                ec = detail::make_system_error();
                if (detail::is_not_found_error(ec)) {
                    return file_status(file_type::not_found, perms::unknown);
                }
                return file_status(file_type::none);
            }
        }
        if (sz) {
            *sz = static_cast<uintmax_t>(st.st_size);
        }
        if (nhl) {
            *nhl = st.st_nlink;
        }
        if (lwt) {
            *lwt = st.st_mtime;
        }
        return fs;
    }
    else {
        ec = detail::make_system_error();
        if (detail::is_not_found_error(ec)) {
            return file_status(file_type::not_found, perms::unknown);
        }
        return file_status(file_type::none);
    }
#endif
}

}  // namespace detail

GHC_INLINE u8arguments::u8arguments(int& argc, char**& argv)
    : _argc(argc)
    , _argv(argv)
    , _refargc(argc)
    , _refargv(argv)
    , _isvalid(false)
{
#ifdef GHC_OS_WINDOWS
    LPWSTR* p;
    p = ::CommandLineToArgvW(::GetCommandLineW(), &argc);
    _args.reserve(static_cast<size_t>(argc));
    _argp.reserve(static_cast<size_t>(argc));
    for (size_t i = 0; i < static_cast<size_t>(argc); ++i) {
        _args.push_back(detail::toUtf8(std::wstring(p[i])));
        _argp.push_back((char*)_args[i].data());
    }
    argv = _argp.data();
    ::LocalFree(p);
    _isvalid = true;
#else
    std::setlocale(LC_ALL, "");
#if defined(__ANDROID__) && __ANDROID_API__ < 26
    _isvalid = true;
#else
    if (detail::equals_simple_insensitive(::nl_langinfo(CODESET), "UTF-8")) {
        _isvalid = true;
    }
#endif
#endif
}

//-----------------------------------------------------------------------------
// [fs.path.construct] constructors and destructor

GHC_INLINE path::path() noexcept {}

GHC_INLINE path::path(const path& p)
    : _path(p._path)
#if defined(GHC_OS_WINDOWS) && defined(GHC_WIN_AUTO_PREFIX_LONG_PATH)
    , _prefixLength(p._prefixLength)
#endif
{
}

GHC_INLINE path::path(path&& p) noexcept
    : _path(std::move(p._path))
#if defined(GHC_OS_WINDOWS) && defined(GHC_WIN_AUTO_PREFIX_LONG_PATH)
    , _prefixLength(p._prefixLength)
#endif
{
}

GHC_INLINE path::path(string_type&& source, format fmt)
    : _path(std::move(source))
{
    postprocess_path_with_format(fmt);
}

#endif  // GHC_EXPAND_IMPL

#ifdef GHC_WITH_EXCEPTIONS
template <class Source, typename>
inline path::path(const Source& source, const std::locale& loc, format fmt)
    : path(source, fmt)
{
    std::string locName = loc.name();
    if (!(locName.length() >= 5 && (locName.substr(locName.length() - 5) == "UTF-8" || locName.substr(locName.length() - 5) == "utf-8"))) {
        throw filesystem_error("This implementation only supports UTF-8 locales!", path(_path), detail::make_error_code(detail::portable_error::not_supported));
    }
}

template <class InputIterator>
inline path::path(InputIterator first, InputIterator last, const std::locale& loc, format fmt)
    : path(std::basic_string<typename std::iterator_traits<InputIterator>::value_type>(first, last), fmt)
{
    std::string locName = loc.name();
    if (!(locName.length() >= 5 && (locName.substr(locName.length() - 5) == "UTF-8" || locName.substr(locName.length() - 5) == "utf-8"))) {
        throw filesystem_error("This implementation only supports UTF-8 locales!", path(_path), detail::make_error_code(detail::portable_error::not_supported));
    }
}
#endif

#ifdef GHC_EXPAND_IMPL

GHC_INLINE path::~path() {}

//-----------------------------------------------------------------------------
// [fs.path.assign] assignments

GHC_INLINE path& path::operator=(const path& p)
{
    _path = p._path;
#if defined(GHC_OS_WINDOWS) && defined(GHC_WIN_AUTO_PREFIX_LONG_PATH)
    _prefixLength = p._prefixLength;
#endif
    return *this;
}

GHC_INLINE path& path::operator=(path&& p) noexcept
{
    _path = std::move(p._path);
#if defined(GHC_OS_WINDOWS) && defined(GHC_WIN_AUTO_PREFIX_LONG_PATH)
    _prefixLength = p._prefixLength;
#endif
    return *this;
}

GHC_INLINE path& path::operator=(path::string_type&& source)
{
    return assign(source);
}

GHC_INLINE path& path::assign(path::string_type&& source)
{
    _path = std::move(source);
    postprocess_path_with_format(native_format);
    return *this;
}

#endif  // GHC_EXPAND_IMPL

template <class Source>
inline path& path::operator=(const Source& source)
{
    return assign(source);
}

template <class Source>
inline path& path::assign(const Source& source)
{
#ifdef GHC_USE_WCHAR_T
    _path.assign(detail::toWChar(source));
#else
    _path.assign(detail::toUtf8(source));
#endif
    postprocess_path_with_format(native_format);
    return *this;
}

template <>
inline path& path::assign<path>(const path& source)
{
    _path = source._path;
#if defined(GHC_OS_WINDOWS) && defined(GHC_WIN_AUTO_PREFIX_LONG_PATH)
    _prefixLength = source._prefixLength;
#endif
    return *this;
}

template <class InputIterator>
inline path& path::assign(InputIterator first, InputIterator last)
{
    _path.assign(first, last);
    postprocess_path_with_format(native_format);
    return *this;
}

#ifdef GHC_EXPAND_IMPL

//-----------------------------------------------------------------------------
// [fs.path.append] appends

GHC_INLINE path& path::operator/=(const path& p)
{
    if (p.empty()) {
        // was: if ((!has_root_directory() && is_absolute()) || has_filename())
        if (!_path.empty() && _path[_path.length() - 1] != preferred_separator && _path[_path.length() - 1] != ':') {
            _path += preferred_separator;
        }
        return *this;
    }
    if ((p.is_absolute() && (_path != root_name()._path || p._path != "/")) || (p.has_root_name() && p.root_name() != root_name())) {
        assign(p);
        return *this;
    }
    if (p.has_root_directory()) {
        assign(root_name());
    }
    else if ((!has_root_directory() && is_absolute()) || has_filename()) {
        _path += preferred_separator;
    }
    auto iter = p.begin();
    bool first = true;
    if (p.has_root_name()) {
        ++iter;
    }
    while (iter != p.end()) {
        if (!first && !(!_path.empty() && _path[_path.length() - 1] == preferred_separator)) {
            _path += preferred_separator;
        }
        first = false;
        _path += (*iter++).native();
    }
    check_long_path();
    return *this;
}

GHC_INLINE void path::append_name(const value_type* name)
{
    if (_path.empty()) {
        this->operator/=(path(name));
    }
    else {
        if (_path.back() != path::preferred_separator) {
            _path.push_back(path::preferred_separator);
        }
        _path += name;
        check_long_path();
    }
}

#endif  // GHC_EXPAND_IMPL

template <class Source>
inline path& path::operator/=(const Source& source)
{
    return append(source);
}

template <class Source>
inline path& path::append(const Source& source)
{
    return this->operator/=(path(source));
}

template <>
inline path& path::append<path>(const path& p)
{
    return this->operator/=(p);
}

template <class InputIterator>
inline path& path::append(InputIterator first, InputIterator last)
{
    std::basic_string<typename std::iterator_traits<InputIterator>::value_type> part(first, last);
    return append(part);
}

#ifdef GHC_EXPAND_IMPL

//-----------------------------------------------------------------------------
// [fs.path.concat] concatenation

GHC_INLINE path& path::operator+=(const path& x)
{
    return concat(x._path);
}

GHC_INLINE path& path::operator+=(const string_type& x)
{
    return concat(x);
}

#ifdef GHC_WITH_STRING_VIEW
GHC_INLINE path& path::operator+=(basic_string_view<value_type> x)
{
    return concat(x);
}
#endif

GHC_INLINE path& path::operator+=(const value_type* x)
{
#ifdef GHC_WITH_STRING_VIEW
    basic_string_view<value_type> part(x);
#else
    string_type part(x);
#endif
    return concat(part);
}

GHC_INLINE path& path::operator+=(value_type x)
{
#ifdef GHC_OS_WINDOWS
    if (x == generic_separator) {
        x = preferred_separator;
    }
#endif
    if (_path.empty() || _path.back() != preferred_separator) {
        _path += x;
    }
    check_long_path();
    return *this;
}

#endif  // GHC_EXPAND_IMPL

template <class Source>
inline path::path_from_string<Source>& path::operator+=(const Source& x)
{
    return concat(x);
}

template <class EcharT>
inline path::path_type_EcharT<EcharT>& path::operator+=(EcharT x)
{
#ifdef GHC_WITH_STRING_VIEW
    basic_string_view<EcharT> part(&x, 1);
#else
    std::basic_string<EcharT> part(1, x);
#endif
    concat(part);
    return *this;
}

template <class Source>
inline path& path::concat(const Source& x)
{
    path p(x);
    _path += p._path;
    postprocess_path_with_format(native_format);
    return *this;
}
template <class InputIterator>
inline path& path::concat(InputIterator first, InputIterator last)
{
    _path.append(first, last);
    postprocess_path_with_format(native_format);
    return *this;
}

#ifdef GHC_EXPAND_IMPL

//-----------------------------------------------------------------------------
// [fs.path.modifiers] modifiers
GHC_INLINE void path::clear() noexcept
{
    _path.clear();
#if defined(GHC_OS_WINDOWS) && defined(GHC_WIN_AUTO_PREFIX_LONG_PATH)
    _prefixLength = 0;
#endif
}

GHC_INLINE path& path::make_preferred()
{
    // as this filesystem implementation only uses generic_format
    // internally, this must be a no-op
    return *this;
}

GHC_INLINE path& path::remove_filename()
{
    if (has_filename()) {
        _path.erase(_path.size() - filename()._path.size());
    }
    return *this;
}

GHC_INLINE path& path::replace_filename(const path& replacement)
{
    remove_filename();
    return append(replacement);
}

GHC_INLINE path& path::replace_extension(const path& replacement)
{
    if (has_extension()) {
        _path.erase(_path.size() - extension()._path.size());
    }
    if (!replacement.empty() && replacement._path[0] != '.') {
        _path += '.';
    }
    return concat(replacement);
}

GHC_INLINE void path::swap(path& rhs) noexcept
{
    _path.swap(rhs._path);
#if defined(GHC_OS_WINDOWS) && defined(GHC_WIN_AUTO_PREFIX_LONG_PATH)
    std::swap(_prefixLength, rhs._prefixLength);
#endif
}

//-----------------------------------------------------------------------------
// [fs.path.native.obs] native format observers
GHC_INLINE const path::string_type& path::native() const noexcept
{
    return _path;
}

GHC_INLINE const path::value_type* path::c_str() const noexcept
{
    return native().c_str();
}

GHC_INLINE path::operator path::string_type() const
{
    return native();
}

#endif  // GHC_EXPAND_IMPL

template <class EcharT, class traits, class Allocator>
inline std::basic_string<EcharT, traits, Allocator> path::string(const Allocator& a) const
{
#ifdef GHC_USE_WCHAR_T
    return detail::fromWChar<std::basic_string<EcharT, traits, Allocator>>(_path, a);
#else
    return detail::fromUtf8<std::basic_string<EcharT, traits, Allocator>>(_path, a);
#endif
}

#ifdef GHC_EXPAND_IMPL

GHC_INLINE std::string path::string() const
{
#ifdef GHC_USE_WCHAR_T
    return detail::toUtf8(native());
#else
    return native();
#endif
}

GHC_INLINE std::wstring path::wstring() const
{
#ifdef GHC_USE_WCHAR_T
    return native();
#else
    return detail::fromUtf8<std::wstring>(native());
#endif
}

#if defined(__cpp_lib_char8_t) && !defined(GHC_FILESYSTEM_ENFORCE_CPP17_API)
GHC_INLINE std::u8string path::u8string() const
{
#ifdef GHC_USE_WCHAR_T
    return std::u8string(reinterpret_cast<const char8_t*>(detail::toUtf8(native()).c_str()));
#else
    return std::u8string(reinterpret_cast<const char8_t*>(c_str()));
#endif
}
#else
GHC_INLINE std::string path::u8string() const
{
#ifdef GHC_USE_WCHAR_T
    return detail::toUtf8(native());
#else
    return native();
#endif
}
#endif

GHC_INLINE std::u16string path::u16string() const
{
    // TODO: optimize
    return detail::fromUtf8<std::u16string>(string());
}

GHC_INLINE std::u32string path::u32string() const
{
    // TODO: optimize
    return detail::fromUtf8<std::u32string>(string());
}

#endif  // GHC_EXPAND_IMPL

//-----------------------------------------------------------------------------
// [fs.path.generic.obs] generic format observers
template <class EcharT, class traits, class Allocator>
inline std::basic_string<EcharT, traits, Allocator> path::generic_string(const Allocator& a) const
{
#ifdef GHC_OS_WINDOWS
#ifdef GHC_USE_WCHAR_T
    auto result = detail::fromWChar<std::basic_string<EcharT, traits, Allocator>, path::string_type>(_path, a);
#else
    auto result = detail::fromUtf8<std::basic_string<EcharT, traits, Allocator>>(_path, a);
#endif
    for (auto& c : result) {
        if (c == preferred_separator) {
            c = generic_separator;
        }
    }
    return result;
#else
    return detail::fromUtf8<std::basic_string<EcharT, traits, Allocator>>(_path, a);
#endif
}

#ifdef GHC_EXPAND_IMPL

GHC_INLINE std::string path::generic_string() const
{
#ifdef GHC_OS_WINDOWS
    return generic_string<std::string::value_type, std::string::traits_type, std::string::allocator_type>();
#else
    return _path;
#endif
}

GHC_INLINE std::wstring path::generic_wstring() const
{
#ifdef GHC_OS_WINDOWS
    return generic_string<std::wstring::value_type, std::wstring::traits_type, std::wstring::allocator_type>();
#else
    return detail::fromUtf8<std::wstring>(_path);
#endif
}  // namespace filesystem

#if defined(__cpp_lib_char8_t) && !defined(GHC_FILESYSTEM_ENFORCE_CPP17_API)
GHC_INLINE std::u8string path::generic_u8string() const
{
#ifdef GHC_OS_WINDOWS
    return generic_string<std::u8string::value_type, std::u8string::traits_type, std::u8string::allocator_type>();
#else
    return std::u8string(reinterpret_cast<const char8_t*>(_path.c_str()));
#endif
}
#else
GHC_INLINE std::string path::generic_u8string() const
{
#ifdef GHC_OS_WINDOWS
    return generic_string<std::string::value_type, std::string::traits_type, std::string::allocator_type>();
#else
    return _path;
#endif
}
#endif

GHC_INLINE std::u16string path::generic_u16string() const
{
#ifdef GHC_OS_WINDOWS
    return generic_string<std::u16string::value_type, std::u16string::traits_type, std::u16string::allocator_type>();
#else
    return detail::fromUtf8<std::u16string>(_path);
#endif
}

GHC_INLINE std::u32string path::generic_u32string() const
{
#ifdef GHC_OS_WINDOWS
    return generic_string<std::u32string::value_type, std::u32string::traits_type, std::u32string::allocator_type>();
#else
    return detail::fromUtf8<std::u32string>(_path);
#endif
}

//-----------------------------------------------------------------------------
// [fs.path.compare] compare
GHC_INLINE int path::compare(const path& p) const noexcept
{
#ifdef LWG_2936_BEHAVIOUR
    auto rnl1 = root_name_length();
    auto rnl2 = p.root_name_length();
#ifdef GHC_OS_WINDOWS
    auto rnc = detail::compare_simple_insensitive(_path.c_str(), rnl1, p._path.c_str(), rnl2);
#else
    auto rnc = _path.compare(0, rnl1, p._path, 0, (std::min(rnl1, rnl2)));
#endif
    if (rnc) {
        return rnc;
    }
    bool hrd1 = has_root_directory(), hrd2 = p.has_root_directory();
    if (hrd1 != hrd2) {
        return hrd1 ? 1 : -1;
    }
    if (hrd1) {
        ++rnl1;
        ++rnl2;
    }
    auto iter1 = _path.begin() + static_cast<int>(rnl1);
    auto iter2 = p._path.begin() + static_cast<int>(rnl2);
    while (iter1 != _path.end() && iter2 != p._path.end() && *iter1 == *iter2) {
        ++iter1;
        ++iter2;
    }
    if (iter1 == _path.end()) {
        return iter2 == p._path.end() ? 0 : -1;
    }
    if (iter2 == p._path.end()) {
        return 1;
    }
    if (*iter1 == preferred_separator) {
        return -1;
    }
    if (*iter2 == preferred_separator) {
        return 1;
    }
    return *iter1 < *iter2 ? -1 : 1;
#else  // LWG_2936_BEHAVIOUR
#ifdef GHC_OS_WINDOWS
    auto rnl1 = root_name_length();
    auto rnl2 = p.root_name_length();
    auto rnc = detail::compare_simple_insensitive(_path.c_str(), rnl1, p._path.c_str(), rnl2);
    if (rnc) {
        return rnc;
    }
    return _path.compare(rnl1, std::string::npos, p._path, rnl2, std::string::npos);
#else
    return _path.compare(p._path);
#endif
#endif
}

GHC_INLINE int path::compare(const string_type& s) const
{
    return compare(path(s));
}

#ifdef GHC_WITH_STRING_VIEW
GHC_INLINE int path::compare(basic_string_view<value_type> s) const
{
    return compare(path(s));
}
#endif

GHC_INLINE int path::compare(const value_type* s) const
{
    return compare(path(s));
}

//-----------------------------------------------------------------------------
// [fs.path.decompose] decomposition
#ifdef GHC_OS_WINDOWS
GHC_INLINE void path::handle_prefixes()
{
#if defined(GHC_WIN_AUTO_PREFIX_LONG_PATH)
    _prefixLength = 0;
    if (_path.length() >= 6 && _path[2] == '?' && std::toupper(static_cast<unsigned char>(_path[4])) >= 'A' && std::toupper(static_cast<unsigned char>(_path[4])) <= 'Z' && _path[5] == ':') {
        if (detail::startsWith(_path, impl_string_type(GHC_PLATFORM_LITERAL("\\\\?\\"))) || detail::startsWith(_path, impl_string_type(GHC_PLATFORM_LITERAL("\\??\\")))) {
            _prefixLength = 4;
        }
    }
#endif  // GHC_WIN_AUTO_PREFIX_LONG_PATH
}
#endif

GHC_INLINE path::string_type::size_type path::root_name_length() const noexcept
{
#ifdef GHC_OS_WINDOWS
    if (_path.length() >= _prefixLength + 2 && std::toupper(static_cast<unsigned char>(_path[_prefixLength])) >= 'A' && std::toupper(static_cast<unsigned char>(_path[_prefixLength])) <= 'Z' && _path[_prefixLength + 1] == ':') {
        return 2;
    }
#endif
    if (_path.length() > _prefixLength + 2 && _path[_prefixLength] == preferred_separator && _path[_prefixLength + 1] == preferred_separator && _path[_prefixLength + 2] != preferred_separator && std::isprint(_path[_prefixLength + 2])) {
        impl_string_type::size_type pos = _path.find(preferred_separator, _prefixLength + 3);
        if (pos == impl_string_type::npos) {
            return _path.length();
        }
        else {
            return pos;
        }
    }
    return 0;
}

GHC_INLINE path path::root_name() const
{
    return path(_path.substr(_prefixLength, root_name_length()), native_format);
}

GHC_INLINE path path::root_directory() const
{
    if (has_root_directory()) {
        static const path _root_dir(std::string(1, preferred_separator), native_format);
        return _root_dir;
    }
    return path();
}

GHC_INLINE path path::root_path() const
{
    return path(root_name().string() + root_directory().string(), native_format);
}

GHC_INLINE path path::relative_path() const
{
    auto rootPathLen = _prefixLength + root_name_length() + (has_root_directory() ? 1 : 0);
    return path(_path.substr((std::min)(rootPathLen, _path.length())), generic_format);
}

GHC_INLINE path path::parent_path() const
{
    auto rootPathLen = _prefixLength + root_name_length() + (has_root_directory() ? 1 : 0);
    if (rootPathLen < _path.length()) {
        if (empty()) {
            return path();
        }
        else {
            auto piter = end();
            auto iter = piter.decrement(_path.end());
            if (iter > _path.begin() + static_cast<long>(rootPathLen) && *iter != preferred_separator) {
                --iter;
            }
            return path(_path.begin(), iter, native_format);
        }
    }
    else {
        return *this;
    }
}

GHC_INLINE path path::filename() const
{
    return !has_relative_path() ? path() : path(*--end());
}

GHC_INLINE path path::stem() const
{
    impl_string_type fn = filename().native();
    if (fn != "." && fn != "..") {
        impl_string_type::size_type pos = fn.rfind('.');
        if (pos != impl_string_type::npos && pos > 0) {
            return path{fn.substr(0, pos), native_format};
        }
    }
    return path{fn, native_format};
}

GHC_INLINE path path::extension() const
{
    if (has_relative_path()) {
        auto iter = end();
        const auto& fn = *--iter;
        impl_string_type::size_type pos = fn._path.rfind('.');
        if (pos != std::string::npos && pos > 0) {
            return path(fn._path.substr(pos), native_format);
        }
    }
    return path();
}

#ifdef GHC_OS_WINDOWS
namespace detail {
GHC_INLINE bool has_executable_extension(const path& p)
{
    if (p.has_relative_path()) {
        auto iter = p.end();
        const auto& fn = *--iter;
        auto pos = fn._path.find_last_of('.');
        if (pos == std::string::npos || pos == 0 || fn._path.length() - pos != 3) {
            return false;
        }
        const path::value_type* ext = fn._path.c_str() + pos + 1;
        if (detail::equals_simple_insensitive(ext, GHC_PLATFORM_LITERAL("exe")) || detail::equals_simple_insensitive(ext, GHC_PLATFORM_LITERAL("cmd")) || detail::equals_simple_insensitive(ext, GHC_PLATFORM_LITERAL("bat")) ||
            detail::equals_simple_insensitive(ext, GHC_PLATFORM_LITERAL("com"))) {
            return true;
        }
    }
    return false;
}
}  // namespace detail
#endif

//-----------------------------------------------------------------------------
// [fs.path.query] query
GHC_INLINE bool path::empty() const noexcept
{
    return _path.empty();
}

GHC_INLINE bool path::has_root_name() const
{
    return root_name_length() > 0;
}

GHC_INLINE bool path::has_root_directory() const
{
    auto rootLen = _prefixLength + root_name_length();
    return (_path.length() > rootLen && _path[rootLen] == preferred_separator);
}

GHC_INLINE bool path::has_root_path() const
{
    return has_root_name() || has_root_directory();
}

GHC_INLINE bool path::has_relative_path() const
{
    auto rootPathLen = _prefixLength + root_name_length() + (has_root_directory() ? 1 : 0);
    return rootPathLen < _path.length();
}

GHC_INLINE bool path::has_parent_path() const
{
    return !parent_path().empty();
}

GHC_INLINE bool path::has_filename() const
{
    return has_relative_path() && !filename().empty();
}

GHC_INLINE bool path::has_stem() const
{
    return !stem().empty();
}

GHC_INLINE bool path::has_extension() const
{
    return !extension().empty();
}

GHC_INLINE bool path::is_absolute() const
{
#ifdef GHC_OS_WINDOWS
    return has_root_name() && has_root_directory();
#else
    return has_root_directory();
#endif
}

GHC_INLINE bool path::is_relative() const
{
    return !is_absolute();
}

//-----------------------------------------------------------------------------
// [fs.path.gen] generation
GHC_INLINE path path::lexically_normal() const
{
    path dest;
    bool lastDotDot = false;
    for (string_type s : *this) {
        if (s == ".") {
            dest /= "";
            continue;
        }
        else if (s == ".." && !dest.empty()) {
            auto root = root_path();
            if (dest == root) {
                continue;
            }
            else if (*(--dest.end()) != "..") {
                if (dest._path.back() == preferred_separator) {
                    dest._path.pop_back();
                }
                dest.remove_filename();
                continue;
            }
        }
        if (!(s.empty() && lastDotDot)) {
            dest /= s;
        }
        lastDotDot = s == "..";
    }
    if (dest.empty()) {
        dest = ".";
    }
    return dest;
}

GHC_INLINE path path::lexically_relative(const path& base) const
{
    if (root_name() != base.root_name() || is_absolute() != base.is_absolute() || (!has_root_directory() && base.has_root_directory())) {
        return path();
    }
    const_iterator a = begin(), b = base.begin();
    while (a != end() && b != base.end() && *a == *b) {
        ++a;
        ++b;
    }
    if (a == end() && b == base.end()) {
        return path(".");
    }
    int count = 0;
    for (const auto& element : input_iterator_range<const_iterator>(b, base.end())) {
        if (element != "." && element != "" && element != "..") {
            ++count;
        }
        else if (element == "..") {
            --count;
        }
    }
    if (count < 0) {
        return path();
    }
    path result;
    for (int i = 0; i < count; ++i) {
        result /= "..";
    }
    for (const auto& element : input_iterator_range<const_iterator>(a, end())) {
        result /= element;
    }
    return result;
}

GHC_INLINE path path::lexically_proximate(const path& base) const
{
    path result = lexically_relative(base);
    return result.empty() ? *this : result;
}

//-----------------------------------------------------------------------------
// [fs.path.itr] iterators
GHC_INLINE path::iterator::iterator() {}

GHC_INLINE path::iterator::iterator(const path& p, const impl_string_type::const_iterator& pos)
    : _first(p._path.begin())
    , _last(p._path.end())
    , _prefix(_first + static_cast<string_type::difference_type>(p._prefixLength))
    , _root(p.has_root_directory() ? _first + static_cast<string_type::difference_type>(p._prefixLength + p.root_name_length()) : _last)
    , _iter(pos)
{
    if (pos != _last) {
        updateCurrent();
    }
}

GHC_INLINE path::impl_string_type::const_iterator path::iterator::increment(const path::impl_string_type::const_iterator& pos) const
{
    path::impl_string_type::const_iterator i = pos;
    bool fromStart = i == _first || i == _prefix;
    if (i != _last) {
        if (fromStart && i == _first && _prefix > _first) {
            i = _prefix;
        }
        else if (*i++ == preferred_separator) {
            // we can only sit on a slash if it is a network name or a root
            if (i != _last && *i == preferred_separator) {
                if (fromStart && !(i + 1 != _last && *(i + 1) == preferred_separator)) {
                    // leadind double slashes detected, treat this and the
                    // following until a slash as one unit
                    i = std::find(++i, _last, preferred_separator);
                }
                else {
                    // skip redundant slashes
                    while (i != _last && *i == preferred_separator) {
                        ++i;
                    }
                }
            }
        }
        else {
            if (fromStart && i != _last && *i == ':') {
                ++i;
            }
            else {
                i = std::find(i, _last, preferred_separator);
            }
        }
    }
    return i;
}

GHC_INLINE path::impl_string_type::const_iterator path::iterator::decrement(const path::impl_string_type::const_iterator& pos) const
{
    path::impl_string_type::const_iterator i = pos;
    if (i != _first) {
        --i;
        // if this is now the root slash or the trailing slash, we are done,
        // else check for network name
        if (i != _root && (pos != _last || *i != preferred_separator)) {
#ifdef GHC_OS_WINDOWS
            static const impl_string_type seps = GHC_PLATFORM_LITERAL("\\:");
            i = std::find_first_of(std::reverse_iterator<path::impl_string_type::const_iterator>(i), std::reverse_iterator<path::impl_string_type::const_iterator>(_first), seps.begin(), seps.end()).base();
            if (i > _first && *i == ':') {
                i++;
            }
#else
            i = std::find(std::reverse_iterator<path::impl_string_type::const_iterator>(i), std::reverse_iterator<path::impl_string_type::const_iterator>(_first), preferred_separator).base();
#endif
            // Now we have to check if this is a network name
            if (i - _first == 2 && *_first == preferred_separator && *(_first + 1) == preferred_separator) {
                i -= 2;
            }
        }
    }
    return i;
}

GHC_INLINE void path::iterator::updateCurrent()
{
    if ((_iter == _last) || (_iter != _first && _iter != _last && (*_iter == preferred_separator && _iter != _root) && (_iter + 1 == _last))) {
        _current.clear();
    }
    else {
        _current.assign(_iter, increment(_iter));
    }
}

GHC_INLINE path::iterator& path::iterator::operator++()
{
    _iter = increment(_iter);
    while (_iter != _last &&                 // we didn't reach the end
           _iter != _root &&                 // this is not a root position
           *_iter == preferred_separator &&  // we are on a separator
           (_iter + 1) != _last              // the slash is not the last char
    ) {
        ++_iter;
    }
    updateCurrent();
    return *this;
}

GHC_INLINE path::iterator path::iterator::operator++(int)
{
    path::iterator i{*this};
    ++(*this);
    return i;
}

GHC_INLINE path::iterator& path::iterator::operator--()
{
    _iter = decrement(_iter);
    updateCurrent();
    return *this;
}

GHC_INLINE path::iterator path::iterator::operator--(int)
{
    auto i = *this;
    --(*this);
    return i;
}

GHC_INLINE bool path::iterator::operator==(const path::iterator& other) const
{
    return _iter == other._iter;
}

GHC_INLINE bool path::iterator::operator!=(const path::iterator& other) const
{
    return _iter != other._iter;
}

GHC_INLINE path::iterator::reference path::iterator::operator*() const
{
    return _current;
}

GHC_INLINE path::iterator::pointer path::iterator::operator->() const
{
    return &_current;
}

GHC_INLINE path::iterator path::begin() const
{
    return iterator(*this, _path.begin());
}

GHC_INLINE path::iterator path::end() const
{
    return iterator(*this, _path.end());
}

//-----------------------------------------------------------------------------
// [fs.path.nonmember] path non-member functions
GHC_INLINE void swap(path& lhs, path& rhs) noexcept
{
    swap(lhs._path, rhs._path);
}

GHC_INLINE size_t hash_value(const path& p) noexcept
{
    return std::hash<std::string>()(p.generic_string());
}

#ifdef GHC_HAS_THREEWAY_COMP
GHC_INLINE std::strong_ordering operator<=>(const path& lhs, const path& rhs) noexcept
{
    return lhs.compare(rhs) <=> 0;
}
#endif

GHC_INLINE bool operator==(const path& lhs, const path& rhs) noexcept
{
    return lhs.compare(rhs) == 0;
}

GHC_INLINE bool operator!=(const path& lhs, const path& rhs) noexcept
{
    return !(lhs == rhs);
}

GHC_INLINE bool operator<(const path& lhs, const path& rhs) noexcept
{
    return lhs.compare(rhs) < 0;
}

GHC_INLINE bool operator<=(const path& lhs, const path& rhs) noexcept
{
    return lhs.compare(rhs) <= 0;
}

GHC_INLINE bool operator>(const path& lhs, const path& rhs) noexcept
{
    return lhs.compare(rhs) > 0;
}

GHC_INLINE bool operator>=(const path& lhs, const path& rhs) noexcept
{
    return lhs.compare(rhs) >= 0;
}

GHC_INLINE path operator/(const path& lhs, const path& rhs)
{
    path result(lhs);
    result /= rhs;
    return result;
}

#endif  // GHC_EXPAND_IMPL

//-----------------------------------------------------------------------------
// [fs.path.io] path inserter and extractor
template <class charT, class traits>
inline std::basic_ostream<charT, traits>& operator<<(std::basic_ostream<charT, traits>& os, const path& p)
{
    os << "\"";
    auto ps = p.string<charT, traits>();
    for (auto c : ps) {
        if (c == '"' || c == '\\') {
            os << '\\';
        }
        os << c;
    }
    os << "\"";
    return os;
}

template <class charT, class traits>
inline std::basic_istream<charT, traits>& operator>>(std::basic_istream<charT, traits>& is, path& p)
{
    std::basic_string<charT, traits> tmp;
    charT c;
    is >> c;
    if (c == '"') {
        auto sf = is.flags();
        is >> std::noskipws;
        while (is) {
            auto c2 = is.get();
            if (is) {
                if (c2 == '\\') {
                    c2 = is.get();
                    if (is) {
                        tmp += static_cast<charT>(c2);
                    }
                }
                else if (c2 == '"') {
                    break;
                }
                else {
                    tmp += static_cast<charT>(c2);
                }
            }
        }
        if ((sf & std::ios_base::skipws) == std::ios_base::skipws) {
            is >> std::skipws;
        }
        p = path(tmp);
    }
    else {
        is >> tmp;
        p = path(static_cast<charT>(c) + tmp);
    }
    return is;
}

#ifdef GHC_EXPAND_IMPL

//-----------------------------------------------------------------------------
// [fs.class.filesystem_error] Class filesystem_error
GHC_INLINE filesystem_error::filesystem_error(const std::string& what_arg, std::error_code ec)
    : std::system_error(ec, what_arg)
    , _what_arg(what_arg)
    , _ec(ec)
{
}

GHC_INLINE filesystem_error::filesystem_error(const std::string& what_arg, const path& p1, std::error_code ec)
    : std::system_error(ec, what_arg)
    , _what_arg(what_arg)
    , _ec(ec)
    , _p1(p1)
{
    if (!_p1.empty()) {
        _what_arg += ": '" + _p1.string() + "'";
    }
}

GHC_INLINE filesystem_error::filesystem_error(const std::string& what_arg, const path& p1, const path& p2, std::error_code ec)
    : std::system_error(ec, what_arg)
    , _what_arg(what_arg)
    , _ec(ec)
    , _p1(p1)
    , _p2(p2)
{
    if (!_p1.empty()) {
        _what_arg += ": '" + _p1.string() + "'";
    }
    if (!_p2.empty()) {
        _what_arg += ", '" + _p2.string() + "'";
    }
}

GHC_INLINE const path& filesystem_error::path1() const noexcept
{
    return _p1;
}

GHC_INLINE const path& filesystem_error::path2() const noexcept
{
    return _p2;
}

GHC_INLINE const char* filesystem_error::what() const noexcept
{
    return _what_arg.c_str();
}

//-----------------------------------------------------------------------------
// [fs.op.funcs] filesystem operations
#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE path absolute(const path& p)
{
    std::error_code ec;
    path result = absolute(p, ec);
    if (ec) {
        throw filesystem_error(detail::systemErrorText(ec.value()), p, ec);
    }
    return result;
}
#endif

GHC_INLINE path absolute(const path& p, std::error_code& ec)
{
    ec.clear();
#ifdef GHC_OS_WINDOWS
    if (p.empty()) {
        return absolute(current_path(ec), ec) / "";
    }
    ULONG size = ::GetFullPathNameW(GHC_NATIVEWP(p), 0, 0, 0);
    if (size) {
        std::vector<wchar_t> buf(size, 0);
        ULONG s2 = GetFullPathNameW(GHC_NATIVEWP(p), size, buf.data(), nullptr);
        if (s2 && s2 < size) {
            path result = path(std::wstring(buf.data(), s2));
            if (p.filename() == ".") {
                result /= ".";
            }
            return result;
        }
    }
    ec = detail::make_system_error();
    return path();
#else
    path base = current_path(ec);
    if (!ec) {
        if (p.empty()) {
            return base / p;
        }
        if (p.has_root_name()) {
            if (p.has_root_directory()) {
                return p;
            }
            else {
                return p.root_name() / base.root_directory() / base.relative_path() / p.relative_path();
            }
        }
        else {
            if (p.has_root_directory()) {
                return base.root_name() / p;
            }
            else {
                return base / p;
            }
        }
    }
    ec = detail::make_system_error();
    return path();
#endif
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE path canonical(const path& p)
{
    std::error_code ec;
    auto result = canonical(p, ec);
    if (ec) {
        throw filesystem_error(detail::systemErrorText(ec.value()), p, ec);
    }
    return result;
}
#endif

GHC_INLINE path canonical(const path& p, std::error_code& ec)
{
    if (p.empty()) {
        ec = detail::make_error_code(detail::portable_error::not_found);
        return path();
    }
    path work = p.is_absolute() ? p : absolute(p, ec);
    path result;

    auto fs = status(work, ec);
    if (ec) {
        return path();
    }
    if (fs.type() == file_type::not_found) {
        ec = detail::make_error_code(detail::portable_error::not_found);
        return path();
    }
    bool redo;
    do {
        auto rootPathLen = work._prefixLength + work.root_name_length() + (work.has_root_directory() ? 1 : 0);
        redo = false;
        result.clear();
        for (auto pe : work) {
            if (pe.empty() || pe == ".") {
                continue;
            }
            else if (pe == "..") {
                result = result.parent_path();
                continue;
            }
            else if ((result / pe).string().length() <= rootPathLen) {
                result /= pe;
                continue;
            }
            auto sls = symlink_status(result / pe, ec);
            if (ec) {
                return path();
            }
            if (is_symlink(sls)) {
                redo = true;
                auto target = read_symlink(result / pe, ec);
                if (ec) {
                    return path();
                }
                if (target.is_absolute()) {
                    result = target;
                    continue;
                }
                else {
                    result /= target;
                    continue;
                }
            }
            else {
                result /= pe;
            }
        }
        work = result;
    } while (redo);
    ec.clear();
    return result;
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE void copy(const path& from, const path& to)
{
    copy(from, to, copy_options::none);
}

GHC_INLINE void copy(const path& from, const path& to, copy_options options)
{
    std::error_code ec;
    copy(from, to, options, ec);
    if (ec) {
        throw filesystem_error(detail::systemErrorText(ec.value()), from, to, ec);
    }
}
#endif

GHC_INLINE void copy(const path& from, const path& to, std::error_code& ec) noexcept
{
    copy(from, to, copy_options::none, ec);
}

GHC_INLINE void copy(const path& from, const path& to, copy_options options, std::error_code& ec) noexcept
{
    std::error_code tec;
    file_status fs_from, fs_to;
    ec.clear();
    if ((options & (copy_options::skip_symlinks | copy_options::copy_symlinks | copy_options::create_symlinks)) != copy_options::none) {
        fs_from = symlink_status(from, ec);
    }
    else {
        fs_from = status(from, ec);
    }
    if (!exists(fs_from)) {
        if (!ec) {
            ec = detail::make_error_code(detail::portable_error::not_found);
        }
        return;
    }
    if ((options & (copy_options::skip_symlinks | copy_options::create_symlinks)) != copy_options::none) {
        fs_to = symlink_status(to, tec);
    }
    else {
        fs_to = status(to, tec);
    }
    if (is_other(fs_from) || is_other(fs_to) || (is_directory(fs_from) && is_regular_file(fs_to)) || (exists(fs_to) && equivalent(from, to, ec))) {
        ec = detail::make_error_code(detail::portable_error::invalid_argument);
    }
    else if (is_symlink(fs_from)) {
        if ((options & copy_options::skip_symlinks) == copy_options::none) {
            if (!exists(fs_to) && (options & copy_options::copy_symlinks) != copy_options::none) {
                copy_symlink(from, to, ec);
            }
            else {
                ec = detail::make_error_code(detail::portable_error::invalid_argument);
            }
        }
    }
    else if (is_regular_file(fs_from)) {
        if ((options & copy_options::directories_only) == copy_options::none) {
            if ((options & copy_options::create_symlinks) != copy_options::none) {
                create_symlink(from.is_absolute() ? from : canonical(from, ec), to, ec);
            }
#ifndef GHC_OS_WEB
            else if ((options & copy_options::create_hard_links) != copy_options::none) {
                create_hard_link(from, to, ec);
            }
#endif
            else if (is_directory(fs_to)) {
                copy_file(from, to / from.filename(), options, ec);
            }
            else {
                copy_file(from, to, options, ec);
            }
        }
    }
#ifdef LWG_2682_BEHAVIOUR
    else if (is_directory(fs_from) && (options & copy_options::create_symlinks) != copy_options::none) {
        ec = detail::make_error_code(detail::portable_error::is_a_directory);
    }
#endif
    else if (is_directory(fs_from) && (options == copy_options::none || (options & copy_options::recursive) != copy_options::none)) {
        if (!exists(fs_to)) {
            create_directory(to, from, ec);
            if (ec) {
                return;
            }
        }
        for (auto iter = directory_iterator(from, ec); iter != directory_iterator(); iter.increment(ec)) {
            if (!ec) {
                copy(iter->path(), to / iter->path().filename(), options | static_cast<copy_options>(0x8000), ec);
            }
            if (ec) {
                return;
            }
        }
    }
    return;
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE bool copy_file(const path& from, const path& to)
{
    return copy_file(from, to, copy_options::none);
}

GHC_INLINE bool copy_file(const path& from, const path& to, copy_options option)
{
    std::error_code ec;
    auto result = copy_file(from, to, option, ec);
    if (ec) {
        throw filesystem_error(detail::systemErrorText(ec.value()), from, to, ec);
    }
    return result;
}
#endif

GHC_INLINE bool copy_file(const path& from, const path& to, std::error_code& ec) noexcept
{
    return copy_file(from, to, copy_options::none, ec);
}

GHC_INLINE bool copy_file(const path& from, const path& to, copy_options options, std::error_code& ec) noexcept
{
    std::error_code tecf, tect;
    auto sf = status(from, tecf);
    auto st = status(to, tect);
    bool overwrite = false;
    ec.clear();
    if (!is_regular_file(sf)) {
        ec = tecf;
        return false;
    }
    if (exists(st) && (!is_regular_file(st) || equivalent(from, to, ec) || (options & (copy_options::skip_existing | copy_options::overwrite_existing | copy_options::update_existing)) == copy_options::none)) {
        ec = tect ? tect : detail::make_error_code(detail::portable_error::exists);
        return false;
    }
    if (exists(st)) {
        if ((options & copy_options::update_existing) == copy_options::update_existing) {
            auto from_time = last_write_time(from, ec);
            if (ec) {
                ec = detail::make_system_error();
                return false;
            }
            auto to_time = last_write_time(to, ec);
            if (ec) {
                ec = detail::make_system_error();
                return false;
            }
            if (from_time <= to_time) {
                return false;
            }
        }
        overwrite = true;
    }
#ifdef GHC_OS_WINDOWS
    if (!::CopyFileW(GHC_NATIVEWP(from), GHC_NATIVEWP(to), !overwrite)) {
        ec = detail::make_system_error();
        return false;
    }
    return true;
#else
    std::vector<char> buffer(16384, '\0');
    int in = -1, out = -1;
    if ((in = ::open(from.c_str(), O_RDONLY)) < 0) {
        ec = detail::make_system_error();
        return false;
    }
    int mode = O_CREAT | O_WRONLY | O_TRUNC;
    if (!overwrite) {
        mode |= O_EXCL;
    }
    if ((out = ::open(to.c_str(), mode, static_cast<int>(sf.permissions() & perms::all))) < 0) {
        ec = detail::make_system_error();
        ::close(in);
        return false;
    }
    ssize_t br, bw;
    while ((br = ::read(in, buffer.data(), buffer.size())) > 0) {
        ssize_t offset = 0;
        do {
            if ((bw = ::write(out, buffer.data() + offset, static_cast<size_t>(br))) > 0) {
                br -= bw;
                offset += bw;
            }
            else if (bw < 0) {
                ec = detail::make_system_error();
                ::close(in);
                ::close(out);
                return false;
            }
        } while (br);
    }
    ::close(in);
    ::close(out);
    return true;
#endif
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE void copy_symlink(const path& existing_symlink, const path& new_symlink)
{
    std::error_code ec;
    copy_symlink(existing_symlink, new_symlink, ec);
    if (ec) {
        throw filesystem_error(detail::systemErrorText(ec.value()), existing_symlink, new_symlink, ec);
    }
}
#endif

GHC_INLINE void copy_symlink(const path& existing_symlink, const path& new_symlink, std::error_code& ec) noexcept
{
    ec.clear();
    auto to = read_symlink(existing_symlink, ec);
    if (!ec) {
        if (exists(to, ec) && is_directory(to, ec)) {
            create_directory_symlink(to, new_symlink, ec);
        }
        else {
            create_symlink(to, new_symlink, ec);
        }
    }
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE bool create_directories(const path& p)
{
    std::error_code ec;
    auto result = create_directories(p, ec);
    if (ec) {
        throw filesystem_error(detail::systemErrorText(ec.value()), p, ec);
    }
    return result;
}
#endif

GHC_INLINE bool create_directories(const path& p, std::error_code& ec) noexcept
{
    path current;
    ec.clear();
    bool didCreate = false;
    auto rootPathLen = p._prefixLength + p.root_name_length() + (p.has_root_directory() ? 1 : 0);
    current = p.native().substr(0, rootPathLen);
    path folders(p._path.substr(rootPathLen));
    for (path::string_type part : folders) {
        current /= part;
        std::error_code tec;
        auto fs = status(current, tec);
        if (tec && fs.type() != file_type::not_found) {
            ec = tec;
            return false;
        }
        if (!exists(fs)) {
            create_directory(current, ec);
            if (ec) {
                std::error_code tmp_ec;
                if (is_directory(current, tmp_ec)) {
                    ec.clear();
                }
                else {
                    return false;
                }
            }
            didCreate = true;
        }
#ifndef LWG_2935_BEHAVIOUR
        else if (!is_directory(fs)) {
            ec = detail::make_error_code(detail::portable_error::exists);
            return false;
        }
#endif
    }
    return didCreate;
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE bool create_directory(const path& p)
{
    std::error_code ec;
    auto result = create_directory(p, path(), ec);
    if (ec) {
        throw filesystem_error(detail::systemErrorText(ec.value()), p, ec);
    }
    return result;
}
#endif

GHC_INLINE bool create_directory(const path& p, std::error_code& ec) noexcept
{
    return create_directory(p, path(), ec);
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE bool create_directory(const path& p, const path& attributes)
{
    std::error_code ec;
    auto result = create_directory(p, attributes, ec);
    if (ec) {
        throw filesystem_error(detail::systemErrorText(ec.value()), p, ec);
    }
    return result;
}
#endif

GHC_INLINE bool create_directory(const path& p, const path& attributes, std::error_code& ec) noexcept
{
    std::error_code tec;
    ec.clear();
    auto fs = status(p, tec);
#ifdef LWG_2935_BEHAVIOUR
    if (status_known(fs) && exists(fs)) {
        return false;
    }
#else
    if (status_known(fs) && exists(fs) && is_directory(fs)) {
        return false;
    }
#endif
#ifdef GHC_OS_WINDOWS
    if (!attributes.empty()) {
        if (!::CreateDirectoryExW(GHC_NATIVEWP(attributes), GHC_NATIVEWP(p), NULL)) {
            ec = detail::make_system_error();
            return false;
        }
    }
    else if (!::CreateDirectoryW(GHC_NATIVEWP(p), NULL)) {
        ec = detail::make_system_error();
        return false;
    }
#else
    ::mode_t attribs = static_cast<mode_t>(perms::all);
    if (!attributes.empty()) {
        struct ::stat fileStat;
        if (::stat(attributes.c_str(), &fileStat) != 0) {
            ec = detail::make_system_error();
            return false;
        }
        attribs = fileStat.st_mode;
    }
    if (::mkdir(p.c_str(), attribs) != 0) {
        ec = detail::make_system_error();
        return false;
    }
#endif
    return true;
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE void create_directory_symlink(const path& to, const path& new_symlink)
{
    std::error_code ec;
    create_directory_symlink(to, new_symlink, ec);
    if (ec) {
        throw filesystem_error(detail::systemErrorText(ec.value()), to, new_symlink, ec);
    }
}
#endif

GHC_INLINE void create_directory_symlink(const path& to, const path& new_symlink, std::error_code& ec) noexcept
{
    detail::create_symlink(to, new_symlink, true, ec);
}

#ifndef GHC_OS_WEB
#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE void create_hard_link(const path& to, const path& new_hard_link)
{
    std::error_code ec;
    create_hard_link(to, new_hard_link, ec);
    if (ec) {
        throw filesystem_error(detail::systemErrorText(ec.value()), to, new_hard_link, ec);
    }
}
#endif

GHC_INLINE void create_hard_link(const path& to, const path& new_hard_link, std::error_code& ec) noexcept
{
    detail::create_hardlink(to, new_hard_link, ec);
}
#endif

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE void create_symlink(const path& to, const path& new_symlink)
{
    std::error_code ec;
    create_symlink(to, new_symlink, ec);
    if (ec) {
        throw filesystem_error(detail::systemErrorText(ec.value()), to, new_symlink, ec);
    }
}
#endif

GHC_INLINE void create_symlink(const path& to, const path& new_symlink, std::error_code& ec) noexcept
{
    detail::create_symlink(to, new_symlink, false, ec);
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE path current_path()
{
    std::error_code ec;
    auto result = current_path(ec);
    if (ec) {
        throw filesystem_error(detail::systemErrorText(ec.value()), ec);
    }
    return result;
}
#endif

GHC_INLINE path current_path(std::error_code& ec)
{
    ec.clear();
#ifdef GHC_OS_WINDOWS
    DWORD pathlen = ::GetCurrentDirectoryW(0, 0);
    std::unique_ptr<wchar_t[]> buffer(new wchar_t[size_t(pathlen) + 1]);
    if (::GetCurrentDirectoryW(pathlen, buffer.get()) == 0) {
        ec = detail::make_system_error();
        return path();
    }
    return path(std::wstring(buffer.get()), path::native_format);
#else
    size_t pathlen = static_cast<size_t>(std::max(int(::pathconf(".", _PC_PATH_MAX)), int(PATH_MAX)));
    std::unique_ptr<char[]> buffer(new char[pathlen + 1]);
    if (::getcwd(buffer.get(), pathlen) == nullptr) {
        ec = detail::make_system_error();
        return path();
    }
    return path(buffer.get());
#endif
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE void current_path(const path& p)
{
    std::error_code ec;
    current_path(p, ec);
    if (ec) {
        throw filesystem_error(detail::systemErrorText(ec.value()), p, ec);
    }
}
#endif

GHC_INLINE void current_path(const path& p, std::error_code& ec) noexcept
{
    ec.clear();
#ifdef GHC_OS_WINDOWS
    if (!::SetCurrentDirectoryW(GHC_NATIVEWP(p))) {
        ec = detail::make_system_error();
    }
#else
    if (::chdir(p.string().c_str()) == -1) {
        ec = detail::make_system_error();
    }
#endif
}

GHC_INLINE bool exists(file_status s) noexcept
{
    return status_known(s) && s.type() != file_type::not_found;
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE bool exists(const path& p)
{
    return exists(status(p));
}
#endif

GHC_INLINE bool exists(const path& p, std::error_code& ec) noexcept
{
    file_status s = status(p, ec);
    if (status_known(s)) {
        ec.clear();
    }
    return exists(s);
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE bool equivalent(const path& p1, const path& p2)
{
    std::error_code ec;
    bool result = equivalent(p1, p2, ec);
    if (ec) {
        throw filesystem_error(detail::systemErrorText(ec.value()), p1, p2, ec);
    }
    return result;
}
#endif

GHC_INLINE bool equivalent(const path& p1, const path& p2, std::error_code& ec) noexcept
{
    ec.clear();
#ifdef GHC_OS_WINDOWS
    detail::unique_handle file1(::CreateFileW(GHC_NATIVEWP(p1), 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, 0, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, 0));
    auto e1 = ::GetLastError();
    detail::unique_handle file2(::CreateFileW(GHC_NATIVEWP(p2), 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, 0, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, 0));
    if (!file1 || !file2) {
#ifdef LWG_2937_BEHAVIOUR
        ec = detail::make_system_error(e1 ? e1 : ::GetLastError());
#else
        if (file1 == file2) {
            ec = detail::make_system_error(e1 ? e1 : ::GetLastError());
        }
#endif
        return false;
    }
    BY_HANDLE_FILE_INFORMATION inf1, inf2;
    if (!::GetFileInformationByHandle(file1.get(), &inf1)) {
        ec = detail::make_system_error();
        return false;
    }
    if (!::GetFileInformationByHandle(file2.get(), &inf2)) {
        ec = detail::make_system_error();
        return false;
    }
    return inf1.ftLastWriteTime.dwLowDateTime == inf2.ftLastWriteTime.dwLowDateTime && inf1.ftLastWriteTime.dwHighDateTime == inf2.ftLastWriteTime.dwHighDateTime && inf1.nFileIndexHigh == inf2.nFileIndexHigh && inf1.nFileIndexLow == inf2.nFileIndexLow &&
           inf1.nFileSizeHigh == inf2.nFileSizeHigh && inf1.nFileSizeLow == inf2.nFileSizeLow && inf1.dwVolumeSerialNumber == inf2.dwVolumeSerialNumber;
#else
    struct ::stat s1, s2;
    auto rc1 = ::stat(p1.c_str(), &s1);
    auto e1 = errno;
    auto rc2 = ::stat(p2.c_str(), &s2);
    if (rc1 || rc2) {
#ifdef LWG_2937_BEHAVIOUR
        ec = detail::make_system_error(e1 ? e1 : errno);
#else
        if (rc1 && rc2) {
            ec = detail::make_system_error(e1 ? e1 : errno);
        }
#endif
        return false;
    }
    return s1.st_dev == s2.st_dev && s1.st_ino == s2.st_ino && s1.st_size == s2.st_size && s1.st_mtime == s2.st_mtime;
#endif
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE uintmax_t file_size(const path& p)
{
    std::error_code ec;
    auto result = file_size(p, ec);
    if (ec) {
        throw filesystem_error(detail::systemErrorText(ec.value()), p, ec);
    }
    return result;
}
#endif

GHC_INLINE uintmax_t file_size(const path& p, std::error_code& ec) noexcept
{
    ec.clear();
#ifdef GHC_OS_WINDOWS
    WIN32_FILE_ATTRIBUTE_DATA attr;
    if (!GetFileAttributesExW(GHC_NATIVEWP(p), GetFileExInfoStandard, &attr)) {
        ec = detail::make_system_error();
        return static_cast<uintmax_t>(-1);
    }
    return static_cast<uintmax_t>(attr.nFileSizeHigh) << (sizeof(attr.nFileSizeHigh) * 8) | attr.nFileSizeLow;
#else
    struct ::stat fileStat;
    if (::stat(p.c_str(), &fileStat) == -1) {
        ec = detail::make_system_error();
        return static_cast<uintmax_t>(-1);
    }
    return static_cast<uintmax_t>(fileStat.st_size);
#endif
}

#ifndef GHC_OS_WEB
#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE uintmax_t hard_link_count(const path& p)
{
    std::error_code ec;
    auto result = hard_link_count(p, ec);
    if (ec) {
        throw filesystem_error(detail::systemErrorText(ec.value()), p, ec);
    }
    return result;
}
#endif

GHC_INLINE uintmax_t hard_link_count(const path& p, std::error_code& ec) noexcept
{
    ec.clear();
#ifdef GHC_OS_WINDOWS
    uintmax_t result = static_cast<uintmax_t>(-1);
    detail::unique_handle file(::CreateFileW(GHC_NATIVEWP(p), 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, 0, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, 0));
    BY_HANDLE_FILE_INFORMATION inf;
    if (!file) {
        ec = detail::make_system_error();
    }
    else {
        if (!::GetFileInformationByHandle(file.get(), &inf)) {
            ec = detail::make_system_error();
        }
        else {
            result = inf.nNumberOfLinks;
        }
    }
    return result;
#else
    uintmax_t result = 0;
    file_status fs = detail::status_ex(p, ec, nullptr, nullptr, &result, nullptr);
    if (fs.type() == file_type::not_found) {
        ec = detail::make_error_code(detail::portable_error::not_found);
    }
    return ec ? static_cast<uintmax_t>(-1) : result;
#endif
}
#endif

GHC_INLINE bool is_block_file(file_status s) noexcept
{
    return s.type() == file_type::block;
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE bool is_block_file(const path& p)
{
    return is_block_file(status(p));
}
#endif

GHC_INLINE bool is_block_file(const path& p, std::error_code& ec) noexcept
{
    return is_block_file(status(p, ec));
}

GHC_INLINE bool is_character_file(file_status s) noexcept
{
    return s.type() == file_type::character;
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE bool is_character_file(const path& p)
{
    return is_character_file(status(p));
}
#endif

GHC_INLINE bool is_character_file(const path& p, std::error_code& ec) noexcept
{
    return is_character_file(status(p, ec));
}

GHC_INLINE bool is_directory(file_status s) noexcept
{
    return s.type() == file_type::directory;
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE bool is_directory(const path& p)
{
    return is_directory(status(p));
}
#endif

GHC_INLINE bool is_directory(const path& p, std::error_code& ec) noexcept
{
    return is_directory(status(p, ec));
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE bool is_empty(const path& p)
{
    if (is_directory(p)) {
        return directory_iterator(p) == directory_iterator();
    }
    else {
        return file_size(p) == 0;
    }
}
#endif

GHC_INLINE bool is_empty(const path& p, std::error_code& ec) noexcept
{
    auto fs = status(p, ec);
    if (ec) {
        return false;
    }
    if (is_directory(fs)) {
        directory_iterator iter(p, ec);
        if (ec) {
            return false;
        }
        return iter == directory_iterator();
    }
    else {
        auto sz = file_size(p, ec);
        if (ec) {
            return false;
        }
        return sz == 0;
    }
}

GHC_INLINE bool is_fifo(file_status s) noexcept
{
    return s.type() == file_type::fifo;
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE bool is_fifo(const path& p)
{
    return is_fifo(status(p));
}
#endif

GHC_INLINE bool is_fifo(const path& p, std::error_code& ec) noexcept
{
    return is_fifo(status(p, ec));
}

GHC_INLINE bool is_other(file_status s) noexcept
{
    return exists(s) && !is_regular_file(s) && !is_directory(s) && !is_symlink(s);
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE bool is_other(const path& p)
{
    return is_other(status(p));
}
#endif

GHC_INLINE bool is_other(const path& p, std::error_code& ec) noexcept
{
    return is_other(status(p, ec));
}

GHC_INLINE bool is_regular_file(file_status s) noexcept
{
    return s.type() == file_type::regular;
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE bool is_regular_file(const path& p)
{
    return is_regular_file(status(p));
}
#endif

GHC_INLINE bool is_regular_file(const path& p, std::error_code& ec) noexcept
{
    return is_regular_file(status(p, ec));
}

GHC_INLINE bool is_socket(file_status s) noexcept
{
    return s.type() == file_type::socket;
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE bool is_socket(const path& p)
{
    return is_socket(status(p));
}
#endif

GHC_INLINE bool is_socket(const path& p, std::error_code& ec) noexcept
{
    return is_socket(status(p, ec));
}

GHC_INLINE bool is_symlink(file_status s) noexcept
{
    return s.type() == file_type::symlink;
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE bool is_symlink(const path& p)
{
    return is_symlink(symlink_status(p));
}
#endif

GHC_INLINE bool is_symlink(const path& p, std::error_code& ec) noexcept
{
    return is_symlink(symlink_status(p, ec));
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE file_time_type last_write_time(const path& p)
{
    std::error_code ec;
    auto result = last_write_time(p, ec);
    if (ec) {
        throw filesystem_error(detail::systemErrorText(ec.value()), p, ec);
    }
    return result;
}
#endif

GHC_INLINE file_time_type last_write_time(const path& p, std::error_code& ec) noexcept
{
    time_t result = 0;
    ec.clear();
    file_status fs = detail::status_ex(p, ec, nullptr, nullptr, nullptr, &result);
    return ec ? (file_time_type::min)() : std::chrono::system_clock::from_time_t(result);
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE void last_write_time(const path& p, file_time_type new_time)
{
    std::error_code ec;
    last_write_time(p, new_time, ec);
    if (ec) {
        throw filesystem_error(detail::systemErrorText(ec.value()), p, ec);
    }
}
#endif

GHC_INLINE void last_write_time(const path& p, file_time_type new_time, std::error_code& ec) noexcept
{
    ec.clear();
    auto d = new_time.time_since_epoch();
#ifdef GHC_OS_WINDOWS
    detail::unique_handle file(::CreateFileW(GHC_NATIVEWP(p), FILE_WRITE_ATTRIBUTES, FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, NULL));
    FILETIME ft;
    auto tt = std::chrono::duration_cast<std::chrono::microseconds>(d).count() * 10 + 116444736000000000;
    ft.dwLowDateTime = static_cast<DWORD>(tt);
    ft.dwHighDateTime = static_cast<DWORD>(tt >> 32);
    if (!::SetFileTime(file.get(), 0, 0, &ft)) {
        ec = detail::make_system_error();
    }
#elif defined(GHC_OS_MACOS)
#ifdef __MAC_OS_X_VERSION_MIN_REQUIRED
#if __MAC_OS_X_VERSION_MIN_REQUIRED < 101300
    struct ::stat fs;
    if (::stat(p.c_str(), &fs) == 0) {
        struct ::timeval tv[2];
        tv[0].tv_sec = fs.st_atimespec.tv_sec;
        tv[0].tv_usec = static_cast<int>(fs.st_atimespec.tv_nsec / 1000);
        tv[1].tv_sec = std::chrono::duration_cast<std::chrono::seconds>(d).count();
        tv[1].tv_usec = static_cast<int>(std::chrono::duration_cast<std::chrono::microseconds>(d).count() % 1000000);
        if (::utimes(p.c_str(), tv) == 0) {
            return;
        }
    }
    ec = detail::make_system_error();
    return;
#else
    struct ::timespec times[2];
    times[0].tv_sec = 0;
    times[0].tv_nsec = UTIME_OMIT;
    times[1].tv_sec = std::chrono::duration_cast<std::chrono::seconds>(d).count();
    times[1].tv_nsec = 0;  // std::chrono::duration_cast<std::chrono::nanoseconds>(d).count() % 1000000000;
    if (::utimensat(AT_FDCWD, p.c_str(), times, AT_SYMLINK_NOFOLLOW) != 0) {
        ec = detail::make_system_error();
    }
    return;
#endif
#endif
#else
#ifndef UTIME_OMIT
#define UTIME_OMIT ((1l << 30) - 2l)
#endif
    struct ::timespec times[2];
    times[0].tv_sec = 0;
    times[0].tv_nsec = UTIME_OMIT;
    times[1].tv_sec = static_cast<decltype(times[1].tv_sec)>(std::chrono::duration_cast<std::chrono::seconds>(d).count());
    times[1].tv_nsec = static_cast<decltype(times[1].tv_nsec)>(std::chrono::duration_cast<std::chrono::nanoseconds>(d).count() % 1000000000);
#if defined(__ANDROID_API__) && __ANDROID_API__ < 12
    if (syscall(__NR_utimensat, AT_FDCWD, p.c_str(), times, AT_SYMLINK_NOFOLLOW) != 0) {
#else
    if (::utimensat((int)AT_FDCWD, p.c_str(), times, AT_SYMLINK_NOFOLLOW) != 0) {
#endif
        ec = detail::make_system_error();
    }
    return;
#endif
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE void permissions(const path& p, perms prms, perm_options opts)
{
    std::error_code ec;
    permissions(p, prms, opts, ec);
    if (ec) {
        throw filesystem_error(detail::systemErrorText(ec.value()), p, ec);
    }
}
#endif

GHC_INLINE void permissions(const path& p, perms prms, std::error_code& ec) noexcept
{
    permissions(p, prms, perm_options::replace, ec);
}

GHC_INLINE void permissions(const path& p, perms prms, perm_options opts, std::error_code& ec) noexcept
{
    if (static_cast<int>(opts & (perm_options::replace | perm_options::add | perm_options::remove)) == 0) {
        ec = detail::make_error_code(detail::portable_error::invalid_argument);
        return;
    }
    auto fs = symlink_status(p, ec);
    if ((opts & perm_options::replace) != perm_options::replace) {
        if ((opts & perm_options::add) == perm_options::add) {
            prms = fs.permissions() | prms;
        }
        else {
            prms = fs.permissions() & ~prms;
        }
    }
#ifdef GHC_OS_WINDOWS
#ifdef __GNUC__
    auto oldAttr = GetFileAttributesW(GHC_NATIVEWP(p));
    if (oldAttr != INVALID_FILE_ATTRIBUTES) {
        DWORD newAttr = ((prms & perms::owner_write) == perms::owner_write) ? oldAttr & ~(static_cast<DWORD>(FILE_ATTRIBUTE_READONLY)) : oldAttr | FILE_ATTRIBUTE_READONLY;
        if (oldAttr == newAttr || SetFileAttributesW(GHC_NATIVEWP(p), newAttr)) {
            return;
        }
    }
    ec = detail::make_system_error();
#else
    int mode = 0;
    if ((prms & perms::owner_read) == perms::owner_read) {
        mode |= _S_IREAD;
    }
    if ((prms & perms::owner_write) == perms::owner_write) {
        mode |= _S_IWRITE;
    }
    if (::_wchmod(p.wstring().c_str(), mode) != 0) {
        ec = detail::make_system_error();
    }
#endif
#else
    if ((opts & perm_options::nofollow) != perm_options::nofollow) {
        if (::chmod(p.c_str(), static_cast<mode_t>(prms)) != 0) {
            ec = detail::make_system_error();
        }
    }
#endif
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE path proximate(const path& p, std::error_code& ec)
{
    auto cp = current_path(ec);
    if (!ec) {
        return proximate(p, cp, ec);
    }
    return path();
}
#endif

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE path proximate(const path& p, const path& base)
{
    return weakly_canonical(p).lexically_proximate(weakly_canonical(base));
}
#endif

GHC_INLINE path proximate(const path& p, const path& base, std::error_code& ec)
{
    return weakly_canonical(p, ec).lexically_proximate(weakly_canonical(base, ec));
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE path read_symlink(const path& p)
{
    std::error_code ec;
    auto result = read_symlink(p, ec);
    if (ec) {
        throw filesystem_error(detail::systemErrorText(ec.value()), p, ec);
    }
    return result;
}
#endif

GHC_INLINE path read_symlink(const path& p, std::error_code& ec)
{
    file_status fs = symlink_status(p, ec);
    if (fs.type() != file_type::symlink) {
        ec = detail::make_error_code(detail::portable_error::invalid_argument);
        return path();
    }
    auto result = detail::resolveSymlink(p, ec);
    return ec ? path() : result;
}

GHC_INLINE path relative(const path& p, std::error_code& ec)
{
    return relative(p, current_path(ec), ec);
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE path relative(const path& p, const path& base)
{
    return weakly_canonical(p).lexically_relative(weakly_canonical(base));
}
#endif

GHC_INLINE path relative(const path& p, const path& base, std::error_code& ec)
{
    return weakly_canonical(p, ec).lexically_relative(weakly_canonical(base, ec));
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE bool remove(const path& p)
{
    std::error_code ec;
    auto result = remove(p, ec);
    if (ec) {
        throw filesystem_error(detail::systemErrorText(ec.value()), p, ec);
    }
    return result;
}
#endif

GHC_INLINE bool remove(const path& p, std::error_code& ec) noexcept
{
    ec.clear();
#ifdef GHC_OS_WINDOWS
#ifdef GHC_USE_WCHAR_T
    auto cstr = p.c_str();
#else
    std::wstring np = detail::fromUtf8<std::wstring>(p.u8string());
    auto cstr = np.c_str();
#endif
    DWORD attr = GetFileAttributesW(cstr);
    if (attr == INVALID_FILE_ATTRIBUTES) {
        auto error = ::GetLastError();
        if (error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND) {
            return false;
        }
        ec = detail::make_system_error(error);
    }
    else if (attr & FILE_ATTRIBUTE_READONLY) {
        auto new_attr = attr & ~static_cast<DWORD>(FILE_ATTRIBUTE_READONLY);
        if (!SetFileAttributesW(cstr, new_attr)) {
            auto error = ::GetLastError();
            ec = detail::make_system_error(error);
        }
    }
    if (!ec) {
        if (attr & FILE_ATTRIBUTE_DIRECTORY) {
            if (!RemoveDirectoryW(cstr)) {
                ec = detail::make_system_error();
            }
        }
        else {
            if (!DeleteFileW(cstr)) {
                ec = detail::make_system_error();
            }
        }
    }
#else
    if (::remove(p.c_str()) == -1) {
        auto error = errno;
        if (error == ENOENT) {
            return false;
        }
        ec = detail::make_system_error();
    }
#endif
    return ec ? false : true;
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE uintmax_t remove_all(const path& p)
{
    std::error_code ec;
    auto result = remove_all(p, ec);
    if (ec) {
        throw filesystem_error(detail::systemErrorText(ec.value()), p, ec);
    }
    return result;
}
#endif

GHC_INLINE uintmax_t remove_all(const path& p, std::error_code& ec) noexcept
{
    ec.clear();
    uintmax_t count = 0;
    if (p == "/") {
        ec = detail::make_error_code(detail::portable_error::not_supported);
        return static_cast<uintmax_t>(-1);
    }
    std::error_code tec;
    auto fs = symlink_status(p, tec);
    if (exists(fs) && is_directory(fs)) {
        for (auto iter = directory_iterator(p, ec); iter != directory_iterator(); iter.increment(ec)) {
            if (ec && !detail::is_not_found_error(ec)) {
                break;
            }
            bool is_symlink_result = iter->is_symlink(ec);
            if (ec)
                return static_cast<uintmax_t>(-1);
            if (!is_symlink_result && iter->is_directory(ec)) {
                count += remove_all(iter->path(), ec);
                if (ec) {
                    return static_cast<uintmax_t>(-1);
                }
            }
            else {
                if (!ec) {
                    remove(iter->path(), ec);
                }
                if (ec) {
                    return static_cast<uintmax_t>(-1);
                }
                ++count;
            }
        }
    }
    if (!ec) {
        if (remove(p, ec)) {
            ++count;
        }
    }
    if (ec) {
        return static_cast<uintmax_t>(-1);
    }
    return count;
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE void rename(const path& from, const path& to)
{
    std::error_code ec;
    rename(from, to, ec);
    if (ec) {
        throw filesystem_error(detail::systemErrorText(ec.value()), from, to, ec);
    }
}
#endif

GHC_INLINE void rename(const path& from, const path& to, std::error_code& ec) noexcept
{
    ec.clear();
#ifdef GHC_OS_WINDOWS
    if (from != to) {
        if (!MoveFileExW(GHC_NATIVEWP(from), GHC_NATIVEWP(to), (DWORD)MOVEFILE_REPLACE_EXISTING)) {
            ec = detail::make_system_error();
        }
    }
#else
    if (from != to) {
        if (::rename(from.c_str(), to.c_str()) != 0) {
            ec = detail::make_system_error();
        }
    }
#endif
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE void resize_file(const path& p, uintmax_t size)
{
    std::error_code ec;
    resize_file(p, size, ec);
    if (ec) {
        throw filesystem_error(detail::systemErrorText(ec.value()), p, ec);
    }
}
#endif

GHC_INLINE void resize_file(const path& p, uintmax_t size, std::error_code& ec) noexcept
{
    ec.clear();
#ifdef GHC_OS_WINDOWS
    LARGE_INTEGER lisize;
    lisize.QuadPart = static_cast<LONGLONG>(size);
    if (lisize.QuadPart < 0) {
#ifdef ERROR_FILE_TOO_LARGE
        ec = detail::make_system_error(ERROR_FILE_TOO_LARGE);
#else
        ec = detail::make_system_error(223);
#endif
        return;
    }
    detail::unique_handle file(CreateFileW(GHC_NATIVEWP(p), GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL));
    if (!file) {
        ec = detail::make_system_error();
    }
    else if (SetFilePointerEx(file.get(), lisize, NULL, FILE_BEGIN) == 0 || SetEndOfFile(file.get()) == 0) {
        ec = detail::make_system_error();
    }
#else
    if (::truncate(p.c_str(), static_cast<off_t>(size)) != 0) {
        ec = detail::make_system_error();
    }
#endif
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE space_info space(const path& p)
{
    std::error_code ec;
    auto result = space(p, ec);
    if (ec) {
        throw filesystem_error(detail::systemErrorText(ec.value()), p, ec);
    }
    return result;
}
#endif

GHC_INLINE space_info space(const path& p, std::error_code& ec) noexcept
{
    ec.clear();
#ifdef GHC_OS_WINDOWS
    ULARGE_INTEGER freeBytesAvailableToCaller = {{ 0, 0 }};
    ULARGE_INTEGER totalNumberOfBytes = {{ 0, 0 }};
    ULARGE_INTEGER totalNumberOfFreeBytes = {{ 0, 0 }};
    if (!GetDiskFreeSpaceExW(GHC_NATIVEWP(p), &freeBytesAvailableToCaller, &totalNumberOfBytes, &totalNumberOfFreeBytes)) {
        ec = detail::make_system_error();
        return {static_cast<uintmax_t>(-1), static_cast<uintmax_t>(-1), static_cast<uintmax_t>(-1)};
    }
    return {static_cast<uintmax_t>(totalNumberOfBytes.QuadPart), static_cast<uintmax_t>(totalNumberOfFreeBytes.QuadPart), static_cast<uintmax_t>(freeBytesAvailableToCaller.QuadPart)};
#else
    struct ::statvfs sfs;
    if (::statvfs(p.c_str(), &sfs) != 0) {
        ec = detail::make_system_error();
        return {static_cast<uintmax_t>(-1), static_cast<uintmax_t>(-1), static_cast<uintmax_t>(-1)};
    }
    return {static_cast<uintmax_t>(sfs.f_blocks) * static_cast<uintmax_t>(sfs.f_frsize), static_cast<uintmax_t>(sfs.f_bfree) * static_cast<uintmax_t>(sfs.f_frsize), static_cast<uintmax_t>(sfs.f_bavail) * static_cast<uintmax_t>(sfs.f_frsize)};
#endif
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE file_status status(const path& p)
{
    std::error_code ec;
    auto result = status(p, ec);
    if (result.type() == file_type::none) {
        throw filesystem_error(detail::systemErrorText(ec.value()), p, ec);
    }
    return result;
}
#endif

GHC_INLINE file_status status(const path& p, std::error_code& ec) noexcept
{
    return detail::status_ex(p, ec);
}

GHC_INLINE bool status_known(file_status s) noexcept
{
    return s.type() != file_type::none;
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE file_status symlink_status(const path& p)
{
    std::error_code ec;
    auto result = symlink_status(p, ec);
    if (result.type() == file_type::none) {
        throw filesystem_error(detail::systemErrorText(ec.value()), ec);
    }
    return result;
}
#endif

GHC_INLINE file_status symlink_status(const path& p, std::error_code& ec) noexcept
{
    return detail::symlink_status_ex(p, ec);
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE path temp_directory_path()
{
    std::error_code ec;
    path result = temp_directory_path(ec);
    if (ec) {
        throw filesystem_error(detail::systemErrorText(ec.value()), ec);
    }
    return result;
}
#endif

GHC_INLINE path temp_directory_path(std::error_code& ec) noexcept
{
    ec.clear();
#ifdef GHC_OS_WINDOWS
    wchar_t buffer[512];
    auto rc = GetTempPathW(511, buffer);
    if (!rc || rc > 511) {
        ec = detail::make_system_error();
        return path();
    }
    return path(std::wstring(buffer));
#else
    static const char* temp_vars[] = {"TMPDIR", "TMP", "TEMP", "TEMPDIR", nullptr};
    const char* temp_path = nullptr;
    for (auto temp_name = temp_vars; *temp_name != nullptr; ++temp_name) {
        temp_path = std::getenv(*temp_name);
        if (temp_path) {
            return path(temp_path);
        }
    }
    return path("/tmp");
#endif
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE path weakly_canonical(const path& p)
{
    std::error_code ec;
    auto result = weakly_canonical(p, ec);
    if (ec) {
        throw filesystem_error(detail::systemErrorText(ec.value()), p, ec);
    }
    return result;
}
#endif

GHC_INLINE path weakly_canonical(const path& p, std::error_code& ec) noexcept
{
    path result;
    ec.clear();
    bool scan = true;
    for (auto pe : p) {
        if (scan) {
            std::error_code tec;
            if (exists(result / pe, tec)) {
                result /= pe;
            }
            else {
                if (ec) {
                    return path();
                }
                scan = false;
                if (!result.empty()) {
                    result = canonical(result, ec) / pe;
                    if (ec) {
                        break;
                    }
                }
                else {
                    result /= pe;
                }
            }
        }
        else {
            result /= pe;
        }
    }
    if (scan) {
        if (!result.empty()) {
            result = canonical(result, ec);
        }
    }
    return ec ? path() : result.lexically_normal();
}

//-----------------------------------------------------------------------------
// [fs.class.file_status] class file_status
// [fs.file_status.cons] constructors and destructor
GHC_INLINE file_status::file_status() noexcept
    : file_status(file_type::none)
{
}

GHC_INLINE file_status::file_status(file_type ft, perms prms) noexcept
    : _type(ft)
    , _perms(prms)
{
}

GHC_INLINE file_status::file_status(const file_status& other) noexcept
    : _type(other._type)
    , _perms(other._perms)
{
}

GHC_INLINE file_status::file_status(file_status&& other) noexcept
    : _type(other._type)
    , _perms(other._perms)
{
}

GHC_INLINE file_status::~file_status() {}

// assignments:
GHC_INLINE file_status& file_status::operator=(const file_status& rhs) noexcept
{
    _type = rhs._type;
    _perms = rhs._perms;
    return *this;
}

GHC_INLINE file_status& file_status::operator=(file_status&& rhs) noexcept
{
    _type = rhs._type;
    _perms = rhs._perms;
    return *this;
}

// [fs.file_status.mods] modifiers
GHC_INLINE void file_status::type(file_type ft) noexcept
{
    _type = ft;
}

GHC_INLINE void file_status::permissions(perms prms) noexcept
{
    _perms = prms;
}

// [fs.file_status.obs] observers
GHC_INLINE file_type file_status::type() const noexcept
{
    return _type;
}

GHC_INLINE perms file_status::permissions() const noexcept
{
    return _perms;
}

//-----------------------------------------------------------------------------
// [fs.class.directory_entry] class directory_entry
// [fs.dir.entry.cons] constructors and destructor
// directory_entry::directory_entry() noexcept = default;
// directory_entry::directory_entry(const directory_entry&) = default;
// directory_entry::directory_entry(directory_entry&&) noexcept = default;
#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE directory_entry::directory_entry(const filesystem::path& p)
    : _path(p)
    , _file_size(static_cast<uintmax_t>(-1))
#ifndef GHC_OS_WINDOWS
    , _hard_link_count(static_cast<uintmax_t>(-1))
#endif
    , _last_write_time(0)
{
    refresh();
}
#endif

GHC_INLINE directory_entry::directory_entry(const filesystem::path& p, std::error_code& ec)
    : _path(p)
    , _file_size(static_cast<uintmax_t>(-1))
#ifndef GHC_OS_WINDOWS
    , _hard_link_count(static_cast<uintmax_t>(-1))
#endif
    , _last_write_time(0)
{
    refresh(ec);
}

GHC_INLINE directory_entry::~directory_entry() {}

// assignments:
// directory_entry& directory_entry::operator=(const directory_entry&) = default;
// directory_entry& directory_entry::operator=(directory_entry&&) noexcept = default;

// [fs.dir.entry.mods] directory_entry modifiers
#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE void directory_entry::assign(const filesystem::path& p)
{
    _path = p;
    refresh();
}
#endif

GHC_INLINE void directory_entry::assign(const filesystem::path& p, std::error_code& ec)
{
    _path = p;
    refresh(ec);
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE void directory_entry::replace_filename(const filesystem::path& p)
{
    _path.replace_filename(p);
    refresh();
}
#endif

GHC_INLINE void directory_entry::replace_filename(const filesystem::path& p, std::error_code& ec)
{
    _path.replace_filename(p);
    refresh(ec);
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE void directory_entry::refresh()
{
    std::error_code ec;
    refresh(ec);
    if (ec) {
        throw filesystem_error(detail::systemErrorText(ec.value()), _path, ec);
    }
}
#endif

GHC_INLINE void directory_entry::refresh(std::error_code& ec) noexcept
{
#ifdef GHC_OS_WINDOWS
    _status = detail::status_ex(_path, ec, &_symlink_status, &_file_size, nullptr, &_last_write_time);
#else
    _status = detail::status_ex(_path, ec, &_symlink_status, &_file_size, &_hard_link_count, &_last_write_time);
#endif
}

// [fs.dir.entry.obs] directory_entry observers
GHC_INLINE const filesystem::path& directory_entry::path() const noexcept
{
    return _path;
}

GHC_INLINE directory_entry::operator const filesystem::path&() const noexcept
{
    return _path;
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE file_type directory_entry::status_file_type() const
{
    return _status.type() != file_type::none ? _status.type() : filesystem::status(path()).type();
}
#endif

GHC_INLINE file_type directory_entry::status_file_type(std::error_code& ec) const noexcept
{
    if (_status.type() != file_type::none) {
        ec.clear();
        return _status.type();
    }
    return filesystem::status(path(), ec).type();
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE bool directory_entry::exists() const
{
    return status_file_type() != file_type::not_found;
}
#endif

GHC_INLINE bool directory_entry::exists(std::error_code& ec) const noexcept
{
    return status_file_type(ec) != file_type::not_found;
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE bool directory_entry::is_block_file() const
{
    return status_file_type() == file_type::block;
}
#endif
GHC_INLINE bool directory_entry::is_block_file(std::error_code& ec) const noexcept
{
    return status_file_type(ec) == file_type::block;
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE bool directory_entry::is_character_file() const
{
    return status_file_type() == file_type::character;
}
#endif

GHC_INLINE bool directory_entry::is_character_file(std::error_code& ec) const noexcept
{
    return status_file_type(ec) == file_type::character;
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE bool directory_entry::is_directory() const
{
    return status_file_type() == file_type::directory;
}
#endif

GHC_INLINE bool directory_entry::is_directory(std::error_code& ec) const noexcept
{
    return status_file_type(ec) == file_type::directory;
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE bool directory_entry::is_fifo() const
{
    return status_file_type() == file_type::fifo;
}
#endif

GHC_INLINE bool directory_entry::is_fifo(std::error_code& ec) const noexcept
{
    return status_file_type(ec) == file_type::fifo;
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE bool directory_entry::is_other() const
{
    auto ft = status_file_type();
    return ft != file_type::none && ft != file_type::not_found && ft != file_type::regular && ft != file_type::directory && !is_symlink();
}
#endif

GHC_INLINE bool directory_entry::is_other(std::error_code& ec) const noexcept
{
    auto ft = status_file_type(ec);
    bool other = ft != file_type::none && ft != file_type::not_found && ft != file_type::regular && ft != file_type::directory && !is_symlink(ec);
    return !ec && other;
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE bool directory_entry::is_regular_file() const
{
    return status_file_type() == file_type::regular;
}
#endif

GHC_INLINE bool directory_entry::is_regular_file(std::error_code& ec) const noexcept
{
    return status_file_type(ec) == file_type::regular;
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE bool directory_entry::is_socket() const
{
    return status_file_type() == file_type::socket;
}
#endif

GHC_INLINE bool directory_entry::is_socket(std::error_code& ec) const noexcept
{
    return status_file_type(ec) == file_type::socket;
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE bool directory_entry::is_symlink() const
{
    return _symlink_status.type() != file_type::none ? _symlink_status.type() == file_type::symlink : filesystem::is_symlink(symlink_status());
}
#endif

GHC_INLINE bool directory_entry::is_symlink(std::error_code& ec) const noexcept
{
    if (_symlink_status.type() != file_type::none) {
        ec.clear();
        return _symlink_status.type() == file_type::symlink;
    }
    return filesystem::is_symlink(symlink_status(ec));
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE uintmax_t directory_entry::file_size() const
{
    if (_file_size != static_cast<uintmax_t>(-1)) {
        return _file_size;
    }
    return filesystem::file_size(path());
}
#endif

GHC_INLINE uintmax_t directory_entry::file_size(std::error_code& ec) const noexcept
{
    if (_file_size != static_cast<uintmax_t>(-1)) {
        ec.clear();
        return _file_size;
    }
    return filesystem::file_size(path(), ec);
}

#ifndef GHC_OS_WEB
#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE uintmax_t directory_entry::hard_link_count() const
{
#ifndef GHC_OS_WINDOWS
    if (_hard_link_count != static_cast<uintmax_t>(-1)) {
        return _hard_link_count;
    }
#endif
    return filesystem::hard_link_count(path());
}
#endif

GHC_INLINE uintmax_t directory_entry::hard_link_count(std::error_code& ec) const noexcept
{
#ifndef GHC_OS_WINDOWS
    if (_hard_link_count != static_cast<uintmax_t>(-1)) {
        ec.clear();
        return _hard_link_count;
    }
#endif
    return filesystem::hard_link_count(path(), ec);
}
#endif

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE file_time_type directory_entry::last_write_time() const
{
    if (_last_write_time != 0) {
        return std::chrono::system_clock::from_time_t(_last_write_time);
    }
    return filesystem::last_write_time(path());
}
#endif

GHC_INLINE file_time_type directory_entry::last_write_time(std::error_code& ec) const noexcept
{
    if (_last_write_time != 0) {
        ec.clear();
        return std::chrono::system_clock::from_time_t(_last_write_time);
    }
    return filesystem::last_write_time(path(), ec);
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE file_status directory_entry::status() const
{
    if (_status.type() != file_type::none && _status.permissions() != perms::unknown) {
        return _status;
    }
    return filesystem::status(path());
}
#endif

GHC_INLINE file_status directory_entry::status(std::error_code& ec) const noexcept
{
    if (_status.type() != file_type::none && _status.permissions() != perms::unknown) {
        ec.clear();
        return _status;
    }
    return filesystem::status(path(), ec);
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE file_status directory_entry::symlink_status() const
{
    if (_symlink_status.type() != file_type::none && _symlink_status.permissions() != perms::unknown) {
        return _symlink_status;
    }
    return filesystem::symlink_status(path());
}
#endif

GHC_INLINE file_status directory_entry::symlink_status(std::error_code& ec) const noexcept
{
    if (_symlink_status.type() != file_type::none && _symlink_status.permissions() != perms::unknown) {
        ec.clear();
        return _symlink_status;
    }
    return filesystem::symlink_status(path(), ec);
}

#ifdef GHC_HAS_THREEWAY_COMP
GHC_INLINE std::strong_ordering directory_entry::operator<=>(const directory_entry& rhs) const noexcept
{
    return _path <=> rhs._path;
}
#endif

GHC_INLINE bool directory_entry::operator<(const directory_entry& rhs) const noexcept
{
    return _path < rhs._path;
}

GHC_INLINE bool directory_entry::operator==(const directory_entry& rhs) const noexcept
{
    return _path == rhs._path;
}

GHC_INLINE bool directory_entry::operator!=(const directory_entry& rhs) const noexcept
{
    return _path != rhs._path;
}

GHC_INLINE bool directory_entry::operator<=(const directory_entry& rhs) const noexcept
{
    return _path <= rhs._path;
}

GHC_INLINE bool directory_entry::operator>(const directory_entry& rhs) const noexcept
{
    return _path > rhs._path;
}

GHC_INLINE bool directory_entry::operator>=(const directory_entry& rhs) const noexcept
{
    return _path >= rhs._path;
}

//-----------------------------------------------------------------------------
// [fs.class.directory_iterator] class directory_iterator

#ifdef GHC_OS_WINDOWS
class directory_iterator::impl
{
public:
    impl(const path& p, directory_options options)
        : _base(p)
        , _options(options)
        , _dirHandle(INVALID_HANDLE_VALUE)
    {
        if (!_base.empty()) {
            ZeroMemory(&_findData, sizeof(WIN32_FIND_DATAW));
            if ((_dirHandle = FindFirstFileW(GHC_NATIVEWP((_base / "*")), &_findData)) != INVALID_HANDLE_VALUE) {
                if (std::wstring(_findData.cFileName) == L"." || std::wstring(_findData.cFileName) == L"..") {
                    increment(_ec);
                }
                else {
                    _dir_entry._path = _base / std::wstring(_findData.cFileName);
                    copyToDirEntry(_ec);
                }
            }
            else {
                auto error = ::GetLastError();
                _base = filesystem::path();
                if (error != ERROR_ACCESS_DENIED || (options & directory_options::skip_permission_denied) == directory_options::none) {
                    _ec = detail::make_system_error();
                }
            }
        }
    }
    impl(const impl& other) = delete;
    ~impl()
    {
        if (_dirHandle != INVALID_HANDLE_VALUE) {
            FindClose(_dirHandle);
            _dirHandle = INVALID_HANDLE_VALUE;
        }
    }
    void increment(std::error_code& ec)
    {
        if (_dirHandle != INVALID_HANDLE_VALUE) {
            do {
                if (FindNextFileW(_dirHandle, &_findData)) {
                    _dir_entry._path = _base;
#ifdef GHC_USE_WCHAR_T
                    _dir_entry._path.append_name(_findData.cFileName);
#else
#ifdef GHC_RAISE_UNICODE_ERRORS
                    try {
                        _dir_entry._path.append_name(detail::toUtf8(_findData.cFileName).c_str());
                    }
                    catch (filesystem_error& fe) {
                        ec = fe.code();
                        return;
                    }
#else
                    _dir_entry._path.append_name(detail::toUtf8(_findData.cFileName).c_str());
#endif
#endif
                    copyToDirEntry(ec);
                }
                else {
                    auto err = ::GetLastError();
                    if (err != ERROR_NO_MORE_FILES) {
                        _ec = ec = detail::make_system_error(err);
                    }
                    FindClose(_dirHandle);
                    _dirHandle = INVALID_HANDLE_VALUE;
                    _dir_entry._path.clear();
                    break;
                }
            } while (std::wstring(_findData.cFileName) == L"." || std::wstring(_findData.cFileName) == L"..");
        }
        else {
            ec = _ec;
        }
    }
    void copyToDirEntry(std::error_code& ec)
    {
        if (_findData.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) {
            _dir_entry._status = detail::status_ex(_dir_entry._path, ec, &_dir_entry._symlink_status, &_dir_entry._file_size, nullptr, &_dir_entry._last_write_time);
        }
        else {
            _dir_entry._status = detail::status_from_INFO(_dir_entry._path, &_findData, ec, &_dir_entry._file_size, &_dir_entry._last_write_time);
            _dir_entry._symlink_status = _dir_entry._status;
        }
        if (ec) {
            if (_dir_entry._status.type() != file_type::none && _dir_entry._symlink_status.type() != file_type::none) {
                ec.clear();
            }
            else {
                _dir_entry._file_size = static_cast<uintmax_t>(-1);
                _dir_entry._last_write_time = 0;
            }
        }
    }
    path _base;
    directory_options _options;
    WIN32_FIND_DATAW _findData;
    HANDLE _dirHandle;
    directory_entry _dir_entry;
    std::error_code _ec;
};
#else
// POSIX implementation
class directory_iterator::impl
{
public:
    impl(const path& path, directory_options options)
        : _base(path)
        , _options(options)
        , _dir(nullptr)
        , _entry(nullptr)
    {
        if (!path.empty()) {
            _dir = ::opendir(path.native().c_str());
            if (!_dir) {
                auto error = errno;
                _base = filesystem::path();
                if ((error != EACCES && error != EPERM) || (options & directory_options::skip_permission_denied) == directory_options::none) {
                    _ec = detail::make_system_error();
                }
            }
            else {
                increment(_ec);
            }
        }
    }
    impl(const impl& other) = delete;
    ~impl()
    {
        if (_dir) {
            ::closedir(_dir);
        }
    }
    void increment(std::error_code& ec)
    {
        if (_dir) {
            bool skip;
            do {
                skip = false;
                errno = 0;
                _entry = ::readdir(_dir);
                if (_entry) {
                    _dir_entry._path = _base;
                    _dir_entry._path.append_name(_entry->d_name);
                    copyToDirEntry();
                    if (ec && (ec.value() == EACCES || ec.value() == EPERM) && (_options & directory_options::skip_permission_denied) == directory_options::skip_permission_denied) {
                        ec.clear();
                        skip = true;
                    }
                }
                else {
                    ::closedir(_dir);
                    _dir = nullptr;
                    _dir_entry._path.clear();
                    if (errno) {
                        ec = detail::make_system_error();
                    }
                    break;
                }
            } while (skip || std::strcmp(_entry->d_name, ".") == 0 || std::strcmp(_entry->d_name, "..") == 0);
        }
    }

    void copyToDirEntry()
    {
        _dir_entry._symlink_status.permissions(perms::unknown);
        auto ft = detail::file_type_from_dirent(*_entry);
        _dir_entry._symlink_status.type(ft);
        if (ft != file_type::symlink) {
            _dir_entry._status = _dir_entry._symlink_status;
        }
        else {
            _dir_entry._status.type(file_type::none);
            _dir_entry._status.permissions(perms::unknown);
        }
        _dir_entry._file_size = static_cast<uintmax_t>(-1);
        _dir_entry._hard_link_count = static_cast<uintmax_t>(-1);
        _dir_entry._last_write_time = 0;
    }
    path _base;
    directory_options _options;
    DIR* _dir;
    struct ::dirent* _entry;
    directory_entry _dir_entry;
    std::error_code _ec;
};
#endif

// [fs.dir.itr.members] member functions
GHC_INLINE directory_iterator::directory_iterator() noexcept
    : _impl(new impl(path(), directory_options::none))
{
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE directory_iterator::directory_iterator(const path& p)
    : _impl(new impl(p, directory_options::none))
{
    if (_impl->_ec) {
        throw filesystem_error(detail::systemErrorText(_impl->_ec.value()), p, _impl->_ec);
    }
    _impl->_ec.clear();
}

GHC_INLINE directory_iterator::directory_iterator(const path& p, directory_options options)
    : _impl(new impl(p, options))
{
    if (_impl->_ec) {
        throw filesystem_error(detail::systemErrorText(_impl->_ec.value()), p, _impl->_ec);
    }
}
#endif

GHC_INLINE directory_iterator::directory_iterator(const path& p, std::error_code& ec) noexcept
    : _impl(new impl(p, directory_options::none))
{
    if (_impl->_ec) {
        ec = _impl->_ec;
    }
}

GHC_INLINE directory_iterator::directory_iterator(const path& p, directory_options options, std::error_code& ec) noexcept
    : _impl(new impl(p, options))
{
    if (_impl->_ec) {
        ec = _impl->_ec;
    }
}

GHC_INLINE directory_iterator::directory_iterator(const directory_iterator& rhs)
    : _impl(rhs._impl)
{
}

GHC_INLINE directory_iterator::directory_iterator(directory_iterator&& rhs) noexcept
    : _impl(std::move(rhs._impl))
{
}

GHC_INLINE directory_iterator::~directory_iterator() {}

GHC_INLINE directory_iterator& directory_iterator::operator=(const directory_iterator& rhs)
{
    _impl = rhs._impl;
    return *this;
}

GHC_INLINE directory_iterator& directory_iterator::operator=(directory_iterator&& rhs) noexcept
{
    _impl = std::move(rhs._impl);
    return *this;
}

GHC_INLINE const directory_entry& directory_iterator::operator*() const
{
    return _impl->_dir_entry;
}

GHC_INLINE const directory_entry* directory_iterator::operator->() const
{
    return &_impl->_dir_entry;
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE directory_iterator& directory_iterator::operator++()
{
    std::error_code ec;
    _impl->increment(ec);
    if (ec) {
        throw filesystem_error(detail::systemErrorText(ec.value()), _impl->_dir_entry._path, ec);
    }
    return *this;
}
#endif

GHC_INLINE directory_iterator& directory_iterator::increment(std::error_code& ec) noexcept
{
    _impl->increment(ec);
    return *this;
}

GHC_INLINE bool directory_iterator::operator==(const directory_iterator& rhs) const
{
    return _impl->_dir_entry._path == rhs._impl->_dir_entry._path;
}

GHC_INLINE bool directory_iterator::operator!=(const directory_iterator& rhs) const
{
    return _impl->_dir_entry._path != rhs._impl->_dir_entry._path;
}

// [fs.dir.itr.nonmembers] directory_iterator non-member functions

GHC_INLINE directory_iterator begin(directory_iterator iter) noexcept
{
    return iter;
}

GHC_INLINE directory_iterator end(const directory_iterator&) noexcept
{
    return directory_iterator();
}

//-----------------------------------------------------------------------------
// [fs.class.rec.dir.itr] class recursive_directory_iterator

GHC_INLINE recursive_directory_iterator::recursive_directory_iterator() noexcept
    : _impl(new recursive_directory_iterator_impl(directory_options::none, true))
{
    _impl->_dir_iter_stack.push(directory_iterator());
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE recursive_directory_iterator::recursive_directory_iterator(const path& p)
    : _impl(new recursive_directory_iterator_impl(directory_options::none, true))
{
    _impl->_dir_iter_stack.push(directory_iterator(p));
}

GHC_INLINE recursive_directory_iterator::recursive_directory_iterator(const path& p, directory_options options)
    : _impl(new recursive_directory_iterator_impl(options, true))
{
    _impl->_dir_iter_stack.push(directory_iterator(p, options));
}
#endif

GHC_INLINE recursive_directory_iterator::recursive_directory_iterator(const path& p, directory_options options, std::error_code& ec) noexcept
    : _impl(new recursive_directory_iterator_impl(options, true))
{
    _impl->_dir_iter_stack.push(directory_iterator(p, options, ec));
}

GHC_INLINE recursive_directory_iterator::recursive_directory_iterator(const path& p, std::error_code& ec) noexcept
    : _impl(new recursive_directory_iterator_impl(directory_options::none, true))
{
    _impl->_dir_iter_stack.push(directory_iterator(p, ec));
}

GHC_INLINE recursive_directory_iterator::recursive_directory_iterator(const recursive_directory_iterator& rhs)
    : _impl(rhs._impl)
{
}

GHC_INLINE recursive_directory_iterator::recursive_directory_iterator(recursive_directory_iterator&& rhs) noexcept
    : _impl(std::move(rhs._impl))
{
}

GHC_INLINE recursive_directory_iterator::~recursive_directory_iterator() {}

// [fs.rec.dir.itr.members] observers
GHC_INLINE directory_options recursive_directory_iterator::options() const
{
    return _impl->_options;
}

GHC_INLINE int recursive_directory_iterator::depth() const
{
    return static_cast<int>(_impl->_dir_iter_stack.size() - 1);
}

GHC_INLINE bool recursive_directory_iterator::recursion_pending() const
{
    return _impl->_recursion_pending;
}

GHC_INLINE const directory_entry& recursive_directory_iterator::operator*() const
{
    return *(_impl->_dir_iter_stack.top());
}

GHC_INLINE const directory_entry* recursive_directory_iterator::operator->() const
{
    return &(*(_impl->_dir_iter_stack.top()));
}

// [fs.rec.dir.itr.members] modifiers recursive_directory_iterator&
GHC_INLINE recursive_directory_iterator& recursive_directory_iterator::operator=(const recursive_directory_iterator& rhs)
{
    _impl = rhs._impl;
    return *this;
}

GHC_INLINE recursive_directory_iterator& recursive_directory_iterator::operator=(recursive_directory_iterator&& rhs) noexcept
{
    _impl = std::move(rhs._impl);
    return *this;
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE recursive_directory_iterator& recursive_directory_iterator::operator++()
{
    std::error_code ec;
    increment(ec);
    if (ec) {
        throw filesystem_error(detail::systemErrorText(ec.value()), _impl->_dir_iter_stack.empty() ? path() : _impl->_dir_iter_stack.top()->path(), ec);
    }
    return *this;
}
#endif

GHC_INLINE recursive_directory_iterator& recursive_directory_iterator::increment(std::error_code& ec) noexcept
{
    bool isSymLink = (*this)->is_symlink(ec);
    bool isDir = !ec && (*this)->is_directory(ec);
    if (isSymLink && detail::is_not_found_error(ec)) {
        ec.clear();
    }
    if (!ec) {
        if (recursion_pending() && isDir && (!isSymLink || (options() & directory_options::follow_directory_symlink) != directory_options::none)) {
            _impl->_dir_iter_stack.push(directory_iterator((*this)->path(), _impl->_options, ec));
        }
        else {
            _impl->_dir_iter_stack.top().increment(ec);
        }
        if (!ec) {
            while (depth() && _impl->_dir_iter_stack.top() == directory_iterator()) {
                _impl->_dir_iter_stack.pop();
                _impl->_dir_iter_stack.top().increment(ec);
            }
        }
        else if (!_impl->_dir_iter_stack.empty()) {
            _impl->_dir_iter_stack.pop();
        }
        _impl->_recursion_pending = true;
    }
    return *this;
}

#ifdef GHC_WITH_EXCEPTIONS
GHC_INLINE void recursive_directory_iterator::pop()
{
    std::error_code ec;
    pop(ec);
    if (ec) {
        throw filesystem_error(detail::systemErrorText(ec.value()), _impl->_dir_iter_stack.empty() ? path() : _impl->_dir_iter_stack.top()->path(), ec);
    }
}
#endif

GHC_INLINE void recursive_directory_iterator::pop(std::error_code& ec)
{
    if (depth() == 0) {
        *this = recursive_directory_iterator();
    }
    else {
        do {
            _impl->_dir_iter_stack.pop();
            _impl->_dir_iter_stack.top().increment(ec);
        } while (depth() && _impl->_dir_iter_stack.top() == directory_iterator());
    }
}

GHC_INLINE void recursive_directory_iterator::disable_recursion_pending()
{
    _impl->_recursion_pending = false;
}

// other members as required by [input.iterators]
GHC_INLINE bool recursive_directory_iterator::operator==(const recursive_directory_iterator& rhs) const
{
    return _impl->_dir_iter_stack.top() == rhs._impl->_dir_iter_stack.top();
}

GHC_INLINE bool recursive_directory_iterator::operator!=(const recursive_directory_iterator& rhs) const
{
    return _impl->_dir_iter_stack.top() != rhs._impl->_dir_iter_stack.top();
}

// [fs.rec.dir.itr.nonmembers] directory_iterator non-member functions
GHC_INLINE recursive_directory_iterator begin(recursive_directory_iterator iter) noexcept
{
    return iter;
}

GHC_INLINE recursive_directory_iterator end(const recursive_directory_iterator&) noexcept
{
    return recursive_directory_iterator();
}

#endif  // GHC_EXPAND_IMPL

}  // namespace filesystem
}  // namespace ghc

// cleanup some macros
#undef GHC_INLINE
#undef GHC_EXPAND_IMPL

#endif  // GHC_FILESYSTEM_H
