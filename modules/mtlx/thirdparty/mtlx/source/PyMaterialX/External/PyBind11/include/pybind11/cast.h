/*
    pybind11/cast.h: Partial template specializations to cast between
    C++ and Python types

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "detail/common.h"
#include "detail/descr.h"
#include "detail/type_caster_base.h"
#include "detail/typeid.h"
#include "pytypes.h"

#include <array>
#include <cstring>
#include <functional>
#include <iosfwd>
#include <iterator>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

PYBIND11_WARNING_DISABLE_MSVC(4127)

PYBIND11_NAMESPACE_BEGIN(detail)

template <typename type, typename SFINAE = void>
class type_caster : public type_caster_base<type> {};
template <typename type>
using make_caster = type_caster<intrinsic_t<type>>;

// Shortcut for calling a caster's `cast_op_type` cast operator for casting a type_caster to a T
template <typename T>
typename make_caster<T>::template cast_op_type<T> cast_op(make_caster<T> &caster) {
    return caster.operator typename make_caster<T>::template cast_op_type<T>();
}
template <typename T>
typename make_caster<T>::template cast_op_type<typename std::add_rvalue_reference<T>::type>
cast_op(make_caster<T> &&caster) {
    return std::move(caster).operator typename make_caster<T>::
        template cast_op_type<typename std::add_rvalue_reference<T>::type>();
}

template <typename type>
class type_caster<std::reference_wrapper<type>> {
private:
    using caster_t = make_caster<type>;
    caster_t subcaster;
    using reference_t = type &;
    using subcaster_cast_op_type = typename caster_t::template cast_op_type<reference_t>;

    static_assert(
        std::is_same<typename std::remove_const<type>::type &, subcaster_cast_op_type>::value
            || std::is_same<reference_t, subcaster_cast_op_type>::value,
        "std::reference_wrapper<T> caster requires T to have a caster with an "
        "`operator T &()` or `operator const T &()`");

public:
    bool load(handle src, bool convert) { return subcaster.load(src, convert); }
    static constexpr auto name = caster_t::name;
    static handle
    cast(const std::reference_wrapper<type> &src, return_value_policy policy, handle parent) {
        // It is definitely wrong to take ownership of this pointer, so mask that rvp
        if (policy == return_value_policy::take_ownership
            || policy == return_value_policy::automatic) {
            policy = return_value_policy::automatic_reference;
        }
        return caster_t::cast(&src.get(), policy, parent);
    }
    template <typename T>
    using cast_op_type = std::reference_wrapper<type>;
    explicit operator std::reference_wrapper<type>() { return cast_op<type &>(subcaster); }
};

#define PYBIND11_TYPE_CASTER(type, py_name)                                                       \
protected:                                                                                        \
    type value;                                                                                   \
                                                                                                  \
public:                                                                                           \
    static constexpr auto name = py_name;                                                         \
    template <typename T_,                                                                        \
              ::pybind11::detail::enable_if_t<                                                    \
                  std::is_same<type, ::pybind11::detail::remove_cv_t<T_>>::value,                 \
                  int>                                                                            \
              = 0>                                                                                \
    static ::pybind11::handle cast(                                                               \
        T_ *src, ::pybind11::return_value_policy policy, ::pybind11::handle parent) {             \
        if (!src)                                                                                 \
            return ::pybind11::none().release();                                                  \
        if (policy == ::pybind11::return_value_policy::take_ownership) {                          \
            auto h = cast(std::move(*src), policy, parent);                                       \
            delete src;                                                                           \
            return h;                                                                             \
        }                                                                                         \
        return cast(*src, policy, parent);                                                        \
    }                                                                                             \
    operator type *() { return &value; }               /* NOLINT(bugprone-macro-parentheses) */   \
    operator type &() { return value; }                /* NOLINT(bugprone-macro-parentheses) */   \
    operator type &&() && { return std::move(value); } /* NOLINT(bugprone-macro-parentheses) */   \
    template <typename T_>                                                                        \
    using cast_op_type = ::pybind11::detail::movable_cast_op_type<T_>

template <typename CharT>
using is_std_char_type = any_of<std::is_same<CharT, char>, /* std::string */
#if defined(PYBIND11_HAS_U8STRING)
                                std::is_same<CharT, char8_t>, /* std::u8string */
#endif
                                std::is_same<CharT, char16_t>, /* std::u16string */
                                std::is_same<CharT, char32_t>, /* std::u32string */
                                std::is_same<CharT, wchar_t>   /* std::wstring */
                                >;

template <typename T>
struct type_caster<T, enable_if_t<std::is_arithmetic<T>::value && !is_std_char_type<T>::value>> {
    using _py_type_0 = conditional_t<sizeof(T) <= sizeof(long), long, long long>;
    using _py_type_1 = conditional_t<std::is_signed<T>::value,
                                     _py_type_0,
                                     typename std::make_unsigned<_py_type_0>::type>;
    using py_type = conditional_t<std::is_floating_point<T>::value, double, _py_type_1>;

public:
    bool load(handle src, bool convert) {
        py_type py_value;

        if (!src) {
            return false;
        }

#if !defined(PYPY_VERSION)
        auto index_check = [](PyObject *o) { return PyIndex_Check(o); };
#else
        // In PyPy 7.3.3, `PyIndex_Check` is implemented by calling `__index__`,
        // while CPython only considers the existence of `nb_index`/`__index__`.
        auto index_check = [](PyObject *o) { return hasattr(o, "__index__"); };
#endif

        if (std::is_floating_point<T>::value) {
            if (convert || PyFloat_Check(src.ptr())) {
                py_value = (py_type) PyFloat_AsDouble(src.ptr());
            } else {
                return false;
            }
        } else if (PyFloat_Check(src.ptr())
                   || (!convert && !PYBIND11_LONG_CHECK(src.ptr()) && !index_check(src.ptr()))) {
            return false;
        } else {
            handle src_or_index = src;
            // PyPy: 7.3.7's 3.8 does not implement PyLong_*'s __index__ calls.
#if PY_VERSION_HEX < 0x03080000 || defined(PYPY_VERSION)
            object index;
            if (!PYBIND11_LONG_CHECK(src.ptr())) { // So: index_check(src.ptr())
                index = reinterpret_steal<object>(PyNumber_Index(src.ptr()));
                if (!index) {
                    PyErr_Clear();
                    if (!convert)
                        return false;
                } else {
                    src_or_index = index;
                }
            }
#endif
            if (std::is_unsigned<py_type>::value) {
                py_value = as_unsigned<py_type>(src_or_index.ptr());
            } else { // signed integer:
                py_value = sizeof(T) <= sizeof(long)
                               ? (py_type) PyLong_AsLong(src_or_index.ptr())
                               : (py_type) PYBIND11_LONG_AS_LONGLONG(src_or_index.ptr());
            }
        }

        // Python API reported an error
        bool py_err = py_value == (py_type) -1 && PyErr_Occurred();

        // Check to see if the conversion is valid (integers should match exactly)
        // Signed/unsigned checks happen elsewhere
        if (py_err
            || (std::is_integral<T>::value && sizeof(py_type) != sizeof(T)
                && py_value != (py_type) (T) py_value)) {
            PyErr_Clear();
            if (py_err && convert && (PyNumber_Check(src.ptr()) != 0)) {
                auto tmp = reinterpret_steal<object>(std::is_floating_point<T>::value
                                                         ? PyNumber_Float(src.ptr())
                                                         : PyNumber_Long(src.ptr()));
                PyErr_Clear();
                return load(tmp, false);
            }
            return false;
        }

        value = (T) py_value;
        return true;
    }

    template <typename U = T>
    static typename std::enable_if<std::is_floating_point<U>::value, handle>::type
    cast(U src, return_value_policy /* policy */, handle /* parent */) {
        return PyFloat_FromDouble((double) src);
    }

    template <typename U = T>
    static typename std::enable_if<!std::is_floating_point<U>::value && std::is_signed<U>::value
                                       && (sizeof(U) <= sizeof(long)),
                                   handle>::type
    cast(U src, return_value_policy /* policy */, handle /* parent */) {
        return PYBIND11_LONG_FROM_SIGNED((long) src);
    }

    template <typename U = T>
    static typename std::enable_if<!std::is_floating_point<U>::value && std::is_unsigned<U>::value
                                       && (sizeof(U) <= sizeof(unsigned long)),
                                   handle>::type
    cast(U src, return_value_policy /* policy */, handle /* parent */) {
        return PYBIND11_LONG_FROM_UNSIGNED((unsigned long) src);
    }

    template <typename U = T>
    static typename std::enable_if<!std::is_floating_point<U>::value && std::is_signed<U>::value
                                       && (sizeof(U) > sizeof(long)),
                                   handle>::type
    cast(U src, return_value_policy /* policy */, handle /* parent */) {
        return PyLong_FromLongLong((long long) src);
    }

    template <typename U = T>
    static typename std::enable_if<!std::is_floating_point<U>::value && std::is_unsigned<U>::value
                                       && (sizeof(U) > sizeof(unsigned long)),
                                   handle>::type
    cast(U src, return_value_policy /* policy */, handle /* parent */) {
        return PyLong_FromUnsignedLongLong((unsigned long long) src);
    }

    PYBIND11_TYPE_CASTER(T, const_name<std::is_integral<T>::value>("int", "float"));
};

template <typename T>
struct void_caster {
public:
    bool load(handle src, bool) {
        if (src && src.is_none()) {
            return true;
        }
        return false;
    }
    static handle cast(T, return_value_policy /* policy */, handle /* parent */) {
        return none().release();
    }
    PYBIND11_TYPE_CASTER(T, const_name("None"));
};

template <>
class type_caster<void_type> : public void_caster<void_type> {};

template <>
class type_caster<void> : public type_caster<void_type> {
public:
    using type_caster<void_type>::cast;

    bool load(handle h, bool) {
        if (!h) {
            return false;
        }
        if (h.is_none()) {
            value = nullptr;
            return true;
        }

        /* Check if this is a capsule */
        if (isinstance<capsule>(h)) {
            value = reinterpret_borrow<capsule>(h);
            return true;
        }

        /* Check if this is a C++ type */
        const auto &bases = all_type_info((PyTypeObject *) type::handle_of(h).ptr());
        if (bases.size() == 1) { // Only allowing loading from a single-value type
            value = values_and_holders(reinterpret_cast<instance *>(h.ptr())).begin()->value_ptr();
            return true;
        }

        /* Fail */
        return false;
    }

    static handle cast(const void *ptr, return_value_policy /* policy */, handle /* parent */) {
        if (ptr) {
            return capsule(ptr).release();
        }
        return none().release();
    }

    template <typename T>
    using cast_op_type = void *&;
    explicit operator void *&() { return value; }
    static constexpr auto name = const_name("capsule");

private:
    void *value = nullptr;
};

template <>
class type_caster<std::nullptr_t> : public void_caster<std::nullptr_t> {};

template <>
class type_caster<bool> {
public:
    bool load(handle src, bool convert) {
        if (!src) {
            return false;
        }
        if (src.ptr() == Py_True) {
            value = true;
            return true;
        }
        if (src.ptr() == Py_False) {
            value = false;
            return true;
        }
        if (convert || (std::strcmp("numpy.bool_", Py_TYPE(src.ptr())->tp_name) == 0)) {
            // (allow non-implicit conversion for numpy booleans)

            Py_ssize_t res = -1;
            if (src.is_none()) {
                res = 0; // None is implicitly converted to False
            }
#if defined(PYPY_VERSION)
            // On PyPy, check that "__bool__" attr exists
            else if (hasattr(src, PYBIND11_BOOL_ATTR)) {
                res = PyObject_IsTrue(src.ptr());
            }
#else
            // Alternate approach for CPython: this does the same as the above, but optimized
            // using the CPython API so as to avoid an unneeded attribute lookup.
            else if (auto *tp_as_number = src.ptr()->ob_type->tp_as_number) {
                if (PYBIND11_NB_BOOL(tp_as_number)) {
                    res = (*PYBIND11_NB_BOOL(tp_as_number))(src.ptr());
                }
            }
#endif
            if (res == 0 || res == 1) {
                value = (res != 0);
                return true;
            }
            PyErr_Clear();
        }
        return false;
    }
    static handle cast(bool src, return_value_policy /* policy */, handle /* parent */) {
        return handle(src ? Py_True : Py_False).inc_ref();
    }
    PYBIND11_TYPE_CASTER(bool, const_name("bool"));
};

// Helper class for UTF-{8,16,32} C++ stl strings:
template <typename StringType, bool IsView = false>
struct string_caster {
    using CharT = typename StringType::value_type;

    // Simplify life by being able to assume standard char sizes (the standard only guarantees
    // minimums, but Python requires exact sizes)
    static_assert(!std::is_same<CharT, char>::value || sizeof(CharT) == 1,
                  "Unsupported char size != 1");
#if defined(PYBIND11_HAS_U8STRING)
    static_assert(!std::is_same<CharT, char8_t>::value || sizeof(CharT) == 1,
                  "Unsupported char8_t size != 1");
#endif
    static_assert(!std::is_same<CharT, char16_t>::value || sizeof(CharT) == 2,
                  "Unsupported char16_t size != 2");
    static_assert(!std::is_same<CharT, char32_t>::value || sizeof(CharT) == 4,
                  "Unsupported char32_t size != 4");
    // wchar_t can be either 16 bits (Windows) or 32 (everywhere else)
    static_assert(!std::is_same<CharT, wchar_t>::value || sizeof(CharT) == 2 || sizeof(CharT) == 4,
                  "Unsupported wchar_t size != 2/4");
    static constexpr size_t UTF_N = 8 * sizeof(CharT);

    bool load(handle src, bool) {
        handle load_src = src;
        if (!src) {
            return false;
        }
        if (!PyUnicode_Check(load_src.ptr())) {
            return load_raw(load_src);
        }

        // For UTF-8 we avoid the need for a temporary `bytes` object by using
        // `PyUnicode_AsUTF8AndSize`.
        if (UTF_N == 8) {
            Py_ssize_t size = -1;
            const auto *buffer
                = reinterpret_cast<const CharT *>(PyUnicode_AsUTF8AndSize(load_src.ptr(), &size));
            if (!buffer) {
                PyErr_Clear();
                return false;
            }
            value = StringType(buffer, static_cast<size_t>(size));
            return true;
        }

        auto utfNbytes
            = reinterpret_steal<object>(PyUnicode_AsEncodedString(load_src.ptr(),
                                                                  UTF_N == 8    ? "utf-8"
                                                                  : UTF_N == 16 ? "utf-16"
                                                                                : "utf-32",
                                                                  nullptr));
        if (!utfNbytes) {
            PyErr_Clear();
            return false;
        }

        const auto *buffer
            = reinterpret_cast<const CharT *>(PYBIND11_BYTES_AS_STRING(utfNbytes.ptr()));
        size_t length = (size_t) PYBIND11_BYTES_SIZE(utfNbytes.ptr()) / sizeof(CharT);
        // Skip BOM for UTF-16/32
        if (UTF_N > 8) {
            buffer++;
            length--;
        }
        value = StringType(buffer, length);

        // If we're loading a string_view we need to keep the encoded Python object alive:
        if (IsView) {
            loader_life_support::add_patient(utfNbytes);
        }

        return true;
    }

    static handle
    cast(const StringType &src, return_value_policy /* policy */, handle /* parent */) {
        const char *buffer = reinterpret_cast<const char *>(src.data());
        auto nbytes = ssize_t(src.size() * sizeof(CharT));
        handle s = decode_utfN(buffer, nbytes);
        if (!s) {
            throw error_already_set();
        }
        return s;
    }

    PYBIND11_TYPE_CASTER(StringType, const_name(PYBIND11_STRING_NAME));

private:
    static handle decode_utfN(const char *buffer, ssize_t nbytes) {
#if !defined(PYPY_VERSION)
        return UTF_N == 8    ? PyUnicode_DecodeUTF8(buffer, nbytes, nullptr)
               : UTF_N == 16 ? PyUnicode_DecodeUTF16(buffer, nbytes, nullptr, nullptr)
                             : PyUnicode_DecodeUTF32(buffer, nbytes, nullptr, nullptr);
#else
        // PyPy segfaults when on PyUnicode_DecodeUTF16 (and possibly on PyUnicode_DecodeUTF32 as
        // well), so bypass the whole thing by just passing the encoding as a string value, which
        // works properly:
        return PyUnicode_Decode(buffer,
                                nbytes,
                                UTF_N == 8    ? "utf-8"
                                : UTF_N == 16 ? "utf-16"
                                              : "utf-32",
                                nullptr);
#endif
    }

    // When loading into a std::string or char*, accept a bytes/bytearray object as-is (i.e.
    // without any encoding/decoding attempt).  For other C++ char sizes this is a no-op.
    // which supports loading a unicode from a str, doesn't take this path.
    template <typename C = CharT>
    bool load_raw(enable_if_t<std::is_same<C, char>::value, handle> src) {
        if (PYBIND11_BYTES_CHECK(src.ptr())) {
            // We were passed raw bytes; accept it into a std::string or char*
            // without any encoding attempt.
            const char *bytes = PYBIND11_BYTES_AS_STRING(src.ptr());
            if (!bytes) {
                pybind11_fail("Unexpected PYBIND11_BYTES_AS_STRING() failure.");
            }
            value = StringType(bytes, (size_t) PYBIND11_BYTES_SIZE(src.ptr()));
            return true;
        }
        if (PyByteArray_Check(src.ptr())) {
            // We were passed a bytearray; accept it into a std::string or char*
            // without any encoding attempt.
            const char *bytearray = PyByteArray_AsString(src.ptr());
            if (!bytearray) {
                pybind11_fail("Unexpected PyByteArray_AsString() failure.");
            }
            value = StringType(bytearray, (size_t) PyByteArray_Size(src.ptr()));
            return true;
        }

        return false;
    }

    template <typename C = CharT>
    bool load_raw(enable_if_t<!std::is_same<C, char>::value, handle>) {
        return false;
    }
};

template <typename CharT, class Traits, class Allocator>
struct type_caster<std::basic_string<CharT, Traits, Allocator>,
                   enable_if_t<is_std_char_type<CharT>::value>>
    : string_caster<std::basic_string<CharT, Traits, Allocator>> {};

#ifdef PYBIND11_HAS_STRING_VIEW
template <typename CharT, class Traits>
struct type_caster<std::basic_string_view<CharT, Traits>,
                   enable_if_t<is_std_char_type<CharT>::value>>
    : string_caster<std::basic_string_view<CharT, Traits>, true> {};
#endif

// Type caster for C-style strings.  We basically use a std::string type caster, but also add the
// ability to use None as a nullptr char* (which the string caster doesn't allow).
template <typename CharT>
struct type_caster<CharT, enable_if_t<is_std_char_type<CharT>::value>> {
    using StringType = std::basic_string<CharT>;
    using StringCaster = make_caster<StringType>;
    StringCaster str_caster;
    bool none = false;
    CharT one_char = 0;

public:
    bool load(handle src, bool convert) {
        if (!src) {
            return false;
        }
        if (src.is_none()) {
            // Defer accepting None to other overloads (if we aren't in convert mode):
            if (!convert) {
                return false;
            }
            none = true;
            return true;
        }
        return str_caster.load(src, convert);
    }

    static handle cast(const CharT *src, return_value_policy policy, handle parent) {
        if (src == nullptr) {
            return pybind11::none().release();
        }
        return StringCaster::cast(StringType(src), policy, parent);
    }

    static handle cast(CharT src, return_value_policy policy, handle parent) {
        if (std::is_same<char, CharT>::value) {
            handle s = PyUnicode_DecodeLatin1((const char *) &src, 1, nullptr);
            if (!s) {
                throw error_already_set();
            }
            return s;
        }
        return StringCaster::cast(StringType(1, src), policy, parent);
    }

    explicit operator CharT *() {
        return none ? nullptr : const_cast<CharT *>(static_cast<StringType &>(str_caster).c_str());
    }
    explicit operator CharT &() {
        if (none) {
            throw value_error("Cannot convert None to a character");
        }

        auto &value = static_cast<StringType &>(str_caster);
        size_t str_len = value.size();
        if (str_len == 0) {
            throw value_error("Cannot convert empty string to a character");
        }

        // If we're in UTF-8 mode, we have two possible failures: one for a unicode character that
        // is too high, and one for multiple unicode characters (caught later), so we need to
        // figure out how long the first encoded character is in bytes to distinguish between these
        // two errors.  We also allow want to allow unicode characters U+0080 through U+00FF, as
        // those can fit into a single char value.
        if (StringCaster::UTF_N == 8 && str_len > 1 && str_len <= 4) {
            auto v0 = static_cast<unsigned char>(value[0]);
            // low bits only: 0-127
            // 0b110xxxxx - start of 2-byte sequence
            // 0b1110xxxx - start of 3-byte sequence
            // 0b11110xxx - start of 4-byte sequence
            size_t char0_bytes = (v0 & 0x80) == 0      ? 1
                                 : (v0 & 0xE0) == 0xC0 ? 2
                                 : (v0 & 0xF0) == 0xE0 ? 3
                                                       : 4;

            if (char0_bytes == str_len) {
                // If we have a 128-255 value, we can decode it into a single char:
                if (char0_bytes == 2 && (v0 & 0xFC) == 0xC0) { // 0x110000xx 0x10xxxxxx
                    one_char = static_cast<CharT>(((v0 & 3) << 6)
                                                  + (static_cast<unsigned char>(value[1]) & 0x3F));
                    return one_char;
                }
                // Otherwise we have a single character, but it's > U+00FF
                throw value_error("Character code point not in range(0x100)");
            }
        }

        // UTF-16 is much easier: we can only have a surrogate pair for values above U+FFFF, thus a
        // surrogate pair with total length 2 instantly indicates a range error (but not a "your
        // string was too long" error).
        else if (StringCaster::UTF_N == 16 && str_len == 2) {
            one_char = static_cast<CharT>(value[0]);
            if (one_char >= 0xD800 && one_char < 0xE000) {
                throw value_error("Character code point not in range(0x10000)");
            }
        }

        if (str_len != 1) {
            throw value_error("Expected a character, but multi-character string found");
        }

        one_char = value[0];
        return one_char;
    }

    static constexpr auto name = const_name(PYBIND11_STRING_NAME);
    template <typename _T>
    using cast_op_type = pybind11::detail::cast_op_type<_T>;
};

// Base implementation for std::tuple and std::pair
template <template <typename...> class Tuple, typename... Ts>
class tuple_caster {
    using type = Tuple<Ts...>;
    static constexpr auto size = sizeof...(Ts);
    using indices = make_index_sequence<size>;

public:
    bool load(handle src, bool convert) {
        if (!isinstance<sequence>(src)) {
            return false;
        }
        const auto seq = reinterpret_borrow<sequence>(src);
        if (seq.size() != size) {
            return false;
        }
        return load_impl(seq, convert, indices{});
    }

    template <typename T>
    static handle cast(T &&src, return_value_policy policy, handle parent) {
        return cast_impl(std::forward<T>(src), policy, parent, indices{});
    }

    // copied from the PYBIND11_TYPE_CASTER macro
    template <typename T>
    static handle cast(T *src, return_value_policy policy, handle parent) {
        if (!src) {
            return none().release();
        }
        if (policy == return_value_policy::take_ownership) {
            auto h = cast(std::move(*src), policy, parent);
            delete src;
            return h;
        }
        return cast(*src, policy, parent);
    }

    static constexpr auto name
        = const_name("Tuple[") + concat(make_caster<Ts>::name...) + const_name("]");

    template <typename T>
    using cast_op_type = type;

    explicit operator type() & { return implicit_cast(indices{}); }
    explicit operator type() && { return std::move(*this).implicit_cast(indices{}); }

protected:
    template <size_t... Is>
    type implicit_cast(index_sequence<Is...>) & {
        return type(cast_op<Ts>(std::get<Is>(subcasters))...);
    }
    template <size_t... Is>
    type implicit_cast(index_sequence<Is...>) && {
        return type(cast_op<Ts>(std::move(std::get<Is>(subcasters)))...);
    }

    static constexpr bool load_impl(const sequence &, bool, index_sequence<>) { return true; }

    template <size_t... Is>
    bool load_impl(const sequence &seq, bool convert, index_sequence<Is...>) {
#ifdef __cpp_fold_expressions
        if ((... || !std::get<Is>(subcasters).load(seq[Is], convert))) {
            return false;
        }
#else
        for (bool r : {std::get<Is>(subcasters).load(seq[Is], convert)...}) {
            if (!r) {
                return false;
            }
        }
#endif
        return true;
    }

    /* Implementation: Convert a C++ tuple into a Python tuple */
    template <typename T, size_t... Is>
    static handle
    cast_impl(T &&src, return_value_policy policy, handle parent, index_sequence<Is...>) {
        PYBIND11_WORKAROUND_INCORRECT_MSVC_C4100(src, policy, parent);
        PYBIND11_WORKAROUND_INCORRECT_GCC_UNUSED_BUT_SET_PARAMETER(policy, parent);
        std::array<object, size> entries{{reinterpret_steal<object>(
            make_caster<Ts>::cast(std::get<Is>(std::forward<T>(src)), policy, parent))...}};
        for (const auto &entry : entries) {
            if (!entry) {
                return handle();
            }
        }
        tuple result(size);
        int counter = 0;
        for (auto &entry : entries) {
            PyTuple_SET_ITEM(result.ptr(), counter++, entry.release().ptr());
        }
        return result.release();
    }

    Tuple<make_caster<Ts>...> subcasters;
};

template <typename T1, typename T2>
class type_caster<std::pair<T1, T2>> : public tuple_caster<std::pair, T1, T2> {};

template <typename... Ts>
class type_caster<std::tuple<Ts...>> : public tuple_caster<std::tuple, Ts...> {};

/// Helper class which abstracts away certain actions. Users can provide specializations for
/// custom holders, but it's only necessary if the type has a non-standard interface.
template <typename T>
struct holder_helper {
    static auto get(const T &p) -> decltype(p.get()) { return p.get(); }
};

/// Type caster for holder types like std::shared_ptr, etc.
/// The SFINAE hook is provided to help work around the current lack of support
/// for smart-pointer interoperability. Please consider it an implementation
/// detail that may change in the future, as formal support for smart-pointer
/// interoperability is added into pybind11.
template <typename type, typename holder_type, typename SFINAE = void>
struct copyable_holder_caster : public type_caster_base<type> {
public:
    using base = type_caster_base<type>;
    static_assert(std::is_base_of<base, type_caster<type>>::value,
                  "Holder classes are only supported for custom types");
    using base::base;
    using base::cast;
    using base::typeinfo;
    using base::value;

    bool load(handle src, bool convert) {
        return base::template load_impl<copyable_holder_caster<type, holder_type>>(src, convert);
    }

    explicit operator type *() { return this->value; }
    // static_cast works around compiler error with MSVC 17 and CUDA 10.2
    // see issue #2180
    explicit operator type &() { return *(static_cast<type *>(this->value)); }
    explicit operator holder_type *() { return std::addressof(holder); }
    explicit operator holder_type &() { return holder; }

    static handle cast(const holder_type &src, return_value_policy, handle) {
        const auto *ptr = holder_helper<holder_type>::get(src);
        return type_caster_base<type>::cast_holder(ptr, &src);
    }

protected:
    friend class type_caster_generic;
    void check_holder_compat() {
        if (typeinfo->default_holder) {
            throw cast_error("Unable to load a custom holder type from a default-holder instance");
        }
    }

    bool load_value(value_and_holder &&v_h) {
        if (v_h.holder_constructed()) {
            value = v_h.value_ptr();
            holder = v_h.template holder<holder_type>();
            return true;
        }
        throw cast_error("Unable to cast from non-held to held instance (T& to Holder<T>) "
#if !defined(PYBIND11_DETAILED_ERROR_MESSAGES)
                         "(#define PYBIND11_DETAILED_ERROR_MESSAGES or compile in debug mode for "
                         "type information)");
#else
                         "of type '"
                         + type_id<holder_type>() + "''");
#endif
    }

    template <typename T = holder_type,
              detail::enable_if_t<!std::is_constructible<T, const T &, type *>::value, int> = 0>
    bool try_implicit_casts(handle, bool) {
        return false;
    }

    template <typename T = holder_type,
              detail::enable_if_t<std::is_constructible<T, const T &, type *>::value, int> = 0>
    bool try_implicit_casts(handle src, bool convert) {
        for (auto &cast : typeinfo->implicit_casts) {
            copyable_holder_caster sub_caster(*cast.first);
            if (sub_caster.load(src, convert)) {
                value = cast.second(sub_caster.value);
                holder = holder_type(sub_caster.holder, (type *) value);
                return true;
            }
        }
        return false;
    }

    static bool try_direct_conversions(handle) { return false; }

    holder_type holder;
};

/// Specialize for the common std::shared_ptr, so users don't need to
template <typename T>
class type_caster<std::shared_ptr<T>> : public copyable_holder_caster<T, std::shared_ptr<T>> {};

/// Type caster for holder types like std::unique_ptr.
/// Please consider the SFINAE hook an implementation detail, as explained
/// in the comment for the copyable_holder_caster.
template <typename type, typename holder_type, typename SFINAE = void>
struct move_only_holder_caster {
    static_assert(std::is_base_of<type_caster_base<type>, type_caster<type>>::value,
                  "Holder classes are only supported for custom types");

    static handle cast(holder_type &&src, return_value_policy, handle) {
        auto *ptr = holder_helper<holder_type>::get(src);
        return type_caster_base<type>::cast_holder(ptr, std::addressof(src));
    }
    static constexpr auto name = type_caster_base<type>::name;
};

template <typename type, typename deleter>
class type_caster<std::unique_ptr<type, deleter>>
    : public move_only_holder_caster<type, std::unique_ptr<type, deleter>> {};

template <typename type, typename holder_type>
using type_caster_holder = conditional_t<is_copy_constructible<holder_type>::value,
                                         copyable_holder_caster<type, holder_type>,
                                         move_only_holder_caster<type, holder_type>>;

template <typename T, bool Value = false>
struct always_construct_holder {
    static constexpr bool value = Value;
};

/// Create a specialization for custom holder types (silently ignores std::shared_ptr)
#define PYBIND11_DECLARE_HOLDER_TYPE(type, holder_type, ...)                                      \
    PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)                                                  \
    namespace detail {                                                                            \
    template <typename type>                                                                      \
    struct always_construct_holder<holder_type> : always_construct_holder<void, ##__VA_ARGS__> {  \
    };                                                                                            \
    template <typename type>                                                                      \
    class type_caster<holder_type, enable_if_t<!is_shared_ptr<holder_type>::value>>               \
        : public type_caster_holder<type, holder_type> {};                                        \
    }                                                                                             \
    PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

// PYBIND11_DECLARE_HOLDER_TYPE holder types:
template <typename base, typename holder>
struct is_holder_type
    : std::is_base_of<detail::type_caster_holder<base, holder>, detail::type_caster<holder>> {};
// Specialization for always-supported unique_ptr holders:
template <typename base, typename deleter>
struct is_holder_type<base, std::unique_ptr<base, deleter>> : std::true_type {};

template <typename T>
struct handle_type_name {
    static constexpr auto name = const_name<T>();
};
template <>
struct handle_type_name<bool_> {
    static constexpr auto name = const_name("bool");
};
template <>
struct handle_type_name<bytes> {
    static constexpr auto name = const_name(PYBIND11_BYTES_NAME);
};
template <>
struct handle_type_name<int_> {
    static constexpr auto name = const_name("int");
};
template <>
struct handle_type_name<iterable> {
    static constexpr auto name = const_name("Iterable");
};
template <>
struct handle_type_name<iterator> {
    static constexpr auto name = const_name("Iterator");
};
template <>
struct handle_type_name<float_> {
    static constexpr auto name = const_name("float");
};
template <>
struct handle_type_name<none> {
    static constexpr auto name = const_name("None");
};
template <>
struct handle_type_name<args> {
    static constexpr auto name = const_name("*args");
};
template <>
struct handle_type_name<kwargs> {
    static constexpr auto name = const_name("**kwargs");
};

template <typename type>
struct pyobject_caster {
    template <typename T = type, enable_if_t<std::is_same<T, handle>::value, int> = 0>
    pyobject_caster() : value() {}

    // `type` may not be default constructible (e.g. frozenset, anyset).  Initializing `value`
    // to a nil handle is safe since it will only be accessed if `load` succeeds.
    template <typename T = type, enable_if_t<std::is_base_of<object, T>::value, int> = 0>
    pyobject_caster() : value(reinterpret_steal<type>(handle())) {}

    template <typename T = type, enable_if_t<std::is_same<T, handle>::value, int> = 0>
    bool load(handle src, bool /* convert */) {
        value = src;
        return static_cast<bool>(value);
    }

    template <typename T = type, enable_if_t<std::is_base_of<object, T>::value, int> = 0>
    bool load(handle src, bool /* convert */) {
        if (!isinstance<type>(src)) {
            return false;
        }
        value = reinterpret_borrow<type>(src);
        return true;
    }

    static handle cast(const handle &src, return_value_policy /* policy */, handle /* parent */) {
        return src.inc_ref();
    }
    PYBIND11_TYPE_CASTER(type, handle_type_name<type>::name);
};

template <typename T>
class type_caster<T, enable_if_t<is_pyobject<T>::value>> : public pyobject_caster<T> {};

// Our conditions for enabling moving are quite restrictive:
// At compile time:
// - T needs to be a non-const, non-pointer, non-reference type
// - type_caster<T>::operator T&() must exist
// - the type must be move constructible (obviously)
// At run-time:
// - if the type is non-copy-constructible, the object must be the sole owner of the type (i.e. it
//   must have ref_count() == 1)h
// If any of the above are not satisfied, we fall back to copying.
template <typename T>
using move_is_plain_type
    = satisfies_none_of<T, std::is_void, std::is_pointer, std::is_reference, std::is_const>;
template <typename T, typename SFINAE = void>
struct move_always : std::false_type {};
template <typename T>
struct move_always<
    T,
    enable_if_t<
        all_of<move_is_plain_type<T>,
               negation<is_copy_constructible<T>>,
               std::is_move_constructible<T>,
               std::is_same<decltype(std::declval<make_caster<T>>().operator T &()), T &>>::value>>
    : std::true_type {};
template <typename T, typename SFINAE = void>
struct move_if_unreferenced : std::false_type {};
template <typename T>
struct move_if_unreferenced<
    T,
    enable_if_t<
        all_of<move_is_plain_type<T>,
               negation<move_always<T>>,
               std::is_move_constructible<T>,
               std::is_same<decltype(std::declval<make_caster<T>>().operator T &()), T &>>::value>>
    : std::true_type {};
template <typename T>
using move_never = none_of<move_always<T>, move_if_unreferenced<T>>;

// Detect whether returning a `type` from a cast on type's type_caster is going to result in a
// reference or pointer to a local variable of the type_caster.  Basically, only
// non-reference/pointer `type`s and reference/pointers from a type_caster_generic are safe;
// everything else returns a reference/pointer to a local variable.
template <typename type>
using cast_is_temporary_value_reference
    = bool_constant<(std::is_reference<type>::value || std::is_pointer<type>::value)
                    && !std::is_base_of<type_caster_generic, make_caster<type>>::value
                    && !std::is_same<intrinsic_t<type>, void>::value>;

// When a value returned from a C++ function is being cast back to Python, we almost always want to
// force `policy = move`, regardless of the return value policy the function/method was declared
// with.
template <typename Return, typename SFINAE = void>
struct return_value_policy_override {
    static return_value_policy policy(return_value_policy p) { return p; }
};

template <typename Return>
struct return_value_policy_override<
    Return,
    detail::enable_if_t<std::is_base_of<type_caster_generic, make_caster<Return>>::value, void>> {
    static return_value_policy policy(return_value_policy p) {
        return !std::is_lvalue_reference<Return>::value && !std::is_pointer<Return>::value
                   ? return_value_policy::move
                   : p;
    }
};

// Basic python -> C++ casting; throws if casting fails
template <typename T, typename SFINAE>
type_caster<T, SFINAE> &load_type(type_caster<T, SFINAE> &conv, const handle &handle) {
    static_assert(!detail::is_pyobject<T>::value,
                  "Internal error: type_caster should only be used for C++ types");
    if (!conv.load(handle, true)) {
#if !defined(PYBIND11_DETAILED_ERROR_MESSAGES)
        throw cast_error("Unable to cast Python instance to C++ type (#define "
                         "PYBIND11_DETAILED_ERROR_MESSAGES or compile in debug mode for details)");
#else
        throw cast_error("Unable to cast Python instance of type "
                         + (std::string) str(type::handle_of(handle)) + " to C++ type '"
                         + type_id<T>() + "'");
#endif
    }
    return conv;
}
// Wrapper around the above that also constructs and returns a type_caster
template <typename T>
make_caster<T> load_type(const handle &handle) {
    make_caster<T> conv;
    load_type(conv, handle);
    return conv;
}

PYBIND11_NAMESPACE_END(detail)

// pytype -> C++ type
template <typename T, detail::enable_if_t<!detail::is_pyobject<T>::value, int> = 0>
T cast(const handle &handle) {
    using namespace detail;
    static_assert(!cast_is_temporary_value_reference<T>::value,
                  "Unable to cast type to reference: value is local to type caster");
    return cast_op<T>(load_type<T>(handle));
}

// pytype -> pytype (calls converting constructor)
template <typename T, detail::enable_if_t<detail::is_pyobject<T>::value, int> = 0>
T cast(const handle &handle) {
    return T(reinterpret_borrow<object>(handle));
}

// C++ type -> py::object
template <typename T, detail::enable_if_t<!detail::is_pyobject<T>::value, int> = 0>
object cast(T &&value,
            return_value_policy policy = return_value_policy::automatic_reference,
            handle parent = handle()) {
    using no_ref_T = typename std::remove_reference<T>::type;
    if (policy == return_value_policy::automatic) {
        policy = std::is_pointer<no_ref_T>::value     ? return_value_policy::take_ownership
                 : std::is_lvalue_reference<T>::value ? return_value_policy::copy
                                                      : return_value_policy::move;
    } else if (policy == return_value_policy::automatic_reference) {
        policy = std::is_pointer<no_ref_T>::value     ? return_value_policy::reference
                 : std::is_lvalue_reference<T>::value ? return_value_policy::copy
                                                      : return_value_policy::move;
    }
    return reinterpret_steal<object>(
        detail::make_caster<T>::cast(std::forward<T>(value), policy, parent));
}

template <typename T>
T handle::cast() const {
    return pybind11::cast<T>(*this);
}
template <>
inline void handle::cast() const {
    return;
}

template <typename T>
detail::enable_if_t<!detail::move_never<T>::value, T> move(object &&obj) {
    if (obj.ref_count() > 1) {
#if !defined(PYBIND11_DETAILED_ERROR_MESSAGES)
        throw cast_error(
            "Unable to cast Python instance to C++ rvalue: instance has multiple references"
            " (#define PYBIND11_DETAILED_ERROR_MESSAGES or compile in debug mode for details)");
#else
        throw cast_error("Unable to move from Python " + (std::string) str(type::handle_of(obj))
                         + " instance to C++ " + type_id<T>()
                         + " instance: instance has multiple references");
#endif
    }

    // Move into a temporary and return that, because the reference may be a local value of `conv`
    T ret = std::move(detail::load_type<T>(obj).operator T &());
    return ret;
}

// Calling cast() on an rvalue calls pybind11::cast with the object rvalue, which does:
// - If we have to move (because T has no copy constructor), do it.  This will fail if the moved
//   object has multiple references, but trying to copy will fail to compile.
// - If both movable and copyable, check ref count: if 1, move; otherwise copy
// - Otherwise (not movable), copy.
template <typename T>
detail::enable_if_t<!detail::is_pyobject<T>::value && detail::move_always<T>::value, T>
cast(object &&object) {
    return move<T>(std::move(object));
}
template <typename T>
detail::enable_if_t<!detail::is_pyobject<T>::value && detail::move_if_unreferenced<T>::value, T>
cast(object &&object) {
    if (object.ref_count() > 1) {
        return cast<T>(object);
    }
    return move<T>(std::move(object));
}
template <typename T>
detail::enable_if_t<!detail::is_pyobject<T>::value && detail::move_never<T>::value, T>
cast(object &&object) {
    return cast<T>(object);
}

// pytype rvalue -> pytype (calls converting constructor)
template <typename T>
detail::enable_if_t<detail::is_pyobject<T>::value, T> cast(object &&object) {
    return T(std::move(object));
}

template <typename T>
T object::cast() const & {
    return pybind11::cast<T>(*this);
}
template <typename T>
T object::cast() && {
    return pybind11::cast<T>(std::move(*this));
}
template <>
inline void object::cast() const & {
    return;
}
template <>
inline void object::cast() && {
    return;
}

PYBIND11_NAMESPACE_BEGIN(detail)

// Declared in pytypes.h:
template <typename T, enable_if_t<!is_pyobject<T>::value, int>>
object object_or_cast(T &&o) {
    return pybind11::cast(std::forward<T>(o));
}

// Placeholder type for the unneeded (and dead code) static variable in the
// PYBIND11_OVERRIDE_OVERRIDE macro
struct override_unused {};
template <typename ret_type>
using override_caster_t = conditional_t<cast_is_temporary_value_reference<ret_type>::value,
                                        make_caster<ret_type>,
                                        override_unused>;

// Trampoline use: for reference/pointer types to value-converted values, we do a value cast, then
// store the result in the given variable.  For other types, this is a no-op.
template <typename T>
enable_if_t<cast_is_temporary_value_reference<T>::value, T> cast_ref(object &&o,
                                                                     make_caster<T> &caster) {
    return cast_op<T>(load_type(caster, o));
}
template <typename T>
enable_if_t<!cast_is_temporary_value_reference<T>::value, T> cast_ref(object &&,
                                                                      override_unused &) {
    pybind11_fail("Internal error: cast_ref fallback invoked");
}

// Trampoline use: Having a pybind11::cast with an invalid reference type is going to
// static_assert, even though if it's in dead code, so we provide a "trampoline" to pybind11::cast
// that only does anything in cases where pybind11::cast is valid.
template <typename T>
enable_if_t<cast_is_temporary_value_reference<T>::value, T> cast_safe(object &&) {
    pybind11_fail("Internal error: cast_safe fallback invoked");
}
template <typename T>
enable_if_t<std::is_void<T>::value, void> cast_safe(object &&) {}
template <typename T>
enable_if_t<detail::none_of<cast_is_temporary_value_reference<T>, std::is_void<T>>::value, T>
cast_safe(object &&o) {
    return pybind11::cast<T>(std::move(o));
}

PYBIND11_NAMESPACE_END(detail)

// The overloads could coexist, i.e. the #if is not strictly speaking needed,
// but it is an easy minor optimization.
#if !defined(PYBIND11_DETAILED_ERROR_MESSAGES)
inline cast_error cast_error_unable_to_convert_call_arg() {
    return cast_error("Unable to convert call argument to Python object (#define "
                      "PYBIND11_DETAILED_ERROR_MESSAGES or compile in debug mode for details)");
}
#else
inline cast_error cast_error_unable_to_convert_call_arg(const std::string &name,
                                                        const std::string &type) {
    return cast_error("Unable to convert call argument '" + name + "' of type '" + type
                      + "' to Python object");
}
#endif

template <return_value_policy policy = return_value_policy::automatic_reference>
tuple make_tuple() {
    return tuple(0);
}

template <return_value_policy policy = return_value_policy::automatic_reference, typename... Args>
tuple make_tuple(Args &&...args_) {
    constexpr size_t size = sizeof...(Args);
    std::array<object, size> args{{reinterpret_steal<object>(
        detail::make_caster<Args>::cast(std::forward<Args>(args_), policy, nullptr))...}};
    for (size_t i = 0; i < args.size(); i++) {
        if (!args[i]) {
#if !defined(PYBIND11_DETAILED_ERROR_MESSAGES)
            throw cast_error_unable_to_convert_call_arg();
#else
            std::array<std::string, size> argtypes{{type_id<Args>()...}};
            throw cast_error_unable_to_convert_call_arg(std::to_string(i), argtypes[i]);
#endif
        }
    }
    tuple result(size);
    int counter = 0;
    for (auto &arg_value : args) {
        PyTuple_SET_ITEM(result.ptr(), counter++, arg_value.release().ptr());
    }
    return result;
}

/// \ingroup annotations
/// Annotation for arguments
struct arg {
    /// Constructs an argument with the name of the argument; if null or omitted, this is a
    /// positional argument.
    constexpr explicit arg(const char *name = nullptr)
        : name(name), flag_noconvert(false), flag_none(true) {}
    /// Assign a value to this argument
    template <typename T>
    arg_v operator=(T &&value) const;
    /// Indicate that the type should not be converted in the type caster
    arg &noconvert(bool flag = true) {
        flag_noconvert = flag;
        return *this;
    }
    /// Indicates that the argument should/shouldn't allow None (e.g. for nullable pointer args)
    arg &none(bool flag = true) {
        flag_none = flag;
        return *this;
    }

    const char *name;        ///< If non-null, this is a named kwargs argument
    bool flag_noconvert : 1; ///< If set, do not allow conversion (requires a supporting type
                             ///< caster!)
    bool flag_none : 1;      ///< If set (the default), allow None to be passed to this argument
};

/// \ingroup annotations
/// Annotation for arguments with values
struct arg_v : arg {
private:
    template <typename T>
    arg_v(arg &&base, T &&x, const char *descr = nullptr)
        : arg(base), value(reinterpret_steal<object>(detail::make_caster<T>::cast(
                         std::forward<T>(x), return_value_policy::automatic, {}))),
          descr(descr)
#if defined(PYBIND11_DETAILED_ERROR_MESSAGES)
          ,
          type(type_id<T>())
#endif
    {
        // Workaround! See:
        // https://github.com/pybind/pybind11/issues/2336
        // https://github.com/pybind/pybind11/pull/2685#issuecomment-731286700
        if (PyErr_Occurred()) {
            PyErr_Clear();
        }
    }

public:
    /// Direct construction with name, default, and description
    template <typename T>
    arg_v(const char *name, T &&x, const char *descr = nullptr)
        : arg_v(arg(name), std::forward<T>(x), descr) {}

    /// Called internally when invoking `py::arg("a") = value`
    template <typename T>
    arg_v(const arg &base, T &&x, const char *descr = nullptr)
        : arg_v(arg(base), std::forward<T>(x), descr) {}

    /// Same as `arg::noconvert()`, but returns *this as arg_v&, not arg&
    arg_v &noconvert(bool flag = true) {
        arg::noconvert(flag);
        return *this;
    }

    /// Same as `arg::nonone()`, but returns *this as arg_v&, not arg&
    arg_v &none(bool flag = true) {
        arg::none(flag);
        return *this;
    }

    /// The default value
    object value;
    /// The (optional) description of the default value
    const char *descr;
#if defined(PYBIND11_DETAILED_ERROR_MESSAGES)
    /// The C++ type name of the default value (only available when compiled in debug mode)
    std::string type;
#endif
};

/// \ingroup annotations
/// Annotation indicating that all following arguments are keyword-only; the is the equivalent of
/// an unnamed '*' argument
struct kw_only {};

/// \ingroup annotations
/// Annotation indicating that all previous arguments are positional-only; the is the equivalent of
/// an unnamed '/' argument (in Python 3.8)
struct pos_only {};

template <typename T>
arg_v arg::operator=(T &&value) const {
    return {*this, std::forward<T>(value)};
}

/// Alias for backward compatibility -- to be removed in version 2.0
template <typename /*unused*/>
using arg_t = arg_v;

inline namespace literals {
/** \rst
    String literal version of `arg`
 \endrst */
constexpr arg operator"" _a(const char *name, size_t) { return arg(name); }
} // namespace literals

PYBIND11_NAMESPACE_BEGIN(detail)

template <typename T>
using is_kw_only = std::is_same<intrinsic_t<T>, kw_only>;
template <typename T>
using is_pos_only = std::is_same<intrinsic_t<T>, pos_only>;

// forward declaration (definition in attr.h)
struct function_record;

/// Internal data associated with a single function call
struct function_call {
    function_call(const function_record &f, handle p); // Implementation in attr.h

    /// The function data:
    const function_record &func;

    /// Arguments passed to the function:
    std::vector<handle> args;

    /// The `convert` value the arguments should be loaded with
    std::vector<bool> args_convert;

    /// Extra references for the optional `py::args` and/or `py::kwargs` arguments (which, if
    /// present, are also in `args` but without a reference).
    object args_ref, kwargs_ref;

    /// The parent, if any
    handle parent;

    /// If this is a call to an initializer, this argument contains `self`
    handle init_self;
};

/// Helper class which loads arguments for C++ functions called from Python
template <typename... Args>
class argument_loader {
    using indices = make_index_sequence<sizeof...(Args)>;

    template <typename Arg>
    using argument_is_args = std::is_same<intrinsic_t<Arg>, args>;
    template <typename Arg>
    using argument_is_kwargs = std::is_same<intrinsic_t<Arg>, kwargs>;
    // Get kwargs argument position, or -1 if not present:
    static constexpr auto kwargs_pos = constexpr_last<argument_is_kwargs, Args...>();

    static_assert(kwargs_pos == -1 || kwargs_pos == (int) sizeof...(Args) - 1,
                  "py::kwargs is only permitted as the last argument of a function");

public:
    static constexpr bool has_kwargs = kwargs_pos != -1;

    // py::args argument position; -1 if not present.
    static constexpr int args_pos = constexpr_last<argument_is_args, Args...>();

    static_assert(args_pos == -1 || args_pos == constexpr_first<argument_is_args, Args...>(),
                  "py::args cannot be specified more than once");

    static constexpr auto arg_names = concat(type_descr(make_caster<Args>::name)...);

    bool load_args(function_call &call) { return load_impl_sequence(call, indices{}); }

    template <typename Return, typename Guard, typename Func>
    // NOLINTNEXTLINE(readability-const-return-type)
    enable_if_t<!std::is_void<Return>::value, Return> call(Func &&f) && {
        return std::move(*this).template call_impl<remove_cv_t<Return>>(
            std::forward<Func>(f), indices{}, Guard{});
    }

    template <typename Return, typename Guard, typename Func>
    enable_if_t<std::is_void<Return>::value, void_type> call(Func &&f) && {
        std::move(*this).template call_impl<remove_cv_t<Return>>(
            std::forward<Func>(f), indices{}, Guard{});
        return void_type();
    }

private:
    static bool load_impl_sequence(function_call &, index_sequence<>) { return true; }

    template <size_t... Is>
    bool load_impl_sequence(function_call &call, index_sequence<Is...>) {
#ifdef __cpp_fold_expressions
        if ((... || !std::get<Is>(argcasters).load(call.args[Is], call.args_convert[Is]))) {
            return false;
        }
#else
        for (bool r : {std::get<Is>(argcasters).load(call.args[Is], call.args_convert[Is])...}) {
            if (!r) {
                return false;
            }
        }
#endif
        return true;
    }

    template <typename Return, typename Func, size_t... Is, typename Guard>
    Return call_impl(Func &&f, index_sequence<Is...>, Guard &&) && {
        return std::forward<Func>(f)(cast_op<Args>(std::move(std::get<Is>(argcasters)))...);
    }

    std::tuple<make_caster<Args>...> argcasters;
};

/// Helper class which collects only positional arguments for a Python function call.
/// A fancier version below can collect any argument, but this one is optimal for simple calls.
template <return_value_policy policy>
class simple_collector {
public:
    template <typename... Ts>
    explicit simple_collector(Ts &&...values)
        : m_args(pybind11::make_tuple<policy>(std::forward<Ts>(values)...)) {}

    const tuple &args() const & { return m_args; }
    dict kwargs() const { return {}; }

    tuple args() && { return std::move(m_args); }

    /// Call a Python function and pass the collected arguments
    object call(PyObject *ptr) const {
        PyObject *result = PyObject_CallObject(ptr, m_args.ptr());
        if (!result) {
            throw error_already_set();
        }
        return reinterpret_steal<object>(result);
    }

private:
    tuple m_args;
};

/// Helper class which collects positional, keyword, * and ** arguments for a Python function call
template <return_value_policy policy>
class unpacking_collector {
public:
    template <typename... Ts>
    explicit unpacking_collector(Ts &&...values) {
        // Tuples aren't (easily) resizable so a list is needed for collection,
        // but the actual function call strictly requires a tuple.
        auto args_list = list();
        using expander = int[];
        (void) expander{0, (process(args_list, std::forward<Ts>(values)), 0)...};

        m_args = std::move(args_list);
    }

    const tuple &args() const & { return m_args; }
    const dict &kwargs() const & { return m_kwargs; }

    tuple args() && { return std::move(m_args); }
    dict kwargs() && { return std::move(m_kwargs); }

    /// Call a Python function and pass the collected arguments
    object call(PyObject *ptr) const {
        PyObject *result = PyObject_Call(ptr, m_args.ptr(), m_kwargs.ptr());
        if (!result) {
            throw error_already_set();
        }
        return reinterpret_steal<object>(result);
    }

private:
    template <typename T>
    void process(list &args_list, T &&x) {
        auto o = reinterpret_steal<object>(
            detail::make_caster<T>::cast(std::forward<T>(x), policy, {}));
        if (!o) {
#if !defined(PYBIND11_DETAILED_ERROR_MESSAGES)
            throw cast_error_unable_to_convert_call_arg();
#else
            throw cast_error_unable_to_convert_call_arg(std::to_string(args_list.size()),
                                                        type_id<T>());
#endif
        }
        args_list.append(std::move(o));
    }

    void process(list &args_list, detail::args_proxy ap) {
        for (auto a : ap) {
            args_list.append(a);
        }
    }

    void process(list & /*args_list*/, arg_v a) {
        if (!a.name) {
#if !defined(PYBIND11_DETAILED_ERROR_MESSAGES)
            nameless_argument_error();
#else
            nameless_argument_error(a.type);
#endif
        }
        if (m_kwargs.contains(a.name)) {
#if !defined(PYBIND11_DETAILED_ERROR_MESSAGES)
            multiple_values_error();
#else
            multiple_values_error(a.name);
#endif
        }
        if (!a.value) {
#if !defined(PYBIND11_DETAILED_ERROR_MESSAGES)
            throw cast_error_unable_to_convert_call_arg();
#else
            throw cast_error_unable_to_convert_call_arg(a.name, a.type);
#endif
        }
        m_kwargs[a.name] = std::move(a.value);
    }

    void process(list & /*args_list*/, detail::kwargs_proxy kp) {
        if (!kp) {
            return;
        }
        for (auto k : reinterpret_borrow<dict>(kp)) {
            if (m_kwargs.contains(k.first)) {
#if !defined(PYBIND11_DETAILED_ERROR_MESSAGES)
                multiple_values_error();
#else
                multiple_values_error(str(k.first));
#endif
            }
            m_kwargs[k.first] = k.second;
        }
    }

    [[noreturn]] static void nameless_argument_error() {
        throw type_error(
            "Got kwargs without a name; only named arguments "
            "may be passed via py::arg() to a python function call. "
            "(#define PYBIND11_DETAILED_ERROR_MESSAGES or compile in debug mode for details)");
    }
    [[noreturn]] static void nameless_argument_error(const std::string &type) {
        throw type_error("Got kwargs without a name of type '" + type
                         + "'; only named "
                           "arguments may be passed via py::arg() to a python function call. ");
    }
    [[noreturn]] static void multiple_values_error() {
        throw type_error(
            "Got multiple values for keyword argument "
            "(#define PYBIND11_DETAILED_ERROR_MESSAGES or compile in debug mode for details)");
    }

    [[noreturn]] static void multiple_values_error(const std::string &name) {
        throw type_error("Got multiple values for keyword argument '" + name + "'");
    }

private:
    tuple m_args;
    dict m_kwargs;
};

// [workaround(intel)] Separate function required here
// We need to put this into a separate function because the Intel compiler
// fails to compile enable_if_t<!all_of<is_positional<Args>...>::value>
// (tested with ICC 2021.1 Beta 20200827).
template <typename... Args>
constexpr bool args_are_all_positional() {
    return all_of<is_positional<Args>...>::value;
}

/// Collect only positional arguments for a Python function call
template <return_value_policy policy,
          typename... Args,
          typename = enable_if_t<args_are_all_positional<Args...>()>>
simple_collector<policy> collect_arguments(Args &&...args) {
    return simple_collector<policy>(std::forward<Args>(args)...);
}

/// Collect all arguments, including keywords and unpacking (only instantiated when needed)
template <return_value_policy policy,
          typename... Args,
          typename = enable_if_t<!args_are_all_positional<Args...>()>>
unpacking_collector<policy> collect_arguments(Args &&...args) {
    // Following argument order rules for generalized unpacking according to PEP 448
    static_assert(constexpr_last<is_positional, Args...>()
                          < constexpr_first<is_keyword_or_ds, Args...>()
                      && constexpr_last<is_s_unpacking, Args...>()
                             < constexpr_first<is_ds_unpacking, Args...>(),
                  "Invalid function call: positional args must precede keywords and ** unpacking; "
                  "* unpacking must precede ** unpacking");
    return unpacking_collector<policy>(std::forward<Args>(args)...);
}

template <typename Derived>
template <return_value_policy policy, typename... Args>
object object_api<Derived>::operator()(Args &&...args) const {
#ifndef NDEBUG
    if (!PyGILState_Check()) {
        pybind11_fail("pybind11::object_api<>::operator() PyGILState_Check() failure.");
    }
#endif
    return detail::collect_arguments<policy>(std::forward<Args>(args)...).call(derived().ptr());
}

template <typename Derived>
template <return_value_policy policy, typename... Args>
object object_api<Derived>::call(Args &&...args) const {
    return operator()<policy>(std::forward<Args>(args)...);
}

PYBIND11_NAMESPACE_END(detail)

template <typename T>
handle type::handle_of() {
    static_assert(std::is_base_of<detail::type_caster_generic, detail::make_caster<T>>::value,
                  "py::type::of<T> only supports the case where T is a registered C++ types.");

    return detail::get_type_handle(typeid(T), true);
}

#define PYBIND11_MAKE_OPAQUE(...)                                                                 \
    PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)                                                  \
    namespace detail {                                                                            \
    template <>                                                                                   \
    class type_caster<__VA_ARGS__> : public type_caster_base<__VA_ARGS__> {};                     \
    }                                                                                             \
    PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

/// Lets you pass a type containing a `,` through a macro parameter without needing a separate
/// typedef, e.g.:
/// `PYBIND11_OVERRIDE(PYBIND11_TYPE(ReturnType<A, B>), PYBIND11_TYPE(Parent<C, D>), f, arg)`
#define PYBIND11_TYPE(...) __VA_ARGS__

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
