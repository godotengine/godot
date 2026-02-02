// Based on
// https://gist.github.com/calebh/fd00632d9c616d4b0c14e7c2865f3085
//
// Modification by Syoyo Fujita.
// - Use tinyusdz::value::TypeTraits for type_id
// - Disable exception
// - Implement set and get, get_if
//

/*
This is free and unencumbered software released into the public domain.
Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.
In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
For more information, please refer to <http://unlicense.org/>
*/
#pragma once

#include <iostream>
#include <string>

#include "value-types.hh" // import TypeTraits

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

#include "nonstd/optional.hpp" // for optional<T>& get()


namespace tinyusdz {

namespace variant_detail {

// Equivalent to std::aligned_storage
template <unsigned int Len, unsigned int Align>
struct aligned_storage {
  struct type {
    alignas(Align) unsigned char data[Len];
  };
};

template <unsigned int arg1, unsigned int... others>
struct static_max;

template <unsigned int arg>
struct static_max<arg> {
  static const unsigned int value = arg;
};

template <unsigned int arg1, unsigned int arg2, unsigned int... others>
struct static_max<arg1, arg2, others...> {
  static const unsigned int value = arg1 >= arg2
                                        ? static_max<arg1, others...>::value
                                        : static_max<arg2, others...>::value;
};

template <class T>
struct remove_reference {
  typedef T type;
};
template <class T>
struct remove_reference<T&> {
  typedef T type;
};
template <class T>
struct remove_reference<T&&> {
  typedef T type;
};

template <typename... Ts>
struct variant_helper_rec;

template <typename F, typename... Ts>
struct variant_helper_rec<F, Ts...> {
  inline static void destroy(uint32_t id, void* data) {
    if (value::TypeTraits<F>::type_id == id) {
      reinterpret_cast<F*>(data)->~F();
    } else {
      variant_helper_rec<Ts...>::destroy(id, data);
    }
  }

  inline static void move(uint32_t id, void* from, void* to) {
    if (value::TypeTraits<F>::type_id == id) {
      // This static_cast and use of remove_reference is equivalent to the use
      // of std::move
      new (to) F(static_cast<typename remove_reference<F>::type&&>(
          *reinterpret_cast<F*>(from)));
    } else {
      variant_helper_rec<Ts...>::move(id, from, to);
    }
  }

  inline static void copy(uint32_t id, const void* from, void* to) {
    if (value::TypeTraits<F>::type_id == id) {
      new (to) F(*reinterpret_cast<const F*>(from));
    } else {
      variant_helper_rec<Ts...>::copy(id, from, to);
    }
  }
};

template <>
struct variant_helper_rec<> {
  inline static void destroy(uint32_t id, void* data) {}
  inline static void move(uint32_t old_t, void* from, void* to) {}
  inline static void copy(uint32_t old_t, const void* from, void* to) {}
};

template <typename... Ts>
struct variant_helper {
  inline static void destroy(uint32_t id, void* data) {
    variant_helper_rec<Ts...>::destroy(id, data);
  }

  inline static void move(uint32_t id, void* from, void* to) {
    variant_helper_rec<Ts...>::move(id, from, to);
  }

  inline static void copy(uint32_t id, const void* old_v, void* new_v) {
    variant_helper_rec<Ts...>::copy(id, old_v, new_v);
  }
};

template <>
struct variant_helper<> {
  inline static void destroy(uint32_t id, void* data) {}
  inline static void move(uint32_t old_t, void* old_v, void* new_v) {}
  inline static void copy(uint32_t old_t, const void* old_v, void* new_v) {}
};

template <typename F>
struct variant_helper_static;

template <typename F>
struct variant_helper_static {
  inline static void move(void* from, void* to) {
    new (to) F(static_cast<typename remove_reference<F>::type&&>(
        *reinterpret_cast<F*>(from)));
  }

  inline static void copy(const void* from, void* to) {
    new (to) F(*reinterpret_cast<const F*>(from));
  }
};

#if 0  // not used
// Given a uint8_t i, selects the ith type from the list of item types
template <uint8_t i, typename... Items>
struct variant_alternative;

template <typename HeadItem, typename... TailItems>
struct variant_alternative<0, HeadItem, TailItems...> {
  using type = HeadItem;
};

template <uint8_t i, typename HeadItem, typename... TailItems>
struct variant_alternative<i, HeadItem, TailItems...> {
  using type = typename variant_alternative<i - 1, TailItems...>::type;
};
#endif

template <uint8_t n, typename... Ts>
struct variant_get_rec;

template <typename...>
struct is_one_of {
  static constexpr bool value = false;
};

template <typename T, typename S, typename... Ts>
struct is_one_of<T, S, Ts...> {
  static constexpr bool value =
      std::is_same<T, S>::value || is_one_of<T, Ts...>::value;
};

} // namespace variant_detail

using namespace variant_detail;

template <typename... Ts>
struct variant {

 private:
  static const unsigned int data_size = static_max<sizeof(Ts)...>::value;
  static const unsigned int data_align = static_max<alignof(Ts)...>::value;

  using data_t = typename aligned_storage<data_size, data_align>::type;

  using helper_t = variant_helper<Ts...>;

  // template <uint8_t i>
  // using alternative = typename variant_alternative<i, Ts...>::type;

  static inline uint32_t invalid_type() {
    return value::TypeTraits<void>::type_id();
  }

  uint32_t variant_id;
  data_t data;

  static void *nulldata() {
    return nullptr;
  }

 public:

  variant() : variant_id(invalid_type()) {}


  variant(const variant<Ts...>& from) : variant_id(from.variant_id) {
    helper_t::copy(from.variant_id, &from.data, &data);
  }

  variant(variant<Ts...>&& from) : variant_id(from.variant_id) {
    helper_t::move(from.variant_id, &from.data, &data);
  }

  variant<Ts...>& operator=(const variant<Ts...>& rhs) {
    helper_t::destroy(variant_id, &data);
    variant_id = rhs.variant_id;
    helper_t::copy(rhs.variant_id, &rhs.data, &data);
    return *this;
  }

  variant<Ts...>& operator=(variant<Ts...>&& rhs) {
    helper_t::destroy(variant_id, &data);
    variant_id = rhs.variant_id;
    helper_t::move(rhs.variant_id, &rhs.data, &data);
    return *this;
  }

  template <typename T>
  bool is() const {
    return variant_id == value::TypeTraits<T>::type_id;
  }

  uint32_t id() const { return variant_id; }

  // template<typename T, typename... Args>
  template <typename T, typename... Args,
            typename =
                typename std::enable_if<is_one_of<T, Ts...>::value, void>::type>
  void set(Args&&... args) {
    helper_t::destroy(variant_id, &data);
    new (&data) T(std::forward<Args>(args)...);
    variant_id = value::TypeTraits<T>::type_id;
    // variant_helper_static<alternative<i>>::copy(&value, &data);
  }

  template<typename T>
  variant(const T &v) {
    set<T>(v);
  }

#if 1
  // allow seg fault.
  template <typename T, typename... Args,
            typename =
                typename std::enable_if<is_one_of<T, Ts...>::value, void>::type>
  T& cast() {
    // It is a dynamic_cast-like behaviour
    if (variant_id == value::TypeTraits<T>::type_id) {
      return *reinterpret_cast<T*>(&data);
    }

    // Will raise null-pointer dereference error.
    return *reinterpret_cast<T*>(nulldata());
  }
#endif

  template <typename T, typename... Args,
            typename =
                typename std::enable_if<is_one_of<T, Ts...>::value, void>::type>
  const nonstd::optional<T> get() {
    // It is a dynamic_cast-like behaviour
    if (variant_id == value::TypeTraits<T>::type_id) {
      return *reinterpret_cast<T*>(&data);
    }

    return nonstd::nullopt;
  }

  template <typename T, typename... Args,
            typename =
                typename std::enable_if<is_one_of<T, Ts...>::value, void>::type>
  const nonstd::optional<T> get() const {
    // It is a dynamic_cast-like behaviour
    if (variant_id == value::TypeTraits<T>::type_id) {
      return *reinterpret_cast<const T*>(&data);
    }

    return nonstd::nullopt;
  }

  template <typename T, typename... Args,
            typename =
                typename std::enable_if<is_one_of<T, Ts...>::value, void>::type>
  const T* get_if() const {
    // It is a dynamic_cast-like behaviour
    if (variant_id == value::TypeTraits<T>::type_id) {
      return reinterpret_cast<const T*>(&data);
    }

    return nullptr;
  }

  ~variant() { helper_t::destroy(variant_id, &data); }
};

struct monostate {};

namespace value {

#define DEFINE_TYPE_TRAIT(__dty, __name, __tyid, __nc)           \
  template <>                                                    \
  struct TypeTraits<__dty> {                                      \
    using value_type = __dty;                                    \
    using value_underlying_type = __dty;                         \
    static constexpr uint32_t ndim = 0; /* array dim */          \
    static constexpr uint32_t ncomp =                            \
        __nc; /* the number of components(e.g. float3 => 3) */   \
    static constexpr uint32_t type_id = __tyid;                  \
    static constexpr uint32_t underlying_type_id = __tyid;       \
    static std::string type_name() { return __name; }            \
    static std::string underlying_type_name() { return __name; } \
  }

DEFINE_TYPE_TRAIT(monostate, "monostate", TYPE_ID_MONOSTATE, 1);

#undef DEFINE_TYPE_TRAIT

} // namespace value

#ifdef __clang__
#pragma clang diagnostic pop
#endif

}  // namespace tinyusdz
