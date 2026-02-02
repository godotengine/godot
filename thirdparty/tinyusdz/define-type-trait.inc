// TODO: Use (inline) constexpr for C++17
// TODO: Deprecate ndim()
#define DEFINE_TYPE_TRAIT(__dty, __name, __tyid, __nc)                \
  template <>                                                         \
  struct TypeTraits<__dty> {                                          \
    using value_type = __dty;                                         \
    using value_underlying_type = __dty;                              \
    static constexpr uint32_t ndim() { return 0; } /* array dim */    \
    static constexpr size_t size() { return sizeof(__dty); }        \
    static constexpr uint32_t ncomp() {                               \
      return __nc;                                                    \
    } /* the number of components(e.g. extent = 2) */                 \
    static constexpr uint32_t type_id() { return __tyid; }            \
    static constexpr uint32_t underlying_type_id() { return __tyid; } \
    static std::string type_name() { return __name; }                 \
    static std::string underlying_type_name() { return __name; }      \
    static bool is_role_type() { return false; }      \
    static bool is_array() { return type_id() & value::TYPE_ID_1D_ARRAY_BIT; }      \
  }

// `role` type. Requies underlying type.
#define DEFINE_ROLE_TYPE_TRAIT(__dty, __name, __tyid, __uty)                 \
  template <>                                                                \
  struct TypeTraits<__dty> {                                                 \
    using value_type = __dty;                                                \
    using value_underlying_type = TypeTraits<__uty>::value_type;             \
    static constexpr uint32_t ndim() { return 0; } /* array dim */           \
    static constexpr size_t size() { return TypeTraits<__uty>::size(); }   \
    static constexpr uint32_t ncomp() { return TypeTraits<__uty>::ncomp(); } \
    static constexpr uint32_t type_id() { return __tyid; }                   \
    static constexpr uint32_t underlying_type_id() {                         \
      return TypeTraits<__uty>::type_id();                                     \
    }                                                                        \
    static std::string type_name() { return __name; }                        \
    static std::string underlying_type_name() {                              \
      return TypeTraits<__uty>::type_name();                                 \
    }                                                                        \
    static bool is_role_type() { return true; }      \
    static bool is_array() { return type_id() & value::TYPE_ID_1D_ARRAY_BIT; }      \
  }
