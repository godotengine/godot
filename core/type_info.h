#ifndef GET_TYPE_INFO_H
#define GET_TYPE_INFO_H

template <bool C, typename T = void>
struct EnableIf {

	typedef T type;
};

template <typename T>
struct EnableIf<false, T> {
};

template <typename, typename>
struct TypesAreSame {

	static bool const value = false;
};

template <typename A>
struct TypesAreSame<A, A> {

	static bool const value = true;
};

template <typename B, typename D>
struct TypeInherits {

	static D *get_d();

	static char (&test(B *))[1];
	static char (&test(...))[2];

	static bool const value = sizeof(test(get_d())) == sizeof(char) &&
							  !TypesAreSame<B volatile const, void volatile const>::value;
};

template <class T, typename = void>
struct GetTypeInfo {
	enum { VARIANT_TYPE = Variant::NIL };

	static inline StringName get_class_name() {
		ERR_PRINT("Fallback type info. Bug!");
		return ""; // Not "Nil", this is an error
	}
};

#define MAKE_TYPE_INFO(m_type, m_var_type)                              \
	template <>                                                         \
	struct GetTypeInfo<m_type> {                                        \
		enum { VARIANT_TYPE = m_var_type };                             \
		static inline StringName get_class_name() {                     \
			return Variant::get_type_name((Variant::Type)VARIANT_TYPE); \
		}                                                               \
	};                                                                  \
	template <>                                                         \
	struct GetTypeInfo<const m_type &> {                                \
		enum { VARIANT_TYPE = m_var_type };                             \
		static inline StringName get_class_name() {                     \
			return Variant::get_type_name((Variant::Type)VARIANT_TYPE); \
		}                                                               \
	};

MAKE_TYPE_INFO(bool, Variant::BOOL)
MAKE_TYPE_INFO(uint8_t, Variant::INT)
MAKE_TYPE_INFO(int8_t, Variant::INT)
MAKE_TYPE_INFO(uint16_t, Variant::INT)
MAKE_TYPE_INFO(int16_t, Variant::INT)
MAKE_TYPE_INFO(uint32_t, Variant::INT)
MAKE_TYPE_INFO(int32_t, Variant::INT)
MAKE_TYPE_INFO(int64_t, Variant::INT)
MAKE_TYPE_INFO(uint64_t, Variant::INT)
MAKE_TYPE_INFO(float, Variant::REAL)
MAKE_TYPE_INFO(double, Variant::REAL)

MAKE_TYPE_INFO(String, Variant::STRING)
MAKE_TYPE_INFO(Vector2, Variant::VECTOR2)
MAKE_TYPE_INFO(Rect2, Variant::RECT2)
MAKE_TYPE_INFO(Vector3, Variant::VECTOR3)
MAKE_TYPE_INFO(Transform2D, Variant::TRANSFORM2D)
MAKE_TYPE_INFO(Plane, Variant::PLANE)
MAKE_TYPE_INFO(Quat, Variant::QUAT)
MAKE_TYPE_INFO(Rect3, Variant::RECT3)
MAKE_TYPE_INFO(Basis, Variant::BASIS)
MAKE_TYPE_INFO(Transform, Variant::TRANSFORM)
MAKE_TYPE_INFO(Color, Variant::COLOR)
MAKE_TYPE_INFO(NodePath, Variant::NODE_PATH)
MAKE_TYPE_INFO(RID, Variant::_RID)
MAKE_TYPE_INFO(Dictionary, Variant::DICTIONARY)
MAKE_TYPE_INFO(Array, Variant::ARRAY)
MAKE_TYPE_INFO(PoolByteArray, Variant::POOL_BYTE_ARRAY)
MAKE_TYPE_INFO(PoolIntArray, Variant::POOL_INT_ARRAY)
MAKE_TYPE_INFO(PoolRealArray, Variant::POOL_REAL_ARRAY)
MAKE_TYPE_INFO(PoolStringArray, Variant::POOL_STRING_ARRAY)
MAKE_TYPE_INFO(PoolVector2Array, Variant::POOL_VECTOR2_ARRAY)
MAKE_TYPE_INFO(PoolVector3Array, Variant::POOL_VECTOR3_ARRAY)
MAKE_TYPE_INFO(PoolColorArray, Variant::POOL_COLOR_ARRAY)

MAKE_TYPE_INFO(StringName, Variant::STRING)
MAKE_TYPE_INFO(IP_Address, Variant::STRING)

class BSP_Tree;
MAKE_TYPE_INFO(BSP_Tree, Variant::DICTIONARY)

#define MAKE_TYPE_INFO_WITH_NAME(m_type, m_var_type, m_class_name) \
	template <>                                                    \
	struct GetTypeInfo<m_type> {                                   \
		enum { VARIANT_TYPE = m_var_type };                        \
		static inline StringName get_class_name() {                \
			return m_class_name;                                   \
		}                                                          \
	};                                                             \
	template <>                                                    \
	struct GetTypeInfo<const m_type &> {                           \
		enum { VARIANT_TYPE = m_var_type };                        \
		static inline StringName get_class_name() {                \
			return m_class_name;                                   \
		}                                                          \
	};

MAKE_TYPE_INFO_WITH_NAME(RefPtr, Variant::OBJECT, "Reference")
MAKE_TYPE_INFO_WITH_NAME(Variant, Variant::NIL, "Variant")

#define MAKE_TEMPLATE_TYPE_INFO(m_template, m_type, m_var_type)         \
	template <>                                                         \
	struct GetTypeInfo<m_template<m_type> > {                           \
		enum { VARIANT_TYPE = m_var_type };                             \
		static inline StringName get_class_name() {                     \
			return Variant::get_type_name((Variant::Type)VARIANT_TYPE); \
		}                                                               \
	};                                                                  \
	template <>                                                         \
	struct GetTypeInfo<const m_template<m_type> &> {                    \
		enum { VARIANT_TYPE = m_var_type };                             \
		static inline StringName get_class_name() {                     \
			return Variant::get_type_name((Variant::Type)VARIANT_TYPE); \
		}                                                               \
	};

MAKE_TEMPLATE_TYPE_INFO(Vector, uint8_t, Variant::POOL_BYTE_ARRAY)
MAKE_TEMPLATE_TYPE_INFO(Vector, int, Variant::POOL_INT_ARRAY)
MAKE_TEMPLATE_TYPE_INFO(Vector, float, Variant::POOL_REAL_ARRAY)
MAKE_TEMPLATE_TYPE_INFO(Vector, String, Variant::POOL_STRING_ARRAY)
MAKE_TEMPLATE_TYPE_INFO(Vector, Vector2, Variant::POOL_VECTOR2_ARRAY)
MAKE_TEMPLATE_TYPE_INFO(Vector, Vector3, Variant::POOL_VECTOR3_ARRAY)
MAKE_TEMPLATE_TYPE_INFO(Vector, Color, Variant::POOL_COLOR_ARRAY)

MAKE_TEMPLATE_TYPE_INFO(Vector, Variant, Variant::ARRAY)
MAKE_TEMPLATE_TYPE_INFO(Vector, RID, Variant::ARRAY)
MAKE_TEMPLATE_TYPE_INFO(Vector, Plane, Variant::ARRAY)

MAKE_TEMPLATE_TYPE_INFO(PoolVector, Plane, Variant::ARRAY)
MAKE_TEMPLATE_TYPE_INFO(PoolVector, Face3, Variant::POOL_VECTOR3_ARRAY)

#define MAKE_ENUM_TYPE_INFO(m_enum)                                     \
	template <>                                                         \
	struct GetTypeInfo<m_enum> {                                        \
		enum { VARIANT_TYPE = Variant::INT };                           \
		static inline StringName get_class_name() {                     \
			return Variant::get_type_name((Variant::Type)VARIANT_TYPE); \
		}                                                               \
	};                                                                  \
	template <>                                                         \
	struct GetTypeInfo<m_enum const> {                                  \
		enum { VARIANT_TYPE = Variant::INT };                           \
		static inline StringName get_class_name() {                     \
			return Variant::get_type_name((Variant::Type)VARIANT_TYPE); \
		}                                                               \
	};                                                                  \
	template <>                                                         \
	struct GetTypeInfo<m_enum &> {                                      \
		enum { VARIANT_TYPE = Variant::INT };                           \
		static inline StringName get_class_name() {                     \
			return Variant::get_type_name((Variant::Type)VARIANT_TYPE); \
		}                                                               \
	};                                                                  \
	template <>                                                         \
	struct GetTypeInfo<const m_enum &> {                                \
		enum { VARIANT_TYPE = Variant::INT };                           \
		static inline StringName get_class_name() {                     \
			return Variant::get_type_name((Variant::Type)VARIANT_TYPE); \
		}                                                               \
	};

template <typename T>
struct GetTypeInfo<T *, typename EnableIf<TypeInherits<Object, T>::value>::type> {
	enum { VARIANT_TYPE = Variant::OBJECT };

	static inline StringName get_class_name() {
		return T::get_class_static();
	}
};

template <typename T>
struct GetTypeInfo<const T *, typename EnableIf<TypeInherits<Object, T>::value>::type> {
	enum { VARIANT_TYPE = Variant::OBJECT };

	static inline StringName get_class_name() {
		return T::get_class_static();
	}
};

#endif // GET_TYPE_INFO_H
