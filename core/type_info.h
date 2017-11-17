#ifndef GET_TYPE_INFO_H
#define GET_TYPE_INFO_H

#ifdef DEBUG_METHODS_ENABLED

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
	static const Variant::Type VARIANT_TYPE = Variant::NIL;
	static inline PropertyInfo get_class_info() {
		ERR_PRINT("GetTypeInfo fallback. Bug!");
		return PropertyInfo(); // Not "Nil", this is an error
	}
};

#define MAKE_TYPE_INFO(m_type, m_var_type)                    \
	template <>                                               \
	struct GetTypeInfo<m_type> {                              \
		static const Variant::Type VARIANT_TYPE = m_var_type; \
		static inline PropertyInfo get_class_info() {         \
			return PropertyInfo(VARIANT_TYPE, String());      \
		}                                                     \
	};                                                        \
	template <>                                               \
	struct GetTypeInfo<const m_type &> {                      \
		static const Variant::Type VARIANT_TYPE = m_var_type; \
		static inline PropertyInfo get_class_info() {         \
			return PropertyInfo(VARIANT_TYPE, String());      \
		}                                                     \
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
MAKE_TYPE_INFO(wchar_t, Variant::INT)
MAKE_TYPE_INFO(float, Variant::REAL)
MAKE_TYPE_INFO(double, Variant::REAL)

MAKE_TYPE_INFO(String, Variant::STRING)
MAKE_TYPE_INFO(Vector2, Variant::VECTOR2)
MAKE_TYPE_INFO(Rect2, Variant::RECT2)
MAKE_TYPE_INFO(Vector3, Variant::VECTOR3)
MAKE_TYPE_INFO(Transform2D, Variant::TRANSFORM2D)
MAKE_TYPE_INFO(Plane, Variant::PLANE)
MAKE_TYPE_INFO(Quat, Variant::QUAT)
MAKE_TYPE_INFO(AABB, Variant::AABB)
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

//for RefPtr
template <>
struct GetTypeInfo<RefPtr> {
	static const Variant::Type VARIANT_TYPE = Variant::OBJECT;
	static inline PropertyInfo get_class_info() {
		return PropertyInfo(Variant::OBJECT, String(), PROPERTY_HINT_RESOURCE_TYPE, "Reference");
	}
};
template <>
struct GetTypeInfo<const RefPtr &> {
	static const Variant::Type VARIANT_TYPE = Variant::OBJECT;
	static inline PropertyInfo get_class_info() {
		return PropertyInfo(Variant::OBJECT, String(), PROPERTY_HINT_RESOURCE_TYPE, "Reference");
	}
};

//for variant
template <>
struct GetTypeInfo<Variant> {
	static const Variant::Type VARIANT_TYPE = Variant::NIL;
	static inline PropertyInfo get_class_info() {
		return PropertyInfo(Variant::NIL, String(), PROPERTY_HINT_NONE, String(), PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_NIL_IS_VARIANT);
	}
};

template <>
struct GetTypeInfo<const Variant &> {
	static const Variant::Type VARIANT_TYPE = Variant::NIL;
	static inline PropertyInfo get_class_info() {
		return PropertyInfo(Variant::NIL, String(), PROPERTY_HINT_NONE, String(), PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_NIL_IS_VARIANT);
	}
};

#define MAKE_TEMPLATE_TYPE_INFO(m_template, m_type, m_var_type) \
	template <>                                                 \
	struct GetTypeInfo<m_template<m_type> > {                   \
		static const Variant::Type VARIANT_TYPE = m_var_type;   \
		static inline PropertyInfo get_class_info() {           \
			return PropertyInfo(VARIANT_TYPE, String());        \
		}                                                       \
	};                                                          \
	template <>                                                 \
	struct GetTypeInfo<const m_template<m_type> &> {            \
		static const Variant::Type VARIANT_TYPE = m_var_type;   \
		static inline PropertyInfo get_class_info() {           \
			return PropertyInfo(VARIANT_TYPE, String());        \
		}                                                       \
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

template <typename T>
struct GetTypeInfo<T *, typename EnableIf<TypeInherits<Object, T>::value>::type> {
	static const Variant::Type VARIANT_TYPE = Variant::OBJECT;
	static inline PropertyInfo get_class_info() {
		return PropertyInfo(StringName(T::get_class_static()));
	}
};

template <typename T>
struct GetTypeInfo<const T *, typename EnableIf<TypeInherits<Object, T>::value>::type> {
	static const Variant::Type VARIANT_TYPE = Variant::OBJECT;
	static inline PropertyInfo get_class_info() {
		return PropertyInfo(StringName(T::get_class_static()));
	}
};

#define TEMPL_MAKE_ENUM_TYPE_INFO(m_enum, m_impl)                                                                                                                                 \
	template <>                                                                                                                                                                   \
	struct GetTypeInfo<m_impl> {                                                                                                                                                  \
		static const Variant::Type VARIANT_TYPE = Variant::INT;                                                                                                                   \
		static inline PropertyInfo get_class_info() {                                                                                                                             \
			return PropertyInfo(Variant::INT, String(), PROPERTY_HINT_NONE, String(), PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_CLASS_IS_ENUM, String(#m_enum).replace("::", ".")); \
		}                                                                                                                                                                         \
	};

#define MAKE_ENUM_TYPE_INFO(m_enum)                 \
	TEMPL_MAKE_ENUM_TYPE_INFO(m_enum, m_enum)       \
	TEMPL_MAKE_ENUM_TYPE_INFO(m_enum, m_enum const) \
	TEMPL_MAKE_ENUM_TYPE_INFO(m_enum, m_enum &)     \
	TEMPL_MAKE_ENUM_TYPE_INFO(m_enum, const m_enum &)

template <typename T>
inline StringName __constant_get_enum_name(T param, const String &p_constant) {
	if (GetTypeInfo<T>::VARIANT_TYPE == Variant::NIL)
		ERR_PRINTS("Missing VARIANT_ENUM_CAST for constant's enum: " + p_constant);
	return GetTypeInfo<T>::get_class_info().class_name;
}

#define CLASS_INFO(m_type)                                    \
	(GetTypeInfo<m_type *>::VARIANT_TYPE != Variant::NIL ?    \
					GetTypeInfo<m_type *>::get_class_info() : \
					GetTypeInfo<m_type>::get_class_info())

#else

#define MAKE_ENUM_TYPE_INFO(m_enum)
#define CLASS_INFO(m_type)

#endif // DEBUG_METHODS_ENABLED

#endif // GET_TYPE_INFO_H
