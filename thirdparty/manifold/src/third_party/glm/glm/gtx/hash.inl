/// @ref gtx_hash
///
/// @see core (dependence)
///
/// @defgroup gtx_hash GLM_GTX_hash
/// @ingroup gtx
///
/// @brief Add std::hash support for glm types
///
/// <glm/gtx/hash.inl> need to be included to use the features of this extension.

namespace glm {
namespace detail
{
	GLM_INLINE void hash_combine(size_t &seed, size_t hash)
	{
		hash += 0x9e3779b9 + (seed << 6) + (seed >> 2);
		seed ^= hash;
	}
}}

namespace std
{
	template<typename T, glm::qualifier Q>
	GLM_FUNC_QUALIFIER size_t hash<glm::vec<1, T, Q>>::operator()(glm::vec<1, T, Q> const& v) const
	{
		hash<T> hasher;
		return hasher(v.x);
	}

	template<typename T, glm::qualifier Q>
	GLM_FUNC_QUALIFIER size_t hash<glm::vec<2, T, Q>>::operator()(glm::vec<2, T, Q> const& v) const
	{
		size_t seed = 0;
		hash<T> hasher;
		glm::detail::hash_combine(seed, hasher(v.x));
		glm::detail::hash_combine(seed, hasher(v.y));
		return seed;
	}

	template<typename T, glm::qualifier Q>
	GLM_FUNC_QUALIFIER size_t hash<glm::vec<3, T, Q>>::operator()(glm::vec<3, T, Q> const& v) const
	{
		size_t seed = 0;
		hash<T> hasher;
		glm::detail::hash_combine(seed, hasher(v.x));
		glm::detail::hash_combine(seed, hasher(v.y));
		glm::detail::hash_combine(seed, hasher(v.z));
		return seed;
	}

	template<typename T, glm::qualifier Q>
	GLM_FUNC_QUALIFIER size_t hash<glm::vec<4, T, Q>>::operator()(glm::vec<4, T, Q> const& v) const
	{
		size_t seed = 0;
		hash<T> hasher;
		glm::detail::hash_combine(seed, hasher(v.x));
		glm::detail::hash_combine(seed, hasher(v.y));
		glm::detail::hash_combine(seed, hasher(v.z));
		glm::detail::hash_combine(seed, hasher(v.w));
		return seed;
	}

	template<typename T, glm::qualifier Q>
	GLM_FUNC_QUALIFIER size_t hash<glm::qua<T, Q>>::operator()(glm::qua<T,Q> const& q) const
	{
		size_t seed = 0;
		hash<T> hasher;
		glm::detail::hash_combine(seed, hasher(q.x));
		glm::detail::hash_combine(seed, hasher(q.y));
		glm::detail::hash_combine(seed, hasher(q.z));
		glm::detail::hash_combine(seed, hasher(q.w));
		return seed;
	}

	template<typename T, glm::qualifier Q>
	GLM_FUNC_QUALIFIER size_t hash<glm::tdualquat<T, Q>>::operator()(glm::tdualquat<T, Q> const& q) const
	{
		size_t seed = 0;
		hash<glm::qua<T, Q>> hasher;
		glm::detail::hash_combine(seed, hasher(q.real));
		glm::detail::hash_combine(seed, hasher(q.dual));
		return seed;
	}

	template<typename T, glm::qualifier Q>
	GLM_FUNC_QUALIFIER size_t hash<glm::mat<2, 2, T, Q>>::operator()(glm::mat<2, 2, T, Q> const& m) const
	{
		size_t seed = 0;
		hash<glm::vec<2, T, Q>> hasher;
		glm::detail::hash_combine(seed, hasher(m[0]));
		glm::detail::hash_combine(seed, hasher(m[1]));
		return seed;
	}

	template<typename T, glm::qualifier Q>
	GLM_FUNC_QUALIFIER size_t hash<glm::mat<2, 3, T, Q>>::operator()(glm::mat<2, 3, T, Q> const& m) const
	{
		size_t seed = 0;
		hash<glm::vec<3, T, Q>> hasher;
		glm::detail::hash_combine(seed, hasher(m[0]));
		glm::detail::hash_combine(seed, hasher(m[1]));
		return seed;
	}

	template<typename T, glm::qualifier Q>
	GLM_FUNC_QUALIFIER size_t hash<glm::mat<2, 4, T, Q>>::operator()(glm::mat<2, 4, T, Q> const& m) const
	{
		size_t seed = 0;
		hash<glm::vec<4, T, Q>> hasher;
		glm::detail::hash_combine(seed, hasher(m[0]));
		glm::detail::hash_combine(seed, hasher(m[1]));
		return seed;
	}

	template<typename T, glm::qualifier Q>
	GLM_FUNC_QUALIFIER size_t hash<glm::mat<3, 2, T, Q>>::operator()(glm::mat<3, 2, T, Q> const& m) const
	{
		size_t seed = 0;
		hash<glm::vec<2, T, Q>> hasher;
		glm::detail::hash_combine(seed, hasher(m[0]));
		glm::detail::hash_combine(seed, hasher(m[1]));
		glm::detail::hash_combine(seed, hasher(m[2]));
		return seed;
	}

	template<typename T, glm::qualifier Q>
	GLM_FUNC_QUALIFIER size_t hash<glm::mat<3, 3, T, Q>>::operator()(glm::mat<3, 3, T, Q> const& m) const
	{
		size_t seed = 0;
		hash<glm::vec<3, T, Q>> hasher;
		glm::detail::hash_combine(seed, hasher(m[0]));
		glm::detail::hash_combine(seed, hasher(m[1]));
		glm::detail::hash_combine(seed, hasher(m[2]));
		return seed;
	}

	template<typename T, glm::qualifier Q>
	GLM_FUNC_QUALIFIER size_t hash<glm::mat<3, 4, T, Q>>::operator()(glm::mat<3, 4, T, Q> const& m) const
	{
		size_t seed = 0;
		hash<glm::vec<4, T, Q>> hasher;
		glm::detail::hash_combine(seed, hasher(m[0]));
		glm::detail::hash_combine(seed, hasher(m[1]));
		glm::detail::hash_combine(seed, hasher(m[2]));
		return seed;
	}

	template<typename T, glm::qualifier Q>
	GLM_FUNC_QUALIFIER size_t hash<glm::mat<4, 2, T,Q>>::operator()(glm::mat<4, 2, T,Q> const& m) const
	{
		size_t seed = 0;
		hash<glm::vec<2, T, Q>> hasher;
		glm::detail::hash_combine(seed, hasher(m[0]));
		glm::detail::hash_combine(seed, hasher(m[1]));
		glm::detail::hash_combine(seed, hasher(m[2]));
		glm::detail::hash_combine(seed, hasher(m[3]));
		return seed;
	}

	template<typename T, glm::qualifier Q>
	GLM_FUNC_QUALIFIER size_t hash<glm::mat<4, 3, T,Q>>::operator()(glm::mat<4, 3, T,Q> const& m) const
	{
		size_t seed = 0;
		hash<glm::vec<3, T, Q>> hasher;
		glm::detail::hash_combine(seed, hasher(m[0]));
		glm::detail::hash_combine(seed, hasher(m[1]));
		glm::detail::hash_combine(seed, hasher(m[2]));
		glm::detail::hash_combine(seed, hasher(m[3]));
		return seed;
	}

	template<typename T, glm::qualifier Q>
	GLM_FUNC_QUALIFIER size_t hash<glm::mat<4, 4, T,Q>>::operator()(glm::mat<4, 4, T, Q> const& m) const
	{
		size_t seed = 0;
		hash<glm::vec<4, T, Q>> hasher;
		glm::detail::hash_combine(seed, hasher(m[0]));
		glm::detail::hash_combine(seed, hasher(m[1]));
		glm::detail::hash_combine(seed, hasher(m[2]));
		glm::detail::hash_combine(seed, hasher(m[3]));
		return seed;
	}
}
