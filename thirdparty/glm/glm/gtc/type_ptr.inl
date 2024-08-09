/// @ref gtc_type_ptr

#include <cstring>

namespace glm
{
	/// @addtogroup gtc_type_ptr
	/// @{

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T const* value_ptr(vec<2, T, Q> const& v)
	{
		return &(v.x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T* value_ptr(vec<2, T, Q>& v)
	{
		return &(v.x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T const * value_ptr(vec<3, T, Q> const& v)
	{
		return &(v.x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T* value_ptr(vec<3, T, Q>& v)
	{
		return &(v.x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T const* value_ptr(vec<4, T, Q> const& v)
	{
		return &(v.x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T* value_ptr(vec<4, T, Q>& v)
	{
		return &(v.x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T const* value_ptr(mat<2, 2, T, Q> const& m)
	{
		return &(m[0].x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T* value_ptr(mat<2, 2, T, Q>& m)
	{
		return &(m[0].x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T const* value_ptr(mat<3, 3, T, Q> const& m)
	{
		return &(m[0].x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T* value_ptr(mat<3, 3, T, Q>& m)
	{
		return &(m[0].x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T const* value_ptr(mat<4, 4, T, Q> const& m)
	{
		return &(m[0].x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T* value_ptr(mat<4, 4, T, Q>& m)
	{
		return &(m[0].x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T const* value_ptr(mat<2, 3, T, Q> const& m)
	{
		return &(m[0].x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T* value_ptr(mat<2, 3, T, Q>& m)
	{
		return &(m[0].x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T const* value_ptr(mat<3, 2, T, Q> const& m)
	{
		return &(m[0].x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T* value_ptr(mat<3, 2, T, Q>& m)
	{
		return &(m[0].x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T const* value_ptr(mat<2, 4, T, Q> const& m)
	{
		return &(m[0].x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T* value_ptr(mat<2, 4, T, Q>& m)
	{
		return &(m[0].x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T const* value_ptr(mat<4, 2, T, Q> const& m)
	{
		return &(m[0].x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T* value_ptr(mat<4, 2, T, Q>& m)
	{
		return &(m[0].x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T const* value_ptr(mat<3, 4, T, Q> const& m)
	{
		return &(m[0].x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T* value_ptr(mat<3, 4, T, Q>& m)
	{
		return &(m[0].x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T const* value_ptr(mat<4, 3, T, Q> const& m)
	{
		return &(m[0].x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T * value_ptr(mat<4, 3, T, Q>& m)
	{
		return &(m[0].x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T const * value_ptr(qua<T, Q> const& q)
	{
		return &(q[0]);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T* value_ptr(qua<T, Q>& q)
	{
		return &(q[0]);
	}

	template <typename T, qualifier Q>
	GLM_FUNC_DECL vec<1, T, Q> make_vec1(vec<1, T, Q> const& v)
	{
		return v;
	}

	template <typename T, qualifier Q>
	GLM_FUNC_DECL vec<1, T, Q> make_vec1(vec<2, T, Q> const& v)
	{
		return vec<1, T, Q>(v);
	}

	template <typename T, qualifier Q>
	GLM_FUNC_DECL vec<1, T, Q> make_vec1(vec<3, T, Q> const& v)
	{
		return vec<1, T, Q>(v);
	}

	template <typename T, qualifier Q>
	GLM_FUNC_DECL vec<1, T, Q> make_vec1(vec<4, T, Q> const& v)
	{
		return vec<1, T, Q>(v);
	}

	template <typename T, qualifier Q>
	GLM_FUNC_DECL vec<2, T, Q> make_vec2(vec<1, T, Q> const& v)
	{
		return vec<2, T, Q>(v.x, static_cast<T>(0));
	}

	template <typename T, qualifier Q>
	GLM_FUNC_DECL vec<2, T, Q> make_vec2(vec<2, T, Q> const& v)
	{
		return v;
	}

	template <typename T, qualifier Q>
	GLM_FUNC_DECL vec<2, T, Q> make_vec2(vec<3, T, Q> const& v)
	{
		return vec<2, T, Q>(v);
	}

	template <typename T, qualifier Q>
	GLM_FUNC_DECL vec<2, T, Q> make_vec2(vec<4, T, Q> const& v)
	{
		return vec<2, T, Q>(v);
	}

	template <typename T, qualifier Q>
	GLM_FUNC_DECL vec<3, T, Q> make_vec3(vec<1, T, Q> const& v)
	{
		return vec<3, T, Q>(v.x, static_cast<T>(0), static_cast<T>(0));
	}

	template <typename T, qualifier Q>
	GLM_FUNC_DECL vec<3, T, Q> make_vec3(vec<2, T, Q> const& v)
	{
		return vec<3, T, Q>(v.x, v.y, static_cast<T>(0));
	}

	template <typename T, qualifier Q>
	GLM_FUNC_DECL vec<3, T, Q> make_vec3(vec<3, T, Q> const& v)
	{
		return v;
	}

	template <typename T, qualifier Q>
	GLM_FUNC_DECL vec<3, T, Q> make_vec3(vec<4, T, Q> const& v)
	{
		return vec<3, T, Q>(v);
	}

	template <typename T, qualifier Q>
	GLM_FUNC_DECL vec<4, T, Q> make_vec4(vec<1, T, Q> const& v)
	{
		return vec<4, T, Q>(v.x, static_cast<T>(0), static_cast<T>(0), static_cast<T>(1));
	}

	template <typename T, qualifier Q>
	GLM_FUNC_DECL vec<4, T, Q> make_vec4(vec<2, T, Q> const& v)
	{
		return vec<4, T, Q>(v.x, v.y, static_cast<T>(0), static_cast<T>(1));
	}

	template <typename T, qualifier Q>
	GLM_FUNC_DECL vec<4, T, Q> make_vec4(vec<3, T, Q> const& v)
	{
		return vec<4, T, Q>(v.x, v.y, v.z, static_cast<T>(1));
	}

	template <typename T, qualifier Q>
	GLM_FUNC_DECL vec<4, T, Q> make_vec4(vec<4, T, Q> const& v)
	{
		return v;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER vec<2, T, defaultp> make_vec2(T const *const ptr)
	{
		vec<2, T, defaultp> Result;
		memcpy(value_ptr(Result), ptr, sizeof(vec<2, T, defaultp>));
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER vec<3, T, defaultp> make_vec3(T const *const ptr)
	{
		vec<3, T, defaultp> Result;
		memcpy(value_ptr(Result), ptr, sizeof(vec<3, T, defaultp>));
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER vec<4, T, defaultp> make_vec4(T const *const ptr)
	{
		vec<4, T, defaultp> Result;
		memcpy(value_ptr(Result), ptr, sizeof(vec<4, T, defaultp>));
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<2, 2, T, defaultp> make_mat2x2(T const *const ptr)
	{
		mat<2, 2, T, defaultp> Result;
		memcpy(value_ptr(Result), ptr, sizeof(mat<2, 2, T, defaultp>));
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<2, 3, T, defaultp> make_mat2x3(T const *const ptr)
	{
		mat<2, 3, T, defaultp> Result;
		memcpy(value_ptr(Result), ptr, sizeof(mat<2, 3, T, defaultp>));
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<2, 4, T, defaultp> make_mat2x4(T const *const ptr)
	{
		mat<2, 4, T, defaultp> Result;
		memcpy(value_ptr(Result), ptr, sizeof(mat<2, 4, T, defaultp>));
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<3, 2, T, defaultp> make_mat3x2(T const *const ptr)
	{
		mat<3, 2, T, defaultp> Result;
		memcpy(value_ptr(Result), ptr, sizeof(mat<3, 2, T, defaultp>));
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<3, 3, T, defaultp> make_mat3x3(T const *const ptr)
	{
		mat<3, 3, T, defaultp> Result;
		memcpy(value_ptr(Result), ptr, sizeof(mat<3, 3, T, defaultp>));
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<3, 4, T, defaultp> make_mat3x4(T const *const ptr)
	{
		mat<3, 4, T, defaultp> Result;
		memcpy(value_ptr(Result), ptr, sizeof(mat<3, 4, T, defaultp>));
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 2, T, defaultp> make_mat4x2(T const *const ptr)
	{
		mat<4, 2, T, defaultp> Result;
		memcpy(value_ptr(Result), ptr, sizeof(mat<4, 2, T, defaultp>));
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 3, T, defaultp> make_mat4x3(T const *const ptr)
	{
		mat<4, 3, T, defaultp> Result;
		memcpy(value_ptr(Result), ptr, sizeof(mat<4, 3, T, defaultp>));
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> make_mat4x4(T const *const ptr)
	{
		mat<4, 4, T, defaultp> Result;
		memcpy(value_ptr(Result), ptr, sizeof(mat<4, 4, T, defaultp>));
		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<2, 2, T, defaultp> make_mat2(T const *const ptr)
	{
		return make_mat2x2(ptr);
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<3, 3, T, defaultp> make_mat3(T const *const ptr)
	{
		return make_mat3x3(ptr);
	}

	template<typename T>
	GLM_FUNC_QUALIFIER mat<4, 4, T, defaultp> make_mat4(T const *const ptr)
	{
		return make_mat4x4(ptr);
	}

	template<typename T>
	GLM_FUNC_QUALIFIER qua<T, defaultp> make_quat(T const *const ptr)
	{
		qua<T, defaultp> Result;
		memcpy(value_ptr(Result), ptr, sizeof(qua<T, defaultp>));
		return Result;
	}

	/// @}
}//namespace glm

