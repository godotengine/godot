/// @ref gtx_type_trait

namespace glm
{
	template<typename T>
	bool const type<T>::is_vec;
	template<typename T>
	bool const type<T>::is_mat;
	template<typename T>
	bool const type<T>::is_quat;
	template<typename T>
	length_t const type<T>::components;
	template<typename T>
	length_t const type<T>::cols;
	template<typename T>
	length_t const type<T>::rows;

	// vec
	template<length_t L, typename T, qualifier Q>
	bool const type<vec<L, T, Q> >::is_vec;
	template<length_t L, typename T, qualifier Q>
	bool const type<vec<L, T, Q> >::is_mat;
	template<length_t L, typename T, qualifier Q>
	bool const type<vec<L, T, Q> >::is_quat;
	template<length_t L, typename T, qualifier Q>
	length_t const type<vec<L, T, Q> >::components;

	// mat
	template<length_t C, length_t R, typename T, qualifier Q>
	bool const type<mat<C, R, T, Q> >::is_vec;
	template<length_t C, length_t R, typename T, qualifier Q>
	bool const type<mat<C, R, T, Q> >::is_mat;
	template<length_t C, length_t R, typename T, qualifier Q>
	bool const type<mat<C, R, T, Q> >::is_quat;
	template<length_t C, length_t R, typename T, qualifier Q>
	length_t const type<mat<C, R, T, Q> >::components;
	template<length_t C, length_t R, typename T, qualifier Q>
	length_t const type<mat<C, R, T, Q> >::cols;
	template<length_t C, length_t R, typename T, qualifier Q>
	length_t const type<mat<C, R, T, Q> >::rows;

	// tquat
	template<typename T, qualifier Q>
	bool const type<qua<T, Q> >::is_vec;
	template<typename T, qualifier Q>
	bool const type<qua<T, Q> >::is_mat;
	template<typename T, qualifier Q>
	bool const type<qua<T, Q> >::is_quat;
	template<typename T, qualifier Q>
	length_t const type<qua<T, Q> >::components;

	// tdualquat
	template<typename T, qualifier Q>
	bool const type<tdualquat<T, Q> >::is_vec;
	template<typename T, qualifier Q>
	bool const type<tdualquat<T, Q> >::is_mat;
	template<typename T, qualifier Q>
	bool const type<tdualquat<T, Q> >::is_quat;
	template<typename T, qualifier Q>
	length_t const type<tdualquat<T, Q> >::components;
}//namespace glm
