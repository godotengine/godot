/*
 * Copyright 2015-2019 Alexey Chernov <4ernov@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#ifndef FLOAXIE_STATIC_POW_H
#define FLOAXIE_STATIC_POW_H

#include <array>
#include <utility>

namespace floaxie
{
	/** \brief Helper class template to calculate integer positive power in
	 * compile time.
	 *
	 * Unspecialized template contains intermediate steps implementation.
	 *
	 * \tparam base base of the power.
	 * \tparam pow exponent to power base.
	 */
	template<unsigned int base, unsigned int pow> struct static_pow_helper
	{
		/** \brief Result of power calculation (in compile time).
		 *
		 * Is calculated recursively in intermediate steps.
		 */
		static const unsigned int value = base * static_pow_helper<base, pow - 1>::value;
	};

	/** \brief Terminal specialization of `static_pow_helper` template. */
	template<unsigned int base> struct static_pow_helper<base, 0>
	{
		/** \brief Any value in zero power is `1`. */
		static const unsigned int value = 1;
	};

	/** \brief Handy wrapper function to calculate positive power in compile
	 * time.
	 *
	 * \tparam base base of the power.
	 * \tparam pow exponent to power base.
	 *
	 * \return Result of power calculation (in compile time).
	 */
	template<unsigned int base, unsigned int pow> constexpr unsigned long static_pow()
	{
		static_assert(base > 0, "Base should be positive");
		return static_pow_helper<base, pow>::value;
	}

	/** \brief Helper structure template to append value to
	 * `std::integer_sequence`.
	 *
	 * \tparam T type of elements of `std::integer_sequence`.
	 * \tparam Add element to add to the specified sequence.
	 * \tparam Seq original `std::integer_sequence` expressed by parameter pack
	 * of its elements in template parameters.
	 */
	template<typename T, T Add, typename Seq> struct concat_sequence;

	/** \brief Implements the concatenation itself.
	 *
	 * Steals parameters (read: contents) of passed `std::integer_sequence`-like
	 * type and creates another `std::integer_sequence`-like type with the
	 * specified value appended at the end.
	 */
	template<typename T, T Add, T... Seq> struct concat_sequence<T, Add, std::integer_sequence<T, Seq...>>
	{
		/** \brief New `std::integer_sequence`-like type with the specified
		 * element added to the end.
		 */
		using type = std::integer_sequence<T, Seq..., Add>;
	};

	/** \brief Helper structure template to convert `std::integer_sequence`
	 * to `std::array`.
	 *
	 * \tparam Seq sequence to convert.
	 */
	template<typename Seq> struct make_integer_array;

	/** \brief Main specialization of `make_integer_array` with the specified
	 * input `std::integer_sequence`.
	 *
	 * \tparam T type of elements of `std::integer_sequence`.
	 * \tparam Ints elements of the `std::integer_sequence` to convert.
	 */
	template<typename T, T... Ints> struct make_integer_array<std::integer_sequence<T, Ints...>>
	{
		/** \brief Type of the resulting `std::array` (specialized with
		 * element type and length). */
		using type = std::array<T, sizeof...(Ints)>;

		/** \brief Instance of the resulting `std::array` filled in with
		 * elements from the specified `std::integer_sequence` in compile time.
		 */
		static constexpr type value = type{{Ints...}};
	};

	/** \brief Creates `std::integer_sequence` with sequence of powers of
	 * \p **base** up to the exponent value, defined by \p **current_pow**.
	 *
	 * For example, if \p base is `10` and \p current_pow is `3`, the result
	 * will contain values `1000`, `100`, `10`, `1`.
	 *
	 * \tparam T type of elements.
	 * \tparam base base of powers to calculate.
	 * \tparam current_pow the maximum exponent value to calculate power for.
	 */
	template<typename T, T base, std::size_t current_pow> struct pow_sequencer
	{
		/** \brief Value of power on the current step (in recursion). */
		static const T value = pow_sequencer<T, base, current_pow - 1>::value * base;

		/** \brief `std::integer_sequence`-like type containing all the
		 * calculated powers at the moment.
		 */
		typedef typename concat_sequence<T, value, typename pow_sequencer<T, base, current_pow - 1>::sequence_type>::type sequence_type;
	};

	/** \brief Terminal step specialization for `pow_sequencer` template. */
	template<typename T, T base> struct pow_sequencer<T, base, 0>
	{
		/** \brief Zero power of base. */
		static constexpr T value = 1;
		/** \brief `std::integer_sequence` with zero power only yet. */
		typedef std::integer_sequence<T, 1> sequence_type;
	};

	/** \brief Handy wrapper function to calculate a sequence of powers.
	 *
	 * \tparam T type of elements.
	 * \tparam base base for powers to calculate.
	 * \tparam max_pow maximum exponent, up to which the calculation will be
	 * performed.
	 *
	 * \return `std::array` filled with powers of the specified arguments in
	 * reverse order. I.e. for \p **T** of `unsigned int`, \p **base** of 10
	 * and \p **max_pow** 3 this would be:
	 * `std::array<unsigned int>({{1000, 100, 10, 1}})`.
	 */
	template<typename T, T base, std::size_t max_pow> constexpr T seq_pow(std::size_t pow)
	{
		typedef make_integer_array<typename pow_sequencer<T, base, max_pow>::sequence_type> maker;
		constexpr typename maker::type arr(maker::value);

		return arr[pow];
	}
}

#endif // FLOAXIE_STATIC_POW_H
