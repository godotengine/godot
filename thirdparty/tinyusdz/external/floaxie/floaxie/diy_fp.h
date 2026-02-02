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
 * diy_fp class and helper functions use code and influenced by
 * Florian Loitsch's original Grisu algorithms implementation
 * (http://florian.loitsch.com/publications/bench.tar.gz)
 * and "Printing Floating-Point Numbers Quickly and Accurately with
 * Integers" paper
 * (http://florian.loitsch.com/publications/dtoa-pldi2010.pdf)
 */

#ifndef FLOAXIE_DIY_FP_H
#define FLOAXIE_DIY_FP_H

#include <algorithm>
#include <limits>
#include <cstdint>
#include <cassert>
#include <ostream>
#include <utility>

#include <external/floaxie/floaxie/bit_ops.h>
#include <external/floaxie/floaxie/print.h>
#include <external/floaxie/floaxie/type_punning_cast.h>
#include <external/floaxie/floaxie/huge_val.h>
#include <external/floaxie/floaxie/conversion_status.h>

namespace floaxie
{

	/** \brief Template structure to define `diy_fp` inner types for
	 * selected floating point types.
	 */
	template<typename FloatType> struct diy_fp_traits;

	/** \brief `diy_fp_traits` specialization associated with single precision
	 * floating point type (`float`).
	 *
	 * **Mantissa** is stored using the fastest natively supported standard
	 * integer type — almost always it's 32-bit unsigned integer value.
	 *
	 * **Exponent** is stored in `int` value. Shorter types aren't used to
	 * avoid any performance impacts of not using default integer types
	 * (though it can be effective in terms of cache usage).
	 */
	template<> struct diy_fp_traits<float>
	{
		/** \brief Integer type to store mantissa value. */
		typedef std::uint32_t mantissa_type;
		/** \brief Integer type to store exponent value. */
		typedef int exponent_type;
	};

	/** \brief `diy_fp_traits` specialization associated with double precision
	 * floating point type (`double`).
	 *
	 * **Mantissa** is stored using the biggest natively supported standard
	 * integer type — currently it's 64-bit unsigned integer value. Emulated
	 * (e.g. *big integer*) integer types are not used, as they don't follow
	 * the ideas behind the fast integer algorithms (they are slower due to
	 * the emulation part).
	 *
	 * **Exponent** is stored in `int` value.
	 *
	 */
	template<> struct diy_fp_traits<double>
	{
		/** \brief Integer type to store mantissa value. */
		typedef std::uint64_t mantissa_type;
		/** \brief Integer type to store exponent value. */
		typedef int exponent_type;
	};

	/** \brief Integer representation of floating point value.
	 *
	 * The templated type represents floating point value using two integer values, one
	 * to store **mantissa** and another to hold **exponent**. Concrete types are
	 * expressed as **diy_fp** specializations, with pre-defined types for **mantissa**
	 * and **exponent**, suitable to process floating point value of the specified
	 * precision with maximum efficiency and without losing accuracy.
	 *
	 * \tparam FloatType floating point type the representation is instantiated for.
	 * \tparam Traits some inner settings (mainly, integer types to store mantissa and
	 * exponent) corresponding to `FloatType`.
	 *
	 * The type is used in **Grisu** and **Krosh** algorithms.
	 */
	template<typename FloatType, typename Traits = diy_fp_traits<FloatType>> class diy_fp
	{
	public:
		/** \brief Mantissa storage type abstraction. */
		typedef typename Traits::mantissa_type mantissa_storage_type;
		/** \brief Exponent storage type abstraction. */
		typedef typename Traits::exponent_type exponent_storage_type;

	private:
		static_assert(std::numeric_limits<FloatType>::is_iec559, "Only IEEE-754 floating point types are supported");
		static_assert(sizeof(FloatType) == sizeof(mantissa_storage_type), "Float type is not compatible with its `diy_fp` representation layout.");

		/** \brief Returns value of hidden bit for the specified floating point type.
		 *
		 * \return integer value of hidden bit of the specified type in
		 * `mantissa_storage_type` variable.
		 */
		static constexpr mantissa_storage_type hidden_bit()
		{
			return raised_bit<mantissa_storage_type>(std::numeric_limits<FloatType>::digits - 1);
		}

	public:
		/** \brief Default constructor. */
		diy_fp() = default;

		/** \brief Copy constructor. */
		diy_fp(const diy_fp&) = default;

		/** \brief Component initialization constructor. */
		constexpr diy_fp(mantissa_storage_type mantissa, exponent_storage_type exponent) noexcept : m_f(mantissa), m_e(exponent) { }

		/** \brief Initializes `diy_fp` value from the value of floating point
		 * type.
		 *
		 * It splits floating point value into mantissa and exponent
		 * components, calculates hidden bit of mantissa and initializes
		 * `diy_fp` value with the results of calculations.
		 */
		explicit diy_fp(FloatType d) noexcept
		{
			constexpr auto full_mantissa_bit_size(std::numeric_limits<FloatType>::digits);
			constexpr auto mantissa_bit_size(full_mantissa_bit_size - 1); // remember hidden bit
			constexpr mantissa_storage_type mantissa_mask(mask<mantissa_storage_type>(mantissa_bit_size));
			constexpr mantissa_storage_type exponent_mask((~mantissa_mask) ^ msb_value<FloatType>()); // ignore sign bit
			constexpr exponent_storage_type exponent_bias(std::numeric_limits<FloatType>::max_exponent - 1 + mantissa_bit_size);

			mantissa_storage_type parts = type_punning_cast<mantissa_storage_type>(d);

			m_f = parts & mantissa_mask;
			m_e = (parts & exponent_mask) >> mantissa_bit_size;

			if (m_e)
			{
				m_f += hidden_bit();
				m_e -= exponent_bias;
			}
			else
			{
				m_e = 1 - exponent_bias;
			}
		}

		/** \brief Downsample result structure.
		 *
		 */
		struct downsample_result
		{
			/** \brief Downsampled floating point result. */
			FloatType value;
			/** \brief Status showing possible under- or overflow found during downsampling. */
			conversion_status status;
			/** \brief Flag indicating if the conversion is accurate (no
			 * [rounding errors] (http://www.exploringbinary.com/decimal-to-floating-point-needs-arbitrary-precision/). */
			bool is_accurate;
		};

		/** \brief Convert `diy_fp` value back to floating point type correctly
		 * downsampling mantissa value.
		 *
		 * The caller should ensure, that the current mantissa value is not null
		 * and the whole `diy_fp` value is normalized, otherwise the behaviour is
		 * undefined.
		 *
		 * \return result structure with floating point value of the specified type.
		 */
		downsample_result downsample()
		{
			downsample_result ret;

			ret.is_accurate = true;
			ret.status = conversion_status::success;

			assert(m_f != 0);

			assert(is_normalized());

			constexpr auto full_mantissa_bit_size(std::numeric_limits<FloatType>::digits);
			constexpr auto mantissa_bit_size(full_mantissa_bit_size - 1); // remember hidden bit
			constexpr mantissa_storage_type my_mantissa_size(bit_size<mantissa_storage_type>());
			constexpr mantissa_storage_type mantissa_mask(mask<mantissa_storage_type>(mantissa_bit_size));
			constexpr exponent_storage_type exponent_bias(std::numeric_limits<FloatType>::max_exponent - 1 + mantissa_bit_size);
			constexpr std::size_t lsb_pow(my_mantissa_size - full_mantissa_bit_size);

			const auto f(m_f);

			if (m_e >= std::numeric_limits<FloatType>::max_exponent)
			{
				ret.value = huge_value<FloatType>();
				ret.status = conversion_status::overflow;
				return ret;
			}

			if (m_e + int(my_mantissa_size) < std::numeric_limits<FloatType>::min_exponent - int(mantissa_bit_size))
			{
				ret.value = FloatType(0);
				ret.status = conversion_status::underflow;
				return ret;
			}

			const std::size_t denorm_exp(positive_part(std::numeric_limits<FloatType>::min_exponent - int(mantissa_bit_size) - m_e - 1));

			assert(denorm_exp < my_mantissa_size);

			const std::size_t shift_amount(std::max(denorm_exp, lsb_pow));

			mantissa_storage_type parts = m_e + shift_amount + exponent_bias - (denorm_exp > lsb_pow);
			parts <<= mantissa_bit_size;

			const auto& round(round_up(f, shift_amount));
			parts |= ((f >> shift_amount) + round.value) & mantissa_mask;

			ret.value = type_punning_cast<FloatType>(parts);
			ret.is_accurate = round.is_accurate;

			return ret;
		}

		/** \brief Mantissa component. */
		constexpr mantissa_storage_type mantissa() const
		{
			return m_f;
		}

		/** \brief Exponent component. */
		constexpr exponent_storage_type exponent() const
		{
			return m_e;
		}

		/** \brief Checks if the value is normalized.
		 *
		 * The behaviour is undefined, if called for null value.
		 *
		 */
		bool is_normalized() const noexcept
		{
			assert(m_f != 0); // normalization of zero is undefined
			return m_f & msb_value<mantissa_storage_type>();
		}


		/** \brief Normalizes the value the common way.
		 *
		 * The caller should ensure, that the current mantissa value is not null,
		 * otherwise the behaviour is undefined.
		 */
		void normalize() noexcept
		{
			assert(m_f != 0); // normalization of zero is undefined

			while (!highest_bit(m_f))
			{
				m_f <<= 1;
				m_e--;
			}
		}

		/** \brief Copy assignment operator. */
		diy_fp& operator=(const diy_fp&) = default;

		/** \brief Subtracts the specified `diy_fp` value from the current.
		 *
		 * Simple mantissa subtraction of `diy_fp` values.
		 *
		 * If exponents of the values differ or mantissa of left value is less,
		 * than mantissa of right value, the behaviour is undefined.
		 *
		 * \param rhs subtrahend.
		 *
		 * \return reference to current value, i.e. the result of the
		 * subtraction.
		 */
		diy_fp& operator-=(const diy_fp& rhs) noexcept
		{
			assert(m_e == rhs.m_e && m_f >= rhs.m_f);

			m_f -= rhs.m_f;

			return *this;
		}

		/** \brief Non-destructive version of `diy_fp::operator-=()`. */
		diy_fp operator-(const diy_fp& rhs) const noexcept
		{
			return diy_fp(*this) -= rhs;
		}

		/** \brief Fast and coarse multiplication.
		 *
		 * Performs multiplication of `diy_fp` values ignoring some carriers
		 * for the sake of performance. This multiplication algorithm is used
		 * in original **Grisu** implementation and also works fine for
		 * **Krosh**.
		 *
		 * \param rhs multiplier.
		 *
		 * \return reference to current value, i.e. the result of the
		 * multiplication.
		 */
		diy_fp& operator*=(const diy_fp& rhs) noexcept
		{
			constexpr std::size_t half_width = bit_size<mantissa_storage_type>() / 2;
			constexpr auto mask_half = mask<mantissa_storage_type>(half_width);

			const mantissa_storage_type a = m_f >> half_width;
			const mantissa_storage_type b = m_f & mask_half;
			const mantissa_storage_type c = rhs.m_f >> half_width;
			const mantissa_storage_type d = rhs.m_f & mask_half;

			const mantissa_storage_type ac = a * c;
			const mantissa_storage_type bc = b * c;
			const mantissa_storage_type ad = a * d;
			const mantissa_storage_type bd = b * d;

			const mantissa_storage_type tmp = (bd >> half_width) + (ad & mask_half) + (bc & mask_half) + raised_bit<mantissa_storage_type>(half_width - 1);

			m_f = ac + (ad >> half_width) + (bc >> half_width) + (tmp >> half_width);
			m_e += rhs.m_e + bit_size<mantissa_storage_type>();

			return *this;
		}

		/** \brief Non-destructive version of `diy_fp::operator*=()`. */
		diy_fp operator*(const diy_fp& rhs) const noexcept
		{
			return diy_fp(*this) *= rhs;
		}

		/** \brief Increment (prefix) with mantissa overflow control. */
		diy_fp& operator++() noexcept
		{
			if (m_f < std::numeric_limits<diy_fp::mantissa_storage_type>::max())
			{
				++m_f;
			}
			else
			{
				m_f >>= 1;
				++m_f;
				++m_e;
			}
			return *this;
		}

		/** \brief Postfix increment version. */
		diy_fp operator++(int) noexcept
		{
			auto temp = *this;
			++(*this);
			return temp;
		}

		/** \brief Decrement (prefix) with mantissa underflow control. */
		diy_fp& operator--() noexcept
		{
			if (m_f > 1)
			{
				--m_f;
			}
			else
			{
				m_f <<= 1;
				--m_f;
				--m_e;
			}
			return *this;
		}

		/** \brief Postfix decrement version. */
		diy_fp operator--(int) noexcept
		{
			auto temp = *this;
			--(*this);
			return temp;
		}

		/** \brief Equality of `diy_fp` values.
		 *
		 * Just member-wise equality check.
		 */
		bool operator==(const diy_fp& d) const noexcept
		{
			return m_f == d.m_f && m_e == d.m_e;
		}


		/** \brief Inequality of `diy_fp` values.
		 *
		 * Negation of `diy_fp::operator==()` for consistency.
		 */
		bool operator!=(const diy_fp& d) const noexcept
		{
			return !operator==(d);
		}

		/** \brief Calculates boundary values (M+ and M-) for the specified
		 * floating point value.
		 *
		 * Helper function for **Grisu2** algorithm, which first converts the
		 * specified floating point value to `diy_fp` and then calculates lower
		 * (M-) and higher (M+) boundaries of it and thus of original accurate
		 * floating point value.
		 *
		 * These two boundary values define the range where all the values are
		 * correctly rounded to the specified floating point value, so any
		 * value within this range can be treated as correct representation of
		 * the specified one.
		 *
		 * \param d floating point value to calculate boundaries for.
		 *
		 * \return `std::pair` of two `diy_fp` values, **M-** and **M+**,
		 * respectively.
		 *
		 * \see [Printing Floating-Point Numbers Quickly and Accurately with
		 * Integers]
		 * (http://florian.loitsch.com/publications/dtoa-pldi2010.pdf)
		 */
		static std::pair<diy_fp, diy_fp> boundaries(FloatType d) noexcept
		{
			std::pair<diy_fp, diy_fp> result;
			diy_fp &mi(result.first), &pl(result.second);
			pl = diy_fp(d);
			mi = pl;

			pl.m_f <<= 1;
			pl.m_f += 1;

			pl.m_e  -= 1;


			pl.normalize_from_ieee754(); // as we increase precision of IEEE-754 type by 1

			const unsigned char shift_amount(1 + (mi.m_f == hidden_bit()));

			mi.m_f <<= shift_amount;
			mi.m_f -= 1;
			mi.m_e -= shift_amount;

			mi.m_f <<= mi.m_e - pl.m_e;
			mi.m_e = pl.m_e;

			return result;
		}

		/** \brief Prints `diy_fp` value.
		 *
		 * \param os `std::basic_ostream` to print to.
		 * \param v `diy_fp` value to print.
		 *
		 * \return `std::basic_ostream` with the \p **v** value printed.
		 */
		template<typename Ch, typename Alloc>
		friend std::basic_ostream<Ch, Alloc>& operator<<(std::basic_ostream<Ch, Alloc>& os, const diy_fp& v)
		{
			os << "(f = " << print_binary(v.m_f) << ", e = " << v.m_e << ')';
			return os;
		}

	private:
		/** \brief Normalizes the value using additional information on
		 * mantissa content of the `FloatType`.

		 * Mantissa value is treated as of the width defined in
		 * `std::numeric_limits`. This information speeds up the normalization,
		 * allowing to shift the value by several positions right at one take,
		 * rather than shifting it by one step and checking if it's still not normalized.
		 *
		 * The caller should ensure, that the current mantissa value is not null
		 * and is really represented in IEEE-754 format, otherwise the behaviour
		 * is undefined.
		 */
		void normalize_from_ieee754() noexcept
		{
			constexpr auto mantissa_bit_width(std::numeric_limits<FloatType>::digits);

			static_assert(mantissa_bit_width >= 0, "Mantissa bit width should be positive.");

			assert(m_f != 0); // normalization of zero is undefined

			while (!nth_bit(m_f, mantissa_bit_width))
			{
				m_f <<= 1;
				m_e--;
			}

			constexpr mantissa_storage_type my_mantissa_size(bit_size<mantissa_storage_type>());
			constexpr mantissa_storage_type e_diff = my_mantissa_size - mantissa_bit_width - 1;

			m_f <<= e_diff;
			m_e -= e_diff;
		}

		mantissa_storage_type m_f;
		exponent_storage_type m_e;
	};
}

#endif // FLOAXIE_DIY_FP_H
