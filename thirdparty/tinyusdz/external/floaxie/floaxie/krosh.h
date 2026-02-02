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
 */

#ifndef FLOAXIE_CROSH_H
#define FLOAXIE_CROSH_H

#include <vector>
#include <locale>
#include <cstddef>
#include <cmath>
#include <cassert>

#include <floaxie/diy_fp.h>
#include <floaxie/static_pow.h>
#include <floaxie/k_comp.h>
#include <floaxie/cached_power.h>
#include <floaxie/bit_ops.h>
#include <floaxie/fraction.h>
#include <floaxie/conversion_status.h>

namespace floaxie
{
	/** \brief Maximum number of decimal digits mantissa of `diy_fp` can hold. */
	template<typename FloatType> constexpr std::size_t decimal_q = std::numeric_limits<typename diy_fp<FloatType>::mantissa_storage_type>::digits10;

	/** \brief Maximum number of necessary binary digits of fraction part. */
	constexpr std::size_t fraction_binary_digits(7);

	/** \brief Maximum number of decimal digits of fraction part, which can be observed. */
	constexpr std::size_t fraction_decimal_digits(4);

	/** \brief Maximum length of input string (2 KB). */
	constexpr std::size_t maximum_offset = 2048;

	/** \brief Maximum number of decimal digits in the exponent value. */
	constexpr std::size_t exponent_decimal_digits(3);

	/** \brief Tries to find and eat NaN representation in one of two forms.
	 *
	 * Searches for either "NAN" or "NAN(<character sequence>)" form of NaN
	 * (not a number) value representation (case insensitive) to help
	 * converting it to quiet NaN and finding the end of the read value.
	 *
	 * \param str character buffer to analyze.
	 *
	 * \return number of consumed characters. Naturally, if it's equal to zero,
	 * NaN representation wasn't found.
	 */
	template<typename CharType> std::size_t eat_nan(const CharType* str) noexcept
	{
		std::size_t eaten(0);

		if ((str[0] == 'a' || str[0] == 'A') && (str[1] == 'n' || str[1] == 'N'))
		{
			const CharType* cp = str + 2;
			eaten = 2;

			/* Match `(n-char-sequence-digit)'.  */
			if (*cp == '(')
			{
				do
					++cp;
				while ((*cp >= '0' && *cp <= '9') ||
					   (std::tolower(*cp, std::locale()) >= 'a' && std::tolower(*cp, std::locale()) <= 'z') ||
						*cp == '_');

				if (*cp == ')')
					eaten = cp - str + 1;
			}
		}

		return eaten;
	}

	/** \brief Tries to find and eat infinity representation.
	 *
	 * Searches for either "inf" or "infinity" sequence (case insensitive)
	 * to determine infinite floating point value representation.
	 *
	 * \param str character buffer to analyze.
	 *
	 * \return number of consumed characters. Naturally, if it's equal to zero,
	 * infinity representation wasn't found.
	 */
	template<typename CharType> std::size_t eat_inf(const CharType* str) noexcept
	{
		std::size_t eaten(0);

		if ((str[0] == 'n' || str[0] == 'N') && (str[1] == 'f' || str[1] == 'F'))
		{
			const CharType* cp = str + 2;
			eaten = 2;

			if (*cp == 'i' || *cp == 'I')
			{
				++cp;

				const std::array<CharType, 4> suffix {{ 'n', 'i', 't', 'y' }};
				auto it = suffix.cbegin();

				while (it != suffix.cend() && std::tolower(*cp, std::locale()) == *it)
				{
					++cp;
					++it;
				}

				if (it == suffix.cend())
					eaten = cp - str;
			}
		}

		return eaten;
	}

	/** \brief Extracts up to \p **kappa** decimal digits from fraction part.
	 *
	 * Extracts decimal digits from fraction part and returns it as numerator
	 * value with denominator equal to \f$10^{\kappa}\f$.
	 *
	 * \tparam kappa maximum number of decimal digits to extract.
	 * \tparam FloatType destination type of floating point value to store the
	 * results.
	 * \tparam CharType character type (typically `char` or `wchar_t`) \p **str**
	 * consists of.
	 *
	 * \param str character buffer to extract from.
	 *
	 * \return Numerator value of the extracted decimal digits (i.e. as they
	 * are actually written after the decimal point).
	 */
	template<std::size_t kappa, typename CharType>
	inline unsigned int extract_fraction_digits(const CharType* str)
	{
		static_assert(kappa <= std::numeric_limits<int>::digits10, "Extracting values, exceeding 'int' capacity, is not supported.");

		std::array<unsigned char, kappa> parsed_digits;
		parsed_digits.fill(0);

		for (std::size_t pos = 0; pos < kappa; ++pos)
		{
			const auto c = str[pos] - '0';
			if (c >= 0 && c <= 9)
				parsed_digits[pos] = c;
			else
				break;
		}

		unsigned int result(0);
		std::size_t pow(0);
		for (auto rit = parsed_digits.rbegin(); rit != parsed_digits.rend(); ++rit)
			result += (*rit) * seq_pow<unsigned int, 10, kappa>(pow++);

		return result;
	}

	/** \brief Type of special value. */
	enum class speciality : unsigned char
	{
		/** \brief Normal value - no special. */
		no = 0,
		/** \brief NaN (not a number) value. */
		nan,
		/** \brief infinity value. */
		inf
	};

	/** \brief Return structure for `parse_digits`.
	 *
	 * \tparam FloatType destination type of floating point value to store the
	 * results.
	 * \tparam CharType character type (typically `char` or `wchar_t`) used.
	 */
	template<typename FloatType, typename CharType> struct digit_parse_result
	{
		/** \brief Pre-initializes members to sane values. */
		digit_parse_result() : value(), K(0), str_end(nullptr), frac(0), special(), sign(true) { }

		/** \brief Parsed mantissa value. */
		typename diy_fp<FloatType>::mantissa_storage_type value;

		/** \brief Decimal exponent, as calculated by exponent part and decimal
		 * point position.
		 */
		int K;

		/** \brief Pointer to the memory after the parsed part of the buffer. */
		const CharType* str_end;

		/** \brief Binary numerator of fractional part, to help correct rounding. */
		unsigned char frac;

		/** \brief Flag of special value possibly occured. */
		speciality special;

		/** \brief Sign of the value. */
		bool sign;
	};

	/** \brief Unified method to extract and parse digits in one pass.
	 *
	 * Goes through the string representation of the floating point value in
	 * the specified buffer, detects the meaning of each digit in its position
	 * and calculates main parts of floating point value — mantissa, exponent,
	 * sign, fractional part.
	 *
	 * \tparam kappa maximum number of digits to expect.
	 * \tparam calc_frac if `true`, try to calculate fractional part, if any.
	 * \tparam FloatType destination type of floating point value to store the
	 * results.
	 * \tparam CharType character type (typically `char` or `wchar_t`) \p **str**
	 * consists of.
	 *
	 * \param str Character buffer with floating point value representation to
	 * parse.
	 *
	 * \return `digit_parse_result` with the parsing results.
	 */
	template<typename FloatType, typename CharType>
	inline digit_parse_result<FloatType, CharType> parse_digits(const CharType* str) noexcept
	{
		digit_parse_result<FloatType, CharType> ret;

		constexpr std::size_t kappa = decimal_q<FloatType>;

		std::vector<unsigned char> parsed_digits;
		parsed_digits.reserve(kappa);

		bool dot_set(false);
		bool sign_set(false);
		bool frac_calculated(false);
		std::size_t pow_gain(0);
		std::size_t zero_substring_length(0), fraction_digits_count(0);

		bool go_to_beach(false);
		std::size_t pos(0);

		while(!go_to_beach)
		{
			const auto c = str[pos];
			switch (c)
			{
			case '0':
				if (!parsed_digits.empty() || dot_set)
				{
					++zero_substring_length;
					pow_gain += !dot_set;
				}
				break;

			case '1':
			case '2':
			case '3':
			case '4':
			case '5':
			case '6':
			case '7':
			case '8':
			case '9':
				if (zero_substring_length && parsed_digits.size() < kappa)
				{
					const std::size_t spare_digits { kappa - parsed_digits.size() };
					auto zero_copy_count = zero_substring_length;
					auto pow_gain_reduced = pow_gain;

					if (!parsed_digits.empty())
					{
						zero_copy_count = std::min(zero_substring_length, spare_digits);
						pow_gain_reduced = std::min(pow_gain, spare_digits);

						parsed_digits.insert(parsed_digits.end(), zero_copy_count, 0);
					}

					fraction_digits_count += zero_copy_count - pow_gain_reduced;
					zero_substring_length -= zero_copy_count;
					pow_gain -= pow_gain_reduced;
				}

				if (parsed_digits.size() < kappa)
				{
					parsed_digits.push_back(c - '0');
					fraction_digits_count += dot_set;
				}
				else
				{
					if (!frac_calculated)
					{
						const std::size_t frac_suffix_size = parsed_digits.size() + zero_substring_length - kappa;
						auto tail = extract_fraction_digits<fraction_decimal_digits>(str + pos - frac_suffix_size);
						ret.frac = convert_numerator<fraction_decimal_digits, fraction_binary_digits>(tail);

						frac_calculated = true;
					}

					pow_gain += !dot_set;
				}
				break;

			case '.':
				go_to_beach = dot_set;
				dot_set = true;
				break;

			case 'n':
			case 'N':
				if (pos == sign_set)
				{
					const std::size_t eaten = eat_nan(str + pos + 1);
					pos += eaten + 1;

					if (eaten)
						ret.special = speciality::nan;
				}

				go_to_beach = true;
				break;

			case 'i':
			case 'I':
				if (pos == sign_set)
				{
					const std::size_t eaten = eat_inf(str + pos + 1);
					pos += eaten + 1;

					if (eaten)
						ret.special = speciality::inf;
				}

				go_to_beach = true;
				break;

			case '-':
			case '+':
				if (pos == 0)
				{
					ret.sign = static_cast<bool>('-' - c); // '+' => true, '-' => false
					sign_set = true;
					break;
				}
				// fall through

			default:
				go_to_beach = true;
				break;
			}

			go_to_beach |= pos > maximum_offset;

			++pos;
		}

		std::size_t pow(0);
		for (auto rit = parsed_digits.rbegin(); rit != parsed_digits.rend(); ++rit)
			ret.value += (*rit) * seq_pow<typename diy_fp<FloatType>::mantissa_storage_type, 10, decimal_q<FloatType>>(pow++);

		ret.str_end = str + (pos - 1);
		ret.K = pow_gain - fraction_digits_count;

		return ret;
	}

	/** \brief Return structure for `parse_mantissa`.
	 *
	 * \tparam FloatType destination value floating point type.
	 * \tparam CharType character type (typically `char` or `wchar_t`) used.
	 */
	template<typename FloatType, typename CharType> struct mantissa_parse_result
	{
		/** \brief Calculated mantissa value. */
		diy_fp<FloatType> value;

		/** \brief Corrected value of decimal exponent value. */
		int K;

		/** \brief Pointer to the memory after the parsed part of the buffer. */
		const CharType* str_end;

		/** \brief Flag of special value. */
		speciality special;

		/** \brief Sign of the value. */
		bool sign;
	};

	/** \brief Tides up results of `parse_digits` for **Krosh** to use.
	 *
	 * Packs mantissa value into `diy_fp` structure and performs the necessary
	 * rounding up according to the fractional part value.
	 *
	 * \tparam FloatType destination type of floating point value to store the
	 * results.
	 * \tparam CharType character type (typically `char` or `wchar_t`) \p **str**
	 * consists of.
	 *
	 * \param str Character buffer with floating point value representation to
	 * parse.
	 *
	 * \return `mantissa_parse_result` structure with the results of parsing
	 * and corrections.
	 */
	template<typename FloatType, typename CharType> inline mantissa_parse_result<FloatType, CharType> parse_mantissa(const CharType* str)
	{
		mantissa_parse_result<FloatType, CharType> ret;

		const auto& digits_parts(parse_digits<FloatType>(str));

		ret.special = digits_parts.special;
		ret.str_end = digits_parts.str_end;
		ret.sign = digits_parts.sign;

		if (digits_parts.special == speciality::no)
		{
			ret.value = diy_fp<FloatType>(digits_parts.value, 0);
			ret.K = digits_parts.K;

			if (digits_parts.value)
			{
				auto& w(ret.value);
				w.normalize();

				// extract additional binary digits and round up gently
				if (digits_parts.frac)
				{
					assert(w.exponent() > (-1) * static_cast<int>(fraction_binary_digits));
					const std::size_t lsb_pow(fraction_binary_digits + w.exponent());

					typename diy_fp<FloatType>::mantissa_storage_type f(w.mantissa());
					f |= digits_parts.frac >> lsb_pow;

					w = diy_fp<FloatType>(f, w.exponent());

					// round correctly avoiding integer overflow, undefined behaviour, pain and suffering
					if (round_up(digits_parts.frac, lsb_pow).value)
					{
						++w;
					}
				}
			}
		}

		return ret;
	}

	/** \brief Return structure for `parse_exponent`.
	 *
	 * \tparam CharType character type (typically `char` or `wchar_t`) used.
	 */
	template<typename CharType> struct exponent_parse_result
	{
		/** \brief Value of the exponent. */
		int value;

		/** \brief Pointer to the memory after the parsed part of the buffer. */
		const CharType* str_end;
	};

	/** \brief Parses exponent part of the floating point string representation.
	 *
	 * \tparam CharType character type (typically `char` or `wchar_t`) of \p **str**.
	 *
	 * \param str Exponent part of character buffer with floating point value
	 * representation to parse.
	 *
	 * \return `exponent_parse_result` structure with parse results.
	 */
	template<typename CharType> inline exponent_parse_result<CharType> parse_exponent(const CharType* str)
	{
		exponent_parse_result<CharType> ret;
		if (*str != 'e' && *str != 'E')
		{
			ret.value = 0;
			ret.str_end = str;
		}
		else
		{
			++str;

			const auto& digit_parts(parse_digits<float>(str));

			ret.value = digit_parts.value * seq_pow<int, 10, exponent_decimal_digits>(digit_parts.K);

			if (!digit_parts.sign)
				ret.value = -ret.value;

			ret.str_end = digit_parts.str_end;
		}

		return ret;
	}

	/** \brief Return structure, containing **Krosh** algorithm results.
	 *
	 * \tparam FloatType destination type of floating point value to store the
	 * results.
	 * \tparam CharType character type (typically `char` or `wchar_t`) used.
	 */
	template<typename FloatType, typename CharType> struct krosh_result
	{
		/** \brief The result floating point value, downsampled to the defined
		 * floating point type.
		 */
		FloatType value;

		/** \brief Pointer to the memory after the parsed part of the buffer. */
		const CharType* str_end;

		/** \brief Status of the performed conversion. */
		conversion_status status;

		/** \brief Flag indicating if the result ensured to be rounded correctly. */
		bool is_accurate;
	};

	/** \brief Implements **Krosh** algorithm.
	 *
	 * \tparam FloatType destination type of floating point value to store the
	 * results.
	 *
	 * \tparam CharType character type (typically `char` or `wchar_t`) \p **str**
	 * consists of.
	 *
	 * \param str Character buffer with floating point value
	 * representation to parse.
	 *
	 * \return `krosh_result` structure with all the results of **Krosh**
	 * algorithm.
	 */
	template<typename FloatType, typename CharType> krosh_result<FloatType, CharType> krosh(const CharType* str)
	{
		krosh_result<FloatType, CharType> ret;

		static_assert(sizeof(FloatType) <= sizeof(typename diy_fp<FloatType>::mantissa_storage_type), "Only floating point types no longer, than 64 bits are supported.");

		auto mp(parse_mantissa<FloatType>(str));

		if (mp.special == speciality::no && mp.value.mantissa())
		{
			diy_fp<FloatType>& w(mp.value);

			const auto& ep(parse_exponent(mp.str_end));

			mp.K += ep.value;

			if (mp.K)
			{
				const bool b1 = mp.K >= powers_ten<FloatType>::boundaries.first;
				const bool b2 = mp.K <= powers_ten<FloatType>::boundaries.second;

				if (b1 && b2)
				{
					w *= cached_power<FloatType>(mp.K);
				}
				else
				{
					if (!b1)
					{
						ret.value = FloatType(0);
						ret.status = conversion_status::underflow;
					}
					else // !b2
					{
						ret.value = huge_value<FloatType>();
						ret.status = conversion_status::overflow;
					}

					ret.str_end = ep.str_end;
					ret.is_accurate = true;

					return ret;
				}
			}

			w.normalize();
			const auto& v(w.downsample());
			ret.value = v.value;
			ret.str_end = ep.str_end;
			ret.is_accurate = v.is_accurate;
			ret.status = v.status;
		}
		else
		{
			switch (mp.special)
			{
			case speciality::nan:
				ret.value = std::numeric_limits<FloatType>::quiet_NaN();
				break;

			case speciality::inf:
				ret.value = std::numeric_limits<FloatType>::infinity();
				break;

			default:
				ret.value = 0;
				break;
			}

			ret.str_end = mp.str_end;
			ret.is_accurate = true;
			ret.status = conversion_status::success;
		}

		if (!mp.sign)
			ret.value = -ret.value;

		return ret;
	}
}

#endif // FLOAXIE_CROSH_H
