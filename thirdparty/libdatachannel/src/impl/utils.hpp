/**
 * Copyright (c) 2020-2022 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_IMPL_UTILS_H
#define RTC_IMPL_UTILS_H

#include "common.hpp"

#include <climits>
#include <limits>
#include <map>
#include <random>
#include <vector>

namespace rtc::impl::utils {

std::vector<string> explode(const string &str, char delim);
string implode(const std::vector<string> &tokens, char delim);

// Decode URL percent-encoding (RFC 3986)
// See https://www.rfc-editor.org/rfc/rfc3986.html#section-2.1
string url_decode(const string &str);

// Encode as base64 (RFC 4648)
// See https://www.rfc-editor.org/rfc/rfc4648.html#section-4
string base64_encode(const binary &data);

// Return a random seed sequence
std::seed_seq random_seed();

template <typename Generator, typename Result = typename Generator::result_type>
struct random_engine_wrapper {
	Generator &engine;
	using result_type = Result;
	static constexpr result_type min() { return static_cast<Result>(Generator::min()); }
	static constexpr result_type max() { return static_cast<Result>(Generator::max()); }
	inline result_type operator()() { return static_cast<Result>(engine()); }
	inline void discard(unsigned long long z) { engine.discard(z); }
};

// Return a wrapped thread-local seeded random number generator
template <typename Generator = std::mt19937, typename Result = typename Generator::result_type>
auto random_engine() {
	static thread_local std::seed_seq seed = random_seed();
	static thread_local Generator engine{seed};
	return random_engine_wrapper<Generator, Result>{engine};
}

// Return a wrapped thread-local seeded random bytes generator
template <typename Generator = std::mt19937> auto random_bytes_engine() {
	using char_independent_bits_engine =
	    std::independent_bits_engine<Generator, CHAR_BIT, unsigned short>;
	static_assert(char_independent_bits_engine::min() == std::numeric_limits<uint8_t>::min());
	static_assert(char_independent_bits_engine::max() == std::numeric_limits<uint8_t>::max());
	return random_engine<char_independent_bits_engine, uint8_t>();
}

namespace this_thread {

void set_name(const string &name);

} // namespace this_thread

} // namespace rtc::impl::utils

#endif
