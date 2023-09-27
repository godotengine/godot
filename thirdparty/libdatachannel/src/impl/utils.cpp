/**
 * Copyright (c) 2020-2022 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "utils.hpp"

#include "impl/internals.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <functional>
#include <iterator>
#include <sstream>
#include <thread>

#if defined(_WIN32)
#include <windows.h>

typedef HRESULT(WINAPI *pfnSetThreadDescription)(HANDLE, PCWSTR);
#endif
#if defined(__linux__)
#include <sys/prctl.h> // for prctl(PR_SET_NAME)
#endif
#if defined(__FreeBSD__)
#include <pthread_np.h> // for pthread_set_name_np
#endif

namespace rtc::impl::utils {

using std::to_integer;

std::vector<string> explode(const string &str, char delim) {
	std::vector<std::string> result;
	std::istringstream ss(str);
	string token;
	while (std::getline(ss, token, delim))
		result.push_back(token);

	return result;
}

string implode(const std::vector<string> &tokens, char delim) {
	string sdelim(1, delim);
	std::ostringstream ss;
	std::copy(tokens.begin(), tokens.end(), std::ostream_iterator<string>(ss, sdelim.c_str()));
	string result = ss.str();
	if (result.size() > 0)
		result.resize(result.size() - 1);

	return result;
}

string url_decode(const string &str) {
	string result;
	size_t i = 0;
	while (i < str.size()) {
		char c = str[i++];
		if (c == '%') {
			auto value = str.substr(i, 2);
			try {
				if (value.size() != 2 || !std::isxdigit(value[0]) || !std::isxdigit(value[1]))
					throw std::exception();

				c = static_cast<char>(std::stoi(value, nullptr, 16));
				i += 2;

			} catch (...) {
				PLOG_WARNING << "Invalid percent-encoded character in URL: \"%" + value + "\"";
			}
		}

		result.push_back(c);
	}

	return result;
}

string base64_encode(const binary &data) {
	static const char tab[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

	string out;
	out.reserve(3 * ((data.size() + 3) / 4));
	int i = 0;
	while (data.size() - i >= 3) {
		auto d0 = to_integer<uint8_t>(data[i]);
		auto d1 = to_integer<uint8_t>(data[i + 1]);
		auto d2 = to_integer<uint8_t>(data[i + 2]);
		out += tab[d0 >> 2];
		out += tab[((d0 & 3) << 4) | (d1 >> 4)];
		out += tab[((d1 & 0x0F) << 2) | (d2 >> 6)];
		out += tab[d2 & 0x3F];
		i += 3;
	}

	int left = int(data.size() - i);
	if (left) {
		auto d0 = to_integer<uint8_t>(data[i]);
		out += tab[d0 >> 2];
		if (left == 1) {
			out += tab[(d0 & 3) << 4];
			out += '=';
		} else { // left == 2
			auto d1 = to_integer<uint8_t>(data[i + 1]);
			out += tab[((d0 & 3) << 4) | (d1 >> 4)];
			out += tab[(d1 & 0x0F) << 2];
		}
		out += '=';
	}

	return out;
}

std::seed_seq random_seed() {
	std::vector<unsigned int> seed;

	// Seed with random device
	try {
		// On some systems an exception might be thrown if the random_device can't be initialized
		std::random_device device;
		// 128 bits should be more than enough
		std::generate_n(std::back_inserter(seed), 4, std::ref(device));
	} catch (...) {
		// Ignore
	}

	// Seed with high-resolution clock
	using std::chrono::high_resolution_clock;
	seed.push_back(
	    static_cast<unsigned int>(high_resolution_clock::now().time_since_epoch().count()));

	// Seed with thread id
	seed.push_back(
	    static_cast<unsigned int>(std::hash<std::thread::id>{}(std::this_thread::get_id())));

	return std::seed_seq(seed.begin(), seed.end());
}

namespace {

void thread_set_name_self(const char *name) {
#if defined(_WIN32)
	int name_length = (int)strlen(name);
	int wname_length =
	    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, name, name_length, nullptr, 0);
	if (wname_length > 0) {
		std::wstring wname(wname_length, L'\0');
		wname_length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, name, name_length,
		                                   &wname[0], wname_length + 1);

		HMODULE kernel32 = GetModuleHandleW(L"kernel32.dll");
		if (kernel32 != nullptr) {
			auto pSetThreadDescription =
			    (pfnSetThreadDescription)GetProcAddress(kernel32, "SetThreadDescription");
			if (pSetThreadDescription != nullptr) {
				pSetThreadDescription(GetCurrentThread(), wname.c_str());
			}
		}
	}
#elif defined(__linux__)
	prctl(PR_SET_NAME, name);
#elif defined(__APPLE__)
	pthread_setname_np(name);
#elif defined(__FreeBSD__)
	pthread_set_name_np(pthread_self(), name);
#else
	(void)name;
#endif
}

} // namespace

namespace this_thread {

void set_name(const string &name) { thread_set_name_self(name.c_str()); }

} // namespace this_thread

} // namespace rtc::impl::utils
