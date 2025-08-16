#pragma once
#include <httplib.h>
#include <string>
#include <vector>

extern void load_file(const std::string& filename, std::vector<uint8_t>&);

extern void compile(const httplib::Request&, httplib::Response&);
extern void execute(
	const httplib::Request&, httplib::Response&, const httplib::ContentReader&);

#include <sys/time.h>
inline uint64_t micros_now()
{
	struct timespec ts;
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
	return ts.tv_sec * 1000000ul + ts.tv_nsec / 1000ul;
}

inline uint64_t monotonic_micros_now()
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000000ul + ts.tv_nsec / 1000ul;
}
