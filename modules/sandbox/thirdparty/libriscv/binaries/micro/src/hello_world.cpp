#include "syscall.hpp"

struct String {
	const char* data;
	unsigned len = 0;

	template<int N>
	constexpr String(const char (&str)[N]) : data(str), len(N-1) {}

	constexpr String(const char* str, unsigned l) : data(str), len(l) {}

	constexpr String(const char* str) : data(str), len(0) {
		while(str[len] != 0) len++;
	}
};

template <typename... Args>
inline void print(Args&&... args) {
	([&] {
		const String str {args};
		write(1, str.data, str.len);
	}(), ...);
}

static struct Test {
	Test() {
		print("Hello, Global Constructor!\n");
	}
} test;

int main(int, char** argv)
{
	print("Hello World from ", argv[0], "!\n");
	return 666;
}
