#include <cstdio>
#include <cstring>
#include <regex>
#include <iostream>
#include <stdexcept>
//#include <unistd.h>
//#include "type_name.hpp"
extern "C" void _exit(int);

inline auto rdcycle()
{
	union {
		uint64_t whole;
		uint32_t word[2];
	};
	asm ("rdcycleh %0\n rdcycle %1\n" : "=r"(word[1]), "=r"(word[0]));
	return whole;
}
inline uint64_t rdtime()
{
	union {
		uint64_t whole;
		uint32_t word[2];
	};
	asm ("rdtimeh %0\n rdtime %1\n" : "=r"(word[1]), "=r"(word[0]));
	return whole;
}

int main()
{
	std::string s = "Some people, when confronted with a problem, think "
        "\"I know, I'll use regular expressions.\" "
        "Now they have two problems.";

	std::regex self_regex("REGULAR EXPRESSIONS",
            std::regex_constants::ECMAScript | std::regex_constants::icase);
    if (std::regex_search(s, self_regex)) {
		std::cout << "Text contains the phrase 'regular expressions'\n";
    }

	std::regex word_regex("(\\w+)");
    auto words_begin =
        std::sregex_iterator(s.begin(), s.end(), word_regex);
    auto words_end = std::sregex_iterator();

	std::cout << "Found "
              << std::distance(words_begin, words_end)
              << " words\n";

	const int N = 6;
	std::cout << "Words longer than " << N << " characters:\n";
	for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
		std::smatch match = *i;
		std::string match_str = match.str();
		if (match_str.size() > N) {
			std::cout << "  " << match_str << '\n';
		}
	}

	std::regex long_word_regex("(\\w{7,})");
	std::string new_s = std::regex_replace(s, long_word_regex, "[$&]");
	printf("%s\n", new_s.c_str());

	printf("Testing exception\n");
	const auto cycle0 = rdcycle();
	try {
		throw std::runtime_error("Hello Exceptions!");
	}
	catch (const std::exception& e) {
		printf("Caught exception: %s\n", e.what());
	}
	const auto cycle1 = rdcycle();
	printf("It took %llu instructions to throw, catch and print the exception\n", cycle1-cycle0);

	// if we don't return from main we can continue calling functions in the VM
	// exit(int) will call destructors, which breaks the C runtime environment
	// instead, call _exit which is just a shortcut for the EXIT system call.
	_exit(666);
}

static std::vector<int> array;

extern "C"
int test(int arg1)
{
	printf("Test called with argument %d\n", arg1);
	array.push_back(arg1);
	for (const int val : array) {
		printf("Array: %d\n", val);
	}
	printf("Returning 777\n");
	return 777;
}
