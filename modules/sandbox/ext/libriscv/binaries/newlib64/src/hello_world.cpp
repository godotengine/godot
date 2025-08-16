#include <cstdio>
#include <cstring>
#include <regex>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

inline uint64_t rdcycle()
{
	uint64_t whole;
	asm ("rdcycle %0\n" : "=r"(whole));
	return whole;
}
inline uint64_t rdtime()
{
	uint64_t whole;
	asm ("rdtime %0\n" : "=r"(whole));
	return whole;
}
inline uint64_t rol(uint64_t val, unsigned shift)
{
	uint64_t result;
	asm("rol %0, %1, %2" : "=r"(result) : "r"(val), "r"(shift));
	return result;
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

	const auto cycle0 = rdcycle();
	try {
		printf("Testing exception\n");
		throw std::runtime_error("Hello Exceptions!");
	}
	catch (const std::exception& e) {
		printf("Caught exception: %s\n", e.what());
	}
	const auto cycle1 = rdcycle();
	printf("It took %lu instructions to throw, catch and print the exception\n", cycle1-cycle0);

	_exit(666);
}

static std::vector<int> array;

extern "C" __attribute__((used, retain))
int test(int arg1, const char *arg2)
{
	printf("Test called with argument %d and string argument '%s'\n", arg1, arg2);
	array.push_back(arg1);
	for (const int val : array) {
		printf("Array: %d\n", val);
	}
	printf("Returning 777\n");
	fflush(stdout); /* stdout is buffered. */
	return 777;
}
