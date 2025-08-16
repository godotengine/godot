<a id="top"></a>
# Why do we need yet another C++ test framework?

Good question. For C++ there are quite a number of established frameworks,
including (but not limited to),
[Google Test](http://code.google.com/p/googletest/),
[Boost.Test](http://www.boost.org/doc/libs/1_49_0/libs/test/doc/html/index.html),
[CppUnit](http://sourceforge.net/apps/mediawiki/cppunit/index.php?title=Main_Page),
[Cute](http://www.cute-test.com), and
[many, many more](http://en.wikipedia.org/wiki/List_of_unit_testing_frameworks#C.2B.2B).

So what does Catch2 bring to the party that differentiates it from these? Apart from the catchy name, of course.


## Key Features

* Quick and easy to get started. Just download two files, add them into your project and you're away.
* No external dependencies. As long as you can compile C++14 and have the C++ standard library available.
* Write test cases as, self-registering, functions (or methods, if you prefer).
* Divide test cases into sections, each of which is run in isolation (eliminates the need for fixtures).
* Use BDD-style Given-When-Then sections as well as traditional unit test cases.
* Only one core assertion macro for comparisons. Standard C/C++ operators are used for the comparison - yet the full expression is decomposed and lhs and rhs values are logged.
* Tests are named using free-form strings - no more couching names in legal identifiers.


## Other core features

* Tests can be tagged for easily running ad-hoc groups of tests.
* Failures can (optionally) break into the debugger on common platforms.
* Output is through modular reporter objects. Basic textual and XML reporters are included. Custom reporters can easily be added.
* JUnit xml output is supported for integration with third-party tools, such as CI servers.
* A default main() function is provided, but you can supply your own for complete control (e.g. integration into your own test runner GUI).
* A command line parser is provided and can still be used if you choose to provide your own main() function.
* Alternative assertion macro(s) report failures but don't abort the test case
* Good set of facilities for floating point comparisons (`Catch::Approx` and full set of matchers)
* Internal and friendly macros are isolated so name clashes can be managed
* Data generators (data driven test support)
* Hamcrest-style Matchers for testing complex properties
* Microbenchmarking support


## Who else is using Catch2?

A whole lot of people. According to [the 2022 JetBrains C++ ecosystem survey](https://www.jetbrains.com/lp/devecosystem-2022/cpp/#Which-unit-testing-frameworks-do-you-regularly-use),
about 12% of C++ programmers use Catch2 for unit testing, making it the
second most popular unit testing framework.

You can also take a look at the (incomplete) list of [open source projects](opensource-users.md#top)
or the (very incomplete) list of [commercial users of Catch2](commercial-users.md#top)
for some idea on who else also uses Catch2.

---

See the [tutorial](tutorial.md#top) to get more of a taste of using
Catch2 in practice.

---

[Home](Readme.md#top)
