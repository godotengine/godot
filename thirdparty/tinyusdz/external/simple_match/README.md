# Simple, Extensible C++ Pattern Matching Library

I have recently been looking at Haskell and Rust. One of the things I wanted in C++ from those languages is pattern matching.

Here is an example from the Rustlang Book (http://static.rust-lang.org/doc/master/book/match.html)

```rust
match x {
    1 => println!("one"),
    2 => println!("two"),
    3 => println!("three"),
    4 => println!("four"),
    5 => println!("five"),
    _ => println!("something else"),
}
```
There is currently a C++ Library Mach7 that does pattern matching (https://github.com/solodon4/Mach7), however it is big, complicated, and uses a lot of macros. I wanted to see if I could use C++14 to write a simple implementation without macros.

This library is the result of that effort. If you are familiar with C++14 especially variadic templates, forwarding, and tuples, this library and implementation should be easy for you to understand and extend.

## Usage
You will need a C++14 compiler. I have used the latest Visual C++ 2015 CTP, GCC 4.9.2, and Clang 3.5 to test this library.

The library consists of 2 headers. `simple_match.hpp` which is the core of the library, and `some_none.hpp` which contains code that lets you match on raw pointers, and unique_ptr, and shared_ptr.

Here is a simple excerpt. Assume you have included simple_match.hpp

```cpp
using namespace simple_match;
using namespace simple_match::placeholders;

int x = 0;

while (true) {
	std::cin >> x;
	match(x,
		1, []() {std::cout << "The answer is one\n"; },
		2, []() {std::cout << "The answer is two\n"; },
		_x < 10, [](auto&& a) {std::cout << "The answer " << a << " is less than 10\n"; },
		10 < _x < 20,	[](auto&& a) {std::cout << "The answer " << a  << " is between 10 and 20 exclusive\n"; },
		_, []() {std::cout << "Did not match\n"; }
	);
}
```

## Example Files
There are 2 files under the test directory: `test.cpp` and `cppcon-matching.cpp`. `test.cpp` contains just some simple tests of matching. `cppcon-matching.cpp`contains the example from Mach7 that was presented at cppcon.

## Extending
There are 2 points of customization provided in namespace `simple_matcher::customization`. They are
```cpp
template<class T, class U>
struct matcher;
```

and

```cpp
template<class Type>
struct tuple_adapter;
```
## License
Licensed under the Boost Software License.

## Tutorial

We are going to assume you have the following at the top of your file

```cpp
#include "simple_match/simple_match.hpp"


using namespace simple_match;
using namespace simple_match::placeholders;
```

Here is how to match exactly

```cpp
int i = 0;
match(i,
	1, [](){std::cout << "The answer is one";}
	2, [](){std::cout << "The answer is two";}
	otherwise, [](){std::cout << "Did not match"}
);
```
The match function will try matching from top to bottom and run the lamba corresponding to the first successful match. `otherwise` always matches, and therefore you should have it at the end. If you find `otherwise` too long, you can also use `_`. It is located in the namespace `simple_match::placeholders`

Match also works for strings.
```
std::string s = "";

match(s,
	"Hello", [](){ std::cout << " You said hello\n";},
	_, [](){std::cout << "I do not know what you said\n";} // _ is the same as otherwise
);

```


You can even return values from a match

```cpp
char digit = '0';

int value = match(digit,
	'0', [](){return 0;},
	'1', [](){return 1;},
	'2', [](){return 2;},
	'3', [](){return 3;},
	'4', [](){return 4;},
	'5', [](){return 5;},
// and so on
);
```

We can also do comparisons, and ranges. To do so use `_x` from the `simple_match::placeholders` namespace.

```cpp
int i = 0;
match(i,
	_x < 0, [](int x){std::cout << x << " is a negative number\n";},
	1 < _x < 10, [](int z){std::cout << z << " is between 1 and 10\n"},
	_x, [](int x){std::cout << x << " is the value\n";}
);

```
There are a some items of interest in the above example. When `_x` is used, it passes its value to the lambda. If `_x` is used without any comparison, it will pass the value to the lambda. Also, because of the way it is overloaded, it is very easy to make ranges using the `<` or `<=` operator as seen in the match above.

### Tuples

Now we can even have more fun! Let's represent a 2d point as a tuple.

```cpp
std::tuple<int,int> p(0,0);

match(p,
	ds(0,0), [](){std::cout << "Point is at the origin";},
	ds(0,_), [](){std::cout << "Point is on the horizontal axis";},
	ds(_,0), [](){std::cout << "Point is on the vertical axis";}.
	ds(_x < 10,_), [](int x){std::cout << x << " is less than 10";},
	ds(_x,_x), [](int x, int y){ std::cout << x << "," << y << " Is not on an axis";}
);
```

`ds` stands for de-structure and splits a tuple into its parts. Notice you can use the same expressions as you could without tuples. As before `_x` results in a value being passed to the lambda. `_` matches anything and ignores it, so no corresponding variable is passed to the lambda.

We can actually use `ds` to deconstruct our own `struct`s and `class`es .
First we have to specialize `simple_match::customization::tuple_adapter` for our type.

```cpp
struct point {
	int x;
	int y;
	point(int x_,int y_):x(x_),y(y_){}
};

// Adapting point to be used with ds
namespace simple_match {
	namespace customization {
		template<>
		struct tuple_adapter<point>{

			enum { tuple_len = 2 };

			template<size_t I, class T>
			static decltype(auto) get(T&& t) {
				return std::get<I>(std::tie(t.x,t.y));
			}
		};
	}
}
```

Then we can use `ds` like we did with a tuple.

```cpp
point p{0,0};

match(p,
	ds(0,0), [](){std::cout << "Point is at the origin";},
	ds(0,_), [](){std::cout << "Point is on the horizontal axis";},
	ds(_,0), [](){std::cout << "Point is on the vertical axis";}.
	ds(_x < 10,_), [](int x){std::cout << x << " is less than 10";},
	ds(_x,_x), [](int x, int y){ std::cout << x << "," << y << " Is not on an axis";}
);
```

### Pointers as option types
Sometimes we have pointer that we want to get a value safely out of. To do this we can use `some` and `none` . To do this we have to include `simple_match/some_none.hpp`

Let us use the same `point` as before

```cpp
point* pp = new point(0,0);

match(pp,
	some(), [](point& p){std::cout << p.x << " is the x-value";}
	none(), [](){std::cout << "Null pointer\n";}
);
```

Notice how `some()` converted the pointer to a reference and passed it to us.

Now, that is now how we should allocate memory with a naked new. We would probably use a `std::unique_ptr`. `some` has built in support for `unique_ptr` and `shared_ptr`. So we can write it like this.

```cpp
auto pp = std::make_unique<point>(0,0);

match(pp,
	some(), [](point& p){std::cout << p.x << " is the x-value";}
	none(), [](){std::cout << "Null pointer\n";}
);
```
Notice, how our match code did not change.

We can do better because `some` composes. Since we specialized `tuple_adapter` we can use `ds` with `point`.

```cpp
auto pp = std::make_unique<point>(0,0);

match(pp,
	some(ds(0,0)), [](){std::cout << "Point is at the origin";},
	some(ds(0,_)), [](){std::cout << "Point is on the horizontal axis";},
	some(ds(_,0)), [](){std::cout << "Point is on the vertical axis";}.
	some(ds(_x < 10,_)), [](int x){std::cout << x << " is less than 10";},
	some(ds(_x,_x)), [](int x, int y){ std::cout << x << "," << y << " Is not on an axis";},
	none(), [](){std::cout << "Null pointer";}
);
```
Notice how `some` and `ds` compose. If we wanted to to, we could have pointers in tuples, and tuples in pointers and it would just work.

`some` can also use RTTI to do downcasting.

Here is an example. We will now make `point` a base class and have point2d, and point3d as subclasses, and adapt them.

```cpp
struct point{
	virtual ~point(){}
};

struct point2d:point{
	int x;
	int y;
	point2d(int x_,int y_):x(x_),y(y_){}
};

struct point3d:point{
	int x;
	int y;
	int z;
	point3d(int x_,int y_, int z_):x(x_),y(y_),z(z_){}
};

// Adapting point2d and point3d to be used with ds
namespace simple_match {
	namespace customization {
		template<>
		struct tuple_adapter<point2d>{

			enum { tuple_len = 2 };

			template<size_t I, class T>
			static decltype(auto) get(T&& t) {
				return std::get<I>(std::tie(t.x,t.y));
			}
		};
		template<>
		struct tuple_adapter<point3d>{

			enum { tuple_len = 3 };

			template<size_t I, class T>
			static decltype(auto) get(T&& t) {
				return std::get<I>(std::tie(t.x,t.y,t.z));
			}
		};
	}
}
```

Then we can use it like this

```cpp
std::unique_ptr<point> pp(new point2d(0,0));

match(pp,
	some<point2d>(ds(_x,_x)), [](int x, int y){std::cout << x << "," << y;},
	some<point3d>(ds(_x,_x,_x)), [](int x, int y, int z){std::cout << x << "," << y << "," << z;},
	some(), [](point& p){std::cout << "Unknown point type\n"},
	none(), [](){std::cout << "Null pointer\n"}
);

```

Notice how we can safely downcast, and use `ds` to destructure the `point`. Everything composes nicely.

# Implementation Details

simple_match actually was easier to implement than I thought it would be. I used the apply sample implementation from http://isocpp.org/files/papers/N3915.pdf to call a function with a tuple as arguments.

Here is the core of the implementation

```cpp
template<class T, class U>
bool match_check(T&& t, U&& u) {
	using namespace customization;
	using m = matcher<std::decay_t<T>, std::decay_t<U>>;
	return m::check(std::forward<T>(t), std::forward<U>(u));
}


template<class T, class U>
auto match_get(T&& t, U&& u) {
	using namespace customization;
	using m = matcher<std::decay_t<T>, std::decay_t<U>>;
	return m::get(std::forward<T>(t), std::forward<U>(u));
}

template<class T, class A1, class F1>
auto match(T&& t, A1&& a, F1&& f) {
	if (match_check(std::forward<T>(t), std::forward<A1>(a))) {
		return detail::apply(f, match_get(std::forward<T>(t), std::forward<A1>(a)));
	}
	else {
		throw std::logic_error("No match");
	}
}


template<class T, class A1, class F1, class A2, class F2, class... Args>
auto match(T&& t, A1&& a, F1&& f, A2&& a2, F2&& f2, Args&&... args) {
	if (match_check(t, a)) {
		return detail::apply(f, match_get(std::forward<T>(t), std::forward<A1>(a)));
	}
	else {
		return match(t, std::forward<A2>(a2), std::forward<F2>(f2), std::forward<Args>(args)...);
	}
}
```

`match` is a variadic function that takes the value to be matched, and then parameters for the match criteria, and lambda to be executed if the criteria succeeds. It goes through calling `match_check` until it returns true. Then it calls `match_get` to get a tuple of the values that need to be forwarded to the lambda, and uses apply to call the lambda.

The match types are implemented by specializing `simple_match::customization::matcher`
```cpp
namespace customization {
	template<class T, class U>
	struct matcher;
}
```

For an example, here is how the matcher that matches to values is implemented. Note, that it does not pass any values to the lambda and so returns an empty tuple.

```cpp
// Match same type
template<class T>
struct matcher<T, T> {
	static bool check(const T& t, const T& v) {
		return t == v;
	}
	static auto get(const T&, const T&) {
		return std::tie();
	}
}
```

I hope you enjoy using this code, as much as I have enjoyed writing it. Please give me any feedback you may have.

-- John R. Bandela, MD

> Written with [StackEdit](https://stackedit.io/).
