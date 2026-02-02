// Copyright 2015 John R. Bandela
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#pragma once
#ifndef SIMPLE_MATCH_SOME_NONE_HPP_JRB_2015_03_21
#define SIMPLE_MATCH_SOME_NONE_HPP_JRB_2015_03_21
#include <memory>
#include <type_traits>

namespace simple_match {

	namespace customization {

		template<class Type>
		struct pointer_getter {};


	}

	namespace detail {

		template<class T, class C>
		struct cv_helper_imp;

		template<class T, class C>
		struct cv_helper_imp<T*, C> {
			using type = std::add_pointer_t<C>;
		};

		template<class T, class C>
		struct cv_helper_imp<const T*, C> {
			using type = std::add_pointer_t<std::add_const_t<C>>;
		};

		template<class T, class C>
		struct cv_helper_imp<volatile T*, C> {
			using type = std::add_pointer_t<std::add_volatile_t<C>>;
		};

		template<class T, class C>
		struct cv_helper_imp<const volatile T*, C> {
			using type = std::add_pointer_t<std::add_cv_t<C>>;
		};

	}
	namespace utils {
		template<class T, class C>
		using cv_helper = typename detail::cv_helper_imp<T, C>::type;

	}

	namespace detail{


		template<class Class, class Matcher>
		struct some_t{
			Matcher m_;

			template<class T>
			bool check(T&& t) {
				auto ptr = customization::pointer_getter<std::decay_t<T>>::template get_pointer<Class>(std::forward<T>(t));
				if (!ptr) {
					return false;
				}
				return match_check(*ptr, m_);

			}

			template<class T>
			auto get(T&& t) {
				auto ptr = customization::pointer_getter<std::decay_t<T>>::template get_pointer<Class>(std::forward<T>(t));
				return match_get(*ptr, m_);
			}


		};

		template<class Matcher>
		struct some_t<void,Matcher> {
			Matcher m_;

			template<class T>
			bool check(T&& t) {
				// If you get an error here, this means that some() without a type is not supported
				// Examples of this are variants and boost::any
				auto ptr = customization::pointer_getter<std::decay_t<T>>::get_pointer_no_cast(std::forward<T>(t));
				if (!ptr) {
					return false;
				}
				return match_check(*ptr, m_);

			}

			template<class T>
			auto get(T&& t) {
				auto ptr = (customization::pointer_getter<std::decay_t<T>>::get_pointer_no_cast(std::forward<T>(t)));
				return match_get(*ptr, m_);
			}


		};

		template<class Class>
		struct some_t<Class,void> {

			template<class T>
			bool check(T&& t) {
				auto ptr = customization::pointer_getter<std::decay_t<T>>::template get_pointer<Class>(std::forward<T>(t));
				if (!ptr) {
					return false;
				}
				return true;

			}

			template<class T>
			auto get(T&& t) {
				auto ptr = customization::pointer_getter<std::decay_t<T>>::template get_pointer<Class>(std::forward<T>(t));
				return std::tie(*ptr);
			}


		};

		template<>
		struct some_t<void, void> {

			template<class T>
			bool check(T&& t) {
				// If you get an error here, this means that some() without a type is not supported
				// Examples of this are variants and boost::any
				auto ptr = customization::pointer_getter<std::decay_t<T>>::get_pointer_no_cast(std::forward<T>(t));
				if (!ptr) {
					return false;
				}
				return true;

			}

			template<class T>
			auto get(T&& t) {
				auto ptr = customization::pointer_getter<std::decay_t<T>>::get_pointer_no_cast(std::forward<T>(t));
				return std::tie(*ptr);
			}


		};


		struct none_t{
			template<class T>
			bool check(T&& t) {
				// If you get an error here, this means that none() is not supported
				// Example is boost::variant which has a never empty guarantee
				return customization::pointer_getter<std::decay_t<T>>::is_null(std::forward<T>(t));
			}

			template<class T>
			auto get(T&&) {
				return std::tie();
			}
		};

	}
	namespace customization {

		template<class Type, class Class, class Matcher>
		struct matcher<Type,detail::some_t<Class,Matcher>> {
			template<class T, class U>
			static bool check(T&& t, U&& u) {
				return u.check(std::forward<T>(t));
			}
			template<class T, class U>
			static auto get(T&& t, U&& u) {
				return u.get(std::forward<T>(t));
			}

		};

		template<class Type>
		struct matcher<Type,detail::none_t> {
			template<class T, class U>
			static bool check(T&& t, U&& u) {
				return u.check(std::forward<T>(t));
			}
			template<class T, class U>
			static auto get(T&& t, U&& u) {
				return u.get(std::forward<T>(t));
			}

		};
	}
	namespace customization {
		template<class Type>
		struct pointer_getter<Type*> {
			template<class To>
			static auto get_pointer(Type* t) {
				return dynamic_cast<utils::cv_helper<decltype(t),To>>(t);
			}
			static auto get_pointer_no_cast(Type* t) {
				return t;
			}
			static auto is_null(Type* t) {
				return !t;
			}
		};


		template<class Type>
		struct pointer_getter<std::shared_ptr<Type>> {

			template<class To, class T>
			static auto get_pointer(T&& t) {
				return dynamic_cast<utils::cv_helper<decltype(t.get()),To>>(t.get());
			}
			template<class T>
			static auto get_pointer_no_cast(T&& t) {
				return t.get();
			}
			template<class T>
			static auto is_null(T&& t) {
				return !t;
			}
		};

		template<class Type, class D>
		struct pointer_getter<std::unique_ptr<Type, D>> {
			template<class To, class T>
			static auto get_pointer(T&& t) {
				return dynamic_cast<utils::cv_helper<decltype(t.get()),To>>(t.get());
			}
			template<class T>
			static auto get_pointer_no_cast(T&& t) {
				return t.get();
			}
			template<class T>
			static auto is_null(T&& t) {
				return !t;
			}
		};

	}


	inline detail::none_t none() { return detail::none_t{}; }

	inline detail::some_t<void, void> some() { return detail::some_t<void, void>{}; }

    // Make Matcher... a variadic template so it is a worse match than some<T>(T())
	template<class... Matcher>
	detail::some_t<void, Matcher...> some(Matcher&&... m) { return detail::some_t<void, Matcher...> { std::forward<Matcher>(m)... }; }

	template<class Class, class Matcher>
	detail::some_t<Class, Matcher> some(Matcher&& m) {
		return detail::some_t<Class, Matcher> { std::forward<Matcher>(m) };
	}
	template<class Class>
	detail::some_t<Class, void> some() {
		return detail::some_t<Class, void>{ };
	}


	// exhaustiveness

	namespace detail {

		template<class T, class Tuple>
		struct type_in_tuple {
			static const bool value = false;
		};
		template<class T, class First, class... Rest>
		struct type_in_tuple<T, std::tuple<First, Rest...>> {
			static const bool value = type_in_tuple<T, std::tuple<Rest...>>::value;
		};
		template<class T, class... Rest>
		struct type_in_tuple<T, std::tuple<T, Rest...>> {
			static const bool value = true;
		};
		template<class T>
		struct type_in_tuple<T, std::tuple<>> {
			static const bool value = false;
		};

		template<class ArgTypes>
		struct has_otherwise {
			static const bool value = type_in_tuple<otherwise_t, ArgTypes>::value;
		};

		template<bool, class ArgTypes,class... RequiredTypes>
		struct some_exhaustiveness_helper {
			static const bool value = true;
		};

		template<class Tuple>
		struct get_some_classes {};

		template<>
		struct get_some_classes<std::tuple<>> {
			using type = std::tuple<>;
		};

		template<class First, class...Rest>
		struct get_some_classes<std::tuple<First, Rest...>> {
			using type = typename get_some_classes<std::tuple<Rest...>>::type;

		};
		template<class Class, class Matcher, class...Rest>
		struct get_some_classes<std::tuple<some_t<Class,Matcher>, Rest...>> {
			using type = cat_tuple_t<std::tuple<Class>, typename get_some_classes<std::tuple<Rest...>>::type>;

		};

		template<class SomeClasses, class... Required>
		struct all_in {};

		template<bool b, class Type>
		struct not_in_match_asserter {
			static const bool value = b;
			static_assert(value, "This type is not in the match");
		};

		template<class SomeClasses, class First, class... Rest>
		struct all_in<SomeClasses,First,Rest...> {
			static const bool myvalue = type_in_tuple<First, SomeClasses>::value;
			static const bool value = not_in_match_asserter<myvalue, First>::value&& all_in<SomeClasses, Rest...>::value;

		};
		template<class SomeClasses, class First>
		struct all_in<SomeClasses,First> {
			static const bool myvalue = type_in_tuple<First, SomeClasses>::value;
//			static_assert(type_in_tuple<First, SomeClasses>::value, "This classes is not in the match");
			static const bool value = not_in_match_asserter<myvalue, First>::value;

		};



		template<class ArgTypes, class... RequiredTypes>
		struct some_exhaustiveness_helper<false,ArgTypes,RequiredTypes...> {
			using some_classes = typename get_some_classes<ArgTypes>::type;
			using a = all_in<some_classes, RequiredTypes...>;
			static const bool value = a::value;
		
		
		};






	}

	template<class... Types>
	struct some_exhaustiveness {
		template<class ArgTypes>
		struct type {
			static const bool v = detail::has_otherwise<ArgTypes>::value;
			using seh = detail::some_exhaustiveness_helper<v, ArgTypes, Types...>;
			static const bool value = seh::value;
		};
	};
}



#endif
