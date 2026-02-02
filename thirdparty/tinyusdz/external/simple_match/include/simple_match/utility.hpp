#pragma once
// Copyright 2015 John R. Bandela
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#ifndef SIMPLE_MATCH_UTILITY_HPP_JRB_2015_09_11
#define SIMPLE_MATCH_UTILITY_HPP_JRB_2015_09_11
#include "simple_match.hpp"
namespace simple_match {
	// tagged_tuple

	template<class Type, class... Args>
	struct tagged_tuple :std::tuple<Args...> {
		using base = std::tuple<Args...>;
		template<class... A>
		tagged_tuple(A&&... a) :base{ std::forward<A>(a)... } {}
	};
	
	// inheriting_tagged_tuple

	template<class Base, class Type, class... Args>
	struct inheriting_tagged_tuple :Base,tagged_tuple<Type,Args...> {
		using base = tagged_tuple<Type,Args...>;
		template<class... A>
		inheriting_tagged_tuple(A&&... a) :base{ std::forward<A>(a)... } {}
	};


}


#endif
