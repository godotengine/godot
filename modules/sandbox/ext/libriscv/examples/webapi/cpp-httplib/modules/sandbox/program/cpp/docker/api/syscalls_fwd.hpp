#pragma once
#include <cstddef>
#include <cstdint>
#include <string_view>
struct Object;
struct Variant;
#ifdef DOUBLE_PRECISION_REAL_T
using real_t = double;
#else
using real_t = float;
#endif

#define PUBLIC extern "C" __attribute__((used, retain))

#define UNLIKELY(x) __builtin_expect(!!(x), 0)

#define EXTERN_SYSCALL(rval, name, ...) \
	extern "C" rval name(__VA_ARGS__);

EXTERN_SYSCALL(void, sys_print, const Variant *, size_t);
EXTERN_SYSCALL(void, sys_throw, const char *, size_t, const char *, size_t, ...);
EXTERN_SYSCALL(unsigned, sys_callable_create, void (*)(), const Variant *, const void *, size_t);
EXTERN_SYSCALL(void, sys_sandbox_add, int, ...);

inline __attribute__((noreturn)) void api_throw_at(std::string_view type, std::string_view msg, const Variant *srcVar, const char *func) {
	sys_throw(type.data(), type.size(), msg.data(), msg.size(), srcVar, func);
	__builtin_unreachable();
}
#define API_THROW(type, msg, srcVar) api_throw_at(type, msg, srcVar, __FUNCTION__)

inline __attribute__((noreturn)) void api_throw(std::string_view type, std::string_view msg, const Variant *srcVar = nullptr) {
	sys_throw(type.data(), type.size(), msg.data(), msg.size(), srcVar, nullptr);
	__builtin_unreachable();
}
#define EXPECT(cond, msg) if (UNLIKELY(!(cond))) { api_throw(__FUNCTION__, msg); }

extern "C" __attribute__((noreturn)) void fast_exit();

// Helper method to call a method on any type that can be wrapped in a Variant
#define VMETHOD(name) \
	template <typename... Args> \
	inline Variant name(Args&&... args) { \
		return operator() (#name, std::forward<Args>(args)...); \
	}

#define METHOD(Type, name) \
	template <typename... Args> \
	inline Type name(Args&&... args) { \
		if constexpr (std::is_same_v<Type, void>) { \
			voidcall(#name, std::forward<Args>(args)...); \
		} else { \
			return operator() (#name, std::forward<Args>(args)...); \
		} \
	}

// Helpers for static method calls
#define SMETHOD(Type, name) METHOD(Type, name)
#define SVMETHOD(name) VMETHOD(name)
