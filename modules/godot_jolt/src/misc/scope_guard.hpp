#pragma once

#define GDJ_CONCATENATE_IMPL(m_a, m_b) m_a##m_b
#define GDJ_CONCATENATE(m_a, m_b) GDJ_CONCATENATE_IMPL(m_a, m_b)
#define GDJ_UNIQUE_IDENTIFIER(m_prefix) GDJ_CONCATENATE(m_prefix, __COUNTER__)

template<typename TCallable>
class ScopeGuard {
public:
	// NOLINTNEXTLINE(hicpp-explicit-conversions)
	ScopeGuard(TCallable p_callable)
		: callable(std::move(p_callable)) { }

	ScopeGuard(const ScopeGuard& p_other) = delete;

	ScopeGuard(ScopeGuard&& p_other) = delete;

	~ScopeGuard() {
		if (!released) {
			callable();
		}
	}

	void release() { released = true; }

	ScopeGuard& operator=(const ScopeGuard& p_other) = delete;

	ScopeGuard& operator=(ScopeGuard&& p_other) = delete;

private:
	TCallable callable;

	bool released = false;
};

struct ScopeGuardHelper {
	template<typename TCallable>
	ScopeGuard<std::decay_t<TCallable>> operator+(TCallable&& p_callable) {
		return {std::forward<TCallable>(p_callable)};
	}
};

// NOLINTNEXTLINE(bugprone-macro-parentheses)
#define ON_SCOPE_EXIT auto GDJ_UNIQUE_IDENTIFIER(scope_guard) = ScopeGuardHelper() + [&]
