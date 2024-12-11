#pragma once

// clang-format off

#define ERR_FAIL_INDEX_D(m_index, m_size) ERR_FAIL_INDEX_V(m_index, m_size, {})
#define ERR_FAIL_INDEX_D_MSG(m_index, m_size, m_msg) ERR_FAIL_INDEX_V_MSG(m_index, m_size, {}, m_msg)
#define ERR_FAIL_UNSIGNED_INDEX_D(m_index, m_size) ERR_FAIL_UNSIGNED_INDEX_V(m_index, m_size, {})
#define ERR_FAIL_UNSIGNED_INDEX_D_MSG(m_index, m_size, m_msg) ERR_FAIL_UNSIGNED_INDEX_V_MSG(m_index, m_size, {}, m_msg)
#define ERR_FAIL_NULL_D(m_param) ERR_FAIL_NULL_V(m_param, {})
#define ERR_FAIL_NULL_D_MSG(m_param, m_msg) ERR_FAIL_NULL_V_MSG(m_param, {}, m_msg)
#define ERR_FAIL_COND_D(m_cond) ERR_FAIL_COND_V(m_cond, {})
#define ERR_FAIL_COND_D_MSG(m_cond, m_msg) ERR_FAIL_COND_V_MSG(m_cond, {}, m_msg)
#define ERR_FAIL_D() ERR_FAIL_V({})
#define ERR_FAIL_D_MSG(m_msg) ERR_FAIL_V_MSG({}, m_msg)

#define GDJ_MSG_NOT_IMPL vformat("%s is not implemented in Godot Jolt.", __FUNCTION__)
#define ERR_FAIL_NOT_IMPL() ERR_FAIL_MSG(GDJ_MSG_NOT_IMPL)
#define ERR_FAIL_V_NOT_IMPL(m_retval) ERR_FAIL_V_MSG(m_retval, GDJ_MSG_NOT_IMPL)
#define ERR_FAIL_D_NOT_IMPL() ERR_FAIL_D_MSG(GDJ_MSG_NOT_IMPL)
#define ERR_BREAK_NOT_IMPL(m_cond) ERR_BREAK_MSG(m_cond, GDJ_MSG_NOT_IMPL)
#define ERR_CONTINUE_NOT_IMPL(m_cond) ERR_CONTINUE_MSG(m_cond, GDJ_MSG_NOT_IMPL)

#define GDJ_FMT_REPORT(m_msg) vformat("%s This should not happen under normal circumstances. Consider reporting this issue in the Godot Jolt repository on GitHub.", m_msg)
#define ERR_FAIL_INDEX_REPORT(m_index, m_size, m_msg) ERR_FAIL_INDEX_MSG(m_index, m_size, GDJ_FMT_REPORT(m_msg))
#define ERR_FAIL_INDEX_D_REPORT(m_index, m_size, m_msg) ERR_FAIL_INDEX_D_MSG(m_index, m_size, GDJ_FMT_REPORT(m_msg))
#define ERR_FAIL_INDEX_V_REPORT(m_index, m_size, m_retval, m_msg) ERR_FAIL_INDEX_V_MSG(m_index, m_size, m_retval, GDJ_FMT_REPORT(m_msg))
#define ERR_FAIL_UNSIGNED_INDEX_REPORT(m_index, m_size, m_msg) ERR_FAIL_UNSIGNED_INDEX_MSG(m_index, m_size, GDJ_FMT_REPORT(m_msg))
#define ERR_FAIL_UNSIGNED_INDEX_D_REPORT(m_index, m_size, m_msg) ERR_FAIL_UNSIGNED_INDEX_D_MSG(m_index, m_size, GDJ_FMT_REPORT(m_msg))
#define ERR_FAIL_UNSIGNED_INDEX_V_REPORT(m_index, m_size, m_retval, m_msg) ERR_FAIL_UNSIGNED_INDEX_V_MSG(m_index, m_size, m_retval, GDJ_FMT_REPORT(m_msg))
#define ERR_BREAK_REPORT(m_cond, m_msg) ERR_BREAK_MSG(m_cond, GDJ_FMT_REPORT(m_msg))
#define ERR_CONTINUE_REPORT(m_cond, m_msg) ERR_CONTINUE_MSG(m_cond, GDJ_FMT_REPORT(m_msg))
#define ERR_FAIL_REPORT(m_msg) ERR_FAIL_MSG(GDJ_FMT_REPORT(m_msg))
#define ERR_FAIL_COND_REPORT(m_cond, m_msg) ERR_FAIL_COND_MSG(m_cond, GDJ_FMT_REPORT(m_msg))
#define ERR_FAIL_COND_D_REPORT(m_cond, m_msg) ERR_FAIL_COND_D_MSG(m_cond, GDJ_FMT_REPORT(m_msg))
#define ERR_FAIL_COND_V_REPORT(m_cond, m_retval, m_msg) ERR_FAIL_COND_V_MSG(m_cond, m_retval, GDJ_FMT_REPORT(m_msg))
#define ERR_FAIL_D_REPORT(m_msg) ERR_FAIL_D_MSG(GDJ_FMT_REPORT(m_msg))
#define ERR_FAIL_NULL_REPORT(m_param, m_msg) ERR_FAIL_NULL_MSG(m_param, GDJ_FMT_REPORT(m_msg))
#define ERR_FAIL_NULL_D_REPORT(m_param, m_msg) ERR_FAIL_NULL_D_MSG(m_param, GDJ_FMT_REPORT(m_msg))
#define ERR_FAIL_NULL_V_REPORT(m_param, m_retval, m_msg) ERR_FAIL_NULL_V_MSG(m_param, m_retval, GDJ_FMT_REPORT(m_msg))
#define ERR_FAIL_V_REPORT(m_retval, m_msg) ERR_FAIL_V_MSG(m_retval, GDJ_FMT_REPORT(m_msg))
#define CRASH_BAD_INDEX_REPORT(m_index, m_size, m_msg) CRASH_BAD_INDEX_MSG(m_index, m_size, GDJ_FMT_REPORT(m_msg))
#define CRASH_BAD_UNSIGNED_INDEX_REPORT(m_index, m_size, m_msg) CRASH_BAD_UNSIGNED_INDEX_MSG(m_index, m_size, GDJ_FMT_REPORT(m_msg))
#define CRASH_COND_REPORT(m_cond, m_msg) CRASH_COND_MSG(m_cond, GDJ_FMT_REPORT(m_msg))
#define CRASH_NOW_REPORT(m_msg) CRASH_NOW_MSG(GDJ_FMT_REPORT(m_msg))

// clang-format on

#define QUIET_FAIL_COND(m_cond) \
	if (unlikely(m_cond)) {     \
		return;                 \
	} else                      \
		((void)0)

#define QUIET_FAIL_COND_V(m_cond, m_retval) \
	if (unlikely(m_cond)) {                 \
		return m_retval;                    \
	} else                                  \
		((void)0)

#define QUIET_BREAK(m_cond) \
	if (unlikely(m_cond)) { \
		break;              \
	} else                  \
		((void)0)

#define QUIET_CONTINUE(m_cond) \
	if (unlikely(m_cond)) {    \
		continue;              \
	} else                     \
		((void)0)

// clang-format off

#define QUIET_FAIL_COND_D(m_cond) QUIET_FAIL_COND_V(m_cond, {})
#define QUIET_FAIL_NULL(m_param) QUIET_FAIL_COND((m_param) == nullptr)
#define QUIET_FAIL_NULL_V(m_param, m_retval) QUIET_FAIL_COND_V((m_param) == nullptr, m_retval)
#define QUIET_FAIL_NULL_D(m_param) QUIET_FAIL_NULL_V(m_param, {})
#define QUIET_FAIL_INDEX(m_index, m_size) QUIET_FAIL_COND((m_index) < 0 || (m_index) >= (m_size))
#define QUIET_FAIL_INDEX_V(m_index, m_size, m_retval) QUIET_FAIL_COND_V((m_index) < 0 || (m_index) >= (m_size), m_retval)
#define QUIET_FAIL_INDEX_D(m_index, m_size) QUIET_FAIL_INDEX_V(m_index, m_size, {})
#define QUIET_FAIL_UNSIGNED_INDEX(m_index, m_size) QUIET_FAIL_COND((m_index) >= (m_size))
#define QUIET_FAIL_UNSIGNED_INDEX_V(m_index, m_size, m_retval) QUIET_FAIL_COND_V((m_index) >= (m_size), m_retval)
#define QUIET_FAIL_UNSIGNED_INDEX_D(m_index, m_size) QUIET_FAIL_UNSIGNED_INDEX_V(m_index, m_size, {})

#ifdef TOOLS_ENABLED

#define QUIET_FAIL_COND_D_ED(m_cond, m_retval) QUIET_FAIL_COND_D(m_cond, m_retval)
#define QUIET_FAIL_NULL_ED(m_param) QUIET_FAIL_NULL(m_param)
#define QUIET_FAIL_NULL_V_ED(m_param, m_retval) QUIET_FAIL_NULL_V(m_param, m_retval)
#define QUIET_FAIL_NULL_D_ED(m_param) QUIET_FAIL_NULL_D(m_param)
#define QUIET_FAIL_INDEX_ED(m_index, m_size) QUIET_FAIL_INDEX(m_index, m_size)
#define QUIET_FAIL_INDEX_V_ED(m_index, m_size, m_retval) QUIET_FAIL_INDEX_V(m_index, m_size, m_retval)
#define QUIET_FAIL_INDEX_D_ED(m_index, m_size) QUIET_FAIL_INDEX_D(m_index, m_size)
#define QUIET_FAIL_UNSIGNED_INDEX_ED(m_index, m_size) QUIET_FAIL_UNSIGNED_INDEX(m_index, m_size)
#define QUIET_FAIL_UNSIGNED_INDEX_V_ED(m_index, m_size, m_retval) QUIET_FAIL_UNSIGNED_INDEX_V(m_index, m_size, m_retval)
#define QUIET_FAIL_UNSIGNED_INDEX_D_ED(m_index, m_size) QUIET_FAIL_UNSIGNED_INDEX_D(m_index, m_size)
#define QUIET_BREAK_ED(m_cond) QUIET_BREAK(m_cond)
#define QUIET_CONTINUE_ED(m_cond) QUIET_CONTINUE(m_cond)

#else // TOOLS_ENABLED

#define QUIET_FAIL_COND_D_ED(m_cond, m_retval)
#define QUIET_FAIL_NULL_ED(m_param)
#define QUIET_FAIL_NULL_V_ED(m_param, m_retval)
#define QUIET_FAIL_NULL_D_ED(m_param)
#define QUIET_FAIL_INDEX_ED(m_index, m_size)
#define QUIET_FAIL_INDEX_V_ED(m_index, m_size, m_retval)
#define QUIET_FAIL_INDEX_D_ED(m_index, m_size)
#define QUIET_FAIL_UNSIGNED_INDEX_ED(m_index, m_size)
#define QUIET_FAIL_UNSIGNED_INDEX_V_ED(m_index, m_size, m_retval)
#define QUIET_FAIL_UNSIGNED_INDEX_D_ED(m_index, m_size)
#define QUIET_BREAK_ED(m_cond)
#define QUIET_CONTINUE_ED(m_cond)

#endif // TOOLS_ENABLED
