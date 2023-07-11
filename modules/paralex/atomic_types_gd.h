#ifndef SAFE_NUMERIC_GD_H
#define SAFE_NUMERIC_GD_H

#include "core/safe_refcount.h"
#include "core/reference.h"

#define REPLACER_0A(m_type) \
_ALWAYS_INLINE_ void m_type() { reciever.m_type(); }

#define REPLACER_0A_C(m_type) \
_ALWAYS_INLINE_ void m_type() const { reciever.m_type(); }

#define REPLACER_0A_R(m_r, m_type) \
_ALWAYS_INLINE_ m_r m_type() { return reciever.m_type(); }

#define REPLACER_0A_RC(m_r, m_type) \
_ALWAYS_INLINE_ m_r m_type() const { return reciever.m_type(); }

#define REPLACER_1A(m_type, p_arg0) \
_ALWAYS_INLINE_ void m_type(p_arg0 m_arg0) { reciever.m_type(m_arg0); }

#define REPLACER_1A_C(m_type, p_arg0) \
_ALWAYS_INLINE_ void m_type(p_arg0 m_arg0) const { reciever.m_type(m_arg0); }

#define REPLACER_1A_R(m_r, m_type, p_arg0) \
_ALWAYS_INLINE_ m_r m_type(p_arg0 m_arg0) { return reciever.m_type(m_arg0); }

#define REPLACER_1A_RC(m_r, m_type, p_arg0) \
_ALWAYS_INLINE_ m_r m_type(p_arg0 m_arg0) const { return reciever.m_type(m_arg0); }

class _SafeNumeric : public Reference {
	GDCLASS(_SafeNumeric, Reference);
private:
	SafeNumeric<uint64_t> reciever;
protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_value", "value"), &_SafeNumeric::set);
		ClassDB::bind_method(D_METHOD("get_value"), &_SafeNumeric::get);

		ClassDB::bind_method(D_METHOD("increment"), &_SafeNumeric::increment);
		ClassDB::bind_method(D_METHOD("postincrement"), &_SafeNumeric::postincrement);
		ClassDB::bind_method(D_METHOD("decrement"), &_SafeNumeric::decrement);
		ClassDB::bind_method(D_METHOD("postdecrement"), &_SafeNumeric::postdecrement);

		ClassDB::bind_method(D_METHOD("add", "value"), &_SafeNumeric::add);
		ClassDB::bind_method(D_METHOD("postadd", "value"), &_SafeNumeric::postadd);
		ClassDB::bind_method(D_METHOD("sub", "value"), &_SafeNumeric::sub);
		ClassDB::bind_method(D_METHOD("postsub", "value"), &_SafeNumeric::postsub);

		ClassDB::bind_method(D_METHOD("exchange_if_greater", "value"), &_SafeNumeric::exchange_if_greater);
		ClassDB::bind_method(D_METHOD("conditional_increment"), &_SafeNumeric::conditional_increment);
	}
public:
	_SafeNumeric()  = default;
	~_SafeNumeric() = default;

	REPLACER_1A(set, uint64_t);
	REPLACER_0A_RC(uint64_t, get);

	REPLACER_0A_R(uint64_t, increment);
	REPLACER_0A_R(uint64_t, postincrement);
	REPLACER_0A_R(uint64_t, decrement);
	REPLACER_0A_R(uint64_t, postdecrement);

	REPLACER_1A_R(uint64_t, add, uint64_t);
	REPLACER_1A_R(uint64_t, postadd, uint64_t);
	REPLACER_1A_R(uint64_t, sub, uint64_t);
	REPLACER_1A_R(uint64_t, postsub, uint64_t);

	REPLACER_1A_R(uint64_t, exchange_if_greater, uint64_t);
	REPLACER_0A_R(uint64_t, conditional_increment);
};

class _SafeBoolean : public Reference {
	GDCLASS(_SafeBoolean, Reference);
private:
	SafeFlag reciever;
protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_to", "value"), &_SafeBoolean::set_to);
		ClassDB::bind_method(D_METHOD("is_set"), &_SafeBoolean::is_set);
		ClassDB::bind_method(D_METHOD("set_flag"), &_SafeBoolean::set);
		ClassDB::bind_method(D_METHOD("clear_flag"), &_SafeBoolean::clear);
	}
public:
	_SafeBoolean()  = default;
	~_SafeBoolean() = default;

	REPLACER_0A_RC(bool, is_set);
	REPLACER_0A(set);
	REPLACER_0A(clear);
	REPLACER_1A(set_to, bool);
};

#endif
