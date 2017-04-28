/*************************************************************************/
/*  variant_op.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#include "variant.h"

#include "core_string_names.h"
#include "object.h"
#include "script_language.h"

Variant::operator bool() const {

	bool b;
	return booleanize(b);
}

bool Variant::booleanize(bool &r_valid) const {

	r_valid = true;
	switch (type) {
		case NIL: return false;
		case BOOL: return _data._bool;
		case INT: return _data._int;
		case REAL: return _data._real;
		case STRING: return (*reinterpret_cast<const String *>(_data._mem)) != "";
		case VECTOR2:
		case RECT2:
		case TRANSFORM2D:
		case VECTOR3:
		case PLANE:
		case RECT3:
		case QUAT:
		case BASIS:
		case TRANSFORM:
		case COLOR:
		case IMAGE: r_valid = false; return false;
		case _RID: return (*reinterpret_cast<const RID *>(_data._mem)).is_valid();
		case OBJECT: return _get_obj().obj;
		case NODE_PATH: return (*reinterpret_cast<const NodePath *>(_data._mem)) != NodePath();
		case INPUT_EVENT:
		case DICTIONARY:
		case ARRAY:
		case POOL_BYTE_ARRAY:
		case POOL_INT_ARRAY:
		case POOL_REAL_ARRAY:
		case POOL_STRING_ARRAY:
		case POOL_VECTOR2_ARRAY:
		case POOL_VECTOR3_ARRAY:
		case POOL_COLOR_ARRAY:
			r_valid = false;
			return false;
		default: {}
	}

	return false;
}

#define _RETURN(m_what) \
	{                   \
		r_ret = m_what; \
		return;         \
	}

#define DEFAULT_OP_NUM(m_op, m_name, m_type)                           \
	case m_name: {                                                     \
		switch (p_b.type) {                                            \
			case BOOL: _RETURN(p_a._data.m_type m_op p_b._data._bool); \
			case INT: _RETURN(p_a._data.m_type m_op p_b._data._int);   \
			case REAL: _RETURN(p_a._data.m_type m_op p_b._data._real); \
			default: {}                                                \
		}                                                              \
		r_valid = false;                                               \
		return;                                                        \
	};

#define DEFAULT_OP_NUM_NEG(m_name, m_type) \
	case m_name: {                         \
                                           \
		_RETURN(-p_a._data.m_type);        \
	};

#define DEFAULT_OP_NUM_POS(m_name, m_type) \
	case m_name: {                         \
                                           \
		_RETURN(p_a._data.m_type);         \
	};

#define DEFAULT_OP_NUM_VEC(m_op, m_name, m_type)                                                             \
	case m_name: {                                                                                           \
		switch (p_b.type) {                                                                                  \
			case BOOL: _RETURN(p_a._data.m_type m_op p_b._data._bool);                                       \
			case INT: _RETURN(p_a._data.m_type m_op p_b._data._int);                                         \
			case REAL: _RETURN(p_a._data.m_type m_op p_b._data._real);                                       \
			case VECTOR2: _RETURN(p_a._data.m_type m_op *reinterpret_cast<const Vector2 *>(p_b._data._mem)); \
			case VECTOR3: _RETURN(p_a._data.m_type m_op *reinterpret_cast<const Vector3 *>(p_b._data._mem)); \
			default: {}                                                                                      \
		}                                                                                                    \
		r_valid = false;                                                                                     \
		return;                                                                                              \
	};

#define DEFAULT_OP_STR(m_op, m_name, m_type)                                                                                                     \
	case m_name: {                                                                                                                               \
		switch (p_b.type) {                                                                                                                      \
			case STRING: _RETURN(*reinterpret_cast<const m_type *>(p_a._data._mem) m_op *reinterpret_cast<const String *>(p_b._data._mem));      \
			case NODE_PATH: _RETURN(*reinterpret_cast<const m_type *>(p_a._data._mem) m_op *reinterpret_cast<const NodePath *>(p_b._data._mem)); \
			default: {}                                                                                                                          \
		}                                                                                                                                        \
		r_valid = false;                                                                                                                         \
		return;                                                                                                                                  \
	};

#define DEFAULT_OP_LOCALMEM(m_op, m_name, m_type)                                                                                           \
	case m_name: {                                                                                                                          \
		switch (p_b.type) {                                                                                                                 \
			case m_name: _RETURN(*reinterpret_cast<const m_type *>(p_a._data._mem) m_op *reinterpret_cast<const m_type *>(p_b._data._mem)); \
			default: {}                                                                                                                     \
		}                                                                                                                                   \
		r_valid = false;                                                                                                                    \
		return;                                                                                                                             \
	}

#define DEFAULT_OP_LOCALMEM_NEG(m_name, m_type)                      \
	case m_name: {                                                   \
		_RETURN(-*reinterpret_cast<const m_type *>(p_a._data._mem)); \
	}

#define DEFAULT_OP_LOCALMEM_POS(m_name, m_type)                     \
	case m_name: {                                                  \
		_RETURN(*reinterpret_cast<const m_type *>(p_a._data._mem)); \
	}

#define DEFAULT_OP_LOCALMEM_NUM(m_op, m_name, m_type)                                                                                       \
	case m_name: {                                                                                                                          \
		switch (p_b.type) {                                                                                                                 \
			case m_name: _RETURN(*reinterpret_cast<const m_type *>(p_a._data._mem) m_op *reinterpret_cast<const m_type *>(p_b._data._mem)); \
			case BOOL: _RETURN(*reinterpret_cast<const m_type *>(p_a._data._mem) m_op p_b._data._bool);                                     \
			case INT: _RETURN(*reinterpret_cast<const m_type *>(p_a._data._mem) m_op p_b._data._int);                                       \
			case REAL: _RETURN(*reinterpret_cast<const m_type *>(p_a._data._mem) m_op p_b._data._real);                                     \
			default: {}                                                                                                                     \
		}                                                                                                                                   \
		r_valid = false;                                                                                                                    \
		return;                                                                                                                             \
	}

#define DEFAULT_OP_PTR(m_op, m_name, m_sub)                             \
	case m_name: {                                                      \
		switch (p_b.type) {                                             \
			case m_name: _RETURN(p_a._data.m_sub m_op p_b._data.m_sub); \
			default: {}                                                 \
		}                                                               \
		r_valid = false;                                                \
		return;                                                         \
	}

#define DEFAULT_OP_PTRREF(m_op, m_name, m_sub)                            \
	case m_name: {                                                        \
		switch (p_b.type) {                                               \
			case m_name: _RETURN(*p_a._data.m_sub m_op *p_b._data.m_sub); \
			default: {}                                                   \
		}                                                                 \
		r_valid = false;                                                  \
		return;                                                           \
	}

#define DEFAULT_OP_ARRAY_EQ(m_name, m_type) \
	DEFAULT_OP_ARRAY_OP(m_name, m_type, !=, !=, true, false, false)

#define DEFAULT_OP_ARRAY_LT(m_name, m_type) \
	DEFAULT_OP_ARRAY_OP(m_name, m_type, <, !=, false, a_len < array_b.size(), true)

#define DEFAULT_OP_ARRAY_OP(m_name, m_type, m_opa, m_opb, m_ret_def, m_ret_s, m_ret_f)                     \
	case m_name: {                                                                                         \
		if (p_a.type != p_b.type) {                                                                        \
			r_valid = false;                                                                               \
			return;                                                                                        \
		}                                                                                                  \
		const PoolVector<m_type> &array_a = *reinterpret_cast<const PoolVector<m_type> *>(p_a._data._mem); \
		const PoolVector<m_type> &array_b = *reinterpret_cast<const PoolVector<m_type> *>(p_b._data._mem); \
                                                                                                           \
		int a_len = array_a.size();                                                                        \
		if (a_len m_opa array_b.size()) {                                                                  \
			_RETURN(m_ret_s);                                                                              \
		} else {                                                                                           \
                                                                                                           \
			PoolVector<m_type>::Read ra = array_a.read();                                                  \
			PoolVector<m_type>::Read rb = array_b.read();                                                  \
                                                                                                           \
			for (int i = 0; i < a_len; i++) {                                                              \
				if (ra[i] m_opb rb[i])                                                                     \
					_RETURN(m_ret_f);                                                                      \
			}                                                                                              \
                                                                                                           \
			_RETURN(m_ret_def);                                                                            \
		}                                                                                                  \
	}

#define DEFAULT_OP_ARRAY_ADD(m_name, m_type)                                                               \
	case m_name: {                                                                                         \
		if (p_a.type != p_b.type) {                                                                        \
			r_valid = false;                                                                               \
			_RETURN(NIL);                                                                                  \
		}                                                                                                  \
		const PoolVector<m_type> &array_a = *reinterpret_cast<const PoolVector<m_type> *>(p_a._data._mem); \
		const PoolVector<m_type> &array_b = *reinterpret_cast<const PoolVector<m_type> *>(p_b._data._mem); \
		PoolVector<m_type> sum = array_a;                                                                  \
		sum.append_array(array_b);                                                                         \
		_RETURN(sum);                                                                                      \
	}

#define DEFAULT_OP_FAIL(m_name) \
	case m_name: {              \
		r_valid = false;        \
		return;                 \
	}

void Variant::evaluate(const Operator &p_op, const Variant &p_a, const Variant &p_b, Variant &r_ret, bool &r_valid) {

	r_valid = true;

	switch (p_op) {

		case OP_EQUAL: {

			if ((int(p_a.type) * int(p_b.type)) == 0) {
				//null case is an exception, one of both is null
				if (p_a.type == p_b.type) //null against null is true
					_RETURN(true);
				//only against object is allowed
				if (p_a.type == Variant::OBJECT) {
					_RETURN(p_a._get_obj().obj == NULL);
				} else if (p_b.type == Variant::OBJECT) {
					_RETURN(p_b._get_obj().obj == NULL);
				}
				//otherwise, always false
				_RETURN(false);
			}

			switch (p_a.type) {

				case NIL: {

					_RETURN(p_b.type == NIL || (p_b.type == Variant::OBJECT && !p_b._get_obj().obj));
				} break;

					DEFAULT_OP_NUM(==, BOOL, _bool);
					DEFAULT_OP_NUM(==, INT, _int);
					DEFAULT_OP_NUM(==, REAL, _real);
					DEFAULT_OP_STR(==, STRING, String);
					DEFAULT_OP_LOCALMEM(==, VECTOR2, Vector2);
					DEFAULT_OP_LOCALMEM(==, RECT2, Rect2);
					DEFAULT_OP_PTRREF(==, TRANSFORM2D, _transform2d);
					DEFAULT_OP_LOCALMEM(==, VECTOR3, Vector3);
					DEFAULT_OP_LOCALMEM(==, PLANE, Plane);
					DEFAULT_OP_LOCALMEM(==, QUAT, Quat);
					DEFAULT_OP_PTRREF(==, RECT3, _rect3);
					DEFAULT_OP_PTRREF(==, BASIS, _basis);
					DEFAULT_OP_PTRREF(==, TRANSFORM, _transform);

					DEFAULT_OP_LOCALMEM(==, COLOR, Color);
					DEFAULT_OP_PTRREF(==, IMAGE, _image);
					DEFAULT_OP_STR(==, NODE_PATH, NodePath);
					DEFAULT_OP_LOCALMEM(==, _RID, RID);
				case OBJECT: {

					if (p_b.type == OBJECT)
						_RETURN((p_a._get_obj().obj == p_b._get_obj().obj));
					if (p_b.type == NIL)
						_RETURN(!p_a._get_obj().obj);
				} break;
					DEFAULT_OP_PTRREF(==, INPUT_EVENT, _input_event);

				case DICTIONARY: {

					if (p_b.type != DICTIONARY)
						_RETURN(false);

					const Dictionary *arr_a = reinterpret_cast<const Dictionary *>(p_a._data._mem);
					const Dictionary *arr_b = reinterpret_cast<const Dictionary *>(p_b._data._mem);

					_RETURN(*arr_a == *arr_b);

				} break;
				case ARRAY: {

					if (p_b.type != ARRAY)
						_RETURN(false);

					const Array *arr_a = reinterpret_cast<const Array *>(p_a._data._mem);
					const Array *arr_b = reinterpret_cast<const Array *>(p_b._data._mem);

					int l = arr_a->size();
					if (arr_b->size() != l)
						_RETURN(false);
					for (int i = 0; i < l; i++) {
						if (!((*arr_a)[i] == (*arr_b)[i])) {
							_RETURN(false);
						}
					}

					_RETURN(true);

				} break;

					DEFAULT_OP_ARRAY_EQ(POOL_BYTE_ARRAY, uint8_t);
					DEFAULT_OP_ARRAY_EQ(POOL_INT_ARRAY, int);
					DEFAULT_OP_ARRAY_EQ(POOL_REAL_ARRAY, real_t);
					DEFAULT_OP_ARRAY_EQ(POOL_STRING_ARRAY, String);
					DEFAULT_OP_ARRAY_EQ(POOL_VECTOR2_ARRAY, Vector3);
					DEFAULT_OP_ARRAY_EQ(POOL_VECTOR3_ARRAY, Vector3);
					DEFAULT_OP_ARRAY_EQ(POOL_COLOR_ARRAY, Color);

				case VARIANT_MAX: {
					r_valid = false;
					return;

				} break;
			}
		} break;
		case OP_NOT_EQUAL: {
			Variant res;
			evaluate(OP_EQUAL, p_a, p_b, res, r_valid);
			if (!r_valid)
				return;
			if (res.type == BOOL)
				res._data._bool = !res._data._bool;
			_RETURN(res);

		} break;
		case OP_LESS: {

			switch (p_a.type) {

				DEFAULT_OP_FAIL(NIL);
				DEFAULT_OP_NUM(<, BOOL, _bool);
				DEFAULT_OP_NUM(<, INT, _int);
				DEFAULT_OP_NUM(<, REAL, _real);
				DEFAULT_OP_STR(<, STRING, String);
				DEFAULT_OP_LOCALMEM(<, VECTOR2, Vector2);
				DEFAULT_OP_FAIL(RECT2);
				DEFAULT_OP_FAIL(TRANSFORM2D);
				DEFAULT_OP_LOCALMEM(<, VECTOR3, Vector3);
				DEFAULT_OP_FAIL(PLANE);
				DEFAULT_OP_FAIL(QUAT);
				DEFAULT_OP_FAIL(RECT3);
				DEFAULT_OP_FAIL(BASIS);
				DEFAULT_OP_FAIL(TRANSFORM);

				DEFAULT_OP_FAIL(COLOR);
				DEFAULT_OP_FAIL(IMAGE);
				DEFAULT_OP_FAIL(NODE_PATH);
				DEFAULT_OP_LOCALMEM(<, _RID, RID);
				case OBJECT: {

					if (p_b.type == OBJECT)
						_RETURN((p_a._get_obj().obj < p_b._get_obj().obj));
				} break;
					DEFAULT_OP_FAIL(INPUT_EVENT);
					DEFAULT_OP_FAIL(DICTIONARY);
				case ARRAY: {

					if (p_b.type != ARRAY)
						_RETURN(false);

					const Array *arr_a = reinterpret_cast<const Array *>(p_a._data._mem);
					const Array *arr_b = reinterpret_cast<const Array *>(p_b._data._mem);

					int l = arr_a->size();
					if (arr_b->size() < l)
						_RETURN(false);
					for (int i = 0; i < l; i++) {
						if (!((*arr_a)[i] < (*arr_b)[i])) {
							_RETURN(true);
						}
					}

					_RETURN(false);

				} break;
					DEFAULT_OP_ARRAY_LT(POOL_BYTE_ARRAY, uint8_t);
					DEFAULT_OP_ARRAY_LT(POOL_INT_ARRAY, int);
					DEFAULT_OP_ARRAY_LT(POOL_REAL_ARRAY, real_t);
					DEFAULT_OP_ARRAY_LT(POOL_STRING_ARRAY, String);
					DEFAULT_OP_ARRAY_LT(POOL_VECTOR2_ARRAY, Vector3);
					DEFAULT_OP_ARRAY_LT(POOL_VECTOR3_ARRAY, Vector3);
					DEFAULT_OP_ARRAY_LT(POOL_COLOR_ARRAY, Color);
				case VARIANT_MAX: {
					r_valid = false;
					return;

				} break;
			}

		} break;
		case OP_LESS_EQUAL: {

			switch (p_a.type) {

				DEFAULT_OP_FAIL(NIL);
				DEFAULT_OP_NUM(<=, BOOL, _bool);
				DEFAULT_OP_NUM(<=, INT, _int);
				DEFAULT_OP_NUM(<=, REAL, _real);
				DEFAULT_OP_STR(<=, STRING, String);
				DEFAULT_OP_LOCALMEM(<=, VECTOR2, Vector2);
				DEFAULT_OP_FAIL(RECT2);
				DEFAULT_OP_FAIL(TRANSFORM2D);
				DEFAULT_OP_LOCALMEM(<=, VECTOR3, Vector3);
				DEFAULT_OP_FAIL(PLANE);
				DEFAULT_OP_FAIL(QUAT);
				DEFAULT_OP_FAIL(RECT3);
				DEFAULT_OP_FAIL(BASIS);
				DEFAULT_OP_FAIL(TRANSFORM);

				DEFAULT_OP_FAIL(COLOR);
				DEFAULT_OP_FAIL(IMAGE);
				DEFAULT_OP_FAIL(NODE_PATH);
				DEFAULT_OP_LOCALMEM(<=, _RID, RID);
				case OBJECT: {

					if (p_b.type == OBJECT)
						_RETURN((p_a._get_obj().obj <= p_b._get_obj().obj));
				} break;
					DEFAULT_OP_FAIL(INPUT_EVENT);
					DEFAULT_OP_FAIL(DICTIONARY);
					DEFAULT_OP_FAIL(ARRAY);
					DEFAULT_OP_FAIL(POOL_BYTE_ARRAY);
					DEFAULT_OP_FAIL(POOL_INT_ARRAY);
					DEFAULT_OP_FAIL(POOL_REAL_ARRAY);
					DEFAULT_OP_FAIL(POOL_STRING_ARRAY);
					DEFAULT_OP_FAIL(POOL_VECTOR2_ARRAY);
					DEFAULT_OP_FAIL(POOL_VECTOR3_ARRAY);
					DEFAULT_OP_FAIL(POOL_COLOR_ARRAY);
				case VARIANT_MAX: {
					r_valid = false;
					return;

				} break;
			}

		} break;
		case OP_GREATER: {

			Variant res;
			evaluate(OP_LESS, p_b, p_a, res, r_valid);
			if (!r_valid)
				return;
			_RETURN(res);

		} break;
		case OP_GREATER_EQUAL: {

			Variant res;
			evaluate(OP_LESS_EQUAL, p_b, p_a, res, r_valid);
			if (!r_valid)
				return;
			_RETURN(res);
		} break;
		//mathematic
		case OP_ADD: {
			switch (p_a.type) {

				DEFAULT_OP_FAIL(NIL);
				DEFAULT_OP_NUM(+, BOOL, _bool);
				DEFAULT_OP_NUM(+, INT, _int);
				DEFAULT_OP_NUM(+, REAL, _real);
				DEFAULT_OP_STR(+, STRING, String);
				DEFAULT_OP_LOCALMEM(+, VECTOR2, Vector2);
				DEFAULT_OP_FAIL(RECT2);
				DEFAULT_OP_FAIL(TRANSFORM2D);
				DEFAULT_OP_LOCALMEM(+, VECTOR3, Vector3);
				DEFAULT_OP_FAIL(PLANE);
				DEFAULT_OP_LOCALMEM(+, QUAT, Quat);
				DEFAULT_OP_FAIL(RECT3);
				DEFAULT_OP_FAIL(BASIS);
				DEFAULT_OP_FAIL(TRANSFORM);

				DEFAULT_OP_FAIL(COLOR);
				DEFAULT_OP_FAIL(IMAGE);
				DEFAULT_OP_FAIL(NODE_PATH);
				DEFAULT_OP_FAIL(_RID);
				DEFAULT_OP_FAIL(OBJECT);
				DEFAULT_OP_FAIL(INPUT_EVENT);
				DEFAULT_OP_FAIL(DICTIONARY);

				case ARRAY: {
					if (p_a.type != p_b.type) {
						r_valid = false;
						return;
					}
					const Array &array_a = *reinterpret_cast<const Array *>(p_a._data._mem);
					const Array &array_b = *reinterpret_cast<const Array *>(p_b._data._mem);
					Array sum;
					int asize = array_a.size();
					int bsize = array_b.size();
					sum.resize(asize + bsize);
					for (int i = 0; i < asize; i++)
						sum[i] = array_a[i];
					for (int i = 0; i < bsize; i++)
						sum[i + asize] = array_b[i];
					_RETURN(sum);
				}
					DEFAULT_OP_ARRAY_ADD(POOL_BYTE_ARRAY, uint8_t);
					DEFAULT_OP_ARRAY_ADD(POOL_INT_ARRAY, int);
					DEFAULT_OP_ARRAY_ADD(POOL_REAL_ARRAY, real_t);
					DEFAULT_OP_ARRAY_ADD(POOL_STRING_ARRAY, String);
					DEFAULT_OP_ARRAY_ADD(POOL_VECTOR2_ARRAY, Vector2);
					DEFAULT_OP_ARRAY_ADD(POOL_VECTOR3_ARRAY, Vector3);
					DEFAULT_OP_ARRAY_ADD(POOL_COLOR_ARRAY, Color);
				case VARIANT_MAX: {
					r_valid = false;
					return;

				} break;
			}
		} break;
		case OP_SUBSTRACT: {
			switch (p_a.type) {

				DEFAULT_OP_FAIL(NIL);
				DEFAULT_OP_NUM(-, BOOL, _bool);
				DEFAULT_OP_NUM(-, INT, _int);
				DEFAULT_OP_NUM(-, REAL, _real);
				DEFAULT_OP_FAIL(STRING);
				DEFAULT_OP_LOCALMEM(-, VECTOR2, Vector2);
				DEFAULT_OP_FAIL(RECT2);
				DEFAULT_OP_FAIL(TRANSFORM2D);
				DEFAULT_OP_LOCALMEM(-, VECTOR3, Vector3);
				DEFAULT_OP_FAIL(PLANE);
				DEFAULT_OP_LOCALMEM(-, QUAT, Quat);
				DEFAULT_OP_FAIL(RECT3);
				DEFAULT_OP_FAIL(BASIS);
				DEFAULT_OP_FAIL(TRANSFORM);

				DEFAULT_OP_FAIL(COLOR);
				DEFAULT_OP_FAIL(IMAGE);
				DEFAULT_OP_FAIL(NODE_PATH);
				DEFAULT_OP_FAIL(_RID);
				DEFAULT_OP_FAIL(OBJECT);
				DEFAULT_OP_FAIL(INPUT_EVENT);
				DEFAULT_OP_FAIL(DICTIONARY);
				DEFAULT_OP_FAIL(ARRAY);
				DEFAULT_OP_FAIL(POOL_BYTE_ARRAY);
				DEFAULT_OP_FAIL(POOL_INT_ARRAY);
				DEFAULT_OP_FAIL(POOL_REAL_ARRAY);
				DEFAULT_OP_FAIL(POOL_STRING_ARRAY);
				DEFAULT_OP_FAIL(POOL_VECTOR2_ARRAY);
				DEFAULT_OP_FAIL(POOL_VECTOR3_ARRAY);
				DEFAULT_OP_FAIL(POOL_COLOR_ARRAY);
				case VARIANT_MAX: {
					r_valid = false;
					return;

				} break;
			}
		} break;
		case OP_MULTIPLY: {
			switch (p_a.type) {

				DEFAULT_OP_FAIL(NIL);
				DEFAULT_OP_NUM(*, BOOL, _bool);
				DEFAULT_OP_NUM_VEC(*, INT, _int);
				DEFAULT_OP_NUM_VEC(*, REAL, _real);
				DEFAULT_OP_FAIL(STRING);
				DEFAULT_OP_LOCALMEM_NUM(*, VECTOR2, Vector2);
				DEFAULT_OP_FAIL(RECT2);
				case TRANSFORM2D: {

					if (p_b.type == TRANSFORM2D) {
						_RETURN(*p_a._data._transform2d * *p_b._data._transform2d);
					};
					if (p_b.type == VECTOR2) {
						_RETURN(p_a._data._transform2d->xform(*(const Vector2 *)p_b._data._mem));
					};
					r_valid = false;
					return;
				} break;
					DEFAULT_OP_LOCALMEM_NUM(*, VECTOR3, Vector3);
					DEFAULT_OP_FAIL(PLANE);
				case QUAT: {

					switch (p_b.type) {
						case VECTOR3: {

							_RETURN(reinterpret_cast<const Quat *>(p_a._data._mem)->xform(*(const Vector3 *)p_b._data._mem));
						} break;
						case QUAT: {

							_RETURN(*reinterpret_cast<const Quat *>(p_a._data._mem) * *reinterpret_cast<const Quat *>(p_b._data._mem));
						} break;
						case REAL: {
							_RETURN(*reinterpret_cast<const Quat *>(p_a._data._mem) * p_b._data._real);
						} break;
						default: {}
					};
					r_valid = false;
					return;
				} break;
					DEFAULT_OP_FAIL(RECT3);
				case BASIS: {

					switch (p_b.type) {
						case VECTOR3: {

							_RETURN(p_a._data._basis->xform(*(const Vector3 *)p_b._data._mem));
						};
						case BASIS: {

							_RETURN(*p_a._data._basis * *p_b._data._basis);
						};
						default: {}
					};
					r_valid = false;
					return;
				} break;
				case TRANSFORM: {

					switch (p_b.type) {
						case VECTOR3: {

							_RETURN(p_a._data._transform->xform(*(const Vector3 *)p_b._data._mem));
						};
						case TRANSFORM: {

							_RETURN(*p_a._data._transform * *p_b._data._transform);
						};
						default: {}
					};
					r_valid = false;
					return;
				} break;
					DEFAULT_OP_FAIL(COLOR);
					DEFAULT_OP_FAIL(IMAGE);
					DEFAULT_OP_FAIL(NODE_PATH);
					DEFAULT_OP_FAIL(_RID);
					DEFAULT_OP_FAIL(OBJECT);
					DEFAULT_OP_FAIL(INPUT_EVENT);
					DEFAULT_OP_FAIL(DICTIONARY);
					DEFAULT_OP_FAIL(ARRAY);
					DEFAULT_OP_FAIL(POOL_BYTE_ARRAY);
					DEFAULT_OP_FAIL(POOL_INT_ARRAY);
					DEFAULT_OP_FAIL(POOL_REAL_ARRAY);
					DEFAULT_OP_FAIL(POOL_STRING_ARRAY);
					DEFAULT_OP_FAIL(POOL_VECTOR2_ARRAY);
					DEFAULT_OP_FAIL(POOL_VECTOR3_ARRAY);
					DEFAULT_OP_FAIL(POOL_COLOR_ARRAY);
				case VARIANT_MAX: {
					r_valid = false;
					return;

				} break;
			}
		} break;
		case OP_DIVIDE: {
			switch (p_a.type) {

				DEFAULT_OP_FAIL(NIL);
				DEFAULT_OP_NUM(/, BOOL, _bool);
				case INT: {
					switch (p_b.type) {
						case BOOL: {
							int64_t b = p_b._data._bool;
							if (b == 0) {

								r_valid = false;
								_RETURN("Division By False");
							}
							_RETURN(p_a._data._int / b);

						} break;
						case INT: {
							int64_t b = p_b._data._int;
							if (b == 0) {

								r_valid = false;
								_RETURN("Division By Zero");
							}
							_RETURN(p_a._data._int / b);

						} break;
						case REAL: _RETURN(p_a._data._int / p_b._data._real);
						default: {}
					}
					r_valid = false;
					return;
				};
					DEFAULT_OP_NUM(/, REAL, _real);
					DEFAULT_OP_FAIL(STRING);
					DEFAULT_OP_LOCALMEM_NUM(/, VECTOR2, Vector2);
					DEFAULT_OP_FAIL(RECT2);
					DEFAULT_OP_FAIL(TRANSFORM2D);
					DEFAULT_OP_LOCALMEM_NUM(/, VECTOR3, Vector3);
					DEFAULT_OP_FAIL(PLANE);
				case QUAT: {
					if (p_b.type != REAL) {
						r_valid = false;
						return;
					}
					_RETURN(*reinterpret_cast<const Quat *>(p_a._data._mem) / p_b._data._real);
				} break;
					DEFAULT_OP_FAIL(RECT3);
					DEFAULT_OP_FAIL(BASIS);
					DEFAULT_OP_FAIL(TRANSFORM);

					DEFAULT_OP_FAIL(COLOR);
					DEFAULT_OP_FAIL(IMAGE);
					DEFAULT_OP_FAIL(NODE_PATH);
					DEFAULT_OP_FAIL(_RID);
					DEFAULT_OP_FAIL(OBJECT);
					DEFAULT_OP_FAIL(INPUT_EVENT);
					DEFAULT_OP_FAIL(DICTIONARY);
					DEFAULT_OP_FAIL(ARRAY);
					DEFAULT_OP_FAIL(POOL_BYTE_ARRAY);
					DEFAULT_OP_FAIL(POOL_INT_ARRAY);
					DEFAULT_OP_FAIL(POOL_REAL_ARRAY);
					DEFAULT_OP_FAIL(POOL_STRING_ARRAY);
					DEFAULT_OP_FAIL(POOL_VECTOR2_ARRAY);
					DEFAULT_OP_FAIL(POOL_VECTOR3_ARRAY);
					DEFAULT_OP_FAIL(POOL_COLOR_ARRAY);
				case VARIANT_MAX: {
					r_valid = false;
					return;

				} break;
			}

		} break;
		case OP_POSITIVE: {
			// Simple case when user defines variable as +value.
			switch (p_a.type) {

				DEFAULT_OP_FAIL(NIL);
				DEFAULT_OP_FAIL(STRING);
				DEFAULT_OP_FAIL(RECT2);
				DEFAULT_OP_FAIL(TRANSFORM2D);
				DEFAULT_OP_FAIL(RECT3);
				DEFAULT_OP_FAIL(BASIS);
				DEFAULT_OP_FAIL(TRANSFORM);
				DEFAULT_OP_NUM_POS(BOOL, _bool);
				DEFAULT_OP_NUM_POS(INT, _int);
				DEFAULT_OP_NUM_POS(REAL, _real);
				DEFAULT_OP_LOCALMEM_POS(VECTOR3, Vector3);
				DEFAULT_OP_LOCALMEM_POS(PLANE, Plane);
				DEFAULT_OP_LOCALMEM_POS(QUAT, Quat);
				DEFAULT_OP_LOCALMEM_POS(VECTOR2, Vector2);

				DEFAULT_OP_FAIL(COLOR);
				DEFAULT_OP_FAIL(IMAGE);
				DEFAULT_OP_FAIL(NODE_PATH);
				DEFAULT_OP_FAIL(_RID);
				DEFAULT_OP_FAIL(OBJECT);
				DEFAULT_OP_FAIL(INPUT_EVENT);
				DEFAULT_OP_FAIL(DICTIONARY);
				DEFAULT_OP_FAIL(ARRAY);
				DEFAULT_OP_FAIL(POOL_BYTE_ARRAY);
				DEFAULT_OP_FAIL(POOL_INT_ARRAY);
				DEFAULT_OP_FAIL(POOL_REAL_ARRAY);
				DEFAULT_OP_FAIL(POOL_STRING_ARRAY);
				DEFAULT_OP_FAIL(POOL_VECTOR2_ARRAY);
				DEFAULT_OP_FAIL(POOL_VECTOR3_ARRAY);
				DEFAULT_OP_FAIL(POOL_COLOR_ARRAY);
				case VARIANT_MAX: {
					r_valid = false;
					return;

				} break;
			}
		} break;
		case OP_NEGATE: {
			switch (p_a.type) {

				DEFAULT_OP_FAIL(NIL);
				DEFAULT_OP_NUM_NEG(BOOL, _bool);
				DEFAULT_OP_NUM_NEG(INT, _int);
				DEFAULT_OP_NUM_NEG(REAL, _real);
				DEFAULT_OP_FAIL(STRING);
				DEFAULT_OP_LOCALMEM_NEG(VECTOR2, Vector2);
				DEFAULT_OP_FAIL(RECT2);
				DEFAULT_OP_FAIL(TRANSFORM2D);
				DEFAULT_OP_LOCALMEM_NEG(VECTOR3, Vector3);
				DEFAULT_OP_LOCALMEM_NEG(PLANE, Plane);
				DEFAULT_OP_LOCALMEM_NEG(QUAT, Quat);
				DEFAULT_OP_FAIL(RECT3);
				DEFAULT_OP_FAIL(BASIS);
				DEFAULT_OP_FAIL(TRANSFORM);

				DEFAULT_OP_FAIL(COLOR);
				DEFAULT_OP_FAIL(IMAGE);
				DEFAULT_OP_FAIL(NODE_PATH);
				DEFAULT_OP_FAIL(_RID);
				DEFAULT_OP_FAIL(OBJECT);
				DEFAULT_OP_FAIL(INPUT_EVENT);
				DEFAULT_OP_FAIL(DICTIONARY);
				DEFAULT_OP_FAIL(ARRAY);
				DEFAULT_OP_FAIL(POOL_BYTE_ARRAY);
				DEFAULT_OP_FAIL(POOL_INT_ARRAY);
				DEFAULT_OP_FAIL(POOL_REAL_ARRAY);
				DEFAULT_OP_FAIL(POOL_STRING_ARRAY);
				DEFAULT_OP_FAIL(POOL_VECTOR2_ARRAY);
				DEFAULT_OP_FAIL(POOL_VECTOR3_ARRAY);
				DEFAULT_OP_FAIL(POOL_COLOR_ARRAY);
				case VARIANT_MAX: {
					r_valid = false;
					return;

				} break;
			}
		} break;
		case OP_MODULE: {
			if (p_a.type == INT && p_b.type == INT) {
#ifdef DEBUG_ENABLED
				if (p_b._data._int == 0) {
					r_valid = false;
					_RETURN("Division By Zero");
				}
#endif
				_RETURN(p_a._data._int % p_b._data._int);

			} else if (p_a.type == STRING) {
				const String *format = reinterpret_cast<const String *>(p_a._data._mem);

				String result;
				bool error;
				if (p_b.type == ARRAY) {
					// e.g. "frog %s %d" % ["fish", 12]
					const Array *args = reinterpret_cast<const Array *>(p_b._data._mem);
					result = format->sprintf(*args, &error);
				} else {
					// e.g. "frog %d" % 12
					Array args;
					args.push_back(p_b);
					result = format->sprintf(args, &error);
				}
				r_valid = !error;
				_RETURN(result);
			}

			r_valid = false;
			return;

		} break;
		case OP_STRING_CONCAT: {

			_RETURN(p_a.operator String() + p_b.operator String());
		} break;
		//bitwise
		case OP_SHIFT_LEFT: {
			if (p_a.type == INT && p_b.type == INT)
				_RETURN(p_a._data._int << p_b._data._int);

			r_valid = false;
			return;

		} break;
		case OP_SHIFT_RIGHT: {
			if (p_a.type == INT && p_b.type == INT)
				_RETURN(p_a._data._int >> p_b._data._int);

			r_valid = false;
			return;

		} break;
		case OP_BIT_AND: {
			if (p_a.type == INT && p_b.type == INT)
				_RETURN(p_a._data._int & p_b._data._int);

			r_valid = false;
			return;

		} break;
		case OP_BIT_OR: {

			if (p_a.type == INT && p_b.type == INT)
				_RETURN(p_a._data._int | p_b._data._int);

			r_valid = false;
			return;

		} break;
		case OP_BIT_XOR: {

			if (p_a.type == INT && p_b.type == INT)
				_RETURN(p_a._data._int ^ p_b._data._int);

			r_valid = false;
			return;

		} break;
		case OP_BIT_NEGATE: {

			if (p_a.type == INT)
				_RETURN(~p_a._data._int);

			r_valid = false;
			return;

		} break;
		//logic
		case OP_AND: {
			bool l = p_a.booleanize(r_valid);
			if (!r_valid)
				return;
			bool r = p_b.booleanize(r_valid);
			if (!r_valid)
				return;

			_RETURN(l && r);
		} break;
		case OP_OR: {
			bool l = p_a.booleanize(r_valid);
			if (!r_valid)
				return;
			bool r = p_b.booleanize(r_valid);
			if (!r_valid)
				return;

			_RETURN(l || r);

		} break;
		case OP_XOR: {
			bool l = p_a.booleanize(r_valid);
			if (!r_valid)
				return;
			bool r = p_b.booleanize(r_valid);
			if (!r_valid)
				return;

			_RETURN((l || r) && !(l && r));
		} break;
		case OP_NOT: {

			bool l = p_a.booleanize(r_valid);
			if (!r_valid)
				return;
			_RETURN(!l);

		} break;
		case OP_IN: {

			_RETURN(p_b.in(p_a, &r_valid));

		} break;
		case OP_MAX: {

			r_valid = false;
			ERR_FAIL();
		}
	}

	r_valid = false;
}

void Variant::set_named(const StringName &p_index, const Variant &p_value, bool *r_valid) {

	if (type == OBJECT) {

#ifdef DEBUG_ENABLED
		if (!_get_obj().obj) {
			if (r_valid)
				*r_valid = false;
			return;
		} else {

			if (ScriptDebugger::get_singleton() && _get_obj().ref.is_null() && !ObjectDB::instance_validate(_get_obj().obj)) {
				if (r_valid)
					*r_valid = false;
				return;
			}
		}

#endif
		_get_obj().obj->set(p_index, p_value, r_valid);
		return;
	}

	set(p_index.operator String(), p_value, r_valid);
}

Variant Variant::get_named(const StringName &p_index, bool *r_valid) const {

	if (type == OBJECT) {

#ifdef DEBUG_ENABLED
		if (!_get_obj().obj) {
			if (r_valid)
				*r_valid = false;
			return "Instance base is null.";
		} else {

			if (ScriptDebugger::get_singleton() && _get_obj().ref.is_null() && !ObjectDB::instance_validate(_get_obj().obj)) {
				if (r_valid)
					*r_valid = false;
				return "Attempted use of stray pointer object.";
			}
		}

#endif

		return _get_obj().obj->get(p_index, r_valid);
	}

	return get(p_index.operator String(), r_valid);
}

#define DEFAULT_OP_ARRAY_CMD(m_name, m_type, skip_test, cmd)                             \
	case m_name: {                                                                       \
		skip_test;                                                                       \
                                                                                         \
		if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) { \
			int index = p_index;                                                         \
			m_type *arr = reinterpret_cast<m_type *>(_data._mem);                        \
                                                                                         \
			if (index < 0)                                                               \
				index += arr->size();                                                    \
			if (index >= 0 && index < arr->size()) {                                     \
				valid = true;                                                            \
				cmd;                                                                     \
			}                                                                            \
		}                                                                                \
	} break;

#define DEFAULT_OP_DVECTOR_SET(m_name, dv_type, skip_cond) \
	DEFAULT_OP_ARRAY_CMD(m_name, PoolVector<dv_type>, if (skip_cond) return;, arr->set(index, p_value); return )

#define DEFAULT_OP_DVECTOR_GET(m_name, dv_type) \
	DEFAULT_OP_ARRAY_CMD(m_name, const PoolVector<dv_type>, ;, return arr->get(index))

void Variant::set(const Variant &p_index, const Variant &p_value, bool *r_valid) {

	static bool _dummy = false;

	bool &valid = r_valid ? *r_valid : _dummy;
	valid = false;

	switch (type) {
		case NIL: {
			return;
		} break;
		case BOOL: {
			return;
		} break;
		case INT: {
			return;
		} break;
		case REAL: {
			return;
		} break;
		case STRING: {

			if (p_index.type != Variant::INT && p_index.type != Variant::REAL)
				return;

			int idx = p_index;
			String *str = reinterpret_cast<String *>(_data._mem);
			int len = str->length();
			if (idx < 0)
				idx += len;
			if (idx < 0 || idx >= len)
				return;

			String chr;
			if (p_value.type == Variant::INT || p_value.type == Variant::REAL) {

				chr = String::chr(p_value);
			} else if (p_value.type == Variant::STRING) {

				chr = p_value;
			} else {
				return;
			}

			*str = str->substr(0, idx) + chr + str->substr(idx + 1, len);
			valid = true;
			return;

		} break;
		case VECTOR2: {

			if (p_value.type != Variant::INT && p_value.type != Variant::REAL)
				return;

			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {
				// scalar index
				int idx = p_index;

				if (idx < 0)
					idx += 2;
				if (idx >= 0 && idx < 2) {

					Vector2 *v = reinterpret_cast<Vector2 *>(_data._mem);
					valid = true;
					(*v)[idx] = p_value;
					return;
				}
			} else if (p_index.get_type() == Variant::STRING) {
				//scalar name

				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				Vector2 *v = reinterpret_cast<Vector2 *>(_data._mem);
				if (*str == "x" || *str == "width") {
					valid = true;
					v->x = p_value;
					return;
				} else if (*str == "y" || *str == "height") {
					valid = true;
					v->y = p_value;
					return;
				}
			}

		} break; // 5
		case RECT2: {

			if (p_value.type != Variant::VECTOR2)
				return;

			if (p_index.get_type() == Variant::STRING) {
				//scalar name

				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				Rect2 *v = reinterpret_cast<Rect2 *>(_data._mem);
				if (*str == "pos") {
					valid = true;
					v->pos = p_value;
					return;
				} else if (*str == "size") {
					valid = true;
					v->size = p_value;
					return;
				} else if (*str == "end") {
					valid = true;
					v->size = Vector2(p_value) - v->pos;
					return;
				}
			}
		} break;
		case TRANSFORM2D: {

			if (p_value.type != Variant::VECTOR2)
				return;

			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {

				int index = p_index;

				if (index < 0)
					index += 3;
				if (index >= 0 && index < 3) {
					Transform2D *v = _data._transform2d;

					valid = true;
					v->elements[index] = p_value;
					return;
				}
			} else if (p_index.get_type() == Variant::STRING && p_value.get_type() == Variant::VECTOR2) {

				//scalar name
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				Transform2D *v = _data._transform2d;
				if (*str == "x") {
					valid = true;
					v->elements[0] = p_value;
					return;
				} else if (*str == "y") {
					valid = true;
					v->elements[1] = p_value;
					return;
				} else if (*str == "o") {
					valid = true;
					v->elements[2] = p_value;
					return;
				}
			}

		} break;
		case VECTOR3: {

			if (p_value.type != Variant::INT && p_value.type != Variant::REAL)
				return;

			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {
				//scalar index
				int idx = p_index;
				if (idx < 0)
					idx += 3;
				if (idx >= 0 && idx < 3) {

					Vector3 *v = reinterpret_cast<Vector3 *>(_data._mem);
					valid = true;
					(*v)[idx] = p_value;
					return;
				}
			} else if (p_index.get_type() == Variant::STRING) {

				//scalar name
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				Vector3 *v = reinterpret_cast<Vector3 *>(_data._mem);
				if (*str == "x") {
					valid = true;
					v->x = p_value;
					return;
				} else if (*str == "y") {
					valid = true;
					v->y = p_value;
					return;
				} else if (*str == "z") {
					valid = true;
					v->z = p_value;
					return;
				}
			}

		} break;
		case PLANE: {

			if (p_index.get_type() == Variant::STRING) {
				//scalar name
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				Plane *v = reinterpret_cast<Plane *>(_data._mem);
				if (*str == "x") {
					if (p_value.type != Variant::INT && p_value.type != Variant::REAL)
						return;

					valid = true;
					v->normal.x = p_value;
					return;
				} else if (*str == "y") {
					if (p_value.type != Variant::INT && p_value.type != Variant::REAL)
						return;

					valid = true;
					v->normal.y = p_value;
					return;
				} else if (*str == "z") {
					if (p_value.type != Variant::INT && p_value.type != Variant::REAL)
						return;

					valid = true;
					v->normal.z = p_value;
					return;
				} else if (*str == "normal") {
					if (p_value.type != Variant::VECTOR3)
						return;

					valid = true;
					v->normal = p_value;
					return;
				} else if (*str == "d") {
					valid = true;
					v->d = p_value;
					return;
				}
			}

		} break;
		case QUAT: {

			if (p_value.type != Variant::INT && p_value.type != Variant::REAL)
				return;

			if (p_index.get_type() == Variant::STRING) {

				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				Quat *v = reinterpret_cast<Quat *>(_data._mem);
				if (*str == "x") {
					valid = true;
					v->x = p_value;
					return;
				} else if (*str == "y") {
					valid = true;
					v->y = p_value;
					return;
				} else if (*str == "z") {
					valid = true;
					v->z = p_value;
					return;
				} else if (*str == "w") {
					valid = true;
					v->w = p_value;
					return;
				}
			}

		} break;
		case RECT3: {

			if (p_value.type != Variant::VECTOR3)
				return;

			if (p_index.get_type() == Variant::STRING) {
				//scalar name

				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				Rect3 *v = _data._rect3;
				if (*str == "pos") {
					valid = true;
					v->pos = p_value;
					return;
				} else if (*str == "size") {
					valid = true;
					v->size = p_value;
					return;
				} else if (*str == "end") {
					valid = true;
					v->size = Vector3(p_value) - v->pos;
					return;
				}
			}
		} break; //sorry naming convention fail :( not like it's used often // 10
		case BASIS: {

			if (p_value.type != Variant::VECTOR3)
				return;

			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {

				int index = p_index;

				if (index < 0)
					index += 3;
				if (index >= 0 && index < 3) {
					Basis *v = _data._basis;

					valid = true;
					v->set_axis(index, p_value);
					return;
				}
			} else if (p_index.get_type() == Variant::STRING) {

				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				Basis *v = _data._basis;

				if (*str == "x") {
					valid = true;
					v->set_axis(0, p_value);
					return;
				} else if (*str == "y") {
					valid = true;
					v->set_axis(1, p_value);
					return;
				} else if (*str == "z") {
					valid = true;
					v->set_axis(2, p_value);
					return;
				}
			}

		} break;
		case TRANSFORM: {

			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {

				if (p_value.type != Variant::VECTOR3)
					return;

				int index = p_index;

				if (index < 0)
					index += 4;
				if (index >= 0 && index < 4) {
					Transform *v = _data._transform;
					valid = true;
					if (index == 3)
						v->origin = p_value;
					else
						v->basis.set_axis(index, p_value);
					return;
				}
			} else if (p_index.get_type() == Variant::STRING) {

				Transform *v = _data._transform;
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);

				if (*str == "basis") {

					if (p_value.type != Variant::BASIS)
						return;
					valid = true;
					v->basis = p_value;
					return;
				}
				if (*str == "origin") {
					if (p_value.type != Variant::VECTOR3)
						return;
					valid = true;
					v->origin = p_value;
					return;
				}
			}

		} break;
		case COLOR: {

			if (p_value.type != Variant::INT && p_value.type != Variant::REAL)
				return;

			if (p_index.get_type() == Variant::STRING) {

				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				Color *v = reinterpret_cast<Color *>(_data._mem);
				if (*str == "r") {
					valid = true;
					v->r = p_value;
					return;
				} else if (*str == "g") {
					valid = true;
					v->g = p_value;
					return;
				} else if (*str == "b") {
					valid = true;
					v->b = p_value;
					return;
				} else if (*str == "a") {
					valid = true;
					v->a = p_value;
					return;
				} else if (*str == "h") {
					valid = true;
					v->set_hsv(p_value, v->get_s(), v->get_v());
					return;
				} else if (*str == "s") {
					valid = true;
					v->set_hsv(v->get_h(), p_value, v->get_v());
					return;
				} else if (*str == "v") {
					valid = true;
					v->set_hsv(v->get_h(), v->get_s(), p_value);
					return;
				} else if (*str == "r8") {
					valid = true;
					v->r = float(p_value) / 255.0;
					return;
				} else if (*str == "g8") {
					valid = true;
					v->g = float(p_value) / 255.0;
					return;
				} else if (*str == "b8") {
					valid = true;
					v->b = float(p_value) / 255.0;
					return;
				} else if (*str == "a8") {
					valid = true;
					v->a = float(p_value) / 255.0;
					return;
				}
			} else if (p_index.get_type() == Variant::INT) {

				int idx = p_index;
				if (idx < 0)
					idx += 4;
				if (idx >= 0 || idx < 4) {
					Color *v = reinterpret_cast<Color *>(_data._mem);
					(*v)[idx] = p_value;
					valid = true;
				}
			}

		} break;
		case IMAGE: {
		} break;
		case NODE_PATH: {
		} break; // 15
		case _RID: {
		} break;
		case OBJECT: {

			Object *obj = _get_obj().obj;
			//only if debugging!

			if (obj) {
#ifdef DEBUG_ENABLED
				if (ScriptDebugger::get_singleton() && _get_obj().ref.is_null()) {

					if (!ObjectDB::instance_validate(obj)) {
						WARN_PRINT("Attempted use of stray pointer object.");
						valid = false;
						return;
					}
				}
#endif

				if (p_index.get_type() != Variant::STRING) {
					obj->setvar(p_index, p_value, r_valid);
					return;
				}

				return obj->set(p_index, p_value, r_valid);
			}
		} break;
		case INPUT_EVENT: {

			InputEvent &ie = *_data._input_event;

			if (p_index.get_type() != Variant::STRING)
				return;

			const String &str = *reinterpret_cast<const String *>(p_index._data._mem);

			if (str == "type") {

				if (p_value.type != Variant::INT && p_value.type != Variant::REAL)
					return;

				int type = p_value;
				if (type < 0 || type >= InputEvent::TYPE_MAX)
					return; //fail
				valid = true;
				ie.type = InputEvent::Type(type);
				return;
			} else if (str == "device") {

				if (p_value.type != Variant::INT && p_value.type != Variant::REAL)
					return;

				valid = true;
				ie.device = p_value;
				return;
			} else if (str == "ID") {

				if (p_value.type != Variant::INT && p_value.type != Variant::REAL)
					return;

				valid = true;
				ie.ID = p_value;
				return;
			}

			if (ie.type == InputEvent::KEY || ie.type == InputEvent::MOUSE_BUTTON || ie.type == InputEvent::MOUSE_MOTION) {

				if (str == "shift") {

					if (p_value.type != Variant::INT && p_value.type != Variant::REAL && p_value.type != Variant::BOOL)
						return;

					valid = true;
					ie.key.mod.shift = p_value;
					return;
				}
				if (str == "alt") {

					if (p_value.type != Variant::INT && p_value.type != Variant::REAL && p_value.type != Variant::BOOL)
						return;

					valid = true;
					ie.key.mod.alt = p_value;
					return;
				}
				if (str == "control") {

					if (p_value.type != Variant::INT && p_value.type != Variant::REAL && p_value.type != Variant::BOOL)
						return;

					valid = true;
					ie.key.mod.control = p_value;
					return;
				}
				if (str == "meta") {

					if (p_value.type != Variant::INT && p_value.type != Variant::REAL && p_value.type != Variant::BOOL)
						return;

					valid = true;
					ie.key.mod.meta = p_value;
					return;
				}
			}

			if (ie.type == InputEvent::KEY) {

				if (str == "pressed") {

					if (p_value.type != Variant::INT && p_value.type != Variant::REAL && p_value.type != Variant::BOOL)
						return;

					valid = true;
					ie.key.pressed = p_value;
					return;
				} else if (str == "scancode") {

					if (p_value.type != Variant::INT && p_value.type != Variant::REAL)
						return;

					valid = true;
					ie.key.scancode = p_value;
					return;
				} else if (str == "unicode") {
					if (p_value.type != Variant::INT && p_value.type != Variant::REAL)
						return;
					valid = true;
					ie.key.unicode = p_value;
					return;
				} else if (str == "echo") {
					if (p_value.type != Variant::INT && p_value.type != Variant::REAL && p_value.type != Variant::BOOL)
						return;

					valid = true;
					ie.key.echo = p_value;
					return;
				}
			}

			if (ie.type == InputEvent::MOUSE_MOTION || ie.type == InputEvent::MOUSE_BUTTON) {

				if (str == "button_mask") {
					if (p_value.type != Variant::INT && p_value.type != Variant::REAL)
						return;
					valid = true;
					ie.mouse_button.button_mask = p_value;
					return;
				} else if (str == "x") {
					if (p_value.type != Variant::INT && p_value.type != Variant::REAL)
						return;
					valid = true;
					ie.mouse_button.x = p_value;
					return;
				} else if (str == "y") {
					if (p_value.type != Variant::INT && p_value.type != Variant::REAL)
						return;
					valid = true;
					ie.mouse_button.y = p_value;
					return;
				} else if (str == "pos") {
					if (p_value.type != Variant::VECTOR2)
						return;
					valid = true;
					Point2 value = p_value;
					ie.mouse_button.x = value.x;
					ie.mouse_button.y = value.y;
					return;
				} else if (str == "global_x") {
					if (p_value.type != Variant::INT && p_value.type != Variant::REAL)
						return;
					valid = true;
					ie.mouse_button.global_x = p_value;
					return;
				} else if (str == "global_y") {
					if (p_value.type != Variant::INT && p_value.type != Variant::REAL)
						return;
					valid = true;
					ie.mouse_button.global_y = p_value;
					return;
				} else if (str == "global_pos") {
					if (p_value.type != Variant::VECTOR2)
						return;
					valid = true;
					Point2 value = p_value;
					ie.mouse_button.global_x = value.x;
					ie.mouse_button.global_y = value.y;
					return;
				} /*else if (str=="pointer_index") {
					valid=true;
					return ie.mouse_button.pointer_index;
				}*/

				if (ie.type == InputEvent::MOUSE_MOTION) {

					if (str == "relative_x") {
						if (p_value.type != Variant::INT && p_value.type != Variant::REAL)
							return;
						valid = true;
						ie.mouse_motion.relative_x = p_value;
						return;
					} else if (str == "relative_y") {
						if (p_value.type != Variant::INT && p_value.type != Variant::REAL)
							return;
						valid = true;
						ie.mouse_motion.relative_y = p_value;
						return;
					} else if (str == "relative_pos") {
						if (p_value.type != Variant::VECTOR2)
							return;
						valid = true;
						Point2 value = p_value;
						ie.mouse_motion.relative_x = value.x;
						ie.mouse_motion.relative_y = value.y;
						return;
					}

					if (str == "speed_x") {
						if (p_value.type != Variant::INT && p_value.type != Variant::REAL)
							return;
						valid = true;
						ie.mouse_motion.speed_x = p_value;
						return;
					} else if (str == "speed_y") {
						if (p_value.type != Variant::INT && p_value.type != Variant::REAL)
							return;
						valid = true;
						ie.mouse_motion.speed_y = p_value;
						return;
					} else if (str == "speed") {
						if (p_value.type != Variant::VECTOR2)
							return;
						valid = true;
						Point2 value = p_value;
						ie.mouse_motion.speed_x = value.x;
						ie.mouse_motion.speed_y = value.y;
						return;
					}

				} else if (ie.type == InputEvent::MOUSE_BUTTON) {

					if (str == "button_index") {
						if (p_value.type != Variant::INT && p_value.type != Variant::REAL)
							return;
						valid = true;
						ie.mouse_button.button_index = p_value;
						return;
					} else if (str == "pressed") {
						if (p_value.type != Variant::INT && p_value.type != Variant::REAL && p_value.type != Variant::BOOL)
							return;
						valid = true;
						ie.mouse_button.pressed = p_value;
						return;
					} else if (str == "doubleclick") {
						if (p_value.type != Variant::INT && p_value.type != Variant::REAL && p_value.type != Variant::BOOL)
							return;
						valid = true;
						ie.mouse_button.doubleclick = p_value;
						return;
					}
				}
			}

			if (ie.type == InputEvent::JOYPAD_BUTTON) {

				if (str == "button_index") {
					if (p_value.type != Variant::REAL && p_value.type != Variant::INT)
						return;
					valid = true;
					ie.joy_button.button_index = p_value;
					return;
				}
				if (str == "pressed") {
					if (p_value.type != Variant::INT && p_value.type != Variant::REAL && p_value.type != Variant::BOOL)
						return;

					valid = true;
					ie.joy_button.pressed = p_value;
					return;
				}
				if (str == "pressure") {
					if (p_value.type != Variant::REAL && p_value.type != Variant::INT)
						return;
					valid = true;
					ie.joy_button.pressure = p_value;
					return;
				}
			}

			if (ie.type == InputEvent::JOYPAD_MOTION) {

				if (str == "axis") {
					if (p_value.type != Variant::REAL && p_value.type != Variant::INT)
						return;
					valid = true;
					ie.joy_motion.axis = p_value;
					return;
				}
				if (str == "value") {
					if (p_value.type != Variant::REAL && p_value.type != Variant::INT)
						return;
					valid = true;
					ie.joy_motion.axis_value = p_value;
					return;
				}
			}

			if (ie.type == InputEvent::SCREEN_TOUCH) {

				if (str == "index") {
					valid = true;
					ie.screen_touch.index = p_value;
					return;
				}
				if (str == "x") {
					valid = true;
					ie.screen_touch.x = p_value;
					return;
				}
				if (str == "y") {
					valid = true;
					ie.screen_touch.y = p_value;
					return;
				}
				if (str == "pos") {
					valid = true;
					Vector2 v = p_value;
					ie.screen_touch.x = v.x;
					ie.screen_touch.y = v.y;
					return;
				}
				if (str == "pressed") {
					valid = true;
					ie.screen_touch.pressed = p_value;
					return;
				}
			}

			if (ie.type == InputEvent::SCREEN_DRAG) {

				if (str == "index") {
					valid = true;
					ie.screen_drag.index = p_value;
					return;
				}
				if (str == "x") {
					valid = true;
					ie.screen_drag.x = p_value;
					return;
				}
				if (str == "y") {
					valid = true;
					ie.screen_drag.y = p_value;
					return;
				}
				if (str == "pos") {
					valid = true;
					Vector2 v = p_value;
					ie.screen_drag.x = v.x;
					ie.screen_drag.y = v.y;
					return;
				}
				if (str == "relative_x") {
					valid = true;
					ie.screen_drag.relative_x = p_value;
					return;
				}
				if (str == "relative_y") {
					valid = true;
					ie.screen_drag.relative_y = p_value;
					return;
				}
				if (str == "relative_pos") {
					valid = true;
					Vector2 v = p_value;
					ie.screen_drag.relative_x = v.x;
					ie.screen_drag.relative_y = v.y;
					return;
				}
				if (str == "speed_x") {
					valid = true;
					ie.screen_drag.speed_x = p_value;
					return;
				}
				if (str == "speed_y") {
					valid = true;
					ie.screen_drag.speed_y = p_value;
					return;
				}
				if (str == "speed") {
					valid = true;
					Vector2 v = p_value;
					ie.screen_drag.speed_x = v.x;
					ie.screen_drag.speed_y = v.y;
					return;
				}
			}
			if (ie.type == InputEvent::ACTION) {

				if (str == "action") {
					valid = true;
					ie.action.action = p_value;
					return;
				} else if (str == "pressed") {
					valid = true;
					ie.action.pressed = p_value;
					return;
				}
			}

		} break;
		case DICTIONARY: {

			Dictionary *dic = reinterpret_cast<Dictionary *>(_data._mem);
			dic->operator[](p_index) = p_value;
			valid = true; //always valid, i guess? should this really be ok?
			return;
		} break; // 20
			DEFAULT_OP_ARRAY_CMD(ARRAY, Array, ;, (*arr)[index] = p_value; return )
			DEFAULT_OP_DVECTOR_SET(POOL_BYTE_ARRAY, uint8_t, p_value.type != Variant::REAL && p_value.type != Variant::INT)
			DEFAULT_OP_DVECTOR_SET(POOL_INT_ARRAY, int, p_value.type != Variant::REAL && p_value.type != Variant::INT)
			DEFAULT_OP_DVECTOR_SET(POOL_REAL_ARRAY, real_t, p_value.type != Variant::REAL && p_value.type != Variant::INT)
			DEFAULT_OP_DVECTOR_SET(POOL_STRING_ARRAY, String, p_value.type != Variant::STRING) // 25
			DEFAULT_OP_DVECTOR_SET(POOL_VECTOR2_ARRAY, Vector2, p_value.type != Variant::VECTOR2)
			DEFAULT_OP_DVECTOR_SET(POOL_VECTOR3_ARRAY, Vector3, p_value.type != Variant::VECTOR3)
			DEFAULT_OP_DVECTOR_SET(POOL_COLOR_ARRAY, Color, p_value.type != Variant::COLOR)
		default: return;
	}
}

Variant Variant::get(const Variant &p_index, bool *r_valid) const {

	static bool _dummy = false;

	bool &valid = r_valid ? *r_valid : _dummy;

	valid = false;

	switch (type) {
		case NIL: {
			return Variant();
		} break;
		case BOOL: {
			return Variant();
		} break;
		case INT: {
			return Variant();
		} break;
		case REAL: {
			return Variant();
		} break;
		case STRING: {

			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {
				//string index

				int idx = p_index;
				const String *str = reinterpret_cast<const String *>(_data._mem);
				if (idx < 0)
					idx += str->length();
				if (idx >= 0 && idx < str->length()) {

					valid = true;
					return str->substr(idx, 1);
				}
			}

		} break;
		case VECTOR2: {

			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {
				// scalar index
				int idx = p_index;
				if (idx < 0)
					idx += 2;
				if (idx >= 0 && idx < 2) {

					const Vector2 *v = reinterpret_cast<const Vector2 *>(_data._mem);
					valid = true;
					return (*v)[idx];
				}
			} else if (p_index.get_type() == Variant::STRING) {
				//scalar name

				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				const Vector2 *v = reinterpret_cast<const Vector2 *>(_data._mem);
				if (*str == "x" || *str == "width") {
					valid = true;
					return v->x;
				} else if (*str == "y" || *str == "height") {
					valid = true;
					return v->y;
				}
			}

		} break; // 5
		case RECT2: {

			if (p_index.get_type() == Variant::STRING) {
				//scalar name

				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				const Rect2 *v = reinterpret_cast<const Rect2 *>(_data._mem);
				if (*str == "pos") {
					valid = true;
					return v->pos;
				} else if (*str == "size") {
					valid = true;
					return v->size;
				} else if (*str == "end") {
					valid = true;
					return v->size + v->pos;
				}
			}
		} break;
		case VECTOR3: {

			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {
				//scalar index
				int idx = p_index;
				if (idx < 0)
					idx += 3;
				if (idx >= 0 && idx < 3) {

					const Vector3 *v = reinterpret_cast<const Vector3 *>(_data._mem);
					valid = true;
					return (*v)[idx];
				}
			} else if (p_index.get_type() == Variant::STRING) {

				//scalar name
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				const Vector3 *v = reinterpret_cast<const Vector3 *>(_data._mem);
				if (*str == "x") {
					valid = true;
					return v->x;
				} else if (*str == "y") {
					valid = true;
					return v->y;
				} else if (*str == "z") {
					valid = true;
					return v->z;
				}
			}

		} break;
		case TRANSFORM2D: {

			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {

				int index = p_index;

				if (index < 0)
					index += 3;
				if (index >= 0 && index < 3) {
					const Transform2D *v = _data._transform2d;

					valid = true;
					return v->elements[index];
				}
			} else if (p_index.get_type() == Variant::STRING) {

				//scalar name
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				const Transform2D *v = _data._transform2d;
				if (*str == "x") {
					valid = true;
					return v->elements[0];
				} else if (*str == "y") {
					valid = true;
					return v->elements[1];
				} else if (*str == "o") {
					valid = true;
					return v->elements[2];
				}
			}

		} break;
		case PLANE: {

			if (p_index.get_type() == Variant::STRING) {
				//scalar name
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				const Plane *v = reinterpret_cast<const Plane *>(_data._mem);
				if (*str == "x") {
					valid = true;
					return v->normal.x;
				} else if (*str == "y") {
					valid = true;
					return v->normal.y;
				} else if (*str == "z") {
					valid = true;
					return v->normal.z;
				} else if (*str == "normal") {
					valid = true;
					return v->normal;
				} else if (*str == "d") {
					valid = true;
					return v->d;
				}
			}

		} break;
		case QUAT: {

			if (p_index.get_type() == Variant::STRING) {

				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				const Quat *v = reinterpret_cast<const Quat *>(_data._mem);
				if (*str == "x") {
					valid = true;
					return v->x;
				} else if (*str == "y") {
					valid = true;
					return v->y;
				} else if (*str == "z") {
					valid = true;
					return v->z;
				} else if (*str == "w") {
					valid = true;
					return v->w;
				}
			}

		} break;
		case RECT3: {

			if (p_index.get_type() == Variant::STRING) {
				//scalar name

				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				const Rect3 *v = _data._rect3;
				if (*str == "pos") {
					valid = true;
					return v->pos;
				} else if (*str == "size") {
					valid = true;
					return v->size;
				} else if (*str == "end") {
					valid = true;
					return v->size + v->pos;
				}
			}
		} break; //sorry naming convention fail :( not like it's used often // 10
		case BASIS: {

			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {

				int index = p_index;
				if (index < 0)
					index += 3;
				if (index >= 0 && index < 3) {
					const Basis *v = _data._basis;

					valid = true;
					return v->get_axis(index);
				}
			} else if (p_index.get_type() == Variant::STRING) {

				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				const Basis *v = _data._basis;

				if (*str == "x") {
					valid = true;
					return v->get_axis(0);
				} else if (*str == "y") {
					valid = true;
					return v->get_axis(1);
				} else if (*str == "z") {
					valid = true;
					return v->get_axis(2);
				}
			}

		} break;
		case TRANSFORM: {

			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {

				int index = p_index;
				if (index < 0)
					index += 4;
				if (index >= 0 && index < 4) {
					const Transform *v = _data._transform;
					valid = true;
					return index == 3 ? v->origin : v->basis.get_axis(index);
				}
			} else if (p_index.get_type() == Variant::STRING) {

				const Transform *v = _data._transform;
				const String *str = reinterpret_cast<const String *>(p_index._data._mem);

				if (*str == "basis") {
					valid = true;
					return v->basis;
				}
				if (*str == "origin") {
					valid = true;
					return v->origin;
				}
			}

		} break;
		case COLOR: {

			if (p_index.get_type() == Variant::STRING) {

				const String *str = reinterpret_cast<const String *>(p_index._data._mem);
				const Color *v = reinterpret_cast<const Color *>(_data._mem);
				if (*str == "r") {
					valid = true;
					return v->r;
				} else if (*str == "g") {
					valid = true;
					return v->g;
				} else if (*str == "b") {
					valid = true;
					return v->b;
				} else if (*str == "a") {
					valid = true;
					return v->a;
				} else if (*str == "h") {
					valid = true;
					return v->get_h();
				} else if (*str == "s") {
					valid = true;
					return v->get_s();
				} else if (*str == "v") {
					valid = true;
					return v->get_v();
				} else if (*str == "r8") {
					valid = true;
					return (int)Math::round(v->r * 255.0);
				} else if (*str == "g8") {
					valid = true;
					return (int)Math::round(v->g * 255.0);
				} else if (*str == "b8") {
					valid = true;
					return (int)Math::round(v->b * 255.0);
				} else if (*str == "a8") {
					valid = true;
					return (int)Math::round(v->a * 255.0);
				}
			} else if (p_index.get_type() == Variant::INT) {

				int idx = p_index;
				if (idx < 0)
					idx += 4;
				if (idx >= 0 || idx < 4) {
					const Color *v = reinterpret_cast<const Color *>(_data._mem);
					valid = true;
					return (*v)[idx];
				}
			}

		} break;
		case IMAGE: {
		} break;
		case NODE_PATH: {
		} break; // 15
		case _RID: {
		} break;
		case OBJECT: {
			Object *obj = _get_obj().obj;
			if (obj) {

#ifdef DEBUG_ENABLED
				if (ScriptDebugger::get_singleton() && _get_obj().ref.is_null()) {
					//only if debugging!
					if (!ObjectDB::instance_validate(obj)) {
						valid = false;
						return "Attempted get on stray pointer.";
					}
				}
#endif

				if (p_index.get_type() != Variant::STRING) {
					return obj->getvar(p_index, r_valid);
				}

				return obj->get(p_index, r_valid);
			}

		} break;
		case INPUT_EVENT: {

			InputEvent ie = operator InputEvent();

			if (p_index.get_type() != Variant::STRING)
				break;

			const String &str = *reinterpret_cast<const String *>(p_index._data._mem);

			if (str == "type") {
				valid = true;
				return ie.type;
			} else if (str == "device") {
				valid = true;
				return ie.device;
			} else if (str == "ID") {
				valid = true;
				return ie.ID;
			}

			if (ie.type == InputEvent::KEY || ie.type == InputEvent::MOUSE_BUTTON || ie.type == InputEvent::MOUSE_MOTION) {

				if (str == "shift") {
					valid = true;
					return ie.key.mod.shift;
				}
				if (str == "alt") {
					valid = true;
					return ie.key.mod.alt;
				}
				if (str == "control") {
					valid = true;
					return ie.key.mod.control;
				}
				if (str == "meta") {
					valid = true;
					return ie.key.mod.meta;
				}
			}

			if (ie.type == InputEvent::KEY) {

				if (str == "pressed") {
					valid = true;
					return ie.key.pressed;
				} else if (str == "scancode") {
					valid = true;
					return ie.key.scancode;
				} else if (str == "unicode") {
					valid = true;
					return ie.key.unicode;
				} else if (str == "echo") {
					valid = true;
					return ie.key.echo;
				}
			}

			if (ie.type == InputEvent::MOUSE_MOTION || ie.type == InputEvent::MOUSE_BUTTON) {

				if (str == "button_mask") {
					valid = true;
					return ie.mouse_button.button_mask;
				} else if (str == "x") {
					valid = true;
					return ie.mouse_button.x;
				} else if (str == "y") {
					valid = true;
					return ie.mouse_button.y;
				} else if (str == "pos") {
					valid = true;
					return Point2(ie.mouse_button.x, ie.mouse_button.y);
				} else if (str == "global_x") {
					valid = true;
					return ie.mouse_button.global_x;
				} else if (str == "global_y") {
					valid = true;
					return ie.mouse_button.global_y;
				} else if (str == "global_pos") {
					valid = true;
					return Point2(ie.mouse_button.global_x, ie.mouse_button.global_y);
				} /*else if (str=="pointer_index") {
					valid=true;
					return ie.mouse_button.pointer_index;
				}*/

				if (ie.type == InputEvent::MOUSE_MOTION) {

					if (str == "relative_x") {
						valid = true;
						return ie.mouse_motion.relative_x;
					} else if (str == "relative_y") {
						valid = true;
						return ie.mouse_motion.relative_y;
					} else if (str == "relative_pos") {
						valid = true;
						return Point2(ie.mouse_motion.relative_x, ie.mouse_motion.relative_y);
					} else if (str == "speed_x") {
						valid = true;
						return ie.mouse_motion.speed_x;
					} else if (str == "speed_y") {
						valid = true;
						return ie.mouse_motion.speed_y;
					} else if (str == "speed") {
						valid = true;
						return Point2(ie.mouse_motion.speed_x, ie.mouse_motion.speed_y);
					}

				} else if (ie.type == InputEvent::MOUSE_BUTTON) {

					if (str == "button_index") {
						valid = true;
						return ie.mouse_button.button_index;
					} else if (str == "pressed") {
						valid = true;
						return ie.mouse_button.pressed;
					} else if (str == "doubleclick") {
						valid = true;
						return ie.mouse_button.doubleclick;
					}
				}
			}

			if (ie.type == InputEvent::JOYPAD_BUTTON) {

				if (str == "button_index") {
					valid = true;
					return ie.joy_button.button_index;
				}
				if (str == "pressed") {
					valid = true;
					return ie.joy_button.pressed;
				}
				if (str == "pressure") {
					valid = true;
					return ie.joy_button.pressure;
				}
			}

			if (ie.type == InputEvent::JOYPAD_MOTION) {

				if (str == "axis") {
					valid = true;
					return ie.joy_motion.axis;
				}
				if (str == "value") {
					valid = true;
					return ie.joy_motion.axis_value;
				}
			}

			if (ie.type == InputEvent::SCREEN_TOUCH) {

				if (str == "index") {
					valid = true;
					return ie.screen_touch.index;
				}
				if (str == "x") {
					valid = true;
					return ie.screen_touch.x;
				}
				if (str == "y") {
					valid = true;
					return ie.screen_touch.y;
				}
				if (str == "pos") {
					valid = true;
					return Vector2(ie.screen_touch.x, ie.screen_touch.y);
				}
				if (str == "pressed") {
					valid = true;
					return ie.screen_touch.pressed;
				}
			}

			if (ie.type == InputEvent::SCREEN_DRAG) {

				if (str == "index") {
					valid = true;
					return ie.screen_drag.index;
				}
				if (str == "x") {
					valid = true;
					return ie.screen_drag.x;
				}
				if (str == "y") {
					valid = true;
					return ie.screen_drag.y;
				}
				if (str == "pos") {
					valid = true;
					return Vector2(ie.screen_drag.x, ie.screen_drag.y);
				}
				if (str == "relative_x") {
					valid = true;
					return ie.screen_drag.relative_x;
				}
				if (str == "relative_y") {
					valid = true;
					return ie.screen_drag.relative_y;
				}
				if (str == "relative_pos") {
					valid = true;
					return Vector2(ie.screen_drag.relative_x, ie.screen_drag.relative_y);
				}
				if (str == "speed_x") {
					valid = true;
					return ie.screen_drag.speed_x;
				}
				if (str == "speed_y") {
					valid = true;
					return ie.screen_drag.speed_y;
				}
				if (str == "speed") {
					valid = true;
					return Vector2(ie.screen_drag.speed_x, ie.screen_drag.speed_y);
				}
			}
			if (ie.type == InputEvent::ACTION) {

				if (str == "action") {
					valid = true;
					return ie.action.action;
				} else if (str == "pressed") {
					valid = true;
					return ie.action.pressed;
				}
			}

		} break;
		case DICTIONARY: {

			const Dictionary *dic = reinterpret_cast<const Dictionary *>(_data._mem);
			const Variant *res = dic->getptr(p_index);
			if (res) {
				valid = true;
				return *res;
			}
		} break; // 20
			DEFAULT_OP_ARRAY_CMD(ARRAY, const Array, ;, return (*arr)[index])
			DEFAULT_OP_DVECTOR_GET(POOL_BYTE_ARRAY, uint8_t)
			DEFAULT_OP_DVECTOR_GET(POOL_INT_ARRAY, int)
			DEFAULT_OP_DVECTOR_GET(POOL_REAL_ARRAY, real_t)
			DEFAULT_OP_DVECTOR_GET(POOL_STRING_ARRAY, String)
			DEFAULT_OP_DVECTOR_GET(POOL_VECTOR2_ARRAY, Vector2)
			DEFAULT_OP_DVECTOR_GET(POOL_VECTOR3_ARRAY, Vector3)
			DEFAULT_OP_DVECTOR_GET(POOL_COLOR_ARRAY, Color)
		default: return Variant();
	}

	return Variant();
}

bool Variant::in(const Variant &p_index, bool *r_valid) const {

	if (r_valid)
		*r_valid = true;

	switch (type) {

		case STRING: {

			if (p_index.get_type() == Variant::STRING) {
				//string index
				String idx = p_index;
				const String *str = reinterpret_cast<const String *>(_data._mem);

				return str->find(idx) != -1;
			}

		} break;
		case OBJECT: {
			Object *obj = _get_obj().obj;
			if (obj) {

				bool valid = false;
#ifdef DEBUG_ENABLED
				if (ScriptDebugger::get_singleton() && _get_obj().ref.is_null()) {
					//only if debugging!
					if (!ObjectDB::instance_validate(obj)) {
						if (r_valid) {
							*r_valid = false;
						}
						return "Attempted get on stray pointer.";
					}
				}
#endif

				if (p_index.get_type() != Variant::STRING) {
					obj->getvar(p_index, &valid);
				} else {
					obj->get(p_index, &valid);
				}

				return valid;
			} else {
				if (r_valid)
					*r_valid = false;
			}
			return false;
		} break;
		case DICTIONARY: {

			const Dictionary *dic = reinterpret_cast<const Dictionary *>(_data._mem);
			return dic->has(p_index);

		} break; // 20
		case ARRAY: {

			const Array *arr = reinterpret_cast<const Array *>(_data._mem);
			int l = arr->size();
			if (l) {
				for (int i = 0; i < l; i++) {

					if (evaluate(OP_EQUAL, (*arr)[i], p_index))
						return true;
				}
			}

			return false;

		} break;
		case POOL_BYTE_ARRAY: {
			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {

				int index = p_index;
				const PoolVector<uint8_t> *arr = reinterpret_cast<const PoolVector<uint8_t> *>(_data._mem);
				int l = arr->size();
				if (l) {
					PoolVector<uint8_t>::Read r = arr->read();
					for (int i = 0; i < l; i++) {
						if (r[i] == index)
							return true;
					}
				}

				return false;
			}

		} break;
		case POOL_INT_ARRAY: {
			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {

				int index = p_index;
				const PoolVector<int> *arr = reinterpret_cast<const PoolVector<int> *>(_data._mem);
				int l = arr->size();
				if (l) {
					PoolVector<int>::Read r = arr->read();
					for (int i = 0; i < l; i++) {
						if (r[i] == index)
							return true;
					}
				}

				return false;
			}
		} break;
		case POOL_REAL_ARRAY: {

			if (p_index.get_type() == Variant::INT || p_index.get_type() == Variant::REAL) {

				real_t index = p_index;
				const PoolVector<real_t> *arr = reinterpret_cast<const PoolVector<real_t> *>(_data._mem);
				int l = arr->size();
				if (l) {
					PoolVector<real_t>::Read r = arr->read();
					for (int i = 0; i < l; i++) {
						if (r[i] == index)
							return true;
					}
				}

				return false;
			}

		} break;
		case POOL_STRING_ARRAY: {
			if (p_index.get_type() == Variant::STRING) {

				String index = p_index;
				const PoolVector<String> *arr = reinterpret_cast<const PoolVector<String> *>(_data._mem);

				int l = arr->size();
				if (l) {
					PoolVector<String>::Read r = arr->read();
					for (int i = 0; i < l; i++) {
						if (r[i] == index)
							return true;
					}
				}

				return false;
			}

		} break; //25
		case POOL_VECTOR2_ARRAY: {
			if (p_index.get_type() == Variant::VECTOR2) {

				Vector2 index = p_index;
				const PoolVector<Vector2> *arr = reinterpret_cast<const PoolVector<Vector2> *>(_data._mem);

				int l = arr->size();
				if (l) {
					PoolVector<Vector2>::Read r = arr->read();
					for (int i = 0; i < l; i++) {
						if (r[i] == index)
							return true;
					}
				}

				return false;
			}

		} break;
		case POOL_VECTOR3_ARRAY: {
			if (p_index.get_type() == Variant::VECTOR3) {

				Vector3 index = p_index;
				const PoolVector<Vector3> *arr = reinterpret_cast<const PoolVector<Vector3> *>(_data._mem);

				int l = arr->size();
				if (l) {
					PoolVector<Vector3>::Read r = arr->read();
					for (int i = 0; i < l; i++) {
						if (r[i] == index)
							return true;
					}
				}

				return false;
			}

		} break;
		case POOL_COLOR_ARRAY: {

			if (p_index.get_type() == Variant::COLOR) {

				Color index = p_index;
				const PoolVector<Color> *arr = reinterpret_cast<const PoolVector<Color> *>(_data._mem);

				int l = arr->size();
				if (l) {
					PoolVector<Color>::Read r = arr->read();
					for (int i = 0; i < l; i++) {
						if (r[i] == index)
							return true;
					}
				}

				return false;
			}
		} break;
		default: {}
	}

	if (r_valid)
		*r_valid = false;
	return false;
}

void Variant::get_property_list(List<PropertyInfo> *p_list) const {

	switch (type) {
		case VECTOR2: {

			p_list->push_back(PropertyInfo(Variant::REAL, "x"));
			p_list->push_back(PropertyInfo(Variant::REAL, "y"));
			p_list->push_back(PropertyInfo(Variant::REAL, "width"));
			p_list->push_back(PropertyInfo(Variant::REAL, "height"));

		} break; // 5
		case RECT2: {

			p_list->push_back(PropertyInfo(Variant::VECTOR2, "pos"));
			p_list->push_back(PropertyInfo(Variant::VECTOR2, "size"));
			p_list->push_back(PropertyInfo(Variant::VECTOR2, "end"));

		} break;
		case VECTOR3: {

			p_list->push_back(PropertyInfo(Variant::REAL, "x"));
			p_list->push_back(PropertyInfo(Variant::REAL, "y"));
			p_list->push_back(PropertyInfo(Variant::REAL, "z"));

		} break;
		case TRANSFORM2D: {

			p_list->push_back(PropertyInfo(Variant::VECTOR2, "x"));
			p_list->push_back(PropertyInfo(Variant::VECTOR2, "y"));
			p_list->push_back(PropertyInfo(Variant::VECTOR2, "o"));

		} break;
		case PLANE: {

			p_list->push_back(PropertyInfo(Variant::VECTOR3, "normal"));
			p_list->push_back(PropertyInfo(Variant::REAL, "x"));
			p_list->push_back(PropertyInfo(Variant::REAL, "y"));
			p_list->push_back(PropertyInfo(Variant::REAL, "z"));
			p_list->push_back(PropertyInfo(Variant::REAL, "d"));

		} break;
		case QUAT: {

			p_list->push_back(PropertyInfo(Variant::REAL, "x"));
			p_list->push_back(PropertyInfo(Variant::REAL, "y"));
			p_list->push_back(PropertyInfo(Variant::REAL, "z"));
			p_list->push_back(PropertyInfo(Variant::REAL, "w"));

		} break;
		case RECT3: {
			p_list->push_back(PropertyInfo(Variant::VECTOR3, "pos"));
			p_list->push_back(PropertyInfo(Variant::VECTOR3, "size"));
			p_list->push_back(PropertyInfo(Variant::VECTOR3, "end"));
		} break; //sorry naming convention fail :( not like it's used often // 10
		case BASIS: {

			p_list->push_back(PropertyInfo(Variant::VECTOR3, "x"));
			p_list->push_back(PropertyInfo(Variant::VECTOR3, "y"));
			p_list->push_back(PropertyInfo(Variant::VECTOR3, "z"));

		} break;
		case TRANSFORM: {

			p_list->push_back(PropertyInfo(Variant::BASIS, "basis"));
			p_list->push_back(PropertyInfo(Variant::VECTOR3, "origin"));

		} break;
		case COLOR: {
			p_list->push_back(PropertyInfo(Variant::REAL, "r"));
			p_list->push_back(PropertyInfo(Variant::REAL, "g"));
			p_list->push_back(PropertyInfo(Variant::REAL, "b"));
			p_list->push_back(PropertyInfo(Variant::REAL, "a"));
			p_list->push_back(PropertyInfo(Variant::REAL, "h"));
			p_list->push_back(PropertyInfo(Variant::REAL, "s"));
			p_list->push_back(PropertyInfo(Variant::REAL, "v"));
			p_list->push_back(PropertyInfo(Variant::INT, "r8"));
			p_list->push_back(PropertyInfo(Variant::INT, "g8"));
			p_list->push_back(PropertyInfo(Variant::INT, "b8"));
			p_list->push_back(PropertyInfo(Variant::INT, "a8"));

		} break;
		case IMAGE: {
		} break;
		case NODE_PATH: {
		} break; // 15
		case _RID: {
		} break;
		case OBJECT: {

			Object *obj = _get_obj().obj;
			if (obj) {
#ifdef DEBUG_ENABLED
				if (ScriptDebugger::get_singleton() && _get_obj().ref.is_null()) {
					//only if debugging!
					if (!ObjectDB::instance_validate(obj)) {
						WARN_PRINT("Attempted get_property list on stray pointer.");
						return;
					}
				}
#endif

				obj->get_property_list(p_list);
			}

		} break;
		case INPUT_EVENT: {

			InputEvent ie = operator InputEvent();

			p_list->push_back(PropertyInfo(Variant::INT, "type"));
			p_list->push_back(PropertyInfo(Variant::INT, "device"));
			p_list->push_back(PropertyInfo(Variant::INT, "ID"));

			if (ie.type == InputEvent::KEY || ie.type == InputEvent::MOUSE_BUTTON || ie.type == InputEvent::MOUSE_MOTION) {

				p_list->push_back(PropertyInfo(Variant::BOOL, "shift"));
				p_list->push_back(PropertyInfo(Variant::BOOL, "alt"));
				p_list->push_back(PropertyInfo(Variant::BOOL, "control"));
				p_list->push_back(PropertyInfo(Variant::BOOL, "meta"));
			}

			if (ie.type == InputEvent::KEY) {

				p_list->push_back(PropertyInfo(Variant::BOOL, "pressed"));
				p_list->push_back(PropertyInfo(Variant::BOOL, "echo"));
				p_list->push_back(PropertyInfo(Variant::INT, "scancode"));
				p_list->push_back(PropertyInfo(Variant::INT, "unicode"));
			}

			if (ie.type == InputEvent::MOUSE_MOTION || ie.type == InputEvent::MOUSE_BUTTON) {

				p_list->push_back(PropertyInfo(Variant::INT, "button_mask"));
				p_list->push_back(PropertyInfo(Variant::REAL, "x"));
				p_list->push_back(PropertyInfo(Variant::REAL, "y"));
				p_list->push_back(PropertyInfo(Variant::VECTOR2, "pos"));
				p_list->push_back(PropertyInfo(Variant::REAL, "global_x"));
				p_list->push_back(PropertyInfo(Variant::REAL, "global_y"));
				p_list->push_back(PropertyInfo(Variant::VECTOR2, "global_pos"));

				if (ie.type == InputEvent::MOUSE_MOTION) {

					p_list->push_back(PropertyInfo(Variant::REAL, "relative_x"));
					p_list->push_back(PropertyInfo(Variant::REAL, "relative_y"));
					p_list->push_back(PropertyInfo(Variant::VECTOR2, "relative_pos"));
					p_list->push_back(PropertyInfo(Variant::REAL, "speed_x"));
					p_list->push_back(PropertyInfo(Variant::REAL, "speed_y"));
					p_list->push_back(PropertyInfo(Variant::VECTOR2, "speed"));

				} else if (ie.type == InputEvent::MOUSE_BUTTON) {

					p_list->push_back(PropertyInfo(Variant::INT, "button_index"));
					p_list->push_back(PropertyInfo(Variant::BOOL, "pressed"));
					p_list->push_back(PropertyInfo(Variant::BOOL, "doubleclick"));
				}
			}

			if (ie.type == InputEvent::JOYPAD_BUTTON) {

				p_list->push_back(PropertyInfo(Variant::INT, "button_index"));
				p_list->push_back(PropertyInfo(Variant::BOOL, "pressed"));
				p_list->push_back(PropertyInfo(Variant::REAL, "pressure"));
			}

			if (ie.type == InputEvent::JOYPAD_MOTION) {

				p_list->push_back(PropertyInfo(Variant::INT, "axis"));
				p_list->push_back(PropertyInfo(Variant::REAL, "value"));
			}

			if (ie.type == InputEvent::SCREEN_TOUCH) {

				p_list->push_back(PropertyInfo(Variant::INT, "index"));
				p_list->push_back(PropertyInfo(Variant::REAL, "x"));
				p_list->push_back(PropertyInfo(Variant::REAL, "y"));
				p_list->push_back(PropertyInfo(Variant::VECTOR2, "pos"));
				p_list->push_back(PropertyInfo(Variant::BOOL, "pressed"));
			}

			if (ie.type == InputEvent::SCREEN_DRAG) {

				p_list->push_back(PropertyInfo(Variant::INT, "index"));
				p_list->push_back(PropertyInfo(Variant::REAL, "x"));
				p_list->push_back(PropertyInfo(Variant::REAL, "y"));
				p_list->push_back(PropertyInfo(Variant::VECTOR2, "pos"));
				p_list->push_back(PropertyInfo(Variant::REAL, "relative_x"));
				p_list->push_back(PropertyInfo(Variant::REAL, "relative_y"));
				p_list->push_back(PropertyInfo(Variant::VECTOR2, "relative_pos"));
				p_list->push_back(PropertyInfo(Variant::REAL, "speed_x"));
				p_list->push_back(PropertyInfo(Variant::REAL, "speed_y"));
				p_list->push_back(PropertyInfo(Variant::VECTOR2, "speed"));
			}

		} break;
		case DICTIONARY: {

			const Dictionary *dic = reinterpret_cast<const Dictionary *>(_data._mem);
			List<Variant> keys;
			dic->get_key_list(&keys);
			for (List<Variant>::Element *E = keys.front(); E; E = E->next()) {
				if (E->get().get_type() == Variant::STRING) {
					p_list->push_back(PropertyInfo(Variant::STRING, E->get()));
				}
			}
		} break; // 20
		case ARRAY:
		case POOL_BYTE_ARRAY:
		case POOL_INT_ARRAY:
		case POOL_REAL_ARRAY:
		case POOL_STRING_ARRAY:
		case POOL_VECTOR3_ARRAY:
		case POOL_COLOR_ARRAY: {

			//nothing
		} break;
		default: {}
	}
}

bool Variant::iter_init(Variant &r_iter, bool &valid) const {

	valid = true;
	switch (type) {
		case INT: {
			r_iter = 0;
			return _data._int > 0;
		} break;
		case REAL: {
			r_iter = 0.0;
			return _data._real > 0.0;
		} break;
		case VECTOR2: {
			real_t from = reinterpret_cast<const Vector2 *>(_data._mem)->x;
			real_t to = reinterpret_cast<const Vector2 *>(_data._mem)->y;

			r_iter = from;

			return from < to;
		} break;
		case VECTOR3: {
			real_t from = reinterpret_cast<const Vector3 *>(_data._mem)->x;
			real_t to = reinterpret_cast<const Vector3 *>(_data._mem)->y;
			real_t step = reinterpret_cast<const Vector3 *>(_data._mem)->z;

			r_iter = from;

			if (from == to) {
				return false;
			} else if (from < to) {
				return step > 0.0;
			} else {
				return step < 0.0;
			}
			//return true;
		} break;
		case OBJECT: {

#ifdef DEBUG_ENABLED
			if (!_get_obj().obj) {
				valid = false;
				return false;
			}

			if (ScriptDebugger::get_singleton() && _get_obj().ref.is_null() && !ObjectDB::instance_validate(_get_obj().obj)) {
				valid = false;
				return false;
			}
#endif
			Variant::CallError ce;
			ce.error = Variant::CallError::CALL_OK;
			Array ref;
			ref.push_back(r_iter);
			Variant vref = ref;
			const Variant *refp[] = { &vref };
			Variant ret = _get_obj().obj->call(CoreStringNames::get_singleton()->_iter_init, refp, 1, ce);

			if (ref.size() != 1 || ce.error != Variant::CallError::CALL_OK) {
				valid = false;
				return false;
			}

			r_iter = ref[0];
			return ret;
		} break;

		case STRING: {

			const String *str = reinterpret_cast<const String *>(_data._mem);
			if (str->empty())
				return false;
			r_iter = 0;
			return true;
		} break;
		case DICTIONARY: {

			const Dictionary *dic = reinterpret_cast<const Dictionary *>(_data._mem);
			if (dic->empty())
				return false;

			const Variant *next = dic->next(NULL);
			r_iter = *next;
			return true;

		} break;
		case ARRAY: {

			const Array *arr = reinterpret_cast<const Array *>(_data._mem);
			if (arr->empty())
				return false;
			r_iter = 0;
			return true;
		} break;
		case POOL_BYTE_ARRAY: {
			const PoolVector<uint8_t> *arr = reinterpret_cast<const PoolVector<uint8_t> *>(_data._mem);
			if (arr->size() == 0)
				return false;
			r_iter = 0;
			return true;

		} break;
		case POOL_INT_ARRAY: {
			const PoolVector<int> *arr = reinterpret_cast<const PoolVector<int> *>(_data._mem);
			if (arr->size() == 0)
				return false;
			r_iter = 0;
			return true;

		} break;
		case POOL_REAL_ARRAY: {
			const PoolVector<real_t> *arr = reinterpret_cast<const PoolVector<real_t> *>(_data._mem);
			if (arr->size() == 0)
				return false;
			r_iter = 0;
			return true;

		} break;
		case POOL_STRING_ARRAY: {
			const PoolVector<String> *arr = reinterpret_cast<const PoolVector<String> *>(_data._mem);
			if (arr->size() == 0)
				return false;
			r_iter = 0;
			return true;
		} break;
		case POOL_VECTOR2_ARRAY: {

			const PoolVector<Vector2> *arr = reinterpret_cast<const PoolVector<Vector2> *>(_data._mem);
			if (arr->size() == 0)
				return false;
			r_iter = 0;
			return true;
		} break;
		case POOL_VECTOR3_ARRAY: {

			const PoolVector<Vector3> *arr = reinterpret_cast<const PoolVector<Vector3> *>(_data._mem);
			if (arr->size() == 0)
				return false;
			r_iter = 0;
			return true;
		} break;
		case POOL_COLOR_ARRAY: {

			const PoolVector<Color> *arr = reinterpret_cast<const PoolVector<Color> *>(_data._mem);
			if (arr->size() == 0)
				return false;
			r_iter = 0;
			return true;

		} break;
		default: {}
	}

	valid = false;
	return false;
}
bool Variant::iter_next(Variant &r_iter, bool &valid) const {

	valid = true;
	switch (type) {
		case INT: {

			int64_t idx = r_iter;
			idx++;
			if (idx >= _data._int)
				return false;
			r_iter = idx;
			return true;
		} break;
		case REAL: {

			double idx = r_iter;
			idx += 1.0;
			if (idx >= _data._real)
				return false;
			r_iter = idx;
			return true;
		} break;
		case VECTOR2: {
			real_t idx = r_iter;
			idx += 1.0;
			if (idx >= reinterpret_cast<const Vector2 *>(_data._mem)->y)
				return false;
			r_iter = idx;
			return true;
		} break;
		case VECTOR3: {
			real_t to = reinterpret_cast<const Vector3 *>(_data._mem)->y;
			real_t step = reinterpret_cast<const Vector3 *>(_data._mem)->z;

			real_t idx = r_iter;
			idx += step;

			if (step < 0.0 && idx <= to)
				return false;

			if (step > 0.0 && idx >= to)
				return false;

			r_iter = idx;
			return true;
		} break;
		case OBJECT: {

#ifdef DEBUG_ENABLED
			if (!_get_obj().obj) {
				valid = false;
				return false;
			}

			if (ScriptDebugger::get_singleton() && _get_obj().ref.is_null() && !ObjectDB::instance_validate(_get_obj().obj)) {
				valid = false;
				return false;
			}
#endif
			Variant::CallError ce;
			ce.error = Variant::CallError::CALL_OK;
			Array ref;
			ref.push_back(r_iter);
			Variant vref = ref;
			const Variant *refp[] = { &vref };
			Variant ret = _get_obj().obj->call(CoreStringNames::get_singleton()->_iter_next, refp, 1, ce);

			if (ref.size() != 1 || ce.error != Variant::CallError::CALL_OK) {
				valid = false;
				return false;
			}

			r_iter = ref[0];

			return ret;
		} break;

		case STRING: {

			const String *str = reinterpret_cast<const String *>(_data._mem);
			int idx = r_iter;
			idx++;
			if (idx >= str->length())
				return false;
			r_iter = idx;
			return true;
		} break;
		case DICTIONARY: {

			const Dictionary *dic = reinterpret_cast<const Dictionary *>(_data._mem);
			const Variant *next = dic->next(&r_iter);
			if (!next)
				return false;

			r_iter = *next;
			return true;

		} break;
		case ARRAY: {

			const Array *arr = reinterpret_cast<const Array *>(_data._mem);
			int idx = r_iter;
			idx++;
			if (idx >= arr->size())
				return false;
			r_iter = idx;
			return true;
		} break;
		case POOL_BYTE_ARRAY: {
			const PoolVector<uint8_t> *arr = reinterpret_cast<const PoolVector<uint8_t> *>(_data._mem);
			int idx = r_iter;
			idx++;
			if (idx >= arr->size())
				return false;
			r_iter = idx;
			return true;

		} break;
		case POOL_INT_ARRAY: {
			const PoolVector<int> *arr = reinterpret_cast<const PoolVector<int> *>(_data._mem);
			int idx = r_iter;
			idx++;
			if (idx >= arr->size())
				return false;
			r_iter = idx;
			return true;

		} break;
		case POOL_REAL_ARRAY: {
			const PoolVector<real_t> *arr = reinterpret_cast<const PoolVector<real_t> *>(_data._mem);
			int idx = r_iter;
			idx++;
			if (idx >= arr->size())
				return false;
			r_iter = idx;
			return true;

		} break;
		case POOL_STRING_ARRAY: {
			const PoolVector<String> *arr = reinterpret_cast<const PoolVector<String> *>(_data._mem);
			int idx = r_iter;
			idx++;
			if (idx >= arr->size())
				return false;
			r_iter = idx;
			return true;
		} break;
		case POOL_VECTOR2_ARRAY: {

			const PoolVector<Vector2> *arr = reinterpret_cast<const PoolVector<Vector2> *>(_data._mem);
			int idx = r_iter;
			idx++;
			if (idx >= arr->size())
				return false;
			r_iter = idx;
			return true;
		} break;
		case POOL_VECTOR3_ARRAY: {

			const PoolVector<Vector3> *arr = reinterpret_cast<const PoolVector<Vector3> *>(_data._mem);
			int idx = r_iter;
			idx++;
			if (idx >= arr->size())
				return false;
			r_iter = idx;
			return true;
		} break;
		case POOL_COLOR_ARRAY: {

			const PoolVector<Color> *arr = reinterpret_cast<const PoolVector<Color> *>(_data._mem);
			int idx = r_iter;
			idx++;
			if (idx >= arr->size())
				return false;
			r_iter = idx;
			return true;
		} break;
		default: {}
	}

	valid = false;
	return false;
}

Variant Variant::iter_get(const Variant &r_iter, bool &r_valid) const {

	r_valid = true;
	switch (type) {
		case INT: {

			return r_iter;
		} break;
		case REAL: {

			return r_iter;
		} break;
		case VECTOR2: {

			return r_iter;
		} break;
		case VECTOR3: {

			return r_iter;
		} break;
		case OBJECT: {

#ifdef DEBUG_ENABLED
			if (!_get_obj().obj) {
				r_valid = false;
				return Variant();
			}

			if (ScriptDebugger::get_singleton() && _get_obj().ref.is_null() && !ObjectDB::instance_validate(_get_obj().obj)) {
				r_valid = false;
				return Variant();
			}
#endif
			Variant::CallError ce;
			ce.error = Variant::CallError::CALL_OK;
			const Variant *refp[] = { &r_iter };
			Variant ret = _get_obj().obj->call(CoreStringNames::get_singleton()->_iter_get, refp, 1, ce);

			if (ce.error != Variant::CallError::CALL_OK) {
				r_valid = false;
				return Variant();
			}

			//r_iter=ref[0];

			return ret;
		} break;

		case STRING: {

			const String *str = reinterpret_cast<const String *>(_data._mem);
			return str->substr(r_iter, 1);
		} break;
		case DICTIONARY: {

			return r_iter; //iterator is the same as the key

		} break;
		case ARRAY: {

			const Array *arr = reinterpret_cast<const Array *>(_data._mem);
			int idx = r_iter;
#ifdef DEBUG_ENABLED
			if (idx < 0 || idx >= arr->size()) {
				r_valid = false;
				return Variant();
			}
#endif
			return arr->get(idx);
		} break;
		case POOL_BYTE_ARRAY: {
			const PoolVector<uint8_t> *arr = reinterpret_cast<const PoolVector<uint8_t> *>(_data._mem);
			int idx = r_iter;
#ifdef DEBUG_ENABLED
			if (idx < 0 || idx >= arr->size()) {
				r_valid = false;
				return Variant();
			}
#endif
			return arr->get(idx);
		} break;
		case POOL_INT_ARRAY: {
			const PoolVector<int> *arr = reinterpret_cast<const PoolVector<int> *>(_data._mem);
			int idx = r_iter;
#ifdef DEBUG_ENABLED
			if (idx < 0 || idx >= arr->size()) {
				r_valid = false;
				return Variant();
			}
#endif
			return arr->get(idx);
		} break;
		case POOL_REAL_ARRAY: {
			const PoolVector<real_t> *arr = reinterpret_cast<const PoolVector<real_t> *>(_data._mem);
			int idx = r_iter;
#ifdef DEBUG_ENABLED
			if (idx < 0 || idx >= arr->size()) {
				r_valid = false;
				return Variant();
			}
#endif
			return arr->get(idx);
		} break;
		case POOL_STRING_ARRAY: {
			const PoolVector<String> *arr = reinterpret_cast<const PoolVector<String> *>(_data._mem);
			int idx = r_iter;
#ifdef DEBUG_ENABLED
			if (idx < 0 || idx >= arr->size()) {
				r_valid = false;
				return Variant();
			}
#endif
			return arr->get(idx);
		} break;
		case POOL_VECTOR2_ARRAY: {

			const PoolVector<Vector2> *arr = reinterpret_cast<const PoolVector<Vector2> *>(_data._mem);
			int idx = r_iter;
#ifdef DEBUG_ENABLED
			if (idx < 0 || idx >= arr->size()) {
				r_valid = false;
				return Variant();
			}
#endif
			return arr->get(idx);
		} break;
		case POOL_VECTOR3_ARRAY: {

			const PoolVector<Vector3> *arr = reinterpret_cast<const PoolVector<Vector3> *>(_data._mem);
			int idx = r_iter;
#ifdef DEBUG_ENABLED
			if (idx < 0 || idx >= arr->size()) {
				r_valid = false;
				return Variant();
			}
#endif
			return arr->get(idx);
		} break;
		case POOL_COLOR_ARRAY: {

			const PoolVector<Color> *arr = reinterpret_cast<const PoolVector<Color> *>(_data._mem);
			int idx = r_iter;
#ifdef DEBUG_ENABLED
			if (idx < 0 || idx >= arr->size()) {
				r_valid = false;
				return Variant();
			}
#endif
			return arr->get(idx);
		} break;
		default: {}
	}

	r_valid = false;
	return Variant();
}

void Variant::blend(const Variant &a, const Variant &b, float c, Variant &r_dst) {
	if (a.type != b.type) {
		if (a.is_num() && b.is_num()) {
			real_t va = a;
			real_t vb = b;
			r_dst = va + vb * c;
		} else {
			r_dst = a;
		}
		return;
	}

	switch (a.type) {
		case NIL: {
			r_dst = Variant();
		}
			return;
		case INT: {
			int64_t va = a._data._int;
			int64_t vb = b._data._int;
			r_dst = int(va + vb * c + 0.5);
		}
			return;
		case REAL: {
			double ra = a._data._real;
			double rb = b._data._real;
			r_dst = ra + rb * c;
		}
			return;
		case VECTOR2: {
			r_dst = *reinterpret_cast<const Vector2 *>(a._data._mem) + *reinterpret_cast<const Vector2 *>(b._data._mem) * c;
		}
			return;
		case RECT2: {
			const Rect2 *ra = reinterpret_cast<const Rect2 *>(a._data._mem);
			const Rect2 *rb = reinterpret_cast<const Rect2 *>(b._data._mem);
			r_dst = Rect2(ra->pos + rb->pos * c, ra->size + rb->size * c);
		}
			return;
		case VECTOR3: {
			r_dst = *reinterpret_cast<const Vector3 *>(a._data._mem) + *reinterpret_cast<const Vector3 *>(b._data._mem) * c;
		}
			return;
		case RECT3: {
			const Rect3 *ra = reinterpret_cast<const Rect3 *>(a._data._mem);
			const Rect3 *rb = reinterpret_cast<const Rect3 *>(b._data._mem);
			r_dst = Rect3(ra->pos + rb->pos * c, ra->size + rb->size * c);
		}
			return;
		case QUAT: {
			Quat empty_rot;
			const Quat *qa = reinterpret_cast<const Quat *>(a._data._mem);
			const Quat *qb = reinterpret_cast<const Quat *>(b._data._mem);
			r_dst = *qa * empty_rot.slerp(*qb, c);
		}
			return;
		case COLOR: {
			const Color *ca = reinterpret_cast<const Color *>(a._data._mem);
			const Color *cb = reinterpret_cast<const Color *>(b._data._mem);
			float r = ca->r + cb->r * c;
			float g = ca->g + cb->g * c;
			float b = ca->b + cb->b * c;
			float a = ca->a + cb->a * c;
			r = r > 1.0 ? 1.0 : r;
			g = g > 1.0 ? 1.0 : g;
			b = b > 1.0 ? 1.0 : b;
			a = a > 1.0 ? 1.0 : a;
			r_dst = Color(r, g, b, a);
		}
			return;
		default: { r_dst = c < 0.5 ? a : b; }
			return;
	}
}

void Variant::interpolate(const Variant &a, const Variant &b, float c, Variant &r_dst) {

	if (a.type != b.type) {
		if (a.is_num() && b.is_num()) {
			//not as efficient but..
			real_t va = a;
			real_t vb = b;
			r_dst = (1.0 - c) * va + vb * c;

		} else {
			r_dst = a;
		}
		return;
	}

	switch (a.type) {

		case NIL: {
			r_dst = Variant();
		}
			return;
		case BOOL: {
			r_dst = a;
		}
			return;
		case INT: {
			int64_t va = a._data._int;
			int64_t vb = b._data._int;
			r_dst = int((1.0 - c) * va + vb * c);
		}
			return;
		case REAL: {
			real_t va = a._data._real;
			real_t vb = b._data._real;
			r_dst = (1.0 - c) * va + vb * c;
		}
			return;
		case STRING: {
			//this is pretty funny and bizarre, but artists like to use it for typewritter effects
			String sa = *reinterpret_cast<const String *>(a._data._mem);
			String sb = *reinterpret_cast<const String *>(b._data._mem);
			String dst;
			int csize = sb.length() * c + sa.length() * (1.0 - c);
			if (csize == 0) {
				r_dst = "";
				return;
			}
			dst.resize(csize + 1);
			dst[csize] = 0;
			int split = csize / 2;

			for (int i = 0; i < csize; i++) {

				CharType chr = ' ';

				if (i < split) {

					if (i < sa.length())
						chr = sa[i];
					else if (i < sb.length())
						chr = sb[i];

				} else {

					if (i < sb.length())
						chr = sb[i];
					else if (i < sa.length())
						chr = sa[i];
				}

				dst[i] = chr;
			}

			r_dst = dst;
		}
			return;
		case VECTOR2: {
			r_dst = reinterpret_cast<const Vector2 *>(a._data._mem)->linear_interpolate(*reinterpret_cast<const Vector2 *>(b._data._mem), c);
		}
			return;
		case RECT2: {
			r_dst = Rect2(reinterpret_cast<const Rect2 *>(a._data._mem)->pos.linear_interpolate(reinterpret_cast<const Rect2 *>(b._data._mem)->pos, c), reinterpret_cast<const Rect2 *>(a._data._mem)->size.linear_interpolate(reinterpret_cast<const Rect2 *>(b._data._mem)->size, c));
		}
			return;
		case VECTOR3: {
			r_dst = reinterpret_cast<const Vector3 *>(a._data._mem)->linear_interpolate(*reinterpret_cast<const Vector3 *>(b._data._mem), c);
		}
			return;
		case TRANSFORM2D: {
			r_dst = a._data._transform2d->interpolate_with(*b._data._transform2d, c);
		}
			return;
		case PLANE: {
			r_dst = a;
		}
			return;
		case QUAT: {
			r_dst = reinterpret_cast<const Quat *>(a._data._mem)->slerp(*reinterpret_cast<const Quat *>(b._data._mem), c);
		}
			return;
		case RECT3: {
			r_dst = Rect3(a._data._rect3->pos.linear_interpolate(b._data._rect3->pos, c), a._data._rect3->size.linear_interpolate(b._data._rect3->size, c));
		}
			return;
		case BASIS: {
			r_dst = Transform(*a._data._basis).interpolate_with(Transform(*b._data._basis), c).basis;
		}
			return;
		case TRANSFORM: {
			r_dst = a._data._transform->interpolate_with(*b._data._transform, c);
		}
			return;
		case COLOR: {
			r_dst = reinterpret_cast<const Color *>(a._data._mem)->linear_interpolate(*reinterpret_cast<const Color *>(b._data._mem), c);
		}
			return;
		case IMAGE: {
			r_dst = a;
		}
			return;
		case NODE_PATH: {
			r_dst = a;
		}
			return;
		case _RID: {
			r_dst = a;
		}
			return;
		case OBJECT: {
			r_dst = a;
		}
			return;
		case INPUT_EVENT: {
			r_dst = a;
		}
			return;
		case DICTIONARY: {
		}
			return;
		case ARRAY: {
			r_dst = a;
		}
			return;
		case POOL_BYTE_ARRAY: {
			r_dst = a;
		}
			return;
		case POOL_INT_ARRAY: {
			r_dst = a;
		}
			return;
		case POOL_REAL_ARRAY: {
			r_dst = a;
		}
			return;
		case POOL_STRING_ARRAY: {
			r_dst = a;
		}
			return;
		case POOL_VECTOR2_ARRAY: {
			const PoolVector<Vector2> *arr_a = reinterpret_cast<const PoolVector<Vector2> *>(a._data._mem);
			const PoolVector<Vector2> *arr_b = reinterpret_cast<const PoolVector<Vector2> *>(b._data._mem);
			int sz = arr_a->size();
			if (sz == 0 || arr_b->size() != sz) {

				r_dst = a;
			} else {

				PoolVector<Vector2> v;
				v.resize(sz);
				{
					PoolVector<Vector2>::Write vw = v.write();
					PoolVector<Vector2>::Read ar = arr_a->read();
					PoolVector<Vector2>::Read br = arr_b->read();

					for (int i = 0; i < sz; i++) {
						vw[i] = ar[i].linear_interpolate(br[i], c);
					}
				}
				r_dst = v;
			}
		}
			return;
		case POOL_VECTOR3_ARRAY: {

			const PoolVector<Vector3> *arr_a = reinterpret_cast<const PoolVector<Vector3> *>(a._data._mem);
			const PoolVector<Vector3> *arr_b = reinterpret_cast<const PoolVector<Vector3> *>(b._data._mem);
			int sz = arr_a->size();
			if (sz == 0 || arr_b->size() != sz) {

				r_dst = a;
			} else {

				PoolVector<Vector3> v;
				v.resize(sz);
				{
					PoolVector<Vector3>::Write vw = v.write();
					PoolVector<Vector3>::Read ar = arr_a->read();
					PoolVector<Vector3>::Read br = arr_b->read();

					for (int i = 0; i < sz; i++) {
						vw[i] = ar[i].linear_interpolate(br[i], c);
					}
				}
				r_dst = v;
			}
		}
			return;
		case POOL_COLOR_ARRAY: {
			r_dst = a;
		}
			return;
		default: {

			r_dst = a;
		}
	}
}

static const char *_op_names[Variant::OP_MAX] = {
	"==",
	"!=",
	"<",
	"<=",
	">",
	">=",
	"+",
	"-",
	"*",
	"/",
	"- (negation)",
	"%",
	"..",
	"<<",
	">>",
	"&",
	"|",
	"^",
	"~",
	"and",
	"or",
	"xor",
	"not",
	"in"

};

String Variant::get_operator_name(Operator p_op) {

	ERR_FAIL_INDEX_V(p_op, OP_MAX, "");
	return _op_names[p_op];
}
