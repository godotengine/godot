/**************************************************************************/
/*  java_class_wrapper.cpp                                                */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "api/java_class_wrapper.h"

#include "jni_utils.h"
#include "thread_jandroid.h"

bool JavaClass::_call_method(JavaObject *p_instance, const StringName &p_method, const Variant **p_args, int p_argcount, Callable::CallError &r_error, Variant &ret) {
	HashMap<StringName, List<MethodInfo>>::Iterator M = methods.find(p_method);
	if (!M) {
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
		return false;
	}

	JNIEnv *env = get_jni_env();
	ERR_FAIL_NULL_V(env, false);

	env->PushLocalFrame(p_argcount);
	if (env->ExceptionCheck()) {
		env->ExceptionDescribe();
		env->ExceptionClear();
		return false;
	}

	MethodInfo *method = nullptr;
	for (MethodInfo &E : M->value) {
		if (!p_instance && !E._static && !E._constructor) {
			r_error.error = Callable::CallError::CALL_ERROR_INSTANCE_IS_NULL;
			continue;
		}

		int pc = E.param_types.size();
		if (p_argcount < pc) {
			r_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
			r_error.expected = pc;
			continue;
		}
		if (p_argcount > pc) {
			r_error.error = Callable::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS;
			r_error.expected = pc;
			continue;
		}
		uint32_t *ptypes = E.param_types.ptrw();
		bool valid = true;

		for (int i = 0; i < pc; i++) {
			Variant::Type arg_expected = Variant::NIL;
			switch (ptypes[i]) {
				case ARG_TYPE_VOID: {
					//bug?
				} break;
				case ARG_TYPE_BOOLEAN: {
					if (p_args[i]->get_type() != Variant::BOOL) {
						arg_expected = Variant::BOOL;
					}
				} break;
				case ARG_NUMBER_CLASS_BIT | ARG_TYPE_BYTE:
				case ARG_NUMBER_CLASS_BIT | ARG_TYPE_CHAR:
				case ARG_NUMBER_CLASS_BIT | ARG_TYPE_SHORT:
				case ARG_NUMBER_CLASS_BIT | ARG_TYPE_INT:
				case ARG_NUMBER_CLASS_BIT | ARG_TYPE_LONG:
				case ARG_TYPE_BYTE:
				case ARG_TYPE_CHAR:
				case ARG_TYPE_SHORT:
				case ARG_TYPE_INT:
				case ARG_TYPE_LONG: {
					if (!p_args[i]->is_num()) {
						arg_expected = Variant::INT;
					}
				} break;
				case ARG_NUMBER_CLASS_BIT | ARG_TYPE_FLOAT:
				case ARG_NUMBER_CLASS_BIT | ARG_TYPE_DOUBLE:
				case ARG_TYPE_FLOAT:
				case ARG_TYPE_DOUBLE: {
					if (!p_args[i]->is_num()) {
						arg_expected = Variant::FLOAT;
					}
				} break;
				case ARG_TYPE_STRING:
				case ARG_TYPE_CHARSEQUENCE: {
					if (!p_args[i]->is_string()) {
						arg_expected = Variant::STRING;
					}
				} break;
				case ARG_TYPE_CALLABLE: {
					if (p_args[i]->get_type() != Variant::CALLABLE) {
						arg_expected = Variant::CALLABLE;
					}
				} break;
				case ARG_TYPE_CLASS: {
					String cn = E.param_sigs[i].string();
					if (cn.begins_with("L") && cn.ends_with(";")) {
						cn = cn.substr(1, cn.length() - 2);
					}
					if (cn == "org/godotengine/godot/Dictionary") {
						if (p_args[i]->get_type() != Variant::DICTIONARY) {
							arg_expected = Variant::DICTIONARY;
						}
					} else if (p_args[i]->get_type() != Variant::OBJECT && p_args[i]->get_type() != Variant::NIL) {
						arg_expected = Variant::OBJECT;
					} else {
						Ref<RefCounted> ref = *p_args[i];
						if (ref.is_valid()) {
							if (Object::cast_to<JavaObject>(ref.ptr())) {
								Ref<JavaObject> jo = ref;
								jclass c = jni_find_class(env, cn.utf8().get_data());
								if (!c || !env->IsInstanceOf(jo->instance, c)) {
									arg_expected = Variant::OBJECT;
								} else {
									//ok
								}
							} else {
								arg_expected = Variant::OBJECT;
							}
						}
					}
				} break;
				case ARG_ARRAY_BIT | ARG_TYPE_BOOLEAN: {
					if (p_args[i]->get_type() == Variant::ARRAY) {
						Array arr = *p_args[i];
						if (arr.is_typed() && arr.get_typed_builtin() != Variant::BOOL) {
							arg_expected = Variant::ARRAY;
						}
					} else {
						arg_expected = Variant::ARRAY;
					}
				} break;
				case ARG_ARRAY_BIT | ARG_TYPE_BYTE:
				case ARG_ARRAY_BIT | ARG_TYPE_CHAR: {
					if (p_args[i]->get_type() != Variant::PACKED_BYTE_ARRAY) {
						arg_expected = Variant::PACKED_BYTE_ARRAY;
					}
				} break;
				case ARG_ARRAY_BIT | ARG_TYPE_SHORT:
				case ARG_ARRAY_BIT | ARG_TYPE_INT: {
					if (p_args[i]->get_type() == Variant::ARRAY) {
						Array arr = *p_args[i];
						if (arr.is_typed() && arr.get_typed_builtin() != Variant::INT) {
							arg_expected = Variant::ARRAY;
						}
					} else if (p_args[i]->get_type() != Variant::PACKED_INT32_ARRAY) {
						arg_expected = Variant::ARRAY;
					}
				} break;
				case ARG_ARRAY_BIT | ARG_TYPE_LONG: {
					if (p_args[i]->get_type() == Variant::ARRAY) {
						Array arr = *p_args[i];
						if (arr.is_typed() && arr.get_typed_builtin() != Variant::INT) {
							arg_expected = Variant::ARRAY;
						}
					} else if (p_args[i]->get_type() != Variant::PACKED_INT64_ARRAY) {
						arg_expected = Variant::ARRAY;
					}
				} break;
				case ARG_ARRAY_BIT | ARG_TYPE_FLOAT: {
					if (p_args[i]->get_type() == Variant::ARRAY) {
						Array arr = *p_args[i];
						if (arr.is_typed() && arr.get_typed_builtin() != Variant::FLOAT) {
							arg_expected = Variant::ARRAY;
						}
					} else if (p_args[i]->get_type() != Variant::PACKED_FLOAT32_ARRAY) {
						arg_expected = Variant::ARRAY;
					}
				} break;
				case ARG_ARRAY_BIT | ARG_TYPE_DOUBLE: {
					if (p_args[i]->get_type() == Variant::ARRAY) {
						Array arr = *p_args[i];
						if (arr.is_typed() && arr.get_typed_builtin() != Variant::FLOAT) {
							arg_expected = Variant::ARRAY;
						}
					} else if (p_args[i]->get_type() != Variant::PACKED_FLOAT64_ARRAY) {
						arg_expected = Variant::ARRAY;
					}
				} break;
				case ARG_ARRAY_BIT | ARG_TYPE_STRING:
				case ARG_ARRAY_BIT | ARG_TYPE_CHARSEQUENCE: {
					if (p_args[i]->get_type() == Variant::ARRAY) {
						Array arr = *p_args[i];
						if (arr.is_typed() && (arr.get_typed_builtin() != Variant::STRING && arr.get_typed_builtin() != Variant::STRING_NAME)) {
							arg_expected = Variant::ARRAY;
						}
					} else if (p_args[i]->get_type() != Variant::PACKED_STRING_ARRAY) {
						arg_expected = Variant::ARRAY;
					}
				} break;
				case ARG_ARRAY_BIT | ARG_TYPE_CALLABLE: {
					if (p_args[i]->get_type() == Variant::ARRAY) {
						Array arr = *p_args[i];
						if (arr.is_typed() && arr.get_typed_builtin() != Variant::CALLABLE) {
							arg_expected = Variant::ARRAY;
						}
					} else {
						arg_expected = Variant::ARRAY;
					}
				} break;
				case ARG_ARRAY_BIT | ARG_TYPE_CLASS: {
					if (p_args[i]->get_type() == Variant::ARRAY) {
						Array arr = *p_args[i];
						if (arr.is_typed() && arr.get_typed_builtin() != Variant::OBJECT) {
							arg_expected = Variant::ARRAY;
						} else {
							String cn = E.param_sigs[i].string();
							if (cn.begins_with("[L") && cn.ends_with(";")) {
								cn = cn.substr(2, cn.length() - 3);
							}
							jclass c = jni_find_class(env, cn.utf8().get_data());
							if (c) {
								for (int j = 0; j < arr.size(); j++) {
									Ref<JavaObject> jo = arr[j];
									if (jo.is_valid()) {
										if (!env->IsInstanceOf(jo->instance, c)) {
											arg_expected = Variant::ARRAY;
											break;
										}
									} else {
										arg_expected = Variant::ARRAY;
										break;
									}
								}
							} else {
								arg_expected = Variant::ARRAY;
							}
						}
					} else {
						arg_expected = Variant::ARRAY;
					}
				} break;
				default: {
					if (p_args[i]->get_type() != Variant::ARRAY) {
						arg_expected = Variant::ARRAY;
					}
				} break;
			}

			if (arg_expected != Variant::NIL) {
				r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_error.argument = i;
				r_error.expected = arg_expected;
				valid = false;
				break;
			}
		}
		if (!valid) {
			continue;
		}

		method = &E;
		break;
	}

	if (!method) {
		if (r_error.error == Callable::CallError::CALL_ERROR_INSTANCE_IS_NULL) {
			ERR_PRINT(vformat(R"(Cannot call static function "%s" on Java class "%s" directly. Make an instance instead.)", p_method, java_class_name));
		}
		return true; //no version convinces
	}

	r_error.error = Callable::CallError::CALL_OK;

	jvalue *argv = nullptr;

	if (method->param_types.size()) {
		argv = (jvalue *)alloca(sizeof(jvalue) * method->param_types.size());
	}

	for (int i = 0; i < method->param_types.size(); i++) {
		_convert_variant_to_jvalue(env, *p_args[i], method->param_types[i], method->param_sigs[i], argv[i]);
	}

	r_error.error = Callable::CallError::CALL_OK;
	bool success = true;

	switch (method->return_type) {
		case ARG_TYPE_VOID: {
			if (method->_static) {
				env->CallStaticVoidMethodA(_class, method->method, argv);
			} else {
				env->CallVoidMethodA(p_instance->instance, method->method, argv);
			}
			ret = Variant();

		} break;
		case ARG_TYPE_BOOLEAN: {
			if (method->_static) {
				ret = env->CallStaticBooleanMethodA(_class, method->method, argv);
			} else {
				ret = env->CallBooleanMethodA(p_instance->instance, method->method, argv);
			}
		} break;
		case ARG_TYPE_BYTE: {
			if (method->_static) {
				ret = env->CallStaticByteMethodA(_class, method->method, argv);
			} else {
				ret = env->CallByteMethodA(p_instance->instance, method->method, argv);
			}
		} break;
		case ARG_TYPE_CHAR: {
			if (method->_static) {
				ret = env->CallStaticCharMethodA(_class, method->method, argv);
			} else {
				ret = env->CallCharMethodA(p_instance->instance, method->method, argv);
			}
		} break;
		case ARG_TYPE_SHORT: {
			if (method->_static) {
				ret = env->CallStaticShortMethodA(_class, method->method, argv);
			} else {
				ret = env->CallShortMethodA(p_instance->instance, method->method, argv);
			}

		} break;
		case ARG_TYPE_INT: {
			if (method->_static) {
				ret = env->CallStaticIntMethodA(_class, method->method, argv);
			} else {
				ret = env->CallIntMethodA(p_instance->instance, method->method, argv);
			}

		} break;
		case ARG_TYPE_LONG: {
			if (method->_static) {
				ret = (int64_t)env->CallStaticLongMethodA(_class, method->method, argv);
			} else {
				ret = (int64_t)env->CallLongMethodA(p_instance->instance, method->method, argv);
			}

		} break;
		case ARG_TYPE_FLOAT: {
			if (method->_static) {
				ret = env->CallStaticFloatMethodA(_class, method->method, argv);
			} else {
				ret = env->CallFloatMethodA(p_instance->instance, method->method, argv);
			}

		} break;
		case ARG_TYPE_DOUBLE: {
			if (method->_static) {
				ret = env->CallStaticDoubleMethodA(_class, method->method, argv);
			} else {
				ret = env->CallDoubleMethodA(p_instance->instance, method->method, argv);
			}

		} break;
		default: {
			jobject obj;
			if (method->_constructor) {
				obj = env->NewObjectA(_class, method->method, argv);
			} else if (method->_static) {
				obj = env->CallStaticObjectMethodA(_class, method->method, argv);
			} else {
				obj = env->CallObjectMethodA(p_instance->instance, method->method, argv);
			}

			if (!obj) {
				ret = Variant();
			} else {
				if (!_convert_object_to_variant(env, obj, ret, method->return_type)) {
					ret = Variant();
					r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
					success = false;
				}
			}

		} break;
	}

	jobject exception = env->ExceptionOccurred();
	if (exception) {
		env->ExceptionClear();

		jclass java_class = env->GetObjectClass(exception);
		Ref<JavaClass> java_class_wrapped = JavaClassWrapper::singleton->wrap_jclass(java_class);
		env->DeleteLocalRef(java_class);

		JavaClassWrapper::singleton->exception.instantiate(java_class_wrapped, exception);
	} else {
		JavaClassWrapper::singleton->exception.unref();
	}

	env->PopLocalFrame(nullptr);

	return success;
}

bool JavaClass::_get(const StringName &p_name, Variant &r_ret) const {
	// Godot properties take precedence.
	if (RefCounted::_get(p_name, r_ret)) {
		return true;
	}

	return _get_field(nullptr, p_name, r_ret);
}

bool JavaClass::_set(const StringName &p_name, const Variant &p_property) {
	// Godot properties take precedence.
	if (RefCounted::_set(p_name, p_property)) {
		return true;
	}

	return _set_field(nullptr, p_name, p_property);
}

bool JavaClass::_get_field_value(JNIEnv *p_env, jobject p_instance, jclass p_clazz, const JavaClass::FieldInfo &p_field_info, Variant &r_ret) {
	ERR_FAIL_NULL_V(p_env, false);
	ERR_FAIL_COND_V(p_field_info._static && !p_clazz, false);
	ERR_FAIL_COND_V(!p_field_info._static && !p_instance, false);

	bool result = true;
	switch (p_field_info.field_type) {
		case ARG_TYPE_BOOLEAN:
			r_ret = p_field_info._static
					? (bool)p_env->GetStaticBooleanField(p_clazz, p_field_info.field)
					: (bool)p_env->GetBooleanField(p_instance, p_field_info.field);
			break;
		case ARG_TYPE_BYTE:
			r_ret = p_field_info._static
					? p_env->GetStaticByteField(p_clazz, p_field_info.field)
					: p_env->GetByteField(p_instance, p_field_info.field);
			break;
		case ARG_TYPE_CHAR:
			r_ret = p_field_info._static
					? p_env->GetStaticCharField(p_clazz, p_field_info.field)
					: p_env->GetCharField(p_instance, p_field_info.field);
			break;
		case ARG_TYPE_SHORT:
			r_ret = p_field_info._static
					? p_env->GetStaticShortField(p_clazz, p_field_info.field)
					: p_env->GetShortField(p_instance, p_field_info.field);
			break;
		case ARG_TYPE_INT:
			r_ret = p_field_info._static
					? p_env->GetStaticIntField(p_clazz, p_field_info.field)
					: p_env->GetIntField(p_instance, p_field_info.field);
			break;
		case ARG_TYPE_LONG:
			r_ret = p_field_info._static
					? (int64_t)p_env->GetStaticLongField(p_clazz, p_field_info.field)
					: (int64_t)p_env->GetLongField(p_instance, p_field_info.field);
			break;
		case ARG_TYPE_FLOAT:
			r_ret = p_field_info._static
					? p_env->GetStaticFloatField(p_clazz, p_field_info.field)
					: p_env->GetFloatField(p_instance, p_field_info.field);
			break;
		case ARG_TYPE_DOUBLE:
			r_ret = p_field_info._static
					? p_env->GetStaticDoubleField(p_clazz, p_field_info.field)
					: p_env->GetDoubleField(p_instance, p_field_info.field);
			break;
		default:
			jobject obj_value = p_field_info._static
					? p_env->GetStaticObjectField(p_clazz, p_field_info.field)
					: p_env->GetObjectField(p_instance, p_field_info.field);
			if (!_convert_object_to_variant(p_env, obj_value, r_ret, p_field_info.field_type)) {
				result = false;
			}
			p_env->DeleteLocalRef(obj_value);
			break;
	}

	if (p_env->ExceptionCheck()) {
		p_env->ExceptionDescribe();
		p_env->ExceptionClear();
		return false;
	}

	return result;
}

bool JavaClass::_set_field_value(JNIEnv *p_env, jobject p_instance, jclass p_clazz, const JavaClass::FieldInfo &p_field_info, const Variant &p_property) {
	ERR_FAIL_NULL_V(p_env, false);
	ERR_FAIL_COND_V(p_field_info._static && !p_clazz, false);
	ERR_FAIL_COND_V(!p_field_info._static && !p_instance, false);

	if (p_field_info._final) {
		// Cannot update final fields.
		return false;
	}

	switch (p_field_info.field_type) {
		case ARG_TYPE_BOOLEAN:
			if (p_field_info._static) {
				p_env->SetStaticBooleanField(p_clazz, p_field_info.field, p_property);
			} else {
				p_env->SetBooleanField(p_instance, p_field_info.field, p_property);
			}
			break;
		case ARG_TYPE_BYTE:
			if (p_field_info._static) {
				p_env->SetStaticByteField(p_clazz, p_field_info.field, p_property);
			} else {
				p_env->SetByteField(p_instance, p_field_info.field, p_property);
			}
			break;
		case ARG_TYPE_CHAR:
			if (p_field_info._static) {
				p_env->SetStaticCharField(p_clazz, p_field_info.field, p_property);
			} else {
				p_env->SetCharField(p_instance, p_field_info.field, p_property);
			}
			break;
		case ARG_TYPE_SHORT:
			if (p_field_info._static) {
				p_env->SetStaticShortField(p_clazz, p_field_info.field, p_property);
			} else {
				p_env->SetShortField(p_instance, p_field_info.field, p_property);
			}
			break;
		case ARG_TYPE_INT:
			if (p_field_info._static) {
				p_env->SetStaticIntField(p_clazz, p_field_info.field, p_property);
			} else {
				p_env->SetIntField(p_instance, p_field_info.field, p_property);
			}
			break;
		case ARG_TYPE_LONG:
			if (p_field_info._static) {
				p_env->SetStaticLongField(p_clazz, p_field_info.field, p_property);
			} else {
				p_env->SetLongField(p_instance, p_field_info.field, p_property);
			}
			break;
		case ARG_TYPE_FLOAT:
			if (p_field_info._static) {
				p_env->SetStaticFloatField(p_clazz, p_field_info.field, p_property);
			} else {
				p_env->SetFloatField(p_instance, p_field_info.field, p_property);
			}
			break;
		case ARG_TYPE_DOUBLE:
			if (p_field_info._static) {
				p_env->SetStaticDoubleField(p_clazz, p_field_info.field, p_property);
			} else {
				p_env->SetDoubleField(p_instance, p_field_info.field, p_property);
			}
			break;
		default:
			jvalue val;
			_convert_variant_to_jvalue(p_env, p_property, p_field_info.field_type, p_field_info.field_sig, val);
			jobject obj_value = val.l;
			if (p_field_info._static) {
				p_env->SetStaticObjectField(p_clazz, p_field_info.field, obj_value);
			} else {
				p_env->SetObjectField(p_instance, p_field_info.field, obj_value);
			}
			p_env->DeleteLocalRef(obj_value);
			break;
	}

	if (p_env->ExceptionCheck()) {
		p_env->ExceptionDescribe();
		p_env->ExceptionClear();
		return false;
	}

	return true;
}

bool JavaClass::_get_field(jobject p_instance, const StringName &p_name, Variant &r_ret) const {
	if (fields.has(p_name)) {
		const JavaClass::FieldInfo field_info = fields[p_name];
		if (field_info._static && field_info._final) {
			r_ret = field_info.constant_value;
			return true;
		}

		JNIEnv *env = get_jni_env();
		ERR_FAIL_NULL_V(env, false);
		return _get_field_value(env, p_instance, _class, field_info, r_ret);
	}

	// Check the parent class.
	Ref<JavaClass> parent_class = get_java_parent_class();
	if (parent_class.is_valid()) {
		return parent_class->_get_field(p_instance, p_name, r_ret);
	}

	return false;
}

bool JavaClass::_set_field(jobject p_instance, const StringName &p_name, const Variant &p_property) {
	if (fields.has(p_name)) {
		const JavaClass::FieldInfo field_info = fields[p_name];

		JNIEnv *env = get_jni_env();
		ERR_FAIL_NULL_V(env, false);
		return _set_field_value(env, p_instance, _class, field_info, p_property);
	}

	// Check the parent class.
	Ref<JavaClass> parent_class = get_java_parent_class();
	if (parent_class.is_valid()) {
		return parent_class->_set_field(p_instance, p_name, p_property);
	}

	return false;
}

Variant JavaClass::callp(const StringName &p_method, const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	// Godot methods take precedence.
	Variant ret = RefCounted::callp(p_method, p_args, p_argcount, r_error);
	if (r_error.error == Callable::CallError::CALL_OK) {
		return ret;
	}

	String method = (p_method == java_constructor_name) ? "<init>" : p_method;
	bool found = _call_method(nullptr, method, p_args, p_argcount, r_error, ret);
	if (found) {
		return ret;
	}

	return Variant();
}

String JavaClass::get_java_class_name() const {
	return java_class_name;
}

TypedArray<Dictionary> JavaClass::get_java_method_list() const {
	TypedArray<Dictionary> method_list;

	for (const KeyValue<StringName, List<MethodInfo>> &item : methods) {
		for (const MethodInfo &mi : item.value) {
			Dictionary method;

			method["name"] = mi._constructor ? java_constructor_name : String(item.key);
			method["id"] = (uint64_t)mi.method;
			method["default_args"] = Array();
			method["flags"] = METHOD_FLAGS_DEFAULT & (mi._static || mi._constructor ? METHOD_FLAG_STATIC : METHOD_FLAG_NORMAL);

			{
				Array a;

				for (uint32_t argtype : mi.param_types) {
					Dictionary d;

					Variant::Type t = Variant::NIL;
					float likelihood = 0.0;
					_convert_to_variant_type(argtype, t, likelihood);
					d["type"] = t;
					if (t == Variant::OBJECT) {
						d["hint"] = PROPERTY_HINT_RESOURCE_TYPE;
						d["hint_string"] = "JavaObject";
					} else {
						d["hint"] = 0;
						d["hint_string"] = "";
					}

					a.push_back(d);
				}

				method["args"] = a;
			}

			{
				Dictionary d;

				if (mi._constructor) {
					d["type"] = Variant::OBJECT;
					d["hint"] = PROPERTY_HINT_RESOURCE_TYPE;
					d["hint_string"] = "JavaObject";
				} else {
					Variant::Type t = Variant::NIL;
					float likelihood = 0.0;
					_convert_to_variant_type(mi.return_type, t, likelihood);
					d["type"] = t;
					if (t == Variant::OBJECT) {
						d["hint"] = PROPERTY_HINT_RESOURCE_TYPE;
						d["hint_string"] = "JavaObject";
					} else {
						d["hint"] = 0;
						d["hint_string"] = "";
					}
				}

				method["return_type"] = d;
			}

			method_list.push_back(method);
		}
	}

	return method_list;
}

Ref<JavaClass> JavaClass::get_java_parent_class() const {
	ERR_FAIL_NULL_V(_class, Ref<JavaClass>());

	JNIEnv *env = get_jni_env();
	ERR_FAIL_NULL_V(env, Ref<JavaClass>());

	jclass superclass = (jclass)env->CallObjectMethod(_class, JavaClassWrapper::singleton->Class_getSuperclass);
	if (!superclass) {
		return Ref<JavaClass>();
	}

	Ref<JavaClass> ret = JavaClassWrapper::singleton->wrap_jclass(superclass);
	env->DeleteLocalRef(superclass);
	return ret;
}

String JavaClass::_to_string() {
	return "<JavaClass:" + java_class_name + ">";
}

bool JavaClass::has_java_method(const StringName &p_method) const {
	String method = (p_method == java_constructor_name) ? "<init>" : p_method;
	return methods.has(method);
}

JavaClass::JavaClass() {
}

JavaClass::~JavaClass() {
	if (_class) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_NULL(env);

		env->DeleteGlobalRef(_class);
	}
}

/////////////////////

Variant JavaObject::callp(const StringName &p_method, const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	// Godot methods take precedence.
	Variant ret = RefCounted::callp(p_method, p_args, p_argcount, r_error);
	if (r_error.error == Callable::CallError::CALL_OK) {
		return ret;
	}

	if (instance) {
		Ref<JavaClass> c = base_class;
		while (c.is_valid()) {
			bool found = c->_call_method(this, p_method, p_args, p_argcount, r_error, ret);
			if (found) {
				return ret;
			}
			c = c->get_java_parent_class();
		}
	}

	return Variant();
}

Ref<JavaClass> JavaObject::get_java_class() const {
	return base_class;
}

bool JavaObject::_get(const StringName &p_name, Variant &r_ret) const {
	// Godot properties take precedence.
	if (RefCounted::_get(p_name, r_ret)) {
		return true;
	}

	if (base_class.is_valid()) {
		return base_class->_get_field(instance, p_name, r_ret);
	}

	return false;
}

bool JavaObject::_set(const StringName &p_name, const Variant &p_property) {
	// Godot properties take precedence.
	if (RefCounted::_set(p_name, p_property)) {
		return true;
	}

	if (base_class.is_valid()) {
		return base_class->_set_field(instance, p_name, p_property);
	}

	return false;
}

String JavaObject::_to_string() {
	if (base_class.is_valid() && instance) {
		return "<JavaObject:" + base_class->java_class_name + " \"" + (String)call("toString") + "\">";
	}
	return RefCounted::_to_string();
}

bool JavaObject::has_java_method(const StringName &p_method) const {
	if (instance) {
		Ref<JavaClass> c = base_class;
		while (c.is_valid()) {
			if (c->has_java_method(p_method)) {
				return true;
			}
			c = c->get_java_parent_class();
		}
	}
	return false;
}

JavaObject::JavaObject() {
}

JavaObject::JavaObject(const Ref<JavaClass> &p_base, jobject p_instance) {
	JNIEnv *env = get_jni_env();
	ERR_FAIL_NULL(env);

	base_class = p_base;
	instance = env->NewGlobalRef(p_instance);
}

JavaObject::~JavaObject() {
	if (instance) {
		JNIEnv *env = get_jni_env();
		ERR_FAIL_NULL(env);

		env->DeleteGlobalRef(instance);
	}
}

////////////////////

bool JavaClassWrapper::_get_type_sig(JNIEnv *env, jobject obj, uint32_t &sig, String &strsig) {
	jstring name2 = (jstring)env->CallObjectMethod(obj, Class_getName);
	String str_type = jstring_to_string(name2, env);
	env->DeleteLocalRef(name2);
	uint32_t t = 0;

	if (str_type.begins_with("[")) {
		t = JavaClass::ARG_ARRAY_BIT;
		strsig = "[";
		str_type = str_type.substr(1);
		if (str_type.begins_with("[")) {
			print_line("Nested arrays not supported for type: " + str_type);
			return false;
		}
		if (str_type.begins_with("L")) {
			str_type = str_type.substr(1, str_type.length() - 2); //ok it's a class
		}
	}

	if (str_type == "void" || str_type == "V") {
		t |= JavaClass::ARG_TYPE_VOID;
		strsig += "V";
	} else if (str_type == "boolean" || str_type == "Z") {
		t |= JavaClass::ARG_TYPE_BOOLEAN;
		strsig += "Z";
	} else if (str_type == "byte" || str_type == "B") {
		t |= JavaClass::ARG_TYPE_BYTE;
		strsig += "B";
	} else if (str_type == "char" || str_type == "C") {
		t |= JavaClass::ARG_TYPE_CHAR;
		strsig += "C";
	} else if (str_type == "short" || str_type == "S") {
		t |= JavaClass::ARG_TYPE_SHORT;
		strsig += "S";
	} else if (str_type == "int" || str_type == "I") {
		t |= JavaClass::ARG_TYPE_INT;
		strsig += "I";
	} else if (str_type == "long" || str_type == "J") {
		t |= JavaClass::ARG_TYPE_LONG;
		strsig += "J";
	} else if (str_type == "float" || str_type == "F") {
		t |= JavaClass::ARG_TYPE_FLOAT;
		strsig += "F";
	} else if (str_type == "double" || str_type == "D") {
		t |= JavaClass::ARG_TYPE_DOUBLE;
		strsig += "D";
	} else if (str_type == "java.lang.String") {
		t |= JavaClass::ARG_TYPE_STRING;
		strsig += "Ljava/lang/String;";
	} else if (str_type == "java.lang.CharSequence") {
		t |= JavaClass::ARG_TYPE_CHARSEQUENCE;
		strsig += "Ljava/lang/CharSequence;";
	} else if (str_type == "org.godotengine.godot.variant.Callable") {
		t |= JavaClass::ARG_TYPE_CALLABLE;
		strsig += "Lorg/godotengine/godot/variant/Callable;";
	} else if (str_type == "java.lang.Boolean") {
		t |= JavaClass::ARG_TYPE_BOOLEAN | JavaClass::ARG_NUMBER_CLASS_BIT;
		strsig += "Ljava/lang/Boolean;";
	} else if (str_type == "java.lang.Byte") {
		t |= JavaClass::ARG_TYPE_BYTE | JavaClass::ARG_NUMBER_CLASS_BIT;
		strsig += "Ljava/lang/Byte;";
	} else if (str_type == "java.lang.Character") {
		t |= JavaClass::ARG_TYPE_CHAR | JavaClass::ARG_NUMBER_CLASS_BIT;
		strsig += "Ljava/lang/Character;";
	} else if (str_type == "java.lang.Short") {
		t |= JavaClass::ARG_TYPE_SHORT | JavaClass::ARG_NUMBER_CLASS_BIT;
		strsig += "Ljava/lang/Short;";
	} else if (str_type == "java.lang.Integer") {
		t |= JavaClass::ARG_TYPE_INT | JavaClass::ARG_NUMBER_CLASS_BIT;
		strsig += "Ljava/lang/Integer;";
	} else if (str_type == "java.lang.Long") {
		t |= JavaClass::ARG_TYPE_LONG | JavaClass::ARG_NUMBER_CLASS_BIT;
		strsig += "Ljava/lang/Long;";
	} else if (str_type == "java.lang.Float") {
		t |= JavaClass::ARG_TYPE_FLOAT | JavaClass::ARG_NUMBER_CLASS_BIT;
		strsig += "Ljava/lang/Float;";
	} else if (str_type == "java.lang.Double") {
		t |= JavaClass::ARG_TYPE_DOUBLE | JavaClass::ARG_NUMBER_CLASS_BIT;
		strsig += "Ljava/lang/Double;";
	} else {
		//a class likely
		strsig += "L" + str_type.replace_char('.', '/') + ";";
		t |= JavaClass::ARG_TYPE_CLASS;
	}

	sig = t;

	return true;
}

bool JavaClass::_convert_variant_to_jvalue(JNIEnv *p_env, const Variant &p_variant, uint32_t p_sig, const StringName &p_strsig, jvalue &r_ret) {
	switch (p_sig) {
		case ARG_TYPE_VOID: {
			r_ret.l = nullptr;
		} break;

		case ARG_TYPE_BOOLEAN: {
			r_ret.z = p_variant;
		} break;
		case ARG_TYPE_BYTE: {
			r_ret.b = p_variant;
		} break;
		case ARG_TYPE_CHAR: {
			r_ret.c = p_variant;
		} break;
		case ARG_TYPE_SHORT: {
			r_ret.s = p_variant;
		} break;
		case ARG_TYPE_INT: {
			r_ret.i = p_variant;
		} break;
		case ARG_TYPE_LONG: {
			r_ret.j = (int64_t)p_variant;
		} break;
		case ARG_TYPE_FLOAT: {
			r_ret.f = p_variant;
		} break;
		case ARG_TYPE_DOUBLE: {
			r_ret.d = p_variant;
		} break;
		case ARG_NUMBER_CLASS_BIT | ARG_TYPE_BOOLEAN: {
			jclass bclass = jni_find_class(p_env, "java/lang/Boolean");
			jmethodID ctor = p_env->GetMethodID(bclass, "<init>", "(Z)V");
			jvalue val;
			val.z = (bool)(p_variant);
			jobject obj = p_env->NewObjectA(bclass, ctor, &val);
			r_ret.l = obj;
			p_env->DeleteLocalRef(bclass);
		} break;
		case ARG_NUMBER_CLASS_BIT | ARG_TYPE_BYTE: {
			jclass bclass = jni_find_class(p_env, "java/lang/Byte");
			jmethodID ctor = p_env->GetMethodID(bclass, "<init>", "(B)V");
			jvalue val;
			val.b = (int)(p_variant);
			jobject obj = p_env->NewObjectA(bclass, ctor, &val);
			r_ret.l = obj;
			p_env->DeleteLocalRef(bclass);
		} break;
		case ARG_NUMBER_CLASS_BIT | ARG_TYPE_CHAR: {
			jclass bclass = jni_find_class(p_env, "java/lang/Character");
			jmethodID ctor = p_env->GetMethodID(bclass, "<init>", "(C)V");
			jvalue val;
			val.c = (int)(p_variant);
			jobject obj = p_env->NewObjectA(bclass, ctor, &val);
			r_ret.l = obj;
			p_env->DeleteLocalRef(bclass);
		} break;
		case ARG_NUMBER_CLASS_BIT | ARG_TYPE_SHORT: {
			jclass bclass = jni_find_class(p_env, "java/lang/Short");
			jmethodID ctor = p_env->GetMethodID(bclass, "<init>", "(S)V");
			jvalue val;
			val.s = (int)(p_variant);
			jobject obj = p_env->NewObjectA(bclass, ctor, &val);
			r_ret.l = obj;
			p_env->DeleteLocalRef(bclass);
		} break;
		case ARG_NUMBER_CLASS_BIT | ARG_TYPE_INT: {
			jclass bclass = jni_find_class(p_env, "java/lang/Integer");
			jmethodID ctor = p_env->GetMethodID(bclass, "<init>", "(I)V");
			jvalue val;
			val.i = (int)(p_variant);
			jobject obj = p_env->NewObjectA(bclass, ctor, &val);
			r_ret.l = obj;
			p_env->DeleteLocalRef(bclass);
		} break;
		case ARG_NUMBER_CLASS_BIT | ARG_TYPE_LONG: {
			jclass bclass = jni_find_class(p_env, "java/lang/Long");
			jmethodID ctor = p_env->GetMethodID(bclass, "<init>", "(J)V");
			jvalue val;
			val.j = (int64_t)(p_variant);
			jobject obj = p_env->NewObjectA(bclass, ctor, &val);
			r_ret.l = obj;
			p_env->DeleteLocalRef(bclass);
		} break;
		case ARG_NUMBER_CLASS_BIT | ARG_TYPE_FLOAT: {
			jclass bclass = jni_find_class(p_env, "java/lang/Float");
			jmethodID ctor = p_env->GetMethodID(bclass, "<init>", "(F)V");
			jvalue val;
			val.f = (float)(p_variant);
			jobject obj = p_env->NewObjectA(bclass, ctor, &val);
			r_ret.l = obj;
			p_env->DeleteLocalRef(bclass);
		} break;
		case ARG_NUMBER_CLASS_BIT | ARG_TYPE_DOUBLE: {
			jclass bclass = jni_find_class(p_env, "java/lang/Double");
			jmethodID ctor = p_env->GetMethodID(bclass, "<init>", "(D)V");
			jvalue val;
			val.d = (double)(p_variant);
			jobject obj = p_env->NewObjectA(bclass, ctor, &val);
			r_ret.l = obj;
			p_env->DeleteLocalRef(bclass);
		} break;
		case ARG_TYPE_STRING:
		case ARG_TYPE_CHARSEQUENCE: {
			String s = p_variant;
			jstring jStr = p_env->NewStringUTF(s.utf8().get_data());
			r_ret.l = jStr;
		} break;
		case ARG_TYPE_CALLABLE: {
			jobject jcallable = callable_to_jcallable(p_env, p_variant);
			r_ret.l = jcallable;
		} break;
		case ARG_TYPE_CLASS: {
			if (p_variant.get_type() == Variant::DICTIONARY) {
				r_ret.l = _variant_to_jobject(p_env, Variant::DICTIONARY, &p_variant);
			} else {
				Ref<JavaObject> jo = p_variant;
				if (jo.is_valid()) {
					r_ret.l = jo->instance;
				} else {
					r_ret.l = nullptr; //I hope this works
				}
			}
		} break;
		case ARG_ARRAY_BIT | ARG_TYPE_BOOLEAN: {
			Array arr = p_variant;
			jbooleanArray a = p_env->NewBooleanArray(arr.size());
			for (int j = 0; j < arr.size(); j++) {
				jboolean val = arr[j];
				p_env->SetBooleanArrayRegion(a, j, 1, &val);
			}
			r_ret.l = a;
		} break;
		case ARG_ARRAY_BIT | ARG_TYPE_BYTE: {
			jbyteArray a = nullptr;

			if (p_variant.get_type() == Variant::ARRAY) {
				Array arr = p_variant;
				a = p_env->NewByteArray(arr.size());
				for (int j = 0; j < arr.size(); j++) {
					jbyte val = arr[j];
					p_env->SetByteArrayRegion(a, j, 1, &val);
				}
			} else if (p_variant.get_type() == Variant::PACKED_BYTE_ARRAY) {
				PackedByteArray arr = p_variant;
				a = p_env->NewByteArray(arr.size());
				p_env->SetByteArrayRegion(a, 0, arr.size(), (const jbyte *)arr.ptr());
			}

			r_ret.l = a;
		} break;
		case ARG_ARRAY_BIT | ARG_TYPE_CHAR: {
			jcharArray a = nullptr;

			if (p_variant.get_type() == Variant::ARRAY) {
				Array arr = p_variant;
				a = p_env->NewCharArray(arr.size());
				for (int j = 0; j < arr.size(); j++) {
					jchar val = arr[j];
					p_env->SetCharArrayRegion(a, j, 1, &val);
				}
			} else if (p_variant.get_type() == Variant::PACKED_BYTE_ARRAY) {
				PackedByteArray arr = p_variant;
				// The data is expected to be UTF-16 encoded, so the length is half the size of the byte array.
				int size = arr.size() / 2;
				a = p_env->NewCharArray(size);
				p_env->SetCharArrayRegion(a, 0, size, (const jchar *)arr.ptr());
			}

			r_ret.l = a;

		} break;
		case ARG_ARRAY_BIT | ARG_TYPE_SHORT: {
			jshortArray a = nullptr;

			if (p_variant.get_type() == Variant::ARRAY) {
				Array arr = p_variant;
				a = p_env->NewShortArray(arr.size());
				for (int j = 0; j < arr.size(); j++) {
					jshort val = arr[j];
					p_env->SetShortArrayRegion(a, j, 1, &val);
				}
			} else if (p_variant.get_type() == Variant::PACKED_INT32_ARRAY) {
				PackedInt32Array arr = p_variant;
				a = p_env->NewShortArray(arr.size());
				for (int j = 0; j < arr.size(); j++) {
					jshort val = arr[j];
					p_env->SetShortArrayRegion(a, j, 1, &val);
				}
			}

			r_ret.l = a;

		} break;
		case ARG_ARRAY_BIT | ARG_TYPE_INT: {
			jintArray a = nullptr;

			if (p_variant.get_type() == Variant::ARRAY) {
				Array arr = p_variant;
				a = p_env->NewIntArray(arr.size());
				for (int j = 0; j < arr.size(); j++) {
					jint val = arr[j];
					p_env->SetIntArrayRegion(a, j, 1, &val);
				}
			} else if (p_variant.get_type() == Variant::PACKED_INT32_ARRAY) {
				PackedInt32Array arr = p_variant;
				a = p_env->NewIntArray(arr.size());
				p_env->SetIntArrayRegion(a, 0, arr.size(), arr.ptr());
			}

			r_ret.l = a;
		} break;
		case ARG_ARRAY_BIT | ARG_TYPE_LONG: {
			jlongArray a = nullptr;

			if (p_variant.get_type() == Variant::ARRAY) {
				Array arr = p_variant;
				a = p_env->NewLongArray(arr.size());
				for (int j = 0; j < arr.size(); j++) {
					jlong val = (int64_t)arr[j];
					p_env->SetLongArrayRegion(a, j, 1, &val);
				}
			} else if (p_variant.get_type() == Variant::PACKED_INT64_ARRAY) {
				PackedInt64Array arr = p_variant;
				a = p_env->NewLongArray(arr.size());
				p_env->SetLongArrayRegion(a, 0, arr.size(), arr.ptr());
			}

			r_ret.l = a;
		} break;
		case ARG_ARRAY_BIT | ARG_TYPE_FLOAT: {
			jfloatArray a = nullptr;

			if (p_variant.get_type() == Variant::ARRAY) {
				Array arr = p_variant;
				a = p_env->NewFloatArray(arr.size());
				for (int j = 0; j < arr.size(); j++) {
					jfloat val = arr[j];
					p_env->SetFloatArrayRegion(a, j, 1, &val);
				}
			} else if (p_variant.get_type() == Variant::PACKED_FLOAT32_ARRAY) {
				PackedFloat32Array arr = p_variant;
				a = p_env->NewFloatArray(arr.size());
				p_env->SetFloatArrayRegion(a, 0, arr.size(), arr.ptr());
			}

			r_ret.l = a;
		} break;
		case ARG_ARRAY_BIT | ARG_TYPE_DOUBLE: {
			jdoubleArray a = nullptr;

			if (p_variant.get_type() == Variant::ARRAY) {
				Array arr = p_variant;
				a = p_env->NewDoubleArray(arr.size());
				for (int j = 0; j < arr.size(); j++) {
					jdouble val = arr[j];
					p_env->SetDoubleArrayRegion(a, j, 1, &val);
				}
			} else if (p_variant.get_type() == Variant::PACKED_FLOAT64_ARRAY) {
				PackedFloat64Array arr = p_variant;
				a = p_env->NewDoubleArray(arr.size());
				p_env->SetDoubleArrayRegion(a, 0, arr.size(), arr.ptr());
			}

			r_ret.l = a;
		} break;
		case ARG_ARRAY_BIT | ARG_TYPE_STRING:
		case ARG_ARRAY_BIT | ARG_TYPE_CHARSEQUENCE: {
			jobjectArray a = nullptr;

			if (p_variant.get_type() == Variant::ARRAY) {
				Array arr = p_variant;
				jclass sclass = jni_find_class(p_env, "java/lang/String");
				a = p_env->NewObjectArray(arr.size(), sclass, nullptr);
				p_env->DeleteLocalRef(sclass);
				for (int j = 0; j < arr.size(); j++) {
					String s = arr[j];
					jstring jStr = p_env->NewStringUTF(s.utf8().get_data());
					p_env->SetObjectArrayElement(a, j, jStr);
					p_env->DeleteLocalRef(jStr);
				}
			} else if (p_variant.get_type() == Variant::PACKED_STRING_ARRAY) {
				PackedStringArray arr = p_variant;
				jclass sclass = jni_find_class(p_env, "java/lang/String");
				a = p_env->NewObjectArray(arr.size(), sclass, nullptr);
				p_env->DeleteLocalRef(sclass);
				for (int j = 0; j < arr.size(); j++) {
					String s = arr[j];
					jstring jStr = p_env->NewStringUTF(s.utf8().get_data());
					p_env->SetObjectArrayElement(a, j, jStr);
					p_env->DeleteLocalRef(jStr);
				}
			}

			r_ret.l = a;
		} break;
		case ARG_ARRAY_BIT | ARG_TYPE_CALLABLE: {
			Array arr = p_variant;
			jclass cclass = jni_find_class(p_env, "org/godotengine/godot/variant/Callable");
			jobjectArray jarr = p_env->NewObjectArray(arr.size(), cclass, nullptr);
			p_env->DeleteLocalRef(cclass);
			for (int j = 0; j < arr.size(); j++) {
				Variant callable = arr[j];
				jobject jcallable = callable_to_jcallable(p_env, callable);
				p_env->SetObjectArrayElement(jarr, j, jcallable);
				p_env->DeleteLocalRef(jcallable);
			}

			r_ret.l = jarr;
		} break;
		case ARG_ARRAY_BIT | ARG_TYPE_CLASS: {
			String cn = p_strsig.string();
			if (cn.begins_with("[L") && cn.ends_with(";")) {
				cn = cn.substr(2, cn.length() - 3);
			}
			jclass c = jni_find_class(p_env, cn.utf8().get_data());
			if (c) {
				Array arr = p_variant;
				jobjectArray jarr = p_env->NewObjectArray(arr.size(), c, nullptr);
				for (int j = 0; j < arr.size(); j++) {
					Ref<JavaObject> jo = arr[j];
					p_env->SetObjectArrayElement(jarr, j, jo->instance);
				}

				r_ret.l = jarr;
				p_env->DeleteLocalRef(c);
			}
		} break;
	}

	return true;
}

bool JavaClass::_convert_object_to_variant(JNIEnv *env, jobject obj, Variant &var, uint32_t p_sig) {
	if (!obj) {
		var = Variant(); //seems null is just null...
		return true;
	}

	switch (p_sig) {
		case ARG_TYPE_VOID: {
			return Variant();
		} break;
		case ARG_TYPE_BOOLEAN | ARG_NUMBER_CLASS_BIT: {
			var = env->CallBooleanMethod(obj, JavaClassWrapper::singleton->Boolean_booleanValue);
			return true;
		} break;
		case ARG_TYPE_BYTE | ARG_NUMBER_CLASS_BIT: {
			var = env->CallByteMethod(obj, JavaClassWrapper::singleton->Byte_byteValue);
			return true;

		} break;
		case ARG_TYPE_CHAR | ARG_NUMBER_CLASS_BIT: {
			var = env->CallCharMethod(obj, JavaClassWrapper::singleton->Character_characterValue);
			return true;

		} break;
		case ARG_TYPE_SHORT | ARG_NUMBER_CLASS_BIT: {
			var = env->CallShortMethod(obj, JavaClassWrapper::singleton->Short_shortValue);
			return true;

		} break;
		case ARG_TYPE_INT | ARG_NUMBER_CLASS_BIT: {
			var = env->CallIntMethod(obj, JavaClassWrapper::singleton->Integer_integerValue);
			return true;

		} break;
		case ARG_TYPE_LONG | ARG_NUMBER_CLASS_BIT: {
			var = (int64_t)env->CallLongMethod(obj, JavaClassWrapper::singleton->Long_longValue);
			return true;

		} break;
		case ARG_TYPE_FLOAT | ARG_NUMBER_CLASS_BIT: {
			var = env->CallFloatMethod(obj, JavaClassWrapper::singleton->Float_floatValue);
			return true;

		} break;
		case ARG_TYPE_DOUBLE | ARG_NUMBER_CLASS_BIT: {
			var = env->CallDoubleMethod(obj, JavaClassWrapper::singleton->Double_doubleValue);
			return true;
		} break;
		case ARG_TYPE_STRING: {
			var = jstring_to_string((jstring)obj, env);
			return true;
		} break;
		case ARG_TYPE_CHARSEQUENCE: {
			var = charsequence_to_string(env, obj);
			return true;
		} break;
		case ARG_TYPE_CALLABLE: {
			var = jcallable_to_callable(env, obj);
			return true;
		} break;
		case ARG_TYPE_CLASS: {
			jclass java_class = env->GetObjectClass(obj);
			Ref<JavaClass> java_class_wrapped = JavaClassWrapper::singleton->wrap_jclass(java_class);
			env->DeleteLocalRef(java_class);

			if (java_class_wrapped.is_valid()) {
				String cn = java_class_wrapped->get_java_class_name();
				if (cn == "org.godotengine.godot.Dictionary") {
					var = _jobject_to_variant(env, obj);
				} else {
					Ref<JavaObject> ret = Ref<JavaObject>(memnew(JavaObject(java_class_wrapped, obj)));
					var = ret;
				}
				return true;
			}

			return false;
		} break;
		case ARG_ARRAY_BIT | ARG_TYPE_VOID: {
			var = Array(); // ?
			return true;
		} break;
		case ARG_ARRAY_BIT | ARG_TYPE_BOOLEAN: {
			Array ret;
			jobjectArray arr = (jobjectArray)obj;

			int count = env->GetArrayLength(arr);
			ret.resize(count);

			for (int i = 0; i < count; i++) {
				jboolean val;
				env->GetBooleanArrayRegion((jbooleanArray)arr, i, 1, &val);
				ret[i] = (bool)val;
			}

			var = ret;
			return true;

		} break;
		case ARG_ARRAY_BIT | ARG_TYPE_BYTE: {
			PackedByteArray ret;
			jobjectArray arr = (jobjectArray)obj;

			int count = env->GetArrayLength(arr);
			ret.resize(count);
			env->GetByteArrayRegion((jbyteArray)arr, 0, count, (int8_t *)ret.ptrw());

			var = ret;
			return true;
		} break;
		case ARG_ARRAY_BIT | ARG_TYPE_CHAR: {
			PackedByteArray ret;
			jobjectArray arr = (jobjectArray)obj;

			int count = env->GetArrayLength(arr);
			// Char arrays are UTF-16 encoded, so it's double the length.
			ret.resize(count * 2);
			env->GetCharArrayRegion((jcharArray)arr, 0, count, (jchar *)ret.ptrw());

			var = ret;
			return true;
		} break;
		case ARG_ARRAY_BIT | ARG_TYPE_SHORT: {
			PackedInt32Array ret;
			jobjectArray arr = (jobjectArray)obj;

			int count = env->GetArrayLength(arr);
			ret.resize(count);

			int32_t *ptr = ret.ptrw();
			for (int i = 0; i < count; i++) {
				jshort val;
				env->GetShortArrayRegion((jshortArray)arr, i, 1, &val);
				ptr[i] = val;
			}

			var = ret;
			return true;
		} break;
		case ARG_ARRAY_BIT | ARG_TYPE_INT: {
			PackedInt32Array ret;
			jobjectArray arr = (jobjectArray)obj;

			int count = env->GetArrayLength(arr);
			ret.resize(count);
			env->GetIntArrayRegion((jintArray)arr, 0, count, ret.ptrw());

			var = ret;
			return true;
		} break;
		case ARG_ARRAY_BIT | ARG_TYPE_LONG: {
			PackedInt64Array ret;
			jobjectArray arr = (jobjectArray)obj;

			int count = env->GetArrayLength(arr);
			ret.resize(count);
			env->GetLongArrayRegion((jlongArray)arr, 0, count, ret.ptrw());

			var = ret;
			return true;
		} break;
		case ARG_ARRAY_BIT | ARG_TYPE_FLOAT: {
			PackedFloat32Array ret;
			jobjectArray arr = (jobjectArray)obj;

			int count = env->GetArrayLength(arr);
			ret.resize(count);
			env->GetFloatArrayRegion((jfloatArray)arr, 0, count, ret.ptrw());

			var = ret;
			return true;
		} break;
		case ARG_ARRAY_BIT | ARG_TYPE_DOUBLE: {
			PackedFloat64Array ret;
			jobjectArray arr = (jobjectArray)obj;

			int count = env->GetArrayLength(arr);
			ret.resize(count);
			env->GetDoubleArrayRegion((jdoubleArray)arr, 0, count, ret.ptrw());

			var = ret;
			return true;
		} break;
		case ARG_NUMBER_CLASS_BIT | ARG_ARRAY_BIT | ARG_TYPE_BOOLEAN: {
			Array ret;
			jobjectArray arr = (jobjectArray)obj;

			int count = env->GetArrayLength(arr);
			ret.resize(count);

			for (int i = 0; i < count; i++) {
				jobject o = env->GetObjectArrayElement(arr, i);
				if (o) {
					bool val = env->CallBooleanMethod(o, JavaClassWrapper::singleton->Boolean_booleanValue);
					ret[i] = val;
				}
				env->DeleteLocalRef(o);
			}

			var = ret;
			return true;

		} break;
		case ARG_NUMBER_CLASS_BIT | ARG_ARRAY_BIT | ARG_TYPE_BYTE: {
			PackedByteArray ret;
			jobjectArray arr = (jobjectArray)obj;

			int count = env->GetArrayLength(arr);
			ret.resize(count);

			uint8_t *ptr = ret.ptrw();
			for (int i = 0; i < count; i++) {
				jobject o = env->GetObjectArrayElement(arr, i);
				if (!o) {
					ptr[i] = 0;
				} else {
					int val = env->CallByteMethod(o, JavaClassWrapper::singleton->Byte_byteValue);
					ptr[i] = (uint8_t)val;
				}
				env->DeleteLocalRef(o);
			}

			var = ret;
			return true;
		} break;
		case ARG_NUMBER_CLASS_BIT | ARG_ARRAY_BIT | ARG_TYPE_CHAR: {
			PackedByteArray ret;
			jobjectArray arr = (jobjectArray)obj;

			int count = env->GetArrayLength(arr);
			// Char arrays are UTF-16 encoded, so it's double the length.
			ret.resize(count * 2);

			jchar *ptr = (jchar *)ret.ptrw();
			for (int i = 0; i < count; i++) {
				jobject o = env->GetObjectArrayElement(arr, i);
				if (!o) {
					count = i;
					break;
				} else {
					int val = env->CallCharMethod(o, JavaClassWrapper::singleton->Character_characterValue);
					ptr[i] = (jchar)val;
				}
				env->DeleteLocalRef(o);
			}

			var = ret;
			return true;
		} break;
		case ARG_NUMBER_CLASS_BIT | ARG_ARRAY_BIT | ARG_TYPE_SHORT: {
			PackedInt32Array ret;
			jobjectArray arr = (jobjectArray)obj;

			int count = env->GetArrayLength(arr);
			ret.resize(count);

			int32_t *ptr = ret.ptrw();
			for (int i = 0; i < count; i++) {
				jobject o = env->GetObjectArrayElement(arr, i);
				if (!o) {
					ptr[i] = 0;
				} else {
					int val = env->CallShortMethod(o, JavaClassWrapper::singleton->Short_shortValue);
					ptr[i] = val;
				}
				env->DeleteLocalRef(o);
			}

			var = ret;
			return true;
		} break;
		case ARG_NUMBER_CLASS_BIT | ARG_ARRAY_BIT | ARG_TYPE_INT: {
			PackedInt32Array ret;
			jobjectArray arr = (jobjectArray)obj;

			int count = env->GetArrayLength(arr);
			ret.resize(count);

			int32_t *ptr = ret.ptrw();
			for (int i = 0; i < count; i++) {
				jobject o = env->GetObjectArrayElement(arr, i);
				if (!o) {
					ptr[i] = 0;
				} else {
					int val = env->CallIntMethod(o, JavaClassWrapper::singleton->Integer_integerValue);
					ptr[i] = val;
				}
				env->DeleteLocalRef(o);
			}

			var = ret;
			return true;
		} break;
		case ARG_NUMBER_CLASS_BIT | ARG_ARRAY_BIT | ARG_TYPE_LONG: {
			PackedInt64Array ret;
			jobjectArray arr = (jobjectArray)obj;

			int count = env->GetArrayLength(arr);
			ret.resize(count);

			int64_t *ptr = ret.ptrw();
			for (int i = 0; i < count; i++) {
				jobject o = env->GetObjectArrayElement(arr, i);
				if (!o) {
					ptr[i] = 0;
				} else {
					int64_t val = env->CallLongMethod(o, JavaClassWrapper::singleton->Long_longValue);
					ptr[i] = val;
				}
				env->DeleteLocalRef(o);
			}

			var = ret;
			return true;
		} break;
		case ARG_NUMBER_CLASS_BIT | ARG_ARRAY_BIT | ARG_TYPE_FLOAT: {
			PackedFloat32Array ret;
			jobjectArray arr = (jobjectArray)obj;

			int count = env->GetArrayLength(arr);
			ret.resize(count);

			float *ptr = ret.ptrw();
			for (int i = 0; i < count; i++) {
				jobject o = env->GetObjectArrayElement(arr, i);
				if (!o) {
					ptr[i] = 0.0;
				} else {
					float val = env->CallFloatMethod(o, JavaClassWrapper::singleton->Float_floatValue);
					ptr[i] = val;
				}
				env->DeleteLocalRef(o);
			}

			var = ret;
			return true;
		} break;
		case ARG_NUMBER_CLASS_BIT | ARG_ARRAY_BIT | ARG_TYPE_DOUBLE: {
			PackedFloat64Array ret;
			jobjectArray arr = (jobjectArray)obj;

			int count = env->GetArrayLength(arr);
			ret.resize(count);

			double *ptr = ret.ptrw();
			for (int i = 0; i < count; i++) {
				jobject o = env->GetObjectArrayElement(arr, i);
				if (!o) {
					ptr[i] = 0.0;
				} else {
					double val = env->CallDoubleMethod(o, JavaClassWrapper::singleton->Double_doubleValue);
					ptr[i] = val;
				}
				env->DeleteLocalRef(o);
			}

			var = ret;
			return true;
		} break;

		case ARG_ARRAY_BIT | ARG_TYPE_STRING: {
			PackedStringArray ret;
			jobjectArray arr = (jobjectArray)obj;

			int count = env->GetArrayLength(arr);
			ret.resize(count);

			String *ptr = ret.ptrw();
			for (int i = 0; i < count; i++) {
				jobject o = env->GetObjectArrayElement(arr, i);
				if (o) {
					String val = jstring_to_string((jstring)o, env);
					ptr[i] = val;
				}
				env->DeleteLocalRef(o);
			}

			var = ret;
			return true;
		} break;
		case ARG_ARRAY_BIT | ARG_TYPE_CHARSEQUENCE: {
			PackedStringArray ret;
			jobjectArray arr = (jobjectArray)obj;

			int count = env->GetArrayLength(arr);
			ret.resize(count);

			String *ptr = ret.ptrw();
			for (int i = 0; i < count; i++) {
				jobject o = env->GetObjectArrayElement(arr, i);
				if (o) {
					String val = charsequence_to_string(env, o);
					ptr[i] = val;
				}
				env->DeleteLocalRef(o);
			}

			var = ret;
			return true;
		} break;
		case ARG_ARRAY_BIT | ARG_TYPE_CALLABLE: {
			Array ret;
			jobjectArray jarr = (jobjectArray)obj;
			int count = env->GetArrayLength(jarr);
			ret.resize(count);
			for (int i = 0; i < count; i++) {
				jobject o = env->GetObjectArrayElement(jarr, i);
				if (o) {
					Callable callable = jcallable_to_callable(env, o);
					ret[i] = callable;
				}
				env->DeleteLocalRef(o);
			}

			var = ret;
			return true;
		} break;
		case ARG_ARRAY_BIT | ARG_TYPE_CLASS: {
			Array ret;
			jobjectArray jarr = (jobjectArray)obj;
			int count = env->GetArrayLength(jarr);
			ret.resize(count);
			for (int i = 0; i < count; i++) {
				jobject obj = env->GetObjectArrayElement(jarr, i);
				if (obj) {
					jclass java_class = env->GetObjectClass(obj);
					Ref<JavaClass> java_class_wrapped = JavaClassWrapper::singleton->wrap_jclass(java_class);
					env->DeleteLocalRef(java_class);

					if (java_class_wrapped.is_valid()) {
						String cn = java_class_wrapped->get_java_class_name();
						if (cn == "org.godotengine.godot.Dictionary") {
							ret[i] = _jobject_to_variant(env, obj);
						} else {
							Ref<JavaObject> java_obj_wrapped = Ref<JavaObject>(memnew(JavaObject(java_class_wrapped, obj)));
							ret[i] = java_obj_wrapped;
						}
					}
				}
				env->DeleteLocalRef(obj);
			}

			var = ret;
			return true;
		} break;
	}

	return false;
}

bool JavaClassWrapper::_wrap_class_components(JNIEnv *p_env, const Ref<JavaClass> &p_java_class, jclass p_class, bool p_allow_non_public_methods_access) {
	ERR_FAIL_NULL_V(p_class, false);

	jobjectArray constructors = (jobjectArray)p_env->CallObjectMethod(p_class, Class_getConstructors);
	if (p_env->ExceptionCheck()) {
		p_env->ExceptionDescribe();
		p_env->ExceptionClear();
	}
	ERR_FAIL_NULL_V(constructors, false);

	jobjectArray methods = (jobjectArray)p_env->CallObjectMethod(p_class, Class_getDeclaredMethods);
	if (p_env->ExceptionCheck()) {
		p_env->ExceptionDescribe();
		p_env->ExceptionClear();
	}
	ERR_FAIL_NULL_V(methods, false);

	int constructor_count = p_env->GetArrayLength(constructors);
	int method_count = p_env->GetArrayLength(methods);

	int methods_and_constructors_count = method_count + constructor_count;
	for (int i = 0; i < methods_and_constructors_count; i++) {
		bool is_constructor = i < constructor_count;
		jobject obj = is_constructor
				? p_env->GetObjectArrayElement(constructors, i)
				: p_env->GetObjectArrayElement(methods, i - constructor_count);
		ERR_CONTINUE(!obj);

		String str_method;
		if (is_constructor) {
			str_method = "<init>";
		} else {
			jstring name = (jstring)p_env->CallObjectMethod(obj, Method_getName);
			str_method = jstring_to_string(name, p_env);
			p_env->DeleteLocalRef(name);
		}

		jint mods = p_env->CallIntMethod(obj, is_constructor ? Constructor_getModifiers : Method_getModifiers);
		bool is_public = (mods & 0x0001) != 0; // java.lang.reflect.Modifier.PUBLIC

		if (!is_public && (is_constructor || !p_allow_non_public_methods_access)) {
			p_env->DeleteLocalRef(obj);
			continue; //not public bye
		}

		jobjectArray param_types = (jobjectArray)p_env->CallObjectMethod(obj, is_constructor ? Constructor_getParameterTypes : Method_getParameterTypes);
		int count = p_env->GetArrayLength(param_types);

		if (!p_java_class->methods.has(str_method)) {
			p_java_class->methods[str_method] = List<JavaClass::MethodInfo>();
		}

		JavaClass::MethodInfo mi;
		mi._public = is_public;
		mi._static = (mods & 0x8) != 0;
		mi._constructor = is_constructor;
		bool valid = true;
		String signature = "(";

		for (int j = 0; j < count; j++) {
			jobject obj2 = p_env->GetObjectArrayElement(param_types, j);
			String strsig;
			uint32_t sig = 0;
			if (!_get_type_sig(p_env, obj2, sig, strsig)) {
				valid = false;
				p_env->DeleteLocalRef(obj2);
				break;
			}
			signature += strsig;
			mi.param_types.push_back(sig);
			mi.param_sigs.push_back(strsig);
			p_env->DeleteLocalRef(obj2);
		}

		if (!valid) {
			print_line("Method can't be bound (unsupported arguments): " + p_java_class->java_class_name + "::" + str_method);
			p_env->DeleteLocalRef(obj);
			p_env->DeleteLocalRef(param_types);
			continue;
		}

		signature += ")";

		if (is_constructor) {
			signature += "V";
			mi.return_type = JavaClass::ARG_TYPE_CLASS;
		} else {
			jobject return_type = (jobject)p_env->CallObjectMethod(obj, Method_getReturnType);

			String strsig;
			uint32_t sig = 0;
			if (!_get_type_sig(p_env, return_type, sig, strsig)) {
				print_line("Method can't be bound (unsupported return type): " + p_java_class->java_class_name + "::" + str_method);
				p_env->DeleteLocalRef(obj);
				p_env->DeleteLocalRef(param_types);
				p_env->DeleteLocalRef(return_type);
				continue;
			}

			signature += strsig;
			mi.return_type = sig;

			p_env->DeleteLocalRef(return_type);
		}

		bool discard = false;

		for (List<JavaClass::MethodInfo>::Element *E = p_java_class->methods[str_method].front(); E; E = E->next()) {
			float new_likeliness = 0;
			float existing_likeliness = 0;

			if (E->get().param_types.size() != mi.param_types.size()) {
				continue;
			}
			bool this_valid = true;
			for (int j = 0; j < E->get().param_types.size(); j++) {
				Variant::Type _new;
				float new_l;
				Variant::Type existing;
				float existing_l;
				JavaClass::_convert_to_variant_type(E->get().param_types[j], existing, existing_l);
				JavaClass::_convert_to_variant_type(mi.param_types[j], _new, new_l);
				if (_new != existing) {
					this_valid = false;
					break;
				} else if ((_new == Variant::OBJECT || _new == Variant::ARRAY) && E->get().param_sigs[j] != mi.param_sigs[j]) {
					this_valid = false;
					break;
				}
				new_likeliness += new_l;
				existing_likeliness += existing_l;
			}

			if (!this_valid) {
				continue;
			}

			if (new_likeliness > existing_likeliness) {
				p_java_class->methods[str_method].erase(E);
				break;
			} else {
				discard = true;
			}
		}

		if (!discard) {
			if (mi._static) {
				mi.method = p_env->GetStaticMethodID(p_class, str_method.utf8().get_data(), signature.utf8().get_data());
			} else {
				mi.method = p_env->GetMethodID(p_class, str_method.utf8().get_data(), signature.utf8().get_data());
			}

			if (p_env->ExceptionCheck()) {
				// Exceptions may be thrown when trying to access hidden methods; write the exception to the logs and continue.
				p_env->ExceptionDescribe();
				p_env->ExceptionClear();
			} else if (mi.method) {
				p_java_class->methods[str_method].push_back(mi);
			}
		}

		p_env->DeleteLocalRef(obj);
		p_env->DeleteLocalRef(param_types);
	}

	p_env->DeleteLocalRef(constructors);
	p_env->DeleteLocalRef(methods);

	jobjectArray fields = (jobjectArray)p_env->CallObjectMethod(p_class, Class_getFields);

	int count = p_env->GetArrayLength(fields);

	for (int i = 0; i < count; i++) {
		jobject obj = p_env->GetObjectArrayElement(fields, i);
		ERR_CONTINUE(!obj);

		jstring name = (jstring)p_env->CallObjectMethod(obj, Field_getName);
		String str_field = jstring_to_string(name, p_env);
		p_env->DeleteLocalRef(name);
		int mods = p_env->CallIntMethod(obj, Field_getModifiers);
		if (mods & 0x1) { // public
			jobject field_type = (jobject)p_env->CallObjectMethod(obj, Field_getType);
			String strsig;
			uint32_t sig = 0;
			if (_get_type_sig(p_env, field_type, sig, strsig)) {
				JavaClass::FieldInfo field_info;
				field_info._static = mods & 0x8;
				field_info._final = mods & 0x10;
				field_info.field_type = sig;
				field_info.field_sig = strsig;
				if (field_info._static) {
					field_info.field = p_env->GetStaticFieldID(
							p_class,
							str_field.utf8().get_data(),
							strsig.utf8().get_data());
					if (field_info._final) {
						// The field value won't change, so we cache it.
						if (!JavaClass::_get_field_value(p_env, nullptr, p_class, field_info, field_info.constant_value)) {
							print_line("Unable to retrieve value for constant: " + p_java_class->java_class_name + "::" + str_field);
						}
					}
				} else {
					field_info.field = p_env->GetFieldID(
							p_class,
							str_field.utf8().get_data(),
							strsig.utf8().get_data());
				}
				if (p_env->ExceptionCheck()) {
					// Exceptions may be thrown when trying to access hidden fields; write the exception to the logs and continue.
					p_env->ExceptionDescribe();
					p_env->ExceptionClear();
				} else if (field_info.field) {
					p_java_class->fields[str_field] = field_info;
				}
			} else {
				print_line("Field can't be bound (unsupported field type): " + p_java_class->java_class_name + "::" + str_field);
			}

			p_env->DeleteLocalRef(field_type);
		}
		p_env->DeleteLocalRef(obj);
	}

	p_env->DeleteLocalRef(fields);
	return true;
}

Ref<JavaClass> JavaClassWrapper::_wrap(const String &p_class, bool p_allow_non_public_methods_access) {
	String class_name_dots = p_class.replace_char('/', '.');
	if (class_cache.has(class_name_dots)) {
		return class_cache[class_name_dots];
	}

	JNIEnv *env = get_jni_env();
	ERR_FAIL_NULL_V(env, Ref<JavaClass>());

	jclass bclass = jni_find_class(env, class_name_dots.replace_char('.', '/').utf8().get_data());
	ERR_FAIL_NULL_V_MSG(bclass, Ref<JavaClass>(), vformat("Java class '%s' not found.", p_class));

	Ref<JavaClass> java_class;
	java_class.instantiate();
	java_class->java_class_name = class_name_dots;
	Vector<String> class_name_parts = class_name_dots.split(".");
	java_class->java_constructor_name = class_name_parts[class_name_parts.size() - 1];
	java_class->_class = (jclass)env->NewGlobalRef(bclass);
	java_class->is_interface = env->CallBooleanMethod(bclass, Class_isInterface);

	bool class_components_configured;
	if (_is_proxy_class(env, bclass)) {
		// Proxy class components must be setup using the interfaces they implement.
		jobjectArray interfaces = (jobjectArray)env->CallObjectMethod(bclass, Class_getInterfaces);
		if (env->ExceptionCheck()) {
			env->ExceptionDescribe();
			env->ExceptionClear();
		}
		int interfaces_count = interfaces == nullptr ? 0 : env->GetArrayLength(interfaces);
		for (int i = 0; i < interfaces_count; i++) {
			jclass interface = (jclass)env->GetObjectArrayElement(interfaces, i);
			ERR_CONTINUE(!interface);

			jstring j_interface_name = (jstring)env->CallObjectMethod(interface, Class_getName);
			String interface_name = jstring_to_string(j_interface_name, env);
			env->DeleteLocalRef(j_interface_name);
			if (!_wrap_class_components(env, java_class, interface, p_allow_non_public_methods_access)) {
				ERR_PRINT(vformat("Unable to set up components for proxy class interface %s.", interface_name));
				continue;
			}

			class_components_configured = true;
		}
	} else {
		class_components_configured = _wrap_class_components(env, java_class, bclass, p_allow_non_public_methods_access);
	}
	env->DeleteLocalRef(bclass);

	if (!class_components_configured) {
		java_class.unref();
		return Ref<JavaClass>();
	}

	// Cache the initialized JavaClass instance.
	class_cache[class_name_dots] = java_class;

	return java_class;
}

Ref<JavaObject> JavaClassWrapper::create_sam_callback(const String &p_sam_interface, const Callable &p_callable) {
	JNIEnv *env = get_jni_env();
	ERR_FAIL_NULL_V(env, Ref<JavaObject>());
	ERR_FAIL_NULL_V(android_runtime_class, Ref<JavaObject>());
	ERR_FAIL_NULL_V(ARP_create_proxy_from_godot_callable, Ref<JavaObject>());

	Ref<JavaClass> interface_class = wrap(p_sam_interface);
	if (interface_class.is_null()) {
		ERR_PRINT(vformat("Invalid java class: %s.", p_sam_interface));
		return Ref<JavaObject>();
	}
	if (!interface_class->is_interface) {
		ERR_PRINT(vformat("Java class %s must be an interface.", p_sam_interface));
		return Ref<JavaObject>();
	}

	if (interface_class->methods.size() != 1) {
		ERR_PRINT(vformat("%s must be a Single Abstract Method (SAM) interface.", p_sam_interface));
		return Ref<JavaObject>();
	}

	jobject j_callable = callable_to_jcallable(env, p_callable);
	jstring j_interface = env->NewStringUTF(p_sam_interface.utf8().get_data());

	jobject proxy = env->CallStaticObjectMethod(android_runtime_class, ARP_create_proxy_from_godot_callable, j_interface, j_callable);
	Ref<JavaObject> result = _jobject_to_variant(env, proxy);

	env->DeleteLocalRef(j_callable);
	env->DeleteLocalRef(j_interface);
	env->DeleteLocalRef(proxy);

	return result;
}

Ref<JavaObject> JavaClassWrapper::create_proxy(const Object *p_object, const PackedStringArray &p_interfaces) {
	ERR_FAIL_NULL_V(p_object, Ref<JavaObject>());
	ERR_FAIL_COND_V(p_interfaces.is_empty(), Ref<JavaObject>());

	JNIEnv *env = get_jni_env();
	ERR_FAIL_NULL_V(env, Ref<JavaObject>());
	ERR_FAIL_NULL_V(android_runtime_class, Ref<JavaObject>());
	ERR_FAIL_NULL_V(ARP_create_proxy_from_godot_callable, Ref<JavaObject>());

	for (const String &interface_name : p_interfaces) {
		Ref<JavaClass> interface_class = wrap(interface_name);
		if (interface_class.is_null()) {
			ERR_PRINT(vformat("Invalid java class: %s.", interface_name));
			return Ref<JavaObject>();
		}

		if (!interface_class->is_interface) {
			ERR_PRINT(vformat("Java class %s must be an interface.", interface_name));
			return Ref<JavaObject>();
		}
	}

	jlong object_id = p_object->get_instance_id();
	jobjectArray j_interfaces = env->NewObjectArray(p_interfaces.size(), jni_find_class(env, "java/lang/String"), nullptr);
	for (int i = 0; i < p_interfaces.size(); i++) {
		jstring j_interface = env->NewStringUTF(p_interfaces[i].utf8().get_data());
		env->SetObjectArrayElement(j_interfaces, i, j_interface);
		env->DeleteLocalRef(j_interface);
	}

	jobject proxy = env->CallStaticObjectMethod(android_runtime_class, ARP_create_proxy_from_godot_object_id, object_id, j_interfaces);
	Ref<JavaObject> result = _jobject_to_variant(env, proxy);

	env->DeleteLocalRef(j_interfaces);
	env->DeleteLocalRef(proxy);

	return result;
}

Ref<JavaClass> JavaClassWrapper::wrap_jclass(jclass p_class, bool p_allow_non_public_methods_access) {
	JNIEnv *env = get_jni_env();
	ERR_FAIL_NULL_V(env, Ref<JavaClass>());

	jstring class_name = (jstring)env->CallObjectMethod(p_class, Class_getName);
	String class_name_string = jstring_to_string(class_name, env);
	env->DeleteLocalRef(class_name);

	return _wrap(class_name_string, p_allow_non_public_methods_access);
}

bool JavaClassWrapper::_is_proxy_class(JNIEnv *env, jclass p_class) {
	ERR_FAIL_NULL_V(proxy_class, false);
	ERR_FAIL_NULL_V(Proxy_isProxyClass, false);
	ERR_FAIL_NULL_V(env, false);

	return env->CallStaticBooleanMethod(proxy_class, Proxy_isProxyClass, p_class);
}

JavaClassWrapper *JavaClassWrapper::singleton = nullptr;

JavaClassWrapper::JavaClassWrapper() {
	singleton = this;

	JNIEnv *env = get_jni_env();
	ERR_FAIL_NULL(env);

	proxy_class = jni_find_class(env, "java/lang/reflect/Proxy");
	if (proxy_class) {
		proxy_class = (jclass)env->NewGlobalRef(proxy_class);
		Proxy_isProxyClass = env->GetStaticMethodID(proxy_class, "isProxyClass", "(Ljava/lang/Class;)Z");
	}

	android_runtime_class = jni_find_class(env, "org/godotengine/godot/plugin/AndroidRuntimePlugin");
	if (android_runtime_class) {
		android_runtime_class = (jclass)env->NewGlobalRef(android_runtime_class);
		ARP_create_proxy_from_godot_callable = env->GetStaticMethodID(android_runtime_class, "createProxyFromGodotCallable", "(Ljava/lang/String;Lorg/godotengine/godot/variant/Callable;)Ljava/lang/Object;");
		ARP_create_proxy_from_godot_object_id = env->GetStaticMethodID(android_runtime_class, "createProxyFromGodotObjectID", "(J[Ljava/lang/String;)Ljava/lang/Object;");
	}

	jclass bclass = jni_find_class(env, "java/lang/Class");
	Class_getConstructors = env->GetMethodID(bclass, "getConstructors", "()[Ljava/lang/reflect/Constructor;");
	Class_getDeclaredMethods = env->GetMethodID(bclass, "getDeclaredMethods", "()[Ljava/lang/reflect/Method;");
	Class_getFields = env->GetMethodID(bclass, "getFields", "()[Ljava/lang/reflect/Field;");
	Class_getInterfaces = env->GetMethodID(bclass, "getInterfaces", "()[Ljava/lang/Class;");
	Class_getName = env->GetMethodID(bclass, "getName", "()Ljava/lang/String;");
	Class_getSuperclass = env->GetMethodID(bclass, "getSuperclass", "()Ljava/lang/Class;");
	Class_isInterface = env->GetMethodID(bclass, "isInterface", "()Z");
	env->DeleteLocalRef(bclass);

	bclass = jni_find_class(env, "java/lang/reflect/Constructor");
	Constructor_getParameterTypes = env->GetMethodID(bclass, "getParameterTypes", "()[Ljava/lang/Class;");
	Constructor_getModifiers = env->GetMethodID(bclass, "getModifiers", "()I");
	env->DeleteLocalRef(bclass);

	bclass = jni_find_class(env, "java/lang/reflect/Method");
	Method_getParameterTypes = env->GetMethodID(bclass, "getParameterTypes", "()[Ljava/lang/Class;");
	Method_getReturnType = env->GetMethodID(bclass, "getReturnType", "()Ljava/lang/Class;");
	Method_getName = env->GetMethodID(bclass, "getName", "()Ljava/lang/String;");
	Method_getModifiers = env->GetMethodID(bclass, "getModifiers", "()I");
	env->DeleteLocalRef(bclass);

	bclass = jni_find_class(env, "java/lang/reflect/Field");
	Field_getName = env->GetMethodID(bclass, "getName", "()Ljava/lang/String;");
	Field_getModifiers = env->GetMethodID(bclass, "getModifiers", "()I");
	Field_get = env->GetMethodID(bclass, "get", "(Ljava/lang/Object;)Ljava/lang/Object;");
	Field_getType = env->GetMethodID(bclass, "getType", "()Ljava/lang/Class;");
	env->DeleteLocalRef(bclass);

	bclass = jni_find_class(env, "java/lang/Boolean");
	Boolean_booleanValue = env->GetMethodID(bclass, "booleanValue", "()Z");
	env->DeleteLocalRef(bclass);

	bclass = jni_find_class(env, "java/lang/Byte");
	Byte_byteValue = env->GetMethodID(bclass, "byteValue", "()B");
	env->DeleteLocalRef(bclass);

	bclass = jni_find_class(env, "java/lang/Character");
	Character_characterValue = env->GetMethodID(bclass, "charValue", "()C");
	env->DeleteLocalRef(bclass);

	bclass = jni_find_class(env, "java/lang/Short");
	Short_shortValue = env->GetMethodID(bclass, "shortValue", "()S");
	env->DeleteLocalRef(bclass);

	bclass = jni_find_class(env, "java/lang/Integer");
	Integer_integerValue = env->GetMethodID(bclass, "intValue", "()I");
	env->DeleteLocalRef(bclass);

	bclass = jni_find_class(env, "java/lang/Long");
	Long_longValue = env->GetMethodID(bclass, "longValue", "()J");
	env->DeleteLocalRef(bclass);

	bclass = jni_find_class(env, "java/lang/Float");
	Float_floatValue = env->GetMethodID(bclass, "floatValue", "()F");
	env->DeleteLocalRef(bclass);

	bclass = jni_find_class(env, "java/lang/Double");
	Double_doubleValue = env->GetMethodID(bclass, "doubleValue", "()D");
	env->DeleteLocalRef(bclass);
}
