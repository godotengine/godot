/**************************************************************************/
/*  jni_utils.cpp                                                         */
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

#include "jni_utils.h"

#include "api/java_class_wrapper.h"

static jobject android_class_loader = nullptr;
static jmethodID load_class_method = nullptr;

jobject callable_to_jcallable(JNIEnv *p_env, const Variant &p_callable) {
	ERR_FAIL_NULL_V(p_env, nullptr);
	if (p_callable.get_type() != Variant::CALLABLE) {
		return nullptr;
	}

	Variant *callable_jcopy = memnew(Variant(p_callable));

	jclass bclass = jni_find_class(p_env, "org/godotengine/godot/variant/Callable");
	jmethodID ctor = p_env->GetMethodID(bclass, "<init>", "(J)V");
	jobject jcallable = p_env->NewObject(bclass, ctor, reinterpret_cast<int64_t>(callable_jcopy));
	p_env->DeleteLocalRef(bclass);

	return jcallable;
}

Callable jcallable_to_callable(JNIEnv *p_env, jobject p_jcallable_obj) {
	ERR_FAIL_NULL_V(p_env, Callable());

	const Variant *callable_variant = nullptr;
	if (p_jcallable_obj) {
		jclass callable_class = jni_find_class(p_env, "org/godotengine/godot/variant/Callable");
		if (callable_class && p_env->IsInstanceOf(p_jcallable_obj, callable_class)) {
			jmethodID get_native_pointer = p_env->GetMethodID(callable_class, "getNativePointer", "()J");
			jlong native_callable = p_env->CallLongMethod(p_jcallable_obj, get_native_pointer);

			callable_variant = reinterpret_cast<const Variant *>(native_callable);
		}

		p_env->DeleteLocalRef(callable_class);
	}

	ERR_FAIL_NULL_V(callable_variant, Callable());
	return *callable_variant;
}

String charsequence_to_string(JNIEnv *p_env, jobject p_charsequence) {
	ERR_FAIL_NULL_V(p_env, String());

	String result;
	if (p_charsequence) {
		jclass bclass = jni_find_class(p_env, "java/lang/CharSequence");
		if (bclass && p_env->IsInstanceOf(p_charsequence, bclass)) {
			jmethodID to_string = p_env->GetMethodID(bclass, "toString", "()Ljava/lang/String;");
			jstring obj_string = (jstring)p_env->CallObjectMethod(p_charsequence, to_string);

			result = jstring_to_string(obj_string, p_env);
			p_env->DeleteLocalRef(obj_string);
		}

		p_env->DeleteLocalRef(bclass);
	}
	return result;
}

jobject _variant_to_jobject(JNIEnv *env, Variant::Type p_type, const Variant *p_arg, int p_depth) {
	jobject ret = nullptr;

	if (p_depth > Variant::MAX_RECURSION_DEPTH) {
		ERR_PRINT("Variant is too deep! Bailing.");
		return ret;
	}

	env->PushLocalFrame(2);
	switch (p_type) {
		case Variant::BOOL: {
			jclass bclass = jni_find_class(env, "java/lang/Boolean");
			jmethodID ctor = env->GetMethodID(bclass, "<init>", "(Z)V");
			jvalue val;
			val.z = (bool)(*p_arg);
			ret = env->NewObjectA(bclass, ctor, &val);
			env->DeleteLocalRef(bclass);
		} break;
		case Variant::INT: {
			jclass bclass = jni_find_class(env, "java/lang/Long");
			jmethodID ctor = env->GetMethodID(bclass, "<init>", "(J)V");
			jvalue val;
			val.j = (jlong)(*p_arg);
			ret = env->NewObjectA(bclass, ctor, &val);
			env->DeleteLocalRef(bclass);
		} break;
		case Variant::FLOAT: {
			jclass bclass = jni_find_class(env, "java/lang/Double");
			jmethodID ctor = env->GetMethodID(bclass, "<init>", "(D)V");
			jvalue val;
			val.d = (double)(*p_arg);
			ret = env->NewObjectA(bclass, ctor, &val);
			env->DeleteLocalRef(bclass);
		} break;
		case Variant::STRING: {
			String s = *p_arg;
			jstring jStr = env->NewStringUTF(s.utf8().get_data());
			ret = jStr;
		} break;
		case Variant::PACKED_STRING_ARRAY: {
			Vector<String> sarray = *p_arg;
			jobjectArray arr = env->NewObjectArray(sarray.size(), jni_find_class(env, "java/lang/String"), env->NewStringUTF(""));

			for (int j = 0; j < sarray.size(); j++) {
				jstring str = env->NewStringUTF(sarray[j].utf8().get_data());
				env->SetObjectArrayElement(arr, j, str);
				env->DeleteLocalRef(str);
			}
			ret = arr;
		} break;

		case Variant::CALLABLE: {
			jobject jcallable = callable_to_jcallable(env, *p_arg);
			ret = jcallable;
		} break;

		case Variant::DICTIONARY: {
			Dictionary dict = *p_arg;
			jclass dclass = jni_find_class(env, "org/godotengine/godot/Dictionary");
			jmethodID ctor = env->GetMethodID(dclass, "<init>", "()V");
			jobject jdict = env->NewObject(dclass, ctor);

			Array keys = dict.keys();

			jobjectArray jkeys = env->NewObjectArray(keys.size(), jni_find_class(env, "java/lang/String"), env->NewStringUTF(""));
			for (int j = 0; j < keys.size(); j++) {
				jstring str = env->NewStringUTF(String(keys[j]).utf8().get_data());
				env->SetObjectArrayElement(jkeys, j, str);
				env->DeleteLocalRef(str);
			}

			jmethodID set_keys = env->GetMethodID(dclass, "set_keys", "([Ljava/lang/String;)V");
			jvalue val;
			val.l = jkeys;
			env->CallVoidMethodA(jdict, set_keys, &val);
			env->DeleteLocalRef(jkeys);

			jobjectArray jvalues = env->NewObjectArray(keys.size(), jni_find_class(env, "java/lang/Object"), nullptr);

			for (int j = 0; j < keys.size(); j++) {
				Variant var = dict[keys[j]];
				jobject jvar = _variant_to_jobject(env, var.get_type(), &var, p_depth + 1);
				env->SetObjectArrayElement(jvalues, j, jvar);
				if (jvar) {
					env->DeleteLocalRef(jvar);
				}
			}

			jmethodID set_values = env->GetMethodID(dclass, "set_values", "([Ljava/lang/Object;)V");
			val.l = jvalues;
			env->CallVoidMethodA(jdict, set_values, &val);
			env->DeleteLocalRef(jvalues);
			env->DeleteLocalRef(dclass);

			ret = jdict;
		} break;

		case Variant::ARRAY: {
			Array array = *p_arg;
			jobjectArray arr = env->NewObjectArray(array.size(), jni_find_class(env, "java/lang/Object"), nullptr);

			for (int j = 0; j < array.size(); j++) {
				Variant var = array[j];
				jobject jvar = _variant_to_jobject(env, var.get_type(), &var, p_depth + 1);
				env->SetObjectArrayElement(arr, j, jvar);
				if (jvar) {
					env->DeleteLocalRef(jvar);
				}
			}
			ret = arr;
		} break;

		case Variant::PACKED_INT32_ARRAY: {
			Vector<int> array = *p_arg;
			jintArray arr = env->NewIntArray(array.size());
			const int *r = array.ptr();
			env->SetIntArrayRegion(arr, 0, array.size(), r);
			ret = arr;
		} break;
		case Variant::PACKED_INT64_ARRAY: {
			Vector<int64_t> array = *p_arg;
			jlongArray arr = env->NewLongArray(array.size());
			const int64_t *r = array.ptr();
			env->SetLongArrayRegion(arr, 0, array.size(), r);
			ret = arr;
		} break;
		case Variant::PACKED_BYTE_ARRAY: {
			Vector<uint8_t> array = *p_arg;
			jbyteArray arr = env->NewByteArray(array.size());
			const uint8_t *r = array.ptr();
			env->SetByteArrayRegion(arr, 0, array.size(), reinterpret_cast<const signed char *>(r));
			ret = arr;
		} break;
		case Variant::PACKED_FLOAT32_ARRAY: {
			Vector<float> array = *p_arg;
			jfloatArray arr = env->NewFloatArray(array.size());
			const float *r = array.ptr();
			env->SetFloatArrayRegion(arr, 0, array.size(), r);
			ret = arr;
		} break;
		case Variant::PACKED_FLOAT64_ARRAY: {
			Vector<double> array = *p_arg;
			jdoubleArray arr = env->NewDoubleArray(array.size());
			const double *r = array.ptr();
			env->SetDoubleArrayRegion(arr, 0, array.size(), r);
			ret = arr;
		} break;
		case Variant::OBJECT: {
			Ref<JavaObject> generic_object = *p_arg;
			if (generic_object.is_valid()) {
				jobject obj = env->NewLocalRef(generic_object->get_instance());
				ret = obj;
			}
		} break;

		// Add default to prevent compiler warning about not handling all types.
		default:
			break;
	}

	return env->PopLocalFrame(ret);
}

String _get_class_name(JNIEnv *env, jclass cls, bool *array) {
	jclass cclass = jni_find_class(env, "java/lang/Class");
	jmethodID getName = env->GetMethodID(cclass, "getName", "()Ljava/lang/String;");
	jstring clsName = (jstring)env->CallObjectMethod(cls, getName);

	if (array) {
		jmethodID isArray = env->GetMethodID(cclass, "isArray", "()Z");
		jboolean isarr = env->CallBooleanMethod(cls, isArray);
		(*array) = isarr != 0;
	}
	String name = jstring_to_string(clsName, env);
	env->DeleteLocalRef(clsName);
	env->DeleteLocalRef(cclass);

	return name;
}

Variant _jobject_to_variant(JNIEnv *env, jobject obj, int p_depth) {
	ERR_FAIL_COND_V_MSG(p_depth > Variant::MAX_RECURSION_DEPTH, Variant(), "Variant is too deep! Bailing.");

	if (obj == nullptr) {
		return Variant();
	}

	jclass c = env->GetObjectClass(obj);
	bool array;
	String name = _get_class_name(env, c, &array);

	if (name == "java.lang.String") {
		return jstring_to_string((jstring)obj, env);
	}

	if (name == "java.lang.CharSequence") {
		return charsequence_to_string(env, obj);
	}

	if (name == "[Ljava.lang.String;") {
		jobjectArray arr = (jobjectArray)obj;
		int stringCount = env->GetArrayLength(arr);
		Vector<String> sarr;

		for (int i = 0; i < stringCount; i++) {
			jstring string = (jstring)env->GetObjectArrayElement(arr, i);
			sarr.push_back(jstring_to_string(string, env));
			env->DeleteLocalRef(string);
		}

		return sarr;
	}

	if (name == "[Ljava.lang.CharSequence;") {
		jobjectArray arr = (jobjectArray)obj;
		int stringCount = env->GetArrayLength(arr);
		Vector<String> sarr;

		for (int i = 0; i < stringCount; i++) {
			jobject charsequence = env->GetObjectArrayElement(arr, i);
			sarr.push_back(charsequence_to_string(env, charsequence));
			env->DeleteLocalRef(charsequence);
		}

		return sarr;
	}

	if (name == "java.lang.Boolean") {
		jmethodID boolValue = env->GetMethodID(c, "booleanValue", "()Z");
		bool ret = env->CallBooleanMethod(obj, boolValue);
		return ret;
	}

	if (name == "java.lang.Integer" || name == "java.lang.Long") {
		jclass nclass = jni_find_class(env, "java/lang/Number");
		jmethodID longValue = env->GetMethodID(nclass, "longValue", "()J");
		jlong ret = env->CallLongMethod(obj, longValue);
		env->DeleteLocalRef(nclass);
		return ret;
	}

	if (name == "[I") {
		jintArray arr = (jintArray)obj;
		int fCount = env->GetArrayLength(arr);
		Vector<int> sarr;
		sarr.resize(fCount);

		int *w = sarr.ptrw();
		env->GetIntArrayRegion(arr, 0, fCount, w);
		return sarr;
	}

	if (name == "[J") {
		jlongArray arr = (jlongArray)obj;
		int fCount = env->GetArrayLength(arr);
		Vector<int64_t> sarr;
		sarr.resize(fCount);

		int64_t *w = sarr.ptrw();
		env->GetLongArrayRegion(arr, 0, fCount, w);
		return sarr;
	}

	if (name == "[B") {
		jbyteArray arr = (jbyteArray)obj;
		int fCount = env->GetArrayLength(arr);
		Vector<uint8_t> sarr;
		sarr.resize(fCount);

		uint8_t *w = sarr.ptrw();
		env->GetByteArrayRegion(arr, 0, fCount, reinterpret_cast<signed char *>(w));
		return sarr;
	}

	if (name == "java.lang.Float" || name == "java.lang.Double") {
		jclass nclass = jni_find_class(env, "java/lang/Number");
		jmethodID doubleValue = env->GetMethodID(nclass, "doubleValue", "()D");
		double ret = env->CallDoubleMethod(obj, doubleValue);
		env->DeleteLocalRef(nclass);
		return ret;
	}

	if (name == "[D") {
		jdoubleArray arr = (jdoubleArray)obj;
		int fCount = env->GetArrayLength(arr);
		PackedFloat64Array packed_array;
		packed_array.resize(fCount);

		double *w = packed_array.ptrw();

		for (int i = 0; i < fCount; i++) {
			double n;
			env->GetDoubleArrayRegion(arr, i, 1, &n);
			w[i] = n;
		}
		return packed_array;
	}

	if (name == "[F") {
		jfloatArray arr = (jfloatArray)obj;
		int fCount = env->GetArrayLength(arr);
		PackedFloat32Array packed_array;
		packed_array.resize(fCount);

		float *w = packed_array.ptrw();

		for (int i = 0; i < fCount; i++) {
			float n;
			env->GetFloatArrayRegion(arr, i, 1, &n);
			w[i] = n;
		}
		return packed_array;
	}

	if (name == "[Ljava.lang.Object;") {
		jobjectArray arr = (jobjectArray)obj;
		int objCount = env->GetArrayLength(arr);
		Array varr;

		for (int i = 0; i < objCount; i++) {
			jobject jobj = env->GetObjectArrayElement(arr, i);
			Variant v = _jobject_to_variant(env, jobj, p_depth + 1);
			varr.push_back(v);
			env->DeleteLocalRef(jobj);
		}

		return varr;
	}

	if (name == "java.util.HashMap" || name == "org.godotengine.godot.Dictionary") {
		Dictionary ret;
		jclass oclass = c;
		jmethodID get_keys = env->GetMethodID(oclass, "get_keys", "()[Ljava/lang/String;");
		jobjectArray arr = (jobjectArray)env->CallObjectMethod(obj, get_keys);

		PackedStringArray keys = _jobject_to_variant(env, arr, p_depth + 1);
		env->DeleteLocalRef(arr);

		jmethodID get_values = env->GetMethodID(oclass, "get_values", "()[Ljava/lang/Object;");
		arr = (jobjectArray)env->CallObjectMethod(obj, get_values);

		Array vals = _jobject_to_variant(env, arr, p_depth + 1);
		env->DeleteLocalRef(arr);

		for (int i = 0; i < keys.size(); i++) {
			ret[keys[i]] = vals[i];
		}

		return ret;
	}

	if (name == "org.godotengine.godot.variant.Callable") {
		return jcallable_to_callable(env, obj);
	}

	Ref<JavaObject> generic_object(memnew(JavaObject(JavaClassWrapper::get_singleton()->wrap(name), obj)));

	env->DeleteLocalRef(c);

	return generic_object;
}

Variant::Type get_jni_type(const String &p_type) {
	static struct {
		const char *name;
		Variant::Type type;
	} _type_to_vtype[] = {
		{ "void", Variant::NIL },
		{ "boolean", Variant::BOOL },
		{ "int", Variant::INT },
		{ "long", Variant::INT },
		{ "float", Variant::FLOAT },
		{ "double", Variant::FLOAT },
		{ "java.lang.String", Variant::STRING },
		{ "java.lang.CharSequence", Variant::STRING },
		{ "[I", Variant::PACKED_INT32_ARRAY },
		{ "[J", Variant::PACKED_INT64_ARRAY },
		{ "[B", Variant::PACKED_BYTE_ARRAY },
		{ "[F", Variant::PACKED_FLOAT32_ARRAY },
		{ "[D", Variant::PACKED_FLOAT64_ARRAY },
		{ "[Ljava.lang.String;", Variant::PACKED_STRING_ARRAY },
		{ "[Ljava.lang.CharSequence;", Variant::PACKED_STRING_ARRAY },
		{ "org.godotengine.godot.Dictionary", Variant::DICTIONARY },
		{ "org.godotengine.godot.variant.Callable", Variant::CALLABLE },
		{ nullptr, Variant::NIL }
	};

	int idx = 0;

	while (_type_to_vtype[idx].name) {
		if (p_type == _type_to_vtype[idx].name) {
			return _type_to_vtype[idx].type;
		}

		idx++;
	}

	return Variant::OBJECT;
}

void setup_android_class_loader() {
	// Find a known class defined in the Godot package and obtain its ClassLoader.
	// This ClassLoader will be used by jni_find_class() to locate classes at runtime
	// in a thread-safe manner, avoiding issues with FindClass in non-main threads.

	if (android_class_loader) {
		cleanup_android_class_loader();
	}

	JNIEnv *env = get_jni_env();
	ERR_FAIL_NULL(env);

	jclass known_class = env->FindClass("org/godotengine/godot/Godot");
	ERR_FAIL_NULL(known_class);

	jclass class_class = env->FindClass("java/lang/Class");
	ERR_FAIL_NULL(class_class);

	jmethodID get_class_loader_method = env->GetMethodID(class_class, "getClassLoader", "()Ljava/lang/ClassLoader;");
	ERR_FAIL_NULL(get_class_loader_method);

	jobject class_loader = env->CallObjectMethod(known_class, get_class_loader_method);
	ERR_FAIL_NULL(class_loader);

	// NOTE: Make global ref so it can be used later.
	android_class_loader = env->NewGlobalRef(class_loader);
	ERR_FAIL_NULL(android_class_loader);

	jclass class_loader_class = env->FindClass("java/lang/ClassLoader");
	ERR_FAIL_NULL(class_loader_class);

	load_class_method = env->GetMethodID(class_loader_class, "loadClass", "(Ljava/lang/String;)Ljava/lang/Class;");
	if (!load_class_method) {
		env->DeleteGlobalRef(android_class_loader);
		android_class_loader = nullptr;
		ERR_FAIL_MSG("Failed to find method ID for ClassLoader::loadClass.");
	}

	env->DeleteLocalRef(class_loader_class);
	env->DeleteLocalRef(class_loader);
	env->DeleteLocalRef(class_class);
	env->DeleteLocalRef(known_class);
}

void cleanup_android_class_loader() {
	if (android_class_loader != nullptr) {
		JNIEnv *env = get_jni_env();
		if (env) {
			env->DeleteGlobalRef(android_class_loader);
		} else {
			ERR_PRINT("Failed to release Android ClassLoader - JNIEnv is not available.");
		}
		android_class_loader = nullptr;
		load_class_method = nullptr;
	}
}

jclass jni_find_class(JNIEnv *p_env, const char *p_class_name) {
	ERR_FAIL_NULL_V(p_env, nullptr);
	ERR_FAIL_NULL_V(p_class_name, nullptr);

	jobject class_object = nullptr;
	if (!android_class_loader || !load_class_method) {
		ERR_PRINT("Android ClassLoader is not initialized. Falling back to FindClass.");
		class_object = p_env->FindClass(p_class_name);
	} else {
		jstring java_class_name = p_env->NewStringUTF(p_class_name);
		class_object = p_env->CallObjectMethod(
				android_class_loader,
				load_class_method,
				java_class_name);
		p_env->DeleteLocalRef(java_class_name);
	}
	if (p_env->ExceptionCheck()) {
		p_env->ExceptionDescribe();
		p_env->ExceptionClear();
	}
	ERR_FAIL_NULL_V_MSG(class_object, nullptr, vformat("Failed to find Java class: \"%s\".", p_class_name));
	return static_cast<jclass>(class_object);
}
