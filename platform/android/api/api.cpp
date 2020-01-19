#include "api.h"

#include "core/engine.h"
#include "java_class_wrapper.h"

#if !defined(ANDROID_ENABLED)
static JavaClassWrapper *java_class_wrapper = NULL;
#endif

void register_android_api() {

#if !defined(ANDROID_ENABLED)
	java_class_wrapper = memnew(JavaClassWrapper); // Dummy
#endif

	ClassDB::register_class<JavaClass>();
	ClassDB::register_class<JavaClassWrapper>();
	Engine::get_singleton()->add_singleton(Engine::Singleton("JavaClassWrapper", JavaClassWrapper::get_singleton()));
}

void unregister_android_api() {

#if !defined(ANDROID_ENABLED)
	memdelete(java_class_wrapper);
#endif
}

void JavaClassWrapper::_bind_methods() {

	ClassDB::bind_method(D_METHOD("wrap", "name"), &JavaClassWrapper::wrap);
}

#if !defined(ANDROID_ENABLED)

Variant JavaClass::call(const StringName &, const Variant **, int, Variant::CallError &) {
	return Variant();
}

JavaClass::JavaClass() {
}

Variant JavaObject::call(const StringName &, const Variant **, int, Variant::CallError &) {
	return Variant();
}

JavaClassWrapper *JavaClassWrapper::singleton = NULL;

Ref<JavaClass> JavaClassWrapper::wrap(const String &) {
	return Ref<JavaClass>();
}

JavaClassWrapper::JavaClassWrapper() {
	singleton = this;
}

#endif
