/*************************************************************************/
/*  android_support.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "android_support.h"

#if defined(ANDROID_ENABLED)

#include <dlfcn.h> // dlopen, dlsym
#include <mono/utils/mono-dl-fallback.h>
#include <sys/system_properties.h>
#include <cstddef>

#if __ANDROID_API__ < 24
#include "thirdparty/misc/ifaddrs-android.h"
#else
#include <ifaddrs.h>
#endif

#include "core/os/os.h"
#include "core/ustring.h"
#include "platform/android/java_godot_wrapper.h"
#include "platform/android/os_android.h"
#include "platform/android/thread_jandroid.h"

#include "../../utils/path_utils.h"
#include "../../utils/string_utils.h"
#include "../gd_mono_cache.h"
#include "../gd_mono_marshal.h"

// Warning: JNI boilerplate ahead... continue at your own risk

namespace gdmono {
namespace android {
namespace support {

template <typename T>
struct ScopedLocalRef {
	JNIEnv *env;
	T local_ref;

	_FORCE_INLINE_ T get() const { return local_ref; }
	_FORCE_INLINE_ operator T() const { return local_ref; }
	_FORCE_INLINE_ operator jvalue() const { return (jvalue)local_ref; }

	_FORCE_INLINE_ operator bool() const { return local_ref != NULL; }

	_FORCE_INLINE_ bool operator==(std::nullptr_t) const {
		return local_ref == nullptr;
	}

	_FORCE_INLINE_ bool operator!=(std::nullptr_t) const {
		return local_ref != nullptr;
	}

	ScopedLocalRef(const ScopedLocalRef &) = delete;
	ScopedLocalRef &operator=(const ScopedLocalRef &) = delete;

	ScopedLocalRef(JNIEnv *p_env, T p_local_ref) :
			env(p_env),
			local_ref(p_local_ref) {
	}

	~ScopedLocalRef() {
		if (local_ref) {
			env->DeleteLocalRef(local_ref);
		}
	}
};

bool jni_exception_check(JNIEnv *p_env) {
	if (p_env->ExceptionCheck()) {
		// Print the exception to logcat
		p_env->ExceptionDescribe();

		p_env->ExceptionClear();
		return true;
	}

	return false;
}

String app_native_lib_dir_cache;

String determine_app_native_lib_dir() {
	// The JNI code is the equivalent of:
	//
	// return godotActivity.getApplicationInfo().nativeLibraryDir;

	JNIEnv *env = get_jni_env();

	GodotJavaWrapper *godot_java = static_cast<OS_Android *>(OS::get_singleton())->get_godot_java();
	jobject activity = godot_java->get_activity();

	ScopedLocalRef<jclass> contextClass(env, env->FindClass("android/content/Context"));
	jmethodID getApplicationInfo = env->GetMethodID(contextClass, "getApplicationInfo", "()Landroid/content/pm/ApplicationInfo;");
	ScopedLocalRef<jobject> applicationInfo(env, env->CallObjectMethod(activity, getApplicationInfo));
	jfieldID nativeLibraryDirField = env->GetFieldID(env->GetObjectClass(applicationInfo), "nativeLibraryDir", "Ljava/lang/String;");
	ScopedLocalRef<jstring> nativeLibraryDir(env, (jstring)env->GetObjectField(applicationInfo, nativeLibraryDirField));

	String result;

	const char *const nativeLibraryDirUtf8 = env->GetStringUTFChars(nativeLibraryDir, NULL);
	if (nativeLibraryDirUtf8) {
		result.parse_utf8(nativeLibraryDirUtf8);
		env->ReleaseStringUTFChars(nativeLibraryDir, nativeLibraryDirUtf8);
	}

	return result;
}

String get_app_native_lib_dir() {
	if (app_native_lib_dir_cache.empty())
		app_native_lib_dir_cache = determine_app_native_lib_dir();
	return app_native_lib_dir_cache;
}

int gd_mono_convert_dl_flags(int flags) {
	// from mono's runtime-bootstrap.c

	int lflags = flags & MONO_DL_LOCAL ? 0 : RTLD_GLOBAL;

	if (flags & MONO_DL_LAZY)
		lflags |= RTLD_LAZY;
	else
		lflags |= RTLD_NOW;

	return lflags;
}

#ifndef GD_MONO_SO_NAME
#define GD_MONO_SO_NAME "libmonosgen-2.0.so"
#endif

const char *mono_so_name = GD_MONO_SO_NAME;
const char *godot_so_name = "libgodot_android.so";

void *mono_dl_handle = NULL;
void *godot_dl_handle = NULL;

void *try_dlopen(const String &p_so_path, int p_flags) {
	if (!FileAccess::exists(p_so_path)) {
		if (OS::get_singleton()->is_stdout_verbose())
			OS::get_singleton()->print("Cannot find shared library: '%s'\n", p_so_path.utf8().get_data());
		return NULL;
	}

	int lflags = gd_mono_convert_dl_flags(p_flags);

	void *handle = dlopen(p_so_path.utf8().get_data(), lflags);

	if (!handle) {
		if (OS::get_singleton()->is_stdout_verbose())
			OS::get_singleton()->print("Failed to open shared library: '%s'. Error: '%s'\n", p_so_path.utf8().get_data(), dlerror());
		return NULL;
	}

	if (OS::get_singleton()->is_stdout_verbose())
		OS::get_singleton()->print("Successfully loaded shared library: '%s'\n", p_so_path.utf8().get_data());

	return handle;
}

void *gd_mono_android_dlopen(const char *p_name, int p_flags, char **r_err, void *p_user_data) {
	if (p_name == NULL) {
		// __Internal

		if (!mono_dl_handle) {
			String app_native_lib_dir = get_app_native_lib_dir();
			String so_path = path::join(app_native_lib_dir, mono_so_name);

			mono_dl_handle = try_dlopen(so_path, p_flags);
		}

		return mono_dl_handle;
	}

	String name = String::utf8(p_name);

	if (name.ends_with(".dll.so") || name.ends_with(".exe.so")) {
		String app_native_lib_dir = get_app_native_lib_dir();

		String orig_so_name = name.get_file();
		String so_name = "lib-aot-" + orig_so_name;
		String so_path = path::join(app_native_lib_dir, so_name);

		return try_dlopen(so_path, p_flags);
	}

	return NULL;
}

void *gd_mono_android_dlsym(void *p_handle, const char *p_name, char **r_err, void *p_user_data) {
	void *sym_addr = dlsym(p_handle, p_name);

	if (sym_addr)
		return sym_addr;

	if (p_handle == mono_dl_handle && godot_dl_handle) {
		// Looking up for '__Internal' P/Invoke. We want to search in both the Mono and Godot shared libraries.
		// This is needed to resolve the monodroid P/Invoke functions that are defined at the bottom of the file.
		sym_addr = dlsym(godot_dl_handle, p_name);

		if (sym_addr)
			return sym_addr;
	}

	if (r_err)
		*r_err = str_format_new("%s\n", dlerror());

	return NULL;
}

void *gd_mono_android_dlclose(void *p_handle, void *p_user_data) {
	dlclose(p_handle);

	// Not sure if this ever happens. Does Mono close the handle for the main module?
	if (p_handle == mono_dl_handle)
		mono_dl_handle = NULL;

	return NULL;
}

int32_t build_version_sdk_int = 0;

int32_t get_build_version_sdk_int() {
	// The JNI code is the equivalent of:
	//
	// android.os.Build.VERSION.SDK_INT

	if (build_version_sdk_int == 0) {
		JNIEnv *env = get_jni_env();

		jclass versionClass = env->FindClass("android/os/Build$VERSION");
		ERR_FAIL_NULL_V(versionClass, 0);

		jfieldID sdkIntField = env->GetStaticFieldID(versionClass, "SDK_INT", "I");
		ERR_FAIL_NULL_V(sdkIntField, 0);

		build_version_sdk_int = (int32_t)env->GetStaticIntField(versionClass, sdkIntField);
	}

	return build_version_sdk_int;
}

jobject certStore = NULL; // KeyStore

MonoBoolean _gd_mono_init_cert_store() {
	// The JNI code is the equivalent of:
	//
	// try {
	// 	certStoreLocal = KeyStore.getInstance("AndroidCAStore");
	// 	certStoreLocal.load(null);
	//	certStore = certStoreLocal;
	//	return true;
	// } catch (Exception e) {
	//	return false;
	// }

	JNIEnv *env = get_jni_env();

	ScopedLocalRef<jclass> keyStoreClass(env, env->FindClass("java/security/KeyStore"));

	jmethodID getInstance = env->GetStaticMethodID(keyStoreClass, "getInstance", "(Ljava/lang/String;)Ljava/security/KeyStore;");
	jmethodID load = env->GetMethodID(keyStoreClass, "load", "(Ljava/security/KeyStore$LoadStoreParameter;)V");

	ScopedLocalRef<jstring> androidCAStoreString(env, env->NewStringUTF("AndroidCAStore"));

	ScopedLocalRef<jobject> certStoreLocal(env, env->CallStaticObjectMethod(keyStoreClass, getInstance, androidCAStoreString.get()));

	if (jni_exception_check(env))
		return 0;

	env->CallVoidMethod(certStoreLocal, load, NULL);

	if (jni_exception_check(env))
		return 0;

	certStore = env->NewGlobalRef(certStoreLocal);

	return 1;
}

MonoArray *_gd_mono_android_cert_store_lookup(MonoString *p_alias) {
	// The JNI code is the equivalent of:
	//
	// Certificate certificate = certStore.getCertificate(alias);
	// if (certificate == null)
	//	return null;
	// return certificate.getEncoded();

	MonoError mono_error;
	char *alias_utf8 = mono_string_to_utf8_checked(p_alias, &mono_error);

	if (!mono_error_ok(&mono_error)) {
		ERR_PRINT(String() + "Failed to convert MonoString* to UTF-8: '" + mono_error_get_message(&mono_error) + "'.");
		mono_error_cleanup(&mono_error);
		return NULL;
	}

	JNIEnv *env = get_jni_env();

	ScopedLocalRef<jstring> js_alias(env, env->NewStringUTF(alias_utf8));
	mono_free(alias_utf8);

	ScopedLocalRef<jclass> keyStoreClass(env, env->FindClass("java/security/KeyStore"));
	ERR_FAIL_NULL_V(keyStoreClass, NULL);
	ScopedLocalRef<jclass> certificateClass(env, env->FindClass("java/security/cert/Certificate"));
	ERR_FAIL_NULL_V(certificateClass, NULL);

	jmethodID getCertificate = env->GetMethodID(keyStoreClass, "getCertificate", "(Ljava/lang/String;)Ljava/security/cert/Certificate;");
	ERR_FAIL_NULL_V(getCertificate, NULL);

	jmethodID getEncoded = env->GetMethodID(certificateClass, "getEncoded", "()[B");
	ERR_FAIL_NULL_V(getEncoded, NULL);

	ScopedLocalRef<jobject> certificate(env, env->CallObjectMethod(certStore, getCertificate, js_alias.get()));

	if (!certificate)
		return NULL;

	ScopedLocalRef<jbyteArray> encoded(env, (jbyteArray)env->CallObjectMethod(certificate, getEncoded));
	jsize encodedLength = env->GetArrayLength(encoded);

	MonoArray *encoded_ret = mono_array_new(mono_domain_get(), CACHED_CLASS_RAW(uint8_t), encodedLength);
	uint8_t *dest = (uint8_t *)mono_array_addr(encoded_ret, uint8_t, 0);

	env->GetByteArrayRegion(encoded, 0, encodedLength, reinterpret_cast<jbyte *>(dest));

	return encoded_ret;
}

void register_internal_calls() {
	GDMonoUtils::add_internal_call("Android.Runtime.AndroidEnvironment::_gd_mono_init_cert_store", _gd_mono_init_cert_store);
	GDMonoUtils::add_internal_call("Android.Runtime.AndroidEnvironment::_gd_mono_android_cert_store_lookup", _gd_mono_android_cert_store_lookup);
}

void initialize() {
	// We need to set this environment variable to make the monodroid BCL use btls instead of legacy as the default provider
	OS::get_singleton()->set_environment("XA_TLS_PROVIDER", "btls");

	mono_dl_fallback_register(gd_mono_android_dlopen, gd_mono_android_dlsym, gd_mono_android_dlclose, NULL);

	String app_native_lib_dir = get_app_native_lib_dir();
	String so_path = path::join(app_native_lib_dir, godot_so_name);

	godot_dl_handle = try_dlopen(so_path, gd_mono_convert_dl_flags(MONO_DL_LAZY));
	ERR_FAIL_COND(!godot_dl_handle);
}

void cleanup() {
	// This is called after shutting down the Mono runtime

	if (mono_dl_handle)
		gd_mono_android_dlclose(mono_dl_handle, NULL);

	if (godot_dl_handle)
		gd_mono_android_dlclose(godot_dl_handle, NULL);

	JNIEnv *env = get_jni_env();

	if (certStore) {
		env->DeleteGlobalRef(certStore);
		certStore = NULL;
	}
}

} // namespace support
} // namespace android
} // namespace gdmono

using namespace gdmono::android::support;

// The following are P/Invoke functions required by the monodroid profile of the BCL.
// These are P/Invoke functions and not internal calls, hence why they use
// 'mono_bool' and 'const char*' instead of 'MonoBoolean' and 'MonoString*'.

#define GD_PINVOKE_EXPORT extern "C" __attribute__((visibility("default")))

GD_PINVOKE_EXPORT int32_t _monodroid_get_android_api_level() {
	return get_build_version_sdk_int();
}

GD_PINVOKE_EXPORT void monodroid_free(void *ptr) {
	free(ptr);
}

GD_PINVOKE_EXPORT int32_t monodroid_get_system_property(const char *p_name, char **r_value) {
	char prop_value_str[PROP_VALUE_MAX + 1] = { 0 };

	int len = __system_property_get(p_name, prop_value_str);

	if (r_value) {
		if (len >= 0) {
			*r_value = (char *)malloc(len + 1);
			ERR_FAIL_NULL_V_MSG(*r_value, -1, "Out of memory.");
			memcpy(*r_value, prop_value_str, len);
			(*r_value)[len] = '\0';
		} else {
			*r_value = NULL;
		}
	}

	return len;
}

GD_PINVOKE_EXPORT mono_bool _monodroid_get_network_interface_up_state(const char *p_ifname, mono_bool *r_is_up) {
	// The JNI code is the equivalent of:
	//
	// NetworkInterface.getByName(p_ifname).isUp()

	if (!r_is_up || !p_ifname || strlen(p_ifname) == 0)
		return 0;

	*r_is_up = 0;

	JNIEnv *env = get_jni_env();

	jclass networkInterfaceClass = env->FindClass("java/net/NetworkInterface");
	ERR_FAIL_NULL_V(networkInterfaceClass, 0);

	jmethodID getByName = env->GetStaticMethodID(networkInterfaceClass, "getByName", "(Ljava/lang/String;)Ljava/net/NetworkInterface;");
	ERR_FAIL_NULL_V(getByName, 0);

	jmethodID isUp = env->GetMethodID(networkInterfaceClass, "isUp", "()Z");
	ERR_FAIL_NULL_V(isUp, 0);

	ScopedLocalRef<jstring> js_ifname(env, env->NewStringUTF(p_ifname));
	ScopedLocalRef<jobject> networkInterface(env, env->CallStaticObjectMethod(networkInterfaceClass, getByName, js_ifname.get()));

	if (!networkInterface)
		return 0;

	*r_is_up = (mono_bool)env->CallBooleanMethod(networkInterface, isUp);

	return 1;
}

GD_PINVOKE_EXPORT mono_bool _monodroid_get_network_interface_supports_multicast(const char *p_ifname, mono_bool *r_supports_multicast) {
	// The JNI code is the equivalent of:
	//
	// NetworkInterface.getByName(p_ifname).supportsMulticast()

	if (!r_supports_multicast || !p_ifname || strlen(p_ifname) == 0)
		return 0;

	*r_supports_multicast = 0;

	JNIEnv *env = get_jni_env();

	jclass networkInterfaceClass = env->FindClass("java/net/NetworkInterface");
	ERR_FAIL_NULL_V(networkInterfaceClass, 0);

	jmethodID getByName = env->GetStaticMethodID(networkInterfaceClass, "getByName", "(Ljava/lang/String;)Ljava/net/NetworkInterface;");
	ERR_FAIL_NULL_V(getByName, 0);

	jmethodID supportsMulticast = env->GetMethodID(networkInterfaceClass, "supportsMulticast", "()Z");
	ERR_FAIL_NULL_V(supportsMulticast, 0);

	ScopedLocalRef<jstring> js_ifname(env, env->NewStringUTF(p_ifname));
	ScopedLocalRef<jobject> networkInterface(env, env->CallStaticObjectMethod(networkInterfaceClass, getByName, js_ifname.get()));

	if (!networkInterface)
		return 0;

	*r_supports_multicast = (mono_bool)env->CallBooleanMethod(networkInterface, supportsMulticast);

	return 1;
}

static const int dns_servers_len = 8;

static void interop_get_active_network_dns_servers(char **r_dns_servers, int *dns_servers_count) {
	// The JNI code is the equivalent of:
	//
	// ConnectivityManager connectivityManager = (ConnectivityManager)getApplicationContext()
	// 		.getSystemService(Context.CONNECTIVITY_SERVICE);
	// Network activeNerwork = connectivityManager.getActiveNetwork();
	// LinkProperties linkProperties = connectivityManager.getLinkProperties(activeNerwork);
	// List<String> dnsServers = linkProperties.getDnsServers().stream()
	// 		.map(inetAddress -> inetAddress.getHostAddress()).collect(Collectors.toList());

#ifdef DEBUG_ENABLED
	CRASH_COND(get_build_version_sdk_int() < 23);
#endif

	JNIEnv *env = get_jni_env();

	GodotJavaWrapper *godot_java = ((OS_Android *)OS::get_singleton())->get_godot_java();
	jobject activity = godot_java->get_activity();

	ScopedLocalRef<jclass> activityClass(env, env->GetObjectClass(activity));
	ERR_FAIL_NULL(activityClass);

	jmethodID getApplicationContext = env->GetMethodID(activityClass, "getApplicationContext", "()Landroid/content/Context;");

	ScopedLocalRef<jobject> applicationContext(env, env->CallObjectMethod(activity, getApplicationContext));

	ScopedLocalRef<jclass> contextClass(env, env->FindClass("android/content/Context"));
	ERR_FAIL_NULL(contextClass);

	jfieldID connectivityServiceField = env->GetStaticFieldID(contextClass, "CONNECTIVITY_SERVICE", "Ljava/lang/String;");
	ScopedLocalRef<jstring> connectivityServiceString(env, (jstring)env->GetStaticObjectField(contextClass, connectivityServiceField));

	jmethodID getSystemService = env->GetMethodID(contextClass, "getSystemService", "(Ljava/lang/String;)Ljava/lang/Object;");

	ScopedLocalRef<jobject> connectivityManager(env, env->CallObjectMethod(applicationContext, getSystemService, connectivityServiceString.get()));

	if (!connectivityManager)
		return;

	ScopedLocalRef<jclass> connectivityManagerClass(env, env->FindClass("android/net/ConnectivityManager"));
	ERR_FAIL_NULL(connectivityManagerClass);

	jmethodID getActiveNetwork = env->GetMethodID(connectivityManagerClass, "getActiveNetwork", "()Landroid/net/Network;");
	ERR_FAIL_NULL(getActiveNetwork);

	ScopedLocalRef<jobject> activeNetwork(env, env->CallObjectMethod(connectivityManager, getActiveNetwork));

	if (!activeNetwork)
		return;

	jmethodID getLinkProperties = env->GetMethodID(connectivityManagerClass,
			"getLinkProperties", "(Landroid/net/Network;)Landroid/net/LinkProperties;");
	ERR_FAIL_NULL(getLinkProperties);

	ScopedLocalRef<jobject> linkProperties(env, env->CallObjectMethod(connectivityManager, getLinkProperties, activeNetwork.get()));

	if (!linkProperties)
		return;

	ScopedLocalRef<jclass> linkPropertiesClass(env, env->FindClass("android/net/LinkProperties"));
	ERR_FAIL_NULL(linkPropertiesClass);

	jmethodID getDnsServers = env->GetMethodID(linkPropertiesClass, "getDnsServers", "()Ljava/util/List;");
	ERR_FAIL_NULL(getDnsServers);

	ScopedLocalRef<jobject> dnsServers(env, env->CallObjectMethod(linkProperties, getDnsServers));

	if (!dnsServers)
		return;

	ScopedLocalRef<jclass> listClass(env, env->FindClass("java/util/List"));
	ERR_FAIL_NULL(listClass);

	jmethodID listSize = env->GetMethodID(listClass, "size", "()I");
	ERR_FAIL_NULL(listSize);

	int dnsServersCount = env->CallIntMethod(dnsServers, listSize);

	if (dnsServersCount > dns_servers_len)
		dnsServersCount = dns_servers_len;

	if (dnsServersCount <= 0)
		return;

	jmethodID listGet = env->GetMethodID(listClass, "get", "(I)Ljava/lang/Object;");
	ERR_FAIL_NULL(listGet);

	ScopedLocalRef<jclass> inetAddressClass(env, env->FindClass("java/net/InetAddress"));
	ERR_FAIL_NULL(inetAddressClass);

	jmethodID getHostAddress = env->GetMethodID(inetAddressClass, "getHostAddress", "()Ljava/lang/String;");
	ERR_FAIL_NULL(getHostAddress);

	for (int i = 0; i < dnsServersCount; i++) {
		ScopedLocalRef<jobject> dnsServer(env, env->CallObjectMethod(dnsServers, listGet, (jint)i));
		if (!dnsServer)
			continue;

		ScopedLocalRef<jstring> hostAddress(env, (jstring)env->CallObjectMethod(dnsServer, getHostAddress));
		const char *host_address = env->GetStringUTFChars(hostAddress, 0);

		r_dns_servers[i] = strdup(host_address); // freed by the BCL
		(*dns_servers_count)++;

		env->ReleaseStringUTFChars(hostAddress, host_address);
	}

	// jesus...
}

GD_PINVOKE_EXPORT int32_t _monodroid_get_dns_servers(void **r_dns_servers_array) {
	if (!r_dns_servers_array)
		return -1;

	*r_dns_servers_array = NULL;

	char *dns_servers[dns_servers_len];
	int dns_servers_count = 0;

	if (_monodroid_get_android_api_level() < 26) {
		// The 'net.dns*' system properties are no longer available in Android 8.0 (API level 26) and greater:
		// https://developer.android.com/about/versions/oreo/android-8.0-changes.html#o-pri

		char prop_name[] = "net.dns*";

		for (int i = 0; i < dns_servers_len; i++) {
			prop_name[7] = (char)(i + 0x31);
			char *prop_value;
			int32_t len = monodroid_get_system_property(prop_name, &prop_value);

			if (len > 0) {
				dns_servers[dns_servers_count] = strndup(prop_value, (size_t)len); // freed by the BCL
				dns_servers_count++;
				free(prop_value);
			}
		}
	} else {
		// Alternative for Oreo and greater
		interop_get_active_network_dns_servers(dns_servers, &dns_servers_count);
	}

	if (dns_servers_count > 0) {
		size_t ret_size = sizeof(char *) * (size_t)dns_servers_count;
		*r_dns_servers_array = malloc(ret_size); // freed by the BCL
		ERR_FAIL_NULL_V_MSG(*r_dns_servers_array, -1, "Out of memory.");
		memcpy(*r_dns_servers_array, dns_servers, ret_size);
	}

	return dns_servers_count;
}

GD_PINVOKE_EXPORT const char *_monodroid_timezone_get_default_id() {
	// The JNI code is the equivalent of:
	//
	// TimeZone.getDefault().getID()

	JNIEnv *env = get_jni_env();

	ScopedLocalRef<jclass> timeZoneClass(env, env->FindClass("java/util/TimeZone"));
	ERR_FAIL_NULL_V(timeZoneClass, NULL);

	jmethodID getDefault = env->GetStaticMethodID(timeZoneClass, "getDefault", "()Ljava/util/TimeZone;");
	ERR_FAIL_NULL_V(getDefault, NULL);

	jmethodID getID = env->GetMethodID(timeZoneClass, "getID", "()Ljava/lang/String;");
	ERR_FAIL_NULL_V(getID, NULL);

	ScopedLocalRef<jobject> defaultTimeZone(env, env->CallStaticObjectMethod(timeZoneClass, getDefault));

	if (!defaultTimeZone)
		return NULL;

	ScopedLocalRef<jstring> defaultTimeZoneID(env, (jstring)env->CallObjectMethod(defaultTimeZone, getID));

	if (!defaultTimeZoneID)
		return NULL;

	const char *default_time_zone_id = env->GetStringUTFChars(defaultTimeZoneID, 0);

	char *result = strdup(default_time_zone_id); // freed by the BCL

	env->ReleaseStringUTFChars(defaultTimeZoneID, default_time_zone_id);

	return result;
}

GD_PINVOKE_EXPORT int32_t _monodroid_getifaddrs(struct ifaddrs **p_ifap) {
	return getifaddrs(p_ifap);
}

GD_PINVOKE_EXPORT void _monodroid_freeifaddrs(struct ifaddrs *p_ifap) {
	freeifaddrs(p_ifap);
}

#endif
