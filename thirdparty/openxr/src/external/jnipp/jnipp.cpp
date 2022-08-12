#ifdef _WIN32
# define WIN32_LEAN_AND_MEAN 1

  // Windows Dependencies
# include <windows.h>
#else
  // UNIX Dependencies
# include <dlfcn.h>
# include <unistd.h>
# include <tuple>
#endif

// External Dependencies
#include <jni.h>

// Standard Dependencies
#include <atomic>
#include <string>

// Local Dependencies
#include "jnipp.h"

namespace jni
{
    // Static Variables
    static std::atomic_bool isVm(false);
    static JavaVM* javaVm = nullptr;

    static bool getEnv(JavaVM *vm, JNIEnv **env) {
        return vm->GetEnv((void **)env, JNI_VERSION_1_2) == JNI_OK;
    }

    static bool isAttached(JavaVM *vm) {
        JNIEnv *env = nullptr;
        return getEnv(vm, &env);
    }
    /**
        Maintains the lifecycle of a JNIEnv.
     */
    class ScopedEnv final
    {
    public:
        ScopedEnv() noexcept : _vm(nullptr), _env(nullptr), _attached(false) {}
        ~ScopedEnv();

        void init(JavaVM* vm);
        JNIEnv* get() const noexcept { return _env; }

    private:
        // Instance Variables
        JavaVM* _vm;
        JNIEnv* _env;
        bool    _attached;    ///< Manually attached, as opposed to already attached.
    };

    ScopedEnv::~ScopedEnv()
    {
        if (_vm && _attached)
            _vm->DetachCurrentThread();
    }

    void ScopedEnv::init(JavaVM* vm)
    {
        if (_env != nullptr)
            return;

        if (vm == nullptr)
            throw InitializationException("JNI not initialized");

        if (!getEnv(vm, &_env))
        {
#ifdef __ANDROID__
            if (vm->AttachCurrentThread(&_env, nullptr) != 0)
#else
            if (vm->AttachCurrentThread((void**)&_env, nullptr) != 0)
#endif
                throw InitializationException("Could not attach JNI to thread");

            _attached = true;
        }

        _vm = vm;
    }

    /*
        Helper Functions
     */

#ifdef _WIN32

    static bool fileExists(const std::string& filePath)
    {
        DWORD attr = ::GetFileAttributesA(filePath.c_str());

        return attr != INVALID_FILE_ATTRIBUTES && !(attr & FILE_ATTRIBUTE_DIRECTORY);
    }

#else

     /**
         Convert from a UTF-16 Java string to a UTF-32 string.
      */
    std::wstring toWString(const jchar* str, jsize length)
    {
        std::wstring result;

        result.reserve(length);

        for (jsize i = 0; i < length; ++i)
        {
            wchar_t ch = str[i];

            // Check for a two-segment character.
            if (ch >= wchar_t(0xD800) && ch <= wchar_t(0xDBFF)) {
                if (i + 1 >= length)
                    break;

                // Create a single, 32-bit character.
                ch = (ch - wchar_t(0xD800)) << 10;
                ch += str[i++] - wchar_t(0x1DC00);
            }

            result += ch;
        }

        return result;
    }

    /**
        Convert from a UTF-32 string to a UTF-16 Java string.
     */
    std::basic_string<jchar> toJString(const wchar_t* str, size_t length)
    {
        std::basic_string<jchar> result;

        result.reserve(length * 2);    // Worst case scenario.

        for (size_t i = 0; i < length; ++i)
        {
            wchar_t ch = str[i];

            // Check for multi-byte UTF-16 character.
            if (ch > wchar_t(0xFFFF)) {
                ch -= uint32_t(0x10000);

                // Add the first of the two-segment character.
                result += jchar(0xD800 + (ch >> 10));
                ch = wchar_t(0xDC00) + (ch & 0x03FF);
            }

            result += jchar(ch);
        }

        return result;
    }

#endif // _WIN32

    JNIEnv* env()
    {
        static thread_local ScopedEnv env;

        if (env.get() != nullptr && !isAttached(javaVm))
        {
            // we got detached, so clear it.
            // will be re-populated from static javaVm below.
            env = ScopedEnv{};
        }

        if (env.get() == nullptr)
        {
            env.init(javaVm);
        }

        return env.get();
    }

    static jclass findClass(const char* name)
    {
        jclass ref = env()->FindClass(name);

        if (ref == nullptr)
        {
            env()->ExceptionClear();
            throw NameResolutionException(name);
        }

        return ref;
    }

    static void handleJavaExceptions()
    {
        JNIEnv* env = jni::env();

        jthrowable exception = env->ExceptionOccurred();

        if (exception != nullptr)
        {
            Object obj(exception, Object::Temporary);

            env->ExceptionClear();
            std::string msg = obj.call<std::string>("toString");
            throw InvocationException(msg.c_str());
        }
    }

    static std::string toString(jobject handle, bool deleteLocal = true)
    {
        std::string result;

        if (handle != nullptr)
        {
            JNIEnv* env = jni::env();

            const char* chars = env->GetStringUTFChars(jstring(handle), nullptr);
            result.assign(chars, env->GetStringUTFLength(jstring(handle)));
            env->ReleaseStringUTFChars(jstring(handle), chars);

            if (deleteLocal)
                env->DeleteLocalRef(handle);
        }

        return result;
    }

    static std::wstring toWString(jobject handle, bool deleteLocal = true)
    {
        std::wstring result;

        if (handle != nullptr)
        {
            JNIEnv* env = jni::env();

            const jchar* chars = env->GetStringChars(jstring(handle), nullptr);
#ifdef _WIN32
            result.assign((const wchar_t*) chars, env->GetStringLength(jstring(handle)));
#else
            result = toWString(chars, env->GetStringLength(jstring(handle)));
#endif
            env->ReleaseStringChars(jstring(handle), chars);

            if (deleteLocal)
                env->DeleteLocalRef(handle);
        }

        return result;
    }


    /*
        Stand-alone Function Implementations
     */

    void init(JNIEnv* env)
    {
        bool expected = false;

        if (isVm.compare_exchange_strong(expected, true))
        {
            if (javaVm == nullptr && env->GetJavaVM(&javaVm) != 0)
                throw InitializationException("Could not acquire Java VM");
        }
    }

    void init(JavaVM* vm) {
        bool expected = false;

        if (isVm.compare_exchange_strong(expected, true))
        {
            javaVm = vm;
        }
    }
    /*
        Object Implementation
     */

    Object::Object() noexcept : _handle(nullptr), _class(nullptr), _isGlobal(false)
    {
    }

    Object::Object(const Object& other) : _handle(nullptr), _class(nullptr), _isGlobal(!other.isNull())
    {
        if (!other.isNull())
            _handle = env()->NewGlobalRef(other._handle);
    }

    Object::Object(Object&& other) noexcept : _handle(other._handle), _class(other._class), _isGlobal(other._isGlobal)
    {
        other._handle   = nullptr;
        other._class    = nullptr;
        other._isGlobal = false;
    }

    Object::Object(jobject ref, int scopeFlags) : _handle(ref), _class(nullptr), _isGlobal((scopeFlags & Temporary) == 0)
    {
        if (!_isGlobal)
            return;

        JNIEnv* env = jni::env();

        _handle = env->NewGlobalRef(ref);

        if (scopeFlags & DeleteLocalInput)
            env->DeleteLocalRef(ref);
    }

    Object::~Object() noexcept
    {
        JNIEnv* env = jni::env();

        if (_isGlobal)
            env->DeleteGlobalRef(_handle);

        if (_class != nullptr)
            env->DeleteGlobalRef(_class);
    }

    Object& Object::operator=(const Object& other)
    {
        if (_handle != other._handle)
        {
            JNIEnv* env = jni::env();

            // Ditch the old reference.
            if (_isGlobal)
                env->DeleteGlobalRef(_handle);
            if (_class != nullptr)
                env->DeleteGlobalRef(_class);

            // Assign the new reference.
            if ((_isGlobal = !other.isNull()) != false)
                _handle = env->NewGlobalRef(other._handle);

            _class = nullptr;
        }

        return *this;
    }

    bool Object::operator==(const Object& other) const
    {
        return env()->IsSameObject(_handle, other._handle) != JNI_FALSE;
    }

    Object& Object::operator=(Object&& other)
    {
        if (_handle != other._handle)
        {
            JNIEnv* env = jni::env();

            // Ditch the old reference.
            if (_isGlobal)
                env->DeleteGlobalRef(_handle);
            if (_class != nullptr)
                env->DeleteGlobalRef(_class);

            // Assign the new reference.
            _handle   = other._handle;
            _isGlobal = other._isGlobal;
            _class    = other._class;

            other._handle   = nullptr;
            other._isGlobal = false;
            other._class    = nullptr;
        }

        return *this;
    }

    bool Object::isNull() const noexcept
    {
        return _handle == nullptr || env()->IsSameObject(_handle, nullptr);
    }

    void Object::callMethod(method_t method, internal::value_t* args, internal::ReturnTypeWrapper<void> const&) const
    {
        env()->CallVoidMethodA(_handle, method, (jvalue*) args);
        handleJavaExceptions();
    }

    bool Object::callMethod(method_t method, internal::value_t* args, internal::ReturnTypeWrapper<bool> const&) const
    {
        auto result = env()->CallBooleanMethodA(_handle, method, (jvalue*) args);
        handleJavaExceptions();
        return result != 0;
    }

    bool Object::getFieldValue(field_t field, internal::ReturnTypeWrapper<bool> const&) const
    {
        return env()->GetBooleanField(_handle, field) != 0;
    }

    template <> void Object::set(field_t field, const bool& value)
    {
        env()->SetBooleanField(_handle, field, value);
    }

    byte_t Object::callMethod(method_t method, internal::value_t* args, internal::ReturnTypeWrapper<byte_t> const&) const
    {
        auto result = env()->CallByteMethodA(_handle, method, (jvalue*) args);
        handleJavaExceptions();
        return result;
    }

    wchar_t Object::callMethod(method_t method, internal::value_t* args, internal::ReturnTypeWrapper<wchar_t> const&) const
    {
        auto result = env()->CallCharMethodA(_handle, method, (jvalue*) args);
        handleJavaExceptions();
        return result;
    }

    short Object::callMethod(method_t method, internal::value_t* args, internal::ReturnTypeWrapper<short> const&) const
    {
        auto result = env()->CallShortMethodA(_handle, method, (jvalue*) args);
        handleJavaExceptions();
        return result;
    }

    int Object::callMethod(method_t method, internal::value_t* args, internal::ReturnTypeWrapper<int> const&) const
    {
        auto result = env()->CallIntMethodA(_handle, method, (jvalue*) args);
        handleJavaExceptions();
        return result;
    }

    long long Object::callMethod(method_t method, internal::value_t* args, internal::ReturnTypeWrapper<long long> const&) const
    {
        auto result = env()->CallLongMethodA(_handle, method, (jvalue*) args);
        handleJavaExceptions();
        return result;
    }

    float Object::callMethod(method_t method, internal::value_t* args, internal::ReturnTypeWrapper<float> const&) const
    {
        auto result = env()->CallFloatMethodA(_handle, method, (jvalue*) args);
        handleJavaExceptions();
        return result;
    }

    double Object::callMethod(method_t method, internal::value_t* args, internal::ReturnTypeWrapper<double> const&) const
    {
        auto result = env()->CallDoubleMethodA(_handle, method, (jvalue*) args);
        handleJavaExceptions();
        return result;
    }

    std::string Object::callMethod(method_t method, internal::value_t* args, internal::ReturnTypeWrapper<std::string> const&) const
    {
        auto result = env()->CallObjectMethodA(_handle, method, (jvalue*) args);
        handleJavaExceptions();
        return toString(result);
    }

    std::wstring Object::callMethod(method_t method, internal::value_t* args, internal::ReturnTypeWrapper<std::wstring> const&) const
    {
        auto result = env()->CallObjectMethodA(_handle, method, (jvalue*) args);
        handleJavaExceptions();
        return toWString(result);
    }

    jni::Object Object::callMethod(method_t method, internal::value_t* args, internal::ReturnTypeWrapper<jni::Object> const&) const
    {
        auto result = env()->CallObjectMethodA(_handle, method, (jvalue*) args);
        handleJavaExceptions();
        return Object(result, DeleteLocalInput);
    }

    byte_t Object::getFieldValue(field_t field, internal::ReturnTypeWrapper<byte_t> const&) const
    {
        return env()->GetByteField(_handle, field);
    }

    wchar_t Object::getFieldValue(field_t field, internal::ReturnTypeWrapper<wchar_t> const&) const
    {
        return env()->GetCharField(_handle, field);
    }

    short Object::getFieldValue(field_t field, internal::ReturnTypeWrapper<short> const&) const
    {
        return env()->GetShortField(_handle, field);
    }

    int Object::getFieldValue(field_t field, internal::ReturnTypeWrapper<int> const&) const
    {
        return env()->GetIntField(_handle, field);
    }

    long long Object::getFieldValue(field_t field, internal::ReturnTypeWrapper<long long> const&) const
    {
        return env()->GetLongField(_handle, field);
    }

    float Object::getFieldValue(field_t field, internal::ReturnTypeWrapper<float> const&) const
    {
        return env()->GetFloatField(_handle, field);
    }

    double Object::getFieldValue(field_t field, internal::ReturnTypeWrapper<double> const&) const
    {
        return env()->GetDoubleField(_handle, field);
    }

    std::string Object::getFieldValue(field_t field, internal::ReturnTypeWrapper<std::string> const&) const
    {
        return toString(env()->GetObjectField(_handle, field));
    }

    std::wstring Object::getFieldValue(field_t field, internal::ReturnTypeWrapper<std::wstring> const&) const
    {
        return toWString(env()->GetObjectField(_handle, field));
    }

    Object Object::getFieldValue(field_t field, internal::ReturnTypeWrapper<Object> const&) const
    {
        return Object(env()->GetObjectField(_handle, field), DeleteLocalInput);
    }

    template <> void Object::set(field_t field, const byte_t& value)
    {
        env()->SetByteField(_handle, field, value);
    }

    template <> void Object::set(field_t field, const wchar_t& value)
    {
        env()->SetCharField(_handle, field, value);
    }

    template <> void Object::set(field_t field, const short& value)
    {
        env()->SetShortField(_handle, field, value);
    }

    template <> void Object::set(field_t field, const int& value)
    {
        env()->SetIntField(_handle, field, value);
    }

    template <> void Object::set(field_t field, const long long& value)
    {
        env()->SetLongField(_handle, field, value);
    }

    template <> void Object::set(field_t field, const float& value)
    {
        env()->SetFloatField(_handle, field, value);
    }

    template <> void Object::set(field_t field, const double& value)
    {
        env()->SetDoubleField(_handle, field, value);
    }

    template <> void Object::set(field_t field, const std::string& value)
    {
        JNIEnv* env = jni::env();

        jobject handle = env->NewStringUTF(value.c_str());
        env->SetObjectField(_handle, field, handle);
        env->DeleteLocalRef(handle);
    }

    template <> void Object::set(field_t field, const std::wstring& value)
    {
        JNIEnv* env = jni::env();

#ifdef _WIN32
        jobject handle = env->NewString((const jchar*) value.c_str(), jsize(value.length()));
#else
        auto jstr = toJString(value.c_str(), value.length());
        jobject handle = env->NewString(jstr.c_str(), jsize(jstr.length()));
#endif
        env->SetObjectField(_handle, field, handle);
        env->DeleteLocalRef(handle);
    }

    template <> void Object::set(field_t field, const wchar_t* const& value)
    {
        JNIEnv* env = jni::env();
#ifdef _WIN32
        jobject handle = env->NewString((const jchar*) value, jsize(std::wcslen(value)));
#else
        auto jstr = toJString(value, std::wcslen(value));
        jobject handle = env->NewString(jstr.c_str(), jsize(jstr.length()));
#endif
        env->SetObjectField(_handle, field, handle);
        env->DeleteLocalRef(handle);
    }

    template <> void Object::set(field_t field, const char* const& value)
    {
        JNIEnv* env = jni::env();

        jobject handle = env->NewStringUTF(value);
        env->SetObjectField(_handle, field, handle);
        env->DeleteLocalRef(handle);
    }

    template <> void Object::set(field_t field, const Object& value)
    {
        env()->SetObjectField(_handle, field, value.getHandle());
    }

    template <> void Object::set(field_t field, const Object* const& value)
    {
        env()->SetObjectField(_handle, field, value ? value->getHandle() : nullptr);
    }

    jclass Object::getClass() const
    {
        if (_class == nullptr)
        {
            JNIEnv* env = jni::env();

            jclass classRef = env->GetObjectClass(_handle);
            _class = jclass(env->NewGlobalRef(classRef));
            env->DeleteLocalRef(classRef);
        }

        return _class;
    }

    method_t Object::getMethod(const char* name, const char* signature) const
    {
        return Class(getClass(), Temporary).getMethod(name, signature);
    }

    method_t Object::getMethod(const char* nameAndSignature) const
    {
        return Class(getClass(), Temporary).getMethod(nameAndSignature);
    }

    field_t Object::getField(const char* name, const char* signature) const
    {
        return Class(getClass(), Temporary).getField(name, signature);
    }

    jobject Object::makeLocalReference() const 
    {
        if (isNull())
            return nullptr;
        return env()->NewLocalRef(_handle);
    }

    /*
        Class Implementation
     */

    Class::Class(const char* name) : Object(findClass(name), DeleteLocalInput)
    {
    }

    Class::Class(jclass ref, int scopeFlags) : Object(ref, scopeFlags)
    {
    }

    Object Class::newInstance() const
    {
        method_t constructor = getMethod("<init>", "()V");
        jobject obj = env()->NewObject(getHandle(), constructor);

        handleJavaExceptions();

        return Object(obj, Object::DeleteLocalInput);
    }

    field_t Class::getField(const char* name, const char* signature) const
    {
        jfieldID id = env()->GetFieldID(getHandle(), name, signature);

        if (id == nullptr)
            throw NameResolutionException(name);

        return id;
    }

    field_t Class::getStaticField(const char* name, const char* signature) const
    {
        jfieldID id = env()->GetStaticFieldID(getHandle(), name, signature);

        if (id == nullptr)
            throw NameResolutionException(name);

        return id;
    }

    method_t Class::getMethod(const char* name, const char* signature) const
    {
        jmethodID id = env()->GetMethodID(getHandle(), name, signature);

        if (id == nullptr)
            throw NameResolutionException(name);

        return id;
    }


    method_t Class::getMethod(const char* nameAndSignature) const
    {
        jmethodID id = nullptr;
        const char* sig = std::strchr(nameAndSignature, '(');

        if (sig != nullptr)
            return getMethod(std::string(nameAndSignature, sig - nameAndSignature).c_str(), sig);

        if (id == nullptr)
            throw NameResolutionException(nameAndSignature);

        return id;
    }

    method_t Class::getStaticMethod(const char* name, const char* signature) const
    {
        jmethodID id = env()->GetStaticMethodID(getHandle(), name, signature);

        if (id == nullptr)
            throw NameResolutionException(name);

        return id;
    }

    method_t Class::getStaticMethod(const char* nameAndSignature) const
    {
        jmethodID id = nullptr;
        const char* sig = std::strchr(nameAndSignature, '(');

        if (sig != nullptr)
            return getStaticMethod(std::string(nameAndSignature, sig - nameAndSignature).c_str(), sig);

        if (id == nullptr)
            throw NameResolutionException(nameAndSignature);

        return id;
    }

    Class Class::getParent() const
    {
        return Class(env()->GetSuperclass(getHandle()), DeleteLocalInput);
    }

    std::string Class::getName() const
    {
        return Object::call<std::string>("getName");
    }

    template <> bool Class::get(field_t field) const
    {
        return env()->GetStaticBooleanField(getHandle(), field) != 0;
    }

    template <> byte_t Class::get(field_t field) const
    {
        return env()->GetStaticByteField(getHandle(), field);
    }

    template <> wchar_t Class::get(field_t field) const
    {
        return env()->GetStaticCharField(getHandle(), field);
    }

    template <> short Class::get(field_t field) const
    {
        return env()->GetStaticShortField(getHandle(), field);
    }

    template <> int Class::get(field_t field) const
    {
        return env()->GetStaticIntField(getHandle(), field);
    }

    template <> long long Class::get(field_t field) const
    {
        return env()->GetStaticLongField(getHandle(), field);
    }

    template <> float Class::get(field_t field) const
    {
        return env()->GetStaticFloatField(getHandle(), field);
    }

    template <> double Class::get(field_t field) const
    {
        return env()->GetStaticDoubleField(getHandle(), field);
    }

    template <> std::string Class::get(field_t field) const
    {
        return toString(env()->GetStaticObjectField(getHandle(), field));
    }

    template <> std::wstring Class::get(field_t field) const
    {
        return toWString(env()->GetStaticObjectField(getHandle(), field));
    }

    template <> Object Class::get(field_t field) const
    {
        return Object(env()->GetStaticObjectField(getHandle(), field), DeleteLocalInput);
    }

    template <> void Class::set(field_t field, const bool& value)
    {
        env()->SetStaticBooleanField(getHandle(), field, value);
    }

    template <> void Class::set(field_t field, const byte_t& value)
    {
        env()->SetStaticByteField(getHandle(), field, value);
    }

    template <> void Class::set(field_t field, const wchar_t& value)
    {
        env()->SetStaticCharField(getHandle(), field, value);
    }

    template <> void Class::set(field_t field, const short& value)
    {
        env()->SetStaticShortField(getHandle(), field, value);
    }

    template <> void Class::set(field_t field, const int& value)
    {
        env()->SetStaticIntField(getHandle(), field, value);
    }

    template <> void Class::set(field_t field, const long long& value)
    {
        env()->SetStaticLongField(getHandle(), field, value);
    }

    template <> void Class::set(field_t field, const float& value)
    {
        env()->SetStaticFloatField(getHandle(), field, value);
    }

    template <> void Class::set(field_t field, const double& value)
    {
        env()->SetStaticDoubleField(getHandle(), field, value);
    }

    template <> void Class::set(field_t field, const Object& value)
    {
        env()->SetStaticObjectField(getHandle(), field, value.getHandle());
    }

    template <> void Class::set(field_t field, const Object* const& value)
    {
        env()->SetStaticObjectField(getHandle(), field, value ? value->getHandle() : nullptr);
    }

    template <> void Class::set(field_t field, const std::string& value)
    {
        JNIEnv* env = jni::env();

        jobject handle = env->NewStringUTF(value.c_str());
        env->SetStaticObjectField(getHandle(), field, handle);
        env->DeleteLocalRef(handle);
    }

    template <> void Class::set(field_t field, const std::wstring& value)
    {
        JNIEnv* env = jni::env();

#ifdef _WIN32
        jobject handle = env->NewString((const jchar*) value.c_str(), jsize(value.length()));
#else
        auto jstr = toJString(value.c_str(), value.length());
        jobject handle = env->NewString(jstr.c_str(), jsize(jstr.length()));
#endif
        env->SetStaticObjectField(getHandle(), field, handle);
        env->DeleteLocalRef(handle);
    }

    template <> void Class::callStaticMethod(method_t method, internal::value_t* args) const
    {
        env()->CallStaticVoidMethodA(getHandle(), method, (jvalue*) args);
        handleJavaExceptions();
    }

    template <> bool Class::callStaticMethod(method_t method, internal::value_t* args) const
    {
        auto result = env()->CallStaticBooleanMethodA(getHandle(), method, (jvalue*) args);
        handleJavaExceptions();
        return result != 0;
    }

    template <> byte_t Class::callStaticMethod(method_t method, internal::value_t* args) const
    {
        auto result = env()->CallStaticByteMethodA(getHandle(), method, (jvalue*) args);
        handleJavaExceptions();
        return result;
    }

    template <> wchar_t Class::callStaticMethod(method_t method, internal::value_t* args) const
    {
        auto result = env()->CallStaticCharMethodA(getHandle(), method, (jvalue*) args);
        handleJavaExceptions();
        return result;
    }

    template <> short Class::callStaticMethod(method_t method, internal::value_t* args) const
    {
        auto result = env()->CallStaticShortMethodA(getHandle(), method, (jvalue*) args);
        handleJavaExceptions();
        return result;
    }

    template <> int Class::callStaticMethod(method_t method, internal::value_t* args) const
    {
        auto result = env()->CallStaticIntMethodA(getHandle(), method, (jvalue*) args);
        handleJavaExceptions();
        return result;
    }

    template <> long long Class::callStaticMethod(method_t method, internal::value_t* args) const
    {
        auto result = env()->CallStaticLongMethodA(getHandle(), method, (jvalue*) args);
        handleJavaExceptions();
        return result;
    }

    template <> float Class::callStaticMethod(method_t method, internal::value_t* args) const
    {
        auto result = env()->CallStaticFloatMethodA(getHandle(), method, (jvalue*) args);
        handleJavaExceptions();
        return result;
    }

    template <> double Class::callStaticMethod(method_t method, internal::value_t* args) const
    {
        auto result = env()->CallStaticDoubleMethodA(getHandle(), method, (jvalue*) args);
        handleJavaExceptions();
        return result;
    }

    template <> std::string Class::callStaticMethod(method_t method, internal::value_t* args) const
    {
        auto result = env()->CallStaticObjectMethodA(getHandle(), method, (jvalue*) args);
        handleJavaExceptions();
        return toString(result);
    }

    template <> std::wstring Class::callStaticMethod(method_t method, internal::value_t* args) const
    {
        auto result = env()->CallStaticObjectMethodA(getHandle(), method, (jvalue*) args);
        handleJavaExceptions();
        return toWString(result);
    }

    template <> jni::Object Class::callStaticMethod(method_t method, internal::value_t* args) const
    {
        auto result = env()->CallStaticObjectMethodA(getHandle(), method, (jvalue*) args);
        handleJavaExceptions();
        return Object(result, DeleteLocalInput);
    }

    template <> void Class::callExactMethod(jobject obj, method_t method, internal::value_t* args) const
    {
        env()->CallNonvirtualVoidMethodA(obj, getHandle(), method, (jvalue*) args);
        handleJavaExceptions();
    }

    template <> bool Class::callExactMethod(jobject obj, method_t method, internal::value_t* args) const
    {
        auto result = env()->CallNonvirtualBooleanMethodA(obj, getHandle(), method, (jvalue*) args);
        handleJavaExceptions();
        return result != 0;
    }

    template <> byte_t Class::callExactMethod(jobject obj, method_t method, internal::value_t* args) const
    {
        auto result = env()->CallNonvirtualByteMethodA(obj, getHandle(), method, (jvalue*) args);
        handleJavaExceptions();
        return result;
    }

    template <> wchar_t Class::callExactMethod(jobject obj, method_t method, internal::value_t* args) const
    {
        auto result = env()->CallNonvirtualCharMethodA(obj, getHandle(), method, (jvalue*) args);
        handleJavaExceptions();
        return result;
    }

    template <> short Class::callExactMethod(jobject obj, method_t method, internal::value_t* args) const
    {
        auto result = env()->CallNonvirtualShortMethodA(obj, getHandle(), method, (jvalue*) args);
        handleJavaExceptions();
        return result;
    }

    template <> int Class::callExactMethod(jobject obj, method_t method, internal::value_t* args) const
    {
        auto result = env()->CallNonvirtualIntMethodA(obj, getHandle(), method, (jvalue*) args);
        handleJavaExceptions();
        return result;
    }

    template <> long long Class::callExactMethod(jobject obj, method_t method, internal::value_t* args) const
    {
        auto result = env()->CallNonvirtualLongMethodA(obj, getHandle(), method, (jvalue*) args);
        handleJavaExceptions();
        return result;
    }

    template <>  float Class::callExactMethod(jobject obj, method_t method, internal::value_t* args) const
    {
        auto result = env()->CallNonvirtualFloatMethodA(obj, getHandle(), method, (jvalue*) args);
        handleJavaExceptions();
        return result;
    }

    template <> double Class::callExactMethod(jobject obj, method_t method, internal::value_t* args) const
    {
        auto result = env()->CallNonvirtualDoubleMethodA(obj, getHandle(), method, (jvalue*) args);
        handleJavaExceptions();
        return result;
    }

    template <> std::string Class::callExactMethod(jobject obj, method_t method, internal::value_t* args) const
    {
        auto result = env()->CallNonvirtualObjectMethodA(obj, getHandle(), method, (jvalue*)args);
        handleJavaExceptions();
        return toString(result);
    }

    template <> std::wstring Class::callExactMethod(jobject obj, method_t method, internal::value_t* args) const
    {
        auto result = env()->CallNonvirtualObjectMethodA(obj, getHandle(), method, (jvalue*)args);
        handleJavaExceptions();
        return toWString(result);
    }

    template <> Object Class::callExactMethod(jobject obj, method_t method, internal::value_t* args) const
    {
        auto result = env()->CallNonvirtualObjectMethodA(obj, getHandle(), method, (jvalue*)args);
        handleJavaExceptions();
        return Object(result, DeleteLocalInput);
    }

    Object Class::newObject(method_t constructor, internal::value_t* args) const
    {
        jobject ref = env()->NewObjectA(getHandle(), constructor, (jvalue*)args);
        handleJavaExceptions();
        return Object(ref, DeleteLocalInput);
    }

    /*
        Enum Implementation
     */

    Enum::Enum(const char* name) : Class(name)
    {
        _name  = "L";
        _name += name;
        _name += ";";
    }

    Object Enum::get(const char* name) const
    {
        return Class::get<Object>(getStaticField(name, _name.c_str()));
    }

    /*
        Array Implementation
     */

    template <> Array<bool>::Array(long length) : Object(env()->NewBooleanArray(length)), _length(length)
    {
    }

    template <> Array<byte_t>::Array(long length) : Object(env()->NewByteArray(length)), _length(length)
    {
    }

    template <> Array<wchar_t>::Array(long length) : Object(env()->NewCharArray(length)), _length(length)
    {
    }

    template <> Array<short>::Array(long length) : Object(env()->NewShortArray(length)), _length(length)
    {
    }

    template <> Array<int>::Array(long length) : Object(env()->NewIntArray(length)), _length(length)
    {
    }

    template <> Array<long long>::Array(long length) : Object(env()->NewLongArray(length)), _length(length)
    {
    }

    template <> Array<float>::Array(long length) : Object(env()->NewFloatArray(length)), _length(length)
    {
    }

    template <> Array<double>::Array(long length) : Object(env()->NewDoubleArray(length)), _length(length)
    {
    }

    template <> Array<std::string>::Array(long length) : Object(env()->NewObjectArray(length, Class("java/lang/String").getHandle(), nullptr)), _length(length)
    {
    }

    template <> Array<std::wstring>::Array(long length) : Object(env()->NewObjectArray(length, Class("java/lang/String").getHandle(), nullptr)), _length(length)
    {
    }

    template <> Array<Object>::Array(long length) : Object(env()->NewObjectArray(length, Class("java/lang/Object").getHandle(), nullptr)), _length(length)
    {
    }

    template <> Array<Object>::Array(long length, const Class& type) : Object(env()->NewObjectArray(length, type.getHandle(), nullptr)), _length(length)
    {
    }

    template <> bool Array<bool>::getElement(long index) const
    {
        jboolean output;
        env()->GetBooleanArrayRegion(jbooleanArray(getHandle()), index, 1, &output);
        handleJavaExceptions();
        return output;
    }

    template <> byte_t Array<byte_t>::getElement(long index) const
    {
        jbyte output;
        env()->GetByteArrayRegion(jbyteArray(getHandle()), index, 1, &output);
        handleJavaExceptions();
        return output;
    }

    template <> wchar_t Array<wchar_t>::getElement(long index) const
    {
        jchar output;
        env()->GetCharArrayRegion(jcharArray(getHandle()), index, 1, &output);
        handleJavaExceptions();
        return output;
    }

    template <> short Array<short>::getElement(long index) const
    {
        jshort output;
        env()->GetShortArrayRegion(jshortArray(getHandle()), index, 1, &output);
        handleJavaExceptions();
        return output;
    }

    template <> int Array<int>::getElement(long index) const
    {
        jint output;
        env()->GetIntArrayRegion(jintArray(getHandle()), index, 1, &output);
        handleJavaExceptions();
        return output;
    }

    template <> long long Array<long long>::getElement(long index) const
    {
        jlong output;
        env()->GetLongArrayRegion(jlongArray(getHandle()), index, 1, &output);
        handleJavaExceptions();
        return output;
    }

    template <> float Array<float>::getElement(long index) const
    {
        jfloat output;
        env()->GetFloatArrayRegion(jfloatArray(getHandle()), index, 1, &output);
        handleJavaExceptions();
        return output;
    }

    template <> double Array<double>::getElement(long index) const
    {
        jdouble output;
        env()->GetDoubleArrayRegion(jdoubleArray(getHandle()), index, 1, &output);
        handleJavaExceptions();
        return output;
    }

    template <> std::string Array<std::string>::getElement(long index) const
    {
        jobject output = env()->GetObjectArrayElement(jobjectArray(getHandle()), index);
        handleJavaExceptions();
        return toString(output);
    }

    template <> std::wstring Array<std::wstring>::getElement(long index) const
    {
        jobject output = env()->GetObjectArrayElement(jobjectArray(getHandle()), index);
        handleJavaExceptions();
        return toWString(output);
    }

    template <> Object Array<Object>::getElement(long index) const
    {
        jobject output = env()->GetObjectArrayElement(jobjectArray(getHandle()), index);
        handleJavaExceptions();
        return Object(output, DeleteLocalInput);
    }

    template <> void Array<bool>::setElement(long index, bool value)
    {
        jboolean jvalue = value;
        env()->SetBooleanArrayRegion(jbooleanArray(getHandle()), index, 1, &jvalue);
        handleJavaExceptions();
    }

    template <> void Array<byte_t>::setElement(long index, byte_t value)
    {
        jbyte jvalue = value;
        env()->SetByteArrayRegion(jbyteArray(getHandle()), index, 1, &jvalue);
        handleJavaExceptions();
    }

    template <> void Array<wchar_t>::setElement(long index, wchar_t value)
    {
        jchar jvalue = value;
        env()->SetCharArrayRegion(jcharArray(getHandle()), index, 1, &jvalue);
        handleJavaExceptions();
    }

    template <> void Array<short>::setElement(long index, short value)
    {
        jshort jvalue = value;
        env()->SetShortArrayRegion(jshortArray(getHandle()), index, 1, &jvalue);
        handleJavaExceptions();
    }

    template <> void Array<int>::setElement(long index, int value)
    {
        jint jvalue = value;
        env()->SetIntArrayRegion(jintArray(getHandle()), index, 1, &jvalue);
        handleJavaExceptions();
    }

    template <> void Array<long long>::setElement(long index, long long value)
    {
        jlong jvalue = value;
        env()->SetLongArrayRegion(jlongArray(getHandle()), index, 1, &jvalue);
        handleJavaExceptions();
    }

    template <> void Array<float>::setElement(long index, float value)
    {
        jfloat jvalue = value;
        env()->SetFloatArrayRegion(jfloatArray(getHandle()), index, 1, &jvalue);
        handleJavaExceptions();
    }

    template <> void Array<double>::setElement(long index, double value)
    {
        jdouble jvalue = value;
        env()->SetDoubleArrayRegion(jdoubleArray(getHandle()), index, 1, &jvalue);
        handleJavaExceptions();
    }

    template <> void Array<std::string>::setElement(long index, std::string value)
    {
        JNIEnv* env = jni::env();

        jobject jvalue = env->NewStringUTF(value.c_str());;
        env->SetObjectArrayElement(jobjectArray(getHandle()), index, jvalue);
        env->DeleteLocalRef(jvalue);
        handleJavaExceptions();
    }

    template <> void Array<std::wstring>::setElement(long index, std::wstring value)
    {
        JNIEnv* env = jni::env();

#ifdef _WIN32
        jobject jvalue = env->NewString((const jchar*) value.c_str(), jsize(value.length()));
#else
        auto jstr = toJString(value.c_str(), value.length());
        jobject jvalue = env->NewString(jstr.c_str(), jsize(jstr.length()));
#endif
        env->SetObjectArrayElement(jobjectArray(getHandle()), index, jvalue);
        env->DeleteLocalRef(jvalue);
        handleJavaExceptions();
    }

    template <> void Array<Object>::setElement(long index, Object value)
    {
        env()->SetObjectArrayElement(jobjectArray(getHandle()), index, value.getHandle());
        handleJavaExceptions();
    }

    /*
        Vm Implementation
     */

    typedef jint (JNICALL *CreateVm_t)(JavaVM**, void**, void*);

#ifndef _WIN32
    static bool fileExists(const std::string& filePath)
    {
        return access(filePath.c_str(), F_OK) != -1;
    }

    template <size_t N>
    static ssize_t readlink_safe(const char *pathname, char (&output)[N]) {
        auto len = readlink(pathname, output, N - 1);
        if (len > 0) {
            output[len] = '\0';
        }
        return len;
    }

    static std::pair<ssize_t, std::string>
    readlink_as_string(const char *pathname) {
        char buf[2048] = {};
        auto len = readlink_safe(pathname, buf);
        if (len <= 0) {
            return {len, {}};
        }
        return {len, std::string{buf, static_cast<size_t>(len)}};
    }
    static std::string readlink_deep(const char *pathname) {
        std::string prev{pathname};
        ssize_t len = 0;
        std::string next;
        while (true) {
            std::tie(len, next) = readlink_as_string(prev.c_str());
            if (!next.empty()) {
                prev = next;
            } else {
                return prev;
            }
        }
    }

    static std::string drop_path_components(const std::string & path, size_t num_components) {
        size_t pos = std::string::npos;
        size_t slash_pos = std::string::npos;
        for (size_t i = 0; i < num_components; ++i) {
            slash_pos = path.find_last_of('/', pos);
            if (slash_pos == std::string::npos || slash_pos == 0) {
                return {};
            }
            pos = slash_pos - 1;
        }
        return std::string{path.c_str(), slash_pos};
    }
#endif
    static std::string detectJvmPath()
    {
        std::string result;

#ifdef _WIN32

        BYTE buffer[1024];
        DWORD size = sizeof(buffer);
        HKEY versionKey;

        // Search via registry entries.
        if (::RegOpenKeyA(HKEY_LOCAL_MACHINE, "Software\\JavaSoft\\Java Runtime Environment\\", &versionKey) == ERROR_SUCCESS)
        {
            if (::RegQueryValueEx(versionKey, "CurrentVersion", NULL, NULL, buffer, &size) == ERROR_SUCCESS)
            {
                HKEY libKey;

                std::string keyName = std::string("Software\\JavaSoft\\Java Runtime Environment\\") + (const char*)buffer;

                ::RegCloseKey(versionKey);

                if (::RegOpenKeyA(HKEY_LOCAL_MACHINE, keyName.c_str(), &libKey) == ERROR_SUCCESS)
                {
                    size = sizeof(buffer);

                    if (::RegQueryValueEx(libKey, "RuntimeLib", NULL, NULL, buffer, &size) == ERROR_SUCCESS)
                    {
                        result = (const char*)buffer;
                    }

                    ::RegCloseKey(libKey);
                }
            }
        }

        if (result.length() == 0)
        {
            // Could not locate via registry. Try the JAVA_HOME environment variable.
            if ((size = ::GetEnvironmentVariableA("JAVA_HOME", (LPSTR) buffer, sizeof(buffer))) != 0)
            {
                std::string javaHome((const char*) buffer, size);

                // Different installers put in different relative locations.
                std::string options[] = {
                    javaHome + "\\jre\\bin\\client\\jvm.dll",
                    javaHome + "\\jre\\bin\\server\\jvm.dll",
                    javaHome + "\\bin\\client\\jvm.dll",
                    javaHome + "\\bin\\server\\jvm.dll"
                };

                for (auto const& i : options)
                    if (fileExists(i))
                        return i;
            }
        }

#else

        const char* javaHome = getenv("JAVA_HOME");
        if (javaHome != nullptr) {
            #ifdef __APPLE__
            std::string libJvmPath = std::string(javaHome) + "/jre/lib/server/libjvm.dylib";
            #else
            std::string libJvmPath = std::string(javaHome) + "/jre/lib/amd64/server/libjvm.so";
            #endif
            result = libJvmPath;
        } else {
            std::string path = readlink_deep("/usr/bin/java");
            if (!path.empty()) {
                // drop bin and java
                auto javaHome = drop_path_components(path, 2);
                if (!javaHome.empty()) {
                    std::string options[] = {
                        javaHome + "/jre/lib/amd64/server/libjvm.so",
                        javaHome + "/jre/lib/amd64/client/libjvm.so",
                        javaHome + "/jre/lib/server/libjvm.so",
                        javaHome + "/jre/lib/client/libjvm.so",
                        javaHome + "/lib/server/libjvm.so",
                        javaHome + "/lib/client/libjvm.so",
                    };

                    for (auto const &i : options) {
                        fprintf(stderr, "trying %s\n", i.c_str());
                        if (fileExists(i)) {
                            return i;
                        }
                    }
                }
            }
            // Best guess so far.
            result = "/usr/lib/jvm/default-java/jre/lib/amd64/server/libjvm.so";
        }

#endif // _WIN32

        return result;
    }

    Vm::Vm(const char* path_)
    {
        bool expected = false;

        std::string path = path_ ? path_ : detectJvmPath();

        if (path.length() == 0)
            throw InitializationException("Could not locate Java Virtual Machine");
        if (!isVm.compare_exchange_strong(expected, true))
            throw InitializationException("Java Virtual Machine already initialized");

        if (javaVm == nullptr)
        {
            JNIEnv* env;
            JavaVMInitArgs args = {};
            args.version = JNI_VERSION_1_2;

#ifdef _WIN32

            HMODULE lib = ::LoadLibraryA(path.c_str());

            if (lib == NULL)
            {
                isVm.store(false);
                throw InitializationException("Could not load JVM library");
            }

            CreateVm_t JNI_CreateJavaVM = (CreateVm_t) ::GetProcAddress(lib, "JNI_CreateJavaVM");

            /**
                Is your debugger catching an error here?  This is normal.  Just continue. The JVM
                intentionally does this to test how the OS handles memory-reference exceptions.
             */
            if (JNI_CreateJavaVM == NULL || JNI_CreateJavaVM(&javaVm, (void**) &env, &args) != 0)
            {
                isVm.store(false);
                ::FreeLibrary(lib);
                throw InitializationException("Java Virtual Machine failed during creation");
            }

#else

            void* lib = ::dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);

            if (lib == NULL)
            {
                isVm.store(false);
                throw InitializationException("Could not load JVM library");
            }

            CreateVm_t JNI_CreateJavaVM = (CreateVm_t) ::dlsym(lib, "JNI_CreateJavaVM");

            if (JNI_CreateJavaVM == NULL || JNI_CreateJavaVM(&javaVm, (void**) &env, &args) != 0)
            {
                isVm.store(false);
                ::dlclose(lib);
                throw InitializationException("Java Virtual Machine failed during creation");
            }

#endif // _WIN32
        }
    }

    Vm::~Vm()
    {
        /*
            Note that you can't ever *really* unload the JavaVM. If you call
            DestroyJavaVM(), you can't then call JNI_CreateJavaVM() again.
            So, instead we just flag it as "gone".
         */
        isVm.store(false);
    }

    // Forward Declarations
    JNIEnv* env();

#ifndef _WIN32
    extern std::basic_string<jchar> toJString(const wchar_t* str, size_t length);
#endif

    namespace internal
    {
        // Base Type Conversions
        void valueArg(value_t* v, bool a)                   { ((jvalue*) v)->z = jboolean(a); }
        void valueArg(value_t* v, byte_t a)                 { ((jvalue*) v)->b = a; }
        void valueArg(value_t* v, wchar_t a)                { ((jvalue*) v)->c = jchar(a); }    // Note: Possible truncation.
        void valueArg(value_t* v, short a)                  { ((jvalue*) v)->s = a; }
        void valueArg(value_t* v, int a)                    { ((jvalue*) v)->i = a; }
        void valueArg(value_t* v, long long a)              { ((jvalue*) v)->j = a; }
        void valueArg(value_t* v, float a)                  { ((jvalue*) v)->f = a; }
        void valueArg(value_t* v, double a)                 { ((jvalue*) v)->d = a; }
        void valueArg(value_t* v, jobject a)                { ((jvalue*) v)->l = a; }
        void valueArg(value_t* v, const Object& a)          { ((jvalue*) v)->l = a.getHandle(); }
        void valueArg(value_t* v, const Object* const& a)   { ((jvalue*) v)->l = a ? a->getHandle() : nullptr; }

        /*
            Object Implementations
         */

        std::string valueSig(const Object* obj)
        {
            if (obj == nullptr || obj->isNull())
                return "Ljava/lang/Object;";    // One can always hope...

            std::string name = Class(obj->getClass(), Object::Temporary).getName();

            // Change from "java.lang.Object" format to "java/lang/Object";
            for (size_t i = 0; i < name.length(); ++i)
                if (name[i] == '.')
                    name[i] = '/';

            return "L" + name + ";";
        }

        /*
            String Implementations
         */

        void valueArg(value_t* v, const std::string& a)
        {
            ((jvalue*) v)->l = env()->NewStringUTF(a.c_str());
        }

        template <> void cleanupArg<std::string>(value_t* v)
        {
            env()->DeleteLocalRef(((jvalue*) v)->l);
        }

        void valueArg(value_t* v, const char* a)
        {
            ((jvalue*) v)->l = env()->NewStringUTF(a);
        }

        void valueArg(value_t* v, std::nullptr_t)
        {
            ((jvalue*) v)->l = nullptr;
        }

        template <> void cleanupArg<const char*>(value_t* v)
        {
            env()->DeleteLocalRef(((jvalue*) v)->l);
        }
#ifdef _WIN32

        void valueArg(value_t* v, const std::wstring& a)
        {
            ((jvalue*) v)->l = env()->NewString((const jchar*) a.c_str(), jsize(a.length()));
        }

        void valueArg(value_t* v, const wchar_t* a)
        {
            ((jvalue*) v)->l = env()->NewString((const jchar*) a, jsize(std::wcslen(a)));
        }
#else

        void valueArg(value_t* v, const std::wstring& a)
        {
            auto jstr = toJString(a.c_str(), a.length());
            ((jvalue*) v)->l = env()->NewString(jstr.c_str(), jsize(jstr.length()));
        }

        void valueArg(value_t* v, const wchar_t* a)
        {
            auto jstr = toJString(a, std::wcslen(a));
            ((jvalue*) v)->l = env()->NewString(jstr.c_str(), jsize(jstr.length()));
        }

#endif

        template <> void cleanupArg<const std::wstring*>(value_t* v)
        {
            env()->DeleteLocalRef(((jvalue*) v)->l);
        }

        template <> void cleanupArg<const wchar_t*>(value_t* v)
        {
            env()->DeleteLocalRef(((jvalue*) v)->l);
        }

        long getArrayLength(jarray array)
        {
            return env()->GetArrayLength(array);
        }
    }
}

