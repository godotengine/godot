#pragma once

#include "NSDefines.hpp"
#include "NSBlocks.hpp"
#include "NSStructs.hpp"
#include "NSBridge.hpp"
#include "NSObject.hpp"
#include "NSTypes.hpp"
#include "NSRange.hpp"

namespace NS {
    class Array;
    class Dictionary;
    class Error;
    class Object;
    class String;
    class URL;
}

namespace NS
{

extern NS::NotificationName const BundleDidLoadNotification __asm__("_NSBundleDidLoadNotification");
extern NS::String* const LoadedClasses __asm__("_NSLoadedClasses");
extern NS::NotificationName const BundleResourceRequestLowDiskSpaceNotification __asm__("_NSBundleResourceRequestLowDiskSpaceNotification");
inline constexpr unsigned int BundleExecutableArchitectureI386 = 0x00000007;
inline constexpr unsigned int BundleExecutableArchitecturePPC = 0x00000012;
inline constexpr unsigned int BundleExecutableArchitectureX86_64 = 0x01000007;
inline constexpr unsigned int BundleExecutableArchitecturePPC64 = 0x01000012;
inline constexpr unsigned int BundleExecutableArchitectureARM64 = 0x0100000c;


class Bundle : public NS::Referencing<Bundle>
{
public:
    static Bundle* alloc();
    Bundle*        init() const;

    static NS::Array*  allBundles();
    static NS::Array*  allFrameworks();
    static NS::Bundle* bundle(NS::String* path);
    static NS::Bundle* bundle(NS::URL* url);
    static NS::Bundle* mainBundle();

    NS::URL*        URLForAuxiliaryExecutable(NS::String* executableName);
    NS::URL*        appStoreReceiptURL() const;
    NS::String*     builtInPlugInsPath() const;
    NS::URL*        builtInPlugInsURL() const;
    NS::String*     bundleIdentifier() const;
    NS::String*     bundlePath() const;
    NS::URL*        bundleURL() const;
    NS::String*     executablePath() const;
    NS::URL*        executableURL() const;
    NS::Dictionary* infoDictionary() const;
    NS::Bundle*     init(NS::String* path);
    NS::Bundle*     init(NS::URL* url);
    bool            isLoaded();
    bool            load();
    bool            loadAndReturnError(NS::Error** error);
    NS::Dictionary* localizedInfoDictionary() const;
    NS::String*     localizedString(NS::String* key, NS::String* value, NS::String* tableName);
    NS::Object*     object(NS::String* key);
    NS::String*     path(NS::String* executableName);
    bool            preflightAndReturnError(NS::Error** error);
    NS::String*     privateFrameworksPath() const;
    NS::URL*        privateFrameworksURL() const;
    NS::String*     resourcePath() const;
    NS::URL*        resourceURL() const;
    NS::String*     sharedFrameworksPath() const;
    NS::URL*        sharedFrameworksURL() const;
    NS::String*     sharedSupportPath() const;
    NS::URL*        sharedSupportURL() const;
    bool            unload();

};

} // namespace NS

// --- Class symbols + inline implementations ---

extern "C" void *OBJC_CLASS_$_NSBundle;

_NS_INLINE NS::Bundle* NS::Bundle::alloc()
{
    return _NS_msg_NS__Bundlep_alloc((const void*)&OBJC_CLASS_$_NSBundle, nullptr);
}

_NS_INLINE NS::Bundle* NS::Bundle::init() const
{
    return _NS_msg_NS__Bundlep_init((const void*)this, nullptr);
}

_NS_INLINE NS::Bundle* NS::Bundle::mainBundle()
{
    return _NS_msg_NS__Bundlep_mainBundle((const void*)&OBJC_CLASS_$_NSBundle, nullptr);
}

_NS_INLINE NS::Array* NS::Bundle::allBundles()
{
    return _NS_msg_NS__Arrayp_allBundles((const void*)&OBJC_CLASS_$_NSBundle, nullptr);
}

_NS_INLINE NS::Array* NS::Bundle::allFrameworks()
{
    return _NS_msg_NS__Arrayp_allFrameworks((const void*)&OBJC_CLASS_$_NSBundle, nullptr);
}

_NS_INLINE NS::Bundle* NS::Bundle::bundle(NS::String* path)
{
    return _NS_msg_NS__Bundlep_bundleWithPath__NS__Stringp((const void*)&OBJC_CLASS_$_NSBundle, nullptr, path);
}

_NS_INLINE NS::Bundle* NS::Bundle::bundle(NS::URL* url)
{
    return _NS_msg_NS__Bundlep_bundleWithURL__NS__URLp((const void*)&OBJC_CLASS_$_NSBundle, nullptr, url);
}

_NS_INLINE NS::URL* NS::Bundle::bundleURL() const
{
    return _NS_msg_NS__URLp_bundleURL((const void*)this, nullptr);
}

_NS_INLINE NS::URL* NS::Bundle::resourceURL() const
{
    return _NS_msg_NS__URLp_resourceURL((const void*)this, nullptr);
}

_NS_INLINE NS::URL* NS::Bundle::executableURL() const
{
    return _NS_msg_NS__URLp_executableURL((const void*)this, nullptr);
}

_NS_INLINE NS::URL* NS::Bundle::privateFrameworksURL() const
{
    return _NS_msg_NS__URLp_privateFrameworksURL((const void*)this, nullptr);
}

_NS_INLINE NS::URL* NS::Bundle::sharedFrameworksURL() const
{
    return _NS_msg_NS__URLp_sharedFrameworksURL((const void*)this, nullptr);
}

_NS_INLINE NS::URL* NS::Bundle::sharedSupportURL() const
{
    return _NS_msg_NS__URLp_sharedSupportURL((const void*)this, nullptr);
}

_NS_INLINE NS::URL* NS::Bundle::builtInPlugInsURL() const
{
    return _NS_msg_NS__URLp_builtInPlugInsURL((const void*)this, nullptr);
}

_NS_INLINE NS::URL* NS::Bundle::appStoreReceiptURL() const
{
    return _NS_msg_NS__URLp_appStoreReceiptURL((const void*)this, nullptr);
}

_NS_INLINE NS::String* NS::Bundle::bundlePath() const
{
    return _NS_msg_NS__Stringp_bundlePath((const void*)this, nullptr);
}

_NS_INLINE NS::String* NS::Bundle::resourcePath() const
{
    return _NS_msg_NS__Stringp_resourcePath((const void*)this, nullptr);
}

_NS_INLINE NS::String* NS::Bundle::executablePath() const
{
    return _NS_msg_NS__Stringp_executablePath((const void*)this, nullptr);
}

_NS_INLINE NS::String* NS::Bundle::privateFrameworksPath() const
{
    return _NS_msg_NS__Stringp_privateFrameworksPath((const void*)this, nullptr);
}

_NS_INLINE NS::String* NS::Bundle::sharedFrameworksPath() const
{
    return _NS_msg_NS__Stringp_sharedFrameworksPath((const void*)this, nullptr);
}

_NS_INLINE NS::String* NS::Bundle::sharedSupportPath() const
{
    return _NS_msg_NS__Stringp_sharedSupportPath((const void*)this, nullptr);
}

_NS_INLINE NS::String* NS::Bundle::builtInPlugInsPath() const
{
    return _NS_msg_NS__Stringp_builtInPlugInsPath((const void*)this, nullptr);
}

_NS_INLINE NS::String* NS::Bundle::bundleIdentifier() const
{
    return _NS_msg_NS__Stringp_bundleIdentifier((const void*)this, nullptr);
}

_NS_INLINE NS::Dictionary* NS::Bundle::infoDictionary() const
{
    return _NS_msg_NS__Dictionaryp_infoDictionary((const void*)this, nullptr);
}

_NS_INLINE NS::Dictionary* NS::Bundle::localizedInfoDictionary() const
{
    return _NS_msg_NS__Dictionaryp_localizedInfoDictionary((const void*)this, nullptr);
}

_NS_INLINE NS::Bundle* NS::Bundle::init(NS::String* path)
{
    return _NS_msg_NS__Bundlep_initWithPath__NS__Stringp((const void*)this, nullptr, path);
}

_NS_INLINE NS::Bundle* NS::Bundle::init(NS::URL* url)
{
    return _NS_msg_NS__Bundlep_initWithURL__NS__URLp((const void*)this, nullptr, url);
}

_NS_INLINE bool NS::Bundle::load()
{
    return _NS_msg_bool_load((const void*)this, nullptr);
}

_NS_INLINE bool NS::Bundle::unload()
{
    return _NS_msg_bool_unload((const void*)this, nullptr);
}

_NS_INLINE bool NS::Bundle::preflightAndReturnError(NS::Error** error)
{
    return _NS_msg_bool_preflightAndReturnError__NS__Errorpp((const void*)this, nullptr, error);
}

_NS_INLINE bool NS::Bundle::loadAndReturnError(NS::Error** error)
{
    return _NS_msg_bool_loadAndReturnError__NS__Errorpp((const void*)this, nullptr, error);
}

_NS_INLINE NS::URL* NS::Bundle::URLForAuxiliaryExecutable(NS::String* executableName)
{
    return _NS_msg_NS__URLp_URLForAuxiliaryExecutable__NS__Stringp((const void*)this, nullptr, executableName);
}

_NS_INLINE NS::String* NS::Bundle::path(NS::String* executableName)
{
    return _NS_msg_NS__Stringp_pathForAuxiliaryExecutable__NS__Stringp((const void*)this, nullptr, executableName);
}

_NS_INLINE NS::String* NS::Bundle::localizedString(NS::String* key, NS::String* value, NS::String* tableName)
{
    return _NS_msg_NS__Stringp_localizedStringForKey_value_table__NS__Stringp_NS__Stringp_NS__Stringp((const void*)this, nullptr, key, value, tableName);
}

_NS_INLINE NS::Object* NS::Bundle::object(NS::String* key)
{
    return _NS_msg_NS__Objectp_objectForInfoDictionaryKey__NS__Stringp((const void*)this, nullptr, key);
}

_NS_INLINE bool NS::Bundle::isLoaded()
{
    return _NS_msg_bool_isLoaded((const void*)this, nullptr);
}
