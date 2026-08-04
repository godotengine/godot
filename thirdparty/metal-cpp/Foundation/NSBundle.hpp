//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Foundation/NSBundle.hpp
//
// Copyright 2020-2024 Apple Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#pragma once

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#include "NSDefines.hpp"
#include "NSNotification.hpp"
#include "NSObject.hpp"
#include "NSTypes.hpp"

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace NS
{
_NS_CONST(NotificationName, BundleDidLoadNotification);
_NS_CONST(NotificationName, BundleResourceRequestLowDiskSpaceNotification);

class String* LocalizedString(const String* pKey, const String*);
class String* LocalizedStringFromTable(const String* pKey, const String* pTbl, const String*);
class String* LocalizedStringFromTableInBundle(const String* pKey, const String* pTbl, const class Bundle* pBdle, const String*);
class String* LocalizedStringWithDefaultValue(const String* pKey, const String* pTbl, const class Bundle* pBdle, const String* pVal, const String*);

class Bundle : public Referencing<Bundle>
{
public:
    static Bundle*      mainBundle();

    static Bundle*      bundle(const class String* pPath);
    static Bundle*      bundle(const class URL* pURL);

    static class Array* allBundles();
    static class Array* allFrameworks();

    static Bundle*      alloc();

    Bundle*             init(const class String* pPath);
    Bundle*             init(const class URL* pURL);

    bool                load();
    bool                unload();

    bool                isLoaded() const;

    bool                preflightAndReturnError(class Error** pError) const;
    bool                loadAndReturnError(class Error** pError);

    class URL*          bundleURL() const;
    class URL*          resourceURL() const;
    class URL*          executableURL() const;
    class URL*          URLForAuxiliaryExecutable(const class String* pExecutableName) const;

    class URL*          privateFrameworksURL() const;
    class URL*          sharedFrameworksURL() const;
    class URL*          sharedSupportURL() const;
    class URL*          builtInPlugInsURL() const;
    class URL*          appStoreReceiptURL() const;

    class String*       bundlePath() const;
    class String*       resourcePath() const;
    class String*       executablePath() const;
    class String*       pathForAuxiliaryExecutable(const class String* pExecutableName) const;

    class String*       privateFrameworksPath() const;
    class String*       sharedFrameworksPath() const;
    class String*       sharedSupportPath() const;
    class String*       builtInPlugInsPath() const;

    class String*       bundleIdentifier() const;
    class Dictionary*   infoDictionary() const;
    class Dictionary*   localizedInfoDictionary() const;
    class Object*       objectForInfoDictionaryKey(const class String* pKey);

    class String*       localizedString(const class String* pKey, const class String* pValue = nullptr, const class String* pTableName = nullptr) const;
};
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_PRIVATE_DEF_CONST(NS::NotificationName, BundleDidLoadNotification);
_NS_PRIVATE_DEF_CONST(NS::NotificationName, BundleResourceRequestLowDiskSpaceNotification);

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::String* NS::LocalizedString(const String* pKey, const String*)
{
    return Bundle::mainBundle()->localizedString(pKey, nullptr, nullptr);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::String* NS::LocalizedStringFromTable(const String* pKey, const String* pTbl, const String*)
{
    return Bundle::mainBundle()->localizedString(pKey, nullptr, pTbl);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::String* NS::LocalizedStringFromTableInBundle(const String* pKey, const String* pTbl, const Bundle* pBdl, const String*)
{
    return pBdl->localizedString(pKey, nullptr, pTbl);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::String* NS::LocalizedStringWithDefaultValue(const String* pKey, const String* pTbl, const Bundle* pBdl, const String* pVal, const String*)
{
    return pBdl->localizedString(pKey, pVal, pTbl);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Bundle* NS::Bundle::mainBundle()
{
    return Object::sendMessage<Bundle*>(_NS_PRIVATE_CLS(NSBundle), _NS_PRIVATE_SEL(mainBundle));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Bundle* NS::Bundle::bundle(const class String* pPath)
{
    return Object::sendMessage<Bundle*>(_NS_PRIVATE_CLS(NSBundle), _NS_PRIVATE_SEL(bundleWithPath_), pPath);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Bundle* NS::Bundle::bundle(const class URL* pURL)
{
    return Object::sendMessage<Bundle*>(_NS_PRIVATE_CLS(NSBundle), _NS_PRIVATE_SEL(bundleWithURL_), pURL);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Array* NS::Bundle::allBundles()
{
    return Object::sendMessage<Array*>(_NS_PRIVATE_CLS(NSBundle), _NS_PRIVATE_SEL(allBundles));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Array* NS::Bundle::allFrameworks()
{
    return Object::sendMessage<Array*>(_NS_PRIVATE_CLS(NSBundle), _NS_PRIVATE_SEL(allFrameworks));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Bundle* NS::Bundle::alloc()
{
    return Object::sendMessage<Bundle*>(_NS_PRIVATE_CLS(NSBundle), _NS_PRIVATE_SEL(alloc));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Bundle* NS::Bundle::init(const String* pPath)
{
    return Object::sendMessage<Bundle*>(this, _NS_PRIVATE_SEL(initWithPath_), pPath);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Bundle* NS::Bundle::init(const URL* pURL)
{
    return Object::sendMessage<Bundle*>(this, _NS_PRIVATE_SEL(initWithURL_), pURL);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE bool NS::Bundle::load()
{
    return Object::sendMessage<bool>(this, _NS_PRIVATE_SEL(load));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE bool NS::Bundle::unload()
{
    return Object::sendMessage<bool>(this, _NS_PRIVATE_SEL(unload));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE bool NS::Bundle::isLoaded() const
{
    return Object::sendMessage<bool>(this, _NS_PRIVATE_SEL(isLoaded));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE bool NS::Bundle::preflightAndReturnError(Error** pError) const
{
    return Object::sendMessage<bool>(this, _NS_PRIVATE_SEL(preflightAndReturnError_), pError);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE bool NS::Bundle::loadAndReturnError(Error** pError)
{
    return Object::sendMessage<bool>(this, _NS_PRIVATE_SEL(loadAndReturnError_), pError);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::URL* NS::Bundle::bundleURL() const
{
    return Object::sendMessage<URL*>(this, _NS_PRIVATE_SEL(bundleURL));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::URL* NS::Bundle::resourceURL() const
{
    return Object::sendMessage<URL*>(this, _NS_PRIVATE_SEL(resourceURL));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::URL* NS::Bundle::executableURL() const
{
    return Object::sendMessage<URL*>(this, _NS_PRIVATE_SEL(executableURL));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::URL* NS::Bundle::URLForAuxiliaryExecutable(const String* pExecutableName) const
{
    return Object::sendMessage<URL*>(this, _NS_PRIVATE_SEL(URLForAuxiliaryExecutable_), pExecutableName);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::URL* NS::Bundle::privateFrameworksURL() const
{
    return Object::sendMessage<URL*>(this, _NS_PRIVATE_SEL(privateFrameworksURL));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::URL* NS::Bundle::sharedFrameworksURL() const
{
    return Object::sendMessage<URL*>(this, _NS_PRIVATE_SEL(sharedFrameworksURL));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::URL* NS::Bundle::sharedSupportURL() const
{
    return Object::sendMessage<URL*>(this, _NS_PRIVATE_SEL(sharedSupportURL));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::URL* NS::Bundle::builtInPlugInsURL() const
{
    return Object::sendMessage<URL*>(this, _NS_PRIVATE_SEL(builtInPlugInsURL));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::URL* NS::Bundle::appStoreReceiptURL() const
{
    return Object::sendMessage<URL*>(this, _NS_PRIVATE_SEL(appStoreReceiptURL));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::String* NS::Bundle::bundlePath() const
{
    return Object::sendMessage<String*>(this, _NS_PRIVATE_SEL(bundlePath));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::String* NS::Bundle::resourcePath() const
{
    return Object::sendMessage<String*>(this, _NS_PRIVATE_SEL(resourcePath));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::String* NS::Bundle::executablePath() const
{
    return Object::sendMessage<String*>(this, _NS_PRIVATE_SEL(executablePath));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::String* NS::Bundle::pathForAuxiliaryExecutable(const String* pExecutableName) const
{
    return Object::sendMessage<String*>(this, _NS_PRIVATE_SEL(pathForAuxiliaryExecutable_), pExecutableName);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::String* NS::Bundle::privateFrameworksPath() const
{
    return Object::sendMessage<String*>(this, _NS_PRIVATE_SEL(privateFrameworksPath));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::String* NS::Bundle::sharedFrameworksPath() const
{
    return Object::sendMessage<String*>(this, _NS_PRIVATE_SEL(sharedFrameworksPath));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::String* NS::Bundle::sharedSupportPath() const
{
    return Object::sendMessage<String*>(this, _NS_PRIVATE_SEL(sharedSupportPath));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::String* NS::Bundle::builtInPlugInsPath() const
{
    return Object::sendMessage<String*>(this, _NS_PRIVATE_SEL(builtInPlugInsPath));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::String* NS::Bundle::bundleIdentifier() const
{
    return Object::sendMessage<String*>(this, _NS_PRIVATE_SEL(bundleIdentifier));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Dictionary* NS::Bundle::infoDictionary() const
{
    return Object::sendMessage<Dictionary*>(this, _NS_PRIVATE_SEL(infoDictionary));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Dictionary* NS::Bundle::localizedInfoDictionary() const
{
    return Object::sendMessage<Dictionary*>(this, _NS_PRIVATE_SEL(localizedInfoDictionary));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::Object* NS::Bundle::objectForInfoDictionaryKey(const String* pKey)
{
    return Object::sendMessage<Object*>(this, _NS_PRIVATE_SEL(objectForInfoDictionaryKey_), pKey);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

_NS_INLINE NS::String* NS::Bundle::localizedString(const String* pKey, const String* pValue /* = nullptr */, const String* pTableName /* = nullptr */) const
{
    return Object::sendMessage<String*>(this, _NS_PRIVATE_SEL(localizedStringForKey_value_table_), pKey, pValue, pTableName);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------
