// Copyright 2020-2021, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
// Author: Ryan Pavlik <ryan.pavlik@collabora.com>

#pragma once

#include "ObjectWrapperBase.h"

namespace wrap {
namespace android::net {
class Uri;
class Uri_Builder;
} // namespace android::net

} // namespace wrap

namespace wrap {
namespace android::net {
/*!
 * Wrapper for android.net.Uri objects.
 */
class Uri : public ObjectWrapperBase {
  public:
    using ObjectWrapperBase::ObjectWrapperBase;
    static constexpr const char *getTypeName() noexcept {
        return "android/net/Uri";
    }

    /*!
     * Wrapper for the toString method
     *
     * Java prototype:
     * `public abstract java.lang.String toString();`
     *
     * JNI signature: ()Ljava/lang/String;
     *
     */
    std::string toString() const;

    /*!
     * Class metadata
     */
    struct Meta : public MetaBaseDroppable {
        jni::method_t toString;

        /*!
         * Singleton accessor
         */
        static Meta &data() {
            static Meta instance{};
            return instance;
        }

      private:
        Meta();
    };
};

/*!
 * Wrapper for android.net.Uri$Builder objects.
 */
class Uri_Builder : public ObjectWrapperBase {
  public:
    using ObjectWrapperBase::ObjectWrapperBase;
    static constexpr const char *getTypeName() noexcept {
        return "android/net/Uri$Builder";
    }

    /*!
     * Wrapper for a constructor
     *
     * Java prototype:
     * `public android.net.Uri$Builder();`
     *
     * JNI signature: ()V
     *
     */
    static Uri_Builder construct();

    /*!
     * Wrapper for the scheme method
     *
     * Java prototype:
     * `public android.net.Uri$Builder scheme(java.lang.String);`
     *
     * JNI signature: (Ljava/lang/String;)Landroid/net/Uri$Builder;
     *
     */
    Uri_Builder &scheme(std::string const &stringParam);

    /*!
     * Wrapper for the authority method
     *
     * Java prototype:
     * `public android.net.Uri$Builder authority(java.lang.String);`
     *
     * JNI signature: (Ljava/lang/String;)Landroid/net/Uri$Builder;
     *
     */
    Uri_Builder &authority(std::string const &stringParam);

    /*!
     * Wrapper for the appendPath method
     *
     * Java prototype:
     * `public android.net.Uri$Builder appendPath(java.lang.String);`
     *
     * JNI signature: (Ljava/lang/String;)Landroid/net/Uri$Builder;
     *
     */
    Uri_Builder &appendPath(std::string const &stringParam);

    /*!
     * Wrapper for the build method
     *
     * Java prototype:
     * `public android.net.Uri build();`
     *
     * JNI signature: ()Landroid/net/Uri;
     *
     */
    Uri build();

    /*!
     * Class metadata
     */
    struct Meta : public MetaBaseDroppable {
        jni::method_t init;
        jni::method_t scheme;
        jni::method_t authority;
        jni::method_t appendPath;
        jni::method_t build;

        /*!
         * Singleton accessor
         */
        static Meta &data() {
            static Meta instance{};
            return instance;
        }

      private:
        Meta();
    };
};

} // namespace android::net
} // namespace wrap
#include "android.net.impl.h"
