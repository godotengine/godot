// Copyright 2020-2021, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
// Author: Ryan Pavlik <ryan.pavlik@collabora.com>

#pragma once

#include "ObjectWrapperBase.h"

namespace wrap {
namespace android::content {
class ComponentName;
class ContentResolver;
class Context;
} // namespace android::content

namespace android::database {
class Cursor;
} // namespace android::database

namespace android::net {
class Uri;
class Uri_Builder;
} // namespace android::net

} // namespace wrap

namespace wrap {
namespace android::content {
/*!
 * Wrapper for android.content.Context objects.
 */
class Context : public ObjectWrapperBase {
  public:
    using ObjectWrapperBase::ObjectWrapperBase;
    static constexpr const char *getTypeName() noexcept {
        return "android/content/Context";
    }

    /*!
     * Wrapper for the getContentResolver method
     *
     * Java prototype:
     * `public abstract android.content.ContentResolver getContentResolver();`
     *
     * JNI signature: ()Landroid/content/ContentResolver;
     *
     */
    ContentResolver getContentResolver() const;

    /*!
     * Class metadata
     */
    struct Meta : public MetaBaseDroppable {
        jni::method_t getContentResolver;

        /*!
         * Singleton accessor
         */
        static Meta &data(bool deferDrop = false) {
            static Meta instance{deferDrop};
            return instance;
        }

      private:
        explicit Meta(bool deferDrop);
    };
};

/*!
 * Wrapper for android.content.ContentUris objects.
 */
class ContentUris : public ObjectWrapperBase {
  public:
    using ObjectWrapperBase::ObjectWrapperBase;
    static constexpr const char *getTypeName() noexcept {
        return "android/content/ContentUris";
    }

    /*!
     * Wrapper for the appendId static method
     *
     * Java prototype:
     * `public static android.net.Uri$Builder appendId(android.net.Uri$Builder,
     * long);`
     *
     * JNI signature: (Landroid/net/Uri$Builder;J)Landroid/net/Uri$Builder;
     *
     */
    static net::Uri_Builder appendId(net::Uri_Builder &uri_Builder,
                                     long long longParam);

    /*!
     * Class metadata
     */
    struct Meta : public MetaBaseDroppable {
        jni::method_t appendId;

        /*!
         * Singleton accessor
         */
        static Meta &data(bool deferDrop = false) {
            static Meta instance{deferDrop};
            return instance;
        }

      private:
        explicit Meta(bool deferDrop);
    };
};

/*!
 * Wrapper for android.content.ComponentName objects.
 */
class ComponentName : public ObjectWrapperBase {
  public:
    using ObjectWrapperBase::ObjectWrapperBase;
    static constexpr const char *getTypeName() noexcept {
        return "android/content/ComponentName";
    }

    /*!
     * Wrapper for a constructor
     *
     * Java prototype:
     * `public android.content.ComponentName(java.lang.String,
     * java.lang.String);`
     *
     * JNI signature: (Ljava/lang/String;Ljava/lang/String;)V
     *
     */
    static ComponentName construct(std::string const &pkg,
                                   std::string const &className);

    /*!
     * Wrapper for a constructor
     *
     * Java prototype:
     * `public android.content.ComponentName(android.content.Context,
     * java.lang.String);`
     *
     * JNI signature: (Landroid/content/Context;Ljava/lang/String;)V
     *
     */
    static ComponentName construct(Context const &context,
                                   std::string const &className);

    /*!
     * Wrapper for a constructor
     *
     * Java prototype:
     * `public android.content.ComponentName(android.content.Context,
     * java.lang.Class<?>);`
     *
     * JNI signature: (Landroid/content/Context;Ljava/lang/Class;)V
     *
     */
    static ComponentName construct(Context const &context,
                                   jni::Object const &cls);

    /*!
     * Wrapper for a constructor
     *
     * Java prototype:
     * `public android.content.ComponentName(android.os.Parcel);`
     *
     * JNI signature: (Landroid/os/Parcel;)V
     *
     */
    static ComponentName construct(jni::Object const &parcel);

    /*!
     * Class metadata
     */
    struct Meta : public MetaBase {
        jni::method_t init;
        jni::method_t init1;
        jni::method_t init2;
        jni::method_t init3;

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
 * Wrapper for android.content.ContentResolver objects.
 */
class ContentResolver : public ObjectWrapperBase {
  public:
    using ObjectWrapperBase::ObjectWrapperBase;
    static constexpr const char *getTypeName() noexcept {
        return "android/content/ContentResolver";
    }

    /*!
     * Wrapper for the query method - overload added in API level 1
     *
     * Java prototype:
     * `public final android.database.Cursor query(android.net.Uri,
     * java.lang.String[], java.lang.String, java.lang.String[],
     * java.lang.String);`
     *
     * JNI signature:
     * (Landroid/net/Uri;[Ljava/lang/String;Ljava/lang/String;[Ljava/lang/String;Ljava/lang/String;)Landroid/database/Cursor;
     *
     */
    database::Cursor query(net::Uri const &uri,
                           jni::Array<std::string> const &projection,
                           std::string const &selection,
                           jni::Array<std::string> const &selectionArgs,
                           std::string const &sortOrder);

    /*!
     * Wrapper for the query method - overload added in API level 1
     *
     * Java prototype:
     * `public final android.database.Cursor query(android.net.Uri,
     * java.lang.String[], java.lang.String, java.lang.String[],
     * java.lang.String);`
     *
     * JNI signature:
     * (Landroid/net/Uri;[Ljava/lang/String;Ljava/lang/String;[Ljava/lang/String;Ljava/lang/String;)Landroid/database/Cursor;
     *
     * This is a way to call the main query() function without the three
     * trailing optional arguments.
     */
    database::Cursor query(net::Uri const &uri,
                           jni::Array<std::string> const &projection);

    /*!
     * Wrapper for the query method - overload added in API level 16
     *
     * Java prototype:
     * `public final android.database.Cursor query(android.net.Uri,
     * java.lang.String[], java.lang.String, java.lang.String[],
     * java.lang.String, android.os.CancellationSignal);`
     *
     * JNI signature:
     * (Landroid/net/Uri;[Ljava/lang/String;Ljava/lang/String;[Ljava/lang/String;Ljava/lang/String;Landroid/os/CancellationSignal;)Landroid/database/Cursor;
     *
     */
    database::Cursor query(net::Uri const &uri,
                           jni::Array<std::string> const &projection,
                           std::string const &selection,
                           jni::Array<std::string> const &selectionArgs,
                           std::string const &sortOrder,
                           jni::Object const &cancellationSignal);

    /*!
     * Wrapper for the query method - overload added in API level 26
     *
     * Java prototype:
     * `public final android.database.Cursor query(android.net.Uri,
     * java.lang.String[], android.os.Bundle, android.os.CancellationSignal);`
     *
     * JNI signature:
     * (Landroid/net/Uri;[Ljava/lang/String;Landroid/os/Bundle;Landroid/os/CancellationSignal;)Landroid/database/Cursor;
     *
     */
    database::Cursor query(net::Uri const &uri,
                           jni::Array<std::string> const &projection,
                           jni::Object const &queryArgs,
                           jni::Object const &cancellationSignal);

    /*!
     * Class metadata
     */
    struct Meta : public MetaBaseDroppable {
        jni::method_t query;
        jni::method_t query1;
        jni::method_t query2;

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

} // namespace android::content
} // namespace wrap
#include "android.content.impl.h"
