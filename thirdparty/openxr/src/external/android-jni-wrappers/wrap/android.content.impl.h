// Copyright 2020-2021, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
// Author: Ryan Pavlik <ryan.pavlik@collabora.com>
// Inline implementations: do not include on its own!

#pragma once

#include "android.database.h"
#include "android.net.h"
#include <string>

namespace wrap {
namespace android::content {
inline ContentResolver Context::getContentResolver() const {
    assert(!isNull());
    return ContentResolver(
        object().call<jni::Object>(Meta::data().getContentResolver));
}

inline net::Uri_Builder ContentUris::appendId(net::Uri_Builder &uri_Builder,
                                              long long longParam) {
    auto &data = Meta::data(true);
    auto ret = net::Uri_Builder(data.clazz().call<jni::Object>(
        data.appendId, uri_Builder.object(), longParam));
    data.dropClassRef();
    return ret;
}

inline ComponentName ComponentName::construct(std::string const &pkg,
                                              std::string const &className) {
    return ComponentName(
        Meta::data().clazz().newInstance(Meta::data().init, pkg, className));
}

inline ComponentName ComponentName::construct(Context const &context,
                                              std::string const &className) {
    return ComponentName(Meta::data().clazz().newInstance(
        Meta::data().init1, context.object(), className));
}

inline ComponentName ComponentName::construct(Context const &context,
                                              jni::Object const &cls) {
    return ComponentName(Meta::data().clazz().newInstance(
        Meta::data().init2, context.object(), cls));
}

inline ComponentName ComponentName::construct(jni::Object const &parcel) {
    return ComponentName(
        Meta::data().clazz().newInstance(Meta::data().init3, parcel));
}

inline database::Cursor ContentResolver::query(
    net::Uri const &uri, jni::Array<std::string> const &projection,
    std::string const &selection, jni::Array<std::string> const &selectionArgs,
    std::string const &sortOrder) {
    assert(!isNull());
    return database::Cursor(
        object().call<jni::Object>(Meta::data().query, uri.object(), projection,
                                   selection, selectionArgs, sortOrder));
}

inline database::Cursor
ContentResolver::query(net::Uri const &uri,
                       jni::Array<std::string> const &projection) {
    assert(!isNull());
    return database::Cursor(
        object().call<jni::Object>(Meta::data().query, uri.object(), projection,
                                   nullptr, nullptr, nullptr));
}

inline database::Cursor ContentResolver::query(
    net::Uri const &uri, jni::Array<std::string> const &projection,
    std::string const &selection, jni::Array<std::string> const &selectionArgs,
    std::string const &sortOrder, jni::Object const &cancellationSignal) {
    assert(!isNull());
    return database::Cursor(object().call<jni::Object>(
        Meta::data().query1, uri.object(), projection, selection, selectionArgs,
        sortOrder, cancellationSignal));
}

inline database::Cursor ContentResolver::query(
    net::Uri const &uri, jni::Array<std::string> const &projection,
    jni::Object const &queryArgs, jni::Object const &cancellationSignal) {
    assert(!isNull());
    return database::Cursor(
        object().call<jni::Object>(Meta::data().query2, uri.object(),
                                   projection, queryArgs, cancellationSignal));
}

} // namespace android::content
} // namespace wrap
