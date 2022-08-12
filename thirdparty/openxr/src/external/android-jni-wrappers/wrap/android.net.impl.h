// Copyright 2020-2021, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
// Author: Ryan Pavlik <ryan.pavlik@collabora.com>
// Inline implementations: do not include on its own!

#pragma once

#include <string>

namespace wrap {
namespace android::net {
inline std::string Uri::toString() const {
    assert(!isNull());
    return object().call<std::string>(Meta::data().toString);
}

inline Uri_Builder Uri_Builder::construct() {
    return Uri_Builder(Meta::data().clazz().newInstance(Meta::data().init));
}

inline Uri_Builder &Uri_Builder::scheme(std::string const &stringParam) {
    assert(!isNull());
    object().call<jni::Object>(Meta::data().scheme, stringParam);
    return *this;
}

inline Uri_Builder &Uri_Builder::authority(std::string const &stringParam) {
    assert(!isNull());
    object().call<jni::Object>(Meta::data().authority, stringParam);
    return *this;
}

inline Uri_Builder &Uri_Builder::appendPath(std::string const &stringParam) {
    assert(!isNull());
    object().call<jni::Object>(Meta::data().appendPath, stringParam);
    return *this;
}

inline Uri Uri_Builder::build() {
    assert(!isNull());
    return Uri(object().call<jni::Object>(Meta::data().build));
}

} // namespace android::net
} // namespace wrap
