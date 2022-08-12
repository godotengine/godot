// Copyright 2020-2021, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
// Author: Ryan Pavlik <ryan.pavlik@collabora.com>

#include "android.net.h"

namespace wrap {
namespace android::net {
Uri::Meta::Meta()
    : MetaBaseDroppable(Uri::getTypeName()),
      toString(classRef().getMethod("toString", "()Ljava/lang/String;")) {
    MetaBaseDroppable::dropClassRef();
}
Uri_Builder::Meta::Meta()
    : MetaBaseDroppable(Uri_Builder::getTypeName()),
      init(classRef().getMethod("<init>", "()V")),
      scheme(classRef().getMethod(
          "scheme", "(Ljava/lang/String;)Landroid/net/Uri$Builder;")),
      authority(classRef().getMethod(
          "authority", "(Ljava/lang/String;)Landroid/net/Uri$Builder;")),
      appendPath(classRef().getMethod(
          "appendPath", "(Ljava/lang/String;)Landroid/net/Uri$Builder;")),
      build(classRef().getMethod("build", "()Landroid/net/Uri;")) {
    MetaBaseDroppable::dropClassRef();
}
} // namespace android::net
} // namespace wrap
