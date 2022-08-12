// Copyright 2020-2021, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
// Author: Ryan Pavlik <ryan.pavlik@collabora.com>

#include "android.content.h"

namespace wrap {
namespace android::content {
Context::Meta::Meta(bool deferDrop)
    : MetaBaseDroppable(Context::getTypeName()),
      getContentResolver(classRef().getMethod(
          "getContentResolver", "()Landroid/content/ContentResolver;")) {
    if (!deferDrop) {
        MetaBaseDroppable::dropClassRef();
    }
}
ContentUris::Meta::Meta(bool deferDrop)
    : MetaBaseDroppable(ContentUris::getTypeName()),
      appendId(classRef().getStaticMethod(
          "appendId",
          "(Landroid/net/Uri$Builder;J)Landroid/net/Uri$Builder;")) {
    if (!deferDrop) {
        MetaBaseDroppable::dropClassRef();
    }
}
ComponentName::Meta::Meta()
    : MetaBase(ComponentName::getTypeName()),
      init(classRef().getMethod("<init>",
                                "(Ljava/lang/String;Ljava/lang/String;)V")),
      init1(classRef().getMethod(
          "<init>", "(Landroid/content/Context;Ljava/lang/String;)V")),
      init2(classRef().getMethod(
          "<init>", "(Landroid/content/Context;Ljava/lang/Class;)V")),
      init3(classRef().getMethod("<init>", "(Landroid/os/Parcel;)V")) {}
ContentResolver::Meta::Meta()
    : MetaBaseDroppable(ContentResolver::getTypeName()),
      query(classRef().getMethod(
          "query",
          "(Landroid/net/Uri;[Ljava/lang/String;Ljava/lang/String;[Ljava/lang/"
          "String;Ljava/lang/String;)Landroid/database/Cursor;")),
      query1(classRef().getMethod(
          "query", "(Landroid/net/Uri;[Ljava/lang/String;Ljava/lang/"
                   "String;[Ljava/lang/String;Ljava/lang/String;Landroid/os/"
                   "CancellationSignal;)Landroid/database/Cursor;")),
      query2(classRef().getMethod(
          "query",
          "(Landroid/net/Uri;[Ljava/lang/String;Landroid/os/Bundle;Landroid/os/"
          "CancellationSignal;)Landroid/database/Cursor;")) {
    MetaBaseDroppable::dropClassRef();
}
} // namespace android::content
} // namespace wrap
