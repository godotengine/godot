// Copyright 2020-2021, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
// Author: Ryan Pavlik <ryan.pavlik@collabora.com>

#include "android.database.h"

namespace wrap {
namespace android::database {
Cursor::Meta::Meta()
    : MetaBaseDroppable(Cursor::getTypeName()),
      getCount(classRef().getMethod("getCount", "()I")),
      moveToFirst(classRef().getMethod("moveToFirst", "()Z")),
      moveToNext(classRef().getMethod("moveToNext", "()Z")),
      getColumnIndex(
          classRef().getMethod("getColumnIndex", "(Ljava/lang/String;)I")),
      getString(classRef().getMethod("getString", "(I)Ljava/lang/String;")),
      getInt(classRef().getMethod("getInt", "(I)I")),
      close(classRef().getMethod("close", "()V")) {
    MetaBaseDroppable::dropClassRef();
}
} // namespace android::database
} // namespace wrap
