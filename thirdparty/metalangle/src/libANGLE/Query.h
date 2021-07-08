//
// Copyright 2012 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Query.h: Defines the gl::Query class

#ifndef LIBANGLE_QUERY_H_
#define LIBANGLE_QUERY_H_

#include "common/PackedEnums.h"
#include "libANGLE/Debug.h"
#include "libANGLE/Error.h"
#include "libANGLE/RefCountObject.h"

#include "common/angleutils.h"

#include "angle_gl.h"

namespace rx
{
class QueryImpl;
}

namespace gl
{

class Query final : public RefCountObject<QueryID>, public LabeledObject
{
  public:
    Query(rx::QueryImpl *impl, QueryID id);
    ~Query() override;
    void onDestroy(const Context *context) override;

    void setLabel(const Context *context, const std::string &label) override;
    const std::string &getLabel() const override;

    angle::Result begin(const Context *context);
    angle::Result end(const Context *context);
    angle::Result queryCounter(const Context *context);
    angle::Result getResult(const Context *context, GLint *params);
    angle::Result getResult(const Context *context, GLuint *params);
    angle::Result getResult(const Context *context, GLint64 *params);
    angle::Result getResult(const Context *context, GLuint64 *params);
    angle::Result isResultAvailable(const Context *context, bool *available);

    QueryType getType() const;

    rx::QueryImpl *getImplementation() const;

  private:
    rx::QueryImpl *mQuery;

    std::string mLabel;
};
}  // namespace gl

#endif  // LIBANGLE_QUERY_H_
