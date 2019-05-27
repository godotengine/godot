# Copyright 2009 The RE2 Authors.  All Rights Reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Defines a Bazel macro that instantiates a native cc_test rule for an RE2 test.
def re2_test(name, deps=[], size="medium"):
  native.cc_test(
      name=name,
      srcs=["re2/testing/%s.cc" % (name)],
      deps=[":test"] + deps,
      size=size,
  )
