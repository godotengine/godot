#extension GL_GOOGLE_cpp_style_line_directive : enable

#error at "0:3"

#line 150 "a.h"
#error at "a.h:150"

#line 24
#error at "a.h:24"

#line 42
#error at "a.h:42"

#line 30 "b.cc"
#error at "b.cc:30"

#line 10 3
#error at "3:10"

#line 48
#error at "3:48"

#line 4
#error at "3:4"

#line 55 100
#error at "100:55"

#line 1000 "c"
#error at "c:1000"

#line 42 1
#error at "1:42"

#line 42 "this-is-a-quite-long-name-maybe-i-should-shorten-it"
#error at "this-is-a-quite-long-name-maybe-i-should-shorten-it:42"
