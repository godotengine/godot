/**************************************************************************
 *
 * Copyright 2014 VMware, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/


/*
 * Including system's headers inside `extern "C" { ... }` is not safe, as system
 * headers may have C++ code in them, and C++ code inside extern "C"
 * leads to syntactically incorrect code.
 *
 * This is because putting code inside extern "C" won't make __cplusplus define
 * go away, that is, the system header being included thinks is free to use C++
 * as it sees fits.
 *
 * Including non-system headers inside extern "C"  is not safe either, because
 * non-system headers end up including system headers, hence fall in the above
 * case too.
 *
 * Conclusion, includes inside extern "C" is simply not portable.
 *
 *
 * This header helps surface these issues.
 */

#ifdef __cplusplus
template<class T> class _IncludeInsideExternCNotPortable;
#endif
