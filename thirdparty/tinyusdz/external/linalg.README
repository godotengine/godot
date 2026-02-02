# linalg.h

[![Release is 2.2-beta](https://img.shields.io/badge/version-2.2--beta-blue.svg)](http://raw.githubusercontent.com/sgorsten/linalg/v3/linalg.h)
[![License is Unlicense](http://img.shields.io/badge/license-Unlicense-blue.svg?style=flat)](http://unlicense.org/)
[![Travis CI build status](http://travis-ci.org/sgorsten/linalg.svg)](https://travis-ci.org/sgorsten/linalg)
[![Appveyor build status](http://ci.appveyor.com/api/projects/status/l4bfv5omodkajuc9?svg=true)](https://ci.appveyor.com/project/sgorsten/linalg)

[`linalg.h`](/linalg.h) is a [single header](http://github.com/nothings/stb/blob/master/docs/other_libs.md), [public domain](http://unlicense.org/), [short vector math](http://www.reedbeta.com/blog/on-vector-math-libraries/) library for [C++](http://en.cppreference.com/w/). It is inspired by the syntax of popular shading and compute languages and is intended to serve as a lightweight alternative to projects such as [GLM](http://glm.g-truc.net/0.9.7/), [Boost.QVM](https://www.boost.org/doc/libs/1_66_0/libs/qvm/doc/index.html) or [Eigen](http://eigen.tuxfamily.org/) in domains such as computer graphics, computational geometry, and physical simulation. It allows you to easily write programs like the following:

```cpp
#include <linalg.h>
using namespace linalg::aliases;

// Compute the coefficients of the equation of a plane containing points a, b, and c
float4 compute_plane(float3 a, float3 b, float3 c)
{
    float3 n = cross(b-a, c-a);
    return {n, -dot(n,a)};
}
```

`linalg.h` aims to be:

* **Lightweight**: The library is defined in a single header file which is less than a thousand lines of code.
* **Dependency free**: There are no dependencies beyond a compliant C++11 compiler and a small subset of the standard library.
* **Standards compliant**: Almost all operations are free of undefined behavior and can be evaluated in a `constexpr` context.
* **Generic**: All types and operations are parameterized over scalar type, and can be mixed within expressions. Type promotion rules roughly match the C standard.
* **Consistent**: Named functions and overloaded operators perform the same conceptual operation on all data types for which they are supported.
* **Complete**: There are very few restrictions on which operations may be applied to which data types.
* **Easy to integrate**: The library defines no symbols in the public namespace, and provides a mechanism for defining implicit conversions to external or user-provided data types.

The documentation for `v2.2` is still in progress.

* [Data structures](#data-structures)
  * [Vectors](#vectors)
  * [Matrices](#matrices)
* [Function listing](#function-listing)
  * [Vector algebra](#vector-algebra)
  * [Quaternion algebra](#quaternion-algebra)  
  * [Matrix algebra](#matrix-algebra)
  * [Component-wise operations](#component-wise-operations)
  * [Reductions](#reductions)
* [Optional features](#optional-features)
  * [Type aliases](#type-aliases)
  * [`ostream` overloads](#ostream-overloads)
  * [User-defined conversions](#user-defined-conversions)
* [Higher order functions](#higher-order-functions)
* [Changes from v2.1](#changes-from-v21)

## Data structures

#### Vectors

`linalg::vec<T,M>` defines a fixed-length vector containing exactly `M` elements of type `T`. Convenience aliases such as `float3`, `float4`, or `int2` are provided in the [`linalg::aliases` namespace](#type-aliases). This data structure can be used to store a wide variety of types of data, including geometric vectors, points, homogeneous coordinates, plane equations, colors, texture coordinates, or any other situation where you need to manipulate a small sequence of numbers. As such, `vec<T,M>` is supported by a set of [algebraic](#vector-algebra) and [component-wise](#component-wise-operations) functions, as well as a set of standard [reductions](#reductions).

`vec<T,M>`:
* is [`DefaultConstructible`](https://en.cppreference.com/w/cpp/named_req/DefaultConstructible):
  ```cpp
  float3 v; // v contains 0,0,0
  ```
* is constructible from `M` elements of type `T`:
  ```cpp
  float3 v {1,2,3}; // v contains 1,2,3
  ```
* is [`CopyConstructible`](https://en.cppreference.com/w/cpp/named_req/CopyConstructible) and [`CopyAssignable`](https://en.cppreference.com/w/cpp/named_req/CopyAssignable): 
  ```cpp
  float3 v {1,2,3}; // v contains 1,2,3
  float3 u {v};     // u contains 1,2,3
  float3 w;         // w contains 0,0,0 
  w = u;            // w contains 1,2,3
  ```
* is [`EqualityComparable`](https://en.cppreference.com/w/cpp/named_req/EqualityComparable) and [`LessThanComparable`](https://en.cppreference.com/w/cpp/named_req/LessThanComparable):
  ```cpp
  if(v == y) cout << "v and u contain equal elements in the same positions" << endl;
  if(v < u) cout << "v precedes u lexicographically" << endl;
  ```  
* is **explicitly** constructible from a single element of type `T`:
  ```cpp
  float3 v = float3{4}; // v contains 4,4,4
  ```
* is **explicitly** constructible from a `vec<U,M>` of some other type `U`:
  ```cpp
  float3 v {1.1f,2.3f,3.5f}; // v contains 1.1,2.3,3.5
  int3 u = int3{v};          // u contains 1,2,3
  ```
* has fields `x,y,z,w`:
  ```cpp
  float y = point.y;    // y contains second element of point
  pixel.w = 0.5;        // fourth element of pixel set to 0.5
  float s = tc.x;       // s contains first element of tc
  ```
* supports indexing: 
  ```cpp
  float x = v[0]; // x contains first element of v
  v[2] = 5;       // third element of v set to 5
  ```
* supports unary operators `+`, `-`, `!` and `~` in component-wise fashion: 
  ```cpp
  auto v = -float{2,3}; // v is float2{-2,-3}
  ```
* supports binary operators `+`, `-`, `*`, `/`, `%`, `|`, `&`, `^`, `<<` and `>>` in component-wise fashion: 
  ```cpp
  auto v = float2{1,1} + float2{2,3}; // v is float2{3,4}
  ```
* supports binary operators with a scalar on the left or the right:
  ```cpp
  auto v = 2 * float3{1,2,3}; // v is float3{2,4,6}
  auto u = float3{1,2,3} + 1; // u is float3{2,3,4}
  ```
* supports operators `+=`, `-=`, `*=`, `/=`, `%=`, `|=`, `&=`, `^=`, `<<=` and `>>=` with vectors or scalars on the right:
  ```cpp
  float2 v {1,2}; v *= 3; // v is float2{3,6}
  ```
* supports operations on mixed element types: 
  ```cpp
  auto v = float3{1,2,3} + int3{4,5,6}; // v is float3{5,7,9}
  ```
* supports [range-based for](https://en.cppreference.com/w/cpp/language/range-for):
  ```cpp
  for(auto elem : float3{1,2,3}) cout << elem << ' '; // prints "1 2 3 "
  ```
* has a flat memory layout: 
  ```cpp
  float3 v {1,2,3}; 
  float * p = v.data(); // &v[i] == p+i
  p[1] = 4; // v contains 1,4,3
  ```

#### Matrices

`linalg::mat<T,M,N>` defines a fixed-size matrix containing exactly `M` rows and `N` columns of type `T`, in column-major order. Convenience aliases such as `float4x4` or `double3x3` are provided in the [`linalg::aliases` namespace](#type-aliases). This data structure is supported by a set of [algebraic](#matrix-algebra) functions and [component-wise](#component-wise-operations) functions, as well as a set of standard [reductions](#reductions).

`mat<T,M,N>`:
* is [`DefaultConstructible`](https://en.cppreference.com/w/cpp/named_req/DefaultConstructible):
  ```cpp
  float2x2 m; // m contains columns 0,0; 0,0
  ```
* is constructible from `N` columns of type `vec<T,M>`: 
  ```cpp
  float2x2 m {{1,2},{3,4}}; // m contains columns 1,2; 3,4
  ```
* is constructible from `linalg::identity`:
  ```cpp
  float3x3 m = linalg::identity; // m contains columns 1,0,0; 0,1,0; 0,0,1
  ```
* is [`CopyConstructible`](https://en.cppreference.com/w/cpp/named_req/CopyConstructible) and [`CopyAssignable`](https://en.cppreference.com/w/cpp/named_req/CopyAssignable): 
  ```cpp
  float2x2 m {{1,2},{3,4}}; // m contains columns 1,2; 3,4
  float2x2 n {m};           // n contains columns 1,2; 3,4
  float2x2 p;               // p contains columns 0,0; 0,0
  p = n;                    // p contains columns 1,2; 3,4
  ```
* is [`EqualityComparable`](https://en.cppreference.com/w/cpp/named_req/EqualityComparable) and [`LessThanComparable`](https://en.cppreference.com/w/cpp/named_req/LessThanComparable):
  ```cpp
  if(m == n) cout << "m and n contain equal elements in the same positions" << endl;
  if(m < n) cout << "m precedes n lexicographically when compared in column-major order" << endl;
  ```  
* is **explicitly** constructible from a single element of type `T`: 
  ```cpp
  float2x2 m {5}; // m contains columns 5,5; 5,5
  ```
* is **explicitly** constructible from a `mat<U,M,N>` of some other type `U`: 
  ```cpp
  float2x2 m {int2x2{{5,6},{7,8}}}; // m contains columns 5,6; 7,8
  ```
* supports indexing into *columns*: 
  ```cpp
  float2x3 m {{1,2},{3,4},{5,6}}; // m contains columns 1,2; 3,4; 5,6
  float2 c = m[0];                // c contains 1,2
  m[1]     = {7,8};               // m contains columns 1,2; 7,8; 5,6
  ```
* supports retrieval (but not assignment) of rows:
  ```cpp
  float2x3 m {{1,2},{3,4},{5,6}}; // m contains columns 1,2; 3,4; 5,6
  float3 r = m.row(1);            // r contains 2,4,6
  ```
  
  
  
* supports unary operators `+`, `-`, `!` and `~` in component-wise fashion:
  ```cpp
  float2x2 m {{1,2},{3,4}}; // m contains columns 1,2; 3,4
  float2x2 n = -m;          // n contains columns -1,-2; -3,-4
  ```
* supports binary operators `+`, `-`, `*`, `/`, `%`, `|`, `&`, `^`, `<<` and `>>` in component-wise fashion:
  ```cpp
  float2x2 a {{0,0},{2,2}}; // a contains columns 0,0; 2,2
  float2x2 b {{1,2},{1,2}}; // b contains columns 1,2; 1,2
  float2x2 c = a + b;       // c contains columns 1,2; 3,4
  ```
  
* supports binary operators with a scalar on the left or the right:
  ```cpp
  auto m = 2 * float2x2{{1,2},{3,4}}; // m is float2x2{{2,4},{6,8}}
  ```  
  
* supports operators `+=`, `-=`, `*=`, `/=`, `%=`, `|=`, `&=`, `^=`, `<<=` and `>>=` with matrices or scalars on the right:
  ```cpp
  float2x2 v {{5,4},{3,2}}; 
  v *= 3; // v is float2x2{{15,12},{9,6}}
  ```  
  
* supports operations on mixed element types: 
  
* supports [range-based for](https://en.cppreference.com/w/cpp/language/range-for) over columns

* has a flat memory layout

## Function listing

#### Vector algebra

* `cross(vec<T,3> a, vec<T,3> b) -> vec<T,3>` is the [cross or vector product](https://en.wikipedia.org/wiki/Cross_product) of vectors `a` and `b`
  * `cross(vec<T,2> a, vec<T,2> b) -> T` is shorthand for `cross({a.x,a.y,0}, {b.x,b.y,0}).z`
  * `cross(T a, vec<T,2> b) -> vec<T,2>` is shorthand for `cross({0,0,a.z}, {b.x,b.y,0}).xy()`
  * `cross(vec<T,2> a, T b) -> vec<T,2>` is shorthand for `cross({a.x,a.y,0}, {0,0,b.z}).xy()`

* `dot(vec<T,M> a, vec<T,M> b) -> T` is the [dot or inner product](https://en.wikipedia.org/wiki/Dot_product) of vectors `a` and `b`

* `length(vec<T,M> a) -> T` is the length or magnitude of a vector `a`
* `length2(vec<T,M> a) -> T` is the *square* of the length or magnitude of vector `a`
* `normalize(vec<T,M> a) -> vec<T,M>` is a unit length vector in the same direction as `a` (undefined for zero-length vectors)

* `distance(vec<T,M> a, vec<T,M> b) -> T` is the [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) between points `a` and `b`
* `distance2(vec<T,M> a, vec<T,M> b) -> T` is the *square* of the [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) between points `a` and `b`

* `angle(vec<T,M> a, vec<T,M> b) -> T` is the angle in [radians](https://en.wikipedia.org/wiki/Radian) between vectors `a` and `b`
* `uangle(vec<T,M> a, vec<T,M> b) -> T` is the angle in [radians](https://en.wikipedia.org/wiki/Radian) between unit vectors `a` and `b` (undefined for non-unit vectors)
* `rot(T a, vec<T,2> v) -> vec<T,2>` is the vector `v` rotated counter-clockwise by the angle `a` in [radians](https://en.wikipedia.org/wiki/Radian)

* `nlerp(vec<T,M> a, vec<T,M> b, T t) -> vec<T,M>` is shorthand for `normalize(lerp(a,b,t))`
* `slerp(vec<T,M> a, vec<T,M> b, T t) -> vec<T,M>` is the [spherical linear interpolation](https://en.wikipedia.org/wiki/Slerp) between unit vectors `a` and `b` (undefined for non-unit vectors) by parameter `t`

#### Quaternion algebra

A small set of functions provides support for quaternion math, using `vec<T,4>` values to represent quaternions of the form `xi + yj + zk + w`.

* `qmul(vec<T,4> a, vec<T,4> b) -> vec<T,4>` is the [Hamilton product](https://en.wikipedia.org/wiki/Quaternion#Hamilton_product) of quaternions `a` and `b`
* `qconj(vec<T,4> q) -> vec<T,4>` is the [conjugate](https://en.wikipedia.org/wiki/Quaternion#Conjugation,_the_norm,_and_reciprocal) of quaternion `q`
* `qinv(vec<T,4> q) -> vec<T,4>` is the [inverse or reciprocal](https://en.wikipedia.org/wiki/Quaternion#Conjugation,_the_norm,_and_reciprocal) of quaternion `q` (undefined for zero-length quaternions)

* `qexp(vec<T,4> q) -> vec<T,4>` is the [exponential](https://en.wikipedia.org/wiki/Quaternion#Exponential,_logarithm,_and_power_functions) of quaternion `q`
* `qlog(vec<T,4> q) -> vec<T,4>` is the [logarithm](https://en.wikipedia.org/wiki/Quaternion#Exponential,_logarithm,_and_power_functions) of quaternion `q`
* `qpow(vec<T,4> q T p) -> vec<T,4>` is the quaternion `q` raised to the exponent `p`

A second set of functions provides support for using unit-length quaternions to represent 3D spatial rotations. Their results are undefined for quaternions which are not of unit-length.

* `qangle(vec<T,4> q)` is the angle in radians of the rotation expressed by quaternion `q`
* `qaxis(vec<T,4> q)` is the axis of rotation expression by quaternion `q` (undefined for zero-angle quaternions)
* `qrot(vec<T,4> q, vec<T,3> v) -> vec<T,3>` is vector `v` rotated via rotation quaternion `q`

* `qmat(vec<T,4> q)` is a 3x3 rotation matrix which performs the same operation as rotation quaternion `q`
* `qxdir(vec<T,4> q)` is (efficient) shorthand for `qrot(q, {1,0,0})`
* `qydir(vec<T,4> q)` is (efficient) shorthand for `qrot(q, {0,1,0})`
* `qzdir(vec<T,4> q)` is (efficient) shorthand for `qrot(q, {0,0,1})`  

It is possible to use the `nlerp` and `slerp` functions to interpolate rotation quaternions as though they were simply four-dimensional vectors. However, the rotation quaternions form a [double cover](https://en.wikipedia.org/wiki/Covering_group) over spatial rotations in three dimensions. This means that there are **two** distinct rotation quaternions representing each spatial rotation. Naively interpolating between two spatial rotations using quaternions could follow the "short path" or the "long path" between these rotations, depending on which specific quaternions are being interpolated. 

* `qnlerp(vec<T,4> a, vec<T,4> b, T t)` is similar to `nlerp(a,b,t)`, but always chooses the "short path" between the rotations represented by `a` and `b`.
* `qslerp(vec<T,4> a, vec<T,4> b, T t)` is similar to `slerp(a,b,t)`, but always chooses the "short path" between the rotations represented by `a` and `b`.

#### Matrix algebra

* `mul(mat<T,M,N> a, mat<T,N,P> b) -> mat<T,M,P>` is the [matrix product](https://en.wikipedia.org/wiki/Matrix_multiplication) of matrices `a` and `b`
** `mul(mat<T,M,N> a, vec<T,N> b) -> vec<T,M>` is the [matrix product](https://en.wikipedia.org/wiki/Matrix_multiplication) of matrix `a` and a column matrix containing the elements of vector `b`
** `mul(a, b, c)` is shorthand for `mul(mul(a, b), c)`

* `outerprod(vec<T,M> a, vec<T,N> b) -> mat<T,M,N>` is the [outer product](https://en.wikipedia.org/wiki/Outer_product) of vectors `a` and `b`

* `diagonal(mat<T,N,N> a) -> vec<T,N>` is a vector containing the elements along the main diagonal of matrix `a`
* `trace(mat<T,N,N> a) -> T` is the sum of the elements along the main diagonal of matrix `a`

* `transpose(mat<T,M,N> a) -> mat<T,N,M>` is the [transpose](https://en.wikipedia.org/wiki/Transpose) of matrix `a`
* `adjugate(mat<T,N,N> a) -> mat<T,N,N>` is the [adjugate or classical adjoint](https://en.wikipedia.org/wiki/Adjugate_matrix) of matrix `a` (the transpose of its cofactor matrix, or the numerator in the expression of its inverse)
* `comatrix(mat<T,N,N> a) -> mat<T,N,N>` is the [comatrix or cofactor matrix](https://en.wikipedia.org/wiki/Minor_(linear_algebra)#Inverse_of_a_matrix) of matrix `a` (the transpose of its adjugate matrix)

* `determinant(mat<T,N,N> a) -> T` is the [determinant](https://en.wikipedia.org/wiki/Determinant) of matrix `a`
* `inverse(mat<T,N,N> a) -> mat<T,N,N>` is the [multiplicative inverse](https://en.wikipedia.org/wiki/Multiplicative_inverse) of the [invertible matrix](https://en.wikipedia.org/wiki/Invertible_matrix) `a` (undefined for singular inputs)

#### Component-wise operations

The unary functions `abs`, `floor`, `ceil`, `exp`, `log`, `log10`, `sqrt`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `round` accept a vector-valued argument and produce a vector-valued result by passing individual elements to the function of the same name in the `std::` namespace, as defined by `<cmath>` or `<cstdlib>`.

```cpp
float4 a {1,-4,9,-16}; // a contains 1,-4,9,-16
float4 b = abs(a);     // b contains 1,4,9,16
float4 c = sqrt(b);    // c contains 1,2,3,4
```

The binary functions `fmod`, `pow`, `atan2`, and `copysign` function similarly, except that either argument can be a vector or a scalar.

```cpp
float2 a {5,4}, b {2,3};
float2 c = pow(a, 2);    // c contains 25,16
float2 d = pow(2, b);    // d contains 4,8
float2 e = pow(a, b);    // e contains 25,64
```

The binary functions `equal`, `nequal`, `less`, `greater`, `lequal`, and `gequal` apply operators `==`, `!=`, `<`, `>`, `<=` and `>=` respectively in a component-wise fashion, returning a `vec<bool,M>`. As before, either argument can be a vector or a scalar.

```cpp
int2 a {2,5}, b {3,4};
bool2 c = less(a,3);    // c contains true, false
bool2 d = equal(4,b);   // d contains false, true
bool2 e = greater(a,b); // e contains false, true
```

`min(a,b) -> vec<T,M>` performs the component-wise selection of lesser elements, as by `a[i] < b[i] ? a[i] : b[i]`. Either argument can be a vector or a scalar.

`max(a,b) -> vec<T,M>` performs the component-wise selection of greater elements, as by `a[i] > b[i] ? a[i] : b[i]`. Either argument can be a vector or a scalar.

`clamp(x,l,h) -> vec<T,M>` performs the component-wise clamping of elements between a low and high boundary, as by `min(max(x,l),h)`. Any argument can be a vector or a scalar.

`select(p,a,b) -> vec<T,M>` performs a component-wise ternary operator, as by `p[i] ? a[i] : b[i]`. Any argument can be a vector or a scalar.

`lerp(a,b,t) -> vec<T,M>` performs a component-wise linear interpolation, as by `a[i]*(1-t[i]) + b[i]*t[i]`. Any argument can be a vector or a scalar.

#### Reductions

* `any(vec<bool,M> a) -> bool` is `true` if any element of the vector `a` is `true`
* `all(vec<bool,M> a) -> bool` is `true` if all elements of the vector `a` are `true`
* `sum(vec<T,M> a) -> T` is the sum of all elements in the vector `a`
* `product(vec<T,M> a) -> T` returns the product of all elements in the vector `a`
* `minelem(vec<T,M> a) -> T` returns the **value** of the least element in the vector `a`
* `maxelem(vec<T,M> a) -> T` returns the **value** of the greatest element in the vector `a`
* `argmin(vec<T,M> a) -> int` returns the **zero-based index** of the least element in the vector `a`
* `argmax(vec<T,M> a) -> int` returns the **zero-based index** of the greatest element in the vector `a`

#### Comparisons

`compare(a,b)` is conceptually equivalent to `operator <=>` from [C++20](https://en.cppreference.com/w/cpp/language/default_comparisons). It compares two values of equivalent shape and returns a value which supports all six standard comparisons against `0`. It provides the same ordering guarantees as the underlying scalar type. That is, a `vec<int,M>` provides a strong ordering, where a `vec<float,M>` provides a partial odering.

## Optional features

#### Type aliases

By default, `linalg.h` does not define any symbols in the global namespace, and a three-element vector of single-precision floating point values must be spelled `linalg::vec<float,3>`. In various libraries and shading languages, such a type might be spelled `float3`, `vec3`, `vec3f`, `point3f`, `simd_float3`, or any one of a hundred other possibilities. `linalg.h` provides a collection of useful aliases in the `linalg::aliases` namespace. If the names specified in this namespace are suitable for a user's purposes, they can quickly be brought into scope as follows:

```cpp
#include <linalg.h>
using namespace linalg::aliases;

float3 a_vector;
float4x4 a_matrix;
```

Note that this **only** brings the type aliases into global scope. The core types and all functions and operator overloads defined by the library remain in `namespace linalg`. 

If the spellings in `namespace linalg::aliases` conflict with other types that have been defined in the global namespace or in other namespaces of interest, the user can choose to omit the `using namespace` directive and instead define their own aliases as desired.

```cpp
#include <linalg.h>
using v3f = linalg::vec<float,3>;
using m44f = linalg::mat<float,4,4>;

v3f a_vector;
m44f a_matrix;
```

It is, of course, always possible to use the core `linalg.h` types directly if operating in an environment where no additional symbols should be defined.

```cpp
#include <linalg.h>

linalg::vec<float,3> a_vector;
linalg::mat<float,4,4> a_matrix;
```

The set of type aliases defined in `namespace linalg::aliases` is as follows:

* `vec<float,M>` aliased to *floatM*, as in: `float1`, `float2`, `float3`, `float4`
* `vec<double,M>` aliased to *doubleM*, as in: `double1`, `double2`, `double3`, `double4`
* `vec<int,M>` aliased to *intM* as in: `int1`, `int2`, `int3`, `int4`
* `vec<unsigned,M>` aliased to *uintM* as in: `uint1`, `uint2`, `uint3`, `uint4`
* `vec<bool,M>` aliased to *boolM* as in: `bool1`, `bool2`, `bool3`, `bool4`
* `vec<int16_t,M>` aliased to *shortM* as in: `short1`, `short2`, `short3`, `short4`
* `vec<uint16_t,M>` aliased to *ushortM* as in: `ushort1`, `ushort2`, `ushort3`, `ushort4`
* `vec<uint8_t,M>` aliased to *byteM* as in: `byte1`, `byte2`, `byte3`, `byte4`
* `mat<float,M,N>` aliased to *floatMxN* as in: `float1x3`, `float3x2`, `float4x4`, etc.
* `mat<double,M,N>` aliased to *doubleMxN* as in: `double1x3`, `double3x2`, `double4x4`, etc.
* `mat<int,M,N>` aliased to *intMxN* as in: `int1x3`, `int3x2`, `int4x4`, etc.
* `mat<bool,M,N>` aliased to *boolMxN* as in: `boolx3`, `bool3x2`, `bool4x4`, etc.

All combinations of up to four elements, rows, or columns are provided.

#### `ostream` overloads

By default, `linalg.h` does not provide operators for interaction with standard library streams. This is to permit maximum flexibility for users who wish to define their own formatting (with or without delimiters, row versus column major matrices, human-readable precision or round-trip exact). However, as it is often useful to simply be able to show something when writing small programs, we provide some default stream operator overloads which can be brought into scope with:

```cpp
#include "linalg.h"
using namespace linalg::ostream_overloads;
```

The provided behavior is to output a string using the currently specified stream properties (width, precision, padding, etc) which matches the braced-initialization syntax that could be used to construct that same value, without any extra whitespace.

```cpp
int3 v {1, 2, 3};
int2x2 m {{4, 5}, {6, 7}};
std::cout << v << std::endl; // Prints {1,2,3}
std::wcout << m << std::endl; // Prints {{4,5},{6,7}}
```

#### User-defined conversions

A mechanism exists to define automatic conversions between `linalg` and user-provided types. As an example, this mechanism has already been used to defined bidirectional conversions between `linalg::vec<T,M>` and `std::array<T,M>`.

**TODO: Explain `converter<T,U>`**

## Higher order functions

#### `linalg::fold(f, a, b)`

`fold(f, a, b)` is a higher order function which accepts a function of the form `A,B => A` and repeatedly invokes `a = f(a, element_of_b)` until all elements have been consumed, before returning `a`. It is approximately equal to a [left fold with an initial value](https://en.wikipedia.org/wiki/Fold_(higher-order_function)). When `b` is a `vec<T,M>`, elements are folded from least to greatest index. When `b` is a `mat<T,M,N>`, elements are folded in column-major order.

See also: [Reductions](#reductions)

#### `linalg::apply(f, a...)`

`apply(f, a...)` is a higher order function which accepts a function of the form `A... => T` and applies it to component-wise sets of elements from data structures of compatible shape and dimensions. It is approximately equal to a [convolution](https://en.wikipedia.org/wiki/Convolution_(computer_science)) followed by a [map](https://en.wikipedia.org/wiki/Map_(higher-order_function)). The shape of the result (that is, whether it is a scalar, vector, or matrix, and the dimensions thereof) is determined by the arguments. If more than one argument is a non-scalar, the shape of those arguments must agree. Scalars can be freely intermixed with non-scalars, and element types can also be freely mixed. The element type of the returned value is determined by the return type of the provided mapping function `f`. The supported call signatures are enumerated in the following table:

| call             | type of `a`  | type of `b`  | type of `c` | result type  | result elements          |
|------------------|--------------|--------------|-------------|--------------|--------------------------|
| `apply(f,a)`     | `A`          |              |             | `T`          | `f(a)`                   |
| `apply(f,a)`     | `vec<A,M>`   |              |             | `vec<T,M>`   | `f(a[i])...`             |
| `apply(f,a)`     | `mat<A,M,N>` |              |             | `mat<T,M,N>` | `f(a[j][i])...`          |
| `apply(f,a,b)`   | `A`          | `B`          |             | `T`          | `f(a, b)...`             |
| `apply(f,a,b)`   | `A`          | `vec<B,M>`   |             | `vec<T,M>`   | `f(a, b[i])...`          |
| `apply(f,a,b)`   | `vec<A,M>`   | `B`          |             | `vec<T,M>`   | `f(a[i], b)...`          |
| `apply(f,a,b)`   | `vec<A,M>`   | `vec<B,M>`   |             | `vec<T,M>`   | `f(a[i], b[i])...`       |
| `apply(f,a,b)`   | `A`          | `mat<B,M,N>` |             | `mat<T,M,N>` | `f(a, b[j][i])...`       |
| `apply(f,a,b)`   | `mat<A,M,N>` | `B`          |             | `mat<T,M,N>` | `f(a[j][i], b)...`       |
| `apply(f,a,b)`   | `mat<A,M,N>` | `mat<B,M,N>` |             | `mat<T,M,N>` | `f(a[j][i], b[j][i])...` |
| `apply(f,a,b,c)` | `A`          | `B`          | `C`         | `T`          | `f(a, b, c)...`          |
| `apply(f,a,b,c)` | `A`          | `B`          | `vec<C,M>`  | `vec<T,M>`   | `f(a, b, c[i])...`       |
| `apply(f,a,b,c)` | `A`          | `vec<B,M>`   | `C`         | `vec<T,M>`   | `f(a, b[i], c)...`       |
| `apply(f,a,b,c)` | `A`          | `vec<B,M>`   | `vec<C,M>`  | `vec<T,M>`   | `f(a, b[i], c[i])...`    |
| `apply(f,a,b,c)` | `vec<A,M>`   | `B`          | `C`         | `vec<T,M>`   | `f(a[i], b, c)...`       |
| `apply(f,a,b,c)` | `vec<A,M>`   | `B`          | `vec<C,M>`  | `vec<T,M>`   | `f(a[i], b, c[i])...`    |
| `apply(f,a,b,c)` | `vec<A,M>`   | `vec<B,M>`   | `C`         | `vec<T,M>`   | `f(a[i], b[i], c)...`    |
| `apply(f,a,b,c)` | `vec<A,M>`   | `vec<B,M>`   | `vec<C,M>`  | `vec<T,M>`   | `f(a[i], b[i], c[i])...` |

**TODO: Explain `apply_t<F, A...>` and SFINAE helpers.**

See also: [Component-wise operations](#component-wise-operations)

## Changes from `v2.1`

#### Improvements in `v2.2`

* `map(a,f)` and `zip(a,b,f)` subsumed by new `apply(f,a...)`
  * `apply(...)` supports unary, binary, and ternary operations for `vec`
  * `apply(...)` supports unary and binary operations for `mat` and `quat`
  * `apply(...)` can also be invoked exclusively with scalars, and supports arbitrary numbers of arguments
  * `apply(...)` supports mixed element types
  * Template type alias `apply_t<F,A...>` provides the return type of `apply(f,a...)`
* `vec<T,1>` and `mat<T,M,1>` specializations are now provided
* `compare(a,b)` provide three-way comparison between compatible types
* `clamp(a,b,c)` can be invoked with three distinct (but compatible) types
* `select(a,b,c)` provides the a component-wise equivalent to `a ? b : c`
* `lerp(a,b,t)` has been generalized to a component-wise operation where any of `a`, `b`, and `t` can be vectors or scalars
* User can specialize `converter<T,U>` to enable implicit conversions from `U` to `T`, if either type is a `vec`, `mat`, or `quat`
  * `identity` is implemented using this facility to serve as an in-library example
* No undefined behavior according to the C++11 standard
* Almost all operations which do not internally call `<cmath>` functions are `constexpr`, except for `argmin` and `argmax`
* No lambdas are used in `linalg.h`, avoiding potential ODR violations

#### Deprecations in `v2.2`

* `operator *` has been deprecated between pairs of matrices.
  * Call `cmul(...)` if the original, component-wise product was intended
  * Call `mul(...)` if the algebraic matrix product was intended

You can `#define LINALG_FORWARD_COMPATIBLE` before including `linalg.h` to remove all deprecated features.

#### Breaking changes in `v2.2-beta`

It is intended that compatibility will be restored before officially tagging `v2.2`

* `linalg.h` no longer supports Visual Studio 2013. However, it is known to work on GCC 4.9+, Clang 3.5+ in C++11 mode and Visual Studio 2015+.
* `vec<T,M>` and `mat<T,M,N>` may only be used with a `T` which is an [arithmetic type](https://en.cppreference.com/w/c/language/arithmetic_types)
  * This requirement will likely be relaxed, but will require specializing some trait type to indicate additional scalar types
