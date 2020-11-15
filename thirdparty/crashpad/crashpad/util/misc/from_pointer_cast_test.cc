// Copyright 2017 The Crashpad Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "util/misc/from_pointer_cast.h"

#include <stdlib.h>
#include <sys/types.h>

#include <limits>

#include "build/build_config.h"
#include "gtest/gtest.h"
#include "test/gtest_death.h"

namespace crashpad {
namespace test {
namespace {

struct SomeType {};

template <typename T>
class FromPointerCastTest : public testing::Test {};

using FromPointerCastTestTypes = testing::Types<void*,
                                                const void*,
                                                volatile void*,
                                                const volatile void*,
                                                SomeType*,
                                                const SomeType*,
                                                volatile SomeType*,
                                                const volatile SomeType*>;

TYPED_TEST_CASE(FromPointerCastTest, FromPointerCastTestTypes);

TYPED_TEST(FromPointerCastTest, ToSigned) {
  EXPECT_EQ(FromPointerCast<int64_t>(nullptr), 0);
  EXPECT_EQ(FromPointerCast<int64_t>(reinterpret_cast<TypeParam>(1)), 1);
  EXPECT_EQ(FromPointerCast<int64_t>(reinterpret_cast<TypeParam>(-1)), -1);
  EXPECT_EQ(FromPointerCast<int64_t>(reinterpret_cast<TypeParam>(
                std::numeric_limits<uintptr_t>::max())),
            static_cast<intptr_t>(std::numeric_limits<uintptr_t>::max()));
  EXPECT_EQ(FromPointerCast<int64_t>(reinterpret_cast<TypeParam>(
                std::numeric_limits<intptr_t>::min())),
            std::numeric_limits<intptr_t>::min());
  EXPECT_EQ(FromPointerCast<int64_t>(reinterpret_cast<TypeParam>(
                std::numeric_limits<intptr_t>::max())),
            std::numeric_limits<intptr_t>::max());
}

TYPED_TEST(FromPointerCastTest, ToUnsigned) {
  EXPECT_EQ(FromPointerCast<uint64_t>(nullptr), 0u);
  EXPECT_EQ(FromPointerCast<uint64_t>(reinterpret_cast<TypeParam>(1)), 1u);
  EXPECT_EQ(FromPointerCast<uint64_t>(reinterpret_cast<TypeParam>(-1)),
            static_cast<uintptr_t>(-1));
  EXPECT_EQ(FromPointerCast<uint64_t>(reinterpret_cast<TypeParam>(
                std::numeric_limits<uintptr_t>::max())),
            std::numeric_limits<uintptr_t>::max());
  EXPECT_EQ(FromPointerCast<uint64_t>(reinterpret_cast<TypeParam>(
                std::numeric_limits<intptr_t>::min())),
            static_cast<uintptr_t>(std::numeric_limits<intptr_t>::min()));
  EXPECT_EQ(FromPointerCast<uint64_t>(reinterpret_cast<TypeParam>(
                std::numeric_limits<intptr_t>::max())),
            static_cast<uintptr_t>(std::numeric_limits<intptr_t>::max()));
}

// MatchCV<YourType, CVQualifiedType>::Type adapts YourType to match the
// const/void qualification of CVQualifiedType.
template <typename Base, typename MatchTo>
struct MatchCV {
 private:
  using NonCVBase = typename std::remove_cv<Base>::type;

 public:
  using Type = typename std::conditional<
      std::is_const<MatchTo>::value,
      typename std::conditional<std::is_volatile<MatchTo>::value,
                                const volatile NonCVBase,
                                const NonCVBase>::type,
      typename std::conditional<std::is_volatile<MatchTo>::value,
                                volatile NonCVBase,
                                NonCVBase>::type>::type;
};

#if defined(COMPILER_MSVC) && _MSC_VER < 1910
// gtest under MSVS 2015 (MSC 19.0) doesn’t handle EXPECT_EQ(a, b) when a or b
// is a pointer to a volatile type, because it can’t figure out how to print
// them.
template <typename T>
typename std::remove_volatile<typename std::remove_pointer<T>::type>::type*
MaybeRemoveVolatile(const T& value) {
  return const_cast<typename std::remove_volatile<
      typename std::remove_pointer<T>::type>::type*>(value);
}
#else  // COMPILER_MSVC && _MSC_VER < 1910
// This isn’t a problem in MSVS 2017 (MSC 19.1) or with other compilers.
template <typename T>
T MaybeRemoveVolatile(const T& value) {
  return value;
}
#endif  // COMPILER_MSVC && _MSC_VER < 1910

TYPED_TEST(FromPointerCastTest, ToPointer) {
  using CVSomeType =
      typename MatchCV<SomeType,
                       typename std::remove_pointer<TypeParam>::type>::Type;

  EXPECT_EQ(MaybeRemoveVolatile(FromPointerCast<CVSomeType*>(nullptr)),
            MaybeRemoveVolatile(static_cast<CVSomeType*>(nullptr)));
  EXPECT_EQ(MaybeRemoveVolatile(
                FromPointerCast<CVSomeType*>(reinterpret_cast<TypeParam>(1))),
            MaybeRemoveVolatile(reinterpret_cast<CVSomeType*>(1)));
  EXPECT_EQ(MaybeRemoveVolatile(
                FromPointerCast<CVSomeType*>(reinterpret_cast<TypeParam>(-1))),
            MaybeRemoveVolatile(reinterpret_cast<CVSomeType*>(-1)));
  EXPECT_EQ(
      MaybeRemoveVolatile(FromPointerCast<CVSomeType*>(
          reinterpret_cast<TypeParam>(std::numeric_limits<uintptr_t>::max()))),
      MaybeRemoveVolatile(reinterpret_cast<CVSomeType*>(
          std::numeric_limits<uintptr_t>::max())));
  EXPECT_EQ(
      MaybeRemoveVolatile(FromPointerCast<CVSomeType*>(
          reinterpret_cast<TypeParam>(std::numeric_limits<intptr_t>::min()))),
      MaybeRemoveVolatile(
          reinterpret_cast<CVSomeType*>(std::numeric_limits<intptr_t>::min())));
  EXPECT_EQ(
      MaybeRemoveVolatile(FromPointerCast<CVSomeType*>(
          reinterpret_cast<TypeParam>(std::numeric_limits<intptr_t>::max()))),
      MaybeRemoveVolatile(
          reinterpret_cast<CVSomeType*>(std::numeric_limits<intptr_t>::max())));
}

TEST(FromPointerCast, FromFunctionPointer) {
  // These casts should work with or without the & in &malloc.

  EXPECT_NE(FromPointerCast<int64_t>(malloc), 0);
  EXPECT_NE(FromPointerCast<int64_t>(&malloc), 0);

  EXPECT_NE(FromPointerCast<uint64_t>(malloc), 0u);
  EXPECT_NE(FromPointerCast<uint64_t>(&malloc), 0u);

  EXPECT_EQ(FromPointerCast<void*>(malloc), reinterpret_cast<void*>(malloc));
  EXPECT_EQ(FromPointerCast<void*>(&malloc), reinterpret_cast<void*>(malloc));
  EXPECT_EQ(FromPointerCast<const void*>(malloc),
            reinterpret_cast<const void*>(malloc));
  EXPECT_EQ(FromPointerCast<const void*>(&malloc),
            reinterpret_cast<const void*>(malloc));
  EXPECT_EQ(MaybeRemoveVolatile(FromPointerCast<volatile void*>(malloc)),
            MaybeRemoveVolatile(reinterpret_cast<volatile void*>(malloc)));
  EXPECT_EQ(MaybeRemoveVolatile(FromPointerCast<volatile void*>(&malloc)),
            MaybeRemoveVolatile(reinterpret_cast<volatile void*>(malloc)));
  EXPECT_EQ(
      MaybeRemoveVolatile(FromPointerCast<const volatile void*>(malloc)),
      MaybeRemoveVolatile(reinterpret_cast<const volatile void*>(malloc)));
  EXPECT_EQ(
      MaybeRemoveVolatile(FromPointerCast<const volatile void*>(&malloc)),
      MaybeRemoveVolatile(reinterpret_cast<const volatile void*>(malloc)));

  EXPECT_EQ(FromPointerCast<SomeType*>(malloc),
            reinterpret_cast<SomeType*>(malloc));
  EXPECT_EQ(FromPointerCast<SomeType*>(&malloc),
            reinterpret_cast<SomeType*>(malloc));
  EXPECT_EQ(FromPointerCast<const SomeType*>(malloc),
            reinterpret_cast<const SomeType*>(malloc));
  EXPECT_EQ(FromPointerCast<const SomeType*>(&malloc),
            reinterpret_cast<const SomeType*>(malloc));
  EXPECT_EQ(MaybeRemoveVolatile(FromPointerCast<volatile SomeType*>(malloc)),
            MaybeRemoveVolatile(reinterpret_cast<volatile SomeType*>(malloc)));
  EXPECT_EQ(MaybeRemoveVolatile(FromPointerCast<volatile SomeType*>(&malloc)),
            MaybeRemoveVolatile(reinterpret_cast<volatile SomeType*>(malloc)));
  EXPECT_EQ(
      MaybeRemoveVolatile(FromPointerCast<const volatile SomeType*>(malloc)),
      MaybeRemoveVolatile(reinterpret_cast<const volatile SomeType*>(malloc)));
  EXPECT_EQ(
      MaybeRemoveVolatile(FromPointerCast<const volatile SomeType*>(&malloc)),
      MaybeRemoveVolatile(reinterpret_cast<const volatile SomeType*>(malloc)));
}

TEST(FromPointerCast, ToNarrowInteger) {
  EXPECT_EQ(FromPointerCast<int>(nullptr), 0);
  EXPECT_EQ(FromPointerCast<int>(reinterpret_cast<void*>(1)), 1);
  EXPECT_EQ(FromPointerCast<int>(reinterpret_cast<void*>(-1)), -1);
  EXPECT_EQ(FromPointerCast<int>(reinterpret_cast<void*>(
                static_cast<intptr_t>(std::numeric_limits<int>::max()))),
            std::numeric_limits<int>::max());
  EXPECT_EQ(FromPointerCast<int>(reinterpret_cast<void*>(
                static_cast<intptr_t>(std::numeric_limits<int>::min()))),
            std::numeric_limits<int>::min());

  EXPECT_EQ(FromPointerCast<unsigned int>(nullptr), 0u);
  EXPECT_EQ(FromPointerCast<unsigned int>(reinterpret_cast<void*>(1)), 1u);
  EXPECT_EQ(
      FromPointerCast<unsigned int>(reinterpret_cast<void*>(
          static_cast<uintptr_t>(std::numeric_limits<unsigned int>::max()))),
      std::numeric_limits<unsigned int>::max());
  EXPECT_EQ(FromPointerCast<unsigned int>(reinterpret_cast<void*>(
                static_cast<uintptr_t>(std::numeric_limits<int>::max()))),
            static_cast<unsigned int>(std::numeric_limits<int>::max()));

  // int and unsigned int may not be narrower than a pointer, so also test short
  // and unsigned short.

  EXPECT_EQ(FromPointerCast<short>(nullptr), 0);
  EXPECT_EQ(FromPointerCast<short>(reinterpret_cast<void*>(1)), 1);
  EXPECT_EQ(FromPointerCast<short>(reinterpret_cast<void*>(-1)), -1);
  EXPECT_EQ(FromPointerCast<short>(reinterpret_cast<void*>(
                static_cast<intptr_t>(std::numeric_limits<short>::max()))),
            std::numeric_limits<short>::max());
  EXPECT_EQ(FromPointerCast<short>(reinterpret_cast<void*>(
                static_cast<intptr_t>(std::numeric_limits<short>::min()))),
            std::numeric_limits<short>::min());

  EXPECT_EQ(FromPointerCast<unsigned short>(nullptr), 0u);
  EXPECT_EQ(FromPointerCast<unsigned short>(reinterpret_cast<void*>(1)), 1u);
  EXPECT_EQ(
      FromPointerCast<unsigned short>(reinterpret_cast<void*>(
          static_cast<uintptr_t>(std::numeric_limits<unsigned short>::max()))),
      std::numeric_limits<unsigned short>::max());
  EXPECT_EQ(FromPointerCast<unsigned short>(reinterpret_cast<void*>(
                static_cast<uintptr_t>(std::numeric_limits<short>::max()))),
            static_cast<unsigned short>(std::numeric_limits<short>::max()));
}

TEST(FromPointerCastDeathTest, ToNarrowInteger) {
  if (sizeof(int) < sizeof(void*)) {
    EXPECT_DEATH_CHECK(
        FromPointerCast<int>(reinterpret_cast<void*>(static_cast<uintptr_t>(
            std::numeric_limits<unsigned int>::max() + 1ull))),
        "");
    EXPECT_DEATH_CHECK(
        FromPointerCast<unsigned int>(
            reinterpret_cast<void*>(static_cast<uintptr_t>(
                std::numeric_limits<unsigned int>::max() + 1ull))),
        "");
  }

  // int and unsigned int may not be narrower than a pointer, so also test short
  // and unsigned short.

  EXPECT_DEATH_CHECK(
      FromPointerCast<short>(reinterpret_cast<void*>(static_cast<uintptr_t>(
          std::numeric_limits<unsigned short>::max() + 1u))),
      "");
  EXPECT_DEATH_CHECK(FromPointerCast<unsigned short>(
                         reinterpret_cast<void*>(static_cast<uintptr_t>(
                             std::numeric_limits<unsigned short>::max() + 1u))),
                     "");
}

}  // namespace
}  // namespace test
}  // namespace crashpad
