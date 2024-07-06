// Copyright (c) 2020 Google LLC
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

#ifndef SOURCE_FUZZ_OVERFLOW_ID_SOURCE_H_
#define SOURCE_FUZZ_OVERFLOW_ID_SOURCE_H_

#include <cstdint>
#include <unordered_set>

namespace spvtools {
namespace fuzz {

// An implementation of this interface can be used to provide fresh ids on
// demand when applying a transformation.
//
// During fuzzing this should never be required: a fuzzer pass should determine
// all the fresh ids it requires to apply a transformation.
//
// However, during shrinking we can have the situation where, after removing
// an early transformation, a later transformation needs more ids.
//
// As an example, suppose a SPIR-V function originally has this form:
//
// main() {
//   stmt1;
//   stmt2;
//   stmt3;
//   stmt4;
// }
//
// Now suppose two *outlining* transformations are applied.  The first
// transformation, T1, outlines "stmt1; stmt2;" into a function foo, giving us:
//
// foo() {
//   stmt1;
//   stmt2;
// }
//
// main() {
//   foo();
//   stmt3;
//   stmt4;
// }
//
// The second transformation, T2, outlines "foo(); stmt3;" from main into a
// function bar, giving us:
//
// foo() {
//   stmt1;
//   stmt2;
// }
//
// bar() {
//   foo();
//   stmt3;
// }
//
// main() {
//   bar();
//   stmt4;
// }
//
// Suppose that T2 used a set of fresh ids, FRESH, in order to perform its
// outlining.
//
// Now suppose that during shrinking we remove T1, but still want to apply T2.
// The fresh ids used by T2 - FRESH - are sufficient to outline "foo(); stmt3;".
// However, because we did not apply T1, "foo();" does not exist and instead the
// task of T2 is to outline "stmt1; stmt2; stmt3;".  The set FRESH contains
// *some* of the fresh ids required to do this (those for "stmt3;"), but not all
// of them (those for "stmt1; stmt2;" are missing).
//
// A source of overflow ids can be used to allow the shrinker to proceed
// nevertheless.
//
// It is desirable to use overflow ids only when needed.  In our worked example,
// T2 should still use the ids from FRESH when handling "stmt3;", because later
// transformations might refer to those ids and will become inapplicable if
// overflow ids are used instead.
class OverflowIdSource {
 public:
  virtual ~OverflowIdSource();

  // Returns true if and only if this source is capable of providing overflow
  // ids.
  virtual bool HasOverflowIds() const = 0;

  // Precondition: HasOverflowIds() must hold.  Returns the next available
  // overflow id.
  virtual uint32_t GetNextOverflowId() = 0;

  // Returns the set of overflow ids from this source that have been previously
  // issued via calls to GetNextOverflowId().
  virtual const std::unordered_set<uint32_t>& GetIssuedOverflowIds() const = 0;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_OVERFLOW_ID_SOURCE_H_
