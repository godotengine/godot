
#include "FuzzerInternal.h"
#include "gtest/gtest.h"
#include <set>

// For now, have LLVMFuzzerTestOneInput just to make it link.
// Later we may want to make unittests that actually call LLVMFuzzerTestOneInput.
extern "C" void LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  abort();
}

TEST(Fuzzer, CrossOver) {
  using namespace fuzzer;
  Unit A({0, 1, 2}), B({5, 6, 7});
  Unit C;
  Unit Expected[] = {
       { 0 },
       { 0, 1 },
       { 0, 5 },
       { 0, 1, 2 },
       { 0, 1, 5 },
       { 0, 5, 1 },
       { 0, 5, 6 },
       { 0, 1, 2, 5 },
       { 0, 1, 5, 2 },
       { 0, 1, 5, 6 },
       { 0, 5, 1, 2 },
       { 0, 5, 1, 6 },
       { 0, 5, 6, 1 },
       { 0, 5, 6, 7 },
       { 0, 1, 2, 5, 6 },
       { 0, 1, 5, 2, 6 },
       { 0, 1, 5, 6, 2 },
       { 0, 1, 5, 6, 7 },
       { 0, 5, 1, 2, 6 },
       { 0, 5, 1, 6, 2 },
       { 0, 5, 1, 6, 7 },
       { 0, 5, 6, 1, 2 },
       { 0, 5, 6, 1, 7 },
       { 0, 5, 6, 7, 1 },
       { 0, 1, 2, 5, 6, 7 },
       { 0, 1, 5, 2, 6, 7 },
       { 0, 1, 5, 6, 2, 7 },
       { 0, 1, 5, 6, 7, 2 },
       { 0, 5, 1, 2, 6, 7 },
       { 0, 5, 1, 6, 2, 7 },
       { 0, 5, 1, 6, 7, 2 },
       { 0, 5, 6, 1, 2, 7 },
       { 0, 5, 6, 1, 7, 2 },
       { 0, 5, 6, 7, 1, 2 }
  };
  for (size_t Len = 1; Len < 8; Len++) {
    std::set<Unit> FoundUnits, ExpectedUnitsWitThisLength;
    for (int Iter = 0; Iter < 3000; Iter++) {
      C.resize(Len);
      size_t NewSize = CrossOver(A.data(), A.size(), B.data(), B.size(),
                                 C.data(), C.size());
      C.resize(NewSize);
      FoundUnits.insert(C);
    }
    for (const Unit &U : Expected)
      if (U.size() <= Len)
        ExpectedUnitsWitThisLength.insert(U);
    EXPECT_EQ(ExpectedUnitsWitThisLength, FoundUnits);
  }
}

TEST(Fuzzer, Hash) {
  uint8_t A[] = {'a', 'b', 'c'};
  fuzzer::Unit U(A, A + sizeof(A));
  EXPECT_EQ("a9993e364706816aba3e25717850c26c9cd0d89d", fuzzer::Hash(U));
  U.push_back('d');
  EXPECT_EQ("81fe8bfe87576c3ecb22426f8e57847382917acf", fuzzer::Hash(U));
}
