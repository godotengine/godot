// Copyright (c) 2017 Google Inc.
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

#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "source/util/ilist.h"

namespace spvtools {
namespace utils {
namespace {

using ::testing::ElementsAre;
using IListTest = ::testing::Test;

class TestNode : public IntrusiveNodeBase<TestNode> {
 public:
  TestNode() : IntrusiveNodeBase<TestNode>() {}
  int data_;
};

class TestList : public IntrusiveList<TestNode> {
 public:
  TestList() = default;
  TestList(TestList&& that) : IntrusiveList<TestNode>(std::move(that)) {}
  TestList& operator=(TestList&& that) {
    static_cast<IntrusiveList<TestNode>&>(*this) =
        static_cast<IntrusiveList<TestNode>&&>(that);
    return *this;
  }
};

// This test checks the push_back method, as well as using an iterator to
// traverse the list from begin() to end().  This implicitly test the
// PreviousNode and NextNode functions.
TEST(IListTest, PushBack) {
  TestNode nodes[10];
  TestList list;
  for (int i = 0; i < 10; i++) {
    nodes[i].data_ = i;
    list.push_back(&nodes[i]);
  }

  std::vector<int> output;
  for (auto& i : list) output.push_back(i.data_);

  EXPECT_THAT(output, ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
}

// Returns a list containing the values 0 to n-1 using the first n elements of
// nodes to build the list.
TestList BuildList(TestNode nodes[], int n) {
  TestList list;
  for (int i = 0; i < n; i++) {
    nodes[i].data_ = i;
    list.push_back(&nodes[i]);
  }
  return list;
}

// Test decrementing begin()
TEST(IListTest, DecrementingBegin) {
  TestNode nodes[10];
  TestList list = BuildList(nodes, 10);
  EXPECT_EQ(--list.begin(), list.end());
}

// Test incrementing end()
TEST(IListTest, IncrementingEnd1) {
  TestNode nodes[10];
  TestList list = BuildList(nodes, 10);
  EXPECT_EQ((++list.end())->data_, 0);
}

// Test incrementing end() should equal begin()
TEST(IListTest, IncrementingEnd2) {
  TestNode nodes[10];
  TestList list = BuildList(nodes, 10);
  EXPECT_EQ(++list.end(), list.begin());
}

// Test decrementing end()
TEST(IListTest, DecrementingEnd) {
  TestNode nodes[10];
  TestList list = BuildList(nodes, 10);
  EXPECT_EQ((--list.end())->data_, 9);
}

// Test the move constructor for the list class.
TEST(IListTest, MoveConstructor) {
  TestNode nodes[10];
  TestList list = BuildList(nodes, 10);
  std::vector<int> output;
  for (auto& i : list) output.push_back(i.data_);

  EXPECT_THAT(output, ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
}

// Using a const list so we can test the const_iterator.
TEST(IListTest, ConstIterator) {
  TestNode nodes[10];
  const TestList list = BuildList(nodes, 10);
  std::vector<int> output;
  for (auto& i : list) output.push_back(i.data_);

  EXPECT_THAT(output, ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
}

// Uses the move assignement instead of the move constructor.
TEST(IListTest, MoveAssignment) {
  TestNode nodes[10];
  TestList list;
  list = BuildList(nodes, 10);
  std::vector<int> output;
  for (auto& i : list) output.push_back(i.data_);

  EXPECT_THAT(output, ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
}

// Test inserting a new element at the end of a list using the IntrusiveNodeBase
// "InsertAfter" function.
TEST(IListTest, InsertAfter1) {
  TestNode nodes[10];
  TestList list = BuildList(nodes, 5);

  nodes[5].data_ = 5;
  nodes[5].InsertAfter(&nodes[4]);

  std::vector<int> output;
  for (auto& i : list) output.push_back(i.data_);

  EXPECT_THAT(output, ElementsAre(0, 1, 2, 3, 4, 5));
}

// Test inserting a new element in the middle of a list using the
// IntrusiveNodeBase "InsertAfter" function.
TEST(IListTest, InsertAfter2) {
  TestNode nodes[10];
  TestList list = BuildList(nodes, 5);

  nodes[5].data_ = 5;
  nodes[5].InsertAfter(&nodes[2]);

  std::vector<int> output;
  for (auto& i : list) output.push_back(i.data_);

  EXPECT_THAT(output, ElementsAre(0, 1, 2, 5, 3, 4));
}

// Test moving an element already in the list in the middle of a list using the
// IntrusiveNodeBase "InsertAfter" function.
TEST(IListTest, MoveUsingInsertAfter1) {
  TestNode nodes[10];
  TestList list = BuildList(nodes, 6);

  nodes[5].InsertAfter(&nodes[2]);

  std::vector<int> output;
  for (auto& i : list) output.push_back(i.data_);

  EXPECT_THAT(output, ElementsAre(0, 1, 2, 5, 3, 4));
}

// Move the element at the start of the list into the middle.
TEST(IListTest, MoveUsingInsertAfter2) {
  TestNode nodes[10];
  TestList list = BuildList(nodes, 6);

  nodes[0].InsertAfter(&nodes[2]);

  std::vector<int> output;
  for (auto& i : list) output.push_back(i.data_);

  EXPECT_THAT(output, ElementsAre(1, 2, 0, 3, 4, 5));
}

// Move an element in the middle of the list to the end.
TEST(IListTest, MoveUsingInsertAfter3) {
  TestNode nodes[10];
  TestList list = BuildList(nodes, 6);

  nodes[2].InsertAfter(&nodes[5]);

  std::vector<int> output;
  for (auto& i : list) output.push_back(i.data_);

  EXPECT_THAT(output, ElementsAre(0, 1, 3, 4, 5, 2));
}

// Removing an element from the middle of a list.
TEST(IListTest, Remove1) {
  TestNode nodes[10];
  TestList list = BuildList(nodes, 6);

  nodes[2].RemoveFromList();

  std::vector<int> output;
  for (auto& i : list) output.push_back(i.data_);

  EXPECT_THAT(output, ElementsAre(0, 1, 3, 4, 5));
}

// Removing an element from the beginning of the list.
TEST(IListTest, Remove2) {
  TestNode nodes[10];
  TestList list = BuildList(nodes, 6);

  nodes[0].RemoveFromList();

  std::vector<int> output;
  for (auto& i : list) output.push_back(i.data_);

  EXPECT_THAT(output, ElementsAre(1, 2, 3, 4, 5));
}

// Removing the last element of a list.
TEST(IListTest, Remove3) {
  TestNode nodes[10];
  TestList list = BuildList(nodes, 6);

  nodes[5].RemoveFromList();

  std::vector<int> output;
  for (auto& i : list) output.push_back(i.data_);

  EXPECT_THAT(output, ElementsAre(0, 1, 2, 3, 4));
}

// Test that operator== and operator!= work properly for the iterator class.
TEST(IListTest, IteratorEqual) {
  TestNode nodes[10];
  TestList list = BuildList(nodes, 6);

  std::vector<int> output;
  for (auto i = list.begin(); i != list.end(); ++i)
    for (auto j = list.begin(); j != list.end(); ++j)
      if (i == j) output.push_back(i->data_);

  EXPECT_THAT(output, ElementsAre(0, 1, 2, 3, 4, 5));
}

// Test MoveBefore.  Moving into middle of a list.
TEST(IListTest, MoveBefore1) {
  TestNode nodes[10];
  TestList list1 = BuildList(nodes, 6);
  TestList list2 = BuildList(nodes + 6, 3);

  TestList::iterator insertion_point = list1.begin();
  ++insertion_point;
  insertion_point.MoveBefore(&list2);

  std::vector<int> output;
  for (auto i = list1.begin(); i != list1.end(); ++i) {
    output.push_back(i->data_);
  }

  EXPECT_THAT(output, ElementsAre(0, 0, 1, 2, 1, 2, 3, 4, 5));
}

// Test MoveBefore.  Moving to the start of a list.
TEST(IListTest, MoveBefore2) {
  TestNode nodes[10];
  TestList list1 = BuildList(nodes, 6);
  TestList list2 = BuildList(nodes + 6, 3);

  TestList::iterator insertion_point = list1.begin();
  insertion_point.MoveBefore(&list2);

  std::vector<int> output;
  for (auto i = list1.begin(); i != list1.end(); ++i) {
    output.push_back(i->data_);
  }

  EXPECT_THAT(output, ElementsAre(0, 1, 2, 0, 1, 2, 3, 4, 5));
}

// Test MoveBefore.  Moving to the end of a list.
TEST(IListTest, MoveBefore3) {
  TestNode nodes[10];
  TestList list1 = BuildList(nodes, 6);
  TestList list2 = BuildList(nodes + 6, 3);

  TestList::iterator insertion_point = list1.end();
  insertion_point.MoveBefore(&list2);

  std::vector<int> output;
  for (auto i = list1.begin(); i != list1.end(); ++i) {
    output.push_back(i->data_);
  }

  EXPECT_THAT(output, ElementsAre(0, 1, 2, 3, 4, 5, 0, 1, 2));
}

// Test MoveBefore.  Moving an empty list.
TEST(IListTest, MoveBefore4) {
  TestNode nodes[10];
  TestList list1 = BuildList(nodes, 6);
  TestList list2;

  TestList::iterator insertion_point = list1.end();
  insertion_point.MoveBefore(&list2);

  std::vector<int> output;
  for (auto i = list1.begin(); i != list1.end(); ++i) {
    output.push_back(i->data_);
  }

  EXPECT_THAT(output, ElementsAre(0, 1, 2, 3, 4, 5));
}

}  // namespace
}  // namespace utils
}  // namespace spvtools
