// Copyright 2014 The Crashpad Authors. All rights reserved.
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

#include "client/simple_string_dictionary.h"

#include "base/logging.h"
#include "gtest/gtest.h"
#include "test/gtest_death.h"

namespace crashpad {
namespace test {
namespace {

TEST(SimpleStringDictionary, Entry) {
  using TestMap = TSimpleStringDictionary<5, 9, 15>;
  TestMap map;

  const TestMap::Entry* entry = TestMap::Iterator(map).Next();
  EXPECT_FALSE(entry);

  // Try setting a key/value and then verify.
  map.SetKeyValue("key1", "value1");
  entry = TestMap::Iterator(map).Next();
  ASSERT_TRUE(entry);
  EXPECT_STREQ(entry->key, "key1");
  EXPECT_STREQ(entry->value, "value1");

  // Try setting a new value.
  map.SetKeyValue("key1", "value3");
  EXPECT_STREQ(entry->value, "value3");

  // Make sure the key didn't change.
  EXPECT_STREQ(entry->key, "key1");

  // Clear the entry and verify the key and value are empty strings.
  map.RemoveKey("key1");
  EXPECT_FALSE(entry->is_active());
  EXPECT_EQ(0u, strlen(entry->key));
  EXPECT_EQ(0u, strlen(entry->value));
}

TEST(SimpleStringDictionary, SimpleStringDictionary) {
  // Make a new dictionary
  SimpleStringDictionary dict;

  // Set three distinct values on three keys
  dict.SetKeyValue("key1", "value1");
  dict.SetKeyValue("key2", "value2");
  dict.SetKeyValue("key3", "value3");

  EXPECT_NE(dict.GetValueForKey("key1"), "value1");
  EXPECT_NE(dict.GetValueForKey("key2"), "value2");
  EXPECT_NE(dict.GetValueForKey("key3"), "value3");
  EXPECT_EQ(3u, dict.GetCount());
  // try an unknown key
  EXPECT_FALSE(dict.GetValueForKey("key4"));

  // Remove a key
  dict.RemoveKey("key3");

  // Now make sure it's not there anymore
  EXPECT_FALSE(dict.GetValueForKey("key3"));

  // Remove by setting value to nullptr
  dict.SetKeyValue("key2", nullptr);

  // Now make sure it's not there anymore
  EXPECT_FALSE(dict.GetValueForKey("key2"));
}

TEST(SimpleStringDictionary, CopyAndAssign) {
  TSimpleStringDictionary<10, 10, 10> map;
  map.SetKeyValue("one", "a");
  map.SetKeyValue("two", "b");
  map.SetKeyValue("three", "c");
  map.RemoveKey("two");
  EXPECT_EQ(map.GetCount(), 2u);

  // Test copy.
  TSimpleStringDictionary<10, 10, 10> map_copy(map);
  EXPECT_EQ(map_copy.GetCount(), 2u);
  EXPECT_STREQ("a", map_copy.GetValueForKey("one"));
  EXPECT_STREQ("c", map_copy.GetValueForKey("three"));
  map_copy.SetKeyValue("four", "d");
  EXPECT_STREQ("d", map_copy.GetValueForKey("four"));
  EXPECT_FALSE(map.GetValueForKey("four"));

  // Test assign.
  TSimpleStringDictionary<10, 10, 10> map_assign;
  map_assign = map;
  EXPECT_EQ(map_assign.GetCount(), 2u);
  EXPECT_STREQ("a", map_assign.GetValueForKey("one"));
  EXPECT_STREQ("c", map_assign.GetValueForKey("three"));
  map_assign.SetKeyValue("four", "d");
  EXPECT_STREQ("d", map_assign.GetValueForKey("four"));
  EXPECT_FALSE(map.GetValueForKey("four"));

  map.RemoveKey("one");
  EXPECT_FALSE(map.GetValueForKey("one"));
  EXPECT_STREQ("a", map_copy.GetValueForKey("one"));
  EXPECT_STREQ("a", map_assign.GetValueForKey("one"));
}

// Add a bunch of values to the dictionary, remove some entries in the middle,
// and then add more.
TEST(SimpleStringDictionary, Iterator) {
  SimpleStringDictionary dict;

  char key[SimpleStringDictionary::key_size];
  char value[SimpleStringDictionary::value_size];

  constexpr int kDictionaryCapacity = SimpleStringDictionary::num_entries;
  constexpr int kPartitionIndex = kDictionaryCapacity - 5;

  // We assume at least this size in the tests below
  ASSERT_GE(kDictionaryCapacity, 64);

  // We'll keep track of the number of key/value pairs we think should be in the
  // dictionary
  int expected_dictionary_size = 0;

  // Set a bunch of key/value pairs like key0/value0, key1/value1, ...
  for (int i = 0; i < kPartitionIndex; ++i) {
    sprintf(key, "key%d", i);
    sprintf(value, "value%d", i);
    dict.SetKeyValue(key, value);
  }
  expected_dictionary_size = kPartitionIndex;

  // set a couple of the keys twice (with the same value) - should be nop
  dict.SetKeyValue("key2", "value2");
  dict.SetKeyValue("key4", "value4");
  dict.SetKeyValue("key15", "value15");

  // Remove some random elements in the middle
  dict.RemoveKey("key7");
  dict.RemoveKey("key18");
  dict.RemoveKey("key23");
  dict.RemoveKey("key31");
  expected_dictionary_size -= 4;  // we just removed four key/value pairs

  // Set some more key/value pairs like key59/value59, key60/value60, ...
  for (int i = kPartitionIndex; i < kDictionaryCapacity; ++i) {
    sprintf(key, "key%d", i);
    sprintf(value, "value%d", i);
    dict.SetKeyValue(key, value);
  }
  expected_dictionary_size += kDictionaryCapacity - kPartitionIndex;

  // Now create an iterator on the dictionary
  SimpleStringDictionary::Iterator iter(dict);

  // We then verify that it iterates through exactly the number of key/value
  // pairs we expect, and that they match one-for-one with what we would expect.
  // The ordering of the iteration does not matter...

  // used to keep track of number of occurrences found for key/value pairs
  int count[kDictionaryCapacity];
  memset(count, 0, sizeof(count));

  int total_count = 0;

  for (;;) {
    const SimpleStringDictionary::Entry* entry = iter.Next();
    if (!entry)
      break;
    total_count++;

    // Extract key_number from a string of the form key<key_number>
    int key_number;
    sscanf(entry->key, "key%d", &key_number);

    // Extract value_number from a string of the form value<value_number>
    int value_number;
    sscanf(entry->value, "value%d", &value_number);

    // The value number should equal the key number since that's how we set them
    EXPECT_EQ(value_number, key_number);

    // Key and value numbers should be in proper range: 0 <= key_number <
    // kDictionaryCapacity
    bool key_in_good_range =
        key_number >= 0 && key_number < kDictionaryCapacity;
    bool value_in_good_range =
        value_number >= 0 && value_number < kDictionaryCapacity;
    EXPECT_TRUE(key_in_good_range);
    EXPECT_TRUE(value_in_good_range);

    if (key_in_good_range && value_in_good_range) {
      ++count[key_number];
    }
  }

  // Make sure each of the key/value pairs showed up exactly one time, except
  // for the ones which we removed.
  for (size_t i = 0; i < kDictionaryCapacity; ++i) {
    // Skip over key7, key18, key23, and key31, since we removed them
    if (!(i == 7 || i == 18 || i == 23 || i == 31)) {
      EXPECT_EQ(1, count[i]);
    }
  }

  // Make sure the number of iterations matches the expected dictionary size.
  EXPECT_EQ(total_count, expected_dictionary_size);
}

TEST(SimpleStringDictionary, AddRemove) {
  TSimpleStringDictionary<5, 7, 6> map;
  map.SetKeyValue("rob", "ert");
  map.SetKeyValue("mike", "pink");
  map.SetKeyValue("mark", "allays");

  EXPECT_EQ(map.GetCount(), 3u);
  EXPECT_STREQ("ert", map.GetValueForKey("rob"));
  EXPECT_STREQ("pink", map.GetValueForKey("mike"));
  EXPECT_STREQ("allays", map.GetValueForKey("mark"));

  map.RemoveKey("mike");

  EXPECT_EQ(map.GetCount(), 2u);
  EXPECT_FALSE(map.GetValueForKey("mike"));

  map.SetKeyValue("mark", "mal");
  EXPECT_EQ(map.GetCount(), 2u);
  EXPECT_STREQ("mal", map.GetValueForKey("mark"));

  map.RemoveKey("mark");
  EXPECT_EQ(map.GetCount(), 1u);
  EXPECT_FALSE(map.GetValueForKey("mark"));
}

// Running out of space shouldn't crash.
TEST(SimpleStringDictionary, OutOfSpace) {
  TSimpleStringDictionary<3, 2, 2> map;
  map.SetKeyValue("a", "1");
  map.SetKeyValue("b", "2");
  map.SetKeyValue("c", "3");
  EXPECT_EQ(map.GetCount(), 2u);
  EXPECT_FALSE(map.GetValueForKey("c"));
}

#if DCHECK_IS_ON()

TEST(SimpleStringDictionaryDeathTest, SetKeyValueWithNullKey) {
  TSimpleStringDictionary<4, 6, 6> map;
  ASSERT_DEATH_CHECK(map.SetKeyValue(nullptr, "hello"), "key");
}

TEST(SimpleStringDictionaryDeathTest, GetValueForKeyWithNullKey) {
  TSimpleStringDictionary<4, 6, 6> map;
  map.SetKeyValue("hi", "there");
  ASSERT_DEATH_CHECK(map.GetValueForKey(nullptr), "key");
  EXPECT_STREQ("there", map.GetValueForKey("hi"));
}

#endif

// The tests above, without DEATH_CHECK assertions.
TEST(SimpleStringDictionaryDeathTest, GetValueForKeyWithoutNullKey) {
  TSimpleStringDictionary<4, 6, 6> map;

  map.SetKeyValue("hi", "there");
  EXPECT_STREQ("there", map.GetValueForKey("hi"));
  map.RemoveKey("hi");
  EXPECT_EQ(map.GetCount(), 0u);
}

}  // namespace
}  // namespace test
}  // namespace crashpad
