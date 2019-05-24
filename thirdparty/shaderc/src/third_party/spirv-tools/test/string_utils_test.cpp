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

#include <string>

#include "gtest/gtest.h"
#include "source/util/string_utils.h"
#include "spirv-tools/libspirv.h"

namespace spvtools {
namespace utils {
namespace {

TEST(ToString, Int) {
  EXPECT_EQ("0", ToString(0));
  EXPECT_EQ("1000", ToString(1000));
  EXPECT_EQ("-1", ToString(-1));
  EXPECT_EQ("0", ToString(0LL));
  EXPECT_EQ("1000", ToString(1000LL));
  EXPECT_EQ("-1", ToString(-1LL));
}

TEST(ToString, Uint) {
  EXPECT_EQ("0", ToString(0U));
  EXPECT_EQ("1000", ToString(1000U));
  EXPECT_EQ("0", ToString(0ULL));
  EXPECT_EQ("1000", ToString(1000ULL));
}

TEST(ToString, Float) {
  EXPECT_EQ("0", ToString(0.f));
  EXPECT_EQ("1000", ToString(1000.f));
  EXPECT_EQ("-1.5", ToString(-1.5f));
}

TEST(ToString, Double) {
  EXPECT_EQ("0", ToString(0.));
  EXPECT_EQ("1000", ToString(1000.));
  EXPECT_EQ("-1.5", ToString(-1.5));
}

TEST(CardinalToOrdinal, Test) {
  EXPECT_EQ("1st", CardinalToOrdinal(1));
  EXPECT_EQ("2nd", CardinalToOrdinal(2));
  EXPECT_EQ("3rd", CardinalToOrdinal(3));
  EXPECT_EQ("4th", CardinalToOrdinal(4));
  EXPECT_EQ("5th", CardinalToOrdinal(5));
  EXPECT_EQ("6th", CardinalToOrdinal(6));
  EXPECT_EQ("7th", CardinalToOrdinal(7));
  EXPECT_EQ("8th", CardinalToOrdinal(8));
  EXPECT_EQ("9th", CardinalToOrdinal(9));
  EXPECT_EQ("10th", CardinalToOrdinal(10));
  EXPECT_EQ("11th", CardinalToOrdinal(11));
  EXPECT_EQ("12th", CardinalToOrdinal(12));
  EXPECT_EQ("13th", CardinalToOrdinal(13));
  EXPECT_EQ("14th", CardinalToOrdinal(14));
  EXPECT_EQ("15th", CardinalToOrdinal(15));
  EXPECT_EQ("16th", CardinalToOrdinal(16));
  EXPECT_EQ("17th", CardinalToOrdinal(17));
  EXPECT_EQ("18th", CardinalToOrdinal(18));
  EXPECT_EQ("19th", CardinalToOrdinal(19));
  EXPECT_EQ("20th", CardinalToOrdinal(20));
  EXPECT_EQ("21st", CardinalToOrdinal(21));
  EXPECT_EQ("22nd", CardinalToOrdinal(22));
  EXPECT_EQ("23rd", CardinalToOrdinal(23));
  EXPECT_EQ("24th", CardinalToOrdinal(24));
  EXPECT_EQ("25th", CardinalToOrdinal(25));
  EXPECT_EQ("26th", CardinalToOrdinal(26));
  EXPECT_EQ("27th", CardinalToOrdinal(27));
  EXPECT_EQ("28th", CardinalToOrdinal(28));
  EXPECT_EQ("29th", CardinalToOrdinal(29));
  EXPECT_EQ("30th", CardinalToOrdinal(30));
  EXPECT_EQ("31st", CardinalToOrdinal(31));
  EXPECT_EQ("32nd", CardinalToOrdinal(32));
  EXPECT_EQ("33rd", CardinalToOrdinal(33));
  EXPECT_EQ("34th", CardinalToOrdinal(34));
  EXPECT_EQ("35th", CardinalToOrdinal(35));
  EXPECT_EQ("100th", CardinalToOrdinal(100));
  EXPECT_EQ("101st", CardinalToOrdinal(101));
  EXPECT_EQ("102nd", CardinalToOrdinal(102));
  EXPECT_EQ("103rd", CardinalToOrdinal(103));
  EXPECT_EQ("104th", CardinalToOrdinal(104));
  EXPECT_EQ("105th", CardinalToOrdinal(105));
  EXPECT_EQ("106th", CardinalToOrdinal(106));
  EXPECT_EQ("107th", CardinalToOrdinal(107));
  EXPECT_EQ("108th", CardinalToOrdinal(108));
  EXPECT_EQ("109th", CardinalToOrdinal(109));
  EXPECT_EQ("110th", CardinalToOrdinal(110));
  EXPECT_EQ("111th", CardinalToOrdinal(111));
  EXPECT_EQ("112th", CardinalToOrdinal(112));
  EXPECT_EQ("113th", CardinalToOrdinal(113));
  EXPECT_EQ("114th", CardinalToOrdinal(114));
  EXPECT_EQ("115th", CardinalToOrdinal(115));
  EXPECT_EQ("116th", CardinalToOrdinal(116));
  EXPECT_EQ("117th", CardinalToOrdinal(117));
  EXPECT_EQ("118th", CardinalToOrdinal(118));
  EXPECT_EQ("119th", CardinalToOrdinal(119));
  EXPECT_EQ("120th", CardinalToOrdinal(120));
  EXPECT_EQ("121st", CardinalToOrdinal(121));
  EXPECT_EQ("122nd", CardinalToOrdinal(122));
  EXPECT_EQ("123rd", CardinalToOrdinal(123));
  EXPECT_EQ("124th", CardinalToOrdinal(124));
  EXPECT_EQ("125th", CardinalToOrdinal(125));
  EXPECT_EQ("126th", CardinalToOrdinal(126));
  EXPECT_EQ("127th", CardinalToOrdinal(127));
  EXPECT_EQ("128th", CardinalToOrdinal(128));
  EXPECT_EQ("129th", CardinalToOrdinal(129));
  EXPECT_EQ("130th", CardinalToOrdinal(130));
  EXPECT_EQ("131st", CardinalToOrdinal(131));
  EXPECT_EQ("132nd", CardinalToOrdinal(132));
  EXPECT_EQ("133rd", CardinalToOrdinal(133));
  EXPECT_EQ("134th", CardinalToOrdinal(134));
  EXPECT_EQ("135th", CardinalToOrdinal(135));
  EXPECT_EQ("1000th", CardinalToOrdinal(1000));
  EXPECT_EQ("1001st", CardinalToOrdinal(1001));
  EXPECT_EQ("1002nd", CardinalToOrdinal(1002));
  EXPECT_EQ("1003rd", CardinalToOrdinal(1003));
  EXPECT_EQ("1004th", CardinalToOrdinal(1004));
  EXPECT_EQ("1005th", CardinalToOrdinal(1005));
  EXPECT_EQ("1006th", CardinalToOrdinal(1006));
  EXPECT_EQ("1007th", CardinalToOrdinal(1007));
  EXPECT_EQ("1008th", CardinalToOrdinal(1008));
  EXPECT_EQ("1009th", CardinalToOrdinal(1009));
  EXPECT_EQ("1010th", CardinalToOrdinal(1010));
  EXPECT_EQ("1011th", CardinalToOrdinal(1011));
  EXPECT_EQ("1012th", CardinalToOrdinal(1012));
  EXPECT_EQ("1013th", CardinalToOrdinal(1013));
  EXPECT_EQ("1014th", CardinalToOrdinal(1014));
  EXPECT_EQ("1015th", CardinalToOrdinal(1015));
  EXPECT_EQ("1016th", CardinalToOrdinal(1016));
  EXPECT_EQ("1017th", CardinalToOrdinal(1017));
  EXPECT_EQ("1018th", CardinalToOrdinal(1018));
  EXPECT_EQ("1019th", CardinalToOrdinal(1019));
  EXPECT_EQ("1020th", CardinalToOrdinal(1020));
  EXPECT_EQ("1021st", CardinalToOrdinal(1021));
  EXPECT_EQ("1022nd", CardinalToOrdinal(1022));
  EXPECT_EQ("1023rd", CardinalToOrdinal(1023));
  EXPECT_EQ("1024th", CardinalToOrdinal(1024));
  EXPECT_EQ("1025th", CardinalToOrdinal(1025));
  EXPECT_EQ("1026th", CardinalToOrdinal(1026));
  EXPECT_EQ("1027th", CardinalToOrdinal(1027));
  EXPECT_EQ("1028th", CardinalToOrdinal(1028));
  EXPECT_EQ("1029th", CardinalToOrdinal(1029));
  EXPECT_EQ("1030th", CardinalToOrdinal(1030));
  EXPECT_EQ("1031st", CardinalToOrdinal(1031));
  EXPECT_EQ("1032nd", CardinalToOrdinal(1032));
  EXPECT_EQ("1033rd", CardinalToOrdinal(1033));
  EXPECT_EQ("1034th", CardinalToOrdinal(1034));
  EXPECT_EQ("1035th", CardinalToOrdinal(1035));
  EXPECT_EQ("1200th", CardinalToOrdinal(1200));
  EXPECT_EQ("1201st", CardinalToOrdinal(1201));
  EXPECT_EQ("1202nd", CardinalToOrdinal(1202));
  EXPECT_EQ("1203rd", CardinalToOrdinal(1203));
  EXPECT_EQ("1204th", CardinalToOrdinal(1204));
  EXPECT_EQ("1205th", CardinalToOrdinal(1205));
  EXPECT_EQ("1206th", CardinalToOrdinal(1206));
  EXPECT_EQ("1207th", CardinalToOrdinal(1207));
  EXPECT_EQ("1208th", CardinalToOrdinal(1208));
  EXPECT_EQ("1209th", CardinalToOrdinal(1209));
  EXPECT_EQ("1210th", CardinalToOrdinal(1210));
  EXPECT_EQ("1211th", CardinalToOrdinal(1211));
  EXPECT_EQ("1212th", CardinalToOrdinal(1212));
  EXPECT_EQ("1213th", CardinalToOrdinal(1213));
  EXPECT_EQ("1214th", CardinalToOrdinal(1214));
  EXPECT_EQ("1215th", CardinalToOrdinal(1215));
  EXPECT_EQ("1216th", CardinalToOrdinal(1216));
  EXPECT_EQ("1217th", CardinalToOrdinal(1217));
  EXPECT_EQ("1218th", CardinalToOrdinal(1218));
  EXPECT_EQ("1219th", CardinalToOrdinal(1219));
  EXPECT_EQ("1220th", CardinalToOrdinal(1220));
  EXPECT_EQ("1221st", CardinalToOrdinal(1221));
  EXPECT_EQ("1222nd", CardinalToOrdinal(1222));
  EXPECT_EQ("1223rd", CardinalToOrdinal(1223));
  EXPECT_EQ("1224th", CardinalToOrdinal(1224));
  EXPECT_EQ("1225th", CardinalToOrdinal(1225));
}

}  // namespace
}  // namespace utils
}  // namespace spvtools
