// Copyright 2015 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Image transform methods for lossless encoder.
//
// Authors: Vikas Arora (vikaas.arora@gmail.com)
//          Jyrki Alakuijala (jyrki@google.com)
//          Urvang Joshi (urvang@google.com)

#include "./dsp.h"

#include <math.h>
#include <stdlib.h>
#include "../dec/vp8li.h"
#include "../utils/endian_inl.h"
#include "./lossless.h"
#include "./yuv.h"

#define MAX_DIFF_COST (1e30f)

// lookup table for small values of log2(int)
const float kLog2Table[LOG_LOOKUP_IDX_MAX] = {
  0.0000000000000000f, 0.0000000000000000f,
  1.0000000000000000f, 1.5849625007211560f,
  2.0000000000000000f, 2.3219280948873621f,
  2.5849625007211560f, 2.8073549220576041f,
  3.0000000000000000f, 3.1699250014423121f,
  3.3219280948873621f, 3.4594316186372973f,
  3.5849625007211560f, 3.7004397181410921f,
  3.8073549220576041f, 3.9068905956085187f,
  4.0000000000000000f, 4.0874628412503390f,
  4.1699250014423121f, 4.2479275134435852f,
  4.3219280948873626f, 4.3923174227787606f,
  4.4594316186372973f, 4.5235619560570130f,
  4.5849625007211560f, 4.6438561897747243f,
  4.7004397181410917f, 4.7548875021634682f,
  4.8073549220576037f, 4.8579809951275718f,
  4.9068905956085187f, 4.9541963103868749f,
  5.0000000000000000f, 5.0443941193584533f,
  5.0874628412503390f, 5.1292830169449663f,
  5.1699250014423121f, 5.2094533656289501f,
  5.2479275134435852f, 5.2854022188622487f,
  5.3219280948873626f, 5.3575520046180837f,
  5.3923174227787606f, 5.4262647547020979f,
  5.4594316186372973f, 5.4918530963296747f,
  5.5235619560570130f, 5.5545888516776376f,
  5.5849625007211560f, 5.6147098441152083f,
  5.6438561897747243f, 5.6724253419714951f,
  5.7004397181410917f, 5.7279204545631987f,
  5.7548875021634682f, 5.7813597135246599f,
  5.8073549220576037f, 5.8328900141647412f,
  5.8579809951275718f, 5.8826430493618415f,
  5.9068905956085187f, 5.9307373375628866f,
  5.9541963103868749f, 5.9772799234999167f,
  6.0000000000000000f, 6.0223678130284543f,
  6.0443941193584533f, 6.0660891904577720f,
  6.0874628412503390f, 6.1085244567781691f,
  6.1292830169449663f, 6.1497471195046822f,
  6.1699250014423121f, 6.1898245588800175f,
  6.2094533656289501f, 6.2288186904958804f,
  6.2479275134435852f, 6.2667865406949010f,
  6.2854022188622487f, 6.3037807481771030f,
  6.3219280948873626f, 6.3398500028846243f,
  6.3575520046180837f, 6.3750394313469245f,
  6.3923174227787606f, 6.4093909361377017f,
  6.4262647547020979f, 6.4429434958487279f,
  6.4594316186372973f, 6.4757334309663976f,
  6.4918530963296747f, 6.5077946401986963f,
  6.5235619560570130f, 6.5391588111080309f,
  6.5545888516776376f, 6.5698556083309478f,
  6.5849625007211560f, 6.5999128421871278f,
  6.6147098441152083f, 6.6293566200796094f,
  6.6438561897747243f, 6.6582114827517946f,
  6.6724253419714951f, 6.6865005271832185f,
  6.7004397181410917f, 6.7142455176661224f,
  6.7279204545631987f, 6.7414669864011464f,
  6.7548875021634682f, 6.7681843247769259f,
  6.7813597135246599f, 6.7944158663501061f,
  6.8073549220576037f, 6.8201789624151878f,
  6.8328900141647412f, 6.8454900509443747f,
  6.8579809951275718f, 6.8703647195834047f,
  6.8826430493618415f, 6.8948177633079437f,
  6.9068905956085187f, 6.9188632372745946f,
  6.9307373375628866f, 6.9425145053392398f,
  6.9541963103868749f, 6.9657842846620869f,
  6.9772799234999167f, 6.9886846867721654f,
  7.0000000000000000f, 7.0112272554232539f,
  7.0223678130284543f, 7.0334230015374501f,
  7.0443941193584533f, 7.0552824355011898f,
  7.0660891904577720f, 7.0768155970508308f,
  7.0874628412503390f, 7.0980320829605263f,
  7.1085244567781691f, 7.1189410727235076f,
  7.1292830169449663f, 7.1395513523987936f,
  7.1497471195046822f, 7.1598713367783890f,
  7.1699250014423121f, 7.1799090900149344f,
  7.1898245588800175f, 7.1996723448363644f,
  7.2094533656289501f, 7.2191685204621611f,
  7.2288186904958804f, 7.2384047393250785f,
  7.2479275134435852f, 7.2573878426926521f,
  7.2667865406949010f, 7.2761244052742375f,
  7.2854022188622487f, 7.2946207488916270f,
  7.3037807481771030f, 7.3128829552843557f,
  7.3219280948873626f, 7.3309168781146167f,
  7.3398500028846243f, 7.3487281542310771f,
  7.3575520046180837f, 7.3663222142458160f,
  7.3750394313469245f, 7.3837042924740519f,
  7.3923174227787606f, 7.4008794362821843f,
  7.4093909361377017f, 7.4178525148858982f,
  7.4262647547020979f, 7.4346282276367245f,
  7.4429434958487279f, 7.4512111118323289f,
  7.4594316186372973f, 7.4676055500829976f,
  7.4757334309663976f, 7.4838157772642563f,
  7.4918530963296747f, 7.4998458870832056f,
  7.5077946401986963f, 7.5156998382840427f,
  7.5235619560570130f, 7.5313814605163118f,
  7.5391588111080309f, 7.5468944598876364f,
  7.5545888516776376f, 7.5622424242210728f,
  7.5698556083309478f, 7.5774288280357486f,
  7.5849625007211560f, 7.5924570372680806f,
  7.5999128421871278f, 7.6073303137496104f,
  7.6147098441152083f, 7.6220518194563764f,
  7.6293566200796094f, 7.6366246205436487f,
  7.6438561897747243f, 7.6510516911789281f,
  7.6582114827517946f, 7.6653359171851764f,
  7.6724253419714951f, 7.6794800995054464f,
  7.6865005271832185f, 7.6934869574993252f,
  7.7004397181410917f, 7.7073591320808825f,
  7.7142455176661224f, 7.7210991887071855f,
  7.7279204545631987f, 7.7347096202258383f,
  7.7414669864011464f, 7.7481928495894605f,
  7.7548875021634682f, 7.7615512324444795f,
  7.7681843247769259f, 7.7747870596011736f,
  7.7813597135246599f, 7.7879025593914317f,
  7.7944158663501061f, 7.8008998999203047f,
  7.8073549220576037f, 7.8137811912170374f,
  7.8201789624151878f, 7.8265484872909150f,
  7.8328900141647412f, 7.8392037880969436f,
  7.8454900509443747f, 7.8517490414160571f,
  7.8579809951275718f, 7.8641861446542797f,
  7.8703647195834047f, 7.8765169465649993f,
  7.8826430493618415f, 7.8887432488982591f,
  7.8948177633079437f, 7.9008668079807486f,
  7.9068905956085187f, 7.9128893362299619f,
  7.9188632372745946f, 7.9248125036057812f,
  7.9307373375628866f, 7.9366379390025709f,
  7.9425145053392398f, 7.9483672315846778f,
  7.9541963103868749f, 7.9600019320680805f,
  7.9657842846620869f, 7.9715435539507719f,
  7.9772799234999167f, 7.9829935746943103f,
  7.9886846867721654f, 7.9943534368588577f
};

const float kSLog2Table[LOG_LOOKUP_IDX_MAX] = {
  0.00000000f,    0.00000000f,  2.00000000f,   4.75488750f,
  8.00000000f,   11.60964047f,  15.50977500f,  19.65148445f,
  24.00000000f,  28.52932501f,  33.21928095f,  38.05374781f,
  43.01955001f,  48.10571634f,  53.30296891f,  58.60335893f,
  64.00000000f,  69.48686830f,  75.05865003f,  80.71062276f,
  86.43856190f,  92.23866588f,  98.10749561f,  104.04192499f,
  110.03910002f, 116.09640474f, 122.21143267f, 128.38196256f,
  134.60593782f, 140.88144886f, 147.20671787f, 153.58008562f,
  160.00000000f, 166.46500594f, 172.97373660f, 179.52490559f,
  186.11730005f, 192.74977453f, 199.42124551f, 206.13068654f,
  212.87712380f, 219.65963219f, 226.47733176f, 233.32938445f,
  240.21499122f, 247.13338933f, 254.08384998f, 261.06567603f,
  268.07820003f, 275.12078236f, 282.19280949f, 289.29369244f,
  296.42286534f, 303.57978409f, 310.76392512f, 317.97478424f,
  325.21187564f, 332.47473081f, 339.76289772f, 347.07593991f,
  354.41343574f, 361.77497759f, 369.16017124f, 376.56863518f,
  384.00000000f, 391.45390785f, 398.93001188f, 406.42797576f,
  413.94747321f, 421.48818752f, 429.04981119f, 436.63204548f,
  444.23460010f, 451.85719280f, 459.49954906f, 467.16140179f,
  474.84249102f, 482.54256363f, 490.26137307f, 497.99867911f,
  505.75424759f, 513.52785023f, 521.31926438f, 529.12827280f,
  536.95466351f, 544.79822957f, 552.65876890f, 560.53608414f,
  568.42998244f, 576.34027536f, 584.26677867f, 592.20931226f,
  600.16769996f, 608.14176943f, 616.13135206f, 624.13628279f,
  632.15640007f, 640.19154569f, 648.24156472f, 656.30630539f,
  664.38561898f, 672.47935976f, 680.58738488f, 688.70955430f,
  696.84573069f, 704.99577935f, 713.15956818f, 721.33696754f,
  729.52785023f, 737.73209140f, 745.94956849f, 754.18016116f,
  762.42375127f, 770.68022275f, 778.94946161f, 787.23135586f,
  795.52579543f, 803.83267219f, 812.15187982f, 820.48331383f,
  828.82687147f, 837.18245171f, 845.54995518f, 853.92928416f,
  862.32034249f, 870.72303558f, 879.13727036f, 887.56295522f,
  896.00000000f, 904.44831595f, 912.90781569f, 921.37841320f,
  929.86002376f, 938.35256392f, 946.85595152f, 955.37010560f,
  963.89494641f, 972.43039537f, 980.97637504f, 989.53280911f,
  998.09962237f, 1006.67674069f, 1015.26409097f, 1023.86160116f,
  1032.46920021f, 1041.08681805f, 1049.71438560f, 1058.35183469f,
  1066.99909811f, 1075.65610955f, 1084.32280357f, 1092.99911564f,
  1101.68498204f, 1110.38033993f, 1119.08512727f, 1127.79928282f,
  1136.52274614f, 1145.25545758f, 1153.99735821f, 1162.74838989f,
  1171.50849518f, 1180.27761738f, 1189.05570047f, 1197.84268914f,
  1206.63852876f, 1215.44316535f, 1224.25654560f, 1233.07861684f,
  1241.90932703f, 1250.74862473f, 1259.59645914f, 1268.45278005f,
  1277.31753781f, 1286.19068338f, 1295.07216828f, 1303.96194457f,
  1312.85996488f, 1321.76618236f, 1330.68055071f, 1339.60302413f,
  1348.53355734f, 1357.47210556f, 1366.41862452f, 1375.37307041f,
  1384.33539991f, 1393.30557020f, 1402.28353887f, 1411.26926400f,
  1420.26270412f, 1429.26381818f, 1438.27256558f, 1447.28890615f,
  1456.31280014f, 1465.34420819f, 1474.38309138f, 1483.42941118f,
  1492.48312945f, 1501.54420843f, 1510.61261078f, 1519.68829949f,
  1528.77123795f, 1537.86138993f, 1546.95871952f, 1556.06319119f,
  1565.17476976f, 1574.29342040f, 1583.41910860f, 1592.55180020f,
  1601.69146137f, 1610.83805860f, 1619.99155871f, 1629.15192882f,
  1638.31913637f, 1647.49314911f, 1656.67393509f, 1665.86146266f,
  1675.05570047f, 1684.25661744f, 1693.46418280f, 1702.67836605f,
  1711.89913698f, 1721.12646563f, 1730.36032233f, 1739.60067768f,
  1748.84750254f, 1758.10076802f, 1767.36044551f, 1776.62650662f,
  1785.89892323f, 1795.17766747f, 1804.46271172f, 1813.75402857f,
  1823.05159087f, 1832.35537170f, 1841.66534438f, 1850.98148244f,
  1860.30375965f, 1869.63214999f, 1878.96662767f, 1888.30716711f,
  1897.65374295f, 1907.00633003f, 1916.36490342f, 1925.72943838f,
  1935.09991037f, 1944.47629506f, 1953.85856831f, 1963.24670620f,
  1972.64068498f, 1982.04048108f, 1991.44607117f, 2000.85743204f,
  2010.27454072f, 2019.69737440f, 2029.12591044f, 2038.56012640f
};

const VP8LPrefixCode kPrefixEncodeCode[PREFIX_LOOKUP_IDX_MAX] = {
  { 0, 0}, { 0, 0}, { 1, 0}, { 2, 0}, { 3, 0}, { 4, 1}, { 4, 1}, { 5, 1},
  { 5, 1}, { 6, 2}, { 6, 2}, { 6, 2}, { 6, 2}, { 7, 2}, { 7, 2}, { 7, 2},
  { 7, 2}, { 8, 3}, { 8, 3}, { 8, 3}, { 8, 3}, { 8, 3}, { 8, 3}, { 8, 3},
  { 8, 3}, { 9, 3}, { 9, 3}, { 9, 3}, { 9, 3}, { 9, 3}, { 9, 3}, { 9, 3},
  { 9, 3}, {10, 4}, {10, 4}, {10, 4}, {10, 4}, {10, 4}, {10, 4}, {10, 4},
  {10, 4}, {10, 4}, {10, 4}, {10, 4}, {10, 4}, {10, 4}, {10, 4}, {10, 4},
  {10, 4}, {11, 4}, {11, 4}, {11, 4}, {11, 4}, {11, 4}, {11, 4}, {11, 4},
  {11, 4}, {11, 4}, {11, 4}, {11, 4}, {11, 4}, {11, 4}, {11, 4}, {11, 4},
  {11, 4}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5},
  {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5},
  {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5},
  {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5}, {12, 5},
  {12, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5},
  {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5},
  {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5},
  {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5}, {13, 5},
  {13, 5}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6},
  {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6},
  {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6},
  {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6},
  {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6},
  {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6},
  {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6},
  {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6}, {14, 6},
  {14, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6},
  {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6},
  {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6},
  {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6},
  {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6},
  {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6},
  {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6},
  {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6}, {15, 6},
  {15, 6}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
  {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
  {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
  {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
  {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
  {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
  {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
  {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
  {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
  {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
  {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
  {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
  {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
  {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
  {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
  {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7}, {16, 7},
  {16, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
  {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
  {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
  {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
  {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
  {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
  {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
  {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
  {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
  {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
  {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
  {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
  {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
  {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
  {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
  {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7}, {17, 7},
};

const uint8_t kPrefixEncodeExtraBitsValue[PREFIX_LOOKUP_IDX_MAX] = {
   0,  0,  0,  0,  0,  0,  1,  0,  1,  0,  1,  2,  3,  0,  1,  2,  3,
   0,  1,  2,  3,  4,  5,  6,  7,  0,  1,  2,  3,  4,  5,  6,  7,
   0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
   0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
   0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
  16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
   0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
  16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
   0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
  16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
  32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
  48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
   0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
  16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
  32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
  48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
   0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
  16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
  32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
  48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
  64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
  80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
  96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
  112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,
  127,
   0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
  16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
  32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
  48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
  64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
  80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
  96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
  112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126
};

// The threshold till approximate version of log_2 can be used.
// Practically, we can get rid of the call to log() as the two values match to
// very high degree (the ratio of these two is 0.99999x).
// Keeping a high threshold for now.
#define APPROX_LOG_WITH_CORRECTION_MAX  65536
#define APPROX_LOG_MAX                   4096
#define LOG_2_RECIPROCAL 1.44269504088896338700465094007086
static float FastSLog2Slow(uint32_t v) {
  assert(v >= LOG_LOOKUP_IDX_MAX);
  if (v < APPROX_LOG_WITH_CORRECTION_MAX) {
    int log_cnt = 0;
    uint32_t y = 1;
    int correction = 0;
    const float v_f = (float)v;
    const uint32_t orig_v = v;
    do {
      ++log_cnt;
      v = v >> 1;
      y = y << 1;
    } while (v >= LOG_LOOKUP_IDX_MAX);
    // vf = (2^log_cnt) * Xf; where y = 2^log_cnt and Xf < 256
    // Xf = floor(Xf) * (1 + (v % y) / v)
    // log2(Xf) = log2(floor(Xf)) + log2(1 + (v % y) / v)
    // The correction factor: log(1 + d) ~ d; for very small d values, so
    // log2(1 + (v % y) / v) ~ LOG_2_RECIPROCAL * (v % y)/v
    // LOG_2_RECIPROCAL ~ 23/16
    correction = (23 * (orig_v & (y - 1))) >> 4;
    return v_f * (kLog2Table[v] + log_cnt) + correction;
  } else {
    return (float)(LOG_2_RECIPROCAL * v * log((double)v));
  }
}

static float FastLog2Slow(uint32_t v) {
  assert(v >= LOG_LOOKUP_IDX_MAX);
  if (v < APPROX_LOG_WITH_CORRECTION_MAX) {
    int log_cnt = 0;
    uint32_t y = 1;
    const uint32_t orig_v = v;
    double log_2;
    do {
      ++log_cnt;
      v = v >> 1;
      y = y << 1;
    } while (v >= LOG_LOOKUP_IDX_MAX);
    log_2 = kLog2Table[v] + log_cnt;
    if (orig_v >= APPROX_LOG_MAX) {
      // Since the division is still expensive, add this correction factor only
      // for large values of 'v'.
      const int correction = (23 * (orig_v & (y - 1))) >> 4;
      log_2 += (double)correction / orig_v;
    }
    return (float)log_2;
  } else {
    return (float)(LOG_2_RECIPROCAL * log((double)v));
  }
}

// Mostly used to reduce code size + readability
static WEBP_INLINE int GetMin(int a, int b) { return (a > b) ? b : a; }

//------------------------------------------------------------------------------
// Methods to calculate Entropy (Shannon).

static float PredictionCostSpatial(const int counts[256], int weight_0,
                                   double exp_val) {
  const int significant_symbols = 256 >> 4;
  const double exp_decay_factor = 0.6;
  double bits = weight_0 * counts[0];
  int i;
  for (i = 1; i < significant_symbols; ++i) {
    bits += exp_val * (counts[i] + counts[256 - i]);
    exp_val *= exp_decay_factor;
  }
  return (float)(-0.1 * bits);
}

// Compute the combined Shanon's entropy for distribution {X} and {X+Y}
static float CombinedShannonEntropy(const int X[256], const int Y[256]) {
  int i;
  double retval = 0.;
  int sumX = 0, sumXY = 0;
  for (i = 0; i < 256; ++i) {
    const int x = X[i];
    const int xy = x + Y[i];
    if (x != 0) {
      sumX += x;
      retval -= VP8LFastSLog2(x);
      sumXY += xy;
      retval -= VP8LFastSLog2(xy);
    } else if (xy != 0) {
      sumXY += xy;
      retval -= VP8LFastSLog2(xy);
    }
  }
  retval += VP8LFastSLog2(sumX) + VP8LFastSLog2(sumXY);
  return (float)retval;
}

static float PredictionCostSpatialHistogram(const int accumulated[4][256],
                                            const int tile[4][256]) {
  int i;
  double retval = 0;
  for (i = 0; i < 4; ++i) {
    const double kExpValue = 0.94;
    retval += PredictionCostSpatial(tile[i], 1, kExpValue);
    retval += CombinedShannonEntropy(tile[i], accumulated[i]);
  }
  return (float)retval;
}

static WEBP_INLINE double BitsEntropyRefine(int nonzeros, int sum, int max_val,
                                            double retval) {
  double mix;
  if (nonzeros < 5) {
    if (nonzeros <= 1) {
      return 0;
    }
    // Two symbols, they will be 0 and 1 in a Huffman code.
    // Let's mix in a bit of entropy to favor good clustering when
    // distributions of these are combined.
    if (nonzeros == 2) {
      return 0.99 * sum + 0.01 * retval;
    }
    // No matter what the entropy says, we cannot be better than min_limit
    // with Huffman coding. I am mixing a bit of entropy into the
    // min_limit since it produces much better (~0.5 %) compression results
    // perhaps because of better entropy clustering.
    if (nonzeros == 3) {
      mix = 0.95;
    } else {
      mix = 0.7;  // nonzeros == 4.
    }
  } else {
    mix = 0.627;
  }

  {
    double min_limit = 2 * sum - max_val;
    min_limit = mix * min_limit + (1.0 - mix) * retval;
    return (retval < min_limit) ? min_limit : retval;
  }
}

// Returns the entropy for the symbols in the input array.
// Also sets trivial_symbol to the code value, if the array has only one code
// value. Otherwise, set it to VP8L_NON_TRIVIAL_SYM.
double VP8LBitsEntropy(const uint32_t* const array, int n,
                       uint32_t* const trivial_symbol) {
  double retval = 0.;
  uint32_t sum = 0;
  uint32_t nonzero_code = VP8L_NON_TRIVIAL_SYM;
  int nonzeros = 0;
  uint32_t max_val = 0;
  int i;
  for (i = 0; i < n; ++i) {
    if (array[i] != 0) {
      sum += array[i];
      nonzero_code = i;
      ++nonzeros;
      retval -= VP8LFastSLog2(array[i]);
      if (max_val < array[i]) {
        max_val = array[i];
      }
    }
  }
  retval += VP8LFastSLog2(sum);
  if (trivial_symbol != NULL) {
    *trivial_symbol = (nonzeros == 1) ? nonzero_code : VP8L_NON_TRIVIAL_SYM;
  }
  return BitsEntropyRefine(nonzeros, sum, max_val, retval);
}

static double InitialHuffmanCost(void) {
  // Small bias because Huffman code length is typically not stored in
  // full length.
  static const int kHuffmanCodeOfHuffmanCodeSize = CODE_LENGTH_CODES * 3;
  static const double kSmallBias = 9.1;
  return kHuffmanCodeOfHuffmanCodeSize - kSmallBias;
}

// Finalize the Huffman cost based on streak numbers and length type (<3 or >=3)
static double FinalHuffmanCost(const VP8LStreaks* const stats) {
  double retval = InitialHuffmanCost();
  retval += stats->counts[0] * 1.5625 + 0.234375 * stats->streaks[0][1];
  retval += stats->counts[1] * 2.578125 + 0.703125 * stats->streaks[1][1];
  retval += 1.796875 * stats->streaks[0][0];
  retval += 3.28125 * stats->streaks[1][0];
  return retval;
}

// Trampolines
static double HuffmanCost(const uint32_t* const population, int length) {
  const VP8LStreaks stats = VP8LHuffmanCostCount(population, length);
  return FinalHuffmanCost(&stats);
}

// Aggregated costs
double VP8LPopulationCost(const uint32_t* const population, int length,
                          uint32_t* const trivial_sym) {
  return
      VP8LBitsEntropy(population, length, trivial_sym) +
      HuffmanCost(population, length);
}

double VP8LGetCombinedEntropy(const uint32_t* const X,
                              const uint32_t* const Y, int length) {
  double bits_entropy_combined;
  double huffman_cost_combined;
  int i;

  // Bit entropy variables.
  double retval = 0.;
  int sum = 0;
  int nonzeros = 0;
  uint32_t max_val = 0;
  int i_prev;
  uint32_t xy;

  // Huffman cost variables.
  int streak = 0;
  uint32_t xy_prev;
  VP8LStreaks stats;
  memset(&stats, 0, sizeof(stats));

  // Treat the first value for the huffman cost: this is keeping the original
  // behavior, even though there is no first streak.
  // TODO(vrabaud): study proper behavior
  xy = X[0] + Y[0];
  ++stats.streaks[xy != 0][0];
  xy_prev = xy;
  i_prev = 0;

  for (i = 1; i < length; ++i) {
    xy = X[i] + Y[i];

    // Process data by streaks for both bit entropy and huffman cost.
    if (xy != xy_prev) {
      streak = i - i_prev;

      // Gather info for the bit entropy.
      if (xy_prev != 0) {
        sum += xy_prev * streak;
        nonzeros += streak;
        retval -= VP8LFastSLog2(xy_prev) * streak;
        if (max_val < xy_prev) {
          max_val = xy_prev;
        }
      }

      // Gather info for the huffman cost.
      stats.counts[xy != 0] += (streak > 3);
      stats.streaks[xy != 0][(streak > 3)] += streak;

      xy_prev = xy;
      i_prev = i;
    }
  }

  // Finish off the last streak for bit entropy.
  if (xy != 0) {
    streak = i - i_prev;
    sum += xy * streak;
    nonzeros += streak;
    retval -= VP8LFastSLog2(xy) * streak;
    if (max_val < xy) {
      max_val = xy;
    }
  }
  // Huffman cost is not updated with the last streak to keep original behavior.
  // TODO(vrabaud): study proper behavior

  retval += VP8LFastSLog2(sum);
  bits_entropy_combined = BitsEntropyRefine(nonzeros, sum, max_val, retval);

  huffman_cost_combined = FinalHuffmanCost(&stats);

  return bits_entropy_combined + huffman_cost_combined;
}

// Estimates the Entropy + Huffman + other block overhead size cost.
double VP8LHistogramEstimateBits(const VP8LHistogram* const p) {
  return
      VP8LPopulationCost(
          p->literal_, VP8LHistogramNumCodes(p->palette_code_bits_), NULL)
      + VP8LPopulationCost(p->red_, NUM_LITERAL_CODES, NULL)
      + VP8LPopulationCost(p->blue_, NUM_LITERAL_CODES, NULL)
      + VP8LPopulationCost(p->alpha_, NUM_LITERAL_CODES, NULL)
      + VP8LPopulationCost(p->distance_, NUM_DISTANCE_CODES, NULL)
      + VP8LExtraCost(p->literal_ + NUM_LITERAL_CODES, NUM_LENGTH_CODES)
      + VP8LExtraCost(p->distance_, NUM_DISTANCE_CODES);
}

double VP8LHistogramEstimateBitsBulk(const VP8LHistogram* const p) {
  return
      VP8LBitsEntropy(p->literal_, VP8LHistogramNumCodes(p->palette_code_bits_),
                  NULL)
      + VP8LBitsEntropy(p->red_, NUM_LITERAL_CODES, NULL)
      + VP8LBitsEntropy(p->blue_, NUM_LITERAL_CODES, NULL)
      + VP8LBitsEntropy(p->alpha_, NUM_LITERAL_CODES, NULL)
      + VP8LBitsEntropy(p->distance_, NUM_DISTANCE_CODES, NULL)
      + VP8LExtraCost(p->literal_ + NUM_LITERAL_CODES, NUM_LENGTH_CODES)
      + VP8LExtraCost(p->distance_, NUM_DISTANCE_CODES);
}

static WEBP_INLINE void UpdateHisto(int histo_argb[4][256], uint32_t argb) {
  ++histo_argb[0][argb >> 24];
  ++histo_argb[1][(argb >> 16) & 0xff];
  ++histo_argb[2][(argb >> 8) & 0xff];
  ++histo_argb[3][argb & 0xff];
}

//------------------------------------------------------------------------------

// Returns best predictor and updates the accumulated histogram.
static int GetBestPredictorForTile(int width, int height,
                                   int tile_x, int tile_y, int bits,
                                   int accumulated[4][256],
                                   const uint32_t* const argb_scratch) {
  const int kNumPredModes = 14;
  const int col_start = tile_x << bits;
  const int row_start = tile_y << bits;
  const int tile_size = 1 << bits;
  const int max_y = GetMin(tile_size, height - row_start);
  const int max_x = GetMin(tile_size, width - col_start);
  float best_diff = MAX_DIFF_COST;
  int best_mode = 0;
  int mode;
  int histo_stack_1[4][256];
  int histo_stack_2[4][256];
  // Need pointers to be able to swap arrays.
  int (*histo_argb)[256] = histo_stack_1;
  int (*best_histo)[256] = histo_stack_2;

  int i, j;
  for (mode = 0; mode < kNumPredModes; ++mode) {
    const uint32_t* current_row = argb_scratch;
    const VP8LPredictorFunc pred_func = VP8LPredictors[mode];
    float cur_diff;
    int y;
    memset(histo_argb, 0, sizeof(histo_stack_1));
    for (y = 0; y < max_y; ++y) {
      int x;
      const int row = row_start + y;
      const uint32_t* const upper_row = current_row;
      current_row = upper_row + width;
      for (x = 0; x < max_x; ++x) {
        const int col = col_start + x;
        uint32_t predict;
        if (row == 0) {
          predict = (col == 0) ? ARGB_BLACK : current_row[col - 1];  // Left.
        } else if (col == 0) {
          predict = upper_row[col];  // Top.
        } else {
          predict = pred_func(current_row[col - 1], upper_row + col);
        }
        UpdateHisto(histo_argb, VP8LSubPixels(current_row[col], predict));
      }
    }
    cur_diff = PredictionCostSpatialHistogram(
        (const int (*)[256])accumulated, (const int (*)[256])histo_argb);
    if (cur_diff < best_diff) {
      int (*tmp)[256] = histo_argb;
      histo_argb = best_histo;
      best_histo = tmp;
      best_diff = cur_diff;
      best_mode = mode;
    }
  }

  for (i = 0; i < 4; i++) {
    for (j = 0; j < 256; j++) {
      accumulated[i][j] += best_histo[i][j];
    }
  }

  return best_mode;
}

static void CopyImageWithPrediction(int width, int height,
                                    int bits, uint32_t* const modes,
                                    uint32_t* const argb_scratch,
                                    uint32_t* const argb) {
  const int tiles_per_row = VP8LSubSampleSize(width, bits);
  const int mask = (1 << bits) - 1;
  // The row size is one pixel longer to allow the top right pixel to point to
  // the leftmost pixel of the next row when at the right edge.
  uint32_t* current_row = argb_scratch;
  uint32_t* upper_row = argb_scratch + width + 1;
  int y;
  VP8LPredictorFunc pred_func = 0;

  for (y = 0; y < height; ++y) {
    int x;
    uint32_t* tmp = upper_row;
    upper_row = current_row;
    current_row = tmp;
    memcpy(current_row, argb + y * width, sizeof(*current_row) * width);
    current_row[width] = (y + 1 < height) ? argb[(y + 1) * width] : ARGB_BLACK;
    for (x = 0; x < width; ++x) {
      uint32_t predict;
      if ((x & mask) == 0) {
        const int mode =
            (modes[(y >> bits) * tiles_per_row + (x >> bits)] >> 8) & 0xff;
        pred_func = VP8LPredictors[mode];
      }
      if (y == 0) {
        predict = (x == 0) ? ARGB_BLACK : current_row[x - 1];  // Left.
      } else if (x == 0) {
        predict = upper_row[x];  // Top.
      } else {
        predict = pred_func(current_row[x - 1], upper_row + x);
      }
      argb[y * width + x] = VP8LSubPixels(current_row[x], predict);
    }
  }
}

void VP8LResidualImage(int width, int height, int bits, int low_effort,
                       uint32_t* const argb, uint32_t* const argb_scratch,
                       uint32_t* const image) {
  const int max_tile_size = 1 << bits;
  const int tiles_per_row = VP8LSubSampleSize(width, bits);
  const int tiles_per_col = VP8LSubSampleSize(height, bits);
  const int kPredLowEffort = 11;
  uint32_t* const upper_row = argb_scratch;
  uint32_t* const current_tile_rows = argb_scratch + width;
  int tile_y;
  int histo[4][256];
  if (!low_effort) memset(histo, 0, sizeof(histo));
  for (tile_y = 0; tile_y < tiles_per_col; ++tile_y) {
    const int tile_y_offset = tile_y * max_tile_size;
    const int this_tile_height =
        (tile_y < tiles_per_col - 1) ? max_tile_size : height - tile_y_offset;
    int tile_x;
    if (tile_y > 0) {
      memcpy(upper_row, current_tile_rows + (max_tile_size - 1) * width,
             width * sizeof(*upper_row));
    }
    memcpy(current_tile_rows, &argb[tile_y_offset * width],
           this_tile_height * width * sizeof(*current_tile_rows));
    for (tile_x = 0; tile_x < tiles_per_row; ++tile_x) {
      const int pred =
          low_effort ? kPredLowEffort :
                       GetBestPredictorForTile(width, height,
                                               tile_x, tile_y, bits,
                                               (int (*)[256])histo,
                                               argb_scratch);
      image[tile_y * tiles_per_row + tile_x] = 0xff000000u | (pred << 8);
    }
  }

  CopyImageWithPrediction(width, height, bits, image, argb_scratch, argb);
}

void VP8LSubtractGreenFromBlueAndRed_C(uint32_t* argb_data, int num_pixels) {
  int i;
  for (i = 0; i < num_pixels; ++i) {
    const uint32_t argb = argb_data[i];
    const uint32_t green = (argb >> 8) & 0xff;
    const uint32_t new_r = (((argb >> 16) & 0xff) - green) & 0xff;
    const uint32_t new_b = ((argb & 0xff) - green) & 0xff;
    argb_data[i] = (argb & 0xff00ff00) | (new_r << 16) | new_b;
  }
}

static WEBP_INLINE void MultipliersClear(VP8LMultipliers* const m) {
  m->green_to_red_ = 0;
  m->green_to_blue_ = 0;
  m->red_to_blue_ = 0;
}

static WEBP_INLINE uint32_t ColorTransformDelta(int8_t color_pred,
                                                int8_t color) {
  return (uint32_t)((int)(color_pred) * color) >> 5;
}

static WEBP_INLINE void ColorCodeToMultipliers(uint32_t color_code,
                                               VP8LMultipliers* const m) {
  m->green_to_red_  = (color_code >>  0) & 0xff;
  m->green_to_blue_ = (color_code >>  8) & 0xff;
  m->red_to_blue_   = (color_code >> 16) & 0xff;
}

static WEBP_INLINE uint32_t MultipliersToColorCode(
    const VP8LMultipliers* const m) {
  return 0xff000000u |
         ((uint32_t)(m->red_to_blue_) << 16) |
         ((uint32_t)(m->green_to_blue_) << 8) |
         m->green_to_red_;
}

void VP8LTransformColor_C(const VP8LMultipliers* const m, uint32_t* data,
                          int num_pixels) {
  int i;
  for (i = 0; i < num_pixels; ++i) {
    const uint32_t argb = data[i];
    const uint32_t green = argb >> 8;
    const uint32_t red = argb >> 16;
    uint32_t new_red = red;
    uint32_t new_blue = argb;
    new_red -= ColorTransformDelta(m->green_to_red_, green);
    new_red &= 0xff;
    new_blue -= ColorTransformDelta(m->green_to_blue_, green);
    new_blue -= ColorTransformDelta(m->red_to_blue_, red);
    new_blue &= 0xff;
    data[i] = (argb & 0xff00ff00u) | (new_red << 16) | (new_blue);
  }
}

static WEBP_INLINE uint8_t TransformColorRed(uint8_t green_to_red,
                                             uint32_t argb) {
  const uint32_t green = argb >> 8;
  uint32_t new_red = argb >> 16;
  new_red -= ColorTransformDelta(green_to_red, green);
  return (new_red & 0xff);
}

static WEBP_INLINE uint8_t TransformColorBlue(uint8_t green_to_blue,
                                              uint8_t red_to_blue,
                                              uint32_t argb) {
  const uint32_t green = argb >> 8;
  const uint32_t red = argb >> 16;
  uint8_t new_blue = argb;
  new_blue -= ColorTransformDelta(green_to_blue, green);
  new_blue -= ColorTransformDelta(red_to_blue, red);
  return (new_blue & 0xff);
}

static float PredictionCostCrossColor(const int accumulated[256],
                                      const int counts[256]) {
  // Favor low entropy, locally and globally.
  // Favor small absolute values for PredictionCostSpatial
  static const double kExpValue = 2.4;
  return CombinedShannonEntropy(counts, accumulated) +
         PredictionCostSpatial(counts, 3, kExpValue);
}

void VP8LCollectColorRedTransforms_C(const uint32_t* argb, int stride,
                                     int tile_width, int tile_height,
                                     int green_to_red, int histo[]) {
  while (tile_height-- > 0) {
    int x;
    for (x = 0; x < tile_width; ++x) {
      ++histo[TransformColorRed(green_to_red, argb[x])];
    }
    argb += stride;
  }
}

static float GetPredictionCostCrossColorRed(
    const uint32_t* argb, int stride, int tile_width, int tile_height,
    VP8LMultipliers prev_x, VP8LMultipliers prev_y, int green_to_red,
    const int accumulated_red_histo[256]) {
  int histo[256] = { 0 };
  float cur_diff;

  VP8LCollectColorRedTransforms(argb, stride, tile_width, tile_height,
                                green_to_red, histo);

  cur_diff = PredictionCostCrossColor(accumulated_red_histo, histo);
  if ((uint8_t)green_to_red == prev_x.green_to_red_) {
    cur_diff -= 3;  // favor keeping the areas locally similar
  }
  if ((uint8_t)green_to_red == prev_y.green_to_red_) {
    cur_diff -= 3;  // favor keeping the areas locally similar
  }
  if (green_to_red == 0) {
    cur_diff -= 3;
  }
  return cur_diff;
}

static void GetBestGreenToRed(
    const uint32_t* argb, int stride, int tile_width, int tile_height,
    VP8LMultipliers prev_x, VP8LMultipliers prev_y, int quality,
    const int accumulated_red_histo[256], VP8LMultipliers* const best_tx) {
  const int kMaxIters = 4 + ((7 * quality) >> 8);  // in range [4..6]
  int green_to_red_best = 0;
  int iter, offset;
  float best_diff = GetPredictionCostCrossColorRed(
      argb, stride, tile_width, tile_height, prev_x, prev_y,
      green_to_red_best, accumulated_red_histo);
  for (iter = 0; iter < kMaxIters; ++iter) {
    // ColorTransformDelta is a 3.5 bit fixed point, so 32 is equal to
    // one in color computation. Having initial delta here as 1 is sufficient
    // to explore the range of (-2, 2).
    const int delta = 32 >> iter;
    // Try a negative and a positive delta from the best known value.
    for (offset = -delta; offset <= delta; offset += 2 * delta) {
      const int green_to_red_cur = offset + green_to_red_best;
      const float cur_diff = GetPredictionCostCrossColorRed(
          argb, stride, tile_width, tile_height, prev_x, prev_y,
          green_to_red_cur, accumulated_red_histo);
      if (cur_diff < best_diff) {
        best_diff = cur_diff;
        green_to_red_best = green_to_red_cur;
      }
    }
  }
  best_tx->green_to_red_ = green_to_red_best;
}

void VP8LCollectColorBlueTransforms_C(const uint32_t* argb, int stride,
                                      int tile_width, int tile_height,
                                      int green_to_blue, int red_to_blue,
                                      int histo[]) {
  while (tile_height-- > 0) {
    int x;
    for (x = 0; x < tile_width; ++x) {
      ++histo[TransformColorBlue(green_to_blue, red_to_blue, argb[x])];
    }
    argb += stride;
  }
}

static float GetPredictionCostCrossColorBlue(
    const uint32_t* argb, int stride, int tile_width, int tile_height,
    VP8LMultipliers prev_x, VP8LMultipliers prev_y,
    int green_to_blue, int red_to_blue, const int accumulated_blue_histo[256]) {
  int histo[256] = { 0 };
  float cur_diff;

  VP8LCollectColorBlueTransforms(argb, stride, tile_width, tile_height,
                                 green_to_blue, red_to_blue, histo);

  cur_diff = PredictionCostCrossColor(accumulated_blue_histo, histo);
  if ((uint8_t)green_to_blue == prev_x.green_to_blue_) {
    cur_diff -= 3;  // favor keeping the areas locally similar
  }
  if ((uint8_t)green_to_blue == prev_y.green_to_blue_) {
    cur_diff -= 3;  // favor keeping the areas locally similar
  }
  if ((uint8_t)red_to_blue == prev_x.red_to_blue_) {
    cur_diff -= 3;  // favor keeping the areas locally similar
  }
  if ((uint8_t)red_to_blue == prev_y.red_to_blue_) {
    cur_diff -= 3;  // favor keeping the areas locally similar
  }
  if (green_to_blue == 0) {
    cur_diff -= 3;
  }
  if (red_to_blue == 0) {
    cur_diff -= 3;
  }
  return cur_diff;
}

#define kGreenRedToBlueNumAxis 8
#define kGreenRedToBlueMaxIters 7
static void GetBestGreenRedToBlue(
    const uint32_t* argb, int stride, int tile_width, int tile_height,
    VP8LMultipliers prev_x, VP8LMultipliers prev_y, int quality,
    const int accumulated_blue_histo[256],
    VP8LMultipliers* const best_tx) {
  const int8_t offset[kGreenRedToBlueNumAxis][2] =
      {{0, -1}, {0, 1}, {-1, 0}, {1, 0}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
  const int8_t delta_lut[kGreenRedToBlueMaxIters] = { 16, 16, 8, 4, 2, 2, 2 };
  const int iters =
      (quality < 25) ? 1 : (quality > 50) ? kGreenRedToBlueMaxIters : 4;
  int green_to_blue_best = 0;
  int red_to_blue_best = 0;
  int iter;
  // Initial value at origin:
  float best_diff = GetPredictionCostCrossColorBlue(
      argb, stride, tile_width, tile_height, prev_x, prev_y,
      green_to_blue_best, red_to_blue_best, accumulated_blue_histo);
  for (iter = 0; iter < iters; ++iter) {
    const int delta = delta_lut[iter];
    int axis;
    for (axis = 0; axis < kGreenRedToBlueNumAxis; ++axis) {
      const int green_to_blue_cur =
          offset[axis][0] * delta + green_to_blue_best;
      const int red_to_blue_cur = offset[axis][1] * delta + red_to_blue_best;
      const float cur_diff = GetPredictionCostCrossColorBlue(
          argb, stride, tile_width, tile_height, prev_x, prev_y,
          green_to_blue_cur, red_to_blue_cur, accumulated_blue_histo);
      if (cur_diff < best_diff) {
        best_diff = cur_diff;
        green_to_blue_best = green_to_blue_cur;
        red_to_blue_best = red_to_blue_cur;
      }
      if (quality < 25 && iter == 4) {
        // Only axis aligned diffs for lower quality.
        break;  // next iter.
      }
    }
    if (delta == 2 && green_to_blue_best == 0 && red_to_blue_best == 0) {
      // Further iterations would not help.
      break;  // out of iter-loop.
    }
  }
  best_tx->green_to_blue_ = green_to_blue_best;
  best_tx->red_to_blue_ = red_to_blue_best;
}
#undef kGreenRedToBlueMaxIters
#undef kGreenRedToBlueNumAxis

static VP8LMultipliers GetBestColorTransformForTile(
    int tile_x, int tile_y, int bits,
    VP8LMultipliers prev_x,
    VP8LMultipliers prev_y,
    int quality, int xsize, int ysize,
    const int accumulated_red_histo[256],
    const int accumulated_blue_histo[256],
    const uint32_t* const argb) {
  const int max_tile_size = 1 << bits;
  const int tile_y_offset = tile_y * max_tile_size;
  const int tile_x_offset = tile_x * max_tile_size;
  const int all_x_max = GetMin(tile_x_offset + max_tile_size, xsize);
  const int all_y_max = GetMin(tile_y_offset + max_tile_size, ysize);
  const int tile_width = all_x_max - tile_x_offset;
  const int tile_height = all_y_max - tile_y_offset;
  const uint32_t* const tile_argb = argb + tile_y_offset * xsize
                                  + tile_x_offset;
  VP8LMultipliers best_tx;
  MultipliersClear(&best_tx);

  GetBestGreenToRed(tile_argb, xsize, tile_width, tile_height,
                    prev_x, prev_y, quality, accumulated_red_histo, &best_tx);
  GetBestGreenRedToBlue(tile_argb, xsize, tile_width, tile_height,
                        prev_x, prev_y, quality, accumulated_blue_histo,
                        &best_tx);
  return best_tx;
}

static void CopyTileWithColorTransform(int xsize, int ysize,
                                       int tile_x, int tile_y,
                                       int max_tile_size,
                                       VP8LMultipliers color_transform,
                                       uint32_t* argb) {
  const int xscan = GetMin(max_tile_size, xsize - tile_x);
  int yscan = GetMin(max_tile_size, ysize - tile_y);
  argb += tile_y * xsize + tile_x;
  while (yscan-- > 0) {
    VP8LTransformColor(&color_transform, argb, xscan);
    argb += xsize;
  }
}

void VP8LColorSpaceTransform(int width, int height, int bits, int quality,
                             uint32_t* const argb, uint32_t* image) {
  const int max_tile_size = 1 << bits;
  const int tile_xsize = VP8LSubSampleSize(width, bits);
  const int tile_ysize = VP8LSubSampleSize(height, bits);
  int accumulated_red_histo[256] = { 0 };
  int accumulated_blue_histo[256] = { 0 };
  int tile_x, tile_y;
  VP8LMultipliers prev_x, prev_y;
  MultipliersClear(&prev_y);
  MultipliersClear(&prev_x);
  for (tile_y = 0; tile_y < tile_ysize; ++tile_y) {
    for (tile_x = 0; tile_x < tile_xsize; ++tile_x) {
      int y;
      const int tile_x_offset = tile_x * max_tile_size;
      const int tile_y_offset = tile_y * max_tile_size;
      const int all_x_max = GetMin(tile_x_offset + max_tile_size, width);
      const int all_y_max = GetMin(tile_y_offset + max_tile_size, height);
      const int offset = tile_y * tile_xsize + tile_x;
      if (tile_y != 0) {
        ColorCodeToMultipliers(image[offset - tile_xsize], &prev_y);
      }
      prev_x = GetBestColorTransformForTile(tile_x, tile_y, bits,
                                            prev_x, prev_y,
                                            quality, width, height,
                                            accumulated_red_histo,
                                            accumulated_blue_histo,
                                            argb);
      image[offset] = MultipliersToColorCode(&prev_x);
      CopyTileWithColorTransform(width, height, tile_x_offset, tile_y_offset,
                                 max_tile_size, prev_x, argb);

      // Gather accumulated histogram data.
      for (y = tile_y_offset; y < all_y_max; ++y) {
        int ix = y * width + tile_x_offset;
        const int ix_end = ix + all_x_max - tile_x_offset;
        for (; ix < ix_end; ++ix) {
          const uint32_t pix = argb[ix];
          if (ix >= 2 &&
              pix == argb[ix - 2] &&
              pix == argb[ix - 1]) {
            continue;  // repeated pixels are handled by backward references
          }
          if (ix >= width + 2 &&
              argb[ix - 2] == argb[ix - width - 2] &&
              argb[ix - 1] == argb[ix - width - 1] &&
              pix == argb[ix - width]) {
            continue;  // repeated pixels are handled by backward references
          }
          ++accumulated_red_histo[(pix >> 16) & 0xff];
          ++accumulated_blue_histo[(pix >> 0) & 0xff];
        }
      }
    }
  }
}

//------------------------------------------------------------------------------
// Bundles multiple (1, 2, 4 or 8) pixels into a single pixel.
void VP8LBundleColorMap(const uint8_t* const row, int width,
                        int xbits, uint32_t* const dst) {
  int x;
  if (xbits > 0) {
    const int bit_depth = 1 << (3 - xbits);
    const int mask = (1 << xbits) - 1;
    uint32_t code = 0xff000000;
    for (x = 0; x < width; ++x) {
      const int xsub = x & mask;
      if (xsub == 0) {
        code = 0xff000000;
      }
      code |= row[x] << (8 + bit_depth * xsub);
      dst[x >> xbits] = code;
    }
  } else {
    for (x = 0; x < width; ++x) dst[x] = 0xff000000 | (row[x] << 8);
  }
}

//------------------------------------------------------------------------------

static double ExtraCost(const uint32_t* population, int length) {
  int i;
  double cost = 0.;
  for (i = 2; i < length - 2; ++i) cost += (i >> 1) * population[i + 2];
  return cost;
}

static double ExtraCostCombined(const uint32_t* X, const uint32_t* Y,
                                int length) {
  int i;
  double cost = 0.;
  for (i = 2; i < length - 2; ++i) {
    const int xy = X[i + 2] + Y[i + 2];
    cost += (i >> 1) * xy;
  }
  return cost;
}

// Returns the various RLE counts
static VP8LStreaks HuffmanCostCount(const uint32_t* population, int length) {
  int i;
  int streak = 0;
  VP8LStreaks stats;
  memset(&stats, 0, sizeof(stats));
  for (i = 0; i < length - 1; ++i) {
    ++streak;
    if (population[i] == population[i + 1]) {
      continue;
    }
    stats.counts[population[i] != 0] += (streak > 3);
    stats.streaks[population[i] != 0][(streak > 3)] += streak;
    streak = 0;
  }
  ++streak;
  stats.counts[population[i] != 0] += (streak > 3);
  stats.streaks[population[i] != 0][(streak > 3)] += streak;
  return stats;
}

//------------------------------------------------------------------------------

static void HistogramAdd(const VP8LHistogram* const a,
                         const VP8LHistogram* const b,
                         VP8LHistogram* const out) {
  int i;
  const int literal_size = VP8LHistogramNumCodes(a->palette_code_bits_);
  assert(a->palette_code_bits_ == b->palette_code_bits_);
  if (b != out) {
    for (i = 0; i < literal_size; ++i) {
      out->literal_[i] = a->literal_[i] + b->literal_[i];
    }
    for (i = 0; i < NUM_DISTANCE_CODES; ++i) {
      out->distance_[i] = a->distance_[i] + b->distance_[i];
    }
    for (i = 0; i < NUM_LITERAL_CODES; ++i) {
      out->red_[i] = a->red_[i] + b->red_[i];
      out->blue_[i] = a->blue_[i] + b->blue_[i];
      out->alpha_[i] = a->alpha_[i] + b->alpha_[i];
    }
  } else {
    for (i = 0; i < literal_size; ++i) {
      out->literal_[i] += a->literal_[i];
    }
    for (i = 0; i < NUM_DISTANCE_CODES; ++i) {
      out->distance_[i] += a->distance_[i];
    }
    for (i = 0; i < NUM_LITERAL_CODES; ++i) {
      out->red_[i] += a->red_[i];
      out->blue_[i] += a->blue_[i];
      out->alpha_[i] += a->alpha_[i];
    }
  }
}

//------------------------------------------------------------------------------

VP8LProcessBlueAndRedFunc VP8LSubtractGreenFromBlueAndRed;

VP8LTransformColorFunc VP8LTransformColor;

VP8LCollectColorBlueTransformsFunc VP8LCollectColorBlueTransforms;
VP8LCollectColorRedTransformsFunc VP8LCollectColorRedTransforms;

VP8LFastLog2SlowFunc VP8LFastLog2Slow;
VP8LFastLog2SlowFunc VP8LFastSLog2Slow;

VP8LCostFunc VP8LExtraCost;
VP8LCostCombinedFunc VP8LExtraCostCombined;

VP8LCostCountFunc VP8LHuffmanCostCount;

VP8LHistogramAddFunc VP8LHistogramAdd;

extern void VP8LEncDspInitSSE2(void);
extern void VP8LEncDspInitSSE41(void);
extern void VP8LEncDspInitNEON(void);
extern void VP8LEncDspInitMIPS32(void);
extern void VP8LEncDspInitMIPSdspR2(void);

static volatile VP8CPUInfo lossless_enc_last_cpuinfo_used =
    (VP8CPUInfo)&lossless_enc_last_cpuinfo_used;

WEBP_TSAN_IGNORE_FUNCTION void VP8LEncDspInit(void) {
  if (lossless_enc_last_cpuinfo_used == VP8GetCPUInfo) return;

  VP8LDspInit();

  VP8LSubtractGreenFromBlueAndRed = VP8LSubtractGreenFromBlueAndRed_C;

  VP8LTransformColor = VP8LTransformColor_C;

  VP8LCollectColorBlueTransforms = VP8LCollectColorBlueTransforms_C;
  VP8LCollectColorRedTransforms = VP8LCollectColorRedTransforms_C;

  VP8LFastLog2Slow = FastLog2Slow;
  VP8LFastSLog2Slow = FastSLog2Slow;

  VP8LExtraCost = ExtraCost;
  VP8LExtraCostCombined = ExtraCostCombined;

  VP8LHuffmanCostCount = HuffmanCostCount;

  VP8LHistogramAdd = HistogramAdd;

  // If defined, use CPUInfo() to overwrite some pointers with faster versions.
  if (VP8GetCPUInfo != NULL) {
#if defined(WEBP_USE_SSE2)
    if (VP8GetCPUInfo(kSSE2)) {
      VP8LEncDspInitSSE2();
#if defined(WEBP_USE_SSE41)
      if (VP8GetCPUInfo(kSSE4_1)) {
        VP8LEncDspInitSSE41();
      }
#endif
    }
#endif
#if defined(WEBP_USE_NEON)
    if (VP8GetCPUInfo(kNEON)) {
      VP8LEncDspInitNEON();
    }
#endif
#if defined(WEBP_USE_MIPS32)
    if (VP8GetCPUInfo(kMIPS32)) {
      VP8LEncDspInitMIPS32();
    }
#endif
#if defined(WEBP_USE_MIPS_DSP_R2)
    if (VP8GetCPUInfo(kMIPSdspR2)) {
      VP8LEncDspInitMIPSdspR2();
    }
#endif
  }
  lossless_enc_last_cpuinfo_used = VP8GetCPUInfo;
}

//------------------------------------------------------------------------------
