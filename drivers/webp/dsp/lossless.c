// Copyright 2012 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Image transforms and color space conversion methods for lossless decoder.
//
// Authors: Vikas Arora (vikaas.arora@gmail.com)
//          Jyrki Alakuijala (jyrki@google.com)
//          Urvang Joshi (urvang@google.com)

#include "./dsp.h"

#if defined(WEBP_USE_SSE2)
#include <emmintrin.h>
#endif

#include <math.h>
#include <stdlib.h>
#include "./lossless.h"
#include "../dec/vp8li.h"
#include "./yuv.h"

#define MAX_DIFF_COST (1e30f)

// lookup table for small values of log2(int)
#define APPROX_LOG_MAX  4096
#define LOG_2_RECIPROCAL 1.44269504088896338700465094007086
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

float VP8LFastSLog2Slow(int v) {
  assert(v >= LOG_LOOKUP_IDX_MAX);
  if (v < APPROX_LOG_MAX) {
    int log_cnt = 0;
    const float v_f = (float)v;
    while (v >= LOG_LOOKUP_IDX_MAX) {
      ++log_cnt;
      v = v >> 1;
    }
    return v_f * (kLog2Table[v] + log_cnt);
  } else {
    return (float)(LOG_2_RECIPROCAL * v * log((double)v));
  }
}

float VP8LFastLog2Slow(int v) {
  assert(v >= LOG_LOOKUP_IDX_MAX);
  if (v < APPROX_LOG_MAX) {
    int log_cnt = 0;
    while (v >= LOG_LOOKUP_IDX_MAX) {
      ++log_cnt;
      v = v >> 1;
    }
    return kLog2Table[v] + log_cnt;
  } else {
    return (float)(LOG_2_RECIPROCAL * log((double)v));
  }
}

//------------------------------------------------------------------------------
// Image transforms.

// In-place sum of each component with mod 256.
static WEBP_INLINE void AddPixelsEq(uint32_t* a, uint32_t b) {
  const uint32_t alpha_and_green = (*a & 0xff00ff00u) + (b & 0xff00ff00u);
  const uint32_t red_and_blue = (*a & 0x00ff00ffu) + (b & 0x00ff00ffu);
  *a = (alpha_and_green & 0xff00ff00u) | (red_and_blue & 0x00ff00ffu);
}

static WEBP_INLINE uint32_t Average2(uint32_t a0, uint32_t a1) {
  return (((a0 ^ a1) & 0xfefefefeL) >> 1) + (a0 & a1);
}

static WEBP_INLINE uint32_t Average3(uint32_t a0, uint32_t a1, uint32_t a2) {
  return Average2(Average2(a0, a2), a1);
}

static WEBP_INLINE uint32_t Average4(uint32_t a0, uint32_t a1,
                                     uint32_t a2, uint32_t a3) {
  return Average2(Average2(a0, a1), Average2(a2, a3));
}

static WEBP_INLINE uint32_t Clip255(uint32_t a) {
  if (a < 256) {
    return a;
  }
  // return 0, when a is a negative integer.
  // return 255, when a is positive.
  return ~a >> 24;
}

static WEBP_INLINE int AddSubtractComponentFull(int a, int b, int c) {
  return Clip255(a + b - c);
}

static WEBP_INLINE uint32_t ClampedAddSubtractFull(uint32_t c0, uint32_t c1,
                                                   uint32_t c2) {
  const int a = AddSubtractComponentFull(c0 >> 24, c1 >> 24, c2 >> 24);
  const int r = AddSubtractComponentFull((c0 >> 16) & 0xff,
                                         (c1 >> 16) & 0xff,
                                         (c2 >> 16) & 0xff);
  const int g = AddSubtractComponentFull((c0 >> 8) & 0xff,
                                         (c1 >> 8) & 0xff,
                                         (c2 >> 8) & 0xff);
  const int b = AddSubtractComponentFull(c0 & 0xff, c1 & 0xff, c2 & 0xff);
  return (a << 24) | (r << 16) | (g << 8) | b;
}

static WEBP_INLINE int AddSubtractComponentHalf(int a, int b) {
  return Clip255(a + (a - b) / 2);
}

static WEBP_INLINE uint32_t ClampedAddSubtractHalf(uint32_t c0, uint32_t c1,
                                                   uint32_t c2) {
  const uint32_t ave = Average2(c0, c1);
  const int a = AddSubtractComponentHalf(ave >> 24, c2 >> 24);
  const int r = AddSubtractComponentHalf((ave >> 16) & 0xff, (c2 >> 16) & 0xff);
  const int g = AddSubtractComponentHalf((ave >> 8) & 0xff, (c2 >> 8) & 0xff);
  const int b = AddSubtractComponentHalf((ave >> 0) & 0xff, (c2 >> 0) & 0xff);
  return (a << 24) | (r << 16) | (g << 8) | b;
}

static WEBP_INLINE int Sub3(int a, int b, int c) {
  const int pb = b - c;
  const int pa = a - c;
  return abs(pb) - abs(pa);
}

static WEBP_INLINE uint32_t Select(uint32_t a, uint32_t b, uint32_t c) {
  const int pa_minus_pb =
      Sub3((a >> 24)       , (b >> 24)       , (c >> 24)       ) +
      Sub3((a >> 16) & 0xff, (b >> 16) & 0xff, (c >> 16) & 0xff) +
      Sub3((a >>  8) & 0xff, (b >>  8) & 0xff, (c >>  8) & 0xff) +
      Sub3((a      ) & 0xff, (b      ) & 0xff, (c      ) & 0xff);
  return (pa_minus_pb <= 0) ? a : b;
}

//------------------------------------------------------------------------------
// Predictors

static uint32_t Predictor0(uint32_t left, const uint32_t* const top) {
  (void)top;
  (void)left;
  return ARGB_BLACK;
}
static uint32_t Predictor1(uint32_t left, const uint32_t* const top) {
  (void)top;
  return left;
}
static uint32_t Predictor2(uint32_t left, const uint32_t* const top) {
  (void)left;
  return top[0];
}
static uint32_t Predictor3(uint32_t left, const uint32_t* const top) {
  (void)left;
  return top[1];
}
static uint32_t Predictor4(uint32_t left, const uint32_t* const top) {
  (void)left;
  return top[-1];
}
static uint32_t Predictor5(uint32_t left, const uint32_t* const top) {
  const uint32_t pred = Average3(left, top[0], top[1]);
  return pred;
}
static uint32_t Predictor6(uint32_t left, const uint32_t* const top) {
  const uint32_t pred = Average2(left, top[-1]);
  return pred;
}
static uint32_t Predictor7(uint32_t left, const uint32_t* const top) {
  const uint32_t pred = Average2(left, top[0]);
  return pred;
}
static uint32_t Predictor8(uint32_t left, const uint32_t* const top) {
  const uint32_t pred = Average2(top[-1], top[0]);
  (void)left;
  return pred;
}
static uint32_t Predictor9(uint32_t left, const uint32_t* const top) {
  const uint32_t pred = Average2(top[0], top[1]);
  (void)left;
  return pred;
}
static uint32_t Predictor10(uint32_t left, const uint32_t* const top) {
  const uint32_t pred = Average4(left, top[-1], top[0], top[1]);
  return pred;
}
static uint32_t Predictor11(uint32_t left, const uint32_t* const top) {
  const uint32_t pred = VP8LSelect(top[0], left, top[-1]);
  return pred;
}
static uint32_t Predictor12(uint32_t left, const uint32_t* const top) {
  const uint32_t pred = VP8LClampedAddSubtractFull(left, top[0], top[-1]);
  return pred;
}
static uint32_t Predictor13(uint32_t left, const uint32_t* const top) {
  const uint32_t pred = VP8LClampedAddSubtractHalf(left, top[0], top[-1]);
  return pred;
}

// TODO(vikasa): Export the predictor array, to allow SSE2 variants.
typedef uint32_t (*PredictorFunc)(uint32_t left, const uint32_t* const top);
static const PredictorFunc kPredictors[16] = {
  Predictor0, Predictor1, Predictor2, Predictor3,
  Predictor4, Predictor5, Predictor6, Predictor7,
  Predictor8, Predictor9, Predictor10, Predictor11,
  Predictor12, Predictor13,
  Predictor0, Predictor0    // <- padding security sentinels
};

// TODO(vikasa): Replace 256 etc with defines.
static float PredictionCostSpatial(const int* counts,
                                   int weight_0, double exp_val) {
  const int significant_symbols = 16;
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
static float CombinedShannonEntropy(const int* const X,
                                    const int* const Y, int n) {
  int i;
  double retval = 0.;
  int sumX = 0, sumXY = 0;
  for (i = 0; i < n; ++i) {
    const int x = X[i];
    const int xy = X[i] + Y[i];
    if (x != 0) {
      sumX += x;
      retval -= VP8LFastSLog2(x);
    }
    if (xy != 0) {
      sumXY += xy;
      retval -= VP8LFastSLog2(xy);
    }
  }
  retval += VP8LFastSLog2(sumX) + VP8LFastSLog2(sumXY);
  return (float)retval;
}

static float PredictionCostSpatialHistogram(int accumulated[4][256],
                                            int tile[4][256]) {
  int i;
  double retval = 0;
  for (i = 0; i < 4; ++i) {
    const double kExpValue = 0.94;
    retval += PredictionCostSpatial(tile[i], 1, kExpValue);
    retval += CombinedShannonEntropy(tile[i], accumulated[i], 256);
  }
  return (float)retval;
}

static int GetBestPredictorForTile(int width, int height,
                                   int tile_x, int tile_y, int bits,
                                   int accumulated[4][256],
                                   const uint32_t* const argb_scratch) {
  const int kNumPredModes = 14;
  const int col_start = tile_x << bits;
  const int row_start = tile_y << bits;
  const int tile_size = 1 << bits;
  const int ymax = (tile_size <= height - row_start) ?
      tile_size : height - row_start;
  const int xmax = (tile_size <= width - col_start) ?
      tile_size : width - col_start;
  int histo[4][256];
  float best_diff = MAX_DIFF_COST;
  int best_mode = 0;

  int mode;
  for (mode = 0; mode < kNumPredModes; ++mode) {
    const uint32_t* current_row = argb_scratch;
    const PredictorFunc pred_func = kPredictors[mode];
    float cur_diff;
    int y;
    memset(&histo[0][0], 0, sizeof(histo));
    for (y = 0; y < ymax; ++y) {
      int x;
      const int row = row_start + y;
      const uint32_t* const upper_row = current_row;
      current_row = upper_row + width;
      for (x = 0; x < xmax; ++x) {
        const int col = col_start + x;
        uint32_t predict;
        uint32_t predict_diff;
        if (row == 0) {
          predict = (col == 0) ? ARGB_BLACK : current_row[col - 1];  // Left.
        } else if (col == 0) {
          predict = upper_row[col];  // Top.
        } else {
          predict = pred_func(current_row[col - 1], upper_row + col);
        }
        predict_diff = VP8LSubPixels(current_row[col], predict);
        ++histo[0][predict_diff >> 24];
        ++histo[1][((predict_diff >> 16) & 0xff)];
        ++histo[2][((predict_diff >> 8) & 0xff)];
        ++histo[3][(predict_diff & 0xff)];
      }
    }
    cur_diff = PredictionCostSpatialHistogram(accumulated, histo);
    if (cur_diff < best_diff) {
      best_diff = cur_diff;
      best_mode = mode;
    }
  }

  return best_mode;
}

static void CopyTileWithPrediction(int width, int height,
                                   int tile_x, int tile_y, int bits, int mode,
                                   const uint32_t* const argb_scratch,
                                   uint32_t* const argb) {
  const int col_start = tile_x << bits;
  const int row_start = tile_y << bits;
  const int tile_size = 1 << bits;
  const int ymax = (tile_size <= height - row_start) ?
      tile_size : height - row_start;
  const int xmax = (tile_size <= width - col_start) ?
      tile_size : width - col_start;
  const PredictorFunc pred_func = kPredictors[mode];
  const uint32_t* current_row = argb_scratch;

  int y;
  for (y = 0; y < ymax; ++y) {
    int x;
    const int row = row_start + y;
    const uint32_t* const upper_row = current_row;
    current_row = upper_row + width;
    for (x = 0; x < xmax; ++x) {
      const int col = col_start + x;
      const int pix = row * width + col;
      uint32_t predict;
      if (row == 0) {
        predict = (col == 0) ? ARGB_BLACK : current_row[col - 1];  // Left.
      } else if (col == 0) {
        predict = upper_row[col];  // Top.
      } else {
        predict = pred_func(current_row[col - 1], upper_row + col);
      }
      argb[pix] = VP8LSubPixels(current_row[col], predict);
    }
  }
}

void VP8LResidualImage(int width, int height, int bits,
                       uint32_t* const argb, uint32_t* const argb_scratch,
                       uint32_t* const image) {
  const int max_tile_size = 1 << bits;
  const int tiles_per_row = VP8LSubSampleSize(width, bits);
  const int tiles_per_col = VP8LSubSampleSize(height, bits);
  uint32_t* const upper_row = argb_scratch;
  uint32_t* const current_tile_rows = argb_scratch + width;
  int tile_y;
  int histo[4][256];
  memset(histo, 0, sizeof(histo));
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
      int pred;
      int y;
      const int tile_x_offset = tile_x * max_tile_size;
      int all_x_max = tile_x_offset + max_tile_size;
      if (all_x_max > width) {
        all_x_max = width;
      }
      pred = GetBestPredictorForTile(width, height, tile_x, tile_y, bits, histo,
                                     argb_scratch);
      image[tile_y * tiles_per_row + tile_x] = 0xff000000u | (pred << 8);
      CopyTileWithPrediction(width, height, tile_x, tile_y, bits, pred,
                             argb_scratch, argb);
      for (y = 0; y < max_tile_size; ++y) {
        int ix;
        int all_x;
        int all_y = tile_y_offset + y;
        if (all_y >= height) {
          break;
        }
        ix = all_y * width + tile_x_offset;
        for (all_x = tile_x_offset; all_x < all_x_max; ++all_x, ++ix) {
          const uint32_t a = argb[ix];
          ++histo[0][a >> 24];
          ++histo[1][((a >> 16) & 0xff)];
          ++histo[2][((a >> 8) & 0xff)];
          ++histo[3][(a & 0xff)];
        }
      }
    }
  }
}

// Inverse prediction.
static void PredictorInverseTransform(const VP8LTransform* const transform,
                                      int y_start, int y_end, uint32_t* data) {
  const int width = transform->xsize_;
  if (y_start == 0) {  // First Row follows the L (mode=1) mode.
    int x;
    const uint32_t pred0 = Predictor0(data[-1], NULL);
    AddPixelsEq(data, pred0);
    for (x = 1; x < width; ++x) {
      const uint32_t pred1 = Predictor1(data[x - 1], NULL);
      AddPixelsEq(data + x, pred1);
    }
    data += width;
    ++y_start;
  }

  {
    int y = y_start;
    const int mask = (1 << transform->bits_) - 1;
    const int tiles_per_row = VP8LSubSampleSize(width, transform->bits_);
    const uint32_t* pred_mode_base =
        transform->data_ + (y >> transform->bits_) * tiles_per_row;

    while (y < y_end) {
      int x;
      const uint32_t pred2 = Predictor2(data[-1], data - width);
      const uint32_t* pred_mode_src = pred_mode_base;
      PredictorFunc pred_func;

      // First pixel follows the T (mode=2) mode.
      AddPixelsEq(data, pred2);

      // .. the rest:
      pred_func = kPredictors[((*pred_mode_src++) >> 8) & 0xf];
      for (x = 1; x < width; ++x) {
        uint32_t pred;
        if ((x & mask) == 0) {    // start of tile. Read predictor function.
          pred_func = kPredictors[((*pred_mode_src++) >> 8) & 0xf];
        }
        pred = pred_func(data[x - 1], data + x - width);
        AddPixelsEq(data + x, pred);
      }
      data += width;
      ++y;
      if ((y & mask) == 0) {   // Use the same mask, since tiles are squares.
        pred_mode_base += tiles_per_row;
      }
    }
  }
}

static void SubtractGreenFromBlueAndRed(uint32_t* argb_data, int num_pixs) {
  int i = 0;
  for (; i < num_pixs; ++i) {
    const uint32_t argb = argb_data[i];
    const uint32_t green = (argb >> 8) & 0xff;
    const uint32_t new_r = (((argb >> 16) & 0xff) - green) & 0xff;
    const uint32_t new_b = ((argb & 0xff) - green) & 0xff;
    argb_data[i] = (argb & 0xff00ff00) | (new_r << 16) | new_b;
  }
}

// Add green to blue and red channels (i.e. perform the inverse transform of
// 'subtract green').
static void AddGreenToBlueAndRed(uint32_t* data, const uint32_t* data_end) {
  while (data < data_end) {
    const uint32_t argb = *data;
    const uint32_t green = ((argb >> 8) & 0xff);
    uint32_t red_blue = (argb & 0x00ff00ffu);
    red_blue += (green << 16) | green;
    red_blue &= 0x00ff00ffu;
    *data++ = (argb & 0xff00ff00u) | red_blue;
  }
}

typedef struct {
  // Note: the members are uint8_t, so that any negative values are
  // automatically converted to "mod 256" values.
  uint8_t green_to_red_;
  uint8_t green_to_blue_;
  uint8_t red_to_blue_;
} Multipliers;

static WEBP_INLINE void MultipliersClear(Multipliers* m) {
  m->green_to_red_ = 0;
  m->green_to_blue_ = 0;
  m->red_to_blue_ = 0;
}

static WEBP_INLINE uint32_t ColorTransformDelta(int8_t color_pred,
                                                int8_t color) {
  return (uint32_t)((int)(color_pred) * color) >> 5;
}

static WEBP_INLINE void ColorCodeToMultipliers(uint32_t color_code,
                                               Multipliers* const m) {
  m->green_to_red_  = (color_code >>  0) & 0xff;
  m->green_to_blue_ = (color_code >>  8) & 0xff;
  m->red_to_blue_   = (color_code >> 16) & 0xff;
}

static WEBP_INLINE uint32_t MultipliersToColorCode(Multipliers* const m) {
  return 0xff000000u |
         ((uint32_t)(m->red_to_blue_) << 16) |
         ((uint32_t)(m->green_to_blue_) << 8) |
         m->green_to_red_;
}

static WEBP_INLINE uint32_t TransformColor(const Multipliers* const m,
                                           uint32_t argb, int inverse) {
  const uint32_t green = argb >> 8;
  const uint32_t red = argb >> 16;
  uint32_t new_red = red;
  uint32_t new_blue = argb;

  if (inverse) {
    new_red += ColorTransformDelta(m->green_to_red_, green);
    new_red &= 0xff;
    new_blue += ColorTransformDelta(m->green_to_blue_, green);
    new_blue += ColorTransformDelta(m->red_to_blue_, new_red);
    new_blue &= 0xff;
  } else {
    new_red -= ColorTransformDelta(m->green_to_red_, green);
    new_red &= 0xff;
    new_blue -= ColorTransformDelta(m->green_to_blue_, green);
    new_blue -= ColorTransformDelta(m->red_to_blue_, red);
    new_blue &= 0xff;
  }
  return (argb & 0xff00ff00u) | (new_red << 16) | (new_blue);
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

static WEBP_INLINE int SkipRepeatedPixels(const uint32_t* const argb,
                                          int ix, int xsize) {
  const uint32_t v = argb[ix];
  if (ix >= xsize + 3) {
    if (v == argb[ix - xsize] &&
        argb[ix - 1] == argb[ix - xsize - 1] &&
        argb[ix - 2] == argb[ix - xsize - 2] &&
        argb[ix - 3] == argb[ix - xsize - 3]) {
      return 1;
    }
    return v == argb[ix - 3] && v == argb[ix - 2] && v == argb[ix - 1];
  } else if (ix >= 3) {
    return v == argb[ix - 3] && v == argb[ix - 2] && v == argb[ix - 1];
  }
  return 0;
}

static float PredictionCostCrossColor(const int accumulated[256],
                                      const int counts[256]) {
  // Favor low entropy, locally and globally.
  // Favor small absolute values for PredictionCostSpatial
  static const double kExpValue = 2.4;
  return CombinedShannonEntropy(counts, accumulated, 256) +
         PredictionCostSpatial(counts, 3, kExpValue);
}

static Multipliers GetBestColorTransformForTile(
    int tile_x, int tile_y, int bits,
    Multipliers prevX,
    Multipliers prevY,
    int step, int xsize, int ysize,
    int* accumulated_red_histo,
    int* accumulated_blue_histo,
    const uint32_t* const argb) {
  float best_diff = MAX_DIFF_COST;
  float cur_diff;
  const int halfstep = step / 2;
  const int max_tile_size = 1 << bits;
  const int tile_y_offset = tile_y * max_tile_size;
  const int tile_x_offset = tile_x * max_tile_size;
  int green_to_red;
  int green_to_blue;
  int red_to_blue;
  int all_x_max = tile_x_offset + max_tile_size;
  int all_y_max = tile_y_offset + max_tile_size;
  Multipliers best_tx;
  MultipliersClear(&best_tx);
  if (all_x_max > xsize) {
    all_x_max = xsize;
  }
  if (all_y_max > ysize) {
    all_y_max = ysize;
  }

  for (green_to_red = -64; green_to_red <= 64; green_to_red += halfstep) {
    int histo[256] = { 0 };
    int all_y;

    for (all_y = tile_y_offset; all_y < all_y_max; ++all_y) {
      int ix = all_y * xsize + tile_x_offset;
      int all_x;
      for (all_x = tile_x_offset; all_x < all_x_max; ++all_x, ++ix) {
        if (SkipRepeatedPixels(argb, ix, xsize)) {
          continue;
        }
        ++histo[TransformColorRed(green_to_red, argb[ix])];  // red.
      }
    }
    cur_diff = PredictionCostCrossColor(&accumulated_red_histo[0], &histo[0]);
    if ((uint8_t)green_to_red == prevX.green_to_red_) {
      cur_diff -= 3;  // favor keeping the areas locally similar
    }
    if ((uint8_t)green_to_red == prevY.green_to_red_) {
      cur_diff -= 3;  // favor keeping the areas locally similar
    }
    if (green_to_red == 0) {
      cur_diff -= 3;
    }
    if (cur_diff < best_diff) {
      best_diff = cur_diff;
      best_tx.green_to_red_ = green_to_red;
    }
  }
  best_diff = MAX_DIFF_COST;
  for (green_to_blue = -32; green_to_blue <= 32; green_to_blue += step) {
    for (red_to_blue = -32; red_to_blue <= 32; red_to_blue += step) {
      int all_y;
      int histo[256] = { 0 };
      for (all_y = tile_y_offset; all_y < all_y_max; ++all_y) {
        int all_x;
        int ix = all_y * xsize + tile_x_offset;
        for (all_x = tile_x_offset; all_x < all_x_max; ++all_x, ++ix) {
          if (SkipRepeatedPixels(argb, ix, xsize)) {
            continue;
          }
          ++histo[TransformColorBlue(green_to_blue, red_to_blue, argb[ix])];
        }
      }
      cur_diff =
          PredictionCostCrossColor(&accumulated_blue_histo[0], &histo[0]);
      if ((uint8_t)green_to_blue == prevX.green_to_blue_) {
        cur_diff -= 3;  // favor keeping the areas locally similar
      }
      if ((uint8_t)green_to_blue == prevY.green_to_blue_) {
        cur_diff -= 3;  // favor keeping the areas locally similar
      }
      if ((uint8_t)red_to_blue == prevX.red_to_blue_) {
        cur_diff -= 3;  // favor keeping the areas locally similar
      }
      if ((uint8_t)red_to_blue == prevY.red_to_blue_) {
        cur_diff -= 3;  // favor keeping the areas locally similar
      }
      if (green_to_blue == 0) {
        cur_diff -= 3;
      }
      if (red_to_blue == 0) {
        cur_diff -= 3;
      }
      if (cur_diff < best_diff) {
        best_diff = cur_diff;
        best_tx.green_to_blue_ = green_to_blue;
        best_tx.red_to_blue_ = red_to_blue;
      }
    }
  }
  return best_tx;
}

static void CopyTileWithColorTransform(int xsize, int ysize,
                                       int tile_x, int tile_y, int bits,
                                       Multipliers color_transform,
                                       uint32_t* const argb) {
  int y;
  int xscan = 1 << bits;
  int yscan = 1 << bits;
  tile_x <<= bits;
  tile_y <<= bits;
  if (xscan > xsize - tile_x) {
    xscan = xsize - tile_x;
  }
  if (yscan > ysize - tile_y) {
    yscan = ysize - tile_y;
  }
  yscan += tile_y;
  for (y = tile_y; y < yscan; ++y) {
    int ix = y * xsize + tile_x;
    const int end_ix = ix + xscan;
    for (; ix < end_ix; ++ix) {
      argb[ix] = TransformColor(&color_transform, argb[ix], 0);
    }
  }
}

void VP8LColorSpaceTransform(int width, int height, int bits, int step,
                             uint32_t* const argb, uint32_t* image) {
  const int max_tile_size = 1 << bits;
  int tile_xsize = VP8LSubSampleSize(width, bits);
  int tile_ysize = VP8LSubSampleSize(height, bits);
  int accumulated_red_histo[256] = { 0 };
  int accumulated_blue_histo[256] = { 0 };
  int tile_y;
  int tile_x;
  Multipliers prevX;
  Multipliers prevY;
  MultipliersClear(&prevY);
  MultipliersClear(&prevX);
  for (tile_y = 0; tile_y < tile_ysize; ++tile_y) {
    for (tile_x = 0; tile_x < tile_xsize; ++tile_x) {
      Multipliers color_transform;
      int all_x_max;
      int y;
      const int tile_y_offset = tile_y * max_tile_size;
      const int tile_x_offset = tile_x * max_tile_size;
      if (tile_y != 0) {
        ColorCodeToMultipliers(image[tile_y * tile_xsize + tile_x - 1], &prevX);
        ColorCodeToMultipliers(image[(tile_y - 1) * tile_xsize + tile_x],
                               &prevY);
      } else if (tile_x != 0) {
        ColorCodeToMultipliers(image[tile_y * tile_xsize + tile_x - 1], &prevX);
      }
      color_transform =
          GetBestColorTransformForTile(tile_x, tile_y, bits,
                                       prevX, prevY,
                                       step, width, height,
                                       &accumulated_red_histo[0],
                                       &accumulated_blue_histo[0],
                                       argb);
      image[tile_y * tile_xsize + tile_x] =
          MultipliersToColorCode(&color_transform);
      CopyTileWithColorTransform(width, height, tile_x, tile_y, bits,
                                 color_transform, argb);

      // Gather accumulated histogram data.
      all_x_max = tile_x_offset + max_tile_size;
      if (all_x_max > width) {
        all_x_max = width;
      }
      for (y = 0; y < max_tile_size; ++y) {
        int ix;
        int all_x;
        int all_y = tile_y_offset + y;
        if (all_y >= height) {
          break;
        }
        ix = all_y * width + tile_x_offset;
        for (all_x = tile_x_offset; all_x < all_x_max; ++all_x, ++ix) {
          if (ix >= 2 &&
              argb[ix] == argb[ix - 2] &&
              argb[ix] == argb[ix - 1]) {
            continue;  // repeated pixels are handled by backward references
          }
          if (ix >= width + 2 &&
              argb[ix - 2] == argb[ix - width - 2] &&
              argb[ix - 1] == argb[ix - width - 1] &&
              argb[ix] == argb[ix - width]) {
            continue;  // repeated pixels are handled by backward references
          }
          ++accumulated_red_histo[(argb[ix] >> 16) & 0xff];
          ++accumulated_blue_histo[argb[ix] & 0xff];
        }
      }
    }
  }
}

// Color space inverse transform.
static void ColorSpaceInverseTransform(const VP8LTransform* const transform,
                                       int y_start, int y_end, uint32_t* data) {
  const int width = transform->xsize_;
  const int mask = (1 << transform->bits_) - 1;
  const int tiles_per_row = VP8LSubSampleSize(width, transform->bits_);
  int y = y_start;
  const uint32_t* pred_row =
      transform->data_ + (y >> transform->bits_) * tiles_per_row;

  while (y < y_end) {
    const uint32_t* pred = pred_row;
    Multipliers m = { 0, 0, 0 };
    int x;

    for (x = 0; x < width; ++x) {
      if ((x & mask) == 0) ColorCodeToMultipliers(*pred++, &m);
      data[x] = TransformColor(&m, data[x], 1);
    }
    data += width;
    ++y;
    if ((y & mask) == 0) pred_row += tiles_per_row;;
  }
}

// Separate out pixels packed together using pixel-bundling.
// We define two methods for ARGB data (uint32_t) and alpha-only data (uint8_t).
#define COLOR_INDEX_INVERSE(FUNC_NAME, TYPE, GET_INDEX, GET_VALUE)             \
void FUNC_NAME(const VP8LTransform* const transform,                           \
               int y_start, int y_end, const TYPE* src, TYPE* dst) {           \
  int y;                                                                       \
  const int bits_per_pixel = 8 >> transform->bits_;                            \
  const int width = transform->xsize_;                                         \
  const uint32_t* const color_map = transform->data_;                          \
  if (bits_per_pixel < 8) {                                                    \
    const int pixels_per_byte = 1 << transform->bits_;                         \
    const int count_mask = pixels_per_byte - 1;                                \
    const uint32_t bit_mask = (1 << bits_per_pixel) - 1;                       \
    for (y = y_start; y < y_end; ++y) {                                        \
      uint32_t packed_pixels = 0;                                              \
      int x;                                                                   \
      for (x = 0; x < width; ++x) {                                            \
        /* We need to load fresh 'packed_pixels' once every                */  \
        /* 'pixels_per_byte' increments of x. Fortunately, pixels_per_byte */  \
        /* is a power of 2, so can just use a mask for that, instead of    */  \
        /* decrementing a counter.                                         */  \
        if ((x & count_mask) == 0) packed_pixels = GET_INDEX(*src++);          \
        *dst++ = GET_VALUE(color_map[packed_pixels & bit_mask]);               \
        packed_pixels >>= bits_per_pixel;                                      \
      }                                                                        \
    }                                                                          \
  } else {                                                                     \
    for (y = y_start; y < y_end; ++y) {                                        \
      int x;                                                                   \
      for (x = 0; x < width; ++x) {                                            \
        *dst++ = GET_VALUE(color_map[GET_INDEX(*src++)]);                      \
      }                                                                        \
    }                                                                          \
  }                                                                            \
}

static WEBP_INLINE uint32_t GetARGBIndex(uint32_t idx) {
  return (idx >> 8) & 0xff;
}

static WEBP_INLINE uint8_t GetAlphaIndex(uint8_t idx) {
  return idx;
}

static WEBP_INLINE uint32_t GetARGBValue(uint32_t val) {
  return val;
}

static WEBP_INLINE uint8_t GetAlphaValue(uint32_t val) {
  return (val >> 8) & 0xff;
}

static COLOR_INDEX_INVERSE(ColorIndexInverseTransform, uint32_t, GetARGBIndex,
                           GetARGBValue)
COLOR_INDEX_INVERSE(VP8LColorIndexInverseTransformAlpha, uint8_t, GetAlphaIndex,
                    GetAlphaValue)

#undef COLOR_INDEX_INVERSE

void VP8LInverseTransform(const VP8LTransform* const transform,
                          int row_start, int row_end,
                          const uint32_t* const in, uint32_t* const out) {
  const int width = transform->xsize_;
  assert(row_start < row_end);
  assert(row_end <= transform->ysize_);
  switch (transform->type_) {
    case SUBTRACT_GREEN:
      VP8LAddGreenToBlueAndRed(out, out + (row_end - row_start) * width);
      break;
    case PREDICTOR_TRANSFORM:
      PredictorInverseTransform(transform, row_start, row_end, out);
      if (row_end != transform->ysize_) {
        // The last predicted row in this iteration will be the top-pred row
        // for the first row in next iteration.
        memcpy(out - width, out + (row_end - row_start - 1) * width,
               width * sizeof(*out));
      }
      break;
    case CROSS_COLOR_TRANSFORM:
      ColorSpaceInverseTransform(transform, row_start, row_end, out);
      break;
    case COLOR_INDEXING_TRANSFORM:
      if (in == out && transform->bits_ > 0) {
        // Move packed pixels to the end of unpacked region, so that unpacking
        // can occur seamlessly.
        // Also, note that this is the only transform that applies on
        // the effective width of VP8LSubSampleSize(xsize_, bits_). All other
        // transforms work on effective width of xsize_.
        const int out_stride = (row_end - row_start) * width;
        const int in_stride = (row_end - row_start) *
            VP8LSubSampleSize(transform->xsize_, transform->bits_);
        uint32_t* const src = out + out_stride - in_stride;
        memmove(src, out, in_stride * sizeof(*src));
        ColorIndexInverseTransform(transform, row_start, row_end, src, out);
      } else {
        ColorIndexInverseTransform(transform, row_start, row_end, in, out);
      }
      break;
  }
}

//------------------------------------------------------------------------------
// Color space conversion.

static int is_big_endian(void) {
  static const union {
    uint16_t w;
    uint8_t b[2];
  } tmp = { 1 };
  return (tmp.b[0] != 1);
}

static void ConvertBGRAToRGB(const uint32_t* src,
                             int num_pixels, uint8_t* dst) {
  const uint32_t* const src_end = src + num_pixels;
  while (src < src_end) {
    const uint32_t argb = *src++;
    *dst++ = (argb >> 16) & 0xff;
    *dst++ = (argb >>  8) & 0xff;
    *dst++ = (argb >>  0) & 0xff;
  }
}

static void ConvertBGRAToRGBA(const uint32_t* src,
                              int num_pixels, uint8_t* dst) {
  const uint32_t* const src_end = src + num_pixels;
  while (src < src_end) {
    const uint32_t argb = *src++;
    *dst++ = (argb >> 16) & 0xff;
    *dst++ = (argb >>  8) & 0xff;
    *dst++ = (argb >>  0) & 0xff;
    *dst++ = (argb >> 24) & 0xff;
  }
}

static void ConvertBGRAToRGBA4444(const uint32_t* src,
                                  int num_pixels, uint8_t* dst) {
  const uint32_t* const src_end = src + num_pixels;
  while (src < src_end) {
    const uint32_t argb = *src++;
    const uint8_t rg = ((argb >> 16) & 0xf0) | ((argb >> 12) & 0xf);
    const uint8_t ba = ((argb >>  0) & 0xf0) | ((argb >> 28) & 0xf);
#ifdef WEBP_SWAP_16BIT_CSP
    *dst++ = ba;
    *dst++ = rg;
#else
    *dst++ = rg;
    *dst++ = ba;
#endif
  }
}

static void ConvertBGRAToRGB565(const uint32_t* src,
                                int num_pixels, uint8_t* dst) {
  const uint32_t* const src_end = src + num_pixels;
  while (src < src_end) {
    const uint32_t argb = *src++;
    const uint8_t rg = ((argb >> 16) & 0xf8) | ((argb >> 13) & 0x7);
    const uint8_t gb = ((argb >>  5) & 0xe0) | ((argb >>  3) & 0x1f);
#ifdef WEBP_SWAP_16BIT_CSP
    *dst++ = gb;
    *dst++ = rg;
#else
    *dst++ = rg;
    *dst++ = gb;
#endif
  }
}

static void ConvertBGRAToBGR(const uint32_t* src,
                             int num_pixels, uint8_t* dst) {
  const uint32_t* const src_end = src + num_pixels;
  while (src < src_end) {
    const uint32_t argb = *src++;
    *dst++ = (argb >>  0) & 0xff;
    *dst++ = (argb >>  8) & 0xff;
    *dst++ = (argb >> 16) & 0xff;
  }
}

static void CopyOrSwap(const uint32_t* src, int num_pixels, uint8_t* dst,
                       int swap_on_big_endian) {
  if (is_big_endian() == swap_on_big_endian) {
    const uint32_t* const src_end = src + num_pixels;
    while (src < src_end) {
      uint32_t argb = *src++;

#if !defined(__BIG_ENDIAN__)
#if !defined(WEBP_REFERENCE_IMPLEMENTATION)
#if defined(__i386__) || defined(__x86_64__)
      __asm__ volatile("bswap %0" : "=r"(argb) : "0"(argb));
      *(uint32_t*)dst = argb;
#elif defined(_MSC_VER)
      argb = _byteswap_ulong(argb);
      *(uint32_t*)dst = argb;
#else
      dst[0] = (argb >> 24) & 0xff;
      dst[1] = (argb >> 16) & 0xff;
      dst[2] = (argb >>  8) & 0xff;
      dst[3] = (argb >>  0) & 0xff;
#endif
#else  // WEBP_REFERENCE_IMPLEMENTATION
      dst[0] = (argb >> 24) & 0xff;
      dst[1] = (argb >> 16) & 0xff;
      dst[2] = (argb >>  8) & 0xff;
      dst[3] = (argb >>  0) & 0xff;
#endif
#else  // __BIG_ENDIAN__
      dst[0] = (argb >>  0) & 0xff;
      dst[1] = (argb >>  8) & 0xff;
      dst[2] = (argb >> 16) & 0xff;
      dst[3] = (argb >> 24) & 0xff;
#endif
      dst += sizeof(argb);
    }
  } else {
    memcpy(dst, src, num_pixels * sizeof(*src));
  }
}

void VP8LConvertFromBGRA(const uint32_t* const in_data, int num_pixels,
                         WEBP_CSP_MODE out_colorspace, uint8_t* const rgba) {
  switch (out_colorspace) {
    case MODE_RGB:
      ConvertBGRAToRGB(in_data, num_pixels, rgba);
      break;
    case MODE_RGBA:
      ConvertBGRAToRGBA(in_data, num_pixels, rgba);
      break;
    case MODE_rgbA:
      ConvertBGRAToRGBA(in_data, num_pixels, rgba);
      WebPApplyAlphaMultiply(rgba, 0, num_pixels, 1, 0);
      break;
    case MODE_BGR:
      ConvertBGRAToBGR(in_data, num_pixels, rgba);
      break;
    case MODE_BGRA:
      CopyOrSwap(in_data, num_pixels, rgba, 1);
      break;
    case MODE_bgrA:
      CopyOrSwap(in_data, num_pixels, rgba, 1);
      WebPApplyAlphaMultiply(rgba, 0, num_pixels, 1, 0);
      break;
    case MODE_ARGB:
      CopyOrSwap(in_data, num_pixels, rgba, 0);
      break;
    case MODE_Argb:
      CopyOrSwap(in_data, num_pixels, rgba, 0);
      WebPApplyAlphaMultiply(rgba, 1, num_pixels, 1, 0);
      break;
    case MODE_RGBA_4444:
      ConvertBGRAToRGBA4444(in_data, num_pixels, rgba);
      break;
    case MODE_rgbA_4444:
      ConvertBGRAToRGBA4444(in_data, num_pixels, rgba);
      WebPApplyAlphaMultiply4444(rgba, num_pixels, 1, 0);
      break;
    case MODE_RGB_565:
      ConvertBGRAToRGB565(in_data, num_pixels, rgba);
      break;
    default:
      assert(0);          // Code flow should not reach here.
  }
}

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

// TODO(vikasa): Move the SSE2 functions to lossless_dsp.c (new file), once
// color-space conversion methods (ConvertFromBGRA) are also updated for SSE2.
#if defined(WEBP_USE_SSE2)
static WEBP_INLINE uint32_t ClampedAddSubtractFullSSE2(uint32_t c0, uint32_t c1,
                                                       uint32_t c2) {
  const __m128i zero = _mm_setzero_si128();
  const __m128i C0 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(c0), zero);
  const __m128i C1 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(c1), zero);
  const __m128i C2 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(c2), zero);
  const __m128i V1 = _mm_add_epi16(C0, C1);
  const __m128i V2 = _mm_sub_epi16(V1, C2);
  const __m128i b = _mm_packus_epi16(V2, V2);
  const uint32_t output = _mm_cvtsi128_si32(b);
  return output;
}

static WEBP_INLINE uint32_t ClampedAddSubtractHalfSSE2(uint32_t c0, uint32_t c1,
                                                       uint32_t c2) {
  const uint32_t ave = Average2(c0, c1);
  const __m128i zero = _mm_setzero_si128();
  const __m128i A0 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(ave), zero);
  const __m128i B0 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(c2), zero);
  const __m128i A1 = _mm_sub_epi16(A0, B0);
  const __m128i BgtA = _mm_cmpgt_epi16(B0, A0);
  const __m128i A2 = _mm_sub_epi16(A1, BgtA);
  const __m128i A3 = _mm_srai_epi16(A2, 1);
  const __m128i A4 = _mm_add_epi16(A0, A3);
  const __m128i A5 = _mm_packus_epi16(A4, A4);
  const uint32_t output = _mm_cvtsi128_si32(A5);
  return output;
}

static WEBP_INLINE uint32_t SelectSSE2(uint32_t a, uint32_t b, uint32_t c) {
  int pa_minus_pb;
  const __m128i zero = _mm_setzero_si128();
  const __m128i A0 = _mm_cvtsi32_si128(a);
  const __m128i B0 = _mm_cvtsi32_si128(b);
  const __m128i C0 = _mm_cvtsi32_si128(c);
  const __m128i AC0 = _mm_subs_epu8(A0, C0);
  const __m128i CA0 = _mm_subs_epu8(C0, A0);
  const __m128i BC0 = _mm_subs_epu8(B0, C0);
  const __m128i CB0 = _mm_subs_epu8(C0, B0);
  const __m128i AC = _mm_or_si128(AC0, CA0);
  const __m128i BC = _mm_or_si128(BC0, CB0);
  const __m128i pa = _mm_unpacklo_epi8(AC, zero);  // |a - c|
  const __m128i pb = _mm_unpacklo_epi8(BC, zero);  // |b - c|
  const __m128i diff = _mm_sub_epi16(pb, pa);
  {
    int16_t out[8];
    _mm_storeu_si128((__m128i*)out, diff);
    pa_minus_pb = out[0] + out[1] + out[2] + out[3];
  }
  return (pa_minus_pb <= 0) ? a : b;
}

static void SubtractGreenFromBlueAndRedSSE2(uint32_t* argb_data, int num_pixs) {
  int i = 0;
  const __m128i mask = _mm_set1_epi32(0x0000ff00);
  for (; i + 4 < num_pixs; i += 4) {
    const __m128i in = _mm_loadu_si128((__m128i*)&argb_data[i]);
    const __m128i in_00g0 = _mm_and_si128(in, mask);     // 00g0|00g0|...
    const __m128i in_0g00 = _mm_slli_epi32(in_00g0, 8);  // 0g00|0g00|...
    const __m128i in_000g = _mm_srli_epi32(in_00g0, 8);  // 000g|000g|...
    const __m128i in_0g0g = _mm_or_si128(in_0g00, in_000g);
    const __m128i out = _mm_sub_epi8(in, in_0g0g);
    _mm_storeu_si128((__m128i*)&argb_data[i], out);
  }
  // fallthrough and finish off with plain-C
  for (; i < num_pixs; ++i) {
    const uint32_t argb = argb_data[i];
    const uint32_t green = (argb >> 8) & 0xff;
    const uint32_t new_r = (((argb >> 16) & 0xff) - green) & 0xff;
    const uint32_t new_b = ((argb & 0xff) - green) & 0xff;
    argb_data[i] = (argb & 0xff00ff00) | (new_r << 16) | new_b;
  }
}

static void AddGreenToBlueAndRedSSE2(uint32_t* data, const uint32_t* data_end) {
  const __m128i mask = _mm_set1_epi32(0x0000ff00);
  for (; data + 4 < data_end; data += 4) {
    const __m128i in = _mm_loadu_si128((__m128i*)data);
    const __m128i in_00g0 = _mm_and_si128(in, mask);     // 00g0|00g0|...
    const __m128i in_0g00 = _mm_slli_epi32(in_00g0, 8);  // 0g00|0g00|...
    const __m128i in_000g = _mm_srli_epi32(in_00g0, 8);  // 000g|000g|...
    const __m128i in_0g0g = _mm_or_si128(in_0g00, in_000g);
    const __m128i out = _mm_add_epi8(in, in_0g0g);
    _mm_storeu_si128((__m128i*)data, out);
  }
  // fallthrough and finish off with plain-C
  while (data < data_end) {
    const uint32_t argb = *data;
    const uint32_t green = ((argb >> 8) & 0xff);
    uint32_t red_blue = (argb & 0x00ff00ffu);
    red_blue += (green << 16) | green;
    red_blue &= 0x00ff00ffu;
    *data++ = (argb & 0xff00ff00u) | red_blue;
  }
}

extern void VP8LDspInitSSE2(void);

void VP8LDspInitSSE2(void) {
  VP8LClampedAddSubtractFull = ClampedAddSubtractFullSSE2;
  VP8LClampedAddSubtractHalf = ClampedAddSubtractHalfSSE2;
  VP8LSelect = SelectSSE2;
  VP8LSubtractGreenFromBlueAndRed = SubtractGreenFromBlueAndRedSSE2;
  VP8LAddGreenToBlueAndRed = AddGreenToBlueAndRedSSE2;
}
#endif
//------------------------------------------------------------------------------

VP8LPredClampedAddSubFunc VP8LClampedAddSubtractFull;
VP8LPredClampedAddSubFunc VP8LClampedAddSubtractHalf;
VP8LPredSelectFunc VP8LSelect;
VP8LSubtractGreenFromBlueAndRedFunc VP8LSubtractGreenFromBlueAndRed;
VP8LAddGreenToBlueAndRedFunc VP8LAddGreenToBlueAndRed;

void VP8LDspInit(void) {
  VP8LClampedAddSubtractFull = ClampedAddSubtractFull;
  VP8LClampedAddSubtractHalf = ClampedAddSubtractHalf;
  VP8LSelect = Select;
  VP8LSubtractGreenFromBlueAndRed = SubtractGreenFromBlueAndRed;
  VP8LAddGreenToBlueAndRed = AddGreenToBlueAndRed;

  // If defined, use CPUInfo() to overwrite some pointers with faster versions.
  if (VP8GetCPUInfo != NULL) {
#if defined(WEBP_USE_SSE2)
    if (VP8GetCPUInfo(kSSE2)) {
      VP8LDspInitSSE2();
    }
#endif
  }
}

//------------------------------------------------------------------------------

