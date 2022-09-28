/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggTheora SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE Theora SOURCE CODE IS COPYRIGHT (C) 2002-2009                *
 * by the Xiph.Org Foundation http://www.xiph.org/                  *
 *                                                                  *
 ********************************************************************

  function:
  last mod: $Id$

 ********************************************************************/
#include <stdlib.h>
#include <string.h>
#include "encint.h"



static unsigned char OC_DCT_EOB_TOKEN[31]={
  0,1,2,3,3,3,3,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5
};

static int oc_make_eob_token(int _run_count){
  return _run_count<32?OC_DCT_EOB_TOKEN[_run_count-1]:OC_DCT_REPEAT_RUN3_TOKEN;
}

static unsigned char OC_DCT_EOB_EB[31]={
  0,0,0,0,1,2,3,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
};

static int oc_make_eob_token_full(int _run_count,int *_eb){
  if(_run_count<32){
    *_eb=OC_DCT_EOB_EB[_run_count-1];
    return OC_DCT_EOB_TOKEN[_run_count-1];
  }
  else{
    *_eb=_run_count;
    return OC_DCT_REPEAT_RUN3_TOKEN;
  }
}

/*Returns the number of blocks ended by an EOB token.*/
static int oc_decode_eob_token(int _token,int _eb){
  return (0x20820C41U>>_token*5&0x1F)+_eb;
}

/*Some tables for fast construction of value tokens.*/

static const unsigned char OC_DCT_VALUE_TOKEN[1161]={
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,21,21,21,21,21,21,21,21,
  21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,
  21,21,21,21,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,
  19,19,19,19,19,19,19,19,18,18,18,18,17,17,16,15,14,13,12,10,
   7,
   9,11,13,14,15,16,17,17,18,18,18,18,19,19,19,19,19,19,19,19,
  20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,21,21,21,21,
  21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,
  21,21,21,21,21,21,21,21,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,
  22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22
};

static const ogg_uint16_t OC_DCT_VALUE_EB[1161]={
  1023,1022,1021,1020,1019,1018,1017,1016,1015,1014,
  1013,1012,1011,1010,1009,1008,1007,1006,1005,1004,
  1003,1002,1001,1000, 999, 998, 997, 996, 995, 994,
   993, 992, 991, 990, 989, 988, 987, 986, 985, 984,
   983, 982, 981, 980, 979, 978, 977, 976, 975, 974,
   973, 972, 971, 970, 969, 968, 967, 966, 965, 964,
   963, 962, 961, 960, 959, 958, 957, 956, 955, 954,
   953, 952, 951, 950, 949, 948, 947, 946, 945, 944,
   943, 942, 941, 940, 939, 938, 937, 936, 935, 934,
   933, 932, 931, 930, 929, 928, 927, 926, 925, 924,
   923, 922, 921, 920, 919, 918, 917, 916, 915, 914,
   913, 912, 911, 910, 909, 908, 907, 906, 905, 904,
   903, 902, 901, 900, 899, 898, 897, 896, 895, 894,
   893, 892, 891, 890, 889, 888, 887, 886, 885, 884,
   883, 882, 881, 880, 879, 878, 877, 876, 875, 874,
   873, 872, 871, 870, 869, 868, 867, 866, 865, 864,
   863, 862, 861, 860, 859, 858, 857, 856, 855, 854,
   853, 852, 851, 850, 849, 848, 847, 846, 845, 844,
   843, 842, 841, 840, 839, 838, 837, 836, 835, 834,
   833, 832, 831, 830, 829, 828, 827, 826, 825, 824,
   823, 822, 821, 820, 819, 818, 817, 816, 815, 814,
   813, 812, 811, 810, 809, 808, 807, 806, 805, 804,
   803, 802, 801, 800, 799, 798, 797, 796, 795, 794,
   793, 792, 791, 790, 789, 788, 787, 786, 785, 784,
   783, 782, 781, 780, 779, 778, 777, 776, 775, 774,
   773, 772, 771, 770, 769, 768, 767, 766, 765, 764,
   763, 762, 761, 760, 759, 758, 757, 756, 755, 754,
   753, 752, 751, 750, 749, 748, 747, 746, 745, 744,
   743, 742, 741, 740, 739, 738, 737, 736, 735, 734,
   733, 732, 731, 730, 729, 728, 727, 726, 725, 724,
   723, 722, 721, 720, 719, 718, 717, 716, 715, 714,
   713, 712, 711, 710, 709, 708, 707, 706, 705, 704,
   703, 702, 701, 700, 699, 698, 697, 696, 695, 694,
   693, 692, 691, 690, 689, 688, 687, 686, 685, 684,
   683, 682, 681, 680, 679, 678, 677, 676, 675, 674,
   673, 672, 671, 670, 669, 668, 667, 666, 665, 664,
   663, 662, 661, 660, 659, 658, 657, 656, 655, 654,
   653, 652, 651, 650, 649, 648, 647, 646, 645, 644,
   643, 642, 641, 640, 639, 638, 637, 636, 635, 634,
   633, 632, 631, 630, 629, 628, 627, 626, 625, 624,
   623, 622, 621, 620, 619, 618, 617, 616, 615, 614,
   613, 612, 611, 610, 609, 608, 607, 606, 605, 604,
   603, 602, 601, 600, 599, 598, 597, 596, 595, 594,
   593, 592, 591, 590, 589, 588, 587, 586, 585, 584,
   583, 582, 581, 580, 579, 578, 577, 576, 575, 574,
   573, 572, 571, 570, 569, 568, 567, 566, 565, 564,
   563, 562, 561, 560, 559, 558, 557, 556, 555, 554,
   553, 552, 551, 550, 549, 548, 547, 546, 545, 544,
   543, 542, 541, 540, 539, 538, 537, 536, 535, 534,
   533, 532, 531, 530, 529, 528, 527, 526, 525, 524,
   523, 522, 521, 520, 519, 518, 517, 516, 515, 514,
   513, 512,  63,  62,  61,  60,  59,  58,  57,  56,
    55,  54,  53,  52,  51,  50,  49,  48,  47,  46,
    45,  44,  43,  42,  41,  40,  39,  38,  37,  36,
    35,  34,  33,  32,  31,  30,  29,  28,  27,  26,
    25,  24,  23,  22,  21,  20,  19,  18,  17,  16,
    15,  14,  13,  12,  11,  10,   9,   8,   7,   6,
     5,   4,   3,   2,   1,   1,   1,   1,   0,   0,
     0,
     0,   0,   0,   0,   0,   0,   0,   1,   0,   1,
     2,   3,   0,   1,   2,   3,   4,   5,   6,   7,
     0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
    10,  11,  12,  13,  14,  15,   0,   1,   2,   3,
     4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
    14,  15,  16,  17,  18,  19,  20,  21,  22,  23,
    24,  25,  26,  27,  28,  29,  30,  31,   0,   1,
     2,   3,   4,   5,   6,   7,   8,   9,  10,  11,
    12,  13,  14,  15,  16,  17,  18,  19,  20,  21,
    22,  23,  24,  25,  26,  27,  28,  29,  30,  31,
    32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
    42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
    52,  53,  54,  55,  56,  57,  58,  59,  60,  61,
    62,  63,  64,  65,  66,  67,  68,  69,  70,  71,
    72,  73,  74,  75,  76,  77,  78,  79,  80,  81,
    82,  83,  84,  85,  86,  87,  88,  89,  90,  91,
    92,  93,  94,  95,  96,  97,  98,  99, 100, 101,
   102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
   112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
   122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
   132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
   142, 143, 144, 145, 146, 147, 148, 149, 150, 151,
   152, 153, 154, 155, 156, 157, 158, 159, 160, 161,
   162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
   172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
   182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
   192, 193, 194, 195, 196, 197, 198, 199, 200, 201,
   202, 203, 204, 205, 206, 207, 208, 209, 210, 211,
   212, 213, 214, 215, 216, 217, 218, 219, 220, 221,
   222, 223, 224, 225, 226, 227, 228, 229, 230, 231,
   232, 233, 234, 235, 236, 237, 238, 239, 240, 241,
   242, 243, 244, 245, 246, 247, 248, 249, 250, 251,
   252, 253, 254, 255, 256, 257, 258, 259, 260, 261,
   262, 263, 264, 265, 266, 267, 268, 269, 270, 271,
   272, 273, 274, 275, 276, 277, 278, 279, 280, 281,
   282, 283, 284, 285, 286, 287, 288, 289, 290, 291,
   292, 293, 294, 295, 296, 297, 298, 299, 300, 301,
   302, 303, 304, 305, 306, 307, 308, 309, 310, 311,
   312, 313, 314, 315, 316, 317, 318, 319, 320, 321,
   322, 323, 324, 325, 326, 327, 328, 329, 330, 331,
   332, 333, 334, 335, 336, 337, 338, 339, 340, 341,
   342, 343, 344, 345, 346, 347, 348, 349, 350, 351,
   352, 353, 354, 355, 356, 357, 358, 359, 360, 361,
   362, 363, 364, 365, 366, 367, 368, 369, 370, 371,
   372, 373, 374, 375, 376, 377, 378, 379, 380, 381,
   382, 383, 384, 385, 386, 387, 388, 389, 390, 391,
   392, 393, 394, 395, 396, 397, 398, 399, 400, 401,
   402, 403, 404, 405, 406, 407, 408, 409, 410, 411,
   412, 413, 414, 415, 416, 417, 418, 419, 420, 421,
   422, 423, 424, 425, 426, 427, 428, 429, 430, 431,
   432, 433, 434, 435, 436, 437, 438, 439, 440, 441,
   442, 443, 444, 445, 446, 447, 448, 449, 450, 451,
   452, 453, 454, 455, 456, 457, 458, 459, 460, 461,
   462, 463, 464, 465, 466, 467, 468, 469, 470, 471,
   472, 473, 474, 475, 476, 477, 478, 479, 480, 481,
   482, 483, 484, 485, 486, 487, 488, 489, 490, 491,
   492, 493, 494, 495, 496, 497, 498, 499, 500, 501,
   502, 503, 504, 505, 506, 507, 508, 509, 510, 511
};

/*The first DCT coefficient that both has a smaller magnitude and gets coded
   with a different token.*/
static const ogg_int16_t OC_DCT_TRELLIS_ALT_VALUE[1161]={
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -68, -68, -68, -68, -68, -68, -68, -68,
   -68, -68, -36, -36, -36, -36, -36, -36, -36, -36,
   -36, -36, -36, -36, -36, -36, -36, -36, -36, -36,
   -36, -36, -36, -36, -36, -36, -36, -36, -36, -36,
   -36, -36, -36, -36, -20, -20, -20, -20, -20, -20,
   -20, -20, -20, -20, -20, -20, -20, -20, -20, -20,
   -12, -12, -12, -12, -12, -12, -12, -12,  -8,  -8,
    -8,  -8,  -6,  -6,  -5,  -4,  -3,  -2,  -1,   0,
     0,
     0,   1,   2,   3,   4,   5,   6,   6,   8,   8,
     8,   8,  12,  12,  12,  12,  12,  12,  12,  12,
    20,  20,  20,  20,  20,  20,  20,  20,  20,  20,
    20,  20,  20,  20,  20,  20,  36,  36,  36,  36,
    36,  36,  36,  36,  36,  36,  36,  36,  36,  36,
    36,  36,  36,  36,  36,  36,  36,  36,  36,  36,
    36,  36,  36,  36,  36,  36,  36,  36,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68,
    68,  68,  68,  68,  68,  68,  68,  68,  68,  68
};

#define OC_DCT_VALUE_TOKEN_PTR (OC_DCT_VALUE_TOKEN+580)
#define OC_DCT_VALUE_EB_PTR (OC_DCT_VALUE_EB+580)
#define OC_DCT_TRELLIS_ALT_VALUE_PTR (OC_DCT_TRELLIS_ALT_VALUE+580)

/*Some tables for fast construction of combo tokens.*/

static const unsigned char OC_DCT_RUN_CAT1_TOKEN[17]={
  23,24,25,26,27,28,28,28,28,29,29,29,29,29,29,29,29
};

static const unsigned char OC_DCT_RUN_CAT1_EB[17][2]={
  {0,1},{0,1},{0, 1},{0, 1},{0, 1},{0, 4},{1, 5},{2, 6},{3,7},
  {0,8},{1,9},{2,10},{3,11},{4,12},{5,13},{6,14},{7,15}
};

static const unsigned char OC_DCT_RUN_CAT2_EB[3][2][2]={
  { {0,1},{2,3} },{ {0,2},{4,6} },{ {1,3},{5,7} }
};

/*Token logging to allow a few fragments of efficient rollback.
  Late SKIP analysis is tied up in the tokenization process, so we need to be
   able to undo a fragment's tokens on a whim.*/

static const unsigned char OC_ZZI_HUFF_OFFSET[64]={
   0,16,16,16,16,16,32,32,
  32,32,32,32,32,32,32,48,
  48,48,48,48,48,48,48,48,
  48,48,48,48,64,64,64,64,
  64,64,64,64,64,64,64,64,
  64,64,64,64,64,64,64,64,
  64,64,64,64,64,64,64,64
};

static int oc_token_bits(oc_enc_ctx *_enc,int _huffi,int _zzi,int _token){
  return _enc->huff_codes[_huffi+OC_ZZI_HUFF_OFFSET[_zzi]][_token].nbits
   +OC_DCT_TOKEN_EXTRA_BITS[_token];
}

static void oc_enc_tokenlog_checkpoint(oc_enc_ctx *_enc,
 oc_token_checkpoint *_cp,int _pli,int _zzi){
  _cp->pli=_pli;
  _cp->zzi=_zzi;
  _cp->eob_run=_enc->eob_run[_pli][_zzi];
  _cp->ndct_tokens=_enc->ndct_tokens[_pli][_zzi];
}

void oc_enc_tokenlog_rollback(oc_enc_ctx *_enc,
 const oc_token_checkpoint *_stack,int _n){
  int i;
  for(i=_n;i-->0;){
    int pli;
    int zzi;
    pli=_stack[i].pli;
    zzi=_stack[i].zzi;
    _enc->eob_run[pli][zzi]=_stack[i].eob_run;
    _enc->ndct_tokens[pli][zzi]=_stack[i].ndct_tokens;
  }
}

static void oc_enc_token_log(oc_enc_ctx *_enc,
 int _pli,int _zzi,int _token,int _eb){
  ptrdiff_t ti;
  ti=_enc->ndct_tokens[_pli][_zzi]++;
  _enc->dct_tokens[_pli][_zzi][ti]=(unsigned char)_token;
  _enc->extra_bits[_pli][_zzi][ti]=(ogg_uint16_t)_eb;
}

static void oc_enc_eob_log(oc_enc_ctx *_enc,
 int _pli,int _zzi,int _run_count){
  int token;
  int eb;
  token=oc_make_eob_token_full(_run_count,&eb);
  oc_enc_token_log(_enc,_pli,_zzi,token,eb);
}


void oc_enc_tokenize_start(oc_enc_ctx *_enc){
  memset(_enc->ndct_tokens,0,sizeof(_enc->ndct_tokens));
  memset(_enc->eob_run,0,sizeof(_enc->eob_run));
  memset(_enc->dct_token_offs,0,sizeof(_enc->dct_token_offs));
  memset(_enc->dc_pred_last,0,sizeof(_enc->dc_pred_last));
}

typedef struct oc_quant_token oc_quant_token;

/*A single node in the Viterbi trellis.
  We maintain up to 2 of these per coefficient:
    - A token to code if the value is zero (EOB, zero run, or combo token).
    - A token to code if the value is not zero (DCT value token).*/
struct oc_quant_token{
  unsigned char next;
  signed char   token;
  ogg_int16_t   eb;
  ogg_uint32_t  cost;
  int           bits;
  int           qc;
};

/*Tokenizes the AC coefficients, possibly adjusting the quantization, and then
   dequantizes and de-zig-zags the result.
  The AC coefficients of _idct must be pre-initialized to zero.*/
int oc_enc_tokenize_ac(oc_enc_ctx *_enc,int _pli,ptrdiff_t _fragi,
 ogg_int16_t *_idct,const ogg_int16_t *_qdct,
 const ogg_uint16_t *_dequant,const ogg_int16_t *_dct,
 int _zzi,oc_token_checkpoint **_stack,int _lambda,int _acmin){
  oc_token_checkpoint *stack;
  ogg_int64_t          zflags;
  ogg_int64_t          nzflags;
  ogg_int64_t          best_flags;
  ogg_uint32_t         d2_accum[64];
  oc_quant_token       tokens[64][2];
  ogg_uint16_t        *eob_run;
  const unsigned char *dct_fzig_zag;
  ogg_uint32_t         cost;
  int                  bits;
  int                  eob;
  int                  token;
  int                  eb;
  int                  next;
  int                  huffi;
  int                  zzi;
  int                  ti;
  int                  zzj;
  int                  qc;
  huffi=_enc->huff_idxs[_enc->state.frame_type][1][_pli+1>>1];
  eob_run=_enc->eob_run[_pli];
  memset(tokens[0],0,sizeof(tokens[0]));
  best_flags=nzflags=0;
  zflags=1;
  d2_accum[0]=0;
  zzj=64;
  for(zzi=OC_MINI(_zzi,63);zzi>0;zzi--){
    ogg_uint32_t best_cost;
    int          best_bits=best_bits;
    int          best_next=best_next;
    int          best_token=best_token;
    int          best_eb=best_eb;
    int          best_qc=best_qc;
    ogg_uint32_t d2;
    int          dq;
    int          qc_m;
    int          e;
    int          c;
    int          s;
    int          tj;
    qc=_qdct[zzi];
    s=-(qc<0);
    qc_m=qc+s^s;
    c=_dct[zzi];
    /*The hard case: try a zero run.*/
    if(qc_m<=1){
      ogg_uint32_t sum_d2;
      int          nzeros;
      int          dc_reserve;
      if(!qc_m){
        /*Skip runs that are already quantized to zeros.
          If we considered each zero coefficient in turn, we might
           theoretically find a better way to partition long zero runs (e.g.,
           a run of > 17 zeros followed by a 1 might be better coded as a short
           zero run followed by a combo token, rather than the longer zero
           token followed by a 1 value token), but zeros are so common that
           this becomes very computationally expensive (quadratic instead of
           linear in the number of coefficients), for a marginal gain.*/
        while(zzi>1&&!_qdct[zzi-1])zzi--;
        /*The distortion of coefficients originally quantized to zero is
           treated as zero (since we'll never quantize them to anything else).*/
        d2=0;
      }
      else{
        d2=c*(ogg_int32_t)c;
        c=c+s^s;
      }
      eob=eob_run[zzi];
      nzeros=zzj-zzi;
      zzj&=63;
      sum_d2=d2+d2_accum[zzj];
      d2_accum[zzi]=sum_d2;
      /*We reserve 1 spot for combo run tokens that start in the 1st AC stack
         to ensure they can be extended to include the DC coefficient if
         necessary; this greatly simplifies stack-rewriting later on.*/
      dc_reserve=zzi+62>>6;
      best_cost=0xFFFFFFFF;
      for(;;){
        if(nzflags>>zzj&1){
          int val;
          int val_s;
          int zzk;
          int tk;
          next=tokens[zzj][1].next;
          tk=next&1;
          zzk=next>>1;
          /*Try a pure zero run to this point.*/
          token=OC_DCT_SHORT_ZRL_TOKEN+(nzeros+55>>6);
          bits=oc_token_bits(_enc,huffi,zzi,token);
          d2=sum_d2-d2_accum[zzj];
          cost=d2+_lambda*bits+tokens[zzj][1].cost;
          if(cost<=best_cost){
            best_next=(zzj<<1)+1;
            best_token=token;
            best_eb=nzeros-1;
            best_cost=cost;
            best_bits=bits+tokens[zzj][1].bits;
            best_qc=0;
          }
          if(nzeros<17+dc_reserve){
            val=_qdct[zzj];
            val_s=-(val<0);
            val=val+val_s^val_s;
            if(val<=2){
              /*Try a +/- 1 combo token.*/
              token=OC_DCT_RUN_CAT1_TOKEN[nzeros-1];
              eb=OC_DCT_RUN_CAT1_EB[nzeros-1][-val_s];
              e=_dct[zzj]-(_dequant[zzj]+val_s^val_s);
              d2=e*(ogg_int32_t)e+sum_d2-d2_accum[zzj];
              bits=oc_token_bits(_enc,huffi,zzi,token);
              cost=d2+_lambda*bits+tokens[zzk][tk].cost;
              if(cost<=best_cost){
                best_next=next;
                best_token=token;
                best_eb=eb;
                best_cost=cost;
                best_bits=bits+tokens[zzk][tk].bits;
                best_qc=1+val_s^val_s;
              }
            }
            if(nzeros<3+dc_reserve&&2<=val&&val<=4){
              int sval;
              /*Try a +/- 2/3 combo token.*/
              token=OC_DCT_RUN_CAT2A+(nzeros>>1);
              bits=oc_token_bits(_enc,huffi,zzi,token);
              val=2+(val>2);
              sval=val+val_s^val_s;
              e=_dct[zzj]-_dequant[zzj]*sval;
              d2=e*(ogg_int32_t)e+sum_d2-d2_accum[zzj];
              cost=d2+_lambda*bits+tokens[zzk][tk].cost;
              if(cost<=best_cost){
                best_cost=cost;
                best_bits=bits+tokens[zzk][tk].bits;
                best_next=next;
                best_token=token;
                best_eb=OC_DCT_RUN_CAT2_EB[nzeros-1][-val_s][val-2];
                best_qc=sval;
              }
            }
          }
          /*zzj can't be coded as a zero, so stop trying to extend the run.*/
          if(!(zflags>>zzj&1))break;
        }
        /*We could try to consider _all_ potentially non-zero coefficients, but
           if we already found a bunch of them not worth coding, it's fairly
           unlikely they would now be worth coding from this position; skipping
           them saves a lot of work.*/
        zzj=(tokens[zzj][0].next>>1)-(tokens[zzj][0].qc!=0)&63;
        if(zzj==0){
          /*We made it all the way to the end of the block; try an EOB token.*/
          if(eob<4095){
            bits=oc_token_bits(_enc,huffi,zzi,oc_make_eob_token(eob+1))
             -(eob>0?oc_token_bits(_enc,huffi,zzi,oc_make_eob_token(eob)):0);
          }
          else bits=oc_token_bits(_enc,huffi,zzi,OC_DCT_EOB1_TOKEN);
          cost=sum_d2+bits*_lambda;
          /*If the best route so far is still a pure zero run to the end of the
             block, force coding it as an EOB.
            Even if it's not optimal for this block, it has a good chance of
             getting combined with an EOB token from subsequent blocks, saving
             bits overall.*/
          if(cost<=best_cost||best_token<=OC_DCT_ZRL_TOKEN&&zzi+best_eb==63){
            best_next=0;
            /*This token is just a marker; in reality we may not emit any
               tokens, but update eob_run[] instead.*/
            best_token=OC_DCT_EOB1_TOKEN;
            best_eb=0;
            best_cost=cost;
            best_bits=bits;
            best_qc=0;
          }
          break;
        }
        nzeros=zzj-zzi;
      }
      tokens[zzi][0].next=(unsigned char)best_next;
      tokens[zzi][0].token=(signed char)best_token;
      tokens[zzi][0].eb=(ogg_int16_t)best_eb;
      tokens[zzi][0].cost=best_cost;
      tokens[zzi][0].bits=best_bits;
      tokens[zzi][0].qc=best_qc;
      zflags|=(ogg_int64_t)1<<zzi;
      if(qc_m){
        dq=_dequant[zzi];
        if(zzi<_acmin)_lambda=0;
        e=dq-c;
        d2=e*(ogg_int32_t)e;
        token=OC_ONE_TOKEN-s;
        bits=oc_token_bits(_enc,huffi,zzi,token);
        zzj=zzi+1&63;
        tj=best_flags>>zzj&1;
        next=(zzj<<1)+tj;
        tokens[zzi][1].next=(unsigned char)next;
        tokens[zzi][1].token=(signed char)token;
        tokens[zzi][1].eb=0;
        tokens[zzi][1].cost=d2+_lambda*bits+tokens[zzj][tj].cost;
        tokens[zzi][1].bits=bits+tokens[zzj][tj].bits;
        tokens[zzi][1].qc=1+s^s;
        nzflags|=(ogg_int64_t)1<<zzi;
        best_flags|=
         (ogg_int64_t)(tokens[zzi][1].cost<tokens[zzi][0].cost)<<zzi;
      }
    }
    else{
      int alt_qc;
      eob=eob_run[zzi];
      if(zzi<_acmin)_lambda=0;
      dq=_dequant[zzi];
      /*No zero run can extend past this point.*/
      d2_accum[zzi]=0;
      e=qc*dq-c;
      d2=e*(ogg_int32_t)e;
      best_token=*(OC_DCT_VALUE_TOKEN_PTR+qc);
      best_bits=oc_token_bits(_enc,huffi,zzi,best_token);
      best_cost=d2+_lambda*best_bits;
      alt_qc=*(OC_DCT_TRELLIS_ALT_VALUE_PTR+qc);
      e=alt_qc*dq-c;
      d2=e*(ogg_int32_t)e;
      token=*(OC_DCT_VALUE_TOKEN_PTR+alt_qc);
      bits=oc_token_bits(_enc,huffi,zzi,token);
      cost=d2+_lambda*bits;
      if(cost<best_cost){
        best_token=token;
        best_bits=bits;
        best_cost=cost;
        qc=alt_qc;
      }
      zzj=zzi+1&63;
      tj=best_flags>>zzj&1;
      next=(zzj<<1)+tj;
      tokens[zzi][1].next=(unsigned char)next;
      tokens[zzi][1].token=(signed char)best_token;
      tokens[zzi][1].eb=*(OC_DCT_VALUE_EB_PTR+qc);
      tokens[zzi][1].cost=best_cost+tokens[zzj][tj].cost;
      tokens[zzi][1].bits=best_bits+tokens[zzj][tj].bits;
      tokens[zzi][1].qc=qc;
      nzflags|=(ogg_int64_t)1<<zzi;
      best_flags|=(ogg_int64_t)1<<zzi;
    }
    zzj=zzi;
  }
  /*Emit the tokens from the best path through the trellis.*/
  stack=*_stack;
  dct_fzig_zag=_enc->state.opt_data.dct_fzig_zag;
  zzi=1;
  ti=best_flags>>1&1;
  bits=tokens[zzi][ti].bits;
  do{
    oc_enc_tokenlog_checkpoint(_enc,stack++,_pli,zzi);
    eob=eob_run[zzi];
    if(tokens[zzi][ti].token<OC_NDCT_EOB_TOKEN_MAX){
      if(++eob>=4095){
        oc_enc_token_log(_enc,_pli,zzi,OC_DCT_REPEAT_RUN3_TOKEN,eob);
        eob=0;
      }
      eob_run[zzi]=eob;
      /*We don't include the actual EOB cost for this block in the return value.
        It is very likely to eventually be spread over several blocks, and
         including it more harshly penalizes the first few blocks in a long EOB
         run.
        Omitting it here gives a small PSNR and SSIM gain.*/
      bits-=tokens[zzi][ti].bits;
      zzi=_zzi;
      break;
    }
    /*Emit pending EOB run if any.*/
    if(eob>0){
      oc_enc_eob_log(_enc,_pli,zzi,eob);
      eob_run[zzi]=0;
    }
    oc_enc_token_log(_enc,_pli,zzi,tokens[zzi][ti].token,tokens[zzi][ti].eb);
    next=tokens[zzi][ti].next;
    qc=tokens[zzi][ti].qc;
    zzj=(next>>1)-1&63;
    /*TODO: It may be worth saving the dequantized coefficient in the trellis
       above; we had to compute it to measure the error anyway.*/
    _idct[dct_fzig_zag[zzj]]=(ogg_int16_t)(qc*(int)_dequant[zzj]);
    zzi=next>>1;
    ti=next&1;
  }
  while(zzi);
  *_stack=stack;
  return bits;
}

/*Simplistic R/D tokenizer.
  The AC coefficients of _idct must be pre-initialized to zero.
  This could be made more accurate by using more sophisticated
   rate predictions for zeros.
  It could be made faster by switching from R/D decisions to static
   lambda-derived rounding biases.*/
int oc_enc_tokenize_ac_fast(oc_enc_ctx *_enc,int _pli,ptrdiff_t _fragi,
 ogg_int16_t *_idct,const ogg_int16_t *_qdct,
 const ogg_uint16_t *_dequant,const ogg_int16_t *_dct,
 int _zzi,oc_token_checkpoint **_stack,int _lambda,int _acmin){
  const unsigned char *dct_fzig_zag;
  ogg_uint16_t        *eob_run;
  oc_token_checkpoint *stack;
  int                  huffi;
  int                  zzi;
  int                  zzj;
  int                  zzk;
  int                  total_bits;
  int                  zr[4];
  stack=*_stack;
  total_bits=0;
  /*The apparent bit-cost of coding a zero from observing the trellis
     quantizer is pre-combined with lambda.
    Four predictive cases are considered: the last optimized value is zero (+2)
     or non-zero and the non-optimized value is zero (+1) or non-zero.*/
  zr[0]=3*_lambda>>1;
  zr[1]=_lambda;
  zr[2]=4*_lambda;
  zr[3]=7*_lambda>>1;
  eob_run=_enc->eob_run[_pli];
  dct_fzig_zag=_enc->state.opt_data.dct_fzig_zag;
  huffi=_enc->huff_idxs[_enc->state.frame_type][1][_pli+1>>1];
  for(zzj=zzi=1;zzj<_zzi&&!_qdct[zzj];zzj++);
  while(zzj<_zzi){
    int v;
    int d0;
    int d1;
    int sign;
    int k;
    int eob;
    int dq0;
    int dq1;
    int dd0;
    int dd1;
    int next_zero;
    int eob_bits;
    int dct_fzig_zzj;
    dct_fzig_zzj=dct_fzig_zag[zzj];
    v=_dct[zzj];
    d0=_qdct[zzj];
    eob=eob_run[zzi];
    for(zzk=zzj+1;zzk<_zzi&&!_qdct[zzk];zzk++);
    next_zero=zzk-zzj+62>>6;
    dq0=d0*_dequant[zzj];
    dd0=dq0-v;
    dd0*=dd0;
    sign=-(d0<0);
    k=d0+sign^sign;
    d1=(k-(zzj>_acmin))+sign^sign;
    dq1=d1*_dequant[zzj];
    dd1=dq1-v;
    dd1*=dd1;
    /*The cost of ending an eob run is included when the alternative is to
       extend this eob run.
      A per qi/zzi weight would probably be useful.
      Including it in the overall tokenization cost was not helpful.
      The same is true at the far end of the zero run plus token case.*/
    if(eob>0&&d1==0&&zzk==_zzi){
      eob_bits=oc_token_bits(_enc,huffi,zzi,OC_DCT_EOB1_TOKEN);
    }
    else eob_bits=0;
    if(zzj==zzi){
      /*No active zero run.*/
      int best_token;
      int best_eb;
      int token;
      int best_bits;
      int bits;
      int cost;
      best_token=*(OC_DCT_VALUE_TOKEN_PTR+d0);
      best_bits=oc_token_bits(_enc,huffi,zzi,best_token);
      if(d1!=0){
        token=*(OC_DCT_VALUE_TOKEN_PTR+d1);
        bits=oc_token_bits(_enc,huffi,zzi,token);
        cost=dd1+(bits+eob_bits)*_lambda;
      }
      else{
        token=bits=0;
        cost=dd1+zr[next_zero];
      }
      if((dd0+(best_bits+eob_bits)*_lambda)>cost){
        _idct[dct_fzig_zzj]=dq1;
        if(d1==0){
          zzj=zzk;
          continue;
        }
        best_bits=bits;
        best_token=token;
        best_eb=*(OC_DCT_VALUE_EB_PTR+d1);
      }
      else{
        best_eb=*(OC_DCT_VALUE_EB_PTR+d0);
        _idct[dct_fzig_zzj]=dq0;
      }
      oc_enc_tokenlog_checkpoint(_enc,stack++,_pli,zzi);
      if(eob>0){
        oc_enc_eob_log(_enc,_pli,zzi,eob);
        eob_run[zzi]=0;
      }
      oc_enc_token_log(_enc,_pli,zzi,best_token,best_eb);
      total_bits+=best_bits;
    }
    else{
      int d;
      int dc_reserve;
      int best_token;
      int best_eb;
      int best_bits;
      int best_cost;
      int best_bits1;
      int best_token1;
      int best_eb1;
      int zr_bits;
      int eob2;
      int eob_bits2;
      int bits;
      int token;
      int nzeros;
      nzeros=zzj-zzi;
      dc_reserve=zzi+62>>6;
      /*A zero run, followed by the value alone.*/
      best_token=best_token1=OC_DCT_SHORT_ZRL_TOKEN+(nzeros+55>>6);
      best_eb=best_eb1=nzeros-1;
      eob2=eob_run[zzj];
      eob_bits2=eob2>0?oc_token_bits(_enc,huffi,zzj,OC_DCT_EOB1_TOKEN):0;
      zr_bits=oc_token_bits(_enc,huffi,zzi,best_token)+eob_bits2;
      best_bits=zr_bits
       +oc_token_bits(_enc,huffi,zzj,*(OC_DCT_VALUE_TOKEN_PTR+d0));
      d=d0;
      best_bits1=0;
      if(d1!=0){
        best_bits1=zr_bits
         +oc_token_bits(_enc,huffi,zzj,*(OC_DCT_VALUE_TOKEN_PTR+d1));
      }
      if(nzeros<17+dc_reserve){
        if(k<=2){
          /*+/- 1 combo token.*/
          token=OC_DCT_RUN_CAT1_TOKEN[nzeros-1];
          bits=oc_token_bits(_enc,huffi,zzi,token);
          if(k==2&&bits<=best_bits1){
            best_bits1=bits;
            best_token1=token;
            best_eb1=OC_DCT_RUN_CAT1_EB[nzeros-1][-sign];
          }
          if(k==1&&bits<=best_bits){
            best_bits=bits;
            best_token=token;
            best_eb=OC_DCT_RUN_CAT1_EB[nzeros-1][-sign];
          }
        }
        if(nzeros<3+dc_reserve&&2<=k&&k<=4){
          /*+/- 2/3 combo token.*/
          token=OC_DCT_RUN_CAT2A+(nzeros>>1);
          bits=oc_token_bits(_enc,huffi,zzi,token);
          if(k==4&&bits<=best_bits1){
            best_bits1=bits;
            best_token1=token;
            best_eb1=OC_DCT_RUN_CAT2_EB[nzeros-1][-sign][1];
          }
          if(k!=4&&bits<=best_bits){
            best_bits=bits;
            best_token=token;
            best_eb=OC_DCT_RUN_CAT2_EB[nzeros-1][-sign][k-2];
          }
        }
      }
      best_cost=dd0+(best_bits+eob_bits)*_lambda;
      if(d1==0&&(dd1+zr[2+next_zero])<=best_cost){
        zzj=zzk;
        continue;
      }
      if(d1!=0&&dd1+(best_bits1+eob_bits)*_lambda<best_cost){
        best_bits=best_bits1;
        best_token=best_token1;
        best_eb=best_eb1;
        d=d1;
        _idct[dct_fzig_zzj]=dq1;
      }
      else _idct[dct_fzig_zzj]=dq0;
      oc_enc_tokenlog_checkpoint(_enc,stack++,_pli,zzi);
      if(eob){
        oc_enc_eob_log(_enc,_pli,zzi,eob);
        eob_run[zzi]=0;
      }
      oc_enc_token_log(_enc,_pli,zzi,best_token,best_eb);
      /*If a zero run won vs. the combo token we still need to code this
         value.*/
      if(best_token<=OC_DCT_ZRL_TOKEN){
        oc_enc_tokenlog_checkpoint(_enc,stack++,_pli,zzj);
        if(eob2){
          oc_enc_eob_log(_enc,_pli,zzj,eob2);
          /*The cost of any EOB run we disrupted is ignored because doing so
             improved PSNR/SSIM by a small amount.*/
          best_bits-=eob_bits2;
          eob_run[zzj]=0;
        }
        oc_enc_token_log(_enc,_pli,zzj,
         *(OC_DCT_VALUE_TOKEN_PTR+d),*(OC_DCT_VALUE_EB_PTR+d));
      }
      total_bits+=best_bits;
    }
    zzi=zzj+1;
    zzj=zzk;
  }
  /*Code an EOB run to complete this block.
    The cost of the EOB run is not included in the total as explained in
     in a comment in the trellis tokenizer above.*/
  if(zzi<64){
    int eob;
    eob=eob_run[zzi]+1;
    oc_enc_tokenlog_checkpoint(_enc,stack++,_pli,zzi);
    if(eob>=4095){
      oc_enc_token_log(_enc,_pli,zzi,OC_DCT_REPEAT_RUN3_TOKEN,eob);
      eob=0;
    }
    eob_run[zzi]=eob;
  }
  *_stack=stack;
  return total_bits;
}

void oc_enc_pred_dc_frag_rows(oc_enc_ctx *_enc,
 int _pli,int _fragy0,int _frag_yend){
  const oc_fragment_plane *fplane;
  const oc_fragment       *frags;
  ogg_int16_t             *frag_dc;
  ptrdiff_t                fragi;
  int                     *pred_last;
  int                      nhfrags;
  int                      fragx;
  int                      fragy;
  fplane=_enc->state.fplanes+_pli;
  frags=_enc->state.frags;
  frag_dc=_enc->frag_dc;
  pred_last=_enc->dc_pred_last[_pli];
  nhfrags=fplane->nhfrags;
  fragi=fplane->froffset+_fragy0*nhfrags;
  for(fragy=_fragy0;fragy<_frag_yend;fragy++){
    if(fragy==0){
      /*For the first row, all of the cases reduce to just using the previous
         predictor for the same reference frame.*/
      for(fragx=0;fragx<nhfrags;fragx++,fragi++){
        if(frags[fragi].coded){
          int refi;
          refi=frags[fragi].refi;
          frag_dc[fragi]=(ogg_int16_t)(frags[fragi].dc-pred_last[refi]);
          pred_last[refi]=frags[fragi].dc;
        }
      }
    }
    else{
      const oc_fragment *u_frags;
      int                l_ref;
      int                ul_ref;
      int                u_ref;
      u_frags=frags-nhfrags;
      l_ref=-1;
      ul_ref=-1;
      u_ref=u_frags[fragi].refi;
      for(fragx=0;fragx<nhfrags;fragx++,fragi++){
        int ur_ref;
        if(fragx+1>=nhfrags)ur_ref=-1;
        else ur_ref=u_frags[fragi+1].refi;
        if(frags[fragi].coded){
          int pred;
          int refi;
          refi=frags[fragi].refi;
          /*We break out a separate case based on which of our neighbors use
             the same reference frames.
            This is somewhat faster than trying to make a generic case which
             handles all of them, since it reduces lots of poorly predicted
             jumps to one switch statement, and also lets a number of the
             multiplications be optimized out by strength reduction.*/
          switch((l_ref==refi)|(ul_ref==refi)<<1|
           (u_ref==refi)<<2|(ur_ref==refi)<<3){
            default:pred=pred_last[refi];break;
            case  1:
            case  3:pred=frags[fragi-1].dc;break;
            case  2:pred=u_frags[fragi-1].dc;break;
            case  4:
            case  6:
            case 12:pred=u_frags[fragi].dc;break;
            case  5:pred=(frags[fragi-1].dc+u_frags[fragi].dc)/2;break;
            case  8:pred=u_frags[fragi+1].dc;break;
            case  9:
            case 11:
            case 13:{
              pred=(75*frags[fragi-1].dc+53*u_frags[fragi+1].dc)/128;
            }break;
            case 10:pred=(u_frags[fragi-1].dc+u_frags[fragi+1].dc)/2;break;
            case 14:{
              pred=(3*(u_frags[fragi-1].dc+u_frags[fragi+1].dc)
               +10*u_frags[fragi].dc)/16;
            }break;
            case  7:
            case 15:{
              int p0;
              int p1;
              int p2;
              p0=frags[fragi-1].dc;
              p1=u_frags[fragi-1].dc;
              p2=u_frags[fragi].dc;
              pred=(29*(p0+p2)-26*p1)/32;
              if(abs(pred-p2)>128)pred=p2;
              else if(abs(pred-p0)>128)pred=p0;
              else if(abs(pred-p1)>128)pred=p1;
            }break;
          }
          frag_dc[fragi]=(ogg_int16_t)(frags[fragi].dc-pred);
          pred_last[refi]=frags[fragi].dc;
          l_ref=refi;
        }
        else l_ref=-1;
        ul_ref=u_ref;
        u_ref=ur_ref;
      }
    }
  }
}

void oc_enc_tokenize_dc_frag_list(oc_enc_ctx *_enc,int _pli,
 const ptrdiff_t *_coded_fragis,ptrdiff_t _ncoded_fragis,
 int _prev_ndct_tokens1,int _prev_eob_run1){
  const ogg_int16_t *frag_dc;
  ptrdiff_t          fragii;
  unsigned char     *dct_tokens0;
  unsigned char     *dct_tokens1;
  ogg_uint16_t      *extra_bits0;
  ogg_uint16_t      *extra_bits1;
  ptrdiff_t          ti0;
  ptrdiff_t          ti1r;
  ptrdiff_t          ti1w;
  int                eob_run0;
  int                eob_run1;
  int                neobs1;
  int                token;
  int                eb;
  int                token1=token1;
  int                eb1=eb1;
  /*Return immediately if there are no coded fragments; otherwise we'd flush
     any trailing EOB run into the AC 1 list and never read it back out.*/
  if(_ncoded_fragis<=0)return;
  frag_dc=_enc->frag_dc;
  dct_tokens0=_enc->dct_tokens[_pli][0];
  dct_tokens1=_enc->dct_tokens[_pli][1];
  extra_bits0=_enc->extra_bits[_pli][0];
  extra_bits1=_enc->extra_bits[_pli][1];
  ti0=_enc->ndct_tokens[_pli][0];
  ti1w=ti1r=_prev_ndct_tokens1;
  eob_run0=_enc->eob_run[_pli][0];
  /*Flush any trailing EOB run for the 1st AC coefficient.
    This is needed to allow us to track tokens to the end of the list.*/
  eob_run1=_enc->eob_run[_pli][1];
  if(eob_run1>0)oc_enc_eob_log(_enc,_pli,1,eob_run1);
  /*If there was an active EOB run at the start of the 1st AC stack, read it
     in and decode it.*/
  if(_prev_eob_run1>0){
    token1=dct_tokens1[ti1r];
    eb1=extra_bits1[ti1r];
    ti1r++;
    eob_run1=oc_decode_eob_token(token1,eb1);
    /*Consume the portion of the run that came before these fragments.*/
    neobs1=eob_run1-_prev_eob_run1;
  }
  else eob_run1=neobs1=0;
  for(fragii=0;fragii<_ncoded_fragis;fragii++){
    int val;
    /*All tokens in the 1st AC coefficient stack are regenerated as the DC
       coefficients are produced.
      This can be done in-place; stack 1 cannot get larger.*/
    if(!neobs1){
      /*There's no active EOB run in stack 1; read the next token.*/
      token1=dct_tokens1[ti1r];
      eb1=extra_bits1[ti1r];
      ti1r++;
      if(token1<OC_NDCT_EOB_TOKEN_MAX){
        neobs1=oc_decode_eob_token(token1,eb1);
        /*It's an EOB run; add it to the current (inactive) one.
          Because we may have moved entries to stack 0, we may have an
           opportunity to merge two EOB runs in stack 1.*/
        eob_run1+=neobs1;
      }
    }
    val=frag_dc[_coded_fragis[fragii]];
    if(val){
      /*There was a non-zero DC value, so there's no alteration to stack 1
         for this fragment; just code the stack 0 token.*/
      /*Flush any pending EOB run.*/
      if(eob_run0>0){
        token=oc_make_eob_token_full(eob_run0,&eb);
        dct_tokens0[ti0]=(unsigned char)token;
        extra_bits0[ti0]=(ogg_uint16_t)eb;
        ti0++;
        eob_run0=0;
      }
      dct_tokens0[ti0]=*(OC_DCT_VALUE_TOKEN_PTR+val);
      extra_bits0[ti0]=*(OC_DCT_VALUE_EB_PTR+val);
      ti0++;
    }
    else{
      /*Zero DC value; that means the entry in stack 1 might need to be coded
         from stack 0.
        This requires a stack 1 fixup.*/
      if(neobs1>0){
        /*We're in the middle of an active EOB run in stack 1.
          Move it to stack 0.*/
        if(++eob_run0>=4095){
          dct_tokens0[ti0]=OC_DCT_REPEAT_RUN3_TOKEN;
          extra_bits0[ti0]=eob_run0;
          ti0++;
          eob_run0=0;
        }
        eob_run1--;
      }
      else{
        /*No active EOB run in stack 1, so we can't extend one in stack 0.
          Flush it if we've got it.*/
        if(eob_run0>0){
          token=oc_make_eob_token_full(eob_run0,&eb);
          dct_tokens0[ti0]=(unsigned char)token;
          extra_bits0[ti0]=(ogg_uint16_t)eb;
          ti0++;
          eob_run0=0;
        }
        /*Stack 1 token is one of: a pure zero run token, a single
           coefficient token, or a zero run/coefficient combo token.
          A zero run token is expanded and moved to token stack 0, and the
           stack 1 entry dropped.
          A single coefficient value may be transformed into combo token that
           is moved to stack 0, or if it cannot be combined, it is left alone
           and a single length-1 zero run is emitted in stack 0.
          A combo token is extended and moved to stack 0.
          During AC coding, we restrict the run lengths on combo tokens for
           stack 1 to guarantee we can extend them.*/
        switch(token1){
          case OC_DCT_SHORT_ZRL_TOKEN:{
            if(eb1<7){
              dct_tokens0[ti0]=OC_DCT_SHORT_ZRL_TOKEN;
              extra_bits0[ti0]=(ogg_uint16_t)(eb1+1);
              ti0++;
              /*Don't write the AC coefficient back out.*/
              continue;
            }
            /*Fall through.*/
          }
          case OC_DCT_ZRL_TOKEN:{
            dct_tokens0[ti0]=OC_DCT_ZRL_TOKEN;
            extra_bits0[ti0]=(ogg_uint16_t)(eb1+1);
            ti0++;
            /*Don't write the AC coefficient back out.*/
          }continue;
          case OC_ONE_TOKEN:
          case OC_MINUS_ONE_TOKEN:{
            dct_tokens0[ti0]=OC_DCT_RUN_CAT1A;
            extra_bits0[ti0]=(ogg_uint16_t)(token1-OC_ONE_TOKEN);
            ti0++;
            /*Don't write the AC coefficient back out.*/
          }continue;
          case OC_TWO_TOKEN:
          case OC_MINUS_TWO_TOKEN:{
            dct_tokens0[ti0]=OC_DCT_RUN_CAT2A;
            extra_bits0[ti0]=(ogg_uint16_t)(token1-OC_TWO_TOKEN<<1);
            ti0++;
            /*Don't write the AC coefficient back out.*/
          }continue;
          case OC_DCT_VAL_CAT2:{
            dct_tokens0[ti0]=OC_DCT_RUN_CAT2A;
            extra_bits0[ti0]=(ogg_uint16_t)((eb1<<1)+1);
            ti0++;
            /*Don't write the AC coefficient back out.*/
          }continue;
          case OC_DCT_RUN_CAT1A:
          case OC_DCT_RUN_CAT1A+1:
          case OC_DCT_RUN_CAT1A+2:
          case OC_DCT_RUN_CAT1A+3:{
            dct_tokens0[ti0]=(unsigned char)(token1+1);
            extra_bits0[ti0]=(ogg_uint16_t)eb1;
            ti0++;
            /*Don't write the AC coefficient back out.*/
          }continue;
          case OC_DCT_RUN_CAT1A+4:{
            dct_tokens0[ti0]=OC_DCT_RUN_CAT1B;
            extra_bits0[ti0]=(ogg_uint16_t)(eb1<<2);
            ti0++;
            /*Don't write the AC coefficient back out.*/
          }continue;
          case OC_DCT_RUN_CAT1B:{
            if((eb1&3)<3){
              dct_tokens0[ti0]=OC_DCT_RUN_CAT1B;
              extra_bits0[ti0]=(ogg_uint16_t)(eb1+1);
              ti0++;
              /*Don't write the AC coefficient back out.*/
              continue;
            }
            eb1=((eb1&4)<<1)-1;
            /*Fall through.*/
          }
          case OC_DCT_RUN_CAT1C:{
            dct_tokens0[ti0]=OC_DCT_RUN_CAT1C;
            extra_bits0[ti0]=(ogg_uint16_t)(eb1+1);
            ti0++;
            /*Don't write the AC coefficient back out.*/
          }continue;
          case OC_DCT_RUN_CAT2A:{
            eb1=(eb1<<1)-1;
            /*Fall through.*/
          }
          case OC_DCT_RUN_CAT2B:{
            dct_tokens0[ti0]=OC_DCT_RUN_CAT2B;
            extra_bits0[ti0]=(ogg_uint16_t)(eb1+1);
            ti0++;
            /*Don't write the AC coefficient back out.*/
          }continue;
        }
        /*We can't merge tokens, write a short zero run and keep going.*/
        dct_tokens0[ti0]=OC_DCT_SHORT_ZRL_TOKEN;
        extra_bits0[ti0]=0;
        ti0++;
      }
    }
    if(!neobs1){
      /*Flush any (inactive) EOB run.*/
      if(eob_run1>0){
        token=oc_make_eob_token_full(eob_run1,&eb);
        dct_tokens1[ti1w]=(unsigned char)token;
        extra_bits1[ti1w]=(ogg_uint16_t)eb;
        ti1w++;
        eob_run1=0;
      }
      /*There's no active EOB run, so log the current token.*/
      dct_tokens1[ti1w]=(unsigned char)token1;
      extra_bits1[ti1w]=(ogg_uint16_t)eb1;
      ti1w++;
    }
    else{
      /*Otherwise consume one EOB from the current run.*/
      neobs1--;
      /*If we have more than 4095 EOBs outstanding in stack1, flush the run.*/
      if(eob_run1-neobs1>=4095){
        dct_tokens1[ti1w]=OC_DCT_REPEAT_RUN3_TOKEN;
        extra_bits1[ti1w]=4095;
        ti1w++;
        eob_run1-=4095;
      }
    }
  }
  /*Save the current state.*/
  _enc->ndct_tokens[_pli][0]=ti0;
  _enc->ndct_tokens[_pli][1]=ti1w;
  _enc->eob_run[_pli][0]=eob_run0;
  _enc->eob_run[_pli][1]=eob_run1;
}

/*Final EOB run welding.*/
void oc_enc_tokenize_finish(oc_enc_ctx *_enc){
  int pli;
  int zzi;
  /*Emit final EOB runs.*/
  for(pli=0;pli<3;pli++)for(zzi=0;zzi<64;zzi++){
    int eob_run;
    eob_run=_enc->eob_run[pli][zzi];
    if(eob_run>0)oc_enc_eob_log(_enc,pli,zzi,eob_run);
  }
  /*Merge the final EOB run of one token list with the start of the next, if
     possible.*/
  for(zzi=0;zzi<64;zzi++)for(pli=0;pli<3;pli++){
    int       old_tok1;
    int       old_tok2;
    int       old_eb1;
    int       old_eb2;
    int       new_tok;
    int       new_eb;
    int       zzj;
    int       plj;
    ptrdiff_t ti=ti;
    int       run_count;
    /*Make sure this coefficient has tokens at all.*/
    if(_enc->ndct_tokens[pli][zzi]<=0)continue;
    /*Ensure the first token is an EOB run.*/
    old_tok2=_enc->dct_tokens[pli][zzi][0];
    if(old_tok2>=OC_NDCT_EOB_TOKEN_MAX)continue;
    /*Search for a previous coefficient that has any tokens at all.*/
    old_tok1=OC_NDCT_EOB_TOKEN_MAX;
    for(zzj=zzi,plj=pli;zzj>=0;zzj--){
      while(plj-->0){
        ti=_enc->ndct_tokens[plj][zzj]-1;
        if(ti>=_enc->dct_token_offs[plj][zzj]){
          old_tok1=_enc->dct_tokens[plj][zzj][ti];
          break;
        }
      }
      if(plj>=0)break;
      plj=3;
    }
    /*Ensure its last token was an EOB run.*/
    if(old_tok1>=OC_NDCT_EOB_TOKEN_MAX)continue;
    /*Pull off the associated extra bits, if any, and decode the runs.*/
    old_eb1=_enc->extra_bits[plj][zzj][ti];
    old_eb2=_enc->extra_bits[pli][zzi][0];
    run_count=oc_decode_eob_token(old_tok1,old_eb1)
     +oc_decode_eob_token(old_tok2,old_eb2);
    /*We can't possibly combine these into one run.
      It might be possible to split them more optimally, but we'll just leave
       them as-is.*/
    if(run_count>=4096)continue;
    /*We CAN combine them into one run.*/
    new_tok=oc_make_eob_token_full(run_count,&new_eb);
    _enc->dct_tokens[plj][zzj][ti]=(unsigned char)new_tok;
    _enc->extra_bits[plj][zzj][ti]=(ogg_uint16_t)new_eb;
    _enc->dct_token_offs[pli][zzi]++;
  }
}
