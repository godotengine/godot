/*
 *  Copyright (c) 2019 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "vp9/common/vp9_mv.h"
#include "vp9/encoder/vp9_non_greedy_mv.h"
// TODO(angiebird): move non_greedy_mv related functions to this file

#define LOG2_TABLE_SIZE 1024
static const int log2_table[LOG2_TABLE_SIZE] = {
  0,  // This is a dummy value
  0,        1048576,  1661954,  2097152,  2434718,  2710530,  2943725,
  3145728,  3323907,  3483294,  3627477,  3759106,  3880192,  3992301,
  4096672,  4194304,  4286015,  4372483,  4454275,  4531870,  4605679,
  4676053,  4743299,  4807682,  4869436,  4928768,  4985861,  5040877,
  5093962,  5145248,  5194851,  5242880,  5289431,  5334591,  5378443,
  5421059,  5462508,  5502851,  5542146,  5580446,  5617800,  5654255,
  5689851,  5724629,  5758625,  5791875,  5824409,  5856258,  5887450,
  5918012,  5947969,  5977344,  6006160,  6034437,  6062195,  6089453,
  6116228,  6142538,  6168398,  6193824,  6218829,  6243427,  6267632,
  6291456,  6314910,  6338007,  6360756,  6383167,  6405252,  6427019,
  6448477,  6469635,  6490501,  6511084,  6531390,  6551427,  6571202,
  6590722,  6609993,  6629022,  6647815,  6666376,  6684713,  6702831,
  6720734,  6738427,  6755916,  6773205,  6790299,  6807201,  6823917,
  6840451,  6856805,  6872985,  6888993,  6904834,  6920510,  6936026,
  6951384,  6966588,  6981641,  6996545,  7011304,  7025920,  7040397,
  7054736,  7068940,  7083013,  7096956,  7110771,  7124461,  7138029,
  7151476,  7164804,  7178017,  7191114,  7204100,  7216974,  7229740,
  7242400,  7254954,  7267405,  7279754,  7292003,  7304154,  7316208,
  7328167,  7340032,  7351805,  7363486,  7375079,  7386583,  7398000,
  7409332,  7420579,  7431743,  7442826,  7453828,  7464751,  7475595,
  7486362,  7497053,  7507669,  7518211,  7528680,  7539077,  7549404,
  7559660,  7569847,  7579966,  7590017,  7600003,  7609923,  7619778,
  7629569,  7639298,  7648964,  7658569,  7668114,  7677598,  7687023,
  7696391,  7705700,  7714952,  7724149,  7733289,  7742375,  7751407,
  7760385,  7769310,  7778182,  7787003,  7795773,  7804492,  7813161,
  7821781,  7830352,  7838875,  7847350,  7855777,  7864158,  7872493,
  7880782,  7889027,  7897226,  7905381,  7913492,  7921561,  7929586,
  7937569,  7945510,  7953410,  7961268,  7969086,  7976864,  7984602,
  7992301,  7999960,  8007581,  8015164,  8022709,  8030217,  8037687,
  8045121,  8052519,  8059880,  8067206,  8074496,  8081752,  8088973,
  8096159,  8103312,  8110431,  8117516,  8124569,  8131589,  8138576,
  8145532,  8152455,  8159347,  8166208,  8173037,  8179836,  8186605,
  8193343,  8200052,  8206731,  8213380,  8220001,  8226593,  8233156,
  8239690,  8246197,  8252676,  8259127,  8265550,  8271947,  8278316,
  8284659,  8290976,  8297266,  8303530,  8309768,  8315981,  8322168,
  8328330,  8334467,  8340579,  8346667,  8352730,  8358769,  8364784,
  8370775,  8376743,  8382687,  8388608,  8394506,  8400381,  8406233,
  8412062,  8417870,  8423655,  8429418,  8435159,  8440878,  8446576,
  8452252,  8457908,  8463542,  8469155,  8474748,  8480319,  8485871,
  8491402,  8496913,  8502404,  8507875,  8513327,  8518759,  8524171,
  8529564,  8534938,  8540293,  8545629,  8550947,  8556245,  8561525,
  8566787,  8572031,  8577256,  8582464,  8587653,  8592825,  8597980,
  8603116,  8608236,  8613338,  8618423,  8623491,  8628542,  8633576,
  8638593,  8643594,  8648579,  8653547,  8658499,  8663434,  8668354,
  8673258,  8678145,  8683017,  8687874,  8692715,  8697540,  8702350,
  8707145,  8711925,  8716690,  8721439,  8726174,  8730894,  8735599,
  8740290,  8744967,  8749628,  8754276,  8758909,  8763528,  8768134,
  8772725,  8777302,  8781865,  8786415,  8790951,  8795474,  8799983,
  8804478,  8808961,  8813430,  8817886,  8822328,  8826758,  8831175,
  8835579,  8839970,  8844349,  8848715,  8853068,  8857409,  8861737,
  8866053,  8870357,  8874649,  8878928,  8883195,  8887451,  8891694,
  8895926,  8900145,  8904353,  8908550,  8912734,  8916908,  8921069,
  8925220,  8929358,  8933486,  8937603,  8941708,  8945802,  8949885,
  8953957,  8958018,  8962068,  8966108,  8970137,  8974155,  8978162,
  8982159,  8986145,  8990121,  8994086,  8998041,  9001986,  9005920,
  9009844,  9013758,  9017662,  9021556,  9025440,  9029314,  9033178,
  9037032,  9040877,  9044711,  9048536,  9052352,  9056157,  9059953,
  9063740,  9067517,  9071285,  9075044,  9078793,  9082533,  9086263,
  9089985,  9093697,  9097400,  9101095,  9104780,  9108456,  9112123,
  9115782,  9119431,  9123072,  9126704,  9130328,  9133943,  9137549,
  9141146,  9144735,  9148316,  9151888,  9155452,  9159007,  9162554,
  9166092,  9169623,  9173145,  9176659,  9180165,  9183663,  9187152,
  9190634,  9194108,  9197573,  9201031,  9204481,  9207923,  9211357,
  9214784,  9218202,  9221613,  9225017,  9228412,  9231800,  9235181,
  9238554,  9241919,  9245277,  9248628,  9251971,  9255307,  9258635,
  9261956,  9265270,  9268577,  9271876,  9275169,  9278454,  9281732,
  9285002,  9288266,  9291523,  9294773,  9298016,  9301252,  9304481,
  9307703,  9310918,  9314126,  9317328,  9320523,  9323711,  9326892,
  9330067,  9333235,  9336397,  9339552,  9342700,  9345842,  9348977,
  9352106,  9355228,  9358344,  9361454,  9364557,  9367654,  9370744,
  9373828,  9376906,  9379978,  9383043,  9386102,  9389155,  9392202,
  9395243,  9398278,  9401306,  9404329,  9407345,  9410356,  9413360,
  9416359,  9419351,  9422338,  9425319,  9428294,  9431263,  9434226,
  9437184,  9440136,  9443082,  9446022,  9448957,  9451886,  9454809,
  9457726,  9460638,  9463545,  9466446,  9469341,  9472231,  9475115,
  9477994,  9480867,  9483735,  9486597,  9489454,  9492306,  9495152,
  9497993,  9500828,  9503659,  9506484,  9509303,  9512118,  9514927,
  9517731,  9520530,  9523324,  9526112,  9528895,  9531674,  9534447,
  9537215,  9539978,  9542736,  9545489,  9548237,  9550980,  9553718,
  9556451,  9559179,  9561903,  9564621,  9567335,  9570043,  9572747,
  9575446,  9578140,  9580830,  9583514,  9586194,  9588869,  9591540,
  9594205,  9596866,  9599523,  9602174,  9604821,  9607464,  9610101,
  9612735,  9615363,  9617987,  9620607,  9623222,  9625832,  9628438,
  9631040,  9633637,  9636229,  9638818,  9641401,  9643981,  9646556,
  9649126,  9651692,  9654254,  9656812,  9659365,  9661914,  9664459,
  9666999,  9669535,  9672067,  9674594,  9677118,  9679637,  9682152,
  9684663,  9687169,  9689672,  9692170,  9694665,  9697155,  9699641,
  9702123,  9704601,  9707075,  9709545,  9712010,  9714472,  9716930,
  9719384,  9721834,  9724279,  9726721,  9729159,  9731593,  9734024,
  9736450,  9738872,  9741291,  9743705,  9746116,  9748523,  9750926,
  9753326,  9755721,  9758113,  9760501,  9762885,  9765266,  9767642,
  9770015,  9772385,  9774750,  9777112,  9779470,  9781825,  9784175,
  9786523,  9788866,  9791206,  9793543,  9795875,  9798204,  9800530,
  9802852,  9805170,  9807485,  9809797,  9812104,  9814409,  9816710,
  9819007,  9821301,  9823591,  9825878,  9828161,  9830441,  9832718,
  9834991,  9837261,  9839527,  9841790,  9844050,  9846306,  9848559,
  9850808,  9853054,  9855297,  9857537,  9859773,  9862006,  9864235,
  9866462,  9868685,  9870904,  9873121,  9875334,  9877544,  9879751,
  9881955,  9884155,  9886352,  9888546,  9890737,  9892925,  9895109,
  9897291,  9899469,  9901644,  9903816,  9905985,  9908150,  9910313,
  9912473,  9914629,  9916783,  9918933,  9921080,  9923225,  9925366,
  9927504,  9929639,  9931771,  9933900,  9936027,  9938150,  9940270,
  9942387,  9944502,  9946613,  9948721,  9950827,  9952929,  9955029,
  9957126,  9959219,  9961310,  9963398,  9965484,  9967566,  9969645,
  9971722,  9973796,  9975866,  9977934,  9980000,  9982062,  9984122,
  9986179,  9988233,  9990284,  9992332,  9994378,  9996421,  9998461,
  10000498, 10002533, 10004565, 10006594, 10008621, 10010644, 10012665,
  10014684, 10016700, 10018713, 10020723, 10022731, 10024736, 10026738,
  10028738, 10030735, 10032729, 10034721, 10036710, 10038697, 10040681,
  10042662, 10044641, 10046617, 10048591, 10050562, 10052530, 10054496,
  10056459, 10058420, 10060379, 10062334, 10064287, 10066238, 10068186,
  10070132, 10072075, 10074016, 10075954, 10077890, 10079823, 10081754,
  10083682, 10085608, 10087532, 10089453, 10091371, 10093287, 10095201,
  10097112, 10099021, 10100928, 10102832, 10104733, 10106633, 10108529,
  10110424, 10112316, 10114206, 10116093, 10117978, 10119861, 10121742,
  10123620, 10125495, 10127369, 10129240, 10131109, 10132975, 10134839,
  10136701, 10138561, 10140418, 10142273, 10144126, 10145976, 10147825,
  10149671, 10151514, 10153356, 10155195, 10157032, 10158867, 10160699,
  10162530, 10164358, 10166184, 10168007, 10169829, 10171648, 10173465,
  10175280, 10177093, 10178904, 10180712, 10182519, 10184323, 10186125,
  10187925, 10189722, 10191518, 10193311, 10195103, 10196892, 10198679,
  10200464, 10202247, 10204028, 10205806, 10207583, 10209357, 10211130,
  10212900, 10214668, 10216435, 10218199, 10219961, 10221721, 10223479,
  10225235, 10226989, 10228741, 10230491, 10232239, 10233985, 10235728,
  10237470, 10239210, 10240948, 10242684, 10244417, 10246149, 10247879,
  10249607, 10251333, 10253057, 10254779, 10256499, 10258217, 10259933,
  10261647, 10263360, 10265070, 10266778, 10268485, 10270189, 10271892,
  10273593, 10275292, 10276988, 10278683, 10280376, 10282068, 10283757,
  10285444, 10287130, 10288814, 10290495, 10292175, 10293853, 10295530,
  10297204, 10298876, 10300547, 10302216, 10303883, 10305548, 10307211,
  10308873, 10310532, 10312190, 10313846, 10315501, 10317153, 10318804,
  10320452, 10322099, 10323745, 10325388, 10327030, 10328670, 10330308,
  10331944, 10333578, 10335211, 10336842, 10338472, 10340099, 10341725,
  10343349, 10344971, 10346592, 10348210, 10349828, 10351443, 10353057,
  10354668, 10356279, 10357887, 10359494, 10361099, 10362702, 10364304,
  10365904, 10367502, 10369099, 10370694, 10372287, 10373879, 10375468,
  10377057, 10378643, 10380228, 10381811, 10383393, 10384973, 10386551,
  10388128, 10389703, 10391276, 10392848, 10394418, 10395986, 10397553,
  10399118, 10400682, 10402244, 10403804, 10405363, 10406920, 10408476,
  10410030, 10411582, 10413133, 10414682, 10416230, 10417776, 10419320,
  10420863, 10422404, 10423944, 10425482, 10427019, 10428554, 10430087,
  10431619, 10433149, 10434678, 10436206, 10437731, 10439256, 10440778,
  10442299, 10443819, 10445337, 10446854, 10448369, 10449882, 10451394,
  10452905, 10454414, 10455921, 10457427, 10458932, 10460435, 10461936,
  10463436, 10464935, 10466432, 10467927, 10469422, 10470914, 10472405,
  10473895, 10475383, 10476870, 10478355, 10479839, 10481322, 10482802,
  10484282,
};

static int mi_size_to_block_size(int mi_bsize, int mi_num) {
  return (mi_num % mi_bsize) ? mi_num / mi_bsize + 1 : mi_num / mi_bsize;
}

Status vp9_alloc_motion_field_info(MotionFieldInfo *motion_field_info,
                                   int frame_num, int mi_rows, int mi_cols) {
  int frame_idx, rf_idx, square_block_idx;
  if (motion_field_info->allocated) {
    // TODO(angiebird): Avoid re-allocate buffer if possible
    vp9_free_motion_field_info(motion_field_info);
  }
  motion_field_info->frame_num = frame_num;
  motion_field_info->motion_field_array =
      vpx_calloc(frame_num, sizeof(*motion_field_info->motion_field_array));
  if (!motion_field_info->motion_field_array) return STATUS_FAILED;
  for (frame_idx = 0; frame_idx < frame_num; ++frame_idx) {
    for (rf_idx = 0; rf_idx < MAX_INTER_REF_FRAMES; ++rf_idx) {
      for (square_block_idx = 0; square_block_idx < SQUARE_BLOCK_SIZES;
           ++square_block_idx) {
        BLOCK_SIZE bsize = square_block_idx_to_bsize(square_block_idx);
        const int mi_height = num_8x8_blocks_high_lookup[bsize];
        const int mi_width = num_8x8_blocks_wide_lookup[bsize];
        const int block_rows = mi_size_to_block_size(mi_height, mi_rows);
        const int block_cols = mi_size_to_block_size(mi_width, mi_cols);
        MotionField *motion_field =
            &motion_field_info
                 ->motion_field_array[frame_idx][rf_idx][square_block_idx];
        Status status =
            vp9_alloc_motion_field(motion_field, bsize, block_rows, block_cols);
        if (status == STATUS_FAILED) {
          return STATUS_FAILED;
        }
      }
    }
  }
  motion_field_info->allocated = 1;
  return STATUS_OK;
}

Status vp9_alloc_motion_field(MotionField *motion_field, BLOCK_SIZE bsize,
                              int block_rows, int block_cols) {
  Status status = STATUS_OK;
  motion_field->ready = 0;
  motion_field->bsize = bsize;
  motion_field->block_rows = block_rows;
  motion_field->block_cols = block_cols;
  motion_field->block_num = block_rows * block_cols;
  motion_field->mf =
      vpx_calloc(motion_field->block_num, sizeof(*motion_field->mf));
  if (motion_field->mf == NULL) {
    status = STATUS_FAILED;
  }
  motion_field->set_mv =
      vpx_calloc(motion_field->block_num, sizeof(*motion_field->set_mv));
  if (motion_field->set_mv == NULL) {
    vpx_free(motion_field->mf);
    motion_field->mf = NULL;
    status = STATUS_FAILED;
  }
  motion_field->local_structure = vpx_calloc(
      motion_field->block_num, sizeof(*motion_field->local_structure));
  if (motion_field->local_structure == NULL) {
    vpx_free(motion_field->mf);
    motion_field->mf = NULL;
    vpx_free(motion_field->set_mv);
    motion_field->set_mv = NULL;
    status = STATUS_FAILED;
  }
  return status;
}

void vp9_free_motion_field(MotionField *motion_field) {
  vpx_free(motion_field->mf);
  vpx_free(motion_field->set_mv);
  vpx_free(motion_field->local_structure);
  vp9_zero(*motion_field);
}

void vp9_free_motion_field_info(MotionFieldInfo *motion_field_info) {
  if (motion_field_info->allocated) {
    int frame_idx, rf_idx, square_block_idx;
    for (frame_idx = 0; frame_idx < motion_field_info->frame_num; ++frame_idx) {
      for (rf_idx = 0; rf_idx < MAX_INTER_REF_FRAMES; ++rf_idx) {
        for (square_block_idx = 0; square_block_idx < SQUARE_BLOCK_SIZES;
             ++square_block_idx) {
          MotionField *motion_field =
              &motion_field_info
                   ->motion_field_array[frame_idx][rf_idx][square_block_idx];
          vp9_free_motion_field(motion_field);
        }
      }
    }
    vpx_free(motion_field_info->motion_field_array);
    motion_field_info->motion_field_array = NULL;
    motion_field_info->frame_num = 0;
    motion_field_info->allocated = 0;
  }
}

MotionField *vp9_motion_field_info_get_motion_field(
    MotionFieldInfo *motion_field_info, int frame_idx, int rf_idx,
    BLOCK_SIZE bsize) {
  int square_block_idx = get_square_block_idx(bsize);
  assert(frame_idx < motion_field_info->frame_num);
  assert(motion_field_info->allocated == 1);
  return &motion_field_info
              ->motion_field_array[frame_idx][rf_idx][square_block_idx];
}

int vp9_motion_field_is_mv_set(const MotionField *motion_field, int brow,
                               int bcol) {
  assert(brow >= 0 && brow < motion_field->block_rows);
  assert(bcol >= 0 && bcol < motion_field->block_cols);
  return motion_field->set_mv[brow * motion_field->block_cols + bcol];
}

int_mv vp9_motion_field_get_mv(const MotionField *motion_field, int brow,
                               int bcol) {
  assert(brow >= 0 && brow < motion_field->block_rows);
  assert(bcol >= 0 && bcol < motion_field->block_cols);
  return motion_field->mf[brow * motion_field->block_cols + bcol];
}

int_mv vp9_motion_field_mi_get_mv(const MotionField *motion_field, int mi_row,
                                  int mi_col) {
  const int mi_height = num_8x8_blocks_high_lookup[motion_field->bsize];
  const int mi_width = num_8x8_blocks_wide_lookup[motion_field->bsize];
  const int brow = mi_row / mi_height;
  const int bcol = mi_col / mi_width;
  assert(mi_row % mi_height == 0);
  assert(mi_col % mi_width == 0);
  return vp9_motion_field_get_mv(motion_field, brow, bcol);
}

void vp9_motion_field_mi_set_mv(MotionField *motion_field, int mi_row,
                                int mi_col, int_mv mv) {
  const int mi_height = num_8x8_blocks_high_lookup[motion_field->bsize];
  const int mi_width = num_8x8_blocks_wide_lookup[motion_field->bsize];
  const int brow = mi_row / mi_height;
  const int bcol = mi_col / mi_width;
  assert(mi_row % mi_height == 0);
  assert(mi_col % mi_width == 0);
  assert(brow >= 0 && brow < motion_field->block_rows);
  assert(bcol >= 0 && bcol < motion_field->block_cols);
  motion_field->mf[brow * motion_field->block_cols + bcol] = mv;
  motion_field->set_mv[brow * motion_field->block_cols + bcol] = 1;
}

void vp9_motion_field_reset_mvs(MotionField *motion_field) {
  memset(motion_field->set_mv, 0,
         motion_field->block_num * sizeof(*motion_field->set_mv));
}

static int64_t log2_approximation(int64_t v) {
  assert(v > 0);
  if (v < LOG2_TABLE_SIZE) {
    return log2_table[v];
  } else {
    // use linear approximation when v >= 2^10
    const int slope =
        1477;  // slope = 1 / (log(2) * 1024) * (1 << LOG2_PRECISION)
    assert(LOG2_TABLE_SIZE == 1 << 10);

    return slope * (v - LOG2_TABLE_SIZE) + (10 << LOG2_PRECISION);
  }
}

int64_t vp9_nb_mvs_inconsistency(const MV *mv, const int_mv *nb_full_mvs,
                                 int mv_num) {
  // The behavior of this function is to compute log2 of mv difference,
  // i.e. min log2(1 + row_diff * row_diff + col_diff * col_diff)
  // against available neighbor mvs.
  // Since the log2 is monotonically increasing, we can compute
  // min row_diff * row_diff + col_diff * col_diff first
  // then apply log2 in the end.
  int i;
  int64_t min_abs_diff = INT64_MAX;
  int cnt = 0;
  assert(mv_num <= NB_MVS_NUM);
  for (i = 0; i < mv_num; ++i) {
    MV nb_mv = nb_full_mvs[i].as_mv;
    const int64_t row_diff = abs(mv->row - nb_mv.row);
    const int64_t col_diff = abs(mv->col - nb_mv.col);
    const int64_t abs_diff = row_diff * row_diff + col_diff * col_diff;
    assert(nb_full_mvs[i].as_int != INVALID_MV);
    min_abs_diff = VPXMIN(abs_diff, min_abs_diff);
    ++cnt;
  }
  if (cnt) {
    return log2_approximation(1 + min_abs_diff);
  }
  return 0;
}

static FloatMV get_smooth_motion_vector(const FloatMV scaled_search_mv,
                                        const FloatMV *tmp_mf,
                                        const int (*M)[MF_LOCAL_STRUCTURE_SIZE],
                                        int rows, int cols, int row, int col,
                                        float alpha) {
  const FloatMV tmp_mv = tmp_mf[row * cols + col];
  int idx_row, idx_col;
  FloatMV avg_nb_mv = { 0.0f, 0.0f };
  FloatMV mv = { 0.0f, 0.0f };
  float filter[3][3] = { { 1.0f / 12.0f, 1.0f / 6.0f, 1.0f / 12.0f },
                         { 1.0f / 6.0f, 0.0f, 1.0f / 6.0f },
                         { 1.0f / 12.0f, 1.0f / 6.0f, 1.0f / 12.0f } };
  for (idx_row = 0; idx_row < 3; ++idx_row) {
    int nb_row = row + idx_row - 1;
    for (idx_col = 0; idx_col < 3; ++idx_col) {
      int nb_col = col + idx_col - 1;
      if (nb_row < 0 || nb_col < 0 || nb_row >= rows || nb_col >= cols) {
        avg_nb_mv.row += (tmp_mv.row) * filter[idx_row][idx_col];
        avg_nb_mv.col += (tmp_mv.col) * filter[idx_row][idx_col];
      } else {
        const FloatMV nb_mv = tmp_mf[nb_row * cols + nb_col];
        avg_nb_mv.row += (nb_mv.row) * filter[idx_row][idx_col];
        avg_nb_mv.col += (nb_mv.col) * filter[idx_row][idx_col];
      }
    }
  }
  {
    // M is the local variance of reference frame
    float M00 = M[row * cols + col][0];
    float M01 = M[row * cols + col][1];
    float M10 = M[row * cols + col][2];
    float M11 = M[row * cols + col][3];

    float det = (M00 + alpha) * (M11 + alpha) - M01 * M10;

    float inv_M00 = (M11 + alpha) / det;
    float inv_M01 = -M01 / det;
    float inv_M10 = -M10 / det;
    float inv_M11 = (M00 + alpha) / det;

    float inv_MM00 = inv_M00 * M00 + inv_M01 * M10;
    float inv_MM01 = inv_M00 * M01 + inv_M01 * M11;
    float inv_MM10 = inv_M10 * M00 + inv_M11 * M10;
    float inv_MM11 = inv_M10 * M01 + inv_M11 * M11;

    mv.row = inv_M00 * avg_nb_mv.row * alpha + inv_M01 * avg_nb_mv.col * alpha +
             inv_MM00 * scaled_search_mv.row + inv_MM01 * scaled_search_mv.col;
    mv.col = inv_M10 * avg_nb_mv.row * alpha + inv_M11 * avg_nb_mv.col * alpha +
             inv_MM10 * scaled_search_mv.row + inv_MM11 * scaled_search_mv.col;
  }
  return mv;
}

void vp9_get_smooth_motion_field(const MV *search_mf,
                                 const int (*M)[MF_LOCAL_STRUCTURE_SIZE],
                                 int rows, int cols, BLOCK_SIZE bsize,
                                 float alpha, int num_iters, MV *smooth_mf) {
  // M is the local variation of reference frame
  // build two buffers
  FloatMV *input = (FloatMV *)malloc(rows * cols * sizeof(FloatMV));
  FloatMV *output = (FloatMV *)malloc(rows * cols * sizeof(FloatMV));
  int idx;
  int row, col;
  int bw = 4 << b_width_log2_lookup[bsize];
  int bh = 4 << b_height_log2_lookup[bsize];
  if (!(input && output)) goto fail;
  // copy search results to input buffer
  for (idx = 0; idx < rows * cols; ++idx) {
    input[idx].row = (float)search_mf[idx].row / bh;
    input[idx].col = (float)search_mf[idx].col / bw;
  }
  for (idx = 0; idx < num_iters; ++idx) {
    FloatMV *tmp;
    for (row = 0; row < rows; ++row) {
      for (col = 0; col < cols; ++col) {
        // note: the scaled_search_mf and smooth_mf are all scaled by macroblock
        // size
        const MV search_mv = search_mf[row * cols + col];
        FloatMV scaled_search_mv = { (float)search_mv.row / bh,
                                     (float)search_mv.col / bw };
        output[row * cols + col] = get_smooth_motion_vector(
            scaled_search_mv, input, M, rows, cols, row, col, alpha);
      }
    }
    // swap buffers
    tmp = input;
    input = output;
    output = tmp;
  }
  // copy smoothed results to output
  for (idx = 0; idx < rows * cols; ++idx) {
    smooth_mf[idx].row = (int)(input[idx].row * bh);
    smooth_mf[idx].col = (int)(input[idx].col * bw);
  }
fail:
  free(input);
  free(output);
}

void vp9_get_local_structure(const YV12_BUFFER_CONFIG *cur_frame,
                             const YV12_BUFFER_CONFIG *ref_frame,
                             const MV *search_mf,
                             const vp9_variance_fn_ptr_t *fn_ptr, int rows,
                             int cols, BLOCK_SIZE bsize,
                             int (*M)[MF_LOCAL_STRUCTURE_SIZE]) {
  const int bw = 4 << b_width_log2_lookup[bsize];
  const int bh = 4 << b_height_log2_lookup[bsize];
  const int cur_stride = cur_frame->y_stride;
  const int ref_stride = ref_frame->y_stride;
  const int width = ref_frame->y_width;
  const int height = ref_frame->y_height;
  int row, col;
  for (row = 0; row < rows; ++row) {
    for (col = 0; col < cols; ++col) {
      int cur_offset = row * bh * cur_stride + col * bw;
      uint8_t *center = cur_frame->y_buffer + cur_offset;
      int ref_h = row * bh + search_mf[row * cols + col].row;
      int ref_w = col * bw + search_mf[row * cols + col].col;
      int ref_offset;
      uint8_t *target;
      uint8_t *nb;
      int search_dist;
      int nb_dist;
      int I_row = 0, I_col = 0;
      // TODO(Dan): handle the case that when reference frame block beyond the
      // boundary
      ref_h = ref_h < 0 ? 0 : (ref_h >= height - bh ? height - bh - 1 : ref_h);
      ref_w = ref_w < 0 ? 0 : (ref_w >= width - bw ? width - bw - 1 : ref_w);
      // compute search results distortion
      // TODO(Dan): maybe need to use vp9 function to find the reference block,
      // to compare with the results of my python code, I first use my way to
      // compute the reference block
      ref_offset = ref_h * ref_stride + ref_w;
      target = ref_frame->y_buffer + ref_offset;
      search_dist = fn_ptr->sdf(center, cur_stride, target, ref_stride);
      // compute target's neighbors' distortions
      // TODO(Dan): if using padding, the boundary condition may vary
      // up
      if (ref_h - bh >= 0) {
        nb = target - ref_stride * bh;
        nb_dist = fn_ptr->sdf(center, cur_stride, nb, ref_stride);
        I_row += nb_dist - search_dist;
      }
      // down
      if (ref_h + bh < height - bh) {
        nb = target + ref_stride * bh;
        nb_dist = fn_ptr->sdf(center, cur_stride, nb, ref_stride);
        I_row += nb_dist - search_dist;
      }
      if (ref_h - bh >= 0 && ref_h + bh < height - bh) {
        I_row /= 2;
      }
      I_row /= (bw * bh);
      // left
      if (ref_w - bw >= 0) {
        nb = target - bw;
        nb_dist = fn_ptr->sdf(center, cur_stride, nb, ref_stride);
        I_col += nb_dist - search_dist;
      }
      // down
      if (ref_w + bw < width - bw) {
        nb = target + bw;
        nb_dist = fn_ptr->sdf(center, cur_stride, nb, ref_stride);
        I_col += nb_dist - search_dist;
      }
      if (ref_w - bw >= 0 && ref_w + bw < width - bw) {
        I_col /= 2;
      }
      I_col /= (bw * bh);
      M[row * cols + col][0] = I_row * I_row;
      M[row * cols + col][1] = I_row * I_col;
      M[row * cols + col][2] = I_col * I_row;
      M[row * cols + col][3] = I_col * I_col;
    }
  }
}
