// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/modular/encoding/enc_debug_tree.h"

#include <cinttypes>  // PRId64
#include <cstdio>
#include <cstdlib>
#include <string>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/encoding/dec_ma.h"
#include "lib/jxl/modular/options.h"

namespace jxl {

namespace {
const char *PredictorName(Predictor p) {
  switch (p) {
    case Predictor::Zero:
      return "Zero";
    case Predictor::Left:
      return "Left";
    case Predictor::Top:
      return "Top";
    case Predictor::Average0:
      return "Avg0";
    case Predictor::Average1:
      return "Avg1";
    case Predictor::Average2:
      return "Avg2";
    case Predictor::Average3:
      return "Avg3";
    case Predictor::Average4:
      return "Avg4";
    case Predictor::Select:
      return "Sel";
    case Predictor::Gradient:
      return "Grd";
    case Predictor::Weighted:
      return "Wgh";
    case Predictor::TopLeft:
      return "TopL";
    case Predictor::TopRight:
      return "TopR";
    case Predictor::LeftLeft:
      return "LL";
    default:
      return "INVALID";
  };
}

std::string PropertyName(size_t i) {
  static_assert(kNumNonrefProperties == 16, "Update this function");
  switch (i) {
    case 0:
      return "c";
    case 1:
      return "g";
    case 2:
      return "y";
    case 3:
      return "x";
    case 4:
      return "|N|";
    case 5:
      return "|W|";
    case 6:
      return "N";
    case 7:
      return "W";
    case 8:
      return "W-WW-NW+NWW";
    case 9:
      return "W+N-NW";
    case 10:
      return "W-NW";
    case 11:
      return "NW-N";
    case 12:
      return "N-NE";
    case 13:
      return "N-NN";
    case 14:
      return "W-WW";
    case 15:
      return "WGH";
    default:
      return "ch[" + ToString(15 - static_cast<int>(i)) + "]";
  }
}

}  // namespace

void PrintTree(const Tree &tree, const std::string &path) {
  FILE *f = fopen((path + ".dot").c_str(), "w");
  fprintf(f, "graph{\n");
  for (size_t cur = 0; cur < tree.size(); cur++) {
    if (tree[cur].property < 0) {
      fprintf(f, "n%05" PRIuS " [label=\"%s%+" PRId64 " (x%u)\"];\n", cur,
              PredictorName(tree[cur].predictor), tree[cur].predictor_offset,
              tree[cur].multiplier);
    } else {
      fprintf(f, "n%05" PRIuS " [label=\"%s>%d\"];\n", cur,
              PropertyName(tree[cur].property).c_str(), tree[cur].splitval);
      fprintf(f, "n%05" PRIuS " -- n%05d;\n", cur, tree[cur].lchild);
      fprintf(f, "n%05" PRIuS " -- n%05d;\n", cur, tree[cur].rchild);
    }
  }
  fprintf(f, "}\n");
  fclose(f);
#if JXL_ENABLE_DOT
  system(("dot " + path + ".dot -T svg -o " + path + ".svg").c_str());
#endif
}

}  // namespace jxl
