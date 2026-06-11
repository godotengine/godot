// Copyright 2017 The Draco Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#include "draco/mesh/mesh_stripifier.h"

namespace draco {

void MeshStripifier::GenerateStripsFromCorner(int local_strip_id,
                                              CornerIndex ci) {
  // Clear the storage for strip faces.
  strip_faces_[local_strip_id].clear();
  // Start corner of the strip (where the strip starts).
  CornerIndex start_ci = ci;
  FaceIndex fi = corner_table_->Face(ci);
  // We need to grow the strip both forward and backward (2 passes).
  // Note that the backward pass can change the start corner of the strip (the
  // start corner is going to be moved to the end of the backward strip).
  for (int pass = 0; pass < 2; ++pass) {
    if (pass == 1) {
      // Backward pass.
      // Start the traversal from the B that is the left sibling of the next
      // corner to the start corner C = |start_ci|.
      //
      //      *-------*-------*-------*
      //     / \     / \C    / \     /
      //    /   \   /   \   /   \   /
      //   /     \ /    B\ /     \ /
      //  *-------*-------*-------*
      //
      // Perform the backward pass only when there is no attribute seam between
      // the initial face and the first face of the backward traversal.
      if (GetOppositeCorner(corner_table_->Previous(start_ci)) ==
          kInvalidCornerIndex) {
        break;  // Attribute seam or a boundary.
      }

      ci = corner_table_->Next(start_ci);
      ci = corner_table_->SwingLeft(ci);
      if (ci == kInvalidCornerIndex) {
        break;
      }

      fi = corner_table_->Face(ci);
    }
    int num_added_faces = 0;
    while (!is_face_visited_[fi]) {
      is_face_visited_[fi] = true;
      strip_faces_[local_strip_id].push_back(fi);
      ++num_added_faces;
      if (num_added_faces > 1) {
        // Move to the correct source corner to traverse to the next face.
        if (num_added_faces & 1) {
          // Odd number of faces added.
          ci = corner_table_->Next(ci);
        } else {
          // Even number of faces added.
          if (pass == 1) {
            // If we are processing the backward pass, update the start corner
            // of the strip on every even face reached (we cannot use odd faces
            // for start of the strip as the strips would start in a wrong
            // direction).
            start_ci = ci;
          }
          ci = corner_table_->Previous(ci);
        }
      }
      ci = GetOppositeCorner(ci);
      if (ci == kInvalidCornerIndex) {
        break;
      }
      fi = corner_table_->Face(ci);
    }
    // Strip end reached.
    if (pass == 1 && (num_added_faces & 1)) {
      // If we processed the backward strip and we add an odd number of faces to
      // the strip, we need to remove the last one as it cannot be used to start
      // the strip (the strip would start in a wrong direction from that face).
      is_face_visited_[strip_faces_[local_strip_id].back()] = false;
      strip_faces_[local_strip_id].pop_back();
    }
  }
  strip_start_corners_[local_strip_id] = start_ci;

  // Reset all visited flags for all faces (we need to process other strips from
  // the given face before we choose the final strip that we are going to use).
  for (int i = 0; i < strip_faces_[local_strip_id].size(); ++i) {
    is_face_visited_[strip_faces_[local_strip_id][i]] = false;
  }
}

}  // namespace draco
