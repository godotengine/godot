#include "animation_node_motion_match.h"
#include "scene\main\node.h"

void AnimationNodeMotionMatch::get_parameter_list(
    List<PropertyInfo> *r_list) const {
  r_list->push_back(
      PropertyInfo(Variant::VECTOR3, vel, PROPERTY_HINT_NONE, ""));
  r_list->push_back(PropertyInfo(Variant::INT, min, PROPERTY_HINT_NONE, ""));
}

void AnimationNodeMotionMatch::add_matching_track(
    const NodePath &p_track_path) {
  matching_tracks.push_back(p_track_path);
}

void AnimationNodeMotionMatch::remove_matching_track(
    const NodePath &p_track_path) {

  matching_tracks.erase(p_track_path);
}

bool AnimationNodeMotionMatch::is_matching_track(
    const NodePath &p_track_path) const {
  return matching_tracks.find(p_track_path) != -1;
}

void AnimationNodeMotionMatch::update_motion_database(
    AnimationPlayer *p_animation_player) {
  for (int i = 0; i < matching_tracks.size(); i++) {
    print_line("track " + itos(i) + ": " + String(matching_tracks[i]));
  }
}

Vector<NodePath> AnimationNodeMotionMatch::get_matching_tracks() {
  return matching_tracks;
}

void AnimationNodeMotionMatch::_bind_methods() {

  ClassDB::bind_method(D_METHOD("add_matching_track", "track"),
                       &AnimationNodeMotionMatch::add_matching_track);
  ClassDB::bind_method(D_METHOD("remove_matching_track", "track"),
                       &AnimationNodeMotionMatch::remove_matching_track);
  ClassDB::bind_method(D_METHOD("is_matching_track", "track"),
                       &AnimationNodeMotionMatch::is_matching_track);

  ClassDB::bind_method(D_METHOD("add_coordinates", "point"),
                       &AnimationNodeMotionMatch::add_coordinates);
  ClassDB::bind_method(D_METHOD("load_coordinates", "points"),
                       &AnimationNodeMotionMatch::load_coordinates);
  ClassDB::bind_method(D_METHOD("clear_coordinates"),
                       &AnimationNodeMotionMatch::clear_coordinates);
  ClassDB::bind_method(D_METHOD("get_coordinates"),
                       &AnimationNodeMotionMatch::get_coordinates);

  ClassDB::bind_method(D_METHOD("set_dim_len", "dim_len"),
                       &AnimationNodeMotionMatch::set_dim_len);
  ClassDB::bind_method(D_METHOD("get_dim_len"),
                       &AnimationNodeMotionMatch::get_dim_len);

  ClassDB::bind_method(D_METHOD("set_min_leaves", "min_leaves"),
                       &AnimationNodeMotionMatch::set_min_leaves);
  ClassDB::bind_method(D_METHOD("get_min_leaves"),
                       &AnimationNodeMotionMatch::get_min_leaves);

  ClassDB::bind_method(D_METHOD("get_root"),
                       &AnimationNodeMotionMatch::get_root);
  ClassDB::bind_method(D_METHOD("build_tree"),
                       &AnimationNodeMotionMatch::build_tree);

  ClassDB::bind_method(D_METHOD("set_start_index", "si"),
                       &AnimationNodeMotionMatch::set_start_index);
  ClassDB::bind_method(D_METHOD("get_start_index"),
                       &AnimationNodeMotionMatch::get_start_index);

  ClassDB::bind_method(D_METHOD("calc_root_threshold"),
                       &AnimationNodeMotionMatch::calc_root_threshold);
  ClassDB::bind_method(D_METHOD("KNNSearch", "point", "k"),
                       &AnimationNodeMotionMatch::KNNSearch);

  ClassDB::bind_method(D_METHOD("calculate_threshold", "point_coordinates"),
                       &AnimationNodeMotionMatch::KDNode::calculate_threshold);

  ClassDB::bind_method(D_METHOD("add_index", "ind"),
                       &AnimationNodeMotionMatch::KDNode::add_index);
  ClassDB::bind_method(D_METHOD("clear_indices"),
                       &AnimationNodeMotionMatch::KDNode::clear_indices);
  ClassDB::bind_method(D_METHOD("get_indices"),
                       &AnimationNodeMotionMatch::KDNode::get_indices);

  ClassDB::bind_method(D_METHOD("leaf_split", "point_coordinates_data",
                                "no_of_dims", "start_dim"),
                       &AnimationNodeMotionMatch::KDNode::leaf_split);

  ClassDB::bind_method(D_METHOD("get_left"),
                       &AnimationNodeMotionMatch::KDNode::get_left);
  ClassDB::bind_method(D_METHOD("get_right"),
                       &AnimationNodeMotionMatch::KDNode::get_right);

  ClassDB::bind_method(D_METHOD("get_prev"),
                       &AnimationNodeMotionMatch::KDNode::get_prev);

  ClassDB::bind_method(D_METHOD("set_th", "th"),
                       &AnimationNodeMotionMatch::KDNode::set_th);
  ClassDB::bind_method(D_METHOD("get_th"),
                       &AnimationNodeMotionMatch::KDNode::get_th);

  ClassDB::bind_method(D_METHOD("Predict_traj", "vel", "samples"),
                       &AnimationNodeMotionMatch::Predict_traj);
}

AnimationNodeMotionMatch::AnimationNodeMotionMatch() {
  root.instance();
  dim_len = 2;
  start_index = 0;
  point_coordinates = {};
  min_leaves = 1;

  this->vel = "velocity";
  this->pos = "position";
  this->min = "min_key";
}

AnimationNodeMotionMatch::~AnimationNodeMotionMatch() {}

void AnimationNodeMotionMatch::add_coordinates(PoolRealArray point) {
  if (point.size() != dim_len) {
    print_line("ERROR: Point is of wrong size.");
    error = true;
    err = LOAD_POINT_ERROR;
  } else {
    for (int i = 0; i < point.size(); i++) {
      this->point_coordinates.append(point[i]);
    }
    root->add_index(root->get_indices().size());
  }
}

void AnimationNodeMotionMatch::load_coordinates(PoolRealArray points) {
  if (points.size() % dim_len != 0) {
    print_line("ERROR: Point is of wrong size.");
    error = true;
    err = LOAD_POINT_ERROR;
  } else {
    for (int i = 0; i < points.size() / dim_len; i++) {
      PoolRealArray point;
      for (int j = 0; j < dim_len; j++) {
        point.append(points[i * dim_len + j]);
      }
      add_coordinates(point);
    }
  }
}

PoolRealArray AnimationNodeMotionMatch::get_coordinates() {
  return this->point_coordinates;
}

void AnimationNodeMotionMatch::clear_coordinates() {
  this->point_coordinates.resize(0);
}

void AnimationNodeMotionMatch::set_dim_len(uint32_t dim_len) {
  this->dim_len = dim_len;
}

uint32_t AnimationNodeMotionMatch::get_dim_len() { return this->dim_len; }

AnimationNodeMotionMatch::KDNode *AnimationNodeMotionMatch::get_root() {
  return root.ptr();
}

void AnimationNodeMotionMatch::clear_root() {
  root->point_indices = {};
  root->split_th = 0;
  root->split_axis = 0;
  root->left = nullptr;
  root->right = nullptr;
}

void AnimationNodeMotionMatch::set_start_index(uint32_t si) {
  this->start_index = si;
}

uint32_t AnimationNodeMotionMatch::get_start_index() {
  return this->start_index;
}

void AnimationNodeMotionMatch::calc_root_threshold() {
  if (this->point_coordinates.size() == 0) {
    print_line("ERROR:Load Points first!");
  } else {
    root->calculate_threshold(point_coordinates, dim_len);
  }
}

void AnimationNodeMotionMatch::set_min_leaves(uint32_t min_l) {
  min_leaves = min_l;
}

uint32_t AnimationNodeMotionMatch::get_min_leaves() { return min_leaves; }

void AnimationNodeMotionMatch::build_tree() {
  if (error == true) {
    if (err == LOAD_POINT_ERROR)
      print_line("ERROR: Check the input points");
    else if (err == QUERY_POINT_ERROR)
      print_line("ERROR: Check your query point");
    else if (err = K_ERROR)
      print_line("ERROR: Invalid value for number of neighbors");

  } else {
    print_line("PROCESS : Building KD-Tree..");
    root->leaf_split(point_coordinates, dim_len, start_index, min_leaves);
    print_line("KD-Tree built SUCCESSFULLY");
  }
} /*Builds the tree with all the given parameters*/

float dist_between(PoolRealArray point_coordinates, PoolRealArray p1,
                   uint32_t index) {
  float n = 0;
  for (int i = 0; i < p1.size(); i++) {
    n += (p1[i] - point_coordinates[p1.size() * index + i]) *
         (p1[i] - point_coordinates[p1.size() * index + i]);
  }
  return sqrt(n);
}

bool in_array(PoolRealArray points, uint32_t query) {
  for (int i = 0; i < points.size(); i++) {
    if (points[i] == query)
      return true;
  }
  return false;
}

PoolRealArray AnimationNodeMotionMatch::KNNSearch(PoolRealArray point,
                                                  uint32_t k) {
  if (error == false && point.size() % dim_len != 0) {
    error = true;
    err = QUERY_POINT_ERROR;
  } else if (error == false && k > point_coordinates.size() / dim_len) {
    error = true;
    err = K_ERROR;
  }

  if (error == true) {
    if (err == LOAD_POINT_ERROR)
      print_line("ERROR: Check the input points");
    else if (err == QUERY_POINT_ERROR)
      print_line("ERROR: Check your query point");
    else if (err = K_ERROR)
      print_line("ERROR: Invalid value for number of neighbors");

    return {};

  } else {
    PoolRealArray Knn = {};
    KDNode *m_node; /*node where the query point can be placed*/
    KDNode *node = root.ptr();

    while (node->get_indices().size() > min_leaves) {
      if (point[node->get_split_axis()] > node->get_th()) {
        node = node->get_left();
      } else {
        node = node->get_right();
      }
    }
    m_node = node;
    for (int j = 0; j < k; j++) {
      uint32_t nn;                                  /*nearest neighbour*/
      float nd = std::numeric_limits<float>::max(); /*nearest distance*/
      for (int i = 0; i < node->get_indices().size(); i++) {
        if (j == 0) {
          float dist =
              dist_between(point_coordinates, point, node->get_indices()[i]);
          if (nd > dist || nd == 0) {
            nd = dist;
            nn = node->get_indices()[i];
          }
        } else if (!in_array(Knn, node->get_indices()[i])) {
          float dist =
              dist_between(point_coordinates, point, node->get_indices()[i]);
          if (nd > dist || nd == 0) {
            nd = dist;
            nn = node->get_indices()[i];
          }
        }
      }
      while (node->get_prev() != NULL) {
        node = node->get_prev();
        if ((point[node->get_split_axis()] - node->get_th()) > nd) {
          break;
        } else {
          for (int i = 0; i < node->get_indices().size(); i++) {
            if (j == 0) {
              float dist = dist_between(point_coordinates, point,
                                        node->get_indices()[i]);
              if (nd > dist || nd == 0) {
                nd = dist;
                nn = node->get_indices()[i];
              }
            } else if (!in_array(Knn, node->get_indices()[i])) {
              float dist = dist_between(point_coordinates, point,
                                        node->get_indices()[i]);
              if (nd > dist || nd == 0) {
                nd = dist;
                nn = node->get_indices()[i];
              }
            }
          }
        }
      }

      Knn.append(nn);
      node = m_node;
      print_line("Round" + itos(j) + ":" + itos(nn));
    }
    return Knn;
  }
} /*returns K nearest neighbours in a PoolRealArray*/

void AnimationNodeMotionMatch::KDNode::calculate_threshold(
    PoolRealArray point_coordinates, uint32_t dim_len) {
  if (point_coordinates.size() % dim_len != 0) {
    print_line("ERROR: Point coordinates array is of wrong size.");
  } else {
    this->split_th = 0;
    for (int i = 0; i < this->point_indices.size(); i++) {
      this->split_th +=
          point_coordinates[(point_indices[i] * dim_len) + split_axis];
    }
    this->split_th = this->split_th / this->point_indices.size();
  }
}

bool AnimationNodeMotionMatch::KDNode::are_all_points_same(
    PoolRealArray point_coordinates, uint32_t dim_len) {
  for (int i = 0; i < dim_len; i++) {
    for (int j = 0; j < point_indices.size(); j++) {
      float p = point_coordinates[(point_indices[j] * dim_len) + i];
      for (int k = 0; k < point_indices.size(); k++) {
        if (point_coordinates[(point_indices[k] * dim_len) + i] != p) {
          return false;
        }
      }
    }
  }

  return true;
}

void AnimationNodeMotionMatch::KDNode::set_th(float t) { this->split_th = t; }

float AnimationNodeMotionMatch::KDNode::get_th() { return this->split_th; }

void AnimationNodeMotionMatch::KDNode::add_index(uint32_t i) {
  point_indices.append(i);
}

void AnimationNodeMotionMatch::KDNode::clear_indices() {
  point_indices.resize(0);
}

PoolRealArray AnimationNodeMotionMatch::KDNode::get_indices() {
  return point_indices;
}

AnimationNodeMotionMatch::KDNode::KDNode() {
  this->point_indices = {};
  this->split_th = 0;
  this->split_axis = 0;
}

void AnimationNodeMotionMatch::KDNode::leaf_split(
    PoolRealArray point_coordinates, uint32_t dim_len, uint32_t dim,
    uint32_t min_leaves) {
  if (point_coordinates.size() % dim_len != 0) {
    print_line("ERROR: Point coordinates array is of wrong size.");
  } else {
    split_axis = dim % dim_len;
    calculate_threshold(point_coordinates, dim_len);

    left.instance();
    right.instance();

    left->prev = this;
    right->prev = this;
    for (int j = 0; j < this->point_indices.size(); j++) {
      if (point_coordinates[(point_indices[j] * dim_len) + split_axis] >
          split_th) {
        left->point_indices.append(point_indices[j]);
      } else {
        right->point_indices.append(point_indices[j]);
      }
    }

    if (left->point_indices.size() > min_leaves &&
        !are_all_points_same(point_coordinates, dim_len)) {
      left->leaf_split(point_coordinates, dim_len, dim + 1, min_leaves);
    }
    if (right->point_indices.size() > min_leaves &&
        !are_all_points_same(point_coordinates, dim_len)) {
      right->leaf_split(point_coordinates, dim_len, dim + 1, min_leaves);
    }
  }
}

AnimationNodeMotionMatch::KDNode *AnimationNodeMotionMatch::KDNode::get_left() {
  return left.ptr();
}

AnimationNodeMotionMatch::KDNode *
AnimationNodeMotionMatch::KDNode::get_right() {
  return right.ptr();
}

float AnimationNodeMotionMatch::process(float p_time, bool p_seek) {
  PoolRealArray future_traj = Predict_traj(get_parameter(vel), 10);
  float min_cost = std::numeric_limits<float>::max();
  float min_cost_time;
  AnimationPlayer *player = state->player;
  int anim = 0;
  int dup;
  if (first_time == true) {
    dup = -1;
  } else {
    dup = get_parameter(min);
  }

  for (int p = 0; p < keys->size(); p++) {
    if (p != dup) {
      float pos_cost = 0.0f;
      float traj_cost = 0.0f;
      float tot_cost = 0.0f;

      PoolVector<frame_model *>::Read read = keys->read();

      for (int i = 0; i < matching_tracks.size(); i++) {

        Vector<String> s = String(matching_tracks[i]).split(":");
        Vector3 pos = skeleton->get_bone_global_pose(skeleton->find_bone(s[1]))
                          .get_origin();

        for (int po = 0; po < 2; po++) {
          pos_cost += (pos[po] - read[p]->bone_data->read()[i][po]) *
                      (pos[po] - read[p]->bone_data->read()[i][po]);
        }
      }

      for (int t = 0; t < 2; t++) {
        traj_cost += (read[p]->traj->read()[t] - future_traj[t]) *
                     (read[p]->traj->read()[t] - future_traj[t]);
      }

      tot_cost = pos_cost + traj_cost;

      if (tot_cost < min_cost) {
        min_cost = tot_cost;
        min_cost_time = keys->read()[p]->time;
        anim = keys->read()[p]->anim_num;
        set_parameter(min, p);
        print_line(rtos(min_cost));
      }
    }
  }

  if (first_time)
    first_time = false;
  List<StringName> a_nam;
  player->get_animation_list(&a_nam);
  player->play(a_nam[anim]);
  player->seek(min_cost_time, true);
  return 0.0;
}

PoolRealArray AnimationNodeMotionMatch::Predict_traj(Vector3 L_Velocity,
                                                     int samples) {
  PoolRealArray futurepath = {};
  Vector3 c_pos = Vector3();
  float time = 1.0f;
  for (int i = 0; i < samples; i++) {
    for (int j = 0; j < 3; j++) {
      if (j != 1) {
        c_pos[j] = c_pos[j] + L_Velocity[j] * (1 - Math::exp(-time));
        futurepath.append(c_pos[j]);
      }
    }

    time += 1.0f;
  }

  return futurepath;
}
