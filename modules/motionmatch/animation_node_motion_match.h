#ifndef ANIMATION_NODE_MOTION_MATCH_H
#define ANIMATION_NODE_MOTION_MATCH_H

#include "core/reference.h"
#include "frame_model.h"
#include "scene/3d/physics_body.h"
#include "scene/animation/animation_tree.h"
#include <limits>
#include "editor/plugins/animation_tree_editor_plugin.h"

class AnimationNodeMotionMatch : public AnimationRootNode {
  GDCLASS(AnimationNodeMotionMatch, AnimationRootNode)

  Vector<NodePath> matching_tracks;
  // parameters
  StringName vel;
  StringName pos;
  StringName min;

  struct KDNode : public Reference {
    /*th -> Threshold*/
    // Variables
    PoolRealArray point_indices;
    Ref<KDNode> left;
    Ref<KDNode> right;
    float split_th;
    uint32_t split_axis;
    // Methods
    bool are_all_points_same(PoolRealArray point_coordinates, uint32_t dim_len);
    KDNode *prev;
    void calculate_threshold(PoolRealArray points, uint32_t dim_len);

    void add_index(uint32_t i);
    void clear_indices();
    PoolRealArray get_indices();

    void leaf_split(PoolRealArray point_coordinates, uint32_t dim_len,
                    uint32_t dim, uint32_t min_leaves);

    KDNode *get_left();
    KDNode *get_right();
    KDNode *get_prev() { return prev; }

    uint32_t get_split_axis() { return split_axis; }

    void set_th(float th);
    float get_th();
    KDNode();
  };

  float delta_time = 0.0f;

  PoolVector<frame_model *> *keys = new PoolVector<frame_model *>();

  PoolRealArray point_coordinates;
  Ref<KDNode> root;
  int dim_len;          /*no of dimensions*/
  uint32_t start_index; /*Axis relative to which the first split occurs*/
  uint32_t min_leaves;  /*Minimum leafs in nodes at the end level*/
  bool error = false;

  enum errortype { LOAD_POINT_ERROR, QUERY_POINT_ERROR, K_ERROR };

protected:
  static void _bind_methods();

public:
  Skeleton *skeleton;

  virtual void get_parameter_list(List<PropertyInfo> *r_list) const;

  float process(float p_time, bool p_seek);
  bool first_time = true;
  void add_matching_track(const NodePath &p_track_path);
  void remove_matching_track(const NodePath &p_track_path);
  bool is_matching_track(const NodePath &p_track_path) const;
  Vector<NodePath> get_matching_tracks();
  void update_motion_database(AnimationPlayer *p_animation_player);

  errortype err;
  void set_start_index(uint32_t si);
  uint32_t get_start_index();

  void set_min_leaves(uint32_t min_l);
  uint32_t get_min_leaves();

  void add_coordinates(PoolRealArray point);
  void load_coordinates(PoolRealArray points);
  PoolRealArray get_coordinates();
  void clear_coordinates();

  void set_dim_len(uint32_t dim_len);
  uint32_t get_dim_len();

  void calc_root_threshold();

  PoolRealArray KNNSearch(PoolRealArray point, uint32_t k);

  void build_tree();
  KDNode *get_root();
  void clear_root();

  PoolVector<frame_model *> *get_keys_data() { return keys; }

  void set_keys_data(PoolVector<frame_model *> *kys) { keys = kys; }
  void clear_keys() { keys->resize(0); }

  float get_delta_time() { return delta_time; }
  void set_delta_time(float p) { delta_time = p; }

  PoolRealArray Predict_traj(Vector3 L_Velocity, int samples);

  AnimationNodeMotionMatch();
  ~AnimationNodeMotionMatch();
};

#endif // ANIMATION_NODE_MOTION_MATCH_H
