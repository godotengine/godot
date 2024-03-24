#include "unknoterImpl.h"

void UnknoterNode::_bind_methods(){
  ClassDB::bind_method(D_METHOD("reset", "players", "width", "height"), &UnknoterNode::reset);
  ClassDB::bind_method(D_METHOD("get_edge_player", "x", "y"), &UnknoterNode::get_edge_player);
  ClassDB::bind_method(D_METHOD("get_upper_vertex_player", "x", "y"), &UnknoterNode::get_upper_vertex_player);
  ClassDB::bind_method(D_METHOD("get_lower_vertex_player", "x", "y"), &UnknoterNode::get_lower_vertex_player);
  ClassDB::bind_method(D_METHOD("can_player_flip_vertex", "player", "x", "y"), &UnknoterNode::can_player_flip_vertex);
  ClassDB::bind_method(D_METHOD("can_player_shift_edges", "player", "x", "y", "select_offset", "perpendicular_offset"), &UnknoterNode::can_player_shift_edges);
  ClassDB::bind_method(D_METHOD("flip_vertex", "x", "y"), &UnknoterNode::flip_vertex);
  ClassDB::bind_method(D_METHOD("shift_edges", "x", "y", "select_offset", "perpendicular_offset"), &UnknoterNode::shift_edges);
}

UnknoterNode::UnknoterNode()
  : impl_(new UnknoterImpl())
{}

UnknoterNode::~UnknoterNode() = default;

void UnknoterNode::reset(int players, int width, int height) {
  impl_->reset(players, width, height);
}

int UnknoterNode::get_edge_player(int x, int y) {
  return impl_->get_edge_player(x, y);
}

int UnknoterNode::get_upper_vertex_player(int x, int y) {
  return impl_->get_upper_vertex_player(x, y);
}

int UnknoterNode::get_lower_vertex_player(int x, int y) {
  return impl_->get_lower_vertex_player(x, y);
}

bool UnknoterNode::can_player_flip_vertex(int player, int x, int y) {
  return impl_->can_player_flip_vertex(player, x, y);
}

bool UnknoterNode::can_player_shift_edges(int player, int x, int y, int select_offset, int perpendicular_offset) {
  return impl_->can_player_shift_edges(player, x, y, select_offset, perpendicular_offset);
}

void UnknoterNode::flip_vertex(int x, int y) {
  impl_->flip_vertex(x, y);
}

void UnknoterNode::shift_edges(int x, int y, int select_offset, int perpendicular_offset) {
  impl_->shift_edges(x, y, select_offset, perpendicular_offset);
}
