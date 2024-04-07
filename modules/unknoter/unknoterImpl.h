#include "unknoterNode.h"

#include <vector>

class UnknoterImpl {
  int width;
  int height;
  std::vector<std::vector<int>> coords_to_player;

public:
  void reset(int players, int width, int height);

  int get_edge_player(int x, int y);
  int get_upper_vertex_player(int x, int y);
  int get_lower_vertex_player(int x, int y);

  int get_width();
  int get_height();

  bool can_player_flip_vertex(int player, int x, int y);
  bool can_player_shift_edges(int player, int x, int y, int select_offset, int perpendicular_offset);

  void flip_vertex(int x, int y);
  void shift_edges(int x, int y, int select_offset, int perpendicular_offset);

  void _set_field(const std::vector<std::vector<int>>& field);
};
