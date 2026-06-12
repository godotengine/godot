// show grids
void showGrids(int block_size) {
  ortho(-width, 0, -height, 0);
  camera(0, 0, 0, 0, 0, 1, 0, 1, 0);
  stroke(0, 0, 255);
  for (int i = 0; i < height; i += block_size) {
    line(0, i, width, i);
  }
  for (int i = 0; i < width; i += block_size) {
    line(i, 0, i, height);
  }
}

// save the point clould information
void savePointCloud(PointCloud point_cloud, String file_name) {
  String[] positions = new String[point_cloud.points.size()];
  String[] colors = new String[point_cloud.points.size()];
  for (int i = 0; i < point_cloud.points.size(); i++) {
    PVector point = point_cloud.getPosition(i);
    color point_color = point_cloud.getColor(i);
    positions[i] = str(point.x) + ' ' + str(point.y) + ' ' + str(point.z);
    colors[i] = str(((point_color >> 16) & 0xFF) / 255.0) + ' ' +
                str(((point_color >> 8) & 0xFF) / 255.0) + ' ' +
                str((point_color & 0xFF) / 255.0);
  }
  saveStrings(file_name + "_pos.txt", positions);
  saveStrings(file_name + "_color.txt", colors);
}
