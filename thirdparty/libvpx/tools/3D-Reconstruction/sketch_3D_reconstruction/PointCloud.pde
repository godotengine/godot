class PointCloud {
  ArrayList<PVector> points;  // array to save points
  IntList point_colors;       // array to save points color
  PVector cloud_mass;
  float[] depth;
  boolean[] real;
  PointCloud() {
    // initialize
    points = new ArrayList<PVector>();
    point_colors = new IntList();
    cloud_mass = new PVector(0, 0, 0);
    depth = new float[width * height];
    real = new boolean[width * height];
  }

  void generate(PImage rgb_img, PImage depth_img, Transform trans) {
    if (depth_img.width != width || depth_img.height != height ||
        rgb_img.width != width || rgb_img.height != height) {
      println("rgb and depth file dimension should be same with window size");
      exit();
    }
    // clear depth and real
    for (int i = 0; i < width * height; i++) {
      depth[i] = 0;
      real[i] = false;
    }
    for (int v = 0; v < height; v++)
      for (int u = 0; u < width; u++) {
        // get depth value (red channel)
        color depth_px = depth_img.get(u, v);
        depth[v * width + u] = depth_px & 0x0000FFFF;
        if (int(depth[v * width + u]) != 0) {
          real[v * width + u] = true;
        }
        point_colors.append(rgb_img.get(u, v));
      }
    for (int v = 0; v < height; v++)
      for (int u = 0; u < width; u++) {
        if (int(depth[v * width + u]) == 0) {
          interpolateDepth(v, u);
        }
        // add transformed pixel as well as pixel color to the list
        PVector pos = trans.transform(u, v, int(depth[v * width + u]));
        points.add(pos);
        // accumulate z value
        cloud_mass = PVector.add(cloud_mass, pos);
      }
  }
  void fillInDepthAlongPath(float d, Node node) {
    node = node.parent;
    while (node != null) {
      int i = node.row;
      int j = node.col;
      if (depth[i * width + j] == 0) {
        depth[i * width + j] = d;
      }
      node = node.parent;
    }
  }
  // interpolate
  void interpolateDepth(int row, int col) {
    if (row < 0 || row >= height || col < 0 || col >= width ||
        int(depth[row * width + col]) != 0) {
      return;
    }
    ArrayList<Node> queue = new ArrayList<Node>();
    queue.add(new Node(row, col, null));
    boolean[] visited = new boolean[width * height];
    for (int i = 0; i < width * height; i++) visited[i] = false;
    visited[row * width + col] = true;
    // Using BFS to Find the Nearest Neighbor
    while (queue.size() > 0) {
      // pop
      Node node = queue.get(0);
      queue.remove(0);
      int i = node.row;
      int j = node.col;
      // if current position have a real depth
      if (depth[i * width + j] != 0 && real[i * width + j]) {
        fillInDepthAlongPath(depth[i * width + j], node);
        break;
      } else {
        // search unvisited 8 neighbors
        for (int r = max(0, i - 1); r < min(height, i + 2); r++) {
          for (int c = max(0, j - 1); c < min(width, j + 2); c++) {
            if (!visited[r * width + c]) {
              visited[r * width + c] = true;
              queue.add(new Node(r, c, node));
            }
          }
        }
      }
    }
  }
  // get point cloud size
  int size() { return points.size(); }
  // get ith position
  PVector getPosition(int i) {
    if (i >= points.size()) {
      println("point position: index " + str(i) + " exceeds");
      exit();
    }
    return points.get(i);
  }
  // get ith color
  color getColor(int i) {
    if (i >= point_colors.size()) {
      println("point color: index " + str(i) + " exceeds");
      exit();
    }
    return point_colors.get(i);
  }
  // get cloud center
  PVector getCloudCenter() {
    if (points.size() > 0) {
      return PVector.div(cloud_mass, points.size());
    }
    return new PVector(0, 0, 0);
  }
  // merge two clouds
  void merge(PointCloud point_cloud) {
    for (int i = 0; i < point_cloud.size(); i++) {
      points.add(point_cloud.getPosition(i));
      point_colors.append(point_cloud.getColor(i));
    }
    cloud_mass = PVector.add(cloud_mass, point_cloud.cloud_mass);
  }
}

class Node {
  int row, col;
  Node parent;
  Node(int row, int col, Node parent) {
    this.row = row;
    this.col = col;
    this.parent = parent;
  }
}
