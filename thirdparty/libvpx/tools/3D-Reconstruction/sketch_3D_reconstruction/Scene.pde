class Scene {
  PointCloud point_cloud;
  ArrayList<Triangle> mesh;
  BVH bvh;
  MotionField motion_field;
  Camera last_cam;
  Camera current_cam;
  int frame_count;

  Scene(Camera camera, PointCloud point_cloud, MotionField motion_field) {
    this.point_cloud = point_cloud;
    this.motion_field = motion_field;
    mesh = new ArrayList<Triangle>();
    for (int v = 0; v < height - 1; v++)
      for (int u = 0; u < width - 1; u++) {
        PVector p1 = point_cloud.getPosition(v * width + u);
        PVector p2 = point_cloud.getPosition(v * width + u + 1);
        PVector p3 = point_cloud.getPosition((v + 1) * width + u + 1);
        PVector p4 = point_cloud.getPosition((v + 1) * width + u);
        color c1 = point_cloud.getColor(v * width + u);
        color c2 = point_cloud.getColor(v * width + u + 1);
        color c3 = point_cloud.getColor((v + 1) * width + u + 1);
        color c4 = point_cloud.getColor((v + 1) * width + u);
        mesh.add(new Triangle(p1, p2, p3, c1, c2, c3));
        mesh.add(new Triangle(p3, p4, p1, c3, c4, c1));
      }
    bvh = new BVH(mesh);
    last_cam = camera.copy();
    current_cam = camera;
    frame_count = 0;
  }

  void run() {
    last_cam = current_cam.copy();
    current_cam.run();
    motion_field.update(last_cam, current_cam, point_cloud, bvh);
    frame_count += 1;
  }

  void render(boolean show_motion_field) {
    // build mesh
    current_cam.open();
    noStroke();
    for (int i = 0; i < mesh.size(); i++) {
      Triangle t = mesh.get(i);
      t.render();
    }
    if (show_motion_field) {
      current_cam.close();
      motion_field.render();
    }
  }

  void save(String path) { saveFrame(path + "_" + str(frame_count) + ".png"); }

  void saveMotionField(String path) {
    motion_field.save(path + "_" + str(frame_count) + ".txt");
  }
}
