/*The dataset is from
 *Computer Vision Group
 *TUM Department of Informatics Technical
 *University of Munich
 *https://vision.in.tum.de/data/datasets/rgbd-dataset/download#freiburg1_xyz
 */
Scene scene;
void setup() {
  size(640, 480, P3D);
  // default settings
  int frame_no = 0;            // frame number
  float fov = PI / 3;          // field of view
  int block_size = 8;          // block size
  float normalizer = 5000.0f;  // normalizer
  // initialize
  PointCloud point_cloud = new PointCloud();
  // synchronized rgb, depth and ground truth
  String head = "../data/";
  String[] rgb_depth_gt = loadStrings(head + "rgb_depth_groundtruth.txt");
  // read in rgb and depth image file paths as well as corresponding camera
  // posiiton and quaternion
  String[] info = split(rgb_depth_gt[frame_no], ' ');
  String rgb_path = head + info[1];
  String depth_path = head + info[3];
  float tx = float(info[7]), ty = float(info[8]),
        tz = float(info[9]);  // real camera position
  float qx = float(info[10]), qy = float(info[11]), qz = float(info[12]),
        qw = float(info[13]);  // quaternion

  // build transformer
  Transform trans =
      new Transform(tx, ty, tz, qx, qy, qz, qw, fov, width, height, normalizer);
  PImage rgb = loadImage(rgb_path);
  PImage depth = loadImage(depth_path);
  // generate point cloud
  point_cloud.generate(rgb, depth, trans);
  // initialize camera
  Camera camera = new Camera(fov, new PVector(0, 0, 0), new PVector(0, 0, 1),
                             new PVector(0, 1, 0));
  // initialize motion field
  MotionField motion_field = new MotionField(block_size);
  // initialize scene
  scene = new Scene(camera, point_cloud, motion_field);
}
boolean inter = false;
void draw() {
  background(0);
  // run camera dragged mouse to rotate camera
  // w: go forward
  // s: go backward
  // a: go left
  // d: go right
  // up arrow: go up
  // down arrow: go down
  //+ increase move speed
  //- decrease move speed
  // r: rotate the camera
  // b: reset to initial position
  scene.run();  // true: make interpolation; false: do not make
                // interpolation
  if (keyPressed && key == 'o') {
    inter = true;
  }
  scene.render(
      false);  // true: turn on motion field; false: turn off motion field
  // save frame with no motion field
  scene.save("../data/frame/raw");
  background(0);
  scene.render(true);
  showGrids(scene.motion_field.block_size);
  // save frame with motion field
  scene.save("../data/frame/raw_mv");
  scene.saveMotionField("../data/frame/mv");
}
