rm -rf ../../thirdparty/manifold

degit github:fire/manifold.git#godot ../../thirdparty/manifold --force
degit github:fire/thrust#1.17.X ../../thirdparty/manifold/src/third_party/thrust --force
degit github:g-truc/glm ../../thirdparty/manifold/src/third_party/glm --force

rm -rf ../../thirdparty/manifold/src/third_party/glm/doc
rm -rf ../../thirdparty/manifold/test
rm -rf ../../thirdparty/manifold/extras
rm -rf ../../thirdparty/manifold/bindings
rm -rf ../../thirdparty/manifold/thrust.diff
rm -rf ../../thirdparty/manifold/assimp.diff
rm -rf ../../thirdparty/manifold/clang-format.sh
rm -rf ../../thirdparty/manifold/CMakeLists.txt
rm -rf ../../thirdparty/manifold/src/CMakeLists.txt
rm -rf ../../thirdparty/manifold/third_party/CMakeLists.txt
rm -rf ../../thirdparty/manifold/utilities/CMakeLists.txt
# Keep # rm -rf ../../thirdparty/manifold/collider
rm -rf ../../thirdparty/manifold/deploy.sh
rm -rf ../../thirdparty/manifold/docker-compose.debug.yml
rm -rf ../../thirdparty/manifold/docker-compose.yml
rm -rf ../../thirdparty/manifold/Dockerfile
rm -rf ../../thirdparty/manifold/docs
rm -rf ../../thirdparty/manifold/Doxyfile
rm -rf ../../thirdparty/manifold/flake.lock
rm -rf ../../thirdparty/manifold/flake.nix
# Keep # rm -rf ../../thirdparty/manifold/LICENSE
# Keep # rm -rf ../../thirdparty/manifold/manifold
# Keep # rm -rf ../../thirdparty/manifold/meshIO
# Keep # rm -rf ../../thirdparty/manifold/polygon
# Keep # rm -rf ../../thirdparty/manifold/README.md
rm -rf ../../thirdparty/manifold/samples
# Keep # rm -rf ../../thirdparty/manifold/test
# Keep # rm -rf ../../thirdparty/manifold/third_party
# Keep # rm -rf ../../thirdparty/manifold/tools
# Keep # rm -rf ../../thirdparty/manifold/utilities
rm -rf ../../thirdparty/manifold/.devcontainer
rm -rf ../../thirdparty/manifold/.github
rm -rf ../../thirdparty/manifold/.vscode
rm -rf ../../thirdparty/manifold/.dockerignore
rm -rf ../../thirdparty/manifold/.gitmodules
rm -rf ../../thirdparty/manifold/third_party/glm/doc
rm -rf ../../thirdparty/manifold/third_party/glm/test
rm -rf ../../thirdparty/manifold/third_party/thrust/testing
rm -rf ../../thirdparty/manifold/third_party/thrust/examples
rm -rf ../../thirdparty/manifold/third_party/thrust/cub
