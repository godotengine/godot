PROJECT_PATH=${1:-.}
pushd $PROJECT_PATH
docker run --name godot-rust-compiler -dv .:/usr/src riscv64-rust
popd
