# Ignore the error if the container is not running
docker stop -t1 godot-rust-compiler || true
docker rm godot-rust-compiler || true
