#!/bin/bash

usage() {
	echo "Usage: docker.sh [command] [options]"
	echo "Commands:"
	echo "  --build          Builds the initial docker image"
	echo "  --enter          Enters the docker container"
	echo "  --clean          Remove all stopped containers and dangling images"
	echo "  --help           Show this help message"
}

build() {
	docker build -t riscv64-linux-gnu .
}

clean() {
	docker rm $(docker ps -a -q)
	docker rmi $(docker images -f "dangling=true" -q)
}

enter() {
	echo "Use './build.sh' to build the RISC-V program"
	docker run -it -v .:/usr/src riscv64-linux-gnu
}

case $1 in
	--build)
		build
		;;
	--clean)
		clean
		;;
	--enter)
		enter
		;;
	--help)
		usage
		;;
	*)
		usage
		;;
esac
