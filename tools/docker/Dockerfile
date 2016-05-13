FROM ubuntu:14.04
MAINTAINER Mohammad Rezai, https://github.com/mrezai
WORKDIR /godot-dev
COPY scripts/install-android-tools /godot-dev/
ENV DEBIAN_FRONTEND noninteractive
RUN dpkg --add-architecture i386 && \
	apt-get update && \
	apt-get upgrade -y && \
	apt-get install --no-install-recommends -y -q \
	build-essential gcc-multilib g++-multilib mingw32 mingw-w64 scons pkg-config libx11-dev libxcursor-dev \
	libasound2-dev libfreetype6-dev libgl1-mesa-dev libglu-dev libssl-dev libxinerama-dev libudev-dev \
	git wget openjdk-7-jdk libbcprov-java libc6:i386 libncurses5:i386 libstdc++6:i386 zlib1g:i386 lib32z1

