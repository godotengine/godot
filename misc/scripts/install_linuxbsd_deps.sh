#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

print_info() {
	echo -e "\e[1m[*] $1\e[0m"
}

print_success() {
	echo -e "\e[1;92m[*] $1\e[0m"
}

print_warning() {
	echo -e "\e[1;93m[!] Warning: $1\e[0m"
}

print_error() {
	echo -e "\e[1;91m[!] Error: $1\e[0m"
}

# The detection below also covers derivatives such as Linux Mint and Manjaro.
if [[ -f "/etc/alpine-release" ]]; then
	distribution="alpine"
elif [[ -f "/etc/arch-release" ]]; then
	distribution="arch"
elif [[ -f "/etc/lsb-release" ]]; then
	# Ubuntu must be checked before checking for Debian as Ubuntu also has `/etc/debian_version`.
	distribution="ubuntu"
	ubuntu_version_year="$(grep DISTRIB_RELEASE /etc/lsb-release | cut -d = -f 2 | cut -d . -f 1)"
	ubuntu_version_month="$(grep DISTRIB_RELEASE /etc/lsb-release | cut -d = -f 2 | cut -d . -f 2)"
	# Ubuntu 18.04 is required for out-of-the-box support.
	if [[ "$ubuntu_version_year" -lt 18 ]]; then
		print_warning "Your Ubuntu version ($ubuntu_version_year.$ubuntu_version_month) does not include GCC 7 or later, which is required to compile Godot.\n    You'll have to install a more recent GCC version using a third-party PPA, or upgrade to a newer Ubuntu version.\n    PPA: https://launchpad.net/~ubuntu-toolchain-r/+archive/ubuntu/test"
	fi
elif [[ -f "/etc/debian_version" ]]; then
	distribution="debian"
	debian_version="$(cut -d "." -f 1 </etc/debian_version)"
	# Debian 11 (bullseye) is required for out-of-the-box support.
	if [[ "$debian_version" -lt 11 ]]; then
		print_warning "Your Debian version ($debian_version) does not include GCC 7 or later, which is required to compile Godot.\n    You'll have to install a more recent GCC version using a third-party repository, or upgrade to a newer Debian version."
	fi
elif [[ -f "/etc/fedora-release" ]]; then
	distribution="fedora"
elif [[ -f "/etc/gentoo-release" ]]; then
	distribution="gentoo"
elif [[ -f "/etc/mageia-release" ]]; then
	# TODO: Is this the correct way to detect Mageia?
	distribution="mageia"
elif [[ -f "/etc/SuSE-release" ]]; then
	# TODO: Is this the correct way to detect openSUSE?
	distribution="opensuse"
elif [[ -f "/etc/solus-release" ]]; then
	# TODO: Is this the correct way to detect Solus?
	distribution="solus"
else
	# TODO: Support FreeBSD, OpenBSD and NetBSD.
	print_error "Couldn't detect your Linux distribution. No dependencies have been installed.\n    You'll have to install dependencies manually using your distribution's package manager."
	exit 1
fi

print_info "Detected Linux distribution: $distribution"

if [[ -n "$(command -v sudo)" ]]; then
	# Use `sudo` when needed.
	sudo="sudo"
else
	# Don't use `sudo`.
	sudo=""
fi

print_info "Installing dependencies... (you may be prompted for your administrator password)"

case "$distribution" in
alpine)
	$sudo apk add \
		py3-pip pkgconf gcc g++ libx11-dev libxcursor-dev libxinerama-dev \
		libxi-dev libxrandr-dev libexecinfo-dev
	;;

arch)
	$sudo pacman -S --needed \
		python-pip pkgconf gcc libxcursor libxinerama libxi libxrandr mesa glu \
		libglvnd alsa-lib pulseaudio yasm
	;;

debian | ubuntu)
	$sudo apt-get update -qq
	$sudo apt-get install \
		build-essential python3-pip pkg-config libx11-dev libxcursor-dev \
		libxinerama-dev libgl1-mesa-dev libglu-dev libasound2-dev libpulse-dev \
		libudev-dev libxi-dev libxrandr-dev yasm
	;;

fedora)
	$sudo dnf install \
		python3-pip pkgconfig gcc-c++ libX11-devel libXcursor-devel \
		libXrandr-devel libXinerama-devel libXi-devel mesa-libGL-devel \
		mesa-libGLU-devel alsa-lib-devel pulseaudio-libs-devel libudev-devel yasm
	;;

gentoo)
	$sudo emerge -an \
		dev-python/pip x11-libs/libX11 x11-libs/libXcursor x11-libs/libXinerama \
		x11-libs/libXi media-libs/mesa media-libs/glu media-libs/alsa-lib \
		media-sound/pulseaudio dev-lang/yasm
	;;

mageia)
	$sudo urpmi \
		task-c++-devel python3-pip pkgconfig "pkgconfig(alsa)" "pkgconfig(glu)" \
		"pkgconfig(libpulse)" "pkgconfig(udev)" "pkgconfig(x11)" \
		"pkgconfig(xcursor)" "pkgconfig(xinerama)" "pkgconfig(xi)" \
		"pkgconfig(xrandr)" yasm
	;;

opensuse)
	$sudo zypper install \
		python3-pip pkgconfig libX11-devel libXcursor-devel libXrandr-devel \
		libXinerama-devel libXi-devel Mesa-libGL-devel alsa-devel \
		libpulse-devel libudev-devel libGLU1 yasm
	;;

solus)
	$sudo eopkg install -c \
		system.devel pip libxcursor-devel libxinerama-devel libxi-devel \
		libxrandr-devel mesalib-devel libglu alsa-lib-devel pulseaudio-devel yasm
	;;
esac

# Install SCons via pip to ensure it uses Python 3 (and not Python 2).
# This also makes it possible to get a more recent SCons version compared to
# what's offered in the distribution repository.
# Use `--user` to avoid interfering with system-wide Python tools.
# `--user` also removes the `sudo` requirement.
python3 -m pip install --user --upgrade scons

# shellcheck disable=SC2016
print_success 'Dependencies successfully installed! You can now build Godot using the `scons` command.'
# shellcheck disable=SC2016
print_info 'Make sure "$HOME/.local/bin" is in your `PATH` environment variable so that `scons` can be found.'
print_info 'If not, run the following command to prepend it for the current session:'
# shellcheck disable=SC2016
echo '    export PATH="$HOME/.local/bin:$PATH"'
