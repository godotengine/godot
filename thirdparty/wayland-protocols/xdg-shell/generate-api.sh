#! /bin/sh

PROTOCOL_PATH="${PROTOCOL_PATH:-/usr/share/wayland-protocols/stable/xdg-shell/xdg-shell.xml}"

if ! test -f "$PROTOCOL_PATH"
then
	printf 'No protocol file found at %s.\n' "$PROTOCOL_PATH"
	return 1
fi

wayland-scanner client-header "$PROTOCOL_PATH" xdg-shell.h
wayland-scanner private-code "$PROTOCOL_PATH" xdg-shell.c
