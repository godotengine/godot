{ pkgs ? import <nixpkgs> {} }:
let
  dlopenLibs = with pkgs; [
    libx11 libxcursor libxinerama libxi libxrandr libxext libxkbcommon
    wayland mesa libglvnd libdecor vulkan-loader
    alsa-lib pulseaudio dbus fontconfig
  ];
in pkgs.mkShell {
  nativeBuildInputs = with pkgs; [ pkg-config scons wayland-scanner autoPatchelfHook ];

  buildInputs = dlopenLibs ++ (with pkgs; [
    freetype systemd
    xcbutilcursor xcbutilwm xcbutilkeysyms xcbutilimage
  ]);

  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath dlopenLibs}:$LD_LIBRARY_PATH"
  '';
}
