# iOS build templates.
scons platform=ios target=template_debug tools=no build_feature_profile="kaetram.build" arch=arm64
scons platform=ios target=template_release tools=no build_feature_profile="kaetram.build" arch=arm64
# scons platform=ios target=template_debug tools=no build_feature_profile="kaetram.build" arch=x86_64
# scons platform=ios target=template_release tools=no build_feature_profile="kaetram.build" arch=x86_64
scons platform=ios target=template_debug tools=no build_feature_profile="kaetram.build" ios_simulator=yes arch=arm64
scons platform=ios target=template_release tools=no build_feature_profile="kaetram.build" ios_simulator=yes arch=arm64
scons platform=ios target=template_debug tools=no build_feature_profile="kaetram.build" ios_simulator=yes arch=x86_64
scons platform=ios target=template_release tools=no build_feature_profile="kaetram.build" ios_simulator=yes arch=x86_64

# Create simulator libraries.
cp -r misc/dist/ios_xcode ./bin

cp bin/libgodot.ios.template_debug.arm64.a bin/ios_xcode/libgodot.ios.debug.xcframework/ios-arm64/libgodot.a
lipo -create bin/libgodot.ios.template_debug.arm64.simulator.a bin/libgodot.ios.template_debug.x86_64.simulator.a -output bin/ios_xcode/libgodot.ios.debug.xcframework/ios-arm64_x86_64-simulator/libgodot.a

cp bin/libgodot.ios.template_release.arm64.a bin/ios_xcode/libgodot.ios.release.xcframework/ios-arm64/libgodot.a
lipo -create bin/libgodot.ios.template_release.arm64.simulator.a bin/libgodot.ios.template_release.x86_64.simulator.a -output bin/ios_xcode/libgodot.ios.release.xcframework/ios-arm64_x86_64-simulator/libgodot.a

# macOS build templates.
scons platform=macos target=template_debug tools=no build_feature_profile="kaetram.build" arch=x86_64
scons platform=macos target=template_release tools=no build_feature_profile="kaetram.build" arch=x86_64
scons platform=macos target=template_debug tools=no build_feature_profile="kaetram.build" arch=arm64
scons platform=macos target=template_release tools=no build_feature_profile="kaetram.build" arch=arm64

# # Combine macOS templates.
lipo -create bin/godot.macos.template_release.x86_64 bin/godot.macos.template_release.arm64 -output bin/godot.macos.template_release.universal
lipo -create bin/godot.macos.template_debug.x86_64 bin/godot.macos.template_debug.arm64 -output bin/godot.macos.template_debug.universal

# Create the MacOS app bundle.
cp -r misc/dist/macos_template.app .
mkdir -p macos_template.app/Contents/MacOS
cp bin/godot.macos.template_release.universal macos_template.app/Contents/MacOS/godot_macos_release.universal
cp bin/godot.macos.template_debug.universal macos_template.app/Contents/MacOS/godot_macos_debug.universal
chmod +x macos_template.app/Contents/MacOS/godot_macos*

mv macos_template.app bin/macos_template.app

cd bin

zip -r macos.zip macos_template.app