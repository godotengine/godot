export SCONS_CACHE="./.SCache"
export SCONS_CACHE_LIMIT=5000


# Clear current compiled things
rm -rf ./bin

#Compile Editor
scons target=editor profile=EditorCustom.py 

# Windows

# Linux
scons profile=TemplatesCustom.py platform=linuxbsd target=template_release arch=x86_64

# Android Templates
export ANDROID_SDK_ROOT=/home/gopher/Android/Sdk/
scons profile=TemplatesCustom.py platform=android target=template_release arch=armv7
scons profile=TemplatesCustom.py platform=android target=template_release arch=arm64v8
cd platform/android/java
call .\gradlew cleanGodotTemplates
call .\gradlew generateGodotTemplates
cd ../../..

# Web Template
scons profile=TemplatesWeb.py platform=web target=template_release javascript_eval=no
echo "Compiled!"