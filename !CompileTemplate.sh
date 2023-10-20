export SCONS_CACHE="./.SCache"
export SCONS_CACHE_LIMIT=5000

scons profile=TemplatesCustom.py platform=windows target=template_release arch=x86_64
# scons platform=linux target=template_release arch=x86_32
scons profile=TemplatesCustom.py platform=linux target=template_release arch=x86_64

# Android Templates
scons profile=TemplatesCustom.py platform=android target=template_release arch=armv7
scons profile=TemplatesCustom.py platform=android target=template_release arch=arm64v8
cd platform/android/java
call .\gradlew cleanGodotTemplates
call .\gradlew generateGodotTemplates
cd ../../..

REM Web Templates
scons profile=TemplatesWeb.py platform=web target=template_release javascript_eval=no
echo "Compiled!"