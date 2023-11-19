set SCONS_CACHE=.\.SCache
set SCONS_CACHE_LIMIT=5000
scons profile=TemplatesCustom.py platform=windows target=template_release arch=x86_64

REM Linux Templates
REM scons platform=linux target=template_release arch=x86_32
REM scons profile=TemplatesCustom.py platform=linux target=template_release arch=x86_64

REM Android Templates
EWM scons profile=TemplatesCustom.py platform=android target=template_release arch=arm32
REM scons profile=TemplatesCustom.py platform=android target=template_release arch=arm64
REM cd platform/android/java
REM call .\gradlew cleanGodotTemplates
REM call .\gradlew generateGodotTemplates
REM cd ../../..

REM Web Templates
REM scons profile=TemplatesWeb.py platform=web target=template_release javascript_eval=no
echo "Compiled!"
