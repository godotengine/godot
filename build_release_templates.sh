#RELEASE TEMPLATES
set -e

BASE_DIR="$( cd "$(dirname "$0")" ; pwd -P )"

cd $BASE_DIR
scons platform=android target=release android_arch=armv7
scons platform=android target=release android_arch=arm64v8
cd platform/android/java
./gradlew generateGodotTemplates    #FOR LINUX AND MAC OS