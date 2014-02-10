
# com.android.godot
# for this, need to push gdbserver to device
# adb push [location]/gdbserver /data/local

# run

DEBUG_PORT=5039
GDB_PATH="$ANDROID_NDK_ROOT/toolchains/arm-linux-androideabi-4.4.3/prebuilt/linux-x86/bin/arm-linux-androideabi-gdb"

#kill existing previous gdbserver if exists
adb shell killall gdbserver
run-as com.android.godot killall com.android.godot
run-as com.android.godot killall gdbserver
#get data dir of the app
adb pull /system/bin/app_process app_process

DATA_DIR=`adb shell run-as com.android.godot /system/bin/sh -c pwd | tr -d '\n\r'`
echo "DATA DIR IS $DATA_DIR"
#start app
adb shell am start -n com.android.godot/com.android.godot.Godot
#get the pid of the app
PID=`adb shell pidof com.android.godot | tr -d '\n\r'`
echo "PID IS: $PID hoho"
#launch gdbserver
DEBUG_SOCKET=debug-socket
#echo adb shell /data/local/gdbserver +debug-socket --attach $PID
adb shell run-as com.android.godot lib/gdbserver +$DEBUG_SOCKET --attach $PID &
sleep 2s
#adb shell /data/local/gdbserver localhost:$DEBUG_PORT --attach $PID &
#setup network connection
adb forward tcp:$DEBUG_PORT localfilesystem:$DATA_DIR/$DEBUG_SOCKET
cp gdb.setup.base gdb.setup
echo "target remote :$DEBUG_PORT" >> gdb.setup
#echo "file 
$GDB_PATH -x gdb.setup




