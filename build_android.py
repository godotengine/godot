import shutil
import subprocess
import os
from datetime import date, datetime

needs_stopping = False

def exec(cmd, current_working_dir="./", err_pattern="auie"):
    global needs_stopping
    needs_stopping = False
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, cwd=current_working_dir, shell=True)
    while True:
        output = process.stdout.readline()

        poll = process.poll()
        if output == '' and poll is not None:
            break
        if not output:
            break
        if output.strip().decode('ascii').find(err_pattern) != -1:
            needs_stopping = True
            break

        if output:
            print(output.strip().decode('ascii'))
        if process.stderr:
            err = process.stderr.readline()
            if err:
                print(err.strip().decode('ascii'))

    return process.poll()

def main() -> bool:
    global needs_stopping
    print("Compiling godot templates")
    # exec(["scons", "platform=android", "target=template_release", "arch=arm64", "debug_symbols=yes"], err_pattern="scons: building terminated because of errors.")

    if needs_stopping:
        return False

    exec(["scons", "platform=android", "target=template_debug", "arch=arm64", "debug_symbols=yes"], err_pattern="scons: building terminated because of errors.")

    if needs_stopping:
        return False

    print("Done Compiling godot templates")

    print("Generating android templates")
    exec(["gradlew", "generateGodotTemplates"], current_working_dir="./platform/android/java/")

    if needs_stopping:
        return False

    print("Done Generating android templates")

    installed_export_templates = [
        "C:/Users/maxim/AppData/Roaming/Godot/export_templates/4.2.2.stable/"
        # "C:/Users/Maxime/AppData/Roaming/Godot/export_templates/4.2.stable/"
    ]

    cwd = os.getcwd()
    android_template_files = [
        "bin/android_source.zip",
        "bin/android_debug.apk",
        # "bin/android_release.apk"
    ]

    print("Copying android templates")
    for out_export_template in installed_export_templates :
        for f in android_template_files:
            source_file= cwd + "/" + f
            print("Copying " + source_file + " to " + out_export_template)
            shutil.copy(source_file, out_export_template)
    print("Done Copying android templates")

    return True


if __name__ == "__main__":
    if main() == False:
        print("Something went wrong")

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("date and time =", dt_string)