#!/usr/bin/env python3
#
# Script to sort the game controller database entries in SDL_gamepad.c

import re


filename = "SDL_gamepad_db.h"
input = open(filename)
output = open(f"{filename}.new", "w")
parsing_controllers = False
controllers = []
controller_guids = {}
conditionals = []
split_pattern = re.compile(r'([^"]*")([^,]*,)([^,]*,)([^"]*)(".*)')
#                                     BUS (1)         CRC (3,2)                       VID (5,4)                       (6)   PID (8,7)                       (9)   VERSION (11,10)                 MISC (12)
standard_guid_pattern = re.compile(r'^([0-9a-fA-F]{4})([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})(0000)([0-9a-fA-F]{2})([0-9a-fA-F]{2})(0000)([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{4},)$')

# These chipsets are used in multiple controllers with different mappings,
# without enough unique information to differentiate them. e.g.
# https://github.com/gabomdq/SDL_GameControllerDB/issues/202
invalid_controllers = (
    ('0079', '0006', '0000'), # DragonRise Inc. Generic USB Joystick
    ('0079', '0006', '6120'), # DragonRise Inc. Generic USB Joystick
    ('04b4', '2412', 'c529'), # Flydigi Vader 2, Vader 2 Pro, Apex 2, Apex 3, Apex 4
    ('16c0', '05e1', '0000'), # Xinmotek Controller
)

def find_element(prefix, bindings):
    i=0
    for element in bindings:
        if element.startswith(prefix):
            return i
        i=(i + 1)

    return -1

def get_crc_from_entry(entry):
    crc = ""
    line = "".join(entry)
    bindings = line.split(",")
    pos = find_element("crc:", bindings)
    if pos >= 0:
        crc = bindings[pos][4:]
    return crc

def save_controller(line):
    global controllers
    match = split_pattern.match(line)
    entry = [ match.group(1), match.group(2), match.group(3) ]
    bindings = sorted(match.group(4).split(","))
    if (bindings[0] == ""):
        bindings.pop(0)

    name = entry[2].rstrip(',')

    crc = ""
    pos = find_element("crc:", bindings)
    if pos >= 0:
        crc = bindings[pos] + ","
        bindings.pop(pos)

    guid_match = standard_guid_pattern.match(entry[1])
    if guid_match:
        groups = guid_match.groups()
        crc_value = groups[2] + groups[1]
        vid_value = groups[4] + groups[3]
        pid_value = groups[7] + groups[6]
        version_value = groups[10] + groups[9]
        #print("CRC: %s, VID: %s, PID: %s, VERSION: %s" % (crc_value, vid_value, pid_value, version_value))

        if crc_value == "0000":
            if crc != "":
                crc_value = crc[4:-1]
        else:
            print("Extracting CRC from GUID of " + name)
            entry[1] = groups[0] + "0000" + "".join(groups[3:])
            crc = "crc:" + crc_value + ","

        if (vid_value, pid_value, crc_value) in invalid_controllers:
            print("Controller '%s' not unique, skipping" % name)
            return

    pos = find_element("type", bindings)
    if pos >= 0:
        bindings.insert(0, bindings.pop(pos))

    pos = find_element("platform", bindings)
    if pos >= 0:
        bindings.insert(0, bindings.pop(pos))

    pos = find_element("sdk", bindings)
    if pos >= 0:
        bindings.append(bindings.pop(pos))

    pos = find_element("hint:", bindings)
    if pos >= 0:
        bindings.append(bindings.pop(pos))

    entry.extend(crc)
    entry.extend(",".join(bindings) + ",")
    entry.append(match.group(5))
    controllers.append(entry)

    entry_id = entry[1] + get_crc_from_entry(entry)
    if ',sdk' in line or ',hint:' in line:
        conditionals.append(entry_id)

def write_controllers():
    global controllers
    global controller_guids
    # Check for duplicates
    for entry in controllers:
        entry_id = entry[1] + get_crc_from_entry(entry)
        if (entry_id in controller_guids and entry_id not in conditionals):
            current_name = entry[2]
            existing_name = controller_guids[entry_id][2]
            print("Warning: entry '%s' is duplicate of entry '%s'" % (current_name, existing_name))

            if (not current_name.startswith("(DUPE)")):
                entry[2] = f"(DUPE) {current_name}"

            if (not existing_name.startswith("(DUPE)")):
                controller_guids[entry_id][2] = f"(DUPE) {existing_name}"

        controller_guids[entry_id] = entry

    for entry in sorted(controllers, key=lambda entry: f"{entry[2]}-{entry[1]}"):
        line = "".join(entry) + "\n"
        line = line.replace("\t", "    ")
        if not line.endswith(",\n") and not line.endswith("*/\n") and not line.endswith(",\r\n") and not line.endswith("*/\r\n"):
            print("Warning: '%s' is missing a comma at the end of the line" % (line))
        output.write(line)

    controllers = []
    controller_guids = {}

for line in input:
    if parsing_controllers:
        if (line.startswith("{")):
            output.write(line)
        elif (line.startswith("    NULL")):
            parsing_controllers = False
            write_controllers()
            output.write(line)
        elif (line.startswith("#if")):
            print(f"Parsing {line.strip()}")
            output.write(line)
        elif ("SDL_PRIVATE_GAMEPAD_DEFINITIONS" in line):
            write_controllers()
            output.write(line)
        elif (line.startswith("#endif")):
            write_controllers()
            output.write(line)
        else:
            save_controller(line)
    else:
        if (line.startswith("static const char *")):
            parsing_controllers = True

        output.write(line)

output.close()
print(f"Finished writing {filename}.new")
