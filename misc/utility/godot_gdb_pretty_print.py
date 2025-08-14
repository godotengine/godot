"""
Load this file to your GDB session to enable pretty-printing of some Godot C++ types.

GDB command: `source misc/utility/godot_gdb_pretty_print.py`.

To load these automatically in Visual Studio Code, add the source command to
the `setupCommands` of your configuration in `launch.json`:
```json
"setupCommands": [
...
    {
        "description": "Load custom pretty-printers for Godot types.",
        "text": "source ${workspaceFolder}/misc/utility/godot_gdb_pretty_print.py"
    }
]
```
Other UIs that use GDB under the hood are likely to have their own ways to achieve this.

To debug this script it's easiest to use the interactive python from a command-line
GDB session. Stop at a breakpoint, then use python-interactive to enter the python shell
and acquire a `Value` object using `gdb.selected_frame().read_var("variable name")`.
From there you can figure out how to print it nicely.
"""

import re

import gdb  # type: ignore
import gdb.printing  # type: ignore


# Printer for Godot StringName variables.
class GodotStringNamePrinter:
    def __init__(self, value):
        self.value = value

    def to_string(self):
        return self.value["_data"]["name"]["_cowdata"]["_ptr"]

    # Hint that the object is string-like.
    def display_hint(self):
        return "string"


# Printer for Godot String variables.
class GodotStringPrinter:
    def __init__(self, value):
        self.value = value

    def to_string(self):
        return self.value["_cowdata"]["_ptr"]

    # Hint that the object is string-like.
    def display_hint(self):
        return "string"


# Printer for Godot Vector variables.
class GodotVectorPrinter:
    def __init__(self, value):
        self.value = value

    # The COW (Copy On Write) object does a bunch of pointer arithmetic to access
    # its members.
    # The offsets are constants on the C++ side, optimized out, so not accessible to us.
    # I'll just hard code the observed values and hope they are the same forever.
    # See core/templates/cowdata.h
    SIZE_OFFSET = 8
    DATA_OFFSET = 16

    # Figures out the number of elements in the vector.
    def get_size(self):
        cowdata = self.value["_cowdata"]
        if cowdata["_ptr"] == 0:
            return 0
        else:
            # The ptr member of cowdata does not point to the beginning of the
            # cowdata. It points to the beginning of the data section of the cowdata.
            # To get to the length section, we must back up to the beginning of the struct,
            # then move back forward to the size.
            # cf. CowData::_get_size
            ptr = cowdata["_ptr"].cast(gdb.lookup_type("uint8_t").pointer())
            return int((ptr - self.DATA_OFFSET + self.SIZE_OFFSET).dereference())

    # Lists children of the value, in this case the vector's items.
    def children(self):
        # Return nothing if ptr is null.
        ptr = self.value["_cowdata"]["_ptr"]
        if ptr == 0:
            return
        # Yield the items one by one.
        for i in range(self.get_size()):
            yield str(i), (ptr + i).dereference()

    def to_string(self):
        return "%s [%d]" % (self.value.type.name, self.get_size())

    # Hint that the object is array-like.
    def display_hint(self):
        return "array"


VECTOR_REGEX = re.compile("^Vector<.*$")


# Tries to find a pretty printer for a debugger value.
def lookup_pretty_printer(value):
    if value.type.name == "StringName":
        return GodotStringNamePrinter(value)
    if value.type.name == "String":
        return GodotStringPrinter(value)
    if value.type.name and VECTOR_REGEX.match(value.type.name):
        return GodotVectorPrinter(value)
    return None


# Register our printer lookup function.
# The first parameter could be used to limit the scope of the printer
# to a specific object file, but that is unnecessary for us.
gdb.printing.register_pretty_printer(None, lookup_pretty_printer)
