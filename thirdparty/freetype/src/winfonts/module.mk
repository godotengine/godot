#
# FreeType 2 Windows FNT/FON module definition
#


# Copyright 1996-2017 by
# David Turner, Robert Wilhelm, and Werner Lemberg.
#
# This file is part of the FreeType project, and may only be used, modified,
# and distributed under the terms of the FreeType project license,
# LICENSE.TXT.  By continuing to use, modify, or distribute this file you
# indicate that you have read the license and understand and accept it
# fully.


FTMODULE_H_COMMANDS += WINDOWS_DRIVER

define WINDOWS_DRIVER
$(OPEN_DRIVER) FT_Driver_ClassRec, winfnt_driver_class $(CLOSE_DRIVER)
$(ECHO_DRIVER)winfnt    $(ECHO_DRIVER_DESC)Windows bitmap fonts with extension *.fnt or *.fon$(ECHO_DRIVER_DONE)
endef

# EOF
