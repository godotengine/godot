#
# FreeType 2 Type42 module definition
#


# Copyright (C) 2002-2019 by
# David Turner, Robert Wilhelm, and Werner Lemberg.
#
# This file is part of the FreeType project, and may only be used, modified,
# and distributed under the terms of the FreeType project license,
# LICENSE.TXT.  By continuing to use, modify, or distribute this file you
# indicate that you have read the license and understand and accept it
# fully.


FTMODULE_H_COMMANDS += TYPE42_DRIVER

define TYPE42_DRIVER
$(OPEN_DRIVER) FT_Driver_ClassRec, t42_driver_class $(CLOSE_DRIVER)
$(ECHO_DRIVER)type42    $(ECHO_DRIVER_DESC)Type 42 font files with no known extension$(ECHO_DRIVER_DONE)
endef

# EOF
