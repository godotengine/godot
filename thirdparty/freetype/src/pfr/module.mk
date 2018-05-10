#
# FreeType 2 PFR module definition
#


# Copyright 2002-2018 by
# David Turner, Robert Wilhelm, and Werner Lemberg.
#
# This file is part of the FreeType project, and may only be used, modified,
# and distributed under the terms of the FreeType project license,
# LICENSE.TXT.  By continuing to use, modify, or distribute this file you
# indicate that you have read the license and understand and accept it
# fully.


FTMODULE_H_COMMANDS += PFR_DRIVER

define PFR_DRIVER
$(OPEN_DRIVER) FT_Driver_ClassRec, pfr_driver_class $(CLOSE_DRIVER)
$(ECHO_DRIVER)pfr       $(ECHO_DRIVER_DESC)PFR/TrueDoc font files with extension *.pfr$(ECHO_DRIVER_DONE)
endef

# EOF
