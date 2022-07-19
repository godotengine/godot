#
# FreeType 2 CFF module definition
#


# Copyright (C) 1996-2020 by
# David Turner, Robert Wilhelm, and Werner Lemberg.
#
# This file is part of the FreeType project, and may only be used, modified,
# and distributed under the terms of the FreeType project license,
# LICENSE.TXT.  By continuing to use, modify, or distribute this file you
# indicate that you have read the license and understand and accept it
# fully.


FTMODULE_H_COMMANDS += CFF_DRIVER

define CFF_DRIVER
$(OPEN_DRIVER) FT_Driver_ClassRec, cff_driver_class $(CLOSE_DRIVER)
$(ECHO_DRIVER)cff       $(ECHO_DRIVER_DESC)OpenType fonts with extension *.otf$(ECHO_DRIVER_DONE)
endef

# EOF
