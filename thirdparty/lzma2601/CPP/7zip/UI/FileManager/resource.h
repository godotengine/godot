#include "resourceGui.h"

#define IDR_MENUBAR1      70
#define IDM_MENU          71
#define IDR_ACCELERATOR1  72

#define IDB_ADD      100
#define IDB_EXTRACT  101
#define IDB_TEST     102
#define IDB_COPY     103
#define IDB_MOVE     104
#define IDB_DELETE   105
#define IDB_INFO     106

#define IDB_ADD2     150
#define IDB_EXTRACT2 151
#define IDB_TEST2    152
#define IDB_COPY2    153
#define IDB_MOVE2    154
#define IDB_DELETE2  155
#define IDB_INFO2    156

#define IDM_HASH_ALL             101
#define IDM_CRC32                102
#define IDM_CRC64                103
#define IDM_SHA1                 104
#define IDM_SHA256               105
#define IDM_SHA384               106
#define IDM_SHA512               107
#define IDM_SHA3_256             108
#define IDM_XXH64                120
#define IDM_BLAKE2SP             121
#define IDM_MD5                  122

#define IDM_FILE                 500
#define IDM_EDIT                 501
#define IDM_VIEW                 502
#define IDM_FAVORITES            503
#define IDM_TOOLS                504
#define IDM_HELP                 505

#define IDM_OPEN                 540
#define IDM_OPEN_INSIDE          541
#define IDM_OPEN_OUTSIDE         542
#define IDM_FILE_VIEW            543
#define IDM_FILE_EDIT            544
#define IDM_RENAME               545
#define IDM_COPY_TO              546
#define IDM_MOVE_TO              547
#define IDM_DELETE               548
#define IDM_SPLIT                549
#define IDM_COMBINE              550
#define IDM_PROPERTIES           551
#define IDM_COMMENT              552
#define IDM_CRC                  553
#define IDM_DIFF                 554
#define IDM_CREATE_FOLDER        555
#define IDM_CREATE_FILE          556
// #define IDM_EXIT                 557
#define IDM_LINK                 558
#define IDM_ALT_STREAMS          559

#define IDM_VER_EDIT             580
#define IDM_VER_COMMIT           581
#define IDM_VER_REVERT           582
#define IDM_VER_DIFF             583

#define IDM_OPEN_INSIDE_ONE      590
#define IDM_OPEN_INSIDE_PARSER   591

#define IDM_SELECT_ALL           600
#define IDM_DESELECT_ALL         601
#define IDM_INVERT_SELECTION     602
#define IDM_SELECT               603
#define IDM_DESELECT             604
#define IDM_SELECT_BY_TYPE       605
#define IDM_DESELECT_BY_TYPE     606

#define IDM_VIEW_LARGE_ICONS     700
#define IDM_VIEW_SMALL_ICONS     701
#define IDM_VIEW_LIST            702
#define IDM_VIEW_DETAILS         703

#define IDM_VIEW_ARANGE_BY_NAME  710
#define IDM_VIEW_ARANGE_BY_TYPE  711
#define IDM_VIEW_ARANGE_BY_DATE  712
#define IDM_VIEW_ARANGE_BY_SIZE  713

#define IDM_VIEW_ARANGE_NO_SORT  730
#define IDM_VIEW_FLAT_VIEW       731
#define IDM_VIEW_TWO_PANELS      732
#define IDM_VIEW_TOOLBARS        733
#define IDM_OPEN_ROOT_FOLDER     734
#define IDM_OPEN_PARENT_FOLDER   735
#define IDM_FOLDERS_HISTORY      736
#define IDM_VIEW_REFRESH         737
#define IDM_VIEW_AUTO_REFRESH    738
// #define IDM_VIEW_SHOW_DELETED    739
// #define IDM_VIEW_SHOW_STREAMS    740

#define IDM_VIEW_ARCHIVE_TOOLBAR            750
#define IDM_VIEW_STANDARD_TOOLBAR           751
#define IDM_VIEW_TOOLBARS_LARGE_BUTTONS     752
#define IDM_VIEW_TOOLBARS_SHOW_BUTTONS_TEXT 753

#define IDM_VIEW_TIME_POPUP      760
#define IDM_VIEW_TIME            761
#define IDM_VIEW_TIME_UTC        799

#define IDM_ADD_TO_FAVORITES     800
#define IDS_BOOKMARK             801

#define IDM_OPTIONS              900
#define IDM_BENCHMARK            901
#define IDM_BENCHMARK2           902
#define IDM_TEMP_DIR             910

#define IDM_HELP_CONTENTS        960
#define IDM_ABOUT                961

#define IDS_OPTIONS                     2100

#define IDS_N_SELECTED_ITEMS            3002

#define IDS_FILE_EXIST                  3008
#define IDS_WANT_UPDATE_MODIFIED_FILE   3009
#define IDS_CANNOT_UPDATE_FILE          3010
#define IDS_CANNOT_START_EDITOR         3011
#define IDS_VIRUS                       3012
#define IDS_MESSAGE_UNSUPPORTED_OPERATION_FOR_LONG_PATH_FOLDER  3013
#define IDS_SELECT_ONE_FILE             3014
#define IDS_SELECT_FILES                3015
#define IDS_TOO_MANY_ITEMS              3016

#define IDS_COPY                        6000
#define IDS_MOVE                        6001
#define IDS_COPY_TO                     6002
#define IDS_MOVE_TO                     6003
#define IDS_COPYING                     6004
// #define IDS_MOVING                      6005
#define IDS_RENAMING                    6006

#define IDS_OPERATION_IS_NOT_SUPPORTED  6008
#define IDS_ERROR_RENAMING              6009
#define IDS_CONFIRM_FILE_COPY           6010
#define IDS_WANT_TO_COPY_FILES          6011

#define IDS_CONFIRM_FILE_DELETE         6100
#define IDS_CONFIRM_FOLDER_DELETE       6101
#define IDS_CONFIRM_ITEMS_DELETE        6102
#define IDS_WANT_TO_DELETE_FILE         6103
#define IDS_WANT_TO_DELETE_FOLDER       6104
#define IDS_WANT_TO_DELETE_ITEMS        6105
#define IDS_DELETING                    6106
#define IDS_ERROR_DELETING              6107
#define IDS_ERROR_LONG_PATH_TO_RECYCLE  6108

#define IDS_CREATE_FOLDER               6300
#define IDS_CREATE_FILE                 6301
#define IDS_CREATE_FOLDER_NAME          6302
#define IDS_CREATE_FILE_NAME            6303
#define IDS_CREATE_FOLDER_DEFAULT_NAME  6304
#define IDS_CREATE_FILE_DEFAULT_NAME    6305
#define IDS_CREATE_FOLDER_ERROR         6306
#define IDS_CREATE_FILE_ERROR           6307

#define IDS_COMMENT                     6400
#define IDS_COMMENT2                    6401
#define IDS_SELECT                      6402
#define IDS_DESELECT                    6403
#define IDS_SELECT_MASK                 6404

#define IDS_PROPERTIES                  6600
#define IDS_FOLDERS_HISTORY             6601

#define IDS_COMPUTER                    7100
#define IDS_NETWORK                     7101
#define IDS_DOCUMENTS                   7102
#define IDS_SYSTEM                      7103

#define IDS_ADD                         7200
#define IDS_EXTRACT                     7201
#define IDS_TEST                        7202
#define IDS_BUTTON_COPY                 7203
#define IDS_BUTTON_MOVE                 7204
#define IDS_BUTTON_DELETE               7205
#define IDS_BUTTON_INFO                 7206

#define IDS_SPLITTING                   7303
#define IDS_SPLIT_CONFIRM_TITLE         7304
#define IDS_SPLIT_CONFIRM_MESSAGE       7305
#define IDS_SPLIT_VOL_MUST_BE_SMALLER   7306

#define IDS_COMBINE                     7400
#define IDS_COMBINE_TO                  7401
#define IDS_COMBINING                   7402
#define IDS_COMBINE_SELECT_ONE_FILE     7403
#define IDS_COMBINE_CANT_DETECT_SPLIT_FILE 7404
#define IDS_COMBINE_CANT_FIND_MORE_THAN_ONE_PART 7405
