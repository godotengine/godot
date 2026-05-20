// XzCrc64Init.cpp

#include "StdAfx.h"

#include "../../C/XzCrc64.h"

static struct CCrc64Gen { CCrc64Gen() { Crc64GenerateTable(); } } g_Crc64TableInit;
