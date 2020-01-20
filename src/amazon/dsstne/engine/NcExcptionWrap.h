#pragma once
#include <ncException.h>
#include <string>
#define NC_EXCEPTION(errorStr, msg, filename, line) NcException(std::string(msg).c_str(), filename, line)
