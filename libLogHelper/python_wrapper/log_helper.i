 /*log_helper.i */
%module log_helper
// Make log_helper.cxx include this header:
%{
#define SWIG_FILE_WITH_INIT
#include "log_helper.h"
%}

// Make SWIG look into this header:
%include "log_helper.h"
