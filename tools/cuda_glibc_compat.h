#pragma once

#if defined(__cplusplus) && defined(__linux__)
#include <features.h>
#if defined(__GLIBC__) && defined(__THROW)
#undef __THROW
#define __THROW
#endif
#endif
