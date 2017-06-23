#include <ctime>
#include <sys/time.h>
#include <iostream>

#define GET_CURRENT_TIME(buffer)                \
  timespec time_spec;                           \
  tm *time_d;                                   \
  clock_gettime(CLOCK_REALTIME, &time_spec);    \
  time_d = localtime(&time_spec.tv_sec);        \
  strftime(buffer, 64, "%T", time_d);

#define LOG_TIME(out)                           \
  char buffer[64];                              \
  GET_CURRENT_TIME(buffer);                     \
  out << "[" << buffer << "] "

#define LOG_INFO(x) {                           \
    LOG_TIME(std::cout) << x << std::endl;      \
  }


#define LOG_ERROR(x) {                          \
    LOG_TIME(std::cerr) << x << std::endl;      \
  }

#define LOG_DEBUG(x)

#ifdef ENABLE_DEBUG

#define LOG_DEBUG(x) {                          \
    LOG_TIME(std::cout) << x << std::endl;      \
  }

#endif // ENABLE_DEBUG
