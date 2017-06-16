#ifndef DEFINES_H
#define DEFINES_H

#include <memory>
#include <vector>
#include <iostream>

#include "config.hpp"


namespace GBDT {
  using std::shared_ptr;
  using std::string;
  using std::vector;

#define LOG_INFO(x) \
  std::cout << x << std::endl

#define LOG_ERROR(x) \
  std::cerr << x << std::endl
}

#endif /* DEFINES_H */
