#ifndef DEFINES_H
#define DEFINES_H

#include <memory>
#include <vector>
#include <iostream>
#include <tuple>
#include <map>
#include <limits>

#include "config.hpp"


namespace GBDT {
  using std::shared_ptr;
  using std::string;
  using std::vector;
  using std::tuple;
  using std::map;
  using size_type = unsigned int;
  using index_type = size_type;
  const index_type NONE = std::numeric_limits<index_type>::max();
  const size_type MAX_SIZE = NONE;

#define LOG_INFO(x) \
  std::cout << x << std::endl

#define LOG_ERROR(x) \
  std::cerr << x << std::endl
}

#endif /* DEFINES_H */
