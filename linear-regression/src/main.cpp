#include "LinearRegression.hpp"

using namespace std;

int main(void) {
  LinearRegression lr;
  // lr.learn(std::numeric_limits<unsigned long long>::max());
  lr.learn(100);
  lr.predict();
  return 0;
}
