#include "SquareLossFunction.hpp"

using namespace GBDT;

double SquareLossFunction::get_1_DF(double point, size_t index) {
  return 2 * (point - m_training_set->getLable(index));
}

double SquareLossFunction::get_2_DF(double, size_t) {
  return 2;
}
