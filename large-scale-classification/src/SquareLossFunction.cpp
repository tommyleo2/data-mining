#include "SquareLossFunction.hpp"

using namespace GBDT;

double SquareLossFunction::get_1_DF(double point, index_type index) {
  return 2 * (point - m_training_set->getLable(index));
}

double SquareLossFunction::get_2_DF(double, index_type) {
  return 2;
}
