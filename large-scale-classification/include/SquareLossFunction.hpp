#ifndef SQUARELOSSFUNCTION_H
#define SQUARELOSSFUNCTION_H

#include "LossFunction.hpp"

namespace GBDT {
  class SquareLossFunction : public LossFunction {
  public:
    SquareLossFunction(shared_ptr<TrainingSet> &training_set) :
      LossFunction(training_set) { }
  protected:
    virtual double get_1_DF(double point, size_t index) override;
    virtual double get_2_DF(double point, size_t index) override;
  };
}

#endif /* SQUARELOSSFUNCTION_H */
