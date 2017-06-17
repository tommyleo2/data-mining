#ifndef LOSSFUNCTION_H
#define LOSSFUNCTION_H

#include "TrainingSet.hpp"


namespace GBDT {

  class LossFunction {
  public:
    LossFunction(shared_ptr<TrainingSet> &training_set) :
      m_training_set(training_set) { }

    virtual void apply_1_DF(const vector<double> &points,
                            vector<double> &result) final {
      for (size_type i = 0; i < points.size(); i++) {
        result.at(i) = get_1_DF(points[i], i);
      }
    }

    virtual void apply_2_DF(const vector<double> &points,
                            vector<double> &result) final {
      for (size_type i = 0; i < points.size(); i++) {
        result.at(i) = get_2_DF(points[i], i);
      }
    }

  protected:
    virtual double get_1_DF(double point, index_type index) = 0;
    virtual double get_2_DF(double point, index_type index) = 0;

    shared_ptr<TrainingSet> m_training_set;
  };

}

#endif /* LOSSFUNCTION_H */
