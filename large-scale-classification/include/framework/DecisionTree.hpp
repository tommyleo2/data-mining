#ifndef DECISIONTREE_H
#define DECISIONTREE_H

#include "defines.hpp"
#include "TrainingSet.hpp"

namespace GBDT {

  class DecisionTree {
  public:
    DecisionTree(const shared_ptr<TrainingSet> &training_set) :
      m_training_set(training_set) { }
    virtual void buildNewTree(vector<double> &residual) = 0;
    virtual double predict(const vector<double> &test_case) = 0;
  protected:
    shared_ptr<TrainingSet> m_training_set;
  };

}  // GBDT

#endif /* DECISIONTREE_H */
