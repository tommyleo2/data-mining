#ifndef DECISIONTREE_H
#define DECISIONTREE_H

#include "defines.hpp"
#include "TrainingSet.hpp"
#include "LossFunction.hpp"

namespace GBDT {

  class DecisionTree {
  public:
    DecisionTree(const shared_ptr<TrainingSet> &training_set,
                 const shared_ptr<LossFunction> &loss_function) :
      m_training_set(training_set), m_loss_function(loss_function) { }
    virtual void buildNewTree(vector<double> &residual) = 0;
    virtual double predict(const TrainingSet::TrainingSetRow_t &test_case) = 0;
  protected:
    shared_ptr<TrainingSet> m_training_set;
    shared_ptr<LossFunction> m_loss_function;
  };

}  // GBDT

#endif /* DECISIONTREE_H */
