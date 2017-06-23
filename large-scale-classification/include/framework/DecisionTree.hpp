#ifndef DECISIONTREE_H
#define DECISIONTREE_H

#include "defines.hpp"
#include "TrainingSet.hpp"
#include "LossFunction.hpp"

namespace GBDT {

  class DecisionTree {
  public:
    DecisionTree(const shared_ptr<TrainingSet> &training_set,
                 const shared_ptr<LossFunction> &loss_function,
                 const string model_file_path) :
      m_training_set(training_set), m_loss_function(loss_function),
      m_model_file_path(model_file_path) { }
    virtual void buildNewTree(vector<double> &residual) = 0;
    virtual double predict(const TrainingSet::TrainingSetRow_t &test_case) = 0;
    virtual double predictOnLastTree(index_type id) = 0;
    virtual void dumpTrees() = 0;
  protected:
    shared_ptr<TrainingSet> m_training_set;
    shared_ptr<LossFunction> m_loss_function;

    string m_model_file_path;
  };

}  // GBDT

#endif /* DECISIONTREE_H */
