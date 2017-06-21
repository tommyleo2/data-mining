#include "defines.hpp"

#include "TrainingSet.hpp"
#include "DecisionTree.hpp"
#include "TestingSet.hpp"

namespace GBDT {
  class GBDTAlgorithm {
  public:
    GBDTAlgorithm(const shared_ptr<TrainingSet> &training_set,
                  const shared_ptr<TestingSet> &testing_set,
                  const shared_ptr<DecisionTree> &decision_tree);
    void learn();
    void predict();
  protected:
    shared_ptr<TrainingSet> m_training_set;
    shared_ptr<TestingSet> m_testing_set;
    shared_ptr<DecisionTree> m_decision_tree;

  private:
    vector<double> m_residual;
  };
}
