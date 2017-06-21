#include "framework/GBDTAlgorithm.hpp"

#include "config.hpp"


using namespace GBDT;

GBDTAlgorithm::GBDTAlgorithm(const shared_ptr<TrainingSet> &training_set,
                             const shared_ptr<TestingSet> &testing_set,
                             const shared_ptr<DecisionTree> &decision_tree) :
  m_training_set(training_set),
  m_testing_set(testing_set),
  m_decision_tree(decision_tree) {
  LOG_INFO("GBDT main module initialization done");
  }

void GBDTAlgorithm::learn() {
  if (m_training_set == nullptr ||
      m_decision_tree == nullptr) {
    LOG_ERROR("Missing modules");
    throw std::runtime_error("Some of the necessary modules are not installed");
  }

  vector<double> residual(m_training_set->getSetSize());

  //  init residual, the init predict function is set to 0
  LOG_INFO("Initializing residual...");
  for (index_type i = 0; i < residual.size(); i++) {
    residual[i] = m_decision_tree->predict(m_training_set->getCase(i)) -
      m_training_set->getLable(i);
  }

  LOG_INFO("Running learning algorithm...");
  for (index_type i = 0; i < config::ITERATION_TIMES; i++) {
    LOG_INFO("Iterating: " << i << " ...");
    // build a new tree
    m_decision_tree->buildNewTree(residual);
    LOG_INFO("Updating residual...");
    for (index_type j = 0; j < residual.size(); j++) {
      residual[j] += m_decision_tree->predictOnLastTree(j);
    }
  }
}

void GBDTAlgorithm::predict() {
  if (m_testing_set == nullptr ||
      m_decision_tree == nullptr) {
    LOG_ERROR("Missing modules");
    throw std::runtime_error("Some of the necessary modules are not installed");
  }

  LOG_INFO("Predicting...");
  for (index_type i = 0; i < m_testing_set->getSetSize(); i++) {
    m_testing_set->getLable(i) =
      m_decision_tree->predict(m_training_set->getCase(i));
  }

  LOG_INFO("Dumping result...");
  m_testing_set->dumpResult();
}
