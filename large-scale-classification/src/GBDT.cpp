#include "GBDT.hpp"

using namespace GBDT;

GBDTAlgorithm::GBDTAlgorithm(const shared_ptr<TrainingSet> &training_set,
                             const shared_ptr<TestingSet> &testing_set,
                             const shared_ptr<LossFunction> &loss_function,
                             const shared_ptr<DecisionTree> &decision_tree) :
  m_training_set(training_set),
  m_testing_set(testing_set),
  m_loss_function(loss_function),
  m_decision_tree(decision_tree) {
  LOG_INFO("GBDT main module initialization done");
  }

void GBDTAlgorithm::learn() {
  if (m_training_set == nullptr ||
      m_loss_function == nullptr ||
      m_decision_tree == nullptr) {
    LOG_ERROR("Missing modules");
    throw std::runtime_error("Some of the necessary modules are not installed");
  }

  vector<double> residual(m_training_set->getSetSize());
  vector<double> first_derived(m_training_set->getSetSize());
  vector<double> second_derived(m_training_set->getSetSize());

  //  init residual, the init predict function is set to 0
  LOG_INFO("Initializing residual...");
  for (size_t i = 0; i < residual.size(); i++) {
    residual[i] = m_training_set->getLable(i) - 0;
  }

  LOG_INFO("Runing learning algorithm...");
  for (int i = 0; i < config::ITERATION_TIMES; i++) {
    LOG_INFO("Iterating: " << i << " ...");
    //  calculate first and second derivative value
    m_loss_function->apply_1_DF(residual, first_derived);
    m_loss_function->apply_2_DF(residual, second_derived);

    // build a new tree
    m_decision_tree->buildNewTree(residual);
  }
}

void GBDTAlgorithm::predict() {
  if (m_testing_set == nullptr ||
      m_decision_tree == nullptr) {
    LOG_ERROR("Missing modules");
    throw std::runtime_error("Some of the necessary modules are not installed");
  }

  LOG_INFO("Predicting...");
  for (size_t i = 0; i < m_testing_set->getSetSize(); i++) {
    m_testing_set->getLable(i) =
      m_decision_tree->predict(m_training_set->getCase(i));
  }

  LOG_INFO("Dumping result...");
  m_testing_set->dumpResult();
}
