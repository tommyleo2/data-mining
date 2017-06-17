#include "DecisionTreeInMemory.hpp"

#include <numeric>

using namespace GBDT;

DecisionTreeInMemory::DecisionTreeInMemory(const shared_ptr<TrainingSet> &training_set,
                                           const shared_ptr<LossFunction> &loss_function) :
  DecisionTree(training_set, loss_function) {
  //  cache all features order
  for (size_type i = 0; i < m_training_set->getFeatureSize(); i++) {
    vector<size_type> order(m_training_set->getSetSize());
    std::iota(order.begin(), order.end(), 0);
    m_training_set->sortSetByFeature(i, order);
    order_cache.push_back(std::move(order));
  }
}

void DecisionTreeInMemory::buildNewTree(vector<double> &residual) {
  vector<double> first_derived(m_training_set->getSetSize());
  vector<double> second_derived(m_training_set->getSetSize());
  //  calculate first and second derivative value
  m_loss_function->apply_1_DF(residual, first_derived);
  m_loss_function->apply_2_DF(residual, second_derived);

  Tree tree(m_training_set, std::move(first_derived), std::move(second_derived));
  vector<index_type> current_layer {0};
  //  split a tree nodes
  for (size_type i = 0; i < config::MAX_TREE_DEPTH; i++) {
    vector<index_type> next_layer;
    for (auto index : current_layer) {
      auto split_result = tree.split(index);
      next_layer.push_back(std::get<0>(split_result));
      next_layer.push_back(std::get<1>(split_result));
    }
    current_layer = std::move(next_layer);
  }
  tree.releaseResources();
  m_trees.push_back(std::move(tree));
}


double DecisionTreeInMemory::predict(const TrainingSet::TrainingSetRow_t &test_case) {
  return 0;
}
