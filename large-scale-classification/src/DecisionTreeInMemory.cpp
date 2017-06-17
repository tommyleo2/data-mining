#include "DecisionTreeInMemory.hpp"

#include <numeric>

using namespace GBDT;

DecisionTreeInMemory::DecisionTreeInMemory(const shared_ptr<TrainingSet> &training_set) :
  DecisionTree(training_set) {
  //  cache all features order
  for (size_t i = 0; i < m_training_set->getFeatureSize(); i++) {
    vector<size_t> order(m_training_set->getSetSize());
    std::iota(order.begin(), order.end(), 0);
    m_training_set->sortSetByFeature(i, order);
    order_cache.push_back(std::move(order));
  }
}

void DecisionTreeInMemory::buildNewTree(vector<double> &residual) {
  // vector<size_t>
  // for (size_t layer = 0; layer < config::MAX_TREE_DEPTH; layer++) {
  //   SplitPoint sp = findBestSplitPoint( const vector<size_t> &cases );
  // }
}


double DecisionTreeInMemory::predict(const vector<double> &test_case) {
  return 0;
}

DecisionTreeInMemory::SplitPoint DecisionTreeInMemory::findBestSplitPoint(const vector<size_t> &cases) {
  return {};
}
