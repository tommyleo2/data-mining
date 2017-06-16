#include "DecisionTreeInMemory.hpp"

#include <numeric>

using namespace GBDT;

DecisionTreeInMemory::DecisionTreeInMemory(const shared_ptr<TrainingSet> &training_set) :
  DecisionTree(training_set) {
  for (size_t i = 0; i < m_training_set->getFeatureSize(); i++) {
    vector<size_t> order(m_training_set->getSetSize());
    std::iota(order.begin(), order.end(), 0);
    m_training_set->sortSetByFeature(i, order);
    order_cache.push_back(std::move(order));
  }
}

void DecisionTreeInMemory::buildNewTree(vector<double> &residual) {

}
