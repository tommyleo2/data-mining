#include "in-memory-impl/DecisionTreeInMemory.hpp"
#include "config.hpp"

#include <numeric>
#include <fstream>

using namespace GBDT;

DecisionTreeInMemory::DecisionTreeInMemory(const shared_ptr<TrainingSet> &training_set,
                                           const shared_ptr<LossFunction> &loss_function) :
  DecisionTree(training_set, loss_function) {
  std::fstream model_file(config::MODEL_PATH, std::fstream::in|std::fstream::binary);
  if (!model_file) {
    LOG_INFO("No existing model file");
    return;
  }
  m_trees.clear();
  size_type size;
  model_file >> size;
  m_trees.reserve(size);
  while (size--) {
    Tree tree;
    model_file >> tree;
    m_trees.push_back(std::move(tree));
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
  for (size_type i = 0; i < config::MAX_TREE_DEPTH; i++) {
    LOG_INFO("Building layer: " << i);
    vector<index_type> next_layer;
    for (auto index : current_layer) {
      LOG_DEBUG("Spliting: " << index);
      //  split a tree node
      auto split_result = tree.split(index);
      if (std::get<0>(split_result) == NONE) {
        continue;
      }
      next_layer.push_back(std::get<0>(split_result));
      next_layer.push_back(std::get<1>(split_result));
    }
    current_layer = std::move(next_layer);
  }
  tree.finishTraining();
  m_trees.push_back(std::move(tree));
}


double DecisionTreeInMemory::predict(const TrainingSet::TrainingSetRow_t &test_case) {
  double result = 0;
  for (auto &&tree : m_trees) {
    const Tree::Node *current_node = &tree[0];
    while(!current_node->is_leaf()) {
      index_type feature = std::get<0>(current_node->m_sp);
      double sp_value = std::get<0>(current_node->m_sp);
      if (test_case[feature] < sp_value) {
        current_node = &tree[current_node->m_left];
      } else {
        current_node = &tree[current_node->m_right];
      }
    }
    result += config::ETA * current_node->m_weight;
  }
  return result;
}

double DecisionTreeInMemory::predictOnLastTree(index_type id) {
  const Tree *last_tree = &m_trees.back();
  const Tree::Node *current_node = &last_tree->operator[](0);
  while(!current_node->is_leaf()) {
    index_type feature = std::get<0>(current_node->m_sp);
    double sp_value = std::get<0>(current_node->m_sp);
    if (m_training_set->getFeature(id, feature) < sp_value) {
      current_node = &last_tree->operator[](current_node->m_left);
    } else {
      current_node = &last_tree->operator[](current_node->m_right);
    }
  }
  return config::ETA * current_node->m_weight;
}
