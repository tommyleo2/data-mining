#include "framework/Tree.hpp"
#include "config.hpp"

#include <algorithm>
#include <cmath>

using namespace GBDT;
using std::pow;

tuple<index_type, index_type> Tree::split(index_type index) {
  if (index >= m_nodes.size()) {
    throw std::runtime_error("No such node");
  }
  if (!m_nodes[index].is_leaf()) {
    throw std::runtime_error("Node has been splited already");
  }

  vector<SplitPoint> best_splits(m_training_set->getFeatureSize());
  Node &current_node = m_nodes[index];

  if (index == 0) {  //  root node
    for (index_type i = 0; i < m_training_set->getFeatureSize(); i++) {
      best_splits[i] = findFeatureSplit(i, m_training_set->sortSetByFeature(i));
    }
  } else {  // other nodes
    for (index_type i = 0; i < m_training_set->getFeatureSize(); i++) {
      best_splits[i] = findFeatureSplit(i, m_training_set->sortSetByFeature(i, current_node.m_attached_cases));
    }
  }
  auto best_split_it = std::max_element(best_splits.begin(),
                                        best_splits.end(),
                                        [](const tuple<index_type, double> &first,
                                           const tuple<index_type, double> &second) {
                                          return std::get<1>(first) < std::get<1>(second);
                                        });
  if (std::get<1>(*best_split_it) <= 0) {
    return std::make_tuple(NONE, NONE);
  }

  index_type best_feature = best_split_it - best_splits.begin();
  //index_type case_index = std::get<0>(*best_split_it);
  double split_value = std::get<1>(*best_split_it);

  current_node.m_sp = std::make_tuple(best_feature, split_value);

  //  build children nodes
  Node left(m_training_set), right(m_training_set);
  left.m_parent = index;
  right.m_parent = index;
  auto split_result = current_node.filter(*best_split_it);
  left.m_attached_cases = std::move(std::get<0>(split_result));
  right.m_attached_cases = std::move(std::get<1>(split_result));

  m_nodes.push_back(std::move(left));
  m_nodes.push_back(std::move(right));

  return std::make_tuple(m_nodes.size() - 2, m_nodes.size() - 1);
}


tuple<index_type, double> Tree::findFeatureSplit(index_type feature_index, const vector<index_type> &cases) {
  // double last_feature = std::numeric_limits<double>::min();
  double max_gain = std::numeric_limits<double>::min();
  index_type max_gain_index = NONE;

  double GL = 0, GR = 0;
  double HL = 0, HR = 0;

  for (auto index : cases) {
    GR += m_first_derived[index];
    HR += m_second_derived[index];
  }

  for (auto it = cases.begin(); it != cases.end(); it++) {
    GR -= m_first_derived[*it];
    GL += m_first_derived[*it];
    HR -= m_second_derived[*it];
    HL += m_second_derived[*it];

    if (it + 1 == cases.end()) {
      break;
    }

    if (m_training_set->getFeature(*it, feature_index) ==
        m_training_set->getFeature(*(it + 1), feature_index)) {
      continue;
    }
    double tmp_gain = gain(GL, GR, HL, HR);
    if (tmp_gain > max_gain) {
      max_gain = tmp_gain;
      max_gain_index = *it;
    }
  }

  return std::make_tuple(max_gain_index, max_gain);
}

double Tree::gain(double GL, double GR, double HL, double HR) {
  return (
          pow(GL, 2) / (HL + config::LAMBDA) +
          pow(GR, 2) / (HR + config::LAMBDA) -
          pow(GL + GR, 2) / (HL + HR + config::LAMBDA)
          ) / 2  - config::GAMMA;
}

void Tree::calculateWeight() {
  for (auto &&node : m_nodes) {
    if (!node.is_leaf()) {
      continue;
    }
    double G = 0, H = 0;
    for (auto index : node.m_attached_cases) {
      G += m_first_derived[index];
      H += m_second_derived[index];
    }
    node.m_weight = - (G / (H + config::LAMBDA));
  }
}

void Tree::releaseResources() {
   m_first_derived.clear();
   m_first_derived.resize(0);
   m_second_derived.clear();
   m_second_derived.resize(0);

   for (auto &&node : m_nodes) {
     node.m_attached_cases.clear();
     node.m_attached_cases.resize(0);
   }
 }
