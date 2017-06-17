#ifndef TREE_H
#define TREE_H

#include "defines.hpp"
#include "TrainingSet.hpp"

#include <numeric>

namespace GBDT {

  class Tree {
  public:
    using SplitPoint = tuple<index_type, double>;

    struct Node {
      Node() : m_parent(-1), m_left(-1), m_right(-1) { }
      index_type m_parent;
      index_type m_left;
      index_type m_right;
      union {
        double m_weight;
        SplitPoint m_sp;
      };
      vector<index_type> m_attached_cases;
    };

  public:
    Tree(const shared_ptr<TrainingSet> &training_set,
         vector<double> &&first_derived,
         vector<double> &&second_derived) :
      m_training_set(training_set),
      m_first_derived(first_derived),
      m_second_derived(second_derived) {
      //  init root node
      Node tmp;
      vector<index_type> all_cases(m_training_set->getSetSize());
      std::iota(all_cases.begin(), all_cases.end(), 0);
      tmp.m_attached_cases = std::move(all_cases);
      m_nodes.push_back(std::move(tmp));
    }

    tuple<index_type, index_type> split(index_type index) {

    }

    void releaseResources() {
      m_first_derived.clear();
      m_first_derived.resize(0);
      m_second_derived.clear();
      m_second_derived.resize(0);

      for (auto &&node : m_nodes) {
        node.m_attached_cases.clear();
        node.m_attached_cases.resize(0);
      }
    }

    index_type getDepth(index_type index) const {
      size_type depth = 0;
      while (m_nodes[index].m_parent != -1) {
        depth++;
        index = m_nodes[index].m_parent;
      }
      return depth;
    }

  private:
    vector<Node> m_nodes;

    shared_ptr<TrainingSet> m_training_set;
    vector<double> m_first_derived;
    vector<double> m_second_derived;
  };


}  // GBDT

#endif /* TREE_H */
