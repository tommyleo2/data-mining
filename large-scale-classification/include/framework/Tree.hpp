#ifndef TREE_H
#define TREE_H

#include "defines.hpp"
#include "TrainingSet.hpp"

#include <numeric>

namespace GBDT {

  class Tree {
  public:
    using SplitPoint = tuple<index_type, double>;
    //                       feature,    value

    struct Node {
      Node(const shared_ptr<TrainingSet> &training_set) :
        m_parent(NONE), m_left(NONE), m_right(NONE),
        m_training_set(training_set) { }

      tuple< vector<index_type>, vector<index_type> > filter(SplitPoint sp) {
        vector<index_type> left, right;
        index_type feature = std::get<0>(sp);
        double value = std::get<1>(sp);
        for (auto id : m_attached_cases) {
          if (m_training_set->getFeature(id, feature) < value) {
            left.push_back(id);
          } else {
            right.push_back(id);
          }
        }
        return std::make_tuple(std::move(left), std::move(right));
      }

      bool is_leaf() const {
        return m_left == NONE && m_right == NONE;
      }

      index_type m_parent;
      index_type m_left;
      index_type m_right;
      union {
        double m_weight;
        SplitPoint m_sp;
      };
      vector<index_type> m_attached_cases;

      shared_ptr<TrainingSet> m_training_set;
    };

  public:
    Tree(const shared_ptr<TrainingSet> &training_set,
         vector<double> &&first_derived,
         vector<double> &&second_derived) :
      m_training_set(training_set),
      m_first_derived(first_derived),
      m_second_derived(second_derived) {
      //  init root node
      Node tmp(m_training_set);
      // vector<index_type> all_cases(m_training_set->getSetSize());
      // std::iota(all_cases.begin(), all_cases.end(), 0);
      // tmp.m_attached_cases = std::move(all_cases);
      // m_nodes.push_back(std::move(tmp));
    }

    tuple<index_type, index_type> split(index_type node_index);
    tuple<index_type, double> findFeatureSplit(index_type feature_index, const vector<index_type> &cases);
    double gain(double GL, double GR, double HL, double HR);

    void finishTraining() {
      calculateWeight();
      releaseResources();
    }

    void calculateWeight();
    void releaseResources();

    index_type getDepth(index_type index) const {
      size_type depth = 0;
      while (m_nodes[index].m_parent != NONE) {
        depth++;
        index = m_nodes[index].m_parent;
      }
      return depth;
    }

    const Node &operator[](index_type index) const {
      return m_nodes[index];
    }

  private:
    vector<Node> m_nodes;

    shared_ptr<TrainingSet> m_training_set;
    vector<double> m_first_derived;
    vector<double> m_second_derived;
  };


}  // GBDT

#endif /* TREE_H */
