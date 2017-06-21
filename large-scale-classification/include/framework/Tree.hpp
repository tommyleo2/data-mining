#ifndef TREE_H
#define TREE_H

#include "defines.hpp"
#include "TrainingSet.hpp"

#include <numeric>
#include <iostream>

namespace GBDT {

  class Tree {
  public:
    using SplitPoint = tuple<index_type, double>;
    //                       feature,    value

    friend std::ostream &operator<<(std::ostream &out, const Tree &tree) {
      out << tree.m_nodes.size() << std::endl;
      for (auto &&node: tree.m_nodes) {
        out << node << std::endl;
      }
      return out;
    }

    friend std::istream &operator>>(std::istream &in, Tree &tree) {
      size_type node_size;
      in >> node_size;

      tree.m_nodes.clear();
      tree.m_nodes.resize(node_size, Node(tree.m_training_set));

      for (auto &&node : tree.m_nodes) {
        in >> node;
      }
      return in;
    }

    struct Node {

      friend std::ostream &operator<<(std::ostream &out, const Node &node) {
        out << node.m_parent << " " << node.m_left << " " << node.m_right << " ";
        if (node.is_leaf()) {
          out << node.m_weight;
        } else {
          out << std::get<0>(node.m_sp) << " " << std::get<1>(node.m_sp);
        }
        return out;
      }

      friend std::istream &operator>>(std::istream &in, Node &node) {
        in >> node.m_parent >> node.m_left >> node.m_right;
        if (node.is_leaf()) {
          in >> node.m_weight;
        } else {
          in >> std::get<0>(node.m_sp) >> std::get<1>(node.m_sp);
        }
        return in;
      }

      explicit Node(const shared_ptr<TrainingSet> &training_set) :
        m_parent(NONE), m_left(NONE), m_right(NONE), m_weight(0),
        m_training_set(training_set) { }
      Node &operator=(const Node &node) {
        m_parent = node.m_parent;
        m_left = node.m_left;
        m_right = node.m_right;
        if (is_leaf()) {
          m_weight = node.m_weight;
        } else {
          m_sp = node.m_sp;
        }
        m_training_set = node.m_training_set;
        return *this;
      }

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
    Tree() { }
    Tree(const shared_ptr<TrainingSet> &training_set,
         vector<double> &&first_derived,
         vector<double> &&second_derived) :
      m_training_set(training_set),
      m_first_derived(std::forward< vector<double> >(first_derived)),
      m_second_derived(std::forward< vector<double> >(second_derived)) {
      //  init root node
      Node tmp(m_training_set);
      m_nodes.push_back(std::move(tmp));
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
