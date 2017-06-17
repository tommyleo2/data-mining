#ifndef TREE_H
#define TREE_H

#include "defines.hpp"

namespace GBDT {

  class Tree {
  public:
    using SplitPoint = tuple<size_t, double>;

    struct Node {
      size_t m_parent = 0;
      size_t m_left = -1;
      size_t m_right = -1;
      union {
        double m_weight;
        SplitPoint m_sp;
      };
    };

  public:
    Tree();
    virtual ~Tree();
  private:
    vector<Node> m_nodes;
  };


}  // GBDT

#endif /* TREE_H */
