#ifndef DECISIONTREEINMEMORY_H
#define DECISIONTREEINMEMORY_H

#include "DecisionTree.hpp"

namespace GBDT {

  class DecisionTreeInMemory : public DecisionTree {
    struct Tree {

    };
  public:
    DecisionTreeInMemory(const shared_ptr<TrainingSet> &training_set);
    virtual void buildNewTree(vector<double> &residual) override;

  private:
    vector< vector<size_t> > order_cache;
  };

}  // GBDT

#endif /* DECISIONTREEINMEMORY_H */
