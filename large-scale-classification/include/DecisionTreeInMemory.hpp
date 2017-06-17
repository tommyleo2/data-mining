#ifndef DECISIONTREEINMEMORY_H
#define DECISIONTREEINMEMORY_H

#include "framework/DecisionTree.hpp"

namespace GBDT {

  class DecisionTreeInMemory : public DecisionTree {
  public:
    using SplitPoint = tuple<size_t, size_t>;

  public:
    DecisionTreeInMemory(const shared_ptr<TrainingSet> &training_set);
    virtual void buildNewTree(vector<double> &residual) override;
    virtual double predict(const vector<double> &test_case) override;

  protected:
    virtual SplitPoint findBestSplitPoint(const vector<size_t> &cases);

    vector< vector<size_t> > order_cache;
  };

}  // GBDT

#endif /* DECISIONTREEINMEMORY_H */
