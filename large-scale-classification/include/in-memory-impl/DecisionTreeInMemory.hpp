#ifndef DECISIONTREEINMEMORY_H
#define DECISIONTREEINMEMORY_H

#include "../framework/DecisionTree.hpp"
#include "../framework/Tree.hpp"

#include "../ext/SimpleThreadPool.hpp"

namespace GBDT {

  class DecisionTreeInMemory : public DecisionTree {
  public:
    using SplitPoint = tuple<size_type, size_type>;

  public:
    DecisionTreeInMemory(const shared_ptr<TrainingSet> &training_set,
                         const shared_ptr<LossFunction> &loss_function,
                         const string model_file_path);
    virtual ~DecisionTreeInMemory();
    virtual void buildNewTree(vector<double> &residual) override;
    virtual double predict(const TrainingSet::TrainingSetRow_t &test_case) override;
    virtual double predictOnLastTree(index_type id) override;
    virtual void dumpTrees() override;

  protected:
    vector<Tree> m_trees;

    SimpleThreadPool::ThreadPool m_thread_pool;
  };

}  // GBDT

#endif /* DECISIONTREEINMEMORY_H */
