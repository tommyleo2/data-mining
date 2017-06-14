#ifndef TRAININGSETINMEMORY_H
#define TRAININGSETINMEMORY_H

#include "TrainingSet.hpp"
#include "config.hpp"
#include <vector>
#include <fstream>
#include <sstream>

namespace GBDT {
  using std::string;
  using std::vector;

  class TrainingSetInMemory : public TrainingSet {
  public:
    using TrainingSetRow_t = vector<double>;
    using TrainingSet_t = vector<TrainingSetRow_t>;

    TrainingSetInMemory(size_t feature_size);
    virtual double &getFeature(int id, int index) override;
    virtual double getFeature(int id, int index) const override;
    virtual double &getLable(int id) override;
    virtual double getLable(int id) const override;
    virtual size_t getSetSize() const override;
    virtual size_t getFeatureSize() const override;
  private:
    TrainingSet_t m_data;
    vector<double> m_label;

    size_t m_feature_size;
  };

}
#endif /* TRAININGSETINMEMORY_H */
