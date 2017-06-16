#ifndef TRAININGSETINMEMORY_H
#define TRAININGSETINMEMORY_H

#include "TrainingSet.hpp"
#include "config.hpp"
#include <vector>
#include <fstream>
#include <sstream>

namespace GBDT {

  class TrainingSetInMemory : public TrainingSet {
  public:
    using TrainingSet_t = vector<TrainingSetRow_t>;

    TrainingSetInMemory(const string &file_path,size_t feature_size);
    virtual double &getFeature(int id, int index) override;
    virtual double getFeature(int id, int index) const override;

    virtual const TrainingSetRow_t &getCase(int id) override;

    virtual double &getLable(int id) override;
    virtual double getLable(int id) const override;
    virtual size_t getSetSize() const override;
    virtual size_t getFeatureSize() const override;

    virtual void sortSetByFeature(int index, vector<size_t> &ids) override;
  protected:
    TrainingSet_t m_data;
    TrainingSetCol_t m_label;

    size_t m_feature_size;
  };

}

#endif /* TRAININGSETINMEMORY_H */
