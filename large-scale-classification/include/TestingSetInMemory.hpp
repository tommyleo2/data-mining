#ifndef TESTINGSETINMEMORY_H
#define TESTINGSETINMEMORY_H

#include "framework/TestingSet.hpp"

namespace GBDT {

  class TestingSetInMemory : public TestingSet {
  public:
    TestingSetInMemory(const string &in_file_path,
                       const string &out_file_path,
                       size_t feature_size);
    virtual double &getFeature(int id, int index) override;
    virtual double getFeature(int id, int index) const override;
    virtual double &getLable(int id) override;
    virtual double getLable(int id) const override;
    virtual size_t getSetSize() const override;
    virtual size_t getFeatureSize() const override;

    virtual void dumpResult() const override;

  protected:
    vector< vector<double> > m_cases;
    vector<double> m_label;

    size_t m_feature_size;
  };

}  // GBDT

#endif /* TESTINGSETINMEMORY_H */
