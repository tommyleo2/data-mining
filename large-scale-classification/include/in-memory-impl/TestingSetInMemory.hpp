#ifndef TESTINGSETINMEMORY_H
#define TESTINGSETINMEMORY_H

#include "../framework/TestingSet.hpp"

namespace GBDT {

  class TestingSetInMemory : public TestingSet {
  public:
    TestingSetInMemory(const string &in_file_path,
                       const string &out_file_path,
                       size_type feature_size);
    virtual double &getFeature(int id, int index) override;
    virtual double getFeature(int id, int index) const override;
    virtual const vector<double> &getCase(index_type id) override;
    virtual double &getLable(int id) override;
    virtual double getLable(int id) const override;
    virtual size_type getSetSize() const override;
    virtual size_type getFeatureSize() const override;

    virtual void dumpResult() const override;

  protected:
    vector< vector<double> > m_cases;
    vector<double> m_label;

    size_type m_feature_size;
  };

}  // GBDT

#endif /* TESTINGSETINMEMORY_H */
