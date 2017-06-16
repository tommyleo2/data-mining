#ifndef TESTINGSET_H
#define TESTINGSET_H

#include "defines.hpp"

namespace GBDT {

  class TestingSet {
  public:
    TestingSet(const string &in_file_path, const string &out_file_path);
    virtual double &getFeature(int id, int index) = 0;
    virtual double getFeature(int id, int index) const = 0;
    virtual double &getLable(int id) = 0;
    virtual double getLable(int id) const = 0;
    virtual size_t getSetSize() const = 0;
    virtual size_t getFeatureSize() const = 0;

    virtual void dumpResult() const = 0;

  protected:
    string m_in_file_path;
    string m_out_file_path;
  };

}
#endif /* TESTINGSET_H */
