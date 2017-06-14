#ifndef TRAININGSET_H
#define TRAININGSET_H

#include <cstddef>

namespace GBDT {

  class TrainingSet {
  public:
    virtual double &getFeature(int id, int index) = 0;
    virtual double getFeature(int id, int index) const = 0;
    virtual double &getLable(int id) = 0;
    virtual double getLable(int id) const = 0;
    virtual size_t getSetSize() const = 0;
    virtual size_t getFeatureSize() const = 0;
  };

}

#endif /* TRAININGSET_H */
