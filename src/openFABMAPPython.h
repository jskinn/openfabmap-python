#ifndef OPEN_FABMAP_PYTHON_H
#define OPEN_FABMAP_PYTHON_H

#include <memory>
#include <vector>
#include <Python.h>
#include <boost/python.hpp>
#include <fabmap.hpp>
#include "FabMapVocabluary.h"
#include "ChowLiuTree.h"

namespace pyof2 {

class OpenFABMAPPython
{
public:
    OpenFABMAPPython(std::shared_ptr<ChowLiuTree> chowLiuTree, boost::python::dict settings = boost::python::dict());
    virtual ~OpenFABMAPPython();
    
    bool loadAndProcessImage(std::string imageFile);
    boost::python::list getAllMatches() const;
    boost::python::list getConfusionMatrix() const;
    
private:
    std::shared_ptr<FabMapVocabluary> vocabluary;
    std::shared_ptr<of2::FabMap> fabmap;
    
    std::vector<std::vector<of2::IMatch>> allMatches;
    int numPlaces;
};

} // namespace pyof2



#endif // OPEN_FABMAP_PYTHON_H
