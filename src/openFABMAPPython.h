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
    int getLastMatch() const;
    boost::python::list getAllLoopClosures() const;
    
private:
    std::shared_ptr<FabMapVocabluary> vocabluary;
    std::shared_ptr<of2::FabMap> fabmap;
    
    int imageIndex;
    int lastMatch;
    boost::python::list loopClosures;
};

} // namespace pyof2



#endif // OPEN_FABMAP_PYTHON_H
