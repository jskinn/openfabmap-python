/*//////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this
//  license. If you do not agree to this license, do not download, install,
//  copy or use the software.
//
// This file originates from the openFABMAP project:
// [http://code.google.com/p/openfabmap/] -or-
// [https://github.com/arrenglover/openfabmap]
//
// For published work which uses all or part of OpenFABMAP, please cite:
// [http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=6224843]
//
// Original Algorithm by Mark Cummins and Paul Newman:
// [http://ijr.sagepub.com/content/27/6/647.short]
// [http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=5613942]
// [http://ijr.sagepub.com/content/30/9/1100.abstract]
//
//                           License Agreement
//
// Copyright (C) 2012 Arren Glover [aj.glover@qut.edu.au] and
//                    Will Maddern [w.maddern@qut.edu.au], all rights reserved.
//
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//  * Redistribution's of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
//  * Redistribution's in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
//  * The name of the copyright holders may not be used to endorse or promote
//    products derived from this software without specific prior written
///   permission.
//
// This software is provided by the copyright holders and contributors "as is"
// and any express or implied warranties, including, but not limited to, the
// implied warranties of merchantability and fitness for a particular purpose
// are disclaimed. In no event shall the Intel Corporation or contributors be
// liable for any direct, indirect, incidental, special, exemplary, or
// consequential damages (including, but not limited to, procurement of
// substitute goods or services; loss of use, data, or profits; or business
// interruption) however caused and on any theory of liability, whether in
// contract, strict liability,or tort (including negligence or otherwise)
// arising in any way out of the use of this software, even if advised of the
// possibility of such damage.
//////////////////////////////////////////////////////////////////////////////*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "openFABMAPPython.h"

// ----------------- OpenFABMAPPython -----------------

pyof2::OpenFABMAPPython::OpenFABMAPPython(std::shared_ptr<pyof2::ChowLiuTree> chowLiuTree, boost::python::dict settings) :
        vocabluary(chowLiuTree->getVocabluary()),
        fabmap(),
        allMatches(),
        numPlaces(0)
{
    // Build the chow liu tree, if it hasn't been already.
    if (!chowLiuTree->isTreeBuilt())
    {
        chowLiuTree->buildChowLiuTree();
    }
    boost::python::dict openFabMapOptions;
    if (settings.has_key("openFabMapOptions")) {
        openFabMapOptions = boost::python::extract<boost::python::dict>(settings.get("openFabMapOptions"));
    }
    
    //create options flags
    std::string newPlaceMethod = "Meanfield";
    std::string bayesMethod = "Naive";
    bool simpleMotionModel = false;
    
    if (openFabMapOptions.has_key("NewPlaceMethod")) {
        newPlaceMethod = boost::python::extract<std::string>(openFabMapOptions.get("NewPlaceMethod"));
    }
    if (openFabMapOptions.has_key("BayesMethod")) {
        bayesMethod = boost::python::extract<std::string>(openFabMapOptions.get("BayesMethod"));
    }
    if (openFabMapOptions.has_key("SimpleMotion")) {
        simpleMotionModel = boost::python::extract<bool>(openFabMapOptions.get("SimpleMotion"));
    }
    
    int options = 0;
    if(newPlaceMethod == "Sampled") {
        options |= of2::FabMap::SAMPLED;
    } else {
        options |= of2::FabMap::MEAN_FIELD;
    }
    if(bayesMethod == "ChowLiu") {
        options |= of2::FabMap::CHOW_LIU;
    } else {
        options |= of2::FabMap::NAIVE_BAYES;
    }
    if(simpleMotionModel) {
        options |= of2::FabMap::MOTION_MODEL;
    }

    //create an instance of the desired type of FabMap
    std::string fabMapVersion = "FABMAP2";
    if (openFabMapOptions.has_key("FabMapVersion")) {
        fabMapVersion = boost::python::extract<std::string>(openFabMapOptions.get("FabMapVersion"));
    }
    
    // Read common settings
    double PzGe = 0.39;
    double PzGne = 0.0;
    int numSamples = 3000;
    if (openFabMapOptions.has_key("PzGe")) {
        PzGe = boost::python::extract<double>(openFabMapOptions.get("PzGe"));
    }
    if (openFabMapOptions.has_key("PzGne")) {
        PzGne = boost::python::extract<double>(openFabMapOptions.get("PzGne"));
    }
    if (openFabMapOptions.has_key("SimpleMotion")) {
        numSamples = boost::python::extract<int>(openFabMapOptions.get("NumSamples"));
    }
    
    // Create the appropriate FABMAP object
    if(fabMapVersion == "FABMAP1") {
        fabmap = std::make_shared<of2::FabMap1>(chowLiuTree->getChowLiuTree(), PzGe, PzGne, options, numSamples);
    } else if(fabMapVersion == "FABMAPLUT") {
        int precision = 6;
        if (openFabMapOptions.has_key("PzGe")) {
            precision = boost::python::extract<int>(openFabMapOptions.get("Precision"));
        }
        
        fabmap = std::make_shared<of2::FabMapLUT>(chowLiuTree->getChowLiuTree(), PzGe, PzGne, options, numSamples, precision);
    } else if(fabMapVersion == "FABMAPFBO") {
        double rejectionThreshold = 1e-8;
        double PsGd = 1e-8;
        int bisectionStart = 512;
        int bisectionIts = 9;
        
        if (openFabMapOptions.has_key("RejectionThreshold")) {
            rejectionThreshold = boost::python::extract<double>(openFabMapOptions.get("RejectionThreshold"));
        }
        if (openFabMapOptions.has_key("PsGd")) {
            PsGd = boost::python::extract<double>(openFabMapOptions.get("PsGd"));
        }
        if (openFabMapOptions.has_key("BisectionStart")) {
            bisectionStart = boost::python::extract<int>(openFabMapOptions.get("BisectionStart"));
        }
        if (openFabMapOptions.has_key("BisectionIts")) {
            bisectionIts = boost::python::extract<int>(openFabMapOptions.get("BisectionIts"));
        }
        
        fabmap = std::make_shared<of2::FabMapFBO>(chowLiuTree->getChowLiuTree(), PzGe, PzGne, options, numSamples, rejectionThreshold, PsGd, bisectionStart, bisectionIts);
    } else {    // Default to FABMAP2
        fabmap = std::make_shared<of2::FabMap2>(chowLiuTree->getChowLiuTree(), PzGe, PzGne, options);
    }

    //add the training data for use with the sampling method
    fabmap->addTraining(chowLiuTree->getTrainingData());
}

pyof2::OpenFABMAPPython::~OpenFABMAPPython()
{
    
}

bool pyof2::OpenFABMAPPython::loadAndProcessImage(std::string imageFile)
{
    cv::Mat frame = cv::imread(imageFile, CV_LOAD_IMAGE_UNCHANGED);
    if (frame.data)
    {
        cv::Mat bow = vocabluary->generateBOWImageDescs(frame);
        if (!bow.empty())
        {    
            std::vector<of2::IMatch> matches;
            fabmap->localize(bow, matches, true);
            ++numPlaces;
            allMatches.push_back(std::move(matches));
            return true;
        }        
    }
    return false;
}

boost::python::list pyof2::OpenFABMAPPython::getAllMatches() const
{
    boost::python::list matchesList;
    for (unsigned int i = 0; i < allMatches.size(); ++i)
    {
        for (std::vector<of2::IMatch>::const_iterator iter = allMatches[i].begin(); iter != allMatches[i].end(); ++iter)
        {
            if (iter->match != 0)
            {
                if(iter->imgIdx < 0) {
                    //add the new place to the confusion matrix 'diagonal'
                    matchesList.append(boost::python::make_tuple(i, (int)allMatches[i].size()-1, iter->match));

                }
                else {
                    matchesList.append(boost::python::make_tuple(i, iter->imgIdx, iter->match));
                }
            }
        }
    }
    return matchesList;
}

boost::python::list pyof2::OpenFABMAPPython::getConfusionMatrix() const
{
    cv::Mat confusion_mat(numPlaces, numPlaces, CV_64FC1);
    confusion_mat.setTo(0); // init to 0's
    for (unsigned int i = 0; i < allMatches.size(); ++i)
    {
        for (std::vector<of2::IMatch>::const_iterator iter = allMatches[i].begin(); iter != allMatches[i].end(); ++iter)
        {
            if(iter->imgIdx < 0) {
                //add the new place to the confusion matrix 'diagonal'
                confusion_mat.at<double>(i, (int)allMatches[i].size()-1) = iter->match;

            } else {
                //add the score to the confusion matrix
                confusion_mat.at<double>(i, iter->imgIdx) = iter->match;
            }
        }
    }
    
    boost::python::list matrix;
    for (int i = 0; i < numPlaces; ++i) 
    {
        boost::python::list row;
        for (int j = 0; j < numPlaces; ++j)
        {
            row.append(confusion_mat.at<double>(j, i));
        }
        matrix.append(row);
    }
    return matrix;
}
