/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */
#ifndef AMAZON_DSSTNE_UTILS_TENSORBOARD_METRICS_LOGGER_H
#define AMAZON_DSSTNE_UTILS_TENSORBOARD_METRICS_LOGGER_H

#include <boost/filesystem.hpp>
#include <tensorflow/core/util/events_writer.h>

#include <string>


class TensorboardMetricsLogger {
public:
    // Creates a logger that will log all events in the given directory.
    // The format of the directory is one suitable for use as Tensorboard's
    // log directory.
    //
    // The log directory must be the absolute path to an existing directory.
    explicit TensorboardMetricsLogger(boost::filesystem::path const& logdir);

    // Log a scalar metric at a given epoch. This scalar value will be part of
    // the graph displayed for the given metric in the Tensorboard dashboard.
    void scalar(int epoch, std::string const& metric, float value);

private:
    tensorflow::EventsWriter events_writer_;
};

#endif // include guard
