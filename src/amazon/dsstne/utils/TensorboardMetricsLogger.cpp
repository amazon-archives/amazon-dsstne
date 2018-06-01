/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#include <amazon/dsstne/utils/TensorboardMetricsLogger.h>

#include <boost/filesystem.hpp>
#include <tensorflow/core/util/events_writer.h>

#include <cassert>
#include <string>


TensorboardMetricsLogger::TensorboardMetricsLogger(boost::filesystem::path const& logdir)
  : events_writer_((logdir / "events").native())
{
    assert(logdir.is_absolute());
    assert(boost::filesystem::is_directory(logdir));
}

void TensorboardMetricsLogger::scalar(int epoch, std::string const& metric, float value) {
    tensorflow::Event event;
    event.set_step(epoch);
    tensorflow::Summary::Value* summary_value = event.mutable_summary()->add_value();
    summary_value->set_tag(metric);
    summary_value->set_simple_value(value);
    events_writer_.WriteEvent(event);
}
