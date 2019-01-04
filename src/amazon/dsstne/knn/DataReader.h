/*
 *  Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License").
 *  You may not use this file except in compliance with the License.
 *  A copy of the License is located at
 *
 *  http://aws.amazon.com/apache2.0/
 *
 *  or in the "license" file accompanying this file.
 *  This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 *  either express or implied.
 *
 *  See the License for the specific language governing permissions and limitations under the License.
 *
 */

#ifndef LIBKNN_DATAREADER_H_
#define LIBKNN_DATAREADER_H_

#include <fstream>

class DataReader
{

  public:

    /**
     * Reads and parses the next row into the key and vector variables.
     */
    virtual bool readRow(std::string *key, float *vector) = 0;

    uint32_t getRows() const;

    int getColumns() const;

    virtual ~DataReader()
    {
    }

  protected:
    uint32_t rows;
    int columns;
};

class TextFileDataReader: public DataReader
{

  public:
    TextFileDataReader(const std::string &fileName, char keyValueDelimiter = '\t', char vectorDelimiter = ' ');

    bool readRow(std::string *key, float *vector);

    static void findDataDimensions(const std::string &fileName, uint32_t &rows, int &columns, char keyValueDelimiter =
        '\t', char vectorDelimiter = ' ');

    ~TextFileDataReader();

  private:
    std::string fileName;
    std::fstream fileStream;
    char keyValueDelimiter;
    char vectorDelimiter;
};

#endif /* LIBKNN_DATAREADER_H_ */
