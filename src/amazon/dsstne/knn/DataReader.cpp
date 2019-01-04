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

#include <exception>
#include <fstream>
#include <string>
#include <stdexcept>
#include <sstream>

#include "DataReader.h"

namespace
{
static const int success = 0;
static const int failure = 1;

int splitKeyVector(const std::string &line, std::string &key, std::string &vector, char keyValueDelimiter)
{
  int keyValDelimIndex = line.find_first_of(keyValueDelimiter);

  if (keyValDelimIndex == std::string::npos)
  {
    return failure;
  }

  key = line.substr(0, keyValDelimIndex);
  vector = line.substr(keyValDelimIndex + 1, line.size());
  return success;
}
}

uint32_t DataReader::getRows() const
{
  return rows;
}

int DataReader::getColumns() const
{
  return columns;
}

TextFileDataReader::TextFileDataReader(const std::string &fileName, char keyValueDelimiter, char vectorDelimiter) :
    fileName(fileName),
    fileStream(fileName, std::ios_base::in),
    keyValueDelimiter(keyValueDelimiter),
    vectorDelimiter(vectorDelimiter)
{

  findDataDimensions(fileName, rows, columns, keyValueDelimiter, vectorDelimiter);
}

void TextFileDataReader::findDataDimensions(const std::string &fileName, uint32_t &rows, int &columns,
    char keyValueDelimiter, char vectorDelimiter)
{
  std::fstream fs(fileName, std::ios_base::in);

  rows = 0;
  columns = 0;

  std::string line;
  while (std::getline(fs, line))
  {
    if (line.length() == 0)
    {
      // skip empty lines
      continue;
    }

    ++rows;

    std::string key;
    std::string vectorStr;

    if (splitKeyVector(line, key, vectorStr, keyValueDelimiter))
    {
      std::stringstream msg;
      msg << "In file: " << fileName << "#" << rows << ". Malformed line. key-value delimiter [" << keyValueDelimiter
          << "] not found in: " << line;
      throw std::invalid_argument(msg.str());
    }

    std::stringstream vectorStrStream(vectorStr);
    std::string elementStr;
    int columnsInRow = 0;
    while (std::getline(vectorStrStream, elementStr, vectorDelimiter))
    {
      ++columnsInRow;
    }

    // check all rows have same nColumns
    if (columns == 0)
    {
      columns = columnsInRow;
    } else
    {
      if (columns != columnsInRow)
      {
        std::stringstream msg;
        msg << "In file: " << fileName << "#" << rows << ". Inconsistent num columns detected. Expected : " << columns
            << " Actual: " << columnsInRow;
        throw std::invalid_argument(msg.str());
      }
    }
  }

  fs.close();
}

bool TextFileDataReader::readRow(std::string *key, float *vector)
{
  std::string line;
  if (std::getline(fileStream, line))
  {
    std::string vectorStr;
    splitKeyVector(line, *key, vectorStr, keyValueDelimiter);

    std::stringstream vectorStrStream(vectorStr);
    std::string elementStr;
    size_t idx;

    for (int i = 0; std::getline(vectorStrStream, elementStr, vectorDelimiter); ++i)
    {
      try
      {
        vector[i] = std::stof(elementStr, &idx);
        if(idx != elementStr.size()) {
            std::stringstream msg;
            msg << "Malformed vector element: " << elementStr;
            throw std::invalid_argument(msg.str());
        }
      } catch (const std::exception& e)
      {
        std::stringstream msg;
        msg << "ERROR: " << elementStr << " cannot be parsed as float. Column " << i << " of: " << line;
        fprintf(stderr, "ERROR: %s cannot be parsed as float. Column %d, line: %s\n", elementStr.c_str(), i,
            line.c_str());
        throw;
      }
    }

    return true;
  } else
  {
    return false;
  }
}

TextFileDataReader::~TextFileDataReader()
{
  fileStream.close();
}

