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
#include <cstdio>
#include <map>
#include <iostream>
#include <string>
#include <sstream>

#include "jni_util.h"

namespace {
const std::string CONSTRUCTOR_METHOD_NAME = "<init>";
}  // namespace

namespace dsstne {
namespace jni {

/* exceptions */
const std::string RuntimeException = "java/lang/RuntimeException";
const std::string NullPointerException = "java/lang/NullPointerException";
const std::string IllegalStateException = "java/lang/IllegalStateException";
const std::string IllegalArgumentException = "java/lang/IllegalArgumentException";
const std::string ClassNotFoundException = "java/lang/ClassNotFoundException";
const std::string NoSuchMethodException = "java/lang/NoSuchMethodException";
const std::string FileNotFoundException = "java/io/FileNotFoundException";
const std::string UnsupportedOperationException = "java/lang/UnsupportedOperationException";

/* collections */
const std::string ArrayList = "java/util/ArrayList";

/* java types */
const std::string String = "java/lang/String";

/* methods */
const std::string NO_ARGS_CONSTRUCTOR = "()V";

void deleteReferences(JNIEnv *env, References &refs) {
  for (auto &entry : refs.classGlobalRefs) {
    env->DeleteGlobalRef(entry.second);
  }
  refs.classGlobalRefs.clear();
}

jclass References::getClassGlobalRef(const std::string &className) const {
  return classGlobalRefs.at(className);
}

bool References::containsClassGlobalRef(const std::string &className) const {
  return classGlobalRefs.find(className) != classGlobalRefs.end();
}

void References::putClassGlobalRef(const std::string &className, jclass classRef) {
  classGlobalRefs[className] = classRef;
}

void throwJavaException(JNIEnv* env, const std::string &exceptionType, const char *fmt, ...) {
  jclass exc = env->FindClass(exceptionType.c_str());

  va_list args;
  va_start(args, fmt);

  static const size_t MAX_MSG_LEN = 1024;
  char buffer[MAX_MSG_LEN];
  if (vsnprintf(buffer, MAX_MSG_LEN, fmt, args) >= 0) {
    env->ThrowNew(exc, buffer);
  } else {
    env->ThrowNew(exc, "");
  }

  va_end(args);
}

/**
 * Finds the provided class by name and adds a global reference to it to References. Subsequent findMethodId
 * calls on the same class do not have to be added as a global reference since the global reference
 * to the class keeps the class from being unloaded and hence also the method/field references.
 * Once done with the class reference, the global reference must be explicitly deleted to prevent
 * memory leaks.
 */
jclass findClassGlobalRef(JNIEnv *env, References &refs, const std::string &className) {
  if (refs.containsClassGlobalRef(className)) {
    return refs.getClassGlobalRef(className);
  }

  // this returns a local ref, need to create a global ref from this
  jclass classLocalRef = env->FindClass(className.c_str());

  jthrowable exc = env->ExceptionOccurred();
  if (exc) {
    env->ExceptionDescribe();
    exit(1);
  }

  if (classLocalRef == NULL) {
    throwJavaException(env, jni::ClassNotFoundException, "%s", className);
  }

  jclass classGlobalRef = (jclass) env->NewGlobalRef(classLocalRef);
  refs.putClassGlobalRef(className, classGlobalRef);
  env->DeleteLocalRef(classLocalRef);
  return classGlobalRef;
}

jmethodID findMethodId(JNIEnv *env, References &refs, const std::string &className, const std::string &methodName,
                       const std::string &methodDescriptor) {
  jclass clazz = findClassGlobalRef(env, refs, className);
  jmethodID methodId = env->GetMethodID(clazz, methodName.c_str(), methodDescriptor.c_str());

  jthrowable exc = env->ExceptionOccurred();
  if (exc) {
    std::cerr << "Error finding method " << className << "#" << methodName << methodDescriptor << std::endl;
    env->ExceptionDescribe();
    exit(1);
  }

  if (methodId == NULL) {
    throwJavaException(env, jni::NoSuchMethodException, "%s#%s%s", className, methodName, methodDescriptor);
  }

  return methodId;
}

jmethodID findConstructorId(JNIEnv *env, References &refs, const std::string &className,
                            const std::string &methodDescriptor) {
  return findMethodId(env, refs, className, CONSTRUCTOR_METHOD_NAME, methodDescriptor);
}

jobject newObject(JNIEnv *env, const References &refs, const std::string &className, jmethodID jConstructor, ...) {

  jclass clazz = refs.getClassGlobalRef(className);

  va_list args;
  va_start(args, jConstructor);
  jobject obj = env->NewObjectV(clazz, jConstructor, args);
  va_end(args);

  jthrowable exc = env->ExceptionOccurred();
  if (exc) {
    env->ExceptionDescribe();
    exit(1);
  }

  if (obj == NULL) {
    throwJavaException(env, jni::RuntimeException, "Unable to create new object: %s#%s", className,
                       CONSTRUCTOR_METHOD_NAME);
  }
  return obj;
}
}  // namespace jni
}  // namsspace dsstne
