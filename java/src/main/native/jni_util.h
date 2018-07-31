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

#ifndef JNI_UTIL_H_
#define JNI_UTIL_H_

#include <jni.h>
#include <map>
#include <string>
#include <tuple>

namespace dsstne {
namespace jni {

/* exceptions */
extern const std::string RuntimeException;
extern const std::string NullPointerException;
extern const std::string IllegalStateException;
extern const std::string IllegalArgumentException;
extern const std::string ClassNotFoundException;
extern const std::string NoSuchMethodException;
extern const std::string FileNotFoundException;
extern const std::string UnsupportedOperationException;

/* collections */
extern const std::string ArrayList;

/* java types */
extern const std::string String;

/* methods */
extern const std::string NO_ARGS_CONSTRUCTOR;

struct References;

void deleteReferences(JNIEnv *env, References &refs);

/**
 * Container for jclass and jmethodID references
 */
struct References {
 private:
  std::map<std::string, jclass> classGlobalRefs;
 public:
  friend void deleteReferences(JNIEnv *env, References &refs);

  jclass getClassGlobalRef(const std::string &className) const;

  bool containsClassGlobalRef(const std::string &className) const;

  void putClassGlobalRef(const std::string &className, jclass classRef);
};

void throwJavaException(JNIEnv* env, const std::string &exceptionType, const char *fmt, ...);

jclass findClassGlobalRef(JNIEnv *env, References &refs, const std::string &className);

jmethodID findMethodId(JNIEnv *env, References &refs, const std::string &className, const std::string &methodName,
                       const std::string &methodDescriptor);

/**
 * Finds constructors. methodDescriptor for constructors should return void (V) (e.g. ([Ljava/lang/String;[FI)V ).
 */
jmethodID findConstructorId(JNIEnv *env, References &refs, const std::string &className,
                            const std::string &methodDescriptor);

jobject newObject(JNIEnv *env, const References &refs, const std::string &className, jmethodID jConstructor, ...);

}  // namespace jni
}  // namespace dsstne
#endif /* JNI_UTIL_H_ */
