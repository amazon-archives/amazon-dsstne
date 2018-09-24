#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestListener.h>
#include <iostream>

class PrintProgressListener: public CppUnit::TestListener
{
public:
  void startTest(CppUnit::Test *test)
  {
    std::cout << "Running [" << test->getName() << "]" << std::endl;
  }

  void endTest(CppUnit::Test *test)
  {

    std::cout << "Finished [" << test->getName() << "]" << std::endl;
  }
};

int main(int argc, char **argv)
{
    PrintProgressListener progressListener;
    CppUnit::TextUi::TestRunner runner;
    CppUnit::TestFactoryRegistry &registry = CppUnit::TestFactoryRegistry::getRegistry();
    runner.addTest(registry.makeTest());
    runner.eventManager().addListener(&progressListener);

    bool wasSuccessful = runner.run("", false);
    return !wasSuccessful;
}
