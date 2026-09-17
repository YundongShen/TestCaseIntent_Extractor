# How to Read the Onboarding Document

Read the document from the suite level to the case level:

> **Test Suite Overview → Test Groups → Test Cases → Test Activities**

## 1. Test Suite Overview

The overview identifies the Test Suite and summarizes its coverage.

- **Suite ID** identifies the Test Suite.
- **Test cases** shows the number of Test Cases in the suite.
- **Suite description** summarizes the behaviors covered by the suite.
- **Unique Dependencies** lists the distinct testing libraries used across the suite.
- **Case list** shows the Test Objects and dependencies associated with each Test Case.

## 2. Test Groups

Each group contains Test Cases that address one shared testing question. The question is a higher-level summary of their Test Goals.

Test Cases in one group may use different inputs, paths, states, or expected results. A shared Test Object or dependency alone is not enough to place Test Cases in the same group.

## 3. Test Cases

Each Test Case contains three types of information:

- **Test Objects — What is tested?**  
  The parts of the system examined by the Test Case. A Test Object may be a component, API endpoint, functional module, or unit of business logic. The Test Case checks its behavior, properties, or state.

- **Test Goals — What must be verified?**  
  What the Test Case is intended to verify about its Test Objects. A Test Goal states the expected behavior or condition used to judge the test result.

- **Test Activities — How is it verified?**  
  The ordered actions used to exercise the Test Objects and check the Test Goal. They may include setup, input, observation, and result validation.

Read the Test Goal before the Test Activities. The Goal explains the purpose of the Test Case. The Activities show how it is checked.

## 4. When More Detail Is Needed

Use this document to understand the Test Suite and locate relevant Test Cases. Check the test code for exact setup, test data, assertions, and execution details.
